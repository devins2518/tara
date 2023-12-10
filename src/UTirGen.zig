// Generates UTir from a fully parsed Ast
const UTirGen = @This();

const std = @import("std");
const Ast = @import("Ast.zig");
const UTir = @import("Utir.zig");
const Inst = UTir.Inst;
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const UTirGenError = error{
    ScopeDeclAlreadyExists,
} || Allocator.Error;

allocator: Allocator,
// The AST to build UTir from
ast: *const Ast,
// The list of in-progress UTir instructions being created
instructions: std.MultiArrayList(Inst) = .{},
// Extra data which instructions might need
extra_data: std.ArrayListUnmanaged(u32) = .{},
// Used to store strings which are used to refer to declarations
string_bytes: std.StringArrayHashMapUnmanaged(void) = .{},
// Allocator used for short lived allocations
arena: std.heap.ArenaAllocator,

const Environment = struct {
    const Scope = enum {
        top,
        namespace,
        var_decl,

        const Top = struct {
            base: Scope = .top,
        };

        const Namespace = struct {
            base: Scope = .namespace,
            decls: std.StringHashMapUnmanaged(Inst.Ref) = .{},
            parent: *Scope,

            pub fn deinit(self: *Namespace, allocator: Allocator) void {
                self.decls.deinit(allocator);
            }
        };

        const VarDecl = struct {
            base: Scope = .var_decl,
            parent: *Scope,
        };

        pub fn addDecl(self: *Scope, allocator: Allocator, decl: []const u8, ref: Inst.Ref) !void {
            var s = self;
            while (true) {
                switch (s.*) {
                    .namespace => {
                        const namespace = s.toNamespace();
                        if (namespace.decls.get(decl)) |_| {
                            return UTirGenError.ScopeDeclAlreadyExists;
                        }
                        s = namespace.parent;
                    },
                    .var_decl => {
                        const var_decl = s.toVarDecl();
                        s = var_decl.parent;
                    },
                    .top => break,
                }
            }

            const namespace = self.toNamespace();

            try namespace.decls.putNoClobber(allocator, decl, ref);
        }

        pub fn toNamespace(self: *Scope) *Namespace {
            assert(self.* == .namespace);
            return @fieldParentPtr(Namespace, "base", self);
        }

        pub fn toVarDecl(self: *Scope) *VarDecl {
            assert(self.* == .var_decl);
            return @fieldParentPtr(VarDecl, "base", self);
        }
    };

    // Used to associate an environment with a scope. This can be useful for
    // things like name resolution.
    scope: *Scope,
    // All environments share the same list of instruction refs
    instructions: *std.ArrayListUnmanaged(Inst.Ref),
    // `bottom` refers to the top of `instructions` at the time the current environment was derived
    bottom: usize,
    // All environments share the same `utir_gen`
    utir_gen: *UTirGen,

    // Derives an environment from an already existing environment.
    pub fn derive(env: *Environment, scope: *Scope) Environment {
        return .{
            .scope = scope,
            .instructions = env.instructions,
            .bottom = env.instructions.items.len,
            .utir_gen = env.utir_gen,
        };
    }

    fn reserveInst(self: *Environment) UTirGenError!Inst.Ref {
        return try self.addInst(undefined);
    }

    fn addInst(self: *Environment, inst: Inst) UTirGenError!Inst.Ref {
        const result: Inst.Ref = @enumFromInt(self.utir_gen.instructions.len);
        try self.utir_gen.instructions.append(self.utir_gen.allocator, inst);
        try self.instructions.append(self.utir_gen.allocator, result);
        return result;
    }

    fn addRef(self: *Environment, ref: Inst.Ref) UTirGenError!void {
        try self.instructions.append(self.utir_gen.allocator, ref);
    }

    fn setInst(self: *Environment, idx: Inst.Ref, inst: Inst) void {
        self.utir_gen.instructions.set(@intFromEnum(idx), inst);
    }

    // Used to assert that no instruction refs have been leaked by environment
    pub fn finish(self: *const Environment) void {
        self.instructions.shrinkRetainingCapacity(self.bottom);
    }
};

pub fn genUTir(allocator: Allocator, ast: *const Ast) UTirGenError!UTir {
    var utir_gen = UTirGen{
        .allocator = allocator,
        .ast = ast,
        .arena = std.heap.ArenaAllocator.init(allocator),
    };
    defer utir_gen.deinit();

    var instructions: std.ArrayListUnmanaged(Inst.Ref) = .{};
    defer instructions.deinit(allocator);
    var top_scope = Environment.Scope.Top{};
    var env = Environment{
        .scope = &top_scope.base,
        .instructions = &instructions,
        .bottom = 0,
        .utir_gen = &utir_gen,
    };
    _ = try utir_gen.genStructInner(&env, @enumFromInt(0));

    return UTir{
        .instructions = utir_gen.instructions.toOwnedSlice(),
        .extra_data = try utir_gen.extra_data.toOwnedSlice(allocator),
        .string_bytes = utir_gen.string_bytes.move(),
    };
}

fn genStructInner(self: *UTirGen, env: *Environment, node_idx: Ast.Node.Idx) UTirGenError!Inst.Ref {
    const struct_decl = try env.reserveInst();
    const full_struct = self.ast.assembledStruct(node_idx).?;

    var namespace = Environment.Scope.Namespace{
        .parent = env.scope,
    };
    defer namespace.deinit(self.allocator);
    var struct_env = env.derive(&namespace.base);
    defer struct_env.finish();

    var fields: u32 = 0;
    var decls: u32 = 0;
    for (full_struct.members) |member| {
        // TODO: have this be all fields followed by all decls to have some sense of order
        switch (self.ast.nodes.items(.tag)[@intFromEnum(member)]) {
            .var_decl => {
                try env.addRef(try self.genVarDecl(&struct_env, member));
                decls += 1;
            },
            .container_field => {
                try env.addRef(try self.genVarDecl(&struct_env, member));
                fields += 1;
            },
            else => unreachable,
        }
    }

    const ed_idx = try self.addExtra(Inst.Struct{ .fields = fields, .decls = decls });
    const inst_refs = struct_env.instructions.items[struct_env.bottom..];
    const casted_refs: []Inst.Struct.Item = @ptrCast(inst_refs);
    _ = try self.addExtraSlice(Inst.Struct.Item, casted_refs);

    env.setInst(struct_decl, .{ .struct_decl = .{ .ed_idx = ed_idx } });
    return struct_decl;
}

// Generates an instruction from a variable declation. Returns reference to instruction
fn genVarDecl(self: *UTirGen, env: *Environment, node_idx: Ast.Node.Idx) UTirGenError!Inst.Ref {
    const full_var_decl = self.ast.assembledVarDecl(node_idx).?;

    var scope = Environment.Scope.VarDecl{
        .parent = env.scope,
    };
    var var_env = env.derive(&scope.base);
    defer var_env.finish();

    const ident = self.tokToString(full_var_decl.token + 1);

    const init_expr = try self.genExpr(&var_env, full_var_decl.expr);

    try env.scope.addDecl(self.allocator, ident, init_expr);
    return init_expr;
}

fn genIdentifier(self: *UTirGen, env: *Environment, node_idx: Ast.Node.Idx) UTirGenError!Inst.Ref {
    const token_idx = self.ast.nodes.items(.main_idx)[@intFromEnum(node_idx)];
    const string = self.tokToString(token_idx);
    const str_idx = try self.addStringBytes(string);
    return try env.addInst(.{ .decl_val = .{ .string_bytes_idx = str_idx } });
}

fn genExpr(self: *UTirGen, env: *Environment, node_idx: Ast.Node.Idx) UTirGenError!Inst.Ref {
    const tags = self.ast.nodes.items(.tag);
    switch (tags[@intFromEnum(node_idx)]) {
        .struct_decl => return self.genStructInner(env, node_idx),
        .identifier => return self.genIdentifier(env, node_idx),
        .root,
        .var_decl,
        .module_decl,
        .module_arg,
        .container_field,
        .@"or",
        .@"and",
        .lt,
        .gt,
        .lte,
        .gte,
        .eq,
        .neq,
        .bit_and,
        .bit_or,
        .bit_xor,
        .add,
        .sub,
        .mul,
        .div,
        .reference,
        .assignment,
        .member,
        => unreachable,
    }
}

fn tokToString(self: *UTirGen, token_idx: Ast.TokenIdx) []const u8 {
    const token = self.ast.tokens[token_idx];
    return self.ast.source[token.loc.start..token.loc.end];
}

fn deinit(self: *UTirGen) void {
    self.arena.deinit();
}

fn addExtra(self: *UTirGen, val: anytype) UTirGenError!Inst.EdIdx {
    const fields = std.meta.fields(@TypeOf(val));
    try self.extra_data.ensureUnusedCapacity(self.allocator, fields.len);
    const result: Inst.EdIdx = @enumFromInt(self.extra_data.items.len);
    inline for (fields) |field| {
        comptime assert(@sizeOf(field.type) == @sizeOf(u32));
        switch (@typeInfo(field.type)) {
            .Enum => self.extra_data.appendAssumeCapacity(@intFromEnum(@field(val, field.name))),
            .Int => self.extra_data.appendAssumeCapacity(@field(val, field.name)),
            else => @compileError("Unexpected type found in addExtra: " ++ @typeName(field.type)),
        }
    }
    return result;
}

fn addExtraSlice(self: *UTirGen, comptime T: type, slice: []const T) UTirGenError!Inst.EdIdx {
    const result: Inst.EdIdx = @enumFromInt(self.extra_data.items.len + slice.len);
    for (slice) |val| {
        _ = try self.addExtra(val);
    }
    return result;
}

fn addStringBytes(self: *UTirGen, string: []const u8) UTirGenError!Inst.StrIdx {
    const result = try self.string_bytes.getOrPut(self.allocator, string);
    return @enumFromInt(result.index);
}

test UTirGen {
    const allocator = std.testing.allocator;
    const tests = struct {
        fn doTheTest(src: [:0]const u8, expected_utir: []const Inst, expected_extra_data: []const u32, expected_utir_str: []const u8) !void {
            var ast = try Ast.parse(allocator, src);
            defer ast.deinit(allocator);
            var utir = try genUTir(allocator, &ast);
            defer utir.deinit(allocator);

            for (expected_utir, 0..) |e, i| {
                try std.testing.expectEqual(e, utir.instructions.get(i));
            }

            for (expected_extra_data, utir.extra_data) |e, a| {
                try std.testing.expectEqual(e, a);
            }

            if (expected_utir_str.len > 0) {
                var actual_array_list = try std.ArrayList(u8).initCapacity(allocator, expected_utir_str.len);
                defer actual_array_list.deinit();
                const actual_writer = actual_array_list.writer();
                try std.fmt.format(actual_writer, "{}", .{utir});
                try std.testing.expectEqualStrings(expected_utir_str, actual_array_list.items);
            } else {
                return error.ZigSkipTest;
            }
        }

        pub fn test0() !void {
            const src =
                \\
            ;
            const expected_utir = [_]Inst{
                .{ .struct_decl = .{ .ed_idx = @enumFromInt(0) } }, // Root
            };
            const expected_extra_data = [_]u32{ 0, 0 };
            const expected_utir_str =
                \\%0 = struct_decl({})
                \\
            ;
            try doTheTest(src, &expected_utir, &expected_extra_data, expected_utir_str);
        }

        pub fn test1() !void {
            const src =
                \\const In = struct {};
                \\const Out = struct {};
            ;
            const expected_utir = [_]Inst{
                .{ .struct_decl = .{ .ed_idx = @enumFromInt(4) } }, // Root
                .{ .struct_decl = .{ .ed_idx = @enumFromInt(0) } }, // In
                .{ .struct_decl = .{ .ed_idx = @enumFromInt(2) } }, // Out
            };
            const expected_extra_data = [_]u32{
                0, // In.fields
                0, // In.decls
                0, // Out.fields
                0, // Out.decls
                0, // Root.fields
                2, // Root.decls
                1, // Root.In
                2, // Root.Out
            };
            const expected_utir_str =
                \\%0 = struct_decl({
                \\    %1 = struct_decl({})
                \\    %2 = struct_decl({})
                \\})
                \\
            ;
            try doTheTest(src, &expected_utir, &expected_extra_data, expected_utir_str);
        }

        pub fn test2() !void {
            const src =
                \\const In = struct {
                \\    const A = struct {};
                \\    const B = struct {
                \\        const C = struct {};
                \\    };
                \\};
                \\const Out = struct {};
            ;
            const expected_utir = [_]Inst{
                .{ .struct_decl = .{ .ed_idx = @enumFromInt(13) } }, // Root
                .{ .struct_decl = .{ .ed_idx = @enumFromInt(7) } }, // In
                .{ .struct_decl = .{ .ed_idx = @enumFromInt(0) } }, // In.A
                .{ .struct_decl = .{ .ed_idx = @enumFromInt(4) } }, // In.B
                .{ .struct_decl = .{ .ed_idx = @enumFromInt(2) } }, // In.B.C
                .{ .struct_decl = .{ .ed_idx = @enumFromInt(11) } }, // Out
            };
            const expected_extra_data = [_]u32{
                0, // In.A.fields
                0, // In.A.decls
                0, // In.B.C.fields
                0, // In.B.C.decls
                0, // In.B.fields
                1, // In.B.decls
                4, // In.B.C
                0, // In.fields
                2, // In.decls
                2, // In.A
                3, // In.B
                0, // Out.fields
                0, // Out.decls
                0, // Root.fields
                2, // Root.decls
                1, // In
                5, // Out
            };

            const expected_utir_str =
                \\%0 = struct_decl({
                \\    %1 = struct_decl({
                \\        %2 = struct_decl({})
                \\        %3 = struct_decl({
                \\            %4 = struct_decl({})
                \\        })
                \\    })
                \\    %5 = struct_decl({})
                \\})
                \\
            ;
            try doTheTest(src, &expected_utir, &expected_extra_data, expected_utir_str);
        }

        pub fn test3() !void {
            const src =
                \\const In = bool;
            ;
            const expected_utir = [_]Inst{
                .{ .struct_decl = .{ .ed_idx = @enumFromInt(0) } }, // Root
                .{ .decl_val = .{ .string_bytes_idx = @enumFromInt(0) } }, // In
            };
            const expected_extra_data = [_]u32{
                0, // Root.fields
                1, // Root.decls
                1, // In
            };
            const expected_utir_str =
                \\%0 = struct_decl({
                \\    %1 = decl_val("bool")
                \\})
                \\
            ;
            try doTheTest(src, &expected_utir, &expected_extra_data, expected_utir_str);
        }
    };

    try tests.test0();
    try tests.test1();
    try tests.test2();
    try tests.test3();
}
