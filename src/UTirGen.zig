// Generates UTir from a fully parsed Ast
const UTirGen = @This();

const std = @import("std");
const Ast = @import("Ast.zig");
const UTir = @import("Utir.zig");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const UTirGenError = error{} || Allocator.Error;

allocator: Allocator,
// The AST to build UTir from
ast: *const Ast,
// The list of in-progress UTir instructions being created
instructions: std.MultiArrayList(UTir.Inst) = .{},
// Extra data which instructions might need
extra_data: std.ArrayListUnmanaged(u32) = .{},
// Used to store strings which are used to refer to declarations
string_bytes: std.ArrayListUnmanaged(u8) = .{},
// Allocator used for short lived allocations
arena: std.heap.ArenaAllocator,

const Environment = struct {
    const Scope = enum {
        top,
        namespace,

        const Top = struct {
            base: Scope = .top,
        };

        const Namespace = struct {
            base: Scope = .namespace,
            decls: std.StringHashMapUnmanaged(UTir.Inst.Ref) = .{},
            parent: *const Scope,

            pub fn deinit(self: *Namespace, allocator: Allocator) void {
                self.decls.deinit(allocator);
            }

            pub fn addDecl(self: *Namespace, allocator: Allocator, decl: []const u8, ref: UTir.Inst.Ref) !void {
                var s = self.base;
                while (true) {
                    switch (s) {
                        .namespace => {
                            const namespace = s.toNamespace();
                            if (namespace.decls.get(decl)) |_| {
                                return error.DeclAlreadyExists;
                            }
                            s = namespace.parent;
                        },
                        .top => break,
                    }
                }

                try self.decls.putNoClobber(allocator, decl, ref);
            }
        };

        pub fn toNamespace(self: *Scope) *Namespace {
            assert(self.* == .namespace);
            return @fieldParentPtr(Namespace, "base", self);
        }
    };

    // Used to associate an environment with a scope. This can be useful for
    // things like name resolution.
    scope: *Scope,
    // All environments share the same list of instruction refs
    instructions: *std.ArrayListUnmanaged(UTir.Inst.Ref),
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

    pub fn addDecl(env: *Environment, decl: []const u8, ref: UTir.Inst.Ref) !void {
        switch (env.scope.*) {
            .namespace => {
                const namespace = env.scope.toNamespace();
                namespace.addDecl(decl, ref);
            },
            .top => unreachable,
        }
    }

    fn reserveInst(self: *Environment) UTirGenError!UTir.Inst.Ref {
        const result: UTir.Inst.Ref = @enumFromInt(self.instructions.items.len);
        try self.utir_gen.instructions.ensureUnusedCapacity(self.utir_gen.allocator, 1);
        self.utir_gen.instructions.len += 1;
        try self.instructions.append(self.utir_gen.allocator, result);
        return result;
    }

    fn addInst(self: *Environment, inst: UTir.Inst) UTirGenError!UTir.Inst.Ref {
        const result = self.instructions.len;
        try self.utir_gen.instructions.append(inst);
        try self.instructions.append(self.utir_gen.allocator, result);
        return @enumFromInt(result);
    }

    fn setInst(self: *Environment, idx: UTir.Inst.Ref, inst: UTir.Inst) void {
        self.utir_gen.instructions.set(@intFromEnum(idx), inst);
    }

    // Used to assert that no instruction refs have been leaked by environment
    pub fn finish(self: *const Environment) void {
        assert(self.instructions.items.len == self.bottom);
    }
};

pub fn genUTir(allocator: Allocator, ast: *const Ast) UTirGenError!UTir {
    var utir_gen = UTirGen{
        .allocator = allocator,
        .ast = ast,
        .arena = std.heap.ArenaAllocator.init(allocator),
    };
    defer utir_gen.deinit();

    var instructions: std.ArrayListUnmanaged(UTir.Inst.Ref) = .{};
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
    };
}

fn genStructInner(self: *UTirGen, env: *Environment, node_idx: Ast.Node.Idx) UTirGenError!UTir.Inst.Ref {
    const struct_decl = try env.reserveInst();
    const full_struct = self.ast.assembledStruct(node_idx).?;

    var namespace = Environment.Scope.Namespace{
        .parent = env.scope,
    };
    defer namespace.deinit(self.allocator);
    const struct_env = env.derive(&namespace.base);
    defer struct_env.finish();

    for (full_struct.fields) |field| {
        switch (self.ast.nodes.items(.tag)[@intFromEnum(field)]) {
            else => unreachable,
        }
    }

    // TODO: This is suspect
    const ed_idx = try self.addExtra(UTir.Inst.Struct{ .fields = @truncate(full_struct.fields.len) });
    _ = try self.addExtraSlice(UTir.Inst.Ref, struct_env.instructions.items[struct_env.bottom..]);

    env.setInst(struct_decl, .{ .struct_decl = .{ .ed_idx = ed_idx } });
    return struct_decl;
}

fn deinit(self: *UTirGen) void {
    self.arena.deinit();
}

fn addExtra(self: *UTirGen, val: anytype) UTirGenError!UTir.Inst.EdIdx {
    const fields = std.meta.fields(@TypeOf(val));
    try self.extra_data.ensureUnusedCapacity(self.allocator, fields.len);
    const result: UTir.Inst.EdIdx = @enumFromInt(self.extra_data.items.len);
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

fn addExtraSlice(self: *UTirGen, comptime T: type, slice: []const T) UTirGenError!UTir.Inst.EdIdx {
    var result: UTir.Inst.EdIdx = @enumFromInt(self.extra_data.items.len + slice.len);
    for (slice) |val| {
        _ = try self.addExtra(val);
    }
    return result;
}

test UTirGen {
    const allocator = std.testing.allocator;
    const tests = struct {
        fn doTheTest(src: [:0]const u8, expected_utir: []const UTir.Inst, expected_extra_data: []const u32) !void {
            var ast = try Ast.parse(allocator, src);
            defer ast.deinit(allocator);
            var utir = try genUTir(allocator, &ast);
            defer utir.deinit(allocator);

            for (0..utir.instructions.len) |i| {
                std.debug.print("{}\n", .{utir.instructions.get(i)});
            }
            for (utir.extra_data) |d| {
                std.debug.print("{}\n", .{d});
            }

            for (expected_utir, 0..) |e, i| {
                try std.testing.expectEqual(e, utir.instructions.get(i));
            }

            for (expected_extra_data, utir.extra_data) |e, a| {
                try std.testing.expectEqual(e, a);
            }
        }

        // %0 = {}
        pub fn test0() !void {
            const src =
                \\
            ;
            const expected_utir = [_]UTir.Inst{
                .{ .struct_decl = .{ .ed_idx = @enumFromInt(0) } }, // Root
            };
            const expected_extra_data = [_]u32{0};
            try doTheTest(src, &expected_utir, &expected_extra_data);
        }

        // %0 = {
        //     %1 = struct({});
        // }
        pub fn test1() !void {
            const src =
                \\const In = struct {};
            ;
            const expected_utir = [_]UTir.Inst{
                .{ .struct_decl = .{ .ed_idx = @enumFromInt(0) } }, // Root
                .{ .struct_decl = .{ .ed_idx = @enumFromInt(2) } }, // In
            };
            const expected_extra_data = [_]u32{
                1, // num of fields or decls
                1, // idx of In
                0,
            };
            try doTheTest(src, &expected_utir, &expected_extra_data);
        }
    };

    try tests.test0();
}
