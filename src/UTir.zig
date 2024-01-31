// Untyped Tara IR
// This is generated from the AST before any types has been assigned as
// a step towards full semantic analysis.
const UTir = @This();

const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const utils = @import("utils.zig");
const unionPayloadPtr = utils.unionPayloadPtr;
const u32s = utils.u32s;

// A list of UTir instructions being built
instructions: std.MultiArrayList(Inst).Slice,
// Extra data which might be needed by a UTir instruction
extra_data: []const u32,
// Used to intern strings needed to refer to declarations
string_bytes: std.StringArrayHashMapUnmanaged(void),

pub const Inst = union(enum(u32)) {
    // Declares the type of a struct
    struct_decl: Payload(StructDecl),
    // Declares the type of a module
    module_decl: Payload(ModuleDecl),
    // Declarares the existance of a subroutine (fn/comb)
    subroutine_decl: Payload(SubroutineDecl),
    // Used to refer to the value of a declaration using a string
    decl_val: Str,
    // Declares a local block which contains an arbitrary number of instructions
    inline_block: Payload(Block),
    // Breaks from `BinOp.lhs` inline block with `BinOp.rhs` instruction
    inline_block_break: BinOp,
    // Used to perform type coercion
    as: Ref,
    // TODO: use payload to support integers > max(u32)
    // Used to refer to small integers
    int_small: Int.Small,
    // Performs boolean `or` operation on the `BinOp.lhs` and `BinOp.rhs`
    @"or": BinOp,
    // Performs boolean `and` operation on the `BinOp.lhs` and `BinOp.rhs`
    @"and": BinOp,
    // Performs `lt` operation on the `BinOp.lhs` and `BinOp.rhs`
    lt: BinOp,
    // Performs `gt` operation on the `BinOp.lhs` and `BinOp.rhs`
    gt: BinOp,
    // Performs `lte` operation on the `BinOp.lhs` and `BinOp.rhs`
    lte: BinOp,
    // Performs `gte` operation on the `BinOp.lhs` and `BinOp.rhs`
    gte: BinOp,
    // Performs `eq` operation on the `BinOp.lhs` and `BinOp.rhs`
    eq: BinOp,
    // Performs `neq` operation on the `BinOp.lhs` and `BinOp.rhs`
    neq: BinOp,
    // Performs bitwise `and` operation on the `BinOp.lhs` and `BinOp.rhs`
    bit_and: BinOp,
    // Performs bitwise `or` operation on the `BinOp.lhs` and `BinOp.rhs`
    bit_or: BinOp,
    // Performs bitwise `xor` operation on the `BinOp.lhs` and `BinOp.rhs`
    bit_xor: BinOp,
    // Performs `add` operation on the `BinOp.lhs` and `BinOp.rhs`
    add: BinOp,
    // Performs `sub` operation on the `BinOp.lhs` and `BinOp.rhs`
    sub: BinOp,
    // Performs `mul` operation on the `BinOp.lhs` and `BinOp.rhs`
    mul: BinOp,
    // Performs `div` operation on the `BinOp.lhs` and `BinOp.rhs`
    div: BinOp,
    // Returns `UnOp.lhs` from the innermost function
    @"return": UnOp,
    // Creates a reference type of `PtrTy.child` type
    ref_ty: PtrTy,
    // Creates a pointer type of `PtrTy.child` type
    ptr_ty: PtrTy,

    pub const Tag = std.meta.Tag(Inst);

    // Index into instructions list
    pub const Ref = enum(u32) { _ };

    // An index into `extra_data`
    pub const EdIdx = enum(u32) { _ };

    // An index into `string_bytes`
    pub const StrIdx = enum(u32) { _ };

    // A payload will contain an index into `extra_data` to a certain `T`. The
    // specific `T` is determined by the tag of the instruction.
    pub fn Payload(comptime T: type) type {
        return struct {
            ed_idx: EdIdx,

            pub fn getFromExtra(self: @This(), extra_data: []const u32) T {
                var t: T = undefined;
                inline for (@typeInfo(T).Struct.fields, 0..) |field, i| {
                    @field(t, field.name) = extra_data[self.ed_idx + i];
                }
                return t;
            }
        };
    }

    pub const PtrTy = struct {
        child: Ref,
    };

    // A `Str` is an index into `string_bytes`
    pub const Str = struct {
        string_bytes_idx: StrIdx,
    };

    // A `StructDecl` is followed by `Struct.fields` number of `ContainerField`s
    // and a `Struct.decls` number of `ContainerDecl`s
    pub const StructDecl = struct {
        fields: u32,
        decls: u32,
    };

    // A `ContainerField` is a binding between a `name` and a `type`
    pub const ContainerField = struct {
        name: StrIdx,
        type: Ref,
    };

    // A `ContainerField` is a reference to an arbitrary expression.
    pub const ContainerDecl = struct {
        ref: Ref,
    };

    // A `ModuleDecl` is followed by `Module.fields` number of `ContainerField`s
    // and a `Module.decls` number of `ContainerDecl`s
    pub const ModuleDecl = struct {
        fields: u32,
        decls: u32,
    };

    // A `Block` is followed by `Block.instrs` number of `Block.Item`
    pub const Block = struct {
        instrs: u32,

        pub const Item = struct {
            ref: Ref,
        };
    };

    // A `SubroutineDecl` is followed by `SubroutineDecl.param_len` number of
    // `Param`s and then `SubroutineDecl.body_len` number of `Ref`s.
    pub const SubroutineDecl = struct {
        name_idx: StrIdx,
        param_len: u32,
        ret_ty: Ref,
        body_len: u32,

        pub const Param = struct {
            name: StrIdx,
            type: Ref,
        };
    };

    pub const Int = struct {
        pub const Small = struct {
            int: u32,
        };
    };

    pub const UnOp = struct {
        lhs: Inst.Ref,

        pub fn tagIsUnOp(tag: Tag) bool {
            return switch (tag) {
                inline else => |t| std.meta.TagPayload(Inst, t) == UnOp,
            };
        }

        pub fn fromUtir(utir: *const UTir, inst_idx: Inst.Ref) UnOp {
            assert(tagIsUnOp(utir.tagFromRef(inst_idx)));
            return unionPayloadPtr(UnOp, utir.instructions.get(@intFromEnum(inst_idx))).?;
        }
    };

    pub const BinOp = struct {
        lhs: Inst.Ref,
        rhs: Inst.Ref,

        pub fn tagIsBinOp(tag: Tag) bool {
            return switch (tag) {
                inline else => |t| std.meta.TagPayload(Inst, t) == BinOp,
            };
        }

        pub fn fromUtir(utir: *const UTir, inst_idx: Inst.Ref) BinOp {
            assert(tagIsBinOp(utir.tagFromRef(inst_idx)));
            return unionPayloadPtr(BinOp, utir.instructions.get(@intFromEnum(inst_idx))).?;
        }
    };

    comptime {
        const InstInfo = @typeInfo(Inst).Union;
        if (!std.debug.runtime_safety) {
            for (InstInfo.fields) |field| {
                // TODO: change to 8 when source tracking is added
                assert(@sizeOf(field.type) <= 4);
            }
        }
    }
};

pub fn deinit(self: *UTir, allocator: Allocator) void {
    self.instructions.deinit(allocator);
    allocator.free(self.extra_data);
    self.string_bytes.deinit(allocator);
}

pub fn format(utir: *const UTir, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
    var stream = Stream(@TypeOf(writer)).init(writer);
    var write = Writer{ .utir = utir };
    try write.writeRoot(stream.writer());
}

fn extra(utir: *const UTir, comptime T: type, idx: u32) T {
    var result: T = undefined;
    switch (@typeInfo(T)) {
        .Struct => {
            const fields = std.meta.fields(T);
            inline for (fields, 0..) |field, i| {
                assert(@sizeOf(field.type) == @sizeOf(u32));
                @field(result, field.name) = utir.extra(field.type, @intCast(idx + i));
            }
        },
        .Enum => {
            assert(@sizeOf(T) == @sizeOf(u32));
            result = @enumFromInt(utir.extra_data[idx]);
        },
        .Int => {
            assert(@sizeOf(T) == @sizeOf(u32));
            result = utir.extra_data[idx];
        },
        else => @compileError("Unexpected type encountered in extra " ++ @typeName(T)),
    }
    return result;
}

inline fn tagFromRef(utir: *const UTir, ref: Inst.Ref) Inst.Tag {
    return utir.instructions.items(.tags)[@intFromEnum(ref)];
}

pub fn Stream(comptime UnderlyingStreamType: type) type {
    return struct {
        const AutoIndentingStream = @This();
        const AutoIndentingWriter = std.io.Writer(*AutoIndentingStream, WriteError, writeFn);
        const WriteError = UnderlyingStreamType.Error;

        stream: UnderlyingStreamType,
        indent: usize = 0,
        pending_indent: bool = false,

        pub fn init(stream: UnderlyingStreamType) AutoIndentingStream {
            return .{ .stream = stream };
        }

        pub fn writer(self: *AutoIndentingStream) AutoIndentingWriter {
            return AutoIndentingWriter{ .context = self };
        }

        pub fn writeFn(self: *AutoIndentingStream, bytes: []const u8) WriteError!usize {
            var count: usize = 0;
            if (std.mem.eql(u8, bytes, "{indent+}")) {
                self.indent += 4;
                count = 9;
            } else if (std.mem.eql(u8, bytes, "{indent-}")) {
                self.indent -= 4;
                count = 9;
            } else {
                if (self.pending_indent) {
                    try self.stream.writeByteNTimes(' ', self.indent);
                }
                for (bytes, 0..) |b, i| {
                    try self.stream.writeByte(b);
                    if (b == '\n' and i != bytes.len - 1) {
                        try self.stream.writeByteNTimes(' ', self.indent);
                    }
                    count += 1;
                }
                self.pending_indent = bytes[bytes.len - 1] == '\n';
            }
            return count;
        }
    };
}

const Writer = struct {
    utir: *const UTir,

    fn incIndent(_: *Writer, stream: anytype) void {
        stream.writeAll("{indent+}") catch unreachable;
    }

    fn decIndent(_: *Writer, stream: anytype) void {
        stream.writeAll("{indent-}") catch unreachable;
    }

    pub fn writeRoot(self: *Writer, stream: anytype) !void {
        return self.writeStructDecl(stream, @enumFromInt(0));
    }

    fn writeExpr(self: *Writer, stream: anytype, inst_idx: Inst.Ref) @TypeOf(stream).Error!void {
        try switch (self.utir.tagFromRef(inst_idx)) {
            .module_decl => self.writeModuleDecl(stream, inst_idx),
            .struct_decl => self.writeStructDecl(stream, inst_idx),
            .decl_val => self.writeDeclVal(stream, inst_idx),
            .int_small => self.writeIntSmall(stream, inst_idx),
            .subroutine_decl => self.writeSubroutineDecl(stream, inst_idx),
            inline .@"or",
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
            .inline_block_break,
            => |op| self.writeBinOp(stream, inst_idx, op),
            .@"return" => |op| self.writeUnOp(stream, inst_idx, op),
            .inline_block => self.writeInlineBlock(stream, inst_idx),
            .ref_ty => self.writeRefTy(stream, inst_idx),
            .as,
            .ptr_ty,
            => unreachable,
        };
    }

    fn writeContainerMember(self: *Writer, stream: anytype, decl: Inst.ContainerDecl) !void {
        try self.writeExpr(stream, decl.ref);
    }

    fn writeContainerField(self: *Writer, stream: anytype, field: Inst.ContainerField) !void {
        try self.writeExpr(stream, field.type);
        try stream.writeAll("\n");
        const name = self.utir.string_bytes.keys()[@intFromEnum(field.name)];
        try stream.print("{s} : %{}", .{ name, @intFromEnum(field.type) });
    }

    fn writeStructDecl(self: *Writer, stream: anytype, inst_idx: Inst.Ref) @TypeOf(stream).Error!void {
        assert(self.utir.tagFromRef(inst_idx) == .struct_decl);
        try stream.print("%{} = struct_decl({{", .{@intFromEnum(inst_idx)});
        const ed_idx = @intFromEnum(self.utir.instructions.items(.data)[@intFromEnum(inst_idx)].struct_decl.ed_idx);
        const struct_decl = self.utir.extra(Inst.StructDecl, ed_idx);
        self.incIndent(stream);
        if (struct_decl.fields + struct_decl.decls > 0) {
            try stream.print("\n", .{});

            const fields_base: u32 = ed_idx + 2;
            for (0..struct_decl.fields) |field_num| {
                const field_offset: u32 = @truncate(field_num);
                const field_idx: u32 = fields_base + field_offset * u32s(Inst.ContainerField);
                const field = self.utir.extra(Inst.ContainerField, field_idx);
                try self.writeContainerField(stream, field);
                try stream.writeAll("\n");
            }

            const decls_base: u32 = ed_idx + 2 + struct_decl.fields * u32s(Inst.ContainerField);
            for (0..struct_decl.decls) |decl_num| {
                const decl_offset: u32 = @truncate(decl_num);
                const decl_idx: u32 = decls_base + decl_offset * u32s(Inst.ContainerDecl);
                const decl = self.utir.extra(Inst.ContainerDecl, decl_idx);
                try self.writeContainerMember(stream, decl);
                try stream.writeAll("\n");
            }
        }
        self.decIndent(stream);
        try stream.writeAll("})");
    }

    fn writeModuleDecl(self: *Writer, stream: anytype, inst_idx: Inst.Ref) @TypeOf(stream).Error!void {
        assert(self.utir.tagFromRef(inst_idx) == .module_decl);
        try stream.print("%{} = module_decl({{", .{@intFromEnum(inst_idx)});
        const ed_idx = @intFromEnum(self.utir.instructions.items(.data)[@intFromEnum(inst_idx)].module_decl.ed_idx);
        const fields_len = self.utir.extra_data[ed_idx];
        const decls_len = self.utir.extra_data[ed_idx + 1];
        self.incIndent(stream);
        if (fields_len + decls_len > 0) {
            try stream.print("\n", .{});

            const fields_base: u32 = ed_idx + 2;
            for (0..fields_len) |field_num| {
                const field_offset: u32 = @truncate(field_num);
                const field_idx: u32 = fields_base + field_offset * u32s(Inst.ContainerField);
                const field = self.utir.extra(Inst.ContainerField, field_idx);
                try self.writeContainerField(stream, field);
                try stream.writeAll("\n");
            }

            const decls_base: u32 = ed_idx + 2 + fields_len * u32s(Inst.ContainerField);
            for (0..decls_len) |decl_num| {
                const decl_offset: u32 = @truncate(decl_num);
                const decl_idx: u32 = decls_base + decl_offset * u32s(Inst.ContainerDecl);
                const decl = self.utir.extra(Inst.ContainerDecl, decl_idx);
                try self.writeContainerMember(stream, decl);
                try stream.writeAll("\n");
            }
        }
        self.decIndent(stream);
        try stream.writeAll("})");
    }

    fn writeDeclVal(self: *Writer, stream: anytype, inst_idx: Inst.Ref) !void {
        assert(self.utir.tagFromRef(inst_idx) == .decl_val);
        const str_idx = @intFromEnum(self.utir.instructions.items(.data)[@intFromEnum(inst_idx)].decl_val.string_bytes_idx);
        try stream.print("%{} = decl_val(\"{s}\")", .{
            @intFromEnum(inst_idx),
            self.utir.string_bytes.keys()[str_idx],
        });
    }

    fn writeIntSmall(self: *Writer, stream: anytype, inst_idx: Inst.Ref) !void {
        assert(self.utir.tagFromRef(inst_idx) == .int_small);
        const int = self.utir.instructions.items(.data)[@intFromEnum(inst_idx)].int_small.int;
        try stream.print("%{} = int({})", .{ @intFromEnum(inst_idx), int });
    }

    fn writeRefTy(self: *Writer, stream: anytype, inst_idx: Inst.Ref) !void {
        assert(self.utir.tagFromRef(inst_idx) == .ref_ty);
        const child = self.utir.instructions.items(.data)[@intFromEnum(inst_idx)].ref_ty.child;
        try stream.print("%{} = ref_ty(%{})", .{ @intFromEnum(inst_idx), @intFromEnum(child) });
    }

    fn writeBinOp(self: *Writer, stream: anytype, inst_idx: Inst.Ref, op: Inst.Tag) !void {
        const bin_op = Inst.BinOp.fromUtir(self.utir, inst_idx);
        try stream.print("%{} = {s}(%{}, %{})", .{
            @intFromEnum(inst_idx),
            @tagName(op),
            @intFromEnum(bin_op.lhs),
            @intFromEnum(bin_op.rhs),
        });
    }

    fn writeUnOp(self: *Writer, stream: anytype, inst_idx: Inst.Ref, op: Inst.Tag) !void {
        const un_op = Inst.UnOp.fromUtir(self.utir, inst_idx);
        try stream.print("%{} = {s}(%{})", .{
            @intFromEnum(inst_idx),
            @tagName(op),
            @intFromEnum(un_op.lhs),
        });
    }

    fn writeInlineBlock(self: *Writer, stream: anytype, inst_idx: Inst.Ref) !void {
        assert(self.utir.tagFromRef(inst_idx) == .inline_block);
        try stream.print("%{} = inline_block({{", .{@intFromEnum(inst_idx)});
        const ed_idx = @intFromEnum(self.utir.instructions.items(.data)[@intFromEnum(inst_idx)].inline_block.ed_idx);
        const instr_len = self.utir.extra_data[ed_idx];
        self.incIndent(stream);
        if (instr_len > 0) {
            try stream.print("\n", .{});
        }
        for (0..instr_len) |i| {
            const instr = self.utir.extra(Inst.Ref, @intCast(ed_idx + 1 + i));
            try self.writeExpr(stream, instr);
            try stream.writeAll("\n");
        }
        self.decIndent(stream);
        try stream.writeAll("})");
    }

    fn writeSubroutineDecl(self: *Writer, stream: anytype, inst_idx: Inst.Ref) !void {
        assert(self.utir.tagFromRef(inst_idx) == .subroutine_decl);

        var ed_idx = @intFromEnum(self.utir.instructions.items(.data)[@intFromEnum(inst_idx)].subroutine_decl.ed_idx);
        const subroutine_decl = self.utir.extra(Inst.SubroutineDecl, ed_idx);
        ed_idx += u32s(Inst.SubroutineDecl);

        const subroutine_name = self.utir.string_bytes.keys()[@intFromEnum(subroutine_decl.name_idx)];

        try stream.print("%{} = subroutine_decl(\n", .{@intFromEnum(inst_idx)});
        self.incIndent(stream);

        try stream.print("\"{s}\",\n{{", .{subroutine_name});

        self.incIndent(stream);
        if (subroutine_decl.param_len > 0) {
            try stream.writeAll("\n");

            for (0..subroutine_decl.param_len) |_| {
                const param = self.utir.extra(Inst.SubroutineDecl.Param, ed_idx);

                try self.writeExpr(stream, param.type);
                try stream.writeAll("\n");

                const name = self.utir.string_bytes.keys()[@intFromEnum(param.name)];
                try stream.print("{s} : %{}\n", .{ name, @intFromEnum(param.type) });

                ed_idx += u32s(Inst.SubroutineDecl.Param);
            }
        }
        self.decIndent(stream);
        try stream.writeAll("},\n");

        try self.writeExpr(stream, subroutine_decl.ret_ty);
        try stream.writeAll(",\n{");

        self.incIndent(stream);
        if (subroutine_decl.body_len > 0) {
            try stream.writeAll("\n");

            for (0..subroutine_decl.body_len) |_| {
                const body = self.utir.extra(Inst.Ref, @intCast(ed_idx));
                try self.writeExpr(stream, body);
                try stream.writeAll("\n");

                ed_idx += u32s(Inst.Ref);
            }
        }
        self.decIndent(stream);
        try stream.writeAll("}\n");

        self.decIndent(stream);
        try stream.writeAll(")");
    }
};
