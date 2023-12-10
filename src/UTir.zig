// Untyped Taaraa IR
// This is generated from the AST before any types has been assigned as
// a step towards full semantic analysis.
const UTir = @This();

const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

// A list of UTir instructions being built
instructions: std.MultiArrayList(Inst).Slice,
// Extra data which might be needed by a UTir instruction
extra_data: []const u32,
// Used to intern strings needed to refer to declarations
string_bytes: std.StringArrayHashMapUnmanaged(void),

pub const Inst = union(enum(u32)) {
    // Declares the type of a struct
    struct_decl: Payload(StructDecl),
    // Used to refer to the value of a declaration using a string
    decl_val: Str,
    // Declares a local block which contains an arbitrary number of instructions
    block: Payload(Block),
    // Used to perform type coercion
    as: Ref,

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

    // A `Str` is an index into `string_bytes`
    pub const Str = struct {
        string_bytes_idx: StrIdx,
    };

    // A `StructDecl` is followed by `Struct.fields` number of `Struct.Field`s
    // and a `Struct.decls` number of `Struct.Decl`s
    pub const StructDecl = struct {
        fields: u32,
        decls: u32,

        pub const Field = struct {
            name: StrIdx,
            type: Ref,
        };

        pub const Decl = struct {
            ref: Ref,
        };
    };

    // A `Block` is followed by `Block.instrs` number of `Block.Item`
    pub const Block = struct {
        instrs: u32,

        pub const Item = struct {
            ref: Ref,
        };
    };

    comptime {
        const InstInfo = @typeInfo(Inst).Union;
        if (!std.debug.runtime_safety) {
            for (InstInfo.fields) |field| {
                assert(@sizeOf(field.type) <= 8);
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
    const fields = std.meta.fields(T);
    inline for (fields, 0..) |field, i| {
        assert(@sizeOf(field.type) == @sizeOf(u32));
        switch (@typeInfo(field.type)) {
            .Enum => @field(result, field.name) = @enumFromInt(utir.extra_data[idx + i]),
            .Int => @field(result, field.name) = utir.extra_data[idx + i],
            else => @compileError("Unexpected type encountered in extra " ++ @typeName(field.type)),
        }
    }
    return result;
}

fn u32s(comptime T: type) comptime_int {
    return @sizeOf(T) / @sizeOf(u32);
}

inline fn tagFromRef(utir: *const UTir, ref: Inst.Ref) @typeInfo(Inst).Union.tag_type.? {
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
        try stream.writeAll("%0 = ");
        return self.writeStructDecl(stream, @enumFromInt(0));
    }

    fn writeExpr(self: *Writer, stream: anytype, inst_idx: Inst.Ref) !void {
        try switch (self.utir.tagFromRef(inst_idx)) {
            .struct_decl => self.writeStructDecl(stream, inst_idx),
            .decl_val => self.writeDeclVal(stream, inst_idx),
            else => unreachable,
        };
    }

    fn writeContainerMember(self: *Writer, stream: anytype, decl: Inst.StructDecl.Decl) !void {
        try stream.print("%{} = ", .{@intFromEnum(decl.ref)});
        try self.writeExpr(stream, decl.ref);
    }

    fn writeContainerField(self: *Writer, stream: anytype, field: Inst.StructDecl.Field) !void {
        const name = self.utir.string_bytes.keys()[@intFromEnum(field.name)];
        try stream.print("{s} (%{}): ", .{ name, @intFromEnum(field.type) });
        try self.writeExpr(stream, field.type);
    }

    fn writeStructDecl(self: *Writer, stream: anytype, inst_idx: Inst.Ref) @TypeOf(stream).Error!void {
        assert(self.utir.tagFromRef(inst_idx) == .struct_decl);
        try stream.writeAll("struct_decl({");
        const ed_idx = @intFromEnum(self.utir.instructions.items(.data)[@intFromEnum(inst_idx)].struct_decl.ed_idx);
        const fields_len = self.utir.extra_data[ed_idx];
        const decls_len = self.utir.extra_data[ed_idx + 1];
        self.incIndent(stream);
        if (fields_len + decls_len > 0) {
            try stream.print("\n", .{});

            const fields_base: u32 = ed_idx + 2;
            for (0..fields_len) |field_num| {
                const field_offset: u32 = @truncate(field_num);
                const field_idx: u32 = fields_base + field_offset * u32s(Inst.StructDecl.Field);
                const field = self.utir.extra(Inst.StructDecl.Field, field_idx);
                try self.writeContainerField(stream, field);
            }

            const decls_base: u32 = ed_idx + 2 + fields_len * u32s(Inst.StructDecl.Field);
            for (0..decls_len) |decl_num| {
                const decl_offset: u32 = @truncate(decl_num);
                const decl_idx: u32 = decls_base + decl_offset * u32s(Inst.StructDecl.Decl);
                const decl = self.utir.extra(Inst.StructDecl.Decl, decl_idx);
                try self.writeContainerMember(stream, decl);
            }
        }
        self.decIndent(stream);
        try stream.writeAll("})\n");
    }

    fn writeDeclVal(self: *Writer, stream: anytype, inst_idx: Inst.Ref) !void {
        assert(self.utir.tagFromRef(inst_idx) == .decl_val);
        const str_idx = @intFromEnum(self.utir.instructions.items(.data)[@intFromEnum(inst_idx)].decl_val.string_bytes_idx);
        try stream.print("decl_val(\"{s}\")\n", .{self.utir.string_bytes.keys()[str_idx]});
    }
};
