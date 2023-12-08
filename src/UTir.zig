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
    struct_decl: Payload(Struct),
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

    // A `Struct` is followed by `Struct.fields` number of `Struct.Item`
    pub const Struct = struct {
        fields: u32,

        pub const Item = struct {
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

    fn writeContainerMembers(self: *Writer, stream: anytype, inst_idx: Inst.Ref) !void {
        try stream.print("%{} = ", .{@intFromEnum(inst_idx)});
        try switch (self.utir.tagFromRef(inst_idx)) {
            .struct_decl => self.writeStructDecl(stream, inst_idx),
            .decl_val => self.writeDeclVal(stream, inst_idx),
            else => unreachable,
        };
    }

    fn writeStructDecl(self: *Writer, stream: anytype, inst_idx: Inst.Ref) @TypeOf(stream).Error!void {
        assert(self.utir.tagFromRef(inst_idx) == .struct_decl);
        try stream.writeAll("struct_decl({");
        const ed_idx = @intFromEnum(self.utir.instructions.items(.data)[@intFromEnum(inst_idx)].struct_decl.ed_idx);
        const root_len = self.utir.extra_data[ed_idx];
        self.incIndent(stream);
        if (root_len > 0) {
            try stream.print("\n", .{});
            const root_decls = self.utir.extra_data[ed_idx + 1 .. ed_idx + root_len + 1];
            for (root_decls) |root_decl| {
                try self.writeContainerMembers(stream, @enumFromInt(root_decl));
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
