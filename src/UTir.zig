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

    pub const Str = struct {
        string_bytes_idx: u32,
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
}

pub fn format(utir: UTir, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
    var write = Writer{ .utir = utir };
    try write.writeRoot(writer);
}

const Writer = struct {
    utir: UTir,
    indent: usize = 0,
    pending_indent: bool = false,

    fn incIndent(self: *Writer) void {
        self.indent += 4;
    }

    fn decIndent(self: *Writer) void {
        self.indent -= 4;
    }

    fn writeAll(self: *Writer, stream: anytype, comptime fmt: []const u8, args: anytype) !void {
        if (self.pending_indent) try stream.writeByteNTimes(' ', self.indent);
        try std.fmt.format(stream, fmt, args);
        self.pending_indent = fmt[fmt.len - 1] == '\n';
    }

    pub fn writeRoot(self: *Writer, stream: anytype) !void {
        try self.writeAll(stream, "%0 = ", .{});
        const ed_idx = @intFromEnum(self.utir.instructions.get(0).struct_decl.ed_idx);
        const root_len = self.utir.extra_data[ed_idx];
        self.incIndent();
        defer self.decIndent();
        if (root_len > 0) {
            const root_decls = self.utir.extra_data[ed_idx + 1 .. ed_idx + root_len + 1];
            for (root_decls) |root_decl| {
                try self.writeStructDecl(stream, @enumFromInt(root_decl));
            }
        } else {
            try self.writeAll(stream, "{{}}", .{});
        }
    }

    fn writeStructDecl(self: *Writer, stream: anytype, inst_idx: Inst.Ref) !void {
        _ = inst_idx;
        try self.writeAll(stream, "hi\n", .{});
    }
};
