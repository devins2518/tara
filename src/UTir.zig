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
