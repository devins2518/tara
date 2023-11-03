const std = @import("std");
const c = @import("c.zig");

pub fn main() !void {}

test "simple test" {
    _ = @import("c.zig");
    _ = @import("tokenize.zig");
    std.testing.refAllDeclsRecursive(@This());
}
