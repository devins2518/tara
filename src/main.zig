const std = @import("std");
const c = @import("c.zig");

pub fn main() !void {}

test "simple test" {
    _ = @import("tokenize.zig");
    _ = @import("Parser.zig");
    _ = @import("Ast.zig");
    std.testing.refAllDeclsRecursive(@This());
}
