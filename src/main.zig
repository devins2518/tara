const std = @import("std");
const Ast = @import("Ast.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer std.debug.assert(gpa.deinit() == .ok);
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    if (args.len != 2) {
        std.debug.print("[ERROR] Expected ./prog <filename>\n", .{});
        std.process.cleanExit();
        std.process.exit(1);
    }

    const file = try std.fs.cwd().openFile(args[1], .{});
    defer file.close();
    const contents = try file.readToEndAllocOptions(allocator, std.math.maxInt(usize), null, @alignOf(u8), 0);
    defer allocator.free(contents);

    var ast = try Ast.parse(allocator, contents);
    defer ast.deinit(allocator);
}

test "simple test" {
    _ = @import("tokenize.zig");
    _ = @import("Parser.zig");
    _ = @import("Ast.zig");
    _ = @import("UTirGen.zig");
    _ = @import("UTir.zig");
    _ = @import("mlir/ir.zig");
    std.testing.refAllDeclsRecursive(@This());
}
