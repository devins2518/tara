const std = @import("std");
const Ast = @import("Ast.zig");
const UTir = @import("UTir.zig");
const UTirGen = @import("UTirGen.zig");
const Inst = UTir.Inst;
const Allocator = std.mem.Allocator;

const allocator = std.testing.allocator;

fn runTestExpectSuccess(src: [:0]const u8, expected_utir: []const Inst, expected_extra_data: []const u32, expected_utir_str: []const u8, debug_print: bool) !void {
    var ast = try Ast.parse(allocator, src);
    defer ast.deinit(allocator);
    var utir = try UTirGen.genUTir(allocator, &ast);
    defer utir.deinit(allocator);

    if (debug_print) {
        std.debug.print("{}\n\n", .{utir});
    }

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

fn testEmptyGen() !void {
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
    try runTestExpectSuccess(src, &expected_utir, &expected_extra_data, expected_utir_str, false);
}

test testEmptyGen {
    try testEmptyGen();
}

fn testManyDeclGen() !void {
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
    try runTestExpectSuccess(src, &expected_utir, &expected_extra_data, expected_utir_str, false);
}

test testManyDeclGen {
    try testManyDeclGen();
}

fn testNestedDeclGen() !void {
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
    try runTestExpectSuccess(src, &expected_utir, &expected_extra_data, expected_utir_str, false);
}

test testNestedDeclGen {
    try testNestedDeclGen();
}

fn testDeclValGen() !void {
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
    try runTestExpectSuccess(src, &expected_utir, &expected_extra_data, expected_utir_str, false);
}

test testDeclValGen {
    try testDeclValGen();
}

fn testManyFieldGen() !void {
    const src =
        \\a: bool,
        \\b: bool,
        \\c: u8,
    ;
    const expected_utir = [_]Inst{
        .{ .struct_decl = .{ .ed_idx = @enumFromInt(0) } }, // Root
        .{ .decl_val = .{ .string_bytes_idx = @enumFromInt(1) } }, // a: bool
        .{ .decl_val = .{ .string_bytes_idx = @enumFromInt(1) } }, // b: bool
        .{ .decl_val = .{ .string_bytes_idx = @enumFromInt(4) } }, // c: u8
    };
    const expected_extra_data = [_]u32{
        3, // Root.fields
        0, // Root.decls
        0, // a.name
        1, // a.type
        2, // b.name
        2, // b.type
        3, // c.name
        3, // c.type
    };
    const expected_utir_str =
        \\%0 = struct_decl({
        \\    %1 = decl_val("bool")
        \\    a : %1
        \\    %2 = decl_val("bool")
        \\    b : %2
        \\    %3 = decl_val("u8")
        \\    c : %3
        \\})
        \\
    ;
    try runTestExpectSuccess(src, &expected_utir, &expected_extra_data, expected_utir_str, false);
}

test testManyFieldGen {
    try testManyFieldGen();
}

fn testMixedFieldDeclGen() !void {
    const src =
        \\a: bool,
        \\b: bool,
        \\const In = struct {};
    ;
    const expected_utir = [_]Inst{
        .{ .struct_decl = .{ .ed_idx = @enumFromInt(2) } }, // Root
        .{ .decl_val = .{ .string_bytes_idx = @enumFromInt(1) } }, // a: bool
        .{ .decl_val = .{ .string_bytes_idx = @enumFromInt(1) } }, // b: bool
        .{ .struct_decl = .{ .ed_idx = @enumFromInt(0) } }, // In
    };
    const expected_extra_data = [_]u32{
        0, // Root.In.fields
        0, // Root.In.decls
        2, // Root.fields
        1, // Root.decls
        0, // a.name
        1, // a.type
        2, // b.name
        2, // b.type
        3, // In
    };
    const expected_utir_str =
        \\%0 = struct_decl({
        \\    %1 = decl_val("bool")
        \\    a : %1
        \\    %2 = decl_val("bool")
        \\    b : %2
        \\    %3 = struct_decl({})
        \\})
        \\
    ;
    try runTestExpectSuccess(src, &expected_utir, &expected_extra_data, expected_utir_str, false);
}

test testMixedFieldDeclGen {
    try testMixedFieldDeclGen();
}

fn testNumberGen() !void {
    const src =
        \\const A = 3 + 4;
    ;
    const expected_utir = [_]Inst{
        .{ .struct_decl = .{ .ed_idx = @enumFromInt(0) } }, // Root
        .{ .int_small = .{ .int = 3 } }, // 3
        .{ .int_small = .{ .int = 4 } }, // 4
        .{ .add = .{ .lhs = @enumFromInt(1), .rhs = @enumFromInt(2) } }, // 4
    };
    const expected_extra_data = [_]u32{
        0, // Root.fields
        1, // Root.decls
        3, // A
    };
    const expected_utir_str =
        \\%0 = struct_decl({
        \\    %1 = int(3)
        \\    %2 = int(4)
        \\    %3 = add(%1, %2)
        \\})
        \\
    ;
    try runTestExpectSuccess(src, &expected_utir, &expected_extra_data, expected_utir_str, false);
}

test testNumberGen {
    try testNumberGen();
}

fn testEmptyModuleGen() !void {
    const src =
        \\const Mod = module {};
    ;
    const expected_utir = [_]Inst{
        .{ .struct_decl = .{ .ed_idx = @enumFromInt(2) } }, // Root
        .{ .module_decl = .{ .ed_idx = @enumFromInt(0) } }, // Mod
    };
    const expected_extra_data = [_]u32{
        0, // Mod.fields
        0, // Mod.decls
        0, // Root.fields
        1, // Root.decls
        1, // Mod
    };
    const expected_utir_str =
        \\%0 = struct_decl({
        \\    %1 = module_decl({})
        \\})
        \\
    ;
    try runTestExpectSuccess(src, &expected_utir, &expected_extra_data, expected_utir_str, false);
}

test testEmptyModuleGen {
    try testEmptyModuleGen();
}
