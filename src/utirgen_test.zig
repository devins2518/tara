const std = @import("std");
const Ast = @import("Ast.zig");
const UTir = @import("UTir.zig");
const UTirGen = @import("UTirGen.zig");
const Inst = UTir.Inst;
const Allocator = std.mem.Allocator;

const allocator = std.testing.allocator;

fn runTestExpectSuccess(src: [:0]const u8, maybe_expected_utir: ?[]const Inst, maybe_expected_extra_data: ?[]const u32, expected_utir_str: []const u8, debug_print: bool) !void {
    var ast = try Ast.parse(allocator, src);
    defer ast.deinit(allocator);
    var utir = try UTirGen.genUTir(allocator, &ast);
    defer utir.deinit(allocator);

    if (debug_print) {
        std.debug.print("{}\n\n", .{utir});
    }

    if (maybe_expected_utir) |expected_utir| {
        for (expected_utir, 0..) |e, i| {
            try std.testing.expectEqual(e, utir.instructions.get(i));
        }
    }

    if (maybe_expected_extra_data) |expected_extra_data| {
        for (expected_extra_data, utir.extra_data) |e, a| {
            try std.testing.expectEqual(e, a);
        }
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
        .{ .struct_decl = .{ .ed_idx = @enumFromInt(5) } }, // Root
        .{ .inline_block = .{ .ed_idx = @enumFromInt(0) } }, // blk
        .{ .int_small = .{ .int = 3 } }, // 3
        .{ .int_small = .{ .int = 4 } }, // 4
        .{ .add = .{ .lhs = @enumFromInt(2), .rhs = @enumFromInt(3) } }, // 3 + 4
        .{ .inline_block_break = .{ .lhs = @enumFromInt(1), .rhs = @enumFromInt(4) } }, // break 3 + 4
    };
    const expected_extra_data = [_]u32{
        4, // Block.instrs
        2, // 3
        3, // 4
        4, // 3 + 4
        5, // break 3 + 4
        0, // Root.fields
        1, // Root.decls
        1, // A
    };
    const expected_utir_str =
        \\%0 = struct_decl({
        \\    %1 = inline_block({
        \\        %2 = int(3)
        \\        %3 = int(4)
        \\        %4 = add(%2, %3)
        \\        %5 = inline_block_break(%1, %4)
        \\    })
        \\})
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
    ;
    try runTestExpectSuccess(src, &expected_utir, &expected_extra_data, expected_utir_str, false);
}

test testEmptyModuleGen {
    try testEmptyModuleGen();
}

fn testModuleWithFieldsGen() !void {
    const src =
        \\const Mod = module {
        \\    a: &bool,
        \\    b: &bool,
        \\};
    ;
    const expected_utir = [_]Inst{
        .{ .struct_decl = .{ .ed_idx = @enumFromInt(14) } }, // Root
        .{ .module_decl = .{ .ed_idx = @enumFromInt(8) } }, // Mod
        .{ .inline_block = .{ .ed_idx = @enumFromInt(0) } }, // blk
        .{ .decl_val = .{ .string_bytes_idx = @enumFromInt(1) } }, // bool
        .{ .ref_ty = .{ .child = @enumFromInt(3) } }, // &bool
        .{ .inline_block_break = .{ .lhs = @enumFromInt(2), .rhs = @enumFromInt(4) } }, // break &bool
        .{ .inline_block = .{ .ed_idx = @enumFromInt(4) } }, // blk
        .{ .decl_val = .{ .string_bytes_idx = @enumFromInt(1) } }, // bool
        .{ .ref_ty = .{ .child = @enumFromInt(7) } }, // &bool
        .{ .inline_block_break = .{ .lhs = @enumFromInt(6), .rhs = @enumFromInt(8) } }, // break &bool
    };
    const expected_extra_data = [_]u32{
        3, // Block.instrs
        3, // bool
        4, // &bool
        5, // break &bool
        3, // Block.instrs
        7, // bool
        8, // &bool
        9, // break &bool
        2, // Mod.fields
        0, // Mod.decls
        0, // Mod.a.name
        2, // Mod.a.type
        2, // Mod.b.name
        6, // Mod.b.type
        0, // Root.fields
        1, // Root.decls
        1, // Mod
    };
    const expected_utir_str =
        \\%0 = struct_decl({
        \\    %1 = module_decl({
        \\        %2 = inline_block({
        \\            %3 = decl_val("bool")
        \\            %4 = ref_ty(%3)
        \\            %5 = inline_block_break(%2, %4)
        \\        })
        \\        a : %2
        \\        %6 = inline_block({
        \\            %7 = decl_val("bool")
        \\            %8 = ref_ty(%7)
        \\            %9 = inline_block_break(%6, %8)
        \\        })
        \\        b : %6
        \\    })
        \\})
    ;
    try runTestExpectSuccess(src, &expected_utir, &expected_extra_data, expected_utir_str, false);
}

test testModuleWithFieldsGen {
    try testModuleWithFieldsGen();
}

fn testModuleWithCombGen() !void {
    const src =
        \\const Mod = module {
        \\    pub comb a(b: &u1, c: &u1) u1 {
        \\        return b & c;
        \\    }
        \\};
    ;
    const expected_utir_str =
        \\%0 = struct_decl({
        \\    %1 = module_decl({
        \\        %2 = subroutine_decl(
        \\            "a",
        \\            {
        \\                %3 = inline_block({
        \\                    %4 = decl_val("u1")
        \\                    %5 = ref_ty(%4)
        \\                    %6 = inline_block_break(%3, %5)
        \\                })
        \\                b : %3
        \\                %7 = inline_block({
        \\                    %8 = decl_val("u1")
        \\                    %9 = ref_ty(%8)
        \\                    %10 = inline_block_break(%7, %9)
        \\                })
        \\                c : %7
        \\            },
        \\            %11 = decl_val("u1"),
        \\            {
        \\                %12 = inline_block({
        \\                    %13 = inline_block({
        \\                        %14 = decl_val("b")
        \\                        %15 = decl_val("c")
        \\                        %16 = bit_and(%14, %15)
        \\                        %17 = inline_block_break(%13, %16)
        \\                    })
        \\                    %18 = return(%13)
        \\                    %19 = inline_block_break(%12, %18)
        \\                })
        \\            }
        \\        )
        \\    })
        \\})
    ;
    try runTestExpectSuccess(src, null, null, expected_utir_str, false);
}

test testModuleWithCombGen {
    try testModuleWithCombGen();
}
