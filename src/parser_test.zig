const std = @import("std");
const Ast = @import("Ast.zig");
const tokenize = @import("tokenize.zig");
const Parser = @import("Parser.zig");
const allocator = std.testing.allocator;
const Node = Ast.Node;
const Token = tokenize.Token;

fn runTestExpectSuccess(src: [:0]const u8, expected_nodes: []const Node, expected_extra_data: []const Node.Idx) !void {
    var t = tokenize.Tokenizer.init(src);
    var tokens = std.ArrayList(Token).init(allocator);
    defer tokens.deinit();
    while (true) {
        const token = t.next();
        try tokens.append(token);
        if (token.tag == .eof) break;
    }

    var p = Parser{
        .allocator = std.testing.allocator,
        .source = src,
        .tokens = try tokens.toOwnedSlice(),
    };
    defer p.deinit();
    defer p.nodes.deinit(allocator);
    defer allocator.free(p.tokens);
    try p.nodes.setCapacity(allocator, p.tokens.len);

    try p.parseRoot();

    for (expected_nodes, 0..) |e, i| {
        try std.testing.expectEqual(e, p.nodes.get(i));
    }
    for (expected_extra_data, p.extra_data.items) |e, a| {
        try std.testing.expectEqual(e, a);
    }
}

fn parseEmptyStruct() !void {
    const src =
        \\const In = struct {
        \\};
    ;
    const expected_nodes = [_]Node{
        .{ .tag = .root, .main_idx = 0, .data = .{ .lhs = @enumFromInt(0), .rhs = @enumFromInt(1) } }, // root
        .{ .tag = .struct_decl, .main_idx = 3, .data = .{ .lhs = @enumFromInt(0), .rhs = @enumFromInt(0) } }, // struct { ... }
        .{ .tag = .var_decl, .main_idx = 0, .data = .{ .lhs = @enumFromInt(0), .rhs = @enumFromInt(1) } }, // const In = struct { ... };
    };
    const expected_extra_data = [_]Node.Idx{@enumFromInt(2)};
    try runTestExpectSuccess(src, &expected_nodes, &expected_extra_data);
}

test parseEmptyStruct {
    try parseEmptyStruct();
}

fn parseStructWithFields() !void {
    const src =
        \\const In = struct {
        \\    a: sig,
        \\    b: sig,
        \\    cin: sig,
        \\};
    ;
    // TODO: main_token points to token idx
    const expected_nodes = [_]Node{
        .{ .tag = .root, .main_idx = 0, .data = .{ .lhs = @enumFromInt(3), .rhs = @enumFromInt(4) } }, // root
        .{ .tag = .identifier, .main_idx = 7, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // sig
        .{ .tag = .container_field, .main_idx = 5, .data = .{ .lhs = @enumFromInt(1), .rhs = Node.null_node } }, // a: sig
        .{ .tag = .identifier, .main_idx = 11, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // sig
        .{ .tag = .container_field, .main_idx = 9, .data = .{ .lhs = @enumFromInt(3), .rhs = Node.null_node } }, // b: sig
        .{ .tag = .identifier, .main_idx = 15, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // sig
        .{ .tag = .container_field, .main_idx = 13, .data = .{ .lhs = @enumFromInt(5), .rhs = Node.null_node } }, // cin: sig
        .{ .tag = .struct_decl, .main_idx = 3, .data = .{ .lhs = @enumFromInt(0), .rhs = @enumFromInt(3) } }, // struct { ... }
        .{ .tag = .var_decl, .main_idx = 0, .data = .{ .lhs = @enumFromInt(0), .rhs = @enumFromInt(7) } }, // const In = struct { ... };
    };
    const expected_extra_data = [_]Node.Idx{
        @enumFromInt(2), // struct field 0
        @enumFromInt(4), // struct field 1
        @enumFromInt(6), // struct field 2
        @enumFromInt(8), // root
    };
    try runTestExpectSuccess(src, &expected_nodes, &expected_extra_data);
}

test parseStructWithFields {
    try parseStructWithFields();
}

fn parseExprWithPrecedence() !void {
    // should be parsed as ((a + (b * c)) - (d / e))
    const src =
        \\const A = a + b * c - d / e;
    ;
    const expected_nodes = [_]Node{
        .{ .tag = .root, .main_idx = 0, .data = .{ .lhs = @enumFromInt(0), .rhs = @enumFromInt(1) } }, // root
        .{ .tag = .identifier, .main_idx = 3, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // a
        .{ .tag = .identifier, .main_idx = 5, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // b
        .{ .tag = .identifier, .main_idx = 7, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // c
        .{ .tag = .mul, .main_idx = 6, .data = .{ .lhs = @enumFromInt(2), .rhs = @enumFromInt(3) } }, // b * c
        .{ .tag = .add, .main_idx = 4, .data = .{ .lhs = @enumFromInt(1), .rhs = @enumFromInt(4) } }, // a + (b * c)
        .{ .tag = .identifier, .main_idx = 9, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // d
        .{ .tag = .identifier, .main_idx = 11, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // e
        .{ .tag = .div, .main_idx = 10, .data = .{ .lhs = @enumFromInt(6), .rhs = @enumFromInt(7) } }, // d / e
        .{ .tag = .sub, .main_idx = 8, .data = .{ .lhs = @enumFromInt(5), .rhs = @enumFromInt(8) } }, // (a + (b * c)) - (d / e)
        .{ .tag = .var_decl, .main_idx = 0, .data = .{ .lhs = Node.null_node, .rhs = @enumFromInt(9) } },
    };
    const expected_extra_data = [_]Node.Idx{@enumFromInt(10)};
    try runTestExpectSuccess(src, &expected_nodes, &expected_extra_data);
}

test parseExprWithPrecedence {
    try parseExprWithPrecedence();
}

fn parseExprWithPrecedenceAndParentheses() !void {
    // should be parsed as (((a + b) * c) - (d / e))
    const src =
        \\const A = (a + b) * c - d / e;
    ;
    const expected_nodes = [_]Node{
        .{ .tag = .root, .main_idx = 0, .data = .{ .lhs = @enumFromInt(0), .rhs = @enumFromInt(1) } }, // root
        .{ .tag = .identifier, .main_idx = 4, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // a
        .{ .tag = .identifier, .main_idx = 6, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // b
        .{ .tag = .add, .main_idx = 5, .data = .{ .lhs = @enumFromInt(1), .rhs = @enumFromInt(2) } }, // a + b
        .{ .tag = .identifier, .main_idx = 9, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // c
        .{ .tag = .mul, .main_idx = 8, .data = .{ .lhs = @enumFromInt(3), .rhs = @enumFromInt(4) } }, // (a + b) * c
        .{ .tag = .identifier, .main_idx = 11, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // d
        .{ .tag = .identifier, .main_idx = 13, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // e
        .{ .tag = .div, .main_idx = 12, .data = .{ .lhs = @enumFromInt(6), .rhs = @enumFromInt(7) } }, // d / e
        .{ .tag = .sub, .main_idx = 10, .data = .{ .lhs = @enumFromInt(5), .rhs = @enumFromInt(8) } }, // ((a + b) * c) - (d / e)
        .{ .tag = .var_decl, .main_idx = 0, .data = .{ .lhs = Node.null_node, .rhs = @enumFromInt(9) } },
    };
    const expected_extra_data = [_]Node.Idx{@enumFromInt(10)};
    try runTestExpectSuccess(src, &expected_nodes, &expected_extra_data);
}

test parseExprWithPrecedenceAndParentheses {
    try parseExprWithPrecedenceAndParentheses();
}

fn parseExprWithPrecedenceAndMemberAccess() !void {
    const src =
        \\const A = a.b;
        \\const B = c.d + (e.f - g.h);
    ;
    const expected_nodes = [_]Node{
        .{ .tag = .root, .main_idx = 0, .data = .{ .lhs = @enumFromInt(0), .rhs = @enumFromInt(2) } }, // root
        .{ .tag = .identifier, .main_idx = 3, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // a
        .{ .tag = .identifier, .main_idx = 5, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // b
        .{ .tag = .member, .main_idx = 4, .data = .{ .lhs = @enumFromInt(1), .rhs = @enumFromInt(2) } }, // a.b
        .{ .tag = .var_decl, .main_idx = 0, .data = .{ .lhs = Node.null_node, .rhs = @enumFromInt(3) } }, // const A = a.b
        .{ .tag = .identifier, .main_idx = 10, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // c
        .{ .tag = .identifier, .main_idx = 12, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // d
        .{ .tag = .member, .main_idx = 11, .data = .{ .lhs = @enumFromInt(5), .rhs = @enumFromInt(6) } }, // c.d
        .{ .tag = .identifier, .main_idx = 15, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // e
        .{ .tag = .identifier, .main_idx = 17, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // f
        .{ .tag = .member, .main_idx = 16, .data = .{ .lhs = @enumFromInt(8), .rhs = @enumFromInt(9) } }, // e.f
        .{ .tag = .identifier, .main_idx = 19, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // g
        .{ .tag = .identifier, .main_idx = 21, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // h
        .{ .tag = .member, .main_idx = 20, .data = .{ .lhs = @enumFromInt(11), .rhs = @enumFromInt(12) } }, // g.h
        .{ .tag = .sub, .main_idx = 18, .data = .{ .lhs = @enumFromInt(10), .rhs = @enumFromInt(13) } }, // e.f - g.h
        .{ .tag = .add, .main_idx = 13, .data = .{ .lhs = @enumFromInt(7), .rhs = @enumFromInt(14) } }, // c.d + (e.f - g.h)
        .{ .tag = .var_decl, .main_idx = 7, .data = .{ .lhs = Node.null_node, .rhs = @enumFromInt(15) } }, // const B = c.d + (e.f - g.h);
    };
    const expected_extra_data = [_]Node.Idx{ @enumFromInt(4), @enumFromInt(16) };
    try runTestExpectSuccess(src, &expected_nodes, &expected_extra_data);
}

test parseExprWithPrecedenceAndMemberAccess {
    try parseExprWithPrecedenceAndMemberAccess();
}

fn parseMultipleStructsAndModule() !void {
    const src =
        \\const In = struct {
        \\    a: bool,
        \\    b: bool,
        \\};
        \\const Out = struct {
        \\    c: bool,
        \\};
        \\const Mod = module(in: &In, out: &var Out) {
        \\    out.c = in.a & in.b;
        \\};
    ;
    const expected_nodes = [_]Node{
        .{ .tag = .root, .main_idx = 0, .data = .{ .lhs = @enumFromInt(10), .rhs = @enumFromInt(13) } }, // root
        .{ .tag = .identifier, .main_idx = 7, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // bool
        .{ .tag = .container_field, .main_idx = 5, .data = .{ .lhs = @enumFromInt(1), .rhs = Node.null_node } }, // a: bool
        .{ .tag = .identifier, .main_idx = 11, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // bool
        .{ .tag = .container_field, .main_idx = 9, .data = .{ .lhs = @enumFromInt(3), .rhs = Node.null_node } }, // b: bool
        .{ .tag = .struct_decl, .main_idx = 3, .data = .{ .lhs = @enumFromInt(0), .rhs = @enumFromInt(2) } }, // struct { ... }
        .{ .tag = .var_decl, .main_idx = 0, .data = .{ .lhs = Node.null_node, .rhs = @enumFromInt(5) } }, // const In = struct { ... };
        .{ .tag = .identifier, .main_idx = 22, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // bool
        .{ .tag = .container_field, .main_idx = 20, .data = .{ .lhs = @enumFromInt(7), .rhs = Node.null_node } }, // c: bool
        .{ .tag = .struct_decl, .main_idx = 18, .data = .{ .lhs = @enumFromInt(2), .rhs = @enumFromInt(3) } }, // struct { ... }
        .{ .tag = .var_decl, .main_idx = 15, .data = .{ .lhs = Node.null_node, .rhs = @enumFromInt(9) } }, // const Out = struct { ... };
        .{ .tag = .identifier, .main_idx = 34, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // In
        .{ .tag = .reference, .main_idx = 33, .data = .{ .lhs = @enumFromInt(11), .rhs = Node.null_node } }, // &In
        .{ .tag = .module_arg, .main_idx = 31, .data = .{ .lhs = @enumFromInt(12), .rhs = Node.null_node } }, // in: &In
        .{ .tag = .identifier, .main_idx = 40, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // Out
        .{ .tag = .reference, .main_idx = 38, .data = .{ .lhs = @enumFromInt(14), .rhs = Node.null_node } }, // &var Out
        .{ .tag = .module_arg, .main_idx = 36, .data = .{ .lhs = @enumFromInt(15), .rhs = Node.null_node } }, // out: &var Out
        .{ .tag = .identifier, .main_idx = 43, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // out
        .{ .tag = .identifier, .main_idx = 45, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // c
        .{ .tag = .member, .main_idx = 44, .data = .{ .lhs = @enumFromInt(17), .rhs = @enumFromInt(18) } }, // out.c
        .{ .tag = .identifier, .main_idx = 47, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // in
        .{ .tag = .identifier, .main_idx = 49, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // a
        .{ .tag = .member, .main_idx = 48, .data = .{ .lhs = @enumFromInt(20), .rhs = @enumFromInt(21) } }, // in.a
        .{ .tag = .identifier, .main_idx = 51, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // in
        .{ .tag = .identifier, .main_idx = 53, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // b
        .{ .tag = .member, .main_idx = 52, .data = .{ .lhs = @enumFromInt(23), .rhs = @enumFromInt(24) } }, // in.b
        .{ .tag = .bit_and, .main_idx = 50, .data = .{ .lhs = @enumFromInt(22), .rhs = @enumFromInt(25) } }, // in.a & in.b
        .{ .tag = .assignment, .main_idx = 46, .data = .{ .lhs = @enumFromInt(19), .rhs = @enumFromInt(26) } }, // out.c = in.a & in.b;
        .{ .tag = .module_decl, .main_idx = 29, .data = .{ .lhs = @enumFromInt(5), .rhs = @enumFromInt(8) } }, // module(...){ ... };
        .{ .tag = .var_decl, .main_idx = 26, .data = .{ .lhs = Node.null_node, .rhs = @enumFromInt(28) } }, // const Mod = module(...){ ... };
    };
    const expected_extra_data = [_]Node.Idx{
        @enumFromInt(2), // In.a
        @enumFromInt(4), // In.b
        @enumFromInt(8), // Out.c
        @enumFromInt(13), // in: &In
        @enumFromInt(16), // out: &Out
        @enumFromInt(3), // ModuleArgs.args_start
        @enumFromInt(5), // ModuleArgs.args_end
        @enumFromInt(27), // out.c = in.a & in.b
        @enumFromInt(7), // ModuleStatements.statements_start
        @enumFromInt(8), // ModuleStatements.statements_end
        @enumFromInt(6), // In
        @enumFromInt(10), // Out
        @enumFromInt(29), // Mod
    };
    try runTestExpectSuccess(src, &expected_nodes, &expected_extra_data);
}

test parseMultipleStructsAndModule {
    try parseMultipleStructsAndModule();
}

fn parseNumbers() !void {
    const src =
        \\const A = 3 + 4;
    ;
    const expected_nodes = [_]Node{
        .{ .tag = .root, .main_idx = 0, .data = .{ .lhs = @enumFromInt(0), .rhs = @enumFromInt(1) } }, // root
        .{ .tag = .int, .main_idx = 3, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // 3
        .{ .tag = .int, .main_idx = 5, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // 4
        .{ .tag = .add, .main_idx = 4, .data = .{ .lhs = @enumFromInt(1), .rhs = @enumFromInt(2) } }, // 3 + 4
        .{ .tag = .var_decl, .main_idx = 0, .data = .{ .lhs = Node.null_node, .rhs = @enumFromInt(3) } }, // const A = 3 + 4
    };
    const expected_extra_data = [_]Node.Idx{
        @enumFromInt(4),
    };
    try runTestExpectSuccess(src, &expected_nodes, &expected_extra_data);
}

test parseNumbers {
    try parseNumbers();
}
