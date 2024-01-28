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

fn parseModuleOnlyFields() !void {
    const src =
        \\const Mod = module {
        \\    a: u1,
        \\    b: u1,
        \\};
    ;
    const expected_nodes = [_]Node{
        .{ .tag = .root, .main_idx = 0, .data = .{ .lhs = @enumFromInt(2), .rhs = @enumFromInt(3) } }, // root,
        .{ .tag = .identifier, .main_idx = 7, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // u1
        .{ .tag = .container_field, .main_idx = 5, .data = .{ .lhs = @enumFromInt(1), .rhs = Node.null_node } }, // a: u1
        .{ .tag = .identifier, .main_idx = 11, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // u1
        .{ .tag = .container_field, .main_idx = 9, .data = .{ .lhs = @enumFromInt(3), .rhs = Node.null_node } }, // b: u1
        .{ .tag = .module_decl, .main_idx = 3, .data = .{ .lhs = @enumFromInt(0), .rhs = @enumFromInt(2) } }, // module { ... }
        .{ .tag = .var_decl, .main_idx = 0, .data = .{ .lhs = @enumFromInt(0), .rhs = @enumFromInt(5) } }, // const Mod = module { ... };
    };
    const expected_extra_data = [_]Node.Idx{
        @enumFromInt(2), // struct field 0
        @enumFromInt(4), // struct field 1
        @enumFromInt(6), // root
    };
    try runTestExpectSuccess(src, &expected_nodes, &expected_extra_data);
}

test parseModuleOnlyFields {
    try parseModuleOnlyFields();
}

fn parseModuleWithFieldsAndDecls() !void {
    const src =
        \\const Mod = module {
        \\    a: u1,
        \\    b: u1,
        \\
        \\    const A = 2;
        \\};
    ;
    const expected_nodes = [_]Node{
        .{ .tag = .root, .main_idx = 0, .data = .{ .lhs = @enumFromInt(3), .rhs = @enumFromInt(4) } }, // root,
        .{ .tag = .identifier, .main_idx = 7, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // u1
        .{ .tag = .container_field, .main_idx = 5, .data = .{ .lhs = @enumFromInt(1), .rhs = Node.null_node } }, // a: u1
        .{ .tag = .identifier, .main_idx = 11, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // u1
        .{ .tag = .container_field, .main_idx = 9, .data = .{ .lhs = @enumFromInt(3), .rhs = Node.null_node } }, // b: u1
        .{ .tag = .int, .main_idx = 16, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // 2
        .{ .tag = .var_decl, .main_idx = 13, .data = .{ .lhs = Node.null_node, .rhs = @enumFromInt(5) } }, // const A = 2;
        .{ .tag = .module_decl, .main_idx = 3, .data = .{ .lhs = @enumFromInt(0), .rhs = @enumFromInt(3) } }, // module { ... }
        .{ .tag = .var_decl, .main_idx = 0, .data = .{ .lhs = @enumFromInt(0), .rhs = @enumFromInt(7) } }, // const Mod = module { ... };
    };
    const expected_extra_data = [_]Node.Idx{
        @enumFromInt(2), // struct field 0
        @enumFromInt(4), // struct field 1
        @enumFromInt(6), // Mod.A
        @enumFromInt(8), // root
    };
    try runTestExpectSuccess(src, &expected_nodes, &expected_extra_data);
}

test parseModuleWithFieldsAndDecls {
    try parseModuleWithFieldsAndDecls();
}

fn parseModuleWithPubComb() !void {
    const src =
        \\const Mod = module {
        \\    pub comb a(a: &bool, b: &bool) bool {
        \\        return a & b;
        \\    }
        \\};
    ;
    const expected_nodes = [_]Node{};
    const expected_extra_data = [_]Node.Idx{};
    try runTestExpectSuccess(src, &expected_nodes, &expected_extra_data);
}

test parseModuleWithPubComb {
    // try parseMultipleStructsWithModuleAndPubComb();
    return error.SkipZigTest;
}
