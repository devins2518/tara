const std = @import("std");
const Ast = @import("Ast.zig");
const Node = Ast.Node;
const Allocator = std.mem.Allocator;
const tokenize = @import("tokenize.zig");
const Token = tokenize.Token;
const Parser = @This();
const assert = std.debug.assert;

const Error = error{} || Allocator.Error;

const log = std.log.scoped(.Parser);

// Allocator used for general memory purposes
allocator: Allocator,
// Source of tokens used for error messaging
source: []const u8,
// Owned slice of tokens used to construct AST. Freed in `deinit`
tokens: []const Token,
// Current index into `tokens`
tok_idx: Ast.TokenIdx = 0,
// In-progress list of nodes being created. Expected to be taken and owned by
// caller. Not freed in `deinit`
nodes: Node.List = .{},
// Scratchpad for tracking nodes being built
scratchpad: std.ArrayListUnmanaged(Node.Idx) = .{},
// Extra data if necessary for `nodes`
extra_data: std.ArrayListUnmanaged(Node.Idx) = .{},

// Parse the file as a root Tara file
// `self.nodes` must have a non-zero capacity
pub fn parseRoot(self: *Parser) !void {
    log.debug("parseRoot\n", .{});
    self.nodes.appendAssumeCapacity(.{
        .tag = .root,
        .main_idx = 0,
        .data = undefined,
    });
    const root_members = try self.parseContainerMembers();
    if (self.tokens[self.tok_idx].tag != .eof) {
        // TODO: error
    }
    self.nodes.items(.data)[0] = .{
        .lhs = root_members.start,
        .rhs = root_members.end,
    };
}

fn parseContainerMembers(self: *Parser) Error!Node.SubList {
    log.debug("parseContainerMembers\n", .{});
    const scratch_top = self.scratchpad.items.len;
    defer self.scratchpad.shrinkRetainingCapacity(scratch_top);

    // TODO: add state machine for tracking seen fields and seen decls
    while (true) {
        switch (self.tokens[self.tok_idx].tag) {
            .keyword_pub => {
                self.tok_idx += 1;
                try self.scratchpad.append(self.allocator, try self.parseVarDeclStatement());
            },
            .keyword_const,
            .keyword_var,
            => try self.scratchpad.append(self.allocator, try self.parseVarDeclStatement()),
            .identifier => try self.scratchpad.append(self.allocator, try self.parseContainerField()),
            .eof, .rbrace => break,
            else => {
                std.debug.print("[ERROR]: unhandled container member: {s}", .{@tagName(self.tokens[self.tok_idx].tag)});
                break;
            },
        }
    }

    // TODO: decide on API for variable length optimizations
    // const num_items = self.scratchpad.items.len - scratch_top;
    // if (num_items == 0)
    //     return .{ .start = Node.null_node, .end = Node.null_node }
    // else if (num_items == 1)
    //     return .{ .start = Node.null_node, .end = self.scratchpad.items[scratch_top] }
    // else
    return try self.scratchToSubList(scratch_top);
}

fn parseVarDeclExpr(self: *Parser) !Node.Idx {
    log.debug("parseVarDecl\n", .{});
    // TODO: error on no mutability token
    const mutability = self.eat(.keyword_var) orelse
        self.eat(.keyword_const) orelse
        0;

    _ = self.eat(.identifier);

    const type_node = if (self.eat(.colon) != null)
        try self.parseTypeExpr()
    else
        Node.null_node;

    _ = self.eat(.equal);

    const init_expr = try self.parseExpr();

    return self.addNode(.{
        .tag = .var_decl,
        .main_idx = mutability,
        .data = .{
            .lhs = type_node,
            .rhs = init_expr,
        },
    });
}

fn parseVarDeclStatement(self: *Parser) !Node.Idx {
    log.debug("parseVarDeclStatement\n", .{});

    const var_decl = try self.parseVarDeclExpr();

    _ = self.eat(.semicolon);

    return var_decl;
}

fn parseContainerField(self: *Parser) !Node.Idx {
    log.debug("parseContainerField\n", .{});
    const main_idx = self.eat(.identifier) orelse return Node.null_node;
    _ = self.eat(.colon);
    const type_expr = try self.parseTypeExpr();
    const expr = if (self.eat(.equal)) |_|
        try self.parseExpr()
    else
        Node.null_node;
    _ = self.eat(.comma);

    return self.addNode(.{
        .tag = .container_field,
        .main_idx = main_idx,
        .data = .{
            .lhs = type_expr,
            .rhs = expr,
        },
    });
}

fn scratchToSubList(self: *Parser, top: usize) !Node.SubList {
    const list = self.scratchpad.items[top..];
    try self.extra_data.appendSlice(self.allocator, list);
    return Node.SubList{
        .start = @enumFromInt(self.extra_data.items.len - list.len),
        .end = @enumFromInt(self.extra_data.items.len),
    };
}

fn parseExpr(self: *Parser) Error!Node.Idx {
    return try self.parseExprWithPrecedence(0);
}

const Assoc = enum { left, none };

const OperationInfo = struct {
    precedence: i8,
    tag: Node.Tag,
    associativity: Assoc,
};

const precedence_table = std.enums.directEnumArrayDefault(
    Token.Tag,
    OperationInfo,
    .{ .precedence = -1, .tag = .root, .associativity = .none },
    0,
    .{
        .keyword_or = .{ .precedence = 10, .tag = .@"or", .associativity = .none },

        .keyword_and = .{ .precedence = 20, .tag = .@"and", .associativity = .none },

        .op_eq = .{ .precedence = 30, .tag = .eq, .associativity = .none },
        .op_neq = .{ .precedence = 30, .tag = .neq, .associativity = .none },
        .op_lt = .{ .precedence = 30, .tag = .lt, .associativity = .none },
        .op_gt = .{ .precedence = 30, .tag = .gt, .associativity = .none },
        .op_lte = .{ .precedence = 30, .tag = .lte, .associativity = .none },
        .op_gte = .{ .precedence = 30, .tag = .gte, .associativity = .none },

        .op_and = .{ .precedence = 40, .tag = .bit_and, .associativity = .left },
        .op_pipe = .{ .precedence = 40, .tag = .bit_or, .associativity = .left },
        .op_xor = .{ .precedence = 40, .tag = .bit_xor, .associativity = .left },

        .op_plus = .{ .precedence = 50, .tag = .add, .associativity = .left },
        .op_minus = .{ .precedence = 50, .tag = .sub, .associativity = .left },

        .op_star = .{ .precedence = 60, .tag = .mul, .associativity = .left },
        .op_slash = .{ .precedence = 60, .tag = .div, .associativity = .left },
    },
);

fn parseExprWithPrecedence(self: *Parser, min_precedence: i8) !Node.Idx {
    log.debug("parseExprWithPrecedence\n", .{});
    assert(min_precedence >= 0);
    var node = try self.parsePrefixExpr();
    if (node == Node.null_node) return node;

    while (true) {
        const tag = self.tokens[self.tok_idx].tag;
        const info = precedence_table[@intFromEnum(tag)];
        if (info.precedence < min_precedence) {
            break;
        }

        const operation_token = self.nextToken();
        const rhs = try self.parseExprWithPrecedence(info.precedence + 1);
        // TODO: handle null rhs

        node = try self.addNode(.{
            .tag = info.tag,
            .main_idx = operation_token,
            .data = .{
                .lhs = node,
                .rhs = rhs,
            },
        });
    }

    return node;
}

fn parsePrefixExpr(self: *Parser) !Node.Idx {
    log.debug("parsePrefixExpr\n", .{});
    const tag: Node.Tag = switch (self.tokens[self.tok_idx].tag) {
        .lparen => {
            _ = self.nextToken();
            const expr = self.parseExpr();
            _ = self.eat(.rparen);
            return expr;
        },
        else => return self.parsePrimaryExpr(),
    };
    return self.addNode(.{
        .tag = tag,
        .main_token = self.nextToken(),
        .data = .{
            .lhs = try self.expectPrefixExpr(),
            .rhs = undefined,
        },
    });
}

fn parsePrimaryExpr(self: *Parser) !Node.Idx {
    log.debug("parsePrimaryExpr\n", .{});
    switch (self.tokens[self.tok_idx].tag) {
        .identifier => return self.addNode(.{
            .tag = .identifier,
            .main_idx = self.nextToken(),
            .data = .{ .lhs = Node.null_node, .rhs = Node.null_node },
        }),
        else => return self.parseTypeExpr(),
    }
}

fn parseTypeExpr(self: *Parser) Error!Node.Idx {
    log.debug("parseTypeExpr\n", .{});
    return switch (self.tokens[self.tok_idx].tag) {
        .keyword_struct, .keyword_union => try self.parseContainerDecl(),
        .identifier => return self.addNode(.{
            .tag = .identifier,
            .main_idx = self.nextToken(),
            .data = .{ .lhs = Node.null_node, .rhs = Node.null_node },
        }),
        else => Node.null_node,
    };
}

fn parseContainerDecl(self: *Parser) !Node.Idx {
    log.debug("parseContainerDecl\n", .{});
    const main_idx = self.nextToken();
    const container_decl: Node.Tag = switch (self.tokens[main_idx].tag) {
        .keyword_struct => .struct_decl,
        .keyword_union => @panic("TODO: add Node.Tag.union_decl"),
        else => return Node.null_node,
    };
    _ = self.eat(.lbrace);
    const members = try self.parseContainerMembers();
    _ = self.eat(.rbrace);

    return self.addNode(.{
        .tag = container_decl,
        .main_idx = main_idx,
        .data = .{
            .lhs = members.start,
            .rhs = members.end,
        },
    });
}

fn eat(self: *Parser, tag: Token.Tag) ?Ast.TokenIdx {
    return if (self.tokens[self.tok_idx].tag == tag) self.nextToken() else null;
}

fn nextToken(self: *Parser) Ast.TokenIdx {
    const result = self.tok_idx;
    self.tok_idx += 1;
    return result;
}

fn addNode(self: *Parser, node: Ast.Node) !Node.Idx {
    const result: Node.Idx = @enumFromInt(self.nodes.len);
    try self.nodes.append(self.allocator, node);
    return result;
}

// Deinit `self` and free all related resources
pub fn deinit(self: *Parser) void {
    self.extra_data.deinit(self.allocator);
    self.scratchpad.deinit(self.allocator);
    self.allocator.free(self.tokens);
    self.* = undefined;
}

test Parser {
    const allocator = std.testing.allocator;
    const tests = struct {
        fn doTheTest(src: [:0]const u8, expected_nodes: []const Node, expected_extra_data: []const Node.Idx) !void {
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
            try p.nodes.setCapacity(allocator, p.tokens.len);

            try p.parseRoot();

            for (expected_nodes, 0..) |e, i| {
                try std.testing.expectEqual(e, p.nodes.get(i));
            }
            for (expected_extra_data, p.extra_data.items) |e, a| {
                try std.testing.expectEqual(e, a);
            }
        }

        pub fn test0() !void {
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
            try doTheTest(src, &expected_nodes, &expected_extra_data);
        }

        pub fn test1() !void {
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
            try doTheTest(src, &expected_nodes, &expected_extra_data);
        }

        pub fn test2() !void {
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
            try doTheTest(src, &expected_nodes, &expected_extra_data);
        }

        pub fn test3() !void {
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
            try doTheTest(src, &expected_nodes, &expected_extra_data);
        }
    };

    try tests.test0();
    try tests.test1();
    try tests.test2();
    try tests.test3();
}
