const std = @import("std");
const Ast = @import("Ast.zig");
const Node = Ast.Node;
const Allocator = std.mem.Allocator;
const tokenize = @import("tokenize.zig");
const Token = tokenize.Token;
const Parser = @This();
const assert = std.debug.assert;

const Error = error{} || Allocator.Error;

// Allocator used for general memory purposes
allocator: Allocator,
// Source of tokens used for error messaging
source: []const u8,
// Owned slice of tokens used to construct AST. Freed in `deinit`
tokens: []const Token,
// Current index into `tokens`
tok_idx: Ast.TokenIdx = 0,
// In-progress list of nodes being created
nodes: Node.List = .{},
// Scratchpad for tracking nodes being built
scratchpad: std.ArrayListUnmanaged(Node.Idx) = .{},
// Extra data if necessary for `nodes`
extra_data: std.ArrayListUnmanaged(Node.Idx) = .{},

// Parse the file as a root Tara file
// `self.nodes` must have a non-zero capacity
pub fn parseRoot(self: *Parser) !void {
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
    const scratch_top = self.scratchpad.items.len;
    defer self.scratchpad.shrinkRetainingCapacity(scratch_top);

    // TODO: add state machine for tracking seen fields and seen decls
    while (true) {
        switch (self.tokens[self.tok_idx].tag) {
            .keyword_pub => {
                self.tok_idx += 1;
                try self.scratchpad.append(self.allocator, try self.parseVarDecl());
            },
            .keyword_const,
            .keyword_var,
            => try self.scratchpad.append(self.allocator, try self.parseVarDecl()),
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

fn parseVarDecl(self: *Parser) !Node.Idx {
    const mutability = self.eat(.keyword_var) orelse
        self.eat(.keyword_const) orelse
        Node.null_node;

    _ = self.eat(.identifier);

    const type_node = if (self.eat(.colon) != null)
        try self.parseTypeExpr()
    else
        Node.null_node;

    _ = self.eat(.equal);

    const init_expr = try self.parseExpr();

    _ = self.eat(.semicolon);

    return self.addNode(.{
        .tag = .var_decl,
        .main_idx = mutability,
        .data = .{
            .lhs = type_node,
            .rhs = init_expr,
        },
    });
}

fn parseContainerField(self: *Parser) !Node.Idx {
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
        .start = @intCast(self.extra_data.items.len - list.len),
        .end = @intCast(self.extra_data.items.len),
    };
}

fn parseExpr(self: *Parser) !Node.Idx {
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
    const tag: Node.Tag = switch (self.tokens[self.tok_idx].tag) {
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
    switch (self.tokens[self.tok_idx].tag) {
        else => return self.parseTypeExpr(),
    }
}

fn parseTypeExpr(self: *Parser) !Node.Idx {
    return self.parseContainerDecl();
}

fn parseContainerDecl(self: *Parser) !Node.Idx {
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
    const result: Node.Idx = @intCast(self.nodes.len);
    try self.nodes.append(self.allocator, node);
    return result;
}

// Deinit `self` and free all related resources
pub fn deinit(self: *Parser) void {
    self.nodes.deinit(self.allocator);
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
            try p.nodes.setCapacity(allocator, p.tokens.len);

            try p.parseRoot();

            for (expected_nodes, 0..) |e, i| {
                std.debug.print("\n{}\n", .{p.nodes.get(i)});
                try std.testing.expectEqual(e, p.nodes.get(i));
            }
            for (expected_extra_data, p.extra_data.items) |e, a| {
                std.debug.print("\n{}\n", .{a});
                try std.testing.expectEqual(e, a);
            }
        }

        pub fn test0() !void {
            const src =
                \\const In = struct {
                \\};
            ;
            const expected_nodes = [_]Node{
                .{ .tag = .root, .main_idx = 0, .data = .{ .lhs = 0, .rhs = 1 } }, // root
                .{ .tag = .struct_decl, .main_idx = 3, .data = .{ .lhs = 0, .rhs = 0 } }, // struct { ... }
                .{ .tag = .var_decl, .main_idx = 0, .data = .{ .lhs = 0, .rhs = 1 } }, // const In = struct { ... };
            };
            const expected_extra_data = [_]Node.Idx{2};
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
                .{ .tag = .root, .main_idx = 0, .data = .{ .lhs = 3, .rhs = 4 } }, // root
                .{ .tag = .container_field, .main_idx = 5, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // a: sig
                .{ .tag = .container_field, .main_idx = 9, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // b: sig
                .{ .tag = .container_field, .main_idx = 13, .data = .{ .lhs = Node.null_node, .rhs = Node.null_node } }, // cin: sig
                .{ .tag = .struct_decl, .main_idx = 3, .data = .{ .lhs = 0, .rhs = 3 } }, // struct { ... }
                .{ .tag = .var_decl, .main_idx = 0, .data = .{ .lhs = 0, .rhs = 4 } }, // const In = struct { ... };
            };
            const expected_extra_data = [_]Node.Idx{ 1, 2, 3, 5 };
            try doTheTest(src, &expected_nodes, &expected_extra_data);
        }
    };

    try tests.test0();
    try tests.test1();
}
