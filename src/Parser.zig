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

    while (true) {
        switch (self.tokens[self.tok_idx].tag) {
            .keyword_pub => {
                self.tok_idx += 1;
                try self.scratchpad.append(self.allocator, try self.parseVarDecl());
            },
            .keyword_const,
            .keyword_var,
            => try self.scratchpad.append(self.allocator, try self.parseVarDecl()),
            .eof, .rbrace => break,
            else => {
                std.debug.print("[ERROR]: unhandled container member: {s}", .{@tagName(self.tokens[self.tok_idx].tag)});
                break;
            },
        }
    }

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

fn parseExprWithPrecedence(self: *Parser, min_precedence: isize) !Node.Idx {
    assert(min_precedence >= 0);
    _ = self;
    @panic("TODO parseExprWithPrecedence");
}

fn parseTypeExpr(self: *Parser) !Node.Idx {
    _ = self;
    @panic("TODO parseTypeExpr");
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
    self.allocator.free(self.tokens);
    self.* = undefined;
}

test {
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
            // TODO: main_token points to token idx
            const expected_nodes = [_]Node{
                .{ .tag = .root, .main_idx = 0, .data = .{ .lhs = 1, .rhs = 3 } }, // root
                .{ .tag = .var_decl, .main_idx = 1, .data = .{ .lhs = 2, .rhs = 2 } }, // const In = struct { ... };
                .{ .tag = .struct_decl, .main_idx = 3, .data = .{ .lhs = 0, .rhs = 0 } }, // struct { ... }
            };
            const expected_extra_data = [_]Node.Idx{};
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
                .{ .tag = .root, .main_idx = 0, .data = .{ .lhs = 1, .rhs = 5 } }, // root
                .{ .tag = .var_decl, .main_idx = 1, .data = .{ .lhs = 2, .rhs = 2 } }, // const In = struct { ... };
                .{ .tag = .struct_decl, .main_idx = undefined, .data = .{ .lhs = undefined, .rhs = undefined } }, // struct { ... }
                .{ .tag = .struct_field, .main_idx = undefined, .data = .{ .lhs = Node.null_node, .rhs = undefined } }, // a: sig
                .{ .tag = .struct_field, .main_idx = undefined, .data = .{ .lhs = Node.null_node, .rhs = undefined } }, // b: sig
                .{ .tag = .struct_field, .main_idx = undefined, .data = .{ .lhs = Node.null_node, .rhs = undefined } }, // c: sig
            };
            const expected_extra_data = [_]Node.Idx{
                3, 5, 0, 0, // StructBody for In
            };
            try doTheTest(src, &expected_nodes, &expected_extra_data);
        }
    };

    try tests.test0();
    // try tests.test1();
}
