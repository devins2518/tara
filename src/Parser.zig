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
// Scratchpad for tracking nodes being built. This is often used to temporarily
// store nodes indices which will be pushed into `extra_data`.
// See `parseContainerMembers` for an example.
scratchpad: std.ArrayListUnmanaged(Node.Idx) = .{},
// Extra data if necessary for `nodes`. This is often used to store
// non-contiguous node indices during parsing.
// See `parseContainerMembers` for an example.
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

fn parseAssignmentExpr(self: *Parser) !Node.Idx {
    log.debug("parseAssignmentExpr\n", .{});

    const lhs = try self.parseExpr();

    const main_idx = self.eat(.equal) orelse {
        log.debug("found {}", .{self.tokens[self.tok_idx].tag});
        @panic("TODO: support more assignment_ops!");
    };

    const rhs = try self.parseExpr();

    return self.addNode(.{
        .tag = .assignment,
        .main_idx = main_idx,
        .data = .{
            .lhs = lhs,
            .rhs = rhs,
        },
    });
}

fn parseAssignmentStatement(self: *Parser) !Node.Idx {
    log.debug("parseAssignmentStatement\n", .{});

    const assignment_expr = try self.parseAssignmentExpr();

    _ = self.eat(.semicolon);

    return assignment_expr;
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

        .period = .{ .precedence = 100, .tag = .member, .associativity = .none },
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
    return switch (self.tokens[self.tok_idx].tag) {
        .identifier => self.addNode(.{
            .tag = .identifier,
            .main_idx = self.nextToken(),
            .data = .{ .lhs = Node.null_node, .rhs = Node.null_node },
        }),
        .number => self.addNode(.{
            .tag = .int,
            .main_idx = self.nextToken(),
            .data = .{ .lhs = Node.null_node, .rhs = Node.null_node },
        }),
        else => return self.parseTypeExpr(),
    };
}

fn parseTypeExpr(self: *Parser) Error!Node.Idx {
    log.debug("parseTypeExpr\n", .{});
    return switch (self.tokens[self.tok_idx].tag) {
        .keyword_module => try self.parseModuleDecl(),
        .keyword_struct, .keyword_union => try self.parseContainerDecl(),
        .identifier => return self.addNode(.{
            .tag = .identifier,
            .main_idx = self.nextToken(),
            .data = .{ .lhs = Node.null_node, .rhs = Node.null_node },
        }),
        .op_and => self.parseReference(),
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

fn parseReference(self: *Parser) !Node.Idx {
    const main_idx = self.eat(.op_and) orelse return Node.null_node;
    _ = self.eat(.keyword_var);
    const expr = try self.parseExpr();
    return self.addNode(.{
        .tag = .reference,
        .main_idx = main_idx,
        .data = .{ .lhs = expr, .rhs = Node.null_node },
    });
}

fn parseModuleDecl(self: *Parser) !Node.Idx {
    log.debug("parseModuleDecl\n", .{});
    const main_idx = self.nextToken();
    const module_decl: Node.Tag = switch (self.tokens[main_idx].tag) {
        .keyword_module => .module_decl,
        else => return Node.null_node,
    };
    _ = self.eat(.lbrace);
    const members = try self.parseModuleMembers();
    _ = self.eat(.rbrace);

    return self.addNode(.{
        .tag = module_decl,
        .main_idx = main_idx,
        .data = .{
            .lhs = members.start,
            .rhs = members.end,
        },
    });
}

fn parseCombDecl(self: *Parser) !Node.Idx {
    _ = self.eat(.keyword_comb);
    const main_idx = self.eat(.identifier) orelse return Node.null_node;
    const comb_sig = try self.parseCombSig();
    const comb_body = try self.parseCombBody();

    return self.addNode(.{
        .tag = .comb_decl,
        .main_idx = main_idx,
        .data = .{
            .lhs = comb_sig,
            .rhs = comb_body,
        },
    });
}

fn parseCombSig(self: *Parser) !Node.Idx {
    log.debug("parseModuleSig\n", .{});
    const main_idx = self.eat(.lparen) orelse return Node.null_node;
    const comb_args = try self.parseCombArgs();
    _ = self.eat(.rparen);
    const ret_ty = try self.parseTypeExpr();
    return self.addNode(.{
        .tag = .comb_sig,
        .main_idx = main_idx,
        .data = .{
            .lhs = comb_args,
            .rhs = ret_ty,
        },
    });
}

fn parseCombArgs(self: *Parser) !Node.Idx {
    log.debug("parseModuleArgs\n", .{});
    const scratch_top = self.scratchpad.items.len;
    defer self.scratchpad.shrinkRetainingCapacity(scratch_top);

    while (true) {
        switch (self.tokens[self.tok_idx].tag) {
            .rparen => break,
            .identifier => {
                const main_idx = self.eat(.identifier).?;
                _ = self.eat(.colon);
                const type_expr = try self.parseTypeExpr();
                try self.scratchpad.append(self.allocator, try self.addNode(.{
                    .tag = .comb_arg,
                    .main_idx = main_idx,
                    .data = .{ .lhs = type_expr, .rhs = Node.null_node },
                }));
                _ = self.eat(.comma);
            },
            // TODO: error handling
            else => @panic("Unexpected token when parsing module_args"),
        }
    }

    const sublist = try self.scratchToSubList(scratch_top);
    return try self.addExtra(Node.CombArgs{
        .args_start = sublist.start,
        .args_end = sublist.end,
    });
}

fn parseCombBody(self: *Parser) !Node.Idx {
    log.debug("parseModuleStatements\n", .{});

    const main_idx = self.eat(.lbrace) orelse return Node.null_node;
    const scratch_top = self.scratchpad.items.len;
    defer self.scratchpad.shrinkRetainingCapacity(scratch_top);

    while (true) {
        switch (self.tokens[self.tok_idx].tag) {
            .keyword_pub => {
                self.tok_idx += 1;
                try self.scratchpad.append(self.allocator, try self.parseVarDeclStatement());
            },
            .keyword_const,
            .keyword_var,
            => try self.scratchpad.append(self.allocator, try self.parseVarDeclStatement()),
            .identifier => try self.scratchpad.append(self.allocator, try self.parseAssignmentStatement()),
            .keyword_return => try self.scratchpad.append(self.allocator, try self.parseReturnStatement()),
            .rbrace => break,
            else => {
                std.debug.print("[ERROR]: unhandled module statement: {s}", .{@tagName(self.tokens[self.tok_idx].tag)});
                break;
            },
        }
    }

    const sublist = try self.scratchToSubList(scratch_top);
    return try self.addNode(.{
        .tag = .comb_body,
        .main_idx = main_idx,
        .data = .{
            .lhs = sublist.start,
            .rhs = sublist.end,
        },
    });
}

fn parseModuleMembers(self: *Parser) !Node.SubList {
    log.debug("parseModuleMembers\n", .{});
    const scratch_top = self.scratchpad.items.len;
    defer self.scratchpad.shrinkRetainingCapacity(scratch_top);

    while (true) {
        switch (self.tokens[self.tok_idx].tag) {
            .keyword_pub => {
                self.tok_idx += 1;
                const member = switch (self.tokens[self.tok_idx].tag) {
                    .identifier => try self.parseVarDeclStatement(),
                    .keyword_comb => try self.parseCombDecl(),
                    else => {
                        std.debug.print("[ERROR]: unexpected public module member: {s}", .{@tagName(self.tokens[self.tok_idx].tag)});
                        break;
                    },
                };
                try self.scratchpad.append(self.allocator, member);
            },
            .keyword_const,
            .keyword_var,
            => try self.scratchpad.append(self.allocator, try self.parseVarDeclStatement()),
            .identifier => try self.scratchpad.append(self.allocator, try self.parseContainerField()),
            .keyword_comb => try self.scratchpad.append(self.allocator, try self.parseCombDecl()),
            .rbrace => break,
            else => {
                std.debug.print("[ERROR]: unhandled module member: {s}", .{@tagName(self.tokens[self.tok_idx].tag)});
                break;
            },
        }
    }

    return try self.scratchToSubList(scratch_top);
}

fn parseReturnStatement(self: *Parser) !Node.Idx {
    const main_idx = self.eat(.keyword_return) orelse return Node.null_node;

    const expr = try self.parseExpr();

    _ = self.eat(.semicolon);

    return try self.addNode(.{
        .tag = .@"return",
        .main_idx = main_idx,
        .data = .{
            .lhs = expr,
            .rhs = Node.null_node,
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

fn addExtra(self: *Parser, extra: anytype) !Node.Idx {
    const fields = std.meta.fields(@TypeOf(extra));
    try self.extra_data.ensureUnusedCapacity(self.allocator, fields.len);
    const result: Node.Idx = @enumFromInt(self.extra_data.items.len);
    inline for (fields) |field| {
        comptime assert(field.type == Node.Idx);
        self.extra_data.appendAssumeCapacity(@field(extra, field.name));
    }
    return result;
}

// Deinit `self` and free all related resources
pub fn deinit(self: *Parser) void {
    self.extra_data.deinit(self.allocator);
    self.scratchpad.deinit(self.allocator);
    self.* = undefined;
}

test Parser {
    _ = @import("parser_test.zig");
}
