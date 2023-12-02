const std = @import("std");
const Allocator = std.mem.Allocator;
const MultiArrayList = std.MultiArrayList;
const Parser = @import("Parser.zig");
const tokenizer = @import("tokenize.zig");
const Token = tokenizer.Token;
const Tokenizer = tokenizer.Tokenizer;
const Ast = @This();

source: [:0]const u8,
nodes: Node.List.Slice,
extra_data: []Node.Idx,

pub const TokenIdx = u32;

pub const Node = struct {
    tag: Tag,
    main_idx: TokenIdx,
    data: Data,

    pub const Idx = enum(u32) { _ };
    // 0 is reserved to refer to the root node, so 0 may be used as `null` index;
    pub const null_node: Idx = @enumFromInt(0);

    pub const Tag = enum {
        // `main_idx` is always 0
        // data is list of decls as extra_data[lhs..rhs]
        root,
        // `pub? (const|var) ident(: type_expr) = expr;`
        // `main_idx` is (const|var)
        // lhs is optional `type_expr`, indexes extra_data
        // rhs is `expr`, indexes nodes
        var_decl,
        // `struct { fields* }`
        // `main_idx` is {
        // TODO: This needs to be revisted wrt root
        // If there are no fields or decls, then data is 0
        // If there is a single field or decl, then lhs is 0 and data.rhs is a single field or decl indexing nodes
        // If there are 2 or more fields or decls, then data.lhs is extra_index index of a `StructBody` and data.rhs is 0
        struct_decl,
        // `module (args) { statements* }`
        // `main_idx` is module
        // lhs is extra_index index of `ModuleArgs` which indexes into `extra_data`
        // rhs is extra_index index of `ModuleStatements` which indexes into `extra_data`
        module_decl,
        // `ident: type_expr`
        // `main_idx` is ident
        // lhs is `nodes` index of `type_expr`
        // rhs is unused
        module_arg,
        // `ident: type_expr( = expr)?`
        // `main_idx` is ident
        // lhs is `type_expr`, indexes `nodes`
        // rhs is optional `expr`, indexes `nodes`
        container_field,
        // `lhs or rhs`
        // `main_idx` is `or`
        @"or",
        // `lhs and rhs`
        // `main_idx` is `and`
        @"and",
        // `lhs < rhs`
        // `main_idx` is <
        lt,
        // `lhs > rhs`
        // `main_idx` is >
        gt,
        // `lhs <= rhs`
        // `main_idx` is <=
        lte,
        // `lhs >= rhs`
        // `main_idx` is >=
        gte,
        // `lhs == rhs`
        // `main_idx` is ==
        eq,
        // `lhs != rhs`
        // `main_idx` is !=
        neq,
        // `lhs & rhs`
        // `main_idx` is &
        bit_and,
        // `lhs | rhs`
        // `main_idx` is |
        bit_or,
        // `lhs ^ rhs`
        // `main_idx` is ^
        bit_xor,
        // `lhs + rhs`
        // `main_idx` is `+`
        add,
        // `lhs - rhs`
        // `main_idx` is `-`
        sub,
        // `lhs * rhs`
        // `main_idx` is `*`
        mul,
        // `lhs / rhs`
        // `main_idx` is `/`
        div,
        // `&var? lhs`
        // `main_idx` is &
        // `lhs` is an expression
        // `rhs` is unused
        reference,
        // `lhs assignment_op rhs`
        // `main_idx` is assignment_op
        // `lhs` is an lvalue expression
        // `rhs` is an expression
        assignment,
        // `lhs.rhs`
        // `main_idx` is `.`
        // `lhs` is an identifier
        // `rhs` is unused
        member,
        // `ident`
        // `main_idx` is `ident`
        // `lhs` and `rhs` unused
        identifier,
    };

    // TODO change this to be bare union between Idx and ExtraIdx for debug type safety
    pub const Data = struct {
        lhs: Idx,
        rhs: Idx,
    };

    pub const SubList = struct {
        start: Idx,
        end: Idx,
    };

    pub const StructBody = struct {
        fields_start: Idx,
        fields_end: Idx,
        decls_start: Idx,
        decls_end: Idx,
    };

    // Indexes into extra data
    pub const ModuleArgs = struct {
        args_start: Idx,
        args_end: Idx,
    };

    // Indexes into extra data
    pub const ModuleStatements = struct {
        statements_start: Idx,
        statements_end: Idx,
    };

    pub const List = MultiArrayList(Node);
};

pub fn parse(allocator: Allocator, source: [:0]const u8) !Ast {
    var tokens = try std.ArrayList(Token).initCapacity(allocator, source.len);
    var t = Tokenizer.init(source);

    while (true) {
        const token = t.next();
        try tokens.append(token);
        if (token.tag == .invalid) @panic("Uh oh invalid tag");
        if (token.tag == .eof) break;
    }

    var parser = Parser{
        .allocator = allocator,
        .source = source,
        .tokens = try tokens.toOwnedSlice(),
    };
    defer parser.deinit();
    try parser.nodes.setCapacity(allocator, parser.tokens.len);
    try parser.parseRoot();

    return Ast{
        .source = source,
        .nodes = parser.nodes.toOwnedSlice(),
        .extra_data = try parser.extra_data.toOwnedSlice(allocator),
    };
}

pub fn deinit(self: *Ast, allocator: Allocator) void {
    self.nodes.deinit(allocator);
    allocator.free(self.extra_data);
}

// Useful types for constructing assembled information about a Node
pub const Assembled = struct {
    pub const Struct = struct {
        token: TokenIdx,
        fields: []const Node.Idx,
    };

    pub const VarDecl = struct {
        token: TokenIdx,
        type_expr: Node.Idx,
        expr: Node.Idx,
    };
};

pub fn assembledStruct(self: *const Ast, node_idx: Node.Idx) ?Assembled.Struct {
    const idx = @intFromEnum(node_idx);
    const lhs = @intFromEnum(self.nodes.items(.data)[idx].lhs);
    const rhs = @intFromEnum(self.nodes.items(.data)[idx].rhs);
    return switch (self.nodes.items(.tag)[idx]) {
        .root, .struct_decl => .{
            .token = self.nodes.items(.main_idx)[idx],
            .fields = self.extra_data[lhs..rhs],
        },
        else => null,
    };
}

pub fn assembledVarDecl(self: *const Ast, node_idx: Node.Idx) ?Assembled.VarDecl {
    const idx = @intFromEnum(node_idx);
    const token = self.nodes.items(.main_idx)[idx];
    const type_expr_idx = self.nodes.items(.data)[idx].lhs;
    const type_expr = if (type_expr_idx != Node.null_node)
        self.extra_data[@intFromEnum(type_expr_idx)]
    else
        Node.null_node;
    const expr = self.nodes.items(.data)[idx].rhs;
    return switch (self.nodes.items(.tag)[idx]) {
        .var_decl => .{
            .token = token,
            .type_expr = type_expr,
            .expr = expr,
        },
        else => null,
    };
}
