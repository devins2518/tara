const std = @import("std");
const Allocator = std.mem.Allocator;
const MultiArrayList = std.MultiArrayList;
const Parser = @import("Parser.zig");
const tokenizer = @import("tokenize.zig");
const Token = tokenizer.Token;
const Tokenizer = tokenizer.Tokenizer;
const Ast = @This();

source: [:0]const u8,
nodes: Node.List,
extra_data: []Node.Idx,

pub const TokenIdx = u32;

pub const Node = struct {
    tag: Tag,
    main_idx: TokenIdx,
    data: Data,

    pub const Idx = u32;
    // 0 is reserved to refer to the root node, so 0 may be used as `null` index;
    pub const null_node = 0;

    pub const Tag = enum {
        // `main_idx` is always 0
        // data is list of decls as nodes[lhs..rhs]
        root,
        // `pub? (const|var) ident(: type_expr) = expr;`
        // `main_idx` is (const|var)
        // lhs is optional `type_expr`, indexes extra_data
        // rhs is `expr`, indexes nodes
        var_decl,
        // `struct { fields* }`
        // `main_idx` is {
        // If there are no fields or decls, then data is 0
        // If there is a single field or decl, then lhs is 0 and data.rhs is a single field or decl
        // If there are 2 or more fields or decls, then data.lhs is extra_index index of a `StructBody` and data.rhs is 0
        struct_decl,
        // `ident: type_expr( = expr)?`
        // `main_idx` is ident
        // lhs is `type_expr`, indexes extra_data
        // rhs is optional `expr`, indexes extra_data
        struct_field,
    };

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

    pub const List = MultiArrayList(Node);
};

pub fn parse(allocator: Allocator, source: [:0]const u8) !Ast {
    const tokens = try std.ArrayList(Token).initCapacity(allocator, source.len);
    var t = Tokenizer.init(source);

    while (true) {
        const token = t.next();
        tokens.appendAssumeCapacity(token);
        if (token == .eof) break;
    }

    var parser = Parser{
        .allocator = allocator,
        .source = source,
        .tokens = try tokens.toOwnedSlice(),
    };
    defer parser.deinit();
    try parser.parseRoot();

    return Ast{
        .source = source,
        .nodes = parser.nodes,
        .extra_data = try parser.extra_data.toOwnedSlice(),
    };
}
