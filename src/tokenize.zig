const std = @import("std");
const utils = @import("utils.zig");
const Location = utils.Location;

pub const Token = struct {
    tag: Tag,
    loc: Location,

    pub const keywords = std.ComptimeStringMap(Tag, .{
        .{ "and", .keyword_and },
        .{ "comb", .keyword_comb },
        .{ "const", .keyword_const },
        .{ "enum", .keyword_enum },
        .{ "fn", .keyword_fn },
        .{ "module", .keyword_module },
        .{ "or", .keyword_or },
        .{ "pub", .keyword_pub },
        .{ "struct", .keyword_struct },
        .{ "union", .keyword_union },
        .{ "var", .keyword_var },
        .{ "void", .keyword_void },
    });

    pub fn tryKeyword(ident: []const u8) ?Tag {
        return keywords.get(ident);
    }

    pub const Tag = enum {
        // Symbols
        lbrace, // {
        rbrace, // }
        lbracket, // [
        rbracket, // ]
        lparen, // (
        rparen, // )
        period, // .
        comma, // ,
        colon, // :
        semicolon, // ;
        equal, // =
        at, // @
        op_xor, // ^
        op_and, // &
        op_pipe, // |
        op_plus, // +
        op_minus, // -
        op_star, // *
        op_slash, // /
        op_lt, // <
        op_gt, // >
        op_lte, // <=
        op_gte, // >=
        op_eq, // ==
        op_neq, // !=
        // String based tokens
        identifier,
        number,
        keyword_and,
        keyword_comb,
        keyword_const,
        keyword_enum,
        keyword_fn,
        keyword_module,
        keyword_or,
        keyword_pub,
        keyword_struct,
        keyword_union,
        keyword_var,
        keyword_void,

        invalid,
        eof,
    };

    test keywords {
        comptime var num: usize = 0;
        comptime for (@typeInfo(Tag).Enum.fields) |field| {
            if (std.mem.startsWith(u8, field.name, "keyword_")) num += 1;
        };
        try std.testing.expectEqual(num, keywords.kvs.len);
    }
};

pub const Tokenizer = struct {
    buf: [:0]const u8,
    index: usize = 0,

    const State = enum {
        start,
        identifier,
        number,
        saw_equal,
        saw_slash,
        line_comment,
    };

    pub fn init(buf: [:0]const u8) Tokenizer {
        return .{ .buf = buf };
    }

    pub fn next(t: *Tokenizer) Token {
        var state: State = .start;
        var result = Token{
            .tag = .eof,
            .loc = .{
                .start = t.index,
                .end = undefined,
            },
        };
        while (true) : (t.index += 1) {
            const c = t.buf[t.index];
            switch (state) {
                .start => switch (c) {
                    0 => {
                        break;
                    },
                    'a'...'z', 'A'...'Z', '_' => {
                        result.tag = .identifier;
                        state = .identifier;
                    },
                    '0'...'9' => {
                        result.tag = .number;
                        state = .number;
                    },
                    '=' => {
                        state = .saw_equal;
                    },
                    ' ', '\t', '\n' => {
                        result.loc.start += 1;
                    },
                    '^' => {
                        result.tag = .op_xor;
                        t.index += 1;
                        break;
                    },
                    '&' => {
                        result.tag = .op_and;
                        t.index += 1;
                        break;
                    },
                    '|' => {
                        result.tag = .op_pipe;
                        t.index += 1;
                        break;
                    },
                    '+' => {
                        result.tag = .op_plus;
                        t.index += 1;
                        break;
                    },
                    '-' => {
                        result.tag = .op_minus;
                        t.index += 1;
                        break;
                    },
                    '*' => {
                        result.tag = .op_star;
                        t.index += 1;
                        break;
                    },
                    '@' => {
                        result.tag = .at;
                        t.index += 1;
                        break;
                    },
                    '.' => {
                        result.tag = .period;
                        t.index += 1;
                        break;
                    },
                    ',' => {
                        result.tag = .comma;
                        t.index += 1;
                        break;
                    },
                    ':' => {
                        result.tag = .colon;
                        t.index += 1;
                        break;
                    },
                    '{' => {
                        result.tag = .lbrace;
                        t.index += 1;
                        break;
                    },
                    '}' => {
                        result.tag = .rbrace;
                        t.index += 1;
                        break;
                    },
                    '[' => {
                        result.tag = .lbracket;
                        t.index += 1;
                        break;
                    },
                    ']' => {
                        result.tag = .rbracket;
                        t.index += 1;
                        break;
                    },
                    '(' => {
                        result.tag = .lparen;
                        t.index += 1;
                        break;
                    },
                    ')' => {
                        result.tag = .rparen;
                        t.index += 1;
                        break;
                    },
                    ';' => {
                        result.tag = .semicolon;
                        t.index += 1;
                        break;
                    },
                    '/' => {
                        state = .saw_slash;
                    },
                    else => {
                        result.tag = .invalid;
                        break;
                    },
                },
                .identifier => switch (c) {
                    'a'...'z', 'A'...'Z', '_', '0'...'9' => {},
                    else => {
                        if (Token.tryKeyword(t.buf[result.loc.start..t.index])) |kw| {
                            result.tag = kw;
                        }
                        break;
                    },
                },
                .number => switch (c) {
                    '0'...'9', '_' => {},
                    else => break,
                },
                .saw_equal => switch (c) {
                    else => {
                        result.tag = .equal;
                        break;
                    },
                },
                .saw_slash => switch (c) {
                    '/' => {
                        state = .line_comment;
                    },
                    else => {
                        result.tag = .op_slash;
                        break;
                    },
                },
                .line_comment => switch (c) {
                    '\n' => {
                        state = .start;
                        t.index += 1;
                        result.loc.start = t.index;
                    },
                    else => {},
                },
            }
        }
        result.loc.end = t.index;
        return result;
    }
};

test Tokenizer {
    _ = @import("tokenize_test.zig");
}
