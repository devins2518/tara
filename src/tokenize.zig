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
        .{ "sig", .keyword_sig },
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
        keyword_sig,
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
    const tests = struct {
        fn doTheTest(src: [:0]const u8, expected: []const Token.Tag) !void {
            var t = Tokenizer.init(src);
            for (expected) |token| {
                try std.testing.expectEqual(token, t.next().tag);
            }
        }

        pub fn test0() !void {
            const src =
                \\const Alu = module {
                \\  sig a;
                \\  sig b;
                \\};
            ;
            const expected = [_]Token.Tag{
                .keyword_const, // const
                .identifier, // Alu
                .equal, // =
                .keyword_module, // module
                .lbrace, // {
                .keyword_sig, // sig
                .identifier, // a
                .semicolon, // ;
                .keyword_sig, // sig
                .identifier, // b
                .semicolon, // ;
                .rbrace, // }
                .semicolon, // ;
                .eof,
            };
            try doTheTest(src, &expected);
        }

        pub fn test1() !void {
            const src =
                \\const adder = fn (a: [3]sig, b: [3]sig) struct {
                \\    pub s: [3]sig,
                \\    pub c: [3]sig,
                \\
                \\    comb {
                \\        s = @foreach(a ^ b ^ cin);
                \\        c = @foreach(a & b | cin & a ^ b);
                \\    }
                \\};
            ;
            const expected = [_]Token.Tag{
                .keyword_const, // const
                .identifier, // adder
                .equal, // =
                .keyword_fn, // fn
                .lparen, // (
                .identifier, // a
                .colon, // :
                .lbracket, // [
                .number, // 3
                .rbracket, // ]
                .keyword_sig, // sig
                .comma, // ,
                .identifier, // b
                .colon, // :
                .lbracket, // [
                .number, // 3
                .rbracket, // ]
                .keyword_sig, // sig
                .rparen, // )
                .keyword_struct, // struct
                .lbrace, // {
                .keyword_pub, // pub
                .identifier, // s
                .colon, // :
                .lbracket, // [
                .number, // 3
                .rbracket, // ]
                .keyword_sig, // sig
                .comma, // ,
                .keyword_pub, // pub
                .identifier, // c
                .colon, // :
                .lbracket, // [
                .number, // 3
                .rbracket, // ]
                .keyword_sig, // sig
                .comma, // ,
                .keyword_comb, // comb
                .lbrace, // {
                .identifier, // s
                .equal, // =
                .at, // @
                .identifier, // foreach
                .lparen, // (
                .identifier, // a
                .op_xor, // ^
                .identifier, // b
                .op_xor, // ^
                .identifier, // cin
                .rparen, // )
                .semicolon, // ;
                .identifier, // c
                .equal, // =
                .at, // @
                .identifier, // foreach
                .lparen, // (
                .identifier, // a
                .op_and, // &
                .identifier, // b
                .op_pipe, // |
                .identifier, // cin
                .op_and, // &
                .identifier, // a
                .op_xor, // ^
                .identifier, // b
                .rparen, // )
                .semicolon, // ;
                .rbrace, // }
                .rbrace, // }
                .semicolon, // ;
                .eof,
            };
            try doTheTest(src, &expected);
        }

        pub fn test2() !void {
            const src =
                \\const addr = fn (in: &In, out: &var Out) void {
                \\    out.s = @foreach(a ^ b ^ cin);
                \\    out.c = @foreach (a & b | cin & a ^ b);
                \\};
            ;
            const expected = [_]Token.Tag{
                .keyword_const, // const
                .identifier, // adder
                .equal, // =
                .keyword_fn, // fn
                .lparen, // (
                .identifier, // in
                .colon, // :
                .op_and, // &
                .identifier, // In
                .comma, // ,
                .identifier, // out
                .colon, // :
                .op_and, // &
                .keyword_var, // var
                .identifier, // Out
                .rparen, // )
                .keyword_void, // void
                .lbrace, // {
                .identifier, // out
                .period, // .
                .identifier, // s
                .equal, // =
                .at, // @
                .identifier, // foreach
                .lparen, // (
                .identifier, // a
                .op_xor, // ^
                .identifier, // b
                .op_xor, // ^
                .identifier, // cin
                .rparen, // )
                .semicolon, // ;
                .identifier, // out
                .period, // .
                .identifier, // c
                .equal, // =
                .at, // @
                .identifier, // foreach
                .lparen, // (
                .identifier, // a
                .op_and, // &
                .identifier, // b
                .op_pipe, // |
                .identifier, // cin
                .op_and, // &
                .identifier, // a
                .op_xor, // ^
                .identifier, // b
                .rparen, // )
                .semicolon, // ;
                .rbrace, // }
                .semicolon, // ;
                .eof,
            };
            try doTheTest(src, &expected);
        }

        pub fn test3() !void {
            const src =
                \\const In = struct {
                \\    a: sig,
                \\    b: sig,
                \\    cin: sig,
                \\};
                \\const Out = struct {
                \\    s: sig,
                \\    c: sig,
                \\};
                \\const adder = module(in: &In, out: &var Out) {
                \\    // ^ defined for buses
                \\    out.s = in.a ^ in.b ^ in.cin;
                \\    out.c = (in.a and in.b) or (in.cin and (in.a ^ in.b));
                \\};
            ;
            const expected = [_]Token.Tag{
                .keyword_const, // const
                .identifier, // In
                .equal, // =
                .keyword_struct, // struct
                .lbrace, // {
                .identifier, // a
                .colon, // :
                .keyword_sig, // sig
                .comma, // ,
                .identifier, // b
                .colon, // :
                .keyword_sig, // sig
                .comma, // ,
                .identifier, // cin
                .colon, // :
                .keyword_sig, // sig
                .comma, // ,
                .rbrace, // }
                .semicolon, // ;
                .keyword_const, // const
                .identifier, // Out
                .equal, // =
                .keyword_struct, // struct
                .lbrace, // {
                .identifier, // s
                .colon, // :
                .keyword_sig, // sig
                .comma, // ,
                .identifier, // c
                .colon, // :
                .keyword_sig, // sig
                .comma, // ,
                .rbrace, // }
                .semicolon, // ;
                .keyword_const, // const
                .identifier, // adder
                .equal, // =
                .keyword_module, // module
                .lparen, // (
                .identifier, // in
                .colon, // :
                .op_and, // &
                .identifier, // In
                .comma, // ,
                .identifier, // out
                .colon, // :
                .op_and, // &
                .keyword_var, // var
                .identifier, // Out
                .rparen, // )
                .lbrace, // {
                .identifier, // out
                .period, // .
                .identifier, // s
                .equal, // =
                .identifier, // in
                .period, // .
                .identifier, // a
                .op_xor, // ^
                .identifier, // in
                .period, // .
                .identifier, // b
                .op_xor, // ^
                .identifier, // in
                .period, // .
                .identifier, // cin
                .semicolon, // ;
                .identifier, // out
                .period, // .
                .identifier, // c
                .equal, // =
                .lparen, // (
                .identifier, // in
                .period, // .
                .identifier, // a
                .keyword_and, // and
                .identifier, // in
                .period, // .
                .identifier, // b
                .rparen, // )
                .keyword_or, // or
                .lparen, // (
                .identifier, // in
                .period, // .
                .identifier, // cin
                .keyword_and, // and
                .lparen, // (
                .identifier, // in
                .period, // .
                .identifier, // a
                .op_xor, // ^
                .identifier, // in
                .period, // .
                .identifier, // b
                .rparen, // )
                .rparen, // )
                .semicolon, // ;
                .rbrace, // }
                .semicolon, // ;
                .eof,
            };
            try doTheTest(src, &expected);
        }
    };

    try tests.test0();
    try tests.test1();
    try tests.test2();
    try tests.test3();
}
