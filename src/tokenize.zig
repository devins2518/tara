const std = @import("std");
const utils = @import("utils.zig");
const Location = utils.Location;

pub const Token = struct {
    tag: Tag,
    loc: Location,

    pub const keywords = std.ComptimeStringMap(Tag, .{
        .{ "const", .keyword_const },
        .{ "enum", .keyword_enum },
        .{ "fn", .keyword_fn },
        .{ "module", .keyword_module },
        .{ "sig", .keyword_sig },
        .{ "struct", .keyword_struct },
        .{ "union", .keyword_union },
        .{ "var", .keyword_var },
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
        semicolon, // ;
        equal, // =
        // String based tokens
        identifier,
        keyword_const,
        keyword_enum,
        keyword_fn,
        keyword_module,
        keyword_sig,
        keyword_struct,
        keyword_union,
        keyword_var,

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
        saw_equal,
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
                    '=' => {
                        state = .saw_equal;
                    },
                    ' ', '\t', '\n' => {
                        result.loc.start += 1;
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
                .saw_equal => switch (c) {
                    else => {
                        result.tag = .equal;
                        break;
                    },
                },
            }
        }
        result.loc.end = t.index;
        return result;
    }
};

test Tokenizer {
    const src =
        \\const Alu = module {
        \\  sig a;
        \\  sig b;
        \\};
    ;
    var t = Tokenizer.init(src);

    const expected = &[_]Token.Tag{
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
    for (expected) |token| {
        try std.testing.expectEqual(token, t.next().tag);
    }
}
