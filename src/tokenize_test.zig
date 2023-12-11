const std = @import("std");
const tokenize = @import("tokenize.zig");
const Token = tokenize.Token;
const Tokenizer = tokenize.Tokenizer;

fn runTestExpectSuccess(src: [:0]const u8, expected: []const Token.Tag) !void {
    var t = Tokenizer.init(src);
    for (expected) |token| {
        try std.testing.expectEqual(token, t.next().tag);
    }
}

fn tokenizeModuleWithFields() !void {
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
        .identifier, // sig
        .identifier, // a
        .semicolon, // ;
        .identifier, // sig
        .identifier, // b
        .semicolon, // ;
        .rbrace, // }
        .semicolon, // ;
        .eof,
    };
    try runTestExpectSuccess(src, &expected);
}

test tokenizeModuleWithFields {
    try tokenizeModuleWithFields();
}

fn tokenizeFnOldGrammar() !void {
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
        .identifier, // sig
        .comma, // ,
        .identifier, // b
        .colon, // :
        .lbracket, // [
        .number, // 3
        .rbracket, // ]
        .identifier, // sig
        .rparen, // )
        .keyword_struct, // struct
        .lbrace, // {
        .keyword_pub, // pub
        .identifier, // s
        .colon, // :
        .lbracket, // [
        .number, // 3
        .rbracket, // ]
        .identifier, // sig
        .comma, // ,
        .keyword_pub, // pub
        .identifier, // c
        .colon, // :
        .lbracket, // [
        .number, // 3
        .rbracket, // ]
        .identifier, // sig
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
    try runTestExpectSuccess(src, &expected);
}

test tokenizeFnOldGrammar {
    try tokenizeFnOldGrammar();
}

fn tokenizeComplexFnOldGrammar() !void {
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
    try runTestExpectSuccess(src, &expected);
}

test tokenizeComplexFnOldGrammar {
    try tokenizeComplexFnOldGrammar();
}

fn tokenizeMultipleStructsAndModule() !void {
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
        .identifier, // sig
        .comma, // ,
        .identifier, // b
        .colon, // :
        .identifier, // sig
        .comma, // ,
        .identifier, // cin
        .colon, // :
        .identifier, // sig
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
        .identifier, // sig
        .comma, // ,
        .identifier, // c
        .colon, // :
        .identifier, // sig
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
    try runTestExpectSuccess(src, &expected);
}

test tokenizeMultipleStructsAndModule {
    try tokenizeMultipleStructsAndModule();
}
