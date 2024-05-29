// RUN: @tara @file --dump-ast
// CHECK: struct_decl(struct_inner((var_decl(priv "A" = identifier("u1"))), (var_decl(priv "B" = bit_and(identifier("a"), identifier("b")))), (var_decl(priv "C" = neg(identifier("a")))), (var_decl(priv "D" = call(access(identifier("a"), identifier("b")), ()))), (var_decl(priv "E" = call(access(identifier("a"), identifier("b")), (identifier("c")identifier("d"))))), (var_decl(priv "F" = bit_or(identifier("a"), identifier("b")))), (var_decl(priv "G" = reference_ty(const identifier("u1")))), (var_decl(priv "H" = reference_ty(mut identifier("u1")))), (var_decl(priv "I" = identifier("undefined"))), (var_decl(priv "J" = identifier("true"))), (var_decl(priv "K" = identifier("false"))), (var_decl(priv "L" = deref(identifier("a")))), (var_decl(priv "M" = return(identifier("a")))), (var_decl(priv "N" = number(0))), (var_decl(priv "O" = number(0))), (var_decl(priv "P" = number(15))), (var_decl(priv "Q" = sized_number(3'd7))), (var_decl(priv "R" = sized_number(32'd4294967295))), (var_decl(priv "S" = if_expr(identifier("a"), identifier("e"), identifier("b")))), (var_decl(priv "R" = builtin_call(builtin, (identifier("S"))))), (var_decl(priv "T" = builtin_call(builtin2, (identifier("S")identifier("R"))))), (var_decl(priv "U" = if_expr(identifier("A"), identifier("M"), ()))), ))

const A = u1;
const B = a & b;
const C = -a;
const D = a.b();
const E = a.b(c, d);
const F = (a | b);
const G = &u1;
const H = &var u1;
const I = undefined;
const J = true;
const K = false;
const L = a.*;
const M = return a;
const N = 0;
const O = 0b0;
const P = 0xF;
const Q = 3b111;
const R = 32xFFFFFFFF;
const S = if (a) e else b;
const R = @builtin(S);
const T = @builtin2(S, R);
const U = if (A) M else {};
