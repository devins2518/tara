// RUN: @tara @file --dump-ast
// CHECK: struct_inner((var_decl(priv "A" = identifier("bool"))), (var_decl(priv "B" = identifier("u1"))), (var_decl(priv "C" = identifier("i1"))), (var_decl(priv "D" = identifier("sig"))), (var_decl(priv "E" = module_decl(module_inner()))), (var_decl(priv "F" = struct_decl(struct_inner()))), (var_decl(priv "G" = identifier("type"))), (var_decl(priv "H" = identifier("clock"))), (var_decl(priv "I" = identifier("reset"))), )

const A = bool;
const B = u1;
const C = i1;
const D = sig;
const E = module {};
const F = struct {};
const G = type;
const H = clock;
const I = reset;
