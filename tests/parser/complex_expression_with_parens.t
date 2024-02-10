// RUN: @tara @file --dump-ast
// CHECK: struct_inner((var_decl(priv "A" = sub(mul(add(identifier("a"), identifier("b")), identifier("c")), div(identifier("d"), identifier("e"))))), )

const A = (a + b) * c - d / e;
