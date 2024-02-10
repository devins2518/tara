// RUN: @tara @file --dump-ast
// CHECK: struct_inner((var_decl(priv "A" = add(identifier("a"), identifier("b")))), )

const A = a + b;
