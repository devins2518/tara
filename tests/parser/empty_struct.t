// RUN: @tara @file --dump-ast
// CHECK: struct_inner((var_decl(priv "S" = struct_decl(struct_inner()))), )

const S = struct {};
