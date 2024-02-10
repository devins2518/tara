// RUN: @tara @file --dump-ast
// CHECK: struct_inner((var_decl(priv "S" = struct_decl(struct_inner((d, identifier("u1")), (var_decl(priv "A" = identifier("bool"))), (var_decl(priv "B" = struct_decl(struct_inner((var_decl(priv "C" = identifier("u1"))), )))), )))), )

const S = struct {
    const A = bool;
    const B = struct {
        const C = u1;
    };

    d: u1,
};
