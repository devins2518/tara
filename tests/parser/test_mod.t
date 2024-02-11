// RUN: @tara @file --dump-ast
// CHECK: struct_inner((var_decl(priv "Mod" = module_decl(module_inner((d, reference_ty(const identifier("u1"))), (var_decl(priv "A" = identifier("bool"))), (var_decl(priv "B" = module_decl(module_inner((var_decl(priv "C" = identifier("u1"))), )))), )))), )

const Mod = module {
    const A = bool;
    const B = module {
        const C = u1;
    };

    d: &u1,
};
