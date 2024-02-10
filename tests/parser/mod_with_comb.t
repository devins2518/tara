// RUN: @tara @file --dump-ast
// CHECK: struct_inner((var_decl(priv "Mod" = module_decl(module_inner((var_decl(priv "A" = identifier("bool"))), (subroutine_decl(pub a((b, identifier("A")), (c, identifier("A")), ) identifier("bool") ((return(and(identifier("b"), identifier("c")))), ))), )))), )

const Mod = module {
    const A = bool;

    pub comb a(b: A, c: A) bool {
        return b and c;
    }
};
