// RUN: @tara @file --dump-ast --exit-early
// CHECK: struct_decl(struct_inner((var_decl(priv "S" = struct_decl(struct_inner((subroutine_decl(pub a() identifier("u1") ((local(priv "b" = identifier("u3"))), (return(number(1))), ))), )))), (var_decl(priv "M" = module_decl(module_inner((subroutine_decl(pub a() identifier("u1") ((local(priv "b" = identifier("u3"))), (return(number(1))), ))), )))), ))

const S = struct {
    pub fn a() u1 {
        const b = u3;
        return 1;
    }
};

const M = module {
    pub comb a() u1 {
        const b = u3;
        return 1;
    }
};
