// RUN: @tara @file --dump-ast
// CHECK: struct_inner((var_decl(priv "HalfAdder" = module_decl(module_inner((subroutine_decl(pub sum((a, reference_ty(const identifier("u1"))), (b, reference_ty(const identifier("u1"))), ) identifier("u1") ((return(bit_xor(identifier("a"), identifier("b")))), ))), (subroutine_decl(pub carry((a, reference_ty(const identifier("u1"))), (b, reference_ty(const identifier("u1"))), ) identifier("u1") ((return(bit_and(identifier("a"), identifier("b")))), ))), )))), )

const HalfAdder = module {
    pub comb sum(a: &u1, b: &u1) u1 {
        return a ^ b;
    }
    pub comb carry(a: &u1, b: &u1) u1 {
        return a & b;
    }
};
