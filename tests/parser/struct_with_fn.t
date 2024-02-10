// RUN: @tara @file --dump-ast
// CHECK: struct_inner((var_decl(priv "S" = struct_decl(struct_inner((var_decl(priv "A" = identifier("bool"))), (subroutine_decl(pub a((b, identifier("A")), (c, identifier("A")), ) identifier("bool") ((return(and(identifier("b"), identifier("c")))), ))), )))), )

const S = struct {
    const A = bool;

    pub fn a(b: A, c: A) bool {
        return b and c;
    }
};
