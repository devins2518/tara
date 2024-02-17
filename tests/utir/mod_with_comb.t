// RUN: @tara @file --dump-utir
// CHECK: %0 = struct_decl({
// CHECK:     "Mod" %1 = module_decl({
// CHECK:         "A" @bool_type
// CHECK:         "a" %2 = subroutine_decl(
// CHECK:             {
// CHECK:                 %3 = decl_val("A")
// CHECK:                 "b" : %3
// CHECK:                 %4 = decl_val("A")
// CHECK:                 "c" : %4
// CHECK:             }
// CHECK:             @bool_type
// CHECK:             %5 = inline_block({
// CHECK:                 %6 = inline_block({
// CHECK:                     %7 = decl_val("b")
// CHECK:                     %8 = decl_val("c")
// CHECK:                     %9 = and(%7, %8)
// CHECK:                     %10 = inline_block_break(%6, %9)
// CHECK:                 })
// CHECK:                 %11 = return(%6)
// CHECK:                 %12 = inline_block_break(%5, %11)
// CHECK:             })
// CHECK:         )
// CHECK:     })
// CHECK: })

const Mod = module {
    const A = bool;

    pub comb a(b: A, c: A) bool {
        return b and c;
    }
};
