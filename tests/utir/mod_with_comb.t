// RUN: @tara @file --dump-utir
// CHECK: %0 = struct_decl({
// CHECK:     "Mod" %1 = module_decl({
// CHECK:         "A" %2 = decl_val("bool")
// CHECK:         "a" %3 = subroutine_decl(
// CHECK:             {
// CHECK:                 %4 = decl_val("A")
// CHECK:                 "b" : %4
// CHECK:                 %5 = decl_val("A")
// CHECK:                 "c" : %5
// CHECK:             }
// CHECK:             %6 = decl_val("bool")
// CHECK:             %7 = inline_block({
// CHECK:                 %8 = inline_block({
// CHECK:                     %9 = decl_val("b")
// CHECK:                     %10 = decl_val("c")
// CHECK:                     %11 = and(%9, %10)
// CHECK:                     %12 = inline_block_break(%8, %11)
// CHECK:                 })
// CHECK:                 %13 = return(%8)
// CHECK:                 %14 = inline_block_break(%7, %13)
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