// RUN: @tara @file --dump-utir
// CHECK: %0 = struct_decl({
// CHECK:     "Mod" %1 = module_decl({
// CHECK:         "A" @bool_type
// CHECK:         "a" %2 = subroutine_decl(
// CHECK:             {
// CHECK:                 %3: @bool_type
// CHECK:                 %4: @bool_type
// CHECK:             }
// CHECK:             @bool_type
// CHECK:             %5 = inline_block({
// CHECK:                 %6 = inline_block({
// CHECK:                     %7 = and(%3, %4)
// CHECK:                     %8 = inline_block_break(%6, %7)
// CHECK:                 })
// CHECK:                 %9 = return(%6)
// CHECK:                 %10 = inline_block_break(%5, %9)
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
