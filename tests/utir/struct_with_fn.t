// RUN: @tara @file --dump-utir
// CHECK: %0 = struct_decl({
// CHECK:     "S" %1 = struct_decl({
// CHECK:         "B" @bool_type
// CHECK:         "C" @bool_type
// CHECK:         "a" %2 = subroutine_decl(
// CHECK:             {
// CHECK:                 %3 = decl_val("B")
// CHECK:                 %4: %3
// CHECK:                 %5 = decl_val("C")
// CHECK:                 %6: %5
// CHECK:             }
// CHECK:             @bool_type
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

const S = struct {
    const B = bool;
    const C = bool;

    pub fn a(b: B, c: C) bool {
        return b and c;
    }
};
