// RUN: @tara @file --dump-utir
// CHECK: %0 = struct_decl({
// CHECK:     "S" %1 = struct_decl({
// CHECK:         "B" @bool_type
// CHECK:         "C" @bool_type
// CHECK:         "a" %2 = subroutine_decl(
// CHECK:             {
// CHECK:                 %3 = decl_val("B")
// CHECK:                 "b" : %3
// CHECK:                 %4 = decl_val("C")
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

const S = struct {
    const B = bool;
    const C = bool;

    pub fn a(b: B, c: C) bool {
        return b and c;
    }
};
