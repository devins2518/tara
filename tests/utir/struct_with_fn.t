// RUN: @tara @file --dump-utir
// CHECK: %0 = struct_decl({
// CHECK:     "S" %1 = struct_decl({
// CHECK:         "B" %2 = decl_val("bool")
// CHECK:         "C" %3 = decl_val("bool")
// CHECK:         "a" %4 = subroutine_decl(
// CHECK:             {
// CHECK:                 %5 = decl_val("B")
// CHECK:                 "b" : %5
// CHECK:                 %6 = decl_val("C")
// CHECK:                 "c" : %6
// CHECK:             }
// CHECK:             %7 = decl_val("bool")
// CHECK:             %8 = inline_block({
// CHECK:                 %9 = inline_block({
// CHECK:                     %10 = decl_val("b")
// CHECK:                     %11 = decl_val("c")
// CHECK:                     %12 = and(%10, %11)
// CHECK:                     %13 = inline_block_break(%9, %12)
// CHECK:                 })
// CHECK:                 %14 = return(%9)
// CHECK:                 %15 = inline_block_break(%8, %14)
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
