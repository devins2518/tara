// RUN: @tara @file --dump-utir
// CHECK: %0 = struct_decl({
// CHECK:     "S" %1 = struct_decl({
// CHECK:         "B" @bool_type
// CHECK:         "C" @bool_type
// CHECK:         "a" %2 = subroutine_decl(
// CHECK:             {
// CHECK:                 %3: @bool_type
// CHECK:                 %4: @bool_type
// CHECK:             }
// CHECK:             @bool_type
// CHECK:             %5 = block({
// CHECK:                 %6 = inline_block({
// CHECK:                     %7 = and(%3, %4)
// CHECK:                     %8 = return(%7)
// CHECK:                     %9 = inline_block_break(%6, %8)
// CHECK:                 })
// CHECK:                 %10 = ret_implicit_void()
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
