// RUN: @tara @file --dump-utir
// CHECK: %0 = struct_decl({
// CHECK:     "A" %1 = inline_block({
// CHECK:         %2 = int_literal(1)
// CHECK:         %3 = int_literal(2)
// CHECK:         %4 = add(%2, %3)
// CHECK:         %5 = int_literal(3)
// CHECK:         %6 = mul(%4, %5)
// CHECK:         %7 = int_literal(4)
// CHECK:         %8 = int_literal(5)
// CHECK:         %9 = div(%7, %8)
// CHECK:         %10 = sub(%6, %9)
// CHECK:         %11 = inline_block_break(%1, %10)
// CHECK:     })
// CHECK: })

const A = (1 + 2) * 3 - 4 / 5;
