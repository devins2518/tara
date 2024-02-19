// RUN: @tara @file --dump-utir
// CHECK: %0 = struct_decl({
// CHECK:     "A" %1 = inline_block({
// CHECK:         %2 = inline_block({
// CHECK:             %3 = int_literal(1)
// CHECK:             %4 = inline_block({
// CHECK:                 %5 = int_literal(2)
// CHECK:                 %6 = int_literal(3)
// CHECK:                 %7 = mul(%5, %6)
// CHECK:                 %8 = inline_block_break(%4, %7)
// CHECK:             })
// CHECK:             %9 = add(%3, %4)
// CHECK:             %10 = inline_block_break(%2, %9)
// CHECK:         })
// CHECK:         %11 = inline_block({
// CHECK:             %12 = int_literal(4)
// CHECK:             %13 = int_literal(5)
// CHECK:             %14 = div(%12, %13)
// CHECK:             %15 = inline_block_break(%11, %14)
// CHECK:         })
// CHECK:         %16 = sub(%2, %11)
// CHECK:         %17 = inline_block_break(%1, %16)
// CHECK:     })
// CHECK: })

const A = 1 + 2 * 3 - 4 / 5;
