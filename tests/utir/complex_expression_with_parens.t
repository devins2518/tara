// RUN: @tara @file --dump-utir
// CHECK: %0 = struct_decl({
// CHECK:     "A" %1 = inline_block({
// CHECK:         %2 = inline_block({
// CHECK:             %3 = inline_block({
// CHECK:                 %4 = decl_val("a")
// CHECK:                 %5 = decl_val("b")
// CHECK:                 %6 = add(%4, %5)
// CHECK:                 %7 = inline_block_break(%3, %6)
// CHECK:             })
// CHECK:             %8 = decl_val("c")
// CHECK:             %9 = mul(%3, %8)
// CHECK:             %10 = inline_block_break(%2, %9)
// CHECK:         })
// CHECK:         %11 = inline_block({
// CHECK:             %12 = decl_val("d")
// CHECK:             %13 = decl_val("e")
// CHECK:             %14 = div(%12, %13)
// CHECK:             %15 = inline_block_break(%11, %14)
// CHECK:         })
// CHECK:         %16 = sub(%2, %11)
// CHECK:         %17 = inline_block_break(%1, %16)
// CHECK:     })
// CHECK: })

const A = (a + b) * c - d / e;
