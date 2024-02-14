// RUN: @tara @file --dump-utir
// CHECK: %0 = struct_decl({
//     "A" %1 = inline_block({
//         %2 = decl_val("a")
//         %3 = decl_val("b")
//         %4 = add(%2, %3)
//         %5 = inline_block_break(%1, %4)
//     })
// })

const A = a + b;
