// RUN: @tara @file --dump-utir
// CHECK: %0 = struct_decl({
//     "A" %1 = inline_block({
//         %2 = int_literal(1)
//         %3 = int_literal(2)
//         %4 = add(%2, %3)
//         %5 = inline_block_break(%1, %4)
//     })
// })

const A = 1 + 2;
