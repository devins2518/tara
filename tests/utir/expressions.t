// RUN: @tara @file --dump-utir
// CHECK: %0 = struct_decl({
// CHECK:     "A" %1 = int_type(u, 1)
// CHECK:     "B" %2 = inline_block({
// CHECK:         %3 = bit_and(%1, %1)
// CHECK:         %4 = inline_block_break(%2, %3)
// CHECK:     })
// CHECK:     "C" %5 = inline_block({
// CHECK:         %6 = negate(%1)
// CHECK:         %7 = inline_block_break(%5, %6)
// CHECK:     })
// CHECK:     "D" %8 = inline_block({
// CHECK:         %9 = inline_block({
// CHECK:             %10 = access(%2, "b")
// CHECK:             %11 = inline_block_break(%9, %10)
// CHECK:         })
// CHECK:         %12 = call(%9, {})
// CHECK:         %13 = inline_block_break(%8, %12)
// CHECK:     })
// CHECK:     "E" %14 = inline_block({
// CHECK:         %15 = inline_block({
// CHECK:             %16 = access(%2, "b")
// CHECK:             %17 = inline_block_break(%15, %16)
// CHECK:         })
// CHECK:         %19 = int_type(u, 1)
// CHECK:         %18 = call(%15, {
// CHECK:             %19,
// CHECK:             %8,
// CHECK:         })
// CHECK:         %20 = inline_block_break(%14, %18)
// CHECK:     })
// CHECK:     "F" %21 = inline_block({
// CHECK:         %22 = bit_or(%5, %8)
// CHECK:         %23 = inline_block_break(%21, %22)
// CHECK:     })
// CHECK:     "G" %24 = inline_block({
// CHECK:         %25 = int_type(u, 1)
// CHECK:         %26 = ref_ty(const %25)
// CHECK:         %27 = inline_block_break(%24, %26)
// CHECK:     })
// CHECK:     "H" %28 = inline_block({
// CHECK:         %29 = int_type(u, 1)
// CHECK:         %30 = ref_ty(mut %29)
// CHECK:         %31 = inline_block_break(%28, %30)
// CHECK:     })
// CHECK:     "I" @undefined
// CHECK:     "J" @bool_true
// CHECK:     "K" @bool_false
// CHECK:     "L" %32 = inline_block({
// CHECK:         %33 = deref(%1)
// CHECK:         %34 = inline_block_break(%32, %33)
// CHECK:     })
// CHECK:     "M" %35 = inline_block({
// CHECK:         %36 = return(%1)
// CHECK:         %37 = inline_block_break(%35, %36)
// CHECK:     })
// CHECK:     "N" %38 = int_literal(0)
// CHECK:     "O" %39 = int_literal(0)
// CHECK:     "P" %40 = int_literal(15)
// CHECK:     "Q" %41 = inline_block({
// CHECK:         %42 = int_literal(7)
// CHECK:         %43 = int_type(u, 3)
// CHECK:         %44 = as(%43, %42)
// CHECK:         %45 = inline_block_break(%41, %44)
// CHECK:     })
// CHECK:     "R" %46 = inline_block({
// CHECK:         %47 = int_literal(4294967295)
// CHECK:         %48 = int_type(u, 32)
// CHECK:         %49 = as(%48, %47)
// CHECK:         %50 = inline_block_break(%46, %49)
// CHECK:     })
// CHECK:     "S" %51 = inline_block({
// CHECK:         %52 = branch(%35, {
// CHECK:             %53 = inline_block_break(%52, %1)
// CHECK:         }, {
// CHECK:             %54 = inline_block({
// CHECK:                 %55 = bit_xor(%1, %2)
// CHECK:                 %56 = inline_block_break(%54, %55)
// CHECK:             })
// CHECK:             %57 = inline_block_break(%52, %54)
// CHECK:         })
// CHECK:         %58 = inline_block_break(%51, %52)
// CHECK:     })
// CHECK: })

const A = u1;
const B = A & A;
const C = -A;
const D = B.b();
const E = B.b(u1, D);
const F = (C | D);
const G = &u1;
const H = &var u1;
const I = undefined;
const J = true;
const K = false;
const L = A.*;
const M = return A;
const N = 0;
const O = 0b0;
const P = 0xF;
const Q = 3b111;
const R = 32xFFFFFFFF;
const S = if (M) A else A ^ B;
