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
// CHECK:         %9 = access(%2, "b")
// CHECK:         %10 = call(%9, {})
// CHECK:         %11 = inline_block_break(%8, %10)
// CHECK:     })
// CHECK:     "E" %12 = inline_block({
// CHECK:         %13 = access(%2, "b")
// CHECK:         %14 = int_type(u, 1)
// CHECK:         %15 = call(%13, {
// CHECK:             %14,
// CHECK:             %8,
// CHECK:         })
// CHECK:         %16 = inline_block_break(%12, %15)
// CHECK:     })
// CHECK:     "F" %17 = inline_block({
// CHECK:         %18 = bit_or(%5, %8)
// CHECK:         %19 = inline_block_break(%17, %18)
// CHECK:     })
// CHECK:     "G" %20 = inline_block({
// CHECK:         %21 = int_type(u, 1)
// CHECK:         %22 = ref_ty(const %21)
// CHECK:         %23 = inline_block_break(%20, %22)
// CHECK:     })
// CHECK:     "H" %24 = inline_block({
// CHECK:         %25 = int_type(u, 1)
// CHECK:         %26 = ref_ty(mut %25)
// CHECK:         %27 = inline_block_break(%24, %26)
// CHECK:     })
// CHECK:     "I" @undefined
// CHECK:     "J" @bool_true
// CHECK:     "K" @bool_false
// CHECK:     "L" %28 = inline_block({
// CHECK:         %29 = deref(%1)
// CHECK:         %30 = inline_block_break(%28, %29)
// CHECK:     })
// CHECK:     "M" %31 = inline_block({
// CHECK:         %32 = return(%1)
// CHECK:         %33 = inline_block_break(%31, %32)
// CHECK:     })
// CHECK:     "N" %34 = int_literal(0)
// CHECK:     "O" %35 = int_literal(0)
// CHECK:     "P" %36 = int_literal(15)
// CHECK:     "Q" %37 = inline_block({
// CHECK:         %38 = int_literal(7)
// CHECK:         %39 = int_type(u, 3)
// CHECK:         %40 = as(%39, %38)
// CHECK:         %41 = inline_block_break(%37, %40)
// CHECK:     })
// CHECK:     "R" %42 = inline_block({
// CHECK:         %43 = int_literal(4294967295)
// CHECK:         %44 = int_type(u, 32)
// CHECK:         %45 = as(%44, %43)
// CHECK:         %46 = inline_block_break(%42, %45)
// CHECK:     })
// CHECK:     "S" %47 = inline_block({
// CHECK:         %48 = branch(%31, {
// CHECK:             %49 = inline_block({
// CHECK:                 %50 = inline_block_break(%49, %1)
// CHECK:             })}, {
// CHECK:             %51 = inline_block({
// CHECK:                 %52 = bit_xor(%1, %2)
// CHECK:                 %53 = inline_block_break(%51, %52)
// CHECK:             })
// CHECK:         })
// CHECK:         %54 = inline_block_break(%47, %48)
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
