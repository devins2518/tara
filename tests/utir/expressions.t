// RUN: @tara @file --dump-utir
// CHECK: %0 = struct_decl({
// CHECK:     "A" %1 = decl_val("u1")
// CHECK:     "B" %2 = inline_block({
// CHECK:         %3 = decl_val("a")
// CHECK:         %4 = decl_val("b")
// CHECK:         %5 = bit_and(%3, %4)
// CHECK:         %6 = inline_block_break(%2, %5)
// CHECK:     })
// CHECK:     "C" %7 = inline_block({
// CHECK:         %8 = decl_val("a")
// CHECK:         %9 = negate(%8)
// CHECK:         %10 = inline_block_break(%7, %9)
// CHECK:     })
// CHECK:     "D" %11 = inline_block({
// CHECK:         %12 = inline_block({
// CHECK:             %13 = decl_val("a")
// CHECK:             %14 = decl_val("b")
// CHECK:             %15 = access(%13, %14)
// CHECK:             %16 = inline_block_break(%12, %15)
// CHECK:         })
// CHECK:         %17 = call(%12, {})
// CHECK:         %18 = inline_block_break(%11, %17)
// CHECK:     })
// CHECK:     "E" %19 = inline_block({
// CHECK:         %20 = inline_block({
// CHECK:             %21 = decl_val("a")
// CHECK:             %22 = decl_val("b")
// CHECK:             %23 = access(%21, %22)
// CHECK:             %24 = inline_block_break(%20, %23)
// CHECK:         })
// CHECK:         %26 = decl_val("c")
// CHECK:         %27 = decl_val("d")
// CHECK:         %25 = call(%20, {
// CHECK:             %26,
// CHECK:             %27,
// CHECK:         })
// CHECK:         %28 = inline_block_break(%19, %25)
// CHECK:     })
// CHECK:     "F" %29 = inline_block({
// CHECK:         %30 = decl_val("a")
// CHECK:         %31 = decl_val("b")
// CHECK:         %32 = bit_or(%30, %31)
// CHECK:         %33 = inline_block_break(%29, %32)
// CHECK:     })
// CHECK:     "G" %34 = inline_block({
// CHECK:         %35 = decl_val("u1")
// CHECK:         %36 = ref_ty(const %35)
// CHECK:         %37 = inline_block_break(%34, %36)
// CHECK:     })
// CHECK:     "H" %38 = inline_block({
// CHECK:         %39 = decl_val("u1")
// CHECK:         %40 = ref_ty(mut %39)
// CHECK:         %41 = inline_block_break(%38, %40)
// CHECK:     })
// CHECK:     "I" %42 = decl_val("undefined")
// CHECK:     "J" @bool_true
// CHECK:     "K" @bool_false
// CHECK:     "L" %43 = inline_block({
// CHECK:         %44 = decl_val("a")
// CHECK:         %45 = deref(%44)
// CHECK:         %46 = inline_block_break(%43, %45)
// CHECK:     })
// CHECK:     "M" %47 = inline_block({
// CHECK:         %48 = decl_val("a")
// CHECK:         %49 = return(%48)
// CHECK:         %50 = inline_block_break(%47, %49)
// CHECK:     })
// CHECK:     "N" %51 = int_literal(0)
// CHECK:     "O" %52 = int_literal(0)
// CHECK:     "P" %53 = int_literal(15)
// CHECK:     "Q" %54 = inline_block({
// CHECK:         %55 = int_literal(7)
// CHECK:         %56 = int_type(u, 3)
// CHECK:         %57 = as(%56, %55)
// CHECK:         %58 = inline_block_break(%54, %57)
// CHECK:     })
// CHECK:     "R" %59 = inline_block({
// CHECK:         %60 = int_literal(4294967295)
// CHECK:         %61 = int_type(u, 32)
// CHECK:         %62 = as(%61, %60)
// CHECK:         %63 = inline_block_break(%59, %62)
// CHECK:     })
// CHECK:     "S" %64 = inline_block({
// CHECK:         %65 = decl_val("a")
// CHECK:         %66 = branch(%65, {
// CHECK:             %67 = int_literal(32)
// CHECK:         }, {
// CHECK:             %68 = int_literal(16)
// CHECK:         })
// CHECK:         %69 = inline_block_break(%64, %66)
// CHECK:     })
// CHECK: })

const A = u1;
const B = a & b;
const C = -a;
const D = a.b();
const E = a.b(c, d);
const F = (a | b);
const G = &u1;
const H = &var u1;
const I = undefined;
const J = true;
const K = false;
const L = a.*;
const M = return a;
const N = 0;
const O = 0b0;
const P = 0xF;
const Q = 3b111;
const R = 32xFFFFFFFF;
const S = if (a) 32 else 16;
