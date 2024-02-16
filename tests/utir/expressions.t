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
// CHECK:         %25 = call(%20, {
// CHECK:             %3,
// CHECK:             %20,
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
// CHECK:     "J" %43 = decl_val("true")
// CHECK:     "K" %44 = decl_val("false")
// CHECK:     "L" %45 = inline_block({
// CHECK:         %46 = decl_val("a")
// CHECK:         %47 = deref(%46)
// CHECK:         %48 = inline_block_break(%45, %47)
// CHECK:     })
// CHECK:     "M" %49 = inline_block({
// CHECK:         %50 = decl_val("a")
// CHECK:         %51 = return(%50)
// CHECK:         %52 = inline_block_break(%49, %51)
// CHECK:     })
// CHECK:     "N" %53 = int_literal(0)
// CHECK:     "O" %54 = int_literal(0)
// CHECK:     "P" %55 = int_literal(15)
// CHECK:     "Q" %56 = inline_block({
// CHECK:         %57 = int_literal(7)
// CHECK:         %58 = int_type(u, 3)
// CHECK:         %59 = as(%58, %57)
// CHECK:         %60 = inline_block_break(%56, %59)
// CHECK:     })
// CHECK:     "R" %61 = inline_block({
// CHECK:         %62 = int_literal(4294967295)
// CHECK:         %63 = int_type(u, 32)
// CHECK:         %64 = as(%63, %62)
// CHECK:         %65 = inline_block_break(%61, %64)
// CHECK:     })
// CHECK:     "S" %66 = inline_block({
// CHECK:         %67 = decl_val("a")
// CHECK:         %68 = branch(%67, {
// CHECK:             %69 = int_literal(32)
// CHECK:         }, {
// CHECK:             %70 = int_literal(16)
// CHECK:         })
// CHECK:         %71 = inline_block_break(%66, %68)
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
