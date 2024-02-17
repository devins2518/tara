// RUN: @tara @file --dump-utir
// CHECK: %0 = struct_decl({
// CHECK:     "HalfAdder" %1 = module_decl({
// CHECK:         "sum" %2 = subroutine_decl(
// CHECK:             {
// CHECK:                 %3 = inline_block({
// CHECK:                     %4 = decl_val("u1")
// CHECK:                     %5 = ref_ty(const %4)
// CHECK:                     %6 = inline_block_break(%3, %5)
// CHECK:                 })
// CHECK:                 %7: %3
// CHECK:                 %8 = inline_block({
// CHECK:                     %9 = decl_val("u1")
// CHECK:                     %10 = ref_ty(const %9)
// CHECK:                     %11 = inline_block_break(%8, %10)
// CHECK:                 })
// CHECK:                 %12: %8
// CHECK:             }
// CHECK:             %13 = decl_val("u1")
// CHECK:             %14 = inline_block({
// CHECK:                 %15 = inline_block({
// CHECK:                     %16 = decl_val("a")
// CHECK:                     %17 = decl_val("b")
// CHECK:                     %18 = bit_xor(%16, %17)
// CHECK:                     %19 = inline_block_break(%15, %18)
// CHECK:                 })
// CHECK:                 %20 = return(%15)
// CHECK:                 %21 = inline_block_break(%14, %20)
// CHECK:             })
// CHECK:         )
// CHECK:         "carry" %22 = subroutine_decl(
// CHECK:             {
// CHECK:                 %23 = inline_block({
// CHECK:                     %24 = decl_val("u1")
// CHECK:                     %25 = ref_ty(const %24)
// CHECK:                     %26 = inline_block_break(%23, %25)
// CHECK:                 })
// CHECK:                 %27: %23
// CHECK:                 %28 = inline_block({
// CHECK:                     %29 = decl_val("u1")
// CHECK:                     %30 = ref_ty(const %29)
// CHECK:                     %31 = inline_block_break(%28, %30)
// CHECK:                 })
// CHECK:                 %32: %28
// CHECK:             }
// CHECK:             %33 = decl_val("u1")
// CHECK:             %34 = inline_block({
// CHECK:                 %35 = inline_block({
// CHECK:                     %36 = decl_val("a")
// CHECK:                     %37 = decl_val("b")
// CHECK:                     %38 = bit_and(%36, %37)
// CHECK:                     %39 = inline_block_break(%35, %38)
// CHECK:                 })
// CHECK:                 %40 = return(%35)
// CHECK:                 %41 = inline_block_break(%34, %40)
// CHECK:             })
// CHECK:         )
// CHECK:     })
// CHECK: })

const HalfAdder = module {
    pub comb sum(a: &u1, b: &u1) u1 {
        return a ^ b;
    }
    pub comb carry(a: &u1, b: &u1) u1 {
        return a & b;
    }
};
