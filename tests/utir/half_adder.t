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
// CHECK:                 "a" : %3
// CHECK:                 %7 = inline_block({
// CHECK:                     %8 = decl_val("u1")
// CHECK:                     %9 = ref_ty(const %8)
// CHECK:                     %10 = inline_block_break(%7, %9)
// CHECK:                 })
// CHECK:                 "b" : %7
// CHECK:             }
// CHECK:             %11 = decl_val("u1")
// CHECK:             %12 = inline_block({
// CHECK:                 %13 = inline_block({
// CHECK:                     %14 = decl_val("a")
// CHECK:                     %15 = decl_val("b")
// CHECK:                     %16 = bit_xor(%14, %15)
// CHECK:                     %17 = inline_block_break(%13, %16)
// CHECK:                 })
// CHECK:                 %18 = return(%13)
// CHECK:                 %19 = inline_block_break(%12, %18)
// CHECK:             })
// CHECK:         )
// CHECK:         "carry" %20 = subroutine_decl(
// CHECK:             {
// CHECK:                 %21 = inline_block({
// CHECK:                     %22 = decl_val("u1")
// CHECK:                     %23 = ref_ty(const %22)
// CHECK:                     %24 = inline_block_break(%21, %23)
// CHECK:                 })
// CHECK:                 "a" : %21
// CHECK:                 %25 = inline_block({
// CHECK:                     %26 = decl_val("u1")
// CHECK:                     %27 = ref_ty(const %26)
// CHECK:                     %28 = inline_block_break(%25, %27)
// CHECK:                 })
// CHECK:                 "b" : %25
// CHECK:             }
// CHECK:             %29 = decl_val("u1")
// CHECK:             %30 = inline_block({
// CHECK:                 %31 = inline_block({
// CHECK:                     %32 = decl_val("a")
// CHECK:                     %33 = decl_val("b")
// CHECK:                     %34 = bit_and(%32, %33)
// CHECK:                     %35 = inline_block_break(%31, %34)
// CHECK:                 })
// CHECK:                 %36 = return(%31)
// CHECK:                 %37 = inline_block_break(%30, %36)
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
