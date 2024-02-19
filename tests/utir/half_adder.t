// RUN: @tara @file --dump-utir
// CHECK: %0 = struct_decl({
// CHECK:     "HalfAdder" %1 = module_decl({
// CHECK:         "sum" %2 = subroutine_decl(
// CHECK:             {
// CHECK:                 %3 = inline_block({
// CHECK:                     %4 = int_type(u, 1)
// CHECK:                     %5 = ref_ty(const %4)
// CHECK:                     %6 = inline_block_break(%3, %5)
// CHECK:                 })
// CHECK:                 %7: %3
// CHECK:                 %8 = inline_block({
// CHECK:                     %9 = int_type(u, 1)
// CHECK:                     %10 = ref_ty(const %9)
// CHECK:                     %11 = inline_block_break(%8, %10)
// CHECK:                 })
// CHECK:                 %12: %8
// CHECK:             }
// CHECK:             %13 = int_type(u, 1)
// CHECK:             %14 = inline_block({
// CHECK:                 %15 = inline_block({
// CHECK:                     %16 = bit_xor(%7, %12)
// CHECK:                     %17 = inline_block_break(%15, %16)
// CHECK:                 })
// CHECK:                 %18 = return(%15)
// CHECK:                 %19 = inline_block_break(%14, %18)
// CHECK:             })
// CHECK:         )
// CHECK:         "carry" %20 = subroutine_decl(
// CHECK:             {
// CHECK:                 %21 = inline_block({
// CHECK:                     %22 = int_type(u, 1)
// CHECK:                     %23 = ref_ty(const %22)
// CHECK:                     %24 = inline_block_break(%21, %23)
// CHECK:                 })
// CHECK:                 %25: %21
// CHECK:                 %26 = inline_block({
// CHECK:                     %27 = int_type(u, 1)
// CHECK:                     %28 = ref_ty(const %27)
// CHECK:                     %29 = inline_block_break(%26, %28)
// CHECK:                 })
// CHECK:                 %30: %26
// CHECK:             }
// CHECK:             %31 = int_type(u, 1)
// CHECK:             %32 = inline_block({
// CHECK:                 %33 = inline_block({
// CHECK:                     %34 = bit_and(%25, %30)
// CHECK:                     %35 = inline_block_break(%33, %34)
// CHECK:                 })
// CHECK:                 %36 = return(%33)
// CHECK:                 %37 = inline_block_break(%32, %36)
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
