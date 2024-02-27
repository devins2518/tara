// RUN: @tara @file --dump-utir --exit-early
// CHECK: %0 = struct_decl({
// CHECK:     "S" %1 = struct_decl({
// CHECK:         "a" %2 = subroutine_decl(
// CHECK:             {}
// CHECK:             %3 = int_type(u, 1)
// CHECK:             %4 = block({
// CHECK:                 %7 = inline_block({
// CHECK:                     %8 = int_literal(1)
// CHECK:                     %9 = return(%8)
// CHECK:                     %10 = inline_block_break(%7, %9)
// CHECK:                 })
// CHECK:                 %11 = ret_implicit_void()
// CHECK:             })
// CHECK:         )
// CHECK:     })
// CHECK:     "M" %12 = module_decl({
// CHECK:         "a" %13 = subroutine_decl(
// CHECK:             {}
// CHECK:             %14 = int_type(u, 1)
// CHECK:             %15 = block({
// CHECK:                 %18 = inline_block({
// CHECK:                     %19 = int_literal(1)
// CHECK:                     %20 = return(%19)
// CHECK:                     %21 = inline_block_break(%18, %20)
// CHECK:                 })
// CHECK:                 %22 = ret_implicit_void()
// CHECK:             })
// CHECK:         )
// CHECK:     })
// CHECK: })

const S = struct {
    pub fn a() u1 {
        const b = u3;
        return 1;
    }
};

const M = module {
    pub comb a() u1 {
        const b = u3;
        return 1;
    }
};

