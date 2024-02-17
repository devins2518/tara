// RUN: @tara @file --dump-utir
// CHECK: %0 = struct_decl({
// CHECK:     "Mod" %1 = module_decl({
// CHECK:         "d": %2 = inline_block({
// CHECK:             %3 = decl_val("u1")
// CHECK:             %4 = ref_ty(const %3)
// CHECK:             %5 = inline_block_break(%2, %4)
// CHECK:         })
// CHECK:         "A" @bool_type
// CHECK:         "B" %6 = module_decl({
// CHECK:             "C" %7 = decl_val("u1")
// CHECK:         })
// CHECK:     })
// CHECK: })

const Mod = module {
    const A = bool;
    const B = module {
        const C = u1;
    };

    d: &u1,
};
