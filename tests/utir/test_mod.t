// RUN: @tara @file --dump-utir
// CHECK: %0 = struct_decl({
// CHECK:     "Mod" %1 = module_decl({
// CHECK:         "A" %6 = decl_val("bool")
// CHECK:         "B" %7 = module_decl({
// CHECK:             "C" %8 = decl_val("u1")
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