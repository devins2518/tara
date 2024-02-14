// RUN: @tara @file --dump-utir
// CHECK: %0 = struct_decl({
// CHECK:     "S" %1 = struct_decl({
// CHECK:         "A" %3 = decl_val("bool")
// CHECK:         "B" %4 = struct_decl({
// CHECK:             "C" %5 = decl_val("u1")
// CHECK:         })
// CHECK:     })
// CHECK: })

const S = struct {
    const A = bool;
    const B = struct {
        const C = u1;
    };

    d: u1,
};
