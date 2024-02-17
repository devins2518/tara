// RUN: @tara @file --dump-utir
// CHECK: %0 = struct_decl({
// CHECK:     "S" %1 = struct_decl({
// CHECK:         "d": %2 = decl_val("u1")
// CHECK:         "A" @bool_type
// CHECK:         "B" %3 = struct_decl({
// CHECK:             "C" %4 = decl_val("u1")
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
