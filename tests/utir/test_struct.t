// RUN: @tara @file --dump-utir
// CHECK: %0 = struct_decl({
// CHECK:     "S" %1 = struct_decl({
// CHECK:         "d": %2 = int_type(u, 1)
// CHECK:         "A" @bool_type
// CHECK:         "B" %3 = struct_decl({
// CHECK:             "C" %4 = int_type(u, 1)
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
