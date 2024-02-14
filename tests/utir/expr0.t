// RUN: @tara @file --dump-utir
// CHECK: %0 = struct_decl({
// CHECK:     "A" %1 = decl_val("a")
// CHECK: })

const A = a;
