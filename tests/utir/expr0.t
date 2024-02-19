// RUN: @tara @file --dump-utir
// CHECK: %0 = struct_decl({
// CHECK:     "A" %1 = int_literal(1)
// CHECK: })

const A = 1;
