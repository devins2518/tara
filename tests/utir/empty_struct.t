// RUN: @tara @file --dump-utir
// CHECK: %0 = struct_decl({
// CHECK:     "S" %1 = struct_decl({})
// CHECK: })

const S = struct {};
