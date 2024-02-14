// RUN: @tara @file --dump-utir
// CHECK: %0 = struct_decl({
// CHECK:     "A" %1 = decl_val("bool")
// CHECK:     "B" %2 = decl_val("u1")
// CHECK:     "C" %3 = decl_val("i1")
// CHECK:     "D" %4 = decl_val("sig")
// CHECK:     "E" %5 = module_decl({})
// CHECK:     "F" %6 = struct_decl({})
// CHECK:     "G" %7 = decl_val("type")
// CHECK:     "H" %8 = decl_val("clock")
// CHECK:     "I" %9 = decl_val("reset")
// CHECK: })

const A = bool;
const B = u1;
const C = i1;
const D = sig;
const E = module {};
const F = struct {};
const G = type;
const H = clock;
const I = reset;
