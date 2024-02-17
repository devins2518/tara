// RUN: @tara @file --dump-utir
// CHECK: %0 = struct_decl({
// CHECK:     "A" @bool_type
// CHECK:     "B" %1 = decl_val("u1")
// CHECK:     "C" %2 = decl_val("i1")
// CHECK:     "D" %3 = decl_val("sig")
// CHECK:     "E" %4 = module_decl({})
// CHECK:     "F" %5 = struct_decl({})
// CHECK:     "G" @type_type
// CHECK:     "H" @clock_type
// CHECK:     "I" @reset_type
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
