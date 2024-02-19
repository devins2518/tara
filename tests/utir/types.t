// RUN: @tara @file --dump-utir
// CHECK: %0 = struct_decl({
// CHECK:     "A" @bool_type
// CHECK:     "B" %1 = int_type(u, 1)
// CHECK:     "C" %2 = int_type(i, 1)
// CHECK:     "D" @sig_type
// CHECK:     "E" %3 = module_decl({})
// CHECK:     "F" %4 = struct_decl({})
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
