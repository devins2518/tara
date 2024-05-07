// RUN: @tara @file --dump-mlir --exit-early 2>&1 | ./install/bin/circt-opt --canonicalize
// CHECK: module {
// CHECK:   arc.define @top(%arg0: i1, %arg1: i1) -> i1 {
// CHECK:     %0 = comb.and %arg0, %arg1 : i1
// CHECK:     arc.output %0 : i1
// CHECK:   }
// CHECK:   hw.module @Top(in %a : i1, in %b : i1, out top : i1) {
// CHECK:     %0 = arc.call @top(%a, %b) : (i1, i1) -> i1
// CHECK:     hw.output %0 : i1
// CHECK:   }
// CHECK: }

const Top = module {
    pub comb top(a: u1, b: u1) u1 {
        return a & b;
    }
};
