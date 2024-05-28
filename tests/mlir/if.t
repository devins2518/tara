// RUN: @tara @file --dump-mlir --exit-early 2>&1

// CHECK: module {
const A = struct {
    // CHECK: func.func @root.A.a(%arg0: i1) -> i1 {
    fn a(b: bool) u1 {
        // CHECK: %true = arith.constant true
        const one: u1 = 1;
        // CHECK: %false = arith.constant false
        // CHECK: %0 = arith.select %arg0, %true, %false : i1
        // CHECK: return %0 : i1
        return if (b) one else 0;
    }
    // CHECK: }
};

const B = module {
    // CHECK: hw.module @root.B.a(in %b : i1, out root.B.a : i1) {
    comb a(b: bool) u1 {
        // CHECK: %true = hw.constant true
        const one: u1 = 1;
        // CHECK: %false = hw.constant false
        // CHECK: %0 = comb.mux bin %b, %true, %false : i1
        // CHECK: hw.output %0 : i1
        return if (b) one else 0;
    }
    // CHECK: }
};
// CHECK: }
