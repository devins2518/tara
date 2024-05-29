// RUN: @tara @file --dump-mlir --exit-early 2>&1

// CHECK: module {
const A = struct {
    // CHECK: func.func @root.A.if_expr(%arg0: i1) -> i1 {
    fn if_expr(b: bool) u1 {
        // CHECK: %true = arith.constant true
        const one: u1 = 1;
        // CHECK: %false = arith.constant false
        // CHECK: %0 = arith.select %arg0, %true, %false : i1
        // CHECK: return %0 : i1
        return if (b) one else 0;
    }
    // CHECK: }

    // CHECK: func.func @root.A.if_stmt(%arg0: i1) -> i1 {
    fn if_stmt(b: bool) u1 {
        // CHECK: %c0_i0 = arith.constant 0 : i0
        // CHECK: %c0_i0_0 = arith.constant 0 : i0
        // CHECK: %0 = arith.select %arg0, %c0_i0, %c0_i0_0 : i0
        if (b) {} else {}
        // CHECK: %true = arith.constant true
        // CHECK: return %true : i1
        return 1;
    }
    // CHECK: }

    // CHECK: func.func @root.A.if_stmt_no_else(%arg0: i1) -> i1 {
    fn if_stmt_no_else(b: bool) u1 {
        // CHECK: %c0_i0 = arith.constant 0 : i0
        // CHECK: %c0_i0_0 = arith.constant 0 : i0
        // CHECK: %0 = arith.select %arg0, %c0_i0, %c0_i0_0 : i0
        if (b) {}
        // CHECK: %true = arith.constant true
        // CHECK: return %true : i1
        return 1;
    }
    // CHECK: }
};

const B = module {
    // CHECK: hw.module @root.B.if_expr(in %b : i1, out root.B.if_expr : i1) {
    comb if_expr(b: bool) u1 {
        // CHECK: %true = hw.constant true
        const one: u1 = 1;
        // CHECK: %false = hw.constant false
        // CHECK: %0 = comb.mux bin %b, %true, %false : i1
        // CHECK: hw.output %0 : i1
        return if (b) one else 0;
    }
    // CHECK: }

    // CHECK: hw.module @root.B.if_stmt(in %b : i1, out root.B.if_stmt : i1) {
    comb if_stmt(b: bool) u1 {
        // CHECK: %c0_i0 = hw.constant 0 : i0
        // CHECK: %c0_i0_0 = hw.constant 0 : i0
        // CHECK: %0 = comb.mux bin %b, %c0_i0, %c0_i0_0 : i0
        if (b) {} else {}
        // CHECK: %true = hw.constant true
        // CHECK: hw.output %true : i1
        return 1;
    }
    // CHECK: }

    // CHECK: hw.module @root.B.if_stmt_no_else(in %b : i1, out root.B.if_stmt_no_else : i1) {
    comb if_stmt_no_else(b: bool) u1 {
        // CHECK: %c0_i0 = hw.constant 0 : i0
        // CHECK: %c0_i0_0 = hw.constant 0 : i0
        // CHECK: %0 = comb.mux bin %b, %c0_i0, %c0_i0_0 : i0
        if (b) {}
        // CHECK: %true = hw.constant true
        // CHECK: hw.output %true : i1
        return 1;
    }
    // CHECK: }
};
// CHECK: }
