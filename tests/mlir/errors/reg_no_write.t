// RUN: TERM=dumb @tara @file --dump-mlir --exit-early 2>&1

const Top = module {
    pub comb test0(clk: clock, rst: reset, in: u1) u1 {
        const reset_val: u1 = 0;
        // CHECK: error: Register not written to!
        // CHECK:    │
        // CHECK: 10 │         const reg = @reg(clk, rst, reset_val);
        // CHECK:    │                     ^^^^^^^^^^^^^^^^^^^^^^^^^
        const reg = @reg(clk, rst, reset_val);
        return reg;
    }

    pub comb test1(clk: clock, rst: reset, in: u1) u1 {
        const reset_val: u1 = 0;
        // CHECK: error: Register not written to!
        // CHECK:    │
        // CHECK: 20 │         return @reg(clk, rst, reset_val);
        // CHECK:    │                ^^^^^^^^^^^^^^^^^^^^^^^^^
        return @reg(clk, rst, reset_val);
    }
};
