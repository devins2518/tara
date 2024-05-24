// RUN: TERM=dumb @tara @file --dump-mlir --exit-early 2>&1

const Top = module {
    pub comb ffSync(clk: clock, rst: reset, in: u1) u1 {
        const reset_val: u1 = 0;
        // CHECK: error: Register not written to!
        // CHECK:    │
        // CHECK: 10 │         const reg = @reg(clk, rst, reset_val);
        // CHECK:    │                     ^^^^^^^^^^^^^^^^^^^^^^^^^
        const reg = @reg(clk, rst, reset_val);
        return reg;
    }
};
