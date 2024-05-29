// RUN: TERM=dumb @tara @file --dump-mlir --exit-early 2>&1

const Top = module {
    pub comb ffSync(clk: clock, rst: reset, in: u1) u1 {
        const reset_val: u1 = 0;
        const reg = @reg(clk, rst, reset_val);
        @regWrite(reg, in ^ reg);
        // CHECK: error: Linear element already consumed when used here!
        // CHECK:    │
        // CHECK: 12 │         @regWrite(reg, in & reg);
        // CHECK:    │         ^^^^^^^^^^^^^^^^^^^^^^^^
        @regWrite(reg, in & reg);
        return reg;
    }
};
