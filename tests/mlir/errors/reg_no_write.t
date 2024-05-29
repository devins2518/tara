// RUN: TERM=dumb @tara @file --dump-mlir --exit-early 2>&1

const Top = module {
    pub comb test0(clk: clock, rst: reset, in: u1) u1 {
        const reset_val: u1 = 0;
        // CHECK: error: Linear value not consumed!
        // CHECK:    │
        // CHECK: 10 │         const reg = @reg(clk, rst, reset_val);
        // CHECK:    │                     ^^^^^^^^^^^^^^^^^^^^^^^^^
        const reg = @reg(clk, rst, reset_val);
        return reg;
    }

    pub comb test1(clk: clock, rst: reset, in: u1) u1 {
        const reset_val: u1 = 0;
        // CHECK: error: Linear value not consumed!
        // CHECK:    │
        // CHECK: 20 │         return @reg(clk, rst, reset_val);
        // CHECK:    │                ^^^^^^^^^^^^^^^^^^^^^^^^^
        return @reg(clk, rst, reset_val);
    }

    pub comb test2(clk: clock, rst: reset, in: u1) u1 {
        const reset_val: u1 = 0;
        // CHECK: error: Linear value not consumed!
        // CHECK:    │
        // CHECK: 29 │         const reg: u1 = @reg(clk, rst, reset_val);
        // CHECK:    │                         ^^^^^^^^^^^^^^^^^^^^^^^^^
        const reg: u1 = @reg(clk, rst, reset_val);
        return reg;
    }

    pub comb test3(clk: clock, rst: reset, in: u1) u1 {
        const reset_val: u1 = 0;
        const reg = @reg(clk, rst, reset_val);
        // CHECK: error: Linear elements have different consumption statuses in the true and false branches!
        // CHECK:    │
        // CHECK: 41 │ ╭         if (in == 0) @regWrite(reg, 0);
        // CHECK: 42 │ │         return reg;
        // CHECK:    │ ╰────────^
        if (in == 0) @regWrite(reg, 0);
        return reg;
    }
};
