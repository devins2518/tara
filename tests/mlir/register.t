// RUN: @tara @file --dump-mlir --exit-early 2>&1

// CHECK: module {
const Top = module {
    // CHECK:   hw.module @root.Top.ffSync(in %clk : !seq.clock, in %rst : i1, in %in : i1, out root.Top.ffSync : i1) {
    pub comb ffSync(clk: clock, rst: reset, in: u1) u1 {
        // CHECK:     %false = hw.constant false
        const reset_val: u1 = 0;
        // CHECK:     %false_0 = hw.constant false
        // CHECK:     %0 = seq.compreg %1, %clk reset %rst, %false : i1
        const reg = @reg(clk, rst, reset_val);
        // CHECK:     %1 = comb.xor bin %in, %0 : i1
        @regWrite(reg, in ^ reg);
        // CHECK:     %2 = comb.and %in, %0 : i1
        // CHECK:     hw.output %2 : i1
        return in & reg;
    }
    // CHECK:   }
};
// CHECK: }
