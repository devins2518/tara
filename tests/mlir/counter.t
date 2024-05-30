// RUN: @tara @file --dump-mlir --exit-early 2>&1

// CHECK: module {
const Counter = module {
    // CHECK: hw.module @root.Counter.count(in %clk : !seq.clock, in %rst : i1, in %rollover : i8, in %count_en : i1, out root.Counter.count : i8) {
    pub comb count(clk: clock, rst: reset, rollover: u8, count_en: bool) u8 {
        // CHECK: %c0_i8 = hw.constant 0 : i8
        const reset_val: u8 = 0;
        // CHECK: %c0_i8_0 = hw.constant 0 : i8
        // CHECK: %0 = seq.compreg %4, %clk reset %rst, %c0_i8 : i8
        const counter = @reg(clk, rst, reset_val);
        // CHECK: %c1_i8 = hw.constant 1 : i8
        // CHECK: %1 = comb.add bin %0, %c1_i8 : i8
        // CHECK: %2 = comb.icmp eq %0, %rollover : i8
        // CHECK: %c0_i8_1 = hw.constant 0 : i8
        // CHECK: %3 = comb.mux bin %2, %c0_i8_1, %0 : i8
        // CHECK: %4 = comb.mux bin %count_en, %1, %3 : i8
        const next_counter = if (count_en) counter + 1 else if (counter == rollover) reset_val else counter;
        @regWrite(counter, next_counter);
        // CHECK: hw.output %0 : i8
        return counter;
    }
    // CHECK: }
};
// CHECK: }
