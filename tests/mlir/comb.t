// RUN: @tara @file --dump-mlir --exit-early 2>&1 | ./install/bin/circt-opt --canonicalize
// CHECK: module {
const Top = module {
    // CHECK:   hw.module @root.Top.bitwise(in %a : i1, in %b : i1, out root.Top.bitwise : i1) {
    // CHECK:     %0 = comb.and %a, %b : i1
    // CHECK:     hw.output %0 : i1
    // CHECK:   }
    pub comb bitwise(a: u1, b: u1) u1 {
        return a & b;
    }

    // CHECK:   hw.module @root.Top.arith(in %a : i1, in %b : i1, out root.Top.arith : i1) {
    // CHECK:     %0 = comb.add bin %a, %b : i1
    // CHECK:     hw.output %0 : i1
    // CHECK:   }
    pub comb arith(a: u1, b: u1) u1 {
        return a + b;
    }

    // CHECK:   hw.module @root.Top.cmp(in %a : i1, in %b : i1, out root.Top.cmp : i1) {
    // CHECK:     %0 = comb.icmp bin ugt %a, %b : i1
    // CHECK:     hw.output %0 : i1
    // CHECK:   }
    pub comb cmp(a: u1, b: u1) bool {
        return a > b;
    }

    // CHECK:   hw.module @root.Top.cmp_bool(in %c : i1, in %d : i1, out root.Top.cmp_bool : i1) {
    // CHECK:     %0 = comb.and %c, %d : i1
    // CHECK:     hw.output %0 : i1
    // CHECK:   }
    pub comb cmp_bool(c: bool, d: bool) bool {
        return c and d;
    }

    // CHECK:   hw.module @root.Top.bitsize_cast(in %e : i2, in %b : i1, out root.Top.bitsize_cast : i2) {
    // CHECK:     %0 = arith.extui %b : i1 to i2
    // CHECK:     %1 = comb.and %e, %0 : i2
    // CHECK:     hw.output %1 : i2
    // CHECK:   }
    pub comb bitsize_cast(e: u2, b: u1) u2 {
        return e & b;
    }

    // CHECK:   hw.module @root.Top.arith_cast(in %e : i2, in %b : i1, out root.Top.arith_cast : i2) {
    // CHECK:     %0 = arith.extui %b : i1 to i2
    // CHECK:     %1 = comb.add bin %e, %0 : i2
    // CHECK:     hw.output %1 : i2
    // CHECK:   }
    pub comb arith_cast(e: u2, b: u1) u2 {
        return e + b;
    }

    // CHECK:   hw.module @root.Top.cmp_cast(in %e : i2, in %b : i1, out root.Top.cmp_cast : i1) {
    // CHECK:     %0 = arith.extui %b : i1 to i2
    // CHECK:     %1 = comb.icmp bin ugt %e, %0 : i2
    // CHECK:     hw.output %1 : i1
    // CHECK:   }
    pub comb cmp_cast(e: u2, b: u1) bool {
        return e > b;
    }

    // CHECK:   hw.module @root.Top.ret_val_cast(in %f : i1, in %g : i1, out root.Top.ret_val_cast : i2) {
    // CHECK:     %0 = comb.and %f, %g : i1
    // CHECK:     %1 = arith.extsi %0 : i1 to i2
    // CHECK:     hw.output %1 : i2
    // CHECK:   }
    pub comb ret_val_cast(f: i1, g: i1) i2 {
        return f & g;
    }

    // CHECK:   hw.module @root.Top.test_void_ret(in %hi : i1, in %there : i2) {
    // CHECK:     hw.output
    // CHECK:   }
    pub comb test_void_ret(hi: u1, there: i2) void {}
};

// CHECK: }
