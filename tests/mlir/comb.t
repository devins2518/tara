// RUN: @tara @file --dump-mlir --exit-early 2>&1 | ./install/bin/circt-opt --canonicalize
// CHECK: module {
const Top = module {
    // CHECK:   arc.define @bitwise(%arg0: i1, %arg1: i1) -> i1 {
    // CHECK:     %0 = comb.and %arg0, %arg1 : i1
    // CHECK:     arc.output %0 : i1
    // CHECK:   }
    pub comb bitwise(a: u1, b: u1) u1 {
        return a & b;
    }

    // CHECK:   arc.define @arith(%arg0: i1, %arg1: i1) -> i1 {
    // CHECK:     %0 = comb.add %arg0, %arg1 : i1
    // CHECK:     arc.output %0 : i1
    // CHECK:   }
    pub comb arith(a: u1, b: u1) u1 {
        return a + b;
    }

    // CHECK:   arc.define @cmp(%arg0: i1, %arg1: i1) -> i1 {
    // CHECK:     %0 = comb.icmp bin ugt %arg0, %arg1 : i1
    // CHECK:     arc.output %0 : i1
    // CHECK:   }
    pub comb cmp(a: u1, b: u1) bool {
        return a > b;
    }

    // CHECK:   arc.define @cmp_bool(%arg0: i1, %arg1: i1) -> i1 {
    // CHECK:     %0 = comb.and %arg0, %arg1 : i1
    // CHECK:     arc.output %0 : i1
    // CHECK:   }
    pub comb cmp_bool(c: bool, d: bool) bool {
        return c and d;
    }

    // CHECK:   arc.define @bitsize_cast(%arg0: i2, %arg1: i1) -> i2 {
    // CHECK:     %0 = arith.extui %arg1 : i1 to i2
    // CHECK:     %1 = comb.and %arg0, %0 : i2
    // CHECK:     arc.output %1 : i2
    // CHECK:   }
    pub comb bitsize_cast(e: u2, b: u1) u2 {
        return e & b;
    }

    // CHECK:   arc.define @arith_cast(%arg0: i2, %arg1: i1) -> i2 {
    // CHECK:     %0 = arith.extui %arg1 : i1 to i2
    // CHECK:     %1 = comb.add %arg0, %0 : i2
    // CHECK:     arc.output %1 : i2
    // CHECK:   }
    pub comb arith_cast(e: u2, b: u1) u2 {
        return e + b;
    }

    // CHECK:   arc.define @cmp_cast(%arg0: i2, %arg1: i1) -> i1 {
    // CHECK:     %0 = arith.extui %arg1 : i1 to i2
    // CHECK:     %1 = comb.icmp bin ugt %arg0, %0 : i2
    // CHECK:     arc.output %1 : i1
    // CHECK:   }
    pub comb cmp_cast(e: u2, b: u1) bool {
        return e > b;
    }

    // CHECK:   arc.define @ret_val_cast(%arg0: i1, %arg1: i1) -> i2 {
    // CHECK:     %0 = comb.and %arg0, %arg1 : i1
    // CHECK:     %1 = arith.extsi %0 : i1 to i2
    // CHECK:     arc.output %1 : i2
    // CHECK:   }
    pub comb ret_val_cast(f: i1, g: i1) i2 {
        return f & g;
    }
};
// CHECK:   hw.module @Top(in %a : i1, in %b : i1, in %c : i1, in %d : i1, in %e : i2, in %f : i1, in %g : i1, out bitwise : i1, out arith : i1, out cmp : i1, out cmp_bool : i1, out bitsize_cast : i2, out arith_cast : i2, out cmp_cast : i1, out ret_val_cast : i2) {
// CHECK:     %0 = arc.call @bitwise(%a, %b) : (i1, i1) -> i1
// CHECK:     %1 = arc.call @arith(%a, %b) : (i1, i1) -> i1
// CHECK:     %2 = arc.call @cmp(%a, %b) : (i1, i1) -> i1
// CHECK:     %3 = arc.call @cmp_bool(%c, %d) : (i1, i1) -> i1
// CHECK:     %4 = arc.call @bitsize_cast(%e, %b) : (i2, i1) -> i2
// CHECK:     %5 = arc.call @arith_cast(%e, %b) : (i2, i1) -> i2
// CHECK:     %6 = arc.call @cmp_cast(%e, %b) : (i2, i1) -> i1
// CHECK:     %7 = arc.call @ret_val_cast(%f, %g) : (i1, i1) -> i2
// CHECK:     hw.output %0, %1, %2, %3, %4, %5, %6, %7 : i1, i1, i1, i1, i2, i2, i1, i2
// CHECK:   }

// CHECK: }
