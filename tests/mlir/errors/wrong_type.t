// RUN: @tara @file --dump-mlir --exit-early

// CHECK: [0m[1m[38;5;9merror[0m[1m: Illegal cast from comptime_int to type[0m
// CHECK:   [0m[34m┌─[0m /Users/devin/Repos/tara/tests/mlir/errors/wrong_type.t:8:18
// CHECK:   [0m[34m│[0m
// CHECK: [0m[34m8[0m [0m[34m│[0m const A_: type = [0m[31m1[0m;
// CHECK:   [0m[34m│[0m                  [0m[31m^[0m
const A_: type = 1;

const B = struct {
    a: u1,

    // CHECK: [0m[1m[38;5;9merror[0m[1m: Illegal cast from bool to u1[0m
    // CHECK:    [0m[34m┌─[0m /Users/devin/Repos/tara/tests/mlir/errors/wrong_type.t:18:23
    // CHECK:    [0m[34m│[0m
    // CHECK: [0m[34m18[0m [0m[34m│[0m     const C = B{ .a = [0m[31mfalse[0m };
    // CHECK:    [0m[34m│[0m                       [0m[31m^^^^^[0m
    const C = B{ .a = false };

    // CHECK: [0m[1m[38;5;9merror[0m[1m: Illegal cast from root.B to u8[0m
    // CHECK:    [0m[34m┌─[0m /Users/devin/Repos/tara/tests/mlir/errors/wrong_type.t:25:20
    // CHECK:    [0m[34m│[0m
    // CHECK: [0m[34m25[0m [0m[34m│[0m     const D_: u8 = [0m[31mB{ .a = 0 }[0m;
    // CHECK:    [0m[34m│[0m                    [0m[31m^^^^^^^^^^^[0m
    const D_: u8 = B{ .a = 0 };
};

const E = module {
    pub comb a(x0: u8) u8 {
        return x0;
    }

    // CHECK: [0m[1m[38;5;9merror[0m[1m: Comb params have different types![0m
    // CHECK:    [0m[34m┌─[0m /Users/devin/Repos/tara/tests/mlir/errors/wrong_type.t:40:5
    // CHECK:    [0m[34m│[0m  
    // CHECK: [0m[34m40[0m [0m[34m│[0m [0m[31m╭[0m     [0m[31mpub comb b(x0: u4) u8 {[0m
    // CHECK: [0m[34m41[0m [0m[34m│[0m [0m[31m│[0m [0m[31m        return x0;[0m
    // CHECK: [0m[34m42[0m [0m[34m│[0m [0m[31m│[0m [0m[31m    }[0m
    // CHECK:    [0m[34m│[0m [0m[31m╰[0m[0m[31m─────^[0m
    pub comb b(x0: u4) u8 {
        return x0;
    }
};
