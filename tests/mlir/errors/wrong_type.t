// RUN: TERM=dumb @tara @file --dump-mlir --exit-early

// CHECK: error: Illegal cast from comptime_int to type
// CHECK:   │
// CHECK: 7 │ const A_: type = 1;
// CHECK:   │                  ^
const A_: type = 1;

const B = struct {
    a: u1,

    // CHECK: error: Illegal cast from bool to u1
    // CHECK:    │
    // CHECK: 16 │     const C = B{ .a = false };
    // CHECK:    │                       ^^^^^
    const C = B{ .a = false };

    // CHECK: error: Illegal cast from root.B to u8
    // CHECK:    │
    // CHECK: 22 │     const D_: u8 = B{ .a = 0 };
    // CHECK:    │                    ^^^^^^^^^^^
    const D_: u8 = B{ .a = 0 };

    // CHECK: error: Block has incorrect return type, wanted u1, found void
    // CHECK:    │
    // CHECK: 28 │     fn a() u1 {}
    // CHECK:    │     ^^^^^^^^^^^^
    fn a() u1 {}

    // CHECK: error: Block has incorrect return type, wanted u1, found void
    // CHECK:    │
    // CHECK: 36 │ ╭     fn b() u1 {
    // CHECK: 37 │ │         false;
    // CHECK: 38 │ │     }
    // CHECK:    │ ╰─────^
    fn b() u1 {
        false;
    }


    // CHECK: error: Illegal cast from comptime_int to void
    // CHECK:    │
    // CHECK: 46 │         return 1;
    // CHECK:    │         ^^^^^^^^
    fn c() void {
        return 1;
    }
};

const E = module {
    pub comb a(x0: u8) u8 {
        return x0;
    }

    // CHECK: error: Comb params have different types!
    // CHECK:    │
    // CHECK: 61 │ ╭     pub comb b(x0: u4) u8 {
    // CHECK: 62 │ │         return x0;
    // CHECK: 63 │ │     }
    // CHECK:    │ ╰─────^
    pub comb b(x0: u4) u8 {
        return x0;
    }
};

const F = module {
    // CHECK: error: Block has incorrect return type, wanted u1, found void
    // CHECK:    │
    // CHECK: 71 │     comb c() u1 {}
    // CHECK:    │     ^^^^^^^^^^^^^^
    comb c() u1 {}

    // CHECK: error: Block has incorrect return type, wanted u1, found void
    // CHECK:    │
    // CHECK: 79 │ ╭     comb d() u1 {
    // CHECK: 80 │ │         false;
    // CHECK: 81 │ │     }
    // CHECK:    │ ╰─────^
    comb d() u1 {
        false;
    }

    // CHECK: error: Illegal cast from comptime_int to bool
    // CHECK:    │
    // CHECK: 88 │         return 1;
    // CHECK:    │         ^^^^^^^^
    comb e() bool {
        return 1;
    }
};
