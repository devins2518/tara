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

    // CHECK: error: Block unexpectedly returns void
    // CHECK:    │
    // CHECK: 28 │     fn a() u1 {}
    // CHECK:    │               ^^
    fn a() u1 {}

    // CHECK: error: Block unexpectedly returns void
    // CHECK:    │
    // CHECK: 37 │       fn b() u1 {
    // CHECK:    │ ╭───────────────^
    // CHECK: 38 │ │         false;
    // CHECK: 39 │ │     }
    // CHECK:    │ ╰─────^
    fn b() u1 {
        false;
    }


    // CHECK: error: Illegal cast from comptime_int to void
    // CHECK:    │
    // CHECK: 47 │         return 1;
    // CHECK:    │         ^^^^^^^^
    fn c() void {
        return 1;
    }
};

const E = module {
    // CHECK: error: Block unexpectedly returns void
    // CHECK:    │
    // CHECK: 56 │     comb a() u1 {}
    // CHECK:    │                 ^^
    comb a() u1 {}

    // CHECK: error: Block unexpectedly returns void
    // CHECK:    │
    // CHECK: 65 │       comb b() u1 {
    // CHECK:    │ ╭─────────────────^
    // CHECK: 66 │ │         false;
    // CHECK: 67 │ │     }
    // CHECK:    │ ╰─────^
    comb b() u1 {
        false;
    }

    // CHECK: error: Illegal cast from comptime_int to bool
    // CHECK:    │
    // CHECK: 74 │         return 1;
    // CHECK:    │         ^^^^^^^^
    comb c() bool {
        return 1;
    }
};
