// RUN: @tara @file --dump-mlir --exit-early 2>&1
// CHECK: module {

const A = struct {
    const C = bool;
    const D = C;
    a: u1,

    const E = A{ .a = 1 };
    
    // CHECK:   func.func @root.A.F(%arg0: !llvm.struct<(i1)>) {
    // CHECK:     return
    // CHECK:   }
    pub fn F(a: A) void {}

    // CHECK:   func.func @root.A.G(%arg0: i1) -> i1 {
    // CHECK:     return %arg0 : i1
    // CHECK:   }
    pub fn G(a: C) D {
        return a;
    }
};

const B = module {
    const C = bool;
    const D = C;

    // CHECK:   arc.define @root.B.F(%arg0: i1) -> i1 {
    // CHECK:     arc.output %arg0 : i1
    // CHECK:   }
    pub comb F(a: C) D {
        return a;
    }
};
// CHECK:   hw.module @root.B(in %a : i1, out F : i1) {
// CHECK:     %0 = arc.call @root.B.F(%a) : (i1) -> i1
// CHECK:     hw.output %0 : i1
// CHECK:   }

// CHECK: }
