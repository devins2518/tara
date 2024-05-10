// RUN: @tara @file --dump-mlir --exit-early 2>&1
// CHECK: module {

// CHECK:   func.func @bitwise(%arg0: i1, %arg1: i1) -> i1 {
// CHECK:     %0 = arith.andi %arg0, %arg1 : i1
// CHECK:     return %0 : i1
// CHECK:   }
pub fn bitwise(a: u1, b: u1) u1 {
    return a & b;
}

// CHECK:   func.func @arith(%arg0: i1, %arg1: i1) -> i1 {
// CHECK:     %0 = arith.addi %arg0, %arg1 : i1
// CHECK:     return %0 : i1
// CHECK:   }
pub fn arith(a: u1, b: u1) u1 {
    return a + b;
}

// CHECK:   func.func @cmp(%arg0: i1, %arg1: i1) -> i1 {
// CHECK:     %0 = arith.cmpi ugt, %arg0, %arg1 : i1
// CHECK:     return %0 : i1
// CHECK:   }
pub fn cmp(a: u1, b: u1) bool {
    return a > b;
}

// CHECK:   func.func @cmp_bool(%arg0: i1, %arg1: i1) -> i1 {
// CHECK:     %0 = arith.andi %arg0, %arg1 : i1
// CHECK:     return %0 : i1
// CHECK:   }
pub fn cmp_bool(a: bool, b: bool) bool {
    return a and b;
}

// CHECK:   func.func @bitsize_cast(%arg0: i2, %arg1: i1) -> i2 {
// CHECK:     %0 = arith.extui %arg1 : i1 to i2
// CHECK:     %1 = arith.andi %arg0, %0 : i2
// CHECK:     return %1 : i2
// CHECK:   }
pub fn bitsize_cast(a: u2, b: u1) u2 {
    return a & b;
}

// CHECK:   func.func @arith_cast(%arg0: i2, %arg1: i1) -> i2 {
// CHECK:     %0 = arith.extui %arg1 : i1 to i2
// CHECK:     %1 = arith.addi %arg0, %0 : i2
// CHECK:     return %1 : i2
// CHECK:   }
pub fn arith_cast(a: u2, b: u1) u2 {
    return a + b;
}

// CHECK:   func.func @cmp_cast(%arg0: i2, %arg1: i1) -> i1 {
// CHECK:     %0 = arith.extui %arg1 : i1 to i2
// CHECK:     %1 = arith.cmpi ugt, %arg0, %0 : i2
// CHECK:     return %1 : i1
// CHECK:   }
pub fn cmp_cast(a: u2, b: u1) bool {
    return a > b;
}

// CHECK: func.func @ret_val_cast(%arg0: i1, %arg1: i1) -> i2 {
// CHECK:   %0 = arith.andi %arg0, %arg1 : i1
// CHECK:   %1 = arith.extsi %0 : i1 to i2
// CHECK:   return %1 : i2
// CHECK: }
pub fn ret_val_cast(a: i1, b: i1) i2 {
    return a & b;
}

pub fn test_void_ret(hi: u1, there: i2) void {}

// CHECK: }
