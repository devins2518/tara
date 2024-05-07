// RUN: @tara @file --dump-mlir --exit-early
// CHECK:[0m[1m[38;5;9merror[0m[1m: Expected bool type, found u8[0m
// CHECK:  [0m[34m┌─[0m /Users/devin/Repos/tara/tests/mlir/type_check.t:10:12
// CHECK:  [0m[34m│[0m
// CHECK:[0m[34m10[0m [0m[34m│[0m     return [0m[31ma[0m and b;
// CHECK:  [0m[34m│[0m            [0m[31m^[0m


fn bad_op_types(a: u8, b: u8) bool {
    return a and b;
}
