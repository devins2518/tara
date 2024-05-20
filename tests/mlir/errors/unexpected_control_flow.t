// RUN: @tara @file --dump-mlir --exit-early
// CHECK:[0m[1m[38;5;9merror[0m[1m: Expected reachable value, control flow unexpectedly diverted[0m
// CHECK:  [0m[34m┌─[0m /Users/devin/Repos/tara/tests/mlir/errors/unexpected_control_flow.t:9:12
// CHECK:  [0m[34m│[0m
// CHECK:[0m[34m9[0m [0m[34m│[0m     return [0m[31mreturn a[0m;
// CHECK:  [0m[34m│[0m            [0m[31m^^^^^^^^[0m

fn test(a: u1) u1 {
    return return a;
}
