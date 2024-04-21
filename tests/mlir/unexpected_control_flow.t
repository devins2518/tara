// RUN: @tara @file --dump-mlir --exit-early 2>&1 || (($?==1 ? 1 : 0))
// CHECK: Error: Expected reachable value, control flow unexpectedly diverted
fn test(a: u1) u1 {
    return return a;
}
