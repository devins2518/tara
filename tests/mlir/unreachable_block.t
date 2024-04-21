// RUN: @tara @file --dump-mlir --exit-early 2>&1 || (($?==1 ? 1 : 0))
// CHECK: Error: Unreachable statement
fn test() void {
    return;
    return;
}
