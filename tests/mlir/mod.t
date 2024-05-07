// RUN: @tara @file --dump-mlir --exit-early 2>&1
// CHECK: module {
// CHECK:   hw.module @Mod() {
// CHECK:     hw.output
// CHECK:   }
// CHECK: }

const Mod = module {};
