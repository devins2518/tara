// RUN: @tara @file --dump-mlir --exit-early 2>&1

// CHECK: module {
pub const Struct = struct {
    const Butterfly = struct { y0: u8, y1: u8 };

    // CHECK:  func.func @root.Struct.top(%arg0: i8, %arg1: i8) -> !llvm.struct<(i8, i8)> {
    pub fn top(x0: u8, x1: u8) Butterfly {
        // CHECK:    %0 = arith.addi %arg0, %arg1 : i8
        // CHECK:    %1 = arith.subi %arg0, %arg1 : i8
        // CHECK:    %2 = llvm.mlir.undef : !llvm.struct<(i8, i8)>
        // CHECK:    %3 = llvm.insertvalue %0, %2[0] : !llvm.struct<(i8, i8)>
        // CHECK:    %4 = llvm.insertvalue %1, %3[1] : !llvm.struct<(i8, i8)>
        // CHECK:    return %4 : !llvm.struct<(i8, i8)>
        return Butterfly{ .y0 = x0 + x1, .y1 = x0 - x1 };
    }
    // CHECK:  }
};

pub const Top = module {
    const Butterfly = struct { y0: u8, y1: u8 };

    // CHECK:   hw.module @root.Top.top(in %x0 : i8, in %x1 : i8, out root.Top.top : !hw.struct<y0: i8, y1: i8>) {
    // CHECK:     %0 = comb.add bin %x0, %x1 : i8
    // CHECK:     %1 = comb.sub bin %x0, %x1 : i8
    // CHECK:     %2 = hw.struct_create (%0, %1) : !hw.struct<y0: i8, y1: i8>
    // CHECK:     hw.output %2 : !hw.struct<y0: i8, y1: i8>
    // CHECK:   }
    pub comb top(x0: u8, x1: u8) Butterfly {
        return Butterfly{ .y0 = x0 + x1, .y1 = x0 - x1 };
    }
};
// CHECK: }
