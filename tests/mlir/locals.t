// RUN: @tara @file --dump-mlir --exit-early 2>&1

// CHECK: module {

pub const Struct = struct {
    const Butterfly = struct { y0: u8, y1: u8 };

    // CHECK:   func.func @root.Struct.top(%arg0: i8, %arg1: i8) -> !llvm.struct<(i8, i8)> {
    // CHECK:     %0 = arith.addi %arg0, %arg1 : i8
    // CHECK:     %1 = arith.subi %arg0, %arg1 : i8
    // CHECK:     %2 = llvm.mlir.undef : !llvm.struct<(i8, i8)>
    // CHECK:     %3 = llvm.insertvalue %0, %2[0] : !llvm.struct<(i8, i8)>
    // CHECK:     %4 = llvm.insertvalue %1, %3[1] : !llvm.struct<(i8, i8)>
    // CHECK:     return %4 : !llvm.struct<(i8, i8)>
    // CHECK:   }
    pub fn top(x0: u8, x1: u8) Butterfly {
        const y0 = x0 + x1;
        const y1 = x0 - x1;
        return Butterfly{ .y0 = y0, .y1 = y1 };
    }

    // CHECK:   func.func @root.Struct.castedTop(%arg0: i4, %arg1: i4) -> !llvm.struct<(i8, i8)> {
    // CHECK:     %0 = arith.addi %arg0, %arg1 : i4
    // CHECK:     %1 = arith.subi %arg0, %arg1 : i4
    // CHECK:     %2 = arith.extui %1 : i4 to i8
    // CHECK:     %3 = arith.extui %0 : i4 to i8
    // CHECK:     %4 = llvm.mlir.undef : !llvm.struct<(i8, i8)>
    // CHECK:     %5 = llvm.insertvalue %3, %4[0] : !llvm.struct<(i8, i8)>
    // CHECK:     %6 = llvm.insertvalue %2, %5[1] : !llvm.struct<(i8, i8)>
    // CHECK:     return %6 : !llvm.struct<(i8, i8)>
    // CHECK:   }
    pub fn castedTop(x0: u4, x1: u4) Butterfly {
        const y0 = x0 + x1;
        const y1: u8 = x0 - x1;
        return Butterfly{ .y0 = y0, .y1 = y1 };
    }
};

pub const Top = module {
    const Butterfly = struct { y0: u8, y1: u8 };

    // CHECK:   arc.define @root.Top.top(%arg0: i8, %arg1: i8) -> !hw.struct<y0: i8, y1: i8> {
    // CHECK:     %0 = comb.add bin %arg0, %arg1 : i8
    // CHECK:     %1 = comb.sub bin %arg0, %arg1 : i8
    // CHECK:     %2 = hw.struct_create (%0, %1) : !hw.struct<y0: i8, y1: i8>
    // CHECK:     arc.output %2 : !hw.struct<y0: i8, y1: i8>
    // CHECK:   }
    pub comb top(x0: u8, x1: u8) Butterfly {
        const y0 = x0 + x1;
        const y1 = x0 - x1;
        return Butterfly{ .y0 = y0, .y1 = y1 };
    }

    // CHECK:   arc.define @root.Top.castedTop(%arg0: i4, %arg1: i4) -> !hw.struct<y0: i8, y1: i8> {
    // CHECK:     %0 = comb.add bin %arg0, %arg1 : i4
    // CHECK:     %1 = comb.sub bin %arg0, %arg1 : i4
    // CHECK:     %2 = arith.extui %1 : i4 to i8
    // CHECK:     %3 = arith.extui %0 : i4 to i8
    // CHECK:     %4 = hw.struct_create (%3, %2) : !hw.struct<y0: i8, y1: i8>
    // CHECK:     arc.output %4 : !hw.struct<y0: i8, y1: i8>
    // CHECK:   }
    pub comb castedTop(casted_x0: u4, casted_x1: u4) Butterfly {
        const y0 = casted_x0 + casted_x1;
        const y1: u8 = casted_x0 - casted_x1;
        return Butterfly{ .y0 = y0, .y1 = y1 };
    }
};
// CHECK:   hw.module @root.Top(in %x0 : i8, in %x1 : i8, in %casted_x0 : i4, in %casted_x1 : i4, out top : !hw.struct<y0: i8, y1: i8>, out castedTop : !hw.struct<y0: i8, y1: i8>) {
// CHECK:     %0 = arc.call @root.Top.top(%x0, %x1) : (i8, i8) -> !hw.struct<y0: i8, y1: i8>
// CHECK:     %1 = arc.call @root.Top.castedTop(%casted_x0, %casted_x1) : (i4, i4) -> !hw.struct<y0: i8, y1: i8>
// CHECK:     hw.output %0, %1 : !hw.struct<y0: i8, y1: i8>, !hw.struct<y0: i8, y1: i8>
// CHECK:   }

// CHECK: }
