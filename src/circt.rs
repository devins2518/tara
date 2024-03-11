melior_macro::dialect! {
    name: "arc",
    table_gen: r#"include "circt/Dialect/Arc/Arc.td""#
}
melior_macro::dialect! {
    name: "calyx",
    table_gen: r#"include "circt/Dialect/Calyx/Calyx.td""#
}
melior_macro::dialect! {
    name: "comb",
    table_gen: r#"include "circt/Dialect/Comb/Comb.td""#
}
melior_macro::dialect! {
    name: "dc",
    table_gen: r#"include "circt/Dialect/DC/DC.td""#
}
melior_macro::dialect! {
    name: "esi",
    table_gen: r#"include "circt/Dialect/ESI/ESI.td""#
}
melior_macro::dialect! {
    name: "emit",
    table_gen: r#"include "circt/Dialect/Emit/Emit.td""#
}
melior_macro::dialect! {
    name: "firrtl",
    include_dirs: ["install/include/circt/Dialect/FIRRTL"],
    table_gen: r#"include "circt/Dialect/FIRRTL/FIRRTL.td""#
}
melior_macro::dialect! {
    name: "fsm",
    table_gen: r#"include "circt/Dialect/FSM/FSM.td""#
}
melior_macro::dialect! {
    name: "hw",
    table_gen: r#"include "circt/Dialect/HW/HW.td""#
}
melior_macro::dialect! {
    name: "hwarith",
    table_gen: r#"include "circt/Dialect/HWArith/HWArith.td""#
}
melior_macro::dialect! {
    name: "handshake",
    table_gen: r#"include "circt/Dialect/Handshake/Handshake.td""#
}
melior_macro::dialect! {
    name: "ibis",
    table_gen: r#"include "circt/Dialect/Ibis/Ibis.td""#
}
melior_macro::dialect! {
    name: "interop",
    table_gen: r#"include "circt/Dialect/Interop/Interop.td""#
}
melior_macro::dialect! {
    name: "llhd",
    include_dirs: ["install/include/circt/Dialect/LLHD/IR"],
    table_gen: r#"include "circt/Dialect/LLHD/IR/LLHD.td""#
}
melior_macro::dialect! {
    name: "ltl",
    table_gen: r#"include "circt/Dialect/LTL/LTL.td""#
}
melior_macro::dialect! {
    name: "loopschedule",
    table_gen: r#"include "circt/Dialect/LoopSchedule/LoopSchedule.td""#
}
melior_macro::dialect! {
    name: "msft",
    table_gen: r#"include "circt/Dialect/MSFT/MSFT.td""#
}
melior_macro::dialect! {
    name: "moore",
    table_gen: r#"include "circt/Dialect/Moore/Moore.td""#
}
melior_macro::dialect! {
    name: "om",
    table_gen: r#"include "circt/Dialect/OM/OM.td""#
}
melior_macro::dialect! {
    name: "pipeline",
    table_gen: r#"include "circt/Dialect/Pipeline/Pipeline.td""#
}
melior_macro::dialect! {
    name: "ssp",
    include_dirs: ["install/include/circt/Dialect/SSP"],
    table_gen: r#"include "circt/Dialect/SSP/SSP.td""#
}
melior_macro::dialect! {
    name: "sv",
    table_gen: r#"include "circt/Dialect/SV/SV.td""#
}
// melior_macro::dialect! {
//     name: "seq",
//     table_gen: r#"include "circt/Dialect/Seq/Seq.td""#
// }
melior_macro::dialect! {
    name: "sim",
    table_gen: r#"include "circt/Dialect/Sim/Sim.td""#
}
melior_macro::dialect! {
    name: "systemc",
    table_gen: r#"include "circt/Dialect/SystemC/SystemC.td""#
}
melior_macro::dialect! {
    name: "verif",
    table_gen: r#"include "circt/Dialect/Verif/Verif.td""#
}
