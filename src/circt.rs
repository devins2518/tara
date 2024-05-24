pub mod sys;

use crate::{ast_codegen::SurroundingContext, types::Type as TaraType};
use melior::{
    dialect::{DialectHandle, DialectRegistry},
    ir::{
        attribute::{AttributeLike, StringAttribute as MlirStringAttribute},
        r#type::TypeLike,
        Type as MlirType,
    },
    Context,
};
use mlir_sys::MlirDialectHandle;

use crate::circt::sys::{
    HWModulePort, MlirAttribute as CirctMlirAttribute, MlirContext as CirctMlirContext,
    MlirType as CirctMlirType,
};

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

extern "C" {
    fn mlirGetDialectHandle__arc__() -> MlirDialectHandle;
    fn mlirGetDialectHandle__calyx__() -> MlirDialectHandle;
    fn mlirGetDialectHandle__comb__() -> MlirDialectHandle;
    fn mlirGetDialectHandle__dc__() -> MlirDialectHandle;
    fn mlirGetDialectHandle__esi__() -> MlirDialectHandle;
    fn mlirGetDialectHandle__emit__() -> MlirDialectHandle;
    fn mlirGetDialectHandle__firrtl__() -> MlirDialectHandle;
    fn mlirGetDialectHandle__fsm__() -> MlirDialectHandle;
    fn mlirGetDialectHandle__hw__() -> MlirDialectHandle;
    fn mlirGetDialectHandle__hwarith__() -> MlirDialectHandle;
    fn mlirGetDialectHandle__handshake__() -> MlirDialectHandle;
    fn mlirGetDialectHandle__ibis__() -> MlirDialectHandle;
    fn mlirGetDialectHandle__interop__() -> MlirDialectHandle;
    fn mlirGetDialectHandle__llhd__() -> MlirDialectHandle;
    fn mlirGetDialectHandle__ltl__() -> MlirDialectHandle;
    fn mlirGetDialectHandle__loopschedule__() -> MlirDialectHandle;
    fn mlirGetDialectHandle__msft__() -> MlirDialectHandle;
    fn mlirGetDialectHandle__moore__() -> MlirDialectHandle;
    fn mlirGetDialectHandle__om__() -> MlirDialectHandle;
    fn mlirGetDialectHandle__pipeline__() -> MlirDialectHandle;
    fn mlirGetDialectHandle__ssp__() -> MlirDialectHandle;
    fn mlirGetDialectHandle__sv__() -> MlirDialectHandle;
    // fn mlirGetDialectHandle__seq__() -> MlirDialectHandle;
    fn mlirGetDialectHandle__sim__() -> MlirDialectHandle;
    fn mlirGetDialectHandle__systemc__() -> MlirDialectHandle;
    fn mlirGetDialectHandle__verif__() -> MlirDialectHandle;
}

// TODO: Link against CIRCT libraries
pub fn register_all_dialects(ctx: &Context) {
    let registry = DialectRegistry::new();
    unsafe {
        DialectHandle::from_raw(mlirGetDialectHandle__arc__()).insert_dialect(&registry);
        // DialectHandle::from_raw(mlirGetDialectHandle__calyx__()).insert_dialect(&registry);
        DialectHandle::from_raw(mlirGetDialectHandle__comb__()).insert_dialect(&registry);
        // DialectHandle::from_raw(mlirGetDialectHandle__dc__()).insert_dialect(&registry);
        DialectHandle::from_raw(mlirGetDialectHandle__esi__()).insert_dialect(&registry);
        DialectHandle::from_raw(mlirGetDialectHandle__emit__()).insert_dialect(&registry);
        DialectHandle::from_raw(mlirGetDialectHandle__firrtl__()).insert_dialect(&registry);
        DialectHandle::from_raw(mlirGetDialectHandle__fsm__()).insert_dialect(&registry);
        DialectHandle::from_raw(mlirGetDialectHandle__hw__()).insert_dialect(&registry);
        DialectHandle::from_raw(mlirGetDialectHandle__hwarith__()).insert_dialect(&registry);
        DialectHandle::from_raw(mlirGetDialectHandle__handshake__()).insert_dialect(&registry);
        // DialectHandle::from_raw(mlirGetDialectHandle__ibis__()).insert_dialect(&registry);
        // DialectHandle::from_raw(mlirGetDialectHandle__interop__()).insert_dialect(&registry);
        DialectHandle::from_raw(mlirGetDialectHandle__llhd__()).insert_dialect(&registry);
        DialectHandle::from_raw(mlirGetDialectHandle__ltl__()).insert_dialect(&registry);
        // DialectHandle::from_raw(mlirGetDialectHandle__loopschedule__()).insert_dialect(&registry);
        DialectHandle::from_raw(mlirGetDialectHandle__msft__()).insert_dialect(&registry);
        DialectHandle::from_raw(mlirGetDialectHandle__moore__()).insert_dialect(&registry);
        DialectHandle::from_raw(mlirGetDialectHandle__om__()).insert_dialect(&registry);
        // DialectHandle::from_raw(mlirGetDialectHandle__pipeline__()).insert_dialect(&registry);
        // DialectHandle::from_raw(mlirGetDialectHandle__ssp__()).insert_dialect(&registry);
        DialectHandle::from_raw(mlirGetDialectHandle__sv__()).insert_dialect(&registry);
        // DialectHandle::from_raw(mlirGetDialectHandle__seq__()).insert_dialect(&registry);
        // DialectHandle::from_raw(mlirGetDialectHandle__sim__()).insert_dialect(&registry);
        // DialectHandle::from_raw(mlirGetDialectHandle__systemc__()).insert_dialect(&registry);
        DialectHandle::from_raw(mlirGetDialectHandle__verif__()).insert_dialect(&registry);
    }
    ctx.append_dialect_registry(&registry);
}

#[repr(u32)]
pub enum ModulePortDirection {
    In = sys::HWModulePortDirection_Input,
    Out = sys::HWModulePortDirection_Output,
    InOut = sys::HWModulePortDirection_InOut,
}

// HACK: Ugly but necessary since circt::sys types aren't the same as melior::sys
pub fn get_module_type<'a>(
    ctx: &Context,
    name: &str,
    inputs: &[(&str, MlirType<'a>)],
    output: TaraType,
) -> MlirType<'static> {
    let mut ports = Vec::new();
    for (name, port_type) in inputs {
        let hw_module_port = HWModulePort {
            name: CirctMlirAttribute {
                ptr: MlirStringAttribute::new(ctx, name).to_raw().ptr,
            },
            dir: sys::HWModulePortDirection_Input,
            type_: CirctMlirType {
                ptr: port_type.to_raw().ptr,
            },
        };
        ports.push(hw_module_port);
    }
    if output != TaraType::Void {
        let hw_module_port = HWModulePort {
            name: CirctMlirAttribute {
                ptr: MlirStringAttribute::new(ctx, name).to_raw().ptr,
            },
            dir: sys::HWModulePortDirection_Output,
            type_: CirctMlirType {
                ptr: output
                    .to_mlir_type(ctx, SurroundingContext::Hw)
                    .to_raw()
                    .ptr,
            },
        };
        ports.push(hw_module_port);
    }
    let module_type = unsafe {
        let raw_type = sys::hwModuleTypeGet(
            CirctMlirContext {
                ptr: ctx.to_raw().ptr,
            },
            ports.len() as isize,
            ports.as_slice().as_ptr(),
        );
        MlirType::from_raw(mlir_sys::MlirType { ptr: raw_type.ptr })
    };
    module_type
}

// Predicate for `cmp.icmp` operation.
#[repr(u64)]
pub enum CombICmpPredicate {
    Eq = 0,
    Ne = 1,
    Slt = 2,
    Sle = 3,
    Sgt = 4,
    Sge = 5,
    Ult = 6,
    Ule = 7,
    Ugt = 8,
    Uge = 9,
    Ceq = 10,
    Cne = 11,
    Weq = 12,
    Wne = 13,
}
