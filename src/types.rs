use crate::{
    circt::{
        self,
        sys::{
            MlirAttribute as CirctMlirAttribute, MlirContext as CirctMlirContext,
            MlirType as CirctMlirType,
        },
    },
    utils::RRC,
};
use melior::{
    ir::{
        attribute::{AttributeLike, StringAttribute as MlirStringAttribute},
        r#type::{IntegerType as MlirIntegerType, TypeLike},
        Type as MlirType,
    },
    Context,
};
use std::{fmt::Display, hash::Hash};
use symbol_table::GlobalSymbol;

// TODO: Remove clone in favor of RRC
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Bool,
    Void,
    Type,
    ComptimeInt,
    Null,
    Undefined,
    InferredAllocMut,
    InferredAllocConst,
    Array,
    Tuple,
    Pointer,
    IntSigned { width: u16 },
    IntUnsigned { width: u16 },
    // TODO: Struct and module layouts
    Struct,
    Module(ModuleInfo),
    TypeType,
}

impl<'ctx> Type {
    pub fn to_mlir_type(&self, ctx: &'ctx Context) -> MlirType<'ctx> {
        match self {
            Type::Bool => MlirIntegerType::new(ctx, 1).into(),
            Type::Void => MlirIntegerType::new(ctx, 0).into(),
            Type::IntSigned { width } | Type::IntUnsigned { width } => {
                MlirIntegerType::new(ctx, (*width).into()).into()
            }
            Type::Module(mod_info) => {
                let mut ports = Vec::new();
                let mut in_ports = Vec::new();
                for in_port in &mod_info.ins {
                    let mlir_type = in_port.ty.borrow().to_mlir_type(ctx);
                    let hw_module_port = circt::sys::HWModulePort {
                        name: CirctMlirAttribute {
                            ptr: MlirStringAttribute::new(ctx, in_port.name.as_str())
                                .to_raw()
                                .ptr,
                        },
                        dir: circt::sys::HWModulePortDirection_Input,
                        type_: CirctMlirType {
                            ptr: mlir_type.to_raw().ptr,
                        },
                    };
                    ports.push(hw_module_port);
                    in_ports.push(mlir_type);
                }
                let mut out_ports = Vec::new();
                for out_port in &mod_info.outs {
                    let mlir_type = out_port.ty.borrow().to_mlir_type(ctx);
                    let hw_module_port = circt::sys::HWModulePort {
                        name: CirctMlirAttribute {
                            ptr: MlirStringAttribute::new(ctx, out_port.name.as_str())
                                .to_raw()
                                .ptr,
                        },
                        dir: circt::sys::HWModulePortDirection_Output,
                        type_: CirctMlirType {
                            ptr: mlir_type.to_raw().ptr,
                        },
                    };
                    ports.push(hw_module_port);
                    out_ports.push(mlir_type);
                }
                let module_type = unsafe {
                    let raw_type = circt::sys::hwModuleTypeGet(
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
            _ => unimplemented!(),
        }
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            Type::Bool => f.write_str("bool"),
            Type::Void => f.write_str("void"),
            Type::Type => f.write_str("type"),
            Type::ComptimeInt => f.write_str("comptime_int"),
            Type::Null => f.write_str("null"),
            Type::Undefined => f.write_str("undefined"),
            Type::Array => unimplemented!(),
            Type::Tuple => unimplemented!(),
            Type::Pointer => unimplemented!(),
            Type::IntSigned { width } => f.write_fmt(format_args!("i{}", width)),
            Type::IntUnsigned { width } => f.write_fmt(format_args!("u{}", width)),
            Type::Struct => unimplemented!(),
            Type::Module(_) => unimplemented!(),
            _ => unreachable!(),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct NamedType {
    pub name: GlobalSymbol,
    pub ty: RRC<Type>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ModuleInfo {
    pub ins: Vec<NamedType>,
    pub outs: Vec<NamedType>,
}
