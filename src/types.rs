use crate::{
    ast_codegen::SurroundingContext,
    circt::{
        self,
        sys::{
            MlirAttribute as CirctMlirAttribute, MlirContext as CirctMlirContext,
            MlirIdentifier as CirctMlirIdentifier, MlirType as CirctMlirType,
        },
    },
    module::{
        comb::Comb, decls::Decl, function::Function, namespace::Namespace, structs::Struct,
        tmodule::TModule,
    },
    utils::RRC,
    values::Value as TaraValue,
};
use melior::{
    dialect::llvm::r#type::r#struct as llvm_struct,
    ir::{
        attribute::{AttributeLike, StringAttribute as MlirStringAttribute},
        r#type::{IntegerType as MlirIntegerType, TypeLike},
        Identifier as MlirIdentifier, Type as MlirType,
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
    Struct(RRC<Struct>),
    Module(RRC<TModule>),
    Function(RRC<Function>),
    Comb(RRC<Comb>),
    NoReturn,
}

impl<'ctx> Type {
    // TODO: switch towards a to_mlir_type, to_hw_mlir_type, and to_sw_mlir_type
    pub fn to_mlir_type(&self, ctx: &'ctx Context, surr_ctx: SurroundingContext) -> MlirType<'ctx> {
        match (surr_ctx, self) {
            (_, Type::Bool) => MlirIntegerType::new(ctx, 1).into(),
            (_, Type::Void) => MlirIntegerType::new(ctx, 0).into(),
            (_, Type::IntSigned { width } | Type::IntUnsigned { width }) => {
                MlirIntegerType::new(ctx, (*width).into()).into()
            }
            (SurroundingContext::Hw, Type::Module(_)) => self.to_hw_mlir_type(ctx),
            (SurroundingContext::Hw, Type::Struct(_)) => self.to_hw_mlir_type(ctx),
            (SurroundingContext::Sw, Type::Struct(_)) => self.to_sw_mlir_type(ctx),
            _ => unimplemented!(),
        }
    }

    fn to_hw_mlir_type(&self, ctx: &'ctx Context) -> MlirType<'ctx> {
        match self {
            Type::Module(mod_info_rrc) => {
                let mod_info = mod_info_rrc.borrow();
                let mut ports = Vec::new();
                for in_port in &mod_info.ins {
                    let mlir_type = in_port.2.to_mlir_type(ctx, SurroundingContext::Hw);
                    let hw_module_port = circt::sys::HWModulePort {
                        name: CirctMlirAttribute {
                            ptr: MlirStringAttribute::new(ctx, in_port.0.as_str())
                                .to_raw()
                                .ptr,
                        },
                        dir: circt::sys::HWModulePortDirection_Input,
                        type_: CirctMlirType {
                            ptr: mlir_type.to_raw().ptr,
                        },
                    };
                    ports.push(hw_module_port);
                }
                for out_port in &mod_info.outs {
                    let mlir_type = out_port.1.to_mlir_type(ctx, SurroundingContext::Hw);
                    let hw_module_port = circt::sys::HWModulePort {
                        name: CirctMlirAttribute {
                            ptr: MlirStringAttribute::new(ctx, out_port.0.as_str())
                                .to_raw()
                                .ptr,
                        },
                        dir: circt::sys::HWModulePortDirection_Output,
                        type_: CirctMlirType {
                            ptr: mlir_type.to_raw().ptr,
                        },
                    };
                    ports.push(hw_module_port);
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
            Type::Struct(struct_info_rrc) => {
                let struct_info = struct_info_rrc.borrow();
                let mut fields = Vec::new();
                // TODO: handle struct offset
                for field in &struct_info.fields {
                    let mlir_type = field.ty.to_mlir_type(ctx, SurroundingContext::Hw);
                    let field_info = circt::sys::HWStructFieldInfo {
                        name: CirctMlirIdentifier {
                            ptr: MlirIdentifier::new(ctx, field.name.as_str()).to_raw().ptr,
                        },
                        type_: CirctMlirType {
                            ptr: mlir_type.to_raw().ptr,
                        },
                    };
                    fields.push(field_info)
                }
                let struct_type = unsafe {
                    let raw_type = circt::sys::hwStructTypeGet(
                        CirctMlirContext {
                            ptr: ctx.to_raw().ptr,
                        },
                        fields.len() as isize,
                        fields.as_slice().as_ptr(),
                    );
                    MlirType::from_raw(mlir_sys::MlirType { ptr: raw_type.ptr })
                };
                struct_type
            }
            _ => unimplemented!(),
        }
    }

    fn to_sw_mlir_type(&self, ctx: &'ctx Context) -> MlirType<'ctx> {
        match self {
            Type::Struct(struct_info_rrc) => {
                let struct_info = struct_info_rrc.borrow();
                let mut fields = Vec::new();
                // TODO: handle struct offset
                for field in &struct_info.fields {
                    let mlir_type = field.ty.to_mlir_type(ctx, SurroundingContext::Sw);
                    fields.push(mlir_type)
                }
                llvm_struct(ctx, &fields, false)
            }
            _ => unimplemented!(),
        }
    }

    pub fn to_value(&self) -> TaraValue {
        let rrc = RRC::new(self.clone());
        TaraValue::Type(rrc)
    }

    pub fn has_namespace(&self) -> bool {
        match self {
            Type::Struct(_) | Type::Module(_) => true,
            _ => false,
        }
    }

    pub fn namespace(&self) -> RRC<Namespace> {
        match self {
            Type::Struct(s) => s.borrow().namespace.clone(),
            Type::Module(_) => unimplemented!(),
            _ => unreachable!(),
        }
    }

    pub fn decl(&self) -> RRC<Decl> {
        match self {
            Type::Struct(s) => s.borrow().decl.clone(),
            Type::Module(m) => m.borrow().decl.clone(),
            _ => unreachable!(),
        }
    }

    pub fn to_struct(&self) -> RRC<Struct> {
        match self {
            Type::Struct(s) => s.clone(),
            _ => unreachable!(),
        }
    }

    pub fn module(&self) -> RRC<TModule> {
        match self {
            Type::Module(m) => m.clone(),
            _ => unreachable!(),
        }
    }

    pub fn is_noreturn(&self) -> bool {
        match self {
            Type::NoReturn => true,
            _ => false,
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
            Type::Struct(struct_rrc) => f.write_fmt(format_args!(
                "{}",
                &struct_rrc.borrow().decl().borrow().name
            )),
            Type::Module(module_rrc) => f.write_fmt(format_args!(
                "{}",
                &module_rrc.borrow().decl().borrow().name
            )),
            _ => unreachable!(),
        }
    }
}

// TODO: replace with hashmap
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct NamedType {
    pub name: GlobalSymbol,
    pub ty: Type,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ModuleInfo {
    pub ins: Vec<NamedType>,
    pub outs: Vec<NamedType>,
}
