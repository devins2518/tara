use crate::{
    ast_codegen::SurroundingContext,
    circt::{
        self,
        sys::{
            MlirContext as CirctMlirContext, MlirIdentifier as CirctMlirIdentifier,
            MlirType as CirctMlirType,
        },
    },
    module::{
        comb::Comb, decls::Decl, function::Function, namespace::Namespace, register::Register,
        structs::Struct,
    },
    utils::RRC,
    values::Value as TaraValue,
};
use melior::{
    dialect::llvm::r#type::r#struct as llvm_struct,
    ir::{
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
    // TODO: add clock and reset info to these
    Clock,
    Reset,
    // TODO: Struct and module layouts
    Struct(RRC<Struct>),
    Module(RRC<Struct>),
    Function(RRC<Function>),
    Comb(RRC<Comb>),
    Register(RRC<Register>),
    // TODO: Remove type reference when UTIR is readded
    NoReturn(RRC<Type>),
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
            (SurroundingContext::Hw, Type::Clock) => self.to_hw_mlir_type(ctx),
            (SurroundingContext::Hw, Type::Reset) => self.to_hw_mlir_type(ctx),
            (SurroundingContext::Hw, Type::Module(_)) => self.to_hw_mlir_type(ctx),
            (SurroundingContext::Hw, Type::Struct(_)) => self.to_hw_mlir_type(ctx),
            (SurroundingContext::Sw, Type::Struct(_)) => self.to_sw_mlir_type(ctx),
            _ => unimplemented!(),
        }
    }

    fn to_hw_mlir_type(&self, ctx: &'ctx Context) -> MlirType<'ctx> {
        let raw_ctx = CirctMlirContext {
            ptr: ctx.to_raw().ptr,
        };
        let raw_type = match self {
            Type::Module(struct_info_rrc) | Type::Struct(struct_info_rrc) => {
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
                let raw_type = unsafe {
                    circt::sys::hwStructTypeGet(
                        raw_ctx,
                        fields.len() as isize,
                        fields.as_slice().as_ptr(),
                    )
                };
                raw_type
            }
            Type::Clock => unsafe { circt::sys::seqClockTypeGet(raw_ctx) },
            Type::Reset => return Type::Bool.to_mlir_type(ctx, SurroundingContext::Sw),
            _ => unimplemented!(),
        };
        unsafe { MlirType::from_raw(mlir_sys::MlirType { ptr: raw_type.ptr }) }
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

    pub fn get_bit_size(&self) -> u16 {
        match self {
            Self::Bool => 1,
            Self::Void => 0,
            Self::IntSigned { width } | Self::IntUnsigned { width } => *width,
            Self::Reset => 1,

            Self::Undefined
            | Self::InferredAllocMut
            | Self::InferredAllocConst
            | Self::Array
            | Self::Tuple
            | Self::Pointer
            | Self::Null
            | Self::Struct(_)
            | Self::Module(_) => unimplemented!(),

            Self::Type
            | Self::ComptimeInt
            | Self::Clock
            | Self::Function(_)
            | Self::Comb(_)
            | Self::Register(_)
            | Self::NoReturn(_) => unreachable!(),
        }
    }

    pub fn to_value(&self) -> TaraValue {
        TaraValue::Type(self.clone())
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

    pub fn module(&self) -> RRC<Struct> {
        match self {
            Type::Module(m) => m.clone(),
            _ => unreachable!(),
        }
    }

    pub fn is_noreturn(&self) -> bool {
        match self {
            Type::NoReturn(_) => true,
            _ => false,
        }
    }

    pub fn inner_type(&self) -> Type {
        match self {
            Type::Register(reg_rrc) => reg_rrc.map(|reg| reg.ty.clone()),
            _ => self.clone(),
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
            Type::Register(_) => f.write_fmt(format_args!("reg({})", self.inner_type())),

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
