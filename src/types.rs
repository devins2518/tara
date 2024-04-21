use crate::{
    module::{structs::Struct, tmodule::TModule},
    utils::RRC,
};
use melior::{
    ir::{r#type::IntegerType as MlirIntegerType, Type as MlirType},
    Context,
};
use std::{fmt::Display, hash::Hash};

#[derive(Clone, PartialEq, Eq)]
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
    Module,
    TypeType,
}

impl Hash for Type {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Type::IntSigned { width } => state.write_u16(*width),
            Type::IntUnsigned { width } => state.write_u16(*width),
            _ => {}
        }
    }
}

impl<'ctx> Type {
    pub fn to_mlir_type(&self, ctx: &'ctx Context) -> MlirType<'ctx> {
        match self {
            Type::Bool => MlirIntegerType::new(ctx, 1).into(),
            Type::Void => MlirIntegerType::new(ctx, 0).into(),
            Type::IntSigned { width } | Type::IntUnsigned { width } => {
                MlirIntegerType::new(ctx, (*width).into()).into()
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
            Type::Module => unimplemented!(),
            _ => unreachable!(),
        }
    }
}
