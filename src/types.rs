use crate::{
    module::{structs::Struct, tmodule::TModule},
    utils::RRC,
};
use melior::{ir::Type as MlirType, Context};
use std::hash::Hash;

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
        unimplemented!()
    }
}
