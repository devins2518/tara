use crate::{
    module::{structs::Struct, tmodule::TModule},
    utils::RRC,
};
use std::hash::Hash;

#[derive(Clone)]
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
    Struct(RRC<Struct>),
    Module(RRC<TModule>),
    TypeType,
}

impl Hash for Type {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Type::IntSigned { width } => state.write_u16(*width),
            Type::IntUnsigned { width } => state.write_u16(*width),
            Type::Struct(s) => (*s).hash(state),
            Type::Module(m) => (*m).hash(state),
            _ => {}
        }
    }
}
