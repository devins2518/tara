use crate::{module::structs::Struct, utils::RRC};
use std::hash::Hash;

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
}

impl Hash for Type {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Type::IntSigned { width } => state.write_u16(*width),
            Type::IntUnsigned { width } => state.write_u16(*width),
            Type::Struct(s) => (*s).hash(state),
            _ => {}
        }
    }
}
