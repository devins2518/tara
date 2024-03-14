use crate::module::structs::Struct;

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum Type<'module> {
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
    Struct(&'module Struct<'module>),
}
