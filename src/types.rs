// Primitive types which can be constructed at compile time through user code. All changes here
// must be ported to user facing type interfacing libraries.

use crate::builtin::Signedness;

pub enum Type<'a> {
    Int(IntInfo),
    Struct(StructInfo<'a>),
    Type,
}

#[repr(C)]
pub struct IntInfo {
    sign: Signedness,
    width: u16,
}

#[repr(C)]
pub struct StructInfo<'a> {
    fields: Box<[StructField<'a>]>,
    decls: Box<[Declaration<'a>]>,
}

#[repr(C)]
pub struct StructField<'a> {
    name: &'a str,
    ty: Type<'a>,
}

#[repr(C)]
pub struct Declaration<'a> {
    name: &'a str,
}
