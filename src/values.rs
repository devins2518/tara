use crate::{
    builtin::Signedness,
    module::{function::Function, variable::Variable},
    types::Type,
    utils::RRC,
};

#[derive(Hash)]
pub enum Value {
    U1Type,
    U8Type,
    I8Type,
    U16Type,
    I16Type,
    U32Type,
    I32Type,
    U64Type,
    I64Type,
    U128Type,
    I128Type,
    UsizeType,
    IsizeType,
    BoolType,
    VoidType,
    TypeType,
    ComptimeIntType,
    UndefinedType,
    EnumLiteralType,
    Undef,
    Zero,
    One,
    VoidValue,
    Unreachable,
    BoolTrue,
    BoolFalse,

    Type(RRC<Type>),
    IntType(IntInfo),
    Function(RRC<Function>),
    Variable(RRC<Variable>),
    /// An instance of a struct.
    Struct(RRC<Vec<Value>>),
    // Struct(Vec<Value<'module>>),
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct IntInfo {
    sign: Signedness,
    width: u16,
}

#[derive(Hash)]
pub struct TypedValue {
    pub ty: Type,
    pub value: Value,
}
