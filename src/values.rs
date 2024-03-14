use crate::{
    builtin::Signedness,
    module::{function::Function, variable::Variable},
    types::Type,
};

#[derive(PartialEq, Eq, Hash)]
pub enum Value<'module> {
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

    Type(Type<'module>),
    IntType(IntInfo),
    Function(&'module Function<'module>),
    Variable(&'module Variable<'module>),
    /// An instance of a struct.
    Struct(&'module [Value<'module>]),
    // Struct(Vec<Value<'module>>),
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct IntInfo {
    sign: Signedness,
    width: u16,
}

#[derive(PartialEq, Eq, Hash)]
pub struct TypedValue<'module> {
    ty: Type<'module>,
    value: Value<'module>,
}
