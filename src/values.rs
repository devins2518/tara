use crate::{
    builtin::Signedness,
    module::{function::Function, namespace::Namespace},
    types::Type as TaraType,
    utils::RRC,
};
use melior::ir::{Value as MlirValue, ValueLike};
use std::hash::Hash;

pub type StaticMlirValue = MlirValue<'static, 'static>;

#[derive(Clone)]
pub enum Value {
    // Well known values
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

    // TODO: Remove rrc from here
    // Compile time types
    Type(RRC<TaraType>),
    Function(RRC<Function>),
    /// An instance of a struct.
    Struct(RRC<Vec<Value>>),

    // Compile time values
    Integer(i64),

    // Value known at run-time
    RuntimeValue(StaticMlirValue),
    // Run time mutable variable
    // Variable(RRC<Variable>),
}

impl Value {
    pub fn get_runtime_value(&self) -> StaticMlirValue {
        match self {
            Value::RuntimeValue(val) => val.clone(),
            _ => unreachable!(),
        }
    }

    pub fn has_runtime_value(&self) -> bool {
        match self {
            Value::RuntimeValue(_) => true,
            _ => false,
        }
    }

    pub fn to_type(&self) -> TaraType {
        match self {
            Value::Type(ty) => ty.borrow().clone(),
            _ => unreachable!(),
        }
    }

    pub fn namespace(&self) -> RRC<Namespace> {
        self.to_type().namespace()
    }

    pub fn integer(&self) -> i64 {
        match self {
            Value::Integer(int) => *int,
            _ => unreachable!(),
        }
    }
}

impl Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let dis = std::mem::discriminant(self);
        dis.hash(state);
        match self {
            Value::Type(ty) => ty.hash(state),
            Value::Function(f) => f.hash(state),
            Value::Struct(s) => s.hash(state),
            Value::Integer(i) => i.hash(state),
            Value::RuntimeValue(v) => v.to_raw().ptr.hash(state),
            _ => unreachable!(),
        }
    }
}

impl<'a, 'b> From<MlirValue<'a, 'b>> for Value {
    fn from(value: MlirValue<'a, 'b>) -> Self {
        let raw = unsafe { MlirValue::from_raw(value.to_raw()) };
        Self::RuntimeValue(raw)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct IntInfo {
    sign: Signedness,
    width: u16,
}

#[derive(Clone, Hash)]
pub struct TypedValue {
    pub ty: TaraType,
    pub value: Value,
}

impl TypedValue {
    pub fn new(ty: TaraType, value: Value) -> Self {
        Self { ty, value }
    }
}
