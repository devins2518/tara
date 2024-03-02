use crate::utils::arena::{Arena, Id};
use crate::{types::Type, values::Value};

pub struct Intern<'a> {
    values: Arena<Value<'a>>,
    types: Arena<Type<'a>>,
}

impl<'a> Intern<'a> {
    pub fn new() -> Self {
        return Self {
            values: Arena::new(),
            types: Arena::new(),
        };
    }

    pub fn intern_val(&mut self, val: Value<'a>) -> ValueId {
        self.values.alloc(val)
    }
}

pub type ValueId<'a> = Id<Value<'a>>;
pub type TypeId<'a> = Id<Type<'a>>;
