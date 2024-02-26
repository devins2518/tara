use crate::{types::Type, values::Value};
use internment::{Arena, ArenaIntern};

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

    pub fn intern_val(&mut self, val: Value<'a>) -> ArenaIntern<Value> {
        self.values.intern(val)
    }
}
