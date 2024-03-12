use crate::utils::arena::Arena;

pub struct Tables {
    extra: Arena<u32>,
}

impl Tables {
    pub fn new() -> Self {
        return Self {
            extra: Arena::new(),
        };
    }
}
