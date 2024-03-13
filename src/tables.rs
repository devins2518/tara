use crate::utils::arena::IdArena;

pub struct Tables {
    extra: IdArena<u32>,
}

impl Tables {
    pub fn new() -> Self {
        return Self {
            extra: IdArena::new(),
        };
    }
}
