use crate::intern::Intern;

pub struct Module<'a> {
    intern: Intern<'a>,
}

impl<'a> Module<'a> {
    pub fn new() -> Self {
        return Self {
            intern: Intern::new(),
        };
    }
}
