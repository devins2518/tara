use crate::{
    module::decls::Decl,
    utils::id_arena::{Id, IdArena},
};

pub struct Tables {
    null_terminated_strings: IdArena<u8>,
    // decls: IdArena<Decl>,
    extra: IdArena<u32>,
}

impl Tables {
    pub fn new() -> Self {
        return Self {
            null_terminated_strings: IdArena::new(),
            extra: IdArena::new(),
        };
    }

    pub fn insert_string(&self, s: &str) -> StringId {
        let mut bytes = s.bytes();
        let start = self.null_terminated_strings.alloc(bytes.next().unwrap());
        for byte in bytes {
            _ = self.null_terminated_strings.alloc(byte);
        }
        _ = self.null_terminated_strings.alloc(b'\0');
        return start;
    }

    pub fn get_string(&self, id: StringId) -> &str {
        let start: u32 = id.into();
        let mut curr = start;
        let end = loop {
            if self.null_terminated_strings.get(curr.into()) == b'\0' {
                break curr;
            }
            curr += 1;
        };
        let slice = self.null_terminated_strings.slice(id, end - start);
        return unsafe { std::str::from_utf8_unchecked(slice) };
    }
}

pub type StringId = Id<u8>;
// pub type DeclId = Id<Decl>;
