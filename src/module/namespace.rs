use crate::{module::decls::Decl, types::Type};
use std::collections::HashMap;
use std::hash::Hash;

#[derive(PartialEq, Eq)]
pub struct Namespace<'module> {
    parent: Option<&'module Namespace<'module>>,
    // file?
    ty: Type<'module>,
    decls: HashMap<&'module str, &'module Decl<'module>>,
}

impl<'module> Hash for Namespace<'module> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.parent.hash(state);
        self.ty.hash(state);
        for decl in self.decls.keys() {
            (*decl).hash(state)
        }
    }
}
