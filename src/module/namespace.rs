use crate::module::file::File;
use crate::{module::decls::Decl, types::Type};
use std::collections::HashMap;
use std::hash::Hash;

#[derive(PartialEq, Eq)]
pub struct Namespace<'module> {
    pub parent: Option<&'module Namespace<'module>>,
    pub file: File<'module>,
    pub ty: Type<'module>,
    pub decls: HashMap<&'module str, &'module Decl<'module>>,
}

impl<'module> Namespace<'module> {
    pub fn new(file: File<'module>, ty: Type<'module>) -> Self {
        return Self {
            parent: None,
            file,
            ty,
            decls: HashMap::new(),
        };
    }
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
