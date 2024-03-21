use crate::module::file::File;
use crate::utils::RRC;
use crate::{module::decls::Decl, types::Type};
use std::collections::HashMap;
use std::hash::Hash;

pub struct Namespace {
    pub parent: Option<RRC<Namespace>>,
    pub file: RRC<File>,
    pub ty: RRC<Type>,
    pub decls: HashMap<String, RRC<Decl>>,
}

impl Namespace {
    pub fn new(file: RRC<File>, ty: RRC<Type>) -> Self {
        Self {
            parent: None,
            file,
            ty,
            decls: HashMap::new(),
        }
    }
}

impl Hash for Namespace {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.parent.hash(state);
        self.ty.hash(state);
        for decl in self.decls.keys() {
            (*decl).hash(state)
        }
    }
}
