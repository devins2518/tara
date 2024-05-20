use crate::{
    ast_codegen::Error,
    module::decls::Decl,
    types::Type,
    utils::{init_field, RRC},
};
use anyhow::Result;
use std::{collections::HashMap, hash::Hash, mem::MaybeUninit};

pub struct Namespace {
    pub parent: Option<RRC<Namespace>>,
    pub ty: Type,
    pub decls: HashMap<String, RRC<Decl>>,
}

impl Namespace {
    pub fn new() -> Self {
        Self {
            parent: None,
            #[allow(invalid_value)]
            ty: unsafe { MaybeUninit::uninit().assume_init() },
            decls: HashMap::new(),
        }
    }

    pub fn init_ty(&mut self, ty: Type) {
        init_field!(self, ty, ty);
    }

    pub fn find_decl(top: RRC<Namespace>, decl_name: &str) -> Option<RRC<Decl>> {
        let search = top.borrow();
        search.decls.get(decl_name).map(RRC::clone).or_else(|| {
            search
                .parent
                .clone()
                .map(|parent| Namespace::find_decl(parent, decl_name))
                .flatten()
        })
    }

    pub fn add_decl<T: Into<String>>(&mut self, name: T, decl: RRC<Decl>) -> Result<()> {
        let name = name.into();
        if let Some(_prev_decl) = self.decls.get(&name) {
            // TODO: add note with previous declaration
            let decl_node = decl.map(Decl::node);
            Err(Error::new(decl_node.span, "Declaration shadowing detected"))?
        }
        self.decls.insert(name, decl);
        Ok(())
    }

    pub fn decl(&self) -> RRC<Decl> {
        self.ty.decl()
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
