use crate::{
    ast::Node,
    module::{decls::Decl, namespace::Namespace},
    types::Type as TaraType,
    utils::{init_field, RRC},
};
use std::mem::MaybeUninit;

#[derive(Hash)]
pub struct TModule {
    pub decl: RRC<Decl>,
    // Ugly, needed for identifier resolution
    pub ins: Vec<(String, *const Node, TaraType)>,
    pub outs: Vec<(String, TaraType)>,
    pub namespace: RRC<Namespace>,
    pub node_ptr: *const Node,
    pub status: ModuleStatus,
}

impl TModule {
    pub fn new(decl: RRC<Decl>, node: &Node) -> Self {
        let node_ptr = node as *const _;
        TModule {
            decl,
            ins: Vec::new(),
            outs: Vec::new(),
            #[allow(invalid_value)]
            namespace: unsafe { MaybeUninit::uninit().assume_init() },
            node_ptr,
            status: ModuleStatus::None,
        }
    }

    pub fn init_namespace(&mut self, namespace: RRC<Namespace>) {
        init_field!(self, namespace, namespace);
    }

    pub fn decl(&self) -> RRC<Decl> {
        self.decl.clone()
    }

    pub fn node<'a, 'b>(&'a self) -> &'b Node {
        unsafe { &*self.node_ptr }
    }
}

impl PartialEq for TModule {
    fn eq(&self, other: &Self) -> bool {
        self.decl == other.decl
    }
}

impl Eq for TModule {}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum ModuleStatus {
    None,
    InProgress,
    SemaError,
    FullyResolved,
}
