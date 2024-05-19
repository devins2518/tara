use crate::{
    ast::Node,
    module::{decls::Decl, namespace::Namespace},
    types::Type,
    utils::{init_field, RRC},
};
use std::{hash::Hash, mem::MaybeUninit};

#[derive(Clone)]
pub struct Struct {
    pub decl: RRC<Decl>,
    pub fields: Vec<Field>,
    pub namespace: RRC<Namespace>,
    pub node_ptr: *const Node,
    pub status: StructStatus,
}

impl Struct {
    // Caller MUST call `init_namespace` after creating a namespace
    pub fn new(node: &Node, decl: RRC<Decl>) -> Self {
        let node_ptr = node as *const _;
        Self {
            decl,
            fields: Vec::new(),
            #[allow(invalid_value)]
            namespace: unsafe { MaybeUninit::uninit().assume_init() },
            node_ptr,
            status: StructStatus::None,
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

impl PartialEq for Struct {
    fn eq(&self, other: &Self) -> bool {
        self.decl == other.decl
    }
}

impl Eq for Struct {}

impl Hash for Struct {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.node_ptr.hash(state);
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum StructStatus {
    None,
    FieldTypeWip,
    HaveFieldTypes,
    LayoutWip,
    HaveLayout,
    FullyResolvedWip,
    FullyResolved,
}

// TODO: default values
#[derive(Clone, Hash)]
pub struct Field {
    pub name: String,
    pub ty: Type,
}
