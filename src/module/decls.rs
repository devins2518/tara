use crate::{
    module::namespace::Namespace,
    types::Type,
    utils::{RRC, WRC},
    utir::inst::UtirInstRef,
    values::{TypedValue, Value},
};
use std::{collections::HashMap, hash::Hash};

#[derive(Hash)]
pub struct Decl {
    pub name: String,
    pub ty: Option<Type>,
    pub value: Option<Value>,
    pub src_namespace: RRC<Namespace>,
    pub src_scope: Option<RRC<CaptureScope>>,
    pub utir_inst: UtirInstRef,
    pub public: bool,
    pub export: bool,
    pub status: DeclStatus,
}

impl Decl {
    pub fn new(
        name: String,
        src_namespace: RRC<Namespace>,
        utir_inst: UtirInstRef,
        src_scope: Option<RRC<CaptureScope>>,
    ) -> Self {
        Self {
            name,
            ty: None,
            value: None,
            src_namespace,
            src_scope,
            utir_inst,
            public: false,
            export: false,
            status: DeclStatus::Unreferenced,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum DeclStatus {
    // This Decl corresponds to an AST Node that has not been referenced yet
    Unreferenced,
    // Semantic analysis for this Decl is running right now. This state is used to detect
    // dependency loops
    InProgress,
    // The file corresponding to this Decl had a parse error or UTIR error
    FileFailure,
    // This Decl might be OK but it depends on another one which did not successfully complete
    // semantic analysis
    DependencyFailure,
    /// Semantic analysis failure
    CodegenFailure,
    // Everything is done
    Complete,
}

pub struct CaptureScope {
    parent: Option<WRC<CaptureScope>>,
    captures: HashMap<UtirInstRef, TypedValue>,
}

impl Hash for CaptureScope {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.parent.hash(state)
    }
}
