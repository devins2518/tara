use crate::{
    module::namespace::Namespace,
    types::Type,
    utir::inst::UtirInstRef,
    values::{TypedValue, Value},
};
use std::{collections::HashMap, hash::Hash};

#[derive(PartialEq, Eq, Hash)]
pub struct Decl<'module> {
    pub name: &'module str,
    pub ty: Option<Type<'module>>,
    pub value: Option<Value<'module>>,
    pub src_namespace: &'module Namespace<'module>,
    pub src_scope: Option<&'module CaptureScope<'module>>,
    pub utir_inst: UtirInstRef,
    pub public: bool,
    pub export: bool,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum DeclStatus {
    /// This Decl corresponds to an AST Node that has not been referenced yet, and therefore
    /// because of Zig's lazy declaration analysis, it will remain unanalyzed until referenced.
    Unreferenced,
    /// Semantic analysis for this Decl is running right now.
    /// This state detects dependency loops.
    InProgress,
    /// The file corresponding to this Decl had a parse error or ZIR error.
    /// There will be a corresponding ErrorMsg in Module.failed_files.
    FileFailure,
    /// This Decl might be OK but it depends on another one which did not successfully complete
    /// semantic analysis.
    DependencyFailure,
    /// Semantic analysis failure.
    /// There will be a corresponding ErrorMsg in Module.failed_decls.
    SemaFailure,
    /// There will be a corresponding ErrorMsg in Module.failed_decls.
    /// This indicates the failure was something like running out of disk space,
    /// and attempting semantic analysis again may succeed.
    SemaFailureRetryable,
    /// There will be a corresponding ErrorMsg in Module.failed_decls.
    CodegenFailure,
    /// There will be a corresponding ErrorMsg in Module.failed_decls.
    /// This indicates the failure was something like running out of disk space,
    /// and attempting codegen again may succeed.
    CodegenFailureRetryable,
    /// Everything is done. During an update, this Decl may be out of date, depending
    /// on its dependencies. The `generation` field can be used to determine if this
    /// completion status occurred before or after a given update.
    Complete,
    /// A Module update is in progress, and this Decl has been flagged as being known
    /// to require re-analysis.
    Outdated,
}

#[derive(PartialEq, Eq)]
pub struct CaptureScope<'module> {
    parent: Option<&'module CaptureScope<'module>>,
    captures: HashMap<UtirInstRef, TypedValue<'module>>,
}

impl<'module> Hash for CaptureScope<'module> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.parent.hash(state)
    }
}
