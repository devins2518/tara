use crate::{
    ast::Node,
    module::{comb::Comb, file::File, function::Function, namespace::Namespace, structs::Struct},
    types::Type as TaraType,
    utils::{init_field, RRC},
    utir::inst::UtirInstRef,
    values::{TypedValue, Value as TaraValue},
};
use std::{collections::HashMap, hash::Hash, mem::MaybeUninit};

// A container, function, or comb
pub struct Decl {
    pub name: String,
    pub ty: Option<TaraType>,
    // TODO: handle this by making parentless one by defualt, init_struct will set this and set the
    // parent
    pub namespace: RRC<Namespace>,
    pub value: Option<TaraValue>,
    pub node_ptr: *const Node,
    pub public: bool,
    pub export: bool,
    pub status: DeclStatus,
}

impl Decl {
    pub fn new<T: Into<String>>(name: T, node: &Node) -> Self {
        let name = name.into();
        let node_ptr = node as *const _;
        Self {
            name,
            ty: None,
            #[allow(invalid_value)]
            namespace: unsafe { MaybeUninit::uninit().assume_init() },
            value: None,
            node_ptr,
            public: false,
            export: false,
            status: DeclStatus::Unreferenced,
        }
    }

    pub fn child_name(&self, child_name: &str) -> String {
        [&self.name, child_name].join(".").to_string()
    }

    pub fn file_scope(&self) -> RRC<File> {
        unimplemented!()
    }

    pub fn namespace(&self) -> RRC<Namespace> {
        self.namespace.clone()
    }

    pub fn init_struct_empty_namespace<S: Into<RRC<Self>>>(decl: S, node: &Node) -> RRC<Self> {
        let rrc = decl.into();

        {
            let mut decl = rrc.borrow_mut();

            decl.ty = Some(TaraType::Type);
            decl.status = DeclStatus::InProgress;

            let struct_obj = Struct::new(node, rrc.clone());
            let struct_obj_rrc = RRC::new(struct_obj);

            let struct_ty = TaraType::Struct(struct_obj_rrc.clone());

            let struct_val = TaraValue::Type(struct_ty.clone());
            decl.value = Some(struct_val);

            let mut namespace = Namespace::new();
            namespace.init_ty(struct_ty);
            namespace.parent = None;
            let namespace_rrc = RRC::new(namespace);
            init_field!(decl, namespace, namespace_rrc.clone());

            struct_obj_rrc
                .borrow_mut()
                .init_namespace(namespace_rrc.clone());
        }

        rrc
    }

    pub fn init_struct<S: Into<RRC<Self>>>(
        decl: S,
        node: &Node,
        parent_namespace: RRC<Namespace>,
    ) -> RRC<Self> {
        let rrc = Self::init_struct_empty_namespace(decl, node);

        {
            let decl = rrc.borrow();
            let mut namespace = decl.namespace.borrow_mut();
            namespace.parent = Some(parent_namespace);
        }

        rrc
    }

    pub fn init_module<S: Into<RRC<Self>>>(
        decl: S,
        node: &Node,
        parent_namespace: RRC<Namespace>,
    ) -> RRC<Self> {
        let rrc = Self::init_struct(decl, node, parent_namespace);

        rrc.map_mut(|decl_rrc| {
            let module = decl_rrc.value.as_ref().unwrap().to_type().to_struct();
            decl_rrc.value = Some(TaraValue::Type(TaraType::Module(module)));
        });

        rrc
    }

    pub fn init_fn<S: Into<RRC<Self>>>(
        decl: S,
        node: &Node,
        parent_namespace: RRC<Namespace>,
    ) -> RRC<Self> {
        let rrc = decl.into();

        {
            rrc.map_mut(|decl| {
                decl.ty = Some(TaraType::Type);
                decl.status = DeclStatus::InProgress;
            });

            let fn_obj = Function::new(rrc.clone(), node);
            let fn_obj_rrc = RRC::new(fn_obj);

            let fn_ty = TaraType::Function(fn_obj_rrc.clone());

            let fn_val = TaraValue::Type(fn_ty.clone());
            rrc.map_mut(|decl| decl.value = Some(fn_val.clone()));

            let mut namespace = Namespace::new();
            namespace.init_ty(fn_ty);
            namespace.parent = Some(parent_namespace);
            let namespace_rrc = RRC::new(namespace);
            rrc.map_mut(|decl| init_field!(decl, namespace, namespace_rrc.clone()));
        }

        rrc
    }

    pub fn init_comb<S: Into<RRC<Self>>>(
        decl: S,
        node: &Node,
        parent_namespace: RRC<Namespace>,
    ) -> RRC<Self> {
        let rrc = decl.into();

        {
            rrc.map_mut(|decl| {
                decl.ty = Some(TaraType::Type);
                decl.status = DeclStatus::InProgress;
            });

            let comb_obj = Comb::new(rrc.clone(), node);
            let comb_obj_rrc = RRC::new(comb_obj);

            let comb_ty = TaraType::Comb(comb_obj_rrc.clone());

            let comb_val = TaraValue::Type(comb_ty.clone());
            rrc.map_mut(|decl| decl.value = Some(comb_val.clone()));

            let mut namespace = Namespace::new();
            namespace.init_ty(comb_ty);
            namespace.parent = Some(parent_namespace);
            let namespace_rrc = RRC::new(namespace);
            rrc.map_mut(|decl| init_field!(decl, namespace, namespace_rrc.clone()));
        }

        rrc
    }

    pub fn node<'a, 'b>(&'a self) -> &'b Node {
        unsafe { &*self.node_ptr }
    }

    pub fn error_guard<S: Into<RRC<Self>>>(decl: S) -> DeclErrorGuard {
        let rrc = decl.into();

        DeclErrorGuard {
            decl: rrc,
            errored: true,
        }
    }
}

impl Hash for Decl {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state)
    }
}

impl PartialEq for Decl {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for Decl {}

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

pub struct DeclErrorGuard {
    decl: RRC<Decl>,
    errored: bool,
}

impl DeclErrorGuard {
    pub fn success(&mut self) {
        self.errored = false;
    }
}

impl Drop for DeclErrorGuard {
    fn drop(&mut self) {
        if self.errored {
            self.decl
                .map_mut(|decl| decl.status = DeclStatus::DependencyFailure);
        }
    }
}

pub struct CaptureScope {
    parent: Option<RRC<CaptureScope>>,
    captures: HashMap<UtirInstRef, TypedValue>,
}

impl CaptureScope {
    pub fn new(parent: Option<RRC<CaptureScope>>) -> Self {
        Self {
            parent,
            captures: HashMap::new(),
        }
    }
}

impl Hash for CaptureScope {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.parent.hash(state)
    }
}
