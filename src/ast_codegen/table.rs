use crate::{
    ast::{Node, NodeKind},
    ast_codegen::Error,
    types::Type as TaraType,
    values::{TypedValue, Value as TaraValue},
};
use anyhow::Result;
use codespan::Span;
use quickscope::ScopeMap;
use symbol_table::GlobalSymbol;

pub struct Table {
    name_table: ScopeMap<GlobalSymbol, *const Node>,
    bindings: ScopeMap<*const Node, TypedValue>,
}

impl Table {
    pub fn new() -> Self {
        Self {
            name_table: ScopeMap::new(),
            bindings: ScopeMap::new(),
        }
    }

    pub fn define_name(&mut self, ident: GlobalSymbol, node: &Node) -> Result<()> {
        if self.name_table.contains_key(&ident) {
            anyhow::bail!(Error::new(node.span, "Shadowing detected".to_string()));
        }
        self.name_table.define(ident, node);
        Ok(())
    }

    // Define a name that can shadow keys above the current layer.
    pub fn define_name_shadow_upper(&mut self, ident: GlobalSymbol, node: &Node) -> Result<()> {
        if self.name_table.contains_key_at_top(&ident) {
            anyhow::bail!(Error::new(node.span, "Shadowing detected".to_string()));
        }
        self.name_table.define(ident, node);
        Ok(())
    }

    pub fn define_ty_val(&mut self, node: &Node, ty_val: TypedValue) {
        let ptr = node as *const _;
        self.bindings.define(ptr, ty_val);
    }

    pub fn get_name(&self, node: &Node) -> Option<GlobalSymbol> {
        for (name, val) in self.name_table.iter() {
            if std::ptr::eq(*val, node) {
                return Some(*name);
            }
        }
        None
    }

    pub fn get_ty_val(&self, node: &Node) -> Result<TypedValue> {
        match &node.kind {
            NodeKind::Identifier(i) => self.get_identifier_ty_val(*i, node.span),
            _ => {
                let ptr: *const Node = node;
                Ok(self.bindings.get(&ptr).unwrap().clone())
            }
        }
    }

    pub fn get_type(&self, node: &Node) -> Result<TaraType> {
        self.get_ty_val(node).map(|ty_val| ty_val.ty)
    }

    pub fn get_value(&self, node: &Node) -> Result<TaraValue> {
        self.get_ty_val(node).map(|ty_val| ty_val.value)
    }

    pub fn get_named_ty_val(&self, ident: GlobalSymbol, loc: Span) -> Result<TypedValue> {
        Ok(self.get_identifier_ty_val(ident, loc)?)
    }

    pub fn get_named_type(&self, ident: GlobalSymbol, loc: Span) -> Result<TaraType> {
        self.get_named_ty_val(ident, loc).map(|ty_val| ty_val.ty)
    }

    pub fn get_named_value(&self, ident: GlobalSymbol, loc: Span) -> Result<TaraValue> {
        self.get_named_ty_val(ident, loc).map(|ty_val| ty_val.value)
    }

    // Pushes a layer onto all tables
    pub fn push(&mut self) {
        self.name_table.push_layer();
        self.bindings.push_layer();
    }

    // Pops a layer from all tables
    pub fn pop(&mut self) {
        self.name_table.pop_layer();
        self.bindings.pop_layer();
    }

    fn get_identifier_ty_val(&self, ident: GlobalSymbol, loc: Span) -> Result<TypedValue> {
        let ident_node: *const Node = *self.name_table.get(&ident).unwrap();
        self.bindings
            .get(&ident_node)
            .ok_or_else(|| Error::new(loc, "Unknown identifier".to_string()).into())
            .cloned()
    }
}
