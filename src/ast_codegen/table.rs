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
    linear_bindings: ScopeMap<*const Node, LinearityStatus>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum LinearityStatus {
    Unconsumed,
    BorrowedRead,
    BorrowedWrite,
    Consumed,
}

impl Table {
    pub fn new() -> Self {
        Self {
            name_table: ScopeMap::new(),
            bindings: ScopeMap::new(),
            linear_bindings: ScopeMap::new(),
        }
    }

    pub fn define_name(&mut self, ident: GlobalSymbol, node: &Node) -> Result<()> {
        if self.name_table.contains_key(&ident) {
            anyhow::bail!(Error::new(node.span, "Shadowing detected".to_string()));
        }
        self.name_table.define(ident, node);
        Ok(())
    }

    pub fn define_ty_val(&mut self, node: &Node, ty_val: TypedValue) {
        let ptr = node as *const _;
        self.bindings.define(ptr, ty_val);
    }

    pub fn define_linearity(&mut self, node: &Node) {
        let ptr = node as *const _;
        self.linear_bindings
            .define(ptr, LinearityStatus::Unconsumed);
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
        self.linear_bindings.push_layer();
    }

    // Pops a layer from all tables
    pub fn pop(&mut self) -> Result<()> {
        let mut maybe_err = None;
        for (node_ptr, status) in self.linear_bindings.iter_top() {
            if status != &LinearityStatus::Consumed {
                let node: &Node = unsafe { std::mem::transmute(*node_ptr) };
                maybe_err = Some(Error::new(node.span, "Linear value not consumed!"));
                break;
            }
        }
        self.name_table.pop_layer();
        self.bindings.pop_layer();
        self.linear_bindings.pop_layer();
        Ok(maybe_err.map(Result::Err).unwrap_or(Ok(()))?)
    }

    pub fn get_linear_bindings(&mut self) -> Vec<(*const Node, LinearityStatus)> {
        self.linear_bindings.iter().map(|(k, v)| (*k, *v)).collect()
    }

    pub fn consume(&mut self, linear_node: &Node, consuming_node: &Node) -> Result<()> {
        let ptr = linear_node as *const _;
        let status = self.linear_bindings.get(&ptr).unwrap();
        match status {
            LinearityStatus::Unconsumed => {}
            LinearityStatus::Consumed => Err(Error::new(
                consuming_node.span,
                "Linear element already consumed when used here!",
            ))?,
            _ => unimplemented!(),
        }
        self.linear_bindings.define(ptr, LinearityStatus::Consumed);
        Ok(())
    }

    fn get_identifier_ty_val(&self, ident: GlobalSymbol, loc: Span) -> Result<TypedValue> {
        let ident_node: *const Node = *self
            .name_table
            .get(&ident)
            .ok_or_else(|| Error::new(loc, "Unknown identifier".to_string()))?;
        self.bindings
            .get(&ident_node)
            .ok_or_else(|| Error::new(loc, "Unknown identifier".to_string()).into())
            .cloned()
    }
}
