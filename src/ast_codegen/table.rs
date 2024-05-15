use crate::{
    ast::{Node, NodeKind},
    ast_codegen::Error,
    types::Type as TaraType,
    values::Value as TaraValue,
};
use anyhow::Result;
use melior::{
    ir::{attribute::FlatSymbolRefAttribute as MlirFlatSymbolRefAttribute, Type as MlirType},
    Context,
};
use quickscope::ScopeMap;
use std::collections::HashMap;
use symbol_table::GlobalSymbol;

pub struct Table<'ctx, 'ast> {
    // TODO: add publicity to this
    name_table: ScopeMap<GlobalSymbol, &'ast Node>,
    // TODO: store params for type checking
    fn_table: ScopeMap<GlobalSymbol, MlirFlatSymbolRefAttribute<'ctx>>,
    value_table: ScopeMap<*const Node, TaraValue>,
    type_table: ScopeMap<*const Node, TaraType>,
    type_conversion_table: HashMap<TaraType, MlirType<'ctx>>,
}

impl<'ctx, 'ast> Table<'ctx, 'ast> {
    pub fn new() -> Self {
        Self {
            name_table: ScopeMap::new(),
            fn_table: ScopeMap::new(),
            value_table: ScopeMap::new(),
            type_table: ScopeMap::new(),
            type_conversion_table: HashMap::new(),
        }
    }

    pub fn define_name(&mut self, ident: GlobalSymbol, node: &'ast Node) -> Result<()> {
        if self.name_table.contains_key(&ident) {
            anyhow::bail!(Error::new(node.span, "Shadowing detected".to_string()));
        }
        self.name_table.define(ident, node);
        Ok(())
    }

    pub fn define_name_intentional_shadow(
        &mut self,
        ident: GlobalSymbol,
        node: &'ast Node,
    ) -> Result<()> {
        self.name_table.define(ident, node);
        Ok(())
    }

    pub fn define_symbol(&mut self, ident: GlobalSymbol, value: TaraValue) {
        assert!(self.name_table.contains_key(&ident));
        let node = self.name_table.get(&ident).unwrap();
        self.define_value(node, value);
    }

    pub fn define_fn(&mut self, ident: GlobalSymbol, fn_name: MlirFlatSymbolRefAttribute<'ctx>) {
        assert!(self.name_table.contains_key(&ident));
        self.fn_table.define(ident, fn_name);
    }

    pub fn define_typed_value(&mut self, node: &'ast Node, ty: TaraType, value: TaraValue) {
        let rrc = ty.into();
        self.define_type(node, rrc);
        self.define_value(node, value);
    }

    pub fn define_type(&mut self, node: &'ast Node, ty: TaraType) {
        self.type_table.define(node, ty);
    }

    pub fn define_value(&mut self, node: &Node, value: TaraValue) {
        let ptr: *const Node = node;
        self.value_table.define(ptr, value);
    }

    pub fn get_name(&self, node: &Node) -> Option<GlobalSymbol> {
        for (name, val) in self.name_table.iter() {
            if std::ptr::eq(*val, node) {
                return Some(*name);
            }
        }
        None
    }

    pub fn get_node(&self, name: GlobalSymbol) -> &Node {
        self.name_table.get(&name).unwrap()
    }

    pub fn get_type(&self, node: &Node) -> Result<TaraType> {
        match &node.kind {
            NodeKind::Identifier(_) => self.get_identifier_type(node),
            _ => {
                let ptr: *const Node = node;
                Ok(self.type_table.get(&ptr).unwrap().clone())
            }
        }
    }

    pub fn get_value(&self, node: &Node) -> Result<TaraValue> {
        let ptr: *const Node = node;
        Ok(self.value_table.get(&ptr).unwrap().clone())
    }

    pub fn get_mlir_type(&mut self, ctx: &'ctx Context, ty: TaraType) -> Result<MlirType<'ctx>> {
        Ok(*self
            .type_conversion_table
            .entry(ty.clone())
            .or_insert_with(|| ty.to_mlir_type(ctx)))
    }

    pub fn get_mlir_type_node(
        &mut self,
        ctx: &'ctx Context,
        node: &'ast Node,
    ) -> Result<MlirType<'ctx>> {
        let ty = self.get_type(node)?;
        self.get_mlir_type(ctx, ty)
    }

    // Pushes a layer onto all tables
    pub fn push(&mut self) {
        self.name_table.push_layer();
        self.fn_table.push_layer();
        self.value_table.push_layer();
        self.type_table.push_layer();
    }

    // Pops a layer from all tables
    pub fn pop(&mut self) {
        self.name_table.pop_layer();
        self.fn_table.pop_layer();
        self.value_table.pop_layer();
        self.type_table.pop_layer();
    }

    pub fn get_identifier_value(&self, node: &Node) -> Result<TaraValue> {
        matches!(node.kind, NodeKind::Identifier(_));
        let ident = match node.kind {
            NodeKind::Identifier(i) => i,
            _ => unreachable!(),
        };
        let ident_node: *const Node = *self.name_table.get(&ident).unwrap();
        self.value_table
            .get(&ident_node)
            .ok_or_else(|| Error::new(node.span, "Unknown identifier".to_string()).into())
            .cloned()
    }

    pub fn get_identifier_type(&self, node: &Node) -> Result<TaraType> {
        matches!(node.kind, NodeKind::Identifier(_));
        let ident = match node.kind {
            NodeKind::Identifier(i) => i,
            _ => unreachable!(),
        };
        let ident_node: *const Node = *self
            .name_table
            .get(&ident)
            .ok_or_else(|| Error::new(node.span, "Use of unknown identifier".to_string()))?;
        self.type_table
            .get(&ident_node)
            .ok_or_else(|| Error::new(node.span, "Unknown identifier".to_string()).into())
            .cloned()
    }
}
