use crate::{
    ast::{Node, NodeKind},
    ast_codegen::Error,
    types::Type as TaraType,
};
use anyhow::Result;
use melior::{
    ir::{
        attribute::StringAttribute as MlirStringAttribute, Type as MlirType, Value as MlirValue,
        ValueLike,
    },
    Context,
};
use quickscope::ScopeMap;
use std::collections::HashMap;
use symbol_table::GlobalSymbol;

pub struct Table<'ctx, 'ast> {
    name_table: ScopeMap<GlobalSymbol, &'ast Node>,
    // TODO: add publicity to this
    symbol_table: ScopeMap<GlobalSymbol, MlirValue<'ctx, 'ctx>>,
    // TODO: store params for type checking
    fn_table: ScopeMap<GlobalSymbol, MlirStringAttribute<'ctx>>,
    type_table: HashMap<*const Node, TaraType>,
    type_conversion_table: HashMap<TaraType, MlirType<'ctx>>,
}

impl<'ctx, 'ast> Table<'ctx, 'ast> {
    pub fn new() -> Self {
        Self {
            name_table: ScopeMap::new(),
            symbol_table: ScopeMap::new(),
            fn_table: ScopeMap::new(),
            type_table: HashMap::new(),
            type_conversion_table: HashMap::new(),
        }
    }

    pub fn define_name(&mut self, ident: GlobalSymbol, node: &'ast Node) -> Result<()> {
        if self.name_table.contains_key(&ident) {
            // TODO: proper error
            anyhow::bail!(Error::new(node.span, "Shadowing detected".to_string()));
        }
        self.name_table.define(ident, node);
        Ok(())
    }

    pub fn define_symbol(&mut self, ident: GlobalSymbol, value: MlirValue<'ctx, '_>) {
        assert!(self.name_table.contains_key(&ident));
        let raw_value = value.to_raw();
        self.symbol_table
            .define(ident, unsafe { MlirValue::from_raw(raw_value) });
    }

    pub fn define_fn(&mut self, ident: GlobalSymbol, fn_name: MlirStringAttribute<'ctx>) {
        assert!(self.name_table.contains_key(&ident));
        self.fn_table.define(ident, fn_name);
    }

    pub fn define_type(&mut self, node: &'ast Node, ty: TaraType) {
        self.type_table.insert(node, ty);
    }

    pub fn get_type(&mut self, node: &'ast Node) -> TaraType {
        let ptr: *const Node = node;
        self.type_table.get(&ptr).unwrap().to_owned()
    }

    pub fn get_mlir_type(&mut self, ctx: &'ctx Context, node: &'ast Node) -> MlirType<'ctx> {
        let ty = self.get_type(node);
        *self
            .type_conversion_table
            .entry(ty.clone())
            .or_insert_with(|| ty.to_mlir_type(ctx))
    }

    // Pushes a layer onto all tables
    pub fn push(&mut self) {
        self.name_table.push_layer();
        self.symbol_table.push_layer();
        self.fn_table.push_layer();
    }

    // Pops a layer from all tables
    pub fn pop(&mut self) {
        self.name_table.pop_layer();
        self.symbol_table.pop_layer();
        self.fn_table.pop_layer();
    }

    pub fn get_identifier(&self, node: &'ast Node) -> Result<MlirValue<'ctx, 'ctx>> {
        matches!(node.kind, NodeKind::Identifier(_));
        let ident = match node.kind {
            NodeKind::Identifier(i) => i,
            _ => unreachable!(),
        };
        self.symbol_table
            .get(&ident)
            .ok_or_else(|| Error::new(node.span, "Unknown identifier".to_string()).into())
            .copied()
    }
}
