mod error;

use crate::{
    ast::{Node, NodeKind},
    ast_codegen::error::Error,
    Ast,
};
use anyhow::Result;
use melior::{
    dialect::{
        arith::CmpiPredicate,
        ods::{
            arith::{
                AddIOperation, AndIOperation, CmpIOperation, MulIOperation, OrIOperation,
                SubIOperation, XOrIOperation,
            },
            func::{r#return, FuncOperation, ReturnOperation},
        },
    },
    ir::{
        attribute::{
            IntegerAttribute as MlirIntegerAttribute, StringAttribute as MlirStringAttribute,
            TypeAttribute as MlirTypeAttribute,
        },
        operation::{
            Operation as MlirOperation, OperationBuilder, OperationRef as MlirOperationRef,
        },
        r#type::{FunctionType as MlirFunctionType, IntegerType as MlirIntegerType},
        Attribute as MlirAttribute, Block as MlirBlock, BlockRef as MlirBlockRef,
        Identifier as MlirIdentifier, Location, Module as MlirModule, Region as MlirRegion,
        Type as MlirType, Value as MlirValue, ValueLike,
    },
    Context,
};
use quickscope::ScopeMap;
use symbol_table::GlobalSymbol;

pub struct AstCodegen<'a, 'ast, 'ctx> {
    ast: &'ast Ast,
    ctx: &'ctx Context,
    module: &'a MlirModule<'ctx>,
    builder: Builder<'ctx, 'a>,
    name_table: ScopeMap<GlobalSymbol, &'ast Node>,
    // TODO: add publicity to this
    symbol_table: ScopeMap<GlobalSymbol, MlirValue<'ctx, 'a>>,
    // TODO: store params for type checking
    fn_table: ScopeMap<GlobalSymbol, MlirStringAttribute<'ctx>>,
}

impl<'a, 'ast, 'ctx> AstCodegen<'a, 'ast, 'ctx> {
    pub fn new(ast: &'ast Ast, ctx: &'ctx Context, module: &'a MlirModule<'ctx>) -> Self {
        Self {
            ast,
            ctx,
            module,
            builder: Builder::new(module.body()),
            name_table: ScopeMap::new(),
            symbol_table: ScopeMap::new(),
            fn_table: ScopeMap::new(),
        }
    }

    fn define_name(&mut self, ident: GlobalSymbol, node: &'ast Node) -> Result<()> {
        if self.name_table.contains_key(&ident) {
            // TODO: proper error
            anyhow::bail!("shadowing detected")
        }
        self.name_table.define(ident, node);
        Ok(())
    }

    fn define_symbol(&mut self, ident: GlobalSymbol, value: MlirValue<'ctx, '_>) -> Result<()> {
        let raw_value = value.to_raw();
        assert!(self.name_table.contains_key(&ident));
        self.symbol_table
            .define(ident, unsafe { MlirValue::from_raw(raw_value) });
        Ok(())
    }

    fn create<'blk, T: Into<MlirOperation<'ctx>> + Clone>(
        &mut self,
        block: MlirBlockRef<'ctx, 'blk>,
        operation: &T,
    ) -> Result<MlirValue<'ctx, 'blk>> {
        let op: T = operation.clone();
        let op_ref = block.append_operation(op.into());
        self.value_from_ref(op_ref)
    }

    fn value_from_ref<'blk>(&self, op_ref: MlirOperationRef) -> Result<MlirValue<'ctx, 'blk>> {
        let result = op_ref.result(0)?;
        let raw_value = result.to_raw();
        Ok(unsafe { MlirValue::from_raw(raw_value) })
    }

    // Pushes a layer on to the name table and symbol table
    fn push(&mut self) {
        self.name_table.push_layer();
        self.symbol_table.push_layer();
        self.fn_table.push_layer();
    }

    // Pops a layer from the name table and symbol table
    fn pop(&mut self) {
        self.name_table.pop_layer();
        self.symbol_table.pop_layer();
        self.fn_table.pop_layer();
    }
}

// MLIR generation methods
impl<'a, 'ast, 'ctx, 'blk> AstCodegen<'a, 'ast, 'ctx>
where
    'a: 'blk,
{
    pub fn gen_root(mut self) -> Result<()> {
        self.gen_struct_decl(&self.ast.root)?;

        Ok(())
    }

    fn _gen_container(
        &mut self,
        node: &'ast Node,
        subroutine_gen: impl Fn(&mut Self, &'ast Node) -> Result<()>,
    ) -> Result<()> {
        matches!(node.kind, NodeKind::StructDecl(_));
        // Setup name table
        self.push();
        let members = match &node.kind {
            NodeKind::StructDecl(s) => &s.members,
            NodeKind::ModuleDecl(m) => &m.members,
            _ => unreachable!(),
        };
        // Members will be either a VarDecl or a SubroutineDecl
        for member in members {
            match &member.kind {
                NodeKind::VarDecl(v_d) => self.define_name(v_d.ident, member)?,
                NodeKind::SubroutineDecl(s_d) => self.define_name(s_d.ident, member)?,
                _ => unreachable!(),
            }
        }

        for member in members {
            match &member.kind {
                NodeKind::VarDecl(v_d) => {
                    let val = self.gen_var_decl(&member)?;
                    self.symbol_table.define(v_d.ident, val);
                }
                NodeKind::SubroutineDecl(s_d) => {
                    let val = subroutine_gen(self, &member)?;
                }
                _ => unreachable!(),
            }
        }

        self.pop();
        Ok(())
    }

    fn gen_struct_decl(&mut self, node: &'ast Node) -> Result<()> {
        self._gen_container(node, Self::gen_fn_decl)
    }

    fn gen_module_decl(&mut self, node: &'ast Node) -> Result<()> {
        self._gen_container(node, Self::gen_comb_decl)
    }

    fn gen_var_decl(&mut self, node: &'ast Node) -> Result<MlirValue<'ctx, 'blk>> {
        matches!(node.kind, NodeKind::VarDecl(_));
        let block = self.module.body();

        let var_decl = match &node.kind {
            NodeKind::VarDecl(v_d) => v_d,
            _ => unreachable!(),
        };
        let maybe_type_expr = if let Some(ty) = &var_decl.ty {
            Some(self.gen_expr_reachable(block, &*ty)?)
        } else {
            None
        };
        let value = self.gen_expr_reachable(block, &var_decl.expr)?;

        if let Some(type_expr) = maybe_type_expr {
            self.type_check(type_expr, value)?;
        }

        Ok(value)
    }

    // FlatSymbolRefAttribute
    fn gen_fn_decl(&mut self, node: &'ast Node) -> Result<()> {
        matches!(node.kind, NodeKind::SubroutineDecl(_));
        let fn_decl = match &node.kind {
            NodeKind::SubroutineDecl(s_d) => s_d,
            _ => unreachable!(),
        };
        self.push();

        let return_type = self.gen_type(&fn_decl.return_type)?;

        let loc = node.loc(self.ctx);
        let body = self.module.body();
        let fn_name = {
            let block = MlirBlock::new(&[]);

            let mut param_types = Vec::new();
            for param in &fn_decl.params {
                self.define_name(param.name, &param.ty)?;
                let param_type = self.gen_type(&param.ty)?;
                param_types.push(param_type);

                let arg = block.add_argument(param_type, param.ty.loc(self.ctx));
                self.define_symbol(param.name, arg)?;
            }

            let region = MlirRegion::new();
            let block = region.append_block(block);

            self.gen_block(block, &fn_decl.block)?;

            // TODO: return type check

            let fn_type = MlirFunctionType::new(self.ctx, &param_types, &[return_type]).into();
            let builder = FuncOperation::builder(self.ctx, loc)
                .body(region)
                .sym_name(MlirStringAttribute::new(self.ctx, fn_decl.ident.as_str()))
                .function_type(MlirTypeAttribute::new(fn_type));
            let func = builder.build();
            let fn_name = func.sym_name()?;
            body.append_operation(func.into());
            fn_name
        };

        self.pop();

        self.fn_table.define(fn_decl.ident, fn_name);

        Ok(())
    }

    fn gen_comb_decl(&mut self, node: &'ast Node) -> Result<()> {
        matches!(node.kind, NodeKind::SubroutineDecl(_));
        self.push();

        self.pop();
        unimplemented!()
    }

    fn gen_return(&mut self, block: MlirBlockRef<'ctx, 'blk>, node: &'ast Node) -> Result<()> {
        matches!(node.kind, NodeKind::Return(_));
        let expr = match &node.kind {
            NodeKind::Return(e) => &e.lhs,
            _ => unreachable!(),
        };

        let return_value = self.gen_expr_reachable(block, &expr)?;

        let loc = node.loc(self.ctx);
        let return_op = ReturnOperation::builder(self.ctx, loc)
            .operands(&[return_value])
            .build();

        block.append_operation(return_op.into());
        Ok(())
    }

    fn gen_expr(
        &mut self,
        block: MlirBlockRef<'ctx, 'blk>,
        node: &'ast Node,
    ) -> Result<Option<MlirValue<'ctx, 'blk>>> {
        let maybe_val = match &node.kind {
            NodeKind::StructDecl(_) => {
                self.gen_struct_decl(node)?;
                None
            }
            NodeKind::Return(_) => {
                self.gen_return(block, node)?;
                None
            }
            NodeKind::Or(_)
            | NodeKind::And(_)
            | NodeKind::Lt(_)
            | NodeKind::Gt(_)
            | NodeKind::Lte(_)
            | NodeKind::Eq(_)
            | NodeKind::Neq(_)
            | NodeKind::BitAnd(_)
            | NodeKind::BitOr(_)
            | NodeKind::BitXor(_)
            | NodeKind::Add(_)
            | NodeKind::Sub(_)
            | NodeKind::Mul(_)
            | NodeKind::Div(_) => Some(self.gen_bin_op(block, node)?),
            NodeKind::Identifier(_) => Some(self.get_identifier(block, node)?),
            _ => unimplemented!(),
        };
        Ok(maybe_val)
    }

    fn gen_expr_reachable(
        &mut self,
        block: MlirBlockRef<'ctx, 'blk>,
        node: &'ast Node,
    ) -> Result<MlirValue<'ctx, 'blk>> {
        self.gen_expr(block, node)?.ok_or_else(|| {
            Error::new(
                node.span,
                "Expected reachable value, control flow unexpectedly diverted".to_string(),
            )
            .into()
        })
    }

    fn gen_bin_op(
        &mut self,
        block: MlirBlockRef<'ctx, 'blk>,
        node: &'ast Node,
    ) -> Result<MlirValue<'ctx, 'blk>> {
        matches!(
            node.kind,
            NodeKind::Or(_)
                | NodeKind::And(_)
                | NodeKind::Lt(_)
                | NodeKind::Gt(_)
                | NodeKind::Lte(_)
                | NodeKind::Gte(_)
                | NodeKind::Eq(_)
                | NodeKind::Neq(_)
                | NodeKind::BitAnd(_)
                | NodeKind::BitOr(_)
                | NodeKind::BitXor(_)
                | NodeKind::Add(_)
                | NodeKind::Sub(_)
                | NodeKind::Mul(_)
                | NodeKind::Div(_)
        );
        let bin_op = match &node.kind {
            NodeKind::Or(bin)
            | NodeKind::And(bin)
            | NodeKind::Lt(bin)
            | NodeKind::Gt(bin)
            | NodeKind::Lte(bin)
            | NodeKind::Gte(bin)
            | NodeKind::Eq(bin)
            | NodeKind::Neq(bin)
            | NodeKind::BitAnd(bin)
            | NodeKind::BitOr(bin)
            | NodeKind::BitXor(bin)
            | NodeKind::Add(bin)
            | NodeKind::Sub(bin)
            | NodeKind::Mul(bin)
            | NodeKind::Div(bin) => bin,
            _ => unreachable!(),
        };
        let lhs = self.gen_expr_reachable(block, &bin_op.lhs)?;
        let rhs = self.gen_expr_reachable(block, &bin_op.rhs)?;

        match &node.kind {
            NodeKind::Or(_)
            | NodeKind::And(_)
            | NodeKind::Lt(_)
            | NodeKind::Gt(_)
            | NodeKind::Lte(_)
            | NodeKind::Gte(_)
            | NodeKind::Eq(_)
            | NodeKind::Neq(_) => self.gen_cmp(block, node, lhs, rhs),
            NodeKind::BitAnd(_) | NodeKind::BitOr(_) | NodeKind::BitXor(_) => {
                self.gen_bitwise(block, node, lhs, rhs)
            }
            NodeKind::Add(_) | NodeKind::Sub(_) | NodeKind::Mul(_) | NodeKind::Div(_) => {
                self.gen_arith(block, node, lhs, rhs)
            }
            _ => unreachable!(),
        }
    }

    fn gen_cmp(
        &mut self,
        block: MlirBlockRef<'ctx, 'blk>,
        node: &'ast Node,
        lhs: MlirValue<'ctx, 'blk>,
        rhs: MlirValue<'ctx, 'blk>,
    ) -> Result<MlirValue<'ctx, 'blk>> {
        matches!(
            node.kind,
            NodeKind::Or(_)
                | NodeKind::And(_)
                | NodeKind::Lt(_)
                | NodeKind::Gt(_)
                | NodeKind::Lte(_)
                | NodeKind::Eq(_)
                | NodeKind::Neq(_)
        );
        let loc = node.loc(self.ctx);
        let i64 = MlirIntegerType::new(self.ctx, 64).into();
        let ret_type = MlirIntegerType::new(self.ctx, 1).into();
        match node.kind {
            NodeKind::Or(_) => {
                // TODO: Type check that lhs and rhs are bools
                let built = OrIOperation::builder(self.ctx, loc)
                    .lhs(lhs)
                    .rhs(rhs)
                    .build();
                let op = built.as_operation();
                Ok(self.create(block, op)?)
            }
            NodeKind::And(_) => {
                // TODO: Type check that lhs and rhs are bools
                let built = AndIOperation::builder(self.ctx, loc)
                    .lhs(lhs)
                    .rhs(rhs)
                    .build();
                let op = built.as_operation();
                Ok(self.create(block, op)?)
            }
            NodeKind::Lt(_) => {
                let built = CmpIOperation::builder(self.ctx, loc)
                    .predicate(MlirIntegerAttribute::new(i64, CmpiPredicate::Ult as i64).into())
                    .lhs(lhs)
                    .rhs(rhs)
                    .result(ret_type)
                    .build();
                let op = built.as_operation();
                Ok(self.create(block, op)?)
            }
            NodeKind::Gt(_) => {
                let built = CmpIOperation::builder(self.ctx, loc)
                    .predicate(MlirIntegerAttribute::new(i64, CmpiPredicate::Ult as i64).into())
                    .lhs(lhs)
                    .rhs(rhs)
                    .result(ret_type)
                    .build();
                let op = built.as_operation();
                Ok(self.create(block, op)?)
            }
            NodeKind::Lte(_) => {
                let built = CmpIOperation::builder(self.ctx, loc)
                    .predicate(MlirIntegerAttribute::new(i64, CmpiPredicate::Ult as i64).into())
                    .lhs(lhs)
                    .rhs(rhs)
                    .result(ret_type)
                    .build();
                let op = built.as_operation();
                Ok(self.create(block, op)?)
            }
            NodeKind::Gte(_) => {
                let built = CmpIOperation::builder(self.ctx, loc)
                    .predicate(MlirIntegerAttribute::new(i64, CmpiPredicate::Ult as i64).into())
                    .lhs(lhs)
                    .rhs(rhs)
                    .result(ret_type)
                    .build();
                let op = built.as_operation();
                Ok(self.create(block, op)?)
            }
            NodeKind::Eq(_) => {
                let built = CmpIOperation::builder(self.ctx, loc)
                    .predicate(MlirIntegerAttribute::new(i64, CmpiPredicate::Ult as i64).into())
                    .lhs(lhs)
                    .rhs(rhs)
                    .result(ret_type)
                    .build();
                let op = built.as_operation();
                Ok(self.create(block, op)?)
            }
            NodeKind::Neq(_) => {
                let built = CmpIOperation::builder(self.ctx, loc)
                    .predicate(MlirIntegerAttribute::new(i64, CmpiPredicate::Ult as i64).into())
                    .lhs(lhs)
                    .rhs(rhs)
                    .result(ret_type)
                    .build();
                let op = built.as_operation();
                Ok(self.create(block, op)?)
            }
            _ => unreachable!(),
        }
    }

    fn gen_arith(
        &mut self,
        block: MlirBlockRef<'ctx, 'blk>,
        node: &'ast Node,
        lhs: MlirValue<'ctx, 'blk>,
        rhs: MlirValue<'ctx, 'blk>,
    ) -> Result<MlirValue<'ctx, 'blk>> {
        matches!(
            node.kind,
            NodeKind::Add(_) | NodeKind::Sub(_) | NodeKind::Mul(_) | NodeKind::Div(_)
        );
        let loc = node.loc(self.ctx);
        match node.kind {
            NodeKind::Add(_) => {
                let built = AddIOperation::builder(self.ctx, loc)
                    .lhs(lhs)
                    .rhs(rhs)
                    .build();
                let op = built.as_operation();
                Ok(self.create(block, op)?)
            }
            NodeKind::Sub(_) => {
                let built = SubIOperation::builder(self.ctx, loc)
                    .lhs(lhs)
                    .rhs(rhs)
                    .build();
                let op = built.as_operation();
                Ok(self.create(block, op)?)
            }
            NodeKind::Mul(_) => {
                let built = MulIOperation::builder(self.ctx, loc)
                    .lhs(lhs)
                    .rhs(rhs)
                    .build();
                let op = built.as_operation();
                Ok(self.create(block, op)?)
            }
            NodeKind::Div(_) => {
                unimplemented!("Signed and unsigned division")
            }
            _ => unreachable!(),
        }
    }

    fn gen_bitwise(
        &mut self,
        block: MlirBlockRef<'ctx, 'blk>,
        node: &'ast Node,
        lhs: MlirValue<'ctx, 'blk>,
        rhs: MlirValue<'ctx, 'blk>,
    ) -> Result<MlirValue<'ctx, 'blk>> {
        matches!(
            node.kind,
            NodeKind::BitAnd(_) | NodeKind::BitOr(_) | NodeKind::BitXor(_)
        );
        let loc = node.loc(self.ctx);
        match node.kind {
            NodeKind::BitAnd(_) => {
                let built = AndIOperation::builder(self.ctx, loc)
                    .lhs(lhs)
                    .rhs(rhs)
                    .build();
                let op = built.as_operation();
                Ok(self.create(block, op)?)
            }
            NodeKind::BitOr(_) => {
                let built = OrIOperation::builder(self.ctx, loc)
                    .lhs(lhs)
                    .rhs(rhs)
                    .build();
                let op = built.as_operation();
                Ok(self.create(block, op)?)
            }
            NodeKind::BitXor(_) => {
                let built = XOrIOperation::builder(self.ctx, loc)
                    .lhs(lhs)
                    .rhs(rhs)
                    .build();
                let op = built.as_operation();
                Ok(self.create(block, op)?)
            }
            _ => unreachable!(),
        }
    }

    // Generates a block. Ensures that the last instruction is a terminator returning void if
    // necessary. It is the callers responsibility to ensure that the terminator type checks.
    fn gen_block(&mut self, block: MlirBlockRef<'ctx, 'blk>, nodes: &'ast [Node]) -> Result<()> {
        for expr in nodes {
            self.gen_expr(block, expr)?;
        }

        if let None = block.terminator() {
            let loc = Location::unknown(self.ctx);
            block.append_operation(r#return(self.ctx, &[], loc).into());
        }

        Ok(())
    }

    fn gen_type(&mut self, node: &'ast Node) -> Result<MlirType<'ctx>> {
        match &node.kind {
            NodeKind::Identifier(ident) => {
                let s = ident.as_str();
                if let Some(ty) = get_maybe_primitive(self.ctx, s) {
                    return Ok(ty);
                }
                unimplemented!()
            }
            _ => unimplemented!(),
        }
    }

    fn get_identifier(
        &mut self,
        _: MlirBlockRef,
        node: &'ast Node,
    ) -> Result<MlirValue<'ctx, 'blk>> {
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

// Type checking methods
impl<'a, 'ast, 'ctx> AstCodegen<'a, 'ast, 'ctx> {
    fn type_check(
        &self,
        expected_type: MlirValue<'ctx, 'a>,
        actual_value: MlirValue<'ctx, 'a>,
    ) -> Result<()> {
        unimplemented!()
    }
}

struct Builder<'ctx, 'blk> {
    block: MlirBlockRef<'ctx, 'blk>,
}

impl<'ctx, 'blk> Builder<'ctx, 'blk> {
    pub fn new(block: MlirBlockRef<'ctx, 'blk>) -> Self {
        Self { block }
    }

    // Set the insertion point of the builder. Returns previous insertion point
    pub fn set_insertion_point(
        &mut self,
        block: MlirBlockRef<'ctx, 'blk>,
    ) -> MlirBlockRef<'ctx, 'blk> {
        std::mem::replace(&mut self.block, block)
    }

    pub fn create(
        &mut self,
        loc: Location<'ctx>,
        name: &str,
        operands: &[MlirValue<'ctx, '_>],
        results: &[MlirType<'ctx>],
        attributes: &[(MlirIdentifier<'ctx>, MlirAttribute<'ctx>)],
        successors: &[&MlirBlock<'ctx>],
    ) -> Result<MlirOperationRef> {
        let state = OperationBuilder::new(name, loc)
            .add_operands(operands)
            .add_results(results)
            .add_attributes(attributes)
            .add_successors(successors);
        let op = state.build()?;
        Ok(self.block.append_operation(op))
    }
}

fn get_maybe_primitive<'ctx>(context: &'ctx Context, s: &str) -> Option<MlirType<'ctx>> {
    let bytes = s.as_bytes();
    // TODO: Use TaraType to keep signedness info
    if bytes[0] == b'u' {
        let size = u16::from_str_radix(&s[1..], 10).ok()?;
        let int_type = MlirIntegerType::new(context, size.into());
        Some(int_type.into())
    } else if bytes[0] == b'i' {
        let size = u16::from_str_radix(&s[1..], 10).ok()?;
        let int_type = MlirIntegerType::new(context, size.into());
        Some(int_type.into())
    } else {
        None
    }
}
