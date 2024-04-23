mod error;
mod table;

use crate::{
    ast::{Node, NodeKind},
    ast_codegen::{error::Error, table::Table},
    types::Type as TaraType,
    utils::RRC,
    Ast,
};
use anyhow::Result;
use melior::{
    dialect::{
        arith::CmpiPredicate,
        ods::{
            arith::{
                AddIOperation, AndIOperation, CmpIOperation, ExtSIOperation, ExtUIOperation,
                MulIOperation, OrIOperation, SubIOperation, XOrIOperation,
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

pub struct AstCodegen<'a, 'ast, 'ctx> {
    ast: &'ast Ast,
    ctx: &'ctx Context,
    module: &'a MlirModule<'ctx>,
    builder: Builder<'ctx, 'a>,
    table: Table<'ctx, 'ast>,
}

impl<'a, 'ast, 'ctx> AstCodegen<'a, 'ast, 'ctx> {
    pub fn new(ast: &'ast Ast, ctx: &'ctx Context, module: &'a MlirModule<'ctx>) -> Self {
        Self {
            ast,
            ctx,
            module,
            builder: Builder::new(module.body()),
            table: Table::new(),
        }
    }

    fn create<'blk, T: Into<MlirOperation<'ctx>> + Clone>(
        &mut self,
        block: MlirBlockRef<'ctx, 'blk>,
        operation: &T,
    ) -> Result<MlirValue<'ctx, 'ctx>> {
        let op: T = operation.clone();
        let op_ref = block.append_operation(op.into());
        self.value_from_ref(op_ref)
    }

    fn value_from_ref<'blk>(&self, op_ref: MlirOperationRef) -> Result<MlirValue<'ctx, 'ctx>> {
        let result = op_ref.result(0)?;
        let raw_value = result.to_raw();
        Ok(unsafe { MlirValue::from_raw(raw_value) })
    }

    // Pushes a layer onto all tables
    fn push(&mut self) {
        self.table.push();
    }

    // Pops a layer from all tables
    fn pop(&mut self) {
        self.table.pop();
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
                NodeKind::VarDecl(v_d) => self.table.define_name(v_d.ident, member)?,
                NodeKind::SubroutineDecl(s_d) => self.table.define_name(s_d.ident, member)?,
                _ => unreachable!(),
            }
        }

        for member in members {
            match &member.kind {
                NodeKind::VarDecl(v_d) => {
                    let val = self.gen_var_decl(&member)?;
                    self.table.define_symbol(v_d.ident, val);
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

        let mut value = self.gen_expr_reachable(block, &var_decl.expr)?;

        if let Some(ty_expr) = &var_decl.ty {
            self.gen_type(ty_expr)?;
            let expected_type = self.table.get_type(ty_expr)?;
            value = self.cast(block, &var_decl.expr, expected_type.clone())?;
            self.table.define_typed_value(node, expected_type, value);
        } else {
            // TODO: infer value type
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
        let mlir_return_type = self.table.get_mlir_type(self.ctx, return_type.clone())?;

        let loc = node.loc(self.ctx);
        let body = self.module.body();
        let fn_name = {
            let block = MlirBlock::new(&[]);

            let mut param_types = Vec::new();
            for param in &fn_decl.params {
                let param_ty = param.ty.as_ref();
                self.table.define_name(param.name, param_ty)?;
                let param_type = self.gen_type(param_ty)?;
                let mlir_param_type = self.table.get_mlir_type(self.ctx, param_type)?;
                param_types.push(mlir_param_type);

                let arg = block.add_argument(mlir_param_type, param.ty.loc(self.ctx));
                self.table.define_symbol(param.name, arg);
                self.table.define_value(param_ty, arg);
            }

            let region = MlirRegion::new();
            let block_ref = region.append_block(block);

            self.gen_block(block_ref, &fn_decl.block)?;
            assert!(block_ref.terminator().is_some());

            // TODO: This assumes that all blocks terminate at their last instruction which might
            // not be true
            if let Some(last_node) = fn_decl.block.last() {
                let return_expr = match &last_node.kind {
                    NodeKind::Return(Some(return_expr)) => &return_expr.lhs,
                    _ => unreachable!(),
                };
                let curr_return_type = self.table.get_type(return_expr)?;
                if curr_return_type != return_type {
                    unimplemented!("TODO: Cast return value of block");
                    /*
                    let mut owned_block = unsafe { MlirBlock::from_raw(block_ref.to_raw()) };
                    let mut terminator = owned_block.terminator_mut().unwrap();
                    let return_value = self.cast(block_ref, return_expr, &return_type)?;
                    let return_op = ReturnOperation::builder(self.ctx, loc)
                        .operands(&[return_value])
                        .build();
                    block_ref.insert_operation_before(terminator, return_op.into());
                    terminator.remove_from_parent();
                    */
                }
            } else {
                if *return_type.borrow() != TaraType::Void {
                    Err(Error::new(
                        node.span,
                        "Body returns void but return type is not void".to_string(),
                    ))?;
                }
            }

            let fn_type = MlirFunctionType::new(self.ctx, &param_types, &[mlir_return_type]).into();
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

        self.table.define_fn(fn_decl.ident, fn_name);

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
        let loc = node.loc(self.ctx);
        let return_op = match &node.kind {
            NodeKind::Return(Some(e)) => {
                let return_value = self.gen_expr_reachable(block, &e.lhs)?;
                let return_ty = self.table.get_type(&e.lhs)?;
                self.table.define_type(node, return_ty);
                ReturnOperation::builder(self.ctx, loc)
                    .operands(&[return_value])
                    .build()
            }
            NodeKind::Return(None) => {
                self.table.define_type(node, TaraType::Void);
                ReturnOperation::builder(self.ctx, loc)
                    .operands(&[])
                    .build()
            }
            _ => unreachable!(),
        };

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
            | NodeKind::Gte(_)
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
        let value = self.gen_expr(block, node)?.ok_or_else(|| {
            Error::new(
                node.span,
                "Expected reachable value, control flow unexpectedly diverted".to_string(),
            )
        })?;
        self.table.define_value(node, value);
        Ok(value)
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
        self.gen_expr_reachable(block, &bin_op.lhs)?;
        self.gen_expr_reachable(block, &bin_op.rhs)?;

        match &node.kind {
            NodeKind::Or(_)
            | NodeKind::And(_)
            | NodeKind::Lt(_)
            | NodeKind::Gt(_)
            | NodeKind::Lte(_)
            | NodeKind::Gte(_)
            | NodeKind::Eq(_)
            | NodeKind::Neq(_) => self.gen_cmp(block, node),
            NodeKind::BitAnd(_) | NodeKind::BitOr(_) | NodeKind::BitXor(_) => {
                self.gen_bitwise(block, node)
            }
            NodeKind::Add(_) | NodeKind::Sub(_) | NodeKind::Mul(_) | NodeKind::Div(_) => {
                self.gen_arith(block, node)
            }
            _ => unreachable!(),
        }
    }

    fn gen_cmp(
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
        );
        let loc = node.loc(self.ctx);
        let (lhs_node, rhs_node) = match &node.kind {
            NodeKind::Or(binop)
            | NodeKind::And(binop)
            | NodeKind::Lt(binop)
            | NodeKind::Gt(binop)
            | NodeKind::Lte(binop)
            | NodeKind::Gte(binop)
            | NodeKind::Eq(binop)
            | NodeKind::Neq(binop) => (&binop.lhs, &binop.rhs),
            _ => unreachable!(),
        };
        let lhs = self.table.get_value(lhs_node)?;
        let lhs_type = self.table.get_type(lhs_node)?;
        // Type check lhs and rhs
        let rhs = match &node.kind {
            NodeKind::Or(_) | NodeKind::And(_) => {
                self.expect_bool_type(lhs_node)?;
                self.expect_bool_type(rhs_node)?;
                self.table.get_value(rhs_node)?
            }
            NodeKind::Lt(_) | NodeKind::Gt(_) | NodeKind::Lte(_) | NodeKind::Gte(_) => {
                self.expect_integral_type(lhs_node)?;
                self.expect_integral_type(rhs_node)?;
                self.cast(block, rhs_node, lhs_type)?
            }
            NodeKind::Eq(_) | NodeKind::Neq(_) => {
                self.expect_integral_type(lhs_node)
                    .or(self.expect_bool_type(lhs_node))?;
                self.expect_integral_type(rhs_node)
                    .or(self.expect_bool_type(rhs_node))?;
                self.cast(block, rhs_node, lhs_type)?
            }
            _ => unreachable!(),
        };

        let i64_ty = TaraType::IntUnsigned { width: 64 };
        let predicate_i64 = self.table.get_mlir_type(self.ctx, i64_ty)?;
        let bool_ty = TaraType::Bool;
        let ret_type = self.table.get_mlir_type(self.ctx, bool_ty.clone())?;
        self.table.define_type(node, bool_ty);

        match node.kind {
            NodeKind::Or(_) => {
                let built = OrIOperation::builder(self.ctx, loc)
                    .lhs(lhs)
                    .rhs(rhs)
                    .build();
                let op = built.as_operation();
                Ok(self.create(block, op)?)
            }
            NodeKind::And(_) => {
                let built = AndIOperation::builder(self.ctx, loc)
                    .lhs(lhs)
                    .rhs(rhs)
                    .build();
                let op = built.as_operation();
                Ok(self.create(block, op)?)
            }
            NodeKind::Lt(_) => {
                let built = CmpIOperation::builder(self.ctx, loc)
                    .predicate(
                        MlirIntegerAttribute::new(predicate_i64, CmpiPredicate::Ult as i64).into(),
                    )
                    .lhs(lhs)
                    .rhs(rhs)
                    .result(ret_type)
                    .build();
                let op = built.as_operation();
                Ok(self.create(block, op)?)
            }
            NodeKind::Gt(_) => {
                let built = CmpIOperation::builder(self.ctx, loc)
                    .predicate(
                        MlirIntegerAttribute::new(predicate_i64, CmpiPredicate::Ult as i64).into(),
                    )
                    .lhs(lhs)
                    .rhs(rhs)
                    .result(ret_type)
                    .build();
                let op = built.as_operation();
                Ok(self.create(block, op)?)
            }
            NodeKind::Lte(_) => {
                let built = CmpIOperation::builder(self.ctx, loc)
                    .predicate(
                        MlirIntegerAttribute::new(predicate_i64, CmpiPredicate::Ult as i64).into(),
                    )
                    .lhs(lhs)
                    .rhs(rhs)
                    .result(ret_type)
                    .build();
                let op = built.as_operation();
                Ok(self.create(block, op)?)
            }
            NodeKind::Gte(_) => {
                let built = CmpIOperation::builder(self.ctx, loc)
                    .predicate(
                        MlirIntegerAttribute::new(predicate_i64, CmpiPredicate::Ult as i64).into(),
                    )
                    .lhs(lhs)
                    .rhs(rhs)
                    .result(ret_type)
                    .build();
                let op = built.as_operation();
                Ok(self.create(block, op)?)
            }
            NodeKind::Eq(_) => {
                let built = CmpIOperation::builder(self.ctx, loc)
                    .predicate(
                        MlirIntegerAttribute::new(predicate_i64, CmpiPredicate::Ult as i64).into(),
                    )
                    .lhs(lhs)
                    .rhs(rhs)
                    .result(ret_type)
                    .build();
                let op = built.as_operation();
                Ok(self.create(block, op)?)
            }
            NodeKind::Neq(_) => {
                let built = CmpIOperation::builder(self.ctx, loc)
                    .predicate(
                        MlirIntegerAttribute::new(predicate_i64, CmpiPredicate::Ult as i64).into(),
                    )
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
    ) -> Result<MlirValue<'ctx, 'blk>> {
        matches!(
            node.kind,
            NodeKind::Add(_) | NodeKind::Sub(_) | NodeKind::Mul(_) | NodeKind::Div(_)
        );
        let loc = node.loc(self.ctx);
        let (lhs_node, rhs_node) = match &node.kind {
            NodeKind::Add(binop)
            | NodeKind::Sub(binop)
            | NodeKind::Mul(binop)
            | NodeKind::Div(binop) => (&binop.lhs, &binop.rhs),
            _ => unreachable!(),
        };
        let lhs = self.table.get_value(lhs_node)?;
        let lhs_type = self.table.get_type(lhs_node)?;
        self.expect_integral_type(&lhs_node)?;

        let rhs = self.cast(block, rhs_node, lhs_type.clone())?;

        self.table.define_type(node, lhs_type);

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
    ) -> Result<MlirValue<'ctx, 'blk>> {
        matches!(
            node.kind,
            NodeKind::BitAnd(_) | NodeKind::BitOr(_) | NodeKind::BitXor(_)
        );
        let loc = node.loc(self.ctx);
        let (lhs_node, rhs_node): (&Node, &Node) = match &node.kind {
            NodeKind::BitAnd(binop) | NodeKind::BitOr(binop) | NodeKind::BitXor(binop) => {
                (binop.lhs.as_ref(), binop.rhs.as_ref())
            }
            _ => unreachable!(),
        };
        let lhs = self.table.get_value(lhs_node)?;
        let lhs_type = self.table.get_type(lhs_node)?;
        self.expect_integral_type(&lhs_node)?;

        let rhs = self.cast(block, rhs_node, lhs_type.clone())?;

        self.table.define_type(node, lhs_type);

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
        for (i, expr) in nodes.iter().enumerate() {
            if let None = self.gen_expr(block, expr)? {
                if i != (nodes.len() - 1) {
                    Err(Error::new(
                        nodes[i + 1].span,
                        "Unreachable statement".to_string(),
                    ))?;
                }
            }
        }

        if let None = block.terminator() {
            let loc = Location::unknown(self.ctx);
            block.append_operation(r#return(self.ctx, &[], loc).into());
        }

        Ok(())
    }

    fn gen_type(&mut self, node: &'ast Node) -> Result<RRC<TaraType>> {
        let ty = match &node.kind {
            NodeKind::Identifier(ident) => {
                let s = ident.as_str();
                if let Some(ty) = get_maybe_primitive(s) {
                    ty
                } else {
                    unimplemented!()
                }
            }
            _ => unimplemented!(),
        };
        self.table.define_type(node, ty.clone());
        Ok(RRC::new(ty))
    }

    fn get_identifier(
        &mut self,
        _: MlirBlockRef,
        node: &'ast Node,
    ) -> Result<MlirValue<'ctx, 'blk>> {
        self.table.get_identifier_value(node)
    }
}

// Type checking methods
impl<'a, 'ast, 'ctx, 'blk> AstCodegen<'a, 'ast, 'ctx> {
    fn expect_integral_type(&self, node: &Node) -> Result<()> {
        let actual_type = self.table.get_type(node)?;
        let actual_type_borrowed = actual_type.borrow();
        match *actual_type_borrowed {
            TaraType::IntSigned { .. } | TaraType::IntUnsigned { .. } => Ok(()),
            _ => Err(Error::new(
                node.span,
                format!("Expected integral type, found {}", actual_type),
            ))?,
        }
    }

    fn expect_bool_type(&self, node: &Node) -> Result<()> {
        let actual_type = self.table.get_type(node)?;
        let actual_type_borrowed = actual_type.borrow();
        match *actual_type_borrowed {
            TaraType::Bool => Ok(()),
            _ => Err(Error::new(
                node.span,
                format!("Expected bool type, found {}", actual_type),
            ))?,
        }
    }

    fn cast(
        &mut self,
        block: MlirBlockRef<'ctx, 'blk>,
        node: &Node,
        expected_type: RRC<TaraType>,
    ) -> Result<MlirValue<'ctx, 'ctx>> {
        let expected_type_borrow = expected_type.borrow();
        let actual_type = self.table.get_type(node)?;
        let actual_type_borrow = actual_type.borrow();
        if actual_type == expected_type {
            return self.table.get_value(node);
        }
        match (&*expected_type_borrow, &*actual_type_borrow) {
            (
                TaraType::IntSigned { width: exp_width },
                TaraType::IntSigned { width: act_width },
            ) => {
                if exp_width > act_width {
                    let actual_mlir_value = self.table.get_value(node)?;
                    let expected_mlir_type =
                        self.table.get_mlir_type(self.ctx, expected_type.clone())?;
                    let built = ExtSIOperation::builder(self.ctx, node.loc(self.ctx))
                        .r#in(actual_mlir_value)
                        .out(expected_mlir_type)
                        .build();
                    let op = built.as_operation();
                    Ok(self.create(block, op)?)
                } else {
                    Err(Error::new(
                        node.span,
                        format!("Expected {}, found {}", expected_type, actual_type),
                    ))?
                }
            }
            (
                TaraType::IntUnsigned { width: exp_width },
                TaraType::IntUnsigned { width: act_width },
            ) => {
                if exp_width > act_width {
                    let actual_mlir_value = self.table.get_value(node)?;
                    let expected_mlir_type =
                        self.table.get_mlir_type(self.ctx, expected_type.clone())?;
                    let built = ExtUIOperation::builder(self.ctx, node.loc(self.ctx))
                        .r#in(actual_mlir_value)
                        .out(expected_mlir_type)
                        .build();
                    let op = built.as_operation();
                    Ok(self.create(block, op)?)
                } else {
                    Err(Error::new(
                        node.span,
                        format!("Expected {}, found {}", expected_type, actual_type),
                    ))?
                }
            }
            _ => Err(Error::new(
                node.span,
                format!(
                    "TODO: Unhandled cast from {} to {}",
                    actual_type, expected_type
                ),
            ))?,
        }
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

fn get_maybe_primitive(s: &str) -> Option<TaraType> {
    let bytes = s.as_bytes();
    if bytes[0] == b'u' {
        let size = u16::from_str_radix(&s[1..], 10).ok()?;
        let int_type = TaraType::IntUnsigned { width: size };
        Some(int_type.into())
    } else if bytes[0] == b'i' {
        let size = u16::from_str_radix(&s[1..], 10).ok()?;
        let int_type = TaraType::IntSigned { width: size };
        Some(int_type.into())
    } else if s == "bool" {
        Some(TaraType::Bool)
    } else if s == "void" {
        Some(TaraType::Void)
    } else {
        None
    }
}
