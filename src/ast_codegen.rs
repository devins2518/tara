mod error;
mod table;

use crate::{
    ast::{Node, NodeKind},
    ast_codegen::{error::Error, table::Table},
    circt::{
        arc::{
            CallOperation as HwProcCallOperation, DefineOperation as HwProcDefineOperation,
            OutputOperation as HwProcReturnOperation,
        },
        comb::{
            ICmpOperation as HwCmpOperation, MulOperation as HwMulOperation,
            OrOperation as HwOrOperation, SubOperation as HwSubOperation,
            XorOperation as HwXorOperation,
        },
        hw::{HWModuleOperation as HwModuleOperation, OutputOperation as HwOutputOperation},
        CombICmpPredicate,
    },
    module::file::File,
    types::{ModuleInfo, NamedType, Type as TaraType},
    utils::RRC,
    Ast,
};
use anyhow::Result;
use codespan_reporting::{
    diagnostic::{Diagnostic, Label},
    term::{
        self,
        termcolor::{ColorChoice, StandardStream},
    },
};
use indexmap::map::{Entry, IndexMap};
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
            ArrayAttribute as MlirArrayAttribute,
            FlatSymbolRefAttribute as MlirFlatSymbolRefAttribute,
            IntegerAttribute as MlirIntegerAttribute, StringAttribute as MlirStringAttribute,
            TypeAttribute as MlirTypeAttribute,
        },
        operation::{
            Operation as MlirOperation, OperationBuilder, OperationRef as MlirOperationRef,
        },
        r#type::FunctionType as MlirFunctionType,
        Attribute as MlirAttribute, Block as MlirBlock, BlockRef as MlirBlockRef,
        Identifier as MlirIdentifier, Location, Module as MlirModule, Region as MlirRegion,
        Type as MlirType, Value as MlirValue, ValueLike,
    },
    Context,
};
use std::collections::HashMap;
use symbol_table::GlobalSymbol;

pub struct AstCodegen<'a, 'ast, 'ctx> {
    ast: &'ast Ast,
    ctx: &'ctx Context,
    module: &'a MlirModule<'ctx>,
    builder: Builder<'ctx, 'a>,
    table: Table<'ctx, 'ast>,
    pub errors: Vec<anyhow::Error>,
    anon_cnt: usize,
}

impl<'a, 'ast, 'ctx> AstCodegen<'a, 'ast, 'ctx> {
    pub fn new(ast: &'ast Ast, ctx: &'ctx Context, module: &'a MlirModule<'ctx>) -> Self {
        Self {
            ast,
            ctx,
            module,
            builder: Builder::new(module.body()),
            table: Table::new(),
            errors: Vec::new(),
            anon_cnt: 0,
        }
    }

    // Pushes a layer onto all tables
    fn push(&mut self) {
        self.table.push();
    }

    // Pops a layer from all tables
    fn pop(&mut self) {
        self.table.pop();
    }

    pub fn report_errors(&self, file: &File) -> Result<()> {
        if self.errors.len() > 0 {
            for error in &self.errors {
                if let Some(err) = error.downcast_ref::<Error>() {
                    let label = Label::primary((), err.span);
                    let diagnostic = Diagnostic::error()
                        .with_message(err.reason.clone())
                        .with_labels(vec![label]);

                    let writer = StandardStream::stdout(ColorChoice::Always);
                    let config = codespan_reporting::term::Config::default();

                    term::emit(&mut writer.lock(), &config, file, &diagnostic)?;
                } else {
                    println!("{}", error);
                }
            }
            anyhow::bail!("compilation errors!")
        }
        Ok(())
    }

    fn anon(&mut self) -> usize {
        let anon = self.anon_cnt;
        self.anon_cnt += 1;
        anon
    }
}

// MLIR generation methods
impl<'a, 'ast, 'ctx, 'blk> AstCodegen<'a, 'ast, 'ctx>
where
    'a: 'blk,
{
    pub fn gen_root(&mut self) -> Result<()> {
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
        let members = match &node.kind {
            NodeKind::StructDecl(s) => &s.members,
            NodeKind::ModuleDecl(m) => &m.members,
            _ => unreachable!(),
        };
        // Members will be either a VarDecl or a SubroutineDecl
        for member in members {
            match &member.kind {
                NodeKind::VarDecl(v_d) => self.table.define_name(v_d.ident, &v_d.expr)?,
                NodeKind::SubroutineDecl(s_d) => self.table.define_name(s_d.ident, member)?,
                _ => unreachable!(),
            }
        }

        for member in members {
            match &member.kind {
                NodeKind::VarDecl(v_d) => match self.gen_var_decl(&member) {
                    Ok(Some(val)) => self.table.define_symbol(v_d.ident, val),
                    Ok(None) => {}
                    Err(e) => {
                        self.pop();
                        self.errors.push(e);
                    }
                },
                NodeKind::SubroutineDecl(_) => match subroutine_gen(self, &member) {
                    Ok(_) => {}
                    Err(e) => {
                        println!("got error");
                        self.pop();
                        self.errors.push(e);
                    }
                },
                _ => unreachable!(),
            }
        }

        Ok(())
    }

    fn gen_struct_decl(&mut self, node: &'ast Node) -> Result<()> {
        self.push();
        self._gen_container(node, Self::gen_fn_decl)?;
        self.pop();
        Ok(())
    }

    fn gen_module_decl(&mut self, node: &'ast Node) -> Result<()> {
        matches!(node.kind, NodeKind::ModuleDecl(_));
        self.push();
        let mod_decl = match &node.kind {
            NodeKind::ModuleDecl(m_d) => m_d,
            _ => unreachable!(),
        };
        let loc = node.loc(self.ctx);
        let prev_context = self.builder.save(SurroundingContext::Hw);

        let region = MlirRegion::new();
        let body = region.append_block(MlirBlock::new(&[]));
        self.builder.set_block(body);

        // Here we'll generate all of the inputs to the module by analyzing each comb.
        // TODO: This should only be done for method combs (i.e. @This() is first parameter), all
        // other combs should be free
        {
            let mut ins: IndexMap<GlobalSymbol, (RRC<TaraType>, &Node)> = IndexMap::new();
            let mut outs = Vec::new();
            for member in &mod_decl.members {
                match &member.kind {
                    NodeKind::SubroutineDecl(s_d) => {
                        let ret_ty = self.gen_type(&s_d.return_type)?;
                        if *ret_ty.borrow() == TaraType::Void {
                            continue;
                        }
                        outs.push(NamedType {
                            name: s_d.ident,
                            ty: self.gen_type(&s_d.return_type)?,
                        });
                        for param in &s_d.params {
                            let param_ty = param.ty.as_ref();
                            let param_type = self.gen_type(param_ty)?;

                            match ins.entry(param.name) {
                                Entry::Occupied(entry) => {
                                    if entry.get().0 != param_type {
                                        Err(Error::new(
                                            member.span,
                                            "Comb params have different types!".to_string(),
                                        ))?;
                                    }
                                }
                                Entry::Vacant(entry) => _ = entry.insert((param_type, param_ty)),
                            }
                        }
                    }
                    _ => {}
                }
            }

            // And now that we have the reduced set of inputs, we can generate module inputs
            let ins = {
                let mut v = Vec::new();
                for (name, ty_and_node) in ins {
                    let param_type = ty_and_node.0;
                    let param_node = ty_and_node.1;
                    self.table.define_name(name, param_node)?;
                    let mlir_param_type = self.table.get_mlir_type(self.ctx, param_type.clone())?;
                    let arg = self
                        .builder
                        .add_argument(mlir_param_type, param_node.loc(self.ctx));
                    self.table.define_symbol(name, arg);
                    self.table
                        .define_typed_value(param_node, param_type.clone(), arg);
                    v.push(NamedType {
                        name,
                        ty: param_type,
                    });
                }
                v
            };
            let mod_info = ModuleInfo { ins, outs };
            self.table.define_type(node, TaraType::Module(mod_info));
        }

        let mod_type = self.table.get_mlir_type_node(self.ctx, node)?;

        self._gen_container(node, Self::gen_comb_decl)?;

        {
            let mut outs = Vec::new();
            for member in &mod_decl.members {
                match &member.kind {
                    NodeKind::SubroutineDecl(s_d) => {
                        let ret_ty = self.gen_type(&s_d.return_type)?;
                        if *ret_ty.borrow() == TaraType::Void {
                            continue;
                        }
                        let val = self.table.get_value(member)?;
                        outs.push(val);
                    }
                    _ => {}
                }
            }
            let output_op = HwOutputOperation::builder(self.ctx, loc)
                .outputs(&outs)
                .build();
            body.append_operation(output_op.into());
        }

        let name = if let Some(name) = self.table.get_name(node) {
            name
        } else {
            let name = GlobalSymbol::new(format!("anon_module{}", self.anon()));
            self.table.define_name(name, node)?;
            name
        };
        let builder = HwModuleOperation::builder(self.ctx, loc)
            .body(region)
            .sym_name(MlirStringAttribute::new(self.ctx, name.as_str()))
            .module_type(MlirTypeAttribute::new(mod_type))
            .parameters(MlirArrayAttribute::new(self.ctx, &[]))
            .build();
        let operation: MlirOperation = builder.into();

        self.module.body().append_operation(operation);

        self.builder.restore(prev_context);
        self.pop();
        Ok(())
    }

    fn gen_var_decl(&mut self, node: &'ast Node) -> Result<Option<MlirValue<'ctx, 'blk>>> {
        matches!(node.kind, NodeKind::VarDecl(_));

        let var_decl = match &node.kind {
            NodeKind::VarDecl(v_d) => v_d,
            _ => unreachable!(),
        };
        let block = self.module.body();
        let prev_context = self.builder.save_with_block(SurroundingContext::Sw, block);

        if let Some(mut value) = self.gen_expr(&var_decl.expr)? {
            if let Some(ty_expr) = &var_decl.ty {
                self.gen_type(ty_expr)?;
                let expected_type = self.table.get_type(ty_expr)?;
                value = self.cast(&var_decl.expr, expected_type.clone())?;
                self.table.define_typed_value(node, expected_type, value);
            } else {
                // TODO: infer value type
            }
            Ok(Some(value))
        } else {
            Ok(None)
        }
    }

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
        let region = MlirRegion::new();
        let block = MlirBlock::new(&[]);
        let block_ref = region.append_block(block);
        let prev_context = self
            .builder
            .save_with_block(SurroundingContext::Sw, block_ref);
        self.builder.block_ret_ty = Some(return_type.clone());

        let fn_name = {
            let mut param_types = Vec::new();
            for param in &fn_decl.params {
                let param_ty = param.ty.as_ref();
                self.table.define_name(param.name, param_ty)?;
                let param_type = self.gen_type(param_ty)?;
                let mlir_param_type = self.table.get_mlir_type(self.ctx, param_type.clone())?;
                param_types.push(mlir_param_type);

                let arg = self
                    .builder
                    .add_argument(mlir_param_type, param.ty.loc(self.ctx));
                self.table.define_symbol(param.name, arg);
                self.table.define_typed_value(param_ty, param_type, arg);
            }

            self.gen_block_terminated(&fn_decl.block)?;

            let mlir_fn_ret_type = if *return_type.borrow() != TaraType::Void {
                Some(mlir_return_type)
            } else {
                None
            };
            let fn_type =
                MlirFunctionType::new(self.ctx, &param_types, mlir_fn_ret_type.as_slice()).into();
            let builder = FuncOperation::builder(self.ctx, loc)
                .body(region)
                .sym_name(MlirStringAttribute::new(self.ctx, fn_decl.ident.as_str()))
                .function_type(MlirTypeAttribute::new(fn_type));
            let func = builder.build();
            let fn_name = MlirFlatSymbolRefAttribute::new(self.ctx, func.sym_name()?.value());
            self.module.body().append_operation(func.into());
            fn_name
        };

        self.pop();
        self.builder.restore(prev_context);

        self.table.define_fn(fn_decl.ident, fn_name);

        Ok(())
    }

    fn gen_comb_decl(&mut self, node: &'ast Node) -> Result<()> {
        matches!(node.kind, NodeKind::SubroutineDecl(_));
        let comb_decl = match &node.kind {
            NodeKind::SubroutineDecl(s_d) => s_d,
            _ => unreachable!(),
        };
        self.push();

        let return_type = self.gen_type(&comb_decl.return_type)?;
        // Don't emit an arc that returns void as all computation can be culled
        if *return_type.borrow() == TaraType::Void {
            return Ok(());
        }
        let mlir_return_type = self.table.get_mlir_type(self.ctx, return_type.clone())?;

        let loc = node.loc(self.ctx);
        let body = self.module.body();
        let comb_name = {
            let region = MlirRegion::new();
            let block = MlirBlock::new(&[]);
            let block_ref = region.append_block(block);
            let prev_context = self
                .builder
                .save_with_block(SurroundingContext::Hw, block_ref);
            self.builder.block_ret_ty = Some(return_type.clone());

            let mut param_types = Vec::new();
            for param in &comb_decl.params {
                let param_ty = param.ty.as_ref();
                let param_type = self.gen_type(param_ty)?;
                let mlir_param_type = self.table.get_mlir_type(self.ctx, param_type.clone())?;
                param_types.push(mlir_param_type);

                let arg = self
                    .builder
                    .add_argument(mlir_param_type, param.ty.loc(self.ctx));
                self.table.define_symbol(param.name, arg);
                self.table.define_typed_value(param_ty, param_type, arg);
            }

            self.gen_block_terminated(&comb_decl.block)?;

            let comb_type =
                MlirFunctionType::new(self.ctx, &param_types, &[mlir_return_type]).into();
            let builder = HwProcDefineOperation::builder(self.ctx, loc)
                .body(region)
                .sym_name(MlirStringAttribute::new(self.ctx, comb_decl.ident.as_str()))
                .function_type(MlirTypeAttribute::new(comb_type));
            let comb = builder.build();
            let comb_name = MlirFlatSymbolRefAttribute::new(self.ctx, comb_decl.ident.as_str());
            let operation: MlirOperation = comb.into();
            body.append_operation(operation);
            self.builder.restore(prev_context);
            comb_name
        };

        self.pop();

        let mut ins = Vec::new();
        for param in &comb_decl.params {
            let real_param_node = self.table.get_node(param.name);
            let param_val = self.table.get_value(real_param_node)?;
            ins.push(param_val);
        }
        let call_op = HwProcCallOperation::builder(self.ctx, loc)
            .arc(comb_name)
            .inputs(&ins)
            .outputs(&[mlir_return_type])
            .build();
        let call_val = self
            .builder
            .insert_operation(call_op.as_operation().clone())?;

        self.table.define_fn(comb_decl.ident, comb_name);
        self.table.define_typed_value(node, return_type, call_val);

        Ok(())
    }

    fn gen_return(&mut self, node: &'ast Node) -> Result<()> {
        matches!(node.kind, NodeKind::Return(_));
        let loc = node.loc(self.ctx);
        match &node.kind {
            NodeKind::Return(Some(e)) => {
                let return_value = self.gen_expr_reachable(&e.lhs)?;
                let return_ty = self.table.get_type(&e.lhs)?;
                self.table.define_typed_value(node, return_ty, return_value);
                let casted_value = self.cast(node, self.builder.ret_ty())?;
                self.builder.gen_return(self.ctx, loc, Some(casted_value))?;
            }
            NodeKind::Return(None) => {
                self.table.define_type(node, TaraType::Void);
                self.builder.gen_return(self.ctx, loc, None)?;
            }
            _ => unreachable!(),
        };

        assert!(self.builder.block.terminator().is_some());
        Ok(())
    }

    fn gen_expr(&mut self, node: &'ast Node) -> Result<Option<MlirValue<'ctx, 'blk>>> {
        let maybe_val = match &node.kind {
            NodeKind::StructDecl(_) => {
                self.gen_struct_decl(node)?;
                None
            }
            NodeKind::ModuleDecl(_) => {
                self.gen_module_decl(node)?;
                None
            }
            NodeKind::Return(_) => {
                self.gen_return(node)?;
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
            | NodeKind::Div(_) => Some(self.gen_bin_op(node)?),
            NodeKind::Identifier(_) => Some(self.get_identifier(node)?),
            _ => unimplemented!(),
        };
        Ok(maybe_val)
    }

    fn gen_expr_reachable(&mut self, node: &'ast Node) -> Result<MlirValue<'ctx, 'blk>> {
        let value = self.gen_expr(node)?.ok_or_else(|| {
            Error::new(
                node.span,
                "Expected reachable value, control flow unexpectedly diverted".to_string(),
            )
        })?;
        self.table.define_value(node, value);
        Ok(value)
    }

    fn gen_bin_op(&mut self, node: &'ast Node) -> Result<MlirValue<'ctx, 'blk>> {
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
        self.gen_expr_reachable(&bin_op.lhs)?;
        self.gen_expr_reachable(&bin_op.rhs)?;

        match &node.kind {
            NodeKind::Or(_)
            | NodeKind::And(_)
            | NodeKind::Lt(_)
            | NodeKind::Gt(_)
            | NodeKind::Lte(_)
            | NodeKind::Gte(_)
            | NodeKind::Eq(_)
            | NodeKind::Neq(_) => self.gen_cmp(node),
            NodeKind::BitAnd(_) | NodeKind::BitOr(_) | NodeKind::BitXor(_) => {
                self.gen_bitwise(node)
            }
            NodeKind::Add(_) | NodeKind::Sub(_) | NodeKind::Mul(_) | NodeKind::Div(_) => {
                self.gen_arith(node)
            }
            _ => unreachable!(),
        }
    }

    fn gen_cmp(&mut self, node: &'ast Node) -> Result<MlirValue<'ctx, 'blk>> {
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
                self.cast(rhs_node, lhs_type)?
            }
            NodeKind::Eq(_) | NodeKind::Neq(_) => {
                self.expect_integral_type(lhs_node)
                    .or(self.expect_bool_type(lhs_node))?;
                self.expect_integral_type(rhs_node)
                    .or(self.expect_bool_type(rhs_node))?;
                self.cast(rhs_node, lhs_type)?
            }
            _ => unreachable!(),
        };

        let bool_ty = TaraType::Bool;
        self.table.define_type(node, bool_ty);

        match node.kind {
            NodeKind::Or(_) => self.builder.gen_log_or(self.ctx, loc, lhs, rhs),
            NodeKind::And(_) => self.builder.gen_log_and(self.ctx, loc, lhs, rhs),
            NodeKind::Lt(_) => self.builder.gen_log_lt(self.ctx, loc, lhs, rhs),
            NodeKind::Gt(_) => self.builder.gen_log_gt(self.ctx, loc, lhs, rhs),
            NodeKind::Lte(_) => self.builder.gen_log_lte(self.ctx, loc, lhs, rhs),
            NodeKind::Gte(_) => self.builder.gen_log_gte(self.ctx, loc, lhs, rhs),
            NodeKind::Eq(_) => self.builder.gen_log_eq(self.ctx, loc, lhs, rhs),
            NodeKind::Neq(_) => self.builder.gen_log_neq(self.ctx, loc, lhs, rhs),
            _ => unreachable!(),
        }
    }

    fn gen_arith(&mut self, node: &'ast Node) -> Result<MlirValue<'ctx, 'blk>> {
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

        let rhs = self.cast(rhs_node, lhs_type.clone())?;

        self.table.define_type(node, lhs_type);

        match &node.kind {
            NodeKind::Add(_) => self.builder.gen_int_add(self.ctx, loc, lhs, rhs),
            NodeKind::Sub(_) => self.builder.gen_int_sub(self.ctx, loc, lhs, rhs),
            NodeKind::Mul(_) => self.builder.gen_int_mul(self.ctx, loc, lhs, rhs),
            NodeKind::Div(_) => self.builder.gen_int_div(self.ctx, loc, lhs, rhs),
            _ => unreachable!(),
        }
    }

    fn gen_bitwise(&mut self, node: &'ast Node) -> Result<MlirValue<'ctx, 'blk>> {
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

        let rhs = self.cast(rhs_node, lhs_type.clone())?;

        self.table.define_type(node, lhs_type.clone());

        match &node.kind {
            NodeKind::BitAnd(_) => self.builder.gen_bit_and(self.ctx, loc, lhs, rhs),
            NodeKind::BitOr(_) => self.builder.gen_bit_or(self.ctx, loc, lhs, rhs),
            NodeKind::BitXor(_) => self.builder.gen_bit_xor(self.ctx, loc, lhs, rhs),
            _ => unreachable!(),
        }
    }

    fn gen_block(&mut self, nodes: &'ast [Node]) -> Result<()> {
        for (i, expr) in nodes.iter().enumerate() {
            if let None = self.gen_expr(expr)? {
                if i != (nodes.len() - 1) {
                    Err(Error::new(
                        nodes[i + 1].span,
                        "Unreachable statement".to_string(),
                    ))?;
                }
            }
        }
        Ok(())
    }

    // Generates a block. Ensures that the last instruction is a terminator returning void if
    // necessary. Performs type checking by casting to `self.builder.ret_ty`.
    fn gen_block_terminated(&mut self, nodes: &'ast [Node]) -> Result<()> {
        self.gen_block(nodes)?;
        if let None = self.builder.block.terminator() {
            let loc = Location::unknown(self.ctx);
            self.builder.gen_return(self.ctx, loc, None)?;
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
            NodeKind::ModuleDecl(m_d) => {
                let mut ins = HashMap::new();
                let mut outs = HashMap::new();
                for comb in &m_d.members {
                    match &comb.kind {
                        NodeKind::SubroutineDecl(s_d) => {
                            for param in &s_d.params {
                                self.table.define_name(param.name, &param.ty)?;
                                let param_ty = self.gen_type(&param.ty)?;
                                let named_in = NamedType {
                                    name: param.name,
                                    ty: param_ty,
                                };
                                ins.insert(param.name, named_in);
                            }
                            let return_ty = self.gen_type(&s_d.return_type)?;
                            let named_out = NamedType {
                                name: s_d.ident,
                                ty: return_ty,
                            };
                            outs.insert(s_d.ident, named_out);
                        }
                        _ => {}
                    }
                }
                TaraType::Module(ModuleInfo {
                    ins: Vec::from_iter(ins.into_values()),
                    outs: Vec::from_iter(outs.into_values()),
                })
            }
            _ => unimplemented!(),
        };
        self.table.define_type(node, ty.clone());
        Ok(RRC::new(ty))
    }

    fn get_identifier(&mut self, node: &'ast Node) -> Result<MlirValue<'ctx, 'blk>> {
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

    fn cast(&mut self, node: &Node, expected_type: RRC<TaraType>) -> Result<MlirValue<'ctx, 'ctx>> {
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
                    self.builder.insert_operation(op.clone())
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
                    self.builder.insert_operation(op.clone())
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

#[derive(Clone)]
struct Builder<'ctx, 'blk> {
    surr_context: SurroundingContext,
    block: MlirBlockRef<'ctx, 'blk>,
    block_ret_ty: Option<RRC<TaraType>>,
}

impl<'ctx, 'blk> Builder<'ctx, 'blk> {
    pub fn new(block: MlirBlockRef<'ctx, 'blk>) -> Self {
        Self {
            surr_context: SurroundingContext::Sw,
            block,
            block_ret_ty: None,
        }
    }

    // Returns the current context and updates self to have a surr_context of `ctx`.
    pub fn save(&mut self, ctx: SurroundingContext) -> Self {
        let save = self.clone();
        self.surr_context = ctx;
        save
    }

    // Returns the current context and updates self to have a surr_context of `ctx` with a new
    // block.
    pub fn save_with_block(
        &mut self,
        ctx: SurroundingContext,
        block: MlirBlockRef<'ctx, '_>,
    ) -> Self {
        let save = self.save(ctx);
        self.set_block(block);
        save
    }

    pub fn ret_ty(&self) -> RRC<TaraType> {
        self.block_ret_ty
            .clone()
            .unwrap_or(RRC::new(TaraType::Void))
    }

    pub fn restore(&mut self, ctx: Self) {
        *self = ctx;
    }

    // Set the insertion point of the builder
    pub fn set_block(&mut self, block: MlirBlockRef<'ctx, '_>) {
        let b = unsafe { MlirBlockRef::from_raw(block.to_raw()) };
        self.block = b;
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

    pub fn add_argument(
        &self,
        r#type: MlirType<'ctx>,
        loc: Location<'ctx>,
    ) -> MlirValue<'ctx, 'blk> {
        let raw = self.block.add_argument(r#type, loc).to_raw();
        unsafe { MlirValue::from_raw(raw) }
    }

    fn insert_operation<T: Into<MlirOperation<'ctx>> + Clone>(
        &self,
        op: T,
    ) -> Result<MlirValue<'ctx, 'ctx>> {
        let op_ref = self.block.append_operation(op.into());
        let res = op_ref.result(0)?;
        let raw_val = res.to_raw();
        Ok(unsafe { MlirValue::from_raw(raw_val) })
    }

    // TODO: melior doesn't generate function to set return types which is what comb
    // expects
    pub fn gen_bit_and(
        &self,
        ctx: &'ctx Context,
        loc: Location<'ctx>,
        lhs: MlirValue<'ctx, 'ctx>,
        rhs: MlirValue<'ctx, 'ctx>,
    ) -> Result<MlirValue<'ctx, 'ctx>> {
        let lhs_type = lhs.r#type();
        match self.surr_context {
            SurroundingContext::Sw => {
                let built = AndIOperation::builder(ctx, loc).lhs(lhs).rhs(rhs).build();
                let op = built.as_operation();
                self.insert_operation(op.clone())
            }
            SurroundingContext::Hw => {
                let op = OperationBuilder::new("comb.and", loc)
                    .add_operands(&[lhs, rhs])
                    .add_results(&[lhs_type])
                    .build()?;
                self.insert_operation(op.clone())
            }
        }
    }

    pub fn gen_bit_or(
        &self,
        ctx: &'ctx Context,
        loc: Location<'ctx>,
        lhs: MlirValue<'ctx, 'ctx>,
        rhs: MlirValue<'ctx, 'ctx>,
    ) -> Result<MlirValue<'ctx, 'ctx>> {
        let unit = MlirAttribute::unit(ctx);
        match self.surr_context {
            SurroundingContext::Sw => {
                let built = OrIOperation::builder(ctx, loc).lhs(lhs).rhs(rhs).build();
                let op = built.as_operation();
                self.insert_operation(op.clone())
            }
            SurroundingContext::Hw => {
                let built = HwOrOperation::builder(ctx, loc)
                    .inputs(&[lhs, rhs])
                    .two_state(unit)
                    .build();
                let op = built.as_operation();
                self.insert_operation(op.clone())
            }
        }
    }

    pub fn gen_bit_xor(
        &self,
        ctx: &'ctx Context,
        loc: Location<'ctx>,
        lhs: MlirValue<'ctx, 'ctx>,
        rhs: MlirValue<'ctx, 'ctx>,
    ) -> Result<MlirValue<'ctx, 'ctx>> {
        let unit = MlirAttribute::unit(ctx);
        match self.surr_context {
            SurroundingContext::Sw => {
                let built = XOrIOperation::builder(ctx, loc).lhs(lhs).rhs(rhs).build();
                let op = built.as_operation();
                self.insert_operation(op.clone())
            }
            SurroundingContext::Hw => {
                let built = HwXorOperation::builder(ctx, loc)
                    .inputs(&[lhs, rhs])
                    .two_state(unit)
                    .build();
                let op = built.as_operation();
                self.insert_operation(op.clone())
            }
        }
    }

    pub fn gen_int_add(
        &self,
        ctx: &'ctx Context,
        loc: Location<'ctx>,
        lhs: MlirValue<'ctx, 'ctx>,
        rhs: MlirValue<'ctx, 'ctx>,
    ) -> Result<MlirValue<'ctx, 'ctx>> {
        let lhs_type = lhs.r#type();
        match self.surr_context {
            SurroundingContext::Sw => {
                let built = AddIOperation::builder(ctx, loc).lhs(lhs).rhs(rhs).build();
                let op = built.as_operation();
                self.insert_operation(op.clone())
            }
            SurroundingContext::Hw => {
                let op = OperationBuilder::new("comb.add", loc)
                    .add_operands(&[lhs, rhs])
                    .add_results(&[lhs_type])
                    .build()?;
                self.insert_operation(op)
            }
        }
    }

    pub fn gen_int_sub(
        &self,
        ctx: &'ctx Context,
        loc: Location<'ctx>,
        lhs: MlirValue<'ctx, 'ctx>,
        rhs: MlirValue<'ctx, 'ctx>,
    ) -> Result<MlirValue<'ctx, 'ctx>> {
        let unit = MlirAttribute::unit(ctx);
        match self.surr_context {
            SurroundingContext::Sw => {
                let built = SubIOperation::builder(ctx, loc).lhs(lhs).rhs(rhs).build();
                let op = built.as_operation();
                self.insert_operation(op.clone())
            }
            SurroundingContext::Hw => {
                let built = HwSubOperation::builder(ctx, loc)
                    .lhs(lhs)
                    .rhs(rhs)
                    .two_state(unit)
                    .build();
                let op = built.as_operation();
                self.insert_operation(op.clone())
            }
        }
    }

    pub fn gen_int_mul(
        &self,
        ctx: &'ctx Context,
        loc: Location<'ctx>,
        lhs: MlirValue<'ctx, 'ctx>,
        rhs: MlirValue<'ctx, 'ctx>,
    ) -> Result<MlirValue<'ctx, 'ctx>> {
        let unit = MlirAttribute::unit(ctx);
        match self.surr_context {
            SurroundingContext::Sw => {
                let built = MulIOperation::builder(ctx, loc).lhs(lhs).rhs(rhs).build();
                let op = built.as_operation();
                self.insert_operation(op.clone())
            }
            SurroundingContext::Hw => {
                let built = HwMulOperation::builder(ctx, loc)
                    .inputs(&[lhs, rhs])
                    .two_state(unit)
                    .build();
                let op = built.as_operation();
                self.insert_operation(op.clone())
            }
        }
    }

    pub fn gen_int_div(
        &self,
        _: &'ctx Context,
        _: Location<'ctx>,
        _: MlirValue<'ctx, 'ctx>,
        _: MlirValue<'ctx, 'ctx>,
    ) -> Result<MlirValue<'ctx, 'ctx>> {
        match self.surr_context {
            SurroundingContext::Sw => {
                unimplemented!("SW: Signed and unsigned division")
            }
            SurroundingContext::Hw => {
                unimplemented!("HW: Signed and unsigned division")
            }
        }
    }

    pub fn gen_log_or(
        &self,
        ctx: &'ctx Context,
        loc: Location<'ctx>,
        lhs: MlirValue<'ctx, 'ctx>,
        rhs: MlirValue<'ctx, 'ctx>,
    ) -> Result<MlirValue<'ctx, 'ctx>> {
        let unit = MlirAttribute::unit(ctx);
        match self.surr_context {
            SurroundingContext::Sw => {
                let built = OrIOperation::builder(ctx, loc).lhs(lhs).rhs(rhs).build();
                let op = built.as_operation();
                self.insert_operation(op.clone())
            }
            SurroundingContext::Hw => {
                let built = HwOrOperation::builder(ctx, loc)
                    .inputs(&[lhs, rhs])
                    .two_state(unit)
                    .build();
                let op = built.as_operation();
                self.insert_operation(op.clone())
            }
        }
    }

    pub fn gen_log_and(
        &self,
        ctx: &'ctx Context,
        loc: Location<'ctx>,
        lhs: MlirValue<'ctx, 'ctx>,
        rhs: MlirValue<'ctx, 'ctx>,
    ) -> Result<MlirValue<'ctx, 'ctx>> {
        let lhs_type = lhs.r#type();
        match self.surr_context {
            SurroundingContext::Sw => {
                let built = AndIOperation::builder(ctx, loc).lhs(lhs).rhs(rhs).build();
                let op = built.as_operation();
                self.insert_operation(op.clone())
            }
            SurroundingContext::Hw => {
                let op = OperationBuilder::new("comb.and", loc)
                    .add_operands(&[lhs, rhs])
                    .add_results(&[lhs_type])
                    .build()?;
                self.insert_operation(op.clone())
            }
        }
    }

    // TODO: Support signed and unsigned lt
    pub fn gen_log_lt(
        &self,
        ctx: &'ctx Context,
        loc: Location<'ctx>,
        lhs: MlirValue<'ctx, 'ctx>,
        rhs: MlirValue<'ctx, 'ctx>,
    ) -> Result<MlirValue<'ctx, 'ctx>> {
        let predicate_type = TaraType::IntUnsigned { width: 64 }.to_mlir_type(ctx);
        let ret_type = TaraType::Bool.to_mlir_type(ctx);
        match self.surr_context {
            SurroundingContext::Sw => {
                let built = CmpIOperation::builder(ctx, loc)
                    .predicate(
                        MlirIntegerAttribute::new(predicate_type, CmpiPredicate::Ult as i64).into(),
                    )
                    .lhs(lhs)
                    .rhs(rhs)
                    .result(ret_type)
                    .build();
                let op = built.as_operation();
                self.insert_operation(op.clone())
            }
            SurroundingContext::Hw => {
                unimplemented!()
            }
        }
    }

    // TODO: Support signed and unsigned gt
    pub fn gen_log_gt(
        &self,
        ctx: &'ctx Context,
        loc: Location<'ctx>,
        lhs: MlirValue<'ctx, 'ctx>,
        rhs: MlirValue<'ctx, 'ctx>,
    ) -> Result<MlirValue<'ctx, 'ctx>> {
        let predicate_type = TaraType::IntUnsigned { width: 64 }.to_mlir_type(ctx);
        let ret_type = TaraType::Bool.to_mlir_type(ctx);
        let unit = MlirAttribute::unit(ctx);
        match self.surr_context {
            SurroundingContext::Sw => {
                let built = CmpIOperation::builder(ctx, loc)
                    .predicate(
                        MlirIntegerAttribute::new(predicate_type, CmpiPredicate::Ugt as i64).into(),
                    )
                    .lhs(lhs)
                    .rhs(rhs)
                    .result(ret_type)
                    .build();
                let op = built.as_operation();
                self.insert_operation(op.clone())
            }
            SurroundingContext::Hw => {
                let built = HwCmpOperation::builder(ctx, loc)
                    .predicate(
                        MlirIntegerAttribute::new(predicate_type, CombICmpPredicate::Ugt as i64)
                            .into(),
                    )
                    .lhs(lhs)
                    .rhs(rhs)
                    .two_state(unit)
                    .result(ret_type)
                    .build();
                let op = built.as_operation();
                self.insert_operation(op.clone())
            }
        }
    }

    // TODO: Support signed and unsigned lte
    pub fn gen_log_lte(
        &self,
        ctx: &'ctx Context,
        loc: Location<'ctx>,
        lhs: MlirValue<'ctx, 'ctx>,
        rhs: MlirValue<'ctx, 'ctx>,
    ) -> Result<MlirValue<'ctx, 'ctx>> {
        let predicate_type = TaraType::IntUnsigned { width: 64 }.to_mlir_type(ctx);
        let ret_type = TaraType::Bool.to_mlir_type(ctx);
        match self.surr_context {
            SurroundingContext::Sw => {
                let built = CmpIOperation::builder(ctx, loc)
                    .predicate(
                        MlirIntegerAttribute::new(predicate_type, CmpiPredicate::Ule as i64).into(),
                    )
                    .lhs(lhs)
                    .rhs(rhs)
                    .result(ret_type)
                    .build();
                let op = built.as_operation();
                self.insert_operation(op.clone())
            }
            SurroundingContext::Hw => {
                unimplemented!()
            }
        }
    }

    // TODO: Support signed and unsigned gte
    pub fn gen_log_gte(
        &self,
        ctx: &'ctx Context,
        loc: Location<'ctx>,
        lhs: MlirValue<'ctx, 'ctx>,
        rhs: MlirValue<'ctx, 'ctx>,
    ) -> Result<MlirValue<'ctx, 'ctx>> {
        let predicate_type = TaraType::IntUnsigned { width: 64 }.to_mlir_type(ctx);
        let ret_type = TaraType::Bool.to_mlir_type(ctx);
        match self.surr_context {
            SurroundingContext::Sw => {
                let built = CmpIOperation::builder(ctx, loc)
                    .predicate(
                        MlirIntegerAttribute::new(predicate_type, CmpiPredicate::Uge as i64).into(),
                    )
                    .lhs(lhs)
                    .rhs(rhs)
                    .result(ret_type)
                    .build();
                let op = built.as_operation();
                self.insert_operation(op.clone())
            }
            SurroundingContext::Hw => {
                unimplemented!()
            }
        }
    }

    pub fn gen_log_eq(
        &self,
        ctx: &'ctx Context,
        loc: Location<'ctx>,
        lhs: MlirValue<'ctx, 'ctx>,
        rhs: MlirValue<'ctx, 'ctx>,
    ) -> Result<MlirValue<'ctx, 'ctx>> {
        let predicate_type = TaraType::IntUnsigned { width: 64 }.to_mlir_type(ctx);
        let ret_type = TaraType::Bool.to_mlir_type(ctx);
        match self.surr_context {
            SurroundingContext::Sw => {
                let built = CmpIOperation::builder(ctx, loc)
                    .predicate(
                        MlirIntegerAttribute::new(predicate_type, CmpiPredicate::Eq as i64).into(),
                    )
                    .lhs(lhs)
                    .rhs(rhs)
                    .result(ret_type)
                    .build();
                let op = built.as_operation();
                self.insert_operation(op.clone())
            }
            SurroundingContext::Hw => {
                unimplemented!()
            }
        }
    }

    pub fn gen_log_neq(
        &self,
        ctx: &'ctx Context,
        loc: Location<'ctx>,
        lhs: MlirValue<'ctx, 'ctx>,
        rhs: MlirValue<'ctx, 'ctx>,
    ) -> Result<MlirValue<'ctx, 'ctx>> {
        let predicate_type = TaraType::IntUnsigned { width: 64 }.to_mlir_type(ctx);
        let ret_type = TaraType::Bool.to_mlir_type(ctx);
        match self.surr_context {
            SurroundingContext::Sw => {
                let built = CmpIOperation::builder(ctx, loc)
                    .predicate(
                        MlirIntegerAttribute::new(predicate_type, CmpiPredicate::Ne as i64).into(),
                    )
                    .lhs(lhs)
                    .rhs(rhs)
                    .result(ret_type)
                    .build();
                let op = built.as_operation();
                self.insert_operation(op.clone())
            }
            SurroundingContext::Hw => {
                unimplemented!()
            }
        }
    }

    pub fn gen_return(
        &self,
        ctx: &'ctx Context,
        loc: Location<'ctx>,
        ret_val: Option<MlirValue<'ctx, 'ctx>>,
    ) -> Result<()> {
        match self.surr_context {
            SurroundingContext::Sw => {
                let built = ReturnOperation::builder(ctx, loc)
                    .operands(ret_val.as_slice())
                    .build();
                let op = built.as_operation();
                self.block.append_operation(op.clone());
            }
            SurroundingContext::Hw => {
                let built = HwProcReturnOperation::builder(ctx, loc)
                    .outputs(ret_val.as_slice())
                    .build();
                let op = built.as_operation();
                self.block.append_operation(op.clone());
            }
        }
        Ok(())
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

// The context surrounding an operation. This changes when crossing software/hardware boundaries
// (i.e. anything->module, anything->fn)
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum SurroundingContext {
    Sw,
    Hw,
}
