mod error;
mod table;

pub use crate::ast_codegen::error::Error;
use crate::{
    ast::{Node, NodeKind},
    ast_codegen::table::Table,
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
        hw::{
            ConstantOperation as HwConstantOperation, HWModuleOperation as HwModuleOperation,
            OutputOperation as HwOutputOperation, StructCreateOperation as HwStructCreateOperation,
        },
        CombICmpPredicate,
    },
    module::{
        decls::{Decl, DeclStatus},
        file::File,
        namespace::Namespace,
        structs::{Field, Struct, StructStatus},
        tmodule::{ModuleStatus, TModule},
    },
    types::{NamedType, Type as TaraType},
    utils::{init_field, RRC},
    values::{StaticMlirValue, TypedValue, Value as TaraValue},
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
                AddIOperation, AndIOperation, CmpIOperation, ConstantOperation, ExtSIOperation,
                ExtUIOperation, MulIOperation, OrIOperation, SubIOperation, XOrIOperation,
            },
            func::{FuncOperation, ReturnOperation},
            llvm::{InsertValueOperation, UndefOperation},
        },
    },
    ir::{
        attribute::{
            ArrayAttribute as MlirArrayAttribute,
            DenseI64ArrayAttribute as MlirDenseI64ArrayAttribute,
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
use num_bigint::ToBigInt;
use std::collections::HashMap;
use symbol_table::GlobalSymbol;

pub struct AstCodegen<'a, 'ast, 'ctx> {
    ast: &'ast Ast,
    ctx: &'ctx Context,
    module: &'a MlirModule<'ctx>,
    builder: Builder<'ctx, 'a>,
    table: Table,
    pub errors: Vec<anyhow::Error>,
    anon_cnt: usize,
}

impl<'a, 'ast, 'ctx> AstCodegen<'a, 'ast, 'ctx> {
    pub fn new(ast: &'ast Ast, ctx: &'ctx Context, module: &'a MlirModule<'ctx>) -> Self {
        let root_node = &ast.root;
        let root = Decl::new("root", root_node);
        let root_rrc = Decl::init_struct_empty_namespace(root, &ast.root);

        Self {
            ast,
            ctx,
            module,
            builder: Builder::new(root_rrc, module.body()),
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
        let node = &self.ast.root;
        self.setup_namespace(node)?;
        self._gen_container(node, Self::gen_fn_decl)?;

        Ok(())
    }

    fn _gen_container<F: Fn(&mut Self, &Node) -> Result<TypedValue>>(
        &mut self,
        node: &Node,
        subroutine_gen: F,
    ) -> Result<()> {
        matches!(node.kind, NodeKind::StructDecl(_) | NodeKind::ModuleDecl(_));

        let members = match &node.kind {
            NodeKind::StructDecl(s) => &s.members,
            NodeKind::ModuleDecl(m) => &m.members,
            _ => unreachable!(),
        };

        let mut error_guard = Decl::error_guard(self.builder.curr_decl());
        for member in members {
            match &member.kind {
                NodeKind::VarDecl(_) => match self.gen_var_decl(&member) {
                    Ok(_) => {}
                    Err(e) => {
                        self.pop();
                        self.errors.push(e);
                    }
                },
                NodeKind::SubroutineDecl(_) => {
                    match self._gen_subroutine(&member, &subroutine_gen) {
                        Ok(_) => {}
                        Err(e) => {
                            self.pop();
                            self.errors.push(e);
                        }
                    }
                }
                _ => unreachable!(),
            }
        }
        error_guard.success();

        Ok(())
    }

    fn gen_struct_decl(&mut self, node: &Node) -> Result<TypedValue> {
        self.push();

        let struct_ty = {
            let curr_decl = self.builder.parent_decl.clone();
            let namespace = self.builder.namespace();
            Decl::init_struct(curr_decl.clone(), node, namespace);
            let struct_obj = match curr_decl.borrow().value.as_ref().unwrap() {
                TaraValue::Type(ty) => match &ty {
                    TaraType::Struct(s) => s.clone(),
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            };
            TaraType::Struct(struct_obj)
        };

        self.setup_namespace(node)?;

        self._gen_container(node, Self::gen_fn_decl)?;

        self.pop();

        let ty_val = TypedValue::new(TaraType::Type, TaraValue::Type(struct_ty));
        Ok(ty_val)
    }

    fn gen_module_decl(&mut self, node: &Node) -> Result<TypedValue> {
        matches!(node.kind, NodeKind::ModuleDecl(_));
        self.push();
        let mod_decl = match &node.kind {
            NodeKind::ModuleDecl(m_d) => m_d,
            _ => unreachable!(),
        };
        let loc = node.loc(self.ctx);
        let prev_context = self.builder.save(SurroundingContext::Hw);

        // Set up module decl
        let module_ty = {
            let curr_decl = self.builder.parent_decl.clone();
            let namespace = self.builder.namespace();
            Decl::init_module(curr_decl.clone(), node, namespace);
            let module_ty = curr_decl.borrow().value.as_ref().unwrap().to_type();
            module_ty
        };

        self.builder
            .curr_decl()
            .map_mut(|decl| decl.status = DeclStatus::InProgress);

        self.setup_namespace(node)?;

        match &module_ty {
            TaraType::Module(m) => self.resolve_module_type(m.clone())?,
            _ => unreachable!(),
        }

        let mod_mlir_type = self.builder.get_mlir_tye(self.ctx, &module_ty);

        let region = MlirRegion::new();
        let body = region.append_block(MlirBlock::new(&[]));
        self.builder.set_block(body);

        for (name, node_ptr, param_type) in &module_ty.module().borrow().ins {
            let param_node = unsafe { &**node_ptr };
            self.table.define_name(name.into(), param_node)?;
            let mlir_param_type = self.builder.get_mlir_tye(self.ctx, &param_type);
            let arg = self
                .builder
                .add_argument(mlir_param_type, param_node.loc(self.ctx));
            let ty_val = TypedValue::new(param_type.clone(), arg);
            self.table.define_ty_val(param_node, ty_val);
        }

        self._gen_container(node, Self::gen_comb_decl)?;

        if self.errors.len() == 0 {
            self.builder
                .curr_decl()
                .map_mut(|decl| decl.status = DeclStatus::CodegenFailure);

            let mut outs = Vec::new();
            for member in &mod_decl.members {
                match &member.kind {
                    NodeKind::SubroutineDecl(s_d) => {
                        let ret_ty = self.gen_type(&s_d.return_type)?;
                        if ret_ty == TaraType::Void {
                            continue;
                        }
                        let val = self.table.get_value(member)?.get_runtime_value();
                        outs.push(val);
                    }
                    _ => {}
                }
            }
            let output_op = HwOutputOperation::builder(self.ctx, loc)
                .outputs(&outs)
                .build();
            body.append_operation(output_op.into());

            let operation: MlirOperation = {
                let name = &self.builder.parent_decl.borrow().name;
                let builder = HwModuleOperation::builder(self.ctx, loc)
                    .body(region)
                    .sym_name(MlirStringAttribute::new(self.ctx, name.as_str()))
                    .module_type(MlirTypeAttribute::new(mod_mlir_type))
                    .parameters(MlirArrayAttribute::new(self.ctx, &[]))
                    .build();
                builder.into()
            };

            self.module.body().append_operation(operation);
        }

        self.builder.restore(prev_context);
        self.pop();

        let ty_val = TypedValue::new(TaraType::Type, TaraValue::Type(module_ty));

        self.builder.curr_decl().map_mut(|decl| {
            let ty_val = ty_val.clone();
            decl.status = DeclStatus::Complete;
            decl.ty = Some(ty_val.ty);
            decl.value = Some(ty_val.value);
        });

        Ok(ty_val)
    }

    fn gen_var_decl(&mut self, node: &Node) -> Result<TypedValue> {
        matches!(node.kind, NodeKind::VarDecl(_));

        let var_decl = match &node.kind {
            NodeKind::VarDecl(v_d) => v_d,
            _ => unreachable!(),
        };
        let block = self.module.body();
        let decl_name = var_decl.ident.as_str();
        // Decl should've already been created if we got to this point
        let decl = self.builder.find_decl(decl_name).unwrap();
        let prev_context =
            self.builder
                .save_with_block_with_decl(SurroundingContext::Sw, block, decl.clone());

        let mut ty_val = self.gen_expr_reachable(&var_decl.expr)?;
        if let Some(ty_expr) = &var_decl.ty {
            let expected_type = self.gen_type(ty_expr)?;
            ty_val.value = self.cast(&var_decl.expr, ty_val.clone(), &expected_type)?;
            ty_val.ty = expected_type;
        };
        decl.map_mut(|decl| {
            let ty_val = ty_val.clone();
            decl.ty = Some(ty_val.ty);
            decl.value = Some(ty_val.value);
        });

        self.builder.restore(prev_context);

        Ok(ty_val)
    }

    fn _gen_subroutine<F: Fn(&mut Self, &Node) -> Result<TypedValue>>(
        &mut self,
        node: &Node,
        subroutine_gen: F,
    ) -> Result<()> {
        matches!(node.kind, NodeKind::SubroutineDecl(_));

        let sr_decl = match &node.kind {
            NodeKind::SubroutineDecl(s_d) => s_d,
            _ => unreachable!(),
        };
        let decl_name = sr_decl.ident.as_str();
        // Decl should've already been created if we got to this point
        let decl = self.builder.find_decl(decl_name).unwrap();
        let prev_context = self.builder.save_with_block_with_decl(
            self.builder.surr_context,
            self.builder.block,
            decl.clone(),
        );

        match decl.borrow().status {
            DeclStatus::Complete => return Ok(()),
            _ => {}
        }

        decl.map_mut(|decl| decl.status = DeclStatus::InProgress);

        let ty_val = subroutine_gen(self, node)?;

        decl.map_mut(|decl| {
            let ty_val = ty_val.clone();
            decl.status = DeclStatus::Complete;
            decl.ty = Some(ty_val.ty);
            decl.value = Some(ty_val.value);
        });

        self.builder.restore(prev_context);

        Ok(())
    }

    fn gen_fn_decl(&mut self, node: &Node) -> Result<TypedValue> {
        matches!(node.kind, NodeKind::SubroutineDecl(_));
        let fn_decl = match &node.kind {
            NodeKind::SubroutineDecl(s_d) => s_d,
            _ => unreachable!(),
        };
        self.push();

        // Set up fn decl
        let fn_ty = {
            let curr_decl = self.builder.parent_decl.clone();
            let namespace = self.builder.namespace();
            Decl::init_fn(curr_decl.clone(), node, namespace);
            let fn_obj = match curr_decl.borrow().value.as_ref().unwrap() {
                TaraValue::Type(ty) => match &ty {
                    TaraType::Function(f) => f.clone(),
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            };
            TaraType::Function(fn_obj)
        };

        let return_type = self.gen_type(&fn_decl.return_type)?;
        let mlir_return_type = self.builder.get_mlir_tye(self.ctx, &return_type);

        let loc = node.loc(self.ctx);
        let region = MlirRegion::new();
        let block = MlirBlock::new(&[]);
        let block_ref = region.append_block(block);
        let prev_context = self
            .builder
            .save_with_block(SurroundingContext::Sw, block_ref);
        self.builder.block_ret_ty = Some(return_type.clone());

        {
            let mut param_types = Vec::new();
            for param in &fn_decl.params {
                let param_ty = param.ty.as_ref();
                self.table.define_name(param.name, param_ty)?;
                let param_type = self.gen_type(param_ty)?;
                let mlir_param_type = self.builder.get_mlir_tye(self.ctx, &param_type);
                param_types.push(mlir_param_type);

                let arg = self
                    .builder
                    .add_argument(mlir_param_type, param.ty.loc(self.ctx));
                let ty_val = TypedValue::new(param_type, arg);
                self.table.define_ty_val(param_ty, ty_val);
            }

            self.gen_block_terminated(&fn_decl.block)?;

            let mlir_fn_ret_type = if return_type != TaraType::Void {
                Some(mlir_return_type)
            } else {
                None
            };
            let fn_name = &self.builder.parent_decl.borrow().name;
            let fn_type =
                MlirFunctionType::new(self.ctx, &param_types, mlir_fn_ret_type.as_slice()).into();
            let builder = FuncOperation::builder(self.ctx, loc)
                .body(region)
                .sym_name(MlirStringAttribute::new(self.ctx, &fn_name))
                .function_type(MlirTypeAttribute::new(fn_type));
            let func = builder.build();
            self.module.body().append_operation(func.into());
        };

        self.pop();
        self.builder.restore(prev_context);

        let ty_val = TypedValue::new(TaraType::Type, TaraValue::Type(fn_ty));

        Ok(ty_val)
    }

    fn gen_comb_decl(&mut self, node: &Node) -> Result<TypedValue> {
        matches!(node.kind, NodeKind::SubroutineDecl(_));
        let comb_decl = match &node.kind {
            NodeKind::SubroutineDecl(s_d) => s_d,
            _ => unreachable!(),
        };
        self.push();

        // Set up comb decl
        let comb_ty = {
            let curr_decl = self.builder.parent_decl.clone();
            let namespace = self.builder.namespace();
            Decl::init_comb(curr_decl.clone(), node, namespace);
            let comb_obj = match curr_decl.borrow().value.as_ref().unwrap() {
                TaraValue::Type(ty) => match &ty {
                    TaraType::Comb(c) => c.clone(),
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            };
            TaraType::Comb(comb_obj)
        };

        let return_type = self.gen_type(&comb_decl.return_type)?;
        // Don't emit an arc that returns void as all computation can be culled
        if return_type == TaraType::Void {
            return Ok(TypedValue::new(TaraType::Void, TaraValue::VoidValue));
        }
        let mlir_return_type = self.builder.get_mlir_tye(self.ctx, &return_type);

        let loc = node.loc(self.ctx);
        let body = self.module.body();
        {
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
                self.table.define_name_shadow_upper(param.name, param_ty)?;
                let param_type = self.gen_type(param_ty)?;
                let mlir_param_type = self.builder.get_mlir_tye(self.ctx, &param_type);
                param_types.push(mlir_param_type);

                let arg = self
                    .builder
                    .add_argument(mlir_param_type, param.ty.loc(self.ctx));
                let ty_val = TypedValue::new(param_type, arg);
                self.table.define_ty_val(param_ty, ty_val);
            }

            self.gen_block_terminated(&comb_decl.block)?;

            let comb_type =
                MlirFunctionType::new(self.ctx, &param_types, &[mlir_return_type]).into();
            let builder = HwProcDefineOperation::builder(self.ctx, loc)
                .body(region)
                .sym_name(MlirStringAttribute::new(
                    self.ctx,
                    &self.builder.parent_decl.borrow().name,
                ))
                .function_type(MlirTypeAttribute::new(comb_type));
            let comb = builder.build();
            let operation: MlirOperation = comb.into();
            body.append_operation(operation);
            self.builder.restore(prev_context);
        };

        self.pop();

        let mut ins = Vec::new();
        for param in &comb_decl.params {
            let param_val = self
                .table
                .get_named_value(param.name, param.ty.span)?
                .get_runtime_value();
            ins.push(param_val);
        }
        let comb_name = &self.builder.parent_decl.borrow().name;
        let call_op = HwProcCallOperation::builder(self.ctx, loc)
            .arc(MlirFlatSymbolRefAttribute::new(self.ctx, &comb_name))
            .inputs(&ins)
            .outputs(&[mlir_return_type])
            .build();
        let call_val = self
            .builder
            .insert_operation(call_op.as_operation().clone())?;

        let rt_ty_val = TypedValue::new(return_type, call_val);
        self.table.define_ty_val(node, rt_ty_val);

        let ty_val = TypedValue::new(TaraType::Type, TaraValue::Type(comb_ty));
        Ok(ty_val)
    }

    fn gen_return(&mut self, node: &Node) -> Result<TypedValue> {
        matches!(node.kind, NodeKind::Return(_));
        let loc = node.loc(self.ctx);
        match &node.kind {
            NodeKind::Return(Some(e)) => {
                let return_ty_val = self.gen_expr_reachable(&e.lhs)?;
                self.table.define_ty_val(node, return_ty_val.clone());
                let ret_ty = self.builder.ret_ty();
                let casted_value = self.cast(node, return_ty_val, &ret_ty)?;
                let mat_value = self
                    .builder
                    .materialize(self.ctx, loc, &ret_ty, &casted_value)?;
                self.builder.gen_return(self.ctx, loc, Some(mat_value))?;
            }
            NodeKind::Return(None) => {
                let ty_val = TypedValue::new(TaraType::Void, TaraValue::VoidValue);
                self.table.define_ty_val(node, ty_val);
                self.builder.gen_return(self.ctx, loc, None)?;
            }
            _ => unreachable!(),
        };

        assert!(self.builder.block.terminator().is_some());
        let ty_val = TypedValue::new(TaraType::NoReturn, TaraValue::VoidValue);
        Ok(ty_val)
    }

    fn gen_expr(&mut self, node: &Node) -> Result<TypedValue> {
        let ty_val = match &node.kind {
            NodeKind::StructDecl(_) => self.gen_struct_decl(node)?,
            NodeKind::ModuleDecl(_) => self.gen_module_decl(node)?,
            NodeKind::Return(_) => self.gen_return(node)?,
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
            | NodeKind::Div(_) => self.gen_bin_op(node)?,
            NodeKind::Identifier(_) => self.get_identifier(node)?,
            NodeKind::StructInit(_) => self.gen_struct_init(node)?,
            NodeKind::NumberLiteral(_) => self.gen_number(node)?,
            NodeKind::VarDecl(_) => self.gen_var_decl(node)?,
            NodeKind::LocalVarDecl(_) => self.gen_local_var_decl(node)?,
            _ => unimplemented!(),
        };
        Ok(ty_val)
    }

    fn gen_expr_reachable(&mut self, node: &Node) -> Result<TypedValue> {
        let ty_val = self.gen_expr(node)?;
        if ty_val.ty.is_noreturn() {
            Err(Error::new(
                node.span,
                "Expected reachable value, control flow unexpectedly diverted".to_string(),
            ))?
        }
        self.table.define_ty_val(node, ty_val.clone());
        Ok(ty_val)
    }

    fn gen_local_var_decl(&mut self, node: &Node) -> Result<TypedValue> {
        matches!(node.kind, NodeKind::LocalVarDecl(_));
        let local_var_decl = match &node.kind {
            NodeKind::LocalVarDecl(l_v_d) => l_v_d,
            _ => unreachable!(),
        };

        let mut ty_val = self.gen_expr_reachable(&local_var_decl.expr)?;
        if let Some(ty_expr) = &local_var_decl.ty {
            let expected_type = self.gen_type(ty_expr)?;
            ty_val.value = self.cast(&local_var_decl.expr, ty_val.clone(), &expected_type)?;
            ty_val.ty = expected_type;
        };

        self.table.define_name(local_var_decl.ident, node)?;
        self.table.define_ty_val(node, ty_val.clone());

        Ok(ty_val)
    }

    fn gen_bin_op(&mut self, node: &Node) -> Result<TypedValue> {
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

    fn gen_cmp(&mut self, node: &Node) -> Result<TypedValue> {
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
        let rhs_ty_val = self.table.get_ty_val(rhs_node)?;
        let rhs = match &node.kind {
            NodeKind::Or(_) | NodeKind::And(_) => {
                self.expect_bool_type(lhs_node)?;
                self.expect_bool_type(rhs_node)?;
                self.table.get_value(rhs_node)?
            }
            NodeKind::Lt(_) | NodeKind::Gt(_) | NodeKind::Lte(_) | NodeKind::Gte(_) => {
                self.expect_integral_type(lhs_node)?;
                self.expect_integral_type(rhs_node)?;
                self.cast(rhs_node, rhs_ty_val, &lhs_type)?
            }
            NodeKind::Eq(_) | NodeKind::Neq(_) => {
                self.expect_integral_type(lhs_node)
                    .or(self.expect_bool_type(lhs_node))?;
                self.expect_integral_type(rhs_node)
                    .or(self.expect_bool_type(rhs_node))?;
                self.cast(rhs_node, rhs_ty_val, &lhs_type)?
            }
            _ => unreachable!(),
        };

        if !lhs.has_runtime_value() && !rhs.has_runtime_value() {
            Err(Error::new(
                node.span,
                "TODO: compile time cmpt operation".to_string(),
            ))?;
        }

        let lhs = self
            .builder
            .materialize(self.ctx, lhs_node.loc(self.ctx), &lhs_type, &lhs)?;
        let rhs = self
            .builder
            .materialize(self.ctx, rhs_node.loc(self.ctx), &lhs_type, &rhs)?;

        let val = match node.kind {
            NodeKind::Or(_) => self.builder.gen_log_or(self.ctx, loc, lhs, rhs),
            NodeKind::And(_) => self.builder.gen_log_and(self.ctx, loc, lhs, rhs),
            NodeKind::Lt(_) => self.builder.gen_log_lt(self.ctx, loc, lhs, rhs),
            NodeKind::Gt(_) => self.builder.gen_log_gt(self.ctx, loc, lhs, rhs),
            NodeKind::Lte(_) => self.builder.gen_log_lte(self.ctx, loc, lhs, rhs),
            NodeKind::Gte(_) => self.builder.gen_log_gte(self.ctx, loc, lhs, rhs),
            NodeKind::Eq(_) => self.builder.gen_log_eq(self.ctx, loc, lhs, rhs),
            NodeKind::Neq(_) => self.builder.gen_log_neq(self.ctx, loc, lhs, rhs),
            _ => unreachable!(),
        }?;

        let ty_val = TypedValue::new(TaraType::Bool, val.clone());
        self.table.define_ty_val(node, ty_val.clone());

        Ok(ty_val)
    }

    fn gen_arith(&mut self, node: &Node) -> Result<TypedValue> {
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

        let rhs_ty_val = self.table.get_ty_val(rhs_node)?;
        let rhs = self.cast(rhs_node, rhs_ty_val, &lhs_type)?;

        if !lhs.has_runtime_value() && !rhs.has_runtime_value() {
            Err(Error::new(
                node.span,
                "TODO: compile time arithmetic operation".to_string(),
            ))?;
        }

        let lhs = self
            .builder
            .materialize(self.ctx, lhs_node.loc(self.ctx), &lhs_type, &lhs)?;
        let rhs = self
            .builder
            .materialize(self.ctx, rhs_node.loc(self.ctx), &lhs_type, &rhs)?;

        let val = match &node.kind {
            NodeKind::Add(_) => self.builder.gen_int_add(self.ctx, loc, lhs, rhs),
            NodeKind::Sub(_) => self.builder.gen_int_sub(self.ctx, loc, lhs, rhs),
            NodeKind::Mul(_) => self.builder.gen_int_mul(self.ctx, loc, lhs, rhs),
            NodeKind::Div(_) => self.builder.gen_int_div(self.ctx, loc, lhs, rhs),
            _ => unreachable!(),
        }?;

        let ty_val = TypedValue::new(lhs_type, val.clone());
        self.table.define_ty_val(node, ty_val.clone());

        Ok(ty_val)
    }

    fn gen_bitwise(&mut self, node: &Node) -> Result<TypedValue> {
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

        let rhs_ty_val = self.table.get_ty_val(rhs_node)?;
        let rhs = self.cast(rhs_node, rhs_ty_val, &lhs_type)?;

        if !lhs.has_runtime_value() && !rhs.has_runtime_value() {
            Err(Error::new(
                node.span,
                "TODO: compile time bitwise operation".to_string(),
            ))?;
        }

        let lhs = self
            .builder
            .materialize(self.ctx, lhs_node.loc(self.ctx), &lhs_type, &lhs)?;
        let rhs = self
            .builder
            .materialize(self.ctx, rhs_node.loc(self.ctx), &lhs_type, &rhs)?;

        let val = match &node.kind {
            NodeKind::BitAnd(_) => self.builder.gen_bit_and(self.ctx, loc, lhs, rhs),
            NodeKind::BitOr(_) => self.builder.gen_bit_or(self.ctx, loc, lhs, rhs),
            NodeKind::BitXor(_) => self.builder.gen_bit_xor(self.ctx, loc, lhs, rhs),
            _ => unreachable!(),
        }?;

        let ty_val = TypedValue::new(lhs_type, val.clone());
        self.table.define_ty_val(node, ty_val.clone());

        Ok(ty_val)
    }

    fn gen_struct_init(&mut self, node: &Node) -> Result<TypedValue> {
        matches!(node.kind, NodeKind::StructInit(_));
        let struct_init = match &node.kind {
            NodeKind::StructInit(s_i) => s_i,
            _ => unreachable!(),
        };

        let struct_type = match &struct_init.ty {
            Some(ty) => self.gen_type(ty)?,
            None => Err(Error::new(
                node.span,
                "TODO: support anonymous struct initialization",
            ))?,
        };

        let struct_val = self.expect_struct_type(node, &struct_type)?;

        self.resolve_struct_layout(struct_val.clone())?;

        let mut field_tracker = struct_val.map(|s| {
            let mut field_tracker = IndexMap::new();
            for field in &s.fields {
                field_tracker.insert(field.name.clone(), (false, field.ty.clone()));
            }
            field_tracker
        });

        let mut fields = Vec::new();
        for field_init in &struct_init.fields {
            match field_tracker.get_mut(field_init.name.as_str()) {
                Some((inited, expected_ty)) => {
                    if *inited {
                        Err(Error::new(
                            field_init.ty.span,
                            format!("Duplicate field initialization of {}", field_init.name),
                        ))?
                    }
                    *inited = true;
                    let init_ty_val = self.gen_expr_reachable(&field_init.ty)?;
                    let casted_value = self.cast(&field_init.ty, init_ty_val, expected_ty)?;
                    fields.push(casted_value);
                }
                None => Err(Error::new(
                    field_init.ty.span,
                    format!("Attempted to init unknown field {}", field_init.name),
                ))?,
            };
        }

        // Check if anything was left uninitialized
        for uninited_field in field_tracker
            .iter()
            .filter(|(_, (inited, _))| !*inited)
            .into_iter()
        {
            Err(Error::new(
                node.span,
                format!("Field {} is not initialized", uninited_field.0),
            ))?;
        }

        let value = TaraValue::Struct(RRC::new(fields));

        let ty_val = TypedValue::new(struct_type, value);
        self.table.define_ty_val(node, ty_val.clone());

        Ok(ty_val)
    }

    fn gen_block(&mut self, nodes: &[Node]) -> Result<()> {
        if let Some((last, rest)) = nodes.split_last() {
            for expr in rest {
                self.gen_expr_reachable(expr)?;
            }
            self.gen_expr(last)?;
        }
        Ok(())
    }

    // Generates a block. Ensures that the last instruction is a terminator returning void if
    // necessary. Performs type checking by casting to `self.builder.ret_ty`.
    fn gen_block_terminated(&mut self, nodes: &[Node]) -> Result<()> {
        self.gen_block(nodes)?;
        if let None = self.builder.block.terminator() {
            let loc = Location::unknown(self.ctx);
            self.builder.gen_return(self.ctx, loc, None)?;
        }

        Ok(())
    }

    fn gen_type(&mut self, node: &Node) -> Result<TaraType> {
        let ty = match &node.kind {
            NodeKind::Identifier(_) => {
                let ty_val = self.get_identifier(node)?;
                let ty = ty_val.value.to_type();
                match &ty {
                    TaraType::Struct(s) => self.resolve_struct_layout(s.clone())?,
                    TaraType::Module(m) => self.resolve_module_type(m.clone())?,
                    _ => {}
                }
                ty
            }
            NodeKind::ModuleDecl(m_d) => {
                let mut ins = HashMap::new();
                let mut outs = HashMap::new();
                for comb in &m_d.members {
                    match &comb.kind {
                        NodeKind::SubroutineDecl(s_d) => {
                            for param in &s_d.params {
                                // self.table.define_name(param.name, &param.ty)?;
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
                // TaraType::Module(ModuleInfo {
                //     ins: Vec::from_iter(ins.into_values()),
                //     outs: Vec::from_iter(outs.into_values()),
                // })
                unimplemented!()
            }
            NodeKind::StructDecl(_) => {
                // If we're reaching here, then we're generating an anonymous struct declaration.
                let block = self.module.body();
                let anon_num = self.anon();
                let decl = self
                    .builder
                    .new_decl(node, &format!("anon_struct{}", anon_num))?;
                let prev_context =
                    self.builder
                        .save_with_block_with_decl(SurroundingContext::Sw, block, decl);

                let ty_val = self.gen_expr_reachable(node)?;
                self.expect_type_type(node, &ty_val.ty)?;
                let struct_val = self.expect_struct_type(node, &ty_val.value.to_type())?;

                self.builder.restore(prev_context);

                TaraType::Struct(struct_val)
            }
            _ => unimplemented!(),
        };
        Ok(ty)
    }

    fn gen_number(&mut self, node: &Node) -> Result<TypedValue> {
        matches!(node.kind, NodeKind::NumberLiteral(_));
        let number: i64 = match &node.kind {
            NodeKind::NumberLiteral(num) => num.to_bigint().unwrap().try_into().unwrap(),
            _ => unreachable!(),
        };
        let ty_val = TypedValue::new(TaraType::ComptimeInt, TaraValue::Integer(number));
        Ok(ty_val)
    }

    fn get_identifier(&mut self, node: &Node) -> Result<TypedValue> {
        let ident = match node.kind {
            NodeKind::Identifier(ident) => ident,
            _ => unreachable!(),
        };
        get_maybe_primitive(ident.as_str())
            .map(Result::Ok)
            .or_else(|| {
                self.builder
                    .find_decl(ident.as_str())
                    .map(|decl| self.resolve_decl_ty_val(decl))
            })
            .unwrap_or_else(|| self.table.get_named_ty_val(ident, node.span))
    }
}

// Decl related methods
impl<'a, 'ast, 'ctx, 'blk> AstCodegen<'a, 'ast, 'ctx>
where
    'a: 'blk,
{
    fn resolve_struct_layout(&mut self, struct_rrc: RRC<Struct>) -> Result<()> {
        let mut struct_obj = struct_rrc.borrow_mut();
        let decl = struct_obj.decl.clone();
        let node = struct_obj.node();
        let struct_node = match &node.kind {
            NodeKind::StructDecl(inner) => inner,
            _ => unreachable!(),
        };

        match struct_obj.status {
            StructStatus::FullyResolved => return Ok(()),
            StructStatus::FieldTypeWip => Err(Error::new(node.span, "Struct depends on itself")),
            _ => Ok(()),
        }?;
        struct_obj.status = StructStatus::FieldTypeWip;
        decl.map_mut(|decl| decl.status = DeclStatus::InProgress);

        let mut decl_error_guard = Decl::error_guard(decl.clone());
        for field_node in &struct_node.fields {
            let field_type = self.gen_type(&field_node.ty)?;
            let field = Field {
                name: field_node.name.to_string(),
                ty: field_type,
            };
            struct_obj.fields.push(field);
        }
        decl_error_guard.success();

        struct_obj.status = StructStatus::FullyResolved;
        decl.map_mut(|decl| decl.status = DeclStatus::Complete);

        Ok(())
    }

    fn resolve_module_type(&mut self, module_rrc: RRC<TModule>) -> Result<()> {
        let node = module_rrc.map(|module| module.node());
        let decl = module_rrc.map(|module| module.decl.clone());
        let module_node = match &node.kind {
            NodeKind::ModuleDecl(inner) => inner,
            _ => unreachable!(),
        };

        match module_rrc.borrow().status {
            ModuleStatus::FullyResolved => return Ok(()),
            ModuleStatus::InProgress => Err(Error::new(node.span, "Module depends on itself")),
            _ => Ok(()),
        }?;
        module_rrc.map_mut(|obj| obj.status = ModuleStatus::InProgress);
        decl.map_mut(|decl| decl.status = DeclStatus::InProgress);

        // Here we'll generate all of the inputs to the module by analyzing each comb.
        // TODO: This should only be done for method combs (i.e. @This() is first parameter), all
        // other combs should be free
        // TODO: Probably should generate all comb methods here as well, can use error guard to
        // detect dependency errors
        let mut reduced_ins: IndexMap<GlobalSymbol, (TaraType, &Node)> = IndexMap::new();
        for member in &module_node.members {
            match &member.kind {
                NodeKind::SubroutineDecl(s_d) => {
                    let ret_ty = self.gen_type(&s_d.return_type)?;
                    // Skip generating combs which don't return anything
                    if ret_ty == TaraType::Void {
                        continue;
                    }
                    module_rrc.map_mut(|obj| -> Result<()> {
                        obj.outs.push((s_d.ident.to_string(), ret_ty.clone()));
                        Ok(())
                    })?;
                    for param in &s_d.params {
                        let param_ty = param.ty.as_ref();
                        let param_type = self.gen_type(param_ty)?;

                        match reduced_ins.entry(param.name) {
                            Entry::Occupied(entry) => {
                                if entry.get().0 != param_type {
                                    module_rrc.map_mut(|obj| obj.status = ModuleStatus::SemaError);
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
        let mut module_obj = module_rrc.borrow_mut();
        for (name, ty_and_node) in reduced_ins {
            let param_type = ty_and_node.0;
            let param_node = ty_and_node.1;
            module_obj
                .ins
                .push((name.to_string(), param_node, param_type));
        }

        module_obj.status = ModuleStatus::FullyResolved;
        decl.map_mut(|decl| decl.status = DeclStatus::Complete);

        Ok(())
    }

    fn resolve_decl_ty_val(&mut self, decl: RRC<Decl>) -> Result<TypedValue> {
        if let (Some(value), Some(ty)) = (&decl.borrow().value, &decl.borrow().ty) {
            let ty_val = TypedValue::new(ty.clone(), value.clone());
            return Ok(ty_val);
        }

        let node = decl.borrow().node();

        if decl.borrow().status == DeclStatus::InProgress {
            Err(Error::new(node.span, "Decl depends on itself"))?
        }

        decl.map_mut(|decl| decl.status = DeclStatus::InProgress);

        let ty_val = self.gen_expr_reachable(node)?;

        decl.map_mut(|decl| {
            let ty_val = ty_val.clone();
            decl.value = Some(ty_val.value.clone());
            decl.ty = Some(ty_val.ty);
            decl.status = DeclStatus::Complete;
        });

        Ok(ty_val)
    }
}

// Housekeeping methods
impl<'a, 'ast, 'ctx, 'blk> AstCodegen<'a, 'ast, 'ctx>
where
    'a: 'blk,
{
    fn setup_namespace(&mut self, node: &Node) -> Result<()> {
        matches!(node.kind, NodeKind::StructDecl(_) | NodeKind::ModuleDecl(_));
        let members = match &node.kind {
            NodeKind::StructDecl(s) => &s.members,
            NodeKind::ModuleDecl(m) => &m.members,
            _ => unreachable!(),
        };
        for member in members {
            match &member.kind {
                NodeKind::VarDecl(v_d) => {
                    // self.table.define_name(v_d.ident, &v_d.expr)?;
                    self.builder.new_decl(member, v_d.ident.as_str())?;
                }
                NodeKind::SubroutineDecl(s_d) => {
                    // self.table.define_name(s_d.ident, member)?;
                    self.builder.new_decl(member, s_d.ident.as_str())?;
                }
                _ => unreachable!(),
            }
        }
        Ok(())
    }
}

// Type checking methods
impl<'a, 'ast, 'ctx, 'blk> AstCodegen<'a, 'ast, 'ctx> {
    fn expect_integral_type(&self, node: &Node) -> Result<()> {
        let actual_type = self.table.get_type(node)?;
        match actual_type {
            TaraType::IntSigned { .. } | TaraType::IntUnsigned { .. } => Ok(()),
            _ => Err(Error::new(
                node.span,
                format!("Expected integral type, found {}", actual_type),
            ))?,
        }
    }

    fn expect_bool_type(&self, node: &Node) -> Result<()> {
        let actual_type = self.table.get_type(node)?;
        match actual_type {
            TaraType::Bool => Ok(()),
            _ => Err(Error::new(
                node.span,
                format!("Expected bool type, found {}", actual_type),
            ))?,
        }
    }

    fn expect_struct_type(&self, node: &Node, actual_type: &TaraType) -> Result<RRC<Struct>> {
        match actual_type {
            TaraType::Struct(s) => Ok(s.clone()),
            _ => Err(Error::new(
                node.span,
                format!("Expected struct type, found {}", actual_type),
            ))?,
        }
    }

    fn expect_type_type(&self, node: &Node, actual_type: &TaraType) -> Result<()> {
        match actual_type {
            TaraType::Type => Ok(()),
            _ => Err(Error::new(
                node.span,
                format!("Expected type type, found {}", actual_type),
            ))?,
        }
    }

    // TODO: Handle values being known at compile time
    fn cast(
        &mut self,
        node: &Node,
        ty_val: TypedValue,
        expected_type: &TaraType,
    ) -> Result<TaraValue> {
        let actual_type = ty_val.ty;
        if actual_type == *expected_type {
            return Ok(ty_val.value);
        }
        match (expected_type, &actual_type) {
            (
                TaraType::IntSigned { width: exp_width },
                TaraType::IntSigned { width: act_width },
            ) => {
                if exp_width > act_width {
                    let actual_mlir_value = ty_val.value.get_runtime_value();
                    let expected_mlir_type = self.builder.get_mlir_tye(self.ctx, expected_type);
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
                    let actual_mlir_value = ty_val.value.get_runtime_value();
                    let expected_mlir_type = self.builder.get_mlir_tye(self.ctx, expected_type);
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
            (TaraType::IntUnsigned { .. }, TaraType::ComptimeInt) => {
                let value = ty_val.value.integer();
                if value < 0 {
                    Err(Error::new(
                        node.span,
                        format!(
                            "Attempted to cast comptime_int {} to {}",
                            value, expected_type
                        ),
                    ))?
                } else {
                    Ok(ty_val.value)
                }
            }
            (TaraType::IntSigned { .. }, TaraType::ComptimeInt) => Ok(ty_val.value),
            _ => Err(Error::new(
                node.span,
                format!("Illegal cast from {} to {}", actual_type, expected_type),
            ))?,
        }
    }
}

#[derive(Clone)]
struct Builder<'ctx, 'blk> {
    parent_decl: RRC<Decl>,
    surr_context: SurroundingContext,
    block: MlirBlockRef<'ctx, 'blk>,
    block_ret_ty: Option<TaraType>,
}

impl<'ctx, 'blk> Builder<'ctx, 'blk> {
    pub fn new(parent_decl: RRC<Decl>, block: MlirBlockRef<'ctx, 'blk>) -> Self {
        Self {
            parent_decl,
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

    pub fn save_with_block_with_decl(
        &mut self,
        ctx: SurroundingContext,
        block: MlirBlockRef<'ctx, '_>,
        decl: RRC<Decl>,
    ) -> Self {
        let save = self.save_with_block(ctx, block);
        self.parent_decl = decl;
        save
    }

    pub fn curr_decl(&self) -> RRC<Decl> {
        self.parent_decl.clone()
    }

    pub fn namespace(&self) -> RRC<Namespace> {
        let curr_decl = self.parent_decl.borrow();
        curr_decl.namespace.clone()
    }

    pub fn add_decl(&self, decl_name: &str, decl: RRC<Decl>) -> Result<()> {
        let curr_decl = self.parent_decl.borrow();
        let namespace = curr_decl.namespace();
        namespace.borrow_mut().add_decl(decl_name, decl.clone())?;
        let mut decl = decl.borrow_mut();
        init_field!(decl, namespace, namespace);
        Ok(())
    }

    pub fn find_decl(&self, decl_name: &str) -> Option<RRC<Decl>> {
        let curr_decl = self.parent_decl.borrow();
        let namespace = curr_decl.namespace();
        Namespace::find_decl(namespace, decl_name)
    }

    pub fn ret_ty(&self) -> TaraType {
        self.block_ret_ty.clone().unwrap_or(TaraType::Void)
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

    pub fn add_argument(&self, r#type: MlirType<'ctx>, loc: Location<'ctx>) -> TaraValue {
        let raw = self.block.add_argument(r#type, loc).to_raw();
        let val = unsafe { MlirValue::from_raw(raw) };
        TaraValue::RuntimeValue(val)
    }

    fn insert_operation<T: Into<MlirOperation<'ctx>> + Clone>(&self, op: T) -> Result<TaraValue> {
        let op_ref = self.block.append_operation(op.into());
        let res = op_ref.result(0)?;
        let raw_val = res.to_raw();
        let mlir_val = unsafe { MlirValue::from_raw(raw_val) };
        Ok(TaraValue::RuntimeValue(mlir_val))
    }

    pub fn new_decl(&self, node: &Node, child_name: &str) -> Result<RRC<Decl>> {
        let curr_decl = self.parent_decl.borrow();
        let decl_name = curr_decl.child_name(child_name);
        let decl = Decl::new(decl_name, node);
        let rrc = RRC::new(decl);
        self.add_decl(child_name, rrc.clone())?;
        Ok(rrc)
    }

    pub fn materialize(
        &self,
        ctx: &'ctx Context,
        loc: Location<'ctx>,
        // Type needed here for constant creation, not type checking
        ty: &TaraType,
        val: &TaraValue,
    ) -> Result<StaticMlirValue> {
        let val = match val {
            TaraValue::RuntimeValue(val) => val.clone(),
            TaraValue::Integer(int) => self
                .gen_constant_int(ctx, loc, ty, *int)?
                .get_runtime_value(),
            TaraValue::Struct(fields_rrc) => {
                let mut materialized_fields = Vec::new();

                let struct_ty_rrc = ty.to_struct();

                let struct_ty = struct_ty_rrc.borrow();
                let fields = fields_rrc.borrow();

                for (field, struct_field) in fields.iter().zip(struct_ty.fields.iter()) {
                    let materialized = self.materialize(ctx, loc, &struct_field.ty, field)?;
                    materialized_fields.push(materialized);
                }

                self.gen_constant_struct(ctx, loc, ty, &materialized_fields)?
                    .get_runtime_value()
            }
            TaraValue::U1Type
            | TaraValue::U8Type
            | TaraValue::I8Type
            | TaraValue::U16Type
            | TaraValue::I16Type
            | TaraValue::U32Type
            | TaraValue::I32Type
            | TaraValue::U64Type
            | TaraValue::I64Type
            | TaraValue::U128Type
            | TaraValue::I128Type
            | TaraValue::UsizeType
            | TaraValue::IsizeType
            | TaraValue::BoolType
            | TaraValue::VoidType
            | TaraValue::TypeType
            | TaraValue::ComptimeIntType
            | TaraValue::UndefinedType
            | TaraValue::EnumLiteralType
            | TaraValue::Undef
            | TaraValue::Zero
            | TaraValue::One
            | TaraValue::VoidValue
            | TaraValue::Unreachable
            | TaraValue::BoolTrue
            | TaraValue::BoolFalse
            | TaraValue::Type(_)
            | TaraValue::Function(_) => unimplemented!(),
        };
        Ok(val)
    }

    // TODO: melior doesn't generate function to set return types which is what comb
    // expects
    pub fn gen_bit_and(
        &self,
        ctx: &'ctx Context,
        loc: Location<'ctx>,
        lhs: StaticMlirValue,
        rhs: StaticMlirValue,
    ) -> Result<TaraValue> {
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
        lhs: StaticMlirValue,
        rhs: StaticMlirValue,
    ) -> Result<TaraValue> {
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
        lhs: StaticMlirValue,
        rhs: StaticMlirValue,
    ) -> Result<TaraValue> {
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
        lhs: StaticMlirValue,
        rhs: StaticMlirValue,
    ) -> Result<TaraValue> {
        let lhs_type = lhs.r#type();
        match self.surr_context {
            SurroundingContext::Sw => {
                let built = AddIOperation::builder(ctx, loc).lhs(lhs).rhs(rhs).build();
                let mut op = built.as_operation().clone();
                // HACK: Currently we don't set any overflow flags, but none should be omitted but
                // the generated bindings insert it anyways
                op.remove_attribute("overflowFlags")?;
                self.insert_operation(op)
            }
            SurroundingContext::Hw => {
                let bin_identifier = MlirIdentifier::new(ctx, "twoState");
                let unit = MlirAttribute::unit(ctx);
                let op = OperationBuilder::new("comb.add", loc)
                    .add_operands(&[lhs, rhs])
                    .add_results(&[lhs_type])
                    .add_attributes(&[(bin_identifier, unit)])
                    .build()?;
                self.insert_operation(op)
            }
        }
    }

    pub fn gen_int_sub(
        &self,
        ctx: &'ctx Context,
        loc: Location<'ctx>,
        lhs: StaticMlirValue,
        rhs: StaticMlirValue,
    ) -> Result<TaraValue> {
        let unit = MlirAttribute::unit(ctx);
        match self.surr_context {
            SurroundingContext::Sw => {
                let built = SubIOperation::builder(ctx, loc).lhs(lhs).rhs(rhs).build();
                let mut op = built.as_operation().clone();
                // HACK: Currently we don't set any overflow flags, but none should be omitted but
                // the generated bindings insert it anyways
                op.remove_attribute("overflowFlags")?;
                self.insert_operation(op)
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
        lhs: StaticMlirValue,
        rhs: StaticMlirValue,
    ) -> Result<TaraValue> {
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
        _: StaticMlirValue,
        _: StaticMlirValue,
    ) -> Result<TaraValue> {
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
        lhs: StaticMlirValue,
        rhs: StaticMlirValue,
    ) -> Result<TaraValue> {
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
        lhs: StaticMlirValue,
        rhs: StaticMlirValue,
    ) -> Result<TaraValue> {
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
        lhs: StaticMlirValue,
        rhs: StaticMlirValue,
    ) -> Result<TaraValue> {
        let predicate_type = self.get_mlir_tye(ctx, &TaraType::IntUnsigned { width: 64 });
        let ret_type = self.get_mlir_tye(ctx, &TaraType::Bool);
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
        lhs: StaticMlirValue,
        rhs: StaticMlirValue,
    ) -> Result<TaraValue> {
        let predicate_type = self.get_mlir_tye(ctx, &TaraType::IntUnsigned { width: 64 });
        let ret_type = self.get_mlir_tye(ctx, &TaraType::Bool);
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
        lhs: StaticMlirValue,
        rhs: StaticMlirValue,
    ) -> Result<TaraValue> {
        let predicate_type = self.get_mlir_tye(ctx, &TaraType::IntUnsigned { width: 64 });
        let ret_type = self.get_mlir_tye(ctx, &TaraType::Bool);
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
        lhs: StaticMlirValue,
        rhs: StaticMlirValue,
    ) -> Result<TaraValue> {
        let predicate_type = self.get_mlir_tye(ctx, &TaraType::IntUnsigned { width: 64 });
        let ret_type = self.get_mlir_tye(ctx, &TaraType::Bool);
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
        lhs: StaticMlirValue,
        rhs: StaticMlirValue,
    ) -> Result<TaraValue> {
        let predicate_type = self.get_mlir_tye(ctx, &TaraType::IntUnsigned { width: 64 });
        let ret_type = self.get_mlir_tye(ctx, &TaraType::Bool);
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
        lhs: StaticMlirValue,
        rhs: StaticMlirValue,
    ) -> Result<TaraValue> {
        let predicate_type = self.get_mlir_tye(ctx, &TaraType::IntUnsigned { width: 64 });
        let ret_type = self.get_mlir_tye(ctx, &TaraType::Bool);
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
        ret_val: Option<StaticMlirValue>,
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

    pub fn gen_constant_int(
        &self,
        ctx: &'ctx Context,
        loc: Location<'ctx>,
        ty: &TaraType,
        val: i64,
    ) -> Result<TaraValue> {
        let mlir_ty = self.get_mlir_tye(ctx, ty);
        let attribute = MlirIntegerAttribute::new(mlir_ty, val);
        match self.surr_context {
            SurroundingContext::Sw => {
                let built = ConstantOperation::builder(ctx, loc)
                    .value(attribute.into())
                    .result(mlir_ty)
                    .build();
                let op = built.as_operation();
                self.insert_operation(op.clone())
            }
            SurroundingContext::Hw => {
                let built = HwConstantOperation::builder(ctx, loc)
                    .value(attribute)
                    .build();
                let op = built.as_operation();
                self.insert_operation(op.clone())
            }
        }
    }

    pub fn gen_constant_struct(
        &self,
        ctx: &'ctx Context,
        loc: Location<'ctx>,
        ty: &TaraType,
        vals: &[StaticMlirValue],
    ) -> Result<TaraValue> {
        let mlir_ty = self.get_mlir_tye(ctx, ty);
        match self.surr_context {
            SurroundingContext::Sw => {
                // Set up initial undefined value
                let mut struct_op: MlirOperation = UndefOperation::builder(ctx, loc)
                    .res(mlir_ty)
                    .build()
                    .into();
                let mut struct_val = self.insert_operation(struct_op.clone())?;

                // Insert into each field
                for (field_i, field) in vals.iter().enumerate() {
                    let position_value =
                        MlirDenseI64ArrayAttribute::new(ctx, &[field_i as i64]).into();
                    struct_op = InsertValueOperation::builder(ctx, loc)
                        .res(mlir_ty)
                        .container(struct_val.get_runtime_value())
                        .value(*field)
                        .position(position_value)
                        .build()
                        .into();
                    struct_val = self.insert_operation(struct_op)?;
                }

                Ok(struct_val)
            }
            SurroundingContext::Hw => {
                let built = HwStructCreateOperation::builder(ctx, loc)
                    .input(vals)
                    .result(mlir_ty)
                    .build();
                let op = built.as_operation();
                self.insert_operation(op.clone())
            }
        }
    }

    pub fn get_mlir_tye(&self, ctx: &'ctx Context, ty: &TaraType) -> MlirType<'ctx> {
        ty.to_mlir_type(ctx, self.surr_context)
    }
}

fn get_maybe_primitive(s: &str) -> Option<TypedValue> {
    let bytes = s.as_bytes();
    let maybe_ty_val = if bytes[0] == b'u' {
        let size = u16::from_str_radix(&s[1..], 10).ok()?;
        let int_type = TaraType::IntUnsigned { width: size };
        Some((TaraType::Type, TaraValue::Type(int_type)))
    } else if bytes[0] == b'i' {
        let size = u16::from_str_radix(&s[1..], 10).ok()?;
        let int_type = TaraType::IntSigned { width: size };
        Some((TaraType::Type, TaraValue::Type(int_type)))
    } else if s == "bool" {
        Some((TaraType::Type, TaraValue::BoolType))
    } else if s == "void" {
        Some((TaraType::Type, TaraValue::VoidType))
    } else if s == "type" {
        Some((TaraType::Type, TaraValue::TypeType))
    } else if s == "true" {
        Some((TaraType::Bool, TaraValue::BoolTrue))
    } else if s == "false" {
        Some((TaraType::Bool, TaraValue::BoolFalse))
    } else {
        None
    };
    maybe_ty_val.map(|(ty, val)| TypedValue::new(ty, val))
}

// The context surrounding an operation. This changes when crossing software/hardware boundaries
// (i.e. anything->module, anything->fn)
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum SurroundingContext {
    Sw,
    Hw,
}
