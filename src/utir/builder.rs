use crate::{
    ast::{Ast, Node, NodeKind, TypedName},
    builtin::{Mutability, Signedness},
    utils::id_arena::{ArenaRef, ExtraArenaContainable, IdArena},
    utir::{error::*, inst::*, Utir},
};
use codespan::Span;
use num_traits::cast::ToPrimitive;
use std::{collections::HashMap, mem::MaybeUninit};
use symbol_table::GlobalSymbol;

type AstResult = Result<UtirInstRef, Failure>;

pub struct Builder<'ast> {
    // Underlying reference to ast used to create UTIR
    ast: &'ast Ast,
    instructions: IdArena<UtirInst>,
    extra_data: IdArena<u32>,
}

impl<'ast> Builder<'ast> {
    pub fn new(ast: &'ast Ast) -> Self {
        return Self {
            ast,
            instructions: IdArena::new(),
            extra_data: IdArena::new(),
        };
    }

    pub fn build(self) -> Result<Utir, Failure> {
        match self.gen_root() {
            Ok(_) => {}
            Err(e) => return Err(e),
        }
        return Ok(self.into());
    }

    fn gen_root(&self) -> AstResult {
        let mut env = Environment::new(self);
        self.gen_struct_inner(&mut env, &self.ast.root)
    }

    fn gen_struct_inner(&self, env: &mut Environment<'_, 'ast, '_>, node: &'ast Node) -> AstResult {
        let struct_inner = match &node.kind {
            NodeKind::StructDecl(inner) => inner,
            _ => unreachable!(),
        };
        let struct_idx = env.reserve_instruction();

        let span = node.span;

        let mut struct_env = env.derive();

        for field in &struct_inner.fields {
            let name = field.name;
            let ty = self.gen_type_expr(&mut struct_env, &*field.ty)?;
            let container_field = ContainerField::new(name, ty);
            struct_env.add_tmp_extra(container_field);
        }

        for member in &struct_inner.members {
            let (name, decl_idx) = match &member.kind {
                NodeKind::VarDecl(var_decl) => {
                    (var_decl.ident, self.gen_var_decl(&mut struct_env, member)?)
                }
                NodeKind::SubroutineDecl(fn_decl) => {
                    (fn_decl.ident, self.gen_fn_decl(&mut struct_env, member)?)
                }
                _ => unreachable!(),
            };
            struct_env.add_binding(name, member, decl_idx);
            let container_member = ContainerMember::new(name, decl_idx);
            struct_env.add_tmp_extra(container_member);
        }

        let fields = struct_inner.fields.len() as u32;
        let decls = struct_inner.members.len() as u32;

        let extra_idx = env.add_extra(ContainerDecl { fields, decls });
        struct_env.finish();

        env.set_instruction(struct_idx, UtirInst::struct_decl(extra_idx, span));
        return Ok(struct_idx);
    }

    fn gen_module_inner(&self, env: &mut Environment<'_, 'ast, '_>, node: &'ast Node) -> AstResult {
        let module_inner = match &node.kind {
            NodeKind::ModuleDecl(inner) => inner,
            _ => unreachable!(),
        };
        let module_idx = env.reserve_instruction();

        let span = node.span;

        let mut module_env = env.derive();

        for field in &module_inner.fields {
            let name = field.name;
            let ty = self.gen_type_expr(&mut module_env, &*field.ty)?;
            let container_field = ContainerField::new(name, ty);
            module_env.add_tmp_extra(container_field);
        }

        for member in &module_inner.members {
            let (name, decl_idx) = match &member.kind {
                NodeKind::VarDecl(var_decl) => {
                    (var_decl.ident, self.gen_var_decl(&mut module_env, member)?)
                }
                NodeKind::SubroutineDecl(comb_decl) => (
                    comb_decl.ident,
                    self.gen_comb_decl(&mut module_env, member)?,
                ),
                _ => unreachable!(),
            };
            module_env.add_binding(name, member, decl_idx);
            let container_member = ContainerMember::new(name, decl_idx);
            module_env.add_tmp_extra(container_member);
        }

        let fields = module_inner.fields.len() as u32;
        let decls = module_inner.members.len() as u32;

        let extra_idx = env.add_extra(ContainerDecl { fields, decls });
        module_env.finish();

        env.set_instruction(module_idx, UtirInst::module_decl(extra_idx, span));
        return Ok(module_idx);
    }

    fn gen_var_decl(&self, env: &mut Environment<'_, 'ast, '_>, node: &'ast Node) -> AstResult {
        let var_decl = match &node.kind {
            NodeKind::VarDecl(inner) => inner,
            _ => unreachable!(),
        };

        // TODO: when instantiating a namespace, include all definitions at once. Doesn't need to
        // refer to instructions, we just need nodes for error reporting. Then when creating local
        // definitions, slowly add to the scope. This method shouldn't require any fixups
        // No this will require fixups because we need InstRefs that don't exist yet

        let ident = var_decl.ident;

        let init_expr = {
            let mut var_env = env.derive();
            self.gen_expr(&mut var_env, &*var_decl.expr)
        }?;

        if let Some(prev_inst) = env.add_binding(ident, node, init_expr) {
            return Err(Failure::shadow(node, prev_inst));
        }
        return Ok(init_expr);
    }

    fn gen_subroutine_decl(
        &self,
        inst: UtirInstRef,
        env: &mut Environment<'_, 'ast, '_>,
        node: &'ast Node,
    ) -> Result<ExtraPayload<SubroutineDecl>, Failure> {
        let subroutine_decl = match &node.kind {
            NodeKind::SubroutineDecl(inner) => inner,
            _ => unreachable!(),
        };

        let span = node.span;

        let ident = subroutine_decl.ident;

        let mut subroutine_env = env.derive();

        for param in &subroutine_decl.params {
            let param_inst = self.gen_param(&mut subroutine_env, param)?;
            subroutine_env.add_tmp_extra(param_inst);
        }

        let return_type = self.gen_type_expr(&mut subroutine_env, &*subroutine_decl.return_type)?;

        let body = {
            let mut body_env = subroutine_env.make_block(InstructionScope::Block, span);
            let body_inst = body_env.block.unwrap();

            let body_nodes = match &subroutine_decl.block.kind {
                NodeKind::Block(block) => &block.body,
                _ => unreachable!(),
            };
            for instr in body_nodes {
                self.gen_expr(&mut body_env, instr)?;
            }
            body_env.add_instruction(UtirInst::RetImplicitVoid);

            body_env.finish();
            body_inst
        };

        let extra_idx = env.add_extra(SubroutineDecl {
            params: subroutine_decl.params.len() as u32,
            return_type,
            body,
        });
        subroutine_env.finish();

        let subroutine_decl = ExtraPayload::new(extra_idx, span);

        if let Some(prev_inst) = env.add_binding(ident, node, inst) {
            return Err(Failure::shadow(node, prev_inst));
        }

        return Ok(subroutine_decl);
    }

    fn gen_fn_decl(&self, env: &mut Environment<'_, 'ast, '_>, node: &'ast Node) -> AstResult {
        let subroutine_idx = env.reserve_instruction();
        let subroutine_decl = self.gen_subroutine_decl(subroutine_idx, env, node)?;
        env.set_instruction(subroutine_idx, UtirInst::FunctionDecl(subroutine_decl));

        return Ok(subroutine_idx);
    }

    fn gen_comb_decl(&self, env: &mut Environment<'_, 'ast, '_>, node: &'ast Node) -> AstResult {
        let subroutine_idx = env.reserve_instruction();
        let subroutine_decl = self.gen_subroutine_decl(subroutine_idx, env, node)?;
        env.set_instruction(subroutine_idx, UtirInst::CombDecl(subroutine_decl));

        return Ok(subroutine_idx);
    }

    fn gen_param(&self, env: &mut Environment<'_, 'ast, '_>, param: &'ast TypedName) -> AstResult {
        let type_expr = self.gen_type_expr(env, &*param.ty)?;
        let span = param.ty.span;

        let inst_ref = env.add_instruction(UtirInst::param(type_expr, span));

        if let Some(prev_inst) = env.add_binding(param.name, &*param.ty, inst_ref) {
            return Err(Failure::shadow(&*param.ty, prev_inst));
        }

        return Ok(inst_ref);
    }

    fn gen_type_expr(&self, env: &mut Environment<'_, 'ast, '_>, node: &'ast Node) -> AstResult {
        match &node.kind {
            NodeKind::StructDecl(_) => self.gen_struct_inner(env, node),
            NodeKind::ModuleDecl(_) => self.gen_module_inner(env, node),
            NodeKind::Identifier(_) => self.resolve_identifier(env, node),
            NodeKind::ReferenceTy(_) => self.gen_inline_block(env, node),
            NodeKind::PointerTy(_) => self.gen_inline_block(env, node),
            _ => unreachable!(),
        }
    }

    fn gen_expr(&self, env: &mut Environment<'_, 'ast, '_>, node: &'ast Node) -> AstResult {
        return match node.kind {
            NodeKind::StructDecl(_) => self.gen_struct_inner(env, node),
            NodeKind::ModuleDecl(_) => self.gen_module_inner(env, node),
            NodeKind::Identifier(_) => self.resolve_identifier(env, node),
            NodeKind::NumberLiteral(_) => self.gen_number_literal(env, node),
            NodeKind::LocalVarDecl(_) => self.gen_local_var_decl(env, node),
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
            | NodeKind::Access(_)
            | NodeKind::Return(_)
            | NodeKind::Negate(_)
            | NodeKind::Deref(_)
            | NodeKind::ReferenceTy(_)
            | NodeKind::PointerTy(_)
            | NodeKind::Call(_)
            | NodeKind::SizedNumberLiteral(_)
            | NodeKind::IfExpr(_)
            | NodeKind::StructInit(_) => self.gen_inline_block(env, node),
            NodeKind::BuiltinCall(_) | NodeKind::Block(_) => unimplemented!(),
            NodeKind::VarDecl(_) | NodeKind::SubroutineDecl(_) => unreachable!(),
        };
    }

    fn gen_struct_init(&self, env: &mut Environment<'_, 'ast, '_>, node: &'ast Node) -> AstResult {
        let struct_init = match &node.kind {
            NodeKind::StructInit(inner) => inner,
            _ => unreachable!(),
        };

        let type_expr = if let Some(type_node) = &struct_init.ty {
            Some(self.gen_type_expr(env, type_node)?)
        } else {
            None
        };

        let mut init_exprs = Vec::new();
        for field in &struct_init.fields {
            let expr_ref = self.gen_expr(env, &*field.ty)?;
            init_exprs.push(FieldInit {
                name: field.name,
                expr: expr_ref,
            });
        }

        let extra_idx = env.add_extra(StructInit {
            type_expr: type_expr.unwrap_or(UtirInstRef::None),
            fields: struct_init.fields.len() as u32,
        });
        for init_expr in init_exprs {
            env.add_extra(init_expr);
        }

        let span = node.span;

        let inst = env.add_instruction(UtirInst::struct_init(extra_idx, span));

        return Ok(inst);
    }

    fn gen_local_var_decl(
        &self,
        env: &mut Environment<'_, 'ast, '_>,
        node: &'ast Node,
    ) -> AstResult {
        let local_var_decl = match &node.kind {
            NodeKind::LocalVarDecl(inner) => inner,
            _ => unreachable!(),
        };
        let span = node.span;

        let ident = local_var_decl.ident;

        let type_inst = if let Some(ty_node) = &local_var_decl.ty {
            Some(self.gen_type_expr(env, &*ty_node)?)
        } else {
            None
        };

        let mut init_expr = self.gen_expr(env, &*local_var_decl.expr)?;

        if let Some(type_inst_ref) = type_inst {
            let bin_op = BinOp {
                lhs: type_inst_ref,
                rhs: init_expr,
            };
            let extra_idx = env.add_extra(bin_op);
            init_expr = env.add_instruction(UtirInst::as_instr(extra_idx, span));
        }

        let ptr_inst = env.add_instruction(UtirInst::MakeAllocConst(init_expr));

        if let Some(prev_inst) = env.add_binding(ident, node, ptr_inst) {
            return Err(Failure::shadow(node, prev_inst));
        }
        return Ok(ptr_inst);
    }

    fn resolve_identifier(
        &self,
        env: &mut Environment<'_, 'ast, '_>,
        node: &'ast Node,
    ) -> AstResult {
        // TODO: do namespace resolving. This requires that parameters become their own
        // instructions so that they can be added to the scopes in functions
        let symbol = match &node.kind {
            NodeKind::Identifier(ident) => ident,
            _ => unreachable!(),
        };
        // Try to resolve to some well known value
        if let Some(inst_ref) = UtirInstRef::from_str(symbol.as_str()) {
            return Ok(inst_ref);
        }
        let span = node.span;
        if let Some(inst) = UtirInst::maybe_primitive(symbol.as_str(), span) {
            return Ok(env.add_instruction(inst));
        }
        let inst = env
            .resolve_binding(*symbol)
            .ok_or_else(|| Failure::unknown(node))?;
        return Ok(inst);
    }

    fn gen_number_literal(
        &self,
        env: &mut Environment<'_, 'ast, '_>,
        node: &'ast Node,
    ) -> AstResult {
        let int = match &node.kind {
            NodeKind::NumberLiteral(inner) => inner,
            _ => unreachable!(),
        };
        if let Some(number) = int.to_u64() {
            let inst = UtirInst::int_literal(number);
            return Ok(env.add_instruction(inst));
        } else {
            unimplemented!("large number literal")
        }
    }

    fn gen_sized_number_literal(
        &self,
        env: &mut Environment<'_, 'ast, '_>,
        node: &'ast Node,
    ) -> AstResult {
        let sized_number_literal = match &node.kind {
            NodeKind::SizedNumberLiteral(inner) => inner,
            _ => unreachable!(),
        };
        if let Some(int) = sized_number_literal.literal.to_u64() {
            let span = node.span;

            let int_inst = env.add_instruction(UtirInst::int_literal(int));
            let type_inst = env.add_instruction(UtirInst::int_type(
                Signedness::Unsigned,
                sized_number_literal.size,
                span,
            ));

            let bin_op = BinOp::new(type_inst, int_inst);
            let extra_idx = env.add_extra(bin_op);

            let span = node.span;

            let as_instr = env.add_instruction(UtirInst::as_instr(extra_idx, span));
            return Ok(as_instr);
        } else {
            unimplemented!("large sized number literal")
        }
    }

    fn gen_inline_expr(&self, env: &mut Environment<'_, 'ast, '_>, node: &'ast Node) -> AstResult {
        let return_value = match &node.kind {
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
            | NodeKind::Div(_) => self.gen_bin_op(env, node),
            NodeKind::Return(_) | NodeKind::Negate(_) | NodeKind::Deref(_) => {
                self.gen_un_op(env, node)
            }
            NodeKind::ReferenceTy(_) | NodeKind::PointerTy(_) => self.gen_ref_ty(env, node),
            NodeKind::Call(_) => self.gen_call(env, node),
            NodeKind::SizedNumberLiteral(_) => self.gen_sized_number_literal(env, node),
            NodeKind::IfExpr(_) => self.gen_branch(env, node),
            NodeKind::Access(_) => self.gen_access(env, node),
            NodeKind::StructInit(_) => self.gen_struct_init(env, node),
            NodeKind::BuiltinCall(_) => unimplemented!(),
            NodeKind::StructDecl(_)
            | NodeKind::ModuleDecl(_)
            | NodeKind::VarDecl(_)
            | NodeKind::LocalVarDecl(_)
            | NodeKind::Identifier(_)
            | NodeKind::NumberLiteral(_)
            | NodeKind::SubroutineDecl(_)
            | NodeKind::Block(_) => unreachable!(),
        }?;

        return Ok(return_value);
    }

    fn gen_inline_block(&self, env: &mut Environment<'_, 'ast, '_>, node: &'ast Node) -> AstResult {
        // Don't add another block instruction if we're already in an inline block
        if env.instruction_scope == InstructionScope::InlineBlock {
            return self.gen_inline_expr(env, node);
        } else {
            let span = node.span;

            let mut inline_block_env = env.make_block(InstructionScope::InlineBlock, span);
            let inline_block = inline_block_env.block.unwrap();

            let return_expr = self.gen_inline_block(&mut inline_block_env, node)?;

            inline_block_env
                .add_instruction(UtirInst::inline_block_break(inline_block, return_expr));

            inline_block_env.finish();

            return Ok(inline_block);
        }
    }

    fn gen_bin_op(&self, env: &mut Environment<'_, 'ast, '_>, node: &'ast Node) -> AstResult {
        let inner = match &node.kind {
            NodeKind::Or(inner)
            | NodeKind::And(inner)
            | NodeKind::Lt(inner)
            | NodeKind::Gt(inner)
            | NodeKind::Lte(inner)
            | NodeKind::Gte(inner)
            | NodeKind::Eq(inner)
            | NodeKind::Neq(inner)
            | NodeKind::BitAnd(inner)
            | NodeKind::BitOr(inner)
            | NodeKind::BitXor(inner)
            | NodeKind::Add(inner)
            | NodeKind::Sub(inner)
            | NodeKind::Mul(inner)
            | NodeKind::Div(inner)
            | NodeKind::Access(inner) => inner,
            NodeKind::StructDecl(_)
            | NodeKind::VarDecl(_)
            | NodeKind::LocalVarDecl(_)
            | NodeKind::ModuleDecl(_)
            | NodeKind::Call(_)
            | NodeKind::Negate(_)
            | NodeKind::Deref(_)
            | NodeKind::Return(_)
            | NodeKind::Identifier(_)
            | NodeKind::ReferenceTy(_)
            | NodeKind::PointerTy(_)
            | NodeKind::NumberLiteral(_)
            | NodeKind::SizedNumberLiteral(_)
            | NodeKind::IfExpr(_)
            | NodeKind::SubroutineDecl(_)
            | NodeKind::StructInit(_)
            | NodeKind::BuiltinCall(_)
            | NodeKind::Block(_) => unreachable!(),
        };

        let lhs_idx = self.gen_expr(env, &*inner.lhs)?;
        let rhs_idx = self.gen_expr(env, &*inner.rhs)?;
        let bin_op = BinOp::new(lhs_idx, rhs_idx);

        let ed_idx = env.add_extra(bin_op);
        let span = node.span;

        let payload = ExtraPayload::new(ed_idx, span);

        let inst = match node.kind {
            NodeKind::Or(_) => UtirInst::Or(payload),
            NodeKind::And(_) => UtirInst::And(payload),
            NodeKind::Lt(_) => UtirInst::Lt(payload),
            NodeKind::Gt(_) => UtirInst::Gt(payload),
            NodeKind::Lte(_) => UtirInst::Lte(payload),
            NodeKind::Gte(_) => UtirInst::Gte(payload),
            NodeKind::Eq(_) => UtirInst::Eq(payload),
            NodeKind::Neq(_) => UtirInst::Neq(payload),
            NodeKind::BitAnd(_) => UtirInst::BitAnd(payload),
            NodeKind::BitOr(_) => UtirInst::BitOr(payload),
            NodeKind::BitXor(_) => UtirInst::BitXor(payload),
            NodeKind::Add(_) => UtirInst::Add(payload),
            NodeKind::Sub(_) => UtirInst::Sub(payload),
            NodeKind::Mul(_) => UtirInst::Mul(payload),
            NodeKind::Div(_) => UtirInst::Div(payload),
            _ => unreachable!(),
        };

        return Ok(env.add_instruction(inst));
    }

    fn gen_un_op(&self, env: &mut Environment<'_, 'ast, '_>, node: &'ast Node) -> AstResult {
        let inner = match &node.kind {
            NodeKind::Negate(inner) | NodeKind::Deref(inner) | NodeKind::Return(Some(inner)) => {
                inner
            }
            _ => unreachable!(),
        };

        let lhs_idx = self.gen_expr(env, &*inner.lhs)?;
        let span = node.span;

        let un_op = UnOp::new(lhs_idx, span);

        let inst = match node.kind {
            NodeKind::Negate(_) => UtirInst::Negate(un_op),
            NodeKind::Deref(_) => UtirInst::Deref(un_op),
            NodeKind::Return(_) => UtirInst::Return(un_op),
            _ => unreachable!(),
        };

        return Ok(env.add_instruction(inst));
    }

    fn gen_ref_ty(&self, env: &mut Environment<'_, 'ast, '_>, node: &'ast Node) -> AstResult {
        let inner = match &node.kind {
            NodeKind::ReferenceTy(inner) | NodeKind::PointerTy(inner) => inner,
            _ => unreachable!(),
        };
        let mutability = match inner.mutability {
            Mutability::Mutable => Mutability::Mutable,
            Mutability::Immutable => Mutability::Immutable,
        };

        let type_expr = self.gen_type_expr(env, &*inner.ty)?;

        let ref_ty = RefTy::new(mutability, type_expr);
        let extra_idx = env.add_extra(ref_ty);

        let span = node.span;

        let payload = ExtraPayload::new(extra_idx, span);

        let inst = match node.kind {
            NodeKind::ReferenceTy(_) => UtirInst::RefTy(payload),
            NodeKind::PointerTy(_) => UtirInst::PtrTy(payload),
            _ => unreachable!(),
        };

        let idx = env.add_instruction(inst);
        return Ok(idx);
    }

    fn gen_call(&self, env: &mut Environment<'_, 'ast, '_>, node: &'ast Node) -> AstResult {
        let call = match &node.kind {
            NodeKind::Call(inner) => inner,
            _ => unreachable!(),
        };

        let lhs = self.gen_expr(env, &*call.call)?;

        let mut args = Vec::new();
        for arg in &call.args {
            args.push(self.gen_expr(env, arg)?);
        }

        let call_args = CallArgs {
            lhs,
            num_args: call.args.len() as u32,
        };

        let extra_idx = env.add_extra(call_args);
        for arg in args {
            env.add_extra(arg);
        }

        let span = node.span;
        let call_inst = env.add_instruction(UtirInst::call(extra_idx, span));

        return Ok(call_inst);
    }

    fn gen_branch(&self, env: &mut Environment<'_, 'ast, '_>, node: &'ast Node) -> AstResult {
        let if_expr = match &node.kind {
            NodeKind::IfExpr(inner) => inner,
            _ => unreachable!(),
        };

        let cond = self.gen_expr(env, &*if_expr.cond)?;

        let branch_instr = env.reserve_instruction();

        let old_scope = env.instruction_scope;
        env.set_instruction_scope(InstructionScope::Global);
        let true_block = {
            let true_span = if_expr.body.span;

            let mut true_env = env.make_block(InstructionScope::InlineBlock, true_span);
            let true_block = true_env.block.unwrap();

            let break_val = self.gen_expr(&mut true_env, &*if_expr.body)?;
            true_env.add_instruction(UtirInst::inline_block_break(true_block, break_val));

            true_env.finish();
            true_block
        };

        let false_block = {
            let false_span = if_expr.else_body.span;

            let mut false_env = env.make_block(InstructionScope::InlineBlock, false_span);
            let false_block = false_env.block.unwrap();

            let break_val = self.gen_expr(&mut false_env, &*if_expr.else_body)?;
            false_env.add_instruction(UtirInst::inline_block_break(false_block, break_val));

            false_env.finish();
            false_block
        };
        env.set_instruction_scope(old_scope);

        let extra_idx = env.add_extra(Branch {
            cond,
            true_block,
            false_block,
        });

        let span = node.span;

        env.set_instruction(branch_instr, UtirInst::branch(extra_idx, span));
        return Ok(branch_instr);
    }

    fn gen_access(&self, env: &mut Environment<'_, 'ast, '_>, node: &'ast Node) -> AstResult {
        let access = match &node.kind {
            NodeKind::Access(inner) => inner,
            _ => unreachable!(),
        };

        let lhs = self.gen_expr(env, &*access.lhs)?;
        let rhs = match access.rhs.kind {
            NodeKind::Identifier(ident) => ident,
            _ => unreachable!(),
        };

        let extra_idx = env.add_extra(Access { lhs, rhs });
        let span = node.span;

        return Ok(env.add_instruction(UtirInst::access(extra_idx, span)));
    }
}

impl<'ast> Into<Utir> for Builder<'ast> {
    fn into(self) -> Utir {
        return Utir {
            instructions: self.instructions.into(),
            extra_data: self.extra_data.into(),
        };
    }
}

struct Environment<'builder, 'inst, 'parent>
where
    'inst: 'builder,
{
    parent: Option<&'parent Environment<'builder, 'inst, 'parent>>,
    // Current bindings of symbols to instructions
    scope: Scope<'inst>,
    // List of arbitrary data which can be used to store temporary data before it is pushed to
    // `Builder.extra_data`
    tmp_extra: IdArena<u32>,
    // Used to control whether instruction refs are added to `extra_data` or not
    instruction_scope: InstructionScope,
    // All enivironments share a builder
    builder: &'builder Builder<'inst>,
    // Possible block instruction that created this environment. While this is an `InstRef`, it is
    // guarenteed to be an `InstIdx`. The instruction value is uninitialized.
    block: Option<UtirInstRef>,
}

impl<'builder, 'inst, 'parent> Environment<'builder, 'inst, 'parent>
where
    'inst: 'builder,
{
    pub fn new(builder: &'builder Builder<'inst>) -> Self {
        return Self {
            parent: None,
            scope: Scope::new(),
            tmp_extra: IdArena::new(),
            instruction_scope: InstructionScope::Global,
            builder,
            block: None,
        };
    }

    pub fn derive(&'parent self) -> Self {
        return Self {
            parent: Some(self),
            scope: Scope::new(),
            tmp_extra: IdArena::new(),
            instruction_scope: self.instruction_scope,
            builder: self.builder,
            block: None,
        };
    }

    pub fn make_block(&'parent self, scope: InstructionScope, span: Span) -> Self {
        let mut env = self.derive();
        env.instruction_scope = scope;

        let extra_idx = unsafe {
            let uninit = MaybeUninit::uninit();
            uninit.assume_init()
        };

        let block = match scope {
            InstructionScope::Block => UtirInst::block(extra_idx, span),
            InstructionScope::InlineBlock => UtirInst::inline_block(extra_idx, span),
            _ => unreachable!(),
        };
        let block_inst = self.add_instruction(block);
        env.block = Some(block_inst);

        return env;
    }

    pub fn add_extra<const N: usize, T: ExtraArenaContainable<N>>(&self, val: T) -> ExtraIdx<T> {
        return self.builder.extra_data.insert_extra(val);
    }

    pub fn add_extra_u32(&self, val: u32) -> ExtraIdx<u32> {
        return self.builder.extra_data.alloc(val);
    }

    pub fn add_tmp_extra<const N: usize, T: ExtraArenaContainable<N>>(&self, val: T) {
        self.tmp_extra.insert_extra(val);
    }

    pub fn add_binding(
        &mut self,
        ident: GlobalSymbol,
        node: &'inst Node,
        inst: UtirInstRef,
    ) -> Option<&'inst Node> {
        let mut env = &*self;
        while let Some(parent) = env.parent {
            if let Some(prev) = parent.scope.bindings.get(&ident) {
                return Some(prev.0);
            }
            env = parent;
        }
        return self.scope.add_binding(ident, node, inst);
    }

    pub fn resolve_binding(&self, ident: GlobalSymbol) -> Option<UtirInstRef> {
        let mut maybe_env = Some(self);
        let mut maybe_binding = None;
        while let Some(env) = maybe_env {
            maybe_binding = env.scope.resolve_binding(ident);

            if maybe_binding.is_some() {
                break;
            }

            maybe_env = env.parent;
        }
        return maybe_binding;
    }

    pub fn add_instruction(&self, inst: UtirInst) -> UtirInstRef {
        let inst_idx = self.builder.instructions.alloc(inst);
        let inst_ref = UtirInstRef::from(inst_idx);
        match self.instruction_scope {
            InstructionScope::Global => {}
            InstructionScope::InlineBlock | InstructionScope::Block => self.add_tmp_extra(inst_ref),
        }
        return inst_ref;
    }

    pub fn reserve_instruction(&self) -> UtirInstRef {
        let uninit = MaybeUninit::uninit();
        let id = self.add_instruction(unsafe { uninit.assume_init() });
        return id;
    }

    pub fn set_instruction(&self, inst: UtirInstRef, val: UtirInst) {
        self.builder.instructions.set(inst.to_inst().unwrap(), val);
    }

    pub fn set_instruction_scope(&mut self, scope: InstructionScope) {
        self.instruction_scope = scope;
    }

    fn finish_block(&mut self) {
        if let Some(block_ref) = self.block {
            let block_idx = block_ref.to_inst().unwrap();

            let block_extra = self.builder.extra_data.insert_extra(Block {
                num_instrs: self.tmp_extra.len() as u32,
            });

            let block = match self.builder.instructions.get(block_idx) {
                UtirInst::Block(inner) => UtirInst::block(block_extra, inner.span),
                UtirInst::InlineBlock(inner) => UtirInst::inline_block(block_extra, inner.span),
                _ => unreachable!(),
            };

            self.builder.instructions.set(block_idx, block);

            self.block = None;

            // Assert that the last instruction in a block is noreturn, this is a required
            // invariant of UTIR blocks
            let last_idx: ExtraIdx<UtirInstRef> = ExtraIdx::from(self.tmp_extra.len() as u32 - 1);
            let last_inst: UtirInstRef = self.tmp_extra.get_extra(last_idx.to_u32());
            if let Some(inst) = last_inst.to_inst() {
                assert!(self.builder.instructions.get(inst).is_no_return());
            } else {
                panic!(
                    "Last instruction in block not noreturn. This is likely a bug in the compiler."
                )
            }
        }
    }

    pub fn finish(mut self) {
        self.finish_block();
        let mut data = ArenaRef::from(self.tmp_extra);
        for extra in data.iter() {
            _ = self.builder.extra_data.alloc(*extra);
        }
    }
}

// Changes behavior of `Environment.add_instruction` in order to allow for flexible instruction
// tracking.
#[derive(Copy, Clone, PartialEq, Eq)]
enum InstructionScope {
    Global,
    Block,
    InlineBlock,
}

struct Scope<'inst> {
    // Current bindings of symbols to nodes
    // TODO: This is currently insufficient for referencing variables defined later in the file
    // even at namespace scope
    bindings: HashMap<GlobalSymbol, (&'inst Node, UtirInstRef)>,
}

impl<'inst> Scope<'inst> {
    pub fn new() -> Self {
        return Self {
            bindings: HashMap::new(),
        };
    }

    pub fn add_binding(
        &mut self,
        ident: GlobalSymbol,
        node: &'inst Node,
        inst: UtirInstRef,
    ) -> Option<&'inst Node> {
        if let Some(prev_inst) = self.bindings.get(&ident) {
            Some(prev_inst.0)
        } else {
            self.bindings.insert(ident, (node, inst));
            None
        }
    }

    pub fn resolve_binding(&self, ident: GlobalSymbol) -> Option<UtirInstRef> {
        if let Some(prev_inst) = self.bindings.get(&ident) {
            Some(prev_inst.1)
        } else {
            None
        }
    }
}
