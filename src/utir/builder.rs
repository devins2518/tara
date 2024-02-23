use crate::{
    arena::{Arena, ArenaRef, ExtraArenaContainable},
    ast::{Ast, Node, NodeKind, TypedName},
    builtin::{Mutability, Signedness},
    utir::{error::*, inst::*, Utir},
};
use num_traits::cast::ToPrimitive;
use std::{collections::HashMap, mem::MaybeUninit};
use symbol_table::GlobalSymbol;

type AstResult = Result<InstRef, Failure>;

pub struct Builder<'ast> {
    // Underlying reference to ast used to create UTIR
    ast: &'ast Ast<'ast>,
    instructions: Arena<Inst<'ast>>,
    extra_data: Arena<u32>,
    nodes: Arena<&'ast Node<'ast>>,
}

impl<'ast> Builder<'ast> {
    pub fn new(ast: &'ast Ast<'ast>) -> Self {
        return Self {
            ast,
            instructions: Arena::new(),
            extra_data: Arena::new(),
            nodes: Arena::new(),
        };
    }

    pub fn build(self) -> Option<Utir<'ast>> {
        match self.gen_root() {
            Ok(_) => {}
            Err(e) => {
                e.report(self.ast).unwrap();
                return None;
            }
        }
        return Some(self.into());
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

        let node_idx = self.add_node(node);

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

        env.set_instruction(struct_idx, Inst::struct_decl(extra_idx, node_idx));
        return Ok(struct_idx);
    }

    fn gen_module_inner(&self, env: &mut Environment<'_, 'ast, '_>, node: &'ast Node) -> AstResult {
        let module_inner = match &node.kind {
            NodeKind::ModuleDecl(inner) => inner,
            _ => unreachable!(),
        };
        let module_idx = env.reserve_instruction();

        let node_idx = self.add_node(node);

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
            let container_member = ContainerMember::new(name, decl_idx);
            module_env.add_tmp_extra(container_member);
        }

        let fields = module_inner.fields.len() as u32;
        let decls = module_inner.members.len() as u32;

        let extra_idx = env.add_extra(ContainerDecl { fields, decls });
        module_env.finish();

        env.set_instruction(module_idx, Inst::module_decl(extra_idx, node_idx));
        return Ok(module_idx);
    }

    fn gen_var_decl(
        &self,
        env: &mut Environment<'_, 'ast, '_>,
        node: &'ast Node<'ast>,
    ) -> AstResult {
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

    fn gen_fn_decl(
        &self,
        env: &mut Environment<'_, 'ast, '_>,
        node: &'ast Node<'ast>,
    ) -> AstResult {
        let subroutine_decl = match &node.kind {
            NodeKind::SubroutineDecl(inner) => inner,
            _ => unreachable!(),
        };
        let subroutine_idx = env.reserve_instruction();

        let ident = subroutine_decl.ident;

        let mut subroutine_env = env.derive();

        for param in &subroutine_decl.params {
            let param_inst = self.gen_param(&mut subroutine_env, param)?;
            subroutine_env.add_tmp_extra(param_inst);
        }

        let return_type = self.gen_type_expr(&mut subroutine_env, &*subroutine_decl.return_type)?;

        subroutine_env.set_instruction_scope(InstructionScope::Block);
        for instr in &subroutine_decl.block {
            self.gen_expr(&mut subroutine_env, instr)?;
        }

        let body_len = subroutine_env.tmp_extra.len() as u32;
        let extra_idx = env.add_extra(SubroutineDecl {
            params: subroutine_decl.params.len() as u32,
            return_type,
            body_len,
        });
        subroutine_env.finish();

        let node_idx = self.add_node(node);

        let subroutine_decl = ExtraPayload::new(extra_idx, node_idx);

        env.set_instruction(subroutine_idx, Inst::FunctionDecl(subroutine_decl));

        if let Some(prev_inst) = env.add_binding(ident, node, subroutine_idx) {
            return Err(Failure::shadow(node, prev_inst));
        }
        return Ok(subroutine_idx);
    }

    fn gen_comb_decl(
        &self,
        env: &mut Environment<'_, 'ast, '_>,
        node: &'ast Node<'ast>,
    ) -> AstResult {
        let subroutine_decl = match &node.kind {
            NodeKind::SubroutineDecl(inner) => inner,
            _ => unreachable!(),
        };
        let subroutine_idx = env.reserve_instruction();

        let ident = subroutine_decl.ident;

        let mut subroutine_env = env.derive();

        for param in &subroutine_decl.params {
            let param_inst = self.gen_param(&mut subroutine_env, param)?;
            subroutine_env.add_tmp_extra(param_inst);
        }

        let return_type = self.gen_type_expr(&mut subroutine_env, &*subroutine_decl.return_type)?;

        subroutine_env.set_instruction_scope(InstructionScope::Block);
        for instr in &subroutine_decl.block {
            self.gen_expr(&mut subroutine_env, instr)?;
        }

        let body_len = subroutine_env.tmp_extra.len() as u32;
        let extra_idx = env.add_extra(SubroutineDecl {
            params: subroutine_decl.params.len() as u32,
            return_type,
            body_len,
        });
        subroutine_env.finish();

        let node_idx = self.add_node(node);

        let subroutine_decl = ExtraPayload::new(extra_idx, node_idx);

        env.set_instruction(subroutine_idx, Inst::CombDecl(subroutine_decl));

        if let Some(prev_inst) = env.add_binding(ident, node, subroutine_idx) {
            return Err(Failure::shadow(node, prev_inst));
        }
        return Ok(subroutine_idx);
    }

    fn gen_param(&self, env: &mut Environment<'_, 'ast, '_>, param: &'ast TypedName) -> AstResult {
        let type_expr = self.gen_type_expr(env, &*param.ty)?;
        let node_idx = self.add_node(&*param.ty);

        let inst_ref = env.add_instruction(Inst::param(type_expr, node_idx));

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
            | NodeKind::IfExpr(_) => self.gen_inline_block(env, node),
            NodeKind::VarDecl(_) | NodeKind::SubroutineDecl(_) => unreachable!(),
        };
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
        if let Some(inst_ref) = InstRef::from_str(symbol.as_str()) {
            return Ok(inst_ref);
        }
        let node_idx = self.add_node(node);
        if let Some(inst) = Inst::maybe_primitive(symbol.as_str(), node_idx) {
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
            let inst = Inst::int_literal(number);
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
            let node_idx = self.add_node(node);

            let int_inst = env.add_instruction(Inst::int_literal(int));
            let type_inst = env.add_instruction(Inst::int_type(
                Signedness::Unsigned,
                sized_number_literal.size,
                node_idx,
            ));

            let bin_op = BinOp::new(type_inst, int_inst);
            let extra_idx = env.add_extra(bin_op);

            let node_idx = self.add_node(node);

            let as_instr = env.add_instruction(Inst::as_instr(extra_idx, node_idx));
            return Ok(as_instr);
        } else {
            unimplemented!("large sized number literal")
        }
    }

    fn gen_inline_block(&self, env: &mut Environment<'_, 'ast, '_>, node: &'ast Node) -> AstResult {
        let inline_block = env.reserve_instruction();

        // TODO: extraneous inline_blocks are created here, we can skip creating one if we detect
        // the parent environment is an inline block. In order to do this, all envs will need to
        // share an arena for extra_data
        let mut inline_block_env = env.derive();
        inline_block_env.set_instruction_scope(InstructionScope::Block);

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
            | NodeKind::Div(_) => self.gen_bin_op(&mut inline_block_env, node),
            NodeKind::Return(_) | NodeKind::Negate(_) | NodeKind::Deref(_) => {
                self.gen_un_op(&mut inline_block_env, node)
            }
            NodeKind::ReferenceTy(_) | NodeKind::PointerTy(_) => {
                self.gen_ref_ty(&mut inline_block_env, node)
            }
            NodeKind::Call(_) => self.gen_call(&mut inline_block_env, node),
            NodeKind::SizedNumberLiteral(_) => {
                self.gen_sized_number_literal(&mut inline_block_env, node)
            }
            NodeKind::IfExpr(_) => self.gen_branch(&mut inline_block_env, node),
            NodeKind::Access(_) => self.gen_access(&mut inline_block_env, node),
            NodeKind::StructDecl(_)
            | NodeKind::ModuleDecl(_)
            | NodeKind::VarDecl(_)
            | NodeKind::Identifier(_)
            | NodeKind::NumberLiteral(_)
            | NodeKind::SubroutineDecl(_) => unreachable!(),
        }?;
        inline_block_env.add_instruction(Inst::inline_block_break(inline_block, return_value));

        let num_instrs = inline_block_env.tmp_extra.len() as u32;
        let extra_idx = env.add_extra(Block::new(num_instrs));
        inline_block_env.finish();

        let node_idx = self.add_node(node);

        env.set_instruction(inline_block, Inst::inline_block(extra_idx, node_idx));
        return Ok(inline_block);
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
            | NodeKind::SubroutineDecl(_) => unreachable!(),
        };

        let lhs_idx = self.gen_expr(env, &*inner.lhs)?;
        let rhs_idx = self.gen_expr(env, &*inner.rhs)?;
        let bin_op = BinOp::new(lhs_idx, rhs_idx);

        let ed_idx = env.add_extra(bin_op);
        let node_idx = self.add_node(node);

        let payload = ExtraPayload::new(ed_idx, node_idx);

        let inst = match node.kind {
            NodeKind::Or(_) => Inst::Or(payload),
            NodeKind::And(_) => Inst::And(payload),
            NodeKind::Lt(_) => Inst::Lt(payload),
            NodeKind::Gt(_) => Inst::Gt(payload),
            NodeKind::Lte(_) => Inst::Lte(payload),
            NodeKind::Gte(_) => Inst::Gte(payload),
            NodeKind::Eq(_) => Inst::Eq(payload),
            NodeKind::Neq(_) => Inst::Neq(payload),
            NodeKind::BitAnd(_) => Inst::BitAnd(payload),
            NodeKind::BitOr(_) => Inst::BitOr(payload),
            NodeKind::BitXor(_) => Inst::BitXor(payload),
            NodeKind::Add(_) => Inst::Add(payload),
            NodeKind::Sub(_) => Inst::Sub(payload),
            NodeKind::Mul(_) => Inst::Mul(payload),
            NodeKind::Div(_) => Inst::Div(payload),
            _ => unreachable!(),
        };

        return Ok(env.add_instruction(inst));
    }

    fn gen_un_op(&self, env: &mut Environment<'_, 'ast, '_>, node: &'ast Node) -> AstResult {
        let inner = match &node.kind {
            NodeKind::Negate(inner) | NodeKind::Deref(inner) | NodeKind::Return(inner) => inner,
            _ => unreachable!(),
        };

        let lhs_idx = self.gen_expr(env, &*inner.lhs)?;
        let node_idx = self.add_node(node);

        let un_op = UnOp::new(lhs_idx, node_idx);

        let inst = match node.kind {
            NodeKind::Negate(_) => Inst::Negate(un_op),
            NodeKind::Deref(_) => Inst::Deref(un_op),
            NodeKind::Return(_) => Inst::Return(un_op),
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

        let node_idx = self.add_node(node);

        let payload = ExtraPayload::new(extra_idx, node_idx);

        let inst = match node.kind {
            NodeKind::ReferenceTy(_) => Inst::RefTy(payload),
            NodeKind::PointerTy(_) => Inst::PtrTy(payload),
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

        let call_inst = env.reserve_instruction();

        let mut call_env = env.derive();
        call_env.set_instruction_scope(InstructionScope::Block);
        for arg in &call.args {
            let arg_inst = self.gen_expr(&mut call_env, arg)?;
            // HACK: if the arg is a reference, a ref won't be added to the tmp vec so we add it
            // here instead
            if u32::from(arg_inst) < u32::from(call_inst) {
                call_env.add_tmp_extra(arg_inst);
            }
        }

        let call_args = CallArgs {
            lhs,
            num_args: call.args.len() as u32,
        };

        let extra_idx = env.add_extra(call_args);
        call_env.finish();

        let node_idx = self.add_node(node);
        env.set_instruction(call_inst, Inst::call(extra_idx, node_idx));

        return Ok(call_inst);
    }

    fn gen_branch(&self, env: &mut Environment<'_, 'ast, '_>, node: &'ast Node) -> AstResult {
        let if_expr = match &node.kind {
            NodeKind::IfExpr(inner) => inner,
            _ => unreachable!(),
        };

        let cond = self.gen_expr(env, &*if_expr.cond)?;

        let branch_instr = env.reserve_instruction();

        let mut branch_env = env.derive();
        branch_env.set_instruction_scope(InstructionScope::Block);

        let true_instr = self.gen_expr(&mut branch_env, &*if_expr.body)?;
        let _ = branch_env.add_instruction(Inst::inline_block_break(branch_instr, true_instr));
        let true_body_len = branch_env.tmp_extra.len() as u32;

        let false_instr = self.gen_expr(&mut branch_env, &*if_expr.else_body)?;
        let _ = branch_env.add_instruction(Inst::inline_block_break(branch_instr, false_instr));
        let false_body_len = branch_env.tmp_extra.len() as u32 - true_body_len;

        let extra_idx = env.add_extra(Branch {
            cond,
            true_body_len,
            false_body_len,
        });
        branch_env.finish();

        let node_idx = self.add_node(node);

        env.set_instruction(branch_instr, Inst::branch(extra_idx, node_idx));
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
        let node_idx = self.add_node(node);

        return Ok(env.add_instruction(Inst::access(extra_idx, node_idx)));
    }

    fn add_node(&self, node: &'ast Node) -> NodeIdx<'ast> {
        return self.nodes.alloc(node);
    }
}

impl<'ast> Into<Utir<'ast>> for Builder<'ast> {
    fn into(self) -> Utir<'ast> {
        return Utir {
            ast: self.ast,
            instructions: self.instructions.into(),
            extra_data: self.extra_data.into(),
            nodes: self.nodes.into(),
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
    tmp_extra: Arena<u32>,
    // Used to control whether instruction refs are added to `extra_data` or not
    instruction_scope: InstructionScope,
    // All enivironments share a builder
    builder: &'builder Builder<'inst>,
}

impl<'builder, 'inst, 'parent> Environment<'builder, 'inst, 'parent> {
    pub fn new(builder: &'builder Builder<'inst>) -> Self {
        return Self {
            parent: None,
            scope: Scope::new(),
            tmp_extra: Arena::new(),
            instruction_scope: InstructionScope::Global,
            builder,
        };
    }

    pub fn derive(&'parent self) -> Self {
        return Self {
            parent: Some(self),
            scope: Scope::new(),
            tmp_extra: Arena::new(),
            instruction_scope: self.instruction_scope,
            builder: self.builder,
        };
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

    pub fn add_tmp_extra_u32(&self, val: u32) {
        self.tmp_extra.alloc(val);
    }

    pub fn add_binding(
        &mut self,
        ident: GlobalSymbol,
        node: &'inst Node<'inst>,
        inst: InstRef,
    ) -> Option<&'inst Node<'inst>> {
        return self.scope.add_binding(ident, node, inst);
    }

    pub fn resolve_binding(&self, ident: GlobalSymbol) -> Option<InstRef> {
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

    pub fn add_instruction(&self, inst: Inst<'inst>) -> InstRef {
        let inst_idx = self.builder.instructions.alloc(inst);
        let inst_ref = InstRef::from(inst_idx);
        match self.instruction_scope {
            InstructionScope::Global => {}
            InstructionScope::Block => self.add_tmp_extra(inst_ref),
        }
        return inst_ref;
    }

    pub fn reserve_instruction(&self) -> InstRef {
        let uninit = MaybeUninit::uninit();
        let id = self.add_instruction(unsafe { uninit.assume_init() });
        return id;
    }

    pub fn set_instruction(&self, inst: InstRef, val: Inst<'inst>) {
        self.builder.instructions.set(inst.to_inst().unwrap(), val);
    }

    pub fn set_instruction_scope(&mut self, scope: InstructionScope) {
        self.instruction_scope = scope;
    }

    pub fn finish(self) {
        if let Some(parent) = self.parent {
            let data = ArenaRef::from(self.tmp_extra);
            for inst in data.data.iter() {
                parent.add_extra_u32(*inst);
            }
        }
    }
}

// Changes behavior of `Environment.add_instruction` in order to allow for flexible instruction
// tracking.
#[derive(Copy, Clone, PartialEq, Eq)]
enum InstructionScope {
    Global,
    Block,
}

struct Scope<'inst> {
    // Current bindings of symbols to nodes
    // TODO: This is currently insufficient for referencing variables defined later in the file
    // even at namespace scope
    bindings: HashMap<GlobalSymbol, (&'inst Node<'inst>, InstRef)>,
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
        node: &'inst Node<'inst>,
        inst: InstRef,
    ) -> Option<&'inst Node<'inst>> {
        if let Some(prev_inst) = self.bindings.get(&ident) {
            Some(prev_inst.0)
        } else {
            self.bindings.insert(ident, (node, inst));
            None
        }
    }

    pub fn resolve_binding(&self, ident: GlobalSymbol) -> Option<InstRef> {
        if let Some(prev_inst) = self.bindings.get(&ident) {
            Some(prev_inst.1)
        } else {
            None
        }
    }
}
