use crate::{
    arena::{Arena, ArenaRef, ExtraArenaContainable},
    ast::{Ast, Node},
    builtin::{Mutability, Signedness},
    utir::{inst::*, Utir},
};
use num_traits::cast::ToPrimitive;
use std::{collections::HashMap, mem::MaybeUninit};
use symbol_table::GlobalSymbol;

pub struct Builder<'ast> {
    // Underlying reference to ast used to create UTIR
    ast: &'ast Ast<'ast>,
    instructions: Arena<Inst<'ast>>,
    extra_data: Arena<u32>,
    nodes: Arena<&'ast Node<'ast>>,
    failure: Failure<'ast>,
}

impl<'ast> Builder<'ast> {
    pub fn new(ast: &'ast Ast) -> Self {
        return Self {
            ast,
            instructions: Arena::new(),
            extra_data: Arena::new(),
            nodes: Arena::new(),
            failure: Failure::new(),
        };
    }

    pub fn build(self) -> Utir<'ast> {
        _ = &self.gen_root();
        return self.into();
    }

    fn gen_root(&self) {
        let mut env = Environment::new(self);
        let _ = self.gen_struct_inner(&mut env, &self.ast.root);
    }

    fn gen_struct_inner(
        &self,
        env: &mut Environment<'_, 'ast, '_>,
        node: &'ast Node,
    ) -> InstIdx<'ast> {
        let struct_inner = match node {
            Node::StructDecl(inner) => inner,
            _ => unreachable!(),
        };
        let struct_idx = env.reserve_instruction();

        let node_idx = self.add_node(node);

        let mut struct_env = env.derive();

        for field in &struct_inner.fields {
            let name = field.name;
            let ty = self.gen_type_expr(&mut struct_env, &*field.ty);
            let container_field = ContainerField { name, ty };
            struct_env.add_tmp_extra(container_field);
        }

        for member in &struct_inner.members {
            let (name, decl_idx) = match member {
                Node::VarDecl(var_decl) => {
                    (var_decl.ident, self.gen_var_decl(&mut struct_env, member))
                }
                Node::SubroutineDecl(fn_decl) => {
                    (fn_decl.ident, self.gen_fn_decl(&mut struct_env, member))
                }
                _ => unreachable!(),
            };
            let container_member = ContainerMember::new(name, decl_idx);
            struct_env.add_tmp_extra(container_member);
        }

        let fields = struct_inner.fields.len() as u32;
        let decls = struct_inner.members.len() as u32;

        let extra_idx = env.add_extra(ContainerDecl { fields, decls });
        struct_env.finish();

        env.set_instruction(struct_idx, Inst::struct_decl(extra_idx, node_idx));
        return struct_idx;
    }

    fn gen_module_inner(
        &self,
        env: &mut Environment<'_, 'ast, '_>,
        node: &'ast Node,
    ) -> InstIdx<'ast> {
        let module_inner = match node {
            Node::ModuleDecl(inner) => inner,
            _ => unreachable!(),
        };
        let module_idx = env.reserve_instruction();

        let node_idx = self.add_node(node);

        let mut module_env = env.derive();

        for field in &module_inner.fields {
            let name = field.name;
            let ty = self.gen_type_expr(&mut module_env, &*field.ty);
            let container_field = ContainerField { name, ty };
            module_env.add_tmp_extra(container_field);
        }

        for member in &module_inner.members {
            let (name, decl_idx) = match member {
                Node::VarDecl(var_decl) => {
                    (var_decl.ident, self.gen_var_decl(&mut module_env, member))
                }
                Node::SubroutineDecl(comb_decl) => {
                    (comb_decl.ident, self.gen_comb_decl(&mut module_env, member))
                }
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
        return module_idx;
    }

    fn gen_var_decl(
        &self,
        env: &mut Environment<'_, 'ast, '_>,
        node: &'ast Node<'ast>,
    ) -> InstIdx<'ast> {
        let var_decl = match node {
            Node::VarDecl(inner) => inner,
            _ => unreachable!(),
        };

        let ident = var_decl.ident;

        let init_expr = {
            let mut var_env = env.derive();
            self.gen_expr(&mut var_env, &*var_decl.expr)
        };

        env.add_binding(ident, init_expr);
        return init_expr;
    }

    fn gen_fn_decl(
        &self,
        env: &mut Environment<'_, 'ast, '_>,
        node: &'ast Node<'ast>,
    ) -> InstIdx<'ast> {
        let subroutine_decl = match node {
            Node::SubroutineDecl(inner) => inner,
            _ => unreachable!(),
        };
        let subroutine_idx = env.reserve_instruction();

        let mut subroutine_env = env.derive();

        for param in &subroutine_decl.params {
            let param_name = param.name;
            let param_type = self.gen_type_expr(&mut subroutine_env, &*param.ty);
            subroutine_env.add_tmp_extra(Param {
                name: param_name,
                ty: param_type,
            });
        }

        let return_type = self.gen_type_expr(&mut subroutine_env, &*subroutine_decl.return_type);

        subroutine_env.set_instruction_scope(InstructionScope::Block);
        for instr in &subroutine_decl.block {
            self.gen_expr(&mut subroutine_env, instr);
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
        return subroutine_idx;
    }

    fn gen_comb_decl(
        &self,
        env: &mut Environment<'_, 'ast, '_>,
        node: &'ast Node<'ast>,
    ) -> InstIdx<'ast> {
        let subroutine_decl = match node {
            Node::SubroutineDecl(inner) => inner,
            _ => unreachable!(),
        };
        let subroutine_idx = env.reserve_instruction();

        let mut subroutine_env = env.derive();

        for param in &subroutine_decl.params {
            let param_name = param.name;
            let param_type = self.gen_type_expr(&mut subroutine_env, &*param.ty);
            subroutine_env.add_tmp_extra(Param {
                name: param_name,
                ty: param_type,
            });
        }

        let return_type = self.gen_type_expr(&mut subroutine_env, &*subroutine_decl.return_type);

        subroutine_env.set_instruction_scope(InstructionScope::Block);
        for instr in &subroutine_decl.block {
            self.gen_expr(&mut subroutine_env, instr);
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
        return subroutine_idx;
    }

    fn gen_type_expr(
        &self,
        env: &mut Environment<'_, 'ast, '_>,
        node: &'ast Node,
    ) -> InstIdx<'ast> {
        match node {
            Node::StructDecl(_) => self.gen_struct_inner(env, node),
            Node::ModuleDecl(_) => self.gen_module_inner(env, node),
            Node::Identifier(_) => self.gen_identifier(env, node),
            Node::ReferenceTy(_) => self.gen_inline_block(env, node),
            Node::PointerTy(_) => self.gen_inline_block(env, node),
            _ => unreachable!(),
        }
    }

    fn gen_expr(&self, env: &mut Environment<'_, 'ast, '_>, node: &'ast Node) -> InstIdx<'ast> {
        return match node {
            Node::StructDecl(_) => self.gen_struct_inner(env, node),
            Node::ModuleDecl(_) => self.gen_module_inner(env, node),
            Node::Identifier(_) => self.gen_identifier(env, node),
            Node::NumberLiteral(_) => self.gen_number_literal(env, node),
            Node::Or(_)
            | Node::And(_)
            | Node::Lt(_)
            | Node::Gt(_)
            | Node::Lte(_)
            | Node::Gte(_)
            | Node::Eq(_)
            | Node::Neq(_)
            | Node::BitAnd(_)
            | Node::BitOr(_)
            | Node::BitXor(_)
            | Node::Add(_)
            | Node::Sub(_)
            | Node::Mul(_)
            | Node::Div(_)
            | Node::Access(_)
            | Node::Return(_)
            | Node::Negate(_)
            | Node::Deref(_)
            | Node::ReferenceTy(_)
            | Node::PointerTy(_)
            | Node::Call(_) => self.gen_inline_block(env, node),
            Node::VarDecl(_) | Node::SubroutineDecl(_) => unreachable!(),
            Node::SizedNumberLiteral(_) => self.gen_inline_block(env, node),
            Node::IfExpr(_) => unimplemented!("todo if expr"),
        };
    }

    fn gen_identifier(
        &self,
        env: &mut Environment<'_, 'ast, '_>,
        node: &'ast Node,
    ) -> InstIdx<'ast> {
        // TODO: do namespace resolving. This requires that parameters become their own
        // instructions so that they can be added to the scopes in functions
        let symbol = match node {
            Node::Identifier(ident) => ident,
            _ => unreachable!(),
        };
        let node_idx = self.add_node(node);
        let idx = env.add_instruction(Inst::decl_val(*symbol, node_idx));
        return idx;
    }

    fn gen_number_literal(
        &self,
        env: &mut Environment<'_, 'ast, '_>,
        node: &'ast Node,
    ) -> InstIdx<'ast> {
        let int = match node {
            Node::NumberLiteral(inner) => inner,
            _ => unreachable!(),
        };
        if let Some(number) = int.to_u64() {
            let inst = Inst::int_literal(number);
            return env.add_instruction(inst);
        } else {
            unimplemented!("large number literal")
        }
    }

    fn gen_sized_number_literal(
        &self,
        env: &mut Environment<'_, 'ast, '_>,
        node: &'ast Node,
    ) -> InstIdx<'ast> {
        let sized_number_literal = match node {
            Node::SizedNumberLiteral(inner) => inner,
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
            return as_instr;
        } else {
            unimplemented!("large sized number literal")
        }
    }

    fn gen_inline_block(
        &self,
        env: &mut Environment<'_, 'ast, '_>,
        node: &'ast Node,
    ) -> InstIdx<'ast> {
        let inline_block = env.reserve_instruction();

        // TODO: extraneous inline_blocks are created here, we can skip creating one if we detect
        // the parent environment is an inline block. In order to do this, all envs will need to
        // share an arena for extra_data
        let mut inline_block_env = env.derive();
        inline_block_env.set_instruction_scope(InstructionScope::Block);

        let return_value = match node {
            Node::Or(_)
            | Node::And(_)
            | Node::Lt(_)
            | Node::Gt(_)
            | Node::Lte(_)
            | Node::Gte(_)
            | Node::Eq(_)
            | Node::Neq(_)
            | Node::BitAnd(_)
            | Node::BitOr(_)
            | Node::BitXor(_)
            | Node::Add(_)
            | Node::Sub(_)
            | Node::Mul(_)
            | Node::Div(_)
            | Node::Access(_) => self.gen_bin_op(&mut inline_block_env, node),
            Node::Return(_) | Node::Negate(_) | Node::Deref(_) => {
                self.gen_un_op(&mut inline_block_env, node)
            }
            Node::ReferenceTy(_) | Node::PointerTy(_) => {
                self.gen_ref_ty(&mut inline_block_env, node)
            }
            Node::Call(_) => self.gen_call(&mut inline_block_env, node),
            Node::SizedNumberLiteral(_) => {
                self.gen_sized_number_literal(&mut inline_block_env, node)
            }
            Node::StructDecl(_)
            | Node::ModuleDecl(_)
            | Node::VarDecl(_)
            | Node::Identifier(_)
            | Node::NumberLiteral(_)
            | Node::SubroutineDecl(_) => unreachable!(),
            Node::IfExpr(_) => {
                unimplemented!("todo: more inline blocks")
            }
        };
        inline_block_env.add_instruction(Inst::inline_block_break(inline_block, return_value));

        let num_instrs = inline_block_env.tmp_extra.len() as u32;
        let extra_idx = env.add_extra(Block::new(num_instrs));
        inline_block_env.finish();

        let node_idx = self.add_node(node);

        env.set_instruction(inline_block, Inst::inline_block(extra_idx, node_idx));
        return inline_block;
    }

    fn gen_bin_op(&self, env: &mut Environment<'_, 'ast, '_>, node: &'ast Node) -> InstIdx<'ast> {
        let inner = match node {
            Node::Or(inner)
            | Node::And(inner)
            | Node::Lt(inner)
            | Node::Gt(inner)
            | Node::Lte(inner)
            | Node::Gte(inner)
            | Node::Eq(inner)
            | Node::Neq(inner)
            | Node::BitAnd(inner)
            | Node::BitOr(inner)
            | Node::BitXor(inner)
            | Node::Add(inner)
            | Node::Sub(inner)
            | Node::Mul(inner)
            | Node::Div(inner)
            | Node::Access(inner) => inner,
            Node::StructDecl(_)
            | Node::VarDecl(_)
            | Node::ModuleDecl(_)
            | Node::Call(_)
            | Node::Negate(_)
            | Node::Deref(_)
            | Node::Return(_)
            | Node::Identifier(_)
            | Node::ReferenceTy(_)
            | Node::PointerTy(_)
            | Node::NumberLiteral(_)
            | Node::SizedNumberLiteral(_)
            | Node::IfExpr(_)
            | Node::SubroutineDecl(_) => unreachable!(),
        };

        let lhs_idx = self.gen_expr(env, &*inner.lhs);
        let rhs_idx = self.gen_expr(env, &*inner.rhs);
        let bin_op = BinOp::new(lhs_idx, rhs_idx);

        let ed_idx = env.add_extra(bin_op);
        let node_idx = self.add_node(node);

        let payload = ExtraPayload::new(ed_idx, node_idx);

        let inst = match node {
            Node::Or(_) => Inst::Or(payload),
            Node::And(_) => Inst::And(payload),
            Node::Lt(_) => Inst::Lt(payload),
            Node::Gt(_) => Inst::Gt(payload),
            Node::Lte(_) => Inst::Lte(payload),
            Node::Gte(_) => Inst::Gte(payload),
            Node::Eq(_) => Inst::Eq(payload),
            Node::Neq(_) => Inst::Neq(payload),
            Node::BitAnd(_) => Inst::BitAnd(payload),
            Node::BitOr(_) => Inst::BitOr(payload),
            Node::BitXor(_) => Inst::BitXor(payload),
            Node::Add(_) => Inst::Add(payload),
            Node::Sub(_) => Inst::Sub(payload),
            Node::Mul(_) => Inst::Mul(payload),
            Node::Div(_) => Inst::Div(payload),
            Node::Access(_) => Inst::Access(payload),
            _ => unreachable!(),
        };

        return env.add_instruction(inst);
    }

    fn gen_un_op(&self, env: &mut Environment<'_, 'ast, '_>, node: &'ast Node) -> InstIdx<'ast> {
        let inner = match node {
            Node::Negate(inner) | Node::Deref(inner) | Node::Return(inner) => inner,
            _ => unreachable!(),
        };

        let lhs_idx = self.gen_expr(env, &*inner.lhs);
        let node_idx = self.add_node(node);

        let un_op = UnOp::new(lhs_idx, node_idx);

        let inst = match node {
            Node::Negate(_) => Inst::Negate(un_op),
            Node::Deref(_) => Inst::Deref(un_op),
            Node::Return(_) => Inst::Return(un_op),
            _ => unreachable!(),
        };

        return env.add_instruction(inst);
    }

    fn gen_ref_ty(&self, env: &mut Environment<'_, 'ast, '_>, node: &'ast Node) -> InstIdx<'ast> {
        let inner = match node {
            Node::ReferenceTy(inner) | Node::PointerTy(inner) => inner,
            _ => unreachable!(),
        };
        let mutability = match inner.mutability {
            Mutability::Mutable => Mutability::Mutable,
            Mutability::Immutable => Mutability::Immutable,
        };

        let type_expr = self.gen_type_expr(env, &*inner.ty);

        let ref_ty = RefTy::new(mutability, type_expr);
        let extra_idx = env.add_extra(ref_ty);

        let node_idx = self.add_node(node);

        let payload = ExtraPayload::new(extra_idx, node_idx);

        let inst = match node {
            Node::ReferenceTy(_) => Inst::RefTy(payload),
            Node::PointerTy(_) => Inst::PtrTy(payload),
            _ => unreachable!(),
        };

        let idx = env.add_instruction(inst);
        return idx;
    }

    fn gen_call(&self, env: &mut Environment<'_, 'ast, '_>, node: &'ast Node) -> InstIdx<'ast> {
        let call = match node {
            Node::Call(inner) => inner,
            _ => unreachable!(),
        };

        let lhs = self.gen_expr(env, &*call.call);

        let call_inst = env.reserve_instruction();

        let mut call_env = env.derive();
        call_env.set_instruction_scope(InstructionScope::Block);
        for arg in &call.args {
            self.gen_expr(&mut call_env, arg);
        }
        call_env.finish();

        let call_args = CallArgs {
            lhs,
            num_args: call.args.len() as u32,
        };

        let extra_idx = env.add_extra(call_args);
        let node_idx = self.add_node(node);
        env.set_instruction(call_inst, Inst::call(extra_idx, node_idx));

        return call_inst;
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
    scope: Scope<'inst, 'parent>,
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
            scope: self.scope.derive(),
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

    pub fn add_binding(&mut self, ident: GlobalSymbol, inst: InstIdx<'inst>) {
        self.scope.add_binding(ident, inst);
    }

    pub fn add_instruction(&self, inst: Inst<'inst>) -> InstIdx<'inst> {
        let inst_idx = self.builder.instructions.alloc(inst);
        match self.instruction_scope {
            InstructionScope::Global => {}
            InstructionScope::Block => self.add_tmp_extra(inst_idx),
        }
        return inst_idx;
    }

    pub fn reserve_instruction(&self) -> InstIdx<'inst> {
        let uninit = MaybeUninit::uninit();
        let id = self.add_instruction(unsafe { uninit.assume_init() });
        return id;
    }

    pub fn set_instruction(&self, inst: InstIdx<'inst>, val: Inst<'inst>) {
        self.builder.instructions.set(inst, val);
    }

    pub fn set_instruction_scope(&mut self, scope: InstructionScope) {
        self.instruction_scope = scope;
    }

    pub fn is_block_scope(&self) -> bool {
        return self.instruction_scope == InstructionScope::Block;
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

struct Scope<'inst, 'parent> {
    parent: Option<&'parent Scope<'inst, 'parent>>,
    // Current bindings of symbols to instruction indexes
    bindings: HashMap<GlobalSymbol, InstIdx<'inst>>,
    // Instructions generated during the current scope that couldn't be immediately found
    fix_ups: Vec<InstIdx<'inst>>,
}

impl<'inst, 'parent> Scope<'inst, 'parent> {
    pub fn new() -> Self {
        return Self {
            parent: None,
            bindings: HashMap::new(),
            fix_ups: Vec::new(),
        };
    }

    pub fn derive(&'parent self) -> Self {
        return Self {
            parent: Some(self),
            bindings: HashMap::new(),
            fix_ups: Vec::new(),
        };
    }

    pub fn add_binding(&mut self, ident: GlobalSymbol, inst: InstIdx<'inst>) {
        if let Some(prev_inst) = self.bindings.get(&ident) {
            todo!("Error reporting for shadowing")
        } else {
            self.bindings.insert(ident, inst);
        }
    }
}

struct Failure<'ast> {
    fail_node: Option<&'ast Node<'ast>>,
    reason: String,
    notes: Vec<String>,
}

impl<'ast> Failure<'ast> {
    pub fn new() -> Self {
        return Self {
            fail_node: None,
            reason: String::new(),
            notes: Vec::new(),
        };
    }

    pub fn fail_with(&mut self, node: &'ast Node<'ast>, reason: &str) {
        self.fail_node = Some(node);
        self.reason = reason.to_owned();
    }

    pub fn add_node(&mut self, note: &str) {
        self.notes.push(note.to_owned());
    }
}
