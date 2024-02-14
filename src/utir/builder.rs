use crate::{
    arena::{Arena, ArenaRef, ExtraArenaContainable},
    ast::{Ast, Node},
    utir::{inst::*, Utir},
};
use std::collections::HashMap;
use symbol_table::GlobalSymbol;

pub struct Builder<'ast> {
    // Underlying reference to ast used to create UTIR
    ast: &'ast Ast<'ast>,
    instructions: Arena<Inst<'ast>>,
    extra_data: Arena<u32>,
    nodes: Arena<&'ast Node<'ast>>,
}

impl<'ast> Builder<'ast> {
    pub fn new(ast: &'ast Ast) -> Self {
        return Self {
            ast,
            instructions: Arena::new(),
            extra_data: Arena::new(),
            nodes: Arena::new(),
        };
    }

    pub fn build(self) -> Utir<'ast> {
        _ = &self.gen_root();
        return self.into();
    }

    fn gen_root(&self) {
        let mut env = Environment::new();
        let _ = self.gen_struct_inner(&mut env, &self.ast.root);
    }

    fn gen_struct_inner(&self, env: &mut Environment<'ast, '_>, node: &'ast Node) -> InstIdx<'ast> {
        let struct_inner = match node {
            Node::StructDecl(inner) => inner,
            _ => unreachable!(),
        };
        let struct_idx = self.instructions.reserve();

        let node_idx = self.add_node(node);

        let mut struct_env = env.derive();

        for field in &struct_inner.fields {
            let name = field.name;
            let ty = self.gen_type_expr(&mut struct_env, &*field.ty);
            let container_field = ContainerField { name, ty };
            struct_env.add_to_extra(container_field);
        }

        for member in &struct_inner.members {
            let name = match member {
                Node::VarDecl(var_decl) => var_decl.ident,
                Node::SubroutineDecl(fn_decl) => fn_decl.ident,
                _ => unreachable!(),
            };
            let decl_idx = self.gen_var_decl(&mut struct_env, member);
            let container_member = ContainerMember::new(name, decl_idx);
            struct_env.add_to_extra(container_member);
        }

        let fields = struct_inner.fields.len() as u32;
        let decls = struct_inner.members.len() as u32;

        let extra_idx = self.add_extra(ContainerDecl { fields, decls });
        for extra in ArenaRef::from(struct_env.extra_data).data.iter() {
            self.extra_data.alloc(*extra);
        }

        self.instructions
            .set(struct_idx, Inst::struct_decl(extra_idx, node_idx));
        return struct_idx;
    }

    fn gen_module_inner(&self, env: &mut Environment<'ast, '_>, node: &'ast Node) -> InstIdx<'ast> {
        let module_inner = match node {
            Node::ModuleDecl(inner) => inner,
            _ => unreachable!(),
        };
        let module_idx = self.instructions.reserve();

        let node_idx = self.add_node(node);

        let mut module_env = env.derive();

        for field in &module_inner.fields {
            let name = field.name;
            let ty = self.gen_type_expr(&mut module_env, &*field.ty);
            let container_field = ContainerField { name, ty };
            module_env.add_to_extra(container_field);
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
            module_env.add_to_extra(container_member);
        }

        let fields = module_inner.fields.len() as u32;
        let decls = module_inner.members.len() as u32;

        let extra_idx = self.add_extra(ContainerDecl { fields, decls });
        for extra in ArenaRef::from(module_env.extra_data).data.iter() {
            self.extra_data.alloc(*extra);
        }

        self.instructions
            .set(module_idx, Inst::module_decl(extra_idx, node_idx));
        return module_idx;
    }

    fn gen_var_decl(
        &self,
        env: &mut Environment<'ast, '_>,
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

    fn gen_comb_decl(
        &self,
        env: &mut Environment<'ast, '_>,
        node: &'ast Node<'ast>,
    ) -> InstIdx<'ast> {
        unimplemented!("gen comb decl")
    }

    fn gen_type_expr(&self, env: &mut Environment<'ast, '_>, node: &'ast Node) -> InstIdx<'ast> {
        match node {
            Node::StructDecl(_) => self.gen_struct_inner(env, node),
            Node::ModuleDecl(_) => unimplemented!("gen_type_expr module decl"),
            Node::Identifier(_) => self.gen_identifier(env, node),
            Node::ReferenceTy(_) => unimplemented!("gen_type_expr reference ty"),
            Node::PointerTy(_) => unimplemented!("gen_type_expr pointer ty"),
            _ => unreachable!(),
        }
    }

    fn gen_expr(&self, env: &mut Environment<'ast, '_>, node: &'ast Node) -> InstIdx<'ast> {
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
            | Node::Return(_)
            | Node::ReferenceTy(_)
            | Node::PointerTy(_) => self.gen_inline_block(env, node),
            _ => unimplemented!("expr"),
        };
    }

    fn gen_identifier(&self, env: &mut Environment<'ast, '_>, node: &'ast Node) -> InstIdx<'ast> {
        let symbol = match node {
            Node::Identifier(ident) => ident,
            _ => unreachable!(),
        };
        let node_idx = self.nodes.alloc(node);
        let idx = self.instructions.alloc(Inst::decl_val(*symbol, node_idx));
        return idx;
    }

    fn gen_number_literal(
        &self,
        env: &mut Environment<'ast, '_>,
        node: &'ast Node,
    ) -> InstIdx<'ast> {
        unimplemented!("number literal")
    }

    fn gen_inline_block(&self, env: &mut Environment<'ast, '_>, node: &'ast Node) -> InstIdx<'ast> {
        let inline_block = self.instructions.reserve();

        let mut inline_block_env = env.derive();

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
            | Node::Div(_) => self.gen_bin_op(&mut inline_block_env, node),
            Node::ReferenceTy(_) | Node::PointerTy(_) | Node::Return(_) => {
                self.gen_un_op(&mut inline_block_env, node)
            }
            Node::StructDecl(_)
            | Node::ModuleDecl(_)
            | Node::VarDecl(_)
            | Node::Identifier(_)
            | Node::NumberLiteral(_)
            | Node::SizedNumberLiteral(_)
            | Node::SubroutineDecl(_) => unreachable!(),
            Node::Access(_)
            | Node::Call(_)
            | Node::Negate(_)
            | Node::Deref(_)
            | Node::IfExpr(_) => {
                unimplemented!("todo: more inline blocks")
            }
        };

        self.instructions
            .alloc(Inst::inline_block_break(inline_block, return_value));
        unimplemented!()
    }

    fn gen_bin_op(&self, env: &mut Environment<'ast, '_>, node: &'ast Node) -> InstIdx<'ast> {
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
            | Node::Div(inner) => inner,
            _ => unreachable!(),
        };

        let lhs_idx = self.gen_expr(env, &*inner.lhs);
        let rhs_idx = self.gen_expr(env, &*inner.rhs);
        let bin_op = BinOp::new(lhs_idx, rhs_idx);

        let ed_idx = self.extra_data.insert_extra(bin_op);
        let node_idx = self.nodes.alloc(node);

        let payload = Payload::new(ed_idx, node_idx);

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
            _ => unreachable!(),
        };

        return self.instructions.alloc(inst);
    }

    fn gen_un_op(&self, env: &mut Environment<'ast, '_>, node: &'ast Node) -> InstIdx<'ast> {
        unimplemented!()
    }

    fn add_extra<const N: usize, T: ExtraArenaContainable<N>>(&self, val: T) -> ExtraIdx<T> {
        return self.extra_data.insert_extra(val);
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

struct Environment<'inst, 'parent> {
    // Current bindings of symbols to instructions
    scope: Scope<'inst, 'parent>,
    // List of arbitrary data which can be used to store temporary data before it is pushed to
    // `Builder.extra_data`
    extra_data: Arena<u32>,
}

impl<'inst, 'parent> Environment<'inst, 'parent> {
    pub fn new() -> Self {
        return Self {
            scope: Scope::new(),
            extra_data: Arena::new(),
        };
    }

    pub fn derive(&'parent self) -> Self {
        return Self {
            scope: self.scope.derive(),
            extra_data: Arena::new(),
        };
    }

    pub fn add_to_extra<const N: usize, T: ExtraArenaContainable<N>>(&mut self, val: T) {
        self.extra_data.insert_extra(val);
    }

    pub fn add_binding(&mut self, ident: GlobalSymbol, inst: InstIdx<'inst>) {
        self.scope.add_binding(ident, inst);
    }
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
