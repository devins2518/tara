use crate::{
    arena::{Arena, ExtraArenaContainable},
    ast::{Ast, Node},
    utir::{ContainerDecl, ExtraIdx, Inst, InstIdx, NodeIdx, Utir},
};
use std::{collections::HashMap, marker::PhantomData};
use symbol_table::GlobalSymbol;

pub struct Builder<'ast, 'build> {
    // Underlying reference to ast used to create UTIR
    ast: &'ast Ast<'ast>,
    instructions: Arena<Inst<'ast>>,
    extra_data: Arena<u32>,
    nodes: Arena<&'ast Node<'ast>>,
    _phantom: PhantomData<&'build ()>,
}

impl<'build, 'ast> Builder<'ast, 'build>
where
    'ast: 'build,
{
    pub fn new(ast: &'ast Ast) -> Self {
        return Self {
            ast,
            instructions: Arena::new(),
            extra_data: Arena::new(),
            nodes: Arena::new(),
            _phantom: PhantomData,
        };
    }

    pub fn build(self) -> Utir<'ast> {
        _ = &self.gen_root();
        return self.into();
    }

    fn gen_root(&self) {
        let _ = self.gen_struct_inner(&self.ast.root);
    }

    fn gen_struct_inner(&self, node: &'ast Node) -> InstIdx<'ast> {
        let struct_inner = match node {
            Node::StructDecl(inner) => inner,
            _ => unreachable!(),
        };
        let struct_idx = self.instructions.reserve();
        let node_idx = self.add_node(node);

        let fields = struct_inner.fields.len() as u32;
        let decls = struct_inner.members.len() as u32;

        let extra_idx = self.add_extra(ContainerDecl { fields, decls });

        self.instructions
            .set(struct_idx, Inst::struct_decl(extra_idx, node_idx));
        return struct_idx;
    }

    fn add_extra<const N: usize, T: ExtraArenaContainable<N>>(&self, val: T) -> ExtraIdx<T> {
        return self.extra_data.insert(val);
    }

    fn add_node(&self, node: &'ast Node) -> NodeIdx<'ast> {
        return self.nodes.alloc(node);
    }
}

impl<'build, 'ast> Into<Utir<'ast>> for Builder<'ast, 'build>
where
    'ast: 'build,
{
    fn into(self) -> Utir<'ast> {
        return Utir {
            ast: self.ast,
            instructions: self.instructions.into(),
            extra_data: self.extra_data.into(),
            nodes: self.nodes.into(),
        };
    }
}
