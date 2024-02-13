mod builder;

use crate::{
    arena::{ArenaRef, ExtraArenaContainable, Id},
    ast::Node,
    Ast,
};
use builder::Builder;
use std::fmt::Display;
use symbol_table::GlobalSymbol;

pub struct Utir<'a> {
    ast: &'a Ast<'a>,
    instructions: ArenaRef<Inst<'a>>,
    extra_data: ArenaRef<u32>,
    nodes: ArenaRef<&'a Node<'a>>,
}

impl<'a> Utir<'a> {
    pub fn gen(ast: &'a Ast) -> Self {
        return Builder::new(ast).build();
    }
}

impl Display for Utir<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unimplemented!()
    }
}

#[derive(Copy, Clone)]
pub enum Inst<'a> {
    StructDecl(Payload<'a, ContainerDecl>),
    ModuleDecl(Payload<'a, ContainerDecl>),
    FunctionDecl(Payload<'a, SubroutineDecl<'a>>),
    CombDecl(Payload<'a, SubroutineDecl<'a>>),
    DeclVal(Str<'a>),
    InlineBlock(Payload<'a, Block<'a>>),
    InlineBlockBreak(Payload<'a, BinOp<'a>>),
    As(Payload<'a, BinOp<'a>>),
    // TODO: integers
    Or(Payload<'a, BinOp<'a>>),
    And(Payload<'a, BinOp<'a>>),
    Lt(Payload<'a, BinOp<'a>>),
    Gt(Payload<'a, BinOp<'a>>),
    Lte(Payload<'a, BinOp<'a>>),
    Gte(Payload<'a, BinOp<'a>>),
    Eq(Payload<'a, BinOp<'a>>),
    Neq(Payload<'a, BinOp<'a>>),
    BitAnd(Payload<'a, BinOp<'a>>),
    BitOr(Payload<'a, BinOp<'a>>),
    BitXor(Payload<'a, BinOp<'a>>),
    Add(Payload<'a, BinOp<'a>>),
    Sub(Payload<'a, BinOp<'a>>),
    Mul(Payload<'a, BinOp<'a>>),
    Div(Payload<'a, BinOp<'a>>),
    Return(UnOp<'a>),
    RefTy(UnOp<'a>),
    PtrTy(UnOp<'a>),
}

impl<'a> Inst<'a> {
    pub fn struct_decl(extra_idx: ExtraIdx<ContainerDecl>, node_idx: NodeIdx<'a>) -> Self {
        return Self::StructDecl(Payload::new(extra_idx, node_idx));
    }
}

#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct Payload<'a, T> {
    extra_idx: ExtraIdx<T>,
    node_idx: NodeIdx<'a>,
}

impl<'a, T> Payload<'a, T> {
    pub fn new(extra_idx: ExtraIdx<T>, node_idx: NodeIdx<'a>) -> Self {
        return Payload {
            extra_idx,
            node_idx,
        };
    }
}

// Followed by a `fields` number of (name: `Str`, type: `InstIdx`) followed by a `decls` number of
// InstIdx
#[derive(Copy, Clone)]
pub struct ContainerDecl {
    fields: u32,
    decls: u32,
}

impl ExtraArenaContainable<2> for ContainerDecl {}

impl From<[u32; 2]> for ContainerDecl {
    fn from(value: [u32; 2]) -> Self {
        return Self {
            fields: value[0],
            decls: value[1],
        };
    }
}

impl From<ContainerDecl> for [u32; 2] {
    fn from(value: ContainerDecl) -> Self {
        return [value.fields, value.decls];
    }
}

// Followed by `params` number of `Param`s, then `body_len` number of instructions which make up
// the body of the subroutine
#[repr(C)]
#[derive(Copy, Clone)]
pub struct SubroutineDecl<'a> {
    params: u32,
    return_type: InstIdx<'a>,
    body_len: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Block<'a>(InstSubList<'a>);

#[repr(C)]
#[derive(Copy, Clone)]
pub struct InstSubList<'a> {
    start: InstIdx<'a>,
    end: InstIdx<'a>,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Str<'a> {
    string: GlobalSymbol,
    node: NodeIdx<'a>,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct BinOp<'a> {
    lhs: InstIdx<'a>,
    rhs: InstIdx<'a>,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct UnOp<'a> {
    lhs: InstIdx<'a>,
    node: NodeIdx<'a>,
}

// An index into `instructions`
type InstIdx<'a> = Id<Inst<'a>>;

// An index into `nodes`
type NodeIdx<'a> = Id<&'a Node<'a>>;

// A type for an index into `extra_data` which contains a T
type ExtraIdx<T> = Id<T>;

#[test]
fn size_enforcement() {
    assert!(std::mem::size_of::<Payload<ContainerDecl>>() == 8);
    assert!(std::mem::size_of::<Payload<SubroutineDecl>>() == 8);
    assert!(std::mem::size_of::<Str>() == 8);
    assert!(std::mem::size_of::<Payload<Block>>() == 8);
    assert!(std::mem::size_of::<Payload<BinOp>>() == 8);
    assert!(std::mem::size_of::<UnOp>() == 8);
}
