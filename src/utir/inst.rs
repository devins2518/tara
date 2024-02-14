use crate::{
    arena::{ExtraArenaContainable, Id},
    ast::Node,
};
use std::num::NonZeroU32;
use symbol_table::GlobalSymbol;

#[derive(Copy, Clone)]
pub enum Inst<'a> {
    StructDecl(Payload<'a, ContainerDecl>),
    ModuleDecl(Payload<'a, ContainerDecl>),
    FunctionDecl(Payload<'a, SubroutineDecl<'a>>),
    CombDecl(Payload<'a, SubroutineDecl<'a>>),
    DeclVal(Str<'a>),
    InlineBlock(Payload<'a, Block<'a>>),
    InlineBlockBreak(BinOp<'a>),
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

    pub fn module_decl(extra_idx: ExtraIdx<ContainerDecl>, node_idx: NodeIdx<'a>) -> Self {
        return Self::ModuleDecl(Payload::new(extra_idx, node_idx));
    }

    pub fn decl_val(ident: GlobalSymbol, node_idx: NodeIdx<'a>) -> Self {
        return Self::DeclVal(Str::new(ident, node_idx));
    }

    pub fn inline_block_break(lhs: InstIdx<'a>, rhs: InstIdx<'a>) -> Self {
        return Self::InlineBlockBreak(BinOp::new(lhs, rhs));
    }
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct Payload<'a, T> {
    pub(super) extra_idx: ExtraIdx<T>,
    pub(super) node_idx: NodeIdx<'a>,
}

impl<'a, T> Payload<'a, T> {
    pub fn new(extra_idx: ExtraIdx<T>, node_idx: NodeIdx<'a>) -> Self {
        return Payload {
            extra_idx,
            node_idx,
        };
    }
}

pub const CONTAINER_DECL_U32S: usize = 2;
// Followed by a `fields` number of `ContainerField` followed by a `decls` number of
// `ContainerVarDecl`
#[derive(Copy, Clone)]
pub struct ContainerDecl {
    pub(super) fields: u32,
    pub(super) decls: u32,
}

impl ExtraArenaContainable<CONTAINER_DECL_U32S> for ContainerDecl {}

impl From<[u32; CONTAINER_DECL_U32S]> for ContainerDecl {
    fn from(value: [u32; CONTAINER_DECL_U32S]) -> Self {
        return Self {
            fields: value[0],
            decls: value[1],
        };
    }
}

impl From<ContainerDecl> for [u32; CONTAINER_DECL_U32S] {
    fn from(value: ContainerDecl) -> Self {
        return [value.fields, value.decls];
    }
}

pub const CONTAINER_FIELD_U32S: usize = 2;
pub struct ContainerField<'a> {
    pub(super) name: GlobalSymbol,
    pub(super) ty: InstIdx<'a>,
}

impl ExtraArenaContainable<CONTAINER_FIELD_U32S> for ContainerField<'_> {}

impl From<[u32; CONTAINER_FIELD_U32S]> for ContainerField<'_> {
    fn from(value: [u32; CONTAINER_FIELD_U32S]) -> Self {
        return Self {
            name: GlobalSymbol::from(NonZeroU32::new(value[0]).unwrap()),
            ty: value[1].into(),
        };
    }
}

impl From<ContainerField<'_>> for [u32; CONTAINER_FIELD_U32S] {
    fn from(value: ContainerField) -> Self {
        let nonzero = NonZeroU32::from(value.name);
        return [nonzero.into(), value.ty.into()];
    }
}

pub const CONTAINER_MEMBER_U32S: usize = 2;
pub struct ContainerMember<'a> {
    pub(super) name: GlobalSymbol,
    pub(super) value: InstIdx<'a>,
}

impl<'a> ContainerMember<'a> {
    pub fn new(name: GlobalSymbol, value: InstIdx<'a>) -> Self {
        return Self { name, value };
    }
}

impl ExtraArenaContainable<CONTAINER_MEMBER_U32S> for ContainerMember<'_> {}
impl From<[u32; CONTAINER_MEMBER_U32S]> for ContainerMember<'_> {
    fn from(value: [u32; CONTAINER_MEMBER_U32S]) -> Self {
        return Self {
            name: GlobalSymbol::from(NonZeroU32::new(value[0]).unwrap()),
            value: InstIdx::from(value[1]),
        };
    }
}

impl From<ContainerMember<'_>> for [u32; CONTAINER_MEMBER_U32S] {
    fn from(value: ContainerMember) -> Self {
        let nonzero = NonZeroU32::from(value.name);
        return [nonzero.into(), u32::from(value.value)];
    }
}

// Followed by `params` number of `Param`s, then `body_len` number of instructions which make up
// the body of the subroutine
#[repr(C)]
#[derive(Copy, Clone)]
pub struct SubroutineDecl<'a> {
    pub(super) params: u32,
    pub(super) return_type: InstIdx<'a>,
    pub(super) body_len: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Block<'a>(InstSubList<'a>);

#[repr(C)]
#[derive(Copy, Clone)]
pub struct InstSubList<'a> {
    pub(super) start: InstIdx<'a>,
    pub(super) end: InstIdx<'a>,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Str<'a> {
    pub(super) string: GlobalSymbol,
    pub(super) node: NodeIdx<'a>,
}

impl<'a> Str<'a> {
    fn new(string: GlobalSymbol, node: NodeIdx<'a>) -> Self {
        return Self { string, node };
    }
}

pub const BIN_OP_U32S: usize = 2;
#[repr(C)]
#[derive(Copy, Clone)]
pub struct BinOp<'a> {
    pub(super) lhs: InstIdx<'a>,
    pub(super) rhs: InstIdx<'a>,
}

impl<'a> BinOp<'a> {
    pub fn new(lhs: InstIdx<'a>, rhs: InstIdx<'a>) -> Self {
        return Self { lhs, rhs };
    }
}

impl ExtraArenaContainable<BIN_OP_U32S> for BinOp<'_> {}
impl From<[u32; BIN_OP_U32S]> for BinOp<'_> {
    fn from(value: [u32; BIN_OP_U32S]) -> Self {
        return Self {
            lhs: InstIdx::from(value[0]),
            rhs: InstIdx::from(value[1]),
        };
    }
}

impl From<BinOp<'_>> for [u32; BIN_OP_U32S] {
    fn from(value: BinOp) -> Self {
        return [u32::from(value.lhs), u32::from(value.rhs)];
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct UnOp<'a> {
    pub(super) lhs: InstIdx<'a>,
    pub(super) node: NodeIdx<'a>,
}

// An index into `instructions`
pub type InstIdx<'a> = Id<Inst<'a>>;

// An index into `nodes`
pub type NodeIdx<'a> = Id<&'a Node<'a>>;

// A type for an index into `extra_data` which contains a T
pub type ExtraIdx<T> = Id<T>;

#[test]
fn size_enforcement() {
    assert!(std::mem::size_of::<Payload<ContainerDecl>>() == 8);
    assert!(std::mem::size_of::<Payload<SubroutineDecl>>() == 8);
    assert!(std::mem::size_of::<Str>() == 8);
    assert!(std::mem::size_of::<Payload<Block>>() == 8);
    assert!(std::mem::size_of::<Payload<BinOp>>() == 8);
    assert!(std::mem::size_of::<UnOp>() == 8);
}
