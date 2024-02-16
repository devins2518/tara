use crate::{
    arena::{ExtraArenaContainable, Id},
    ast::Node,
};
use std::{fmt::Display, num::NonZeroU32};
use symbol_table::GlobalSymbol;

#[derive(Copy, Clone)]
pub enum Inst<'a> {
    StructDecl(ExtraPayload<'a, ContainerDecl>),
    ModuleDecl(ExtraPayload<'a, ContainerDecl>),
    FunctionDecl(ExtraPayload<'a, SubroutineDecl<'a>>),
    CombDecl(ExtraPayload<'a, SubroutineDecl<'a>>),
    DeclVal(Str<'a>),
    InlineBlock(ExtraPayload<'a, Block>),
    InlineBlockBreak(BinOp<'a>),
    As(ExtraPayload<'a, BinOp<'a>>),
    // TODO: integers
    Or(ExtraPayload<'a, BinOp<'a>>),
    And(ExtraPayload<'a, BinOp<'a>>),
    Lt(ExtraPayload<'a, BinOp<'a>>),
    Gt(ExtraPayload<'a, BinOp<'a>>),
    Lte(ExtraPayload<'a, BinOp<'a>>),
    Gte(ExtraPayload<'a, BinOp<'a>>),
    Eq(ExtraPayload<'a, BinOp<'a>>),
    Neq(ExtraPayload<'a, BinOp<'a>>),
    BitAnd(ExtraPayload<'a, BinOp<'a>>),
    BitOr(ExtraPayload<'a, BinOp<'a>>),
    BitXor(ExtraPayload<'a, BinOp<'a>>),
    Add(ExtraPayload<'a, BinOp<'a>>),
    Sub(ExtraPayload<'a, BinOp<'a>>),
    Mul(ExtraPayload<'a, BinOp<'a>>),
    Div(ExtraPayload<'a, BinOp<'a>>),
    // TODO: lhs should be instruction, rhs should be ident
    Access(ExtraPayload<'a, BinOp<'a>>),
    Negate(UnOp<'a>),
    Deref(UnOp<'a>),
    Return(UnOp<'a>),
    RefTy(ExtraPayload<'a, RefTy<'a>>),
    PtrTy(ExtraPayload<'a, RefTy<'a>>),
    Call(ExtraPayload<'a, CallArgs<'a>>),
}

impl<'a> Inst<'a> {
    pub fn struct_decl(extra_idx: ExtraIdx<ContainerDecl>, node_idx: NodeIdx<'a>) -> Self {
        return Self::StructDecl(ExtraPayload::new(extra_idx, node_idx));
    }

    pub fn module_decl(extra_idx: ExtraIdx<ContainerDecl>, node_idx: NodeIdx<'a>) -> Self {
        return Self::ModuleDecl(ExtraPayload::new(extra_idx, node_idx));
    }

    pub fn decl_val(ident: GlobalSymbol, node_idx: NodeIdx<'a>) -> Self {
        return Self::DeclVal(Str::new(ident, node_idx));
    }

    pub fn inline_block(extra_idx: ExtraIdx<Block>, node_idx: NodeIdx<'a>) -> Self {
        return Self::InlineBlock(ExtraPayload::new(extra_idx, node_idx));
    }

    pub fn inline_block_break(lhs: InstIdx<'a>, rhs: InstIdx<'a>) -> Self {
        return Self::InlineBlockBreak(BinOp::new(lhs, rhs));
    }

    pub fn call(extra_idx: ExtraIdx<CallArgs<'a>>, node_idx: NodeIdx<'a>) -> Self {
        return Self::Call(ExtraPayload::new(extra_idx, node_idx));
    }
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct ExtraPayload<'a, T> {
    pub(super) extra_idx: ExtraIdx<T>,
    pub(super) node_idx: NodeIdx<'a>,
}

impl<'a, T> ExtraPayload<'a, T> {
    pub fn new(extra_idx: ExtraIdx<T>, node_idx: NodeIdx<'a>) -> Self {
        return ExtraPayload {
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
pub const SUBROUTINE_DECL_U32S: usize = 3;
#[repr(C)]
#[derive(Copy, Clone)]
pub struct SubroutineDecl<'a> {
    pub(super) params: u32,
    pub(super) return_type: InstIdx<'a>,
    pub(super) body_len: u32,
}

impl ExtraArenaContainable<SUBROUTINE_DECL_U32S> for SubroutineDecl<'_> {}
impl From<[u32; SUBROUTINE_DECL_U32S]> for SubroutineDecl<'_> {
    fn from(value: [u32; SUBROUTINE_DECL_U32S]) -> Self {
        return Self {
            params: value[0],
            return_type: InstIdx::from(value[1]),
            body_len: value[2],
        };
    }
}

impl From<SubroutineDecl<'_>> for [u32; SUBROUTINE_DECL_U32S] {
    fn from(value: SubroutineDecl) -> Self {
        return [value.params, value.return_type.into(), value.body_len];
    }
}

pub const PARAM_U32S: usize = 2;
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Param<'a> {
    pub(super) name: GlobalSymbol,
    pub(super) ty: InstIdx<'a>,
}

impl ExtraArenaContainable<PARAM_U32S> for Param<'_> {}
impl From<[u32; PARAM_U32S]> for Param<'_> {
    fn from(value: [u32; PARAM_U32S]) -> Self {
        return Self {
            name: GlobalSymbol::from(NonZeroU32::new(value[0]).unwrap()),
            ty: InstIdx::from(value[1]),
        };
    }
}

impl From<Param<'_>> for [u32; PARAM_U32S] {
    fn from(value: Param) -> Self {
        let nonzero = NonZeroU32::from(value.name);
        return [nonzero.into(), u32::from(value.ty)];
    }
}

// Followed by `Block.num_instrs` number of `InstIdx`s
pub const BLOCK_U32S: usize = 1;
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Block {
    pub(super) num_instrs: u32,
}

impl Block {
    pub fn new(num_instrs: u32) -> Self {
        return Self { num_instrs };
    }
}

impl ExtraArenaContainable<BLOCK_U32S> for Block {}
impl From<[u32; BLOCK_U32S]> for Block {
    fn from(value: [u32; BLOCK_U32S]) -> Self {
        return Self {
            num_instrs: value[0],
        };
    }
}

impl From<Block> for [u32; BLOCK_U32S] {
    fn from(value: Block) -> Self {
        return [value.num_instrs];
    }
}

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

impl<'a> UnOp<'a> {
    pub fn new(lhs: InstIdx<'a>, node: NodeIdx<'a>) -> Self {
        return Self { lhs, node };
    }
}

#[repr(u32)]
#[derive(Copy, Clone)]
pub enum Mutability {
    Mutable,
    Immutable,
}

impl Display for Mutability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Mutability::Mutable => f.write_str("mut"),
            Mutability::Immutable => f.write_str("const"),
        }
    }
}

pub const REF_TY_U32S: usize = 2;
#[repr(C)]
#[derive(Copy, Clone)]
pub struct RefTy<'a> {
    pub(super) mutability: Mutability,
    pub(super) ty: InstIdx<'a>,
}

impl<'a> RefTy<'a> {
    pub fn new(mutability: Mutability, ty: InstIdx<'a>) -> Self {
        return Self { mutability, ty };
    }
}

impl ExtraArenaContainable<REF_TY_U32S> for RefTy<'_> {}
impl From<[u32; REF_TY_U32S]> for RefTy<'_> {
    fn from(value: [u32; REF_TY_U32S]) -> Self {
        let mutability = match value[0] {
            0 => Mutability::Mutable,
            1 => Mutability::Immutable,
            _ => unreachable!(),
        };
        return Self {
            mutability,
            ty: InstIdx::from(value[1]),
        };
    }
}

impl From<RefTy<'_>> for [u32; REF_TY_U32S] {
    fn from(value: RefTy) -> Self {
        let mut_val = match value.mutability {
            Mutability::Mutable => 0,
            Mutability::Immutable => 1,
        };
        return [mut_val, u32::from(value.ty)];
    }
}

// Followed by `CallArgs.num_args` number of `InstIdx`s
pub const CALL_ARGS_U32S: usize = 2;
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CallArgs<'a> {
    pub(super) lhs: InstIdx<'a>,
    pub(super) num_args: u32,
}

impl ExtraArenaContainable<CALL_ARGS_U32S> for CallArgs<'_> {}
impl From<[u32; CALL_ARGS_U32S]> for CallArgs<'_> {
    fn from(value: [u32; CALL_ARGS_U32S]) -> Self {
        return Self {
            lhs: InstIdx::from(value[0]),
            num_args: value[1],
        };
    }
}

impl From<CallArgs<'_>> for [u32; CALL_ARGS_U32S] {
    fn from(value: CallArgs) -> Self {
        return [value.lhs.into(), value.num_args];
    }
}

// An index into `instructions`
pub type InstIdx<'a> = Id<Inst<'a>>;

// An index into `nodes`
pub type NodeIdx<'a> = Id<&'a Node<'a>>;

// A type for an index into `extra_data` which contains a T
pub type ExtraIdx<T> = Id<T>;

#[test]
fn size_enforcement() {
    assert!(std::mem::size_of::<ExtraPayload<ContainerDecl>>() == 8);
    assert!(std::mem::size_of::<ExtraPayload<SubroutineDecl>>() == 8);
    assert!(std::mem::size_of::<Str>() == 8);
    assert!(std::mem::size_of::<ExtraPayload<Block>>() == 8);
    assert!(std::mem::size_of::<ExtraPayload<BinOp>>() == 8);
    assert!(std::mem::size_of::<UnOp>() == 8);
}
