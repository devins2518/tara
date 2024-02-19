use crate::{
    arena::{ExtraArenaContainable, Id},
    ast::Node,
    builtin::{Mutability, Signedness},
};
use std::{fmt::Display, num::NonZeroU32};
use symbol_table::GlobalSymbol;

#[derive(Copy, Clone)]
pub enum Inst<'a> {
    StructDecl(ExtraPayload<'a, ContainerDecl>),
    ModuleDecl(ExtraPayload<'a, ContainerDecl>),
    FunctionDecl(ExtraPayload<'a, SubroutineDecl>),
    CombDecl(ExtraPayload<'a, SubroutineDecl>),
    Param(NodePayload<'a, InstRef>),
    DeclVal(Str<'a>),
    InlineBlock(ExtraPayload<'a, Block>),
    InlineBlockBreak(BinOp),
    As(ExtraPayload<'a, BinOp>),
    // TODO: integers
    Or(ExtraPayload<'a, BinOp>),
    And(ExtraPayload<'a, BinOp>),
    Lt(ExtraPayload<'a, BinOp>),
    Gt(ExtraPayload<'a, BinOp>),
    Lte(ExtraPayload<'a, BinOp>),
    Gte(ExtraPayload<'a, BinOp>),
    Eq(ExtraPayload<'a, BinOp>),
    Neq(ExtraPayload<'a, BinOp>),
    BitAnd(ExtraPayload<'a, BinOp>),
    BitOr(ExtraPayload<'a, BinOp>),
    BitXor(ExtraPayload<'a, BinOp>),
    Add(ExtraPayload<'a, BinOp>),
    Sub(ExtraPayload<'a, BinOp>),
    Mul(ExtraPayload<'a, BinOp>),
    Div(ExtraPayload<'a, BinOp>),
    // TODO: lhs should be instruction, rhs should be ident
    Access(ExtraPayload<'a, Access>),
    Negate(UnOp<'a>),
    Deref(UnOp<'a>),
    Return(UnOp<'a>),
    RefTy(ExtraPayload<'a, RefTy>),
    PtrTy(ExtraPayload<'a, RefTy>),
    Call(ExtraPayload<'a, CallArgs>),
    IntLiteral(u64),
    IntType(IntType<'a>),
    Branch(ExtraPayload<'a, Branch>),
}

impl<'a> Inst<'a> {
    pub fn struct_decl(extra_idx: ExtraIdx<ContainerDecl>, node_idx: NodeIdx<'a>) -> Self {
        return Self::StructDecl(ExtraPayload::new(extra_idx, node_idx));
    }

    pub fn module_decl(extra_idx: ExtraIdx<ContainerDecl>, node_idx: NodeIdx<'a>) -> Self {
        return Self::ModuleDecl(ExtraPayload::new(extra_idx, node_idx));
    }

    pub fn param(inst_ref: InstRef, node_idx: NodeIdx<'a>) -> Self {
        return Self::Param(NodePayload::new(inst_ref, node_idx));
    }

    pub fn decl_val(ident: GlobalSymbol, node_idx: NodeIdx<'a>) -> Self {
        return Self::DeclVal(Str::new(ident, node_idx));
    }

    pub fn inline_block(extra_idx: ExtraIdx<Block>, node_idx: NodeIdx<'a>) -> Self {
        return Self::InlineBlock(ExtraPayload::new(extra_idx, node_idx));
    }

    pub fn inline_block_break(lhs: InstRef, rhs: InstRef) -> Self {
        return Self::InlineBlockBreak(BinOp::new(lhs, rhs));
    }

    pub fn call(extra_idx: ExtraIdx<CallArgs>, node_idx: NodeIdx<'a>) -> Self {
        return Self::Call(ExtraPayload::new(extra_idx, node_idx));
    }

    pub fn int_literal(int: u64) -> Self {
        return Self::IntLiteral(int);
    }

    pub fn int_type(signedness: Signedness, size: u16, node: NodeIdx<'a>) -> Self {
        return Self::IntType(NodePayload::new(IntInfo { signedness, size }, node));
    }

    pub fn as_instr(extra_idx: ExtraIdx<BinOp>, node_idx: NodeIdx<'a>) -> Self {
        return Self::As(ExtraPayload::new(extra_idx, node_idx));
    }

    pub fn branch(extra_idx: ExtraIdx<Branch>, node_idx: NodeIdx<'a>) -> Self {
        return Self::Branch(ExtraPayload::new(extra_idx, node_idx));
    }

    pub fn access(extra_idx: ExtraIdx<Access>, node_idx: NodeIdx<'a>) -> Self {
        return Self::Access(ExtraPayload::new(extra_idx, node_idx));
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

#[derive(Copy, Clone)]
#[repr(C)]
pub struct NodePayload<'a, T> {
    pub(super) val: T,
    pub(super) node_idx: NodeIdx<'a>,
}

impl<'a, T> NodePayload<'a, T> {
    pub fn new(val: T, node_idx: NodeIdx<'a>) -> Self {
        return NodePayload { val, node_idx };
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

pub const CONTAINER_FIELD_U32S: usize = NAMED_REF_U32S;
pub type ContainerField = NamedRef;

pub const CONTAINER_MEMBER_U32S: usize = NAMED_REF_U32S;
pub type ContainerMember = NamedRef;

pub const NAMED_REF_U32S: usize = 2;
#[derive(Copy, Clone)]
pub struct NamedRef {
    pub(super) name: GlobalSymbol,
    pub(super) inst_ref: InstRef,
}

impl NamedRef {
    pub fn new(name: GlobalSymbol, inst_ref: InstRef) -> Self {
        return Self { name, inst_ref };
    }
}

impl ExtraArenaContainable<NAMED_REF_U32S> for NamedRef {}
impl From<[u32; NAMED_REF_U32S]> for NamedRef {
    fn from(value: [u32; NAMED_REF_U32S]) -> Self {
        return Self {
            name: GlobalSymbol::from(NonZeroU32::new(value[0]).unwrap()),
            inst_ref: value[1].into(),
        };
    }
}

impl From<NamedRef> for [u32; NAMED_REF_U32S] {
    fn from(value: NamedRef) -> Self {
        let nonzero = NonZeroU32::from(value.name);
        return [nonzero.into(), value.inst_ref.into()];
    }
}

// Followed by `params` number of `InstRef`s which are indexes of `Param`s, then `body_len` number of instructions which make up
// the body of the subroutine
pub const SUBROUTINE_DECL_U32S: usize = 3;
#[repr(C)]
#[derive(Copy, Clone)]
pub struct SubroutineDecl {
    pub(super) params: u32,
    pub(super) return_type: InstRef,
    pub(super) body_len: u32,
}

impl ExtraArenaContainable<SUBROUTINE_DECL_U32S> for SubroutineDecl {}
impl From<[u32; SUBROUTINE_DECL_U32S]> for SubroutineDecl {
    fn from(value: [u32; SUBROUTINE_DECL_U32S]) -> Self {
        return Self {
            params: value[0],
            return_type: InstRef::from(value[1]),
            body_len: value[2],
        };
    }
}

impl From<SubroutineDecl> for [u32; SUBROUTINE_DECL_U32S] {
    fn from(value: SubroutineDecl) -> Self {
        return [value.params, value.return_type.into(), value.body_len];
    }
}

// Followed by `Block.num_instrs` number of `InstRef`s
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

pub type Str<'a> = NodePayload<'a, GlobalSymbol>;

pub const BIN_OP_U32S: usize = 2;
#[repr(C)]
#[derive(Copy, Clone)]
pub struct BinOp {
    pub(super) lhs: InstRef,
    pub(super) rhs: InstRef,
}

impl BinOp {
    pub fn new(lhs: InstRef, rhs: InstRef) -> Self {
        return Self { lhs, rhs };
    }
}

impl ExtraArenaContainable<BIN_OP_U32S> for BinOp {}
impl From<[u32; BIN_OP_U32S]> for BinOp {
    fn from(value: [u32; BIN_OP_U32S]) -> Self {
        return Self {
            lhs: InstRef::from(value[0]),
            rhs: InstRef::from(value[1]),
        };
    }
}

impl From<BinOp> for [u32; BIN_OP_U32S] {
    fn from(value: BinOp) -> Self {
        return [u32::from(value.lhs), u32::from(value.rhs)];
    }
}

pub type UnOp<'a> = NodePayload<'a, InstRef>;

pub const REF_TY_U32S: usize = 2;
#[repr(C)]
#[derive(Copy, Clone)]
pub struct RefTy {
    pub(super) mutability: Mutability,
    pub(super) ty: InstRef,
}

impl RefTy {
    pub fn new(mutability: Mutability, ty: InstRef) -> Self {
        return Self { mutability, ty };
    }
}

impl ExtraArenaContainable<REF_TY_U32S> for RefTy {}
impl From<[u32; REF_TY_U32S]> for RefTy {
    fn from(value: [u32; REF_TY_U32S]) -> Self {
        let mutability = match value[0] {
            0 => Mutability::Mutable,
            1 => Mutability::Immutable,
            _ => unreachable!(),
        };
        return Self {
            mutability,
            ty: InstRef::from(value[1]),
        };
    }
}

impl From<RefTy> for [u32; REF_TY_U32S] {
    fn from(value: RefTy) -> Self {
        let mut_val = match value.mutability {
            Mutability::Mutable => 0,
            Mutability::Immutable => 1,
        };
        return [mut_val, u32::from(value.ty)];
    }
}

// Followed by `CallArgs.num_args` number of `InstRef`s
pub const CALL_ARGS_U32S: usize = 2;
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CallArgs {
    pub(super) lhs: InstRef,
    pub(super) num_args: u32,
}

impl ExtraArenaContainable<CALL_ARGS_U32S> for CallArgs {}
impl From<[u32; CALL_ARGS_U32S]> for CallArgs {
    fn from(value: [u32; CALL_ARGS_U32S]) -> Self {
        return Self {
            lhs: InstRef::from(value[0]),
            num_args: value[1],
        };
    }
}

impl From<CallArgs> for [u32; CALL_ARGS_U32S] {
    fn from(value: CallArgs) -> Self {
        return [value.lhs.into(), value.num_args];
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct IntInfo {
    pub(super) signedness: Signedness,
    pub(super) size: u16,
}

pub type IntType<'a> = NodePayload<'a, IntInfo>;

// Followed by `Branch.true_body_len` number of `InstRef`s followed by `Branch.false_body_len`
// number of `InstRef`s
pub const BRANCH_U32S: usize = 3;
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Branch {
    pub(super) cond: InstRef,
    pub(super) true_body_len: u32,
    pub(super) false_body_len: u32,
}

impl ExtraArenaContainable<BRANCH_U32S> for Branch {}
impl From<[u32; BRANCH_U32S]> for Branch {
    fn from(value: [u32; BRANCH_U32S]) -> Self {
        return Self {
            cond: InstRef::from(value[0]),
            true_body_len: value[1],
            false_body_len: value[1],
        };
    }
}

impl From<Branch> for [u32; BRANCH_U32S] {
    fn from(value: Branch) -> Self {
        return [value.cond.into(), value.true_body_len, value.false_body_len];
    }
}

pub const ACCESS_U32S: usize = 2;
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Access {
    pub(super) lhs: InstRef,
    pub(super) rhs: GlobalSymbol,
}

impl ExtraArenaContainable<ACCESS_U32S> for Access {}
impl From<[u32; ACCESS_U32S]> for Access {
    fn from(value: [u32; ACCESS_U32S]) -> Self {
        return Self {
            lhs: InstRef::from(value[0]),
            rhs: GlobalSymbol::from(NonZeroU32::new(value[1]).unwrap()),
        };
    }
}

impl From<Access> for [u32; ACCESS_U32S] {
    fn from(value: Access) -> Self {
        let nonzero = NonZeroU32::from(value.rhs);
        return [value.lhs.into(), nonzero.into()];
    }
}

// An index into `instructions`
pub type InstIdx<'a> = Id<Inst<'a>>;

// Refs include well known and well typed commonly used values
pub const INST_REF_U32S: usize = 1;
#[repr(u32)]
#[non_exhaustive]
#[derive(Copy, Clone)]
pub enum InstRef {
    IntTypeU8 = 0,
    IntTypeU16 = 1,
    IntTypeU32 = 2,
    IntTypeU64 = 3,
    IntTypeI8 = 4,
    IntTypeI16 = 5,
    IntTypeI32 = 6,
    IntTypeI64 = 7,

    NumLiteral0 = 8,
    NumLiteral1 = 9,

    VoidType = 10,

    BoolType = 11,
    BoolValTrue = 12,
    BoolValFalse = 13,

    ClockType = 14,
    ResetType = 15,

    TypeType = 16,

    // Used to indicate end of known values
    None = 17,
}

impl InstRef {
    pub fn to_inst<'a>(&self) -> Option<InstIdx<'a>> {
        return (*self).into();
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "u8" => Some(Self::IntTypeU8),
            "u16" => Some(Self::IntTypeU16),
            "u32" => Some(Self::IntTypeU32),
            "u64" => Some(Self::IntTypeU64),
            "i8" => Some(Self::IntTypeI8),
            "i16" => Some(Self::IntTypeI16),
            "i32" => Some(Self::IntTypeI32),
            "i64" => Some(Self::IntTypeI64),
            "0" => Some(Self::NumLiteral0),
            "1" => Some(Self::NumLiteral1),
            "void" => Some(Self::VoidType),
            "bool" => Some(Self::BoolType),
            "true" => Some(Self::BoolValTrue),
            "false" => Some(Self::BoolValFalse),
            "clock" => Some(Self::ClockType),
            "reset" => Some(Self::ResetType),
            "type" => Some(Self::TypeType),
            _ => None,
        }
    }
}

impl From<InstRef> for u32 {
    fn from(value: InstRef) -> Self {
        return unsafe { std::mem::transmute(value) };
    }
}

impl From<u32> for InstRef {
    fn from(value: u32) -> Self {
        return unsafe { std::mem::transmute(value) };
    }
}

impl From<InstIdx<'_>> for InstRef {
    fn from(value: InstIdx) -> Self {
        return unsafe { std::mem::transmute(u32::from(value) + u32::from(Self::None) + 1) };
    }
}

impl From<InstRef> for Option<InstIdx<'_>> {
    fn from(value: InstRef) -> Self {
        let u32_val = u32::from(value);
        if u32_val > u32::from(InstRef::None) {
            return Some(InstIdx::from(u32_val - u32::from(InstRef::None) - 1));
        } else {
            return None;
        }
    }
}

impl ExtraArenaContainable<INST_REF_U32S> for InstRef {}
impl From<[u32; INST_REF_U32S]> for InstRef {
    fn from(value: [u32; INST_REF_U32S]) -> Self {
        return value[0].into();
    }
}

impl From<InstRef> for [u32; INST_REF_U32S] {
    fn from(value: InstRef) -> Self {
        return [value.into()];
    }
}

impl Display for InstRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(inst_idx) = self.to_inst() {
            f.write_fmt(format_args!("%{}", u32::from(inst_idx)))?;
        } else {
            let s = match self {
                Self::IntTypeU8 => "@int_type_u8",
                Self::IntTypeU16 => "@int_type_u16",
                Self::IntTypeU32 => "@int_type_u32",
                Self::IntTypeU64 => "@int_type_u64",
                Self::IntTypeI8 => "@int_type_i8",
                Self::IntTypeI16 => "@int_type_i16",
                Self::IntTypeI32 => "@int_type_i32",
                Self::IntTypeI64 => "@int_type_i64",
                Self::NumLiteral0 => "@num_literal_0",
                Self::NumLiteral1 => "@num_literal_1",
                Self::VoidType => "@void_type",
                Self::BoolType => "@bool_type",
                Self::BoolValTrue => "@bool_true",
                Self::BoolValFalse => "@bool_false",
                Self::ClockType => "@clock_type",
                Self::ResetType => "@reset_type",
                Self::TypeType => "@type_type",
                _ => unreachable!(),
            };
            f.write_str(s)?;
        }
        return Ok(());
    }
}

// An index into `nodes`
pub type NodeIdx<'a> = Id<&'a Node<'a>>;

// A type for an index into `extra_data` which contains a T
pub type ExtraIdx<T> = Id<T>;

#[test]
fn size_enforcement() {
    assert!(std::mem::size_of::<ExtraPayload<ContainerDecl>>() == 8);
    assert!(std::mem::size_of::<ExtraPayload<SubroutineDecl>>() == 8);
    assert!(std::mem::size_of::<ExtraPayload<Block>>() == 8);
    assert!(std::mem::size_of::<ExtraPayload<BinOp>>() == 8);
    assert!(std::mem::size_of::<BinOp>() == 8);
    assert!(std::mem::size_of::<UnOp>() == 8);
    assert!(std::mem::size_of::<ExtraPayload<RefTy>>() == 8);
    assert!(std::mem::size_of::<ExtraPayload<CallArgs>>() == 8);
    assert!(std::mem::size_of::<u64>() == 8);
    assert!(std::mem::size_of::<IntType>() == 8);
    assert!(std::mem::size_of::<ExtraPayload<Branch>>() == 8);
}
