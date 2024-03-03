use crate::{
    ast::Node,
    builtin::{Mutability, Signedness},
    utils::arena::{ExtraArenaContainable, Id},
};
use std::{fmt::Display, num::NonZeroU32};
use symbol_table::GlobalSymbol;

#[derive(Copy, Clone)]
pub enum UtirInst<'a> {
    StructDecl(ExtraPayload<'a, ContainerDecl>),
    ModuleDecl(ExtraPayload<'a, ContainerDecl>),
    FunctionDecl(ExtraPayload<'a, SubroutineDecl>),
    CombDecl(ExtraPayload<'a, SubroutineDecl>),
    Alloc(ExtraPayload<'a, BinOp>),
    MakeAllocConst(UtirInstRef),
    Param(NodePayload<'a, UtirInstRef>),
    Block(ExtraPayload<'a, Block>),
    BlockBreak(BinOp),
    InlineBlock(ExtraPayload<'a, Block>),
    InlineBlockBreak(BinOp),
    As(ExtraPayload<'a, BinOp>),
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
    StructInit(ExtraPayload<'a, StructInit>),
    // Used to maintain noreturn invariant for subroutine blocks.
    RetImplicitVoid,
}

impl<'a> UtirInst<'a> {
    pub fn struct_decl(extra_idx: ExtraIdx<ContainerDecl>, node_idx: NodeIdx<'a>) -> Self {
        return Self::StructDecl(ExtraPayload::new(extra_idx, node_idx));
    }

    pub fn module_decl(extra_idx: ExtraIdx<ContainerDecl>, node_idx: NodeIdx<'a>) -> Self {
        return Self::ModuleDecl(ExtraPayload::new(extra_idx, node_idx));
    }

    pub fn param(inst_ref: UtirInstRef, node_idx: NodeIdx<'a>) -> Self {
        return Self::Param(NodePayload::new(inst_ref, node_idx));
    }

    pub fn block(extra_idx: ExtraIdx<Block>, node_idx: NodeIdx<'a>) -> Self {
        return Self::Block(ExtraPayload::new(extra_idx, node_idx));
    }

    pub fn block_break(lhs: UtirInstRef, rhs: UtirInstRef) -> Self {
        return Self::BlockBreak(BinOp::new(lhs, rhs));
    }

    pub fn inline_block(extra_idx: ExtraIdx<Block>, node_idx: NodeIdx<'a>) -> Self {
        return Self::InlineBlock(ExtraPayload::new(extra_idx, node_idx));
    }

    pub fn inline_block_break(lhs: UtirInstRef, rhs: UtirInstRef) -> Self {
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

    pub fn struct_init(extra_idx: ExtraIdx<StructInit>, node_idx: NodeIdx<'a>) -> Self {
        return Self::StructInit(ExtraPayload::new(extra_idx, node_idx));
    }

    pub fn maybe_primitive(s: &'a str, node_idx: NodeIdx<'a>) -> Option<Self> {
        let bytes = s.as_bytes();
        if bytes[0] == b'u' {
            let size = u16::from_str_radix(&s[1..], 10).ok()?;
            let int_type = UtirInst::int_type(Signedness::Unsigned, size, node_idx);
            return Some(int_type);
        } else if bytes[0] == b'i' {
            let size = u16::from_str_radix(&s[1..], 10).ok()?;
            let int_type = UtirInst::int_type(Signedness::Signed, size, node_idx);
            return Some(int_type);
        }
        None
    }

    pub fn is_no_return(&self) -> bool {
        match self {
            Self::StructDecl(_)
            | Self::ModuleDecl(_)
            | Self::FunctionDecl(_)
            | Self::CombDecl(_)
            | Self::Param(_)
            | Self::Block(_)
            | Self::InlineBlock(_)
            | Self::As(_)
            | Self::Or(_)
            | Self::And(_)
            | Self::Lt(_)
            | Self::Gt(_)
            | Self::Lte(_)
            | Self::Gte(_)
            | Self::Eq(_)
            | Self::Neq(_)
            | Self::BitAnd(_)
            | Self::BitOr(_)
            | Self::BitXor(_)
            | Self::Add(_)
            | Self::Sub(_)
            | Self::Mul(_)
            | Self::Div(_)
            | Self::Access(_)
            | Self::Negate(_)
            | Self::Deref(_)
            | Self::Return(_)
            | Self::RefTy(_)
            | Self::PtrTy(_)
            | Self::Call(_)
            | Self::IntLiteral(_)
            | Self::IntType(_)
            | Self::Alloc(_)
            | Self::MakeAllocConst(_)
            | Self::StructInit(_) => false,
            Self::BlockBreak(_)
            | Self::InlineBlockBreak(_)
            | Self::Branch(_)
            | Self::RetImplicitVoid => true,
        }
    }
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct ExtraPayload<'a, T> {
    pub extra_idx: ExtraIdx<T>,
    pub node_idx: NodeIdx<'a>,
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
    pub val: T,
    pub node_idx: NodeIdx<'a>,
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
    pub fields: u32,
    pub decls: u32,
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
    pub name: GlobalSymbol,
    pub inst_ref: UtirInstRef,
}

impl NamedRef {
    pub fn new(name: GlobalSymbol, inst_ref: UtirInstRef) -> Self {
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

// Followed by `params` number of `InstRef`s which are indexes of `Param`s, then a `body` InstRef
// which is a block
pub const SUBROUTINE_DECL_U32S: usize = 3;
#[repr(C)]
#[derive(Copy, Clone)]
pub struct SubroutineDecl {
    pub params: u32,
    pub return_type: UtirInstRef,
    pub body: UtirInstRef,
}

impl ExtraArenaContainable<SUBROUTINE_DECL_U32S> for SubroutineDecl {}
impl From<[u32; SUBROUTINE_DECL_U32S]> for SubroutineDecl {
    fn from(value: [u32; SUBROUTINE_DECL_U32S]) -> Self {
        return Self {
            params: value[0],
            return_type: UtirInstRef::from(value[1]),
            body: UtirInstRef::from(value[2]),
        };
    }
}

impl From<SubroutineDecl> for [u32; SUBROUTINE_DECL_U32S] {
    fn from(value: SubroutineDecl) -> Self {
        return [value.params, value.return_type.into(), value.body.into()];
    }
}

// Followed by `Block.num_instrs` number of `InstRef`s
pub const BLOCK_U32S: usize = 1;
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Block {
    pub num_instrs: u32,
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

pub const BIN_OP_U32S: usize = 2;
#[repr(C)]
#[derive(Copy, Clone)]
pub struct BinOp {
    pub lhs: UtirInstRef,
    pub rhs: UtirInstRef,
}

impl BinOp {
    pub fn new(lhs: UtirInstRef, rhs: UtirInstRef) -> Self {
        return Self { lhs, rhs };
    }
}

impl ExtraArenaContainable<BIN_OP_U32S> for BinOp {}
impl From<[u32; BIN_OP_U32S]> for BinOp {
    fn from(value: [u32; BIN_OP_U32S]) -> Self {
        return Self {
            lhs: UtirInstRef::from(value[0]),
            rhs: UtirInstRef::from(value[1]),
        };
    }
}

impl From<BinOp> for [u32; BIN_OP_U32S] {
    fn from(value: BinOp) -> Self {
        return [u32::from(value.lhs), u32::from(value.rhs)];
    }
}

pub type UnOp<'a> = NodePayload<'a, UtirInstRef>;

pub const REF_TY_U32S: usize = 2;
#[repr(C)]
#[derive(Copy, Clone)]
pub struct RefTy {
    pub mutability: Mutability,
    pub ty: UtirInstRef,
}

impl RefTy {
    pub fn new(mutability: Mutability, ty: UtirInstRef) -> Self {
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
            ty: UtirInstRef::from(value[1]),
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
    pub lhs: UtirInstRef,
    pub num_args: u32,
}

impl ExtraArenaContainable<CALL_ARGS_U32S> for CallArgs {}
impl From<[u32; CALL_ARGS_U32S]> for CallArgs {
    fn from(value: [u32; CALL_ARGS_U32S]) -> Self {
        return Self {
            lhs: UtirInstRef::from(value[0]),
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
    pub signedness: Signedness,
    pub size: u16,
}

pub type IntType<'a> = NodePayload<'a, IntInfo>;

// Followed by `Branch.true_body_len` number of `InstRef`s followed by `Branch.false_body_len`
// number of `InstRef`s
pub const BRANCH_U32S: usize = 3;
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Branch {
    pub cond: UtirInstRef,
    pub true_block: UtirInstRef,
    pub false_block: UtirInstRef,
}

impl ExtraArenaContainable<BRANCH_U32S> for Branch {}
impl From<[u32; BRANCH_U32S]> for Branch {
    fn from(value: [u32; BRANCH_U32S]) -> Self {
        return Self {
            cond: UtirInstRef::from(value[0]),
            true_block: UtirInstRef::from(value[1]),
            false_block: UtirInstRef::from(value[2]),
        };
    }
}

impl From<Branch> for [u32; BRANCH_U32S] {
    fn from(value: Branch) -> Self {
        return [
            value.cond.into(),
            value.true_block.into(),
            value.false_block.into(),
        ];
    }
}

pub const ACCESS_U32S: usize = 2;
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Access {
    pub lhs: UtirInstRef,
    pub rhs: GlobalSymbol,
}

impl ExtraArenaContainable<ACCESS_U32S> for Access {}
impl From<[u32; ACCESS_U32S]> for Access {
    fn from(value: [u32; ACCESS_U32S]) -> Self {
        return Self {
            lhs: UtirInstRef::from(value[0]),
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

pub const STRUCT_INIT_U32S: usize = 2;
// Followed by `StructInit.fields` number of `FieldInit`s
#[repr(C)]
#[derive(Copy, Clone)]
pub struct StructInit {
    pub type_expr: UtirInstRef,
    pub fields: u32,
}

impl ExtraArenaContainable<STRUCT_INIT_U32S> for StructInit {}
impl From<[u32; STRUCT_INIT_U32S]> for StructInit {
    fn from(value: [u32; STRUCT_INIT_U32S]) -> Self {
        return Self {
            type_expr: UtirInstRef::from(value[0]),
            fields: value[1],
        };
    }
}

impl From<StructInit> for [u32; STRUCT_INIT_U32S] {
    fn from(value: StructInit) -> Self {
        return [value.type_expr.into(), value.fields];
    }
}

pub const FIELD_INIT_U32S: usize = 2;
#[repr(C)]
#[derive(Copy, Clone)]
pub struct FieldInit {
    pub name: GlobalSymbol,
    pub expr: UtirInstRef,
}

impl ExtraArenaContainable<FIELD_INIT_U32S> for FieldInit {}
impl From<[u32; FIELD_INIT_U32S]> for FieldInit {
    fn from(value: [u32; FIELD_INIT_U32S]) -> Self {
        return Self {
            name: GlobalSymbol::from(NonZeroU32::new(value[0]).unwrap()),
            expr: UtirInstRef::from(value[1]),
        };
    }
}

impl From<FieldInit> for [u32; FIELD_INIT_U32S] {
    fn from(value: FieldInit) -> Self {
        let nonzero = NonZeroU32::from(value.name);
        return [nonzero.into(), value.expr.into()];
    }
}

// An index into `instructions`
pub type UtirInstIdx<'a> = Id<UtirInst<'a>>;

// Refs include well known and well typed commonly used values
pub const INST_REF_U32S: usize = 1;
#[repr(u32)]
#[non_exhaustive]
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum UtirInstRef {
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

    SigType = 17,

    Undefined = 18,

    // Used to indicate end of known values
    None = 19,
}

impl UtirInstRef {
    pub fn to_inst<'a>(&self) -> Option<UtirInstIdx<'a>> {
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
            "sig" => Some(Self::SigType),
            "undefined" => Some(Self::Undefined),
            _ => None,
        }
    }
}

impl From<UtirInstRef> for u32 {
    fn from(value: UtirInstRef) -> Self {
        return unsafe { std::mem::transmute(value) };
    }
}

impl From<u32> for UtirInstRef {
    fn from(value: u32) -> Self {
        return unsafe { std::mem::transmute(value) };
    }
}

impl From<UtirInstIdx<'_>> for UtirInstRef {
    fn from(value: UtirInstIdx) -> Self {
        return unsafe { std::mem::transmute(u32::from(value) + u32::from(Self::None) + 1) };
    }
}

impl From<UtirInstRef> for Option<UtirInstIdx<'_>> {
    fn from(value: UtirInstRef) -> Self {
        let u32_val = u32::from(value);
        if u32_val > u32::from(UtirInstRef::None) {
            return Some(UtirInstIdx::from(
                u32_val - u32::from(UtirInstRef::None) - 1,
            ));
        } else {
            return None;
        }
    }
}

impl ExtraArenaContainable<INST_REF_U32S> for UtirInstRef {}
impl From<[u32; INST_REF_U32S]> for UtirInstRef {
    fn from(value: [u32; INST_REF_U32S]) -> Self {
        return value[0].into();
    }
}

impl From<UtirInstRef> for [u32; INST_REF_U32S] {
    fn from(value: UtirInstRef) -> Self {
        return [value.into()];
    }
}

impl Display for UtirInstRef {
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
                Self::SigType => "@sig_type",
                Self::Undefined => "@undefined",
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
