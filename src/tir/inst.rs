use crate::{
    tir::sema::Sema,
    utils::id_arena::{ExtraArenaContainable, Id},
};
use std::fmt::Display;

#[derive(Copy, Clone)]
pub enum TirInst {
    // Basic arithmetic and logical operators
    Or(BinOp),
    And(BinOp),
    Lt(BinOp),
    Gt(BinOp),
    Lte(BinOp),
    Gte(BinOp),
    Eq(BinOp),
    Neq(BinOp),
    BitAnd(BinOp),
    BitOr(BinOp),
    BitXor(BinOp),
    Add(BinOp),
    Sub(BinOp),
    Mul(BinOp),
    Div(BinOp),
    // Allocates space for a type and returns the pointer to it.
    // Alloc(TypeId<'a>),
}

#[derive(Copy, Clone)]
pub struct UnOp {
    lhs: TirInstRef,
    rhs: TirInstRef,
}

#[derive(Copy, Clone)]
pub struct BinOp {
    lhs: TirInstRef,
    rhs: TirInstRef,
}

impl TirInst {
    pub fn is_no_return(&self) -> bool {
        match self {
            Self::Or(_)
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
            | Self::Div(_) => false,
        }
    }
}

pub type TirInstIdx = Id<TirInst>;

// Refs include well known and well typed commonly used values
pub const INST_REF_U32S: usize = 1;
#[repr(u32)]
#[non_exhaustive]
#[derive(Copy, Clone)]
pub enum TirInstRef {
    // Used to indicate end of known values
    None = 0,
    AlwaysNoReturn = std::u32::MAX,
}

impl TirInstRef {
    pub fn to_inst(&self) -> Option<TirInstIdx> {
        return (*self).into();
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            _ => None,
        }
    }

    /*
    pub fn is_no_return<'sema>(&self, sema: &'sema Sema<'_, '_>) -> bool {
        if let Some(idx) = self.to_inst() {
            return sema.get_instruction(idx).is_no_return();
        } else {
            match self {
                _ => unreachable!(),
            }
        }
    }
    */
}

impl From<TirInstRef> for u32 {
    fn from(value: TirInstRef) -> Self {
        return unsafe { std::mem::transmute(value) };
    }
}

impl From<u32> for TirInstRef {
    fn from(value: u32) -> Self {
        return unsafe { std::mem::transmute(value) };
    }
}

impl From<TirInstIdx> for TirInstRef {
    fn from(value: TirInstIdx) -> Self {
        return unsafe { std::mem::transmute(u32::from(value) + u32::from(Self::None) + 1) };
    }
}

impl From<TirInstRef> for Option<TirInstIdx> {
    fn from(value: TirInstRef) -> Self {
        let u32_val = u32::from(value);
        if u32_val > u32::from(TirInstRef::None) {
            return Some(TirInstIdx::from(u32_val - u32::from(TirInstRef::None) - 1));
        } else {
            return None;
        }
    }
}

impl ExtraArenaContainable<INST_REF_U32S> for TirInstRef {}
impl From<[u32; INST_REF_U32S]> for TirInstRef {
    fn from(value: [u32; INST_REF_U32S]) -> Self {
        return value[0].into();
    }
}

impl From<TirInstRef> for [u32; INST_REF_U32S] {
    fn from(value: TirInstRef) -> Self {
        return [value.into()];
    }
}

impl Display for TirInstRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(inst_idx) = self.to_inst() {
            f.write_fmt(format_args!("%{}", u32::from(inst_idx)))?;
        } else {
            let s = match self {
                _ => unreachable!(),
            };
            f.write_str(s)?;
        }
        return Ok(());
    }
}
