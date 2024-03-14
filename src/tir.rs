mod error;
mod inst;
mod sema;

use crate::{tir::inst::TirInst, utils::id_arena::IdArena};

// Typed IR
pub struct Tir {
    instructions: IdArena<TirInst>,
    extra_data: IdArena<u32>,
}
