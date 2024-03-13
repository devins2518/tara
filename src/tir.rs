mod error;
mod inst;
mod sema;

use crate::{
    module::Module,
    tir::{error::Failure, inst::TirInst, sema::Sema},
    utils::id_arena::{Id, IdArena},
    utir::Utir,
};
use anyhow::Result;

use self::sema::Block;

// Typed IR
pub struct Tir {
    instructions: IdArena<TirInst>,
    extra_data: IdArena<u32>,
}

impl Tir {
    pub fn gen<'comp, 'utir>(
        module: &'comp mut Module<'comp>,
        utir: &'utir Utir<'utir>,
    ) -> Result<Self, Failure> {
        let sema = Sema::new(module, utir);
        let mut top_block = Block::new(&sema);

        // Find Top
        let top = utir
            .get_decl(Id::from(0), "Top")
            .ok_or(Failure::could_not_find_top())?;

        // Find Top.top
        let top_idx = top.to_inst().ok_or(Failure::TopNotModule)?;

        sema.analyze_top(top_idx)?;

        Ok(sema.into())
    }
}
