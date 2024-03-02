mod error;
mod inst;
mod sema;

use crate::{
    module::Module,
    tir::{error::Failure, inst::TirInst, sema::Sema},
    utils::arena::{Arena, Id},
    utir::Utir,
};
use anyhow::Result;

use self::sema::Block;

// Typed IR
pub struct Tir<'module> {
    instructions: Arena<TirInst<'module>>,
    extra_data: Arena<u32>,
}

impl<'module> Tir<'module> {
    pub fn gen<'comp, 'utir>(
        module: &'comp mut Module<'module>,
        utir: &'utir Utir<'utir>,
    ) -> Result<Self, Failure> {
        let sema = Sema::new(module, utir);
        let mut top_block = Block::new(&sema);

        // Find Top
        let top = utir
            .get_decl(Id::from(0), "Top")
            .ok_or(Failure::could_not_find_top())?;

        // Find Top.top
        let top_top = top
            .to_inst()
            .ok_or(Failure::TopNotModule)
            .map(|x| utir.get_decl(x, "top"))?
            .ok_or(Failure::could_not_find_top())?;

        // Get body of Top.top
        let top_body_idxs = top_top
            .to_inst()
            .ok_or(Failure::TopTopNotComb)
            .map(|x| utir.get_body(x))?
            .ok_or(Failure::TopTopNotComb)?;

        sema.analyze_body(&mut top_block, &top_body_idxs)?;

        Ok(sema.into())
    }
}
