use crate::{
    module::Module,
    tir::{
        error::Failure,
        inst::{Inst, InstIdx},
        Tir,
    },
    utils::arena::Arena,
    utir::{inst::Inst as UtirInst, Utir},
};
use std::collections::HashMap;

pub struct Sema<'comp, 'ast, 'utir, 'module> {
    utir: &'utir Utir<'ast>,
    module: &'comp mut Module<'module>,
    instructions: Arena<Inst>,
    extra_data: Arena<u32>,
    utir_map: HashMap<UtirInst<'utir>, InstIdx>,
}

impl<'comp, 'ast, 'utir, 'module> Sema<'comp, 'ast, 'utir, 'module> {
    pub fn new(module: &'comp mut Module<'module>, utir: &'utir Utir<'ast>) -> Self {
        return Self {
            module,
            utir,
            instructions: Arena::new(),
            extra_data: Arena::new(),
            utir_map: HashMap::new(),
        };
    }

    pub fn analyze_body_inner(&self, block: &mut Block, body: &[UtirInst]) -> Result<(), Failure> {
        println!("got this:");
        for inst in body {
            println!("{:#?}", std::mem::discriminant(inst));
        }
        return Ok(());
    }

    pub fn to_tir(self) -> Tir {
        return Tir {
            instructions: self.instructions,
            extra_data: self.extra_data,
        };
    }
}

pub struct Block<'parent, 'sema, 'comp, 'ast, 'utir, 'module> {
    pub parent: Option<&'parent Block<'parent, 'sema, 'comp, 'ast, 'utir, 'module>>,
    pub sema: &'sema Sema<'comp, 'ast, 'utir, 'module>,
    pub instructions: Vec<InstIdx>,
}
