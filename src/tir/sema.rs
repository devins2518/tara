use crate::{
    tir::{
        error::Failure,
        inst::{Inst, InstIdx},
        Tir,
    },
    utils::arena::Arena,
    utir::{inst::Inst as UtirInst, Utir},
};
use std::collections::HashMap;

pub struct Sema<'utir, 'ast> {
    utir: &'utir Utir<'ast>,
    instructions: Arena<Inst>,
    extra_data: Arena<u32>,
    utir_map: HashMap<UtirInst<'utir>, InstIdx>,
}

impl Unpin for Sema<'_, '_> {}

impl<'utir, 'ast> Sema<'utir, 'ast> {
    pub fn new(utir: &'utir Utir<'ast>) -> Self {
        return Self {
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

pub struct Block<'sema, 'ast, 'parent> {
    pub parent: Option<&'parent Block<'sema, 'ast, 'parent>>,
    pub sema: &'sema Sema<'sema, 'ast>,
    pub instructions: Vec<InstIdx>,
}
