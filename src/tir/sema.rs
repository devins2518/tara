use crate::{
    tir::{
        error::Failure,
        inst::{TirInst, TirInstIdx, TirInstRef},
        Tir,
    },
    utils::id_arena::IdArena,
    utir::{inst::UtirInstIdx, Utir},
};
use std::collections::HashMap;

pub struct Sema<'utir> {
    utir: &'utir Utir<'utir>,
    instructions: IdArena<TirInst>,
    extra_data: IdArena<u32>,
    utir_map: HashMap<UtirInstIdx<'utir>, TirInstRef>,
}

type SemaResult = Result<TirInstRef, Failure>;

impl<'utir> Sema<'utir> {
    pub fn new(utir: &'utir Utir<'utir>) -> Self {
        return Self {
            utir,
            instructions: IdArena::new(),
            extra_data: IdArena::new(),
            utir_map: HashMap::new(),
        };
    }
}

impl From<Sema<'_>> for Tir {
    fn from(value: Sema<'_>) -> Self {
        return Self {
            instructions: value.instructions,
            extra_data: value.extra_data,
        };
    }
}

pub struct Block<'parent, 'sema, 'utir> {
    pub parent: Option<&'parent Block<'parent, 'sema, 'utir>>,
    pub sema: &'sema Sema<'utir>,
    pub instructions: IdArena<TirInstIdx>,
}

impl<'parent, 'sema, 'utir> Block<'parent, 'sema, 'utir> {
    pub fn new(sema: &'sema Sema<'utir>) -> Self {
        return Self {
            parent: None,
            sema,
            instructions: IdArena::new(),
        };
    }

    pub fn derive(parent: &'parent Self) -> Self {
        return Self {
            parent: Some(parent),
            sema: parent.sema,
            instructions: IdArena::new(),
        };
    }
}
