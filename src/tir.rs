mod error;
mod inst;
mod sema;

use crate::{
    tir::{error::Failure, inst::Inst, sema::Sema},
    utils::arena::{Arena, Id},
    utir::{
        inst::{
            ContainerDecl, ContainerMember, Inst as UtirInst, InstIdx, SubroutineDecl,
            CONTAINER_DECL_U32S, CONTAINER_FIELD_U32S, SUBROUTINE_DECL_U32S,
        },
        Utir,
    },
};
use anyhow::Result;

use self::sema::Block;

// Typed IR
pub struct Tir {
    instructions: Arena<Inst>,
    extra_data: Arena<u32>,
}

impl Tir {
    pub fn gen(utir: &Utir) -> Result<Tir, Failure> {
        let sema = Sema::new(utir);
        let mut top_block = Block {
            parent: None,
            sema: &sema,
            instructions: Vec::new(),
        };

        // Find Top
        let top = 'top: {
            let root = utir.get_inst(Id::from(0));
            let extra_idx = match root {
                UtirInst::StructDecl(payload) => payload.extra_idx,
                _ => unreachable!(),
            };
            let top_decl = utir.get_extra(extra_idx);
            let field_base = extra_idx + CONTAINER_DECL_U32S;
            let decls_base = field_base + (top_decl.fields * CONTAINER_FIELD_U32S as u32);
            for i in 0..top_decl.decls {
                let decl_offset = decls_base + (i * CONTAINER_FIELD_U32S as u32);
                let decl: ContainerMember = utir.get_extra(decl_offset.to_u32().from_u32());
                if decl.name.as_str() == "Top" {
                    break 'top decl.inst_ref;
                }
            }
            return Err(Failure::could_not_find_top());
        };

        // Find Top.top
        let top_top = 'top_top: {
            if let Some(idx) = top.to_inst() {
                let payload = match utir.get_inst(idx) {
                    UtirInst::ModuleDecl(payload) => payload,
                    _ => return Err(Failure::TopNotModule),
                };

                let extra_idx = payload.extra_idx;

                let top_decl = utir.get_extra(extra_idx);
                let field_base = extra_idx + CONTAINER_DECL_U32S;
                let decls_base = field_base + (top_decl.fields * CONTAINER_FIELD_U32S as u32);
                for i in 0..top_decl.decls {
                    let decl_offset = decls_base + (i * CONTAINER_FIELD_U32S as u32);
                    let decl: ContainerMember = utir.get_extra(decl_offset.to_u32().from_u32());
                    if decl.name.as_str() == "top" {
                        break 'top_top decl.inst_ref;
                    }
                }
                return Err(Failure::could_not_find_top_top(
                    utir.get_node(payload.node_idx),
                ));
            } else {
                return Err(Failure::TopNotModule);
            };
        };

        let top_body_idxs = 'top_body: {
            let extra_idx = if let Some(idx) = top_top.to_inst() {
                match utir.get_inst(idx) {
                    UtirInst::CombDecl(payload) => payload.extra_idx,
                    _ => unreachable!(),
                }
            } else {
                return Err(Failure::TopTopNotComb);
            };
            let subroutine: SubroutineDecl = utir.get_extra(extra_idx);
            let body_start: Id<InstIdx> =
                extra_idx.to_u32().from_u32() + SUBROUTINE_DECL_U32S + subroutine.params;
            let body_end: Id<InstIdx> = body_start + subroutine.body_len;
            let body = utir.slice(body_start, body_end);
            break 'top_body body;
        };

        let mut top_body = Vec::new();
        for idx in top_body_idxs {
            top_body.push(utir.get_inst(*idx));
        }

        sema.analyze_body_inner(&mut top_block, &top_body)?;

        Ok(sema.to_tir())
    }
}
