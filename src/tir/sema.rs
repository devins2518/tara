use crate::{
    module::Module,
    tir::{
        error::Failure,
        inst::{TirInst, TirInstIdx, TirInstRef},
        Tir,
    },
    utils::arena::Arena,
    utir::{
        inst::{UtirInst, UtirInstIdx, UtirInstRef},
        Utir,
    },
};
use std::collections::HashMap;

pub struct Sema<'comp, 'utir, 'module, 'tir> {
    utir: &'utir Utir<'utir>,
    module: &'comp mut Module<'module>,
    instructions: Arena<TirInst<'tir>>,
    extra_data: Arena<u32>,
    utir_map: HashMap<UtirInstIdx<'utir>, TirInstRef>,
}

type SemaResult = Result<TirInstRef, Failure>;

impl<'comp, 'utir, 'module, 'tir> Sema<'comp, 'utir, 'module, 'tir> {
    pub fn new(module: &'comp mut Module<'module>, utir: &'utir Utir<'utir>) -> Self {
        return Self {
            module,
            utir,
            instructions: Arena::new(),
            extra_data: Arena::new(),
            utir_map: HashMap::new(),
        };
    }

    pub fn analyze_body(&self, block: &mut Block, body: &[UtirInstRef]) -> Result<(), Failure> {
        let mut i = 0;
        let result = loop {
            let inst = body[i].to_inst().unwrap();
            let tir_inst: TirInstRef = match self.utir.get_inst(inst) {
                UtirInst::StructDecl(_) => self.utir_struct_decl(block, inst)?,
                UtirInst::ModuleDecl(_) => self.utir_module_decl(block, inst)?,
                UtirInst::FunctionDecl(_) => self.utir_func_decl(block, inst)?,
                UtirInst::CombDecl(_) => self.utir_comb_decl(block, inst)?,
                UtirInst::Alloc(_) => self.utir_alloc(block, inst)?,
                UtirInst::MakeAllocConst(_) => self.utir_make_alloc_const(block, inst)?,
                UtirInst::Param(_) => self.utir_param(block, inst)?,
                UtirInst::Block(_) => unimplemented!(),
                UtirInst::BlockBreak(_) => unimplemented!(),
                UtirInst::InlineBlock(_) => unimplemented!(),
                UtirInst::InlineBlockBreak(_) => unimplemented!(),
                UtirInst::As(_) => self.utir_as(block, inst)?,
                UtirInst::Or(_) => self.utir_or(block, inst)?,
                UtirInst::And(_) => self.utir_and(block, inst)?,
                UtirInst::Lt(_) => self.utir_lt(block, inst)?,
                UtirInst::Gt(_) => self.utir_gt(block, inst)?,
                UtirInst::Lte(_) => self.utir_lte(block, inst)?,
                UtirInst::Gte(_) => self.utir_gte(block, inst)?,
                UtirInst::Eq(_) => self.utir_eq(block, inst)?,
                UtirInst::Neq(_) => self.utir_neq(block, inst)?,
                UtirInst::BitAnd(_) => self.utir_bit_and(block, inst)?,
                UtirInst::BitOr(_) => self.utir_bit_or(block, inst)?,
                UtirInst::BitXor(_) => self.utir_bit_xor(block, inst)?,
                UtirInst::Add(_) => self.utir_add(block, inst)?,
                UtirInst::Sub(_) => self.utir_sub(block, inst)?,
                UtirInst::Mul(_) => self.utir_mul(block, inst)?,
                UtirInst::Div(_) => self.utir_div(block, inst)?,
                UtirInst::Access(_) => self.utir_access(block, inst)?,
                UtirInst::Negate(_) => self.utir_negate(block, inst)?,
                UtirInst::Deref(_) => self.utir_deref(block, inst)?,
                UtirInst::Return(_) => unimplemented!(),
                UtirInst::RefTy(_) => self.utir_ref_ty(block, inst)?,
                UtirInst::PtrTy(_) => self.utir_ptr_ty(block, inst)?,
                UtirInst::Call(_) => self.utir_call(block, inst)?,
                UtirInst::IntLiteral(_) => self.utir_int_literal(block, inst)?,
                UtirInst::IntType(_) => self.utir_int_type(block, inst)?,
                UtirInst::Branch(_) => unimplemented!(),
                UtirInst::StructInit(_) => self.utir_struct_init(block, inst)?,
                UtirInst::RetImplicitVoid => unimplemented!(),
            };
            if tir_inst.is_no_return(self) {
                assert!(block.instructions.len() > 0);
                assert!(TirInstRef::from(
                    block
                        .instructions
                        .get((block.instructions.len() as u32 - 1).into())
                )
                .is_no_return(self));
                break TirInstRef::AlwaysNoReturn;
            }
            // self.utir_map.insert(inst, tir_inst);
            i += 1;
        };

        return Ok(());
    }

    pub fn add_instruction(&self, inst: TirInst<'tir>) -> TirInstIdx<'tir> {
        return self.instructions.alloc(inst).into();
    }

    pub fn get_instruction(&self, idx: TirInstIdx<'tir>) -> TirInst<'tir> {
        return self.instructions.get(idx);
    }

    pub fn utir_struct_decl(&self, _block: &mut Block, _inst: UtirInstIdx) -> SemaResult {
        unimplemented!()
    }

    pub fn utir_module_decl(&self, _block: &mut Block, _inst: UtirInstIdx) -> SemaResult {
        unimplemented!()
    }

    pub fn utir_func_decl(&self, _block: &mut Block, _inst: UtirInstIdx) -> SemaResult {
        unimplemented!()
    }

    pub fn utir_comb_decl(&self, _block: &mut Block, _inst: UtirInstIdx) -> SemaResult {
        unimplemented!()
    }

    pub fn utir_alloc(&self, _block: &mut Block, _inst: UtirInstIdx) -> SemaResult {
        unimplemented!()
    }

    pub fn utir_make_alloc_const(&self, _block: &mut Block, _inst: UtirInstIdx) -> SemaResult {
        unimplemented!()
    }

    pub fn utir_param(&self, _block: &mut Block, _inst: UtirInstIdx) -> SemaResult {
        unimplemented!()
    }

    pub fn utir_as(&self, _block: &mut Block, _inst: UtirInstIdx) -> SemaResult {
        unimplemented!()
    }

    pub fn utir_or(&self, _block: &mut Block, _inst: UtirInstIdx) -> SemaResult {
        unimplemented!()
    }

    pub fn utir_and(&self, _block: &mut Block, _inst: UtirInstIdx) -> SemaResult {
        unimplemented!()
    }

    pub fn utir_lt(&self, _block: &mut Block, _inst: UtirInstIdx) -> SemaResult {
        unimplemented!()
    }

    pub fn utir_gt(&self, _block: &mut Block, _inst: UtirInstIdx) -> SemaResult {
        unimplemented!()
    }

    pub fn utir_lte(&self, _block: &mut Block, _inst: UtirInstIdx) -> SemaResult {
        unimplemented!()
    }

    pub fn utir_gte(&self, _block: &mut Block, _inst: UtirInstIdx) -> SemaResult {
        unimplemented!()
    }

    pub fn utir_eq(&self, _block: &mut Block, _inst: UtirInstIdx) -> SemaResult {
        unimplemented!()
    }

    pub fn utir_neq(&self, _block: &mut Block, _inst: UtirInstIdx) -> SemaResult {
        unimplemented!()
    }

    pub fn utir_bit_and(&self, _block: &mut Block, _inst: UtirInstIdx) -> SemaResult {
        unimplemented!()
    }

    pub fn utir_bit_or(&self, _block: &mut Block, _inst: UtirInstIdx) -> SemaResult {
        unimplemented!()
    }

    pub fn utir_bit_xor(&self, _block: &mut Block, _inst: UtirInstIdx) -> SemaResult {
        unimplemented!()
    }

    pub fn utir_add(&self, _block: &mut Block, _inst: UtirInstIdx) -> SemaResult {
        unimplemented!()
    }

    pub fn utir_sub(&self, _block: &mut Block, _inst: UtirInstIdx) -> SemaResult {
        unimplemented!()
    }

    pub fn utir_mul(&self, _block: &mut Block, _inst: UtirInstIdx) -> SemaResult {
        unimplemented!()
    }

    pub fn utir_div(&self, _block: &mut Block, _inst: UtirInstIdx) -> SemaResult {
        unimplemented!()
    }

    pub fn utir_access(&self, _block: &mut Block, _inst: UtirInstIdx) -> SemaResult {
        unimplemented!()
    }

    pub fn utir_negate(&self, _block: &mut Block, _inst: UtirInstIdx) -> SemaResult {
        unimplemented!()
    }

    pub fn utir_deref(&self, _block: &mut Block, _inst: UtirInstIdx) -> SemaResult {
        unimplemented!()
    }

    pub fn utir_ref_ty(&self, _block: &mut Block, _inst: UtirInstIdx) -> SemaResult {
        unimplemented!()
    }

    pub fn utir_ptr_ty(&self, _block: &mut Block, _inst: UtirInstIdx) -> SemaResult {
        unimplemented!()
    }

    pub fn utir_call(&self, _block: &mut Block, _inst: UtirInstIdx) -> SemaResult {
        unimplemented!()
    }

    pub fn utir_int_literal(&self, _block: &mut Block, _inst: UtirInstIdx) -> SemaResult {
        unimplemented!()
    }

    pub fn utir_int_type(&self, _block: &mut Block, _inst: UtirInstIdx) -> SemaResult {
        unimplemented!()
    }

    pub fn utir_struct_init(&self, _block: &mut Block, _inst: UtirInstIdx) -> SemaResult {
        unimplemented!()
    }
}

impl<'tir> From<Sema<'_, '_, '_, 'tir>> for Tir<'tir> {
    fn from(value: Sema<'_, '_, '_, 'tir>) -> Self {
        return Self {
            instructions: value.instructions,
            extra_data: value.extra_data,
        };
    }
}

pub struct Block<'parent, 'sema, 'comp, 'utir, 'module, 'tir> {
    pub parent: Option<&'parent Block<'parent, 'sema, 'comp, 'utir, 'module, 'tir>>,
    pub sema: &'sema Sema<'comp, 'utir, 'module, 'tir>,
    pub instructions: Arena<TirInstIdx<'tir>>,
}

impl<'parent, 'sema, 'comp, 'utir, 'module, 'tir>
    Block<'parent, 'sema, 'comp, 'utir, 'module, 'tir>
{
    pub fn new(sema: &'sema Sema<'comp, 'utir, 'module, 'tir>) -> Self {
        return Self {
            parent: None,
            sema,
            instructions: Arena::new(),
        };
    }

    pub fn derive(parent: &'parent Self) -> Self {
        return Self {
            parent: Some(parent),
            sema: parent.sema,
            instructions: Arena::new(),
        };
    }

    pub fn add_instruction(&self, inst: TirInst<'tir>) -> TirInstRef {
        let idx = self.sema.add_instruction(inst);
        self.instructions.alloc(idx);
        return idx.into();
    }
}
