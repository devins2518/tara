mod error;
mod tld;

use crate::{
    builtin::Signedness,
    circt::hw::HWModuleOperationBuilder,
    codegen::error::Failure,
    module::{
        comb::Comb,
        decls::{CaptureScope, Decl},
        function::Function,
        tmodule::TModule,
        Module,
    },
    types::Type as TaraType,
    utils::RRC,
    utir::{
        inst::{UtirInst, UtirInstIdx, UtirInstRef},
        Utir,
    },
    values::{TypedValue, Value as TaraValue},
};
use anyhow::Result;
use melior::{
    ir::{
        attribute::StringAttribute, Block as MlirBlock, Location, Module as MlirModule,
        OperationRef, Region,
    },
    Context,
};
use std::collections::HashMap;

pub struct Codegen<'cg> {
    pub module: &'cg Module,
    pub ctx: &'cg Context,
    pub errors: Vec<Failure>,
    pub mlir_module: MlirModule<'cg>,
    pub utir_type_info: HashMap<UtirInstRef, TaraType>,
    pub utir: &'cg Utir,
    pub func: Option<RRC<Function>>,
}

impl<'cg> Codegen<'cg> {
    pub fn new(
        module: &'cg Module,
        ctx: &'cg Context,
        utir: &'cg Utir,
        /* TODO: build_mode */
    ) -> Self {
        let loc = Location::unknown(ctx);
        let mlir_module = MlirModule::new(loc);
        Self {
            module,
            ctx,
            mlir_module,
            errors: Vec::new(),
            utir_type_info: HashMap::new(),
            utir,
            func: None,
        }
    }

    pub fn analyze_struct_decl(&mut self, decl: RRC<Decl>) -> Result<()> {
        let struct_decl = decl.clone();
        let namespace = {
            let decl = struct_decl.borrow();
            decl.src_namespace.clone()
        };
        let inner_decls = self
            .utir
            .get_container_decl_decls(struct_decl.borrow().utir_inst.to_inst().unwrap());
        self.module
            .scan_namespace(self.ctx, namespace, inner_decls, decl)?;
        Ok(())
    }

    pub fn analyze_module_decl(&mut self, module: RRC<Decl>) -> Result<()> {
        let module_decl = module.clone();
        let namespace = {
            let decl = module.borrow();
            decl.src_namespace.clone()
        };
        let utir_idx = module.borrow().utir_inst.to_inst().unwrap();
        let inner_decls = self.utir.get_container_decl_decls(utir_idx);
        self.module
            .scan_namespace(self.ctx, namespace, inner_decls, module)?;
        Ok(())
    }

    // Because top level decls don't necesarily need to be blocks or inline_blocks, the return
    // value of this function may be `decl_ref` in the case where the implicit block's return value
    // is trivially known.
    pub fn analyze_top_level_decl(
        &mut self,
        decl: RRC<Decl>,
        decl_ref: UtirInstRef,
        capture_scope: RRC<CaptureScope>,
    ) -> Result<UtirInstRef> {
        log::debug!("analyzing top decl {}", u32::from(decl_ref));
        match decl_ref {
            UtirInstRef::IntTypeU8
            | UtirInstRef::IntTypeU16
            | UtirInstRef::IntTypeU32
            | UtirInstRef::IntTypeU64
            | UtirInstRef::IntTypeI8
            | UtirInstRef::IntTypeI16
            | UtirInstRef::IntTypeI32
            | UtirInstRef::IntTypeI64
            | UtirInstRef::NumLiteral0
            | UtirInstRef::NumLiteral1
            | UtirInstRef::VoidType
            | UtirInstRef::BoolType
            | UtirInstRef::BoolValTrue
            | UtirInstRef::BoolValFalse
            | UtirInstRef::ClockType
            | UtirInstRef::ResetType
            | UtirInstRef::TypeType
            | UtirInstRef::SigType
            | UtirInstRef::Undefined => Ok(decl_ref),
            _ => {
                let inst_idx = decl_ref.to_inst().unwrap();
                match self.utir.get_inst(inst_idx) {
                    UtirInst::StructDecl(_) => unimplemented!(),
                    UtirInst::ModuleDecl(_) => {
                        self.utir_module_decl(decl, decl_ref, capture_scope)?;
                        Ok(decl_ref)
                    }
                    UtirInst::FunctionDecl(_) => unimplemented!(),
                    UtirInst::CombDecl(_) => {
                        self.utir_comb_decl(decl, decl_ref, capture_scope)?;
                        Ok(decl_ref)
                    }
                    UtirInst::InlineBlock(_) | UtirInst::Block(_) => unimplemented!(),
                    UtirInst::Alloc(_)
                    | UtirInst::MakeAllocConst(_)
                    | UtirInst::Param(_)
                    | UtirInst::BlockBreak(_)
                    | UtirInst::InlineBlockBreak(_)
                    | UtirInst::As(_)
                    | UtirInst::Or(_)
                    | UtirInst::And(_)
                    | UtirInst::Lt(_)
                    | UtirInst::Gt(_)
                    | UtirInst::Lte(_)
                    | UtirInst::Gte(_)
                    | UtirInst::Eq(_)
                    | UtirInst::Neq(_)
                    | UtirInst::BitAnd(_)
                    | UtirInst::BitOr(_)
                    | UtirInst::BitXor(_)
                    | UtirInst::Add(_)
                    | UtirInst::Sub(_)
                    | UtirInst::Mul(_)
                    | UtirInst::Div(_)
                    | UtirInst::Access(_)
                    | UtirInst::Negate(_)
                    | UtirInst::Deref(_)
                    | UtirInst::Return(_)
                    | UtirInst::RefTy(_)
                    | UtirInst::PtrTy(_)
                    | UtirInst::Call(_)
                    | UtirInst::IntLiteral(_)
                    | UtirInst::IntType(_)
                    | UtirInst::Branch(_)
                    | UtirInst::StructInit(_)
                    | UtirInst::RetImplicitVoid => unreachable!(),
                }
            }
        }
    }

    pub fn resolve_ref_value(&self, utir_ref: UtirInstRef) -> TypedValue {
        unimplemented!()
    }

    pub fn resolve_type_layout(&self, ty: TaraType) {
        unimplemented!()
    }

    pub fn analyze_inst(&mut self, block: &mut Block, utir_ref: UtirInstRef) -> Result<()> {
        match utir_ref {
            UtirInstRef::IntTypeU8
            | UtirInstRef::IntTypeU16
            | UtirInstRef::IntTypeU32
            | UtirInstRef::IntTypeU64
            | UtirInstRef::IntTypeI8
            | UtirInstRef::IntTypeI16
            | UtirInstRef::IntTypeI32
            | UtirInstRef::IntTypeI64
            | UtirInstRef::NumLiteral0
            | UtirInstRef::NumLiteral1
            | UtirInstRef::VoidType
            | UtirInstRef::BoolType
            | UtirInstRef::BoolValTrue
            | UtirInstRef::BoolValFalse
            | UtirInstRef::ClockType
            | UtirInstRef::ResetType
            | UtirInstRef::TypeType
            | UtirInstRef::SigType
            | UtirInstRef::Undefined => Ok(()),
            _ => {
                let inst_idx = utir_ref.to_inst().unwrap();
                match self.utir.get_inst(inst_idx) {
                    UtirInst::StructDecl(_) => unimplemented!(),
                    UtirInst::ModuleDecl(_) => unimplemented!(),
                    UtirInst::FunctionDecl(_) => unimplemented!(),
                    UtirInst::CombDecl(_) => unimplemented!(),
                    UtirInst::InlineBlock(_) | UtirInst::Block(_) => unimplemented!(),
                    UtirInst::Alloc(_) => unimplemented!(),
                    UtirInst::MakeAllocConst(_) => unimplemented!(),
                    UtirInst::Param(_) => self.utir_param(block, utir_ref),
                    UtirInst::BlockBreak(_) => unimplemented!(),
                    UtirInst::InlineBlockBreak(_) => unimplemented!(),
                    UtirInst::As(_) => unimplemented!(),
                    UtirInst::Or(_) => unimplemented!(),
                    UtirInst::And(_) => unimplemented!(),
                    UtirInst::Lt(_) => unimplemented!(),
                    UtirInst::Gt(_) => unimplemented!(),
                    UtirInst::Lte(_) => unimplemented!(),
                    UtirInst::Gte(_) => unimplemented!(),
                    UtirInst::Eq(_) => unimplemented!(),
                    UtirInst::Neq(_) => unimplemented!(),
                    UtirInst::BitAnd(_) => unimplemented!(),
                    UtirInst::BitOr(_) => unimplemented!(),
                    UtirInst::BitXor(_) => unimplemented!(),
                    UtirInst::Add(_) => unimplemented!(),
                    UtirInst::Sub(_) => unimplemented!(),
                    UtirInst::Mul(_) => unimplemented!(),
                    UtirInst::Div(_) => unimplemented!(),
                    UtirInst::Access(_) => unimplemented!(),
                    UtirInst::Negate(_) => unimplemented!(),
                    UtirInst::Deref(_) => unimplemented!(),
                    UtirInst::Return(_) => unimplemented!(),
                    UtirInst::RefTy(_) => unimplemented!(),
                    UtirInst::PtrTy(_) => unimplemented!(),
                    UtirInst::Call(_) => unimplemented!(),
                    UtirInst::IntLiteral(_) => unimplemented!(),
                    UtirInst::IntType(_) => self.utir_int_type(block, utir_ref),
                    UtirInst::Branch(_) => unimplemented!(),
                    UtirInst::StructInit(_) => unimplemented!(),
                    UtirInst::RetImplicitVoid => unimplemented!(),
                }
            }
        }
    }

    // Returns instruction that breaks out of this block
    pub fn analyze_block(&self, block: &mut Block, utir_ref: UtirInstRef) -> Result<UtirInstRef> {
        match utir_ref {
            UtirInstRef::IntTypeU8
            | UtirInstRef::IntTypeU16
            | UtirInstRef::IntTypeU32
            | UtirInstRef::IntTypeU64
            | UtirInstRef::IntTypeI8
            | UtirInstRef::IntTypeI16
            | UtirInstRef::IntTypeI32
            | UtirInstRef::IntTypeI64
            | UtirInstRef::NumLiteral0
            | UtirInstRef::NumLiteral1
            | UtirInstRef::VoidType
            | UtirInstRef::BoolType
            | UtirInstRef::BoolValTrue
            | UtirInstRef::BoolValFalse
            | UtirInstRef::ClockType
            | UtirInstRef::ResetType
            | UtirInstRef::TypeType
            | UtirInstRef::SigType
            | UtirInstRef::Undefined => Ok(utir_ref),
            _ => {
                let inst_idx = utir_ref.to_inst().unwrap();
                match self.utir.get_inst(inst_idx) {
                    UtirInst::StructDecl(_) => unimplemented!(),
                    UtirInst::ModuleDecl(_) => unimplemented!(),
                    UtirInst::FunctionDecl(_) => unimplemented!(),
                    UtirInst::CombDecl(_) => unimplemented!(),
                    UtirInst::InlineBlock(_) | UtirInst::Block(_) => unimplemented!(),
                    UtirInst::Alloc(_) => unimplemented!(),
                    UtirInst::MakeAllocConst(_) => unimplemented!(),
                    UtirInst::Param(_) => unimplemented!(),
                    UtirInst::BlockBreak(_) => unimplemented!(),
                    UtirInst::InlineBlockBreak(_) => unimplemented!(),
                    UtirInst::As(_) => unimplemented!(),
                    UtirInst::Or(_) => unimplemented!(),
                    UtirInst::And(_) => unimplemented!(),
                    UtirInst::Lt(_) => unimplemented!(),
                    UtirInst::Gt(_) => unimplemented!(),
                    UtirInst::Lte(_) => unimplemented!(),
                    UtirInst::Gte(_) => unimplemented!(),
                    UtirInst::Eq(_) => unimplemented!(),
                    UtirInst::Neq(_) => unimplemented!(),
                    UtirInst::BitAnd(_) => unimplemented!(),
                    UtirInst::BitOr(_) => unimplemented!(),
                    UtirInst::BitXor(_) => unimplemented!(),
                    UtirInst::Add(_) => unimplemented!(),
                    UtirInst::Sub(_) => unimplemented!(),
                    UtirInst::Mul(_) => unimplemented!(),
                    UtirInst::Div(_) => unimplemented!(),
                    UtirInst::Access(_) => unimplemented!(),
                    UtirInst::Negate(_) => unimplemented!(),
                    UtirInst::Deref(_) => unimplemented!(),
                    UtirInst::Return(_) => unimplemented!(),
                    UtirInst::RefTy(_) => unimplemented!(),
                    UtirInst::PtrTy(_) => unimplemented!(),
                    UtirInst::Call(_) => unimplemented!(),
                    UtirInst::IntLiteral(_) => unimplemented!(),
                    UtirInst::IntType(_) => unimplemented!(),
                    UtirInst::Branch(_) => unimplemented!(),
                    UtirInst::StructInit(_) => unimplemented!(),
                    UtirInst::RetImplicitVoid => unimplemented!(),
                }
            }
        }
    }

    // Analyzes the instruction as a type, creating an entry in `type_info`
    fn analyze_as_type(&mut self, block: &mut Block, utir_ref: UtirInstRef) -> Result<TaraType> {
        match self.utir_type_info.get(&utir_ref) {
            Some(ty) => Ok(ty.clone()),
            None => {
                self.analyze_inst(block, utir_ref)?;
                Ok(self.utir_type_info.get(&utir_ref).unwrap().clone())
            }
        }
    }
}

// Codegen related methods
impl<'cg> Codegen<'cg> {
    pub fn utir_module_decl(
        &mut self,
        decl: RRC<Decl>,
        decl_ref: UtirInstRef,
        capture_scope: RRC<CaptureScope>,
    ) -> Result<OperationRef> {
        {
            let module_obj = RRC::new(TModule::new(
                decl.clone(),
                decl.borrow().src_namespace.clone(),
            ));
            let mut decl = decl.borrow_mut();
        }

        self.analyze_module_decl(decl.clone())?;

        let name = &decl.borrow().name;
        let body = {
            let region = Region::new();
            region
        };
        let loc = Location::unknown(self.ctx);
        let hw_module_builder = HWModuleOperationBuilder::new(self.ctx, loc)
            .sym_name(StringAttribute::new(self.ctx, name))
            .body(body);
        unimplemented!()
    }

    pub fn utir_comb_decl(
        &mut self,
        decl: RRC<Decl>,
        decl_ref: UtirInstRef,
        capture_scope: RRC<CaptureScope>,
    ) -> Result<OperationRef> {
        let decl_idx = decl_ref.to_inst().unwrap();

        let comb_obj = RRC::new(Comb::new(decl.clone(), decl_ref));

        let mut mlir_block = Block::new(self.ctx, decl_ref);

        // Setup params
        let params = self.utir.get_subroutine_params(decl_idx);
        for param in params {
            self.analyze_inst(&mut mlir_block, *param)?;
        }

        // let block_inst = self.utir.get_subroutine_block(decl_idx);
        // let body_slice = self.utir.get_block_body(block_inst);
        // let return_type = self.resolve_block(block, body_slice, block_inst);

        unimplemented!()
    }

    pub fn utir_param(&mut self, block: &mut Block, utir_ref: UtirInstRef) -> Result<()> {
        let param_inst = match self.utir.get_inst(utir_ref.to_inst().unwrap()) {
            UtirInst::Param(inner) => inner.val,
            _ => unreachable!(),
        };
        self.analyze_inst(block, param_inst)?;
        let param_ty = self.analyze_as_type(block, param_inst);
        unimplemented!()
    }

    pub fn utir_int_type(&mut self, block: &mut Block, utir_ref: UtirInstRef) -> Result<()> {
        let inst = self.utir.get_inst(utir_ref.to_inst().unwrap());
        let int_inst = match inst {
            UtirInst::IntType(ty) => ty.val,
            _ => unreachable!(),
        };
        let int_type = match int_inst.signedness {
            Signedness::Unsigned => TaraType::IntUnsigned {
                width: int_inst.size,
            },
            Signedness::Signed => TaraType::IntSigned {
                width: int_inst.size,
            },
        };
        self.utir_type_info.insert(utir_ref, int_type);
        Ok(())
    }
}

pub struct Block<'ctx> {
    mlir_block: MlirBlock<'ctx>,
    // Ref to the instruction which created this block
    inst_ref: UtirInstRef,
}

impl<'ctx> Block<'ctx> {
    pub fn new(ctx: &'ctx Context, inst_ref: UtirInstRef) -> Self {
        Self {
            mlir_block: MlirBlock::new(&[]),
            inst_ref,
        }
    }
}
