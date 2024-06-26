mod builder;
mod error;
pub(super) mod inst;

use self::error::Failure;
use crate::{
    auto_indenting_stream::AutoIndentingStream,
    utils::id_arena::{ExtraArenaContainable, Id, IdArena},
    utir::inst::*,
    Ast,
};
use builder::Builder;
use std::fmt::{Display, Write};

// Untyped IR
pub struct Utir {
    // TODO: make this private and force use of get_inst
    instructions: IdArena<UtirInst>,
    extra_data: IdArena<u32>,
}

impl Utir {
    pub fn gen(ast: &Ast) -> Result<Self, Failure> {
        return Builder::new(ast).build();
    }

    pub fn get_inst(&self, inst: UtirInstIdx) -> UtirInst {
        return self.instructions.get(inst);
    }

    // Returns the instruction of decl.name
    pub fn get_decl(&self, decl: UtirInstIdx, name: &str) -> Option<UtirInstRef> {
        let extra_idx = match self.get_inst(decl) {
            UtirInst::StructDecl(payload) => payload.extra_idx,
            UtirInst::ModuleDecl(payload) => payload.extra_idx,
            _ => return None,
        };
        let container_decl = self.get_extra(extra_idx);
        let field_base = extra_idx + CONTAINER_DECL_U32S;
        let decls_base = field_base + (container_decl.fields * CONTAINER_FIELD_U32S as u32);
        for i in 0..container_decl.decls {
            let member_offset = decls_base.to_u32().from_u32() + (i * CONTAINER_FIELD_U32S as u32);
            let member: ContainerMember = self.get_extra(member_offset);
            if member.name.as_str() == name {
                return Some(member.inst_ref);
            }
        }
        return None;
    }

    pub fn get_body(&self, decl: UtirInstIdx) -> Option<&[UtirInstRef]> {
        let extra_idx = match self.get_inst(decl) {
            UtirInst::FunctionDecl(payload) => payload.extra_idx,
            UtirInst::CombDecl(payload) => payload.extra_idx,
            _ => return None,
        };
        let subroutine = self.get_extra(extra_idx);
        let body_ref = subroutine.body;
        return Some(self.get_block_body(body_ref.to_inst().unwrap()));
    }

    pub fn get_block_body(&self, block: UtirInstIdx) -> &[UtirInstRef] {
        let extra_idx = match self.get_inst(block) {
            UtirInst::Block(payload) => payload.extra_idx,
            UtirInst::InlineBlock(payload) => payload.extra_idx,
            _ => unreachable!(),
        };
        let block = self.get_extra(extra_idx);
        let body_start: ExtraIdx<UtirInstRef> = (extra_idx + 1 as u32).to_u32().from_u32();
        return self.slice(body_start, block.num_instrs);
    }

    pub fn get_container_decl_fields(&self, decl: UtirInstIdx) -> &[ContainerField] {
        let extra_idx = match self.get_inst(decl) {
            UtirInst::StructDecl(inner) => inner.extra_idx,
            UtirInst::ModuleDecl(inner) => inner.extra_idx,
            _ => unreachable!(),
        };
        let container_decl = self.get_extra(extra_idx);
        let start: Id<ContainerField> = (extra_idx + CONTAINER_DECL_U32S).to_u32().from_u32();
        return self.slice(start, container_decl.fields);
    }

    pub fn get_container_decl_decls(&self, decl: UtirInstIdx) -> &[ContainerMember] {
        let extra_idx = match self.get_inst(decl) {
            UtirInst::StructDecl(inner) => inner.extra_idx,
            UtirInst::ModuleDecl(inner) => inner.extra_idx,
            _ => unreachable!(),
        };
        let container_decl = self.get_extra(extra_idx);
        let start: Id<ContainerMember> = (extra_idx
            + CONTAINER_DECL_U32S
            + (container_decl.fields * CONTAINER_FIELD_U32S as u32))
            .to_u32()
            .from_u32();
        return self.slice(start, container_decl.decls);
    }

    pub fn get_struct_init_fields(&self, struct_init: UtirInstIdx) -> &[FieldInit] {
        let extra_idx = match self.get_inst(struct_init) {
            UtirInst::StructInit(inner) => inner.extra_idx,
            _ => unreachable!(),
        };
        let container_decl = self.get_extra(extra_idx);
        let start: Id<FieldInit> = (extra_idx + STRUCT_INIT_U32S).to_u32().from_u32();
        return self.slice(start, container_decl.fields);
    }

    pub fn get_subroutine_params(&self, subroutine: UtirInstIdx) -> &[UtirInstRef] {
        let extra_idx = match self.get_inst(subroutine) {
            UtirInst::FunctionDecl(inner) => inner.extra_idx,
            UtirInst::CombDecl(inner) => inner.extra_idx,
            _ => unreachable!(),
        };
        let subroutine_decl = self.get_extra(extra_idx);
        let start: Id<UtirInstRef> = (extra_idx + SUBROUTINE_DECL_U32S).to_u32().from_u32();
        return self.slice(start, subroutine_decl.params);
    }

    pub fn get_subroutine_block(&self, subroutine: UtirInstIdx) -> UtirInstRef {
        let extra_idx: Id<SubroutineDecl> = match self.get_inst(subroutine) {
            UtirInst::FunctionDecl(inner) => inner.extra_idx,
            UtirInst::CombDecl(inner) => inner.extra_idx,
            _ => unreachable!(),
        };
        let subroutine_decl = self.get_extra(extra_idx);
        subroutine_decl.body
    }

    pub fn get_call_args(&self, call: UtirInstIdx) -> &[UtirInstRef] {
        let extra_idx = match self.get_inst(call) {
            UtirInst::Call(inner) => inner.extra_idx,
            _ => unreachable!(),
        };
        let call = self.get_extra(extra_idx);
        let start: Id<UtirInstRef> = (extra_idx + CALL_ARGS_U32S).to_u32().from_u32();
        return self.slice(start, call.num_args);
    }

    pub fn get_extra<const N: usize, T: ExtraArenaContainable<N>>(&self, extra: ExtraIdx<T>) -> T {
        return self.extra_data.get_extra(extra.to_u32());
    }

    pub fn slice<const N: usize, T: ExtraArenaContainable<N>>(
        &self,
        start: ExtraIdx<T>,
        len: u32,
    ) -> &[T] {
        return self.extra_data.slice_u32(start, len);
    }
}

struct UtirWriter<'a, 'b, 'c> {
    utir: &'a Utir,
    stream: AutoIndentingStream<'b, 'c>,
}

impl<'a, 'b, 'c> UtirWriter<'a, 'b, 'c> {
    pub fn new(utir: &'a Utir, stream: AutoIndentingStream<'b, 'c>) -> Self {
        return Self { utir, stream };
    }
    fn indent(&mut self) {
        let _ = write!(self, "indent+");
    }
    fn deindent(&mut self) {
        let _ = write!(self, "indent-");
    }

    pub fn write_root(&mut self) -> std::fmt::Result {
        self.write_container_decl(UtirInstIdx::from(0))?;
        Ok(())
    }

    fn write_container_decl(&mut self, idx: UtirInstIdx) -> std::fmt::Result {
        let (ed_idx, name) = match self.utir.instructions.get(idx) {
            UtirInst::StructDecl(payload) => (payload.extra_idx.to_u32(), "struct_decl"),
            UtirInst::ModuleDecl(payload) => (payload.extra_idx.to_u32(), "module_decl"),
            _ => unreachable!(),
        };
        let container_decl: ContainerDecl = self.utir.extra_data.get_extra(ed_idx);
        write!(self, "%{} = {}({{", u32::from(idx), name)?;

        if container_decl.fields + container_decl.decls > 0 {
            self.indent();
            write!(self, "\n")?;

            for field in self.utir.get_container_decl_fields(idx) {
                self.write_container_field(*field)?;
                write!(self, "\n")?;
            }

            for decl in self.utir.get_container_decl_decls(idx) {
                self.write_container_member(*decl)?;
                write!(self, "\n")?;
            }

            self.deindent();
        }

        write!(self, "}})")?;
        Ok(())
    }

    fn write_container_member(&mut self, member: ContainerMember) -> std::fmt::Result {
        let name = member.name;
        write!(self, "\"{}\" ", name.as_str())?;
        self.write_expr(member.inst_ref)?;
        Ok(())
    }

    fn write_container_field(&mut self, field: ContainerField) -> std::fmt::Result {
        let name = field.name;
        write!(self, "\"{}\": ", name.as_str())?;
        self.write_expr(field.inst_ref)?;
        Ok(())
    }

    fn write_expr(&mut self, inst_ref: UtirInstRef) -> std::fmt::Result {
        if let Some(inst_idx) = inst_ref.into() {
            let inst = self.utir.instructions.get(inst_idx);
            match inst {
                UtirInst::StructDecl(_) | UtirInst::ModuleDecl(_) => {
                    self.write_container_decl(inst_idx)?
                }
                UtirInst::Alloc(_)
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
                | UtirInst::BlockBreak(_)
                | UtirInst::InlineBlockBreak(_)
                | UtirInst::As(_) => self.write_bin_op(inst_idx)?,
                UtirInst::Negate(_) | UtirInst::Deref(_) | UtirInst::Return(_) => {
                    self.write_un_op(inst_idx)?
                }
                UtirInst::InlineBlock(_) | UtirInst::Block(_) => self.write_block(inst_idx)?,
                UtirInst::FunctionDecl(_) => self.write_subroutine_decl(inst_idx)?,
                UtirInst::CombDecl(_) => self.write_subroutine_decl(inst_idx)?,
                UtirInst::RefTy(_) | UtirInst::PtrTy(_) => self.write_ref_ty(inst_idx)?,
                UtirInst::Call(_) => self.write_call(inst_idx)?,
                UtirInst::IntLiteral(_) => self.write_int_literal(inst_idx)?,
                UtirInst::IntType(_) => self.write_int_type(inst_idx)?,
                UtirInst::Branch(_) => self.write_branch(inst_idx)?,
                UtirInst::Param(_) => self.write_param(inst_idx)?,
                UtirInst::Access(_) => self.write_access(inst_idx)?,
                UtirInst::RetImplicitVoid => self.write_ret_implicit_void(inst_idx)?,
                UtirInst::MakeAllocConst(_) => self.write_make_alloc_const(inst_idx)?,
                UtirInst::StructInit(_) => self.write_struct_init(inst_idx)?,
            }
        } else {
            self.write_ref(inst_ref)?;
        }
        Ok(())
    }

    fn write_param(&mut self, idx: UtirInstIdx) -> std::fmt::Result {
        let param = match self.utir.instructions.get(idx) {
            UtirInst::Param(inner) => inner.val,
            _ => unreachable!(),
        };
        // HACK: don't print out things that we've already printed out
        if let Some(param_idx) = param.to_inst() {
            if u32::from(param_idx) < u32::from(idx) {
                self.write_expr(param)?;
                write!(self, "\n")?;
            }
        }

        write!(self, "%{}: {}", u32::from(idx), param)?;
        return Ok(());
    }

    fn write_bin_op(&mut self, idx: UtirInstIdx) -> std::fmt::Result {
        let instr = self.utir.instructions.get(idx);
        let (payload, name) = match instr {
            UtirInst::Alloc(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "alloc",
            ),
            UtirInst::Or(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "or",
            ),
            UtirInst::And(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "and",
            ),
            UtirInst::Lt(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "lt",
            ),
            UtirInst::Gt(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "gt",
            ),
            UtirInst::Lte(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "lte",
            ),
            UtirInst::Gte(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "gte",
            ),
            UtirInst::Eq(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "eq",
            ),
            UtirInst::Neq(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "neq",
            ),
            UtirInst::BitAnd(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "bit_and",
            ),
            UtirInst::BitOr(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "bit_or",
            ),
            UtirInst::BitXor(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "bit_xor",
            ),
            UtirInst::Add(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "add",
            ),
            UtirInst::Sub(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "sub",
            ),
            UtirInst::Mul(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "mul",
            ),
            UtirInst::Div(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "div",
            ),
            UtirInst::As(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "as",
            ),
            UtirInst::InlineBlockBreak(payload) => (payload, "inline_block_break"),
            UtirInst::BlockBreak(payload) => (payload, "block_break"),
            UtirInst::StructDecl(_)
            | UtirInst::ModuleDecl(_)
            | UtirInst::FunctionDecl(_)
            | UtirInst::CombDecl(_)
            | UtirInst::Block(_)
            | UtirInst::MakeAllocConst(_)
            | UtirInst::Param(_)
            | UtirInst::InlineBlock(_)
            | UtirInst::Negate(_)
            | UtirInst::Deref(_)
            | UtirInst::Return(_)
            | UtirInst::RefTy(_)
            | UtirInst::PtrTy(_)
            | UtirInst::Call(_)
            | UtirInst::IntLiteral(_)
            | UtirInst::IntType(_)
            | UtirInst::Branch(_)
            | UtirInst::Access(_)
            | UtirInst::StructInit(_)
            | UtirInst::RetImplicitVoid => unreachable!(),
        };
        write!(
            self,
            "%{} = {}({}, {})",
            u32::from(idx),
            name,
            payload.lhs,
            payload.rhs,
        )?;
        Ok(())
    }

    fn write_un_op(&mut self, idx: UtirInstIdx) -> std::fmt::Result {
        let instr = self.utir.instructions.get(idx);
        let (payload, name) = match instr {
            UtirInst::Negate(payload) => (payload, "negate"),
            UtirInst::Deref(payload) => (payload, "deref"),
            UtirInst::Return(payload) => (payload, "return"),
            UtirInst::StructDecl(_)
            | UtirInst::ModuleDecl(_)
            | UtirInst::FunctionDecl(_)
            | UtirInst::CombDecl(_)
            | UtirInst::Block(_)
            | UtirInst::BlockBreak(_)
            | UtirInst::Alloc(_)
            | UtirInst::MakeAllocConst(_)
            | UtirInst::Param(_)
            | UtirInst::InlineBlock(_)
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
            | UtirInst::RefTy(_)
            | UtirInst::PtrTy(_)
            | UtirInst::Call(_)
            | UtirInst::IntLiteral(_)
            | UtirInst::IntType(_)
            | UtirInst::Branch(_)
            | UtirInst::StructInit(_)
            | UtirInst::RetImplicitVoid => unreachable!(),
        };
        write!(self, "%{} = {}({})", u32::from(idx), name, payload.val)?;
        Ok(())
    }

    fn write_block(&mut self, idx: UtirInstIdx) -> std::fmt::Result {
        let name = match self.utir.instructions.get(idx) {
            UtirInst::Block(_) => "block",
            UtirInst::InlineBlock(_) => "inline_block",
            _ => unreachable!(),
        };
        write!(self, "%{} = {}({{\n", u32::from(idx), name)?;

        {
            self.indent();

            for instr in self.utir.get_block_body(idx) {
                self.write_expr(*instr)?;
                write!(self, "\n")?;
            }

            self.deindent();
        }

        write!(self, "}})")?;
        Ok(())
    }

    fn write_subroutine_decl(&mut self, idx: UtirInstIdx) -> std::fmt::Result {
        let ed_idx = match self.utir.instructions.get(idx) {
            UtirInst::FunctionDecl(payload) | UtirInst::CombDecl(payload) => {
                payload.extra_idx.to_u32()
            }
            _ => unreachable!(),
        };
        let subroutine_decl: SubroutineDecl = self.utir.extra_data.get_extra(ed_idx);

        write!(self, "%{} = subroutine_decl(\n", u32::from(idx))?;
        {
            self.indent();

            write!(self, "{{")?;
            if subroutine_decl.params > 0 {
                self.indent();

                for param in self.utir.get_subroutine_params(idx) {
                    write!(self, "\n")?;
                    self.write_expr(*param)?;
                }
                write!(self, "\n")?;

                self.deindent();
            }
            write!(self, "}}\n")?;

            self.write_expr(subroutine_decl.return_type.into())?;
            write!(self, "\n")?;
            self.write_expr(subroutine_decl.body.into())?;

            self.deindent();
        }
        write!(self, "\n)")?;

        return Ok(());
    }

    fn write_ref_ty(&mut self, idx: UtirInstIdx) -> std::fmt::Result {
        let instr = self.utir.instructions.get(idx);
        let (ref_ty, name): (RefTy, &'static str) = match instr {
            UtirInst::RefTy(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "ref_ty",
            ),
            UtirInst::PtrTy(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "ptr_ty",
            ),
            _ => unreachable!(),
        };
        write!(
            self,
            "%{} = {}({} {})",
            u32::from(idx),
            name,
            ref_ty.mutability,
            ref_ty.ty
        )?;
        return Ok(());
    }

    fn write_call(&mut self, idx: UtirInstIdx) -> std::fmt::Result {
        let ed_idx = match self.utir.instructions.get(idx) {
            UtirInst::Call(payload) => payload.extra_idx,
            _ => unreachable!(),
        };
        let call: CallArgs = self.utir.extra_data.get_extra(ed_idx.to_u32());

        write!(self, "%{} = call({}, {{", u32::from(idx), call.lhs)?;
        if call.num_args > 0 {
            self.indent();
            write!(self, "\n")?;

            for arg in self.utir.get_call_args(idx) {
                write!(self, "{},\n", arg)?;
            }

            self.deindent();
        }
        write!(self, "}})")?;
        return Ok(());
    }

    fn write_int_literal(&mut self, idx: UtirInstIdx) -> std::fmt::Result {
        let number = match self.utir.instructions.get(idx) {
            UtirInst::IntLiteral(num) => num,
            _ => unreachable!(),
        };
        write!(self, "%{} = int_literal({})", u32::from(idx), number)?;
        return Ok(());
    }

    fn write_int_type(&mut self, idx: UtirInstIdx) -> std::fmt::Result {
        let int_type = match self.utir.instructions.get(idx) {
            UtirInst::IntType(num) => num.val,
            _ => unreachable!(),
        };
        write!(
            self,
            "%{} = int_type({}, {})",
            u32::from(idx),
            int_type.signedness,
            int_type.size
        )?;
        return Ok(());
    }

    fn write_branch(&mut self, idx: UtirInstIdx) -> std::fmt::Result {
        let extra_idx = match self.utir.instructions.get(idx) {
            UtirInst::Branch(payload) => payload.extra_idx.to_u32(),
            _ => unreachable!(),
        };
        let branch: Branch = self.utir.extra_data.get_extra(extra_idx);

        write!(self, "%{} = branch({}, {{\n", u32::from(idx), branch.cond)?;
        {
            self.indent();

            self.write_expr(branch.true_block)?;

            self.deindent();
        }
        write!(self, "}}, {{\n")?;
        {
            self.indent();

            self.write_expr(branch.false_block)?;

            self.deindent();
            write!(self, "\n")?;
        }
        write!(self, "}})")?;

        return Ok(());
    }

    fn write_access(&mut self, idx: UtirInstIdx) -> std::fmt::Result {
        let extra_idx = match self.utir.instructions.get(idx) {
            UtirInst::Access(access) => access.extra_idx,
            _ => unreachable!(),
        };
        let access: Access = self.utir.extra_data.get_extra(extra_idx.to_u32());
        write!(
            self,
            "%{} = access({}, \"{}\")",
            u32::from(idx),
            access.lhs,
            access.rhs.as_str()
        )?;
        return Ok(());
    }

    fn write_ret_implicit_void(&mut self, idx: UtirInstIdx) -> std::fmt::Result {
        write!(self, "%{} = ret_implicit_void()", u32::from(idx))?;
        return Ok(());
    }

    fn write_ref(&mut self, inst_ref: UtirInstRef) -> std::fmt::Result {
        write!(self, "{}", inst_ref)?;
        return Ok(());
    }

    fn write_make_alloc_const(&mut self, idx: UtirInstIdx) -> std::fmt::Result {
        let ptr = match self.utir.instructions.get(idx) {
            UtirInst::MakeAllocConst(ptr) => ptr,
            _ => unreachable!(),
        };
        write!(self, "%{} = make_alloc_const({})", u32::from(idx), ptr)?;
        return Ok(());
    }

    fn write_struct_init(&mut self, idx: UtirInstIdx) -> std::fmt::Result {
        let extra_idx = match self.utir.instructions.get(idx) {
            UtirInst::StructInit(inner) => inner.extra_idx,
            _ => unreachable!(),
        };
        let struct_init: StructInit = self.utir.extra_data.get_extra(extra_idx.to_u32());

        write!(self, "%{} = struct_init(", u32::from(idx),)?;

        if struct_init.type_expr == UtirInstRef::None {
            write!(self, "anon, ")?;
        } else {
            write!(self, "{}, ", struct_init.type_expr)?;
        }

        write!(self, "{{")?;

        if struct_init.fields > 0 {
            self.indent();
            write!(self, "\n")?;

            for field in self.utir.get_struct_init_fields(idx) {
                write!(self, ".{} = {}\n", field.name.as_str(), field.expr)?;
            }

            self.deindent();
        }

        write!(self, "}})")?;
        return Ok(());
    }
}

impl Write for UtirWriter<'_, '_, '_> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        write!(self.stream, "{}", s)?;
        Ok(())
    }
}

impl Display for Utir {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stream = AutoIndentingStream::new(f);
        let mut writer = UtirWriter::new(self, stream);
        writer.write_root()?;
        Ok(())
    }
}
