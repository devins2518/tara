mod builder;
mod error;
pub(super) mod inst;

use self::error::Failure;
use crate::{
    ast::Node,
    auto_indenting_stream::AutoIndentingStream,
    utils::arena::{Arena, ExtraArenaContainable, Id},
    utir::inst::*,
    Ast,
};
use builder::Builder;
use std::fmt::{Display, Write};

// Untyped IR
pub struct Utir<'a> {
    pub ast: &'a Ast<'a>,
    // TODO: make this private and force use of get_inst
    instructions: Arena<Inst<'a>>,
    extra_data: Arena<u32>,
    nodes: Arena<&'a Node<'a>>,
}

impl<'a> Utir<'a> {
    pub fn gen(ast: &'a Ast) -> Result<Self, Failure> {
        return Builder::new(ast).build();
    }

    pub fn get_inst(&self, inst: InstIdx<'a>) -> Inst<'a> {
        return self.instructions.get(inst);
    }

    // Returns the instruction of decl.name
    pub fn get_decl(&self, decl: InstIdx<'a>, name: &str) -> Option<InstRef> {
        let extra_idx = match self.get_inst(decl) {
            Inst::StructDecl(payload) => payload.extra_idx,
            Inst::ModuleDecl(payload) => payload.extra_idx,
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

    pub fn get_body(&self, decl: InstIdx<'a>) -> Option<&[InstIdx]> {
        let extra_idx = match self.get_inst(decl) {
            Inst::FunctionDecl(payload) => payload.extra_idx,
            Inst::CombDecl(payload) => payload.extra_idx,
            _ => return None,
        };
        let subroutine = self.get_extra(extra_idx);
        let body_ref = subroutine.body;
        let block_extra_idx = match self.get_inst(body_ref.to_inst().unwrap()) {
            Inst::Block(payload) => payload.extra_idx,
            _ => unreachable!(),
        };
        let block = self.get_extra(block_extra_idx);
        let body_start: ExtraIdx<InstIdx> = (block_extra_idx + 1 as u32).to_u32().from_u32();
        let body_end = body_start + block.num_instrs;
        let body = self.slice(body_start, body_end);
        return Some(body);
    }

    pub fn get_extra<const N: usize, T: ExtraArenaContainable<N>>(&self, extra: ExtraIdx<T>) -> T {
        return self.extra_data.get_extra(extra.to_u32());
    }

    pub fn slice<const N: usize, T: ExtraArenaContainable<N>>(
        &self,
        start: ExtraIdx<T>,
        end: ExtraIdx<T>,
    ) -> &[T] {
        return self.extra_data.slice(start, end);
    }

    pub fn get_node(&self, node: NodeIdx<'a>) -> &'a Node<'a> {
        return self.nodes.get(node);
    }
}

struct UtirWriter<'a, 'b, 'c, 'd> {
    utir: &'a Utir<'b>,
    stream: AutoIndentingStream<'c, 'd>,
}

impl<'a, 'b, 'c, 'd> UtirWriter<'a, 'b, 'c, 'd> {
    pub fn new(utir: &'a Utir<'b>, stream: AutoIndentingStream<'c, 'd>) -> Self {
        return Self { utir, stream };
    }
    fn indent(&mut self) {
        let _ = write!(self, "indent+");
    }
    fn deindent(&mut self) {
        let _ = write!(self, "indent-");
    }

    pub fn write_root(&mut self) -> std::fmt::Result {
        self.write_container_decl(InstIdx::from(0))?;
        Ok(())
    }

    fn write_container_decl(&mut self, idx: InstIdx<'b>) -> std::fmt::Result {
        let (ed_idx, name) = match self.utir.instructions.get(idx) {
            Inst::StructDecl(payload) => (payload.extra_idx.to_u32(), "struct_decl"),
            Inst::ModuleDecl(payload) => (payload.extra_idx.to_u32(), "module_decl"),
            _ => unreachable!(),
        };
        let container_decl: ContainerDecl = self.utir.extra_data.get_extra(ed_idx);
        write!(self, "%{} = {}({{", u32::from(idx), name)?;

        if container_decl.fields + container_decl.decls > 0 {
            self.indent();
            write!(self, "\n")?;

            let field_base = u32::from(ed_idx) + 2;
            for i in 0..container_decl.fields {
                let field_offset = field_base + (i * CONTAINER_FIELD_U32S as u32);
                let field: ContainerField = self.utir.extra_data.get_extra(field_offset.into());
                self.write_container_field(field)?;
                write!(self, "\n")?;
            }

            let decls_base = field_base + (container_decl.fields * CONTAINER_FIELD_U32S as u32);
            for i in 0..container_decl.decls {
                let decl_offset = decls_base + (i * CONTAINER_FIELD_U32S as u32);
                let decl: ContainerMember = self.utir.extra_data.get_extra(decl_offset.into());
                self.write_container_member(decl)?;
                self.stream.write_char('\n')?;
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

    fn write_expr(&mut self, inst_ref: InstRef) -> std::fmt::Result {
        if let Some(inst_idx) = inst_ref.into() {
            let inst = self.utir.instructions.get(inst_idx);
            match inst {
                Inst::StructDecl(_) | Inst::ModuleDecl(_) => self.write_container_decl(inst_idx)?,
                Inst::Or(_)
                | Inst::And(_)
                | Inst::Lt(_)
                | Inst::Gt(_)
                | Inst::Lte(_)
                | Inst::Gte(_)
                | Inst::Eq(_)
                | Inst::Neq(_)
                | Inst::BitAnd(_)
                | Inst::BitOr(_)
                | Inst::BitXor(_)
                | Inst::Add(_)
                | Inst::Sub(_)
                | Inst::Mul(_)
                | Inst::Div(_)
                | Inst::BlockBreak(_)
                | Inst::InlineBlockBreak(_)
                | Inst::As(_) => self.write_bin_op(inst_idx)?,
                Inst::Negate(_) | Inst::Deref(_) | Inst::Return(_) => self.write_un_op(inst_idx)?,
                Inst::InlineBlock(_) | Inst::Block(_) => self.write_block(inst_idx)?,
                Inst::FunctionDecl(_) => self.write_subroutine_decl(inst_idx)?,
                Inst::CombDecl(_) => self.write_subroutine_decl(inst_idx)?,
                Inst::RefTy(_) | Inst::PtrTy(_) => self.write_ref_ty(inst_idx)?,
                Inst::Call(_) => self.write_call(inst_idx)?,
                Inst::IntLiteral(_) => self.write_int_literal(inst_idx)?,
                Inst::IntType(_) => self.write_int_type(inst_idx)?,
                Inst::Branch(_) => self.write_branch(inst_idx)?,
                Inst::Param(_) => self.write_param(inst_idx)?,
                Inst::Access(_) => self.write_access(inst_idx)?,
                Inst::RetImplicitVoid => self.write_ret_implicit_void(inst_idx)?,
            }
        } else {
            self.write_ref(inst_ref)?;
        }
        Ok(())
    }

    fn write_param(&mut self, idx: InstIdx<'b>) -> std::fmt::Result {
        let param = match self.utir.instructions.get(idx) {
            Inst::Param(inner) => inner.val,
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

    fn write_bin_op(&mut self, idx: InstIdx<'b>) -> std::fmt::Result {
        let instr = self.utir.instructions.get(idx);
        let (payload, name) = match instr {
            Inst::Or(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "or",
            ),
            Inst::And(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "and",
            ),
            Inst::Lt(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "lt",
            ),
            Inst::Gt(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "gt",
            ),
            Inst::Lte(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "lte",
            ),
            Inst::Gte(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "gte",
            ),
            Inst::Eq(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "eq",
            ),
            Inst::Neq(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "neq",
            ),
            Inst::BitAnd(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "bit_and",
            ),
            Inst::BitOr(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "bit_or",
            ),
            Inst::BitXor(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "bit_xor",
            ),
            Inst::Add(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "add",
            ),
            Inst::Sub(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "sub",
            ),
            Inst::Mul(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "mul",
            ),
            Inst::Div(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "div",
            ),
            Inst::As(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "as",
            ),
            Inst::InlineBlockBreak(payload) => (payload, "inline_block_break"),
            Inst::BlockBreak(payload) => (payload, "block_break"),
            Inst::StructDecl(_)
            | Inst::ModuleDecl(_)
            | Inst::FunctionDecl(_)
            | Inst::CombDecl(_)
            | Inst::Block(_)
            | Inst::Param(_)
            | Inst::InlineBlock(_)
            | Inst::Negate(_)
            | Inst::Deref(_)
            | Inst::Return(_)
            | Inst::RefTy(_)
            | Inst::PtrTy(_)
            | Inst::Call(_)
            | Inst::IntLiteral(_)
            | Inst::IntType(_)
            | Inst::Branch(_)
            | Inst::Access(_)
            | Inst::RetImplicitVoid => unreachable!(),
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

    fn write_un_op(&mut self, idx: InstIdx<'b>) -> std::fmt::Result {
        let instr = self.utir.instructions.get(idx);
        let (payload, name) = match instr {
            Inst::Negate(payload) => (payload, "negate"),
            Inst::Deref(payload) => (payload, "deref"),
            Inst::Return(payload) => (payload, "return"),
            Inst::StructDecl(_)
            | Inst::ModuleDecl(_)
            | Inst::FunctionDecl(_)
            | Inst::CombDecl(_)
            | Inst::Block(_)
            | Inst::BlockBreak(_)
            | Inst::Param(_)
            | Inst::InlineBlock(_)
            | Inst::InlineBlockBreak(_)
            | Inst::As(_)
            | Inst::Or(_)
            | Inst::And(_)
            | Inst::Lt(_)
            | Inst::Gt(_)
            | Inst::Lte(_)
            | Inst::Gte(_)
            | Inst::Eq(_)
            | Inst::Neq(_)
            | Inst::BitAnd(_)
            | Inst::BitOr(_)
            | Inst::BitXor(_)
            | Inst::Add(_)
            | Inst::Sub(_)
            | Inst::Mul(_)
            | Inst::Div(_)
            | Inst::Access(_)
            | Inst::RefTy(_)
            | Inst::PtrTy(_)
            | Inst::Call(_)
            | Inst::IntLiteral(_)
            | Inst::IntType(_)
            | Inst::Branch(_)
            | Inst::RetImplicitVoid => unreachable!(),
        };
        write!(self, "%{} = {}({})", u32::from(idx), name, payload.val)?;
        Ok(())
    }

    fn write_block(&mut self, idx: InstIdx<'b>) -> std::fmt::Result {
        let (ed_idx, name) = match self.utir.instructions.get(idx) {
            Inst::Block(payload) => (payload.extra_idx.to_u32(), "block"),
            Inst::InlineBlock(payload) => (payload.extra_idx.to_u32(), "inline_block"),
            _ => unreachable!(),
        };
        let block: Block = self.utir.extra_data.get_extra(ed_idx);
        write!(self, "%{} = {}({{\n", u32::from(idx), name)?;

        {
            self.indent();

            for instr in 0..block.num_instrs {
                let inst_idx = u32::from(ed_idx) + instr + 1;
                let instr: InstRef = self.utir.extra_data.get_extra(inst_idx.into());
                self.write_expr(instr)?;
                write!(self, "\n")?;
            }

            self.deindent();
        }

        write!(self, "}})")?;
        Ok(())
    }

    fn write_subroutine_decl(&mut self, idx: InstIdx<'b>) -> std::fmt::Result {
        let ed_idx = match self.utir.instructions.get(idx) {
            Inst::FunctionDecl(payload) | Inst::CombDecl(payload) => payload.extra_idx.to_u32(),
            _ => unreachable!(),
        };
        let subroutine_decl: SubroutineDecl = self.utir.extra_data.get_extra(ed_idx);

        write!(self, "%{} = subroutine_decl(\n", u32::from(idx))?;
        {
            self.indent();

            write!(self, "{{")?;
            {
                self.indent();

                let param_base = u32::from(ed_idx) + SUBROUTINE_DECL_U32S as u32;
                for param_num in 0..subroutine_decl.params {
                    write!(self, "\n")?;

                    let param_offset = param_base + (param_num * INST_REF_U32S as u32);
                    let param_ref: InstRef = self.utir.extra_data.get_extra(param_offset.into());

                    self.write_expr(param_ref)?;
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

    fn write_ref_ty(&mut self, idx: InstIdx<'b>) -> std::fmt::Result {
        let instr = self.utir.instructions.get(idx);
        let (ref_ty, name): (RefTy, &'static str) = match instr {
            Inst::RefTy(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "ref_ty",
            ),
            Inst::PtrTy(payload) => (
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

    fn write_call(&mut self, idx: InstIdx<'b>) -> std::fmt::Result {
        let ed_idx = match self.utir.instructions.get(idx) {
            Inst::Call(payload) => payload.extra_idx,
            _ => unreachable!(),
        };
        let call: CallArgs = self.utir.extra_data.get_extra(ed_idx.to_u32());

        let arg_base = u32::from(ed_idx) + CALL_ARGS_U32S as u32;

        write!(self, "%{} = call({}, {{", u32::from(idx), call.lhs)?;
        if call.num_args > 0 {
            self.indent();
            write!(self, "\n")?;

            for arg_num in 0..call.num_args {
                let arg_ed_idx = arg_base + arg_num;
                let arg_idx: InstRef = self.utir.extra_data.get_extra(Id::from(arg_ed_idx));
                write!(self, "{},\n", arg_idx)?;
            }

            self.deindent();
        }
        write!(self, "}})")?;
        return Ok(());
    }

    fn write_int_literal(&mut self, idx: InstIdx<'b>) -> std::fmt::Result {
        let number = match self.utir.instructions.get(idx) {
            Inst::IntLiteral(num) => num,
            _ => unreachable!(),
        };
        write!(self, "%{} = int_literal({})", u32::from(idx), number)?;
        return Ok(());
    }

    fn write_int_type(&mut self, idx: InstIdx<'b>) -> std::fmt::Result {
        let int_type = match self.utir.instructions.get(idx) {
            Inst::IntType(num) => num.val,
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

    fn write_branch(&mut self, idx: InstIdx<'b>) -> std::fmt::Result {
        let extra_idx = match self.utir.instructions.get(idx) {
            Inst::Branch(payload) => payload.extra_idx.to_u32(),
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

    fn write_access(&mut self, idx: InstIdx<'b>) -> std::fmt::Result {
        let extra_idx = match self.utir.instructions.get(idx) {
            Inst::Access(access) => access.extra_idx,
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

    fn write_ret_implicit_void(&mut self, idx: InstIdx<'b>) -> std::fmt::Result {
        write!(self, "%{} = ret_implicit_void()", u32::from(idx))?;
        return Ok(());
    }

    fn write_ref(&mut self, inst_ref: InstRef) -> std::fmt::Result {
        write!(self, "{}", inst_ref)?;
        return Ok(());
    }
}

impl Write for UtirWriter<'_, '_, '_, '_> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        write!(self.stream, "{}", s)?;
        Ok(())
    }
}

impl Display for Utir<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stream = AutoIndentingStream::new(f);
        let mut writer = UtirWriter::new(self, stream);
        writer.write_root()?;
        Ok(())
    }
}
