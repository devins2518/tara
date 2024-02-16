mod builder;
mod inst;

use crate::{
    arena::{Arena, ExtraArenaContainable, Id},
    ast::Node,
    auto_indenting_stream::AutoIndentingStream,
    utir::inst::*,
    Ast,
};
use builder::Builder;
use std::{
    fmt::{Display, Write},
    num::NonZeroU32,
};
use symbol_table::GlobalSymbol;

pub struct Utir<'a> {
    ast: &'a Ast<'a>,
    instructions: Arena<Inst<'a>>,
    extra_data: Arena<u32>,
    nodes: Arena<&'a Node<'a>>,
}

impl<'a> Utir<'a> {
    pub fn gen(ast: &'a Ast) -> Self {
        return Builder::new(ast).build();
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
        self.write_struct_decl(InstIdx::from(0))?;
        Ok(())
    }

    fn write_struct_decl(&mut self, idx: InstIdx<'b>) -> std::fmt::Result {
        let ed_idx = match self.utir.instructions.get(idx) {
            Inst::StructDecl(payload) => payload.extra_idx.to_u32(),
            _ => unreachable!(),
        };
        let struct_decl: ContainerDecl = self.utir.extra_data.get_extra(ed_idx);
        write!(self, "%{} = struct_decl({{", u32::from(idx))?;

        if struct_decl.fields + struct_decl.decls > 0 {
            self.indent();
            write!(self, "\n")?;

            let field_base = u32::from(ed_idx) + 2;
            for i in 0..struct_decl.fields {}

            let decls_base = field_base + (struct_decl.fields * CONTAINER_FIELD_U32S as u32);
            for i in 0..struct_decl.decls {
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

    fn write_module_decl(&mut self, idx: InstIdx<'b>) -> std::fmt::Result {
        let ed_idx = match self.utir.instructions.get(idx) {
            Inst::ModuleDecl(payload) => payload.extra_idx.to_u32(),
            _ => unreachable!(),
        };
        let module_decl: ContainerDecl = self.utir.extra_data.get_extra(ed_idx);
        write!(self, "%{} = module_decl({{", u32::from(idx))?;

        if module_decl.fields + module_decl.decls > 0 {
            self.indent();
            write!(self, "\n")?;

            let field_base = u32::from(ed_idx) + 2;
            for i in 0..module_decl.fields {}

            let decls_base = field_base + (module_decl.fields * CONTAINER_FIELD_U32S as u32);
            for i in 0..module_decl.decls {
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

    fn write_container_member(&mut self, member: ContainerMember<'b>) -> std::fmt::Result {
        let name = member.name;
        write!(self, "\"{}\" ", name.as_str())?;
        self.write_expr(member.value)?;
        Ok(())
    }

    fn write_expr(&mut self, idx: InstIdx<'b>) -> std::fmt::Result {
        let inst = self.utir.instructions.get(idx);
        match inst {
            Inst::StructDecl(_) => self.write_struct_decl(idx)?,
            Inst::ModuleDecl(_) => self.write_module_decl(idx)?,
            Inst::DeclVal(_) => self.write_decl_val(idx)?,
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
            | Inst::InlineBlockBreak(_)
            | Inst::As(_)
            | Inst::Access(_) => self.write_bin_op(idx)?,
            Inst::Negate(_) | Inst::Deref(_) | Inst::Return(_) => self.write_un_op(idx)?,
            Inst::InlineBlock(_) => self.write_inline_block(idx)?,
            Inst::FunctionDecl(_) => self.write_subroutine_decl(idx)?,
            Inst::CombDecl(_) => self.write_subroutine_decl(idx)?,
            Inst::RefTy(_) | Inst::PtrTy(_) => self.write_ref_ty(idx)?,
            Inst::Call(_) => self.write_call(idx)?,
        }
        Ok(())
    }

    fn write_decl_val(&mut self, idx: InstIdx<'b>) -> std::fmt::Result {
        let decl_val = match self.utir.instructions.get(idx) {
            Inst::DeclVal(decl_val) => decl_val,
            _ => unreachable!(),
        };
        write!(
            self,
            "%{} = decl_val(\"{}\")",
            u32::from(idx),
            decl_val.val.as_str()
        )?;
        Ok(())
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
            Inst::Access(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "access",
            ),
            Inst::As(payload) => (
                self.utir.extra_data.get_extra(payload.extra_idx.to_u32()),
                "as",
            ),
            Inst::InlineBlockBreak(payload) => (payload, "inline_block_break"),
            Inst::StructDecl(_)
            | Inst::ModuleDecl(_)
            | Inst::FunctionDecl(_)
            | Inst::CombDecl(_)
            | Inst::DeclVal(_)
            | Inst::InlineBlock(_)
            | Inst::Negate(_)
            | Inst::Deref(_)
            | Inst::Return(_)
            | Inst::RefTy(_)
            | Inst::PtrTy(_)
            | Inst::Call(_) => unreachable!(),
        };
        write!(
            self,
            "%{} = {}(%{}, %{})",
            u32::from(idx),
            name,
            u32::from(payload.lhs),
            u32::from(payload.rhs),
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
            | Inst::DeclVal(_)
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
            | Inst::Call(_) => unreachable!(),
        };
        write!(
            self,
            "%{} = {}(%{})",
            u32::from(idx),
            name,
            u32::from(payload.val)
        )?;
        Ok(())
    }

    fn write_inline_block(&mut self, idx: InstIdx<'b>) -> std::fmt::Result {
        let ed_idx = match self.utir.instructions.get(idx) {
            Inst::InlineBlock(payload) => payload.extra_idx.to_u32(),
            _ => unreachable!(),
        };
        let block: Block = self.utir.extra_data.get_extra(ed_idx);
        write!(self, "%{} = inline_block({{\n", u32::from(idx))?;

        {
            self.indent();

            for instr in 0..block.num_instrs {
                let instr: InstIdx = self
                    .utir
                    .extra_data
                    .get(ExtraIdx::from(u32::from(ed_idx) + instr + 1))
                    .into();
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

                    let param_offset = param_base + (param_num * PARAM_U32S as u32);
                    let param: Param = self.utir.extra_data.get_extra(param_offset.into());

                    self.write_expr(param.ty)?;
                    write!(
                        self,
                        "\n\"{}\" : %{}",
                        param.name.as_str(),
                        u32::from(param.ty)
                    )?;
                }
                write!(self, "\n")?;

                self.deindent();
            }
            write!(self, "}}\n")?;

            self.write_expr(subroutine_decl.return_type)?;
            write!(self, "\n")?;

            let body_base = (u32::from(ed_idx) + SUBROUTINE_DECL_U32S as u32)
                + (subroutine_decl.params * PARAM_U32S as u32);
            for body_num in 0..subroutine_decl.body_len {
                write!(self, "\n")?;

                let body_offset = body_base + body_num;
                let body_idx = self.utir.extra_data.get(body_offset.into());
                self.write_expr(body_idx.into())?;
                break;
            }
            write!(self, "\n")?;

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
            "%{} = {}({} %{})",
            u32::from(idx),
            name,
            ref_ty.mutability,
            u32::from(ref_ty.ty)
        )?;
        return Ok(());
    }

    fn write_call(&mut self, idx: InstIdx<'b>) -> std::fmt::Result {
        let ed_idx = match self.utir.instructions.get(idx) {
            Inst::Call(payload) => payload.extra_idx,
            _ => unreachable!(),
        };
        let call: CallArgs<'a> = self.utir.extra_data.get_extra(ed_idx.to_u32());
        write!(
            self,
            "%{} = call(%{}, {{\n",
            u32::from(idx),
            u32::from(call.lhs)
        )?;
        {
            self.indent();

            let arg_base = u32::from(ed_idx) + CALL_ARGS_U32S as u32;
            for arg_num in 0..call.num_args {
                let arg_ed_idx = arg_base + arg_num;
                let arg_idx = self.utir.extra_data.get(Id::from(arg_ed_idx));
                write!(self, "%{},\n", arg_idx)?;
            }

            self.deindent();
        }
        write!(self, "}})")?;
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
