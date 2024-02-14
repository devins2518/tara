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
            _ => unreachable!()
        }
        Ok(())
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
