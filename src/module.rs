pub mod decls;
pub mod file;
pub mod function;
pub mod namespace;
pub mod structs;
pub mod variable;

use crate::{ast::Ast, comp::Compilation, module::file::File, utir::Utir};
use anyhow::Result;
use kioku::Arena;

pub struct Module<'comp> {
    comp: &'comp Compilation,
    arena: Arena,
}

impl<'comp> Module<'comp> {
    pub fn new(comp: &'comp Compilation) -> Self {
        Self {
            comp,
            arena: Arena::new(),
        }
    }

    pub fn compile_file<'file>(
        &mut self,
        file: &'file mut File<'comp>,
        exit_early: bool,
        compile_ast: bool,
        compile_utir: bool,
    ) -> Result<()>
    where
        'comp: 'file,
    {
        if exit_early && !compile_ast && !compile_utir {
            return Ok(());
        }

        let source = file.source();
        let ast = match Ast::parse(source) {
            Ok(ast) => self.comp.alloc(ast),
            Err(_) => {
                file.fail_ast();
                return Ok(());
            }
        };
        file.add_ast(ast);

        if exit_early && !compile_utir {
            return Ok(());
        }

        let utir = match Utir::gen(ast) {
            Ok(utir) => self.comp.alloc(utir),
            Err(fail) => {
                file.fail_utir();
                fail.report(file)?;
                return Ok(());
            }
        };
        file.add_utir(utir);
        Ok(())
    }

    pub fn analyze_file(&self, file: &File) {
        if let Some(root_decl) = file.root_decl {}
    }
}
