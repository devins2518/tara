use crate::{
    ast::Ast,
    types::Type,
    utils::slice::{OwnedSlice, OwnedString},
    utir::Utir,
};
use anyhow::Result;
use codespan_reporting::files::SimpleFiles;
use internment::Arena;
use symbol_table::GlobalSymbol;

pub struct Compilation<'a> {
    files: SimpleFiles<GlobalSymbol, OwnedString>,
    types: Arena<Type<'a>>,
}

impl<'a> Compilation<'a> {
    pub fn new() -> Self {
        return Self {
            files: SimpleFiles::new(),
            types: Arena::new(),
        };
    }

    pub fn compile_all(&mut self) -> Result<()> {
        let options = CompilationOptions::from_args();
        for file in options.files.into_iter() {
            let contents = std::fs::read_to_string(file.as_str())?;
            let file_id = self.files.add(*file, OwnedString::from(contents));

            let ast = Ast::parse(self.files.get(file_id)?)?;
            if options.dump_ast {
                println!("{}", ast);
            }

            let utir = match Utir::gen(&ast) {
                Ok(utir) => utir,
                Err(fail) => return fail.report(&ast),
            };
            if options.dump_utir {
                println!("{}", utir);
            }
        }
        return Ok(());
    }
}

struct CompilationOptions {
    files: OwnedSlice<GlobalSymbol>,
    dump_ast: bool,
    dump_utir: bool,
}

impl CompilationOptions {
    pub fn from_args() -> CompilationOptions {
        let mut files = Vec::new();
        let mut dump_ast = false;
        let mut dump_utir = false;
        for arg in std::env::args().skip(1) {
            match arg.as_str() {
                "--dump-ast" => dump_ast = true,
                "--dump-utir" => dump_utir = true,
                _ => files.push(GlobalSymbol::new(arg)),
            }
        }
        return CompilationOptions {
            files: OwnedSlice::from(files),
            dump_ast,
            dump_utir,
        };
    }
}
