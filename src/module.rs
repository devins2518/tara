pub mod decls;
pub mod file;
pub mod function;
pub mod namespace;
pub mod structs;
pub mod variable;

use crate::{
    ast::Ast,
    codegen::package::Package,
    comp::Compilation,
    module::{
        decls::{CaptureScope, Decl},
        file::File,
        namespace::Namespace,
        structs::Struct,
    },
    types::Type,
    utir::{inst::UtirInstRef, Utir},
    values::Value,
};
use anyhow::Result;
use kioku::Arena;
use std::{
    collections::{hash_map::Entry, HashMap},
    mem::MaybeUninit,
};

pub struct Module<'comp, 'arena> {
    comp: &'comp Compilation,
    arena: &'arena Arena,
    // Keys are fully resolved paths
    pub import_table: HashMap<&'arena str, File<'arena>>,
}

impl<'comp, 'arena> Module<'comp, 'arena> {
    pub fn new(comp: &'comp Compilation, arena: &'arena Arena) -> Self {
        Self {
            comp,
            arena,
            import_table: HashMap::new(),
        }
    }

    pub fn analyze_pkg(
        &mut self,
        pkg: &'arena Package<'arena>,
        exit_early: bool,
        compile_ast: bool,
        compile_utir: bool,
    ) -> Result<()> {
        let mut file = self.import_pkg(pkg)?.file;
        self.analyze_file(&mut file, exit_early, compile_ast, compile_utir)?;
        self.import_table.insert(pkg.src_dir, file);
        Ok(())
    }

    fn analyze_file(
        &mut self,
        file: &mut File<'arena>,
        exit_early: bool,
        compile_ast: bool,
        compile_utir: bool,
    ) -> Result<()> {
        let contents = {
            use std::io::prelude::*;
            let mut fp = std::fs::File::open(file.path)?;
            let mut vec = Vec::new_in(self.arena);
            let len = fp.metadata()?.len() as usize;
            vec.reserve(len);
            unsafe {
                vec.set_len(len);
            }
            fp.read_exact(vec.as_mut_slice())?;
            // Leaking here is fine since the arena will clean it up later
            let slice = vec.leak();
            // Tara files don't necessarily need to be UTF-8 encoded
            unsafe { std::str::from_utf8_unchecked(slice) }
        };
        file.add_source(contents);

        if exit_early && !compile_ast && !compile_utir {
            return Ok(());
        }

        let source = file.source();
        let ast = match Ast::parse(source) {
            Ok(ast) => self.arena.alloc_no_copy(ast),
            Err(_) => {
                file.fail_ast();
                return Ok(());
            }
        };
        file.add_ast(ast);
        if compile_ast {
            println!("{}", file.ast());
        }

        if exit_early && !compile_utir {
            return Ok(());
        }

        let utir = match Utir::gen(ast) {
            Ok(utir) => self.arena.alloc_no_copy(utir),
            Err(fail) => {
                file.fail_utir();
                fail.report(&file)?;
                return Ok(());
            }
        };
        file.add_utir(utir);
        if compile_utir {
            println!("{}", file.utir());
        }

        self.sema_file(file);

        Ok(())
    }

    fn sema_file(&self, file: &File<'arena>) {
        if file.root_decl.is_some() {
            return;
        }

        let struct_obj: &mut Struct = self.alloc_uninit();
        let struct_ty = Type::Struct(struct_obj);
        let struct_val = Value::Type(struct_ty);
        let ty_ty = Value::TypeType;
        let namespace = Namespace::new(*file, struct_ty);
        // struct_obj.fields = Vec::new();
        // struct_obj.namespace = namespace;
        /*
        *struct_obj = Struct {
            owner_decl: todo!(),
            fields: Vec::new(),
            namespace,
            utir_ref: todo!(),
            status: todo!(),
        };
        */
    }

    fn allocate_decl(
        &self,
        name: &'arena str,
        namespace: &'arena Namespace<'arena>,
        src_scope: Option<&'arena CaptureScope<'arena>>,
    ) -> &'arena Decl {
        self.arena.alloc_no_copy(Decl {
            name,
            ty: None,
            value: None,
            src_namespace: namespace,
            src_scope,
            utir_inst: UtirInstRef::from(UtirInstRef::None as u32 + 1),
            public: false,
            export: false,
        })
    }

    fn alloc_uninit<T>(&self) -> &'arena mut T {
        self.arena
            .alloc_no_copy(unsafe { MaybeUninit::uninit().assume_init() })
    }

    // If changing the returned file, remember to update it by reinserting it
    fn import_pkg(&mut self, pkg: &'arena Package) -> Result<ImportResult<'arena>> {
        let resolved_path = std::fs::canonicalize(pkg.src_dir)?.join(pkg.src_path);
        let resolved_str = resolved_path.to_str().unwrap();
        match self.import_table.get(resolved_str) {
            Some(file) => Ok(ImportResult {
                file: *file,
                new: false,
            }),
            None => {
                let arena_str = self.arena.copy_str(resolved_str);
                let file = File::new(arena_str);
                self.import_table.insert(arena_str, file);
                Ok(ImportResult {
                    file: *self.import_table.get(arena_str).unwrap(),
                    new: true,
                })
            }
        }
    }
}

struct ImportResult<'arena> {
    file: File<'arena>,
    new: bool,
}
