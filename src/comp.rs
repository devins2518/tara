use crate::{
    ast::Ast,
    codegen::Codegen,
    module::{file::File, Module},
    tir::Tir,
    utils::slice::OwnedString,
    utir::Utir,
};
use anyhow::Result;
use kioku::Arena;
use std::io::prelude::*;
use std::mem::MaybeUninit;
use symbol_table::GlobalSymbol;

pub struct Compilation {
    arena: Arena,
}

impl Compilation {
    pub fn new() -> Self {
        return Self {
            arena: Arena::new(),
        };
    }

    pub fn compile(&mut self) -> Result<()> {
        let options = CompilationOptions::from_args();
        let mut file = File::new(options.top_file.as_str());
        let contents = {
            let mut fp = std::fs::File::open(options.top_file.as_str())?;
            let mut vec = Vec::new_in(&self.arena);
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

        let mut module = Module::new(self);
        module.compile_file(
            &mut file,
            options.exit_early,
            options.dump_ast,
            options.dump_utir,
        )?;

        if options.dump_ast {
            println!("{}", file.ast());
            if options.exit_early {
                return Ok(());
            }
        }

        if options.dump_utir {
            println!("{}", file.utir());
            if options.exit_early {
                return Ok(());
            }
        }

        /*
        let mut codegen = Codegen::new(utir);
        // match codegen.gen(&utir) {
        //     Ok(_) => {}
        //     Err(fail) => return fail.report(&ast),
        // }
        codegen.dump_module();
        */

        return Ok(());
    }

    pub fn alloc<'comp, T>(&'comp self, val: T) -> &'comp mut T {
        let memory = self.arena.alloc_raw(std::alloc::Layout::new::<T>()) as *mut MaybeUninit<T>;
        let memory_ref = unsafe { memory.as_mut().unwrap() };
        memory_ref.write(val)
    }
}

struct CompilationOptions {
    top_file: GlobalSymbol,
    exit_early: bool,
    dump_ast: bool,
    dump_utir: bool,
    dump_tir: bool,
}

impl CompilationOptions {
    pub fn from_args() -> CompilationOptions {
        let mut top_file = MaybeUninit::uninit();
        let mut found_top = false;
        let mut exit_early = false;
        let mut dump_ast = false;
        let mut dump_utir = false;
        let mut dump_tir = false;
        for arg in std::env::args().skip(1) {
            match arg.as_str() {
                "--exit-early" => exit_early = true,
                "--dump-ast" => dump_ast = true,
                "--dump-utir" => dump_utir = true,
                "--dump-tir" => dump_tir = true,
                _ => {
                    if found_top {
                        println!("[ERROR] Found multiple top files in arguments");
                        std::process::exit(1);
                    } else {
                        top_file.write(GlobalSymbol::from(arg));
                        found_top = true;
                    }
                }
            }
        }

        if !found_top {
            println!("[ERROR] Could not find top file in arguments");
            std::process::exit(1);
        }

        return CompilationOptions {
            top_file: unsafe { top_file.assume_init() },
            exit_early,
            dump_ast,
            dump_utir,
            dump_tir,
        };
    }
}
