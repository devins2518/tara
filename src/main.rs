#![feature(allocator_api)]
#![allow(dead_code)]

mod ast;
mod ast_codegen;
mod auto_indenting_stream;
mod builtin;
mod circt;
mod comp;
mod module;
mod parser;
mod tables;
mod tir;
mod types;
mod utils;
mod utir;
mod values;

use anyhow::Result;
use ast::Ast;
use comp::Compilation;
use std::pin::Pin;

fn main() -> Result<()> {
    simple_logger::init_with_level(log::Level::Debug).unwrap();

    let args: Vec<_> = std::env::args().collect();
    if args.len() < 2 {
        println!("[ERROR] Expected ./prog <filename>");
        std::process::exit(1);
    }

    let mut unpinned_comp = Compilation::new();
    let mut compilation = Pin::new(&mut unpinned_comp);
    compilation.compile()?;

    return Ok(());
}
