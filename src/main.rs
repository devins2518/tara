#[deny(unused_variables)]
mod arena;
mod ast;
mod auto_indenting_stream;
mod parser;
mod utir;

use anyhow::Result;
use ast::Ast;
use std::path::PathBuf;
use utir::Utir;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 2 {
        println!("[ERROR] Expected ./prog <filename>");
        std::process::exit(1);
    }

    let file_path = PathBuf::from(&args[1]);
    let contents = std::fs::read_to_string(&file_path)?;

    let ast = Ast::parse(&contents)?;

    if &args[2] == "--dump-ast" {
        println!("{}", ast);
        return Ok(());
    }

    let utir = Utir::gen(&ast);

    if &args[2] == "--dump-utir" {
        println!("{}", utir);
    }

    return Ok(());
}
