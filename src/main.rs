mod ast;
mod parser;

use anyhow::Result;
use ast::Ast;
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 2 {
        println!("[ERROR] Expected ./prog <filename>");
        std::process::exit(1);
    }

    let file_path = PathBuf::from(&args[1]);
    let contents = std::fs::read_to_string(&file_path)?;

    let ast = Ast::parse(&contents)?;

    println!("{}", ast.root.fields.len());
    println!("{}", ast.root.members.len());

    return Ok(());
}
