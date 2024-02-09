use crate::parser::TaraParser;
use anyhow::Result;
use num_bigint::BigUint;
use std::{fmt::Display, marker::PhantomData};
use symbol_table::GlobalSymbol;

pub enum Node<'a> {
    StructDecl(StructInner<'a>),
    VarDecl(VarDecl<'a>),
    ModuleDecl(ModuleInner<'a>),
    Or(BinOp<'a>),
    And(BinOp<'a>),
    Lt(BinOp<'a>),
    Gt(BinOp<'a>),
    Lte(BinOp<'a>),
    Gte(BinOp<'a>),
    Eq(BinOp<'a>),
    Neq(BinOp<'a>),
    BitAnd(BinOp<'a>),
    BitOr(BinOp<'a>),
    BitXor(BinOp<'a>),
    Add(BinOp<'a>),
    Sub(BinOp<'a>),
    Mul(BinOp<'a>),
    Div(BinOp<'a>),
    Access(BinOp<'a>),
    Call(Call<'a>),
    Negate(UnOp<'a>),
    Deref(UnOp<'a>),
    Return(UnOp<'a>),
    Identifier(GlobalSymbol),
    ReferenceTy(UnOp<'a>),
    PointerTy(UnOp<'a>),
    NumberLiteral(BigUint),
    SizedNumberLiteral(SizedNumberLiteral),
    IfExpr(IfExpr<'a>),
}

pub struct StructInner<'a> {
    pub fields: Vec<TypedName<'a>>,
    pub members: Vec<Node<'a>>,
    _phantom: PhantomData<&'a Node<'a>>,
}

impl<'a> StructInner<'a> {
    pub fn new(fields: Vec<TypedName<'a>>, members: Vec<Node<'a>>) -> Self {
        return Self {
            fields,
            members,
            _phantom: PhantomData,
        };
    }
}

pub struct ModuleInner<'a> {
    pub fields: Vec<TypedName<'a>>,
    pub members: Vec<Node<'a>>,
    _phantom: PhantomData<&'a Node<'a>>,
}

impl<'a> ModuleInner<'a> {
    pub fn new(fields: Vec<TypedName<'a>>, members: Vec<Node<'a>>) -> Self {
        return Self {
            fields,
            members,
            _phantom: PhantomData,
        };
    }
}

pub enum Publicity {
    Public,
    Private,
}

pub struct VarDecl<'a> {
    pub publicity: Publicity,
    pub ident: GlobalSymbol,
    pub ty: Option<Box<Node<'a>>>,
    pub expr: Box<Node<'a>>,
}

impl<'a> VarDecl<'a> {
    pub fn new(
        publicity: Publicity,
        ident: GlobalSymbol,
        ty: Option<Node<'a>>,
        expr: Node<'a>,
    ) -> Self {
        return Self {
            publicity,
            ident,
            ty: ty.map(Box::new),
            expr: Box::new(expr),
        };
    }
}

pub struct TypedName<'a> {
    ty: Box<Node<'a>>,
    name: GlobalSymbol,
}

impl<'a> TypedName<'a> {
    pub fn new(ty: Node<'a>, name: GlobalSymbol) -> Self {
        return Self {
            ty: Box::new(ty),
            name,
        };
    }
}

pub struct BinOp<'a> {
    pub lhs: Box<Node<'a>>,
    pub rhs: Box<Node<'a>>,
}

pub struct UnOp<'a> {
    pub lhs: Box<Node<'a>>,
}

pub struct Call<'a> {
    pub call: Box<Node<'a>>,
    pub args: Vec<Node<'a>>,
}

pub struct SizedNumberLiteral {
    pub size: u16,
    pub literal: BigUint,
}

pub struct IfExpr<'a> {
    pub cond: Box<Node<'a>>,
    pub body: Box<Node<'a>>,
    pub else_body: Box<Node<'a>>,
}

pub struct Ast<'a> {
    source: &'a str,
    pub root: StructInner<'a>,
}

impl<'a> Ast<'a> {
    pub fn parse(source: &'a str) -> Result<Ast<'a>> {
        let root = TaraParser::parse_source(&source)?;

        return Ok(Self { source, root });
    }
}

impl Display for Ast<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return Ok(());
    }
}
