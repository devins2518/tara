use crate::utils::slice::OwnedString;
use crate::{builtin::Mutability, parser::TaraParser};
use anyhow::Result;
use codespan::{FileId, Span};
use codespan_reporting::files::SimpleFile;
use num_bigint::BigUint;
use std::{fmt::Display, marker::PhantomData};
use symbol_table::GlobalSymbol;

pub struct Node<'a> {
    pub span: Span,
    pub kind: NodeKind<'a>,
}

impl<'a> Node<'a> {
    pub fn new(kind: NodeKind<'a>, span: pest::Span) -> Self {
        return Self {
            span: Span::new(span.start() as u32, span.end() as u32),
            kind,
        };
    }
}

impl Display for Node<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.kind)?;
        return Ok(());
    }
}

pub enum NodeKind<'a> {
    StructDecl(StructInner<'a>),
    VarDecl(VarDecl<'a>),
    ModuleDecl(ModuleInner<'a>),
    LocalVarDecl(VarDecl<'a>),
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
    ReferenceTy(RefTy<'a>),
    PointerTy(RefTy<'a>),
    NumberLiteral(BigUint),
    SizedNumberLiteral(SizedNumberLiteral),
    IfExpr(IfExpr<'a>),
    SubroutineDecl(SubroutineDecl<'a>),
    StructInit(StructInit<'a>),
}

impl Display for NodeKind<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeKind::StructDecl(struct_inner) => {
                f.write_fmt(format_args!("struct_decl({})", struct_inner))?
            }
            NodeKind::VarDecl(var_decl) => f.write_fmt(format_args!("var_decl({})", var_decl))?,
            NodeKind::ModuleDecl(mod_inner) => {
                f.write_fmt(format_args!("module_decl({})", mod_inner))?
            }
            NodeKind::LocalVarDecl(var_decl) => f.write_fmt(format_args!("local({})", var_decl))?,
            NodeKind::Or(expr) => f.write_fmt(format_args!("or({})", expr))?,
            NodeKind::And(expr) => f.write_fmt(format_args!("and({})", expr))?,
            NodeKind::Lt(expr) => f.write_fmt(format_args!("lt({})", expr))?,
            NodeKind::Gt(expr) => f.write_fmt(format_args!("gt({})", expr))?,
            NodeKind::Lte(expr) => f.write_fmt(format_args!("lte({})", expr))?,
            NodeKind::Gte(expr) => f.write_fmt(format_args!("gte({})", expr))?,
            NodeKind::Eq(expr) => f.write_fmt(format_args!("eq({})", expr))?,
            NodeKind::Neq(expr) => f.write_fmt(format_args!("neq({})", expr))?,
            NodeKind::BitAnd(expr) => f.write_fmt(format_args!("bit_and({})", expr))?,
            NodeKind::BitOr(expr) => f.write_fmt(format_args!("bit_or({})", expr))?,
            NodeKind::BitXor(expr) => f.write_fmt(format_args!("bit_xor({})", expr))?,
            NodeKind::Add(expr) => f.write_fmt(format_args!("add({})", expr))?,
            NodeKind::Sub(expr) => f.write_fmt(format_args!("sub({})", expr))?,
            NodeKind::Mul(expr) => f.write_fmt(format_args!("mul({})", expr))?,
            NodeKind::Div(expr) => f.write_fmt(format_args!("div({})", expr))?,
            NodeKind::Access(expr) => f.write_fmt(format_args!("access({})", expr))?,
            NodeKind::Call(expr) => f.write_fmt(format_args!("call({})", expr))?,
            NodeKind::Identifier(ident) => {
                f.write_fmt(format_args!("identifier(\"{}\")", ident.as_str()))?
            }
            NodeKind::Negate(expr) => f.write_fmt(format_args!("neg({})", expr))?,
            NodeKind::Deref(expr) => f.write_fmt(format_args!("deref({})", expr))?,
            NodeKind::Return(expr) => f.write_fmt(format_args!("return({})", expr))?,
            NodeKind::ReferenceTy(expr) => f.write_fmt(format_args!("reference_ty({})", expr))?,
            NodeKind::PointerTy(expr) => f.write_fmt(format_args!("pointer_ty({})", expr))?,
            NodeKind::NumberLiteral(num) => f.write_fmt(format_args!("number({})", num))?,
            NodeKind::SizedNumberLiteral(num) => {
                f.write_fmt(format_args!("sized_number({})", num))?
            }
            NodeKind::IfExpr(expr) => f.write_fmt(format_args!("if_expr({})", expr))?,
            NodeKind::SubroutineDecl(subroutine) => {
                f.write_fmt(format_args!("subroutine_decl({})", subroutine))?
            }
            NodeKind::StructInit(expr) => f.write_fmt(format_args!("struct_init({})", expr))?,
        }
        Ok(())
    }
}

pub struct StructInner<'a> {
    pub fields: Vec<TypedName<'a>>,
    pub members: Vec<Node<'a>>,
}

impl<'a> StructInner<'a> {
    pub fn new(fields: Vec<TypedName<'a>>, members: Vec<Node<'a>>) -> Self {
        return Self { fields, members };
    }
}

impl Display for StructInner<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("struct_inner(")?;
        for field in &self.fields {
            f.write_fmt(format_args!("({}, {}), ", field.name, *field.ty))?;
        }
        for member in &self.members {
            f.write_fmt(format_args!("({}), ", member))?;
        }
        f.write_str(")")?;
        Ok(())
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

impl Display for ModuleInner<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("module_inner(")?;
        for field in &self.fields {
            f.write_fmt(format_args!("({}, {}), ", field.name, *field.ty))?;
        }
        for member in &self.members {
            f.write_fmt(format_args!("({}), ", member))?;
        }
        f.write_str(")")?;
        Ok(())
    }
}

pub enum Publicity {
    Public,
    Private,
}

impl Display for Publicity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Publicity::Public => "pub",
            Publicity::Private => "priv",
        };
        f.write_str(s)?;
        Ok(())
    }
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

impl Display for VarDecl<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.ty {
            Some(node) => f.write_fmt(format_args!(
                "{} \"{}\": {} = {}",
                self.publicity,
                self.ident.as_str(),
                *node,
                *self.expr
            ))?,
            None => f.write_fmt(format_args!(
                "{} \"{}\" = {}",
                self.publicity,
                self.ident.as_str(),
                *self.expr
            ))?,
        }
        Ok(())
    }
}

pub struct TypedName<'a> {
    pub ty: Box<Node<'a>>,
    pub name: GlobalSymbol,
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

impl Display for BinOp<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{}, {}", *self.lhs, *self.rhs))?;
        Ok(())
    }
}

pub struct UnOp<'a> {
    pub lhs: Box<Node<'a>>,
}

impl Display for UnOp<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{}", *self.lhs))?;
        Ok(())
    }
}

pub struct Call<'a> {
    pub call: Box<Node<'a>>,
    pub args: Vec<Node<'a>>,
}

impl Display for Call<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{}, (", *self.call))?;
        for arg in &self.args {
            f.write_fmt(format_args!("{}", arg))?;
        }
        f.write_str(")")?;
        Ok(())
    }
}

pub struct SizedNumberLiteral {
    pub size: u16,
    pub literal: BigUint,
}

impl Display for SizedNumberLiteral {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{}'d{}", self.size, self.literal))?;
        Ok(())
    }
}

pub struct RefTy<'a> {
    pub mutability: Mutability,
    pub ty: Box<Node<'a>>,
}

impl<'a> RefTy<'a> {
    pub fn new(mutability: Mutability, ty: Node<'a>) -> Self {
        return Self {
            mutability,
            ty: Box::new(ty),
        };
    }
}

impl Display for RefTy<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{} {}", self.mutability, *self.ty))?;
        Ok(())
    }
}

pub struct IfExpr<'a> {
    pub cond: Box<Node<'a>>,
    pub body: Box<Node<'a>>,
    pub else_body: Box<Node<'a>>,
}

impl Display for IfExpr<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "{}, {}, {}",
            *self.cond, *self.body, *self.else_body
        ))?;
        Ok(())
    }
}

pub struct SubroutineDecl<'a> {
    pub publicity: Publicity,
    pub ident: GlobalSymbol,
    pub params: Vec<TypedName<'a>>,
    pub return_type: Box<Node<'a>>,
    pub block: Vec<Node<'a>>,
}

impl<'a> SubroutineDecl<'a> {
    pub fn new(
        publicity: Publicity,
        ident: GlobalSymbol,
        params: Vec<TypedName<'a>>,
        return_type: Node<'a>,
        block: Vec<Node<'a>>,
    ) -> Self {
        return Self {
            publicity,
            ident,
            params,
            return_type: Box::new(return_type),
            block,
        };
    }
}

impl Display for SubroutineDecl<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{} {}(", self.publicity, self.ident))?;
        for param in &self.params {
            f.write_fmt(format_args!("({}, {}), ", param.name, *param.ty))?;
        }
        f.write_fmt(format_args!(") {} (", *self.return_type))?;
        for statement in &self.block {
            f.write_fmt(format_args!("({}), ", statement))?;
        }
        f.write_str(")")?;
        Ok(())
    }
}

pub struct StructInit<'a> {
    pub ty: Box<Node<'a>>,
    // Technically not a typed name, but a named value but whatever
    pub fields: Vec<TypedName<'a>>,
}

impl<'a> StructInit<'a> {
    pub fn new(ty: Node<'a>, fields: Vec<TypedName<'a>>) -> Self {
        return Self {
            ty: Box::new(ty),
            fields,
        };
    }
}

impl Display for StructInit<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{}, (", *self.ty))?;
        for field in &self.fields {
            f.write_fmt(format_args!(".{} = {}, ", field.name, *field.ty))?;
        }
        f.write_str(")")?;
        Ok(())
    }
}

pub struct Ast<'a> {
    pub source: &'a SimpleFile<GlobalSymbol, OwnedString>,
    pub root: Node<'a>,
}

impl<'a> Ast<'a> {
    pub fn parse(source: &'a SimpleFile<GlobalSymbol, OwnedString>) -> Result<Ast<'a>> {
        let root = TaraParser::parse_source(source.source())?;

        return Ok(Self { source, root });
    }
}

impl Display for Ast<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{}", self.root))
    }
}
