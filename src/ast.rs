use crate::{builtin::Mutability, parser::TaraParser};
use anyhow::Result;
use codespan::Span;
use num_bigint::BigUint;
use std::fmt::Display;
use symbol_table::GlobalSymbol;

pub struct Node {
    pub span: Span,
    pub kind: NodeKind,
}

impl Node {
    pub fn new(kind: NodeKind, span: pest::Span) -> Self {
        return Self {
            span: Span::new(span.start() as u32, span.end() as u32),
            kind,
        };
    }
}

impl Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.kind)?;
        return Ok(());
    }
}

pub enum NodeKind {
    StructDecl(StructInner),
    VarDecl(VarDecl),
    ModuleDecl(ModuleInner),
    LocalVarDecl(VarDecl),
    Or(BinOp),
    And(BinOp),
    Lt(BinOp),
    Gt(BinOp),
    Lte(BinOp),
    Gte(BinOp),
    Eq(BinOp),
    Neq(BinOp),
    BitAnd(BinOp),
    BitOr(BinOp),
    BitXor(BinOp),
    Add(BinOp),
    Sub(BinOp),
    Mul(BinOp),
    Div(BinOp),
    Access(BinOp),
    Call(Call),
    Negate(UnOp),
    Deref(UnOp),
    Return(UnOp),
    Identifier(GlobalSymbol),
    ReferenceTy(RefTy),
    PointerTy(RefTy),
    NumberLiteral(BigUint),
    SizedNumberLiteral(SizedNumberLiteral),
    IfExpr(IfExpr),
    SubroutineDecl(SubroutineDecl),
    StructInit(StructInit),
}

impl Display for NodeKind {
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

pub struct StructInner {
    pub fields: Vec<TypedName>,
    pub members: Vec<Node>,
}

impl StructInner {
    pub fn new(fields: Vec<TypedName>, members: Vec<Node>) -> Self {
        return Self { fields, members };
    }
}

impl Display for StructInner {
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

pub struct ModuleInner {
    pub fields: Vec<TypedName>,
    pub members: Vec<Node>,
}

impl ModuleInner {
    pub fn new(fields: Vec<TypedName>, members: Vec<Node>) -> Self {
        return Self { fields, members };
    }
}

impl Display for ModuleInner {
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

pub struct VarDecl {
    pub publicity: Publicity,
    pub ident: GlobalSymbol,
    pub ty: Option<Box<Node>>,
    pub expr: Box<Node>,
}

impl VarDecl {
    pub fn new(publicity: Publicity, ident: GlobalSymbol, ty: Option<Node>, expr: Node) -> Self {
        return Self {
            publicity,
            ident,
            ty: ty.map(Box::new),
            expr: Box::new(expr),
        };
    }
}

impl Display for VarDecl {
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

pub struct TypedName {
    pub ty: Box<Node>,
    pub name: GlobalSymbol,
}

impl TypedName {
    pub fn new(ty: Node, name: GlobalSymbol) -> Self {
        return Self {
            ty: Box::new(ty),
            name,
        };
    }
}

pub struct BinOp {
    pub lhs: Box<Node>,
    pub rhs: Box<Node>,
}

impl Display for BinOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{}, {}", *self.lhs, *self.rhs))?;
        Ok(())
    }
}

pub struct UnOp {
    pub lhs: Box<Node>,
}

impl Display for UnOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{}", *self.lhs))?;
        Ok(())
    }
}

pub struct Call {
    pub call: Box<Node>,
    pub args: Vec<Node>,
}

impl Display for Call {
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

pub struct RefTy {
    pub mutability: Mutability,
    pub ty: Box<Node>,
}

impl RefTy {
    pub fn new(mutability: Mutability, ty: Node) -> Self {
        return Self {
            mutability,
            ty: Box::new(ty),
        };
    }
}

impl Display for RefTy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{} {}", self.mutability, *self.ty))?;
        Ok(())
    }
}

pub struct IfExpr {
    pub cond: Box<Node>,
    pub body: Box<Node>,
    pub else_body: Box<Node>,
}

impl Display for IfExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "{}, {}, {}",
            *self.cond, *self.body, *self.else_body
        ))?;
        Ok(())
    }
}

pub struct SubroutineDecl {
    pub publicity: Publicity,
    pub ident: GlobalSymbol,
    pub params: Vec<TypedName>,
    pub return_type: Box<Node>,
    pub block: Vec<Node>,
}

impl SubroutineDecl {
    pub fn new(
        publicity: Publicity,
        ident: GlobalSymbol,
        params: Vec<TypedName>,
        return_type: Node,
        block: Vec<Node>,
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

impl Display for SubroutineDecl {
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

pub struct StructInit {
    pub ty: Option<Box<Node>>,
    // Technically not a typed name, but a named value but whatever
    pub fields: Vec<TypedName>,
}

impl StructInit {
    pub fn new_with_ty(ty: Node, fields: Vec<TypedName>) -> Self {
        return Self {
            ty: Some(Box::new(ty)),
            fields,
        };
    }

    pub fn new_anon(fields: Vec<TypedName>) -> Self {
        return Self { ty: None, fields };
    }
}

impl Display for StructInit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(expr) = &self.ty {
            f.write_fmt(format_args!("{}, (", expr))?;
        } else {
            f.write_str("anon, (")?;
        }
        for field in &self.fields {
            f.write_fmt(format_args!(".{} = {}, ", field.name, *field.ty))?;
        }
        f.write_str(")")?;
        Ok(())
    }
}

pub struct Ast {
    pub root: Node,
}

impl Ast {
    pub fn parse(source: &str) -> Result<Ast> {
        let root = TaraParser::parse_source(source)?;

        return Ok(Self { root });
    }
}

impl Display for Ast {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{}", self.root))
    }
}
