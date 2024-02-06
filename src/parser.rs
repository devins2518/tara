use crate::ast::{BinOp, Node, Publicity, StructInner, TypedName, VarDecl};
use anyhow::Result as AResult;
use pest::pratt_parser::{Assoc, Op, PrattParser};
use pest_consume::{match_nodes, Error, Parser};
use symbol_table::GlobalSymbol;

type ParseResult<T> = std::result::Result<T, Error<Rule>>;
type ParseNode<'i> = pest_consume::Node<'i, Rule, ()>;

// Rebuild with grammar changes
const _GRAMMAR_FILE: &str = include_str!("syntax.pest");

#[derive(Debug)]
pub enum ParseError {
    ReadFailure(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("Error occurred during parsing")?;
        Ok(())
    }
}

impl ParseError {
    pub fn read_failure(reason: String) -> ParseError {
        return ParseError::ReadFailure(reason);
    }
}

impl std::error::Error for ParseError {}

lazy_static::lazy_static! {
    static ref PRATT: PrattParser<Rule> = PrattParser::new()
        .op(Op::infix(Rule::or_operator, Assoc::Left))
        .op(Op::infix(Rule::and_operator, Assoc::Left))
        .op(Op::infix(Rule::lte_operator, Assoc::Left) | Op::infix(Rule::lt_operator, Assoc::Left) |
            Op::infix(Rule::gte_operator, Assoc::Left) | Op::infix(Rule::gt_operator, Assoc::Left) |
            Op::infix(Rule::eq_operator, Assoc::Left)| Op::infix(Rule::neq_operator, Assoc::Left))
        .op(Op::infix(Rule::bitwise_and_operator, Assoc::Left) |
            Op::infix(Rule::bitwise_or_operator, Assoc::Left)  |
            Op::infix(Rule::bitwise_xor_operator, Assoc::Left))
        .op(Op::infix(Rule::add_operator, Assoc::Left) | Op::infix(Rule::sub_operator, Assoc::Left))
        .op(Op::infix(Rule::mul_operator, Assoc::Left) | Op::infix(Rule::div_operator, Assoc::Left))
        .op(Op::infix(Rule::infix_operator, Assoc::Left))
        .op(Op::prefix(Rule::neg_operator));
}

#[derive(Parser)]
#[grammar = "syntax.pest"]
pub struct TaraParser;

impl TaraParser {
    pub fn parse_source<'a>(source: &'a str) -> AResult<StructInner<'a>> {
        let inputs = TaraParser::parse(Rule::root, source)?;

        let root_node = inputs.single()?;

        let root = TaraParser::root(root_node)?;

        return Ok(root);
    }

    fn expr_helper(pairs: pest::iterators::Pairs<Rule>) -> ParseResult<Node> {
        PRATT
            .map_primary(|primary| TaraParser::primary_expr(ParseNode::new(primary)))
            .map_infix(|lhs, op, rhs| {
                let lhs_box = Box::new(lhs?);
                let rhs_box = Box::new(rhs?);
                let bin_op = BinOp {
                    lhs: lhs_box,
                    rhs: rhs_box,
                };
                let node = match op.into_inner().next().unwrap().as_rule() {
                    Rule::or_operator => Node::Or(bin_op),
                    Rule::and_operator => Node::And(bin_op),
                    Rule::lt_operator => Node::Lt(bin_op),
                    Rule::lte_operator => Node::Lte(bin_op),
                    Rule::gt_operator => Node::Gt(bin_op),
                    Rule::gte_operator => Node::Gte(bin_op),
                    Rule::eq_operator => Node::Eq(bin_op),
                    Rule::neq_operator => Node::Neq(bin_op),
                    Rule::bitwise_and_operator => Node::BitAnd(bin_op),
                    Rule::bitwise_or_operator => Node::BitOr(bin_op),
                    Rule::bitwise_xor_operator => Node::BitXor(bin_op),
                    Rule::add_operator => Node::Add(bin_op),
                    Rule::sub_operator => Node::Sub(bin_op),
                    Rule::mul_operator => Node::Mul(bin_op),
                    Rule::div_operator => Node::Div(bin_op),
                    _ => unreachable!(),
                };

                Ok(node)
            })
            .parse(pairs)
    }
}

#[pest_consume::parser]
impl TaraParser {
    fn EOI(_input: ParseNode) -> ParseResult<()> {
        return Ok(());
    }

    fn root(input: ParseNode) -> ParseResult<StructInner> {
        match_nodes!(
            input.clone().into_children();
            [struct_inner(struct_inner), EOI(_)] => {
                return Ok(struct_inner);
            },
        );
    }

    fn struct_inner(input: ParseNode) -> ParseResult<StructInner> {
        let mut fields = Vec::new();

        let mut members = Vec::new();

        let node = match_nodes!(
            input.into_children();
            [container_decls(container_decls_pre), container_fields(container_fields), container_decls(container_decls_post)] => {
                members.extend(container_decls_pre);
                members.extend(container_decls_post);
                fields.extend(container_fields);
            },
            [container_decl(container_decl)..] => {
                members.extend(container_decl);
            }
        );

        return Ok(StructInner::new(fields, members));
    }

    fn container_fields(input: ParseNode) -> ParseResult<Vec<TypedName>> {
        match_nodes!(
            input.into_children();
            [container_field(container_fields)..] => {
                return Ok(container_fields.collect())
            }
        );
    }

    fn container_field(input: ParseNode) -> ParseResult<TypedName> {
        match_nodes!(
            input.into_children();
            [identifier(ident), type_expr(ty)] => {
                return Ok(TypedName::new(ty, ident));
            }
        );
    }

    fn container_decls(input: ParseNode) -> ParseResult<Vec<Node>> {
        match_nodes!(
            input.into_children();
            [container_decl(container_decls)..] => {
                return Ok(container_decls.collect())
            }
        );
    }

    fn container_decl(input: ParseNode) -> ParseResult<Node> {
        match_nodes!(
            input.into_children();
            [decl(decl)] => return Ok(decl),
            // TODO: parse functions
            // [fn_decl(fn_decl)] => return Ok(fn_decl),
        );
    }

    fn decl(input: ParseNode) -> ParseResult<Node> {
        let node = match_nodes!(
            input.into_children();
            [publicity(publicity), identifier(ident), expr(init_expr)] => {
                Node::VarDecl(VarDecl {
                    publicity,
                    ident,
                    ty: Box::new(None),
                    expr: Box::new(init_expr),
                })
            },
            [identifier(ident), type_expr(ty_expr), type_expr(init_expr)] => {
                print_type_of(&ident);
                print_type_of(&ty_expr);
                print_type_of(&init_expr);
                unimplemented!()
            },
        );

        return Ok(node);
    }

    fn publicity(input: ParseNode) -> ParseResult<Publicity> {
        let publicity = if input.as_str().len() == "pub".len() {
            Publicity::Public
        } else {
            Publicity::Private
        };
        return Ok(publicity);
    }

    fn identifier(input: ParseNode) -> ParseResult<GlobalSymbol> {
        return Ok(GlobalSymbol::new(input.as_str()));
    }

    fn type_expr(input: ParseNode) -> ParseResult<Node> {
        let node = match_nodes!(
            input.into_children();
            [struct_decl(struct_decl)] => unimplemented!(),
            [module_decl(module_decl)] => unimplemented!(),
            [identifier(identifier)] => Node::Identifier(identifier),
            [reference_ty(identifier)] => unimplemented!(),
            [pointer_ty(identifier)] => unimplemented!(),
            [expr(identifier)] => unimplemented!(),
        );

        return Ok(node);
    }

    fn struct_decl(input: ParseNode) -> ParseResult<Node> {
        unimplemented!()
    }

    fn module_decl(input: ParseNode) -> ParseResult<Node> {
        unimplemented!()
    }

    fn reference_ty(input: ParseNode) -> ParseResult<Node> {
        unimplemented!()
    }

    fn pointer_ty(input: ParseNode) -> ParseResult<Node> {
        unimplemented!()
    }

    fn expr(input: ParseNode) -> ParseResult<Node> {
        return TaraParser::expr_helper(input.into_pair().into_inner());
    }

    fn primary_expr(input: ParseNode) -> ParseResult<Node> {
        let node = match_nodes!(
            input.into_children();
            [parened_expr(parened_expr)] => parened_expr,
            [identifier(identifier)] => Node::Identifier(identifier),
        );

        return Ok(node);
    }

    fn parened_expr(input: ParseNode) -> ParseResult<Node> {
        Ok(match_nodes!(
                input.into_children();
                [expr(expr)] => expr,
        ))
    }
}

fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}
