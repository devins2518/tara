use crate::{
    ast::{
        BinOp, Call, IfExpr, ModuleInner, Node, NodeKind, Publicity, RefTy, SizedNumberLiteral,
        StructInit, StructInner, SubroutineDecl, TypedName, UnOp, VarDecl,
    },
    builtin::Mutability,
};
use anyhow::Result as AResult;
use num_bigint::BigUint;
use pest::{
    pratt_parser::{Assoc, Op, PrattParser},
    Span,
};
use pest_consume::{match_nodes, Error, Parser};
use symbol_table::GlobalSymbol;

type ParseResult<T> = std::result::Result<T, Error<Rule>>;
type ParseNode<'i> = pest_consume::Node<'i, Rule, ()>;

// Rebuild with grammar changes
const _GRAMMAR_FILE: &str = include_str!("syntax.pest");

lazy_static::lazy_static! {
    static ref PRATT: PrattParser<Rule> = PrattParser::new()
        .op(Op::prefix(Rule::neg_operator))
        .op(Op::postfix(Rule::call_operator))
        .op(Op::postfix(Rule::deref_operator))
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
        .op(Op::infix(Rule::access_operator, Assoc::Left));
}

#[derive(Parser)]
#[grammar = "syntax.pest"]
pub struct TaraParser;

impl TaraParser {
    pub fn parse_source<'a>(source: &'a str) -> AResult<Node> {
        let inputs = TaraParser::parse(Rule::root, source)?;

        let root_node = inputs.single()?;

        let root = TaraParser::root(root_node)?;

        let node = Node::new(NodeKind::StructDecl(root), Span::new(source, 0, 0).unwrap());

        return Ok(node);
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
                let node = match op.as_rule() {
                    Rule::or_operator => NodeKind::Or(bin_op),
                    Rule::and_operator => NodeKind::And(bin_op),
                    Rule::lt_operator => NodeKind::Lt(bin_op),
                    Rule::lte_operator => NodeKind::Lte(bin_op),
                    Rule::gt_operator => NodeKind::Gt(bin_op),
                    Rule::gte_operator => NodeKind::Gte(bin_op),
                    Rule::eq_operator => NodeKind::Eq(bin_op),
                    Rule::neq_operator => NodeKind::Neq(bin_op),
                    Rule::bitwise_and_operator => NodeKind::BitAnd(bin_op),
                    Rule::bitwise_or_operator => NodeKind::BitOr(bin_op),
                    Rule::bitwise_xor_operator => NodeKind::BitXor(bin_op),
                    Rule::add_operator => NodeKind::Add(bin_op),
                    Rule::sub_operator => NodeKind::Sub(bin_op),
                    Rule::mul_operator => NodeKind::Mul(bin_op),
                    Rule::div_operator => NodeKind::Div(bin_op),
                    Rule::access_operator => NodeKind::Access(bin_op),
                    _ => unreachable!(),
                };

                Ok(Node::new(node, op.as_span()))
            })
            .map_prefix(|op, rhs| {
                let rhs_box = Box::new(rhs?);
                let un_op = UnOp { lhs: rhs_box };
                let node = match op.as_rule() {
                    Rule::neg_operator => NodeKind::Negate(un_op),
                    _ => unreachable!(),
                };

                Ok(Node::new(node, op.as_span()))
            })
            .map_postfix(|lhs, op| {
                let span = op.as_span();
                let lhs_box = Box::new(lhs?);
                let node = match op.as_rule() {
                    Rule::call_operator => {
                        let args = TaraParser::call_operator(ParseNode::new(op))?;
                        let call = Call {
                            call: lhs_box,
                            args,
                        };

                        NodeKind::Call(call)
                    }
                    Rule::deref_operator => {
                        let un_op = UnOp { lhs: lhs_box };
                        NodeKind::Deref(un_op)
                    }
                    _ => unreachable!(),
                };
                Ok(Node::new(node, span))
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
            [fn_decl(fn_decl)] => return Ok(fn_decl),
        );
    }

    fn decl(input: ParseNode) -> ParseResult<Node> {
        let span = input.as_span();
        let node = match_nodes!(
            input.into_children();
            [publicity(publicity), identifier(ident), expr(init_expr)] => {
                NodeKind::VarDecl(VarDecl::new (publicity,
                        ident,
                        None,
                        init_expr,
                ))
            },
            [publicity(publicity), identifier(ident), type_expr(type_expr), expr(init_expr)] => {
                NodeKind::VarDecl(VarDecl::new( publicity, ident, Some(type_expr), init_expr))
            },
        );

        return Ok(Node::new(node, span));
    }

    fn fn_decl(input: ParseNode) -> ParseResult<Node> {
        let span = input.as_span();
        let node = match_nodes!(
            input.into_children();
            [publicity(publicity), identifier(ident), param_list(params), type_expr(ret_type), block(body)] => {
                NodeKind::SubroutineDecl(SubroutineDecl::new(publicity, ident, params, ret_type, body))
            }
        );

        return Ok(Node::new(node, span));
    }

    fn param(input: ParseNode) -> ParseResult<TypedName> {
        return Ok(match_nodes!(
            input.into_children();
            [identifier(ident), type_expr(ty)] => TypedName::new(ty, ident),
        ));
    }

    fn param_list(input: ParseNode) -> ParseResult<Vec<TypedName>> {
        return Ok(match_nodes!(
            input.into_children();
            [param(param)..] => param.collect()
        ));
    }

    fn block(input: ParseNode) -> ParseResult<Vec<Node>> {
        return Ok(match_nodes!(
                input.into_children();
                [statement(statements)..] => statements.collect()
        ));
    }

    fn statement(input: ParseNode) -> ParseResult<Node> {
        return Ok(match_nodes!(
                input.into_children();
                [decl_statement(decl_statement)] => decl_statement,
                [expr(expr)] => expr,
        ));
    }

    fn decl_statement(input: ParseNode) -> ParseResult<Node> {
        let span = input.as_span();
        let node = match_nodes!(
            input.into_children();
            [identifier(ident), expr(init_expr)] => NodeKind::LocalVarDecl(VarDecl::new(
                Publicity::Private,
                ident,
                None,
                init_expr,
            )),
            [identifier(ident), type_expr(type_expr), expr(init_expr)] => NodeKind::LocalVarDecl(VarDecl::new(
                Publicity::Private,
                ident,
                Some(type_expr),
                init_expr
            )),
        );

        return Ok(Node::new(node, span));
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
        let span = input.as_span();
        let node = match_nodes!(
            input.into_children();
            [struct_decl(struct_decl)] => struct_decl,
            [module_decl(module_decl)] => module_decl,
            [identifier(identifier)] => Node::new(NodeKind::Identifier(identifier), span),
            [reference_ty(reference_ty)] => reference_ty,
            [pointer_ty(pointer_ty)] => pointer_ty,
        );

        return Ok(node);
    }

    fn struct_decl(input: ParseNode) -> ParseResult<Node> {
        let span = input.as_span();
        Ok(match_nodes!(
                input.into_children();
                [struct_inner(struct_inner)] => Node::new(NodeKind::StructDecl(struct_inner), span),
        ))
    }

    fn struct_inner(input: ParseNode) -> ParseResult<StructInner> {
        let (fields, members) = match_nodes!(
            input.into_children();
            [container_decls(container_decls_pre), field_list(fields), container_decls(container_decls_post)] => {
                let mut members = Vec::new();
                members.extend(container_decls_pre) ;
                members.extend(container_decls_post) ;
                (fields, members)
            }
        );

        let struct_inner = StructInner::new(fields, members);
        return Ok(struct_inner);
    }

    fn module_decl(input: ParseNode) -> ParseResult<Node> {
        let span = input.as_span();
        Ok(match_nodes!(
                input.into_children();
                [module_inner(module_inner)] => Node::new(NodeKind::ModuleDecl(module_inner), span),
        ))
    }

    fn module_inner(input: ParseNode) -> ParseResult<ModuleInner> {
        let (fields, members) = match_nodes!(
            input.into_children();
            [module_decls(module_decls_pre), field_list(fields), module_decls(module_decls_post)] => {
                let mut members = Vec::new();
                members.extend(module_decls_pre) ;
                members.extend(module_decls_post) ;
                (fields, members)
            }
        );

        let module_inner = ModuleInner::new(fields, members);
        return Ok(module_inner);
    }

    fn module_decls(input: ParseNode) -> ParseResult<Vec<Node>> {
        match_nodes!(
            input.into_children();
            [module_inner_decl(module_inner_decls)..] => {
                return Ok(module_inner_decls.collect())
            }
        );
    }

    fn module_inner_decl(input: ParseNode) -> ParseResult<Node> {
        match_nodes!(
            input.into_children();
            [decl(decl)] => return Ok(decl),
            [comb_decl(comb_decl)] => return Ok(comb_decl),
        );
    }

    fn comb_decl(input: ParseNode) -> ParseResult<Node> {
        let span = input.as_span();
        let node = match_nodes!(
            input.into_children();
            [publicity(publicity), identifier(ident), param_list(params), type_expr(ret_type), block(body)] => {
                NodeKind::SubroutineDecl(SubroutineDecl::new(publicity, ident, params, ret_type, body))
            }
        );

        return Ok(Node::new(node, span));
    }

    fn field_list(input: ParseNode) -> ParseResult<Vec<TypedName>> {
        return Ok(match_nodes!(
            input.into_children();
            [field(fields)..] => fields.collect()
        ));
    }

    fn field(input: ParseNode) -> ParseResult<TypedName> {
        return Ok(match_nodes!(
            input.into_children();
            [identifier(ident), type_expr(ty)] => TypedName::new(ty, ident)
        ));
    }

    fn ptr_var(input: ParseNode) -> ParseResult<()> {
        Ok(())
    }

    fn reference_ty(input: ParseNode) -> ParseResult<Node> {
        let span = input.as_span();
        let ref_ty = match_nodes!(
            input.into_children();
            [expr(expr)] => RefTy::new(Mutability::Immutable, expr),
            [ptr_var(_), expr(expr)] => RefTy::new(Mutability::Mutable, expr),
        );

        return Ok(Node::new(NodeKind::ReferenceTy(ref_ty), span));
    }

    fn pointer_ty(input: ParseNode) -> ParseResult<Node> {
        let span = input.as_span();
        let ref_ty = match_nodes!(
            input.into_children();
            [expr(expr)] => RefTy::new(Mutability::Immutable, expr),
            [ptr_var(_), expr(expr)] => RefTy::new(Mutability::Mutable, expr),
        );

        return Ok(Node::new(NodeKind::PointerTy(ref_ty), span));
    }

    fn expr(input: ParseNode) -> ParseResult<Node> {
        return TaraParser::expr_helper(input.into_pair().into_inner());
    }

    fn primary_expr(input: ParseNode) -> ParseResult<Node> {
        let span = input.as_span();
        let node = match_nodes!(
            input.into_children();
            [parened_expr(parened_expr)] => parened_expr,
            [return_expr(return_expr)] => return_expr,
            [number(number)] => number,
            [if_expr(if_expr)] => if_expr,
            [struct_init_expr(struct_init_expr)] => struct_init_expr,
            [type_expr(type_expr)] => type_expr,
            [identifier(identifier)] => Node::new(NodeKind::Identifier(identifier), span),
        );

        return Ok(node);
    }

    fn parened_expr(input: ParseNode) -> ParseResult<Node> {
        Ok(match_nodes!(
            input.into_children();
            [expr(expr)] => expr,
        ))
    }

    fn return_expr(input: ParseNode) -> ParseResult<Node> {
        let span = input.as_span();
        Ok(match_nodes!(
            input.into_children();
            [expr(expr)] => {
                let un_op = UnOp{lhs: Box::new(expr)};
                Node::new(NodeKind::Return(un_op), span)
            },
        ))
    }

    fn if_expr(input: ParseNode) -> ParseResult<Node> {
        let span = input.as_span();
        Ok(match_nodes!(
            input.into_children();
            [expr(cond), expr(body), expr(else_body)] => {
                let if_expr = IfExpr{
                    cond: Box::new(cond),
                    body: Box::new(body),
                    else_body: Box::new(else_body),
                };
                Node::new(NodeKind::IfExpr(if_expr), span)
            },
        ))
    }

    fn struct_init_expr(input: ParseNode) -> ParseResult<Node> {
        let span = input.as_span();
        Ok(match_nodes!(
            input.into_children();
            [type_expr(ty), field_init_list(fields)] => {
                let struct_init = StructInit::new(ty, fields);
                Node::new(NodeKind::StructInit(struct_init), span)
            },
        ))
    }

    fn field_init_list(input: ParseNode) -> ParseResult<Vec<TypedName>> {
        return Ok(match_nodes!(
            input.into_children();
            [field_init(fields)..] => fields.collect()
        ));
    }

    fn field_init(input: ParseNode) -> ParseResult<TypedName> {
        return Ok(match_nodes!(
            input.into_children();
            [identifier(ident), expr(expr)] => TypedName::new(expr, ident),
        ));
    }

    fn number(input: ParseNode) -> ParseResult<Node> {
        let span = input.as_span();
        Ok(match_nodes!(
            input.into_children();
            [decimal_number(decimal)] => Node::new(NodeKind::NumberLiteral(decimal), span),
            [binary_number(binary)] => Node::new(NodeKind::NumberLiteral(binary), span),
            [hex_number(hex)] => Node::new(NodeKind::NumberLiteral(hex), span),
            [bitwidth_binary_literal(binary)] => binary,
            [bitwidth_hex_literal(hex)] => hex,
        ))
    }

    fn nonzero_decimal_number(input: ParseNode) -> ParseResult<BigUint> {
        Ok(BigUint::parse_bytes(input.as_str().as_bytes(), 10).unwrap())
    }

    fn decimal_number(input: ParseNode) -> ParseResult<BigUint> {
        Ok(BigUint::parse_bytes(input.as_str().as_bytes(), 10).unwrap())
    }

    fn binary_number(input: ParseNode) -> ParseResult<BigUint> {
        Ok(BigUint::parse_bytes(input.as_str().as_bytes(), 2).unwrap())
    }

    fn hex_number(input: ParseNode) -> ParseResult<BigUint> {
        Ok(BigUint::parse_bytes(input.as_str().as_bytes(), 16).unwrap())
    }

    fn bitwidth_binary_literal(input: ParseNode) -> ParseResult<Node> {
        let span = input.as_span();
        Ok(match_nodes!(
            input.into_children();
            [nonzero_decimal_number(big_size), binary_number(literal)] => {
                // TODO: do size checks here
                let size = big_size.to_u32_digits()[0] as u16;
                let sized_number_literal = SizedNumberLiteral{size,literal};
                let node = NodeKind::SizedNumberLiteral(sized_number_literal);

                Node::new(node, span)
            },
        ))
    }

    fn bitwidth_hex_literal(input: ParseNode) -> ParseResult<Node> {
        let span = input.as_span();
        Ok(match_nodes!(
            input.into_children();
            [nonzero_decimal_number(big_size), hex_number(literal)] => {
                // TODO: do size checks here
                let size = big_size.to_u32_digits()[0] as u16;
                let sized_number_literal = SizedNumberLiteral{size,literal};
                let node = NodeKind::SizedNumberLiteral(sized_number_literal);

                Node::new(node, span)
            },
        ))
    }

    fn call_operator(input: ParseNode) -> ParseResult<Vec<Node>> {
        Ok(match_nodes!(
            input.into_children();
            [expr(expr)..] => expr.collect(),
        ))
    }
}

fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}
