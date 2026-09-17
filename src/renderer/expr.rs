//! `uop/render.py` `renderer` PM port (subset) — the expr strings python
//! interpolates into cstyle output (SPECIAL extent comments).

use std::collections::HashMap;

use crate::dtype::{self, ConstVal};
use crate::uop::arg::Arg;
use crate::uop::ctx::UOpCtx;
use crate::uop::ops::Ops;
use crate::uop::sint::SVal;
use crate::uop::UopId;

/// python float repr (2.0 → "2.0", nan → "nan", inf → "inf").
fn py_float(v: f64) -> String {
    if v.is_nan() {
        return "nan".to_string();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf".to_string() } else { "-inf".to_string() };
    }
    let s = format!("{v}");
    if s.contains('.') || s.contains('e') {
        s
    } else {
        format!("{s}.0")
    }
}

fn const_str(val: &ConstVal) -> String {
    match val {
        ConstVal::Int(v) => v.to_string(),
        ConstVal::Float(f) => py_float(*f),
        ConstVal::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        other => panic!("TGERR: expr_render const {other:?}"),
    }
}

/// `precedence` (render.py:25).
fn prec(op: Option<Ops>) -> i32 {
    match op {
        Some(Ops::Mul | Ops::Floordiv | Ops::Floormod) => 1,
        Some(Ops::Add | Ops::Sub) => 2,
        Some(Ops::Shl | Ops::Shr) => 3,
        Some(Ops::And) => 4,
        Some(Ops::Xor) => 5,
        Some(Ops::Or) => 6,
        _ => 99,
    }
}

/// `strip_binary_parens` (render.py:26) — drop parens where child precedence allows.
fn binary(op: Ops, a: &str, b: &str, left_op: Option<Ops>, right_op: Option<Ops>) -> String {
    let p = prec(Some(op));
    let strip = |s: &str| -> String { s[1..s.len() - 1].to_string() };
    let left = if a.starts_with('(') && prec(left_op) <= p { strip(a) } else { a.to_string() };
    let right = if b.starts_with('(') && prec(right_op) < p { strip(b) } else { b.to_string() };
    format!("({left}{}{right})", sym_of(op))
}

fn sym_of(op: Ops) -> &'static str {
    match op {
        Ops::Add => "+",
        Ops::Sub => "-",
        Ops::Floordiv => "//",
        Ops::Floormod => "%",
        Ops::Shl => "<<",
        Ops::Shr => ">>",
        Ops::Mul => "*",
        Ops::And => "&",
        Ops::Or => "|",
        Ops::Xor => "^",
        _ => panic!("TGERR: expr_render sym {op:?}"),
    }
}

/// `UOp.render(simplify=True)` — the expr string for a (small) symbolic graph.
pub fn expr_render(ctx: &mut UOpCtx, u: UopId) -> String {
    // simplify first (python: self.simplify()); const-folded → plain number
    match ctx.ssimplify(u.clone()) {
        Ok(SVal::I(v)) => v.to_string(),
        Ok(SVal::U(n)) => {
            let mut cache: HashMap<UopId, String> = HashMap::new();
            expr_node(ctx, n, &mut cache)
        }
        Err(_) => {
            let mut cache: HashMap<UopId, String> = HashMap::new();
            expr_node(ctx, u, &mut cache)
        }
    }
}

fn expr_node(ctx: &mut UOpCtx, u: UopId, cache: &mut HashMap<UopId, String>) -> String {
    if let Some(s) = cache.get(&u) {
        return s.clone();
    }
    let (op, dt, srcs, _arg, _tag) = ctx.snapshot(&u);
    let s = |ctx: &mut UOpCtx, id: UopId, cache: &mut HashMap<UopId, String>| expr_node(ctx, id, cache);
    let out = match op {
        Ops::Param => match ctx.arg_of(&u).clone() {
            Arg::Param(p) => match &p.name {
                Some(n) => n.clone(),
                None => format!("p{}", p.slot),
            },
            other => panic!("TGERR: expr_render PARAM {other:?}"),
        },
        Ops::Buffer => match ctx.arg_of(&u) {
            Arg::Param(p) => match &p.name {
                Some(n) => n.clone(),
                None => format!("b{}", p.slot),
            },
            other => panic!("TGERR: expr_render BUFFER {other:?}"),
        },
        Ops::After => s(ctx, (srcs[0]).clone(), cache),
        Ops::Special => match ctx.arg_of(&u) {
            Arg::Str(s) => s.clone(),
            other => panic!("TGERR: expr_render SPECIAL {other:?}"),
        },
        Ops::Range => {
            if dt == dtype::VOID {
                let n = match ctx.arg_of(&u) {
                    Arg::Seq { items, .. } => items.last().map(|it| match it {
                        Arg::Int(v) => v.to_string(),
                        _ => "?".to_string(),
                    }).unwrap_or_else(|| "?".to_string()),
                    other => panic!("TGERR: expr_render RANGE {other:?}"),
                };
                format!("loop{n}")
            } else {
                let rs = range_str_of(ctx, u.clone());
                format!("r{rs}")
            }
        }
        Ops::Const => const_str(&ctx.val(&u)),
        Ops::Cast => {
            // weak/bool consts state the value; other casts state the width
            let c = srcs[0].clone();
            if ctx.op_of(&c) == Ops::Const && (dtype::is_weak(dt) || dt == dtype::BOOL) {
                const_str(&ctx.val(&c))
            } else {
                format!("({})({})", ctx.dt_of(&u).attr(), s(ctx, c, cache))
            }
        }
        Ops::Neg => format!("(-{})", s(ctx, (srcs[0]).clone(), cache)),
        Ops::Reciprocal => format!("(1/{})", s(ctx, (srcs[0]).clone(), cache)),
        Ops::Max => format!("max({}, {})", s(ctx, (srcs[0]).clone(), cache), s(ctx, (srcs[1]).clone(), cache)),
        Ops::Mulacc => format!("({}*{}+{})", s(ctx, (srcs[0]).clone(), cache), s(ctx, (srcs[1]).clone(), cache), s(ctx, (srcs[2]).clone(), cache)),
        Ops::Where => format!("({} if {} else {})", s(ctx, (srcs[1]).clone(), cache), s(ctx, (srcs[0]).clone(), cache), s(ctx, (srcs[2]).clone(), cache)),
        Ops::Cdiv => format!("cdiv({}, {})", s(ctx, (srcs[0]).clone(), cache), s(ctx, (srcs[1]).clone(), cache)),
        Ops::Cmod => format!("cmod({}, {})", s(ctx, (srcs[0]).clone(), cache), s(ctx, (srcs[1]).clone(), cache)),
        Ops::Add | Ops::Sub | Ops::Mul | Ops::Floordiv | Ops::Floormod | Ops::Shl | Ops::Shr | Ops::And | Ops::Or | Ops::Xor => {
            let a = s(ctx, (srcs[0]).clone(), cache);
            let b = s(ctx, (srcs[1]).clone(), cache);
            let (lo, ro) = (Some(ctx.op_of(&srcs[0])), Some(ctx.op_of(&srcs[1])));
            binary(op, &a, &b, lo, ro)
        }
        other => panic!("TGERR: expr_render op {other:?} not ported"),
    };
    cache.insert(u, out.clone());
    out
}

/// `range_str` — '_'.join(str(x) for x in arg[0..-1]).
fn range_str_of(ctx: &UOpCtx, u: UopId) -> String {
    match ctx.arg_of(&u) {
        Arg::Seq { items, .. } => {
            let parts: Vec<String> = items[..items.len().saturating_sub(1)]
                .iter()
                .map(|it| match it {
                    Arg::Int(v) => {
                        if *v >= 0 {
                            v.to_string()
                        } else {
                            format!("m{}", -v)
                        }
                    }
                    Arg::Const(ConstVal::Int(v)) => {
                        if *v >= 0 {
                            v.to_string()
                        } else {
                            format!("m{}", -v)
                        }
                    }
                    other => format!("{other:?}"),
                })
                .collect();
            parts.join("_")
        }
        _ => String::new(),
    }
}
