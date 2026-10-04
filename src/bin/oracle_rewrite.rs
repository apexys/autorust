//! M2 differential oracle: reads one JSON request per line
//!   {"pm": "<pm-name>", "graph": <FlatGraph>}
//! rewrites the graph with the named Rust PatternMatcher, and answers
//!   {"ok": true, "graph": <FlatGraph>}
//! or {"ok": false, "error": "..."}  (panics caught, like Python exceptions).
use std::io::{BufRead, Write};

use autorust::uop::ctx::UOpCtx;
use autorust::uop::flat::{parse_flat, to_string_flat};
use autorust::uop::graph::Graph;
use autorust::uop::sym::{pm_commutative, pm_mop_cleanup, symbolic, symbolic_simple};
use autorust::uop::upat::{graph_rewrite, CtxVal};

static PM_LOAD_COLLAPSE: autorust::uop::upat::PmSlot = autorust::uop::upat::PmSlot::new();
static PM_REDUCE_LOAD_COLLAPSE: autorust::uop::upat::PmSlot = autorust::uop::upat::PmSlot::new();
static PM_SPLIT_RANGES: autorust::uop::upat::PmSlot = autorust::uop::upat::PmSlot::new();
static PM_ADD_GPUDIMS: autorust::uop::upat::PmSlot = autorust::uop::upat::PmSlot::new();
static PM_DECOMP: autorust::uop::upat::PmSlot = autorust::uop::upat::PmSlot::new();
static PM_DTYPE_DECOMP: autorust::uop::upat::PmSlot = autorust::uop::upat::PmSlot::new();

fn pm_by_name(name: &str) -> Option<&'static autorust::uop::upat::PatternMatcher> {
    match name {
        "symbolic_simple" => Some(symbolic_simple()),
        "commutative" => Some(pm_commutative()),
        "mop_cleanup" => Some(pm_mop_cleanup()),
        "div_and_mod_symbolic" => Some(autorust::uop::divandmod::div_and_mod_symbolic()),
        "symbolic" => Some(symbolic()),
        "pm_load_collapse" => Some(PM_LOAD_COLLAPSE.get_or_init(autorust::codegen::simplify::pm_load_collapse)),
        "pm_reduce_load_collapse" => Some(PM_REDUCE_LOAD_COLLAPSE.get_or_init(autorust::codegen::simplify::pm_reduce_load_collapse)),
        "pm_split_ranges" => Some(PM_SPLIT_RANGES.get_or_init(autorust::codegen::simplify::pm_split_ranges)),
        "pm_add_gpudims" => Some(PM_ADD_GPUDIMS.get_or_init(autorust::codegen::gpudims::pm_add_gpudims)),
        "pm_decomp" => Some(PM_DECOMP.get_or_init(|| autorust::codegen::decomp::pm_decomp(true))),
        "pm_dtype_decomps" => Some(PM_DTYPE_DECOMP.get_or_init(autorust::codegen::decomp_dtype::pm_dtype_decomps)),
        _ => None,
    }
}

fn main() {
    let stdin = std::io::stdin();
    let mut out = std::io::stdout();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let resp = std::panic::catch_unwind(|| process_line(&line)).unwrap_or_else(|p| {
            let msg = panic_msg(&p);
            serde_json_err(&format!("panic: {msg}"))
        });
        let _ = writeln!(out, "{resp}");
        let _ = out.flush();
    }
}

fn panic_msg(p: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

fn serde_json_err(e: &str) -> String {
    // minimal JSON error envelope (avoid adding a json dependency)
    format!("{{\"ok\": false, \"error\": \"{}\"}}", escape(e))
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', " ")
}

fn process_line(line: &str) -> String {
    // minimal field extraction: {"pm": ..., "graph": {...}}
    let pm_name = extract_string_field(line, "pm").unwrap_or_default();
    let graph_json = extract_object_field(line, "graph").unwrap_or_default();
    let pm = match pm_by_name(&pm_name) {
        Some(p) => p,
        None => return serde_json_err(&format!("unknown pm {pm_name:?}")),
    };
    let fg = match parse_flat(&graph_json) {
        Ok(f) => f,
        Err(e) => return serde_json_err(&format!("parse: {e}")),
    };
    let g = match Graph::from_flat(&fg) {
        Ok(g) => g,
        Err(e) => return serde_json_err(&format!("from_flat: {e}")),
    };
    let root = match g.roots.last() {
        Some(r) => r.clone(),
        None => return serde_json_err("no roots"),
    };
    let mut ctx = UOpCtx::new();
    ctx.g = g;
    let ctxv = if pm_name == "pm_split_ranges" {
        CtxVal::Dict(std::rc::Rc::new(std::cell::RefCell::new(std::collections::HashMap::new())))
    } else if pm_name == "pm_dtype_decomps" {
        let dctx = autorust::codegen::decomp_dtype::DtypeDecompCtx {
            dtypes: Default::default(),
            supported: autorust::dtype::ALL.to_vec(),
            emulated: autorust::codegen::decomp_dtype::emulated_dtypes(),
        };
        CtxVal::Decomp(std::rc::Rc::new(std::cell::RefCell::new(dctx)))
    } else {
        CtxVal::None
    };
    let rewritten = match graph_rewrite(&mut ctx, root, pm, false, None, false, false, ctxv) {
        Ok(r) => r,
        Err(e) => return serde_json_err(&format!("rewrite: {e}")),
    };
    let out = ctx.g.clone();
    let mut out = out;
    out.roots = vec![rewritten];
    let flat = out.to_flat();
    format!("{{\"ok\": true, \"graph\": {}}}", to_string_flat(&flat))
}

fn extract_string_field(s: &str, field: &str) -> Option<String> {
    let key = format!("\"{field}\":");
    let i = s.find(&key)? + key.len();
    let rest = &s[i..];
    let rest = rest.trim_start();
    if rest.starts_with('"') {
        let end = rest[1..].find('"')?;
        Some(rest[1..1 + end].to_string())
    } else {
        None
    }
}

fn extract_object_field(s: &str, field: &str) -> Option<String> {
    let key = format!("\"{field}\":");
    let i = s.find(&key)? + key.len();
    let rest = &s[i..];
    let rest = rest.trim_start();
    if !rest.starts_with('{') {
        return None;
    }
    // brace matching, respecting strings
    let mut depth = 0usize;
    let mut in_str = false;
    let mut esc = false;
    for (j, c) in rest.char_indices() {
        if esc {
            esc = false;
            continue;
        }
        match c {
            '\\' if in_str => esc = true,
            '"' => in_str = !in_str,
            '{' if !in_str => depth += 1,
            '}' if !in_str => {
                depth -= 1;
                if depth == 0 {
                    return Some(rest[..=j].to_string());
                }
            }
            _ => {}
        }
    }
    None
}
