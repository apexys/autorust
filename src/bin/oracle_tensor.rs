//! M3 differential oracle: replay a tensor-construction program and emit the
//! final graph. Program = list of steps; each step builds one Tensor (handle i
//! = the i-th step). The Python side replays the same program with real
//! tinygrad `Tensor`s and compares UOp keys.
//!
//! Step forms:
//!   {"op": "from_slice", "data": [1.0, 2.0], "kind": "f32",
//!    "dtype": "float32"|null, "device": "PYTHON"|null}
//!   {"op": "from_bytes", "data": "<hex>", "dtype": "uint8", "device": null}
//!   {"op": "arange", "start": 5, "stop": null, "step": 1, "dtype": null}
//!   {"op": "eye", "n": 3, "m": null, "dtype": null}
//!   {"op": "zeros" | "ones", "shape": [2, 3]}
//!   {"op": "reshape", "x": 0, "shape": [3, 2]}
//!   {"op": "getitem", "x": 0, "ixs": [{"slice": [1, 3, 1]} | 1 | "newaxis" | "ellipsis"]}

use autorust::dtype::{self, ConstVal, DTypeId};
use autorust::tensor::ix::Ix;
use autorust::tensor::{ShapeArg, Tensor};
use autorust::uop::ctx::UOpCtx;
use autorust::uop::flat::{to_string_flat, parse_flat};
use autorust::uop::sint::SVal;

fn dtype_from_name(name: Option<String>) -> Option<DTypeId> {
    name.as_ref().and_then(|n| dtype::to_dtype(n))
}

fn sv(shape: &[i64]) -> Vec<SVal> {
    shape.iter().map(|&i| SVal::I(i as i128)).collect()
}

fn main() {
    use std::io::Read;
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).expect("read stdin");
    #[derive(nanoserde::DeJson, Default, Clone)]
    struct IxSpec {
        int: Option<i64>,
        newaxis: Option<bool>,
        ellipsis: Option<bool>,
        slice: Option<Vec<Option<i64>>>,
    }
    #[derive(nanoserde::DeJson, Default)]
    struct Step {
        op: String,
        data: Option<Vec<f64>>,
        hex: Option<String>,
        kind: Option<String>,
        dtype: Option<String>,
        device: Option<String>,
        start: Option<f64>,
        stop: Option<f64>,
        step: Option<f64>,
        n: Option<i64>,
        m: Option<i64>,
        shape: Option<Vec<i64>>,
        x: Option<usize>,
        y: Option<usize>,
        val: Option<f64>,
        axis: Option<Vec<i64>>,
        keepdim: Option<bool>,
        ixs: Option<Vec<IxSpec>>,
    }
    #[derive(nanoserde::DeJson)]
    struct Program {
        prog: Vec<Step>,
        #[nserde(default)]
        mode: Option<String>,
    }

    let prog: Program = nanoserde::DeJson::deserialize_json(&input).unwrap_or_else(|e| {
        eprintln!("{{\"ok\": false, \"error\": \"parse: {e}\"}}");
        std::process::exit(1);
    });
    let mut ctx = UOpCtx::new();
    let mut tensors: Vec<Tensor> = Vec::new();
    let none_str: Option<String> = None;
    for (i, st) in prog.prog.iter().enumerate() {
        let t = match st.op.as_str() {
            "from_slice" => {
                let data = st.data.clone().unwrap_or_default();
                let kind = st.kind.clone().unwrap_or_default();
                let dt = dtype_from_name(st.dtype.clone());
                let dev = st.device.as_deref();
                match kind.as_str() {
                    "f32" => Tensor::from_slice(&mut ctx, &data.iter().map(|&v| v as f32).collect::<Vec<_>>(), dt, dev),
                    "f64" => Tensor::from_slice(&mut ctx, &data.iter().map(|&v| v as f64).collect::<Vec<_>>(), dt, dev),
                    "i32" => Tensor::from_slice(&mut ctx, &data.iter().map(|&v| v as i32).collect::<Vec<_>>(), dt, dev),
                    "i64" => Tensor::from_slice(&mut ctx, &data.iter().map(|&v| v as i64).collect::<Vec<_>>(), dt, dev),
                    "u8" => Tensor::from_slice(&mut ctx, &data.iter().map(|&v| v as u8).collect::<Vec<_>>(), dt, dev),
                    "bool" => Tensor::from_slice(&mut ctx, &data.iter().map(|&v| v != 0.0).collect::<Vec<_>>(), dt, dev),
                    other => panic!("TGERR: oracle_tensor: unknown slice kind {other} (step {i})"),
                }
            }
            "from_bytes" => {
                let hex = st.hex.clone().unwrap_or_default();
                let bytes: Vec<u8> = (0..hex.len() / 2).map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap()).collect();
                let dname = st.dtype.clone().unwrap_or_else(|| "uint8".to_string());
                let dt = dtype::to_dtype(&dname).unwrap_or_else(|| panic!("TGERR: bad dtype {dname}"));
                Tensor::from_bytes(&mut ctx, &bytes, dt, st.device.as_deref())
            }
            "arange" => Tensor::arange(&mut ctx, st.start.unwrap_or(0.0), st.stop, st.step.unwrap_or(1.0), dtype_from_name(st.dtype.clone())),
            "eye" => Tensor::eye(&mut ctx, st.n.unwrap_or(0), st.m, dtype_from_name(st.dtype.clone())),
            "zeros" => Tensor::zeros(&mut ctx, sv(&st.shape.clone().unwrap_or_default()).as_slice()),
            "ones" => Tensor::ones(&mut ctx, sv(&st.shape.clone().unwrap_or_default()).as_slice()),
            "reshape" => tensors[st.x.unwrap_or(0)].clone().reshape(&mut ctx, sv(&st.shape.clone().unwrap_or_default()).as_slice()),
            "getitem" => {
                let parsed: Vec<Ix> = st
                    .ixs
                    .clone()
                    .unwrap_or_default()
                    .iter()
                    .map(|ix| {
                        if let Some(i) = ix.int {
                            Ix::Index(i)
                        } else if ix.newaxis.unwrap_or(false) {
                            Ix::NewAxis
                        } else if ix.ellipsis.unwrap_or(false) {
                            Ix::Ellipsis
                        } else if let Some(s) = &ix.slice {
                            Ix::Range { start: s[0], stop: s[1], step: s[2].unwrap_or(1) }
                        } else {
                            panic!("TGERR: bad IxSpec (step {i})")
                        }
                    })
                    .collect();
                tensors[st.x.unwrap_or(0)].clone().getitem(&mut ctx, &parsed)
            }
            "add" => tensors[st.x.unwrap_or(0)].clone().add(&mut ctx, tensors[st.y.unwrap_or(0)].clone()),
            "mul" => tensors[st.x.unwrap_or(0)].clone().mul(&mut ctx, tensors[st.y.unwrap_or(0)].clone()),
            "addc" => tensors[st.x.unwrap_or(0)].clone().add(&mut ctx, st.val.unwrap_or(0.0)),
            "mulc" => tensors[st.x.unwrap_or(0)].clone().mul(&mut ctx, st.val.unwrap_or(1.0)),
            "neg" => tensors[st.x.unwrap_or(0)].clone().neg(&mut ctx),
            "sum" => tensors[st.x.unwrap_or(0)].clone().sum(
                &mut ctx,
                st.axis.clone().map(|a| a.iter().map(|&v| v as usize).collect::<Vec<_>>()).as_deref(),
                st.keepdim.unwrap_or(false),
                None,
            ),
            "max" => tensors[st.x.unwrap_or(0)].clone().max(
                &mut ctx,
                st.axis.clone().map(|a| a.iter().map(|&v| v as usize).collect::<Vec<_>>()).as_deref(),
                st.keepdim.unwrap_or(false),
            ),
            "contiguous" => tensors[st.x.unwrap_or(0)].clone().contiguous(&mut ctx),
            other => panic!("TGERR: oracle_tensor: unknown op {other} (step {i})"),
        };
        tensors.push(t);
    }
    if prog.mode.as_deref() == Some("schedule") {
        // full scheduler pipeline: sink all tensors, callify, schedule, resolve
        let sink_srcs: Vec<autorust::uop::UopId> = tensors.iter().map(|t| t.uop).collect();
        let big_sink = autorust::tensor::sink(&mut ctx, sink_srcs);
        let (call, _map) = autorust::tensor::callify::transform_to_call(&mut ctx, big_sink);
        let (linear, _planned) = autorust::schedule::sched::create_linear_with_vars(&mut ctx, call);
        let root = linear;
        if std::env::var("TG_M5_CODEGEN").is_ok() {
            // lower the first compute kernel body through the M5 pipeline
            let mut ast = None;
            for &c in ctx.srcs(root).iter() {
                let body = ctx.srcs(c)[0];
                if ctx.op_of(body) == autorust::uop::Ops::Sink {
                    let mut stack = vec![body];
                    let mut seen = std::collections::HashSet::new();
                    let mut has_alu = false;
                    while let Some(n) = stack.pop() {
                        if !seen.insert(n.0) { continue; }
                        if matches!(ctx.op_of(n), autorust::uop::Ops::Add | autorust::uop::Ops::Mul) { has_alu = true; }
                        stack.extend(ctx.srcs(n));
                    }
                    if has_alu {
                        ast = Some(body);
                        break;
                    }
                }
            }
            let ast = ast.expect("TGERR: no compute kernel body");
            let lowered = autorust::codegen::pipeline::full_rewrite_to_sink(&mut ctx, ast);
            ctx.g.roots = vec![lowered];
            let flat = ctx.g.to_flat();
            println!("{{\"ok\": true, \"graph\": {}}}", to_string_flat(&flat));
            return;
        }
        ctx.g.roots = vec![root];
        let flat = ctx.g.to_flat();
        println!("{{\"ok\": true, \"graph\": {}}}", to_string_flat(&flat));
        return;
    }
    let root = tensors.last().map(|t| t.uop).unwrap_or_else(|| panic!("TGERR: empty program"));
    ctx.g.roots = vec![root];
    let flat = ctx.g.to_flat();
    println!("{{\"ok\": true, \"graph\": {}}}", to_string_flat(&flat));
    let _ = none_str;
}
