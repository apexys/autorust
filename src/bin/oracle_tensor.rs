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

use autorust::dtype::{self, DTypeId};
use autorust::tensor::ix::Ix;
use autorust::tensor::Tensor;
use autorust::uop::ctx::UOpCtx;
use autorust::uop::flat::to_string_flat;
use autorust::uop::sint::SVal;
use autorust::uop::UopId;

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
        /// Advanced indexing: reference a prior program step's tensor as the index.
        tensor: Option<usize>,
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
        /// `cat` sources (step indices); see the "cat" op.
        ys: Option<Vec<usize>>,
        /// `randint` bounds.
        low: Option<i64>,
        high: Option<i64>,
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
                    "f64" => Tensor::from_slice(&mut ctx, &data.iter().copied().collect::<Vec<_>>(), dt, dev),
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
                        } else if let Some(ti) = ix.tensor {
                            Ix::Tensor(tensors[ti].clone())
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
            "cat" => {
                let others: Vec<Tensor> = st.ys.clone().unwrap_or_default().iter().map(|&i| tensors[i].clone()).collect();
                tensors[st.x.unwrap_or(0)].clone().cat(&mut ctx, &others)
            }
            "rand" => autorust::tensor::rand::rand(
                &mut ctx,
                sv(&st.shape.clone().unwrap_or_default()).as_slice(),
                dtype_from_name(st.dtype.clone()),
                st.device.as_deref(),
            ),
            "randn" => autorust::tensor::rand::randn(
                &mut ctx,
                sv(&st.shape.clone().unwrap_or_default()).as_slice(),
                dtype_from_name(st.dtype.clone()),
                st.device.as_deref(),
            ),
            "randint" => autorust::tensor::rand::randint(
                &mut ctx,
                sv(&st.shape.clone().unwrap_or_default()).as_slice(),
                st.low.unwrap_or(0) as i32,
                st.high.unwrap_or(1) as i32,
                dtype_from_name(st.dtype.clone()),
                st.device.as_deref(),
            ),
            other => panic!("TGERR: oracle_tensor: unknown op {other} (step {i})"),
        };
        tensors.push(t);
    }
            if prog.mode.as_deref() == Some("native") {
            // M6 native gate: python fully out of the loop. Realize the final
            // tensor twice (tolist semantics); caches must prevent recompiles.
            let mut bufs = autorust::runtime::buffer::BufTable::default();
            let mut cache = autorust::engine::realize::ProgCache::default();
            let last = &tensors.last().expect("TGERR: empty program").uop().clone();
            let vals1 = autorust::engine::realize::tensor_values(&mut ctx, last.clone(), &mut bufs, &mut cache);
            let vals2 = autorust::engine::realize::tensor_values(&mut ctx, last.clone(), &mut bufs, &mut cache);
            let outs_json: Vec<String> = [&vals1, &vals2]
                .iter()
                .map(|vals| format!("[{}]", vals.iter().map(|v| format!("{v:?}")).collect::<Vec<_>>().join(", ")))
                .collect();
            println!(
                "{{\"ok\": true, \"outs\": [{}], \"compiles\": {}, \"stable\": {}}}",
                outs_json.join(", "),
                autorust::runtime::device::total_compiles(),
                vals1 == vals2
            );
            return;
        }
    if prog.mode.as_deref() == Some("schedule") || prog.mode.as_deref() == Some("schedule_last") {
        // full scheduler pipeline: sink all tensors, callify, schedule, resolve
        let sink_srcs: Vec<autorust::uop::UopId> = if prog.mode.as_deref() == Some("schedule_last") {
            vec![tensors.last().expect("TGERR: empty program").uop().clone()]
        } else {
            tensors.iter().map(|t| t.uop().clone()).collect()
        };
        let big_sink = autorust::tensor::sink(&mut ctx, sink_srcs);
        let (call, _map) = autorust::tensor::callify::transform_to_call(&mut ctx, big_sink);
        let (linear, planned, _held) = autorust::schedule::sched::create_linear_with_vars(&mut ctx, call);
        let root = linear;
        if std::env::var("TG_M5_RENDER").is_ok() {
            render_calls(&mut ctx, planned);
            return;
        }
        if std::env::var("TG_OCL_RENDER").is_ok() {
            render_calls_opencl(&mut ctx, planned);
            return;
        }
        if std::env::var("TG_M5_EXEC").is_ok() {
            let steps: Vec<ExecStep> = prog.prog.iter().map(|st| ExecStep {
                op: st.op.clone(), data: st.data.clone(),
            }).collect();
            exec_linear(&mut ctx, planned, &steps);
            return;
        }
        if std::env::var("TG_M5_CODEGEN").is_ok() {
            // lower the first compute kernel body through the M5 pipeline
            let mut ast: Option<UopId> = None;
            for c in ctx.srcs(&root).iter() {
                let body = ctx.srcs(c)[0].clone();
                if ctx.op_of(&body) == autorust::uop::Ops::Sink {
                    let mut stack = vec![body.clone()];
                    let mut seen = std::collections::HashSet::new();
                    let mut has_alu = false;
                    while let Some(n) = stack.pop() {
                        if !seen.insert(n.id()) { continue; }
                        if matches!(ctx.op_of(&n), autorust::uop::Ops::Add | autorust::uop::Ops::Mul) { has_alu = true; }
                        stack.extend(ctx.srcs(&n).iter().cloned());
                    }
                    if has_alu {
                        ast = Some(body);
                        break;
                    }
                }
            }
            let ast = ast.expect("TGERR: no compute kernel body");
            let lowered = autorust::codegen::pipeline::full_rewrite_to_sink(&mut ctx, ast.clone());
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
    let root = tensors.last().map(|t| t.uop().clone()).unwrap_or_else(|| panic!("TGERR: empty program"));
    ctx.g.roots = vec![root];
    let flat = ctx.g.to_flat();
    println!("{{\"ok\": true, \"graph\": {}}}", to_string_flat(&flat));
    let _ = none_str;
}

/// M8 OCL gate: render each compute kernel's linearized uops with the
/// OpenCL-flavored cstyle renderer and emit the text per call.
fn render_calls_opencl(ctx: &mut UOpCtx, planned: UopId) {
    let ren = autorust::renderer::Renderer::opencl();
    let calls: Vec<UopId> = ctx.srcs(&planned).to_vec();
    let mut outs: Vec<String> = Vec::new();
    for call in calls.iter() {
        let body = &ctx.srcs(call)[0].clone();
        if ctx.op_of(body) == autorust::uop::Ops::Copy {
            outs.push("null".to_string());
            continue;
        }
        let sunk = autorust::codegen::pipeline::full_rewrite_to_sink_with(ctx, body.clone(), &ren, true);
        let lst = autorust::codegen::linearize::linearize(ctx, sunk);
        let text = autorust::renderer::cstyle::render_opencl(ctx, &lst);
        let mut j = String::from("\"");
for ch in text.chars() {
    match ch {
        '"' => j.push_str("\\\""),
        '\\' => j.push_str("\\\\"),
        '\n' => j.push_str("\\n"),
        '\t' => j.push_str("\\t"),
        '\r' => j.push_str("\\r"),
        c if (c as u32) < 0x20 => j.push_str(&format!("\\u{:04x}", c as u32)),
        c => j.push(c),
    }
}
j.push('"');
outs.push(j);
    }
    println!("{{\"ok\": true, \"renders\": [{}]}}", outs.join(", "));
}

/// M5 cstyle gate: render each compute kernel's linearized uops with the
/// clang-flavored cstyle renderer and emit the text per call.
fn render_calls(ctx: &mut UOpCtx, planned: UopId) {
    use autorust::uop::Ops;
    let ren = autorust::renderer::Renderer::clang();
    let calls: Vec<UopId> = ctx.srcs(&planned).to_vec();
    let mut outs: Vec<String> = Vec::new();
    for call in calls.iter() {
        let body = &ctx.srcs(call)[0].clone();
        if ctx.op_of(body) == Ops::Copy {
            outs.push("null".to_string());
            continue;
        }
        let sunk = autorust::codegen::pipeline::full_rewrite_to_sink_with(ctx, body.clone(), &ren, true);
        let lst = autorust::codegen::linearize::linearize(ctx, sunk);
        let text = autorust::renderer::cstyle::render_clang(ctx, &lst);
        let mut j = String::from("\"");
for ch in text.chars() {
    match ch {
        '"' => j.push_str("\\\""),
        '\\' => j.push_str("\\\\"),
        '\n' => j.push_str("\\n"),
        '\t' => j.push_str("\\t"),
        '\r' => j.push_str("\\r"),
        c if (c as u32) < 0x20 => j.push_str(&format!("\\u{:04x}", c as u32)),
        c => j.push(c),
    }
}
j.push('"');
outs.push(j);
    }
    println!("{{\"ok\": true, \"renders\": [{}]}}", outs.join(", "));
}

/// M5 numeric gate: execute every kernel call through the uops interpreter
/// (`PythonProgram.__call__` model) and emit the output buffer values.
#[derive(nanoserde::DeJson, Default, Clone)]
struct ExecStep {
    op: String,
    data: Option<Vec<f64>>,
}

fn exec_linear(ctx: &mut UOpCtx, linear: UopId, prog: &[ExecStep]) {
    use autorust::runtime::interp::{exec_kernel, from_storage_scalar, to_storage_scalar, MemView};
    use autorust::uop::arg::Arg;
    use autorust::uop::Ops;

    // from_slice data (in program order) fills the input views
    let input_data: Vec<Vec<f64>> = prog
        .iter()
        .filter(|st| st.op == "from_slice")
        .map(|st| st.data.clone().unwrap_or_default())
        .collect();
    let mut input_iter = input_data.iter();

    // resolve each call's args to (buffer uop, byte offset, view dtype, view numel)
    struct ArgView {
        mem: std::rc::Rc<std::cell::RefCell<Vec<u8>>>,
        byte_off: usize,
        itemsize: usize,
        numel: usize,
        dt: autorust::dtype::DTypeId,
    }
    impl ArgView {
        fn memview(&self) -> MemView { MemView(self.mem.clone(), self.byte_off, self.itemsize, self.numel) }
    }

    let mut allocs: std::collections::HashMap<u32, std::rc::Rc<std::cell::RefCell<Vec<u8>>>> = std::collections::HashMap::new();
    let mut outs_json: Vec<String> = Vec::new();
    let calls: Vec<UopId> = ctx.srcs(&linear).to_vec();
    for call in calls.iter() {
        let body = ctx.srcs(call)[0].clone();
        // resolve arg views (walk BITCAST/SHRINK chains down to the BUFFER)
        let mut arg_views: Vec<ArgView> = Vec::new();
        let call_args: Vec<UopId> = ctx.srcs(call)[1..].to_vec();
        for arg in call_args.iter() {
            let mut cur = arg.clone();
            let mut byte_off = 0usize;
            let mut view_dt = ctx.dt_of(arg);
            let mut view_nbytes = ctx.max_numel(arg.clone()) as usize * view_dt.itemsize();
            while ctx.op_of(&cur) != Ops::Buffer {
                match ctx.op_of(&cur) {
                    Ops::Bitcast => { view_dt = ctx.dt_of(&cur); }
                    Ops::Shrink => {
                        let base_dt = ctx.dt_of(&ctx.srcs(&cur)[0]);
                        let off = match ctx.val(&ctx.srcs(&cur)[1]) {
                            autorust::dtype::ConstVal::Int(v) => v as usize,
                            other => panic!("TGERR: exec SHRINK offset {other:?}"),
                        };
                        let sz = match ctx.val(&ctx.srcs(&cur)[2]) {
                            autorust::dtype::ConstVal::Int(v) => v as usize,
                            other => panic!("TGERR: exec SHRINK size {other:?}"),
                        };
                        byte_off += off * base_dt.itemsize();
                        view_nbytes = sz * base_dt.itemsize();
                    }
                    Ops::Cast | Ops::Index => {}
                    other => panic!("TGERR: exec arg walk op {other:?}"),
                }
                cur = ctx.srcs(&cur)[0].clone();
            }
            let buf = allocs.entry(cur.id()).or_insert_with(|| {
                let nel = ctx.max_numel(cur.clone()) as usize;
                let isz = ctx.dt_of(&cur).itemsize();
                std::rc::Rc::new(std::cell::RefCell::new(vec![0u8; nel * isz]))
            }).clone();
            let isz = view_dt.itemsize();
            arg_views.push(ArgView { mem: buf, byte_off, itemsize: isz, numel: view_nbytes / isz, dt: view_dt });
        }
        // bind PARAM slots (call args are in sorted-slot order)
        let mut slots: Vec<i64> = ctx.toposort(&body).iter().filter_map(|pu| {
            if ctx.op_of(pu) == Ops::Param
                && let Arg::Param(p) = ctx.arg_of(pu) { return Some(p.slot); }
            None
        }).collect();
        slots.sort();
        slots.dedup();
        let mut views: std::collections::HashMap<i64, MemView> = std::collections::HashMap::new();
        for (i, &s) in slots.iter().enumerate() {
            views.insert(s, arg_views.get(i).unwrap_or_else(|| panic!("TGERR: exec missing arg for slot {s}")).memview());
        }
        // SDMA copy call: fill the src view (arg 1, the PYTHON const buffer) with
        // the from_slice data (python copyin), then memcpy dst <- src
        if ctx.op_of(&body) == Ops::Copy {
            let (dst, src) = (&arg_views[0], &arg_views[1]);
            let data = input_iter.next().unwrap_or_else(|| panic!("TGERR: exec no input data for copy"));
            for (j, &v) in data.iter().enumerate() {
                let sv = to_storage_scalar(v, src.dt);
                let mut b = src.mem.borrow_mut();
                for (k, x) in sv.iter().enumerate() { b[src.byte_off + j * src.itemsize + k] = *x; }
            }
            let n = dst.numel * dst.itemsize;
            let dstb = dst.mem.clone();
            let srcb = src.mem.clone();
            let dst_bytes: Vec<u8> = {
                let s = srcb.borrow();
                (0..n).map(|k| s[src.byte_off + k]).collect()
            };
            {
                let mut d = dstb.borrow_mut();
                for (k, x) in dst_bytes.into_iter().enumerate() { d[dst.byte_off + k] = x; }
            }
            continue;
        }
        // ins/outs analysis (ProgramInfo.from_sink): LOAD/STORE src[0] chain to PARAM
        let sunk = autorust::codegen::pipeline::full_rewrite_to_sink(ctx, body);
        let lst = autorust::codegen::linearize::linearize(ctx, sunk);
        let mut ins: Vec<i64> = Vec::new();
        let mut outs: Vec<i64> = Vec::new();
        for u in &lst {
            if matches!(ctx.op_of(u), Ops::Load | Ops::Store) {
                let mut idx = ctx.srcs(u)[0].clone();
                while matches!(ctx.op_of(&idx), Ops::Index | Ops::Shrink | Ops::Cast | Ops::Bitcast) {
                    idx = ctx.srcs(&idx)[0].clone();
                }
                if ctx.op_of(&idx) == Ops::Param
                    && let Arg::Param(p) = ctx.arg_of(&idx) {
                        let list = if ctx.op_of(u) == Ops::Store { &mut outs } else { &mut ins };
                        if !list.contains(&p.slot) { list.push(p.slot); }
                    }
            }
        }
        // fill inputs: from_slice data enters via the SDMA copy's src only
        // (compute-call inputs are intermediates written by prior kernels)
        // run
        exec_kernel(ctx, &lst, &views, &[]);
        // collect outs
        for &slot in &outs {
            let av = &arg_views[slots.iter().position(|&s| s == slot).unwrap_or_else(|| panic!("TGERR: slot {slot}"))];
            let vals: Vec<String> = (0..av.numel).map(|j| {
                let boff = av.byte_off + j * av.itemsize;
                let bytes: Vec<u8> = (0..av.itemsize).map(|k| av.mem.borrow()[boff + k]).collect();
                format!("{:?}", from_storage_scalar(&bytes, av.dt))
            }).collect();
            outs_json.push(format!("[{}]", vals.join(", ")));
        }
    }
    println!("{{\"ok\": true, \"outs\": [{}]}}", outs_json.join(", "));
}
