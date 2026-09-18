//! `renderer/cstyle.py` port (M5) — the `ClangRenderer` subset for the corpus.
//!
//! Renders linearized uops to C text: naming/SSA inlining decisions, const
//! rules, index/load/store, ALU, ranges, and the kernel wrapper + defines.

use crate::dtype::{self, AddrSpace, ConstVal, DTypeId};
use crate::uop::arg::Arg;
use crate::uop::ctx::UOpCtx;
use crate::uop::ops::{group, Ops};
use crate::uop::range::AxisType;
use crate::uop::UopId;
use std::collections::HashMap;

/// One precomputed uop fact (shape/numel lookups need `&mut ctx`, so snapshot).
#[derive(Clone)]
struct Info {
    op: Ops,
    dt: DTypeId,
    srcs: Vec<UopId>,
    addrspace: Option<AddrSpace>,
    derived: Option<AddrSpace>,
    numel: i64,
    shape: Vec<i64>,
    volatile: bool,
}

/// `CStyleLanguage` (renderer/cstyle.py:117) — the language options a C-style
/// backend tweaks. A new C-style backend is a `CStyleLang` literal + facts.
pub struct CStyleLang {
    /// clang flavor swaps sqrt/trunc for __builtin_* with f64 dispatch
    clang_builtins: bool,
    buffer_prefix: &'static str,
    buffer_suffix: &'static str,
    var_prefix: &'static str,
    var_suffix: &'static str,
    smem_align: &'static str,
    smem_prefix: &'static str,
    smem_prefix_for_cast: bool,
    kernel_typedef: &'static str,
    gep_arr_threshold: i64,
    barrier: &'static str,
    infinity: &'static str,
    nan: &'static str,
    /// `type_map` — dtype name overrides (OpenCL C names).
    type_map: &'static [(DTypeId, &'static str)],
    /// `code_for_workitem` — (kind, fn name) for SPECIAL g/l/i.
    workitem: Option<(&'static str, &'static str, &'static str)>,
    /// ClangRenderer-only vector typedef defines block.
    vector_defines: bool,
    /// `float4_style` — Stack literal delimiters.
    float4_style: (&'static str, &'static str),
}

impl CStyleLang {
    /// `ClangRenderer` language options.
    pub fn clang() -> CStyleLang {
        CStyleLang {
            clang_builtins: true,
            buffer_prefix: "",
            buffer_suffix: " restrict",
            var_prefix: "const ",
            var_suffix: "",
            smem_align: "",
            smem_prefix: "",
            smem_prefix_for_cast: true,
            gep_arr_threshold: 0,
            barrier: "",
            infinity: "__builtin_inff()",
            nan: "__builtin_nanf(\"\")",
            type_map: &[(dtype::BOOL, "_Bool"), (dtype::FLOAT16, "__fp16")],
            workitem: None,
            vector_defines: true,
            kernel_typedef: "void",
            float4_style: ("{", "}"),
        }
    }

    /// `OpenCLRenderer` language options (cstyle.py:308).
    pub fn opencl() -> CStyleLang {
        CStyleLang {
            clang_builtins: false,
            buffer_prefix: "__global ",
            buffer_suffix: "",
            var_prefix: "const ",
            var_suffix: "",
            smem_align: "__attribute__ ((aligned (16))) ",
            smem_prefix: "__local ",
            smem_prefix_for_cast: true,
            gep_arr_threshold: 4,
            barrier: "barrier(CLK_LOCAL_MEM_FENCE);",
            infinity: "INFINITY",
            nan: "NAN",
            type_map: &[
                (dtype::INT8, "char"),
                (dtype::UINT8, "uchar"),
                (dtype::UINT32, "uint"),
                (dtype::UINT16, "ushort"),
                (dtype::UINT64, "ulong"),
                (dtype::BFLOAT16, "ushort"),
            ],
            workitem: Some(("get_group_id", "get_local_id", "get_global_id")),
            vector_defines: false,
            kernel_typedef: "__kernel void",
            float4_style: ("(", ")"),
        }
    }

    /// `_render_dtype` — type_map lookup (space-normalized), prefix/suffix by
    /// addrspace, vector size suffix.
    fn render_dtype(&self, dt: DTypeId, sz: i64, addrspace: Option<AddrSpace>, override_ptr: bool) -> String {
        let as_ = addrspace.unwrap_or(AddrSpace::Alu);
        let mut prefix = String::new();
        if matches!(as_, AddrSpace::Local | AddrSpace::Global) {
            if as_ == AddrSpace::Local && self.smem_prefix_for_cast {
                prefix = self.smem_prefix.to_string();
            }
            if as_ == AddrSpace::Global {
                prefix = self.buffer_prefix.to_string();
            }
        }
        let suffix = if matches!(as_, AddrSpace::Local | AddrSpace::Global) || override_ptr { "*" } else { "" };
        let base = self.type_name(dt);
        if sz > 1 {
            format!("{prefix}{base}{sz}{suffix}")
        } else {
            format!("{prefix}{base}{suffix}")
        }
    }

    /// `type_map.get(dtype, dtype.name).replace(" ", "_")`.
    fn type_name(&self, dt: DTypeId) -> String {
        for (d, name) in self.type_map {
            if *d == dt {
                return name.to_string();
            }
        }
        dt.name().replace(' ', "_")
    }

    fn render_type(&self, i: &Info) -> String {
        self.render_dtype(i.dt, i.numel, i.addrspace, false)
    }

    /// `render_access` — u = the INDEX/SHRINK bidx, src[0] = the buffer.
    /// (python uses `u.addrspace` on the bidx — our pipeline carries that in
    /// the derived addrspace, so fall through to it.)
    fn render_access(&self, info: &HashMap<UopId, Info>, id: UopId, bidx: &str) -> String {
        let i = &info[&id];
        let src_dt = info[&i.srcs[0]].dt;
        let as_ = i.addrspace.or(i.derived);
        if i.numel > 1 || i.dt != src_dt {
            format!("*(({})({}))", self.render_dtype(i.dt, i.numel, as_, true), bidx)
        } else {
            format!("*{bidx}")
        }
    }
}

/// `strip_parens` — drop one balanced outer paren pair.
fn strip_parens(s: &str) -> &str {
    if s.starts_with('(') && s.ends_with(')') {
        let mut depth = 0i32;
        for (i, c) in s.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 && i != s.len() - 1 {
                        return s;
                    }
                }
                _ => {}
            }
        }
        &s[1..s.len() - 1]
    } else {
        s
    }
}

/// `to_function_name` (helpers.py:154) — ansistrip then keep ident chars.
pub fn to_function_name_pub(s: &str) -> String {
    to_function_name(s)
}

/// `to_function_name` (helpers.py:154) — ansistrip then keep ident chars.
fn to_function_name(s: &str) -> String {
    let mut clean = String::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            for c2 in chars.by_ref() {
                if c2 == 'm' {
                    break;
                }
            }
        } else {
            clean.push(c);
        }
    }
    clean
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c.to_string()
            } else {
                format!("{:02X}", c as u32)
            }
        })
        .collect()
}

/// `range_str` — '_'.join(str(x) for x in arg[0..-1]).
fn range_str(arg: &Arg) -> String {
    match arg {
        Arg::Seq { tuple: _, items } => {
            let parts: Vec<String> = items[..items.len().saturating_sub(1)]
                .iter()
                .map(|it| {
                    let v = match it {
                        Arg::Int(v) => Some(*v),
                        Arg::Const(ConstVal::Int(v)) => Some(*v as i64),
                        _ => None,
                    };
                    match v {
                        Some(v) if v >= 0 => v.to_string(),
                        Some(v) => format!("m{}", -v),
                        None => format!("{it:?}"),
                    }
                })
                .collect();
            parts.join("_")
        }
        _ => String::new(),
    }
}

/// `ClangRenderer.render(lst)`.
pub fn render_clang(ctx: &mut UOpCtx, lst: &[UopId]) -> String {
    render_cstyle(ctx, lst, &CStyleLang::clang())
}

/// `OpenCLRenderer.render(lst)`.
pub fn render_opencl(ctx: &mut UOpCtx, lst: &[UopId]) -> String {
    render_cstyle(ctx, lst, &CStyleLang::opencl())
}

fn render_cstyle(ctx: &mut UOpCtx, lst: &[UopId], lang: &CStyleLang) -> String {
    // snapshot facts
    let mut info: HashMap<UopId, Info> = HashMap::new();
    for u in lst {
        let op = ctx.op_of(u);
        let srcs = ctx.srcs(u).to_vec();
        let addrspace = ctx.addrspace_of(u.clone());
        let derived = ctx.addrspace_derived(u.clone());
        let (numel, shape) = match ctx.shape(u) {
            Ok(Some(s)) if !s.is_empty() => {
                let prods: i64 = s.iter().map(|x| x.clone().as_i64_or_panic()).product();
                (prods, s.iter().map(|x| x.clone().as_i64_or_panic()).collect())
            }
            _ => (1, vec![]),
        };
        let volatile = match ctx.arg_of(u) {
            Arg::Param(p) => p.volatile,
            _ => false,
        };
        info.insert(u.clone(), Info { op, dt: ctx.dt_of(u), srcs, addrspace, derived, numel, shape, volatile });
    }
    let mut child_count: HashMap<UopId, usize> = HashMap::new();
    for u in lst {
        for s in &info[u].srcs {
            *child_count.entry(s.clone()).or_insert(0) += 1;
        }
    }
    // writable params: toposort of STORE src[0]s, gated below END (unused in output)
    let mut writable: std::collections::HashSet<UopId> = std::collections::HashSet::new();
    {
        let mut stack: Vec<UopId> = lst.iter().filter(|&u| info[u].op == Ops::Store).cloned().map(|u| (info[&u].srcs[0]).clone()).collect();
        let mut seen = std::collections::HashSet::new();
        while let Some(u) = stack.pop() {
            if !seen.insert(u.id()) {
                continue;
            }
            if info[&u].op == Ops::Param {
                writable.insert(u.clone());
            }
            if info[&u].op != Ops::End {
                stack.extend(info[&u].srcs.iter().cloned());
            }
        }
    }

    let mut r: HashMap<UopId, String> = HashMap::new();
    // CONST nodes are not named in `r` (they render inline). tinygrad's
    // `code_for_op[Ops.CONST]` renders them on demand, so any op that consumes a
    // CONST *directly* (e.g. `x != 0`, or a LOAD's fill) can inline it too.
    let mut consts: HashMap<UopId, String> = HashMap::new();
    for u in lst {
        if info[u].op == Ops::Const {
            consts.insert(u.clone(), const_text(ctx, info[u].dt, u.clone(), lang));
        }
    }
    // bufs: (name, param info) in first-encounter order
    let mut bufs: Vec<(String, Info, bool)> = Vec::new();
    let mut kernel: Vec<String> = Vec::new();
    let mut depth = 1usize;
    let mut counters: HashMap<&'static str, usize> = HashMap::new();
    let mut name = "test".to_string();

    for u in lst {
        let i = info[u].clone();
        if matches!(i.op, Ops::Noop | Ops::Group | Ops::Const) {
            continue;
        }
        if i.op == Ops::Stack && i.srcs.is_empty() {
            continue;
        }
        if i.op == Ops::After {
            r.insert(u.clone(), r[&i.srcs[0]].clone());
            continue;
        }
        if i.op == Ops::Sink {
            if let Arg::Data(d) = ctx.arg_of(u)
                && d.tag == "KernelInfo"
                    && let Some(Arg::Str(n)) = d.fields.iter().find_map(|(k, v)| if k == "name" { Some(v) } else { None }) {
                        name = to_function_name(n);
                    }
            continue;
        }
        if i.op == Ops::Param {
            let slot = match ctx.arg_of(u) {
                Arg::Param(p) => p.slot,
                _ => 0,
            };
            let shape: Vec<String> = i.shape.iter().map(|x| x.to_string()).collect();
            let nm = format!("data{slot}_{}", shape.join("_"));
            bufs.push((nm.clone(), i.clone(), writable.contains(u)));
            r.insert(u.clone(), nm);
            continue;
        }

        // naming
        let prefix: Option<&'static str>;
        match i.op {
            Ops::Special => {
                // r[u] = u.arg (the plain string, e.g. "lidx0")
                let nm = match ctx.arg_of(u) {
                    Arg::Str(s) => s.clone(),
                    other => panic!("TGERR: cstyle SPECIAL arg {other:?}"),
                };
                r.insert(u.clone(), nm);
                prefix = None;
            }
            Ops::Range => {
                let letter = match AxisType::range_axis_type(ctx, u.clone()) {
                    AxisType::Device => "d",
                    AxisType::Global => "g",
                    AxisType::Thread => "t",
                    AxisType::Local => "l",
                    AxisType::Warp => "w",
                    AxisType::Weak => "L",
                    AxisType::Loop => "L",
                    AxisType::Upcast => "u",
                    AxisType::GroupReduce => "G",
                    AxisType::Reduce => "R",
                    AxisType::Unroll => "r",
                    _ => "L",
                };
                r.insert(u.clone(), format!("{letter}idx{}", range_str(ctx.arg_of(u))));
                prefix = None;
            }
            _ => {
                let p = match i.op {
                    Ops::Wmma => "wmma",
                    Ops::Buffer => "buf",
                    Ops::Cast | Ops::Bitcast | Ops::Stack => "cast",
                    Ops::Index => "bidx",
                    Ops::Load => "val",
                    _ => "alu",
                };
                let c0 = counters.entry(p).or_insert(0);
                r.insert(u.clone(), format!("{p}{c0}"));
                prefix = Some(p);
            }
        }

        // render the op to a string
        let g = |m: &HashMap<UopId, String>, id: &UopId| -> String {
            match m.get(id).or_else(|| consts.get(id)) {
                Some(s) => s.clone(),
                None => {
                    let op = info.get(id).map(|i| format!("{:?}", i.op)).unwrap_or_else(|| "NOT-IN-LST".to_string());
                    let parent = info
                        .iter()
                        .find(|(_, inf)| inf.srcs.contains(id))
                        .map(|(_, inf)| format!("{:?}", inf.op))
                        .unwrap_or_else(|| "?".to_string());
                    panic!("TGERR: cstyle render missing name for node {} op {op}, referenced by {parent}", id.id())
                }
            }
        };
        let l: String = match i.op {
            Ops::Range => {
                if i.dt == dtype::VOID {
                    "for (;;) {".to_string()
                } else {
                    let nm = &r[u];
                    format!("for ({} {nm} = 0; {nm} < {}; {nm}++) {{", lang.render_dtype(i.dt, 1, None, false), g(&r, &i.srcs[0]))
                }
            }
            Ops::End => {
                if i.srcs.len() == 3 && info[&i.srcs[1]].op == Ops::Range && info[&i.srcs[2]].dt == dtype::BOOL {
                    format!("  if (!({})) {{ break; }}\n}}", g(&r, &i.srcs[2]))
                } else {
                    "}".to_string()
                }
            }
            Ops::If => format!("if ({}) {{", g(&r, &i.srcs[0])),
            Ops::Endif => "}".to_string(),
            Ops::Barrier => lang.barrier.to_string(),
            Ops::Cast => {
                let src0 = &info[&i.srcs[0]];
                if i.numel > 1 && i.addrspace == Some(AddrSpace::Reg) {
                    format!("__builtin_convertvector({}, {})", g(&r, &i.srcs[0]), lang.render_type(&i))
                } else if src0.op == Ops::Const {
                    const_text(ctx, i.dt, (i.srcs[0]).clone(), lang)
                } else {
                    format!("({})({})", lang.render_type(&i), g(&r, &i.srcs[0]))
                }
            }
            Ops::Bitcast => {
                if matches!(i.addrspace, Some(AddrSpace::Global) | Some(AddrSpace::Local)) {
                    format!("(({})({}))", lang.render_dtype(i.dt, 1, i.addrspace, false), g(&r, &i.srcs[0]))
                } else {
                    format!("__builtin_bit_cast({}, ({})({}))", lang.render_type(&i), lang.render_type(&info[&i.srcs[0]]), g(&r, &i.srcs[0]))
                }
            }
            Ops::Special => {
                // code_for_workitem[arg[0]](arg[-1]) + "; /* {src[0].render()} */"
                let (gfn, lfn, ifn) = lang.workitem.expect("TGERR: cstyle SPECIAL without code_for_workitem");
                let arg = ctx.arg_of(u).clone();
                let (kind, dim) = match &arg {
                    Arg::Str(s) => (s.as_bytes()[0], s[s.len() - 1..].parse::<usize>().unwrap_or(0)),
                    other => panic!("TGERR: cstyle SPECIAL arg {other:?}"),
                };
                let f = match kind {
                    b'g' => gfn,
                    b'l' => lfn,
                    _ => ifn,
                };
                // the extent comment: src[0].render() (expr, simplified)
                let ext = crate::renderer::expr::expr_render(ctx, (i.srcs[0]).clone());
                format!("{f}({dim}); /* {ext} */")
            }
            Ops::Index | Ops::Shrink => {
                let buf = &info[&i.srcs[0]];
                let idx = &info[&i.srcs[1]];
                let buf_s = g(&r, &i.srcs[0]);
                let idx_s = g(&r, &i.srcs[1]);
                if buf.derived == Some(AddrSpace::Alu) {
                    // lane access
                    let idx_is_const_lane = idx.op == Ops::Cast && idx.srcs.first().map(|c| info[c].op == Ops::Const).unwrap_or(false);
                    if !idx_is_const_lane {
                        format!("({buf_s})[{idx_s}]")
                    } else {
                        let lane = idx.srcs.first().and_then(|c| const_int(ctx, c.clone())).unwrap_or(0);
                        if buf.numel > lang.gep_arr_threshold {
                            format!("{buf_s}[{lane}]")
                        } else {
                            let ch = "xyzwabcd".chars().nth(lane as usize).unwrap_or('w');
                            format!("{buf_s}.{ch}")
                        }
                    }
                } else if idx.op == Ops::Add {
                    format!("({buf_s}+{})", strip_parens(&idx_s))
                } else {
                    format!("({buf_s}+{idx_s})")
                }
            }
            Ops::Stack => {
                let sty = lang.render_type(&i);
                let float4 = "(float4)".replace("float4", &sty);
                let inner: Vec<String> = i.srcs.iter().map(|s| g(&r, s)).collect();
                let (open, close) = lang.float4_style;
                format!("{float4}{open}{}{close}", inner.join(","))
            }
            Ops::Load => {
                if i.srcs.len() >= 3 {
                    format!(
                        "({}?{}:{})",
                        g(&r, &i.srcs[2]),
                        lang.render_access(&info, (i.srcs[0]).clone(), &g(&r, &i.srcs[0])),
                        g(&r, &i.srcs[1])
                    )
                } else {
                    format!("({})", lang.render_access(&info, (i.srcs[0]).clone(), &g(&r, &i.srcs[0])))
                }
            }
            Ops::Store => {
                let bidx = lang.render_access(&info, (i.srcs[0]).clone(), &g(&r, &i.srcs[0]));
                format!("{bidx} = {};", g(&r, &i.srcs[1]))
            }
            Ops::Wmma => {
                let nm = wmma_name(ctx, u.clone());
                format!("__{}({}, {}, {})", nm, g(&r, &i.srcs[0]), g(&r, &i.srcs[1]), g(&r, &i.srcs[2]))
            }
            Ops::Buffer => {
                // render_buffer (cstyle.py:169): the (local) storage declaration.
                // `_render_dtype(dtype, sz=1)` uses the default ALU addrspace, so the
                // only `__local ` is the one in `smem_prefix` (not a pointer).
                let prefix = if i.addrspace == Some(AddrSpace::Local) {
                    format!("{}{}", lang.smem_align, lang.smem_prefix)
                } else {
                    String::new()
                };
                format!("{prefix}{} {}[{}];", lang.render_dtype(i.dt, 1, None, false), r[u], i.numel)
            }
            _ if group::in_(group::ALU, i.op) => {
                let args: Vec<String> = i
                    .srcs
                    .iter()
                    .map(|v| {
                        if info[v].op == i.op && matches!(i.op, Ops::Add | Ops::Mul | Ops::Xor | Ops::Or | Ops::And) {
                            strip_parens(&g(&r, v)).to_string()
                        } else {
                            g(&r, v)
                        }
                    })
                    .collect();
                code_for_op(i.op, &args, lang.clang_builtins, i.dt)
            }
            other => panic!("TGERR: cstyle op {other:?} not ported"),
        };

        if matches!(i.op, Ops::Endif | Ops::End) {
            depth = depth.saturating_sub(1);
        }
        let is_const_cast = i.op == Ops::Cast && info[&i.srcs[0]].op == Ops::Const;
        let reg_load = i.op == Ops::Load && info[&i.srcs[0]].derived == Some(AddrSpace::Reg) && child_count[u] == 1;
        let ptr_cast = matches!(i.op, Ops::Cast | Ops::Bitcast) && matches!(i.addrspace, Some(AddrSpace::Global) | Some(AddrSpace::Local));
        let single_use_inline = matches!(i.op, Ops::Stack | Ops::Cast | Ops::Bitcast)
            || (group::in_(group::ALU, i.op) && i.op != Ops::Where);
        let inline = (i.op != Ops::Cast || i.numel == 1)
            && (is_const_cast
                || matches!(i.op, Ops::Index | Ops::Shrink | Ops::Customi)
                || reg_load
                || ptr_cast
                || (single_use_inline && child_count[u] == 1));
        if inline {
            r.insert(u.clone(), l);
        } else {
            let full = if !matches!(i.op, Ops::Range | Ops::Store | Ops::Buffer) && i.dt != dtype::VOID {
                let semi = if i.op == Ops::Special { "" } else { ";" };
                format!("{} {} = {}{semi}", lang.render_type(&i), r[u], l)
            } else {
                l
            };
            let indented: Vec<String> = full.split('\n').map(|line| format!("{}{}", "  ".repeat(depth), line)).collect();
            kernel.push(indented.join("\n"));
            if let Some(p) = prefix {
                *counters.entry(p).or_insert(0) += 1;
            }
        }
        if matches!(i.op, Ops::If | Ops::Range) {
            depth += 1;
        }
    }

    // defines: vector typedefs for (dtype, numel>1) uops with raw addrspace ALU/None
    let mut seen_dts: Vec<(DTypeId, i64)> = Vec::new();
    if lang.vector_defines {
        for u in lst {
            let i = &info[u];
            if matches!(i.addrspace, None | Some(AddrSpace::Alu)) && i.dt != dtype::VOID && !i.shape.is_empty() {
                let key = (i.dt, i.numel);
                if !seen_dts.contains(&key) {
                    seen_dts.push(key);
                }
            }
        }
    }
    let mut define_lines: Vec<String> = Vec::new();
    for (dt, count) in &seen_dts {
        if *count > 1 {
            let alignment = if *dt == dtype::BOOL { 1 } else { 2u64.pow((dt.itemsize() as u32 * *count as u32).ilog2()) };
            let vec = format!("{}{}", lang.type_name(*dt), count);
            define_lines.push(format!(
                "typedef {} {} __attribute__((aligned({}),ext_vector_type({})));",
                lang.type_name(*dt),
                vec,
                alignment,
                count
            ));
        }
    }
    // buftypes
    let buftypes: Vec<String> = bufs
        .iter()
        .map(|(nm, i, _writable)| {
            let as_ = i.addrspace;
            let vol = if i.volatile { "volatile " } else { "" };
            let (pre, suf) = if as_ == Some(AddrSpace::Alu) {
                (lang.var_prefix, lang.var_suffix)
            } else {
                ("", lang.buffer_suffix)
            };
            format!("{vol}{pre}{}{suf} {nm}", lang.render_dtype(i.dt, 1, as_, false))
        })
        .collect();
    // launch_bounds: product of local SPECIAL vmaxs — corpus: no SPECIALs → 1
    let prg = format!(
        "{} {}({}) {{\n{}\n}}",
        lang.kernel_typedef,
        name,
        buftypes.join(", "),
        kernel.join("\n")
    );
    if lang.vector_defines {
        let defines = define_lines.join("\n");
        format!("{defines}\n{prg}\n")
    } else {
        // OpenCLRenderer.render_kernel: optional fp16 pragma prefix, no defines,
        // and no trailing newline (python joins prefix + prg)
        let pragma = if lst.iter().any(|u| info[u].dt == dtype::FLOAT16) {
            "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
        } else {
            ""
        };
        format!("{pragma}{prg}")
    }
}

fn const_int(ctx: &mut UOpCtx, u: UopId) -> Option<i128> {
    match ctx.val(&u) {
        ConstVal::Int(v) => Some(v),
        _ => None,
    }
}

/// The base_rewrite const rules (in python match order) — cast dtype decides.
fn const_text(ctx: &mut UOpCtx, dt: DTypeId, c: UopId, lang: &CStyleLang) -> String {
    let val = ctx.val(&c);
    let vint = match val {
        ConstVal::Int(v) => Some(v),
        ConstVal::Bool(b) => Some(b as i128),
        ConstVal::Float(_f) => None,
        other => panic!("TGERR: cstyle const {other:?}"),
    };
    let vfloat = match val {
        ConstVal::Float(f) => f,
        ConstVal::Int(v) => v as f64,
        ConstVal::Bool(b) => b as i64 as f64,
        other => panic!("TGERR: cstyle const {other:?}"),
    };
    if dtype::is_float(dt) {
        if !vfloat.is_finite() {
            let inf = lang.infinity;
            let nan = lang.nan;
            let inner = if vfloat.is_nan() {
                nan.to_string()
            } else if vfloat > 0.0 {
                inf.to_string()
            } else {
                format!("-{inf}")
            };
            // tinygrad: f"({render_cast(x, ...)})" -> ((type)(inner))
            return format!("(({})({}))", lang.type_name(dt), inner);
        }
        if dt == dtype::FLOAT32 {
            return format!("{}f", py_float(vfloat));
        }
        // consts are rendered to a larger type and casted (fp8s/bf16/f16)
        if dtype::in_set(dtype::FP8S, dt) || dtype::in_set(&[dtype::BFLOAT16, dtype::FLOAT16], dt) {
            return format!("(({})({}f))", lang.type_name(dt), py_float(vfloat));
        }
        // f64 (and other native floats) render as the plain value
        return py_float(vfloat);
    }
    if dt == dtype::INT64 {
        return format!("{}l", vint.unwrap_or(0));
    }
    if dt == dtype::UINT64 {
        return format!("{}ul", trunc_int(vint.unwrap_or(0), dt));
    }
    if dt == dtype::UINT32 {
        return format!("{}u", trunc_int(vint.unwrap_or(0), dt));
    }
    if dt == dtype::BOOL {
        return if vint.unwrap_or(0) != 0 { "1".to_string() } else { "0".to_string() };
    }
    // tinygrad wraps these as f"({render_cast(x, ...)})" -> ((type)(val))
    if dt == dtype::UINT8 || dt == dtype::UINT16 {
        return format!("(({})({}u))", lang.type_name(dt), vint.unwrap_or(0));
    }
    if dt == dtype::INT8 || dt == dtype::INT16 {
        return format!("(({})({}))", lang.type_name(dt), vint.unwrap_or(0));
    }
    format!("{}", vint.unwrap_or(0))
}

/// python-style float repr: 2.0 prints "2.0" in python, "2" in Rust.
fn py_float(v: f64) -> String {
    let s = format!("{v}");
    if s.contains('.') || s.contains('e') || s.contains("inf") || s.contains("nan") {
        s
    } else {
        format!("{s}.0")
    }
}

/// int truncation to the dtype's range.
fn trunc_int(v: i128, dt: DTypeId) -> i128 {
    match (dtype::dtype_min(dt), dtype::dtype_max(dt)) {
        (ConstVal::Int(l), ConstVal::Int(h)) => v.clamp(l, h),
        _ => v,
    }
}

/// `_wmma_name`.
fn wmma_name(ctx: &mut UOpCtx, u: UopId) -> String {
    let arg = ctx.arg_of(&u).clone();
    match arg {
        Arg::Seq { tuple: _, items } => {
            let dims: Vec<String> = items
                .iter()
                .filter_map(|it| match it {
                    Arg::Const(ConstVal::Int(v)) => Some(v.to_string()),
                    _ => None,
                })
                .collect();
            format!("WMMA_{}", dims.join("_")).replace(' ', "_")
        }
        _ => "WMMA".to_string(),
    }
}

/// `CStyleLanguage.code_for_op` (clang subset).
fn code_for_op(op: Ops, a: &[String], clang: bool, dt: crate::dtype::DTypeId) -> String {
    match op {
        Ops::Sqrt => {
            if clang {
                if dt == crate::dtype::FLOAT64 { format!("__builtin_sqrt({})", a[0]) } else { format!("__builtin_sqrtf({})", a[0]) }
            } else {
                format!("sqrt({})", a[0])
            }
        }
        Ops::Exp2 => format!("exp2({})", a[0]),
        Ops::Log2 => format!("log2({})", a[0]),
        Ops::Sin => format!("sin({})", a[0]),
        Ops::Trunc => {
            if clang {
                if dt == crate::dtype::FLOAT64 { format!("__builtin_trunc({})", a[0]) } else { format!("__builtin_truncf({})", a[0]) }
            } else {
                format!("trunc({})", a[0])
            }
        }
        Ops::Reciprocal => format!("(1/{})", a[0]),
        Ops::Neg => format!("-{}", a[0]),
        Ops::And => format!("({}&{})", a[0], a[1]),
        Ops::Xor => format!("({}^{})", a[0], a[1]),
        Ops::Or => format!("({}|{})", a[0], a[1]),
        Ops::Add => format!("({}+{})", a[0], a[1]),
        Ops::Sub => format!("({}-{})", a[0], a[1]),
        Ops::Mul => format!("({}*{})", a[0], a[1]),
        Ops::Cmod => format!("({}%{})", a[0], a[1]),
        Ops::Cdiv => format!("({}/{})", a[0], a[1]),
        Ops::Cmpne => format!("({}!={})", a[0], a[1]),
        Ops::Shr => format!("({}>>{})", a[0], a[1]),
        Ops::Shl => format!("({}<<{})", a[0], a[1]),
        Ops::Cmplt => format!("({}<{})", a[0], a[1]),
        Ops::Where => format!("({}?{}:{})", a[0], a[1], a[2]),
        Ops::Cmpeq => format!("({}=={})", a[0], a[1]),
        Ops::Fdiv => format!("({}/{})", a[0], a[1]),
        other => panic!("TGERR: cstyle ALU {other:?} not ported"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Render a scalar const through `const_text` (the `base_rewrite` const rules).
    fn ctext(v: ConstVal, dt: DTypeId, lang: &CStyleLang) -> String {
        let mut ctx = UOpCtx::new();
        let c = ctx.const_py(v, Some(dt));
        const_text(&mut ctx, dt, c, lang)
    }

    /// The uint8/int8 const casts must keep the parens around the type name:
    /// tinygrad wraps `render_cast` in another pair (`((uchar)(0u))`), and the
    /// old `(uchar(0u))` parsed as a function call and failed to build on CL.
    #[test]
    fn const_cast_rendering_matches_tinygrad() {
        let cl = CStyleLang::opencl();
        assert_eq!(ctext(ConstVal::Int(0), dtype::UINT8, &cl), "((uchar)(0u))");
        assert_eq!(ctext(ConstVal::Int(3), dtype::INT8, &cl), "((char)(3))");
        assert_eq!(ctext(ConstVal::Int(3), dtype::UINT16, &cl), "((ushort)(3u))");
        assert_eq!(ctext(ConstVal::Int(3), dtype::INT16, &cl), "((short)(3))");
        assert_eq!(ctext(ConstVal::Int(3), dtype::UINT32, &cl), "3u");
        assert_eq!(ctext(ConstVal::Int(3), dtype::INT32, &cl), "3");
        assert_eq!(ctext(ConstVal::Bool(true), dtype::BOOL, &cl), "1");
        assert_eq!(ctext(ConstVal::Float(2.0), dtype::FLOAT32, &cl), "2.0f");
        // OpenCL has no half type_map entry -> tinygrad renders `half`; Clang maps it.
        assert_eq!(ctext(ConstVal::Float(2.0), dtype::FLOAT16, &cl), "((half)(2.0f))");
        let clang = CStyleLang::clang();
        assert_eq!(ctext(ConstVal::Float(2.0), dtype::FLOAT16, &clang), "((__fp16)(2.0f))");
    }

    /// Non-finite float consts are rendered through a cast of the lang's
    /// `nan`/`infinity` (tinygrad: `((float)(NAN))`, not a bare `(NAN)`).
    #[test]
    fn const_nonfinite_rendering_matches_tinygrad() {
        let cl = CStyleLang::opencl();
        assert_eq!(ctext(ConstVal::Float(f64::NAN), dtype::FLOAT32, &cl), "((float)(NAN))");
        assert_eq!(ctext(ConstVal::Float(f64::INFINITY), dtype::FLOAT32, &cl), "((float)(INFINITY))");
        assert_eq!(ctext(ConstVal::Float(f64::NEG_INFINITY), dtype::FLOAT32, &cl), "((float)(-INFINITY))");
        let clang = CStyleLang::clang();
        assert_eq!(ctext(ConstVal::Float(f64::NAN), dtype::FLOAT32, &clang), "((float)(__builtin_nanf(\"\")))");
    }
}
