//! `renderer/cstyle.py` port (M5) — the `ClangRenderer` subset for the corpus.
//!
//! Renders linearized uops to C text: naming/SSA inlining decisions, const
//! rules, index/load/store, ALU, ranges, and the kernel wrapper + defines.

use crate::dtype::{self, AddrSpace, ConstVal, DTypeId};
use crate::uop::UopId;
use crate::uop::arg::Arg;
use crate::uop::ctx::UOpCtx;
use crate::uop::ops::{Ops, group};
use crate::uop::range::AxisType;
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
    /// `u._shape is not None` — scalars have an empty-but-present shape.
    has_shape: bool,
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
    /// OpenCL stores bf16 as `ushort`, so bf16 constants render as their
    /// 16-bit bit pattern (`{bits}u`) instead of a float cast (cstyle.py:325).
    bf16_const_bits: bool,
    /// ALU `BITCAST` rendering: clang uses `__builtin_bit_cast`, OpenCL uses
    /// `as_<type>(...)` (cstyle.py:41 vs the OpenCL `string_rewrite` at :322).
    bitcast_builtin: bool,
    /// CUDA (`CUDARenderer`, cstyle.py:397) flavor: `nv_bfloat16` type map,
    /// `__launch_bounds__` typedef, native bf16 consts, `tg_bitcast`, the
    /// `_nms` vector prefix, and the CUDA `render_kernel` prefix block.
    cuda: bool,
    /// `float4` (cstyle.py:129) — the Stack literal base; `float4` is replaced by
    /// the vector type name (`(float4)` clang/opencl, `make_float4` CUDA).
    float4: &'static str,
    /// Rust target: emit a `#[unsafe(no_mangle)] extern "C"` function instead of
    /// C. The kernel takes the fixed `(bufs, vals, gidx)` ABI and extracts typed
    /// pointers in its prologue (see [`render_rust`]).
    rust: bool,
    /// CPU C target: the fixed `(bufs, vals, gidx)` ABI, `gidx` chunking, and
    /// portable vector types. Works with clang and gcc (see [`render_cpu_c`]).
    cpu: bool,
    /// Metal Shading Language flavor (`MetalRenderer`, cstyle.py:251): `kernel
    /// void` with `gid`/`lid` thread attributes, `device`/`threadgroup`
    /// address spaces, `as_type<>` bitcasts, and `constant int&` scalar args.
    metal: bool,
    /// Whether a vector type name is a C typedef identifier (`unsigned_char2`)
    /// rather than a language spelling (`unsigned char2`). The CPU/clang
    /// vector typedefs need the former; Metal needs the latter.
    vector_underscore: bool,
    /// Emit an explicit truncating float→int cast (`(long)` first, clamping
    /// negatives for unsigned destinations) instead of relying on the GPU
    /// language's `float -> small_int` cast, which is implementation-defined
    /// and saturates on Apple OpenCL/Metal. The interpreter/CPU semantics are
    /// truncation + wrap (`ctypes.c_<n>`).
    trunc_float_to_int: bool,
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
            bf16_const_bits: false,
            bitcast_builtin: true,
            cuda: false,
            float4: "(float4)",
            rust: false,
            cpu: false,
            metal: false,
            vector_underscore: true,
            trunc_float_to_int: false,
        }
    }

    /// `RUST` language options — emit Rust source with a fixed C ABI.
    ///
    /// The generated function takes `(bufs, vals, gidx)` so the host does not
    /// need a per-kernel function-pointer type; see [`render_rust`].
    pub fn rust() -> CStyleLang {
        CStyleLang {
            clang_builtins: false,
            buffer_prefix: "",
            buffer_suffix: "",
            var_prefix: "",
            var_suffix: "",
            smem_align: "",
            smem_prefix: "",
            smem_prefix_for_cast: false,
            gep_arr_threshold: 0,
            barrier: "",
            infinity: "f32::INFINITY",
            nan: "f32::NAN",
            type_map: &[],
            workitem: None,
            vector_defines: false,
            kernel_typedef: "",
            float4_style: ("[", "]"),
            bf16_const_bits: false,
            bitcast_builtin: false,
            cuda: false,
            float4: "",
            rust: true,
            cpu: false,
            metal: false,
            vector_underscore: false,
            trunc_float_to_int: false,
        }
    }

    /// `CC` language options — portable C with the fixed CPU ABI.
    ///
    /// The kernel takes `(uint8_t** bufs, int64_t* vals, int64_t* gidx)` and
    /// extracts typed `restrict` pointers in its prologue. The `gidx[0..2]`
    /// slots are the `[start, end)` chunk of the first global dimension.
    pub fn cpu() -> CStyleLang {
        CStyleLang {
            clang_builtins: true,
            buffer_prefix: "",
            buffer_suffix: " restrict",
            var_prefix: "",
            var_suffix: "",
            smem_align: "",
            smem_prefix: "",
            smem_prefix_for_cast: false,
            gep_arr_threshold: 0,
            barrier: "",
            infinity: "__builtin_inff()",
            nan: "__builtin_nanf(\"\")",
            type_map: &[
                (dtype::BOOL, "_Bool"),
                (dtype::FLOAT16, "_Float16"),
                (dtype::BFLOAT16, "unsigned short"),
            ],
            workitem: None,
            vector_defines: true,
            kernel_typedef: "",
            float4_style: ("{", "}"),
            bf16_const_bits: true,
            bitcast_builtin: false,
            cuda: false,
            float4: "(float4)",
            rust: false,
            cpu: true,
            metal: false,
            vector_underscore: true,
            trunc_float_to_int: false,
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
            bf16_const_bits: true,
            bitcast_builtin: false,
            cuda: false,
            float4: "(float4)",
            rust: false,
            cpu: false,
            metal: false,
            vector_underscore: false,
            trunc_float_to_int: true,
        }
    }

    /// `CUDARenderer` language options (cstyle.py:397).
    pub fn cuda() -> CStyleLang {
        CStyleLang {
            clang_builtins: false,
            buffer_prefix: "",
            buffer_suffix: "",
            var_prefix: "const ",
            var_suffix: "",
            smem_align: "",
            smem_prefix: "__shared__ __align__(16) ",
            smem_prefix_for_cast: false,
            gep_arr_threshold: 8,
            barrier: "__syncthreads();",
            infinity: "INFINITY",
            nan: "NAN",
            type_map: &[
                (dtype::UINT32, "uint"),
                (dtype::BFLOAT16, "nv_bfloat16"),
                (dtype::FP8E4M3, "__nv_fp8_e4m3"),
                (dtype::FP8E5M2, "__nv_fp8_e5m2"),
            ],
            workitem: Some(("blockIdx", "threadIdx", "global")),
            vector_defines: false,
            kernel_typedef: "extern \"C\" __global__ void __launch_bounds__({launch_bounds})",
            float4_style: ("(", ")"),
            bf16_const_bits: false,
            bitcast_builtin: false,
            cuda: true,
            float4: "make_float4",
            rust: false,
            cpu: false,
            metal: false,
            vector_underscore: false,
            trunc_float_to_int: false,
        }
    }

    /// `MetalRenderer` language options (cstyle.py:251).
    ///
    /// Metal uses `kernel void` with implicit `gid`/`lid` thread attributes,
    /// `device` global memory, `threadgroup` local memory, `as_type<>` for
    /// bitcasts, and passes scalar `int` params as `constant int&` (the
    /// runtime sets them with `setBytes`).
    pub fn metal() -> CStyleLang {
        CStyleLang {
            clang_builtins: false,
            buffer_prefix: "device ",
            buffer_suffix: "",
            var_prefix: "const ",
            var_suffix: "",
            smem_align: "",
            smem_prefix: "threadgroup ",
            smem_prefix_for_cast: true,
            gep_arr_threshold: 0,
            barrier: "threadgroup_barrier(mem_flags::mem_threadgroup);",
            infinity: "INFINITY",
            nan: "NAN",
            type_map: &[
                (dtype::INT8, "char"),
                (dtype::UINT8, "uchar"),
                (dtype::UINT16, "ushort"),
                (dtype::UINT32, "uint"),
                (dtype::UINT64, "ulong"),
                (dtype::BFLOAT16, "bfloat"),
            ],
            workitem: Some(("gid", "lid", "gid")),
            vector_defines: false,
            kernel_typedef: "kernel void",
            float4_style: ("(", ")"),
            bf16_const_bits: false,
            bitcast_builtin: false,
            cuda: false,
            float4: "float4",
            rust: false,
            cpu: false,
            metal: true,
            vector_underscore: false,
            trunc_float_to_int: true,
        }
    }

    /// `_render_dtype` — type_map lookup (space-normalized), prefix/suffix by
    /// addrspace, vector size suffix.
    fn render_dtype(
        &self,
        dt: DTypeId,
        sz: i64,
        addrspace: Option<AddrSpace>,
        override_ptr: bool,
    ) -> String {
        if self.rust {
            return rust_dtype(dt, sz, addrspace, override_ptr);
        }
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
        let suffix = if matches!(as_, AddrSpace::Local | AddrSpace::Global) || override_ptr {
            "*"
        } else {
            ""
        };
        let base = if sz > 1 && self.vector_underscore {
            self.type_name(dt).replace(' ', "_")
        } else {
            self.type_name(dt)
        };
        if sz > 1 {
            format!("{prefix}{base}{sz}{suffix}")
        } else {
            format!("{prefix}{base}{suffix}")
        }
    }

    /// `type_map.get(dtype, dtype.name)` — the space normalization happens in
    /// `render_dtype` only for `sz > 1` (cstyle.py:184).
    fn type_name(&self, dt: DTypeId) -> String {
        for (d, name) in self.type_map {
            if *d == dt {
                return name.to_string();
            }
        }
        dt.name().to_string()
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
        if self.rust {
            // A BUFFER/ALU/REG source is already an array element (a place or
            // value); only global/local pointers need a dereference.
            let is_array = if matches!(i.op, Ops::Index | Ops::Shrink) {
                let b = &info[&i.srcs[0]];
                b.op == Ops::Buffer
                    || matches!(b.addrspace, Some(AddrSpace::Alu) | Some(AddrSpace::Reg))
                    || matches!(b.derived, Some(AddrSpace::Alu) | Some(AddrSpace::Reg))
            } else {
                matches!(as_, Some(AddrSpace::Alu) | Some(AddrSpace::Reg))
            };
            return if is_array {
                bidx.to_string()
            } else if i.numel > 1 || i.dt != src_dt {
                format!("*(({bidx}) as {})", rust_dtype(i.dt, i.numel, as_, true))
            } else {
                format!("*{bidx}")
            };
        }
        if i.numel > 1 || i.dt != src_dt {
            format!(
                "*(({})({}))",
                self.render_dtype(i.dt, i.numel, as_, true),
                bidx
            )
        } else {
            format!("*{bidx}")
        }
    }
}

/// Rust scalar type for a dtype.
///
/// f16/bf16/fp8 are storage-only here (`u16`/`u8`); the backend should fall
/// back for models that need native half arithmetic.
fn rust_type_name(dt: DTypeId) -> &'static str {
    if dt == dtype::BOOL {
        "bool"
    } else if dt == dtype::WEAKINT {
        "i32"
    } else if dt == dtype::WEAKFLOAT {
        "f32"
    } else if dt == dtype::INT8 {
        "i8"
    } else if dt == dtype::INT16 {
        "i16"
    } else if dt == dtype::INT32 {
        "i32"
    } else if dt == dtype::INT64 {
        "i64"
    } else if dt == dtype::UINT8 {
        "u8"
    } else if dt == dtype::UINT16 {
        "u16"
    } else if dt == dtype::UINT32 {
        "u32"
    } else if dt == dtype::UINT64 {
        "u64"
    } else if dt == dtype::FLOAT32 {
        "f32"
    } else if dt == dtype::FLOAT64 {
        "f64"
    } else if dt == dtype::FLOAT16 || dt == dtype::BFLOAT16 {
        "u16"
    } else if dtype::in_set(dtype::FP8S, dt) {
        "u8"
    } else if dt == dtype::VOID {
        "()"
    } else {
        panic!("TGERR: rust renderer has no type for {dt:?}")
    }
}

/// Rust type for a `(dtype, width, addrspace)` as the renderer sees it.
///
/// A vector is `[T; N]`; a global/local pointer is `*mut T` (or `*mut [T; N]`
/// for a vector access). `override_ptr` forces a pointer for a cast site.
fn rust_dtype(dt: DTypeId, sz: i64, addrspace: Option<AddrSpace>, override_ptr: bool) -> String {
    let as_ = addrspace.unwrap_or(AddrSpace::Alu);
    let base = rust_type_name(dt);
    let ty = if sz > 1 {
        format!("[{base}; {sz}]")
    } else {
        base.to_string()
    };
    let is_ptr = matches!(as_, AddrSpace::Global | AddrSpace::Local) || override_ptr;
    if is_ptr && dt != dtype::VOID {
        format!("*mut {ty}")
    } else {
        ty
    }
}

/// Rust zero literal for a buffer initialization.
fn rust_zero(dt: DTypeId) -> String {
    if dt == dtype::BOOL {
        "false".to_string()
    } else if dt == dtype::FLOAT32 {
        "0.0f32".to_string()
    } else if dt == dtype::FLOAT64 {
        "0.0f64".to_string()
    } else {
        format!("0{}", rust_type_name(dt))
    }
}

/// A scalar Rust cast expression.
fn rust_cast_scalar(src: &str, src_dt: DTypeId, dst_dt: DTypeId) -> String {
    if src_dt == dtype::BFLOAT16 && dst_dt == dtype::FLOAT32 {
        // bf16 is `u16` storage: widen by placing the 16 bits in the high half.
        return format!("f32::from_bits((({src}) as u32) << 16)");
    }
    if dst_dt == dtype::BOOL {
        if src_dt == dtype::BOOL {
            src.to_string()
        } else {
            format!("({src} != 0)")
        }
    } else if src_dt == dtype::BOOL {
        if dtype::is_float(dst_dt) {
            format!("(({src} as u8) as {})", rust_type_name(dst_dt))
        } else {
            format!("({src} as {})", rust_type_name(dst_dt))
        }
    } else if dtype::is_float(src_dt) && !dtype::is_float(dst_dt) {
        // tinygrad/`interp::truncate`: unsigned destinations clamp negatives
        // to zero first, then the integer wraps to the destination width.
        let dst = rust_type_name(dst_dt);
        if dtype::in_set(dtype::UINTS, dst_dt) {
            format!(
                "{{ let __autorust_v = ({src}); ((if __autorust_v < 0.0 {{ 0.0 }} else {{ __autorust_v }}) as i64) as {dst} }}"
            )
        } else {
            format!("{{ let __autorust_v = ({src}); (__autorust_v as i64) as {dst} }}")
        }
    } else {
        format!("({src} as {})", rust_type_name(dst_dt))
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

/// `RUST` renderer — Rust source with a fixed `(bufs, vals, gidx)` C ABI.
pub fn render_rust(ctx: &mut UOpCtx, lst: &[UopId]) -> String {
    render_cstyle(ctx, lst, &CStyleLang::rust())
}

/// `CC` renderer — portable C with the fixed `(bufs, vals, gidx)` ABI.
pub fn render_cpu_c(ctx: &mut UOpCtx, lst: &[UopId]) -> String {
    render_cstyle(ctx, lst, &CStyleLang::cpu())
}

/// `OpenCLRenderer.render(lst)`.
pub fn render_opencl(ctx: &mut UOpCtx, lst: &[UopId]) -> String {
    render_cstyle(ctx, lst, &CStyleLang::opencl())
}

/// `CUDARenderer.render(lst)`.
pub fn render_cuda(ctx: &mut UOpCtx, lst: &[UopId]) -> String {
    render_cstyle(ctx, lst, &CStyleLang::cuda())
}

/// `MetalRenderer.render(lst)`.
pub fn render_metal(ctx: &mut UOpCtx, lst: &[UopId]) -> String {
    render_cstyle(ctx, lst, &CStyleLang::metal())
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
                (
                    prods,
                    s.iter().map(|x| x.clone().as_i64_or_panic()).collect(),
                )
            }
            _ => (1, vec![]),
        };
        let has_shape = matches!(ctx.shape(u), Ok(Some(_)));
        let volatile = match ctx.arg_of(u) {
            Arg::Param(p) => p.volatile,
            _ => false,
        };
        info.insert(
            u.clone(),
            Info {
                op,
                dt: ctx.dt_of(u),
                srcs,
                addrspace,
                derived,
                numel,
                shape,
                has_shape,
                volatile,
            },
        );
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
        let mut stack: Vec<UopId> = lst
            .iter()
            .filter(|&u| info[u].op == Ops::Store)
            .map(|u| info[u].srcs[0].clone())
            .collect();
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
    // Rust only: scalar ALU params come from the `vals` array instead.
    let mut vals: Vec<(String, Info)> = Vec::new();
    // Rust only: SPECIAL global ids the kernel reads from the `gidx` array.
    let mut gidx_params: Vec<(String, DTypeId)> = Vec::new();
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
                && let Some(Arg::Str(n)) = d
                    .fields
                    .iter()
                    .find_map(|(k, v)| if k == "name" { Some(v) } else { None })
            {
                name = to_function_name(n);
            }
            continue;
        }
        if i.op == Ops::Param {
            let slot = match ctx.arg_of(u) {
                Arg::Param(p) => p.slot,
                _ => 0,
            };
            let is_alu = match ctx.arg_of(u) {
                Arg::Param(p) => p.addrspace == Some(AddrSpace::Alu),
                _ => false,
            };
            let shape: Vec<String> = i.shape.iter().map(|x| x.to_string()).collect();
            let nm = format!("data{slot}_{}", shape.join("_"));
            if (lang.rust || lang.cpu) && is_alu {
                vals.push((nm.clone(), i.clone()));
            } else {
                bufs.push((nm.clone(), i.clone(), writable.contains(u)));
            }
            r.insert(u.clone(), nm);
            continue;
        }

        // naming
        let prefix: Option<&'static str> = match i.op {
            Ops::Special => {
                // r[u] = u.arg (the plain string, e.g. "lidx0")
                let nm = match ctx.arg_of(u) {
                    Arg::Str(s) => s.clone(),
                    other => panic!("TGERR: cstyle SPECIAL arg {other:?}"),
                };
                if lang.rust || lang.cpu {
                    if !nm.starts_with('g') {
                        panic!("TGERR: CPU renderer supports only global SPECIAL, got {nm:?}");
                    }
                    if !gidx_params.iter().any(|(n, _)| n == &nm) {
                        gidx_params.push((nm.clone(), i.dt));
                    }
                }
                r.insert(u.clone(), nm);
                None
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
                r.insert(
                    u.clone(),
                    format!("{letter}idx{}", range_str(ctx.arg_of(u))),
                );
                None
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
                Some(p)
            }
        };

        // render the op to a string
        let g = |m: &HashMap<UopId, String>, id: &UopId| -> String {
            match m.get(id).or_else(|| consts.get(id)) {
                Some(s) => s.clone(),
                None => {
                    let op = info
                        .get(id)
                        .map(|i| format!("{:?}", i.op))
                        .unwrap_or_else(|| "NOT-IN-LST".to_string());
                    let parent = info
                        .iter()
                        .find(|(_, inf)| inf.srcs.contains(id))
                        .map(|(_, inf)| format!("{:?}", inf.op))
                        .unwrap_or_else(|| "?".to_string());
                    panic!(
                        "TGERR: cstyle render missing name for node {} op {op}, referenced by {parent}",
                        id.id()
                    )
                }
            }
        };
        let l: String = match i.op {
            Ops::Range => {
                if lang.rust {
                    if i.dt == dtype::VOID {
                        "loop {".to_string()
                    } else {
                        let nm = &r[u];
                        format!("for {nm} in 0..{} {{", g(&r, &i.srcs[0]))
                    }
                } else if i.dt == dtype::VOID {
                    "for (;;) {".to_string()
                } else {
                    let nm = &r[u];
                    format!(
                        "for ({} {nm} = 0; {nm} < {}; {nm}++) {{",
                        lang.render_dtype(i.dt, 1, None, false),
                        g(&r, &i.srcs[0])
                    )
                }
            }
            Ops::End => {
                if i.srcs.len() == 3
                    && info[&i.srcs[1]].op == Ops::Range
                    && (info[&i.srcs[2]].dt == dtype::BOOL || lang.rust)
                {
                    let cond = g(&r, &i.srcs[2]);
                    if lang.rust {
                        if info[&i.srcs[2]].dt == dtype::BOOL {
                            format!("  if !({cond}) {{ break; }}\n}}")
                        } else {
                            format!("  if ({cond}) == 0 {{ break; }}\n}}")
                        }
                    } else {
                        format!("  if (!({cond})) {{ break; }}\n}}")
                    }
                } else {
                    "}".to_string()
                }
            }
            Ops::If => {
                if lang.rust {
                    let cond = g(&r, &i.srcs[0]);
                    if info[&i.srcs[0]].dt == dtype::BOOL {
                        format!("if {cond} {{")
                    } else {
                        format!("if ({cond}) != 0 {{")
                    }
                } else {
                    format!("if ({}) {{", g(&r, &i.srcs[0]))
                }
            }
            Ops::Endif => "}".to_string(),
            Ops::Barrier => lang.barrier.to_string(),
            Ops::Cast => {
                let src0 = &info[&i.srcs[0]];
                if lang.rust {
                    let src = g(&r, &i.srcs[0]);
                    if src0.op == Ops::Const {
                        let lit = const_text(ctx, i.dt, (i.srcs[0]).clone(), lang);
                        if i.numel > 1 {
                            format!("[{lit}; {}]", i.numel)
                        } else {
                            lit
                        }
                    } else if matches!(
                        i.addrspace,
                        Some(AddrSpace::Global) | Some(AddrSpace::Local)
                    ) {
                        format!(
                            "({src} as {})",
                            rust_dtype(i.dt, i.numel, i.addrspace, true)
                        )
                    } else if i.numel > 1 {
                        if src0.numel > 1 {
                            format!("({src}).map(|v| {})", rust_cast_scalar("v", src0.dt, i.dt))
                        } else {
                            format!(
                                "core::array::from_fn(|_| {})",
                                rust_cast_scalar(&src, src0.dt, i.dt)
                            )
                        }
                    } else {
                        rust_cast_scalar(&src, src0.dt, i.dt)
                    }
                } else if i.numel > 1 && i.addrspace == Some(AddrSpace::Reg) {
                    format!(
                        "__builtin_convertvector({}, {})",
                        g(&r, &i.srcs[0]),
                        lang.render_type(&i)
                    )
                } else if lang.cpu
                    && dtype::is_float(src0.dt)
                    && !dtype::is_float(i.dt)
                    && i.dt != dtype::BOOL
                    && dtype::in_set(dtype::UINTS, i.dt)
                {
                    // tinygrad truncate: unsigned destinations clamp negatives
                    // to zero before the C cast (which would be UB otherwise).
                    format!("TG_F2U({}, {})", lang.render_type(&i), g(&r, &i.srcs[0]))
                } else if lang.trunc_float_to_int
                    && i.numel == 1
                    && dtype::is_float(src0.dt)
                    && !dtype::is_float(i.dt)
                    && i.dt != dtype::BOOL
                {
                    // The language float->small-int cast is implementation-
                    // defined (Apple GPUs saturate). tinygrad's interpreter
                    // truncates toward zero and wraps, so go through `long`
                    // explicitly and clamp negatives for unsigned targets.
                    let dst = lang.render_type(&i);
                    let src = g(&r, &i.srcs[0]);
                    if dtype::in_set(dtype::UINTS, i.dt) {
                        format!("(({dst})(long)((({src}) < 0) ? 0 : ({src})))")
                    } else {
                        format!("(({dst})(long)({src}))")
                    }
                } else if src0.op == Ops::Const {
                    const_text(ctx, i.dt, (i.srcs[0]).clone(), lang)
                } else {
                    // base_rewrite CAST rule: `f"({render_cast(x, val)})"` and
                    // `render_cast` is `(type)(val)` — so double parens.
                    format!("(({})({}))", lang.render_type(&i), g(&r, &i.srcs[0]))
                }
            }
            Ops::Bitcast => {
                if lang.rust {
                    let src = g(&r, &i.srcs[0]);
                    if matches!(
                        i.addrspace,
                        Some(AddrSpace::Global) | Some(AddrSpace::Local)
                    ) {
                        format!(
                            "({src} as {})",
                            rust_dtype(i.dt, i.numel, i.addrspace, true)
                        )
                    } else {
                        let st =
                            rust_dtype(info[&i.srcs[0]].dt, info[&i.srcs[0]].numel, None, false);
                        let dt = rust_dtype(i.dt, i.numel, None, false);
                        format!("core::mem::transmute::<{st}, {dt}>({src})")
                    }
                } else if lang.cpu {
                    let src = g(&r, &i.srcs[0]);
                    if matches!(
                        i.addrspace,
                        Some(AddrSpace::Global) | Some(AddrSpace::Local)
                    ) {
                        format!(
                            "(({})({}))",
                            lang.render_dtype(i.dt, 1, i.addrspace, false),
                            src
                        )
                    } else {
                        format!(
                            "TG_BITCAST({}, ({})({}))",
                            lang.render_type(&i),
                            lang.render_type(&info[&i.srcs[0]]),
                            src
                        )
                    }
                } else if matches!(
                    i.addrspace,
                    Some(AddrSpace::Global) | Some(AddrSpace::Local)
                ) {
                    format!(
                        "(({})({}))",
                        lang.render_dtype(i.dt, 1, i.addrspace, false),
                        g(&r, &i.srcs[0])
                    )
                } else if lang.bitcast_builtin {
                    format!(
                        "__builtin_bit_cast({}, ({})({}))",
                        lang.render_type(&i),
                        lang.render_type(&info[&i.srcs[0]]),
                        g(&r, &i.srcs[0])
                    )
                } else if lang.metal {
                    // MetalRenderer.string_rewrite (cstyle.py:282):
                    // `as_type<{dtype}>({src})`.
                    format!("as_type<{}>({})", lang.render_type(&i), g(&r, &i.srcs[0]))
                } else if lang.cuda {
                    // CUDARenderer string_rewrite (cstyle.py:434): tg_bitcast<dst>((src)(x))
                    let dst = lang.render_dtype(i.dt, 1, None, false);
                    let src = lang.render_dtype(info[&i.srcs[0]].dt, 1, None, false);
                    format!("tg_bitcast<{dst}>(({src})({}))", g(&r, &i.srcs[0]))
                } else {
                    // OpenCL (cstyle.py:322): as_<type>((<src_type>)(src))
                    let dst = lang.render_dtype(i.dt, 1, None, false);
                    let src = lang.render_dtype(info[&i.srcs[0]].dt, 1, None, false);
                    format!("as_{dst}(({src})({}))", g(&r, &i.srcs[0]))
                }
            }
            Ops::Special => {
                if lang.rust || lang.cpu {
                    r[u].clone()
                } else {
                    // code_for_workitem[arg[0]](arg[-1]) + "; /* {src[0].render()} */"
                    let (gfn, lfn, ifn) = lang
                        .workitem
                        .expect("TGERR: cstyle SPECIAL without code_for_workitem");
                    let arg = ctx.arg_of(u).clone();
                    let (kind, dim) = match &arg {
                        Arg::Str(s) => (
                            s.as_bytes()[0],
                            s[s.len() - 1..].parse::<usize>().unwrap_or(0),
                        ),
                        other => panic!("TGERR: cstyle SPECIAL arg {other:?}"),
                    };
                    let f = match kind {
                        b'g' => gfn,
                        b'l' => lfn,
                        _ => ifn,
                    };
                    // the extent comment: src[0].render() (expr, simplified)
                    let ext = crate::renderer::expr::expr_render(ctx, (i.srcs[0]).clone());
                    if lang.cuda {
                        // CUDARenderer.code_for_workitem (cstyle.py:407): `blockIdx.x` etc.
                        let ch = match dim {
                            0 => 'x',
                            1 => 'y',
                            _ => 'z',
                        };
                        let s = match kind {
                            b'g' => format!("blockIdx.{ch}"),
                            b'l' => format!("threadIdx.{ch}"),
                            _ => format!("(blockIdx.{ch}*blockDim.{ch}+threadIdx.{ch})"),
                        };
                        format!("{s}; /* {ext} */")
                    } else if lang.metal {
                        // MetalRenderer.code_for_workitem (cstyle.py:266): `gid.x` / `lid.x`.
                        let ch = match dim {
                            0 => 'x',
                            1 => 'y',
                            _ => 'z',
                        };
                        let s = match kind {
                            b'g' => format!("gid.{ch}"),
                            b'l' => format!("lid.{ch}"),
                            other => panic!("TGERR: Metal has no workitem for {:?}", other as char),
                        };
                        format!("{s}; /* {ext} */")
                    } else {
                        format!("{f}({dim}); /* {ext} */")
                    }
                }
            }
            Ops::Index | Ops::Shrink => {
                let buf = &info[&i.srcs[0]];
                let idx = &info[&i.srcs[1]];
                let buf_s = g(&r, &i.srcs[0]);
                let idx_s = g(&r, &i.srcs[1]);
                let idx_is_const_lane = idx.op == Ops::Cast
                    && idx
                        .srcs
                        .first()
                        .map(|c| info[c].op == Ops::Const)
                        .unwrap_or(false);
                let lane = if idx_is_const_lane {
                    idx.srcs
                        .first()
                        .and_then(|c| const_int(ctx, c.clone()))
                        .unwrap_or(0)
                } else {
                    0
                };
                if lang.rust {
                    let is_array = buf.op == Ops::Buffer
                        || matches!(buf.derived, Some(AddrSpace::Alu) | Some(AddrSpace::Reg))
                        || matches!(buf.addrspace, Some(AddrSpace::Alu) | Some(AddrSpace::Reg));
                    if is_array {
                        if !idx_is_const_lane {
                            format!("({buf_s})[({idx_s}) as usize]")
                        } else {
                            format!("{buf_s}[{lane} as usize]")
                        }
                    } else {
                        format!("({buf_s}).offset(({idx_s}) as isize)")
                    }
                } else if buf.derived == Some(AddrSpace::Alu) {
                    // lane access
                    if !idx_is_const_lane {
                        format!("({buf_s})[{idx_s}]")
                    } else if buf.numel > lang.gep_arr_threshold {
                        format!("{buf_s}[{lane}]")
                    } else {
                        let ch = "xyzwabcd".chars().nth(lane as usize).unwrap_or('w');
                        format!("{buf_s}.{ch}")
                    }
                } else if idx.op == Ops::Add {
                    format!("({buf_s}+{})", strip_parens(&idx_s))
                } else {
                    format!("({buf_s}+{idx_s})")
                }
            }
            Ops::Stack => {
                let inner: Vec<String> = i.srcs.iter().map(|s| g(&r, s)).collect();
                if lang.rust {
                    format!("[{}]", inner.join(", "))
                } else {
                    let sty = lang.render_type(&i);
                    let float4 = lang.float4.replace("float4", &sty);
                    let (open, close) = lang.float4_style;
                    format!("{float4}{open}{}{close}", inner.join(","))
                }
            }
            Ops::Load => {
                let access = lang.render_access(&info, (i.srcs[0]).clone(), &g(&r, &i.srcs[0]));
                if i.srcs.len() >= 3 {
                    let gate = g(&r, &i.srcs[2]);
                    let var = g(&r, &i.srcs[1]);
                    if lang.rust {
                        if info[&i.srcs[2]].dt == dtype::BOOL {
                            format!("(if {gate} {{ {access} }} else {{ {var} }})")
                        } else {
                            format!("(if ({gate}) != 0 {{ {access} }} else {{ {var} }})")
                        }
                    } else {
                        format!("({gate}?{access}:{var})")
                    }
                } else {
                    format!("({access})")
                }
            }
            Ops::Store => {
                let bidx = lang.render_access(&info, (i.srcs[0]).clone(), &g(&r, &i.srcs[0]));
                format!("{bidx} = {};", g(&r, &i.srcs[1]))
            }
            Ops::Wmma => {
                let nm = wmma_name(ctx, u.clone());
                format!(
                    "__{}({}, {}, {})",
                    nm,
                    g(&r, &i.srcs[0]),
                    g(&r, &i.srcs[1]),
                    g(&r, &i.srcs[2])
                )
            }
            Ops::Buffer => {
                if lang.rust {
                    if i.addrspace == Some(AddrSpace::Local) {
                        panic!("TGERR: rust renderer does not support local memory");
                    }
                    format!(
                        "let mut {}: [{}; {}] = [{}; {}];",
                        r[u],
                        rust_type_name(i.dt),
                        i.numel,
                        rust_zero(i.dt),
                        i.numel
                    )
                } else {
                    // render_buffer (cstyle.py:169): the (local) storage declaration.
                    // `_render_dtype(dtype, sz=1)` uses the default ALU addrspace, so the
                    // only `__local ` is the one in `smem_prefix` (not a pointer).
                    let prefix = if i.addrspace == Some(AddrSpace::Local) {
                        format!("{}{}", lang.smem_align, lang.smem_prefix)
                    } else {
                        String::new()
                    };
                    format!(
                        "{prefix}{} {}[{}];",
                        lang.render_dtype(i.dt, 1, None, false),
                        r[u],
                        i.numel
                    )
                }
            }
            _ if group::in_(group::ALU, i.op) => {
                let args: Vec<String> = i
                    .srcs
                    .iter()
                    .map(|v| {
                        if info[v].op == i.op
                            && matches!(i.op, Ops::Add | Ops::Mul | Ops::Xor | Ops::Or | Ops::And)
                        {
                            strip_parens(&g(&r, v)).to_string()
                        } else {
                            g(&r, v)
                        }
                    })
                    .collect();
                if lang.rust {
                    let idx_args: Vec<String> = if i.numel > 1 {
                        i.srcs
                            .iter()
                            .zip(args.iter())
                            .map(|(v, s)| {
                                if info[v].numel > 1 {
                                    format!("{s}[k]")
                                } else {
                                    s.clone()
                                }
                            })
                            .collect()
                    } else {
                        args.clone()
                    };
                    let mut idx_args = idx_args;
                    if i.op == Ops::Where && info[&i.srcs[0]].dt != dtype::BOOL {
                        idx_args[0] = format!("({}) != 0", idx_args[0]);
                    }
                    let scalar = code_for_op(i.op, &idx_args, false, false, true, false, i.dt);
                    if i.numel > 1 {
                        format!("core::array::from_fn(|k| {scalar})")
                    } else {
                        scalar
                    }
                } else {
                    code_for_op(
                        i.op,
                        &args,
                        lang.clang_builtins,
                        lang.cuda,
                        false,
                        lang.metal,
                        i.dt,
                    )
                }
            }
            other => panic!("TGERR: cstyle op {other:?} not ported"),
        };

        if matches!(i.op, Ops::Endif | Ops::End) {
            depth = depth.saturating_sub(1);
        }
        let is_const_cast = i.op == Ops::Cast && info[&i.srcs[0]].op == Ops::Const;
        let reg_load = i.op == Ops::Load
            && info[&i.srcs[0]].derived == Some(AddrSpace::Reg)
            && child_count[u] == 1;
        let ptr_cast = matches!(i.op, Ops::Cast | Ops::Bitcast)
            && matches!(
                i.addrspace,
                Some(AddrSpace::Global) | Some(AddrSpace::Local)
            );
        let single_use_inline = matches!(i.op, Ops::Stack | Ops::Cast | Ops::Bitcast)
            || (group::in_(group::ALU, i.op) && i.op != Ops::Where);
        let inline = (i.op != Ops::Cast || i.numel == 1)
            && (is_const_cast
                || matches!(i.op, Ops::Index | Ops::Shrink | Ops::Customi)
                || ((lang.rust || lang.cpu) && i.op == Ops::Special)
                || reg_load
                || ptr_cast
                || (single_use_inline && child_count[u] == 1));
        if inline {
            r.insert(u.clone(), l);
        } else {
            let full = if lang.rust {
                if matches!(i.op, Ops::Range | Ops::Store | Ops::Buffer | Ops::Special)
                    || i.dt == dtype::VOID
                {
                    l
                } else {
                    format!("let mut {}: {} = {};", r[u], lang.render_type(&i), l)
                }
            } else if !matches!(i.op, Ops::Range | Ops::Store | Ops::Buffer) && i.dt != dtype::VOID
            {
                let semi = if i.op == Ops::Special { "" } else { ";" };
                format!("{} {} = {}{semi}", lang.render_type(&i), r[u], l)
            } else {
                l
            };
            let indented: Vec<String> = full
                .split('\n')
                .map(|line| format!("{}{}", "  ".repeat(depth), line))
                .collect();
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
    if lang.vector_defines || lang.cuda {
        for u in lst {
            let i = &info[u];
            if matches!(i.addrspace, None | Some(AddrSpace::Alu))
                && i.dt != dtype::VOID
                && i.has_shape
            {
                let key = (i.dt, i.numel);
                if !seen_dts.contains(&key) {
                    seen_dts.push(key);
                }
            }
        }
    }
    let mut define_lines: Vec<String> = Vec::new();
    if lang.vector_defines {
        for (dt, count) in &seen_dts {
            if *count > 1 {
                let alignment = if *dt == dtype::BOOL {
                    1
                } else {
                    2u64.pow((dt.itemsize() as u32 * *count as u32).ilog2())
                };
                let scal = lang.render_dtype(*dt, 1, Some(AddrSpace::Reg), false);
                let vec = lang.render_dtype(*dt, *count, Some(AddrSpace::Reg), false);
                if lang.cpu {
                    let bytes = dt.itemsize() * *count as usize;
                    define_lines.push(format!(
                        "typedef {} {} __attribute__((vector_size({}), aligned({})));",
                        scal, vec, bytes, alignment
                    ));
                } else {
                    define_lines.push(format!(
                        "typedef {} {} __attribute__((aligned({}),ext_vector_type({})));",
                        scal, vec, alignment, count
                    ));
                }
            }
        }
    }
    if lang.cpu {
        let mut prologue: Vec<String> = Vec::new();
        for (i, (nm, inf, _)) in bufs.iter().enumerate() {
            let ty = lang.render_dtype(inf.dt, 1, inf.addrspace, true);
            prologue.push(format!("{ty} restrict {nm} = ({ty})bufs[{i}];"));
        }
        for (i, (nm, inf)) in vals.iter().enumerate() {
            let ty = lang.render_dtype(inf.dt, 1, None, false);
            prologue.push(format!("{ty} {nm} = ({ty})vals[{i}];"));
        }
        let mut has_gidx0 = false;
        for (nm, dt) in &gidx_params {
            let dim: usize = nm[nm.len() - 1..].parse().unwrap_or(0);
            let ty = lang.render_dtype(*dt, 1, None, false);
            if dim == 0 {
                has_gidx0 = true;
                prologue.push(format!("{ty} gidx0_start = ({ty})gidx[0];"));
                prologue.push(format!("{ty} gidx0_end = ({ty})gidx[1];"));
            } else {
                prologue.push(format!("{ty} {nm} = ({ty})gidx[{}];", dim + 1));
            }
        }
        let body = kernel.join("\n");
        let body = if has_gidx0 {
            format!("for (int gidx0 = gidx0_start; gidx0 < gidx0_end; gidx0++) {{\n{body}\n}}")
        } else {
            body
        };
        let defines = define_lines.join("\n");
        return format!(
            "{defines}\nvoid {name}(uint8_t** bufs, int64_t* vals, int64_t* gidx) {{\n{}\n{}\n}}\n",
            prologue.join("\n"),
            body
        );
    }

    if lang.rust {
        // Fixed ABI: the host passes raw buffer pointers, scalar `vals`, and the
        // global id. The prologue turns those into typed locals.
        let mut prologue: Vec<String> = Vec::new();
        for (i, (nm, inf, _)) in bufs.iter().enumerate() {
            prologue.push(format!(
                "let {nm} = *bufs.add({i}) as {};",
                rust_dtype(inf.dt, 1, inf.addrspace, true)
            ));
        }
        for (i, (nm, inf)) in vals.iter().enumerate() {
            prologue.push(format!(
                "let {nm} = *vals.add({i}) as {};",
                rust_type_name(inf.dt)
            ));
        }
        for (nm, dt) in &gidx_params {
            let dim: usize = nm[nm.len() - 1..].parse().unwrap_or(0);
            let ty = rust_type_name(*dt);
            if dim == 0 {
                prologue.push(format!("let gidx0_start = *gidx.add(0) as {ty};"));
                prologue.push(format!("let gidx0_end = *gidx.add(1) as {ty};"));
            } else {
                // gidx0 is a [start, end) chunk (slots 0/1); the other dims
                // follow at 2/3.
                prologue.push(format!("let {nm} = *gidx.add({}) as {ty};", dim + 1));
            }
        }
        let body = kernel.join("\n");
        // `rustc`'s loop vectorizer is exponential in the length of an integer
        // shift/xor chain: measured on rustc 1.97 at `-C opt-level=3 -C
        // target-cpu=native`, a loop body with 20 shifts compiles in 0.26 s,
        // 24 in 1.8 s, 28 in 16 s and the RNG kernels' 80+ shifts take over
        // ten minutes. The hash/`threefry` kernels are exactly that shape, so
        // route their loop index through `black_box`: the vectorizer cannot
        // vectorize an opaque induction variable and falls back to the (already
        // vectorized, tiny) body. It costs nothing at runtime and leaves every
        // other kernel's gidx loop vectorizable.
        const SHIFT_HEAVY: usize = 24;
        let shifts = body.matches(" << ").count() + body.matches(" >> ").count();
        let body = if gidx_params.iter().any(|(n, _)| n == "gidx0") {
            let idx = if shifts >= SHIFT_HEAVY {
                "core::hint::black_box(__gidx0)"
            } else {
                "__gidx0"
            };
            format!("for __gidx0 in gidx0_start..gidx0_end {{\nlet gidx0 = {idx};\n{body}\n}}")
        } else {
            body
        };
        return format!(
            "#[unsafe(no_mangle)]\n\
             #[allow(dead_code, unused_mut, unused_variables, unused_parens, non_snake_case, unsafe_op_in_unsafe_fn)]\n\
             pub unsafe extern \"C\" fn {name}(\n    \
                 bufs: *const *mut u8,\n    \
                 vals: *const i64,\n    \
                 gidx: *const i64,\n\
             ) {{\n{}\n{}\n}}",
            prologue.join("\n"),
            body
        );
    }

    // buftypes
    let buftypes: Vec<String> = bufs
        .iter()
        .map(|(nm, i, _writable)| {
            let as_ = i.addrspace;
            if lang.metal && as_ == Some(AddrSpace::Alu) && i.dt == dtype::INT32 {
                // MetalRenderer.arg_int_prefix (cstyle.py:263): scalar int
                // params are `constant int&` set with `setBytes`.
                return format!("constant int& {nm}");
            }
            let vol = if i.volatile { "volatile " } else { "" };
            let (pre, suf) = if as_ == Some(AddrSpace::Alu) {
                (lang.var_prefix, lang.var_suffix)
            } else {
                ("", lang.buffer_suffix)
            };
            format!(
                "{vol}{pre}{}{suf} {nm}",
                lang.render_dtype(i.dt, 1, as_, false)
            )
        })
        .collect();
    // launch_bounds: product of the local SPECIAL ranges' extents (cstyle.py:158).
    let mut launch_bounds: i64 = 1;
    for u in lst {
        if info[u].op == Ops::Special
            && let Arg::Str(s) = ctx.arg_of(u)
            && s.starts_with('l')
        {
            let rng = info[u].srcs[0].clone();
            if let Some(ext) = info
                .get(&rng)
                .and_then(|ri| ri.srcs.first())
                .and_then(|e| const_int(ctx, e.clone()))
            {
                launch_bounds *= ext as i64;
            }
        }
    }
    let ktypedef = lang
        .kernel_typedef
        .replace("{launch_bounds}", &launch_bounds.to_string());
    if lang.metal {
        // MetalRenderer.render_kernel (cstyle.py:286): the implicit `gid`/`lid`
        // thread attributes are the last two kernel arguments.
        let mut args = buftypes.clone();
        args.push("uint3 gid [[threadgroup_position_in_grid]]".to_string());
        args.push("uint3 lid [[thread_position_in_threadgroup]]".to_string());
        let prg = format!(
            "{} {}({}) {{\n{}\n}}",
            ktypedef,
            name,
            args.join(", "),
            kernel.join("\n")
        );
        // MetalRenderer.render_kernel (cstyle.py:378): one simdgroup helper per
        // distinct `(name, dtype_in, dtype_out)` WMMA shape.
        let mut prefix = String::from("#include <metal_stdlib>\nusing namespace metal;\n");
        let mut seen: Vec<(String, DTypeId, DTypeId)> = Vec::new();
        for (wmma, _dims, dtype_in, dtype_out, _upcast) in wmma_args(ctx, lst) {
            if seen
                .iter()
                .any(|(n, a, b)| n == &wmma && *a == dtype_in && *b == dtype_out)
            {
                continue;
            }
            seen.push((wmma.clone(), dtype_in, dtype_out));
            let in_vec = lang.render_dtype(dtype_in, 2, Some(AddrSpace::Reg), false);
            let out_vec = lang.render_dtype(dtype_out, 2, Some(AddrSpace::Reg), false);
            let in_mat = lang.render_dtype(dtype_in, 1, None, false);
            let out_mat = lang.render_dtype(dtype_out, 1, None, false);
            prefix.push_str(&format!(
                "{out_vec} __{wmma}({in_vec} a, {in_vec} b, {out_vec} c) {{\n  \
                 simdgroup_{in_mat}8x8 mat_a, mat_b; simdgroup_{out_mat}8x8 mat_c;\n  \
                 mat_a.thread_elements()[0] = a[0]; mat_b.thread_elements()[0] = b[0]; mat_c.thread_elements()[0] = c[0];\n  \
                 mat_a.thread_elements()[1] = a[1]; mat_b.thread_elements()[1] = b[1]; mat_c.thread_elements()[1] = c[1];\n  \
                 simdgroup_multiply_accumulate(mat_c, mat_a, mat_b, mat_c);\n  \
                 return {out_vec}(mat_c.thread_elements()[0], mat_c.thread_elements()[1]);\n}}\n"
            ));
        }
        return format!("{prefix}{prg}");
    }
    let prg = format!(
        "{} {}({}) {{\n{}\n}}",
        ktypedef,
        name,
        buftypes.join(", "),
        kernel.join("\n")
    );
    if lang.cuda {
        // CUDARenderer.render_kernel prefix (cstyle.py:436-455).
        let mut prefix: Vec<String> = vec![
            "typedef unsigned int uint;".to_string(),
            "#define INFINITY (__int_as_float(0x7f800000))".to_string(),
            "#define NAN (__int_as_float(0x7fffffff))".to_string(),
            "template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }".to_string(),
        ];
        if seen_dts.iter().any(|(d, _)| dtype::in_set(dtype::FP8S, *d)) {
            prefix.push("#include <cuda_fp8.h>".to_string());
        }
        if seen_dts.iter().any(|(d, _)| *d == dtype::FLOAT16) {
            prefix.push("#include <cuda_fp16.h>".to_string());
        }
        if seen_dts.iter().any(|(d, _)| *d == dtype::BFLOAT16) {
            prefix.push("#include <cuda_bf16.h>".to_string());
        }
        for (dt, count) in &seen_dts {
            let fp8 = dtype::in_set(dtype::FP8S, *dt);
            let halfbf = *dt == dtype::FLOAT16 || *dt == dtype::BFLOAT16;
            if (halfbf && (*count == 4 || *count == 8)) || (fp8 && matches!(*count, 2 | 4 | 8 | 16))
            {
                prefix.push(cuda_vector_prefix(lang, *dt, *count));
            }
        }
        prefix.extend(wmma_asm_prefix(lang, ctx, lst));
        return format!("{}\n{prg}", prefix.join("\n"));
    }
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

/// CUDARenderer.`render_vector_prefix` (cstyle.py:438-442) — the `_nms` struct
/// plus a `make_<vec>` constructor.
fn cuda_vector_prefix(lang: &CStyleLang, dt: DTypeId, count: i64) -> String {
    let vec = lang.render_dtype(dt, count, Some(AddrSpace::Reg), false);
    let scal = lang.render_dtype(dt, 1, None, false);
    let nms: Vec<String> = "xyzwabcdefghijkl"
        .chars()
        .map(|c| c.to_string())
        .chain((16..32).map(|i| format!("v{i}")))
        .collect();
    let elems: Vec<String> = nms[..count as usize].to_vec();
    let elems_join = elems.join(", ");
    let header: Vec<String> = elems.iter().map(|e| format!("{scal} {e}")).collect();
    format!(
        "struct __align__({}) {vec} {{ {scal} {elems_join}; }}; __device__ {vec} make_{vec}({}) {{ {vec} r={{{elems_join}}}; return r; }}",
        dt.itemsize() * count as usize,
        header.join(", ")
    )
}

fn const_int(ctx: &mut UOpCtx, u: UopId) -> Option<i128> {
    match ctx.val(&u) {
        ConstVal::Int(v) => Some(v),
        _ => None,
    }
}

/// One WMMA call site: `(name, dims, dtype_in, dtype_out, upcast_sizes)`.
type WmmaArg = (String, (i64, i64, i64), DTypeId, DTypeId, (i64, i64, i64));

/// `wmma_args` (cstyle.py:112) — the deduped WMMA call sites in the linear list.
fn wmma_args(ctx: &mut UOpCtx, lst: &[UopId]) -> Vec<WmmaArg> {
    use crate::uop::Ops;
    let mut seen: Vec<String> = Vec::new();
    let mut out = Vec::new();
    for u in lst {
        if ctx.op_of(u) != Ops::Wmma {
            continue;
        }
        let name = wmma_name(ctx, u.clone());
        if seen.contains(&name) {
            continue;
        }
        seen.push(name.clone());
        let (dims, dtype_in) = match ctx.arg_of(u).clone() {
            Arg::Seq { tuple: true, items } if items.len() >= 2 => {
                let dims = match &items[0] {
                    Arg::Seq { items: d, .. } => {
                        let v: Vec<i64> = d.iter().filter_map(|it| it.as_int()).collect();
                        if v.len() >= 3 {
                            (v[0], v[1], v[2])
                        } else {
                            continue;
                        }
                    }
                    _ => continue,
                };
                let di = match &items[1] {
                    Arg::Dtype(d) => *d,
                    _ => continue,
                };
                (dims, di)
            }
            _ => continue,
        };
        let dtype_out = ctx.dt_of(u);
        let mut t = [0i64; 3];
        for (i, slot) in t.iter_mut().enumerate() {
            let s = ctx.srcs(u)[i].clone();
            if let Ok(Some(sh)) = ctx.shape(&s)
                && let Some(last) = sh.last()
            {
                *slot = last.clone().as_i64_or_panic();
            }
        }
        out.push((name, dims, dtype_in, dtype_out, (t[0], t[1], t[2])));
    }
    out
}

/// The CUDARenderer `mma.sync.aligned` asm wrapper (cstyle.py:449-455).
fn wmma_asm_prefix(lang: &CStyleLang, ctx: &mut UOpCtx, lst: &[UopId]) -> Vec<String> {
    let map_in = |d: DTypeId| -> &'static str {
        if d == dtype::FLOAT32 {
            "tf32"
        } else if d == dtype::FLOAT16 {
            "f16"
        } else if d == dtype::BFLOAT16 {
            "bf16"
        } else if d == dtype::FP8E4M3 {
            "e4m3"
        } else if d == dtype::FP8E5M2 {
            "e5m2"
        } else {
            "?"
        }
    };
    let map_out = |d: DTypeId| -> &'static str {
        if d == dtype::FLOAT32 {
            "f32"
        } else if d == dtype::FLOAT16 {
            "f16"
        } else {
            "?"
        }
    };
    let mut out = Vec::new();
    for (name, (n, m, k), dtype_in, dtype_out, upcast) in wmma_args(ctx, lst) {
        let sizes = [upcast.0, upcast.1, upcast.2];
        let dts = [dtype_in, dtype_in, dtype_out];
        let wd: Vec<String> = dts
            .iter()
            .zip(sizes.iter())
            .map(|(d, i)| lang.render_dtype(*d, *i, Some(AddrSpace::Reg), false))
            .collect();
        let nops: Vec<i64> = dts
            .iter()
            .zip(sizes.iter())
            .map(|(d, i)| i * d.itemsize() as i64 / 4)
            .collect();
        let total: i64 = nops.iter().sum();
        let operands: Vec<String> = (0..total).map(|i| format!("%{i}")).collect();
        let c = &operands[..nops[2] as usize];
        let a = &operands[nops[2] as usize..(nops[2] + nops[0]) as usize];
        let b = &operands[operands.len() - nops[1] as usize..];
        let c0s = c.join(", ");
        let aops = a.join(", ");
        let bops = b.join(", ");
        let outc: Vec<String> = (0..nops[2]).map(|i| format!("\"+r\"(c_pk[{i}])")).collect();
        let ain: Vec<String> = (0..nops[0]).map(|i| format!("\"r\"(a_pk[{i}])")).collect();
        let bin: Vec<String> = (0..nops[1]).map(|i| format!("\"r\"(b_pk[{i}])")).collect();
        let (wmc, wma, wmb) = (&wd[2], &wd[0], &wd[1]);
        let din = map_in(dtype_in);
        let dout = map_out(dtype_out);
        let mut s = String::new();
        s.push_str(&format!(
            "__device__ {wmc} __{name}({wma} a, {wmb} b, {wmc} c){{\n"
        ));
        s.push_str("  int *a_pk = (int *)(&a), *b_pk = (int *)(&b), *c_pk = (int *)(&c);\n");
        s.push_str(&format!(
            "  asm(\"mma.sync.aligned.m{m}n{n}k{k}.row.col.{dout}.{din}.{din}.{dout}\"\n"
        ));
        s.push_str(&format!("      \"{{{c0s}}}, {{{aops}}},\"\n"));
        s.push_str(&format!("      \"{{{bops}}}, {{{c0s}}};\"\n"));
        s.push_str(&format!("    : {}\n", outc.join(", ")));
        s.push_str(&format!("    : {}, {});\n", ain.join(", "), bin.join(", ")));
        s.push_str("  return c;\n}");
        out.push(s);
    }
    out
}

/// The base_rewrite const rules (in python match order) — cast dtype decides.
fn const_text(ctx: &mut UOpCtx, dt: DTypeId, c: UopId, lang: &CStyleLang) -> String {
    let val = ctx.val(&c);
    if lang.rust {
        return const_text_rust(dt, val);
    }
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
        // OpenCL stores bf16 as `ushort`, so a bf16 const renders as its bit
        // pattern, not a float cast (cstyle.py:325). Applies to any value.
        if dt == dtype::BFLOAT16 && lang.bf16_const_bits {
            let bits = (dtype::float_to_bf16(vfloat) as f32).to_bits() >> 16;
            return format!("{}u", bits);
        }
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
        return if vint.unwrap_or(0) != 0 {
            "1".to_string()
        } else {
            "0".to_string()
        };
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

/// Rust literal for a const node (mirrors `const_text`'s float/int rules).
fn const_text_rust(dt: DTypeId, val: ConstVal) -> String {
    let vint = match val {
        ConstVal::Int(v) => Some(v),
        ConstVal::Bool(b) => Some(b as i128),
        ConstVal::Float(_) => None,
        ConstVal::Big(_) | ConstVal::Invalid => None,
    };
    let vfloat = match val {
        ConstVal::Float(f) => f,
        ConstVal::Int(v) => v as f64,
        ConstVal::Bool(b) => b as i64 as f64,
        ConstVal::Big(_) | ConstVal::Invalid => 0.0,
    };
    if dtype::is_float(dt) {
        if dt == dtype::WEAKFLOAT {
            return py_float(vfloat);
        }
        if dt == dtype::BFLOAT16 {
            // bf16 is stored as u16; render the rounded bit pattern.
            let bits = (dtype::float_to_bf16(vfloat) as f32).to_bits() >> 16;
            return format!("{bits}u16");
        }
        if dt == dtype::FLOAT16 {
            panic!("TGERR: rust renderer does not support native f16 constants");
        }
        if dt == dtype::FLOAT32 {
            let v = vfloat as f32;
            if v.is_nan() {
                return "f32::NAN".to_string();
            }
            if v == f32::INFINITY {
                return "f32::INFINITY".to_string();
            }
            if v == f32::NEG_INFINITY {
                return "(f32::NEG_INFINITY)".to_string();
            }
            return format!("{}f32", py_float(v as f64));
        }
        if vfloat.is_nan() {
            return "f64::NAN".to_string();
        }
        if vfloat == f64::INFINITY {
            return "f64::INFINITY".to_string();
        }
        if vfloat == f64::NEG_INFINITY {
            return "(f64::NEG_INFINITY)".to_string();
        }
        return format!("{}f64", py_float(vfloat));
    }
    let v = vint.unwrap_or(0);
    if dt == dtype::BOOL {
        return if v != 0 { "true" } else { "false" }.to_string();
    }
    if dt == dtype::INT8 {
        return format!("{}i8", v as i8);
    }
    if dt == dtype::INT16 {
        return format!("{}i16", v as i16);
    }
    if dt == dtype::INT32 {
        return format!("{}i32", v as i32);
    }
    if dt == dtype::INT64 {
        return format!("{}i64", v as i64);
    }
    if dt == dtype::UINT8 {
        return format!("{}u8", v as u8);
    }
    if dt == dtype::UINT16 {
        return format!("{}u16", v as u16);
    }
    if dt == dtype::UINT32 {
        return format!("{}u32", v as u32);
    }
    if dt == dtype::UINT64 {
        return format!("{}u64", v as u64);
    }
    if dt == dtype::WEAKINT {
        return format!("{v}");
    }
    panic!("TGERR: rust renderer has no const for {dt:?}");
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
    // `truncate[dt]` for ints is the C cast to the dtype width — it wraps, it
    // does not clamp (dtype.py:289-293, `ctypes.c_<n>(x).value`). Clamping made
    // `-1` render as `0u`, which broke `-x` on uint (bf16 bit math).
    if dtype::in_set(dtype::UINTS, dt) {
        match dt.bitsize() {
            8 => (v as u8) as i128,
            16 => (v as u16) as i128,
            32 => (v as u32) as i128,
            64 => (v as u64) as i128,
            _ => v,
        }
    } else {
        match dt.bitsize() {
            8 => v as i8 as i128,
            16 => v as i16 as i128,
            32 => v as i32 as i128,
            64 => v as i64 as i128,
            _ => v,
        }
    }
}

/// `_wmma_name` (cstyle.py:107) — `WMMA_{N}_{M}_{K}_{dtype_in.name}_{dtype_out.name}`.
fn wmma_name(ctx: &mut UOpCtx, u: UopId) -> String {
    let dt_out = ctx.dt_of(&u);
    let arg = ctx.arg_of(&u).clone();
    if let Arg::Seq { tuple: true, items } = arg
        && items.len() >= 2
    {
        let dims: Vec<String> = match &items[0] {
            Arg::Seq { items: d, .. } => d
                .iter()
                .filter_map(|it| it.as_int())
                .map(|v| v.to_string())
                .collect(),
            _ => vec![],
        };
        let dtype_in = match &items[1] {
            Arg::Dtype(d) => d.name().to_string(),
            _ => "?".to_string(),
        };
        return format!("WMMA_{}_{}_{}", dims.join("_"), dtype_in, dt_out.name()).replace(' ', "_");
    }
    "WMMA".to_string()
}

/// `CStyleLanguage.code_for_op` (clang subset + CUDARenderer overrides).
fn code_for_op(
    op: Ops,
    a: &[String],
    clang: bool,
    cuda: bool,
    rust: bool,
    metal: bool,
    dt: crate::dtype::DTypeId,
) -> String {
    if rust {
        return code_for_op_rust(op, a, dt);
    }
    // CUDARenderer (cstyle.py:413): half/bf16 use the `h*` intrinsics.
    let h = cuda && (dt == crate::dtype::FLOAT16 || dt == crate::dtype::BFLOAT16);
    match op {
        Ops::Sqrt => {
            if h {
                format!("hsqrt({})", a[0])
            } else if clang {
                if dt == crate::dtype::FLOAT64 {
                    format!("__builtin_sqrt({})", a[0])
                } else {
                    format!("__builtin_sqrtf({})", a[0])
                }
            } else {
                format!("sqrt({})", a[0])
            }
        }
        Ops::Exp2 => {
            if h {
                format!("hexp2({})", a[0])
            } else {
                format!("exp2({})", a[0])
            }
        }
        Ops::Log2 => {
            if h {
                format!("hlog2({})", a[0])
            } else {
                format!("log2({})", a[0])
            }
        }
        Ops::Sin => {
            if h {
                format!("hsin({})", a[0])
            } else if metal {
                // MetalRenderer.code_for_op (cstyle.py:272): `precise::sin`.
                format!("precise::sin({})", a[0])
            } else {
                format!("sin({})", a[0])
            }
        }
        Ops::Trunc => {
            if h {
                format!("htrunc({})", a[0])
            } else if clang {
                if dt == crate::dtype::FLOAT64 {
                    format!("__builtin_trunc({})", a[0])
                } else {
                    format!("__builtin_truncf({})", a[0])
                }
            } else {
                format!("trunc({})", a[0])
            }
        }
        Ops::Reciprocal => {
            if h {
                format!("hrcp({})", a[0])
            } else {
                format!("(1/{})", a[0])
            }
        }
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

/// `code_for_op` for the Rust target. All math goes through `libm` (the
/// generated crate is `#![no_std]`); the names are declared by the compiler.
fn code_for_op_rust(op: Ops, a: &[String], dt: crate::dtype::DTypeId) -> String {
    let f64_ = dt == crate::dtype::FLOAT64;
    let f32_ = dt == crate::dtype::FLOAT32;
    match op {
        Ops::Sqrt => {
            if f64_ {
                format!("sqrt({})", a[0])
            } else {
                format!("sqrtf({})", a[0])
            }
        }
        Ops::Exp2 => {
            if f64_ {
                format!("exp2({})", a[0])
            } else {
                format!("exp2f({})", a[0])
            }
        }
        Ops::Log2 => {
            if f64_ {
                format!("log2({})", a[0])
            } else {
                format!("log2f({})", a[0])
            }
        }
        Ops::Sin => {
            if f64_ {
                format!("sin({})", a[0])
            } else {
                format!("sinf({})", a[0])
            }
        }
        Ops::Trunc => {
            if f64_ {
                format!("trunc({})", a[0])
            } else {
                format!("truncf({})", a[0])
            }
        }
        Ops::Reciprocal => {
            if f32_ {
                format!("(1.0f32 / {})", a[0])
            } else if f64_ {
                format!("(1.0f64 / {})", a[0])
            } else {
                format!("(1 / {})", a[0])
            }
        }
        Ops::Neg => format!("-{}", a[0]),
        Ops::And => format!("({} & {})", a[0], a[1]),
        Ops::Xor => format!("({} ^ {})", a[0], a[1]),
        Ops::Or => format!("({} | {})", a[0], a[1]),
        Ops::Add => format!("({} + {})", a[0], a[1]),
        Ops::Sub => format!("({} - {})", a[0], a[1]),
        Ops::Mul => format!("({} * {})", a[0], a[1]),
        Ops::Cmod => format!("({} % {})", a[0], a[1]),
        Ops::Cdiv => format!("({} / {})", a[0], a[1]),
        Ops::Cmpne => format!("({} != {})", a[0], a[1]),
        Ops::Shr => format!("({} >> {})", a[0], a[1]),
        Ops::Shl => format!("({} << {})", a[0], a[1]),
        Ops::Cmplt => format!("({} < {})", a[0], a[1]),
        Ops::Where => format!("if {} {{ {} }} else {{ {} }}", a[0], a[1], a[2]),
        Ops::Cmpeq => format!("({} == {})", a[0], a[1]),
        Ops::Fdiv => format!("({} / {})", a[0], a[1]),
        other => panic!("TGERR: rust ALU {other:?} not ported"),
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
        assert_eq!(
            ctext(ConstVal::Int(3), dtype::UINT16, &cl),
            "((ushort)(3u))"
        );
        assert_eq!(ctext(ConstVal::Int(3), dtype::INT16, &cl), "((short)(3))");
        assert_eq!(ctext(ConstVal::Int(3), dtype::UINT32, &cl), "3u");
        assert_eq!(ctext(ConstVal::Int(3), dtype::INT32, &cl), "3");
        assert_eq!(ctext(ConstVal::Bool(true), dtype::BOOL, &cl), "1");
        assert_eq!(ctext(ConstVal::Float(2.0), dtype::FLOAT32, &cl), "2.0f");
        // OpenCL has no half type_map entry -> tinygrad renders `half`; Clang maps it.
        assert_eq!(
            ctext(ConstVal::Float(2.0), dtype::FLOAT16, &cl),
            "((half)(2.0f))"
        );
        let clang = CStyleLang::clang();
        assert_eq!(
            ctext(ConstVal::Float(2.0), dtype::FLOAT16, &clang),
            "((__fp16)(2.0f))"
        );
    }

    /// Non-finite float consts are rendered through a cast of the lang's
    /// `nan`/`infinity` (tinygrad: `((float)(NAN))`, not a bare `(NAN)`).
    #[test]
    fn const_nonfinite_rendering_matches_tinygrad() {
        let cl = CStyleLang::opencl();
        assert_eq!(
            ctext(ConstVal::Float(f64::NAN), dtype::FLOAT32, &cl),
            "((float)(NAN))"
        );
        assert_eq!(
            ctext(ConstVal::Float(f64::INFINITY), dtype::FLOAT32, &cl),
            "((float)(INFINITY))"
        );
        assert_eq!(
            ctext(ConstVal::Float(f64::NEG_INFINITY), dtype::FLOAT32, &cl),
            "((float)(-INFINITY))"
        );
        let clang = CStyleLang::clang();
        assert_eq!(
            ctext(ConstVal::Float(f64::NAN), dtype::FLOAT32, &clang),
            "((float)(__builtin_nanf(\"\")))"
        );
    }

    /// Rust consts are suffixed so they can be used without a type annotation.
    #[test]
    fn rust_const_rendering() {
        let r = CStyleLang::rust();
        assert_eq!(ctext(ConstVal::Float(2.0), dtype::FLOAT32, &r), "2.0f32");
        assert_eq!(ctext(ConstVal::Float(2.0), dtype::FLOAT64, &r), "2.0f64");
        assert_eq!(ctext(ConstVal::Int(3), dtype::INT8, &r), "3i8");
        assert_eq!(ctext(ConstVal::Int(3), dtype::INT16, &r), "3i16");
        assert_eq!(ctext(ConstVal::Int(3), dtype::INT32, &r), "3i32");
        assert_eq!(ctext(ConstVal::Int(3), dtype::INT64, &r), "3i64");
        assert_eq!(ctext(ConstVal::Int(3), dtype::UINT8, &r), "3u8");
        assert_eq!(ctext(ConstVal::Int(3), dtype::UINT16, &r), "3u16");
        assert_eq!(ctext(ConstVal::Int(3), dtype::UINT32, &r), "3u32");
        assert_eq!(ctext(ConstVal::Int(3), dtype::UINT64, &r), "3u64");
        assert_eq!(ctext(ConstVal::Bool(true), dtype::BOOL, &r), "true");
        assert_eq!(ctext(ConstVal::Bool(false), dtype::BOOL, &r), "false");
        assert_eq!(
            ctext(ConstVal::Float(f64::NAN), dtype::FLOAT32, &r),
            "f32::NAN"
        );
        assert_eq!(
            ctext(ConstVal::Float(f64::INFINITY), dtype::FLOAT64, &r),
            "f64::INFINITY"
        );
        assert_eq!(
            ctext(ConstVal::Float(f64::NEG_INFINITY), dtype::FLOAT32, &r),
            "(f32::NEG_INFINITY)"
        );
    }
}
