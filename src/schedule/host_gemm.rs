//! Recognize a pure `A[m,k] * B[k,n] -> REDUCE(ADD)` kernel and describe it as
//! a library GEMM.
//!
//! The scheduler lowers each contraction to its own kernel with an affine
//! `INDEX` per operand. Rather than pattern-matching the high-level tensor
//! graph, we read the already-rangeified store (the same place
//! `try_matmul_tiling` looks) and recover the layout from the index maps:
//!
//! ```text
//! STORE(INDEX(C, m*N + n),
//!       REDUCE(ADD, MUL(INDEX(A, m*K + k), INDEX(B, k*N + n)), k))
//! ```
//!
//! Any load/store pattern that is not a plain strided matrix (masked stores,
//! non-affine offsets, mixed dtypes, multiple reductions) is rejected and the
//! normal generated kernel is used. The descriptor is attached to the kernel's
//! `KernelInfo`; `engine/realize.rs` executes it through cuBLAS when available.

use crate::dtype::DTypeId;
use crate::uop::UopId;
use crate::uop::arg::{Arg, DataArg, DeviceSpec};
use crate::uop::ctx::UOpCtx;
use crate::uop::ops::Ops;
use crate::uop::sint::SVal;

/// A GEMM expressed in the backend's row-major view of the tensors.
///
/// `C[m,n] = alpha * A[m,k] @ B[k,n]`, with optional strided batching along a
/// third axis. `*_off` and `*_b` are in elements.
#[derive(Clone, Debug, PartialEq)]
pub struct HostGemm {
    /// Buffer slot of `A`.
    pub a_slot: i64,
    /// Buffer slot of `B`.
    pub b_slot: i64,
    /// Buffer slot of `C`.
    pub c_slot: i64,
    /// Constant element offset of `A`.
    pub a_off: i64,
    /// Constant element offset of `B`.
    pub b_off: i64,
    /// Constant element offset of `C`.
    pub c_off: i64,
    /// Rows of `C`.
    pub m: i64,
    /// Columns of `C`.
    pub n: i64,
    /// Contraction length.
    pub k: i64,
    /// Stride of the `m` axis in `A` (elements).
    pub a_m: i64,
    /// Stride of the `k` axis in `A`.
    pub a_k: i64,
    /// Stride of the `k` axis in `B`.
    pub b_k: i64,
    /// Stride of the `n` axis in `B`.
    pub b_n: i64,
    /// Stride of the `m` axis in `C`.
    pub c_m: i64,
    /// Stride of the `n` axis in `C`.
    pub c_n: i64,
    /// Number of strided batches (>= 1).
    pub batch: i64,
    /// Batch stride in `A` (0 when shared).
    pub a_b: i64,
    /// Batch stride in `B` (0 when shared).
    pub b_b: i64,
    /// Batch stride in `C`.
    pub c_b: i64,
    /// Storage dtype of `A`/`B` (the multiplication dtype).
    pub dtype: DTypeId,
    /// Storage dtype of `A` (equals [`Self::dtype`] unless the pipeline promoted
    /// one operand, e.g. f32 activations against 16-bit weights).
    pub a_dtype: DTypeId,
    /// Storage dtype of `B`.
    pub b_dtype: DTypeId,
    /// Storage dtype of `C`. Usually equal to `dtype`; Metal's f16 inputs with
    /// an f32 accumulator use `(FLOAT16, FLOAT32)`.
    pub c_dtype: DTypeId,
    /// Scalar multiplier.
    pub alpha: f32,
    /// Optional bias vector slot (-1 when absent), added to every row.
    pub bias_slot: i64,
    /// Element offset of the bias vector.
    pub bias_off: i64,
    /// Storage dtype of the bias vector (`dtype` unless it was promoted).
    pub bias_dtype: DTypeId,
    /// Optional per-column scale slot (-1 when absent).
    pub scale_slot: i64,
    /// Element offset of the scale vector.
    pub scale_off: i64,
    /// Optional residual tensor slot (-1 when absent), added elementwise.
    pub res_slot: i64,
    /// Element offset of the residual tensor.
    pub res_off: i64,
}

impl HostGemm {
    /// A Rust expression that reconstructs this descriptor.
    ///
    /// Used by the AOT emitter: a frozen plan may contain host GEMM steps, so
    /// the descriptor has to survive as source.
    pub fn to_rust_literal(&self) -> String {
        format!(
            "autorust::schedule::host_gemm::HostGemm {{ a_slot: {}, b_slot: {}, \
             c_slot: {}, a_off: {}, b_off: {}, c_off: {}, m: {}, n: {}, k: {}, \
             a_m: {}, a_k: {}, b_k: {}, b_n: {}, c_m: {}, c_n: {}, batch: {}, \
             a_b: {}, b_b: {}, c_b: {}, dtype: autorust::dtype::DTypeId({}), \
             c_dtype: autorust::dtype::DTypeId({}), alpha: {:?}f32, bias_slot: {}, bias_off: {}, scale_slot: {}, \
             scale_off: {}, res_slot: {}, res_off: {} }}",
            self.a_slot,
            self.b_slot,
            self.c_slot,
            self.a_off,
            self.b_off,
            self.c_off,
            self.m,
            self.n,
            self.k,
            self.a_m,
            self.a_k,
            self.b_k,
            self.b_n,
            self.c_m,
            self.c_n,
            self.batch,
            self.a_b,
            self.b_b,
            self.c_b,
            self.dtype.0,
            self.c_dtype.0,
            self.alpha,
            self.bias_slot,
            self.bias_off,
            self.scale_slot,
            self.scale_off,
            self.res_slot,
            self.res_off
        )
    }

    /// Serialize into the kernel's `KernelInfo` arg.
    pub fn to_data(&self) -> DataArg {
        let int = |v: i64| Arg::Int(v);
        DataArg {
            tag: "HostGemm".into(),
            fields: vec![
                ("a_slot".into(), int(self.a_slot)),
                ("b_slot".into(), int(self.b_slot)),
                ("c_slot".into(), int(self.c_slot)),
                ("a_off".into(), int(self.a_off)),
                ("b_off".into(), int(self.b_off)),
                ("c_off".into(), int(self.c_off)),
                ("m".into(), int(self.m)),
                ("n".into(), int(self.n)),
                ("k".into(), int(self.k)),
                ("a_m".into(), int(self.a_m)),
                ("a_k".into(), int(self.a_k)),
                ("b_k".into(), int(self.b_k)),
                ("b_n".into(), int(self.b_n)),
                ("c_m".into(), int(self.c_m)),
                ("c_n".into(), int(self.c_n)),
                ("batch".into(), int(self.batch)),
                ("a_b".into(), int(self.a_b)),
                ("b_b".into(), int(self.b_b)),
                ("c_b".into(), int(self.c_b)),
                ("dtype".into(), int(self.dtype.0 as i64)),
                ("a_dtype".into(), int(self.a_dtype.0 as i64)),
                ("b_dtype".into(), int(self.b_dtype.0 as i64)),
                ("c_dtype".into(), int(self.c_dtype.0 as i64)),
                ("alpha".into(), Arg::Float(self.alpha as f64)),
                ("bias_slot".into(), int(self.bias_slot)),
                ("bias_off".into(), int(self.bias_off)),
                ("bias_dtype".into(), int(self.bias_dtype.0 as i64)),
                ("scale_slot".into(), int(self.scale_slot)),
                ("scale_off".into(), int(self.scale_off)),
                ("res_slot".into(), int(self.res_slot)),
                ("res_off".into(), int(self.res_off)),
            ],
        }
    }

    /// Parse the `host_gemm` field of a kernel's `KernelInfo`.
    pub fn from_arg(arg: &Arg) -> Option<HostGemm> {
        let Arg::Data(d) = arg else { return None };
        if d.tag != "HostGemm" {
            return None;
        }
        let i = |name: &str| match d.get(name) {
            Arg::Int(v) => *v,
            _ => 0,
        };
        let alpha = match d.get("alpha") {
            Arg::Float(v) => *v as f32,
            _ => 1.0,
        };
        let desc = HostGemm {
            a_slot: i("a_slot"),
            b_slot: i("b_slot"),
            c_slot: i("c_slot"),
            a_off: i("a_off"),
            b_off: i("b_off"),
            c_off: i("c_off"),
            m: i("m"),
            n: i("n"),
            k: i("k"),
            a_m: i("a_m"),
            a_k: i("a_k"),
            b_k: i("b_k"),
            b_n: i("b_n"),
            c_m: i("c_m"),
            c_n: i("c_n"),
            batch: i("batch").max(1),
            a_b: i("a_b"),
            b_b: i("b_b"),
            c_b: i("c_b"),
            dtype: DTypeId(i("dtype") as u16),
            a_dtype: if d.fields.iter().any(|(n, _)| n == "a_dtype") {
                DTypeId(i("a_dtype") as u16)
            } else {
                DTypeId(i("dtype") as u16)
            },
            b_dtype: if d.fields.iter().any(|(n, _)| n == "b_dtype") {
                DTypeId(i("b_dtype") as u16)
            } else {
                DTypeId(i("dtype") as u16)
            },
            c_dtype: if d.fields.iter().any(|(n, _)| n == "c_dtype") {
                DTypeId(i("c_dtype") as u16)
            } else {
                DTypeId(i("dtype") as u16)
            },
            alpha,
            bias_slot: {
                let v = i("bias_slot");
                if v == 0 && d.fields.iter().all(|(n, _)| n != "bias_slot") {
                    -1
                } else {
                    v
                }
            },
            bias_off: i("bias_off"),
            bias_dtype: if d.fields.iter().any(|(n, _)| n == "bias_dtype") {
                DTypeId(i("bias_dtype") as u16)
            } else {
                DTypeId(i("c_dtype") as u16)
            },
            scale_slot: if d.fields.iter().any(|(n, _)| n == "scale_slot") {
                i("scale_slot")
            } else {
                -1
            },
            scale_off: i("scale_off"),
            res_slot: if d.fields.iter().any(|(n, _)| n == "res_slot") {
                i("res_slot")
            } else {
                -1
            },
            res_off: i("res_off"),
        };
        Some(desc)
    }
}

/// Look up the `host_gemm` field on a kernel SINK (or any `KernelInfo` arg).
pub fn host_gemm_of(ctx: &UOpCtx, sink: &UopId) -> Option<HostGemm> {
    match ctx.arg_of(sink) {
        Arg::Data(d) if d.tag == "KernelInfo" => {
            let field = d.fields.iter().find(|(n, _)| n == "host_gemm")?;
            HostGemm::from_arg(&field.1)
        }
        _ => None,
    }
}

/// A range keyed by its (axis id, axis type) rather than its arena id.
type RKey = (i64, u16);

fn range_key(ctx: &UOpCtx, u: &UopId) -> Option<RKey> {
    if ctx.op_of(u) != Ops::Range {
        return None;
    }
    match ctx.arg_of(u) {
        Arg::Seq { tuple: true, items } => {
            let id = match items.first()? {
                Arg::Int(i) => *i,
                _ => return None,
            };
            let ty = match items.last()? {
                Arg::Enum(e) => e.value,
                _ => return None,
            };
            Some((id, ty))
        }
        _ => None,
    }
}

fn const_int(ctx: &UOpCtx, u: &UopId) -> Option<i64> {
    match ctx.arg_of(u) {
        Arg::Const(crate::dtype::ConstVal::Int(v)) => i64::try_from(*v).ok(),
        Arg::Int(v) => Some(*v),
        Arg::Const(crate::dtype::ConstVal::Bool(b)) => Some(*b as i64),
        Arg::Const(crate::dtype::ConstVal::Float(f)) if f.fract() == 0.0 => Some(*f as i64),
        Arg::Float(f) if f.fract() == 0.0 => Some(*f as i64),
        _ => None,
    }
}

/// Split an index expression into `const + sum(coeff * range)`.
fn affine(ctx: &UOpCtx, u: &UopId) -> Option<(i64, Vec<(RKey, i64)>)> {
    if let Some(c) = const_int(ctx, u) {
        return Some((c, vec![]));
    }
    if let Some(k) = range_key(ctx, u) {
        return Some((0, vec![(k, 1)]));
    }
    let op = ctx.op_of(u);
    let srcs = ctx.srcs(u);
    if srcs.len() != 2 {
        return None;
    }
    match op {
        Ops::Add | Ops::Sub => {
            let (c0, m0) = affine(ctx, &srcs[0])?;
            let (c1, m1) = affine(ctx, &srcs[1])?;
            let sign = if op == Ops::Sub { -1 } else { 1 };
            let mut m = m0;
            for (k, v) in m1 {
                add_coeff(&mut m, k, sign * v);
            }
            Some((c0 + sign * c1, m))
        }
        Ops::Mul => {
            let (c0, m0) = affine(ctx, &srcs[0])?;
            let (c1, m1) = affine(ctx, &srcs[1])?;
            match (m0.is_empty(), m1.is_empty()) {
                (true, _) => {
                    let mut m = m1;
                    for v in m.iter_mut() {
                        v.1 *= c0;
                    }
                    Some((c0 * c1, m))
                }
                (_, true) => {
                    let mut m = m0;
                    for v in m.iter_mut() {
                        v.1 *= c1;
                    }
                    Some((c0 * c1, m))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

fn add_coeff(m: &mut Vec<(RKey, i64)>, k: RKey, v: i64) {
    if let Some(e) = m.iter_mut().find(|(kk, _)| *kk == k) {
        e.1 += v;
        if e.1 == 0 {
            m.retain(|(kk, _)| *kk != k);
        }
    } else if v != 0 {
        m.push((k, v));
    }
}

fn coeff(m: &[(RKey, i64)], k: RKey) -> i64 {
    m.iter()
        .find(|(kk, _)| *kk == k)
        .map(|(_, v)| *v)
        .unwrap_or(0)
}

fn param_slot(ctx: &UOpCtx, u: &UopId) -> Option<(i64, DTypeId)> {
    match ctx.arg_of(u) {
        Arg::Param(p) if p.addrspace == Some(crate::dtype::AddrSpace::Global) => {
            Some((p.slot, p.dtype))
        }
        _ => None,
    }
}

/// Walk a pointer expression down to its backing `PARAM` (through `AFTER`,
/// casts and reshapes). The index offset is returned separately by `affine`;
/// adding it here too would double-count slice starts.
fn base_of(ctx: &UOpCtx, u: &UopId) -> Option<(i64, DTypeId, i64)> {
    let mut cur = u.clone();
    let off = 0i64;
    for _ in 0..32 {
        match ctx.op_of(&cur) {
            Ops::Index => {
                cur = ctx.srcs(&cur)[0].clone();
            }
            Ops::After | Ops::Cast | Ops::Bitcast | Ops::Contiguous | Ops::Reshape => {
                cur = ctx.srcs(&cur)[0].clone();
            }
            _ => break,
        }
    }
    let (slot, dt) = param_slot(ctx, &cur)?;
    Some((slot, dt, off))
}

/// The affine offset expression of a pointer (`INDEX(buf, off)` → `off`).
fn index_offset(ctx: &UOpCtx, u: &UopId) -> UopId {
    let mut cur = u.clone();
    for _ in 0..8 {
        match ctx.op_of(&cur) {
            Ops::Index => return ctx.srcs(&cur)[1].clone(),
            Ops::After | Ops::Cast | Ops::Bitcast | Ops::Contiguous | Ops::Reshape => {
                cur = ctx.srcs(&cur)[0].clone()
            }
            _ => return cur,
        }
    }
    cur
}

/// Whether `u` contains a reduction without crossing a materialization point.
fn contains_reduce(ctx: &UOpCtx, u: &UopId) -> bool {
    match ctx.op_of(u) {
        Ops::Reduce => true,
        Ops::After | Ops::Buffer | Ops::Param => false,
        _ => ctx.srcs(u).iter().any(|s| contains_reduce(ctx, s)),
    }
}

fn unwrap(ctx: &UOpCtx, mut u: UopId) -> UopId {
    loop {
        match ctx.op_of(&u) {
            Ops::Cast | Ops::Bitcast | Ops::Contiguous | Ops::Reshape => {
                u = ctx.srcs(&u)[0].clone();
            }
            _ => return u,
        }
    }
}

fn library_ready(ctx: &UOpCtx, kind: GemmDevice) -> bool {
    let backend = match kind {
        GemmDevice::Cuda => "cuda",
        GemmDevice::Cl => "cl",
        GemmDevice::Metal => "metal",
        GemmDevice::Vk => "vk",
        GemmDevice::Cpu => "cpu",
    };
    if ctx.settings.library_gemm_for(backend) == crate::settings::LibraryGemm::Never {
        return false;
    }
    match kind {
        GemmDevice::Cpu => {
            #[cfg(any(
                feature = "backend_rust",
                feature = "backend_clang",
                feature = "backend_rayon"
            ))]
            {
                crate::runtime::cpu_gemm::available()
            }
            #[cfg(not(any(
                feature = "backend_rust",
                feature = "backend_clang",
                feature = "backend_rayon"
            )))]
            {
                false
            }
        }
        GemmDevice::Cuda => {
            #[cfg(all(
                feature = "backend_cuda",
                any(target_os = "linux", target_os = "windows")
            ))]
            {
                crate::runtime::cublas::available()
            }
            #[cfg(not(all(
                feature = "backend_cuda",
                any(target_os = "linux", target_os = "windows")
            )))]
            {
                false
            }
        }
        GemmDevice::Cl => {
            #[cfg(feature = "backend_cl")]
            {
                crate::runtime::clblast::available()
            }
            #[cfg(not(feature = "backend_cl"))]
            {
                false
            }
        }
        GemmDevice::Metal => {
            #[cfg(all(feature = "backend_metal", target_os = "macos"))]
            {
                crate::runtime::ops_metal::mps_available()
            }
            #[cfg(not(all(feature = "backend_metal", target_os = "macos")))]
            {
                false
            }
        }
        GemmDevice::Vk => {
            #[cfg(feature = "backend_vk")]
            {
                crate::runtime::ops_vk::coop_gemm_available()
            }
            #[cfg(not(feature = "backend_vk"))]
            {
                false
            }
        }
    }
}

/// Whether any dynamically loaded GEMM library is usable in this process.
///
/// The per-device check happens in [`extract`]: a CUDA store needs cuBLAS and an
/// OpenCL store needs CLBlast.
pub fn library_available(settings: &crate::settings::Settings) -> bool {
    if !settings.any_library_gemm() {
        return false;
    }
    #[cfg(any(
        feature = "backend_rust",
        feature = "backend_clang",
        feature = "backend_rayon"
    ))]
    if crate::runtime::cpu_gemm::available() {
        return true;
    }
    #[cfg(all(
        feature = "backend_cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    if crate::runtime::cublas::available() {
        return true;
    }
    #[cfg(feature = "backend_cl")]
    if crate::runtime::clblast::available() {
        return true;
    }
    #[cfg(all(feature = "backend_metal", target_os = "macos"))]
    if crate::runtime::ops_metal::mps_available() {
        return true;
    }
    #[cfg(feature = "backend_vk")]
    if crate::runtime::ops_vk::coop_gemm_available() {
        return true;
    }
    false
}

/// Which library-backed device a kernel belongs to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum GemmDevice {
    Cpu,
    Cuda,
    Cl,
    Metal,
    Vk,
}

fn gemm_device(ctx: &UOpCtx, u: &UopId) -> Option<GemmDevice> {
    let mut cur = u.clone();
    for _ in 0..8 {
        if let Arg::Param(p) = ctx.arg_of(&cur) {
            let s = match &p.device {
                Some(DeviceSpec::Single(s)) => s.as_str(),
                Some(DeviceSpec::Multi(v)) => v.first().map(|s| s.as_str())?,
                None => return None,
            };
            if s.starts_with("CUDA") {
                return Some(GemmDevice::Cuda);
            }
            if s.starts_with("CL") {
                return Some(GemmDevice::Cl);
            }
            if s.starts_with("METAL") {
                return Some(GemmDevice::Metal);
            }
            if s.starts_with("VK") {
                return Some(GemmDevice::Vk);
            }
            // `CC`/`CLANG` only exist as compiled backends with the feature;
            // without it the name falls back to the interpreter, which has no
            // host-op dispatcher.
            if s == "RUST"
                || s == "RAYON"
                || (cfg!(feature = "backend_clang") && matches!(s, "CC" | "CLANG"))
            {
                return Some(GemmDevice::Cpu);
            }
            return None;
        }
        if ctx.srcs(&cur).is_empty() {
            return None;
        }
        cur = ctx.srcs(&cur)[0].clone();
    }
    None
}

/// Try to interpret a lowered kernel store as a library GEMM.
///
/// Returns `None` for anything that is not a plain strided matrix product, or
/// when the kernel's backend has no usable library GEMM.
pub fn extract(ctx: &mut UOpCtx, store: UopId) -> Option<HostGemm> {
    // Cheap reject for elementwise kernels: only the store value matters, and
    // we must not descend into AFTER (which embeds whole earlier kernels).
    fn has_reduce(ctx: &UOpCtx, u: &UopId) -> bool {
        match ctx.op_of(u) {
            Ops::After => false,
            Ops::Reduce => true,
            _ => ctx.srcs(u).iter().any(|s| has_reduce(ctx, s)),
        }
    }
    let st0 = if ctx.op_of(&store) == Ops::End {
        ctx.srcs(&store)[0].clone()
    } else {
        store.clone()
    };
    if ctx.op_of(&st0) != Ops::Store {
        if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] bail 16"); }
        return None;
    }
    if !has_reduce(ctx, &ctx.srcs(&st0)[1].clone()) {
        if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] bail 19"); }
        return None;
    }
    let kind = gemm_device(ctx, &store)?;
    if !library_ready(ctx, kind) {
        if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] bail 23"); }
        return None;
    }

    let st = if ctx.op_of(&store) == Ops::End {
        ctx.srcs(&store)[0].clone()
    } else {
        store.clone()
    };
    if ctx.op_of(&st) != Ops::Store {
        if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] bail 32"); }
        return None;
    }
    let dst_ptr = ctx.srcs(&st)[0].clone();
    if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] ext: c-affine"); }
    let (c_off0, c_map) = affine(ctx, &index_offset(ctx, &dst_ptr))?;
    if c_map.len() < 2 {
        if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] bail 38"); }
        return None;
    }
    let (c_slot, c_dt, c_base) = base_of(ctx, &dst_ptr)?;
    let c_off = c_off0 + c_base;

    // Operands are either `LOAD(INDEX(...))` or the bare `INDEX(...)` the
    // lowerer emits (an INDEX used as a value is a load in this IR).
    let operand = |ctx: &UOpCtx, u: UopId| -> Option<UopId> {
        let u = unwrap(ctx, u);
        match ctx.op_of(&u) {
            Ops::Load => Some(ctx.srcs(&u)[0].clone()),
            Ops::Index => Some(u),
            _ => None,
        }
    };

    // Peel the fused epilogue, outermost first. A transformer store looks like
    //   `ADD(residual, MUL(ADD(REDUCE(...), bias), scale))`
    // (`x + linear(x)*layer_scale`), a plain linear like
    //   `ADD(REDUCE(...), bias)`.
    // Scalar factors become `alpha`; every other non-reduce operand is kept and
    // classified afterwards by its affine index map (a MUL operand is the
    // per-column scale, an ADD operand is the residual or the bias).
    struct Extra {
        from_mul: bool,
        expr: UopId,
    }
    let mut alpha = 1.0f32;
    let mut extras: Vec<Extra> = Vec::new();
    let mut value = ctx.srcs(&st)[1].clone();
    for _ in 0..12 {
        match ctx.op_of(&value) {
            Ops::Cast | Ops::Bitcast => value = ctx.srcs(&value)[0].clone(),
            Ops::Mul => {
                let srcs = ctx.srcs(&value).to_vec();
                let (sa, sb) = (srcs[0].clone(), srcs[1].clone());
                if ctx.srcs(&sb).is_empty() {
                    alpha *= const_float(ctx, &sb)? as f32;
                    value = sa;
                } else if ctx.srcs(&sa).is_empty() {
                    alpha *= const_float(ctx, &sa)? as f32;
                    value = sb;
                } else if contains_reduce(ctx, &sa) && !contains_reduce(ctx, &sb) {
                    extras.push(Extra {
                        from_mul: true,
                        expr: sb,
                    });
                    value = sa;
                } else if contains_reduce(ctx, &sb) && !contains_reduce(ctx, &sa) {
                    extras.push(Extra {
                        from_mul: true,
                        expr: sa,
                    });
                    value = sb;
                } else {
                    break;
                }
            }
            Ops::Add => {
                let srcs = ctx.srcs(&value).to_vec();
                let (sa, sb) = (srcs[0].clone(), srcs[1].clone());
                if contains_reduce(ctx, &sa) && !contains_reduce(ctx, &sb) {
                    extras.push(Extra {
                        from_mul: false,
                        expr: sb,
                    });
                    value = sa;
                } else if contains_reduce(ctx, &sb) && !contains_reduce(ctx, &sa) {
                    extras.push(Extra {
                        from_mul: false,
                        expr: sa,
                    });
                    value = sb;
                } else {
                    break;
                }
            }
            _ => break,
        }
    }
    if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] ext: value-reduce"); }
    value = unwrap(ctx, value);
    if ctx.op_of(&value) != Ops::Reduce {
        if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] bail 121"); }
        return None;
    }
    let reduce = value;
    let red_dt = ctx.dt_of(&reduce);
    if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] ext: rsrcs"); }
    let rsrcs = ctx.srcs(&reduce).to_vec();
    if rsrcs.len() != 2 || reduce_op(ctx, &reduce) != Some(Ops::Add) {
        if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] bail 128"); }
        return None;
    }
    let red_k = range_key(ctx, &rsrcs[1])?;

    if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] ext: mul"); }
    let mul = unwrap(ctx, rsrcs[0].clone());
    if ctx.op_of(&mul) != Ops::Mul {
        if crate::helpers::env::get("VK_TRACE").is_some() {
            let mut cur = rsrcs[0].clone();
            for depth in 0..4 {
                let op = ctx.op_of(&cur);
                let dt = ctx.dt_of(&cur);
                let kids: Vec<String> = ctx
                    .srcs(&cur)
                    .iter()
                    .map(|k| format!("{:?}/{:?}", ctx.op_of(k), ctx.dt_of(k)))
                    .collect();
                let arg = format!("{:?}", ctx.arg_of(&cur));
                let arg: String = arg.chars().take(90).collect();
                eprintln!("[vk] hg body[{depth}] {op:?} dt={dt:?} kids=[{}] arg={arg}", kids.join(", "));
                let ks = ctx.srcs(&cur).to_vec();
                if ks.is_empty() {
                    break;
                }
                cur = ks[0].clone();
            }
        }
        if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] bail 155"); }
        return None;
    }
    let msrcs = ctx.srcs(&mul).to_vec();
    if msrcs.len() != 2 {
        if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] bail 159"); }
        return None;
    }
    let mul_dt = ctx.dt_of(&mul);
    // A reduced-precision GEMM whose *reduce* is f32 is the scheduler's
    // "promote the multiply to the f32 accumulator" form, which is exactly what
    // the cooperative matrix computes. Metal's MPS path only takes the
    // f16-in/f32-out variant; the Vulkan module also takes a 16-bit result (it
    // rounds the f32 accumulator) and bf16 inputs.
    let mixed_f16_f32 = matches!(kind, GemmDevice::Metal | GemmDevice::Vk)
        && (mul_dt == crate::dtype::FLOAT16
            || (kind == GemmDevice::Vk && mul_dt == crate::dtype::BFLOAT16))
        && red_dt == crate::dtype::FLOAT32
        && (c_dt == crate::dtype::FLOAT32
            || (kind == GemmDevice::Vk && c_dt == mul_dt));
    if (mul_dt != red_dt || mul_dt != c_dt) && !mixed_f16_f32 {
        if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] bail 174"); }
        return None;
    }
    if !matches!(
        ctx.dt_of(&mul),
        crate::dtype::FLOAT32 | crate::dtype::FLOAT16 | crate::dtype::BFLOAT16
    ) {
        if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] bail 180"); }
        return None;
    }
    // CLBlast's `Sgemm` is the only entry point wired up on OpenCL.
    if kind == GemmDevice::Cl && ctx.dt_of(&mul) != crate::dtype::FLOAT32 {
        if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] bail 184"); }
        return None;
    }
    // The CPU host GEMM packs f32 panels; other dtypes stay generated.
    if kind == GemmDevice::Cpu && ctx.dt_of(&mul) != crate::dtype::FLOAT32 {
        if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] bail 188"); }
        return None;
    }

    let dbg = crate::helpers::env::get("AUTORUST_MM_DEBUG").is_some();
    let a_idx = match operand(ctx, msrcs[0].clone()) {
        Some(v) => v,
        None => {
            if dbg {
                eprintln!(
                    "[hg] reject: A operand is {:?}, not a load",
                    ctx.op_of(&unwrap(ctx, msrcs[0].clone()))
                );
            }
        if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] bail 201"); }
            return None;
        }
    };
    let b_idx = match operand(ctx, msrcs[1].clone()) {
        Some(v) => v,
        None => {
            if dbg {
                eprintln!(
                    "[hg] reject: B operand is {:?}, not a load",
                    ctx.op_of(&unwrap(ctx, msrcs[1].clone()))
                );
            }
        if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] bail 213"); }
            return None;
        }
    };
    if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] ext: a-affine"); }
    let (a_off0, a_map) = affine(ctx, &index_offset(ctx, &a_idx))?;
    let (b_off0, b_map) = affine(ctx, &index_offset(ctx, &b_idx))?;
    let (a_slot, a_dt, a_base) = base_of(ctx, &a_idx)?;
    let a_off = a_off0 + a_base;
    let (b_slot, b_dt, b_base) = base_of(ctx, &b_idx)?;
    let b_off = b_off0 + b_base;
    // The library GEMM reads the operand buffers with the contraction dtype.
    // A casted staging buffer (e.g. f32 -> f16 `from_slice`) must stay on the
    // generated kernel.
    if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] ext: ab-dtype"); }
    if (a_dt != mul_dt || b_dt != mul_dt)
        && !(kind == GemmDevice::Vk
            && matches!(
                a_dt,
                crate::dtype::FLOAT32 | crate::dtype::FLOAT16 | crate::dtype::BFLOAT16
            )
            && matches!(
                b_dt,
                crate::dtype::FLOAT32 | crate::dtype::FLOAT16 | crate::dtype::BFLOAT16
            ))
    {
        if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] bail 238"); }
        return None;
    }

    // n is the only range B iterates that A does not, and C stores it
    // contiguously (stride 1). m is the A-only range whose C stride equals the
    // n extent (row-major output). Shared ranges are batch axes.
    let a_keys: Vec<RKey> = a_map.iter().map(|(k, _)| *k).collect();
    let b_keys: Vec<RKey> = b_map.iter().map(|(k, _)| *k).collect();
    let mut n_candidates: Vec<RKey> = b_keys
        .iter()
        .copied()
        .filter(|k| *k != red_k && !a_keys.contains(k) && coeff(&c_map, *k) == 1)
        .collect();
    n_candidates.dedup();
    if n_candidates.len() != 1 {
        if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] bail 253"); }
        return None;
    }
    let n_key = n_candidates[0];
    let n = range_extent(ctx, &st, n_key)?;
    if n <= 1 {
        if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] bail 258"); }
        return None;
    }
    let mut m_candidates: Vec<(RKey, i64)> = a_keys
        .iter()
        .copied()
        .filter(|k| *k != red_k && !b_keys.contains(k))
        .map(|k| (k, coeff(&c_map, k)))
        .filter(|(_, c)| *c >= n)
        .collect();
    m_candidates.dedup();
    // The m axis is the innermost remaining output axis (smallest C stride).
    m_candidates.sort_by_key(|(_, c)| *c);
    if m_candidates.is_empty() || (m_candidates.len() > 1 && m_candidates[0].1 == m_candidates[1].1)
    {
        if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] bail 272"); }
        return None;
    }
    let m_key = m_candidates[0].0;
    let m = range_extent(ctx, &st, m_key)?;
    let k = range_extent(ctx, &st, red_k)?;
    if m <= 0 || k <= 0 {
        if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] bail 278"); }
        return None;
    }

    let a_m = coeff(&a_map, m_key);
    let a_k = coeff(&a_map, red_k);
    let b_k = coeff(&b_map, red_k);
    let b_n = coeff(&b_map, n_key);
    let c_m = coeff(&c_map, m_key);
    let c_n = coeff(&c_map, n_key);
    // The CPU blocked kernel stores rows contiguously; anything else keeps the
    // generated kernel.
    if kind == GemmDevice::Cpu && c_n != 1 {
        if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] bail 290"); }
        return None;
    }
    if c_n != 1 || c_m < n {
        if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] bail 293"); }
        return None;
    }
    // A row-major (k contiguous) or column-major (m contiguous). The leading
    // dimension may be larger than the tight size: ViT's attention operands are
    // transposed views interleaving the head axis into the row stride.
    let a_row = a_k == 1 && a_m >= k;
    let a_col = a_m == 1 && a_k >= m;
    let b_row = b_n == 1 && b_k >= n;
    let b_col = b_k == 1 && b_n >= k;
    if !(a_row || a_col) || !(b_row || b_col) {
        if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] bail 303"); }
        return None;
    }

    // Classify the peeled epilogue operands. A MUL operand is the per-column
    // scale; an ADD operand is the residual (same affine layout as C) or the
    // bias (depends only on the output columns).
    let mut bias: (i64, i64) = (-1, 0);
    let mut bias_dt = c_dt;
    let mut scale: (i64, i64) = (-1, 0);
    let mut residual: (i64, i64) = (-1, 0);
    for e in &extras {
        let idx = operand(ctx, e.expr.clone())?;
        let (off0, map) = affine(ctx, &index_offset(ctx, &idx))?;
        let (slot, dt, base) = base_of(ctx, &idx)?;
        // The Vulkan cooperative-matrix epilogue can read a 16-bit bias for an
        // f32 result (a mixed-precision `Linear`); everything else needs the
        // bias in the result dtype.
        let bias_ok = kind == GemmDevice::Vk
            && matches!(
                dt,
                crate::dtype::FLOAT32 | crate::dtype::FLOAT16 | crate::dtype::BFLOAT16
            );
        if dt != c_dt && !bias_ok {
        if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] bail 326"); }
            return None;
        }
        let off = off0 + base;
        if map.len() == 1 && coeff(&map, n_key) == 1 {
            // A MUL operand is the per-column scale; an ADD operand is additive.
            // Two additive n-only terms (e.g. `bias_ih + bias_hh`) cannot be
            // expressed by this epilogue, so leave the kernel generated.
            if e.from_mul {
                if scale.0 >= 0 {
        if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] bail 335"); }
                    return None;
                }
                scale = (slot, off);
            } else {
                if bias.0 >= 0 {
        if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] bail 340"); }
                    return None;
                }
                bias = (slot, off);
                bias_dt = dt;
            }
            continue;
        }
        // The residual must be elementwise-aligned with the output store.
        let same_layout =
            map.len() == c_map.len() && c_map.iter().all(|(k, v)| coeff(&map, *k) == *v);
        if !same_layout || e.from_mul || residual.0 >= 0 {
        if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] bail 351"); }
            return None;
        }
        residual = (slot, off);
    }
    let (bias_slot, bias_off) = bias;
    let (scale_slot, scale_off) = scale;
    let (res_slot, res_off) = residual;

    // Batch axes: every range in C that is neither m nor n. Fold a contiguous
    // set into one strided batch axis.
    let mut batch: Vec<(RKey, i64, i64, i64, i64)> = Vec::new(); // key, ext, a, b, c
    for (key, _) in c_map.iter() {
        if *key == m_key || *key == n_key {
            continue;
        }
        let ext = range_extent(ctx, &st, *key)?;
        // An extent-1 axis contributes a constant zero; drop it.
        if ext == 1 {
            continue;
        }
        batch.push((
            *key,
            ext,
            coeff(&a_map, *key),
            coeff(&b_map, *key),
            coeff(&c_map, *key),
        ));
    }
    // A and B must not iterate any other range we ignored.
    for (key, _) in a_map.iter().chain(b_map.iter()) {
        if *key != m_key && *key != n_key && *key != red_k && !batch.iter().any(|(k, ..)| k == key)
        {
        if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] bail 383"); }
            return None;
        }
    }
    if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] ext: fold-batches"); }
    let (batch_count, a_b, b_b, c_b) = fold_batches(&mut batch)?;

    // Kernel-only measurements (2026-11, RTX 3060) show cuBLAS winning f32 at
    // the square and batch-8 projection shapes and tying/winning elsewhere;
    // the old "generated wins on clean squares" rule came from readback-
    // inflated numbers. The generated f32 kernel has no shared-memory tiling
    // (register-only, DRAM-bound), so CUDA Auto now prefers cuBLAS for f32.
    // bf16 is a wash (generated 10.7 vs cuBLAS 9.8 TFLOP/s at 2048^3), so the
    // divisibility heuristic stays for 16-bit, CL and the CPU host path.
    // `AUTORUST_CUBLAS=all` still forces the library for every contraction and
    // `=0` keeps the generated kernel.
    let backend = match kind {
        GemmDevice::Cuda => "cuda",
        GemmDevice::Cl => "cl",
        GemmDevice::Metal => "metal",
        GemmDevice::Vk => "vk",
        GemmDevice::Cpu => "cpu",
    };
    let policy = ctx.settings.library_gemm_for(backend);
    let force = policy == crate::settings::LibraryGemm::Always;
    let cuda_f32 = kind == GemmDevice::Cuda && ctx.dt_of(&mul) == crate::dtype::FLOAT32;
    if kind == GemmDevice::Metal {
        // MPS is opt-in (`AUTORUST_MPS=all`): measured on an M2 it is within
        // noise of (sometimes slower than) the generated kernel, while adding
        // matrix/pipeline setup. `Auto` keeps the deterministic generated path.
        // MPS only produces a row-major result, and a fused bias/scale/residual
        // epilogue would need a separate kernel, so those stay generated too.
        if !force || c_n != 1 || bias_slot >= 0 || scale_slot >= 0 || res_slot >= 0 {
        if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] bail 415"); }
            return None;
        }
        // MPS supports f32/f32, f16/f16 and f16 inputs with an f32 result.
        // bfloat16 is not accepted by MPSMatrixMultiplication on macOS.
        let ok_dtypes = matches!(
            (mul_dt, c_dt),
            (crate::dtype::FLOAT32, crate::dtype::FLOAT32)
                | (crate::dtype::FLOAT16, crate::dtype::FLOAT16)
                | (crate::dtype::FLOAT16, crate::dtype::FLOAT32)
        );
        if !ok_dtypes {
        if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] bail 426"); }
            return None;
        }
    }
    // The CPU host GEMM is a *different* implementation, not just a library
    // swap: it blocks and parallelizes, so it wins on every shape where the
    // generated kernel's 1-row tile loses its `B` reuse. Do not apply the
    // divisibility heuristic there; require a minimum amount of work instead.
    if kind == GemmDevice::Cpu {
        #[cfg(any(
            feature = "backend_rust",
            feature = "backend_clang",
            feature = "backend_rayon"
        ))]
        {
            if !force && !crate::runtime::cpu_gemm::worth_dispatching(m, n, k) {
        if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] bail 441"); }
                return None;
            }
        }
        // No host-memory backend is compiled in, so nothing can execute it.
        #[cfg(not(any(
            feature = "backend_rust",
            feature = "backend_clang",
            feature = "backend_rayon"
        )))]
        {
        if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] bail 451"); }
            return None;
        }
    } else if kind == GemmDevice::Vk {
        // The cooperative-matrix module is 2.6-4.6x the generated kernel at these
        // shapes, so the historical "clean multiples of 4 stay generated" rule
        // (measured for cuBLAS/CLBlast) does not apply; `coopmat::eligible` is
        // the filter.
    } else if !force && !cuda_f32 && m % 4 == 0 && n % 4 == 0 && k % 4 == 0 {
        if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] bail 459"); }
        return None;
    }

    if crate::helpers::env::get("AUTORUST_CUBLAS_DEBUG").is_some() {
        eprintln!(
            "CUBLAS m={m} n={n} k={k} batch={batch_count} \
             a=[{a_m},{a_k}] b=[{b_k},{b_n}] c=[{c_m},{c_n}] alpha={alpha} bias={bias_slot} scale={scale_slot} res={res_slot} \
             off=[{a_off},{b_off},{c_off}] slots=[{a_slot},{b_slot},{c_slot}]"
        );
    }
    let desc = HostGemm {
        a_slot,
        b_slot,
        c_slot,
        a_off,
        b_off,
        c_off,
        m,
        n,
        k,
        a_m,
        a_k,
        b_k,
        b_n,
        c_m,
        c_n,
        batch: batch_count,
        a_b,
        b_b,
        c_b,
        dtype: ctx.dt_of(&mul),
        a_dtype: a_dt,
        b_dtype: b_dt,
        c_dtype: c_dt,
        alpha,
        bias_slot,
        bias_off,
        bias_dtype: bias_dt,
        scale_slot,
        scale_off,
        res_slot,
        res_off,
    };
    // The cooperative-matrix module is the only VK library GEMM, so it has to
    // accept the descriptor (f16 operands, f32 accumulate, contiguous row-major,
    // no epilogue) or the kernel must keep its generated form.
    #[cfg(feature = "backend_vk")]
    if kind == GemmDevice::Vk {
        // A plain bf16 dot accumulates in bf16 in the graph, while the MMA
        // accumulates in f32, so that case is a numerics change and rides the
        // same switch as the CUDA tensor-core path (`USE_TC`, on by default).
        let bf16_accum =
            desc.c_dtype == crate::dtype::BFLOAT16 && red_dt == crate::dtype::BFLOAT16;
        let tc_on = ctx.settings.tensor_cores
            && !matches!(
                crate::helpers::env::get("USE_TC").as_deref(),
                Some("0") | Some("false")
            );
        if !crate::runtime::coopmat::eligible(&desc) || (bf16_accum && !tc_on) {
            if crate::helpers::env::get("VK_TRACE").is_some() {
                eprintln!(
                    "[vk] coopmat declined (bf16_accum={bf16_accum} tc={tc_on}): {desc:?}"
                );
            }
        if crate::helpers::env::get("VK_TRACE").is_some() { eprintln!("[vk] bail 523"); }
            return None;
        }
    }
    Some(desc)
}

fn reduce_op(ctx: &UOpCtx, reduce: &UopId) -> Option<Ops> {
    match ctx.arg_of(reduce) {
        Arg::Seq { tuple: true, items } => match items.first()? {
            Arg::Enum(e) => Ops::from_u16(e.value),
            _ => None,
        },
        _ => None,
    }
}

fn const_float(ctx: &UOpCtx, u: &UopId) -> Option<f64> {
    match ctx.arg_of(u) {
        Arg::Float(f) => Some(*f),
        Arg::Int(i) => Some(*i as f64),
        Arg::Const(crate::dtype::ConstVal::Float(f)) => Some(*f),
        Arg::Const(crate::dtype::ConstVal::Int(i)) => Some(*i as f64),
        _ => None,
    }
}

/// Find a range's extent by scanning the (small) store AST for its extent node.
fn range_extent(ctx: &mut UOpCtx, root: &UopId, key: RKey) -> Option<i64> {
    for u in ctx.toposort(root) {
        if range_key(ctx, &u) == Some(key) {
            let ext = ctx.srcs(&u)[0].clone();
            if let Ok(SVal::I(v)) = ctx.ssimplify(ext) {
                return i64::try_from(v).ok();
            }
        }
    }
    None
}

/// Collapse a set of batch ranges into `(count, a_stride, b_stride, c_stride)`.
fn fold_batches(batch: &mut [(RKey, i64, i64, i64, i64)]) -> Option<(i64, i64, i64, i64)> {
    if batch.is_empty() {
        return Some((1, 0, 0, 0));
    }
    // Innermost (smallest C stride) first.
    batch.sort_by_key(|(_, _, _, _, c)| *c);
    let mut count = 1i64;
    for w in batch.windows(2) {
        let (_, e0, a0, b0, c0) = w[0];
        let (_, _, a1, b1, c1) = w[1];
        if c1 != c0.saturating_mul(e0) || a1 != a0.saturating_mul(e0) || b1 != b0.saturating_mul(e0)
        {
            return None;
        }
        count = count.checked_mul(e0)?;
    }
    count = count.checked_mul(batch.last()?.1)?;
    let (_, _, a, b, c) = batch[0];
    Some((count, a, b, c))
}
