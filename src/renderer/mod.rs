//! `renderer/` — `tinygrad/renderer/`: the `Renderer` facts struct the codegen
//! pipeline consumes, plus the `UopRenderer` seam (facts + render) each
//! backend implements, and the C-style family.

pub mod cstyle;
pub mod expr;
pub mod extra;
pub mod tc;

use crate::uop::Ops;

/// `python_alu.keys()` (uop/ops.py:1287) — the interp's native ALU ops.
fn python_alu_keys() -> Vec<Ops> {
    vec![
        Ops::Log2,
        Ops::Exp2,
        Ops::Sqrt,
        Ops::Reciprocal,
        Ops::Sin,
        Ops::Pow,
        Ops::Trunc,
        Ops::Neg,
        Ops::Add,
        Ops::Sub,
        Ops::Mul,
        Ops::Cmpne,
        Ops::Cmplt,
        Ops::Xor,
        Ops::Or,
        Ops::And,
        Ops::Shr,
        Ops::Shl,
        Ops::Max,
        Ops::Cmod,
        Ops::Cdiv,
        Ops::Floordiv,
        Ops::Floormod,
        Ops::Mulacc,
        Ops::Where,
        Ops::Cmpeq,
    ]
}

/// `CStyleLanguage.code_for_op.keys()` — the C-style table.
fn cstyle_alu_keys() -> Vec<Ops> {
    // python cstyle.py:136 — note NO MAX (max decomposes via WHERE)
    vec![
        Ops::Sqrt,
        Ops::Reciprocal,
        Ops::Neg,
        Ops::Exp2,
        Ops::Log2,
        Ops::Sin,
        Ops::Trunc,
        Ops::And,
        Ops::Xor,
        Ops::Or,
        Ops::Add,
        Ops::Sub,
        Ops::Mul,
        Ops::Cmod,
        Ops::Cdiv,
        Ops::Cmpne,
        Ops::Shr,
        Ops::Shl,
        Ops::Cmplt,
        Ops::Where,
        Ops::Cmpeq,
    ]
}

/// `ClangRenderer.code_for_op` — the C-style table with the transcendental
/// family removed and FDIV added.
fn cstyle_alu_keys_clang() -> Vec<Ops> {
    cstyle_alu_keys()
        .into_iter()
        .filter(|&o| {
            !matches!(
                o,
                Ops::Exp2 | Ops::Log2 | Ops::Sin | Ops::Trunc | Ops::Reciprocal
            )
        })
        .chain([Ops::Fdiv])
        .collect()
}

/// `Renderer` (renderer/__init__.py) — the renderer facts the codegen
/// pipeline and opts consume (`has_local`, dim maxima, `code_for_op` keys).
#[derive(Clone)]
pub struct Renderer {
    pub device: String,
    pub has_local: bool,
    pub has_shared: bool,
    pub has_threads: bool,
    pub global_max: Option<[i64; 3]>,
    pub local_max: Option<[i64; 3]>,
    pub shared_max: i64,
    /// `tensor_cores` (renderer/__init__.py:72) — the WMMA descriptions this
    /// renderer supports (empty for the interp/CL/clang backends).
    pub tensor_cores: Vec<tc::TensorCore>,
    /// `code_for_op.keys()` — ops this renderer renders natively (everything
    /// else decomposes in the pipeline). python_alu for the interp, the
    /// C-style table for cstyle renderers.
    pub code_for_op: Vec<crate::uop::Ops>,
    /// `supported_dtypes()` — the dtype set the backend computes in.
    pub supported_dtypes: Vec<crate::dtype::DTypeId>,
    /// `renderer.extra_matcher` (codegen/__init__.py:377) — renderer-specific
    /// rewrites run in the pipeline's "final rewrite" stage (e.g. bf16
    /// compute-through-f32). Empty for the interpreter.
    pub extra_matcher: crate::uop::upat::PatternMatcher,
}

impl Renderer {
    /// `PythonRenderer` for `Target.parse("cpu")` — base Renderer defaults.
    pub fn cpu_interp() -> Renderer {
        Renderer {
            device: "INTERP".to_string(),
            has_local: true,
            has_shared: true,
            has_threads: false,
            global_max: Some([0x8FFFFFFF, 0x8FFFFFFF, 0x8FFFFFFF]),
            local_max: Some([0x8FFFFFFF, 0x8FFFFFFF, 0x8FFFFFFF]),
            shared_max: 32768,
            tensor_cores: Vec::new(),
            code_for_op: python_alu_keys(),
            // ops_python.py:235 — 3.12+ keeps half, everything else stays.
            supported_dtypes: crate::dtype::ALL.to_vec(),
            // PythonRenderer.extra_matcher is None (empty).
            extra_matcher: Default::default(),
        }
    }

    /// `ops_rayon` (`RAYON`) — the multithreaded CPU backend facts. For now the
    /// same capabilities as the interpreter (it composes the interpreter until
    /// the range-driven fast path lands, §9.8); only the device label differs.
    pub fn rayon() -> Renderer {
        let mut r = Renderer::cpu_interp();
        r.device = "RAYON".to_string();
        r
    }

    /// `ClangRenderer` — has_local=False, global_max=(NUM_CPU_THREADS, 0, 0).
    pub fn clang() -> Renderer {
        Renderer {
            device: "CPU".to_string(),
            has_local: false,
            has_shared: true,
            has_threads: false,
            global_max: Some([1024, 0, 0]),
            local_max: Some([0x8FFFFFFF, 0x8FFFFFFF, 0x8FFFFFFF]),
            shared_max: 32768,
            tensor_cores: Vec::new(),
            code_for_op: cstyle_alu_keys_clang(),
            // cstyle.py:300 — fp8s excluded everywhere; bf16 only on x86/arm arch.
            supported_dtypes: crate::dtype::ALL
                .iter()
                .cloned()
                .filter(|&d| !in_fp8s(d))
                .collect(),
            extra_matcher: extra::clang_extra_matcher(),
        }
    }

    /// `OpenCLRenderer` — base defaults, OpenCL C flavor.
    pub fn opencl() -> Renderer {
        Renderer {
            device: "CL".to_string(),
            has_local: true,
            has_shared: true,
            has_threads: false,
            global_max: Some([0x8FFFFFFF, 0x8FFFFFFF, 0x8FFFFFFF]),
            local_max: Some([0x8FFFFFFF, 0x8FFFFFFF, 0x8FFFFFFF]),
            shared_max: 32768,
            tensor_cores: Vec::new(),
            code_for_op: cstyle_alu_keys(),
            supported_dtypes: opencl_supported_dtypes(""),
            // cstyle.py:319 — bf16 compute-through-f32 + manual bit casts.
            extra_matcher: extra::create_non_native_float_pats(&[crate::dtype::BFLOAT16], true)
                .add(&extra::pm_manual_bf16_cast()),
        }
    }

    /// `OpenCLRenderer.supported_dtypes` (cstyle.py:339) — fp8s never, half
    /// only with `cl_khr_fp16` in the arch, double only with `cl_khr_fp64`.
    pub fn opencl_arch(arch: &str) -> Renderer {
        let mut r = Renderer::opencl();
        r.supported_dtypes = opencl_supported_dtypes(arch);
        r
    }

    /// `CUDARenderer` (cstyle.py:397) — native bf16 on sm_80+, half on sm_53+,
    /// fp8e4m3/e5m2 on sm_89+; fp8_fnuz never.
    pub fn cuda(arch: &str) -> Renderer {
        Renderer {
            device: "CUDA".to_string(),
            has_local: true,
            has_shared: true,
            has_threads: false,
            global_max: Some([2147483647, 65535, 65535]),
            local_max: Some([1024, 1024, 64]),
            shared_max: 49152,
            tensor_cores: tc::get_cuda(arch),
            code_for_op: cstyle_alu_keys(),
            supported_dtypes: cuda_supported_dtypes(arch),
            extra_matcher: extra::cuda_extra_matcher(),
        }
    }
}

/// cstyle.py:470 — the CUDA dtype gate over the SM version.
fn cuda_supported_dtypes(arch: &str) -> Vec<crate::dtype::DTypeId> {
    let ver: i64 = arch
        .strip_prefix("sm_")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    crate::dtype::ALL
        .iter()
        .cloned()
        .filter(|&d| {
            let half = d == crate::dtype::FLOAT16;
            let bf16 = d == crate::dtype::BFLOAT16;
            let fp8_ocp = crate::dtype::in_set(crate::dtype::FP8_OCP, d);
            let fp8_fnuz = crate::dtype::in_set(crate::dtype::FP8_FNUZ, d);
            !fp8_fnuz && (!half || ver >= 53) && (!bf16 || ver >= 80) && (!fp8_ocp || ver >= 89)
        })
        .collect()
}

fn in_fp8s(d: crate::dtype::DTypeId) -> bool {
    crate::dtype::in_set(crate::dtype::FP8S, d)
}

/// cstyle.py:339 — the CL dtype gate over the device's extension arch string.
fn opencl_supported_dtypes(arch: &str) -> Vec<crate::dtype::DTypeId> {
    crate::dtype::ALL
        .iter()
        .cloned()
        .filter(|&d| {
            let fp8 = in_fp8s(d);
            let half = d == crate::dtype::FLOAT16;
            let dbl = d == crate::dtype::FLOAT64;
            !fp8 && (!half || arch.contains("cl_khr_fp16"))
                && (!dbl || arch.contains("cl_khr_fp64"))
        })
        .collect()
}

/// The per-device render seam: pipeline facts + linear-uops-to-source text.
/// (python: `Renderer.render` on the device's renderer instance.)
pub trait UopRenderer {
    fn facts(&self) -> &Renderer;
    fn render(&self, ctx: &mut crate::uop::ctx::UOpCtx, lst: &[crate::uop::UopId]) -> String;
}
