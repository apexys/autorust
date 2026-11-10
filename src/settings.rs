//! Per-context configuration.
//!
//! Every knob is read from the environment **once** ([`global`]), so a process
//! that starts with `AUTORUST_GEMM=0` behaves the same for its whole lifetime
//! and no lowering path pays a `getenv` per kernel. A [`Ctx`] clones the
//! process defaults at creation and can then be reconfigured per context
//! (`ctx.borrow_mut().settings.attention = AttentionKind::Sdpa`), which is what
//! a tuning pass or a per-model policy wants.
//!
//! [`Ctx`]: crate::tensor::Ctx

use std::sync::OnceLock;

use crate::models::flash_attention::AttentionKind;

/// How the library GEMM (cuBLAS / CLBlast / packed CPU kernel) is selected.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LibraryGemm {
    /// Use the library where it wins. On CUDA, f32 now always takes the
    /// library (measured faster at square and batch-8 projection shapes); 16-bit,
    /// OpenCL and the CPU path keep the shape heuristic.
    Auto,
    /// Use it for every eligible contraction.
    Always,
    /// Never use it.
    Never,
}

/// Per-backend library-GEMM policy, so `AUTORUST_CLBLAST=0` does not silently
/// disable cuBLAS (or the CPU path) as well.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LibraryGemmPolicy {
    /// CUDA driver (cuBLAS).
    pub cuda: LibraryGemm,
    /// OpenCL driver (CLBlast).
    pub cl: LibraryGemm,
    /// Apple Metal (MetalPerformanceShaders).
    pub metal: LibraryGemm,
    /// Native CPU backends (packed kernel).
    pub cpu: LibraryGemm,
}

impl LibraryGemmPolicy {
    /// The same policy on every backend.
    pub const fn uniform(p: LibraryGemm) -> LibraryGemmPolicy {
        LibraryGemmPolicy {
            cuda: p,
            cl: p,
            metal: p,
            cpu: p,
        }
    }
}

impl Default for LibraryGemmPolicy {
    fn default() -> Self {
        LibraryGemmPolicy::uniform(LibraryGemm::Auto)
    }
}

/// Configuration for one context.
#[derive(Clone, Debug)]
pub struct Settings {
    /// Attention implementation used by `Tensor::attention`.
    pub attention: AttentionKind,
    /// Library-GEMM policy, per backend.
    pub library_gemm: LibraryGemmPolicy,
    /// Use TF32 (f32 mantissa, 10 bits) for cuBLAS f32 GEMMs.
    ///
    /// **Off by default.** Measured on sm_86, TF32 cuBLAS is within noise of
    /// pedantic f32 at our shapes (789x1536x384 and 2048^3), while it changes
    /// the model output; there is no reason to pay for it. `AUTORUST_CUBLAS_TF32=1`
    /// enables it, and this does not affect the tensor-core attention kernel —
    /// that is governed by [`Settings::tensor_cores`].
    pub allow_tf32: bool,
    /// Allow generated matmuls and hand-written kernels to use warp-level
    /// tensor-core instructions.
    ///
    /// On by default. Only takes effect where the device has the instructions
    /// (`sm_80+`), so older cards and non-NVIDIA backends keep the scalar
    /// kernels. `AUTORUST_TENSOR_CORES=0` forces the scalar path everywhere
    /// (useful for bit-exact comparisons).
    ///
    /// There is no full-precision f32 tensor core: the f32-storage format is
    /// TF32 (10-bit mantissa), so this is independent of
    /// [`Settings::allow_tf32`], which only controls cuBLAS.
    pub tensor_cores: bool,
    /// CPU packed-GEMM tile `(mc, nc, kc)`, when overridden.
    pub cpu_gemm_tile: Option<(usize, usize, usize)>,
    /// BEAM search width (`0` = off).
    pub beam: usize,
    /// On-disk cache for compiled kernels and tuning results.
    pub cache: bool,
    /// `SDPA_TILE`: pad attention to a multiple of this (0 = off).
    pub sdpa_tile: i64,
    /// `AUTORUST_MM_{M,N,K,LM,LN}` overrides for the generated matmul tile.
    pub mm_tile: Option<(usize, usize, usize, usize, usize)>,
    /// `DEFAULT_DEVICE`: device used by tensor creation when none is given.
    pub device: Option<String>,
    /// Double-buffer K/V tiles with `cp.async` in the fused attention kernels.
    ///
    /// On by default (`AUTORUST_ASYNC=0` disables). Used by the bf16 kernel,
    /// where it measures 1.3-1.4x over synchronous staging; the TF32 kernel
    /// keeps synchronous staging by default because there the copy cannot round
    /// to TF32 and the doubled shared memory halves occupancy (measured
    /// neutral-to-worse). `cp.async` needs `sm_80+`, the same gate as the
    /// tensor-core kernels, so older cards are unaffected.
    pub async_pipeline: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            attention: AttentionKind::Flash,
            library_gemm: LibraryGemmPolicy::default(),
            allow_tf32: false,
            tensor_cores: true,
            cpu_gemm_tile: None,
            beam: 0,
            cache: true,
            sdpa_tile: 0,
            mm_tile: None,
            device: None,
            async_pipeline: true,
        }
    }
}

fn env(name: &str) -> Option<String> {
    crate::helpers::env::get(name)
}

impl Settings {
    /// Read the process defaults from the environment. Called exactly once by
    /// [`global`].
    pub fn from_env() -> Settings {
        // Default is "use the library whenever it is there": `Auto` resolves to
        // the library on every backend, and is only downgraded by an explicit
        // opt-out. Per-backend switches win over the device-neutral one.
        fn parse(v: Option<&str>) -> LibraryGemm {
            match v {
                Some("0") | Some("never") => LibraryGemm::Never,
                Some("all") | Some("force") | Some("always") => LibraryGemm::Always,
                _ => LibraryGemm::Auto,
            }
        }
        let neutral = env("AUTORUST_GEMM");
        let mut library_gemm = LibraryGemmPolicy::uniform(parse(neutral.as_deref()));
        if let Some(v) = env("AUTORUST_CUBLAS") {
            library_gemm.cuda = parse(Some(&v));
        }
        if let Some(v) = env("AUTORUST_CLBLAST") {
            library_gemm.cl = parse(Some(&v));
        }
        if let Some(v) = env("AUTORUST_MPS").or_else(|| env("AUTORUST_METAL_GEMM")) {
            library_gemm.metal = parse(Some(&v));
        }
        if let Some(v) = env("AUTORUST_CPU_GEMM") {
            library_gemm.cpu = parse(Some(&v));
        }
        let attention = match env("AUTORUST_ATTN").as_deref() {
            Some("sdpa") | Some("0") => AttentionKind::Sdpa,
            Some("sage") => AttentionKind::Sage,
            _ => AttentionKind::Flash,
        };
        // Opt-in: TF32 cuBLAS is within noise of pedantic f32 at our shapes
        // (measured 789x1536x384 and 2048^3) and changes the model output.
        let allow_tf32 = matches!(
            env("AUTORUST_CUBLAS_TF32")
                .or_else(|| env("ALLOW_TF32"))
                .as_deref(),
            Some("1") | Some("true")
        );
        // Master switch for hand-written tensor-core kernels. Independent of
        // `allow_tf32` so a caller can keep TF32 cuBLAS but demand the exact
        // scalar attention kernel (or the other way round).
        let tensor_cores = !matches!(
            env("AUTORUST_TENSOR_CORES")
                .or_else(|| env("AUTORUST_TC"))
                .as_deref(),
            Some("0") | Some("false")
        );
        let cpu_gemm_tile = match (
            env("AUTORUST_CPU_GEMM_MC").and_then(|v| v.parse().ok()),
            env("AUTORUST_CPU_GEMM_NC").and_then(|v| v.parse().ok()),
            env("AUTORUST_CPU_GEMM_KC").and_then(|v| v.parse().ok()),
        ) {
            (Some(mc), Some(nc), Some(kc)) => Some((mc, nc, kc)),
            _ => None,
        };
        let beam = env("BEAM")
            .or_else(|| env("AUTORUST_BEAM"))
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let cache = env("CACHELEVEL").as_deref() != Some("0");
        let sdpa_tile = env("SDPA_TILE")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(0);
        let mm = |k: &str| env(k).and_then(|v| v.parse::<usize>().ok());
        let mm_tile = match (
            mm("AUTORUST_MM_M"),
            mm("AUTORUST_MM_N"),
            mm("AUTORUST_MM_K"),
            mm("AUTORUST_MM_LM"),
            mm("AUTORUST_MM_LN"),
        ) {
            (Some(m), Some(n), Some(k), Some(lm), Some(ln)) => Some((m, n, k, lm, ln)),
            _ => None,
        };
        let device = env("DEFAULT_DEVICE").filter(|d| !d.is_empty());
        let async_pipeline = !matches!(env("AUTORUST_ASYNC").as_deref(), Some("0") | Some("false"));
        Settings {
            attention,
            library_gemm,
            allow_tf32,
            tensor_cores,
            cpu_gemm_tile,
            beam,
            cache,
            sdpa_tile,
            mm_tile,
            device,
            async_pipeline,
        }
    }

    /// Builder: `Settings::default().with_attention(kind)`.
    pub fn with_attention(mut self, attention: AttentionKind) -> Settings {
        self.attention = attention;
        self
    }

    /// Builder: the same library-GEMM policy on every backend.
    pub fn with_library_gemm(mut self, policy: LibraryGemm) -> Settings {
        self.library_gemm = LibraryGemmPolicy::uniform(policy);
        self
    }

    /// Policy for a backend (`"cuda"`, `"cl"`, `"cpu"`).
    pub fn library_gemm_for(&self, backend: &str) -> LibraryGemm {
        match backend {
            "cuda" => self.library_gemm.cuda,
            "cl" => self.library_gemm.cl,
            "metal" => self.library_gemm.metal,
            _ => self.library_gemm.cpu,
        }
    }

    /// Whether any backend may use a library GEMM.
    pub fn any_library_gemm(&self) -> bool {
        let p = self.library_gemm;
        p.cuda != LibraryGemm::Never
            || p.cl != LibraryGemm::Never
            || p.metal != LibraryGemm::Never
            || p.cpu != LibraryGemm::Never
    }

    /// Builder: BEAM width.
    pub fn with_beam(mut self, width: usize) -> Settings {
        self.beam = width;
        self
    }
}

/// Process defaults, initialised from the environment on first use.
pub fn global() -> &'static Settings {
    static G: OnceLock<Settings> = OnceLock::new();
    G.get_or_init(Settings::from_env)
}
