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
    /// Use it only where the generated kernel loses (clean shapes stay
    /// generated on CUDA/OpenCL; the CPU host path wins above a size floor).
    Auto,
    /// Use it for every eligible contraction.
    Always,
    /// Never use it.
    Never,
}

/// Configuration for one context.
#[derive(Clone, Debug)]
pub struct Settings {
    /// Attention implementation used by `Tensor::attention`.
    pub attention: AttentionKind,
    /// Library-GEMM policy.
    pub library_gemm: LibraryGemm,
    /// Allow TF32 for f32 CUDA GEMMs.
    pub allow_tf32: bool,
    /// CPU packed-GEMM tile `(mc, nc, kc)`, when overridden.
    pub cpu_gemm_tile: Option<(usize, usize, usize)>,
    /// BEAM search width (`0` = off).
    pub beam: usize,
    /// On-disk cache for compiled kernels and tuning results.
    pub cache: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            attention: AttentionKind::Flash,
            library_gemm: LibraryGemm::Auto,
            allow_tf32: false,
            cpu_gemm_tile: None,
            beam: 0,
            cache: true,
        }
    }
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

impl Settings {
    /// Read the process defaults from the environment. Called exactly once by
    /// [`global`].
    pub fn from_env() -> Settings {
        let gemm = env("AUTORUST_GEMM").or_else(|| env("AUTORUST_CUBLAS"));
        let library_gemm = match gemm.as_deref() {
            Some("0") => LibraryGemm::Never,
            Some("all") | Some("force") => LibraryGemm::Always,
            _ => match env("AUTORUST_CLBLAST").as_deref() {
                Some("0") => LibraryGemm::Never,
                _ => LibraryGemm::Auto,
            },
        };
        let attention = match env("AUTORUST_ATTN").as_deref() {
            Some("sdpa") | Some("0") => AttentionKind::Sdpa,
            Some("sage") => AttentionKind::Sage,
            _ => AttentionKind::Flash,
        };
        let allow_tf32 = matches!(
            env("AUTORUST_CUBLAS_TF32")
                .or_else(|| env("ALLOW_TF32"))
                .as_deref(),
            Some("1") | Some("true")
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
        Settings {
            attention,
            library_gemm,
            allow_tf32,
            cpu_gemm_tile,
            beam,
            cache,
        }
    }

    /// Builder: `Settings::default().with_attention(kind)`.
    pub fn with_attention(mut self, attention: AttentionKind) -> Settings {
        self.attention = attention;
        self
    }

    /// Builder: library-GEMM policy.
    pub fn with_library_gemm(mut self, policy: LibraryGemm) -> Settings {
        self.library_gemm = policy;
        self
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
