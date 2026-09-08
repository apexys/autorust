//! `renderer/` — `tinygrad/renderer/`: the `Renderer` facts struct the codegen
//! pipeline consumes, plus the `UopRenderer` seam (facts + render) each
//! backend implements, and the C-style family.

pub mod cstyle;
pub mod expr;

use crate::uop::Ops;

/// `python_alu.keys()` (uop/ops.py:1287) — the interp's native ALU ops.
fn python_alu_keys() -> Vec<Ops> {
    vec![
        Ops::Log2, Ops::Exp2, Ops::Sqrt, Ops::Reciprocal, Ops::Sin, Ops::Pow, Ops::Trunc, Ops::Neg, Ops::Add, Ops::Sub,
        Ops::Mul, Ops::Cmpne, Ops::Cmplt, Ops::Xor, Ops::Or, Ops::And, Ops::Shr, Ops::Shl, Ops::Max, Ops::Cmod, Ops::Cdiv,
        Ops::Floordiv, Ops::Floormod, Ops::Mulacc, Ops::Where, Ops::Cmpeq,
    ]
}

/// `CStyleLanguage.code_for_op.keys()` — the C-style table.
fn cstyle_alu_keys() -> Vec<Ops> {
    vec![
        Ops::Sqrt, Ops::Reciprocal, Ops::Neg, Ops::Exp2, Ops::Log2, Ops::Sin, Ops::Trunc, Ops::And, Ops::Xor, Ops::Or,
        Ops::Add, Ops::Sub, Ops::Mul, Ops::Cmod, Ops::Cdiv, Ops::Cmpne, Ops::Shr, Ops::Shl, Ops::Cmplt, Ops::Where, Ops::Cmpeq,
    ]
}

/// `ClangRenderer.code_for_op` — the C-style table with the transcendental
/// family removed and FDIV added.
fn cstyle_alu_keys_clang() -> Vec<Ops> {
    cstyle_alu_keys()
        .into_iter()
        .filter(|&o| !matches!(o, Ops::Exp2 | Ops::Log2 | Ops::Sin | Ops::Trunc | Ops::Reciprocal))
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
    /// `code_for_op.keys()` — ops this renderer renders natively (everything
    /// else decomposes in the pipeline). python_alu for the interp, the
    /// C-style table for cstyle renderers.
    pub code_for_op: Vec<crate::uop::Ops>,
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
            code_for_op: python_alu_keys(),
        }
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
            code_for_op: cstyle_alu_keys_clang(),
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
            code_for_op: cstyle_alu_keys(),
        }
    }
}

/// The per-device render seam: pipeline facts + linear-uops-to-source text.
/// (python: `Renderer.render` on the device's renderer instance.)
pub trait UopRenderer {
    fn facts(&self) -> &Renderer;
    fn render(&self, ctx: &mut crate::uop::ctx::UOpCtx, lst: &[crate::uop::UopId]) -> String;
}
