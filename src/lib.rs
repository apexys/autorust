//! A faithful Rust port of [tinygrad](https://github.com/tinygrad/tinygrad)
//! v0.14: an interned UOp graph, a rangeify scheduler, a codegen pipeline, and
//! native CPU/OpenCL/CUDA backends.
//!
//! The Python tree is the reference oracle. Rust never imports Python at
//! runtime; the `py/` bridge is a development-only differential harness.
//!
//! # Quick start
//!
//! ```no_run
//! use autorust::engine::session::Session;
//! use autorust::tensor::{Ctx, Tensor};
//!
//! let ctx = Ctx::new();
//! let mut sess = Session::new();
//! let a = Tensor::from_slice(&ctx, &[1.0f32, 2.0, 3.0], None, None);
//! let b = Tensor::from_slice(&ctx, &[10.0f32, 20.0, 30.0], None, None);
//! let c = a.add(&b);
//! assert_eq!(c.to_list::<f32>(&mut sess), vec![11.0, 22.0, 33.0]);
//! ```
//!
//! [`UOpCtx`](uop::ctx::UOpCtx) owns the interned graph; [`Session`](engine::session::Session)
//! owns the realize state (buffers + compiled-program cache + symbolic values).
//! Both are cheap to create but should be reused across steps.
//!
//! # Public surface
//!
//! Stable, user-facing modules:
//!
//! - [`tensor`] — the `Tensor` front-end, `nn` layers, optimizers, autograd.
//! - [`models`] — reference model definitions (ViT, transformer, FlashAttention).
//! - [`dtype`] / [`helpers`] — dtypes and environment/context helpers.
//! - [`engine`] — `Session`, `Jit`, realize, disk cache.
//! - [`runtime`] / [`renderer`] — the backend seams and built-in backends.
//!
//! The lower-level [`uop`], [`schedule`], and [`codegen`] modules are also
//! public for backend authors and differential tooling, but their APIs are
//! explicitly **unstable** in 0.1.x.
//!
//! # Features
//!
//! Default: `capi` + `backend_rayon`. `backend_cl` and `backend_cuda` are
//! opt-in. `--no-default-features` is std-only (plus `nanoserde`). See
//! `Cargo.toml` and the README for the full table.
//!
//! # Threading and panics
//!
//! [`Tensor`](tensor::Tensor) and [`UOpCtx`](uop::ctx::UOpCtx) are `Rc`-based
//! and therefore `!Send`/`!Sync`; the front-end is single-threaded. The RAYON
//! backend parallelizes internally. Most API misuse (shape/axis errors,
//! unsupported ops) panics with a `TGERR:` message, mirroring the Python
//! oracle; file/device/parse APIs are being converted to `Result` as they land.
//!
//! # License
//!
//! MIT. This is a derivative work of tinygrad (MIT); see the `LICENSE` file.
// `UopId`/`Node` carry interior-mutable property caches (`RefCell`), which trips
// `mutable_key_type`; but `Hash`/`Eq` are defined solely by the immutable node
// `id`, so using them as `HashMap` keys is sound.
#![allow(clippy::mutable_key_type)]

pub mod codegen;
pub mod engine;
pub mod renderer;
pub mod dtype;
pub mod helpers;

pub mod models;
pub mod runtime;
pub mod schedule;
pub mod tensor;
pub mod uop;

/// C ABI for the Python `ctypes` bridge (default feature `capi`).
#[cfg(feature = "capi")]
pub mod capi;

/// JSON graph round-trip through the interned graph (used by oracle binaries
/// and the cdylib face).
pub fn rebuild(json: &str) -> Result<String, String> {
    let fg = uop::parse_flat(json)?;
    let g = uop::Graph::from_flat(&fg)?;
    Ok(uop::to_string_flat(&g.to_flat()))
}

/// Rebuild and return root summary (op names + dtype attrs) for quick CLI checks.
pub fn graph_summary(json: &str) -> Result<String, String> {
    let fg = uop::parse_flat(json)?;
    let g = uop::Graph::from_flat(&fg)?;
    let order = g.toposort();
    let mut out = String::new();
    for id in order {
        out.push_str(&format!(
            "{} {} {} srcs={} arg={:?}\n",
            id.id(),
            id.op.pyname(),
            id.dtype.attr(),
            id.srcs.iter().map(|s| s.id().to_string()).collect::<Vec<_>>().join(","),
            id.arg
        ));
    }
    Ok(out)
}
