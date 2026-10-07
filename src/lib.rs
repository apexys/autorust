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
