//! Small utilities shared by the port: tinygrad's `ContextVar`-style flags
//! ([`context`]), env-var reading ([`mod@env`]), and the bignum helpers used by
//! symbolic index arithmetic ([`bigint`]).
//!
//! Runtime configuration is read through [`mod@env`] once and cached where the
//! value is hot; build-time diagnostics are Cargo features.
pub mod bigint;
pub mod context;
pub mod env;

pub use bigint::*;
pub use context::*;
pub use env::*;
