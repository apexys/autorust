//! `extra/models/` ports: model definitions built on the `Tensor` front-end.
//!
//! Currently [`transformer`] (`TransformerBlock`/`Transformer`), [`vit`]
//! (`ViT`), and [`flash_attention`] (op-level FA2). Models are plain structs
//! with a hand-written `forward` (like tinygrad's `class Model`), not a generic
//! `Sequential` container. Add new ports here and gate them with goldens under
//! `tests/`.
pub mod flash_attention;
pub mod transformer;
pub mod vit;
