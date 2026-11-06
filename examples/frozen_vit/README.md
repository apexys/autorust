# Frozen ViT build-script example

This is a **standalone package** (it has its own `[workspace]`) because a Cargo
`build.rs` cannot import the library of the package it is building. Since the
model is a packaged `autorust::models::vit::ViT`, no model subcrate is needed:
`src/make_vit.rs` is included by both `build.rs` and `main.rs`.

Run it:

```sh
cargo run --release --manifest-path examples/frozen_vit/Cargo.toml
```

The defaults are a small ViT so the build is quick. For ViT-B/16:

```sh
VIT_LAYERS=12 VIT_EMBED=768 VIT_HEADS=12 VIT_IMG=224 VIT_PATCH=16 \
  cargo run --release --manifest-path examples/frozen_vit/Cargo.toml
```

The build script runs the forward once, captures the schedule, emits the Rust
kernels + plan into `OUT_DIR`, and the app links them. No `rustc` or scheduler
is used at run time. `Engine` selects the eager model when a GPU backend is
compiled in/available; otherwise it uses the frozen CPU plan.
