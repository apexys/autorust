//! Device autotune: the global default can be replaced, the benchmark ranks the
//! available backends, and the winner becomes the default.

use autorust::runtime::autotune;
use autorust::tensor::creation::default_device;

#[test]
fn autotune_selects_a_default() {
    // `Device.DEFAULT = name` semantics: the setter canonicalizes and wins over
    // the priority pick for subsequent tensor creation.
    autorust::set_default_device("rayon");
    assert_eq!(default_device(), "RAYON");

    let available = autotune::available_devices();
    assert!(
        !available.iter().any(|d| d == "INTERP" || d == "CPU"),
        "the interpreter must not be a candidate: {available:?}"
    );
    if cfg!(feature = "backend_rayon") {
        assert!(available.contains(&"RAYON".to_string()), "{available:?}");
    }

    // The suite is sized for release builds; skip the measurement in a debug
    // `cargo test` (the plumbing above is the part that should always run).
    if cfg!(debug_assertions) {
        return;
    }
    let ranked = autotune::rank_devices();
    if ranked.is_empty() {
        // No accelerated backend on this machine: the priority fallback stands.
        return;
    }
    for pair in ranked.windows(2) {
        assert!(
            pair[0].millis <= pair[1].millis,
            "rank not sorted: {ranked:?}"
        );
    }
    for entry in &ranked {
        assert!(entry.millis.is_finite() && entry.millis > 0.0, "{entry:?}");
        assert_eq!(entry.cases.len(), 3, "{entry:?}");
        let sum: f64 = entry.cases.iter().map(|c| c.millis).sum();
        assert!((sum - entry.millis).abs() < 1e-9, "{entry:?}");
    }

    let chosen = autorust::select_fastest_device();
    assert!(
        ranked.iter().any(|entry| entry.device == chosen),
        "chose {chosen} not in {ranked:?}"
    );
    assert_eq!(default_device(), chosen);
}
