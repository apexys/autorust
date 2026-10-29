//! `Tensor.dropout` + the global `TRAINING` mode (rand.py:274-296,
//! helpers.py:237). Dropout is inverted (kept × 1/(1-p)) and only active under
//! training; eval mode is the identity.

use autorust::engine::session::Session;
use autorust::helpers::context::with_training;
use autorust::tensor::{Ctx, Tensor};

mod dropout_golden;
use dropout_golden::DROPOUT_GOLDEN;

fn read(t: &Tensor, sess: &mut Session) -> Vec<f32> {
    t.to_list::<f32>(sess)
}

#[test]
fn dropout_matches_tinygrad_goldens() {
    // Exact match vs tinygrad: same Threefry RNG, same masked/inverted scaling.
    for g in DROPOUT_GOLDEN {
        let ctx = Ctx::new();
        let mut sess = Session::new();
        Tensor::manual_seed(g.seed);
        let x = Tensor::ones(&ctx, g.shape.iter().map(|&s| s as i64).collect::<Vec<_>>());
        let y = with_training(true, || x.dropout(g.p));
        let got = read(&y, &mut sess);
        assert_eq!(got.len(), g.out.len(), "{}: length", g.name);
        for (i, (a, b)) in got.iter().zip(g.out.iter()).enumerate() {
            assert_eq!(a, b, "{}[{i}]: got {a}, want {b}", g.name);
        }
    }
}

#[test]
fn dropout_eval_is_identity() {
    // Outside training, dropout must be a no-op regardless of p.
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let x = Tensor::ones(&ctx, [7, 5]);
    for p in [0.0, 0.25, 0.5, 1.0] {
        let y = x.dropout(p);
        assert_eq!(read(&y, &mut sess), vec![1.0f32; 35], "eval dropout p={p}");
    }
}

#[test]
fn dropout_p0_and_p1() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let x = Tensor::ones(&ctx, [4, 4]);
    with_training(true, || {
        let y0 = x.dropout(0.0);
        assert_eq!(read(&y0, &mut sess), vec![1.0f32; 16], "p=0 identity");
        let y1 = x.dropout(1.0);
        assert_eq!(read(&y1, &mut sess), vec![0.0f32; 16], "p=1 all zero");
    });
}

#[test]
fn dropout_training_scales_and_masks() {
    // Input all ones, p=0.5 → every output is 0 or 2, ~half zeros, mean ≈ 1.
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let x = Tensor::ones(&ctx, [200, 50]); // 10_000 elements
    Tensor::manual_seed(1234);
    let y = with_training(true, || x.dropout(0.5));
    let vals = read(&y, &mut sess);

    let zeros = vals.iter().filter(|&&v| v == 0.0).count();
    let twos = vals.iter().filter(|&&v| v == 2.0).count();
    assert_eq!(zeros + twos, vals.len(), "only 0 or 1/(1-p)=2 values");
    let frac_zero = zeros as f64 / vals.len() as f64;
    assert!(
        (0.45..0.55).contains(&frac_zero),
        "zero fraction {frac_zero} should be ~0.5"
    );

    let mean = vals.iter().map(|&v| v as f64).sum::<f64>() / vals.len() as f64;
    assert!(
        (mean - 1.0).abs() < 0.05,
        "inverted-dropout mean {mean} should be ~1.0"
    );
}

#[test]
fn session_training_switch_gates_dropout() {
    // `Session::training(bool)` mirrors the global flag, like `Session::jit`.
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let x = Tensor::ones(&ctx, [1000]);

    sess.training(false);
    assert!(!sess.training_enabled());
    let eval = x.dropout(0.5);
    assert_eq!(read(&eval, &mut sess), vec![1.0f32; 1000], "eval identity");

    sess.training(true);
    assert!(sess.training_enabled());
    let train = x.dropout(0.5);
    let v = read(&train, &mut sess);
    assert!(
        v.contains(&0.0) && v.contains(&2.0),
        "training masks+scales"
    );
}
