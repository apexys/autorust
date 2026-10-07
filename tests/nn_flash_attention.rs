//! FlashAttention vs `scaled_dot_product_attention`: forward and backward must
//! match (`extra/thunder/tiny/fa.py` computes the same softmax attention). Small
//! block sizes force multiple Q/KV blocks, remainders, and mask padding.

use autorust::engine::session::Session;
use autorust::models::flash_attention::flash_attention_blocks;
use autorust::tensor::Tensor;
use autorust::uop::ctx::UOpCtx;

fn close(got: &[f32], want: &[f32], rtol: f32, name: &str) {
    assert_eq!(got.len(), want.len(), "{name}: length {} != {}", got.len(), want.len());
    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        let tol = rtol * b.abs().max(1.0);
        assert!((a - b).abs() <= tol, "{name}[{i}]: got {a}, want {b} (|d|={})", (a - b).abs());
    }
}

fn tensor(ctx: &mut UOpCtx, vals: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_slice(ctx, vals, None, None).reshape(ctx, shape)
}

/// Returns `(out, gq, gk, gv)` for one attention variant.
#[allow(clippy::too_many_arguments)]
fn run(
    flash: bool,
    q: &[f32], qs: &[usize],
    k: &[f32], ks: &[usize],
    v: &[f32], vs: &[usize],
    mask: Option<(&[usize], &[f32], bool)>,
    is_causal: bool,
    gqa: bool,
    bq: i64, bk: i64,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut ctx = UOpCtx::new();
    let mut sess = Session::new();
    let qt = tensor(&mut ctx, q, qs);
    let kt = tensor(&mut ctx, k, ks);
    let vt = tensor(&mut ctx, v, vs);
    let mt = mask.map(|(ms, mv, mb)| {
        let m = tensor(&mut ctx, mv, ms);
        if mb { m.cast(&mut ctx, autorust::dtype::BOOL) } else { m }
    });
    let out = if flash {
        flash_attention_blocks(&qt, &kt, &vt, mt.as_ref(), is_causal, bq, bk, &mut ctx)
    } else {
        qt.scaled_dot_product_attention(&mut ctx, &kt, &vt, mt.as_ref(), 0.0, is_causal, gqa)
    };
    let loss = out.mul(&mut ctx, &out).sum(&mut ctx, None, false, None);
    loss.backward(&mut ctx, &[qt.clone(), kt.clone(), vt.clone()], None);
    let gq = qt.grad(&ctx).unwrap().to_list::<f32>(&mut ctx, &mut sess);
    let gk = kt.grad(&ctx).unwrap().to_list::<f32>(&mut ctx, &mut sess);
    let gv = vt.grad(&ctx).unwrap().to_list::<f32>(&mut ctx, &mut sess);
    let o = out.to_list::<f32>(&mut ctx, &mut sess);
    (o, gq, gk, gv)
}

#[allow(clippy::too_many_arguments)]
fn check(name: &str, q: &[f32], qs: &[usize], k: &[f32], ks: &[usize], v: &[f32], vs: &[usize],
         mask: Option<(&[usize], &[f32], bool)>, is_causal: bool, gqa: bool, bq: i64, bk: i64) {
    let (fo, fgq, fgk, fgv) = run(true, q, qs, k, ks, v, vs, mask, is_causal, gqa, bq, bk);
    let (so, sgq, sgk, sgv) = run(false, q, qs, k, ks, v, vs, mask, is_causal, gqa, bq, bk);
    close(&fo, &so, 1e-4, &format!("{name} forward"));
    close(&fgq, &sgq, 1e-3, &format!("{name} grad_q"));
    close(&fgk, &sgk, 1e-3, &format!("{name} grad_k"));
    close(&fgv, &sgv, 1e-3, &format!("{name} grad_v"));
}

fn seq(shape: &[usize], scale: f32, off: f32) -> Vec<f32> {
    let n: usize = shape.iter().product();
    (0..n).map(|i| i as f32 * scale + off).collect()
}

#[test]
fn flash_basic_and_multihead() {
    let (q, k, v) = (seq(&[2, 3, 8], 0.3, -0.4), seq(&[2, 3, 8], 0.2, -0.1), seq(&[2, 3, 8], 0.3, 0.5));
    check("basic", &q, &[2, 3, 8], &k, &[2, 3, 8], &v, &[2, 3, 8], None, false, false, 2, 2);
    let (q, k, v) = (seq(&[2, 2, 4, 8], 0.3, -0.4), seq(&[2, 2, 4, 8], 0.2, -0.1), seq(&[2, 2, 4, 8], 0.3, 0.5));
    check("multihead", &q, &[2, 2, 4, 8], &k, &[2, 2, 4, 8], &v, &[2, 2, 4, 8], None, false, false, 3, 3);
}

#[test]
fn flash_causal_with_remainder() {
    // nq = nk = 5 with bq=3, bk=2 -> padded grids (6, 6), nontrivial masking.
    let (q, k, v) = (seq(&[1, 5, 8], 0.3, -0.4), seq(&[1, 5, 8], 0.2, -0.1), seq(&[1, 5, 8], 0.3, 0.5));
    check("causal", &q, &[1, 5, 8], &k, &[1, 5, 8], &v, &[1, 5, 8], None, true, false, 3, 2);
}

#[test]
fn flash_masks() {
    let (q, k, v) = (seq(&[2, 4, 4], 0.3, -0.4), seq(&[2, 4, 4], 0.2, -0.1), seq(&[2, 4, 4], 0.3, 0.5));
    let bm = [1.0f32, 1., 0., 0., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 0.];
    check("bool_mask", &q, &[2, 4, 4], &k, &[2, 4, 4], &v, &[2, 4, 4], Some((&[1, 4, 4], &bm, true)), false, false, 3, 3);
    let fm = [0.0f32, 0., -1e9, -1e9, 0., 0., 0., -1e9, 0., 0., 0., 0., 0., 0., 0., -1e9];
    check("float_mask", &q, &[2, 4, 4], &k, &[2, 4, 4], &v, &[2, 4, 4], Some((&[1, 4, 4], &fm, false)), false, false, 3, 3);
}

#[test]
fn flash_gqa() {
    let (q, k, v) = (seq(&[1, 4, 5, 8], 0.3, -0.4), seq(&[1, 2, 5, 8], 0.2, -0.1), seq(&[1, 2, 5, 8], 0.3, 0.5));
    check("gqa", &q, &[1, 4, 5, 8], &k, &[1, 2, 5, 8], &v, &[1, 2, 5, 8], None, false, true, 2, 2);
}

#[test]
fn flash_default_blocks_and_promote() {
    // Default-sized blocks: one block each; also check 3D -> 4D promotion.
    let (q, k, v) = (seq(&[1, 2, 10, 8], 0.3, -0.4), seq(&[1, 2, 10, 8], 0.2, -0.1), seq(&[1, 2, 10, 8], 0.3, 0.5));
    check("default_blocks", &q, &[1, 2, 10, 8], &k, &[1, 2, 10, 8], &v, &[1, 2, 10, 8], None, false, false, autorust::models::flash_attention::Q_BLOCK_SIZE, autorust::models::flash_attention::KV_BLOCK_SIZE);

    let (q, k, v) = (seq(&[3, 5, 8], 0.3, -0.4), seq(&[3, 5, 8], 0.2, -0.1), seq(&[3, 5, 8], 0.3, 0.5));
    let (f3, _, _, _) = run(true, &q, &[3, 5, 8], &k, &[3, 5, 8], &v, &[3, 5, 8], None, false, false, 2, 2);
    let (f4, _, _, _) = run(true, &q, &[1, 3, 5, 8], &k, &[1, 3, 5, 8], &v, &[1, 3, 5, 8], None, false, false, 2, 2);
    close(&f3, &f4, 1e-6, "3d vs 4d");
}
