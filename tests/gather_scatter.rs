//! Fast gather / interpolation / segment-sum / scatter-add paths.
//!
//! These exercise the public APIs in `src/tensor/indexing_fast.rs`: forward
//! numerics against a naive host reference, autograd (analytic + finite
//! differences), determinism, and the CL backend when a device is present.

use autorust::engine::session::Session;
use autorust::tensor::{Ctx, Tensor};

fn f32s(ctx: &Ctx, data: &[f32]) -> Tensor {
    Tensor::from_slice(ctx, data, None, None)
}
fn i32s(ctx: &Ctx, data: &[i32]) -> Tensor {
    Tensor::from_slice(ctx, data, None, None)
}

fn check(got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(
        got.len(),
        want.len(),
        "len {} != {} (got {got:?}, want {want:?})",
        got.len(),
        want.len()
    );
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!(
            (a - b).abs() <= tol,
            "[{i}] {a} != {b} (got {got:?}, want {want:?})"
        );
    }
}

/// Relative comparison for finite-difference checks (single-precision f32).
fn check_close(got: &[f32], want: &[f32]) {
    assert_eq!(
        got.len(),
        want.len(),
        "len {} != {} (got {got:?}, want {want:?})",
        got.len(),
        want.len()
    );
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        let tol = 5e-3 * a.abs().max(b.abs()) + 1e-4;
        assert!(
            (a - b).abs() <= tol,
            "[{i}] {a} != {b} (got {got:?}, want {want:?})"
        );
    }
}

fn naive_gather_1d(src: &[f32], idx: &[i32]) -> Vec<f32> {
    idx.iter()
        .map(|&i| {
            let n = src.len() as i32;
            let j = if i < 0 { i + n } else { i };
            if j >= 0 && j < n {
                src[j as usize]
            } else {
                0.0
            }
        })
        .collect()
}

fn naive_gather_dim0(src: &[f32], rows: usize, cols: usize, idx: &[i32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(idx.len() * cols);
    for &r in idx {
        let rr = if r < 0 { r + rows as i32 } else { r };
        for c in 0..cols {
            if rr >= 0 && (rr as usize) < rows {
                out.push(src[rr as usize * cols + c]);
            } else {
                out.push(0.0);
            }
        }
    }
    out
}

#[test]
fn gather_direct_1d_negative_and_oob() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let src = f32s(&ctx, &[10.0, 11.0, 12.0, 13.0]);
    let idx = i32s(&ctx, &[0, 3, 3, -1, -4, 4, -5, 2]);
    let got = src.gather_direct(0, &idx).to_list::<f32>(&mut sess);
    check(
        &got,
        &naive_gather_1d(&[10.0, 11.0, 12.0, 13.0], &[0, 3, 3, -1, -4, 4, -5, 2]),
        0.0,
    );
}

#[test]
fn gather_direct_1d_index_any_shape() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let src = f32s(&ctx, &[10.0, 11.0, 12.0, 13.0]);
    let idx = i32s(&ctx, &[1, 0, 2, 3, 1, 0]).reshape([2, 3]);
    let got = src.gather_direct(0, &idx);
    assert_eq!(got.shape().ints(), Some(vec![2, 3]));
    check(
        &got.to_list::<f32>(&mut sess),
        &naive_gather_1d(&[10.0, 11.0, 12.0, 13.0], &[1, 0, 2, 3, 1, 0]),
        0.0,
    );
}

#[test]
fn gather_direct_2d_dim0() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let src: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();
    let t = f32s(&ctx, &src).reshape([3, 4]);
    let idx = i32s(&ctx, &[0, 2, -1, 5, -4, 1]).reshape([2, 3]);
    let got = t.gather_direct(0, &idx);
    assert_eq!(got.shape().ints(), Some(vec![2, 3, 4]));
    check(
        &got.to_list::<f32>(&mut sess),
        &naive_gather_dim0(&src, 3, 4, &[0, 2, -1, 5, -4, 1]),
        0.0,
    );
}

#[test]
fn gather_direct_2d_dim1() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let src: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();
    let t = f32s(&ctx, &src).reshape([3, 4]);
    let idx = i32s(&ctx, &[0, 3, -1, 5, 2, 1]).reshape([3, 2]);
    let got = t.gather_direct(1, &idx);
    assert_eq!(got.shape().ints(), Some(vec![3, 2]));
    let want = vec![
        src[0], src[3], // row 0: col 0, col 3
        src[7], 0.0, // row 1: col -1 -> 3, col 5 oob
        src[10], src[9], // row 2: col 2, col 1
    ];
    check(&got.to_list::<f32>(&mut sess), &want, 0.0);
}

#[test]
fn gather_direct_empty_and_large_index() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let src = f32s(&ctx, &[1.0, 2.0, 3.0]);
    let empty = i32s(&ctx, &[]);
    assert!(
        src.gather_direct(0, &empty)
            .to_list::<f32>(&mut sess)
            .is_empty()
    );

    let big_src: Vec<f32> = (0..2000).map(|i| i as f32).collect();
    let t = f32s(&ctx, &big_src);
    let mut idx_v = Vec::with_capacity(10_000);
    for i in 0..10_000i32 {
        idx_v.push(match i % 7 {
            0 => i % 2000,
            1 => -(i % 2000) - 1,
            2 => 2000 + i,
            3 => -3000 - i,
            _ => (i * 37) % 2000,
        });
    }
    let idx = i32s(&ctx, &idx_v);
    let got = t.gather_direct(0, &idx).to_list::<f32>(&mut sess);
    check(&got, &naive_gather_1d(&big_src, &idx_v), 0.0);
}

fn naive_interp(atlas: &[f32], rows: usize, cols: usize, row: i32, q: f32) -> (f32, f32) {
    if row < 0 || row as usize >= rows {
        return (0.0, 0.0);
    }
    let s = cols as i32;
    let i = (q.floor() as i32).clamp(0, s - 2);
    let f = if q <= 0.0 {
        0.0
    } else if q >= (s - 1) as f32 {
        1.0
    } else {
        q - i as f32
    };
    let a = atlas[row as usize * cols + i as usize];
    let b = atlas[row as usize * cols + i as usize + 1];
    (a * (1.0 - f) + b * f, b - a)
}

#[test]
fn gather_interp_matches_naive() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let src: Vec<f32> = (0..24).map(|i| (i as f32 * 0.37).sin()).collect();
    let atlas = f32s(&ctx, &src).reshape([3, 8]);
    let rows_v = [0i32, 2, -1, 9, 1, 0, 2, 1];
    let q_v = [0.0f32, 7.0, 3.5, 2.0, 1.0, -2.0, 9.5, 6.999];
    let rows = i32s(&ctx, &rows_v);
    let q = f32s(&ctx, &q_v);
    let (value, slope) = atlas.gather_interp(&rows, &q);
    let mut want_v = Vec::new();
    let mut want_s = Vec::new();
    for (&r, &x) in rows_v.iter().zip(&q_v) {
        let (v, s) = naive_interp(&src, 3, 8, r, x);
        want_v.push(v);
        want_s.push(s);
    }
    check(&value.to_list::<f32>(&mut sess), &want_v, 1e-6);
    check(&slope.to_list::<f32>(&mut sess), &want_s, 1e-6);
}

#[test]
fn gather_interp_empty_and_q_edges() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let atlas = f32s(&ctx, &[0.0, 1.0, 2.0, 3.0]).reshape([2, 2]);
    let empty_r = i32s(&ctx, &[]);
    let empty_q = f32s(&ctx, &[]);
    let (v, s) = atlas.gather_interp(&empty_r, &empty_q);
    assert!(v.to_list::<f32>(&mut sess).is_empty());
    assert!(s.to_list::<f32>(&mut sess).is_empty());

    let rows = i32s(&ctx, &[0, 0, 1]);
    // q <= 0 -> (0,0); q == S-1 -> (S-2,1); q above -> clamped
    let q = f32s(&ctx, &[-1.0, 1.0, 5.0]);
    let (v, s) = atlas.gather_interp(&rows, &q);
    check(&v.to_list::<f32>(&mut sess), &[0.0, 1.0, 3.0], 1e-6);
    check(&s.to_list::<f32>(&mut sess), &[1.0, 1.0, 1.0], 1e-6);
}

#[test]
fn segment_sum_cases() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let vals = f32s(&ctx, &[1.0, 2.0, 3.0, 4.0, 5.0]);
    // empty segments at both ends and in the middle
    let ptr = i32s(&ctx, &[0, 0, 2, 2, 5, 5]);
    let got = Tensor::segment_sum(&ptr, &vals).to_list::<f32>(&mut sess);
    check(&got, &[0.0, 3.0, 0.0, 12.0, 0.0], 0.0);

    // one segment
    let ptr = i32s(&ctx, &[0, 5]);
    check(
        &Tensor::segment_sum(&ptr, &vals).to_list::<f32>(&mut sess),
        &[15.0],
        0.0,
    );

    // all singleton segments
    let ptr = i32s(&ctx, &[0, 1, 2, 3, 4, 5]);
    check(
        &Tensor::segment_sum(&ptr, &vals).to_list::<f32>(&mut sess),
        &[1.0, 2.0, 3.0, 4.0, 5.0],
        0.0,
    );

    // 1024 values exercises the split-cumsum path (E > 512)
    let long: Vec<f32> = (0..1024).map(|i| (i % 13) as f32).collect();
    let long_t = f32s(&ctx, &long);
    let ptr = i32s(&ctx, &[0, 10, 10, 500, 1024]);
    let got = Tensor::segment_sum(&ptr, &long_t).to_list::<f32>(&mut sess);
    let sums = [
        long[..10].iter().sum::<f32>(),
        0.0,
        long[10..500].iter().sum::<f32>(),
        long[500..].iter().sum::<f32>(),
    ];
    check(&got, &sums, 1e-3);
}

#[test]
#[should_panic(expected = "nondecreasing")]
fn segment_sum_rejects_nonmonotone() {
    let ctx = Ctx::new();
    let vals = f32s(&ctx, &[1.0, 2.0]);
    let ptr = i32s(&ctx, &[0, 2, 1, 2]);
    let _ = Tensor::segment_sum(&ptr, &vals);
}

#[test]
#[should_panic(expected = "last row_ptr")]
fn segment_sum_rejects_bad_total() {
    let ctx = Ctx::new();
    let vals = f32s(&ctx, &[1.0, 2.0]);
    let ptr = i32s(&ctx, &[0, 1, 1]);
    let _ = Tensor::segment_sum(&ptr, &vals);
}

fn naive_scatter(len: usize, idx: &[i32], vals: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; len];
    for (&i, &v) in idx.iter().zip(vals) {
        if i >= 0 && (i as usize) < len {
            out[i as usize] += v;
        }
    }
    out
}

#[test]
fn scatter_add_repeated_shuffled_and_oob() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let idx_v = [2i32, 0, 2, 2, -1, 7, 0, 1];
    let val_v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let idx = i32s(&ctx, &idx_v);
    let vals = f32s(&ctx, &val_v);
    let got = Tensor::scatter_add(4, &idx, &vals).to_list::<f32>(&mut sess);
    check(&got, &naive_scatter(4, &idx_v, &val_v), 0.0);

    // A shuffled input order must give the same totals.
    let idx2_v = [0i32, 1, 0, 2, 7, -1, 2, 2];
    let idx2 = i32s(&ctx, &idx2_v);
    let vals2 = f32s(&ctx, &val_v);
    let got2 = Tensor::scatter_add(4, &idx2, &vals2).to_list::<f32>(&mut sess);
    check(&got2, &naive_scatter(4, &idx2_v, &val_v), 0.0);
}

#[test]
fn scatter_add_is_deterministic() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let idx_v: Vec<i32> = (0..4096).map(|i| (i * 37) % 257).collect();
    let val_v: Vec<f32> = (0..4096).map(|i| ((i * 13) % 97) as f32 * 0.01).collect();
    let idx = i32s(&ctx, &idx_v);
    let vals = f32s(&ctx, &val_v);
    let a = Tensor::scatter_add(257, &idx, &vals).to_list::<f32>(&mut sess);
    for _ in 0..4 {
        let b = Tensor::scatter_add(257, &idx, &vals).to_list::<f32>(&mut sess);
        assert_eq!(a, b, "scatter_add changed across identical runs");
    }
    check(&a, &naive_scatter(257, &idx_v, &val_v), 1e-2);
}

#[test]
fn scatter_add_dynamic_index_fallback() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let base = i32s(&ctx, &[2, 0, 2, 5, -1]);
    // `+0` makes the index a graph value rather than a static leaf.
    let idx = base.add(0);
    let vals = f32s(&ctx, &[1.0, 2.0, 3.0, 4.0, 5.0]);
    let got = Tensor::scatter_add(4, &idx, &vals).to_list::<f32>(&mut sess);
    check(
        &got,
        &naive_scatter(4, &[2, 0, 2, 5, -1], &[1.0, 2.0, 3.0, 4.0, 5.0]),
        0.0,
    );
}

#[test]
fn scatter_add_dynamic_sorted_path() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let len = 16_384usize;
    let e = 1000usize;
    let idx_v: Vec<i32> = (0..e)
        .map(|i| match i % 17 {
            0 => -((i % 31) as i32 + 1),
            1 => len as i32 + (i % 19) as i32,
            _ => ((i * 37) % len) as i32,
        })
        .collect();
    let val_v: Vec<f32> = (0..e).map(|i| ((i * 13) % 97) as f32 * 0.01).collect();
    let idx = i32s(&ctx, &idx_v).add(0);
    let vals = f32s(&ctx, &val_v);
    let want = naive_scatter(len, &idx_v, &val_v);
    let got = Tensor::scatter_add(len as i64, &idx, &vals).to_list::<f32>(&mut sess);
    check(&got, &want, 1e-3);
    for _ in 0..2 {
        let again = Tensor::scatter_add(len as i64, &idx, &vals).to_list::<f32>(&mut sess);
        assert_eq!(got, again, "dynamic sorted scatter_add changed across runs");
    }
}

// ---- autograd ----

fn numeric_grad<F: Fn(&[f32]) -> f32>(x: &[f32], f: F) -> Vec<f32> {
    let h = 1e-3f32;
    let mut g = Vec::with_capacity(x.len());
    for i in 0..x.len() {
        let mut a = x.to_vec();
        let mut b = x.to_vec();
        a[i] += h;
        b[i] -= h;
        g.push((f(&a) - f(&b)) / (2.0 * h));
    }
    g
}

#[test]
fn grad_gather_direct_wrt_source() {
    let idx_v = [2i32, 0, 2, 7, -1];
    let w_v = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let src_v = [1.0f32, 2.0, 3.0, 4.0];
    let analytic = {
        let ctx = Ctx::new();
        let src = f32s(&ctx, &src_v);
        let idx = i32s(&ctx, &idx_v);
        let w = f32s(&ctx, &w_v);
        let loss = src.gather_direct(0, &idx).mul(&w).sum(None, false, None);
        let g = loss.gradient(&[src], None);
        g[0].to_list_fresh::<f32>()
    };
    let numeric = numeric_grad(&src_v, |x| {
        let ctx = Ctx::new();
        let src = f32s(&ctx, x);
        let idx = i32s(&ctx, &idx_v);
        let w = f32s(&ctx, &w_v);
        src.gather_direct(0, &idx)
            .mul(&w)
            .sum(None, false, None)
            .to_list_fresh::<f32>()[0]
    });
    check_close(&analytic, &numeric);
    // explicit expectation: idx 2 gets w0+w2=4, idx 0 gets w1=2, -1->3 gets w4=5
    check(&analytic, &[2.0, 0.0, 4.0, 5.0], 1e-4);
}

#[test]
fn grad_gather_interp_wrt_q_and_atlas() {
    let src: Vec<f32> = (0..12).map(|i| (i as f32 * 0.7).sin()).collect();
    let rows_v = [0i32, 2, 1];
    let q_v = [0.3f32, 1.7, 2.4];
    let wv = [1.0f32, 2.0, 3.0];
    let ws = [0.5f32, -1.0, 2.0];
    let (gq, ga) = {
        let ctx = Ctx::new();
        let atlas = f32s(&ctx, &src).reshape([3, 4]);
        let rows = i32s(&ctx, &rows_v);
        let q = f32s(&ctx, &q_v);
        let wv = f32s(&ctx, &wv);
        let ws = f32s(&ctx, &ws);
        let (v, s) = atlas.gather_interp(&rows, &q);
        let loss = v
            .mul(&wv)
            .sum(None, false, None)
            .add(s.mul(&ws).sum(None, false, None));
        let gs = loss.gradient(&[q, atlas], None);
        (gs[0].to_list_fresh::<f32>(), gs[1].to_list_fresh::<f32>())
    };
    // VJP w.r.t. q is grad_value * slope.
    let expected_q: Vec<f32> = rows_v
        .iter()
        .zip(&q_v)
        .zip(&wv)
        .map(|((&r, &qq), &w)| w * naive_interp(&src, 3, 4, r, qq).1)
        .collect();
    check(&gq, &expected_q, 1e-4);

    let numeric_q = numeric_grad(&q_v, |x| {
        let ctx = Ctx::new();
        let atlas = f32s(&ctx, &src).reshape([3, 4]);
        let rows = i32s(&ctx, &rows_v);
        let q = f32s(&ctx, x);
        let wv = f32s(&ctx, &wv);
        let ws = f32s(&ctx, &ws);
        let (v, s) = atlas.gather_interp(&rows, &q);
        v.mul(&wv).sum(None, false, None).to_list_fresh::<f32>()[0]
            + s.mul(&ws).sum(None, false, None).to_list_fresh::<f32>()[0]
    });
    check_close(&gq, &numeric_q);

    let numeric_a = numeric_grad(&src, |x| {
        let ctx = Ctx::new();
        let atlas = f32s(&ctx, x).reshape([3, 4]);
        let rows = i32s(&ctx, &rows_v);
        let q = f32s(&ctx, &q_v);
        let wv = f32s(&ctx, &wv);
        let ws = f32s(&ctx, &ws);
        let (v, s) = atlas.gather_interp(&rows, &q);
        v.mul(&wv).sum(None, false, None).to_list_fresh::<f32>()[0]
            + s.mul(&ws).sum(None, false, None).to_list_fresh::<f32>()[0]
    });
    check_close(&ga, &numeric_a);
}

#[test]
fn grad_segment_sum_wrt_values() {
    let vals_v = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let ptr_v = [0i32, 2, 2, 5];
    let w_v = [10.0f32, 20.0, 30.0];
    let analytic = {
        let ctx = Ctx::new();
        let vals = f32s(&ctx, &vals_v);
        let ptr = i32s(&ctx, &ptr_v);
        let w = f32s(&ctx, &w_v);
        let loss = Tensor::segment_sum(&ptr, &vals)
            .mul(&w)
            .sum(None, false, None);
        loss.gradient(&[vals], None)[0].to_list_fresh::<f32>()
    };
    let numeric = numeric_grad(&vals_v, |x| {
        let ctx = Ctx::new();
        let vals = f32s(&ctx, x);
        let ptr = i32s(&ctx, &ptr_v);
        let w = f32s(&ctx, &w_v);
        Tensor::segment_sum(&ptr, &vals)
            .mul(&w)
            .sum(None, false, None)
            .to_list_fresh::<f32>()[0]
    });
    check_close(&analytic, &numeric);
    check(&analytic, &[10.0, 10.0, 30.0, 30.0, 30.0], 1e-4);
}

#[test]
fn grad_scatter_add_wrt_values() {
    let idx_v = [2i32, 0, 2, 7, -1];
    let vals_v = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let w_v = [10.0f32, 20.0, 30.0, 40.0];
    let analytic = {
        let ctx = Ctx::new();
        let idx = i32s(&ctx, &idx_v);
        let vals = f32s(&ctx, &vals_v);
        let w = f32s(&ctx, &w_v);
        let loss = Tensor::scatter_add(4, &idx, &vals)
            .mul(&w)
            .sum(None, false, None);
        loss.gradient(&[vals], None)[0].to_list_fresh::<f32>()
    };
    let numeric = numeric_grad(&vals_v, |x| {
        let ctx = Ctx::new();
        let idx = i32s(&ctx, &idx_v);
        let vals = f32s(&ctx, x);
        let w = f32s(&ctx, &w_v);
        Tensor::scatter_add(4, &idx, &vals)
            .mul(&w)
            .sum(None, false, None)
            .to_list_fresh::<f32>()[0]
    });
    check_close(&analytic, &numeric);
    check(&analytic, &[30.0, 10.0, 30.0, 0.0, 0.0], 1e-4);
}

#[test]
fn cl_scatter_add_dynamic_sorted_path() {
    if !autorust::runtime::ops_cl::available() {
        return;
    }
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let len = 4096usize;
    let e = 64usize;
    let idx_v: Vec<i32> = (0..e)
        .map(|i| match i % 13 {
            0 => -((i % 7) as i32 + 1),
            1 => len as i32 + (i % 5) as i32,
            _ => ((i * 37) % len) as i32,
        })
        .collect();
    let val_v: Vec<f32> = (0..e).map(|i| ((i * 13) % 97) as f32 * 0.01).collect();
    let idx = Tensor::from_slice(&ctx, &idx_v, None, Some("CL")).add(0);
    let vals = Tensor::from_slice(&ctx, &val_v, None, Some("CL"));
    let got = Tensor::scatter_add(len as i64, &idx, &vals).to_list::<f32>(&mut sess);
    check(&got, &naive_scatter(len, &idx_v, &val_v), 1e-4);
}

#[test]
fn cl_backend_smoke() {
    if !autorust::runtime::ops_cl::available() {
        return;
    }
    let ctx = Ctx::new();
    let mut sess = Session::new();
    let src: Vec<f32> = (0..20).map(|i| i as f32).collect();
    let t = Tensor::from_slice(&ctx, &src, None, Some("CL")).reshape([4, 5]);
    let idx = Tensor::from_slice(&ctx, &[0i32, -1, 9], None, Some("CL"));
    let got = t.gather_direct(0, &idx).to_list::<f32>(&mut sess);
    check(&got, &naive_gather_dim0(&src, 4, 5, &[0, -1, 9]), 0.0);

    let rows = Tensor::from_slice(&ctx, &[0i32, 2], None, Some("CL"));
    let q = Tensor::from_slice(&ctx, &[0.5f32, 3.5], None, Some("CL"));
    let (v, _s) = t.gather_interp(&rows, &q);
    check(&v.to_list::<f32>(&mut sess), &[0.5, 13.5], 1e-6);

    let ptr = Tensor::from_slice(&ctx, &[0i32, 2, 5], None, Some("CL"));
    let vals = Tensor::from_slice(&ctx, &[1.0f32, 2.0, 3.0, 4.0, 5.0], None, Some("CL"));
    check(
        &Tensor::segment_sum(&ptr, &vals).to_list::<f32>(&mut sess),
        &[3.0, 12.0],
        1e-6,
    );

    let idx = Tensor::from_slice(&ctx, &[0i32, 2, 2, 9], None, Some("CL"));
    let vals = Tensor::from_slice(&ctx, &[1.0f32, 2.0, 3.0, 4.0], None, Some("CL"));
    check(
        &Tensor::scatter_add(4, &idx, &vals).to_list::<f32>(&mut sess),
        &[1.0, 0.0, 5.0, 0.0],
        1e-6,
    );
}
