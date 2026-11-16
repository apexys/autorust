//! C0/C1 classical-ML gates.

use autorust::classic::linalg;
use autorust::classic::metrics;
use autorust::classic::{
    KMeans, KNNClassifier, LinearRegression, LinearSVC, LinearSVR, LogisticRegression, Persist,
    StandardScaler, PCA,
};
use autorust::dtype;
use autorust::engine::session::Session;
use autorust::tensor::{Ctx, Tensor};
use std::sync::atomic::{AtomicUsize, Ordering};

static FILE_ID: AtomicUsize = AtomicUsize::new(0);

fn tensor(ctx: &Ctx, vals: &[f64], shape: &[i64]) -> Tensor {
    Tensor::from_slice(ctx, vals, Some(dtype::FLOAT64), None).reshape(shape.to_vec())
}

fn read(t: &Tensor, sess: &mut Session) -> Vec<f64> {
    t.to_list::<f64>(sess)
}

fn close(got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len());
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!(
            (a - b).abs() <= tol,
            "[{i}] {a} != {b} (diff {})",
            (a - b).abs()
        );
    }
}

fn close_mat(got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len());
    let max = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(max <= tol, "max abs diff {max} > {tol}");
}

#[test]
fn linalg_solve_cholesky_qr_svd_eigh() {
    let a = [4.0, 1.0, 1.0, 3.0];
    let b = [1.0, 2.0];
    let x = linalg::solve(&a, &b, 2, 1).unwrap();
    close(&x, &[1.0 / 11.0, 7.0 / 11.0], 1e-12);
    let x = linalg::cholesky_solve(&a, &b, 2, 1).unwrap();
    close(&x, &[1.0 / 11.0, 7.0 / 11.0], 1e-12);

    // QR reconstruction.
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0];
    let (q, r) = linalg::qr(&a, 3, 2).unwrap();
    let mut qr = vec![0.0; 6];
    for i in 0..3 {
        for j in 0..2 {
            qr[i * 2 + j] = (0..3).map(|k| q[i * 3 + k] * r[k * 2 + j]).sum();
        }
    }
    close_mat(&qr, &a, 1e-10);

    // Symmetric eigendecomposition: A = V diag(ev) V^T.
    let a = [2.0, 1.0, 1.0, 2.0];
    let (ev, v) = linalg::symmetric_eigh(&a, 2).unwrap();
    assert!((ev[0] - 3.0).abs() < 1e-10 && (ev[1] - 1.0).abs() < 1e-10);
    let mut recon = vec![0.0; 4];
    for i in 0..2 {
        for j in 0..2 {
            recon[i * 2 + j] = (0..2).map(|k| v[i * 2 + k] * ev[k] * v[j * 2 + k]).sum();
        }
    }
    close_mat(&recon, &a, 1e-10);

    let w = linalg::least_squares(&[1.0, 2.0, 3.0, 4.0], &[1.0, 2.0], 2, 2, 1, 0.0).unwrap();
    close(&w, &[0.0, 0.5], 1e-10);
    // SVD reconstruction.
    let (u, s, vt) = linalg::svd(&a, 3, 2).unwrap_or_else(|_| {
        // The 2x2 A is padded to 3x2 for this test below.
        linalg::svd(&[2.0, 1.0, 1.0, 2.0, 0.0, 0.0], 3, 2).unwrap()
    });
    let _ = (u, s, vt);
    let a3 = [2.0, 1.0, 1.0, 2.0, 0.0, 0.0];
    let (u, s, vt) = linalg::svd(&a3, 3, 2).unwrap();
    let mut recon = vec![0.0; 6];
    for i in 0..3 {
        for j in 0..2 {
            recon[i * 2 + j] = (0..2).map(|k| u[i * 2 + k] * s[k] * vt[k * 2 + j]).sum();
        }
    }
    close_mat(&recon, &a3, 1e-10);
}

#[test]
fn least_squares_underdetermined_and_silhouette() {
    // n=2, d=3: minimum-norm solution is [1, 1, 0].
    let w =
        linalg::least_squares(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[1.0, 1.0], 2, 3, 1, 0.0).unwrap();
    close(&w, &[1.0, 1.0, 0.0], 1e-9);

    let x = [0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1];
    let labels = [0i64, 0, 0, 1, 1, 1];
    let s = metrics::silhouette_score(&x, &labels, 6, 2).unwrap();
    assert!(s > 0.9, "silhouette {s}");
}

#[test]
fn metrics_and_scaler() {
    let y = [0i64, 1, 1, 2];
    let p = [0i64, 1, 2, 2];
    assert!((metrics::accuracy(&y, &p).unwrap() - 0.75).abs() < 1e-12);
    let cm = metrics::confusion_matrix(&y, &p, 3).unwrap();
    assert_eq!(cm, vec![1, 0, 0, 0, 1, 1, 0, 0, 1]);
    assert!(
        (metrics::macro_f1(&y, &p, 3).unwrap() - (2.0 / 3.0 + 1.0 + 2.0 / 3.0) / 3.0).abs() < 1e-12
    );

    let ctx = Ctx::new();
    let x = tensor(&ctx, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let mut sc = StandardScaler::new(&ctx);
    let z = sc.fit_transform(&x).unwrap();
    let mut sess = Session::new();
    let zv = read(&z, &mut sess);
    let n = 3.0;
    let mean0 = zv[0] + zv[2] + zv[4];
    let mean1 = zv[1] + zv[3] + zv[5];
    assert!(mean0.abs() < 1e-12 && mean1.abs() < 1e-12);
    let var0 = (zv[0].powi(2) + zv[2].powi(2) + zv[4].powi(2)) / n;
    let var1 = (zv[1].powi(2) + zv[3].powi(2) + zv[5].powi(2)) / n;
    assert!((var0 - 1.0).abs() < 1e-12 && (var1 - 1.0).abs() < 1e-12);
    let back = sc.inverse_transform(&z).unwrap();
    close(&read(&back, &mut sess), &read(&x, &mut sess), 1e-12);
}

#[test]
fn linear_and_ridge_regression() {
    let ctx = Ctx::new();
    // y = 3*x0 - 2*x1 + 1.
    let mut xv = Vec::new();
    let mut yv = Vec::new();
    for i in 0..6 {
        let x0 = i as f64 - 2.0;
        let x1 = (i % 3) as f64 - 1.0;
        xv.push(x0);
        xv.push(x1);
        yv.push(3.0 * x0 - 2.0 * x1 + 1.0);
    }
    let x = tensor(&ctx, &xv, &[6, 2]);
    let y = tensor(&ctx, &yv, &[6]);
    let mut m = LinearRegression::new(&ctx);
    m.fit(&x, &y).unwrap();
    let pred = m.predict(&x).unwrap();
    let mut sess = Session::new();
    close(&read(&pred, &mut sess), &yv, 1e-9);
    let coef = read(m.coef().unwrap(), &mut sess);
    close(&coef, &[3.0, -2.0], 1e-9);

    let mut ridge = LinearRegression::ridge(&ctx, 1e-6);
    ridge.fit(&x, &y).unwrap();
    let p = ridge.predict(&x).unwrap();
    close(&read(&p, &mut sess), &yv, 1e-4);
}

#[test]
fn logistic_regression_binary_and_multiclass() {
    let ctx = Ctx::new();
    // Two well-separated clusters.
    let mut xv = Vec::new();
    let mut yv = Vec::new();
    for i in 0..20 {
        let t = i as f64 / 20.0;
        xv.extend([-1.0 + t * 0.2, -1.0 - t * 0.1]);
        yv.push(0i64);
        xv.extend([1.0 + t * 0.2, 1.0 - t * 0.1]);
        yv.push(1i64);
    }
    let x = tensor(&ctx, &xv, &[40, 2]);
    let y = Tensor::from_slice(&ctx, &yv, Some(dtype::INT64), None);
    let mut m = LogisticRegression::new(&ctx).with_max_iter(500);
    m.fit(&x, &y).unwrap();
    let mut sess = Session::new();
    let pred = m.predict(&x).unwrap();
    let p = pred.to_list::<i64>(&mut sess);
    assert!(metrics::accuracy(&yv, &p).unwrap() > 0.95);
    let proba = m.predict_proba(&x).unwrap();
    let pv = read(&proba, &mut sess);
    for row in 0..40 {
        assert!((pv[row * 2] + pv[row * 2 + 1] - 1.0).abs() < 1e-8);
    }

    // Three separated clusters.
    let mut xv = Vec::new();
    let mut yv = Vec::new();
    for i in 0..15 {
        let t = i as f64 * 0.05;
        xv.extend([-2.0 + t, 0.0]);
        yv.push(0i64);
        xv.extend([2.0, -2.0 + t]);
        yv.push(1i64);
        xv.extend([0.0, 2.0 + t]);
        yv.push(2i64);
    }
    let x = tensor(&ctx, &xv, &[45, 2]);
    let y = Tensor::from_slice(&ctx, &yv, Some(dtype::INT64), None);
    let mut m = LogisticRegression::new(&ctx).with_max_iter(1000);
    m.fit(&x, &y).unwrap();
    let p = m.predict(&x).unwrap().to_list::<i64>(&mut sess);
    assert!(metrics::accuracy(&yv, &p).unwrap() > 0.9);
}

#[test]
fn pca_projection_and_inverse() {
    let ctx = Ctx::new();
    let mut xv = Vec::new();
    for i in 0..20 {
        let t = i as f64;
        xv.extend([t, 2.0 * t]);
    }
    let x = tensor(&ctx, &xv, &[20, 2]);
    let mut pca = PCA::new(&ctx, 1);
    let z = pca.fit_transform(&x).unwrap();
    let mut sess = Session::new();
    let zv = read(&z, &mut sess);
    assert_eq!(zv.len(), 20);
    let ratio = pca.explained_variance_ratio().unwrap();
    assert!(ratio[0] > 0.99);
    let back = pca.inverse_transform(&z).unwrap();
    close(&read(&back, &mut sess), &read(&x, &mut sess), 1e-8);
}

#[test]
fn kmeans_finds_clusters() {
    let ctx = Ctx::new();
    let mut xv = Vec::new();
    for i in 0..10 {
        let t = i as f64 * 0.01;
        xv.extend([0.0 + t, 0.0]);
        xv.extend([5.0 + t, 0.0]);
        xv.extend([0.0, 5.0 + t]);
    }
    let x = tensor(&ctx, &xv, &[30, 2]);
    let mut m = KMeans::new(&ctx, 3).with_seed(7);
    m.fit(&x).unwrap();
    let mut sess = Session::new();
    let labels = m.predict(&x).unwrap().to_list::<i64>(&mut sess);
    // The three true clusters must map to three distinct labels.
    assert_ne!(labels[0], labels[1]);
    assert_ne!(labels[0], labels[2]);
    assert_ne!(labels[1], labels[2]);
    assert!(m.inertia() < 1.0);
}

#[test]
fn knn_classifies_nearest_points() {
    let ctx = Ctx::new();
    let xv = vec![0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1];
    let yv = vec![0i64, 0, 0, 1, 1, 1];
    let x = tensor(&ctx, &xv, &[6, 2]);
    let y = Tensor::from_slice(&ctx, &yv, Some(dtype::INT64), None);
    let mut m = KNNClassifier::new(&ctx, 3);
    m.fit(&x, &y).unwrap();
    let query = tensor(&ctx, &[0.05, 0.0, 5.0, 5.05], &[2, 2]);
    let mut sess = Session::new();
    let p = m.predict(&query).unwrap().to_list::<i64>(&mut sess);
    assert_eq!(p, vec![0, 1]);
    let proba = m.predict_proba(&query).unwrap();
    let pv = read(&proba, &mut sess);
    assert!((pv[0] - 1.0).abs() < 1e-6 && (pv[3] - 1.0).abs() < 1e-6);
}

#[test]
fn linear_svm_and_svr() {
    let ctx = Ctx::new();
    let xv = vec![
        -2.0, -1.0, -1.5, -0.5, -1.0, -1.5, 1.0, 1.0, 2.0, 1.5, 1.5, 2.0,
    ];
    let yv = vec![0i64, 0, 0, 1, 1, 1];
    let x = tensor(&ctx, &xv, &[6, 2]);
    let y = Tensor::from_slice(&ctx, &yv, Some(dtype::INT64), None);
    let mut svc = LinearSVC::new(&ctx).with_l2(1e-3).with_max_iter(2000);
    svc.fit(&x, &y).unwrap();
    let mut sess = Session::new();
    let pred = svc.predict(&x).unwrap().to_list::<i64>(&mut sess);
    assert!(metrics::accuracy(&yv, &pred).unwrap() > 0.95);

    let mut rx = Vec::new();
    let mut ry = Vec::new();
    for i in 0..12 {
        let v = i as f64 * 0.5 - 3.0;
        rx.push(v);
        ry.push(2.0 * v + 1.0);
    }
    let x = tensor(&ctx, &rx, &[12, 1]);
    let y = tensor(&ctx, &ry, &[12]);
    let mut svr = LinearSVR::new(&ctx, 0.05).with_l2(1e-4).with_max_iter(2000);
    svr.fit(&x, &y).unwrap();
    let pred = svr.predict(&x).unwrap();
    let pv = read(&pred, &mut sess);
    assert!(metrics::mse(&ry, &pv).unwrap() < 0.02);
}

fn temp_path(name: &str) -> std::path::PathBuf {
    let id = FILE_ID.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "autorust_classic_{}_{}_{}.safetensors",
        std::process::id(),
        name,
        id
    ))
}

#[test]
fn persistence_round_trip() {
    let ctx = Ctx::new();
    let mut xv = Vec::new();
    let mut yv = Vec::new();
    for i in 0..8 {
        let x0 = i as f64;
        let x1 = (i as f64 * 0.5).sin();
        xv.extend([x0, x1]);
        yv.push(2.0 * x0 + 3.0 * x1 - 1.0);
    }
    let x = tensor(&ctx, &xv, &[8, 2]);
    let y = tensor(&ctx, &yv, &[8]);
    let mut lin = LinearRegression::new(&ctx);
    lin.fit(&x, &y).unwrap();
    let mut sess = Session::new();
    let p0 = read(&lin.predict(&x).unwrap(), &mut sess);
    let path = temp_path("lin");
    lin.save(&path).unwrap();
    let state = autorust::classic::load_state_file(&path).unwrap();
    let mut lin2 = LinearRegression::new(&ctx);
    lin2.load_state(&state).unwrap();
    let p1 = read(&lin.predict(&x).unwrap(), &mut sess);
    let p2 = read(&lin2.predict(&x).unwrap(), &mut sess);
    close(&p0, &p2, 1e-12);
    close(&p1, &p2, 1e-12);
    let _ = std::fs::remove_file(&path);

    let mut pca = PCA::new(&ctx, 1);
    pca.fit(&x).unwrap();
    let path = temp_path("pca");
    pca.save(&path).unwrap();
    let state = autorust::classic::load_state_file(&path).unwrap();
    let mut pca2 = PCA::new(&ctx, 1);
    pca2.load_state(&state).unwrap();
    let z1 = read(&pca.transform(&x).unwrap(), &mut sess);
    let z2 = read(&pca2.transform(&x).unwrap(), &mut sess);
    close(&z1, &z2, 1e-12);
    let _ = std::fs::remove_file(&path);

    let mut km = KMeans::new(&ctx, 2).with_seed(3);
    km.fit(&x).unwrap();
    let path = temp_path("km");
    km.save(&path).unwrap();
    let state = autorust::classic::load_state_file(&path).unwrap();
    let mut km2 = KMeans::new(&ctx, 2);
    km2.load_state(&state).unwrap();
    assert_eq!(
        km.predict(&x).unwrap().to_list::<i64>(&mut sess),
        km2.predict(&x).unwrap().to_list::<i64>(&mut sess)
    );
    let _ = std::fs::remove_file(&path);

    // Logistic and kNN persistence.
    let labels = Tensor::from_slice(&ctx, &[0i64, 0, 0, 0, 1, 1, 1, 1], Some(dtype::INT64), None);
    let mut logit = LogisticRegression::new(&ctx).with_max_iter(300);
    logit.fit(&x, &labels).unwrap();
    let path = temp_path("logit");
    logit.save(&path).unwrap();
    let state = autorust::classic::load_state_file(&path).unwrap();
    let mut logit2 = LogisticRegression::new(&ctx);
    logit2.load_state(&state).unwrap();
    assert_eq!(
        logit.predict(&x).unwrap().to_list::<i64>(&mut sess),
        logit2.predict(&x).unwrap().to_list::<i64>(&mut sess)
    );
    let _ = std::fs::remove_file(&path);

    let mut knn = KNNClassifier::new(&ctx, 2);
    knn.fit(&x, &labels).unwrap();
    let path = temp_path("knn");
    knn.save(&path).unwrap();
    let state = autorust::classic::load_state_file(&path).unwrap();
    let mut knn2 = KNNClassifier::new(&ctx, 2);
    knn2.load_state(&state).unwrap();
    assert_eq!(
        knn.predict(&x).unwrap().to_list::<i64>(&mut sess),
        knn2.predict(&x).unwrap().to_list::<i64>(&mut sess)
    );
    let _ = std::fs::remove_file(&path);
}
