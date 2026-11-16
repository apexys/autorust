//! Small dense host linear algebra for classical estimators.
//!
//! All matrices are row-major. These routines are intentionally plain `f64`
//! implementations: classic tabular problems are usually small enough that a
//! host solve/eigendecomposition is cheaper than a device round-trip, while
//! the large products (`XᵀX`, predictions) can still use the Tensor stack.

use super::{ClassicError, Result};

fn check_len(name: &str, actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        return Err(ClassicError::Shape(format!(
            "{name} has {actual} values, expected {expected}"
        )));
    }
    Ok(())
}

/// Solve the general `n x n` system `A X = B` by LU with partial pivoting.
///
/// `a` is `n*n` row-major; `b` is `n*m` row-major. Returns `X` (`n*m`).
pub fn solve(a: &[f64], b: &[f64], n: usize, m: usize) -> Result<Vec<f64>> {
    check_len("A", a.len(), n * n)?;
    check_len("B", b.len(), n * m)?;
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut lu = a.to_vec();
    let mut x = b.to_vec();
    for k in 0..n {
        let mut pivot = k;
        let mut best = lu[k * n + k].abs();
        for i in k + 1..n {
            let v = lu[i * n + k].abs();
            if v > best {
                best = v;
                pivot = i;
            }
        }
        if best < 1e-14 {
            return Err(ClassicError::Singular(format!("zero pivot at column {k}")));
        }
        if pivot != k {
            for j in 0..n {
                lu.swap(k * n + j, pivot * n + j);
            }
            for j in 0..m {
                x.swap(k * m + j, pivot * m + j);
            }
        }
        let d = lu[k * n + k];
        for i in k + 1..n {
            let f = lu[i * n + k] / d;
            lu[i * n + k] = 0.0;
            for j in k + 1..n {
                lu[i * n + j] -= f * lu[k * n + j];
            }
            for j in 0..m {
                x[i * m + j] -= f * x[k * m + j];
            }
        }
    }
    for k in (0..n).rev() {
        let d = lu[k * n + k];
        for j in 0..m {
            let mut v = x[k * m + j];
            for i in k + 1..n {
                v -= lu[k * n + i] * x[i * m + j];
            }
            x[k * m + j] = v / d;
        }
    }
    Ok(x)
}

/// Solve a symmetric positive-definite system by Cholesky factorization.
pub fn cholesky_solve(a: &[f64], b: &[f64], n: usize, m: usize) -> Result<Vec<f64>> {
    check_len("A", a.len(), n * n)?;
    check_len("B", b.len(), n * m)?;
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 0.0 || !s.is_finite() {
                    return Err(ClassicError::Singular(format!(
                        "matrix is not positive definite (diagonal {i} = {s})"
                    )));
                }
                l[i * n + j] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    let mut y = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            let mut s = b[i * m + j];
            for k in 0..i {
                s -= l[i * n + k] * y[k * m + j];
            }
            y[i * m + j] = s / l[i * n + i];
        }
    }
    let mut x = vec![0.0; n * m];
    for i in (0..n).rev() {
        for j in 0..m {
            let mut s = y[i * m + j];
            for k in i + 1..n {
                s -= l[k * n + i] * x[k * m + j];
            }
            x[i * m + j] = s / l[i * n + i];
        }
    }
    Ok(x)
}

/// Householder QR: returns `(Q, R)` with `Q` `n*n` and `R` `n*d`.
pub fn qr(a: &[f64], n: usize, d: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    check_len("A", a.len(), n * d)?;
    let mut r = a.to_vec();
    let mut q = vec![0.0; n * n];
    for i in 0..n {
        q[i * n + i] = 1.0;
    }
    let mut v = vec![0.0; n];
    for k in 0..n.min(d) {
        let norm = (k..n)
            .map(|i| r[i * d + k] * r[i * d + k])
            .sum::<f64>()
            .sqrt();
        if norm < 1e-15 {
            continue;
        }
        let alpha = if r[k * d + k] >= 0.0 { -norm } else { norm };
        for i in 0..n {
            v[i] = if i >= k { r[i * d + k] } else { 0.0 };
        }
        v[k] -= alpha;
        let vnorm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if vnorm < 1e-15 {
            continue;
        }
        for x in &mut v {
            *x /= vnorm;
        }
        // R <- H R
        for j in k..d {
            let dot: f64 = (k..n).map(|i| v[i] * r[i * d + j]).sum();
            for i in k..n {
                r[i * d + j] -= 2.0 * v[i] * dot;
            }
        }
        // Q <- Q H
        for i in 0..n {
            let dot: f64 = (k..n).map(|l| q[i * n + l] * v[l]).sum();
            for j in k..n {
                q[i * n + j] -= 2.0 * dot * v[j];
            }
        }
    }
    Ok((q, r))
}

/// Symmetric eigendecomposition by Jacobi rotations.
///
/// Returns `(eigenvalues, eigenvectors)` sorted by descending eigenvalue.
/// Eigenvectors are columns: component `i` of eigenvector `j` is `v[i*n+j]`.
pub fn symmetric_eigh(a: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    check_len("A", a.len(), n * n)?;
    let mut m = a.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _ in 0..100 {
        let mut off = 0.0;
        for p in 0..n {
            for q in p + 1..n {
                off += m[p * n + q] * m[p * n + q];
            }
        }
        if off < 1e-24 {
            break;
        }
        for p in 0..n.saturating_sub(1) {
            for q in p + 1..n {
                let apq = m[p * n + q];
                if apq.abs() < 1e-18 {
                    continue;
                }
                let app = m[p * n + p];
                let aqq = m[q * n + q];
                // J = [[c, s], [-s, c]] with theta = 1/2 atan2(2 apq, app-aqq).
                let theta = 0.5 * (2.0 * apq).atan2(aqq - app);
                let c = theta.cos();
                let s = theta.sin();
                // Rows p/q: R <- J^T R.
                for j in 0..n {
                    let mp = m[p * n + j];
                    let mq = m[q * n + j];
                    m[p * n + j] = c * mp - s * mq;
                    m[q * n + j] = s * mp + c * mq;
                }
                // Columns p/q: R <- R J.
                for i in 0..n {
                    let mp = m[i * n + p];
                    let mq = m[i * n + q];
                    m[i * n + p] = c * mp - s * mq;
                    m[i * n + q] = s * mp + c * mq;
                }
                // V <- V J.
                for i in 0..n {
                    let vp = v[i * n + p];
                    let vq = v[i * n + q];
                    v[i * n + p] = c * vp - s * vq;
                    v[i * n + q] = s * vp + c * vq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| m[j * n + j].partial_cmp(&m[i * n + i]).unwrap());
    let mut evals = Vec::with_capacity(n);
    let mut vecs = vec![0.0; n * n];
    for (new, &old) in order.iter().enumerate() {
        evals.push(m[old * n + old]);
        for i in 0..n {
            vecs[i * n + new] = v[i * n + old];
        }
    }
    Ok((evals, vecs))
}

/// Thin SVD of an `n x d` matrix via the eigendecomposition of `AᵀA`.
///
/// Returns `(u, s, vt)` with `u` `n*d`, `s` length `d`, and `vt` `d*d`.
pub fn svd(a: &[f64], n: usize, d: usize) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    check_len("A", a.len(), n * d)?;
    let mut ata = vec![0.0; d * d];
    for i in 0..n {
        for p in 0..d {
            let x = a[i * d + p];
            for q in p..d {
                ata[p * d + q] += x * a[i * d + q];
            }
        }
    }
    for p in 0..d {
        for q in 0..p {
            ata[p * d + q] = ata[q * d + p];
        }
    }
    let (evals, v) = symmetric_eigh(&ata, d)?;
    let s: Vec<f64> = evals.iter().map(|&e| e.max(0.0).sqrt()).collect();
    let tol = s.first().copied().unwrap_or(0.0) * 1e-12;
    let mut u = vec![0.0; n * d];
    for i in 0..n {
        for j in 0..d {
            if s[j] <= tol {
                continue;
            }
            let mut acc = 0.0;
            for k in 0..d {
                acc += a[i * d + k] * v[k * d + j];
            }
            u[i * d + j] = acc / s[j];
        }
    }
    let mut vt = vec![0.0; d * d];
    for j in 0..d {
        for i in 0..d {
            vt[j * d + i] = v[i * d + j];
        }
    }
    Ok((u, s, vt))
}

/// Least-squares/ridge solve of `A W = B`, `A` `n*d`, `B` `n*m`.
///
/// Uses the SVD, so rank-deficient problems fall back to the pseudoinverse.
/// The ridge term is applied as `(s² + ridge)`.
pub fn least_squares(
    a: &[f64],
    b: &[f64],
    n: usize,
    d: usize,
    m: usize,
    ridge: f64,
) -> Result<Vec<f64>> {
    check_len("A", a.len(), n * d)?;
    check_len("B", b.len(), n * m)?;
    let (_, s, vt) = svd(a, n, d)?;
    let mut aty = vec![0.0; d * m];
    for i in 0..n {
        for j in 0..d {
            let x = a[i * d + j];
            for k in 0..m {
                aty[j * m + k] += x * b[i * m + k];
            }
        }
    }
    // z = V^T aty = vt * aty
    let mut z = vec![0.0; d * m];
    for j in 0..d {
        for k in 0..m {
            let mut acc = 0.0;
            for i in 0..d {
                acc += vt[j * d + i] * aty[i * m + k];
            }
            z[j * m + k] = acc;
        }
    }
    let max_s = s.first().copied().unwrap_or(0.0);
    for j in 0..d {
        let denom = s[j] * s[j] + ridge;
        let scale = if denom > max_s * max_s * 1e-24 && denom > 0.0 {
            1.0 / denom
        } else {
            0.0
        };
        for k in 0..m {
            z[j * m + k] *= scale;
        }
    }
    // W = V z; V[i,j] = vt[j*d+i]
    let mut w = vec![0.0; d * m];
    for i in 0..d {
        for k in 0..m {
            let mut acc = 0.0;
            for j in 0..d {
                acc += vt[j * d + i] * z[j * m + k];
            }
            w[i * m + k] = acc;
        }
    }
    Ok(w)
}

/// Dot product of two equal-length vectors.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
