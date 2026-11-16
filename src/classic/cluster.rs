//! k-means clustering.

use super::{
    dtype_from_debug, float_dtype, meta_f64, meta_i64, meta_str_optional, put_meta, read_matrix,
    require_fitted, shape_i64, state_f64, tensor_from_f64, ClassicError, Persist, Result,
};
use crate::dtype::DTypeId;
use crate::tensor::state::State;
use crate::tensor::{Ctx, Tensor};

/// A small deterministic xorshift RNG for k-means++ initialization.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

fn sqdist(a: &[f64], b: &[f64], d: usize) -> f64 {
    (0..d).map(|j| (a[j] - b[j]) * (a[j] - b[j])).sum()
}

/// k-means clustering with k-means++ initialization.
pub struct KMeans {
    ctx: Ctx,
    centroids: Option<Tensor>,
    k: usize,
    seed: u64,
    max_iter: usize,
    tol: f64,
    inertia: f64,
    n_features: usize,
    dtype: DTypeId,
}

impl KMeans {
    /// Create an unfitted k-means model with `k` clusters.
    pub fn new(ctx: &Ctx, k: usize) -> Self {
        KMeans {
            ctx: ctx.clone(),
            centroids: None,
            k,
            seed: 0,
            max_iter: 300,
            tol: 1e-8,
            inertia: 0.0,
            n_features: 0,
            dtype: crate::dtype::FLOAT32,
        }
    }

    /// Set the initialization/Lloyd seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the maximum number of Lloyd iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter.max(1);
        self
    }

    /// Set the centroid-shift tolerance.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol.max(0.0);
        self
    }

    /// Fit `k` centroids on `x`.
    pub fn fit(&mut self, x: &Tensor) -> Result<()> {
        let (xv, n, d) = read_matrix(x, "KMeans.x")?;
        if self.k == 0 || self.k > n {
            return Err(ClassicError::Invalid(format!(
                "KMeans k must be in 1..={n}, got {}",
                self.k
            )));
        }
        let dtype = float_dtype(x)?;
        let device = x.device();
        let k = self.k;
        let mut rng = Rng::new(self.seed.wrapping_add(0x9E37_79B9_7F4A_7C15));
        // k-means++ initialization.
        let mut centroids = vec![0.0; k * d];
        let first = (rng.next_f64() * n as f64) as usize % n;
        centroids[..d].copy_from_slice(&xv[first * d..first * d + d]);
        let mut best = vec![f64::INFINITY; n];
        for c in 1..k {
            let prev = c - 1;
            let mut total = 0.0;
            for i in 0..n {
                let dist = sqdist(&xv[i * d..i * d + d], &centroids[prev * d..prev * d + d], d);
                if dist < best[i] {
                    best[i] = dist;
                }
                total += best[i];
            }
            let mut pick = if total > 0.0 {
                let target = rng.next_f64() * total;
                let mut acc = 0.0;
                let mut chosen = n - 1;
                for (i, &b) in best.iter().enumerate() {
                    acc += b;
                    if acc >= target {
                        chosen = i;
                        break;
                    }
                }
                chosen
            } else {
                (rng.next_f64() * n as f64) as usize % n
            };
            if pick >= n {
                pick = n - 1;
            }
            centroids[c * d..c * d + d].copy_from_slice(&xv[pick * d..pick * d + d]);
        }
        let mut assignment = vec![0usize; n];
        let mut inertia = 0.0;
        for _ in 0..self.max_iter {
            inertia = 0.0;
            for i in 0..n {
                let mut b = 0usize;
                let mut bd = f64::INFINITY;
                for c in 0..k {
                    let dist = sqdist(&xv[i * d..i * d + d], &centroids[c * d..c * d + d], d);
                    if dist < bd {
                        bd = dist;
                        b = c;
                    }
                }
                assignment[i] = b;
                inertia += bd;
            }
            let mut sums = vec![0.0; k * d];
            let mut counts = vec![0usize; k];
            for i in 0..n {
                let c = assignment[i];
                counts[c] += 1;
                for j in 0..d {
                    sums[c * d + j] += xv[i * d + j];
                }
            }
            // Empty clusters: reseed with the point farthest from its centroid.
            for c in 0..k {
                if counts[c] == 0 {
                    let mut far = 0usize;
                    let mut far_d = f64::NEG_INFINITY;
                    for i in 0..n {
                        let dist = sqdist(
                            &xv[i * d..i * d + d],
                            &centroids[assignment[i] * d..assignment[i] * d + d],
                            d,
                        );
                        if dist > far_d {
                            far_d = dist;
                            far = i;
                        }
                    }
                    counts[c] = 1;
                    for j in 0..d {
                        sums[c * d + j] = xv[far * d + j];
                    }
                }
            }
            let mut shift = 0.0;
            for c in 0..k {
                for j in 0..d {
                    let new = sums[c * d + j] / counts[c] as f64;
                    let diff = new - centroids[c * d + j];
                    shift += diff * diff;
                    centroids[c * d + j] = new;
                }
            }
            if shift <= self.tol {
                break;
            }
        }
        self.centroids = Some(tensor_from_f64(
            &self.ctx,
            &centroids,
            &[k as i64, d as i64],
            dtype,
            device.as_deref(),
        ));
        self.inertia = inertia;
        self.n_features = d;
        self.dtype = dtype;
        Ok(())
    }

    /// Cluster index `[n]` (int64) for each row of `x`.
    pub fn predict(&self, x: &Tensor) -> Result<Tensor> {
        let centroids = require_fitted(&self.centroids, "KMeans")?;
        let sh = shape_i64(x);
        if sh.len() != 2 || sh[1] as usize != self.n_features {
            return Err(ClassicError::Shape(format!(
                "KMeans expects [n, {}], got {sh:?}",
                self.n_features
            )));
        }
        let x = if x.dtype() != centroids.dtype() {
            x.cast(centroids.dtype())
        } else {
            x.clone()
        };
        let x2 = x.square().sum(Some(&[1]), true, None);
        let c2 = centroids.square().sum(Some(&[1]), false, None).unsqueeze(0);
        let dot = x.matmul(centroids.transpose(0, 1));
        let dist = x2.add(c2).sub(dot.mul(2.0));
        Ok(dist.argmin(Some(1), false).cast(crate::dtype::INT64))
    }

    /// Fitted centroids `[k, n_features]`.
    pub fn centroids(&self) -> Option<&Tensor> {
        self.centroids.as_ref()
    }

    /// Sum of squared distances of the training points to their centroids.
    pub fn inertia(&self) -> f64 {
        self.inertia
    }

    /// Number of clusters.
    pub fn k(&self) -> usize {
        self.k
    }
}

impl Persist for KMeans {
    fn state(&self) -> Result<State> {
        let centroids = require_fitted(&self.centroids, "KMeans")?;
        let mut s = State::new();
        s.insert("centroids", super::detach_tensor(centroids)?);
        put_meta(&mut s, "classic.kind", "kmeans");
        put_meta(&mut s, "classic.version", 1);
        put_meta(&mut s, "classic.k", self.k);
        put_meta(&mut s, "classic.seed", self.seed);
        put_meta(&mut s, "classic.max_iter", self.max_iter);
        put_meta(&mut s, "classic.tol", self.tol);
        put_meta(&mut s, "classic.inertia", self.inertia);
        put_meta(&mut s, "classic.n_features", self.n_features);
        put_meta(&mut s, "classic.dtype", super::dtype_label(self.dtype));
        Ok(s)
    }

    fn load_state(&mut self, state: &State) -> Result<()> {
        if meta_str_optional(state, "classic.kind") != Some("kmeans") {
            return Err(ClassicError::State("state is not a KMeans".into()));
        }
        let k = meta_i64(state, "classic.k")? as usize;
        let d = meta_i64(state, "classic.n_features")? as usize;
        let dtype =
            dtype_from_debug(meta_str_optional(state, "classic.dtype").unwrap_or("FLOAT32"))?;
        let centroids = state_f64(state, "centroids")?;
        if centroids.len() != k * d {
            return Err(ClassicError::State(
                "KMeans state has inconsistent dimensions".into(),
            ));
        }
        self.centroids = Some(tensor_from_f64(
            &self.ctx,
            &centroids,
            &[k as i64, d as i64],
            dtype,
            None,
        ));
        self.k = k;
        self.seed = meta_i64(state, "classic.seed").unwrap_or(0) as u64;
        self.max_iter = meta_i64(state, "classic.max_iter").unwrap_or(300) as usize;
        self.tol = meta_f64(state, "classic.tol").unwrap_or(1e-8);
        self.inertia = meta_f64(state, "classic.inertia").unwrap_or(0.0);
        self.n_features = d;
        self.dtype = dtype;
        Ok(())
    }
}
