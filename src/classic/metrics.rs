//! Metrics for classical estimators.

use super::{ClassicError, Result};

fn same_len(a: usize, b: usize, name: &str) -> Result<()> {
    if a != b {
        return Err(ClassicError::Shape(format!(
            "{name} length mismatch: {a} != {b}"
        )));
    }
    Ok(())
}

/// Classification accuracy.
pub fn accuracy(y_true: &[i64], y_pred: &[i64]) -> Result<f64> {
    same_len(y_true.len(), y_pred.len(), "accuracy")?;
    if y_true.is_empty() {
        return Err(ClassicError::Data("accuracy on empty data".into()));
    }
    let correct = y_true.iter().zip(y_pred).filter(|(a, b)| a == b).count();
    Ok(correct as f64 / y_true.len() as f64)
}

/// Row-major confusion matrix, `true` rows and `pred` columns.
pub fn confusion_matrix(y_true: &[i64], y_pred: &[i64], n_classes: usize) -> Result<Vec<usize>> {
    same_len(y_true.len(), y_pred.len(), "confusion matrix")?;
    if n_classes == 0 {
        return Err(ClassicError::Invalid("n_classes must be positive".into()));
    }
    let mut out = vec![0usize; n_classes * n_classes];
    for (&t, &p) in y_true.iter().zip(y_pred) {
        if t < 0 || p < 0 || t as usize >= n_classes || p as usize >= n_classes {
            return Err(ClassicError::Data(format!(
                "label out of range for {n_classes} classes: true={t}, pred={p}"
            )));
        }
        out[t as usize * n_classes + p as usize] += 1;
    }
    Ok(out)
}

/// Precision, recall and F1 for one class.
pub fn precision_recall_f1(y_true: &[i64], y_pred: &[i64], class: i64) -> Result<(f64, f64, f64)> {
    same_len(y_true.len(), y_pred.len(), "precision/recall")?;
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut fne = 0usize;
    for (&t, &p) in y_true.iter().zip(y_pred) {
        if t == class && p == class {
            tp += 1;
        } else if p == class {
            fp += 1;
        } else if t == class {
            fne += 1;
        }
    }
    let precision = if tp + fp == 0 {
        0.0
    } else {
        tp as f64 / (tp + fp) as f64
    };
    let recall = if tp + fne == 0 {
        0.0
    } else {
        tp as f64 / (tp + fne) as f64
    };
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    Ok((precision, recall, f1))
}

/// Macro-averaged F1 over classes `0..n_classes`.
pub fn macro_f1(y_true: &[i64], y_pred: &[i64], n_classes: usize) -> Result<f64> {
    if n_classes == 0 {
        return Err(ClassicError::Invalid("n_classes must be positive".into()));
    }
    let mut sum = 0.0;
    for c in 0..n_classes as i64 {
        sum += precision_recall_f1(y_true, y_pred, c)?.2;
    }
    Ok(sum / n_classes as f64)
}

/// Mean squared error.
pub fn mse(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
    same_len(y_true.len(), y_pred.len(), "mse")?;
    if y_true.is_empty() {
        return Err(ClassicError::Data("mse on empty data".into()));
    }
    Ok(y_true
        .iter()
        .zip(y_pred)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        / y_true.len() as f64)
}

/// Root mean squared error.
pub fn rmse(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
    Ok(mse(y_true, y_pred)?.sqrt())
}

/// Mean absolute error.
pub fn mae(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
    same_len(y_true.len(), y_pred.len(), "mae")?;
    if y_true.is_empty() {
        return Err(ClassicError::Data("mae on empty data".into()));
    }
    Ok(y_true
        .iter()
        .zip(y_pred)
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / y_true.len() as f64)
}

/// Coefficient of determination `R²`.
pub fn r2(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
    same_len(y_true.len(), y_pred.len(), "r2")?;
    if y_true.is_empty() {
        return Err(ClassicError::Data("r2 on empty data".into()));
    }
    let mean = y_true.iter().sum::<f64>() / y_true.len() as f64;
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred)
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    let ss_tot: f64 = y_true.iter().map(|a| (a - mean) * (a - mean)).sum();
    if ss_tot == 0.0 {
        return Ok(if ss_res == 0.0 { 1.0 } else { 0.0 });
    }
    Ok(1.0 - ss_res / ss_tot)
}

/// Mean silhouette coefficient for k-means/label diagnostics.
///
/// `x` is `[n, d]` row-major. Singleton clusters contribute `0`.
#[allow(clippy::needless_range_loop)]
pub fn silhouette_score(x: &[f64], labels: &[i64], n: usize, d: usize) -> Result<f64> {
    same_len(labels.len(), n, "silhouette")?;
    if x.len() != n * d {
        return Err(ClassicError::Shape(format!(
            "silhouette X has {} values, expected {}",
            x.len(),
            n * d
        )));
    }
    if n < 2 || d == 0 {
        return Err(ClassicError::Data(
            "silhouette needs at least 2 samples and 1 feature".into(),
        ));
    }
    let mut unique = labels.to_vec();
    unique.sort_unstable();
    unique.dedup();
    if unique.len() < 2 {
        return Ok(0.0);
    }
    let dist = |a: usize, b: usize| -> f64 {
        (0..d)
            .map(|k| {
                let dx = x[a * d + k] - x[b * d + k];
                dx * dx
            })
            .sum::<f64>()
            .sqrt()
    };
    let mut total = 0.0;
    for i in 0..n {
        let own = labels[i];
        let mut own_sum = 0.0;
        let mut own_count = 0usize;
        let mut other_best = f64::INFINITY;
        for &c in &unique {
            if c == own {
                continue;
            }
            let mut sum = 0.0;
            let mut count = 0usize;
            for j in 0..n {
                if labels[j] == c {
                    sum += dist(i, j);
                    count += 1;
                }
            }
            if count > 0 {
                other_best = other_best.min(sum / count as f64);
            }
        }
        for j in 0..n {
            if i != j && labels[j] == own {
                own_sum += dist(i, j);
                own_count += 1;
            }
        }
        if own_count == 0 || !other_best.is_finite() {
            continue;
        }
        let a = own_sum / own_count as f64;
        let b = other_best;
        let s = (b - a) / a.max(b);
        total += s;
    }
    Ok(total / n as f64)
}
