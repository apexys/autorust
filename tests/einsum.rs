//! `Tensor::einsum_many`: multi-operand, ellipsis, repeated labels and implicit
//! output. The reference is an explicit loop for each equation.

use autorust::engine::session::Session;
use autorust::tensor::{Ctx, Tensor};

fn t(ctx: &Ctx, len: usize, shape: &[i64], seed: f32) -> Tensor {
    let data: Vec<f32> = (0..len)
        .map(|i| ((i as f32) * 0.17 + seed).sin())
        .collect();
    Tensor::from_slice(ctx, &data, None, Some("INTERP")).reshape(shape)
}

fn vals(x: &Tensor) -> Vec<f32> {
    let mut s = Session::new();
    x.to_list::<f32>(&mut s)
}

fn close(got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "len {got:?} vs {want:?}");
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!((a - b).abs() <= tol, "[{i}] {a} != {b}");
    }
}

#[test]
fn three_operands() {
    let ctx = Ctx::new();
    let a = t(&ctx, 6, &[2, 3], 0.1);
    let b = t(&ctx, 12, &[3, 4], 0.2);
    let c = t(&ctx, 8, &[4, 2], 0.3);
    let got = vals(&Tensor::einsum_many(&[&a, &b, &c], "ij,jk,kl->il"));
    let (av, bv, cv) = (vals(&a), vals(&b), vals(&c));
    let mut want = vec![0.0f32; 2 * 2];
    for i in 0..2 {
        for l in 0..2 {
            let mut s = 0.0;
            for j in 0..3 {
                for k in 0..4 {
                    s += av[i * 3 + j] * bv[j * 4 + k] * cv[k * 2 + l];
                }
            }
            want[i * 2 + l] = s;
        }
    }
    close(&got, &want, 1e-5);
}

#[test]
fn ellipsis_batched_matmul() {
    let ctx = Ctx::new();
    let a = t(&ctx, 24, &[2, 3, 4], 0.4);
    let b = t(&ctx, 40, &[2, 4, 5], 0.5);
    let got = vals(&Tensor::einsum_many(&[&a, &b], "...ij,...jk->...ik"));
    let (av, bv) = (vals(&a), vals(&b));
    let mut want = vec![0.0f32; 2 * 3 * 5];
    for bt in 0..2 {
        for i in 0..3 {
            for k in 0..5 {
                let mut s = 0.0;
                for j in 0..4 {
                    s += av[(bt * 3 + i) * 4 + j] * bv[(bt * 4 + j) * 5 + k];
                }
                want[(bt * 3 + i) * 5 + k] = s;
            }
        }
    }
    close(&got, &want, 1e-5);
}

#[test]
fn ellipsis_broadcasts_a_lower_rank_operand() {
    let ctx = Ctx::new();
    let a = t(&ctx, 24, &[2, 3, 4], 0.6);
    let b = t(&ctx, 20, &[4, 5], 0.7);
    let got = vals(&Tensor::einsum_many(&[&a, &b], "...ij,jk->...ik"));
    let (av, bv) = (vals(&a), vals(&b));
    let mut want = vec![0.0f32; 2 * 3 * 5];
    for bt in 0..2 {
        for i in 0..3 {
            for k in 0..5 {
                let mut s = 0.0;
                for j in 0..4 {
                    s += av[(bt * 3 + i) * 4 + j] * bv[j * 5 + k];
                }
                want[(bt * 3 + i) * 5 + k] = s;
            }
        }
    }
    close(&got, &want, 1e-5);
}

#[test]
fn repeated_label_is_a_diagonal() {
    let ctx = Ctx::new();
    let a = t(&ctx, 9, &[3, 3], 0.8);
    let av = vals(&a);
    let diag = vals(&Tensor::einsum_many(&[&a], "ii->i"));
    close(&diag, &[av[0], av[4], av[8]], 1e-6);
    let trace = vals(&Tensor::einsum_many(&[&a], "ii->"));
    close(&trace, &[av[0] + av[4] + av[8]], 1e-6);
    // Implicit output drops the repeated label: `ii` is the trace.
    let trace2 = vals(&Tensor::einsum_many(&[&a], "ii"));
    close(&trace2, &trace, 1e-6);
}

#[test]
fn repeated_label_with_a_free_axis() {
    let ctx = Ctx::new();
    let a = t(&ctx, 12, &[2, 2, 3], 0.9);
    let av = vals(&a);
    let got = vals(&Tensor::einsum_many(&[&a], "iij->j"));
    let mut want = vec![0.0f32; 3];
    for i in 0..2 {
        for j in 0..3 {
            want[j] += av[(i * 2 + i) * 3 + j];
        }
    }
    close(&got, &want, 1e-6);
}

#[test]
fn implicit_output_matches_the_explicit_form() {
    let ctx = Ctx::new();
    let a = t(&ctx, 6, &[2, 3], 1.1);
    let b = t(&ctx, 12, &[3, 4], 1.2);
    let implicit = vals(&Tensor::einsum_many(&[&a, &b], "ij,jk"));
    let explicit = vals(&Tensor::einsum_many(&[&a, &b], "ij,jk->ik"));
    close(&implicit, &explicit, 1e-6);
}
