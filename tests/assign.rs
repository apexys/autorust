//! In-place parameter updates (`Tensor::assign`) — the faithful tinygrad path.
//!
//! tinygrad's `Tensor.assign` (tensor.py:436-466) builds the write as a
//! `STORE`+`AFTER` effect — `self.uop.after(self.uop.store(x.uop))` — so the
//! scheduler *fuses* the parameter write into the schedule and orders any read
//! of the parameter before the store. The port used to realize the update and
//! `buffer_copy_into` the parameter instead (a separate pass). These tests pin
//! the faithful path:
//! - the effect is co-realizable with other roots, and a read in the same
//!   schedule sees the *old* value while the parameter ends up updated;
//! - repeated assigns accumulate (the parameter stays a stable leaf);
//! - the parameter owns its buffer, so a schedule that wrote it is not lost to
//!   `BufTable::prune` before the next read.
use autorust::engine::session::Session;
use autorust::tensor::Tensor;
use autorust::uop::ctx::UOpCtx;

fn approx(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "length {} != {}", a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() < 1e-5, "[{i}] {x} != {y} (got {a:?}, want {b:?})");
    }
}

fn scalar(ctx: &mut UOpCtx, v: f64) -> Tensor {
    Tensor::full_const(ctx, &[] as &[i64], autorust::dtype::ConstVal::Float(v), None)
}

#[test]
fn assign_updates_in_place() {
    let mut ctx = UOpCtx::new();
    let mut sess = Session::new();
    let p = Tensor::zeros_param(&mut ctx, [3], None, None);
    approx(&p.to_list::<f32>(&mut ctx, &mut sess), &[0.0, 0.0, 0.0]);

    let one = scalar(&mut ctx, 1.0);
    let v = p.add(&mut ctx, &one);
    p.assign(&mut ctx, &mut sess, &v);
    approx(&p.to_list::<f32>(&mut ctx, &mut sess), &[1.0, 1.0, 1.0]);
}

#[test]
fn assign_accumulates_across_steps() {
    let mut ctx = UOpCtx::new();
    let mut sess = Session::new();
    let p = Tensor::zeros_param(&mut ctx, [3], None, None);
    for step in 1..=5 {
        let one = scalar(&mut ctx, 1.0);
        let v = p.add(&mut ctx, &one);
        p.assign(&mut ctx, &mut sess, &v);
        approx(&p.to_list::<f32>(&mut ctx, &mut sess), &[step as f32; 3]);
    }
}

/// Co-realize a **read** of `p` with the **write** to `p` in one schedule: the
/// read must see the old value, the write must land (tinygrad's AFTER ordering).
#[test]
fn assign_effect_orders_read_before_store() {
    let mut ctx = UOpCtx::new();
    let mut sess = Session::new();
    let p = Tensor::zeros_param(&mut ctx, [3], None, None);

    let two = scalar(&mut ctx, 2.0);
    let read = p.mul(&mut ctx, &two); // elementwise, not a reduce
    let one = scalar(&mut ctx, 1.0);
    let applied = p.add(&mut ctx, &one);
    let effect = p.assign_effect(&mut ctx, &applied);
    Tensor::realize_many(&mut ctx, &mut sess, &[read.clone(), effect]);
    p.pin_buffer(&mut ctx, &mut sess);

    approx(&read.to_list::<f32>(&mut ctx, &mut sess), &[0.0, 0.0, 0.0]);
    approx(&p.to_list::<f32>(&mut ctx, &mut sess), &[1.0, 1.0, 1.0]);
}

/// Same but the read is a reduction (exercises the reduce-local path coexisting
/// with a fused parameter store), and the parameter is updated through an
/// optimizer-like co-realization.
#[test]
fn assign_effect_with_reduced_read() {
    let mut ctx = UOpCtx::new();
    let mut sess = Session::new();
    let p = Tensor::zeros_param(&mut ctx, [3], None, None);

    let loss = p.sum(&mut ctx, None, false, None);
    let one = scalar(&mut ctx, 1.0);
    let applied = p.add(&mut ctx, &one);
    let effect = p.assign_effect(&mut ctx, &applied);
    Tensor::realize_many(&mut ctx, &mut sess, &[loss.clone(), effect]);
    p.pin_buffer(&mut ctx, &mut sess);

    approx(&loss.to_list::<f32>(&mut ctx, &mut sess), &[0.0]);
    approx(&p.to_list::<f32>(&mut ctx, &mut sess), &[1.0, 1.0, 1.0]);
}
