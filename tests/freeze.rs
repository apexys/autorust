//! Freeze-plan gate: lower a captured JIT graph to a plan, execute it with the
//! loaded kernel table, and compare the bytes to the eager captured outputs.

#![cfg(feature = "freeze")]

use autorust::engine::jit::Jit;
use autorust::engine::session::Session;
use autorust::freeze::{FrozenRunner, ModuleForward, plan};
use autorust::tensor::{Ctx, Tensor};

fn decode(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn engine_selects_frozen_without_gpu() {
    struct MulModel {
        w: Tensor,
    }
    impl autorust::freeze::ModuleForward for MulModel {
        type Input = Tensor;
        type Output = Tensor;
        fn forward(&self, x: Tensor) -> Tensor {
            x.mul(&self.w)
        }
    }

    let ctx = Ctx::new();
    let mut sess = Session::new();
    sess.enable_jit();
    let x = Tensor::from_slice(&ctx, &[1.0f32, 2.0, 3.0, 4.0], None, Some("RUST")).to_param();
    let w = Tensor::from_slice(&ctx, &[2.0f32, 3.0, 4.0, 5.0], None, Some("RUST")).to_param();
    let _ = x.realize_fresh();
    let _ = w.realize_fresh();
    let mut jit = Jit::new();
    let outs = jit.call_many(&ctx, &mut sess, |_c, _s| vec![x.mul(&w)]);
    let built = plan::build(
        &ctx,
        &mut sess,
        &jit,
        std::slice::from_ref(&x),
        std::slice::from_ref(&w),
    );
    let want = outs[0].to_list::<f32>(&mut sess);

    let eager = MulModel { w: w.clone() };
    assert_eq!(eager.forward(x.clone()).to_list::<f32>(&mut sess), want);

    let frozen =
        autorust::freeze::Frozen::<MulModel>::new_jit(built.plan, built.progs, vec![w.clone()]);
    let mut engine = autorust::freeze::Engine::new(eager, frozen);
    assert!(!engine.is_eager(), "no GPU feature should select Frozen");
    assert_eq!(engine.forward(x.clone()).to_list::<f32>(&mut sess), want);
}

#[test]
fn freeze_plan_matches_eager_and_reuses_buffers() {
    let ctx = Ctx::new();
    let mut sess = Session::new();
    sess.enable_jit();
    let x = Tensor::from_slice(&ctx, &[1.0f32, 2.0, 3.0, 4.0], None, Some("RUST")).to_param();
    let w = Tensor::from_slice(&ctx, &[2.0f32, 3.0, 4.0, 5.0], None, Some("RUST")).to_param();
    let mut jit = Jit::new();
    let outs = jit.call_many(&ctx, &mut sess, |_c, _s| {
        let a = x.mul(&w);
        let b = a.add(&x);
        vec![a, b]
    });

    let built = plan::build(
        &ctx,
        &mut sess,
        &jit,
        std::slice::from_ref(&x),
        std::slice::from_ref(&w),
    );
    assert!(!built.plan.steps.is_empty());
    assert_eq!(built.plan.inputs.len(), 1);
    assert_eq!(built.plan.weights.len(), 1);
    assert_eq!(built.plan.outputs.len(), 2);
    assert_eq!(built.plan.external_count(), 2);

    let want0 = outs[0].to_list::<f32>(&mut sess);
    let want1 = outs[1].to_list::<f32>(&mut sess);
    let mut runner = FrozenRunner::new(built.plan, built.progs);
    let got = runner.run(std::slice::from_ref(&x), std::slice::from_ref(&w));
    assert_eq!(decode(&got[0]), want0);
    assert_eq!(decode(&got[1]), want1);

    // Second run reuses the internal allocations and produces the same bytes.
    let again = runner.run(std::slice::from_ref(&x), std::slice::from_ref(&w));
    assert_eq!(again, got);
}
