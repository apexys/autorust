//! 9.1 — persistent realize state (`Session`): buffers + compiled-program cache
//! are reused across realizes instead of rebuilt per `to_list`/`to_bytes`.

use autorust::engine::realize::ProgCache;
use autorust::engine::session::Session;
use autorust::runtime::buffer::BufTable;
use autorust::tensor::Tensor;
use autorust::uop::ctx::UOpCtx;

/// A second realize of the same tensor through one `Session` must not recompile
/// or allocate new base buffers.
#[test]
fn session_realize_reuses_compile_and_buffers() {
    let mut ctx = UOpCtx::new();
    let mut sess = Session::new();

    let a = Tensor::from_slice(&mut ctx, &[1.0f32, 2.0, 3.0], None, None);
    let b = Tensor::from_slice(&mut ctx, &[4.0f32, 5.0, 6.0], None, None);
    let c = a.add(&mut ctx, &b);

    let v1 = c.to_list_in::<f32>(&mut ctx, &mut sess);
    assert_eq!(v1, vec![5.0, 7.0, 9.0]);
    let bases1 = sess.buffers.n_bases();
    let misses1 = sess.cache.misses;
    assert_eq!(misses1, 1, "one compute kernel should compile on the first realize");

    let v2 = c.to_list_in::<f32>(&mut ctx, &mut sess);
    assert_eq!(v2, v1);
    assert_eq!(sess.cache.misses, misses1, "second realize must not recompile");
    assert_eq!(sess.buffers.n_bases(), bases1, "second realize must not allocate new buffers");
    assert_eq!(sess.n_realized(), 1, "the tensor is memoized after the first realize");
}

/// A memoized realize must not be corrupted by later realizes in the same
/// session (each scheduled buffer uop owns its base, so no cross-schedule
/// aliasing on the concrete corpus).
#[test]
fn session_memo_survives_later_realizes() {
    let mut ctx = UOpCtx::new();
    let mut sess = Session::new();

    let a = Tensor::from_slice(&mut ctx, &[1.0f32, 2.0, 3.0, 4.0], None, None);
    let first = a.to_list_in::<f32>(&mut ctx, &mut sess);

    for i in 0..4 {
        let x = Tensor::from_slice(&mut ctx, &[10.0f32 + i as f32; 4], None, None);
        let y = Tensor::from_slice(&mut ctx, &[100.0f32; 4], None, None);
        let z = x.mul(&mut ctx, &y);
        let _ = z.to_list_in::<f32>(&mut ctx, &mut sess);
    }

    assert_eq!(a.to_list_in::<f32>(&mut ctx, &mut sess), first, "memoized buffer was corrupted by a later realize");
    assert_eq!(first, vec![1.0, 2.0, 3.0, 4.0]);
}

/// Two identical graphs built in **separate arenas** sharing one `ProgCache`
/// compile exactly once (the cache is keyed by AST content, not by uop id).
/// This is what would let compiled kernels survive a fresh `UOpCtx` per step.
#[test]
fn prog_cache_reuses_across_arenas() {
    let mut cache = ProgCache::default();
    let mut values = Vec::new();
    for _ in 0..3 {
        let mut ctx = UOpCtx::new();
        let a = Tensor::from_slice(&mut ctx, &[1.0f32, 2.0, 3.0], None, None);
        let b = Tensor::from_slice(&mut ctx, &[4.0f32, 5.0, 6.0], None, None);
        let c = a.mul(&mut ctx, &b);
        let mut bufs = BufTable::default();
        values.push(c.to_list_with::<f32>(&mut ctx, &mut bufs, &mut cache));
    }
    assert_eq!(values[0], vec![4.0, 10.0, 18.0]);
    assert_eq!(values[0], values[1]);
    assert_eq!(values[1], values[2]);
    assert_eq!(cache.misses, 1, "identical ASTs across arenas must compile once");
    assert_eq!(cache.hits, 2, "the 2nd and 3rd arena must hit the cache");
    assert_eq!(cache.len(), 1);
}
