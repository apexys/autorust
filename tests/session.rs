//! 9.1 — persistent realize state (`Session`): buffers + compiled-program cache
//! are reused across realizes instead of rebuilt per `to_list`/`to_bytes`.

use autorust::engine::realize::{buffer_bytes, buffer_copy_into, buffer_copyin};
use autorust::engine::session::Session;
use autorust::runtime::buffer::{BufTable, Buffer};
use autorust::tensor::Tensor;
use autorust::uop::ctx::UOpCtx;
use autorust::uop::ops::Ops;
use std::rc::Rc;

/// A second realize of the same tensor through one `Session` must not recompile
/// or allocate new base buffers (the tensor caches its realized buffer).
#[test]
fn session_realize_reuses_compile_and_buffers() {
    let mut ctx = UOpCtx::new();
    let mut sess = Session::new();

    let a = Tensor::from_slice(&mut ctx, &[1.0f32, 2.0, 3.0], None, None);
    let b = Tensor::from_slice(&mut ctx, &[4.0f32, 5.0, 6.0], None, None);
    let c = a.add(&mut ctx, &b);

    let v1 = c.to_list::<f32>(&mut ctx, &mut sess);
    assert_eq!(v1, vec![5.0, 7.0, 9.0]);
    let bases1 = sess.buffers.n_bases();
    let misses1 = sess.cache.misses;
    assert_eq!(misses1, 1, "one compute kernel should compile on the first realize");

    let v2 = c.to_list::<f32>(&mut ctx, &mut sess);
    assert_eq!(v2, v1);
    assert_eq!(sess.cache.misses, misses1, "second realize must not recompile");
    assert_eq!(sess.buffers.n_bases(), bases1, "second realize must not allocate new buffers");
}

/// A memoized realize must not be corrupted by later realizes in the same
/// session (each scheduled buffer uop owns its base, so no cross-schedule
/// aliasing on the concrete corpus).
#[test]
fn session_memo_survives_later_realizes() {
    let mut ctx = UOpCtx::new();
    let mut sess = Session::new();

    let a = Tensor::from_slice(&mut ctx, &[1.0f32, 2.0, 3.0, 4.0], None, None);
    let first = a.to_list::<f32>(&mut ctx, &mut sess);

    for i in 0..4 {
        let x = Tensor::from_slice(&mut ctx, &[10.0f32 + i as f32; 4], None, None);
        let y = Tensor::from_slice(&mut ctx, &[100.0f32; 4], None, None);
        let z = x.mul(&mut ctx, &y);
        let _ = z.to_list::<f32>(&mut ctx, &mut sess);
    }

    assert_eq!(a.to_list::<f32>(&mut ctx, &mut sess), first, "memoized buffer was corrupted by a later realize");
    assert_eq!(first, vec![1.0, 2.0, 3.0, 4.0]);
}

/// Two identical graphs built in **separate arenas** sharing one `ProgCache`
/// compile exactly once (the cache is keyed by AST content, not by uop id).
/// This is what would let compiled kernels survive a fresh `UOpCtx` per step.
#[test]
fn prog_cache_reuses_across_arenas() {
    let mut sess = Session::new();
    let mut values = Vec::new();
    for _ in 0..3 {
        let mut ctx = UOpCtx::new();
        // Buffers are keyed by arena-local uop ids, so reset them per arena
        // while keeping the content-keyed program cache.
        sess.buffers = BufTable::default();
        let a = Tensor::from_slice(&mut ctx, &[1.0f32, 2.0, 3.0], None, None);
        let b = Tensor::from_slice(&mut ctx, &[4.0f32, 5.0, 6.0], None, None);
        let c = a.mul(&mut ctx, &b);
        values.push(c.to_list::<f32>(&mut ctx, &mut sess));
    }
    assert_eq!(values[0], vec![4.0, 10.0, 18.0]);
    assert_eq!(values[0], values[1]);
    assert_eq!(values[1], values[2]);
    assert_eq!(sess.cache.misses, 1, "identical ASTs across arenas must compile once");
    assert_eq!(sess.cache.hits, 2, "the 2nd and 3rd arena must hit the cache");
    assert_eq!(sess.cache.len(), 1);
}

/// `buffer_copy_into` copies correctly whether the two buffers are on the same
/// device (direct device-to-device path) or on different devices (host
/// round-trip fallback). The interp backend backs every device name here.
#[test]
fn buffer_copy_into_same_and_cross_device() {
    let one = [1u8, 2, 3, 4];

    let a = Buffer::new_base("CPU", 1, autorust::dtype::FLOAT32);
    let b = Buffer::new_base("CPU", 1, autorust::dtype::FLOAT32);
    buffer_copyin(&a, &one);
    buffer_copy_into(&b, &a);
    assert_eq!(buffer_bytes(&b), one, "same-device copy");

    let src = Buffer::new_base("NV", 1, autorust::dtype::FLOAT32);
    let dst = Buffer::new_base("CPU", 1, autorust::dtype::FLOAT32);
    buffer_copyin(&src, &[5u8, 6, 7, 8]);
    buffer_copy_into(&dst, &src);
    assert_eq!(buffer_bytes(&dst), vec![5u8, 6, 7, 8], "cross-device copy");
}

/// 9.2 residual — the staged `initial_data` bytes and the attached autograd
/// `grad` live on the graph node, so both are freed with the tensor's subgraph
/// instead of leaking in a ctx-wide side table.
#[test]
fn node_side_data_dies_with_the_node() {
    let mut ctx = UOpCtx::new();
    let t = Tensor::from_slice(&mut ctx, &[1.0f32, 2.0, 3.0, 4.0], None, None);
    let bytes = {
        let mut u = t.uop.clone();
        while u.op != Ops::Buffer {
            u = u.srcs[0].clone();
        }
        Rc::downgrade(u.initial_data.borrow().as_ref().expect("staged bytes on the buffer node"))
    };

    {
        let g = ctx.cint(7);
        ctx.set_grad(&t.uop, g);
    }
    let grad_weak = {
        let b = t.uop.node.grad.borrow();
        Rc::downgrade(&b.as_ref().expect("grad attached to the node").node)
    };

    drop(t);
    assert!(bytes.upgrade().is_none(), "initial_data must be freed with its node");
    assert!(grad_weak.upgrade().is_none(), "grad must be freed with its node");
}
