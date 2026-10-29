use autorust::uop::ctx::UOpCtx;
use autorust::tensor::{Ctx, Tensor};
use autorust::engine::realize::{tensor_values, ProgCache};
use autorust::runtime::buffer::BufTable;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn main() {
    let res = catch_unwind(AssertUnwindSafe(|| {
        let mut ctx = Ctx::new();
        let t = Tensor::arange(&ctx, 0.0, Some(6.0), 1.0, None).reshape([2usize, 3]).sum(None, false, None);
        let mut cache = ProgCache::default();
        let vals = tensor_values(&mut ctx.raw(), t.uop(), &mut BufTable::default(), &mut cache);
        println!("VALUES: {vals:?}");
    }));
    match res {
        Ok(()) => {}
        Err(e) => {
            let msg = e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()));
            eprintln!("PANIC: {msg:?}");
        }
    }
}
