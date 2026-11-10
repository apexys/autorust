//! Runtime selection between the eager (GPU-capable) model and a frozen CPU
//! plan.
//!
//! ```ignore
//! impl autorust::freeze::ModuleForward for MyModel {
//!     type Input = Tensor;
//!     type Output = Tensor;
//!     fn forward(&self, x: Tensor) -> Tensor { MyModel::forward(self, &x) }
//! }
//!
//! let weights = model.named_params().map(|(_, t)| t).collect();
//! let frozen = autorust::freeze::Frozen::<MyModel>::new(plan(), KERNELS.to_vec(), weights);
//! let mut engine = autorust::freeze::Engine::new(model, frozen);
//! let y = engine.forward(x);
//! ```

use std::marker::PhantomData;

use crate::runtime::ops_rust::{KernelFn, RustProgram};
use crate::tensor::Tensor;

use super::plan::FrozenPlan;
use super::runner::FrozenRunner;

pub use crate::tensor::nn::ModuleForward;

/// A frozen plan plus its weights, runnable without the scheduler or rustc.
pub struct Frozen<M: ModuleForward> {
    runner: FrozenRunner,
    weights: Vec<Tensor>,
    _marker: PhantomData<M>,
}

impl<M: ModuleForward> Frozen<M> {
    /// Build a frozen executor from the generated plan/kernel table and the
    /// model's weight tensors (in the same order used at freeze time).
    pub fn new(plan: FrozenPlan, kernels: Vec<KernelFn>, weights: Vec<Tensor>) -> Frozen<M> {
        Frozen {
            runner: FrozenRunner::from_fns(plan, kernels),
            weights,
            _marker: PhantomData,
        }
    }

    /// The underlying runner (for diagnostics/custom multi-output calls).
    pub fn runner_mut(&mut self) -> &mut FrozenRunner {
        &mut self.runner
    }

    /// In-process variant using JIT-loaded kernels (`RustProgram`).
    pub fn new_jit(plan: FrozenPlan, progs: Vec<RustProgram>, weights: Vec<Tensor>) -> Frozen<M> {
        Frozen {
            runner: FrozenRunner::new(plan, progs),
            weights,
            _marker: PhantomData,
        }
    }
}

impl<M> Frozen<M>
where
    M: ModuleForward<Input = Tensor, Output = Tensor>,
{
    /// Run the frozen plan on one input tensor and return one output tensor.
    pub fn forward(&mut self, input: Tensor) -> Tensor {
        let bytes = self.runner.run(std::slice::from_ref(&input), &self.weights);
        let plan = self.runner.plan();
        let view = &plan.views[plan.outputs[0]];
        let shape = plan.output_shapes[0].clone();
        let dev = input.device();
        let ctx = input.ctx();
        let out = Tensor::from_bytes(ctx, &bytes[0], view.dtype, dev.as_deref());
        let mut c = ctx.borrow_mut();
        out.reshape_ctx(&mut c, shape)
    }
}

/// Eager model (GPU when available) with a frozen CPU fallback.
#[allow(clippy::large_enum_variant)]
pub enum Engine<M: ModuleForward> {
    /// The normal realized model path.
    Eager(M),
    /// The AOT/frozen CPU plan.
    Frozen(Frozen<M>),
}

impl<M: ModuleForward> Engine<M> {
    /// Select the eager path when a GPU backend is available, else frozen.
    pub fn new(model: M, frozen: Frozen<M>) -> Engine<M> {
        if gpu_available() {
            Engine::Eager(model)
        } else {
            Engine::Frozen(frozen)
        }
    }

    /// Whether the eager path was selected.
    pub fn is_eager(&self) -> bool {
        matches!(self, Engine::Eager(_))
    }
}

impl<M> Engine<M>
where
    M: ModuleForward<Input = Tensor, Output = Tensor>,
{
    /// Run one forward through the selected path.
    pub fn forward(&mut self, input: Tensor) -> Tensor {
        match self {
            Engine::Eager(model) => model.forward(input),
            Engine::Frozen(frozen) => frozen.forward(input),
        }
    }
}

fn gpu_available() -> bool {
    #[cfg(all(
        feature = "backend_cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    if crate::runtime::ops_cuda::available() {
        return true;
    }
    #[cfg(feature = "backend_cl")]
    if crate::runtime::ops_cl::available() {
        return true;
    }
    false
}
