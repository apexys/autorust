//! CUDA driver+NVTRC smoke test (the `backend_cuda` S0 gate).
//!
//! Confirms cudarc dynamically loads `libcuda`/`libnvrtc`, allocates, compiles a
//! kernel with NVRTC, and launches it. Mirrors `ocl_smoke` for the CUDA path.

use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

fn main() {
    let ctx = CudaContext::new(0).expect("CudaContext::new");
    println!("CUDA: {}", ctx.name().expect("name"));
    let stream = ctx.default_stream();

    let src = r#"
extern "C" __global__ void add_kernel(float *out, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
"#;
    let ptx = compile_ptx(src).expect("nvrtc compile");
    let module = ctx.load_module(ptx).expect("load_module");
    let f = module.load_function("add_kernel").expect("load_function");

    let a = stream.clone_htod(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    let b = stream.clone_htod(&[10.0f32, 20.0, 30.0, 40.0]).unwrap();
    let out = stream.alloc_zeros::<f32>(4).unwrap();

    let mut builder = stream.launch_builder(&f);
    builder.arg(&out);
    builder.arg(&a);
    builder.arg(&b);
    builder.arg(&4i32);
    unsafe { builder.launch(LaunchConfig::for_num_elems(4)) }.expect("launch");

    let host: Vec<f32> = stream.clone_dtoh(&out).unwrap();
    println!("out = {host:?}");
    assert_eq!(host, vec![11.0, 22.0, 33.0, 44.0]);
    println!("cuda_smoke: OK");
}
