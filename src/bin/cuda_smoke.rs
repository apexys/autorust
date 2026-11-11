//! CUDA driver+NVTRC smoke test (the `backend_cuda` S0 gate).
//!
//! Confirms cudarc dynamically loads `libcuda`/`libnvrtc`, allocates, compiles a
//! kernel with NVRTC, and launches it. Mirrors `ocl_smoke` for the CUDA path.

#[cfg(all(
    feature = "backend_cuda",
    any(target_os = "linux", target_os = "windows")
))]
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};

#[cfg(all(
    feature = "backend_cuda",
    any(target_os = "linux", target_os = "windows")
))]
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
    let (major, minor) = ctx.compute_capability().expect("compute capability");
    let arch = format!("sm_{major}{minor}");
    let (_, module) = autorust::runtime::nvrtc::compile_validated(
        src,
        &arch,
        &[
            "/usr/local/cuda/include".to_string(),
            "/usr/include".to_string(),
            "/opt/cuda/include".to_string(),
        ],
        |ptx| {
            let text = String::from_utf8_lossy(ptx).into_owned();
            ctx.load_module(cudarc::nvrtc::Ptx::from_src(text))
                .map_err(|e| e.to_string())
        },
    )
    .expect("nvrtc compile");
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

#[cfg(not(all(
    feature = "backend_cuda",
    any(target_os = "linux", target_os = "windows")
)))]
fn main() {
    eprintln!("cuda_smoke: the CUDA backend is only available on Linux and Windows");
}
