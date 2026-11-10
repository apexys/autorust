//! Fragment layout of `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`.
//!
//! Pins the (lane, register) → (row, column) maps the bf16 flash kernel relies
//! on (`g = lane >> 2`, `t = lane & 3`; each register holds two consecutive k
//! values):
//!
//! | A | B | C/D |
//! |---|---|---|
//! | a0 = A[g][2t:2t+2]      | b0 = B[2t:2t+2][g]    | c0 = C[g][2t]     |
//! | a1 = A[g+8][2t:2t+2]    | b1 = B[2t+8:2t+10][g] | c1 = C[g][2t+1]   |
//! | a2 = A[g][2t+8:2t+10]   |                       | c2 = C[g+8][2t]   |
//! | a3 = A[g+8][2t+8:2t+10] |                       | c3 = C[g+8][2t+1] |
//!
//! The C layout is therefore "two consecutive score tiles" in the A layout, so
//! `P·V` consumes the softmaxed scores from registers with no cross-lane shuffle.

#![cfg(all(
    feature = "backend_cuda",
    any(target_os = "linux", target_os = "windows")
))]

use autorust::dtype;
use autorust::runtime::device::device;
use autorust::tensor::{Ctx, Tensor};

const SRC: &str = r#"
__device__ __forceinline__ unsigned short ar_bf16(float x) {
    unsigned short r; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(x)); return r;
}
__device__ __forceinline__ unsigned ar_pack(float lo, float hi) {
    return (unsigned)ar_bf16(lo) | ((unsigned)ar_bf16(hi) << 16);
}
extern "C" __global__ void probe(const float* A, const float* Bc, float* C) {
    int lane = threadIdx.x & 31;
    int g = lane >> 2, t = lane & 3;
    unsigned a[4] = {
        ar_pack(A[g*16+2*t],       A[g*16+2*t+1]),
        ar_pack(A[(g+8)*16+2*t],   A[(g+8)*16+2*t+1]),
        ar_pack(A[g*16+2*t+8],     A[g*16+2*t+9]),
        ar_pack(A[(g+8)*16+2*t+8], A[(g+8)*16+2*t+9]),
    };
    /* Bc is col-major: Bc[n*16+k] = B[k][n]. */
    unsigned b[2] = {
        ar_pack(Bc[g*16+2*t],   Bc[g*16+2*t+1]),
        ar_pack(Bc[g*16+2*t+8], Bc[g*16+2*t+9]),
    };
    float c[4] = {0.f,0.f,0.f,0.f};
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(c[0]),"=f"(c[1]),"=f"(c[2]),"=f"(c[3])
                 : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),
                   "r"(b[0]),"r"(b[1]),
                   "f"(0.f),"f"(0.f),"f"(0.f),"f"(0.f));
    C[g*8+2*t+0] = c[0];
    C[g*8+2*t+1] = c[1];
    C[(g+8)*8+2*t+0] = c[2];
    C[(g+8)*8+2*t+1] = c[3];
}
"#;

#[test]
fn mma_m16n8k16_bf16_layout() {
    if !autorust::runtime::ops_cuda::has_tensor_cores() {
        return;
    }
    let ctx = Ctx::new();
    let mut sess = autorust::engine::session::Session::new();
    let a: Vec<f32> = (0..256).map(|i| ((i * 13) % 23) as f32 - 11.0).collect();
    let b: Vec<f32> = (0..128).map(|i| ((i * 7) % 17) as f32 - 8.0).collect();
    let bc: Vec<f32> = (0..128).map(|i| b[(i % 16) * 8 + (i / 16)]).collect();
    let mut want = vec![0.0f32; 128];
    for i in 0..16 {
        for j in 0..8 {
            let mut s = 0.0f32;
            for k in 0..16 {
                s += a[i * 16 + k] * b[k * 8 + j];
            }
            want[i * 8 + j] = s;
        }
    }
    let ta = Tensor::from_slice(&ctx, &a, None, Some("CUDA"));
    let tb = Tensor::from_slice(&ctx, &bc, None, Some("CUDA"));
    let tc = Tensor::from_slice(&ctx, &vec![0.0f32; 128], None, Some("CUDA")).to_param();
    for t in [&ta, &tb, &tc] {
        let _ = t.to_list::<f32>(&mut sess);
    }
    let dev = device("CUDA");
    let ov = |t: &Tensor| t.realized_buffer().expect("realized").get_buf(&dev);
    dev.run_host_kernel(
        SRC,
        "probe",
        &[
            (0, dtype::FLOAT32),
            (1, dtype::FLOAT32),
            (2, dtype::FLOAT32),
        ],
        &[ov(&ta), ov(&tb), ov(&tc)],
        [1, 1, 1],
        Some([32, 1, 1]),
        &[],
        true,
    )
    .expect("launch");
    let got = tc.to_list::<f32>(&mut sess);
    let maxd = got
        .iter()
        .zip(&want)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(maxd < 1e-2, "bf16 mma layout mismatch: {maxd:e}");
}
