//! Fragment layout of `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`.
//!
//! The warp-tiled flash kernel depends on the exact (lane, register) → (row,
//! column) maps; they are not obvious from the instruction name, so this pins
//! them down with a single 16x8x8 matmul. It also documents the one useful
//! consequence: the accumulator layout is the A-operand layout with registers
//! 1 and 2 swapped, which is why the flash kernel feeds the softmaxed scores
//! straight back into `P·V` without a shared-memory round trip.
//!
//! Layouts (lane `l`, `g = l >> 2`, `t = l & 3`):
//!
//! | A | B | C/D |
//! |---|---|---|
//! | a0 = A[g][2t]     | b0 = B[2t][g]     | c0 = C[g][2t]     |
//! | a1 = A[g+8][2t]   | b1 = B[2t+1][g]   | c1 = C[g][2t+1]   |
//! | a2 = A[g][2t+1]   |                   | c2 = C[g+8][2t]   |
//! | a3 = A[g+8][2t+1] |                   | c3 = C[g+8][2t+1] |

#![cfg(feature = "backend_cuda")]

use autorust::dtype;
use autorust::runtime::device::device;
use autorust::tensor::{Ctx, Tensor};

const SRC: &str = r#"
__device__ __forceinline__ unsigned ar_tf32(float x) {
    unsigned r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r;
}
extern "C" __global__ void probe(const float* A, const float* Bc, float* C) {
    int lane = threadIdx.x & 31;
    int g = lane >> 2, t = lane & 3;
    unsigned a[4] = { ar_tf32(A[g*8+2*t]), ar_tf32(A[(g+8)*8+2*t]),
                      ar_tf32(A[g*8+2*t+1]), ar_tf32(A[(g+8)*8+2*t+1]) };
    /* Bc is col-major: Bc[n*8+k] = B[k][n]. */
    unsigned b[2] = { ar_tf32(Bc[g*8+2*t]), ar_tf32(Bc[g*8+2*t+1]) };
    float c[4] = {0.f,0.f,0.f,0.f};
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
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
fn mma_m16n8k8_fragment_layout() {
    if !autorust::runtime::ops_cuda::has_tensor_cores() {
        return;
    }
    let ctx = Ctx::new();
    let mut sess = autorust::engine::session::Session::new();
    let a: Vec<f32> = (0..128)
        .map(|i| ((i * 7) % 19) as f32 * 0.25 - 2.0)
        .collect();
    let b: Vec<f32> = (0..64)
        .map(|i| ((i * 11) % 17) as f32 * 0.5 - 3.0)
        .collect();
    let bc: Vec<f32> = (0..64).map(|i| b[(i % 8) * 8 + (i / 8)]).collect();
    let mut want = vec![0.0f32; 128];
    for i in 0..16 {
        for j in 0..8 {
            let mut s = 0.0f32;
            for k in 0..8 {
                s += a[i * 8 + k] * b[k * 8 + j];
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
    // TF32 rounds both operands, so the reference is only accurate to ~1e-3.
    let maxd = got
        .iter()
        .zip(&want)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(maxd < 1e-2, "mma fragment layout mismatch: {maxd:e}");
}
