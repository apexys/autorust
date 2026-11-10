//! `ldmatrix.m8n8.x2` fragment orientation for the bf16 attention kernel.
//!
//! Empirically pins the two layouts the kernel relies on (`g = lane >> 2`,
//! `t = lane & 3`, source `s[row][col]`):
//!
//! | instruction | r0 | r1 |
//! |---|---|---|
//! | `ldmatrix.x2`       | `{s[g][2t],   s[g][2t+1]}`   | `{s[8+g][2t], s[8+g][2t+1]}` |
//! | `ldmatrix.x2.trans` | `{s[2t][g],   s[2t+1][g]}`   | `{s[8+2t][g], s[9+2t][g]}`   |
//!
//! K is stored `[key][d]` and the logical B is `B[d][key]`, so it uses the
//! non-transposed form; V is stored `[key][d]` and the logical B is
//! `B[key][d]`, so it uses the transposed form. Getting this backwards is easy
//! (and the sawtooth test data hides it), hence the explicit check.

#![cfg(all(
    feature = "backend_cuda",
    any(target_os = "linux", target_os = "windows")
))]

use autorust::dtype;
use autorust::runtime::device::device;
use autorust::tensor::{Ctx, Tensor};

const SRC: &str = r#"
extern "C" __global__ void probe(const unsigned short* B, unsigned* R0, unsigned* R1,
                                 int trans) {
    /* B is a 16x8 row-major matrix s[k][n]. */
    __shared__ __align__(16) unsigned short s[16][8];
    int lane = threadIdx.x & 31;
    for (int i = threadIdx.x; i < 128; i += 32) s[i / 8][i % 8] = B[i];
    __syncwarp();
    int l = lane & 15;
    unsigned a;
    asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }"
        : "=r"(a) : "l"(&s[l][0]));
    unsigned r0, r1;
    if (trans) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                     : "=r"(r0), "=r"(r1) : "r"(a));
    } else {
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                     : "=r"(r0), "=r"(r1) : "r"(a));
    }
    R0[lane] = r0;
    R1[lane] = r1;
}
"#;

fn bf16_bits(value: u32) -> u16 {
    ((value as f32).to_bits() >> 16) as u16
}

fn pack(lo: u16, hi: u16) -> u32 {
    (lo as u32) | ((hi as u32) << 16)
}

#[test]
fn ldmatrix_x2_orientation() {
    if !autorust::runtime::ops_cuda::has_tensor_cores() {
        return;
    }
    let ctx = Ctx::new();
    let mut sess = autorust::engine::session::Session::new();
    // s[k][n] = k*8 + n + 1 (exact in bf16).
    let val = |k: usize, n: usize| bf16_bits((k * 8 + n + 1) as u32);
    let b: Vec<f32> = (0..128).map(|i| (i + 1) as f32).collect();
    let tb = Tensor::from_slice(&ctx, &b, None, Some("CUDA")).cast(dtype::BFLOAT16);
    let r0 = Tensor::from_slice(&ctx, &[0u32; 32], Some(dtype::UINT32), Some("CUDA")).to_param();
    let r1 = Tensor::from_slice(&ctx, &[0u32; 32], Some(dtype::UINT32), Some("CUDA")).to_param();
    {
        let mut c = ctx.borrow_mut();
        for t in [&tb, &r0, &r1] {
            t.realize_device_ctx(&mut c, &mut sess);
        }
    }
    let dev = device("CUDA");
    let ov = |t: &Tensor| t.realized_buffer().expect("realized").get_buf(&dev);
    for trans in [0i64, 1] {
        dev.run_host_kernel(
            SRC,
            "probe",
            &[
                (0, dtype::BFLOAT16),
                (1, dtype::UINT32),
                (2, dtype::UINT32),
                (3, dtype::INT32),
            ],
            &[ov(&tb), ov(&r0), ov(&r1)],
            [1, 1, 1],
            Some([32, 1, 1]),
            &[trans],
            true,
        )
        .expect("launch");
        let a0 = r0.to_list::<u32>(&mut sess);
        let a1 = r1.to_list::<u32>(&mut sess);
        for lane in 0..32 {
            let g = lane / 4;
            let t = lane % 4;
            let (want0, want1) = if trans == 0 {
                (
                    pack(val(g, 2 * t), val(g, 2 * t + 1)),
                    pack(val(8 + g, 2 * t), val(8 + g, 2 * t + 1)),
                )
            } else {
                (
                    pack(val(2 * t, g), val(2 * t + 1, g)),
                    pack(val(8 + 2 * t, g), val(9 + 2 * t, g)),
                )
            };
            assert_eq!(
                (a0[lane], a1[lane]),
                (want0, want1),
                "trans={trans} lane={lane}"
            );
        }
    }
}
