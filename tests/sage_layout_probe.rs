//! Fragment layout of `mma.m16n8k32.row.col.s32.s8.s8.s32`.
//!
//! Pins the layout the SageAttention kernel relies on (`g = lane >> 2`, `t =
//! lane & 3`, four consecutive 8-bit k values per 32-bit register):
//!
//! | A | B | C/D |
//! |---|---|---|
//! | a0 = A[g][4t:4t+4]      | b0 = B[4t:4t+4][g]       | c0 = C[g][2t]     |
//! | a1 = A[g+8][4t:4t+4]    | b1 = B[4t+16:4t+20][g]   | c1 = C[g][2t+1]   |
//! | a2 = A[g][4t+16:4t+20]  |                          | c2 = C[g+8][2t]   |
//! | a3 = A[g+8][4t+16:4t+20]|                          | c3 = C[g+8][2t+1] |
//!
//! The FP8 E4M3 instruction (`mma.m16n8k32...e4m3.e4m3`) uses the same
//! fragment layout, so this also pins the FP8 kernel (which can only run on
//! `sm_89+`).

#![cfg(all(
    feature = "backend_cuda",
    any(target_os = "linux", target_os = "windows")
))]

use autorust::dtype;
use autorust::runtime::device::device;
use autorust::tensor::{Ctx, Tensor};

const SRC: &str = r#"
__device__ __forceinline__ unsigned ar_pack4(signed char a, signed char b,
                                             signed char c, signed char d) {
    return (unsigned)(unsigned char)a | ((unsigned)(unsigned char)b << 8)
         | ((unsigned)(unsigned char)c << 16) | ((unsigned)(unsigned char)d << 24);
}
extern "C" __global__ void probe(const signed char* A, const signed char* B,
                                 int* C) {
    int lane = threadIdx.x & 31;
    int g = lane >> 2, t = lane & 3;
    unsigned a[4] = {
        ar_pack4(A[g*32 + 4*t + 0],     A[g*32 + 4*t + 1],
                 A[g*32 + 4*t + 2],     A[g*32 + 4*t + 3]),
        ar_pack4(A[(g+8)*32 + 4*t + 0], A[(g+8)*32 + 4*t + 1],
                 A[(g+8)*32 + 4*t + 2], A[(g+8)*32 + 4*t + 3]),
        ar_pack4(A[g*32 + 4*t + 16],    A[g*32 + 4*t + 17],
                 A[g*32 + 4*t + 18],    A[g*32 + 4*t + 19]),
        ar_pack4(A[(g+8)*32 + 4*t + 16], A[(g+8)*32 + 4*t + 17],
                 A[(g+8)*32 + 4*t + 18], A[(g+8)*32 + 4*t + 19]),
    };
    unsigned b[2] = {
        ar_pack4(B[(4*t + 0)*8 + g], B[(4*t + 1)*8 + g],
                 B[(4*t + 2)*8 + g], B[(4*t + 3)*8 + g]),
        ar_pack4(B[(4*t + 16)*8 + g], B[(4*t + 17)*8 + g],
                 B[(4*t + 18)*8 + g], B[(4*t + 19)*8 + g]),
    };
    int c[4] = {0,0,0,0};
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=r"(c[0]),"=r"(c[1]),"=r"(c[2]),"=r"(c[3])
                 : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),
                   "r"(b[0]),"r"(b[1]),
                   "r"(0),"r"(0),"r"(0),"r"(0));
    C[g*8+2*t+0] = c[0];
    C[g*8+2*t+1] = c[1];
    C[(g+8)*8+2*t+0] = c[2];
    C[(g+8)*8+2*t+1] = c[3];
}
"#;

#[test]
fn int8_m16n8k32_layout() {
    if !autorust::runtime::ops_cuda::has_tensor_cores() {
        return;
    }
    let ctx = Ctx::new();
    let mut sess = autorust::engine::session::Session::new();
    let a: Vec<i8> = (0..512).map(|i| ((i * 7) % 19) as i8 - 9).collect();
    let b: Vec<i8> = (0..256).map(|i| ((i * 5) % 13) as i8 - 6).collect();
    let mut want = vec![0i32; 128];
    for i in 0..16 {
        for j in 0..8 {
            let mut s = 0i32;
            for k in 0..32 {
                s += a[i * 32 + k] as i32 * b[k * 8 + j] as i32;
            }
            want[i * 8 + j] = s;
        }
    }
    let ta = Tensor::from_slice(&ctx, &a, Some(dtype::INT8), Some("CUDA"));
    let tb = Tensor::from_slice(&ctx, &b, Some(dtype::INT8), Some("CUDA"));
    let tc =
        Tensor::from_slice(&ctx, &vec![0i32; 128], Some(dtype::INT32), Some("CUDA")).to_param();
    {
        let mut c = ctx.borrow_mut();
        for t in [&ta, &tb, &tc] {
            t.realize_device_ctx(&mut c, &mut sess);
        }
    }
    let dev = device("CUDA");
    let ov = |t: &Tensor| t.realized_buffer().expect("realized").get_buf(&dev);
    dev.run_host_kernel(
        SRC,
        "probe",
        &[(0, dtype::INT8), (1, dtype::INT8), (2, dtype::INT32)],
        &[ov(&ta), ov(&tb), ov(&tc)],
        [1, 1, 1],
        Some([32, 1, 1]),
        &[],
        true,
    )
    .expect("launch");
    let got = tc.to_list::<i32>(&mut sess);
    assert_eq!(got, want, "m16n8k32 fragment layout mismatch");
}
