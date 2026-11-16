//! Bounded tensor-core implicit-GEMM convolution for CUDA.
//!
//! One block computes a 64-row (batch*oy*ox) by 64-output-channel tile with
//! four warps, each running a 16x16 TF32 `wmma` tile. The patches are generated
//! directly into the A staging buffer, so there is no im2col copy in global
//! memory. The K loop is `cin*kh*kw`; the B tile is the corresponding weight
//! taps.
//!
//! Covered: square `k x k`, `groups=1`, `stride` 1 or 2, `dilation=1`,
//! `padding=(k-1)/2`, f32 in/out. Everything else keeps the graph convolution.
//! This is a forward-only host kernel, so training keeps the autograd path.
//!
//! The dispatch only picks it when the graph would otherwise use the slow
//! direct convolution (`cin*k*k` below the im2col floor, e.g. ResNet's 7x7
//! stem at K=147) or for the measured 3x3 stride-1 case.

/// Threads per block: four warps, one 16x16 output tile each.
pub const THREADS: i64 = 128;
/// Rows per block.
pub const M_TILE: i64 = 64;
/// Output channels per block.
pub const N_TILE: i64 = 64;
/// Smallest output worth a host launch.
pub const MIN_OUT: i64 = 1 << 14;
/// Kernel entry point.
pub const NAME: &str = "ar_conv_tc";

/// CUDA source.
pub const SRC: &str = r#"
#include <mma.h>
using namespace nvcuda;

extern "C" __global__ void ar_conv_tc(
    const float* __restrict__ X, const float* __restrict__ WT,
    const float* __restrict__ Bias, float* __restrict__ Y,
    int B, int C, int H, int Wd, int O, int has_bias,
    int KH, int KW, int pad, int stride) {
  const int warp = threadIdx.x >> 5;
  const int m0 = blockIdx.x * 64 + warp * 16;
  const int n0 = blockIdx.y * 64;
  const int OH = (H + 2*pad - KH) / stride + 1;
  const int OW = (Wd + 2*pad - KW) / stride + 1;
  const int KS = KH * KW, K = C * KS;
  __shared__ float As[64][8];
  __shared__ float Bs[8][64];
  __shared__ float Cs[64][64];

  wmma::fragment<wmma::accumulator,16,16,8,float> acc;
  wmma::fill_fragment(acc, 0.0f);

  for (int kb = 0; kb < K; kb += 8) {
    for (int i = threadIdx.x; i < 64*8; i += blockDim.x) {
      int r = i / 8, kk = i % 8;
      int k = kb + kk;
      int m = m0 + r;
      float v = 0.0f;
      if (k < K && m < B*OH*OW) {
        int bi = m / (OH*OW);
        int rem = m % (OH*OW);
        int oy = rem / OW, ox = rem % OW;
        int ci = k / KS, tap = k % KS;
        int ky = tap / KW, kx = tap % KW;
        int iy = oy * stride + ky - pad, ix = ox * stride + kx - pad;
        if (iy >= 0 && iy < H && ix >= 0 && ix < Wd)
          v = X[((bi*C + ci)*H + iy)*Wd + ix];
      }
      As[r][kk] = v;
    }
    for (int i = threadIdx.x; i < 8*64; i += blockDim.x) {
      int kk = i / 64, n = i % 64;
      int k = kb + kk;
      float v = 0.0f;
      if (k < K && n0 + n < O) {
        int ci = k / KS, tap = k % KS;
        v = WT[((n0+n)*C + ci)*KS + tap];
      }
      Bs[kk][n] = v;
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> a;
    wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> b;
    wmma::load_matrix_sync(a, &As[warp*16][0], 8);
    wmma::load_matrix_sync(b, &Bs[0][warp*16], 64);
    wmma::mma_sync(acc, a, b, acc);
    __syncthreads();
  }
  wmma::store_matrix_sync(&Cs[warp*16][warp*16], acc, 64, wmma::mem_row_major);
  __syncthreads();
  for (int i = threadIdx.x; i < 64*64; i += blockDim.x) {
    int r = i / 64, n = i % 64;
    int m = m0 + r;
    if (m < B*OH*OW && n0 + n < O) {
      int bi = m / (OH*OW);
      int rem = m % (OH*OW);
      int oy = rem / OW, ox = rem % OW;
      float v = Cs[r][n];
      if (has_bias) v += Bias[n0+n];
      Y[((bi*O + (n0+n))*OH + oy)*OW + ox] = v;
    }
  }
}
"#;

/// `(slot, dtype)`: four pointers, then eight int scalars.
pub fn signature() -> Vec<(i64, crate::dtype::DTypeId)> {
    use crate::dtype::{FLOAT32, INT32};
    vec![
        (0, FLOAT32),
        (1, FLOAT32),
        (2, FLOAT32),
        (3, FLOAT32),
        (4, INT32),
        (5, INT32),
        (6, INT32),
        (7, INT32),
        (8, INT32),
        (9, INT32),
        (10, INT32),
        (11, INT32),
        (12, INT32),
        (13, INT32),
    ]
}

/// Captured launch for one tensor-core convolution.
#[allow(clippy::too_many_arguments)]
pub fn host_call(
    bufs: Vec<crate::runtime::buffer::Buffer>,
    b: i64,
    c: i64,
    h: i64,
    w: i64,
    o: i64,
    has_bias: bool,
    kh: i64,
    kw: i64,
    pad: i64,
    stride: i64,
) -> crate::engine::session::HostCall {
    let oh = (h + 2 * pad - kh) / stride + 1;
    let ow = (w + 2 * pad - kw) / stride + 1;
    let rows = b * oh * ow;
    crate::engine::session::HostCall {
        device: "CUDA".into(),
        src: SRC.to_string(),
        name: NAME.to_string(),
        signature: signature(),
        bufs,
        global: [(rows + M_TILE - 1) / M_TILE, (o + N_TILE - 1) / N_TILE, 1],
        local: Some([THREADS, 1, 1]),
        vals: vec![b, c, h, w, o, has_bias as i64, kh, kw, pad, stride],
    }
}
