//! A cooperative-matrix GEMM module for the `VK` backend.
//!
//! `renderer/spirv.rs` does not emit `OpCooperativeMatrix*KHR` yet, so the
//! tensor-core path ships as one pre-assembled module whose three shape
//! constants (M, N, K) are patched in at load time (see [`module`]). The
//! readable source of truth is `tests/data/coopmat_gemm.spvasm`; the words below
//! were produced from it by
//!
//! ```text
//! sed 's/\$M\$/1196444482/;s/\$N\$/1314344526/;s/\$K\$/1263947082/' \
//!     tests/data/coopmat_gemm.spvasm > /tmp/sent.spvasm
//! spirv-as --target-env spv1.6 /tmp/sent.spvasm -o /tmp/sent.spv
//! ```
//!
//! and `tests/vk_coopmat.rs` re-assembles the fixture and compares, so the two
//! cannot drift.
//!
//! The kernel is a 16x64-tile GEMM for one 32-lane subgroup per tile:
//! `C[M,N] = A[M,K] * B[K,N]` with 16-bit inputs (row-major, row stride `N`) and
//! an f32 accumulator, four `OpCooperativeMatrixMulAddKHR` per A load. The last
//! tile in each direction is clamped to the tensor edge, so `M`/`N` need not be
//! multiples of the tile size (the overlap recomputes identical values).

/// Sentinel value of the `M` constant in [`GEMM_WORDS`].
pub const SENTINEL_M: u32 = 1196444482;
/// Sentinel value of the `N` constant.
pub const SENTINEL_N: u32 = 1314344526;
/// Sentinel value of the `K` constant.
pub const SENTINEL_K: u32 = 1263947082;
/// Sentinel value of A's row stride.
pub const SENTINEL_SA: u32 = 1515870810;

/// The assembled module, with the three shape constants left as sentinels.
pub const GEMM_WORDS: &[u32] = &[
    0x07230203, 0x00010600, 0x00070000, 0x00000068, 0x00000000, 0x00020011, 0x00000001, 0x00020011,
    0x00000009, 0x00020011, 0x00001151, 0x00020011, 0x00001786, 0x00020011, 0x000014e1, 0x0007000a,
    0x5f565053, 0x5f52484b, 0x69623631, 0x74735f74, 0x6761726f, 0x00000065, 0x0008000a, 0x5f565053,
    0x5f52484b, 0x706f6f63, 0x74617265, 0x5f657669, 0x7274616d, 0x00007869, 0x0003000e, 0x00000000,
    0x00000003, 0x0009000f, 0x00000005, 0x00000001, 0x6e69616d, 0x00000000, 0x00000002, 0x00000003,
    0x00000004, 0x00000005, 0x00060010, 0x00000001, 0x00000011, 0x00000020, 0x00000001, 0x00000001,
    0x00040047, 0x00000002, 0x0000000b, 0x0000001a, 0x00040047, 0x00000006, 0x00000006, 0x00000002,
    0x00040047, 0x00000007, 0x00000006, 0x00000002, 0x00040047, 0x00000008, 0x00000006, 0x00000004,
    0x00050048, 0x00000009, 0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x0000000a, 0x00000000,
    0x00000023, 0x00000000, 0x00050048, 0x0000000b, 0x00000000, 0x00000023, 0x00000000, 0x00030047,
    0x00000009, 0x00000002, 0x00030047, 0x0000000a, 0x00000002, 0x00030047, 0x0000000b, 0x00000002,
    0x00040047, 0x00000003, 0x00000022, 0x00000000, 0x00040047, 0x00000003, 0x00000021, 0x00000000,
    0x00040047, 0x00000004, 0x00000022, 0x00000000, 0x00040047, 0x00000004, 0x00000021, 0x00000001,
    0x00040047, 0x00000005, 0x00000022, 0x00000000, 0x00040047, 0x00000005, 0x00000021, 0x00000002,
    0x00020013, 0x0000000c, 0x00030021, 0x0000000d, 0x0000000c, 0x00040015, 0x0000000e, 0x00000020,
    0x00000000, 0x00020014, 0x0000000f, 0x00030016, 0x00000010, 0x00000010, 0x00030016, 0x00000011,
    0x00000020, 0x00040017, 0x00000012, 0x0000000e, 0x00000003, 0x0004002b, 0x0000000e, 0x00000013,
    0x00000010, 0x0004002b, 0x0000000e, 0x00000014, 0x00000000, 0x0004002b, 0x0000000e, 0x00000015,
    0x4e574e4e, 0x0004002b, 0x0000000e, 0x00000016, 0x4b564d4a, 0x0004002b, 0x0000000e, 0x00000017,
    0x47504b42, 0x0004002b, 0x0000000e, 0x00000018, 0x5a5a5a5a, 0x0004002b, 0x0000000e, 0x00000019,
    0x00000010, 0x0004002b, 0x0000000e, 0x0000001a, 0x00000040, 0x0004002b, 0x0000000e, 0x0000001b,
    0x00000003, 0x0004002b, 0x0000000e, 0x0000001c, 0x00000000, 0x0004002b, 0x0000000e, 0x0000001d,
    0x00000001, 0x0004002b, 0x0000000e, 0x0000001e, 0x00000002, 0x0004002b, 0x0000000e, 0x0000001f,
    0x00000040, 0x0004002b, 0x0000000e, 0x00000020, 0x00000020, 0x0004002b, 0x0000000e, 0x00000021,
    0x00000030, 0x0004002b, 0x00000011, 0x00000022, 0x00000000, 0x0003001d, 0x00000006, 0x00000010,
    0x0003001d, 0x00000007, 0x00000010, 0x0003001d, 0x00000008, 0x00000011, 0x0003001e, 0x00000009,
    0x00000006, 0x0003001e, 0x0000000a, 0x00000007, 0x0003001e, 0x0000000b, 0x00000008, 0x00040020,
    0x00000023, 0x0000000c, 0x00000009, 0x00040020, 0x00000024, 0x0000000c, 0x0000000a, 0x00040020,
    0x00000025, 0x0000000c, 0x0000000b, 0x00040020, 0x00000026, 0x0000000c, 0x00000010, 0x00040020,
    0x00000027, 0x0000000c, 0x00000011, 0x00040020, 0x00000028, 0x00000001, 0x00000012, 0x0004003b,
    0x00000028, 0x00000002, 0x00000001, 0x0004003b, 0x00000023, 0x00000003, 0x0000000c, 0x0004003b,
    0x00000024, 0x00000004, 0x0000000c, 0x0004003b, 0x00000025, 0x00000005, 0x0000000c, 0x00071168,
    0x00000029, 0x00000010, 0x0000001b, 0x00000013, 0x00000013, 0x0000001c, 0x00071168, 0x0000002a,
    0x00000010, 0x0000001b, 0x00000013, 0x00000013, 0x0000001d, 0x00071168, 0x0000002b, 0x00000011,
    0x0000001b, 0x00000013, 0x00000013, 0x0000001e, 0x00050036, 0x0000000c, 0x00000001, 0x00000000,
    0x0000000d, 0x000200f8, 0x0000002c, 0x0004003d, 0x00000012, 0x0000002d, 0x00000002, 0x00050051,
    0x0000000e, 0x0000002e, 0x0000002d, 0x00000000, 0x00050051, 0x0000000e, 0x0000002f, 0x0000002d,
    0x00000001, 0x00050084, 0x0000000e, 0x00000030, 0x0000002e, 0x00000013, 0x00050084, 0x0000000e,
    0x00000031, 0x0000002f, 0x0000001f, 0x00050082, 0x0000000e, 0x00000032, 0x00000017, 0x00000019,
    0x000500ac, 0x0000000f, 0x00000033, 0x00000030, 0x00000032, 0x000600a9, 0x0000000e, 0x00000034,
    0x00000033, 0x00000032, 0x00000030, 0x00050082, 0x0000000e, 0x00000035, 0x00000015, 0x0000001a,
    0x000500ac, 0x0000000f, 0x00000036, 0x00000031, 0x00000035, 0x000600a9, 0x0000000e, 0x00000037,
    0x00000036, 0x00000035, 0x00000031, 0x00040050, 0x0000002b, 0x00000038, 0x00000022, 0x00040050,
    0x0000002b, 0x00000039, 0x00000022, 0x00040050, 0x0000002b, 0x0000003a, 0x00000022, 0x00040050,
    0x0000002b, 0x0000003b, 0x00000022, 0x000200f9, 0x0000003c, 0x000200f8, 0x0000003c, 0x000700f5,
    0x0000000e, 0x0000003d, 0x00000014, 0x0000002c, 0x0000003e, 0x0000003f, 0x000700f5, 0x0000002b,
    0x00000040, 0x00000038, 0x0000002c, 0x00000041, 0x0000003f, 0x000700f5, 0x0000002b, 0x00000042,
    0x00000039, 0x0000002c, 0x00000043, 0x0000003f, 0x000700f5, 0x0000002b, 0x00000044, 0x0000003a,
    0x0000002c, 0x00000045, 0x0000003f, 0x000700f5, 0x0000002b, 0x00000046, 0x0000003b, 0x0000002c,
    0x00000047, 0x0000003f, 0x000400f6, 0x00000048, 0x0000003f, 0x00000000, 0x000200f9, 0x0000003f,
    0x000200f8, 0x0000003f, 0x00050084, 0x0000000e, 0x00000049, 0x00000034, 0x00000018, 0x00050080,
    0x0000000e, 0x0000004a, 0x00000049, 0x0000003d, 0x00050084, 0x0000000e, 0x0000004b, 0x0000003d,
    0x00000015, 0x00050080, 0x0000000e, 0x0000004c, 0x0000004b, 0x00000037, 0x00060041, 0x00000026,
    0x0000004d, 0x00000003, 0x00000014, 0x0000004a, 0x00060041, 0x00000026, 0x0000004e, 0x00000004,
    0x00000014, 0x0000004c, 0x00061169, 0x00000029, 0x0000004f, 0x0000004d, 0x00000014, 0x00000018,
    0x00061169, 0x0000002a, 0x00000050, 0x0000004e, 0x00000014, 0x00000015, 0x0006116b, 0x0000002b,
    0x00000041, 0x0000004f, 0x00000050, 0x00000040, 0x00050080, 0x0000000e, 0x00000051, 0x0000004c,
    0x00000013, 0x00060041, 0x00000026, 0x00000052, 0x00000004, 0x00000014, 0x00000051, 0x00061169,
    0x0000002a, 0x00000053, 0x00000052, 0x00000014, 0x00000015, 0x0006116b, 0x0000002b, 0x00000043,
    0x0000004f, 0x00000053, 0x00000042, 0x00050080, 0x0000000e, 0x00000054, 0x0000004c, 0x00000020,
    0x00060041, 0x00000026, 0x00000055, 0x00000004, 0x00000014, 0x00000054, 0x00061169, 0x0000002a,
    0x00000056, 0x00000055, 0x00000014, 0x00000015, 0x0006116b, 0x0000002b, 0x00000045, 0x0000004f,
    0x00000056, 0x00000044, 0x00050080, 0x0000000e, 0x00000057, 0x0000004c, 0x00000021, 0x00060041,
    0x00000026, 0x00000058, 0x00000004, 0x00000014, 0x00000057, 0x00061169, 0x0000002a, 0x00000059,
    0x00000058, 0x00000014, 0x00000015, 0x0006116b, 0x0000002b, 0x00000047, 0x0000004f, 0x00000059,
    0x00000046, 0x00040053, 0x0000002b, 0x0000005a, 0x00000041, 0x00040053, 0x0000002b, 0x0000005b,
    0x00000043, 0x00040053, 0x0000002b, 0x0000005c, 0x00000045, 0x00040053, 0x0000002b, 0x0000005d,
    0x00000047, 0x00050080, 0x0000000e, 0x0000003e, 0x0000003d, 0x00000013, 0x000500b0, 0x0000000f,
    0x0000005e, 0x0000003e, 0x00000016, 0x000400fa, 0x0000005e, 0x0000003c, 0x00000048, 0x000200f8,
    0x00000048, 0x00050084, 0x0000000e, 0x0000005f, 0x00000034, 0x00000015, 0x00050080, 0x0000000e,
    0x00000060, 0x0000005f, 0x00000037, 0x00060041, 0x00000027, 0x00000061, 0x00000005, 0x00000014,
    0x00000060, 0x0005116a, 0x00000061, 0x0000005a, 0x00000014, 0x00000015, 0x00050080, 0x0000000e,
    0x00000062, 0x00000060, 0x00000013, 0x00060041, 0x00000027, 0x00000063, 0x00000005, 0x00000014,
    0x00000062, 0x0005116a, 0x00000063, 0x0000005b, 0x00000014, 0x00000015, 0x00050080, 0x0000000e,
    0x00000064, 0x00000060, 0x00000020, 0x00060041, 0x00000027, 0x00000065, 0x00000005, 0x00000014,
    0x00000064, 0x0005116a, 0x00000065, 0x0000005c, 0x00000014, 0x00000015, 0x00050080, 0x0000000e,
    0x00000066, 0x00000060, 0x00000021, 0x00060041, 0x00000027, 0x00000067, 0x00000005, 0x00000014,
    0x00000066, 0x0005116a, 0x00000067, 0x0000005d, 0x00000014, 0x00000015, 0x000100fd, 0x00010038,
];

/// Tile shape of the module: `M_TILE x N_TILE` output, `K_TILE` per MMA.
pub const M_TILE: usize = 16;
/// Columns of the output tile.
pub const N_TILE: usize = 64;
/// Contraction depth of one `OpCooperativeMatrixMulAddKHR`.
pub const K_TILE: usize = 16;

/// The module specialised to `C[m,n] = A[m,k] * B[k,n]`.
///
/// Returns `None` when the shape cannot be tiled (the caller then keeps the
/// generated kernel) or the module does not contain each sentinel exactly once.
pub fn module(m: usize, n: usize, k: usize) -> Option<Vec<u32>> {
    if m < M_TILE || n < N_TILE || k < K_TILE || !k.is_multiple_of(K_TILE) {
        return None;
    }
    let mut out = GEMM_WORDS.to_vec();
    // A's row stride is its own width (contiguous row-major), which is what
    // `eligible` requires of the descriptor.
    for (sent, val) in [
        (SENTINEL_M, m),
        (SENTINEL_N, n),
        (SENTINEL_K, k),
        (SENTINEL_SA, k),
    ] {
        let mut hits = out.iter_mut().filter(|w| **w == sent);
        let first = hits.next()?;
        if hits.next().is_some() {
            return None;
        }
        *first = val as u32;
    }
    Some(out)
}

/// Whether a library-GEMM descriptor can run as [`module`].
///
/// The module takes f16 operands and accumulates in f32, so the graph must have
/// asked for an f32 dot: a plain 16-bit dot (which accumulates in 16 bits) never
/// takes this path, so the numerics of such a graph are unchanged. It also needs
/// contiguous row-major tensors whose row strides are their own widths, no
/// epilogue (`alpha`/bias/batch), and a shape that tiles.
pub fn eligible(g: &crate::schedule::host_gemm::HostGemm) -> bool {
    use crate::dtype;
    g.dtype == dtype::FLOAT16
        && g.c_dtype == dtype::FLOAT32
        && g.batch == 1
        && g.alpha == 1.0
        && g.bias_slot < 0
        && g.a_k == 1
        && g.b_n == 1
        && g.c_n == 1
        && g.a_m == g.k
        && g.b_k == g.n
        && g.c_m == g.n
        && g.a_off >= 0
        && g.b_off >= 0
        && g.c_off >= 0
        && module(g.m as usize, g.n as usize, g.k as usize).is_some()
}

/// Workgroup count for [`module`] over an `m x n` output tile grid.
pub fn grid(m: usize, n: usize) -> [u32; 3] {
    [
        m.div_ceil(M_TILE) as u32,
        n.div_ceil(N_TILE) as u32,
        1,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_replaces_each_sentinel_once() {
        let w = module(1576, 768, 3072).expect("patch");
        assert_eq!(w[0], 0x0723_0203, "SPIR-V magic");
        assert_eq!(w[1], 0x0001_0600, "SPIR-V 1.6 (OpCooperativeMatrixKHR)");
        assert!(!w.contains(&SENTINEL_M));
        assert!(!w.contains(&SENTINEL_N));
        assert!(!w.contains(&SENTINEL_K));
        assert!(!w.contains(&SENTINEL_SA));
        assert!(w.contains(&1576) && w.contains(&3072));
        assert_eq!(grid(1576, 3072), [99, 48, 1]);
        assert_eq!(grid(16, 64), [1, 1, 1]);
    }

    #[test]
    fn rejects_untileable_shapes() {
        assert!(module(8, 768, 768).is_none()); // m < tile
        assert!(module(1576, 32, 768).is_none()); // n < tile
        assert!(module(1576, 768, 760).is_none()); // k not a multiple of 16
    }
}
