//! `RandMixin` port (`mixin/rand.py`) + the stateful per-device RNG counters
//! (`tensor.py:_next_counter`).
//!
//! tinygrad builds `Tensor.rand` as a *graph* of uint32 Threefry + bit-level
//! mantissa assembly. We reproduce the exact same integer sequence on the host
//! (u32/u64 wrapping math) and materialize the result via `from_slice`: the
//! INTERP backend stores every value as `f64`, which cannot carry a uint64
//! bit pattern or perform `bitcast`, so the graph route cannot run there. The
//! `codegen::decomp::threefry2x32` rule is the faithful graph lowering used by
//! hardware backends; this module is the reference host evaluator.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::dtype::{self, DTypeId};
use crate::dtype::ConstVal;
use crate::uop::ctx::UOpCtx;
use crate::uop::sint::SVal;

use super::creation;
use super::Tensor;

// ---------------------------------------------------------------------------
// module-level RNG state (thread-local, mirrors tinygrad's class-level state)
// ---------------------------------------------------------------------------

#[derive(Default)]
struct RngState {
    seed: i64,
    device_seeds: HashMap<String, (u32, u32)>,
    counters: HashMap<String, (u32, u32)>,
}

thread_local! {
    static RNG: RefCell<RngState> = RefCell::new(RngState { seed: 0, ..Default::default() });
}

/// `Tensor.manual_seed(seed)` (tensor.py:624): reset seed + device RNG state.
pub fn manual_seed(seed: i64) {
    RNG.with(|r| {
        let mut r = r.borrow_mut();
        r.seed = seed;
        r.device_seeds.clear();
        r.counters.clear();
    });
}

/// `Tensor._next_counter(device, num)` (tensor.py:642): on first use of a
/// device, seed it from `sha256(len(_device_seeds) as 4 bytes)` + `Tensor._seed`
/// (truncated to uint32); advance the counter by `num`; return
/// `(key, pre_increment_counter)` as host u32 pairs.
fn next_counter(device: &str, num: u32) -> ((u32, u32), (u32, u32)) {
    RNG.with(|r| {
        let mut r = r.borrow_mut();
        if !r.device_seeds.contains_key(device) {
            let n = r.device_seeds.len() as u32;
            let digest = sha256(&n.to_be_bytes());
            // int.from_bytes(digest, "big") cast to uint32 == low 32 bits == last 4 bytes.
            let s0 = u32::from_be_bytes(digest[28..32].try_into().unwrap());
            let s1 = r.seed as u32;
            r.device_seeds.insert(device.to_string(), (s0, s1));
            r.counters.insert(device.to_string(), (0u32, 0u32));
        }
        let key = r.device_seeds[device];
        let (low, high) = r.counters[device];
        let new_low = low.wrapping_add(num);
        let new_high = high.wrapping_add(0).wrapping_add((new_low < low) as u32);
        r.counters.insert(device.to_string(), (new_low, new_high));
        (key, (low, high))
    })
}

// ---------------------------------------------------------------------------
// Threefry (host evaluator, mirrors codegen::decomp::threefry2x32 exactly)
// ---------------------------------------------------------------------------

/// `threefry2x32(x, key)` (`codegen/decomp/op.py`): two uint32 lanes from a
/// uint64 `x`, 5x4 rounds of the two-cycle Threefry keyed by `key`, recombined.
fn threefry2x32(x: u64, key: u64) -> u64 {
    let mut xr0 = x as u32;
    let mut xr1 = (x >> 32) as u32;
    let key0 = key as u32;
    let key1 = (key >> 32) as u32;
    let kx = key0 ^ key1 ^ 0x1BD11BDA;
    let ks = [key1, kx, key0];
    xr0 = xr0.wrapping_add(ks[2]); // + key0
    xr1 = xr1.wrapping_add(ks[0]); // + key1
    let rot: [[u32; 4]; 2] = [[13, 15, 26, 6], [17, 29, 16, 24]];
    for i in 0..5 {
        for &r in rot[i % 2].iter() {
            let new0 = xr0.wrapping_add(xr1);
            let rotated = (xr1 << r).wrapping_add(xr1 >> (32 - r));
            let new1 = new0 ^ rotated;
            xr0 = new0;
            xr1 = new1;
        }
        xr0 = xr0.wrapping_add(ks[i % 3]);
        let k = ks[(i + 1) % 3].wrapping_add((i + 1) as u32);
        xr1 = xr1.wrapping_add(k);
    }
    ((xr1 as u64) << 32) | (xr0 as u64)
}

/// `RandMixin.random_bits` (rand.py:18): produce `num` uint32 bits from the
/// device `key` and (pre-increment) `counter`.
fn random_bits(key: (u32, u32), counter: (u32, u32), num: u32) -> Vec<u32> {
    // new_key = _threefry_random_bits(key, c_low, c_high) for the scalar counter.
    let x_scalar = ((counter.1 as u64) << 32) | (counter.0 as u64);
    let key_scalar = ((key.1 as u64) << 32) | (key.0 as u64);
    let out = threefry2x32(x_scalar, key_scalar);
    let nk0 = out as u32; // low
    let nk1 = (out >> 32) as u32; // high
    let key2 = ((nk1 as u64) << 32) | (nk0 as u64);

    let half = ((num + 1) / 2) as u32;
    let mut bits: Vec<u32> = Vec::with_capacity(2 * half as usize);
    let mut highs: Vec<u32> = Vec::with_capacity(half as usize);
    for j in 0..half {
        let counts1 = j.wrapping_add(half);
        let xj = ((counts1 as u64) << 32) | (j as u64);
        let o = threefry2x32(xj, key2);
        bits.push(o as u32);
        highs.push((o >> 32) as u32);
    }
    bits.extend(highs); // `.cat` of (low-bits vec, high-bits vec)
    bits.truncate(num as usize);
    bits
}

/// `RandMixin._bits_to_rand` (rand.py:31): assemble float values from uint32
/// bits — `(((bit >> (bitsize-nmant)) | float_one_bits) bitcast float) - 1`.
fn bits_to_rand_f32(bits: &[u32], numel: usize) -> Vec<f32> {
    let float_one: u32 = 1.0f32.to_bits();
    (0..numel)
        .map(|i| {
            let m = (bits[i] >> 9) | float_one;
            f32::from_bits(m) - 1.0
        })
        .collect()
}

fn bits_to_rand_f64(bits: &[u32], numel: usize) -> Vec<f64> {
    let float_one: u64 = 1.0f64.to_bits();
    (0..numel)
        .map(|i| {
            let lo = bits[2 * i] as u64;
            let hi = bits[2 * i + 1] as u64;
            let v = (hi << 32) | lo;
            let m = (v >> 12) | float_one;
            f64::from_bits(m) - 1.0
        })
        .collect()
}

fn numel_int(shape: &[SVal]) -> i64 {
    shape.iter().map(|s| s.as_i64().unwrap_or(1)).product()
}

/// `RandMixin.rand(*shape, dtype, device)` (rand.py:50).
pub fn rand(ctx: &mut UOpCtx, shape: &[SVal], dtype_opt: Option<DTypeId>, device: Option<&str>) -> Tensor {
    let dt = dtype_opt.unwrap_or_else(dtype::default_float);
    let dev = creation::canonicalize_device(device);
    let isz = dt.itemsize() as i64;
    let numel = numel_int(shape);
    let num = (numel * isz + 3) / 4;
    let (key, counter) = next_counter(&dev, num as u32);
    let bits = random_bits(key, counter, num as u32);
    let out = if dt == dtype::FLOAT64 {
        let vals = bits_to_rand_f64(&bits, numel as usize);
        Tensor::from_slice(ctx, &vals, Some(dt), Some(&dev))
    } else {
        let vals = bits_to_rand_f32(&bits, numel as usize);
        Tensor::from_slice(ctx, &vals, Some(dt), Some(&dev))
    };
    if shape.len() == 1 && shape[0].as_i64().unwrap_or(1) == numel {
        out
    } else {
        out.reshape(ctx, shape)
    }
}

/// `RandMixin.uniform(*shape, low, high, dtype)` (rand.py:158):
/// `((high-low) * rand()).cast(dtype) + low`.
pub fn uniform(
    ctx: &mut UOpCtx,
    shape: &[SVal],
    low: f64,
    high: f64,
    dtype_opt: Option<DTypeId>,
    device: Option<&str>,
) -> Tensor {
    let dt = dtype_opt.unwrap_or_else(dtype::default_float);
    let dev = creation::canonicalize_device(device);
    // `uniform` draws its bits in the *default float* dtype, scales, casts to
    // `dt`, then shifts by `low` (rand.py:158).
    let r = rand(ctx, shape, None, Some(&dev));
    let fdt = dtype::default_float();
    let range = Tensor::full(ctx, shape, ConstVal::Float(high - low), Some(fdt), Some(&dev));
    let scaled = r.mul(ctx, range).cast(ctx, dt);
    let lo = Tensor::full(ctx, shape, ConstVal::Float(low), Some(dt), Some(&dev));
    scaled.add(ctx, lo)
}

/// `RandMixin.randn(*shape, dtype)` (rand.py:106) via `randn_like` (rand.py:88):
/// a single `rand` over `[2, *shape]` in float32 feeds the polar Box–Muller
/// transform `cos(2*pi*u0) * sqrt(-2*ln(1-u1))`, computed on the host (the
/// graph route can't run under INTERP) and materialized with `from_slice`.
pub fn randn(ctx: &mut UOpCtx, shape: &[SVal], dtype_opt: Option<DTypeId>, device: Option<&str>) -> Tensor {
    let dt = dtype_opt.unwrap_or_else(dtype::default_float);
    let dev = creation::canonicalize_device(device);
    let mut s2 = shape.to_vec();
    s2.insert(0, SVal::I(2));
    let numel2 = numel_int(&s2);
    let (key, counter) = next_counter(&dev, numel2 as u32);
    let bits = random_bits(key, counter, numel2 as u32);
    let all = bits_to_rand_f32(&bits, numel2 as usize);
    let n = numel_int(shape) as usize;
    let (u0, u1) = all.split_at(n);
    // `2*pi` is a Python float rounded through the float32 tensor's dtype.
    let two_pi: f32 = (2.0f64 * std::f64::consts::PI) as f32;
    let vals: Vec<f32> = (0..n)
        .map(|i| {
            let a = u0[i] * two_pi;
            let c = a.cos();
            let b = 1.0f32 - u1[i];
            let l = b.ln();
            let s = (l * -2.0f32).sqrt();
            c * s
        })
        .collect();
    let out = if dt == dtype::FLOAT64 {
        let d: Vec<f64> = vals.iter().map(|&v| v as f64).collect();
        Tensor::from_slice(ctx, &d, Some(dt), Some(&dev))
    } else {
        Tensor::from_slice(ctx, &vals, Some(dt), Some(&dev))
    };
    if shape.len() == 1 && shape[0].as_i64().unwrap_or(1) == n as i64 {
        out
    } else {
        out.reshape(ctx, shape)
    }
}

/// `RandMixin.randint(*shape, low, high, dtype=int32)` (rand.py:122):
/// `uniform(low, high, dtype).cast(dtype)` — i.e. truncate
/// `(high-low)*rand` to the int dtype then add `low`. Bits are drawn in the
/// default float dtype; the value is computed on the host.
pub fn randint(
    ctx: &mut UOpCtx,
    shape: &[SVal],
    low: i32,
    high: i32,
    dtype_opt: Option<DTypeId>,
    device: Option<&str>,
) -> Tensor {
    assert!(low < high, "TGERR: Tensor.randint requires low < high, got low={low}, high={high}");
    let dt = dtype_opt.unwrap_or(dtype::INT32);
    let dev = creation::canonicalize_device(device);
    let numel = numel_int(shape);
    let (key, counter) = next_counter(&dev, numel as u32);
    let bits = random_bits(key, counter, numel as u32);
    let r = bits_to_rand_f32(&bits, numel as usize);
    let range = (high - low) as f32;
    let vals: Vec<i32> = r.iter().map(|&v| ((range * v) as i32) + low).collect();
    let out = Tensor::from_slice(ctx, &vals, Some(dt), Some(&dev));
    if shape.len() == 1 && shape[0].as_i64().unwrap_or(1) == numel {
        out
    } else {
        out.reshape(ctx, shape)
    }
}

// ---------------------------------------------------------------------------
// SHA-256 (needed only for `_next_counter`'s device seeding)
// ---------------------------------------------------------------------------

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn sha256(msg: &[u8]) -> [u8; 32] {
    let mut h: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let ml = (msg.len() as u64) * 8;
    let mut data = msg.to_vec();
    data.push(0x80);
    while data.len() % 64 != 56 {
        data.push(0);
    }
    data.extend_from_slice(&ml.to_be_bytes());
    for chunk in data.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(chunk[i * 4..i * 4 + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut abcdefgh = h;
        for i in 0..64 {
            let s1 = abcdefgh[4].rotate_right(6) ^ abcdefgh[4].rotate_right(11) ^ abcdefgh[4].rotate_right(25);
            let ch = (abcdefgh[4] & abcdefgh[5]) ^ ((!abcdefgh[4]) & abcdefgh[6]);
            let t1 = abcdefgh[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = abcdefgh[0].rotate_right(2) ^ abcdefgh[0].rotate_right(13) ^ abcdefgh[0].rotate_right(22);
            let maj = (abcdefgh[0] & abcdefgh[1]) ^ (abcdefgh[0] & abcdefgh[2]) ^ (abcdefgh[1] & abcdefgh[2]);
            let t2 = s0.wrapping_add(maj);
            abcdefgh[7] = abcdefgh[6];
            abcdefgh[6] = abcdefgh[5];
            abcdefgh[5] = abcdefgh[4];
            abcdefgh[4] = abcdefgh[3].wrapping_add(t1);
            abcdefgh[3] = abcdefgh[2];
            abcdefgh[2] = abcdefgh[1];
            abcdefgh[1] = abcdefgh[0];
            abcdefgh[0] = t1.wrapping_add(t2);
        }
        for i in 0..8 {
            h[i] = h[i].wrapping_add(abcdefgh[i]);
        }
    }
    let mut out = [0u8; 32];
    for i in 0..8 {
        out[i * 4..i * 4 + 4].copy_from_slice(&h[i].to_be_bytes());
    }
    out
}

#[cfg(test)]
mod sha_tests {
    use super::sha256;
    fn hex(d:&[u8;32])->String{d.iter().map(|b|format!("{b:02x}")).collect()}
    #[test] fn abc(){assert_eq!(hex(&sha256(b"abc")),"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");}
    #[test] fn empty(){assert_eq!(hex(&sha256(b"")),"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");}
}
