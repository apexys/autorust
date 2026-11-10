//! Fast non-cryptographic hashing for the hot interning/rewrite maps.
//!
//! `std`'s default `RandomState` is SipHash-1-3, which showed up as ~16% of a
//! cold ViT schedule (`hash_one::<&UopId>` / `<&u32>`). These maps are keyed by
//! process-unique node ids and never face untrusted input, so a small Fx-style
//! hasher is enough. The aliases keep the call sites readable.

use std::hash::{BuildHasherDefault, Hasher};

/// A hasher for interned node ids and small integer keys.
#[derive(Default)]
pub(crate) struct FxHasher {
    hash: u64,
}

const K: u64 = 0x517c_c1b7_2722_0a95;

impl FxHasher {
    #[inline]
    fn add(&mut self, i: u64) {
        self.hash = (self.hash.rotate_left(5) ^ i).wrapping_mul(K);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut b = bytes;
        while b.len() >= 8 {
            let (head, rest) = b.split_at(8);
            self.add(u64::from_ne_bytes(head.try_into().unwrap()));
            b = rest;
        }
        if !b.is_empty() {
            let mut v = [0u8; 8];
            v[..b.len()].copy_from_slice(b);
            self.add(u64::from_ne_bytes(v));
        }
    }
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(i as u64);
    }
    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.add(i as u64);
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(i as u64);
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

/// `HashMap` with [`FxHasher`].
pub(crate) type FastMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<FxHasher>>;
/// `HashSet` with [`FxHasher`].
pub(crate) type FastSet<K> = std::collections::HashSet<K, BuildHasherDefault<FxHasher>>;
