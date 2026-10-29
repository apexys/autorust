//! Minimal arbitrary-precision integer (sign + u64 limbs), std-only.
//!
//! Python ints are pervasive in tinygrad args (vmin/vmax, RANGE bounds, weak
//! dtype bounds like 2^800-1), so we need just enough bignum to represent,
//! compare, and (de)serialize them. Not a general bignum library.

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BigInt {
    neg: bool,
    /// little-endian u64 limbs, normalized (no trailing zeros; zero = empty)
    mag: Vec<u64>,
}

impl std::hash::Hash for BigInt {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.neg.hash(state);
        self.mag.hash(state);
    }
}

impl PartialOrd for BigInt {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BigInt {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self.is_negative(), other.is_negative()) {
            (false, true) => std::cmp::Ordering::Greater,
            (true, false) => std::cmp::Ordering::Less,
            (false, false) => Self::cmp_mag(&self.mag, &other.mag),
            (true, true) => Self::cmp_mag(&other.mag, &self.mag),
        }
    }
}

impl BigInt {
    pub fn zero() -> Self {
        BigInt {
            neg: false,
            mag: vec![],
        }
    }

    /// low 64 bits as an unsigned value (v mod 2^64 in two's complement)
    pub fn low64(&self) -> u64 {
        self.mag.first().cloned().unwrap_or(0)
    }

    pub fn limbs(&self) -> &[u64] {
        &self.mag
    }

    pub fn is_negative(&self) -> bool {
        self.neg && !self.mag.is_empty()
    }

    pub fn from_i128(v: i128) -> Self {
        let neg = v < 0;
        let mut m = v.unsigned_abs();
        let mut mag = Vec::new();
        while m != 0 {
            mag.push(m as u64);
            m >>= 64;
        }
        BigInt { neg, mag }
    }

    /// Raw constructor from (sign, little-endian limbs); normalizes zero.
    pub fn from_parts(neg: bool, mag: Vec<u64>) -> Self {
        let mut mag = mag;
        while mag.last() == Some(&0) {
            mag.pop();
        }
        BigInt {
            neg: neg && !mag.is_empty(),
            mag,
        }
    }

    pub fn pow2(exp: u64) -> Self {
        let mut mag = vec![0u64; (exp / 64) as usize + 1];
        mag[(exp / 64) as usize] = 1u64 << (exp % 64);
        BigInt { neg: false, mag }
    }

    /// 2^exp - 1
    pub fn pow2_minus_one(exp: u64) -> Self {
        if exp == 0 {
            return BigInt::zero();
        }
        let mut mag = vec![u64::MAX; (exp / 64) as usize];
        mag.push((1u64 << (exp % 64)) - 1);
        if mag.last() == Some(&0) {
            mag.pop();
        }
        BigInt { neg: false, mag }
    }

    pub fn is_zero(&self) -> bool {
        self.mag.is_empty()
    }

    pub fn negate(&self) -> Self {
        BigInt {
            neg: !self.neg && !self.is_zero(),
            mag: self.mag.clone(),
        }
    }

    fn cmp_mag(a: &[u64], b: &[u64]) -> std::cmp::Ordering {
        if a.len() != b.len() {
            return a.len().cmp(&b.len());
        }
        for i in (0..a.len()).rev() {
            if a[i] != b[i] {
                return a[i].cmp(&b[i]);
            }
        }
        std::cmp::Ordering::Equal
    }

    fn add_mag(a: &[u64], b: &[u64]) -> Vec<u64> {
        let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
        let mut out = Vec::with_capacity(long.len() + 1);
        let mut carry = 0u128;
        for i in 0..long.len() {
            let s = carry + long[i] as u128 + if i < short.len() { short[i] as u128 } else { 0 };
            out.push(s as u64);
            carry = s >> 64;
        }
        if carry != 0 {
            out.push(carry as u64);
        }
        out
    }

    /// a - b, requires a >= b
    fn sub_mag(a: &[u64], b: &[u64]) -> Vec<u64> {
        let mut out = Vec::with_capacity(a.len());
        let mut borrow = 0i128;
        for i in 0..a.len() {
            let mut d = a[i] as i128 - borrow - if i < b.len() { b[i] as i128 } else { 0 };
            if d < 0 {
                d += 1i128 << 64;
                borrow = 1;
            } else {
                borrow = 0;
            }
            out.push(d as u64);
        }
        while out.last() == Some(&0) {
            out.pop();
        }
        out
    }

    pub fn add(&self, other: &BigInt) -> BigInt {
        if self.neg == other.neg {
            BigInt {
                neg: self.neg,
                mag: Self::add_mag(&self.mag, &other.mag),
            }
        } else {
            match Self::cmp_mag(&self.mag, &other.mag) {
                std::cmp::Ordering::Equal => BigInt::zero(),
                std::cmp::Ordering::Greater => BigInt {
                    neg: self.neg,
                    mag: Self::sub_mag(&self.mag, &other.mag),
                },
                std::cmp::Ordering::Less => BigInt {
                    neg: other.neg,
                    mag: Self::sub_mag(&other.mag, &self.mag),
                },
            }
        }
    }

    pub fn sub(&self, other: &BigInt) -> BigInt {
        self.add(&other.negate())
    }

    pub fn cmp_i128(&self, other: i128) -> std::cmp::Ordering {
        self.cmp(&BigInt::from_i128(other))
    }

    /// Approximate f64 value (exact while the magnitude fits f64's 53-bit grid;
    /// approximate afterwards — only used where Python would also round via float).
    pub fn to_f64(&self) -> f64 {
        if self.is_zero() {
            return 0.0;
        }
        // take top 64 bits + scale
        let top = self.mag[self.mag.len() - 1];
        let shift = 64 * (self.mag.len() - 1) as i32;
        let v = top as f64;
        let scale = 2f64.powi(shift);
        if self.is_negative() {
            -v * scale
        } else {
            v * scale
        }
    }

    /// exact truncation of a finite float toward zero (Python `int(x)`).
    /// Returns None for inf/nan (Python raises).
    pub fn from_f64_trunc(x: f64) -> Option<BigInt> {
        let bits = x.to_bits();
        let neg = (bits >> 63) == 1;
        let exp = ((bits >> 52) & 0x7FF) as i64;
        let mant = bits & 0x000FFFFFFFFFFFFF;
        if exp == 0x7FF {
            return None; // inf / nan
        }
        let (mag, e): (Vec<u64>, i64) = if exp == 0 {
            if mant == 0 {
                return Some(BigInt::zero());
            }
            (vec![mant], -1074)
        } else {
            (vec![mant | (1u64 << 52)], exp - 1075)
        };
        let mut m = BigInt { neg: false, mag };
        if e >= 0 {
            m.shl(e as u64);
        } else {
            m.shr((-e) as u64);
        }
        m.neg = neg && !m.is_zero();
        Some(m)
    }

    fn shl(&mut self, k: u64) {
        if self.is_zero() {
            return;
        }
        let (limbs, bits) = (k / 64, k % 64);
        if bits > 0 {
            let mut carry = 0u64;
            for limb in self.mag.iter_mut() {
                let nc = *limb >> (64 - bits);
                *limb = (*limb << bits) | carry;
                carry = nc;
            }
            if carry != 0 {
                self.mag.push(carry);
            }
        }
        if limbs > 0 {
            let mut m = vec![0u64; limbs as usize];
            m.extend_from_slice(&self.mag);
            self.mag = m;
        }
    }

    fn shr(&mut self, k: u64) {
        let (limbs, bits) = (k / 64, k % 64);
        if limbs as usize >= self.mag.len() {
            self.mag.clear();
            self.neg = false;
            return;
        }
        if limbs > 0 {
            self.mag = self.mag[limbs as usize..].to_vec();
        }
        if bits > 0 {
            for i in 0..self.mag.len() {
                let hi = if i + 1 < self.mag.len() {
                    self.mag[i + 1] << (64 - bits)
                } else {
                    0
                };
                self.mag[i] = (self.mag[i] >> bits) | hi;
            }
        }
        self.normalize();
    }

    /// magnitude = magnitude * m + a  (m, a < 2^64) — for decimal (de)serialization
    fn mul_add_small(&mut self, m: u64, a: u64) {
        let mut carry = a as u128;
        for limb in self.mag.iter_mut() {
            let v = (*limb as u128) * m as u128 + carry;
            *limb = v as u64;
            carry = v >> 64;
        }
        while carry != 0 {
            self.mag.push(carry as u64);
            carry >>= 64;
        }
        self.normalize();
    }

    /// magnitude = magnitude / d, returns remainder (< d)
    fn divmod_small(&mut self, d: u64) -> u64 {
        let mut rem = 0u128;
        for limb in self.mag.iter_mut().rev() {
            let cur = (rem << 64) | *limb as u128;
            *limb = (cur / d as u128) as u64;
            rem = cur % d as u128;
        }
        self.normalize();
        rem as u64
    }

    fn normalize(&mut self) {
        while self.mag.last() == Some(&0) {
            self.mag.pop();
        }
        if self.mag.is_empty() {
            self.neg = false;
        }
    }

    /// decimal string (no leading zeros, '-' for negatives)
    pub fn to_decimal(&self) -> String {
        if self.is_zero() {
            return "0".into();
        }
        let mut mag = self.mag.clone();
        let mut chunks: Vec<String> = Vec::new();
        while !mag.is_empty() {
            let mut cur = BigInt {
                neg: false,
                mag: mag.clone(),
            };
            let rem = cur.divmod_small(10_000_000_000_000_000_000);
            mag = cur.mag;
            chunks.push(if mag.is_empty() {
                rem.to_string()
            } else {
                format!("{rem:019}")
            });
        }
        chunks.reverse();
        let mut s = String::new();
        if self.neg {
            s.push('-');
        }
        s.push_str(&chunks.concat());
        s
    }

    /// parse a decimal string (optional leading '-')
    pub fn from_decimal(s: &str) -> Self {
        let (neg, digits) = match s.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, s),
        };
        let mut out = BigInt::zero();
        for chunk in digits.as_bytes().chunks(19) {
            let v: u64 = std::str::from_utf8(chunk).unwrap().parse().unwrap();
            out.mul_add_small(10u64.pow(chunk.len() as u32), v);
        }
        out.neg = neg && !out.is_zero();
        out
    }

    /// i64 if representable (common fast path for arg ints)
    pub fn as_i64(&self) -> Option<i64> {
        if self.mag.len() > 2 {
            return None;
        }
        let m = match self.mag.len() {
            0 => 0u128,
            1 => self.mag[0] as u128,
            _ => ((self.mag[1] as u128) << 64) | self.mag[0] as u128,
        };
        if self.neg {
            if m <= (1u128 << 63) {
                Some((m as i128).wrapping_neg() as i64)
            } else {
                None
            }
        } else if m <= i64::MAX as u128 {
            Some(m as i64)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// decimal string of 2^exp via naive doubling (test reference)
    fn ref_pow2(exp: u32) -> String {
        let mut s = "1".to_string();
        for _ in 0..exp {
            let mut carry = 0u8;
            let mut out = Vec::new();
            for &c in s.as_bytes().iter().rev() {
                let d = (c - b'0') * 2 + carry;
                out.push((d % 10) + b'0');
                carry = d / 10;
            }
            while carry > 0 {
                out.push((carry % 10) + b'0');
                carry /= 10;
            }
            out.reverse();
            s = String::from_utf8(out).unwrap();
        }
        s
    }

    #[test]
    fn basics() {
        assert_eq!(
            BigInt::from_decimal("-12345678901234567890123456789").to_decimal(),
            "-12345678901234567890123456789"
        );
        assert_eq!(BigInt::pow2(800).to_decimal(), ref_pow2(800));
        assert_eq!(
            BigInt::pow2_minus_one(64).to_decimal(),
            "18446744073709551615"
        );
        assert_eq!(BigInt::pow2_minus_one(800).to_decimal(), {
            let p = ref_pow2(800);
            // subtract one from decimal string
            let mut b = p.into_bytes();
            let mut i = b.len() - 1;
            loop {
                if b[i] > b'0' {
                    b[i] -= 1;
                    break;
                }
                b[i] = b'9';
                i -= 1;
            }
            String::from_utf8(b).unwrap()
        });
        assert!(BigInt::from_i128(-5).cmp(&BigInt::from_i128(3)) == std::cmp::Ordering::Less);
        assert_eq!(
            BigInt::from_i128(7)
                .add(&BigInt::from_i128(-10))
                .to_decimal(),
            "-3"
        );
        assert_eq!(
            BigInt::pow2(64).sub(&BigInt::pow2(63)).to_decimal(),
            "9223372036854775808"
        );
        assert_eq!(BigInt::pow2(63).negate().as_i64(), Some(i64::MIN));
        assert_eq!(BigInt::pow2(63).as_i64(), None);
        assert_eq!(BigInt::from_i128(-9).as_i64(), Some(-9));
    }
}
