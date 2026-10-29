//! Typed data-marshalling tests: `Tensor::to_bytes` / `to_list_fresh::<T>` / `item_fresh::<T>`
//! and dtype inference on the input side (`from_slice` infers width from `T`).

use autorust::dtype;
use autorust::tensor::{Ctx, Tensor};

#[test]
fn from_slice_infers_dtype_from_generic() {
    let ctx = Ctx::new();
    let t16 = Tensor::from_slice(&ctx, &[1i16, 2, 3], None, None);
    let t32 = Tensor::from_slice(&ctx, &[1i32, 2, 3], None, None);
    let tu8 = Tensor::from_slice(&ctx, &[1u8, 2, 3], None, None);
    let tf32 = Tensor::from_slice(&ctx, &[1.0f32, 2.0, 3.0], None, None);
    let tf64 = Tensor::from_slice(&ctx, &[1.0f64, 2.0, 3.0], None, None);
    let tb = Tensor::from_slice(&ctx, &[true, false, true], None, None);
    assert_eq!(t16.dtype(), dtype::INT16);
    assert_eq!(t32.dtype(), dtype::INT32);
    assert_eq!(tu8.dtype(), dtype::UINT8);
    assert_eq!(tf32.dtype(), dtype::FLOAT32);
    assert_eq!(tf64.dtype(), dtype::FLOAT64);
    assert_eq!(tb.dtype(), dtype::BOOL);
}

#[test]
fn to_list_roundtrips_native_dtype() {
    let ctx = Ctx::new();
    let a = Tensor::from_slice(&ctx, &[-5i16, 300, -200, 0], None, None);
    assert_eq!(a.to_list_fresh::<i16>(), vec![-5i16, 300, -200, 0]);

    let b = Tensor::from_slice(&ctx, &[7u32, 1000, 65536], None, None);
    assert_eq!(b.to_list_fresh::<u32>(), vec![7u32, 1000, 65536]);

    let c = Tensor::from_slice(&ctx, &[1.5f64, -2.25, 3.0], None, None);
    assert_eq!(c.to_list_fresh::<f64>(), vec![1.5f64, -2.25, 3.0]);

    let d = Tensor::from_slice(&ctx, &[true, false, true], None, None);
    assert_eq!(d.to_list_fresh::<bool>(), vec![true, false, true]);
}

#[test]
fn to_bytes_is_native_width_storage() {
    let ctx = Ctx::new();
    let t = Tensor::from_slice(&ctx, &[1i16, 2, 3, 4], None, None);
    let bytes = t.to_bytes_fresh();
    assert_eq!(bytes.len(), 4 * dtype::INT16.itemsize());
    assert_eq!(bytes, vec![1u8, 0, 2, 0, 3, 0, 4, 0]); // little-endian i16

    let big = Tensor::from_slice(&ctx, &[1_000_000i64, -1], None, None);
    let bb = big.to_bytes_fresh();
    assert_eq!(bb.len(), 2 * dtype::INT64.itemsize());
    assert_eq!(
        i64::from_le_bytes(bb[..8].try_into().unwrap()),
        1_000_000i64
    );
}

#[test]
fn to_list_casts_across_dtypes() {
    let ctx = Ctx::new();
    // widen i16 -> i32
    let t16 = Tensor::from_slice(&ctx, &[-5i16, 300, -200], None, None);
    assert_eq!(t16.to_list_fresh::<i32>(), vec![-5i32, 300, -200]);

    // f32 -> i32 (integer-valued floats so truncation is unambiguous)
    let tf = Tensor::from_slice(&ctx, &[1.0f32, -2.0, 3.0], None, None);
    assert_eq!(tf.to_list_fresh::<i32>(), vec![1i32, -2, 3]);

    // bool -> u8 (true becomes 1)
    let tb = Tensor::from_slice(&ctx, &[true, false, true], None, None);
    assert_eq!(tb.to_list_fresh::<u8>(), vec![1u8, 0, 1]);
}

#[test]
fn item_reads_single_element() {
    let ctx = Ctx::new();
    let s = Tensor::from_slice(&ctx, &[42i64], None, None);
    assert_eq!(s.item_fresh::<i64>(), 42i64);
    let f = Tensor::from_slice(&ctx, &[3.5f32], None, None);
    assert_eq!(f.item_fresh::<f32>(), 3.5f32);
}

#[test]
#[should_panic(expected = "item() requires exactly one element")]
fn item_panics_on_multi_element() {
    let ctx = Ctx::new();
    let t = Tensor::from_slice(&ctx, &[1.0f32, 2.0, 3.0], None, None);
    let _ = t.item_fresh::<f32>();
}
