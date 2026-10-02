//! M3 creation mixin (mixin/creation.py + the `Tensor.*` factories on UOp-level
//! sugar): full/ones/zeros/eye/arange/linspace/_tri/triu/tril and `new_buffer`/
//! `empty` (the BUFFER-slot spine shared with `Tensor.empty`).

use crate::dtype::{self, DTypeId};
use crate::dtype::ConstVal;
use crate::uop::arg::{Arg, ParamArg};
use crate::tensor::Shape;
use crate::uop::ctx::UOpCtx;
use crate::uop::graph::UopId;
use crate::uop::ops::Ops;
use crate::uop::sint::{SInt, SVal};

use super::movement;
use super::Tensor;

/// Register the initial host bytes for a buffer-backed tensor (python
/// `Buffer(opaque=memoryview(bytearray(initial_value)))` at Tensor(data) time):
/// walk to the base BUFFER uop and record the bytes for the engine.
pub(crate) fn register_initial(ctx: &mut UOpCtx, mut u: UopId, bytes: Vec<u8>) {
    while ctx.op_of(&u) != Ops::Buffer {
        u = ctx.srcs(&u)[0].clone();
    }
    u.node.initial_data.replace(Some(std::rc::Rc::new(std::cell::RefCell::new(bytes))));
}

/// `UOp.unique_num` — the global BUFFER slot counter (ops.py:814).
pub static UNIQUE_NUM: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);

/// `UOp.new_buffer(device, size, dtype, num=None)` (ops.py:815-820).
/// `device` here is the Python-side canonical string (e.g. "PYTHON" — tinygrad's DEFAULT staging name —, "CL").
pub fn new_buffer(ctx: &mut UOpCtx, device: &str, size: SVal, dt: DTypeId, num: Option<i64>) -> UopId {
    assert!(!dtype::is_weak(dt), "TGERR: cannot create storage for weak dtype {dt:?}");
    let slot = num.unwrap_or_else(|| UNIQUE_NUM.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
    let arg = ParamArg {
        slot,
        dtype: dt,
        vmin_vmax: None,
        multiple_of: None,
        name: None,
        addrspace: Some(crate::dtype::AddrSpace::Global), // ParamArg default
        axis: None,
        device: Some(crate::uop::arg::DeviceSpec::Single(device.to_string())),
        volatile: false,
    };
    let src = ctx.shape_arg(&[size]);
    ctx.uop(Ops::Buffer, None, vec![src], Arg::Param(arg))
}

/// `UOp.empty(shape, dtype, device)` (ops.py:846-850) — `new_buffer(...).reshape(max_shape).shrink_to(shape)`.
pub fn empty(ctx: &mut UOpCtx, shape: &[SVal], dt: DTypeId, device: &str) -> UopId {
    let max_shape = to_max_shape(shape);
    let numel = super::movement::prod_sval(ctx, &max_shape);
    let buf = new_buffer(ctx, device, numel, dt, None);
    let reshaped = movement::reshape(ctx, buf, &max_shape);
    movement::shrink_to(ctx, reshaped, shape)
}

/// `to_max_shape` (ops.py:1754): symbolic dims replaced by their (int) vmax.
pub fn to_max_shape(shape: &[SVal]) -> Vec<SVal> {
    shape.to_vec() // resolved-to-int shapes only in M3 corpus; symbolic vmax path lands with the callify oracle
}

/// `CreationMixin.full(shape, fill_value, dtype=None, device=None, buffer=True)`
/// (creation.py:61-85). `fill_value` arrives as a `ConstVal`; `buffer=false`
/// returns the broadcast const (no store).
pub fn full(ctx: &mut UOpCtx, shape: &[SVal], fill_value: ConstVal, dtype: Option<DTypeId>, device: Option<&str>, buffer: bool) -> Tensor {
    let new_shape = shape.to_vec();
    let dt = dtype.unwrap_or(match &fill_value {
        ConstVal::Bool(_) => dtype::BOOL,
        ConstVal::Int(_) | ConstVal::Big(_) => dtype::WEAKINT,
        ConstVal::Float(_) => dtype::WEAKFLOAT,
        ConstVal::Invalid => dtype::WEAKINT,
    });
    let val = Tensor::const_(ctx, fill_value, Some(dt));
    let ones_shape: Vec<SVal> = vec![SVal::I(1); new_shape.len()];
    let val = { let __u = movement::reshape(ctx, val.uop(), &ones_shape); Tensor::from_uop(ctx, __u) };
    let val = val.expand(ctx, new_shape.clone());
    if !buffer {
        return val;
    }
    // `val.empty_like(None, device)`: device = canonicalize(device or val.device
    // (None)) -> the default device; dtype = strong_dtype(val.dtype)
    let dev = canonicalize_device(device);
    let ret = empty(ctx, &new_shape, dtype::strong_dtype(dt), &dev);
    let store = super::store(ctx, ret.clone(), val.uop(), None);
    { let __u = super::after(ctx, ret, vec![store]); Tensor::from_uop(ctx, __u) }
}

/// `CreationMixin.zeros(*shape, **kwargs)` (creation.py:106) — `full(shape, 0.0, dtype)`.
pub fn zeros(ctx: &mut UOpCtx, shape: &[SVal], dtype: Option<DTypeId>, device: Option<&str>) -> Tensor {
    // Python passes dtype straight through; fill_value 0.0 -> from_py -> weakfloat
    full(ctx, shape, ConstVal::Float(0.0), dtype, device, true)
}

/// `CreationMixin.ones(*shape, **kwargs)` (creation.py:136) — `full(shape, 1.0, dtype)`.
pub fn ones(ctx: &mut UOpCtx, shape: &[SVal], dtype: Option<DTypeId>, device: Option<&str>) -> Tensor {
    full(ctx, shape, ConstVal::Float(1.0), dtype, device, true)
}

/// `RandMixin.rand(*shape, **kwargs)` (rand.py:50): uniform [0, 1) via the
/// Threefry graph RNG (`tensor::rand`).
pub fn rand(ctx: &mut UOpCtx, shape: &[SVal], dtype: Option<DTypeId>, device: Option<&str>) -> Tensor {
    crate::tensor::rand::rand(ctx, shape, dtype, device)
}

/// `RandMixin.uniform(*shape, low=0.0, high=1.0, **kwargs)` (rand.py:158):
/// `(high-low) * rand() + low`.
pub fn uniform(ctx: &mut UOpCtx, shape: &[SVal], low: f64, high: f64, dtype: Option<DTypeId>, device: Option<&str>) -> Tensor {
    crate::tensor::rand::uniform(ctx, shape, low, high, dtype, device)
}

/// `OpMixin.arange(start, stop=None, step=1, dtype=None)` (op.py:165-197).
/// All-int args only in the M3 corpus (symbolic stops come with the Variable
/// smoke test); `dtype=None` picks default_float iff any arg is a float.
pub fn arange(ctx: &mut UOpCtx, start: f64, stop: Option<f64>, step: f64, dtype: Option<DTypeId>) -> Tensor {
    let (start, stop) = match stop {
        Some(s) => (start, s),
        None => (0.0, start),
    };
    let any_float = start.fract() != 0.0 || stop.fract() != 0.0 || step.fract() != 0.0;
    let (lo, hi) = if step > 0.0 { (start, stop - step) } else { (stop - step, start) };
    let dt = match dtype {
        Some(d) => d,
        None => {
            if any_float {
                dtype::default_float()
            } else {
                // an int range too large for default_int picks int64
                let (loi, hii) = (lo as i128, hi as i128);
                let d = dtype::default_int();
                let (min, max) = (dtype::dtype_min(d).as_f64(), dtype::dtype_max(d).as_f64());
                if (loi as f64) < min || max < hii as f64 {
                    crate::dtype::by_attr("int64").expect("int64 dtype")
                } else {
                    d
                }
            }
        }
    };
    // overflow check against the chosen dtype (ints only; float ranges always fit)
    if !dtype::is_float(dt) {
        let (min, max) = (dtype::dtype_min(dt).as_f64(), dtype::dtype_max(dt).as_f64());
        if lo < min || hi > max {
            panic!("TGERR: OverflowError arange [{start}, {stop}) is not representable in dtype {dt:?}");
        }
    }
    // python ceildiv(num, amt) = -(num // -amt) with python floor semantics
    // (works for float steps too)
    let output_len = (-(stop - start) / -step).floor() as i128;
    if output_len <= 0 {
        return full(ctx, &[SVal::I(0)], ConstVal::Int(0), Some(dt), None, false);
    }
    // full((output_len,), step, buffer=False)._cumalu(0, ADD) + (start - step)
    let t = full(ctx, &[SVal::I(output_len)], const_f64(step, dt), Some(dt), None, false);
    let cum = super::reduce::cumalu(ctx, t.uop(), 0, Ops::Add);
    let start_step = const_f64(start - step, dt);
    let added = super::elementwise::binop_c(ctx, Ops::Add, cum, start_step);
    { let __u = ctx.cast(added, dt); Tensor::from_uop(ctx, __u) }
}

/// `OpMixin.eye(n, m=None, dtype=None)` (op.py:215-232):
/// `arange(n).unsqueeze(-1).eq(arange(m)).cast(dtype)`.
pub fn eye(ctx: &mut UOpCtx, n: i64, m: Option<i64>, dtype: Option<DTypeId>) -> Tensor {
    let m_ = m.unwrap_or(n);
    assert!(n >= 0 && m_ >= 0, "TGERR: cannot have negative n, m");
    let out_dtype = dtype.unwrap_or_else(dtype::default_float);
    let a = arange(ctx, n as f64, None, 1.0, None).unsqueeze(ctx, -1);
    let b = arange(ctx, m_ as f64, None, 1.0, None);
    let eq = super::elementwise::eq(ctx, a.uop(), b.uop());
    { let __u = ctx.cast(eq, out_dtype); Tensor::from_uop(ctx, __u) }
}

/// `OpMixin._tri(r, c, diagonal=0)` (op.py:233-235):
/// `arange(r).unsqueeze(-1) + diagonal <= arange(c)`.
pub fn _tri(ctx: &mut UOpCtx, r: SVal, c: SVal, diagonal: i64) -> Tensor {
    let ri = r.as_i128_or_panic() as f64;
    let ci = c.as_i128_or_panic() as f64;
    let a = arange(ctx, ri, None, 1.0, None).unsqueeze(ctx, -1);
    let b = arange(ctx, ci, None, 1.0, None);
    let diag_val = ConstVal::Int(diagonal as i128);
    let added = super::elementwise::binop_c(ctx, Ops::Add, a.uop(), diag_val);
    let le = super::elementwise::le(ctx, added, b.uop()); // a+diag <= arange(c)
    { let __u = le; Tensor::from_uop(ctx, __u) }
}

/// `OpMixin.linspace(start, stop, steps, dtype=None)` (op.py:198-214).
pub fn linspace(ctx: &mut UOpCtx, start: f64, stop: f64, steps: i64, dtype: Option<DTypeId>) -> Tensor {
    assert!(steps >= 0, "TGERR: number of steps must be non-negative");
    let dt = dtype.unwrap_or_else(dtype::default_float);
    assert!(!dtype::is_bool(dt), "TGERR: linspace with bool dtype is not supported");
    if steps == 1 {
        return full(ctx, &[SVal::I(1)], const_f64(start, dt), Some(dt), None, false);
    }
    // (start + arange(steps, float) * ((stop - start) / (steps - 1))).cast(dtype)
    let idx = arange(ctx, steps as f64, None, 1.0, Some(dtype::default_float()));
    let scale = (stop - start) / (steps - 1) as f64;
    let scaled = super::elementwise::binop_c(ctx, Ops::Mul, idx.uop(), const_f64(scale, dtype::default_float()));
    let biased = super::elementwise::binop_c(ctx, Ops::Add, scaled, const_f64(start, dtype::default_float()));
    { let __u = ctx.cast(biased, dt); Tensor::from_uop(ctx, __u) }
}

/// `OpMixin.triu(diagonal=0)` (op.py:236-258):
/// `_tri(shape[-2], shape[-1], diagonal).where(self, 0)`.
pub fn triu(ctx: &mut UOpCtx, x: UopId, diagonal: i64) -> Tensor {
    let shape = ctx.shape(&x).unwrap().unwrap_or_default();
    let (r, c) = (<SInt as Clone>::clone(&shape[shape.len() - 2]), <SInt as Clone>::clone(&shape[shape.len() - 1]));
    let tri = _tri(ctx, r, c, diagonal);
    let zero = ctx.const_like_dt(x.clone(), ConstVal::Int(0), None);
    { let __u = ctx.where_(tri.uop(), x, zero); Tensor::from_uop(ctx, __u) }
}

/// `OpMixin.tril(diagonal=0)` (op.py:259-281):
/// `_tri(shape[-2], shape[-1], diagonal+1).where(0, self)`.
pub fn tril(ctx: &mut UOpCtx, x: UopId, diagonal: i64) -> Tensor {
    let shape = ctx.shape(&x).unwrap().unwrap_or_default();
    let (r, c) = (<SInt as Clone>::clone(&shape[shape.len() - 2]), <SInt as Clone>::clone(&shape[shape.len() - 1]));
    let tri = _tri(ctx, r, c, diagonal + 1);
    let zero = ctx.const_like_dt(x.clone(), ConstVal::Int(0), None);
    { let __u = ctx.where_(tri.uop(), zero, x); Tensor::from_uop(ctx, __u) }
}

/// ConstVal of the right kind for `dt` from an f64 step/start value
/// (float dtypes -> Float; int dtypes -> Int, only for integral inputs).
pub fn const_f64(v: f64, dt: DTypeId) -> ConstVal {
    if dtype::is_float(dt) {
        ConstVal::Float(v)
    } else {
        assert_eq!(v.fract(), 0.0, "TGERR: non-integral {v} for int dtype");
        ConstVal::Int(v as i128)
    }
}

/// `Shape` helper for creation fns taking `ShapeArg` at the `Tensor` level.
pub fn shape_from(args: impl super::ShapeArg) -> Shape {
    Shape::new(args.into_shape_vec())
}

// **** constructors from host data (`Tensor.__init__` list/bytes paths -> `UOp._frompy`) ****

/// Python `Device.DEFAULT` (device.py): env-configurable, else the best backend
/// actually available. There is no "NV" backend ported, so the fallback is the
/// interpreter (always present) — or the OpenCL backend when it is compiled in
/// (`backend_cl`) and a device is present.
pub static DEFAULT_DEVICE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

pub fn default_device() -> String {
    DEFAULT_DEVICE
        .get_or_init(|| {
            if let Ok(dev) = std::env::var("DEFAULT_DEVICE") {
                return dev;
            }
            #[cfg(feature = "backend_cl")]
            if crate::runtime::ops_cl::available() {
                return "CL".to_string();
            }
            "INTERP".to_string()
        })
        .clone()
}

/// `canonicalize_device` (device.py:59) for the single-device case.
pub fn canonicalize_device(device: Option<&str>) -> String {
    device.unwrap_or(default_device().as_str()).to_string()
}

/// Host element types usable with `Tensor::from_slice`. The DEFAULT dtype
/// mirrors Python's `dtypes.from_py(list)` inference: every int width ->
/// `default_int`, every float width -> `default_float`, bool -> `bool`
/// (python lists carry no width information). Pass `Some(dtype)` for exact
/// control — the values then pack at that width on the Python side.
pub trait SliceElem: Sized {
    fn default_dtype() -> DTypeId;
    fn to_f64(&self) -> f64;
    /// Little-endian bytes at `default_dtype()` width (no `f64` round-trip).
    fn encode_native(&self) -> Vec<u8>;
}

macro_rules! impl_slice_elem {
    ($($t:ty => $dt:expr);* $(;)?) => {$(
        impl SliceElem for $t {
            fn default_dtype() -> DTypeId {
                $dt
            }
            fn to_f64(&self) -> f64 {
                *self as f64
            }
            fn encode_native(&self) -> Vec<u8> {
                self.to_le_bytes().to_vec()
            }
        }
    )*};
}
impl SliceElem for bool {
    fn default_dtype() -> DTypeId {
        dtype::BOOL
    }
    fn to_f64(&self) -> f64 {
        if *self { 1.0 } else { 0.0 }
    }
    fn encode_native(&self) -> Vec<u8> {
        vec![*self as u8]
    }
}
impl_slice_elem!(
    i8 => dtype::INT8; i16 => dtype::INT16; i32 => dtype::INT32; i64 => dtype::INT64; isize => dtype::INT64;
    u8 => dtype::UINT8; u16 => dtype::UINT16; u32 => dtype::UINT32; u64 => dtype::UINT64; usize => dtype::UINT64;
    f32 => dtype::FLOAT32; f64 => dtype::FLOAT64
);

/// `copy_to_device` (ops.py:745-750): `COPY(pad_to(max_shape), arg=device)
/// .shrink_to(shape)` — both noops for static shapes.
pub fn copy_to_device(ctx: &mut UOpCtx, x: UopId, device: &str) -> UopId {
    let shape = ctx.shape(&x).unwrap().unwrap_or_default();
    assert!(!dtype::is_weak(ctx.dt_of(&x)), "TGERR: cannot create storage for weak dtype (node {} op {:?})", x.id(), ctx.op_of(&x));
    let padded = movement::pad_to(ctx, x, &shape);
    let copy = ctx.uop(Ops::Copy, None, vec![padded], Arg::Str(device.to_string()));
    movement::shrink_to(ctx, copy, &shape)
}

/// `Tensor(data=list, dtype=?, device=?)` -> `UOp._frompy` (ops.py:829-842):
/// `UOp.empty(shape, bdtype, "INTERP")` (float32 storage for bf16/fp8 targets,
/// then a cast), then `COPY` to the canonical device when it is not "PYTHON".
/// The byte packing itself is a Python-side buffer effect and does not appear
/// in the graph.
pub fn from_slice<T: SliceElem>(ctx: &mut UOpCtx, data: &[T], dtype: Option<DTypeId>, device: Option<&str>) -> Tensor {
    let target = dtype.unwrap_or_else(T::default_dtype);
    assert!(!dtype::is_weak(target), "TGERR: cannot create storage for weak dtype {target:?}");
    assert!(!data.is_empty(), "TGERR: cannot create Tensor from an empty list");
    let shape = [SVal::I(data.len() as i128)];
    // bfloat16 and fp8 have no struct format: pack a float32 buffer and cast
    let bdtype = if dtype::in_set(&[dtype::BFLOAT16, dtype::FP8E4M3, dtype::FP8E5M2, dtype::FP8E4M3FNUZ, dtype::FP8E5M2FNUZ], target) {
            dtype::FLOAT32
        } else {
            target
        };
    let ret = empty(ctx, &shape, bdtype, "PYTHON");
    // pack the host data into the PYTHON base storage (Buffer(opaque=...)).
    // When the requested width matches the element's own dtype we encode the
    // native little-endian bytes directly (no f64); an explicit `dtype` override
    // that changes width goes through the f64 cast bridge.
    let native = T::default_dtype();
    let packed: Vec<u8> = if bdtype == native {
        data.iter().flat_map(|v| v.encode_native()).collect()
    } else {
        data.iter().flat_map(|v| crate::runtime::interp::to_storage_scalar(v.to_f64(), bdtype)).collect()
    };
    register_initial(ctx, ret.clone(), packed);
    let ret = if bdtype != target { ctx.cast(ret, target) } else { ret };
    let dev = canonicalize_device(device);
    let ret = if dev == "PYTHON" { ret } else { copy_to_device(ctx, ret, &dev) };
    { let __u = ret; Tensor::from_uop(ctx, __u) }
}

/// `Tensor(data=bytes, dtype=?, device=?)` -> the bytes arm of `_frompy`:
/// a 1-D `BUFFER("PYTHON", len(data)/itemsize)` — no reshape, no cast.
pub fn from_bytes(ctx: &mut UOpCtx, data: &[u8], dtype: DTypeId, device: Option<&str>) -> Tensor {
    assert!(!dtype::is_weak(dtype), "TGERR: cannot create storage for weak dtype {dtype:?}");
    assert!(!data.is_empty(), "TGERR: cannot create Tensor from empty bytes");
    let size = SVal::I((data.len() / dtype.itemsize()) as i128);
    let _ = &size;
    let ret = new_buffer(ctx, "PYTHON", size, dtype, None);
    register_initial(ctx, ret.clone(), data.to_vec());
    let dev = canonicalize_device(device);
    let ret = if dev == "PYTHON" { ret } else { copy_to_device(ctx, ret, &dev) };
    { let __u = ret; Tensor::from_uop(ctx, __u) }
}
