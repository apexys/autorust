"""Generate tests/golden.rs — golden numerics for the backend harness.

Programs are realized once on tinygrad's default device (the pure-python
interp) and emitted as full-precision f64 consts with a per-program rtol
(0 = exact, for arange-derived and algebraic values; ~1e-5 for the
transcendental family whose OpenCL/interp implementations differ by ULPs).
The rust harness builds the identical programs and asserts every configured
backend reproduces the goldens.

Run from autorust/: ../tinygrad/.venv/bin/python py/dump_golden.py
"""
import sys
sys.path.insert(0, "py"); sys.path.insert(0, "../tinygrad")

import math
from tinygrad import Tensor
from tinygrad.dtype import dtypes, from_storage_scalar

def buffer_values(t):
    """Read a realized tensor like the rust harness does: raw storage bytes
    decoded per dtype. tolist() would const-fold sub-f32 casts (fp8/bf16/f16
    roundtrips would skip quantization, diverging from buffer semantics)."""
    import struct as _struct
    if t.dtype in (dtypes.half, dtypes.bfloat16, *dtypes.fp8s):
        raw = bytes(t._buffer().as_memoryview())
        if t.dtype == dtypes.half:
            isz = 2
            return [_struct.unpack("<e", raw[i * isz:(i + 1) * isz])[0] for i in range(len(raw) // isz)]
        if t.dtype == dtypes.bfloat16:
            isz = 2
            # bf16 = the top half of the f32 bit pattern: f32 LE bytes are
            # [0, 0, bf16_lo, bf16_hi]
            return [_struct.unpack("<f", b"\x00\x00" + raw[i * isz:(i + 1) * isz])[0] for i in range(len(raw) // isz)]
        isz = 1
        return [from_storage_scalar(raw[i], t.dtype) for i in range(len(raw))]
    flat = t.tolist()

    def flat_list(v):
        if isinstance(v, list):
            out = []
            for sub in v:
                out.extend(flat_list(sub))
            return out
        return [float(v)]

    return flat_list(flat)

FX = [-2.0, -1.0, 0.5, 1.0, 2.0, 3.5, 4.0, 10.0]
FX2 = [3.0, -2.0, 2.5, -1.0, 1.0, 0.5, -2.0, 4.0]
PX = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
IX = [3, -7, 0, 12, 5, -2, 8, 1]
IY = [2, 1, 3, 3, 1, 2, 5, 1]  # nonzero: no div/mod by zero
SH = [0, 1, 2, 3, 0, 1, 2, 3]

def tf(vals): return Tensor(vals)

# name -> (builder() -> Tensor, rtol)
# Builder order matters only for readability; each program is realized alone.
def make_programs():
    p = {}
    add = lambda n, t, r=0.0: p.__setitem__(n, (t, r))

    # -- smoke: upload/compute/download on any backend --
    add("smoke_add", lambda: tf([1, 2, 3, 4, 5, 6, 7, 8]) + 2.0)
    add("smoke_two_tensors", lambda: tf([1, 2, 3, 4]) + tf([10, 20, 30, 40]))
    add("smoke_mulc_neg", lambda: tf([0.5, 1.5, -2.0]) * 3)

    # -- ALU ops (one program per code_for_op entry) --
    add("op_neg", lambda: -tf(FX))
    add("op_exp2", lambda: tf(PX).exp2(), 1e-5)
    add("op_log2", lambda: tf(PX).log2(), 1e-5)
    add("op_sqrt", lambda: tf(PX).sqrt(), 1e-5)
    add("op_sin", lambda: tf(FX).sin(), 1e-5)
    add("op_trunc", lambda: tf(FX).trunc())
    add("op_reciprocal", lambda: tf(PX).reciprocal(), 1e-5)
    add("op_pow", lambda: tf(PX).pow(1.5), 1e-5)
    add("op_add", lambda: tf(FX) + tf(FX2))
    add("op_sub", lambda: tf(FX) - tf(FX2))
    add("op_mul", lambda: tf(FX) * tf(FX2))
    add("op_fdiv", lambda: tf(FX) / tf(FX2), 1e-6)
    add("op_cmplt", lambda: tf(FX) < tf(FX2))
    add("op_cmpeq", lambda: tf(FX) == tf(FX2))
    add("op_cmpne", lambda: tf(FX) != tf(FX2))
    add("op_max", lambda: tf(FX).maximum(tf(FX2)))
    add("op_where", lambda: (tf(FX) > 0).where(tf(FX2), tf(PX)))
    ix, iy, sh = tf(IX).cast(dtypes.int32), tf(IY).cast(dtypes.int32), tf(SH).cast(dtypes.int32)
    add("op_shl", lambda: ix << sh)
    add("op_shr", lambda: ix >> sh)
    add("op_and", lambda: ix & iy)
    add("op_or", lambda: ix | iy)
    add("op_xor", lambda: ix ^ iy)
    add("op_floordiv", lambda: ix // iy)
    add("op_floormod", lambda: ix % iy)
    add("op_mulacc", lambda: (tf(FX) * tf(FX2)).sum())
    # CDIV/CMOD are reached inside floordiv/floormod decomp (no tensor sugar).

    # -- reductions over aranges (exact integer-valued f32) --
    add("arange4", lambda: Tensor.arange(4))
    add("arange_step", lambda: Tensor.arange(3, 7, 2))
    add("red_sum_arange", lambda: Tensor.arange(100).sum())
    add("red_mean_arange", lambda: Tensor.arange(100).mean())
    add("red_max_arange", lambda: Tensor.arange(100).max())
    add("red_sum_axis", lambda: Tensor.arange(12).reshape(3, 4).sum(axis=1))
    add("red_max_axis_keepdim", lambda: Tensor.arange(12).reshape(3, 4).max(axis=0, keepdim=True))
    add("red_sum_2d_all", lambda: Tensor.arange(6).reshape(2, 3).sum())

    # -- movement / matmul battery (integer-valued, exact) --
    add("mm_2x3_3x2", lambda: tf([[1, 2, 3], [4, 5, 6]]).matmul(tf([[7, 8], [9, 10], [11, 12]])))
    add("mm_3x3", lambda: tf([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).matmul(tf([[1, 0], [0, 1], [1, 0]])))
    add("permute_2x4", lambda: tf([[1, 2, 3, 4], [5, 6, 7, 8]]).permute(1, 0))
    add("expand_1x3_2x3", lambda: tf([[1, 2, 3]]).expand(2, 3))
    add("cumsum_axis1", lambda: tf([[1, 2, 3], [4, 5, 6]]).cumsum(1))
    add("cumsum_axis0", lambda: tf([[1, 2, 3], [4, 5, 6]]).cumsum(0))

    # -- casts (ctypes truncation semantics) --
    add("cast_f2i", lambda: tf([-2.0, -1.0, 1.7, 2.5]).cast(dtypes.int32))
    add("cast_i2f", lambda: tf([3, -7, 0, 12]).cast(dtypes.int32).cast(dtypes.float32))
    add("cast_f2u8", lambda: tf([-2.0, 3.9, 300.0]).cast(dtypes.uint8))
    add("cast_f2i8", lambda: tf([-300.0, 3.9, 127.0]).cast(dtypes.int8))
    add("cast_b2i32", lambda: tf([1.0, 0.0, 2.0, 0.5]).cast(dtypes.bool).cast(dtypes.int32))
    add("cast_f2b", lambda: tf([0.0, 1.5, -3.0]).cast(dtypes.bool))
    add("cast_f32_f64", lambda: tf(FX).cast(dtypes.float64))
    add("cast_f64_f32", lambda: tf([1e10, -2.5, 3.14159265358979]).cast(dtypes.float64).cast(dtypes.float32))

    # -- dtype battery: staging cast -> roundtrip / ew / sum per dtype --
    float_vals = [0.5, -1.25, 3.75, 100.0]
    signed_vals = [-5, -1, 0, 7, 100]
    unsigned_vals = [5, 1, 0, 7, 100]
    for name, vals in [("float16", float_vals), ("bfloat16", float_vals), ("float64", float_vals),
                       ("fp8e4m3", float_vals), ("fp8e5m2", float_vals),
                       ("fp8e4m3fnuz", float_vals), ("fp8e5m2fnuz", float_vals),
                       ("int8", signed_vals), ("int16", signed_vals), ("int32", signed_vals), ("int64", signed_vals),
                       ("uint8", unsigned_vals), ("uint16", unsigned_vals), ("uint32", unsigned_vals), ("uint64", unsigned_vals)]:
        add(f"dt_{name}_roundtrip", (lambda v, d: lambda: tf(v).cast(d))(vals, getattr(dtypes, name)))
        add(f"dt_{name}_ew", (lambda d: lambda: (tf(float_vals).cast(d) * 2) + 1)(getattr(dtypes, name)))
        add(f"dt_{name}_sum", (lambda d: lambda: tf(float_vals).cast(d).sum())(getattr(dtypes, name)), 1e-6 if name.startswith("fp8") else 0.0)
    add("dt_bool_roundtrip", lambda: tf([0.0, 1.5, -3.0, 0.5]).cast(dtypes.bool))
    add("dt_bool_sum", lambda: tf([0.0, 1.5, -3.0, 0.5]).cast(dtypes.bool).sum())
    return p


def emit():
    lines = []
    lines.append("// GENERATED by py/dump_golden.py — golden numerics for the backend harness.")
    lines.append("// Programs are mirrored in tests/backend_harness.rs; regenerate with")
    lines.append("// `../tinygrad/.venv/bin/python py/dump_golden.py` after editing either side.")
    lines.append("")
    lines.append("pub struct Golden {")
    lines.append("    pub name: &'static str,")
    lines.append("    pub shape: &'static [usize],")
    lines.append("    pub vals: &'static [f64],")
    lines.append("    /// relative tolerance for |got - want| <= rtol * max(|want|, 1.0)")
    lines.append("    pub rtol: f64,")
    lines.append("}")
    lines.append("")
    entries = []
    progs = make_programs()
    for name, (builder, rtol) in progs.items():
        t = builder()
        vals = buffer_values(t)
        shape = [int(s) for s in t.shape]
        for v in vals:
            assert math.isfinite(v), f"{name}: non-finite golden {v}"
        vals_s = ", ".join(f"{v!r}" for v in vals)
        shape_s = ", ".join(str(s) for s in shape)
        entries.append(
            f'    Golden {{ name: "{name}", shape: &[{shape_s}], vals: &[{vals_s}], rtol: {rtol!r} }},')
    lines.append("pub static GOLDEN: &[Golden] = &[")
    lines.extend(entries)
    lines.append("];")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    print(emit())
