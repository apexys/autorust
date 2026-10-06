"""Generate tests/nn_golden.rs — conv2d / pooling / one-hot / cross-entropy
numerics vs tinygrad (the M7 nn gate).

Inputs are deterministic sequences so the Rust side only has to rebuild them
from the emitted flat values. Run from autorust/:
  ../tinygrad/.venv/bin/python py/dump_nn_golden.py
"""
import sys
sys.path.insert(0, "../tinygrad")

from tinygrad import Tensor
from tinygrad import Context, nn
from tinygrad.dtype import dtypes


def seq(shape, scale=0.1, off=-0.3):
    n = 1
    for s in shape:
        n *= s
    return Tensor.arange(n).reshape(shape).float() * scale + off


def flat(t):
    return [float(v) for v in t.flatten().tolist()]


def shape_of(t):
    return [int(s) for s in t.shape]


CONV = []
POOL = []
CE = []
OH = []


def cadd(name, x, w, bias, groups, stride, dilation, padding):
    out = x.conv2d(w, bias, groups, stride, dilation, padding)
    padding = list(padding) if isinstance(padding, (list, tuple)) else [padding]
    CONV.append((name, x, w, bias, groups, stride, dilation, padding, out))


cadd("conv_basic", seq((1, 1, 3, 3)), seq((1, 1, 2, 2)), None, 1, 1, 1, 0)
cadd("conv_bias_stride_pad", seq((1, 2, 5, 5)), seq((3, 2, 3, 3)), seq((3,)), 1, 2, 1, 1)
cadd("conv_groups", seq((1, 4, 5, 5)), seq((4, 2, 2, 2)), None, 2, 1, 1, 0)
cadd("conv_dilation", seq((1, 1, 6, 6)), seq((1, 1, 2, 2)), seq((1,)), 1, 1, 2, 0)
cadd("conv_pad_lrtb", seq((1, 1, 5, 5)), seq((1, 1, 3, 3)), None, 1, 1, 1, [1, 1, 1, 1])


def padd(name, x, k, stride, dilation, padding, ceil_mode, return_indices):
    r = x.max_pool2d(tuple(k), stride, tuple(dilation), padding, ceil_mode, return_indices)
    padding = list(padding) if isinstance(padding, (list, tuple)) else [padding]
    if return_indices:
        pooled, idx = r
        POOL.append((name, x, k, stride, dilation, padding, ceil_mode, True, pooled, idx))
    else:
        POOL.append((name, x, k, stride, dilation, padding, ceil_mode, False, r, None))


padd("pool_basic", seq((1, 1, 4, 4)), [2, 2], None, [1, 1], 0, False, False)
padd("pool_pad", seq((1, 1, 4, 4)), [2, 2], None, [1, 1], 1, False, False)
padd("pool_ceil", seq((1, 1, 5, 5)), [2, 2], None, [1, 1], 1, True, False)
padd("pool_indices", seq((1, 1, 4, 4)), [2, 2], None, [1, 1], 0, False, True)


def ce(name, logits, y, ignore_index, label_smoothing, reduction):
    out = logits.sparse_categorical_crossentropy(
        Tensor(y, dtype=dtypes.int32), ignore_index=ignore_index, label_smoothing=label_smoothing, reduction=reduction
    )
    CE.append((name, logits, y, ignore_index, label_smoothing, reduction, out))


ce("ce_mean", seq((2, 3)), [1, 2], -1, 0.0, "mean")
ce("ce_sum", seq((2, 3)), [1, 2], -1, 0.0, "sum")
ce("ce_ignore", seq((2, 3)), [1, 2], 1, 0.0, "mean")
ce("ce_soft", seq((2, 3)), [0, 2], -1, 0.2, "mean")

OH.append(("one_hot", Tensor([0, 1, 3, 3, 4], dtype=dtypes.int32), 5, Tensor([0, 1, 3, 3, 4], dtype=dtypes.int32).one_hot(5)))


BN = []
GN = []
IN = []
L2 = []
RMS = []
EMB = []
LSTM = []
C1D = []
CT = []
ARGMAX = []


def amax(name, x, axis, keepdim):
    ARGMAX.append((name, x, axis, keepdim, x.argmax(axis=axis, keepdim=keepdim)))


amax("argmax_axis1", Tensor([[-1.0, 2.0, -3.0], [1.0, -2.0, 3.0]]), 1, False)
amax("argmax_axis0", Tensor([[1.0, 0.0, 2.0], [5.0, 4.0, 3.0]]), 0, False)
amax("argmax_axis1_keepdim", Tensor([[1.0, 0.0, 2.0], [5.0, 4.0, 3.0]]), 1, True)
amax("argmax_none", Tensor([[1.0, 0.0, 2.0], [5.0, 4.0, 3.0]]), None, False)
amax("argmax_ties", Tensor([[1.0, 1.0, 0.0], [2.0, 2.0, 2.0]]), 1, False)



def badd(name, sz, weight, bias, x1, x2, eps=1e-5, momentum=0.1):
    bn = nn.BatchNorm(sz, eps=eps, affine=True, track_running_stats=True, momentum=momentum)
    bn.weight, bn.bias = Tensor(list(weight), dtype=dtypes.float32), Tensor(list(bias), dtype=dtypes.float32)
    with Context(TRAINING=1):
        y1, y2 = bn(x1), bn(x2)
        rmean, rvar, nbt = bn.running_mean, bn.running_var, bn.num_batches_tracked
    BN.append((name, sz, eps, momentum, weight, bias, x1, x2, y1, y2, rmean, rvar, nbt, bn(x1)))


badd("bn_4d", 4, [1.0, 2.0, 0.5, -1.0], [0.0, 0.1, 0.2, 0.3], seq((2, 4, 3, 3), 0.1, 0.0), seq((2, 4, 3, 3), -1 / 7, 0.0))
badd("bn_2d", 3, [1.5, -0.5, 2.0], [0.1, -0.2, 0.3], seq((4, 3), 0.3, 0.5), seq((4, 3), -0.2, -0.7))


def gadd(name, num_groups, num_channels, eps, weight, bias, x):
    gn = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=True)
    gn.weight, gn.bias = Tensor(list(weight), dtype=dtypes.float32), Tensor(list(bias), dtype=dtypes.float32)
    GN.append((name, num_groups, num_channels, eps, weight, bias, x, gn(x)))


gadd("gn_4d", 2, 6, 1e-5, [1.5, -0.5, 2.0, 0.3, -1.2, 0.7], [0.1, 0.2, -0.3, 0.4, 0.5, -0.6], seq((2, 6, 2, 2), 1.0, 0.0))
gadd("gn_3d", 3, 6, 1e-5, [1.0, 0.5, -1.0, 2.0, 0.25, -0.75], [0.0, 0.1, -0.1, 0.2, -0.2, 0.3], seq((2, 6, 4), 0.5, -0.5))




def iadd(name, num_features, eps, weight, bias, x):
    m = nn.InstanceNorm(num_features, eps=eps, affine=True)
    m.weight, m.bias = Tensor(list(weight), dtype=dtypes.float32), Tensor(list(bias), dtype=dtypes.float32)
    IN.append((name, num_features, eps, weight, bias, x, m(x)))


iadd("in_4d", 3, 1e-5, [1.5, -0.5, 2.0], [0.1, -0.2, 0.3], seq((2, 3, 3, 3), 0.3, 0.5))
iadd("in_3d", 4, 1e-5, [1.0, 0.5, -1.0, 2.0], [0.0, 0.1, -0.1, 0.2], seq((2, 4, 5), 0.5, -0.5))


def l2add(name, num_channels, eps, weight, bias, x):
    m = nn.LayerNorm2d(num_channels, eps=eps)
    m.weight, m.bias = Tensor(list(weight), dtype=dtypes.float32), Tensor(list(bias), dtype=dtypes.float32)
    L2.append((name, num_channels, eps, weight, bias, x, m(x)))


l2add("ln2d_3ch", 3, 1e-5, [1.0, 0.5, -1.0], [0.1, -0.2, 0.3], seq((2, 3, 3, 3), 0.4, 0.1))
l2add("ln2d_4ch", 4, 1e-5, [1.0, 1.5, -0.5, 0.25], [-0.1, 0.2, 0.3, -0.4], seq((1, 4, 2, 4), 0.5, 0.0))


def rmsadd(name, dim, eps, weight, x):
    m = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
    m.weight = Tensor(list(weight), dtype=dtypes.float32)
    RMS.append((name, dim, eps, weight, x, m(x)))


rmsadd("rms_4", 4, 1e-6, [0.5, 1.0, 1.5, -2.0], seq((3, 4), 0.5, 0.1))
rmsadd("rms_5", 5, 1e-6, [1.0, -0.5, 0.25, 2.0, 0.75], seq((2, 3, 5), 0.3, -0.2))



def eadd(name, vocab, embed, weight, idx):
    m = nn.Embedding(vocab, embed)
    m.weight = Tensor(weight, dtype=dtypes.float32).reshape(vocab, embed)
    EMB.append((name, vocab, embed, m.weight, Tensor(idx, dtype=dtypes.int32), m(Tensor(idx, dtype=dtypes.int32))))


eadd("emb_small", 10, 3, [float(i) * 0.1 - 0.6 for i in range(30)], [1, 2, 3, 1])
eadd("emb_2d", 5, 2, [float(i) * 0.2 - 0.4 for i in range(10)], [[0, 1], [2, 4]])

def ladd(name, inp, hid, x, h0, c0):
    m = nn.LSTMCell(inp, hid)
    m.weight_ih = seq((hid * 4, inp), 0.05, -0.4)
    m.weight_hh = seq((hid * 4, hid), 0.04, -0.3)
    m.bias_ih = seq((hid * 4,), 0.1, -1.0)
    m.bias_hh = seq((hid * 4,), 0.05, -0.5)
    h1, c1 = m(x, (h0, c0))
    hz, cz = m(x)
    LSTM.append((name, inp, hid, m.weight_ih, m.weight_hh, m.bias_ih, m.bias_hh, x, h0, c0, h1, c1, hz, cz))

ladd("lstm_2x5", 4, 5, seq((2, 4), 0.3, 0.2), seq((2, 5), 0.2, -0.5), seq((2, 5), 0.1, 0.3))


def c1add(name, inc, outc, k, stride, padding, dilation, groups, x, weight, bias):
    m = nn.Conv1d(inc, outc, k, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=(bias is not None))
    m.weight = weight.reshape(outc, inc // groups, k)
    if bias is not None:
        m.bias = Tensor(bias, dtype=dtypes.float32)
    C1D.append((name, inc, outc, k, stride, padding, dilation, groups, m.weight, x, m(x), bias))


c1add("conv1d_basic", 1, 1, 3, 1, 0, 1, 1, seq((1, 1, 5), 0.3, 0.1), seq((1, 1, 3), 0.1, -0.5), [0.2])
c1add("conv1d_stride_pad", 2, 3, 2, 2, 1, 1, 1, seq((1, 2, 6), 0.2, -0.3), seq((3, 2, 2), 0.1, -0.4), None)


def ctadd(name, inc, outc, k, stride, padding, output_padding, dilation, groups, x, weight, bias):
    if isinstance(k, int):
        m = nn.ConvTranspose1d(inc, outc, k, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups)
        kt = (k,)
    else:
        m = nn.ConvTranspose2d(inc, outc, k, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups)
        kt = k
    m.weight = weight.reshape(inc, outc // groups, *kt)
    if bias is not None:
        m.bias = Tensor(bias, dtype=dtypes.float32)
    CT.append((name, inc, outc, list(kt), stride, padding, output_padding, dilation, groups, m.weight, x, m(x), bias))


ctadd("convt1d_basic", 1, 1, 3, 1, 0, 0, 1, 1, seq((1, 1, 4), 0.3, 0.1), seq((1, 1, 3), 0.1, -0.5), [0.05])
ctadd("convt2d_basic", 1, 1, (2, 2), 1, 0, 0, 1, 1, seq((1, 1, 3, 3), 0.3, 0.1), seq((1, 1, 2, 2), 0.1, -0.4), [0.1])
ctadd("convt2d_stride", 2, 3, (3, 3), 2, 1, 1, 1, 1, seq((1, 2, 4, 4), 0.2, -0.3), seq((2, 3, 3, 3), 0.1, -0.4), [0.2, -0.1, 0.05])
def s(vals):
    return ", ".join(repr(float(v)) for v in vals)


def ishape(shape):
    return ", ".join(str(s) for s in shape)


def emit():
    L = []
    L.append("// GENERATED by py/dump_nn_golden.py — nn numerics vs tinygrad.")
    L.append("// Regenerate with `../tinygrad/.venv/bin/python py/dump_nn_golden.py`.")
    L.append("#![allow(clippy::excessive_precision)]")
    L.append("#![allow(dead_code)]")
    L.append("")
    L.append("pub struct ConvGolden {")
    L.append("    pub name: &'static str, pub x_shape: &'static [usize], pub x: &'static [f32],")
    L.append("    pub w_shape: &'static [usize], pub w: &'static [f32], pub bias: &'static [f32],")
    L.append("    pub groups: i64, pub stride: i64, pub dilation: i64, pub padding: &'static [i64],")
    L.append("    pub out_shape: &'static [usize], pub out: &'static [f32],")
    L.append("}")
    L.append("")
    L.append("pub struct PoolGolden {")
    L.append("    pub name: &'static str, pub x_shape: &'static [usize], pub x: &'static [f32],")
    L.append("    pub k: &'static [i64], pub stride: &'static [i64], pub dilation: &'static [i64], pub padding: &'static [i64],")
    L.append("    pub ceil_mode: bool, pub return_indices: bool,")
    L.append("    pub out_shape: &'static [usize], pub out: &'static [f32],")
    L.append("    pub idx_shape: &'static [usize], pub idx: &'static [f32],")
    L.append("}")
    L.append("")
    L.append("pub struct CeGolden {")
    L.append("    pub name: &'static str, pub logits_shape: &'static [usize], pub logits: &'static [f32],")
    L.append("    pub y: &'static [i32], pub ignore_index: i64, pub label_smoothing: f64, pub reduction: &'static str,")
    L.append("    pub out: &'static [f32],")
    L.append("}")
    L.append("")
    L.append("pub struct OneHotGolden {")
    L.append("    pub name: &'static str, pub x: &'static [i32], pub num_classes: i64,")
    L.append("    pub out_shape: &'static [usize], pub out: &'static [f32],")
    L.append("}")
    L.append("")

    L.append("pub static CONV_GOLDEN: &[ConvGolden] = &[")
    for name, x, w, bias, g, s_, d, pad, out in CONV:
        L.append(
            f'    ConvGolden {{ name: "{name}", x_shape: &[{ishape(shape_of(x))}], x: &[{s(flat(x))}], '
            f"w_shape: &[{ishape(shape_of(w))}], w: &[{s(flat(w))}], "
            f'bias: &[{"" if bias is None else s(flat(bias))}], groups: {g}, stride: {s_}, dilation: {d}, '
            f'padding: &[{", ".join(str(p) for p in pad)}], out_shape: &[{ishape(shape_of(out))}], out: &[{s(flat(out))}] }},'
        )
    L.append("];")
    L.append("")

    L.append("pub static POOL_GOLDEN: &[PoolGolden] = &[")
    for name, x, k, stride, dilation, pad, ceil_mode, ri, out, idx in POOL:
        stride_s = "&[]" if stride is None else f'&[{", ".join(str(v) for v in stride)}]'
        idx_shape = "&[]" if idx is None else f"&[{ishape(shape_of(idx))}]"
        idx_vals = "" if idx is None else s(flat(idx))
        L.append(
            f'    PoolGolden {{ name: "{name}", x_shape: &[{ishape(shape_of(x))}], x: &[{s(flat(x))}], '
            f'k: &[{", ".join(str(v) for v in k)}], stride: {stride_s}, '
            f'dilation: &[{", ".join(str(v) for v in dilation)}], padding: &[{", ".join(str(p) for p in pad)}], '
            f'ceil_mode: {str(ceil_mode).lower()}, return_indices: {str(ri).lower()}, '
            f'out_shape: &[{ishape(shape_of(out))}], out: &[{s(flat(out))}], idx_shape: {idx_shape}, idx: &[{idx_vals}] }},'
        )
    L.append("];")
    L.append("")

    L.append("pub static CE_GOLDEN: &[CeGolden] = &[")
    for name, logits, y, ig, ls, red, out in CE:
        L.append(
            f'    CeGolden {{ name: "{name}", logits_shape: &[{ishape(shape_of(logits))}], logits: &[{s(flat(logits))}], '
            f'y: &[{", ".join(str(int(v)) for v in y)}], ignore_index: {ig}, label_smoothing: {ls!r}, '
            f'reduction: "{red}", out: &[{s(flat(out))}] }},'
        )
    L.append("];")
    L.append("")

    L.append("pub static ONE_HOT_GOLDEN: &[OneHotGolden] = &[")
    for name, x, nc, out in OH:
        L.append(
            f'    OneHotGolden {{ name: "{name}", x: &[{", ".join(str(int(v)) for v in x.tolist())}], '
            f"num_classes: {nc}, out_shape: &[{ishape(shape_of(out))}], out: &[{s(flat(out))}] }},"
        )
    L.append("];")
    L.append("")

    L.append("pub struct BnGolden {")
    L.append("    pub name: &'static str, pub sz: i64, pub eps: f64, pub momentum: f64,")
    L.append("    pub weight: &'static [f32], pub bias: &'static [f32],")
    L.append("    pub x1_shape: &'static [usize], pub x1: &'static [f32],")
    L.append("    pub x2_shape: &'static [usize], pub x2: &'static [f32],")
    L.append("    pub y1_shape: &'static [usize], pub y1: &'static [f32],")
    L.append("    pub y2_shape: &'static [usize], pub y2: &'static [f32],")
    L.append("    pub running_mean: &'static [f32], pub running_var: &'static [f32], pub num_batches_tracked: i64,")
    L.append("    pub eval_shape: &'static [usize], pub eval_out: &'static [f32],")
    L.append("}")
    L.append("")
    L.append("pub static BN_GOLDEN: &[BnGolden] = &[")
    for name, sz, eps, momentum, weight, bias, x1, x2, y1, y2, rmean, rvar, nbt, eval_out in BN:
        L.append(
            f'    BnGolden {{ name: "{name}", sz: {sz}, eps: {eps!r}, momentum: {momentum!r}, '
            f"weight: &[{s(weight)}], bias: &[{s(bias)}], "
            f"x1_shape: &[{ishape(shape_of(x1))}], x1: &[{s(flat(x1))}], x2_shape: &[{ishape(shape_of(x2))}], x2: &[{s(flat(x2))}], "
            f"y1_shape: &[{ishape(shape_of(y1))}], y1: &[{s(flat(y1))}], y2_shape: &[{ishape(shape_of(y2))}], y2: &[{s(flat(y2))}], "
            f"running_mean: &[{s(flat(rmean))}], running_var: &[{s(flat(rvar))}], num_batches_tracked: {int(nbt.tolist())}, "
            f"eval_shape: &[{ishape(shape_of(eval_out))}], eval_out: &[{s(flat(eval_out))}] }},"
        )
    L.append("];")
    L.append("")

    L.append("pub struct GnGolden {")
    L.append("    pub name: &'static str, pub num_groups: i64, pub num_channels: i64, pub eps: f64,")
    L.append("    pub weight: &'static [f32], pub bias: &'static [f32],")
    L.append("    pub x_shape: &'static [usize], pub x: &'static [f32],")
    L.append("    pub out_shape: &'static [usize], pub out: &'static [f32],")
    L.append("}")
    L.append("")
    L.append("pub static GN_GOLDEN: &[GnGolden] = &[")
    for name, ng, nc, eps, weight, bias, x, out in GN:
        L.append(
            f'    GnGolden {{ name: "{name}", num_groups: {ng}, num_channels: {nc}, eps: {eps!r}, '
            f"weight: &[{s(weight)}], bias: &[{s(bias)}], "
            f"x_shape: &[{ishape(shape_of(x))}], x: &[{s(flat(x))}], out_shape: &[{ishape(shape_of(out))}], out: &[{s(flat(out))}] }},"
        )
    L.append("];")
    L.append("")

    L.append("pub struct InGolden {")
    L.append("    pub name: &'static str, pub num_features: i64, pub eps: f64,")
    L.append("    pub weight: &'static [f32], pub bias: &'static [f32],")
    L.append("    pub x_shape: &'static [usize], pub x: &'static [f32],")
    L.append("    pub out_shape: &'static [usize], pub out: &'static [f32],")
    L.append("}")
    L.append("")
    L.append("pub static IN_GOLDEN: &[InGolden] = &[")
    for name, nf, eps, weight, bias, x, out in IN:
        L.append(
            f'    InGolden {{ name: "{name}", num_features: {nf}, eps: {eps!r}, '
            f"weight: &[{s(weight)}], bias: &[{s(bias)}], "
            f"x_shape: &[{ishape(shape_of(x))}], x: &[{s(flat(x))}], out_shape: &[{ishape(shape_of(out))}], out: &[{s(flat(out))}] }},"
        )
    L.append("];")
    L.append("")
    L.append("pub struct L2Golden {")
    L.append("    pub name: &'static str, pub num_channels: i64, pub eps: f64,")
    L.append("    pub weight: &'static [f32], pub bias: &'static [f32],")
    L.append("    pub x_shape: &'static [usize], pub x: &'static [f32],")
    L.append("    pub out_shape: &'static [usize], pub out: &'static [f32],")
    L.append("}")
    L.append("")
    L.append("pub static L2_GOLDEN: &[L2Golden] = &[")
    for name, nc, eps, weight, bias, x, out in L2:
        L.append(
            f'    L2Golden {{ name: "{name}", num_channels: {nc}, eps: {eps!r}, '
            f"weight: &[{s(weight)}], bias: &[{s(bias)}], "
            f"x_shape: &[{ishape(shape_of(x))}], x: &[{s(flat(x))}], out_shape: &[{ishape(shape_of(out))}], out: &[{s(flat(out))}] }},"
        )
    L.append("];")
    L.append("")
    L.append("pub struct RmsGolden {")
    L.append("    pub name: &'static str, pub dim: i64, pub eps: f64,")
    L.append("    pub weight: &'static [f32],")
    L.append("    pub x_shape: &'static [usize], pub x: &'static [f32],")
    L.append("    pub out_shape: &'static [usize], pub out: &'static [f32],")
    L.append("}")
    L.append("")
    L.append("pub static RMS_GOLDEN: &[RmsGolden] = &[")
    for name, dim, eps, weight, x, out in RMS:
        L.append(
            f'    RmsGolden {{ name: "{name}", dim: {dim}, eps: {eps!r}, '
            f"weight: &[{s(weight)}], "
            f"x_shape: &[{ishape(shape_of(x))}], x: &[{s(flat(x))}], out_shape: &[{ishape(shape_of(out))}], out: &[{s(flat(out))}] }},"
        )
    L.append("];")
    L.append("")
    L.append("pub struct EmbGolden {")
    L.append("    pub name: &'static str, pub vocab_size: i64, pub embed_size: i64,")
    L.append("    pub weight_shape: &'static [usize], pub weight: &'static [f32],")
    L.append("    pub idx_shape: &'static [usize], pub idx: &'static [i32],")
    L.append("    pub out_shape: &'static [usize], pub out: &'static [f32],")
    L.append("}")
    L.append("")
    L.append("pub static EMB_GOLDEN: &[EmbGolden] = &[")
    for name, vocab, embed, weight, idx, out in EMB:
        L.append(
            f'    EmbGolden {{ name: "{name}", vocab_size: {vocab}, embed_size: {embed}, '
            f"weight_shape: &[{ishape(shape_of(weight))}], weight: &[{s(flat(weight))}], "
            f'idx_shape: &[{ishape(shape_of(idx))}], idx: &[{", ".join(str(int(v)) for v in idx.flatten().tolist())}], '
            f"out_shape: &[{ishape(shape_of(out))}], out: &[{s(flat(out))}] }},"
        )
    L.append("];")
    L.append("")
    L.append("pub struct LstmGolden {")
    L.append("    pub name: &'static str, pub input_size: i64, pub hidden_size: i64,")
    L.append("    pub weight_ih_shape: &'static [usize], pub weight_ih: &'static [f32],")
    L.append("    pub weight_hh_shape: &'static [usize], pub weight_hh: &'static [f32],")
    L.append("    pub bias_ih: &'static [f32], pub bias_hh: &'static [f32],")
    L.append("    pub x_shape: &'static [usize], pub x: &'static [f32],")
    L.append("    pub h0_shape: &'static [usize], pub h0: &'static [f32], pub c0_shape: &'static [usize], pub c0: &'static [f32],")
    L.append("    pub h1_shape: &'static [usize], pub h1: &'static [f32], pub c1_shape: &'static [usize], pub c1: &'static [f32],")
    L.append("    pub hz_shape: &'static [usize], pub hz: &'static [f32], pub cz_shape: &'static [usize], pub cz: &'static [f32],")
    L.append("}")
    L.append("")
    L.append("pub static LSTM_GOLDEN: &[LstmGolden] = &[")
    for name, inp, hid, wih, whh, bih, bhh, x, h0, c0, h1, c1, hz, cz in LSTM:
        L.append(
            f'    LstmGolden {{ name: "{name}", input_size: {inp}, hidden_size: {hid}, '
            f"weight_ih_shape: &[{ishape(shape_of(wih))}], weight_ih: &[{s(flat(wih))}], "
            f"weight_hh_shape: &[{ishape(shape_of(whh))}], weight_hh: &[{s(flat(whh))}], "
            f"bias_ih: &[{s(flat(bih))}], bias_hh: &[{s(flat(bhh))}], "
            f"x_shape: &[{ishape(shape_of(x))}], x: &[{s(flat(x))}], "
            f"h0_shape: &[{ishape(shape_of(h0))}], h0: &[{s(flat(h0))}], c0_shape: &[{ishape(shape_of(c0))}], c0: &[{s(flat(c0))}], "
            f"h1_shape: &[{ishape(shape_of(h1))}], h1: &[{s(flat(h1))}], c1_shape: &[{ishape(shape_of(c1))}], c1: &[{s(flat(c1))}], "
            f"hz_shape: &[{ishape(shape_of(hz))}], hz: &[{s(flat(hz))}], cz_shape: &[{ishape(shape_of(cz))}], cz: &[{s(flat(cz))}] }},"
        )
    L.append("];")
    L.append("")
    L.append("pub struct Conv1dGolden {")
    L.append("    pub name: &'static str, pub in_channels: i64, pub out_channels: i64, pub kernel_size: i64,")
    L.append("    pub stride: i64, pub padding: i64, pub dilation: i64, pub groups: i64,")
    L.append("    pub weight_shape: &'static [usize], pub weight: &'static [f32], pub bias: &'static [f32],")
    L.append("    pub x_shape: &'static [usize], pub x: &'static [f32],")
    L.append("    pub out_shape: &'static [usize], pub out: &'static [f32],")
    L.append("}")
    L.append("")
    L.append("pub static C1D_GOLDEN: &[Conv1dGolden] = &[")
    for name, inc, outc, k, stride, padding, dilation, groups, weight, x, out, bias in C1D:
        L.append(
            f'    Conv1dGolden {{ name: "{name}", in_channels: {inc}, out_channels: {outc}, kernel_size: {k}, '
            f'stride: {stride}, padding: {padding}, dilation: {dilation}, groups: {groups}, '
            f"weight_shape: &[{ishape(shape_of(weight))}], weight: &[{s(flat(weight))}], bias: &[{'' if bias is None else s(bias)}], "
            f"x_shape: &[{ishape(shape_of(x))}], x: &[{s(flat(x))}], out_shape: &[{ishape(shape_of(out))}], out: &[{s(flat(out))}] }},"
        )
    L.append("];")
    L.append("")
    L.append("pub struct ConvTGolden {")
    L.append("    pub name: &'static str, pub in_channels: i64, pub out_channels: i64, pub kernel_size: &'static [i64],")
    L.append("    pub stride: i64, pub padding: i64, pub output_padding: i64, pub dilation: i64, pub groups: i64,")
    L.append("    pub weight_shape: &'static [usize], pub weight: &'static [f32], pub bias: &'static [f32],")
    L.append("    pub x_shape: &'static [usize], pub x: &'static [f32],")
    L.append("    pub out_shape: &'static [usize], pub out: &'static [f32],")
    L.append("}")
    L.append("")
    L.append("pub static CT_GOLDEN: &[ConvTGolden] = &[")
    for name, inc, outc, kt, stride, padding, opad, dilation, groups, weight, x, out, bias in CT:
        L.append(
            f'    ConvTGolden {{ name: "{name}", in_channels: {inc}, out_channels: {outc}, kernel_size: &[{", ".join(str(v) for v in kt)}], '
            f"stride: {stride}, padding: {padding}, output_padding: {opad}, dilation: {dilation}, groups: {groups}, "
            f"weight_shape: &[{ishape(shape_of(weight))}], weight: &[{s(flat(weight))}], bias: &[{'' if bias is None else s(bias)}], "
            f"x_shape: &[{ishape(shape_of(x))}], x: &[{s(flat(x))}], out_shape: &[{ishape(shape_of(out))}], out: &[{s(flat(out))}] }},"
        )
    L.append("];")
    L.append("")
    L.append("pub struct ArgmaxGolden {")
    L.append("    pub name: &'static str, pub x_shape: &'static [usize], pub x: &'static [f32],")
    L.append("    pub axis: Option<i64>, pub keepdim: bool,")
    L.append("    pub out_shape: &'static [usize], pub out: &'static [i32],")
    L.append("}")
    L.append("")
    L.append("pub static ARGMAX_GOLDEN: &[ArgmaxGolden] = &[")
    for name, x, axis, keepdim, out in ARGMAX:
        axis_s = "None" if axis is None else f"Some({axis})"
        L.append(
            f'    ArgmaxGolden {{ name: "{name}", x_shape: &[{ishape(shape_of(x))}], x: &[{s(flat(x))}], '
            f"axis: {axis_s}, keepdim: {str(keepdim).lower()}, "
            f"out_shape: &[{ishape(shape_of(out))}], out: &[{', '.join(str(int(v)) for v in out.flatten().tolist())}] }},"
        )
    L.append("];")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    with open("tests/nn_golden.rs", "w") as f:
        f.write(emit() + "\n")
    print("wrote tests/nn_golden.rs")