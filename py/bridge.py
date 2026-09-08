"""Bridge between tinygrad's Python UOps and the Rust `autorust` crate.

JSON interchange (v1) mirroring nanoserde's derived shapes exactly:
  node  = {"op": int, "dtype": tbl_idx, "srcs": [idx...], "arg": arg, "tag": arg}
  graph = {"dtypes": [attr name...], "nodes": [...], "roots": [idx...]}
  arg   = "None" | {"Int":[i]} | {"UInt":[u]} | {"Float":[f]} | {"NonFinite":["Nan"|"PosInf"|"NegInf"]}
        | {"Bool":[b]} | {"Str":[s]} | {"Bytes":[[u8...]]} | {"Enum":[{"class":c,"value":v}]}
        | {"Dtype":[idx]} | {"Seq":{"tuple":b,"items":[...]}} | {"Pickle":[[u8...]]}

Float args reconstruct as `ConstFloat` (tinygrad's canonical float const type —
`str` matches plain float, but `repr` differs, and repr feeds the key).
"""

import ctypes
import json
import math
import pickle
import pathlib

from tinygrad.dtype import DType, AddrSpace, ConstFloat, INVERSE_DTYPES_DICT, dtypes, Invalid
from tinygrad.uop.ops import UOp, Ops, AxisType, ParamArg, KernelInfo, CallInfo
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.schedule.indexing import BufferizeOpts

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_LIB = ctypes.CDLL(str(_ROOT / "target" / "debug" / "libautorust.so"))
_LIB.autorust_rebuild.restype = ctypes.POINTER(ctypes.c_ubyte)
_LIB.autorust_rebuild.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
_LIB.autorust_exec_add.restype = None

_ATTR_TO_DTYPE = {attr: getattr(dtypes, attr) for attr in INVERSE_DTYPES_DICT.values()}


# ---------- Python UOp -> JSON ----------

def _arg_json(x, dtable, didx):
    if x is None:
        return "None"
    if x is Invalid:
        return {"ConstInvalid": []}
    if isinstance(x, ParamArg):
        vv = x.vmin_vmax
        dev = x.device
        return {
            "Param": {
                "slot": x.slot,
                "dtype": _dtype_idx(x.dtype, dtable, didx),
                "vmin_vmax": [_arg_json(vv is not None, [], {}), _const_json(vv[0]) if vv else "None", _const_json(vv[1]) if vv else "None"],
                "multiple_of": [x.multiple_of is not None],
                "multiple_of_val": x.multiple_of if x.multiple_of is not None else 0,
                "name": [x.name is not None],
                "name_val": x.name if x.name is not None else "",
                "addrspace": 0 if x.addrspace is None else int(x.addrspace),
                "axis": [x.axis is not None],
                "axis_val": x.axis if x.axis is not None else 0,
                "device": [dev is not None, isinstance(dev, tuple)],
                "device_single": dev if isinstance(dev, str) else "",
                "device_multi": list(dev) if isinstance(dev, tuple) else [],
                "volatile": x.volatile,
            }
        }
    if isinstance(x, bool):
        return {"Bool": [x]}
    if isinstance(x, (Ops, AxisType, AddrSpace)):
        cls = ("Ops" if isinstance(x, Ops) else "AxisType" if isinstance(x, AxisType) else "AddrSpace")
        return {"Enum": [{"class": cls, "value": int(x)}]}
    if isinstance(x, DType):
        key = INVERSE_DTYPES_DICT[x.name]
        if key not in didx:
            didx[key] = len(dtable)
            dtable.append(key)
        return {"Dtype": [didx[key]]}
    if isinstance(x, int):
        if -(2**63) <= x < 2**63:
            return {"Int": [x]}
        if 0 <= x < 2**64:
            return {"UInt": [x]}
        neg = 1 if x < 0 else 0
        mag = abs(x)
        limbs = []
        while mag:
            limbs.append(mag & 0xFFFF_FFFF_FFFF_FFFF)
            mag >>= 64
        return {"Big": [neg, limbs]}
    elif isinstance(x, float):
        if math.isnan(x):
            return {"NonFinite": ["Nan"]}
        if math.isinf(x):
            return {"NonFinite": ["PosInf" if x > 0 else "NegInf"]}
        return {"Float": [x]}
    elif isinstance(x, str):
        return {"Str": [x]}
    elif isinstance(x, (bytes, bytearray)):
        return {"Bytes": [list(x)]}
    elif isinstance(x, (tuple, list)):
        return {"Seq": {"tuple": isinstance(x, tuple), "items": [_arg_json(i, dtable, didx) for i in x]}}
    return {"Pickle": [list(pickle.dumps(x))]}


def _const_json(v):
    """PyConst (bool|int|float) -> FlatArg json (dtypes never appear here)."""
    return _arg_json(v, [], {})


def uop_to_json(*roots: UOp) -> str:
    """Serialize roots' graph to the interchange JSON (post-order, id-deduped)."""
    dtable, didx = [], {}
    nodes, nidx = [], {}

    def visit(u: UOp) -> int:
        k = id(u)
        if k in nidx:
            return nidx[k]
        srcs = [visit(s) for s in u.src]
        node = {
            "op": int(u.op),
            "dtype": _dtype_idx(u.dtype, dtable, didx),
            "srcs": srcs,
            "arg": _arg_json(u.arg, dtable, didx),
            "tag": _arg_json(u.tag, dtable, didx),
        }
        nidx[k] = len(nodes)
        nodes.append(node)
        return nidx[k]

    root_idx = [visit(r) for r in roots]
    return json.dumps({"dtypes": dtable, "nodes": nodes, "roots": root_idx}, separators=(",", ":"))


def _dtype_idx(dt: DType, dtable, didx) -> int:
    key = INVERSE_DTYPES_DICT[dt.name]
    if key not in didx:
        didx[key] = len(dtable)
        dtable.append(key)
    return didx[key]


# ---------- JSON -> Python UOp ----------

def _nonfinite(x: str) -> float:
    return {"Nan": math.nan, "PosInf": math.inf, "NegInf": -math.inf}[x]


def _arg_py(j, dts):
    if j == "None":
        return None
    kind = next(iter(j))
    v = j[kind]
    if kind == "Int":
        return v[0]
    if kind == "UInt":
        return v[0]
    if kind == "Big":
        # rust serde emits the struct variant as a map, older bridge a 2-list
        if isinstance(v, dict):
            neg, limbs = v["neg"], v["limbs"]
        else:
            neg, limbs = v[0], v[1]
        mag = 0
        for limb in reversed(limbs):
            mag = (mag << 64) | limb
        return -mag if neg else mag
    if kind == "ConstInvalid":
        return Invalid
    if kind == "Param":
        p = v
        vv = None
        if _arg_py(p["vmin_vmax"][0], dts):
            vv = (_arg_py(p["vmin_vmax"][1], dts), _arg_py(p["vmin_vmax"][2], dts))
        addrspace = None if p["addrspace"] == 0 else AddrSpace(p["addrspace"])
        device = None
        if p["device"][0]:
            device = p["device_single"] if p["device"][1] == 0 else tuple(p["device_multi"])
        return ParamArg(
            p["slot"], dts[p["dtype"]], vv,
            p["multiple_of_val"] if p["multiple_of"][0] else None,
            p["name_val"] if p["name"][0] else None,
            addrspace,
            p["axis_val"] if p["axis"][0] else None,
            device, p["volatile"],
        )
    if kind == "Float":
        return ConstFloat(v[0])
    if kind == "NonFinite":
        return ConstFloat(_nonfinite(v[0]))
    if kind == "Bool":
        return v[0]
    if kind == "Str":
        return v[0]
    if kind == "Bytes":
        return bytes(v[0])
    if kind == "Enum":
        e = v[0]
        cls = {"Ops": Ops, "AxisType": AxisType, "AddrSpace": AddrSpace}[e["class"]]
        return cls(e["value"])
    if kind == "Dtype":
        return dts[v[0]]
    if kind == "Seq":
        items = [_arg_py(i, dts) for i in v["items"]]
        return tuple(items) if v["tuple"] else items
    if kind == "Pickle":
        return pickle.loads(bytes(v[0]))
    if kind == "Data":
        fs = {k: _arg_py(f, dts) for k, f in v["fields"]}
        if v["tag"] == "KernelInfo":
            return KernelInfo(name=fs["name"], axis_types=tuple(fs["axis_types"]),
                              dont_use_locals=fs["dont_use_locals"], applied_opts=tuple(fs["applied_opts"]),
                              opts_to_apply=fs["opts_to_apply"], beam=fs["beam"])
        if v["tag"] == "CallInfo":
            return CallInfo(None, fs["name"], fs["precompile"], fs["precompile_backward"])
        if v["tag"] == "BufferizeOpts":
            return BufferizeOpts(fs["device"], fs["addrspace"], fs["removable"])
        if v["tag"] == "Opt":
            op = OptOps[fs["op"]] if isinstance(fs["op"], str) else fs["op"]
            return Opt(op=op, axis=fs["axis"], arg=fs["arg"])
        raise ValueError(f"unknown dataclass {v['tag']}")
    raise ValueError(f"unknown arg kind {kind}")


def json_to_uop(s: str) -> list[UOp]:
    """Reconstruct the roots of a serialized graph as Python UOps."""
    data = json.loads(s)
    dts = [_ATTR_TO_DTYPE[name] for name in data["dtypes"]]
    nodes = []
    for n in data["nodes"]:
        srcs = tuple(nodes[i] for i in n["srcs"])
        op = Ops(n["op"])
        nodes.append(UOp(op, dts[n["dtype"]], srcs, _arg_py(n["arg"], dts), _arg_py(n["tag"], dts)))
    return [nodes[i] for i in data["roots"]]


# ---------- cdylib calls ----------

def rust_rebuild(data: bytes) -> bytes:
    """Round-trip a JSON graph through the Rust interned graph."""
    buf = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    out_len = ctypes.c_size_t()
    ptr = _LIB.autorust_rebuild(buf, len(data), ctypes.byref(out_len))
    if not ptr:
        raise RuntimeError("rust rebuild failed (parse error?)")
    out = ctypes.string_at(ptr, out_len.value)
    _LIB.autorust_free(ctypes.cast(ptr, ctypes.c_void_p), out_len.value)
    return out


def rust_rebuild_json(json_str: str) -> str:
    return rust_rebuild(json_str.encode()).decode()


def rust_exec_add(a, b):
    """M0 smoke kernel: elementwise f32 add via the cdylib."""
    n = len(a)
    at = (ctypes.c_float * n)(*a)
    bt = (ctypes.c_float * n)(*b)
    out = (ctypes.c_float * n)()
    _LIB.autorust_exec_add(at, bt, out, n)
    return list(out)


# ---------- oracles / execution polyfill ----------

def graph_key(*roots: UOp) -> list[str]:
    return [r.key.hex() for r in roots]


def realize_uops(*uops: UOp) -> None:
    """Polyfill executor: realize arbitrary UOps through real tinygrad (M3+)."""
    from tinygrad.tensor import Tensor
    from tinygrad.engine.realize import run_linear
    ts = [Tensor(u) for u in uops]
    linear, var_vals = ts[0].linear_with_vars(*ts[1:])
    run_linear(linear, var_vals)
