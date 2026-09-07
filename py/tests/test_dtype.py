"""M1 gate: differential test of the Rust dtype table against tinygrad.

Runs `oracle_dtype` and compares every line against Python's own dtype logic.
Float comparisons go through bit patterns (so -0.0 and NaN are exact).
"""

import math
import pathlib
import struct
import subprocess
import unittest

from tinygrad.dtype import dtypes, truncate, least_upper_dtype, can_lossless_cast, DType, INVERSE_DTYPES_DICT

BIN = pathlib.Path(__file__).resolve().parent.parent.parent / "target" / "debug" / "oracle_dtype"

# the closed set under test: dtypes.all + weak dtypes (mirrors oracle_dtype)
ALL = list(dtypes.all) + [dtypes.weakint, dtypes.weakfloat]


def INVERSE_ATTR(d: DType) -> str:
    return INVERSE_DTYPES_DICT[d.name]


BY_ATTR = {INVERSE_ATTR(d): d for d in ALL}
_CANNOT = {"weakint", "weakfloat", "_uint128", "_uint256", "void"}


def truncate_is_int(d: DType) -> bool:
    return d in dtypes.ints


def parse_py(s: str):
    kind, _, rest = s.partition(":")
    if kind == "int":
        return int(rest)
    if kind == "bool":
        return rest == "True"
    if kind == "float":
        return float(rest)
    if kind == "invalid":
        return "invalid"
    raise ValueError(s)


def bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def py_trunc_repr(v, dt):
    """Python-side canonical value for a truncate result, comparable to parse_py."""
    if isinstance(v, bool):
        return ("bool", v)
    if isinstance(v, int):
        return ("int", v)
    return ("float", bits(float(v)))


class TestDtypeDifferential(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        out = subprocess.run([str(BIN)], capture_output=True, text=True, check=True).stdout
        cls.lu, cls.cl, cls.minmax, cls.trunc = {}, {}, {}, []
        for line in out.splitlines():
            parts = line.split("\t")
            if parts[0] == "LU":
                cls.lu[(parts[1], parts[2])] = parts[3]
            elif parts[0] == "CL":
                cls.cl[(parts[1], parts[2])] = parts[3] == "1"
            elif parts[0] == "MINMAX":
                cls.minmax[parts[1]] = (parse_py(parts[2]), parse_py(parts[3]))
            elif parts[0] == "TRUNC":
                cls.trunc.append((parts[1], parts[2], parse_py(parts[3])))

    def test_least_upper_all_pairs(self):
        for a in ALL:
            for b in ALL:
                rust = self.lu[(INVERSE_ATTR(a), INVERSE_ATTR(b))]
                mine = INVERSE_ATTR(least_upper_dtype(a, b))
                self.assertEqual(rust, mine, f"least_upper({INVERSE_ATTR(a)},{INVERSE_ATTR(b)})")

    def test_can_lossless_all_pairs(self):
        for a in ALL:
            for b in ALL:
                rust = self.cl[(INVERSE_ATTR(a), INVERSE_ATTR(b))]
                mine = can_lossless_cast(a, b)
                self.assertEqual(rust, mine, f"can_lossless({INVERSE_ATTR(a)},{INVERSE_ATTR(b)})")

    def test_minmax(self):
        for attr, (rmn, rmx) in self.minmax.items():
            d = BY_ATTR[attr]
            pmn, pmx = d.min, d.max
            def norm(v):
                if isinstance(v, bool):
                    return ("bool", v)
                if isinstance(v, int):
                    return ("int", v)
                return ("float", bits(float(v)))
            self.assertEqual(norm(rmn), norm(pmn), f"min {attr}")
            self.assertEqual(norm(rmx), norm(pmx), f"max {attr}")

    def test_truncate_sweep(self):
        def norm_scalar(v):
            if isinstance(v, bool):
                return ("bool", v)
            if isinstance(v, int):
                return ("int", v)
            return ("float", bits(float(v)))
        for attr, val_s, rust in self.trunc:
            if attr in _CANNOT:
                continue
            d = BY_ATTR[attr]
            v = float(val_s)
            if truncate_is_int(d):
                # tinygrad only feeds the int arms of truncate already-const
                # values (Python ints): truncate[d](d.const(v)). NaN/Inf raise
                # in CPython (ctypes) and are unreachable from tinygrad flows.
                if not math.isfinite(v):
                    continue
                mine = truncate[d](d.const(v))
            else:
                mine = truncate[d](v)
            self.assertEqual(norm_scalar(rust), norm_scalar(mine), f"truncate[{attr}]({val_s})")


if __name__ == "__main__":
    unittest.main()
