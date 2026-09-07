"""M0 gate: UOp -> JSON -> Rust -> JSON -> UOp preserves `key` and `repr`.

Corpus = tensor programs harvested from tinygrad's own test surface. Each case
builds lazy tensors; we take `.uop` roots (no realization), round-trip through
the Rust interned graph, and compare Python-side keys/reprs.
"""

import math
import sys
import unittest

sys.path.insert(0, str(__file__).rsplit("/py/", 1)[0] + "/py")
from bridge import rust_rebuild_json, rust_exec_add, json_to_uop, uop_to_json  # noqa: E402

from tinygrad import Tensor, Variable, dtypes  # noqa: E402

CORPUS = {}


def case(fn):
    CORPUS[fn.__name__] = fn
    return fn


@case
def scalar(): return [Tensor(2.0)]


@case
def add(): return [Tensor([1.0, 2, 3]) + Tensor([4.0, 5, 6])]


@case
def add_int(): return [Tensor([1, 2, 3], dtype=dtypes.int32) + Tensor([4, 5, 6], dtype=dtypes.int32)]


@case
def sum_axis(): return [Tensor([[1.0, 2], [3, 4]]).sum(axis=1)]


@case
def sum_full(): return [Tensor.ones(4, 4).contiguous().sum()]


@case
def max_axis(): return [Tensor([[1.0, 2], [3, 4]]).max(axis=0)]


@case
def mean(): return [Tensor([[1.0, 2], [3, 4]]).mean()]


@case
def reshape_permute(): return [Tensor([[1.0, 2], [3, 4]]).reshape(2, 2).permute(1, 0)]


@case
def expand(): return [Tensor([1.0, 2]).expand(2, 2)]


@case
def slice_(): return [Tensor.arange(0, 8).reshape(2, 4)[:, 1:3]]


@case
def flip(): return [Tensor.arange(0, 4).flip(0)]


@case
def pad(): return [Tensor.arange(0, 4).reshape(2, 2).pad(((1, 1), (0, 0)))]


@case
def cat(): return [Tensor.cat(Tensor.ones(4).contiguous(), Tensor.zeros(4).contiguous())]


@case
def matmul(): return [Tensor.ones(3, 3).contiguous() @ Tensor.eye(3)]


@case
def cast(): return [Tensor([1.5, 2.5]).cast(dtypes.int32)]


@case
def bitcast(): return [Tensor([1.5, 2.5], dtype=dtypes.float32).bitcast(dtypes.int32)]


@case
def where(): return [(Tensor([1.0, 2]) < Tensor([2.0, 1])).where(Tensor([1.0, 2]), Tensor([3.0, 4]))]


@case
def arange(): return [Tensor.arange(0, 10)]


@case
def eye(): return [Tensor.eye(3)]


@case
def unary(): return [Tensor([0.5, 2.0]).sqrt().exp2(), Tensor([0.5]).log2(), Tensor([0.5]).sin()]


@case
def neg(): return [-Tensor([1.0, 2])]


@case
def pow(): return [Tensor([2.0, 3.0]) ** 2]


@case
def floordiv(): return [Tensor([7, 8]) // Tensor([2, 3])]


@case
def cmp_reduce(): return [(Tensor([[1.0, 2], [3, 4]]) == Tensor([[1.0, 2], [3, 4]])).sum()]


@case
def nan_const(): return [Tensor(math.nan)]


@case
def negzero(): return [Tensor(-0.0)]


@case
def inf_const(): return [Tensor(math.inf)]


@case
def bool_tensor(): return [Tensor(True)]


@case
def contiguous(): return [Tensor.ones(4, 4).contiguous()]


@case
def symbolic():
    i = Variable("i", 1, 10)
    return [Tensor.ones(10).contiguous()[:i.bind(5)] + 1]


@case
def chained(): return [(Tensor([[1.0, 2], [3, 4]]) @ Tensor.eye(2)).sum(axis=0).sqrt()]


class TestRoundtrip(unittest.TestCase):

    def test_roundtrip_keys_and_reprs(self):
        for name, fn in CORPUS.items():
            with self.subTest(case=name):
                roots = [t.uop for t in fn()]
                json1 = uop_to_json(*roots)
                json2 = rust_rebuild_json(json1)
                roots2 = json_to_uop(json2)
                self.assertEqual(len(roots), len(roots2))
                for a, b in zip(roots, roots2):
                    self.assertEqual(a.key, b.key, f"key mismatch in {name}")
                    self.assertEqual(repr(a), repr(b), f"repr mismatch in {name}")

    def test_cdylib_exec_add(self):
        self.assertEqual(rust_exec_add([1.0, 2.0, 3.5], [4.0, 5.0, 0.5]), [5.0, 7.0, 4.0])

    def test_nan_survives(self):
        t = Tensor(math.nan) + 1
        roots = [t.uop]
        roots2 = json_to_uop(rust_rebuild_json(uop_to_json(*roots)))
        self.assertEqual(roots[0].key, roots2[0].key)


if __name__ == "__main__":
    unittest.main()
