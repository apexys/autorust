#!/usr/bin/env python
"""M2 differential gate: Rust graph_rewrite vs Python's, per PatternMatcher.

For a corpus of random symbolic / movement graphs, each is rewritten by BOTH
engines:
  - Python:   graph_rewrite(uop, PM_py)
  - Rust:     oracle_rewrite subprocess with the same-named PM
and the results must be key-identical (structural identity via the M0 bridge).
Python exceptions and Rust panics both mark the case "error": both-error is a
skip, single-side error is a failure.
"""

import json
import os
import pathlib
import random
import subprocess
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "tinygrad"))

from bridge import uop_to_json, json_to_uop  # noqa: E402

from tinygrad.dtype import dtypes  # noqa: E402
from tinygrad.uop.ops import UOp, Ops, graph_rewrite  # noqa: E402
from tinygrad.uop.symbolic import symbolic, symbolic_simple, commutative  # noqa: E402
from tinygrad.uop.movement import mop_cleanup  # noqa: E402
from tinygrad.uop.divandmod import div_and_mod_symbolic  # noqa: E402

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_ORACLE = _ROOT / "target" / "debug" / "oracle_rewrite"


def uconst(val, dtype=None):
    return UOp.const(val, dtype)


def Variable(name, lo, hi, dtype=dtypes.weakint):
    return UOp.variable(name, lo, hi, dtype, param=True)


class Oracle:
    def __init__(self):
        self.p = subprocess.Popen(
            [str(_ORACLE)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )

    def rewrite(self, pm_name, uop):
        line = '{"pm": "%s", "graph": %s}' % (pm_name, uop_to_json(uop))
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        resp = json.loads(self.p.stdout.readline())
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "oracle error"))
        return json_to_uop(json.dumps(resp["graph"]))


# ---------- random graph generators ----------

BINOPS = [
    ("add", lambda a, b: a + b),
    ("sub", lambda a, b: a - b),
    ("mul", lambda a, b: a * b),
    ("div", lambda a, b: a // b),
    ("mod", lambda a, b: a % b),
    ("lt", lambda a, b: a < b),
    ("ne", lambda a, b: a != b),
    ("eq", lambda a, b: a == b),
    ("and", lambda a, b: a & b),
    ("or", lambda a, b: a | b),
    ("xor", lambda a, b: a ^ b),
    ("shl", lambda a, b: a << b),
    ("shr", lambda a, b: a >> b),
    ("max", lambda a, b: a.maximum(b)),
    ("min", lambda a, b: a.minimum(b)),
]
UNOPS = [lambda a: -a, lambda a: a.logical_not(), lambda a: a.sqrt(), lambda a: a.reciprocal()]


INT_ONLY_OPS = {"shl", "shr", "and", "or", "xor", "div", "mod"}


def rand_expr(rng, vars_, depth):
    if depth == 0 or rng.random() < 0.35:
        r = rng.random()
        if r < 0.5 and vars_:
            return rng.choice(vars_)
        if r < 0.75:
            return uconst(rng.randint(-20, 20))
        return uconst(round(rng.uniform(-10, 10), 2))
    r = rng.random()
    a, b = rand_expr(rng, vars_, depth - 1), rand_expr(rng, vars_, depth - 1)
    if r < 0.8:
        name, op = rng.choice(BINOPS)
        if name in ("div", "mod") and not isinstance(b, UOp):
            # force a non-zero const denominator (0 raises in Python)
            denom = uconst(rng.choice([2, 3, 4, 5, 7, 8]))
            return (a // denom) if name == "div" else (a % denom)
        if name in INT_ONLY_OPS:
            # shifts/bitwise/floor-div are int-only: coerce both sides
            a, b = a.cast(dtypes.weakint), b.cast(dtypes.weakint)
        return op(a, b)
    if r < 0.9:
        return rng.choice(UNOPS)(a)
    # where
    cond = (a < b) if rng.random() < 0.5 else uconst(rng.choice([True, False]))
    return cond.where(a, b)


def rand_symbolic_graph(rng):
    for _ in range(20):
        try:
            vars_ = []
            for i in range(rng.randint(1, 3)):
                lo = rng.randint(-4, 4)
                hi = lo + rng.randint(1, 12)
                vars_.append(Variable(f"v{i}", lo, hi))
            return rand_expr(rng, vars_, rng.randint(2, 4))
        except Exception:
            continue
    raise AssertionError("generator failed 20x")


def rand_movement_graph(rng):
    for _ in range(20):
        try:
            return _rand_movement_graph_inner(rng)
        except Exception:
            continue
    raise AssertionError("movement generator failed 20x")


def _rand_movement_graph_inner(rng):
    n = rng.randint(2, 8)
    base = UOp.stack(*[uconst(rng.randint(-5, 5)) for _ in range(n)])
    u = base
    for _ in range(rng.randint(1, 4)):
        r = rng.random()
        shape = list(u.shape)
        if r < 0.3:
            # reshape to a compatible shape
            import math

            numel = math.prod(shape)
            choices = [(2, numel // 2), (numel,), (1, numel), (numel, 1)]
            new = rng.choice(choices)
            if math.prod(new) == numel:
                u = u.reshape(new)
        elif r < 0.5 and len(shape) > 1:
            perm = list(range(len(shape)))
            rng.shuffle(perm)
            u = u.permute(tuple(perm))
        elif r < 0.65:
            u = u.flip(tuple(rng.randint(0, 1) for _ in shape))
        elif r < 0.8:
            u = u.expand((1,) + tuple(shape))
        else:
            i = rng.randint(0, n - 1)
            u = u.index(i)
    return u


def rand_index_graph(rng):
    """Graphs exercising `pm_load_collapse` (indexing a tensor with a tensor)."""
    for _ in range(20):
        try:
            n = rng.randint(2, 6)
            base = UOp.stack(*[uconst(rng.randint(-5, 5)) for _ in range(n)])
            r = UOp.range(n, rng.randint(0, 3))
            idx = Variable("idx", -2, n + 2)
            expr = base.index(idx.cast(dtypes.weakint))
            style = rng.random()
            if style < 0.45:
                # the gather pattern: (idx != r).where(0, load(base, idx)).reduce(r, ADD)
                return (idx.cast(dtypes.weakint) != r).where(uconst(0), expr).reduce(r, arg=Ops.ADD)
            if style < 0.75:
                # (weakint x + y) < c with x containing an INDEX (needs no-load y,c)
                x = base.index(r)
                y = Variable("y", 0, n + 3)
                c = uconst(rng.randint(-4, 10))
                return (x + y) < c
            # bare reduce over an index expression
            return expr.reduce(r, arg=Ops.ADD)
        except Exception:
            continue
    raise AssertionError("index generator failed 20x")


def rand_split_graph(rng):
    """Graphs exercising `pm_split_ranges` (a range used modulo a divisor of its length)."""
    for _ in range(20):
        try:
            n = rng.choice([4, 6, 8, 9, 12, 16])
            c = rng.choice([d for d in range(2, n + 1) if n % d == 0])
            r = UOp.range(n, rng.randint(0, 3))
            parts = [r % uconst(c)]
            if rng.random() < 0.6:
                parts.append(r)
            if rng.random() < 0.5:
                parts.append(uconst(rng.randint(-5, 5)))
            if rng.random() < 0.4:
                parts.append(Variable("w", -3, 5))
            expr = parts[0]
            for p in parts[1:]:
                expr = expr + p
            if rng.random() < 0.4:
                expr = expr * uconst(rng.choice([2, 3]))
            return UOp.sink(expr)
        except Exception:
            continue
    raise AssertionError("split generator failed 20x")


def rand_threefry_graph(rng):
    """Graphs exercising the THREEFRY decomposition in `pm_decomp`."""
    for _ in range(20):
        try:
            x = Variable("x", 0, 1000, dtypes.uint64)
            key = Variable("key", 0, 1000, dtypes.uint64)
            g = UOp(Ops.THREEFRY, dtypes.uint64, (x, key))
            if rng.random() < 0.5:
                g = (g & uconst(0xFFFF, dtypes.uint64)) + uconst(rng.randint(0, 3), dtypes.uint64)
            return g
        except Exception:
            continue
    raise AssertionError("threefry generator failed 20x")


class TestRewriteDifferential(unittest.TestCase):
    ORACLE = None

    @classmethod
    def setUpClass(cls):
        if not _ORACLE.exists():
            raise unittest.SkipTest("oracle_rewrite not built")
        cls.ORACLE = Oracle()

    def _diff(self, pm_name, pm_py, graphs, ctx=None):
        n_ok, n_skip, n_fail = 0, 0, 0
        for gi, g in enumerate(graphs):
            py_err = None
            py_out = None
            try:
                py_out = graph_rewrite(g, pm_py, ctx=ctx)
            except Exception as e:
                py_err = f"{type(e).__name__}: {e}"
            rust_err = None
            rust_out = None
            try:
                rust_out = self.ORACLE.rewrite(pm_name, g)
            except Exception as e:
                rust_err = f"{type(e).__name__}: {e}"
            if py_err is not None and rust_err is not None:
                n_skip += 1
                continue
            if py_err is not None or rust_err is not None:
                n_fail += 1
                self.fail(f"{pm_name} graph#{gi}: error mismatch py={py_err} rust={rust_err}")
                continue
            if py_out.key != rust_out[0].key:
                n_fail += 1
                self.fail(
                    f"{pm_name} graph#{gi}: key mismatch\n py: {py_out.pyrender().splitlines()[-1]}\n"
                    f" rust: {rust_out[0].pyrender().splitlines()[-1]}"
                )
            n_ok += 1
        print(f" [{pm_name}] ok={n_ok} skip={n_skip} fail={n_fail}")

    def test_symbolic_simple(self):
        rng = random.Random(42)
        graphs = [rand_symbolic_graph(rng) for _ in range(300)]
        self._diff("symbolic_simple", symbolic_simple, graphs)

    def test_commutative(self):
        rng = random.Random(43)
        graphs = [rand_symbolic_graph(rng) for _ in range(200)]
        self._diff("commutative", commutative, graphs)

    def test_mop_cleanup(self):
        rng = random.Random(44)
        graphs = [rand_movement_graph(rng) for _ in range(200)]
        self._diff("mop_cleanup", mop_cleanup, graphs)

    def test_div_and_mod(self):
        rng = random.Random(45)
        graphs = [rand_symbolic_graph(rng) for _ in range(200)]
        self._diff("div_and_mod_symbolic", div_and_mod_symbolic, graphs)

    def test_symbolic(self):
        rng = random.Random(46)
        graphs = [rand_symbolic_graph(rng) for _ in range(300)]
        self._diff("symbolic", symbolic, graphs)

    def test_pm_load_collapse(self):
        from tinygrad.codegen.simplify import pm_load_collapse
        rng = random.Random(47)
        graphs = [rand_index_graph(rng) for _ in range(300)]
        self._diff("pm_load_collapse", pm_load_collapse, graphs)

    def test_pm_reduce_load_collapse(self):
        from tinygrad.codegen.simplify import pm_reduce_load_collapse
        rng = random.Random(48)
        graphs = [rand_index_graph(rng) for _ in range(300)]
        self._diff("pm_reduce_load_collapse", pm_reduce_load_collapse, graphs)

    def test_pm_split_ranges(self):
        from tinygrad.codegen.simplify import pm_split_ranges
        rng = random.Random(49)
        graphs = [rand_split_graph(rng) for _ in range(300)]
        self._diff("pm_split_ranges", pm_split_ranges, graphs, ctx={})

    def test_pm_decomp(self):
        from tinygrad.codegen.decomp.op import get_simplifying_rewrite_patterns
        from tinygrad.runtime.ops_python import PythonRenderer
        from tinygrad.helpers import Target
        ren = PythonRenderer.__new__(PythonRenderer)
        PythonRenderer.__init__(ren, Target.parse("cpu"))
        pm = symbolic_simple + get_simplifying_rewrite_patterns(tuple(ren.code_for_op.keys()))
        rng = random.Random(50)
        graphs = [rand_symbolic_graph(rng) for _ in range(300)]
        graphs += [rand_threefry_graph(rng) for _ in range(50)]
        self._diff("pm_decomp", pm, graphs)


if __name__ == "__main__":
    unittest.main()
