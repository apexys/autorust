#!/usr/bin/env python3
"""Rule-parity diff: this Rust port's codegen PatternMatchers vs tinygrad.

Builds a canonical, order-sensitive signature for every rule in every codegen
pipeline `PatternMatcher` on both sides and reports missing/extra rules.

Usage:
    python3 py/pm_inventory_diff.py            # summary table
    python3 py/pm_inventory_diff.py -v PM      # per-rule diff for one PM

The Rust side is `src/bin/pm_inventory.rs` (run via cargo). The tinygrad side is
imported directly, so set TG_ROOT if tinygrad isn't at ../tinygrad.
"""
import os, re, sys, subprocess, itertools
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TG = os.environ.get("TG_ROOT", os.path.join(ROOT, "..", "tinygrad"))
sys.path.insert(0, TG)

import tinygrad.uop.symbolic as SY
import tinygrad.uop.movement as MV
import tinygrad.uop.weak as WK
import tinygrad.codegen as C
import tinygrad.codegen.gpudims as GD
import tinygrad.codegen.late.coalesce as CO
import tinygrad.codegen.late.gater as GA
import tinygrad.codegen.late.linearizer as LN
import tinygrad.codegen.simplify as SI
import tinygrad.schedule.rangeify as RF


def sig(p):
    # `UPat.any(...)` / `or_casted` / `or_after`: normalize to Rust `SrcPat::Any`.
    if getattr(p, "is_any", False):
        ops, name, req, strict = "ANY", "", 0, False
    else:
        ops = "ANY" if p.op is None else ",".join(sorted("Ops." + o.name for o in p.op))
        name = p.name or ""
        req, strict = p.required_len, bool(p.strict_length)
    s = "%s|%s|req=%d|strict=%s" % (ops, name, req, str(strict).lower())
    kids = []
    if p.src is not None:
        first = p.src[0]
        if isinstance(first, itertools.repeat):
            kids = [sig(next(iter(first))) + "*"]
        else:
            kids = [sig(x) for x in first]
    if kids:
        s += "<" + ",".join(kids) + ">"
    return s


def tg_inventory():
    def pm_decomp():
        from tinygrad.codegen.decomp.op import get_simplifying_rewrite_patterns
        from tinygrad.runtime.ops_python import PythonRenderer
        from tinygrad.helpers import Target
        ren = PythonRenderer.__new__(PythonRenderer)
        PythonRenderer.__init__(ren, Target.parse("cpu"))
        return SY.symbolic_simple + get_simplifying_rewrite_patterns(tuple(ren.code_for_op.keys()))
    pms = [
        ("symbolic_simple", SY.symbolic_simple), ("symbolic", SY.symbolic), ("sym", SY.sym),
        ("pm_simplify_valid", SY.pm_simplify_valid), ("pm_commutative", getattr(SY, "commutative", None)),
        ("mop_cleanup", MV.mop_cleanup), ("devectorizer2", C.devectorizer2), ("ew_devectorizer", C.ew_devectorizer),
        ("expander2", C.expander2), ("pm_expand_broadcast", C.pm_expand_broadcast), ("pm_add_loads", C.pm_add_loads),
        ("pm_reduce_local", C.pm_reduce_local), ("pm_add_local_buffers", C.pm_add_local_buffers),
        ("pm_add_gpudims", GD.pm_add_gpudims), ("indexing_simplify", CO.indexing_simplify),
        ("pm_simplify_add_image", CO.pm_simplify_add_image), ("pm_decomp", pm_decomp()),
        ("pm_dtype_decomps", C.pm_dtype_decomps),
        ("pm_move_gates_from_index", GA.pm_move_gates_from_index), ("pm_cast_float_alu", C.pm_cast_float_alu),
        ("pm_commit_weak", WK.pm_commit_weak), ("pm_cast_weak", WK.pm_cast_weak),
        ("pm_lower_index_dtype", WK.pm_lower_index_dtype), ("pm_casted_consts", C.pm_casted_consts),
        ("pm_number_params", C.pm_number_params), ("pm_implicit_barriers", C.pm_implicit_barriers),
        ("pm_split_ends", LN.pm_split_ends), ("pm_move_where_on_load", SY.pm_move_where_on_load),
        ("pm_flatten_range", SI.pm_flatten_range), ("pm_reduce_unparented", SI.pm_reduce_unparented),
        ("pm_split_ranges", SI.pm_split_ranges),
        ("pm_reduce_load_collapse", SI.pm_reduce_load_collapse), ("pm_load_collapse", SI.pm_load_collapse),
        ("pm_mops", RF.pm_mops),
    ]
    return {n: [sig(p) for p, _ in pm.patterns] for n, pm in pms if pm is not None}


def rust_inventory():
    out = subprocess.run(["cargo", "run", "--quiet", "--bin", "pm_inventory"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    pms, cur = {}, None
    for line in out.splitlines():
        m = re.match(r"^PM (\S+) (\S+)$", line)
        if m:
            cur = m.group(1); pms[cur] = []; continue
        m = re.match(r"^R (\d+) (.*)$", line)
        if m and cur:
            pms[cur].append(m.group(2))
    return pms


def main():
    tg, rs = tg_inventory(), rust_inventory()
    verbose = "-v" in sys.argv
    target = sys.argv[sys.argv.index("-v") + 1] if verbose and sys.argv.index("-v") + 1 < len(sys.argv) else None
    names = list(dict.fromkeys(list(tg) + list(rs)))
    if target:
        r, t = Counter(rs.get(target, [])), Counter(tg.get(target, []))
        for s in (t - r).elements(): print("TG-ONLY:", s)
        for s in (r - t).elements(): print("RS-ONLY:", s)
        return
    print(f"{'PM':<28}{'rust':>6}{'tg':>6}{'miss':>6}{'extra':>7}")
    for n in names:
        if n == "pm_sym_extra":
            continue
        r, t = rs.get(n, []), tg.get(n, [])
        miss, extra = sum((Counter(t) - Counter(r)).values()), sum((Counter(r) - Counter(t)).values())
        flag = "" if miss == 0 and extra == 0 else "  <--"
        print(f"{n:<28}{len(r):>6}{len(t):>6}{miss:>6}{extra:>7}{flag}")


if __name__ == "__main__":
    main()
