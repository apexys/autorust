//! Port of the pattern matcher (ops.py:1317-1479): `UPat` matching semantics,
//! `PatternMatcher` (interpret path — the Python bytecode compiler is a pure
//! optimization), and the rewrite driver context values.
//!
//! Semantics faithfully mirrored:
//! - `match` returns a LIST of binding stores; the rule runs on the first
//!   store whose function returns non-None (ops.py:1434-1446).
//! - a rule result equal to the rewritten node counts as "no match" and the
//!   next pattern is tried (ops.py:1478).
//! - src patterns: exact tuples, permutations of lists, and `repeat` (one
//!   pattern over all srcs); `zip` semantics (extra srcs ignored unless
//!   strict_length).
//! - `early_reject` = ops of single-op src patterns, precomputed (ops.py:1344-1347).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::arg::Arg;
use super::ctx::UOpCtx;
use super::graph::UopId;
use super::ops::{Ops, group};
use super::sint::SVal;
use crate::dtype::{ConstVal, DTypeId, is_bool};
use ConstVal::*;

#[derive(Clone, PartialEq)]
pub enum SrcPat {
    None,
    /// exact tuple (strict when allow_any_len is false)
    Tup(Vec<UPat>),
    /// permutations of a list (precomputed)
    List(Vec<Vec<UPat>>),
    /// one pattern repeated over all srcs
    Repeat(Box<UPat>),
    /// `is_any`: try each alternative against the same node
    Any(Vec<UPat>),
}

#[derive(Clone, PartialEq)]
pub struct UPat {
    pub op: Option<Vec<Ops>>,
    pub dtype: Option<Vec<DTypeId>>,
    pub tag: Option<Vec<Arg>>,
    pub arg: Option<Arg>,
    pub name: Option<&'static str>,
    pub src: SrcPat,
    pub strict_length: bool,
    pub required_len: usize,
    pub early_reject: Vec<Ops>,
}

pub type Store = HashMap<&'static str, UopId>;

// `add`/`sub`/`mul`/`shl`/`shr` build *patterns* for those Ops (named after
// tinygrad's `UPat` helpers), not arithmetic on `UPat`; they take
// `impl Into<UPatOrConst>`, which a std operator could not cover.
#[allow(clippy::should_implement_trait)]
impl UPat {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        op: Option<Vec<Ops>>,
        dtype: Option<Vec<DTypeId>>,
        src: SrcPat,
        arg: Option<Arg>,
        name: Option<&'static str>,
        allow_any_len: bool,
        tag: Option<Vec<Arg>>,
    ) -> UPat {
        let (strict_length, required_len) = match &src {
            SrcPat::Tup(v) => (!allow_any_len, v.len()),
            SrcPat::List(ps) => (!allow_any_len, ps[0].len()),
            SrcPat::Repeat(_) => (false, 0),
            SrcPat::Any(_) => (false, 0),
            SrcPat::None => (false, 0),
        };
        // early reject: ops of single-op direct src patterns
        let direct: Vec<&UPat> = match &src {
            SrcPat::Repeat(p) => vec![p],
            SrcPat::None => vec![],
            SrcPat::Tup(v) => v.iter().collect(),
            SrcPat::List(ps) => ps[0].iter().collect(),
            SrcPat::Any(_) => vec![],
        };
        let mut early_reject = Vec::new();
        for p in direct {
            if let Some(ops) = &p.op
                && ops.len() == 1
                && !early_reject.contains(&ops[0])
            {
                early_reject.push(ops[0]);
            }
        }
        UPat {
            op,
            dtype,
            tag,
            arg,
            name,
            src,
            strict_length,
            required_len,
            early_reject,
        }
    }

    pub fn op_var(ops: Vec<Ops>) -> UPat {
        UPat::new(Some(ops), None, SrcPat::None, None, None, false, None)
    }

    pub fn ops1(op: Ops) -> UPat {
        UPat::new(Some(vec![op]), None, SrcPat::None, None, None, false, None)
    }

    /// `UPat.var(name, dtype)`
    pub fn var(name: Option<&'static str>, dtype: Option<Vec<DTypeId>>) -> UPat {
        UPat::new(None, dtype, SrcPat::None, None, name, false, None)
    }

    /// `UPat.cvar(name, dtype, arg)`
    pub fn cvar(name: Option<&'static str>, dtype: Option<Vec<DTypeId>>, arg: Option<Arg>) -> UPat {
        UPat::new(
            Some(vec![Ops::Const]),
            dtype,
            SrcPat::None,
            arg,
            name,
            false,
            None,
        )
    }

    /// `UPat.const(b, dtype)`
    pub fn const_pat(v: super::super::dtype::ConstVal, dtype: Option<Vec<DTypeId>>) -> UPat {
        UPat::cvar(None, dtype, Some(super::arg::Arg::Const(v)))
    }

    pub fn any(srcs: Vec<UPat>) -> UPat {
        UPat::new(None, None, SrcPat::Any(srcs), None, None, false, None)
    }

    pub fn named(&self, name: &'static str) -> UPat {
        let mut p = self.clone();
        p.name = Some(name);
        p
    }

    pub fn with_arg(&self, arg: Option<Arg>) -> UPat {
        let mut p = self.clone();
        p.arg = arg;
        p
    }

    pub fn with_dtype(&self, dtype: Option<Vec<DTypeId>>) -> UPat {
        let mut p = self.clone();
        p.dtype = dtype;
        p
    }

    pub fn with_tag(&self, tag: Option<Vec<Arg>>) -> UPat {
        let mut p = self.clone();
        p.tag = tag;
        p
    }

    /// `p.f(op, ...)` — child pattern.
    pub fn f(&self, op: Ops) -> UPat {
        UPat::ops1(op).with_src_tup(vec![self.clone()])
    }

    /// match any number of srcs (sets required_len to the fixed prefix only)
    pub fn allow_all_srcs(&self) -> UPat {
        let mut p = self.clone();
        p.strict_length = false;
        p.required_len = match &p.src {
            SrcPat::Tup(v) => v.len(),
            SrcPat::List(_) => 1,
            _ => 0,
        };
        p.recompute_early_reject();
        p
    }

    /// all srcs are RESHAPE(var) patterns (for MSELECT/MSTACK reshape hoisting)
    pub fn with_src_all_reshape(&self) -> UPat {
        let srcs = self.srcs_clone();
        let new_srcs: Vec<UPat> = srcs
            .into_iter()
            .map(|s| {
                UPat::ops1(Ops::Reshape)
                    .with_src_tup(vec![s])
                    .allow_all_srcs()
            })
            .collect();
        let mut p = self.clone();
        p.src = SrcPat::Tup(new_srcs);
        p.strict_length = false;
        p.recompute_early_reject();
        p
    }

    fn srcs_clone(&self) -> Vec<UPat> {
        match &self.src {
            SrcPat::Tup(v) => v.clone(),
            _ => vec![],
        }
    }

    pub fn with_src_tup(&self, srcs: Vec<UPat>) -> UPat {
        let mut p = self.clone();
        p.src = SrcPat::Tup(srcs);
        p.strict_length = true;
        p.required_len = {
            let SrcPat::Tup(v) = &p.src else {
                unreachable!()
            };
            v.len()
        };
        p.recompute_early_reject();
        p
    }

    pub(crate) fn recompute_early_reject_rep(&mut self, p: &UPat) {
        let mut er = Vec::new();
        if let Some(ops) = &p.op
            && ops.len() == 1
        {
            er.push(ops[0]);
        }
        self.early_reject = er;
    }

    pub fn recompute_early_reject(&mut self) {
        let mut er = Vec::new();
        if let SrcPat::Tup(v) = &self.src {
            for p in v {
                if let Some(ops) = &p.op
                    && ops.len() == 1
                    && !er.contains(&ops[0])
                {
                    er.push(ops[0]);
                }
            }
        }
        self.early_reject = er;
    }

    // ---- compositional builders mirroring the mixin operator expansions ----

    fn ufix(x: UPatOrConst) -> UPat {
        match x {
            UPatOrConst::Pat(p) => p,
            UPatOrConst::Val(v) => UPat::cvar(None, None, Some(Arg::Const(v))),
        }
    }

    pub fn alu(self, op: Ops, src: Vec<UPat>) -> UPat {
        let mut asrc = vec![self];
        asrc.extend(src);
        let dt = match op {
            Ops::Cmplt | Ops::Cmpne => Some(vec![crate::dtype::BOOL]),
            _ => asrc.last().and_then(|p| p.dtype.clone()),
        };
        let src = if group::in_(group::COMMUTATIVE, op) {
            SrcPat::List(permutations(asrc))
        } else {
            SrcPat::Tup(asrc)
        };
        UPat::new(Some(vec![op]), dt, src, None, None, false, None)
    }

    fn _binop(self, op: Ops, x: UPatOrConst, reverse: bool) -> UPat {
        let y = Self::ufix(x);
        let (a, b) = if reverse { (y, self) } else { (self, y) };
        a.alu(op, vec![b])
    }

    pub fn mul(self, x: impl Into<UPatOrConst>) -> UPat {
        self._binop(Ops::Mul, x.into(), false)
    }

    pub fn add(self, x: impl Into<UPatOrConst>) -> UPat {
        self._binop(Ops::Add, x.into(), false)
    }

    pub fn sub(self, x: impl Into<UPatOrConst>) -> UPat {
        self._binop(Ops::Sub, x.into(), false)
    }

    pub fn floordiv(self, x: impl Into<UPatOrConst>) -> UPat {
        self._binop(Ops::Floordiv, x.into(), false)
    }

    pub fn truediv(self, x: impl Into<UPatOrConst>) -> UPat {
        // elementwise div: a * RECIPROCAL(b) for non-int pairs, CDIV/FLOORDIV otherwise;
        // the M2 rules only use `/` on const patterns where the composite matches
        // Python's expansion (MUL + RECIPROCAL).
        let y = Self::ufix(x.into());
        self.mul(y.alu(Ops::Reciprocal, vec![]))
    }

    pub fn mod_(self, x: impl Into<UPatOrConst>) -> UPat {
        self._binop(Ops::Floormod, x.into(), false)
    }

    pub fn and_(self, x: impl Into<UPatOrConst>) -> UPat {
        self._binop(Ops::And, x.into(), false)
    }

    pub fn or_(self, x: impl Into<UPatOrConst>) -> UPat {
        self._binop(Ops::Or, x.into(), false)
    }

    pub fn xor(self, x: impl Into<UPatOrConst>) -> UPat {
        self._binop(Ops::Xor, x.into(), false)
    }

    pub fn shl(self, x: impl Into<UPatOrConst>) -> UPat {
        self._binop(Ops::Shl, x.into(), false)
    }

    pub fn shr(self, x: impl Into<UPatOrConst>) -> UPat {
        self._binop(Ops::Shr, x.into(), false)
    }

    pub fn lt(self, x: impl Into<UPatOrConst>) -> UPat {
        self._binop(Ops::Cmplt, x.into(), false)
    }

    pub fn gt(self, x: impl Into<UPatOrConst>) -> UPat {
        self._binop(Ops::Cmplt, x.into(), true)
    }

    pub fn ne(self, x: impl Into<UPatOrConst>) -> UPat {
        self._binop(Ops::Cmpne, x.into(), false)
    }

    pub fn maximum(self, x: impl Into<UPatOrConst>) -> UPat {
        self._binop(Ops::Max, x.into(), false)
    }

    pub fn where_(self, x: impl Into<UPatOrConst>, y: impl Into<UPatOrConst>) -> UPat {
        let (x, y) = (Self::ufix(x.into()), Self::ufix(y.into()));
        self.alu(Ops::Where, vec![x, y])
    }

    /// UPat-specific `cast` (ops.py:1380-1382).
    pub fn cast(self, dtype: Option<DTypeId>) -> UPat {
        if let Some(d) = dtype {
            if self.dtype == Some(vec![d]) {
                return self;
            }
            UPat::new(
                Some(vec![Ops::Cast]),
                Some(vec![d]),
                SrcPat::Tup(vec![self]),
                None,
                None,
                false,
                None,
            )
        } else {
            UPat::new(
                Some(vec![Ops::Cast]),
                None,
                SrcPat::Tup(vec![self]),
                None,
                None,
                false,
                None,
            )
        }
    }

    /// cast with a dtype SET (Python `cast(dtypes.ints + ...)`)
    pub fn cast_dts(self, dts: Vec<DTypeId>) -> UPat {
        UPat::new(
            Some(vec![Ops::Cast]),
            Some(dts),
            SrcPat::Tup(vec![self]),
            None,
            None,
            false,
            None,
        )
    }

    pub fn bitcast(self, dtype: Option<DTypeId>) -> UPat {
        UPat::new(
            Some(vec![Ops::Bitcast]),
            dtype.map(|d| vec![d]),
            SrcPat::Tup(vec![self]),
            None,
            None,
            false,
            None,
        )
    }

    pub fn logical_not(self) -> UPat {
        // mixin: self.cast(bool).ne(True)
        self.cast(Some(crate::dtype::BOOL))
            .ne(UPatOrConst::Val(ConstVal::Bool(true)))
    }

    pub fn reciprocal(self) -> UPat {
        self.alu(Ops::Reciprocal, vec![])
    }

    pub fn trunc(self) -> UPat {
        self.alu(Ops::Trunc, vec![])
    }

    pub fn reduce(self, src: Vec<UPat>, arg: Option<Arg>, allow_any_len: bool) -> UPat {
        let dt = self.dtype.clone();
        let mut asrc = vec![self];
        asrc.extend(src);
        let mut p = UPat::new(
            Some(vec![Ops::Reduce]),
            dt,
            SrcPat::Tup(asrc),
            arg,
            None,
            allow_any_len,
            None,
        );
        p.recompute_early_reject();
        p
    }

    pub fn load(self, src: Vec<UPat>, allow_any_len: bool) -> UPat {
        let dt = self.dtype.clone();
        let mut asrc = vec![self];
        asrc.extend(src);
        let mut p = UPat::new(
            Some(vec![Ops::Load]),
            dt,
            SrcPat::Tup(asrc),
            None,
            None,
            allow_any_len,
            None,
        );
        p.recompute_early_reject();
        p
    }

    pub fn store(self, src: Vec<UPat>) -> UPat {
        let dt = self.dtype.clone();
        let mut asrc = vec![self];
        asrc.extend(src);
        let mut p = UPat::new(
            Some(vec![Ops::Store]),
            dt,
            SrcPat::Tup(asrc),
            None,
            None,
            false,
            None,
        );
        p.recompute_early_reject();
        p
    }

    pub fn index(self, srcs: Vec<UPat>) -> UPat {
        let dt = self.dtype.clone();
        let mut asrc = vec![self];
        asrc.extend(srcs);
        let mut p = UPat::new(
            Some(vec![Ops::Index]),
            dt,
            SrcPat::Tup(asrc),
            None,
            None,
            false,
            None,
        );
        p.recompute_early_reject();
        p
    }

    pub fn after(self, src: Vec<UPat>, allow_any_len: bool) -> UPat {
        let dt = self.dtype.clone();
        let mut asrc = vec![self];
        asrc.extend(src);
        let mut p = UPat::new(
            Some(vec![Ops::After]),
            dt,
            SrcPat::Tup(asrc),
            None,
            None,
            allow_any_len,
            None,
        );
        p.recompute_early_reject();
        p
    }

    pub fn end(self, src: Vec<UPat>) -> UPat {
        let dt = self.dtype.clone();
        let mut asrc = vec![self];
        asrc.extend(src);
        let mut p = UPat::new(
            Some(vec![Ops::End]),
            dt,
            SrcPat::Tup(asrc),
            None,
            None,
            false,
            None,
        );
        p.recompute_early_reject();
        p
    }

    /// `UPat(Ops.STACK, self.match_dtype, src=self, ...)` — repeat src.
    pub fn broadcast(self, allow_any_len: bool) -> UPat {
        let dt = self.dtype.clone();
        let src = SrcPat::Repeat(Box::new(self));
        UPat::new(
            Some(vec![Ops::Stack]),
            dt,
            src,
            None,
            None,
            allow_any_len,
            None,
        )
    }

    /// `or_casted(name)` (ops.py:1360).
    pub fn or_casted(self, name: Option<&'static str>) -> UPat {
        let a = match name {
            None => self.clone(),
            Some(n) => self.named(n),
        };
        let b = UPat::new(
            Some(vec![Ops::Cast]),
            None,
            SrcPat::Tup(vec![self.clone()]),
            None,
            name,
            false,
            None,
        );
        UPat::any(vec![a, b])
    }

    /// `or_after(name)`.
    pub fn or_after(self, name: Option<&'static str>) -> UPat {
        let a = match name {
            None => self.clone(),
            Some(n) => self.named(n),
        };
        let b = {
            UPat::new(
                Some(vec![Ops::After]),
                None,
                SrcPat::Tup(vec![self.clone()]),
                None,
                name,
                true,
                None,
            )
        };
        UPat::any(vec![a, b])
    }

    // ---------- matching ----------

    pub fn match_uop(&self, ctx: &UOpCtx, uop: UopId, store: Store, out: &mut Vec<Store>) {
        if let SrcPat::Any(alts) = &self.src {
            for alt in alts {
                alt.match_uop(ctx, uop.clone(), store.clone(), out);
            }
            return;
        }
        // op check
        if let Some(ops) = &self.op
            && !ops.contains(&ctx.op_of(&uop))
        {
            return;
        }
        // name binding (identity)
        let mut store = store;
        if let Some(n) = self.name {
            match store.get(n) {
                Some(prev) if *prev != uop => return,
                Some(_) => {}
                None => {
                    store.insert(n, uop.clone());
                }
            }
        }
        // dtype
        if let Some(dts) = &self.dtype
            && !dts.contains(&ctx.dt_of(&uop))
        {
            return;
        }
        // arg (Python `!=` — bools compare as ints)
        if let Some(pa) = &self.arg
            && !arg_matches(pa, ctx.arg_of(&uop))
        {
            return;
        }
        // tag
        if let Some(tags) = &self.tag
            && !tags.iter().any(|t| t == ctx.tag_of(&uop))
        {
            return;
        }
        let srcs = ctx.srcs(&uop);
        if srcs.len() < self.required_len || (self.strict_length && srcs.len() != self.required_len)
        {
            return;
        }
        match &self.src {
            SrcPat::None => out.push(store),
            SrcPat::Tup(vp) => self.match_zip(ctx, srcs, vp.iter(), store, out),
            SrcPat::List(perms) => {
                for vp in perms {
                    self.match_zip(ctx, srcs, vp.iter(), store.clone(), out);
                }
            }
            SrcPat::Repeat(p) => {
                // zip with an infinite repeat: all srcs against the same pattern
                let mut stores = vec![store];
                for uu in srcs {
                    let mut next = Vec::new();
                    for s in stores {
                        let mut o = Vec::new();
                        p.match_uop(ctx, uu.clone(), s, &mut o);
                        next.extend(o);
                    }
                    stores = next;
                    if stores.is_empty() {
                        return;
                    }
                }
                out.extend(stores);
            }
            SrcPat::Any(_) => unreachable!(),
        }
    }

    fn match_zip<'p>(
        &self,
        ctx: &UOpCtx,
        srcs: &[UopId],
        mut vp: impl Iterator<Item = &'p UPat>,
        store: Store,
        out: &mut Vec<Store>,
    ) {
        let mut stores = vec![store];
        for uu in srcs {
            let Some(vv) = vp.next() else { break };
            let mut new_stores = Vec::new();
            for s in stores {
                let mut o = Vec::new();
                vv.match_uop(ctx, uu.clone(), s, &mut o);
                new_stores.extend(o);
            }
            stores = new_stores;
            if stores.is_empty() {
                return;
            }
        }
        out.extend(stores);
    }
}

/// Memo table used by `pm_long_decomp` while splitting long index chains.
pub type L2iMemo =
    std::cell::RefCell<HashMap<(Ops, crate::dtype::DTypeId, Vec<UopId>), Vec<UopId>>>;

#[derive(Clone)]
#[allow(clippy::large_enum_variant)] // patterns are built once; boxing would complicate every match
pub enum UPatOrConst {
    Pat(UPat),
    Val(crate::dtype::ConstVal),
}

impl From<UPat> for UPatOrConst {
    fn from(p: UPat) -> Self {
        UPatOrConst::Pat(p)
    }
}

impl From<crate::dtype::ConstVal> for UPatOrConst {
    fn from(v: crate::dtype::ConstVal) -> Self {
        UPatOrConst::Val(v)
    }
}

impl From<i64> for UPatOrConst {
    fn from(v: i64) -> Self {
        UPatOrConst::Val(crate::dtype::ConstVal::Int(v as i128))
    }
}

impl From<bool> for UPatOrConst {
    fn from(v: bool) -> Self {
        UPatOrConst::Val(crate::dtype::ConstVal::Bool(v))
    }
}

impl From<f64> for UPatOrConst {
    fn from(v: f64) -> Self {
        UPatOrConst::Val(crate::dtype::ConstVal::Float(v))
    }
}

/// Pattern-arg vs node-arg equality with Python's loose numeric equality
/// (bools are ints), float identity by bits, and `Const` unboxing.
pub fn arg_matches(pat: &Arg, node: &Arg) -> bool {
    match (pat, node) {
        (Arg::Const(c), other) => const_arg_matches(c, other),
        (other, Arg::Const(c)) => const_arg_matches(c, other),
        (
            Arg::Seq {
                tuple: t1,
                items: i1,
            },
            Arg::Seq {
                tuple: t2,
                items: i2,
            },
        ) => t1 == t2 && i1.len() == i2.len() && i1.iter().zip(i2).all(|(a, b)| arg_matches(a, b)),
        (a, b) => a == b,
    }
}

fn const_arg_matches(c: &crate::dtype::ConstVal, node: &Arg) -> bool {
    use crate::dtype::ConstVal;
    let node_c = match node {
        Arg::Const(c2) => c2.clone(),
        Arg::Int(i) => ConstVal::Int(*i as i128),
        Arg::UInt(u) => ConstVal::Int(*u as i128),
        Arg::Bool(b) => ConstVal::Bool(*b),
        Arg::Float(f) => ConstVal::Float(*f),
        Arg::NonFinite(super::arg::NonFinite::Nan) => ConstVal::Float(f64::NAN),
        Arg::NonFinite(super::arg::NonFinite::PosInf) => ConstVal::Float(f64::INFINITY),
        Arg::NonFinite(super::arg::NonFinite::NegInf) => ConstVal::Float(f64::NEG_INFINITY),
        _ => return false,
    };
    py_const_eq(c, &node_c)
}

/// Python `==` on ConstType values.
pub fn py_const_eq(a: &ConstVal, b: &ConstVal) -> bool {
    match (a, b) {
        // bool == int numerically in Python
        (Bool(x), Int(y)) | (Int(y), Bool(x)) => (*x as i128) == *y,
        (Bool(x), Bool(y)) => x == y,
        (Int(x), Int(y)) => x == y,
        (Big(x), Big(y)) => x == y,
        (Big(x), Int(y)) | (Int(y), Big(x)) => x.as_i64().map(|i| i as i128) == Some(*y),
        (Float(x), Float(y)) => x.to_bits() == y.to_bits(),
        (Float(x), Int(y)) | (Int(y), Float(x)) => *x == *y as f64,
        (Float(x), Bool(y)) | (Bool(y), Float(x)) => *x == (*y as i128 as f64),
        (Invalid, Invalid) => true,
        _ => false,
    }
}

pub fn permutations<T: Clone + PartialEq>(v: Vec<T>) -> Vec<Vec<T>> {
    fn helper<T: Clone>(
        items: &[T],
        cur: &mut Vec<T>,
        used: &mut Vec<bool>,
        out: &mut Vec<Vec<T>>,
    ) {
        if cur.len() == items.len() {
            out.push(cur.clone());
            return;
        }
        for i in 0..items.len() {
            if used[i] {
                continue;
            }
            used[i] = true;
            cur.push(items[i].clone());
            helper(items, cur, used, out);
            cur.pop();
            used[i] = false;
        }
    }
    // Python: permutations of a list with all-equal elements collapse to one
    let mut all_same = true;
    for w in v.windows(2) {
        if w[0] != w[1] {
            all_same = false;
            break;
        }
    }
    if all_same && !v.is_empty() {
        return vec![v];
    }
    let mut out = Vec::new();
    helper(&v, &mut Vec::new(), &mut vec![false; v.len()], &mut out);
    out
}

// ---------- PatternMatcher ----------

pub type RuleFn = Arc<dyn Fn(&mut UOpCtx, &Store, &CtxVal) -> Option<UopId> + Send + Sync>;

/// Lazy `PatternMatcher` slot for a `static`.
///
/// `PatternMatcher` is `Send + Sync` (patterns are owned data and rules are
/// `Arc<dyn Fn + Send + Sync>` that capture no arena state), so a `OnceLock`
/// hands out a process-wide `&'static` with thread-safe initialization — no
/// leak, no `unsafe`, and no per-thread copies. Rules that need arena-specific
/// data receive it through [`CtxVal`] rather than capturing it.
pub struct PmSlot(std::sync::OnceLock<PatternMatcher>);

impl Default for PmSlot {
    fn default() -> Self {
        Self::new()
    }
}

impl PmSlot {
    pub const fn new() -> Self {
        PmSlot(std::sync::OnceLock::new())
    }
    pub fn get_or_init(
        &'static self,
        build: impl FnOnce() -> PatternMatcher,
    ) -> &'static PatternMatcher {
        self.0.get_or_init(|| build().interned())
    }
}

pub struct Rule {
    pub pat: UPat,
    pub f: RuleFn,
    pub early_reject: Vec<Ops>,
}

/// Threading context for rule functions (M2: substitution dicts).
#[derive(Clone, Default)]
pub enum CtxVal {
    #[default]
    None,
    Subst(HashMap<UopId, UopId>),
    /// Scheduler IndexingContext (run_rangeify rules).
    Idx(std::rc::Rc<std::cell::RefCell<crate::schedule::indexing::IndexingContext>>),
    /// Device string for pm_fix_deviceless.
    Dev(Option<crate::uop::arg::DeviceSpec>),
    /// Shared counter (itertools.count contexts).
    Cnt(std::rc::Rc<std::cell::Cell<i64>>),
    /// LocalAddBufferContext for kernel splitting.
    La(std::rc::Rc<std::cell::RefCell<LocalAddBufferContext>>),
    /// AllocCtx for callify.
    Al(std::rc::Rc<std::cell::RefCell<crate::tensor::callify::AllocCtx>>),
    /// (cache, args) for pm_post_sched_cache.
    Ca(std::rc::Rc<std::cell::RefCell<crate::schedule::sched::CallCtx>>),
    /// Mutable mark dict context (e.g. `pm_split_ranges`).
    Dict(std::rc::Rc<std::cell::RefCell<HashMap<UopId, UopId>>>),
    /// Read-only (from, to) dtype pair for `pm_float_decomp`.
    Pair(crate::dtype::DTypeId, crate::dtype::DTypeId),
    /// Dtype set context for `pm_dtype_decomps`.
    Decomp(std::rc::Rc<std::cell::RefCell<crate::codegen::decomp_dtype::DtypeDecompCtx>>),
    /// Memo dict for `pm_long_decomp` `split_l2i`.
    L2i(std::rc::Rc<L2iMemo>),
}

/// `LocalAddBufferContext` (rangeify.py:469).
#[derive(Default)]
pub struct LocalAddBufferContext {
    pub dg: i64,
    /// insertion-ordered buf -> replacement map
    pub map_keys: Vec<UopId>,
    pub map_vals: Vec<UopId>,
    pub range: i64,
    pub opts: Option<Vec<Arg>>,
}

impl LocalAddBufferContext {
    pub fn map_get(&self, k: UopId) -> Option<UopId> {
        self.map_keys
            .iter()
            .position(|x| *x == k)
            .map(|i| (self.map_vals[i]).clone())
    }
    pub fn map_insert(&mut self, k: UopId, v: UopId) {
        if self.map_get(k.clone()).is_none() {
            self.map_keys.push(k);
            self.map_vals.push(v);
        }
    }
}

impl CtxVal {
    pub fn get_subst(&self, u: UopId) -> Option<UopId> {
        match self {
            CtxVal::Subst(m) => m.get(&u).cloned(),
            _ => None,
        }
    }
}

/// rule helper: closure over (IndexingContext, UOpCtx, store)
pub fn r_idx<F>(f: F) -> RuleFn
where
    F: Fn(&mut crate::schedule::indexing::IndexingContext, &mut UOpCtx, &Store) -> Option<UopId>
        + Send
        + Sync
        + 'static,
{
    Arc::new(move |ctx, st, ctxv| match ctxv {
        CtxVal::Idx(ictx) => f(&mut ictx.borrow_mut(), ctx, st),
        _ => panic!("TGERR: rule requires IndexingContext"),
    })
}

/// rule helper: closure over (UOpCtx, store) with no special ctx
pub fn r_ctx<F>(f: F) -> RuleFn
where
    F: Fn(&mut UOpCtx, &Store) -> Option<UopId> + Send + Sync + 'static,
{
    Arc::new(move |ctx, st, _ctxv| f(ctx, st))
}

/// rule helper: closure over (AllocCtx, UOpCtx, store)
pub fn r_al<F>(f: F) -> RuleFn
where
    F: Fn(&mut crate::tensor::callify::AllocCtx, &mut UOpCtx, &Store) -> Option<UopId>
        + Send
        + Sync
        + 'static,
{
    Arc::new(move |ctx, st, ctxv| match ctxv {
        CtxVal::Al(a) => f(&mut a.borrow_mut(), ctx, st),
        _ => panic!("TGERR: rule requires AllocCtx"),
    })
}

/// rule helper: closure over (LocalAddBufferContext, UOpCtx, store)
pub fn r_la<F>(f: F) -> RuleFn
where
    F: Fn(&mut LocalAddBufferContext, &mut UOpCtx, &Store) -> Option<UopId> + Send + Sync + 'static,
{
    Arc::new(move |ctx, st, ctxv| match ctxv {
        CtxVal::La(l) => f(&mut l.borrow_mut(), ctx, st),
        _ => panic!("TGERR: rule requires LocalAddBufferContext"),
    })
}

/// rule helper: closure over (counter, UOpCtx, store)
pub fn r_cnt<F>(f: F) -> RuleFn
where
    F: Fn(&std::cell::Cell<i64>, &mut UOpCtx, &Store) -> Option<UopId> + Send + Sync + 'static,
{
    Arc::new(move |ctx, st, ctxv| match ctxv {
        CtxVal::Cnt(c) => f(c, ctx, st),
        _ => panic!("TGERR: rule requires counter ctx"),
    })
}

/// rule helper: closure over (device ctx, UOpCtx, store)
pub fn r_dev<F>(f: F) -> RuleFn
where
    F: Fn(&Option<crate::uop::arg::DeviceSpec>, &mut UOpCtx, &Store) -> Option<UopId>
        + Send
        + Sync
        + 'static,
{
    Arc::new(move |ctx, st, ctxv| match ctxv {
        CtxVal::Dev(d) => f(d, ctx, st),
        _ => panic!("TGERR: rule requires device ctx"),
    })
}

/// rule helper: closure over (mutable mark dict, UOpCtx, store)
pub fn r_dict<F>(f: F) -> RuleFn
where
    F: Fn(&mut HashMap<UopId, UopId>, &mut UOpCtx, &Store) -> Option<UopId> + Send + Sync + 'static,
{
    Arc::new(move |ctx, st, ctxv| match ctxv {
        CtxVal::Dict(d) => f(&mut d.borrow_mut(), ctx, st),
        _ => panic!("TGERR: rule requires dict ctx"),
    })
}

/// rule helper: closure over ((from, to) dtype pair, UOpCtx, store)
pub fn r_pair<F>(f: F) -> RuleFn
where
    F: Fn(crate::dtype::DTypeId, crate::dtype::DTypeId, &mut UOpCtx, &Store) -> Option<UopId>
        + Send
        + Sync
        + 'static,
{
    Arc::new(move |ctx, st, ctxv| match ctxv {
        CtxVal::Pair(fr, to) => f(*fr, *to, ctx, st),
        _ => panic!("TGERR: rule requires Pair ctx"),
    })
}

/// rule helper: closure over (DtypeDecompCtx, UOpCtx, store)
pub fn r_decomp<F>(f: F) -> RuleFn
where
    F: Fn(&mut crate::codegen::decomp_dtype::DtypeDecompCtx, &mut UOpCtx, &Store) -> Option<UopId>
        + Send
        + Sync
        + 'static,
{
    Arc::new(move |ctx, st, ctxv| match ctxv {
        CtxVal::Decomp(d) => f(&mut d.borrow_mut(), ctx, st),
        _ => panic!("TGERR: rule requires Decomp ctx"),
    })
}

/// rule helper: closure over (L2i memo dict, UOpCtx, store)
pub fn r_l2i<F>(f: F) -> RuleFn
where
    F: Fn(
            &std::rc::Rc<
                std::cell::RefCell<HashMap<(Ops, crate::dtype::DTypeId, Vec<UopId>), Vec<UopId>>>,
            >,
            &mut UOpCtx,
            &Store,
        ) -> Option<UopId>
        + Send
        + Sync
        + 'static,
{
    Arc::new(move |ctx, st, ctxv| match ctxv {
        CtxVal::L2i(d) => f(d, ctx, st),
        _ => panic!("TGERR: rule requires L2i ctx"),
    })
}
#[derive(Default)]
struct PmInner {
    pdict: HashMap<Ops, Vec<usize>>,
    rules: Vec<Arc<Rule>>,
    /// Whether this matcher has a stable process-wide identity (built once by a
    /// [`PmSlot`] or an `add` whose operands were stable). Only stable operands
    /// are eligible for `add` memoization, so freshly built temporaries cannot
    /// grow the cache without bound.
    stable: bool,
}

/// A compiled, immutable rule set. Cheap to `clone`: internally an `Arc`, so
/// clones share the same identity (used by `add`'s memo).
#[derive(Clone)]
pub struct PatternMatcher(Arc<PmInner>);

impl Default for PatternMatcher {
    fn default() -> Self {
        PatternMatcher(Arc::new(PmInner::default()))
    }
}

/// Memo for `add`, keyed by the identity (inner `Arc` address) of the two
/// operands. Mirrors tinygrad's `@functools.cache` on `PatternMatcher.__add__`
/// (ops.py:1470). Entries hold strong refs, so operand addresses stay valid; the
/// set is bounded by the number of distinct *stable* compositions.
type AddCache = std::sync::Mutex<HashMap<(usize, usize), PatternMatcher>>;

fn add_cache() -> &'static AddCache {
    static CACHE: std::sync::OnceLock<AddCache> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

impl PatternMatcher {
    pub fn new(patterns: Vec<(UPat, RuleFn)>) -> PatternMatcher {
        let mut inner = PmInner::default();
        for (pat, f) in patterns {
            let ops = pat.op.clone().expect("every UPat must have ops");
            let idx = inner.rules.len();
            inner.rules.push(Arc::new(Rule {
                early_reject: pat.early_reject.clone(),
                pat,
                f,
            }));
            for op in ops {
                inner.pdict.entry(op).or_default().push(idx);
            }
        }
        PatternMatcher(Arc::new(inner))
    }

    /// Mark this matcher as having a stable identity (see `PmInner::stable`).
    /// No-op if the inner `Arc` is already shared.
    pub fn interned(mut self) -> Self {
        if let Some(inner) = Arc::get_mut(&mut self.0) {
            inner.stable = true;
        }
        self
    }

    pub fn rules(&self) -> &[Arc<Rule>] {
        &self.0.rules
    }

    /// `pm1 + pm2` — concatenate rule sets. Rules are shared (`Arc<Rule>`), so
    /// this copies pointers, not the `UPat` trees. When both operands have a
    /// stable identity the result is memoized, so the same composition is built
    /// only once (tinygrad's `functools.cache` on `__add__`, ops.py:1470).
    pub fn add(&self, more: &PatternMatcher) -> PatternMatcher {
        let memo = self.0.stable && more.0.stable;
        let key = (Arc::as_ptr(&self.0) as usize, Arc::as_ptr(&more.0) as usize);
        if memo && let Some(hit) = add_cache().lock().unwrap().get(&key) {
            return hit.clone();
        }
        let mut inner = PmInner {
            stable: memo,
            ..Default::default()
        };
        inner.rules.extend(self.0.rules.iter().cloned());
        inner.rules.extend(more.0.rules.iter().cloned());
        {
            let PmInner { pdict, rules, .. } = &mut inner;
            for (idx, rule) in rules.iter().enumerate() {
                if let Some(ops) = &rule.pat.op {
                    for op in ops {
                        pdict.entry(*op).or_default().push(idx);
                    }
                }
            }
        }
        let pm = PatternMatcher(Arc::new(inner));
        if memo {
            add_cache().lock().unwrap().insert(key, pm.clone());
        }
        pm
    }

    /// `pm.rewrite(uop, ctx)` — first rule match wins.
    pub fn rewrite(&self, ctx: &mut UOpCtx, uop: UopId, ctxval: &CtxVal) -> Option<UopId> {
        let idxs = self.0.pdict.get(&ctx.op_of(&uop))?;
        let ler: HashSet<Ops> = ctx.srcs(&uop).iter().map(|s| ctx.op_of(s)).collect();
        for &i in idxs {
            let rule = &self.0.rules[i];
            if !rule.early_reject.iter().all(|o| ler.contains(o)) {
                continue;
            }
            let mut stores = Vec::new();
            rule.pat
                .match_uop(ctx, uop.clone(), Store::new(), &mut stores);
            for store in &stores {
                #[cfg(feature = "trace")]
                if std::env::var("TG_TRACE").is_ok() {
                    eprintln!(
                        "RULE[{}/{}] pat={:?} ops={:?} fire on u{} op {:?}",
                        i,
                        self.0.rules.len(),
                        rule.pat.name,
                        rule.pat.op,
                        uop.id(),
                        ctx.op_of(&uop)
                    );
                }
                let _r = (rule.f)(ctx, store, ctxval);
                #[cfg(feature = "trace")]
                if std::env::var("TG_TRACE").is_ok() {
                    eprintln!(
                        "  -> u{} op {:?}",
                        _r.clone().map(|x| x.id()).unwrap_or(0),
                        _r.as_ref().map(|x| ctx.op_of(x))
                    );
                }
                if let Some(ret) = _r
                    && ret != uop
                {
                    return Some(ret);
                }
                // python: rules returning self are skipped, try the next rule
            }
        }
        None
    }
}

/// helper for rules: `st["x"]`
pub fn sget(store: &Store, name: &str) -> UopId {
    store
        .get(name)
        .unwrap_or_else(|| {
            let keys: Vec<&str> = store.keys().cloned().collect();
            panic!("TGERR: missing pattern binding {name} (have {keys:?})")
        })
        .clone()
}

pub fn sget_opt(store: &Store, name: &str) -> Option<UopId> {
    store.get(name).cloned()
}

/// `const_arg(u)` (symbolic.py:30-34): CONST val, CAST-of-CONST canonicalized
/// to the cast dtype, or STACK tuple of vals.
pub fn const_arg(ctx: &mut UOpCtx, u: UopId) -> Option<ConstArgVal> {
    let (op, dt, srcs, _, _) = ctx.snapshot(&u);
    match op {
        Ops::Const => Some(ConstArgVal::One(ctx.val(&u))),
        Ops::Cast if ctx.op_of(&srcs[0]) == Ops::Const => {
            let v = crate::dtype::dtype_const(dt, ctx.val(&srcs[0]));
            Some(ConstArgVal::One(v))
        }
        Ops::Stack if srcs.iter().all(|s| ctx.op_of(s) == Ops::Const) => {
            Some(ConstArgVal::Vec(srcs.iter().map(|s| ctx.val(s)).collect()))
        }
        _ => None,
    }
}

#[derive(Clone, Debug)]
pub enum ConstArgVal {
    One(crate::dtype::ConstVal),
    Vec(Vec<crate::dtype::ConstVal>),
}

impl ConstArgVal {
    pub fn one(self) -> Option<crate::dtype::ConstVal> {
        match self {
            ConstArgVal::One(v) => Some(v),
            _ => None,
        }
    }
}

/// op set helpers for rule definitions
pub fn opset_union(a: &[Ops], b: &[Ops]) -> Vec<Ops> {
    let mut v = a.to_vec();
    for &o in b {
        if !v.contains(&o) {
            v.push(o);
        }
    }
    v
}

pub fn opset_diff(a: &[Ops], b: &[Ops]) -> Vec<Ops> {
    a.iter().filter(|o| !b.contains(o)).cloned().collect()
}

pub fn opset_single(set: &[Ops]) -> Vec<Ops> {
    set.to_vec()
}

pub fn tuplize_lt(ctx: &UOpCtx, a: UopId, b: UopId) -> std::cmp::Ordering {
    ctx.tuplize(&a).cmp(&ctx.tuplize(&b))
}

/// `bool` dtype membership used by several rules.
pub fn is_bool_dtype(ctx: &UOpCtx, u: UopId) -> bool {
    is_bool(ctx.dt_of(&u))
}

// ---------- graph rewrite engine (ops.py:1627-1746) ----------

/// Raised by bottom-up rules to stop rewriting a node (used by rangeify in M4).
#[derive(Debug)]
pub struct BottomUpGate;

pub const REWRITE_STACK_LIMIT: usize = 250000;

/// `_OPAQUE_CALL_BODIES` (ops.py:1179).
pub const OPAQUE_CALL_BODIES: &[Ops] = &[
    Ops::Sink,
    Ops::Program,
    Ops::Linear,
    Ops::Copy,
    Ops::CustomFunction,
];

fn is_opaque_body_call(ctx: &UOpCtx, n: UopId) -> bool {
    let op = ctx.op_of(&n);
    if op == Ops::Function {
        return true;
    }
    if op == Ops::Call {
        let srcs = ctx.srcs(&n);
        return !srcs.is_empty() && group::in_(OPAQUE_CALL_BODIES, ctx.op_of(&srcs[0]));
    }
    false
}

/// `_rebuild_dtype` (ops.py:1748-1751).
fn rebuild_dtype(ctx: &mut UOpCtx, n: UopId, new_src: &[UopId]) -> DTypeId {
    let old = ctx.srcs(&n);
    if old.len() == new_src.len()
        && old
            .iter()
            .zip(new_src)
            .all(|(a, b)| ctx.dt_of(a) == ctx.dt_of(b))
    {
        return ctx.dt_of(&n);
    }
    let (_, _, _, arg, _) = ctx.snapshot(&n);
    ctx.dtype_from_uop(ctx.op_of(&n), new_src, &arg)
        .unwrap_or_else(|| panic!("TGERR: dtype derivation failed in rebuild"))
}

struct RewriteCtxt<'a> {
    pm: Option<&'a PatternMatcher>,
    bpm: Option<&'a PatternMatcher>,
    bpm_cache: HashMap<UopId, Option<UopId>>,
    ctxval: &'a CtxVal,
    replace: HashMap<UopId, UopId>,
    enter_calls: bool,
}

impl<'a> RewriteCtxt<'a> {
    fn pm_rewrite(&mut self, ctx: &mut UOpCtx, x: UopId) -> Option<UopId> {
        self.pm.unwrap().rewrite(ctx, x, self.ctxval)
    }

    fn cached_bpm_rewrite(
        &mut self,
        ctx: &mut UOpCtx,
        x: UopId,
    ) -> Result<Option<UopId>, BottomUpGate> {
        if let Some(ret) = self.bpm_cache.get(&x) {
            return Ok(ret.clone());
        }
        let ret = self.bpm.unwrap().rewrite(ctx, x.clone(), self.ctxval);
        self.bpm_cache.insert(x, ret.clone());
        Ok(ret)
    }

    /// MLIR-style walk driver (ops.py:1649-1673).
    fn walk_rewrite(&mut self, ctx: &mut UOpCtx, root: UopId) -> Result<UopId, String> {
        enum Step {
            Enter(UopId),
            Rebuild(UopId),
        }
        let mut stack = vec![Step::Enter(root.clone())];
        let mut iters = 0usize;
        while let Some(step) = stack.pop() {
            iters += 1;
            if iters > 400_000 {
                return Err("rewrite step limit (walk)".into());
            }
            match step {
                Step::Enter(n) => {
                    if self.replace.contains_key(&n) {
                        continue;
                    }
                    // bottom-up: try bpm on the original node first, use result as-is
                    if self.bpm.is_some() {
                        let rewritten = self
                            .cached_bpm_rewrite(ctx, n.clone())
                            .map_err(|_| "bottom-up gate in walk".to_string())?;
                        if let Some(r) = rewritten {
                            self.replace.insert(n, r);
                            continue;
                        }
                    }
                    stack.push(Step::Rebuild(n.clone()));
                    if !self.enter_calls && is_opaque_body_call(ctx, n.clone()) {
                        let body = &ctx.srcs(&n)[0].clone();
                        self.replace.insert(body.clone(), body.clone());
                    }
                    for x in ctx.srcs(&n).iter().rev() {
                        if !self.replace.contains_key(x) {
                            stack.push(Step::Enter(x.clone()));
                        }
                    }
                }
                Step::Rebuild(n) => {
                    let srcs = ctx.srcs(&n).to_vec();
                    let new_src: Vec<UopId> = srcs
                        .iter()
                        .map(|x| self.replace.get(x).cloned().unwrap_or(x.clone()))
                        .collect();
                    let new_n = if new_src != srcs {
                        let (op, _, _, arg, tag) = ctx.snapshot(&n);
                        let dt = rebuild_dtype(ctx, n.clone(), &new_src);
                        ctx.intern_raw(op, dt, new_src, arg, tag)
                    } else {
                        n.clone()
                    };
                    // top-down: try pm on the rebuilt node, use result as-is
                    let new_n = if self.pm.is_some() {
                        self.pm_rewrite(ctx, new_n.clone()).unwrap_or(new_n)
                    } else {
                        new_n
                    };
                    self.replace.insert(n, new_n);
                }
            }
        }
        Ok(self.replace.get(&root).cloned().unwrap_or(root))
    }

    /// the main driver (ops.py:1675-1741).
    fn unified_rewrite(&mut self, ctx: &mut UOpCtx, root: UopId) -> Result<UopId, String> {
        #[derive(Clone, Copy, PartialEq, Debug)]
        enum Stage {
            S0,
            S1,
            S2,
        }
        let mut stack: Vec<(UopId, Stage, UopId)> = vec![(root.clone(), Stage::S0, root.clone())];
        let mut on_stack: HashSet<UopId> = HashSet::from([root.clone()]);
        let mut waitlist: HashMap<UopId, Vec<(UopId, Stage, UopId)>> = HashMap::new();
        let mut iters = 0usize;
        while let Some((n, stage, new_n)) = stack.pop() {
            iters += 1;
            if iters > 400_000 {
                return Err("rewrite step limit".into());
            }
            if stack.len() > REWRITE_STACK_LIMIT {
                return Err("infinite loop in graph_rewrite (stack too big)".into());
            }
            if self.replace.contains_key(&n) {
                // NOTE: python can't hit this for freshly rebuilt nodes (identity
                // objects); structural interning can, so fire the waiters.
                if let Some(w) = waitlist.remove(&n) {
                    stack.extend(w);
                }
                continue;
            }
            match stage {
                Stage::S0 => {
                    let mut new_n = new_n;
                    if let Some(_bpm) = self.bpm {
                        // apply rewrite rules until a fixed point is reached
                        let mut test_n: Option<UopId> = Some(new_n.clone());
                        let mut seen: HashSet<UopId> = HashSet::new();
                        let mut gated: Option<UopId> = None;
                        while let Some(tn) = test_n {
                            if seen.contains(&tn) {
                                return Err("infinite loop in fixed_point_rewrite".into());
                            }
                            seen.insert(tn.clone());
                            new_n = tn.clone();
                            match self.cached_bpm_rewrite(ctx, tn.clone()) {
                                Ok(Some(next)) => test_n = Some(next),
                                Ok(None) => test_n = None,
                                Err(BottomUpGate) => {
                                    gated = Some(tn);
                                    break;
                                }
                            }
                        }
                        if let Some(tn) = gated {
                            // gate: done with this node, don't continue down the srcs
                            self.replace.insert(n.clone(), tn);
                            if let Some(w) = waitlist.remove(&n) {
                                stack.extend(w);
                            }
                            continue;
                        }
                    }
                    stack.push((n, Stage::S1, new_n.clone()));
                    if !self.enter_calls && is_opaque_body_call(ctx, new_n.clone()) {
                        let body = &ctx.srcs(&new_n)[0].clone();
                        self.replace.insert(body.clone(), body.clone());
                    }
                    for x in ctx.srcs(&new_n).iter().rev() {
                        if on_stack.contains(x) {
                            continue;
                        }
                        stack.push((x.clone(), Stage::S0, x.clone()));
                        on_stack.insert(x.clone());
                    }
                }
                Stage::S1 => {
                    let srcs = ctx.srcs(&new_n).to_vec();
                    let mut tmp = Vec::with_capacity(srcs.len());
                    let mut ready = true;
                    for x in &srcs {
                        match self.replace.get(x) {
                            Some(rx) => tmp.push(rx.clone()),
                            None => {
                                waitlist.entry(x.clone()).or_default().push((
                                    n.clone(),
                                    Stage::S1,
                                    new_n.clone(),
                                ));
                                ready = false;
                                break;
                            }
                        }
                    }
                    if !ready {
                        continue;
                    }
                    let new_src = tmp;
                    if new_src == srcs {
                        // top-down rewrite of the node as-is
                        let rewritten = if self.pm.is_some() {
                            self.pm_rewrite(ctx, new_n.clone())
                        } else {
                            None
                        };
                        match rewritten {
                            None => {
                                self.replace.insert(n.clone(), new_n);
                                if let Some(w) = waitlist.remove(&n) {
                                    stack.extend(w);
                                }
                                continue;
                            }
                            Some(new_src_n) => {
                                stack.push((n, Stage::S2, new_src_n.clone()));
                                stack.push((new_src_n.clone(), Stage::S0, new_src_n));
                            }
                        }
                    } else {
                        let (op, _, _, arg, tag) = ctx.snapshot(&new_n);
                        let dt = rebuild_dtype(ctx, new_n, &new_src);
                        let new_src_n = ctx.intern_raw(op, dt, new_src, arg, tag);
                        stack.push((n, Stage::S2, new_src_n.clone()));
                        stack.push((new_src_n.clone(), Stage::S0, new_src_n));
                    }
                }
                Stage::S2 => match self.replace.get(&new_n) {
                    None => {
                        waitlist
                            .entry(new_n.clone())
                            .or_default()
                            .push((n, Stage::S2, new_n));
                    }
                    Some(rn) => {
                        self.replace.insert(n.clone(), rn.clone());
                        if let Some(w) = waitlist.remove(&n) {
                            stack.extend(w);
                        }
                    }
                },
            }
        }
        match self.replace.get(&root) {
            Some(r) => Ok(r.clone()),
            None => Err(format!("rewrite did not complete root {:?}", root.id())),
        }
    }
}

/// `graph_rewrite` (ops.py:1743-1746).
#[allow(clippy::too_many_arguments)] // signature mirrors the tinygrad oracle
pub fn graph_rewrite(
    ctx: &mut UOpCtx,
    sink: UopId,
    pm: &PatternMatcher,
    bottom_up: bool,
    bpm: Option<&PatternMatcher>,
    walk: bool,
    enter_calls: bool,
    ctxval: CtxVal,
) -> Result<UopId, String> {
    let mut rc = RewriteCtxt {
        pm: if bottom_up { None } else { Some(pm) },
        bpm: if bottom_up { Some(pm) } else { bpm },
        bpm_cache: HashMap::new(),
        ctxval: &ctxval,
        replace: HashMap::new(),
        enter_calls,
    };
    let _ = &mut rc;
    if walk {
        rc.walk_rewrite(ctx, sink)
    } else {
        rc.unified_rewrite(ctx, sink)
    }
}

/// `UOp.substitute` (ops.py:529-534) — bottom-up rewrite with `_substitute`.
/// `substitute` with an extra PM composed before `_substitute`
/// (python `substitute(..., extra_pm=...)`).
pub fn substitute_extra(
    ctx: &mut UOpCtx,
    root: UopId,
    dvars: HashMap<UopId, UopId>,
    extra: Option<&PatternMatcher>,
    walk: bool,
    enter_calls: bool,
) -> UopId {
    if dvars.is_empty() {
        return root;
    }
    let extra = extra.expect("TGERR: substitute_extra requires extra pm");
    let pm = extra.add(substitute_pm());
    graph_rewrite(
        ctx,
        root,
        &pm,
        true,
        None,
        walk,
        enter_calls,
        CtxVal::Subst(dvars),
    )
    .unwrap_or_else(|e| panic!("TGERR: substitute_extra failed: {e}"))
}

pub fn substitute(
    ctx: &mut UOpCtx,
    root: UopId,
    dvars: HashMap<UopId, UopId>,
    walk: bool,
    enter_calls: bool,
) -> UopId {
    let dvars: HashMap<UopId, UopId> = dvars.into_iter().filter(|(k, v)| k != v).collect();
    if dvars.is_empty() {
        return root;
    }
    let pm = substitute_pm();
    graph_rewrite(
        ctx,
        root,
        pm,
        true,
        None,
        walk,
        enter_calls,
        CtxVal::Subst(dvars),
    )
    .unwrap_or_else(|e| panic!("TGERR: substitute failed: {e}"))
}

static SUBSTITUTE_PM: PmSlot = PmSlot::new();

pub fn substitute_pm() -> &'static PatternMatcher {
    SUBSTITUTE_PM.get_or_init(|| {
        let pat = UPat::op_var((0..=79u16).filter_map(Ops::from_u16).collect()).named("x");
        PatternMatcher::new(vec![(
            pat,
            Arc::new(|_, x, ctxv: &CtxVal| ctxv.get_subst(sget(x, "x"))),
        )])
    })
}

/// `UOp.simplify` (ops.py:510-516) with `symbolic` (sym.rs).
pub fn simplify(ctx: &mut UOpCtx, u: UopId) -> UopId {
    if ctx.op_of(&u) == Ops::Const {
        return u;
    }
    if ctx.op_of(&u) == Ops::Sink {
        let ok = ctx.srcs(&u).iter().all(|s| {
            ctx.op_of(s) == Ops::Const || { ctx.op_of(s) == Ops::Stack && ctx.srcs(s).is_empty() }
        });
        if ok {
            return u;
        }
    }
    let pm = super::sym::symbolic();
    graph_rewrite(ctx, u, pm, false, None, false, false, CtxVal::None)
        .unwrap_or_else(|e| panic!("TGERR: simplify failed: {e}"))
}

/// SymVal: what `ssimplify` returns in Python (ConstType | UOp).
#[derive(Clone, Debug)]
pub enum SymVal {
    C(crate::dtype::ConstVal),
    U(UopId),
}

impl SymVal {
    pub fn to_sval(&self) -> SVal {
        match self {
            SymVal::C(c) => {
                let i = match c {
                    crate::dtype::ConstVal::Int(v) => *v,
                    crate::dtype::ConstVal::Bool(b) => *b as i128,
                    crate::dtype::ConstVal::Big(b) => {
                        b.as_i64().map(|v| v as i128).unwrap_or(if b.is_negative() {
                            i128::MIN / 2
                        } else {
                            i128::MAX / 2
                        })
                    }
                    crate::dtype::ConstVal::Float(f) => *f as i128,
                    crate::dtype::ConstVal::Invalid => 0,
                };
                SVal::I(i)
            }
            SymVal::U(u) => SVal::U(u.clone()),
        }
    }
}

/// `ssimplify` (ops.py:517-519): simplify, then const-ify `CAST(CONST)`/`CONST`.
pub fn ssimplify(ctx: &mut UOpCtx, u: UopId) -> SymVal {
    let ret = simplify(ctx, u);
    let (op, dt, srcs, _, _) = ctx.snapshot(&ret);
    if op == Ops::Cast && ctx.op_of(&srcs[0]) == Ops::Const {
        let v = crate::dtype::dtype_const(dt, ctx.val(&srcs[0]));
        return SymVal::C(v);
    }
    if op == Ops::Const {
        return SymVal::C(ctx.val(&ret));
    }
    SymVal::U(ret)
}

/// `resolve` (ops.py:54-58).
pub fn resolve(ctx: &mut UOpCtx, x: UopId, default: bool) -> bool {
    if !is_bool(ctx.dt_of(&x)) {
        panic!("TGERR: UOp in resolve must be bool");
    }
    let sx = simplify(ctx, x);
    let (vmin, vmax) = ctx.min_max(&sx).unwrap_or_else(|e| panic!("TGERR: {e}"));
    if vmin == vmax {
        // bool(vmin)
        match vmin {
            SVal::I(v) => v != 0,
            SVal::U(_) => panic!("TGERR: symbolic vmin in resolve"),
        }
    } else {
        default
    }
}

/// Canonical, order-sensitive signature of a `UPat`, used by the rule-parity
/// inventory (`pm_inventory`) to diff this port against tinygrad rule-by-rule.
pub fn pat_sig(p: &UPat) -> String {
    let ops = match &p.op {
        None => "ANY".to_string(),
        Some(v) => {
            let mut ns: Vec<&str> = v.iter().map(|o| o.pyname()).collect();
            ns.sort_unstable();
            ns.join(",")
        }
    };
    let name = p.name.unwrap_or("");
    let mut s = format!(
        "{}|{}|req={}|strict={}",
        ops, name, p.required_len, p.strict_length
    );
    let kids: Vec<String> = match &p.src {
        SrcPat::None => Vec::new(),
        SrcPat::Tup(v) => v.iter().map(pat_sig).collect(),
        SrcPat::List(ps) => ps
            .first()
            .map(|x| x.iter().map(pat_sig).collect())
            .unwrap_or_default(),
        SrcPat::Repeat(x) => vec![format!("{}*", pat_sig(x))],
        SrcPat::Any(v) => v.iter().map(pat_sig).collect(),
    };
    if !kids.is_empty() {
        s.push('<');
        s.push_str(&kids.join(","));
        s.push('>');
    }
    s
}

impl PatternMatcher {
    /// One canonical signature per rule, in rule order.
    pub fn rule_sigs(&self) -> Vec<String> {
        self.rules().iter().map(|r| pat_sig(&r.pat)).collect()
    }
}
