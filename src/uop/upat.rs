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
use std::sync::{Arc, OnceLock};

use super::arg::Arg;
use super::ctx::UOpCtx;
use super::graph::UopId;
use super::ops::{group, Ops};
use super::sint::SVal;
use crate::dtype::{is_bool, ConstVal, DTypeId};
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
            if let Some(ops) = &p.op {
                if ops.len() == 1 && !early_reject.contains(&ops[0]) {
                    early_reject.push(ops[0]);
                }
            }
        }
        UPat { op, dtype, tag, arg, name, src, strict_length, required_len, early_reject }
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
        UPat::new(Some(vec![Ops::Const]), dtype, SrcPat::None, arg, name, false, None)
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

    pub fn with_src_tup(&self, srcs: Vec<UPat>) -> UPat {
        let mut p = self.clone();
        p.src = SrcPat::Tup(srcs);
        p.strict_length = true;
        p.required_len = {
            let SrcPat::Tup(v) = &p.src else { unreachable!() };
            v.len()
        };
        p.recompute_early_reject();
        p
    }

    pub(crate) fn recompute_early_reject_rep(&mut self, p: &UPat) {
        let mut er = Vec::new();
        if let Some(ops) = &p.op {
            if ops.len() == 1 {
                er.push(ops[0]);
            }
        }
        self.early_reject = er;
    }

    fn recompute_early_reject(&mut self) {
        let mut er = Vec::new();
        if let SrcPat::Tup(v) = &self.src {
            for p in v {
                if let Some(ops) = &p.op {
                    if ops.len() == 1 && !er.contains(&ops[0]) {
                        er.push(ops[0]);
                    }
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
        let src = if group::in_(group::COMMUTATIVE, op) { SrcPat::List(permutations(asrc)) } else { SrcPat::Tup(asrc) };
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
            UPat::new(Some(vec![Ops::Cast]), Some(vec![d]), SrcPat::Tup(vec![self]), None, None, false, None)
        } else {
            UPat::new(Some(vec![Ops::Cast]), None, SrcPat::Tup(vec![self]), None, None, false, None)
        }
    }

    /// cast with a dtype SET (Python `cast(dtypes.ints + ...)`)
    pub fn cast_dts(self, dts: Vec<DTypeId>) -> UPat {
        UPat::new(Some(vec![Ops::Cast]), Some(dts), SrcPat::Tup(vec![self]), None, None, false, None)
    }

    pub fn bitcast(self, dtype: Option<DTypeId>) -> UPat {
        UPat::new(Some(vec![Ops::Bitcast]), dtype.map(|d| vec![d]), SrcPat::Tup(vec![self]), None, None, false, None)
    }

    pub fn logical_not(self) -> UPat {
        // mixin: self.cast(bool).ne(True)
        self.cast(Some(crate::dtype::BOOL)).ne(UPatOrConst::Val(ConstVal::Bool(true)))
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
        let mut p = UPat::new(Some(vec![Ops::Reduce]), dt, SrcPat::Tup(asrc), arg, None, allow_any_len, None);
        p.recompute_early_reject();
        p
    }

    pub fn load(self, src: Vec<UPat>, allow_any_len: bool) -> UPat {
        let dt = self.dtype.clone();
        let mut asrc = vec![self];
        asrc.extend(src);
        let mut p = UPat::new(Some(vec![Ops::Load]), dt, SrcPat::Tup(asrc), None, None, allow_any_len, None);
        p.recompute_early_reject();
        p
    }

    pub fn store(self, src: Vec<UPat>) -> UPat {
        let dt = self.dtype.clone();
        let mut asrc = vec![self];
        asrc.extend(src);
        let mut p = UPat::new(Some(vec![Ops::Store]), dt, SrcPat::Tup(asrc), None, None, false, None);
        p.recompute_early_reject();
        p
    }

    pub fn index(self, srcs: Vec<UPat>) -> UPat {
        let dt = self.dtype.clone();
        let mut asrc = vec![self];
        asrc.extend(srcs);
        let mut p = UPat::new(Some(vec![Ops::Index]), dt, SrcPat::Tup(asrc), None, None, false, None);
        p.recompute_early_reject();
        p
    }

    pub fn after(self, src: Vec<UPat>, allow_any_len: bool) -> UPat {
        let dt = self.dtype.clone();
        let mut asrc = vec![self];
        asrc.extend(src);
        let mut p = UPat::new(Some(vec![Ops::After]), dt, SrcPat::Tup(asrc), None, None, allow_any_len, None);
        p.recompute_early_reject();
        p
    }

    pub fn end(self, src: Vec<UPat>) -> UPat {
        let dt = self.dtype.clone();
        let mut asrc = vec![self];
        asrc.extend(src);
        let mut p = UPat::new(Some(vec![Ops::End]), dt, SrcPat::Tup(asrc), None, None, false, None);
        p.recompute_early_reject();
        p
    }

    /// `UPat(Ops.STACK, self.match_dtype, src=self, ...)` — repeat src.
    pub fn broadcast(self, allow_any_len: bool) -> UPat {
        let dt = self.dtype.clone();
        let src = SrcPat::Repeat(Box::new(self));
        UPat::new(Some(vec![Ops::Stack]), dt, src, None, None, allow_any_len, None)
    }

    /// `or_casted(name)` (ops.py:1360).
    pub fn or_casted(self, name: Option<&'static str>) -> UPat {
        let a = match name {
            None => self.clone(),
            Some(n) => self.named(n),
        };
        let b = UPat::new(Some(vec![Ops::Cast]), None, SrcPat::Tup(vec![self.clone()]), None, name, false, None);
        UPat::any(vec![a, b])
    }

    /// `or_after(name)`.
    pub fn or_after(self, name: Option<&'static str>) -> UPat {
        let a = match name {
            None => self.clone(),
            Some(n) => self.named(n),
        };
        let b = {
            let p = UPat::new(Some(vec![Ops::After]), None, SrcPat::Tup(vec![self.clone()]), None, name, true, None);
            p
        };
        UPat::any(vec![a, b])
    }

    // ---------- matching ----------

    pub fn match_uop(&self, ctx: &UOpCtx, uop: UopId, store: Store, out: &mut Vec<Store>) {
        if let SrcPat::Any(alts) = &self.src {
            for alt in alts {
                alt.match_uop(ctx, uop, store.clone(), out);
            }
            return;
        }
        // op check
        if let Some(ops) = &self.op {
            if !ops.contains(&ctx.op_of(uop)) {
                return;
            }
        }
        // name binding (identity)
        let mut store = store;
        if let Some(n) = self.name {
            match store.get(n) {
                Some(&prev) if prev != uop => return,
                Some(_) => {}
                None => {
                    store.insert(n, uop);
                }
            }
        }
        // dtype
        if let Some(dts) = &self.dtype {
            if !dts.contains(&ctx.dt_of(uop)) {
                return;
            }
        }
        // arg (Python `!=` — bools compare as ints)
        if let Some(pa) = &self.arg {
            if !arg_matches(pa, ctx.arg_of(uop)) {
                return;
            }
        }
        // tag
        if let Some(tags) = &self.tag {
            if !tags.iter().any(|t| t == ctx.tag_of(uop)) {
                return;
            }
        }
        let srcs = ctx.srcs(uop);
        if srcs.len() < self.required_len || (self.strict_length && srcs.len() != self.required_len) {
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
                for &uu in srcs {
                    let mut next = Vec::new();
                    for s in stores {
                        let mut o = Vec::new();
                        p.match_uop(ctx, uu, s, &mut o);
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
        for &uu in srcs {
            let Some(vv) = vp.next() else { break };
            let mut new_stores = Vec::new();
            for s in stores {
                let mut o = Vec::new();
                vv.match_uop(ctx, uu, s, &mut o);
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

#[derive(Clone)]
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
        (Arg::Seq { tuple: t1, items: i1 }, Arg::Seq { tuple: t2, items: i2 }) => {
            t1 == t2 && i1.len() == i2.len() && i1.iter().zip(i2).all(|(a, b)| arg_matches(a, b))
        }
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
    fn helper<T: Clone>(items: &[T], cur: &mut Vec<T>, used: &mut Vec<bool>, out: &mut Vec<Vec<T>>) {
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
}

impl CtxVal {
    pub fn get_subst(&self, u: UopId) -> Option<UopId> {
        match self {
            CtxVal::Subst(m) => m.get(&u).copied(),
            _ => None,
        }
    }
}

#[derive(Default)]
pub struct PatternMatcher {
    pdict: HashMap<Ops, Vec<usize>>,
    pub rules: Vec<Rule>,
}

impl PatternMatcher {
    pub fn new(patterns: Vec<(UPat, RuleFn)>) -> PatternMatcher {
        let mut pm = PatternMatcher::default();
        for (pat, f) in patterns {
            let ops = pat.op.clone().expect("every UPat must have ops");
            let idx = pm.rules.len();
            pm.rules.push(Rule { early_reject: pat.early_reject.clone(), pat, f });
            for op in ops {
                pm.pdict.entry(op).or_default().push(idx);
            }
        }
        pm
    }

    pub fn add(&self, more: &PatternMatcher) -> PatternMatcher {
        let mut patterns: Vec<(UPat, RuleFn)> = self.rules.iter().map(|r| (r.pat.clone(), r.f.clone())).collect();
        patterns.extend(more.rules.iter().map(|r| (r.pat.clone(), r.f.clone())));
        PatternMatcher::new(patterns)
    }

    /// `pm.rewrite(uop, ctx)` — first rule match wins.
    pub fn rewrite(&self, ctx: &mut UOpCtx, uop: UopId, ctxval: &CtxVal) -> Option<UopId> {
        let idxs = self.pdict.get(&ctx.op_of(uop))?;
        let ler: HashSet<Ops> = ctx.srcs(uop).iter().map(|&s| ctx.op_of(s)).collect();
        for &i in idxs {
            let rule = &self.rules[i];
            if !rule.early_reject.iter().all(|o| ler.contains(o)) {
                continue;
            }
            let mut stores = Vec::new();
            rule.pat.match_uop(ctx, uop, Store::new(), &mut stores);
            for store in &stores {
                if std::env::var("TG_TRACE").is_ok() {
                    eprintln!("RULE[{}] fire on op {:?}", i, ctx.op_of(uop));
                }
                let _ = &rule;
                if let Some(ret) = (rule.f)(ctx, store, ctxval) {
                    if ret != uop {
                        return Some(ret);
                    }
                    break; // first non-None result wins, even if it is the node itself
                }
            }
        }
        None
    }
}

/// helper for rules: `st["x"]`
pub fn sget(store: &Store, name: &str) -> UopId {
    *store.get(name).unwrap_or_else(|| {
        let keys: Vec<&str> = store.keys().copied().collect();
        panic!("TGERR: missing pattern binding {name} (have {keys:?})")
    })
}

pub fn sget_opt(store: &Store, name: &str) -> Option<UopId> {
    store.get(name).copied()
}

/// `const_arg(u)` (symbolic.py:30-34): CONST val, CAST-of-CONST canonicalized
/// to the cast dtype, or STACK tuple of vals.
pub fn const_arg(ctx: &mut UOpCtx, u: UopId) -> Option<ConstArgVal> {
    let (op, dt, srcs, _, _) = ctx.snapshot(u);
    match op {
        Ops::Const => Some(ConstArgVal::One(ctx.val(u))),
        Ops::Cast if ctx.op_of(srcs[0]) == Ops::Const => {
            let v = crate::dtype::dtype_const(dt, ctx.val(srcs[0]));
            Some(ConstArgVal::One(v))
        }
        Ops::Stack if srcs.iter().all(|&s| ctx.op_of(s) == Ops::Const) => {
            Some(ConstArgVal::Vec(srcs.iter().map(|&s| ctx.val(s)).collect()))
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
    a.iter().filter(|o| !b.contains(o)).copied().collect()
}

pub fn opset_single(set: &[Ops]) -> Vec<Ops> {
    set.to_vec()
}

pub fn tuplize_lt(ctx: &UOpCtx, a: UopId, b: UopId) -> std::cmp::Ordering {
    ctx.tuplize(a).cmp(&ctx.tuplize(b))
}

/// `bool` dtype membership used by several rules.
pub fn is_bool_dtype(ctx: &UOpCtx, u: UopId) -> bool {
    is_bool(ctx.dt_of(u))
}

// ---------- graph rewrite engine (ops.py:1627-1746) ----------

/// Raised by bottom-up rules to stop rewriting a node (used by rangeify in M4).
#[derive(Debug)]
pub struct BottomUpGate;

pub const REWRITE_STACK_LIMIT: usize = 250000;

/// `_OPAQUE_CALL_BODIES` (ops.py:1179).
pub const OPAQUE_CALL_BODIES: &[Ops] = &[Ops::Sink, Ops::Program, Ops::Linear, Ops::Copy, Ops::CustomFunction];

fn is_opaque_body_call(ctx: &UOpCtx, n: UopId) -> bool {
    let op = ctx.op_of(n);
    if op == Ops::Function {
        return true;
    }
    if op == Ops::Call {
        let srcs = ctx.srcs(n);
        return !srcs.is_empty() && group::in_(OPAQUE_CALL_BODIES, ctx.op_of(srcs[0]));
    }
    false
}

/// `_rebuild_dtype` (ops.py:1748-1751).
fn rebuild_dtype(ctx: &mut UOpCtx, n: UopId, new_src: &[UopId]) -> DTypeId {
    let old = ctx.srcs(n);
    if old.len() == new_src.len() && old.iter().zip(new_src).all(|(a, b)| ctx.dt_of(*a) == ctx.dt_of(*b)) {
        return ctx.dt_of(n);
    }
    let (_, _, _, arg, _) = ctx.snapshot(n);
    ctx.dtype_from_uop(ctx.op_of(n), new_src, &arg).unwrap_or_else(|| panic!("TGERR: dtype derivation failed in rebuild"))
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

    fn cached_bpm_rewrite(&mut self, ctx: &mut UOpCtx, x: UopId) -> Result<Option<UopId>, BottomUpGate> {
        if let Some(&ret) = self.bpm_cache.get(&x) {
            return Ok(ret);
        }
        let ret = self.bpm.unwrap().rewrite(ctx, x, self.ctxval);
        self.bpm_cache.insert(x, ret);
        Ok(ret)
    }

    /// MLIR-style walk driver (ops.py:1649-1673).
    fn walk_rewrite(&mut self, ctx: &mut UOpCtx, root: UopId) -> Result<UopId, String> {
        enum Step {
            Enter(UopId),
            Rebuild(UopId),
        }
        let mut stack = vec![Step::Enter(root)];
        while let Some(step) = stack.pop() {
            match step {
                Step::Enter(n) => {
                    if self.replace.contains_key(&n) {
                        continue;
                    }
                    // bottom-up: try bpm on the original node first, use result as-is
                    if self.bpm.is_some() {
                        let rewritten = self.cached_bpm_rewrite(ctx, n).map_err(|_| "bottom-up gate in walk".to_string())?;
                        if let Some(r) = rewritten {
                            self.replace.insert(n, r);
                            continue;
                        }
                    }
                    stack.push(Step::Rebuild(n));
                    if !self.enter_calls && is_opaque_body_call(ctx, n) {
                        let body = ctx.srcs(n)[0];
                        self.replace.insert(body, body);
                    }
                    for &x in ctx.srcs(n).iter().rev() {
                        if !self.replace.contains_key(&x) {
                            stack.push(Step::Enter(x));
                        }
                    }
                }
                Step::Rebuild(n) => {
                    let srcs = ctx.srcs(n).to_vec();
                    let new_src: Vec<UopId> = srcs.iter().map(|&x| self.replace.get(&x).copied().unwrap_or(x)).collect();
                    let new_n = if new_src != srcs {
                        let (op, _, _, arg, tag) = ctx.snapshot(n);
                        let dt = rebuild_dtype(ctx, n, &new_src);
                        ctx.intern_raw(op, dt, new_src, arg, tag)
                    } else {
                        n
                    };
                    // top-down: try pm on the rebuilt node, use result as-is
                    let new_n = if self.pm.is_some() { self.pm_rewrite(ctx, new_n).unwrap_or(new_n) } else { new_n };
                    self.replace.insert(n, new_n);
                }
            }
        }
        Ok(self.replace.get(&root).copied().unwrap_or(root))
    }

    /// the main driver (ops.py:1675-1741).
    fn unified_rewrite(&mut self, ctx: &mut UOpCtx, root: UopId) -> Result<UopId, String> {
        #[derive(Clone, Copy, PartialEq)]
        enum Stage {
            S0,
            S1,
            S2,
        }
        let mut stack: Vec<(UopId, Stage, UopId)> = vec![(root, Stage::S0, root)];
        let mut on_stack: HashSet<UopId> = HashSet::from([root]);
        let mut waitlist: HashMap<UopId, Vec<(UopId, Stage, UopId)>> = HashMap::new();
        while let Some((n, stage, new_n)) = stack.pop() {
            if stack.len() > REWRITE_STACK_LIMIT {
                return Err("infinite loop in graph_rewrite (stack too big)".into());
            }
            if self.replace.contains_key(&n) {
                continue;
            }
            match stage {
                Stage::S0 => {
                    let mut new_n = new_n;
                    if let Some(_bpm) = self.bpm {
                        // apply rewrite rules until a fixed point is reached
                        let mut test_n: Option<UopId> = Some(new_n);
                        let mut seen: HashSet<UopId> = HashSet::new();
                        let mut gated: Option<UopId> = None;
                        while let Some(tn) = test_n {
                            if seen.contains(&tn) {
                                return Err("infinite loop in fixed_point_rewrite".into());
                            }
                            seen.insert(tn);
                            new_n = tn;
                            match self.cached_bpm_rewrite(ctx, tn) {
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
                            self.replace.insert(n, tn);
                            if let Some(w) = waitlist.remove(&n) {
                                stack.extend(w);
                            }
                            continue;
                        }
                    }
                    stack.push((n, Stage::S1, new_n));
                    if !self.enter_calls && is_opaque_body_call(ctx, new_n) {
                        let body = ctx.srcs(new_n)[0];
                        self.replace.insert(body, body);
                    }
                    for &x in ctx.srcs(new_n).iter().rev() {
                        if on_stack.contains(&x) {
                            continue;
                        }
                        stack.push((x, Stage::S0, x));
                        on_stack.insert(x);
                    }
                }
                Stage::S1 => {
                    let srcs = ctx.srcs(new_n).to_vec();
                    let mut tmp = Vec::with_capacity(srcs.len());
                    let mut ready = true;
                    for &x in &srcs {
                        match self.replace.get(&x) {
                            Some(&rx) => tmp.push(rx),
                            None => {
                                waitlist.entry(x).or_default().push((n, Stage::S1, new_n));
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
                        let rewritten = if self.pm.is_some() { self.pm_rewrite(ctx, new_n) } else { None };
                        match rewritten {
                            None => {
                                self.replace.insert(n, new_n);
                                if let Some(w) = waitlist.remove(&n) {
                                    stack.extend(w);
                                }
                                continue;
                            }
                            Some(new_src_n) => {
                                stack.push((n, Stage::S2, new_src_n));
                                stack.push((new_src_n, Stage::S0, new_src_n));
                            }
                        }
                    } else {
                        let (op, _, _, arg, tag) = ctx.snapshot(new_n);
                        let dt = rebuild_dtype(ctx, new_n, &new_src);
                        let new_src_n = ctx.intern_raw(op, dt, new_src, arg, tag);
                        stack.push((n, Stage::S2, new_src_n));
                        stack.push((new_src_n, Stage::S0, new_src_n));
                    }
                }
                Stage::S2 => {
                    match self.replace.get(&new_n) {
                        None => {
                            waitlist.entry(new_n).or_default().push((n, Stage::S2, new_n));
                        }
                        Some(&rn) => {
                            self.replace.insert(n, rn);
                            if let Some(w) = waitlist.remove(&n) {
                                stack.extend(w);
                            }
                        }
                    }
                }
            }
        }
        Ok(self.replace[&root])
    }
}

/// `graph_rewrite` (ops.py:1743-1746).
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
    if walk {
        rc.walk_rewrite(ctx, sink)
    } else {
        rc.unified_rewrite(ctx, sink)
    }
}

/// `UOp.substitute` (ops.py:529-534) — bottom-up rewrite with `_substitute`.
pub fn substitute(ctx: &mut UOpCtx, root: UopId, dvars: HashMap<UopId, UopId>, walk: bool, enter_calls: bool) -> UopId {
    let dvars: HashMap<UopId, UopId> = dvars.into_iter().filter(|(k, v)| k != v).collect();
    if dvars.is_empty() {
        return root;
    }
    let pm = substitute_pm();
    graph_rewrite(ctx, root, &pm, true, None, walk, enter_calls, CtxVal::Subst(dvars))
        .unwrap_or_else(|e| panic!("TGERR: substitute failed: {e}"))
}

static SUBSTITUTE_PM: OnceLock<PatternMatcher> = OnceLock::new();

pub fn substitute_pm() -> &'static PatternMatcher {
    SUBSTITUTE_PM.get_or_init(|| {
        let pat = UPat::op_var((0..=79u16).filter_map(Ops::from_u16).collect()).named("x");
        PatternMatcher::new(vec![(pat, Arc::new(|_, x, ctxv: &CtxVal| ctxv.get_subst(sget(x, "x"))))])
    })
}

/// `UOp.simplify` (ops.py:510-516) with `symbolic` (sym.rs).
pub fn simplify(ctx: &mut UOpCtx, u: UopId) -> UopId {
    if ctx.op_of(u) == Ops::Const {
        return u;
    }
    if ctx.op_of(u) == Ops::Sink {
        let ok = ctx.srcs(u).iter().all(|&s| ctx.op_of(s) == Ops::Const || {
            ctx.op_of(s) == Ops::Stack && ctx.srcs(s).is_empty()
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
                    crate::dtype::ConstVal::Big(b) => b.as_i64().map(|v| v as i128).unwrap_or(if b.is_negative() { i128::MIN / 2 } else { i128::MAX / 2 }),
                    crate::dtype::ConstVal::Float(f) => *f as i128,
                    crate::dtype::ConstVal::Invalid => 0,
                };
                SVal::I(i)
            }
            SymVal::U(u) => SVal::U(*u),
        }
    }
}

/// `ssimplify` (ops.py:517-519): simplify, then const-ify `CAST(CONST)`/`CONST`.
pub fn ssimplify(ctx: &mut UOpCtx, u: UopId) -> SymVal {
    let ret = simplify(ctx, u);
    let (op, dt, srcs, _, _) = ctx.snapshot(ret);
    if op == Ops::Cast && ctx.op_of(srcs[0]) == Ops::Const {
        let v = crate::dtype::dtype_const(dt, ctx.val(srcs[0]));
        return SymVal::C(v);
    }
    if op == Ops::Const {
        return SymVal::C(ctx.val(ret));
    }
    SymVal::U(ret)
}

/// `resolve` (ops.py:54-58).
pub fn resolve(ctx: &mut UOpCtx, x: UopId, default: bool) -> bool {
    if !is_bool(ctx.dt_of(x)) {
        panic!("TGERR: UOp in resolve must be bool");
    }
    let sx = simplify(ctx, x);
    let (vmin, vmax) = ctx.min_max(sx).unwrap_or_else(|e| panic!("TGERR: {e}"));
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
