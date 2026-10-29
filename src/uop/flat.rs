//! JSON interchange for UOp graphs (nanoserde), port-side of `autorust/py/bridge.py`.
//!
//! Schema (v1):
//!   { "dtypes": [&lt;attr name&gt;...],        # name table referenced by index
//!     "nodes": [ { "op": u16, "dtype": u16, "srcs": [u32...], "arg": FlatArg,
//!                  "tag": FlatArg } ... ], # post-order (srcs < index)
//!     "roots": [u32...] }
//!
//! Non-finite floats are `{"NonFinite": "Nan"|"PosInf"|"NegInf"}` (JSON has no
//! bare NaN/Inf — verified empirically for nanoserde and miniserde).

use nanoserde::{DeJson, SerJson};
use std::collections::HashMap;

use super::arg::{Arg, DataArg, DeviceSpec, EnumVal, NonFinite, ParamArg};
use super::graph::{Graph, UopId};
use crate::dtype::{self, AddrSpace, ConstVal, DTypeId};
use crate::helpers::bigint::BigInt;

#[derive(SerJson, DeJson, Debug, Clone, PartialEq)]
pub struct FlatGraph {
    pub dtypes: Vec<String>,
    pub nodes: Vec<FlatNode>,
    pub roots: Vec<u32>,
}

#[derive(SerJson, DeJson, Debug, Clone, PartialEq)]
pub struct FlatNode {
    pub op: u16,
    pub dtype: u16,
    pub srcs: Vec<u32>,
    pub arg: FlatArg,
    pub tag: FlatArg,
}

#[derive(SerJson, DeJson, Debug, Clone, PartialEq)]
pub enum FlatArg {
    None,
    Int(i64),
    UInt(u64),
    Float(f64),
    NonFinite(NonFinite),
    Bool(bool),
    Str(String),
    Bytes(Vec<u8>),
    Enum(EnumVal),
    Dtype(u16),
    Seq {
        tuple: bool,
        items: Vec<FlatArg>,
    },
    /// Python `ParamArg` — no Option fields on the wire (nanoserde omits
    /// `None` when serializing, which would make the format non-round-trippable):
    /// presence flags lead each optional value.
    Param {
        slot: i64,
        dtype: u16,
        /// [has, vmin, vmax]
        vmin_vmax: [Box<FlatArg>; 3],
        multiple_of: [bool; 1],
        multiple_of_val: i64,
        name: [bool; 1],
        name_val: String,
        /// 0 = Python None, 1..4 = GLOBAL/LOCAL/REG/ALU
        addrspace: u8,
        axis: [bool; 1],
        axis_val: i64,
        /// [has, kind(0=single,1=multi)]
        device: [bool; 2],
        device_single: String,
        device_multi: Vec<String>,
        volatile: bool,
    },
    /// Python int beyond i64/u64 (exact, sign + little-endian limbs).
    Big {
        neg: bool,
        limbs: Vec<u64>,
    },
    /// The `Invalid` const sentinel.
    ConstInvalid,
    Pickle(Vec<u8>),
    /// Named python dataclass (KernelInfo/CallInfo/BufferizeOpts/...).
    Data {
        tag: String,
        fields: Vec<(String, FlatArg)>,
    },
}

pub fn parse_flat(json: &str) -> Result<FlatGraph, String> {
    FlatGraph::deserialize_json(json).map_err(|e| format!("{e:?}"))
}

pub fn to_string_flat(g: &FlatGraph) -> String {
    g.serialize_json()
}

// ---- FlatArg <-> Arg ----

pub fn arg_from_flat(f: &FlatArg) -> Result<Arg, String> {
    arg_from_flat_with(f, &|i| {
        Err(format!("dtype idx {i} in arg outside graph decode"))
    })
}

/// `dtmap` resolves interchange-table indices to interned `DTypeId`s (required
/// during graph decode; dtype args share the node dtype table).
pub fn arg_from_flat_with(
    f: &FlatArg,
    dtmap: &dyn Fn(u16) -> Result<DTypeId, String>,
) -> Result<Arg, String> {
    Ok(match f {
        FlatArg::None => Arg::None,
        FlatArg::Int(i) => Arg::Int(*i),
        FlatArg::UInt(u) => Arg::UInt(*u),
        FlatArg::Float(x) => {
            if x.is_nan() {
                Arg::NonFinite(NonFinite::Nan)
            } else if x.is_infinite() {
                Arg::NonFinite(if *x > 0.0 {
                    NonFinite::PosInf
                } else {
                    NonFinite::NegInf
                })
            } else {
                Arg::Float(*x)
            }
        }
        FlatArg::NonFinite(n) => Arg::NonFinite(*n),
        FlatArg::Bool(b) => Arg::Bool(*b),
        FlatArg::Str(s) => Arg::Str(s.clone()),
        FlatArg::Bytes(b) => Arg::Bytes(b.clone()),
        FlatArg::Enum(e) => Arg::Enum(*e),
        FlatArg::Dtype(i) => Arg::Dtype(dtmap(*i)?),
        FlatArg::Seq { tuple, items } => {
            let items = items
                .iter()
                .map(|i| arg_from_flat_with(i, dtmap))
                .collect::<Result<Vec<_>, _>>()?;
            Arg::Seq {
                tuple: *tuple,
                items,
            }
        }
        FlatArg::Big { neg, limbs } => {
            Arg::Const(ConstVal::Big(BigInt::from_parts(*neg, limbs.clone())))
        }
        FlatArg::ConstInvalid => Arg::Const(ConstVal::Invalid),
        FlatArg::Param {
            slot,
            dtype,
            vmin_vmax,
            multiple_of,
            multiple_of_val,
            name,
            name_val,
            addrspace,
            axis,
            axis_val,
            device,
            device_single,
            device_multi,
            volatile,
        } => {
            let dtype = dtmap(*dtype)?;
            let vv = if vmin_vmax[0] == Box::new(FlatArg::Bool(true)) {
                Some((
                    flat_to_constval(&vmin_vmax[1])?,
                    flat_to_constval(&vmin_vmax[2])?,
                ))
            } else {
                None
            };
            let addrspace = if *addrspace == 0 {
                None
            } else {
                Some(
                    AddrSpace::from_u8(*addrspace)
                        .ok_or_else(|| format!("bad addrspace {addrspace}"))?,
                )
            };
            let device = if device[0] && !device[1] {
                Some(DeviceSpec::Single(device_single.clone()))
            } else if device[0] {
                Some(DeviceSpec::Multi(device_multi.clone()))
            } else {
                None
            };
            Arg::Param(ParamArg {
                slot: *slot,
                dtype,
                vmin_vmax: vv,
                multiple_of: if multiple_of[0] {
                    Some(*multiple_of_val)
                } else {
                    None
                },
                name: if name[0] {
                    Some(name_val.clone())
                } else {
                    None
                },
                addrspace,
                axis: if axis[0] { Some(*axis_val) } else { None },
                device,
                volatile: *volatile,
            })
        }
        FlatArg::Pickle(p) => Arg::Pickle(p.clone()),
        FlatArg::Data { tag, fields } => {
            let mut fs = Vec::with_capacity(fields.len());
            for (n, f) in fields {
                fs.push((n.clone(), arg_from_flat_with(f, dtmap)?));
            }
            Arg::Data(DataArg {
                tag: tag.clone(),
                fields: fs,
            })
        }
    })
}

fn flat_to_constval(f: &FlatArg) -> Result<ConstVal, String> {
    Ok(match f {
        FlatArg::Int(i) => ConstVal::Int(*i as i128),
        FlatArg::Bool(b) => ConstVal::Bool(*b),
        FlatArg::Float(x) => ConstVal::Float(*x),
        FlatArg::NonFinite(_) => ConstVal::Float(f64::NAN),
        FlatArg::Big { neg, limbs } => ConstVal::Big(BigInt::from_parts(*neg, limbs.clone())),
        FlatArg::ConstInvalid => ConstVal::Invalid,
        _ => return Err("bad const in ParamArg vmin_vmax".into()),
    })
}

fn constval_to_flat(v: &ConstVal) -> FlatArg {
    match v {
        ConstVal::Bool(b) => FlatArg::Bool(*b),
        ConstVal::Int(i) => {
            if *i >= i64::MIN as i128 && *i <= i64::MAX as i128 {
                FlatArg::Int(*i as i64)
            } else {
                let b = BigInt::from_i128(*i);
                FlatArg::Big {
                    neg: b.is_negative(),
                    limbs: b.limbs().to_vec(),
                }
            }
        }
        ConstVal::Big(b) => FlatArg::Big {
            neg: b.is_negative(),
            limbs: b.limbs().to_vec(),
        },
        ConstVal::Float(f) => {
            if f.is_nan() {
                FlatArg::NonFinite(NonFinite::Nan)
            } else if f.is_infinite() {
                FlatArg::NonFinite(if *f > 0.0 {
                    NonFinite::PosInf
                } else {
                    NonFinite::NegInf
                })
            } else {
                FlatArg::Float(*f)
            }
        }
        ConstVal::Invalid => FlatArg::ConstInvalid,
    }
}

pub fn arg_to_flat(a: &Arg) -> FlatArg {
    arg_to_flat_with(a, None)
}

/// `dtidx` remaps internal `DTypeId`s to the output dtype-table positions
/// (must be provided when encoding a graph whose table was reindexed).
pub fn arg_to_flat_with(a: &Arg, dtidx: Option<&[u16]>) -> FlatArg {
    let dmap = |d: DTypeId| -> u16 {
        match dtidx {
            Some(m) => m[d.0 as usize],
            None => d.0,
        }
    };
    match a {
        Arg::None => FlatArg::None,
        Arg::Int(i) => FlatArg::Int(*i),
        Arg::UInt(u) => FlatArg::UInt(*u),
        Arg::Float(x) => {
            if x.is_nan() {
                FlatArg::NonFinite(NonFinite::Nan)
            } else if x.is_infinite() {
                FlatArg::NonFinite(if *x > 0.0 {
                    NonFinite::PosInf
                } else {
                    NonFinite::NegInf
                })
            } else {
                FlatArg::Float(*x)
            }
        }
        Arg::NonFinite(n) => FlatArg::NonFinite(*n),
        Arg::Bool(b) => FlatArg::Bool(*b),
        Arg::Str(s) => FlatArg::Str(s.clone()),
        Arg::Bytes(b) => FlatArg::Bytes(b.clone()),
        Arg::Enum(e) => FlatArg::Enum(*e),
        Arg::Dtype(d) => FlatArg::Dtype(dmap(*d)),
        Arg::Param(p) => {
            let vv: [Box<FlatArg>; 3] = match &p.vmin_vmax {
                None => [
                    Box::new(FlatArg::Bool(false)),
                    Box::new(FlatArg::None),
                    Box::new(FlatArg::None),
                ],
                Some((a, b)) => [
                    Box::new(FlatArg::Bool(true)),
                    Box::new(constval_to_flat(a)),
                    Box::new(constval_to_flat(b)),
                ],
            };
            let addrspace = match p.addrspace {
                None => 0,
                Some(a) => a as u8,
            };
            let device: [bool; 2] = match &p.device {
                None => [false, false],
                Some(DeviceSpec::Single(_)) => [true, false],
                Some(DeviceSpec::Multi(_)) => [true, true],
            };
            FlatArg::Param {
                slot: p.slot,
                dtype: dmap(p.dtype),
                vmin_vmax: vv,
                multiple_of: [p.multiple_of.is_some()],
                multiple_of_val: p.multiple_of.unwrap_or(0),
                name: [p.name.is_some()],
                name_val: p.name.clone().unwrap_or_default(),
                addrspace,
                axis: [p.axis.is_some()],
                axis_val: p.axis.unwrap_or(0),
                device,
                device_single: match &p.device {
                    Some(DeviceSpec::Single(s)) => s.clone(),
                    _ => String::new(),
                },
                device_multi: match &p.device {
                    Some(DeviceSpec::Multi(v)) => v.clone(),
                    _ => Vec::new(),
                },
                volatile: p.volatile,
            }
        }
        Arg::Seq { tuple, items } => FlatArg::Seq {
            tuple: *tuple,
            items: items.iter().map(|i| arg_to_flat_with(i, dtidx)).collect(),
        },
        Arg::Const(c) => constval_to_flat(c),
        Arg::Pickle(p) => FlatArg::Pickle(p.clone()),
        Arg::Data(d) => FlatArg::Data {
            tag: d.tag.clone(),
            fields: d
                .fields
                .iter()
                .map(|(n, a)| (n.clone(), arg_to_flat_with(a, dtidx)))
                .collect(),
        },
    }
}

// ---- FlatGraph <-> Graph ----

impl Graph {
    pub fn from_flat(fg: &FlatGraph) -> Result<Graph, String> {
        // map dtype names to the interned table
        let mut dtmap: Vec<DTypeId> = Vec::with_capacity(fg.dtypes.len());
        for name in &fg.dtypes {
            let d = dtype::by_attr(name).ok_or_else(|| format!("unknown dtype {name:?}"))?;
            dtmap.push(d);
        }
        let mut g = Graph::new();
        let mut idmap: Vec<Option<UopId>> = vec![None; fg.nodes.len()];
        for (i, n) in fg.nodes.iter().enumerate() {
            let op = super::ops::Ops::from_u16(n.op).ok_or_else(|| format!("bad op {}", n.op))?;
            let dtype = *dtmap
                .get(n.dtype as usize)
                .ok_or_else(|| format!("bad dtype idx {}", n.dtype))?;
            let mut srcs = Vec::with_capacity(n.srcs.len());
            for &s in &n.srcs {
                let id = idmap
                    .get(s as usize)
                    .and_then(|o| o.clone())
                    .ok_or_else(|| {
                        format!("src {s} referenced before definition (not post-order)")
                    })?;
                srcs.push(id);
            }
            let arg = arg_from_flat_with(&n.arg, &|i| {
                dtmap
                    .get(i as usize)
                    .cloned()
                    .ok_or_else(|| format!("bad dtype idx {i} in arg"))
            })?;
            let tag = arg_from_flat_with(&n.tag, &|i| {
                dtmap
                    .get(i as usize)
                    .cloned()
                    .ok_or_else(|| format!("bad dtype idx {i} in tag"))
            })?;
            let id = g.intern(op, dtype, srcs, arg, tag);
            idmap[i] = Some(id);
        }
        for &r in &fg.roots {
            let id = idmap
                .get(r as usize)
                .and_then(|o| o.clone())
                .ok_or_else(|| format!("root {r} referenced before definition"))?;
            g.roots.push(id);
        }
        Ok(g)
    }

    pub fn to_flat(&self) -> FlatGraph {
        // Only serialize nodes reachable from the roots, in post-order. Compiled
        // PROGRAM/BINARY nodes from earlier kernels are not reachable from the
        // current kernel's roots, so they are dropped here instead of being
        // re-embedded (each BINARY carries the prior arena's JSON, which would
        // otherwise make every later kernel's lib grow exponentially).
        let order: Vec<UopId> = self.toposort();
        let mut remap: HashMap<u32, u32> = HashMap::with_capacity(order.len());
        for (i, id) in order.iter().enumerate() {
            remap.insert(id.id(), i as u32);
        }
        let mut used: Vec<DTypeId> = Vec::new();
        fn collect_arg_dtypes(a: &Arg, used: &mut Vec<DTypeId>) {
            match a {
                Arg::Dtype(d) => {
                    if !used.contains(d) {
                        used.push(*d);
                    }
                }
                Arg::Param(p) => {
                    if !used.contains(&p.dtype) {
                        used.push(p.dtype);
                    }
                }
                Arg::Seq { items, .. } => {
                    for i in items {
                        collect_arg_dtypes(i, used);
                    }
                }
                Arg::Data(d) => {
                    for (_, f) in &d.fields {
                        collect_arg_dtypes(f, used);
                    }
                }
                _ => {}
            }
        }
        for id in &order {
            if !used.contains(&id.dtype) {
                used.push(id.dtype);
            }
            collect_arg_dtypes(&id.arg, &mut used);
            collect_arg_dtypes(&id.tag, &mut used);
        }
        used.sort_by_key(|d| d.0);
        let dtypes = used.iter().map(|d| d.attr().to_string()).collect();
        let mut dtidx = vec![u16::MAX; dtype::DTYPES.len()];
        for (i, d) in used.iter().enumerate() {
            dtidx[d.0 as usize] = i as u16;
        }
        let nodes = order
            .iter()
            .map(|id| FlatNode {
                op: id.op as u16,
                dtype: dtidx[id.dtype.0 as usize],
                srcs: id
                    .srcs
                    .iter()
                    .map(|s| *remap.get(&s.id()).unwrap())
                    .collect(),
                arg: arg_to_flat_with(&id.arg, Some(&dtidx)),
                tag: arg_to_flat_with(&id.tag, Some(&dtidx)),
            })
            .collect();
        let roots = self
            .roots
            .iter()
            .filter_map(|r| remap.get(&r.id()).cloned())
            .collect();
        FlatGraph {
            dtypes,
            nodes,
            roots,
        }
    }
}
