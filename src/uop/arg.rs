//! The `arg` value space for UOp nodes (Python `arg: Any`).
//!
//! M0 keeps this generic: every arg decodable from a tensor graph must fit one
//! of these variants; anything exotic travels as `Pickle` (opaque bytes,
//! reconstructed Python-side). Later milestones replace pickles with typed
//! variants as they need to *interpret* those args.
//!
//! Float identity follows `ConstFloat`: equality and hashing go through the
//! bit pattern, so `-0.0 != 0.0` and `NaN == NaN` (canonical NaN only).

use nanoserde::{DeJson, SerJson};

use crate::dtype::{AddrSpace, ConstVal, DTypeId};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, SerJson, DeJson)]
pub enum NonFinite {
    Nan,
    PosInf,
    NegInf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, SerJson, DeJson)]
pub enum EnumClass {
    Ops,
    AxisType,
    AddrSpace,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, SerJson, DeJson)]
pub struct EnumVal {
    pub class: EnumClass,
    pub value: u16,
}

#[derive(Clone, Debug)]
pub enum Arg {
    None,
    Int(i64),
    UInt(u64),
    Float(f64),
    NonFinite(NonFinite),
    Bool(bool),
    Str(String),
    Bytes(Vec<u8>),
    Enum(EnumVal),
    Dtype(DTypeId),
    Seq { tuple: bool, items: Vec<Arg> },
    /// Python `ParamArg` dataclass (BUFFER/PARAM args): the Variables/buffers spine.
    Param(ParamArg),
    /// A CONST arg carried as an exact `ConstVal` (ints may exceed i64, and
    /// `Invalid` needs a first-class variant). Non-const args use the scalar
    /// variants; the node `op` is CONST when this appears.
    Const(ConstVal),
    /// Escape hatch: pickle bytes, opaque to Rust until we model this arg.
    Pickle(Vec<u8>),
}

/// Device field of ParamArg: `str | tuple[str, ...] | None`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DeviceSpec {
    Single(String),
    Multi(Vec<String>),
}

/// Python `ParamArg` (frozen dataclass, order=True):
/// `slot, dtype, vmin_vmax=None, multiple_of=None, name=None, addrspace=GLOBAL, axis=None, device=None, volatile=False`.
#[derive(Clone, Debug)]
pub struct ParamArg {
    pub slot: i64,
    pub dtype: DTypeId,
    pub vmin_vmax: Option<(ConstVal, ConstVal)>,
    pub multiple_of: Option<i64>,
    pub name: Option<String>,
    /// `0` encodes Python `None`; 1..4 are GLOBAL/LOCAL/REG/ALU.
    pub addrspace: Option<AddrSpace>,
    pub axis: Option<i64>,
    pub device: Option<DeviceSpec>,
    pub volatile: bool,
}

impl PartialEq for ParamArg {
    fn eq(&self, other: &Self) -> bool {
        self.slot == other.slot
            && self.dtype == other.dtype
            && self.vmin_vmax == other.vmin_vmax
            && self.multiple_of == other.multiple_of
            && self.name == other.name
            && self.addrspace == other.addrspace
            && self.axis == other.axis
            && self.device == other.device
            && self.volatile == other.volatile
    }
}
impl Eq for ParamArg {}

impl std::hash::Hash for ParamArg {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.slot.hash(state);
        self.dtype.hash(state);
        match &self.vmin_vmax {
            None => 0u8.hash(state),
            Some((a, b)) => {
                1u8.hash(state);
                a.hash(state);
                b.hash(state);
            }
        }
        self.multiple_of.hash(state);
        self.name.hash(state);
        self.addrspace.hash(state);
        self.axis.hash(state);
        self.device.hash(state);
        self.volatile.hash(state);
    }
}

impl ParamArg {
    pub fn vmin(&self) -> ConstVal {
        match &self.vmin_vmax {
            Some((v, _)) => v.clone(),
            None => ConstVal::Bool(false),
        }
    }
    pub fn vmax(&self) -> ConstVal {
        match &self.vmin_vmax {
            Some((_, v)) => v.clone(),
            None => ConstVal::Bool(true),
        }
    }
}

impl Arg {
    pub fn is_none(&self) -> bool {
        matches!(self, Arg::None)
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            Arg::Int(i) => Some(*i),
            Arg::UInt(u) if *u <= i64::MAX as u64 => Some(*u as i64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Arg::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_seq(&self) -> Option<(&[Arg], bool)> {
        match self {
            Arg::Seq { tuple, items } => Some((items, *tuple)),
            _ => None,
        }
    }

    pub fn as_param(&self) -> Option<&ParamArg> {
        match self {
            Arg::Param(p) => Some(p),
            _ => None,
        }
    }

    pub fn addr_space(&self) -> Option<AddrSpace> {
        match self {
            Arg::Enum(EnumVal { class: EnumClass::AddrSpace, value }) => AddrSpace::from_u8(*value as u8),
            _ => None,
        }
    }

    /// Canonical NaN so bits-identity is stable across round-trips.
    pub fn nan() -> Arg {
        Arg::Float(f64::NAN)
    }
}

impl PartialEq for Arg {
    fn eq(&self, other: &Self) -> bool {
        use Arg::*;
        match (self, other) {
            (Float(a), Float(b)) => a.to_bits() == b.to_bits(),
            (Int(a), Int(b)) => a == b,
            (UInt(a), UInt(b)) => a == b,
            (NonFinite(a), NonFinite(b)) => a == b,
            (Bool(a), Bool(b)) => a == b,
            (Str(a), Str(b)) => a == b,
            (Bytes(a), Bytes(b)) | (Pickle(a), Pickle(b)) => a == b,
            (Enum(a), Enum(b)) => a == b,
            (Dtype(a), Dtype(b)) => a == b,
            (Param(a), Param(b)) => a == b,
            (Const(a), Const(b)) => a == b,
            (Seq { tuple: ta, items: ia }, Seq { tuple: tb, items: ib }) => ta == tb && ia == ib,
            (None, None) => true,
            _ => false,
        }
    }
}
impl Eq for Arg {}

impl std::hash::Hash for Arg {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        use Arg::*;
        std::mem::discriminant(self).hash(state);
        match self {
            Float(a) => a.to_bits().hash(state),
            Int(a) => a.hash(state),
            UInt(a) => a.hash(state),
            NonFinite(a) => a.hash(state),
            Bool(a) => a.hash(state),
            Str(a) => a.hash(state),
            Bytes(a) | Pickle(a) => a.hash(state),
            Enum(a) => a.hash(state),
            Dtype(a) => a.hash(state),
            Param(a) => a.hash(state),
            Const(a) => {
                0xC0u8.hash(state);
                a.hash(state)
            }
            Seq { tuple, items } => {
                tuple.hash(state);
                items.hash(state);
            }
            None => {}
        }
    }
}
