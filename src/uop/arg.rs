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

/// A non-finite float arg, encoded separately so bit identity survives JSON.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, SerJson, DeJson)]
pub enum NonFinite {
    /// Canonical NaN.
    Nan,
    /// Positive infinity.
    PosInf,
    /// Negative infinity.
    NegInf,
}

/// Which Python `IntEnum` class an `EnumVal` belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, SerJson, DeJson)]
pub enum EnumClass {
    /// `tinygrad.uop.Ops`.
    Ops,
    /// `tinygrad.uop.ops.AxisType`.
    AxisType,
    /// `tinygrad.dtype.AddrSpace`.
    AddrSpace,
}

/// An enum member arg, carried as class tag plus numeric value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, SerJson, DeJson)]
pub struct EnumVal {
    /// The enum class this member came from.
    pub class: EnumClass,
    /// The member's numeric discriminant.
    pub value: u16,
}

/// The value space of a UOp `arg` (Python's untyped `arg: Any`).
#[derive(Clone, Debug)]
pub enum Arg {
    /// Python `None`.
    None,
    /// Python `int` that fits in `i64`.
    Int(i64),
    /// Python `int` above `i64::MAX`, kept unsigned.
    UInt(u64),
    /// Python finite `float`, compared and hashed by bit pattern.
    Float(f64),
    /// Python non-finite `float`.
    NonFinite(NonFinite),
    /// Python `bool`.
    Bool(bool),
    /// Python `str`.
    Str(String),
    /// Python `bytes`.
    Bytes(Vec<u8>),
    /// A Python enum member.
    Enum(EnumVal),
    /// A `DType` arg.
    Dtype(DTypeId),
    /// Python `tuple`/`list` of args.
    Seq {
        /// Whether the Python object is a `tuple` rather than a `list`.
        tuple: bool,
        /// The elements, in order.
        items: Vec<Arg>,
    },
    /// Python `ParamArg` dataclass (BUFFER/PARAM args): the Variables/buffers spine.
    Param(ParamArg),
    /// A CONST arg carried as an exact `ConstVal` (ints may exceed i64, and
    /// `Invalid` needs a first-class variant). Non-const args use the scalar
    /// variants; the node `op` is CONST when this appears.
    Const(ConstVal),
    /// Escape hatch: pickle bytes, opaque to Rust until we model this arg.
    Pickle(Vec<u8>),
    /// A named python dataclass arg (KernelInfo, CallInfo, BufferizeOpts, Opt):
    /// tag + named fields, rebuilt to the real dataclass on the bridge.
    Data(DataArg),
}

/// Generic dataclass-shaped arg: every field is an Arg (None/Bool/Int/Str/Enum/
/// Seq/Data). Equality is structural, matching python dataclass __eq__ for the
/// fields we carry.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DataArg {
    /// Dataclass name used to rebuild the Python object (e.g. `KernelInfo`).
    pub tag: String,
    /// Fields in Python declaration order.
    pub fields: Vec<(String, Arg)>,
}

impl DataArg {
    /// Look up a field by name, panicking when it is absent.
    pub fn get(&self, name: &str) -> &Arg {
        &self
            .fields
            .iter()
            .find(|(n, _)| n == name)
            .expect("TGERR: missing dataclass field")
            .1
    }
    /// Read a field that must be a `bool`.
    pub fn as_bool(&self, name: &str) -> bool {
        match self.get(name) {
            Arg::Bool(b) => *b,
            other => panic!("TGERR: field {name} not bool: {other:?}"),
        }
    }
    /// Read a field that must be a `str`.
    pub fn as_str(&self, name: &str) -> &str {
        match self.get(name) {
            Arg::Str(s) => s,
            other => panic!("TGERR: field {name} not str: {other:?}"),
        }
    }
    /// True when the named field is Python `None`.
    pub fn is_none(&self, name: &str) -> bool {
        matches!(self.get(name), Arg::None)
    }
}

/// `KernelInfo(name="test", axis_types=(), dont_use_locals=False, applied_opts=(), opts_to_apply=None, estimates=None, beam=0)`
pub fn kernel_info(name: &str, opts_to_apply: Option<Vec<Arg>>) -> DataArg {
    DataArg {
        tag: "KernelInfo".into(),
        fields: vec![
            ("name".into(), Arg::Str(name.into())),
            (
                "axis_types".into(),
                Arg::Seq {
                    tuple: true,
                    items: vec![],
                },
            ),
            ("dont_use_locals".into(), Arg::Bool(false)),
            (
                "applied_opts".into(),
                Arg::Seq {
                    tuple: true,
                    items: vec![],
                },
            ),
            (
                "opts_to_apply".into(),
                opts_to_apply
                    .map(|o| Arg::Seq {
                        tuple: true,
                        items: o,
                    })
                    .unwrap_or(Arg::None),
            ),
            ("beam".into(), Arg::Int(0)),
        ],
    }
}

/// `CallInfo(grad_fxn=None, name=None, precompile=False, precompile_backward=False, aux=None)`
pub fn call_info(name: Option<&str>) -> DataArg {
    DataArg {
        tag: "CallInfo".into(),
        fields: vec![
            (
                "name".into(),
                name.map(|n| Arg::Str(n.into())).unwrap_or(Arg::None),
            ),
            ("precompile".into(), Arg::Bool(false)),
            ("precompile_backward".into(), Arg::Bool(false)),
        ],
    }
}

/// `BufferizeOpts(device, addrspace=GLOBAL, removable=True)`; device is
/// str|tuple|None (LOCAL ids appear as Int once assigned).
pub fn bufferize_opts(device: Arg, addrspace: Option<AddrSpace>, removable: bool) -> DataArg {
    DataArg {
        tag: "BufferizeOpts".into(),
        fields: vec![
            ("device".into(), device),
            (
                "addrspace".into(),
                addrspace
                    .map(|a| {
                        Arg::Enum(EnumVal {
                            class: EnumClass::AddrSpace,
                            value: a as u16,
                        })
                    })
                    .unwrap_or(Arg::None),
            ),
            ("removable".into(), Arg::Bool(removable)),
        ],
    }
}

/// Device field of ParamArg: `str | tuple[str, ...] | None`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DeviceSpec {
    /// A single device name, e.g. `"CL"`.
    Single(String),
    /// A multi-device tuple of names.
    Multi(Vec<String>),
}

/// Python `ParamArg` (frozen dataclass, order=True):
/// `slot, dtype, vmin_vmax=None, multiple_of=None, name=None, addrspace=GLOBAL, axis=None, device=None, volatile=False`.
#[derive(Clone, Debug)]
pub struct ParamArg {
    /// Buffer slot this parameter occupies.
    pub slot: i64,
    /// Element dtype of the buffer.
    pub dtype: DTypeId,
    /// Optional known `(vmin, vmax)` bounds on the values.
    pub vmin_vmax: Option<(ConstVal, ConstVal)>,
    /// Optional known divisibility of the values.
    pub multiple_of: Option<i64>,
    /// Optional human-readable name.
    pub name: Option<String>,
    /// `0` encodes Python `None`; 1..4 are GLOBAL/LOCAL/REG/ALU.
    pub addrspace: Option<AddrSpace>,
    /// Optional axis index for multi-device parameters.
    pub axis: Option<i64>,
    /// Optional device (or device tuple) the buffer lives on.
    pub device: Option<DeviceSpec>,
    /// Whether the buffer is marked volatile.
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
    /// Lower bound of `vmin_vmax`, defaulting to `False` like tinygrad.
    pub fn vmin(&self) -> ConstVal {
        match &self.vmin_vmax {
            Some((v, _)) => v.clone(),
            None => ConstVal::Bool(false),
        }
    }
    /// Upper bound of `vmin_vmax`, defaulting to `True` like tinygrad.
    pub fn vmax(&self) -> ConstVal {
        match &self.vmin_vmax {
            Some((_, v)) => v.clone(),
            None => ConstVal::Bool(true),
        }
    }
}

impl Arg {
    /// True when the arg is Python `None`.
    pub fn is_none(&self) -> bool {
        matches!(self, Arg::None)
    }

    /// The arg as an `i64`, if it is an int that fits.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Arg::Int(i) => Some(*i),
            Arg::UInt(u) if *u <= i64::MAX as u64 => Some(*u as i64),
            _ => None,
        }
    }

    /// The arg as a `str`, if it is one.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Arg::Str(s) => Some(s),
            _ => None,
        }
    }

    /// The arg as a sequence: `(items, is_tuple)`.
    pub fn as_seq(&self) -> Option<(&[Arg], bool)> {
        match self {
            Arg::Seq { tuple, items } => Some((items, *tuple)),
            _ => None,
        }
    }

    /// The arg as a `ParamArg`, if it is one.
    pub fn as_param(&self) -> Option<&ParamArg> {
        match self {
            Arg::Param(p) => Some(p),
            _ => None,
        }
    }

    /// The `AddrSpace` when the arg is an `AddrSpace` enum member.
    pub fn addr_space(&self) -> Option<AddrSpace> {
        match self {
            Arg::Enum(EnumVal {
                class: EnumClass::AddrSpace,
                value,
            }) => AddrSpace::from_u8(*value as u8),
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
            (
                Seq {
                    tuple: ta,
                    items: ia,
                },
                Seq {
                    tuple: tb,
                    items: ib,
                },
            ) => ta == tb && ia == ib,
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
            Data(d) => d.hash(state),
            None => {}
        }
    }
}
