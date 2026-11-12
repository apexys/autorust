//! Direct SPIR-V emission for the Vulkan compute backend (`VK`).
//!
//! tinygrad has no Vulkan target, so there is no Python oracle for this file:
//! it is the same linear-UOp walk as [`crate::renderer::cstyle`] (same facts,
//! same `SPECIAL`/`RANGE`/`END`/`IF` conventions), re-encoded as a SPIR-V
//! module instead of C text. The invariants that matter:
//!
//! - `gidx{d}` is the **workgroup** id and `lidx{d}` the **local invocation**
//!   id, exactly like OpenCL's `get_group_id`/`get_local_id` (the scheduler's
//!   `global_size` is a group count, see `ProgramInfo::launch_dims`), so the
//!   same kernels and the same tuning transfer from the `CL` backend.
//! - Buffers are SSBOs (`Uniform` storage class + `BufferBlock`, the SPIR-V 1.0
//!   shape every Vulkan 1.0 driver accepts). SPIR-V has no pointer casts, so a
//!   buffer *slot* gets one descriptor binding per `(scalar type, lane count)`
//!   it is viewed as: a `float4` view is a separate `RuntimeArray<vec4<float>>`
//!   variable bound to the same `VkBuffer`. The binding table travels with the
//!   payload ([`VkModule::slots`]).
//! - Vectorization maps to SPIR-V vectors. Vulkan requires naturally aligned
//!   accesses; that is the same "ALIGNED" assumption the OpenCL backend makes
//!   when it renders `*(__global float4*)`.
//! - `bf16` never reaches the ALU: the renderer's extra matcher
//!   (`extra::pm_manual_bf16_cast`) expands bf16 arithmetic through `f32`, so
//!   BFLOAT16 only appears as a 16-bit storage/bit-pattern type here.
//!
//! The rendered seam is `String` (it is also persisted in the on-disk render
//! cache as JSON), so a module is wrapped in a small text envelope.

use std::collections::HashMap;

use crate::dtype::{self, DTypeId};
use crate::renderer::{Renderer, UopRenderer};
use crate::uop::UopId;
use crate::uop::arg::Arg;
use crate::uop::ctx::UOpCtx;
use crate::uop::ops::{Ops, group};

// ---------------- device capabilities ----------------

/// What the `VK` renderer may emit for the device it will run on.
///
/// Each flag is both a SPIR-V capability and a Vulkan device feature; the
/// device is only registered when every feature the renderer advertises is
/// present (`ops_vk::factory`), so a module never asks for more than the device
/// can execute.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Caps {
    /// `Float16` capability + `shaderFloat16`: native `f16` storage and math.
    pub f16: bool,
    /// `StorageBuffer16BitAccess`: 16-bit storage (required for `bf16`, and for
    /// `f16` storage). Not core Vulkan, so it is probed.
    pub storage16: bool,
    /// `Int64` + `shaderInt64`: 64-bit integers (and `u64`/`f64` storage).
    pub int64: bool,
    /// `Float64` + `shaderFloat64`.
    pub f64: bool,
    /// `StorageBuffer8BitAccess`: 1-byte storage (`bool`, `u8`, `i8`).
    pub storage8: bool,
    /// `shaderInt8`: 8-bit arithmetic (needed to compare/store those bytes).
    pub int8: bool,
    /// `shaderInt16`: 16-bit arithmetic (`bf16` bit patterns).
    pub int16: bool,
    /// Emit SPIR-V 1.3 with the `StorageBuffer` storage class + `Block`
    /// decoration (Vulkan 1.1+) instead of the SPIR-V 1.0 `Uniform` +
    /// `BufferBlock` form. The newer form is what shader compilers emit and is
    /// measurably faster on some drivers.
    pub spirv13: bool,
}

impl Caps {
    /// Stable tag for the kernel name, so a module built for one capability set
    /// is never executed against another.
    pub fn tag(&self) -> String {
        format!(
            "f16{}{}{}{}{}{}{}",
            self.f16 as u8,
            self.storage16 as u8,
            self.int8 as u8,
            self.int64 as u8,
            self.f64 as u8,
            self.storage8 as u8,
            self.spirv13 as u8
        )
    }
}

// ---------------- SPIR-V opcodes (only what is emitted) ----------------

mod op {
    pub const NAME: u16 = 5;
    pub const EXTENSION: u16 = 10;
    pub const EXT_INST_IMPORT: u16 = 11;
    pub const EXT_INST: u16 = 12;
    pub const MEMORY_MODEL: u16 = 14;
    pub const ENTRY_POINT: u16 = 15;
    pub const EXECUTION_MODE: u16 = 16;
    pub const CAPABILITY: u16 = 17;

    pub const TYPE_VOID: u16 = 19;
    pub const TYPE_BOOL: u16 = 20;
    pub const TYPE_INT: u16 = 21;
    pub const TYPE_FLOAT: u16 = 22;
    pub const TYPE_VECTOR: u16 = 23;
    pub const TYPE_ARRAY: u16 = 28;
    pub const TYPE_RUNTIME_ARRAY: u16 = 29;
    pub const TYPE_STRUCT: u16 = 30;
    pub const TYPE_POINTER: u16 = 32;
    pub const TYPE_FUNCTION: u16 = 33;

    pub const CONSTANT_TRUE: u16 = 41;
    pub const CONSTANT_FALSE: u16 = 42;
    pub const CONSTANT: u16 = 43;

    pub const FUNCTION: u16 = 54;
    pub const FUNCTION_END: u16 = 56;

    pub const VARIABLE: u16 = 59;
    pub const LOAD: u16 = 61;
    pub const STORE: u16 = 62;
    pub const ACCESS_CHAIN: u16 = 65;

    pub const DECORATE: u16 = 71;
    pub const MEMBER_DECORATE: u16 = 72;

    pub const VECTOR_EXTRACT_DYNAMIC: u16 = 77;
    pub const COPY_OBJECT: u16 = 83;
    pub const COMPOSITE_CONSTRUCT: u16 = 80;
    pub const COMPOSITE_EXTRACT: u16 = 81;

    pub const CONVERT_F_TO_S: u16 = 110;
    pub const CONVERT_S_TO_F: u16 = 111;
    pub const CONVERT_U_TO_F: u16 = 112;
    pub const U_CONVERT: u16 = 113;
    pub const S_CONVERT: u16 = 114;
    pub const F_CONVERT: u16 = 115;
    pub const BITCAST: u16 = 124;

    pub const S_NEGATE: u16 = 126;
    pub const F_NEGATE: u16 = 127;
    pub const I_ADD: u16 = 128;
    pub const F_ADD: u16 = 129;
    pub const I_SUB: u16 = 130;
    pub const F_SUB: u16 = 131;
    pub const I_MUL: u16 = 132;
    pub const F_MUL: u16 = 133;
    pub const U_DIV: u16 = 134;
    pub const S_DIV: u16 = 135;
    pub const F_DIV: u16 = 136;
    pub const U_MOD: u16 = 137;
    pub const S_REM: u16 = 138;
    pub const F_REM: u16 = 140;

    pub const SELECT: u16 = 169;
    pub const I_EQUAL: u16 = 170;
    pub const I_NOT_EQUAL: u16 = 171;
    pub const U_LESS_THAN: u16 = 176;
    pub const S_LESS_THAN: u16 = 177;
    pub const F_ORD_EQUAL: u16 = 180;
    pub const F_UNORD_NOT_EQUAL: u16 = 183;
    pub const F_ORD_LESS_THAN: u16 = 184;

    pub const SHIFT_RIGHT_LOGICAL: u16 = 194;
    pub const SHIFT_RIGHT_ARITHMETIC: u16 = 195;
    pub const SHIFT_LEFT_LOGICAL: u16 = 196;
    pub const BITWISE_OR: u16 = 197;
    pub const BITWISE_XOR: u16 = 198;
    pub const BITWISE_AND: u16 = 199;

    pub const CONTROL_BARRIER: u16 = 224;

    pub const PHI: u16 = 245;
    pub const LOOP_MERGE: u16 = 246;
    pub const SELECTION_MERGE: u16 = 247;
    pub const LABEL: u16 = 248;
    pub const BRANCH: u16 = 249;
    pub const BRANCH_CONDITIONAL: u16 = 250;
    pub const RETURN: u16 = 253;
}

mod cap {
    pub const SHADER: u32 = 1;
    pub const FLOAT16: u32 = 9;
    pub const FLOAT64: u32 = 10;
    pub const INT64: u32 = 11;
    pub const STORAGE_BUFFER_16: u32 = 4433;
    pub const STORAGE_BUFFER_8: u32 = 4448;
}

mod deco {
    pub const BLOCK: u32 = 2;
    pub const BUFFER_BLOCK: u32 = 3;
    pub const ARRAY_STRIDE: u32 = 6;
    pub const BUILT_IN: u32 = 11;
    pub const BINDING: u32 = 33;
    pub const DESCRIPTOR_SET: u32 = 34;
    pub const OFFSET: u32 = 35;
}

mod builtin {
    pub const WORKGROUP_ID: u32 = 26;
    pub const LOCAL_INVOCATION_ID: u32 = 27;
}

mod storage {
    pub const INPUT: u32 = 1;
    pub const UNIFORM: u32 = 2;
    pub const STORAGE_BUFFER: u32 = 12;
    pub const WORKGROUP: u32 = 4;
    pub const FUNCTION: u32 = 7;
}

/// `GLSL.std.450` instruction numbers.
mod glsl {
    pub const TRUNC: u32 = 3;
    pub const SIN: u32 = 13;
    pub const POW: u32 = 26;
    pub const EXP2: u32 = 29;
    pub const LOG2: u32 = 30;
    pub const SQRT: u32 = 31;
    pub const FMA: u32 = 50;
}

// ---------------- scalar categories ----------------

/// A scalar arithmetic category: what a dtype computes as.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cat {
    /// SPIR-V `bool` (a mask type, not a byte).
    Bool,
    /// Integer of `bits` with signedness.
    Int {
        /// Register width in bits.
        bits: u32,
        /// Two's-complement signedness.
        signed: bool,
    },
    /// IEEE float of `bits`.
    Float {
        /// Register width in bits.
        bits: u32,
    },
}

impl Cat {
    fn size(self) -> u32 {
        match self {
            Cat::Bool => 1,
            Cat::Int { bits, .. } => bits / 8,
            Cat::Float { bits } => bits / 8,
        }
    }
    fn bits(self) -> u32 {
        match self {
            Cat::Bool => 1,
            Cat::Int { bits, .. } => bits,
            Cat::Float { bits } => bits,
        }
    }
    fn is_float(self) -> bool {
        matches!(self, Cat::Float { .. })
    }
    fn is_signed(self) -> bool {
        matches!(self, Cat::Int { signed: true, .. })
    }
}

/// The category a value is *computed* in. GPU registers are at least 32 bits,
/// and Vulkan restricts 8/16-bit integer arithmetic, so narrow integers live in
/// 32-bit registers (zero/sign-extended at load, truncated at store), exactly
/// like C's integer promotion in the OpenCL backend.
fn reg_cat(cat: Cat) -> Cat {
    match cat {
        Cat::Int { bits, signed } if bits < 32 => Cat::Int { bits: 32, signed },
        other => other,
    }
}

fn is_narrow_int(cat: Cat) -> bool {
    matches!(cat, Cat::Int { bits, .. } if bits < 32)
}

/// The category a value is *stored* as. SPIR-V `bool` has no physical size, so
/// a bool buffer is a byte array (1 byte per element, like OpenCL's C `bool`).
fn storage_cat(cat: Cat) -> Cat {
    match cat {
        Cat::Bool => Cat::Int {
            bits: 8,
            signed: false,
        },
        other => other,
    }
}

fn dtype_size(dt: DTypeId) -> u32 {
    if dt == dtype::BOOL {
        1
    } else {
        dt.itemsize() as u32
    }
}

fn cat_of(dt: DTypeId, caps: &Caps) -> Cat {
    if dt == dtype::BOOL {
        return Cat::Bool;
    }
    if dt == dtype::BFLOAT16 {
        // bf16 arithmetic is expanded to f32 by the extra matcher; storage and
        // bitcasts are a 16-bit pattern (the CL renderer's `ushort` mapping).
        return Cat::Int {
            bits: 16,
            signed: false,
        };
    }
    if dt == dtype::WEAKINT {
        return Cat::Int {
            bits: 32,
            signed: true,
        };
    }
    if dtype::in_set(dtype::SINTS, dt) {
        return Cat::Int {
            bits: dt.bitsize(),
            signed: true,
        };
    }
    if dtype::in_set(dtype::UINTS, dt) {
        return Cat::Int {
            bits: dt.bitsize(),
            signed: false,
        };
    }
    if dt == dtype::FLOAT16 && caps.f16 {
        return Cat::Float { bits: 16 };
    }
    if dt == dtype::FLOAT64 && caps.f64 {
        return Cat::Float { bits: 64 };
    }
    if dtype::is_float(dt) {
        // f16 without the feature, weak floats: the pipeline casts to f32.
        return Cat::Float { bits: 32 };
    }
    panic!("TGERR: SPIR-V renderer has no scalar for dtype {dt:?}");
}

/// Whether the device can store and compute `dt` natively.
pub fn supported_dtype(dt: DTypeId, caps: &Caps) -> bool {
    if dtype::in_set(dtype::FP8S, dt) {
        return false;
    }
    if dt == dtype::BOOL || dtype::in_set(dtype::UINTS, dt) || dtype::in_set(dtype::SINTS, dt) {
        return match dtype_size(dt) {
            1 => caps.storage8 && caps.int8,
            2 => caps.storage16 && caps.int16,
            8 => caps.int64,
            _ => true,
        };
    }
    if dt == dtype::BFLOAT16 {
        return caps.storage16;
    }
    if dt == dtype::FLOAT16 {
        return caps.f16;
    }
    if dt == dtype::FLOAT64 {
        return caps.f64;
    }
    true
}

// ---------------- rendered payload ----------------

/// One rendered kernel: the descriptor binding table plus the SPIR-V module.
///
/// The text envelope is
///
/// ```text
/// vk1 <nbindings> <slot>... <lx> <ly> <lz>
/// <base64 of the little-endian SPIR-V words>
/// ```
///
/// `slots[i]` is the kernel buffer slot that descriptor binding `i` refers to;
/// a slot repeats when one buffer is viewed as several element types. The local
/// size is baked into the module (`OpExecutionMode LocalSize`) and repeated here
/// so the runtime never has to parse SPIR-V.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VkModule {
    /// Buffer slot per descriptor binding, in binding order.
    pub slots: Vec<i64>,
    /// Workgroup size.
    pub local: [u32; 3],
    /// SPIR-V words.
    pub words: Vec<u32>,
}

const VK_HEADER: &str = "vk1";

impl VkModule {
    /// Encode as the render seam's `String`.
    pub fn encode(&self) -> String {
        let mut line = format!("{VK_HEADER} {}", self.slots.len());
        for s in &self.slots {
            line.push(' ');
            line.push_str(&s.to_string());
        }
        line.push_str(&format!(
            " {} {} {}",
            self.local[0], self.local[1], self.local[2]
        ));
        let mut bytes = Vec::with_capacity(self.words.len() * 4);
        for w in &self.words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        format!("{line}\n{}", crate::helpers::base64::encode(&bytes))
    }

    /// Decode a payload rendered by [`VkModule::encode`].
    pub fn decode(src: &str) -> Option<VkModule> {
        let (line, body) = src.split_once('\n')?;
        let mut it = line.split_whitespace();
        if it.next()? != VK_HEADER {
            return None;
        }
        let n: usize = it.next()?.parse().ok()?;
        let mut slots = Vec::with_capacity(n);
        for _ in 0..n {
            slots.push(it.next()?.parse().ok()?);
        }
        let local = [
            it.next()?.parse().ok()?,
            it.next()?.parse().ok()?,
            it.next()?.parse().ok()?,
        ];
        let bytes = crate::helpers::base64::decode(body.trim())?;
        if bytes.len() % 4 != 0 {
            return None;
        }
        let words = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Some(VkModule {
            slots,
            local,
            words,
        })
    }
}

// ---------------- the emitter ----------------

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum TyKey {
    Void,
    Bool,
    Int(u32, bool),
    Float(u32),
    Vec(Cat, u32),
    Ptr(u32, u32),
    Array(u32, u32),
    RuntimeArray(u32),
    Function,
}

impl TyKey {
    fn of_cat(cat: Cat) -> TyKey {
        match cat {
            Cat::Bool => TyKey::Bool,
            Cat::Int { bits, signed } => TyKey::Int(bits, signed),
            Cat::Float { bits } => TyKey::Float(bits),
        }
    }
}

/// A byte offset, folded when static. `Id` carries the register width of the
/// index (32 or 64 bits): SPIR-V has no implicit integer promotion, so offset
/// arithmetic must stay in the width the graph produced.
#[derive(Clone, Copy)]
enum Off {
    C(i64),
    Id(u32, u32),
}

#[derive(Clone, Copy)]
enum Root {
    /// A global buffer slot (the `PARAM`'s slot number).
    Buf(i64),
    /// A `BUFFER` uop (register array or local memory), already declared.
    /// `scalar` marks a 1-element buffer, declared as a plain variable (no
    /// access chain, which is what lets the driver keep it in a register).
    Local {
        var: u32,
        cat: Cat,
        stride: u32,
        sc: u32,
        scalar: bool,
    },
}

#[derive(Clone, Copy)]
struct PtrVal {
    root: Root,
    off: Off,
}

#[derive(Clone, Copy)]
enum V {
    /// `(ssa value, category, lanes)`.
    Val(u32, Cat, u32),
    Ptr(PtrVal),
}

struct Info {
    op: Ops,
    dt: DTypeId,
    srcs: Vec<UopId>,
    numel: u32,
    addrspace: Option<dtype::AddrSpace>,
}

struct LoopFrame {
    header: u32,
    cont: u32,
    merge: u32,
    /// `(phi value, category)` of the induction variable.
    iv: Option<(u32, Cat)>,
    /// pre-allocated id of the incremented induction variable
    iv_next: u32,
    end: u32,
}

/// One loop-carried register element: the `FUNCTION` variable and element it
/// lives in, the header `OpPhi`, the id carrying the value out of the continue
/// block, its category and its SPIR-V type.
type LoopReg = (u32, i64, u32, u32, Cat, u32);

/// Emits one SPIR-V compute module from a linear UOp list.
struct Emitter<'a> {
    ctx: &'a mut UOpCtx,
    caps: Caps,
    name: String,
    info: HashMap<UopId, Info>,
    vals: HashMap<UopId, V>,
    next_id: u32,
    caps_sec: Vec<u32>,
    ext_sec: Vec<u32>,
    header: Vec<u32>,
    debug: Vec<u32>,
    annotations: Vec<u32>,
    types: Vec<u32>,
    func: Vec<u32>,
    type_cache: HashMap<TyKey, u32>,
    const_cache: HashMap<(u32, u64), u32>,
    /// Constant values of 32-bit SSA ids, so address arithmetic folds to
    /// literals (an obviously-constant index lets the driver promote a
    /// register array out of Function memory).
    known: HashMap<u32, i64>,
    /// How many times each uop is referenced (for FMA contraction).
    uses: HashMap<UopId, u32>,
    /// `BUFFER` uops whose every access is a constant index and whose stores are
    /// not inside an `IF`, so their elements can live in SSA registers instead of
    /// `Function` memory.
    promotable: HashMap<UopId, bool>,
    /// Register-resident element values: `(variable, element) -> SSA id`.
    regs: HashMap<(u32, i64), u32>,
    /// `RANGE` uop -> the `(array, element)` pairs stored in its body.
    loop_writes: HashMap<UopId, Vec<(UopId, i64)>>,
    /// Elements carried around each open loop.
    loop_regs: Vec<Vec<LoopReg>>,
    /// `(slot, cat, lanes)` -> `(variable, byte stride)`
    binding_map: HashMap<(i64, Cat, u32), (u32, u32)>,
    bindings: Vec<i64>,
    /// whether any 2-byte storage element was bound
    uses_16: bool,
    builtin_var: HashMap<u32, u32>,
    local_var: HashMap<u32, (u32, bool)>,
    /// `Function`-storage variables, emitted first in the entry block.
    locals: Vec<u32>,
    block: Vec<u32>,
    label: u32,
    terminated: bool,
    loops: Vec<LoopFrame>,
    ifs: Vec<u32>,
    glsl: u32,
    main: u32,
    void: u32,
    /// Storage class + `Block`-style decoration for SSBOs (SPIR-V 1.0 vs 1.3).
    ssbo_sc: u32,
    ssbo_block: u32,
    local_size: [u32; 3],
}

impl<'a> Emitter<'a> {
    fn id(&mut self) -> u32 {
        let v = self.next_id;
        self.next_id += 1;
        v
    }

    /// Append a raw instruction (opcode + operand words) to a section.
    fn push(sec: &mut Vec<u32>, opcode: u16, args: &[u32]) {
        sec.push(((args.len() as u32 + 1) << 16) | opcode as u32);
        sec.extend_from_slice(args);
    }

    /// Emit an instruction with `%ty %result` into the current block.
    fn d(&mut self, opcode: u16, ty: u32, args: &[u32]) -> u32 {
        let id = self.id();
        self.d_at(id, opcode, ty, args);
        id
    }

    /// Emit an instruction that defines the pre-allocated `id`.
    fn d_at(&mut self, id: u32, opcode: u16, ty: u32, args: &[u32]) {
        let mut words = Vec::with_capacity(args.len() + 3);
        words.push(ty);
        words.push(id);
        words.extend_from_slice(args);
        Self::push(&mut self.block, opcode, &words);
    }

    /// Emit an instruction without a result.
    fn e(&mut self, opcode: u16, args: &[u32]) {
        Self::push(&mut self.block, opcode, args);
    }

    // ---- types and constants ----

    fn ty(&mut self, key: TyKey) -> u32 {
        if let Some(id) = self.type_cache.get(&key) {
            return *id;
        }
        let id = self.id();
        match key {
            TyKey::Void => Self::push(&mut self.types, op::TYPE_VOID, &[id]),
            TyKey::Bool => Self::push(&mut self.types, op::TYPE_BOOL, &[id]),
            TyKey::Int(bits, signed) => {
                Self::push(&mut self.types, op::TYPE_INT, &[id, bits, signed as u32])
            }
            TyKey::Float(bits) => Self::push(&mut self.types, op::TYPE_FLOAT, &[id, bits]),
            TyKey::Vec(comp, n) => {
                let c = self.ty(TyKey::of_cat(comp));
                Self::push(&mut self.types, op::TYPE_VECTOR, &[id, c, n]);
            }
            TyKey::Ptr(sc, pointee) => {
                Self::push(&mut self.types, op::TYPE_POINTER, &[id, sc, pointee])
            }
            TyKey::Array(elem, len) => {
                let n = self.const_u32(len);
                Self::push(&mut self.types, op::TYPE_ARRAY, &[id, elem, n]);
            }
            TyKey::RuntimeArray(elem) => {
                Self::push(&mut self.types, op::TYPE_RUNTIME_ARRAY, &[id, elem])
            }
            TyKey::Function => {
                let v = self.ty(TyKey::Void);
                Self::push(&mut self.types, op::TYPE_FUNCTION, &[id, v]);
            }
        }
        self.type_cache.insert(key, id);
        id
    }

    fn cat_ty(&mut self, cat: Cat) -> u32 {
        self.ty(TyKey::of_cat(cat))
    }

    /// The SPIR-V type of a *register* value of `cat`.
    fn val_ty(&mut self, cat: Cat, lanes: u32) -> u32 {
        let cat = reg_cat(cat);
        if lanes <= 1 {
            self.cat_ty(cat)
        } else {
            self.ty(TyKey::Vec(cat, lanes))
        }
    }

    /// `f32 -> bf16` bit pattern (round to nearest even, matching
    /// `extra::cast_float_to_bf16`), in a 32-bit register.
    fn f32_to_bf16(&mut self, id: u32, lanes: u32) -> u32 {
        let u32c = Cat::Int { bits: 32, signed: false };
        let ty = self.val_ty(u32c, lanes);
        let bits = {
            let bty = ty;
            self.d(op::BITCAST, bty, &[id])
        };
        let c16 = self.const_splat(u32c, lanes, 16);
        let c1 = self.const_splat(u32c, lanes, 1);
        let c7fff = self.const_splat(u32c, lanes, 0x7fff);
        let cffff = self.const_splat(u32c, lanes, 0xffff);
        let c10000 = self.const_splat(u32c, lanes, 0x10000);
        let cmask = self.const_splat(u32c, lanes, 0x7f80_0000);
        let zero = self.const_splat(u32c, lanes, 0);
        let bty = ty;
        // `(-x & 0x7f800000) != 0`: the value is normal.
        let neg = self.d(op::I_SUB, bty, &[zero, bits]);
        let sel = self.d(op::BITWISE_AND, bty, &[neg, cmask]);
        let bty_b = self.bool_ty_or_vec(lanes);
        let is_normal = self.d(op::I_NOT_EQUAL, bty_b, &[sel, zero]);
        let sh = self.d(op::SHIFT_RIGHT_LOGICAL, bty, &[bits, c16]);
        let lo = self.d(op::BITWISE_AND, bty, &[sh, c1]);
        let sum = self.d(op::I_ADD, bty, &[bits, lo]);
        let round = self.d(op::I_ADD, bty, &[sum, c7fff]);
        let frac = self.d(op::BITWISE_AND, bty, &[bits, cffff]);
        let nonzero = self.d(op::I_NOT_EQUAL, bty_b, &[frac, zero]);
        let or = self.d(op::BITWISE_OR, bty, &[bits, c10000]);
        let sticky = self.d(op::SELECT, bty, &[nonzero, or, bits]);
        let rounded = self.d(op::SELECT, bty, &[is_normal, round, sticky]);
        let hi = self.d(op::SHIFT_RIGHT_LOGICAL, bty, &[rounded, c16]);
        // Narrow to the 16-bit pattern and widen back into the register.
        let n16 = Cat::Int {
            bits: 16,
            signed: false,
        };
        let narrow = self.narrow_to(hi, n16, lanes);
        self.to_reg(narrow, n16, lanes)
    }

    /// `bf16 -> f32`: shift the pattern into the high half.
    fn bf16_to_f32(&mut self, id: u32, lanes: u32) -> u32 {
        let u32c = Cat::Int { bits: 32, signed: false };
        let ty = self.val_ty(u32c, lanes);
        let c16 = self.const_splat(u32c, lanes, 16);
        let sh = self.d(op::SHIFT_LEFT_LOGICAL, ty, &[id, c16]);
        let fty = self.val_ty(Cat::Float { bits: 32 }, lanes);
        self.d(op::BITCAST, fty, &[sh])
    }

    fn bool_ty_or_vec(&mut self, lanes: u32) -> u32 {
        if lanes <= 1 {
            self.bool_ty()
        } else {
            self.ty(TyKey::Vec(Cat::Bool, lanes))
        }
    }

    /// Widen a narrow integer value to its register form.
    #[allow(clippy::wrong_self_convention)]
    fn to_reg(&mut self, v: u32, cat: Cat, lanes: u32) -> u32 {
        if !is_narrow_int(cat) {
            return v;
        }
        let ty = self.val_ty(cat, lanes);
        let o = if cat.is_signed() {
            op::S_CONVERT
        } else {
            op::U_CONVERT
        };
        self.d(o, ty, &[v])
    }

    fn bool_ty(&mut self) -> u32 {
        self.ty(TyKey::Bool)
    }

    fn const_u32(&mut self, v: u32) -> u32 {
        self.const_bits(
            Cat::Int {
                bits: 32,
                signed: false,
            },
            v as u64,
        )
    }

    /// `OpConstant` of `cat` from raw bits (deduplicated). `bool` uses
    /// `OpConstantTrue`/`OpConstantFalse`: SPIR-V forbids `OpConstant` on a
    /// bool type.
    fn const_bits(&mut self, cat: Cat, bits: u64) -> u32 {
        let ty = self.cat_ty(cat);
        if let Some(id) = self.const_cache.get(&(ty, bits)) {
            return *id;
        }
        let id = self.id();
        if cat == Cat::Bool {
            let o = if bits & 1 != 0 {
                op::CONSTANT_TRUE
            } else {
                op::CONSTANT_FALSE
            };
            Self::push(&mut self.types, o, &[ty, id]);
            self.const_cache.insert((ty, bits & 1), id);
            return id;
        }
        let mut args = vec![ty, id, bits as u32];
        if cat.bits() > 32 {
            args.push((bits >> 32) as u32);
        }
        Self::push(&mut self.types, op::CONSTANT, &args);
        self.const_cache.insert((ty, bits), id);
        if matches!(cat, Cat::Int { bits: 32, .. }) {
            let v = bits as u32 as i32 as i64;
            self.known.insert(id, v);
        }
        id
    }

    /// The constant value of `id`, when it is known.
    fn known_of(&self, id: u32) -> Option<i64> {
        self.known.get(&id).copied()
    }

    /// A scalar constant usable inside a block.
    ///
    /// Vulkan forbids `OpConstant` of 8/16-bit *integer* types ("cannot form
    /// constants of 8- or 16-bit types"), so narrow integers are widened from a
    /// 32-bit literal with `OpUConvert`/`OpSConvert` (exact: the value is
    /// masked). The result lives in the current block, so it is not cached
    /// across blocks.
    fn const_scalar(&mut self, cat: Cat, bits: u64) -> u32 {
        // Register form: narrow integers are 32-bit literals of the (already
        // masked) value.
        self.const_bits(reg_cat(cat), mask_bits(bits, cat))
    }

    fn const_splat(&mut self, cat: Cat, lanes: u32, bits: u64) -> u32 {
        let c = self.const_scalar(cat, bits);
        if lanes <= 1 {
            return c;
        }
        let ty = self.val_ty(cat, lanes);
        let args: Vec<u32> = std::iter::repeat_n(c, lanes as usize).collect();
        self.d(op::COMPOSITE_CONSTRUCT, ty, &args)
    }

    // ---- blocks ----

    fn start_block(&mut self, label: u32) {
        // Close the current block (it must end in a terminator), *then* open the
        // new one.
        if !self.terminated {
            Self::push(&mut self.block, op::BRANCH, &[label]);
            self.terminated = true;
        }
        self.func.append(&mut self.block);
        Self::push(&mut self.func, op::LABEL, &[label]);
        self.terminated = false;
        self.label = label;
    }

    /// End the current block unless it is already terminated.
    fn end_block(&mut self, opcode: u16, args: &[u32]) {
        if !self.terminated {
            Self::push(&mut self.block, opcode, args);
            self.terminated = true;
        }
    }

    // ---- values ----

    fn value(&mut self, u: &UopId) -> (u32, Cat, u32) {
        if let Some(v) = self.vals.get(u).copied() {
            return match v {
                V::Val(id, cat, lanes) => (id, cat, lanes),
                V::Ptr(_) => panic!(
                    "TGERR: SPIR-V renderer needs a value but uop {} is a pointer",
                    u.id()
                ),
            };
        }
        // CONSTs are not materialized by the walk (the C renderer inlines them);
        // emit one on first use.
        if self.info.get(u).map(|i| i.op) == Some(Ops::Const) {
            let v = self.const_uop(u);
            self.vals.insert(u.clone(), v);
            if let V::Val(id, cat, lanes) = v {
                return (id, cat, lanes);
            }
        }
        panic!(
            "TGERR: SPIR-V renderer has no value for uop {} (op {:?})",
            u.id(),
            self.info.get(u).map(|i| i.op)
        )
    }

    /// Materialize a `CONST` uop as an `OpConstant` (splatted to its shape).
    fn const_uop(&mut self, u: &UopId) -> V {
        let Arg::Const(val) = self.ctx.arg_of(u) else {
            panic!("TGERR: SPIR-V CONST without a value");
        };
        let dt = self.info[u].dt;
        let lanes = self.info[u].numel.max(1);
        let cat = cat_of(dt, &self.caps);
        let nodes = dtype::dtype_const(dt, val.clone());
        let bits: u64 = match (&nodes, cat) {
            (crate::dtype::ConstVal::Bool(b), Cat::Bool) => *b as u64,
            (crate::dtype::ConstVal::Bool(b), _) => *b as u64,
            (crate::dtype::ConstVal::Int(i), Cat::Float { bits: 16 }) => {
                f16_bits(*i as f32) as u64
            }
            (crate::dtype::ConstVal::Int(i), Cat::Float { bits: 32 }) => {
                (*i as f32).to_bits() as u64
            }
            (crate::dtype::ConstVal::Int(i), Cat::Float { bits: 64 }) => (*i as f64).to_bits(),
            (crate::dtype::ConstVal::Int(i), _) => *i as u64,
            (crate::dtype::ConstVal::Float(f), Cat::Int { bits: 16, .. }) => {
                bf16_bits(*f as f32) as u64
            }
            (crate::dtype::ConstVal::Float(f), Cat::Float { bits: 16 }) => {
                f16_bits(*f as f32) as u64
            }
            (crate::dtype::ConstVal::Float(f), Cat::Float { bits: 32 }) => {
                (*f as f32).to_bits() as u64
            }
            (crate::dtype::ConstVal::Float(f), Cat::Float { bits: 64 }) => f.to_bits(),
            (crate::dtype::ConstVal::Float(f), Cat::Bool) => (*f != 0.0) as u64,
            (crate::dtype::ConstVal::Big(b), _) => {
                b.as_i64().unwrap_or(0) as u64
            }
            (other, cat) => panic!("TGERR: SPIR-V const {other:?} for {cat:?}"),
        };
        let bits = mask_bits(bits, cat);
        let id = self.const_splat(cat, lanes, bits);
        V::Val(id, reg_cat(cat), lanes)
    }

    fn ptr(&self, u: &UopId) -> PtrVal {
        match self.vals.get(u) {
            Some(V::Ptr(p)) => *p,
            _ => panic!(
                "TGERR: SPIR-V renderer needs a pointer but uop {} is not one",
                u.id()
            ),
        }
    }

    fn val_id(&mut self, u: &UopId) -> u32 {
        self.value(u).0
    }

    fn cat_lanes(&self, u: &UopId) -> (Cat, u32) {
        let i = &self.info[u];
        (cat_of(i.dt, &self.caps), i.numel.max(1))
    }

    /// Broadcast a scalar value to `lanes` components (SPIR-V has no implicit
    /// scalar/vector promotion).
    fn widen(&mut self, id: u32, cat: Cat, from: u32, to: u32) -> u32 {
        if from >= to || to <= 1 {
            return id;
        }
        let ty = self.val_ty(cat, to);
        let args: Vec<u32> = std::iter::repeat_n(id, to as usize).collect();
        self.d(op::COMPOSITE_CONSTRUCT, ty, &args)
    }

    fn as_bool(&mut self, id: u32, cat: Cat, lanes: u32) -> u32 {
        if cat == Cat::Bool {
            return id;
        }
        let ty = if lanes <= 1 {
            self.bool_ty()
        } else {
            self.ty(TyKey::Vec(Cat::Bool, lanes))
        };
        let z = self.const_splat(cat, lanes, 0);
        if cat.is_float() {
            self.d(op::F_UNORD_NOT_EQUAL, ty, &[id, z])
        } else {
            self.d(op::I_NOT_EQUAL, ty, &[id, z])
        }
    }

    fn one_bits(cat: Cat) -> u64 {
        match cat {
            Cat::Bool => 1,
            Cat::Int { .. } => 1,
            Cat::Float { bits: 16 } => 0x3c00,
            Cat::Float { bits: 32 } => 0x3f80_0000,
            Cat::Float { bits: 64 } => 0x3ff0_0000_0000_0000,
            Cat::Float { bits } => panic!("TGERR: no float of {bits} bits"),
        }
    }

    /// Numeric cast (`Ops::Cast`). `from` is a register value, so narrow
    /// integers are already widened; a narrow *result* is truncated to its dtype
    /// and then zero/sign-extended back into the register (so `f32 -> u8 -> f32`
    /// wraps like the interpreter).
    fn cast_num(&mut self, id: u32, from: Cat, to: Cat, lanes: u32) -> u32 {
        let from = reg_cat(from);
        if from == to {
            return id;
        }
        let narrow = is_narrow_int(to);
        let ty = if narrow {
            self.storage_ty(to, lanes)
        } else {
            self.val_ty(to, lanes)
        };
        let out = match (from, to) {
            (Cat::Bool, _) => {
                let one = self.const_splat(to, lanes, Self::one_bits(to));
                let zero = self.const_splat(to, lanes, 0);
                self.d(op::SELECT, ty, &[id, one, zero])
            }
            (_, Cat::Bool) => self.as_bool(id, from, lanes),
            (Cat::Float { .. }, Cat::Int { signed, .. }) => {
                // tinygrad's `truncate` truncates toward zero and then *wraps*
                // to the target width (`c_ubyte(300.0) == 44`), so convert to
                // the widest int first and narrow from there.
                let wide = if self.caps.int64 { 64 } else { 32 };
                let wty = self.cat_ty(Cat::Int { bits: wide, signed: true });
                let w = self.d(op::CONVERT_F_TO_S, wty, &[id]);
                // Unsigned destinations clamp negatives to 0 first (the interp's
                // C-cast model), then wrap to the target width.
                let w = if signed {
                    w
                } else {
                    let zero = self.const_bits(Cat::Int { bits: wide, signed: true }, 0);
                    let bty = self.bool_ty();
                    let neg = self.d(op::S_LESS_THAN, bty, &[w, zero]);
                    self.d(op::SELECT, wty, &[neg, zero, w])
                };
                let o = if signed {
                    op::S_CONVERT
                } else {
                    op::U_CONVERT
                };
                self.d(o, ty, &[w])
            }
            (Cat::Int { signed, .. }, Cat::Float { .. }) => {
                let o = if signed {
                    op::CONVERT_S_TO_F
                } else {
                    op::CONVERT_U_TO_F
                };
                self.d(o, ty, &[id])
            }
            (Cat::Int { .. }, Cat::Int { .. }) => {
                if from.bits() == to.bits() {
                    self.d(op::BITCAST, ty, &[id])
                } else if from.is_signed() {
                    self.d(op::S_CONVERT, ty, &[id])
                } else {
                    self.d(op::U_CONVERT, ty, &[id])
                }
            }
            (Cat::Float { .. }, Cat::Float { .. }) => self.d(op::F_CONVERT, ty, &[id]),
        };
        self.to_reg(out, to, lanes)
    }

    /// Bit reinterpretation (`Ops::Bitcast`).
    ///
    /// Values live in register form, so a bitcast between two narrow integer
    /// types (e.g. `bf16` -> `u16`) is the identity: both sides are the same
    /// 16-bit pattern zero-extended into a 32-bit register.
    fn bitcast_val(&mut self, id: u32, from: Cat, to: Cat, lanes: u32) -> u32 {
        let (fr, tr) = (reg_cat(from), reg_cat(to));
        if fr == tr {
            return id;
        }
        if from.bits() != to.bits() {
            panic!("TGERR: SPIR-V bitcast of unequal widths ({from:?} -> {to:?})");
        }
        if fr == Cat::Bool {
            let ty = self.val_ty(to, lanes);
            let one = self.const_splat(to, lanes, Self::one_bits(to));
            let zero = self.const_splat(to, lanes, 0);
            return self.d(op::SELECT, ty, &[id, one, zero]);
        }
        if tr == Cat::Bool {
            return self.as_bool(id, reg_cat(from), lanes);
        }
        let ty = self.val_ty(to, lanes);
        self.d(op::BITCAST, ty, &[id])
    }

    // ---- byte offsets ----

    fn off_add(&mut self, a: Off, b: Off) -> Off {
        if let (Some(x), Some(y)) = (self.off_known(a), self.off_known(b)) {
            return Off::C(x + y);
        }
        if let (Off::C(0), v) | (v, Off::C(0)) = (a, b) {
            return v;
        }
        let w = self.off_width(a).max(self.off_width(b));
        let x = self.off_widen(a, w);
        let y = self.off_widen(b, w);
        let ty = self.ty(TyKey::Int(w, true));
        Off::Id(self.d(op::I_ADD, ty, &[x, y]), w)
    }

    fn off_scale(&mut self, a: Off, k: i64) -> Off {
        if k == 1 {
            return a;
        }
        if let Some(v) = self.off_known(a) {
            return Off::C(v * k);
        }
        let w = self.off_width(a);
        let id = self.off_widen(a, w);
        let ty = self.ty(TyKey::Int(w, true));
        let kc = self.const_bits(Cat::Int { bits: w, signed: true }, k as u64);
        Off::Id(self.d(op::I_MUL, ty, &[id, kc]), w)
    }

    fn off_width(&self, o: Off) -> u32 {
        match o {
            Off::C(_) => 32,
            Off::Id(_, w) => w,
        }
    }

    /// `o` as an integer SSA value of width `w` (widening with `OpSConvert`).
    fn off_widen(&mut self, o: Off, w: u32) -> u32 {
        let (id, have) = match o {
            Off::C(v) => {
                return self.const_bits(Cat::Int { bits: w, signed: true }, v as u64);
            }
            Off::Id(id, have) => (id, have),
        };
        if have == w {
            return id;
        }
        let ty = self.ty(TyKey::Int(w, true));
        self.d(op::S_CONVERT, ty, &[id])
    }

    /// The constant value of an offset, when known.
    fn off_known(&self, o: Off) -> Option<i64> {
        match o {
            Off::C(v) => Some(v),
            Off::Id(id, _) => self.known_of(id),
        }
    }

    /// `byte_off / stride` (stride is a power of two).
    fn elem_index(&mut self, off: Off, stride: u32) -> u32 {
        let w = self.off_width(off);
        if let Some(v) = self.off_known(off) {
            return self.const_bits(Cat::Int { bits: w, signed: false }, (v / stride as i64) as u64);
        }
        let bits = stride.trailing_zeros();
        let o = self.off_widen(off, w);
        if bits == 0 {
            return o;
        }
        let shift = self.const_bits(Cat::Int { bits: w, signed: false }, bits as u64);
        let ty = self.ty(TyKey::Int(w, false));
        self.d(op::SHIFT_RIGHT_LOGICAL, ty, &[o, shift])
    }

    fn access_chain(&mut self, var: u32, ptr_ty: u32, index: u32, two_level: bool) -> u32 {
        let zero = self.const_u32(0);
        if two_level {
            self.d(op::ACCESS_CHAIN, ptr_ty, &[var, zero, index])
        } else {
            self.d(op::ACCESS_CHAIN, ptr_ty, &[var, index])
        }
    }

    // ---- buffer bindings ----

    fn binding(&mut self, slot: i64, cat: Cat, lanes: u32) -> (u32, u32) {
        let cat = storage_cat(cat);
        if let Some(b) = self.binding_map.get(&(slot, cat, lanes)) {
            return *b;
        }
        let lanes = lanes.max(1);
        let stride = cat.size() * lanes;
        if stride == 2 {
            self.uses_16 = true;
        }
        let elem = self.storage_ty(cat, lanes);
        let fresh = !self.type_cache.contains_key(&TyKey::RuntimeArray(elem));
        let arr = self.ty(TyKey::RuntimeArray(elem));
        // One struct per binding: the BufferBlock decoration is per type, so
        // bindings cannot share it.
        let st = self.id();
        Self::push(&mut self.types, op::TYPE_STRUCT, &[st, arr]);
        let ptr = self.ty(TyKey::Ptr(self.ssbo_sc, st));
        let var = self.id();
        Self::push(&mut self.types, op::VARIABLE, &[ptr, var, self.ssbo_sc]);
        let idx = self.bindings.len() as u32;
        Self::push(&mut self.annotations, op::DECORATE, &[st, self.ssbo_block]);
        Self::push(
            &mut self.annotations,
            op::MEMBER_DECORATE,
            &[st, 0, deco::OFFSET, 0],
        );
        if fresh {
            Self::push(
                &mut self.annotations,
                op::DECORATE,
                &[arr, deco::ARRAY_STRIDE, stride],
            );
        }
        Self::push(
            &mut self.annotations,
            op::DECORATE,
            &[var, deco::DESCRIPTOR_SET, 0],
        );
        Self::push(&mut self.annotations, op::DECORATE, &[var, deco::BINDING, idx]);
        self.bindings.push(slot);
        self.binding_map.insert((slot, cat, lanes), (var, stride));
        (var, stride)
    }

    /// The storage type of a `(cat, lanes)` access (no register widening).
    fn storage_ty(&mut self, cat: Cat, lanes: u32) -> u32 {
        let cat = storage_cat(cat);
        if lanes <= 1 {
            self.cat_ty(cat)
        } else {
            self.ty(TyKey::Vec(cat, lanes))
        }
    }

    /// Truncate a register value to the `cat` storage width (per lane when
    /// `lanes > 1`).
    fn narrow_to(&mut self, v: u32, cat: Cat, lanes: u32) -> u32 {
        if !is_narrow_int(cat) {
            return v;
        }
        let ty = self.storage_ty(cat, lanes);
        let o = if cat.is_signed() {
            op::S_CONVERT
        } else {
            op::U_CONVERT
        };
        self.d(o, ty, &[v])
    }

    /// Load `(cat, lanes)` through `p`. The value is returned in register form
    /// (narrow integers zero/sign-extended, byte-stored bools as a mask).
    fn load_ptr(&mut self, p: PtrVal, cat: Cat, lanes: u32) -> u32 {
        let lanes = lanes.max(1);
        let scat = storage_cat(cat);
        if lanes == 1
            && !is_narrow_int(scat)
            && let Some((var, k)) = self.promoted_elem(&p)
            && let Some(v) = self.regs.get(&(var, k)).copied()
        {
            return v;
        }
        let raw = self.load_raw(p, scat, lanes);
        let reg = self.to_reg(raw, scat, lanes);
        if cat == Cat::Bool {
            self.as_bool(reg, reg_cat(scat), lanes)
        } else {
            reg
        }
    }

    fn load_raw(&mut self, p: PtrVal, cat: Cat, lanes: u32) -> u32 {
        let lanes = lanes.max(1);
        match p.root {
            Root::Buf(slot) => {
                let (var, stride) = self.binding(slot, cat, lanes);
                let ty = self.storage_ty(cat, lanes);
                let ptr_ty = self.ty(TyKey::Ptr(self.ssbo_sc, ty));
                let idx = self.elem_index(p.off, stride);
                let ptr = self.access_chain(var, ptr_ty, idx, true);
                self.d(op::LOAD, ty, &[ptr])
            }
            Root::Local {
                var,
                cat: bcat,
                stride,
                sc,
                scalar,
            } => {
                if lanes == 1 && cat == bcat {
                    let ty = self.cat_ty(cat);
                    if scalar {
                        return self.d(op::LOAD, ty, &[var]);
                    }
                    let ptr_ty = self.ty(TyKey::Ptr(sc, ty));
                    let idx = self.elem_index(p.off, stride);
                    let ptr = self.access_chain(var, ptr_ty, idx, false);
                    return self.d(op::LOAD, ty, &[ptr]);
                }
                let elem_sz = cat.size();
                let mut ids = Vec::with_capacity(lanes as usize);
                for l in 0..lanes {
                    let off = self.off_add(p.off, Off::C((l * elem_sz) as i64));
                    let ty = self.cat_ty(bcat);
                    let ptr_ty = self.ty(TyKey::Ptr(sc, ty));
                    let idx = self.elem_index(off, stride);
                    let ptr = self.access_chain(var, ptr_ty, idx, false);
                    let v = self.d(op::LOAD, ty, &[ptr]);
                    let v = if bcat == cat {
                        v
                    } else if bcat.bits() == cat.bits() {
                        let ty = self.cat_ty(cat);
                        self.d(op::BITCAST, ty, &[v])
                    } else if bcat.bits() > cat.bits() {
                        let ty = self.cat_ty(cat);
                        let o = if bcat.is_signed() {
                            op::S_CONVERT
                        } else {
                            op::U_CONVERT
                        };
                        self.d(o, ty, &[v])
                    } else {
                        panic!("TGERR: SPIR-V local-memory view wider than its array element");
                    };
                    ids.push(v);
                }
                let ty = self.storage_ty(cat, lanes);
                self.d(op::COMPOSITE_CONSTRUCT, ty, &ids)
            }
        }
    }

    /// Store `(cat, lanes)` through `p`. `val` is a register value.
    fn store_ptr(&mut self, p: PtrVal, cat: Cat, lanes: u32, val: u32) {
        let lanes = lanes.max(1);
        if lanes == 1
            && cat != Cat::Bool
            && !is_narrow_int(cat)
            && let Some((var, k)) = self.promoted_elem(&p)
        {
            self.regs.insert((var, k), val);
            return;
        }
        // `bool` is stored as a byte: convert the mask to 0/1 in register form.
        let (cat, val) = match cat {
            Cat::Bool => {
                let s = storage_cat(cat);
                let rty = self.val_ty(s, lanes);
                let one = self.const_splat(s, lanes, 1);
                let zero = self.const_splat(s, lanes, 0);
                (s, self.d(op::SELECT, rty, &[val, one, zero]))
            }
            other => (other, val),
        };
        // Narrow the register value to the storage width. A vector of 8/16-bit
        // elements cannot be built with `OpCompositeConstruct` (Vulkan forbids
        // composites of those types), so the narrowed lanes are assembled in
        // register width and converted once.
        let val = if lanes == 1 {
            self.narrow_to(val, cat, 1)
        } else if is_narrow_int(cat) {
            let lane_ty = self.val_ty(reg_cat(cat), 1);
            let mut parts = Vec::with_capacity(lanes as usize);
            for l in 0..lanes {
                let ex = self.d(op::COMPOSITE_EXTRACT, lane_ty, &[val, l]);
                let nar = self.narrow_to(ex, cat, 1);
                parts.push(self.to_reg(nar, cat, 1));
            }
            let vty = self.val_ty(reg_cat(cat), lanes);
            let wide = self.d(op::COMPOSITE_CONSTRUCT, vty, &parts);
            let sty = self.storage_ty(cat, lanes);
            self.d(op::U_CONVERT, sty, &[wide])
        } else {
            val
        };
        match p.root {
            Root::Buf(slot) => {
                let (var, stride) = self.binding(slot, cat, lanes);
                let ty = self.storage_ty(cat, lanes);
                let ptr_ty = self.ty(TyKey::Ptr(self.ssbo_sc, ty));
                let idx = self.elem_index(p.off, stride);
                let ptr = self.access_chain(var, ptr_ty, idx, true);
                self.e(op::STORE, &[ptr, val]);
            }
            Root::Local {
                var,
                cat: bcat,
                stride,
                sc,
                scalar,
            } => {
                if lanes == 1 && cat == bcat {
                    let ty = self.cat_ty(cat);
                    if scalar {
                        self.e(op::STORE, &[var, val]);
                        return;
                    }
                    let ptr_ty = self.ty(TyKey::Ptr(sc, ty));
                    let idx = self.elem_index(p.off, stride);
                    let ptr = self.access_chain(var, ptr_ty, idx, false);
                    self.e(op::STORE, &[ptr, val]);
                    return;
                }
                let elem_sz = cat.size();
                for l in 0..lanes {
                    let off = self.off_add(p.off, Off::C((l * elem_sz) as i64));
                    let ty = self.cat_ty(bcat);
                    let ptr_ty = self.ty(TyKey::Ptr(sc, ty));
                    let idx = self.elem_index(off, stride);
                    let ptr = self.access_chain(var, ptr_ty, idx, false);
                    let lane_ty = self.cat_ty(cat);
                    let v = if lanes == 1 {
                        val
                    } else {
                        self.d(op::COMPOSITE_EXTRACT, lane_ty, &[val, l])
                    };
                    let v = if bcat == cat {
                        v
                    } else if bcat.bits() == cat.bits() {
                        self.d(op::BITCAST, ty, &[v])
                    } else if bcat.bits() > cat.bits() {
                        let o = if bcat.is_signed() {
                            op::S_CONVERT
                        } else {
                            op::U_CONVERT
                        };
                        self.d(o, ty, &[v])
                    } else {
                        panic!("TGERR: SPIR-V local-memory store wider than its array element");
                    };
                    self.e(op::STORE, &[ptr, v]);
                }
            }
        }
    }

    fn builtin_input(&mut self, which: u32) -> u32 {
        if let Some(v) = self.builtin_var.get(&which) {
            return *v;
        }
        let uvec3 = self.ty(TyKey::Vec(
            Cat::Int {
                bits: 32,
                signed: false,
            },
            3,
        ));
        let ptr = self.ty(TyKey::Ptr(storage::INPUT, uvec3));
        let var = self.id();
        Self::push(&mut self.types, op::VARIABLE, &[ptr, var, storage::INPUT]);
        Self::push(&mut self.annotations, op::DECORATE, &[var, deco::BUILT_IN, which]);
        self.builtin_var.insert(which, var);
        var
    }

    // ---- walks ----

    /// Unroll factor for hot, small, constant-trip-count loops.
    ///
    /// The NVIDIA SPIR-V path does not unroll a 1024-iteration reduce loop,
    /// while its OpenCL compiler unrolls the *same* loop 4x (measured: a 64 MiB
    /// row-reduce takes 0.78 ms un-unrolled and 0.26 ms with a manual 4x
    /// unroll, matching OpenCL's 0.26 ms). The extra loads in flight are what
    /// makes a strided reduce reach memory bandwidth, so unroll here where the
    /// body is small enough not to blow up register pressure.
    const UNROLL: usize = 8;

    /// The index of the `END` matching the `RANGE` at `u`, when that loop
    /// should be unrolled.
    fn unroll_end(&mut self, u: &UopId, lst: &[UopId]) -> Option<usize> {
        let info = self.info.get(u)?;
        if info.op != Ops::Range || info.dt == dtype::VOID {
            return None;
        }
        // Constant trip count. A remainder is fine: `unroll_loop` runs
        // `ext / UNROLL` full chunks and then emits the leftover iterations as
        // straight-line copies, which keeps the order of loads and of the
        // loop-carried accumulator identical.
        let Ok(crate::uop::sint::SVal::I(ext)) = self.ctx.ssimplify(info.srcs[0].clone()) else {
            return None;
        };
        let ext = ext as i64;
        if ext < (2 * Self::UNROLL) as i64 {
            return None;
        }
        // Find the matching END (nested RANGE/IF depth).
        let start = lst.iter().position(|x| x.id() == u.id())?;
        let mut depth = 0usize;
        let mut end = None;
        for (i, x) in lst.iter().enumerate().skip(start) {
            match self.info.get(x).map(|i| i.op) {
                Some(Ops::Range) | Some(Ops::If) => depth += 1,
                Some(Ops::End) | Some(Ops::Endif) => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let end = end?;
        // Only unroll small bodies that actually read memory; a large fused
        // body (the GEMM reduce) would spill registers instead.
        let body = &lst[start + 1..end];
        if crate::helpers::env::get("VK_TRACE").is_some() {
            let reads = body
                .iter()
                .filter(|x| self.info.get(x).map(|i| i.op == Ops::Load).unwrap_or(false))
                .count();
            eprintln!(
                "[vk] loop {ext}: body {} uops, {reads} loads",
                body.len()
            );
        }
        if body.len() > 48 {
            return None;
        }
        let reads = body.iter().any(|x| {
            self.info
                .get(x)
                .map(|i| i.op == Ops::Load)
                .unwrap_or(false)
        });
        reads.then_some(end)
    }

    /// Emit `range` with its body duplicated `UNROLL` times and a step of
    /// `UNROLL` (the trip count is a multiple of it, checked by
    /// [`Self::unroll_end`]).
    fn unroll_loop(&mut self, range: &UopId, body: &[UopId], end: &UopId) {
        let cat = cat_of(self.info[range].dt, &self.caps);
        let end_v = self.val_id(&self.info[range].srcs[0].clone());
        let end_v = if cat_of(self.info[&self.info[range].srcs[0]].dt, &self.caps) == cat {
            end_v
        } else {
            let fcat = cat_of(self.info[&self.info[range].srcs[0]].dt, &self.caps);
            self.cast_num(end_v, fcat, cat, 1)
        };
        let pred = self.label;
        let hdr = self.id();
        let body_l = self.id();
        let cont = self.id();
        let merge = self.id();
        self.end_block(op::BRANCH, &[hdr]);
        self.start_block(hdr);
        let vty = self.cat_ty(cat);
        let zero = self.const_bits(cat, 0);
        let iv_next = self.id();
        let phi = self.d(op::PHI, vty, &[zero, pred, iv_next, cont]);
        let carried = self.open_loop_regs(range, pred, cont);
        self.loop_regs.push(carried);
        self.e(op::LOOP_MERGE, &[merge, cont, 0]);
        self.end_block(op::BRANCH, &[body_l]);
        self.start_block(body_l);
        for k in 0..Self::UNROLL {
            let iv = if k == 0 {
                phi
            } else {
                let kc = self.const_bits(cat, k as u64);
                self.d(op::I_ADD, vty, &[phi, kc])
            };
            self.vals.insert(
                range.clone(),
                V::Val(
                    iv,
                    cat,
                    1,
                ),
            );
            for u in body {
                self.step(u);
            }
        }
        self.end_block(op::BRANCH, &[cont]);
        self.start_block(cont);
        self.close_loop_regs();
        let step = self.const_bits(cat, Self::UNROLL as u64);
        self.d_at(iv_next, op::I_ADD, vty, &[phi, step]);
        let bty = self.bool_ty();
        // The loop covers whole chunks only; a trailing partial chunk would read
        // past the extent, so it is emitted below instead. The extent is the
        // `RANGE`'s source (`end` here is the matching `END` uop).
        let bound = match self.ctx.ssimplify(self.info[range].srcs[0].clone()) {
            Ok(crate::uop::sint::SVal::I(ext)) if ext > 0 => {
                let rem = ext as i64 % Self::UNROLL as i64;
                if rem == 0 {
                    None
                } else {
                    Some((ext as i64 - rem, rem))
                }
            }
            _ => None,
        };
        let limit = match bound {
            Some((lim, _)) => self.const_bits(cat, lim as u64),
            None => end_v,
        };
        let cmp = if cat.is_signed() {
            self.d(op::S_LESS_THAN, bty, &[iv_next, limit])
        } else {
            self.d(op::U_LESS_THAN, bty, &[iv_next, limit])
        };
        self.end_block(op::BRANCH_CONDITIONAL, &[cmp, hdr, merge]);
        self.start_block(merge);
        // The leftover iterations, in order, as straight-line body copies. The
        // loop-carried registers already hold the value carried out of the
        // continue block, so these continue from it exactly like another chunk.
        if let Some((lim, rem)) = bound {
            let prev = self.vals.insert(range.clone(), V::Val(limit, cat, 1));
            for k in 0..rem {
                // Re-emit the body at iteration `lim + k`. The range's id is
                // re-mapped to a constant so nested uses/indexing see the value.
                let kc = self.const_bits(cat, (lim + k) as u64);
                self.vals.insert(range.clone(), V::Val(kc, cat, 1));
                for u in body {
                    self.step(u);
                }
            }
            // Restore what the loop head left behind, so anything outside the
            // loop sees the same value it would have without the peel.
            match prev {
                Some(v) => {
                    self.vals.insert(range.clone(), v);
                }
                None => {
                    self.vals.remove(range);
                }
            }
        }
        let _ = end;
    }

    /// Walk the pointer chain of `u` down to the `BUFFER` uop it started from.
    fn buffer_root(&self, u: &UopId) -> Option<UopId> {
        let mut cur = u.clone();
        for _ in 0..64 {
            let i = self.info.get(&cur)?;
            match i.op {
                Ops::Buffer => return Some(cur),
                Ops::Param => return None,
                Ops::Index | Ops::Shrink | Ops::Cast | Ops::Bitcast | Ops::After => {
                    cur = i.srcs.first()?.clone();
                }
                _ => return None,
            }
        }
        None
    }

    /// A constant-index pointer chain: `(array uop, element index)`.
    fn const_elem(&mut self, ptr: &UopId) -> Option<(UopId, i64)> {
        let mut cur = ptr.clone();
        let mut off = 0i64;
        for _ in 0..64 {
            let i = self.info.get(&cur)?;
            match i.op {
                Ops::Buffer => {
                    let size = dtype_size(i.dt) as i64;
                    return Some((cur.clone(), off / size));
                }
                Ops::Index | Ops::Shrink => {
                    let idx = i.srcs.get(1)?;
                    let Ok(crate::uop::sint::SVal::I(v)) = self.ctx.ssimplify(idx.clone()) else {
                        return None;
                    };
                    let base = self.info.get(&i.srcs[0])?;
                    off += v as i64 * dtype_size(base.dt) as i64;
                    cur = i.srcs[0].clone();
                }
                Ops::Cast | Ops::Bitcast | Ops::After => cur = i.srcs[0].clone(),
                _ => return None,
            }
        }
        None
    }

    /// Decide which register arrays can live in SSA (see `promotable`) and record
    /// each loop's body span.
    fn scan_buffers(&mut self, lst: &[UopId]) {
        let mut idx_const: HashMap<UopId, bool> = HashMap::new();
        let mut store_in_if: HashMap<UopId, bool> = HashMap::new();
        let mut if_depth = 0i64;
        let mut ranges: Vec<(UopId, usize)> = Vec::new();
        for (i, u) in lst.iter().enumerate() {
            let Some(info) = self.info.get(u).map(|i| (i.op, i.srcs.clone())) else { continue };
            let info = Info { op: info.0, dt: dtype::VOID, srcs: info.1, numel: 1, addrspace: None };
            match info.op {
                Ops::If => if_depth += 1,
                Ops::Endif => if_depth -= 1,
                Ops::Range => {
                    ranges.push((u.clone(), i));
                    self.loop_writes.entry(u.clone()).or_default();
                }
                Ops::End => {
                    ranges.pop();
                }
                Ops::Index | Ops::Shrink => {
                    if let Some(root) = self.buffer_root(&info.srcs[0]) {
                        let k = info
                            .srcs
                            .get(1)
                            .map(|s| {
                                matches!(
                                    self.ctx.ssimplify(s.clone()),
                                    Ok(crate::uop::sint::SVal::I(_))
                                )
                            })
                            .unwrap_or(true);
                        let e = idx_const.entry(root).or_insert(true);
                        *e = *e && k;
                    }
                }
                Ops::Store => {
                    if if_depth > 0
                        && let Some(root) = self.buffer_root(&info.srcs[0])
                    {
                        store_in_if.insert(root, true);
                    }
                    if let Some((buf, k)) = self.const_elem(&info.srcs[0])
                        && let Some((r, _)) = ranges.last()
                    {
                        self.loop_writes
                            .entry(r.clone())
                            .or_default()
                            .push((buf, k));
                    }
                }
                _ => {}
            }
        }
        // Buffers that are never indexed (or only with constants) and never
        // stored inside a conditional can be promoted.
        let bufs: Vec<UopId> = lst
            .iter()
            .filter(|u| self.info.get(u).map(|i| i.op) == Some(Ops::Buffer))
            .cloned()
            .collect();
        for b in bufs {
            let ok = *idx_const.get(&b).unwrap_or(&true) && !store_in_if.contains_key(&b);
            let local = self
                .info
                .get(&b)
                .map(|i| i.addrspace == Some(dtype::AddrSpace::Local))
                .unwrap_or(true);
            self.promotable.insert(b, ok && !local);
        }
    }

    /// The promoted element `(variable, element)` of a local pointer, if the
    /// pointer is a constant-index access into a promotable register array.
    fn promoted_elem(&self, p: &PtrVal) -> Option<(u32, i64)> {
        let Root::Local { var, stride, sc, .. } = p.root else {
            return None;
        };
        if sc != storage::FUNCTION {
            return None;
        }
        let Off::C(off) = p.off else { return None };
        Some((var, off / stride.max(1) as i64))
    }

    /// Emit the loop-carried `OpPhi`s for register elements written in `range`'s
    /// body. Must run right after the loop's other `OpPhi`s (SPIR-V requires all
    /// phis at the top of the header block).
    fn open_loop_regs(
        &mut self,
        range: &UopId,
        pred: u32,
        cont: u32,
    ) -> Vec<LoopReg> {
        let mut out = Vec::new();
        let writes = self.loop_writes.get(range).cloned().unwrap_or_default();
        let mut seen: Vec<(u32, i64)> = Vec::new();
        for (buf, k) in writes {
            if !self.promotable.get(&buf).copied().unwrap_or(false) {
                continue;
            }
            let Some((var, scalar)) = self.local_var.get(&buf.id()).copied() else {
                continue;
            };
            let cat = cat_of(self.info[&buf].dt, &self.caps);
            if is_narrow_int(cat) || seen.contains(&(var, k)) {
                continue;
            }
            seen.push((var, k));
            let ty = self.cat_ty(cat);
            let pre = match self.regs.get(&(var, k)).copied() {
                Some(v) => v,
                None if scalar => self.d(op::LOAD, ty, &[var]),
                None => {
                    let ptr_ty = self.ty(TyKey::Ptr(storage::FUNCTION, ty));
                    let kidx = self.const_bits(
                        Cat::Int {
                            bits: 32,
                            signed: false,
                        },
                        k as u64,
                    );
                    let ptr = self.access_chain(var, ptr_ty, kidx, false);
                    self.d(op::LOAD, ty, &[ptr])
                }
            };
            let next = self.id();
            let phi = self.d(op::PHI, ty, &[pre, pred, next, cont]);
            self.regs.insert((var, k), phi);
            out.push((var, k, phi, next, cat, ty));
        }
        out
    }

    /// Close a loop's register phis: the value at the end of the body becomes the
    /// back-edge operand, and after the loop the phi is the element's value.
    fn close_loop_regs(&mut self) {
        let Some(carried) = self.loop_regs.pop() else {
            return;
        };
        for (var, k, phi, next, _cat, ty) in carried {
            let cur = self.regs.get(&(var, k)).copied().unwrap_or(phi);
            self.d_at(next, op::COPY_OBJECT, ty, &[cur]);
            // After the loop the element's value is the one carried out of the
            // *continue* block, not the header phi: the exit branch is taken from
            // the continue block, so the header is not re-entered and its phi
            // still holds the previous iteration's value (off by one iteration).
            self.regs.insert((var, k), next);
        }
    }

    fn step(&mut self, u: &UopId) {
        let (op, dt, srcs, numel, addrspace) = {
            let i = &self.info[u];
            (i.op, i.dt, i.srcs.clone(), i.numel, i.addrspace)
        };
        match op {
            Ops::Noop | Ops::Group | Ops::Const | Ops::Sink => {}
            Ops::After => {
                if let Some(s) = srcs.first()
                    && let Some(v) = self.vals.get(s).copied()
                {
                    self.vals.insert(u.clone(), v);
                }
            }
            Ops::Param => {
                let Arg::Param(p) = self.ctx.arg_of(u) else {
                    panic!("TGERR: SPIR-V PARAM without a ParamArg");
                };
                if p.addrspace == Some(dtype::AddrSpace::Alu) {
                    panic!(
                        "TGERR: VK does not support ALU-addrspace scalar params (slot {})",
                        p.slot
                    );
                }
                self.vals.insert(
                    u.clone(),
                    V::Ptr(PtrVal {
                        root: Root::Buf(p.slot),
                        off: Off::C(0),
                    }),
                );
            }
            Ops::Buffer => {
                let (var, scalar) = *self
                    .local_var
                    .get(&u.id())
                    .unwrap_or_else(|| panic!("TGERR: SPIR-V BUFFER not declared"));
                let cat = cat_of(dt, &self.caps);
                let sc = if addrspace == Some(dtype::AddrSpace::Local) {
                    storage::WORKGROUP
                } else {
                    storage::FUNCTION
                };
                self.vals.insert(
                    u.clone(),
                    V::Ptr(PtrVal {
                        root: Root::Local {
                            var,
                            cat,
                            stride: cat.size(),
                            sc,
                            scalar,
                        },
                        off: Off::C(0),
                    }),
                );
            }
            Ops::Stack => {
                let lanes = numel.max(1);
                let cat = cat_of(dt, &self.caps);
                let ids: Vec<u32> = srcs
                    .iter()
                    .map(|s| {
                        let (id, _c, _n) = self.value(s);
                        id
                    })
                    .collect();
                let ty = self.val_ty(cat, lanes.max(ids.len() as u32));
                let id = self.d(op::COMPOSITE_CONSTRUCT, ty, &ids);
                self.vals
                    .insert(u.clone(), V::Val(id, reg_cat(cat), ids.len() as u32));
            }
            Ops::Index | Ops::Shrink => {
                match self.vals.get(&srcs[0]).copied() {
                    Some(V::Ptr(base)) => {
                        let base_cat = cat_of(self.info[&srcs[0]].dt, &self.caps);
                        let (idx, icat, _) = self.value(&srcs[1]);
                        let scaled = Off::Id(idx, icat.bits().max(32));
                        let scaled = self.off_scale(scaled, base_cat.size() as i64);
                        let off = self.off_add(base.off, scaled);
                        self.vals.insert(
                            u.clone(),
                            V::Ptr(PtrVal {
                                root: base.root,
                                off,
                            }),
                        );
                    }
                    Some(V::Val(vid, cat, lanes)) => {
                        let idx = self.val_id(&srcs[1]);
                        let lane = self
                            .known_of(idx)
                            .or_else(|| match self.ctx.ssimplify(srcs[1].clone()) {
                                Ok(crate::uop::sint::SVal::I(l)) => Some(l as i64),
                                _ => None,
                            });
                        let id = match lane {
                            Some(l) => {
                                let ty = self.cat_ty(cat);
                                self.d(op::COMPOSITE_EXTRACT, ty, &[vid, l as u32])
                            }
                            None => {
                                let ty = self.cat_ty(cat);
                                self.d(op::VECTOR_EXTRACT_DYNAMIC, ty, &[vid, idx])
                            }
                        };
                        let _ = lanes;
                        self.vals.insert(u.clone(), V::Val(id, cat, 1));
                    }
                    None => panic!(
                        "TGERR: SPIR-V renderer: INDEX base {} not rendered",
                        srcs[0].id()
                    ),
                }
            }
            Ops::Cast => {
                if let Some(V::Ptr(p)) = self.vals.get(&srcs[0]).copied() {
                    self.vals.insert(u.clone(), V::Ptr(p));
                } else {
                    let from_dt = self.info[&srcs[0]].dt;
                    let (id, fcat, flanes) = self.value(&srcs[0]);
                    let (tcat, tlanes) = self.cat_lanes(u);
                    let id = self.widen(id, fcat, flanes, tlanes);
                    // bf16 is a *bit pattern* in a 16-bit int, not a number:
                    // `cast(f32, bf16)` rounds the mantissa and `cast(bf16, f32)`
                    // shifts it back. The pipeline normally expands these
                    // (`extra::pm_manual_bf16_cast`), but a folded constant can
                    // reach the renderer directly.
                    let out = if dt == dtype::BFLOAT16 && fcat.is_float() {
                        let f32_id = self.cast_num(
                            id,
                            fcat,
                            Cat::Float { bits: 32 },
                            tlanes,
                        );
                        self.f32_to_bf16(f32_id, tlanes)
                    } else if from_dt == dtype::BFLOAT16 && tcat.is_float() {
                        let w = self.bf16_to_f32(id, tlanes);
                        self.cast_num(w, Cat::Float { bits: 32 }, tcat, tlanes)
                    } else {
                        self.cast_num(id, fcat, tcat, tlanes)
                    };
                    self.vals.insert(u.clone(), V::Val(out, tcat, tlanes));
                }
            }
            Ops::Bitcast => {
                if let Some(V::Ptr(p)) = self.vals.get(&srcs[0]).copied() {
                    self.vals.insert(u.clone(), V::Ptr(p));
                } else {
                    let (id, fcat, flanes) = self.value(&srcs[0]);
                    let (tcat, tlanes) = self.cat_lanes(u);
                    let id = self.widen(id, fcat, flanes, tlanes);
                    let out = self.bitcast_val(id, fcat, tcat, tlanes);
                    self.vals.insert(u.clone(), V::Val(out, reg_cat(tcat), tlanes));
                }
            }
            Ops::Load => {
                let p = self.ptr(&srcs[0]);
                let cat = cat_of(dt, &self.caps);
                let lanes = numel.max(1);
                let id = if srcs.len() >= 3 {
                    let (g, gcat, glanes) = self.value(&srcs[2]);
                    let g = self.as_bool(g, gcat, glanes);
                    let (dflt, dcat, dlanes) = self.value(&srcs[1]);
                    let dflt = self.widen(dflt, dcat, dlanes, lanes);
                    let lt = self.id();
                    let lf = self.id();
                    let m = self.id();
                    Self::push(&mut self.block, op::SELECTION_MERGE, &[m, 0]);
                    self.end_block(op::BRANCH_CONDITIONAL, &[g, lt, lf]);
                    self.start_block(lt);
                    let x = self.load_ptr(p, cat, lanes);
                    self.end_block(op::BRANCH, &[m]);
                    self.start_block(lf);
                    self.end_block(op::BRANCH, &[m]);
                    self.start_block(m);
                    let ty = self.val_ty(cat, lanes);
                    self.d(op::PHI, ty, &[x, lt, dflt, lf])
                } else {
                    self.load_ptr(p, cat, lanes)
                };
                self.vals.insert(u.clone(), V::Val(id, reg_cat(cat), lanes));
            }
            Ops::Store => {
                let p = self.ptr(&srcs[0]);
                let (cat, lanes) = self.cat_lanes(&srcs[0]);
                let (v, vcat, vlanes) = self.value(&srcs[1]);
                let v = self.widen(v, vcat, vlanes, lanes);
                let v = if vcat == cat {
                    v
                } else {
                    self.cast_num(v, vcat, cat, lanes)
                };
                self.store_ptr(p, cat, lanes, v);
            }
            Ops::Special => {
                let Arg::Str(s) = self.ctx.arg_of(u) else {
                    panic!("TGERR: SPIR-V SPECIAL without a name");
                };
                let which = s.as_bytes()[0];
                let dim: usize = s[s.len() - 1..].parse().unwrap_or(0);
                let b = match which {
                    b'g' => builtin::WORKGROUP_ID,
                    b'l' => builtin::LOCAL_INVOCATION_ID,
                    other => panic!(
                        "TGERR: SPIR-V renderer has no workitem for {:?}",
                        other as char
                    ),
                };
                let var = self.builtin_input(b);
                let uvec3 = self.ty(TyKey::Vec(
                    Cat::Int {
                        bits: 32,
                        signed: false,
                    },
                    3,
                ));
                let ptr = self.ty(TyKey::Ptr(storage::INPUT, uvec3));
                let v = self.d(op::LOAD, uvec3, &[var]);
                let _ = ptr;
                let comp_ty = self.cat_ty(Cat::Int {
                    bits: 32,
                    signed: false,
                });
                let comp = self.d(op::COMPOSITE_EXTRACT, comp_ty, &[v, dim as u32]);
                let ty = self.cat_ty(Cat::Int {
                    bits: 32,
                    signed: true,
                });
                let id = self.d(op::BITCAST, ty, &[comp]);
                self.vals.insert(
                    u.clone(),
                    V::Val(
                        id,
                        Cat::Int {
                            bits: 32,
                            signed: true,
                        },
                        1,
                    ),
                );
            }
            Ops::Range => {
                let cat = cat_of(dt, &self.caps);
                let end = self.val_id(&srcs[0]);
                let end = if cat_of(self.info[&srcs[0]].dt, &self.caps) == cat {
                    end
                } else {
                    let fcat = cat_of(self.info[&srcs[0]].dt, &self.caps);
                    self.cast_num(end, fcat, cat, 1)
                };
                let pred = self.label;
                let hdr = self.id();
                let body = self.id();
                let cont = self.id();
                let merge = self.id();
                self.end_block(op::BRANCH, &[hdr]);
                self.start_block(hdr);
                let iv = if dt == dtype::VOID {
                    None
                } else {
                    let vty = self.cat_ty(cat);
                    let zero = self.const_bits(cat, 0);
                    let iv_next = self.id();
                    let phi = self.d(op::PHI, vty, &[zero, pred, iv_next, cont]);
                    Some((phi, cat, iv_next))
                };
                let carried = self.open_loop_regs(u, pred, cont);
                self.loop_regs.push(carried);
                self.e(op::LOOP_MERGE, &[merge, cont, 0]);
                self.end_block(op::BRANCH, &[body]);
                self.start_block(body);
                self.loops.push(LoopFrame {
                    header: hdr,
                    cont,
                    merge,
                    iv: iv.map(|(p, c, _)| (p, c)),
                    iv_next: iv.map(|(_, _, n)| n).unwrap_or(0),
                    end,
                });
                if let Some((phi, cat, _)) = iv {
                    self.vals.insert(u.clone(), V::Val(phi, cat, 1));
                }
            }
            Ops::End => {
                let f = self
                    .loops
                    .pop()
                    .unwrap_or_else(|| panic!("TGERR: SPIR-V END without a RANGE"));
                // Close the body unconditionally and finish the loop in the
                // continue block: a phi's operand must be defined in the block it
                // is the parent of (or a block dominating it), and putting the
                // loop-carried copies there keeps that trivially true.
                self.end_block(op::BRANCH, &[f.cont]);
                self.start_block(f.cont);
                self.close_loop_regs();
                let cond = if srcs.len() >= 3 {
                    let (g, gcat, glanes) = self.value(&srcs[2]);
                    Some(self.as_bool(g, gcat, glanes))
                } else {
                    None
                };
                match f.iv {
                    Some((phi, cat)) => {
                        let one = self.const_bits(cat, 1);
                        let ty = self.cat_ty(cat);
                        self.d_at(f.iv_next, op::I_ADD, ty, &[phi, one]);
                        let bty = self.bool_ty();
                        let cmp = if cat.is_signed() {
                            self.d(op::S_LESS_THAN, bty, &[f.iv_next, f.end])
                        } else {
                            self.d(op::U_LESS_THAN, bty, &[f.iv_next, f.end])
                        };
                        let cont = match cond {
                            Some(c) => self.d(op::BITWISE_AND, bty, &[cmp, c]),
                            None => cmp,
                        };
                        self.end_block(op::BRANCH_CONDITIONAL, &[cont, f.header, f.merge]);
                    }
                    None => match cond {
                        Some(c) => {
                            self.end_block(op::BRANCH_CONDITIONAL, &[c, f.header, f.merge])
                        }
                        None => self.end_block(op::BRANCH, &[f.header]),
                    },
                }
                self.start_block(f.merge);
            }
            Ops::If => {
                let (c, ccat, clanes) = self.value(&srcs[0]);
                let c = self.as_bool(c, ccat, clanes);
                let m = self.id();
                let then = self.id();
                Self::push(&mut self.block, op::SELECTION_MERGE, &[m, 0]);
                self.end_block(op::BRANCH_CONDITIONAL, &[c, then, m]);
                self.start_block(then);
                self.ifs.push(m);
            }
            Ops::Endif => {
                let m = self
                    .ifs
                    .pop()
                    .unwrap_or_else(|| panic!("TGERR: SPIR-V ENDIF without an IF"));
                self.end_block(op::BRANCH, &[m]);
                self.start_block(m);
            }
            Ops::Barrier => {
                let scope = self.const_u32(2); // Workgroup
                let sem = self.const_u32(0x108); // AcquireRelease | WorkgroupMemory
                self.e(op::CONTROL_BARRIER, &[scope, scope, sem]);
            }
            Ops::Wmma => panic!("TGERR: SPIR-V renderer does not emit WMMA yet"),
            other if group::in_(group::ALU, other) => {
                self.alu(u, other, dt, &srcs, numel, addrspace);
            }
            other => panic!("TGERR: SPIR-V op {other:?} not implemented"),
        }
    }

    fn alu(
        &mut self,
        u: &UopId,
        op: Ops,
        dt: DTypeId,
        srcs: &[UopId],
        _numel: u32,
        _addrspace: Option<dtype::AddrSpace>,
    ) {
        let _ = dt;
        let (rcat, rlanes) = self.result_cat(op, srcs);
        let lanes = rlanes.max(1);
        // Operands converted to the result category and lane count.
        let mut ops: Vec<u32> = Vec::with_capacity(srcs.len());
        for s in srcs {
            let (id, c, n) = self.value(s);
            let id = self.widen(id, c, n, lanes);
            let id = if c == rcat || op == Ops::Where || is_cmp(op) {
                id
            } else {
                self.cast_num(id, c, rcat, lanes)
            };
            ops.push(id);
        }
        let ty = self.val_ty(rcat, lanes);
        let out = match op {
            Ops::Neg => {
                let (id, cat, _) = self.value(&srcs[0]);
                if cat.is_float() {
                    self.d(op::F_NEGATE, ty, &[id])
                } else if cat.is_signed() {
                    self.d(op::S_NEGATE, ty, &[id])
                } else {
                    let z = self.const_splat(cat, lanes, 0);
                    self.d(op::I_SUB, ty, &[z, id])
                }
            }
            Ops::Sqrt | Ops::Exp2 | Ops::Log2 | Ops::Sin | Ops::Trunc => {
                let g = match op {
                    Ops::Sqrt => glsl::SQRT,
                    Ops::Exp2 => glsl::EXP2,
                    Ops::Log2 => glsl::LOG2,
                    Ops::Sin => glsl::SIN,
                    _ => glsl::TRUNC,
                };
                self.d(op::EXT_INST, ty, &[self.glsl, g, ops[0]])
            }
            Ops::Reciprocal => {
                let one = self.const_splat(rcat, lanes, Self::one_bits(rcat));
                self.d(op::F_DIV, ty, &[one, ops[0]])
            }
            Ops::Pow => {
                let g = self.glsl;
                self.d(op::EXT_INST, ty, &[g, glsl::POW, ops[0], ops[1]])
            }
            Ops::Add => {
                // Contract a single-use `Mul` into a fused multiply-add, which
                // C compilers do implicitly for `acc + a*b` (and is what the
                // OpenCL/CUDA backends get). Without it the GEMM spends two
                // instructions per MAC.
                if rcat.is_float() && crate::helpers::env::get("VK_NO_FMA").is_none() {
                    if let Some(id) = self.try_fma(&srcs[0], &srcs[1], rcat, lanes) {
                        id
                    } else if let Some(id) = self.try_fma(&srcs[1], &srcs[0], rcat, lanes) {
                        id
                    } else {
                        self.d(op::F_ADD, ty, &[ops[0], ops[1]])
                    }
                } else {
                    self.d(op::I_ADD, ty, &[ops[0], ops[1]])
                }
            }
            Ops::Sub => self.f_bin(op, rcat, ty, &ops, op::F_SUB, op::I_SUB),
            Ops::Mul => self.f_bin(op, rcat, ty, &ops, op::F_MUL, op::I_MUL),
            Ops::Fdiv => self.f_bin(op, rcat, ty, &ops, op::F_DIV, op::S_DIV),
            Ops::Floordiv => self.f_bin(op, rcat, ty, &ops, op::F_DIV, op::S_DIV),
            Ops::Mulacc => {
                if rcat.is_float() {
                    self.d(op::EXT_INST, ty, &[self.glsl, glsl::FMA, ops[0], ops[1], ops[2]])
                } else {
                    let ty_i = ty;
                    let m = self.d(op::I_MUL, ty_i, &[ops[0], ops[1]]);
                    self.d(op::I_ADD, ty_i, &[m, ops[2]])
                }
            }
            Ops::Cdiv => {
                if rcat.is_float() {
                    self.d(op::F_DIV, ty, &[ops[0], ops[1]])
                } else if rcat.is_signed() {
                    self.d(op::S_DIV, ty, &[ops[0], ops[1]])
                } else {
                    self.d(op::U_DIV, ty, &[ops[0], ops[1]])
                }
            }
            Ops::Cmod => {
                if rcat.is_float() {
                    self.d(op::F_REM, ty, &[ops[0], ops[1]])
                } else if rcat.is_signed() {
                    self.d(op::S_REM, ty, &[ops[0], ops[1]])
                } else {
                    self.d(op::U_MOD, ty, &[ops[0], ops[1]])
                }
            }
            Ops::Floormod => {
                if rcat.is_float() {
                    self.d(op::F_REM, ty, &[ops[0], ops[1]])
                } else if rcat.is_signed() {
                    self.d(op::S_REM, ty, &[ops[0], ops[1]])
                } else {
                    self.d(op::U_MOD, ty, &[ops[0], ops[1]])
                }
            }
            Ops::Xor => self.d(op::BITWISE_XOR, ty, &[ops[0], ops[1]]),
            Ops::Or => self.d(op::BITWISE_OR, ty, &[ops[0], ops[1]]),
            Ops::And => self.d(op::BITWISE_AND, ty, &[ops[0], ops[1]]),
            Ops::Shl => {
                let sh = self.shift_amount(&srcs[1], rcat, lanes);
                self.d(op::SHIFT_LEFT_LOGICAL, ty, &[ops[0], sh])
            }
            Ops::Shr => {
                let sh = self.shift_amount(&srcs[1], rcat, lanes);
                let o = if rcat.is_signed() {
                    op::SHIFT_RIGHT_ARITHMETIC
                } else {
                    op::SHIFT_RIGHT_LOGICAL
                };
                self.d(o, ty, &[ops[0], sh])
            }
            Ops::Cmplt => {
                let (id, cat, _) = self.value(&srcs[0]);
                let (b0, b1, bty) = (ops[0], ops[1], ty);
                let _ = (id, cat);
                if cat.is_float() {
                    self.d(op::F_ORD_LESS_THAN, bty, &[b0, b1])
                } else if cat.is_signed() {
                    self.d(op::S_LESS_THAN, bty, &[b0, b1])
                } else {
                    self.d(op::U_LESS_THAN, bty, &[b0, b1])
                }
            }
            Ops::Cmpeq => {
                let (_, cat, _) = self.value(&srcs[0]);
                if cat.is_float() {
                    self.d(op::F_ORD_EQUAL, ty, &[ops[0], ops[1]])
                } else {
                    self.d(op::I_EQUAL, ty, &[ops[0], ops[1]])
                }
            }
            Ops::Cmpne => {
                let (_, cat, _) = self.value(&srcs[0]);
                if cat.is_float() {
                    self.d(op::F_UNORD_NOT_EQUAL, ty, &[ops[0], ops[1]])
                } else {
                    self.d(op::I_NOT_EQUAL, ty, &[ops[0], ops[1]])
                }
            }
            Ops::Where => {
                let (c, ccat, clanes) = self.value(&srcs[0]);
                let c = self.as_bool(c, ccat, clanes.max(1));
                let c = self.widen(c, Cat::Bool, clanes.max(1), lanes);
                let (x, xcat, xl) = self.value(&srcs[1]);
                let x = self.widen(x, xcat, xl, lanes);
                let (y, ycat, yl) = self.value(&srcs[2]);
                let y = self.widen(y, ycat, yl, lanes);
                let (x, y) = if xcat == ycat {
                    (x, y)
                } else {
                    (
                        self.cast_num(x, xcat, ycat, lanes),
                        y,
                    )
                };
                let ty = self.val_ty(ycat, lanes);
                self.d(op::SELECT, ty, &[c, x, y])
            }
            Ops::Max => {
                // `Ops::MAX` is decomposed by the pipeline; keep a native form.
                let (_, cat, _) = self.value(&srcs[0]);
                let _ = cat;
                if rcat.is_float() {
                    self.d(op::EXT_INST, ty, &[self.glsl, 40, ops[0], ops[1]])
                } else if rcat.is_signed() {
                    self.d(op::EXT_INST, ty, &[self.glsl, 42, ops[0], ops[1]])
                } else {
                    self.d(op::EXT_INST, ty, &[self.glsl, 41, ops[0], ops[1]])
                }
            }
            other => panic!(
                "TGERR: SPIR-V ALU op {other:?} on {:?} not implemented (uop {})",
                self.info[u].dt,
                u.id()
            ),
        };
        self.vals.insert(u.clone(), V::Val(out, rcat, lanes));
    }

    /// `mul + x` as a single `Fma` when `mul` is a single-use float `MUL`.
    fn try_fma(&mut self, mul: &UopId, add: &UopId, cat: Cat, lanes: u32) -> Option<u32> {
        let (mop, mnumel, msrcs) = {
            let mi = self.info.get(mul)?;
            (mi.op, mi.numel, mi.srcs.clone())
        };
        if mop != Ops::Mul || mnumel != lanes || msrcs.len() != 2 {
            return None;
        }
        if self.uses.get(mul).copied().unwrap_or(0) != 1 {
            return None;
        }
        let (a, acat, al) = self.value(&msrcs[0]);
        let (b, bcat, bl) = self.value(&msrcs[1]);
        let (c, ccat, cl) = self.value(add);
        if !acat.is_float() || !bcat.is_float() || !ccat.is_float() {
            return None;
        }
        let a = self.widen(a, acat, al, lanes);
        let b = self.widen(b, bcat, bl, lanes);
        let c = self.widen(c, ccat, cl, lanes);
        let ty = self.val_ty(cat, lanes);
        let g = self.glsl;
        Some(self.d(op::EXT_INST, ty, &[g, glsl::FMA, a, b, c]))
    }

    fn f_bin(&mut self, _op: Ops, cat: Cat, ty: u32, ops: &[u32], fop: u16, iop: u16) -> u32 {
        if cat.is_float() {
            self.d(fop, ty, &[ops[0], ops[1]])
        } else if cat == (Cat::Int { bits: 32, signed: false }) && iop == op::S_DIV {
            self.d(op::U_DIV, ty, &[ops[0], ops[1]])
        } else {
            self.d(iop, ty, &[ops[0], ops[1]])
        }
    }

    fn shift_amount(&mut self, s: &UopId, cat: Cat, lanes: u32) -> u32 {
        let (id, scat, slanes) = self.value(s);
        let id = self.widen(id, scat, slanes, lanes);
        if scat == cat {
            id
        } else {
            let ty = self.val_ty(cat, lanes);
            self.d(op::BITCAST, ty, &[id])
        }
    }

    fn result_cat(&mut self, op: Ops, srcs: &[UopId]) -> (Cat, u32) {
        // `WHERE` is `cond ? x : y`: the result is the arm's dtype, not the
        // condition's.
        let pick = if op == Ops::Where {
            srcs.get(1).or(srcs.first())
        } else {
            srcs.first()
        };
        let (cat, lanes) = match pick {
            Some(s) => self.cat_lanes(s),
            None => (
                Cat::Int {
                    bits: 32,
                    signed: true,
                },
                1,
            ),
        };
        if is_cmp(op) {
            (Cat::Bool, lanes)
        } else {
            (cat, lanes)
        }
    }
}

fn is_cmp(op: Ops) -> bool {
    matches!(op, Ops::Cmplt | Ops::Cmpne | Ops::Cmpeq)
}

impl<'a> Emitter<'a> {
    /// Render the whole kernel.
    fn run(&mut self, lst: &[UopId]) -> VkModule {
        // The newer SSBO form needs SPIR-V 1.3 (`StorageBuffer` + `Block`).
        if self.caps.spirv13 {
            self.ssbo_sc = storage::STORAGE_BUFFER;
            self.ssbo_block = deco::BLOCK;
        }
        // ---- snapshot the linear list -------------------------------------
        for u in lst {
            let numel = match self.ctx.shape(u) {
                Ok(Some(s)) if !s.is_empty() => {
                    s.iter().map(|x| x.clone().as_i64_or_panic()).product::<i64>() as u32
                }
                _ => 1,
            };
            let info = Info {
                op: self.ctx.op_of(u),
                dt: self.ctx.dt_of(u),
                srcs: self.ctx.srcs(u).to_vec(),
                numel,
                addrspace: self.ctx.addrspace_of(u.clone()),
            };
            let srcs = info.srcs.clone();
            self.info.insert(u.clone(), info);
            for s in srcs {
                *self.uses.entry(s).or_insert(0) += 1;
            }
        }
        self.scan_buffers(lst);
        // The kernel name (and the workgroup size) come from the SINK.
        let mut name = "kernel".to_string();
        for u in lst {
            let i = &self.info[u];
            if i.op == Ops::Sink
                && let Arg::Data(d) = self.ctx.arg_of(u)
                && d.tag == "KernelInfo"
                && let Some(Arg::Str(n)) = d
                    .fields
                    .iter()
                    .find_map(|(k, v)| if k == "name" { Some(v) } else { None })
            {
                name = crate::renderer::cstyle::to_function_name_pub(n);
            }
        }
        self.name = format!("{}_{}", name, self.caps.tag());
        for u in lst {
            let i = &self.info[u];
            if i.op != Ops::Special {
                continue;
            }
            let Arg::Str(s) = self.ctx.arg_of(u) else {
                continue;
            };
            let dim: usize = s[s.len() - 1..].parse().unwrap_or(0);
            match s.as_bytes()[0] {
                b'l' => {
                    if dim < 3
                        && let Ok(crate::uop::sint::SVal::I(v)) =
                            self.ctx.ssimplify(self.info[u].srcs[0].clone())
                    {
                        self.local_size[dim] = (v as u32).max(1);
                    }
                }
                b'g' => {}
                other => panic!(
                    "TGERR: SPIR-V renderer only supports gidx/lidx specials, got {:?}",
                    other as char
                ),
            }
        }
        for d in &mut self.local_size {
            *d = (*d).max(1);
        }

        // ---- module preamble ----------------------------------------------
        let void = self.ty(TyKey::Void);
        let _ = void;
        self.void = void;
        let glsl = self.id();
        self.glsl = glsl;
        let main = self.id();
        self.main = main;
        let fnty = self.ty(TyKey::Function);

        // ---- local arrays must be declared in the entry block ------------
        for u in lst {
            let (op, dt, numel, addrspace) = {
                let i = &self.info[u];
                (i.op, i.dt, i.numel, i.addrspace)
            };
            if op != Ops::Buffer {
                continue;
            }
            let cat = cat_of(dt, &self.caps);
            let elem = self.cat_ty(cat);
            let arr = self.ty(TyKey::Array(elem, numel.max(1)));
            let sc = if addrspace == Some(dtype::AddrSpace::Local) {
                storage::WORKGROUP
            } else {
                storage::FUNCTION
            };
            // A one-element buffer is just a scalar variable: declaring
            // `float[1]` and indexing it with a constant access chain is the
            // shape drivers are least likely to promote out of Function storage,
            // and the reduce accumulators in softmax/layernorm are exactly that.
            let scalar = numel <= 1;
            let ptr_ty = if scalar {
                self.ty(TyKey::Ptr(sc, elem))
            } else {
                self.ty(TyKey::Ptr(sc, arr))
            };
            let var = self.id();
            let words = vec![ptr_ty, var, sc];
            if sc == storage::FUNCTION {
                // `Function` storage is only legal inside a function, and all
                // such variables must be declared in the entry block.
                Self::push(&mut self.locals, op::VARIABLE, &words);
            } else {
                Self::push(&mut self.types, op::VARIABLE, &words);
            }
            self.local_var.insert(u.id(), (var, scalar));
        }

        // ---- function -----------------------------------------------------
        Self::push(&mut self.func, op::FUNCTION, &[void, main, 0, fnty]);
        let entry = self.id();
        Self::push(&mut self.func, op::LABEL, &[entry]);
        self.func.append(&mut self.locals);
        self.label = entry;
        self.terminated = false;

        // Walk by index so a qualifying `RANGE` can consume its whole body at
        // once (see `unroll_loop`).
        let mut i = 0usize;
        while i < lst.len() {
            let u = &lst[i];
            if let Some(end) = self.unroll_end(u, lst) {
                let (body, end_uop) = (lst[i + 1..end].to_vec(), lst[end].clone());
                self.unroll_loop(u, &body, &end_uop);
                i = end + 1;
                continue;
            }
            self.step(u);
            i += 1;
        }
        self.end_block(op::RETURN, &[]);
        self.func.append(&mut self.block);
        Self::push(&mut self.func, op::FUNCTION_END, &[]);

        // ---- capability/extension section (needs the walk's bindings) -----
        let mut caps: Vec<u32> = vec![cap::SHADER];
        let mut exts: Vec<&str> = Vec::new();
        if self.caps.storage16 || self.uses_16 {
            caps.push(cap::STORAGE_BUFFER_16);
            exts.push("SPV_KHR_16bit_storage");
        }
        if self.caps.f16 {
            caps.push(cap::FLOAT16);
        }
        if self.caps.storage8 {
            caps.push(cap::STORAGE_BUFFER_8);
            exts.push("SPV_KHR_8bit_storage");
        }
        if self.caps.int64 {
            caps.push(cap::INT64);
        }
        if self.caps.f64 {
            caps.push(cap::FLOAT64);
        }
        for c in caps {
            Self::push(&mut self.caps_sec, op::CAPABILITY, &[c]);
        }
        for e in exts {
            let words = lit(e);
            Self::push(&mut self.ext_sec, op::EXTENSION, &words);
        }
        let mut imp = vec![glsl];
        imp.extend(lit("GLSL.std.450"));
        Self::push(&mut self.ext_sec, op::EXT_INST_IMPORT, &imp);
        Self::push(&mut self.header, op::MEMORY_MODEL, &[0, 1]);
        // The entry point is always `main` (the Vulkan convention); the kernel
        // name is preserved as a debug name so `spirv-dis` stays readable.
        let mut ep = vec![5u32, main]; // ExecutionModel GLCompute
        ep.extend(lit("main"));
        // SPIR-V 1.0 requires every Input/Output variable in the interface.
        let mut ifaces: Vec<u32> = self.builtin_var.values().copied().collect();
        ifaces.sort_unstable();
        ep.extend(ifaces);
        Self::push(&mut self.header, op::ENTRY_POINT, &ep);
        let mut dbg = vec![main];
        dbg.extend(lit(&self.name));
        Self::push(&mut self.debug, op::NAME, &dbg);
        Self::push(
            &mut self.header,
            op::EXECUTION_MODE,
            &[
                main,
                17, // LocalSize
                self.local_size[0],
                self.local_size[1],
                self.local_size[2],
            ],
        );

        // ---- assemble ------------------------------------------------------
        let mut out = Vec::with_capacity(
            self.caps_sec.len()
                + self.ext_sec.len()
                + self.header.len()
                + self.types.len()
                + self.func.len(),
        );
        out.push(0x0723_0203); // magic
        out.push(if self.caps.spirv13 { 0x0001_0300 } else { 0x0001_0000 });
        out.push(0);
        out.push(0); // bound, patched below
        out.push(0); // schema
        out.extend_from_slice(&self.caps_sec);
        out.extend_from_slice(&self.ext_sec);
        out.extend_from_slice(&self.header);
        out.extend_from_slice(&self.debug);
        out.extend_from_slice(&self.annotations);
        out.extend_from_slice(&self.types);
        out.extend_from_slice(&self.func);
        out[3] = self.next_id;
        VkModule {
            slots: self.bindings.clone(),
            local: self.local_size,
            words: out,
        }
    }
}

// ---------------- helpers ----------------

/// Truncate a constant to the bit width of `cat` (SPIR-V requires the literal
/// to fit its type).
fn mask_bits(bits: u64, cat: Cat) -> u64 {
    if cat == Cat::Bool {
        return bits & 1;
    }
    let n = cat.bits();
    if n >= 64 {
        bits
    } else {
        bits & ((1u64 << n) - 1)
    }
}

/// IEEE binary16 bit pattern (round to nearest even).
fn f16_bits(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let man = x & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = (man | 0x80_0000) >> (1 - e);
        let round = (m >> 13) & 1;
        let half = 0x1000u32;
        let m = (m + half + round.wrapping_sub(1)) >> 13;
        return sign | (m as u16 & 0x3ff);
    }
    let m = man >> 13;
    let round = (man >> 12) & 1;
    let mut out = (e as u16) << 10 | m as u16;
    if round == 1 && (man & 0xfff) != 0 || (round == 1 && (m & 1) == 1) {
        out += 1;
    }
    sign | out
}

/// bf16 bit pattern (the top 16 bits of f32, round to nearest even).
fn bf16_bits(v: f32) -> u16 {
    let x = v.to_bits();
    let rounded = x.wrapping_add(0x7fff + ((x >> 16) & 1));
    (rounded >> 16) as u16
}

/// A SPIR-V literal string: `\0`-terminated and word-padded.
fn lit(s: &str) -> Vec<u32> {
    let mut bytes = s.as_bytes().to_vec();
    bytes.push(0);
    while !bytes.len().is_multiple_of(4) {
        bytes.push(0);
    }
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ---------------- renderer seam ----------------

/// The `VK` renderer facts.
///
/// The scheduler's tiling heuristics depend on these bounds, so the device's
/// real limits are used (the OpenCL defaults are unbounded, which would make the
/// scheduler emit grids Vulkan rejects).
pub fn facts_with_limits(caps: &Caps, global: [i64; 3], local: [i64; 3], shared: i64) -> Renderer {
    let mut r = facts(caps);
    r.global_max = Some(global);
    r.local_max = Some(local);
    r.shared_max = shared;
    r
}

/// The `VK` renderer facts with the conservative Vulkan 1.0 minimum limits.
pub fn facts(caps: &Caps) -> Renderer {
    let mut r = Renderer::opencl();
    r.device = "VK".to_string();
    r.global_max = Some([65535, 65535, 65535]);
    r.local_max = Some([1024, 1024, 64]);
    r.supported_dtypes = dtype::ALL
        .iter()
        .cloned()
        .filter(|&d| supported_dtype(d, caps))
        .collect();
    r
}

/// Render a linear list to a SPIR-V module.
pub fn render(ctx: &mut UOpCtx, lst: &[UopId], caps: Caps) -> VkModule {
    let mut em = Emitter {
        ctx,
        caps,
        name: String::new(),
        info: HashMap::new(),
        vals: HashMap::new(),
        next_id: 1,
        caps_sec: Vec::new(),
        ext_sec: Vec::new(),
        header: Vec::new(),
        debug: Vec::new(),
        annotations: Vec::new(),
        types: Vec::new(),
        func: Vec::new(),
        type_cache: HashMap::new(),
        const_cache: HashMap::new(),
        known: HashMap::new(),
        uses: HashMap::new(),
        promotable: HashMap::new(),
        regs: HashMap::new(),
        loop_writes: HashMap::new(),
        loop_regs: Vec::new(),
        binding_map: HashMap::new(),
        bindings: Vec::new(),
        uses_16: false,
        builtin_var: HashMap::new(),
        local_var: HashMap::new(),
        locals: Vec::new(),
        block: Vec::new(),
        label: 0,
        terminated: false,
        loops: Vec::new(),
        ifs: Vec::new(),
        glsl: 0,
        main: 0,
        void: 0,
        ssbo_sc: storage::UNIFORM,
        ssbo_block: deco::BUFFER_BLOCK,
        local_size: [1, 1, 1],
    };
    em.run(lst)
}

/// `VulkanRenderer` — the `UopRenderer` seam over [`render`].
pub struct VkUopRenderer {
    facts: Renderer,
    caps: Caps,
}

impl VkUopRenderer {
    /// Build the renderer for a device with `caps`.
    pub fn new(caps: Caps) -> VkUopRenderer {
        VkUopRenderer {
            facts: facts(&caps),
            caps,
        }
    }

    /// Build the renderer with device-derived facts (`max_buffers` from the
    /// driver's descriptor limit, limits from `VkPhysicalDeviceLimits`).
    pub fn with_facts(caps: Caps, facts: Renderer) -> VkUopRenderer {
        VkUopRenderer { facts, caps }
    }
}

impl UopRenderer for VkUopRenderer {
    fn facts(&self) -> &Renderer {
        &self.facts
    }
    fn render(&self, ctx: &mut UOpCtx, lst: &[UopId]) -> String {
        render(ctx, lst, self.caps).encode()
    }
}
