//! `runtime/ops_vk.py`-style backend for Vulkan (device `VK`).
//!
//! There is no tinygrad Vulkan target, so this backend composes the same public
//! seams as the other GPU backends: `OpaqueT` (a `VkBuffer` allocation),
//! `BackendAllocator` (host-visible, persistently mapped memory under the LRU),
//! `Comp` (identity: the rendered payload *is* the SPIR-V module), `Prog`
//! (descriptor bind + dispatch) and `UopRenderer` ([`crate::renderer::spirv`]).
//!
//! Graceful absence: registration only happens when the loader, a compute
//! device, the features the renderer advertises (8/16-bit storage, int64) and a
//! working pipeline probe are all present. Without them `VK` is simply not in
//! the device registry, exactly like a machine without CUDA.
//!
//! Every allocation is `HOST_VISIBLE | HOST_COHERENT` and permanently mapped,
//! so upload/download are `memcpy`; kernels read the same memory as storage
//! buffers. `VK:N` selects the N-th enumerated device.

use std::cell::RefCell;
use std::rc::Rc;

use crate::dtype::DTypeId;
use crate::renderer::spirv::{self, Caps, VkModule, VkUopRenderer};
use crate::runtime::buffer::OpaqueT;
use crate::runtime::device::{
    BackendAllocator, Comp, Compiled, DeviceFactory, LruAllocator, MemoryInfo, Prog, TinyElf,
    register_device_family, register_device_with_limit,
};
use crate::runtime::coopmat;
use crate::runtime::vk_sys::{self, DeviceInfo, GpuBuffer};
use crate::schedule::host_gemm::HostGemm;

// ---------------- device inner ----------------

/// The opened device shared by the allocator and every program.
/// Cache key of a cooperative-matrix pipeline: shape plus the operand dtypes.
type CoopKey = (usize, usize, usize, DTypeId, DTypeId);
/// Cache key of an element-wise conversion pipeline: module id plus every value
/// the module is patched with (shape *and* strides — keying on the element count
/// alone made a transposed staging pass reuse the row-major kernel).
type ConvKey = (u8, usize, usize, usize, usize);
/// A compiled module plus the descriptor sets it dispatches with (the sets are
/// referenced by pending submissions, so they live as long as the pipeline).
type CachedPipe = (Rc<vk_sys::GpuPipeline>, Vec<ash::vk::DescriptorSet>);

struct VkInner {
    dev: Rc<vk_sys::VkDevice>,
    /// Pipelines for the cooperative-matrix GEMM, with their descriptor sets.
    coop: RefCell<std::collections::HashMap<CoopKey, CachedPipe>>,
    /// Pipelines for the conversion/staging/epilogue modules, with their sets.
    conv: RefCell<std::collections::HashMap<ConvKey, CachedPipe>>,
    /// Reused scratch for the cooperative-matrix staging (A, B, C). Allocating
    /// tens of megabytes through `vkAllocateMemory` on every GEMM dominated the
    /// call, so the buffers grow in place instead.
    scratch: RefCell<std::collections::HashMap<u8, vk_sys::GpuBuffer>>,
}

impl VkInner {
    fn open(device: &str) -> VkInner {
        let index = device
            .split_once(':')
            .and_then(|(_, i)| i.parse::<usize>().ok())
            .unwrap_or(0);
        let devs = vk_devices();
        let info = devs.get(index).unwrap_or_else(|| {
            panic!(
                "TGERR: {device}: only {} Vulkan device(s) available",
                devs.len()
            )
        });
        let lib = vk_sys::VkLib::get().expect("TGERR: no Vulkan loader");
        let dev = vk_sys::VkDevice::open(&lib.instance, lib.api, info).unwrap_or_else(|e| {
            panic!(
                "TGERR: {device}: could not open Vulkan device {}: {e}",
                info.name
            )
        });
        eprintln!("VKDevice: opening {}", dev.describe());
        VkInner {
            dev: Rc::new(dev),
            coop: RefCell::new(std::collections::HashMap::new()),
            conv: RefCell::new(std::collections::HashMap::new()),
            scratch: RefCell::new(std::collections::HashMap::new()),
        }
    }
}

/// Whether `VK_TRACE` is on (cached: the hot path must not read the
/// environment per dispatch).
pub(crate) fn trace_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::helpers::env::get("VK_TRACE").is_some())
}

/// [`trace`], for `vk_sys` (which cannot reach the private helper).
pub(crate) fn trace_public(msg: &str) {
    trace(msg);
}

/// Print the backend probe (set `VK_TRACE=1`).
fn trace(msg: &str) {
    if trace_on() {
        eprintln!("[vk] {msg}");
    }
}

/// `VK_TIME` (cached).
pub(crate) fn time_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::helpers::env::get("VK_TIME").is_some())
}

/// `VK_SYNC` (cached).
fn sync_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::helpers::env::get("VK_SYNC").is_some())
}

/// `VK_DUMP_SPV` (cached).
fn dump_dir() -> Option<String> {
    static ON: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    ON.get_or_init(|| crate::helpers::env::get("VK_DUMP_SPV")).clone()
}

/// The usable physical devices (cached per thread: enumeration is not free).
fn vk_devices() -> Vec<DeviceInfo> {
    thread_local! {
        static DEVS: RefCell<Option<Rc<Vec<DeviceInfo>>>> = const { RefCell::new(None) };
    }
    DEVS.with(|d| {
        if let Some(v) = d.borrow().as_ref() {
            return (**v).clone();
        }
        let Some(lib) = vk_sys::VkLib::get() else {
            trace("no libvulkan loader");
            return Vec::new();
        };
        let mut out = Vec::new();
        for info in lib.devices() {
            // 1-byte storage and 8-bit arithmetic are required: `bool` and `u8`
            // tensors are everywhere, and a bool buffer is a byte array (SPIR-V
            // has no sized bool), compared/stored with 8-bit ops.
            if !info.caps.storage8 || !info.caps.int8 {
                trace(&format!(
                    "{}: no storageBuffer8BitAccess/shaderInt8",
                    info.name
                ));
                continue;
            }
            if let Err(e) = vk_sys::device_probe(&lib.instance, lib.api, &info) {
                trace(&format!("{}: probe failed: {e}", info.name));
                continue;
            }
            trace(&format!(
                "{}: usable, caps {:?}, coopmat {:?}",
                info.name, info.caps, info.coop
            ));
            out.push(info);
        }
        *d.borrow_mut() = Some(Rc::new(out.clone()));
        out
    })
}

// ---------------- opaque ----------------

/// One live `VkBuffer` plus the memory it owns.
struct Alloc {
    dev: Rc<vk_sys::VkDevice>,
    buf: GpuBuffer,
    id: usize,
}

impl Drop for Alloc {
    fn drop(&mut self) {
        self.dev.free(&self.buf);
    }
}

/// A window into an [`Alloc`] (a whole buffer or a `_offset` view of it).
struct VkOpaque {
    alloc: Rc<Alloc>,
    offset: usize,
    size: usize,
}

impl OpaqueT for VkOpaque {
    fn size(&self) -> usize {
        self.size
    }
    fn storage_id(&self) -> usize {
        self.alloc.id
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn vk_of(o: &dyn OpaqueT) -> &VkOpaque {
    o.as_any()
        .downcast_ref::<VkOpaque>()
        .expect("TGERR: expected a VK opaque")
}

// ---------------- allocator ----------------

struct VkAllocator {
    inner: Rc<VkInner>,
}

impl BackendAllocator for VkAllocator {
    fn _alloc(&self, size: usize) -> Rc<dyn OpaqueT> {
        let dev = self.inner.dev.clone();
        let buf = dev
            .alloc(size)
            .unwrap_or_else(|| panic!("TGERR: VK: could not allocate {size} bytes"));
        if trace_on() {
            trace(&format!("alloc {size} bytes"));
        }
        let id = dev.next_id.get();
        dev.next_id.set(id + 1);
        Rc::new(VkOpaque {
            alloc: Rc::new(Alloc { dev, buf, id }),
            offset: 0,
            size,
        })
    }

    fn _free(&self, o: Rc<dyn OpaqueT>) {
        drop(o); // last reference frees the VkBuffer
    }

    fn _copyin(&self, dest: &dyn OpaqueT, src: &[u8]) {
        if trace_on() {
            trace(&format!("copyin {}", src.len()));
        }
        // A queued dispatch may still be reading this buffer.
        self.inner.dev.wait_idle();
        let dev = &self.inner.dev;
        let d = vk_of(dest);
        assert_eq!(src.len(), d.size, "TGERR: VK copyin size mismatch");
        match d.alloc.buf.ptr {
            // Mapped (small/BAR) buffer: a plain host copy.
            Some(p) => unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), p.add(d.offset), src.len())
            },
            // Device-local: stage through host-visible memory.
            None => {
                let mut st = dev.staging(src.len());
                let stage = st.as_mut().expect("staging");
                unsafe {
                    std::ptr::copy_nonoverlapping(src.as_ptr(), stage.ptr.unwrap(), src.len())
                };
                dev.copy_buffer(stage, 0, &d.alloc.buf, d.offset as u64, src.len() as u64, true);
            }
        }
    }

    fn _copyout(&self, dest: &mut [u8], src: &dyn OpaqueT) {
        if trace_on() {
            trace(&format!("copyout {}", dest.len()));
        }
        let dev = &self.inner.dev;
        let s = vk_of(src);
        assert_eq!(dest.len(), s.size, "TGERR: VK copyout size mismatch");
        match s.alloc.buf.ptr {
            Some(p) => {
                dev.wait_idle();
                unsafe {
                    std::ptr::copy_nonoverlapping(p.add(s.offset), dest.as_mut_ptr(), s.size)
                };
            }
            None => {
                let t0 = std::time::Instant::now();
                let mut st = dev.staging(dest.len());
                {
                    let stage = st.as_mut().expect("staging");
                    dev.copy_buffer(
                        &s.alloc.buf,
                        s.offset as u64,
                        stage,
                        0,
                        s.size as u64,
                        true,
                    );
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            stage.ptr.unwrap(),
                            dest.as_mut_ptr(),
                            s.size,
                        )
                    };
                }
                if time_on() {
                    eprintln!(
                        "[vk] copyout {} bytes {:.3} ms",
                        dest.len(),
                        t0.elapsed().as_secs_f64() * 1e3
                    );
                }
            }
        }
    }

    fn _copy_into(&self, dest: &dyn OpaqueT, src: &dyn OpaqueT) {
        let dev = &self.inner.dev;
        let (d, s) = (vk_of(dest), vk_of(src));
        assert_eq!(d.size, s.size, "TGERR: copy size mismatch");
        match (d.alloc.buf.ptr, s.alloc.buf.ptr) {
            (Some(dp), Some(sp)) => {
                dev.wait_idle();
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        sp.add(s.offset),
                        dp.add(d.offset),
                        s.size,
                    )
                };
            }
            // Device-to-device: the queue orders it after any producer, so no
            // host round-trip is needed (CL's `clEnqueueCopyBuffer` is async too).
            _ => dev.copy_buffer(
                &s.alloc.buf,
                s.offset as u64,
                &d.alloc.buf,
                d.offset as u64,
                s.size as u64,
                false,
            ),
        }
    }

    fn _offset(&self, buf: &dyn OpaqueT, size: usize, offset: usize) -> Rc<dyn OpaqueT> {
        let o = vk_of(buf);
        assert!(
            offset + size <= o.size,
            "TGERR: VK offset view out of range ({offset}+{size} > {})",
            o.size
        );
        Rc::new(VkOpaque {
            alloc: o.alloc.clone(),
            offset: o.offset + offset,
            size,
        })
    }

    fn memory_info(&self) -> Option<MemoryInfo> {
        Some(MemoryInfo {
            total: self.inner.dev.device_local_memory(),
            free: None,
        })
    }
}

// ---------------- program ----------------

/// A compiled SPIR-V program: the pipeline plus the binding table.
struct VkProgram {
    inner: Rc<VkInner>,
    name: String,
    pipeline: vk_sys::GpuPipeline,
    /// Descriptor sets, one per command-buffer ring slot.
    sets: Vec<ash::vk::DescriptorSet>,
    /// Buffer slot per descriptor binding.
    slots: Vec<i64>,
    /// Workgroup size baked into the module.
    local: [u32; 3],
}

impl VkProgram {
    fn new(inner: Rc<VkInner>, elf: &TinyElf) -> VkProgram {
        let text = std::str::from_utf8(&elf.lib).expect("TGERR: VK elF lib is not utf8");
        let m = VkModule::decode(text)
            .unwrap_or_else(|| panic!("TGERR: bad VK payload for {}", elf.name));
        if let Some(dir) = dump_dir() {
            let mut bytes = Vec::with_capacity(m.words.len() * 4);
            for w in &m.words {
                bytes.extend_from_slice(&w.to_le_bytes());
            }
            let path = std::path::Path::new(&dir).join(format!("{}.spv", elf.name));
            let _ = std::fs::create_dir_all(&dir);
            let _ = std::fs::write(path, bytes);
        }
        if trace_on() {
            trace(&format!(
                "program {} bindings={} local={:?}",
                elf.name,
                m.slots.len(),
                m.local
            ));
        }
        let pipeline = inner
            .dev
            .pipeline(&m.words, m.slots.len() as u32)
            .unwrap_or_else(|e| {
                panic!(
                    "TGERR: VK pipeline {} failed: {e}\n--- dump ---\n{}",
                    elf.name,
                    dump(&m.words)
                )
            });
        let sets = inner
            .dev
            .alloc_sets(pipeline.set_layout, vk_sys::SETS_PER_PROGRAM as u32);
        VkProgram {
            inner,
            name: elf.name.clone(),
            sets,
            pipeline,
            slots: m.slots,
            local: m.local,
        }
    }
}

/// `spirv-dis`-ish hex dump for a pipeline failure (no external tools needed).
fn dump(words: &[u32]) -> String {
    let mut s = String::new();
    for (i, w) in words.iter().enumerate() {
        if i % 8 == 0 {
            s.push_str(&format!("\n{i:04}: "));
        }
        s.push_str(&format!("{w:08x} "));
    }
    s
}

impl Prog for VkProgram {
    fn call(
        &mut self,
        bufs: &[Rc<dyn OpaqueT>],
        global_size: [usize; 3],
        local_size: Option<[usize; 3]>,
        _vals: &[i64],
        wait: bool,
    ) -> Option<f64> {
        // The workgroup size is baked into the module; the scheduler derives the
        // same value from the `lidx` extents, so a mismatch is a renderer bug.
        if let Some(ls) = local_size {
            let ls = [ls[0] as u32, ls[1] as u32, ls[2] as u32];
            assert_eq!(
                ls, self.local,
                "TGERR: VK: launch local size {ls:?} != shader LocalSize {:?}",
                self.local
            );
        }
        let mut bound: Vec<(&GpuBuffer, u64, u64)> = Vec::with_capacity(self.slots.len());
        for slot in &self.slots {
            let o = vk_of(&*bufs[*slot as usize]);
            bound.push((&o.alloc.buf, o.offset as u64, o.size as u64));
        }
        let groups = [
            global_size[0] as u32,
            global_size[1] as u32,
            global_size[2] as u32,
        ];
        if trace_on() {
            trace(&format!(
                "dispatch {} groups={groups:?} local={:?} bufs={}",
                self.name,
                self.local,
                bound.len()
            ));
        }
        // `VK_SYNC=1` waits after every dispatch, so a faulting kernel is
        // attributable (a lost device otherwise hides which one it was).
        let wait = wait || sync_on();
        self.inner
            .dev
            .dispatch(&self.pipeline, &self.sets, &bound, groups, wait, &self.name)
            .unwrap_or_else(|e| panic!("TGERR: VK dispatch failed: {e}"));
        None
    }

    fn flush(&mut self) {
        // Batched launches are submitted at the linear boundary (and before any
        // host access), so the GPU is never idle while the host builds a graph.
        let _ = self.inner.dev.flush();
    }
}

// ---------------- compiler ----------------

/// `VK` "compiler": the rendered payload already holds the SPIR-V module, so
/// the seam only carries it through (the module is built at pipeline creation).
struct VkCompiler;

impl Comp for VkCompiler {
    fn compile(&self, src: &str) -> Vec<u8> {
        src.as_bytes().to_vec()
    }
    fn compile_cached(&self, src: &str) -> Vec<u8> {
        src.as_bytes().to_vec()
    }
}

// ---------------- renderer ----------------

// ---------------- device + registration ----------------

/// Whether Vulkan can run this backend here: loader, a compute device with the
/// features the renderer advertises, and a device that creates a pipeline.
pub fn available() -> bool {
    !vk_devices().is_empty()
}

/// Whether this device can run the cooperative-matrix GEMM
/// (`runtime::coopmat`): the extension must be present and the advertised MMA
/// shape must cover the module's tile.
pub fn coop_gemm_available() -> bool {
    coop_shape().is_some()
}

/// Number of cooperative-matrix GEMM dispatches since process start (diagnostics
/// and tests: proves the library-GEMM seam took the tensor-core path).
pub fn coop_gemm_count() -> usize {
    COOP_GEMMS.with(|c| c.get())
}

thread_local! {
    static COOP_GEMMS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The cooperative-matrix shape of the first usable device, when the module's
/// 16x16x16 f16 -> f32 tile is among the advertised ones.
fn coop_shape() -> Option<vk_sys::CoopShape> {
    let s = vk_devices().first()?.coop?;
    (s.m as usize >= coopmat::M_TILE
        && s.n as usize >= coopmat::M_TILE
        && s.k as usize >= coopmat::K_TILE)
        .then_some(s)
}

/// One element-wise conversion module: `bf16 -> f16` or `f32 -> bf16`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Conversion {
    /// `f32` (4-byte) to `bf16` (2-byte), round to nearest even.
    F32ToBf16,
}

impl Conversion {
    fn id(self) -> u8 {
        match self {
            Conversion::F32ToBf16 => 1,
        }
    }
    fn words(self, n: usize) -> Option<Vec<u32>> {
        match self {
            Conversion::F32ToBf16 => coopmat::convert_f32_to_bf16(n),
        }
    }
}

impl VkInner {
    /// A scratch buffer of at least `size` bytes for `slot` (0 = A, 1 = B,
    /// 2 = C), reused across calls.
    fn scratch(&self, slot: u8, size: usize) -> vk_sys::GpuBuffer {
        let mut pool = self.scratch.borrow_mut();
        if let Some(b) = pool.get(&slot)
            && b.size >= size
        {
            return b.clone();
        }
        let b = self
            .dev
            .alloc(size)
            .unwrap_or_else(|| panic!("TGERR: VK coopmat scratch {size} bytes"));
        pool.insert(slot, b.clone());
        b
    }
}

/// A device buffer plus the window `coop_gemm` binds for it.
#[derive(Clone, Copy)]
struct Bind<'a> {
    buf: &'a vk_sys::GpuBuffer,
    off: u64,
    size: u64,
}

/// Record `n` element conversions from `src` to `dst`.
fn convert(
    inner: &VkInner,
    c: Conversion,
    n: usize,
    src: Bind<'_>,
    dst: Bind<'_>,
) {
    let words = c.words(n).expect("coopmat: conversion module");
    let key = (c.id(), n, 0, 0, 0);
    let (pipe, sets) = {
        let mut cache = inner.conv.borrow_mut();
        if let Some(p) = cache.get(&key) {
            p.clone()
        } else {
            let p = Rc::new(
                inner
                    .dev
                    .pipeline(&words, 2)
                    .unwrap_or_else(|e| panic!("TGERR: VK convert pipeline: {e}")),
            );
            let sets = inner
                .dev
                .alloc_sets(p.set_layout, vk_sys::SETS_PER_PROGRAM as u32);
            cache.insert(key, (p.clone(), sets.clone()));
            (p, sets)
        }
    };
    let bound = [(src.buf, src.off, src.size), (dst.buf, dst.off, dst.size)];
    inner
        .dev
        .dispatch(
            &pipe,
            &sets,
            &bound,
            coopmat::element_grid(n),
            false,
            "convert",
        )
        .unwrap_or_else(|e| panic!("TGERR: VK convert dispatch: {e}"));
}

/// Record a `rows x cols` bf16 -> row-major f16 staging pass.
#[allow(clippy::too_many_arguments)]
fn stage(
    inner: &VkInner,
    kind: coopmat::Stage,
    rows: usize,
    cols: usize,
    in_rs: usize,
    in_cs: usize,
    src: Bind<'_>,
    dst: Bind<'_>,
) {
    let words = match kind {
        coopmat::Stage::Bf16 => coopmat::stage_bf16(rows, cols, in_rs, in_cs),
        coopmat::Stage::F32 => coopmat::stage_f32(rows, cols, in_rs, in_cs),
        coopmat::Stage::Direct => return,
    }
    .expect("coopmat: staging module");
    let key = (2, rows, cols, in_rs, in_cs);
    let (pipe, sets) = {
        let mut cache = inner.conv.borrow_mut();
        if let Some(p) = cache.get(&key) {
            p.clone()
        } else {
            let p = Rc::new(
                inner
                    .dev
                    .pipeline(&words, 2)
                    .unwrap_or_else(|e| panic!("TGERR: VK stage pipeline: {e}")),
            );
            let sets = inner
                .dev
                .alloc_sets(p.set_layout, vk_sys::SETS_PER_PROGRAM as u32);
            cache.insert(key, (p.clone(), sets.clone()));
            (p, sets)
        }
    };
    if trace_on() {
        trace(&format!(
            "stage {rows}x{cols} rs={in_rs} cs={in_cs} count={}",
            rows * cols
        ));
    }
    let bound = [(src.buf, src.off, src.size), (dst.buf, dst.off, dst.size)];
    inner
        .dev
        .dispatch(
            &pipe,
            &sets,
            &bound,
            coopmat::element_grid(rows * cols),
            false,
            "stage",
        )
        .unwrap_or_else(|e| panic!("TGERR: VK stage dispatch: {e}"));
}

/// Record the fused bias + round epilogue over an `m x n` f32 accumulator.
fn epilogue(
    inner: &VkInner,
    kind: coopmat::Epilogue,
    n: usize,
    count: usize,
    src: Bind<'_>,
    bias: Bind<'_>,
    dst: Bind<'_>,
) {
    let words = match kind {
        coopmat::Epilogue::BiasBf16OutBf16 => coopmat::epilogue_bf16(n, count),
        coopmat::Epilogue::BiasBf16OutF32 => coopmat::epilogue_bf16_bias_f32(n, count),
        coopmat::Epilogue::BiasF32OutF32 => coopmat::epilogue_f32(n, count),
        coopmat::Epilogue::None => return,
    }
    .expect("coopmat: epilogue module");
    let key = (3, n, count, kind as usize, 0);
    let (pipe, sets) = {
        let mut cache = inner.conv.borrow_mut();
        if let Some(p) = cache.get(&key) {
            p.clone()
        } else {
            let p = Rc::new(
                inner
                    .dev
                    .pipeline(&words, 3)
                    .unwrap_or_else(|e| panic!("TGERR: VK epilogue pipeline: {e}")),
            );
            let sets = inner
                .dev
                .alloc_sets(p.set_layout, vk_sys::SETS_PER_PROGRAM as u32);
            cache.insert(key, (p.clone(), sets.clone()));
            (p, sets)
        }
    };
    let bound = [
        (src.buf, src.off, src.size),
        (bias.buf, bias.off, bias.size),
        (dst.buf, dst.off, dst.size),
    ];
    if trace_on() {
        trace(&format!("epilogue n={n} count={count}"));
    }
    inner
        .dev
        .dispatch(
            &pipe,
            &sets,
            &bound,
            coopmat::element_grid(count),
            false,
            "epilogue",
        )
        .unwrap_or_else(|e| panic!("TGERR: VK epilogue dispatch: {e}"));
}

/// Run one `HostGemm` descriptor as a cooperative-matrix kernel.
///
/// The library-GEMM seam (`schedule::host_gemm`) has already checked
/// `coopmat::eligible`, so the layouts are contiguous row-major. The module reads
/// f16 operands and accumulates in f32, so `bf16` storage is staged through f16
/// scratch buffers (exact for in-range values: f16 has *more* mantissa bits than
/// bf16, only a narrower exponent) and a bf16 result is rounded from an f32
/// scratch. Dispatches are recorded into the open command buffer like any other
/// kernel, so the queue orders them after their producers.
fn coop_gemm(inner: &VkInner, bufs: &[crate::runtime::buffer::OpaqueView], g: &HostGemm) {
    let (m, n, k) = (g.m as usize, g.n as usize, g.k as usize);
    let dev = &inner.dev;
    let view = |slot: i64| {
        let o = vk_of(&*bufs[slot as usize]);
        Bind {
            buf: &o.alloc.buf,
            off: o.offset as u64,
            size: o.size as u64,
        }
    };
    // The bias vector is 2 bytes per element, so its element offset scales.
    let view_elem = |slot: i64, off: i64, elem_bytes: u64| {
        let mut b = view(slot);
        b.off += (off as u64) * elem_bytes;
        b
    };
    let (a, b, c) = (view(g.a_slot), view(g.b_slot), view(g.c_slot));

    // Operands that are not already f16 are staged into packed f16 tiles inside
    // the per-batch loop below (the pass also transposes a column-major operand).
    let plan = coopmat::plan(g).expect("coopmat: eligible descriptor");
    let nb = g.batch as usize;
    let a_slice = m * k;
    let c_slice = m * n;
    // A batch slice starts exactly one slice after the previous one when the
    // tensor is contiguous — then the whole tensor can be staged (and the whole
    // result converted) in *one* dispatch instead of one per slice.
    let a_packed = nb == 1 || g.a_b == m as i64 * plan.sa1 as i64;
    let c_packed = nb == 1 || g.c_b == m as i64 * plan.sc as i64;
    let need_a = plan.stage_a != coopmat::Stage::Direct;
    let need_b = plan.stage_b != coopmat::Stage::Direct;
    let need_scratch =
        g.c_dtype == crate::dtype::BFLOAT16 || plan.epilogue != coopmat::Epilogue::None;

    let pack_a = if need_a {
        Some(inner.scratch(0, if a_packed { nb } else { 1 } * a_slice * 2))
    } else {
        None
    };
    let pack_b = if need_b {
        Some(inner.scratch(1, k * n * 2))
    } else {
        None
    };
    fn bind_whole(b: &vk_sys::GpuBuffer) -> Bind<'_> {
        Bind {
            buf: b,
            off: 0,
            size: b.size as u64,
        }
    }
    let pa = pack_a.as_ref().map(bind_whole);
    let pb = pack_b.as_ref().map(bind_whole);
    let scratch_c = if need_scratch {
        Some(inner.scratch(2, if c_packed { nb } else { 1 } * c_slice * 4))
    } else {
        None
    };
    let pc = scratch_c.as_ref().map(bind_whole);
    let elem_a: u64 = if g.a_dtype == crate::dtype::FLOAT32 { 4 } else { 2 };
    let elem_b: u64 = if g.b_dtype == crate::dtype::FLOAT32 { 4 } else { 2 };
    let elem_c: u64 = if g.c_dtype == crate::dtype::FLOAT32 { 4 } else { 2 };

    let dense = coopmat::Plan {
        sa1: k,
        sa2: 1,
        la: 0,
        sla: k,
        sb1: n,
        sb2: 1,
        lb: 0,
        slb: n,
        ..plan
    };
    let words = coopmat::module(dense).expect("coopmat: module");
    let key = (m, n, k, g.dtype, g.c_dtype);
    let (pipe, sets) = {
        let mut cache = inner.coop.borrow_mut();
        if let Some(p) = cache.get(&key) {
            p.clone()
        } else {
            let p = Rc::new(
                dev.pipeline(&words, 3)
                    .unwrap_or_else(|e| panic!("TGERR: VK coopmat pipeline: {e}")),
            );
            let sets = dev.alloc_sets(p.set_layout, vk_sys::SETS_PER_PROGRAM as u32);
            cache.insert(key, (p.clone(), sets.clone()));
            (p, sets)
        }
    };
    let groups = coopmat::grid(m, n);
    if trace_on() {
        trace(&format!(
            "coopmat {m}x{n}x{k} batch={nb} packed={a_packed}/{c_packed} groups={groups:?} {:?}",
            g.dtype
        ));
    }

    // Stage B once: it is either shared across the batch or packed.
    if need_b && g.b_b == 0 {
        stage(inner, plan.stage_b, k, n, plan.sb1, plan.sb2, b, pb.unwrap());
    }
    // Stage A once when the batch is contiguous, otherwise per slice below.
    if need_a && a_packed {
        stage(inner, plan.stage_a, nb * m, k, plan.sa1, plan.sa2, a, pa.unwrap());
    }

    for bi in 0..nb as i64 {
        let biu = bi as usize;
        let off_a = (g.a_b * bi) as u64 * elem_a;
        let off_b = (g.b_b * bi) as u64 * elem_b;
        let off_c = (g.c_b * bi) as u64 * elem_c;
        if need_a && !a_packed {
            let src = Bind {
                off: a.off + off_a,
                size: a.size.saturating_sub(off_a),
                ..a
            };
            stage(inner, plan.stage_a, m, k, plan.sa1, plan.sa2, src, pa.unwrap());
        }
        if need_b && g.b_b != 0 {
            let src = Bind {
                off: b.off + off_b,
                size: b.size.saturating_sub(off_b),
                ..b
            };
            stage(inner, plan.stage_b, k, n, plan.sb1, plan.sb2, src, pb.unwrap());
        }
        let a_final = match pa {
            Some(pa) => Bind {
                off: if a_packed {
                    (biu * a_slice) as u64 * 2
                } else {
                    0
                },
                ..pa
            },
            None => Bind {
                off: a.off + off_a,
                size: a.size.saturating_sub(off_a),
                ..a
            },
        };
        let b_final = pb.unwrap_or(Bind {
            off: b.off + off_b,
            size: b.size.saturating_sub(off_b),
            ..b
        });
        // Where the MMA writes this slice's accumulator.
        let (acc, acc_off) = match pc {
            Some(pc) => {
                let off = if c_packed {
                    (biu * c_slice) as u64 * 4
                } else {
                    0
                };
                (pc, off)
            }
            None => (c, off_c),
        };
        let bound = [
            (a_final.buf, a_final.off, a_final.size),
            (b_final.buf, b_final.off, b_final.size),
            (acc.buf, acc.off + acc_off, acc.size.saturating_sub(acc_off)),
        ];
        COOP_GEMMS.with(|c| c.set(c.get() + 1));
        dev.dispatch(&pipe, &sets, &bound, groups, false, "coopmat")
            .unwrap_or_else(|e| panic!("TGERR: VK coopmat dispatch: {e}"));
        // A packed batch is converted once after the loop.
        if !c_packed {
            let slice_in = Bind {
                off: acc.off + acc_off,
                size: acc.size.saturating_sub(acc_off),
                ..acc
            };
            let slice_out = Bind {
                off: c.off + off_c,
                size: c.size.saturating_sub(off_c),
                ..c
            };
            match plan.epilogue {
                coopmat::Epilogue::None => {
                    if g.c_dtype == crate::dtype::BFLOAT16 {
                        convert(inner, Conversion::F32ToBf16, c_slice, slice_in, slice_out);
                    }
                }
                kind => {
                    let elem = if g.bias_dtype == crate::dtype::FLOAT32 { 4 } else { 2 };
                    let bias = view_elem(g.bias_slot, g.bias_off, elem);
                    epilogue(inner, kind, n, c_slice, slice_in, bias, slice_out);
                }
            }
        }
    }
    // One conversion for the whole packed batch.
    if c_packed && let Some(pc) = pc {
        let count = nb * c_slice;
        let dst = Bind {
            off: c.off,
            size: c.size,
            ..c
        };
        match plan.epilogue {
            coopmat::Epilogue::None => {
                if g.c_dtype == crate::dtype::BFLOAT16 {
                    convert(inner, Conversion::F32ToBf16, count, pc, dst);
                }
            }
            kind => {
                let elem = if g.bias_dtype == crate::dtype::FLOAT32 { 4 } else { 2 };
                let bias = view_elem(g.bias_slot, g.bias_off, elem);
                epilogue(inner, kind, n, count, pc, bias, dst);
            }
        }
    }
}

/// Number of usable Vulkan devices.
pub fn device_count() -> usize {
    vk_devices().len()
}

/// Whether the `index`-th device passes the probe (cached).
pub fn device_usable(index: usize) -> bool {
    index < vk_devices().len()
}

/// The kernel-buffer limit of the `index`-th device.
fn max_buffers(index: usize) -> usize {
    vk_devices()
        .get(index)
        .map(|d| {
            (d.props.limits.max_per_stage_descriptor_storage_buffers as usize).max(1)
        })
        .unwrap_or(usize::MAX)
}

/// A factory for the device named `name` (`VK` or `VK:N`).
fn factory(name: String) -> DeviceFactory {
    Rc::new(move || {
        let index = name
            .split_once(':')
            .and_then(|(_, i)| i.parse::<usize>().ok())
            .unwrap_or(0);
        let inner = Rc::new(VkInner::open(&name));
        let info = vk_devices()[index].clone();
        let caps: Caps = info.caps;
        let lim = info.props.limits;
        // The scheduler's tiling depends on these: use what the device reports
        // (Vulkan's guaranteed minimum is far smaller than a desktop GPU's).
        let global = [
            lim.max_compute_work_group_count[0] as i64,
            lim.max_compute_work_group_count[1] as i64,
            lim.max_compute_work_group_count[2] as i64,
        ];
        let local = [
            (lim.max_compute_work_group_size[0] as i64).min(1024),
            (lim.max_compute_work_group_size[1] as i64).min(1024),
            (lim.max_compute_work_group_size[2] as i64).min(64),
        ];
        let mut facts = spirv::facts_with_limits(
            &caps,
            global,
            local,
            lim.max_compute_shared_memory_size as i64,
        );
        // Bound the scheduler's fusion budget by the device's descriptor limit.
        facts.max_buffers = max_buffers(index);
        let renderer: Box<dyn crate::renderer::UopRenderer> =
            Box::new(VkUopRenderer::with_facts(caps, facts));
        let compiler: Rc<dyn Comp> = Rc::new(VkCompiler);
        let allocator = LruAllocator::with_backend(Box::new(VkAllocator {
            inner: inner.clone(),
        }));
        let ctor_inner = inner.clone();
        let ctor: crate::runtime::device::RuntimeCtor = Rc::new(move |elf| {
            Rc::new(RefCell::new(VkProgram::new(ctor_inner.clone(), elf)))
        });
        let compiled = Compiled::new(&name, renderer, allocator, compiler, ctor);
        if coop_gemm_available() {
            // The library-GEMM seam (`schedule::host_gemm`): a kernel that is
            // exactly a contiguous f16 matmul with an f32 accumulator becomes one
            // cooperative-matrix dispatch instead of a generated reduce loop.
            let he_inner = inner.clone();
            let f: Box<crate::runtime::device::HostExec> = Box::new(move |bufs, desc| {
                let g = HostGemm::from_arg(&crate::uop::arg::Arg::Data(desc.clone()))
                    .unwrap_or_else(|| panic!("TGERR: bad HostGemm descriptor"));
                coop_gemm(&he_inner, bufs, &g);
            });
            compiled.host_exec.replace(Some(f));
        }
        compiled
    })
}

/// The `VKDevice` composition + `Device._DeviceMap["VK"]` assignment.
pub fn register() {
    // A binary built with `backend_vk` must start on machines without Vulkan;
    // only register the device when the loader and a usable device are there.
    if !available() {
        return;
    }
    register_device_with_limit("VK", max_buffers(0), factory("VK".into()));
    register_device_family("VK", Rc::new(|name| factory(name.to_string())));
}

/// Dtype support for diagnostics/tests.
pub fn supported_dtype(dt: DTypeId) -> bool {
    let caps = vk_devices().first().map(|d| d.caps).unwrap_or_default();
    spirv::supported_dtype(dt, &caps)
}

/// Diagnostic: run a raw SPIR-V module on the first usable Vulkan device.
///
/// Allocates one storage buffer per binding (`sizes[i]` bytes, descriptor
/// binding `i`), dispatches `groups` workgroups of `local` invocations `iters`
/// times with a fence wait per dispatch, and returns the median milliseconds.
/// This exists so the Vulkan IR can be A/B-tested against another compiler's
/// output for the same kernel; it is a development tool, not a stable API.
pub fn run_raw_module(
    spv: &[u32],
    sizes: &[usize],
    groups: [u32; 3],
    iters: usize,
) -> Option<f64> {
    let (ms, _) = run_raw_module_io(spv, sizes, &[], groups, iters)?;
    Some(ms)
}

/// [`run_raw_module`] with host-side input/output: `fills[i]` preloads buffer `i`
/// (empty = zeroed), and the device contents are read back afterwards.
///
/// This is what lets a hand-written module be checked numerically as well as
/// timed — the same two things you need before trusting a new instruction path.
pub fn run_raw_module_io(
    spv: &[u32],
    sizes: &[usize],
    fills: &[&[u8]],
    groups: [u32; 3],
    iters: usize,
) -> Option<(f64, Vec<Vec<u8>>)> {
    let devs = vk_devices();
    let info = devs.first()?;
    let lib = vk_sys::VkLib::get()?;
    let dev = match vk_sys::VkDevice::open(&lib.instance, lib.api, info) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[vk] raw: {e}");
            return None;
        }
    };
    let pipe = match dev.pipeline(spv, sizes.len() as u32) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[vk] raw pipeline: {e}");
            return None;
        }
    };
    let sets = dev.alloc_sets(pipe.set_layout, vk_sys::SETS_PER_PROGRAM as u32);
    let bufs: Vec<vk_sys::GpuBuffer> = sizes
        .iter()
        .map(|s| dev.alloc(*s).expect("vk raw alloc"))
        .collect();
    for (i, b) in bufs.iter().enumerate() {
        if let Some(src) = fills.get(i)
            && !src.is_empty()
        {
            assert_eq!(src.len(), b.size, "raw fill {i} size");
            dev.upload(b, src);
        }
    }
    let bound: Vec<(&vk_sys::GpuBuffer, u64, u64)> =
        bufs.iter().map(|b| (b, 0, b.size as u64)).collect();
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..=iters {
        let t = std::time::Instant::now();
        if let Err(e) = dev.dispatch(&pipe, &sets, &bound, groups, true, "raw") {
            eprintln!("[vk] raw dispatch: {e}");
            return None;
        }
        samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(f64::total_cmp);
    dev.wait_idle();
    let out = bufs
        .iter()
        .map(|b| {
            let mut v = vec![0u8; b.size];
            dev.download(b, &mut v);
            v
        })
        .collect();
    Some((samples[samples.len() / 2], out))
}
