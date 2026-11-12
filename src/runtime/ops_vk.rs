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
use crate::runtime::vk_sys::{self, DeviceInfo, GpuBuffer};

// ---------------- device inner ----------------

/// The opened device shared by the allocator and every program.
struct VkInner {
    dev: Rc<vk_sys::VkDevice>,
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
        VkInner { dev: Rc::new(dev) }
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
            trace(&format!("{}: usable, caps {:?}", info.name, info.caps));
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
        let ctor: crate::runtime::device::RuntimeCtor = Rc::new(move |elf| {
            Rc::new(RefCell::new(VkProgram::new(inner.clone(), elf)))
        });
        Compiled::new(&name, renderer, allocator, compiler, ctor)
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
    let devs = vk_devices();
    let info = devs.first()?;
    let lib = vk_sys::VkLib::get()?;
    let dev = vk_sys::VkDevice::open(&lib.instance, lib.api, info).ok()?;
    let pipe = dev.pipeline(spv, sizes.len() as u32).ok()?;
    let sets = dev.alloc_sets(pipe.set_layout, vk_sys::SETS_PER_PROGRAM as u32);
    let bufs: Vec<vk_sys::GpuBuffer> = sizes
        .iter()
        .map(|s| dev.alloc(*s).expect("vk raw alloc"))
        .collect();
    let bound: Vec<(&vk_sys::GpuBuffer, u64, u64)> =
        bufs.iter().map(|b| (b, 0, b.size as u64)).collect();
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..=iters {
        let t = std::time::Instant::now();
        dev.dispatch(&pipe, &sets, &bound, groups, true, "raw").ok()?;
        samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(f64::total_cmp);
    Some(samples[samples.len() / 2])
}
