//! Dynamic Vulkan plumbing for the `VK` backend.
//!
//! `ash` is used only for the bindings: its `loaded` feature dlopens
//! `libvulkan.so.1` (or the platform equivalent) at run time, so a binary built
//! with `backend_vk` starts and reports "no device" on machines without a
//! Vulkan loader, a driver, or the features this backend needs. Nothing links
//! against the loader at build time.
//!
//! The module keeps the surface small and RAII-ish:
//!
//! - [`VkLib`] — the loader `Entry` plus one `Instance` (one per thread, like the
//!   rest of the runtime).
//! - [`DeviceInfo`] — a physical device that passed the capability probe.
//! - [`VkDevice`] — the opened device: one compute queue, one command buffer,
//!   one fence, host-visible+coherent memory, and descriptor/pipeline helpers.
//!
//! Memory model: every allocation is host-visible and host-coherent and stays
//! mapped, so `_copyin`/`_copyout` are `memcpy` (the same choice Metal makes
//! with `MTLResourceStorageModeShared`). Kernels read and write those buffers
//! directly as storage buffers.

use std::ffi::CStr;
use std::sync::OnceLock;

use ash::vk;

use crate::renderer::spirv::Caps;

/// The Vulkan loader plus a created instance.
pub struct VkLib {
    /// Kept alive: the instance's function pointers come from it.
    _entry: ash::Entry,
    /// The instance.
    pub instance: ash::Instance,
    /// The instance's API version.
    pub api: u32,
}

/// Constructor for a process-wide (per-thread is not needed: `Instance` is
/// `Send + Sync`) loaded library.
impl VkLib {
    /// Load `libvulkan` and create an instance. `None` when there is no loader.
    pub fn get() -> Option<&'static VkLib> {
        static LIB: OnceLock<Option<VkLib>> = OnceLock::new();
        LIB.get_or_init(|| unsafe {
            let entry = match ash::Entry::load() {
                Ok(e) => e,
                Err(_) => return None,
            };
            // Prefer 1.3 (feature chains, `vkGetPhysicalDeviceFeatures2` in
            // core), but accept 1.0: everything used here is 1.0 except the
            // feature query, which is 1.1.
            let avail = entry
                .try_enumerate_instance_version()
                .ok()
                .flatten()
                .unwrap_or(vk::API_VERSION_1_0);
            let api = avail.min(vk::API_VERSION_1_3);
            let app = vk::ApplicationInfo::default()
                .application_name(c"autorust")
                .application_version(vk::make_api_version(0, 0, 1, 1))
                .engine_name(c"autorust")
                .api_version(api);
            let info = vk::InstanceCreateInfo::default().application_info(&app);
            match entry.create_instance(&info, None) {
                Ok(instance) => Some(VkLib {
                    _entry: entry,
                    instance,
                    api,
                }),
                Err(_) => None,
            }
        })
        .as_ref()
    }

    /// The loader entry (for extension entry points).
    pub fn entry(&self) -> &ash::Entry {
        &self._entry
    }

    /// Physical devices with a compute queue (best first).
    pub fn devices(&self) -> Vec<DeviceInfo> {
        let Ok(pds) = (unsafe { self.instance.enumerate_physical_devices() }) else {
            return Vec::new();
        };
        let mut out: Vec<DeviceInfo> = Vec::new();
        for pd in pds {
            let props = unsafe { self.instance.get_physical_device_properties(pd) };
            let qf = unsafe { self.instance.get_physical_device_queue_family_properties(pd) };
            let queue_family = qf.iter().position(|q| {
                q.queue_flags.contains(vk::QueueFlags::COMPUTE) && q.queue_count > 0
            });
            let Some(queue_family) = queue_family else {
                continue;
            };
            let caps = unsafe { query_caps(&self.instance, pd, self.api) };
            let name = props
                .device_name_as_c_str()
                .map(|c| c.to_string_lossy().into_owned())
                .unwrap_or_else(|_| "Vulkan device".to_string());
            out.push(DeviceInfo {
                pd,
                name,
                kind: props.device_type,
                queue_family: queue_family as u32,
                api: props.api_version.min(self.api),
                props,
                caps,
                coop: unsafe { query_coop(&self._entry, &self.instance, pd) },
            });
        }
        // Discrete > integrated > virtual > CPU, stable by name.
        out.sort_by_key(|d| {
            let rank = match d.kind {
                vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                vk::PhysicalDeviceType::CPU => 3,
                _ => 4,
            };
            (rank, d.name.clone())
        });
        out
    }
}

/// A cooperative-matrix ("tensor core") configuration the device supports for
/// `A[m,k] * B[k,n] + C[m,n]` with 16-bit inputs and an f32 accumulator.
///
/// `VK_KHR_cooperative_matrix` has no bf16 component type, so this is the f16
/// form: bf16 operands would have to be converted to f16 first, which changes
/// both the exponent range and the mantissa precision. It is therefore used for
/// `f16` graphs (and any graph whose accumulator dtype is f32), not to silently
/// reinterpret a bf16 dot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoopShape {
    /// Tile rows (also the result rows).
    pub m: u32,
    /// Tile columns.
    pub n: u32,
    /// Contraction depth of one `OpCooperativeMatrixMulAddKHR`.
    pub k: u32,
    /// `VkScopeKHR` the matrix must be built in (`3` = `Subgroup`).
    pub scope: i32,
}

/// A physical device that can run this backend.
#[derive(Clone)]
pub struct DeviceInfo {
    /// The physical device handle.
    pub pd: vk::PhysicalDevice,
    /// `VkPhysicalDeviceProperties.deviceName`.
    pub name: String,
    /// Discrete/integrated/CPU.
    pub kind: vk::PhysicalDeviceType,
    /// The queue family used for compute.
    pub queue_family: u32,
    /// Effective API version.
    pub api: u32,
    /// Physical device properties (limits, memory heaps).
    pub props: vk::PhysicalDeviceProperties,
    /// Feature probe result.
    pub caps: Caps,
    /// The cooperative-matrix configuration to use, when the device has one.
    pub coop: Option<CoopShape>,
}

/// The device extension that provides `OpCooperativeMatrix*KHR`.
const COOPMAT_EXT: &std::ffi::CStr = c"VK_KHR_cooperative_matrix";

/// Find a cooperative-matrix shape: 16-bit A/B, f32 C, subgroup scope.
///
/// This is a *capability* probe only — the enumerant must be matched exactly
/// (the driver accepts only the shapes it reports), which is why the shape is
/// queried instead of hard-coded.
unsafe fn query_coop(
    entry: &ash::Entry,
    instance: &ash::Instance,
    pd: vk::PhysicalDevice,
) -> Option<CoopShape> {
    let exts = unsafe { instance.enumerate_device_extension_properties(pd) }.ok()?;
    let has = exts
        .iter()
        .filter_map(|e| e.extension_name_as_c_str().ok())
        .any(|n| n == COOPMAT_EXT);
    if !has {
        return None;
    }
    let cm = ash::khr::cooperative_matrix::Instance::new(entry, instance);
    let props = unsafe { cm.get_physical_device_cooperative_matrix_properties(pd) }.ok()?;
    props
        .iter()
        .filter(|p| {
            p.a_type == vk::ComponentTypeKHR::FLOAT16
                && p.b_type == vk::ComponentTypeKHR::FLOAT16
                && p.c_type == vk::ComponentTypeKHR::FLOAT32
                && p.result_type == vk::ComponentTypeKHR::FLOAT32
                && p.scope == vk::ScopeKHR::SUBGROUP
                && p.saturating_accumulation == vk::FALSE
        })
        .max_by_key(|p| (p.m_size * p.n_size, p.k_size))
        .map(|p| CoopShape {
            m: p.m_size,
            n: p.n_size,
            k: p.k_size,
            scope: p.scope.as_raw(),
        })
}

/// Query the features the renderer would use.
unsafe fn query_caps(instance: &ash::Instance, pd: vk::PhysicalDevice, api: u32) -> Caps {
    let mut storage16 = vk::PhysicalDevice16BitStorageFeatures::default();
    let mut storage8 = vk::PhysicalDevice8BitStorageFeatures::default();
    let mut f16i8 = vk::PhysicalDeviceShaderFloat16Int8Features::default();
    let base;
    if api >= vk::API_VERSION_1_1 {
        {
            let mut f2 = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut storage16)
                .push_next(&mut storage8)
                .push_next(&mut f16i8);
            unsafe { instance.get_physical_device_features2(pd, &mut f2) };
            base = f2.features;
        }
    } else {
        base = unsafe { instance.get_physical_device_features(pd) };
    }
    Caps {
        f16: f16i8.shader_float16 == vk::TRUE
            && storage16.storage_buffer16_bit_access == vk::TRUE,
        storage16: storage16.storage_buffer16_bit_access == vk::TRUE,
        int8: f16i8.shader_int8 == vk::TRUE,
        int16: base.shader_int16 == vk::TRUE,
        int64: base.shader_int64 == vk::TRUE,
        f64: base.shader_float64 == vk::TRUE,
        storage8: storage8.storage_buffer8_bit_access == vk::TRUE,
        // SPIR-V 1.3 (`StorageBuffer` + `Block`) needs Vulkan 1.1.
        spirv13: api >= vk::API_VERSION_1_1,
    }
}

/// Dispatches recorded into one command buffer before it is submitted.
///
/// One submit + fence wait costs ~25 us on this driver, while recording a
/// dispatch into an already-open command buffer costs ~1 us: batching is what
/// makes a graph with hundreds of small kernels (LLM decode) competitive with
/// OpenCL, where `clEnqueueNDRangeKernel` is nearly free.
const BATCH: usize = 8;

/// Descriptor sets a program must own: one per (ring slot, batch position).
///
/// A set may not be rewritten while the batch referencing it is still in
/// flight, so the index has to include the ring slot, not just the position in
/// the batch: reusing `sets[0]` across consecutive batches makes the driver
/// serialize (measured: 0.9 -> 1.5 ms for a single-kernel step).
pub const SETS_PER_PROGRAM: usize = 4 * BATCH;

/// One GPU buffer (handles only; cloning shares the allocation).
#[derive(Clone)]
pub struct GpuBuffer {
    /// The Vulkan buffer.
    pub buf: vk::Buffer,
    /// Its backing memory.
    pub mem: vk::DeviceMemory,
    /// Persistent host mapping, when the memory type is host-visible.
    pub ptr: Option<*mut u8>,
    /// Allocation size in bytes.
    pub size: usize,
}

/// A compiled compute pipeline and its descriptor layout.
pub struct GpuPipeline {
    /// The shader module (destroyed with the pipeline).
    pub shader: vk::ShaderModule,
    /// Descriptor set layout (one SSBO per binding).
    pub set_layout: vk::DescriptorSetLayout,
    /// Pipeline layout.
    pub layout: vk::PipelineLayout,
    /// The compute pipeline.
    pub pipeline: vk::Pipeline,
    /// Number of storage-buffer bindings.
    pub bindings: u32,
}

/// An opened Vulkan device with one compute queue.
pub struct VkDevice {
    /// The loader/instance this device came from.
    pub instance: &'static ash::Instance,
    /// The logical device.
    pub device: ash::Device,
    /// The physical device.
    pub pd: vk::PhysicalDevice,
    /// The compute queue.
    pub queue: vk::Queue,
    /// The queue family of `queue`.
    pub queue_family: u32,
    /// Effective API version.
    pub api: u32,
    /// Physical device properties.
    pub props: vk::PhysicalDeviceProperties,
    /// The feature set the renderer was built for.
    pub caps: Caps,
    /// Command pool for the reusable command buffers.
    pub cmd_pool: vk::CommandPool,
    /// Ring of command buffers: slot `i` is only reused after its fence
    /// signals, which is what makes resetting it legal while later slots may
    /// still be executing.
    pub cmds: [vk::CommandBuffer; 4],
    /// Descriptor pool for program descriptor sets.
    pub desc_pool: vk::DescriptorPool,
    /// One fence per ring slot.
    pub fences: [vk::Fence; 4],
    /// Next ring slot.
    pub slot: std::cell::Cell<usize>,
    /// Whether work is queued that the host has not waited for.
    pub in_flight: std::cell::Cell<bool>,
    /// Reused host-visible buffer for device-local transfers.
    pub staging: std::cell::RefCell<Option<GpuBuffer>>,
    /// Buffers whose last reference was dropped while work may still be in
    /// flight. Vulkan forbids destroying a buffer a pending command buffer
    /// references, so destruction is deferred to the next idle point.
    pub graveyard: std::cell::RefCell<Vec<GpuBuffer>>,
    /// Timestamp query pool (`BATCH` pairs per ring slot) for `VK_TIME=1`.
    ts_pool: vk::QueryPool,
    /// `timestampPeriod` in nanoseconds.
    ts_period: f32,
    /// Whether the current ring slot's command buffer is open for recording.
    recording: std::cell::Cell<bool>,
    /// Dispatches recorded into the open command buffer.
    batch: std::cell::Cell<usize>,
    /// Kernel names per ring slot, for the timestamp report.
    batch_names: std::cell::RefCell<Vec<Vec<String>>>,
    /// `storage_id` counter, so views of one allocation share an id.
    pub next_id: std::cell::Cell<usize>,
}

fn cstr(p: &[std::ffi::c_char]) -> String {
    let p: &[u8] = unsafe { std::slice::from_raw_parts(p.as_ptr() as *const u8, p.len()) };
    CStr::from_bytes_until_nul(p)
        .map(|c| c.to_string_lossy().into_owned())
        .unwrap_or_default()
}

impl VkDevice {
    /// Open `info` with every feature its `caps` advertise.
    pub fn open(
        instance: &'static ash::Instance,
        _api: u32,
        info: &DeviceInfo,
    ) -> Result<VkDevice, String> {
        let prio = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(info.queue_family)
            .queue_priorities(&prio)];
        let mut storage16 = vk::PhysicalDevice16BitStorageFeatures::default()
            .storage_buffer16_bit_access(info.caps.storage16)
            .uniform_and_storage_buffer16_bit_access(info.caps.storage16);
        let mut storage8 = vk::PhysicalDevice8BitStorageFeatures::default()
            .storage_buffer8_bit_access(info.caps.storage8)
            .uniform_and_storage_buffer8_bit_access(info.caps.storage8);
        let mut f16i8 = vk::PhysicalDeviceShaderFloat16Int8Features::default()
            .shader_float16(info.caps.f16)
            .shader_int8(info.caps.int8);
        let features = vk::PhysicalDeviceFeatures::default()
            .shader_int64(info.caps.int64)
            .shader_float64(info.caps.f64)
            .robust_buffer_access(robust_access())
            .shader_int16(info.caps.int16);
        let mut create = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_features(&features);
        let mut coop = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default()
            .cooperative_matrix(info.coop.is_some());
        let ext_ptrs: Vec<*const std::ffi::c_char> = if info.coop.is_some() {
            vec![COOPMAT_EXT.as_ptr()]
        } else {
            Vec::new()
        };
        if info.api >= vk::API_VERSION_1_1 {
            create = create
                .push_next(&mut storage16)
                .push_next(&mut storage8)
                .push_next(&mut f16i8)
                .enabled_extension_names(&ext_ptrs);
            if info.coop.is_some() {
                create = create.push_next(&mut coop);
            }
        }
        let device = unsafe { instance.create_device(info.pd, &create, None) }
            .map_err(|e| format!("vkCreateDevice: {e}"))?;
        crate::runtime::ops_vk::trace_public("device created");
        let queue = unsafe { device.get_device_queue(info.queue_family, 0) };
        crate::runtime::ops_vk::trace_public("queue");
        let cmd_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(info.queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .map_err(|e| format!("vkCreateCommandPool: {e}"))?;
        crate::runtime::ops_vk::trace_public("cmd pool");
        let cmd_v = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(4),
            )
        }
        .map_err(|e| format!("vkAllocateCommandBuffers: {e}"))?;
        let cmds: [vk::CommandBuffer; 4] = cmd_v
            .try_into()
            .map_err(|_| "vkAllocateCommandBuffers: expected 4 buffers")?;
        crate::runtime::ops_vk::trace_public("cmd buffer");
        let desc_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    // `SETS_PER_PROGRAM` sets per cached program (programs live
                    // for the process); generous so a many-kernel model never
                    // exhausts it and falls back to one set per program (which
                    // serializes batches, see `SETS_PER_PROGRAM`).
                    .max_sets(32768)
                    .pool_sizes(&[vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(131072)])
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET),
                None,
            )
        }
        .map_err(|e| format!("vkCreateDescriptorPool: {e}"))?;
        // Timestamp queries: exact per-dispatch GPU time for `VK_TIME=1`. The
        // queue must support timestamps at all (`timestampValidBits`).
        let ts_pool = if info.props.limits.timestamp_compute_and_graphics != 0 {
            unsafe {
                device.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count((2 * BATCH * 4) as u32),
                    None,
                )
            }
            .unwrap_or(vk::QueryPool::null())
        } else {
            vk::QueryPool::null()
        };
        crate::runtime::ops_vk::trace_public("desc pool");
        // Created signalled so the first wait on each slot returns immediately.
        let mut fences = [vk::Fence::null(); 4];
        for f in fences.iter_mut() {
            *f = unsafe {
                device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )
            }
            .map_err(|e| format!("vkCreateFence: {e}"))?;
        }
        crate::runtime::ops_vk::trace_public("fence");
        Ok(VkDevice {
            instance,
            device,
            pd: info.pd,
            queue,
            queue_family: info.queue_family,
            api: info.api,
            props: info.props,
            caps: info.caps,
            cmd_pool,
            cmds,
            desc_pool,
            fences,
            slot: std::cell::Cell::new(0),
            in_flight: std::cell::Cell::new(false),
            staging: std::cell::RefCell::new(None),
            graveyard: std::cell::RefCell::new(Vec::new()),
            ts_pool,
            ts_period: info.props.limits.timestamp_period,
            recording: std::cell::Cell::new(false),
            batch: std::cell::Cell::new(0),
            batch_names: std::cell::RefCell::new(vec![Vec::new(); 4]),
            next_id: std::cell::Cell::new(1),
        })
    }

    /// Wait for everything submitted so far (flushing the open batch first).
    pub fn wait_idle(&self) {
        let _ = self.flush();
        if self.in_flight.get() {
            unsafe {
                let _ = self.device.device_wait_idle();
            }
            self.in_flight.set(false);
            for slot in 0..self.cmds.len() {
                self.report_batch(slot);
            }
            self.drain();
        }
    }

    /// Print the GPU time of each dispatch in `slot`'s last batch (timestamp
    /// query readback: exact device time, no host attribution).
    fn report_batch(&self, slot: usize) {
        if !crate::runtime::ops_vk::time_on() || self.ts_pool == vk::QueryPool::null() {
            return;
        }
        let names: Vec<String> = {
            let mut n = self.batch_names.borrow_mut();
            std::mem::take(&mut n[slot])
        };
        if names.is_empty() {
            return;
        }
        let mut data = vec![0u64; 2 * names.len()];
        let ok = unsafe {
            self.device.get_query_pool_results(
                self.ts_pool,
                (2 * BATCH * slot) as u32,
                &mut data,
                vk::QueryResultFlags::TYPE_64,
            )
        };
        if ok.is_err() {
            return;
        }
        for (i, name) in names.iter().enumerate() {
            let ticks = data[2 * i + 1].saturating_sub(data[2 * i]);
            let ms = ticks as f64 * self.ts_period as f64 * 1e-6;
            eprintln!("[vk] gpu {name} {ms:.3} ms");
        }
    }

    /// Pick a memory type whose flags are a superset of `want`.
    fn memory_type(&self, bits: u32, want: vk::MemoryPropertyFlags) -> Option<u32> {
        let mp = unsafe { self.instance.get_physical_device_memory_properties(self.pd) };
        (0..mp.memory_type_count).find(|&i| {
            bits & (1 << i) != 0 && mp.memory_types[i as usize].property_flags.contains(want)
        })
    }

    fn memory_flags(&self, ty: u32) -> vk::MemoryPropertyFlags {
        let mp = unsafe { self.instance.get_physical_device_memory_properties(self.pd) };
        mp.memory_types[ty as usize].property_flags
    }

    /// Allocate a storage buffer.
    ///
    /// Small buffers prefer `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` (the
    /// BAR window): they are mapped, so upload/download is a `memcpy`. Larger
    /// ones use plain `DEVICE_LOCAL` (VRAM) and are copied through a staging
    /// buffer — host-visible memory beyond the BAR window is PCIe-resident, and
    /// reading it from a shader costs an order of magnitude.
    pub fn alloc(&self, size: usize) -> Option<GpuBuffer> {
        let size = size.max(4);
        let small = size <= (1 << 20);
        let buf = unsafe {
            self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size as u64)
                    .usage(
                        vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::TRANSFER_SRC
                            | vk::BufferUsageFlags::TRANSFER_DST,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }
        .ok()?;
        let req = unsafe { self.device.get_buffer_memory_requirements(buf) };
        use vk::MemoryPropertyFlags as F;
        let local = F::DEVICE_LOCAL;
        let host = F::HOST_VISIBLE | F::HOST_COHERENT;
        let ty = if small {
            self.memory_type(req.memory_type_bits, local | host)
                .or_else(|| self.memory_type(req.memory_type_bits, local))
                .or_else(|| self.memory_type(req.memory_type_bits, host))
        } else {
            self.memory_type(req.memory_type_bits, local)
                .or_else(|| self.memory_type(req.memory_type_bits, host))
        }?;
        let mem = unsafe {
            self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(ty),
                None,
            )
        }
        .ok()?;
        unsafe {
            if self.device.bind_buffer_memory(buf, mem, 0).is_err() {
                self.device.destroy_buffer(buf, None);
                self.device.free_memory(mem, None);
                return None;
            }
        }
        let flags = self.memory_flags(ty);
        let host_visible = flags.contains(F::HOST_VISIBLE);
        if crate::runtime::ops_vk::trace_on() {
            eprintln!(
                "[vk] alloc {size} bytes -> memory type {ty} ({}{})",
                if flags.contains(F::DEVICE_LOCAL) { "device-local" } else { "host" },
                if host_visible { ", mapped" } else { "" }
            );
        }
        let ptr = if host_visible {
            Some(unsafe {
                self.device
                    .map_memory(mem, 0, req.size, vk::MemoryMapFlags::empty())
            }
            .ok()? as *mut u8)
        } else {
            None
        };
        // Vulkan leaves new memory undefined; the other backends' allocations are
        // zeroed and some kernels rely on a zero fill, so match that.
        if let Some(p) = ptr {
            unsafe { std::ptr::write_bytes(p, 0, size) };
        }
        Some(GpuBuffer {
            buf,
            mem,
            ptr,
            size,
        })
    }

    /// Release a buffer's memory at the next idle point.
    ///
    /// The queue is ordered, so a buffer freed after its last dispatch is safe to
    /// *reuse* immediately; destroying it while the GPU still reads it is not
    /// (and loses the device). Deferring until nothing is in flight keeps the
    /// common "free intermediate, allocate the next one" loop valid.
    pub fn free(&self, b: &GpuBuffer) {
        self.graveyard.borrow_mut().push(b.clone());
        // Bound the graveyard for long GPU-only loops that never read back.
        if !self.in_flight.get() || self.graveyard.borrow().len() > 512 {
            self.wait_idle();
            self.drain();
        }
    }

    /// Destroy deferred buffers. Only call when no work is in flight.
    pub fn drain(&self) {
        let dead: Vec<GpuBuffer> = self.graveyard.borrow_mut().drain(..).collect();
        for b in dead {
            unsafe {
                if b.ptr.is_some() {
                    self.device.unmap_memory(b.mem);
                }
                self.device.destroy_buffer(b.buf, None);
                self.device.free_memory(b.mem, None);
            }
        }
    }

    /// Build a compute pipeline for `spv` with `bindings` storage buffers.
    pub fn pipeline(&self, spv: &[u32], bindings: u32) -> Result<GpuPipeline, String> {
        let shader = unsafe {
            self.device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(spv),
                None,
            )
        }
        .map_err(|e| format!("vkCreateShaderModule: {e}"))?;
        let bindings_v: Vec<vk::DescriptorSetLayoutBinding> = (0..bindings)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let set_layout = match unsafe {
            self.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings_v),
                None,
            )
        } {
            Ok(s) => s,
            Err(e) => {
                unsafe { self.device.destroy_shader_module(shader, None) };
                return Err(format!("vkCreateDescriptorSetLayout: {e}"));
            }
        };
        let layout = match unsafe {
            self.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]),
                None,
            )
        } {
            Ok(l) => l,
            Err(e) => {
                unsafe {
                    self.device
                        .destroy_descriptor_set_layout(set_layout, None);
                    self.device.destroy_shader_module(shader, None);
                }
                return Err(format!("vkCreatePipelineLayout: {e}"));
            }
        };
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(c"main");
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);
        let pipeline = match unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
        } {
            Ok(mut v) => v.remove(0),
            Err((_, e)) => {
                unsafe {
                    self.device.destroy_pipeline_layout(layout, None);
                    self.device
                        .destroy_descriptor_set_layout(set_layout, None);
                    self.device.destroy_shader_module(shader, None);
                }
                return Err(format!("vkCreateComputePipelines: {e}"));
            }
        };
        Ok(GpuPipeline {
            shader,
            set_layout,
            layout,
            pipeline,
            bindings,
        })
    }

    /// Destroy a pipeline built by [`VkDevice::pipeline`].
    pub fn destroy_pipeline(&self, p: &GpuPipeline) {
        unsafe {
            self.device.destroy_pipeline(p.pipeline, None);
            self.device.destroy_pipeline_layout(p.layout, None);
            self.device.destroy_descriptor_set_layout(p.set_layout, None);
            self.device.destroy_shader_module(p.shader, None);
        }
    }

    /// A host-visible staging buffer of at least `size` bytes (grown, cached).
    ///
    /// Transfers to/from device-local memory go through this; it is reused so a
    /// training loop does not allocate one per upload/readback.
    pub fn staging(&self, size: usize) -> std::cell::RefMut<'_, Option<GpuBuffer>> {
        let mut slot = self.staging.borrow_mut();
        let need = size
            .max(1 << 16)
            .next_power_of_two()
            .min(1 << 30);
        let grow = match slot.as_ref() {
            Some(b) => b.size < need,
            None => true,
        };
        if grow {
            if let Some(old) = slot.take() {
                self.free(&old);
            }
            *slot = Some(self.alloc_raw_host(need));
        }
        slot
    }

    /// A plain host-visible (mapped) buffer; used for staging and the raw-module
    /// diagnostic runner (which reads/writes host-side test data).
    pub fn alloc_raw_host(&self, size: usize) -> GpuBuffer {
        let size = size.max(4);
        let buf = unsafe {
            self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size as u64)
                    .usage(
                        vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::TRANSFER_SRC
                            | vk::BufferUsageFlags::TRANSFER_DST,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }
        .expect("TGERR: vk staging buffer");
        let req = unsafe { self.device.get_buffer_memory_requirements(buf) };
        use vk::MemoryPropertyFlags as F;
        // Plain system memory: the driver's DMA path to it is much faster than
        // reading back from the (small) BAR window.
        let ty = self
            .memory_type(
                req.memory_type_bits,
                F::HOST_VISIBLE | F::HOST_COHERENT,
            )
            .expect("TGERR: VK: no host-visible memory type for staging");
        let mem = unsafe {
            self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(ty),
                None,
            )
        }
        .expect("TGERR: vk staging memory");
        unsafe { self.device.bind_buffer_memory(buf, mem, 0) }.expect("TGERR: vk staging bind");
        let ptr = unsafe {
            self.device
                .map_memory(mem, 0, req.size, vk::MemoryMapFlags::empty())
        }
        .expect("TGERR: vk staging map") as *mut u8;
        GpuBuffer {
            buf,
            mem,
            ptr: Some(ptr),
            size,
        }
    }

    /// Record a buffer-to-buffer copy `src[src_off ..+ size]` →
    /// `dst[dst_off ..+ size]` into the open batch. `wait` flushes and blocks
    /// (needed before the host touches the staging buffer); a device-to-device
    /// copy is ordered by the queue and needs no host round-trip.
    pub fn copy_buffer(
        &self,
        src: &GpuBuffer,
        src_off: u64,
        dst: &GpuBuffer,
        dst_off: u64,
        size: u64,
        wait: bool,
    ) {
        if wait {
            let _ = self.flush();
        }
        if !self.recording.get() {
            let _ = self.begin_batch();
        }
        let region = [vk::BufferCopy::default()
            .src_offset(src_off)
            .dst_offset(dst_off)
            .size(size)];
        let cmd = self.cmds[self.slot.get()];
        unsafe { self.device.cmd_copy_buffer(cmd, src.buf, dst.buf, &region) };
        self.batch.set(self.batch.get() + 1);
        if wait {
            let _ = self.flush();
            self.wait_idle();
        }
    }

    /// The descriptor sets one program needs (one per ring slot).
    pub fn alloc_sets(&self, layout: vk::DescriptorSetLayout, n: u32) -> Vec<vk::DescriptorSet> {
        for n in [n, 1] {
            let layouts = vec![layout; n as usize];
            if let Ok(v) = unsafe {
                self.device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.desc_pool)
                        .set_layouts(&layouts),
                )
            } {
                return v;
            }
        }
        // A single set always forces a flush per dispatch, so batching degrades
        // instead of failing if the pool is ever exhausted.
        panic!("TGERR: vkAllocateDescriptorSets: pool exhausted")
    }

    /// Copy host bytes into `b`: a `memcpy` for a mapped buffer, a staged
    /// transfer for device-local memory.
    pub fn upload(&self, b: &GpuBuffer, src: &[u8]) {
        match b.ptr {
            Some(p) => unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), p, src.len())
            },
            None => {
                let mut st = self.staging(src.len());
                let stage = st.as_mut().expect("staging");
                unsafe {
                    std::ptr::copy_nonoverlapping(src.as_ptr(), stage.ptr.unwrap(), src.len())
                };
                self.copy_buffer(stage, 0, b, 0, src.len() as u64, true);
            }
        }
    }

    /// Copy `b` out to the host (`memcpy` when mapped, staged otherwise).
    pub fn download(&self, b: &GpuBuffer, dst: &mut [u8]) {
        match b.ptr {
            Some(p) => {
                self.wait_idle();
                unsafe { std::ptr::copy_nonoverlapping(p, dst.as_mut_ptr(), dst.len()) };
            }
            None => {
                let mut st = self.staging(dst.len());
                let stage = st.as_mut().expect("staging");
                self.copy_buffer(b, 0, stage, 0, dst.len() as u64, true);
                unsafe {
                    std::ptr::copy_nonoverlapping(stage.ptr.unwrap(), dst.as_mut_ptr(), dst.len())
                };
            }
        }
    }

    /// Open the current ring slot's command buffer for recording.
    fn begin_batch(&self) -> Result<(), String> {
        let dev = &self.device;
        let slot = self.slot.get();
        let fence = self.fences[slot];
        let cmd = self.cmds[slot];
        unsafe {
            dev.wait_for_fences(&[fence], true, u64::MAX)
                .map_err(|e| format!("vkWaitForFences: {e}"))?;
            dev.reset_fences(&[fence])
                .map_err(|e| format!("vkResetFences: {e}"))?;
        }
        // That slot's previous batch has completed: its timestamps are ready.
        self.report_batch(slot);
        unsafe {
            dev.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .map_err(|e| format!("vkResetCommandBuffer: {e}"))?;
            dev.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(|e| format!("vkBeginCommandBuffer: {e}"))?;
            if crate::runtime::ops_vk::time_on() && self.ts_pool != vk::QueryPool::null() {
                dev.cmd_reset_query_pool(
                    cmd,
                    self.ts_pool,
                    (2 * BATCH * slot) as u32,
                    (2 * BATCH) as u32,
                );
            }
        }
        self.recording.set(true);
        self.batch.set(0);
        self.batch_names.borrow_mut()[slot].clear();
        Ok(())
    }

    /// Close and submit the open batch (if any).
    pub fn flush(&self) -> Result<(), String> {
        if !self.recording.get() {
            return Ok(());
        }
        let dev = &self.device;
        let slot = self.slot.get();
        let cmd = self.cmds[slot];
        unsafe {
            dev.end_command_buffer(cmd)
                .map_err(|e| format!("vkEndCommandBuffer: {e}"))?;
            let cmds = [cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            dev.queue_submit(self.queue, &[submit], self.fences[slot])
                .map_err(|e| format!("vkQueueSubmit: {e}"))?;
        }
        self.slot.set((slot + 1) % self.cmds.len());
        self.recording.set(false);
        self.in_flight.set(true);
        Ok(())
    }

    /// Run one dispatch of `pipeline` over `buffers` (binding order) on a
    /// `groups` workgroup grid, recorded into the open command buffer.
    ///
    /// `sets[i]` is the descriptor set for the i-th dispatch of a batch; the
    /// caller provides one per batch position because a descriptor set may only
    /// be rewritten once the batch referencing it has completed.
    pub fn dispatch(
        &self,
        pipeline: &GpuPipeline,
        sets: &[vk::DescriptorSet],
        buffers: &[(&GpuBuffer, u64, u64)],
        groups: [u32; 3],
        wait: bool,
        name: &str,
    ) -> Result<(), String> {
        let slot0 = self.slot.get();
        let idx0 = self.batch.get();
        let open = self.recording.get()
            && idx0 < BATCH
            && slot0 * BATCH + idx0 < sets.len();
        if !open {
            self.flush()?;
        }
        let slot = self.slot.get();
        let idx = if open { idx0 } else { 0 };
        let set = sets[slot * BATCH + idx];
        // Descriptors are updated *before* the command buffer is reopened, which
        // is the order that keeps the driver from serializing.
        let infos: Vec<vk::DescriptorBufferInfo> = buffers
            .iter()
            .map(|(b, off, size)| {
                vk::DescriptorBufferInfo::default()
                    .buffer(b.buf)
                    .offset(*off)
                    .range(*size)
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = infos
            .iter()
            .enumerate()
            .map(|(i, bi)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(bi))
            })
            .collect();
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
        if !open {
            self.begin_batch()?;
        }
        let cmd = self.cmds[slot];
        let timed = crate::runtime::ops_vk::time_on() && self.ts_pool != vk::QueryPool::null();
        let dev = &self.device;
        unsafe {
            if timed {
                dev.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    self.ts_pool,
                    (2 * (BATCH * slot + idx)) as u32,
                );
            }
            // Make the previous dispatch's storage-buffer writes visible. The
            // first dispatch of a batch also has to cover the previous batch and
            // any recorded copies, so it uses ALL_COMMANDS (free: nothing else is
            // recorded yet); later ones only need the compute->compute edge,
            // which avoids flushing every cache mid-batch.
            let (src_stage, src_access) = if idx == 0 {
                (vk::PipelineStageFlags::ALL_COMMANDS, vk::AccessFlags::MEMORY_WRITE)
            } else {
                (vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::SHADER_WRITE)
            };
            let barriers = [vk::MemoryBarrier::default()
                .src_access_mask(src_access)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
            dev.cmd_pipeline_barrier(
                cmd,
                src_stage,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &barriers,
                &[],
                &[],
            );
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
            dev.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout,
                0,
                &[set],
                &[],
            );
            dev.cmd_dispatch(cmd, groups[0].max(1), groups[1].max(1), groups[2].max(1));
            if timed {
                dev.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.ts_pool,
                    (2 * (BATCH * slot + idx) + 1) as u32,
                );
            }
        }
        self.batch_names.borrow_mut()[slot].push(name.to_string());
        self.batch.set(idx + 1);
        if wait {
            self.flush()?;
            self.wait_idle();
        }
        Ok(())
    }

    /// Total device-local memory in bytes (the `MemoryInfo` total).
    pub fn device_local_memory(&self) -> usize {
        let mp = unsafe { self.instance.get_physical_device_memory_properties(self.pd) };
        (0..mp.memory_heap_count)
            .filter(|&i| {
                mp.memory_heaps[i as usize]
                    .flags
                    .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
            })
            .map(|i| mp.memory_heaps[i as usize].size as usize)
            .max()
            .unwrap_or(0)
    }

    /// `driverVersion`/`apiVersion`/`deviceName`, for the startup log.
    pub fn describe(&self) -> String {
        format!(
            "{} (api {}.{}.{}, driver {}.{}.{})",
            cstr(&self.props.device_name),
            vk::api_version_major(self.props.api_version),
            vk::api_version_minor(self.props.api_version),
            vk::api_version_patch(self.props.api_version),
            (self.props.driver_version >> 22) & 0x3ff,
            (self.props.driver_version >> 12) & 0x3ff,
            self.props.driver_version & 0xfff,
        )
    }
}

/// `robustBufferAccess` (on unless `VK_NO_ROBUST` asks otherwise).
fn robust_access() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::helpers::env::get("VK_NO_ROBUST").is_none())
}

/// Whether the loader and a usable compute device exist.
pub fn loader_present() -> bool {
    VkLib::get().is_some()
}

/// A tiny valid SPIR-V module (empty compute kernel) used to prove that a
/// device can compile and create a pipeline. This catches drivers that
/// enumerate but cannot execute anything (a broken ICD, a headless VM, ...).
pub const PROBE_SPV: &[u32] = &[
    0x0723_0203, 0x0001_0000, 0, 17, 0, // header: magic, version, gen, bound, schema
    (2 << 16) | 17, 1,           // OpCapability Shader
    (3 << 16) | 14, 0, 1,        // OpMemoryModel Logical GLSL450
    (5 << 16) | 15, 5, 2, 0x6e69_616d, 0x0000_0000, // OpEntryPoint GLCompute %2 "main"
    (6 << 16) | 16, 2, 17, 1, 1, 1, // OpExecutionMode %2 LocalSize 1 1 1
    (2 << 16) | 19, 3,           // OpTypeVoid %3
    (3 << 16) | 33, 4, 3,        // OpTypeFunction %4 -> %3
    (5 << 16) | 54, 3, 2, 0, 4,  // OpFunction %3 %2 None %4
    (2 << 16) | 248, 5,          // OpLabel %5
    (1 << 16) | 253,             // OpReturn
    (1 << 16) | 56,              // OpFunctionEnd
];

/// Check that `info` can open a device and create a compute pipeline.
///
/// Returns the reason on failure, so `VK_TRACE=1` can explain why a device was
/// skipped instead of silently omitting it.
pub fn device_probe(
    instance: &'static ash::Instance,
    api: u32,
    info: &DeviceInfo,
) -> Result<(), String> {
    let dev = VkDevice::open(instance, api, info)?;
    let out = dev.pipeline(PROBE_SPV, 0).map(|p| dev.destroy_pipeline(&p));
    // The probe must not leak a logical device: only one instance of the real
    // device is ever opened per process.
    unsafe { dev.device.destroy_device(None) };
    out
}

/// Whether `info` passes [`device_probe`].
pub fn device_usable(instance: &'static ash::Instance, api: u32, info: &DeviceInfo) -> bool {
    device_probe(instance, api, info).is_ok()
}
