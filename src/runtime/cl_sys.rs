//! Dynamic OpenCL FFI for the `CL` backend.
//!
//! `opencl3`/`cl3`/`opencl-sys` hard-link `libOpenCL`, so a binary built with
//! `backend_cl` cannot start on a machine without the ICD loader. This module
//! loads the library at run time with `libloading`, exposes only the ~30
//! `cl_*` entry points the backend uses, and wraps the raw handles in RAII
//! types. `ops_cl` sees a small safe API; all FFI is contained here.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::{CString, c_char, c_void};
use std::sync::OnceLock;

pub type cl_int = i32;
pub type cl_uint = u32;
pub type cl_ulong = u64;
pub type cl_bitfield = u64;
pub type cl_bool = u32;
pub type cl_device_type = u64;
pub type cl_mem_flags = u64;
pub type cl_command_queue_properties = u64;
pub type cl_platform_id = *mut c_void;
pub type cl_device_id = *mut c_void;
pub type cl_context = *mut c_void;
pub type cl_command_queue = *mut c_void;
pub type cl_program = *mut c_void;
pub type cl_kernel = *mut c_void;
pub type cl_mem = *mut c_void;
pub type cl_event = *mut c_void;
pub type cl_context_properties = isize;
pub type cl_device_info = u32;
pub type cl_program_info = u32;
pub type cl_program_build_info = u32;
pub type cl_profiling_info = u32;

pub const CL_SUCCESS: cl_int = 0;
pub const CL_TRUE: cl_bool = 1;
pub const CL_DEVICE_TYPE_DEFAULT: cl_device_type = 1 << 0;
pub const CL_DEVICE_TYPE_GPU: cl_device_type = 1 << 2;
pub const CL_MEM_READ_WRITE: cl_mem_flags = 1 << 0;
pub const CL_QUEUE_PROFILING_ENABLE: cl_command_queue_properties = 1 << 1;
pub const CL_DEVICE_NAME: cl_device_info = 0x102B;
pub const CL_DRIVER_VERSION: cl_device_info = 0x102D;
pub const CL_DEVICE_EXTENSIONS: cl_device_info = 0x1030;
pub const CL_PROGRAM_BUILD_LOG: cl_program_build_info = 0x1183;
pub const CL_PROGRAM_BINARY_SIZES: cl_program_info = 0x1165;
pub const CL_PROGRAM_BINARIES: cl_program_info = 0x1166;
pub const CL_PROFILING_COMMAND_START: cl_profiling_info = 0x1282;
pub const CL_PROFILING_COMMAND_END: cl_profiling_info = 0x1283;

/// An OpenCL error code.
#[derive(Clone, Copy, Debug)]
pub struct ClError(pub cl_int);

impl std::fmt::Display for ClError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OpenCL error {}", self.0)
    }
}
impl std::error::Error for ClError {}

type ClResult<T> = Result<T, ClError>;

fn check(code: cl_int) -> ClResult<()> {
    if code == CL_SUCCESS {
        Ok(())
    } else {
        Err(ClError(code))
    }
}

/// Validate a freshly created handle. If the driver returned both an error and
/// a non-null handle (not required by the spec, but cheap to guard), release it
/// instead of leaking it.
fn check_handle(
    raw: *mut c_void,
    err: cl_int,
    release: unsafe extern "C" fn(*mut c_void) -> cl_int,
) -> ClResult<*mut c_void> {
    if err != CL_SUCCESS {
        if !raw.is_null() {
            unsafe {
                let _ = release(raw);
            }
        }
        return Err(ClError(err));
    }
    if raw.is_null() {
        return Err(ClError(err));
    }
    Ok(raw)
}

fn noptr<T>() -> *const T {
    std::ptr::null()
}
fn mutptr<T>() -> *mut T {
    std::ptr::null_mut()
}

type FnGetPlatformIDs = unsafe extern "C" fn(cl_uint, *mut cl_platform_id, *mut cl_uint) -> cl_int;
type FnGetDeviceIDs = unsafe extern "C" fn(
    cl_platform_id,
    cl_device_type,
    cl_uint,
    *mut cl_device_id,
    *mut cl_uint,
) -> cl_int;
type FnGetDeviceInfo =
    unsafe extern "C" fn(cl_device_id, cl_device_info, usize, *mut c_void, *mut usize) -> cl_int;
type FnCreateContext = unsafe extern "C" fn(
    *const cl_context_properties,
    cl_uint,
    *const cl_device_id,
    *const c_void,
    *mut c_void,
    *mut cl_int,
) -> cl_context;
type FnReleaseContext = unsafe extern "C" fn(cl_context) -> cl_int;
type FnCreateCommandQueue = unsafe extern "C" fn(
    cl_context,
    cl_device_id,
    cl_command_queue_properties,
    *mut cl_int,
) -> cl_command_queue;
type FnReleaseCommandQueue = unsafe extern "C" fn(cl_command_queue) -> cl_int;
type FnFinish = unsafe extern "C" fn(cl_command_queue) -> cl_int;
type FnCreateProgramWithSource = unsafe extern "C" fn(
    cl_context,
    cl_uint,
    *const *const c_char,
    *const usize,
    *mut cl_int,
) -> cl_program;
type FnCreateProgramWithBinary = unsafe extern "C" fn(
    cl_context,
    cl_uint,
    *const cl_device_id,
    *const usize,
    *const *const u8,
    *mut cl_int,
    *mut cl_int,
) -> cl_program;
type FnBuildProgram = unsafe extern "C" fn(
    cl_program,
    cl_uint,
    *const cl_device_id,
    *const c_char,
    *const c_void,
    *mut c_void,
) -> cl_int;
type FnGetProgramBuildInfo = unsafe extern "C" fn(
    cl_program,
    cl_device_id,
    cl_program_build_info,
    usize,
    *mut c_void,
    *mut usize,
) -> cl_int;
type FnGetProgramInfo =
    unsafe extern "C" fn(cl_program, cl_program_info, usize, *mut c_void, *mut usize) -> cl_int;
type FnReleaseProgram = unsafe extern "C" fn(cl_program) -> cl_int;
type FnCreateKernel = unsafe extern "C" fn(cl_program, *const c_char, *mut cl_int) -> cl_kernel;
type FnReleaseKernel = unsafe extern "C" fn(cl_kernel) -> cl_int;
type FnSetKernelArg = unsafe extern "C" fn(cl_kernel, cl_uint, usize, *const c_void) -> cl_int;
type FnEnqueueNDRangeKernel = unsafe extern "C" fn(
    cl_command_queue,
    cl_kernel,
    cl_uint,
    *const usize,
    *const usize,
    *const usize,
    cl_uint,
    *const cl_event,
    *mut cl_event,
) -> cl_int;
type FnCreateBuffer =
    unsafe extern "C" fn(cl_context, cl_mem_flags, usize, *mut c_void, *mut cl_int) -> cl_mem;
type FnReleaseMemObject = unsafe extern "C" fn(cl_mem) -> cl_int;
type FnEnqueueWriteBuffer = unsafe extern "C" fn(
    cl_command_queue,
    cl_mem,
    cl_bool,
    usize,
    usize,
    *const c_void,
    cl_uint,
    *const cl_event,
    *mut cl_event,
) -> cl_int;
type FnEnqueueReadBuffer = unsafe extern "C" fn(
    cl_command_queue,
    cl_mem,
    cl_bool,
    usize,
    usize,
    *mut c_void,
    cl_uint,
    *const cl_event,
    *mut cl_event,
) -> cl_int;
type FnEnqueueCopyBuffer = unsafe extern "C" fn(
    cl_command_queue,
    cl_mem,
    cl_mem,
    usize,
    usize,
    usize,
    cl_uint,
    *const cl_event,
    *mut cl_event,
) -> cl_int;
type FnWaitForEvents = unsafe extern "C" fn(cl_uint, *const cl_event) -> cl_int;
type FnGetEventProfilingInfo =
    unsafe extern "C" fn(cl_event, cl_profiling_info, usize, *mut c_void, *mut usize) -> cl_int;
type FnReleaseEvent = unsafe extern "C" fn(cl_event) -> cl_int;

/// The loaded OpenCL library and its entry points.
pub struct ClLib {
    _lib: libloading::Library,
    get_platform_ids: FnGetPlatformIDs,
    get_device_ids: FnGetDeviceIDs,
    get_device_info: FnGetDeviceInfo,
    create_context: FnCreateContext,
    release_context: FnReleaseContext,
    create_command_queue: FnCreateCommandQueue,
    release_command_queue: FnReleaseCommandQueue,
    finish: FnFinish,
    create_program_with_source: FnCreateProgramWithSource,
    create_program_with_binary: FnCreateProgramWithBinary,
    build_program: FnBuildProgram,
    get_program_build_info: FnGetProgramBuildInfo,
    get_program_info: FnGetProgramInfo,
    release_program: FnReleaseProgram,
    create_kernel: FnCreateKernel,
    release_kernel: FnReleaseKernel,
    set_kernel_arg: FnSetKernelArg,
    enqueue_nd_range_kernel: FnEnqueueNDRangeKernel,
    create_buffer: FnCreateBuffer,
    release_mem_object: FnReleaseMemObject,
    enqueue_write_buffer: FnEnqueueWriteBuffer,
    enqueue_read_buffer: FnEnqueueReadBuffer,
    enqueue_copy_buffer: FnEnqueueCopyBuffer,
    wait_for_events: FnWaitForEvents,
    get_event_profiling_info: FnGetEventProfilingInfo,
    release_event: FnReleaseEvent,
}

impl ClLib {
    /// Load the ICD loader once. Returns `None` (without panicking) if the
    /// library is absent or lacks an entry point.
    pub fn get() -> Option<&'static ClLib> {
        static CL: OnceLock<Option<ClLib>> = OnceLock::new();
        CL.get_or_init(ClLib::load).as_ref()
    }

    fn load() -> Option<ClLib> {
        #[cfg(target_os = "linux")]
        let default_names: &[&str] = &["libOpenCL.so.1", "libOpenCL.so"];
        #[cfg(target_os = "macos")]
        let default_names: &[&str] = &[
            "/System/Library/Frameworks/OpenCL.framework/OpenCL",
            "OpenCL",
        ];
        #[cfg(target_os = "windows")]
        let default_names: &[&str] = &["OpenCL.dll"];
        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        let default_names: &[&str] = &["libOpenCL.so"];
        // Test/deployment override: trying a single explicit path makes it
        // possible to verify the missing-library fallback without touching the
        // system loader configuration.
        let names: Vec<String> = match std::env::var("AUTORUST_CL_LIB") {
            Ok(n) => vec![n],
            Err(_) => default_names.iter().map(|s| s.to_string()).collect(),
        };

        for name in &names {
            let Ok(lib) = (unsafe { libloading::Library::new(name) }) else {
                continue;
            };
            // SAFETY: symbol names/types come from the OpenCL C API.
            if let Some(cl) = unsafe { ClLib::from_lib(lib) } {
                return Some(cl);
            }
        }
        None
    }

    unsafe fn from_lib(lib: libloading::Library) -> Option<ClLib> {
        macro_rules! sym {
            ($name:literal, $ty:ty) => {
                *unsafe { lib.get::<$ty>(concat!($name, "\0").as_bytes()).ok()? }
            };
        }
        Some(ClLib {
            get_platform_ids: sym!("clGetPlatformIDs", FnGetPlatformIDs),
            get_device_ids: sym!("clGetDeviceIDs", FnGetDeviceIDs),
            get_device_info: sym!("clGetDeviceInfo", FnGetDeviceInfo),
            create_context: sym!("clCreateContext", FnCreateContext),
            release_context: sym!("clReleaseContext", FnReleaseContext),
            create_command_queue: sym!("clCreateCommandQueue", FnCreateCommandQueue),
            release_command_queue: sym!("clReleaseCommandQueue", FnReleaseCommandQueue),
            finish: sym!("clFinish", FnFinish),
            create_program_with_source: sym!(
                "clCreateProgramWithSource",
                FnCreateProgramWithSource
            ),
            create_program_with_binary: sym!(
                "clCreateProgramWithBinary",
                FnCreateProgramWithBinary
            ),
            build_program: sym!("clBuildProgram", FnBuildProgram),
            get_program_build_info: sym!("clGetProgramBuildInfo", FnGetProgramBuildInfo),
            get_program_info: sym!("clGetProgramInfo", FnGetProgramInfo),
            release_program: sym!("clReleaseProgram", FnReleaseProgram),
            create_kernel: sym!("clCreateKernel", FnCreateKernel),
            release_kernel: sym!("clReleaseKernel", FnReleaseKernel),
            set_kernel_arg: sym!("clSetKernelArg", FnSetKernelArg),
            enqueue_nd_range_kernel: sym!("clEnqueueNDRangeKernel", FnEnqueueNDRangeKernel),
            create_buffer: sym!("clCreateBuffer", FnCreateBuffer),
            release_mem_object: sym!("clReleaseMemObject", FnReleaseMemObject),
            enqueue_write_buffer: sym!("clEnqueueWriteBuffer", FnEnqueueWriteBuffer),
            enqueue_read_buffer: sym!("clEnqueueReadBuffer", FnEnqueueReadBuffer),
            enqueue_copy_buffer: sym!("clEnqueueCopyBuffer", FnEnqueueCopyBuffer),
            wait_for_events: sym!("clWaitForEvents", FnWaitForEvents),
            get_event_profiling_info: sym!("clGetEventProfilingInfo", FnGetEventProfilingInfo),
            release_event: sym!("clReleaseEvent", FnReleaseEvent),
            _lib: lib,
        })
    }
}

/// A raw OpenCL device id (not refcounted).
#[derive(Clone, Copy)]
pub struct Device {
    id: cl_device_id,
}

impl Device {
    /// Wrap a device id.
    pub fn new(id: cl_device_id) -> Device {
        Device { id }
    }
    /// The raw `cl_device_id`.
    pub fn id(&self) -> cl_device_id {
        self.id
    }
    fn info_string(&self, param: cl_device_info, what: &str) -> ClResult<String> {
        let cl = ClLib::get().ok_or(ClError(-1))?;
        let mut size = 0usize;
        check(unsafe { (cl.get_device_info)(self.id, param, 0, mutptr(), &mut size) })?;
        if size == 0 {
            return Ok(String::new());
        }
        let mut buf = vec![0u8; size];
        check(unsafe {
            (cl.get_device_info)(
                self.id,
                param,
                size,
                buf.as_mut_ptr() as *mut c_void,
                &mut size,
            )
        })?;
        buf.truncate(size);
        let _ = what;
        Ok(String::from_utf8_lossy(&buf)
            .trim_end_matches('\0')
            .to_string())
    }
    /// `CL_DEVICE_NAME`.
    pub fn name(&self) -> ClResult<String> {
        self.info_string(CL_DEVICE_NAME, "name")
    }
    /// `CL_DRIVER_VERSION`.
    pub fn driver_version(&self) -> ClResult<String> {
        self.info_string(CL_DRIVER_VERSION, "driver")
    }
    /// `CL_DEVICE_EXTENSIONS`.
    pub fn extensions(&self) -> ClResult<String> {
        self.info_string(CL_DEVICE_EXTENSIONS, "extensions")
    }
}

/// Enumerate devices of `ty` across all platforms.
pub fn get_all_devices(ty: cl_device_type) -> ClResult<Vec<cl_device_id>> {
    let cl = ClLib::get().ok_or(ClError(-1))?;
    let mut nplat: cl_uint = 0;
    check(unsafe { (cl.get_platform_ids)(0, mutptr(), &mut nplat) })?;
    if nplat == 0 {
        return Ok(Vec::new());
    }
    let mut plats = vec![std::ptr::null_mut(); nplat as usize];
    check(unsafe { (cl.get_platform_ids)(nplat, plats.as_mut_ptr(), &mut nplat) })?;
    plats.truncate(nplat as usize);
    let mut out = Vec::new();
    for p in plats {
        let mut count: cl_uint = 0;
        let code = unsafe { (cl.get_device_ids)(p, ty, 0, mutptr(), &mut count) };
        if code != CL_SUCCESS || count == 0 {
            continue;
        }
        let mut ids = vec![std::ptr::null_mut(); count as usize];
        if check(unsafe { (cl.get_device_ids)(p, ty, count, ids.as_mut_ptr(), &mut count) }).is_ok()
        {
            ids.truncate(count as usize);
            out.extend(ids);
        }
    }
    Ok(out)
}

/// An OpenCL context (released on drop).
pub struct Context {
    raw: cl_context,
}

impl Context {
    /// `clCreateContext` for one device.
    pub fn from_device(dev: &Device) -> ClResult<Context> {
        let cl = ClLib::get().ok_or(ClError(-1))?;
        let devices = [dev.id()];
        let mut err: cl_int = 0;
        let raw = unsafe {
            (cl.create_context)(noptr(), 1, devices.as_ptr(), noptr(), mutptr(), &mut err)
        };
        Ok(Context {
            raw: check_handle(raw, err, cl.release_context)? as cl_context,
        })
    }
    /// The raw `cl_context`.
    pub fn raw(&self) -> cl_context {
        self.raw
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if let Some(cl) = ClLib::get() {
            unsafe {
                let _ = (cl.release_context)(self.raw);
            }
        }
    }
}

/// An OpenCL command queue (released on drop).
pub struct CommandQueue {
    raw: cl_command_queue,
}

impl CommandQueue {
    /// `clCreateCommandQueue` for one device.
    pub fn create(
        ctx: &Context,
        dev: cl_device_id,
        props: cl_command_queue_properties,
    ) -> ClResult<CommandQueue> {
        let cl = ClLib::get().ok_or(ClError(-1))?;
        let mut err: cl_int = 0;
        let raw = unsafe { (cl.create_command_queue)(ctx.raw(), dev, props, &mut err) };
        Ok(CommandQueue {
            raw: check_handle(raw, err, cl.release_command_queue)? as cl_command_queue,
        })
    }

    /// Raw `cl_command_queue` (for library calls such as CLBlast).
    pub fn raw(&self) -> cl_command_queue {
        self.raw
    }

    /// `clFinish`.
    pub fn finish(&self) -> ClResult<()> {
        let cl = ClLib::get().ok_or(ClError(-1))?;
        check(unsafe { (cl.finish)(self.raw) })
    }

    /// Blocking `clEnqueueWriteBuffer`.
    pub fn enqueue_write_buffer(
        &self,
        mem: cl_mem,
        blocking: bool,
        offset: usize,
        src: &[u8],
    ) -> ClResult<()> {
        if src.is_empty() {
            return Ok(());
        }
        let cl = ClLib::get().ok_or(ClError(-1))?;
        check(unsafe {
            (cl.enqueue_write_buffer)(
                self.raw,
                mem,
                if blocking { CL_TRUE } else { 0 },
                offset,
                src.len(),
                src.as_ptr() as *const c_void,
                0,
                noptr(),
                mutptr(),
            )
        })
    }

    /// Blocking `clEnqueueReadBuffer`.
    pub fn enqueue_read_buffer(
        &self,
        mem: cl_mem,
        blocking: bool,
        offset: usize,
        dst: &mut [u8],
    ) -> ClResult<()> {
        if dst.is_empty() {
            return Ok(());
        }
        let cl = ClLib::get().ok_or(ClError(-1))?;
        check(unsafe {
            (cl.enqueue_read_buffer)(
                self.raw,
                mem,
                if blocking { CL_TRUE } else { 0 },
                offset,
                dst.len(),
                dst.as_mut_ptr() as *mut c_void,
                0,
                noptr(),
                mutptr(),
            )
        })
    }

    /// `clEnqueueCopyBuffer`.
    pub fn enqueue_copy_buffer(
        &self,
        src: cl_mem,
        dst: cl_mem,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> ClResult<()> {
        if size == 0 {
            return Ok(());
        }
        let cl = ClLib::get().ok_or(ClError(-1))?;
        check(unsafe {
            (cl.enqueue_copy_buffer)(
                self.raw,
                src,
                dst,
                src_offset,
                dst_offset,
                size,
                0,
                noptr(),
                mutptr(),
            )
        })
    }

    /// `clEnqueueNDRangeKernel`; returns the event for wait/profiling.
    pub fn enqueue_nd_range_kernel(
        &self,
        kernel: cl_kernel,
        global: &[usize; 3],
        local: Option<&[usize; 3]>,
    ) -> ClResult<Event> {
        let cl = ClLib::get().ok_or(ClError(-1))?;
        let local_ptr = local.map(|l| l.as_ptr()).unwrap_or(std::ptr::null());
        let mut ev: cl_event = std::ptr::null_mut();
        let code = unsafe {
            (cl.enqueue_nd_range_kernel)(
                self.raw,
                kernel,
                3,
                noptr(),
                global.as_ptr(),
                local_ptr,
                0,
                noptr(),
                &mut ev,
            )
        };
        if let Err(e) = check(code) {
            if !ev.is_null() {
                unsafe {
                    let _ = (cl.release_event)(ev);
                }
            }
            return Err(e);
        }
        if ev.is_null() {
            return Err(ClError(-2));
        }
        Ok(Event { raw: ev })
    }
}

impl Drop for CommandQueue {
    fn drop(&mut self) {
        if let Some(cl) = ClLib::get() {
            unsafe {
                let _ = (cl.release_command_queue)(self.raw);
            }
        }
    }
}

/// An OpenCL memory object (released on drop).
pub struct Mem {
    raw: cl_mem,
}

impl Mem {
    /// `clCreateBuffer`.
    pub fn create(
        ctx: &Context,
        flags: cl_mem_flags,
        size: usize,
        host_ptr: *mut c_void,
    ) -> ClResult<Mem> {
        let cl = ClLib::get().ok_or(ClError(-1))?;
        let mut err: cl_int = 0;
        let raw = unsafe { (cl.create_buffer)(ctx.raw(), flags, size, host_ptr, &mut err) };
        Ok(Mem {
            raw: check_handle(raw, err, cl.release_mem_object)? as cl_mem,
        })
    }
    /// The raw `cl_mem`.
    pub fn get(&self) -> cl_mem {
        self.raw
    }
}

impl Drop for Mem {
    fn drop(&mut self) {
        if let Some(cl) = ClLib::get() {
            unsafe {
                let _ = (cl.release_mem_object)(self.raw);
            }
        }
    }
}

/// An OpenCL program (released on drop).
pub struct Program {
    raw: cl_program,
}

impl Program {
    /// `clCreateProgramWithSource`.
    pub fn create_from_source(ctx: &Context, src: &str) -> ClResult<Program> {
        let cl = ClLib::get().ok_or(ClError(-1))?;
        let c = CString::new(src).map_err(|_| ClError(-1))?;
        let strings = [c.as_ptr()];
        let lengths = [src.len()];
        let mut err: cl_int = 0;
        let raw = unsafe {
            (cl.create_program_with_source)(
                ctx.raw(),
                1,
                strings.as_ptr(),
                lengths.as_ptr(),
                &mut err,
            )
        };
        Ok(Program {
            raw: check_handle(raw, err, cl.release_program)? as cl_program,
        })
    }

    /// `clCreateProgramWithBinary`.
    pub fn create_from_binary(
        ctx: &Context,
        devices: &[cl_device_id],
        binaries: &[&[u8]],
    ) -> ClResult<Program> {
        assert_eq!(devices.len(), binaries.len(), "TGERR: CL binary count");
        let cl = ClLib::get().ok_or(ClError(-1))?;
        let lengths: Vec<usize> = binaries.iter().map(|b| b.len()).collect();
        let ptrs: Vec<*const u8> = binaries.iter().map(|b| b.as_ptr()).collect();
        let mut err: cl_int = 0;
        let raw = unsafe {
            (cl.create_program_with_binary)(
                ctx.raw(),
                devices.len() as cl_uint,
                devices.as_ptr(),
                lengths.as_ptr(),
                ptrs.as_ptr(),
                mutptr(),
                &mut err,
            )
        };
        Ok(Program {
            raw: check_handle(raw, err, cl.release_program)? as cl_program,
        })
    }

    /// `clBuildProgram`.
    pub fn build(&self, devices: &[cl_device_id], options: &str) -> ClResult<()> {
        let cl = ClLib::get().ok_or(ClError(-1))?;
        let opts = CString::new(options).map_err(|_| ClError(-1))?;
        check(unsafe {
            (cl.build_program)(
                self.raw,
                devices.len() as cl_uint,
                devices.as_ptr(),
                opts.as_ptr(),
                noptr(),
                mutptr(),
            )
        })
    }

    /// `CL_PROGRAM_BUILD_LOG`.
    pub fn get_build_log(&self, dev: cl_device_id) -> ClResult<String> {
        let cl = ClLib::get().ok_or(ClError(-1))?;
        let mut size = 0usize;
        check(unsafe {
            (cl.get_program_build_info)(self.raw, dev, CL_PROGRAM_BUILD_LOG, 0, mutptr(), &mut size)
        })?;
        if size == 0 {
            return Ok(String::new());
        }
        let mut buf = vec![0u8; size];
        check(unsafe {
            (cl.get_program_build_info)(
                self.raw,
                dev,
                CL_PROGRAM_BUILD_LOG,
                size,
                buf.as_mut_ptr() as *mut c_void,
                &mut size,
            )
        })?;
        buf.truncate(size);
        Ok(String::from_utf8_lossy(&buf)
            .trim_end_matches('\0')
            .to_string())
    }

    /// `CL_PROGRAM_BINARIES` (one entry per device the program was built for).
    pub fn get_binaries(&self) -> ClResult<Vec<Vec<u8>>> {
        let cl = ClLib::get().ok_or(ClError(-1))?;
        let mut sizes = [0usize; 1];
        check(unsafe {
            (cl.get_program_info)(
                self.raw,
                CL_PROGRAM_BINARY_SIZES,
                std::mem::size_of::<usize>(),
                sizes.as_mut_ptr() as *mut c_void,
                mutptr(),
            )
        })?;
        if sizes[0] == 0 {
            return Ok(vec![Vec::new()]);
        }
        let mut bufs: Vec<Vec<u8>> = sizes.iter().map(|&s| vec![0u8; s]).collect();
        let mut ptrs: Vec<*mut u8> = bufs.iter_mut().map(|b| b.as_mut_ptr()).collect();
        check(unsafe {
            (cl.get_program_info)(
                self.raw,
                CL_PROGRAM_BINARIES,
                ptrs.len() * std::mem::size_of::<*mut u8>(),
                ptrs.as_mut_ptr() as *mut c_void,
                mutptr(),
            )
        })?;
        Ok(bufs)
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        if let Some(cl) = ClLib::get() {
            unsafe {
                let _ = (cl.release_program)(self.raw);
            }
        }
    }
}

/// An OpenCL kernel (released on drop).
pub struct Kernel {
    raw: cl_kernel,
}

impl Kernel {
    /// `clCreateKernel`.
    pub fn create(program: &Program, name: &str) -> ClResult<Kernel> {
        let cl = ClLib::get().ok_or(ClError(-1))?;
        let c = CString::new(name).map_err(|_| ClError(-1))?;
        let mut err: cl_int = 0;
        let raw = unsafe { (cl.create_kernel)(program.raw, c.as_ptr(), &mut err) };
        Ok(Kernel {
            raw: check_handle(raw, err, cl.release_kernel)? as cl_kernel,
        })
    }

    /// `clSetKernelArg` for a `Copy` value (buffer handles and scalars).
    pub fn set_arg<T>(&self, index: cl_uint, value: &T) -> ClResult<()> {
        let cl = ClLib::get().ok_or(ClError(-1))?;
        check(unsafe {
            (cl.set_kernel_arg)(
                self.raw,
                index,
                std::mem::size_of::<T>(),
                value as *const T as *const c_void,
            )
        })
    }

    /// The raw `cl_kernel`.
    pub fn get(&self) -> cl_kernel {
        self.raw
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        if let Some(cl) = ClLib::get() {
            unsafe {
                let _ = (cl.release_kernel)(self.raw);
            }
        }
    }
}

/// An OpenCL event (released on drop).
pub struct Event {
    raw: cl_event,
}

impl Event {
    /// `clWaitForEvents`.
    pub fn wait(&self) -> ClResult<()> {
        let cl = ClLib::get().ok_or(ClError(-1))?;
        check(unsafe { (cl.wait_for_events)(1, &self.raw) })
    }
    fn profiling(&self, param: cl_profiling_info) -> ClResult<u64> {
        let cl = ClLib::get().ok_or(ClError(-1))?;
        let mut v: cl_ulong = 0;
        check(unsafe {
            (cl.get_event_profiling_info)(
                self.raw,
                param,
                std::mem::size_of::<cl_ulong>(),
                &mut v as *mut cl_ulong as *mut c_void,
                mutptr(),
            )
        })?;
        Ok(v)
    }
    /// `CL_PROFILING_COMMAND_START` in ns.
    pub fn profiling_command_start(&self) -> ClResult<u64> {
        self.profiling(CL_PROFILING_COMMAND_START)
    }
    /// `CL_PROFILING_COMMAND_END` in ns.
    pub fn profiling_command_end(&self) -> ClResult<u64> {
        self.profiling(CL_PROFILING_COMMAND_END)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if let Some(cl) = ClLib::get() {
            unsafe {
                let _ = (cl.release_event)(self.raw);
            }
        }
    }
}
