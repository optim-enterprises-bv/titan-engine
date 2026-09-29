//! Pure-Rust twins of the group-2 extern "C" host launchers of mistralrs-paged-attn
//! (libmistralrspagedattention.a: flashinfer_decode.cu and flashinfer_mla_decode.cu): same names, same
//! parameter lists and C ABI as the C definitions (declared in mistralrs-paged-attn/src/cuda/ffi.rs), the
//! same host-side control flow (dtype / head-dim / sliding-window / soft-cap / GQA-group dispatch, the
//! compute-capability pipeline depth, `uint_fastdiv` construction, dynamic shared memory sizing, split-KV
//! partitioning and the VariableLengthMergeStates pass, error returns and messages), launching the oxide
//! kernels of this crate on the caller's stream through the CUDA driver API.
//!
//! - The reference objects are built with `--default-stream per-thread`: a null stream means the
//!   per-thread default stream (`CU_STREAM_PER_THREAD`).
//! - The oxide module is loaded once per CUDA context (the stream's context, or the current one) and its
//!   functions are cached. The image is the `#[cuda_module]` bundle embedded in the running executable;
//!   if there is none, `$MISTRALRS_PAGED_ATTN_B_PTX`, else `<crate>/mistralrs_paged_attn_b.ptx`.
//! - The merge grid is `num_sms * min(occupancy, ceil(batch * heads / num_sms))` with the occupancy of
//!   the oxide merge kernel (the grid-stride loop makes the output independent of the grid size).
//! - C++ exceptions of the reference (unsupported GQA group size) become the same stderr line and the
//!   same `cudaErrorUnknown` (999) return; runtime errors (`cudaFuncSetAttribute` refusing the dynamic
//!   shared memory, invalid launch configurations, launch failures) return the same cudaError_t codes.
#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
use crate::instances::{INSTANCES, Kind};
use cuda_core::sys as cu;
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::sync::Mutex;

const CU_STREAM_PER_THREAD: usize = 0x2;

struct Module {
    module: cu::CUmodule,
    funcs: HashMap<&'static str, cu::CUfunction>,
}
unsafe impl Send for Module {}

static MODULES: Mutex<Option<HashMap<usize, Module>>> = Mutex::new(None);
/// Every kernel name launched by these launchers (for the gate's coverage check).
pub static LAUNCHED: Mutex<Option<HashSet<&'static str>>> = Mutex::new(None);
static CC_MAJOR: Mutex<Option<i32>> = Mutex::new(None);

fn module_image() -> Vec<u8> {
    if let Ok(bundles) = cuda_core::embedded::artifact_bundles_from_current_exe() {
        for b in &bundles {
            if b.name == "mistralrs_paged_attn_b" {
                if let Some(p) = b
                    .payload(cuda_core::embedded::ArtifactPayloadKind::Cubin)
                    .or_else(|| b.payload(cuda_core::embedded::ArtifactPayloadKind::Ptx))
                {
                    let mut v = p.to_vec();
                    v.push(0);
                    return v;
                }
            }
        }
    }
    let path = std::env::var("MISTRALRS_PAGED_ATTN_B_PTX")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/mistralrs_paged_attn_b.ptx").to_string());
    let mut v = std::fs::read(&path).unwrap_or_else(|e| panic!("mistralrs-paged-attn-b: no embedded module and cannot read {path}: {e}"));
    v.push(0);
    v
}

fn check(r: cu::CUresult, what: &str) {
    if r != cu::cudaError_enum_CUDA_SUCCESS {
        panic!("mistralrs-paged-attn-b: {what} failed: {r:?}");
    }
}

fn cu_stream(stream: *mut c_void) -> cu::CUstream {
    if stream.is_null() { CU_STREAM_PER_THREAD as cu::CUstream } else { stream as cu::CUstream }
}

/// The context owning `stream` (made current if nothing is).
unsafe fn stream_ctx(stream: cu::CUstream) -> cu::CUcontext {
    unsafe {
        check(cu::cuInit(0), "cuInit");
        let mut ctx: cu::CUcontext = std::ptr::null_mut();
        if stream as usize != CU_STREAM_PER_THREAD {
            check(cu::cuStreamGetCtx(stream, &mut ctx), "cuStreamGetCtx");
        }
        if ctx.is_null() {
            check(cu::cuCtxGetCurrent(&mut ctx), "cuCtxGetCurrent");
        }
        if ctx.is_null() {
            let mut dev: cu::CUdevice = 0;
            check(cu::cuDeviceGet(&mut dev, 0), "cuDeviceGet");
            check(cu::cuDevicePrimaryCtxRetain(&mut ctx, dev), "cuDevicePrimaryCtxRetain");
            check(cu::cuCtxSetCurrent(ctx), "cuCtxSetCurrent");
        }
        ctx
    }
}

/// The function `name` of this crate's module in the context owning `stream`.
unsafe fn function(stream: cu::CUstream, name: &'static str) -> cu::CUfunction {
    unsafe {
        let ctx = stream_ctx(stream);
        let mut guard = MODULES.lock().unwrap();
        let map = guard.get_or_insert_with(HashMap::new);
        let m = map.entry(ctx as usize).or_insert_with(|| {
            let image = module_image();
            let mut module: cu::CUmodule = std::ptr::null_mut();
            check(cu::cuCtxPushCurrent_v2(ctx), "cuCtxPushCurrent");
            check(cu::cuModuleLoadData(&mut module, image.as_ptr() as *const c_void), "cuModuleLoadData");
            let mut popped: cu::CUcontext = std::ptr::null_mut();
            check(cu::cuCtxPopCurrent_v2(&mut popped), "cuCtxPopCurrent");
            Module { module, funcs: HashMap::new() }
        });
        let module = m.module;
        *m.funcs.entry(name).or_insert_with(|| {
            let mut f: cu::CUfunction = std::ptr::null_mut();
            let c = std::ffi::CString::new(name).unwrap();
            check(cu::cuModuleGetFunction(&mut f, module, c.as_ptr()), name);
            f
        })
    }
}

/// Whether this build's module contains the kernel `name` (the gate skips cases of kernels a partial
/// development build left out; the final coverage check requires all of them).
pub fn available(name: &str) -> bool {
    unsafe {
        let s = CU_STREAM_PER_THREAD as cu::CUstream;
        let _ = function(s, INSTANCES.iter().find(|i| i.kind == Kind::Mla && i.stages == 2).unwrap().name);
        let guard = MODULES.lock().unwrap();
        let ctx = stream_ctx(s);
        let m = &guard.as_ref().unwrap()[&(ctx as usize)];
        let mut f: cu::CUfunction = std::ptr::null_mut();
        let c = std::ffi::CString::new(name).unwrap();
        cu::cuModuleGetFunction(&mut f, m.module, c.as_ptr()) == cu::cudaError_enum_CUDA_SUCCESS
    }
}

/// The decode kernel `flashinfer_decode` would pick (for the gate's partial-build skip).
pub fn decode_kernel_name(dtype: u32, head_dim: u32, group: u32, sw: bool, sc: bool) -> Option<&'static str> {
    let sz = dtype_size(dtype);
    let vec = (16 / sz).max(head_dim / 32);
    let bdx = head_dim / vec;
    let bdz = 128u32.max(bdx * group) / (bdx * group);
    let tile = if group == 1 { 4 } else { 1 };
    INSTANCES
        .iter()
        .find(|i| i.kind == Kind::Decode && i.stages == 2 && i.dtype == dtype && i.vec == vec && i.bdx == bdx && i.bdy == group && i.bdz == bdz && i.tile == tile && i.sw == sw && i.sc == sc)
        .map(|i| i.name)
}

/// Device of the stream's context.
unsafe fn device(stream: cu::CUstream) -> cu::CUdevice {
    unsafe {
        let ctx = stream_ctx(stream);
        check(cu::cuCtxPushCurrent_v2(ctx), "cuCtxPushCurrent");
        let mut dev: cu::CUdevice = 0;
        check(cu::cuCtxGetDevice(&mut dev), "cuCtxGetDevice");
        let mut popped: cu::CUcontext = std::ptr::null_mut();
        check(cu::cuCtxPopCurrent_v2(&mut popped), "cuCtxPopCurrent");
        dev
    }
}

unsafe fn dev_attr(stream: cu::CUstream, a: cu::CUdevice_attribute) -> i32 {
    unsafe {
        let mut v = 0;
        check(cu::cuDeviceGetAttribute(&mut v, a, device(stream)), "cuDeviceGetAttribute");
        v
    }
}

/// `GetCudaComputeCapability().first` (queried once per process, like the C function's static).
unsafe fn cc_major(stream: cu::CUstream) -> i32 {
    let mut g = CC_MAJOR.lock().unwrap();
    *g.get_or_insert_with(|| unsafe { dev_attr(stream, cu::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) })
}

/// DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM
unsafe fn num_stages_smem(stream: cu::CUstream) -> u32 {
    if unsafe { cc_major(stream) } >= 8 { 2 } else { 1 }
}

pub const CUDA_SUCCESS: i32 = 0;
pub const CUDA_ERROR_INVALID_VALUE: i32 = 1;
pub const CUDA_ERROR_INVALID_CONFIGURATION: i32 = 9;
pub const CUDA_ERROR_UNKNOWN: i32 = 999;

/// driver CUresult -> runtime cudaError_t (the codes a launch / attribute call can produce here).
fn rt_err(r: cu::CUresult) -> i32 {
    match r as i32 {
        0 => 0,
        1 => 1,     // invalid value -> cudaErrorInvalidValue
        2 => 2,     // out of memory -> cudaErrorMemoryAllocation
        701 => 701, // launch out of resources
        400 => 400, // invalid handle -> cudaErrorInvalidResourceHandle
        e => e,
    }
}

fn cuda_error_string(e: i32) -> &'static str {
    match e {
        0 => "no error",
        1 => "invalid argument",
        2 => "out of memory",
        9 => "invalid configuration argument",
        400 => "invalid resource handle",
        701 => "too many resources requested for launch",
        999 => "unknown error",
        _ => "unrecognized error code",
    }
}

/// FLASHINFER_CUDA_CALL's report (built without NDEBUG).
fn report(e: i32, func: &str) {
    eprintln!("CUDA Error: {} ({e}) {}: line 0 at function {func}", cuda_error_string(e), file!());
}

/// A kernel argument list: each value in its own 8-byte slot.
pub struct Args(Vec<[u8; 8]>);
impl Args {
    pub fn new() -> Self {
        Args(Vec::new())
    }
    pub fn p<T>(mut self, v: *const T) -> Self {
        self.0.push((v as u64).to_le_bytes());
        self
    }
    pub fn i(mut self, v: i32) -> Self {
        self.0.push((v as u32 as u64).to_le_bytes());
        self
    }
    pub fn u(mut self, v: u32) -> Self {
        self.0.push((v as u64).to_le_bytes());
        self
    }
    pub fn l(mut self, v: i64) -> Self {
        self.0.push((v as u64).to_le_bytes());
        self
    }
    pub fn f(mut self, v: f32) -> Self {
        self.0.push((v.to_bits() as u64).to_le_bytes());
        self
    }
}
impl Default for Args {
    fn default() -> Self {
        Self::new()
    }
}

/// `cudaLaunchKernel` with the runtime's configuration checks; returns the cudaError_t.
pub unsafe fn launch(stream: *mut c_void, name: &'static str, grid: (u32, u32, u32), block: (u32, u32, u32), smem: u32, args: Args) -> i32 {
    // measured: the CUDA 13 runtime reports a zero grid / block dimension as cudaErrorInvalidValue
    if grid.0 == 0 || grid.1 == 0 || grid.2 == 0 || block.0 == 0 || block.1 == 0 || block.2 == 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if block.0 as u64 * block.1 as u64 * block.2 as u64 > 1024 || block.2 > 64 || grid.1 > 65535 || grid.2 > 65535 {
        return CUDA_ERROR_INVALID_CONFIGURATION;
    }
    unsafe {
        let s = cu_stream(stream);
        let f = function(s, name);
        LAUNCHED.lock().unwrap().get_or_insert_with(HashSet::new).insert(name);
        let mut slots = args.0;
        let mut ptrs: Vec<*mut c_void> = slots.iter_mut().map(|s| s.as_mut_ptr() as *mut c_void).collect();
        let r = cu::cuLaunchKernel(f, grid.0, grid.1, grid.2, block.0, block.1, block.2, smem, s, ptrs.as_mut_ptr(), std::ptr::null_mut());
        rt_err(r)
    }
}

/// `SetMaxDynamicSmemOnce` = `cudaFuncSetAttribute(kernel, MaxDynamicSharedMemorySize, bytes)`.
pub unsafe fn set_max_dynamic_smem(stream: *mut c_void, name: &'static str, bytes: u32) -> i32 {
    unsafe {
        let f = function(cu_stream(stream), name);
        let r = cu::cuFuncSetAttribute(f, cu::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes as i32);
        rt_err(r)
    }
}

/// `uint_fastdiv(d)` host constructor.
pub fn fastdiv(d: u32) -> (u32, u32, u32, u32) {
    let mut a: u32 = 0;
    let nc = u32::MAX.wrapping_sub(0u32.wrapping_sub(d) % d);
    let mut p: u32 = 31;
    let mut q1 = 0x8000_0000u32 / nc;
    let mut r1 = 0x8000_0000u32.wrapping_sub(q1.wrapping_mul(nc));
    let mut q2 = 0x7FFF_FFFFu32 / d;
    let mut r2 = 0x7FFF_FFFFu32.wrapping_sub(q2.wrapping_mul(d));
    loop {
        p = p.wrapping_add(1);
        if r1 >= nc.wrapping_sub(r1) {
            q1 = q1.wrapping_mul(2).wrapping_add(1);
            r1 = r1.wrapping_mul(2).wrapping_sub(nc);
        } else {
            q1 = q1.wrapping_mul(2);
            r1 = r1.wrapping_mul(2);
        }
        if r2.wrapping_add(1) >= d.wrapping_sub(r2) {
            if q2 >= 0x7FFF_FFFF {
                a = 1;
            }
            q2 = q2.wrapping_mul(2).wrapping_add(1);
            r2 = r2.wrapping_mul(2).wrapping_add(1).wrapping_sub(d);
        } else {
            if q2 >= 0x8000_0000 {
                a = 1;
            }
            q2 = q2.wrapping_mul(2);
            r2 = r2.wrapping_mul(2).wrapping_add(1);
        }
        let delta = d.wrapping_sub(1).wrapping_sub(r2);
        if !(p < 64 && (q1 < delta || (q1 == delta && r1 == 0))) {
            break;
        }
    }
    (d, q2.wrapping_add(1), p.wrapping_sub(32), a)
}

fn dtype_size(dtype: u32) -> u32 {
    if dtype == 2 { 4 } else { 2 }
}

fn find(kind: Kind, pred: impl Fn(&crate::instances::Inst) -> bool) -> &'static str {
    INSTANCES
        .iter()
        .find(|i| i.kind == kind && pred(i))
        .unwrap_or_else(|| panic!("mistralrs-paged-attn-b: no {kind:?} instance for this configuration"))
        .name
}

// ================================================================================================
// VariableLengthMergeStates<DType, DType, int32_t>(v, s, indptr, v_merged, s_merged=nullptr,
//   max_seq_len, seq_len=nullptr, num_heads, head_dim, enable_pdl=false, stream)

unsafe fn variable_length_merge_states(
    dtype: u32, v: *mut c_void, s: *mut f32, indptr: *const i32, v_merged: *mut c_void, s_merged: *mut f32, max_seq_len: u32,
    num_heads: u32, head_dim: u32, stream: *mut c_void,
) -> i32 {
    unsafe {
        let cs = cu_stream(stream);
        let num_sms = dev_attr(cs, cu::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
        let sz = dtype_size(dtype);
        let vec = (16 / sz).max(head_dim / 32);
        let bdx = head_dim / vec;
        let num_threads = 128u32;
        let bdy = num_threads / bdx;
        let smem_size = 4 * bdy * head_dim * sz + num_threads * 4;
        let name = find(Kind::Merge, |i| i.dtype == dtype && i.vec == vec && i.bdx == bdx && i.bdy == bdy);
        let f = function(cs, name);
        let mut occ: i32 = 0;
        let r = rt_err(cu::cuOccupancyMaxActiveBlocksPerMultiprocessor(&mut occ, f, num_threads as i32, smem_size as usize));
        if r != 0 {
            report(r, "cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel, num_threads, smem_size)");
            return r;
        }
        // min(int, unsigned) -> unsigned
        let cd = max_seq_len.wrapping_mul(num_heads).wrapping_add(num_sms as u32).wrapping_sub(1) / num_sms as u32;
        let num_blocks_per_sm = (occ as u32).min(cd) as i32;
        let nblks = (num_sms as u32).wrapping_mul(num_blocks_per_sm as u32);
        let r = set_max_dynamic_smem(stream, name, smem_size);
        if r != 0 {
            report(r, "::flashinfer::SetMaxDynamicSmemOnce((const void*)(kernel), smem_size)");
            return r;
        }
        let args = Args::new().p(v).p(s).p(indptr).p(v_merged).p(s_merged).u(max_seq_len).p(std::ptr::null::<u32>()).u(num_heads);
        let r = launch(stream, name, (nblks, 1, 1), (bdx, bdy, 1), smem_size, args);
        if r != 0 {
            report(r, "cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream)");
            return r;
        }
        0
    }
}

// ================================================================================================
// flashinfer_decode.cu

/// BatchDecodeWithPagedKVCacheDispatched<HEAD_DIM, kNone, DefaultAttention<false, SW, SC, false>, ...>
/// (through run_flashinfer_decode). Err(msg) = a thrown flashinfer::Error.
unsafe fn run_flashinfer_decode(
    dtype: u32, head_dim: u32, sw: bool, sc: bool, q: *mut c_void, key_cache: *mut c_void, value_cache: *mut c_void, kv_indptr: *const i32,
    kv_indices: *const i32, kv_last_page_len: *const i32, request_indices: *const i32, kv_tile_indices: *const i32, o_indptr: *const i32,
    kv_chunk_size_ptr: *const i32, block_valid_mask: *const u8, o: *mut c_void, tmp_v: *mut c_void, tmp_s: *mut c_void, batch_size: i32,
    padded_batch_size: i32, num_qo_heads: i32, num_kv_heads: i32, page_size: i32, q_stride_n: i32, q_stride_h: i32, sm_scale: f32,
    window_left: i32, logits_soft_cap: f32, stream: *mut c_void,
) -> Result<i32, String> {
    unsafe {
        let (fd, fm, fs, fa) = fastdiv(page_size as u32);
        let nkv = num_kv_heads as u32;
        let ps = page_size as u32;
        let stride_page = nkv.wrapping_mul(ps).wrapping_mul(head_dim);
        let stride_n = head_dim;
        let stride_h = ps.wrapping_mul(head_dim);
        let nqo = num_qo_heads as u32;
        let padded = padded_batch_size as u32;
        let sz = dtype_size(dtype);
        let vec = (16 / sz).max(head_dim / 32);
        let bdx = head_dim / vec;
        let group = nqo / nkv;
        if ![1, 2, 3, 4, 8, 16].contains(&group) {
            return Err(format!(
                "Error in function 'BatchDecodeWithPagedKVCacheDispatched' at {}:{}: Unsupported group_size: {group}",
                "src/cuda/flashinfer/attention/decode.cuh", 755
            ));
        }
        let bdy = group;
        let num_threads = 128u32.max(bdx * bdy);
        let bdz = num_threads / (bdx * bdy);
        let tile = if group == 1 { 4 } else { 1 };
        let stages = num_stages_smem(cu_stream(stream));
        let smem_size = 2 * stages * tile * bdy * bdz * head_dim * sz + (tile * num_threads * 8).max(2 * bdy * bdz * 4);
        let name = find(Kind::Decode, |i| {
            i.dtype == dtype && i.stages == stages && i.tile == tile && i.vec == vec && i.bdx == bdx && i.bdy == bdy && i.bdz == bdz && i.sw == sw && i.sc == sc
        });
        let r = set_max_dynamic_smem(stream, name, smem_size);
        if r != 0 {
            report(r, "::flashinfer::SetMaxDynamicSmemOnce((const void*)(kernel), smem_size)");
            return Ok(r);
        }
        let grid = (padded, nkv, 1);
        let block = (bdx, bdy, bdz);
        let partition = !tmp_v.is_null();
        let (ko, klse) = if partition { (tmp_v, tmp_s as *mut f32) } else { (o, std::ptr::null_mut()) };
        let args = Args::new()
            .p(q)
            .p(key_cache)
            .p(value_cache)
            .p(kv_indices)
            .p(kv_indptr)
            .p(kv_last_page_len)
            .p(ko)
            .p(klse)
            .p(request_indices)
            .p(kv_tile_indices)
            .p(kv_chunk_size_ptr)
            .p(block_valid_mask)
            .u(nqo)
            .i(q_stride_n)
            .i(q_stride_h)
            .i(window_left)
            .f(logits_soft_cap)
            .f(sm_scale)
            .u(partition as u32)
            .u(fd)
            .u(fm)
            .u(fs)
            .u(fa)
            .u(batch_size as u32)
            .u(stride_page)
            .u(stride_n)
            .u(stride_h);
        let r = launch(stream, name, grid, block, smem_size, args);
        if r != 0 {
            report(r, "cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream)");
            return Ok(r);
        }
        if partition {
            let r = variable_length_merge_states(
                dtype,
                tmp_v,
                tmp_s as *mut f32,
                o_indptr,
                o,
                std::ptr::null_mut(),
                batch_size as u32,
                nqo,
                head_dim,
                stream,
            );
            if r != 0 {
                report(r, "VariableLengthMergeStates(...)");
                return Ok(r);
            }
        }
        Ok(0)
    }
}

pub unsafe extern "C" fn flashinfer_decode(
    q: *mut c_void, key_cache: *mut c_void, value_cache: *mut c_void, kv_indptr: *const i32, kv_indices: *const i32,
    kv_last_page_len: *const i32, request_indices: *const i32, kv_tile_indices: *const i32, o_indptr: *const i32,
    kv_chunk_size_ptr: *const i32, block_valid_mask: *const u8, o: *mut c_void, tmp_v: *mut c_void, tmp_s: *mut c_void, batch_size: i32,
    padded_batch_size: i32, num_qo_heads: i32, num_kv_heads: i32, head_size: i32, page_size: i32, q_stride_n: i32, q_stride_h: i32,
    sm_scale: f32, window_left: i32, logits_soft_cap: f32, dtype: u32, stream: *mut c_void,
) -> i32 {
    if dtype > 2 {
        eprintln!("FlashInfer decode received unsupported dtype {dtype}");
        return CUDA_ERROR_INVALID_VALUE;
    }
    if ![64, 128, 256, 512].contains(&head_size) {
        eprintln!("FlashInfer decode received unsupported head_size {head_size}");
        return CUDA_ERROR_INVALID_VALUE;
    }
    let sw = window_left >= 0;
    let sc = logits_soft_cap > 0.0;
    let r = unsafe {
        run_flashinfer_decode(
            dtype,
            head_size as u32,
            sw,
            sc,
            q,
            key_cache,
            value_cache,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            request_indices,
            kv_tile_indices,
            o_indptr,
            kv_chunk_size_ptr,
            block_valid_mask,
            o,
            tmp_v,
            tmp_s,
            batch_size,
            padded_batch_size,
            num_qo_heads,
            num_kv_heads,
            page_size,
            q_stride_n,
            q_stride_h,
            sm_scale,
            window_left,
            logits_soft_cap,
            stream,
        )
    };
    match r {
        Ok(0) => 0,
        Ok(e) => {
            eprintln!("FlashInfer decode failed: {}", cuda_error_string(e));
            e
        }
        Err(msg) => {
            eprintln!("FlashInfer decode failed: {msg}");
            CUDA_ERROR_UNKNOWN
        }
    }
}

pub unsafe extern "C" fn reshape_and_cache_flashinfer(
    key: *mut c_void, value: *mut c_void, key_cache: *mut c_void, value_cache: *mut c_void, slot_mapping: *mut i64, num_tokens: i32,
    num_heads: i32, head_size: i32, block_size: i32, key_stride: i32, value_stride: i32, dtype: u32, stream: *mut c_void,
) {
    if dtype > 2 {
        eprintln!("reshape_and_cache_flashinfer received unsupported dtype {dtype}");
        return;
    }
    let grid = (num_tokens as u32, 1, 1);
    let block = (num_heads.wrapping_mul(head_size).min(512) as u32, 1, 1);
    let name = find(Kind::Reshape, |i| i.dtype == dtype);
    let args = Args::new()
        .p(key)
        .p(value)
        .p(key_cache)
        .p(value_cache)
        .p(slot_mapping)
        .i(num_heads)
        .i(head_size)
        .i(block_size)
        .i(key_stride)
        .i(value_stride);
    unsafe { launch(stream, name, grid, block, 0, args) };
}

pub unsafe extern "C" fn gather_kv_cache_flashinfer(
    key_cache: *mut c_void, value_cache: *mut c_void, k_out: *mut c_void, v_out: *mut c_void, block_table: *const i32,
    cu_seq_lens: *const i32, num_tokens: i32, _num_seqs: i32, block_size: i32, block_table_stride: i32, num_kv_heads: i32, head_size: i32,
    dtype: u32, stream: *mut c_void,
) {
    if dtype > 2 {
        eprintln!("gather_kv_cache_flashinfer received unsupported dtype {dtype}");
        return;
    }
    let grid = (num_tokens as u32, 1, 1);
    let block = (num_kv_heads.wrapping_mul(head_size).min(512) as u32, 1, 1);
    let name = find(Kind::Gather, |i| i.dtype == dtype);
    let args = Args::new()
        .p(key_cache)
        .p(value_cache)
        .p(k_out)
        .p(v_out)
        .p(block_table)
        .p(cu_seq_lens)
        .i(num_tokens)
        .i(block_size)
        .i(block_table_stride)
        .i(num_kv_heads)
        .i(head_size);
    unsafe { launch(stream, name, grid, block, 0, args) };
}

// ================================================================================================
// flashinfer_mla_decode.cu (run_mla_decode<DType>: HEAD_DIM_CKV 512, HEAD_DIM_KPE 64, tmp_v = nullptr)

pub unsafe extern "C" fn flashinfer_mla_decode(
    q_nope: *mut c_void, q_pe: *mut c_void, ckv_cache: *mut c_void, kpe_cache: *mut c_void, kv_indptr: *const i32, kv_indices: *const i32,
    kv_last_page_len: *const i32, o: *mut c_void, batch_size: i32, num_qo_heads: i32, page_size: i32, sm_scale: f32, _window_left: i32,
    _logits_soft_cap: f32, _rope_scale: f32, _rope_theta: f32, request_indices: *const i32, kv_tile_indices: *const i32,
    _o_indptr: *const i32, kv_chunk_size_ptr: *const i32, dtype: u32, stream: *mut c_void,
) -> i32 {
    if dtype > 2 {
        eprintln!("FlashInfer MLA decode received unsupported dtype {dtype}");
        return CUDA_ERROR_INVALID_VALUE;
    }
    unsafe {
        let (fd, fm, fs, fa) = fastdiv(page_size as u32);
        let ps = page_size as u32;
        let sz = dtype_size(dtype);
        let nqo = num_qo_heads as u32;
        let gdy = nqo.wrapping_add(15) / 16;
        let stages = num_stages_smem(cu_stream(stream));
        let smem_size = stages * 8 * (512 + 64) * sz + 4096;
        let name = find(Kind::Mla, |i| i.dtype == dtype && i.stages == stages);
        let mut r = set_max_dynamic_smem(stream, name, smem_size);
        if r != 0 {
            report(r, "::flashinfer::SetMaxDynamicSmemOnce((const void*)(kernel), smem_size)");
        } else {
            let args = Args::new()
                .p(q_nope)
                .p(q_pe)
                .p(ckv_cache)
                .p(kpe_cache)
                .p(kv_indices)
                .p(kv_indptr)
                .p(kv_last_page_len)
                .p(o)
                .p(std::ptr::null::<f32>())
                .p(request_indices)
                .p(kv_tile_indices)
                .p(kv_chunk_size_ptr)
                .p(std::ptr::null::<u8>())
                .u(nqo)
                .f(sm_scale)
                .u(0)
                .u(fd)
                .u(fm)
                .u(fs)
                .u(fa)
                .u(batch_size as u32)
                .u(ps.wrapping_mul(512))
                .u(ps.wrapping_mul(64))
                .u(512)
                .u(64);
            r = launch(stream, name, (batch_size as u32, gdy, 1), (32, 8, 1), smem_size, args);
            if r != 0 {
                report(r, "cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream)");
            }
        }
        if r != 0 {
            eprintln!("FlashInfer MLA decode failed: {}", cuda_error_string(r));
        }
        r
    }
}
