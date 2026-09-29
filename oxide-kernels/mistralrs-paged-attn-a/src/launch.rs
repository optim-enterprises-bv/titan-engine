//! Pure-Rust twins of the group-1 extern "C" host launchers of mistralrs-paged-attn
//! (libmistralrspagedattention.a): same names, same parameter lists and C ABI as the C definitions
//! (mistralrs-paged-attn/src/cuda/*.cu, declared in src/cuda/ffi.rs), same grid / block /
//! shared-memory selection and host-side control flow, launching the oxide kernels of this crate on
//! the caller's stream through the CUDA driver API.
//!
//! - The reference objects are built with `--default-stream per-thread`: a null stream means the
//!   per-thread default stream (`CU_STREAM_PER_THREAD`), not the legacy stream.
//! - The oxide module is loaded once per CUDA context (the stream's context, or the current one for
//!   the per-thread stream) and its functions are cached. The image is the `#[cuda_module]` bundle
//!   embedded in the running executable; if there is none, `$MISTRALRS_PAGED_ATTN_A_PTX`, else
//!   `<crate>/mistralrs_paged_attn_a.ptx`.
//! - `CUDA_CHECK(cudaGetLastError())` after a launch is reproduced for the launch-configuration
//!   errors a Rust launcher can observe itself (a zero grid / block dimension, a dynamic shared memory
//!   request the device refuses): it prints `CUDA error at <file>:<line>: <msg>` and `exit`s with the
//!   runtime error code, as the C code does. Errors left pending by unrelated earlier runtime calls
//!   cannot be seen from the driver API and are not reproduced.
#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
use cuda_core::sys as cu;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;

const CU_STREAM_PER_THREAD: usize = 0x2;

struct Module {
    module: cu::CUmodule,
    funcs: HashMap<&'static str, cu::CUfunction>,
}
unsafe impl Send for Module {}

static MODULES: Mutex<Option<HashMap<usize, Module>>> = Mutex::new(None);

fn module_image() -> Vec<u8> {
    if let Ok(bundles) = cuda_core::embedded::artifact_bundles_from_current_exe() {
        for b in &bundles {
            if b.name == "mistralrs_paged_attn_a" {
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
    let path = std::env::var("MISTRALRS_PAGED_ATTN_A_PTX")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/mistralrs_paged_attn_a.ptx").to_string());
    let mut v = std::fs::read(&path).unwrap_or_else(|e| panic!("mistralrs-paged-attn-a: no embedded module and cannot read {path}: {e}"));
    v.push(0);
    v
}

fn check(r: cu::CUresult, what: &str) {
    if r != cu::cudaError_enum_CUDA_SUCCESS {
        panic!("mistralrs-paged-attn-a: {what} failed: {r:?}");
    }
}

fn cu_stream(stream: *mut c_void) -> cu::CUstream {
    if stream.is_null() { CU_STREAM_PER_THREAD as cu::CUstream } else { stream as cu::CUstream }
}

/// The function `name` of this crate's module in the context owning `stream`.
unsafe fn function(stream: cu::CUstream, name: &'static str) -> cu::CUfunction {
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

/// cudaError_t values the launchers can report.

fn cuda_error_string(e: i32) -> &'static str {
    match e {
        1 => "invalid argument",
        9 => "invalid configuration argument",
        701 => "too many resources requested for launch",
        _ => "unknown error",
    }
}

/// `CUDA_CHECK(err)` of the .cu files: print and exit with the error code.
fn cuda_check(err: i32, file: &str, line: u32) {
    if err != 0 {
        eprintln!("CUDA error at {file}:{line}: {}", cuda_error_string(err));
        std::process::exit(err);
    }
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
    pub fn l(mut self, v: i64) -> Self {
        self.0.push((v as u64).to_le_bytes());
        self
    }
    pub fn f(mut self, v: f32) -> Self {
        self.0.push((v.to_bits() as u64).to_le_bytes());
        self
    }
}

/// `kernel<<<grid, block, smem, stream>>>(args)` with the runtime's configuration checks; returns
/// the cudaError_t the runtime would record (0 = launched).
unsafe fn launch(stream: *mut c_void, name: &'static str, grid: (u32, u32, u32), block: (u32, u32, u32), smem: u32, args: Args) -> i32 {
    // Invalid configurations (a zero grid or block dimension, too many threads, too much shared
    // memory) are refused by cuLaunchKernel with CUDA_ERROR_INVALID_VALUE, which is also what the
    // CUDA 13 runtime reports for them (cudaErrorInvalidValue, "invalid argument").
    unsafe {
        let s = cu_stream(stream);
        let f = function(s, name);
        let mut slots = args.0;
        let mut ptrs: Vec<*mut c_void> = slots.iter_mut().map(|s| s.as_mut_ptr() as *mut c_void).collect();
        let r = cu::cuLaunchKernel(f, grid.0, grid.1, grid.2, block.0, block.1, block.2, smem, s, ptrs.as_mut_ptr(), std::ptr::null_mut());
        r as i32
    }
}

/// `cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)`.
unsafe fn set_max_dynamic_smem(stream: *mut c_void, name: &'static str, bytes: i32) -> i32 {
    unsafe {
        let f = function(cu_stream(stream), name);
        let r = cu::cuFuncSetAttribute(f, cu::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes);
        if r == cu::cudaError_enum_CUDA_SUCCESS { 0 } else { r as i32 }
    }
}

// ================================================================================================
// reshape_and_cache_kernel.cu

pub unsafe extern "C" fn reshape_and_cache(
    key: *mut c_void, value: *mut c_void, key_cache: *mut c_void, value_cache: *mut c_void, slot_mapping: *mut i64, num_tokens: i32,
    num_heads: i32, head_size: i32, block_size: i32, x: i32, key_stride: i32, value_stride: i32, stream: *mut c_void, dtype: u32,
    cache_dtype: u32, k_scale: *mut f32, v_scale: *mut f32,
) {
    let grid = (num_tokens as u32, 1, 1);
    let block = ((num_heads.wrapping_mul(head_size)).min(512) as u32, 1, 1);
    let name = match (cache_dtype == 3, dtype) {
        (true, 0) => Some("rac_f16_fp8"),
        (true, 1) => Some("rac_bf16_fp8"),
        (true, 2) => Some("rac_f32_fp8"),
        (false, 0) => Some("rac_f16"),
        (false, 1) => Some("rac_bf16"),
        (false, 2) => Some("rac_f32"),
        _ => None,
    };
    let mut err = 0;
    if let Some(name) = name {
        let a = Args::new()
            .p(key)
            .p(value)
            .p(key_cache)
            .p(value_cache)
            .p(slot_mapping)
            .i(key_stride)
            .i(value_stride)
            .i(num_heads)
            .i(head_size)
            .i(block_size)
            .i(x)
            .p(k_scale)
            .p(v_scale);
        err = unsafe { launch(stream, name, grid, block, 0, a) };
    }
    cuda_check(err, "src/cuda/reshape_and_cache_kernel.cu", 140);
}

// ================================================================================================
// gather_kv_cache_kernel.cu

pub unsafe extern "C" fn gather_kv_cache(
    key_cache: *mut c_void, value_cache: *mut c_void, k_out: *mut c_void, v_out: *mut c_void, k_scale: *mut c_void, v_scale: *mut c_void,
    block_table: *const i32, cu_seq_lens: *const i32, num_tokens: i32, num_seqs: i32, block_size: i32, block_table_stride: i32,
    num_kv_heads: i32, head_size: i32, x: i32, stream: *mut c_void, out_dtype: u32, cache_dtype: u32,
) {
    if num_tokens <= 0 {
        return;
    }
    let grid = (num_tokens as u32, 1, 1);
    let block = ((num_kv_heads.wrapping_mul(head_size)).min(512) as u32, 1, 1);
    let name = match (cache_dtype == 3, out_dtype) {
        (true, 0) => Some("gkv_fp8_f16"),
        (true, 1) => Some("gkv_fp8_bf16"),
        (true, 2) => Some("gkv_fp8_f32"),
        (false, 0) => Some("gkv_f16"),
        (false, 1) => Some("gkv_bf16"),
        (false, 2) => Some("gkv_f32"),
        _ => None,
    };
    let mut err = 0;
    if let Some(name) = name {
        let a = Args::new()
            .p(key_cache)
            .p(value_cache)
            .p(k_out)
            .p(v_out)
            .p(k_scale)
            .p(v_scale)
            .p(block_table)
            .p(cu_seq_lens)
            .i(num_tokens)
            .i(num_seqs)
            .i(block_size)
            .i(block_table_stride)
            .i(num_kv_heads)
            .i(head_size)
            .i(x);
        err = unsafe { launch(stream, name, grid, block, 0, a) };
    }
    cuda_check(err, "src/cuda/gather_kv_cache_kernel.cu", 175);
}

// ================================================================================================
// update_kvscales.cu (no error check)

unsafe fn update_scales(name: &'static str, k: *mut c_void, v: *mut c_void, num_elements: i64, k_scales: *mut f32, v_scales: *mut f32, stream: i64) {
    const THREADS: i64 = 512;
    let mut blocks = (num_elements.wrapping_add(THREADS - 1)) / THREADS;
    if blocks == 0 {
        blocks = 1;
    }
    if blocks > 65535 {
        blocks = 65535;
    }
    let smem = (2 * THREADS * 4) as u32;
    let a = Args::new().p(k).p(v).l(num_elements).p(k_scales).p(v_scales);
    unsafe { launch(stream as usize as *mut c_void, name, (blocks as i32 as u32, 1, 1), (THREADS as u32, 1, 1), smem, a) };
}
pub unsafe extern "C" fn update_kv_scales_f32(k: *mut c_void, v: *mut c_void, num_elements: i64, k_scales: *mut f32, v_scales: *mut f32, stream: i64) {
    unsafe { update_scales("kvscales_f32", k, v, num_elements, k_scales, v_scales, stream) }
}
pub unsafe extern "C" fn update_kv_scales_f16(k: *mut c_void, v: *mut c_void, num_elements: i64, k_scales: *mut f32, v_scales: *mut f32, stream: i64) {
    unsafe { update_scales("kvscales_f16", k, v, num_elements, k_scales, v_scales, stream) }
}
pub unsafe extern "C" fn update_kv_scales_bf16(k: *mut c_void, v: *mut c_void, num_elements: i64, k_scales: *mut f32, v_scales: *mut f32, stream: i64) {
    unsafe { update_scales("kvscales_bf16", k, v, num_elements, k_scales, v_scales, stream) }
}

// ================================================================================================
// copy_blocks_kernel.cu (no error check)

unsafe fn copy_blocks(
    name: &'static str, key_cache_ptrs: *mut i64, value_cache_ptrs: *mut i64, block_mapping: *const i64, num_layers: i32, num_pairs: i32,
    numel_per_block_key: i32, numel_per_block_value: i32, stream: i64,
) {
    let mut num_threads = numel_per_block_key;
    if numel_per_block_value > num_threads {
        num_threads = numel_per_block_value;
    }
    if num_threads > 1024 {
        num_threads = 1024;
    }
    let a = Args::new().p(key_cache_ptrs).p(value_cache_ptrs).p(block_mapping).i(numel_per_block_key).i(numel_per_block_value);
    unsafe { launch(stream as usize as *mut c_void, name, (num_layers as u32, num_pairs as u32, 1), (num_threads as u32, 1, 1), 0, a) };
}
pub unsafe extern "C" fn copy_blocks_f32(k: *mut i64, v: *mut i64, m: *const i64, nl: i32, np: i32, nk: i32, nv: i32, stream: i64) {
    unsafe { copy_blocks("copy_blocks_kernel_f32", k, v, m, nl, np, nk, nv, stream) }
}
pub unsafe extern "C" fn copy_blocks_f16(k: *mut i64, v: *mut i64, m: *const i64, nl: i32, np: i32, nk: i32, nv: i32, stream: i64) {
    unsafe { copy_blocks("copy_blocks_kernel_f16", k, v, m, nl, np, nk, nv, stream) }
}
pub unsafe extern "C" fn copy_blocks_bf16(k: *mut i64, v: *mut i64, m: *const i64, nl: i32, np: i32, nk: i32, nv: i32, stream: i64) {
    unsafe { copy_blocks("copy_blocks_kernel_bf16", k, v, m, nl, np, nk, nv, stream) }
}

// ================================================================================================
// concat_and_cache_mla_kernel.cu

pub unsafe extern "C" fn concat_and_cache_mla(
    ckv: *mut c_void, k_pe: *mut c_void, ckv_cache: *mut c_void, kpe_cache: *mut c_void, slot_mapping: *mut i64, num_tokens: i32,
    kv_lora_rank: i32, kpe_head_dim: i32, block_size: i32, ckv_stride: i32, kpe_stride: i32, stream: *mut c_void, dtype: u32,
) {
    let max_dim = if kv_lora_rank > kpe_head_dim { kv_lora_rank } else { kpe_head_dim };
    let grid = (num_tokens as u32, 1, 1);
    let block = (max_dim.min(512) as u32, 1, 1);
    let name = match dtype {
        0 => Some("ccmla_f16"),
        1 => Some("ccmla_bf16"),
        2 => Some("ccmla_f32"),
        _ => None,
    };
    let mut err = 0;
    if let Some(name) = name {
        let a = Args::new()
            .p(ckv)
            .p(k_pe)
            .p(ckv_cache)
            .p(kpe_cache)
            .p(slot_mapping)
            .i(ckv_stride)
            .i(kpe_stride)
            .i(kv_lora_rank)
            .i(kpe_head_dim)
            .i(block_size);
        err = unsafe { launch(stream, name, grid, block, 0, a) };
    }
    cuda_check(err, "src/cuda/concat_and_cache_mla_kernel.cu", 87);
}

// ================================================================================================
// gather_mla_cache_kernel.cu

pub unsafe extern "C" fn gather_mla_cache(
    ckv_cache: *mut c_void, kpe_cache: *mut c_void, ckv_out: *mut c_void, kpe_out: *mut c_void, block_table: *const i32,
    cu_seq_lens: *const i32, token_to_seq: *const i32, num_tokens: i32, block_size: i32, block_table_stride: i32, kv_lora_rank: i32,
    kpe_head_dim: i32, stream: *mut c_void, dtype: u32,
) {
    if num_tokens <= 0 {
        return;
    }
    let name = match dtype {
        0 => Some("gmla_f16"),
        1 => Some("gmla_bf16"),
        2 => Some("gmla_f32"),
        _ => None,
    };
    let mut err = 0;
    if let Some(name) = name {
        let a = Args::new()
            .p(ckv_cache)
            .p(kpe_cache)
            .p(ckv_out)
            .p(kpe_out)
            .p(block_table)
            .p(cu_seq_lens)
            .p(token_to_seq)
            .i(num_tokens)
            .i(block_size)
            .i(block_table_stride)
            .i(kv_lora_rank)
            .i(kpe_head_dim);
        err = unsafe { launch(stream, name, (num_tokens as u32, 1, 1), (256, 1, 1), 0, a) };
    }
    cuda_check(err, "src/cuda/gather_mla_cache_kernel.cu", 95);
}

// ================================================================================================
// pagedattention_v{1,2}_{f32,f16,bf16}.cu (+ pagedattention.cuh launchers)

const PA_HEADS: [i32; 8] = [64, 80, 96, 112, 128, 192, 256, 512];

fn pa_name(v: u32, dt: &str, fp8: bool, head: i32, block: i32) -> &'static str {
    // names are interned once per (v, dt, cache, head, block)
    static NAMES: Mutex<Option<HashMap<String, &'static str>>> = Mutex::new(None);
    let key = if v == 3 { format!("pa2r_{dt}_{head}") } else { format!("pa{v}_{dt}_{}_{head}_{block}", if fp8 { "e" } else { "a" }) };
    let mut g = NAMES.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    if let Some(n) = m.get(&key) {
        return n;
    }
    let s: &'static str = Box::leak(key.clone().into_boxed_str());
    m.insert(key, s);
    s
}

/// `DIVIDE_ROUND_UP(a, b)` in int.
fn div_round_up(a: i32, b: i32) -> i32 {
    a.wrapping_add(b).wrapping_sub(1).wrapping_div(b)
}

unsafe fn pa_v1(
    dt: &'static str, file: &'static str, line: u32, out: *mut c_void, query: *mut c_void, key_cache: *mut c_void, value_cache: *mut c_void,
    alibi_slopes: *mut c_void, num_kv_heads: i32, scale: f32, softcapping: f32, block_tables: *mut u32, context_lens: *mut u32, block_size: i32,
    max_context_len: i32, num_seqs: i32, num_heads: i32, head_size: i32, max_num_blocks_per_seq: i32, q_stride: i32, kv_block_stride: i32,
    kv_head_stride: i32, stream: *mut c_void, cache_dtype: u32, k_scale: *mut f32, v_scale: *mut f32, sinks: *const f32,
) {
    let fp8 = cache_dtype == 3;
    let mut err = 0;
    if matches!(block_size, 8 | 16 | 32) && PA_HEADS.contains(&head_size) {
        const NUM_WARPS: i32 = 4;
        let padded_max_context_len = div_round_up(max_context_len, block_size).wrapping_mul(block_size);
        let logits_size = padded_max_context_len.wrapping_mul(4);
        let outputs_size = (NUM_WARPS / 2) * head_size * 4;
        let shared_mem_size = logits_size.max(outputs_size);
        let name = pa_name(1, dt, fp8, head_size, block_size);
        err = unsafe { set_max_dynamic_smem(stream, name, shared_mem_size) };
        let a = Args::new()
            .p(out)
            .p(query)
            .p(key_cache)
            .p(value_cache)
            .i(num_kv_heads)
            .f(scale)
            .f(softcapping)
            .p(block_tables)
            .p(context_lens)
            .i(max_num_blocks_per_seq)
            .p(alibi_slopes)
            .i(q_stride)
            .i(kv_block_stride)
            .i(kv_head_stride)
            .p(k_scale)
            .p(v_scale)
            .p(sinks);
        let e = unsafe { launch(stream, name, (num_heads as u32, num_seqs as u32, 1), (128, 1, 1), shared_mem_size as u32, a) };
        if err == 0 {
            err = e;
        }
    }
    cuda_check(err, file, line);
}

unsafe fn pa_v2(
    dt: &'static str, file: &'static str, line: u32, out: *mut c_void, exp_sums: *mut f32, max_logits: *mut f32, tmp_out: *mut c_void,
    query: *mut c_void, key_cache: *mut c_void, value_cache: *mut c_void, alibi_slopes: *mut c_void, num_kv_heads: i32, scale: f32,
    softcapping: f32, block_tables: *mut u32, context_lens: *mut u32, block_size: i32, max_context_len: i32, num_seqs: i32, num_heads: i32,
    head_size: i32, max_num_blocks_per_seq: i32, q_stride: i32, kv_block_stride: i32, kv_head_stride: i32, stream: *mut c_void,
    cache_dtype: u32, k_scale: *mut f32, v_scale: *mut f32, sinks: *const f32,
) {
    let fp8 = cache_dtype == 3;
    let mut err = 0;
    if matches!(block_size, 8 | 16 | 32) && PA_HEADS.contains(&head_size) {
        const NUM_WARPS: i32 = 4;
        const PARTITION_SIZE: i32 = 512;
        let max_num_partitions = div_round_up(max_context_len, PARTITION_SIZE);
        let logits_size = PARTITION_SIZE * 4;
        let outputs_size = (NUM_WARPS / 2) * head_size * 4;
        let shared_mem_size = logits_size.max(outputs_size);
        let reduce_shared_mem_size = 2 * max_num_partitions.wrapping_mul(4);
        let a = Args::new()
            .p(exp_sums)
            .p(max_logits)
            .p(tmp_out)
            .p(query)
            .p(key_cache)
            .p(value_cache)
            .i(num_kv_heads)
            .f(scale)
            .f(softcapping)
            .p(block_tables)
            .p(context_lens)
            .i(max_num_blocks_per_seq)
            .p(alibi_slopes)
            .i(q_stride)
            .i(kv_block_stride)
            .i(kv_head_stride)
            .p(k_scale)
            .p(v_scale)
            .p(sinks);
        err = unsafe {
            launch(stream, pa_name(2, dt, fp8, head_size, block_size), (num_heads as u32, num_seqs as u32, max_num_partitions as u32), (128, 1, 1), shared_mem_size as u32, a)
        };
        let a = Args::new().p(out).p(exp_sums).p(max_logits).p(tmp_out).p(context_lens).i(max_num_partitions).p(sinks);
        let e = unsafe { launch(stream, pa_name(3, dt, fp8, head_size, block_size), (num_heads as u32, num_seqs as u32, 1), (128, 1, 1), reduce_shared_mem_size as u32, a) };
        if err == 0 {
            err = e;
        }
    }
    cuda_check(err, file, line);
}

macro_rules! pa_launchers {
    ($($v1:ident $v2:ident $dt:literal $l1:literal $l2:literal;)*) => {$(
        pub unsafe extern "C" fn $v1(
            out: *mut c_void, query: *mut c_void, key_cache: *mut c_void, value_cache: *mut c_void, alibi_slopes: *mut c_void, num_kv_heads: i32,
            scale: f32, softcapping: f32, block_tables: *mut u32, context_lens: *mut u32, block_size: i32, max_context_len: i32, num_seqs: i32,
            num_heads: i32, head_size: i32, max_num_blocks_per_seq: i32, q_stride: i32, kv_block_stride: i32, kv_head_stride: i32,
            stream: *mut c_void, cache_dtype: u32, k_scale: *mut f32, v_scale: *mut f32, sinks: *const f32,
        ) {
            unsafe {
                pa_v1($dt, concat!("src/cuda/pagedattention_v1_", $dt, ".cu"), $l1, out, query, key_cache, value_cache, alibi_slopes, num_kv_heads,
                      scale, softcapping, block_tables, context_lens, block_size, max_context_len, num_seqs, num_heads, head_size,
                      max_num_blocks_per_seq, q_stride, kv_block_stride, kv_head_stride, stream, cache_dtype, k_scale, v_scale, sinks)
            }
        }
        pub unsafe extern "C" fn $v2(
            out: *mut c_void, exp_sums: *mut f32, max_logits: *mut f32, tmp_out: *mut c_void, query: *mut c_void, key_cache: *mut c_void,
            value_cache: *mut c_void, alibi_slopes: *mut c_void, num_kv_heads: i32, scale: f32, softcapping: f32, block_tables: *mut u32,
            context_lens: *mut u32, block_size: i32, max_context_len: i32, num_seqs: i32, num_heads: i32, head_size: i32,
            max_num_blocks_per_seq: i32, q_stride: i32, kv_block_stride: i32, kv_head_stride: i32, stream: *mut c_void, cache_dtype: u32,
            k_scale: *mut f32, v_scale: *mut f32, sinks: *const f32,
        ) {
            unsafe {
                pa_v2($dt, concat!("src/cuda/pagedattention_v2_", $dt, ".cu"), $l2, out, exp_sums, max_logits, tmp_out, query, key_cache,
                      value_cache, alibi_slopes, num_kv_heads, scale, softcapping, block_tables, context_lens, block_size, max_context_len,
                      num_seqs, num_heads, head_size, max_num_blocks_per_seq, q_stride, kv_block_stride, kv_head_stride, stream, cache_dtype,
                      k_scale, v_scale, sinks)
            }
        }
    )*};
}

pa_launchers! {
    paged_attention_v1_f32 paged_attention_v2_f32 "f32" 29 33;
    paged_attention_v1_f16 paged_attention_v2_f16 "f16" 30 33;
    paged_attention_v1_bf16 paged_attention_v2_bf16 "bf16" 30 33;
}

// ================================================================================================
// flash_attn_sinks.cu (errors are printed, not fatal: FA_CUDA_CHECK)

fn fa_bc(head_dim: i32) -> bool {
    matches!(head_dim, 64 | 80 | 96 | 112 | 128 | 192 | 256)
}

fn fa_name(prefix: &str, dt: &str, head: i32) -> &'static str {
    static NAMES: Mutex<Option<HashMap<String, &'static str>>> = Mutex::new(None);
    let key = format!("{prefix}_{dt}_{head}");
    let mut g = NAMES.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    if let Some(n) = m.get(&key) {
        return n;
    }
    let s: &'static str = Box::leak(key.clone().into_boxed_str());
    m.insert(key, s);
    s
}

/// `__LINE__` of the `LAUNCH_KERNEL(D, BC)` / `LAUNCH_VARLEN_KERNEL(D, BC)` case for `head_dim`.
fn fa_line(first: u32, head_dim: i32) -> u32 {
    first + 3 * [64, 80, 96, 112, 128, 192, 256].iter().position(|&h| h == head_dim).unwrap() as u32
}

fn fa_check(err: i32, line: u32) {
    if err != 0 {
        eprintln!("CUDA error at src/cuda/flash_attn_sinks.cu:{line}: {}", cuda_error_string(err));
    }
}

unsafe fn fa(
    dt: &str, q: *const c_void, k: *const c_void, v: *const c_void, o: *mut c_void, sinks: *const f32, scale: f32, batch_size: i32, q_len: i32,
    kv_len: i32, num_heads: i32, num_kv_heads: i32, head_dim: i32, window_size: i32, stream: *mut c_void,
) {
    if !fa_bc(head_dim) {
        eprintln!("flash_attn_sinks: unsupported head_dim={head_dim}. Supported: 64, 80, 96, 112, 128, 192, 256");
        return;
    }
    let grid = (num_heads as u32, batch_size as u32, ((q_len + 7) / 8) as u32);
    let a = Args::new().p(q).p(k).p(v).p(o).p(sinks).f(scale).i(q_len).i(kv_len).i(num_heads).i(num_kv_heads).i(window_size);
    let e = unsafe { launch(stream, fa_name("fas", dt, head_dim), grid, (256, 1, 1), 0, a) };
    fa_check(e, fa_line(542, head_dim));
}

unsafe fn fav(
    dt: &str, q: *const c_void, k: *const c_void, v: *const c_void, o: *mut c_void, sinks: *const f32, cu_q: *const u32, cu_k: *const u32,
    scale: f32, batch_size: i32, max_q_len: i32, num_heads: i32, num_kv_heads: i32, head_dim: i32, window_size: i32, stream: *mut c_void,
) {
    if !fa_bc(head_dim) {
        eprintln!("flash_attn_sinks_varlen: unsupported head_dim={head_dim}. Supported: 64, 80, 96, 112, 128, 192, 256");
        return;
    }
    let grid = (num_heads as u32, batch_size as u32, ((max_q_len + 7) / 8) as u32);
    let a = Args::new().p(q).p(k).p(v).p(o).p(sinks).p(cu_q).p(cu_k).f(scale).i(max_q_len).i(num_heads).i(num_kv_heads).i(window_size);
    let e = unsafe { launch(stream, fa_name("fasv", dt, head_dim), grid, (256, 1, 1), 0, a) };
    fa_check(e, fa_line(634, head_dim));
}

macro_rules! fa_launchers {
    ($($f:ident $fv:ident $dt:literal;)*) => {$(
        pub unsafe extern "C" fn $f(
            q: *const c_void, k: *const c_void, v: *const c_void, o: *mut c_void, sinks: *const f32, scale: f32, batch_size: i32, q_len: i32,
            kv_len: i32, num_heads: i32, num_kv_heads: i32, head_dim: i32, window_size: i32, stream: *mut c_void,
        ) {
            unsafe { fa($dt, q, k, v, o, sinks, scale, batch_size, q_len, kv_len, num_heads, num_kv_heads, head_dim, window_size, stream) }
        }
        pub unsafe extern "C" fn $fv(
            q: *const c_void, k: *const c_void, v: *const c_void, o: *mut c_void, sinks: *const f32, cu_seqlens_q: *const u32,
            cu_seqlens_k: *const u32, scale: f32, batch_size: i32, max_q_len: i32, num_heads: i32, num_kv_heads: i32, head_dim: i32,
            window_size: i32, stream: *mut c_void,
        ) {
            unsafe {
                fav($dt, q, k, v, o, sinks, cu_seqlens_q, cu_seqlens_k, scale, batch_size, max_q_len, num_heads, num_kv_heads, head_dim,
                    window_size, stream)
            }
        }
    )*};
}

fa_launchers! {
    flash_attn_sinks_f16 flash_attn_sinks_varlen_f16 "f16";
    flash_attn_sinks_bf16 flash_attn_sinks_varlen_bf16 "bf16";
    flash_attn_sinks_f32 flash_attn_sinks_varlen_f32 "f32";
}
