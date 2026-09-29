//! Pure-Rust twins of the group-A extern "C" host launchers of mistralrs-quant
//! (libmistralrsquant.a): same names, same parameter lists and C ABI as the `extern "C"`
//! declarations in mistralrs-quant/src/**/ffi.rs, same grid / block / shared-memory selection and
//! host-side control flow, launching the oxide kernels on the caller's stream (a `CUstream` /
//! `cudaStream_t`; null = the legacy default stream) through the CUDA driver API.
//!
//! The oxide module is loaded once per CUDA context (the stream's context, or the current one for
//! the null stream; if no context is current, device 0's primary context, as the runtime API would)
//! and its functions are cached. The module image is the `#[cuda_module]` bundle embedded in the
//! running executable; if there is none, the PTX file named by `$MISTRALRS_QUANT_A_PTX`, else
//! `<crate>/mistralrs_quant_a.ptx`.
//!
//! Like the C launchers (`<<<>>>` without error checks), launch failures are dropped, except where
//! the C code checks them (ops.cu's `CUDA_CHECK`, which prints and `exit`s): those are reproduced.
//! Launchers that are `assert(false)` stubs in the host pass of nvcc (the f16/bf16 hqq dequantizers
//! and nonzero_{f16,bf16}: their `#if __CUDA_ARCH__ >= ...` is false on the host) call the same
//! `__assert_fail`; count_nonzero_{f16,bf16} are stubs returning 0.
#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
use cuda_core::sys;
use std::collections::HashMap;
use std::ffi::{CString, c_char, c_void};
use std::sync::{Mutex, OnceLock};

struct Module {
    module: sys::CUmodule,
    functions: HashMap<&'static str, sys::CUfunction>,
}
// CUmodule / CUfunction are driver handles, valid from any thread of the process.
unsafe impl Send for Module {}

static MODULES: OnceLock<Mutex<HashMap<usize, Module>>> = OnceLock::new();

fn module_image() -> Vec<u8> {
    if let Ok(bundles) = cuda_core::embedded::artifact_bundles_from_current_exe() {
        for b in &bundles {
            if b.name == "mistralrs_quant_a" {
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
    let path = std::env::var("MISTRALRS_QUANT_A_PTX")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/mistralrs_quant_a.ptx").to_string());
    let mut v = std::fs::read(&path).unwrap_or_else(|e| panic!("mistralrs-quant-a: no embedded module and cannot read {path}: {e}"));
    v.push(0);
    v
}

fn check(r: sys::CUresult, what: &str) {
    assert!(r == 0, "mistralrs-quant-a: {what} failed: {r:?}");
}

/// The context `stream` belongs to (the current context for the null stream).
unsafe fn stream_context(stream: sys::CUstream) -> sys::CUcontext {
    unsafe {
        check(sys::cuInit(0), "cuInit");
        let mut ctx: sys::CUcontext = std::ptr::null_mut();
        if !stream.is_null() {
            check(sys::cuStreamGetCtx(stream, &mut ctx), "cuStreamGetCtx");
            return ctx;
        }
        check(sys::cuCtxGetCurrent(&mut ctx), "cuCtxGetCurrent");
        if ctx.is_null() {
            let mut dev: sys::CUdevice = 0;
            check(sys::cuDeviceGet(&mut dev, 0), "cuDeviceGet");
            check(sys::cuDevicePrimaryCtxRetain(&mut ctx, dev), "cuDevicePrimaryCtxRetain");
            check(sys::cuCtxSetCurrent(ctx), "cuCtxSetCurrent");
        }
        ctx
    }
}

/// The oxide kernel `name` in `stream`'s context (loading the module on first use).
unsafe fn function(name: &'static str, stream: sys::CUstream) -> sys::CUfunction {
    unsafe {
        let ctx = stream_context(stream);
        let mut map = MODULES.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap();
        let m = map.entry(ctx as usize).or_insert_with(|| {
            let mut cur: sys::CUcontext = std::ptr::null_mut();
            check(sys::cuCtxGetCurrent(&mut cur), "cuCtxGetCurrent");
            if cur != ctx {
                check(sys::cuCtxPushCurrent_v2(ctx), "cuCtxPushCurrent");
            }
            let image = module_image();
            let mut module: sys::CUmodule = std::ptr::null_mut();
            check(sys::cuModuleLoadData(&mut module, image.as_ptr() as *const c_void), "cuModuleLoadData");
            if cur != ctx {
                let mut popped: sys::CUcontext = std::ptr::null_mut();
                check(sys::cuCtxPopCurrent_v2(&mut popped), "cuCtxPopCurrent");
            }
            Module { module, functions: HashMap::new() }
        });
        let module = m.module;
        *m.functions.entry(name).or_insert_with(|| {
            let mut f: sys::CUfunction = std::ptr::null_mut();
            let c = CString::new(name).unwrap();
            check(sys::cuModuleGetFunction(&mut f, module, c.as_ptr()), name);
            f
        })
    }
}

/// Launch `name`; returns the driver's result (the C `<<<>>>` drops it, so most callers do too).
unsafe fn launch(
    name: &'static str, grid: (u32, u32, u32), block: (u32, u32, u32), smem: u32, stream: *mut c_void,
    args: &mut [*mut c_void],
) -> sys::CUresult {
    unsafe {
        let s = stream as sys::CUstream;
        let f = function(name, s);
        sys::cuLaunchKernel(f, grid.0, grid.1, grid.2, block.0, block.1, block.2, smem, s, args.as_mut_ptr(), std::ptr::null_mut())
    }
}

/// Pointers to each argument, in order, for `cuLaunchKernel`.
macro_rules! args {
    ($($a:expr),* $(,)?) => { [$( &raw mut $a as *mut c_void ),*] };
}

unsafe extern "C" {
    fn __assert_fail(assertion: *const c_char, file: *const c_char, line: u32, function: *const c_char) -> !;
}

/// The C `assert(false)` of an nvcc host-pass stub (glibc prints and aborts).
fn assert_false(file: &'static [u8], line: u32, function: &'static [u8]) -> ! {
    unsafe { __assert_fail(c"false".as_ptr(), file.as_ptr() as *const c_char, line, function.as_ptr() as *const c_char) }
}

// ================================================================================================
// hqq_bitpack.cu

/// `(total_threads + 255) / 256` in size_t, truncated to int, as the grid's x.
fn blocks_256(total: usize) -> u32 {
    (total.wrapping_add(255) / 256) as i32 as u32
}

unsafe fn pack(name: &'static str, i: *const c_void, o: *mut c_void, n: usize, width: usize, per: usize, stream: *mut c_void) {
    let total = (n / per).wrapping_mul(width);
    let (mut i, mut o, mut n, mut width) = (i, o, n, width);
    unsafe { launch(name, (blocks_256(total), 1, 1), (256, 1, 1), 0, stream, &mut args!(i, o, n, width)) };
}

pub unsafe extern "C" fn launch_pack_1bit_kernel(d_input: *const u8, d_output: *mut u8, n: usize, width: usize, stream: *mut c_void) {
    unsafe { pack("pack_1bit_kernel", d_input as _, d_output as _, n, width, 8, stream) }
}
pub unsafe extern "C" fn launch_pack_2bit_kernel(d_input: *const u8, d_output: *mut u8, n: usize, width: usize, stream: *mut c_void) {
    unsafe { pack("pack_2bit_kernel", d_input as _, d_output as _, n, width, 4, stream) }
}
pub unsafe extern "C" fn launch_pack_3bit_kernel(d_input: *const u32, d_output: *mut i32, n: usize, width: usize, stream: *mut c_void) {
    unsafe { pack("pack_3bit_kernel", d_input as _, d_output as _, n, width, 10, stream) }
}
pub unsafe extern "C" fn launch_pack_4bit_kernel(d_input: *const u8, d_output: *mut u8, n: usize, width: usize, stream: *mut c_void) {
    unsafe { pack("pack_4bit_kernel", d_input as _, d_output as _, n, width, 2, stream) }
}
pub unsafe extern "C" fn launch_pack_8bit_kernel(d_input: *const u8, d_output: *mut u8, n: usize, stream: *mut c_void) {
    let (mut i, mut o, mut n) = (d_input, d_output, n);
    // `(num_elements + 255) / 256` in size_t, truncated to int.
    unsafe { launch("pack_8bit_kernel", (blocks_256(n), 1, 1), (256, 1, 1), 0, stream, &mut args!(i, o, n)) };
}

// ================================================================================================
// hqq.cu (no stream parameter: the legacy default stream)

unsafe fn hqq(name: &'static str, wq: *const c_void, scale: *const c_void, zero: *const c_void, out: *const c_void, h: i32, w: i32) {
    // cdiv(unsigned h*w, 256)
    let blocks = (h.wrapping_mul(w) as u32).wrapping_add(255) / 256;
    let (mut wq, mut s, mut z, mut o, mut h, mut w) = (wq, scale, zero, out, h, w);
    unsafe { launch(name, (blocks, 1, 1), (256, 1, 1), 0, std::ptr::null_mut(), &mut args!(wq, s, z, o, h, w)) };
}

const HQQ_FILE: &[u8] = b"kernels/hqq/hqq.cu\0";

macro_rules! hqq_launchers {
    ($($f32:ident $f16:ident $bf16:ident $wq:ty, $kf32:literal, $l16:literal, $s16:literal, $l_bf:literal, $s_bf:literal;)*) => {$(
        pub unsafe extern "C" fn $f32(wq: *const $wq, scale: *const f32, zero: *const f32, out: *const f32, h: i32, w: i32) {
            unsafe { hqq($kf32, wq as _, scale as _, zero as _, out as _, h, w) }
        }
        pub unsafe extern "C" fn $f16(_wq: *const $wq, _scale: *const u16, _zero: *const u16, _out: *const u16, _h: i32, _w: i32) {
            assert_false(HQQ_FILE, $l16, $s16)
        }
        pub unsafe extern "C" fn $bf16(_wq: *const $wq, _scale: *const u16, _zero: *const u16, _out: *const u16, _h: i32, _w: i32) {
            assert_false(HQQ_FILE, $l_bf, $s_bf)
        }
    )*};
}

hqq_launchers! {
    dequantize_8bit_u8_kernel_f32 dequantize_8bit_u8_kernel_f16 dequantize_8bit_u8_kernel_bf16 u8, "hqq_8bit_f32",
        57, b"void dequantize_8bit_u8_kernel_f16(unsigned char*, uint16_t*, uint16_t*, uint16_t*, int, int)\0",
        75, b"void dequantize_8bit_u8_kernel_bf16(unsigned char*, uint16_t*, uint16_t*, uint16_t*, int, int)\0";
    dequantize_4bit_u8_kernel_f32 dequantize_4bit_u8_kernel_f16 dequantize_4bit_u8_kernel_bf16 u8, "hqq_4bit_f32",
        132, b"void dequantize_4bit_u8_kernel_f16(unsigned char*, uint16_t*, uint16_t*, uint16_t*, int, int)\0",
        150, b"void dequantize_4bit_u8_kernel_bf16(unsigned char*, uint16_t*, uint16_t*, uint16_t*, int, int)\0";
    dequantize_2bit_u8_kernel_f32 dequantize_2bit_u8_kernel_f16 dequantize_2bit_u8_kernel_bf16 u8, "hqq_2bit_f32",
        208, b"void dequantize_2bit_u8_kernel_f16(unsigned char*, uint16_t*, uint16_t*, uint16_t*, int, int)\0",
        226, b"void dequantize_2bit_u8_kernel_bf16(unsigned char*, uint16_t*, uint16_t*, uint16_t*, int, int)\0";
    dequantize_1bit_u8_kernel_f32 dequantize_1bit_u8_kernel_f16 dequantize_1bit_u8_kernel_bf16 u8, "hqq_1bit_f32",
        323, b"void dequantize_1bit_u8_kernel_f16(unsigned char*, uint16_t*, uint16_t*, uint16_t*, int, int)\0",
        341, b"void dequantize_1bit_u8_kernel_bf16(unsigned char*, uint16_t*, uint16_t*, uint16_t*, int, int)\0";
    dequantize_3bit_32_kernel_f32 dequantize_3bit_32_kernel_f16 dequantize_3bit_32_kernel_bf16 i32, "hqq_3bit_f32",
        449, b"void dequantize_3bit_32_kernel_f16(unsigned char*, uint16_t*, uint16_t*, uint16_t*, int, int)\0",
        467, b"void dequantize_3bit_32_kernel_bf16(unsigned char*, uint16_t*, uint16_t*, uint16_t*, int, int)\0";
}

// ================================================================================================
// bitsandbytes dequant.cu: grid (n + tile - 1) / tile (tile 512 for int8, 1024 for fp4/nf4), block
// 64; the 4-bit kernels get blocksize / 2.

unsafe fn bnb(name: &'static str, dt: u32, code: *const f32, a: *const u8, absmax: *const f32, out: *mut c_void, blocksize: i32, n: i32, stream: *mut c_void) {
    let tile: i32 = if dt > 0 { 1024 } else { 512 };
    let grid = (n.wrapping_add(tile - 1) / tile) as u32;
    let (mut c, mut a, mut am, mut o, mut bs, mut n) = (code, a, absmax, out, if dt > 0 { blocksize / 2 } else { blocksize }, n);
    unsafe { launch(name, (grid, 1, 1), (64, 1, 1), 0, stream, &mut args!(c, a, am, o, bs, n)) };
}

macro_rules! bnb_launchers {
    ($($name:ident $k:literal $dt:literal $t:ty;)*) => {$(
        pub unsafe extern "C" fn $name(code: *const f32, a: *const u8, absmax: *const f32, out: *mut $t, blocksize: i32, n: i32, stream: *mut c_void) {
            unsafe { bnb($k, $dt, code, a, absmax, out as _, blocksize, n, stream) }
        }
    )*};
}
bnb_launchers! {
    dequantize_blockwise_f32_int8 "bnb_f32_int8" 0 f32;
    dequantize_blockwise_f32_fp4 "bnb_f32_fp4" 1 f32;
    dequantize_blockwise_f32_nf4 "bnb_f32_nf4" 2 f32;
    dequantize_blockwise_f16_int8 "bnb_f16_int8" 0 u16;
    dequantize_blockwise_f16_fp4 "bnb_f16_fp4" 1 u16;
    dequantize_blockwise_f16_nf4 "bnb_f16_nf4" 2 u16;
    dequantize_blockwise_bf16_int8 "bnb_bf16_int8" 0 u16;
    dequantize_blockwise_bf16_fp4 "bnb_bf16_fp4" 1 u16;
    dequantize_blockwise_bf16_nf4 "bnb_bf16_nf4" 2 u16;
}

// ================================================================================================
// moe/gelu_tanh_and_mul.cu

unsafe fn act_and_mul(name: &'static str, out: *mut c_void, input: *const c_void, num_tokens: i32, d: i32, stream: *mut c_void) {
    if num_tokens == 0 {
        return;
    }
    let block = if d < 1024 { d } else { 1024 } as u32;
    let (mut o, mut i, mut d) = (out, input, d);
    unsafe { launch(name, (num_tokens as u32, 1, 1), (block, 1, 1), 0, stream, &mut args!(o, i, d)) };
}
pub unsafe extern "C" fn launch_gelu_tanh_and_mul_bf16(out: *mut c_void, input: *const c_void, num_tokens: i32, d: i32, stream: *mut c_void) {
    unsafe { act_and_mul("gelu_tanh_and_mul_bf16", out, input, num_tokens, d, stream) }
}
pub unsafe extern "C" fn launch_silu_and_mul_bf16(out: *mut c_void, input: *const c_void, num_tokens: i32, d: i32, stream: *mut c_void) {
    unsafe { act_and_mul("silu_and_mul_bf16", out, input, num_tokens, d, stream) }
}
pub unsafe extern "C" fn launch_gelu_tanh_and_mul_f16(out: *mut c_void, input: *const c_void, num_tokens: i32, d: i32, stream: *mut c_void) {
    unsafe { act_and_mul("gelu_tanh_and_mul_f16", out, input, num_tokens, d, stream) }
}
pub unsafe extern "C" fn launch_moe_sum_bf16(out: *mut c_void, input: *const c_void, num_tokens: i32, hidden: i32, topk: i32, stream: *mut c_void) {
    if num_tokens == 0 || hidden == 0 || topk == 0 {
        return;
    }
    let grid = (num_tokens as u32, (hidden.wrapping_add(255) / 256) as u32, 1);
    let (mut o, mut i, mut n, mut h, mut k) = (out, input, num_tokens, hidden, topk);
    unsafe { launch("moe_sum_bf16", grid, (256, 1, 1), 0, stream, &mut args!(o, i, n, h, k)) };
}

// ================================================================================================
// rotary.cu: grid (num_tokens), block min(num_heads * rot_dim, 512); dtype 0 f16, 1 bf16, 2 f32
// (anything else launches nothing).

const ROTARY: [[&str; 3]; 2] = [
    ["rotary_f16_gptj", "rotary_bf16_gptj", "rotary_f32_gptj"],
    ["rotary_f16_neox", "rotary_bf16_neox", "rotary_f32_neox"],
];
const ROTARY_POS: [[&str; 3]; 2] = [
    ["rotary_pos_f16_gptj", "rotary_pos_bf16_gptj", "rotary_pos_f32_gptj"],
    ["rotary_pos_f16_neox", "rotary_pos_bf16_neox", "rotary_pos_f32_neox"],
];

pub unsafe extern "C" fn rotary_embedding(
    query: *const c_void, key: *const c_void, cos_cache: *const c_void, sin_cache: *const c_void, is_neox: i32,
    head_size: i32, num_tokens: i64, rot_dim: i32, num_heads: i32, num_kv_heads: i32, query_stride: i64, key_stride: i64,
    dtype: u32, stream: i64,
) {
    if dtype > 2 {
        return;
    }
    let name = ROTARY[(is_neox != 0) as usize][dtype as usize];
    let block = num_heads.wrapping_mul(rot_dim).min(512) as u32;
    let (mut q, mut k, mut c, mut s) = (query, key, cos_cache, sin_cache);
    let (mut rd, mut qs, mut ks, mut nh, mut nkv, mut hs) = (rot_dim, query_stride, key_stride, num_heads, num_kv_heads, head_size);
    unsafe {
        launch(name, (num_tokens as u32, 1, 1), (block, 1, 1), 0, stream as *mut c_void, &mut args!(q, k, c, s, rd, qs, ks, nh, nkv, hs))
    };
}

pub unsafe extern "C" fn rotary_embedding_positions(
    query: *const c_void, key: *const c_void, cos_cache: *const c_void, sin_cache: *const c_void, positions: *const c_void,
    is_neox: i32, head_size: i32, num_tokens: i64, rot_dim: i32, seq_len: i32, num_heads: i32, num_kv_heads: i32,
    query_stride: i64, key_stride: i64, dtype: u32, stream: i64,
) {
    if dtype > 2 {
        return;
    }
    let name = ROTARY_POS[(is_neox != 0) as usize][dtype as usize];
    let block = num_heads.wrapping_mul(rot_dim).min(512) as u32;
    let (mut q, mut k, mut c, mut s, mut p) = (query, key, cos_cache, sin_cache, positions);
    let (mut rd, mut sl, mut qs, mut ks, mut nh, mut nkv, mut hs) = (rot_dim, seq_len, query_stride, key_stride, num_heads, num_kv_heads, head_size);
    unsafe {
        launch(name, (num_tokens as u32, 1, 1), (block, 1, 1), 0, stream as *mut c_void, &mut args!(q, k, c, s, p, rd, sl, qs, ks, nh, nkv, hs))
    };
}

// ================================================================================================
// moe/moe_align.cu

pub unsafe extern "C" fn launch_moe_align(
    topk_ids: *const i32, sorted_token_ids: *mut i32, expert_ids: *mut i32, num_tokens_post_pad: *mut i32, cumsum: *mut i32,
    num_experts: i32, block_size: i32, numel: i32, max_num_tokens_padded: i32, stream: *mut c_void,
) {
    let padded_num_experts = (num_experts.wrapping_add(31) / 32).wrapping_mul(32);
    let experts_per_warp = 32i32;
    let num_warps = padded_num_experts.wrapping_add(experts_per_warp - 1) / experts_per_warp;
    let smem = (num_warps as i64 as usize).wrapping_mul(32 * 4) as u32;
    let (mut ti, mut st, mut ei, mut nt, mut cs) = (topk_ids, sorted_token_ids, expert_ids, num_tokens_post_pad, cumsum);
    let (mut ne, mut pne, mut epw, mut bs, mut nel, mut mx) =
        (num_experts, padded_num_experts, experts_per_warp, block_size, numel as i64 as usize, max_num_tokens_padded);
    unsafe {
        launch("moe_align_block_size_kernel", (2, 1, 1), (1024, 1, 1), smem, stream, &mut args!(ti, st, ei, nt, ne, pne, epw, bs, nel, cs, mx));
    }
    let n_blocks = numel.wrapping_add(255) / 256;
    let actual = n_blocks.min(65535);
    let (mut ti, mut st, mut cs, mut nel, mut ne) = (topk_ids, sorted_token_ids, cumsum, numel as i64 as usize, num_experts);
    unsafe {
        launch("count_and_sort_expert_tokens_kernel", (1, actual as u32, 1), (256, 1, 1), 0, stream, &mut args!(ti, st, cs, nel, ne));
    }
}

pub unsafe extern "C" fn launch_hunyuan_moe_capacity_mask(
    ids: *const c_void, weights: *const c_void, masked_weights: *mut c_void, n_tokens: i32, n_experts: i32, top_k: i32,
    expert_capacity: i32, stream: *mut c_void,
) {
    if n_tokens <= 0 || n_experts <= 0 || top_k <= 0 || expert_capacity <= 0 {
        return;
    }
    let total = n_tokens.wrapping_mul(top_k);
    let grid = total.wrapping_add(255) / 256;
    let (mut d, mut v, mut n) = (masked_weights, 0.0f32, total);
    unsafe { launch("fill_f32_kernel", (grid as u32, 1, 1), (256, 1, 1), 0, stream, &mut args!(d, v, n)) };
    let (mut i, mut w, mut m, mut nt, mut ne, mut tk, mut cap) = (ids, weights, masked_weights, n_tokens, n_experts, top_k, expert_capacity);
    unsafe {
        launch("hunyuan_moe_capacity_mask_kernel", (n_experts as u32, top_k as u32, 1), (1, 1, 1), 0, stream, &mut args!(i, w, m, nt, ne, tk, cap))
    };
}

// ================================================================================================
// cutlass_moe/moe_data.cu

pub unsafe extern "C" fn launch_cutlass_moe_problem_sizes(
    topk_ids: *const i32, problem_sizes1: *mut i32, problem_sizes2: *mut i32, atomic_buffer: *mut i32, num_experts: i32,
    topk_length: i32, n: i32, k: i32, is_gated: bool, stream: *mut c_void,
) {
    unsafe {
        // cudaMemsetAsync(atomic_buffer, 0, num_experts * sizeof(int32_t), stream)
        let _ = sys::cuMemsetD8Async(atomic_buffer as sys::CUdeviceptr, 0, (num_experts as i64 as usize).wrapping_mul(4), stream as sys::CUstream);
        let (mut t, mut p1, mut p2, mut ab, mut tl, mut n, mut k, mut g) = (topk_ids, problem_sizes1, problem_sizes2, atomic_buffer, topk_length, n, k, is_gated as u8);
        launch("compute_problem_sizes_kernel", (num_experts as u32, 1, 1), (512, 1, 1), 0, stream, &mut args!(t, p1, p2, ab, tl, n, k, g));
    }
}

pub unsafe extern "C" fn launch_cutlass_moe_expert_offsets(
    problem_sizes1: *const i32, expert_offsets: *mut i32, atomic_buffer: *mut i32, num_experts: i32, stream: *mut c_void,
) {
    let (mut p, mut eo, mut ab, mut ne) = (problem_sizes1, expert_offsets, atomic_buffer, num_experts);
    unsafe { launch("compute_expert_offsets_kernel", (1, 1, 1), (1, 1, 1), 0, stream, &mut args!(p, eo, ab, ne)) };
}

pub unsafe extern "C" fn launch_cutlass_moe_arg_sorts(
    topk_ids: *const i32, input_permutation: *mut i32, output_permutation: *mut i32, atomic_buffer: *mut i32, num_experts: i32,
    topk_length: i32, topk: i32, stream: *mut c_void,
) {
    let (mut t, mut ip, mut op, mut ab, mut tl, mut tk) = (topk_ids, input_permutation, output_permutation, atomic_buffer, topk_length, topk);
    unsafe { launch("compute_arg_sorts_kernel", (num_experts as u32, 1, 1), (512, 1, 1), 0, stream, &mut args!(t, ip, op, ab, tl, tk)) };
}

pub unsafe extern "C" fn launch_cutlass_moe_gather_rows_bf16(dst: *mut c_void, src: *const c_void, map: *const i32, num_rows: i32, k: i32, stream: *mut c_void) {
    let (mut d, mut s, mut m, mut nr, mut k) = (dst, src, map, num_rows, k);
    unsafe { launch("gather_rows_bf16_kernel", (num_rows as u32, 1, 1), (256, 1, 1), 0, stream, &mut args!(d, s, m, nr, k)) };
}

pub unsafe extern "C" fn launch_cutlass_moe_gather_weighted_bf16(
    out: *mut c_void, input: *const c_void, out_perm: *const i32, weights: *const f32, num_rows: i32, k: i32, stream: *mut c_void,
) {
    let (mut o, mut i, mut p, mut w, mut nr, mut k) = (out, input, out_perm, weights, num_rows, k);
    unsafe { launch("gather_weighted_bf16_kernel", (num_rows as u32, 1, 1), (256, 1, 1), 0, stream, &mut args!(o, i, p, w, nr, k)) };
}

pub unsafe extern "C" fn launch_cutlass_moe_group_starts_bf16(
    expert_offsets: *const i32, a_base: *const c_void, b_base: *const c_void, d_base: *mut c_void, a_ptrs: *const *const c_void,
    b_ptrs: *const *const c_void, d_ptrs: *mut *mut c_void, lda: *mut i64, ldb: *mut i64, ldd: *mut i64, num_experts: i32, n: i64,
    k: i64, stream: *mut c_void,
) {
    let (mut eo, mut a, mut b, mut d, mut ap, mut bp, mut dp) = (expert_offsets, a_base, b_base, d_base, a_ptrs, b_ptrs, d_ptrs);
    let (mut la, mut lb, mut ld, mut n, mut k) = (lda, ldb, ldd, n, k);
    unsafe {
        launch("get_group_starts_bf16_kernel", (1, 1, 1), (num_experts as u32, 1, 1), 0, stream, &mut args!(eo, a, b, d, ap, bp, dp, la, lb, ld, n, k))
    };
}

// ================================================================================================
// ops/ops.cu

const OPS_FILE: &str = "kernels/ops/ops.cu";

/// ops.cu's `CUDA_CHECK(cudaGetLastError())` after a launch: on failure print and `exit(err)`.
/// A bad launch configuration (e.g. a zero grid) is CUDA_ERROR_INVALID_VALUE from the driver and
/// cudaErrorInvalidValue (1, "invalid argument") from the CUDA 13 runtime (measured: not 9); the
/// codes and strings used here are the driver's, which agree for these errors.
fn cuda_check(r: sys::CUresult, line: u32) {
    if r == 0 {
        return;
    }
    let mut p: *const c_char = std::ptr::null();
    unsafe { sys::cuGetErrorString(r, &mut p) };
    let msg = if p.is_null() { "unknown error".to_string() } else { unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy().into_owned() };
    eprintln!("CUDA error at {OPS_FILE}:{line}: {msg}");
    std::process::exit(r as i32);
}

/// ops.cu `next_power_of_2` (int result compared against an unsigned count).
fn next_power_of_2(num: u32) -> i32 {
    let mut result: i32 = 1;
    while (result as u32) < num {
        result = result.wrapping_shl(1);
    }
    result
}

/// The `bitwise_*` / `leftshift` host helpers: nthreads = min(next_power_of_2(N), 1024), legacy stream.
fn small_grid(n: i32) -> (u32, u32) {
    let mut nthreads = next_power_of_2(n as u32);
    if nthreads > 1024 {
        nthreads = 1024;
    }
    let nblocks = n.wrapping_add(nthreads - 1) / nthreads;
    (nblocks as u32, nthreads as u32)
}

macro_rules! bitwise_launchers {
    ($($f:ident $k:literal $t:ty, $line:literal;)*) => {$(
        pub unsafe extern "C" fn $f(d_in1: *const c_void, d_in2: *const c_void, d_out: *mut c_void, n: u32) {
            let (g, b) = small_grid(n as i32);
            let (mut a, mut bb, mut o, mut n) = (d_in1, d_in2, d_out, n);
            cuda_check(unsafe { launch($k, (g, 1, 1), (b, 1, 1), 0, std::ptr::null_mut(), &mut args!(a, bb, o, n)) }, $line);
        }
    )*};
}
bitwise_launchers! {
    bitwise_and_u8 "bitwise_and_u8k" u8, 238; bitwise_or_u8 "bitwise_or_u8k" u8, 249; bitwise_xor_u8 "bitwise_xor_u8k" u8, 260;
    bitwise_and_u32 "bitwise_and_u32k" u32, 238; bitwise_or_u32 "bitwise_or_u32k" u32, 249; bitwise_xor_u32 "bitwise_xor_u32k" u32, 260;
    bitwise_and_i64 "bitwise_and_i64k" i64, 238; bitwise_or_i64 "bitwise_or_i64k" i64, 249; bitwise_xor_i64 "bitwise_xor_i64k" i64, 260;
    bitwise_and_i32 "bitwise_and_i32k" i32, 238; bitwise_or_i32 "bitwise_or_i32k" i32, 249; bitwise_xor_i32 "bitwise_xor_i32k" i32, 260;
}

macro_rules! leftshift_launchers {
    ($($f:ident $k:literal;)*) => {$(
        pub unsafe extern "C" fn $f(d_in1: *const c_void, d_out: *mut c_void, n: u32, k: i32) {
            let (g, b) = small_grid(n as i32);
            let (mut a, mut o, mut n, mut k) = (d_in1, d_out, n, k);
            cuda_check(unsafe { launch($k, (g, 1, 1), (b, 1, 1), 0, std::ptr::null_mut(), &mut args!(a, o, n, k)) }, 302);
        }
    )*};
}
leftshift_launchers! { leftshift_u8 "leftshift_u8k"; leftshift_u32 "leftshift_u32k"; leftshift_i64 "leftshift_i64k"; leftshift_i32 "leftshift_i32k"; }

/// gptoss_swiglu_*: the vec4 kernel when N % 4 == 0 (N4 = N / 4 as int), else the scalar one.
unsafe fn swiglu_launch(k4: &'static str, k1: &'static str, gate: *const c_void, up: *const c_void, out: *mut c_void, n: u32, alpha: f32, limit: f32, stream: *mut c_void, line: u32) {
    let (name, cnt) = if n % 4 == 0 { (k4, (n / 4) as i32) } else { (k1, n as i32) };
    let nblocks = cnt.wrapping_add(255) / 256;
    let (mut g, mut u, mut o, mut c, mut a, mut l) = (gate, up, out, cnt as u32, alpha, limit);
    cuda_check(unsafe { launch(name, (nblocks as u32, 1, 1), (256, 1, 1), 0, stream, &mut args!(g, u, o, c, a, l)) }, line);
}
pub unsafe extern "C" fn gptoss_swiglu_f16(gate: *const c_void, up: *const c_void, output: *mut c_void, n: u32, alpha: f32, limit: f32, stream: *mut c_void) {
    unsafe { swiglu_launch("gptoss_swiglu4_f16k", "gptoss_swiglu_f16k", gate, up, output, n, alpha, limit, stream, 426) }
}
pub unsafe extern "C" fn gptoss_swiglu_bf16(gate: *const c_void, up: *const c_void, output: *mut c_void, n: u32, alpha: f32, limit: f32, stream: *mut c_void) {
    unsafe { swiglu_launch("gptoss_swiglu4_bf16k", "gptoss_swiglu_bf16k", gate, up, output, n, alpha, limit, stream, 449) }
}
pub unsafe extern "C" fn gptoss_swiglu_f32(gate: *const c_void, up: *const c_void, output: *mut c_void, n: u32, alpha: f32, limit: f32, stream: *mut c_void) {
    unsafe { swiglu_launch("gptoss_swiglu4_f32k", "gptoss_swiglu_f32k", gate, up, output, n, alpha, limit, stream, 469) }
}

unsafe fn swiglu_il_launch(name: &'static str, gate_up: *const c_void, out: *mut c_void, n: u32, isz: u32, alpha: f32, limit: f32, stream: *mut c_void, line: u32) {
    let total = n.wrapping_mul(isz);
    let nblocks = (total.wrapping_add(255) / 256) as i32;
    let (mut g, mut o, mut n, mut i, mut a, mut l) = (gate_up, out, n, isz, alpha, limit);
    cuda_check(unsafe { launch(name, (nblocks as u32, 1, 1), (256, 1, 1), 0, stream, &mut args!(g, o, n, i, a, l)) }, line);
}
pub unsafe extern "C" fn gptoss_swiglu_interleaved_f16(gate_up: *const c_void, output: *mut c_void, n: u32, isz: u32, alpha: f32, limit: f32, stream: *mut c_void) {
    unsafe { swiglu_il_launch("gptoss_swiglu_il_f16k", gate_up, output, n, isz, alpha, limit, stream, 524) }
}
pub unsafe extern "C" fn gptoss_swiglu_interleaved_bf16(gate_up: *const c_void, output: *mut c_void, n: u32, isz: u32, alpha: f32, limit: f32, stream: *mut c_void) {
    unsafe { swiglu_il_launch("gptoss_swiglu_il_bf16k", gate_up, output, n, isz, alpha, limit, stream, 535) }
}
pub unsafe extern "C" fn gptoss_swiglu_interleaved_f32(gate_up: *const c_void, output: *mut c_void, n: u32, isz: u32, alpha: f32, limit: f32, stream: *mut c_void) {
    unsafe { swiglu_il_launch("gptoss_swiglu_il_f32k", gate_up, output, n, isz, alpha, limit, stream, 548) }
}

unsafe fn softmax_sinks_launch(
    name: &'static str, logits: *const c_void, sinks: *const c_void, mask: *const c_void, output: *mut c_void, batch: i32, heads: i32,
    q_len: i32, k_len: i32, scale: f32, stream: *mut c_void, line: u32,
) {
    let total = batch.wrapping_mul(heads).wrapping_mul(q_len);
    let block = if k_len <= 64 { 64 } else if k_len <= 128 { 128 } else if k_len <= 256 { 256 } else if k_len <= 512 { 512 } else { 1024 };
    let (mut l, mut s, mut m, mut o, mut b, mut h, mut q, mut k, mut sc) = (logits, sinks, mask, output, batch, heads, q_len, k_len, scale);
    cuda_check(unsafe { launch(name, (total as u32, 1, 1), (block, 1, 1), 0, stream, &mut args!(l, s, m, o, b, h, q, k, sc)) }, line);
}
pub unsafe extern "C" fn softmax_with_sinks_f16(l: *const c_void, s: *const c_void, m: *const c_void, o: *mut c_void, b: i32, h: i32, q: i32, k: i32, scale: f32, stream: *mut c_void) {
    unsafe { softmax_sinks_launch("softmax_with_sinks_f16k", l, s, m, o, b, h, q, k, scale, stream, 738) }
}
pub unsafe extern "C" fn softmax_with_sinks_bf16(l: *const c_void, s: *const c_void, m: *const c_void, o: *mut c_void, b: i32, h: i32, q: i32, k: i32, scale: f32, stream: *mut c_void) {
    unsafe { softmax_sinks_launch("softmax_with_sinks_bf16k", l, s, m, o, b, h, q, k, scale, stream, 763) }
}
pub unsafe extern "C" fn softmax_with_sinks_f32(l: *const c_void, s: *const c_void, m: *const c_void, o: *mut c_void, b: i32, h: i32, q: i32, k: i32, scale: f32, stream: *mut c_void) {
    unsafe { softmax_sinks_launch("softmax_with_sinks_f32k", l, s, m, o, b, h, q, k, scale, stream, 787) }
}

unsafe fn fused_glu_launch(k4: &'static str, k1: &'static str, a: *const c_void, b: *const c_void, out: *mut c_void, n: u32, act: i32, stream: *mut c_void, line: u32) {
    let (name, cnt) = if n % 4 == 0 { (k4, (n / 4) as i32) } else { (k1, n as i32) };
    let nblocks = cnt.wrapping_add(255) / 256;
    let (mut a, mut b, mut o, mut c, mut act) = (a, b, out, cnt as u32, act);
    cuda_check(unsafe { launch(name, (nblocks as u32, 1, 1), (256, 1, 1), 0, stream, &mut args!(a, b, o, c, act)) }, line);
}
pub unsafe extern "C" fn fused_glu_f16(a: *const c_void, b: *const c_void, output: *mut c_void, n: u32, activation: i32, stream: *mut c_void) {
    unsafe { fused_glu_launch("fused_glu4_f16k", "fused_glu_f16k", a, b, output, n, activation, stream, 909) }
}
pub unsafe extern "C" fn fused_glu_bf16(a: *const c_void, b: *const c_void, output: *mut c_void, n: u32, activation: i32, stream: *mut c_void) {
    unsafe { fused_glu_launch("fused_glu4_bf16k", "fused_glu_bf16k", a, b, output, n, activation, stream, 929) }
}
pub unsafe extern "C" fn fused_glu_f32(a: *const c_void, b: *const c_void, output: *mut c_void, n: u32, activation: i32, stream: *mut c_void) {
    unsafe { fused_glu_launch("fused_glu4_f32k", "fused_glu_f32k", a, b, output, n, activation, stream, 946) }
}

unsafe fn softcap_launch(name: &'static str, input: *const c_void, out: *mut c_void, n: u32, cap: f32, stream: *mut c_void, line: u32) {
    let nblocks = (n as i32).wrapping_add(255) / 256;
    let (mut i, mut o, mut n, mut c) = (input, out, n, cap);
    cuda_check(unsafe { launch(name, (nblocks as u32, 1, 1), (256, 1, 1), 0, stream, &mut args!(i, o, n, c)) }, line);
}
pub unsafe extern "C" fn softcap_f32(input: *const c_void, output: *mut c_void, n: u32, cap: f32, stream: *mut c_void) {
    unsafe { softcap_launch("softcap_f32k", input, output, n, cap, stream, 986) }
}
pub unsafe extern "C" fn softcap_f16_to_f32(input: *const c_void, output: *mut c_void, n: u32, cap: f32, stream: *mut c_void) {
    unsafe { softcap_launch("softcap_f16k", input, output, n, cap, stream, 996) }
}
pub unsafe extern "C" fn softcap_bf16_to_f32(input: *const c_void, output: *mut c_void, n: u32, cap: f32, stream: *mut c_void) {
    unsafe { softcap_launch("softcap_bf16k", input, output, n, cap, stream, 1006) }
}

/// ops.cu `CUDA_CHECK(call)` for a driver call standing in for the runtime call at `line`.
fn ops_check(r: sys::CUresult, line: u32) {
    cuda_check(r, line)
}

unsafe fn alloc_async(bytes: usize, stream: sys::CUstream, line: u32) -> sys::CUdeviceptr {
    let mut p: sys::CUdeviceptr = 0;
    if bytes > 0 {
        ops_check(unsafe { sys::cuMemAllocAsync(&mut p, bytes, stream) }, line);
    }
    p
}
unsafe fn free_async(p: sys::CUdeviceptr, stream: sys::CUstream, line: u32) {
    if p != 0 {
        ops_check(unsafe { sys::cuMemFreeAsync(p, stream) }, line);
    }
}

/// count_nonzero<T>: an atomic count kernel into a stream-ordered u32, copied back to the host.
unsafe fn count_nonzero(name: &'static str, d_in: *const c_void, n: u32, stream: *mut c_void) -> u32 {
    let s = stream as sys::CUstream;
    unsafe {
        let d = alloc_async(4, s, 66);
        ops_check(sys::cuMemsetD8Async(d, 0, 4, s), 68);
        if n > 0 {
            let grid = n.div_ceil(256).min(1024);
            let (mut i, mut nn, mut c) = (d_in, n, d);
            ops_check(launch(name, (grid, 1, 1), (256, 1, 1), 0, stream, &mut args!(i, nn, c)), 72);
        }
        let mut result: u32 = 0;
        ops_check(sys::cuMemcpyDtoHAsync_v2(&mut result as *mut u32 as *mut c_void, d, 4, s), 74);
        ops_check(sys::cuStreamSynchronize(s), 74);
        free_async(d, s, 76);
        result
    }
}

/// nonzero<T>: tile counts, a scan, an ordered scatter of the flat indices (cub::DeviceSelect::Flagged
/// in the reference), then ops.cu's own transform_indices launch and CUDA_CHECK.
unsafe fn nonzero(tag: usize, d_in: *const c_void, n: u32, num_nonzero: u32, dims: *const u32, num_dims: u32, d_out: *mut u32, stream: *mut c_void) {
    const TC: [&str; 7] = ["tile_count_nz_f32", "tile_count_nz_f64", "tile_count_nz_u8", "tile_count_nz_u32", "tile_count_nz_i16", "tile_count_nz_i32", "tile_count_nz_i64"];
    const TS: [&str; 7] = ["tile_select_nz_f32", "tile_select_nz_f64", "tile_select_nz_u8", "tile_select_nz_u32", "tile_select_nz_i16", "tile_select_nz_i32", "tile_select_nz_i64"];
    let s = stream as sys::CUstream;
    unsafe {
        let out_temp = alloc_async((num_nonzero as usize) * 4, s, 140);
        let ntiles = n.div_ceil(2048);
        let counts = alloc_async(ntiles as usize * 8, s, 149);
        if ntiles > 0 {
            let offsets = counts + ntiles as u64 * 4;
            let (mut i, mut nn, mut c) = (d_in, n, counts);
            ops_check(launch(TC[tag], (ntiles, 1, 1), (256, 1, 1), 0, stream, &mut args!(i, nn, c)), 150);
            let (mut c, mut o, mut nt) = (counts, offsets, ntiles);
            ops_check(launch("nz_scan_tiles", (1, 1, 1), (1024, 1, 1), 0, stream, &mut args!(c, o, nt)), 150);
            let (mut i, mut nn, mut o, mut t, mut cap) = (d_in, n, offsets, out_temp, num_nonzero);
            ops_check(launch(TS[tag], (ntiles, 1, 1), (256, 1, 1), 0, stream, &mut args!(i, nn, o, t, cap)), 150);
        }
        let mut nthreads = next_power_of_2(num_nonzero);
        if nthreads > 1024 {
            nthreads = 1024;
        }
        let nblocks = (num_nonzero as i32).wrapping_add(nthreads - 1) / nthreads;
        let (mut t, mut nz, mut d, mut nd, mut o) = (out_temp, num_nonzero, dims, num_dims, d_out);
        cuda_check(launch("transform_indices", (nblocks as u32, 1, 1), (nthreads as u32, 1, 1), 0, stream, &mut args!(t, nz, d, nd, o)), 160);
        free_async(out_temp, s, 162);
        free_async(counts, s, 163);
    }
}

macro_rules! nonzero_launchers {
    ($($cf:ident $nf:ident $ck:literal $tag:literal;)*) => {$(
        pub unsafe extern "C" fn $cf(d_in: *const c_void, n: u32, stream: *mut c_void) -> u32 {
            unsafe { count_nonzero($ck, d_in, n, stream) }
        }
        pub unsafe extern "C" fn $nf(d_in: *const c_void, n: u32, num_nonzero: u32, dims: *const c_void, num_dims: u32, d_out: *mut c_void, stream: *mut c_void) {
            unsafe { nonzero($tag, d_in, n, num_nonzero, dims as _, num_dims, d_out as _, stream) }
        }
    )*};
}
nonzero_launchers! {
    count_nonzero_f32 nonzero_f32 "count_nz_f32" 0;
    count_nonzero_f64 nonzero_f64 "count_nz_f64" 1;
    count_nonzero_u8 nonzero_u8 "count_nz_u8" 2;
    count_nonzero_u32 nonzero_u32 "count_nz_u32" 3;
    count_nonzero_i16 nonzero_i16 "count_nz_i16" 4;
    count_nonzero_i32 nonzero_i32 "count_nz_i32" 5;
    count_nonzero_i64 nonzero_i64 "count_nz_i64" 6;
}

/// count_nonzero_{bf16,f16}: host-pass stubs (`#if __CUDA_ARCH__ >= 800` is false on the host).
pub unsafe extern "C" fn count_nonzero_bf16(_d_in: *const c_void, _n: u32, _stream: *mut c_void) -> u32 {
    0
}
pub unsafe extern "C" fn count_nonzero_f16(_d_in: *const c_void, _n: u32, _stream: *mut c_void) -> u32 {
    0
}
/// nonzero_{bf16,f16}: host-pass `assert(false)` stubs.
pub unsafe extern "C" fn nonzero_bf16(_d_in: *const c_void, _n: u32, _nz: u32, _dims: *const c_void, _nd: u32, _o: *mut c_void, _s: *mut c_void) {
    assert_false(b"kernels/ops/ops.cu\0", 186, b"void nonzero_bf16(const uint16_t*, uint32_t, uint32_t, const uint32_t*, uint32_t, uint32_t*, cudaStream_t)\0")
}
pub unsafe extern "C" fn nonzero_f16(_d_in: *const c_void, _n: u32, _nz: u32, _dims: *const c_void, _nd: u32, _o: *mut c_void, _s: *mut c_void) {
    assert_false(b"kernels/ops/ops.cu\0", 192, b"void nonzero_f16(const uint16_t*, uint32_t, uint32_t, const uint32_t*, uint32_t, uint32_t*, cudaStream_t)\0")
}

// ================================================================================================
// gemv/gemv.cu: block = get_optimal_block_size(K) (32/64/128/256 by K/2), grid (M); batch sizes
// 1..8 pick the instance, anything else the batch-1 instance.

const GEMV_BLOCKS: [u32; 4] = [32, 64, 128, 256];

fn gemv_name(t: &str, block: u32, batch: i32) -> &'static str {
    let b = if (1..=8).contains(&batch) { batch } else { 1 };
    // Names are generated in main.rs as gemv_{t}_{block}_{batch}; intern them once.
    static NAMES: OnceLock<HashMap<String, &'static str>> = OnceLock::new();
    let map = NAMES.get_or_init(|| {
        let mut m = HashMap::new();
        for t in ["f32", "f16", "bf16"] {
            for bl in GEMV_BLOCKS {
                for bt in 1..=8 {
                    let s = format!("gemv_{t}_{bl}_{bt}");
                    m.insert(s.clone(), &*Box::leak(s.into_boxed_str()));
                }
            }
        }
        m
    });
    map[&format!("gemv_{t}_{block}_{b}")]
}

/// gemv.cu `get_optimal_block_size`.
pub fn get_optimal_block_size(k: i32) -> u32 {
    let k2 = k / 2;
    if k2 <= 32 { 32 } else if k2 <= 64 { 64 } else if k2 <= 128 { 128 } else { 256 }
}

unsafe fn gemv(t: &str, a: *const c_void, x: *const c_void, bias: *const c_void, y: *mut c_void, m: i32, k: i32, batch: i32, has_bias: bool, stream: *mut c_void) {
    let block = get_optimal_block_size(k);
    let name = gemv_name(t, block, batch);
    let (mut a, mut x, mut bi, mut y, mut m2, mut k, mut hb) = (a, x, bias, y, m, k, has_bias as u8);
    unsafe { launch(name, (m as u32, 1, 1), (block, 1, 1), 0, stream, &mut args!(a, x, bi, y, m2, k, hb)) };
}
pub unsafe extern "C" fn launch_gemv_bf16(a: *const u16, x: *const u16, bias: *const u16, y: *mut u16, m: i32, k: i32, batch_size: i32, has_bias: bool, stream: *mut c_void) {
    unsafe { gemv("bf16", a as _, x as _, bias as _, y as _, m, k, batch_size, has_bias, stream) }
}
pub unsafe extern "C" fn launch_gemv_f16(a: *const u16, x: *const u16, bias: *const u16, y: *mut u16, m: i32, k: i32, batch_size: i32, has_bias: bool, stream: *mut c_void) {
    unsafe { gemv("f16", a as _, x as _, bias as _, y as _, m, k, batch_size, has_bias, stream) }
}
pub unsafe extern "C" fn launch_gemv_f32(a: *const f32, x: *const f32, bias: *const f32, y: *mut f32, m: i32, k: i32, batch_size: i32, has_bias: bool, stream: *mut c_void) {
    unsafe { gemv("f32", a as _, x as _, bias as _, y as _, m, k, batch_size, has_bias, stream) }
}

// ================================================================================================
// indexed_moe/indexed_moe.cu

pub unsafe extern "C" fn launch_quantize_q8_1(x: *const f32, vy: *mut c_void, kx: i32, kx_padded: i32, num_blocks_x: i32, num_rows: i32, stream: *mut c_void) {
    let (mut x, mut y, mut kx, mut kp) = (x, vy, kx, kx_padded);
    unsafe { launch("quantize_q8_1", (num_blocks_x as u32, num_rows as u32, 1), (256, 1, 1), 0, stream, &mut args!(x, y, kx, kp)) };
}
unsafe fn quantize_q8_1_t(name: &'static str, x: *const c_void, vy: *mut c_void, kx: i32, kx_padded: i32, num_rows: i32, stream: *mut c_void) {
    let nbx = kx_padded.wrapping_add(255) / 256;
    let (mut x, mut y, mut kx, mut kp) = (x, vy, kx, kx_padded);
    unsafe { launch(name, (nbx as u32, num_rows as u32, 1), (256, 1, 1), 0, stream, &mut args!(x, y, kx, kp)) };
}
pub unsafe extern "C" fn launch_quantize_q8_1_bf16(x: *const c_void, vy: *mut c_void, kx: i32, kx_padded: i32, num_rows: i32, stream: *mut c_void) {
    unsafe { quantize_q8_1_t("quantize_q8_1_bf16", x, vy, kx, kx_padded, num_rows, stream) }
}
pub unsafe extern "C" fn launch_quantize_q8_1_f16(x: *const c_void, vy: *mut c_void, kx: i32, kx_padded: i32, num_rows: i32, stream: *mut c_void) {
    unsafe { quantize_q8_1_t("quantize_q8_1_f16", x, vy, kx, kx_padded, num_rows, stream) }
}

macro_rules! moe_launchers {
    ($($imf:ident $imk:literal $fg:ident $fgk:literal $da:ident $dak:literal;)*) => {$(
        /// grid (n, batch, topk), block (32, 4).
        pub unsafe extern "C" fn $imf(
            all_weights: *const c_void, all_inputs: *const c_void, indices: *const u32, all_outputs: *mut f32, n: i32, k: i32,
            batch: i32, topk: i32, k_padded: i32, input_dim1: i32, stream: *mut c_void,
        ) {
            let (mut w, mut x, mut i, mut o, mut n2, mut k, mut b, mut t, mut kp, mut d1) = (all_weights, all_inputs, indices, all_outputs, n, k, batch, topk, k_padded, input_dim1);
            unsafe { launch($imk, (n as u32, batch as u32, topk as u32), (32, 4, 1), 0, stream, &mut args!(w, x, i, o, n2, k, b, t, kp, d1)) };
        }
        /// grid ((n + 7) / 8, topk, batch), block (32, 4).
        pub unsafe extern "C" fn $fg(
            gate_weights: *const c_void, up_weights: *const c_void, all_inputs: *const c_void, indices: *const u32, all_outputs: *mut f32,
            n: i32, k: i32, batch: i32, topk: i32, k_padded: i32, act_type: i32, stream: *mut c_void,
        ) {
            let gx = n.wrapping_add(7) / 8;
            let (mut g, mut u, mut x, mut i, mut o, mut n, mut k, mut b, mut t, mut kp, mut a) = (gate_weights, up_weights, all_inputs, indices, all_outputs, n, k, batch, topk, k_padded, act_type);
            unsafe { launch($fgk, (gx as u32, topk as u32, batch as u32), (32, 4, 1), 0, stream, &mut args!(g, u, x, i, o, n, k, b, t, kp, a)) };
        }
        /// grid ((n + 15) / 16, topk, batch), block (32, 4).
        pub unsafe extern "C" fn $da(
            all_weights: *const c_void, all_inputs: *const c_void, indices: *const u32, topk_weights: *const f32, all_outputs: *mut f32,
            n: i32, k: i32, batch: i32, topk: i32, k_padded: i32, stream: *mut c_void,
        ) {
            let gx = n.wrapping_add(15) / 16;
            let (mut w, mut x, mut i, mut tw, mut o, mut n, mut k, mut b, mut t, mut kp) = (all_weights, all_inputs, indices, topk_weights, all_outputs, n, k, batch, topk, k_padded);
            unsafe { launch($dak, (gx as u32, topk as u32, batch as u32), (32, 4, 1), 0, stream, &mut args!(w, x, i, tw, o, n, k, b, t, kp)) };
        }
    )*};
}
moe_launchers! {
    launch_indexed_moe_forward_q4_0_q8_1 "indexed_moe_forward_q4_0_q8_1" launch_moe_gemv_fused_gate_up_q4_0_q8_1 "moe_gemv_fused_gate_up_q4_0_q8_1" launch_moe_gemv_down_aggregate_q4_0_q8_1 "moe_gemv_down_aggregate_q4_0_q8_1";
    launch_indexed_moe_forward_q4_1_q8_1 "indexed_moe_forward_q4_1_q8_1" launch_moe_gemv_fused_gate_up_q4_1_q8_1 "moe_gemv_fused_gate_up_q4_1_q8_1" launch_moe_gemv_down_aggregate_q4_1_q8_1 "moe_gemv_down_aggregate_q4_1_q8_1";
    launch_indexed_moe_forward_q5_0_q8_1 "indexed_moe_forward_q5_0_q8_1" launch_moe_gemv_fused_gate_up_q5_0_q8_1 "moe_gemv_fused_gate_up_q5_0_q8_1" launch_moe_gemv_down_aggregate_q5_0_q8_1 "moe_gemv_down_aggregate_q5_0_q8_1";
    launch_indexed_moe_forward_q5_1_q8_1 "indexed_moe_forward_q5_1_q8_1" launch_moe_gemv_fused_gate_up_q5_1_q8_1 "moe_gemv_fused_gate_up_q5_1_q8_1" launch_moe_gemv_down_aggregate_q5_1_q8_1 "moe_gemv_down_aggregate_q5_1_q8_1";
    launch_indexed_moe_forward_q8_0_q8_1 "indexed_moe_forward_q8_0_q8_1" launch_moe_gemv_fused_gate_up_q8_0_q8_1 "moe_gemv_fused_gate_up_q8_0_q8_1" launch_moe_gemv_down_aggregate_q8_0_q8_1 "moe_gemv_down_aggregate_q8_0_q8_1";
    launch_indexed_moe_forward_q2k_q8_1 "indexed_moe_forward_q2k_q8_1" launch_moe_gemv_fused_gate_up_q2k_q8_1 "moe_gemv_fused_gate_up_q2k_q8_1" launch_moe_gemv_down_aggregate_q2k_q8_1 "moe_gemv_down_aggregate_q2k_q8_1";
    launch_indexed_moe_forward_q3k_q8_1 "indexed_moe_forward_q3k_q8_1" launch_moe_gemv_fused_gate_up_q3k_q8_1 "moe_gemv_fused_gate_up_q3k_q8_1" launch_moe_gemv_down_aggregate_q3k_q8_1 "moe_gemv_down_aggregate_q3k_q8_1";
    launch_indexed_moe_forward_q4k_q8_1 "indexed_moe_forward_q4k_q8_1" launch_moe_gemv_fused_gate_up_q4k_q8_1 "moe_gemv_fused_gate_up_q4k_q8_1" launch_moe_gemv_down_aggregate_q4k_q8_1 "moe_gemv_down_aggregate_q4k_q8_1";
    launch_indexed_moe_forward_q5k_q8_1 "indexed_moe_forward_q5k_q8_1" launch_moe_gemv_fused_gate_up_q5k_q8_1 "moe_gemv_fused_gate_up_q5k_q8_1" launch_moe_gemv_down_aggregate_q5k_q8_1 "moe_gemv_down_aggregate_q5k_q8_1";
    launch_indexed_moe_forward_q6k_q8_1 "indexed_moe_forward_q6k_q8_1" launch_moe_gemv_fused_gate_up_q6k_q8_1 "moe_gemv_fused_gate_up_q6k_q8_1" launch_moe_gemv_down_aggregate_q6k_q8_1 "moe_gemv_down_aggregate_q6k_q8_1";
    launch_indexed_moe_forward_q8_1_q8_1 "indexed_moe_forward_q8_1_q8_1" launch_moe_gemv_fused_gate_up_q8_1_q8_1 "moe_gemv_fused_gate_up_q8_1_q8_1" launch_moe_gemv_down_aggregate_q8_1_q8_1 "moe_gemv_down_aggregate_q8_1_q8_1";
}

// ================================================================================================
// moe_grouped/moe_grouped.cu

pub unsafe extern "C" fn launch_moe_dispatch(
    topk_ids: *const i32, expert_bounds: *mut i32, sorted_token_ids: *mut i32, sorted_source_ids: *mut i32, total_assignments: i32,
    num_experts: i32, topk: i32, expert_counts: *mut i32, expert_cursors: *mut i32, stream: *mut c_void,
) {
    let s = stream as sys::CUstream;
    let bytes = (num_experts as i64 as usize).wrapping_mul(4);
    unsafe {
        let _ = sys::cuMemsetD8Async(expert_counts as sys::CUdeviceptr, 0, bytes, s);
        let blocks = total_assignments.wrapping_add(255) / 256;
        let (mut t, mut c, mut n) = (topk_ids, expert_counts, total_assignments);
        launch("moe_dispatch_count_kernel", (blocks as u32, 1, 1), (256, 1, 1), 0, stream, &mut args!(t, c, n));
        let (mut c, mut b, mut ne) = (expert_counts, expert_bounds, num_experts);
        launch("moe_dispatch_prefix_sum_kernel", (1, 1, 1), (1, 1, 1), 0, stream, &mut args!(c, b, ne));
        let _ = sys::cuMemcpyDtoDAsync_v2(expert_cursors as sys::CUdeviceptr, expert_bounds as sys::CUdeviceptr, bytes, s);
        let (mut t, mut cu, mut so, mut ss, mut n, mut tk) = (topk_ids, expert_cursors, sorted_token_ids, sorted_source_ids, total_assignments, topk);
        launch("moe_dispatch_scatter_kernel", (blocks as u32, 1, 1), (256, 1, 1), 0, stream, &mut args!(t, cu, so, ss, n, tk));
    }
}

unsafe fn weighted_reduce(name: &'static str, inputs: *const f32, topk_weights: *const f32, outputs: *mut c_void, num_tokens: i32, hidden: i32, topk: i32, stream: *mut c_void) {
    let grid = (num_tokens as u32, (hidden.wrapping_add(255) / 256) as u32, 1);
    let smem = (topk as i64 as usize).wrapping_mul(4) as u32;
    let (mut i, mut w, mut o, mut t, mut h, mut k) = (inputs, topk_weights, outputs, num_tokens, hidden, topk);
    unsafe { launch(name, grid, (256, 1, 1), smem, stream, &mut args!(i, w, o, t, h, k)) };
}
pub unsafe extern "C" fn launch_moe_weighted_reduce_flat(inputs: *const f32, topk_weights: *const f32, outputs: *mut f32, num_tokens: i32, hidden: i32, topk: i32, stream: *mut c_void) {
    unsafe { weighted_reduce("moe_weighted_reduce_flat_f32", inputs, topk_weights, outputs as _, num_tokens, hidden, topk, stream) }
}
pub unsafe extern "C" fn launch_moe_weighted_reduce_flat_bf16(inputs: *const f32, topk_weights: *const f32, outputs: *mut c_void, num_tokens: i32, hidden: i32, topk: i32, stream: *mut c_void) {
    unsafe { weighted_reduce("moe_weighted_reduce_flat_bf16", inputs, topk_weights, outputs, num_tokens, hidden, topk, stream) }
}

macro_rules! grouped_launchers {
    ($($f:ident $k:literal;)*) => {$(
        /// grid (ceil(N / 64), num_experts), block (32, 8). (The reference's dynamic shared memory
        /// holds byte tiles this port does not need; every size used stays under the 48 KB default,
        /// so no launch-failure behaviour is lost.)
        pub unsafe extern "C" fn $f(
            all_weights: *const c_void, all_inputs: *const c_void, expert_bounds: *const i32, sorted_token_ids: *const i32,
            topk_weights: *const f32, all_outputs: *mut f32, n: i32, k: i32, k_padded: i32, num_experts: i32, topk: i32,
            input_dim1: i32, stream: *mut c_void,
        ) {
            let gx = n.wrapping_add(63) / 64;
            let (mut w, mut x, mut eb, mut st, mut tw, mut o) = (all_weights, all_inputs, expert_bounds, sorted_token_ids, topk_weights, all_outputs);
            let (mut n, mut k, mut kp, mut ne, mut tk, mut d1) = (n, k, k_padded, num_experts, topk, input_dim1);
            unsafe { launch($k, (gx as u32, num_experts as u32, 1), (32, 8, 1), 0, stream, &mut args!(w, x, eb, st, tw, o, n, k, kp, ne, tk, d1)) };
        }
    )*};
}
grouped_launchers! {
    launch_moe_grouped_gemm_q8_0 "moe_grouped_gemm_q8_0";
    launch_moe_grouped_gemm_q4_0 "moe_grouped_gemm_q4_0";
    launch_moe_grouped_gemm_q4_1 "moe_grouped_gemm_q4_1";
    launch_moe_grouped_gemm_q5_0 "moe_grouped_gemm_q5_0";
    launch_moe_grouped_gemm_q5_1 "moe_grouped_gemm_q5_1";
    launch_moe_grouped_gemm_q8_1 "moe_grouped_gemm_q8_1";
    launch_moe_grouped_gemm_q2k "moe_grouped_gemm_q2k";
    launch_moe_grouped_gemm_q3k "moe_grouped_gemm_q3k";
    launch_moe_grouped_gemm_q4k "moe_grouped_gemm_q4k";
    launch_moe_grouped_gemm_q5k "moe_grouped_gemm_q5k";
    launch_moe_grouped_gemm_q6k "moe_grouped_gemm_q6k";
}
