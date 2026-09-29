//! Pure-Rust twins of the 33 extern "C" host launchers in candle-kernels `mmvq_gguf.cu`
//! (`launch_mmvq_gguf_<q>_<bf16|f16|f32>_plain`, `launch_mmvq_gguf_quantize_q8_1_<t>`).
//!
//! Same parameter lists and C ABI as `candle_kernels::ffi` (so each can be used as candle's
//! `PlainLauncher` fn pointer), same grid / block / shared-memory selection, and the oxide kernels
//! (same entry names as the nvcc ones) are launched on the caller's `stream` (a `CUstream` /
//! `cudaStream_t`, null = legacy default stream) through the CUDA driver API.
//!
//! The oxide module is loaded once per CUDA context (the stream's context, or the current one for
//! the null stream; if no context is current, device 0's primary context, as the runtime API would)
//! and its functions are cached. The module image is the `#[cuda_module]` bundle embedded in the
//! running executable; if there is none, the PTX file named by `$CANDLE_MMVQ_PTX`, else
//! `<crate>/candle_mmvq.ptx`.
//!
//! Like the C launchers (`<<<>>>` without error checks), launch failures are not reported:
//! `b_size` outside 1..=8 launches nothing, `nrows_x <= 0` gives an empty grid that the driver
//! rejects. Unlike the runtime API, a failed driver launch here does not set `cudaGetLastError`.
use cuda_core::sys;
use std::collections::HashMap;
use std::ffi::{CString, c_void};
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
            if b.name == "candle_mmvq" {
                if let Some(p) = b.payload(cuda_core::embedded::ArtifactPayloadKind::Cubin)
                    .or_else(|| b.payload(cuda_core::embedded::ArtifactPayloadKind::Ptx))
                {
                    let mut v = p.to_vec();
                    v.push(0);
                    return v;
                }
            }
        }
    }
    let path = std::env::var("CANDLE_MMVQ_PTX").unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/candle_mmvq.ptx").to_string());
    let mut v = std::fs::read(&path).unwrap_or_else(|e| panic!("candle-mmvq: no embedded module and cannot read {path}: {e}"));
    v.push(0);
    v
}

fn check(r: sys::CUresult, what: &str) {
    assert!(r == 0, "candle-mmvq: {what} failed: {r:?}");
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

unsafe fn launch(name: &'static str, grid: (u32, u32, u32), block: (u32, u32, u32), stream: *mut c_void, args: &mut [*mut c_void]) {
    unsafe {
        let s = stream as sys::CUstream;
        let f = function(name, s);
        // Errors are dropped, as the C launcher's `<<<>>>` drops them.
        let _ = sys::cuLaunchKernel(f, grid.0, grid.1, grid.2, block.0, block.1, block.2, 0, s, args.as_mut_ptr(), std::ptr::null_mut());
    }
}

/// `MMVQ_LAUNCHER_PLAIN`: block (32, nwarps), grid (ceil(nrows_x / rows_per_block)), no dynamic
/// shared memory; `b_size` picks the `_cudaN` instance, anything outside 1..=8 launches nothing.
#[allow(clippy::too_many_arguments)]
unsafe fn plain(
    names: &[&'static str; 8], vx: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32, nrows_x: i32,
    stride_col_y: i32, stride_col_dst: i32, b_size: i32, stream: *mut c_void,
) {
    let rows_per_block: u32 = if b_size <= 1 { 1 } else { 2 };
    let nblocks = (nrows_x as u32).wrapping_add(rows_per_block - 1) / rows_per_block;
    let nwarps: u32 = if b_size <= 4 { 4 } else { 2 };
    if !(1..=8).contains(&b_size) {
        return;
    }
    let (mut vx, mut vy, mut dst) = (vx, vy, dst);
    let (mut a, mut b, mut c, mut d) = (ncols_x, nrows_x, stride_col_y, stride_col_dst);
    let mut args: [*mut c_void; 7] = [
        &raw mut vx as _, &raw mut vy as _, &raw mut dst as _, &raw mut a as _, &raw mut b as _, &raw mut c as _, &raw mut d as _,
    ];
    unsafe { launch(names[(b_size - 1) as usize], (nblocks, 1, 1), (32, nwarps, 1), stream, &mut args) }
}

/// Quantize launchers: grid ((kx_padded + 255) / 256, num_rows), block 256.
unsafe fn quantize(name: &'static str, x: *const c_void, vy: *mut c_void, kx: i32, kx_padded: i32, num_rows: i32, stream: *mut c_void) {
    let num_blocks_x = kx_padded.wrapping_add(255) / 256;
    let (mut x, mut vy, mut kx, mut kp) = (x, vy, kx, kx_padded);
    let mut args: [*mut c_void; 4] = [&raw mut x as _, &raw mut vy as _, &raw mut kx as _, &raw mut kp as _];
    unsafe { launch(name, (num_blocks_x as u32, num_rows as u32, 1), (256, 1, 1), stream, &mut args) }
}

macro_rules! plain_launcher {
    ($fn:ident, $q:literal, $d:literal) => {
        #[allow(clippy::too_many_arguments)]
        pub unsafe extern "C" fn $fn(
            vx: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32, nrows_x: i32,
            stride_col_y: i32, stride_col_dst: i32, b_size: i32, stream: *mut c_void,
        ) {
            const NAMES: [&str; 8] = [
                concat!("mmvq_gguf_", $q, "_", $d, "_plain_cuda1"), concat!("mmvq_gguf_", $q, "_", $d, "_plain_cuda2"),
                concat!("mmvq_gguf_", $q, "_", $d, "_plain_cuda3"), concat!("mmvq_gguf_", $q, "_", $d, "_plain_cuda4"),
                concat!("mmvq_gguf_", $q, "_", $d, "_plain_cuda5"), concat!("mmvq_gguf_", $q, "_", $d, "_plain_cuda6"),
                concat!("mmvq_gguf_", $q, "_", $d, "_plain_cuda7"), concat!("mmvq_gguf_", $q, "_", $d, "_plain_cuda8"),
            ];
            unsafe { plain(&NAMES, vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, b_size, stream) }
        }
    };
}

plain_launcher!(launch_mmvq_gguf_q4_0_bf16_plain, "q4_0", "bf16");
plain_launcher!(launch_mmvq_gguf_q4_1_bf16_plain, "q4_1", "bf16");
plain_launcher!(launch_mmvq_gguf_q5_0_bf16_plain, "q5_0", "bf16");
plain_launcher!(launch_mmvq_gguf_q5_1_bf16_plain, "q5_1", "bf16");
plain_launcher!(launch_mmvq_gguf_q8_0_bf16_plain, "q8_0", "bf16");
plain_launcher!(launch_mmvq_gguf_q2_k_bf16_plain, "q2_k", "bf16");
plain_launcher!(launch_mmvq_gguf_q3_k_bf16_plain, "q3_k", "bf16");
plain_launcher!(launch_mmvq_gguf_q4_k_bf16_plain, "q4_k", "bf16");
plain_launcher!(launch_mmvq_gguf_q5_k_bf16_plain, "q5_k", "bf16");
plain_launcher!(launch_mmvq_gguf_q6_k_bf16_plain, "q6_k", "bf16");
plain_launcher!(launch_mmvq_gguf_q4_0_f16_plain, "q4_0", "f16");
plain_launcher!(launch_mmvq_gguf_q4_1_f16_plain, "q4_1", "f16");
plain_launcher!(launch_mmvq_gguf_q5_0_f16_plain, "q5_0", "f16");
plain_launcher!(launch_mmvq_gguf_q5_1_f16_plain, "q5_1", "f16");
plain_launcher!(launch_mmvq_gguf_q8_0_f16_plain, "q8_0", "f16");
plain_launcher!(launch_mmvq_gguf_q2_k_f16_plain, "q2_k", "f16");
plain_launcher!(launch_mmvq_gguf_q3_k_f16_plain, "q3_k", "f16");
plain_launcher!(launch_mmvq_gguf_q4_k_f16_plain, "q4_k", "f16");
plain_launcher!(launch_mmvq_gguf_q5_k_f16_plain, "q5_k", "f16");
plain_launcher!(launch_mmvq_gguf_q6_k_f16_plain, "q6_k", "f16");
plain_launcher!(launch_mmvq_gguf_q4_0_f32_plain, "q4_0", "f32");
plain_launcher!(launch_mmvq_gguf_q4_1_f32_plain, "q4_1", "f32");
plain_launcher!(launch_mmvq_gguf_q5_0_f32_plain, "q5_0", "f32");
plain_launcher!(launch_mmvq_gguf_q5_1_f32_plain, "q5_1", "f32");
plain_launcher!(launch_mmvq_gguf_q8_0_f32_plain, "q8_0", "f32");
plain_launcher!(launch_mmvq_gguf_q2_k_f32_plain, "q2_k", "f32");
plain_launcher!(launch_mmvq_gguf_q3_k_f32_plain, "q3_k", "f32");
plain_launcher!(launch_mmvq_gguf_q4_k_f32_plain, "q4_k", "f32");
plain_launcher!(launch_mmvq_gguf_q5_k_f32_plain, "q5_k", "f32");
plain_launcher!(launch_mmvq_gguf_q6_k_f32_plain, "q6_k", "f32");

pub unsafe extern "C" fn launch_mmvq_gguf_quantize_q8_1_bf16(x: *const c_void, vy: *mut c_void, kx: i32, kx_padded: i32, num_rows: i32, stream: *mut c_void) {
    unsafe { quantize("mmvq_gguf_quantize_q8_1_bf16", x, vy, kx, kx_padded, num_rows, stream) }
}
pub unsafe extern "C" fn launch_mmvq_gguf_quantize_q8_1_f16(x: *const c_void, vy: *mut c_void, kx: i32, kx_padded: i32, num_rows: i32, stream: *mut c_void) {
    unsafe { quantize("mmvq_gguf_quantize_q8_1_f16", x, vy, kx, kx_padded, num_rows, stream) }
}
pub unsafe extern "C" fn launch_mmvq_gguf_quantize_q8_1_f32(x: *const c_void, vy: *mut c_void, kx: i32, kx_padded: i32, num_rows: i32, stream: *mut c_void) {
    unsafe { quantize("mmvq_gguf_quantize_q8_1_f32", x, vy, kx, kx_padded, num_rows, stream) }
}
