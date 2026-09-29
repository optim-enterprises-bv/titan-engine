//! Pure-Rust replacements for candle-kernels' MMQ C launchers (`ffi.rs`):
//! `launch_mmq_quantize_q8_1_{D4,DS4,D2S6}` and `launch_mmq_gguf_{q4_0,...,q6_k}`.
//!
//! Same signatures, same launch decisions (tile size mmq_x, need_check, stream-k fixup) as the C
//! code in mmq_instance_*.cu / mmq_quantize.cu as compiled into libmoe.a (nvcc for sm_120a:
//! `__CUDA_ARCH_LIST__` = 1200, so `ggml_cuda_highest_compiled_arch(cc)` is 1200 for cc >= 1200
//! and -1 below), launching the cuda-oxide kernels on the caller's stream through the driver API.
//! The module is loaded once per CUDA context.
//!
//! These functions are deliberately not `#[no_mangle]`: the gate links the real C symbols of the
//! same names from libmoe.a and calls both.
#![allow(clippy::too_many_arguments)]

use cuda_core::sys;
use std::collections::HashMap;
use std::ffi::{CString, c_void};
use std::sync::Mutex;

struct Module {
    module: sys::CUmodule,
    funcs: HashMap<String, sys::CUfunction>,
}
unsafe impl Send for Module {}

static MODULES: Mutex<Option<HashMap<usize, Module>>> = Mutex::new(None);

/// The kernel image: `CANDLE_MMQ_PTX` (a file path) if set, else this executable's embedded
/// cuda-oxide artifact (cubin preferred over PTX).
fn image() -> Vec<u8> {
    if let Ok(p) = std::env::var("CANDLE_MMQ_PTX") {
        let mut v = std::fs::read(&p).unwrap_or_else(|e| panic!("CANDLE_MMQ_PTX={p}: {e}"));
        v.push(0);
        return v;
    }
    use cuda_core::embedded::{ArtifactPayloadKind, artifact_bundles_from_current_exe};
    let bundles = artifact_bundles_from_current_exe().expect("embedded cuda-oxide artifacts");
    let b = bundles.iter().find(|b| b.name == "candle-mmq").or_else(|| bundles.first()).expect("candle-mmq artifact bundle");
    if let Some(c) = b.payload(ArtifactPayloadKind::Cubin) {
        return c.to_vec();
    }
    let mut v = b.payload(ArtifactPayloadKind::Ptx).expect("ptx payload").to_vec();
    if v.last() != Some(&0) {
        v.push(0);
    }
    v
}

/// Resolve `name` in the module of the stream's context (loading the module on first use).
unsafe fn function(stream: sys::CUstream, name: &str) -> sys::CUfunction {
    unsafe {
        let mut ctx: sys::CUcontext = std::ptr::null_mut();
        if stream.is_null() || sys::cuStreamGetCtx(stream, &mut ctx) != sys::cudaError_enum_CUDA_SUCCESS || ctx.is_null() {
            sys::cuCtxGetCurrent(&mut ctx);
        }
        let mut guard = MODULES.lock().unwrap();
        let map = guard.get_or_insert_with(HashMap::new);
        let m = map.entry(ctx as usize).or_insert_with(|| {
            let img = image();
            let mut module: sys::CUmodule = std::ptr::null_mut();
            let r = sys::cuModuleLoadData(&mut module, img.as_ptr() as *const c_void);
            assert_eq!(r, sys::cudaError_enum_CUDA_SUCCESS, "cuModuleLoadData (candle-mmq)");
            Module { module, funcs: HashMap::new() }
        });
        if let Some(f) = m.funcs.get(name) {
            return *f;
        }
        let mut f: sys::CUfunction = std::ptr::null_mut();
        let cname = CString::new(name).unwrap();
        let r = sys::cuModuleGetFunction(&mut f, m.module, cname.as_ptr());
        assert_eq!(r, sys::cudaError_enum_CUDA_SUCCESS, "cuModuleGetFunction {name}");
        m.funcs.insert(name.to_string(), f);
        f
    }
}

/// One kernel argument, passed by value.
#[derive(Clone, Copy)]
enum A {
    P(*const c_void),
    I(i32),
    U(u64),
    L(i64),
}

unsafe fn launch(stream: *mut c_void, name: &str, grid: (u32, u32, u32), block: (u32, u32, u32), shared: u32, args: &[A]) {
    unsafe {
        let stream = stream as sys::CUstream;
        let f = function(stream, name);
        if shared > 0 {
            // CUDA_SET_SHARED_MEMORY_LIMIT (errors ignored, as in the C launcher).
            sys::cuFuncSetAttribute(f, sys::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared as i32);
        }
        let mut vals: Vec<[u8; 8]> = args
            .iter()
            .map(|a| match *a {
                A::P(p) => (p as u64).to_le_bytes(),
                A::I(v) => (v as u32 as u64).to_le_bytes(),
                A::U(v) => v.to_le_bytes(),
                A::L(v) => v.to_le_bytes(),
            })
            .collect();
        let mut ptrs: Vec<*mut c_void> = vals.iter_mut().map(|v| v.as_mut_ptr() as *mut c_void).collect();
        // Launch errors are not checked by the C launchers either (no cudaGetLastError).
        sys::cuLaunchKernel(f, grid.0, grid.1, grid.2, block.0, block.1, block.2, shared, stream, ptrs.as_mut_ptr(), std::ptr::null_mut());
    }
}

// ------------------------------------------------------------------------------------------------
// Host-side architecture queries (mmq_common.cuh / mmq_gguf.cuh), for the sm_120a build.

const MAX_CPY_OFFSET_MTHREADS: i32 = 0x0100000;
fn is_nvidia(cc: i32) -> bool {
    cc < MAX_CPY_OFFSET_MTHREADS
}
/// ggml_cuda_highest_compiled_arch with __CUDA_ARCH_LIST__ = 1200.
fn highest_compiled_arch(cc: i32) -> i32 {
    if 1200 <= cc { 1200 } else { -1 }
}
fn turing_mma_available(cc: i32) -> bool {
    is_nvidia(cc) && highest_compiled_arch(cc) >= 750
}
fn volta_plus(cc: i32) -> bool {
    is_nvidia(cc) && highest_compiled_arch(cc) >= 700
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Ty {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
}

impl Ty {
    fn name(self) -> &'static str {
        match self {
            Ty::Q4_0 => "q4_0",
            Ty::Q4_1 => "q4_1",
            Ty::Q5_0 => "q5_0",
            Ty::Q5_1 => "q5_1",
            Ty::Q8_0 => "q8_0",
            Ty::Q2K => "q2_k",
            Ty::Q3K => "q3_k",
            Ty::Q4K => "q4_k",
            Ty::Q5K => "q5_k",
            Ty::Q6K => "q6_k",
        }
    }
    fn qk(self) -> i32 {
        match self {
            Ty::Q2K | Ty::Q3K | Ty::Q4K | Ty::Q5K | Ty::Q6K => 256,
            _ => 32,
        }
    }
    /// mmq_get_mma_tile_x_k.
    fn mma_tile_x_k(self) -> i64 {
        match self {
            Ty::Q2K => 100,
            Ty::Q3K => 84,
            _ => 76,
        }
    }
    /// mmq_get_dp4a_tile_x_sizes (qs, dm, sc).
    fn dp4a_txs(self, mmq_y: i64) -> (i64, i64, i64) {
        const K: i64 = 32;
        let q8_0 = (mmq_y * K * 2 + mmq_y, mmq_y * K * 2 / 8 + mmq_y / (8 / 2), 0);
        match self {
            Ty::Q4_0 | Ty::Q4_1 => (mmq_y * K + mmq_y, mmq_y * K / 4 + mmq_y / 4, 0),
            Ty::Q5_0 | Ty::Q8_0 => q8_0,
            Ty::Q5_1 => (mmq_y * K * 2 + mmq_y, mmq_y * K * 2 / 8 + mmq_y / (8 / 2), 0),
            Ty::Q2K => (mmq_y * K * 2 + mmq_y, mmq_y * K + mmq_y, 0),
            Ty::Q3K => (mmq_y * K * 2 + mmq_y, mmq_y, mmq_y * K / 8 + mmq_y / 8),
            Ty::Q4K => (mmq_y * K + mmq_y, mmq_y * K / 32, mmq_y * K / 8 + mmq_y / 8),
            Ty::Q5K => (mmq_y * K * 2 + mmq_y, mmq_y * K / 32 + mmq_y / 32, mmq_y * K / 8 + mmq_y / 8),
            Ty::Q6K => (mmq_y * K * 2 + mmq_y, mmq_y * K / 32 + mmq_y / 32, mmq_y * K / 8 + mmq_y / 8),
        }
    }
}

/// mmq_get_nbytes_shared<type> (size_t arithmetic).
fn nbytes_shared(t: Ty, mmq_x: i32, mmq_y: i32, cc: i32, warp_size: i32, nwarps: i32) -> u64 {
    let (qs, dm, sc) = t.dp4a_txs(mmq_y as i64);
    let nbs_ids = (mmq_x as i64 as u64).wrapping_mul(4);
    let nbs_x = if turing_mma_available(cc) {
        ((mmq_y as i64 * t.mma_tile_x_k()) as u64).wrapping_mul(4)
    } else {
        (qs as u64).wrapping_mul(4).wrapping_add((dm as u64).wrapping_mul(4)).wrapping_add((sc as u64).wrapping_mul(4))
    };
    let nbs_y = (mmq_x as i64 as u64).wrapping_mul(144);
    let n = ((nwarps.wrapping_mul(warp_size)) as i64 as u64).wrapping_mul(4);
    let pad = (nbs_y.wrapping_add(n).wrapping_sub(1)) & !(n.wrapping_sub(1));
    nbs_ids.wrapping_add(nbs_x).wrapping_add(pad)
}

/// launch_mmq_case_<type> + instantiate_mmq_<type><mmq_x>.
unsafe fn launch_mmq(
    t: Ty, tmp_fixup: *mut c_void, x: *const c_void, y: *const c_void, dst: *mut c_void,
    ncols_x: i64, nrows_x: i64, ncols_y: i64, stride_row_x: i64, cc: i32, nsm: i32, smpbo: i64, warp_size: i32,
    stream: *mut c_void,
) {
    let use_stream_k = volta_plus(cc);
    let ncols_max = ncols_y;
    let mmq_x_max = if turing_mma_available(cc) { 128 } else { 64 };
    let mmq_y: i32 = if volta_plus(cc) { 128 } else { 64 };
    let nwarps = 256 / warp_size;
    let smpbo = smpbo as u64; // size_t parameter

    let mut mmq_x_best = 0;
    let mut ntiles_x_best = i32::MAX;
    let mut mmq_x = 8;
    while mmq_x <= mmq_x_max && ntiles_x_best > 1 {
        let granularity = if turing_mma_available(cc) && mmq_x >= 48 { 16 } else { 8 };
        if mmq_x % granularity == 0 {
            let nbs = nbytes_shared(t, mmq_x, mmq_y, cc, warp_size, nwarps);
            if nbs <= smpbo {
                let ntiles_x = ((ncols_max + mmq_x as i64 - 1) / mmq_x as i64) as i32;
                if ntiles_x < ntiles_x_best {
                    mmq_x_best = mmq_x;
                    ntiles_x_best = ntiles_x;
                }
            }
        }
        mmq_x += 8;
    }
    if !matches!(mmq_x_best, 8 | 16 | 24 | 32 | 40 | 48 | 56 | 64 | 72 | 80 | 88 | 96 | 104 | 112 | 120 | 128) {
        return;
    }
    let mmq_x = mmq_x_best;

    let nbytes = nbytes_shared(t, mmq_x, mmq_y, cc, warp_size, nwarps) as i32;
    let nty = ((nrows_x + mmq_y as i64 - 1) / mmq_y as i64) as i32;
    let ntx = ((ncols_max + mmq_x as i64 - 1) / mmq_x as i64) as i32;
    let ntzw: i32 = 1;
    let block = (warp_size as u32, nwarps as u32, 1);
    let need_check = nrows_x % mmq_y as i64 != 0;
    let name = format!("mmq_{}_x{}_nc{}", t.name(), mmq_x, need_check as i32);
    let args = |fixup: *mut c_void| {
        [
            A::P(x), A::P(y), A::P(dst), A::P(fixup),
            A::I(ncols_x as i32), A::I(nrows_x as i32), A::I(ncols_y as i32), A::I(stride_row_x as i32),
            A::I(ncols_y as i32), A::I(nrows_x as i32), A::I(ncols_max as i32),
        ]
    };
    unsafe {
        if !use_stream_k {
            launch(stream, &name, (nty as u32, ntx as u32, ntzw as u32), block, nbytes as u32, &args(std::ptr::null_mut()));
            return;
        }
        let grid_sk = (nsm as u32, 1, 1);
        let fixup_needed = ntx.wrapping_mul(nty).wrapping_mul(ntzw) % nsm != 0;
        launch(stream, &name, grid_sk, block, nbytes as u32, &args(tmp_fixup));
        if fixup_needed {
            let fname = format!("mmq_fixup_qk{}_x{}_nc{}", t.qk(), mmq_x, need_check as i32);
            launch(stream, &fname, grid_sk, block, 0, &[
                A::P(dst), A::P(tmp_fixup), A::I(ncols_x as i32), A::I(nrows_x as i32), A::I(ncols_y as i32),
                A::U(nrows_x as u64), A::I(ncols_max as i32),
            ]);
        }
    }
}

macro_rules! mmq_launcher {
    ($fn:ident, $t:expr) => {
        /// Same signature and launch logic as the candle C launcher of this name.
        pub unsafe extern "C" fn $fn(
            tmp_fixup: *mut c_void, x: *const c_void, y: *const c_void, dst: *mut c_void,
            ncols_x: i64, nrows_x: i64, ncols_y: i64, stride_row_x: i64, _stride_col_dst: i64,
            cc: i32, nsm: i32, smpbo: i64, warp_size: i32, stream: *mut c_void,
        ) {
            // The C launcher ignores stride_col_dst: the dst column stride is nrows_x.
            unsafe { launch_mmq($t, tmp_fixup, x, y, dst, ncols_x, nrows_x, ncols_y, stride_row_x, cc, nsm, smpbo, warp_size, stream) }
        }
    };
}
mmq_launcher!(launch_mmq_gguf_q4_0, Ty::Q4_0);
mmq_launcher!(launch_mmq_gguf_q4_1, Ty::Q4_1);
mmq_launcher!(launch_mmq_gguf_q5_0, Ty::Q5_0);
mmq_launcher!(launch_mmq_gguf_q5_1, Ty::Q5_1);
mmq_launcher!(launch_mmq_gguf_q8_0, Ty::Q8_0);
mmq_launcher!(launch_mmq_gguf_q2_k, Ty::Q2K);
mmq_launcher!(launch_mmq_gguf_q3_k, Ty::Q3K);
mmq_launcher!(launch_mmq_gguf_q4_k, Ty::Q4K);
mmq_launcher!(launch_mmq_gguf_q5_k, Ty::Q5K);
mmq_launcher!(launch_mmq_gguf_q6_k, Ty::Q6K);

/// launch_mmq_quantize_q8_1_{D4,DS4,D2S6}: grid (ne1, ceil(ne0/512), ne2*ne3), 128 threads.
unsafe fn launch_quantize(
    name: &str, x: *const c_void, ids: *const i32, vy: *mut c_void,
    ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i64, ne2: i64, ne3: i64, stream: *mut c_void,
) {
    let block_num_y = (ne0 + 4 * 128 - 1) / (4 * 128);
    let grid = (ne1 as u32, block_num_y as u32, ne2.wrapping_mul(ne3) as u32);
    unsafe {
        launch(stream, name, grid, (128, 1, 1), 0, &[
            A::P(x), A::P(ids as *const c_void), A::P(vy), A::L(ne00), A::L(s01), A::L(s02), A::L(s03), A::L(ne0),
            A::I(ne1 as i32), A::I(ne2 as i32),
        ]);
    }
}

macro_rules! quantize_launcher {
    ($fn:ident, $k:expr) => {
        /// Same signature and launch logic as the candle C launcher of this name.
        #[allow(non_snake_case)]
        pub unsafe extern "C" fn $fn(
            x: *const c_void, ids: *const i32, vy: *mut c_void, _type_x: i32,
            ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i64, ne2: i64, ne3: i64, stream: *mut c_void,
        ) {
            unsafe { launch_quantize($k, x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2, ne3, stream) }
        }
    };
}
quantize_launcher!(launch_mmq_quantize_q8_1_D4, "quantize_mmq_q8_1_d4");
quantize_launcher!(launch_mmq_quantize_q8_1_DS4, "quantize_mmq_q8_1_ds4");
quantize_launcher!(launch_mmq_quantize_q8_1_D2S6, "quantize_mmq_q8_1_d2s6");
