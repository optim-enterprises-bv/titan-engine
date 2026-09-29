//! CUDA fast path for GGUF tiled matmul (prompt/prefill phase).
//! Handles batch > 8 (complement to fast_mmvq which handles batch 1-8).

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::thread::ThreadId;

use candle_core::cuda::cudarc::driver::{
    CudaSlice, CudaStream, DevicePtrMut, DeviceRepr, SyncOnDrop,
};
use candle_core::cuda_backend::CudaDType;
use candle_core::{
    quantized::{GgmlDType, QTensor},
    CudaDevice, CudaStorage, DType, Device, Result, Shape, Storage, Tensor,
};

use super::ffi;
use crate::{
    utils::{slice_ptr, slice_ptr_mut_on_stream, slice_ptr_on_stream},
    GluActivationType,
};

const QK8_1: usize = 32;
const BLOCK_Q8_1_MMQ_SIZE: usize = 4 * QK8_1 + 4 * 4; // 128 qs + 16 scale bytes = 144
const MATRIX_ROW_PADDING: usize = 512;
const MMQ_X_MAX: usize = 128;
const MMQ_Y_MAX: usize = 128;

#[inline]
fn pad(p: usize, q: usize) -> usize {
    p.div_ceil(q) * q
}

fn output_shape(xs: &Tensor, nrows: usize) -> Shape {
    let mut out_dims = xs.dims().to_vec();
    let last = out_dims.len() - 1;
    out_dims[last] = nrows;
    Shape::from(out_dims)
}

fn wrap_cuda_output<T: CudaDType + DeviceRepr>(
    out: CudaSlice<T>,
    dev: &CudaDevice,
    shape: Shape,
) -> Tensor {
    Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
        shape,
    ))
}

/// Quant types supported by MMQ kernels (same as MMVQ).
pub fn supports(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q4_0
            | GgmlDType::Q4_1
            | GgmlDType::Q5_0
            | GgmlDType::Q5_1
            | GgmlDType::Q8_0
            | GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
    )
}

/// qk (block quantization size) per dtype.
fn qk_for(dtype: GgmlDType) -> usize {
    match dtype {
        GgmlDType::Q4_0 | GgmlDType::Q4_1 | GgmlDType::Q5_0 | GgmlDType::Q5_1 | GgmlDType::Q8_0 => {
            32
        }
        GgmlDType::Q2K | GgmlDType::Q3K | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K => 256,
        _ => unreachable!(),
    }
}

// ds_layout mapping: which Q8_1_mmq scale layout to use per weight type.
// D4 = scale only, DS4 = scale+partial_sum, D2S6 = 2 scales + 6 partial_sums
enum DsLayout {
    D4,
    DS4,
    D2S6,
}

fn ds_layout_for(dtype: GgmlDType) -> DsLayout {
    match dtype {
        GgmlDType::Q4_0 | GgmlDType::Q4_1 => DsLayout::DS4,
        GgmlDType::Q5_0 => DsLayout::D4,
        GgmlDType::Q5_1 => DsLayout::DS4,
        GgmlDType::Q8_0 => DsLayout::D4,
        GgmlDType::Q2K => DsLayout::D2S6,
        GgmlDType::Q3K => DsLayout::D4,
        GgmlDType::Q4K | GgmlDType::Q5K => DsLayout::DS4,
        GgmlDType::Q6K => DsLayout::D4,
        _ => unreachable!(),
    }
}

type QuantizeLauncher = unsafe extern "C" fn(
    x: *const std::ffi::c_void,
    ids: *const i32,
    vy: *mut std::ffi::c_void,
    type_x: i32,
    ne00: i64,
    s01: i64,
    s02: i64,
    s03: i64,
    ne0: i64,
    ne1: i64,
    ne2: i64,
    ne3: i64,
    stream: *mut std::ffi::c_void,
);

type QuantizeGluF32Launcher = unsafe extern "C" fn(
    gate: *const f32,
    up: *const f32,
    ids: *const i32,
    vy: *mut std::ffi::c_void,
    ne00: i64,
    s01: i64,
    ne0: i64,
    ne1: i64,
    activation: i32,
    stream: *mut std::ffi::c_void,
);

type QuantizeGluLauncher = unsafe extern "C" fn(
    gate: *const std::ffi::c_void,
    up: *const std::ffi::c_void,
    ids: *const i32,
    vy: *mut std::ffi::c_void,
    type_x: i32,
    ne00: i64,
    s01: i64,
    ne0: i64,
    ne1: i64,
    activation: i32,
    stream: *mut std::ffi::c_void,
);

fn quantize_launcher(layout: DsLayout) -> QuantizeLauncher {
    match layout {
        DsLayout::D4 => ffi::launch_mmq_quantize_q8_1_D4,
        DsLayout::DS4 => ffi::launch_mmq_quantize_q8_1_DS4,
        DsLayout::D2S6 => ffi::launch_mmq_quantize_q8_1_D2S6,
    }
}

fn quantize_glu_f32_launcher(layout: DsLayout) -> QuantizeGluF32Launcher {
    match layout {
        DsLayout::D4 => ffi::launch_mmq_quantize_glu_q8_1_D4_f32,
        DsLayout::DS4 => ffi::launch_mmq_quantize_glu_q8_1_DS4_f32,
        DsLayout::D2S6 => ffi::launch_mmq_quantize_glu_q8_1_D2S6_f32,
    }
}

fn quantize_glu_launcher(layout: DsLayout) -> QuantizeGluLauncher {
    match layout {
        DsLayout::D4 => ffi::launch_mmq_quantize_glu_q8_1_D4,
        DsLayout::DS4 => ffi::launch_mmq_quantize_glu_q8_1_DS4,
        DsLayout::D2S6 => ffi::launch_mmq_quantize_glu_q8_1_D2S6,
    }
}

type MmqLauncher = unsafe extern "C" fn(
    tmp_fixup: *mut std::ffi::c_void,
    x: *const std::ffi::c_void,
    y: *const std::ffi::c_void,
    dst: *mut std::ffi::c_void,
    ncols_x: i64,
    nrows_x: i64,
    ncols_y: i64,
    stride_row_x: i64,
    stride_col_dst: i64,
    cc: i32,
    nsm: i32,
    smpbo: i64,
    warp_size: i32,
    type_dst: i32,
    stream: *mut std::ffi::c_void,
);

type MmqMoeLauncher = unsafe extern "C" fn(
    tmp_fixup: *mut std::ffi::c_void,
    x: *const std::ffi::c_void,
    y: *const std::ffi::c_void,
    ids_dst: *const i32,
    expert_bounds: *const i32,
    dst: *mut std::ffi::c_void,
    ncols_x: i64,
    nrows_x: i64,
    ncols_dst: i64,
    stride_row_x: i64,
    stride_col_dst: i64,
    num_experts: i64,
    ncols_max: i64,
    cc: i32,
    nsm: i32,
    smpbo: i64,
    warp_size: i32,
    stream: *mut std::ffi::c_void,
);

fn mmq_launcher(dtype: GgmlDType) -> Option<MmqLauncher> {
    let f: MmqLauncher = match dtype {
        GgmlDType::Q4_0 => ffi::launch_mmq_gguf_q4_0,
        GgmlDType::Q4_1 => ffi::launch_mmq_gguf_q4_1,
        GgmlDType::Q5_0 => ffi::launch_mmq_gguf_q5_0,
        GgmlDType::Q5_1 => ffi::launch_mmq_gguf_q5_1,
        GgmlDType::Q8_0 => ffi::launch_mmq_gguf_q8_0,
        GgmlDType::Q2K => ffi::launch_mmq_gguf_q2_k,
        GgmlDType::Q3K => ffi::launch_mmq_gguf_q3_k,
        GgmlDType::Q4K => ffi::launch_mmq_gguf_q4_k,
        GgmlDType::Q5K => ffi::launch_mmq_gguf_q5_k,
        GgmlDType::Q6K => ffi::launch_mmq_gguf_q6_k,
        _ => return None,
    };
    Some(f)
}

fn mmq_moe_launcher(dtype: GgmlDType) -> Option<MmqMoeLauncher> {
    let f: MmqMoeLauncher = match dtype {
        GgmlDType::Q4_0 => ffi::launch_mmq_gguf_q4_0_moe,
        GgmlDType::Q4_1 => ffi::launch_mmq_gguf_q4_1_moe,
        GgmlDType::Q5_0 => ffi::launch_mmq_gguf_q5_0_moe,
        GgmlDType::Q5_1 => ffi::launch_mmq_gguf_q5_1_moe,
        GgmlDType::Q8_0 => ffi::launch_mmq_gguf_q8_0_moe,
        GgmlDType::Q2K => ffi::launch_mmq_gguf_q2_k_moe,
        GgmlDType::Q3K => ffi::launch_mmq_gguf_q3_k_moe,
        GgmlDType::Q4K => ffi::launch_mmq_gguf_q4_k_moe,
        GgmlDType::Q5K => ffi::launch_mmq_gguf_q5_k_moe,
        GgmlDType::Q6K => ffi::launch_mmq_gguf_q6_k_moe,
        _ => return None,
    };
    Some(f)
}

struct WorkspaceSlot {
    slice: CudaSlice<u8>,
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct WorkspaceKey {
    device: candle_core::cuda::DeviceId,
    stream: usize,
    thread: ThreadId,
    capacity: usize,
}

struct WorkspaceGuard<'a> {
    slot: MutexGuard<'static, WorkspaceSlot>,
    stream: &'a CudaStream,
}

impl WorkspaceGuard<'_> {
    fn ptr_mut(&mut self) -> (u64, SyncOnDrop<'_>) {
        self.slot.slice.device_ptr_mut(self.stream)
    }
}

type WsMap = Mutex<HashMap<WorkspaceKey, &'static Mutex<WorkspaceSlot>>>;

static MMQ_WORKSPACE: OnceLock<WsMap> = OnceLock::new();
static FIXUP_WORKSPACE: OnceLock<WsMap> = OnceLock::new();

#[derive(Clone, Copy)]
struct DeviceInfo {
    cc: i32,
    nsm: i32,
    smpbo: i64,
    warp_size: i32,
}

static DEVICE_INFO: OnceLock<Mutex<HashMap<candle_core::cuda::DeviceId, DeviceInfo>>> =
    OnceLock::new();

fn get_device_info(dev: &CudaDevice) -> DeviceInfo {
    use candle_core::cuda::cudarc::driver::{result, sys};
    let map = DEVICE_INFO.get_or_init(|| Mutex::new(HashMap::new()));
    let key = dev.id();
    let mut guard = map.lock().unwrap();
    if let Some(info) = guard.get(&key) {
        return *info;
    }
    let cu_device = dev.cuda_stream().context().cu_device();
    let major = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
    }
    .unwrap_or(8);
    let minor = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
    }
    .unwrap_or(0);
    let nsm = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )
    }
    .unwrap_or(1);
    let smpbo = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        )
    }
    .unwrap_or(49152);
    let warp_size = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE,
        )
    }
    .unwrap_or(32);
    let info = DeviceInfo {
        cc: major * 100 + minor * 10,
        nsm,
        smpbo: smpbo as i64,
        warp_size,
    };
    guard.insert(key, info);
    info
}

fn fixup_workspace_bytes(dev: &CudaDevice) -> usize {
    get_device_info(dev).nsm as usize * MMQ_X_MAX * MMQ_Y_MAX * std::mem::size_of::<f32>()
}

fn workspace_ensure<'a>(
    ws: &'static OnceLock<WsMap>,
    dev: &CudaDevice,
    bytes: usize,
    stream: &'a CudaStream,
) -> Result<WorkspaceGuard<'a>> {
    let map = ws.get_or_init(|| Mutex::new(HashMap::new()));
    let capacity = bytes.max(1).next_power_of_two();
    let key = WorkspaceKey {
        device: dev.id(),
        stream: stream.cu_stream() as usize,
        thread: std::thread::current().id(),
        capacity,
    };
    let device_mtx: &'static Mutex<WorkspaceSlot> = {
        let mut guard = map.lock().unwrap();
        match guard.get(&key).copied() {
            Some(mtx) => mtx,
            None => {
                let slice = unsafe { dev.alloc::<u8>(capacity)? };
                // CUDA graph replay requires process-stable workspace addresses.
                let leaked = Box::leak(Box::new(Mutex::new(WorkspaceSlot { slice })));
                guard.insert(key, leaked);
                leaked
            }
        }
    };
    let slot = device_mtx.lock().unwrap();
    Ok(WorkspaceGuard { slot, stream })
}

// ---------------------------------------------------------------------------------------------
// GGML Q1_0 (type 41) prefill. cuda-oxide kernels from titan-engine/oxide-kernels/q1_0,
// bit-identical to llama.cpp acecd56: `quantize_mmq_q8_1<D4>` on the activations (f16 / bf16
// widened exactly), then `mul_mat_q<Q1_0, J, fallback>` (stream-k, MMA) and, when the stream-k
// split leaves partial tiles, `mul_mat_q_stream_k_fixup`, with the host logic of mmq.cuh
// (`mul_mat_q_switch_J`, `launch_mul_mat_q`). The f32 result is then cast to the input dtype.
// Regenerate q1_0_mmq_oxide.ptx with oxide-kernels/q1_0/export_ptx.py.

const Q1_0_MMQ_PTX: &str = include_str!("q1_0_mmq_oxide.ptx");
const Q1_0_MMQ_MODULE: &str = "titan_q1_0_mmq";
/// Quantized prefill activations (block_q8_1_mmq), tagged with the input they hold.
static Q1_0_MMQ_WS: OnceLock<super::fast_mmvq::Q1WsMap> = OnceLock::new();
/// J (tile width in the token direction) instances of mmq-config-ampere.cuh for Q1_0.
const Q1_0_J_NC: [u32; 11] = [8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128];
const Q1_0_J_FB: [u32; 5] = [8, 16, 32, 64, 128];

/// mmq_get_nbytes_shared: ids[J] + tile_x (128 rows x 76 ints) + tile_y (J block_q8_1_mmq,
/// padded to 256 threads x 4 bytes).
fn q1_0_mmq_smem(j: u32) -> u32 {
    j * 4 + 128 * 76 * 4 + (j * 144).div_ceil(1024) * 1024
}

/// mul_mat_q_switch_J: the smallest configured J minimising ceil(ncols / J).
fn q1_0_mmq_pick_j(ncols: usize, fallback: bool) -> u32 {
    let set: &[u32] = if fallback { &Q1_0_J_FB } else { &Q1_0_J_NC };
    let (mut best, mut nt_best) = (0u32, usize::MAX);
    for &j in set {
        if nt_best <= 1 {
            break;
        }
        let nt = ncols.div_ceil(j as usize);
        if nt < nt_best {
            best = j;
            nt_best = nt;
        }
    }
    best
}

/// [`plain`] for Q1_0.
fn q1_0_plain(w: &QTensor, xs: &Tensor, dev: &CudaDevice, nrows: usize, k: usize, b_size: usize) -> Result<Tensor> {
    use candle_core::cuda::cudarc::driver::sys::CUfunction_attribute_enum;
    use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle_core::cuda::WrapErr;
    if k % 128 != 0 {
        candle_core::bail!("fast_mmq q1_0: k={k} not a multiple of 128");
    }
    let input_ty = xs.dtype();
    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        candle_core::bail!("fast_mmq: input must live on CUDA");
    };
    let xs_offset = xs_layout.start_offset();
    let stream = dev.cuda_stream();

    // quantize_mmq_q8_1 into block_q8_1_mmq [k_padded / 128][b_size], plus 128 spare blocks
    // (the tile_y copy of the last column tile reads up to J_max blocks past the end).
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let y_bytes = (b_size * (k_padded / (4 * QK8_1)) + 128) * BLOCK_Q8_1_MMQ_SIZE;
    let mut y_ws = super::fast_mmvq::q1_0_workspace(&Q1_0_MMQ_WS, dev, y_bytes)?;
    let tag: super::fast_mmvq::Q1Tag = (xs.id(), input_ty, xs_offset, xs.dims().to_vec());
    let quantize = y_ws.tag.as_ref() != Some(&tag);
    y_ws.tag = None; // set again once the launches went through
    let (y_ptr, y_guard) = {
        use candle_core::cuda::cudarc::driver::DevicePtrMut;
        y_ws.slice.device_ptr_mut(&stream)
    };

    let fallback = nrows % 128 != 0;
    let j = q1_0_mmq_pick_j(b_size, fallback);
    let nty = nrows.div_ceil(128) as u32;
    let ntx = b_size.div_ceil(j as usize) as u32;
    let ntiles = ntx * nty;
    let nsm = get_device_info(dev).nsm.max(1) as u32;
    let nwaves = ntiles.div_ceil(nsm);
    let nblocks = if 100 * ntiles / (nsm * nwaves) >= 90 { ntiles } else { nsm };
    let fixup_needed = ntiles % nblocks != 0;
    let fixup_bytes = if fixup_needed { nblocks as usize * j as usize * 128 * 4 } else { 4 };
    let mut fixup_ws = workspace_ensure(&FIXUP_WORKSPACE, dev, fixup_bytes, &stream)?;
    let (fixup_ptr, _fixup_guard) = fixup_ws.ptr_mut();

    let (w_ptr, _w_guard) = w.device_ptr_with_guard(&stream)?;
    let w_ptr = w_ptr as u64;
    let mut out = unsafe { dev.alloc::<f32>(nrows * b_size)? };
    {
        let (x_ptr, _x_guard) = match input_ty {
            DType::BF16 => slice_ptr_on_stream(xs_cuda.as_cuda_slice::<half::bf16>()?, xs_offset, &stream),
            DType::F16 => slice_ptr_on_stream(xs_cuda.as_cuda_slice::<half::f16>()?, xs_offset, &stream),
            DType::F32 => slice_ptr_on_stream(xs_cuda.as_cuda_slice::<f32>()?, xs_offset, &stream),
            other => candle_core::bail!("fast_mmq q1_0: input dtype must be BF16, F16, or F32, got {other:?}"),
        };
        let (dst_ptr, _dst_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);
        let quant_name = match input_ty {
            DType::BF16 => "q1_0_quantize_mmq_d4_bf16",
            DType::F16 => "q1_0_quantize_mmq_d4_f16",
            _ => "q1_0_quantize_mmq_d4_f32",
        };
        // quantize_mmq_q8_1: x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2, n_expert_used
        let ti = match input_ty {
            DType::BF16 => 2,
            DType::F16 => 1,
            _ => 0,
        };
        let quant = super::fast_mmvq::q1_0_func(dev, 100 + ti, Q1_0_MMQ_MODULE, Q1_0_MMQ_PTX, || quant_name.to_string())?;
        let (ne00, s01, s02, ne0) = (k as i64, k as i64, (k * b_size) as i64, k_padded as i64);
        let (ne1, ne2, n_expert_used) = (b_size as i32, 1i32, 0i32);
        let z64 = 0u64;
        if quantize {
            let mut b = stream.launch_builder(&quant);
            b.arg(&x_ptr).arg(&z64).arg(&y_ptr).arg(&ne00).arg(&s01).arg(&s02).arg(&s02).arg(&ne0);
            b.arg(&ne1).arg(&ne2).arg(&n_expert_used);
            let cfg = LaunchConfig { grid_dim: (b_size as u32, (k_padded as u32).div_ceil(512), 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
            unsafe { b.launch(cfg) }.w()?;
        }

        let fb = fallback as u32;
        let bpn = super::fast_mmvq::q1_0_fastdiv((k / 128) as u32);
        let ntx_fd = super::fast_mmvq::q1_0_fastdiv(ntx);
        let one = super::fast_mmvq::q1_0_fastdiv(1);
        let (nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst) =
            (nrows as i32, b_size as i32, (k / 128) as i32, b_size as i32, nrows as i32);
        let z32 = 0i32;
        let smem = q1_0_mmq_smem(j);
        // cache slots 104.. (mmq) and 204.. (fixup), by J / 8 and fallback
        let slot = (j / 8) as usize * 2 + fb as usize;
        let mmq = super::fast_mmvq::q1_0_func(dev, 104 + slot, Q1_0_MMQ_MODULE, Q1_0_MMQ_PTX, || format!("q1_0_mmq_j{j}_f{fb}"))?;
        mmq.set_attribute(CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32).w()?;
        let mut b = stream.launch_builder(&mmq);
        // x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, blocks_per_ne00, nrows_x,
        // ncols_dst, stride_row_x, ncols_y, stride_col_dst, channel_ratio, nchannels_y,
        // stride_channel_{x,y,dst}, sample_ratio, nsamples_y, stride_sample_{x,y,dst}, ntx
        b.arg(&w_ptr).arg(&y_ptr).arg(&z64).arg(&z64).arg(&dst_ptr).arg(&fixup_ptr).arg(&z64);
        b.arg(&bpn[0]).arg(&bpn[1]).arg(&bpn[2]);
        b.arg(&nrows_x).arg(&ncols_dst).arg(&stride_row_x).arg(&ncols_y).arg(&stride_col_dst);
        b.arg(&one[0]).arg(&one[1]).arg(&one[2]).arg(&one[0]).arg(&one[1]).arg(&one[2]).arg(&z32).arg(&z32).arg(&z32);
        b.arg(&one[0]).arg(&one[1]).arg(&one[2]).arg(&one[0]).arg(&one[1]).arg(&one[2]).arg(&z32).arg(&z32).arg(&z32);
        b.arg(&ntx_fd[0]).arg(&ntx_fd[1]).arg(&ntx_fd[2]);
        let cfg = LaunchConfig { grid_dim: (nblocks, 1, 1), block_dim: (32, 8, 1), shared_mem_bytes: smem };
        unsafe { b.launch(cfg) }.w()?;

        if fixup_needed {
            let fix = super::fast_mmvq::q1_0_func(dev, 204 + slot, Q1_0_MMQ_MODULE, Q1_0_MMQ_PTX, || format!("q1_0_mmq_fixup_j{j}_f{fb}"))?;
            let mut b = stream.launch_builder(&fix);
            // ids_dst, expert_bounds, dst, tmp_last_tile, blocks_per_ne00, nrows_x, ncols_dst,
            // stride_col_dst, nchannels_y, stride_channel_dst, nsamples_y, stride_sample_dst, ntx
            b.arg(&z64).arg(&z64).arg(&dst_ptr).arg(&fixup_ptr);
            b.arg(&bpn[0]).arg(&bpn[1]).arg(&bpn[2]);
            b.arg(&nrows_x).arg(&ncols_dst).arg(&stride_col_dst);
            b.arg(&one[0]).arg(&one[1]).arg(&one[2]).arg(&z32).arg(&one[0]).arg(&one[1]).arg(&one[2]).arg(&z32);
            b.arg(&ntx_fd[0]).arg(&ntx_fd[1]).arg(&ntx_fd[2]);
            let cfg = LaunchConfig { grid_dim: (nblocks, 4, 1), block_dim: (32, 4, 1), shared_mem_bytes: 0 };
            unsafe { b.launch(cfg) }.w()?;
        }
    }
    drop(y_guard);
    y_ws.tag = Some(tag);
    let out = wrap_cuda_output(out, dev, output_shape(&xs, nrows));
    if input_ty == DType::F32 {
        Ok(out)
    } else {
        out.to_dtype(input_ty)
    }
}

// ---------------------------------------------------------------------------------------------
// GGML IQ4_NL (type 20), MXFP4 (39), NVFP4 (40) prefill. cuda-oxide kernels from
// titan-engine/oxide-kernels/fmt_mmq, bit-identical to llama.cpp acecd56 on sm_120a:
// - IQ4_NL: `quantize_mmq_q8_1<D4>` + `mul_mat_q<IQ4_NL, J, fallback>` (mmq-config-ampere.cuh,
//   s8 m16n8k32 MMA);
// - MXFP4 / NVFP4: the native-FP4 Blackwell path, `quantize_mmq_mxfp4` / `quantize_mmq_nvfp4`
//   (FP4 activations; NVFP4 also writes a per-column f32 scale) + `mul_mat_q<type, J, fallback>`
//   (mmq-config-blackwell.cuh, block-scaled m16n8k64 FP4 MMA);
// then `mul_mat_q_stream_k_fixup` when the stream-k split leaves partial tiles, with the host
// logic of mmq.cuh (`mul_mat_q_switch_J`, `launch_mul_mat_q`). Regenerate the
// `<fmt>_mmq_oxide.ptx` modules with oxide-kernels/fmt_mmq/export_ptx.py.

struct LlamaMmq {
    /// kernel name prefix / GGML type name
    name: &'static str,
    ptx: &'static str,
    module: &'static str,
    /// values per weight block
    qk: usize,
    /// values per activation block (block_q8_1_mmq 128, block_fp4_mmq 256)
    neb: usize,
    /// K_vram (values per tile iteration)
    iter_k: usize,
    /// index of this format in the function cache (slots 1000 + 200 * fi ..)
    fi: usize,
    ws: &'static OnceLock<super::fast_mmvq::Q1WsMap>,
}

static IQ4_NL_MMQ_WS: OnceLock<super::fast_mmvq::Q1WsMap> = OnceLock::new();
static MXFP4_MMQ_WS: OnceLock<super::fast_mmvq::Q1WsMap> = OnceLock::new();
static NVFP4_MMQ_WS: OnceLock<super::fast_mmvq::Q1WsMap> = OnceLock::new();

static IQ4_NL_MMQ: LlamaMmq = LlamaMmq {
    name: "iq4_nl",
    ptx: include_str!("iq4_nl_mmq_oxide.ptx"),
    module: "titan_iq4_nl_mmq",
    qk: 32,
    neb: 128,
    iter_k: 256,
    fi: 0,
    ws: &IQ4_NL_MMQ_WS,
};
static MXFP4_MMQ: LlamaMmq = LlamaMmq {
    name: "mxfp4",
    ptx: include_str!("mxfp4_mmq_oxide.ptx"),
    module: "titan_mxfp4_mmq",
    qk: 32,
    neb: 256,
    iter_k: 512,
    fi: 1,
    ws: &MXFP4_MMQ_WS,
};
static NVFP4_MMQ: LlamaMmq = LlamaMmq {
    name: "nvfp4",
    ptx: include_str!("nvfp4_mmq_oxide.ptx"),
    module: "titan_nvfp4_mmq",
    qk: 64,
    neb: 256,
    iter_k: 512,
    fi: 2,
    ws: &NVFP4_MMQ_WS,
};

fn llama_mmq(dtype: GgmlDType) -> Option<&'static LlamaMmq> {
    match dtype {
        GgmlDType::IQ4NL => Some(&IQ4_NL_MMQ),
        GgmlDType::MXFP4 => Some(&MXFP4_MMQ),
        GgmlDType::NVFP4 => Some(&NVFP4_MMQ),
        _ => None,
    }
}

/// Quant types with a llama.cpp MMQ port in [`plain`] (prefill only: their decode stays on
/// candle's mmvq).
pub fn supports_llama_mmq(dtype: GgmlDType) -> bool {
    llama_mmq(dtype).is_some()
}

/// [`plain`] for IQ4_NL / MXFP4 / NVFP4. `None` when the shape is outside what the port covers
/// (K not a whole number of tile iterations), so the caller keeps its fallback.
fn llama_mmq_plain(f: &LlamaMmq, w: &QTensor, xs: &Tensor, dev: &CudaDevice, nrows: usize, k: usize, b_size: usize) -> Result<Option<Tensor>> {
    use candle_core::cuda::cudarc::driver::sys::CUfunction_attribute_enum;
    use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle_core::cuda::WrapErr;
    if k % f.iter_k != 0 || !matches!(xs.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        return Ok(None);
    }
    let input_ty = xs.dtype();
    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        candle_core::bail!("fast_mmq: input must live on CUDA");
    };
    let xs_offset = xs_layout.start_offset();
    let stream = dev.cuda_stream();

    // Activations: [k_padded / neb][b_size] blocks of 144 bytes, 128 spare blocks (the tile_y copy
    // of the last column tile reads up to J_max blocks past the end), then NVFP4's per-column f32
    // scales; tagged with the input they hold (Q/K/V and gate/up share one quantization).
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let y_main = (b_size * (k_padded / f.neb) + 128) * BLOCK_Q8_1_MMQ_SIZE;
    let y_bytes = y_main + b_size * 4;
    let mut y_ws = super::fast_mmvq::q1_0_workspace(f.ws, dev, y_bytes)?;
    let tag: super::fast_mmvq::Q1Tag = (xs.id(), input_ty, xs_offset, xs.dims().to_vec());
    let quantize = y_ws.tag.as_ref() != Some(&tag);
    y_ws.tag = None; // set again once the launches went through
    let (y_ptr, y_guard) = {
        use candle_core::cuda::cudarc::driver::DevicePtrMut;
        y_ws.slice.device_ptr_mut(&stream)
    };
    let scale_ptr: u64 = y_ptr + y_main as u64;

    let fallback = nrows % 128 != 0;
    let j = q1_0_mmq_pick_j(b_size, fallback);
    let nty = nrows.div_ceil(128) as u32;
    let ntx = b_size.div_ceil(j as usize) as u32;
    let ntiles = ntx * nty;
    let nsm = get_device_info(dev).nsm.max(1) as u32;
    let nwaves = ntiles.div_ceil(nsm);
    let nblocks = if 100 * ntiles / (nsm * nwaves) >= 90 { ntiles } else { nsm };
    let fixup_needed = ntiles % nblocks != 0;
    let fixup_bytes = if fixup_needed { nblocks as usize * j as usize * 128 * 4 } else { 4 };
    let mut fixup_ws = workspace_ensure(&FIXUP_WORKSPACE, dev, fixup_bytes, &stream)?;
    let (fixup_ptr, _fixup_guard) = fixup_ws.ptr_mut();

    let (w_ptr, _w_guard) = w.device_ptr_with_guard(&stream)?;
    let w_ptr = w_ptr as u64;
    let mut out = unsafe { dev.alloc::<f32>(nrows * b_size)? };
    let slot0 = 1000 + 200 * f.fi;
    {
        let (x_ptr, _x_guard) = match input_ty {
            DType::BF16 => slice_ptr_on_stream(xs_cuda.as_cuda_slice::<half::bf16>()?, xs_offset, &stream),
            DType::F16 => slice_ptr_on_stream(xs_cuda.as_cuda_slice::<half::f16>()?, xs_offset, &stream),
            _ => slice_ptr_on_stream(xs_cuda.as_cuda_slice::<f32>()?, xs_offset, &stream),
        };
        let (dst_ptr, _dst_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);
        let (ti, ts) = match input_ty {
            DType::BF16 => (2, "bf16"),
            DType::F16 => (1, "f16"),
            _ => (0, "f32"),
        };
        let z64 = 0u64;
        if quantize {
            let quant = super::fast_mmvq::q1_0_func(dev, slot0 + ti, f.module, f.ptx, || match f.fi {
                0 => format!("iq4_nl_quantize_mmq_d4_{ts}"),
                _ => format!("{}_quantize_mmq_{ts}", f.name),
            })?;
            let (ne00, s01, s02, ne0) = (k as i64, k as i64, (k * b_size) as i64, k_padded as i64);
            let (ne1_32, ne2_32, ne1_64, ne2_64, neu) = (b_size as i32, 1i32, b_size as i64, 1i64, 0i32);
            let mut b = stream.launch_builder(&quant);
            let cfg = match f.fi {
                // quantize_mmq_q8_1 / quantize_mmq_mxfp4: x, ids, vy, ne00, s01, s02, s03, ne0,
                // ne1, ne2, n_expert_used (int)
                0 | 1 => {
                    b.arg(&x_ptr).arg(&z64).arg(&y_ptr).arg(&ne00).arg(&s01).arg(&s02).arg(&s02).arg(&ne0);
                    b.arg(&ne1_32).arg(&ne2_32).arg(&neu);
                    let block_dim = if f.fi == 0 { (128, 1, 1) } else { (32, 8, 1) };
                    LaunchConfig { grid_dim: (b_size as u32, (k_padded as u32).div_ceil(512), 1), block_dim, shared_mem_bytes: 0 }
                }
                // quantize_mmq_nvfp4: x, ids, vy, scale, ne00, s01, s02, s03, ne0, ne1, ne2
                // (int64), n_expert_used
                _ => {
                    b.arg(&x_ptr).arg(&z64).arg(&y_ptr).arg(&scale_ptr).arg(&ne00).arg(&s01).arg(&s02).arg(&s02).arg(&ne0);
                    b.arg(&ne1_64).arg(&ne2_64).arg(&neu);
                    LaunchConfig { grid_dim: (b_size as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 }
                }
            };
            unsafe { b.launch(cfg) }.w()?;
        }

        let fb = fallback as u32;
        let bpn = super::fast_mmvq::q1_0_fastdiv((k / f.qk) as u32);
        let ntx_fd = super::fast_mmvq::q1_0_fastdiv(ntx);
        let one = super::fast_mmvq::q1_0_fastdiv(1);
        let (nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst) =
            (nrows as i32, b_size as i32, (k / f.qk) as i32, b_size as i32, nrows as i32);
        let z32 = 0i32;
        let ys = if f.fi == 2 { scale_ptr } else { 0u64 };
        let smem = q1_0_mmq_smem(j);
        let slot = (j / 8) as usize * 2 + fb as usize;
        let name = f.name;
        let mmq = super::fast_mmvq::q1_0_func(dev, slot0 + 10 + slot, f.module, f.ptx, || format!("{name}_mmq_j{j}_f{fb}"))?;
        mmq.set_attribute(CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32).w()?;
        let mut b = stream.launch_builder(&mmq);
        // x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, blocks_per_ne00, nrows_x,
        // ncols_dst, stride_row_x, ncols_y, stride_col_dst, channel_ratio, nchannels_y,
        // stride_channel_{x,y,dst}, sample_ratio, nsamples_y, stride_sample_{x,y,dst}, ntx
        b.arg(&w_ptr).arg(&y_ptr).arg(&z64).arg(&z64).arg(&dst_ptr).arg(&fixup_ptr).arg(&ys);
        b.arg(&bpn[0]).arg(&bpn[1]).arg(&bpn[2]);
        b.arg(&nrows_x).arg(&ncols_dst).arg(&stride_row_x).arg(&ncols_y).arg(&stride_col_dst);
        b.arg(&one[0]).arg(&one[1]).arg(&one[2]).arg(&one[0]).arg(&one[1]).arg(&one[2]).arg(&z32).arg(&z32).arg(&z32);
        b.arg(&one[0]).arg(&one[1]).arg(&one[2]).arg(&one[0]).arg(&one[1]).arg(&one[2]).arg(&z32).arg(&z32).arg(&z32);
        b.arg(&ntx_fd[0]).arg(&ntx_fd[1]).arg(&ntx_fd[2]);
        let cfg = LaunchConfig { grid_dim: (nblocks, 1, 1), block_dim: (32, 8, 1), shared_mem_bytes: smem };
        unsafe { b.launch(cfg) }.w()?;

        if fixup_needed {
            let fix = super::fast_mmvq::q1_0_func(dev, slot0 + 100 + slot, f.module, f.ptx, || format!("{name}_mmq_fixup_j{j}_f{fb}"))?;
            let mut b = stream.launch_builder(&fix);
            // ids_dst, expert_bounds, dst, tmp_last_tile, blocks_per_ne00, nrows_x, ncols_dst,
            // stride_col_dst, nchannels_y, stride_channel_dst, nsamples_y, stride_sample_dst, ntx
            b.arg(&z64).arg(&z64).arg(&dst_ptr).arg(&fixup_ptr);
            b.arg(&bpn[0]).arg(&bpn[1]).arg(&bpn[2]);
            b.arg(&nrows_x).arg(&ncols_dst).arg(&stride_col_dst);
            b.arg(&one[0]).arg(&one[1]).arg(&one[2]).arg(&z32).arg(&one[0]).arg(&one[1]).arg(&one[2]).arg(&z32);
            b.arg(&ntx_fd[0]).arg(&ntx_fd[1]).arg(&ntx_fd[2]);
            let cfg = LaunchConfig { grid_dim: (nblocks, 4, 1), block_dim: (32, 4, 1), shared_mem_bytes: 0 };
            unsafe { b.launch(cfg) }.w()?;
        }
    }
    drop(y_guard);
    y_ws.tag = Some(tag);
    let out = wrap_cuda_output(out, dev, output_shape(&xs, nrows));
    Ok(Some(if input_ty == DType::F32 { out } else { out.to_dtype(input_ty)? }))
}

/// Prefill (batch > 8) for IQ4_NL / MXFP4 / NVFP4 through the llama.cpp MMQ port; `None` when
/// the weight / shape is not covered (the caller keeps its dequantize + matmul fallback).
pub fn llama_fmt_plain(w: &QTensor, xs: &Tensor) -> Result<Option<Tensor>> {
    let Some(f) = llama_mmq(w.dtype()) else {
        return Ok(None);
    };
    if std::env::var_os("TITAN_LLAMA_MMQ").is_some_and(|v| v == "0") {
        return Ok(None);
    }
    let Device::Cuda(dev) = w.device() else {
        return Ok(None);
    };
    // 2-D weights only (expert stacks go through the MoE paths)
    let Ok((nrows, ncols)) = w.shape().dims2() else {
        return Ok(None);
    };
    let (b_size, k) = match xs.dims() {
        [b, k] => (*b, *k),
        [b, m, k] => (*b * *m, *k),
        _ => return Ok(None),
    };
    if k != ncols || b_size == 0 {
        return Ok(None);
    }
    llama_mmq_plain(f, w, xs, &dev, nrows, k, b_size)
}


struct DenseMmqRun<'a> {
    weights: &'a [&'a QTensor],
    xs: &'a Tensor,
    dev: &'a CudaDevice,
    stream: &'a CudaStream,
    scratch_ptr: *mut std::ffi::c_void,
    fixup_ptr: *mut std::ffi::c_void,
    quantize: QuantizeLauncher,
    launcher: MmqLauncher,
    device_info: DeviceInfo,
    k: usize,
    k_padded: usize,
    batch_size: usize,
    qk: usize,
    type_x: i32,
}

impl DenseMmqRun<'_> {
    fn launch<T: CudaDType + DeviceRepr>(
        &self,
        xs_slice: &CudaSlice<T>,
        xs_offset: usize,
    ) -> Result<Vec<Tensor>> {
        let stream_ptr = self.stream.cu_stream() as *mut std::ffi::c_void;
        let (xs_ptr, _xs_guard) = slice_ptr_on_stream(xs_slice, xs_offset, self.stream);
        unsafe {
            (self.quantize)(
                xs_ptr as *const std::ffi::c_void,
                std::ptr::null(),
                self.scratch_ptr,
                self.type_x,
                self.k as i64,
                self.k as i64,
                0,
                0,
                self.k_padded as i64,
                self.batch_size as i64,
                1,
                1,
                stream_ptr,
            );
        }

        let mut outputs = Vec::with_capacity(self.weights.len());
        for weight in self.weights {
            let (nrows, _) = weight.shape().dims2()?;
            let (weight_ptr, _weight_guard) = weight.device_ptr_with_guard(self.stream)?;
            let mut out = unsafe { self.dev.alloc::<T>(nrows * self.batch_size)? };
            {
                let (out_ptr, _out_guard) = slice_ptr_mut_on_stream(&mut out, 0, self.stream);
                unsafe {
                    (self.launcher)(
                        self.fixup_ptr,
                        weight_ptr as *const std::ffi::c_void,
                        self.scratch_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        self.k as i64,
                        nrows as i64,
                        self.batch_size as i64,
                        (self.k / self.qk) as i64,
                        nrows as i64,
                        self.device_info.cc,
                        self.device_info.nsm,
                        self.device_info.smpbo,
                        self.device_info.warp_size,
                        self.type_x,
                        stream_ptr,
                    );
                }
            }
            outputs.push(wrap_cuda_output(
                out,
                self.dev,
                output_shape(self.xs, nrows),
            ));
        }
        Ok(outputs)
    }
}

struct DenseGluDownRun<'a> {
    down: &'a QTensor,
    gate: &'a Tensor,
    dev: &'a CudaDevice,
    stream: &'a CudaStream,
    scratch_ptr: *mut std::ffi::c_void,
    fixup_ptr: *mut std::ffi::c_void,
    quantize: QuantizeGluLauncher,
    launcher: MmqLauncher,
    device_info: DeviceInfo,
    k: usize,
    k_padded: usize,
    batch_size: usize,
    qk: usize,
    type_x: i32,
    activation: i32,
}

impl DenseGluDownRun<'_> {
    fn launch<T: CudaDType + DeviceRepr>(
        &self,
        gate_slice: &CudaSlice<T>,
        gate_offset: usize,
        up_slice: &CudaSlice<T>,
        up_offset: usize,
    ) -> Result<Tensor> {
        let stream_ptr = self.stream.cu_stream() as *mut std::ffi::c_void;
        let (gate_ptr, _gate_guard) = slice_ptr_on_stream(gate_slice, gate_offset, self.stream);
        let (up_ptr, _up_guard) = slice_ptr_on_stream(up_slice, up_offset, self.stream);
        unsafe {
            (self.quantize)(
                gate_ptr as *const std::ffi::c_void,
                up_ptr as *const std::ffi::c_void,
                std::ptr::null(),
                self.scratch_ptr,
                self.type_x,
                self.k as i64,
                self.k as i64,
                self.k_padded as i64,
                self.batch_size as i64,
                self.activation,
                stream_ptr,
            );
        }

        let (nrows, _) = self.down.shape().dims2()?;
        let (weight_ptr, _weight_guard) = self.down.device_ptr_with_guard(self.stream)?;
        let mut out = unsafe { self.dev.alloc::<T>(nrows * self.batch_size)? };
        {
            let (out_ptr, _out_guard) = slice_ptr_mut_on_stream(&mut out, 0, self.stream);
            unsafe {
                (self.launcher)(
                    self.fixup_ptr,
                    weight_ptr as *const std::ffi::c_void,
                    self.scratch_ptr as *const std::ffi::c_void,
                    out_ptr as *mut std::ffi::c_void,
                    self.k as i64,
                    nrows as i64,
                    self.batch_size as i64,
                    (self.k / self.qk) as i64,
                    nrows as i64,
                    self.device_info.cc,
                    self.device_info.nsm,
                    self.device_info.smpbo,
                    self.device_info.warp_size,
                    self.type_x,
                    stream_ptr,
                );
            }
        }
        Ok(wrap_cuda_output(
            out,
            self.dev,
            output_shape(self.gate, nrows),
        ))
    }
}

fn shared_lhs(weights: &[&QTensor], xs: &Tensor) -> Result<Vec<Tensor>> {
    let Some(first) = weights.first() else {
        candle_core::bail!("fast_mmq shared_lhs: at least one weight is required");
    };
    let dtype = first.dtype();
    if !supports(dtype) {
        candle_core::bail!("fast_mmq shared_lhs: unsupported quant dtype {dtype:?}");
    }
    let Device::Cuda(dev) = first.device() else {
        candle_core::bail!("fast_mmq shared_lhs: weights must live on CUDA");
    };
    let (_, ncols) = first.shape().dims2()?;
    for weight in &weights[1..] {
        if weight.dtype() != dtype {
            candle_core::bail!("fast_mmq shared_lhs: weight dtype mismatch");
        }
        let Device::Cuda(weight_dev) = weight.device() else {
            candle_core::bail!("fast_mmq shared_lhs: weights must live on CUDA");
        };
        if weight_dev.id() != dev.id() {
            candle_core::bail!("fast_mmq shared_lhs: weights are on different CUDA devices");
        }
        let (_, weight_ncols) = weight.shape().dims2()?;
        if weight_ncols != ncols {
            candle_core::bail!(
                "fast_mmq shared_lhs: weight ncols mismatch {ncols} vs {weight_ncols}"
            );
        }
    }
    if !xs.device().same_device(&first.device()) {
        candle_core::bail!("fast_mmq shared_lhs: input and weights are on different devices");
    }

    let Some((&k, batch_dims)) = xs.dims().split_last() else {
        candle_core::bail!("fast_mmq shared_lhs: input must have at least one dimension");
    };
    let batch_size = batch_dims.iter().product::<usize>();
    if batch_size == 0 {
        candle_core::bail!("fast_mmq shared_lhs: batch size must be greater than zero");
    }
    if k != ncols {
        candle_core::bail!(
            "fast_mmq shared_lhs: weight ncols {ncols} does not match input tail {k}"
        );
    }

    let qk = qk_for(dtype);
    if k % qk != 0 {
        candle_core::bail!("fast_mmq shared_lhs: k={k} not divisible by qk={qk}");
    }
    let input_ty = xs.dtype();
    if !matches!(input_ty, DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!(
            "fast_mmq shared_lhs: input dtype must be BF16, F16, or F32, got {input_ty:?}"
        );
    }

    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        candle_core::bail!("fast_mmq shared_lhs: input must live on CUDA");
    };
    let xs_offset = xs_layout.start_offset();
    let type_x = match input_ty {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 30,
        _ => unreachable!(),
    };

    let stream = dev.cuda_stream();
    let k_padded = pad(pad(k, MATRIX_ROW_PADDING), 4 * QK8_1);
    let blocks_per_row = k_padded / (4 * QK8_1);
    let workspace_main = batch_size * blocks_per_row * BLOCK_Q8_1_MMQ_SIZE;
    let workspace_extra = MMQ_X_MAX * BLOCK_Q8_1_MMQ_SIZE;
    let workspace_bytes = workspace_main + workspace_extra;
    let mut workspace = workspace_ensure(&MMQ_WORKSPACE, &dev, workspace_bytes, &stream)?;
    let (scratch_ptr, _scratch_guard) = workspace.ptr_mut();
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;

    let fixup_bytes = fixup_workspace_bytes(&dev);
    let mut fixup_workspace = workspace_ensure(&FIXUP_WORKSPACE, &dev, fixup_bytes, &stream)?;
    let (fixup_ptr, _fixup_guard) = fixup_workspace.ptr_mut();
    let fixup_ptr = fixup_ptr as *mut std::ffi::c_void;

    let run = DenseMmqRun {
        weights,
        xs: &xs,
        dev: &dev,
        stream: &stream,
        scratch_ptr,
        fixup_ptr,
        quantize: quantize_launcher(ds_layout_for(dtype)),
        launcher: mmq_launcher(dtype).expect("supports() checked"),
        device_info: get_device_info(&dev),
        k,
        k_padded,
        batch_size,
        qk,
        type_x,
    };
    match input_ty {
        DType::BF16 => run.launch(xs_cuda.as_cuda_slice::<half::bf16>()?, xs_offset),
        DType::F16 => run.launch(xs_cuda.as_cuda_slice::<half::f16>()?, xs_offset),
        DType::F32 => run.launch(xs_cuda.as_cuda_slice::<f32>()?, xs_offset),
        _ => unreachable!(),
    }
}

fn down_from_glu(
    down: &QTensor,
    gate: &Tensor,
    up: &Tensor,
    activation: GluActivationType,
) -> Result<Tensor> {
    let dtype = down.dtype();
    if !supports(dtype) {
        candle_core::bail!("fast_mmq down_from_glu: unsupported quant dtype {dtype:?}");
    }
    let Device::Cuda(dev) = down.device() else {
        candle_core::bail!("fast_mmq down_from_glu: weight must live on CUDA");
    };
    if gate.shape() != up.shape() {
        candle_core::bail!(
            "fast_mmq down_from_glu: gate/up shape mismatch {:?} vs {:?}",
            gate.shape(),
            up.shape()
        );
    }
    if gate.dtype() != up.dtype() {
        candle_core::bail!(
            "fast_mmq down_from_glu: gate/up dtype mismatch {:?} vs {:?}",
            gate.dtype(),
            up.dtype()
        );
    }
    if !gate.device().same_device(&down.device()) || !up.device().same_device(&down.device()) {
        candle_core::bail!("fast_mmq down_from_glu: tensors are on different devices");
    }

    let Some((&k, batch_dims)) = gate.dims().split_last() else {
        candle_core::bail!("fast_mmq down_from_glu: input must have at least one dimension");
    };
    let batch_size = batch_dims.iter().product::<usize>();
    if batch_size == 0 {
        candle_core::bail!("fast_mmq down_from_glu: batch size must be greater than zero");
    }
    let (_, ncols) = down.shape().dims2()?;
    if k != ncols {
        candle_core::bail!(
            "fast_mmq down_from_glu: weight ncols {ncols} does not match input tail {k}"
        );
    }
    let qk = qk_for(dtype);
    if k % qk != 0 {
        candle_core::bail!("fast_mmq down_from_glu: k={k} not divisible by qk={qk}");
    }

    let input_ty = gate.dtype();
    let type_x = match input_ty {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 30,
        other => candle_core::bail!(
            "fast_mmq down_from_glu: input dtype must be BF16, F16, or F32, got {other:?}"
        ),
    };
    let gate = gate.contiguous()?;
    let up = up.contiguous()?;
    let (gate_storage, gate_layout) = gate.storage_and_layout();
    let Storage::Cuda(gate_cuda) = &*gate_storage else {
        candle_core::bail!("fast_mmq down_from_glu: gate must live on CUDA");
    };
    let (up_storage, up_layout) = up.storage_and_layout();
    let Storage::Cuda(up_cuda) = &*up_storage else {
        candle_core::bail!("fast_mmq down_from_glu: up must live on CUDA");
    };

    let stream = dev.cuda_stream();
    let k_padded = pad(pad(k, MATRIX_ROW_PADDING), 4 * QK8_1);
    let blocks_per_row = k_padded / (4 * QK8_1);
    let workspace_main = batch_size * blocks_per_row * BLOCK_Q8_1_MMQ_SIZE;
    let workspace_extra = MMQ_X_MAX * BLOCK_Q8_1_MMQ_SIZE;
    let workspace_bytes = workspace_main + workspace_extra;
    let mut workspace = workspace_ensure(&MMQ_WORKSPACE, &dev, workspace_bytes, &stream)?;
    let (scratch_ptr, _scratch_guard) = workspace.ptr_mut();
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;

    let fixup_bytes = fixup_workspace_bytes(&dev);
    let mut fixup_workspace = workspace_ensure(&FIXUP_WORKSPACE, &dev, fixup_bytes, &stream)?;
    let (fixup_ptr, _fixup_guard) = fixup_workspace.ptr_mut();
    let fixup_ptr = fixup_ptr as *mut std::ffi::c_void;

    let run = DenseGluDownRun {
        down,
        gate: &gate,
        dev: &dev,
        stream: &stream,
        scratch_ptr,
        fixup_ptr,
        quantize: quantize_glu_launcher(ds_layout_for(dtype)),
        launcher: mmq_launcher(dtype).expect("supports() checked"),
        device_info: get_device_info(&dev),
        k,
        k_padded,
        batch_size,
        qk,
        type_x,
        activation: activation as i32,
    };
    match input_ty {
        DType::BF16 => run.launch(
            gate_cuda.as_cuda_slice::<half::bf16>()?,
            gate_layout.start_offset(),
            up_cuda.as_cuda_slice::<half::bf16>()?,
            up_layout.start_offset(),
        ),
        DType::F16 => run.launch(
            gate_cuda.as_cuda_slice::<half::f16>()?,
            gate_layout.start_offset(),
            up_cuda.as_cuda_slice::<half::f16>()?,
            up_layout.start_offset(),
        ),
        DType::F32 => run.launch(
            gate_cuda.as_cuda_slice::<f32>()?,
            gate_layout.start_offset(),
            up_cuda.as_cuda_slice::<f32>()?,
            up_layout.start_offset(),
        ),
        _ => unreachable!(),
    }
}

/// Compute one GGUF-quantized projection while preserving the input dtype.
pub fn plain(w: &QTensor, xs: &Tensor) -> Result<Tensor> {
    // titan: GGML Q1_0 prefill through the cuda-oxide llama.cpp MMQ port (not an upstream quant type)
    if w.dtype() == GgmlDType::Q1_0 {
        let Device::Cuda(dev) = w.device() else {
            candle_core::bail!("fast_mmq: weight must live on CUDA");
        };
        let (nrows, ncols) = w.shape().dims2()?;
        let (b_size, k) = match xs.dims() {
            [b, k] => (*b, *k),
            [b, m, k] => (*b * *m, *k),
            other => candle_core::bail!("fast_mmq: unexpected input rank {other:?}"),
        };
        if k != ncols {
            candle_core::bail!(
                "fast_mmq: shape mismatch — weight [{nrows}, {ncols}] vs input tail {k}"
            );
        }
        if b_size == 0 {
            candle_core::bail!("fast_mmq: batch size must be > 0");
        }
        return q1_0_plain(w, xs, &dev, nrows, k, b_size);
    }
    let mut outputs = shared_lhs(&[w], xs)?;
    Ok(outputs.pop().expect("one weight produces one output"))
}

/// Compute Q, K, and V projections with one activation quantization pass.
pub(crate) fn fused_qkv(
    q_w: &QTensor,
    k_w: &QTensor,
    v_w: &QTensor,
    xs: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let mut outputs = shared_lhs(&[q_w, k_w, v_w], xs)?;
    let v = outputs.pop().expect("three weights produce three outputs");
    let k = outputs.pop().expect("three weights produce three outputs");
    let q = outputs.pop().expect("three weights produce three outputs");
    Ok((q, k, v))
}

/// Compute gate and up projections with one activation quantization pass.
pub(crate) fn fused_glu(
    gate_w: &QTensor,
    up_w: &QTensor,
    xs: &Tensor,
    activation: GluActivationType,
) -> Result<Tensor> {
    if gate_w.shape() != up_w.shape() {
        candle_core::bail!(
            "fast_mmq fused_glu: gate/up shape mismatch {:?} vs {:?}",
            gate_w.shape(),
            up_w.shape()
        );
    }
    let mut outputs = shared_lhs(&[gate_w, up_w], xs)?;
    let up = outputs.pop().expect("two weights produce two outputs");
    let gate = outputs.pop().expect("two weights produce two outputs");
    crate::fused_glu(&gate, &up, activation)
}

pub(crate) fn fused_ffn(
    gate_w: &QTensor,
    up_w: &QTensor,
    down_w: &QTensor,
    xs: &Tensor,
    activation: GluActivationType,
) -> Result<Tensor> {
    if gate_w.shape() != up_w.shape() {
        candle_core::bail!(
            "fast_mmq fused_ffn: gate/up shape mismatch {:?} vs {:?}",
            gate_w.shape(),
            up_w.shape()
        );
    }
    let mut outputs = shared_lhs(&[gate_w, up_w], xs)?;
    let up = outputs.pop().expect("two weights produce two outputs");
    let gate = outputs.pop().expect("two weights produce two outputs");
    down_from_glu(down_w, &gate, &up, activation)
}

/// Run one GGUF-quantized MoE projection with llama.cpp-style grouped MMQ.
///
/// `ids_src` maps compact expert-sorted rows to input token rows. `ids_dst`
/// maps those same compact rows to output assignment rows. For Gemma4 MoE this
/// lets callers produce rows in flat assignment order for downstream grouped
/// MoE stages.
#[allow(clippy::too_many_arguments)]
pub fn grouped(
    weight: &QTensor,
    xs: &Tensor,
    ids_src: &CudaSlice<u32>,
    ids_dst: &CudaSlice<u32>,
    expert_bounds: &CudaSlice<u32>,
    total_assignments: usize,
    ncols_max: usize,
    num_experts: usize,
    dev: &CudaDevice,
) -> Result<Tensor> {
    let dtype = weight.dtype();
    if !supports(dtype) {
        candle_core::bail!("fast_mmq grouped: unsupported quant dtype {dtype:?}");
    }

    let (_, k) = xs.dims2()?;

    let (weight_experts, nrows, ncols) = weight.shape().dims3()?;
    if weight_experts != num_experts {
        candle_core::bail!(
            "fast_mmq grouped: expected {num_experts} experts, got {weight_experts}"
        );
    }
    if k != ncols {
        candle_core::bail!(
            "fast_mmq grouped: shape mismatch: weight cols {ncols} vs input tail {k}"
        );
    }
    let qk = qk_for(dtype);
    if k % qk != 0 {
        candle_core::bail!("fast_mmq grouped: k={k} not divisible by qk={qk}");
    }

    let input_ty = xs.dtype();
    if !matches!(input_ty, DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!(
            "fast_mmq grouped: input dtype must be BF16, F16, or F32, got {input_ty:?}"
        );
    }

    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        candle_core::bail!("fast_mmq grouped: input must live on CUDA");
    };
    let xs_offset = xs_layout.start_offset();
    let type_x = match input_ty {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 30,
        _ => unreachable!(),
    };

    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let k_padded = pad(pad(k, MATRIX_ROW_PADDING), 4 * QK8_1);

    let blocks_per_row = k_padded / (4 * QK8_1);
    let workspace_main = total_assignments * blocks_per_row * BLOCK_Q8_1_MMQ_SIZE;
    let workspace_extra = MMQ_X_MAX * BLOCK_Q8_1_MMQ_SIZE;
    let workspace_bytes = workspace_main + workspace_extra;
    let mut workspace = workspace_ensure(&MMQ_WORKSPACE, dev, workspace_bytes, &stream)?;
    let (scratch_ptr, _scratch_guard) = workspace.ptr_mut();
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;

    let fixup_bytes = fixup_workspace_bytes(dev);
    let mut fixup_workspace = workspace_ensure(&FIXUP_WORKSPACE, dev, fixup_bytes, &stream)?;
    let (fixup_ptr, _fixup_guard) = fixup_workspace.ptr_mut();
    let fixup_ptr = fixup_ptr as *mut std::ffi::c_void;

    let out = unsafe { dev.alloc::<f32>(total_assignments * nrows)? };

    let weight_ptr = weight.device_ptr()? as *const std::ffi::c_void;
    let stride_row_x = (k / qk) as i64;
    let stride_col_dst = nrows as i64;
    let di = get_device_info(dev);

    let quantize = quantize_launcher(ds_layout_for(dtype));
    let launcher = mmq_moe_launcher(dtype).expect("supports() checked");

    let (ids_src_ptr, _ids_src_guard) = slice_ptr(ids_src, 0);
    let (ids_dst_ptr, _ids_dst_guard) = slice_ptr(ids_dst, 0);
    let (bounds_ptr, _bounds_guard) = slice_ptr(expert_bounds, 0);
    let (out_ptr, _out_guard) = slice_ptr(&out, 0);

    unsafe {
        match input_ty {
            DType::BF16 => {
                let slice = xs_cuda.as_cuda_slice::<half::bf16>()?;
                let (xs_ptr, _xs_guard) = slice_ptr(slice, xs_offset);
                quantize(
                    xs_ptr as *const std::ffi::c_void,
                    ids_src_ptr as *const i32,
                    scratch_ptr,
                    type_x,
                    k as i64,
                    k as i64,
                    0,
                    0,
                    k_padded as i64,
                    total_assignments as i64,
                    1,
                    1,
                    stream_ptr,
                );
            }
            DType::F16 => {
                let slice = xs_cuda.as_cuda_slice::<half::f16>()?;
                let (xs_ptr, _xs_guard) = slice_ptr(slice, xs_offset);
                quantize(
                    xs_ptr as *const std::ffi::c_void,
                    ids_src_ptr as *const i32,
                    scratch_ptr,
                    type_x,
                    k as i64,
                    k as i64,
                    0,
                    0,
                    k_padded as i64,
                    total_assignments as i64,
                    1,
                    1,
                    stream_ptr,
                );
            }
            DType::F32 => {
                let slice = xs_cuda.as_cuda_slice::<f32>()?;
                let (xs_ptr, _xs_guard) = slice_ptr(slice, xs_offset);
                quantize(
                    xs_ptr as *const std::ffi::c_void,
                    ids_src_ptr as *const i32,
                    scratch_ptr,
                    type_x,
                    k as i64,
                    k as i64,
                    0,
                    0,
                    k_padded as i64,
                    total_assignments as i64,
                    1,
                    1,
                    stream_ptr,
                );
            }
            _ => unreachable!(),
        }

        launcher(
            fixup_ptr,
            weight_ptr,
            scratch_ptr as *const std::ffi::c_void,
            ids_dst_ptr as *const i32,
            bounds_ptr as *const i32,
            out_ptr as *mut std::ffi::c_void,
            k as i64,
            nrows as i64,
            total_assignments as i64,
            stride_row_x,
            stride_col_dst,
            num_experts as i64,
            ncols_max as i64,
            di.cc,
            di.nsm,
            di.smpbo,
            di.warp_size,
            stream_ptr,
        );
    }

    drop(_out_guard);
    drop(_bounds_guard);
    drop(_ids_dst_guard);
    drop(_ids_src_guard);

    let out_shape: Shape = vec![total_assignments, nrows].into();
    Ok(Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
        out_shape,
    )))
}

/// One device buffer of experts for [`grouped_parts`]: the dispatch's experts `first..first + count`,
/// stored back to back at `weight` (`[count, nrows, ncols]` of the projection's dtype).
pub struct GroupedPart {
    pub weight: u64,
    pub first: usize,
    pub count: usize,
    /// Most dispatch rows any of these experts has.
    pub ncols_max: usize,
}

/// [`grouped`] for an f32 input whose experts live in several device buffers (`parts`): the input rows
/// `ids_src` (in dispatch order) are quantized once, then one MMQ launch per part writes that part's
/// dispatch rows into `out` (`[.., nrows]` f32; dispatch row j goes to row `ids_dst[j]`).
#[allow(clippy::too_many_arguments)]
pub fn grouped_parts(
    dtype: GgmlDType,
    nrows: usize,
    xs: &Tensor,
    ids_src: &CudaSlice<u32>,
    ids_dst: &CudaSlice<u32>,
    expert_bounds: &CudaSlice<u32>,
    total_assignments: usize,
    parts: &[GroupedPart],
    out: &CudaSlice<f32>,
    dev: &CudaDevice,
) -> Result<()> {
    if !supports(dtype) {
        candle_core::bail!("fast_mmq grouped_parts: unsupported quant dtype {dtype:?}");
    }
    let (_, k) = xs.dims2()?;
    let qk = qk_for(dtype);
    if k % qk != 0 || xs.dtype() != DType::F32 {
        candle_core::bail!("fast_mmq grouped_parts: input {:?} {:?} (k must be a multiple of {qk}, f32)", xs.dims(), xs.dtype());
    }
    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        candle_core::bail!("fast_mmq grouped_parts: input must live on CUDA");
    };
    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let k_padded = pad(pad(k, MATRIX_ROW_PADDING), 4 * QK8_1);
    let blocks_per_row = k_padded / (4 * QK8_1);
    let workspace_bytes = total_assignments * blocks_per_row * BLOCK_Q8_1_MMQ_SIZE + MMQ_X_MAX * BLOCK_Q8_1_MMQ_SIZE;
    let mut workspace = workspace_ensure(&MMQ_WORKSPACE, dev, workspace_bytes, &stream)?;
    let (scratch_ptr, _scratch_guard) = workspace.ptr_mut();
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;
    let mut fixup_workspace = workspace_ensure(&FIXUP_WORKSPACE, dev, fixup_workspace_bytes(dev), &stream)?;
    let (fixup_ptr, _fixup_guard) = fixup_workspace.ptr_mut();
    let fixup_ptr = fixup_ptr as *mut std::ffi::c_void;
    let di = get_device_info(dev);
    let quantize = quantize_launcher(ds_layout_for(dtype));
    let launcher = mmq_moe_launcher(dtype).expect("supports() checked");
    let (xs_ptr, _xs_guard) = slice_ptr(xs_cuda.as_cuda_slice::<f32>()?, xs_layout.start_offset());
    let (ids_src_ptr, _src_guard) = slice_ptr(ids_src, 0);
    let (ids_dst_ptr, _dst_guard) = slice_ptr(ids_dst, 0);
    let (bounds_ptr, _bounds_guard) = slice_ptr(expert_bounds, 0);
    let (out_ptr, _out_guard) = slice_ptr(out, 0);
    unsafe {
        quantize(
            xs_ptr as *const std::ffi::c_void,
            ids_src_ptr as *const i32,
            scratch_ptr,
            0,
            k as i64,
            k as i64,
            0,
            0,
            k_padded as i64,
            total_assignments as i64,
            1,
            1,
            stream_ptr,
        );
        for p in parts.iter().filter(|p| p.count > 0 && p.ncols_max > 0) {
            launcher(
                fixup_ptr,
                p.weight as *const std::ffi::c_void,
                scratch_ptr as *const std::ffi::c_void,
                ids_dst_ptr as *const i32,
                (bounds_ptr as *const i32).add(p.first),
                out_ptr as *mut std::ffi::c_void,
                k as i64,
                nrows as i64,
                total_assignments as i64,
                (k / qk) as i64,
                nrows as i64,
                p.count as i64,
                p.ncols_max as i64,
                di.cc,
                di.nsm,
                di.smpbo,
                di.warp_size,
                stream_ptr,
            );
        }
    }
    Ok(())
}

struct GroupedGluRun<'a> {
    weight: &'a QTensor,
    gate: &'a Tensor,
    up: &'a Tensor,
    row_stride: usize,
    ids_src: Option<&'a CudaSlice<u32>>,
    ids_dst: &'a CudaSlice<u32>,
    expert_bounds: &'a CudaSlice<u32>,
    total_assignments: usize,
    ncols_max: usize,
    num_experts: usize,
    activation: i32,
    dev: &'a CudaDevice,
}

fn grouped_from_glu(run: GroupedGluRun<'_>) -> Result<Tensor> {
    let GroupedGluRun {
        weight,
        gate,
        up,
        row_stride,
        ids_src,
        ids_dst,
        expert_bounds,
        total_assignments,
        ncols_max,
        num_experts,
        activation,
        dev,
    } = run;
    let dtype = weight.dtype();
    if !supports(dtype) {
        candle_core::bail!("fast_mmq grouped_from_glu_pair: unsupported quant dtype {dtype:?}");
    }

    let (gate_rows, k) = gate.dims2()?;
    let (up_rows, up_k) = up.dims2()?;
    if gate_rows != total_assignments || up_rows != total_assignments || up_k != k {
        candle_core::bail!(
            "fast_mmq grouped_from_glu_pair: gate/up shape mismatch {:?} vs {:?}, total_assignments={total_assignments}",
            gate.shape(),
            up.shape()
        );
    }
    if gate.dtype() != DType::F32 || up.dtype() != DType::F32 {
        candle_core::bail!(
            "fast_mmq grouped_from_glu_pair: gate/up must be F32, got {:?} and {:?}",
            gate.dtype(),
            up.dtype()
        );
    }

    let (weight_experts, nrows, ncols) = weight.shape().dims3()?;
    if weight_experts != num_experts {
        candle_core::bail!(
            "fast_mmq grouped_from_glu_pair: expected {num_experts} experts, got {weight_experts}"
        );
    }
    if k != ncols {
        candle_core::bail!(
            "fast_mmq grouped_from_glu_pair: shape mismatch: weight cols {ncols} vs input tail {k}"
        );
    }
    let qk = qk_for(dtype);
    if k % qk != 0 {
        candle_core::bail!("fast_mmq grouped_from_glu_pair: k={k} not divisible by qk={qk}");
    }

    let (gate_storage, gate_layout) = gate.storage_and_layout();
    let Storage::Cuda(gate_cuda) = &*gate_storage else {
        candle_core::bail!("fast_mmq grouped_from_glu_pair: gate must live on CUDA");
    };
    let (up_storage, up_layout) = up.storage_and_layout();
    let Storage::Cuda(up_cuda) = &*up_storage else {
        candle_core::bail!("fast_mmq grouped_from_glu_pair: up must live on CUDA");
    };
    if gate_layout.stride() != [row_stride, 1] || up_layout.stride() != [row_stride, 1] {
        candle_core::bail!("fast_mmq grouped_from_glu_pair: invalid gate/up row stride");
    }

    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let k_padded = pad(pad(k, MATRIX_ROW_PADDING), 4 * QK8_1);

    let blocks_per_row = k_padded / (4 * QK8_1);
    let workspace_main = total_assignments * blocks_per_row * BLOCK_Q8_1_MMQ_SIZE;
    let workspace_extra = MMQ_X_MAX * BLOCK_Q8_1_MMQ_SIZE;
    let workspace_bytes = workspace_main + workspace_extra;
    let mut workspace = workspace_ensure(&MMQ_WORKSPACE, dev, workspace_bytes, &stream)?;
    let (scratch_ptr, _scratch_guard) = workspace.ptr_mut();
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;

    let fixup_bytes = fixup_workspace_bytes(dev);
    let mut fixup_workspace = workspace_ensure(&FIXUP_WORKSPACE, dev, fixup_bytes, &stream)?;
    let (fixup_ptr, _fixup_guard) = fixup_workspace.ptr_mut();
    let fixup_ptr = fixup_ptr as *mut std::ffi::c_void;

    let out = unsafe { dev.alloc::<f32>(total_assignments * nrows)? };

    let weight_ptr = weight.device_ptr()? as *const std::ffi::c_void;
    let stride_row_x = (k / qk) as i64;
    let stride_col_dst = nrows as i64;
    let di = get_device_info(dev);

    let quantize = quantize_glu_f32_launcher(ds_layout_for(dtype));
    let launcher = mmq_moe_launcher(dtype).expect("supports() checked");

    let gate_slice = gate_cuda.as_cuda_slice::<f32>()?;
    let up_slice = up_cuda.as_cuda_slice::<f32>()?;
    let (gate_ptr, _gate_guard) = slice_ptr(gate_slice, gate_layout.start_offset());
    let (up_ptr, _up_guard) = slice_ptr(up_slice, up_layout.start_offset());
    let (ids_src_ptr, _ids_src_guard) = match ids_src {
        Some(ids_src) => {
            let (ptr, guard) = slice_ptr(ids_src, 0);
            (ptr, Some(guard))
        }
        None => (0, None),
    };
    let (ids_dst_ptr, _ids_dst_guard) = slice_ptr(ids_dst, 0);
    let (bounds_ptr, _bounds_guard) = slice_ptr(expert_bounds, 0);
    let (out_ptr, _out_guard) = slice_ptr(&out, 0);

    unsafe {
        quantize(
            gate_ptr as *const f32,
            up_ptr as *const f32,
            ids_src_ptr as *const i32,
            scratch_ptr,
            k as i64,
            row_stride as i64,
            k_padded as i64,
            total_assignments as i64,
            activation,
            stream_ptr,
        );

        launcher(
            fixup_ptr,
            weight_ptr,
            scratch_ptr as *const std::ffi::c_void,
            ids_dst_ptr as *const i32,
            bounds_ptr as *const i32,
            out_ptr as *mut std::ffi::c_void,
            k as i64,
            nrows as i64,
            total_assignments as i64,
            stride_row_x,
            stride_col_dst,
            num_experts as i64,
            ncols_max as i64,
            di.cc,
            di.nsm,
            di.smpbo,
            di.warp_size,
            stream_ptr,
        );
    }

    drop(_out_guard);
    drop(_bounds_guard);
    drop(_ids_dst_guard);
    drop(_ids_src_guard);
    drop(_up_guard);
    drop(_gate_guard);

    let out_shape: Shape = vec![total_assignments, nrows].into();
    Ok(Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
        out_shape,
    )))
}

/// Run one grouped MoE projection after fusing `activation(gate) * up` directly
/// into the MMQ activation quantization layout.
#[allow(clippy::too_many_arguments)]
pub fn grouped_from_glu_pair(
    weight: &QTensor,
    gate: &Tensor,
    up: &Tensor,
    ids_src: &CudaSlice<u32>,
    ids_dst: &CudaSlice<u32>,
    expert_bounds: &CudaSlice<u32>,
    total_assignments: usize,
    ncols_max: usize,
    num_experts: usize,
    activation: i32,
    dev: &CudaDevice,
) -> Result<Tensor> {
    let gate = gate.contiguous()?;
    let up = up.contiguous()?;
    let row_stride = gate.dim(1)?;
    grouped_from_glu(GroupedGluRun {
        weight,
        gate: &gate,
        up: &up,
        row_stride,
        ids_src: Some(ids_src),
        ids_dst,
        expert_bounds,
        total_assignments,
        ncols_max,
        num_experts,
        activation,
        dev,
    })
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn grouped_from_glu_sorted_pair(
    weight: &QTensor,
    gate: &Tensor,
    up: &Tensor,
    ids_dst: &CudaSlice<u32>,
    expert_bounds: &CudaSlice<u32>,
    total_assignments: usize,
    ncols_max: usize,
    num_experts: usize,
    activation: i32,
    dev: &CudaDevice,
) -> Result<Tensor> {
    let gate = gate.contiguous()?;
    let up = up.contiguous()?;
    let row_stride = gate.dim(1)?;
    grouped_from_glu(GroupedGluRun {
        weight,
        gate: &gate,
        up: &up,
        row_stride,
        ids_src: None,
        ids_dst,
        expert_bounds,
        total_assignments,
        ncols_max,
        num_experts,
        activation,
        dev,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn grouped_from_glu_packed(
    weight: &QTensor,
    gate_up: &Tensor,
    ids_src: &CudaSlice<u32>,
    ids_dst: &CudaSlice<u32>,
    expert_bounds: &CudaSlice<u32>,
    total_assignments: usize,
    ncols_max: usize,
    num_experts: usize,
    activation: i32,
    dev: &CudaDevice,
) -> Result<Tensor> {
    let gate_up = gate_up.contiguous()?;
    let (_, _, k) = weight.shape().dims3()?;
    if gate_up.dims2()? != (total_assignments, 2 * k) {
        candle_core::bail!("fast_mmq grouped_from_glu_packed: gate/up shape mismatch");
    }
    let gate = gate_up.narrow(1, 0, k)?;
    let up = gate_up.narrow(1, k, k)?;
    grouped_from_glu(GroupedGluRun {
        weight,
        gate: &gate,
        up: &up,
        row_stride: 2 * k,
        ids_src: Some(ids_src),
        ids_dst,
        expert_bounds,
        total_assignments,
        ncols_max,
        num_experts,
        activation,
        dev,
    })
}

/// Run two GGUF-quantized MoE projections with llama.cpp-style grouped MMQ.
///
/// Gate/up share one MMQ activation quantization pass and one packed output.
#[allow(clippy::too_many_arguments)]
pub fn grouped_pair_packed(
    gate: &QTensor,
    up: &QTensor,
    xs: &Tensor,
    ids_src: &CudaSlice<u32>,
    ids_dst: &CudaSlice<u32>,
    expert_bounds: &CudaSlice<u32>,
    total_assignments: usize,
    topk: usize,
    num_experts: usize,
    dev: &CudaDevice,
) -> Result<Tensor> {
    let dtype = gate.dtype();
    if dtype != up.dtype() {
        candle_core::bail!(
            "fast_mmq grouped_pair requires matching gate/up dtypes, got {:?} and {:?}",
            dtype,
            up.dtype()
        );
    }
    if !supports(dtype) {
        candle_core::bail!("fast_mmq grouped_pair: unsupported quant dtype {dtype:?}");
    }

    let (num_tokens, k) = xs.dims2()?;
    if total_assignments != num_tokens * topk {
        candle_core::bail!(
            "fast_mmq grouped_pair: total_assignments={total_assignments} does not match num_tokens={num_tokens} * topk={topk}"
        );
    }

    let (gate_experts, nrows, ncols) = gate.shape().dims3()?;
    let (up_experts, up_nrows, up_ncols) = up.shape().dims3()?;
    if gate_experts != num_experts || up_experts != num_experts {
        candle_core::bail!(
            "fast_mmq grouped_pair: expected {num_experts} experts, got gate={gate_experts} up={up_experts}"
        );
    }
    if nrows != up_nrows || ncols != up_ncols {
        candle_core::bail!(
            "fast_mmq grouped_pair: gate/up shape mismatch {:?} vs {:?}",
            gate.shape(),
            up.shape()
        );
    }
    if k != ncols {
        candle_core::bail!(
            "fast_mmq grouped_pair: shape mismatch: weight cols {ncols} vs input tail {k}"
        );
    }
    let qk = qk_for(dtype);
    if k % qk != 0 {
        candle_core::bail!("fast_mmq grouped_pair: k={k} not divisible by qk={qk}");
    }

    let input_ty = xs.dtype();
    if !matches!(input_ty, DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!(
            "fast_mmq grouped_pair: input dtype must be BF16, F16, or F32, got {input_ty:?}"
        );
    }

    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        candle_core::bail!("fast_mmq grouped_pair: input must live on CUDA");
    };
    let xs_offset = xs_layout.start_offset();
    let type_x = match input_ty {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 30,
        _ => unreachable!(),
    };

    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let k_padded = pad(pad(k, MATRIX_ROW_PADDING), 4 * QK8_1);

    let blocks_per_row = k_padded / (4 * QK8_1);
    let workspace_main = total_assignments * blocks_per_row * BLOCK_Q8_1_MMQ_SIZE;
    let workspace_extra = MMQ_X_MAX * BLOCK_Q8_1_MMQ_SIZE;
    let workspace_bytes = workspace_main + workspace_extra;
    let mut workspace = workspace_ensure(&MMQ_WORKSPACE, dev, workspace_bytes, &stream)?;
    let (scratch_ptr, _scratch_guard) = workspace.ptr_mut();
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;

    let fixup_bytes = fixup_workspace_bytes(dev);
    let mut fixup_workspace = workspace_ensure(&FIXUP_WORKSPACE, dev, fixup_bytes, &stream)?;
    let (fixup_ptr, _fixup_guard) = fixup_workspace.ptr_mut();
    let fixup_ptr = fixup_ptr as *mut std::ffi::c_void;

    let output = unsafe { dev.alloc::<f32>(total_assignments * nrows * 2)? };

    let gate_ptr = gate.device_ptr()? as *const std::ffi::c_void;
    let up_ptr = up.device_ptr()? as *const std::ffi::c_void;
    let stride_row_x = (k / qk) as i64;
    let stride_col_dst = (2 * nrows) as i64;
    let di = get_device_info(dev);

    let quantize = quantize_launcher(ds_layout_for(dtype));
    let launcher = mmq_moe_launcher(dtype).expect("supports() checked");

    let (ids_src_ptr, _ids_src_guard) = slice_ptr(ids_src, 0);
    let (ids_dst_ptr, _ids_dst_guard) = slice_ptr(ids_dst, 0);
    let (bounds_ptr, _bounds_guard) = slice_ptr(expert_bounds, 0);
    let (gate_out_ptr, _gate_out_guard) = slice_ptr(&output, 0);
    let (up_out_ptr, _up_out_guard) = slice_ptr(&output, nrows);

    unsafe {
        match input_ty {
            DType::BF16 => {
                let slice = xs_cuda.as_cuda_slice::<half::bf16>()?;
                let (xs_ptr, _xs_guard) = slice_ptr(slice, xs_offset);
                quantize(
                    xs_ptr as *const std::ffi::c_void,
                    ids_src_ptr as *const i32,
                    scratch_ptr,
                    type_x,
                    k as i64,
                    k as i64,
                    0,
                    0,
                    k_padded as i64,
                    total_assignments as i64,
                    1,
                    1,
                    stream_ptr,
                );
            }
            DType::F16 => {
                let slice = xs_cuda.as_cuda_slice::<half::f16>()?;
                let (xs_ptr, _xs_guard) = slice_ptr(slice, xs_offset);
                quantize(
                    xs_ptr as *const std::ffi::c_void,
                    ids_src_ptr as *const i32,
                    scratch_ptr,
                    type_x,
                    k as i64,
                    k as i64,
                    0,
                    0,
                    k_padded as i64,
                    total_assignments as i64,
                    1,
                    1,
                    stream_ptr,
                );
            }
            DType::F32 => {
                let slice = xs_cuda.as_cuda_slice::<f32>()?;
                let (xs_ptr, _xs_guard) = slice_ptr(slice, xs_offset);
                quantize(
                    xs_ptr as *const std::ffi::c_void,
                    ids_src_ptr as *const i32,
                    scratch_ptr,
                    type_x,
                    k as i64,
                    k as i64,
                    0,
                    0,
                    k_padded as i64,
                    total_assignments as i64,
                    1,
                    1,
                    stream_ptr,
                );
            }
            _ => unreachable!(),
        }

        for (weight_ptr, out_ptr) in [
            (gate_ptr, gate_out_ptr as *mut std::ffi::c_void),
            (up_ptr, up_out_ptr as *mut std::ffi::c_void),
        ] {
            launcher(
                fixup_ptr,
                weight_ptr,
                scratch_ptr as *const std::ffi::c_void,
                ids_dst_ptr as *const i32,
                bounds_ptr as *const i32,
                out_ptr,
                k as i64,
                nrows as i64,
                total_assignments as i64,
                stride_row_x,
                stride_col_dst,
                num_experts as i64,
                num_tokens as i64,
                di.cc,
                di.nsm,
                di.smpbo,
                di.warp_size,
                stream_ptr,
            );
        }
    }

    drop(_gate_out_guard);
    drop(_up_out_guard);
    drop(_bounds_guard);
    drop(_ids_dst_guard);
    drop(_ids_src_guard);

    let out_shape: Shape = vec![total_assignments, 2 * nrows].into();
    Ok(Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, dev.clone())),
        out_shape,
    )))
}

/// Run two GGUF-quantized MoE projections with llama.cpp-style grouped MMQ.
#[allow(clippy::too_many_arguments)]
pub fn grouped_pair(
    gate: &QTensor,
    up: &QTensor,
    xs: &Tensor,
    ids_src: &CudaSlice<u32>,
    ids_dst: &CudaSlice<u32>,
    expert_bounds: &CudaSlice<u32>,
    total_assignments: usize,
    topk: usize,
    num_experts: usize,
    dev: &CudaDevice,
) -> Result<(Tensor, Tensor)> {
    let output = grouped_pair_packed(
        gate,
        up,
        xs,
        ids_src,
        ids_dst,
        expert_bounds,
        total_assignments,
        topk,
        num_experts,
        dev,
    )?;
    let (_, nrows, _) = gate.shape().dims3()?;
    let gate = output.narrow(1, 0, nrows)?.contiguous()?;
    let up = output.narrow(1, nrows, nrows)?.contiguous()?;
    Ok((gate, up))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::fast_mmvq;
    use candle_core::quantized::GgmlDType;

    const BATCH: usize = 3;
    const ROWS: usize = 4;
    const HIDDEN: usize = 256;
    const INTERMEDIATE: usize = 512;
    const TOLERANCE: f32 = 5e-3;

    fn patterned(shape: impl Into<Shape>, salt: usize, scale: f32) -> Result<Tensor> {
        let shape = shape.into();
        let values = (0..shape.elem_count())
            .map(|index| {
                let value = (index.wrapping_mul(37) + salt.wrapping_mul(19)) % 211;
                (value as f32 / 105.0 - 1.0) * scale
            })
            .collect::<Vec<_>>();
        Tensor::from_vec(values, shape, &Device::Cpu)
    }

    fn assert_close(actual: &Tensor, expected: &Tensor) -> Result<()> {
        assert_eq!(actual.dims(), expected.dims());
        assert_eq!(actual.dtype(), expected.dtype());
        let actual = actual
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = expected
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!(actual.iter().all(|value| value.is_finite()));
        assert!(expected.iter().all(|value| value.is_finite()));
        let max_error = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs() / (1.0 + expected.abs()))
            .fold(0.0f32, f32::max);
        assert!(
            max_error <= TOLERANCE,
            "relative error {max_error} exceeds {TOLERANCE}"
        );
        Ok(())
    }

    #[test]
    fn dense_q4k_shared_lhs_matches_independent_projections() -> Result<()> {
        let cuda = Device::new_cuda(0)?;
        let weights = [11, 29, 47]
            .into_iter()
            .map(|salt| {
                QTensor::quantize_onto(
                    &patterned((INTERMEDIATE, HIDDEN), salt, 0.03)?,
                    GgmlDType::Q4K,
                    &cuda,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        for dtype in [DType::F16, DType::BF16, DType::F32] {
            let xs = patterned((1, BATCH, ROWS, HIDDEN), 3, 0.2)?
                .to_dtype(dtype)?
                .to_device(&cuda)?;
            let refs = weights.iter().collect::<Vec<_>>();
            let fused = shared_lhs(&refs, &xs)?;
            for (weight, actual) in weights.iter().zip(fused) {
                assert_close(&actual, &plain(weight, &xs)?)?;
            }
        }
        Ok(())
    }

    #[test]
    fn mmvq_q4k_rank4_unequal_qkv_matches_independent() -> Result<()> {
        let cuda = Device::new_cuda(0)?;
        let q =
            QTensor::quantize_onto(&patterned((384, HIDDEN), 11, 0.03)?, GgmlDType::Q4K, &cuda)?;
        let k =
            QTensor::quantize_onto(&patterned((256, HIDDEN), 29, 0.03)?, GgmlDType::Q4K, &cuda)?;
        let v =
            QTensor::quantize_onto(&patterned((128, HIDDEN), 47, 0.03)?, GgmlDType::Q4K, &cuda)?;
        let xs = patterned((1, 2, 2, HIDDEN), 3, 0.2)?
            .to_dtype(DType::BF16)?
            .to_device(&cuda)?;

        let (q_out, k_out, v_out) = fast_mmvq::fused_qkv(&q, &k, &v, &xs)?;
        assert_close(&q_out, &fast_mmvq::plain(&q, &xs)?)?;
        assert_close(&k_out, &fast_mmvq::plain(&k, &xs)?)?;
        assert_close(&v_out, &fast_mmvq::plain(&v, &xs)?)
    }

    #[test]
    fn dense_q4k_glu_down_matches_materialized_path() -> Result<()> {
        let cuda = Device::new_cuda(0)?;
        let gate = QTensor::quantize_onto(
            &patterned((INTERMEDIATE, HIDDEN), 11, 0.03)?,
            GgmlDType::Q4K,
            &cuda,
        )?;
        let up = QTensor::quantize_onto(
            &patterned((INTERMEDIATE, HIDDEN), 29, 0.03)?,
            GgmlDType::Q4K,
            &cuda,
        )?;
        let down = QTensor::quantize_onto(
            &patterned((HIDDEN, INTERMEDIATE), 47, 0.03)?,
            GgmlDType::Q4K,
            &cuda,
        )?;

        for dtype in [DType::F16, DType::BF16, DType::F32] {
            let xs = patterned((BATCH, ROWS, HIDDEN), 3, 0.2)?
                .to_dtype(dtype)?
                .to_device(&cuda)?;
            for activation in [
                GluActivationType::Silu,
                GluActivationType::Gelu,
                GluActivationType::Relu,
                GluActivationType::GeluErf,
                GluActivationType::Sigmoid,
            ] {
                let mut pair = shared_lhs(&[&gate, &up], &xs)?;
                let up_out = pair.pop().unwrap();
                let gate_out = pair.pop().unwrap();
                let intermediate = crate::fused_glu(&gate_out, &up_out, activation)?;
                let expected = plain(&down, &intermediate)?;
                let actual = fused_ffn(&gate, &up, &down, &xs, activation)?;
                assert_close(&actual, &expected)?;
            }
        }
        Ok(())
    }

    #[test]
    fn dense_glu_down_covers_scale_layouts() -> Result<()> {
        let cuda = Device::new_cuda(0)?;
        let xs = patterned((BATCH, ROWS, HIDDEN), 3, 0.2)?
            .to_dtype(DType::BF16)?
            .to_device(&cuda)?;

        for quant in [GgmlDType::Q6K, GgmlDType::Q2K] {
            let gate = QTensor::quantize_onto(
                &patterned((INTERMEDIATE, HIDDEN), 11, 0.03)?,
                quant,
                &cuda,
            )?;
            let up = QTensor::quantize_onto(
                &patterned((INTERMEDIATE, HIDDEN), 29, 0.03)?,
                quant,
                &cuda,
            )?;
            let down = QTensor::quantize_onto(
                &patterned((HIDDEN, INTERMEDIATE), 47, 0.03)?,
                quant,
                &cuda,
            )?;
            let mut pair = shared_lhs(&[&gate, &up], &xs)?;
            let up_out = pair.pop().unwrap();
            let gate_out = pair.pop().unwrap();
            let intermediate = crate::fused_glu(&gate_out, &up_out, GluActivationType::Silu)?;
            let expected = plain(&down, &intermediate)?;
            let actual = fused_ffn(&gate, &up, &down, &xs, GluActivationType::Silu)?;
            assert_close(&actual, &expected)?;
        }
        Ok(())
    }
}
