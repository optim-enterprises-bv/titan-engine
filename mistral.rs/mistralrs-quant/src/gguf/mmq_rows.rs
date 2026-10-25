//! MMQ-exact matmul for a few activation columns (titan-engine speculative verification).
//!
//! `fast_mmq` runs 1..=8 columns as one x8 MMQ tile, so a column's result does not depend on how many columns
//! share the launch; verification relies on that. These kernels reproduce that tile's output bit for bit (each
//! output's f32 chain over its stream-k segments, and the fixup's combination of them) at matvec cost, so a
//! 2-column verification no longer pays for a mostly empty tensor-core tile. Gated against the MMQ launcher by
//! examples/mmq_rows_gate.rs. Regenerate mmq_rows_oxide.ptx with oxide-kernels/mmq-rows/export_ptx.sh.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use candle_core::{
    cuda::{
        cudarc::driver::{sys::CUfunction_attribute, CudaFunction, CudaStream, LaunchConfig, PushKernelArg},
        CudaDevice, DeviceId, WrapErr,
    },
    quantized::{GgmlDType, QTensor},
    Result,
};

const PTX: &str = include_str!("mmq_rows_oxide.ptx");
const MODULE: &str = "titan_mmq_rows";
/// MMQ's tile height (mmq_y on Volta+), the kernels' rows per block, and the most segments a block holds.
const MMQ_Y: usize = 128;
const ROWS_PER_BLOCK: usize = 32;
const MAX_SEGMENTS: usize = 32;
pub const MAX_COLS: usize = 8;
/// MMQ's stream-k launch policy: one block per tile when the tiles fill this share of the last wave.
const TILE_EFFICIENCY_PERCENT: usize = 90;
/// Columns up to which `fast_mmq` takes these kernels (`TITAN_MMQ_ROWS`, 0 disables), and the fewest weight rows:
/// past 3 columns, or below ~4k rows (too few warps to fill the GPU), MMQ's tensor cores win (L2-cold, sm_120).
const DEFAULT_MAX_COLS: usize = 3;
const MIN_ROWS: usize = 4096;

/// Kernel name suffix and staged 32-bit words per weight row (the kernels' RS_Q4 / RS_Q5). Q6_K's 2-byte aligned
/// blocks do not stage at speed, and MMQ already gives the same bits for it.
fn fmt_info(dtype: GgmlDType) -> Option<(&'static str, usize)> {
    Some(match dtype {
        GgmlDType::Q4K => ("q4_k", 37),
        GgmlDType::Q5K => ("q5_k", 45),
        _ => return None,
    })
}
/// The kernels' KB_PER_STAGE: k blocks each warp stages per round.
const KB_PER_STAGE: usize = 2;

/// `max_cols` as set by `set_max_cols` (the gate's MMQ side); `usize::MAX` = unset.
static MAX_COLS_OVERRIDE: AtomicUsize = AtomicUsize::new(usize::MAX);

fn max_cols() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    match MAX_COLS_OVERRIDE.load(Ordering::Relaxed) {
        usize::MAX => *N.get_or_init(|| {
            crate::titan_cfg::var("TITAN_MMQ_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_MAX_COLS).min(MAX_COLS)
        }),
        n => n,
    }
}

/// Override the column limit (0 sends every launch to MMQ) and lift MIN_ROWS; `usize::MAX` restores the defaults.
/// For gates.
pub fn set_max_cols(n: usize) {
    MAX_COLS_OVERRIDE.store(n, Ordering::Relaxed);
}

type FnCache = Mutex<HashMap<(DeviceId, GgmlDType, usize), CudaFunction>>;
static FNS: OnceLock<FnCache> = OnceLock::new();

fn func(dev: &CudaDevice, dtype: GgmlDType, name: &str, ncols: usize, smem_optin: usize) -> Result<CudaFunction> {
    let key = (dev.id(), dtype, ncols);
    let mut map = FNS.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap();
    if let Some(f) = map.get(&key) {
        return Ok(f.clone());
    }
    let f = dev.get_or_load_custom_func(&format!("mmq_rows_{name}_nc{ncols}"), MODULE, PTX)?.into_cuda_function();
    f.set_attribute(CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_optin as i32).w()?;
    map.insert(key, f.clone());
    Ok(f)
}

fn kstart(b: usize, total: usize, grid: usize) -> usize {
    b * total / grid
}

/// The stream-k block holding k block `kbc` (the last one starting at or before it).
fn block_of(kbc: usize, total: usize, grid: usize) -> usize {
    let mut b = kbc * grid / total;
    while b + 1 < grid && kstart(b + 1, total, grid) <= kbc {
        b += 1;
    }
    while b > 0 && kstart(b, total, grid) > kbc {
        b -= 1;
    }
    b
}

/// The most stream-k blocks any tile spans.
fn max_segments(nty: usize, bpn: usize, grid: usize) -> usize {
    static CACHE: OnceLock<Mutex<HashMap<(usize, usize, usize), usize>>> = OnceLock::new();
    let mut map = CACHE.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap();
    *map.entry((nty, bpn, grid)).or_insert_with(|| {
        let total = nty * bpn;
        (0..nty)
            .map(|t| block_of(t * bpn + bpn - 1, total, grid) - block_of(t * bpn, total, grid) + 1)
            .max()
            .unwrap_or(1)
    })
}

/// The output `fast_mmq`'s launcher writes for `w` and `ncols` activation columns already quantized at `y`
/// (block_q8_1_mmq), when these kernels cover it; `false` leaves the launch to MMQ.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_launch(
    w: &QTensor,
    w_ptr: u64,
    y_ptr: u64,
    out_ptr: u64,
    ncols: usize,
    type_dst: i32,
    nsm: usize,
    smem_optin: usize,
    dev: &CudaDevice,
    stream: &CudaStream,
) -> Result<bool> {
    let Some((name, row_words)) = fmt_info(w.dtype()) else {
        return Ok(false);
    };
    if ncols == 0 || ncols > max_cols() {
        return Ok(false);
    }
    let (nrows, k) = w.shape().dims2()?;
    if nrows < MIN_ROWS && MAX_COLS_OVERRIDE.load(Ordering::Relaxed) == usize::MAX {
        return Ok(false);
    }
    let bpn = k / w.dtype().block_size();
    let nty = nrows.div_ceil(MMQ_Y);
    let waves = nty.div_ceil(nsm);
    let grid = if 100 * nty / (nsm * waves) >= TILE_EFFICIENCY_PERCENT { nty } else { nsm };
    let segments = max_segments(nty, bpn, grid);
    // [segment][column][lane] partials, then each segment warp's staged rows
    let smem = segments * ROWS_PER_BLOCK * (ncols + KB_PER_STAGE * row_words) * std::mem::size_of::<f32>();
    if segments > MAX_SEGMENTS || smem > smem_optin {
        return Ok(false);
    }
    let f = func(dev, w.dtype(), name, ncols, smem_optin)?;
    let cfg = LaunchConfig {
        grid_dim: (nrows.div_ceil(ROWS_PER_BLOCK) as u32, 1, 1),
        block_dim: (ROWS_PER_BLOCK as u32, segments as u32, 1),
        shared_mem_bytes: smem as u32,
    };
    let (nrows_i, bpn_i, ncols_i, total_i, grid_i) =
        (nrows as i32, bpn as i32, ncols as i32, (nty * bpn) as i32, grid as i32);
    let mut b = stream.launch_builder(&f);
    b.arg(&w_ptr)
        .arg(&y_ptr)
        .arg(&out_ptr)
        .arg(&nrows_i)
        .arg(&bpn_i)
        .arg(&bpn_i)
        .arg(&ncols_i)
        .arg(&total_i)
        .arg(&grid_i)
        .arg(&type_dst);
    unsafe { b.launch(cfg) }.w()?;
    Ok(true)
}

/// JIT the module and resolve every kernel now, not inside the first verification (or a graph capture).
pub fn warm_up(dev: &CudaDevice) -> Result<()> {
    let smem_optin = dev.cuda_stream().context().attribute(
        candle_core::cuda::cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
    ).w()? as usize;
    for dtype in [GgmlDType::Q4K, GgmlDType::Q5K] {
        let (name, _) = fmt_info(dtype).expect("covered format");
        for ncols in 1..=MAX_COLS {
            func(dev, dtype, name, ncols, smem_optin)?;
        }
    }
    Ok(())
}

