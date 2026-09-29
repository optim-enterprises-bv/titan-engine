//! Tensor-core flash attention for CUDA prompt chunks (titan-engine, `TITAN_ATTN_FLASH_PREFILL`).
//!
//! Causal attention of a chunk's query rows `q` (1, h, s, 256) over the KV cache `k`, `v` (1, kvh, >= past + s,
//! 256), row `i` seeing keys `< past + i + 1`, in one launch (plus a split merge): no score matrix, no mask
//! tensor, so its cost and memory stay flat per chunk as the context grows. The kernels are cuda-oxide
//! (`titan-engine/oxide-kernels/flash-prefill`, llama.cpp fattn-mma numerics: f32 scores, bf16 probabilities),
//! embedded as PTX; regenerate flash_prefill_oxide.ptx with its export_ptx.sh.
//!
//! Chunks from `TITAN_ATTN_FLASH_PREFILL_MIN` keys (default 1024, like flash-decode) take it; shorter prompts
//! keep the eager path and its exact results. `TITAN_ATTN_FLASH_PREFILL=0` turns it off.
//!
//! `TITAN_ATTN_FLASH_PREFILL_KERNEL` picks the kernel: `w8` (default: Q in registers, cp.async K/V overlapped
//! with the MMAs, 8 warps, 16 positions per block), `v2` (the same with 4 warps and v1's split ranges, so v1's
//! bits exactly), `qg` / `qgw8` (Q fragments from L1), `v1` (the first kernel, synchronous tile loads). With one
//! split all of them give v1's bits; `w8` splits the keys differently, which moves the f32 rounding only.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::{
    cuda::{
        cudarc::driver::{CudaFunction, DevicePtr, LaunchConfig, PushKernelArg},
        CudaDevice, DeviceId, WrapErr,
    },
    CudaStorage, DType, Device, Result, Shape, Storage, Tensor,
};

use super::SdpaParams;

const PTX: &str = include_str!("flash_prefill_oxide.ptx");
const MODULE: &str = "titan_flash_prefill";
const HEAD_DIM: usize = 256;
/// Query heads per KV head the kernel packs into a block.
const N_REP: usize = 8;
/// Kernels (name, warps per block; a block takes 2 positions per warp) and the split merge.
const KERNELS: [(&str, &str, usize); 5] = [
    ("w8", "flash_prefill_w8", 8),
    ("v2", "flash_prefill", 4),
    ("qg", "flash_prefill_qg", 4),
    ("qgw8", "flash_prefill_qg_w8", 8),
    ("v1", "flash_prefill_v1", 4),
];
const COMBINE: usize = KERNELS.len();
/// Keys per tile.
const TILE: usize = 32;
const DEFAULT_MIN_KV: usize = 1024;
/// Default `TITAN_ATTN_FLASH_PREFILL_UNITS`: blocks to aim for (split-KV fills in what the positions leave):
/// ~4 waves at 2 blocks per SM on a 60-SM part.
const DEFAULT_UNITS: usize = 480;
/// Fewest key tiles a split gets.
const MIN_SPLIT_TILES: usize = 8;
const MAX_SPLITS: usize = 16;

struct Cfg {
    on: bool,
    min_kv: usize,
    units: usize,
    /// index into `KERNELS`
    kernel: usize,
}

fn cfg() -> &'static Cfg {
    static C: OnceLock<Cfg> = OnceLock::new();
    C.get_or_init(|| {
        let env = |n: &str, d: usize| std::env::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let on = std::env::var("TITAN_ATTN_FLASH_PREFILL").map(|v| v != "0").unwrap_or(true);
        let min_kv = env("TITAN_ATTN_FLASH_PREFILL_MIN", DEFAULT_MIN_KV).max(1);
        let units = env("TITAN_ATTN_FLASH_PREFILL_UNITS", DEFAULT_UNITS).max(1);
        let want = std::env::var("TITAN_ATTN_FLASH_PREFILL_KERNEL").unwrap_or_else(|_| "w8".into());
        let kernel = KERNELS.iter().position(|k| k.0 == want).unwrap_or(0);
        tracing::info!(
            "titan attention: flash-prefill {} from {min_kv} keys, {units} blocks target, kernel {} ({} warps)",
            if on { "on" } else { "off" },
            KERNELS[kernel].1,
            KERNELS[kernel].2
        );
        Cfg { on, min_kv, units, kernel }
    })
}

type FnCache = Mutex<HashMap<(DeviceId, usize), CudaFunction>>;
static FNS: OnceLock<FnCache> = OnceLock::new();

fn func(dev: &CudaDevice, idx: usize) -> Result<CudaFunction> {
    let key = (dev.id(), idx);
    let mut map = FNS.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap();
    if let Some(f) = map.get(&key) {
        return Ok(f.clone());
    }
    let name = if idx == COMBINE { "flash_prefill_combine" } else { KERNELS[idx].1 };
    let f = dev.get_or_load_custom_func(name, MODULE, PTX)?.into_cuda_function();
    map.insert(key, f.clone());
    Ok(f)
}

/// Whether a prompt chunk of `rows` query rows ending at `kv_len` keys takes the kernel (the model shapes
/// aside, which `supported` checks per call).
pub(crate) fn wanted(rows: usize, kv_len: usize) -> bool {
    let c = cfg();
    c.on && rows > 1 && kv_len >= c.min_kv
}

/// Shape, dtype and layout support: CUDA, batch 1, bf16, head_dim 256, 8 query heads per KV head, contiguous
/// q, K/V rows contiguous and 16-byte aligned, no softcap, sliding window or sinks.
pub(crate) fn supported(q: &Tensor, k: &Tensor, v: &Tensor, p: &SdpaParams) -> bool {
    let (Ok((b, h, s, d)), Ok((kb, kvh, kl, kd)), Ok((vb, vh, vl, vd))) = (q.dims4(), k.dims4(), v.dims4()) else {
        return false;
    };
    let rows_ok = |t: &Tensor| {
        let st = t.stride();
        st[3] == 1 && st[2] == HEAD_DIM && st[1] % 8 == 0 && t.layout().start_offset() % 8 == 0
    };
    q.device().is_cuda()
        && [q.dtype(), k.dtype(), v.dtype()].iter().all(|t| *t == DType::BF16)
        && (b, kb, vb) == (1, 1, 1)
        && d == HEAD_DIM
        && kd == HEAD_DIM
        && vd == HEAD_DIM
        && kvh == vh
        && kl == vl
        && kl >= s
        && kvh > 0
        && h == kvh * N_REP
        && p.n_kv_groups == N_REP
        && p.softcap.is_none()
        && p.sliding_window.is_none()
        && p.sinks.is_none()
        && rows_ok(k)
        && rows_ok(v)
        && !mistralrs_quant::distributed::use_nccl()
}

fn dptr(t: &Tensor, stream: &std::sync::Arc<candle_core::cuda::cudarc::driver::CudaStream>) -> Result<u64> {
    let (st, l) = t.storage_and_layout();
    let Storage::Cuda(c) = &*st else {
        candle_core::bail!("flash-prefill: tensor must live on CUDA");
    };
    let s = c.as_cuda_slice::<half::bf16>()?;
    let (p, _g) = s.device_ptr(stream);
    Ok(p + (l.start_offset() * 2) as u64)
}

/// Causal attention of the last `s` positions: `q` (1, h, s, 256), `k`, `v` (1, kvh, kv_len, 256) with the
/// queries at positions `kv_len - s ..`. Returns (1, h, s, 256) bf16.
pub(crate) fn attend(q: &Tensor, k: &Tensor, v: &Tensor, p: &SdpaParams) -> Result<Tensor> {
    let Device::Cuda(dev) = q.device() else {
        candle_core::bail!("flash-prefill: CUDA only");
    };
    let q = q.contiguous()?;
    if !supported(&q, k, v, p) {
        candle_core::bail!("flash-prefill: unsupported q {:?} k {:?} v {:?}", q.layout(), k.layout(), v.layout());
    }
    let (_, h, s, _) = q.dims4()?;
    let (_, kvh, kv_len, _) = k.dims4()?;
    let past = kv_len - s;
    let stream = dev.cuda_stream();
    let (q_ptr, k_ptr, v_ptr) = (dptr(&q, &stream)?, dptr(k, &stream)?, dptr(v, &stream)?);
    let (k_hs, v_hs) = (k.stride()[1] as u64, v.stride()[1] as u64);
    if (k_ptr | v_ptr | q_ptr) % 16 != 0 {
        candle_core::bail!("flash-prefill: q/K/V not 16-byte aligned");
    }
    let kernel = cfg().kernel;
    let nw = KERNELS[kernel].2;
    let npb = s.div_ceil(2 * nw);
    let tiles = kv_len.div_ceil(TILE);
    let nsplit = cfg()
        .units
        .div_ceil(npb * kvh)
        .min(tiles.div_ceil(MIN_SPLIT_TILES))
        .clamp(1, MAX_SPLITS);
    let out = unsafe { dev.alloc::<half::bf16>(h * s * HEAD_DIM)? };
    let out_ptr = {
        let (p, _g) = out.device_ptr(&stream);
        p
    };
    let (part, ml) = if nsplit > 1 {
        (
            Some(unsafe { dev.alloc::<f32>(nsplit * h * s * HEAD_DIM)? }),
            Some(unsafe { dev.alloc::<f32>(nsplit * h * s * 2)? }),
        )
    } else {
        (None, None)
    };
    let ptr_of = |b: &Option<candle_core::cuda::cudarc::driver::CudaSlice<f32>>| -> u64 {
        b.as_ref().map_or(0, |b| {
            let (p, _g) = b.device_ptr(&stream);
            p
        })
    };
    let (part_ptr, ml_ptr) = (ptr_of(&part), ptr_of(&ml));
    let scale_log2 = p.softmax_scale * std::f32::consts::LOG2_E;
    let (s_i, past_i, kv_i, h_i, ns_i) = (s as i32, past as i32, kv_len as i32, h as i32, nsplit as i32);
    let (q_hs, o_hs) = ((s * HEAD_DIM) as i32, (s * HEAD_DIM) as i32);
    let main = func(dev, kernel)?;
    let mut b = stream.launch_builder(&main);
    b.arg(&q_ptr).arg(&k_ptr).arg(&v_ptr).arg(&out_ptr).arg(&part_ptr).arg(&ml_ptr);
    b.arg(&scale_log2).arg(&s_i).arg(&past_i).arg(&kv_i).arg(&h_i).arg(&ns_i);
    b.arg(&q_hs).arg(&o_hs).arg(&k_hs).arg(&v_hs);
    let lc = LaunchConfig { grid_dim: (npb as u32, kvh as u32, nsplit as u32), block_dim: ((32 * nw) as u32, 1, 1), shared_mem_bytes: 0 };
    unsafe { b.launch(lc) }.w()?;
    if nsplit > 1 {
        let comb = func(dev, COMBINE)?;
        let mut b = stream.launch_builder(&comb);
        b.arg(&part_ptr).arg(&ml_ptr).arg(&out_ptr).arg(&s_i).arg(&h_i).arg(&ns_i).arg(&o_hs);
        let lc = LaunchConfig { grid_dim: (s as u32, h as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe { b.launch(lc) }.w()?;
    }
    // part / ml are freed stream-ordered after the launches
    drop((part, ml));
    Ok(Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
        Shape::from((1, h, s, HEAD_DIM)),
    )))
}
