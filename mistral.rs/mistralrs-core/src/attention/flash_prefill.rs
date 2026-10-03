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
    static C: mistralrs_quant::titan_cfg::GenCell<Cfg> = mistralrs_quant::titan_cfg::GenCell::new();
    C.get_or_init(|| {
        let env = |n: &str, d: usize| mistralrs_quant::titan_cfg::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let on = mistralrs_quant::titan_cfg::var("TITAN_ATTN_FLASH_PREFILL").map(|v| v != "0").unwrap_or(true);
        let min_kv = env("TITAN_ATTN_FLASH_PREFILL_MIN", DEFAULT_MIN_KV).max(1);
        let units = env("TITAN_ATTN_FLASH_PREFILL_UNITS", DEFAULT_UNITS).max(1);
        let want = mistralrs_quant::titan_cfg::var("TITAN_ATTN_FLASH_PREFILL_KERNEL").unwrap_or_else(|_| "w8".into());
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

type FnCache = Mutex<HashMap<(DeviceId, &'static str), CudaFunction>>;
static FNS: OnceLock<FnCache> = OnceLock::new();

fn func(dev: &CudaDevice, idx: usize) -> Result<CudaFunction> {
    func_named(dev, if idx == COMBINE { "flash_prefill_combine" } else { KERNELS[idx].1 })
}

fn func_named(dev: &CudaDevice, name: &'static str) -> Result<CudaFunction> {
    let key = (dev.id(), name);
    let mut map = FNS.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap();
    if let Some(f) = map.get(&key) {
        return Ok(f.clone());
    }
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
        && h % kvh == 0
        && h / kvh <= N_REP
        && p.n_kv_groups == h / kvh
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
    let p = if t.dtype() == DType::F16 {
        let (p, _g) = c.as_cuda_slice::<half::f16>()?.device_ptr(stream);
        p
    } else {
        let (p, _g) = c.as_cuda_slice::<half::bf16>()?.device_ptr(stream);
        p
    };
    Ok(p + (l.start_offset() * 2) as u64)
}

/// Causal attention of the last `s` positions: `q` (1, h, s, 256), `k`, `v` (1, kvh, kv_len, 256) with the
/// queries at positions `kv_len - s ..`; `win > 0` also limits each query to its last `win` keys (itself included).
/// Returns (1, h, s, 256) bf16.
pub(crate) fn attend(q: &Tensor, k: &Tensor, v: &Tensor, p: &SdpaParams, win: usize) -> Result<Tensor> {
    let Device::Cuda(dev) = q.device() else {
        candle_core::bail!("flash-prefill: CUDA only");
    };
    let q = q.contiguous()?;
    if !supported(&q, k, v, p) {
        candle_core::bail!("flash-prefill: unsupported q {:?} k {:?} v {:?}", q.layout(), k.layout(), v.layout());
    }
    let (_, h0, s0, _) = q.dims4()?;
    let kvh0 = k.dim(1)?;
    let n_rep = h0 / kvh0;
    if n_rep < N_REP {
        // fewer query heads per KV head than the kernel packs (Qwen3.8 27B: 6): zero heads fill each group
        // (heads never mix), their rows are dropped after
        let pad = Tensor::zeros((kvh0, N_REP - n_rep, s0, HEAD_DIM), q.dtype(), q.device())?;
        let qp = Tensor::cat(&[&q.reshape((kvh0, n_rep, s0, HEAD_DIM))?, &pad], 1)?.reshape((1, kvh0 * N_REP, s0, HEAD_DIM))?;
        let pp = SdpaParams {
            n_kv_groups: N_REP,
            softcap: p.softcap,
            softmax_scale: p.softmax_scale,
            sliding_window: p.sliding_window,
            sinks: p.sinks.clone(),
        };
        let o = attend(&qp, k, v, &pp, win)?;
        return o.reshape((kvh0, N_REP, s0, HEAD_DIM))?.narrow(1, 0, n_rep)?.reshape((1, h0, s0, HEAD_DIM))?.contiguous();
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
    let tiles = (if win > 0 { kv_len.min(win + 2 * nw) } else { kv_len }).div_ceil(TILE);
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
    let win_i = win as i32;
    let (q_hs, o_hs) = ((s * HEAD_DIM) as i32, (s * HEAD_DIM) as i32);
    let main = func(dev, kernel)?;
    let mut b = stream.launch_builder(&main);
    b.arg(&q_ptr).arg(&k_ptr).arg(&v_ptr).arg(&out_ptr).arg(&part_ptr).arg(&ml_ptr);
    b.arg(&scale_log2).arg(&s_i).arg(&past_i).arg(&kv_i).arg(&h_i).arg(&ns_i);
    // v1 has no `win` parameter; the extra argument is ignored there
    b.arg(&q_hs).arg(&o_hs).arg(&k_hs).arg(&v_hs).arg(&win_i);
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

/// Head dims, dtypes and GQA layouts `attend_any` takes: CUDA, batch 1, q / k / v all bf16 or all f16, head dim
/// 256 (at most 8 query heads per KV head, fewer padded with zero heads) or 512 (a multiple of 8 query heads per KV
/// head), K/V rows contiguous and 16-byte aligned, no tensor parallelism.
pub(crate) fn supported_any(q: &Tensor, k: &Tensor, v: &Tensor) -> bool {
    let (Ok((b, h, s, d)), Ok((kb, kvh, kl, kd)), Ok((vb, vh, vl, vd))) = (q.dims4(), k.dims4(), v.dims4()) else {
        return false;
    };
    let rows_ok = |t: &Tensor| {
        let st = t.stride();
        st[3] == 1 && st[2] == d && st[1] % 8 == 0 && t.layout().start_offset() % 8 == 0
    };
    let dt = q.dtype();
    let rep = if kvh > 0 && h % kvh == 0 { h / kvh } else { 0 };
    q.device().is_cuda()
        && matches!(dt, DType::BF16 | DType::F16)
        && k.dtype() == dt
        && v.dtype() == dt
        && (b, kb, vb) == (1, 1, 1)
        && (d == HEAD_DIM || d == 512 || d == 128)
        && kd == d
        && vd == d
        && kvh == vh
        && kl == vl
        && kl >= s
        && rep > 0
        && if d == 512 { rep % N_REP == 0 } else { rep <= N_REP }
        && rows_ok(k)
        && rows_ok(v)
        && !mistralrs_quant::distributed::use_nccl()
}

/// Causal attention of the last `s` positions of `q` (1, h, s, d) over `k`, `v` (1, kvh, kv_len, d), scale
/// `softmax_scale`, `win > 0`: each query sees only its last `win` keys; `s` may be 1 (a decode row). Kernels:
/// - head dim 256, bf16, 2 query heads per KV head: `flash_prefill_r2_w8` (no padded heads); other bf16: `attend`
///   (the configured kernel, bit-identical to before);
/// - head dim 256, f16: 2 per KV head `flash_prefill_r2_w8_f16o32` (f32 out only), else zero-padded to 8 heads per KV
///   head on `flash_prefill_w8_f16` / `_f16o32`;
/// - head dim 512 (multiple of 8 query heads per KV head): `flash_prefill_d512` (bf16), `flash_prefill_d512_f16` /
///   `_f16o32`; TITAN_ATTN_FLASH_PREFILL_D512_W8=1 takes the 8-warp twins where they exist;
/// - head dim 128 (qwen3; at most 8 query heads per KV head, zero-padded to 8): `flash_prefill_d128_w8` (bf16), on f16
///   `flash_prefill_d128_w8_f16o32` (f32 out only).
/// Returns (1, h, s, d), f32 when `out32` (f16 inputs only), else bf16.
pub(crate) fn attend_any(q: &Tensor, k: &Tensor, v: &Tensor, softmax_scale: f32, win: usize, out32: bool) -> Result<Tensor> {
    let q = q.contiguous()?;
    if !supported_any(&q, k, v) {
        candle_core::bail!("flash-prefill: unsupported q {:?} {:?} k {:?} v {:?}", q.layout(), q.dtype(), k.layout(), v.layout());
    }
    let (_, h, s, d) = q.dims4()?;
    let kvh = k.dim(1)?;
    let f16 = q.dtype() == DType::F16;
    if out32 && !f16 {
        candle_core::bail!("flash-prefill: f32 output needs f16 inputs");
    }
    let n_rep = h / kvh;
    if d == HEAD_DIM && n_rep == 2 && (out32 || !f16) {
        let name = if f16 { "flash_prefill_r2_w8_f16o32" } else { "flash_prefill_r2_w8" };
        return launch_named(&q, k, v, softmax_scale, win, name, 8, 2, out32);
    }
    let p = SdpaParams { n_kv_groups: n_rep, softcap: None, softmax_scale, sliding_window: None, sinks: None };
    if d == HEAD_DIM && !f16 {
        return attend(&q, k, v, &p, win);
    }
    if d == 128 && f16 && !out32 {
        candle_core::bail!("flash-prefill: head dim 128 on f16 has f32 output only");
    }
    if (d == HEAD_DIM || d == 128) && n_rep < N_REP {
        let pad = Tensor::zeros((kvh, N_REP - n_rep, s, d), q.dtype(), q.device())?;
        let qp = Tensor::cat(&[&q.reshape((kvh, n_rep, s, d))?, &pad], 1)?.reshape((1, kvh * N_REP, s, d))?;
        let o = attend_any(&qp, k, v, softmax_scale, win, out32)?;
        return o.reshape((kvh, N_REP, s, d))?.narrow(1, 0, n_rep)?.reshape((1, h, s, d))?.contiguous();
    }
    let d512_w8 = {
        static W8: mistralrs_quant::titan_cfg::GenCell<bool> = mistralrs_quant::titan_cfg::GenCell::new();
        *W8.get_or_init(|| mistralrs_quant::titan_cfg::var("TITAN_ATTN_FLASH_PREFILL_D512_W8").is_ok_and(|v| v == "1"))
    };
    let (name, nw): (&'static str, usize) = match (d, f16, out32, d512_w8) {
        (128, false, _, _) => ("flash_prefill_d128_w8", 8),
        (128, true, _, _) => ("flash_prefill_d128_w8_f16o32", 8),
        (HEAD_DIM, _, false, _) => ("flash_prefill_w8_f16", 8),
        (HEAD_DIM, _, true, _) => ("flash_prefill_w8_f16o32", 8),
        (_, false, _, false) => ("flash_prefill_d512", 4),
        (_, false, _, true) => ("flash_prefill_d512_w8", 8),
        (_, true, true, _) => ("flash_prefill_d512_f16o32", 4),
        (_, true, false, false) => ("flash_prefill_d512_f16", 4),
        (_, true, false, true) => ("flash_prefill_d512_w8_f16", 8),
    };
    launch_named(&q, k, v, softmax_scale, win, name, nw, N_REP, out32)
}

/// One launch (+ split merge) of kernel `name` (`nw` warps, `krep` query heads per KV head in its warp tile: 8, or
/// 2 for the r2 kernels) on supported, GQA-ready q / k / v; f32 output when `out32`.
#[allow(clippy::too_many_arguments)]
fn launch_named(q: &Tensor, k: &Tensor, v: &Tensor, softmax_scale: f32, win: usize, name: &'static str, nw: usize, krep: usize, out32: bool) -> Result<Tensor> {
    let Device::Cuda(dev) = q.device() else {
        candle_core::bail!("flash-prefill: CUDA only");
    };
    let (_, h, s, d) = q.dims4()?;
    let (_, kvh, kv_len, _) = k.dims4()?;
    let past = kv_len - s;
    let stream = dev.cuda_stream();
    let (q_ptr, k_ptr, v_ptr) = (dptr(q, &stream)?, dptr(k, &stream)?, dptr(v, &stream)?);
    let (k_hs, v_hs) = (k.stride()[1] as u64, v.stride()[1] as u64);
    if (k_ptr | v_ptr | q_ptr) % 16 != 0 {
        candle_core::bail!("flash-prefill: q/K/V not 16-byte aligned");
    }
    // head dim 512: grid y = query-head groups of 8 (gq per KV head), z = 2 output halves per split
    let (groups, halves) = if d == 512 { (h / N_REP, 2) } else { (kvh, 1) };
    let gq = (h / kvh / N_REP).max(1) as i32;
    let ppb = (16 / krep) * nw;
    let npb = s.div_ceil(ppb);
    let tiles = (if win > 0 { kv_len.min(win + ppb) } else { kv_len }).div_ceil(TILE);
    let nsplit = cfg()
        .units
        .div_ceil(npb * groups * halves)
        .min(tiles.div_ceil(MIN_SPLIT_TILES))
        .clamp(1, MAX_SPLITS);
    let n = h * s * d;
    let (out_bf, out_f) = if out32 {
        (None, Some(unsafe { dev.alloc::<f32>(n)? }))
    } else {
        (Some(unsafe { dev.alloc::<half::bf16>(n)? }), None)
    };
    let out_ptr = match (&out_bf, &out_f) {
        (Some(o), _) => o.device_ptr(&stream).0,
        (_, Some(o)) => o.device_ptr(&stream).0,
        _ => unreachable!(),
    };
    let (part, ml) = if nsplit > 1 {
        (Some(unsafe { dev.alloc::<f32>(nsplit * h * s * d)? }), Some(unsafe { dev.alloc::<f32>(nsplit * h * s * 2)? }))
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
    let scale_log2 = softmax_scale * std::f32::consts::LOG2_E;
    let (s_i, past_i, kv_i, h_i, ns_i) = (s as i32, past as i32, kv_len as i32, h as i32, nsplit as i32);
    let win_i = win as i32;
    let (q_hs, o_hs) = ((s * d) as i32, (s * d) as i32);
    let main = func_named(dev, name)?;
    let mut b = stream.launch_builder(&main);
    b.arg(&q_ptr).arg(&k_ptr).arg(&v_ptr).arg(&out_ptr).arg(&part_ptr).arg(&ml_ptr);
    b.arg(&scale_log2).arg(&s_i).arg(&past_i).arg(&kv_i).arg(&h_i).arg(&ns_i);
    b.arg(&q_hs).arg(&o_hs).arg(&k_hs).arg(&v_hs).arg(&win_i);
    if d == 512 {
        b.arg(&gq);
    }
    let lc = LaunchConfig {
        grid_dim: (npb as u32, groups as u32, (halves * nsplit) as u32),
        block_dim: ((32 * nw) as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { b.launch(lc) }.w()?;
    if nsplit > 1 {
        let cn = match (d, out32) {
            (512, false) => "flash_prefill_combine512",
            (512, true) => "flash_prefill_combine512_f32",
            (128, false) => "flash_prefill_combine128",
            (128, true) => "flash_prefill_combine128_f32",
            (_, false) => "flash_prefill_combine",
            (_, true) => "flash_prefill_combine_f32",
        };
        let comb = func_named(dev, cn)?;
        let mut b = stream.launch_builder(&comb);
        b.arg(&part_ptr).arg(&ml_ptr).arg(&out_ptr).arg(&s_i).arg(&h_i).arg(&ns_i).arg(&o_hs);
        let cb = if d == 128 { 128 } else { 256 };
        let lc = LaunchConfig { grid_dim: (s as u32, h as u32, 1), block_dim: (cb, 1, 1), shared_mem_bytes: 0 };
        unsafe { b.launch(lc) }.w()?;
    }
    drop((part, ml));
    let storage = match (out_bf, out_f) {
        (Some(o), _) => CudaStorage::wrap_cuda_slice(o, dev.clone()),
        (_, Some(o)) => CudaStorage::wrap_cuda_slice(o, dev.clone()),
        _ => unreachable!(),
    };
    Ok(Tensor::from((Storage::Cuda(storage), Shape::from((1, h, s, d)))))
}
