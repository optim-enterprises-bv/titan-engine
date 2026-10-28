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
const HOST_TILE_SPLITS: usize = 2;

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

fn dptr_f32(t: &Tensor, stream: &std::sync::Arc<candle_core::cuda::cudarc::driver::CudaStream>) -> Result<u64> {
    let (st, l) = t.storage_and_layout();
    let Storage::Cuda(c) = &*st else {
        candle_core::bail!("flash-prefill: tensor must live on CUDA");
    };
    let (p, _g) = c.as_cuda_slice::<f32>()?.device_ptr(stream);
    Ok(p + (l.start_offset() * 4) as u64)
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

/// Keys `a .. a + len` of a host F8E4M3 cache `t` (1, kvh, >= a + len, 256) into the device F8E4M3 buffer `dst`
/// (1, kvh, cap, 256), head `g` at row 0 of its `cap` rows.
fn upload_keys(t: &Tensor, a: usize, len: usize, dst: &Tensor, dev: &CudaDevice) -> Result<()> {
    let (_, kvh, cap, hd) = dst.dims4()?;
    let (st, l) = t.storage_and_layout();
    let Storage::Cpu(c) = &*st else {
        candle_core::bail!("flash-prefill host tiles: K/V must live on the CPU");
    };
    let src = c.as_slice::<float8::F8E4M3>()?;
    let hs = t.stride()[1];
    let stream = dev.cuda_stream();
    let (dst_st, _) = dst.storage_and_layout();
    let Storage::Cuda(dc) = &*dst_st else {
        candle_core::bail!("flash-prefill host tiles: staging buffer must live on CUDA");
    };
    let (base, _g) = dc.as_cuda_slice::<float8::F8E4M3>()?.device_ptr(&stream);
    for g in 0..kvh {
        let off = l.start_offset() + g * hs + a * hd;
        // pageable source: the driver stages it before returning, ordered on the stream
        unsafe {
            candle_core::cuda::cudarc::driver::result::memcpy_htod_async(
                base + (g * cap * hd) as u64,
                &src[off..off + len * hd],
                stream.cu_stream(),
            )
        }
        .w()?;
    }
    Ok(())
}

/// Causal attention of the last `s` positions of `q` (1, h, s, 256) bf16 over a host F8E4M3 cache `k`, `v`
/// (1, kvh, kv_len, 256): the cached keys stream to the device `tile` at a time and each tile's split partials
/// merge into running f32 (o, max, sum) as they come, so device memory stays one tile plus the chunk.
pub(crate) fn attend_host_tiles(q: &Tensor, k: &Tensor, v: &Tensor, p: &SdpaParams, tile: usize) -> Result<Tensor> {
    let Device::Cuda(dev) = q.device() else {
        candle_core::bail!("flash-prefill host tiles: q must live on CUDA");
    };
    let q = q.contiguous()?;
    let (_, h0, s, d) = q.dims4()?;
    let (_, kvh, kv_len, kd) = k.dims4()?;
    if (d != HEAD_DIM && d != 128) || kd != d || h0 % kvh != 0 || h0 / kvh > N_REP || kv_len < s || k.dtype() != DType::F8E4M3 {
        candle_core::bail!("flash-prefill host tiles: unsupported q {:?} k {:?} {:?}", q.shape(), k.shape(), k.dtype());
    }
    let n_rep = h0 / kvh;
    let q = if n_rep < N_REP {
        let pad = Tensor::zeros((kvh, N_REP - n_rep, s, d), q.dtype(), q.device())?;
        Tensor::cat(&[&q.reshape((kvh, n_rep, s, d))?, &pad], 1)?.reshape((1, kvh * N_REP, s, d))?
    } else {
        q
    };
    let h = kvh * N_REP;
    let past = kv_len - s;
    // cached keys in whole tiles (every row sees all of them), then the chunk's own keys (causal)
    let mut ranges: Vec<(usize, usize)> = (0..past).step_by(tile.max(TILE)).map(|a| (a, (a + tile.max(TILE)).min(past))).collect();
    ranges.push((past, kv_len));

    let stream = dev.cuda_stream();
    // head dim 128 drives the d128 w8 kernel (same launch signature as the 256 w8 family) and its 128-stride merge
    let (main, nw, merge) = if d == 128 {
        (func_named(dev, "flash_prefill_d128_w8")?, 8usize, func_named(dev, "flash_prefill_host_merge128")?)
    } else {
        (func(dev, cfg().kernel)?, KERNELS[cfg().kernel].2, func_named(dev, "flash_prefill_host_merge")?)
    };
    let npb = s.div_ceil(2 * nw);
    let q_ptr = dptr(&q, &stream)?;
    let scale_log2 = p.softmax_scale * std::f32::consts::LOG2_E;
    let (s_i, h_i, q_hs, o_hs, win_i) = (s as i32, h as i32, (s * d) as i32, (s * d) as i32, 0i32);
    // every buffer of the loop has one size for the whole call: varying tile lengths fragmented the device pool
    // until a 248k-token prompt ran out of memory
    let cap = tile.max(TILE).max(s);
    let (kf8, vf8) = (
        Tensor::zeros((1, kvh, cap, d), DType::F8E4M3, q.device())?,
        Tensor::zeros((1, kvh, cap, d), DType::F8E4M3, q.device())?,
    );
    let hs = (cap * d) as u64;
    // 2 splits: the kernel writes partials rather than normalised output, and each split's f32 partial is
    // h x s x d floats, the largest buffer next to the weights
    let ns = HOST_TILE_SPLITS;
    let part = Tensor::zeros((ns, h, s, d), DType::F32, q.device())?;
    let ml = Tensor::zeros((ns, h, s, 2), DType::F32, q.device())?;
    let (part_ptr, ml_ptr) = (dptr_f32(&part, &stream)?, dptr_f32(&ml, &stream)?);
    // running (o, m, l) merged in place by one kernel per tile: the tensor merge's temporaries were the largest
    // allocations of a prompt step's last layers
    let run_o = unsafe { dev.alloc::<f32>(h * s * d)? };
    let run_ml = unsafe { dev.alloc::<f32>(h * s * 2)? };
    let out = unsafe { dev.alloc::<half::bf16>(h0 * s * d)? };
    let ptr = |b: &candle_core::cuda::cudarc::driver::CudaSlice<f32>| b.device_ptr(&stream).0;
    let (run_o_ptr, run_ml_ptr, out_ptr) = (ptr(&run_o), ptr(&run_ml), out.device_ptr(&stream).0);
    let n_rep_i = n_rep as i32;
    let n = ranges.len();
    for (i, (a, b)) in ranges.into_iter().enumerate() {
        let len = b - a;
        upload_keys(k, a, len, &kf8, dev)?;
        upload_keys(v, a, len, &vf8, dev)?;
        let (kt, vt) = (kf8.to_dtype(DType::BF16)?, vf8.to_dtype(DType::BF16)?);
        let (k_ptr, v_ptr) = (dptr(&kt, &stream)?, dptr(&vt, &stream)?);
        let (past_i, kv_i, ns_i) = ((past - a) as i32, len as i32, ns as i32);
        let mut lb = stream.launch_builder(&main);
        lb.arg(&q_ptr).arg(&k_ptr).arg(&v_ptr).arg(&part_ptr).arg(&part_ptr).arg(&ml_ptr);
        lb.arg(&scale_log2).arg(&s_i).arg(&past_i).arg(&kv_i).arg(&h_i).arg(&ns_i);
        lb.arg(&q_hs).arg(&o_hs).arg(&hs).arg(&hs).arg(&win_i);
        let lc = LaunchConfig { grid_dim: (npb as u32, kvh as u32, ns as u32), block_dim: ((32 * nw) as u32, 1, 1), shared_mem_bytes: 0 };
        unsafe { lb.launch(lc) }.w()?;
        let (first, last) = ((i == 0) as i32, (i + 1 == n) as i32);
        let mut mb = stream.launch_builder(&merge);
        mb.arg(&part_ptr).arg(&ml_ptr).arg(&run_o_ptr).arg(&run_ml_ptr).arg(&out_ptr);
        mb.arg(&s_i).arg(&h_i).arg(&first).arg(&last).arg(&n_rep_i);
        let lc = LaunchConfig { grid_dim: (s as u32, h as u32, 1), block_dim: (d as u32, 1, 1), shared_mem_bytes: 0 };
        unsafe { mb.launch(lc) }.w()?;
    }
    // run_o / run_ml are freed stream-ordered after the launches
    drop((run_o, run_ml));
    Ok(Tensor::from((Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())), Shape::from((1, h0, s, d)))))
}

/// Shapes `attend_any` takes, without tensors: head dim 128 / 256 (at most 8 query heads per KV head) or 512 (a multiple
/// of 8), bf16 or f16 (the planner's twin of `supported_any`).
pub(crate) fn supported_any_shape(dtype: DType, head_dim: usize, q_heads: usize, kv_heads: usize) -> bool {
    let rep = if kv_heads > 0 && q_heads.is_multiple_of(kv_heads) { q_heads / kv_heads } else { 0 };
    matches!(dtype, DType::BF16 | DType::F16)
        && rep > 0
        && match head_dim {
            128 | HEAD_DIM => rep <= N_REP,
            512 => rep.is_multiple_of(N_REP),
            _ => false,
        }
        && !mistralrs_quant::distributed::use_nccl()
}

/// Peak device bytes `attend_any` allocates for `s` query rows over `kv_len` keys (`win` as in `attend_any`): the
/// zero-padded query, the output, and the split-KV partials (the same split count the launch picks; an upper bound for
/// the head-dim-256 bf16 `attend` path, which packs 2 positions per warp like the 8-warp kernels).
pub(crate) fn attend_any_workspace_bytes(dtype: DType, head_dim: usize, q_heads: usize, kv_heads: usize, s: usize, kv_len: usize, win: usize) -> usize {
    let d = head_dim;
    let rep = (q_heads / kv_heads.max(1)).max(1);
    let hp = if d == 512 { q_heads } else { kv_heads * N_REP.max(rep) };
    let es = dtype.size_in_bytes();
    let q_pad = if hp > q_heads { hp * s * d * es * 2 } else { 0 };
    let out = hp * s * d * 4;
    let (groups, halves) = if d == 512 { (q_heads / N_REP, 2) } else { (kv_heads, 1) };
    let ppb = 16;
    let npb = s.div_ceil(ppb);
    let tiles = (if win > 0 { kv_len.min(win + ppb) } else { kv_len }).div_ceil(TILE);
    let nsplit = cfg().units.div_ceil((npb * groups * halves).max(1)).min(tiles.div_ceil(MIN_SPLIT_TILES)).clamp(1, MAX_SPLITS);
    let part = if nsplit > 1 { nsplit * hp * s * (d + 2) * 4 } else { 0 };
    q_pad + out * 2 + part
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
