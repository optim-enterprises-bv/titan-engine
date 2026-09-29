//! Split-K flash-decoding for CUDA decode rows (titan-engine, `TITAN_ATTN_FLASH`).
//!
//! One `split` + one `combine` launch per attention layer and step covers every query row of the
//! step (the decode row, or the MTP verify rows): each KV head's K and V are read once for its
//! `n_rep` query heads x rows, in fixed 128-key tiles grouped into `chunk`-key blocks with an
//! online softmax, then merged per row in chunk order. Row `i` of a call sees keys
//! `< past + i + 1`. A row's result depends only on its own key count, not on the other rows of the
//! call (oxide-kernels/flash-decode gates this bit for bit), so a verify row still equals the decode
//! step it replaces, as MTP requires.
//!
//! Numerics: llama.cpp fattn-vec style (f32 scaled Q, f32 dots and V accumulation, fast-math expf,
//! combine by max-rescaled sums), not the cuBLASLt path's bf16-rounded scores and probabilities;
//! hence the threshold `TITAN_ATTN_FLASH_MIN`: shorter contexts keep the existing path and its
//! exact results.
//!
//! The kernels are cuda-oxide (`titan-engine/oxide-kernels/flash-decode`), embedded as PTX and
//! launched from Rust, so the nvcc and the nvcc-free `oxide` builds share them. Regenerate
//! flash_decode_oxide.ptx with oxide-kernels/flash-decode/export_ptx.sh.
//!
//! Capture: not capture-safe as written (kv_total is a by-value argument and the partials are
//! allocated per call). With kv_total read from a device buffer and a persistent workspace sized for
//! the context limit, the pair could move inside the CUDA-graph segments.

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

const PTX: &str = include_str!("flash_decode_oxide.ptx");
const MODULE: &str = "titan_flash_decode";
const HEAD_DIM: usize = 256;
const TILE: usize = 128;
/// Queries (n_rep x rows) per block, the largest kernel instance.
const GMAX: usize = 24;
/// Default `TITAN_ATTN_FLASH_MIN`: keys from which decode rows take the flash kernel.
const DEFAULT_MIN_KV: usize = 1024;
/// Default `TITAN_ATTN_FLASH_CHUNK`: keys per split block (a multiple of 128).
const DEFAULT_CHUNK: usize = 256;

struct Cfg {
    on: bool,
    min_kv: usize,
    chunk: usize,
}

fn cfg() -> &'static Cfg {
    static C: OnceLock<Cfg> = OnceLock::new();
    C.get_or_init(|| {
        let on = std::env::var("TITAN_ATTN_FLASH")
            .map(|v| v != "0")
            .unwrap_or(true);
        let min_kv = std::env::var("TITAN_ATTN_FLASH_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_MIN_KV)
            .max(1);
        let chunk = std::env::var("TITAN_ATTN_FLASH_CHUNK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(DEFAULT_CHUNK)
            .div_ceil(TILE)
            .max(1)
            * TILE;
        tracing::info!(
            "titan attention: flash-decode {} from {min_kv} keys, {chunk} keys per block",
            if on { "on" } else { "off" }
        );
        Cfg { on, min_kv, chunk }
    })
}

type FnCache = Mutex<HashMap<(DeviceId, usize), CudaFunction>>;
static FNS: OnceLock<FnCache> = OnceLock::new();
const NAMES: [&str; 4] = [
    "flash_decode_split_g8",
    "flash_decode_split_g16",
    "flash_decode_split_g24",
    "flash_decode_combine",
];

fn func(dev: &CudaDevice, idx: usize) -> Result<CudaFunction> {
    let key = (dev.id(), idx);
    let mut map = FNS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap();
    if let Some(f) = map.get(&key) {
        return Ok(f.clone());
    }
    let f = dev
        .get_or_load_custom_func(NAMES[idx], MODULE, PTX)?
        .into_cuda_function();
    map.insert(key, f.clone());
    Ok(f)
}

/// Shape, dtype and layout support (the key count aside): CUDA, batch 1, bf16, head_dim 256, GQA
/// with at most 24 query heads per KV head, K/V rows contiguous and 16-byte aligned, no softcap,
/// sliding window or sinks.
fn supported(q: &Tensor, k: &Tensor, v: &Tensor, p: &SdpaParams) -> bool {
    let (Ok((b, h, _, d)), Ok((kb, kvh, kl, kd)), Ok((vb, vh, vl, vd))) =
        (q.dims4(), k.dims4(), v.dims4())
    else {
        return false;
    };
    let rows_ok = |t: &Tensor| {
        let st = t.stride();
        st[3] == 1 && st[2] == HEAD_DIM && st[1] % 8 == 0 && t.layout().start_offset() % 8 == 0
    };
    q.device().is_cuda()
        && [q.dtype(), k.dtype(), v.dtype()]
            .iter()
            .all(|t| *t == DType::BF16)
        && b == 1
        && kb == 1
        && vb == 1
        && d == HEAD_DIM
        && kd == HEAD_DIM
        && vd == HEAD_DIM
        && kvh == vh
        && kl == vl
        && kvh > 0
        && h % kvh == 0
        && h / kvh <= GMAX
        && h / kvh == p.n_kv_groups.max(1)
        && p.softcap.is_none()
        && p.sliding_window.is_none()
        && p.sinks.is_none()
        && rows_ok(k)
        && rows_ok(v)
        && !mistralrs_quant::distributed::use_nccl()
}

/// For query rows `0..n` of `q` (1, h, n, 256) at key counts `past + 1 ..= past + n`: the first row
/// the flash kernel takes (`n` if none). Rows from the threshold on take it, so a row's path
/// depends on its own key count only (decode and verify agree).
pub(crate) fn first_row(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    past: usize,
    n: usize,
    p: &SdpaParams,
) -> usize {
    let c = cfg();
    if !c.on
        || n == 0
        || past + n < c.min_kv
        || k.dim(2).map_or(true, |l| l < past + n)
        || !supported(q, k, v, p)
    {
        return n;
    }
    // row i has past + i + 1 keys
    (c.min_kv.saturating_sub(past + 1)).min(n)
}

/// `TITAN_ATTN_FLASH_MIN` when the kernel is on.
pub(crate) fn min_kv() -> Option<usize> {
    let c = cfg();
    c.on.then_some(c.min_kv)
}

fn dptr(
    t: &Tensor,
    stream: &std::sync::Arc<candle_core::cuda::cudarc::driver::CudaStream>,
) -> Result<u64> {
    let (st, l) = t.storage_and_layout();
    let Storage::Cuda(c) = &*st else {
        candle_core::bail!("flash-decode: tensor must live on CUDA");
    };
    let s = c.as_cuda_slice::<half::bf16>()?;
    let (p, _g) = s.device_ptr(stream);
    Ok(p + (l.start_offset() * 2) as u64)
}

/// Attention for query rows `q` (1, h, m, 256) where row `i` sees keys `< past + i + 1` of `k`, `v`
/// (1, kvh, >= past + m, 256). Returns `(1, m, h * 256)` bf16. Call only where `first_row` said so.
pub(crate) fn decode_rows(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    past: usize,
    p: &SdpaParams,
) -> Result<Tensor> {
    let Device::Cuda(dev) = q.device() else {
        candle_core::bail!("flash-decode: CUDA only");
    };
    let (_, h, m, _) = q.dims4()?;
    let (_, kvh, kl, _) = k.dims4()?;
    if kl < past + m || m == 0 {
        candle_core::bail!("flash-decode: {m} rows after {past} keys, cache holds {kl}");
    }
    let n_rep = h / kvh;
    let rows_per = (GMAX / n_rep).max(1);
    let chunk = cfg().chunk;
    let q = q.contiguous()?;
    let stream = dev.cuda_stream();
    let (q_ptr, k_ptr, v_ptr) = (dptr(&q, &stream)?, dptr(k, &stream)?, dptr(v, &stream)?);
    let (k_hs, v_hs) = (k.stride()[1] as u64, v.stride()[1] as u64);
    if (k_ptr | v_ptr) % 16 != 0 {
        candle_core::bail!("flash-decode: K/V not 16-byte aligned");
    }
    let mut out = unsafe { dev.alloc::<half::bf16>(m * h * HEAD_DIM)? };
    let out_ptr = {
        let (p, _g) = out.device_ptr(&stream);
        p
    };
    let comb = func(&dev, 3)?;
    let mut r0 = 0;
    while r0 < m {
        let rows = rows_per.min(m - r0);
        let l = past + r0 + rows;
        let nchunks = l.div_ceil(chunk);
        let g = n_rep * rows;
        let split = func(
            &dev,
            if g <= 8 {
                0
            } else if g <= 16 {
                1
            } else {
                2
            },
        )?;
        let nq = rows * h;
        let part = unsafe { dev.alloc::<f32>(nq * nchunks * HEAD_DIM)? };
        let meta = unsafe { dev.alloc::<f32>(nq * nchunks * 2)? };
        let (part_ptr, meta_ptr) = {
            let (a, _ga) = part.device_ptr(&stream);
            let (b, _gb) = meta.device_ptr(&stream);
            (a, b)
        };
        let qp = q_ptr + (r0 * HEAD_DIM * 2) as u64;
        let op = out_ptr + (r0 * h * HEAD_DIM * 2) as u64;
        let (n_rep_i, h_i, l_i, rows_i, chunk_i, nch_i) = (
            n_rep as i32,
            h as i32,
            l as i32,
            rows as i32,
            chunk as i32,
            nchunks as i32,
        );
        let (q_hs, q_rs) = ((m * HEAD_DIM) as i32, HEAD_DIM as i32);
        let (out_rs, out_hs) = ((h * HEAD_DIM) as i32, HEAD_DIM as i32);
        let scale = p.softmax_scale;
        let mut b = stream.launch_builder(&split);
        b.arg(&qp)
            .arg(&k_ptr)
            .arg(&v_ptr)
            .arg(&part_ptr)
            .arg(&meta_ptr)
            .arg(&scale);
        b.arg(&n_rep_i)
            .arg(&h_i)
            .arg(&l_i)
            .arg(&rows_i)
            .arg(&chunk_i)
            .arg(&nch_i);
        b.arg(&q_hs).arg(&q_rs).arg(&k_hs).arg(&v_hs);
        let cfg_s = LaunchConfig {
            grid_dim: (nchunks as u32, kvh as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(cfg_s) }.w()?;
        let mut b = stream.launch_builder(&comb);
        b.arg(&part_ptr)
            .arg(&meta_ptr)
            .arg(&op)
            .arg(&h_i)
            .arg(&rows_i)
            .arg(&l_i);
        b.arg(&chunk_i).arg(&nch_i).arg(&out_rs).arg(&out_hs);
        let cfg_c = LaunchConfig {
            grid_dim: (h as u32, rows as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(cfg_c) }.w()?;
        // part / meta are freed stream-ordered after the launches
        drop((part, meta));
        r0 += rows;
    }
    Ok(Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
        Shape::from((1, m, h * HEAD_DIM)),
    )))
}
