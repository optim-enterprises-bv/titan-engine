//! titan noblas: candle's CUDA float matmul and rand_* on cuda-oxide kernels (titan-engine/oxide-kernels/gemm: tensor
//! cores for f16 / bf16, SIMT for f32 and any layout, GEMV for m <= 8, split-K, Philox4x32-10 fills; gated there
//! against f64 references and timed against cuBLAS on the roster's inventoried shapes). Compiled when candle-core is
//! built without its `cublas` feature, so the binary links neither cuBLAS, cuBLASLt nor cuRAND.
//!
//! gemm_oxide.ptx and oxide_gemm_plan.rs are copies of the crate's gemm.ptx and src/plan.rs (its export.sh).
//! `TITAN_GEMM_LOG=<file>` logs one line per launch plan (kernel, split, problem).
use super::{gemmlog, gemmlog_on, CudaDevice, DeviceId, WrapErr};
use crate::{Layout, Result};
use cudarc::driver::{CudaFunction, CudaSlice, DevicePtr, DevicePtrMut, DeviceRepr, LaunchConfig, PushKernelArg};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

#[path = "oxide_gemm_plan.rs"]
pub mod plan;
pub use plan::{Dt, Force, Launch, Problem};

const PTX: &str = include_str!("gemm_oxide.ptx");
const MODULE: &str = "titan_gemm";

fn func(dev: &CudaDevice, name: &str) -> Result<CudaFunction> {
    type Fns = Mutex<HashMap<(DeviceId, String), CudaFunction>>;
    static FNS: OnceLock<Fns> = OnceLock::new();
    let key = (dev.id(), name.to_string());
    let mut map = FNS.get_or_init(Default::default).lock().unwrap();
    if let Some(f) = map.get(&key) {
        return Ok(f.clone());
    }
    let f = dev.get_or_load_custom_func(name, MODULE, PTX)?.into_cuda_function();
    map.insert(key, f.clone());
    Ok(f)
}

/// `TITAN_GEMM_REF` (diagnostics, read once): comma-separated selectors routing matching calls to the f64-accumulating
/// reference kernel: `all`, `b1` (unbatched), `bN` (batched), a dtype (`f32` / `f16` / `bf16`), `kNNN` (that k).
fn use_ref(p: &Problem) -> bool {
    static SEL: OnceLock<Vec<String>> = OnceLock::new();
    let sel = SEL.get_or_init(|| {
        std::env::var("TITAN_GEMM_REF").map(|v| v.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect()).unwrap_or_default()
    });
    // the reference grid holds m in y (<= 65535): larger problems (e.g. the k = 1 RoPE table outer product) keep the plan
    p.m <= 65535 && sel.iter().any(|t| match t.as_str() {
        "all" => true,
        "b1" => p.batch == 1,
        "bN" => p.batch > 1,
        "f32" | "f16" | "bf16" => t == p.dt.name(),
        _ => t.strip_prefix('k').and_then(|k| k.parse::<i64>().ok()) == Some(p.k),
    })
}

/// D = alpha A B + beta C + bias for `p` (`p.a_addr` / `b_addr` / `d_addr` are device addresses; `c` and `bias`
/// device addresses or 0), on the device's stream.
pub fn gemm(dev: &CudaDevice, p: &Problem, c: u64, bias: u64) -> Result<()> {
    if p.m == 0 || p.n == 0 || p.batch == 0 {
        return Ok(());
    }
    let force = if use_ref(p) { Force { family: 5, ..Default::default() } } else { Force::default() };
    let l = plan::plan(p, dev.sm_count(), force).map_err(crate::Error::Msg)?;
    if gemmlog_on() {
        let q = &l.p;
        gemmlog(format_args!(
            "X {} split={} {} b={} m={} n={} k={} a=({},{}) b=({},{}) d=({},{}) s=({},{},{}) alpha={} beta={} c={} bias={}",
            l.kernel, l.nsplit, q.dt.name(), q.batch, q.m, q.n, q.k, q.a_s0, q.a_s1, q.b_s0, q.b_s1, q.d_s0, q.d_s1, q.sa,
            q.sb, q.sd, q.alpha, q.beta, c != 0, bias != 0
        ));
    }
    let stream = dev.cuda_stream();
    let ws: Option<CudaSlice<f32>> = if l.ws_floats > 0 { Some(unsafe { dev.alloc::<f32>(l.ws_floats)? }) } else { None };
    let (ws_ptr, ws_guard) = match &ws {
        Some(w) => {
            let (p, g) = w.device_ptr(&stream);
            (p, Some(g))
        }
        None => (0u64, None),
    };
    let q = &l.p;
    let (m, n, k, ns) = (q.m as i32, q.n as i32, q.k as i32, l.nsplit);
    let f = func(dev, &l.kernel)?;
    let mut b = stream.launch_builder(&f);
    b.arg(&q.a_addr).arg(&q.b_addr).arg(&q.d_addr).arg(&c).arg(&bias).arg(&m).arg(&n).arg(&k);
    b.arg(&q.a_s0).arg(&q.a_s1).arg(&q.b_s0).arg(&q.b_s1).arg(&q.d_s0).arg(&q.d_s1).arg(&q.c_s0).arg(&q.c_s1);
    b.arg(&q.bias_s0).arg(&q.bias_s1).arg(&q.sa).arg(&q.sb).arg(&q.sd).arg(&q.sc).arg(&q.alpha).arg(&q.beta);
    b.arg(&ws_ptr).arg(&ns);
    let lc = LaunchConfig { grid_dim: l.grid, block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    unsafe { b.launch(lc) }.w()?;
    if let Some((rn, rg)) = &l.reduce {
        let f = func(dev, rn)?;
        let mut b = stream.launch_builder(&f);
        b.arg(&ws_ptr).arg(&ns).arg(&q.d_addr).arg(&c).arg(&bias).arg(&m).arg(&n).arg(&q.d_s0).arg(&q.d_s1);
        b.arg(&q.c_s0).arg(&q.c_s1).arg(&q.bias_s0).arg(&q.bias_s1).arg(&q.sd).arg(&q.sc).arg(&q.alpha).arg(&q.beta);
        let lc = LaunchConfig { grid_dim: *rg, block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe { b.launch(lc) }.w()?;
    }
    // the workspace is freed stream-ordered after the launches
    drop(ws_guard);
    drop(ws);
    Ok(())
}

/// The batch stride of a matmul operand whose batch dims flatten to one stride (candle's cuBLAS rule).
fn batch_stride(l: &Layout, default: usize) -> Option<usize> {
    let s = l.stride();
    let d = l.dims();
    match s[..s.len() - 2] {
        [s1, stride] if s1 == stride * d[1] => Some(stride),
        [_, stride] if d[0] == 1 => Some(stride),
        [stride, _] if d[1] == 1 => Some(stride),
        [stride] => Some(stride),
        [] => Some(default),
        _ => None,
    }
}

/// candle's `lhs (b, m, k) @ rhs (b, k, n)` into a new contiguous `(b, m, n)` buffer.
pub fn matmul<T: DeviceRepr>(
    dev: &CudaDevice,
    dt: Dt,
    lhs: &CudaSlice<T>,
    rhs: &CudaSlice<T>,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_l: &Layout,
    rhs_l: &Layout,
) -> Result<CudaSlice<T>> {
    let (Some(sa), Some(sb)) = (batch_stride(lhs_l, m * k), batch_stride(rhs_l, n * k)) else {
        Err(super::CudaError::MatMulNonContiguous { lhs_stride: lhs_l.clone(), rhs_stride: rhs_l.clone(), mnk: (m, n, k) })?
    };
    let mut out = unsafe { dev.alloc::<T>(b * m * n)? };
    let stream = dev.cuda_stream();
    let es = dt.size() as u64;
    let (lp, _gl) = lhs.device_ptr(&stream);
    let (rp, _gr) = rhs.device_ptr(&stream);
    let (dp, _gd) = out.device_ptr_mut(&stream);
    let ls = lhs_l.stride();
    let rs = rhs_l.stride();
    let p = Problem {
        dt,
        batch: b as i64,
        m: m as i64,
        n: n as i64,
        k: k as i64,
        a_s0: ls[ls.len() - 2] as i64,
        a_s1: ls[ls.len() - 1] as i64,
        b_s0: rs[rs.len() - 2] as i64,
        b_s1: rs[rs.len() - 1] as i64,
        d_s0: n as i64,
        d_s1: 1,
        c_s0: n as i64,
        c_s1: 1,
        bias_s0: 0,
        bias_s1: 0,
        sa: sa as i64,
        sb: sb as i64,
        sd: (m * n) as i64,
        sc: (m * n) as i64,
        alpha: 1.0,
        beta: 0.0,
        has_c: false,
        has_bias: false,
        a_addr: lp + lhs_l.start_offset() as u64 * es,
        b_addr: rp + rhs_l.start_offset() as u64 * es,
        d_addr: dp,
    };
    gemm(dev, &p, 0, 0)?;
    drop((_gl, _gr, _gd));
    Ok(out)
}

/// Fills `n` elements of `data` (f32, or f64 when `f64_`) with Philox uniforms in (0, 1] or, with `normal` =
/// (mean, std), Box-Muller normals, continuing the device's counter (set_seed restarts it).
pub fn philox_fill<T: DeviceRepr>(
    dev: &CudaDevice,
    data: &mut CudaSlice<T>,
    n: usize,
    f64_: bool,
    normal: Option<(f64, f64)>,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let per = if f64_ { 2 } else { 4 };
    let threads = n.div_ceil(per) as u64;
    let (seed, offset) = dev.philox_take(threads);
    let stream = dev.cuda_stream();
    let name = match (f64_, normal.is_some()) {
        (false, false) => "philox_uniform_f32",
        (false, true) => "philox_normal_f32",
        (true, false) => "philox_uniform_f64",
        (true, true) => "philox_normal_f64",
    };
    let f = func(dev, name)?;
    let (ptr, _g) = data.device_ptr_mut(&stream);
    let n64 = n as u64;
    let (mean, std) = normal.unwrap_or((0.0, 1.0));
    let (mean32, std32) = (mean as f32, std as f32);
    let mut b = stream.launch_builder(&f);
    b.arg(&ptr).arg(&n64).arg(&seed).arg(&offset);
    if normal.is_some() {
        if f64_ {
            b.arg(&mean).arg(&std);
        } else {
            b.arg(&mean32).arg(&std32);
        }
    }
    let lc = LaunchConfig { grid_dim: (threads.div_ceil(256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    unsafe { b.launch(lc) }.w()?;
    Ok(())
}
