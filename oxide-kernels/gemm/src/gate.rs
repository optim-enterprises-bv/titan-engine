//! Gate for the titan noblas kernels (new kernels: there is no nvcc twin to kdiff against).
//!
//! (1) accuracy: every case runs the plan's kernel (and, forced, every other family / tile / split that can take it)
//!     and compares each output (sampled above 40k) with an f64 reference over the same rounded inputs:
//!       |D - ref| <= r_out |ref| + (k / 4 + 16) 2^-24 S + 3 * 2^-24 (|alpha acc| + |beta C| + |bias|) + tiny
//!     with S = |alpha| sum_k |A(i, k) B(k, j)|, r_out = 2^-8 (bf16) / 2^-11 (f16) / 2^-23 (f32): the output
//!     rounding plus a worst-case f32 accumulation bound (k / 16 tensor-core steps or k sequential FMAs with margin).
//!     Elements of D's buffer outside the view must keep their canary bytes; C may alias D (in place).
//! (2) mutations (each must FAIL the check): k - 1, a wrong batch stride, alpha x 1.01, the opposite B layout kernel,
//!     a dropped bias, a split-K partial left out.
//! (3) Philox: bits equal a host Philox4x32-10 replica, uniform in (0, 1], moments, counter continuity across calls,
//!     seed sensitivity; normals within 1e-5 of the host Box-Muller.
//! (4) GEMM_BENCH=<shapes file>: time per call vs cuBLAS (cublasGemmStridedBatchedEx, 32F compute, the call candle
//!     makes) on each inventoried shape; both sides' best of 3 x reps.
use crate::plan::{self, Dt, Force, Launch, Problem};
use kdiff::cuda_core::{CudaContext, CudaFunction, CudaModule, CudaStream, DeviceBuffer};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::c_void;
use std::sync::Arc;

pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn unif(&mut self) -> f32 {
        let u = (self.next() >> 40) as f32 / (1u64 << 24) as f32;
        2.0 * u - 1.0
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

pub fn f2bf(x: f32) -> u16 {
    let b = x.to_bits();
    ((b.wrapping_add(0x7fff + ((b >> 16) & 1))) >> 16) as u16
}
pub fn bf2f(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}
pub fn f2h(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let mut mant = b & 0x7f_ffff;
    if exp >= 31 {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        mant |= 0x80_0000;
        let shift = (14 - exp) as u32;
        let mut h = mant >> shift;
        let rem = mant & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        if rem > half || (rem == half && (h & 1) == 1) {
            h += 1;
        }
        return sign | h as u16;
    }
    let mut h = ((exp as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    sign | h as u16
}
pub fn h2f(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f64 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f64;
    let v = if e == 0 { m * 2f64.powi(-24) } else if e == 31 { f64::INFINITY } else { (1.0 + m / 1024.0) * 2f64.powi(e - 15) };
    (sign * v) as f32
}

/// Random host values of a type: (raw bytes, exact f64 values).
fn rand_vals(r: &mut Rng, dt: Dt, len: usize, scale: f32) -> (Vec<u8>, Vec<f64>) {
    let mut bytes = Vec::with_capacity(len * dt.size() as usize);
    let mut vals = Vec::with_capacity(len);
    for i in 0..len {
        let x = if i % 97 == 13 { 0.0 } else { r.unif() * scale };
        match dt {
            Dt::F32 => {
                bytes.extend_from_slice(&x.to_bits().to_le_bytes());
                vals.push(x as f64);
            }
            Dt::F16 => {
                let h = f2h(x);
                bytes.extend_from_slice(&h.to_le_bytes());
                vals.push(h2f(h) as f64);
            }
            Dt::Bf16 => {
                let h = f2bf(x);
                bytes.extend_from_slice(&h.to_le_bytes());
                vals.push(bf2f(h) as f64);
            }
        }
    }
    (bytes, vals)
}

fn decode(dt: Dt, bytes: &[u8], idx: usize) -> f64 {
    match dt {
        Dt::F32 => f32::from_le_bytes(bytes[4 * idx..4 * idx + 4].try_into().unwrap()) as f64,
        Dt::F16 => h2f(u16::from_le_bytes([bytes[2 * idx], bytes[2 * idx + 1]])) as f64,
        Dt::Bf16 => bf2f(u16::from_le_bytes([bytes[2 * idx], bytes[2 * idx + 1]])) as f64,
    }
}

fn extent(batch: i64, sz: i64, d0: i64, s0: i64, d1: i64, s1: i64) -> usize {
    if batch == 0 || d0 == 0 || d1 == 0 {
        return 1;
    }
    ((batch - 1) * sz + (d0 - 1) * s0 + (d1 - 1) * s1 + 1) as usize
}

pub struct H {
    pub ctx: Arc<CudaContext>,
    pub st: Arc<CudaStream>,
    pub module: Arc<CudaModule>,
    pub fns: std::cell::RefCell<BTreeMap<String, CudaFunction>>,
    pub used: std::cell::RefCell<BTreeSet<String>>,
    pub sms: i64,
}

impl H {
    fn func(&self, name: &str) -> CudaFunction {
        self.used.borrow_mut().insert(name.to_string());
        self.fns.borrow_mut().entry(name.to_string()).or_insert_with(|| self.module.load_function(name).unwrap_or_else(|e| panic!("{name}: {e:?}"))).clone()
    }
    fn launch(&self, name: &str, grid: (u32, u32, u32), args: &[u64]) {
        let f = self.func(name);
        let mut a: Vec<[u8; 8]> = args.iter().map(|v| v.to_le_bytes()).collect();
        let mut p: Vec<*mut c_void> = a.iter_mut().map(|v| v.as_mut_ptr() as *mut c_void).collect();
        unsafe { kdiff::cuda_core::simt::launch_kernel_on_stream(&f, grid, (256, 1, 1), 0, &self.st, &mut p).unwrap_or_else(|e| panic!("{name} {grid:?}: {e:?}")) };
    }
    /// Launches a plan (kernel + split-K merge); `ws` must hold `l.ws_floats` floats.
    pub fn run(&self, l: &Launch, c: u64, bias: u64, ws: u64) {
        let p = &l.p;
        let f = |x: f32| x.to_bits() as u64;
        let args = [
            p.a_addr, p.b_addr, p.d_addr, c, bias, p.m as u64, p.n as u64, p.k as u64, p.a_s0 as u64, p.a_s1 as u64,
            p.b_s0 as u64, p.b_s1 as u64, p.d_s0 as u64, p.d_s1 as u64, p.c_s0 as u64, p.c_s1 as u64, p.bias_s0 as u64,
            p.bias_s1 as u64, p.sa as u64, p.sb as u64, p.sd as u64, p.sc as u64, f(p.alpha), f(p.beta), ws, l.nsplit as u64,
        ];
        self.launch(&l.kernel, l.grid, &args);
        if let Some((rn, rg)) = &l.reduce {
            let args = [
                ws, l.nsplit as u64, p.d_addr, c, bias, p.m as u64, p.n as u64, p.d_s0 as u64, p.d_s1 as u64, p.c_s0 as u64,
                p.c_s1 as u64, p.bias_s0 as u64, p.bias_s1 as u64, p.sd as u64, p.sc as u64, f(p.alpha), f(p.beta),
            ];
            self.launch(rn, *rg, &args);
        }
    }
}

/// A test case: the problem without addresses plus the data options.
#[derive(Clone, Copy, Debug)]
pub struct Case {
    pub p: Problem,
    /// element offsets of a / b / d in their buffers (misalignment)
    pub a_off: usize,
    pub b_off: usize,
    pub d_off: usize,
    /// C aliases D (in place, C strides = D strides)
    pub c_alias: bool,
}

pub fn prob(dt: Dt, batch: i64, m: i64, n: i64, k: i64, at: bool, bt: bool) -> Problem {
    // A: k-contiguous rows of k (at: m-contiguous columns of m); B: k-contiguous columns (bt: n-contiguous rows)
    let (a_s0, a_s1) = if at { (1, m) } else { (k, 1) };
    let (b_s0, b_s1) = if bt { (n, 1) } else { (1, k) };
    Problem {
        dt,
        batch,
        m,
        n,
        k,
        a_s0,
        a_s1,
        b_s0,
        b_s1,
        d_s0: n,
        d_s1: 1,
        c_s0: n,
        c_s1: 1,
        bias_s0: 0,
        bias_s1: 1,
        sa: m * k,
        sb: k * n,
        sd: m * n,
        sc: m * n,
        alpha: 1.0,
        beta: 0.0,
        has_c: false,
        has_bias: false,
        a_addr: 0,
        b_addr: 0,
        d_addr: 0,
    }
}

pub struct Res {
    pub kernel: String,
    pub checked: usize,
    pub bad: usize,
    pub worst: f64,
    pub canary_bad: usize,
    pub first: String,
}

pub struct Data {
    pub a: (Vec<u8>, Vec<f64>),
    pub b: (Vec<u8>, Vec<f64>),
    pub c: (Vec<u8>, Vec<f64>),
    pub bias: (Vec<u8>, Vec<f64>),
    pub d_len: usize,
}

pub fn make(r: &mut Rng, cs: &Case) -> Data {
    let p = &cs.p;
    let a_len = cs.a_off + extent(p.batch, p.sa, p.m, p.a_s0, p.k, p.a_s1);
    let b_len = cs.b_off + extent(p.batch, p.sb, p.k, p.b_s0, p.n, p.b_s1);
    let d_len = cs.d_off + extent(p.batch, p.sd, p.m, p.d_s0, p.n, p.d_s1);
    let c_len = if cs.c_alias { d_len } else { extent(p.batch, p.sc, p.m, p.c_s0, p.n, p.c_s1) };
    let bias_len = extent(1, 0, p.m, p.bias_s0, p.n, p.bias_s1);
    Data {
        a: rand_vals(r, p.dt, a_len, 1.0),
        b: rand_vals(r, p.dt, b_len, 1.0),
        c: if p.has_c { rand_vals(r, p.dt, c_len, 1.0) } else { (vec![], vec![]) },
        bias: if p.has_bias { rand_vals(r, p.dt, bias_len, 1.0) } else { (vec![], vec![]) },
        d_len,
    }
}

/// Runs `cs` (plan with `force`, optionally a mutated launch) and checks it against the f64 reference of the
/// UNMUTATED case. `mutate`: 0 none, 1 k - 1 (k - 8 on tensor cores), 2 batch stride + 8, 3 alpha x 1.01, 4 opposite B-layout kernel,
/// 5 no bias, 6 split-K merge over nsplit - 1 partials.
pub fn check(h: &H, r: &mut Rng, cs: &Case, force: Force, mutate: u8, dat: Option<&Data>) -> Res {
    let own;
    let dat = match dat {
        Some(d) => d,
        None => {
            own = make(r, cs);
            &own
        }
    };
    let p0 = cs.p;
    let es = p0.dt.size() as usize;
    let st = &h.st;
    let da = DeviceBuffer::from_host(st, &dat.a.0).unwrap();
    let db = DeviceBuffer::from_host(st, &dat.b.0).unwrap();
    // D buffer: canary 0xff everywhere, then C's values in place when C aliases D
    let mut dinit = vec![0xffu8; dat.d_len * es];
    if p0.has_c && cs.c_alias {
        for z in 0..p0.batch {
            for i in 0..p0.m {
                for j in 0..p0.n {
                    let o = (cs.d_off as i64 + z * p0.sd + i * p0.d_s0 + j * p0.d_s1) as usize;
                    dinit[o * es..(o + 1) * es].copy_from_slice(&dat.c.0[o * es..(o + 1) * es]);
                }
            }
        }
    }
    let dd = DeviceBuffer::from_host(st, &dinit).unwrap();
    let dc = if p0.has_c && !cs.c_alias { Some(DeviceBuffer::from_host(st, &dat.c.0).unwrap()) } else { None };
    let dbias = if p0.has_bias { Some(DeviceBuffer::from_host(st, &dat.bias.0).unwrap()) } else { None };
    let mut p = p0;
    p.a_addr = da.cu_deviceptr() + (cs.a_off * es) as u64;
    p.b_addr = db.cu_deviceptr() + (cs.b_off * es) as u64;
    p.d_addr = dd.cu_deviceptr() + (cs.d_off * es) as u64;
    let c_addr = if !p0.has_c { 0 } else if cs.c_alias { p.d_addr } else { dc.as_ref().unwrap().cu_deviceptr() };
    if p0.has_c && cs.c_alias {
        p.c_s0 = p.d_s0;
        p.c_s1 = p.d_s1;
        p.sc = p.sd;
    }
    let bias_addr = dbias.as_ref().map(|b| b.cu_deviceptr()).unwrap_or(0);
    let mut l = plan::plan(&p, h.sms, force).unwrap_or_else(|e| panic!("plan {p:?}: {e}"));
    match mutate {
        // the last k element (tensor cores load 8-element chunks, so they drop the last chunk)
        1 => l.p.k -= if l.kernel.starts_with("tc_") { 8 } else { 1 },
        2 => {
            l.p.sa += 8;
            l.p.sb += 8;
        }
        3 => l.p.alpha *= 1.01,
        4 => {
            l.kernel = if l.kernel.contains("_kk_") {
                l.kernel.replace("_kk_", "_kn_")
            } else if l.kernel.contains("_kn_") {
                l.kernel.replace("_kn_", "_kk_")
            } else if l.kernel.contains("_mk_") {
                l.kernel.replace("_mk_", "_mn_")
            } else {
                l.kernel.replace("_mn_", "_mk_")
            }
        }
        _ => {}
    }
    let ws = DeviceBuffer::<u8>::from_host(st, &vec![0u8; (l.ws_floats.max(1)) * 4]).unwrap();
    let bias_launch = if mutate == 5 { 0 } else { bias_addr };
    if mutate == 6 && l.nsplit > 1 {
        let mut l2 = l.clone();
        // merge one partial fewer: launch the split kernel, then the merge with nsplit - 1
        l2.reduce = None;
        h.run(&l2, c_addr, bias_launch, ws.cu_deviceptr());
        let mut l3 = l.clone();
        l3.nsplit -= 1;
        let (rn, rg) = l.reduce.clone().unwrap();
        let pp = &l3.p;
        let f = |x: f32| x.to_bits() as u64;
        let args = [
            ws.cu_deviceptr(), l3.nsplit as u64, pp.d_addr, c_addr, bias_launch, pp.m as u64, pp.n as u64, pp.d_s0 as u64,
            pp.d_s1 as u64, pp.c_s0 as u64, pp.c_s1 as u64, pp.bias_s0 as u64, pp.bias_s1 as u64, pp.sd as u64, pp.sc as u64,
            f(pp.alpha), f(pp.beta),
        ];
        h.launch(&rn, rg, &args);
    } else {
        h.run(&l, c_addr, bias_launch, ws.cu_deviceptr());
    }
    st.synchronize().unwrap_or_else(|e| panic!("{}: {e:?}", l.kernel));
    let out = dd.to_host_vec(st).unwrap();
    // reference over the original problem
    let total = (p0.batch * p0.m * p0.n) as usize;
    let mut picks: Vec<(i64, i64, i64)> = Vec::new();
    if total <= 40_000 {
        for z in 0..p0.batch {
            for i in 0..p0.m {
                for j in 0..p0.n {
                    picks.push((z, i, j));
                }
            }
        }
    } else {
        for _ in 0..4000 {
            picks.push((r.below(p0.batch as u64) as i64, r.below(p0.m as u64) as i64, r.below(p0.n as u64) as i64));
        }
        for z in [0, p0.batch - 1] {
            for i in [0, p0.m - 1] {
                for j in [0, p0.n - 1] {
                    picks.push((z, i, j));
                }
            }
        }
    }
    let (r_out, tiny) = match p0.dt {
        Dt::F32 => (2f64.powi(-23), 1e-30),
        Dt::F16 => (2f64.powi(-11), 2f64.powi(-24)),
        Dt::Bf16 => (2f64.powi(-8), 1e-30),
    };
    let u = 2f64.powi(-24);
    let mut res = Res { kernel: l.kernel.clone(), checked: 0, bad: 0, worst: 0.0, canary_bad: 0, first: String::new() };
    for &(z, i, j) in &picks {
        let mut acc = 0f64;
        let mut s = 0f64;
        for kk in 0..p0.k {
            let a = dat.a.1[cs.a_off + (z * p0.sa + i * p0.a_s0 + kk * p0.a_s1) as usize];
            let b = dat.b.1[cs.b_off + (z * p0.sb + kk * p0.b_s0 + j * p0.b_s1) as usize];
            acc += a * b;
            s += (a * b).abs();
        }
        let alpha = p0.alpha as f64;
        let mut refv = alpha * acc;
        let mut mag = (alpha * acc).abs();
        if p0.has_c && p0.beta != 0.0 {
            let ci = if cs.c_alias {
                (cs.d_off as i64 + z * p0.sd + i * p0.d_s0 + j * p0.d_s1) as usize
            } else {
                (z * p0.sc + i * p0.c_s0 + j * p0.c_s1) as usize
            };
            let cv = p0.beta as f64 * dat.c.1[ci];
            refv += cv;
            mag += cv.abs();
        }
        if p0.has_bias {
            let bv = dat.bias.1[(i * p0.bias_s0 + j * p0.bias_s1) as usize];
            refv += bv;
            mag += bv.abs();
        }
        let o = (cs.d_off as i64 + z * p0.sd + i * p0.d_s0 + j * p0.d_s1) as usize;
        let got = decode(p0.dt, &out, o);
        let tol = r_out * refv.abs() + (p0.k as f64 / 4.0 + 16.0) * u * alpha.abs() * s + 3.0 * u * mag + tiny;
        let err = (got - refv).abs();
        let ratio = if err.is_nan() { f64::INFINITY } else { err / tol };
        res.checked += 1;
        if ratio > res.worst {
            res.worst = ratio;
        }
        if !(ratio <= 1.0) {
            res.bad += 1;
            if res.first.is_empty() {
                res.first = format!("({z},{i},{j}) got {got} ref {refv} tol {tol:.3e}");
            }
        }
    }
    // canary: D-buffer elements outside the view keep 0xff (small buffers only)
    if dat.d_len <= 4_000_000 {
        let mut inview = vec![false; dat.d_len];
        for z in 0..p0.batch {
            for i in 0..p0.m {
                for j in 0..p0.n {
                    inview[(cs.d_off as i64 + z * p0.sd + i * p0.d_s0 + j * p0.d_s1) as usize] = true;
                }
            }
        }
        for (e, v) in inview.iter().enumerate() {
            if !v && out[e * es..(e + 1) * es].iter().any(|&x| x != 0xff) {
                res.canary_bad += 1;
            }
        }
    }
    res
}

// ------------------------------------------------------------------------------------------------ cuBLAS (bench)
type CreateFn = unsafe extern "C" fn(*mut *mut c_void) -> i32;
type SetStreamFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;
#[allow(clippy::type_complexity)]
type GemmExFn = unsafe extern "C" fn(
    *mut c_void, i32, i32, i32, i32, i32, *const c_void, *const c_void, i32, i32, i64, *const c_void, i32, i32, i64,
    *const c_void, *mut c_void, i32, i32, i64, i32, i32, i32,
) -> i32;
unsafe extern "C" {
    fn dlopen(name: *const std::ffi::c_char, flags: i32) -> *mut c_void;
    fn dlsym(h: *mut c_void, name: *const std::ffi::c_char) -> *mut c_void;
}
pub struct Cublas {
    h: *mut c_void,
    gemm: GemmExFn,
}
impl Cublas {
    pub fn new(st: &CudaStream) -> Option<Self> {
        unsafe {
            let lib = dlopen(c"libcublas.so.13".as_ptr(), 2);
            if lib.is_null() {
                return None;
            }
            let create: CreateFn = std::mem::transmute(dlsym(lib, c"cublasCreate_v2".as_ptr()));
            let set: SetStreamFn = std::mem::transmute(dlsym(lib, c"cublasSetStream_v2".as_ptr()));
            let gemm: GemmExFn = std::mem::transmute(dlsym(lib, c"cublasGemmStridedBatchedEx".as_ptr()));
            let mut h = std::ptr::null_mut();
            if create(&mut h) != 0 {
                return None;
            }
            set(h, st.cu_stream() as *mut c_void);
            Some(Self { h, gemm })
        }
    }
    /// candle's call for the row-major problem (D contiguous along n); None if cuBLAS cannot express it.
    pub fn gemm(&self, p: &Problem, c_in_d: bool) -> Option<i32> {
        if p.d_s1 != 1 {
            return None;
        }
        let (ta, lda) = if p.b_s1 == 1 { (0, p.b_s0) } else if p.b_s0 == 1 { (1, p.b_s1) } else { return None };
        let (tb, ldb) = if p.a_s1 == 1 { (0, p.a_s0) } else if p.a_s0 == 1 { (1, p.a_s1) } else { return None };
        let rows_a = if ta == 0 { p.n } else { p.k };
        let rows_b = if tb == 0 { p.k } else { p.m };
        let (lda, ldb) = (lda.max(rows_a).max(1), ldb.max(rows_b).max(1));
        let ty = match p.dt {
            Dt::F32 => 0,
            Dt::F16 => 2,
            Dt::Bf16 => 14,
        };
        let alpha = p.alpha;
        let beta = if c_in_d { p.beta } else { 0.0 };
        let st = unsafe {
            (self.gemm)(
                self.h, ta, tb, p.n as i32, p.m as i32, p.k as i32, &alpha as *const f32 as *const c_void, p.b_addr as *const c_void, ty,
                lda as i32, p.sb, p.a_addr as *const c_void, ty, ldb as i32, p.sa, &beta as *const f32 as *const c_void,
                p.d_addr as *mut c_void, ty, p.d_s0.max(p.n) as i32, p.sd, p.batch as i32, 68, 99,
            )
        };
        Some(st)
    }
}

// ------------------------------------------------------------------------------------------------ Philox
fn philox_host(ctr: u64, seed: u64) -> [u32; 4] {
    let (mut c0, mut c1, mut c2, mut c3) = (ctr as u32, (ctr >> 32) as u32, 0u32, 0u32);
    let (mut k0, mut k1) = (seed as u32, (seed >> 32) as u32);
    for _ in 0..10 {
        let p0 = 0xD251_1F53u64 * c0 as u64;
        let p1 = 0xCD9E_8D57u64 * c2 as u64;
        let (hi0, lo0, hi1, lo1) = ((p0 >> 32) as u32, p0 as u32, (p1 >> 32) as u32, p1 as u32);
        let n0 = hi1 ^ c1 ^ k0;
        let n2 = hi0 ^ c3 ^ k1;
        c0 = n0;
        c1 = lo1;
        c2 = n2;
        c3 = lo0;
        k0 = k0.wrapping_add(0x9E37_79B9);
        k1 = k1.wrapping_add(0xBB67_AE85);
    }
    [c0, c1, c2, c3]
}

fn philox_gate(h: &H) -> bool {
    let st = &h.st;
    let mut ok = true;
    let genr = |name: &str, n: usize, seed: u64, offset: u64, esz: usize, per: usize, extra: &[u64]| -> Vec<u8> {
        let out = DeviceBuffer::<u8>::from_host(st, &vec![0u8; n * esz + 8]).unwrap();
        let mut args = vec![out.cu_deviceptr(), n as u64, seed, offset];
        args.extend_from_slice(extra);
        h.launch(name, ((n.div_ceil(per)).div_ceil(256).max(1) as u32, 1, 1), &args);
        st.synchronize().unwrap();
        out.to_host_vec(st).unwrap()
    };
    let n = 1_000_003usize;
    let seed = 0x5EED_0000_2A2Au64;
    // uniform f32: exact host replica, range, moments
    let u = genr("philox_uniform_f32", n, seed, 7, 4, 4, &[]);
    let uf: Vec<f32> = (0..n).map(|i| f32::from_le_bytes(u[4 * i..4 * i + 4].try_into().unwrap())).collect();
    let mut bad = 0;
    for (i, &v) in uf.iter().enumerate().step_by(997) {
        let x = philox_host(7 + (i / 4) as u64, seed)[i % 4];
        if v.to_bits() != (((x >> 8) + 1) as f32 * 2f32.powi(-24)).to_bits() {
            bad += 1;
        }
    }
    let inrange = uf.iter().all(|&v| v > 0.0 && v <= 1.0);
    let mean = uf.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let var = uf.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n as f64;
    let g = bad == 0 && inrange && (mean - 0.5).abs() < 0.002 && (var - 1.0 / 12.0).abs() < 0.001;
    println!("philox uniform f32: host replica mismatches {bad}, in (0, 1] {inrange}, mean {mean:.5}, var {var:.5} -> {}", if g { "PASS" } else { "FAIL" });
    ok &= g;
    // determinism and counter continuity: [offset 7, 2048 values] ++ [offset 7 + 512, 2048] == [offset 7, 4096]
    let a1 = genr("philox_uniform_f32", 2048, seed, 7, 4, 4, &[]);
    let a2 = genr("philox_uniform_f32", 2048, seed, 7 + 512, 4, 4, &[]);
    let whole = genr("philox_uniform_f32", 4096, seed, 7, 4, 4, &[]);
    let cont = a1[..8192] == whole[..8192] && a2[..8192] == whole[8192..16384] && a1[..8192] == u[..8192];
    let other = genr("philox_uniform_f32", 2048, seed ^ 1, 7, 4, 4, &[]);
    let differs = other[..8192] != a1[..8192];
    // mutation: the host replica with a different seed must not match
    let mm = (0..64).filter(|&i| {
        let x = philox_host(7 + (i / 4) as u64, seed ^ 1)[i % 4];
        uf[i].to_bits() == (((x >> 8) + 1) as f32 * 2f32.powi(-24)).to_bits()
    }).count();
    let g = cont && differs && mm < 4;
    println!("philox determinism / counter continuity {cont}, seed sensitivity {differs}, wrong-seed replica matches {mm}/64 -> {}", if g { "PASS" } else { "FAIL" });
    ok &= g;
    // normal f32: Box-Muller from the same counters, moments
    let mean_p = 0.25f32;
    let std_p = 2.0f32;
    let nv = genr("philox_normal_f32", n, seed, 3, 4, 4, &[mean_p.to_bits() as u64, std_p.to_bits() as u64]);
    let nf: Vec<f64> = (0..n).map(|i| f32::from_le_bytes(nv[4 * i..4 * i + 4].try_into().unwrap()) as f64).collect();
    let mut worst = 0f64;
    for i in (0..n).step_by(1009) {
        let x = philox_host(3 + (i / 4) as u64, seed);
        let (u0, u1) = if i % 4 < 2 { (x[0], x[1]) } else { (x[2], x[3]) };
        let uu0 = ((u0 >> 8) + 1) as f64 * 2f64.powi(-24);
        let uu1 = ((u1 >> 8) + 1) as f64 * 2f64.powi(-24);
        let rr = (-2.0 * uu0.ln()).sqrt();
        let a = 2.0 * std::f64::consts::PI * uu1;
        let z = if i % 2 == 0 { rr * a.cos() } else { rr * a.sin() };
        let want = mean_p as f64 + std_p as f64 * z;
        worst = worst.max((nf[i] - want).abs() / (1.0 + want.abs()));
    }
    let m = nf.iter().sum::<f64>() / n as f64;
    let v = nf.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n as f64;
    let g = worst < 1e-5 && (m - 0.25).abs() < 0.01 && (v - 4.0).abs() < 0.03;
    println!("philox normal f32 (mean 0.25, std 2): worst vs host Box-Muller {worst:.2e}, mean {m:.4}, var {v:.4} -> {}", if g { "PASS" } else { "FAIL" });
    ok &= g;
    // f64 uniform / normal
    let n2 = 200_001usize;
    let u = genr("philox_uniform_f64", n2, seed, 11, 8, 2, &[]);
    let ud: Vec<f64> = (0..n2).map(|i| f64::from_le_bytes(u[8 * i..8 * i + 8].try_into().unwrap())).collect();
    let mut bad = 0;
    for i in (0..n2).step_by(101) {
        let x = philox_host(11 + (i / 2) as u64, seed);
        let (lo, hi) = if i % 2 == 0 { (x[0], x[1]) } else { (x[2], x[3]) };
        let want = ((((hi as u64) << 21) ^ (lo as u64 >> 11)) + 1) as f64 * 2f64.powi(-53);
        if ud[i].to_bits() != want.to_bits() {
            bad += 1;
        }
    }
    let inr = ud.iter().all(|&v| v > 0.0 && v <= 1.0);
    let mean = ud.iter().sum::<f64>() / n2 as f64;
    let nv = genr("philox_normal_f64", n2, seed, 5, 8, 2, &[0f64.to_bits(), 1f64.to_bits()]);
    let nd: Vec<f64> = (0..n2).map(|i| f64::from_le_bytes(nv[8 * i..8 * i + 8].try_into().unwrap())).collect();
    let m = nd.iter().sum::<f64>() / n2 as f64;
    let v = nd.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n2 as f64;
    let g = bad == 0 && inr && (mean - 0.5).abs() < 0.005 && m.abs() < 0.01 && (v - 1.0).abs() < 0.02;
    println!("philox f64: uniform replica mismatches {bad}, in (0, 1] {inr}, mean {mean:.4}; normal mean {m:.4} var {v:.4} -> {}", if g { "PASS" } else { "FAIL" });
    ok &= g;
    ok
}

// ------------------------------------------------------------------------------------------------ cases
fn cases(quick: bool) -> Vec<(String, Case)> {
    let mut v = Vec::new();
    let mut add = |tag: &str, p: Problem| {
        v.push((tag.to_string(), Case { p, a_off: 0, b_off: 0, d_off: 0, c_alias: false }));
    };
    let shapes: &[(i64, i64, i64, i64)] = &[
        (1, 1, 1, 1),
        (1, 1, 7, 13),
        (1, 3, 256, 2048),
        (1, 8, 512, 264),
        (1, 1, 300, 4096),
        (1, 9, 33, 40),
        (1, 37, 129, 72),
        (1, 128, 128, 64),
        (2, 200, 300, 520),
        (1, 512, 32, 2048),
        (1, 512, 1, 2048),
        (3, 64, 72, 16),
        (16, 27, 27, 256),
        (16, 27, 256, 27),
        (1, 1000, 520, 136),
        (1, 2, 4096, 256),
    ];
    for dt in [Dt::F32, Dt::F16, Dt::Bf16] {
        for &(b, m, n, k) in shapes {
            for (at, bt) in [(false, false), (false, true), (true, false), (true, true)] {
                if quick && (at || bt) && m * n * k > 100_000 {
                    continue;
                }
                add(&format!("{} b{b} {m}x{n}x{k} {}{}", dt.name(), if at { "m" } else { "k" }, if bt { "n" } else { "k" }), prob(dt, b, m, n, k, at, bt));
            }
        }
        // epilogue: alpha, beta with C (separate and in place), bias by column and by row
        let mut p = prob(dt, 2, 70, 90, 48, false, false);
        p.alpha = 0.5;
        p.beta = -1.5;
        p.has_c = true;
        add(&format!("{} epi alpha/beta/C", dt.name()), p);
        p.has_bias = true;
        add(&format!("{} epi alpha/beta/C/bias", dt.name()), p);
        let mut q = prob(dt, 1, 5, 300, 512, false, false);
        q.has_bias = true;
        q.bias_s0 = 1;
        q.bias_s1 = 0;
        add(&format!("{} gemv bias by row", dt.name()), q);
        // GQA decode shape: B broadcast over the batch (folds into rows)
        let mut g = prob(dt, 8, 1, 700, 256, false, false);
        g.sb = 0;
        g.alpha = 0.0625;
        add(&format!("{} gqa scores", dt.name()), g);
        let mut g2 = prob(dt, 6, 1, 256, 45, false, false);
        g2.sb = 0;
        add(&format!("{} gqa pv", dt.name()), g2);
        // strided D (gaps between rows: canary), broadcast A
        let mut s = prob(dt, 2, 40, 24, 32, false, true);
        s.d_s0 = 40;
        s.sd = 40 * 40 + 8;
        add(&format!("{} strided D", dt.name()), s);
        let mut s2 = prob(dt, 3, 20, 48, 64, false, false);
        s2.sa = 0;
        add(&format!("{} broadcast A", dt.name()), s2);
        // transposed D (column-major output)
        let mut t = prob(dt, 1, 50, 70, 40, false, false);
        t.d_s0 = 1;
        t.d_s1 = 50;
        add(&format!("{} column-major D", dt.name()), t);
    }
    v
}

pub fn run() -> bool {
    let root = env!("CARGO_MANIFEST_DIR");
    let ctx = CudaContext::new(0).expect("cuda context");
    let st = ctx.default_stream();
    let t = std::time::Instant::now();
    let module = ctx.load_module_from_file(&format!("{root}/gemm.ptx")).expect("load gemm.ptx");
    println!("gemm.ptx: loaded (JIT) in {:.2}s", t.elapsed().as_secs_f64());
    let sms = ctx.multiprocessor_count().unwrap() as i64;
    let h = H { ctx: ctx.clone(), st: st.clone(), module, fns: Default::default(), used: Default::default(), sms };
    let quick = std::env::var("GEMM_QUICK").is_ok_and(|v| v == "1");
    let mut ok = true;
    if let Ok(path) = std::env::var("GEMM_BENCH") {
        return crate::bench::bench(&h, &path);
    }
    ok &= philox_gate(&h);

    let mut r = Rng(0x7177_A1B5);
    let (mut n_cases, mut n_bad_cases, mut worst, mut checked) = (0usize, 0usize, 0f64, 0usize);
    let mut worst_by: BTreeMap<String, (f64, String)> = BTreeMap::new();
    let forces = [
        Force::default(),
        Force { family: 1, ..Default::default() },
        Force { family: 2, ..Default::default() },
        Force { family: 3, tile: 128, ..Default::default() },
        Force { family: 3, tile: 64, ..Default::default() },
        Force { family: 3, tile: 32, nsplit: 3 },
        Force { family: 4, tile: 128, ..Default::default() },
        Force { family: 4, tile: 64, nsplit: 2 },
        Force { family: 5, ..Default::default() },
    ];
    for (tag, base) in cases(quick) {
        for (oi, off) in [(0usize, 0usize), (1, 0), (0, 1)].iter().enumerate() {
            if oi > 0 && base.p.m * base.p.n * base.p.k > 3_000_000 {
                continue;
            }
            let mut cs = base;
            cs.a_off = off.0;
            cs.b_off = off.1;
            for alias in [false, true] {
                if alias && !cs.p.has_c {
                    continue;
                }
                cs.c_alias = alias;
                let dat = make(&mut r, &cs);
                let mut seen = BTreeSet::new();
                for f in forces {
                    let es = cs.p.dt.size() as usize;
                    let pl = match plan::plan(&Problem { a_addr: (cs.a_off * es) as u64, b_addr: (cs.b_off * es) as u64, ..cs.p }, sms, f) {
                        Ok(l) => l,
                        Err(_) => continue,
                    };
                    if !seen.insert((pl.kernel.clone(), pl.nsplit)) {
                        continue;
                    }
                    let res = check(&h, &mut r, &cs, f, 0, Some(&dat));
                    n_cases += 1;
                    checked += res.checked;
                    worst = worst.max(res.worst);
                    let fam = res.kernel.split('_').next().unwrap_or("").to_string() + " " + cs.p.dt.name();
                    let e = worst_by.entry(fam).or_insert((0.0, String::new()));
                    if res.worst > e.0 {
                        *e = (res.worst, format!("{tag} via {} split {}", res.kernel, pl.nsplit));
                    }
                    if res.bad > 0 || res.canary_bad > 0 {
                        n_bad_cases += 1;
                        ok = false;
                        println!(
                            "FAIL {tag} off {off:?} alias {alias} force {f:?}: {} split {}: {} / {} outside tol (worst {:.2}), canary {}; first {}",
                            res.kernel, pl.nsplit, res.bad, res.checked, res.worst, res.canary_bad, res.first
                        );
                    }
                }
            }
        }
    }
    for (fam, (w, at)) in &worst_by {
        println!("  worst error / tolerance {fam:<14} {w:.3}  ({at})");
    }
    println!("accuracy: {n_cases} launches, {checked} outputs checked, worst error / tolerance {worst:.3}, {n_bad_cases} failing -> {}", if n_bad_cases == 0 { "PASS" } else { "FAIL" });

    // mutations: each must fail
    let mut n_mut = 0;
    let mut caught = 0;
    let mut mutants: Vec<(u8, Case, Force)> = Vec::new();
    for dt in [Dt::F32, Dt::F16, Dt::Bf16] {
        let p = prob(dt, 2, 96, 136, 264, false, false);
        let cs = Case { p, a_off: 0, b_off: 0, d_off: 0, c_alias: false };
        for f in [Force::default(), Force { family: 4, tile: 64, ..Default::default() }, Force { family: 3, tile: 64, ..Default::default() }] {
            if dt == Dt::F32 && f.family == 3 {
                continue;
            }
            // (a SIMT kernel's layout only picks its thread mapping, so the opposite-layout mutant is a tensor-core one)
            for mt in [1u8, 2, 3, 4] {
                if mt == 4 && (dt == Dt::F32 || f.family == 4) {
                    continue;
                }
                mutants.push((mt, cs, f));
            }
        }
        let g = Case { p: prob(dt, 1, 3, 300, 1024, false, false), a_off: 0, b_off: 0, d_off: 0, c_alias: false };
        for mt in [1u8, 2, 3] {
            mutants.push((mt, Case { p: Problem { batch: 2, ..g.p }, ..g }, Force::default()));
        }
        let mut bp = prob(dt, 1, 40, 64, 128, false, false);
        bp.has_bias = true;
        mutants.push((5, Case { p: bp, a_off: 0, b_off: 0, d_off: 0, c_alias: false }, Force::default()));
        let sk = Case { p: prob(dt, 1, 64, 64, 2048, false, false), a_off: 0, b_off: 0, d_off: 0, c_alias: false };
        mutants.push((6, sk, Force { family: 4, tile: 64, nsplit: 4 }));
        if dt != Dt::F32 {
            mutants.push((6, sk, Force { family: 3, tile: 64, nsplit: 4 }));
        }
    }
    for (mt, cs, f) in mutants {
        let res = check(&h, &mut r, &cs, f, mt, None);
        n_mut += 1;
        if res.bad > 0 {
            caught += 1;
        } else {
            println!("MUTANT SURVIVED: mutation {mt} on {:?} {}x{}x{} via {} (worst {:.3})", cs.p.dt, cs.p.m, cs.p.n, cs.p.k, res.kernel, res.worst);
        }
    }
    let g = caught == n_mut;
    println!("mutations: {caught}/{n_mut} caught -> {}", if g { "PASS" } else { "FAIL" });
    ok &= g;

    // coverage: every GEMM / GEMV kernel of the module exercised
    let all: Vec<String> = std::fs::read_to_string(format!("{root}/gemm.ptx"))
        .unwrap()
        .lines()
        .filter_map(|l| l.strip_prefix(".visible .entry ").or_else(|| l.strip_prefix(".entry ")))
        .map(|l| l.split('(').next().unwrap().trim().to_string())
        .collect();
    let used = h.used.borrow();
    let missing: Vec<&String> = all.iter().filter(|k| !used.contains(*k)).collect();
    println!("coverage: {} of {} entries launched; not launched: {:?}", all.len() - missing.len(), all.len(), missing);
    ok &= missing.is_empty();
    drop(used);
    for k in all.iter().filter(|k| k.starts_with("tc_") || k.starts_with("simt_") || k.ends_with("8_f32_v") || k.ends_with("8_bf16_v")) {
        let f = h.func(k);
        println!("  {k}: {} registers, {} B local (spills)", f.num_registers().unwrap_or(0), f.local_size_bytes().unwrap_or(0));
    }

    println!("gemm gate -> {}", if ok { "PASS" } else { "FAIL" });
    ok
}
