//! Differential gate: every oxide entry against the reference instance in the rebuilt llama.cpp
//! acecd56 cubins (ref/quantize.cubin, ref/mmq_<fmt>.cubin; SASS identical to libggml-cuda.so).
//! Both sides get the same packed parameter buffer (CU_LAUNCH_PARAM_BUFFER_POINTER), private
//! copies of the same input bytes and a pre-filled output buffer; every output byte is compared.
//!
//! - quantizers: `<fmt>_quantize_mmq*_{f32,f16,bf16}` vs `quantize_mmq_q8_1<D4>` (IQ4_NL),
//!   `quantize_mmq_mxfp4<false>`, `quantize_mmq_nvfp4<false, aligned>` (both variants; vy and the
//!   per-row scale), the half inputs widened on the host for the reference (exact).
//! - MMQ: `<fmt>_mmq_j*` + `<fmt>_mmq_fixup_j*` vs `mul_mat_q<type, J, fb>` +
//!   `mul_mat_q_stream_k_fixup`, launched with mmq.cuh's host logic (J choice, stream-k grid,
//!   fixup grid, shared memory) and over forced grids; NVFP4 with and without `y_scale`.
//! - IQ4_XS (type 23): `iq4_xs_mmq_j*` (+ fixups) vs `mul_mat_q<IQ4_XS, J, fb>` from
//!   ref/mmq_iq4_xs.cubin (ref/nvcc_mmq_iq4_xs.sh); its activations are IQ4_NL's quantizer output
//!   (block_q8_1_mmq D4), so the quantizers are gated once, under IQ4_NL. Two loader mutants
//!   (`iq4_xs_mut{1,2}_mmq_j32_f0`) must fail.
use cuda_core::sys as cu;
use kdiff::{Rng, Tally};
use std::ffi::c_void;

macro_rules! ck {
    ($e:expr) => {{
        let r = $e;
        assert_eq!(r, cu::cudaError_enum_CUDA_SUCCESS, "{}", stringify!($e));
    }};
}

/// llama.cpp `init_fastdiv_values`: <mp, L, d>.
pub fn fastdiv_values(d: u64) -> [u32; 3] {
    assert!(d != 0 && d <= u32::MAX as u64);
    let d = d as u32;
    let mut l = 0u32;
    while l < 32 && (1u64 << l) < d as u64 {
        l += 1;
    }
    let mp = (((1u64 << 32) * ((1u64 << l) - d as u64)) / d as u64 + 1) as u32;
    [mp, l, d]
}

/// Packed kernel parameters, laid out with C alignment.
#[derive(Default, Clone)]
pub struct Params(pub Vec<u8>);
impl Params {
    pub fn put<T: Copy>(&mut self, v: T) -> &mut Self {
        let a = std::mem::align_of::<T>();
        while self.0.len() % a != 0 {
            self.0.push(0);
        }
        let b = unsafe { std::slice::from_raw_parts(&v as *const T as *const u8, std::mem::size_of::<T>()) };
        self.0.extend_from_slice(b);
        self
    }
}

pub fn load(path: &str) -> cu::CUmodule {
    let mut img = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    img.push(0);
    let mut m: cu::CUmodule = std::ptr::null_mut();
    unsafe { ck!(cu::cuModuleLoadData(&mut m, img.as_ptr() as *const c_void)) };
    m
}

pub fn func(m: cu::CUmodule, name: &str) -> cu::CUfunction {
    let mut f: cu::CUfunction = std::ptr::null_mut();
    let c = std::ffi::CString::new(name).unwrap();
    let r = unsafe { cu::cuModuleGetFunction(&mut f, m, c.as_ptr()) };
    assert_eq!(r, cu::cudaError_enum_CUDA_SUCCESS, "missing function {name}");
    f
}

pub struct Buf(pub cu::CUdeviceptr, pub usize);
impl Buf {
    pub fn up(b: &[u8]) -> Buf {
        let mut p: cu::CUdeviceptr = 0;
        unsafe {
            ck!(cu::cuMemAlloc_v2(&mut p, b.len().max(1)));
            ck!(cu::cuMemcpyHtoD_v2(p, b.as_ptr() as *const c_void, b.len()));
        }
        Buf(p, b.len())
    }
    pub fn zeros(n: usize) -> Buf {
        Buf::up(&vec![0u8; n])
    }
    pub fn down(&self) -> Vec<u8> {
        let mut h = vec![0u8; self.1];
        unsafe {
            ck!(cu::cuCtxSynchronize());
            ck!(cu::cuMemcpyDtoH_v2(h.as_mut_ptr() as *mut c_void, self.0, h.len()));
        }
        h
    }
}
impl Drop for Buf {
    fn drop(&mut self) {
        unsafe { cu::cuMemFree_v2(self.0) };
    }
}

/// Oxide kernel launches (the gate's launch count; each is paired with a reference launch).
pub static mut OX_LAUNCHES: usize = 0;

/// Launch from a packed parameter buffer (CU_LAUNCH_PARAM_BUFFER_POINTER), then synchronize.
pub fn launch(f: cu::CUfunction, grid: (u32, u32, u32), block: (u32, u32, u32), smem: u32, p: &Params) {
    unsafe {
        if smem > 48 * 1024 {
            ck!(cu::cuFuncSetAttribute(f, cu::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32));
        }
        let mut pb = p.0.clone();
        let mut size = pb.len();
        let mut extra: [*mut c_void; 5] =
            [1 as *mut c_void, pb.as_mut_ptr() as *mut c_void, 2 as *mut c_void, &mut size as *mut usize as *mut c_void, std::ptr::null_mut()];
        ck!(cu::cuLaunchKernel(f, grid.0, grid.1, grid.2, block.0, block.1, block.2, smem, std::ptr::null_mut(), std::ptr::null_mut(), extra.as_mut_ptr()));
        ck!(cu::cuCtxSynchronize());
    }
}
fn launch_ox(f: cu::CUfunction, grid: (u32, u32, u32), block: (u32, u32, u32), smem: u32, p: &Params) {
    unsafe { OX_LAUNCHES += 1 };
    launch(f, grid, block, smem, p);
}

fn cmp(t: &mut Tally, label: &str, a: &[u8], b: &[u8]) -> usize {
    let mut d = kdiff::Diff { bytes: a.len(), differing: 0, first: None };
    for i in 0..a.len() {
        if a[i] != b[i] {
            d.differing += 1;
            if d.first.is_none() {
                d.first = Some((0, i, a[i], b[i]));
            }
        }
    }
    let n = d.differing;
    t.record(label, &d);
    n
}

/// f32 activations: mostly N(0, s)-like magnitudes, `1/rate` specials.
fn acts(r: &mut Rng, n: usize, rate: u64, nan: bool) -> Vec<f32> {
    const SP: [f32; 10] = [0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 1e-40, -1e-42, 3e38, -3e38, 0.49999997];
    (0..n)
        .map(|_| {
            if r.next() % rate == 0 {
                let mut s = SP[(r.next() % 10) as usize];
                if !nan && s.is_nan() {
                    s = 7.5;
                }
                return s;
            }
            let u = (r.next() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
            let e = (r.next() % 12) as i32 - 6;
            (u * 2f64.powi(e)) as f32
        })
        .collect()
}

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h as u32) & 0x8000) << 16;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    if e == 0 {
        if m == 0 {
            return f32::from_bits(s);
        }
        let v = m as f32 * 2f32.powi(-24);
        return if s != 0 { -v } else { v };
    }
    if e == 31 {
        return f32::from_bits(s | 0x7f80_0000 | (m << 13));
    }
    f32::from_bits(s | ((e + 112) << 23) | (m << 13))
}
/// Any f16 bit pattern close to `v` (truncation is fine: only the device reads it back).
fn f32_to_f16_trunc(v: f32) -> u16 {
    let b = v.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ffff;
    if e == 255 {
        return s | 0x7c00 | if m != 0 { 0x200 } else { 0 };
    }
    let e2 = e - 127 + 15;
    if e2 >= 31 {
        return s | 0x7c00;
    }
    if e2 <= 0 {
        if e2 < -10 {
            return s;
        }
        let m = (m | 0x80_0000) >> (1 - e2);
        return s | (m >> 13) as u16;
    }
    s | ((e2 as u16) << 10) | (m >> 13) as u16
}

/// Input of `n` values as (oxide input bytes, f32 bytes for the reference); `it`: 0 f32, 1 f16,
/// 2 bf16 (no NaN in the half inputs: the host widening is then exactly cvt.f32.{f16,bf16}).
fn input(r: &mut Rng, it: usize, n: usize, rate: u64) -> (Vec<u8>, Vec<u8>) {
    let v = acts(r, n, rate, it == 0);
    match it {
        0 => {
            let b = kdiff::as_bytes(&v);
            (b.clone(), b)
        }
        1 => {
            let h: Vec<u16> = v.iter().map(|&x| f32_to_f16_trunc(x)).collect();
            let f: Vec<f32> = h.iter().map(|&x| f16_to_f32(x)).collect();
            (kdiff::as_bytes(&h), kdiff::as_bytes(&f))
        }
        _ => {
            let h: Vec<u16> = v.iter().map(|&x| (x.to_bits() >> 16) as u16).collect();
            let f: Vec<f32> = h.iter().map(|&x| f32::from_bits((x as u32) << 16)).collect();
            (kdiff::as_bytes(&h), kdiff::as_bytes(&f))
        }
    }
}

pub struct M {
    pub quant: cu::CUmodule,
    pub mmq: [cu::CUmodule; 4],
    pub ox: cu::CUmodule,
}

const FMT: [&str; 4] = ["iq4_nl", "mxfp4", "nvfp4", "iq4_xs"];
const GGML_TYPE: [u32; 4] = [20, 39, 40, 23];
const QK: [i32; 4] = [32, 32, 64, 256];
const BYTES: [usize; 4] = [18, 17, 36, 136];
/// values per activation block (block_q8_1_mmq / block_fp4_mmq)
const NEB: [i64; 4] = [128, 256, 256, 128];
const TY: [&str; 3] = ["f32", "f16", "bf16"];

// ---------------------------------------------------------------------------------------------
// quantizers

struct QCase {
    ne00: i64,
    ne1: i64,
    ne2: i64,
    ne3: i64,
    pad: i64,
    ids: bool,
}

fn quant_case(m: &M, t: &mut Tally, r: &mut Rng, f: usize, it: usize, c: &QCase, rate: u64) {
    let ne0 = (c.ne00 + 511) / 512 * 512;
    let s01 = c.ne00 + 8 * c.pad; // keeps 32-byte row alignment for the aligned NVFP4 variant
    let nrows_src = if c.ids { c.ne1 + 3 } else { c.ne1 };
    let s02 = s01 * nrows_src;
    let s03 = s02 * c.ne2;
    let nx = (s03 * c.ne3) as usize + 16;
    let (x_in, x_f32) = input(r, it, nx, rate);
    let x_in = Buf::up(&x_in);
    let x_f32 = Buf::up(&x_f32);
    let idsv: Vec<i32> = (0..c.ne1).map(|_| (r.next() % nrows_src as u64) as i32).collect();
    let idb = Buf::up(&kdiff::as_bytes(&idsv));
    let idp = if c.ids { idb.0 } else { 0u64 };
    let out_bytes = (ne0 / NEB[f]) as usize * 144 * (c.ne1 * c.ne2 * c.ne3) as usize;
    let fill = r.bytes(out_bytes);
    let (ya, yb) = (Buf::up(&fill), Buf::up(&fill));
    let sfill = r.bytes((c.ne1 * c.ne2 * c.ne3) as usize * 4);
    let (sa, sb) = (Buf::up(&sfill), Buf::up(&sfill));
    let label = format!("{}_quantize {} ne00={} ne1={} ne2={} ne3={} ids={} rate={rate}", FMT[f], TY[it], c.ne00, c.ne1, c.ne2, c.ne3, c.ids);
    match f {
        0 | 1 => {
            let params = |x: &Buf, y: &Buf| {
                let mut p = Params::default();
                p.put(x.0).put(idp).put(y.0).put(c.ne00).put(s01).put(s02).put(s03).put(ne0).put(c.ne1 as i32).put(c.ne2 as i32).put(0i32);
                p
            };
            let (rname, oname, grid, block) = if f == 0 {
                (
                    "_Z17quantize_mmq_q8_1IL18mmq_q8_1_ds_layout0ELb0EEvPKfPKiPvllllliii".to_string(),
                    format!("iq4_nl_quantize_mmq_d4_{}", TY[it]),
                    (c.ne1 as u32, ((ne0 + 511) / 512) as u32, (c.ne2 * c.ne3) as u32),
                    (128, 1, 1),
                )
            } else {
                (
                    "_Z18quantize_mmq_mxfp4ILb0EEvPKfPKiPvllllliii".to_string(),
                    format!("mxfp4_quantize_mmq_{}", TY[it]),
                    (c.ne1 as u32, ((ne0 + 511) / 512) as u32, (c.ne2 * c.ne3) as u32),
                    (32, 8, 1),
                )
            };
            launch(func(m.quant, &rname), grid, block, 0, &params(&x_f32, &ya));
            launch_ox(func(m.ox, &oname), grid, block, 0, &params(&x_in, &yb));
            cmp(t, &label, &ya.down(), &yb.down());
        }
        _ => {
            let params = |x: &Buf, y: &Buf, s: &Buf| {
                let mut p = Params::default();
                p.put(x.0).put(idp).put(y.0).put(s.0).put(c.ne00).put(s01).put(s02).put(s03).put(ne0).put(c.ne1).put(c.ne2).put(0i32);
                p
            };
            let grid = (c.ne1 as u32, (c.ne2 * c.ne3) as u32, 1);
            let oname = format!("nvfp4_quantize_mmq_{}", TY[it]);
            launch_ox(func(m.ox, &oname), grid, (128, 1, 1), 0, &params(&x_in, &yb, &sb));
            let (ob, os) = (yb.down(), sb.down());
            for aligned in [true, false] {
                let rname = format!("_Z18quantize_mmq_nvfp4ILb0ELb{}EEvPKfPKiPvPfllllllli", aligned as u32);
                let (ya2, sa2) = (Buf::up(&fill), Buf::up(&sfill));
                launch(func(m.quant, &rname), grid, (128, 1, 1), 0, &params(&x_f32, &ya2, &sa2));
                cmp(t, &format!("{label} aligned={aligned} vy"), &ya2.down(), &ob);
                cmp(t, &format!("{label} aligned={aligned} scale"), &sa2.down(), &os);
            }
            drop((ya, sa));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// MMQ

pub const MMQ_J_NC: [i32; 11] = [8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128];
pub const MMQ_J_FB: [i32; 5] = [8, 16, 32, 64, 128];

/// mmq_get_nbytes_shared (MMA layout, I = 128, SRAM stride 76, 256 threads).
pub fn mmq_smem(j: i32) -> u32 {
    (j * 4 + 128 * 76 * 4 + (j * 144 + 1023) / 1024 * 1024) as u32
}

/// mul_mat_q_switch_J: the smallest J minimising ceil(ncols / J).
pub fn mmq_pick_j(ncols: i64, fallback: bool) -> i32 {
    let set: &[i32] = if fallback { &MMQ_J_FB } else { &MMQ_J_NC };
    let (mut best, mut nt_best) = (0, i64::MAX);
    for &j in set {
        if nt_best <= 1 {
            break;
        }
        let nt = (ncols + j as i64 - 1) / j as i64;
        if nt < nt_best {
            best = j;
            nt_best = nt;
        }
    }
    best
}

/// Weight blocks: realistic scales, `1/rate` special scales, random nibbles.
fn x_blocks(r: &mut Rng, f: usize, n: usize, rate: u64) -> Vec<u8> {
    let mut v = Vec::with_capacity(n * BYTES[f]);
    for _ in 0..n {
        match f {
            0 => {
                let d = if r.next() % rate == 0 {
                    [0u16, 0x8000, 0x7c00, 0xfc00, 0x7e00, 0x0001, 0x03ff, 0x7bff][(r.next() % 8) as usize]
                } else {
                    0x1800 + (r.next() % 0x1000) as u16 | (((r.next() & 1) as u16) << 15)
                };
                v.extend_from_slice(&d.to_le_bytes());
                v.extend_from_slice(&r.bytes(16));
            }
            1 => {
                let e = if r.next() % rate == 0 { [0u8, 1, 254, 255, 127][(r.next() % 5) as usize] } else { 112 + (r.next() % 24) as u8 };
                v.push(e);
                v.extend_from_slice(&r.bytes(16));
            }
            3 => {
                // block_iq4_xs: d, scales_h, scales_l[4], qs[128]
                let d = if r.next() % rate == 0 {
                    [0u16, 0x8000, 0x7c00, 0xfc00, 0x7e00, 0x0001, 0x03ff, 0x7bff][(r.next() % 8) as usize]
                } else {
                    0x1000 + (r.next() % 0x1000) as u16 | (((r.next() & 1) as u16) << 15)
                };
                v.extend_from_slice(&d.to_le_bytes());
                v.extend_from_slice(&r.bytes(134));
            }
            _ => {
                for _ in 0..4 {
                    let s = if r.next() % rate == 0 { [0u8, 1, 0x7e, 0x7f, 0xff, 0x80][(r.next() % 6) as usize] } else { 0x28 + (r.next() % 0x30) as u8 };
                    v.push(s);
                }
                v.extend_from_slice(&r.bytes(32));
            }
        }
    }
    v
}

/// Activation blocks as the quantizers lay them out (block_q8_1_mmq D4 / block_fp4_mmq).
fn y_blocks(r: &mut Rng, f: usize, n: usize, rate: u64) -> Vec<u8> {
    let mut v = Vec::with_capacity(n * 144);
    for _ in 0..n {
        match f {
            0 | 3 => {
                for _ in 0..4 {
                    let d = acts(r, 1, rate, true)[0].abs() / 127.0;
                    v.extend_from_slice(&d.to_le_bytes());
                }
            }
            1 => {
                for _ in 0..4 {
                    let mut e = [0u8; 4];
                    for b in e.iter_mut().take(2) {
                        *b = if r.next() % rate == 0 { [0u8, 255, 1, 254][(r.next() % 4) as usize] } else { 116 + (r.next() % 16) as u8 };
                    }
                    v.extend_from_slice(&e);
                }
            }
            _ => {
                for _ in 0..16 {
                    v.push(if r.next() % rate == 0 { [0u8, 0x7f, 0xff, 0x7e][(r.next() % 4) as usize] } else { 0x30 + (r.next() % 0x28) as u8 });
                }
            }
        }
        v.extend_from_slice(&r.bytes(128));
    }
    v
}

pub struct MmqCase {
    pub nrows: i32,
    pub k: i32,
    pub ncols: i32,
    pub pad_row: i32,
    pub grid: Option<u32>, // None: launch_mul_mat_q's choice for nsm
    pub nch: i32,          // nchannels_y (x shared: channel_ratio = nch)
    pub yscale: bool,      // NVFP4: pass y_scale
}

fn mmq_case(m: &M, t: &mut Tally, r: &mut Rng, f: usize, c: &MmqCase, nsm: u32, rate: u64) {
    mmq_case_named(m, t, r, f, c, nsm, rate, FMT[f]);
}

/// [`mmq_case`] against the oxide entries `<oxfmt>_mmq_j*` (the gate mutants use their own prefix);
/// returns the number of differing bytes.
#[allow(clippy::too_many_arguments)]
fn mmq_case_named(m: &M, t: &mut Tally, r: &mut Rng, f: usize, c: &MmqCase, nsm: u32, rate: u64, oxfmt: &str) -> usize {
    let fallback = c.nrows % 128 != 0;
    let j = mmq_pick_j(c.ncols as i64, fallback);
    let bpn = c.k / QK[f];
    let stride_row_x = bpn + c.pad_row;
    let x = Buf::up(&x_blocks(r, f, (stride_row_x * c.nrows + 64) as usize, rate));
    let kp = (c.k as i64 + 511) / 512 * 512;
    let nyb_ch = (kp / NEB[f]) as i32 * c.ncols; // activation blocks per channel
    let y = Buf::up(&y_blocks(r, f, (nyb_ch * c.nch + 128) as usize, rate));
    let ysv: Vec<f32> = (0..c.ncols * c.nch + 128)
        .map(|_| if r.next() % rate == 0 { [0.0f32, 1e-40, f32::INFINITY, 1e30][(r.next() % 4) as usize] } else { acts(r, 1, 1 << 40, false)[0].abs() / 2688.0 })
        .collect();
    let ys = Buf::up(&kdiff::as_bytes(&ysv));
    let ysp = if f == 2 && c.yscale { ys.0 } else { 0u64 };
    let stride_col_dst = c.nrows + (c.pad_row % 2);
    let stride_ch_dst = stride_col_dst * c.ncols;
    let nout = (stride_ch_dst * c.nch) as usize;
    let fill = r.bytes(nout * 4);
    let (da, db) = (Buf::up(&fill), Buf::up(&fill));

    let nty = (c.nrows + 127) / 128;
    let ntx = (c.ncols + j - 1) / j;
    let ntiles = (ntx * nty * c.nch) as u32;
    let nblocks = c.grid.unwrap_or_else(|| {
        let nwaves = ntiles.div_ceil(nsm);
        let eff = 100 * ntiles / (nsm * nwaves);
        if eff >= 90 { ntiles } else { nsm }
    });
    let fixup_needed = ntiles % nblocks != 0;
    let tmp_a = Buf::zeros((nblocks * j as u32 * 128) as usize * 4);
    let tmp_b = Buf::zeros((nblocks * j as u32 * 128) as usize * 4);
    let fd_bpn = fastdiv_values(bpn as u64);
    let fd_ntx = fastdiv_values(ntx as u64);
    let fd_ch = fastdiv_values(c.nch as u64);
    let one = fastdiv_values(1);
    let smem = mmq_smem(j);
    let stride_ch_y = nyb_ch * 36;
    let mmq_params = |dst: &Buf, tmp: &Buf| {
        let mut p = Params::default();
        p.put(x.0).put(y.0).put(0u64).put(0u64).put(dst.0).put(tmp.0).put(ysp);
        p.put(fd_bpn[0]).put(fd_bpn[1]).put(fd_bpn[2]);
        p.put(c.nrows).put(c.ncols).put(stride_row_x).put(c.ncols).put(stride_col_dst);
        // channel_ratio (= nch: every channel reads the same x), nchannels_y, strides x / y / dst
        p.put(fd_ch[0]).put(fd_ch[1]).put(fd_ch[2]).put(fd_ch[0]).put(fd_ch[1]).put(fd_ch[2]).put(0i32).put(stride_ch_y).put(stride_ch_dst);
        p.put(one[0]).put(one[1]).put(one[2]).put(one[0]).put(one[1]).put(one[2]).put(0i32).put(0i32).put(0i32);
        p.put(fd_ntx[0]).put(fd_ntx[1]).put(fd_ntx[2]);
        p
    };
    let fix_params = |dst: &Buf, tmp: &Buf| {
        let mut p = Params::default();
        p.put(0u64).put(0u64).put(dst.0).put(tmp.0);
        p.put(fd_bpn[0]).put(fd_bpn[1]).put(fd_bpn[2]);
        p.put(c.nrows).put(c.ncols).put(stride_col_dst);
        p.put(fd_ch[0]).put(fd_ch[1]).put(fd_ch[2]).put(stride_ch_dst).put(one[0]).put(one[1]).put(one[2]).put(0i32);
        p.put(fd_ntx[0]).put(fd_ntx[1]).put(fd_ntx[2]);
        p
    };
    let fb = fallback as u32;
    let ty = GGML_TYPE[f];
    let rname = format!("_Z9mul_mat_qIL9ggml_type{ty}ELi{j}ELb{fb}EEvPKcPKiS4_S4_PfS5_PKf5uint3iiiiiS8_S8_iiiS8_S8_iiiS8_");
    let rfix = format!("_Z24mul_mat_q_stream_k_fixupIL9ggml_type{ty}ELi{j}ELb{fb}EEvPKiS2_PfS3_5uint3iiiS4_iS4_iS4_");
    launch(func(m.mmq[f], &rname), (nblocks, 1, 1), (32, 8, 1), smem, &mmq_params(&da, &tmp_a));
    launch_ox(func(m.ox, &format!("{oxfmt}_mmq_j{j}_f{fb}")), (nblocks, 1, 1), (32, 8, 1), smem, &mmq_params(&db, &tmp_b));
    if fixup_needed {
        launch(func(m.mmq[f], &rfix), (nblocks, 4, 1), (32, 4, 1), 0, &fix_params(&da, &tmp_a));
        launch_ox(func(m.ox, &format!("{oxfmt}_mmq_fixup_j{j}_f{fb}")), (nblocks, 4, 1), (32, 4, 1), 0, &fix_params(&db, &tmp_b));
    }
    let (ra, rb) = (da.down(), db.down());
    let label = format!(
        "{oxfmt} mmq J={j} fb={fallback} nrows={} k={} ncols={} nch={} grid={nblocks} fixup={fixup_needed} yscale={} rate={rate}",
        c.nrows, c.k, c.ncols, c.nch, ysp != 0
    );
    if ra != rb {
        // rule 31: is the reference itself deterministic here?
        let dc = Buf::up(&fill);
        let tmp_c = Buf::zeros((nblocks * j as u32 * 128) as usize * 4);
        launch(func(m.mmq[f], &rname), (nblocks, 1, 1), (32, 8, 1), smem, &mmq_params(&dc, &tmp_c));
        if fixup_needed {
            launch(func(m.mmq[f], &rfix), (nblocks, 4, 1), (32, 4, 1), 0, &fix_params(&dc, &tmp_c));
        }
        if dc.down() != ra {
            println!("  note: reference disagrees with itself for {label}");
        }
    }
    cmp(t, &label, &ra, &rb)
}

/// MoE mode of `mul_mat_q` (MUL_MAT_ID; redcell2): one channel per expert, the experts' columns
/// `bounds[e]..bounds[e + 1]` of a compact activation buffer, dst column j written to row `ids[j]`
/// (column stride nrows), with llama.cpp's ggml_cuda_mul_mat_q arguments for that path. `counts` tokens
/// per expert, ncols_max = max + extra; K may leave a partial last tile iteration (REDCELL's 704).
#[allow(clippy::too_many_arguments)]
fn moe_mmq_case(m: &M, t: &mut Tally, r: &mut Rng, f: usize, nrows: i32, k: i32, counts: &[i32], extra: i32, grid: Option<u32>, nsm: u32, rate: u64) {
    let ne = counts.len() as i32;
    let total: i32 = counts.iter().sum();
    let ncols_max = counts.iter().copied().max().unwrap_or(0) + extra;
    let fallback = nrows % 128 != 0;
    let j = mmq_pick_j(ncols_max as i64, fallback);
    let bpn = k / QK[f];
    let x = Buf::up(&x_blocks(r, f, (bpn * nrows * ne + 64) as usize, rate));
    let kp = (k as i64 + 511) / 512 * 512;
    let nyb = (kp / NEB[f]) as i32 * total.max(1);
    let y = Buf::up(&y_blocks(r, f, (nyb + 256) as usize, rate));
    let mut bounds = vec![0i32];
    for &c in counts {
        bounds.push(bounds.last().unwrap() + c);
    }
    let mut ids: Vec<i32> = (0..total).collect();
    for i in (1..ids.len()).rev() {
        let jj = (r.next() % (i as u64 + 1)) as usize;
        ids.swap(i, jj);
    }
    // the kernels read ids up to J entries past the last expert's columns
    for _ in 0..256 {
        ids.push((r.next() % total.max(1) as u64) as i32);
    }
    let idb = Buf::up(&kdiff::as_bytes(&ids));
    let eb = Buf::up(&kdiff::as_bytes(&bounds));
    let fill = r.bytes((nrows * total.max(1)) as usize * 4);
    let (da, db) = (Buf::up(&fill), Buf::up(&fill));
    let nty = (nrows + 127) / 128;
    let ntx = (ncols_max + j - 1) / j;
    let ntiles = (ntx * nty * ne) as u32;
    let nblocks = grid.unwrap_or_else(|| {
        let nwaves = ntiles.div_ceil(nsm);
        if 100 * ntiles / (nsm * nwaves) >= 90 { ntiles } else { nsm }
    });
    let fixup_needed = ntiles % nblocks != 0;
    let tmp_a = Buf::zeros((nblocks * j as u32 * 128) as usize * 4);
    let tmp_b = Buf::zeros((nblocks * j as u32 * 128) as usize * 4);
    let (fd_bpn, fd_ntx, fd_ne, one) = (fastdiv_values(bpn as u64), fastdiv_values(ntx as u64), fastdiv_values(ne as u64), fastdiv_values(1));
    let smem = mmq_smem(j);
    let mmq_params = |dst: &Buf, tmp: &Buf| {
        let mut p = Params::default();
        p.put(x.0).put(y.0).put(idb.0).put(eb.0).put(dst.0).put(tmp.0).put(0u64);
        p.put(fd_bpn[0]).put(fd_bpn[1]).put(fd_bpn[2]);
        p.put(nrows).put(total).put(bpn).put(total).put(nrows);
        p.put(one[0]).put(one[1]).put(one[2]).put(fd_ne[0]).put(fd_ne[1]).put(fd_ne[2]).put(nrows * bpn).put(0i32).put(0i32);
        p.put(one[0]).put(one[1]).put(one[2]).put(one[0]).put(one[1]).put(one[2]).put(0i32).put(0i32).put(0i32);
        p.put(fd_ntx[0]).put(fd_ntx[1]).put(fd_ntx[2]);
        p
    };
    let fix_params = |dst: &Buf, tmp: &Buf| {
        let mut p = Params::default();
        p.put(idb.0).put(eb.0).put(dst.0).put(tmp.0);
        p.put(fd_bpn[0]).put(fd_bpn[1]).put(fd_bpn[2]);
        p.put(nrows).put(total).put(nrows);
        p.put(fd_ne[0]).put(fd_ne[1]).put(fd_ne[2]).put(0i32).put(one[0]).put(one[1]).put(one[2]).put(0i32);
        p.put(fd_ntx[0]).put(fd_ntx[1]).put(fd_ntx[2]);
        p
    };
    let fb = fallback as u32;
    let ty = GGML_TYPE[f];
    let rname = format!("_Z9mul_mat_qIL9ggml_type{ty}ELi{j}ELb{fb}EEvPKcPKiS4_S4_PfS5_PKf5uint3iiiiiS8_S8_iiiS8_S8_iiiS8_");
    let rfix = format!("_Z24mul_mat_q_stream_k_fixupIL9ggml_type{ty}ELi{j}ELb{fb}EEvPKiS2_PfS3_5uint3iiiS4_iS4_iS4_");
    launch(func(m.mmq[f], &rname), (nblocks, 1, 1), (32, 8, 1), smem, &mmq_params(&da, &tmp_a));
    launch_ox(func(m.ox, &format!("{}_mmq_j{j}_f{fb}", FMT[f])), (nblocks, 1, 1), (32, 8, 1), smem, &mmq_params(&db, &tmp_b));
    if fixup_needed {
        launch(func(m.mmq[f], &rfix), (nblocks, 4, 1), (32, 4, 1), 0, &fix_params(&da, &tmp_a));
        launch_ox(func(m.ox, &format!("{}_mmq_fixup_j{j}_f{fb}", FMT[f])), (nblocks, 4, 1), (32, 4, 1), 0, &fix_params(&db, &tmp_b));
    }
    let label = format!(
        "{} moe-mmq J={j} fb={fallback} nrows={nrows} k={k} experts={ne} total={total} ncols_max={ncols_max} grid={nblocks} fixup={fixup_needed} rate={rate}",
        FMT[f]
    );
    cmp(t, &label, &da.down(), &db.down());
}

pub fn nsm() -> u32 {
    let mut v = 0i32;
    unsafe {
        let mut dev = 0;
        ck!(cu::cuDeviceGet(&mut dev, 0));
        ck!(cu::cuDeviceGetAttribute(&mut v, cu::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    }
    v as u32
}

fn ctx() {
    let _ctx = Box::leak(Box::new(kdiff::cuda_core::CudaContext::new(0).expect("cuda context")));
    unsafe {
        let mut c: cu::CUcontext = std::ptr::null_mut();
        ck!(cu::cuCtxGetCurrent(&mut c));
        if c.is_null() {
            let mut dev = 0;
            ck!(cu::cuDeviceGet(&mut dev, 0));
            ck!(cu::cuDevicePrimaryCtxRetain(&mut c, dev));
            ck!(cu::cuCtxSetCurrent(c));
        }
    }
}

fn modules() -> M {
    let dir = env!("CARGO_MANIFEST_DIR");
    M {
        quant: load(&format!("{dir}/ref/quantize.cubin")),
        mmq: [
            load(&format!("{dir}/ref/mmq_iq4_nl.cubin")),
            load(&format!("{dir}/ref/mmq_mxfp4.cubin")),
            load(&format!("{dir}/ref/mmq_nvfp4.cubin")),
            load(&format!("{dir}/ref/mmq_iq4_xs.cubin")),
        ],
        ox: load(&format!("{dir}/fmt_mmq.ptx")),
    }
}

pub fn run_gate(quick: bool, only: Option<usize>) -> bool {
    ctx();
    let m = modules();
    let mut t = Tally::default();
    let mut r = Rng(0xF4_17_A5_7A_11_0C_E5_EDu64);
    let fmts: Vec<usize> = match only {
        Some(f) => vec![f],
        None => vec![0, 1, 2, 3],
    };
    // quantizers (IQ4_XS uses IQ4_NL's)
    for &f in fmts.iter().filter(|&&f| f != 3) {
        let min_k = if f == 2 { 64 } else { 128 };
        for it in 0..3 {
            let shapes: &[(i64, i64, i64, i64, i64, bool)] = &[
                (min_k, 1, 1, 1, 0, false),
                (512, 9, 1, 1, 1, false),
                (4096, 17, 1, 1, 0, false),
                (5120, 33, 1, 1, 2, false),
                (17408, 9, 1, 1, 0, false),
                (1024, 5, 3, 2, 1, false),
                (1000 / 64 * 64, 40, 1, 1, 3, false),
                (768, 12, 1, 1, 0, true),
                (2560, 3, 2, 1, 0, true),
            ];
            for &(ne00, ne1, ne2, ne3, pad, ids) in shapes {
                let c = QCase { ne00, ne1, ne2, ne3, pad, ids };
                for &rate in &[1000u64, 7] {
                    quant_case(&m, &mut t, &mut r, f, it, &c, rate);
                }
            }
            // targeted: all-zero rows, one huge value per row, tiny rows (denormal amax)
            for &rate in &[1u64 << 40, 2, 1] {
                quant_case(&m, &mut t, &mut r, f, it, &QCase { ne00: 1536, ne1: 6, ne2: 1, ne3: 1, pad: 0, ids: false }, rate);
            }
        }
    }
    // MMQ
    let nsm = nsm();
    for &f in &fmts {
        let k_unit = if f == 0 || f == 3 { 256 } else { 512 };
        let mut cases = vec![];
        for &j in &MMQ_J_NC {
            cases.push(MmqCase { nrows: 256, k: 2 * k_unit, ncols: j, pad_row: 0, grid: None, nch: 1, yscale: true });
            cases.push(MmqCase { nrows: 384, k: 3 * k_unit, ncols: j - 3, pad_row: 1, grid: Some(7), nch: 1, yscale: true });
        }
        for &j in &MMQ_J_FB {
            cases.push(MmqCase { nrows: 200, k: 2 * k_unit, ncols: j, pad_row: 0, grid: None, nch: 1, yscale: true });
            cases.push(MmqCase { nrows: 48, k: 3 * k_unit, ncols: j - 5, pad_row: 3, grid: Some(5), nch: 1, yscale: false });
        }
        // stream-k decompositions: odd grids, more blocks than tiles, tiles split mid-row
        for &g in &[1u32, 2, 3, 13, 29, 64, 97] {
            cases.push(MmqCase { nrows: 512, k: 4 * k_unit, ncols: 100, pad_row: 0, grid: Some(g), nch: 1, yscale: true });
            cases.push(MmqCase { nrows: 130, k: 5 * k_unit, ncols: 77, pad_row: 2, grid: Some(g), nch: 1, yscale: g % 2 == 0 });
        }
        // K not a multiple of the tile iteration (the reference reads the next blocks: identical bytes)
        // (IQ4_XS: one block is one tile iteration, K is always a whole number of them)
        if f != 3 {
            cases.push(MmqCase { nrows: 256, k: 3 * k_unit / 2, ncols: 33, pad_row: 1, grid: None, nch: 1, yscale: true });
        }
        // channels (x broadcast over 2 / 3 y channels)
        cases.push(MmqCase { nrows: 256, k: 2 * k_unit, ncols: 40, pad_row: 0, grid: None, nch: 2, yscale: true });
        cases.push(MmqCase { nrows: 136, k: 2 * k_unit, ncols: 70, pad_row: 1, grid: Some(11), nch: 3, yscale: true });
        // model shapes (Qwen3-14B: 5120 -> 1024 / 5120 / 17408, 17408 -> 5120) at the launcher's grid
        if !quick {
            for &(nrows, k) in &[(5120, 5120), (1024, 5120), (17408, 5120), (5120, 17408), (6144, 5120)] {
                for &ncols in &[9, 17, 64, 200, 512] {
                    cases.push(MmqCase { nrows, k, ncols, pad_row: 0, grid: None, nch: 1, yscale: true });
                }
            }
            // IQ4_XS: the rest of Qwen3.5-27B (OrcaSAQ-2): 5120 -> 48 / 12288, 6144 / 10240 -> 5120
            if f == 3 {
                for &(nrows, k) in &[(48, 5120), (12288, 5120), (5120, 6144), (5120, 10240)] {
                    for &ncols in &[9, 64, 512] {
                        cases.push(MmqCase { nrows, k, ncols, pad_row: 0, grid: None, nch: 1, yscale: true });
                    }
                }
            }
        }
        // random shapes (fallback and not, launcher or forced grids, 1-3 channels)
        for n in 0..60 {
            let nrows = if n % 3 == 0 { 128 * (1 + (r.next() % 8) as i32) } else { 1 + (r.next() % 1100) as i32 };
            let k = QK[f] * (1 + (r.next() % (6 * k_unit as u64 / QK[f] as u64)) as i32);
            let ncols = 9 + (r.next() % 300) as i32;
            let grid = if n % 2 == 0 { None } else { Some(1 + (r.next() % 150) as u32) };
            let nch = 1 + (n % 3 == 1) as i32 + (n % 5 == 2) as i32;
            cases.push(MmqCase { nrows, k, ncols, pad_row: (n % 4) as i32, grid, nch, yscale: n % 7 != 3 });
        }
        for c in &cases {
            mmq_case(&m, &mut t, &mut r, f, c, nsm, 256);
        }
        for c in cases.iter().filter(|c| c.nrows <= 512) {
            mmq_case(&m, &mut t, &mut r, f, c, nsm, 5);
        }
    }
    // IQ4_XS loader mutants: each must differ from the reference
    if fmts.contains(&3) {
        let mut detected = 0;
        for k in 1..=2 {
            let mut tm = Tally::default();
            let mut caught = 0;
            for &(nrows, kk, ncols) in &[(256i32, 512i32, 32i32), (384, 1280, 30)] {
                let c = MmqCase { nrows, k: kk, ncols, pad_row: 0, grid: None, nch: 1, yscale: true };
                caught += mmq_case_named(&m, &mut tm, &mut r, 3, &c, nsm, 256, &format!("iq4_xs_mut{k}"));
            }
            println!("  mutation iq4_xs mmq mut{k}: {}", if caught > 0 { "DETECTED (fails vs reference, as it must)" } else { "NOT DETECTED" });
            detected += (caught > 0) as u32;
        }
        let d = kdiff::Diff { bytes: 4, differing: if detected == 2 { 0 } else { 1 }, first: None };
        t.record(&format!("iq4_xs mutation sensitivity ({detected}/2 mutants detected)"), &d);
    }
    // MoE mode (redcell2), IQ4_NL: REDCELL's expert down shape (K = 704) and generic shapes, launcher and forced grids
    if fmts.contains(&0) {
        let mut sets: Vec<Vec<i32>> = vec![vec![5, 0, 17, 1], vec![130, 2, 64], vec![0, 0, 3], vec![200, 150, 7, 90], vec![16, 16, 16, 16]];
        for tokens in [72i32, 547] {
            // 128 experts, top-8 routing of `tokens` tokens, skewed
            let mut c = vec![0i32; 128];
            for _ in 0..tokens * 8 {
                let e = ((r.next() % 128) * (r.next() % 128) / 127) as usize;
                c[e] += 1;
            }
            sets.push(c);
        }
        for (si, counts) in sets.iter().enumerate() {
            for &(nrows, k) in &[(256i32, 704i32), (300, 704), (128, 512), (256, 2816)] {
                for &grid in &[None, Some(7u32)] {
                    if counts.len() == 128 && (nrows != 256 || grid.is_some()) && k != 704 {
                        continue;
                    }
                    moe_mmq_case(&m, &mut t, &mut r, 0, nrows, k, counts, (si % 3) as i32 * 9, grid, nsm, if si % 2 == 0 { 256 } else { 5 });
                }
            }
        }
    }
    println!("oxide kernel launches: {}", unsafe { OX_LAUNCHES });
    let mut cats: std::collections::BTreeMap<String, usize> = Default::default();
    for f in &t.failures {
        let k = f.split(" J=").next().unwrap().split(" ne00=").next().unwrap().to_string();
        *cats.entry(k).or_default() += 1;
    }
    for (k, n) in &cats {
        println!("  failing category: {k}: {n}");
    }
    t.finish("fmt_mmq")
}

/// `--bench`: reference vs oxide mul_mat_q (+ fixup) on Qwen3-14B prefill shapes.
pub fn bench() {
    ctx();
    let m = modules();
    let mut r = Rng(11);
    let nsm = nsm();
    for f in 0..3 {
        for &(nrows, k, ncols) in &[(17408i32, 5120i32, 512i32), (5120, 17408, 512), (5120, 5120, 512), (1024, 5120, 512), (5120, 5120, 64), (17408, 5120, 200)] {
            let j = mmq_pick_j(ncols as i64, nrows % 128 != 0);
            let bpn = k / QK[f];
            let x = Buf::up(&x_blocks(&mut r, f, (bpn * nrows + 64) as usize, 1 << 40));
            let ny = ((k + 511) / 512 * 512) as i64 / NEB[f] * ncols as i64 + 128;
            let y = Buf::up(&y_blocks(&mut r, f, ny as usize, 1 << 40));
            let ys = Buf::up(&kdiff::as_bytes(&vec![0.01f32; ncols as usize + 128]));
            let ysp = if f == 2 { ys.0 } else { 0 };
            let dst = Buf::zeros((nrows * ncols) as usize * 4);
            let nty = (nrows + 127) / 128;
            let ntx = (ncols + j - 1) / j;
            let ntiles = (ntx * nty) as u32;
            let nwaves = ntiles.div_ceil(nsm);
            let nblocks = if 100 * ntiles / (nsm * nwaves) >= 90 { ntiles } else { nsm };
            let fixup = ntiles % nblocks != 0;
            let tmp = Buf::zeros((nblocks * j as u32 * 128) as usize * 4);
            let (fd_bpn, fd_ntx, one) = (fastdiv_values(bpn as u64), fastdiv_values(ntx as u64), fastdiv_values(1));
            let mut p = Params::default();
            p.put(x.0).put(y.0).put(0u64).put(0u64).put(dst.0).put(tmp.0).put(ysp);
            p.put(fd_bpn[0]).put(fd_bpn[1]).put(fd_bpn[2]).put(nrows).put(ncols).put(bpn).put(ncols).put(nrows);
            p.put(one[0]).put(one[1]).put(one[2]).put(one[0]).put(one[1]).put(one[2]).put(0i32).put(0i32).put(0i32);
            p.put(one[0]).put(one[1]).put(one[2]).put(one[0]).put(one[1]).put(one[2]).put(0i32).put(0i32).put(0i32);
            p.put(fd_ntx[0]).put(fd_ntx[1]).put(fd_ntx[2]);
            let mut pf = Params::default();
            pf.put(0u64).put(0u64).put(dst.0).put(tmp.0).put(fd_bpn[0]).put(fd_bpn[1]).put(fd_bpn[2]).put(nrows).put(ncols).put(nrows);
            pf.put(one[0]).put(one[1]).put(one[2]).put(0i32).put(one[0]).put(one[1]).put(one[2]).put(0i32).put(fd_ntx[0]).put(fd_ntx[1]).put(fd_ntx[2]);
            let fb = (nrows % 128 != 0) as u32;
            let smem = mmq_smem(j);
            let ty = GGML_TYPE[f];
            let sides = [
                (func(m.mmq[f], &format!("_Z9mul_mat_qIL9ggml_type{ty}ELi{j}ELb{fb}EEvPKcPKiS4_S4_PfS5_PKf5uint3iiiiiS8_S8_iiiS8_S8_iiiS8_")),
                 func(m.mmq[f], &format!("_Z24mul_mat_q_stream_k_fixupIL9ggml_type{ty}ELi{j}ELb{fb}EEvPKiS2_PfS3_5uint3iiiS4_iS4_iS4_"))),
                (func(m.ox, &format!("{}_mmq_j{j}_f{fb}", FMT[f])), func(m.ox, &format!("{}_mmq_fixup_j{j}_f{fb}", FMT[f]))),
            ];
            let mut us = [0f32; 2];
            for (s, &(fnc, ff)) in sides.iter().enumerate() {
                unsafe {
                    ck!(cu::cuFuncSetAttribute(fnc, cu::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32));
                    let run = |pp: &Params, fnc: cu::CUfunction, g: (u32, u32, u32), b: (u32, u32, u32), sm: u32| {
                        let mut pb = pp.0.clone();
                        let mut size = pb.len();
                        let mut extra: [*mut c_void; 5] = [1 as *mut c_void, pb.as_mut_ptr() as *mut c_void, 2 as *mut c_void, &mut size as *mut usize as *mut c_void, std::ptr::null_mut()];
                        ck!(cu::cuLaunchKernel(fnc, g.0, g.1, g.2, b.0, b.1, b.2, sm, std::ptr::null_mut(), std::ptr::null_mut(), extra.as_mut_ptr()));
                    };
                    let (mut e0, mut e1): (cu::CUevent, cu::CUevent) = (std::ptr::null_mut(), std::ptr::null_mut());
                    ck!(cu::cuEventCreate(&mut e0, 0));
                    ck!(cu::cuEventCreate(&mut e1, 0));
                    for it in 0..60 {
                        if it == 10 {
                            ck!(cu::cuEventRecord(e0, std::ptr::null_mut()));
                        }
                        run(&p, fnc, (nblocks, 1, 1), (32, 8, 1), smem);
                        if fixup {
                            run(&pf, ff, (nblocks, 4, 1), (32, 4, 1), 0);
                        }
                    }
                    ck!(cu::cuEventRecord(e1, std::ptr::null_mut()));
                    ck!(cu::cuEventSynchronize(e1));
                    let mut ms = 0f32;
                    ck!(cu::cuEventElapsedTime_v2(&mut ms, e0, e1));
                    us[s] = ms * 1000.0 / 50.0;
                }
            }
            let tops = 2.0 * nrows as f64 * k as f64 * ncols as f64 / 1e12;
            println!("{} mmq N={nrows} K={k} M={ncols} J={j} grid={nblocks} fixup={fixup}: ref {:.1} us ({:.0} TOPS), oxide {:.1} us ({:.0} TOPS) -> {:.2}x",
                FMT[f], us[0], tops / (us[0] as f64 * 1e-6), us[1], tops / (us[1] as f64 * 1e-6), us[1] / us[0]);
        }
    }
}

pub fn run() -> bool {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--bench") {
        bench();
        return true;
    }
    let quick = args.iter().any(|a| a == "--quick");
    let only = args.iter().find_map(|a| a.strip_prefix("--fmt=")).map(|s| FMT.iter().position(|&n| n == s).expect("--fmt=iq4_nl|mxfp4|nvfp4|iq4_xs"));
    run_gate(quick, only)
}
