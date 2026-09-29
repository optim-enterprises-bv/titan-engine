//! Differential gate for the fast-path entries: activation quantizers (q8_1 / mmq D4 layout,
//! f32 / f16 / bf16 inputs), mmvq with f16 / bf16 output, and the Q1_0 MMQ (stream-k tile
//! matmul + fixup), each against the llama.cpp acecd56 reference cubins (ref/quantize.cubin,
//! ref/mmvq.cubin, ref/convert.cubin, ref/mmq.cubin; SASS-identical to libggml-cuda.so).
//!
//! Compositions are compared end to end on the device:
//! - `q1_0_quantize_*_{f16,bf16}(x)` vs `quantize_*(convert_unary<half|bf16, float>(x))`
//! - `q1_0_mmvq_*_{f16,bf16}` vs `convert_unary<float, half|bf16>(mul_mat_vec_q<Q1_0,..>)`
//! - `q1_0_mmq_j*` + `q1_0_mmq_fixup_j*` vs `mul_mat_q<Q1_0,J,fb>` + `mul_mat_q_stream_k_fixup`,
//!   both launched with the host logic of mmq.cuh (J choice, stream-k grid, fixup grid, shared
//!   memory), over several grid sizes.
use crate::gate::{Params, fastdiv_values};
use cuda_core::sys as cu;
use kdiff::{Rng, Tally};
use std::ffi::c_void;

macro_rules! ck {
    ($e:expr) => {{
        let r = $e;
        assert_eq!(r, cu::cudaError_enum_CUDA_SUCCESS, "{}", stringify!($e));
    }};
}

pub struct M {
    pub quant: cu::CUmodule,
    pub mmvq: cu::CUmodule,
    pub convert: cu::CUmodule,
    pub mmq: cu::CUmodule,
    pub ox: cu::CUmodule,
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

fn cmp(t: &mut Tally, label: &str, a: &[u8], b: &[u8]) {
    let mut d = kdiff::Diff { bytes: a.len(), differing: 0, first: None };
    for i in 0..a.len() {
        if a[i] != b[i] {
            d.differing += 1;
            if d.first.is_none() {
                d.first = Some((0, i, a[i], b[i]));
            }
        }
    }
    t.record(label, &d);
}

/// f32 activations: mostly N(0, s)-like magnitudes, `1/rate` specials.
fn acts(r: &mut Rng, n: usize, rate: u64) -> Vec<f32> {
    const SP: [f32; 10] = [0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 1e-40, -1e-42, 3e38, -3e38, 0.49999997];
    (0..n)
        .map(|_| {
            if r.next() % rate == 0 {
                return SP[(r.next() % 10) as usize];
            }
            let u = (r.next() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
            let e = (r.next() % 12) as i32 - 6;
            (u * 2f64.powi(e)) as f32
        })
        .collect()
}

/// Half-precision patterns (bits): values like `acts`, rounded, plus raw specials.
fn halfs(r: &mut Rng, n: usize, bf16: bool, rate: u64) -> Vec<u16> {
    acts(r, n, rate)
        .into_iter()
        .map(|v| {
            if r.next() % (4 * rate) == 0 {
                return r.next() as u16; // raw bit pattern: NaN payloads, denormals, ...
            }
            if bf16 { (v.to_bits() >> 16) as u16 } else { f32_to_f16_trunc(v) }
        })
        .collect()
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

const CONV_TO_F32: [&str; 2] = [
    "_Z13convert_unaryI6__halffEvPKvPT0_lll5uint3lll",
    "_Z13convert_unaryI13__nv_bfloat16fEvPKvPT0_lll5uint3lll",
];
const CONV_FROM_F32: [&str; 2] = ["_Z13convert_unaryIf6__halfEvPKvPT0_lll5uint3lll", "_Z13convert_unaryIf13__nv_bfloat16EvPKvPT0_lll5uint3lll"];

/// `convert_unary_cont_cuda` (k contiguous elements) on the reference module.
fn convert(m: &M, name: &str, src: &Buf, dst: &Buf, k: i64) {
    let one = fastdiv_values(1);
    let mut p = Params::default();
    p.put(src.0).put(dst.0).put(k).put(1i64).put(1i64).put(one[0]).put(one[1]).put(one[2]).put(k).put(k).put(k);
    launch(func(m.convert, name), (((k + 255) / 256) as u32, 1, 1), (256, 1, 1), 0, &p);
}

// ---------------------------------------------------------------------------------------------
// quantizers

fn quant_q8_1_case(m: &M, t: &mut Tally, r: &mut Rng, it: usize, ne00: i64, ne1: u32, pad: i64, rate: u64) {
    // it: 0 f32, 1 f16, 2 bf16
    let ne0 = (ne00 + pad + 511) / 512 * 512;
    let s01 = ne00 + (pad % 3);
    let nx = (s01 * ne1 as i64) as usize;
    let (x_in, x_f32) = match it {
        0 => {
            let v = Buf::up(&kdiff::as_bytes(&acts(r, nx, rate)));
            let c = Buf::up(&v.down());
            (v, c)
        }
        _ => {
            let h = Buf::up(&kdiff::as_bytes(&halfs(r, nx, it == 2, rate)));
            let f = Buf::zeros(nx * 4);
            convert(m, CONV_TO_F32[it - 1], &h, &f, nx as i64);
            (h, f)
        }
    };
    let out_bytes = (ne0 / 32) as usize * 36 * ne1 as usize;
    let fill = r.bytes(out_bytes);
    let ya = Buf::up(&fill);
    let yb = Buf::up(&fill);
    let one = fastdiv_values(1);
    let params = |x: &Buf, y: &Buf| {
        let mut p = Params::default();
        p.put(x.0).put(y.0).put(ne00).put(s01).put(s01 * ne1 as i64).put(s01 * ne1 as i64).put(ne0).put(ne1).put(one[0]).put(one[1]).put(one[2]);
        p
    };
    let grid = (((ne0 + 255) / 256) as u32, ne1, 1);
    launch(func(m.quant, "_Z13quantize_q8_1PKfPvlllllj5uint3"), grid, (256, 1, 1), 0, &params(&x_f32, &ya));
    let name = ["q1_0_quantize_q8_1_f32", "q1_0_quantize_q8_1_f16", "q1_0_quantize_q8_1_bf16"][it];
    launch(func(m.ox, name), grid, (256, 1, 1), 0, &params(&x_in, &yb));
    cmp(t, &format!("{name} ne00={ne00} ne1={ne1} rate={rate}"), &ya.down(), &yb.down());
}

fn quant_mmq_case(m: &M, t: &mut Tally, r: &mut Rng, it: usize, ne00: i64, ne1: i32, ids: bool, rate: u64) {
    let ne0 = (ne00 + 511) / 512 * 512;
    let s01 = ne00 + 4 * (ne1 as i64 % 3);
    let nrows_src = if ids { ne1 + 3 } else { ne1 };
    let nx = (s01 * nrows_src as i64) as usize + 8;
    let (x_in, x_f32) = match it {
        0 => {
            let v = Buf::up(&kdiff::as_bytes(&acts(r, nx, rate)));
            let c = Buf::up(&v.down());
            (v, c)
        }
        _ => {
            let h = Buf::up(&kdiff::as_bytes(&halfs(r, nx, it == 2, rate)));
            let f = Buf::zeros(nx * 4);
            convert(m, CONV_TO_F32[it - 1], &h, &f, nx as i64);
            (h, f)
        }
    };
    let idsv: Vec<i32> = (0..ne1).map(|_| (r.next() % nrows_src as u64) as i32).collect();
    let idb = Buf::up(&kdiff::as_bytes(&idsv));
    let out_bytes = (ne0 / 128) as usize * 144 * ne1 as usize;
    let fill = r.bytes(out_bytes);
    let ya = Buf::up(&fill);
    let yb = Buf::up(&fill);
    let params = |x: &Buf, y: &Buf| {
        let mut p = Params::default();
        p.put(x.0).put(if ids { idb.0 } else { 0u64 }).put(y.0).put(ne00).put(s01).put(s01 * nrows_src as i64).put(s01 * nrows_src as i64);
        p.put(ne0).put(ne1).put(1i32).put(0i32);
        p
    };
    let grid = (ne1 as u32, ((ne0 + 511) / 512) as u32, 1);
    launch(func(m.quant, "_Z17quantize_mmq_q8_1IL18mmq_q8_1_ds_layout0ELb0EEvPKfPKiPvllllliii"), grid, (128, 1, 1), 0, &params(&x_f32, &ya));
    let name = ["q1_0_quantize_mmq_d4_f32", "q1_0_quantize_mmq_d4_f16", "q1_0_quantize_mmq_d4_bf16"][it];
    launch(func(m.ox, name), grid, (128, 1, 1), 0, &params(&x_in, &yb));
    cmp(t, &format!("{name} ne00={ne00} ne1={ne1} ids={ids} rate={rate}"), &ya.down(), &yb.down());
}

// ---------------------------------------------------------------------------------------------
// mmvq with half outputs: one channel / sample (the fast path's shapes) plus a channel case.

fn q1_0_blocks(r: &mut Rng, n: usize, rate: u64) -> Vec<u8> {
    crate::gate::q1_0_blocks_pub(r, n, rate)
}
fn q8_1_blocks(r: &mut Rng, n: usize, rate: u64) -> Vec<u8> {
    crate::gate::q8_1_blocks_pub(r, n, rate)
}

fn mmvq_half_case(m: &M, t: &mut Tally, r: &mut Rng, nc: usize, small_k: bool, h: usize, bpr: u32, nrows: u32, rate: u64) {
    // h: 0 f16, 1 bf16
    let nwarps: u32 = if nc <= 4 { 4 } else { 2 };
    let rpb: u32 = if nc == 1 { if small_k { nwarps } else { 1 } } else { 2 };
    let stride_row_x = bpr;
    let stride_col_y = (bpr * 128).div_ceil(512) * 512 / 32;
    let x = Buf::up(&q1_0_blocks(r, (bpr * (nrows + rpb)) as usize + 4, rate));
    let y = Buf::up(&q8_1_blocks(r, (stride_col_y * nc as u32) as usize + 8, rate));
    let nout = (nrows * nc as u32) as usize;
    let dref = Buf::zeros(nout * 4);
    let href = Buf::up(&r.bytes(nout * 2));
    let hox = Buf::up(&href.down());
    let one = fastdiv_values(1);
    let params = |dst: &Buf| {
        let mut p = Params::default();
        p.put(x.0).put(y.0).put(0u64);
        for _ in 0..5 {
            p.put(0u64);
        }
        p.put(0u32).put(0f32).put(dst.0).put(bpr * 128).put(0u32).put(0u32).put(0u32);
        p.put(stride_row_x).put(stride_col_y).put(nrows);
        p.put(one[0]).put(one[1]).put(one[2]).put(0u32).put(0u32).put(0u32);
        p.put(one[0]).put(one[1]).put(one[2]).put(0u32).put(0u32).put(0u32).put(0u32);
        p
    };
    let grid = (nrows.div_ceil(rpb), 1, 1);
    let rname = format!(
        "_Z13mul_mat_vec_qIL9ggml_type41ELi{nc}ELb0ELb{}ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj",
        small_k as u32
    );
    launch(func(m.mmvq, &rname), grid, (32, nwarps, 1), 0, &params(&dref));
    convert(m, CONV_FROM_F32[h], &dref, &href, nout as i64);
    let oname = format!("q1_0_mmvq_{nc}{}{}", if small_k { "_small_k" } else { "" }, ["_f16", "_bf16"][h]);
    launch(func(m.ox, &oname), grid, (32, nwarps, 1), 0, &params(&hox));
    cmp(t, &format!("{oname} bpr={bpr} nrows={nrows} rate={rate}"), &href.down(), &hox.down());
}

// ---------------------------------------------------------------------------------------------
// MMQ

pub const MMQ_J_NC: [i32; 11] = [8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128];
pub const MMQ_J_FB: [i32; 5] = [8, 16, 32, 64, 128];

/// mmq_get_nbytes_shared for Q1_0 (MMA layout, I = 128, SRAM stride 76, 256 threads).
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

struct MmqCase {
    nrows: i32,
    k: i32,
    ncols: i32,
    pad_row: i32,
    grid: Option<u32>, // None: launch_mul_mat_q's choice for nsm
}

#[allow(clippy::too_many_arguments)]
fn mmq_case(m: &M, t: &mut Tally, r: &mut Rng, c: &MmqCase, nsm: u32, rate: u64) {
    let fallback = c.nrows % 128 != 0;
    let j = mmq_pick_j(c.ncols as i64, fallback);
    let bpn = c.k / 128;
    let stride_row_x = bpn + c.pad_row;
    // x: rows past nrows are never read (fallback clamps); 2 spare blocks per row for odd bpn.
    let x = Buf::up(&q1_0_blocks(r, (stride_row_x * c.nrows + 4) as usize, rate));
    // y: block_q8_1_mmq [k/128][ncols] + J_max spare blocks (the tile_y copy reads past the end)
    let kp = (c.k + 511) / 512 * 512;
    let ny = (kp / 128 * c.ncols + 128) as usize;
    let mut yb = Vec::with_capacity(ny * 144);
    for _ in 0..ny {
        for _ in 0..4 {
            let v = acts(r, 1, rate)[0].abs() / 127.0;
            yb.extend_from_slice(&v.to_le_bytes());
        }
        yb.extend_from_slice(&r.bytes(128));
    }
    let y = Buf::up(&yb);
    let stride_col_dst = c.nrows + (c.pad_row % 2);
    let nout = (stride_col_dst * c.ncols) as usize;
    let fill = r.bytes(nout * 4);
    let da = Buf::up(&fill);
    let db = Buf::up(&fill);

    let nty = (c.nrows + 127) / 128;
    let ntx = (c.ncols + j - 1) / j;
    let ntiles = (ntx * nty) as u32;
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
    let one = fastdiv_values(1);
    let smem = mmq_smem(j);
    let mmq_params = |dst: &Buf, tmp: &Buf| {
        let mut p = Params::default();
        p.put(x.0).put(y.0).put(0u64).put(0u64).put(dst.0).put(tmp.0).put(0u64);
        p.put(fd_bpn[0]).put(fd_bpn[1]).put(fd_bpn[2]);
        p.put(c.nrows).put(c.ncols).put(stride_row_x).put(c.ncols).put(stride_col_dst);
        p.put(one[0]).put(one[1]).put(one[2]).put(one[0]).put(one[1]).put(one[2]).put(0i32).put(0i32).put(0i32);
        p.put(one[0]).put(one[1]).put(one[2]).put(one[0]).put(one[1]).put(one[2]).put(0i32).put(0i32).put(0i32);
        p.put(fd_ntx[0]).put(fd_ntx[1]).put(fd_ntx[2]);
        p
    };
    let fix_params = |dst: &Buf, tmp: &Buf| {
        let mut p = Params::default();
        p.put(0u64).put(0u64).put(dst.0).put(tmp.0);
        p.put(fd_bpn[0]).put(fd_bpn[1]).put(fd_bpn[2]);
        p.put(c.nrows).put(c.ncols).put(stride_col_dst);
        p.put(one[0]).put(one[1]).put(one[2]).put(0i32).put(one[0]).put(one[1]).put(one[2]).put(0i32);
        p.put(fd_ntx[0]).put(fd_ntx[1]).put(fd_ntx[2]);
        p
    };
    let fb = fallback as u32;
    let rname = format!("_Z9mul_mat_qIL9ggml_type41ELi{j}ELb{fb}EEvPKcPKiS4_S4_PfS5_PKf5uint3iiiiiS8_S8_iiiS8_S8_iiiS8_");
    let rfix = format!("_Z24mul_mat_q_stream_k_fixupIL9ggml_type41ELi{j}ELb{fb}EEvPKiS2_PfS3_5uint3iiiS4_iS4_iS4_");
    launch(func(m.mmq, &rname), (nblocks, 1, 1), (32, 8, 1), smem, &mmq_params(&da, &tmp_a));
    launch(func(m.ox, &format!("q1_0_mmq_j{j}_f{fb}")), (nblocks, 1, 1), (32, 8, 1), smem, &mmq_params(&db, &tmp_b));
    if fixup_needed {
        launch(func(m.mmq, &rfix), (nblocks, 4, 1), (32, 4, 1), 0, &fix_params(&da, &tmp_a));
        launch(func(m.ox, &format!("q1_0_mmq_fixup_j{j}_f{fb}")), (nblocks, 4, 1), (32, 4, 1), 0, &fix_params(&db, &tmp_b));
    }
    cmp(t, &format!("mmq J={j} fb={fallback} nrows={} k={} ncols={} grid={nblocks} fixup={fixup_needed} rate={rate}", c.nrows, c.k, c.ncols), &da.down(), &db.down());
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

pub fn run(m: &M, t: &mut Tally, quick: bool) {
    let mut r = Rng(0x51_1F_A5_7A_11_0C_E5_EDu64);
    // quantizers
    for it in 0..3 {
        for &(ne00, ne1) in &[(32i64, 1u32), (128, 3), (4096, 1), (4096, 8), (5120, 2), (12288, 5), (17408, 1), (1000, 4)] {
            for &rate in &[1000u64, 7] {
                quant_q8_1_case(m, t, &mut r, it, ne00, ne1, (ne00 % 7) * 32, rate);
            }
        }
        for &(ne00, ne1) in &[(128i64, 1i32), (512, 9), (4096, 17), (5120, 33), (12288, 12), (17408, 9), (1000, 40)] {
            for &rate in &[1000u64, 7] {
                quant_mmq_case(m, t, &mut r, it, ne00, ne1, false, rate);
            }
            quant_mmq_case(m, t, &mut r, it, ne00, ne1, true, 50);
        }
    }
    // mmvq with half outputs
    for h in 0..2 {
        for nc in 1..=8usize {
            for &small_k in if nc == 1 { &[false, true][..] } else { &[false][..] } {
                for &(bpr, nrows) in &[(1u32, 5u32), (3, 37), (32, 1024), (32, 4096), (96, 4096), (40, 17408), (136, 5120), (40, 48)] {
                    mmvq_half_case(m, t, &mut r, nc, small_k, h, bpr, nrows, if nrows < 100 { 3 } else { 512 });
                }
            }
        }
    }
    // MMQ
    let nsm = nsm();
    let mut cases = vec![];
    // every J of both instance sets (ncols chosen so mmq_pick_j lands on it)
    for &j in &MMQ_J_NC {
        cases.push(MmqCase { nrows: 256, k: 512, ncols: j, pad_row: 0, grid: None });
        cases.push(MmqCase { nrows: 384, k: 1024, ncols: j - 3, pad_row: 1, grid: Some(7) });
    }
    for &j in &MMQ_J_FB {
        cases.push(MmqCase { nrows: 200, k: 512, ncols: j, pad_row: 0, grid: None });
        cases.push(MmqCase { nrows: 48, k: 768, ncols: j - 5, pad_row: 3, grid: Some(5) });
    }
    // stream-k decompositions: odd grids, more blocks than tiles, tiles split mid-row
    for &g in &[1u32, 2, 3, 13, 29, 64, 97] {
        cases.push(MmqCase { nrows: 512, k: 2048, ncols: 100, pad_row: 0, grid: Some(g) });
        cases.push(MmqCase { nrows: 130, k: 1280, ncols: 77, pad_row: 2, grid: Some(g) });
    }
    // model shapes (Bonsai-8B / 27B prefill), at the launcher's grid choice
    if !quick {
        for &(nrows, k) in &[(4096, 4096), (1024, 4096), (12288, 4096), (4096, 12288), (6144, 5120), (10240, 5120), (17408, 5120), (5120, 17408), (48, 5120), (5120, 6144)] {
            for &ncols in &[9, 17, 64, 200, 512] {
                cases.push(MmqCase { nrows, k, ncols, pad_row: 0, grid: None });
            }
        }
    }
    for c in &cases {
        mmq_case(m, t, &mut r, c, nsm, 256);
    }
    for c in cases.iter().filter(|c| c.nrows <= 512) {
        mmq_case(m, t, &mut r, c, nsm, 5);
    }
}

/// `--bench-mmq`: reference vs oxide mul_mat_q (+ fixup) on Bonsai prefill shapes.
pub fn bench_mmq(m: &M) {
    let mut r = Rng(11);
    let nsm = nsm();
    for &(nrows, k, ncols) in &[(12288i32, 4096i32, 512i32), (4096, 12288, 512), (4096, 4096, 512), (1024, 4096, 512), (17408, 5120, 512), (4096, 4096, 17), (12288, 4096, 64), (17408, 5120, 200)] {
        let j = mmq_pick_j(ncols as i64, nrows % 128 != 0);
        let bpn = k / 128;
        let x = Buf::up(&q1_0_blocks(&mut r, (bpn * nrows + 4) as usize, 512));
        let ny = ((k + 511) / 512 * 512 / 128 * ncols + 128) as usize;
        let mut yb = Vec::with_capacity(ny * 144);
        for _ in 0..ny {
            for _ in 0..4 {
                yb.extend_from_slice(&0.01f32.to_le_bytes());
            }
            yb.extend_from_slice(&r.bytes(128));
        }
        let y = Buf::up(&yb);
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
        p.put(x.0).put(y.0).put(0u64).put(0u64).put(dst.0).put(tmp.0).put(0u64);
        p.put(fd_bpn[0]).put(fd_bpn[1]).put(fd_bpn[2]).put(nrows).put(ncols).put(bpn).put(ncols).put(nrows);
        p.put(one[0]).put(one[1]).put(one[2]).put(one[0]).put(one[1]).put(one[2]).put(0i32).put(0i32).put(0i32);
        p.put(one[0]).put(one[1]).put(one[2]).put(one[0]).put(one[1]).put(one[2]).put(0i32).put(0i32).put(0i32);
        p.put(fd_ntx[0]).put(fd_ntx[1]).put(fd_ntx[2]);
        let mut pf = Params::default();
        pf.put(0u64).put(0u64).put(dst.0).put(tmp.0).put(fd_bpn[0]).put(fd_bpn[1]).put(fd_bpn[2]).put(nrows).put(ncols).put(nrows);
        pf.put(one[0]).put(one[1]).put(one[2]).put(0i32).put(one[0]).put(one[1]).put(one[2]).put(0i32).put(fd_ntx[0]).put(fd_ntx[1]).put(fd_ntx[2]);
        let fb = (nrows % 128 != 0) as u32;
        let smem = mmq_smem(j);
        let sides = [
            (func(m.mmq, &format!("_Z9mul_mat_qIL9ggml_type41ELi{j}ELb{fb}EEvPKcPKiS4_S4_PfS5_PKf5uint3iiiiiS8_S8_iiiS8_S8_iiiS8_")),
             func(m.mmq, &format!("_Z24mul_mat_q_stream_k_fixupIL9ggml_type41ELi{j}ELb{fb}EEvPKiS2_PfS3_5uint3iiiS4_iS4_iS4_"))),
            (func(m.ox, &format!("q1_0_mmq_j{j}_f{fb}")), func(m.ox, &format!("q1_0_mmq_fixup_j{j}_f{fb}"))),
        ];
        let mut us = [0f32; 2];
        for (s, &(f, ff)) in sides.iter().enumerate() {
            unsafe {
                ck!(cu::cuFuncSetAttribute(f, cu::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32));
                let run = |pp: &Params, f: cu::CUfunction, g: (u32, u32, u32), b: (u32, u32, u32), sm: u32| {
                    let mut pb = pp.0.clone();
                    let mut size = pb.len();
                    let mut extra: [*mut c_void; 5] = [1 as *mut c_void, pb.as_mut_ptr() as *mut c_void, 2 as *mut c_void, &mut size as *mut usize as *mut c_void, std::ptr::null_mut()];
                    ck!(cu::cuLaunchKernel(f, g.0, g.1, g.2, b.0, b.1, b.2, sm, std::ptr::null_mut(), std::ptr::null_mut(), extra.as_mut_ptr()));
                };
                let (mut e0, mut e1): (cu::CUevent, cu::CUevent) = (std::ptr::null_mut(), std::ptr::null_mut());
                ck!(cu::cuEventCreate(&mut e0, 0));
                ck!(cu::cuEventCreate(&mut e1, 0));
                for it in 0..60 {
                    if it == 10 {
                        ck!(cu::cuEventRecord(e0, std::ptr::null_mut()));
                    }
                    run(&p, f, (nblocks, 1, 1), (32, 8, 1), smem);
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
        println!("mmq N={nrows} K={k} M={ncols} J={j} grid={nblocks} fixup={fixup}: ref {:.1} us ({:.0} TOPS), oxide {:.1} us ({:.0} TOPS)",
            us[0], tops / (us[0] as f64 * 1e-6), us[1], tops / (us[1] as f64 * 1e-6));
    }
}
