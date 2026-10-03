//! Differential gate: every ptq1_0 entry against the reference instances rebuilt from sudoingX/llama.cpp bonsai2
//! v1.1 (ref/{quantize,fwht,mmvq,mmq}.cubin, see ref/nvcc_ref.py). Both sides get the same packed parameter
//! buffer (CU_LAUNCH_PARAM_BUFFER_POINTER), private copies of the same input bytes and a pre-filled output
//! buffer; every output byte is compared. Compositions (bf16 inputs, bf16 outputs, the fused FWHT + quantize, the
//! post-signs FWHT) are compared against the reference kernels plus exact host steps or the reference chain.
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
    assert_eq!(a.len(), b.len(), "{label}");
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

/// f32 activations: mostly N(0, s)-like magnitudes over a range of scales, `1/rate` specials.
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

/// bf16 bit patterns (values like `acts`, truncated, plus raw patterns) and their exact f32 widening.
fn bf16s(r: &mut Rng, n: usize, rate: u64) -> (Vec<u16>, Vec<f32>) {
    let h: Vec<u16> = acts(r, n, rate)
        .into_iter()
        .map(|v| if r.next() % (4 * rate) == 0 { r.next() as u16 } else { (v.to_bits() >> 16) as u16 })
        .collect();
    let f = h.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
    (h, f)
}

/// Random f16 scale bits: mostly normal magnitudes, `1/rate` specials (0, -0, inf, nan, denormal, max).
fn f16_bits(r: &mut Rng, rate: u64) -> u16 {
    const SP: [u16; 8] = [0x0000, 0x8000, 0x7C00, 0xFC00, 0x7E00, 0x0001, 0x83FF, 0x7BFF];
    if r.next() % rate == 0 {
        return SP[(r.next() % 8) as usize];
    }
    (0x1800 + (r.next() % 0x3000) as u16) | (((r.next() & 1) as u16) << 15)
}

/// PTQ1_0 blocks: any byte is a valid 5-trit (qs) / 4-trit (qh) code.
fn ptq1_0_blocks(r: &mut Rng, n: usize, rate: u64) -> Vec<u8> {
    let mut v = Vec::with_capacity(n * 28);
    for _ in 0..n {
        for _ in 0..26 {
            v.push(r.next() as u8);
        }
        v.extend_from_slice(&f16_bits(r, rate).to_le_bytes());
    }
    v
}

/// Host `cvt.rn.bf16.f32` (NaN -> 0x7FFF).
fn bf16_rne(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7FFF;
    }
    let b = x.to_bits();
    let lsb = (b >> 16) & 1;
    ((b + 0x7FFF + lsb) >> 16) as u16
}

pub struct M {
    pub quant: cu::CUmodule,
    pub fwht: cu::CUmodule,
    pub mmvq: cu::CUmodule,
    pub mmq: cu::CUmodule,
    pub ox: cu::CUmodule,
}

const REF_QPT: &str = "_Z13quantize_q8_1ILb1EEvPKfPvlllllj5uint3";
const REF_QMMQ: &str = "_Z17quantize_mmq_q8_1IL18mmq_q8_1_ds_layout0ELb0ELb0ELb0EEvPKfPKiPvllllliiiS2_S2_S2_";
fn ref_fwht(signs: bool) -> String {
    format!("_Z15fwht_cuda_blockILi1024ELi256EfLb{}EEvPKT1_PflfPKfi", signs as u32)
}
fn ref_mmvq(nc: usize, glu: bool) -> String {
    let g = glu as u32;
    format!("_Z21mul_mat_vec_ptq1_0_ptILi{nc}ELi4ELb{g}ELb{g}EEvPKvS1_31ggml_cuda_mm_fusion_args_devicePfiiiiii5uint3")
}

// ---------------------------------------------------------------------------------------------
// PT quantizer

fn qpt_params(x: &Buf, y: &Buf, ne00: i64, s01: i64, ne0: i64, ne1: u32) -> Params {
    let one = fastdiv_values(1);
    let mut p = Params::default();
    p.put(x.0).put(y.0).put(ne00).put(s01).put(s01 * ne1 as i64).put(s01 * ne1 as i64).put(ne0).put(ne1).put(one[0]).put(one[1]).put(one[2]);
    p
}

fn quant_pt_case(m: &M, t: &mut Tally, r: &mut Rng, bf16: bool, ne00: i64, ne1: u32, rate: u64) {
    let ne0 = (ne00 + 511) / 512 * 512;
    let s01 = ne00 + 32 * (ne1 as i64 % 3);
    let nx = (s01 * ne1 as i64) as usize;
    let (x_in, x_f32) = if bf16 {
        let (h, f) = bf16s(r, nx, rate);
        (Buf::up(&kdiff::as_bytes(&h)), Buf::up(&kdiff::as_bytes(&f)))
    } else {
        let v = kdiff::as_bytes(&acts(r, nx, rate));
        (Buf::up(&v), Buf::up(&v))
    };
    let fill = r.bytes((ne0 / 32) as usize * 36 * ne1 as usize);
    let (ya, yb) = (Buf::up(&fill), Buf::up(&fill));
    let grid = (((ne0 + 255) / 256) as u32, ne1, 1);
    launch(func(m.quant, REF_QPT), grid, (256, 1, 1), 0, &qpt_params(&x_f32, &ya, ne00, s01, ne0, ne1));
    let name = if bf16 { "ptq1_0_quantize_pt_bf16" } else { "ptq1_0_quantize_pt_f32" };
    launch(func(m.ox, name), grid, (256, 1, 1), 0, &qpt_params(&x_in, &yb, ne00, s01, ne0, ne1));
    cmp(t, &format!("{name} ne00={ne00} ne1={ne1} rate={rate}"), &ya.down(), &yb.down());
}

// ---------------------------------------------------------------------------------------------
// FWHT (+ fused quantize)

fn fwht_params(src: u64, dst: u64, rows: i64, signs: u64, n_blk: i32) -> Params {
    let mut p = Params::default();
    p.put(src).put(dst).put(rows).put(1.0f32 / 32.0).put(signs).put(n_blk);
    p
}

fn sign_vec(r: &mut Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| if r.next() & 1 == 0 { 1.0 } else { -1.0 }).collect()
}

/// mode 0/1/2 x f32/bf16 input, `tokens` rows of `n_blk` 1024-blocks.
fn fwht_case(m: &M, t: &mut Tally, r: &mut Rng, mode: u32, bf16: bool, tokens: usize, n_blk: usize, rate: u64) {
    let n = tokens * n_blk * 1024;
    let (x_in, x_f32) = if bf16 {
        let (h, f) = bf16s(r, n, rate);
        (Buf::up(&kdiff::as_bytes(&h)), Buf::up(&kdiff::as_bytes(&f)))
    } else {
        let v = kdiff::as_bytes(&acts(r, n, rate));
        (Buf::up(&v), Buf::up(&v))
    };
    let sv = sign_vec(r, n_blk * 1024);
    let signs = Buf::up(&kdiff::as_bytes(&sv));
    let fill = r.bytes(n * 4);
    let (da, db) = (Buf::up(&fill), Buf::up(&fill));
    let rows = (tokens * n_blk) as i64;
    let grid = (rows as u32, 1, 1);
    let (rs, rn) = if mode == 1 { (signs.0, n_blk as i32) } else { (0u64, 1) };
    launch(func(m.fwht, &ref_fwht(mode == 1)), grid, (256, 1, 1), 0, &fwht_params(x_f32.0, da.0, rows, rs, rn));
    let mut want = da.down();
    if mode == 2 {
        for (i, c) in want.chunks_exact_mut(4).enumerate() {
            let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            let o = if v.is_nan() { f32::from_bits(0x7FFF_FFFF) } else if sv[i % (n_blk * 1024)] < 0.0 { -v } else { v };
            c.copy_from_slice(&o.to_le_bytes());
        }
    }
    let name = format!("ptq1_0_fwht_{}_m{mode}", if bf16 { "bf16" } else { "f32" });
    let (os, on) = if mode == 0 { (0u64, 1) } else { (signs.0, n_blk as i32) };
    launch(func(m.ox, &name), grid, (256, 1, 1), 0, &fwht_params(x_in.0, db.0, rows, os, on));
    cmp(t, &format!("{name} tokens={tokens} n_blk={n_blk} rate={rate}"), &want, &db.down());
}

fn fwht_q8pt_case(m: &M, t: &mut Tally, r: &mut Rng, mode: u32, bf16: bool, tokens: usize, n_blk: usize, rate: u64) {
    let n = tokens * n_blk * 1024;
    let (x_in, x_f32) = if bf16 {
        let (h, f) = bf16s(r, n, rate);
        (Buf::up(&kdiff::as_bytes(&h)), Buf::up(&kdiff::as_bytes(&f)))
    } else {
        let v = kdiff::as_bytes(&acts(r, n, rate));
        (Buf::up(&v), Buf::up(&v))
    };
    let signs = Buf::up(&kdiff::as_bytes(&sign_vec(r, n_blk * 1024)));
    let rows = (tokens * n_blk) as i64;
    let (s, sn) = if mode == 1 { (signs.0, n_blk as i32) } else { (0u64, 1) };
    // reference chain: fwht -> f32 -> quantize_q8_1<pt>
    let tmp = Buf::zeros(n * 4);
    launch(func(m.fwht, &ref_fwht(mode == 1)), (rows as u32, 1, 1), (256, 1, 1), 0, &fwht_params(x_f32.0, tmp.0, rows, s, sn));
    let ne0 = (n_blk * 1024) as i64;
    let fill = r.bytes(n / 32 * 36);
    let (ya, yb) = (Buf::up(&fill), Buf::up(&fill));
    launch(func(m.quant, REF_QPT), (((ne0 + 255) / 256) as u32, tokens as u32, 1), (256, 1, 1), 0, &qpt_params(&tmp, &ya, ne0, ne0, ne0, tokens as u32));
    let name = format!("ptq1_0_fwht_q8pt_{}_m{mode}", if bf16 { "bf16" } else { "f32" });
    // the fused kernel's n_blk is also the row width (1024-blocks per token), signs or not
    launch(func(m.ox, &name), (rows as u32, 1, 1), (256, 1, 1), 0, &fwht_params(x_in.0, yb.0, rows, s, n_blk as i32));
    cmp(t, &format!("{name} tokens={tokens} n_blk={n_blk} rate={rate}"), &ya.down(), &yb.down());
}

// ---------------------------------------------------------------------------------------------
// PT mat-vec

/// ptq1_0_pt_rows_per_cta (mmvq-ptq1_0.cuh).
pub fn rows_per_cta(bpr: i32, nc: i32, rows_per_item: i32) -> i32 {
    let mut rmax = 4096 / (nc * bpr);
    rmax = if rmax < rows_per_item { rows_per_item } else if rmax > 16 { 16 } else { rmax };
    rmax -= rmax % rows_per_item;
    let (mut best, mut best_util) = (rows_per_item, 0.0f64);
    let mut rr = rows_per_item;
    while rr <= rmax {
        let items = (rr / rows_per_item) * bpr;
        let iters = (items + 127) / 128;
        let util = items as f64 / (iters * 128) as f64;
        if util > best_util + 1e-9 {
            best_util = util;
            best = rr;
        }
        if util > 0.999 {
            break;
        }
        rr += rows_per_item;
    }
    best
}

struct MvCase {
    nc: usize,
    k: i32,
    nrows: i32,
    pad_row: i32,
    pad_col_y: i32,
    pad_dst: i32,
}

/// kind: 0 f32 out, 1 bf16 out, 2 GLU SWIGLU, 3 GLU with glu_op 0 (plain product), 4 SWIGLU with x/gate bias.
fn mmvq_case(m: &M, t: &mut Tally, r: &mut Rng, c: &MvCase, kind: u32, rate: u64) {
    let bpr = c.k / 128;
    let stride_row_x = bpr + c.pad_row;
    let kp = (c.k + 511) / 512 * 512;
    let stride_col_y = kp / 32 + c.pad_col_y;
    let stride_col_dst = c.nrows + c.pad_dst;
    let nblocks_x = (stride_row_x * c.nrows + 8) as usize;
    let x = Buf::up(&ptq1_0_blocks(r, nblocks_x, rate));
    let g = Buf::up(&ptq1_0_blocks(r, nblocks_x, rate));
    // PT columns: random plane bytes, random f16 (d, s) pairs
    let ycol = (stride_col_y * 36) as usize;
    let mut yv = r.bytes(ycol * c.nc + 64);
    for j in 0..c.nc {
        let base = j * ycol + (kp as usize / 128) * 128;
        for s in 0..(kp as usize / 32) {
            let o = base + 4 * s;
            yv[o..o + 2].copy_from_slice(&f16_bits(r, rate).to_le_bytes());
        }
    }
    let y = Buf::up(&yv);
    let nout = (stride_col_dst as usize) * c.nc;
    let xb = Buf::up(&kdiff::as_bytes(&acts(r, nout, 1000)));
    let gb = Buf::up(&kdiff::as_bytes(&acts(r, nout, 1000)));
    let es = if kind == 1 { 2 } else { 4 };
    let fill = r.bytes(nout * es);
    let (da, db) = (Buf::up(&fill), Buf::up(&fill));
    let glu = kind >= 2;
    let rpc = rows_per_cta(bpr, c.nc as i32, 4);
    let fd = fastdiv_values(bpr as u64);
    let smem = (c.nc as i32 * rpc * bpr * 4 * if glu { 2 } else { 1 }) as u32;
    let grid = ((c.nrows + rpc - 1) / rpc) as u32;
    let params = |dst: u64| {
        let mut p = Params::default();
        p.put(x.0).put(y.0);
        let (xbias, gbias) = if kind == 4 { (xb.0, gb.0) } else { (0u64, 0u64) };
        p.put(xbias).put(if glu { g.0 } else { 0u64 }).put(gbias).put(0u64).put(0u64);
        p.put(if kind == 3 { 0u32 } else { 2u32 }).put(0u32);
        p.put(dst).put(c.k).put(c.nrows).put(stride_row_x).put(stride_col_y).put(stride_col_dst).put(rpc);
        p.put(fd[0]).put(fd[1]).put(fd[2]);
        assert_eq!(p.0.len(), 108);
        p
    };
    let want = if kind == 1 {
        let tmp = Buf::up(&r.bytes(nout * 4));
        launch(func(m.mmvq, &ref_mmvq(c.nc, false)), (grid, 1, 1), (128, 1, 1), smem, &params(tmp.0));
        let f = tmp.down();
        let mut h = da.down();
        for j in 0..c.nc {
            for row in 0..c.nrows as usize {
                let i = j * stride_col_dst as usize + row;
                let v = f32::from_le_bytes(f[4 * i..4 * i + 4].try_into().unwrap());
                h[2 * i..2 * i + 2].copy_from_slice(&bf16_rne(v).to_le_bytes());
            }
        }
        h
    } else {
        launch(func(m.mmvq, &ref_mmvq(c.nc, glu)), (grid, 1, 1), (128, 1, 1), smem, &params(da.0));
        da.down()
    };
    let name = match kind {
        0 => format!("ptq1_0_mmvq_pt_c{}", c.nc),
        1 => format!("ptq1_0_mmvq_pt_c{}_bf16", c.nc),
        _ => format!("ptq1_0_mmvq_pt_glu_c{}", c.nc),
    };
    launch(func(m.ox, &name), (grid, 1, 1), (128, 1, 1), smem, &params(db.0));
    cmp(t, &format!("{name} kind={kind} k={} nrows={} pads={}/{}/{} rpc={rpc} rate={rate}", c.k, c.nrows, c.pad_row, c.pad_col_y, c.pad_dst), &want, &db.down());
}

// ---------------------------------------------------------------------------------------------
// MMQ quantizer + MMQ

fn quant_mmq_case(m: &M, t: &mut Tally, r: &mut Rng, bf16: bool, ne00: i64, ne1: i32, rate: u64) {
    let ne0 = (ne00 + 511) / 512 * 512;
    let s01 = ne00 + 4 * (ne1 as i64 % 3);
    let nx = (s01 * ne1 as i64) as usize + 8;
    let (x_in, x_f32) = if bf16 {
        let (h, f) = bf16s(r, nx, rate);
        (Buf::up(&kdiff::as_bytes(&h)), Buf::up(&kdiff::as_bytes(&f)))
    } else {
        let v = kdiff::as_bytes(&acts(r, nx, rate));
        (Buf::up(&v), Buf::up(&v))
    };
    let fill = r.bytes((ne0 / 128) as usize * 144 * ne1 as usize);
    let (ya, yb) = (Buf::up(&fill), Buf::up(&fill));
    let params = |x: &Buf, y: &Buf| {
        let mut p = Params::default();
        p.put(x.0).put(0u64).put(y.0).put(ne00).put(s01).put(s01 * ne1 as i64).put(s01 * ne1 as i64);
        p.put(ne0).put(ne1).put(1i32).put(0i32).put(0u64).put(0u64).put(0u64);
        p
    };
    let grid = (ne1 as u32, ((ne0 + 511) / 512) as u32, 1);
    launch(func(m.quant, REF_QMMQ), grid, (128, 1, 1), 0, &params(&x_f32, &ya));
    let name = if bf16 { "ptq1_0_quantize_mmq_d4_bf16" } else { "ptq1_0_quantize_mmq_d4_f32" };
    launch(func(m.ox, name), grid, (128, 1, 1), 0, &params(&x_in, &yb));
    cmp(t, &format!("{name} ne00={ne00} ne1={ne1} rate={rate}"), &ya.down(), &yb.down());
}

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

struct MmqCase {
    nrows: i32,
    k: i32,
    ncols: i32,
    pad_row: i32,
    grid: Option<u32>,
}

fn mmq_case(m: &M, t: &mut Tally, r: &mut Rng, c: &MmqCase, nsm: u32, rate: u64) {
    let fallback = c.nrows % 128 != 0;
    let j = mmq_pick_j(c.ncols as i64, fallback);
    let bpn = c.k / 128;
    let stride_row_x = bpn + c.pad_row;
    let x = Buf::up(&ptq1_0_blocks(r, (stride_row_x * c.nrows + 8) as usize, rate));
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
    let (da, db) = (Buf::up(&fill), Buf::up(&fill));
    let nty = (c.nrows + 127) / 128;
    let ntx = (c.ncols + j - 1) / j;
    let ntiles = (ntx * nty) as u32;
    let nblocks = c.grid.unwrap_or_else(|| {
        let nwaves = ntiles.div_ceil(nsm);
        if 100 * ntiles / (nsm * nwaves) >= 90 { ntiles } else { nsm }
    });
    let fixup_needed = ntiles % nblocks != 0;
    let tmp_a = Buf::zeros((nblocks * j as u32 * 128) as usize * 4);
    let tmp_b = Buf::zeros((nblocks * j as u32 * 128) as usize * 4);
    let (fd_bpn, fd_ntx, one) = (fastdiv_values(bpn as u64), fastdiv_values(ntx as u64), fastdiv_values(1));
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
    let rname = format!("_Z9mul_mat_qIL9ggml_type143ELi{j}ELb{fb}EEvPKcPKiS4_S4_PfS5_PKf5uint3iiiiiS8_S8_iiiS8_S8_iiiS8_");
    let rfix = format!("_Z24mul_mat_q_stream_k_fixupIL9ggml_type143ELi{j}ELb{fb}EEvPKiS2_PfS3_5uint3iiiS4_iS4_iS4_");
    launch(func(m.mmq, &rname), (nblocks, 1, 1), (32, 8, 1), smem, &mmq_params(&da, &tmp_a));
    launch(func(m.ox, &format!("ptq1_0_mmq_j{j}_f{fb}")), (nblocks, 1, 1), (32, 8, 1), smem, &mmq_params(&db, &tmp_b));
    if fixup_needed {
        launch(func(m.mmq, &rfix), (nblocks, 4, 1), (32, 4, 1), 0, &fix_params(&da, &tmp_a));
        launch(func(m.ox, &format!("ptq1_0_mmq_fixup_j{j}_f{fb}")), (nblocks, 4, 1), (32, 4, 1), 0, &fix_params(&db, &tmp_b));
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

fn modules() -> M {
    let dir = env!("CARGO_MANIFEST_DIR");
    // keep the primary context retained for the whole run
    std::mem::forget(kdiff::cuda_core::CudaContext::new(0).expect("cuda context"));
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
    M {
        quant: load(&format!("{dir}/ref/quantize.cubin")),
        fwht: load(&format!("{dir}/ref/fwht.cubin")),
        mmvq: load(&format!("{dir}/ref/mmvq.cubin")),
        mmq: load(&format!("{dir}/ref/mmq.cubin")),
        ox: load(&format!("{dir}/ptq1_0.ptx")),
    }
}

pub fn run_gate(only: Option<&str>, quick: bool) -> bool {
    let m = modules();
    let mut t = Tally::default();
    let mut r = Rng(0x7071_3130_B0B5_A1u64);
    let on = |s: &str| only.is_none_or(|o| o == s);

    if on("quant") {
        for bf16 in [false, true] {
            for &(ne00, ne1) in &[(128i64, 1u32), (512, 3), (1000, 2), (5120, 1), (5120, 4), (6144, 3), (17408, 2), (4096, 5)] {
                for &rate in &[1000u64, 7] {
                    quant_pt_case(&m, &mut t, &mut r, bf16, ne00, ne1, rate);
                }
            }
            for &(ne00, ne1) in &[(128i64, 1i32), (512, 9), (5120, 33), (6144, 12), (17408, 9), (1000, 40), (5120, 512)] {
                for &rate in &[1000u64, 7] {
                    quant_mmq_case(&m, &mut t, &mut r, bf16, ne00, ne1, rate);
                }
            }
        }
    }
    if on("fwht") {
        for mode in 0..3u32 {
            for bf16 in [false, true] {
                for &(tokens, n_blk) in &[(1usize, 1usize), (1, 5), (1, 6), (1, 17), (3, 5), (2, 17), (7, 6), (64, 5)] {
                    for &rate in &[100_000u64, 11] {
                        fwht_case(&m, &mut t, &mut r, mode, bf16, tokens, n_blk, rate);
                    }
                }
            }
        }
        for mode in 0..2u32 {
            for bf16 in [false, true] {
                for &(tokens, n_blk) in &[(1usize, 5usize), (1, 6), (1, 17), (2, 5), (3, 6), (4, 17), (1, 1)] {
                    for &rate in &[100_000u64, 11] {
                        fwht_q8pt_case(&m, &mut t, &mut r, mode, bf16, tokens, n_blk, rate);
                    }
                }
            }
        }
    }
    if on("mmvq") {
        let mut cases = vec![];
        for nc in 1..=4usize {
            for &k in &[128i32, 256, 384, 640, 1152, 4224] {
                for &nrows in &[1i32, 3, 5, 17, 37, 100] {
                    let pad = (k / 128 + nrows) % 3;
                    cases.push(MvCase { nc, k, nrows, pad_row: pad, pad_col_y: pad * 4, pad_dst: pad % 2 });
                }
            }
            // Bonsai 2 27B projections: K 5120 -> qkv 10240, gate 6144, q+gate 12288, k/v 1024, ffn 17408, lm_head
            // (a 20000-row slice of 248320); K 6144 -> 5120 (ssm_out, attn_output); K 17408 -> 5120 (ffn_down)
            if !quick {
                for &(k, nrows) in &[(5120i32, 10240i32), (5120, 6144), (5120, 12288), (5120, 1024), (5120, 17408), (5120, 20000), (6144, 5120), (17408, 5120), (10240, 5120)] {
                    cases.push(MvCase { nc, k, nrows, pad_row: 0, pad_col_y: 0, pad_dst: 0 });
                }
            }
        }
        for c in &cases {
            for kind in 0..5u32 {
                if kind == 3 && c.nrows > 1000 {
                    continue;
                }
                mmvq_case(&m, &mut t, &mut r, c, kind, 512);
            }
        }
        for c in cases.iter().filter(|c| c.nrows <= 100) {
            for kind in [0u32, 2] {
                mmvq_case(&m, &mut t, &mut r, c, kind, 3);
            }
        }
    }
    if on("mmq") {
        let nsm = nsm();
        let mut cases = vec![];
        for &j in &MMQ_J_NC {
            cases.push(MmqCase { nrows: 256, k: 512, ncols: j, pad_row: 0, grid: None });
            cases.push(MmqCase { nrows: 384, k: 1024, ncols: j - 3, pad_row: 1, grid: Some(7) });
        }
        for &j in &MMQ_J_FB {
            cases.push(MmqCase { nrows: 200, k: 512, ncols: j, pad_row: 0, grid: None });
            cases.push(MmqCase { nrows: 48, k: 768, ncols: j - 5, pad_row: 3, grid: Some(5) });
        }
        for &g in &[1u32, 2, 3, 13, 29, 64, 97] {
            cases.push(MmqCase { nrows: 512, k: 2048, ncols: 100, pad_row: 0, grid: Some(g) });
            cases.push(MmqCase { nrows: 130, k: 1280, ncols: 77, pad_row: 2, grid: Some(g) });
        }
        if !quick {
            for &(nrows, k) in &[(10240, 5120), (6144, 5120), (12288, 5120), (1024, 5120), (17408, 5120), (5120, 6144), (5120, 17408), (48, 5120)] {
                for &ncols in &[5, 9, 64, 200, 512] {
                    cases.push(MmqCase { nrows, k, ncols, pad_row: 0, grid: None });
                }
            }
        }
        for c in &cases {
            mmq_case(&m, &mut t, &mut r, c, nsm, 256);
        }
        for c in cases.iter().filter(|c| c.nrows <= 512) {
            mmq_case(&m, &mut t, &mut r, c, nsm, 5);
        }
    }
    t.finish("ptq1_0 (quantize pt x2, fwht x6 + fused x4, mmvq pt x12, quantize mmq x2, mmq x16 + fixup x16) vs sudoingX bonsai2 v1.1 cubins")
}

/// `--bench`: reference vs oxide PT mat-vec (and the fused FWHT + quantize) on the model's decode shapes.
fn bench() {
    let m = modules();
    let mut r = Rng(7);
    for &(k, nrows) in &[(5120i32, 10240i32), (5120, 17408), (17408, 5120), (6144, 5120), (5120, 248320)] {
        for nc in [1usize, 2, 3] {
            let bpr = k / 128;
            let x = Buf::up(&ptq1_0_blocks(&mut r, (bpr * nrows + 8) as usize, 1 << 30));
            let kp = (k + 511) / 512 * 512;
            let y = Buf::up(&r.bytes((kp as usize / 32) * 36 * nc + 64));
            let d = Buf::zeros(nrows as usize * nc * 4);
            let rpc = rows_per_cta(bpr, nc as i32, 4);
            let fd = fastdiv_values(bpr as u64);
            let smem = (nc as i32 * rpc * bpr * 4) as u32;
            let grid = ((nrows + rpc - 1) / rpc) as u32;
            let mut p = Params::default();
            p.put(x.0).put(y.0).put(0u64).put(0u64).put(0u64).put(0u64).put(0u64).put(2u32).put(0u32);
            p.put(d.0).put(k).put(nrows).put(bpr).put(kp / 32).put(nrows).put(rpc).put(fd[0]).put(fd[1]).put(fd[2]);
            let mut us = [0f32; 2];
            for (s, f) in [func(m.mmvq, &ref_mmvq(nc, false)), func(m.ox, &format!("ptq1_0_mmvq_pt_c{nc}"))].into_iter().enumerate() {
                unsafe {
                    let mut pb = p.0.clone();
                    let mut size = pb.len();
                    let mut extra: [*mut c_void; 5] = [1 as *mut c_void, pb.as_mut_ptr() as *mut c_void, 2 as *mut c_void, &mut size as *mut usize as *mut c_void, std::ptr::null_mut()];
                    let (mut e0, mut e1): (cu::CUevent, cu::CUevent) = (std::ptr::null_mut(), std::ptr::null_mut());
                    ck!(cu::cuEventCreate(&mut e0, 0));
                    ck!(cu::cuEventCreate(&mut e1, 0));
                    for it in 0..120 {
                        if it == 20 {
                            ck!(cu::cuEventRecord(e0, std::ptr::null_mut()));
                        }
                        ck!(cu::cuLaunchKernel(f, grid, 1, 1, 128, 1, 1, smem, std::ptr::null_mut(), std::ptr::null_mut(), extra.as_mut_ptr()));
                    }
                    ck!(cu::cuEventRecord(e1, std::ptr::null_mut()));
                    ck!(cu::cuEventSynchronize(e1));
                    let mut ms = 0f32;
                    ck!(cu::cuEventElapsedTime_v2(&mut ms, e0, e1));
                    us[s] = ms * 10.0;
                }
            }
            let gb = (bpr as f64 * nrows as f64 * 28.0) / 1e9;
            println!("mmvq_pt nc={nc} K={k} N={nrows}: ref {:.1} us ({:.0} GB/s), oxide {:.1} us ({:.0} GB/s)", us[0], gb / (us[0] as f64 * 1e-6), us[1], gb / (us[1] as f64 * 1e-6));
        }
    }
}

pub fn run() -> bool {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--bench") {
        bench();
        return true;
    }
    let only = args.iter().position(|a| a == "--only").and_then(|i| args.get(i + 1)).map(|s| s.as_str());
    run_gate(only, args.iter().any(|a| a == "--quick"))
}
