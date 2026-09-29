//! Launcher-level differential gate: every one of the 33 `launch_mmvq_gguf_*` host launchers is
//! called twice on identical inputs in the same (primary) context and stream -- once the REAL C
//! launcher from candle-kernels' libmoe.a (nvcc kernels), once the pure-Rust twin in
//! `crate::launch` (oxide kernels) -- and every output byte is compared (including the untouched
//! gaps of strided outputs). A kernel-level kdiff of every entry against mmvq_gguf.cubin follows.
use crate::launch as ox;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use kdiff::{Arg, Harness, Rng, Tally, as_bytes};
use std::ffi::c_void;
use std::sync::Arc;

/// The C launchers (libmoe.a).
mod cref {
    use std::ffi::c_void;
    macro_rules! decl_plain {
        ($($n:ident),*) => { unsafe extern "C" { $(pub fn $n(vx: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32, nrows_x: i32,
                                                        stride_col_y: i32, stride_col_dst: i32, b_size: i32, stream: *mut c_void);)* } };
    }
    decl_plain!(
        launch_mmvq_gguf_q4_0_bf16_plain, launch_mmvq_gguf_q4_1_bf16_plain, launch_mmvq_gguf_q5_0_bf16_plain,
        launch_mmvq_gguf_q5_1_bf16_plain, launch_mmvq_gguf_q8_0_bf16_plain, launch_mmvq_gguf_q2_k_bf16_plain,
        launch_mmvq_gguf_q3_k_bf16_plain, launch_mmvq_gguf_q4_k_bf16_plain, launch_mmvq_gguf_q5_k_bf16_plain,
        launch_mmvq_gguf_q6_k_bf16_plain,
        launch_mmvq_gguf_q4_0_f16_plain, launch_mmvq_gguf_q4_1_f16_plain, launch_mmvq_gguf_q5_0_f16_plain,
        launch_mmvq_gguf_q5_1_f16_plain, launch_mmvq_gguf_q8_0_f16_plain, launch_mmvq_gguf_q2_k_f16_plain,
        launch_mmvq_gguf_q3_k_f16_plain, launch_mmvq_gguf_q4_k_f16_plain, launch_mmvq_gguf_q5_k_f16_plain,
        launch_mmvq_gguf_q6_k_f16_plain,
        launch_mmvq_gguf_q4_0_f32_plain, launch_mmvq_gguf_q4_1_f32_plain, launch_mmvq_gguf_q5_0_f32_plain,
        launch_mmvq_gguf_q5_1_f32_plain, launch_mmvq_gguf_q8_0_f32_plain, launch_mmvq_gguf_q2_k_f32_plain,
        launch_mmvq_gguf_q3_k_f32_plain, launch_mmvq_gguf_q4_k_f32_plain, launch_mmvq_gguf_q5_k_f32_plain,
        launch_mmvq_gguf_q6_k_f32_plain
    );
    unsafe extern "C" {
        pub fn launch_mmvq_gguf_quantize_q8_1_bf16(x: *const c_void, vy: *mut c_void, kx: i32, kx_padded: i32, num_rows: i32, stream: *mut c_void);
        pub fn launch_mmvq_gguf_quantize_q8_1_f16(x: *const c_void, vy: *mut c_void, kx: i32, kx_padded: i32, num_rows: i32, stream: *mut c_void);
        pub fn launch_mmvq_gguf_quantize_q8_1_f32(x: *const c_void, vy: *mut c_void, kx: i32, kx_padded: i32, num_rows: i32, stream: *mut c_void);
    }
}

type Plain = unsafe extern "C" fn(*const c_void, *const c_void, *mut c_void, i32, i32, i32, i32, i32, *mut c_void);
type Quant = unsafe extern "C" fn(*const c_void, *mut c_void, i32, i32, i32, *mut c_void);

/// A GGUF weight format: name, block bytes, values per block, f16 scale-field offsets.
struct Fmt {
    name: &'static str,
    bs: usize,
    qk: usize,
    f16s: &'static [usize],
}
const FMTS: [Fmt; 10] = [
    Fmt { name: "q4_0", bs: 18, qk: 32, f16s: &[0] },
    Fmt { name: "q4_1", bs: 20, qk: 32, f16s: &[0, 2] },
    Fmt { name: "q5_0", bs: 22, qk: 32, f16s: &[0] },
    Fmt { name: "q5_1", bs: 24, qk: 32, f16s: &[0, 2] },
    Fmt { name: "q8_0", bs: 34, qk: 32, f16s: &[0] },
    Fmt { name: "q2_k", bs: 84, qk: 256, f16s: &[80, 82] },
    Fmt { name: "q3_k", bs: 110, qk: 256, f16s: &[108] },
    Fmt { name: "q4_k", bs: 144, qk: 256, f16s: &[0, 2] },
    Fmt { name: "q5_k", bs: 176, qk: 256, f16s: &[0, 2] },
    Fmt { name: "q6_k", bs: 210, qk: 256, f16s: &[208] },
];

fn plain_pair(fi: usize, dst: usize) -> (Plain, Plain) {
    use cref as c;
    let t: [[(Plain, Plain); 10]; 3] = [
        [
            (c::launch_mmvq_gguf_q4_0_bf16_plain, ox::launch_mmvq_gguf_q4_0_bf16_plain),
            (c::launch_mmvq_gguf_q4_1_bf16_plain, ox::launch_mmvq_gguf_q4_1_bf16_plain),
            (c::launch_mmvq_gguf_q5_0_bf16_plain, ox::launch_mmvq_gguf_q5_0_bf16_plain),
            (c::launch_mmvq_gguf_q5_1_bf16_plain, ox::launch_mmvq_gguf_q5_1_bf16_plain),
            (c::launch_mmvq_gguf_q8_0_bf16_plain, ox::launch_mmvq_gguf_q8_0_bf16_plain),
            (c::launch_mmvq_gguf_q2_k_bf16_plain, ox::launch_mmvq_gguf_q2_k_bf16_plain),
            (c::launch_mmvq_gguf_q3_k_bf16_plain, ox::launch_mmvq_gguf_q3_k_bf16_plain),
            (c::launch_mmvq_gguf_q4_k_bf16_plain, ox::launch_mmvq_gguf_q4_k_bf16_plain),
            (c::launch_mmvq_gguf_q5_k_bf16_plain, ox::launch_mmvq_gguf_q5_k_bf16_plain),
            (c::launch_mmvq_gguf_q6_k_bf16_plain, ox::launch_mmvq_gguf_q6_k_bf16_plain),
        ],
        [
            (c::launch_mmvq_gguf_q4_0_f16_plain, ox::launch_mmvq_gguf_q4_0_f16_plain),
            (c::launch_mmvq_gguf_q4_1_f16_plain, ox::launch_mmvq_gguf_q4_1_f16_plain),
            (c::launch_mmvq_gguf_q5_0_f16_plain, ox::launch_mmvq_gguf_q5_0_f16_plain),
            (c::launch_mmvq_gguf_q5_1_f16_plain, ox::launch_mmvq_gguf_q5_1_f16_plain),
            (c::launch_mmvq_gguf_q8_0_f16_plain, ox::launch_mmvq_gguf_q8_0_f16_plain),
            (c::launch_mmvq_gguf_q2_k_f16_plain, ox::launch_mmvq_gguf_q2_k_f16_plain),
            (c::launch_mmvq_gguf_q3_k_f16_plain, ox::launch_mmvq_gguf_q3_k_f16_plain),
            (c::launch_mmvq_gguf_q4_k_f16_plain, ox::launch_mmvq_gguf_q4_k_f16_plain),
            (c::launch_mmvq_gguf_q5_k_f16_plain, ox::launch_mmvq_gguf_q5_k_f16_plain),
            (c::launch_mmvq_gguf_q6_k_f16_plain, ox::launch_mmvq_gguf_q6_k_f16_plain),
        ],
        [
            (c::launch_mmvq_gguf_q4_0_f32_plain, ox::launch_mmvq_gguf_q4_0_f32_plain),
            (c::launch_mmvq_gguf_q4_1_f32_plain, ox::launch_mmvq_gguf_q4_1_f32_plain),
            (c::launch_mmvq_gguf_q5_0_f32_plain, ox::launch_mmvq_gguf_q5_0_f32_plain),
            (c::launch_mmvq_gguf_q5_1_f32_plain, ox::launch_mmvq_gguf_q5_1_f32_plain),
            (c::launch_mmvq_gguf_q8_0_f32_plain, ox::launch_mmvq_gguf_q8_0_f32_plain),
            (c::launch_mmvq_gguf_q2_k_f32_plain, ox::launch_mmvq_gguf_q2_k_f32_plain),
            (c::launch_mmvq_gguf_q3_k_f32_plain, ox::launch_mmvq_gguf_q3_k_f32_plain),
            (c::launch_mmvq_gguf_q4_k_f32_plain, ox::launch_mmvq_gguf_q4_k_f32_plain),
            (c::launch_mmvq_gguf_q5_k_f32_plain, ox::launch_mmvq_gguf_q5_k_f32_plain),
            (c::launch_mmvq_gguf_q6_k_f32_plain, ox::launch_mmvq_gguf_q6_k_f32_plain),
        ],
    ];
    t[dst][fi]
}

const DSTS: [(&str, usize); 3] = [("bf16", 2), ("f16", 2), ("f32", 4)];

fn quant_pair(t: usize) -> (Quant, Quant) {
    [
        (cref::launch_mmvq_gguf_quantize_q8_1_bf16 as Quant, ox::launch_mmvq_gguf_quantize_q8_1_bf16 as Quant),
        (cref::launch_mmvq_gguf_quantize_q8_1_f16, ox::launch_mmvq_gguf_quantize_q8_1_f16),
        (cref::launch_mmvq_gguf_quantize_q8_1_f32, ox::launch_mmvq_gguf_quantize_q8_1_f32),
    ][t]
}

struct G {
    ctx: Arc<CudaContext>,
    streams: Vec<Arc<CudaStream>>,
    rng: Rng,
    t: Tally,
    calls: usize,
    /// Non-NaN output elements seen (sanity: the data is not all NaN / inf).
    finite: usize,
    elems: usize,
    /// 1 in `srate` f16 scale fields is a special value (nan, inf, denormal, raw bits, ...).
    srate: u64,
}

impl G {
    /// An f16 scale: mostly normal magnitudes 2^-12..2^3, 1 in `srate` a special value or raw bits.
    fn f16_scale(&mut self) -> u16 {
        let r = self.rng.next();
        match if r % self.srate == 0 { (r >> 40) % 2 } else { 2 } {
            0 => [0x0000, 0x8000, 0x7c00, 0xfc00, 0x7e00, 0x0001, 0x83ff, 0x7bff, 0x0400, 0xfe01, 0x7c01, 0x03ff][((r >> 8) % 12) as usize],
            1 => (r >> 16) as u16,
            _ => {
                let sign = ((r >> 8) & 1) as u16;
                let exp = 3 + ((r >> 9) % 16) as u16;
                (sign << 15) | (exp << 10) | ((r >> 16) & 0x3ff) as u16
            }
        }
    }
    /// `n` random weight blocks with f16 scale fields from `f16_scale`, plus trailing pad bytes.
    fn blocks(&mut self, f: &Fmt, n: usize, pad: usize) -> Vec<u8> {
        let mut b = self.rng.bytes(n * f.bs + pad);
        for i in 0..n {
            for &o in f.f16s {
                let v = self.f16_scale();
                b[i * f.bs + o..i * f.bs + o + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
        b
    }
    /// `n` random Q8_1 blocks (36 bytes): random int8 quants, `ds` from `f16_scale`.
    fn q8_1(&mut self, n: usize) -> Vec<u8> {
        let mut b = self.rng.bytes(n * 36);
        for i in 0..n {
            let (d, s) = (self.f16_scale(), self.f16_scale());
            b[i * 36..i * 36 + 2].copy_from_slice(&d.to_le_bytes());
            b[i * 36 + 2..i * 36 + 4].copy_from_slice(&s.to_le_bytes());
        }
        b
    }
    /// Activations of element type `t` (0 bf16, 1 f16, 2 f32): normal values with edge cases.
    fn acts(&mut self, t: usize, n: usize, style: usize) -> Vec<u8> {
        let v: Vec<f32> = match style {
            // all zero (amax == 0)
            1 => vec![0.0; n],
            // tiny (denormal quotients) and exact .5 ties at d = 1
            2 => (0..n).map(|i| if i % 32 == 0 { 127.0 } else { ((i % 64) as f32 - 32.0) * 0.5 }).collect(),
            3 => (0..n).map(|i| (i as f32 - 300.0) * 1e-38).collect(),
            // blocks of 32 whose quotients x / (amax / 127) sit where roundf's add.rz differs from add.rn
            5 => tie_values(t, n, &mut self.rng),
            // mostly plain values, about one special per `srate` elements (long GEMV rows)
            4 => (0..n)
                .map(|_| {
                    let r = self.rng.next();
                    if r % self.srate == 0 { self.rng.f32s(9)[(r >> 32) as usize % 9] } else { ((r >> 11) as f64 / (1u64 << 53) as f64 * 16.0 - 8.0) as f32 }
                })
                .collect(),
            _ => self.rng.f32s(n),
        };
        match t {
            2 => as_bytes(&v),
            _ => {
                let h: Vec<u16> = v
                    .iter()
                    .map(|&x| {
                        let r = self.rng.next();
                        if (style == 0 && r % 12 == 0) || (style == 4 && r % self.srate == 0) {
                            r as u16 // raw 16-bit pattern: every class incl. nan/inf/denormal
                        } else if t == 0 {
                            (x.to_bits() >> 16) as u16
                        } else {
                            f32_to_f16_trunc(x)
                        }
                    })
                    .collect();
                as_bytes(&h)
            }
        }
    }

    fn up(&self, b: &[u8]) -> DeviceBuffer<u8> {
        DeviceBuffer::from_host(&self.streams[0], if b.is_empty() { &[0u8][..] } else { b }).unwrap()
    }
    fn down(&self, b: &DeviceBuffer<u8>) -> Vec<u8> {
        b.to_host_vec(&self.streams[0]).unwrap()
    }

    fn compare(&mut self, label: &str, a: &[u8], b: &[u8], elem: usize) {
        self.calls += 2;
        let d = kdiff::Diff {
            bytes: a.len(),
            differing: a.iter().zip(b).filter(|(x, y)| x != y).count(),
            first: a.iter().zip(b).position(|(x, y)| x != y).map(|i| (0, i, a[i], b[i])),
        };
        if elem == 4 {
            for c in a.chunks_exact(4) {
                self.elems += 1;
                self.finite += !f32::from_le_bytes([c[0], c[1], c[2], c[3]]).is_nan() as usize;
            }
        }
        self.t.record(label, &d);
    }

    /// One plain-launcher case: identical weights / q8_1 / pre-filled dst for both launchers.
    #[allow(clippy::too_many_arguments)]
    fn plain(&mut self, fi: usize, di: usize, ncols: i32, nrows: i32, scy: i32, scd: i32, b: i32, stream: usize) {
        let f = &FMTS[fi];
        let bpr = (ncols.max(0) as usize) / f.qk;
        // Specials in roughly a third of the outputs: every scale field of a row feeds one output.
        self.srate = (3 * bpr * (f.f16s.len() + 2 * f.qk / 32)).max(16) as u64;
        // One spare row of blocks: 2-row CUDA blocks read row `nrows` when nrows is odd.
        let w = self.blocks(f, (nrows.max(0) as usize + 1) * bpr.max(1), 64);
        let ncol_y = b.clamp(1, 8) as usize;
        let yblocks = (ncol_y - 1) * scy as usize + bpr * f.qk / 32 + 8;
        let y = self.q8_1(yblocks);
        let es = DSTS[di].1;
        let dlen = ((ncol_y - 1) * scd.max(0) as usize + nrows.max(0) as usize + 3) * es;
        let dinit = self.rng.bytes(dlen);
        let (cf, of) = plain_pair(fi, di);
        let (wd, yd, d_ref, d_ox) = (self.up(&w), self.up(&y), self.up(&dinit), self.up(&dinit));
        let s = if stream == 0 { std::ptr::null_mut() } else { self.streams[stream].cu_stream() as *mut c_void };
        unsafe {
            cf(wd.cu_deviceptr() as _, yd.cu_deviceptr() as _, d_ref.cu_deviceptr() as _, ncols, nrows, scy, scd, b, s);
            of(wd.cu_deviceptr() as _, yd.cu_deviceptr() as _, d_ox.cu_deviceptr() as _, ncols, nrows, scy, scd, b, s);
        }
        self.ctx.synchronize().unwrap();
        let (a, o) = (self.down(&d_ref), self.down(&d_ox));
        let label = format!("{}_{} ncols={ncols} nrows={nrows} scy={scy} scd={scd} b={b} s={stream}", f.name, DSTS[di].0);
        self.compare(&label, &a, &o, es);
    }

    /// One quantize-launcher case.
    fn quant(&mut self, t: usize, kx: i32, kxp: i32, rows: i32, style: usize, stream: usize) {
        let x = self.acts(t, (rows * kx.max(1)) as usize, style);
        let ylen = (rows * kxp) as usize / 32 * 36 + 36;
        let yinit = self.rng.bytes(ylen);
        let (cf, of) = quant_pair(t);
        let (xd, y_ref, y_ox) = (self.up(&x), self.up(&yinit), self.up(&yinit));
        let s = if stream == 0 { std::ptr::null_mut() } else { self.streams[stream].cu_stream() as *mut c_void };
        unsafe {
            cf(xd.cu_deviceptr() as _, y_ref.cu_deviceptr() as _, kx, kxp, rows, s);
            of(xd.cu_deviceptr() as _, y_ox.cu_deviceptr() as _, kx, kxp, rows, s);
        }
        self.ctx.synchronize().unwrap();
        let (a, o) = (self.down(&y_ref), self.down(&y_ox));
        self.compare(&format!("quantize_q8_1_{} kx={kx} kxp={kxp} rows={rows} style={style} s={stream}", ["bf16", "f16", "f32"][t]), &a, &o, 1);
    }

    /// candle's fast_mmvq call sequence: quantize then GEMV, each side with its own launchers.
    fn pipeline(&mut self, fi: usize, di: usize, k: usize, nrows: usize, b: usize, stream: usize) {
        let f = &FMTS[fi];
        let kp = k.div_ceil(512) * 512;
        let bpr = k / f.qk;
        self.srate = (3 * bpr * (f.f16s.len() + 2 * f.qk / 32)).max(16) as u64;
        let w = self.blocks(f, (nrows + 1) * bpr, 64);
        self.srate = 3 * k as u64;
        let x = self.acts(di, b * k, 4);
        let es = DSTS[di].1;
        let dinit = self.rng.bytes(b * nrows * es);
        let (cq, oq) = quant_pair(di);
        let (cf, of) = plain_pair(fi, di);
        let (wd, xd) = (self.up(&w), self.up(&x));
        let yinit = vec![0u8; b * kp / 32 * 36];
        let (yr, yo, dr, dox) = (self.up(&yinit), self.up(&yinit), self.up(&dinit), self.up(&dinit));
        let s = if stream == 0 { std::ptr::null_mut() } else { self.streams[stream].cu_stream() as *mut c_void };
        let (ki, kpi, bi, ni, scy) = (k as i32, kp as i32, b as i32, nrows as i32, (kp / 32) as i32);
        unsafe {
            cq(xd.cu_deviceptr() as _, yr.cu_deviceptr() as _, ki, kpi, bi, s);
            cf(wd.cu_deviceptr() as _, yr.cu_deviceptr() as _, dr.cu_deviceptr() as _, ki, ni, scy, ni, bi, s);
            oq(xd.cu_deviceptr() as _, yo.cu_deviceptr() as _, ki, kpi, bi, s);
            of(wd.cu_deviceptr() as _, yo.cu_deviceptr() as _, dox.cu_deviceptr() as _, ki, ni, scy, ni, bi, s);
        }
        self.ctx.synchronize().unwrap();
        let (a, o) = (self.down(&dr), self.down(&dox));
        let label = format!("pipeline {}_{} k={k} nrows={nrows} b={b} s={stream}", f.name, DSTS[di].0);
        self.compare(&label, &a, &o, es);
        self.calls += 2; // the two quantize launches
    }
}

/// roundf(t) as nvcc emits it (`trunc(add.rz(t, copysign(0.5, t)))`) and with add.rn instead.
fn round_rz_rn(t: f32) -> (i64, i64) {
    let h = if t.is_sign_negative() { -0.5f32 } else { 0.5 };
    let exact = t as f64 + h as f64;
    let mut rz = exact as f32;
    if (rz as f64).abs() > exact.abs() {
        rz = f32::from_bits(rz.to_bits() - 1);
    }
    (rz.trunc() as i64, (t + h).trunc() as i64)
}

/// Activations (as f32 values exactly representable in type `t`) laid out in 32-element blocks:
/// lane 0 holds the block's amax, the other lanes hold x with x / (amax / 127) at a value where
/// the rounding mode of roundf's add decides the result (found by search on the host).
fn tie_values(t: usize, n: usize, rng: &mut Rng) -> Vec<f32> {
    let to_f = |bits: u32| -> f32 {
        match t {
            0 => f32::from_bits(bits << 16),
            1 => f16_bits_to_f32(bits as u16),
            _ => f32::from_bits(bits),
        }
    };
    let from_f = |x: f32| -> Option<u32> {
        match t {
            0 => ((x.to_bits() & 0xFFFF) == 0).then_some(x.to_bits() >> 16),
            1 => {
                let h = f32_to_f16_trunc(x);
                (f16_bits_to_f32(h) == x).then_some(h as u32)
            }
            _ => Some(x.to_bits()),
        }
    };
    let mut pairs: Vec<(f32, f32)> = Vec::new();
    let mut sens = 0;
    let mut tries = 0;
    while pairs.len() < 64 && tries < 2_000_000 {
        tries += 1;
        let abits = match t {
            0 => 0x0080 + (rng.next() % 0x7e00) as u32,
            1 => 0x0001 + (rng.next() % 0x7bfe) as u32,
            _ => 0x0080_0000 + (rng.next() % 0x7e00_0000) as u32,
        };
        let amax = to_f(abits);
        let d = amax / 127.0;
        if !(d > 0.0) || !d.is_finite() {
            continue;
        }
        // x near (k + 0.5) * d for small k, stepped by a few representable values.
        let k = (rng.next() % 8) as f32;
        let Some(c) = from_f(to_f_round(t, (k + 0.5) * d)) else { continue };
        for dx in -3i32..=3 {
            let xb = (c as i32 + dx) as u32;
            let x = to_f(xb);
            if !(x.abs() < amax) {
                continue;
            }
            let q = x / d;
            let (a, b) = round_rz_rn(q);
            // rz-vs-rn sensitive quotients exist only for f32 input (exhaustive search: none for
            // bf16 / f16), so those types get exact .5 ties (roundf rounds them away from zero).
            if a != b {
                sens += 1;
            }
            if a != b || (q.fract().abs() == 0.5 && pairs.len() < 48) {
                pairs.push((amax, if rng.next() % 2 == 0 { x } else { -x }));
            }
        }
    }
    assert!(!pairs.is_empty() && (t != 2 || sens > 0), "no rounding-sensitive activations found for type {t}");
    (0..n).map(|i| { let p = pairs[(i / 32) % pairs.len()]; if i % 32 == 0 { p.0 } else { p.1 } }).collect()
}

/// Round an f32 to the nearest value of type `t` (for search seeds only).
fn to_f_round(t: usize, x: f32) -> f32 {
    match t {
        0 => f32::from_bits(x.to_bits() & 0xFFFF_0000),
        1 => f16_bits_to_f32(f32_to_f16_trunc(x)),
        _ => x,
    }
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let v = if e == 0 {
        (m as f32) * (2.0f32).powi(-24)
    } else if e == 31 {
        return f32::from_bits(sign | 0x7f80_0000 | (m << 13));
    } else {
        return f32::from_bits(sign | ((e + 112) << 23) | (m << 13));
    };
    f32::from_bits(v.to_bits() | sign)
}

/// f32 -> f16 bits (truncating; only used to make test data).
fn f32_to_f16_trunc(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    if x.is_nan() {
        return sign | 0x7e00;
    }
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = (b & 0x7fffff) | 0x800000;
        return sign | (m >> (14 - e)) as u16;
    }
    sign | ((e as u16) << 10) | ((b >> 13) & 0x3ff) as u16
}

pub fn run() -> bool {
    let ctx = CudaContext::new(0).expect("cuda context");
    ctx.bind_to_thread().unwrap();
    let streams = vec![ctx.default_stream(), ctx.new_stream().unwrap()];
    let mut g = G { ctx: ctx.clone(), streams, rng: Rng(0x3317_9A1D), t: Tally::default(), calls: 0, finite: 0, elems: 0, srate: 16 };

    // ---- plain launchers: every b_size branch (incl. the no-launch default), odd / large shapes,
    // padded y and dst strides, both the null stream and a created stream.
    for fi in 0..FMTS.len() {
        let qk = FMTS[fi].qk as i32;
        for di in 0..3 {
            for b in 0..=9 {
                // (ncols, nrows, extra y stride, extra dst stride)
                let shapes: [(i32, i32, i32, i32); 4] = [
                    (qk * 3, 7, 0, 0),
                    (qk * 17 + qk / 2 + 3, 33, 5, 3),
                    (qk * 2, 1, 16, 1),
                    (if qk == 32 { 4096 } else { 4608 }, 131, 16, 0),
                ];
                for (si, &(ncols, nrows, ey, ed)) in shapes.iter().enumerate() {
                    let scy = (ncols + 511) / 512 * 512 / 32 + ey;
                    let scd = nrows + ed;
                    g.plain(fi, di, ncols, nrows, scy, scd, b, (si + b as usize) % 2);
                }
            }
            // larger row counts at the batch sizes that switch geometry
            for &b in &[1, 2, 4, 5, 8] {
                g.plain(fi, di, qk * 8, 1025, qk * 8 / 32 + 3, 1030, b, 1);
            }
            // ncols smaller than one block (no loop iterations) and one block
            for &b in &[1, 3, 6] {
                g.plain(fi, di, qk - 1, 5, 4, 9, b, 0);
                g.plain(fi, di, qk, 6, 9, 6, b, 1);
            }
            // empty grid (nrows = 0): the launch fails in both and dst stays untouched
            for &b in &[1, 5] {
                g.plain(fi, di, qk * 2, 0, 8, 4, b, b as usize % 2);
            }
            // candle's own call sequence
            for &(k, n, b) in &[(2048usize, 96usize, 1usize), (1536, 61, 3), (4096, 40, 8), (2560, 17, 5)] {
                g.pipeline(fi, di, k, n, b, (k / 512) % 2);
            }
        }
    }

    // ---- quantize launchers: padded tails, odd kx, rows beyond 8, a partial 256-thread block,
    // edge-value activations (zeros, ties, denormals, nan / inf, raw 16-bit patterns).
    for t in 0..3 {
        for &(kx, kxp) in &[(512, 512), (1000, 1024), (1, 512), (511, 512), (2048, 2560), (4096, 4096), (0, 512), (96, 96), (70, 96), (3000, 3072)] {
            for &rows in &[1, 2, 5, 8, 13] {
                for style in 0..6 {
                    if (style > 0 && rows > 2) || style == 4 {
                        continue;
                    }
                    g.quant(t, kx, kxp, rows, style, (rows as usize + style) % 2);
                }
            }
        }
    }

    // ---- kernel level: every one of the 243 entries against the cubin by name (kdiff).
    let root = format!("{}/titan-engine/oxide-kernels", std::env::var("HOME").unwrap());
    let h = Harness::from_files(&format!("{root}/reference/candle-ffi/mmvq_gguf.cubin"), &format!("{root}/candle-mmvq/candle_mmvq.ptx"));
    let mut kt = Tally::default();
    let mut r = Rng(0x51DE);
    for f in FMTS.iter() {
        for (dn, es) in DSTS.iter() {
            for n in 1..=8usize {
                let name = format!("mmvq_gguf_{}_{dn}_plain_cuda{n}", f.name);
                let (ncols, nrows) = (f.qk * 9, 45usize);
                let bpr = ncols / f.qk;
                let rows_per = if n == 1 { 1 } else { 2 };
                let nw = if n <= 4 { 4 } else { 2 };
                let scy = ncols / 32 + 2;
                let scd = nrows + 5;
                g.rng = Rng(r.next() | 1);
                g.srate = (3 * bpr * (f.f16s.len() + 2 * f.qk / 32)) as u64;
                let w = g.blocks(f, (nrows + 1) * bpr, 64);
                let y = g.q8_1((n - 1) * scy + bpr * f.qk / 32 + 8);
                let d = r.bytes(((n - 1) * scd + nrows) * es);
                let args = [Arg::Buf(0), Arg::Buf(1), Arg::Buf(2), Arg::I32(ncols as i32), Arg::I32(nrows as i32), Arg::I32(scy as i32), Arg::I32(scd as i32)];
                let dd = h.diff(&name, ((nrows as u32).div_ceil(rows_per), 1, 1), (32, nw, 1), 0, &args, &[w, y, d], &[2]);
                kt.record(&name, &dd);
            }
        }
    }
    for (t, tn) in ["bf16", "f16", "f32"].iter().enumerate() {
        let name = format!("mmvq_gguf_quantize_q8_1_{tn}");
        let (kx, kxp, rows) = (1000i32, 1024i32, 3i32);
        g.rng = Rng(r.next() | 1);
        let x = g.acts(t, (kx * rows) as usize, 0);
        let y = r.bytes((kxp * rows) as usize / 32 * 36);
        let args = [Arg::Buf(0), Arg::Buf(1), Arg::I32(kx), Arg::I32(kxp)];
        let dd = h.diff(&name, ((kxp as u32).div_ceil(256), rows as u32, 1), (256, 1, 1), 0, &args, &[x, y], &[1]);
        kt.record(&name, &dd);
    }
    let kok = kt.finish("mmvq kernel-level kdiff vs cubin");

    println!("f32 outputs: {} of {} not NaN", g.finite, g.elems);
    for f in g.t.failures.iter().take(40) {
        println!("  FAIL {f}");
    }
    let ok = g.t.failures.is_empty() && kok && g.finite * 2 > g.elems;
    println!(
        "mmvq: {} launcher calls, {} bytes compared, {} failing -> {}",
        g.calls,
        g.t.bytes,
        g.t.failures.len() + kt.failures.len(),
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}
