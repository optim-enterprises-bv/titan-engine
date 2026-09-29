//! Launcher-level differential gate: every extern "C" launcher of group B is called twice on
//! identical inputs in the same (primary) context -- once the REAL C launcher from
//! libmistralrsquant.a (nvcc kernels), once the pure-Rust twin in `crate::launch` (oxide kernels) --
//! and every byte of every output buffer is compared (including untouched gaps). Each output
//! buffer starts as identical random bytes, so a launch that writes nothing, or writes too much,
//! shows up too. A call where neither side changed any output byte is counted as "inert"; every
//! family must have non-inert calls.
use crate::launch as ox;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use kdiff::{Rng, Tally, as_bytes};
use std::ffi::c_void;
use std::sync::Arc;

/// The C launchers (libmistralrsquant.a), declared with the same types as `crate::launch`.
pub mod cref {
    use std::ffi::c_void;
    include!("gen_cref.rs");
    unsafe extern "C" {
        pub fn gemm_half_q_half_cuda_part(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, perm: *const i32, c: *mut u16, m: i32, n: i32, k: i32, m_count: i32, groups: i32, bit: i32);
        pub fn reconstruct_exllama(bq: *const u32, qz: *const u32, sc: *const u16, perm: *const i32, out: *mut u16, height: i32, width: i32, groups: i32, bit: i32);
        pub fn gemm_half_q_half_alt(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, g_idx: *const i32, c: *mut u16, m: i32, n: i32, k: i32, bit: i32);
        pub fn reconstruct_gptq(bq: *const u32, qz: *const u32, sc: *const u16, g_idx: *const i32, out: *mut u16, height: i32, width: i32, groups: i32, bit: i32);
        pub fn launch_mxfp4_matmul_f16(i: *const u16, w: *const u8, ws: *const u8, b: *const u16, o: *mut u16, m: i32, n: i32, k: i32, hb: bool, s: *mut c_void);
        pub fn launch_mxfp4_matmul_bf16(i: *const u16, w: *const u8, ws: *const u8, b: *const u16, o: *mut u16, m: i32, n: i32, k: i32, hb: bool, s: *mut c_void);
        pub fn launch_mxfp4_matmul_wmma_f16(i: *const u16, w: *const u8, ws: *const u8, b: *const u16, o: *mut u16, m: i32, n: i32, k: i32, hb: bool, s: *mut c_void);
        pub fn launch_mxfp4_matmul_wmma_bf16(i: *const u16, w: *const u8, ws: *const u8, b: *const u16, o: *mut u16, m: i32, n: i32, k: i32, hb: bool, s: *mut c_void);
        pub fn launch_mxfp4_indexed_moe_gemm_f16(i: *const u16, w: *const u8, ws: *const u8, b: *const u16, idx: *const u32, o: *mut u16, nt: i32, topk: i32, ne: i32, n: i32, k: i32, hb: bool, ht: bool, s: *mut c_void);
        pub fn launch_mxfp4_indexed_moe_gemm_bf16(i: *const u16, w: *const u8, ws: *const u8, b: *const u16, idx: *const u32, o: *mut u16, nt: i32, topk: i32, ne: i32, n: i32, k: i32, hb: bool, ht: bool, s: *mut c_void);
        pub fn launch_mxfp4_moe_grouped_gemm_f16(i: *const u16, w: *const u8, ws: *const u8, b: *const u16, idx: *const u32, o: *mut u16, nt: i32, topk: i32, ne: i32, n: i32, k: i32, hb: bool, ht: bool, s: *mut c_void);
        pub fn launch_mxfp4_moe_grouped_gemm_bf16(i: *const u16, w: *const u8, ws: *const u8, b: *const u16, idx: *const u32, o: *mut u16, nt: i32, topk: i32, ne: i32, n: i32, k: i32, hb: bool, ht: bool, s: *mut c_void);
        pub fn launch_mxfp4_moe_grouped_gemm_wmma_f16(i: *const u16, w: *const u8, ws: *const u8, b: *const u16, idx: *const u32, o: *mut u16, nt: i32, topk: i32, ne: i32, n: i32, k: i32, hb: bool, ht: bool, s: *mut c_void);
        pub fn launch_mxfp4_moe_grouped_gemm_wmma_bf16(i: *const u16, w: *const u8, ws: *const u8, b: *const u16, idx: *const u32, o: *mut u16, nt: i32, topk: i32, ne: i32, n: i32, k: i32, hb: bool, ht: bool, s: *mut c_void);
        pub fn mxfp4_get_max_smem_optin() -> i32;
        pub fn marlin_gptq_4bit_f16(a: *const c_void, b: *const c_void, s: *mut c_void, z: *mut c_void, c: *mut c_void, m: i32, k: i32, n: i32, ws: *mut c_void, gs: i32, st: i64);
        pub fn marlin_gptq_4bit_bf16(a: *const c_void, b: *const c_void, s: *mut c_void, z: *mut c_void, c: *mut c_void, m: i32, k: i32, n: i32, ws: *mut c_void, gs: i32, st: i64);
        pub fn marlin_awq_4bit_f16(a: *const c_void, b: *const c_void, s: *mut c_void, z: *mut c_void, c: *mut c_void, m: i32, k: i32, n: i32, ws: *mut c_void, gs: i32, st: i64);
        pub fn marlin_awq_4bit_bf16(a: *const c_void, b: *const c_void, s: *mut c_void, z: *mut c_void, c: *mut c_void, m: i32, k: i32, n: i32, ws: *mut c_void, gs: i32, st: i64);
        pub fn gptq_marlin_repack(w: *mut c_void, perm: *mut c_void, out: *mut c_void, k: i32, n: i32, bits: i32, st: i64);
        pub fn awq_marlin_repack(w: *mut c_void, perm: *mut c_void, out: *mut c_void, k: i32, n: i32, bits: i32, st: i64);
        pub fn dequantize_blockwise_f32_int8(code: *mut f32, a: *mut u8, absmax: *mut f32, out: *mut f32, bs: i32, n: i32, s: *mut c_void);
        pub fn dequantize_blockwise_f32_fp4(code: *mut f32, a: *mut u8, absmax: *mut f32, out: *mut f32, bs: i32, n: i32, s: *mut c_void);
        pub fn dequantize_blockwise_f32_nf4(code: *mut f32, a: *mut u8, absmax: *mut f32, out: *mut f32, bs: i32, n: i32, s: *mut c_void);
        pub fn dequantize_blockwise_f16_int8(code: *mut f32, a: *mut u8, absmax: *mut f32, out: *mut u16, bs: i32, n: i32, s: *mut c_void);
        pub fn dequantize_blockwise_f16_fp4(code: *mut f32, a: *mut u8, absmax: *mut f32, out: *mut u16, bs: i32, n: i32, s: *mut c_void);
        pub fn dequantize_blockwise_f16_nf4(code: *mut f32, a: *mut u8, absmax: *mut f32, out: *mut u16, bs: i32, n: i32, s: *mut c_void);
        pub fn dequantize_blockwise_bf16_int8(code: *mut f32, a: *mut u8, absmax: *mut f32, out: *mut u16, bs: i32, n: i32, s: *mut c_void);
        pub fn dequantize_blockwise_bf16_fp4(code: *mut f32, a: *mut u8, absmax: *mut f32, out: *mut u16, bs: i32, n: i32, s: *mut c_void);
        pub fn dequantize_blockwise_bf16_nf4(code: *mut f32, a: *mut u8, absmax: *mut f32, out: *mut u16, bs: i32, n: i32, s: *mut c_void);
        pub fn launch_cutlass_moe_grouped_gemm_2x_bf16(a: *mut *const c_void, b: *mut *const c_void, d: *mut *mut c_void, ps: *const i32, pc: i32, lda: *mut i64, ldb: *mut i64, ldd: *mut i64, ws: *mut c_void, wss: usize, cfg: i32, s: *mut c_void) -> i32;
        pub fn cutlass_moe_grouped_gemm_2x_workspace_size(pc: i32) -> usize;
        pub fn launch_fp8_to_f32_kernel(d_input: *const u8, d_output: *mut f32, num_elements: usize, stream: *mut c_void);
        pub fn launch_fp8_to_f16_kernel(d_input: *const u8, d_output: *mut u16, num_elements: usize, stream: *mut c_void);
        pub fn launch_fp8_to_bf16_kernel(d_input: *const u8, d_output: *mut u16, num_elements: usize, stream: *mut c_void);
        pub fn launch_f32_to_fp8_kernel(d_input: *const f32, d_output: *mut u8, num_elements: usize, stream: *mut c_void);
        pub fn launch_f16_to_fp8_kernel(d_input: *const u16, d_output: *mut u8, num_elements: usize, stream: *mut c_void);
        pub fn launch_bf16_to_fp8_kernel(d_input: *const u16, d_output: *mut u8, num_elements: usize, stream: *mut c_void);
        pub fn launch_dequant_fp8_vector_kernel_f32(w: *const u8, s: *const f32, o: *mut f32, n: usize, stream: *mut c_void);
        pub fn launch_dequant_fp8_vector_kernel_f16(w: *const u8, s: *const f32, o: *mut u16, n: usize, stream: *mut c_void);
        pub fn launch_dequant_fp8_vector_kernel_bf16(w: *const u8, s: *const f32, o: *mut u16, n: usize, stream: *mut c_void);
        pub fn launch_quant_fp8_vector_kernel_f32(i: *const f32, w: *mut u8, s: *mut f32, n: usize, stream: *mut c_void);
        pub fn launch_quant_fp8_vector_kernel_f16(i: *const u16, w: *mut u8, s: *mut f32, n: usize, stream: *mut c_void);
        pub fn launch_quant_fp8_vector_kernel_bf16(i: *const u16, w: *mut u8, s: *mut f32, n: usize, stream: *mut c_void);
        pub fn launch_dequant_fp8_blockwise_kernel_f32(w: *const u8, s: *const f32, o: *mut f32, h: i32, wd: i32, rs: i32, ss: i32, by: i32, bx: i32, stream: *mut c_void);
        pub fn launch_dequant_fp8_blockwise_kernel_f16(w: *const u8, s: *const f32, o: *mut u16, h: i32, wd: i32, rs: i32, ss: i32, by: i32, bx: i32, stream: *mut c_void);
        pub fn launch_dequant_fp8_blockwise_kernel_bf16(w: *const u8, s: *const f32, o: *mut u16, h: i32, wd: i32, rs: i32, ss: i32, by: i32, bx: i32, stream: *mut c_void);
        pub fn launch_quant_fp8_blockwise_kernel_f32(i: *const f32, w: *mut u8, s: *mut f32, h: i32, wd: i32, rs: i32, ss: i32, by: i32, bx: i32, stream: *mut c_void);
        pub fn launch_quant_fp8_blockwise_kernel_f16(i: *const u16, w: *mut u8, s: *mut f32, h: i32, wd: i32, rs: i32, ss: i32, by: i32, bx: i32, stream: *mut c_void);
        pub fn launch_quant_fp8_blockwise_kernel_bf16(i: *const u16, w: *mut u8, s: *mut f32, h: i32, wd: i32, rs: i32, ss: i32, by: i32, bx: i32, stream: *mut c_void);
        pub fn launch_fp8_matmul_f16(i: *const u16, w: *const u8, ws: *const f32, o: *mut u16, m: i32, n: i32, k: i32, srs: i32, by: i32, bx: i32, stream: *mut c_void);
        pub fn launch_fp8_matmul_bf16(i: *const u16, w: *const u8, ws: *const f32, o: *mut u16, m: i32, n: i32, k: i32, srs: i32, by: i32, bx: i32, stream: *mut c_void);
        pub fn launch_fp8_indexed_moe_gemm_f16(i: *const u16, w: *const u8, ws: *const f32, idx: *const u32, o: *mut u16, nt: i32, topk: i32, ne: i32, n: i32, k: i32, srs: i32, by: i32, bx: i32, has_topk: bool, stream: *mut c_void);
        pub fn launch_fp8_indexed_moe_gemm_bf16(i: *const u16, w: *const u8, ws: *const f32, idx: *const u32, o: *mut u16, nt: i32, topk: i32, ne: i32, n: i32, k: i32, srs: i32, by: i32, bx: i32, has_topk: bool, stream: *mut c_void);
    }
}

pub struct G {
    pub ctx: Arc<CudaContext>,
    pub streams: Vec<Arc<CudaStream>>,
    pub rng: Rng,
    pub t: Tally,
    pub calls: usize,
    pub inert: usize,
    pub family_live: usize,
}

/// Element types: 0 f32, 1 f16, 2 bf16.
pub const TN: [&str; 3] = ["f32", "f16", "bf16"];
pub const TS: [usize; 3] = [4, 2, 2];

impl G {
    pub fn up(&self, b: &[u8]) -> DeviceBuffer<u8> {
        DeviceBuffer::from_host(&self.streams[0], if b.is_empty() { &[0u8][..] } else { b }).unwrap()
    }
    pub fn stream(&self, i: usize) -> *mut c_void {
        if i == 0 { std::ptr::null_mut() } else { self.streams[i].cu_stream() as *mut c_void }
    }

    /// Runs `f(rust, ptrs)` for the C (`rust == false`) and the Rust launcher, each on its own
    /// copies of `bufs`, then compares the buffers listed in `outs`.
    pub fn pair(&mut self, label: &str, bufs: &[Vec<u8>], outs: &[usize], f: &dyn Fn(bool, &[u64])) {
        let a: Vec<DeviceBuffer<u8>> = bufs.iter().map(|b| self.up(b)).collect();
        let b: Vec<DeviceBuffer<u8>> = bufs.iter().map(|b| self.up(b)).collect();
        self.ctx.synchronize().unwrap();
        let pa: Vec<u64> = a.iter().map(|x| x.cu_deviceptr()).collect();
        let pb: Vec<u64> = b.iter().map(|x| x.cu_deviceptr()).collect();
        f(false, &pa);
        self.ctx.synchronize().unwrap_or_else(|e| panic!("{label}: C launcher: {e:?}"));
        // GATE_REF_TWICE=1 runs the C launcher on both sides (reference determinism probe).
        f(std::env::var("GATE_REF_TWICE").is_err(), &pb);
        self.ctx.synchronize().unwrap_or_else(|e| panic!("{label}: Rust launcher: {e:?}"));
        self.calls += 2;
        let mut d = kdiff::Diff { bytes: 0, differing: 0, first: None };
        let mut changed = false;
        for &o in outs {
            let x = a[o].to_host_vec(&self.streams[0]).unwrap();
            let y = b[o].to_host_vec(&self.streams[0]).unwrap();
            let n = bufs[o].len();
            d.bytes += n;
            for i in 0..n {
                if x[i] != y[i] {
                    d.differing += 1;
                    if d.first.is_none() {
                        d.first = Some((o, i, x[i], y[i]));
                    }
                }
                changed |= x[i] != bufs[o][i] || y[i] != bufs[o][i];
            }
        }
        if changed {
            self.family_live += 1;
        } else {
            self.inert += 1;
            if self.inert <= 5 {
                println!("  inert: {label}");
            }
        }
        self.t.record(label, &d);
    }

    /// `n` elements of type `t`: f32 from `Rng::f32s`; 16-bit types mostly rounded f32 values in
    /// [-8, 8) with 1 in 8 raw bit patterns (nan / inf / denormal / huge).
    pub fn vals(&mut self, t: usize, n: usize) -> Vec<u8> {
        match t {
            0 => as_bytes(&self.rng.f32s(n)),
            _ => {
                let v = self.rng.f32s(n);
                let h: Vec<u16> = v
                    .iter()
                    .map(|&x| {
                        let r = self.rng.next();
                        if r % 8 == 0 { (r >> 8) as u16 } else if t == 2 { (x.to_bits() >> 16) as u16 } else { f32_to_f16(x) }
                    })
                    .collect();
                as_bytes(&h)
            }
        }
    }
    /// `n` ordinary elements of type `t` in [-2, 2) (no specials): GEMM data whose results are
    /// not swamped by NaN / inf, so accumulation-order differences show.
    pub fn clean(&mut self, t: usize, n: usize) -> Vec<u8> {
        let v: Vec<f32> = (0..n).map(|_| ((self.rng.next() >> 11) as f64 / (1u64 << 53) as f64 * 4.0 - 2.0) as f32).collect();
        match t {
            0 => as_bytes(&v),
            1 => as_bytes(&v.iter().map(|&x| f32_to_f16(x)).collect::<Vec<u16>>()),
            _ => as_bytes(&v.iter().map(|&x| (x.to_bits() >> 16) as u16).collect::<Vec<u16>>()),
        }
    }
    /// Positive / negative scales 2^-8..2^2 without specials.
    pub fn clean_scales(&mut self, t: usize, n: usize) -> Vec<u8> {
        let v: Vec<f32> = (0..n)
            .map(|_| {
                let r = self.rng.next();
                let m = 1.0 + ((r >> 12) & 0xfffff) as f32 / 1048576.0;
                let e = ((r >> 40) % 11) as i32 - 8;
                let s = if (r >> 50) & 1 == 1 { -1.0 } else { 1.0 };
                s * m * 2f32.powi(e)
            })
            .collect();
        match t {
            0 => as_bytes(&v),
            1 => as_bytes(&v.iter().map(|&x| f32_to_f16(x)).collect::<Vec<u16>>()),
            _ => as_bytes(&v.iter().map(|&x| (x.to_bits() >> 16) as u16).collect::<Vec<u16>>()),
        }
    }
    /// Scale-like values: positive and negative magnitudes 2^-8..2^2, with 1 in 16 specials.
    pub fn scales(&mut self, t: usize, n: usize) -> Vec<u8> {
        let v: Vec<f32> = (0..n)
            .map(|i| {
                let r = self.rng.next();
                if r % 16 == 0 {
                    [0.0, -0.0, f32::INFINITY, f32::NAN, 1e-30, 6e-8, -1e-40, 65504.0][i % 8]
                } else {
                    let m = 1.0 + ((r >> 12) & 0xfffff) as f32 / 1048576.0;
                    let e = ((r >> 40) % 11) as i32 - 8;
                    let s = if (r >> 50) & 1 == 1 { -1.0 } else { 1.0 };
                    s * m * 2f32.powi(e)
                }
            })
            .collect();
        match t {
            0 => as_bytes(&v),
            1 => as_bytes(&v.iter().map(|&x| f32_to_f16(x)).collect::<Vec<u16>>()),
            _ => as_bytes(&v.iter().map(|&x| (x.to_bits() >> 16) as u16).collect::<Vec<u16>>()),
        }
    }
    pub fn family(&mut self, name: &str, start_calls: usize, start_fail: usize, start_live: usize) -> bool {
        let live = self.family_live - start_live;
        let fails = self.t.failures.len() - start_fail;
        println!("  {name}: {} launcher calls, {} failing, {} live", self.calls - start_calls, fails, live);
        fails == 0 && live > 0
    }
}

/// f32 -> f16 bits, round to nearest even (test data only).
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    if x.is_nan() {
        return sign | 0x7e00;
    }
    let a = x.abs();
    if a >= 65520.0 {
        return sign | 0x7c00;
    }
    if a < 6.103515625e-05 {
        // subnormal: multiples of 2^-24
        let q = (a as f64 / 5.960464477539063e-08).round_ties_even() as u16;
        return sign | q;
    }
    let e = ((b >> 23) & 0xff) as i32 - 127;
    let m = b & 0x7fffff;
    let mut h = (((e + 15) as u32) << 10) | (m >> 13);
    let rem = m & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    sign | h as u16
}

/// Cancellation data for dot products: in each of `rows` rows of `row_bytes` bytes, copy the first
/// half onto the second half, negating each element of `neg` bytes (sign bit flip) if given. With
/// x mirrored-negated and w mirrored, every exact dot product is 0 and the result is pure
/// accumulation rounding residue, which exposes any change of operation order or fusion.
pub fn mirror(b: &mut [u8], rows: usize, row_bytes: usize, neg: Option<usize>) {
    let h = row_bytes / 2;
    for r in 0..rows {
        let base = r * row_bytes;
        for i in 0..h {
            b[base + h + i] = b[base + i];
        }
        if let Some(es) = neg {
            let mut i = h + es - 1;
            while i < row_bytes {
                b[base + i] ^= 0x80;
                i += es;
            }
        }
    }
}

// ================================================================================ AFQ

fn afq(g: &mut G) -> bool {
    let (c0, f0, l0) = (g.calls, g.t.failures.len(), g.family_live);
    for t in 0..3 {
        let es = TS[t];
        for &bits in &[2i32, 3, 4, 6, 8] {
            for (gi, &gs) in [32i32, 64, 128].iter().enumerate() {
                // ---- dequantize: (rows, cols)
                for &(rows, cols) in &[(3, gs * 2), (17, gs * 5), (1, gs), (64, 256), (5, gs * 3 + 32)] {
                    let bytes_row = if bits == 3 || bits == 6 { ((cols * bits + 7) / 8) as usize } else { (cols * bits / 32) as usize * 4 };
                    let w = g.rng.bytes(rows as usize * bytes_row + 8);
                    let ng = (rows * cols / gs) as usize + 2;
                    let (s, b) = (g.scales(t, ng), g.vals(t, ng));
                    let out = g.rng.bytes((rows * cols) as usize * es + 6);
                    let (cf, of) = afq_dequant_pair(t, bits, gi);
                    g.pair(&format!("afq_dequantize_{bits}bit_gs{gs}_{} rows={rows} cols={cols}", TN[t]), &[w, s, b, out], &[3], &|r, p| unsafe {
                        (if r { of } else { cf })(p[0] as _, p[1] as _, p[2] as _, p[3] as _, rows, cols)
                    });
                }
                // ---- qmv: (M, N, K)
                for (si, &(m, n, k)) in [(1, 7, gs * 2), (3, 33, gs * 3), (2, 64, 256), (1, 1, gs), (5, 9, 512)].iter().enumerate() {
                    let bytes_row = if bits == 3 || bits == 6 { ((k * bits + 7) / 8) as usize } else { (k * bits / 32) as usize * 4 };
                    let w = g.rng.bytes(n as usize * bytes_row + 8);
                    let ng = (n * k / gs) as usize + 2;
                    let sp = si == 1;
                    let (s, b) = if sp { (g.scales(t, ng), g.vals(t, ng)) } else { (g.clean_scales(t, ng), g.clean(t, ng)) };
                    let x = if sp { g.vals(t, (m * k) as usize) } else { g.clean(t, (m * k) as usize) };
                    let y = g.rng.bytes((m * n) as usize * es + 6);
                    let (cf, of) = afq_qmv_pair(t, bits, gi);
                    g.pair(&format!("afq_qmv_{bits}bit_gs{gs}_{} m={m} n={n} k={k}", TN[t]), &[x, w, s, b, y], &[4], &|r, p| unsafe {
                        (if r { of } else { cf })(p[0] as _, p[1] as _, p[2] as _, p[3] as _, p[4] as _, m, n, k)
                    });
                }
                // ---- qmv / qmm cancellation (K = 4 groups, second half mirrors the first)
                {
                    let (m, n, k) = (3, 20, gs * 4);
                    let bytes_row = if bits == 3 || bits == 6 { ((k * bits + 7) / 8) as usize } else { (k * bits / 32) as usize * 4 };
                    let mut w = g.rng.bytes(n as usize * bytes_row);
                    mirror(&mut w, n as usize, bytes_row, None);
                    let (mut s, mut b) = (g.clean_scales(t, (n * 4) as usize), g.clean(t, (n * 4) as usize));
                    mirror(&mut s, n as usize, 4 * es, None);
                    mirror(&mut b, n as usize, 4 * es, None);
                    let mut x = g.clean(t, (m * k) as usize);
                    mirror(&mut x, m as usize, k as usize * es, Some(es));
                    let y = g.rng.bytes((m * n) as usize * es + 6);
                    let (cf, of) = afq_qmv_pair(t, bits, gi);
                    g.pair(&format!("afq_qmv_{bits}bit_gs{gs}_{} cancel", TN[t]), &[x.clone(), w.clone(), s.clone(), b.clone(), y.clone()], &[4], &|r, p| unsafe {
                        (if r { of } else { cf })(p[0] as _, p[1] as _, p[2] as _, p[3] as _, p[4] as _, m, n, k)
                    });
                    if bits != 3 && bits != 6 {
                        let (cf, of) = afq_qmm_pair(t, bits, gi);
                        g.pair(&format!("afq_qmm_{bits}bit_gs{gs}_{} cancel", TN[t]), &[x, w, s, b, y], &[4], &|r, p| unsafe {
                            (if r { of } else { cf })(p[0] as _, p[1] as _, p[2] as _, p[3] as _, p[4] as _, m, n, k)
                        });
                    }
                }
                if bits == 3 || bits == 6 {
                    continue;
                }
                // ---- quantize: (rows, cols), inputs with and without specials
                for (ci, &(rows, cols)) in [(3, gs * 2), (9, gs * 5), (1, gs), (40, 256), (7, gs * 3)].iter().enumerate() {
                    let wv = if ci % 2 == 0 {
                        g.vals(t, (rows * cols) as usize)
                    } else {
                        // ordinary weights: no specials
                        let v: Vec<f32> = (0..rows * cols).map(|_| ((g.rng.next() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 * 0.1).collect();
                        match t {
                            0 => as_bytes(&v),
                            1 => as_bytes(&v.iter().map(|&x| f32_to_f16(x)).collect::<Vec<u16>>()),
                            _ => as_bytes(&v.iter().map(|&x| (x.to_bits() >> 16) as u16).collect::<Vec<u16>>()),
                        }
                    };
                    let wq = g.rng.bytes((rows * cols * bits / 32) as usize * 4 + 8);
                    let ng = (rows * cols / gs) as usize;
                    let (s, b) = (g.rng.bytes(ng * es + 4), g.rng.bytes(ng * es + 4));
                    let (cf, of) = afq_quant_pair(t, bits, gi);
                    g.pair(&format!("afq_quantize_{bits}bit_gs{gs}_{} rows={rows} cols={cols}", TN[t]), &[wv, wq, s, b], &[1, 2, 3], &|r, p| unsafe {
                        (if r { of } else { cf })(p[0] as _, p[1] as _, p[2] as _, p[3] as _, rows, cols)
                    });
                }
                // ---- quantize, targeted: a group with min 0 and max 2^k (exact inv_scale) and a value v
                // with (v - 0) * inv == 0.5 - 2^-25, the one input where roundf's add.rz differs from rn.
                if t == 0 {
                    let (rows, cols) = (2, gs * 2);
                    let range = match bits { 2 => 4.0f32, 4 => 16.0, _ => 256.0 };
                    let inv = ((1 << bits) - 1) as f32 / range;
                    let target = f32::from_bits(0.5f32.to_bits() - 1);
                    let mut v = target / inv;
                    for _ in 0..64 {
                        if v * inv == target { break; }
                        v = f32::from_bits(if v * inv < target { v.to_bits() + 1 } else { v.to_bits() - 1 });
                    }
                    assert!(v * inv == target, "no tie input for bits {bits}");
                    let mut wv: Vec<f32> = (0..rows * cols).map(|i| (i % 7) as f32 * range / 8.0).collect();
                    for grp in 0..(rows * cols / gs) as usize {
                        wv[grp * gs as usize] = 0.0;
                        wv[grp * gs as usize + 1] = range;
                        wv[grp * gs as usize + 2] = v;
                        wv[grp * gs as usize + 5] = v;
                    }
                    let wq = g.rng.bytes((rows * cols * bits / 32) as usize * 4 + 8);
                    let ng = (rows * cols / gs) as usize;
                    let (s, b) = (g.rng.bytes(ng * es + 4), g.rng.bytes(ng * es + 4));
                    let (cf, of) = afq_quant_pair(t, bits, gi);
                    g.pair(&format!("afq_quantize_{bits}bit_gs{gs}_f32 round-tie"), &[as_bytes(&wv), wq, s, b], &[1, 2, 3], &|r, p| unsafe {
                        (if r { of } else { cf })(p[0] as _, p[1] as _, p[2] as _, p[3] as _, rows, cols)
                    });
                }
                // ---- qmm: (M, N, K) incl. partial tiles
                for (si, &(m, n, k)) in [(33, 40, gs * 2), (64, 32, 256), (5, 70, gs * 3), (1, 1, gs), (100, 17, 512)].iter().enumerate() {
                    let w = g.rng.bytes((n * k * bits / 32) as usize * 4 + 8);
                    let ng = (n * k / gs) as usize + 2;
                    let sp = si == 2;
                    let (s, b) = if sp { (g.scales(t, ng), g.vals(t, ng)) } else { (g.clean_scales(t, ng), g.clean(t, ng)) };
                    let x = if sp { g.vals(t, (m * k) as usize) } else { g.clean(t, (m * k) as usize) };
                    let y = g.rng.bytes((m * n) as usize * es + 6);
                    let (cf, of) = afq_qmm_pair(t, bits, gi);
                    g.pair(&format!("afq_qmm_{bits}bit_gs{gs}_{} m={m} n={n} k={k}", TN[t]), &[x, w, s, b, y], &[4], &|r, p| unsafe {
                        (if r { of } else { cf })(p[0] as _, p[1] as _, p[2] as _, p[3] as _, p[4] as _, m, n, k)
                    });
                }
            }
        }
    }
    g.family("afq", c0, f0, l0)
}


// ================================================================================ FP8

/// Transmute a launcher of any pointer types to `F` (identical C ABI).
macro_rules! fnp {
    ($f:path, $F:ty) => { unsafe { std::mem::transmute::<*const (), $F>($f as *const ()) } };
}
type Conv = unsafe extern "C" fn(*const u8, *mut u8, usize, *mut c_void);
type Vec3 = unsafe extern "C" fn(*const u8, *mut u8, *mut u8, usize, *mut c_void);
type Blk = unsafe extern "C" fn(*const u8, *mut u8, *mut u8, i32, i32, i32, i32, i32, i32, *mut c_void);
type Mm = unsafe extern "C" fn(*const u8, *const u8, *const u8, *mut u8, i32, i32, i32, i32, i32, i32, *mut c_void);
type Moe = unsafe extern "C" fn(*const u8, *const u8, *const u8, *const u8, *mut u8, i32, i32, i32, i32, i32, i32, i32, i32, bool, *mut c_void);

/// Every e4m3 byte value 0..=255 in order, then random bytes.
fn fp8_bytes(g: &mut G, n: usize) -> Vec<u8> {
    (0..n).map(|i| if i < 256 { i as u8 } else { g.rng.next() as u8 }).collect()
}
/// Random e4m3 bytes without NaN (0x7f / 0xff).
fn fp8_finite(g: &mut G, n: usize) -> Vec<u8> {
    (0..n).map(|_| { let b = g.rng.next() as u8; if b & 0x7f == 0x7f { b & 0xf0 } else { b } }).collect()
}
/// Values for fp8 quantization: mostly |x| <= ~600 (some beyond +-448), specials.
fn fp8_src(g: &mut G, t: usize, n: usize, big: bool) -> Vec<u8> {
    let v: Vec<f32> = (0..n)
        .map(|i| {
            let r = g.rng.next();
            match r % 16 {
                0 => [0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 1e-40, -3e-39, 448.0, -448.0, 464.0, 480.0, 449.0, 1e-7, 2e-3, 6e-8, 65504.0][i % 16],
                1 if t == 0 => f32::from_bits((r >> 16) as u32),
                _ => {
                    let x = ((r >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32;
                    if big { x * 700.0 } else { x * 3.0 }
                }
            }
        })
        .collect();
    match t {
        0 => as_bytes(&v),
        1 => as_bytes(&v.iter().map(|&x| f32_to_f16(x)).collect::<Vec<u16>>()),
        _ => as_bytes(&v.iter().map(|&x| (x.to_bits() >> 16) as u16).collect::<Vec<u16>>()),
    }
}

fn fp8_scalar(g: &mut G) -> bool {
    let (c0, f0, l0) = (g.calls, g.t.failures.len(), g.family_live);
    let to: [(Conv, Conv); 3] = [
        (fnp!(cref::launch_fp8_to_f32_kernel, Conv), fnp!(ox::launch_fp8_to_f32_kernel, Conv)),
        (fnp!(cref::launch_fp8_to_f16_kernel, Conv), fnp!(ox::launch_fp8_to_f16_kernel, Conv)),
        (fnp!(cref::launch_fp8_to_bf16_kernel, Conv), fnp!(ox::launch_fp8_to_bf16_kernel, Conv)),
    ];
    let from: [(Conv, Conv); 3] = [
        (fnp!(cref::launch_f32_to_fp8_kernel, Conv), fnp!(ox::launch_f32_to_fp8_kernel, Conv)),
        (fnp!(cref::launch_f16_to_fp8_kernel, Conv), fnp!(ox::launch_f16_to_fp8_kernel, Conv)),
        (fnp!(cref::launch_bf16_to_fp8_kernel, Conv), fnp!(ox::launch_bf16_to_fp8_kernel, Conv)),
    ];
    for t in 0..3 {
        for (si, &n) in [256usize, 1, 255, 257, 1000, 4096 + 33].iter().enumerate() {
            let s = si % 2;
            let w = fp8_bytes(g, n);
            let out = g.rng.bytes(n * TS[t] + 8);
            let (cf, of) = to[t];
            let st = g.stream(s);
            g.pair(&format!("launch_fp8_to_{}_kernel n={n}", TN[t]), &[w, out], &[1], &|r, p| unsafe {
                (if r { of } else { cf })(p[0] as _, p[1] as _, n, st)
            });
            for big in [false, true] {
                let x = fp8_src(g, t, n, big);
                let out = g.rng.bytes(n + 8);
                let (cf, of) = from[t];
                g.pair(&format!("launch_{}_to_fp8_kernel n={n} big={big}", TN[t]), &[x, out], &[1], &|r, p| unsafe {
                    (if r { of } else { cf })(p[0] as _, p[1] as _, n, st)
                });
            }
        }
        // all 65536 16-bit patterns through the 16-bit -> fp8 path
        if t > 0 {
            let x: Vec<u16> = (0..=65535u16).collect();
            let out = g.rng.bytes(65536 + 8);
            let (cf, of) = from[t];
            let st = g.stream(1);
            g.pair(&format!("launch_{}_to_fp8_kernel all-patterns", TN[t]), &[as_bytes(&x), out], &[1], &|r, p| unsafe {
                (if r { of } else { cf })(p[0] as _, p[1] as _, 65536, st)
            });
        }
    }
    g.family("scalar_fp8", c0, f0, l0)
}

fn fp8_vector(g: &mut G) -> bool {
    let (c0, f0, l0) = (g.calls, g.t.failures.len(), g.family_live);
    let dq: [(Vec3, Vec3); 3] = [
        (fnp!(cref::launch_dequant_fp8_vector_kernel_f32, Vec3), fnp!(ox::launch_dequant_fp8_vector_kernel_f32, Vec3)),
        (fnp!(cref::launch_dequant_fp8_vector_kernel_f16, Vec3), fnp!(ox::launch_dequant_fp8_vector_kernel_f16, Vec3)),
        (fnp!(cref::launch_dequant_fp8_vector_kernel_bf16, Vec3), fnp!(ox::launch_dequant_fp8_vector_kernel_bf16, Vec3)),
    ];
    let q: [(Vec3, Vec3); 3] = [
        (fnp!(cref::launch_quant_fp8_vector_kernel_f32, Vec3), fnp!(ox::launch_quant_fp8_vector_kernel_f32, Vec3)),
        (fnp!(cref::launch_quant_fp8_vector_kernel_f16, Vec3), fnp!(ox::launch_quant_fp8_vector_kernel_f16, Vec3)),
        (fnp!(cref::launch_quant_fp8_vector_kernel_bf16, Vec3), fnp!(ox::launch_quant_fp8_vector_kernel_bf16, Vec3)),
    ];
    for t in 0..3 {
        for (si, &n) in [128usize, 1, 127, 129, 1000, 128 * 40 + 77].iter().enumerate() {
            let st = g.stream(si % 2);
            let nv = n.div_ceil(128);
            let w = fp8_bytes(g, n);
            let sc = g.scales(0, nv + 1);
            let out = g.rng.bytes(n * TS[t] + 8);
            let (cf, of) = dq[t];
            g.pair(&format!("launch_dequant_fp8_vector_kernel_{} n={n}", TN[t]), &[w, sc, out], &[2], &|r, p| unsafe {
                (if r { of } else { cf })(p[0] as _, p[1] as _, p[2] as _, n, st)
            });
            for mode in 0..3 {
                let mut x = fp8_src(g, t, n, mode == 1);
                if mode == 2 {
                    // tiny vectors: absmax / 448 below 1e-12 (scale clamp) and denormal inputs
                    let v: Vec<f32> = (0..n).map(|i| [1e-10, -3e-40, 2e-38, 0.0, -1e-11, 4e-10][i % 6]).collect();
                    x = match t {
                        0 => as_bytes(&v),
                        1 => as_bytes(&v.iter().map(|&x| f32_to_f16(x * 1e6)).collect::<Vec<u16>>()),
                        _ => as_bytes(&v.iter().map(|&x| (x.to_bits() >> 16) as u16).collect::<Vec<u16>>()),
                    };
                }
                let wq = g.rng.bytes(n + 8);
                let sc = g.rng.bytes(nv * 4 + 8);
                let (cf, of) = q[t];
                g.pair(&format!("launch_quant_fp8_vector_kernel_{} n={n} mode={mode}", TN[t]), &[x, wq, sc], &[1, 2], &|r, p| unsafe {
                    (if r { of } else { cf })(p[0] as _, p[1] as _, p[2] as _, n, st)
                });
            }
        }
    }
    g.family("vector_fp8", c0, f0, l0)
}

fn fp8_blockwise(g: &mut G) -> bool {
    let (c0, f0, l0) = (g.calls, g.t.failures.len(), g.family_live);
    let dq: [(Blk, Blk); 3] = [
        (fnp!(cref::launch_dequant_fp8_blockwise_kernel_f32, Blk), fnp!(ox::launch_dequant_fp8_blockwise_kernel_f32, Blk)),
        (fnp!(cref::launch_dequant_fp8_blockwise_kernel_f16, Blk), fnp!(ox::launch_dequant_fp8_blockwise_kernel_f16, Blk)),
        (fnp!(cref::launch_dequant_fp8_blockwise_kernel_bf16, Blk), fnp!(ox::launch_dequant_fp8_blockwise_kernel_bf16, Blk)),
    ];
    let q: [(Blk, Blk); 3] = [
        (fnp!(cref::launch_quant_fp8_blockwise_kernel_f32, Blk), fnp!(ox::launch_quant_fp8_blockwise_kernel_f32, Blk)),
        (fnp!(cref::launch_quant_fp8_blockwise_kernel_f16, Blk), fnp!(ox::launch_quant_fp8_blockwise_kernel_f16, Blk)),
        (fnp!(cref::launch_quant_fp8_blockwise_kernel_bf16, Blk), fnp!(ox::launch_quant_fp8_blockwise_kernel_bf16, Blk)),
    ];
    for t in 0..3 {
        // (height, width, row stride, block y, block x)
        for (si, &(h, w, rs, by, bx)) in [(256, 256, 256, 128, 128), (200, 300, 300, 128, 128), (70, 50, 64, 16, 16), (33, 97, 101, 64, 32), (5, 7, 7, 128, 128), (130, 64, 64, 48, 20)].iter().enumerate() {
            let st = g.stream(si % 2);
            let (gy, gx) = ((h + by - 1) / by, (w + bx - 1) / bx);
            let ss = gx + (si as i32 % 3);
            let n = (h * rs) as usize;
            let wt = fp8_bytes(g, n + 8);
            let sc = g.scales(0, (gy * ss) as usize + 2);
            let out = g.rng.bytes(n * TS[t] + 8);
            let (cf, of) = dq[t];
            g.pair(&format!("launch_dequant_fp8_blockwise_kernel_{} h={h} w={w} rs={rs} b={by}x{bx}", TN[t]), &[wt, sc, out], &[2], &|r, p| unsafe {
                (if r { of } else { cf })(p[0] as _, p[1] as _, p[2] as _, h, w, rs, ss, by, bx, st)
            });
            for big in [false, true] {
                let x = fp8_src(g, t, n + 4, big);
                let wq = g.rng.bytes(n + 8);
                let sc = g.rng.bytes((gy * ss) as usize * 4 + 8);
                let (cf, of) = q[t];
                g.pair(&format!("launch_quant_fp8_blockwise_kernel_{} h={h} w={w} rs={rs} b={by}x{bx} big={big}", TN[t]), &[x, wq, sc], &[1, 2], &|r, p| unsafe {
                    (if r { of } else { cf })(p[0] as _, p[1] as _, p[2] as _, h, w, rs, ss, by, bx, st)
                });
            }
        }
    }
    g.family("blockwise_fp8", c0, f0, l0)
}

fn fp8_gemm(g: &mut G) -> bool {
    let (c0, f0, l0) = (g.calls, g.t.failures.len(), g.family_live);
    let mm: [(Mm, Mm); 2] = [
        (fnp!(cref::launch_fp8_matmul_f16, Mm), fnp!(ox::launch_fp8_matmul_f16, Mm)),
        (fnp!(cref::launch_fp8_matmul_bf16, Mm), fnp!(ox::launch_fp8_matmul_bf16, Mm)),
    ];
    let moe: [(Moe, Moe); 2] = [
        (fnp!(cref::launch_fp8_indexed_moe_gemm_f16, Moe), fnp!(ox::launch_fp8_indexed_moe_gemm_f16, Moe)),
        (fnp!(cref::launch_fp8_indexed_moe_gemm_bf16, Moe), fnp!(ox::launch_fp8_indexed_moe_gemm_bf16, Moe)),
    ];
    for ti in 0..2 {
        let t = ti + 1;
        for (si, &(m, n, k, by, bx)) in [(1, 128, 256, 128, 128), (7, 200, 300, 128, 128), (33, 65, 96, 32, 32), (64, 64, 64, 64, 16), (3, 5, 7, 128, 128)].iter().enumerate() {
            let st = g.stream(si % 2);
            let srs = (k + bx - 1) / bx + (si as i32 % 2);
            let sp = si == 2;
            let x = if sp { g.vals(t, (m * k) as usize) } else { g.clean(t, (m * k) as usize) };
            let w = if sp { fp8_bytes(g, (n * k) as usize) } else { fp8_finite(g, (n * k) as usize) };
            let nsc = (((n + by - 1) / by) * srs) as usize + 1;
            let sc = if sp { g.scales(0, nsc) } else { g.clean_scales(0, nsc) };
            let out = g.rng.bytes((m * n) as usize * 2 + 4);
            let (cf, of) = mm[ti];
            g.pair(&format!("launch_fp8_matmul_{} m={m} n={n} k={k} b={by}x{bx}", TN[t]), &[x, w, sc, out], &[3], &|r, p| unsafe {
                (if r { of } else { cf })(p[0] as _, p[1] as _, p[2] as _, p[3] as _, m, n, k, srs, by, bx, st)
            });
        }
        // cancellation: x second half = -first half, weight and scale columns mirrored
        for &(m, n, k, by, bx) in &[(8, 64, 256, 64, 128), (40, 33, 512, 128, 128)] {
            let srs = k / bx;
            let mut x = g.clean(t, (m * k) as usize);
            mirror(&mut x, m as usize, k as usize * 2, Some(2));
            let mut w = fp8_finite(g, (n * k) as usize);
            mirror(&mut w, n as usize, k as usize, None);
            let nsr = ((n + by - 1) / by) as usize;
            let mut sc = g.clean_scales(0, nsr * srs as usize);
            mirror(&mut sc, nsr, srs as usize * 4, None);
            let out = g.rng.bytes((m * n) as usize * 2 + 4);
            let (cf, of) = mm[ti];
            let st = g.stream(1);
            g.pair(&format!("launch_fp8_matmul_{} m={m} n={n} k={k} cancel", TN[t]), &[x, w, sc, out], &[3], &|r, p| unsafe {
                (if r { of } else { cf })(p[0] as _, p[1] as _, p[2] as _, p[3] as _, m, n, k, srs, by, bx, st)
            });
        }
        for &(nt, topk, ne, n, k, by, bx) in &[(4, 2, 3, 64, 512, 64, 128), (3, 1, 2, 16, 384, 16, 64)] {
            let srs = k / bx;
            let mut x = g.clean(t, (nt * k) as usize);
            mirror(&mut x, nt as usize, k as usize * 2, Some(2));
            let mut w = fp8_finite(g, (ne * n * k) as usize);
            mirror(&mut w, (ne * n) as usize, k as usize, None);
            let nsr = (ne * ((n + by - 1) / by)) as usize;
            let mut sc = g.clean_scales(0, nsr * srs as usize);
            mirror(&mut sc, nsr, srs as usize * 4, None);
            let idx: Vec<u32> = (0..nt * topk).map(|_| (g.rng.next() >> 8) as u32 % ne as u32).collect();
            let out = g.rng.bytes((nt * topk * n) as usize * 2 + 4);
            let (cf, of) = moe[ti];
            let st = g.stream(1);
            g.pair(&format!("launch_fp8_indexed_moe_gemm_{} nt={nt} k={k} cancel", TN[t]), &[x, w, sc, as_bytes(&idx), out], &[4], &|r, p| unsafe {
                (if r { of } else { cf })(p[0] as _, p[1] as _, p[2] as _, p[3] as _, p[4] as _, nt, topk, ne, n, k, srs, by, bx, false, st)
            });
        }
        // (tokens, topk, experts, N, K, block y, block x)
        for (si, &(nt, topk, ne, n, k, by, bx)) in [(4, 2, 4, 128, 256, 128, 128), (3, 3, 5, 96, 200, 64, 64), (1, 1, 2, 33, 128, 128, 128), (5, 2, 3, 40, 68, 16, 32), (2, 4, 8, 256, 512, 128, 128)].iter().enumerate() {
            let st = g.stream(si % 2);
            let srs = (k + bx - 1) / bx;
            for has_topk in [false, true] {
                let rows = if has_topk { nt * topk } else { nt };
                let sp = si == 1 && has_topk;
                let x = if sp { g.vals(t, (rows * k) as usize + 4) } else { g.clean(t, (rows * k) as usize + 4) };
                let w = if sp { fp8_bytes(g, (ne * n * k) as usize + 4) } else { fp8_finite(g, (ne * n * k) as usize + 4) };
                let nsc = (ne * ((n + by - 1) / by) * srs) as usize + 1;
                let sc = if sp { g.scales(0, nsc) } else { g.clean_scales(0, nsc) };
                // expert ids, some out of range (skipped rows)
                let idx: Vec<u32> = (0..nt * topk).map(|_| { let r = g.rng.next(); if r % 7 == 0 { ne as u32 + (r >> 20) as u32 % 3 } else { (r >> 8) as u32 % ne as u32 } }).collect();
                let out = g.rng.bytes((nt * topk * n) as usize * 2 + 4);
                let (cf, of) = moe[ti];
                g.pair(&format!("launch_fp8_indexed_moe_gemm_{} nt={nt} topk={topk} ne={ne} n={n} k={k} b={by}x{bx} topk_dim={has_topk}", TN[t]), &[x, w, sc, as_bytes(&idx), out], &[4], &|r, p| unsafe {
                    (if r { of } else { cf })(p[0] as _, p[1] as _, p[2] as _, p[3] as _, p[4] as _, nt, topk, ne, n, k, srs, by, bx, has_topk, st)
                });
            }
        }
    }
    g.family("blockwise_fp8_gemm", c0, f0, l0)
}

// ================================================================================ GPTQ

/// GPTQ test tensors for (K, N, groups, bit): q_weight [K*bit/32, N], qzeros [groups, N*bit/32],
/// scales [groups, N] f16, a random permutation of 0..K and a sorted g_idx.
struct Gq {
    qw: Vec<u8>,
    qz: Vec<u8>,
    sc: Vec<u8>,
    perm: Vec<u8>,
    gidx: Vec<u8>,
}
fn gq(g: &mut G, k: i32, n: i32, groups: i32, bit: i32, special: bool) -> Gq {
    let qw = g.rng.bytes((k * bit / 32 * n) as usize * 4 + 16);
    let qz = g.rng.bytes(((groups * n * bit + 31) / 32) as usize * 4 + 16);
    let sc = if special { g.scales(1, (groups * n) as usize + 8) } else { g.clean_scales(1, (groups * n) as usize + 8) };
    let mut perm: Vec<i32> = (0..k).collect();
    for i in (1..k as usize).rev() {
        let j = (g.rng.next() % (i as u64 + 1)) as usize;
        perm.swap(i, j);
    }
    let gs = k / groups;
    let gidx: Vec<i32> = (0..k).map(|i| if special && i % 7 == 3 { (g.rng.next() % groups as u64) as i32 } else { i / gs }).collect();
    Gq { qw, qz, sc, perm: as_bytes(&perm), gidx: as_bytes(&gidx) }
}

/// The exllama / alt GEMMs split K over grid.z blocks of 128 and sum the per-block f16 partials
/// with atomicAdd, so for K > 128 the reference is not deterministic (measured: the C launcher
/// disagrees with itself run to run). To still test blocks with offset_k > 0 bit-exactly, zero
/// every activation outside one K block (chosen by `sel`): the other blocks then add exact zeros
/// and the result is independent of atomic order. `perm` maps kernel k to activation column.
fn split_k_single_block(a: &mut [u8], m: i32, k: i32, sel: usize, perm: Option<&[u8]>) {
    let nz = (k + 127) / 128;
    if nz <= 1 {
        return;
    }
    let keep = (sel as i32 * 7 + 1) % nz;
    let mut live = vec![false; k as usize];
    for i in 0..k {
        if i / 128 == keep {
            let col = match perm {
                Some(p) => i32::from_le_bytes([p[4 * i as usize], p[4 * i as usize + 1], p[4 * i as usize + 2], p[4 * i as usize + 3]]),
                None => i,
            };
            live[col as usize] = true;
        }
    }
    for r in 0..m as usize {
        for c in 0..k as usize {
            if !live[c] {
                a[2 * (r * k as usize + c)] = 0;
                a[2 * (r * k as usize + c) + 1] = 0;
            }
        }
    }
}

/// Activations in f16: ordinary, or with specials.
fn f16s(g: &mut G, n: usize, special: bool) -> Vec<u8> {
    if special { g.vals(1, n) } else { g.clean(1, n) }
}

fn gptq(g: &mut G) -> bool {
    let (c0, f0, l0) = (g.calls, g.t.failures.len(), g.family_live);
    let null = std::ptr::null::<i32>();
    for &bit in &[2i32, 3, 4, 8] {
        // ---- exllama GEMM, one K block (grid.z == 1: no cross-block atomics) and several.
        for (si, &(m_count, n, k, groups)) in [(1, 512, 128, 1), (4, 1024, 128, 4), (8, 516, 96, 3), (3, 128, 128, 2), (5, 2048, 64, 1), (2, 512, 256, 2), (6, 520, 384, 3), (7, 64, 512, 4)].iter().enumerate() {
            let sp = si == 2;
            let t = gq(g, k, n, groups, bit, sp);
            for use_perm in [false, true] {
                let m = m_count * if si % 2 == 0 { 1 } else { 2 };
                let mut a = f16s(g, (m * k) as usize, sp);
                split_k_single_block(&mut a, m, k, si, if use_perm { Some(&t.perm) } else { None });
                let c = g.rng.bytes((m * n) as usize * 2 + 8);
                g.pair(&format!("gemm_half_q_half_cuda_part bit={bit} m={m} m_count={m_count} n={n} k={k} groups={groups} perm={use_perm}"),
                    &[a, t.qw.clone(), t.qz.clone(), t.sc.clone(), t.perm.clone(), c], &[5], &|r, p| unsafe {
                    let pp = if use_perm { p[4] as *const i32 } else { null };
                    if r { ox::gemm_half_q_half_cuda_part(p[0] as _, p[1] as _, p[2] as _, p[3] as _, pp, p[5] as _, m, n, k, m_count, groups, bit) }
                    else { cref::gemm_half_q_half_cuda_part(p[0] as _, p[1] as _, p[2] as _, p[3] as _, pp, p[5] as _, m, n, k, m_count, groups, bit) }
                });
            }
        }
        // ---- cancellation (bits 4 and 8, K = 128): within every 8-element dot product the last four
        // weights repeat the first four and the activations are negated, so each dot product is pure
        // rounding residue and any change of accumulation order shows.
        if bit == 4 || bit == 8 {
            for &(m_count, n, k, groups) in &[(1, 512, 128, 1), (4, 512, 128, 2), (8, 128, 128, 1)] {
                let mut t = gq(g, k, n, groups, bit, false);
                let rows = (k * bit / 32) as usize;
                for r in 0..rows {
                    for c in 0..n as usize {
                        let o = 4 * (r * n as usize + c);
                        let w = u32::from_le_bytes([t.qw[o], t.qw[o + 1], t.qw[o + 2], t.qw[o + 3]]);
                        let nw = if bit == 4 {
                            (w & 0x00ff_00ff) * 0x101
                        } else if r % 2 == 1 {
                            let p = 4 * ((r - 1) * n as usize + c);
                            u32::from_le_bytes([t.qw[p], t.qw[p + 1], t.qw[p + 2], t.qw[p + 3]])
                        } else {
                            w
                        };
                        t.qw[o..o + 4].copy_from_slice(&nw.to_le_bytes());
                    }
                }
                let mut a = g.clean(1, (m_count * k) as usize);
                for i in 0..(m_count * k) as usize {
                    if i % 8 >= 4 {
                        let j = i - 4;
                        a[2 * i] = a[2 * j];
                        a[2 * i + 1] = a[2 * j + 1] ^ 0x80;
                    }
                }
                let c = g.rng.bytes((m_count * n) as usize * 2 + 8);
                let m = m_count;
                g.pair(&format!("gemm_half_q_half_cuda_part bit={bit} m={m} n={n} k={k} cancel"),
                    &[a, t.qw, t.qz, t.sc, c], &[4], &|r, p| unsafe {
                    if r { ox::gemm_half_q_half_cuda_part(p[0] as _, p[1] as _, p[2] as _, p[3] as _, null, p[4] as _, m, n, k, m_count, groups, bit) }
                    else { cref::gemm_half_q_half_cuda_part(p[0] as _, p[1] as _, p[2] as _, p[3] as _, null, p[4] as _, m, n, k, m_count, groups, bit) }
                });
            }
        }
        // ---- reconstruct_exllama
        for (si, &(n, k, groups)) in [(512, 128, 1), (516, 256, 4), (128, 384, 3), (1024, 96, 1), (64, 512, 16)].iter().enumerate() {
            let t = gq(g, k, n, groups, bit, si == 1);
            for use_perm in [false, true] {
                let out = g.rng.bytes((k * n) as usize * 2 + 8);
                g.pair(&format!("reconstruct_exllama bit={bit} n={n} k={k} groups={groups} perm={use_perm}"),
                    &[t.qw.clone(), t.qz.clone(), t.sc.clone(), t.perm.clone(), out], &[4], &|r, p| unsafe {
                    let pp = if use_perm { p[3] as *const i32 } else { null };
                    if r { ox::reconstruct_exllama(p[0] as _, p[1] as _, p[2] as _, pp, p[4] as _, k, n, groups, bit) }
                    else { cref::reconstruct_exllama(p[0] as _, p[1] as _, p[2] as _, pp, p[4] as _, k, n, groups, bit) }
                });
            }
        }
        // ---- reconstruct_gptq (g_idx act-order)
        for (si, &(n, k, groups)) in [(128, 128, 1), (256, 256, 4), (130, 96, 3), (512, 64, 2)].iter().enumerate() {
            let t = gq(g, k, n, groups, bit, si == 1);
            let out = g.rng.bytes((k * n) as usize * 2 + 8);
            g.pair(&format!("reconstruct_gptq bit={bit} n={n} k={k} groups={groups}"), &[t.qw, t.qz, t.sc, t.gidx, out], &[4], &|r, p| unsafe {
                if r { ox::reconstruct_gptq(p[0] as _, p[1] as _, p[2] as _, p[3] as _, p[4] as _, k, n, groups, bit) }
                else { cref::reconstruct_gptq(p[0] as _, p[1] as _, p[2] as _, p[3] as _, p[4] as _, k, n, groups, bit) }
            });
        }
    }
    // ---- alt GEMM (4-bit kernel for every bit but 8)
    for &bit in &[4i32, 8, 2] {
        for (si, &(m, n, k, groups)) in [(1, 128, 128, 1), (8, 256, 128, 4), (5, 384, 128, 2), (11, 128, 128, 1), (3, 256, 256, 2), (2, 128, 512, 4)].iter().enumerate() {
            let sp = si == 2;
            let t = gq(g, k, n, groups, if bit == 2 { 4 } else { bit }, sp);
            let mut a = f16s(g, (m * k) as usize, sp);
            split_k_single_block(&mut a, m, k, si, None);
            let c = g.rng.bytes((m * n) as usize * 2 + 8);
            g.pair(&format!("gemm_half_q_half_alt bit={bit} m={m} n={n} k={k} groups={groups}"), &[a, t.qw, t.qz, t.sc, t.gidx, c], &[5], &|r, p| unsafe {
                if r { ox::gemm_half_q_half_alt(p[0] as _, p[1] as _, p[2] as _, p[3] as _, p[4] as _, p[5] as _, m, n, k, bit) }
                else { cref::gemm_half_q_half_alt(p[0] as _, p[1] as _, p[2] as _, p[3] as _, p[4] as _, p[5] as _, m, n, k, bit) }
            });
        }
    }
    // (No test of pick_gemm_half_q_half_gptq_kernel's NULL pick: the C launcher then calls a NULL
    // host stub and segfaults; m_count == 0 divides by zero. The Rust twin returns instead.)
    let ok = g.family("gptq", c0, f0, l0);
    ok & gptq_kernels(g)
}

/// Kernel-level kdiff for the q_gemm.cubin instances no launcher reaches (shuffle / make_sequential).
fn gptq_kernels(g: &mut G) -> bool {
    use kdiff::{Arg, Harness};
    let root = format!("{}/titan-engine/oxide-kernels", std::env::var("HOME").unwrap());
    let h = Harness::from_files(&format!("{root}/reference/mistralrs-quant/q_gemm.cubin"), &format!("{root}/mistralrs-quant-b/mistralrs_quant_b.ptx"));
    let mut kt = Tally::default();
    for &bit in &[2i32, 3, 4, 8] {
        for &(k, n) in &[(128i32, 64i32), (256, 100), (96, 33)] {
            let rows = k * bit / 32;
            let w = g.rng.bytes((rows * n) as usize * 4);
            let rn = format!("_Z19shuffle_{bit}bit_kernelPjii");
            let d = h.diff_pair(&rn, &format!("gptq_shuffle_{bit}bit"), ((n as u32).div_ceil(32), 1, 1), (32, 1, 1), 0,
                &[Arg::Buf(0), Arg::I32(k), Arg::I32(n)], &[w.clone()], &[0]);
            kt.record(&format!("shuffle_{bit}bit k={k} n={n}"), &d);
            if n % 2 == 1 && bit != 3 {
                continue;
            }
            let mut perm: Vec<i32> = (0..k).collect();
            for i in (1..k as usize).rev() {
                let j = (g.rng.next() % (i as u64 + 1)) as usize;
                perm.swap(i, j);
            }
            let out = g.rng.bytes((rows * n) as usize * 4);
            let rn = format!("_Z27make_sequential_{bit}bit_kernelPKjPjPKii");
            let gx = if bit == 3 { (n as u32).div_ceil(32) } else { ((n / 2) as u32).div_ceil(32) };
            let d = h.diff_pair(&rn, &format!("make_sequential_{bit}bit"), (gx, (k * bit / 32 / if bit == 3 { 3 } else { 1 }) as u32, 1), (32, 1, 1), 0,
                &[Arg::Buf(0), Arg::Buf(1), Arg::Buf(2), Arg::I32(n)], &[w, out, as_bytes(&perm)], &[1]);
            kt.record(&format!("make_sequential_{bit}bit k={k} n={n}"), &d);
        }
    }
    g.calls += kt.launches * 2;
    g.t.bytes += kt.bytes;
    g.t.failures.extend(kt.failures.iter().cloned());
    kt.finish("gptq shuffle / make_sequential kernel-level kdiff vs q_gemm.cubin")
}

// ================================================================================ MXFP4
type MxMm = unsafe extern "C" fn(*const u16, *const u8, *const u8, *const u16, *mut u16, i32, i32, i32, bool, *mut c_void);
type MxMoe = unsafe extern "C" fn(*const u16, *const u8, *const u8, *const u16, *const u32, *mut u16, i32, i32, i32, i32, i32, bool, bool, *mut c_void);

/// E8M0 scales: mostly 2^-9..2^3, specials (0, 1 = denormal after * 0.5, 254, 255 = inf) if `special`.
fn e8m0s(g: &mut G, n: usize, special: bool) -> Vec<u8> {
    (0..n).map(|i| { let r = g.rng.next(); if special && r % 11 == 0 { [0u8, 1, 2, 254, 255, 127][i % 6] } else { 118 + (r >> 8) as u8 % 13 } }).collect()
}
/// Mirror every 32-element MXFP4 block: activations k+16 = -(k), packed weight bytes 8..16 = 0..8.
fn mx_cancel(x: &mut [u8], rows: usize, k: usize, w: &mut [u8], wrows: usize) {
    for r in 0..rows {
        for b in 0..k / 32 {
            for i in 0..16 {
                let (s, d) = (r * k + b * 32 + i, r * k + b * 32 + 16 + i);
                x[2 * d] = x[2 * s];
                x[2 * d + 1] = x[2 * s + 1] ^ 0x80;
            }
        }
    }
    for r in 0..wrows {
        for b in 0..k / 32 {
            for i in 0..8 {
                w[r * k / 2 + b * 16 + 8 + i] = w[r * k / 2 + b * 16 + i];
            }
        }
    }
}

fn mxfp4(g: &mut G) -> bool {
    let (c0, f0, l0) = (g.calls, g.t.failures.len(), g.family_live);
    let mm: [[(MxMm, MxMm); 2]; 2] = [
        [(fnp!(cref::launch_mxfp4_matmul_f16, MxMm), fnp!(ox::launch_mxfp4_matmul_f16, MxMm)), (fnp!(cref::launch_mxfp4_matmul_bf16, MxMm), fnp!(ox::launch_mxfp4_matmul_bf16, MxMm))],
        [(fnp!(cref::launch_mxfp4_matmul_wmma_f16, MxMm), fnp!(ox::launch_mxfp4_matmul_wmma_f16, MxMm)), (fnp!(cref::launch_mxfp4_matmul_wmma_bf16, MxMm), fnp!(ox::launch_mxfp4_matmul_wmma_bf16, MxMm))],
    ];
    let moe: [[(MxMoe, MxMoe); 2]; 3] = [
        [(fnp!(cref::launch_mxfp4_indexed_moe_gemm_f16, MxMoe), fnp!(ox::launch_mxfp4_indexed_moe_gemm_f16, MxMoe)), (fnp!(cref::launch_mxfp4_indexed_moe_gemm_bf16, MxMoe), fnp!(ox::launch_mxfp4_indexed_moe_gemm_bf16, MxMoe))],
        [(fnp!(cref::launch_mxfp4_moe_grouped_gemm_f16, MxMoe), fnp!(ox::launch_mxfp4_moe_grouped_gemm_f16, MxMoe)), (fnp!(cref::launch_mxfp4_moe_grouped_gemm_bf16, MxMoe), fnp!(ox::launch_mxfp4_moe_grouped_gemm_bf16, MxMoe))],
        [(fnp!(cref::launch_mxfp4_moe_grouped_gemm_wmma_f16, MxMoe), fnp!(ox::launch_mxfp4_moe_grouped_gemm_wmma_f16, MxMoe)), (fnp!(cref::launch_mxfp4_moe_grouped_gemm_wmma_bf16, MxMoe), fnp!(ox::launch_mxfp4_moe_grouped_gemm_wmma_bf16, MxMoe))],
    ];
    const MMN: [&str; 2] = ["launch_mxfp4_matmul", "launch_mxfp4_matmul_wmma"];
    const MOEN: [&str; 3] = ["launch_mxfp4_indexed_moe_gemm", "launch_mxfp4_moe_grouped_gemm", "launch_mxfp4_moe_grouped_gemm_wmma"];
    for ti in 0..2 {
        let t = ti + 1;
        for v in 0..2 {
            // (M, N, K): M <= 4 -> vecmat, else tiled / wmma; K up to 257 blocks (vecmat thread loop).
            for (si, &(m, n, k)) in [(1, 64, 128), (3, 130, 256), (4, 7, 8224), (5, 64, 64), (64, 128, 256), (100, 70, 160), (2, 256, 1024), (33, 33, 96)].iter().enumerate() {
                for mode in 0..3 {
                    // mode 0: clean data; 1: specials; 2: cancellation within each 32-block
                    if mode == 1 && si % 3 != 1 { continue; }
                    if mode == 2 && si % 2 == 1 { continue; }
                    let sp = mode == 1;
                    let mut x = if sp { g.vals(t, (m * k) as usize) } else { g.clean(t, (m * k) as usize) };
                    let mut w = g.rng.bytes((n * k / 2) as usize);
                    if mode == 2 { mx_cancel(&mut x, m as usize, k as usize, &mut w, n as usize); }
                    let ws = e8m0s(g, (n * k / 32) as usize, sp);
                    let bias = if sp { g.vals(t, n as usize) } else { g.clean(t, n as usize) };
                    let out = g.rng.bytes((m * n) as usize * 2 + 4);
                    let st = g.stream(si % 2);
                    for (hb, bias_null) in [(false, false), (true, false), (true, true)] {
                        if bias_null && si % 4 != 0 { continue; }
                        let (cf, of) = mm[v][ti];
                        g.pair(&format!("{}_{} m={m} n={n} k={k} mode={mode} has_bias={hb} null_bias={bias_null}", MMN[v], TN[t]),
                            &[x.clone(), w.clone(), ws.clone(), bias.clone(), out.clone()], &[4], &|r, p| unsafe {
                            let b = if bias_null { std::ptr::null() } else { p[3] as *const u16 };
                            (if r { of } else { cf })(p[0] as _, p[1] as _, p[2] as _, b, p[4] as _, m, n, k, hb, st)
                        });
                    }
                }
            }
        }
        // vecmat cross-warp reduction: K = 256 blocks, one per thread. Blocks 128..160 (warp 4)
        // mirror blocks 0..32 (warp 0) negated, every other block is small, so the second-stage
        // shuffle order (4, 2, 1 over the 8 warp partials) decides the rounding.
        for v in 0..2 {
            let (m, n, k) = (2i32, 8i32, 8192i32);
            let mut x = g.clean(t, (m * k) as usize);
            let mut w = g.rng.bytes((n * k / 2) as usize);
            let mut ws = e8m0s(g, (n * k / 32) as usize, false);
            for r in 0..m as usize {
                for b in 0..256usize {
                    for i in 0..32usize {
                        let e = r * k as usize + b * 32 + i;
                        if (128..160).contains(&b) {
                            let s = e - 128 * 32;
                            x[2 * e] = x[2 * s];
                            x[2 * e + 1] = x[2 * s + 1] ^ 0x80;
                        } else if b >= 32 {
                            // small: scale by 2^-8 (exponent field)
                            let h = u16::from_le_bytes([x[2 * e], x[2 * e + 1]]);
                            let h = if t == 1 { if (h & 0x7c00) > 0x2000 { h - 0x2000 } else { h & 0x8000 } } else if (h & 0x7f80) > 0x0400 { h - 0x0400 } else { h & 0x8000 };
                            x[2 * e..2 * e + 2].copy_from_slice(&h.to_le_bytes());
                        }
                    }
                }
            }
            for c in 0..n as usize {
                for b in 128..160usize {
                    for i in 0..16usize {
                        w[c * (k / 2) as usize + b * 16 + i] = w[c * (k / 2) as usize + (b - 128) * 16 + i];
                    }
                    ws[c * (k / 32) as usize + b] = ws[c * (k / 32) as usize + b - 128];
                }
            }
            let out = g.rng.bytes((m * n) as usize * 2);
            let (cf, of) = mm[v][ti];
            g.pair(&format!("{}_{} vecmat cross-warp cancel", MMN[v], TN[t]), &[x, w, ws, vec![0u8; 16], out], &[4], &|r, p| unsafe {
                (if r { of } else { cf })(p[0] as _, p[1] as _, p[2] as _, std::ptr::null(), p[4] as _, m, n, k, false, std::ptr::null_mut())
            });
        }
        for v in 0..3 {
            // (tokens, topk, experts, N, K); N a multiple of 8 for the indexed kernel (a partial
            // 8-column chunk returns warps before the shared-input load: undefined in the C code).
            for (si, &(nt, topk, ne, n, k)) in [(4, 2, 3, 64, 128), (7, 3, 5, 136, 1056), (1, 1, 2, 8, 32), (40, 4, 3, 72, 96), (70, 2, 2, 64, 64)].iter().enumerate() {
                for has_topk in [false, true] {
                    for mode in 0..3 {
                        if mode == 1 && si % 2 == 0 { continue; }
                        if mode == 2 && (si % 2 == 1 || has_topk) { continue; }
                        let sp = mode == 1;
                        let rows = if has_topk { nt * topk } else { nt };
                        let mut x = if sp { g.vals(t, (rows * k) as usize) } else { g.clean(t, (rows * k) as usize) };
                        let mut w = g.rng.bytes((ne * n * k / 2) as usize);
                        if mode == 2 { mx_cancel(&mut x, rows as usize, k as usize, &mut w, (ne * n) as usize); }
                        let ws = e8m0s(g, (ne * n * k / 32) as usize, sp);
                        let bias = if sp { g.vals(t, (ne * n) as usize) } else { g.clean(t, (ne * n) as usize) };
                        let idx: Vec<u32> = (0..nt * topk).map(|_| { let r = g.rng.next(); if r % 9 == 0 { ne as u32 + (r >> 20) as u32 % 3 } else { (r >> 8) as u32 % ne as u32 } }).collect();
                        let out = g.rng.bytes((nt * topk * n) as usize * 2 + 4);
                        let st = g.stream((si + mode) % 2);
                        let hb = si % 2 == 0 || sp;
                        let (cf, of) = moe[v][ti];
                        g.pair(&format!("{}_{} nt={nt} topk={topk} ne={ne} n={n} k={k} topk_dim={has_topk} mode={mode} has_bias={hb} idx0={}", MOEN[v], TN[t], idx[0]),
                            &[x, w, ws, bias, as_bytes(&idx), out], &[5], &|r, p| unsafe {
                            (if r { of } else { cf })(p[0] as _, p[1] as _, p[2] as _, p[3] as _, p[4] as _, p[5] as _, nt, topk, ne, n, k, hb, has_topk, st)
                        });
                    }
                }
            }
        }
    }
    let (a, b) = unsafe { (cref::mxfp4_get_max_smem_optin(), ox::mxfp4_get_max_smem_optin()) };
    g.calls += 2;
    if a != b {
        g.t.failures.push(format!("mxfp4_get_max_smem_optin: C {a} vs Rust {b}"));
    }
    g.family("mxfp4", c0, f0, l0)
}

// ================================================================================ Marlin
type MarlinFn = unsafe extern "C" fn(*const c_void, *const c_void, *mut c_void, *mut c_void, *mut c_void, i32, i32, i32, *mut c_void, i32, i64);
type RepackFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, i32, i32, i32, i64);

fn marlin(g: &mut G) -> bool {
    let (c0, f0, l0) = (g.calls, g.t.failures.len(), g.family_live);
    let fns: [(&str, bool, usize, MarlinFn, MarlinFn); 4] = [
        ("marlin_gptq_4bit_f16", false, 1, cref::marlin_gptq_4bit_f16, ox::marlin_gptq_4bit_f16),
        ("marlin_gptq_4bit_bf16", false, 2, cref::marlin_gptq_4bit_bf16, ox::marlin_gptq_4bit_bf16),
        ("marlin_awq_4bit_f16", true, 1, cref::marlin_awq_4bit_f16, ox::marlin_awq_4bit_f16),
        ("marlin_awq_4bit_bf16", true, 2, cref::marlin_awq_4bit_bf16, ox::marlin_awq_4bit_bf16),
    ];
    // (k, n): each picks a different thread config for small and large m.
    let kn: [(i32, i32); 7] = [(128, 128), (256, 64), (64, 256), (192, 128), (256, 256), (4096, 256), (1024, 1024)];
    let ms: [i32; 9] = [1, 7, 16, 17, 40, 64, 65, 130, 300];
    let mut case = 0usize;
    for &(name, awq, t, cf, of) in fns.iter() {
        for &(k, n) in kn.iter() {
            for &gs in &[-1i32, 64, 128] {
                if gs > 0 && k % gs != 0 {
                    continue;
                }
                for (mi, &m) in ms.iter().enumerate() {
                    // keep the run small: every shape at a few m, the big shapes at two
                    if (k >= 1024 && mi % 4 != 1) || (k < 1024 && (mi + case) % 2 == 1) {
                        continue;
                    }
                    case += 1;
                    let sp = case % 7 == 3;
                    let a = if sp { g.vals(t, (m * k) as usize) } else { g.clean(t, (m * k) as usize) };
                    let b = g.rng.bytes((k * n / 2) as usize);
                    let ng = if gs == -1 { 1 } else { k / gs };
                    let sc = if sp { g.scales(t, (ng * n) as usize) } else { g.clean_scales(t, (ng * n) as usize) };
                    let zp = g.rng.bytes((ng * n / 2) as usize + 16);
                    let c = g.rng.bytes((m * n) as usize * 2);
                    let ws = vec![0u8; ((n / 64) * 16 + 64) as usize * 4];
                    let st = g.stream(case % 2) as usize as i64;
                    g.pair(&format!("{name} m={m} k={k} n={n} gs={gs}{}", if sp { " specials" } else { "" }), &[a, b, sc, zp, c, ws], &[4, 5], &|r, p| unsafe {
                        (if r { of } else { cf })(p[0] as _, p[1] as _, p[2] as _, if awq { p[3] as _ } else { std::ptr::null_mut() }, p[4] as _, m, k, n, p[5] as _, gs, st)
                    });
                }
            }
        }
    }
    // ---- repack: gptq (perm is always used by the C launcher) and awq, 4 and 8 bit.
    for &bits in &[4i32, 8] {
        let pack = 32 / bits;
        for &(k, n) in &[(16, 64), (128, 256), (4096, 128), (256, 1024)] {
            let w = g.rng.bytes((k / pack * n) as usize * 4);
            let mut perm: Vec<u32> = (0..k as u32).collect();
            for i in (1..k as usize).rev() {
                let j = (g.rng.next() % (i as u64 + 1)) as usize;
                perm.swap(i, j);
            }
            let out = g.rng.bytes((k * n / pack) as usize * 4);
            let st = g.stream(1) as usize as i64;
            let (cf, of): (RepackFn, RepackFn) = (cref::gptq_marlin_repack, ox::gptq_marlin_repack);
            g.pair(&format!("gptq_marlin_repack bits={bits} k={k} n={n}"), &[w.clone(), as_bytes(&perm), out.clone()], &[2], &|r, p| unsafe {
                (if r { of } else { cf })(p[0] as _, p[1] as _, p[2] as _, k, n, bits, st)
            });
            // awq: input [k, n / pack] packed along n
            let (cf, of): (RepackFn, RepackFn) = (cref::awq_marlin_repack, ox::awq_marlin_repack);
            g.pair(&format!("awq_marlin_repack bits={bits} k={k} n={n}"), &[w, out], &[1], &|r, p| unsafe {
                (if r { of } else { cf })(p[0] as _, std::ptr::null_mut(), p[1] as _, k, n / pack, bits, st)
            });
        }
    }
    let ok = g.family("marlin", c0, f0, l0);
    // Kernel level: gptq_marlin_repack_kernel<256, bits, has_perm = false> is compiled but no
    // launcher reaches it (the C launcher hard-wires has_perm = true); kdiff it against the cubin.
    use kdiff::{Arg, Harness};
    let root = format!("{}/titan-engine/oxide-kernels", std::env::var("HOME").unwrap());
    let ptx = std::env::var("MISTRALRS_QUANT_B_MARLIN_PTX").unwrap_or(format!("{root}/mistralrs-quant-b/marlin-kernels/marlin_kernels.ptx"));
    let h = Harness::from_files(&format!("{root}/reference/mistralrs-quant/marlin_repack.cubin"), &ptx);
    let mut kt = Tally::default();
    for &bits in &[4i32, 8] {
        let pack = 32 / bits;
        for &(k, n, grid) in &[(16i32, 64i32, 1u32), (128, 256, 3), (512, 128, 84)] {
            let w = g.rng.bytes((k / pack * n) as usize * 4);
            let out = g.rng.bytes((k * n / pack) as usize * 4);
            let rn = format!("_Z25gptq_marlin_repack_kernelILi256ELi{bits}ELb0EEvPKjS1_Pjii");
            let d = h.diff_pair(&rn, &format!("gptq_marlin_repack_{bits}_noperm"), (grid, 1, 1), (256, 1, 1), 16384,
                &[Arg::Buf(0), Arg::Null, Arg::Buf(1), Arg::I32(k), Arg::I32(n)], &[w, out], &[1]);
            kt.record(&format!("gptq_marlin_repack noperm bits={bits} k={k} n={n} grid={grid}"), &d);
        }
    }
    g.calls += kt.launches * 2;
    g.t.bytes += kt.bytes;
    g.t.failures.extend(kt.failures.iter().cloned());
    let ok = ok & kt.finish("marlin repack (has_perm = false) kernel-level kdiff vs marlin_repack.cubin");
    ok & marlin_kernels(g, &root, &ptx)
}

/// Kernel level: every one of the 192 Marlin instances against its reference cubin entry, at a
/// shape it tiles exactly (and one with partial rows), with 1 block (no global reduction) and 3
/// blocks (cross-block lock-serialized fp16 reduction). The launchers cannot reach 64 of the
/// instances on sm_120 (determine_thread_config's shared-memory estimate rejects them).
fn marlin_kernels(g: &mut G, root: &str, ptx: &str) -> bool {
    use kdiff::{Arg, Harness};
    let mut kt = Tally::default();
    let mut hs: Vec<(&str, Harness)> = Vec::new();
    for cub in ["marlin_matmul_f16", "marlin_matmul_bf16", "marlin_matmul_awq_f16", "marlin_matmul_awq_bf16"] {
        hs.push((cub, Harness::from_files(&format!("{root}/reference/mistralrs-quant/{cub}.cubin"), ptx)));
    }
    let mut skipped = Vec::new();
    for &(cub, rname, oname, th, mb, nb, kb, gb, awq, bf16) in crate::gen_names_marlin::MARLIN_INSTANCES {
        let (need_1, need_3) = marlin_smem_need(th as i32, mb, nb, kb, gb, awq);
        if need_1 > 96 * 1024 {
            skipped.push(oname);
            continue;
        }
        let h = &hs.iter().find(|x| x.0 == cub).unwrap().1;
        for (fname, m) in [(rname, &h.reference), (oname, &h.oxide)] {
            let f = m.load_function(fname).unwrap_or_else(|e| panic!("{fname}: {e:?}"));
            unsafe {
                cuda_core::sys::cuFuncSetAttribute(f.cu_function(), cuda_core::sys::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 96 * 1024);
            }
        }
        let t = if bf16 { 2 } else { 1 };
        let (n, k) = (16 * nb * 2, 16 * kb * 4);
        let gs = if gb == -1 { k } else { 16 * gb };
        let ng = k / gs;
        for (mi, &(m, grid)) in [(16 * mb, 1u32), (16 * mb - 5, 3u32)].iter().enumerate() {
            if grid > 1 && need_3 > 96 * 1024 {
                continue; // the cross-block reduction would overflow the 96 KB (reference faults too)
            }
            let a = if mi == 0 { g.clean(t, (16 * mb * k) as usize) } else { g.vals(t, (16 * mb * k) as usize) };
            let b = g.rng.bytes((k * n / 2) as usize);
            let c = g.rng.bytes((m * n) as usize * 2);
            let sc = g.clean_scales(t, (ng * n) as usize);
            let zp = g.rng.bytes((ng * n / 2) as usize);
            let locks = vec![0u8; 256];
            let args = [Arg::Buf(0), Arg::Buf(1), Arg::Buf(2), Arg::Buf(3), if awq { Arg::Buf(4) } else { Arg::Null }, Arg::Null,
                Arg::I32(m), Arg::I32(n), Arg::I32(k), Arg::I32(if gb == -1 { 1 } else { ng }), Arg::Buf(5)];
            let d = h.diff_pair(rname, oname, (grid, 1, 1), (th, 1, 1), 96 * 1024, &args, &[a, b, c, sc, zp, locks], &[2, 5]);
            kt.record(&format!("{oname} m={m} n={n} k={k} grid={grid}"), &d);
        }
    }
    g.calls += kt.launches * 2;
    g.t.bytes += kt.bytes;
    g.t.failures.extend(kt.failures.iter().cloned());
    println!("  marlin instances whose shared-memory layout exceeds the launcher's fixed 96 KB (never launchable, not diffed): {skipped:?}");
    kt.finish("marlin: every launchable instance, kernel-level kdiff vs the marlin_matmul_*.cubin entries")
}

/// Bytes of dynamic shared memory a Marlin instance touches: the 4-stage pipeline plus the
/// largest `sh_red` extent of thread_block_reduce / write_result (single block) and, with a
/// cross-block reduction, global_reduce. Index expressions copied from marlin_kernel.cuh.
fn marlin_smem_need(th: i32, mb: i32, nb: i32, kb: i32, gb: i32, awq: bool) -> (usize, usize) {
    let a_sh_stage = 2 * kb * 16 * mb;
    let b_sh_stage = nb * 8 * kb;
    let s_tb = if gb != -1 && gb < kb { kb / gb } else { 1 };
    let s_sh_stage = s_tb * 2 * nb;
    let zp_sh_stage = if awq { s_tb * nb / 2 } else { 0 };
    let base = 4 * (a_sh_stage + b_sh_stage + zp_sh_stage + s_sh_stage);
    let mut red = 0; // max int4 index + 1 used in sh_red
    let bst = nb * 8;
    let red_off = th / bst / 2;
    if red_off >= 1 {
        let stride = bst * 8;
        for t in 0..th {
            let red_idx = t / bst;
            let rd = stride * red_idx + t % bst;
            let mut i = red_off;
            while i > 0 {
                if i <= red_idx && red_idx < 2 * i {
                    for j in 0..8 {
                        red = red.max(bst * j + rd - stride * i + 1);
                        if i < red_off {
                            red = red.max(bst * j + rd + 1);
                        }
                    }
                }
                i /= 2;
            }
            if red_idx == 0 {
                red = red.max(bst * 7 + rd + 1);
            }
        }
    }
    let c_sh_stride = 2 * nb + 1;
    for t in 0..th {
        if t / 32 < nb / 4 {
            let wr0 = (4 * c_sh_stride) * ((t % 32) / 4) + (t % 32) % 4 + 32 * (t / 32);
            let last = wr0 + 16 * (4 * c_sh_stride) * (mb - 1) + 8 * 3 + (4 * c_sh_stride) * 8 + 4;
            red = red.max(last / 4 + 1); // half2 index -> int4
        }
        let c_sh_rd = c_sh_stride * (t / (2 * nb)) + t % (2 * nb);
        let iters = (16 * mb + th / (2 * nb) - 1) / (th / (2 * nb));
        red = red.max(c_sh_rd + c_sh_stride * (th / (2 * nb)) * (iters - 1) + 1);
    }
    let one = base + red;
    let active = 32 * nb / 4;
    let three = one.max(base + active * mb * 4);
    (one as usize * 16, three as usize * 16)
}

// ================================================================================ bitsandbytes
type BnbFn = unsafe extern "C" fn(*mut f32, *mut u8, *mut f32, *mut u8, i32, i32, *mut c_void);
fn bnb(g: &mut G) -> bool {
    let (c0, f0, l0) = (g.calls, g.t.failures.len(), g.family_live);
    let fns: [[(BnbFn, BnbFn); 3]; 3] = [
        [(fnp!(cref::dequantize_blockwise_f32_int8, BnbFn), fnp!(ox::dequantize_blockwise_f32_int8, BnbFn)), (fnp!(cref::dequantize_blockwise_f32_fp4, BnbFn), fnp!(ox::dequantize_blockwise_f32_fp4, BnbFn)), (fnp!(cref::dequantize_blockwise_f32_nf4, BnbFn), fnp!(ox::dequantize_blockwise_f32_nf4, BnbFn))],
        [(fnp!(cref::dequantize_blockwise_f16_int8, BnbFn), fnp!(ox::dequantize_blockwise_f16_int8, BnbFn)), (fnp!(cref::dequantize_blockwise_f16_fp4, BnbFn), fnp!(ox::dequantize_blockwise_f16_fp4, BnbFn)), (fnp!(cref::dequantize_blockwise_f16_nf4, BnbFn), fnp!(ox::dequantize_blockwise_f16_nf4, BnbFn))],
        [(fnp!(cref::dequantize_blockwise_bf16_int8, BnbFn), fnp!(ox::dequantize_blockwise_bf16_int8, BnbFn)), (fnp!(cref::dequantize_blockwise_bf16_fp4, BnbFn), fnp!(ox::dequantize_blockwise_bf16_fp4, BnbFn)), (fnp!(cref::dequantize_blockwise_bf16_nf4, BnbFn), fnp!(ox::dequantize_blockwise_bf16_nf4, BnbFn))],
    ];
    const DN: [&str; 3] = ["int8", "fp4", "nf4"];
    for t in 0..3 {
        for dt in 0..3 {
            for (si, &(n, bs)) in [(4096, 64), (1000, 64), (1, 64), (513, 128), (1025, 256), (5000, 4096), (2047, 32)].iter().enumerate() {
                let bytes = if dt == 0 { n } else { (n + 1) / 2 };
                let a = g.rng.bytes(bytes as usize);
                let nb = (n + bs - 1) / bs + 8;
                // absmax: normal, plus denormals / specials in some cases (the FP4 1.0 code keeps a denormal)
                let am: Vec<f32> = (0..nb).map(|i| { let r = g.rng.next(); if si % 2 == 1 && r % 5 == 0 { [1e-39f32, -2e-40, f32::INFINITY, f32::NAN, 0.0, -0.0, 3e38][i as usize % 7] } else { ((r >> 11) as f64 / (1u64 << 53) as f64 * 4.0 - 2.0) as f32 } }).collect();
                let code = g.rng.f32s(256);
                let es = TS[t];
                let out = g.rng.bytes(n as usize * es + 64);
                let st = g.stream(si % 2);
                let (cf, of) = fns[t][dt];
                g.pair(&format!("dequantize_blockwise_{}_{} n={n} blocksize={bs}", TN[t], DN[dt]), &[as_bytes(&code), a, as_bytes(&am), out], &[3], &|r, p| unsafe {
                    (if r { of } else { cf })(p[0] as _, p[1] as _, p[2] as _, p[3] as _, bs, n, st)
                });
            }
        }
    }
    g.family("bitsandbytes", c0, f0, l0)
}

// ================================================================================ CUTLASS grouped GEMM
fn grouped_mm(g: &mut G) -> bool {
    let (c0, f0, l0) = (g.calls, g.t.failures.len(), g.family_live);
    for pc in [0i32, 1, 3, 8] {
        let (a, b) = unsafe { (cref::cutlass_moe_grouped_gemm_2x_workspace_size(pc), ox::cutlass_moe_grouped_gemm_2x_workspace_size(pc)) };
        g.calls += 2;
        if a != b {
            g.t.failures.push(format!("cutlass_moe_grouped_gemm_2x_workspace_size({pc}): C {a} vs Rust {b}"));
        }
    }
    // problem sets: (m, n, k) per expert; K multiples of 8 (alignment), incl. K % 32 != 0 residues.
    let sets: [&[(i32, i32, i32)]; 6] = [
        &[(64, 128, 256)],
        &[(17, 96, 64), (0, 96, 64), (130, 96, 64), (5, 96, 64)],
        &[(33, 256, 40), (70, 256, 40)],
        &[(1, 64, 8), (7, 72, 24), (100, 136, 200)],
        &[(200, 128, 512), (3, 128, 512), (48, 128, 512)],
        &[(9, 40, 1032), (16, 8, 1032)],
    ];
    let bytes_of = |g: &G, v: &Vec<u8>| g.up(v);
    let _ = bytes_of;
    for (si, set) in sets.iter().enumerate() {
        for cfg in 0..3 {
            for mode in 0..2 {
                let sp = mode == 1;
                // pack every tensor into one buffer each (A, B, D); pointer arrays are built per side.
                let (mut a_all, mut b_all, mut d_all) = (Vec::new(), Vec::new(), Vec::new());
                let mut offs = Vec::new();
                for &(m, n, k) in set.iter() {
                    let lda = k + if si == 3 { 8 } else { 0 };
                    let ldb = k;
                    let ldd = n + if si == 2 { 8 } else { 0 };
                    let a = if sp { g.vals(2, (m.max(1) * lda) as usize) } else { g.clean(2, (m.max(1) * lda) as usize) };
                    let b = if sp { g.vals(2, (n * ldb) as usize) } else { g.clean(2, (n * ldb) as usize) };
                    let d = g.rng.bytes((m.max(1) * ldd) as usize * 2);
                    offs.push((a_all.len(), b_all.len(), d_all.len(), lda, ldb, ldd));
                    a_all.extend(a);
                    b_all.extend(b);
                    d_all.extend(d);
                }
                let ps: Vec<i32> = set.iter().flat_map(|&(m, n, k)| [m, n, k]).collect();
                let pc = set.len() as i32;
                let lds: Vec<i64> = offs.iter().flat_map(|o| [o.3 as i64, o.4 as i64, o.5 as i64]).collect();
                let st = g.stream(1);
                let offs2 = offs.clone();
                let status = std::cell::Cell::new((0i32, 0i32));
                // bufs: 0 A, 1 B, 2 D, 3 problem sizes, 4 ptr arrays (3 x pc u64), 5 lds (3 x pc i64: interleaved -> split below)
                let lda: Vec<i64> = lds.chunks(3).map(|c| c[0]).collect();
                let ldb: Vec<i64> = lds.chunks(3).map(|c| c[1]).collect();
                let ldd: Vec<i64> = lds.chunks(3).map(|c| c[2]).collect();
                g.pair(&format!("launch_cutlass_moe_grouped_gemm_2x_bf16 set={si} tile_cfg={cfg} mode={mode}"),
                    &[a_all, b_all, d_all, as_bytes(&ps), vec![0u8; 24 * pc as usize + 8], as_bytes(&lda), as_bytes(&ldb), as_bytes(&ldd)], &[2], &|r, p| unsafe {
                    // fill the device pointer arrays for this side
                    let mut ptrs: Vec<u64> = Vec::new();
                    for o in &offs2 { ptrs.push(p[0] + o.0 as u64); }
                    for o in &offs2 { ptrs.push(p[1] + o.1 as u64); }
                    for o in &offs2 { ptrs.push(p[2] + o.2 as u64); }
                    let hb = as_bytes(&ptrs);
                    cuda_core::sys::cuMemcpyHtoD_v2(p[4], hb.as_ptr() as *const c_void, hb.len());
                    // A pageable HtoD copy may return before the DMA lands; the launch below goes
                    // to a non-blocking stream, so wait for it (an early run read stale pointers).
                    cuda_core::sys::cuCtxSynchronize();
                    let pa = p[4] as *mut *const c_void;
                    let pb = (p[4] + 8 * pc as u64) as *mut *const c_void;
                    let pd = (p[4] + 16 * pc as u64) as *mut *mut c_void;
                    let s = if r {
                        ox::launch_cutlass_moe_grouped_gemm_2x_bf16(pa, pb, pd, p[3] as _, pc, p[5] as _, p[6] as _, p[7] as _, std::ptr::null_mut(), 0, cfg, st)
                    } else {
                        cref::launch_cutlass_moe_grouped_gemm_2x_bf16(pa, pb, pd, p[3] as _, pc, p[5] as _, p[6] as _, p[7] as _, std::ptr::null_mut(), 0, cfg, st)
                    };
                    let (x, y) = status.get();
                    status.set(if r { (x, s) } else { (s, y) });
                });
                let (x, y) = status.get();
                if x != y {
                    g.t.failures.push(format!("grouped gemm status set={si}: C {x} vs Rust {y}"));
                }
            }
        }
    }
    g.family("grouped_mm_2x", c0, f0, l0)
}

include!("gen_pairs.rs");

pub fn run() -> bool {
    let ctx = CudaContext::new(0).expect("cuda context");
    ctx.bind_to_thread().unwrap();
    let streams = vec![ctx.default_stream(), ctx.new_stream().unwrap()];
    let mut g = G { ctx: ctx.clone(), streams, rng: Rng(0x5EED_B00B), t: Tally::default(), calls: 0, inert: 0, family_live: 0 };
    let only = std::env::var("GATE_ONLY").unwrap_or_default();
    let mut ok = true;
    let fams: [(&str, fn(&mut G) -> bool); 10] = [("afq", afq), ("scalar_fp8", fp8_scalar), ("vector_fp8", fp8_vector), ("blockwise_fp8", fp8_blockwise), ("blockwise_fp8_gemm", fp8_gemm), ("gptq", gptq), ("mxfp4", mxfp4), ("marlin", marlin), ("grouped_mm_2x", grouped_mm), ("bitsandbytes", bnb)];
    for (name, f) in fams {
        if only.is_empty() || only.split(',').any(|o| o == name) {
            ok &= f(&mut g);
        }
    }
    let nshow = if std::env::var("GATE_VERBOSE").is_ok() { usize::MAX } else { 40 };
    for f in g.t.failures.iter().take(nshow) {
        println!("  FAIL {f}");
    }
    ok &= g.t.failures.is_empty();
    println!("inert calls (no output byte changed on either side): {}", g.inert);
    if only.is_empty() {
        let got: std::collections::HashSet<&str> = ox::launched().into_iter().collect();
        let missed: Vec<&&str> = crate::gen_names::ALL_KERNELS.iter().filter(|n| !got.contains(**n)).collect();
        println!("oxide kernel instances launched by the Rust launchers: {} of {}; not reached: {:?}", crate::gen_names::ALL_KERNELS.len() - missed.len(), crate::gen_names::ALL_KERNELS.len(), missed);
    }
    println!(
        "mistralrs-quant-b: {} launcher calls, {} bytes compared, {} failing -> {}",
        g.calls,
        g.t.bytes,
        g.t.failures.len(),
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}
