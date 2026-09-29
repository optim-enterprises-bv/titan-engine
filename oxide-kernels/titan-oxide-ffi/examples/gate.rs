//! Launcher-vs-launcher gate for titan-oxide-ffi: for a sample of every launcher family, the REAL C
//! launcher from the reference static libraries (renamed rm_ / rq_ / rc_ / rp_ by make_ref.sh) and
//! this crate's exported twin (called through its module path) run on identical inputs in the same
//! primary context, alternating the per-thread default stream (null handle) and a created stream.
//! Every byte of every buffer the call can see is compared (inputs, outputs, and a random guard
//! region after each buffer), except buffers the reference itself does not write deterministically
//! (checked by running the C launcher twice). Prints one line per family and a final PASS / FAIL.
//!
//! Run: ./make_ref.sh && cargo run --release --example gate
//! (LD_LIBRARY_PATH=$HOME/titan-engine/lib:/usr/local/cuda/lib64)
#![allow(clippy::too_many_arguments, clippy::type_complexity)]
use std::collections::BTreeMap;
use std::ffi::c_void;
use titan_oxide_ffi as ox;

#[cfg(not(has_ref))]
compile_error!("the gate links the renamed reference libraries: run ./make_ref.sh first");

type P = *const c_void;
type M = *mut c_void;

// ================================================================================================
// driver + reference declarations

mod drv {
    use std::ffi::{c_int, c_uint, c_void};
    unsafe extern "C" {
        pub fn cuInit(f: c_uint) -> c_int;
        pub fn cuDeviceGet(d: *mut c_int, o: c_int) -> c_int;
        pub fn cuDevicePrimaryCtxRetain(c: *mut *mut c_void, d: c_int) -> c_int;
        pub fn cuCtxSetCurrent(c: *mut c_void) -> c_int;
        pub fn cuCtxSynchronize() -> c_int;
        pub fn cuStreamCreate(s: *mut *mut c_void, flags: c_uint) -> c_int;
        pub fn cuMemAlloc_v2(p: *mut u64, n: usize) -> c_int;
        pub fn cuMemFree_v2(p: u64) -> c_int;
        pub fn cuMemcpyHtoD_v2(d: u64, s: *const c_void, n: usize) -> c_int;
        pub fn cuMemcpyDtoH_v2(d: *mut c_void, s: u64, n: usize) -> c_int;
    }
    unsafe extern "C" {
        pub fn cudaGetLastError() -> c_int;
        pub fn cudaFree(p: *mut c_void) -> c_int;
    }
}

/// Declares reference launchers under their renamed symbols (`$pre` + name).
macro_rules! cref {
    ($pre:literal; $( fn $name:ident($($a:ident: $t:ty),* $(,)?) $(-> $r:ty)?; )*) => {
        unsafe extern "C" { $(
            #[link_name = concat!($pre, stringify!($name))]
            pub fn $name($($a: $t),*) $(-> $r)?;
        )* }
    };
}

/// libmistralrsquant.a
mod rq {
    use super::{M, P};
    use std::ffi::c_void;
    cref! { "rq_";
        fn launch_mmvq_gguf_quantize_q8_1_f32(x: P, vy: M, kx: i32, kxp: i32, rows: i32, s: M);
        fn launch_mmvq_gguf_q4_k_f32_plain(vx: P, vy: P, dst: M, nc: i32, nr: i32, scy: i32, scd: i32, b: i32, s: M);
        fn launch_mmvq_gguf_q4_0_bf16_plain(vx: P, vy: P, dst: M, nc: i32, nr: i32, scy: i32, scd: i32, b: i32, s: M);
        fn launch_mmvq_gguf_q8_0_bf16_fused_glu(g: P, u: P, vy: P, dst: M, nc: i32, nr: i32, scy: i32, scd: i32, b: i32, act: i32, s: M);
        fn launch_mmvq_gguf_q6_k_f16_fused_qkv(q: P, k: P, v: P, vy: P, qd: M, kd: M, vd: M, nc: i32, nq: i32, nk: i32, nv: i32, scy: i32, b: i32, s: M);
        fn launch_mmq_quantize_q8_1_DS4(x: P, ids: *const i32, vy: M, t: i32, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i64, ne2: i64, ne3: i64, s: M);
        fn launch_mmq_quantize_q8_1_D4(x: P, ids: *const i32, vy: M, t: i32, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i64, ne2: i64, ne3: i64, s: M);
        fn launch_mmq_gguf_q4_k(f: M, x: P, y: P, d: M, a: i64, b: i64, c: i64, e: i64, g: i64, cc: i32, nsm: i32, smpbo: i64, ws: i32, td: i32, s: M);
        fn launch_mmq_gguf_q8_0(f: M, x: P, y: P, d: M, a: i64, b: i64, c: i64, e: i64, g: i64, cc: i32, nsm: i32, smpbo: i64, ws: i32, td: i32, s: M);
        fn rotary_embedding(q: P, k: P, c: P, s: P, neox: i32, hs: i32, nt: i64, rd: i32, nh: i32, nkv: i32, qs: i64, ks: i64, dtype: u32, stream: i64);
        fn launch_gemv_bf16(a: *const u16, x: *const u16, b: *const u16, y: *mut u16, m: i32, k: i32, bs: i32, hb: bool, s: M);
        fn launch_quantize_q8_1(x: *const f32, vy: M, kx: i32, kp: i32, nbx: i32, rows: i32, s: M);
        fn launch_indexed_moe_forward_q4k_q8_1(w: P, x: P, i: *const u32, o: *mut f32, n: i32, k: i32, b: i32, t: i32, kp: i32, d1: i32, s: M);
        fn fused_glu_f16(a: P, b: P, o: M, n: u32, act: i32, s: M);
        fn gptoss_swiglu_bf16(g: P, u: P, o: M, n: u32, alpha: f32, limit: f32, s: M);
        fn dequantize_blockwise_f32_nf4(c: *const f32, a: *const u8, am: *const f32, o: *mut f32, bs: i32, n: i32, s: M);
        fn dequantize_blockwise_bf16_int8(c: *const f32, a: *const u8, am: *const f32, o: *mut u16, bs: i32, n: i32, s: M);
        fn bitwise_and_u32(a: P, b: P, o: M, n: u32);
        fn dequantize_4bit_u8_kernel_f32(wq: *const u8, s: *const f32, z: *const f32, o: *const f32, h: i32, w: i32);
        fn afq_qmv_4bit_gs64_bf16(x: *const u16, w: *const u32, s: *const u16, b: *const u16, y: *mut u16, m: i32, n: i32, k: i32);
        fn afq_dequantize_4bit_gs64_bf16(w: *const u32, s: *const u16, b: *const u16, o: *mut u16, rows: i32, cols: i32);
        fn launch_dequant_fp8_blockwise_kernel_bf16(w: *const u8, s: *const f32, o: *mut u16, h: i32, wd: i32, rs: i32, ss: i32, by: i32, bx: i32, st: M);
        fn launch_fp8_to_f32_kernel(i: *const u8, o: *mut f32, n: usize, s: M);
        fn gemm_half_q_half_cuda_part(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, perm: *const i32, c: *mut u16, m: i32, n: i32, k: i32, mc: i32, groups: i32, bit: i32);
        fn launch_mxfp4_matmul_bf16(i: *const u16, w: *const u8, ws: *const u8, b: *const u16, o: *mut u16, m: i32, n: i32, k: i32, hb: bool, s: M);
        fn marlin_gptq_4bit_bf16(a: P, b: P, s: M, z: M, c: M, m: i32, k: i32, n: i32, ws: M, gs: i32, st: i64);
    }
    #[allow(unused)]
    fn _unused(_: *mut c_void) {}
}

/// libmistralrscuda.a
mod rc {
    use super::{M, P};
    cref! { "rc_";
        fn cuda_graph_copy_bytes(src: P, dst: M, n: i64, stream: i64) -> i32;
        fn gated_delta_rule_recurrence(q: *const f32, k: *const f32, v: *const f32, g: *const f32, beta: *const f32, state: *mut f32, output: *mut f32, bh: i32, seq_len: i32, k_dim: i32, v_dim: i32, stream: i64);
        fn moe_gemm_wmma(input: P, weights: P, sorted: *const i32, eids: *const i32, tw: *const f32, output: M, e: i32, topk: i32, m: i32, n: i32, k: i32, dtype: i32, stream: i64);
        fn moe_gemv(input: P, weights: P, sorted: *const i32, eids: *const i32, tw: *const f32, output: M, e: i32, topk: i32, m: i32, n: i32, k: i32, dtype: i32, stream: i64);
        fn rms_norm_residual_f16(x: P, r: P, w: P, s: P, d: M, nrows: i32, ncols: i32, eps: f32, stream: i64);
        fn topk_f32(input: P, v: M, i: M, nrows: i32, ncols: i32, k: i32, stream: i64);
        fn asort_asc_f32(x: M, dst: M, nrows: i32, ncols: i32, inplace: bool, stream: i64);
    }
}

/// libmistralrspagedattention.a
mod rp {
    use super::P;
    use std::ffi::c_void;
    cref! { "rp_";
        fn reshape_and_cache(key: *mut c_void, value: *mut c_void, kc: *mut c_void, vc: *mut c_void, slots: *mut i64, nt: i32, nh: i32, hs: i32, bs: i32, x: i32, ks: i32, vs: i32, stream: *mut c_void, dtype: u32, cache_dtype: u32, k_scale: *mut f32, v_scale: *mut f32);
        fn paged_attention_v1_f16(out: *mut c_void, q: *mut c_void, kc: *mut c_void, vc: *mut c_void, alibi: *mut c_void, nkv: i32, scale: f32, softcap: f32, bt: *mut u32, cl: *mut u32, bs: i32, max_ctx: i32, nseq: i32, nh: i32, hs: i32, mbps: i32, qs: i32, kvbs: i32, kvhs: i32, stream: *mut c_void, cache_dtype: u32, ks: *mut f32, vs: *mut f32, sinks: *const f32);
        fn paged_attention_v2_bf16(out: *mut c_void, es: *mut f32, ml: *mut f32, tmp: *mut c_void, q: *mut c_void, kc: *mut c_void, vc: *mut c_void, alibi: *mut c_void, nkv: i32, scale: f32, softcap: f32, bt: *mut u32, cl: *mut u32, bs: i32, max_ctx: i32, nseq: i32, nh: i32, hs: i32, mbps: i32, qs: i32, kvbs: i32, kvhs: i32, stream: *mut c_void, cache_dtype: u32, ks: *mut f32, vs: *mut f32, sinks: *const f32);
        fn flash_attn_sinks_f16(q: P, k: P, v: P, o: *mut c_void, sinks: *const f32, scale: f32, b: i32, ql: i32, kl: i32, nh: i32, nkv: i32, hd: i32, w: i32, s: *mut c_void);
        fn flashinfer_decode(q: *mut c_void, kc: *mut c_void, vc: *mut c_void, indptr: *const i32, indices: *const i32, last: *const i32, req: *const i32, tiles: *const i32, o_indptr: *const i32, chunk: *const i32, mask: *const u8, o: *mut c_void, tmp_v: *mut c_void, tmp_s: *mut c_void, b: i32, padded: i32, nqo: i32, nkv: i32, hd: i32, ps: i32, qsn: i32, qsh: i32, sm: f32, wl: i32, cap: f32, dtype: u32, s: *mut c_void) -> i32;
        fn reshape_and_cache_flashinfer(key: *mut c_void, value: *mut c_void, kc: *mut c_void, vc: *mut c_void, slots: *mut i64, nt: i32, nh: i32, hs: i32, bs: i32, ks: i32, vs: i32, dtype: u32, s: *mut c_void);
    }
}

/// candle's libmoe.a
mod rm {
    use super::M;
    use std::ffi::c_void;
    cref! { "rm_";
        fn moe_gemm_gguf(input: *const f32, w: *const c_void, sorted: *const i32, eids: *const i32, tw: *const f32, out: M, e: i32, topk: i32, m: i32, n: i32, k: i32, t: i32, stream: i64);
        fn moe_gemm_gguf_prefill(input: *const c_void, w: *const u8, sorted: *const i32, eids: *const i32, tw: *const f32, out: M, e: i32, topk: i32, m: i32, n: i32, k: i32, dt: i32, t: i32, stream: i64);
    }
}

// ================================================================================================
// harness

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn uni(&mut self) -> f32 {
        ((self.next() >> 11) as f64 / (1u64 << 53) as f64) as f32
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
}

fn f32_to_f16(v: f32) -> u16 {
    let x = v as f64;
    let sign = if x.is_sign_negative() { 0x8000u16 } else { 0 };
    if v.is_nan() {
        return 0x7e00;
    }
    let a = x.abs();
    if a == 0.0 {
        return sign;
    }
    if a >= 65520.0 {
        return sign | 0x7c00;
    }
    let e = a.log2().floor() as i32;
    if e < -14 {
        return sign | (a / 2f64.powi(-24)).round_ties_even() as u16;
    }
    let mut mi = ((a / 2f64.powi(e) - 1.0) * 1024.0).round_ties_even() as u32;
    let mut ee = e + 15;
    if mi == 1024 {
        mi = 0;
        ee += 1;
    }
    sign | ((ee as u16) << 10) | mi as u16
}
fn f32_to_bf16(v: f32) -> u16 {
    let b = v.to_bits();
    if v.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}
fn bytes_of<T: Copy>(v: &[T]) -> Vec<u8> {
    let n = std::mem::size_of_val(v);
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n).to_vec() }
}

/// dtype codes of the generators: 0 f16, 1 bf16, 2 f32.
fn enc(t: usize, v: &[f32]) -> Vec<u8> {
    match t {
        0 => bytes_of(&v.iter().map(|&x| f32_to_f16(x)).collect::<Vec<u16>>()),
        1 => bytes_of(&v.iter().map(|&x| f32_to_bf16(x)).collect::<Vec<u16>>()),
        _ => bytes_of(v),
    }
}

const GUARD: usize = 64;

struct Dev(u64);
impl Dev {
    fn new(data: &[u8]) -> Dev {
        let mut p = 0u64;
        unsafe {
            ck(drv::cuMemAlloc_v2(&mut p, data.len().max(1)), "cuMemAlloc");
            ck(drv::cuMemcpyHtoD_v2(p, data.as_ptr() as *const c_void, data.len()), "HtoD");
        }
        Dev(p)
    }
    fn read(&self, n: usize) -> Vec<u8> {
        let mut v = vec![0u8; n];
        unsafe { ck(drv::cuMemcpyDtoH_v2(v.as_mut_ptr() as *mut c_void, self.0, n), "DtoH") };
        v
    }
}
impl Drop for Dev {
    fn drop(&mut self) {
        unsafe { drv::cuMemFree_v2(self.0) };
    }
}
fn ck(r: i32, what: &str) {
    assert!(r == 0, "{what}: CUDA error {r}");
}
fn sync(what: &str) {
    let r = unsafe { drv::cuCtxSynchronize() };
    assert!(r == 0, "{what}: CUDA error {r} (context is lost; cannot continue)");
}

struct G {
    rng: Rng,
    stream: M,
    k: usize,
    fam: &'static str,
    /// family -> (cases, launcher calls, bytes compared, failures, nondeterministic buffers skipped)
    stats: BTreeMap<&'static str, (usize, usize, usize, usize, usize)>,
    failures: Vec<String>,
}

impl G {
    /// The stream for the next case: alternately the per-thread default (null) and the created one.
    fn s(&mut self) -> M {
        self.k += 1;
        if self.k % 2 == 0 { std::ptr::null_mut() } else { self.stream }
    }
    fn family(&mut self, f: &'static str) {
        self.fam = f;
        self.stats.entry(f).or_default();
    }
    fn f32s(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| lo + (hi - lo) * self.rng.uni()).collect()
    }
    /// Values of dtype `t` in [lo, hi), with one special (nan / inf / denormal / -0) every `sp` (0: none).
    fn vals(&mut self, t: usize, n: usize, lo: f32, hi: f32, sp: u64) -> Vec<u8> {
        let mut v = self.f32s(n, lo, hi);
        if sp > 0 {
            for x in v.iter_mut() {
                if self.rng.below(sp) == 0 {
                    *x = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0, 1e-40, 6e-8, 0.0][self.rng.below(7) as usize];
                }
            }
        }
        enc(t, &v)
    }
    /// f16 scale bits: mostly normal 2^-12..2^3.
    fn f16_scale(&mut self) -> u16 {
        let r = self.rng.next();
        let sign = ((r >> 8) & 1) as u16;
        let exp = 3 + ((r >> 9) % 16) as u16;
        (sign << 15) | (exp << 10) | ((r >> 16) & 0x3ff) as u16
    }
    /// `n` GGUF weight blocks of `bs` bytes with f16 scale fields at `f16s`, plus `pad` tail bytes.
    fn blocks(&mut self, bs: usize, f16s: &[usize], n: usize, pad: usize) -> Vec<u8> {
        let mut b = self.rng.bytes(n * bs + pad);
        for i in 0..n {
            for &o in f16s {
                let v = self.f16_scale();
                b[i * bs + o..i * bs + o + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
        b
    }
    /// `n` random Q8_1 blocks (36 bytes: d, s as f16, 32 int8 quants).
    fn q8_1(&mut self, n: usize) -> Vec<u8> {
        let mut b = self.rng.bytes(n * 36);
        for i in 0..n {
            let (d, s) = (self.f16_scale(), self.f16_scale());
            b[i * 36..i * 36 + 2].copy_from_slice(&d.to_le_bytes());
            b[i * 36 + 2..i * 36 + 4].copy_from_slice(&s.to_le_bytes());
        }
        b
    }

    /// Runs `f` (C side: rust=false; twin: rust=true) on private device copies of `bufs` (each
    /// followed by a random guard), the C side twice, and compares every byte of every buffer not in
    /// `skip`. Buffers where the two C runs disagree are reported as nondeterministic and skipped.
    /// `f` returns the launcher's return value (0 for void launchers), which is compared too.
    fn case(&mut self, label: &str, bufs: Vec<Vec<u8>>, skip: &[usize], f: impl Fn(bool, &[u64]) -> i64) {
        let bufs: Vec<Vec<u8>> = bufs
            .into_iter()
            .map(|mut b| {
                b.extend(self.rng.bytes(GUARD));
                b
            })
            .collect();
        let run = |rust: bool| -> (Vec<Vec<u8>>, i64) {
            let d: Vec<Dev> = bufs.iter().map(|b| Dev::new(b)).collect();
            sync("upload");
            // clear the runtime's last-error slot (a dropped launch error would leak into
            // launchers that CUDA_CHECK(cudaGetLastError()))
            unsafe { drv::cudaGetLastError() };
            let p: Vec<u64> = d.iter().map(|x| x.0).collect();
            let r = f(rust, &p);
            sync(&format!("{label} ({})", if rust { "Rust" } else { "C" }));
            (d.iter().zip(&bufs).map(|(x, b)| x.read(b.len())).collect(), r)
        };
        let (c1, r1) = run(false);
        let (o, ro) = run(true);
        let (c2, _) = run(false);
        let st = self.stats.entry(self.fam).or_default();
        st.0 += 1;
        st.1 += 3;
        let mut bad = vec![];
        let mut written = false;
        for i in 0..bufs.len() {
            written |= c1[i] != bufs[i];
            if skip.contains(&i) {
                continue;
            }
            if c1[i] != c2[i] {
                st.4 += 1;
                continue;
            }
            st.2 += bufs[i].len();
            let n = c1[i].iter().zip(&o[i]).filter(|(a, b)| a != b).count();
            if n > 0 {
                let first = c1[i].iter().zip(&o[i]).position(|(a, b)| a != b).unwrap();
                bad.push(format!("buf {i}: {n} of {} bytes differ (first at {first}: C {:#04x} Rust {:#04x})", bufs[i].len(), c1[i][first], o[i][first]));
            }
        }
        if r1 != ro {
            bad.push(format!("return value: C {r1} Rust {ro}"));
        }
        if !written {
            bad.push("the C launcher wrote nothing (vacuous case)".into());
        }
        if !bad.is_empty() {
            st.3 += 1;
            self.failures.push(format!("[{}] {label}: {}", self.fam, bad.join("; ")));
        }
    }
}

// ================================================================================================
// families

/// GGUF formats: (name, block bytes, values per block, f16 scale offsets).
const Q4_0: (usize, usize, &[usize]) = (18, 32, &[0]);
const Q8_0: (usize, usize, &[usize]) = (34, 32, &[0]);
const Q4_K: (usize, usize, &[usize]) = (144, 256, &[0, 2]);
const Q6_K: (usize, usize, &[usize]) = (210, 256, &[208]);

fn quant_c(g: &mut G) {
    g.family("quant_c mmvq/mmq (libmistralrsquant.a)");
    use ox::quant_c as q;
    // quantize f32 -> q8_1
    for &(kx, kxp, rows) in &[(1000, 1024, 3), (4096, 4096, 1), (70, 96, 5)] {
        let x = g.vals(2, rows * kx, -8.0, 8.0, 97);
        let y = g.rng.bytes(rows * kxp / 32 * 36 + 36);
        let s = g.s();
        g.case(&format!("mmvq quantize_q8_1_f32 kx={kx} kxp={kxp} rows={rows}"), vec![x, y], &[], |r, p| unsafe {
            let f = if r { q::launch_mmvq_gguf_quantize_q8_1_f32 } else { rq::launch_mmvq_gguf_quantize_q8_1_f32 };
            f(p[0] as P, p[1] as M, kx as i32, kxp as i32, rows as i32, s);
            0
        });
    }
    // plain GEMV: q4_k -> f32 and q4_0 -> bf16 at several batch sizes (geometry switches at 1 / <= 4 / > 4)
    for &(fmt, bf16, ncols, nrows, b) in &[(Q4_K, false, 2048, 33, 1), (Q4_K, false, 4608, 131, 3), (Q4_0, true, 4096, 64, 8), (Q4_0, true, 96, 7, 1)] {
        let (bs, qk, f16s) = fmt;
        let bpr = ncols / qk;
        let scy = ncols.div_ceil(512) * 512 / 32;
        let w = g.blocks(bs, f16s, (nrows + 1) * bpr, 64);
        let y = g.q8_1((b - 1) * scy + bpr * qk / 32 + 8);
        let es = if bf16 { 2 } else { 4 };
        let d = g.rng.bytes((b * nrows + 3) * es);
        let s = g.s();
        g.case(&format!("mmvq plain {} ncols={ncols} nrows={nrows} b={b}", if bf16 { "q4_0_bf16" } else { "q4_k_f32" }), vec![w, y, d], &[], |r, p| unsafe {
            let f = match (bf16, r) {
                (false, false) => rq::launch_mmvq_gguf_q4_k_f32_plain,
                (false, true) => q::launch_mmvq_gguf_q4_k_f32_plain,
                (true, false) => rq::launch_mmvq_gguf_q4_0_bf16_plain,
                (true, true) => q::launch_mmvq_gguf_q4_0_bf16_plain,
            };
            f(p[0] as P, p[1] as P, p[2] as M, ncols as i32, nrows as i32, scy as i32, nrows as i32, b as i32, s);
            0
        });
    }
    // fused GLU (q8_0 -> bf16), every activation
    for act in 0..4 {
        let (bs, qk, f16s) = Q8_0;
        let (ncols, nrows, b) = (2048usize, 64usize, 1 + act % 2);
        let bpr = ncols / qk;
        let scy = ncols / 32;
        let wg = g.blocks(bs, f16s, (nrows + 1) * bpr, 64);
        let wu = g.blocks(bs, f16s, (nrows + 1) * bpr, 64);
        let y = g.q8_1(b * scy + 8);
        let d = g.rng.bytes((b * nrows + 3) * 2);
        let s = g.s();
        g.case(&format!("mmvq fused_glu q8_0_bf16 act={act} b={b}"), vec![wg, wu, y, d], &[], |r, p| unsafe {
            let f = if r { q::launch_mmvq_gguf_q8_0_bf16_fused_glu } else { rq::launch_mmvq_gguf_q8_0_bf16_fused_glu };
            f(p[0] as P, p[1] as P, p[2] as P, p[3] as M, ncols as i32, nrows as i32, scy as i32, nrows as i32, b as i32, act as i32, s);
            0
        });
    }
    // fused QKV (q6_k -> f16)
    for &(nr, b) in &[([1024usize, 256usize, 256usize], 1usize), ([300, 280, 290], 3)] {
        let (bs, qk, f16s) = Q6_K;
        let ncols = 2048usize;
        let bpr = ncols / qk;
        let scy = ncols / 32;
        let ws: Vec<Vec<u8>> = (0..3).map(|m| g.blocks(bs, f16s, (nr[m] + 1) * bpr, 64)).collect();
        let y = g.q8_1((b - 1) * scy + bpr * qk / 32 + 8);
        let ds: Vec<Vec<u8>> = (0..3).map(|m| g.rng.bytes((b * nr[m] + 3) * 2)).collect();
        let s = g.s();
        let mut bufs = ws;
        bufs.push(y);
        bufs.extend(ds);
        g.case(&format!("mmvq fused_qkv q6_k_f16 rows={nr:?} b={b}"), bufs, &[], |r, p| unsafe {
            let f = if r { q::launch_mmvq_gguf_q6_k_f16_fused_qkv } else { rq::launch_mmvq_gguf_q6_k_f16_fused_qkv };
            f(p[0] as P, p[1] as P, p[2] as P, p[3] as P, p[4] as M, p[5] as M, p[6] as M, ncols as i32, nr[0] as i32, nr[1] as i32,
              nr[2] as i32, scy as i32, b as i32, s);
            0
        });
    }
    // MMQ: quantize (block_q8_1_mmq, DS4 / D4 layouts) then the dense GEMM; tmp_fixup (stream-k
    // scratch) is not compared: the reference leaves slots the fixup never reads nondeterministic.
    for &(fmt, name, k, nrows, ncols, td) in &[(Q4_K, "q4_k", 1024usize, 128usize, 64usize, 0i32), (Q8_0, "q8_0", 2048, 200, 17, 1), (Q4_K, "q4_k", 1536, 256, 300, 30)] {
        let (bs, qk, f16s) = fmt;
        let ds4 = name == "q4_k";
        let k_padded = k.div_ceil(512) * 512;
        let xf = g.vals(2, ncols * k, -4.0, 4.0, 0);
        let ybytes = ncols * (k_padded / 128) * 144 + 300 * 144;
        let y0 = g.rng.bytes(ybytes);
        let s = g.s();
        g.case(&format!("mmq quantize_q8_1_{} k={k} cols={ncols}", if ds4 { "DS4" } else { "D4" }), vec![xf.clone(), y0.clone()], &[], |r, p| unsafe {
            let f = match (ds4, r) {
                (true, false) => rq::launch_mmq_quantize_q8_1_DS4,
                (true, true) => q::launch_mmq_quantize_q8_1_DS4,
                (false, false) => rq::launch_mmq_quantize_q8_1_D4,
                (false, true) => q::launch_mmq_quantize_q8_1_D4,
            };
            f(p[0] as P, std::ptr::null(), p[1] as M, 0, k as i64, k as i64, 0, 0, k_padded as i64, ncols as i64, 1, 1, s);
            0
        });
        // activations for the GEMM: made once with the C quantizer
        let yd = Dev::new(&y0);
        let xd = Dev::new(&xf);
        sync("upload");
        unsafe {
            (if ds4 { rq::launch_mmq_quantize_q8_1_DS4 } else { rq::launch_mmq_quantize_q8_1_D4 })(
                xd.0 as P, std::ptr::null(), yd.0 as M, 0, k as i64, k as i64, 0, 0, k_padded as i64, ncols as i64, 1, 1, std::ptr::null_mut())
        };
        sync("activation quantize");
        let y = yd.read(ybytes);
        let mut x = g.blocks(bs, f16s, nrows * k / qk, 0);
        x.extend(g.rng.bytes(8 * bs + 256));
        let es = if td == 1 || td == 30 { 2 } else { 4 };
        let dst = g.rng.bytes(nrows * ncols * es + 64);
        let (nsm, smpbo) = (60i32, 101376i64);
        let fix = g.rng.bytes(nsm as usize * 128 * 128 * 4);
        let s = g.s();
        g.case(&format!("mmq_gguf_{name} k={k} rows={nrows} cols={ncols} type_dst={td}"), vec![fix, x, y, dst], &[0], |r, p| unsafe {
            let f = match (ds4, r) {
                (true, false) => rq::launch_mmq_gguf_q4_k,
                (true, true) => q::launch_mmq_gguf_q4_k,
                (false, false) => rq::launch_mmq_gguf_q8_0,
                (false, true) => q::launch_mmq_gguf_q8_0,
            };
            f(p[0] as M, p[1] as P, p[2] as P, p[3] as M, k as i64, nrows as i64, ncols as i64, (k / qk) as i64, nrows as i64, 1200, nsm,
              smpbo, 32, td, s);
            0
        });
    }
}

fn quant_a(g: &mut G) {
    g.family("quant_a (libmistralrsquant.a)");
    use ox::quant_a as q;
    // rotary: dtype 0 f16 / 1 bf16 / 2 f32, neox and gpt-j, null key
    for &(dtype, neox, nt, nh, nkv, hs, rd) in &[(1u32, 1i32, 5i64, 8i32, 2i32, 128i32, 64i32), (0, 0, 3, 4, 2, 64, 32), (2, 1, 2, 32, 8, 128, 32), (1, 0, 4, 2, 0, 64, 16)] {
        let ty = dtype as usize;
        let (qs, ks) = (nh as i64 * hs as i64, nkv as i64 * hs as i64);
        let qv = g.vals(ty, (nt * qs) as usize, -2.0, 2.0, 61);
        let kv = g.vals(ty, ((nt * ks) as usize).max(1), -2.0, 2.0, 0);
        let c = g.vals(ty, nt as usize * rd as usize, -1.0, 1.0, 0);
        let sn = g.vals(ty, nt as usize * rd as usize, -1.0, 1.0, 0);
        let st = g.s() as i64;
        g.case(&format!("rotary_embedding dtype={dtype} neox={neox} nt={nt} nh={nh} nkv={nkv} hs={hs} rd={rd}"), vec![qv, kv, c, sn], &[], |r, p| unsafe {
            let f = if r { q::rotary_embedding } else { rq::rotary_embedding };
            let key = if nkv == 0 { std::ptr::null() } else { p[1] as P };
            f(p[0] as P, key, p[2] as P, p[3] as P, neox, hs, nt, rd, nh, nkv, qs, ks, dtype, st);
            0
        });
    }
    // gemv bf16
    for &(m, k, batch, hb) in &[(33i32, 1024i32, 1i32, true), (17, 4096, 3, false), (5, 64, 8, true)] {
        let a = g.vals(1, (m * k) as usize, -1.0, 1.0, 0);
        let x = g.vals(1, (batch * k) as usize, -1.0, 1.0, 0);
        let bias = g.vals(1, m as usize, -1.0, 1.0, 0);
        let y = g.rng.bytes((batch * m) as usize * 2);
        let s = g.s();
        g.case(&format!("launch_gemv_bf16 m={m} k={k} batch={batch} bias={hb}"), vec![a, x, bias, y], &[], |r, p| unsafe {
            let f = if r { q::launch_gemv_bf16 } else { rq::launch_gemv_bf16 };
            f(p[0] as _, p[1] as _, p[2] as _, p[3] as _, m, k, batch, hb, s);
            0
        });
    }
    // quantize_q8_1 (f32) and the indexed-MoE expert GEMV on q4_k
    {
        let (kx, kxp, rows) = (1000i32, 1024i32, 3i32);
        let x = g.vals(2, (rows * kx) as usize, -8.0, 8.0, 0);
        let y = g.rng.bytes((rows * kxp) as usize / 32 * 36);
        let s = g.s();
        g.case("launch_quantize_q8_1 kx=1000 kxp=1024 rows=3", vec![x, y], &[], |r, p| unsafe {
            (if r { q::launch_quantize_q8_1 } else { rq::launch_quantize_q8_1 })(p[0] as _, p[1] as M, kx, kxp, (kxp + 255) / 256, rows, s);
            0
        });
    }
    for &(n, kk, batch, topk, ne, d1) in &[(33i32, 2048i32, 2i32, 3i32, 5usize, 1i32), (130, 2304, 1, 8, 8, 0)] {
        let (bs, qk, f16s) = Q4_K;
        let bpr = (kk as usize).div_ceil(qk);
        let kp = ((kk + 511) / 512 * 512).max((bpr * qk) as i32);
        let tasks = (batch * topk) as usize;
        let w = g.blocks(bs, f16s, ne * n as usize * bpr, 64);
        let ids: Vec<u32> = (0..tasks).map(|_| g.rng.below(ne as u64) as u32).collect();
        let in_rows = if d1 == 1 { batch as usize } else { tasks };
        let x = g.q8_1(in_rows * kp as usize / 32);
        let out = g.rng.bytes(tasks * n as usize * 4);
        let s = g.s();
        g.case(&format!("indexed_moe_forward_q4k_q8_1 n={n} k={kk} batch={batch} topk={topk} d1={d1}"), vec![w, x, bytes_of(&ids), out], &[], |r, p| unsafe {
            let f = if r { q::launch_indexed_moe_forward_q4k_q8_1 } else { rq::launch_indexed_moe_forward_q4k_q8_1 };
            f(p[0] as P, p[1] as P, p[2] as _, p[3] as _, n, kk, batch, topk, kp, d1, s);
            0
        });
    }
    // elementwise: fused GLU f16 (every activation id), gpt-oss swiglu bf16, bitwise and (no stream)
    for act in [0i32, 1, 2, 3, 99] {
        let n = 4097u32;
        let a = g.vals(0, n as usize, -10.0, 10.0, 53);
        let b = g.vals(0, n as usize, -3.0, 3.0, 0);
        let o = g.rng.bytes(n as usize * 2);
        let s = g.s();
        g.case(&format!("fused_glu_f16 n={n} act={act}"), vec![a, b, o], &[], |r, p| unsafe {
            (if r { q::fused_glu_f16 } else { rq::fused_glu_f16 })(p[0] as P, p[1] as P, p[2] as M, n, act, s);
            0
        });
    }
    {
        let n = 65537u32;
        let (a, b, o) = (g.vals(1, n as usize, -12.0, 12.0, 41), g.vals(1, n as usize, -12.0, 12.0, 0), g.rng.bytes(n as usize * 2));
        let s = g.s();
        g.case("gptoss_swiglu_bf16 n=65537 alpha=1.702 limit=7", vec![a, b, o], &[], |r, p| unsafe {
            (if r { q::gptoss_swiglu_bf16 } else { rq::gptoss_swiglu_bf16 })(p[0] as P, p[1] as P, p[2] as M, n, 1.702, 7.0, s);
            0
        });
        let (a, b, o) = (g.rng.bytes(4 * 1001), g.rng.bytes(4 * 1001), g.rng.bytes(4 * 1001));
        g.case("bitwise_and_u32 n=1001", vec![a, b, o], &[], |r, p| unsafe {
            (if r { q::bitwise_and_u32 } else { rq::bitwise_and_u32 })(p[0] as P, p[1] as P, p[2] as M, 1001);
            0
        });
    }
    // bitsandbytes (dequant.o) and hqq (no stream parameter)
    for &(n, bs) in &[(4096i32, 64i32), (5003, 256)] {
        let code = bytes_of(&g.f32s(256, -1.0, 1.0));
        let a = g.rng.bytes((n as usize).div_ceil(2));
        let nabs = (n as usize).div_ceil(1024) * 1024 / (bs as usize / 2) + 2;
        let am = bytes_of(&g.f32s(nabs, -4.0, 4.0));
        let out = g.rng.bytes(n as usize * 4);
        let s = g.s();
        g.case(&format!("dequantize_blockwise_f32_nf4 n={n} bs={bs}"), vec![code.clone(), a, am, out], &[], |r, p| unsafe {
            (if r { q::dequantize_blockwise_f32_nf4 } else { rq::dequantize_blockwise_f32_nf4 })(p[0] as _, p[1] as _, p[2] as _, p[3] as _, bs, n, s);
            0
        });
        let a = g.rng.bytes(n as usize);
        let am = bytes_of(&g.f32s(n as usize / 512 + 4, -4.0, 4.0));
        let out = g.rng.bytes(n as usize * 2);
        let s = g.s();
        g.case(&format!("dequantize_blockwise_bf16_int8 n={n} bs={bs}"), vec![code, a, am, out], &[], |r, p| unsafe {
            (if r { q::dequantize_blockwise_bf16_int8 } else { rq::dequantize_blockwise_bf16_int8 })(p[0] as _, p[1] as _, p[2] as _, p[3] as _, bs, n, s);
            0
        });
    }
    for &(h, w) in &[(64i32, 128i32), (17, 255)] {
        let n = (h * w) as usize;
        let wq = g.rng.bytes(n);
        let (sc, z) = (bytes_of(&g.f32s(w as usize, -2.0, 2.0)), bytes_of(&(0..w).map(|i| (i % 9) as f32).collect::<Vec<f32>>()));
        let out = g.rng.bytes(n * 2 * 4);
        g.case(&format!("hqq dequantize_4bit_u8_kernel_f32 h={h} w={w}"), vec![wq, sc, z, out], &[], |r, p| unsafe {
            (if r { q::dequantize_4bit_u8_kernel_f32 } else { rq::dequantize_4bit_u8_kernel_f32 })(p[0] as _, p[1] as _, p[2] as _, p[3] as _, h, w);
            0
        });
    }
}

fn quant_b(g: &mut G) {
    g.family("quant_b (libmistralrsquant.a)");
    use ox::quant_b as q;
    // AFQ 4-bit gs64 bf16: dequantize and qmv (no stream parameter: per-thread default stream)
    {
        let (rows, cols) = (17i32, 320i32);
        let w = g.rng.bytes((rows * cols / 8 * 4) as usize + 8);
        let ng = (rows * cols / 64) as usize + 2;
        let (sc, bi) = (g.vals(1, ng, -0.5, 0.5, 0), g.vals(1, ng, -1.0, 1.0, 0));
        let out = g.rng.bytes((rows * cols) as usize * 2 + 6);
        g.case("afq_dequantize_4bit_gs64_bf16 rows=17 cols=320", vec![w, sc, bi, out], &[], |r, p| unsafe {
            (if r { q::afq_dequantize_4bit_gs64_bf16 } else { rq::afq_dequantize_4bit_gs64_bf16 })(p[0] as _, p[1] as _, p[2] as _, p[3] as _, rows, cols);
            0
        });
    }
    for &(m, n, k) in &[(1i32, 7i32, 128i32), (3, 33, 192), (5, 9, 512)] {
        let w = g.rng.bytes((n * k / 8 * 4) as usize + 8);
        let ng = (n * k / 64) as usize + 2;
        let (sc, bi) = (g.vals(1, ng, -0.5, 0.5, 0), g.vals(1, ng, -1.0, 1.0, 0));
        let x = g.vals(1, (m * k) as usize, -2.0, 2.0, 0);
        let y = g.rng.bytes((m * n) as usize * 2 + 6);
        g.case(&format!("afq_qmv_4bit_gs64_bf16 m={m} n={n} k={k}"), vec![x, w, sc, bi, y], &[], |r, p| unsafe {
            (if r { q::afq_qmv_4bit_gs64_bf16 } else { rq::afq_qmv_4bit_gs64_bf16 })(p[0] as _, p[1] as _, p[2] as _, p[3] as _, p[4] as _, m, n, k);
            0
        });
    }
    // FP8: blockwise dequant (bf16), e4m3 -> f32
    for &(h, w, rs, by, bx) in &[(200i32, 300i32, 300i32, 128i32, 128i32), (33, 97, 101, 64, 32)] {
        let (gy, gx) = ((h + by - 1) / by, (w + bx - 1) / bx);
        let ss = gx + 1;
        let n = (h * rs) as usize;
        let wt = g.rng.bytes(n + 8);
        let sc = bytes_of(&g.f32s((gy * ss) as usize + 2, -2.0, 2.0));
        let out = g.rng.bytes(n * 2 + 8);
        let s = g.s();
        g.case(&format!("launch_dequant_fp8_blockwise_kernel_bf16 h={h} w={w} rs={rs} b={by}x{bx}"), vec![wt, sc, out], &[], |r, p| unsafe {
            (if r { q::launch_dequant_fp8_blockwise_kernel_bf16 } else { rq::launch_dequant_fp8_blockwise_kernel_bf16 })(
                p[0] as _, p[1] as _, p[2] as _, h, w, rs, ss, by, bx, s);
            0
        });
    }
    {
        let n = 4099usize;
        let (i, o) = (g.rng.bytes(n), g.rng.bytes(n * 4));
        let s = g.s();
        g.case("launch_fp8_to_f32_kernel n=4099", vec![i, o], &[], |r, p| unsafe {
            (if r { q::launch_fp8_to_f32_kernel } else { rq::launch_fp8_to_f32_kernel })(p[0] as _, p[1] as _, n, s);
            0
        });
    }
    // GPTQ exllama GEMM, 4-bit, one K block (K <= 128: no cross-block atomics)
    for &(m_count, n, k, groups, use_perm) in &[(1i32, 512i32, 128i32, 1i32, false), (4, 1024, 128, 4, true)] {
        let bit = 4;
        let qw = g.rng.bytes((k * bit / 32 * n) as usize * 4 + 16);
        let qz = g.rng.bytes(((groups * n * bit + 31) / 32) as usize * 4 + 16);
        let sc = g.vals(0, (groups * n) as usize + 8, -0.05, 0.05, 0);
        let mut perm: Vec<i32> = (0..k).collect();
        for i in (1..k as usize).rev() {
            let j = g.rng.below(i as u64 + 1) as usize;
            perm.swap(i, j);
        }
        let a = g.vals(0, (m_count * k) as usize, -2.0, 2.0, 0);
        let c = g.rng.bytes((m_count * n) as usize * 2 + 8);
        g.case(&format!("gemm_half_q_half_cuda_part bit=4 m={m_count} n={n} k={k} groups={groups} perm={use_perm}"),
               vec![a, qw, qz, sc, bytes_of(&perm), c], &[], |r, p| unsafe {
            let pp = if use_perm { p[4] as *const i32 } else { std::ptr::null() };
            (if r { q::gemm_half_q_half_cuda_part } else { rq::gemm_half_q_half_cuda_part })(
                p[0] as _, p[1] as _, p[2] as _, p[3] as _, pp, p[5] as _, m_count, n, k, m_count, groups, bit);
            0
        });
    }
    // MXFP4 matmul bf16: vecmat (M <= 4) and tiled
    for &(m, n, k, hb) in &[(3i32, 130i32, 256i32, true), (64, 128, 256, false), (100, 70, 160, true)] {
        let x = g.vals(1, (m * k) as usize, -2.0, 2.0, 0);
        let w = g.rng.bytes((n * k / 2) as usize);
        let ws: Vec<u8> = (0..(n * k / 32) as usize).map(|_| 118 + g.rng.below(18) as u8).collect();
        let bias = g.vals(1, n as usize, -2.0, 2.0, 0);
        let out = g.rng.bytes((m * n) as usize * 2 + 4);
        let s = g.s();
        g.case(&format!("launch_mxfp4_matmul_bf16 m={m} n={n} k={k} bias={hb}"), vec![x, w, ws, bias, out], &[], |r, p| unsafe {
            (if r { q::launch_mxfp4_matmul_bf16 } else { rq::launch_mxfp4_matmul_bf16 })(p[0] as _, p[1] as _, p[2] as _, p[3] as _, p[4] as _, m, n, k, hb, s);
            0
        });
    }
    // Marlin GPTQ 4-bit bf16 (the separate marlin PTX module)
    for &(m, k, n, gs) in &[(16i32, 256i32, 256i32, 128i32), (7, 128, 128, -1), (65, 1024, 1024, 128)] {
        let a = g.vals(1, (m * k) as usize, -2.0, 2.0, 0);
        let b = g.rng.bytes((k * n / 2) as usize);
        let ng = if gs == -1 { 1 } else { k / gs };
        let sc = g.vals(1, (ng * n) as usize, -0.05, 0.05, 0);
        let zp = g.rng.bytes((ng * n / 2) as usize + 16);
        let c = g.rng.bytes((m * n) as usize * 2);
        let ws = vec![0u8; ((n / 64) * 16 + 64) as usize * 4];
        let st = g.s() as i64;
        g.case(&format!("marlin_gptq_4bit_bf16 m={m} k={k} n={n} gs={gs}"), vec![a, b, sc, zp, c, ws], &[], |r, p| unsafe {
            (if r { q::marlin_gptq_4bit_bf16 } else { rq::marlin_gptq_4bit_bf16 })(
                p[0] as P, p[1] as P, p[2] as M, std::ptr::null_mut(), p[4] as M, m, k, n, p[5] as M, gs, st);
            0
        });
    }
}

/// Routing grouped by expert: (sorted_token_ids, expert_ids).
fn routing(g: &mut G, tokens: usize, topk: usize, e: usize) -> (Vec<i32>, Vec<i32>) {
    let mut ex: Vec<(i32, i32)> = (0..tokens * topk).map(|p| (g.rng.below(e as u64) as i32, p as i32)).collect();
    ex.sort();
    (ex.iter().map(|x| x.1).collect(), ex.iter().map(|x| x.0).collect())
}

fn core_cuda(g: &mut G) {
    g.family("core_cuda (libmistralrscuda.a)");
    use ox::core_cuda as c;
    for &n in &[0i64, 4099, 65536] {
        let len = n.max(1) as usize;
        let (a, b) = (g.rng.bytes(len), g.rng.bytes(len));
        let st = g.s() as i64;
        let label = format!("cuda_graph_copy_bytes n={n}");
        if n == 0 {
            // n == 0 copies nothing; only the return code is compared
            let (x, y) = unsafe { (rc::cuda_graph_copy_bytes(std::ptr::null(), std::ptr::null_mut(), 0, st), c::cuda_graph_copy_bytes(std::ptr::null(), std::ptr::null_mut(), 0, st)) };
            if x != y {
                g.failures.push(format!("{label}: return C {x} Rust {y}"));
            }
            continue;
        }
        g.case(&label, vec![a, b], &[], |r, p| unsafe {
            (if r { c::cuda_graph_copy_bytes } else { rc::cuda_graph_copy_bytes })(p[0] as P, p[1] as M, n, st) as i64
        });
    }
    for &(bh, seq, kd, vd) in &[(2usize, 3usize, 128usize, 64usize), (1, 65, 64, 128)] {
        let q = bytes_of(&g.f32s(bh * seq * kd, -0.3, 0.3));
        let k = bytes_of(&g.f32s(bh * seq * kd, -0.3, 0.3));
        let v = bytes_of(&g.f32s(bh * seq * vd, -2.0, 2.0));
        let gg = bytes_of(&g.f32s(bh * seq, -1.5, 0.05));
        let beta = bytes_of(&g.f32s(bh * seq, 0.0, 1.0));
        let state = bytes_of(&g.f32s(bh * kd * vd, -1.0, 1.0));
        let out = g.rng.bytes(bh * seq * vd * 4);
        let st = g.s() as i64;
        g.case(&format!("gated_delta_rule_recurrence bh={bh} seq={seq} k={kd} v={vd}"), vec![q, k, v, gg, beta, state, out], &[], |r, p| unsafe {
            (if r { c::gated_delta_rule_recurrence } else { rc::gated_delta_rule_recurrence })(
                p[0] as _, p[1] as _, p[2] as _, p[3] as _, p[4] as _, p[5] as _, p[6] as _, bh as i32, seq as i32, kd as i32, vd as i32, st);
            0
        });
    }
    // MoE GEMMs (moe_gemm_wmma: the libmistralrscuda.a definition the binary links)
    for &(wmma, dtype, tw, e, topk, t, n, k) in &[(true, 0i32, true, 8usize, 2usize, 5usize, 64usize, 64usize), (true, 1, false, 16, 4, 9, 96, 128),
                                                  (false, 1, true, 8, 2, 1, 64, 256), (false, 0, false, 4, 1, 3, 50, 2056)] {
        let (sorted, eids) = routing(g, t, topk, e);
        let m = t * topk;
        let rows = if tw { m } else { t };
        let input = g.vals(dtype as usize, rows * k, -1.0, 1.0, 0);
        let weights = g.vals(dtype as usize, e * n * k, -1.0, 1.0, 0);
        let tws = bytes_of(&g.f32s(m, 0.0, 1.0));
        let out = g.rng.bytes(m * n * 2);
        let st = g.s() as i64;
        let name = if wmma { "moe_gemm_wmma" } else { "moe_gemv" };
        g.case(&format!("{name} dtype={dtype} tw={tw} E={e} topk={topk} T={t} n={n} k={k}"), vec![input, weights, bytes_of(&sorted), bytes_of(&eids), tws, out], &[], |r, p| unsafe {
            let f = match (wmma, r) {
                (true, false) => rc::moe_gemm_wmma,
                (true, true) => c::moe_gemm_wmma,
                (false, false) => rc::moe_gemv,
                (false, true) => c::moe_gemv,
            };
            let twp = if tw { p[4] as *const f32 } else { std::ptr::null() };
            f(p[0] as P, p[1] as P, p[2] as _, p[3] as _, twp, p[5] as M, e as i32, topk as i32, m as i32, n as i32, k as i32, dtype, st);
            0
        });
    }
    for &(nrows, ncols) in &[(3usize, 1000usize), (2, 2048)] {
        let x = g.vals(0, nrows * ncols, -3.0, 3.0, 0);
        let res = g.vals(0, nrows * ncols, -3.0, 3.0, 0);
        let w = g.vals(0, ncols, -2.0, 2.0, 0);
        let sc = g.vals(0, 1, 0.25, 2.0, 0);
        let d = g.rng.bytes(nrows * ncols * 2);
        let st = g.s() as i64;
        g.case(&format!("rms_norm_residual_f16 rows={nrows} cols={ncols}"), vec![x, res, w, sc, d], &[], |r, p| unsafe {
            (if r { c::rms_norm_residual_f16 } else { rc::rms_norm_residual_f16 })(p[0] as P, p[1] as P, p[2] as P, p[3] as P, p[4] as M, nrows as i32, ncols as i32, 1e-6, st);
            0
        });
    }
    {
        let (nrows, ncols, k) = (4usize, 1000usize, 8usize);
        let x = bytes_of(&g.f32s(nrows * ncols, -5.0, 5.0));
        let (v, i) = (g.rng.bytes(nrows * k * 4), g.rng.bytes(nrows * k * 4));
        let st = g.s() as i64;
        g.case("topk_f32 rows=4 cols=1000 k=8", vec![x, v, i], &[], |r, p| unsafe {
            (if r { c::topk_f32 } else { rc::topk_f32 })(p[0] as P, p[1] as M, p[2] as M, nrows as i32, ncols as i32, k as i32, st);
            0
        });
        let (nrows, ncols) = (3usize, 100usize);
        let x = bytes_of(&g.f32s(nrows * ncols, -5.0, 5.0));
        let d = g.rng.bytes(nrows * ncols * 4);
        let st = g.s() as i64;
        g.case("asort_asc_f32 rows=3 cols=100", vec![x, d], &[], |r, p| unsafe {
            (if r { c::asort_asc_f32 } else { rc::asort_asc_f32 })(p[0] as M, p[1] as M, nrows as i32, ncols as i32, false, st);
            0
        });
    }
}

fn paged_attn(g: &mut G) {
    g.family("paged_attn_a (libmistralrspagedattention.a)");
    use ox::paged_attn_a as a;
    // reshape_and_cache, f16 cache
    {
        let (nt, nh, hs, bs, dtype) = (13usize, 4usize, 128usize, 16usize, 0u32);
        let x = 16 / 2;
        let (ks, vs) = (nh * hs + 128, nh * hs);
        let nb = (nt + 2).div_ceil(bs) + 2;
        let mut perm: Vec<i64> = (0..(nb * bs) as i64).collect();
        for i in (1..perm.len()).rev() {
            let j = g.rng.below(i as u64 + 1) as usize;
            perm.swap(i, j);
        }
        let slots: Vec<i64> = (0..nt).map(|t| if t % 5 == 3 { -1 } else { perm[t] }).collect();
        let key = g.vals(0, nt * ks, -2.0, 2.0, 31);
        let val = g.vals(0, nt * vs, -2.0, 2.0, 0);
        let kc = g.rng.bytes(nb * nh * hs * bs * 2);
        let vc = g.rng.bytes(nb * nh * hs * bs * 2);
        let s = g.s();
        g.case("reshape_and_cache nt=13 nh=4 hs=128 bs=16 f16", vec![key, val, kc, vc, bytes_of(&slots)], &[], |r, p| unsafe {
            (if r { a::reshape_and_cache } else { rp::reshape_and_cache })(p[0] as M, p[1] as M, p[2] as M, p[3] as M, p[4] as _, nt as i32, nh as i32,
                hs as i32, bs as i32, x as i32, ks as i32, vs as i32, s, dtype, dtype, std::ptr::null_mut(), std::ptr::null_mut());
            0
        });
    }
    // paged attention v1 (f16) and v2 (bf16)
    for &(v2, lens, nh, nkv, head, bs, alibi, sinks) in &[
        (false, &[1u32, 16, 17, 37][..], 4usize, 4usize, 128usize, 16usize, false, false),
        (false, &[300, 3][..], 8, 2, 64, 32, true, true),
        (true, &[700, 3][..], 8, 2, 128, 16, false, true),
        (true, &[1030][..], 2, 1, 256, 8, true, false),
    ] {
        let dtype = if v2 { 1usize } else { 0 };
        let x = 8;
        let nseq = lens.len();
        let max_ctx = *lens.iter().max().unwrap() as usize;
        let mbps = max_ctx.div_ceil(bs) + 1;
        let (qs, khs) = (nh * head, head * bs);
        let kbs = nkv * khs;
        let nblocks = nseq * mbps + 2;
        let mut perm: Vec<u32> = (0..nblocks as u32).collect();
        for i in (1..perm.len()).rev() {
            let j = g.rng.below(i as u64 + 1) as usize;
            perm.swap(i, j);
        }
        let bt: Vec<u32> = perm[..nseq * mbps].to_vec();
        let q = g.vals(dtype, nseq * qs, -1.0, 1.0, 0);
        let kc = g.vals(dtype, nblocks * kbs, -1.0, 1.0, 0);
        let vc = g.vals(dtype, nblocks * kbs, -2.0, 2.0, 0);
        let al: Vec<f32> = (0..nh).map(|h| [0.0625, -0.25, 0.0, 0.5][h % 4]).collect();
        let sk: Vec<f32> = (0..nh).map(|h| [0.5, -3.0, 12.0, 0.0][h % 4]).collect();
        let out = g.rng.bytes(nseq * nh * head * 2);
        let mp = max_ctx.div_ceil(512);
        let mut bufs = vec![out, q, kc, vc, bytes_of(&al), bytes_of(&bt), bytes_of(lens), bytes_of(&sk)];
        if v2 {
            bufs.push(g.rng.bytes(nseq * nh * mp * 4));
            bufs.push(g.rng.bytes(nseq * nh * mp * 4));
            bufs.push(g.rng.bytes(nseq * nh * mp * head * 2));
        }
        let sm = 1.0 / (head as f32).sqrt();
        let s = g.s();
        let _ = x;
        g.case(&format!("paged_attention_v{} lens={lens:?} nh={nh} nkv={nkv} head={head} bs={bs} alibi={alibi} sinks={sinks}", if v2 { 2 } else { 1 }), bufs, &[], |r, p| unsafe {
            let al = if alibi { p[4] as M } else { std::ptr::null_mut() };
            let skp = if sinks { p[7] as *const f32 } else { std::ptr::null() };
            if v2 {
                (if r { a::paged_attention_v2_bf16 } else { rp::paged_attention_v2_bf16 })(p[0] as M, p[8] as _, p[9] as _, p[10] as M, p[1] as M, p[2] as M,
                    p[3] as M, al, nkv as i32, sm, 1.0, p[5] as _, p[6] as _, bs as i32, max_ctx as i32, nseq as i32, nh as i32, head as i32,
                    mbps as i32, qs as i32, kbs as i32, khs as i32, s, 1, std::ptr::null_mut(), std::ptr::null_mut(), skp)
            } else {
                (if r { a::paged_attention_v1_f16 } else { rp::paged_attention_v1_f16 })(p[0] as M, p[1] as M, p[2] as M, p[3] as M, al, nkv as i32, sm,
                    1.0, p[5] as _, p[6] as _, bs as i32, max_ctx as i32, nseq as i32, nh as i32, head as i32, mbps as i32, qs as i32, kbs as i32,
                    khs as i32, s, 0, std::ptr::null_mut(), std::ptr::null_mut(), skp)
            }
            0
        });
    }
    // flash_attn_sinks f16
    for &(b, ql, kl, nh, nkv, hd, w) in &[(2usize, 13usize, 13usize, 4usize, 2usize, 128usize, 0i32), (1, 9, 150, 4, 4, 64, 33)] {
        let q = g.vals(0, b * nh * ql * hd, -2.0, 2.0, 0);
        let k = g.vals(0, b * nkv * kl * hd, -2.0, 2.0, 0);
        let v = g.vals(0, b * nkv * kl * hd, -3.0, 3.0, 0);
        let o = g.rng.bytes(b * nh * ql * hd * 2);
        let sk: Vec<f32> = (0..nh).map(|h| [0.5, -3.0, 12.0, 0.0][h % 4]).collect();
        let s = g.s();
        g.case(&format!("flash_attn_sinks_f16 b={b} q={ql} kv={kl} nh={nh} nkv={nkv} hd={hd} w={w}"), vec![q, k, v, o, bytes_of(&sk)], &[], |r, p| unsafe {
            (if r { a::flash_attn_sinks_f16 } else { rp::flash_attn_sinks_f16 })(p[0] as P, p[1] as P, p[2] as P, p[3] as M, p[4] as _,
                1.0 / (hd as f32).sqrt(), b as i32, ql as i32, kl as i32, nh as i32, nkv as i32, hd as i32, w, s);
            0
        });
    }

    g.family("paged_attn_b flashinfer (libmistralrspagedattention.a)");
    use ox::paged_attn_b as fb;
    // flashinfer decode: f16 (dtype 0) / bf16 (1), non-split and split-KV with padding tiles
    for &(dtype, hd, group, nkv, ps, split) in &[(0u32, 128usize, 4usize, 2usize, 16usize, None), (1, 128, 1, 2, 16, Some(2usize)), (0, 64, 8, 1, 7, Some(1))] {
        let lens = vec![1usize, ps, ps + 1, 100];
        let b = lens.len();
        let nqo = nkv * group;
        // paged CSR over a shuffled page pool
        let blocks: Vec<usize> = lens.iter().map(|l| l.div_ceil(ps)).collect();
        let total: usize = blocks.iter().sum();
        let num_pages = total + 2;
        let mut pool: Vec<i32> = (0..num_pages as i32).collect();
        for i in (1..pool.len()).rev() {
            let j = g.rng.below(i as u64 + 1) as usize;
            pool.swap(i, j);
        }
        let (mut indptr, mut indices, mut last) = (vec![0i32], vec![], vec![]);
        let mut kk = 0;
        for (nb, &l) in blocks.iter().zip(&lens) {
            for _ in 0..*nb {
                indices.push(pool[kk]);
                kk += 1;
            }
            indptr.push(indices.len() as i32);
            last.push(if *nb == 0 { 0 } else { (l - (nb - 1) * ps) as i32 });
        }
        // plan: request / tile indices per chunk of `split` pages, 2 padding tiles when split
        let chunk_pages = split.unwrap_or(usize::MAX);
        let (mut req, mut tiles, mut o_indptr) = (vec![], vec![], vec![0i32]);
        for (bi, l) in lens.iter().enumerate() {
            let nc = l.div_ceil(ps).max(1).div_ceil(chunk_pages);
            for t in 0..nc {
                req.push(bi as i32);
                tiles.push(t as i32);
            }
            o_indptr.push(req.len() as i32);
        }
        let valid = req.len();
        let pad = if split.is_some() { 2 } else { 0 };
        req.resize(valid + pad, 0);
        tiles.resize(valid + pad, 0);
        let mut mask = vec![1u8; valid];
        mask.resize(valid + pad, 0);
        let chunk = (split.unwrap_or(1) * ps) as i32;
        let padded = req.len();
        let use_tmp = padded > b;
        let t = dtype as usize;
        let q = g.vals(t, b * nqo * hd, -2.0, 2.0, 0);
        let kc = g.vals(t, num_pages * nkv * ps * hd, -2.0, 2.0, 0);
        let vc = g.vals(t, num_pages * nkv * ps * hd, -2.0, 2.0, 0);
        let out = g.rng.bytes(b * nqo * hd * 2);
        let tv = g.rng.bytes(padded * nqo * hd * 2);
        let ts = g.rng.bytes(padded * nqo * 4);
        let bufs = vec![q, kc, vc, bytes_of(&indptr), bytes_of(&indices), bytes_of(&last), bytes_of(&req), bytes_of(&tiles), bytes_of(&o_indptr),
                        bytes_of(&[chunk]), mask, out, tv, ts];
        let sm = 1.0 / (hd as f32).sqrt();
        let s = g.s();
        g.case(&format!("flashinfer_decode dtype={dtype} hd={hd} group={group} nkv={nkv} ps={ps} split={split:?}"), bufs, &[], |r, p| unsafe {
            let (tvp, tsp) = if use_tmp { (p[12] as M, p[13] as M) } else { (std::ptr::null_mut(), std::ptr::null_mut()) };
            (if r { fb::flashinfer_decode } else { rp::flashinfer_decode })(p[0] as M, p[1] as M, p[2] as M, p[3] as _, p[4] as _, p[5] as _,
                p[6] as _, p[7] as _, p[8] as _, p[9] as _, p[10] as _, p[11] as M, tvp, tsp, b as i32, padded as i32, nqo as i32, nkv as i32,
                hd as i32, ps as i32, (nqo * hd) as i32, hd as i32, sm, -1, 0.0, dtype, s) as i64
        });
    }
    {
        let (nt, nh, hs, bs) = (9usize, 2usize, 128usize, 16usize);
        let nb = nt.div_ceil(bs) + 2;
        let slots: Vec<i64> = (0..nt as i64).map(|t| if t == 4 { -1 } else { (t * 3 + 1) % (nb * bs) as i64 }).collect();
        let key = g.vals(1, nt * nh * hs, -2.0, 2.0, 0);
        let val = g.vals(1, nt * nh * hs, -2.0, 2.0, 0);
        let kc = g.rng.bytes(nb * bs * nh * hs * 2);
        let vc = g.rng.bytes(nb * bs * nh * hs * 2);
        let s = g.s();
        g.case("reshape_and_cache_flashinfer nt=9 nh=2 hs=128 bs=16 bf16", vec![key, val, kc, vc, bytes_of(&slots)], &[], |r, p| unsafe {
            (if r { fb::reshape_and_cache_flashinfer } else { rp::reshape_and_cache_flashinfer })(p[0] as M, p[1] as M, p[2] as M, p[3] as M, p[4] as _,
                nt as i32, nh as i32, hs as i32, bs as i32, (nh * hs) as i32, (nh * hs) as i32, 1, s);
            0
        });
    }
}

fn candle_moe(g: &mut G) {
    g.family("candle_moe (libmoe.a)");
    use ox::candle_moe as cm;
    const LAYOUT: [(usize, usize, &[usize]); 6] = [(32, 34, &[0]), (256, 144, &[0, 2]), (256, 84, &[80, 82]), (256, 110, &[108]), (256, 176, &[0, 2]), (256, 210, &[208])];
    // decode: f32 input, q8_1-quantized in the launcher
    for &(t, tw, e, topk, tokens, n, kb) in &[(1usize, true, 8usize, 2usize, 3usize, 64usize, 2usize), (0, false, 16, 4, 2, 7, 3), (5, true, 5, 2, 40, 36, 2)] {
        let (qk, bs, sc) = LAYOUT[t];
        let k = if qk == 32 { kb * 96 + 32 } else { kb * qk };
        let (sorted, eids) = routing(g, tokens, topk, e);
        let m = tokens * topk;
        let rows = if tw { m } else { tokens };
        let input = g.vals(2, rows * k, -4.0, 4.0, 0);
        let w = g.blocks(bs, sc, e * n * k / qk, 0);
        let tws = bytes_of(&g.f32s(m, 0.0, 1.0));
        let out = g.rng.bytes(m * n * 4);
        let st = g.s() as i64;
        g.case(&format!("moe_gemm_gguf t={t} tw={tw} E={e} topk={topk} T={tokens} n={n} k={k}"), vec![input, w, bytes_of(&sorted), bytes_of(&eids), tws, out], &[], |r, p| unsafe {
            let twp = if tw { p[4] as *const f32 } else { std::ptr::null() };
            (if r { cm::moe_gemm_gguf } else { rm::moe_gemm_gguf })(p[0] as _, p[1] as P, p[2] as _, p[3] as _, twp, p[5] as M, e as i32, topk as i32,
                m as i32, n as i32, k as i32, t as i32, st);
            0
        });
    }
    // prefill: f16 / bf16 input
    for &(dt, t, tw, e, topk, tokens, n, kb) in &[(0i32, 0usize, false, 8usize, 2usize, 150usize, 96usize, 2usize), (1, 1, true, 4, 1, 9, 50, 1), (0, 3, true, 3, 3, 4, 33, 4)] {
        let (qk, bs, sc) = LAYOUT[t];
        let k = if qk == 32 { kb * 96 + 32 } else { kb * qk };
        let (sorted, eids) = routing(g, tokens, topk, e);
        let m = tokens * topk;
        let rows = if tw { m } else { tokens };
        let input = g.vals(dt as usize, rows * k, -2.0, 2.0, 0);
        let w = g.blocks(bs, sc, e * n * k / qk, 0);
        let tws = bytes_of(&g.f32s(m, 0.0, 1.0));
        let out = g.rng.bytes(m * n * 4);
        let st = g.s() as i64;
        g.case(&format!("moe_gemm_gguf_prefill dt={dt} t={t} tw={tw} E={e} topk={topk} T={tokens} n={n} k={k}"), vec![input, w, bytes_of(&sorted), bytes_of(&eids), tws, out], &[], |r, p| unsafe {
            let twp = if tw { p[4] as *const f32 } else { std::ptr::null() };
            (if r { cm::moe_gemm_gguf_prefill } else { rm::moe_gemm_gguf_prefill })(p[0] as P, p[1] as _, p[2] as _, p[3] as _, twp, p[5] as M, e as i32,
                topk as i32, m as i32, n as i32, k as i32, dt, t as i32, st);
            0
        });
    }
}

fn main() {
    let t0 = std::time::Instant::now();
    unsafe {
        ck(drv::cuInit(0), "cuInit");
        let mut dev = 0;
        ck(drv::cuDeviceGet(&mut dev, 0), "cuDeviceGet");
        let mut ctx = std::ptr::null_mut();
        ck(drv::cuDevicePrimaryCtxRetain(&mut ctx, dev), "cuDevicePrimaryCtxRetain");
        ck(drv::cuCtxSetCurrent(ctx), "cuCtxSetCurrent");
        drv::cudaFree(std::ptr::null_mut()); // bind cudart to the same (primary) context
    }
    let mut stream = std::ptr::null_mut();
    unsafe { ck(drv::cuStreamCreate(&mut stream, 0), "cuStreamCreate") };
    let mut g = G { rng: Rng(0x7171_7a7a_0bad_f00d), stream, k: 0, fam: "", stats: BTreeMap::new(), failures: vec![] };
    let only = std::env::var("GATE_ONLY").unwrap_or_default();
    let fams: [(&str, fn(&mut G)); 6] =
        [("quant_c", quant_c), ("quant_a", quant_a), ("quant_b", quant_b), ("core_cuda", core_cuda), ("paged_attn", paged_attn), ("candle_moe", candle_moe)];
    for (name, f) in fams {
        if only.is_empty() || only.split(',').any(|o| o == name) {
            f(&mut g);
        }
    }
    let (mut cases, mut calls, mut bytes, mut fails, mut nd) = (0, 0, 0, 0, 0);
    for (fam, &(c, l, b, f, n)) in &g.stats {
        println!("  {fam}: {c} cases, {l} launcher calls, {b} bytes compared, {f} failing{}", if n > 0 { format!(", {n} nondeterministic reference buffers skipped") } else { String::new() });
        cases += c;
        calls += l;
        bytes += b;
        fails += f;
        nd += n;
    }
    for f in g.failures.iter().take(40) {
        println!("  FAIL {f}");
    }
    let ok = g.failures.is_empty();
    let _ = fails;
    println!(
        "titan-oxide-ffi gate: {cases} cases, {calls} launcher calls, {bytes} bytes compared, {nd} nondeterministic buffers skipped, {} failing, {:.1}s -> {}",
        g.failures.len(), t0.elapsed().as_secs_f64(), if ok { "PASS" } else { "FAIL" }
    );
    std::process::exit(if ok { 0 } else { 1 });
}
