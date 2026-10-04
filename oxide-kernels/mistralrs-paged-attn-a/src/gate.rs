//! Launcher-level differential gate for mistralrs-paged-attn group 1: every extern "C" launcher is
//! called twice on identical inputs in the same (primary) context -- once the REAL C launcher from
//! libmistralrspagedattention.a (nvcc kernels, -O3 --use_fast_math), once the pure-Rust twin in
//! `crate::launch` (oxide kernels) -- and every byte of every buffer the call can see is compared:
//! outputs, in-place caches, inputs (no stray writes) and a random guard region after each buffer.
use crate::launch as ox;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use kdiff::{Rng, Tally, as_bytes};
use std::ffi::c_void;
use std::sync::Arc;

/// The C launchers (libmistralrspagedattention.a).
pub mod cref {
    use std::ffi::c_void;
    unsafe extern "C" {
        /// cudart: clears the runtime's last-error slot.
        pub fn cudaGetLastError() -> i32;
        pub fn reshape_and_cache(
            key: *mut c_void, value: *mut c_void, key_cache: *mut c_void, value_cache: *mut c_void, slot_mapping: *mut i64, num_tokens: i32,
            num_heads: i32, head_size: i32, block_size: i32, x: i32, key_stride: i32, value_stride: i32, stream: *mut c_void, dtype: u32,
            cache_dtype: u32, k_scale: *mut f32, v_scale: *mut f32,
        );
        pub fn gather_kv_cache(
            key_cache: *mut c_void, value_cache: *mut c_void, k_out: *mut c_void, v_out: *mut c_void, k_scale: *mut c_void,
            v_scale: *mut c_void, block_table: *const i32, cu_seq_lens: *const i32, num_tokens: i32, num_seqs: i32, block_size: i32,
            block_table_stride: i32, num_kv_heads: i32, head_size: i32, x: i32, stream: *mut c_void, out_dtype: u32, cache_dtype: u32,
        );
        pub fn update_kv_scales_f32(k: *mut c_void, v: *mut c_void, n: i64, ks: *mut f32, vs: *mut f32, stream: i64);
        pub fn update_kv_scales_f16(k: *mut c_void, v: *mut c_void, n: i64, ks: *mut f32, vs: *mut f32, stream: i64);
        pub fn update_kv_scales_bf16(k: *mut c_void, v: *mut c_void, n: i64, ks: *mut f32, vs: *mut f32, stream: i64);
        pub fn copy_blocks_f32(k: *mut i64, v: *mut i64, m: *const i64, nl: i32, np: i32, nk: i32, nv: i32, stream: i64);
        pub fn copy_blocks_f16(k: *mut i64, v: *mut i64, m: *const i64, nl: i32, np: i32, nk: i32, nv: i32, stream: i64);
        pub fn copy_blocks_bf16(k: *mut i64, v: *mut i64, m: *const i64, nl: i32, np: i32, nk: i32, nv: i32, stream: i64);
        pub fn copy_blocks_u8(k: *mut i64, v: *mut i64, m: *const i64, nl: i32, np: i32, nk: i32, nv: i32, stream: i64);
        pub fn concat_and_cache_mla(
            ckv: *mut c_void, k_pe: *mut c_void, ckv_cache: *mut c_void, kpe_cache: *mut c_void, slot_mapping: *mut i64, num_tokens: i32,
            kv_lora_rank: i32, kpe_head_dim: i32, block_size: i32, ckv_stride: i32, kpe_stride: i32, stream: *mut c_void, dtype: u32,
        );
        pub fn gather_mla_cache(
            ckv_cache: *mut c_void, kpe_cache: *mut c_void, ckv_out: *mut c_void, kpe_out: *mut c_void, block_table: *const i32,
            cu_seq_lens: *const i32, token_to_seq: *const i32, num_tokens: i32, block_size: i32, block_table_stride: i32, kv_lora_rank: i32,
            kpe_head_dim: i32, stream: *mut c_void, dtype: u32,
        );
        pub fn paged_attention_v1_f32(
            out: *mut c_void, query: *mut c_void, key_cache: *mut c_void, value_cache: *mut c_void, alibi_slopes: *mut c_void, num_kv_heads: i32,
            scale: f32, softcapping: f32, block_tables: *mut u32, context_lens: *mut u32, block_size: i32, max_context_len: i32, num_seqs: i32,
            num_heads: i32, head_size: i32, max_num_blocks_per_seq: i32, q_stride: i32, kv_block_stride: i32, kv_head_stride: i32,
            stream: *mut c_void, cache_dtype: u32, k_scale: *mut f32, v_scale: *mut f32, sinks: *const f32,
        );
        pub fn paged_attention_v1_f16(
            out: *mut c_void, query: *mut c_void, key_cache: *mut c_void, value_cache: *mut c_void, alibi_slopes: *mut c_void, num_kv_heads: i32,
            scale: f32, softcapping: f32, block_tables: *mut u32, context_lens: *mut u32, block_size: i32, max_context_len: i32, num_seqs: i32,
            num_heads: i32, head_size: i32, max_num_blocks_per_seq: i32, q_stride: i32, kv_block_stride: i32, kv_head_stride: i32,
            stream: *mut c_void, cache_dtype: u32, k_scale: *mut f32, v_scale: *mut f32, sinks: *const f32,
        );
        pub fn paged_attention_v1_bf16(
            out: *mut c_void, query: *mut c_void, key_cache: *mut c_void, value_cache: *mut c_void, alibi_slopes: *mut c_void, num_kv_heads: i32,
            scale: f32, softcapping: f32, block_tables: *mut u32, context_lens: *mut u32, block_size: i32, max_context_len: i32, num_seqs: i32,
            num_heads: i32, head_size: i32, max_num_blocks_per_seq: i32, q_stride: i32, kv_block_stride: i32, kv_head_stride: i32,
            stream: *mut c_void, cache_dtype: u32, k_scale: *mut f32, v_scale: *mut f32, sinks: *const f32,
        );
        pub fn paged_attention_v2_f32(
            out: *mut c_void, exp_sums: *mut f32, max_logits: *mut f32, tmp_out: *mut c_void, query: *mut c_void, key_cache: *mut c_void,
            value_cache: *mut c_void, alibi_slopes: *mut c_void, num_kv_heads: i32, scale: f32, softcapping: f32, block_tables: *mut u32,
            context_lens: *mut u32, block_size: i32, max_context_len: i32, num_seqs: i32, num_heads: i32, head_size: i32,
            max_num_blocks_per_seq: i32, q_stride: i32, kv_block_stride: i32, kv_head_stride: i32, stream: *mut c_void, cache_dtype: u32,
            k_scale: *mut f32, v_scale: *mut f32, sinks: *const f32,
        );
        pub fn paged_attention_v2_f16(
            out: *mut c_void, exp_sums: *mut f32, max_logits: *mut f32, tmp_out: *mut c_void, query: *mut c_void, key_cache: *mut c_void,
            value_cache: *mut c_void, alibi_slopes: *mut c_void, num_kv_heads: i32, scale: f32, softcapping: f32, block_tables: *mut u32,
            context_lens: *mut u32, block_size: i32, max_context_len: i32, num_seqs: i32, num_heads: i32, head_size: i32,
            max_num_blocks_per_seq: i32, q_stride: i32, kv_block_stride: i32, kv_head_stride: i32, stream: *mut c_void, cache_dtype: u32,
            k_scale: *mut f32, v_scale: *mut f32, sinks: *const f32,
        );
        pub fn paged_attention_v2_bf16(
            out: *mut c_void, exp_sums: *mut f32, max_logits: *mut f32, tmp_out: *mut c_void, query: *mut c_void, key_cache: *mut c_void,
            value_cache: *mut c_void, alibi_slopes: *mut c_void, num_kv_heads: i32, scale: f32, softcapping: f32, block_tables: *mut u32,
            context_lens: *mut u32, block_size: i32, max_context_len: i32, num_seqs: i32, num_heads: i32, head_size: i32,
            max_num_blocks_per_seq: i32, q_stride: i32, kv_block_stride: i32, kv_head_stride: i32, stream: *mut c_void, cache_dtype: u32,
            k_scale: *mut f32, v_scale: *mut f32, sinks: *const f32,
        );
        pub fn flash_attn_sinks_f16(q: *const c_void, k: *const c_void, v: *const c_void, o: *mut c_void, s: *const f32, sc: f32, b: i32, ql: i32, kl: i32, nh: i32, nkv: i32, hd: i32, w: i32, st: *mut c_void);
        pub fn flash_attn_sinks_bf16(q: *const c_void, k: *const c_void, v: *const c_void, o: *mut c_void, s: *const f32, sc: f32, b: i32, ql: i32, kl: i32, nh: i32, nkv: i32, hd: i32, w: i32, st: *mut c_void);
        pub fn flash_attn_sinks_f32(q: *const c_void, k: *const c_void, v: *const c_void, o: *mut c_void, s: *const f32, sc: f32, b: i32, ql: i32, kl: i32, nh: i32, nkv: i32, hd: i32, w: i32, st: *mut c_void);
        pub fn flash_attn_sinks_varlen_f16(q: *const c_void, k: *const c_void, v: *const c_void, o: *mut c_void, s: *const f32, cq: *const u32, ck: *const u32, sc: f32, b: i32, mq: i32, nh: i32, nkv: i32, hd: i32, w: i32, st: *mut c_void);
        pub fn flash_attn_sinks_varlen_bf16(q: *const c_void, k: *const c_void, v: *const c_void, o: *mut c_void, s: *const f32, cq: *const u32, ck: *const u32, sc: f32, b: i32, mq: i32, nh: i32, nkv: i32, hd: i32, w: i32, st: *mut c_void);
        pub fn flash_attn_sinks_varlen_f32(q: *const c_void, k: *const c_void, v: *const c_void, o: *mut c_void, s: *const f32, cq: *const u32, ck: *const u32, sc: f32, b: i32, mq: i32, nh: i32, nkv: i32, hd: i32, w: i32, st: *mut c_void);
    }
}

/// Bytes of random guard data appended to every buffer (compared like the data).
const GUARD: usize = 64;

pub struct G {
    pub ctx: Arc<CudaContext>,
    pub streams: Vec<Arc<CudaStream>>,
    pub rng: Rng,
    pub t: Tally,
    pub calls: usize,
    pub family_calls: usize,
}

/// Device copies of a case's buffers for one side (C or Rust).
pub struct Side {
    pub bufs: Vec<DeviceBuffer<u8>>,
}
impl Side {
    pub fn p(&self, i: usize) -> *mut c_void {
        self.bufs[i].cu_deviceptr() as *mut c_void
    }
}

impl G {
    /// Stream `i`: 0 = null (the per-thread default stream of the -default-stream per-thread
    /// reference), 1 = a created stream.
    pub fn stream(&self, i: usize) -> *mut c_void {
        if i == 0 { std::ptr::null_mut() } else { self.streams[1].cu_stream() as *mut c_void }
    }

    fn upload(&self, bufs: &[Vec<u8>]) -> Side {
        Side { bufs: bufs.iter().map(|b| DeviceBuffer::from_host(&self.streams[0], b).unwrap()).collect() }
    }

    /// Run `f` for the C side then the Rust side, each on private copies of `bufs` (each followed by
    /// a random guard), and compare every byte of every buffer afterwards.
    pub fn case(&mut self, label: &str, bufs: Vec<Vec<u8>>, f: impl Fn(bool, &Side)) {
        let bufs: Vec<Vec<u8>> = bufs
            .into_iter()
            .map(|mut b| {
                b.extend(self.rng.bytes(GUARD));
                b
            })
            .collect();
        let (a, b) = (self.upload(&bufs), self.upload(&bufs));
        self.ctx.synchronize().unwrap();
        unsafe { cref::cudaGetLastError() };
        f(false, &a);
        self.ctx.synchronize().unwrap_or_else(|e| panic!("{label}: C side: {e:?}"));
        f(true, &b);
        self.ctx.synchronize().unwrap_or_else(|e| panic!("{label}: Rust side: {e:?}"));
        let down = |s: &Side| -> Vec<Vec<u8>> { s.bufs.iter().map(|x| x.to_host_vec(&self.streams[0]).unwrap()).collect() };
        let (x, y) = (down(&a), down(&b));
        let mut d = kdiff::Diff { bytes: 0, differing: 0, first: None };
        for (k, (x, y)) in x.iter().zip(&y).enumerate() {
            d.bytes += x.len();
            for i in 0..x.len() {
                if x[i] != y[i] {
                    d.differing += 1;
                    if d.first.is_none() {
                        d.first = Some((k, i, x[i], y[i]));
                    }
                }
            }
        }
        self.calls += 2;
        self.family_calls += 2;
        self.t.record(label, &d);
    }

    pub fn f32s(&mut self, n: usize) -> Vec<u8> {
        as_bytes(&self.rng.f32s(n))
    }
    /// Values of dtype `t` (0 f16, 1 bf16, 2 f32): f32s with edge cases (converted), plus raw
    /// 16-bit patterns for the half types.
    pub fn vals(&mut self, t: u32, n: usize) -> Vec<u8> {
        let v = self.rng.f32s(n);
        match t {
            2 => as_bytes(&v),
            _ => {
                let h: Vec<u16> = v
                    .iter()
                    .map(|&x| {
                        let r = self.rng.next();
                        if r % 8 == 0 {
                            r as u16
                        } else if t == 1 {
                            (x.to_bits() >> 16) as u16
                        } else {
                            f32_to_f16(x)
                        }
                    })
                    .collect();
                as_bytes(&h)
            }
        }
    }
    /// Well-behaved values (|x| < `amp`, a few exact zeros) of dtype `t`, for softmax-style inputs
    /// where special values would make every output NaN.
    pub fn nice(&mut self, t: u32, n: usize, amp: f32) -> Vec<u8> {
        let v: Vec<f32> = (0..n)
            .map(|_| {
                let r = self.rng.next();
                if r % 64 == 0 { 0.0 } else { ((r >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 * amp }
            })
            .collect();
        match t {
            2 => as_bytes(&v),
            1 => as_bytes(&v.iter().map(|x| (x.to_bits() >> 16) as u16).collect::<Vec<u16>>()),
            _ => as_bytes(&v.iter().map(|&x| f32_to_f16(x)).collect::<Vec<u16>>()),
        }
    }
    /// Finite values of dtype `t` with a wide exponent range: +-m * 2^e, e in [-20, emax].
    pub fn wide(&mut self, t: u32, n: usize, emax: i32) -> Vec<u8> {
        let v: Vec<f32> = (0..n)
            .map(|_| {
                let r = self.rng.next();
                let m = 1.0 + ((r >> 20) & 0xffff) as f32 / 65536.0;
                let e = -20 + ((r >> 8) % (emax + 21) as u64) as i32;
                let x = m * (2.0f32).powi(e);
                if r & 1 == 1 { -x } else { x }
            })
            .collect();
        match t {
            2 => as_bytes(&v),
            1 => as_bytes(&v.iter().map(|x| (x.to_bits() >> 16) as u16).collect::<Vec<u16>>()),
            _ => as_bytes(&v.iter().map(|&x| f32_to_f16(x)).collect::<Vec<u16>>()),
        }
    }
    /// Random fp8 e4m3 bytes (every class: +-0, denormals, NaN 0x7f/0xff).
    pub fn fp8s(&mut self, n: usize) -> Vec<u8> {
        self.rng.bytes(n)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.rng.next() % n as u64) as usize
    }
    /// A random permutation of 0..n.
    pub fn perm(&mut self, n: usize) -> Vec<usize> {
        let mut v: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
        v
    }

    pub fn family(&mut self, name: &str) {
        println!("  {name}: {} launcher calls", self.family_calls);
        self.family_calls = 0;
    }
}

/// f32 -> f16 bits (truncating; only used to make test data).
pub fn f32_to_f16(x: f32) -> u16 {
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

pub fn esize(dtype: u32) -> usize {
    if dtype == 2 { 4 } else { 2 }
}
pub fn i32s(v: &[i32]) -> Vec<u8> {
    as_bytes(v)
}
pub fn i64s(v: &[i64]) -> Vec<u8> {
    as_bytes(v)
}
pub fn f32b(v: &[f32]) -> Vec<u8> {
    as_bytes(v)
}

/// Scale values for fp8 caches: typical, tiny, huge and degenerate ones.
const SCALES: [f32; 10] = [0.0371, 1.0, 0.25, 7.5, 1e-3, 3e-39, 0.0, -0.5, f32::INFINITY, f32::NAN];

// ================================================================================================
// cache kernels

fn reshape_and_cache(g: &mut G) {
    // (num_tokens, num_heads, head_size, block_size, key_stride_extra, value_stride_extra)
    let shapes: [(usize, usize, usize, usize, usize, usize); 7] = [
        (1, 1, 64, 16, 0, 0),
        (5, 2, 80, 8, 0, 160),
        (13, 4, 128, 16, 128, 0),
        (7, 8, 128, 32, 1024, 2048),
        (3, 3, 256, 8, 0, 256),
        (9, 1, 512, 16, 64, 0),
        (33, 2, 96, 32, 0, 0),
    ];
    for (si, &(nt, nh, hs, bs, kx, vx)) in shapes.iter().enumerate() {
        for dtype in 0..3u32 {
            for fp8 in [false, true] {
                let cache_dtype = if fp8 { 3 } else { dtype };
                let ce = if fp8 { 1 } else { esize(dtype) };
                let x = 16 / ce;
                let ks = nh * hs + kx;
                let vs = nh * hs + vx;
                let num_blocks = (nt + 2).div_ceil(bs) + 2;
                // slots: distinct, some padding (-1)
                let perm = g.perm(num_blocks * bs);
                let slots: Vec<i64> = (0..nt).map(|t| if t % 5 == 3 { -1 } else { perm[t] as i64 }).collect();
                let key = g.vals(dtype, nt * ks);
                let value = g.vals(dtype, nt * vs);
                let kc = g.rng.bytes(num_blocks * nh * hs * bs * ce);
                let vc = g.rng.bytes(num_blocks * nh * hs * bs * ce);
                let sc = [SCALES[si % SCALES.len()], SCALES[(si + 3 + dtype as usize) % SCALES.len()]];
                let s = g.stream(si % 2);
                let label = format!("reshape_and_cache nt={nt} nh={nh} hs={hs} bs={bs} dtype={dtype} cache={cache_dtype}");
                g.case(&label, vec![key, value, kc, vc, i64s(&slots), f32b(&sc[..1]), f32b(&sc[1..])], |rust, b| unsafe {
                    let f = if rust { ox::reshape_and_cache } else { cref::reshape_and_cache };
                    let (kp, vp) = if fp8 { (b.p(5) as *mut f32, b.p(6) as *mut f32) } else { (std::ptr::null_mut(), std::ptr::null_mut()) };
                    f(b.p(0), b.p(1), b.p(2), b.p(3), b.p(4) as _, nt as i32, nh as i32, hs as i32, bs as i32, x as i32, ks as i32, vs as i32, s,
                      dtype, cache_dtype, kp, vp)
                });
            }
        }
    }
    g.family("reshape_and_cache");
}

fn gather_kv_cache(g: &mut G) {
    // (seq lens, num_kv_heads, head_size, block_size)
    let shapes: [(&[usize], usize, usize, usize); 6] = [
        (&[1], 1, 64, 16),
        (&[5, 17, 1], 2, 128, 8),
        (&[33, 0, 64, 2], 4, 80, 16),
        (&[40, 9], 8, 128, 32),
        (&[3, 3, 3, 3, 3], 1, 256, 16),
        (&[70], 2, 512, 8),
    ];
    for (si, &(lens, nkv, hs, bs)) in shapes.iter().enumerate() {
        for dtype in 0..3u32 {
            for fp8 in [false, true] {
                let cache_dtype = if fp8 { 3 } else { dtype };
                let ce = if fp8 { 1 } else { esize(dtype) };
                let x = 16 / ce;
                let nseq = lens.len();
                let max_blocks = lens.iter().map(|l| l.div_ceil(bs)).max().unwrap() + 1;
                let num_blocks = nseq * max_blocks + 3;
                let perm = g.perm(num_blocks);
                let bt: Vec<i32> = (0..nseq * max_blocks).map(|i| perm[i] as i32).collect();
                let mut cu = vec![0i32];
                for l in lens {
                    cu.push(cu.last().unwrap() + *l as i32);
                }
                // v0.9.4: tokens past cu_seq_lens[num_seqs] (num_tokens > the total) are left untouched
                let nt = *cu.last().unwrap() as usize + [0usize, 1, 3][si % 3];
                let kc = if fp8 { g.fp8s(num_blocks * nkv * hs * bs) } else { g.vals(dtype, num_blocks * nkv * hs * bs) };
                let vc = if fp8 { g.fp8s(num_blocks * nkv * hs * bs) } else { g.vals(dtype, num_blocks * nkv * hs * bs) };
                let es = esize(dtype);
                let ko = g.rng.bytes((nt + 1) * nkv * hs * es);
                let vo = g.rng.bytes((nt + 1) * nkv * hs * es);
                let sc = [SCALES[si % SCALES.len()], SCALES[(si + 5 + dtype as usize) % SCALES.len()]];
                let s = g.stream((si + 1) % 2);
                let label = format!("gather_kv_cache lens={lens:?} nkv={nkv} hs={hs} bs={bs} dtype={dtype} cache={cache_dtype}");
                g.case(&label, vec![kc, vc, ko, vo, f32b(&sc[..1]), f32b(&sc[1..]), i32s(&bt), i32s(&cu)], |rust, b| unsafe {
                    let f = if rust { ox::gather_kv_cache } else { cref::gather_kv_cache };
                    let (kp, vp) = if fp8 { (b.p(4), b.p(5)) } else { (std::ptr::null_mut(), std::ptr::null_mut()) };
                    f(b.p(0), b.p(1), b.p(2), b.p(3), kp, vp, b.p(6) as _, b.p(7) as _, nt as i32, nseq as i32, bs as i32, max_blocks as i32,
                      nkv as i32, hs as i32, x as i32, s, dtype, cache_dtype)
                });
            }
        }
    }
    g.family("gather_kv_cache");
}

fn update_kvscales(g: &mut G) {
    type F = unsafe extern "C" fn(*mut c_void, *mut c_void, i64, *mut f32, *mut f32, i64);
    let fns: [(&str, u32, F, F); 3] = [
        ("f32", 2, cref::update_kv_scales_f32, ox::update_kv_scales_f32),
        ("f16", 0, cref::update_kv_scales_f16, ox::update_kv_scales_f16),
        ("bf16", 1, cref::update_kv_scales_bf16, ox::update_kv_scales_bf16),
    ];
    let inits: [(f32, f32); 5] = [(0.0, 0.0), (0.01, 1e9), (f32::NAN, 0.5), (1e-40, -1.0), (f32::INFINITY, 0.0)];
    for (ci, &n) in [0usize, 1, 7, 511, 512, 513, 4099, 70000, 300001].iter().enumerate() {
        for &(name, dtype, cf, of) in &fns {
            for (ii, &(k0, v0)) in inits.iter().enumerate() {
                if ii > 1 && n > 5000 {
                    continue;
                }
                let k = if ii == 1 { g.nice(dtype, n, 3.0) } else { g.vals(dtype, n) };
                let v = if ii == 1 { g.nice(dtype, n, 500.0) } else { g.vals(dtype, n) };
                let s = if (ci + ii) % 2 == 0 { 0i64 } else { g.stream(1) as i64 };
                g.case(&format!("update_kv_scales_{name} n={n} init=({k0},{v0})"), vec![k, v, f32b(&[k0]), f32b(&[v0])], |rust, b| unsafe {
                    let f = if rust { of } else { cf };
                    f(b.p(0), b.p(1), n as i64, b.p(2) as _, b.p(3) as _, s)
                });
            }
        }
    }
    g.family("update_kvscales");
}

fn copy_blocks(g: &mut G) {
    type F = unsafe extern "C" fn(*mut i64, *mut i64, *const i64, i32, i32, i32, i32, i64);
    let fns: [(&str, usize, F, F); 4] = [
        ("f32", 4, cref::copy_blocks_f32, ox::copy_blocks_f32),
        ("f16", 2, cref::copy_blocks_f16, ox::copy_blocks_f16),
        ("bf16", 2, cref::copy_blocks_bf16, ox::copy_blocks_bf16),
        ("u8", 1, cref::copy_blocks_u8, ox::copy_blocks_u8),
    ];
    // (num_layers, num_pairs, numel_per_block_key, numel_per_block_value)
    for (ci, &(nl, np, nk, nv)) in [(1usize, 1usize, 16usize, 16usize), (2, 3, 1024, 1024), (3, 2, 2048, 1000), (1, 4, 300, 5000), (2, 1, 1, 1), (2, 3, 4097, 4099), (1, 2, 3, 7)].iter().enumerate() {
        for &(name, es, cf, of) in &fns {
            let nblocks = 2 * np + 2;
            let perm = g.perm(nblocks);
            let map: Vec<i64> = (0..np).flat_map(|p| [perm[2 * p] as i64, perm[2 * p + 1] as i64]).collect();
            // buffers: 0 = key ptr table, 1 = value ptr table, 2 = mapping, then per layer key, value caches
            let mut bufs = vec![vec![0u8; nl * 8], vec![0u8; nl * 8], i64s(&map)];
            for _ in 0..nl {
                bufs.push(g.rng.bytes(nblocks * nk * es));
                bufs.push(g.rng.bytes(nblocks * nv * es));
            }
            let s = if ci % 2 == 0 { 0i64 } else { g.stream(1) as i64 };
            let ctx = g.ctx.clone();
            g.case(&format!("copy_blocks_{name} nl={nl} np={np} nk={nk} nv={nv}"), bufs, |rust, b| unsafe {
                // per-side pointer tables into this side's own caches; restored to zero afterwards
                let kt: Vec<u64> = (0..nl).map(|l| b.p(3 + 2 * l) as u64).collect();
                let vt: Vec<u64> = (0..nl).map(|l| b.p(4 + 2 * l) as u64).collect();
                cuda_core::sys::cuMemcpyHtoD_v2(b.p(0) as u64, kt.as_ptr() as *const c_void, nl * 8);
                cuda_core::sys::cuMemcpyHtoD_v2(b.p(1) as u64, vt.as_ptr() as *const c_void, nl * 8);
                // pageable HtoD may still be in DMA when the call returns: finish it before a
                // launch on a non-blocking stream (PORTING.md rule 32)
                ctx.synchronize().unwrap();
                let f = if rust { of } else { cf };
                f(b.p(0) as _, b.p(1) as _, b.p(2) as _, nl as i32, np as i32, nk as i32, nv as i32, s);
                ctx.synchronize().unwrap();
                cuda_core::sys::cuMemsetD8_v2(b.p(0) as u64, 0, nl * 8);
                cuda_core::sys::cuMemsetD8_v2(b.p(1) as u64, 0, nl * 8);
            });
        }
    }
    g.family("copy_blocks");
}

fn concat_and_cache_mla(g: &mut G) {
    // (num_tokens, kv_lora_rank, kpe_head_dim, block_size, ckv extra stride, kpe extra stride)
    for (si, &(nt, r, d, bs, cx, kx)) in
        [(1usize, 512usize, 64usize, 16usize, 0usize, 0usize), (9, 512, 64, 32, 64, 576), (17, 256, 32, 8, 0, 0), (4, 1024, 700, 16, 3, 5), (40, 64, 64, 64, 0, 0)]
            .iter()
            .enumerate()
    {
        for dtype in 0..3u32 {
            let es = esize(dtype);
            let (cs, ks) = (r + cx, d + kx);
            let nblocks = (nt + 2).div_ceil(bs) + 1;
            let perm = g.perm(nblocks * bs);
            let slots: Vec<i64> = (0..nt).map(|t| if t % 4 == 2 { -1 } else { perm[t] as i64 }).collect();
            let ckv = g.vals(dtype, nt * cs);
            let kpe = g.vals(dtype, nt * ks);
            let cc = g.rng.bytes(nblocks * bs * r * es);
            let kc = g.rng.bytes(nblocks * bs * d * es);
            let s = g.stream(si % 2);
            g.case(&format!("concat_and_cache_mla nt={nt} r={r} d={d} bs={bs} dtype={dtype}"), vec![ckv, kpe, cc, kc, i64s(&slots)], |rust, b| unsafe {
                let f = if rust { ox::concat_and_cache_mla } else { cref::concat_and_cache_mla };
                f(b.p(0), b.p(1), b.p(2), b.p(3), b.p(4) as _, nt as i32, r as i32, d as i32, bs as i32, cs as i32, ks as i32, s, dtype)
            });
        }
    }
    g.family("concat_and_cache_mla");
}

fn gather_mla_cache(g: &mut G) {
    for (si, &(lens, r, d, bs)) in
        [(&[1usize][..], 512usize, 64usize, 16usize), (&[5, 30, 2], 512, 64, 32), (&[16, 16], 256, 32, 8), (&[3, 0, 77], 1000, 300, 16)].iter().enumerate()
    {
        for dtype in 0..3u32 {
            let es = esize(dtype);
            let nseq = lens.len();
            let max_blocks = lens.iter().map(|l| l.div_ceil(bs)).max().unwrap() + 1;
            let nblocks = nseq * max_blocks + 2;
            let perm = g.perm(nblocks);
            let bt: Vec<i32> = (0..nseq * max_blocks).map(|i| perm[i] as i32).collect();
            let mut cu = vec![0i32];
            let mut t2s = vec![];
            for (b, l) in lens.iter().enumerate() {
                cu.push(cu.last().unwrap() + *l as i32);
                t2s.extend(std::iter::repeat(b as i32).take(*l));
            }
            let nt = t2s.len();
            let cc = g.vals(dtype, nblocks * bs * r);
            let kc = g.vals(dtype, nblocks * bs * d);
            let co = g.rng.bytes((nt + 1) * r * es);
            let ko = g.rng.bytes((nt + 1) * d * es);
            let s = g.stream((si + 1) % 2);
            g.case(&format!("gather_mla_cache lens={lens:?} r={r} d={d} bs={bs} dtype={dtype}"), vec![cc, kc, co, ko, i32s(&bt), i32s(&cu), i32s(&t2s)], |rust, b| unsafe {
                let f = if rust { ox::gather_mla_cache } else { cref::gather_mla_cache };
                f(b.p(0), b.p(1), b.p(2), b.p(3), b.p(4) as _, b.p(5) as _, b.p(6) as _, nt as i32, bs as i32, max_blocks as i32, r as i32, d as i32, s, dtype)
            });
        }
    }
    g.family("gather_mla_cache");
}


// ================================================================================================
// paged attention v1 / v2

type PaV1 = unsafe extern "C" fn(
    *mut c_void, *mut c_void, *mut c_void, *mut c_void, *mut c_void, i32, f32, f32, *mut u32, *mut u32, i32, i32, i32, i32, i32, i32, i32, i32,
    i32, *mut c_void, u32, *mut f32, *mut f32, *const f32,
);
type PaV2 = unsafe extern "C" fn(
    *mut c_void, *mut f32, *mut f32, *mut c_void, *mut c_void, *mut c_void, *mut c_void, *mut c_void, i32, f32, f32, *mut u32, *mut u32, i32,
    i32, i32, i32, i32, i32, i32, i32, i32, *mut c_void, u32, *mut f32, *mut f32, *const f32,
);

/// One paged-attention scenario.
#[derive(Clone)]
pub struct PaCfg {
    pub lens: Vec<u32>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub alibi: bool,
    pub softcap: f32,
    pub sinks: bool,
    /// raw special values (NaN / inf / denormals) in q, k, v
    pub specials: bool,
    /// extra elements in q_stride, kv_block_stride, kv_head_stride
    pub pad: bool,
    /// max_context_len passed (>= max(lens))
    pub max_extra: u32,
    /// finite data with a wide exponent range (exposes summation order / rounding points)
    pub wide: bool,
}

fn pa_cfgs(bs: usize) -> Vec<PaCfg> {
    let b = bs as u32;
    vec![
        PaCfg { lens: vec![1, b, b + 1, 37], num_heads: 4, num_kv_heads: 4, alibi: false, softcap: 1.0, sinks: false, specials: false, pad: false, max_extra: 0, wide: true },
        PaCfg { lens: vec![700, 3], num_heads: 8, num_kv_heads: 2, alibi: true, softcap: 30.0, sinks: true, specials: false, pad: true, max_extra: 5, wide: false },
        PaCfg { lens: vec![1030], num_heads: 2, num_kv_heads: 1, alibi: true, softcap: 1.0, sinks: true, specials: true, pad: false, max_extra: 0, wide: false },
        PaCfg { lens: vec![512, 513, 1024, 1], num_heads: 8, num_kv_heads: 8, alibi: false, softcap: 0.7, sinks: false, specials: false, pad: true, max_extra: 0, wide: true },
        PaCfg { lens: vec![2 * b - 1, 511, 1537], num_heads: 6, num_kv_heads: 3, alibi: false, softcap: 1.0, sinks: true, specials: false, pad: false, max_extra: 31, wide: true },
        PaCfg { lens: vec![5, 64], num_heads: 16, num_kv_heads: 2, alibi: true, softcap: 1.0, sinks: false, specials: true, pad: true, max_extra: 0, wide: false },
    ]
}

fn pa_case(g: &mut G, v2: bool, dtype: u32, fp8: bool, head: usize, bs: usize, c: &PaCfg, stream: usize) {
    let es = esize(dtype);
    let ce = if fp8 { 1 } else { es };
    let x = 16 / ce;
    let (nh, nkv) = (c.num_heads, c.num_kv_heads);
    let nseq = c.lens.len();
    let max_len = *c.lens.iter().max().unwrap();
    let max_ctx = (max_len + c.max_extra) as usize;
    let mbps = max_ctx.div_ceil(bs) + 1;
    let q_stride = nh * head + if c.pad { 3 * head } else { 0 };
    let kv_head_stride = head * bs + if c.pad { x * 2 } else { 0 };
    let kv_block_stride = nkv * kv_head_stride + if c.pad { 16 * x } else { 0 };
    let num_blocks = nseq * mbps + 2;
    let perm = g.perm(num_blocks);
    let bt: Vec<u32> = (0..nseq * mbps).map(|i| perm[i] as u32).collect();
    let q = if c.specials { g.vals(dtype, nseq * q_stride) } else if c.wide { g.wide(dtype, nseq * q_stride, 4) } else { g.nice(dtype, nseq * q_stride, 1.0) };
    let cache_elems = num_blocks * kv_block_stride;
    let (kc, vc) = if fp8 {
        let mut k = g.fp8s(cache_elems);
        let mut v = g.fp8s(cache_elems);
        if c.wide {
            // every finite code (NaN is S.1111.111)
            for b in k.iter_mut().chain(v.iter_mut()) {
                if *b & 0x7f == 0x7f {
                    *b ^= 1;
                }
            }
        } else if !c.specials {
            // no NaN codes, magnitudes below 2^3
            for b in k.iter_mut().chain(v.iter_mut()) {
                *b &= 0xbf;
            }
        }
        (k, v)
    } else if c.specials {
        (g.vals(dtype, cache_elems), g.vals(dtype, cache_elems))
    } else if c.wide {
        (g.wide(dtype, cache_elems, 2), g.wide(dtype, cache_elems, 6))
    } else {
        (g.nice(dtype, cache_elems, 1.0), g.nice(dtype, cache_elems, 2.0))
    };
    let alibi: Vec<f32> = (0..nh).map(|h| [0.0625, -0.25, 0.0, 1e-40, 0.5, -1e-3, 2.0, 0.125][h % 8]).collect();
    let sinks: Vec<f32> = (0..nh).map(|h| [0.5, -3.0, 12.0, 0.0, f32::NEG_INFINITY, 1.5, -0.25, 60.0][h % 8]).collect();
    let scales = [[0.0625f32, 0.02], [0.037, 1.3], [3.1, 0.0071]][g.below(3)];
    let sm_scale = 1.0 / (head as f32).sqrt();
    let mp = max_ctx.div_ceil(512);
    let out = g.rng.bytes(nseq * nh * head * es);
    let mut bufs = vec![out, q, kc, vc, f32b(&alibi), as_bytes(&bt), as_bytes(&c.lens), f32b(&scales[..1]), f32b(&scales[1..]), f32b(&sinks)];
    if v2 {
        bufs.push(g.rng.bytes(nseq * nh * mp * 4));
        bufs.push(g.rng.bytes(nseq * nh * mp * 4));
        bufs.push(g.rng.bytes(nseq * nh * mp * head * es));
    }
    let s = g.stream(stream);
    let cache_dtype = if fp8 { 3 } else { dtype };
    let label = format!(
        "paged_attention_v{} dtype={dtype} cache={cache_dtype} head={head} bs={bs} lens={:?} nh={nh} nkv={nkv} alibi={} cap={} sinks={} specials={} pad={}",
        if v2 { 2 } else { 1 }, c.lens, c.alibi, c.softcap, c.sinks, c.specials, c.pad
    );
    let (alibi_on, sinks_on) = (c.alibi, c.sinks);
    let softcap = c.softcap;
    let v1f: [(PaV1, PaV1); 3] = [
        (cref::paged_attention_v1_f16, ox::paged_attention_v1_f16),
        (cref::paged_attention_v1_bf16, ox::paged_attention_v1_bf16),
        (cref::paged_attention_v1_f32, ox::paged_attention_v1_f32),
    ];
    let v2f: [(PaV2, PaV2); 3] = [
        (cref::paged_attention_v2_f16, ox::paged_attention_v2_f16),
        (cref::paged_attention_v2_bf16, ox::paged_attention_v2_bf16),
        (cref::paged_attention_v2_f32, ox::paged_attention_v2_f32),
    ];
    g.case(&label, bufs, |rust, b| unsafe {
        let al = if alibi_on { b.p(4) } else { std::ptr::null_mut() };
        let sk = if sinks_on { b.p(9) as *const f32 } else { std::ptr::null() };
        let (ks, vs) = if fp8 { (b.p(7) as *mut f32, b.p(8) as *mut f32) } else { (std::ptr::null_mut(), std::ptr::null_mut()) };
        if v2 {
            let f = if rust { v2f[dtype as usize].1 } else { v2f[dtype as usize].0 };
            f(b.p(0), b.p(10) as _, b.p(11) as _, b.p(12), b.p(1), b.p(2), b.p(3), al, nkv as i32, sm_scale, softcap, b.p(5) as _, b.p(6) as _,
              bs as i32, max_ctx as i32, nseq as i32, nh as i32, head as i32, mbps as i32, q_stride as i32, kv_block_stride as i32,
              kv_head_stride as i32, s, cache_dtype, ks, vs, sk)
        } else {
            let f = if rust { v1f[dtype as usize].1 } else { v1f[dtype as usize].0 };
            f(b.p(0), b.p(1), b.p(2), b.p(3), al, nkv as i32, sm_scale, softcap, b.p(5) as _, b.p(6) as _, bs as i32, max_ctx as i32,
              nseq as i32, nh as i32, head as i32, mbps as i32, q_stride as i32, kv_block_stride as i32, kv_head_stride as i32, s,
              cache_dtype, ks, vs, sk)
        }
    });
}

fn paged_attention(g: &mut G, v2: bool) {
    let heads: Vec<usize> = std::env::var("MPA_HEADS").ok().map(|s| s.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![64, 80, 96, 112, 128, 192, 256, 512]);
    let per = std::env::var("MPA_PA_CFGS").ok().and_then(|s| s.parse().ok()).unwrap_or(3usize);
    let mut n = 0usize;
    for dtype in [2u32, 0, 1] {
        for fp8 in [false, true] {
            for bs in [8usize, 16, 32] {
                let cfgs = pa_cfgs(bs);
                for &head in heads.iter() {
                    for k in 0..per {
                        let c = cfgs[(n + k * 7) % cfgs.len()].clone();
                        pa_case(g, v2, dtype, fp8, head, bs, &c, (n + k) % 2);
                    }
                    n += 1;
                }
            }
        }
    }
    // unsupported head size / block size: nothing launched, nothing written
    for (head, bs) in [(48usize, 16usize), (128, 64)] {
        let c = pa_cfgs(16)[0].clone();
        pa_case(g, v2, 0, false, head, bs, &c, 0);
    }
    g.family(if v2 { "paged_attention_v2" } else { "paged_attention_v1" });
}


// ================================================================================================
// flash_attn_sinks (+ varlen)

type Fa = unsafe extern "C" fn(*const c_void, *const c_void, *const c_void, *mut c_void, *const f32, f32, i32, i32, i32, i32, i32, i32, i32, *mut c_void);
type Fav = unsafe extern "C" fn(*const c_void, *const c_void, *const c_void, *mut c_void, *const f32, *const u32, *const u32, f32, i32, i32, i32, i32, i32, i32, *mut c_void);

fn flash_attn_sinks(g: &mut G) {
    let fas: [(Fa, Fa); 3] = [
        (cref::flash_attn_sinks_f16, ox::flash_attn_sinks_f16),
        (cref::flash_attn_sinks_bf16, ox::flash_attn_sinks_bf16),
        (cref::flash_attn_sinks_f32, ox::flash_attn_sinks_f32),
    ];
    let favs: [(Fav, Fav); 3] = [
        (cref::flash_attn_sinks_varlen_f16, ox::flash_attn_sinks_varlen_f16),
        (cref::flash_attn_sinks_varlen_bf16, ox::flash_attn_sinks_varlen_bf16),
        (cref::flash_attn_sinks_varlen_f32, ox::flash_attn_sinks_varlen_f32),
    ];
    // (batch, q_len, kv_len, nh, nkv, window, sinks, specials)
    let cases: [(usize, usize, usize, usize, usize, i32, bool, bool); 7] = [
        (1, 1, 1, 1, 1, 0, true, false),
        (2, 13, 13, 4, 2, 0, true, false),
        (1, 37, 100, 8, 1, 0, false, false),
        (2, 70, 70, 2, 2, 16, true, false),
        (1, 9, 150, 4, 4, 33, true, true),
        (3, 5, 40, 6, 3, 0, false, true),
        (1, 64, 64, 2, 1, 1, true, false),
    ];
    let mut heads: Vec<usize> = std::env::var("MPA_HEADS").ok().map(|s| s.split(',').map(|x| x.parse().unwrap()).filter(|&h| h != 512).collect()).unwrap_or(vec![64, 80, 96, 112, 128, 192, 256]);
    heads.push(48);
    let mut n = 0;
    for dtype in [2u32, 0, 1] {
        let es = esize(dtype);
        for &hd in &heads {
            for k in 0..3 {
                let (b, ql, kl, nh, nkv, w, sk, sp) = cases[(n + 3 * k) % cases.len()];
                n += 1;
                let wd = !sp && n % 2 == 0;
                let q = if sp { g.vals(dtype, b * nh * ql * hd) } else if wd { g.wide(dtype, b * nh * ql * hd, 3) } else { g.nice(dtype, b * nh * ql * hd, 2.0) };
                let kk = if sp { g.vals(dtype, b * nkv * kl * hd) } else if wd { g.wide(dtype, b * nkv * kl * hd, 2) } else { g.nice(dtype, b * nkv * kl * hd, 2.0) };
                let vv = if sp { g.vals(dtype, b * nkv * kl * hd) } else if wd { g.wide(dtype, b * nkv * kl * hd, 6) } else { g.nice(dtype, b * nkv * kl * hd, 3.0) };
                let o = g.rng.bytes(b * nh * ql * hd * es);
                let sinks: Vec<f32> = (0..nh).map(|h| [0.5, -3.0, 12.0, 0.0, f32::NEG_INFINITY, 1.5, 40.0, -0.25][h % 8]).collect();
                let scale = 1.0 / (hd as f32).sqrt();
                let s = g.stream(n % 2);
                let (cf, of) = fas[dtype as usize];
                g.case(&format!("flash_attn_sinks dtype={dtype} hd={hd} b={b} q={ql} kv={kl} nh={nh} nkv={nkv} w={w} sinks={sk} specials={sp}"),
                       vec![q, kk, vv, o, f32b(&sinks)], |rust, bf| unsafe {
                    let f = if rust { of } else { cf };
                    let skp = if sk { bf.p(4) as *const f32 } else { std::ptr::null() };
                    f(bf.p(0), bf.p(1), bf.p(2), bf.p(3), skp, scale, b as i32, ql as i32, kl as i32, nh as i32, nkv as i32, hd as i32, w, s)
                });
            }
            // varlen: ragged q / kv lengths (incl. 0-length q, q longer than kv impossible -> kv >= q)
            for k in 0..3 {
                let qls: Vec<usize> = [vec![1usize, 17, 0, 9], vec![40], vec![3, 3, 64], vec![8, 1]][(n + k) % 4].clone();
                let kls: Vec<usize> = qls.iter().enumerate().map(|(i, &q)| q + [0usize, 5, 31, 100][(i + k) % 4]).collect();
                n += 1;
                let (nh, nkv) = [(4usize, 2usize), (2, 2), (6, 1)][k];
                let w = [0i32, 7, 0][k];
                let sp = k == 2 && hd % 64 == 0;
                let bsz = qls.len();
                let max_q = *qls.iter().max().unwrap();
                let tot_k: usize = kls.iter().sum();
                let mut cq = vec![0u32];
                let mut ck = vec![0u32];
                for i in 0..bsz {
                    cq.push(cq[i] + qls[i] as u32);
                    ck.push(ck[i] + kls[i] as u32);
                }
                let q = if sp { g.vals(dtype, bsz * nh * max_q * hd) } else { g.nice(dtype, bsz * nh * max_q * hd, 2.0) };
                let kk = if sp { g.vals(dtype, tot_k * nkv * hd) } else { g.nice(dtype, tot_k * nkv * hd, 2.0) };
                let vv = if k == 0 { g.wide(dtype, tot_k * nkv * hd, 6) } else { g.nice(dtype, tot_k * nkv * hd, 3.0) };
                let o = g.rng.bytes(bsz * nh * max_q * hd * es);
                let sinks: Vec<f32> = (0..nh).map(|h| [2.0, -1.0, 0.0, 7.0, -9.0, 0.5][h % 6]).collect();
                let sk = k != 1;
                let scale = 1.0 / (hd as f32).sqrt();
                let s = g.stream(n % 2);
                let (cf, of) = favs[dtype as usize];
                g.case(&format!("flash_attn_sinks_varlen dtype={dtype} hd={hd} q={qls:?} kv={kls:?} nh={nh} nkv={nkv} w={w} sinks={sk}"),
                       vec![q, kk, vv, o, f32b(&sinks), as_bytes(&cq), as_bytes(&ck)], |rust, bf| unsafe {
                    let f = if rust { of } else { cf };
                    let skp = if sk { bf.p(4) as *const f32 } else { std::ptr::null() };
                    f(bf.p(0), bf.p(1), bf.p(2), bf.p(3), skp, bf.p(5) as _, bf.p(6) as _, scale, bsz as i32, max_q as i32, nh as i32, nkv as i32,
                      hd as i32, w, s)
                });
            }
        }
    }
    g.family("flash_attn_sinks");
}


// ================================================================================================
// Exit / stderr paths: CUDA_CHECK(cudaGetLastError()) -> exit(err), FA_CUDA_CHECK / unsupported
// head_dim -> stderr only. Each case runs in a child process per side; exit status and stderr
// must match exactly.

pub const EXIT_CASES: usize = 7;

pub fn exit_case(rust: bool, id: usize) {
    let ctx = CudaContext::new(0).expect("cuda context");
    ctx.bind_to_thread().unwrap();
    let st = ctx.default_stream();
    let buf = DeviceBuffer::<u8>::from_host(&st, &vec![0u8; 1 << 16]).unwrap();
    let p = buf.cu_deviceptr() as *mut c_void;
    let nul = std::ptr::null_mut::<c_void>();
    macro_rules! pick { ($f:ident) => { if rust { ox::$f } else { cref::$f } }; }
    unsafe {
        match id {
            0 => pick!(reshape_and_cache)(p, p, p, p, p as _, 0, 2, 64, 16, 8, 128, 128, nul, 0, 0, std::ptr::null_mut(), std::ptr::null_mut()),
            1 => pick!(concat_and_cache_mla)(p, p, p, p, p as _, 0, 512, 64, 16, 512, 64, nul, 1),
            // dynamic shared memory above the device limit: cudaFuncSetAttribute fails
            2 => pick!(paged_attention_v1_f16)(p, p, p, p, nul, 1, 1.0, 1.0, p as _, p as _, 16, 200_000, 1, 1, 64, 1, 64, 1024, 1024, nul, 0,
                                               std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null()),
            // max_context_len 0: zero partitions -> zero grid
            3 => pick!(paged_attention_v2_f32)(p, p as _, p as _, p, p, p, p, nul, 1, 1.0, 1.0, p as _, p as _, 16, 0, 1, 1, 64, 1, 64, 1024,
                                               1024, nul, 2, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null()),
            4 => pick!(flash_attn_sinks_bf16)(p, p, p, p, std::ptr::null(), 1.0, 1, 4, 4, 1, 1, 72, 0, nul),
            5 => pick!(flash_attn_sinks_f32)(p, p, p, p, std::ptr::null(), 1.0, 1, 0, 4, 1, 1, 64, 0, nul),
            6 => pick!(flash_attn_sinks_varlen_f16)(p, p, p, p, std::ptr::null(), p as _, p as _, 1.0, 1, 4, 1, 1, 40, 0, nul),
            _ => unreachable!(),
        }
    }
    ctx.synchronize().unwrap();
}

fn exit_cases(g: &mut G) {
    let exe = std::env::current_exe().unwrap();
    for id in 0..EXIT_CASES {
        let run = |side: &str| {
            let o = std::process::Command::new(&exe).args(["--exit-case", side, &id.to_string()]).output().unwrap();
            (o.status.code(), o.status.to_string(), String::from_utf8_lossy(&o.stderr).to_string())
        };
        let (c, r) = (run("c"), run("rust"));
        let d = kdiff::Diff {
            bytes: c.2.len().max(1),
            differing: if c == r { 0 } else { 1 },
            first: if c == r { None } else { Some((0, 0, 0, 1)) },
        };
        if c != r {
            println!("  exit case {id}: C {:?} / Rust {:?}", c, r);
        } else {
            println!("  exit case {id}: {} {:?}", c.1, c.2.trim());
        }
        g.calls += 2;
        g.family_calls += 2;
        g.t.record(&format!("exit case {id}"), &d);
    }
    g.family("exit/stderr paths");
}

// ================================================================================================

pub fn run() -> bool {
    let ctx = CudaContext::new(0).expect("cuda context");
    ctx.bind_to_thread().unwrap();
    let streams = vec![ctx.default_stream(), ctx.new_stream().unwrap()];
    let mut g = G { ctx: ctx.clone(), streams, rng: Rng(0x9A6E_D0A7), t: Tally::default(), calls: 0, family_calls: 0 };
    let only = std::env::var("MPA_ONLY").ok();
    let want = |f: &str| only.as_deref().map(|o| o.split(',').any(|x| x == f)).unwrap_or(true);

    if want("reshape_and_cache") { reshape_and_cache(&mut g); }
    if want("gather_kv_cache") { gather_kv_cache(&mut g); }
    if want("update_kvscales") { update_kvscales(&mut g); }
    if want("copy_blocks") { copy_blocks(&mut g); }
    if want("concat_and_cache_mla") { concat_and_cache_mla(&mut g); }
    if want("gather_mla_cache") { gather_mla_cache(&mut g); }
    if want("paged_attention_v1") { paged_attention(&mut g, false); }
    if want("paged_attention_v2") { paged_attention(&mut g, true); }
    if want("flash_attn_sinks") { flash_attn_sinks(&mut g); }
    if want("exit") { exit_cases(&mut g); }

    for f in g.t.failures.iter().take(40) {
        println!("  FAIL {f}");
    }
    let ok = g.t.failures.is_empty() && g.calls > 0;
    println!(
        "mistralrs-paged-attn-a: {} launcher calls, {} bytes compared, {} failing -> {}",
        g.calls,
        g.t.bytes,
        g.t.failures.len(),
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}
