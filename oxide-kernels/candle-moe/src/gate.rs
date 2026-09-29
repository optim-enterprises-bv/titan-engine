//! Differential gate: candle's real C launchers (libmoe.a, linked statically) vs the pure-Rust
//! launchers of `launch.rs`, called on identical inputs in the same primary context and stream.
//! Every output byte is compared, plus the expert_counts / expert_offsets buffers of
//! `moe_gemm_wmma`. Outputs start as identical random bytes (the launchers never clear them).
use crate::launch;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use kdiff::Rng;
use std::ffi::c_void;
use std::sync::Arc;

unsafe extern "C" {
    #[link_name = "moe_gemm_wmma"]
    fn c_moe_gemm_wmma(
        input: *const c_void, weights: *const c_void, sorted_token_ids: *const i32, expert_ids: *const i32,
        topk_weights: *const f32, output: *mut c_void, expert_counts: *mut i32, expert_offsets: *mut i32,
        num_experts: i32, topk: i32, size_m: i32, size_n: i32, size_k: i32, dtype: i32, is_prefill: bool, stream: i64,
    );
    #[link_name = "moe_gemm_gguf"]
    fn c_moe_gemm_gguf(
        input: *const f32, weights: *const c_void, sorted_token_ids: *const i32, expert_ids: *const i32,
        topk_weights: *const f32, output: *mut c_void, num_experts: i32, topk: i32, size_m: i32, size_n: i32,
        size_k: i32, gguf_dtype: i32, stream: i64,
    );
    #[link_name = "moe_gemm_gguf_prefill"]
    fn c_moe_gemm_gguf_prefill(
        input: *const c_void, weights: *const u8, sorted_token_ids: *const i32, expert_ids: *const i32,
        topk_weights: *const f32, output: *mut c_void, num_experts: i32, topk: i32, size_m: i32, size_n: i32,
        size_k: i32, input_dtype: i32, gguf_dtype: i32, stream: i64,
    );
}

const GGUF_NAMES: [&str; 6] = ["q8_0", "q4_k", "q2_k", "q3_k", "q5_k", "q6_k"];
/// (qk, block bytes, byte offsets of the f16 scale fields).
fn gguf_layout(t: usize) -> (usize, usize, &'static [usize]) {
    match t {
        0 => (32, 34, &[0]),
        1 => (256, 144, &[0, 2]),
        2 => (256, 84, &[80, 82]),
        3 => (256, 110, &[108]),
        4 => (256, 176, &[0, 2]),
        _ => (256, 210, &[208]),
    }
}

struct Gate {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    rng: Rng,
    calls: usize,
    bytes: usize,
    elems: usize,
    written: usize,
    finite: usize,
    failures: Vec<String>,
}

fn f16_bits(r: &mut Rng, special_every: u64) -> u16 {
    if special_every > 0 && r.next() % special_every == 0 {
        const S: [u16; 10] = [0x0000, 0x8000, 0x7c00, 0xfc00, 0x7e00, 0xfe01, 0x0001, 0x83ff, 0x7bff, 0x0400];
        let k = r.next() % 11;
        return if k == 10 { r.next() as u16 } else { S[k as usize] };
    }
    let sign = (r.next() & 1) as u16;
    let exp = 10 + (r.next() % 7) as u16; // 2^-5 .. 2^1
    (sign << 15) | (exp << 10) | (r.next() as u16 & 0x3ff)
}

fn bf16_bits(r: &mut Rng, special_every: u64) -> u16 {
    if special_every > 0 && r.next() % special_every == 0 {
        const S: [u16; 10] = [0x0000, 0x8000, 0x7f80, 0xff80, 0x7fc0, 0xffc1, 0x0001, 0x807f, 0x7f7f, 0x0080];
        let k = r.next() % 11;
        return if k == 10 { r.next() as u16 } else { S[k as usize] };
    }
    let sign = (r.next() & 1) as u16;
    let exp = 122 + (r.next() % 7) as u16;
    (sign << 15) | (exp << 7) | (r.next() as u16 & 0x7f)
}

fn f32_val(r: &mut Rng, special_every: u64) -> f32 {
    if special_every > 0 && r.next() % special_every == 0 {
        const S: [f32; 9] = [0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 1e-40, -1e-42, 3e38, -3e38];
        let k = r.next() % 10;
        return if k == 9 { f32::from_bits(r.next() as u32) } else { S[k as usize] };
    }
    ((r.next() >> 11) as f64 / (1u64 << 53) as f64 * 8.0 - 4.0) as f32
}

/// Quantised expert weights: random bytes with f16 scale fields drawn from a small-scale
/// distribution plus NaN / inf / denormal / zero / random patterns.
fn gguf_weights(r: &mut Rng, t: usize, blocks: usize) -> Vec<u8> {
    let (_, bs, scales) = gguf_layout(t);
    let mut w = r.bytes(blocks * bs);
    for b in 0..blocks {
        for &o in scales {
            let v = if r.next() % 24 == 0 {
                const S: [u16; 8] = [0x7c00, 0xfc00, 0x7e00, 0x0001, 0x8200, 0x0000, 0x8000, 0x7bff];
                let k = r.next() % 9;
                if k == 8 { r.next() as u16 } else { S[k as usize] }
            } else {
                let sign = if o == scales[0] { 0 } else { (r.next() & 1) as u16 };
                (sign << 15) | ((3 + r.next() % 11) as u16) << 10 | (r.next() as u16 & 0x3ff)
            };
            w[b * bs + o..b * bs + o + 2].copy_from_slice(&v.to_le_bytes());
        }
    }
    w
}

/// Routing for `tokens` tokens with top-`topk` experts out of `num_experts`: returns
/// (sorted_token_ids, expert_ids) of length tokens*topk, grouped by expert (ascending).
/// pattern 0: uniform; 1: skewed to few experts; 2: only 3 experts used; 3: one expert.
fn routing(r: &mut Rng, tokens: usize, topk: usize, num_experts: usize, pattern: u32) -> (Vec<i32>, Vec<i32>) {
    let pairs = tokens * topk;
    let mut ex: Vec<(i32, i32)> = (0..pairs)
        .map(|p| {
            let e = match pattern {
                0 => r.next() % num_experts as u64,
                1 => {
                    if r.next() % 4 != 0 { r.next() % (num_experts as u64).min(2) } else { r.next() % num_experts as u64 }
                }
                2 => [0u64, num_experts as u64 / 2, num_experts as u64 - 1][(r.next() % 3) as usize],
                _ => (num_experts - 1) as u64,
            };
            (e as i32, p as i32)
        })
        .collect();
    ex.sort();
    (ex.iter().map(|x| x.1).collect(), ex.iter().map(|x| x.0).collect())
}

fn b16s(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn i32s(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

impl Gate {
    fn buf(&self, bytes: &[u8]) -> DeviceBuffer<u8> {
        DeviceBuffer::from_host(&self.stream, if bytes.is_empty() { &[0u8][..] } else { bytes }).unwrap()
    }

    fn stream_handle(&self, use_null: bool) -> i64 {
        if use_null { 0 } else { self.stream.cu_stream() as usize as i64 }
    }

    fn sync(&self) {
        self.ctx.synchronize().unwrap_or_else(|e| panic!("synchronize: {e:?}"));
    }

    /// Guards against a vacuous pass: counts reference output elements that were written (differ
    /// from the initial bytes) and, of those, the finite ones. A case that writes nothing fails.
    fn coverage(&mut self, label: &str, out: &DeviceBuffer<u8>, init: &[u8], kind: u32) {
        let x = out.to_host_vec(&self.stream).unwrap();
        let es = if kind == 2 { 4 } else { 2 };
        let (mut w, mut f) = (0usize, 0usize);
        for (a, b) in x.chunks_exact(es).zip(init.chunks_exact(es)) {
            if a != b {
                w += 1;
                let finite = match kind {
                    0 => (u16::from_le_bytes([a[0], a[1]]) & 0x7c00) != 0x7c00,
                    1 => (u16::from_le_bytes([a[0], a[1]]) & 0x7f80) != 0x7f80,
                    _ => f32::from_le_bytes([a[0], a[1], a[2], a[3]]).is_finite(),
                };
                f += finite as usize;
            }
        }
        self.elems += x.len() / es;
        self.written += w;
        self.finite += f;
        if w == 0 {
            self.failures.push(format!("{label}: reference wrote no output (vacuous case)"));
        }
    }

    fn compare(&mut self, label: &str, what: &str, a: &DeviceBuffer<u8>, b: &DeviceBuffer<u8>, n: usize) {
        let x = a.to_host_vec(&self.stream).unwrap();
        let y = b.to_host_vec(&self.stream).unwrap();
        self.bytes += n;
        let bad: Vec<usize> = (0..n).filter(|&i| x[i] != y[i]).collect();
        if !bad.is_empty() {
            let i = bad[0];
            if std::env::var("MOE_DEBUG").is_ok() {
                let h = |v: &Vec<u8>| (0..16).map(|j| format!("{:04x}", u16::from_le_bytes([v[2*j], v[2*j+1]]))).collect::<Vec<_>>().join(" ");
                println!("{label} {what}\n  ref {}\n  ox  {}", h(&x), h(&y));
            }
            self.failures.push(format!("{label} [{what}]: {} of {n} bytes differ (first at byte {i}: ref {:#04x} oxide {:#04x})", bad.len(), x[i], y[i]));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn wmma_case(&mut self, dtype: i32, is_prefill: bool, with_tw: bool, num_experts: usize, topk: usize, tokens: usize,
                 n: usize, k: usize, pattern: u32, specials: u64, use_null: bool) {
        let r = &mut self.rng;
        let (sorted, eids) = routing(r, tokens, topk, num_experts, pattern);
        let size_m = tokens * topk;
        let in_rows = if with_tw { size_m } else { tokens };
        let mk = |r: &mut Rng, cnt: usize| -> Vec<u16> {
            (0..cnt).map(|_| if dtype == 0 { f16_bits(r, specials) } else { bf16_bits(r, specials) }).collect()
        };
        let input = mk(r, in_rows * k);
        let weights = mk(r, num_experts * n * k);
        let tw: Vec<f32> = (0..size_m).map(|_| f32_val(r, if specials > 0 { 97 } else { 0 }).abs().min(1.0)).collect();
        let out0 = r.bytes(size_m * n * 2);
        let cnt0 = r.bytes(num_experts * 4);
        let off0 = r.bytes((num_experts + 1) * 4);
        let d_in = self.buf(&b16s(&input));
        let d_w = self.buf(&b16s(&weights));
        let d_s = self.buf(&i32s(&sorted));
        let d_e = self.buf(&i32s(&eids));
        let d_tw = self.buf(&f32s(&tw));
        let (o_ref, o_ox) = (self.buf(&out0), self.buf(&out0));
        let (c_ref, c_ox) = (self.buf(&cnt0), self.buf(&cnt0));
        let (f_ref, f_ox) = (self.buf(&off0), self.buf(&off0));
        let twp = if with_tw { d_tw.cu_deviceptr() as *const f32 } else { std::ptr::null() };
        let st = self.stream_handle(use_null);
        unsafe {
            c_moe_gemm_wmma(d_in.cu_deviceptr() as _, d_w.cu_deviceptr() as _, d_s.cu_deviceptr() as _, d_e.cu_deviceptr() as _, twp,
                            o_ref.cu_deviceptr() as _, c_ref.cu_deviceptr() as _, f_ref.cu_deviceptr() as _,
                            num_experts as i32, topk as i32, size_m as i32, n as i32, k as i32, dtype, is_prefill, st);
            self.sync();
            launch::moe_gemm_wmma(d_in.cu_deviceptr() as _, d_w.cu_deviceptr() as _, d_s.cu_deviceptr() as _, d_e.cu_deviceptr() as _, twp,
                                  o_ox.cu_deviceptr() as _, c_ox.cu_deviceptr() as _, f_ox.cu_deviceptr() as _,
                                  num_experts as i32, topk as i32, size_m as i32, n as i32, k as i32, dtype, is_prefill, st);
            self.sync();
        }
        self.calls += 1;
        let label = format!("moe_gemm_wmma dtype={dtype} prefill={is_prefill} tw={with_tw} E={num_experts} topk={topk} T={tokens} n={n} k={k} pat={pattern} null_stream={use_null}");
        self.compare(&label, "output", &o_ref, &o_ox, size_m * n * 2);
        if dtype <= 1 {
            self.coverage(&label, &o_ref, &out0, dtype as u32);
        }
        self.compare(&label, "expert_counts", &c_ref, &c_ox, num_experts * 4);
        self.compare(&label, "expert_offsets", &f_ref, &f_ox, (num_experts + 1) * 4);
    }

    /// Decode (`moe_gemm_gguf`); `bad_ids` injects out-of-range expert ids (skipped by the kernel).
    #[allow(clippy::too_many_arguments)]
    fn gguf_case(&mut self, t: usize, with_tw: bool, num_experts: usize, topk: usize, tokens: usize, n: usize, k: usize,
                 pattern: u32, bad_ids: bool, specials: u64, use_null: bool) {
        let (qk, bs, _) = gguf_layout(t);
        let r = &mut self.rng;
        let (sorted, mut eids) = routing(r, tokens, topk, num_experts, pattern);
        if bad_ids {
            for (i, e) in eids.iter_mut().enumerate() {
                if i % 5 == 3 {
                    *e = if i % 2 == 0 { -1 } else { num_experts as i32 + i as i32 };
                }
            }
        }
        let size_m = tokens * topk;
        let in_rows = if with_tw { size_m } else { tokens };
        let mut input: Vec<f32> = (0..in_rows * k).map(|_| f32_val(r, specials)).collect();
        if in_rows >= 3 {
            // an all-zero row (amax == 0), a row of denormals down to the smallest (d rounds to 0),
            // and a row of exact .5 ties
            for v in &mut input[0..k] { *v = 0.0; }
            for (i, v) in input[k..2 * k].iter_mut().enumerate() { *v = f32::from_bits(1 + (i as u32 % 7)) * if i % 2 == 0 { 1.0 } else { -1.0 }; }
            for (i, v) in input[2 * k..3 * k].iter_mut().enumerate() { *v = ((i % 64) as f32 - 32.0) * 0.5; }
        }
        if in_rows >= 4 {
            // amax = 127 makes d = 1 exactly, so t = x: exact .5 ties (roundf rounds away) and
            // 0.5 - 2^-25, where t + 0.5 is inexact and only add.rz gives roundf's answer.
            const T: [f32; 8] = [0.5, -0.5, 1.5, -2.5, 0.49999997, -0.49999997, 0.29999998, 126.5];
            for (i, v) in input[3 * k..4 * k].iter_mut().enumerate() { *v = if i % 32 == 0 { 127.0 } else { T[i % 8] }; }
        }
        let weights = gguf_weights(r, t, num_experts * n * k / qk);
        let tw: Vec<f32> = (0..size_m).map(|_| f32_val(r, if specials > 0 { 97 } else { 0 }).abs().min(1.0)).collect();
        let out0 = r.bytes(size_m * n * 4);
        let _ = bs;
        let d_in = self.buf(&f32s(&input));
        let d_w = self.buf(&weights);
        let d_s = self.buf(&i32s(&sorted));
        let d_e = self.buf(&i32s(&eids));
        let d_tw = self.buf(&f32s(&tw));
        let (o_ref, o_ox) = (self.buf(&out0), self.buf(&out0));
        let twp = if with_tw { d_tw.cu_deviceptr() as *const f32 } else { std::ptr::null() };
        let st = self.stream_handle(use_null);
        unsafe {
            c_moe_gemm_gguf(d_in.cu_deviceptr() as _, d_w.cu_deviceptr() as _, d_s.cu_deviceptr() as _, d_e.cu_deviceptr() as _, twp,
                            o_ref.cu_deviceptr() as _, num_experts as i32, topk as i32, size_m as i32, n as i32, k as i32, t as i32, st);
            self.sync();
            launch::moe_gemm_gguf(d_in.cu_deviceptr() as _, d_w.cu_deviceptr() as _, d_s.cu_deviceptr() as _, d_e.cu_deviceptr() as _, twp,
                                  o_ox.cu_deviceptr() as _, num_experts as i32, topk as i32, size_m as i32, n as i32, k as i32, t as i32, st);
            self.sync();
        }
        self.calls += 1;
        let label = format!("moe_gemm_gguf {} tw={with_tw} E={num_experts} topk={topk} T={tokens} n={n} k={k} pat={pattern} bad_ids={bad_ids} null_stream={use_null}", GGUF_NAMES.get(t).unwrap_or(&"invalid"));
        self.compare(&label, "output", &o_ref, &o_ox, size_m * n * 4);
        if t < 6 {
            self.coverage(&label, &o_ref, &out0, 2);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_case(&mut self, dtype: i32, t: usize, with_tw: bool, num_experts: usize, topk: usize, tokens: usize, n: usize, k: usize,
                    pattern: u32, specials: u64, use_null: bool) {
        let (qk, _, _) = gguf_layout(t);
        let r = &mut self.rng;
        let (sorted, eids) = routing(r, tokens, topk, num_experts, pattern);
        let size_m = tokens * topk;
        let in_rows = if with_tw { size_m } else { tokens };
        let input: Vec<u16> = (0..in_rows * k).map(|_| if dtype == 0 { f16_bits(r, specials) } else { bf16_bits(r, specials) }).collect();
        let weights = gguf_weights(r, t, num_experts * n * k / qk);
        let tw: Vec<f32> = (0..size_m).map(|_| f32_val(r, if specials > 0 { 97 } else { 0 }).abs().min(1.0)).collect();
        let out0 = r.bytes(size_m * n * 4);
        let d_in = self.buf(&b16s(&input));
        let d_w = self.buf(&weights);
        let d_s = self.buf(&i32s(&sorted));
        let d_e = self.buf(&i32s(&eids));
        let d_tw = self.buf(&f32s(&tw));
        let (o_ref, o_ox) = (self.buf(&out0), self.buf(&out0));
        let twp = if with_tw { d_tw.cu_deviceptr() as *const f32 } else { std::ptr::null() };
        let st = self.stream_handle(use_null);
        unsafe {
            c_moe_gemm_gguf_prefill(d_in.cu_deviceptr() as _, d_w.cu_deviceptr() as _, d_s.cu_deviceptr() as _, d_e.cu_deviceptr() as _, twp,
                                    o_ref.cu_deviceptr() as _, num_experts as i32, topk as i32, size_m as i32, n as i32, k as i32, dtype, t as i32, st);
            self.sync();
            launch::moe_gemm_gguf_prefill(d_in.cu_deviceptr() as _, d_w.cu_deviceptr() as _, d_s.cu_deviceptr() as _, d_e.cu_deviceptr() as _, twp,
                                          o_ox.cu_deviceptr() as _, num_experts as i32, topk as i32, size_m as i32, n as i32, k as i32, dtype, t as i32, st);
            self.sync();
        }
        self.calls += 1;
        let label = format!("moe_gemm_gguf_prefill dtype={dtype} {} tw={with_tw} E={num_experts} topk={topk} T={tokens} n={n} k={k} pat={pattern} null_stream={use_null}", GGUF_NAMES.get(t).unwrap_or(&"invalid"));
        self.compare(&label, "output", &o_ref, &o_ox, size_m * n * 4);
        if t < 6 {
            self.coverage(&label, &o_ref, &out0, 2);
        }
    }
}

pub fn run() -> bool {
    let ctx = CudaContext::new(0).expect("cuda context");
    ctx.bind_to_thread().unwrap();
    let stream = ctx.new_stream().expect("stream");
    let mut g = Gate { ctx, stream, rng: Rng(0x5eed_1234_abcd), calls: 0, bytes: 0, elems: 0, written: 0, finite: 0, failures: Vec::new() };

    // ---- moe_gemm_wmma: (E, topk, tokens, n, k, pattern)
    let wmma_shapes: [(usize, usize, usize, usize, usize, u32); 8] = [
        (8, 2, 5, 64, 64, 0),
        (4, 1, 37, 50, 40, 1),   // n % 32 != 0, k % 16 == 8, segments > 32 rows
        (16, 4, 9, 96, 128, 2),  // most experts empty
        (1, 1, 70, 33, 16, 3),   // one expert, 3 m-tiles
        (64, 8, 12, 32, 256, 0), // many experts, some empty
        (3, 2, 100, 128, 72, 1), // 200 pairs, long segments
        (40, 3, 2, 17, 8, 0),    // tiny n / k
        (6, 2, 1, 200, 48, 0),   // single token (decode shape)
    ];
    // Four rounds over the whole matrix; the generator keeps running, so every round draws new data.
    for _round in 0..4 {
    for dtype in [0, 1] {
        for is_prefill in [true, false] {
            for with_tw in [false, true] {
                for (si, &(e, tk, t, n, k, p)) in wmma_shapes.iter().enumerate() {
                    let specials = if si % 2 == 0 { 0 } else { 61 };
                    g.wmma_case(dtype, is_prefill, with_tw, e, tk, t, n, k, p, specials, si == 3 && with_tw);
                }
            }
        }
    }
    }
    // thrust-path scan over > 1024 experts (multi-chunk)
    g.wmma_case(0, true, false, 1500, 4, 300, 32, 16, 0, 0, false);
    g.wmma_case(1, true, true, 2048, 2, 700, 40, 32, 1, 0, false);

    // ---- moe_gemm_gguf (decode) and moe_gemm_gguf_prefill: (E, topk, tokens, n, k-blocks, pattern)
    let gguf_shapes: [(usize, usize, usize, usize, usize, u32); 8] = [
        (8, 2, 150, 96, 2, 0), // 300 pairs: several m-tiles per expert
        (8, 2, 3, 64, 2, 0),
        (4, 1, 9, 50, 1, 1),
        (16, 4, 2, 7, 3, 2),
        (1, 1, 5, 130, 2, 3),
        (32, 8, 1, 40, 1, 0),
        (5, 2, 40, 36, 2, 1),
        (3, 3, 4, 33, 4, 0),
    ];
    for _round in 0..4 {
    for t in 0..6 {
        let (qk, _, _) = gguf_layout(t);
        for with_tw in [false, true] {
            for (si, &(e, tk, tok, n, kb, p)) in gguf_shapes.iter().enumerate() {
                // q8_0 gets longer K (in 32-blocks) so several lanes/iterations are exercised
                let k = if qk == 32 { kb * 96 + if si % 2 == 0 { 32 } else { 0 } } else { kb * qk };
                let specials = if si % 3 == 1 { 53 } else { 0 };
                g.gguf_case(t, with_tw, e, tk, tok, n, k, p, si % 2 == 1, specials, si == 4 && with_tw);
            }
        }
        for dtype in [0, 1] {
            for with_tw in [false, true] {
                for (si, &(e, tk, tok, n, kb, p)) in gguf_shapes.iter().enumerate() {
                    let k = if qk == 32 { kb * 96 + if si % 2 == 0 { 32 } else { 0 } } else { kb * qk };
                    let specials = if si % 3 == 1 { 53 } else { 0 };
                    g.prefill_case(dtype, t, with_tw, e, tk, tok, n, k, p, specials, si == 4 && with_tw);
                }
            }
        }
    }

    }
    // Selector values outside the documented ranges: wmma dtype 2 and gguf type 6 launch no GEMM
    // (offsets / quantisation still run); any non-zero prefill input_dtype selects bf16.
    g.wmma_case(2, true, false, 8, 2, 5, 64, 64, 0, 0, false);
    g.wmma_case(2, false, true, 8, 2, 5, 64, 64, 0, 0, false);
    g.gguf_case(6, false, 8, 2, 3, 64, 512, 0, false, 0, false);
    g.prefill_case(7, 1, false, 8, 2, 3, 64, 512, 0, 0, false);
    g.prefill_case(0, 6, true, 8, 2, 3, 64, 512, 0, 0, false);

    for f in g.failures.iter().take(40) {
        println!("  FAIL {f}");
    }
    println!("coverage: {} output elements, {} written by the reference ({:.1}%), {:.1}% of those finite",
             g.elems, g.written, 100.0 * g.written as f64 / g.elems as f64, 100.0 * g.finite as f64 / g.written.max(1) as f64);
    let ok = g.failures.is_empty();
    println!("moe: {} launcher calls, {} bytes compared, {} failing -> {}", g.calls, g.bytes, g.failures.len(),
             if ok { "PASS (bit-identical)" } else { "FAIL" });
    ok
}
