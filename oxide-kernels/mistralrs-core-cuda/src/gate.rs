//! Differential gate: mistralrs-core's real C launchers (libmistralrscuda.a, linked statically)
//! vs the pure-Rust launchers of `launch.rs`, called on identical inputs in the same primary
//! context and stream. Every byte of every buffer a launcher may write is compared: outputs and the
//! state buffers updated in place (recurrent states, conv states). Writable buffers start as
//! identical random bytes (the launchers never clear them).
use crate::launch;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use kdiff::Rng;
use std::ffi::c_void;
use std::sync::Arc;

type P = *const c_void;
type M = *mut c_void;

mod c {
    use super::{M, P};
    unsafe extern "C" {
        pub fn cuda_graph_copy_bytes(src: P, dst: M, n: i64, stream: i64) -> i32;
        pub fn v094_cuda_graph_copy_2d_bytes(src: P, dst: M, width: i64, height: i64, src_pitch: i64, dst_pitch: i64, stream: i64) -> i32;
        pub fn v094_pad_decode_input_u32(input: P, output: M, input_rows: i32, output_rows: i32, width: i32, stream: i64) -> i32;
        pub fn v094_pack_completion_input_u32(host: P, staged_rows: *const P, output: M, rows: i32, host_width: i32, staged_width: i32, stream: i64) -> i32;
        pub fn gated_delta_rule_recurrence(q: *const f32, k: *const f32, v: *const f32, g: *const f32, beta: *const f32, state: *mut f32, output: *mut f32, bh: i32, seq_len: i32, k_dim: i32, v_dim: i32, stream: i64);
        pub fn warp_gated_delta_rule_recurrence(q: *const f32, k: *const f32, v: *const f32, g: *const f32, beta: *const f32, state: *mut f32, output: *mut f32, bh: i32, seq_len: i32, k_dim: i32, v_dim: i32, stream: i64);
        pub fn chunked_gated_delta_rule_recurrence(q: *const f32, k: *const f32, v: *const f32, g: *const f32, beta: *const f32, state: *mut f32, output: *mut f32, bh: i32, seq_len: i32, k_dim: i32, v_dim: i32, stream: i64);
        pub fn causal_conv1d_update(x: P, weight: P, conv_state: M, output: M, batch_size: i32, conv_dim: i32, kernel_size: i32, dtype: i32, stream: i64);
        pub fn causal_conv1d_full(x: P, weight: P, conv_state_out: M, output: M, batch_size: i32, conv_dim: i32, seq_len: i32, kernel_size: i32, dtype: i32, stream: i64);
        pub fn gdn_rmsnorm_gated(x: P, gate: P, weight: P, output: M, rows: i32, hidden_dim: i32, eps: f32, dtype: i32, stream: i64);
        pub fn fused_gdn_gating(b: P, a: P, a_log: *const f32, dt_bias: *const f32, beta_out: M, g_out: M, total_elements: i32, num_heads: i32, dtype: i32, stream: i64);
        pub fn gdn_prepare_recurrence(mixed_qkv: P, b: P, a: P, a_log: *const f32, dt_bias: *const f32, q_out: *mut f32, k_out: *mut f32, v_out: *mut f32, g_out: *mut f32, beta_out: *mut f32, batch_size: i32, seq_len: i32, num_k_heads: i32, num_v_heads: i32, head_k_dim: i32, head_v_dim: i32, dtype: i32, stream: i64);
        pub fn gdn_decode_recurrence(mixed_qkv: P, b: P, a: P, a_log: *const f32, dt_bias: *const f32, state: *mut f32, output: *mut f32, batch_size: i32, num_k_heads: i32, num_v_heads: i32, head_k_dim: i32, head_v_dim: i32, dtype: i32, stream: i64);
        pub fn moe_gemv(input: P, weights: P, sorted_token_ids: *const i32, expert_ids: *const i32, topk_weights: *const f32, output: M, num_experts: i32, topk: i32, size_m: i32, size_n: i32, size_k: i32, dtype: i32, stream: i64);
        pub fn moe_gemm(input: P, weights: P, sorted_token_ids: *const i32, expert_ids: *const i32, topk_weights: *const f32, output: M, num_experts: i32, topk: i32, size_m: i32, size_n: i32, size_k: i32, dtype: i32, stream: i64);
        pub fn moe_gemm_wmma(input: P, weights: P, sorted_token_ids: *const i32, expert_ids: *const i32, topk_weights: *const f32, output: M, num_experts: i32, topk: i32, size_m: i32, size_n: i32, size_k: i32, dtype: i32, stream: i64);
        pub fn qk_rms_norm_rope(q: P, k: P, q_weight: P, k_weight: P, cos: P, sin: P, q_out: M, k_out: M, q_stride_b: i64, q_stride_h: i64, q_stride_s: i64, q_stride_d: i64, k_stride_b: i64, k_stride_h: i64, k_stride_s: i64, k_stride_d: i64, batch: i32, q_heads: i32, k_heads: i32, seq_len: i32, head_dim: i32, rot_dim: i32, cos_batch_stride: i32, q_eps: f32, k_eps: f32, is_neox: i32, dtype: i32, stream: i64);
        pub fn qk_rms_norm_rope_positions(q: P, k: P, q_weight: P, k_weight: P, cos: P, sin: P, positions: P, q_out: M, k_out: M, q_stride_b: i64, q_stride_h: i64, q_stride_s: i64, q_stride_d: i64, k_stride_b: i64, k_stride_h: i64, k_stride_s: i64, k_stride_d: i64, batch: i32, q_heads: i32, k_heads: i32, seq_len: i32, head_dim: i32, rot_dim: i32, q_eps: f32, k_eps: f32, is_neox: i32, dtype: i32, stream: i64);
        pub fn qkv_rms_norm_rope_positions(q: P, k: P, v: P, q_weight: P, k_weight: P, v_weight: P, cos: P, sin: P, positions: P, q_out: M, k_out: M, v_out: M, q_stride_b: i64, q_stride_h: i64, q_stride_s: i64, q_stride_d: i64, k_stride_b: i64, k_stride_h: i64, k_stride_s: i64, k_stride_d: i64, v_stride_b: i64, v_stride_h: i64, v_stride_s: i64, v_stride_d: i64, batch: i32, q_heads: i32, k_heads: i32, seq_len: i32, head_dim: i32, rot_dim: i32, q_eps: f32, k_eps: f32, v_eps: f32, is_neox: i32, dtype: i32, stream: i64);
        pub fn apply_sparse_penalties_f32(x: P, dst: M, token_ids: *const u32, counts: *const f32, n: i32, n_tokens: i32, f: f32, p: f32, r: f32, stream: i64);
        pub fn apply_sparse_logits_bias_f32(x: P, dst: M, token_ids: *const u32, biases: *const f32, n: i32, n_tokens: i32, stream: i64);
        pub fn apply_causal_mask_f32(scores: M, batch_heads: i32, q_len: i32, kv_len: i32, q_offset: i32, prefix_len: i32, stream: i64);
        pub fn rms_norm_residual_f32(x: P, r: P, w: P, s: P, d: M, nrows: i32, ncols: i32, eps: f32, stream: i64);
        pub fn rms_norm_residual_f16(x: P, r: P, w: P, s: P, d: M, nrows: i32, ncols: i32, eps: f32, stream: i64);
        pub fn rms_norm_residual_bf16(x: P, r: P, w: P, s: P, d: M, nrows: i32, ncols: i32, eps: f32, stream: i64);
        pub fn rms_norm_residual_then_rms_norm_f32(x: P, r: P, rw: P, s: P, nw: P, rd: M, nd: M, nrows: i32, ncols: i32, re: f32, ne: f32, stream: i64);
        pub fn rms_norm_residual_then_rms_norm_f16(x: P, r: P, rw: P, s: P, nw: P, rd: M, nd: M, nrows: i32, ncols: i32, re: f32, ne: f32, stream: i64);
        pub fn rms_norm_residual_then_rms_norm_bf16(x: P, r: P, rw: P, s: P, nw: P, rd: M, nd: M, nrows: i32, ncols: i32, re: f32, ne: f32, stream: i64);
        pub fn rms_norm_strided_4d_f32(x: P, w: P, d: M, sb: i64, sh: i64, ss: i64, sd: i64, b: i32, h: i32, s: i32, hd: i32, eps: f32, stream: i64);
        pub fn rms_norm_strided_4d_f16(x: P, w: P, d: M, sb: i64, sh: i64, ss: i64, sd: i64, b: i32, h: i32, s: i32, hd: i32, eps: f32, stream: i64);
        pub fn rms_norm_strided_4d_bf16(x: P, w: P, d: M, sb: i64, sh: i64, ss: i64, sd: i64, b: i32, h: i32, s: i32, hd: i32, eps: f32, stream: i64);
        pub fn asort_asc_f32(x: M, dst: M, nrows: i32, ncols: i32, inplace: bool, stream: i64);
        pub fn asort_asc_f16(x: M, dst: M, nrows: i32, ncols: i32, inplace: bool, stream: i64);
        pub fn asort_asc_bf16(x: M, dst: M, nrows: i32, ncols: i32, inplace: bool, stream: i64);
        pub fn asort_asc_f64(x: M, dst: M, nrows: i32, ncols: i32, inplace: bool, stream: i64);
        pub fn asort_asc_u8(x: M, dst: M, nrows: i32, ncols: i32, inplace: bool, stream: i64);
        pub fn asort_asc_u32(x: M, dst: M, nrows: i32, ncols: i32, inplace: bool, stream: i64);
        pub fn asort_asc_i64(x: M, dst: M, nrows: i32, ncols: i32, inplace: bool, stream: i64);
        pub fn asort_desc_f32(x: M, dst: M, nrows: i32, ncols: i32, inplace: bool, stream: i64);
        pub fn asort_desc_f16(x: M, dst: M, nrows: i32, ncols: i32, inplace: bool, stream: i64);
        pub fn asort_desc_bf16(x: M, dst: M, nrows: i32, ncols: i32, inplace: bool, stream: i64);
        pub fn asort_desc_f64(x: M, dst: M, nrows: i32, ncols: i32, inplace: bool, stream: i64);
        pub fn asort_desc_u8(x: M, dst: M, nrows: i32, ncols: i32, inplace: bool, stream: i64);
        pub fn asort_desc_u32(x: M, dst: M, nrows: i32, ncols: i32, inplace: bool, stream: i64);
        pub fn asort_desc_i64(x: M, dst: M, nrows: i32, ncols: i32, inplace: bool, stream: i64);
        pub fn topk_f32(input: P, v: M, i: M, nrows: i32, ncols: i32, k: i32, stream: i64);
        pub fn topk_f16(input: P, v: M, i: M, nrows: i32, ncols: i32, k: i32, stream: i64);
        pub fn topk_bf16(input: P, v: M, i: M, nrows: i32, ncols: i32, k: i32, stream: i64);
        pub fn moe_router_topk_f32(l: P, w: M, ids: M, b: P, es: P, nrows: i32, ne: i32, tk: i32, sm: i32, wm: i32, rn: bool, cl: bool, cmin: f32, cmax: f32, nmin: f32, os: f32, stream: i64);
        pub fn moe_router_topk_f16(l: P, w: M, ids: M, b: P, es: P, nrows: i32, ne: i32, tk: i32, sm: i32, wm: i32, rn: bool, cl: bool, cmin: f32, cmax: f32, nmin: f32, os: f32, stream: i64);
        pub fn moe_router_topk_bf16(l: P, w: M, ids: M, b: P, es: P, nrows: i32, ne: i32, tk: i32, sm: i32, wm: i32, rn: bool, cl: bool, cmin: f32, cmax: f32, nmin: f32, os: f32, stream: i64);
        pub fn topk_large_f32(input: *const f32, bv: *mut f32, bi: *mut u32, bm: *mut f32, bs: *mut f32, vo: *mut f32, io: *mut u32, so: *mut f32, ncols: i32, k: i32, chunk: i32, nblocks: i32, it: f32, stream: i64);
        pub fn topk_large_f32_packed(input: *const f32, bv: *mut f32, bi: *mut u32, bm: *mut f32, bs: *mut f32, po: *mut f32, ncols: i32, k: i32, chunk: i32, nblocks: i32, it: f32, stream: i64);
        pub fn top1_large_f32_packed(input: *const f32, bv: *mut f32, bi: *mut u32, po: *mut f32, ncols: i32, chunk: i32, nblocks: i32, stream: i64);
        pub fn selective_scan_cuda(x: *const f32, dt: *const f32, a: *const f32, b: *const f32, c: *const f32, d: *const f32, dt_bias: *const f32, state: *mut f32, y: *mut f32, batch_size: i32, n_heads: i32, head_dim: i32, d_state: i32, seq_len: i32, dt_min: f32, dt_max: f32, stream: i64);
    }
}

/// A launcher argument buffer: read-only input (one copy shared by both launchers), writable
/// (one copy per launcher with identical initial bytes; compared afterwards) or a null pointer.
pub enum Arg {
    In(Vec<u8>),
    Out(&'static str, Vec<u8>),
    Null,
}

pub struct Gate {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    pub rng: Rng,
    calls: usize,
    bytes: usize,
    written: usize,
    elems: usize,
    failures: Vec<String>,
    /// Current family, for the per-family summary.
    family: &'static str,
    fam_stats: Vec<(&'static str, usize, usize, usize)>,
}

// ------------------------------------------------------------------------------------------------
// value generators

pub fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
pub fn u16s(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
pub fn i32s(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
pub fn u32s(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

impl Gate {
    pub fn uni(&mut self) -> f64 {
        (self.rng.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// f32 uniform in [lo, hi), with an adversarial value every `special` draws (0: never).
    pub fn f32v(&mut self, lo: f32, hi: f32, special: u64) -> f32 {
        if special > 0 && self.rng.next() % special == 0 {
            const S: [u32; 12] = [0x0000_0000, 0x8000_0000, 0x7f80_0000, 0xff80_0000, 0x7fc0_0000, 0xffc0_1234, 0x0000_0001, 0x8000_0123, 0x007f_ffff, 0x7f7f_ffff, 0xff7f_ffff, 0x0080_0000];
            let k = self.rng.next() % 13;
            return f32::from_bits(if k == 12 { self.rng.next() as u32 } else { S[k as usize] });
        }
        (lo as f64 + (hi - lo) as f64 * self.uni()) as f32
    }
    pub fn f32vec(&mut self, n: usize, lo: f32, hi: f32, special: u64) -> Vec<f32> {
        (0..n).map(|_| self.f32v(lo, hi, special)).collect()
    }
    /// A half value (dtype 0: f16, else bf16) uniform in [lo, hi) rounded to nearest, with
    /// adversarial patterns every `special` draws.
    pub fn h16(&mut self, dtype: i32, lo: f32, hi: f32, special: u64) -> u16 {
        if special > 0 && self.rng.next() % special == 0 {
            const SH: [u16; 11] = [0x0000, 0x8000, 0x7c00, 0xfc00, 0x7e00, 0xfe01, 0x0001, 0x83ff, 0x7bff, 0xfbff, 0x0400];
            const SB: [u16; 11] = [0x0000, 0x8000, 0x7f80, 0xff80, 0x7fc0, 0xffc1, 0x0001, 0x807f, 0x7f7f, 0xff7f, 0x0080];
            let k = self.rng.next() % 12;
            return if k == 11 { self.rng.next() as u16 } else if dtype == 0 { SH[k as usize] } else { SB[k as usize] };
        }
        let v = (lo as f64 + (hi - lo) as f64 * self.uni()) as f32;
        if dtype == 0 { f32_to_f16(v) } else { f32_to_bf16(v) }
    }
    pub fn h16vec(&mut self, n: usize, dtype: i32, lo: f32, hi: f32, special: u64) -> Vec<u16> {
        (0..n).map(|_| self.h16(dtype, lo, hi, special)).collect()
    }
    pub fn rng_h16vec(&mut self, n: usize, dtype: i32, lo: f32, hi: f32) -> Vec<u16> {
        self.h16vec(n, dtype, lo, hi, 0)
    }
    pub fn bytes(&mut self, n: usize) -> Vec<u8> {
        self.rng.bytes(n)
    }
    pub fn pick<T: Copy>(&mut self, v: &[T]) -> T {
        v[(self.rng.next() % v.len() as u64) as usize]
    }

    fn buf(&self, bytes: &[u8]) -> DeviceBuffer<u8> {
        DeviceBuffer::from_host(&self.stream, if bytes.is_empty() { &[0u8][..] } else { bytes }).unwrap()
    }
    fn sync(&self, what: &str) {
        self.ctx.synchronize().unwrap_or_else(|e| panic!("synchronize after {what}: {e:?}"));
    }
    pub fn stream_handle(&self, use_null: bool) -> i64 {
        if use_null { 0 } else { self.stream.cu_stream() as usize as i64 }
    }

    pub fn family(&mut self, name: &'static str) {
        self.close_family();
        self.family = name;
    }
    fn close_family(&mut self) {
        if !self.family.is_empty() {
            let prev: (usize, usize, usize) = self.fam_stats.iter().fold((0, 0, 0), |a, s| (a.0 + s.1, a.1 + s.2, a.2 + s.3));
            let (c, b, f) = (self.calls - prev.0, self.bytes - prev.1, self.failures.len() - prev.2);
            self.fam_stats.push((self.family, c, b, f));
            self.family = "";
        }
    }

    /// Run one launcher call pair. `call(rust, ptrs, stream)` invokes the C (`rust == false`) or
    /// Rust launcher with the device pointers of `args` (null for `Arg::Null`). `expect_write`:
    /// the reference must change at least one writable byte (guards against vacuous cases).
    pub fn case(&mut self, label: String, args: Vec<Arg>, use_null: bool, expect_write: bool, call: impl Fn(bool, &[u64], i64)) {
        let st = self.stream_handle(use_null);
        let mut ins: Vec<Option<DeviceBuffer<u8>>> = Vec::new();
        let mut outs: Vec<(usize, &'static str, Vec<u8>, DeviceBuffer<u8>, DeviceBuffer<u8>)> = Vec::new();
        for (i, a) in args.iter().enumerate() {
            match a {
                Arg::In(v) => ins.push(Some(self.buf(v))),
                Arg::Out(name, v) => {
                    ins.push(None);
                    outs.push((i, name, v.clone(), self.buf(v), self.buf(v)));
                }
                Arg::Null => ins.push(None),
            }
        }
        let ptrs = |which: usize| -> Vec<u64> {
            (0..args.len())
                .map(|i| {
                    if let Some(b) = &ins[i] {
                        b.cu_deviceptr()
                    } else if let Some(o) = outs.iter().find(|o| o.0 == i) {
                        if which == 0 { o.3.cu_deviceptr() } else { o.4.cu_deviceptr() }
                    } else {
                        0
                    }
                })
                .collect()
        };
        self.sync("upload");
        call(false, &ptrs(0), st);
        self.sync(&format!("C {label}"));
        call(true, &ptrs(1), st);
        self.sync(&format!("Rust {label}"));
        self.calls += 1;
        let mut any_written = false;
        let mut bad_bufs = Vec::new();
        for (_, name, init, r, o) in &outs {
            let x = r.to_host_vec(&self.stream).unwrap();
            let y = o.to_host_vec(&self.stream).unwrap();
            self.bytes += x.len().min(init.len().max(1));
            let n = init.len();
            let w = (0..n).filter(|&i| x[i] != init[i]).count();
            self.written += w;
            self.elems += n;
            any_written |= w > 0;
            let bad: Vec<usize> = (0..n).filter(|&i| x[i] != y[i]).collect();
            if !bad.is_empty() {
                let i = bad[0];
                let j = i & !3;
                let word = |v: &Vec<u8>| if j + 4 <= n { format!("{:08x}", u32::from_le_bytes([v[j], v[j + 1], v[j + 2], v[j + 3]])) } else { String::new() };
                bad_bufs.push(format!(
                    "[{name}] {} of {n} bytes differ (first at byte {i}: ref {:#04x} oxide {:#04x}; word ref {} oxide {})",
                    bad.len(), x[i], y[i], word(&x), word(&y)
                ));
            }
        }
        if !bad_bufs.is_empty() {
            // Is the reference itself deterministic on this input?
            let fresh: Vec<DeviceBuffer<u8>> = outs.iter().map(|o| self.buf(&o.2)).collect();
            let p2: Vec<u64> = (0..args.len())
                .map(|i| {
                    if let Some(b) = &ins[i] {
                        b.cu_deviceptr()
                    } else if let Some(k) = outs.iter().position(|o| o.0 == i) {
                        fresh[k].cu_deviceptr()
                    } else {
                        0
                    }
                })
                .collect();
            self.sync("upload");
            call(false, &p2, st);
            self.sync("C rerun");
            let mut stable = true;
            for (k, o) in outs.iter().enumerate() {
                if fresh[k].to_host_vec(&self.stream).unwrap() != o.3.to_host_vec(&self.stream).unwrap() {
                    stable = false;
                }
            }
            self.failures.push(format!("{label}: {}{}", bad_bufs.join("; "), if stable { "" } else { " (REFERENCE NON-DETERMINISTIC)" }));
        }
        if expect_write && !any_written {
            self.failures.push(format!("{label}: reference wrote nothing (vacuous case)"));
        }
    }

    /// Scalar-result comparison (e.g. a launcher's return code).
    pub fn check_eq<T: PartialEq + std::fmt::Debug>(&mut self, label: &str, what: &str, r: T, o: T) {
        if r != o {
            self.failures.push(format!("{label} [{what}]: ref {r:?} oxide {o:?}"));
        }
    }
}

pub fn f32_to_bf16(v: f32) -> u16 {
    let b = v.to_bits();
    if v.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}
pub fn f32_to_f16(v: f32) -> u16 {
    // round-to-nearest-even via f64 arithmetic on the value (inputs are finite and moderate)
    let x = v as f64;
    let sign = if x.is_sign_negative() { 0x8000u16 } else { 0 };
    let a = x.abs();
    if a == 0.0 {
        return sign;
    }
    if a >= 65520.0 {
        return sign | 0x7c00;
    }
    let e = a.log2().floor() as i32;
    if e < -14 {
        let m = (a / 2f64.powi(-24)).round_ties_even() as u16;
        return sign | m;
    }
    let m = (a / 2f64.powi(e) - 1.0) * 1024.0;
    let mut mi = m.round_ties_even() as u32;
    let mut ee = e + 15;
    if mi == 1024 {
        mi = 0;
        ee += 1;
    }
    sign | ((ee as u16) << 10) | mi as u16
}

// ================================================================================================
// families

fn graph(g: &mut Gate) {
    g.family("graph");
    for &(n, use_null) in &[(-1i64, false), (0, false), (1, false), (7, true), (4099, false), (65536, true)] {
        let len = n.max(0) as usize;
        let src = g.bytes(len.max(1));
        let dst = g.bytes(len.max(1));
        let rc = std::cell::Cell::new((0, 0));
        let label = format!("cuda_graph_copy_bytes n={n} null_stream={use_null}");
        g.case(label.clone(), vec![Arg::In(src), Arg::Out("dst", dst)], use_null, n > 0, |rust, p, st| unsafe {
            let r = if rust {
                launch::cuda_graph_copy_bytes(p[0] as P, p[1] as M, n, st)
            } else {
                c::cuda_graph_copy_bytes(p[0] as P, p[1] as M, n, st)
            };
            let mut v = rc.get();
            if rust { v.1 = r } else { v.0 = r }
            rc.set(v);
        });
        let (a, b) = rc.get();
        g.check_eq(&label, "return", a, b);
    }
}

/// v0.9.4 CUDA-graph helpers: cuda_graph_copy_2d_bytes (cuMemcpy2DAsync), pad_decode_input_u32,
/// pack_completion_input_u32 (64-row chunks, the row-pointer table by value).
fn graph094(g: &mut Gate) {
    g.family("graph094");
    for &(w, h, sp, dp, use_null) in &[(-1i64, 1i64, 4i64, 4i64, false), (4, 3, 2, 8, false), (0, 5, 8, 8, true), (7, 0, 8, 8, false),
                                       (1, 1, 1, 1, false), (13, 9, 16, 32, true), (256, 31, 260, 512, false), (4096, 4, 4096, 8192, true)] {
        let src = g.bytes((sp.max(1) * h.max(1)) as usize + 64);
        let dst = g.bytes((dp.max(1) * h.max(1)) as usize + 64);
        let rc = std::cell::Cell::new((0, 0));
        let label = format!("cuda_graph_copy_2d_bytes w={w} h={h} sp={sp} dp={dp} null_stream={use_null}");
        g.case(label.clone(), vec![Arg::In(src), Arg::Out("dst", dst)], use_null, w > 0 && h > 0 && sp >= w && dp >= w, |rust, p, st| unsafe {
            let r = if rust {
                launch::cuda_graph_copy_2d_bytes(p[0] as P, p[1] as M, w, h, sp, dp, st)
            } else {
                c::v094_cuda_graph_copy_2d_bytes(p[0] as P, p[1] as M, w, h, sp, dp, st)
            };
            let mut v = rc.get();
            if rust { v.1 = r } else { v.0 = r }
            rc.set(v);
        });
        let (a, b) = rc.get();
        g.check_eq(&label, "return", a, b);
    }
    for &(ir, or, w, use_null) in &[(1i32, 1i32, 1i32, false), (1, 4, 1, true), (3, 8, 2, false), (5, 5, 7, false), (2, 64, 3, true), (31, 37, 11, false),
                                    (0, 4, 1, false), (4, 3, 1, false), (2, 4, 0, false)] {
        let input = g.bytes((ir.max(1) * w.max(1)) as usize * 4);
        let output = g.bytes((or.max(1) * w.max(1)) as usize * 4 + 16);
        let rc = std::cell::Cell::new((0, 0));
        let label = format!("pad_decode_input_u32 in={ir} out={or} w={w} null_stream={use_null}");
        g.case(label.clone(), vec![Arg::In(input), Arg::Out("out", output)], use_null, ir > 0 && or >= ir && w > 0, |rust, p, st| unsafe {
            let r = if rust {
                launch::pad_decode_input_u32(p[0] as P, p[1] as M, ir, or, w, st)
            } else {
                c::v094_pad_decode_input_u32(p[0] as P, p[1] as M, ir, or, w, st)
            };
            let mut v = rc.get();
            if rust { v.1 = r } else { v.0 = r }
            rc.set(v);
        });
        let (a, b) = rc.get();
        g.check_eq(&label, "return", a, b);
    }
    // rows spanning 1, 2 and 3 launches (64 rows each); staged rows are separate device buffers
    for &(rows, hw, sw, use_null) in &[(1i32, 1i32, 1i32, false), (3, 5, 2, true), (64, 7, 1, false), (65, 3, 4, false), (130, 1, 9, true), (0, 1, 1, false), (2, 0, 1, false)] {
        let host = g.bytes((rows.max(1) * hw.max(1)) as usize * 4);
        let out = g.bytes((rows.max(1) as usize) * ((hw.max(0) + sw.max(0)) as usize) * 4 + 16);
        let mut args = vec![Arg::In(host), Arg::Out("out", out)];
        for _ in 0..rows.max(0) {
            args.push(Arg::In(g.bytes(sw.max(1) as usize * 4)));
        }
        let rc = std::cell::Cell::new((0, 0));
        let label = format!("pack_completion_input_u32 rows={rows} host_width={hw} staged_width={sw} null_stream={use_null}");
        g.case(label.clone(), args, use_null, rows > 0 && hw > 0 && sw > 0, |rust, p, st| unsafe {
            let staged: Vec<P> = p[2..].iter().map(|&x| x as P).collect();
            let r = if rust {
                launch::pack_completion_input_u32(p[0] as P, staged.as_ptr() as *const *const c_void, p[1] as M, rows, hw, sw, st)
            } else {
                c::v094_pack_completion_input_u32(p[0] as P, staged.as_ptr(), p[1] as M, rows, hw, sw, st)
            };
            let mut v = rc.get();
            if rust { v.1 = r } else { v.0 = r }
            rc.set(v);
        });
        let (a, b) = rc.get();
        g.check_eq(&label, "return", a, b);
    }
}

/// q, k, v, g, beta, state, output for the recurrence launchers.
#[allow(clippy::type_complexity)]
fn gdr_inputs(g: &mut Gate, bh: usize, seq: usize, kd: usize, vd: usize, sp: u64) -> Vec<Arg> {
    let q = g.f32vec(bh * seq * kd, -0.3, 0.3, sp);
    let k = g.f32vec(bh * seq * kd, -0.3, 0.3, sp);
    let v = g.f32vec(bh * seq * vd, -2.0, 2.0, sp);
    let gg = g.f32vec(bh * seq, -1.5, 0.05, sp);
    let beta = g.f32vec(bh * seq, 0.0, 1.0, sp);
    let state = g.f32vec(bh * kd * vd, -1.0, 1.0, sp);
    let out = g.bytes(bh * seq * vd * 4);
    vec![Arg::In(f32s(&q)), Arg::In(f32s(&k)), Arg::In(f32s(&v)), Arg::In(f32s(&gg)), Arg::In(f32s(&beta)), Arg::Out("state", f32s(&state)), Arg::Out("output", out)]
}

fn gdn_recurrence(g: &mut Gate) {
    g.family("gdn recurrence");
    type L = unsafe extern "C" fn(*const f32, *const f32, *const f32, *const f32, *const f32, *mut f32, *mut f32, i32, i32, i32, i32, i64);
    let launchers: [(&str, L, L); 3] = [
        ("gated_delta_rule_recurrence", c::gated_delta_rule_recurrence, launch::gated_delta_rule_recurrence),
        ("warp_gated_delta_rule_recurrence", c::warp_gated_delta_rule_recurrence, launch::warp_gated_delta_rule_recurrence),
        ("chunked_gated_delta_rule_recurrence", c::chunked_gated_delta_rule_recurrence, launch::chunked_gated_delta_rule_recurrence),
    ];
    // (bh, seq_len, k_dim, v_dim). Tiled/chunked/fallback kernels need v_dim % 64 == 0 (otherwise
    // the reference reads unloaded shared memory); the warp kernel takes any v_dim.
    let shapes: [(usize, usize, usize, usize); 14] = [
        (1, 1, 128, 128),
        (2, 3, 128, 64),
        (3, 64, 128, 128),
        (1, 65, 128, 64),
        (2, 130, 128, 128),
        (1, 7, 64, 64),
        (3, 64, 64, 128),
        (2, 97, 64, 192),
        (1, 5, 32, 64),
        (2, 9, 96, 128),
        (1, 4, 100, 64),
        (1, 3, 256, 64),
        (2, 2, 1, 64),
        (1, 0, 128, 64),
    ];
    for (li, &(name, cl, rl)) in launchers.iter().enumerate() {
        for (si, &(bh, seq, kd, vd)) in shapes.iter().enumerate() {
            for sp in [0u64, 41] {
                let args = gdr_inputs(g, bh, seq, kd, vd, sp);
                let use_null = (si + li) % 4 == 1;
                let label = format!("{name} bh={bh} seq={seq} k={kd} v={vd} specials={sp} null_stream={use_null}");
                g.case(label, args, use_null, seq > 0, |rust, p, st| unsafe {
                    let f = if rust { rl } else { cl };
                    f(p[0] as _, p[1] as _, p[2] as _, p[3] as _, p[4] as _, p[5] as _, p[6] as _, bh as i32, seq as i32, kd as i32, vd as i32, st)
                });
            }
        }
        // the warp kernel with v_dim not a multiple of 4 or 64
        if name.starts_with("warp") {
            for &(bh, seq, kd, vd) in &[(2usize, 5usize, 128usize, 70usize), (1, 3, 64, 5), (3, 8, 128, 1), (1, 33, 64, 131)] {
                let args = gdr_inputs(g, bh, seq, kd, vd, 53);
                let label = format!("{name} bh={bh} seq={seq} k={kd} v={vd} odd-v");
                g.case(label, args, false, true, |rust, p, st| unsafe {
                    let f = if rust { rl } else { cl };
                    f(p[0] as _, p[1] as _, p[2] as _, p[3] as _, p[4] as _, p[5] as _, p[6] as _, bh as i32, seq as i32, kd as i32, vd as i32, st)
                });
            }
        }
    }
}

fn gdn_conv(g: &mut Gate) {
    g.family("gdn causal_conv1d");
    for dtype in [0, 1, 5] {
        for &(bs, cd, ks) in &[(1usize, 1usize, 4usize), (2, 255, 4), (3, 256, 2), (1, 300, 1), (2, 600, 5), (1, 64, 3)] {
            for sp in [0u64, 13] {
                let x = g.h16vec(bs * cd, dtype, -3.0, 3.0, sp);
                let w = g.h16vec(cd * ks, dtype, -1.0, 1.0, sp);
                let cs = g.h16vec(bs * cd * ks, dtype, -3.0, 3.0, sp);
                let out = g.bytes(bs * cd * 2);
                let label = format!("causal_conv1d_update dtype={dtype} B={bs} C={cd} K={ks} sp={sp}");
                let use_null = cd == 300;
                g.case(label, vec![Arg::In(u16s(&x)), Arg::In(u16s(&w)), Arg::Out("conv_state", u16s(&cs)), Arg::Out("output", out)], use_null, true, |rust, p, st| unsafe {
                    if rust {
                        launch::causal_conv1d_update(p[0] as P, p[1] as P, p[2] as M, p[3] as M, bs as i32, cd as i32, ks as i32, dtype, st)
                    } else {
                        c::causal_conv1d_update(p[0] as P, p[1] as P, p[2] as M, p[3] as M, bs as i32, cd as i32, ks as i32, dtype, st)
                    }
                });
            }
        }
        for &(bs, cd, sl, ks) in &[(1usize, 1usize, 1usize, 4usize), (2, 255, 3, 4), (1, 256, 4, 4), (2, 300, 17, 4), (1, 64, 5, 6), (3, 70, 2, 1), (1, 513, 9, 3)] {
            for sp in [0u64, 13] {
                let x = g.h16vec(bs * cd * sl, dtype, -3.0, 3.0, sp);
                let w = g.h16vec(cd * ks, dtype, -1.0, 1.0, sp);
                let cs = g.bytes(bs * cd * ks * 2);
                let out = g.bytes(bs * cd * sl * 2);
                let label = format!("causal_conv1d_full dtype={dtype} B={bs} C={cd} S={sl} K={ks} sp={sp}");
                let use_null = cd == 300;
                g.case(label, vec![Arg::In(u16s(&x)), Arg::In(u16s(&w)), Arg::Out("conv_state_out", cs), Arg::Out("output", out)], use_null, true, |rust, p, st| unsafe {
                    if rust {
                        launch::causal_conv1d_full(p[0] as P, p[1] as P, p[2] as M, p[3] as M, bs as i32, cd as i32, sl as i32, ks as i32, dtype, st)
                    } else {
                        c::causal_conv1d_full(p[0] as P, p[1] as P, p[2] as M, p[3] as M, bs as i32, cd as i32, sl as i32, ks as i32, dtype, st)
                    }
                });
            }
        }
    }
}

fn gdn_misc(g: &mut Gate) {
    g.family("gdn gating/rmsnorm");
    for dtype in [0, 1, 3] {
        for &(total, heads) in &[(1usize, 1usize), (255, 3), (256, 16), (1000, 32), (37, 37)] {
            for sp in [0u64, 7] {
                let b = g.h16vec(total, dtype, -8.0, 8.0, sp);
                let a = g.h16vec(total, dtype, -30.0, 30.0, sp);
                let al = g.f32vec(heads, -3.0, 2.0, sp);
                let db = g.f32vec(heads, -3.0, 3.0, sp);
                let label = format!("fused_gdn_gating dtype={dtype} total={total} heads={heads} sp={sp}");
                let (o1, o2) = (g.bytes(total * 2), g.bytes(total * 2));
                g.case(label, vec![Arg::In(u16s(&b)), Arg::In(u16s(&a)), Arg::In(f32s(&al)), Arg::In(f32s(&db)), Arg::Out("beta_out", o1), Arg::Out("g_out", o2)], total == 37, true, |rust, p, st| unsafe {
                    if rust {
                        launch::fused_gdn_gating(p[0] as P, p[1] as P, p[2] as _, p[3] as _, p[4] as M, p[5] as M, total as i32, heads as i32, dtype, st)
                    } else {
                        c::fused_gdn_gating(p[0] as P, p[1] as P, p[2] as _, p[3] as _, p[4] as M, p[5] as M, total as i32, heads as i32, dtype, st)
                    }
                });
            }
        }
        for &(rows, hd) in &[(1usize, 1usize), (3, 64), (2, 128), (5, 129), (1, 1000), (4, 256)] {
            for sp in [0u64, 11] {
                let x = g.h16vec(rows * hd, dtype, -4.0, 4.0, sp);
                let gate = g.h16vec(rows * hd, dtype, -12.0, 12.0, sp);
                let w = g.h16vec(hd, dtype, -2.0, 2.0, sp);
                let eps = g.pick(&[1e-6f32, 1e-5, 0.0, 0.5]);
                let out = g.bytes(rows * hd * 2);
                let label = format!("gdn_rmsnorm_gated dtype={dtype} rows={rows} hidden={hd} eps={eps} sp={sp}");
                g.case(label, vec![Arg::In(u16s(&x)), Arg::In(u16s(&gate)), Arg::In(u16s(&w)), Arg::Out("output", out)], hd == 129, true, |rust, p, st| unsafe {
                    if rust {
                        launch::gdn_rmsnorm_gated(p[0] as P, p[1] as P, p[2] as P, p[3] as M, rows as i32, hd as i32, eps, dtype, st)
                    } else {
                        c::gdn_rmsnorm_gated(p[0] as P, p[1] as P, p[2] as P, p[3] as M, rows as i32, hd as i32, eps, dtype, st)
                    }
                });
            }
        }
    }
}

fn gdn_prepare_decode(g: &mut Gate) {
    g.family("gdn prepare/decode");
    for dtype in [0, 1, 2] {
        // (batch, seq, nk, nv, hk, hv)
        for &(bs, sl, nk, nv, hk, hv) in &[
            (1usize, 1usize, 1usize, 1usize, 128usize, 128usize),
            (2, 3, 2, 4, 128, 128),
            (1, 2, 1, 2, 64, 64),
            (2, 1, 2, 2, 100, 50),
            (1, 2, 1, 1, 300, 300),
            (1, 1, 2, 6, 32, 70),
            (3, 2, 1, 3, 256, 16),
        ] {
            for sp in [0u64, 17] {
                let conv = 2 * nk * hk + nv * hv;
                let m = g.h16vec(bs * sl * conv, dtype, -2.0, 2.0, sp);
                let b = g.h16vec(bs * sl * nv, dtype, -6.0, 6.0, sp);
                let a = g.h16vec(bs * sl * nv, dtype, -25.0, 25.0, sp);
                let al = g.f32vec(nv, -3.0, 1.0, sp);
                let db = g.f32vec(nv, -2.0, 2.0, sp);
                let bh = bs * nv;
                let (qo, ko, vo) = (g.bytes(bh * sl * hk * 4), g.bytes(bh * sl * hk * 4), g.bytes(bh * sl * hv * 4));
                let (go, bo) = (g.bytes(bh * sl * 4), g.bytes(bh * sl * 4));
                let label = format!("gdn_prepare_recurrence dtype={dtype} B={bs} S={sl} nk={nk} nv={nv} hk={hk} hv={hv} sp={sp}");
                let args = vec![
                    Arg::In(u16s(&m)), Arg::In(u16s(&b)), Arg::In(u16s(&a)), Arg::In(f32s(&al)), Arg::In(f32s(&db)),
                    Arg::Out("q_out", qo), Arg::Out("k_out", ko), Arg::Out("v_out", vo), Arg::Out("g_out", go), Arg::Out("beta_out", bo),
                ];
                g.case(label, args, hk == 100, true, |rust, p, st| unsafe {
                    let f = if rust { launch::gdn_prepare_recurrence } else { c::gdn_prepare_recurrence };
                    f(p[0] as P, p[1] as P, p[2] as P, p[3] as _, p[4] as _, p[5] as _, p[6] as _, p[7] as _, p[8] as _, p[9] as _,
                      bs as i32, sl as i32, nk as i32, nv as i32, hk as i32, hv as i32, dtype, st)
                });
                if hk <= 256 {
                    // decode: one token per sequence
                    let m = g.h16vec(bs * conv, dtype, -2.0, 2.0, sp);
                    let b = g.h16vec(bs * nv, dtype, -6.0, 6.0, sp);
                    let a = g.h16vec(bs * nv, dtype, -25.0, 25.0, sp);
                    let state = g.f32vec(bh * hk * hv, -1.0, 1.0, sp);
                    let out = g.bytes(bh * hv * 4);
                    let label = format!("gdn_decode_recurrence dtype={dtype} B={bs} nk={nk} nv={nv} hk={hk} hv={hv} sp={sp}");
                    let args = vec![
                        Arg::In(u16s(&m)), Arg::In(u16s(&b)), Arg::In(u16s(&a)), Arg::In(f32s(&al)), Arg::In(f32s(&db)),
                        Arg::Out("state", f32s(&state)), Arg::Out("output", out),
                    ];
                    g.case(label, args, hk == 32, true, |rust, p, st| unsafe {
                        let f = if rust { launch::gdn_decode_recurrence } else { c::gdn_decode_recurrence };
                        f(p[0] as P, p[1] as P, p[2] as P, p[3] as _, p[4] as _, p[5] as _, p[6] as _, bs as i32, nk as i32, nv as i32, hk as i32, hv as i32, dtype, st)
                    });
                }
            }
        }
    }
}

fn ssm(g: &mut Gate) {
    g.family("ssm");
    // d_state must be a multiple of 32 (<= 256) or > 256 for a race-free reference.
    for &(bs, nh, hd, ds, sl) in &[
        (1usize, 1usize, 1usize, 32usize, 1usize),
        (2, 3, 4, 64, 5),
        (1, 2, 8, 128, 33),
        (2, 1, 2, 256, 7),
        (1, 2, 3, 288, 4),
        (1, 4, 16, 128, 2),
        (3, 2, 5, 32, 17),
        (1, 1, 2, 64, 0),
    ] {
        for sp in [0u64, 19] {
            let x = g.f32vec(bs * sl * nh * hd, -2.0, 2.0, sp);
            let dt = g.f32vec(bs * sl * nh, -6.0, 25.0, sp);
            let a = g.f32vec(nh, -4.0, -0.01, sp);
            let b = g.f32vec(bs * sl * nh * ds, -1.0, 1.0, sp);
            let cc = g.f32vec(bs * sl * nh * ds, -1.0, 1.0, sp);
            let d = g.f32vec(nh, -1.0, 1.0, sp);
            let db = g.f32vec(nh, -2.0, 2.0, sp);
            let state = g.f32vec(bs * nh * hd * ds, -1.0, 1.0, sp);
            let y = g.bytes(bs * sl * nh * hd * 4);
            let (dmin, dmax) = g.pick(&[(0.0f32, f32::INFINITY), (0.001, 0.1), (0.01, 10.0), (0.5, 0.25)]);
            let label = format!("selective_scan_cuda B={bs} H={nh} hd={hd} ds={ds} S={sl} dt=[{dmin},{dmax}] sp={sp}");
            let args = vec![
                Arg::In(f32s(&x)), Arg::In(f32s(&dt)), Arg::In(f32s(&a)), Arg::In(f32s(&b)), Arg::In(f32s(&cc)), Arg::In(f32s(&d)), Arg::In(f32s(&db)),
                Arg::Out("state", f32s(&state)), Arg::Out("y", y),
            ];
            g.case(label, args, hd == 5, sl > 0, |rust, p, st| unsafe {
                let f = if rust { launch::selective_scan_cuda } else { c::selective_scan_cuda };
                f(p[0] as _, p[1] as _, p[2] as _, p[3] as _, p[4] as _, p[5] as _, p[6] as _, p[7] as _, p[8] as _, bs as i32, nh as i32, hd as i32, ds as i32, sl as i32, dmin, dmax, st)
            });
        }
    }
}


/// Routing for `tokens` tokens with top-`topk` experts out of `e`: (sorted_token_ids, expert_ids)
/// of length tokens*topk grouped by expert (ascending). pattern 0: uniform; 1: skewed to few
/// experts; 2: only 3 experts used; 3: one expert.
fn routing(g: &mut Gate, tokens: usize, topk: usize, e: usize, pattern: u32) -> (Vec<i32>, Vec<i32>) {
    let pairs = tokens * topk;
    let mut ex: Vec<(i32, i32)> = (0..pairs)
        .map(|p| {
            let x = match pattern {
                0 => g.rng.next() % e as u64,
                1 => {
                    if g.rng.next() % 4 != 0 { g.rng.next() % (e as u64).min(2) } else { g.rng.next() % e as u64 }
                }
                2 => [0u64, e as u64 / 2, e as u64 - 1][(g.rng.next() % 3) as usize],
                _ => (e - 1) as u64,
            };
            (x as i32, p as i32)
        })
        .collect();
    ex.sort();
    (ex.iter().map(|x| x.1).collect(), ex.iter().map(|x| x.0).collect())
}

type MoeL = unsafe extern "C" fn(P, P, *const i32, *const i32, *const f32, M, i32, i32, i32, i32, i32, i32, i64);

#[allow(clippy::too_many_arguments)]
fn moe_case(g: &mut Gate, name: &str, cl: MoeL, rl: MoeL, dtype: i32, with_tw: bool, e: usize, topk: usize, tokens: usize, n: usize, k: usize,
            pattern: u32, bad_ids: bool, sp: u64, use_null: bool) {
    let (sorted, mut eids) = routing(g, tokens, topk, e, pattern);
    if bad_ids {
        for (i, x) in eids.iter_mut().enumerate() {
            if i % 5 == 3 {
                *x = if i % 2 == 0 { -1 } else { (e + i) as i32 };
            }
        }
    }
    let size_m = tokens * topk;
    let in_rows = if with_tw { size_m } else { tokens };
    let dt = if dtype == 0 { 0 } else { 1 };
    let mut input = g.h16vec(in_rows * k, dt, -1.0, 1.0, sp);
    let mut weights = g.h16vec(e * n * k, dt, -1.0, 1.0, sp);
    if name == "moe_gemm" && k > 64 {
        // moe_gemm's reference races across K tiles (a warp may load tile t+1 into shared memory
        // while another still reads tile t). Rows periodic in K with period 64 make every tile
        // identical, so the racing stores rewrite the same bytes and the result is deterministic.
        for v in [&mut input, &mut weights] {
            for i in 0..v.len() {
                if i % k >= 64 {
                    v[i] = v[i - 64];
                }
            }
        }
    }
    let tw = g.f32vec(size_m, 0.0, 1.0, if sp > 0 { 31 } else { 0 });
    let out = g.bytes(size_m * n * 2);
    let label = format!("{name} dtype={dtype} tw={with_tw} E={e} topk={topk} T={tokens} n={n} k={k} pat={pattern} bad_ids={bad_ids} sp={sp} null_stream={use_null}");
    let args = vec![
        Arg::In(u16s(&input)), Arg::In(u16s(&weights)), Arg::In(i32s(&sorted)), Arg::In(i32s(&eids)),
        if with_tw { Arg::In(f32s(&tw)) } else { Arg::Null }, Arg::Out("output", out),
    ];
    g.case(label, args, use_null, dtype <= 1, |rust, p, st| unsafe {
        let f = if rust { rl } else { cl };
        f(p[0] as P, p[1] as P, p[2] as _, p[3] as _, p[4] as _, p[5] as M, e as i32, topk as i32, size_m as i32, n as i32, k as i32, dtype, st)
    });
}

fn moe(g: &mut Gate) {
    g.family("moe_gemv");
    // (E, topk, tokens, n, k, pattern)
    let gemv_shapes: [(usize, usize, usize, usize, usize, u32); 8] = [
        (8, 2, 1, 64, 256, 0),
        (4, 1, 3, 50, 2056, 1),
        (16, 4, 2, 33, 64, 2),
        (1, 1, 5, 17, 8, 3),
        (32, 8, 1, 40, 128, 0),
        (6, 2, 4, 7, 5, 0), // k < 8: remainder loop only
        (3, 3, 2, 96, 520, 1),
        (8, 2, 8, 12, 4096, 0),
    ];
    for dtype in [0, 1] {
        for with_tw in [false, true] {
            for (si, &(e, tk, t, n, k, pat)) in gemv_shapes.iter().enumerate() {
                let sp = if si % 2 == 0 { 0 } else { 23 };
                moe_case(g, "moe_gemv", c::moe_gemv, launch::moe_gemv, dtype, with_tw, e, tk, t, n, k, pat, si % 3 == 1, sp, si == 2);
            }
        }
    }
    moe_case(g, "moe_gemv", c::moe_gemv, launch::moe_gemv, 2, true, 8, 2, 1, 64, 256, 0, false, 0, false);

    g.family("moe_gemm");
    // K % 64 == 0 (else the reference throws); the last N tile must hold >= 8 columns (else the
    // reference reads unloaded shared memory). K > 64 races in the reference (no barrier between
    // a tile's compute and the next tile's loads): see moe_case (K-periodic rows).
    let gemm_shapes: [(usize, usize, usize, usize, usize, u32); 7] = [
        (8, 2, 1, 64, 64, 0),
        (4, 1, 3, 72, 64, 1),
        (16, 4, 2, 128, 64, 2),
        (1, 1, 5, 100, 64, 3),
        (32, 8, 1, 200, 128, 0),
        (6, 2, 4, 64, 256, 0),
        (3, 3, 2, 136, 64, 1),
    ];
    for dtype in [0, 1] {
        for with_tw in [false, true] {
            for (si, &(e, tk, t, n, k, pat)) in gemm_shapes.iter().enumerate() {
                let sp = if si % 2 == 0 { 0 } else { 23 };
                moe_case(g, "moe_gemm", c::moe_gemm, launch::moe_gemm, dtype, with_tw, e, tk, t, n, k, pat, si % 3 == 1, sp, si == 2);
            }
        }
    }
    moe_case(g, "moe_gemm", c::moe_gemm, launch::moe_gemm, 5, false, 8, 2, 1, 64, 64, 0, false, 0, false);

    g.family("moe_gemm_wmma");
    let wmma_shapes: [(usize, usize, usize, usize, usize, u32); 8] = [
        (8, 2, 5, 64, 64, 0),
        (4, 1, 37, 50, 40, 1),
        (16, 4, 9, 96, 128, 2),
        (1, 1, 70, 33, 16, 3),
        (64, 8, 12, 32, 256, 0),
        (3, 2, 100, 128, 72, 1),
        (40, 3, 2, 17, 8, 0),
        (6, 2, 1, 200, 48, 0),
    ];
    for dtype in [0, 1] {
        for with_tw in [false, true] {
            for (si, &(e, tk, t, n, k, pat)) in wmma_shapes.iter().enumerate() {
                let sp = if si % 2 == 0 { 0 } else { 23 };
                moe_case(g, "moe_gemm_wmma", c::moe_gemm_wmma, launch::moe_gemm_wmma, dtype, with_tw, e, tk, t, n, k, pat, false, sp, si == 3);
            }
        }
    }
    // > 1024 experts (multi-chunk scan); an unsupported dtype (offsets only, no GEMM)
    moe_case(g, "moe_gemm_wmma", c::moe_gemm_wmma, launch::moe_gemm_wmma, 0, false, 1500, 4, 300, 32, 16, 0, false, 0, false);
    moe_case(g, "moe_gemm_wmma", c::moe_gemm_wmma, launch::moe_gemm_wmma, 1, true, 2048, 2, 700, 40, 32, 1, false, 0, false);
    moe_case(g, "moe_gemm_wmma", c::moe_gemm_wmma, launch::moe_gemm_wmma, 2, false, 8, 2, 5, 64, 64, 0, false, 0, false);
}

/// Element values of `dtype` (0 f16, 1 bf16, 2 f32) as bytes.
fn tvals(g: &mut Gate, dtype: i32, n: usize, lo: f32, hi: f32, sp: u64) -> Vec<u8> {
    if dtype == 2 { f32s(&g.f32vec(n, lo, hi, sp)) } else { u16s(&g.h16vec(n, dtype.min(1), lo, hi, sp)) }
}

/// A strided [b, h, s, d] tensor: strides from a dimension order (0: b h s d, 1: b s h d,
/// 2: d-interleaved b s h (d*2), 3: s b h d with padding). Returns (element count, strides).
fn ap_layout(layout: u32, b: usize, h: usize, s: usize, d: usize) -> (usize, [i64; 4]) {
    let (b, h, s, d) = (b as i64, h as i64, s as i64, d as i64);
    let st = match layout {
        0 => [h * s * d, s * d, d, 1],
        1 => [s * h * d, d, h * d, 1],
        2 => [s * h * d * 2, d * 2, h * d * 2, 2],
        _ => [h * (d + 3), (d + 3), b * h * (d + 3), 1],
    };
    let max = (b - 1) * st[0] + (h - 1) * st[1] + (s - 1) * st[2] + (d - 1) * st[3] + 1;
    (max as usize, st)
}

fn attention_prep(g: &mut Gate) {
    g.family("attention_prep");
    // (batch, q_heads, k_heads, seq, head_dim, rot_dim)
    let shapes: [(usize, usize, usize, usize, usize, usize); 8] = [
        (1, 2, 1, 3, 64, 32),
        (2, 4, 2, 2, 128, 32),
        (1, 3, 3, 1, 80, 40),
        (2, 1, 1, 5, 2, 1),
        (1, 2, 1, 2, 1100, 64),
        (1, 1, 0, 4, 64, 16),
        (3, 2, 2, 1, 256, 128),
        (1, 5, 1, 7, 96, 1),
    ];
    let mut ci = 0usize;
    for dtype in [0, 1, 2] {
        for neox in [0, 1] {
            for &(ba, qh, kh, sl, hd, rd) in &shapes {
                ci += 1;
                let es = if dtype == 2 { 4 } else { 2 };
                let sp = if ci % 2 == 0 { 0 } else { 29 };
                let (ql, qs) = ap_layout((ci % 4) as u32, ba, qh, sl, hd);
                let (kl, ks) = ap_layout(((ci + 1) % 4) as u32, ba, kh.max(1), sl, hd);
                let (vl, vs) = ap_layout(((ci + 2) % 4) as u32, ba, kh.max(1), sl, hd);
                let q = tvals(g, dtype, ql, -3.0, 3.0, sp);
                let k = tvals(g, dtype, kl, -3.0, 3.0, sp);
                let v = tvals(g, dtype, vl, -3.0, 3.0, sp);
                let qw = tvals(g, dtype, hd, -2.0, 2.0, sp);
                let kw = tvals(g, dtype, hd, -2.0, 2.0, sp);
                let vw = tvals(g, dtype, hd, -2.0, 2.0, sp);
                let max_pos = 37usize;
                let cos = tvals(g, dtype, max_pos * rd, -1.0, 1.0, 0);
                let sin = tvals(g, dtype, max_pos * rd, -1.0, 1.0, 0);
                let positions: Vec<u32> = (0..ba * sl).map(|_| (g.rng.next() % max_pos as u64) as u32).collect();
                let cbs = if ci % 3 == 0 { 0 } else { sl + (ci % 2) };
                let q_eps = g.pick(&[1e-6f32, 1e-5, 0.0, 0.25]);
                let k_eps = g.pick(&[1e-6f32, 1e-5, 0.0, 0.25]);
                let v_eps = g.pick(&[1e-6f32, 1e-5, 0.0, 0.25]);
                let qo = g.bytes(ba * qh * sl * hd * es);
                let ko = g.bytes((ba * kh * sl * hd).max(1) * es);
                let vo = g.bytes((ba * kh * sl * hd).max(1) * es);
                let use_null = ci % 5 == 0;
                let (b_, qh_, kh_, sl_, hd_, rd_) = (ba as i32, qh as i32, kh as i32, sl as i32, hd as i32, rd as i32);
                let label = format!("qk_rms_norm_rope dtype={dtype} neox={neox} B={ba} qh={qh} kh={kh} S={sl} hd={hd} rot={rd} cbs={cbs} sp={sp}");
                let args = vec![Arg::In(q.clone()), Arg::In(k.clone()), Arg::In(qw.clone()), Arg::In(kw.clone()), Arg::In(cos.clone()), Arg::In(sin.clone()),
                                Arg::Out("q_out", qo.clone()), Arg::Out("k_out", ko.clone())];
                g.case(label, args, use_null, true, |rust, p, st| unsafe {
                    let f = if rust { launch::qk_rms_norm_rope } else { c::qk_rms_norm_rope };
                    f(p[0] as P, p[1] as P, p[2] as P, p[3] as P, p[4] as P, p[5] as P, p[6] as M, p[7] as M, qs[0], qs[1], qs[2], qs[3], ks[0], ks[1], ks[2], ks[3],
                      b_, qh_, kh_, sl_, hd_, rd_, cbs as i32, q_eps, k_eps, neox, dtype, st)
                });
                let label = format!("qk_rms_norm_rope_positions dtype={dtype} neox={neox} B={ba} qh={qh} kh={kh} S={sl} hd={hd} rot={rd} sp={sp}");
                let args = vec![Arg::In(q.clone()), Arg::In(k.clone()), Arg::In(qw.clone()), Arg::In(kw.clone()), Arg::In(cos.clone()), Arg::In(sin.clone()),
                                Arg::In(u32s(&positions)), Arg::Out("q_out", qo.clone()), Arg::Out("k_out", ko.clone())];
                g.case(label, args, use_null, true, |rust, p, st| unsafe {
                    let f = if rust { launch::qk_rms_norm_rope_positions } else { c::qk_rms_norm_rope_positions };
                    f(p[0] as P, p[1] as P, p[2] as P, p[3] as P, p[4] as P, p[5] as P, p[6] as P, p[7] as M, p[8] as M, qs[0], qs[1], qs[2], qs[3], ks[0], ks[1], ks[2], ks[3],
                      b_, qh_, kh_, sl_, hd_, rd_, q_eps, k_eps, neox, dtype, st)
                });
                let label = format!("qkv_rms_norm_rope_positions dtype={dtype} neox={neox} B={ba} qh={qh} kh={kh} S={sl} hd={hd} rot={rd} sp={sp}");
                let args = vec![Arg::In(q), Arg::In(k), Arg::In(v), Arg::In(qw), Arg::In(kw), Arg::In(vw), Arg::In(cos), Arg::In(sin), Arg::In(u32s(&positions)),
                                Arg::Out("q_out", qo), Arg::Out("k_out", ko), Arg::Out("v_out", vo)];
                g.case(label, args, use_null, kh > 0, |rust, p, st| unsafe {
                    let f = if rust { launch::qkv_rms_norm_rope_positions } else { c::qkv_rms_norm_rope_positions };
                    f(p[0] as P, p[1] as P, p[2] as P, p[3] as P, p[4] as P, p[5] as P, p[6] as P, p[7] as P, p[8] as P, p[9] as M, p[10] as M, p[11] as M,
                      qs[0], qs[1], qs[2], qs[3], ks[0], ks[1], ks[2], ks[3], vs[0], vs[1], vs[2], vs[3], b_, qh_, kh_, sl_, hd_, rd_, q_eps, k_eps, v_eps, neox, dtype, st)
                });
            }
        }
    }
    // unsupported dtype and degenerate sizes: nothing launches
    let q = g.bytes(64 * 4);
    let out = g.bytes(64 * 4);
    for (dt, rd) in [(3, 8), (0, 0)] {
        let label = format!("qk_rms_norm_rope no-op dtype={dt} rot={rd}");
        g.case(label, vec![Arg::In(q.clone()), Arg::Out("q_out", out.clone())], false, false, |rust, p, st| unsafe {
            let f = if rust { launch::qk_rms_norm_rope } else { c::qk_rms_norm_rope };
            f(p[0] as P, p[0] as P, p[0] as P, p[0] as P, p[0] as P, p[0] as P, p[1] as M, p[1] as M, 16, 16, 16, 1, 16, 16, 16, 1, 1, 1, 0, 1, 16, rd, 0, 1e-6, 1e-6, 1, dt, st)
        });
    }
}

fn sort_misc(g: &mut Gate) {
    g.family("sort sampling");
    for &(n, nt, sp) in &[(1usize, 1usize, 0u64), (1000, 300, 0), (257, 257, 7), (5000, 1000, 13), (64, 0, 0), (0, 5, 0)] {
        let x = g.f32vec(n.max(1), -10.0, 10.0, sp);
        let dst = g.bytes(n.max(1) * 4);
        // unique token ids (duplicates race in the reference), some out of range
        let mut ids: Vec<u32> = (0..n as u32 + 8).collect();
        for i in (1..ids.len()).rev() {
            let j = (g.rng.next() % (i as u64 + 1)) as usize;
            ids.swap(i, j);
        }
        ids.truncate(nt);
        let counts: Vec<f32> = (0..nt).map(|i| match i % 9 { 0 => 0.0, 1 => -1.0, 2 => f32::from_bits(1), 3 => f32::NAN, _ => (1 + g.rng.next() % 5) as f32 }).collect();
        let biases = g.f32vec(nt, -3.0, 3.0, sp);
        for &(f, pr, r) in &[(0.5f32, 0.25f32, 1.0f32), (0.1, 0.0, 1.3), (1.0, 2.0, 0.7), (0.0, 0.0, f32::NAN)] {
            let label = format!("apply_sparse_penalties_f32 n={n} n_tokens={nt} f={f} p={pr} r={r}");
            let args = vec![Arg::In(f32s(&x)), Arg::Out("dst", dst.clone()), Arg::In(u32s(&ids)), Arg::In(f32s(&counts))];
            g.case(label, args, n == 257, n > 0, |rust, p, st| unsafe {
                let fun = if rust { launch::apply_sparse_penalties_f32 } else { c::apply_sparse_penalties_f32 };
                fun(p[0] as P, p[1] as M, p[2] as _, p[3] as _, n as i32, nt as i32, f, pr, r, st)
            });
        }
        let label = format!("apply_sparse_logits_bias_f32 n={n} n_tokens={nt}");
        let args = vec![Arg::In(f32s(&x)), Arg::Out("dst", dst.clone()), Arg::In(u32s(&ids)), Arg::In(f32s(&biases))];
        g.case(label, args, false, n > 0, |rust, p, st| unsafe {
            let fun = if rust { launch::apply_sparse_logits_bias_f32 } else { c::apply_sparse_logits_bias_f32 };
            fun(p[0] as P, p[1] as M, p[2] as _, p[3] as _, n as i32, nt as i32, st)
        });
    }
    for &(bh, ql, kl, qo, pl) in &[(1usize, 1usize, 1usize, 0i32, 0i32), (3, 5, 9, 2, 1), (2, 7, 7, 0, 0), (1, 4, 300, 100, 50), (2, 3, 5, -4, 0), (0, 3, 3, 0, 0)] {
        let sc = g.f32vec((bh * ql * kl).max(1), -5.0, 5.0, 0);
        let label = format!("apply_causal_mask_f32 bh={bh} q={ql} kv={kl} off={qo} prefix={pl}");
        g.case(label, vec![Arg::Out("scores", f32s(&sc))], bh == 2, false, |rust, p, st| unsafe {
            let fun = if rust { launch::apply_causal_mask_f32 } else { c::apply_causal_mask_f32 };
            fun(p[0] as M, bh as i32, ql as i32, kl as i32, qo, pl, st)
        });
    }
}

type RmsL = unsafe extern "C" fn(P, P, P, P, M, i32, i32, f32, i64);
type RmsThenL = unsafe extern "C" fn(P, P, P, P, P, M, M, i32, i32, f32, f32, i64);
type RmsStrL = unsafe extern "C" fn(P, P, M, i64, i64, i64, i64, i32, i32, i32, i32, f32, i64);

fn sort_rms(g: &mut Gate) {
    g.family("sort rms_norm");
    let res: [(RmsL, RmsL); 3] = [
        (c::rms_norm_residual_f32, launch::rms_norm_residual_f32),
        (c::rms_norm_residual_f16, launch::rms_norm_residual_f16),
        (c::rms_norm_residual_bf16, launch::rms_norm_residual_bf16),
    ];
    let then: [(RmsThenL, RmsThenL); 3] = [
        (c::rms_norm_residual_then_rms_norm_f32, launch::rms_norm_residual_then_rms_norm_f32),
        (c::rms_norm_residual_then_rms_norm_f16, launch::rms_norm_residual_then_rms_norm_f16),
        (c::rms_norm_residual_then_rms_norm_bf16, launch::rms_norm_residual_then_rms_norm_bf16),
    ];
    let strided: [(RmsStrL, RmsStrL); 3] = [
        (c::rms_norm_strided_4d_f32, launch::rms_norm_strided_4d_f32),
        (c::rms_norm_strided_4d_f16, launch::rms_norm_strided_4d_f16),
        (c::rms_norm_strided_4d_bf16, launch::rms_norm_strided_4d_bf16),
    ];
    let mut ci = 0;
    for t in 0..3usize {
        let dt = [2, 0, 1][t]; // tvals dtype code
        let es = if t == 0 { 4 } else { 2 };
        for &(nrows, ncols) in &[(1usize, 8usize), (3, 64), (2, 1000), (1, 1024), (2, 2048), (4, 13), (1, 8200), (2, 256)] {
            for &(with_scale, shift) in &[(false, 0usize), (true, 0), (true, 1)] {
                ci += 1;
                let sp = if ci % 2 == 0 { 0 } else { 17 };
                // `shift` offsets every pointer by one element (misaligned: forces the scalar kernel)
                let sh = shift * es;
                let n = nrows * ncols + shift;
                let x = tvals(g, dt, n, -3.0, 3.0, sp);
                let r = tvals(g, dt, n, -3.0, 3.0, sp);
                let w = tvals(g, dt, ncols + shift, -2.0, 2.0, sp);
                let nw = tvals(g, dt, ncols + shift, -2.0, 2.0, sp);
                let scale = tvals(g, dt, 1, 0.25, 2.0, 0);
                let d = g.bytes(n * es);
                let d2 = g.bytes(n * es);
                let eps = g.pick(&[1e-6f32, 1e-5, 0.0]);
                let neps = g.pick(&[1e-6f32, 1e-5, 0.3]);
                let (cl, rl) = res[t];
                let label = format!("rms_norm_residual t={t} rows={nrows} cols={ncols} scale={with_scale} shift={shift} sp={sp}");
                let args = vec![Arg::In(x.clone()), Arg::In(r.clone()), Arg::In(w.clone()), if with_scale { Arg::In(scale.clone()) } else { Arg::Null }, Arg::Out("dst", d.clone())];
                g.case(label, args, ci % 5 == 0, true, |rust, p, st| unsafe {
                    let f = if rust { rl } else { cl };
                    let a = |i: usize| if p[i] == 0 { 0 } else { p[i] + sh as u64 };
                    f(a(0) as P, a(1) as P, a(2) as P, p[3] as P, a(4) as M, nrows as i32, ncols as i32, eps, st)
                });
                let (cl, rl) = then[t];
                let label = format!("rms_norm_residual_then_rms_norm t={t} rows={nrows} cols={ncols} scale={with_scale} shift={shift} sp={sp}");
                let args = vec![Arg::In(x.clone()), Arg::In(r.clone()), Arg::In(w.clone()), if with_scale { Arg::In(scale.clone()) } else { Arg::Null },
                                Arg::In(nw.clone()), Arg::Out("residual_dst", d.clone()), Arg::Out("norm_dst", d2.clone())];
                g.case(label, args, ci % 5 == 1, true, |rust, p, st| unsafe {
                    let f = if rust { rl } else { cl };
                    let a = |i: usize| if p[i] == 0 { 0 } else { p[i] + sh as u64 };
                    f(a(0) as P, a(1) as P, a(2) as P, p[3] as P, a(4) as P, a(5) as M, a(6) as M, nrows as i32, ncols as i32, eps, neps, st)
                });
            }
        }
        for &(b, h, sl, hd) in &[(1usize, 2usize, 3usize, 64usize), (2, 3, 2, 128), (1, 1, 5, 80), (2, 2, 2, 1100), (1, 4, 1, 1), (3, 1, 2, 256)] {
            ci += 1;
            let (len, strides) = ap_layout((ci % 4) as u32, b, h, sl, hd);
            let x = tvals(g, dt, len, -3.0, 3.0, 23);
            let w = tvals(g, dt, hd, -2.0, 2.0, 0);
            let d = g.bytes(b * h * sl * hd * es);
            let eps = g.pick(&[1e-6f32, 0.0, 0.1]);
            let (cl, rl) = strided[t];
            let label = format!("rms_norm_strided_4d t={t} b={b} h={h} s={sl} hd={hd} strides={strides:?}");
            g.case(label, vec![Arg::In(x), Arg::In(w), Arg::Out("dst", d)], ci % 3 == 0, true, |rust, p, st| unsafe {
                let f = if rust { rl } else { cl };
                f(p[0] as P, p[1] as P, p[2] as M, strides[0], strides[1], strides[2], strides[3], b as i32, h as i32, sl as i32, hd as i32, eps, st)
            });
        }
    }
}

type SortL = unsafe extern "C" fn(M, M, i32, i32, bool, i64);

fn sort_asort(g: &mut Gate) {
    g.family("sort asort");
    let ops: [(&str, usize, SortL, SortL); 14] = [
        ("asort_asc_f32", 0, c::asort_asc_f32, launch::asort_asc_f32),
        ("asort_desc_f32", 0, c::asort_desc_f32, launch::asort_desc_f32),
        ("asort_asc_f16", 1, c::asort_asc_f16, launch::asort_asc_f16),
        ("asort_desc_f16", 1, c::asort_desc_f16, launch::asort_desc_f16),
        ("asort_asc_bf16", 2, c::asort_asc_bf16, launch::asort_asc_bf16),
        ("asort_desc_bf16", 2, c::asort_desc_bf16, launch::asort_desc_bf16),
        ("asort_asc_f64", 3, c::asort_asc_f64, launch::asort_asc_f64),
        ("asort_desc_f64", 3, c::asort_desc_f64, launch::asort_desc_f64),
        ("asort_asc_u8", 4, c::asort_asc_u8, launch::asort_asc_u8),
        ("asort_desc_u8", 4, c::asort_desc_u8, launch::asort_desc_u8),
        ("asort_asc_u32", 5, c::asort_asc_u32, launch::asort_asc_u32),
        ("asort_desc_u32", 5, c::asort_desc_u32, launch::asort_desc_u32),
        ("asort_asc_i64", 6, c::asort_asc_i64, launch::asort_asc_i64),
        ("asort_desc_i64", 6, c::asort_desc_i64, launch::asort_desc_i64),
    ];
    for (oi, &(name, t, cl, rl)) in ops.iter().enumerate() {
        let es = [4usize, 2, 2, 8, 1, 4, 8][t];
        for &(nrows, ncols) in &[(1usize, 1usize), (2, 2), (3, 5), (1, 16), (2, 100), (1, 1024), (1, 1500), (4, 64), (2, 4096)] {
            // f16/bf16 pad with uninitialised host memory (numeric_limits<__half> is not
            // specialised): only power-of-two rows (no padding) are well defined.
            if (t == 1 || t == 2) && !ncols.is_power_of_two() {
                continue;
            }
            for inplace in [false, true] {
                let n = nrows * ncols;
                let x: Vec<u8> = match t {
                    0 => f32s(&(0..n).map(|_| if g.rng.next() % 3 == 0 { (g.rng.next() % 7) as f32 - 3.0 } else { g.f32v(-100.0, 100.0, 11) }).collect::<Vec<_>>()),
                    1 | 2 => u16s(&(0..n).map(|_| g.h16((t == 2) as i32, -50.0, 50.0, 9)).collect::<Vec<_>>()),
                    3 => (0..n).flat_map(|i| (if i % 13 == 0 { f64::from_bits(g.rng.next()) } else if i % 7 == 0 { f64::NAN } else { (g.rng.next() % 1000) as f64 - 500.0 }).to_le_bytes()).collect(),
                    4 => (0..n).map(|_| (g.rng.next() % 40) as u8 * if g.rng.next() % 5 == 0 { 6 } else { 1 }).collect(),
                    5 => u32s(&(0..n).map(|_| if g.rng.next() % 4 == 0 { u32::MAX - (g.rng.next() % 3) as u32 } else { (g.rng.next() % 5000) as u32 }).collect::<Vec<_>>()),
                    _ => (0..n).flat_map(|_| (if g.rng.next() % 4 == 0 { g.rng.next() as i64 } else { (g.rng.next() % 1000) as i64 - 500 }).to_le_bytes()).collect(),
                };
                let dst = g.bytes(n * 4);
                let _ = es;
                let label = format!("{name} rows={nrows} cols={ncols} inplace={inplace}");
                g.case(label, vec![Arg::Out("x", x), Arg::Out("dst", dst)], (oi + ncols) % 4 == 0, n > 1, |rust, p, st| unsafe {
                    let f = if rust { rl } else { cl };
                    f(p[0] as M, p[1] as M, nrows as i32, ncols as i32, inplace, st)
                });
            }
        }
    }
}

type TopkL = unsafe extern "C" fn(P, M, M, i32, i32, i32, i64);
type RouterL = unsafe extern "C" fn(P, M, M, P, P, i32, i32, i32, i32, i32, bool, bool, f32, f32, f32, f32, i64);

fn sort_topk(g: &mut Gate) {
    g.family("sort topk");
    let tk: [(&str, i32, TopkL, TopkL); 3] = [
        ("topk_f32", 2, c::topk_f32, launch::topk_f32),
        ("topk_f16", 0, c::topk_f16, launch::topk_f16),
        ("topk_bf16", 1, c::topk_bf16, launch::topk_bf16),
    ];
    for &(name, dt, cl, rl) in &tk {
        let es = if dt == 2 { 4 } else { 2 };
        for &(nrows, ncols, k) in &[(1usize, 5usize, 2usize), (3, 64, 8), (2, 100, 1), (1, 256, 16), (2, 300, 4), (1, 1000, 10), (2, 7, 9), (4, 128, 128)] {
            let x = tvals(g, dt, nrows * ncols, -5.0, 5.0, 5);
            let (v, i) = (g.bytes(nrows * k * es), g.bytes(nrows * k * 4));
            let label = format!("{name} rows={nrows} cols={ncols} k={k}");
            g.case(label, vec![Arg::In(x), Arg::Out("values", v), Arg::Out("indices", i)], ncols == 100, true, |rust, p, st| unsafe {
                let f = if rust { rl } else { cl };
                f(p[0] as P, p[1] as M, p[2] as M, nrows as i32, ncols as i32, k as i32, st)
            });
        }
    }

    g.family("sort moe_router_topk");
    let rt: [(&str, i32, RouterL, RouterL); 3] = [
        ("moe_router_topk_f32", 2, c::moe_router_topk_f32, launch::moe_router_topk_f32),
        ("moe_router_topk_f16", 0, c::moe_router_topk_f16, launch::moe_router_topk_f16),
        ("moe_router_topk_bf16", 1, c::moe_router_topk_bf16, launch::moe_router_topk_bf16),
    ];
    let mut ci = 0usize;
    for &(name, dt, cl, rl) in &rt {
        for &ne in &[1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 576, 3] {
            for hb in [false, true] {
                for hs in [false, true] {
                    for rep in 0..2 {
                        ci += 1;
                        let nrows = [1usize, 5, 4, 9][ci % 4];
                        let vpt = if ne > 32 { ne / 32 } else { 1 };
                        let top_k = match (ci + rep) % 4 {
                            0 => 1,
                            1 => ne.min(8),
                            2 => ne.min(2),
                            _ => if ne > 32 { (ne / 2).min(32 * vpt).min(40) } else { ne },
                        };
                        let sm = (g.rng.next() % 3) as i32;
                        let wm = (g.rng.next() % 3) as i32;
                        let renorm = g.rng.next() % 2 == 0;
                        let clamp = g.rng.next() % 3 == 0;
                        let (cmin, cmax) = g.pick(&[(-3.0f32, 3.0f32), (-1.0, 0.5), (0.0, 10.0)]);
                        let nmin = g.pick(&[1e-20f32, 0.5, 0.0]);
                        let os = g.pick(&[1.0f32, 2.5, 0.125]);
                        let sp = if rep == 0 { 0 } else { 11 };
                        let logits = tvals(g, dt, nrows * ne, -6.0, 6.0, sp);
                        let bias = g.f32vec(ne, -0.5, 0.5, 0);
                        let escale = g.f32vec(ne, 0.5, 2.0, 0);
                        let w = g.bytes(nrows * top_k * 4);
                        let ids = g.bytes(nrows * top_k * 4);
                        let label = format!("{name} ne={ne} bias={hb} scale={hs} rows={nrows} top_k={top_k} score={sm} weight={wm} renorm={renorm} clamp={clamp} sp={sp}");
                        let args = vec![Arg::In(logits), Arg::Out("weights", w), Arg::Out("ids", ids), if hb { Arg::In(f32s(&bias)) } else { Arg::Null },
                                        if hs { Arg::In(f32s(&escale)) } else { Arg::Null }];
                        g.case(label, args, ci % 7 == 0, ne != 3, |rust, p, st| unsafe {
                            let f = if rust { rl } else { cl };
                            f(p[0] as P, p[1] as M, p[2] as M, p[3] as P, p[4] as P, nrows as i32, ne as i32, top_k as i32, sm, wm, renorm, clamp, cmin, cmax, nmin, os, st)
                        });
                    }
                }
            }
        }
    }

    // Tie-heavy logits (a handful of distinct values): the butterfly's `other_expert < best_expert`
    // tie-break and the selection masking decide the ids here; random data almost never ties.
    for &(name, dt, cl, rl) in &rt {
        for &ne in &[8usize, 64, 256, 576] {
            for (ti, &top_k) in [1usize, 8, 5].iter().enumerate() {
                for sm in 0..3 {
                    let nrows = [1usize, 3, 4][ti];
                    let vals = [-1.0f32, 0.0, -0.0, 0.5, 2.0];
                    let lv: Vec<f32> = (0..nrows * ne).map(|_| g.pick(&vals)).collect();
                    let logits = match dt {
                        2 => f32s(&lv),
                        0 => u16s(&lv.iter().map(|&v| f32_to_f16(v)).collect::<Vec<_>>()),
                        _ => u16s(&lv.iter().map(|&v| f32_to_bf16(v)).collect::<Vec<_>>()),
                    };
                    let wm = (g.rng.next() % 3) as i32;
                    let renorm = g.rng.next() % 2 == 0;
                    let w = g.bytes(nrows * top_k * 4);
                    let ids = g.bytes(nrows * top_k * 4);
                    let label = format!("{name} ties ne={ne} rows={nrows} top_k={top_k} score={sm} weight={wm} renorm={renorm}");
                    let args = vec![Arg::In(logits), Arg::Out("weights", w), Arg::Out("ids", ids), Arg::Null, Arg::Null];
                    g.case(label, args, false, true, |rust, p, st| unsafe {
                        let f = if rust { rl } else { cl };
                        f(p[0] as P, p[1] as M, p[2] as M, p[3] as P, p[4] as P, nrows as i32, ne as i32, top_k as i32, sm, wm, renorm, false, 0.0, 0.0, 0.0, 1.0, st)
                    });
                }
            }
        }
    }

    g.family("sort topk_large");
    for &(ncols, chunk, extra, k, it) in &[(5000usize, 1024usize, 0usize, 5usize, 1.0f32), (1000, 256, 1, 1, 0.7), (70000, 4096, 0, 20, 1.3), (300, 512, 2, 3, 1.0), (4096, 1024, 0, 64, 0.5)] {
        let nblocks = ncols.div_ceil(chunk) + extra;
        let x = g.f32vec(ncols, -20.0, 20.0, 97);
        let bv = g.bytes(nblocks * k * 4);
        let bi = g.bytes(nblocks * k * 4);
        let bm = g.bytes(nblocks * 4);
        let bsm = g.bytes(nblocks * 4);
        let vo = g.bytes(k * 4);
        let io = g.bytes(k * 4);
        let so = g.bytes(8);
        let po = g.bytes((2 * k + 2) * 4);
        let label = format!("topk_large_f32 ncols={ncols} chunk={chunk} nblocks={nblocks} k={k} inv_t={it}");
        let args = vec![Arg::In(f32s(&x)), Arg::Out("block_values", bv.clone()), Arg::Out("block_indices", bi.clone()), Arg::Out("block_maxes", bm.clone()),
                        Arg::Out("block_sums", bsm.clone()), Arg::Out("values", vo), Arg::Out("indices", io), Arg::Out("softmax_info", so)];
        g.case(label, args, k == 1, true, |rust, p, st| unsafe {
            let f = if rust { launch::topk_large_f32 } else { c::topk_large_f32 };
            f(p[0] as _, p[1] as _, p[2] as _, p[3] as _, p[4] as _, p[5] as _, p[6] as _, p[7] as _, ncols as i32, k as i32, chunk as i32, nblocks as i32, it, st)
        });
        let label = format!("topk_large_f32_packed ncols={ncols} chunk={chunk} nblocks={nblocks} k={k} inv_t={it}");
        let args = vec![Arg::In(f32s(&x)), Arg::Out("block_values", bv.clone()), Arg::Out("block_indices", bi.clone()), Arg::Out("block_maxes", bm),
                        Arg::Out("block_sums", bsm), Arg::Out("packed", po)];
        g.case(label, args, false, true, |rust, p, st| unsafe {
            let f = if rust { launch::topk_large_f32_packed } else { c::topk_large_f32_packed };
            f(p[0] as _, p[1] as _, p[2] as _, p[3] as _, p[4] as _, p[5] as _, ncols as i32, k as i32, chunk as i32, nblocks as i32, it, st)
        });
        let po1 = g.bytes(8);
        let label = format!("top1_large_f32_packed ncols={ncols} chunk={chunk} nblocks={nblocks}");
        let args = vec![Arg::In(f32s(&x)), Arg::Out("block_values", bv), Arg::Out("block_indices", bi), Arg::Out("packed", po1)];
        g.case(label, args, extra > 0, true, |rust, p, st| unsafe {
            let f = if rust { launch::top1_large_f32_packed } else { c::top1_large_f32_packed };
            f(p[0] as _, p[1] as _, p[2] as _, p[3] as _, ncols as i32, chunk as i32, nblocks as i32, st)
        });
    }
}

// ------------------------------------------------------------------------------------------------
// GATE_BENCH=1: time the decode-hot launchers, C vs Rust, on decode shapes (cudaEvent timing).

unsafe extern "C" {
    fn cudaEventCreate(e: *mut *mut c_void) -> i32;
    fn cudaEventRecord(e: *mut c_void, s: *mut c_void) -> i32;
    fn cudaEventSynchronize(e: *mut c_void) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, a: *mut c_void, b: *mut c_void) -> i32;
}

fn bench_one(g: &Gate, name: &str, iters: usize, run: &dyn Fn(bool, i64)) {
    let st = g.stream_handle(false);
    let mut t = [0f64; 2];
    unsafe {
        let (mut e0, mut e1) = (std::ptr::null_mut(), std::ptr::null_mut());
        cudaEventCreate(&mut e0);
        cudaEventCreate(&mut e1);
        for rep in 0..6 {
            for side in [false, true] {
                for _ in 0..20 {
                    run(side, st);
                }
                cudaEventRecord(e0, st as usize as *mut c_void);
                for _ in 0..iters {
                    run(side, st);
                }
                cudaEventRecord(e1, st as usize as *mut c_void);
                cudaEventSynchronize(e1);
                let mut ms = 0f32;
                cudaEventElapsedTime(&mut ms, e0, e1);
                if rep > 0 {
                    t[side as usize] += ms as f64;
                }
            }
        }
    }
    let n = (5 * iters) as f64;
    let (c, r) = (t[0] * 1e3 / n, t[1] * 1e3 / n);
    println!("bench {name:<40} ref {c:8.3} us  oxide {r:8.3} us  ratio {:.3}", r / c);
}

fn bench(g: &mut Gate) {
    let iters: usize = std::env::var("GATE_BENCH_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(2000);
    // gdn_decode_recurrence: bf16, B=1, 16 key heads x 128, 32 value heads x 128.
    {
        let (bs, nk, nv, hk, hv) = (1usize, 16usize, 32usize, 128usize, 128usize);
        let conv = 2 * nk * hk + nv * hv;
        let m = { let v = u16s(&g.rng_h16vec(bs * conv, 1, -2.0, 2.0)); g.buf(&v) };
        let b = { let v = u16s(&g.rng_h16vec(bs * nv, 1, -6.0, 6.0)); g.buf(&v) };
        let a = { let v = u16s(&g.rng_h16vec(bs * nv, 1, -25.0, 25.0)); g.buf(&v) };
        let al = { let v = f32s(&g.f32vec(nv, -3.0, 1.0, 0)); g.buf(&v) };
        let db = { let v = f32s(&g.f32vec(nv, -2.0, 2.0, 0)); g.buf(&v) };
        let st0 = f32s(&g.f32vec(bs * nv * hk * hv, -1.0, 1.0, 0));
        let sts = [g.buf(&st0), g.buf(&st0)];
        let outs = [g.buf(&vec![0u8; bs * nv * hv * 4]), g.buf(&vec![0u8; bs * nv * hv * 4])];
        g.sync("bench upload");
        bench_one(g, "gdn_decode_recurrence bf16 128x128 nv=32", iters, &|rust, st| unsafe {
            let i = rust as usize;
            let f = if rust { launch::gdn_decode_recurrence } else { c::gdn_decode_recurrence };
            f(m.cu_deviceptr() as P, b.cu_deviceptr() as P, a.cu_deviceptr() as P, al.cu_deviceptr() as _, db.cu_deviceptr() as _,
              sts[i].cu_deviceptr() as _, outs[i].cu_deviceptr() as _, bs as i32, nk as i32, nv as i32, hk as i32, hv as i32, 1, st)
        });
    }
    // moe_router_topk_f32: 1 row, 256 experts, top 8, softmax score, renormalize.
    {
        let (ne, tk) = (256usize, 8usize);
        let l = { let v = f32s(&g.f32vec(ne, -6.0, 6.0, 0)); g.buf(&v) };
        let ws = [g.buf(&vec![0u8; tk * 4]), g.buf(&vec![0u8; tk * 4])];
        let is = [g.buf(&vec![0u8; tk * 4]), g.buf(&vec![0u8; tk * 4])];
        g.sync("bench upload");
        bench_one(g, "moe_router_topk_f32 ne=256 k=8", iters, &|rust, st| unsafe {
            let i = rust as usize;
            let f: RouterL = if rust { launch::moe_router_topk_f32 } else { c::moe_router_topk_f32 };
            f(l.cu_deviceptr() as P, ws[i].cu_deviceptr() as M, is[i].cu_deviceptr() as M, std::ptr::null(), std::ptr::null(), 1, ne as i32, tk as i32, 1, 0, true, false, 0.0, 0.0, 0.0, 1.0, st)
        });
    }
    // gated_delta_rule_recurrence (tiled 128x64): bh=32, k=v=128, seq 64 (prefill).
    {
        let (bh, seq, kd, vd) = (32usize, 64usize, 128usize, 128usize);
        let q = { let v = f32s(&g.f32vec(bh * seq * kd, -0.3, 0.3, 0)); g.buf(&v) };
        let k = { let v = f32s(&g.f32vec(bh * seq * kd, -0.3, 0.3, 0)); g.buf(&v) };
        let v = { let v = f32s(&g.f32vec(bh * seq * vd, -2.0, 2.0, 0)); g.buf(&v) };
        let gg = { let v = f32s(&g.f32vec(bh * seq, -1.5, 0.05, 0)); g.buf(&v) };
        let be = { let v = f32s(&g.f32vec(bh * seq, 0.0, 1.0, 0)); g.buf(&v) };
        let st0 = f32s(&g.f32vec(bh * kd * vd, -1.0, 1.0, 0));
        let sts = [g.buf(&st0), g.buf(&st0)];
        let outs = [g.buf(&vec![0u8; bh * seq * vd * 4]), g.buf(&vec![0u8; bh * seq * vd * 4])];
        g.sync("bench upload");
        bench_one(g, "gated_delta_rule_recurrence 128 seq=64 bh=32", (iters / 10).max(1), &|rust, st| unsafe {
            let i = rust as usize;
            let f = if rust { launch::gated_delta_rule_recurrence } else { c::gated_delta_rule_recurrence };
            f(q.cu_deviceptr() as _, k.cu_deviceptr() as _, v.cu_deviceptr() as _, gg.cu_deviceptr() as _, be.cu_deviceptr() as _,
              sts[i].cu_deviceptr() as _, outs[i].cu_deviceptr() as _, bh as i32, seq as i32, kd as i32, vd as i32, st)
        });
    }
}

pub fn run() -> bool {
    let ctx = CudaContext::new(0).expect("cuda context");
    ctx.bind_to_thread().unwrap();
    let stream = ctx.new_stream().expect("stream");
    let mut g = Gate { ctx, stream, rng: Rng(0x6d69_7374_7261_6c21), calls: 0, bytes: 0, written: 0, elems: 0, failures: Vec::new(), family: "", fam_stats: Vec::new() };
    if std::env::var("GATE_BENCH").is_ok() {
        bench(&mut g);
        return true;
    }
    let only = std::env::var("GATE_ONLY").unwrap_or_default();
    let want = |f: &str| only.is_empty() || only.split(',').any(|o| f.contains(o));
    // Several rounds over every family; the generator keeps running, so each round draws new data.
    let rounds: usize = std::env::var("GATE_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    for _ in 0..rounds {
        if want("gdn") {
            gdn_recurrence(&mut g);
            gdn_conv(&mut g);
            gdn_misc(&mut g);
            gdn_prepare_decode(&mut g);
        }
        if want("moe") {
            moe(&mut g);
        }
        if want("attention") {
            attention_prep(&mut g);
        }
        if want("sort") {
            sort_misc(&mut g);
            sort_rms(&mut g);
            sort_asort(&mut g);
            sort_topk(&mut g);
        }
        if want("ssm") {
            ssm(&mut g);
        }
        if want("graph") {
            graph(&mut g);
            graph094(&mut g);
        }
    }
    g.close_family();
    // Instance coverage: every reference kernel (except the cub/thrust scan kernels, replaced
    // by one integer scan) must have been launched through its oxide twin.
    if only.is_empty() {
        let home = std::env::var("HOME").unwrap();
        let launched = launch::launched_kernels();
        let mut total = 0;
        let mut missing = Vec::new();
        for m in ["attention_prep", "gdn", "graph", "moe_gemm", "moe_gemm_wmma", "moe_gemv", "sort", "ssm"] {
            let out = std::process::Command::new(format!("{home}/titan-engine/oxide-kernels/tools/cuobjdump"))
                .args(["-sass", &format!("{home}/titan-engine/oxide-kernels/reference/mistralrs-core/{m}.cubin")])
                .output()
                .expect("cuobjdump");
            for l in String::from_utf8_lossy(&out.stdout).lines().filter(|l| l.contains("Function :")) {
                let f = l.split_whitespace().nth(2).unwrap().to_string();
                if f.starts_with("_ZN3cub") {
                    continue;
                }
                total += 1;
                if !launched.contains(&f.as_str()) {
                    missing.push(f);
                }
            }
        }
        println!("instances: {} of {} reference kernels launched through their oxide twins (cub scan kernels excluded)", total - missing.len(), total);
        for f in &missing {
            g.failures.push(format!("reference kernel never exercised: {f}"));
        }
    }
    let mut fams: Vec<(&str, usize, usize, usize)> = Vec::new();
    for &(f, c, b, x) in &g.fam_stats {
        if let Some(e) = fams.iter_mut().find(|e| e.0 == f) {
            e.1 += c;
            e.2 += b;
            e.3 += x;
        } else {
            fams.push((f, c, b, x));
        }
    }
    for (f, calls, bytes, fails) in &fams {
        println!("  {f:<24} {calls:>6} calls {bytes:>12} bytes {fails:>4} failing");
    }
    for f in g.failures.iter().take(60) {
        println!("  FAIL {f}");
    }
    println!(
        "coverage: {} writable bytes, {} written by the reference ({:.1}%)",
        g.elems, g.written, 100.0 * g.written as f64 / g.elems.max(1) as f64
    );
    let ok = g.failures.is_empty() && g.calls > 0;
    println!("mistralrs-core-cuda: {} launcher calls, {} bytes compared, {} failing -> {}", g.calls, g.bytes, g.failures.len(), if ok { "PASS" } else { "FAIL" });
    ok
}
