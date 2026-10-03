//! Launcher-level differential gate for mistralrs-quant group A: every extern "C" launcher is
//! called twice on identical inputs in the same (primary) context and stream -- once the REAL C
//! launcher from libmistralrsquant.a (nvcc kernels, -O3 --use_fast_math), once the pure-Rust twin
//! in `crate::launch` (oxide kernels) -- and every byte of every buffer the call can see is
//! compared: outputs, inputs (no stray writes) and a random guard region after each buffer.
//! Kernel instances that no launcher reaches (host-pass stubs) are compared kernel-by-kernel
//! against the reference cubin with kdiff.
use crate::launch as ox;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use kdiff::{Arg, Harness, Rng, Tally, as_bytes};
use std::ffi::c_void;
use std::sync::Arc;

/// The C launchers (libmistralrsquant.a).
pub mod cref {
    use std::ffi::c_void;
    unsafe extern "C" {
        /// cudart: clears the runtime's last-error slot.
        pub fn cudaGetLastError() -> i32;
        pub fn launch_pack_1bit_kernel(i: *const u8, o: *mut u8, n: usize, w: usize, s: *mut c_void);
        pub fn launch_pack_2bit_kernel(i: *const u8, o: *mut u8, n: usize, w: usize, s: *mut c_void);
        pub fn launch_pack_3bit_kernel(i: *const u32, o: *mut i32, n: usize, w: usize, s: *mut c_void);
        pub fn launch_pack_4bit_kernel(i: *const u8, o: *mut u8, n: usize, w: usize, s: *mut c_void);
        pub fn launch_pack_8bit_kernel(i: *const u8, o: *mut u8, n: usize, s: *mut c_void);

        pub fn dequantize_8bit_u8_kernel_f32(wq: *const u8, s: *const f32, z: *const f32, o: *const f32, h: i32, w: i32);
        pub fn dequantize_4bit_u8_kernel_f32(wq: *const u8, s: *const f32, z: *const f32, o: *const f32, h: i32, w: i32);
        pub fn dequantize_2bit_u8_kernel_f32(wq: *const u8, s: *const f32, z: *const f32, o: *const f32, h: i32, w: i32);
        pub fn dequantize_1bit_u8_kernel_f32(wq: *const u8, s: *const f32, z: *const f32, o: *const f32, h: i32, w: i32);
        pub fn dequantize_3bit_32_kernel_f32(wq: *const i32, s: *const f32, z: *const f32, o: *const f32, h: i32, w: i32);

        pub fn launch_gelu_tanh_and_mul_bf16(o: *mut c_void, i: *const c_void, t: i32, d: i32, s: *mut c_void);
        pub fn launch_silu_and_mul_bf16(o: *mut c_void, i: *const c_void, t: i32, d: i32, s: *mut c_void);
        pub fn launch_gelu_tanh_and_mul_f16(o: *mut c_void, i: *const c_void, t: i32, d: i32, s: *mut c_void);
        pub fn launch_moe_sum_bf16(o: *mut c_void, i: *const c_void, t: i32, h: i32, k: i32, s: *mut c_void);

        pub fn rotary_embedding(q: *const c_void, k: *const c_void, c: *const c_void, s: *const c_void, neox: i32, hs: i32, nt: i64,
                                rd: i32, nh: i32, nkv: i32, qs: i64, ks: i64, dtype: u32, stream: i64);
        pub fn rotary_embedding_positions(q: *const c_void, k: *const c_void, c: *const c_void, s: *const c_void, p: *const c_void,
                                          neox: i32, hs: i32, nt: i64, rd: i32, sl: i32, nh: i32, nkv: i32, qs: i64, ks: i64,
                                          dtype: u32, stream: i64);

        pub fn launch_moe_align(ti: *const i32, st: *mut i32, ei: *mut i32, nt: *mut i32, cs: *mut i32, ne: i32, bs: i32, numel: i32,
                                mx: i32, s: *mut c_void);
        pub fn launch_hunyuan_moe_capacity_mask(ids: *const c_void, w: *const c_void, m: *mut c_void, nt: i32, ne: i32, tk: i32, cap: i32,
                                                s: *mut c_void);

        pub fn launch_cutlass_moe_problem_sizes(t: *const i32, p1: *mut i32, p2: *mut i32, ab: *mut i32, ne: i32, tl: i32, n: i32, k: i32,
                                                g: bool, s: *mut c_void);
        pub fn launch_cutlass_moe_expert_offsets(p1: *const i32, eo: *mut i32, ab: *mut i32, ne: i32, s: *mut c_void);
        pub fn launch_cutlass_moe_arg_sorts(t: *const i32, ip: *mut i32, op: *mut i32, ab: *mut i32, ne: i32, tl: i32, tk: i32, s: *mut c_void);
        pub fn launch_cutlass_moe_gather_rows_bf16(d: *mut c_void, s: *const c_void, m: *const i32, nr: i32, k: i32, st: *mut c_void);
        pub fn launch_cutlass_moe_gather_weighted_bf16(o: *mut c_void, i: *const c_void, p: *const i32, w: *const f32, nr: i32, k: i32,
                                                       s: *mut c_void);
        pub fn launch_cutlass_moe_group_starts_bf16(eo: *const i32, a: *const c_void, b: *const c_void, d: *mut c_void,
                                                    ap: *const *const c_void, bp: *const *const c_void, dp: *mut *mut c_void,
                                                    la: *mut i64, lb: *mut i64, ld: *mut i64, ne: i32, n: i64, k: i64, s: *mut c_void);

        pub fn bitwise_and_u8(a: *const c_void, b: *const c_void, o: *mut c_void, n: u32);
        pub fn bitwise_or_u8(a: *const c_void, b: *const c_void, o: *mut c_void, n: u32);
        pub fn bitwise_xor_u8(a: *const c_void, b: *const c_void, o: *mut c_void, n: u32);
        pub fn bitwise_and_u32(a: *const c_void, b: *const c_void, o: *mut c_void, n: u32);
        pub fn bitwise_or_u32(a: *const c_void, b: *const c_void, o: *mut c_void, n: u32);
        pub fn bitwise_xor_u32(a: *const c_void, b: *const c_void, o: *mut c_void, n: u32);
        pub fn bitwise_and_i64(a: *const c_void, b: *const c_void, o: *mut c_void, n: u32);
        pub fn bitwise_or_i64(a: *const c_void, b: *const c_void, o: *mut c_void, n: u32);
        pub fn bitwise_xor_i64(a: *const c_void, b: *const c_void, o: *mut c_void, n: u32);
        pub fn bitwise_and_i32(a: *const c_void, b: *const c_void, o: *mut c_void, n: u32);
        pub fn bitwise_or_i32(a: *const c_void, b: *const c_void, o: *mut c_void, n: u32);
        pub fn bitwise_xor_i32(a: *const c_void, b: *const c_void, o: *mut c_void, n: u32);
        pub fn leftshift_u8(a: *const c_void, o: *mut c_void, n: u32, k: i32);
        pub fn leftshift_u32(a: *const c_void, o: *mut c_void, n: u32, k: i32);
        pub fn leftshift_i64(a: *const c_void, o: *mut c_void, n: u32, k: i32);
        pub fn leftshift_i32(a: *const c_void, o: *mut c_void, n: u32, k: i32);
        pub fn gptoss_swiglu_f16(g: *const c_void, u: *const c_void, o: *mut c_void, n: u32, a: f32, l: f32, s: *mut c_void);
        pub fn gptoss_swiglu_bf16(g: *const c_void, u: *const c_void, o: *mut c_void, n: u32, a: f32, l: f32, s: *mut c_void);
        pub fn gptoss_swiglu_f32(g: *const c_void, u: *const c_void, o: *mut c_void, n: u32, a: f32, l: f32, s: *mut c_void);
        pub fn gptoss_swiglu_interleaved_f16(g: *const c_void, o: *mut c_void, n: u32, i: u32, a: f32, l: f32, s: *mut c_void);
        pub fn gptoss_swiglu_interleaved_bf16(g: *const c_void, o: *mut c_void, n: u32, i: u32, a: f32, l: f32, s: *mut c_void);
        pub fn gptoss_swiglu_interleaved_f32(g: *const c_void, o: *mut c_void, n: u32, i: u32, a: f32, l: f32, s: *mut c_void);
        pub fn softmax_with_sinks_f16(l: *const c_void, s: *const c_void, m: *const c_void, o: *mut c_void, b: i32, h: i32, q: i32, k: i32, sc: f32, st: *mut c_void);
        pub fn softmax_with_sinks_bf16(l: *const c_void, s: *const c_void, m: *const c_void, o: *mut c_void, b: i32, h: i32, q: i32, k: i32, sc: f32, st: *mut c_void);
        pub fn softmax_with_sinks_f32(l: *const c_void, s: *const c_void, m: *const c_void, o: *mut c_void, b: i32, h: i32, q: i32, k: i32, sc: f32, st: *mut c_void);
        pub fn fused_glu_f16(a: *const c_void, b: *const c_void, o: *mut c_void, n: u32, act: i32, s: *mut c_void);
        pub fn fused_glu_bf16(a: *const c_void, b: *const c_void, o: *mut c_void, n: u32, act: i32, s: *mut c_void);
        pub fn fused_glu_f32(a: *const c_void, b: *const c_void, o: *mut c_void, n: u32, act: i32, s: *mut c_void);
        pub fn softcap_f32(i: *const c_void, o: *mut c_void, n: u32, cap: f32, s: *mut c_void);
        pub fn softcap_f16_to_f32(i: *const c_void, o: *mut c_void, n: u32, cap: f32, s: *mut c_void);
        pub fn softcap_bf16_to_f32(i: *const c_void, o: *mut c_void, n: u32, cap: f32, s: *mut c_void);
        pub fn count_nonzero_bf16(d: *const c_void, n: u32, s: *mut c_void) -> u32;
        pub fn count_nonzero_f16(d: *const c_void, n: u32, s: *mut c_void) -> u32;
        pub fn count_nonzero_f32(d: *const c_void, n: u32, s: *mut c_void) -> u32;
        pub fn count_nonzero_f64(d: *const c_void, n: u32, s: *mut c_void) -> u32;
        pub fn count_nonzero_u8(d: *const c_void, n: u32, s: *mut c_void) -> u32;
        pub fn count_nonzero_u32(d: *const c_void, n: u32, s: *mut c_void) -> u32;
        pub fn count_nonzero_i16(d: *const c_void, n: u32, s: *mut c_void) -> u32;
        pub fn count_nonzero_i64(d: *const c_void, n: u32, s: *mut c_void) -> u32;
        pub fn count_nonzero_i32(d: *const c_void, n: u32, s: *mut c_void) -> u32;
        pub fn nonzero_f32(d: *const c_void, n: u32, nz: u32, dims: *const c_void, nd: u32, o: *mut c_void, s: *mut c_void);
        pub fn nonzero_f64(d: *const c_void, n: u32, nz: u32, dims: *const c_void, nd: u32, o: *mut c_void, s: *mut c_void);
        pub fn nonzero_u8(d: *const c_void, n: u32, nz: u32, dims: *const c_void, nd: u32, o: *mut c_void, s: *mut c_void);
        pub fn nonzero_u32(d: *const c_void, n: u32, nz: u32, dims: *const c_void, nd: u32, o: *mut c_void, s: *mut c_void);
        pub fn nonzero_i64(d: *const c_void, n: u32, nz: u32, dims: *const c_void, nd: u32, o: *mut c_void, s: *mut c_void);
        pub fn nonzero_i16(d: *const c_void, n: u32, nz: u32, dims: *const c_void, nd: u32, o: *mut c_void, s: *mut c_void);
        pub fn nonzero_i32(d: *const c_void, n: u32, nz: u32, dims: *const c_void, nd: u32, o: *mut c_void, s: *mut c_void);

        pub fn launch_gemv_bf16(a: *const u16, x: *const u16, b: *const u16, y: *mut u16, m: i32, k: i32, bs: i32, hb: bool, s: *mut c_void);
        pub fn launch_gemv_f16(a: *const u16, x: *const u16, b: *const u16, y: *mut u16, m: i32, k: i32, bs: i32, hb: bool, s: *mut c_void);
        pub fn launch_gemv_f32(a: *const f32, x: *const f32, b: *const f32, y: *mut f32, m: i32, k: i32, bs: i32, hb: bool, s: *mut c_void);

        pub fn launch_indexed_moe_forward_q4_0_q8_1(w: *const c_void, x: *const c_void, i: *const u32, o: *mut f32, n: i32, k: i32, b: i32, t: i32, kp: i32, d1: i32, s: *mut c_void);
        pub fn launch_moe_gemv_fused_gate_up_q4_0_q8_1(g: *const c_void, u: *const c_void, x: *const c_void, i: *const u32, o: *mut f32, n: i32, k: i32, b: i32, t: i32, kp: i32, a: i32, s: *mut c_void);
        pub fn launch_moe_gemv_down_aggregate_q4_0_q8_1(w: *const c_void, x: *const c_void, i: *const u32, tw: *const f32, o: *mut f32, n: i32, k: i32, b: i32, t: i32, kp: i32, s: *mut c_void);
        pub fn launch_indexed_moe_forward_q4_1_q8_1(w: *const c_void, x: *const c_void, i: *const u32, o: *mut f32, n: i32, k: i32, b: i32, t: i32, kp: i32, d1: i32, s: *mut c_void);
        pub fn launch_moe_gemv_fused_gate_up_q4_1_q8_1(g: *const c_void, u: *const c_void, x: *const c_void, i: *const u32, o: *mut f32, n: i32, k: i32, b: i32, t: i32, kp: i32, a: i32, s: *mut c_void);
        pub fn launch_moe_gemv_down_aggregate_q4_1_q8_1(w: *const c_void, x: *const c_void, i: *const u32, tw: *const f32, o: *mut f32, n: i32, k: i32, b: i32, t: i32, kp: i32, s: *mut c_void);
        pub fn launch_indexed_moe_forward_q5_0_q8_1(w: *const c_void, x: *const c_void, i: *const u32, o: *mut f32, n: i32, k: i32, b: i32, t: i32, kp: i32, d1: i32, s: *mut c_void);
        pub fn launch_moe_gemv_fused_gate_up_q5_0_q8_1(g: *const c_void, u: *const c_void, x: *const c_void, i: *const u32, o: *mut f32, n: i32, k: i32, b: i32, t: i32, kp: i32, a: i32, s: *mut c_void);
        pub fn launch_moe_gemv_down_aggregate_q5_0_q8_1(w: *const c_void, x: *const c_void, i: *const u32, tw: *const f32, o: *mut f32, n: i32, k: i32, b: i32, t: i32, kp: i32, s: *mut c_void);
        pub fn launch_indexed_moe_forward_q5_1_q8_1(w: *const c_void, x: *const c_void, i: *const u32, o: *mut f32, n: i32, k: i32, b: i32, t: i32, kp: i32, d1: i32, s: *mut c_void);
        pub fn launch_moe_gemv_fused_gate_up_q5_1_q8_1(g: *const c_void, u: *const c_void, x: *const c_void, i: *const u32, o: *mut f32, n: i32, k: i32, b: i32, t: i32, kp: i32, a: i32, s: *mut c_void);
        pub fn launch_moe_gemv_down_aggregate_q5_1_q8_1(w: *const c_void, x: *const c_void, i: *const u32, tw: *const f32, o: *mut f32, n: i32, k: i32, b: i32, t: i32, kp: i32, s: *mut c_void);
        pub fn launch_indexed_moe_forward_q8_0_q8_1(w: *const c_void, x: *const c_void, i: *const u32, o: *mut f32, n: i32, k: i32, b: i32, t: i32, kp: i32, d1: i32, s: *mut c_void);
        pub fn launch_moe_gemv_fused_gate_up_q8_0_q8_1(g: *const c_void, u: *const c_void, x: *const c_void, i: *const u32, o: *mut f32, n: i32, k: i32, b: i32, t: i32, kp: i32, a: i32, s: *mut c_void);
        pub fn launch_moe_gemv_down_aggregate_q8_0_q8_1(w: *const c_void, x: *const c_void, i: *const u32, tw: *const f32, o: *mut f32, n: i32, k: i32, b: i32, t: i32, kp: i32, s: *mut c_void);
        pub fn launch_indexed_moe_forward_q2k_q8_1(w: *const c_void, x: *const c_void, i: *const u32, o: *mut f32, n: i32, k: i32, b: i32, t: i32, kp: i32, d1: i32, s: *mut c_void);
        pub fn launch_moe_gemv_fused_gate_up_q2k_q8_1(g: *const c_void, u: *const c_void, x: *const c_void, i: *const u32, o: *mut f32, n: i32, k: i32, b: i32, t: i32, kp: i32, a: i32, s: *mut c_void);
        pub fn launch_moe_gemv_down_aggregate_q2k_q8_1(w: *const c_void, x: *const c_void, i: *const u32, tw: *const f32, o: *mut f32, n: i32, k: i32, b: i32, t: i32, kp: i32, s: *mut c_void);
        pub fn launch_indexed_moe_forward_q3k_q8_1(w: *const c_void, x: *const c_void, i: *const u32, o: *mut f32, n: i32, k: i32, b: i32, t: i32, kp: i32, d1: i32, s: *mut c_void);
        pub fn launch_moe_gemv_fused_gate_up_q3k_q8_1(g: *const c_void, u: *const c_void, x: *const c_void, i: *const u32, o: *mut f32, n: i32, k: i32, b: i32, t: i32, kp: i32, a: i32, s: *mut c_void);
        pub fn launch_moe_gemv_down_aggregate_q3k_q8_1(w: *const c_void, x: *const c_void, i: *const u32, tw: *const f32, o: *mut f32, n: i32, k: i32, b: i32, t: i32, kp: i32, s: *mut c_void);
        pub fn launch_indexed_moe_forward_q4k_q8_1(w: *const c_void, x: *const c_void, i: *const u32, o: *mut f32, n: i32, k: i32, b: i32, t: i32, kp: i32, d1: i32, s: *mut c_void);
        pub fn launch_moe_gemv_fused_gate_up_q4k_q8_1(g: *const c_void, u: *const c_void, x: *const c_void, i: *const u32, o: *mut f32, n: i32, k: i32, b: i32, t: i32, kp: i32, a: i32, s: *mut c_void);
        pub fn launch_moe_gemv_down_aggregate_q4k_q8_1(w: *const c_void, x: *const c_void, i: *const u32, tw: *const f32, o: *mut f32, n: i32, k: i32, b: i32, t: i32, kp: i32, s: *mut c_void);
        pub fn launch_indexed_moe_forward_q5k_q8_1(w: *const c_void, x: *const c_void, i: *const u32, o: *mut f32, n: i32, k: i32, b: i32, t: i32, kp: i32, d1: i32, s: *mut c_void);
        pub fn launch_moe_gemv_fused_gate_up_q5k_q8_1(g: *const c_void, u: *const c_void, x: *const c_void, i: *const u32, o: *mut f32, n: i32, k: i32, b: i32, t: i32, kp: i32, a: i32, s: *mut c_void);
        pub fn launch_moe_gemv_down_aggregate_q5k_q8_1(w: *const c_void, x: *const c_void, i: *const u32, tw: *const f32, o: *mut f32, n: i32, k: i32, b: i32, t: i32, kp: i32, s: *mut c_void);
        pub fn launch_indexed_moe_forward_q6k_q8_1(w: *const c_void, x: *const c_void, i: *const u32, o: *mut f32, n: i32, k: i32, b: i32, t: i32, kp: i32, d1: i32, s: *mut c_void);
        pub fn launch_moe_gemv_fused_gate_up_q6k_q8_1(g: *const c_void, u: *const c_void, x: *const c_void, i: *const u32, o: *mut f32, n: i32, k: i32, b: i32, t: i32, kp: i32, a: i32, s: *mut c_void);
        pub fn launch_moe_gemv_down_aggregate_q6k_q8_1(w: *const c_void, x: *const c_void, i: *const u32, tw: *const f32, o: *mut f32, n: i32, k: i32, b: i32, t: i32, kp: i32, s: *mut c_void);
        pub fn launch_indexed_moe_forward_q8_1_q8_1(w: *const c_void, x: *const c_void, i: *const u32, o: *mut f32, n: i32, k: i32, b: i32, t: i32, kp: i32, d1: i32, s: *mut c_void);
        pub fn launch_moe_gemv_fused_gate_up_q8_1_q8_1(g: *const c_void, u: *const c_void, x: *const c_void, i: *const u32, o: *mut f32, n: i32, k: i32, b: i32, t: i32, kp: i32, a: i32, s: *mut c_void);
        pub fn launch_moe_gemv_down_aggregate_q8_1_q8_1(w: *const c_void, x: *const c_void, i: *const u32, tw: *const f32, o: *mut f32, n: i32, k: i32, b: i32, t: i32, kp: i32, s: *mut c_void);
        pub fn launch_quantize_q8_1(x: *const f32, vy: *mut c_void, kx: i32, kp: i32, nbx: i32, rows: i32, s: *mut c_void);
        pub fn launch_quantize_q8_1_bf16(x: *const c_void, vy: *mut c_void, kx: i32, kp: i32, rows: i32, s: *mut c_void);
        pub fn launch_quantize_q8_1_f16(x: *const c_void, vy: *mut c_void, kx: i32, kp: i32, rows: i32, s: *mut c_void);

        pub fn launch_moe_dispatch(t: *const i32, eb: *mut i32, st: *mut i32, ss: *mut i32, n: i32, ne: i32, tk: i32, ec: *mut i32, cu: *mut i32, s: *mut c_void);
        pub fn launch_moe_weighted_reduce_flat(i: *const f32, w: *const f32, o: *mut f32, t: i32, h: i32, k: i32, s: *mut c_void);
        pub fn launch_moe_weighted_reduce_flat_bf16(i: *const f32, w: *const f32, o: *mut c_void, t: i32, h: i32, k: i32, s: *mut c_void);
        pub fn launch_moe_grouped_gemm_q8_0(w: *const c_void, x: *const c_void, eb: *const i32, st: *const i32, tw: *const f32, o: *mut f32, n: i32, k: i32, kp: i32, ne: i32, tk: i32, d1: i32, s: *mut c_void);
        pub fn launch_moe_grouped_gemm_q4_0(w: *const c_void, x: *const c_void, eb: *const i32, st: *const i32, tw: *const f32, o: *mut f32, n: i32, k: i32, kp: i32, ne: i32, tk: i32, d1: i32, s: *mut c_void);
        pub fn launch_moe_grouped_gemm_q4_1(w: *const c_void, x: *const c_void, eb: *const i32, st: *const i32, tw: *const f32, o: *mut f32, n: i32, k: i32, kp: i32, ne: i32, tk: i32, d1: i32, s: *mut c_void);
        pub fn launch_moe_grouped_gemm_q5_0(w: *const c_void, x: *const c_void, eb: *const i32, st: *const i32, tw: *const f32, o: *mut f32, n: i32, k: i32, kp: i32, ne: i32, tk: i32, d1: i32, s: *mut c_void);
        pub fn launch_moe_grouped_gemm_q5_1(w: *const c_void, x: *const c_void, eb: *const i32, st: *const i32, tw: *const f32, o: *mut f32, n: i32, k: i32, kp: i32, ne: i32, tk: i32, d1: i32, s: *mut c_void);
        pub fn launch_moe_grouped_gemm_q8_1(w: *const c_void, x: *const c_void, eb: *const i32, st: *const i32, tw: *const f32, o: *mut f32, n: i32, k: i32, kp: i32, ne: i32, tk: i32, d1: i32, s: *mut c_void);
        pub fn launch_moe_grouped_gemm_q2k(w: *const c_void, x: *const c_void, eb: *const i32, st: *const i32, tw: *const f32, o: *mut f32, n: i32, k: i32, kp: i32, ne: i32, tk: i32, d1: i32, s: *mut c_void);
        pub fn launch_moe_grouped_gemm_q3k(w: *const c_void, x: *const c_void, eb: *const i32, st: *const i32, tw: *const f32, o: *mut f32, n: i32, k: i32, kp: i32, ne: i32, tk: i32, d1: i32, s: *mut c_void);
        pub fn launch_moe_grouped_gemm_q4k(w: *const c_void, x: *const c_void, eb: *const i32, st: *const i32, tw: *const f32, o: *mut f32, n: i32, k: i32, kp: i32, ne: i32, tk: i32, d1: i32, s: *mut c_void);
        pub fn launch_moe_grouped_gemm_q5k(w: *const c_void, x: *const c_void, eb: *const i32, st: *const i32, tw: *const f32, o: *mut f32, n: i32, k: i32, kp: i32, ne: i32, tk: i32, d1: i32, s: *mut c_void);
        pub fn launch_moe_grouped_gemm_q6k(w: *const c_void, x: *const c_void, eb: *const i32, st: *const i32, tw: *const f32, o: *mut f32, n: i32, k: i32, kp: i32, ne: i32, tk: i32, d1: i32, s: *mut c_void);

        pub fn dequantize_8bit_u8_kernel_f16(wq: *const u8, s: *const u16, z: *const u16, o: *const u16, h: i32, w: i32);
        pub fn dequantize_8bit_u8_kernel_bf16(wq: *const u8, s: *const u16, z: *const u16, o: *const u16, h: i32, w: i32);
        pub fn dequantize_4bit_u8_kernel_f16(wq: *const u8, s: *const u16, z: *const u16, o: *const u16, h: i32, w: i32);
        pub fn dequantize_4bit_u8_kernel_bf16(wq: *const u8, s: *const u16, z: *const u16, o: *const u16, h: i32, w: i32);
        pub fn dequantize_2bit_u8_kernel_f16(wq: *const u8, s: *const u16, z: *const u16, o: *const u16, h: i32, w: i32);
        pub fn dequantize_2bit_u8_kernel_bf16(wq: *const u8, s: *const u16, z: *const u16, o: *const u16, h: i32, w: i32);
        pub fn dequantize_1bit_u8_kernel_f16(wq: *const u8, s: *const u16, z: *const u16, o: *const u16, h: i32, w: i32);
        pub fn dequantize_1bit_u8_kernel_bf16(wq: *const u8, s: *const u16, z: *const u16, o: *const u16, h: i32, w: i32);
        pub fn dequantize_3bit_32_kernel_f16(wq: *const i32, s: *const u16, z: *const u16, o: *const u16, h: i32, w: i32);
        pub fn dequantize_3bit_32_kernel_bf16(wq: *const i32, s: *const u16, z: *const u16, o: *const u16, h: i32, w: i32);
        pub fn nonzero_f16(d: *const c_void, n: u32, nz: u32, dims: *const c_void, nd: u32, o: *mut c_void, s: *mut c_void);
        pub fn nonzero_bf16(d: *const c_void, n: u32, nz: u32, dims: *const c_void, nd: u32, o: *mut c_void, s: *mut c_void);

        pub fn dequantize_blockwise_f32_int8(c: *const f32, a: *const u8, am: *const f32, o: *mut f32, bs: i32, n: i32, s: *mut c_void);
        pub fn dequantize_blockwise_f32_fp4(c: *const f32, a: *const u8, am: *const f32, o: *mut f32, bs: i32, n: i32, s: *mut c_void);
        pub fn dequantize_blockwise_f32_nf4(c: *const f32, a: *const u8, am: *const f32, o: *mut f32, bs: i32, n: i32, s: *mut c_void);
        pub fn dequantize_blockwise_f16_int8(c: *const f32, a: *const u8, am: *const f32, o: *mut u16, bs: i32, n: i32, s: *mut c_void);
        pub fn dequantize_blockwise_f16_fp4(c: *const f32, a: *const u8, am: *const f32, o: *mut u16, bs: i32, n: i32, s: *mut c_void);
        pub fn dequantize_blockwise_f16_nf4(c: *const f32, a: *const u8, am: *const f32, o: *mut u16, bs: i32, n: i32, s: *mut c_void);
        pub fn dequantize_blockwise_bf16_int8(c: *const f32, a: *const u8, am: *const f32, o: *mut u16, bs: i32, n: i32, s: *mut c_void);
        pub fn dequantize_blockwise_bf16_fp4(c: *const f32, a: *const u8, am: *const f32, o: *mut u16, bs: i32, n: i32, s: *mut c_void);
        pub fn dequantize_blockwise_bf16_nf4(c: *const f32, a: *const u8, am: *const f32, o: *mut u16, bs: i32, n: i32, s: *mut c_void);
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
    /// canonicalized cases whose raw bytes differed (order-only differences).
    pub order_only: usize,
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
    pub fn stream(&self, i: usize) -> *mut c_void {
        if i == 0 { std::ptr::null_mut() } else { self.streams[i].cu_stream() as *mut c_void }
    }

    fn upload(&self, bufs: &[Vec<u8>]) -> Side {
        Side { bufs: bufs.iter().map(|b| DeviceBuffer::from_host(&self.streams[0], b).unwrap()).collect() }
    }

    /// Run `f` for the C side then the Rust side, each on private copies of `bufs` (each followed by
    /// a random guard), and compare every byte of every buffer afterwards.
    pub fn case(&mut self, label: &str, bufs: Vec<Vec<u8>>, f: impl Fn(bool, &Side)) {
        self.case_canon(label, bufs, f, |_| {});
    }

    /// Run `f` on each side and download every buffer (data + guard).
    fn run_sides(&mut self, label: &str, bufs: &[Vec<u8>], f: &impl Fn(bool, &Side), rust_second: bool) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
        let (a, b) = (self.upload(bufs), self.upload(bufs));
        self.ctx.synchronize().unwrap();
        // A launch error the C launchers dropped earlier (e.g. an empty grid) stays in the runtime's
        // last-error slot, and ops.cu's CUDA_CHECK(cudaGetLastError()) would report it at the next
        // call: start every C call from a clean slot (the Rust twins cannot see runtime state).
        unsafe { cref::cudaGetLastError() };
        f(false, &a);
        self.ctx.synchronize().unwrap_or_else(|e| panic!("{label}: C side: {e:?}"));
        f(rust_second, &b);
        self.ctx.synchronize().unwrap_or_else(|e| panic!("{label}: second side: {e:?}"));
        let down = |s: &Side| -> Vec<Vec<u8>> { s.bufs.iter().map(|x| x.to_host_vec(&self.streams[0]).unwrap()).collect() };
        (down(&a), down(&b))
    }

    /// `case` for launchers whose reference output is only defined up to an order the hardware
    /// picks (atomics): `canon` maps both sides' downloaded buffers to a canonical form first. Cases
    /// that were also byte-identical before canonicalizing are counted in `exact_canon`.
    pub fn case_canon(&mut self, label: &str, bufs: Vec<Vec<u8>>, f: impl Fn(bool, &Side), canon: impl Fn(&mut Vec<Vec<u8>>)) {
        let bufs: Vec<Vec<u8>> = bufs
            .into_iter()
            .map(|mut b| {
                b.extend(self.rng.bytes(GUARD));
                b
            })
            .collect();
        let (mut x, mut y) = self.run_sides(label, &bufs, &f, true);
        let exact = x == y;
        canon(&mut x);
        canon(&mut y);
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
        if !exact && d.differing == 0 {
            self.order_only += 1;
        }
        self.calls += 2;
        self.family_calls += 2;
        self.t.record(label, &d);
    }

    /// Diagnostic: is the C launcher itself byte-deterministic on this input? (C vs C.)
    pub fn ref_deterministic(&mut self, label: &str, bufs: Vec<Vec<u8>>, f: impl Fn(bool, &Side)) -> bool {
        let (x, y) = self.run_sides(label, &bufs, &f, false);
        x == y
    }

    /// Run `f` on the Rust side only (fresh device copies of `bufs`) and download every buffer.
    pub fn rust_only(&mut self, label: &str, bufs: &[Vec<u8>], f: impl Fn(&Side)) -> Vec<Vec<u8>> {
        let a = self.upload(bufs);
        self.ctx.synchronize().unwrap();
        f(&a);
        self.ctx.synchronize().unwrap_or_else(|e| panic!("{label}: rust side: {e:?}"));
        a.bufs.iter().map(|x| x.to_host_vec(&self.streams[0]).unwrap()).collect()
    }

    /// Record a host-side check as one launch (`ok == false` is a failure).
    pub fn check(&mut self, label: &str, ok: bool) {
        let d = kdiff::Diff { bytes: 1, differing: (!ok) as usize, first: if ok { None } else { Some((0, 0, 0, 1)) } };
        self.calls += 1;
        self.family_calls += 1;
        self.t.record(label, &d);
    }

    pub fn f32s(&mut self, n: usize) -> Vec<u8> {
        as_bytes(&self.rng.f32s(n))
    }
    pub fn b16s(&mut self, n: usize) -> Vec<u8> {
        as_bytes(&self.rng.b16s(n))
    }
    /// Values of dtype `t` (0 f16, 1 bf16, 2 f32) from f32s with edge cases, plus raw 16-bit patterns.
    pub fn vals(&mut self, t: usize, n: usize) -> Vec<u8> {
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

    /// Compare two host-side results (e.g. a launcher's return value).
    pub fn scalar(&mut self, label: &str, a: u64, b: u64) {
        let d = kdiff::Diff { bytes: 8, differing: (a != b) as usize * 8, first: if a != b { Some((0, 0, a as u8, b as u8)) } else { None } };
        self.t.record(label, &d);
    }

    fn family(&mut self, name: &str) {
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

// ================================================================================================
// families

fn hqq_bitpack(g: &mut G) {
    type P = unsafe extern "C" fn(*const u8, *mut u8, usize, usize, *mut c_void);
    let packs: [(&str, P, P, usize); 3] = [
        ("pack_1bit", cref::launch_pack_1bit_kernel, ox::launch_pack_1bit_kernel, 8),
        ("pack_2bit", cref::launch_pack_2bit_kernel, ox::launch_pack_2bit_kernel, 4),
        ("pack_4bit", cref::launch_pack_4bit_kernel, ox::launch_pack_4bit_kernel, 2),
    ];
    let shapes: [(usize, usize); 8] = [(64, 1), (80, 33), (7, 5), (1003, 17), (16, 300), (3, 3), (0, 9), (4096, 64)];
    for (si, &(n, w)) in shapes.iter().enumerate() {
        for &(name, cf, of, per) in &packs {
            let inp = g.rng.bytes(n * w);
            let out = g.rng.bytes((n / per) * w);
            let s = g.stream(si % 2);
            g.case(&format!("{name} n={n} w={w}"), vec![inp, out], |rust, b| unsafe {
                let f = if rust { of } else { cf };
                f(b.p(0) as _, b.p(1) as _, n, w, s)
            });
        }
        // 3-bit: u32 in, i32 out
        let inp = g.rng.bytes(n * w * 4);
        let out = g.rng.bytes((n / 10) * w * 4);
        let s = g.stream((si + 1) % 2);
        g.case(&format!("pack_3bit n={n} w={w}"), vec![inp, out], |rust, b| unsafe {
            if rust {
                ox::launch_pack_3bit_kernel(b.p(0) as _, b.p(1) as _, n, w, s)
            } else {
                cref::launch_pack_3bit_kernel(b.p(0) as _, b.p(1) as _, n, w, s)
            }
        });
        let ne = n * w;
        let inp = g.rng.bytes(ne);
        let out = g.rng.bytes(ne);
        g.case(&format!("pack_8bit n={ne}"), vec![inp, out], |rust, b| unsafe {
            if rust {
                ox::launch_pack_8bit_kernel(b.p(0) as _, b.p(1) as _, ne, s)
            } else {
                cref::launch_pack_8bit_kernel(b.p(0) as _, b.p(1) as _, ne, s)
            }
        });
    }
    g.family("hqq_bitpack");
}

fn hqq(g: &mut G) {
    type D = unsafe extern "C" fn(*const u8, *const f32, *const f32, *const f32, i32, i32);
    let fns: [(&str, D, D, usize, usize); 5] = [
        ("8bit", cref::dequantize_8bit_u8_kernel_f32, ox::dequantize_8bit_u8_kernel_f32, 1, 1),
        ("4bit", cref::dequantize_4bit_u8_kernel_f32, ox::dequantize_4bit_u8_kernel_f32, 2, 1),
        ("2bit", cref::dequantize_2bit_u8_kernel_f32, ox::dequantize_2bit_u8_kernel_f32, 4, 1),
        ("1bit", cref::dequantize_1bit_u8_kernel_f32, ox::dequantize_1bit_u8_kernel_f32, 8, 1),
        ("3bit", unsafe { std::mem::transmute::<unsafe extern "C" fn(*const i32, *const f32, *const f32, *const f32, i32, i32), D>(cref::dequantize_3bit_32_kernel_f32) },
                 unsafe { std::mem::transmute::<unsafe extern "C" fn(*const i32, *const f32, *const f32, *const f32, i32, i32), D>(ox::dequantize_3bit_32_kernel_f32) }, 10, 4),
    ];
    for &(h, w) in &[(1, 1), (3, 7), (64, 128), (17, 255), (256, 1), (1, 1000), (5, 64)] {
        for &(name, cf, of, chunks, wbytes) in &fns {
            let n = h * w;
            let wq = g.rng.bytes(n * wbytes);
            let (s, z) = (g.f32s(w), g.f32s(w));
            // small integer-ish zeros too, so (q - z) hits exact zero / sign changes
            let z = if h % 2 == 1 { as_bytes(&(0..w).map(|i| (i % 9) as f32 - 0.0).collect::<Vec<f32>>()) } else { z };
            let out = g.rng.bytes(n * chunks * 4);
            g.case(&format!("hqq_{name}_f32 h={h} w={w}"), vec![wq, s, z, out], |rust, b| unsafe {
                let f = if rust { of } else { cf };
                f(b.p(0) as _, b.p(1) as _, b.p(2) as _, b.p(3) as _, h as i32, w as i32)
            });
        }
    }
    g.family("hqq (f32 launchers)");
}

fn bnb(g: &mut G) {
    type D = unsafe extern "C" fn(*const f32, *const u8, *const f32, *mut c_void, i32, i32, *mut c_void);
    macro_rules! pair { ($n:ident) => { unsafe { (std::mem::transmute::<*const (), D>(cref::$n as *const ()), std::mem::transmute::<*const (), D>(ox::$n as *const ())) } }; }
    let fns: [(&str, (D, D), usize, usize); 9] = [
        ("f32_int8", pair!(dequantize_blockwise_f32_int8), 4, 0),
        ("f32_fp4", pair!(dequantize_blockwise_f32_fp4), 4, 1),
        ("f32_nf4", pair!(dequantize_blockwise_f32_nf4), 4, 2),
        ("f16_int8", pair!(dequantize_blockwise_f16_int8), 2, 0),
        ("f16_fp4", pair!(dequantize_blockwise_f16_fp4), 2, 1),
        ("f16_nf4", pair!(dequantize_blockwise_f16_nf4), 2, 2),
        ("bf16_int8", pair!(dequantize_blockwise_bf16_int8), 2, 0),
        ("bf16_fp4", pair!(dequantize_blockwise_bf16_fp4), 2, 1),
        ("bf16_nf4", pair!(dequantize_blockwise_bf16_nf4), 2, 2),
    ];
    let mut k = 0usize;
    for &n in &[1i32, 7, 512, 513, 1000, 1024, 1025, 4096, 5003] {
        for &bs in &[64i32, 128, 256, 4096] {
            for &(name, (cf, of), es, dt) in &fns {
                k += 1;
                let tile = if dt > 0 { 1024 } else { 512 };
                let grid = (n + tile - 1) / tile;
                let nbytes = if dt > 0 { (n as usize).div_ceil(2) } else { n as usize };
                let bse = if dt > 0 { bs / 2 } else { bs } as usize;
                let nabs = (grid as usize * 512 + 512) / bse + 2;
                let code = g.f32s(256);
                let a = g.rng.bytes(nbytes);
                // absmax: the edge-case mix, sometimes denormal-heavy
                let am: Vec<f32> = if k % 3 == 0 { (0..nabs).map(|i| [1e-40f32, -3e-39, 1.0, -0.0][i % 4]).collect() } else { g.rng.f32s(nabs) };
                let am = as_bytes(&am);
                let out = g.rng.bytes(n as usize * es);
                let s = g.stream(k % 2);
                g.case(&format!("bnb_{name} n={n} bs={bs}"), vec![code, a, am, out], |rust, b| unsafe {
                    let f = if rust { of } else { cf };
                    f(b.p(0) as _, b.p(1) as _, b.p(2) as _, b.p(3), bs, n, s)
                });
            }
        }
    }
    g.family("dequant (bitsandbytes)");
}

fn gelu_tanh_and_mul(g: &mut G) {
    type A = unsafe extern "C" fn(*mut c_void, *const c_void, i32, i32, *mut c_void);
    let fns: [(&str, A, A, usize); 3] = [
        ("gelu_tanh_and_mul_bf16", cref::launch_gelu_tanh_and_mul_bf16, ox::launch_gelu_tanh_and_mul_bf16, 1),
        ("silu_and_mul_bf16", cref::launch_silu_and_mul_bf16, ox::launch_silu_and_mul_bf16, 1),
        ("gelu_tanh_and_mul_f16", cref::launch_gelu_tanh_and_mul_f16, ox::launch_gelu_tanh_and_mul_f16, 0),
    ];
    let mut k = 0;
    for &(t, d) in &[(1i32, 1i32), (3, 7), (2, 1024), (5, 1500), (1, 4097), (0, 64), (7, 33), (4, 2048)] {
        for &(name, cf, of, ty) in &fns {
            k += 1;
            let inp = g.vals(ty, (t * 2 * d) as usize);
            let out = g.rng.bytes((t * d) as usize * 2);
            let s = g.stream(k % 2);
            g.case(&format!("{name} tokens={t} d={d}"), vec![inp, out], |rust, b| unsafe {
                (if rust { of } else { cf })(b.p(1), b.p(0), t, d, s)
            });
        }
    }
    for &(t, h, topk) in &[(1i32, 1i32, 1i32), (3, 300, 8), (2, 4096, 2), (5, 257, 3), (0, 10, 2), (4, 64, 0), (2, 1000, 17)] {
        k += 1;
        let inp = g.vals(1, (t * h * topk) as usize);
        let out = g.rng.bytes((t * h) as usize * 2);
        let s = g.stream(k % 2);
        g.case(&format!("moe_sum_bf16 tokens={t} hidden={h} topk={topk}"), vec![inp, out], |rust, b| unsafe {
            if rust { ox::launch_moe_sum_bf16(b.p(1), b.p(0), t, h, topk, s) } else { cref::launch_moe_sum_bf16(b.p(1), b.p(0), t, h, topk, s) }
        });
    }
    g.family("gelu_tanh_and_mul");
}

fn rotary(g: &mut G) {
    let mut k = 0;
    // (tokens, heads, kv_heads, head_size, rot_dim, extra q stride, extra k stride)
    let shapes: [(i64, i32, i32, i32, i32, i64, i64); 7] = [
        (1, 1, 0, 8, 4, 0, 0),
        (3, 4, 2, 64, 32, 0, 0),
        (5, 8, 8, 128, 64, 16, 8),
        (2, 32, 8, 128, 32, 0, 0), // 1024 pairs: block capped at 512
        (7, 3, 1, 20, 7, 5, 3),
        (4, 2, 0, 64, 16, 0, 0),
        (1, 1, 1, 6, 3, 1, 0),
    ];
    for &(nt, nh, nkv, hs, rd, eq, ek) in &shapes {
        for dtype in 0u32..4 {
            for neox in [0i32, 1, 7] {
                for pos in [false, true] {
                    k += 1;
                    let ty = [0usize, 1, 2, 2][dtype as usize];
                    let es = if ty == 2 { 4 } else { 2 };
                    let qs = nh as i64 * hs as i64 + eq;
                    let ks = nkv as i64 * hs as i64 + ek;
                    let max_pos = 11usize;
                    let rows = if pos { max_pos } else { nt as usize };
                    let q = g.vals(ty, nt as usize * qs as usize);
                    let kk = g.vals(ty, (nt as usize * ks as usize).max(1));
                    let c = g.vals(ty, rows * rd as usize);
                    let sn = g.vals(ty, rows * rd as usize);
                    let p: Vec<u32> = (0..nt).map(|_| (g.rng.next() % max_pos as u64) as u32).collect();
                    let null_key = nkv == 0;
                    let s = if k % 2 == 0 { 0i64 } else { g.stream(1) as i64 };
                    let label = format!("rotary{} dtype={dtype} neox={neox} nt={nt} nh={nh} nkv={nkv} hs={hs} rd={rd}", if pos { "_positions" } else { "" });
                    let _ = es;
                    g.case(&label, vec![q, kk, c, sn, as_bytes(&p)], |rust, b| unsafe {
                        let key = if null_key { std::ptr::null() } else { b.p(1) as *const c_void };
                        if pos {
                            let f = if rust { ox::rotary_embedding_positions } else { cref::rotary_embedding_positions };
                            f(b.p(0), key, b.p(2), b.p(3), b.p(4), neox, hs, nt, rd, 11, nh, nkv, qs, ks, dtype, s)
                        } else {
                            let f = if rust { ox::rotary_embedding } else { cref::rotary_embedding };
                            f(b.p(0), key, b.p(2), b.p(3), neox, hs, nt, rd, nh, nkv, qs, ks, dtype, s)
                        }
                    });
                }
            }
        }
    }
    g.family("rotary");
}

fn i32s_of(b: &[u8]) -> Vec<i32> {
    b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn moe_align(g: &mut G) {
    let mut k = 0;
    let mut nondet = 0;
    // (num_tokens, topk, num_experts, block_size, invalid-id fraction)
    let shapes: [(usize, usize, usize, usize, u64); 9] = [
        (1, 1, 4, 16, 0), (7, 2, 8, 16, 0), (64, 8, 64, 64, 0), (300, 4, 60, 32, 5), (1, 8, 128, 16, 0),
        (513, 2, 7, 128, 3), (2000, 8, 256, 64, 0), (3, 6, 1000, 16, 0), (100, 2, 33, 1, 0),
    ];
    for &(nt, topk, ne, bs, inv) in &shapes {
        for skew in [false, true] {
            k += 1;
            let numel = nt * topk;
            let em = { let em = numel + ne * (bs - 1); if numel < ne { (numel * bs).min(em) } else { em } };
            let nblocks = em.div_ceil(bs);
            let ids: Vec<i32> = (0..numel)
                .map(|_| {
                    let r = g.rng.next();
                    if inv > 0 && r % 16 < inv { (ne + (r >> 8) as usize % 5) as i32 } else if skew { ((r >> 8) % 3) as i32 } else { ((r >> 8) % ne as u64) as i32 }
                })
                .collect();
            let bufs = vec![as_bytes(&ids), g.rng.bytes(em * 4), g.rng.bytes(nblocks * 4), g.rng.bytes(4), g.rng.bytes((ne + 1) * 4)];
            let s = g.stream(k % 2);
            let call = move |rust: bool, b: &Side| unsafe {
                let f = if rust { ox::launch_moe_align } else { cref::launch_moe_align };
                f(b.p(0) as _, b.p(1) as _, b.p(2) as _, b.p(3) as _, b.p(4) as _, ne as i32, bs as i32, numel as i32, em as i32, s)
            };
            if !g.ref_deterministic("moe_align", bufs.iter().map(|b| { let mut b = b.clone(); b.extend(vec![0u8; GUARD]); b }).collect(), call) {
                nondet += 1;
            }
            // sorted_token_ids is only defined up to the order of tokens within one expert (atomics):
            // sort each expert's positions (grouped by the exactly-compared expert_ids).
            let canon = move |v: &mut Vec<Vec<u8>>| {
                let eids = i32s_of(&v[2][..nblocks * 4]);
                let mut sorted = i32s_of(&v[1][..em * 4]);
                let mut groups: std::collections::BTreeMap<i32, Vec<usize>> = Default::default();
                for p in 0..em {
                    groups.entry(eids[p / bs]).or_default().push(p);
                }
                for (_, ps) in groups {
                    let mut vals: Vec<i32> = ps.iter().map(|&p| sorted[p]).collect();
                    vals.sort();
                    for (i, &p) in ps.iter().enumerate() {
                        sorted[p] = vals[i];
                    }
                }
                v[1][..em * 4].copy_from_slice(&as_bytes(&sorted));
            };
            g.case_canon(&format!("moe_align tokens={nt} topk={topk} experts={ne} bs={bs} skew={skew}"), bufs, call, canon);
        }
    }
    println!("  moe_align: reference itself order-nondeterministic in {nondet} of {k} cases");
    for &(nt, ne, tk, cap) in &[(1i32, 4i32, 1i32, 1i32), (16, 8, 2, 3), (64, 16, 4, 20), (5, 3, 8, 2), (0, 4, 2, 1), (8, 4, 2, 0), (33, 7, 3, 5)] {
        k += 1;
        let n = (nt.max(0) * tk) as usize;
        let ids: Vec<u32> = (0..n).map(|_| (g.rng.next() % (ne as u64 + 1)) as u32).collect();
        let w = g.f32s(n);
        let m = g.rng.bytes(n * 4);
        let s = g.stream(k % 2);
        g.case(&format!("hunyuan_moe_capacity_mask tokens={nt} experts={ne} topk={tk} cap={cap}"), vec![as_bytes(&ids), w, m], |rust, b| unsafe {
            let f = if rust { ox::launch_hunyuan_moe_capacity_mask } else { cref::launch_hunyuan_moe_capacity_mask };
            f(b.p(0), b.p(1), b.p(2), nt, ne, tk, cap, s)
        });
    }
    g.family("moe_align");
}

fn moe_data(g: &mut G) {
    let mut k = 0;
    for &(ne, tl, topk) in &[(1usize, 1usize, 1usize), (8, 64, 2), (64, 2048, 8), (5, 3000, 6), (128, 16, 4), (33, 1000, 1)] {
        k += 1;
        let ids: Vec<i32> = (0..tl).map(|_| { let r = g.rng.next(); if r % 23 == 0 { -1 - (r >> 40) as i32 % 3 } else if r % 29 == 0 { ne as i32 } else { ((r >> 8) % ne as u64) as i32 } }).collect();
        let s = g.stream(k % 2);
        for gated in [false, true] {
            let (n, kk) = (1 + (g.rng.next() % 5000) as i32, 1 + (g.rng.next() % 5000) as i32);
            let bufs = vec![as_bytes(&ids), g.rng.bytes(ne * 12), g.rng.bytes(ne * 12), g.rng.bytes(ne * 4)];
            g.case(&format!("cutlass_moe_problem_sizes ne={ne} tl={tl} gated={gated}"), bufs, |rust, b| unsafe {
                let f = if rust { ox::launch_cutlass_moe_problem_sizes } else { cref::launch_cutlass_moe_problem_sizes };
                f(b.p(0) as _, b.p(1) as _, b.p(2) as _, b.p(3) as _, ne as i32, tl as i32, n, kk, gated, s)
            });
        }
        // expert offsets from per-expert counts (problem_sizes1 rows)
        let mut counts = vec![0i32; ne];
        for &e in &ids {
            if e >= 0 && (e as usize) < ne {
                counts[e as usize] += 1;
            }
        }
        let ps1: Vec<i32> = counts.iter().flat_map(|&c| [c, 7, 9]).collect();
        let bufs = vec![as_bytes(&ps1), g.rng.bytes((ne + 1) * 4), g.rng.bytes(ne * 4)];
        g.case(&format!("cutlass_moe_expert_offsets ne={ne}"), bufs, |rust, b| unsafe {
            let f = if rust { ox::launch_cutlass_moe_expert_offsets } else { cref::launch_cutlass_moe_expert_offsets };
            f(b.p(0) as _, b.p(1) as _, b.p(2) as _, ne as i32, s)
        });
        // arg sorts with the offsets the pipeline would produce
        let mut offs = vec![0i32; ne];
        let mut t = 0;
        for e in 0..ne {
            offs[e] = t;
            t += counts[e];
        }
        let total = t as usize;
        let bufs = vec![as_bytes(&ids), g.rng.bytes(tl * 4), g.rng.bytes(tl * 4), as_bytes(&offs)];
        let ids2 = ids.clone();
        let offs2 = offs.clone();
        // Positions within one expert are handed out by atomics: canonicalize by giving each expert's
        // rows (ascending i) its positions in ascending order, carrying input_permutation's value along.
        let canon = move |v: &mut Vec<Vec<u8>>| {
            let mut ip = i32s_of(&v[1][..tl * 4]);
            let mut op = i32s_of(&v[2][..tl * 4]);
            let (ip0, op0) = (ip.clone(), op.clone());
            for e in 0..ne {
                let rows: Vec<usize> = (0..tl).filter(|&i| ids2[i] == e as i32).collect();
                let mut pos: Vec<i32> = rows.iter().map(|&i| op0[i]).collect();
                pos.sort();
                for (r, &i) in rows.iter().enumerate() {
                    let old = op0[i];
                    let val = if old >= 0 && (old as usize) < tl { ip0[old as usize] } else { i32::MIN };
                    op[i] = pos[r];
                    if pos[r] >= 0 && (pos[r] as usize) < tl {
                        ip[pos[r] as usize] = val;
                    }
                }
                let _ = offs2[e];
            }
            v[1][..tl * 4].copy_from_slice(&as_bytes(&ip));
            v[2][..tl * 4].copy_from_slice(&as_bytes(&op));
        };
        g.case_canon(&format!("cutlass_moe_arg_sorts ne={ne} tl={tl} topk={topk}"), bufs, move |rust, b| unsafe {
            let f = if rust { ox::launch_cutlass_moe_arg_sorts } else { cref::launch_cutlass_moe_arg_sorts };
            f(b.p(0) as _, b.p(1) as _, b.p(2) as _, b.p(3) as _, ne as i32, tl as i32, topk as i32, s)
        }, canon);
        let _ = total;
        // gathers
        for &kw in &[1usize, 255, 256, 257, 2048] {
            let src_rows = 1 + tl / topk.max(1);
            let rows = tl;
            let src = g.vals(1, src_rows * kw);
            let map: Vec<i32> = (0..rows).map(|_| (g.rng.next() % src_rows as u64) as i32).collect();
            let dst = g.rng.bytes(rows * kw * 2);
            let (nr, kk) = (rows as i32, kw as i32);
            g.case(&format!("cutlass_moe_gather_rows_bf16 rows={rows} k={kw}"), vec![dst, src.clone(), as_bytes(&map)], |rust, b| unsafe {
                let f = if rust { ox::launch_cutlass_moe_gather_rows_bf16 } else { cref::launch_cutlass_moe_gather_rows_bf16 };
                f(b.p(0), b.p(1), b.p(2) as _, nr, kk, s)
            });
            let w = g.f32s(rows);
            let out = g.rng.bytes(rows * kw * 2);
            g.case(&format!("cutlass_moe_gather_weighted_bf16 rows={rows} k={kw}"), vec![out, src, as_bytes(&map), w], |rust, b| unsafe {
                let f = if rust { ox::launch_cutlass_moe_gather_weighted_bf16 } else { cref::launch_cutlass_moe_gather_weighted_bf16 };
                f(b.p(0), b.p(1), b.p(2) as _, b.p(3) as _, nr, kk, s)
            });
        }
        // group starts (pointer arithmetic only: the bases are never dereferenced)
        let eo: Vec<i32> = (0..=ne).map(|e| (e as i32) * 37 - 5).collect();
        let (n, kk) = (4096i64 + k as i64, 2880i64 - k as i64);
        let (ab, bb, db) = (0x7f00_1234_5000u64, 0x7e00_0000_0000u64, 0x10u64);
        let bufs = vec![as_bytes(&eo), g.rng.bytes(ne * 8), g.rng.bytes(ne * 8), g.rng.bytes(ne * 8), g.rng.bytes(ne * 8), g.rng.bytes(ne * 8), g.rng.bytes(ne * 8)];
        g.case(&format!("cutlass_moe_group_starts_bf16 ne={ne}"), bufs, |rust, b| unsafe {
            let f = if rust { ox::launch_cutlass_moe_group_starts_bf16 } else { cref::launch_cutlass_moe_group_starts_bf16 };
            f(b.p(0) as _, ab as _, bb as _, db as _, b.p(1) as _, b.p(2) as _, b.p(3) as _, b.p(4) as _, b.p(5) as _, b.p(6) as _, ne as i32, n, kk, s)
        });
    }
    g.family("moe_data");
}

fn ops(g: &mut G) {
    let mut k = 0usize;
    // ---- bitwise / leftshift (legacy default stream, no stream parameter)
    type BW = unsafe extern "C" fn(*const c_void, *const c_void, *mut c_void, u32);
    type LS = unsafe extern "C" fn(*const c_void, *mut c_void, u32, i32);
    let bws: [(&str, BW, BW, usize); 12] = [
        ("bitwise_and_u8", cref::bitwise_and_u8, ox::bitwise_and_u8, 1), ("bitwise_or_u8", cref::bitwise_or_u8, ox::bitwise_or_u8, 1),
        ("bitwise_xor_u8", cref::bitwise_xor_u8, ox::bitwise_xor_u8, 1), ("bitwise_and_u32", cref::bitwise_and_u32, ox::bitwise_and_u32, 4),
        ("bitwise_or_u32", cref::bitwise_or_u32, ox::bitwise_or_u32, 4), ("bitwise_xor_u32", cref::bitwise_xor_u32, ox::bitwise_xor_u32, 4),
        ("bitwise_and_i64", cref::bitwise_and_i64, ox::bitwise_and_i64, 8), ("bitwise_or_i64", cref::bitwise_or_i64, ox::bitwise_or_i64, 8),
        ("bitwise_xor_i64", cref::bitwise_xor_i64, ox::bitwise_xor_i64, 8), ("bitwise_and_i32", cref::bitwise_and_i32, ox::bitwise_and_i32, 4),
        ("bitwise_or_i32", cref::bitwise_or_i32, ox::bitwise_or_i32, 4), ("bitwise_xor_i32", cref::bitwise_xor_i32, ox::bitwise_xor_i32, 4),
    ];
    let lss: [(&str, LS, LS, usize); 4] = [
        ("leftshift_u8", cref::leftshift_u8, ox::leftshift_u8, 1), ("leftshift_u32", cref::leftshift_u32, ox::leftshift_u32, 4),
        ("leftshift_i64", cref::leftshift_i64, ox::leftshift_i64, 8), ("leftshift_i32", cref::leftshift_i32, ox::leftshift_i32, 4),
    ];
    for &n in &[1u32, 2, 3, 31, 33, 1000, 1024, 1025, 70000] {
        for &(name, cf, of, es) in &bws {
            let (a, b, o) = (g.rng.bytes(n as usize * es), g.rng.bytes(n as usize * es), g.rng.bytes(n as usize * es));
            g.case(&format!("{name} n={n}"), vec![a, b, o], |rust, b| unsafe { (if rust { of } else { cf })(b.p(0), b.p(1), b.p(2), n) });
        }
        for &(name, cf, of, es) in &lss {
            for &sh in &[0i32, 1, 5, 7, 8, 13, 31, 32, 33, 63, 64, 100, -1, -33] {
                if n > 1025 && sh % 3 != 0 { continue; }
                let (a, o) = (g.rng.bytes(n as usize * es), g.rng.bytes(n as usize * es));
                g.case(&format!("{name} n={n} k={sh}"), vec![a, o], |rust, b| unsafe { (if rust { of } else { cf })(b.p(0), b.p(1), n, sh) });
            }
        }
    }
    // ---- gptoss swiglu (scalar and vec4 paths) and interleaved
    type SW = unsafe extern "C" fn(*const c_void, *const c_void, *mut c_void, u32, f32, f32, *mut c_void);
    type SI = unsafe extern "C" fn(*const c_void, *mut c_void, u32, u32, f32, f32, *mut c_void);
    let sws: [(&str, SW, SW, usize); 3] = [
        ("gptoss_swiglu_f16", cref::gptoss_swiglu_f16, ox::gptoss_swiglu_f16, 0), ("gptoss_swiglu_bf16", cref::gptoss_swiglu_bf16, ox::gptoss_swiglu_bf16, 1),
        ("gptoss_swiglu_f32", cref::gptoss_swiglu_f32, ox::gptoss_swiglu_f32, 2),
    ];
    let sis: [(&str, SI, SI, usize); 3] = [
        ("gptoss_swiglu_interleaved_f16", cref::gptoss_swiglu_interleaved_f16, ox::gptoss_swiglu_interleaved_f16, 0),
        ("gptoss_swiglu_interleaved_bf16", cref::gptoss_swiglu_interleaved_bf16, ox::gptoss_swiglu_interleaved_bf16, 1),
        ("gptoss_swiglu_interleaved_f32", cref::gptoss_swiglu_interleaved_f32, ox::gptoss_swiglu_interleaved_f32, 2),
    ];
    let params: [(f32, f32); 6] = [(1.702, 7.0), (1.0, 1e30), (0.5, 0.25), (2.0, -3.0), (1.702, f32::INFINITY), (-1.5, 7.0)];
    for &n in &[1u32, 3, 4, 8, 1000, 1001, 4096, 70001] {
        for (pi, &(alpha, limit)) in params.iter().enumerate() {
            if n > 4096 && pi > 1 { continue; }
            for &(name, cf, of, ty) in &sws {
                k += 1;
                let es = if ty == 2 { 4 } else { 2 };
                let (a, b, o) = (g.vals(ty, n as usize), g.vals(ty, n as usize), g.rng.bytes(n as usize * es));
                let s = g.stream(k % 2);
                g.case(&format!("{name} n={n} alpha={alpha} limit={limit}"), vec![a, b, o], |rust, b| unsafe { (if rust { of } else { cf })(b.p(0), b.p(1), b.p(2), n, alpha, limit, s) });
            }
        }
    }
    for &(n, isz) in &[(1u32, 1u32), (3, 7), (8, 2880), (5, 1000), (64, 33)] {
        for &(alpha, limit) in &params[..3] {
            for &(name, cf, of, ty) in &sis {
                k += 1;
                let es = if ty == 2 { 4 } else { 2 };
                let tot = (n * isz) as usize;
                let (a, o) = (g.vals(ty, tot * 2), g.rng.bytes(tot * es));
                let s = g.stream(k % 2);
                g.case(&format!("{name} n={n} isz={isz} alpha={alpha} limit={limit}"), vec![a, o], |rust, b| unsafe { (if rust { of } else { cf })(b.p(0), b.p(1), n, isz, alpha, limit, s) });
            }
        }
    }
    // ---- softmax with sinks
    type SM = unsafe extern "C" fn(*const c_void, *const c_void, *const c_void, *mut c_void, i32, i32, i32, i32, f32, *mut c_void);
    let sms: [(&str, SM, SM, usize); 3] = [
        ("softmax_with_sinks_f16", cref::softmax_with_sinks_f16, ox::softmax_with_sinks_f16, 0),
        ("softmax_with_sinks_bf16", cref::softmax_with_sinks_bf16, ox::softmax_with_sinks_bf16, 1),
        ("softmax_with_sinks_f32", cref::softmax_with_sinks_f32, ox::softmax_with_sinks_f32, 2),
    ];
    for &(b, h, q, kl) in &[(1i32, 1i32, 1i32, 1i32), (1, 2, 3, 63), (2, 3, 2, 64), (1, 4, 5, 65), (1, 2, 2, 128), (2, 2, 1, 200), (1, 1, 3, 256),
                             (1, 2, 1, 511), (1, 1, 2, 512), (1, 2, 1, 700), (1, 1, 1, 1024), (1, 1, 2, 3000), (3, 2, 4, 33)] {
        for masked in [false, true] {
            for &(name, cf, of, ty) in &sms {
                k += 1;
                let es = if ty == 2 { 4 } else { 2 };
                let rows = (b * h * q) as usize;
                // logits: moderate values plus the edge mix; softmax cares about spread
                let lg = if k % 3 == 0 { g.vals(ty, rows * kl as usize) } else {
                    let v: Vec<f32> = (0..rows * kl as usize).map(|_| ((g.rng.next() >> 11) as f64 / (1u64 << 53) as f64 * 20.0 - 10.0) as f32).collect();
                    to_ty(ty, &v)
                };
                let sinks = g.vals(ty, h as usize);
                let mv: Vec<f32> = (0..(b * q * kl) as usize).map(|_| match g.rng.next() % 5 { 0 => f32::NEG_INFINITY, 1 => -1.5, _ => 0.0 }).collect();
                let mask = to_ty(ty, &mv);
                let o = g.rng.bytes(rows * kl as usize * es);
                let s = g.stream(k % 2);
                g.case(&format!("{name} b={b} h={h} q={q} k={kl} mask={masked}"), vec![lg, sinks, mask, o], |rust, bf| unsafe {
                    let m = if masked { bf.p(2) as *const c_void } else { std::ptr::null() };
                    (if rust { of } else { cf })(bf.p(0), bf.p(1), m, bf.p(3), b, h, q, kl, 1.0, s)
                });
            }
        }
    }
    // ---- fused GLU: every activation id (incl. out-of-range defaults), both paths
    type FG = unsafe extern "C" fn(*const c_void, *const c_void, *mut c_void, u32, i32, *mut c_void);
    let fgs: [(&str, FG, FG, usize); 3] = [
        ("fused_glu_f16", cref::fused_glu_f16, ox::fused_glu_f16, 0), ("fused_glu_bf16", cref::fused_glu_bf16, ox::fused_glu_bf16, 1),
        ("fused_glu_f32", cref::fused_glu_f32, ox::fused_glu_f32, 2),
    ];
    for &n in &[1u32, 4, 7, 1000, 4097, 65536] {
        for &act in &[0i32, 1, 2, 3, 4, -1, 99] {
            if n == 65536 && act > 3 { continue; }
            for &(name, cf, of, ty) in &fgs {
                k += 1;
                let es = if ty == 2 { 4 } else { 2 };
                // activation inputs: the edge mix, or a dense sweep over the erf/tanh ranges
                let a = if k % 2 == 0 { g.vals(ty, n as usize) } else {
                    let v: Vec<f32> = (0..n as usize).map(|i| (i as f32 - n as f32 / 2.0) * (40.0 / n as f32) + ((g.rng.next() % 1000) as f32) * 1e-4).collect();
                    to_ty(ty, &v)
                };
                let (b, o) = (g.vals(ty, n as usize), g.rng.bytes(n as usize * es));
                let s = g.stream(k % 2);
                g.case(&format!("{name} n={n} act={act}"), vec![a, b, o], |rust, bf| unsafe { (if rust { of } else { cf })(bf.p(0), bf.p(1), bf.p(2), n, act, s) });
            }
        }
    }
    // a dense erf sweep: every f32 exponent / many mantissas through GELU_ERF
    {
        let v: Vec<f32> = (0..(1u32 << 16)).map(|i| f32::from_bits(i.wrapping_mul(0x9E37_79B9) ^ (i << 16))).collect();
        let n = v.len() as u32;
        let (a, b, o) = (as_bytes(&v), as_bytes(&vec![1.0f32; v.len()]), g.rng.bytes(v.len() * 4));
        g.case("fused_glu_f32 erf sweep", vec![a, b, o], |rust, bf| unsafe {
            (if rust { ox::fused_glu_f32 } else { cref::fused_glu_f32 })(bf.p(0), bf.p(1), bf.p(2), n - 1, 3, std::ptr::null_mut())
        });
    }
    // ---- softcap
    type SC = unsafe extern "C" fn(*const c_void, *mut c_void, u32, f32, *mut c_void);
    let scs: [(&str, SC, SC, usize); 3] = [
        ("softcap_f32", cref::softcap_f32, ox::softcap_f32, 2), ("softcap_f16_to_f32", cref::softcap_f16_to_f32, ox::softcap_f16_to_f32, 0),
        ("softcap_bf16_to_f32", cref::softcap_bf16_to_f32, ox::softcap_bf16_to_f32, 1),
    ];
    for &n in &[1u32, 255, 256, 257, 5000] {
        for &cap in &[30.0f32, 50.0, 0.5, -3.0, 0.0, f32::INFINITY, 1e-40] {
            for &(name, cf, of, ty) in &scs {
                k += 1;
                let (a, o) = (g.vals(ty, n as usize), g.rng.bytes(n as usize * 4));
                let s = g.stream(k % 2);
                g.case(&format!("{name} n={n} cap={cap}"), vec![a, o], |rust, bf| unsafe { (if rust { of } else { cf })(bf.p(0), bf.p(1), n, cap, s) });
            }
        }
    }
    // ---- count_nonzero / nonzero
    type CN = unsafe extern "C" fn(*const c_void, u32, *mut c_void) -> u32;
    type NZ = unsafe extern "C" fn(*const c_void, u32, u32, *const c_void, u32, *mut c_void, *mut c_void);
    let nzs: [(&str, CN, CN, NZ, NZ, usize); 7] = [
        ("f32", cref::count_nonzero_f32, ox::count_nonzero_f32, cref::nonzero_f32, ox::nonzero_f32, 4),
        ("f64", cref::count_nonzero_f64, ox::count_nonzero_f64, cref::nonzero_f64, ox::nonzero_f64, 8),
        ("u8", cref::count_nonzero_u8, ox::count_nonzero_u8, cref::nonzero_u8, ox::nonzero_u8, 1),
        ("u32", cref::count_nonzero_u32, ox::count_nonzero_u32, cref::nonzero_u32, ox::nonzero_u32, 4),
        ("i16", cref::count_nonzero_i16, ox::count_nonzero_i16, cref::nonzero_i16, ox::nonzero_i16, 2),
        ("i32", cref::count_nonzero_i32, ox::count_nonzero_i32, cref::nonzero_i32, ox::nonzero_i32, 4),
        ("i64", cref::count_nonzero_i64, ox::count_nonzero_i64, cref::nonzero_i64, ox::nonzero_i64, 8),
    ];
    let shapes: [&[u32]; 7] = [&[1], &[7], &[3, 5], &[2048], &[2049, 3], &[4, 1, 7, 9], &[100, 700]];
    for (si, dims) in shapes.iter().enumerate() {
        let n: u32 = dims.iter().product();
        for &(name, ccf, cof, ncf, nof, es) in &nzs {
            k += 1;
            let density = [2u64, 8, 50, 100][si % 4];
            let mut data = vec![0u8; n as usize * es];
            for i in 0..n as usize {
                let r = g.rng.next();
                if r % 100 < density {
                    let mut v = g.rng.bytes(es);
                    if es == 4 && name == "f32" && r % 7 == 0 {
                        v = [0x0000_0001u32, 0x8000_0003, 0x7fc0_0000, 0x8000_0000, 0x007f_ffff][(r >> 8) as usize % 5].to_le_bytes().to_vec();
                    }
                    if es == 8 && r % 7 == 0 {
                        v = [1u64, 0x8000_0000_0000_0000, 0x7ff8_0000_0000_0000][(r >> 8) as usize % 3].to_le_bytes().to_vec();
                    }
                    data[i * es..(i + 1) * es].copy_from_slice(&v);
                }
            }
            // count (host result compared, plus every buffer)
            let res = std::cell::Cell::new([0u32; 2]);
            let s = g.stream(k % 2);
            g.case(&format!("count_nonzero_{name} n={n}"), vec![data.clone()], |rust, b| unsafe {
                let mut r = res.get();
                r[rust as usize] = (if rust { cof } else { ccf })(b.p(0), n, s);
                res.set(r);
            });
            let [rc, ro] = res.get();
            g.scalar(&format!("count_nonzero_{name} n={n} result"), rc as u64, ro as u64);
            if rc == 0 {
                continue; // nonzero with num_nonzero == 0 exits the process (checked in the exit cases)
            }
            let nd = dims.len() as u32;
            let out = g.rng.bytes(rc as usize * nd as usize * 4);
            g.case(&format!("nonzero_{name} dims={dims:?} nz={rc}"), vec![data, as_bytes(dims), out], |rust, b| unsafe {
                (if rust { nof } else { ncf })(b.p(0), n, rc, b.p(1), nd, b.p(2), s)
            });
        }
    }
    // f16/bf16 counts are host stubs returning 0
    for rust in [false, true] {
        let _ = rust;
    }
    let r = unsafe { [cref::count_nonzero_f16(std::ptr::null(), 10, std::ptr::null_mut()), ox::count_nonzero_f16(std::ptr::null(), 10, std::ptr::null_mut()),
                     cref::count_nonzero_bf16(std::ptr::null(), 10, std::ptr::null_mut()), ox::count_nonzero_bf16(std::ptr::null(), 10, std::ptr::null_mut())] };
    g.scalar("count_nonzero_f16 stub", r[0] as u64, r[1] as u64);
    g.scalar("count_nonzero_bf16 stub", r[2] as u64, r[3] as u64);
    g.family("ops");
}

/// f32 values -> dtype `t` bytes (0 f16 truncating, 1 bf16 truncating, 2 f32).
pub fn to_ty(t: usize, v: &[f32]) -> Vec<u8> {
    match t {
        2 => as_bytes(v),
        1 => as_bytes(&v.iter().map(|x| (x.to_bits() >> 16) as u16).collect::<Vec<u16>>()),
        _ => as_bytes(&v.iter().map(|&x| f32_to_f16(x)).collect::<Vec<u16>>()),
    }
}

fn gemv(g: &mut G) {
    type GV = unsafe extern "C" fn(*const c_void, *const c_void, *const c_void, *mut c_void, i32, i32, i32, bool, *mut c_void);
    let fns: [(&str, GV, GV, usize); 3] = unsafe {
        [
            ("bf16", std::mem::transmute::<*const (), GV>(cref::launch_gemv_bf16 as *const ()), std::mem::transmute::<*const (), GV>(ox::launch_gemv_bf16 as *const ()), 1),
            ("f16", std::mem::transmute::<*const (), GV>(cref::launch_gemv_f16 as *const ()), std::mem::transmute::<*const (), GV>(ox::launch_gemv_f16 as *const ()), 0),
            ("f32", std::mem::transmute::<*const (), GV>(cref::launch_gemv_f32 as *const ()), std::mem::transmute::<*const (), GV>(ox::launch_gemv_f32 as *const ()), 2),
        ]
    };
    let mut k = 0;
    // K spans every block-size choice (K/2 <= 32, 64, 128, else 256); even K except the M=1 odd-K
    // cases (odd K with M > 1 or batch > 1 is a misaligned vector load in the reference).
    for &(m, kk) in &[(1i32, 2i32), (5, 64), (3, 66), (7, 128), (4, 130), (9, 256), (2, 258), (33, 1024), (17, 4096), (3, 6000), (1, 1), (1, 63), (1, 257), (1, 1001)] {
        for batch in 0..=9 {
            if kk % 2 == 1 && batch > 1 && batch != 9 { continue; }
            for hb in [false, true] {
                if kk > 1024 && batch % 3 != 1 { continue; }
                for &(name, cf, of, ty) in &fns {
                    k += 1;
                    let es = if ty == 2 { 4 } else { 2 };
                    let nb = if (1..=8).contains(&batch) { batch } else { 1 } as usize;
                    // weights/activations: moderate values (the edge mix every third case)
                    let mk = |g: &mut G, n: usize| -> Vec<u8> {
                        if k % 3 == 0 { g.vals(ty, n) } else {
                            let v: Vec<f32> = (0..n).map(|_| ((g.rng.next() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32).collect();
                            to_ty(ty, &v)
                        }
                    };
                    let a = mk(g, (m * kk) as usize);
                    let x = mk(g, nb * kk as usize);
                    let bias = g.vals(ty, m as usize);
                    let y = g.rng.bytes(nb * m as usize * es);
                    let s = g.stream(k % 2);
                    if kk % 2 == 1 && batch == 9 && m > 1 { continue; }
                    g.case(&format!("gemv_{name} m={m} k={kk} batch={batch} bias={hb}"), vec![a, x, bias, y], |rust, b| unsafe {
                        (if rust { of } else { cf })(b.p(0), b.p(1), b.p(2), b.p(3), m, kk, batch, hb, s)
                    });
                }
            }
        }
    }
    g.family("gemv");
}

/// A GGUF weight format: name, block bytes, values per block, f16 scale-field offsets.
pub struct Fmt {
    pub name: &'static str,
    pub bs: usize,
    pub qk: usize,
    pub f16s: &'static [usize],
}
pub const FMTS: [Fmt; 11] = [
    Fmt { name: "q4_0", bs: 18, qk: 32, f16s: &[0] },
    Fmt { name: "q4_1", bs: 20, qk: 32, f16s: &[0, 2] },
    Fmt { name: "q5_0", bs: 22, qk: 32, f16s: &[0] },
    Fmt { name: "q5_1", bs: 24, qk: 32, f16s: &[0, 2] },
    Fmt { name: "q8_0", bs: 34, qk: 32, f16s: &[0] },
    Fmt { name: "q2k", bs: 84, qk: 256, f16s: &[80, 82] },
    Fmt { name: "q3k", bs: 110, qk: 256, f16s: &[108] },
    Fmt { name: "q4k", bs: 144, qk: 256, f16s: &[0, 2] },
    Fmt { name: "q5k", bs: 176, qk: 256, f16s: &[0, 2] },
    Fmt { name: "q6k", bs: 210, qk: 256, f16s: &[208] },
    Fmt { name: "q8_1", bs: 36, qk: 32, f16s: &[0, 2] },
];

type IM = unsafe extern "C" fn(*const c_void, *const c_void, *const u32, *mut f32, i32, i32, i32, i32, i32, i32, *mut c_void);
type FG = unsafe extern "C" fn(*const c_void, *const c_void, *const c_void, *const u32, *mut f32, i32, i32, i32, i32, i32, i32, *mut c_void);
type DA = unsafe extern "C" fn(*const c_void, *const c_void, *const u32, *const f32, *mut f32, i32, i32, i32, i32, i32, *mut c_void);

fn moe_fns(fi: usize) -> (IM, IM, FG, FG, DA, DA) {
    let t: [(IM, IM, FG, FG, DA, DA); 11] = [
        (cref::launch_indexed_moe_forward_q4_0_q8_1 as IM, ox::launch_indexed_moe_forward_q4_0_q8_1 as IM, cref::launch_moe_gemv_fused_gate_up_q4_0_q8_1 as FG, ox::launch_moe_gemv_fused_gate_up_q4_0_q8_1 as FG, cref::launch_moe_gemv_down_aggregate_q4_0_q8_1 as DA, ox::launch_moe_gemv_down_aggregate_q4_0_q8_1 as DA),
        (cref::launch_indexed_moe_forward_q4_1_q8_1 as IM, ox::launch_indexed_moe_forward_q4_1_q8_1 as IM, cref::launch_moe_gemv_fused_gate_up_q4_1_q8_1 as FG, ox::launch_moe_gemv_fused_gate_up_q4_1_q8_1 as FG, cref::launch_moe_gemv_down_aggregate_q4_1_q8_1 as DA, ox::launch_moe_gemv_down_aggregate_q4_1_q8_1 as DA),
        (cref::launch_indexed_moe_forward_q5_0_q8_1 as IM, ox::launch_indexed_moe_forward_q5_0_q8_1 as IM, cref::launch_moe_gemv_fused_gate_up_q5_0_q8_1 as FG, ox::launch_moe_gemv_fused_gate_up_q5_0_q8_1 as FG, cref::launch_moe_gemv_down_aggregate_q5_0_q8_1 as DA, ox::launch_moe_gemv_down_aggregate_q5_0_q8_1 as DA),
        (cref::launch_indexed_moe_forward_q5_1_q8_1 as IM, ox::launch_indexed_moe_forward_q5_1_q8_1 as IM, cref::launch_moe_gemv_fused_gate_up_q5_1_q8_1 as FG, ox::launch_moe_gemv_fused_gate_up_q5_1_q8_1 as FG, cref::launch_moe_gemv_down_aggregate_q5_1_q8_1 as DA, ox::launch_moe_gemv_down_aggregate_q5_1_q8_1 as DA),
        (cref::launch_indexed_moe_forward_q8_0_q8_1 as IM, ox::launch_indexed_moe_forward_q8_0_q8_1 as IM, cref::launch_moe_gemv_fused_gate_up_q8_0_q8_1 as FG, ox::launch_moe_gemv_fused_gate_up_q8_0_q8_1 as FG, cref::launch_moe_gemv_down_aggregate_q8_0_q8_1 as DA, ox::launch_moe_gemv_down_aggregate_q8_0_q8_1 as DA),
        (cref::launch_indexed_moe_forward_q2k_q8_1 as IM, ox::launch_indexed_moe_forward_q2k_q8_1 as IM, cref::launch_moe_gemv_fused_gate_up_q2k_q8_1 as FG, ox::launch_moe_gemv_fused_gate_up_q2k_q8_1 as FG, cref::launch_moe_gemv_down_aggregate_q2k_q8_1 as DA, ox::launch_moe_gemv_down_aggregate_q2k_q8_1 as DA),
        (cref::launch_indexed_moe_forward_q3k_q8_1 as IM, ox::launch_indexed_moe_forward_q3k_q8_1 as IM, cref::launch_moe_gemv_fused_gate_up_q3k_q8_1 as FG, ox::launch_moe_gemv_fused_gate_up_q3k_q8_1 as FG, cref::launch_moe_gemv_down_aggregate_q3k_q8_1 as DA, ox::launch_moe_gemv_down_aggregate_q3k_q8_1 as DA),
        (cref::launch_indexed_moe_forward_q4k_q8_1 as IM, ox::launch_indexed_moe_forward_q4k_q8_1 as IM, cref::launch_moe_gemv_fused_gate_up_q4k_q8_1 as FG, ox::launch_moe_gemv_fused_gate_up_q4k_q8_1 as FG, cref::launch_moe_gemv_down_aggregate_q4k_q8_1 as DA, ox::launch_moe_gemv_down_aggregate_q4k_q8_1 as DA),
        (cref::launch_indexed_moe_forward_q5k_q8_1 as IM, ox::launch_indexed_moe_forward_q5k_q8_1 as IM, cref::launch_moe_gemv_fused_gate_up_q5k_q8_1 as FG, ox::launch_moe_gemv_fused_gate_up_q5k_q8_1 as FG, cref::launch_moe_gemv_down_aggregate_q5k_q8_1 as DA, ox::launch_moe_gemv_down_aggregate_q5k_q8_1 as DA),
        (cref::launch_indexed_moe_forward_q6k_q8_1 as IM, ox::launch_indexed_moe_forward_q6k_q8_1 as IM, cref::launch_moe_gemv_fused_gate_up_q6k_q8_1 as FG, ox::launch_moe_gemv_fused_gate_up_q6k_q8_1 as FG, cref::launch_moe_gemv_down_aggregate_q6k_q8_1 as DA, ox::launch_moe_gemv_down_aggregate_q6k_q8_1 as DA),
        (cref::launch_indexed_moe_forward_q8_1_q8_1 as IM, ox::launch_indexed_moe_forward_q8_1_q8_1 as IM, cref::launch_moe_gemv_fused_gate_up_q8_1_q8_1 as FG, ox::launch_moe_gemv_fused_gate_up_q8_1_q8_1 as FG, cref::launch_moe_gemv_down_aggregate_q8_1_q8_1 as DA, ox::launch_moe_gemv_down_aggregate_q8_1_q8_1 as DA),
    ];
    t[fi]
}

impl G {
    /// An f16 scale: mostly normal magnitudes 2^-12..2^3, 1 in 16 a special value or raw bits.
    pub fn f16_scale(&mut self) -> u16 {
        let r = self.rng.next();
        match if r % 16 == 0 { (r >> 40) % 2 } else { 2 } {
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
    pub fn blocks(&mut self, f: &Fmt, n: usize, pad: usize) -> Vec<u8> {
        let mut b = self.rng.bytes(n * f.bs + pad);
        for i in 0..n {
            for &o in f.f16s {
                let v = self.f16_scale();
                b[i * f.bs + o..i * f.bs + o + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
        b
    }
    /// `blocks` with every f16 scale field a normal magnitude 2^-12..2^3 (no inf / nan / subnormal).
    pub fn sane_blocks(&mut self, f: &Fmt, n: usize, pad: usize) -> Vec<u8> {
        let mut b = self.rng.bytes(n * f.bs + pad);
        for i in 0..n {
            for &o in f.f16s {
                let r = self.rng.next();
                let v = ((((r >> 8) & 1) as u16) << 15) | ((3 + ((r >> 9) % 16) as u16) << 10) | ((r >> 16) & 0x3ff) as u16;
                b[i * f.bs + o..i * f.bs + o + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
        b
    }
    /// `n` random Q8_1 blocks (36 bytes): random int8 quants, `ds` from `f16_scale`.
    pub fn q8_1(&mut self, n: usize) -> Vec<u8> {
        let f = &FMTS[10];
        self.blocks(f, n, 0)
    }
}

/// `add.rn.ftz.f32` on the host: subnormal inputs and results become zero of the same sign.
fn ftz_add(a: f32, b: f32) -> f32 {
    let ftz = |x: f32| if x.is_subnormal() { 0f32.copysign(x) } else { x };
    ftz(ftz(a) + ftz(b))
}

fn indexed_moe(g: &mut G) {
    let mut k = 0usize;
    // ---- quantizers
    type Q = unsafe extern "C" fn(*const c_void, *mut c_void, i32, i32, i32, *mut c_void);
    for &(kx, kxp) in &[(512i32, 512i32), (1000, 1024), (1, 512), (511, 512), (2048, 2560), (96, 96), (70, 96), (3000, 3072), (0, 256)] {
        for &rows in &[1i32, 2, 5, 13] {
            for style in 0..4 {
                if style > 0 && rows > 2 { continue; }
                for t in 0..3usize {
                    k += 1;
                    let n = (rows * kx.max(1)) as usize;
                    let v: Vec<f32> = match style {
                        1 => vec![0.0; n],
                        2 => (0..n).map(|i| if i % 32 == 0 { 127.0 } else { ((i % 64) as f32 - 32.0) * 0.5 }).collect(),
                        3 => (0..n).map(|i| (i as f32 - 300.0) * 1e-38).collect(),
                        _ => g.rng.f32s(n),
                    };
                    let x = if style == 0 { g.vals(t, n) } else { to_ty(t, &v) };
                    let y = g.rng.bytes((rows * kxp) as usize / 32 * 36);
                    let s = g.stream(k % 2);
                    let nbx = (kxp + 255) / 256 + (k % 2) as i32; // f32 launcher: caller-chosen num_blocks_x
                    g.case(&format!("quantize_q8_1[{t}] kx={kx} kxp={kxp} rows={rows} style={style}"), vec![x, y], |rust, b| unsafe {
                        match t {
                            2 => (if rust { ox::launch_quantize_q8_1 } else { cref::launch_quantize_q8_1 })(b.p(0) as _, b.p(1), kx, kxp, nbx, rows, s),
                            1 => (if rust { ox::launch_quantize_q8_1_bf16 as Q } else { cref::launch_quantize_q8_1_bf16 as Q })(b.p(0), b.p(1), kx, kxp, rows, s),
                            _ => (if rust { ox::launch_quantize_q8_1_f16 as Q } else { cref::launch_quantize_q8_1_f16 as Q })(b.p(0), b.p(1), kx, kxp, rows, s),
                        }
                    });
                }
            }
        }
    }
    // ---- expert GEMVs
    for fi in 0..FMTS.len() {
        let f = &FMTS[fi];
        let (imc, imo, fgc, fgo, dac, dao) = moe_fns(fi);
        let qk = f.qk as i32;
        // (n, k, batch, topk, experts): odd n, k not a multiple of qk (ceil blocks), decode and prefill
        let shapes: [(i32, i32, i32, i32, usize); 5] = [
            (7, qk * 3, 1, 1, 2),
            (33, qk * 8, 2, 3, 5),
            (130, if qk == 32 { 2048 } else { 2304 }, 1, 8, 8),
            (17, qk * 2 + qk / 2 + 3, 3, 2, 4),
            (64, if qk == 32 { 4096 } else { 4608 }, 4, 2, 3),
        ];
        for (si, &(n, kk, batch, topk, ne)) in shapes.iter().enumerate() {
            k += 1;
            let bpr = ((kk as usize) + f.qk - 1) / f.qk;
            let kp = ((kk + 511) / 512 * 512).max((bpr * f.qk) as i32);
            let xblocks_row = kp as usize / 32;
            let tasks = (batch * topk) as usize;
            let w = g.blocks(f, ne * n as usize * bpr, 64);
            let w2 = g.blocks(f, ne * n as usize * bpr, 64);
            let ids: Vec<u32> = (0..tasks).map(|_| (g.rng.next() % ne as u64) as u32).collect();
            let s = g.stream(k % 2);
            for d1 in [1i32, 0] {
                let in_rows = if d1 == 1 { batch as usize } else { tasks };
                let x = g.q8_1(in_rows * xblocks_row);
                let out = g.rng.bytes(tasks * n as usize * 4);
                g.case(&format!("indexed_moe_forward_{}_q8_1 n={n} k={kk} batch={batch} topk={topk} d1={d1}", f.name), vec![w.clone(), x, as_bytes(&ids), out], |rust, b| unsafe {
                    (if rust { imo } else { imc })(b.p(0), b.p(1), b.p(2) as _, b.p(3) as _, n, kk, batch, topk, kp, d1, s)
                });
            }
            for act in [0i32, 1, 5] {
                if si > 2 && act == 5 { continue; }
                let x = g.q8_1(batch as usize * xblocks_row);
                let out = g.rng.bytes(tasks * n as usize * 4);
                g.case(&format!("moe_gemv_fused_gate_up_{}_q8_1 n={n} k={kk} batch={batch} topk={topk} act={act}", f.name), vec![w.clone(), w2.clone(), x, as_bytes(&ids), out], |rust, b| unsafe {
                    (if rust { fgo } else { fgc })(b.p(0), b.p(1), b.p(2), b.p(3) as _, b.p(4) as _, n, kk, batch, topk, kp, act, s)
                });
            }
            // down + aggregate: float atomics across the top-k slots of a token are order-dependent,
            // so every call has exactly one non-zero routing weight per token (rotating through the
            // slots); outputs start from normal values.
            for rot in 0..topk.min(3) {
                let x = g.q8_1(tasks * xblocks_row);
                let tw: Vec<f32> = (0..tasks).map(|t| if (t as i32 % topk) == rot { [0.5f32, -1.25, 3.0, 1e-3][t % 4] } else { 0.0 }).collect();
                let init: Vec<f32> = (0..batch as usize * n as usize).map(|i| ((i % 13) as f32 - 6.5) * 0.37).collect();
                g.case(&format!("moe_gemv_down_aggregate_{}_q8_1 n={n} k={kk} batch={batch} topk={topk} rot={rot}", f.name), vec![w.clone(), x, as_bytes(&ids), as_bytes(&tw), as_bytes(&init)], |rust, b| unsafe {
                    (if rust { dao } else { dac })(b.p(0), b.p(1), b.p(2) as _, b.p(3) as _, b.p(4) as _, n, kk, batch, topk, kp, s)
                });
            }
            // Every slot weighted (redcell): the oxide sum must be the slot-order fold of the one-slot
            // results, bit for bit (.ftz adds), and the same on a second run. Finite scales only, so
            // a zero weight never meets an inf/nan dot product.
            {
                let wn = g.sane_blocks(f, ne * n as usize * bpr, 64);
                let xq = g.sane_blocks(&FMTS[10], tasks * xblocks_row, 0);
                let tw_all: Vec<f32> = (0..tasks).map(|t| [0.5f32, -1.25, 3.0, 1e-3, 0.37, -0.0625, 2.5, 0.8][t % 8]).collect();
                let init: Vec<f32> = (0..batch as usize * n as usize).map(|i| ((i % 13) as f32 - 6.5) * 0.37).collect();
                let label = format!("moe_gemv_down_aggregate_{}_q8_1 n={n} k={kk} batch={batch} topk={topk} slot-order", f.name);
                let run = |g: &mut G, tw: &[f32], init: &[f32]| -> Vec<f32> {
                    let o = g.rust_only(&label, &[wn.clone(), xq.clone(), as_bytes(&ids), as_bytes(tw), as_bytes(init)], |b| unsafe {
                        dao(b.p(0), b.p(1), b.p(2) as _, b.p(3) as _, b.p(4) as _, n, kk, batch, topk, kp, s)
                    });
                    o[4].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
                };
                let full = run(&mut *g, &tw_all, &init);
                let again = run(&mut *g, &tw_all, &init);
                let zeros = vec![0f32; init.len()];
                let mut want = init.clone();
                for slot in 0..topk {
                    let tws: Vec<f32> = (0..tasks).map(|t| if t as i32 % topk == slot { tw_all[t] } else { 0.0 }).collect();
                    let p = run(&mut *g, &tws, &zeros);
                    for (a, b) in want.iter_mut().zip(&p) {
                        *a = ftz_add(*a, *b);
                    }
                }
                let same = |a: f32, b: f32| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()) || (a == 0.0 && b == 0.0);
                let bad = full.iter().zip(&want).filter(|(a, b)| !same(**a, **b)).count();
                if bad > 0 {
                    eprintln!("{label}: {bad} of {} outputs differ from the slot-order fold", full.len());
                }
                g.check(&label, bad == 0);
                g.check(&format!("{label} rerun"), full.iter().zip(&again).all(|(a, b)| a.to_bits() == b.to_bits()));
            }
        }
    }
    g.family("indexed_moe");
}

type GG = unsafe extern "C" fn(*const c_void, *const c_void, *const i32, *const i32, *const f32, *mut f32, i32, i32, i32, i32, i32, i32, *mut c_void);

/// Launcher pair for FMTS[fi] (moe_grouped has every format).
fn grouped_fns(fi: usize) -> (GG, GG) {
    // order of FMTS: q4_0 q4_1 q5_0 q5_1 q8_0 q2k q3k q4k q5k q6k q8_1
    let t: [GG; 22] = [
        cref::launch_moe_grouped_gemm_q8_0 as GG, ox::launch_moe_grouped_gemm_q8_0 as GG,
        cref::launch_moe_grouped_gemm_q4_0 as GG, ox::launch_moe_grouped_gemm_q4_0 as GG,
        cref::launch_moe_grouped_gemm_q4_1 as GG, ox::launch_moe_grouped_gemm_q4_1 as GG,
        cref::launch_moe_grouped_gemm_q5_0 as GG, ox::launch_moe_grouped_gemm_q5_0 as GG,
        cref::launch_moe_grouped_gemm_q5_1 as GG, ox::launch_moe_grouped_gemm_q5_1 as GG,
        cref::launch_moe_grouped_gemm_q8_1 as GG, ox::launch_moe_grouped_gemm_q8_1 as GG,
        cref::launch_moe_grouped_gemm_q2k as GG, ox::launch_moe_grouped_gemm_q2k as GG,
        cref::launch_moe_grouped_gemm_q3k as GG, ox::launch_moe_grouped_gemm_q3k as GG,
        cref::launch_moe_grouped_gemm_q4k as GG, ox::launch_moe_grouped_gemm_q4k as GG,
        cref::launch_moe_grouped_gemm_q5k as GG, ox::launch_moe_grouped_gemm_q5k as GG,
        cref::launch_moe_grouped_gemm_q6k as GG, ox::launch_moe_grouped_gemm_q6k as GG,
    ];
    let order = ["q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "q8_1", "q2k", "q3k", "q4k", "q5k", "q6k"];
    let i = order.iter().position(|&n| n == FMTS[fi].name).unwrap();
    (t[2 * i], t[2 * i + 1])
}

fn moe_grouped(g: &mut G) {
    let mut k = 0usize;
    // ---- dispatch (positions within an expert come from atomics: canonicalized)
    for &(tokens, topk, ne) in &[(1usize, 1usize, 1usize), (7, 2, 4), (64, 8, 32), (300, 4, 60), (1000, 2, 7), (5, 6, 128)] {
        for with_src in [true, false] {
            k += 1;
            let total = tokens * topk;
            let ids: Vec<i32> = (0..total).map(|_| (g.rng.next() % ne as u64) as i32).collect();
            let bufs = vec![as_bytes(&ids), g.rng.bytes((ne + 1) * 4), g.rng.bytes(total * 4), g.rng.bytes(total * 4), g.rng.bytes(ne * 4), g.rng.bytes(ne * 4)];
            let s = g.stream(k % 2);
            let canon = move |v: &mut Vec<Vec<u8>>| {
                let b = i32s_of(&v[1][..(ne + 1) * 4]);
                let mut st = i32s_of(&v[2][..total * 4]);
                let mut ss = i32s_of(&v[3][..total * 4]);
                for e in 0..ne {
                    let (lo, hi) = (b[e].clamp(0, total as i32) as usize, b[e + 1].clamp(0, total as i32) as usize);
                    if lo < hi {
                        let mut pairs: Vec<(i32, i32)> = (lo..hi).map(|p| (st[p], if with_src { ss[p] } else { 0 })).collect();
                        pairs.sort();
                        for (i, p) in (lo..hi).enumerate() {
                            st[p] = pairs[i].0;
                            if with_src {
                                ss[p] = pairs[i].1;
                            }
                        }
                    }
                }
                v[2][..total * 4].copy_from_slice(&as_bytes(&st));
                v[3][..total * 4].copy_from_slice(&as_bytes(&ss));
            };
            {
                // redcell: the oxide scatter is stable (ascending assignment index within an expert)
                let o = g.rust_only("moe_dispatch stable", &bufs, |b| unsafe {
                    let src = if with_src { b.p(3) as *mut i32 } else { std::ptr::null_mut() };
                    ox::launch_moe_dispatch(b.p(0) as _, b.p(1) as _, b.p(2) as _, src, total as i32, ne as i32, topk as i32, b.p(4) as _, b.p(5) as _, s)
                });
                let st = i32s_of(&o[2][..total * 4]);
                let mut want: Vec<i32> = (0..total as i32).collect();
                want.sort_by_key(|&i| (ids[i as usize], i));
                g.check(&format!("moe_dispatch stable tokens={tokens} topk={topk} experts={ne} src={with_src}"), st == want);
            }
            g.case_canon(&format!("moe_dispatch tokens={tokens} topk={topk} experts={ne} src={with_src}"), bufs, move |rust, b| unsafe {
                let src = if with_src { b.p(3) as *mut i32 } else { std::ptr::null_mut() };
                (if rust { ox::launch_moe_dispatch } else { cref::launch_moe_dispatch })(b.p(0) as _, b.p(1) as _, b.p(2) as _, src, total as i32, ne as i32, topk as i32, b.p(4) as _, b.p(5) as _, s)
            }, canon);
        }
    }
    // ---- weighted reduce
    for &(t, h, topk) in &[(1i32, 1i32, 1i32), (3, 300, 8), (2, 4096, 2), (5, 257, 3), (4, 64, 17), (2, 1000, 300)] {
        for bf in [false, true] {
            k += 1;
            let inp = g.f32s((t * h * topk) as usize);
            let w = g.f32s((t * topk) as usize);
            let o = g.rng.bytes((t * h) as usize * if bf { 2 } else { 4 });
            let s = g.stream(k % 2);
            g.case(&format!("moe_weighted_reduce_flat{} t={t} h={h} topk={topk}", if bf { "_bf16" } else { "" }), vec![inp, w, o], |rust, b| unsafe {
                if bf {
                    (if rust { ox::launch_moe_weighted_reduce_flat_bf16 } else { cref::launch_moe_weighted_reduce_flat_bf16 })(b.p(0) as _, b.p(1) as _, b.p(2), t, h, topk, s)
                } else {
                    (if rust { ox::launch_moe_weighted_reduce_flat } else { cref::launch_moe_weighted_reduce_flat })(b.p(0) as _, b.p(1) as _, b.p(2) as _, t, h, topk, s)
                }
            });
        }
    }
    // ---- grouped GEMMs
    for fi in 0..FMTS.len() {
        let f = &FMTS[fi];
        let (cf, of) = grouped_fns(fi);
        let qk = f.qk as i32;
        // (N, K, tokens, topk, experts). K: multiples of 256 and (for q8_0's 256-wide K tiles) a
        // partial last tile (2880 = 11.25 tiles); N not a multiple of 64; > 64 tokens per expert.
        let shapes: [(i32, i32, usize, usize, usize); 4] = [
            (70, if qk == 32 { 512 } else { 512 }, 5, 2, 4),
            (64, if qk == 32 { 2880 } else { 2816 }, 40, 4, 5),
            (130, qk * 8 + if qk == 32 { 0 } else { 0 }, 150, 2, 2),
            (33, if qk == 32 { 768 } else { 1024 }, 9, 3, 8),
        ];
        for &(n, kk, tokens, topk, ne) in &shapes {
            let total = tokens * topk;
            // routing: distinct experts per token
            let mut ids = vec![0i32; total];
            for t in 0..tokens {
                let mut used = vec![false; ne];
                for s in 0..topk {
                    let mut e = (g.rng.next() % ne as u64) as usize;
                    while used[e] { e = (e + 1) % ne; }
                    used[e] = true;
                    ids[t * topk + s] = e as i32;
                }
            }
            let mut order: Vec<usize> = (0..total).collect();
            order.sort_by_key(|&i| (ids[i], i));
            let sorted: Vec<i32> = order.iter().map(|&i| i as i32).collect();
            let mut bounds = vec![0i32; ne + 1];
            for &e in &ids { bounds[e as usize + 1] += 1; }
            for e in 0..ne { bounds[e + 1] += bounds[e]; }
            let bprw = kk as usize / f.qk;
            let kp = (kk + 511) / 512 * 512;
            let w = g.blocks(f, ne * n as usize * bprw, 64);
            for d1 in [0i32, 1, 2] {
                for weighted in [false, true] {
                    if weighted && d1 == 2 { continue; }
                    for rot in 0..(if weighted { topk.min(2) } else { 1 }) {
                        k += 1;
                        let in_rows = if d1 == 1 { tokens } else { total };
                        let x = g.q8_1(in_rows * kp as usize / 32);
                        let tw: Vec<f32> = (0..total).map(|i| if weighted && i % topk == rot { [0.75f32, -2.0, 1.5e-2][i % 3] } else { 0.0 }).collect();
                        let out_len = if weighted { tokens } else { total } * n as usize;
                        let init: Vec<f32> = (0..out_len).map(|i| ((i % 11) as f32 - 5.5) * 0.25).collect();
                        let s = g.stream(k % 2);
                        let label = format!("moe_grouped_gemm_{} n={n} k={kk} tokens={tokens} topk={topk} experts={ne} d1={d1} weighted={weighted} rot={rot}", f.name);
                        g.case(&label, vec![w.clone(), x, as_bytes(&bounds), as_bytes(&sorted), as_bytes(&tw), as_bytes(&init)], |rust, b| unsafe {
                            let twp = if weighted { b.p(4) as *const f32 } else { std::ptr::null() };
                            (if rust { of } else { cf })(b.p(0), b.p(1), b.p(2) as _, b.p(3) as _, twp, b.p(5) as _, n, kk, kp, ne as i32, topk as i32, d1, s)
                        });
                    }
                }
            }
        }
    }
    g.family("moe_grouped");
}

/// hqq f16/bf16 instances: their launchers are assert(false) host stubs, so the kernels are compared
/// directly against the reference cubin.
fn hqq_kernels(root: &str, kt: &mut Tally) {
    let h = Harness::from_files(&format!("{root}/reference/mistralrs-quant/hqq.cubin"), &format!("{root}/mistralrs-quant-a/mistralrs_quant_a.ptx"));
    let mut r = Rng(0x4A11);
    let bits = [("8bit_u8", "8bit", 1, 1), ("4bit_u8", "4bit", 2, 1), ("2bit_u8", "2bit", 4, 1), ("1bit_u8", "1bit", 8, 1), ("3bit_32", "3bit", 10, 4)];
    for (refb, oxb, chunks, wb) in bits {
        for (ty, mangled_ty, sz) in [("f16", "I6__halfEvP", 2usize), ("bf16", "I13__nv_bfloat16EvP", 2), ("f32", "IfEvP", 4)] {
            let wtag = if wb == 4 { "i" } else { "h" };
            let tail = if ty == "f32" { "T_S2_S2_ii" } else { "T_S3_S3_ii" };
            let refname = format!("_Z25dequantize_{refb}_kernel{mangled_ty}{wtag}P{tail}");
            let oxname = format!("hqq_{oxb}_{ty}");
            for &(hh, ww) in &[(3usize, 7usize), (40, 33), (1, 513)] {
                let n = hh * ww;
                let vals = |r: &mut Rng, k: usize| -> Vec<u8> {
                    if sz == 4 { as_bytes(&r.f32s(k)) } else { as_bytes(&r.b16s(k)) }
                };
                let bufs = vec![r.bytes(n * wb), vals(&mut r, ww), vals(&mut r, ww), r.bytes(n * chunks * sz)];
                let args = [Arg::Buf(0), Arg::Buf(1), Arg::Buf(2), Arg::Buf(3), Arg::I32(hh as i32), Arg::I32(ww as i32)];
                let d = h.diff_pair(&refname, &oxname, ((n as u32).div_ceil(256), 1, 1), (256, 1, 1), 0, &args, &bufs, &[3]);
                kt.record(&format!("{oxname} h={hh} w={ww}"), &d);
            }
        }
    }
}

pub fn run() -> bool {
    let ctx = CudaContext::new(0).expect("cuda context");
    ctx.bind_to_thread().unwrap();
    let streams = vec![ctx.default_stream(), ctx.new_stream().unwrap()];
    let mut g = G { ctx: ctx.clone(), streams, rng: Rng(0x6A7E_A11C), t: Tally::default(), calls: 0, family_calls: 0, order_only: 0 };
    let only = std::env::var("MQA_ONLY").ok();
    let want = |f: &str| only.as_deref().map(|o| o.split(',').any(|x| x == f)).unwrap_or(true);

    if want("hqq_bitpack") { hqq_bitpack(&mut g); }
    if want("hqq") { hqq(&mut g); }
    if want("dequant") { bnb(&mut g); }
    if want("gelu_tanh_and_mul") { gelu_tanh_and_mul(&mut g); }
    if want("rotary") { rotary(&mut g); }
    if want("moe_align") { moe_align(&mut g); }
    if want("moe_data") { moe_data(&mut g); }
    if want("ops") { ops(&mut g); }
    if want("gemv") { gemv(&mut g); }
    if want("indexed_moe") { indexed_moe(&mut g); }
    if want("moe_grouped") { moe_grouped(&mut g); }
    if want("exit") { exit_cases(&mut g); }

    let root = format!("{}/titan-engine/oxide-kernels", std::env::var("HOME").unwrap());
    let mut kt = Tally::default();
    if want("hqq") { hqq_kernels(&root, &mut kt); }
    let kok = kt.finish("kernel-level kdiff (instances no launcher reaches)");

    for f in g.t.failures.iter().take(40) {
        println!("  FAIL {f}");
    }
    println!("  order-only (atomic-order) differences accepted after canonicalizing: {} cases", g.order_only);
    let ok = g.t.failures.is_empty() && kok && g.calls > 0;
    println!(
        "mistralrs-quant-a: {} launcher calls, {} bytes compared, {} failing -> {}",
        g.calls,
        g.t.bytes + kt.bytes,
        g.t.failures.len() + kt.failures.len(),
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}


// ================================================================================================
// Exit paths: host-pass assert(false) stubs and ops.cu's CUDA_CHECK exit(). Each case runs in a
// child process per side; exit status (code or signal) and stderr must match exactly.

pub const EXIT_CASES: usize = 22;

/// Run exit case `id` on one side (called in a child process).
pub fn exit_case(rust: bool, id: usize) {
    let ctx = CudaContext::new(0).expect("cuda context");
    ctx.bind_to_thread().unwrap();
    let st = ctx.default_stream();
    let buf = DeviceBuffer::<u8>::from_host(&st, &vec![1u8; 4096]).unwrap();
    let p = buf.cu_deviceptr() as *mut c_void;
    let dims = DeviceBuffer::<u32>::from_host(&st, &[4u32, 4]).unwrap();
    let dp = dims.cu_deviceptr() as *const c_void;
    let nul = std::ptr::null_mut::<c_void>();
    macro_rules! pick { ($f:ident) => { if rust { ox::$f } else { cref::$f } }; }
    unsafe {
        match id {
            0 => pick!(dequantize_8bit_u8_kernel_f16)(p as _, p as _, p as _, p as _, 2, 2),
            1 => pick!(dequantize_8bit_u8_kernel_bf16)(p as _, p as _, p as _, p as _, 2, 2),
            2 => pick!(dequantize_4bit_u8_kernel_f16)(p as _, p as _, p as _, p as _, 2, 2),
            3 => pick!(dequantize_4bit_u8_kernel_bf16)(p as _, p as _, p as _, p as _, 2, 2),
            4 => pick!(dequantize_2bit_u8_kernel_f16)(p as _, p as _, p as _, p as _, 2, 2),
            5 => pick!(dequantize_2bit_u8_kernel_bf16)(p as _, p as _, p as _, p as _, 2, 2),
            6 => pick!(dequantize_1bit_u8_kernel_f16)(p as _, p as _, p as _, p as _, 2, 2),
            7 => pick!(dequantize_1bit_u8_kernel_bf16)(p as _, p as _, p as _, p as _, 2, 2),
            8 => pick!(dequantize_3bit_32_kernel_f16)(p as _, p as _, p as _, p as _, 2, 2),
            9 => pick!(dequantize_3bit_32_kernel_bf16)(p as _, p as _, p as _, p as _, 2, 2),
            10 => pick!(nonzero_f16)(p, 16, 3, dp, 2, p, nul),
            11 => pick!(nonzero_bf16)(p, 16, 3, dp, 2, p, nul),
            12 => pick!(bitwise_and_u8)(p, p, p, 0),
            13 => pick!(bitwise_or_i64)(p, p, p, 0),
            14 => pick!(bitwise_xor_i32)(p, p, p, 0),
            15 => pick!(leftshift_u32)(p, p, 0, 3),
            16 => pick!(gptoss_swiglu_bf16)(p, p, p, 0, 1.0, 7.0, nul),
            17 => pick!(gptoss_swiglu_interleaved_f32)(p, p, 5, 0, 1.0, 7.0, nul),
            18 => pick!(softmax_with_sinks_f16)(p, p, std::ptr::null(), p, 0, 2, 2, 8, 1.0, nul),
            19 => pick!(fused_glu_f32)(p, p, p, 0, 1, nul),
            20 => pick!(softcap_bf16_to_f32)(p, p, 0, 30.0, nul),
            21 => pick!(nonzero_f32)(p, 16, 0, dp, 2, p, nul),
            _ => panic!("no exit case {id}"),
        }
    }
    ctx.synchronize().unwrap();
}

/// Run every exit case on both sides in child processes and compare status + stderr.
pub fn exit_cases(g: &mut G) {
    let exe = std::env::current_exe().unwrap();
    let mut nonzero_exits = 0;
    for id in 0..EXIT_CASES {
        let run = |side: &str| {
            let o = std::process::Command::new(&exe).args(["--exit-case", side, &id.to_string()]).output().unwrap();
            use std::os::unix::process::ExitStatusExt;
            (o.status.code(), o.status.signal(), String::from_utf8_lossy(&o.stderr).into_owned())
        };
        let (c, r) = (run("c"), run("rust"));
        if c.0 != Some(0) {
            nonzero_exits += 1;
        }
        let same = c == r;
        if !same {
            println!("  exit case {id}: C {c:?} vs Rust {r:?}");
        }
        g.calls += 2;
        g.family_calls += 2;
        let d = kdiff::Diff { bytes: c.2.len(), differing: (!same) as usize, first: if same { None } else { Some((0, 0, 0, 0)) } };
        g.t.record(&format!("exit case {id}"), &d);
    }
    // every case must actually terminate abnormally on the reference side
    g.scalar("exit cases terminating", EXIT_CASES as u64, nonzero_exits as u64);
    g.family("exit paths (assert stubs, CUDA_CHECK)");
}
