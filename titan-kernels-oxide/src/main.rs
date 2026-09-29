//! titan-engine expert kernels (cuda-oxide), bit-matched against mistral.rs 84b53bf
//! `indexed_moe_forward_{q4k,q5k,q6k}_q8_1` (kernels/indexed_moe/indexed_moe.cu).
//!
//! The reference is llama.cpp's mmvq algorithm compiled by nvcc with
//! `-O3 --use_fast_math`. Bit-identity needs its exact float program, and the
//! spec is the SASS, not the PTX (m0/ref/q4k.sass, q5k.sass):
//!
//! - per K-quant block: `sumf_d = fma(d8[1], x1, fma(d8[0], x0, 0))` (and `sumf_m`);
//! - the epilogue nvcc writes as `dm.x*sumf_d - dm.y*sumf_m` is contracted by
//!   ptxas to `FMUL p = sumf_m*dm.y; FFMA r = sumf_d*dm.x - p`;
//! - everything flushes denormals (`.ftz`);
//! - warps 1..3 hand partials to warp 0 through shared memory, added in warp
//!   order, then a butterfly over xor distances 16, 8, 4, 2, 1.
//!
//! Rust never contracts `a * b + c`, so every operation is the intrinsic ptxas chose.
//!
//! Beyond the reference, each kernel reads the expert through a slot map
//! (`slot_map[expert]` = the slot in `w` holding it). Tiering (titan-engine M1+)
//! moves experts between slots without the kernel knowing; with the identity map
//! the kernel is the reference.
mod cpu;
use cuda_core::{CudaContext, CudaModule, DeviceBuffer};
use cuda_device::{SharedArray, convert, device, dotprod, float, kernel, launch_bounds, ptx_asm, thread, warp};
use cuda_host::cuda_module;
use std::ffi::c_void;
use std::sync::Arc;

#[cuda_module]
mod kernels {
    use super::*;

    /// block_q4_K: half2 dm (word 0), 12 scale bytes (words 1..4), 128 nibble bytes (4..36).
    const Q4K_WORDS: usize = 36;
    const Q4K_W_QS: usize = 4;
    /// block_q5_K: half2 dm (0), 12 scale bytes (1..4), 32 high-bit bytes (4..12), 128 nibble bytes (12..44).
    const Q5K_WORDS: usize = 44;
    const Q5K_W_QH: usize = 4;
    const Q5K_W_QS: usize = 12;
    /// block_q6_K is 210 bytes, so a block is only 2-byte aligned and is read by byte offset:
    /// ql 128 bytes (0..128), qh 64 (128..192), int8 scales 16 (192..208), half d (208..210).
    const Q6K_BYTES: usize = 210;
    const Q6K_QH: usize = 128;
    const Q6K_SCALES: usize = 192;
    const Q6K_D: usize = 208;
    /// Format ids for `moe_gemv`.
    const FMT_Q4K: u32 = 0;
    const FMT_Q5K: u32 = 1;
    const FMT_Q6K: u32 = 2;
    /// GGML Q1_0 (type 41): half d + 128 sign bits, 18 bytes (2-byte aligned: byte-addressed).
    const FMT_Q1_0: u32 = 3;
    const Q1_0_BYTES: usize = 18;
    /// llama.cpp mul_mat_vec_q<Q1_0> geometry: qi 4, vdr 1 -> 4 threads per block (one 32-value
    /// chunk each), 4 warps -> 32 blocks per iteration.
    const Q1_0_THREADS_PER_BLOCK: usize = 4;
    const Q1_0_BLOCKS_PER_ITER: usize = 32;
    /// GGML IQ4_NL (type 20): half d + 16 nibble bytes indexing kvalues_iq4nl, 18 bytes.
    const FMT_IQ4_NL: u32 = 4;
    const IQ4_NL_BYTES: usize = 18;
    /// llama.cpp mul_mat_vec_q<IQ4_NL> geometry: qi 4, vdr 2 -> 2 threads per block (two
    /// 4-byte chunks each), 4 warps -> 64 blocks per iteration.
    const IQ4_NL_THREADS_PER_BLOCK: usize = 2;
    const IQ4_NL_BLOCKS_PER_ITER: usize = 64;
    /// kvalues_iq4nl as the little-endian words get_int_from_table_16 reads.
    const IQ4_NL_TABLE: [u32; 4] = [0xBFAD9881, 0xF6EADDCF, 0x26190D01, 0x71594535];
    /// GGML MXFP4 (type 39): E8M0 byte e + 16 nibble bytes indexing kvalues_mxfp4, 17 bytes.
    /// Same mul_mat_vec_q geometry as IQ4_NL (qi 4, vdr 2).
    const FMT_MXFP4: u32 = 5;
    const MXFP4_BYTES: usize = 17;
    const MXFP4_TABLE: [u32; 4] = [0x03020100, 0x0C080604, 0xFDFEFF00, 0xF4F8FAFC];
    /// GGML NVFP4 (type 40): 4 UE4M3 sub-block scales + 32 nibble bytes indexing kvalues_mxfp4,
    /// 36 bytes, 64 values. llama.cpp mul_mat_vec_q<NVFP4> geometry: qi 8, vdr 4 -> 2 threads per
    /// block (two 16-value sub-blocks each), 64 blocks per iteration, 2 Q8_1 blocks per block.
    const FMT_NVFP4: u32 = 6;
    const NVFP4_BYTES: usize = 36;
    /// Shared by Q4_K and Q5_K: scales start at word 1.
    const W_SCALES: usize = 1;
    /// block_q8_1: half2 ds (word 0), 32 int8 (words 1..9).
    const Q8_WORDS: usize = 9;
    /// QK_K / QK8_1.
    const Q8_PER_KBLOCK: usize = 8;
    const NWARPS: usize = 4;
    const THREADS: usize = NWARPS * 32;
    /// qi / vdr for Q4_K and Q5_K (32 / 2): threads cooperating on one K-quant block.
    const THREADS_PER_BLOCK: usize = 16;
    /// vdr * nwarps * WARP_SIZE / qi.
    const BLOCKS_PER_ITER: usize = 8;
    /// The same for Q6_K (qi 32, vdr 1).
    const Q6K_THREADS_PER_BLOCK: usize = 32;
    const Q6K_BLOCKS_PER_ITER: usize = 4;
    /// Slot-map value for an expert that is not in any GPU slot.
    const NOT_RESIDENT: usize = u32::MAX as usize;

    /// The reference's float ops. FTZ = mistralrs-quant's indexed_moe.cu (built with
    /// --use_fast_math); !FTZ = candle-kernels' quantized.cu (plain -O3), which is what
    /// mistral.rs's GGUF models call through `QMatMul::indexed_moe_forward`.
    #[inline(always)]
    fn fma<const FTZ: bool>(a: f32, b: f32, c: f32) -> f32 {
        if FTZ { float::fma_rn_ftz_f32(a, b, c) } else { float::fma_rn_f32(a, b, c) }
    }
    #[inline(always)]
    fn mul<const FTZ: bool>(a: f32, b: f32) -> f32 {
        if FTZ { float::mul_rn_ftz_f32(a, b) } else { float::mul_rn_f32(a, b) }
    }
    #[inline(always)]
    fn add<const FTZ: bool>(a: f32, b: f32) -> f32 {
        if FTZ { float::add_rn_ftz_f32(a, b) } else { float::add_rn_f32(a, b) }
    }

    #[inline(always)]
    fn half_lo(w: u32) -> f32 {
        convert::cvt_f32_f16x2_lo(w)
    }

    #[inline(always)]
    fn half_hi(w: u32) -> f32 {
        convert::cvt_f32_f16x2_lo(w >> 16)
    }

    /// The 6-bit scale and min for this thread's pair of sub-blocks, shared by
    /// Q4_K and Q5_K (`aux` in the reference). Returns (sc[0], sc[1], m[0], m[1]).
    #[inline(always)]
    unsafe fn k_scales(w: &[u32], wb: usize, bq8_offset: usize) -> [u32; 4] {
        let s16 = |i: usize| -> u32 {
            let word = *w.get_unchecked(wb + W_SCALES + i / 2);
            (word >> (16 * (i % 2))) & 0xFFFF
        };
        let j = bq8_offset / 2;
        let (aux0, aux1) = if j < 2 {
            (s16(j) & 0x3f3f, s16(j + 2) & 0x3f3f)
        } else {
            (
                ((s16(j + 2) >> 0) & 0x0f0f) | ((s16(j - 2) & 0xc0c0) >> 2),
                ((s16(j + 2) >> 4) & 0x0f0f) | ((s16(j) & 0xc0c0) >> 2),
            )
        };
        [aux0 & 0xFF, aux0 >> 8, aux1 & 0xFF, aux1 >> 8]
    }

    /// The shared float tail of `vec_dot_q{4,5}_K_q8_1_impl_vmmq`: `v[i]` are the
    /// two 4-weight words for sub-block pair i, `u` the matching Q8_1 words.
    #[inline(always)]
    unsafe fn k_dot<const FTZ: bool>(v: [[u32; 2]; 2], x: &[u32], xb: usize, bq8_offset: usize, iqs: usize, s: [u32; 4], dm: u32) -> f32 {
        let mut sumf_d = 0f32;
        let mut sumf_m = 0f32;
        let mut i = 0;
        while i < 2 {
            let bq8 = xb + (bq8_offset + i) * Q8_WORDS;
            let d8 = half_lo(*x.get_unchecked(bq8));
            let u0 = *x.get_unchecked(bq8 + 1 + (iqs / 2) % 4);
            let u1 = *x.get_unchecked(bq8 + 1 + (iqs / 2) % 4 + 4);
            let dot1 = dotprod::dp4a_s32(v[i][1], u1, dotprod::dp4a_s32(v[i][0], u0, 0));
            let dot2 = dotprod::dp4a_s32(0x0101_0101, u1, dotprod::dp4a_s32(0x0101_0101, u0, 0));
            // cvt.rn.f32.s32 of the integer product, then fma.rn.ftz.
            let x_d = (dot1 * s[i] as i32) as f32;
            let x_m = (dot2 * s[2 + i] as i32) as f32;
            sumf_d = fma::<FTZ>(d8, x_d, sumf_d);
            sumf_m = fma::<FTZ>(d8, x_m, sumf_m);
            i += 1;
        }
        // ptxas's contraction of nvcc's `dm.x*sumf_d - dm.y*sumf_m`.
        let p = mul::<FTZ>(sumf_m, half_hi(dm));
        fma::<FTZ>(sumf_d, half_lo(dm), -p)
    }

    /// `vec_dot_q4_K_q8_1`. SAFETY: `wb`/`xb` start a Q4_K block and its 8 Q8_1 blocks; `iqs` even, < 32.
    #[inline(always)]
    unsafe fn dot_q4k<const FTZ: bool>(w: &[u32], wb: usize, x: &[u32], xb: usize, iqs: usize) -> f32 {
        let bq8_offset = 2 * ((iqs / 2) / 4);
        let q4 = wb + Q4K_W_QS + 4 * bq8_offset + (iqs / 2) % 4;
        let (a, b) = (*w.get_unchecked(q4), *w.get_unchecked(q4 + 4));
        let v = [[a & 0x0F0F_0F0F, b & 0x0F0F_0F0F], [(a >> 4) & 0x0F0F_0F0F, (b >> 4) & 0x0F0F_0F0F]];
        k_dot::<FTZ>(v, x, xb, bq8_offset, iqs, k_scales(w, wb, bq8_offset), *w.get_unchecked(wb))
    }

    /// `vec_dot_q5_K_q8_1`: Q4_K's nibbles plus a fifth bit from the qh plane.
    #[inline(always)]
    unsafe fn dot_q5k<const FTZ: bool>(w: &[u32], wb: usize, x: &[u32], xb: usize, iqs: usize) -> f32 {
        let bq8_offset = 2 * ((iqs / 2) / 4);
        let ql = wb + Q5K_W_QS + 4 * bq8_offset + (iqs / 2) % 4;
        let qh = wb + Q5K_W_QH + (iqs / 2) % 4;
        let (l0, l1) = (*w.get_unchecked(ql), *w.get_unchecked(ql + 4));
        let (h0, h1) = (*w.get_unchecked(qh) >> bq8_offset, *w.get_unchecked(qh + 4) >> bq8_offset);
        let mut v = [[0u32; 2]; 2];
        let mut i = 0;
        while i < 2 {
            v[i][0] = ((l0 >> (4 * i)) & 0x0F0F_0F0F) | (((h0 >> i) << 4) & 0x1010_1010);
            v[i][1] = ((l1 >> (4 * i)) & 0x0F0F_0F0F) | (((h1 >> i) << 4) & 0x1010_1010);
            i += 1;
        }
        k_dot::<FTZ>(v, x, xb, bq8_offset, iqs, k_scales(w, wb, bq8_offset), *w.get_unchecked(wb))
    }

    /// 16-bit load at a byte offset (Q6_K blocks are only 2-byte aligned).
    #[inline(always)]
    unsafe fn ld16(p: *const u8, off: usize) -> u32 {
        *(p.add(off) as *const u16) as u32
    }

    /// `get_int_from_uint8`: a 32-bit word from two 16-bit loads.
    #[inline(always)]
    unsafe fn ld32(p: *const u8, off: usize) -> u32 {
        ld16(p, off) | (ld16(p, off + 2) << 16)
    }

    /// `tmp += vec_dot_q6_K_q8_1(...)`: nvcc inlines the dot into the loop and contracts
    /// `tmp + d*sumf` to `fma(d, sumf, tmp)` (candle and mistralrs-quant PTX alike).
    /// SAFETY: `b` starts a Q6_K block, `xb` its 8 Q8_1 blocks; `iqs` < 32.
    #[inline(always)]
    unsafe fn acc_q6k<const FTZ: bool>(tmp: f32, b: *const u8, x: &[u32], xb: usize, iqs: usize) -> f32 {
        let bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        let scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        let vh_shift = 2 * ((iqs % 16) / 8);
        let vl = ld32(b, 4 * iqs);
        let vh = (ld32(b, Q6K_QH + 4 * (8 * (iqs / 16) + iqs % 8)) as i32) >> vh_shift;
        let mut sumf = 0f32;
        let mut i = 0;
        while i < 2 {
            let bq8 = xb + (bq8_offset + 2 * i) * Q8_WORDS;
            let d8 = half_lo(*x.get_unchecked(bq8));
            let u = *x.get_unchecked(bq8 + 1 + iqs % 8);
            let sc = *b.add(Q6K_SCALES + scale_offset + 4 * i) as i8 as i32;
            let vil = (vl >> (4 * i)) & 0x0F0F_0F0F;
            let vih = (((vh >> (4 * i)) << 4) as u32) & 0x3030_3030;
            // __vsubss4(vil | vih, 0x20202020): bytes are in [0, 63], so per-byte x - 32
            // never saturates and never borrows once each byte is offset by 128.
            let vi = ((vil | vih | 0x8080_8080).wrapping_sub(0x2020_2020)) ^ 0x8080_8080;
            sumf = fma::<FTZ>(d8, dotprod::dp4a_s32(vi, u, 0).wrapping_mul(sc) as f32, sumf);
            i += 1;
        }
        fma::<FTZ>(half_lo(ld16(b, Q6K_D)), sumf, tmp)
    }

    /// `__byte_perm(a, b, s)`: prmt with the selector masked to its 3-bit byte indices.
    #[inline(always)]
    fn byte_perm(a: u32, b: u32, s: u32) -> u32 {
        let r: u32;
        let s = s & 0x7777;
        unsafe { ptx_asm!("prmt.b32 %0, %1, %2, %3;", out("=r") r, in("r") a, in("r") b, in("r") s, options(register_only)); }
        r
    }

    /// `tmp += vec_dot_q1_0_q8_1(...)` as llama.cpp acecd56 builds it (-use_fast_math, SASS):
    /// 32 sign bits -> +-1 bytes by three `__byte_perm`s, four dp4a against the Q8_1 chunk, then
    /// `fma.ftz(mul.ftz(d, d8), (float)sumi, tmp)`.
    /// SAFETY: `b` starts a Q1_0 block, `xb` its 4 Q8_1 blocks; `iqs` < 4.
    #[inline(always)]
    unsafe fn acc_q1_0<const FTZ: bool>(tmp: f32, b: *const u8, x: &[u32], xb: usize, iqs: usize) -> f32 {
        let bq8 = xb + iqs * Q8_WORDS;
        let mut sumi = 0i32;
        let mut j = 0;
        while j < 2 {
            let q = (ld16(b, 2 + 4 * iqs + 2 * j) as u16 as i16) as i32 as u32;
            let n0 = byte_perm(0x11100100, 0x11100100, q);
            let n1 = byte_perm(0x11100100, 0x11100100, ((q as i32) >> 2) as u32);
            let s0 = byte_perm(0x01FF, 0x01FF, n0);
            let s1 = byte_perm(0x01FF, 0x01FF, n1);
            let s2 = byte_perm(0x01FF, 0x01FF, ((n0 as i32) >> 16) as u32);
            let s3 = byte_perm(0x01FF, 0x01FF, ((n1 as i32) >> 16) as u32);
            sumi = dotprod::dp4a_s32(byte_perm(s0, s1, 0x5410), *x.get_unchecked(bq8 + 1 + 4 * j), sumi);
            sumi = dotprod::dp4a_s32(byte_perm(s0, s1, 0x7632), *x.get_unchecked(bq8 + 2 + 4 * j), sumi);
            sumi = dotprod::dp4a_s32(byte_perm(s2, s3, 0x5410), *x.get_unchecked(bq8 + 3 + 4 * j), sumi);
            sumi = dotprod::dp4a_s32(byte_perm(s2, s3, 0x7632), *x.get_unchecked(bq8 + 4 + 4 * j), sumi);
            j += 1;
        }
        let d = half_lo(ld16(b, 0));
        let d8 = half_lo(*x.get_unchecked(bq8));
        fma::<FTZ>(mul::<FTZ>(d, d8), sumi as f32, tmp)
    }

    /// llama.cpp `get_int_from_table_16(q4, table)`: (table bytes of the 4 low nibbles, of the 4 high nibbles).
    #[inline(always)]
    fn table16(q4: u32, t: [u32; 4]) -> (u32, u32) {
        let sel = 0x32103210 | ((q4 & 0x88888888) >> 1);
        let t0 = byte_perm(byte_perm(t[0], t[1], q4), byte_perm(t[2], t[3], q4), sel);
        let t1 = byte_perm(byte_perm(t[0], t[1], q4 >> 16), byte_perm(t[2], t[3], q4 >> 16), sel >> 16);
        (byte_perm(t0, t1, 0x6420), byte_perm(t0, t1, 0x7531))
    }

    /// `tmp += vec_dot_iq4_nl_q8_1(...)` as llama.cpp acecd56 builds it (-use_fast_math, SASS):
    /// two 32-bit nibble words -> table bytes, four dp4a against the Q8_1 block, then
    /// `fma.ftz(mul.ftz(d, d8), (float)sumi, tmp)`.
    /// SAFETY: `b` starts an IQ4_NL block, `xb` its Q8_1 block; `iqs` is 0 or 2.
    #[inline(always)]
    unsafe fn acc_iq4_nl<const FTZ: bool>(tmp: f32, b: *const u8, x: &[u32], xb: usize, iqs: usize) -> f32 {
        let mut sumi = 0i32;
        let mut l = 0;
        while l < 2 {
            let (lo, hi) = table16(ld32(b, 2 + 4 * (iqs + l)), IQ4_NL_TABLE);
            sumi = dotprod::dp4a_s32(lo, *x.get_unchecked(xb + 1 + iqs + l), sumi);
            sumi = dotprod::dp4a_s32(hi, *x.get_unchecked(xb + 5 + iqs + l), sumi);
            l += 1;
        }
        let d = half_lo(ld16(b, 0));
        let d8 = half_lo(*x.get_unchecked(xb));
        fma::<FTZ>(mul::<FTZ>(d, d8), sumi as f32, tmp)
    }

    /// `ggml_cuda_e8m0_to_fp32` (CUDART >= 12.8: `cvt.rn.bf16x2.ue8m0x2`, then bf16 -> f32) without
    /// the sm_120a-only instruction (this module stays plain sm_120): 2^(e - 127), e = 255 NaN,
    /// e = 0 the denormal 2^-127. Every consumer is a mul.ftz, which flushes the denormal and
    /// canonicalises the NaN, so the result is the reference's bit for bit (gated).
    #[inline(always)]
    fn e8m0(e: u32) -> f32 {
        if e == 255 {
            f32::from_bits(0x7FFF_FFFF)
        } else if e == 0 {
            f32::from_bits(0x0040_0000)
        } else {
            f32::from_bits(e << 23)
        }
    }

    /// `tmp += vec_dot_mxfp4_q8_1(...)` as llama.cpp acecd56 builds it (-use_fast_math, SASS):
    /// byte-loaded nibble words -> table bytes, four dp4a against the Q8_1 block, then
    /// `fma.ftz(mul.ftz(mul.ftz(e8m0(e), 0.5), d8), (float)sumi, tmp)`.
    /// SAFETY: `b` starts an MXFP4 block, `xb` its Q8_1 block; `iqs` is 0 or 2.
    #[inline(always)]
    unsafe fn acc_mxfp4<const FTZ: bool>(tmp: f32, b: *const u8, x: &[u32], xb: usize, iqs: usize) -> f32 {
        let mut sumi = 0i32;
        let mut l = 0;
        while l < 2 {
            let q = b.add(1 + 4 * (iqs + l));
            let w = (*q as u32) | ((*q.add(1) as u32) << 8) | ((*q.add(2) as u32) << 16) | ((*q.add(3) as u32) << 24);
            let (lo, hi) = table16(w, MXFP4_TABLE);
            sumi = dotprod::dp4a_s32(lo, *x.get_unchecked(xb + 1 + iqs + l), sumi);
            sumi = dotprod::dp4a_s32(hi, *x.get_unchecked(xb + 5 + iqs + l), sumi);
            l += 1;
        }
        let d = mul::<FTZ>(mul::<FTZ>(e8m0(*b as u32), 0.5), half_lo(*x.get_unchecked(xb)));
        fma::<FTZ>(d, sumi as f32, tmp)
    }

    /// `ggml_cuda_ue4m3_to_fp32` (FP8_AVAILABLE): the byte as a signed E4M3 (0x7F / 0xFF -> 0),
    /// `cvt.rn.f16x2.e4m3x2`, f16 -> f32, then `/ 2` as fast-math `div.approx.ftz`.
    #[inline(always)]
    fn ue4m3(b: u32) -> f32 {
        let b = if b & 0x7F == 0x7F { 0u16 } else { b as u16 };
        let r: u32;
        unsafe { ptx_asm!("cvt.rn.f16x2.e4m3x2 %0, %1;", out("=r") r, in("h") b, options(register_only)); }
        let v = half_lo(r);
        let q: f32;
        unsafe { ptx_asm!("div.approx.ftz.f32 %0, %1, 0f40000000;", out("=f") q, in("f") v, options(register_only)); }
        q
    }

    /// `tmp += vec_dot_nvfp4_q8_1(...)` as llama.cpp acecd56 builds it (-use_fast_math, SASS): per
    /// sub-block two nibble words -> table bytes, four dp4a, `sum = fma.ftz(mul.ftz(d_s, d8),
    /// (float)sumi, sum)` from sum = +0, then `add.ftz(tmp, sum)` (not contracted).
    /// SAFETY: `b` starts an NVFP4 block, `xb` its first Q8_1 block; `iqs` is 0 or 4.
    #[inline(always)]
    unsafe fn acc_nvfp4<const FTZ: bool>(tmp: f32, b: *const u8, x: &[u32], xb: usize, iqs: usize) -> f32 {
        let mut sum = 0f32;
        let mut i = 0;
        while i < 2 {
            let iqs0 = iqs + 2 * i;
            let is = iqs0 >> 1;
            let (v0x, v0y) = table16(ld32(b, 4 + 4 * iqs0), MXFP4_TABLE);
            let (v1x, v1y) = table16(ld32(b, 8 + 4 * iqs0), MXFP4_TABLE);
            let bq8 = xb + (is >> 1) * Q8_WORDS;
            let i8 = (is & 1) << 2;
            let mut sumi = dotprod::dp4a_s32(v0x, *x.get_unchecked(bq8 + 1 + i8), 0);
            sumi = dotprod::dp4a_s32(v0y, *x.get_unchecked(bq8 + 3 + i8), sumi);
            sumi = dotprod::dp4a_s32(v1x, *x.get_unchecked(bq8 + 2 + i8), sumi);
            sumi = dotprod::dp4a_s32(v1y, *x.get_unchecked(bq8 + 4 + i8), sumi);
            let d = mul::<FTZ>(ue4m3(*b.add(is) as u32), half_lo(*x.get_unchecked(bq8)));
            sum = fma::<FTZ>(d, sumi as f32, sum);
            i += 1;
        }
        add::<FTZ>(tmp, sum)
    }

    /// Sum across the CUDA block exactly as the reference: shared-memory handoff
    /// in warp order, then the xor butterfly. Lane 0 of warp 0 gets the result.
    #[device]
    #[inline(always)]
    unsafe fn block_sum<const FTZ: bool>(mut tmp: f32, tid: usize) -> (bool, f32) {
        static mut PARTIAL: SharedArray<f32, 96> = SharedArray::UNINIT;
        let warp_id = tid / 32;
        let lane = tid % 32;
        let partial = SharedArray::as_raw_mut_ptr(&raw mut PARTIAL);
        if warp_id > 0 {
            *partial.add((warp_id - 1) * 32 + lane) = tmp;
        }
        thread::sync_threads();
        if warp_id != 0 {
            return (false, 0.0);
        }
        let mut l = 0;
        while l < NWARPS - 1 {
            tmp = add::<FTZ>(tmp, *partial.add(l * 32 + lane));
            l += 1;
        }
        let mut mask = 16;
        while mask > 0 {
            tmp = add::<FTZ>(tmp, warp::shuffle_xor_f32_sync(0xffff_ffff, tmp, mask));
            mask >>= 1;
        }
        (lane == 0, tmp)
    }

    /// Launch geometry shared by the formats: one CUDA block of 4 warps per
    /// (output row, task), the reference's grid (n, batch, topk) flattened to 1-D
    /// as blk = task * n + row, task = batch_index * topk + k.
    /// Returns (tid, row0, task, input_row, slot).
    #[device]
    #[inline(always)]
    unsafe fn geometry(ids: &[u32], slot_map: &[u32], n: u32, topk: u32, input_dim1: u32) -> (usize, usize, usize, usize, usize) {
        let g = thread::index_1d().get();
        let blk = g / THREADS;
        let tid = g % THREADS;
        let row0 = blk % n as usize;
        let task = blk / n as usize;
        let input_row = if input_dim1 == 1 { task / topk as usize } else { task };
        let slot = *slot_map.get_unchecked(*ids.get_unchecked(task) as usize) as usize;
        (tid, row0, task, input_row, slot)
    }

    /// Q8_1 input quantization, candle's `quantize_q8_1` (plain -O3): per 32 values a
    /// warp max of |x| and a warp sum, `d = amax / 127` (IEEE div.rn), and
    /// `q = trunc_rz(t + copysign(0.5, t))` with `t = x / d` — nvcc's own roundf,
    /// read off candle's PTX. `ds` = half2(d, sum), round-to-nearest.
    /// Launch: one thread per padded element, rows * kx_padded threads, block 256.
    /// SAFETY: `y` holds rows * kx_padded / 32 blocks of 36 bytes; `x` rows * kx floats.
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn quantize_q8_1(x: &[f32], y: *mut u8, kx: u32, kx_padded: u32) {
        let g = thread::index_1d().get();
        let kp = kx_padded as usize;
        let (iy, ix) = (g / kp, g % kp);
        let (ib, iqs) = (g / 32, g % 32);
        let xi = if ix < kx as usize { *x.get_unchecked(iy * kx as usize + ix) } else { 0.0 };
        let mut amax = f32::from_bits(xi.to_bits() & 0x7FFF_FFFF); // abs.f32: +0 for -0
        let mut sum = xi;
        let mut mask = 16;
        while mask > 0 {
            amax = amax.max(warp::shuffle_xor_f32_sync(0xffff_ffff, amax, mask));
            mask >>= 1;
        }
        let mut mask = 16;
        while mask > 0 {
            sum = float::add_rn_f32(sum, warp::shuffle_xor_f32_sync(0xffff_ffff, sum, mask));
            mask >>= 1;
        }
        let d = amax / 127.0;
        let q = if amax == 0.0 {
            0i32
        } else {
            let t = xi / d;
            let half = f32::from_bits(0x3F00_0000 | (t.to_bits() & 0x8000_0000));
            float::add_rz_f32(t, half) as i32
        };
        *y.add(ib * 36 + 4 + iqs) = q as u8;
        if iqs == 0 {
            *(y.add(ib * 36) as *mut u32) = convert::cvt_f16x2_f32(d, sum);
        }
    }

    /// Shared body of every expert GEMV: format via `FMT`, float program via `FTZ`.
    /// SAFETY: `out` holds tasks * n floats; `x` holds the Q8_1 rows (k_padded / 32
    /// blocks each); every id maps to a filled slot of `FMT`-format blocks.
    #[device]
    #[inline(always)]
    unsafe fn moe_gemv<const FTZ: bool, const FMT: u32>(
        w: &[u32], x: &[u32], ids: &[u32], slot_map: &[u32], out: *mut f32,
        n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32,
    ) {
        let (tid, row0, task, input_row, slot) = geometry(ids, slot_map, n, topk, input_dim1);
        // Not resident on the GPU: the CPU computes this task. Every thread of the block
        // shares the task, so the whole block leaves together and no barrier is split.
        if slot == NOT_RESIDENT {
            return;
        }
        let bpr = k as usize
            / if FMT == FMT_Q1_0 {
                128
            } else if FMT == FMT_IQ4_NL || FMT == FMT_MXFP4 {
                32
            } else if FMT == FMT_NVFP4 {
                64
            } else {
                256
            };
        let x_base = input_row * (k_padded as usize / 32) * Q8_WORDS;
        let mut tmp = 0f32;
        if FMT == FMT_NVFP4 {
            let row_base = (slot * n as usize + row0) * bpr * NVFP4_BYTES;
            let wp = w.as_ptr() as *const u8;
            let mut kbx = tid / IQ4_NL_THREADS_PER_BLOCK;
            let kqs = 4 * (tid % IQ4_NL_THREADS_PER_BLOCK);
            while kbx < bpr {
                tmp = acc_nvfp4::<FTZ>(tmp, wp.add(row_base + kbx * NVFP4_BYTES), x, x_base + kbx * 2 * Q8_WORDS, kqs);
                kbx += IQ4_NL_BLOCKS_PER_ITER;
            }
        } else if FMT == FMT_MXFP4 {
            // Byte-addressed: a 17-byte block has no alignment.
            let row_base = (slot * n as usize + row0) * bpr * MXFP4_BYTES;
            let wp = w.as_ptr() as *const u8;
            let mut kbx = tid / IQ4_NL_THREADS_PER_BLOCK;
            let kqs = 2 * (tid % IQ4_NL_THREADS_PER_BLOCK);
            while kbx < bpr {
                tmp = acc_mxfp4::<FTZ>(tmp, wp.add(row_base + kbx * MXFP4_BYTES), x, x_base + kbx * Q8_WORDS, kqs);
                kbx += IQ4_NL_BLOCKS_PER_ITER;
            }
        } else if FMT == FMT_IQ4_NL {
            // Byte-addressed: an 18-byte block is only 2-byte aligned.
            let row_base = (slot * n as usize + row0) * bpr * IQ4_NL_BYTES;
            let wp = w.as_ptr() as *const u8;
            let mut kbx = tid / IQ4_NL_THREADS_PER_BLOCK;
            let kqs = 2 * (tid % IQ4_NL_THREADS_PER_BLOCK);
            while kbx < bpr {
                tmp = acc_iq4_nl::<FTZ>(tmp, wp.add(row_base + kbx * IQ4_NL_BYTES), x, x_base + kbx * Q8_WORDS, kqs);
                kbx += IQ4_NL_BLOCKS_PER_ITER;
            }
        } else if FMT == FMT_Q1_0 {
            // Byte-addressed: an 18-byte block is not word aligned.
            let row_base = (slot * n as usize + row0) * bpr * Q1_0_BYTES;
            let wp = w.as_ptr() as *const u8;
            let mut kbx = tid / Q1_0_THREADS_PER_BLOCK;
            let kqs = tid % Q1_0_THREADS_PER_BLOCK;
            while kbx < bpr {
                let xb = x_base + kbx * 4 * Q8_WORDS;
                tmp = acc_q1_0::<FTZ>(tmp, wp.add(row_base + kbx * Q1_0_BYTES), x, xb, kqs);
                kbx += Q1_0_BLOCKS_PER_ITER;
            }
        } else if FMT == FMT_Q6K {
            // Byte-addressed: a 210-byte block is not word aligned.
            let row_base = (slot * n as usize + row0) * bpr * Q6K_BYTES;
            let wp = w.as_ptr() as *const u8;
            let mut kbx = tid / Q6K_THREADS_PER_BLOCK;
            let kqs = tid % Q6K_THREADS_PER_BLOCK;
            while kbx < bpr {
                let xb = x_base + kbx * Q8_PER_KBLOCK * Q8_WORDS;
                tmp = acc_q6k::<FTZ>(tmp, wp.add(row_base + kbx * Q6K_BYTES), x, xb, kqs);
                kbx += Q6K_BLOCKS_PER_ITER;
            }
        } else {
            let words = if FMT == FMT_Q5K { Q5K_WORDS } else { Q4K_WORDS };
            let row_base = (slot * n as usize + row0) * bpr * words;
            let mut kbx = tid / THREADS_PER_BLOCK;
            let kqs = 2 * (tid % THREADS_PER_BLOCK);
            while kbx < bpr {
                let wb = row_base + kbx * words;
                let xb = x_base + kbx * Q8_PER_KBLOCK * Q8_WORDS;
                let r = if FMT == FMT_Q5K { dot_q5k::<FTZ>(w, wb, x, xb, kqs) } else { dot_q4k::<FTZ>(w, wb, x, xb, kqs) };
                tmp = add::<FTZ>(tmp, r);
                kbx += BLOCKS_PER_ITER;
            }
        }
        let (write, sum) = block_sum::<FTZ>(tmp, tid);
        if write {
            *out.add(task * n as usize + row0) = sum;
        }
    }

    /// `moe_gemv` for Q4_K / Q5_K with at most 2 K-quant blocks per row (k <= 512): there only warp 0
    /// of the reference's block has non-zero partials (thread t reads block t / 16), so warps 1..3 add
    /// +0.0 three times (which turns a -0 sum into +0, kept here) before warp 0's butterfly. Each warp
    /// runs that program for its own output row: 4 rows per CUDA block instead of 1, no shared memory.
    /// Launch: ceil(n / 4) * tasks blocks of 128 threads.
    /// SAFETY: as `moe_gemv`, and k <= 512.
    #[device]
    #[inline(always)]
    unsafe fn moe_gemv_warp_rows<const FTZ: bool, const FMT: u32>(
        w: &[u32], x: &[u32], ids: &[u32], slot_map: &[u32], out: *mut f32,
        n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32,
    ) {
        let g = thread::index_1d().get();
        let blk = g / THREADS;
        let lane = g % 32;
        let row_blocks = (n as usize).div_ceil(NWARPS);
        let row0 = (blk % row_blocks) * NWARPS + (g % THREADS) / 32;
        let task = blk / row_blocks;
        if row0 >= n as usize {
            return;
        }
        let input_row = if input_dim1 == 1 { task / topk as usize } else { task };
        let slot = *slot_map.get_unchecked(*ids.get_unchecked(task) as usize) as usize;
        if slot == NOT_RESIDENT {
            return;
        }
        let bpr = k as usize / 256;
        let x_base = input_row * (k_padded as usize / 32) * Q8_WORDS;
        let words = if FMT == FMT_Q5K { Q5K_WORDS } else { Q4K_WORDS };
        let row_base = (slot * n as usize + row0) * bpr * words;
        let mut tmp = 0f32;
        let mut kbx = lane / THREADS_PER_BLOCK;
        let kqs = 2 * (lane % THREADS_PER_BLOCK);
        while kbx < bpr {
            let wb = row_base + kbx * words;
            let xb = x_base + kbx * Q8_PER_KBLOCK * Q8_WORDS;
            let r = if FMT == FMT_Q5K { dot_q5k::<FTZ>(w, wb, x, xb, kqs) } else { dot_q4k::<FTZ>(w, wb, x, xb, kqs) };
            tmp = add::<FTZ>(tmp, r);
            kbx += BLOCKS_PER_ITER;
        }
        let mut l = 0;
        while l < NWARPS - 1 {
            tmp = add::<FTZ>(tmp, 0.0);
            l += 1;
        }
        let mut mask = 16;
        while mask > 0 {
            tmp = add::<FTZ>(tmp, warp::shuffle_xor_f32_sync(0xffff_ffff, tmp, mask));
            mask >>= 1;
        }
        if lane == 0 {
            *out.add(task * n as usize + row0) = tmp;
        }
    }

    /// Q4_K, candle's float program, k <= 512: one output row per warp (see `moe_gemv_warp_rows`).
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn q4k_q8_1_moe_gemv_w(w: &[u32], x: &[u32], ids: &[u32], slot_map: &[u32], out: *mut f32,
                                      n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) {
        moe_gemv_warp_rows::<false, FMT_Q4K>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1)
    }
    /// Q5_K, candle's float program, k <= 512: one output row per warp.
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn q5k_q8_1_moe_gemv_w(w: &[u32], x: &[u32], ids: &[u32], slot_map: &[u32], out: *mut f32,
                                      n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) {
        moe_gemv_warp_rows::<false, FMT_Q5K>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1)
    }

    /// Scatter CPU-computed expert rows into a GEMV output: `packed` = m task indices, then m rows of
    /// n f32 (bit patterns); `out[task_i * n + j] = row_i[j]`. One launch for all of a projection's
    /// CPU misses instead of one upload per row. Launch: ceil(m * n / 256) blocks of 256.
    /// SAFETY: every task index < tasks of `out` (tasks * n floats); packed holds m + m * n words.
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn scatter_rows(packed: &[u32], m: u32, out: *mut f32, n: u32) {
        let g = thread::index_1d().get();
        let (m, n) = (m as usize, n as usize);
        if g >= m * n {
            return;
        }
        let (i, j) = (g / n, g % n);
        let task = *packed.get_unchecked(i) as usize;
        *out.add(task * n + j) = f32::from_bits(*packed.get_unchecked(m + g));
    }

    /// Q4_K, candle's float program (what mistral.rs GGUF models run).
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn q4k_q8_1_moe_gemv(w: &[u32], x: &[u32], ids: &[u32], slot_map: &[u32], out: *mut f32,
                                    n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) {
        moe_gemv::<false, FMT_Q4K>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1)
    }
    /// Q5_K, candle's float program.
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn q5k_q8_1_moe_gemv(w: &[u32], x: &[u32], ids: &[u32], slot_map: &[u32], out: *mut f32,
                                    n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) {
        moe_gemv::<false, FMT_Q5K>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1)
    }
    /// Q6_K, candle's float program.
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn q6k_q8_1_moe_gemv(w: &[u32], x: &[u32], ids: &[u32], slot_map: &[u32], out: *mut f32,
                                    n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) {
        moe_gemv::<false, FMT_Q6K>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1)
    }
    /// Q1_0: llama.cpp's `mul_mat_vec_q<GGML_TYPE_Q1_0, 1, false>` float program (fast-math, the
    /// only one that exists for Q1_0), one expert row per CUDA block through the slot map.
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn q1_0_q8_1_moe_gemv(w: &[u32], x: &[u32], ids: &[u32], slot_map: &[u32], out: *mut f32,
                                     n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) {
        moe_gemv::<true, FMT_Q1_0>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1)
    }
    /// IQ4_NL: llama.cpp's `mul_mat_vec_q<GGML_TYPE_IQ4_NL, 1, false>` float program (fast-math).
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn iq4_nl_q8_1_moe_gemv(w: &[u32], x: &[u32], ids: &[u32], slot_map: &[u32], out: *mut f32,
                                       n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) {
        moe_gemv::<true, FMT_IQ4_NL>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1)
    }
    /// MXFP4: llama.cpp's `mul_mat_vec_q<GGML_TYPE_MXFP4, 1, false>` float program (fast-math).
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn mxfp4_q8_1_moe_gemv(w: &[u32], x: &[u32], ids: &[u32], slot_map: &[u32], out: *mut f32,
                                      n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) {
        moe_gemv::<true, FMT_MXFP4>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1)
    }
    /// NVFP4: llama.cpp's `mul_mat_vec_q<GGML_TYPE_NVFP4, 1, false>` float program (fast-math).
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn nvfp4_q8_1_moe_gemv(w: &[u32], x: &[u32], ids: &[u32], slot_map: &[u32], out: *mut f32,
                                      n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) {
        moe_gemv::<true, FMT_NVFP4>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1)
    }
    /// Q4_K, mistralrs-quant's (--use_fast_math) float program.
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn q4k_q8_1_moe_gemv_ftz(w: &[u32], x: &[u32], ids: &[u32], slot_map: &[u32], out: *mut f32,
                                        n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) {
        moe_gemv::<true, FMT_Q4K>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1)
    }
    /// Q5_K, mistralrs-quant's float program.
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn q5k_q8_1_moe_gemv_ftz(w: &[u32], x: &[u32], ids: &[u32], slot_map: &[u32], out: *mut f32,
                                        n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) {
        moe_gemv::<true, FMT_Q5K>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1)
    }
    /// Q6_K, mistralrs-quant's float program.
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn q6k_q8_1_moe_gemv_ftz(w: &[u32], x: &[u32], ids: &[u32], slot_map: &[u32], out: *mut f32,
                                        n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) {
        moe_gemv::<true, FMT_Q6K>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1)
    }
}

/// Raw launch of a kernel by name from PTX text; `args` are pointers to each argument value.
unsafe fn launch(module: &Arc<CudaModule>, name: &str, grid: (u32, u32, u32), block: (u32, u32, u32),
                 stream: &cuda_core::CudaStream, args: &mut [*mut c_void]) {
    let f = module.load_function(name).expect(name);
    cuda_core::simt::launch_kernel_on_stream(&f, grid, block, 0, stream, args).expect(name);
}

/// Deterministic pseudo-random stream.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f32(&mut self) -> f32 {
        ((self.next() >> 11) as f64 / (1u64 << 53) as f64 * 4.0 - 2.0) as f32
    }
}

struct Case {
    tensor: String,
    /// Weight bytes: a tensor file, or None for adversarial synthetic blocks (and Q8_1 input).
    file: Option<String>,
    /// Use only the first `max_experts` experts of the file (keeps VRAM small).
    max_experts: usize,
    fmt: cpu::Format,
    kernel_ref: &'static str,
    kernel_ox: &'static str,
    k: usize,
    n: usize,
}

impl Case {
    fn real(file: String, fmt: cpu::Format, k: usize, n: usize) -> Case {
        let tensor = std::path::Path::new(&file).file_stem().unwrap().to_string_lossy().into_owned();
        Case { tensor, file: Some(file), max_experts: usize::MAX, fmt, kernel_ref: ref_kernel(fmt), kernel_ox: ox_kernel(fmt), k, n }
    }
    fn adversarial(fmt: cpu::Format, k: usize, n: usize) -> Case {
        Case { tensor: format!("adversarial.{fmt:?}"), file: None, max_experts: usize::MAX, fmt, kernel_ref: ref_kernel(fmt), kernel_ox: ox_kernel(fmt), k, n }
    }
}

fn ref_kernel(fmt: cpu::Format) -> &'static str {
    match fmt {
        cpu::Format::Q4K => "indexed_moe_forward_q4k_q8_1",
        cpu::Format::Q5K => "indexed_moe_forward_q5k_q8_1",
        cpu::Format::Q6K => "indexed_moe_forward_q6k_q8_1",
        // llama.cpp mul_mat_vec_q<type, 1> (different ABI: launched by llama_gate, never by the main loop)
        cpu::Format::Q1_0 | cpu::Format::IQ4NL | cpu::Format::MXFP4 | cpu::Format::NVFP4 => "mul_mat_vec_q",
    }
}

fn ox_kernel(fmt: cpu::Format) -> &'static str {
    match fmt {
        cpu::Format::Q4K => "q4k_q8_1_moe_gemv",
        cpu::Format::Q5K => "q5k_q8_1_moe_gemv",
        cpu::Format::Q6K => "q6k_q8_1_moe_gemv",
        cpu::Format::Q1_0 => "q1_0_q8_1_moe_gemv",
        cpu::Format::IQ4NL => "iq4_nl_q8_1_moe_gemv",
        cpu::Format::MXFP4 => "mxfp4_q8_1_moe_gemv",
        cpu::Format::NVFP4 => "nvfp4_q8_1_moe_gemv",
    }
}

/// An adversarial f16: mostly random finite values of any exponent and sign, plus
/// subnormals, signed zeros, infinities and NaNs with random payloads.
fn adversarial_f16(r: &mut Rng) -> u16 {
    let p = r.next() % 1000;
    let sign = ((r.next() & 1) as u16) << 15;
    let man = (r.next() % 0x3FF) as u16 + 1;
    sign | match p {
        0..=2 => 0x7C00 | man,                                 // NaN
        3..=5 => 0x7C00,                                       // inf
        6..=199 => man,                                        // subnormal
        200..=249 => 0,                                        // zero
        _ => (((r.next() % 30) as u16 + 1) << 10) | (man - 1), // normal
    }
}

/// Random blocks of `fmt` with adversarial f16 scales (`d`, and `dmin` for Q4_K/Q5_K).
fn adversarial_weights(fmt: cpu::Format, blocks: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed);
    let bb = fmt.block_bytes();
    let mut b: Vec<u8> = (0..blocks * bb).map(|_| r.next() as u8).collect();
    let halves: &[usize] = match fmt {
        cpu::Format::Q6K => &[208],
        cpu::Format::Q1_0 | cpu::Format::IQ4NL => &[0],
        // the e byte is drawn above
        cpu::Format::MXFP4 => &[],
        // the 4 scale bytes stay uniformly random: every E4M3 byte (0x7F / 0xFF -> 0, sign bit)
        cpu::Format::NVFP4 => &[],
        _ => &[0, 2],
    };
    for blk in b.chunks_exact_mut(bb) {
        if fmt == cpu::Format::MXFP4 {
            // E8M0: 1% NaN (255), 40% tiny exponents (flushed / subnormal products and sums),
            // 30% near 1.0, the rest anything else.
            let p = r.next() % 100;
            blk[0] = match p {
                0 => 255,
                1..=40 => (r.next() % 12) as u8,
                41..=70 => 120 + (r.next() % 16) as u8,
                _ => (r.next() % 255) as u8,
            };
        }
        for &o in halves {
            blk[o..o + 2].copy_from_slice(&adversarial_f16(&mut r).to_le_bytes());
        }
    }
    b
}

/// Q8_1 rows of random int8 quants (full range, -128 included) and adversarial d.
fn adversarial_q8(words: usize, seed: u64) -> Vec<u32> {
    let mut r = Rng(seed);
    let mut w: Vec<u32> = (0..words).map(|_| r.next() as u32).collect();
    for blk in w.chunks_exact_mut(9) {
        blk[0] = adversarial_f16(&mut r) as u32 | ((r.next() as u32) << 16);
    }
    w
}

fn main() {
    let root = std::env::var("M0_ROOT").unwrap_or_else(|_| format!("{}/titan-engine/m0", std::env::var("HOME").unwrap()));
    let ctx = CudaContext::new(0).expect("cuda context");
    let stream = ctx.default_stream();
    // (label, reference PTX, oxide kernel suffix): mistralrs-quant's fast-math build and candle's.
    let refs = [("mistralrs-quant", format!("{root}/ref/indexed_moe.ptx"), "_ftz"),
                ("candle", format!("{root}/../m1/ref/indexed_moe.ptx"), "")];
    // The PTX this build just wrote (OXIDE_PTX overrides, e.g. m0/oxide/titan_kernels_oxide.ptx).
    let ptx = std::env::var("OXIDE_PTX").unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/titan_kernels_oxide.ptx").to_string());
    let oxide = ctx.load_module_from_ptx_src(&std::fs::read_to_string(&ptx).unwrap()).expect("oxide ptx");
    // Q6_K tensors: extract_q6k.py (Qwen3.6-35B-A3B UD-Q4_K_XL ffn_down_exps, K=512, 256 experts).
    let q6k_data = std::env::var("Q6K_DATA").unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/data").to_string());
    // CASES=q6k runs only the Q6_K cases.
    let only = std::env::var("CASES").ok();

    use cpu::Format::{Q4K, Q5K, Q6K};
    let mut cases: Vec<Case> = vec![
        Case::real(format!("{root}/data/blk.0.ffn_gate_exps.q4k"), Q4K, 2048, 768),
        Case::real(format!("{root}/data/blk.0.ffn_down_exps.q4k"), Q4K, 768, 2048),
        Case::real(format!("{root}/data/blk.1.ffn_down_exps.q5k"), Q5K, 768, 2048),
    ];
    // MAX_EXPERTS caps the Q4_K/Q5_K files too (default: all 128) when VRAM is short.
    if let Some(m) = std::env::var("MAX_EXPERTS").ok().and_then(|v| v.parse().ok()) {
        cases.iter_mut().for_each(|c| c.max_experts = m);
    }
    // Q6K_EXPERTS (default 64 of 256, 55 MB per copy) bounds the VRAM these cases take.
    let q6k_experts = std::env::var("Q6K_EXPERTS").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
    for layer in [1, 34, 38, 39] {
        let mut c = Case::real(format!("{q6k_data}/blk.{layer}.ffn_down_exps.q6k"), Q6K, 512, 2048);
        c.max_experts = q6k_experts;
        cases.push(c);
    }
    // K=2048 gives two K-quant blocks per thread for Q6_K (the accumulating fma), K=768 a partial warp.
    for fmt in [Q4K, Q5K, Q6K] {
        cases.push(Case::adversarial(fmt, 2048, 64));
        cases.push(Case::adversarial(fmt, 768, 64));
        // k <= 512: the one-row-per-warp kernels (Q4_K / Q5_K), Qwen3.6's expert down projections
        cases.push(Case::adversarial(fmt, 512, 2048));
        cases.push(Case::adversarial(fmt, 256, 64));
    }
    if let Some(o) = &only {
        cases.retain(|c| c.tensor.to_lowercase().contains(o.as_str()) || format!("{:?}", c.fmt).to_lowercase() == *o);
    }
    let mut per_fmt: std::collections::BTreeMap<String, [usize; 4]> = Default::default();
    let mut all_ok = true;
    let mut total = 0usize;
    let mut cpu_total = 0usize;

    // Quantizer gate against candle's quantize_q8_1 (the one the stock GGUF path runs).
    {
        let candle = ctx.load_module_from_ptx_src(&std::fs::read_to_string(&refs[1].1).unwrap()).expect("candle ptx");
        let mut qbytes = 0usize;
        let mut qbad = 0usize;
        for &(rows, k) in &[(1usize, 2048usize), (7, 768), (3, 2048), (64, 768)] {
            let kp = k.div_ceil(512) * 512;
            let mut r = Rng(rows as u64 * 31 + k as u64);
            let mut xh: Vec<f32> = (0..rows * k).map(|_| r.f32()).collect();
            // Edge rows: all zero (amax == 0), tiny values, exact half-way ties.
            if rows >= 3 {
                for v in &mut xh[0..k] { *v = 0.0; }
                for (i, v) in xh[k..2 * k].iter_mut().enumerate() { *v = (i as f32 - 300.0) * 1e-30; }
                for (i, v) in xh[2 * k..3 * k].iter_mut().enumerate() { *v = ((i % 64) as f32 - 32.0) * 0.5; }
            }
            let xf = DeviceBuffer::from_host(&stream, &xh).unwrap();
            let yr = DeviceBuffer::<u32>::zeroed(&stream, rows * kp / 32 * 9).unwrap();
            let yo = DeviceBuffer::<u32>::zeroed(&stream, rows * kp / 32 * 9).unwrap();
            unsafe {
                let (mut px, mut pr, mut po) = (xf.cu_deviceptr(), yr.cu_deviceptr(), yo.cu_deviceptr());
                let (mut ki, mut kpi) = (k as i32, kp as i32);
                launch(&candle, "quantize_q8_1", ((kp as u32).div_ceil(256), rows as u32, 1), (256, 1, 1), &stream,
                    &mut [&raw mut px as _, &raw mut pr as _, &raw mut ki as _, &raw mut kpi as _]);
                let mut xl = xh.len() as u64;
                let (mut ku, mut kpu) = (k as u32, kp as u32);
                launch(&oxide, "quantize_q8_1", (((rows * kp) as u32).div_ceil(256), 1, 1), (256, 1, 1), &stream,
                    &mut [&raw mut px as _, &raw mut xl as _, &raw mut po as _, &raw mut ku as _, &raw mut kpu as _]);
            }
            stream.synchronize().unwrap();
            let (a, b) = (yr.to_host_vec(&stream).unwrap(), yo.to_host_vec(&stream).unwrap());
            let bad = a.iter().zip(&b).filter(|(x, y)| x != y).count();
            println!("quantize_q8_1 rows={rows} k={k}: {} words, {bad} differ", a.len());
            qbytes += a.len() * 4;
            qbad += bad;
        }
        println!("quantizer: {qbytes} bytes, {qbad} differing words");
        all_ok &= qbad == 0;
    }

    // scatter_rows (a copy): every scattered row lands bit-exact, untouched rows keep their bytes.
    {
        let (tasks, n) = (24usize, 2048usize);
        let mut r = Rng(0x5CA7);
        let base: Vec<u32> = (0..tasks * n).map(|_| r.next() as u32).collect();
        let picked: Vec<u32> = vec![3, 0, 17, 9, 23];
        let m = picked.len();
        let mut packed = picked.clone();
        packed.extend((0..m * n).map(|_| r.next() as u32));
        let mut want = base.clone();
        for (i, &t) in picked.iter().enumerate() {
            want[t as usize * n..(t as usize + 1) * n].copy_from_slice(&packed[m + i * n..m + (i + 1) * n]);
        }
        let out = DeviceBuffer::from_host(&stream, &base).unwrap();
        let pk = DeviceBuffer::from_host(&stream, &packed).unwrap();
        unsafe {
            let (mut pp, mut pl, mut mu, mut po, mut nu) = (pk.cu_deviceptr(), packed.len() as u64, m as u32, out.cu_deviceptr(), n as u32);
            launch(&oxide, "scatter_rows", (((m * n) as u32).div_ceil(256), 1, 1), (256, 1, 1), &stream,
                &mut [&raw mut pp as _, &raw mut pl as _, &raw mut mu as _, &raw mut po as _, &raw mut nu as _]);
        }
        stream.synchronize().unwrap();
        let got = out.to_host_vec(&stream).unwrap();
        let bad = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        println!("scatter_rows: {} words, {bad} differ", got.len());
        all_ok &= bad == 0;
    }

    for (label, ref_path, suffix) in &refs {
    let reference = ctx.load_module_from_ptx_src(&std::fs::read_to_string(ref_path).unwrap()).expect("reference ptx");
    println!("== reference: {label}");
    for c in &cases {
        let kernel_ox = format!("{}{}", c.kernel_ox, suffix);
        let expert_bytes = c.n * (c.k / 256) * c.fmt.block_bytes();
        let bytes = match &c.file {
            Some(f) => {
                use std::io::Read;
                let mut v = Vec::new();
                let cap = (c.max_experts as u64).saturating_mul(expert_bytes as u64);
                std::fs::File::open(f).unwrap().take(cap).read_to_end(&mut v).unwrap();
                v
            }
            None => adversarial_weights(c.fmt, 16 * expert_bytes / c.fmt.block_bytes(), c.k as u64 * 1_000_003 + c.fmt.block_bytes() as u64),
        };
        let experts = bytes.len() / expert_bytes;
        assert_eq!(bytes.len() % 4, 0);
        let words: Vec<u32> = bytes.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let w_ref = DeviceBuffer::from_host(&stream, &words).unwrap();

        // The oxide side reads a *permuted* slot buffer through a slot map, so the
        // test also proves the indirection: expert e lives in slot perm[e].
        let mut rng = Rng(0xC0FFEE);
        let mut perm: Vec<u32> = (0..experts as u32).collect();
        for i in (1..perm.len()).rev() {
            perm.swap(i, (rng.next() % (i as u64 + 1)) as usize);
        }
        // Byte-granular: a Q6_K expert need not be a whole number of words.
        let mut slotted_b = vec![0u8; bytes.len()];
        for e in 0..experts {
            let s = perm[e] as usize;
            slotted_b[s * expert_bytes..(s + 1) * expert_bytes].copy_from_slice(&bytes[e * expert_bytes..(e + 1) * expert_bytes]);
        }
        let slotted: Vec<u32> = slotted_b.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        drop(slotted_b);
        let w_ox = DeviceBuffer::from_host(&stream, &slotted).unwrap();
        let slot_map = DeviceBuffer::from_host(&stream, &perm).unwrap();
        let k_padded = c.k.div_ceil(512) * 512;

        // (batch, topk, input_dim1): decode (1 token, all experts), a prefill-like
        // batch sharing input across its top-k, and per-task inputs (the down projection's shape).
        for &(batch, topk, input_dim1) in &[(1usize, experts, 1u32), (5, 8, 1), (5, 8, 0)] {
            for seed in 1..=4u64 {
                let tasks = batch * topk;
                let in_rows = if input_dim1 == 1 { batch } else { tasks };
                let mut r = Rng(seed * 7919 + batch as u64);
                let xh: Vec<f32> = (0..in_rows * c.k).map(|_| r.f32()).collect();
                let ids: Vec<u32> = if topk == experts { (0..experts as u32).collect() }
                                    else { (0..tasks).map(|_| (r.next() % experts as u64) as u32).collect() };
                let xf = DeviceBuffer::from_host(&stream, &xh).unwrap();
                let ids_d = DeviceBuffer::from_host(&stream, &ids).unwrap();
                let xq = match c.file {
                    Some(_) => DeviceBuffer::<u32>::zeroed(&stream, in_rows * k_padded / 32 * 9).unwrap(),
                    None => DeviceBuffer::from_host(&stream, &adversarial_q8(in_rows * k_padded / 32 * 9, seed * 104_729 + batch as u64)).unwrap(),
                };
                let out_ref = DeviceBuffer::<f32>::zeroed(&stream, tasks * c.n).unwrap();
                let out_ox = DeviceBuffer::<f32>::zeroed(&stream, tasks * c.n).unwrap();
                unsafe {
                    let (mut px, mut pq) = (xf.cu_deviceptr(), xq.cu_deviceptr());
                    let (mut ki, mut kp) = (c.k as i32, k_padded as i32);
                    if c.file.is_some() {
                        launch(&reference, "quantize_q8_1", ((k_padded as u32).div_ceil(256), in_rows as u32, 1), (256, 1, 1), &stream,
                            &mut [&raw mut px as _, &raw mut pq as _, &raw mut ki as _, &raw mut kp as _]);
                    }
                    let (mut pw, mut pi, mut po) = (w_ref.cu_deviceptr(), ids_d.cu_deviceptr(), out_ref.cu_deviceptr());
                    let (mut ni, mut bi, mut ti, mut d1) = (c.n as i32, batch as i32, topk as i32, input_dim1 as i32);
                    launch(&reference, c.kernel_ref, (c.n as u32, batch as u32, topk as u32), (32, 4, 1), &stream,
                        &mut [&raw mut pw as _, &raw mut pq as _, &raw mut pi as _, &raw mut po as _, &raw mut ni as _,
                              &raw mut ki as _, &raw mut bi as _, &raw mut ti as _, &raw mut kp as _, &raw mut d1 as _]);
                    // Oxide ABI: each slice is (ptr, len).
                    let (mut pwo, mut wl) = (w_ox.cu_deviceptr(), slotted.len() as u64);
                    let mut xl = (in_rows * k_padded / 32 * 9) as u64;
                    let mut il = ids.len() as u64;
                    let (mut ps, mut sl) = (slot_map.cu_deviceptr(), perm.len() as u64);
                    let mut po2 = out_ox.cu_deviceptr();
                    let (mut nu, mut ku, mut kpu, mut tu, mut du) = (c.n as u32, c.k as u32, k_padded as u32, topk as u32, input_dim1);
                    launch(&oxide, &kernel_ox, ((c.n * tasks) as u32, 1, 1), (128, 1, 1), &stream,
                        &mut [&raw mut pwo as _, &raw mut wl as _, &raw mut pq as _, &raw mut xl as _, &raw mut pi as _,
                              &raw mut il as _, &raw mut ps as _, &raw mut sl as _, &raw mut po2 as _, &raw mut nu as _,
                              &raw mut ku as _, &raw mut kpu as _, &raw mut tu as _, &raw mut du as _]);
                }
                // k <= 512 Q4_K / Q5_K: the one-row-per-warp variant must equal the reference too.
                let warp_rows = suffix.is_empty() && c.k <= 512 && matches!(c.fmt, Q4K | Q5K);
                let out_w = DeviceBuffer::<f32>::zeroed(&stream, tasks * c.n).unwrap();
                if warp_rows {
                    unsafe {
                        let (mut pwo, mut wl) = (w_ox.cu_deviceptr(), slotted.len() as u64);
                        let mut pq = xq.cu_deviceptr();
                        let mut xl = (in_rows * k_padded / 32 * 9) as u64;
                        let (mut pi, mut il) = (ids_d.cu_deviceptr(), ids.len() as u64);
                        let (mut ps, mut sl) = (slot_map.cu_deviceptr(), perm.len() as u64);
                        let mut pw3 = out_w.cu_deviceptr();
                        let (mut nu, mut ku, mut kpu, mut tu, mut du) = (c.n as u32, c.k as u32, k_padded as u32, topk as u32, input_dim1);
                        launch(&oxide, &format!("{kernel_ox}_w"), ((c.n.div_ceil(4) * tasks) as u32, 1, 1), (128, 1, 1), &stream,
                            &mut [&raw mut pwo as _, &raw mut wl as _, &raw mut pq as _, &raw mut xl as _, &raw mut pi as _,
                                  &raw mut il as _, &raw mut ps as _, &raw mut sl as _, &raw mut pw3 as _, &raw mut nu as _,
                                  &raw mut ku as _, &raw mut kpu as _, &raw mut tu as _, &raw mut du as _]);
                    }
                }
                stream.synchronize().unwrap();
                let (a, b) = (out_ref.to_host_vec(&stream).unwrap(), out_ox.to_host_vec(&stream).unwrap());
                let diff = a.iter().zip(&b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
                if warp_rows {
                    let bw = out_w.to_host_vec(&stream).unwrap();
                    let wdiff = a.iter().zip(&bw).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
                    let pf = per_fmt.entry(format!("{:?} {label} one-row-per-warp", c.fmt)).or_default();
                    pf[0] += a.len();
                    pf[1] += a.len() - wdiff;
                    if seed == 1 || wdiff > 0 {
                        println!("    {kernel_ox}_w: {} outputs, {wdiff} mismatches", a.len());
                    }
                    all_ok &= wdiff == 0;
                }
                let nonzero = a.iter().filter(|v| **v != 0.0).count();
                let nan = a.iter().filter(|v| v.is_nan()).count();
                total += a.len();
                let pf = per_fmt.entry(format!("{:?} {label}", c.fmt)).or_default();
                pf[0] += a.len();
                pf[1] += a.len() - diff;
                if seed == 1 || diff > 0 {
                    println!("{} k={} n={} batch={batch} topk={topk} input_dim1={input_dim1} seed={seed}: {} outputs, {nonzero} nonzero, {nan} NaN, {diff} mismatches",
                             c.tensor, c.k, c.n, a.len());
                }
                all_ok &= diff == 0 && nonzero > a.len() / 2;
                // CPU twin vs the candle-program GPU output (suffix "" = non-ftz), sampled tasks.
                if suffix.is_empty() {
                    let xqh: Vec<u32> = xq.to_host_vec(&stream).unwrap();
                    let xqb: Vec<u8> = xqh.iter().flat_map(|w| w.to_le_bytes()).collect();
                    let fmt = c.fmt;
                    let row_bytes = (c.k / 256) * c.fmt.block_bytes();
                    let q8_row = k_padded / 32 * 36;
                    let mut cbad = 0usize;
                    let mut cn = 0usize;
                    let (mut t_scalar, mut t_simd) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
                    for task in (0..tasks).step_by(if tasks > 64 { 7 } else { 1 }) {
                        let e = ids[task] as usize;
                        let in_row = if input_dim1 == 1 { task / topk } else { task };
                        let xrow = &xqb[in_row * q8_row..];
                        let t0 = std::time::Instant::now();
                        for r in 0..c.n {
                            let w_row = &bytes[e * expert_bytes + r * row_bytes..][..row_bytes];
                            let v = cpu::row(fmt, w_row, xrow, c.k);
                            cbad += (v.to_bits() != a[task * c.n + r].to_bits()) as usize;
                            cn += 1;
                        }
                        t_scalar += t0.elapsed();
                        let t0 = std::time::Instant::now();
                        for r0 in (0..c.n).step_by(8) {
                            let rows: [&[u8]; 8] = std::array::from_fn(|i| &bytes[e * expert_bytes + (r0 + i) * row_bytes..][..row_bytes]);
                            let v = unsafe { cpu::rows8(fmt, &rows, xrow, c.k) };
                            for i in 0..8 {
                                cbad += (v[i].to_bits() != a[task * c.n + r0 + i].to_bits()) as usize;
                            }
                            cn += 8;
                        }
                        t_simd += t0.elapsed();
                        if matches!(fmt, Q4K | Q5K | Q6K) {
                            let rows: Vec<&[u8]> = (0..c.n).map(|r| &bytes[e * expert_bytes + r * row_bytes..][..row_bytes]).collect();
                            let mut v = vec![0f32; c.n];
                            unsafe { cpu::rows_lanes(fmt, &rows, xrow, c.k, &mut v) };
                            for r in 0..c.n {
                                cbad += (v[r].to_bits() != a[task * c.n + r].to_bits()) as usize;
                            }
                            cn += c.n;
                        }
                    }
                    if seed == 1 {
                        println!("    cpu twin timing: scalar {:?}, avx2 x8 {:?}", t_scalar, t_simd);
                    }
                    if seed == 1 || cbad > 0 {
                        println!("    cpu twin: {cn} rows, {cbad} mismatches vs GPU");
                    }
                    cpu_total += cn;
                    let pf = per_fmt.entry(format!("{:?} cpu-twin", c.fmt)).or_default();
                    pf[2] += cn;
                    pf[3] += cn - cbad;
                    all_ok &= cbad == 0;
                }
            }
        }
    }
    }
    for spec in llama_specs() {
        if only.as_deref().is_none_or(|o| o == spec.name) {
            all_ok &= llama_gate(&spec, &ctx, &stream, &oxide, &mut per_fmt, &mut total, &mut cpu_total);
        }
    }
    for (k, v) in &per_fmt {
        if k.ends_with("cpu-twin") {
            println!("{k}: {} / {} rows (scalar + avx2 rows8 + avx2 rows_lanes) bit-identical to the GPU", v[3], v[2]);
        } else {
            println!("{k}: {} / {} outputs bit-identical to the reference", v[1], v[0]);
        }
    }
    println!("{total} outputs compared, {cpu_total} CPU-twin rows");
    println!("{}", if all_ok { "KERNEL GATE: PASS (bit-identical, Q4_K + Q5_K + Q6_K + Q1_0 + IQ4_NL + MXFP4 + NVFP4, permuted slots)" } else { "KERNEL GATE: FAIL" });
    std::process::exit(if all_ok { 0 } else { 1 });
}

/// llama.cpp `init_fastdiv_values`: <mp, L, d>.
fn fastdiv_values(d: u32) -> [u32; 3] {
    let mut l = 0u32;
    while l < 32 && (1u64 << l) < d as u64 {
        l += 1;
    }
    [(((1u64 << 32) * ((1u64 << l) - d as u64)) / d as u64 + 1) as u32, l, d]
}

/// Launch from a packed parameter buffer (CU_LAUNCH_PARAM_BUFFER_POINTER) on the null stream.
unsafe fn launch_packed(f: cuda_core::sys::CUfunction, grid: (u32, u32, u32), block: (u32, u32, u32), params: &mut Vec<u8>) {
    use cuda_core::sys as cu;
    let mut size = params.len();
    let mut extra: [*mut c_void; 5] =
        [1 as *mut c_void, params.as_mut_ptr() as *mut c_void, 2 as *mut c_void, &mut size as *mut usize as *mut c_void, std::ptr::null_mut()];
    let r = unsafe {
        cu::cuLaunchKernel(f, grid.0, grid.1, grid.2, block.0, block.1, block.2, 0, std::ptr::null_mut(), std::ptr::null_mut(), extra.as_mut_ptr())
    };
    assert_eq!(r, cu::cudaError_enum_CUDA_SUCCESS, "launch");
}

fn put<T: Copy>(b: &mut Vec<u8>, v: T) {
    let a = std::mem::align_of::<T>();
    while b.len() % a != 0 {
        b.push(0);
    }
    b.extend_from_slice(unsafe { std::slice::from_raw_parts(&v as *const T as *const u8, std::mem::size_of::<T>()) });
}

/// A GGML format whose only CUDA float program is llama.cpp's `mul_mat_vec_q`.
struct LlamaSpec {
    fmt: cpu::Format,
    /// oxide-kernels crate holding the reference cubin (ref/mmvq.cubin) and the extract script's name
    name: &'static str,
    ggml_type: u32,
    /// VRAM cap for the real-tensor cases: env var and default expert count
    env: &'static str,
    /// real tensors (file stem under data/, k, n) from extract_<name>.py
    files: &'static [(&'static str, usize, usize)],
    adversarial: &'static [(usize, usize)],
}

const QWEN3_EXPS: &[(&str, usize, usize)] = &[
    ("blk.0.ffn_gate_exps", 2048, 768),
    ("blk.0.ffn_down_exps", 768, 2048),
    ("blk.23.ffn_up_exps", 2048, 768),
];

fn llama_specs() -> Vec<LlamaSpec> {
    vec![
        // 4096 = 32 blocks (one full iteration per thread), 384 = 3 blocks (most threads idle), 12288 = 3 iterations.
        LlamaSpec { fmt: cpu::Format::Q1_0, name: "q1_0", ggml_type: 41, env: "Q1_0_EXPERTS", files: QWEN3_EXPS,
                    adversarial: &[(2048, 64), (384, 64), (12288, 40), (4096, 33)] },
        // 2048 = 64 blocks (one iteration), 384 = 12 blocks, 12288 = 6 iterations, 4128 = a ragged 2nd iteration.
        LlamaSpec { fmt: cpu::Format::IQ4NL, name: "iq4_nl", ggml_type: 20, env: "IQ4_NL_EXPERTS", files: QWEN3_EXPS,
                    adversarial: &[(2048, 64), (384, 64), (12288, 40), (4128, 33)] },
        LlamaSpec { fmt: cpu::Format::MXFP4, name: "mxfp4", ggml_type: 39, env: "MXFP4_EXPERTS", files: QWEN3_EXPS,
                    adversarial: &[(2048, 64), (384, 64), (12288, 40), (4128, 33)] },
        // 64-value blocks: 4096 = 64 blocks (one iteration), 384 = 6, 12288 = 3 iterations, 8256 = a ragged 3rd.
        LlamaSpec { fmt: cpu::Format::NVFP4, name: "nvfp4", ggml_type: 40, env: "NVFP4_EXPERTS", files: QWEN3_EXPS,
                    adversarial: &[(4096, 64), (384, 64), (12288, 40), (8256, 33)] },
    ]
}

/// Expert GEMV gate for a llama.cpp-kernel format. Reference: llama.cpp acecd56's own
/// `mul_mat_vec_q<type, 1, false, false, false>` (the MUL_MAT_ID decode kernel: channel = task,
/// `ids` = experts), from the nvcc rebuild in oxide-kernels/<name>/ref/mmvq.cubin (SASS identical
/// to libggml-cuda.so), launched once per token. Oxide reads permuted slots through the slot map.
/// The CPU twin (scalar and AVX2) is checked against the oxide GPU output.
fn llama_gate(spec: &LlamaSpec, ctx: &Arc<CudaContext>, stream: &Arc<cuda_core::CudaStream>, oxide: &Arc<CudaModule>,
              per_fmt: &mut std::collections::BTreeMap<String, [usize; 4]>, total: &mut usize, cpu_total: &mut usize) -> bool {
    use cuda_core::sys as cu;
    let home = std::env::var("HOME").unwrap();
    let cubin = format!("{home}/titan-engine/oxide-kernels/{}/ref/mmvq.cubin", spec.name);
    let reference = ctx.load_module_from_file(&cubin).unwrap_or_else(|e| panic!("{cubin}: {e:?}"));
    let rf_fn = reference
        .load_function(&format!("_Z13mul_mat_vec_qIL9ggml_type{}ELi1ELb0ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj", spec.ggml_type))
        .expect("reference mul_mat_vec_q<type, 1>");
    let rf = unsafe { rf_fn.cu_function() };
    let data = concat!(env!("CARGO_MANIFEST_DIR"), "/data");
    let fmt = spec.fmt;
    let (qk, bb) = (fmt.block_values(), fmt.block_bytes());
    let max_experts = std::env::var(spec.env).ok().and_then(|v| v.parse().ok()).unwrap_or(48usize);
    // extract_<name>.py: the requantized qwen3coder30b-first24 experts (128 x [768, 2048] / [2048, 768]).
    let mut cases: Vec<Case> = spec.files.iter().map(|&(f, k, n)| Case::real(format!("{data}/{f}.{}", spec.name), fmt, k, n)).collect();
    cases.iter_mut().for_each(|c| c.max_experts = max_experts);
    for &(k, n) in spec.adversarial {
        cases.push(Case::adversarial(fmt, k, n));
    }
    let mut ok = true;
    println!("== reference: llama.cpp mul_mat_vec_q<{:?}> ({} gate)", fmt, spec.name);
    for c in &cases {
        let expert_bytes = c.n * (c.k / qk) * bb;
        let bytes = match &c.file {
            Some(f) => {
                use std::io::Read;
                let mut v = Vec::new();
                std::fs::File::open(f).unwrap_or_else(|e| panic!("{f}: {e} (run extract_{}.py)", spec.name))
                    .take((c.max_experts * expert_bytes) as u64).read_to_end(&mut v).unwrap();
                v
            }
            None => adversarial_weights(fmt, 12 * c.n * c.k / qk, c.k as u64 * 7_000_003 + c.n as u64),
        };
        let experts = bytes.len() / expert_bytes;
        let mut padded = bytes.clone();
        while padded.len() % 4 != 0 {
            padded.push(0);
        }
        let words: Vec<u32> = padded.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let w_ref = DeviceBuffer::from_host(stream, &words).unwrap();
        let mut rng = Rng(0xC0FFEE ^ c.k as u64);
        let mut perm: Vec<u32> = (0..experts as u32).collect();
        for i in (1..perm.len()).rev() {
            perm.swap(i, (rng.next() % (i as u64 + 1)) as usize);
        }
        let mut slotted_b = vec![0u8; padded.len()];
        for e in 0..experts {
            let s = perm[e] as usize;
            slotted_b[s * expert_bytes..(s + 1) * expert_bytes].copy_from_slice(&bytes[e * expert_bytes..(e + 1) * expert_bytes]);
        }
        let slotted: Vec<u32> = slotted_b.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let w_ox = DeviceBuffer::from_host(stream, &slotted).unwrap();
        let slot_map = DeviceBuffer::from_host(stream, &perm).unwrap();
        let k_padded = c.k.div_ceil(512) * 512;
        let q8_row_words = k_padded / 32 * 9;
        let bpr = (c.k / qk) as u32;
        for &(batch, topk, input_dim1) in &[(1usize, experts, 1u32), (5, 8, 1), (5, 8, 0), (3, 6, 0)] {
            for seed in 1..=3u64 {
                let tasks = batch * topk;
                let in_rows = if input_dim1 == 1 { batch } else { tasks };
                let mut r = Rng(seed * 7919 + batch as u64 + c.k as u64);
                let ids: Vec<u32> = if batch == 1 && topk == experts { (0..experts as u32).collect() } else { (0..tasks).map(|_| (r.next() % experts as u64) as u32).collect() };
                let ids_d = DeviceBuffer::from_host(stream, &ids).unwrap();
                let xq = match c.file {
                    Some(_) => {
                        let xh: Vec<f32> = (0..in_rows * c.k).map(|_| r.f32()).collect();
                        let xf = DeviceBuffer::from_host(stream, &xh).unwrap();
                        let xq = DeviceBuffer::<u32>::zeroed(stream, in_rows * q8_row_words).unwrap();
                        unsafe {
                            let (mut px, mut xl, mut pq) = (xf.cu_deviceptr(), xh.len() as u64, xq.cu_deviceptr());
                            let (mut ku, mut kpu) = (c.k as u32, k_padded as u32);
                            launch(oxide, "quantize_q8_1", (((in_rows * k_padded) as u32).div_ceil(256), 1, 1), (256, 1, 1), stream,
                                &mut [&raw mut px as _, &raw mut xl as _, &raw mut pq as _, &raw mut ku as _, &raw mut kpu as _]);
                        }
                        stream.synchronize().unwrap();
                        xq
                    }
                    None => DeviceBuffer::from_host(stream, &adversarial_q8(in_rows * q8_row_words, seed * 104_729 + batch as u64 + c.k as u64)).unwrap(),
                };
                let out_ref = DeviceBuffer::<f32>::zeroed(stream, tasks * c.n).unwrap();
                let out_ox = DeviceBuffer::<f32>::zeroed(stream, tasks * c.n).unwrap();
                stream.synchronize().unwrap();
                unsafe {
                    // reference: one launch per token, channel = task within the token.
                    let one = fastdiv_values(1);
                    let ncy = fastdiv_values(if input_dim1 == 1 { 1 } else { topk as u32 });
                    for b in 0..batch {
                        let mut p = Vec::new();
                        put(&mut p, w_ref.cu_deviceptr());
                        let y_row = if input_dim1 == 1 { b } else { b * topk };
                        put(&mut p, xq.cu_deviceptr() + (y_row * q8_row_words * 4) as u64);
                        put(&mut p, ids_d.cu_deviceptr() + (b * topk * 4) as u64);
                        for _ in 0..5 {
                            put(&mut p, 0u64);
                        }
                        put(&mut p, 0u32);
                        put(&mut p, 0f32);
                        put(&mut p, out_ref.cu_deviceptr() + (b * topk * c.n * 4) as u64);
                        for v in [c.k as u32, ncy[0], ncy[1], ncy[2], bpr, (k_padded / 32) as u32, c.n as u32, 0, 0, 0,
                                  c.n as u32 * bpr, (k_padded / 32) as u32, c.n as u32, one[0], one[1], one[2], 0, 0, 0, 0] {
                            put(&mut p, v);
                        }
                        assert_eq!(p.len(), 160);
                        launch_packed(rf, (c.n as u32, topk as u32, 1), (32, 4, 1), &mut p);
                    }
                    assert_eq!(cu::cuCtxSynchronize(), cu::cudaError_enum_CUDA_SUCCESS);
                    let (mut pwo, mut wl) = (w_ox.cu_deviceptr(), slotted.len() as u64);
                    let (mut pq, mut xl) = (xq.cu_deviceptr(), (in_rows * q8_row_words) as u64);
                    let (mut pi, mut il) = (ids_d.cu_deviceptr(), ids.len() as u64);
                    let (mut ps, mut sl) = (slot_map.cu_deviceptr(), perm.len() as u64);
                    let mut po = out_ox.cu_deviceptr();
                    let (mut nu, mut ku, mut kpu, mut tu, mut du) = (c.n as u32, c.k as u32, k_padded as u32, topk as u32, input_dim1);
                    launch(oxide, ox_kernel(fmt), ((c.n * tasks) as u32, 1, 1), (128, 1, 1), stream,
                        &mut [&raw mut pwo as _, &raw mut wl as _, &raw mut pq as _, &raw mut xl as _, &raw mut pi as _,
                              &raw mut il as _, &raw mut ps as _, &raw mut sl as _, &raw mut po as _, &raw mut nu as _,
                              &raw mut ku as _, &raw mut kpu as _, &raw mut tu as _, &raw mut du as _]);
                }
                stream.synchronize().unwrap();
                let (a, b) = (out_ref.to_host_vec(stream).unwrap(), out_ox.to_host_vec(stream).unwrap());
                let diff = a.iter().zip(&b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
                let nonzero = a.iter().filter(|v| **v != 0.0).count();
                let nan = a.iter().filter(|v| v.is_nan()).count();
                *total += a.len();
                let pf = per_fmt.entry(format!("{fmt:?} llama.cpp")).or_default();
                pf[0] += a.len();
                pf[1] += a.len() - diff;
                if seed == 1 || diff > 0 {
                    println!("{} k={} n={} batch={batch} topk={topk} input_dim1={input_dim1} seed={seed}: {} outputs, {nonzero} nonzero, {nan} NaN, {diff} mismatches",
                             c.tensor, c.k, c.n, a.len());
                }
                ok &= diff == 0 && nonzero > a.len() / 2;
                // CPU twin vs the GPU output.
                let xqh: Vec<u32> = xq.to_host_vec(stream).unwrap();
                let xqb: Vec<u8> = xqh.iter().flat_map(|w| w.to_le_bytes()).collect();
                let row_bytes = (c.k / qk) * bb;
                let q8_row = k_padded / 32 * 36;
                let (mut cbad, mut cn) = (0usize, 0usize);
                for task in (0..tasks).step_by(if tasks > 64 { 5 } else { 1 }) {
                    let e = ids[task] as usize;
                    let in_row = if input_dim1 == 1 { task / topk } else { task };
                    let xrow = &xqb[in_row * q8_row..];
                    for r0 in (0..c.n).step_by(8) {
                        let rows: [&[u8]; 8] = std::array::from_fn(|i| &bytes[e * expert_bytes + ((r0 + i).min(c.n - 1)) * row_bytes..][..row_bytes]);
                        let v = unsafe { cpu::rows8(fmt, &rows, xrow, c.k) };
                        for i in 0..8.min(c.n - r0) {
                            let s = cpu::row(fmt, rows[i], xrow, c.k);
                            let g = b[task * c.n + r0 + i].to_bits();
                            cbad += (s.to_bits() != g) as usize + (v[i].to_bits() != g) as usize;
                            cn += 2;
                        }
                    }
                }
                if seed == 1 || cbad > 0 {
                    println!("    cpu twin: {cn} rows (scalar + avx2), {cbad} mismatches vs GPU");
                }
                *cpu_total += cn;
                let pf = per_fmt.entry(format!("{fmt:?} cpu-twin")).or_default();
                pf[2] += cn;
                pf[3] += cn - cbad;
                ok &= cbad == 0;
            }
        }
    }
    ok
}
