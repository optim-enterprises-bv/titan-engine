#![allow(unsafe_op_in_unsafe_fn)]
#![allow(non_snake_case, clippy::missing_safety_doc, clippy::too_many_arguments, dead_code)]
//! Batched MoE expert GEMV for titan-engine's tiered expert path: llama.cpp acecd56's
//! `mul_mat_vec_q_moe` (ggml-cuda/mmvq.cu, PRs #20905 / #27621) and the `mul_mat_vec_q<type, 1>`
//! MUL_MAT_ID grid, in cuda-oxide, reading experts through titan's slot map.
//!
//! Geometry:
//! - `*_moe_mmvq*`: llama.cpp's MoE kernel. Grid (ceil(n / 2), topk), block (32, b): one warp per
//!   token, two output rows per block, no shared memory. Warp `t` of block (r, k) computes rows
//!   2r, 2r + 1 of task t * topk + k. A warp whose expert is not resident returns on its own.
//! - `*_moe_b1*`: llama.cpp's b1 MUL_MAT_ID grid. Grid (ceil(n / rpb), tasks), block (32, 4); rpb 1,
//!   or 4 when k is small (`small_k`: fewer quant blocks per row than the 4 warps cover in one
//!   iteration), the warps' partials handed through shared memory in warp order, then a butterfly.
//!
//! Float programs (each lane's k loop, then the reduction). The SASS is the spec; each helper
//! below is the per-iteration program of the reference, gated bit for bit:
//! - `*_llama`: llama.cpp's own (fast-math, `.ftz`). In the MoE kernel each lane accumulates every
//!   32 / (qi / vdr)-th quant block of the row into ONE register, so a fused `tmp += d * sumf`
//!   (Q6_K, IQ4_NL, MXFP4, NVFP4, Q1_0) chains across iterations: it reduces differently from the
//!   b1 kernel, which gives each of 4 warps its own partial.
//! - service (no suffix): each column exactly as titan's b1 expert GEMV (titan-kernels-oxide
//!   `*_q8_1_moe_gemv[_w]`, what titan_cpu.rs twins) reduces it: in the MoE kernel every lane runs
//!   the 4 warps of the b1 block as 4 virtual warps with 4 partials, added in warp order before the
//!   butterfly. Q4_K / Q5_K / Q6_K use candle's float program (no `.ftz`), the other formats
//!   llama.cpp's b1 program. So a verify row reduces exactly like the decode step it verifies, and
//!   the CPU twin needs no change.
mod gate;

use cuda_device::{SharedArray, convert, device, dotprod, kernel, launch_bounds, ptx_asm, thread, warp};
use cuda_host::cuda_module;

#[cuda_module]
pub mod kernels {
    use super::*;

    macro_rules! unroll {
        () => {
            cuda_device::thread::__unroll_config::<0>();
        };
    }

    pub const FMT_Q4K: u32 = 0;
    pub const FMT_Q5K: u32 = 1;
    pub const FMT_Q6K: u32 = 2;
    pub const FMT_Q1_0: u32 = 3;
    pub const FMT_IQ4_NL: u32 = 4;
    pub const FMT_MXFP4: u32 = 5;
    pub const FMT_NVFP4: u32 = 6;
    /// Slot-map value of an expert that is not in any GPU slot (the CPU twin computes it).
    pub const NOT_RESIDENT: u32 = u32::MAX;
    /// kvalues_iq4nl / kvalues_mxfp4 as the little-endian words get_int_from_table_16 reads.
    pub const IQ4_NL_TABLE: [u32; 4] = [0xBFAD9881, 0xF6EADDCF, 0x26190D01, 0x71594535];
    pub const MXFP4_TABLE: [u32; 4] = [0x03020100, 0x0C080604, 0xFDFEFF00, 0xF4F8FAFC];

    /// Per-format constants (llama.cpp ggml_cuda_type_traits / VDR_*_Q8_1_MMVQ).
    pub struct G<const FMT: u32>;
    impl<const FMT: u32> G<FMT> {
        /// values per quant block
        pub const QK: u32 = if FMT <= FMT_Q6K { 256 } else if FMT == FMT_Q1_0 { 128 } else if FMT == FMT_NVFP4 { 64 } else { 32 };
        /// bytes per quant block
        pub const BB: u32 = if FMT == FMT_Q4K {
            144
        } else if FMT == FMT_Q5K {
            176
        } else if FMT == FMT_Q6K {
            210
        } else if FMT == FMT_MXFP4 {
            17
        } else if FMT == FMT_NVFP4 {
            36
        } else {
            18
        };
        /// qi / vdr: threads sharing one quant block
        pub const TPB: u32 = if FMT <= FMT_Q5K { 16 } else if FMT == FMT_Q6K { 32 } else if FMT == FMT_Q1_0 { 4 } else { 2 };
        pub const VDR: u32 = if FMT <= FMT_Q5K { 2 } else if FMT <= FMT_Q1_0 { 1 } else if FMT == FMT_NVFP4 { 4 } else { 2 };
        /// Q8_1 blocks per quant block
        pub const Q8PB: u32 = Self::QK / 32;
        /// quant blocks one warp covers per iteration (vdr * 32 / qi)
        pub const BPI1: u32 = 32 / Self::TPB;
    }

    // ------------------------------------------------------------------------------------------
    // f32 ops: FTZ = llama.cpp / mistralrs-quant (--use_fast_math), !FTZ = candle-kernels (plain -O3).
    // ptx_asm! with the whitespace comment keeps cuda-oxide's f32x2 scan linear (PORTING rule 38).

    #[inline(always)]
    pub fn fma<const FTZ: bool>(a: f32, b: f32, c: f32) -> f32 {
        let r: f32;
        if FTZ {
            unsafe { ptx_asm!("fma.rn.ftz.f32 %0, %1, %2, %3; //\t\x0b\x0c\r\n", out("=f") r, in("f") a, in("f") b, in("f") c, options(register_only)); }
        } else {
            unsafe { ptx_asm!("fma.rn.f32 %0, %1, %2, %3; //\t\x0b\x0c\r\n", out("=f") r, in("f") a, in("f") b, in("f") c, options(register_only)); }
        }
        r
    }
    #[inline(always)]
    pub fn mul<const FTZ: bool>(a: f32, b: f32) -> f32 {
        let r: f32;
        if FTZ {
            unsafe { ptx_asm!("mul.rn.ftz.f32 %0, %1, %2; //\t\x0b\x0c\r\n", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        } else {
            unsafe { ptx_asm!("mul.rn.f32 %0, %1, %2; //\t\x0b\x0c\r\n", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        }
        r
    }
    #[inline(always)]
    pub fn add<const FTZ: bool>(a: f32, b: f32) -> f32 {
        let r: f32;
        if FTZ {
            unsafe { ptx_asm!("add.rn.ftz.f32 %0, %1, %2; //\t\x0b\x0c\r\n", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        } else {
            unsafe { ptx_asm!("add.rn.f32 %0, %1, %2; //\t\x0b\x0c\r\n", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        }
        r
    }
    #[inline(always)]
    pub fn half_lo(w: u32) -> f32 {
        convert::cvt_f32_f16x2_lo(w & 0xFFFF)
    }
    #[inline(always)]
    pub fn half_hi(w: u32) -> f32 {
        convert::cvt_f32_f16x2_lo(w >> 16)
    }
    #[inline(always)]
    pub fn dp4a(a: u32, b: u32, c: i32) -> i32 {
        dotprod::dp4a_s32(a, b, c)
    }
    /// `__byte_perm(a, b, s)`: prmt with the selector masked to its 3-bit byte indices.
    #[inline(always)]
    pub fn byte_perm(a: u32, b: u32, s: u32) -> u32 {
        let r: u32;
        let s = s & 0x7777;
        unsafe { ptx_asm!("prmt.b32 %0, %1, %2, %3;", out("=r") r, in("r") a, in("r") b, in("r") s, options(register_only)); }
        r
    }

    // Read-only global loads (`const __restrict__` in the reference: LDG.E.CONSTANT).
    #[inline(always)]
    pub unsafe fn ldw(p: *const u8) -> u32 {
        let r: u32;
        ptx_asm!("ld.global.nc.b32 %0, [%1];", out("=r") r, in("l") p as u64, options(register_only));
        r
    }
    #[inline(always)]
    pub unsafe fn ldh(p: *const u8) -> u32 {
        let r: u32;
        ptx_asm!("ld.global.nc.u16 %0, [%1];", out("=r") r, in("l") p as u64, options(register_only));
        r
    }
    #[inline(always)]
    pub unsafe fn ldb(p: *const u8) -> u32 {
        let r: u32;
        ptx_asm!("ld.global.nc.u8 %0, [%1];", out("=r") r, in("l") p as u64, options(register_only));
        r
    }
    /// `get_int_b2`: a 32-bit word of a 2-byte aligned block from two 16-bit loads.
    #[inline(always)]
    pub unsafe fn ld2h(p: *const u8) -> u32 {
        ldh(p) | (ldh(p.add(2)) << 16)
    }

    // ------------------------------------------------------------------------------------------
    // Per-iteration float programs (titan-kernels-oxide's gated helpers, pointer-addressed).
    // `wb`: the quant block; `xb`: the first Q8_1 block (36 bytes: half2 ds, 32 int8) aligned with it.

    /// The 6-bit scale and min of this thread's sub-block pair (Q4_K / Q5_K `aux`): (sc0, sc1, m0, m1).
    /// The three 16-bit scale words are loaded unconditionally and selected (as llama.cpp's SASS).
    #[inline(always)]
    pub unsafe fn k_scales(wb: *const u8, bq8_offset: u32) -> [u32; 4] {
        let j = bq8_offset / 2;
        let m = (j & 1) as usize;
        let s0 = ldh(wb.add(4 + 2 * m));
        let s1 = ldh(wb.add(8 + 2 * m));
        let s2 = ldh(wb.add(12 + 2 * m));
        let (aux0, aux1) = if j < 2 {
            (s0 & 0x3f3f, s1 & 0x3f3f)
        } else {
            ((s2 & 0x0f0f) | ((s0 & 0xc0c0) >> 2), ((s2 >> 4) & 0x0f0f) | ((s1 & 0xc0c0) >> 2))
        };
        [aux0 & 0xFF, aux0 >> 8, aux1 & 0xFF, aux1 >> 8]
    }

    /// The float tail of `vec_dot_q{4,5}_K_q8_1_impl_vmmq`: `v*` are the 4-weight words of sub-block
    /// pair 0 (`va0`, `vb0`) and 1 (`va1`, `vb1`); ptxas's `FMUL p = sumf_m * dm.y; FFMA sumf_d * dm.x - p`.
    #[inline(always)]
    pub unsafe fn k_dot<const FTZ: bool>(va0: u32, vb0: u32, va1: u32, vb1: u32, xb: *const u8, bq8_offset: u32, iqs: u32, s: [u32; 4], dm: u32) -> f32 {
        let q = (4 * ((iqs / 2) % 4)) as usize;
        let b0 = xb.add((bq8_offset * 36) as usize);
        let b1 = b0.add(36);
        let d80 = half_lo(ldw(b0));
        let d81 = half_lo(ldw(b1));
        let (u00, u01) = (ldw(b0.add(4 + q)), ldw(b0.add(20 + q)));
        let (u10, u11) = (ldw(b1.add(4 + q)), ldw(b1.add(20 + q)));
        let dot10 = dp4a(vb0, u01, dp4a(va0, u00, 0));
        let dot20 = dp4a(0x0101_0101, u01, dp4a(0x0101_0101, u00, 0));
        let dot11 = dp4a(vb1, u11, dp4a(va1, u10, 0));
        let dot21 = dp4a(0x0101_0101, u11, dp4a(0x0101_0101, u10, 0));
        let mut sumf_d = fma::<FTZ>(d80, dot10.wrapping_mul(s[0] as i32) as f32, 0.0);
        let mut sumf_m = fma::<FTZ>(d80, dot20.wrapping_mul(s[2] as i32) as f32, 0.0);
        sumf_d = fma::<FTZ>(d81, dot11.wrapping_mul(s[1] as i32) as f32, sumf_d);
        sumf_m = fma::<FTZ>(d81, dot21.wrapping_mul(s[3] as i32) as f32, sumf_m);
        let p = mul::<FTZ>(sumf_m, half_hi(dm));
        fma::<FTZ>(sumf_d, half_lo(dm), -p)
    }

    /// `vec_dot_q4_K_q8_1`. block_q4_K: half2 dm (0..4), 12 scale bytes (4..16), 128 nibble bytes (16..144).
    #[inline(always)]
    pub unsafe fn dot_q4k<const FTZ: bool>(wb: *const u8, xb: *const u8, iqs: u32) -> f32 {
        let bq8_offset = 2 * ((iqs / 2) / 4);
        let q4 = wb.add((16 + 16 * bq8_offset + 4 * ((iqs / 2) % 4)) as usize);
        let (a, b) = (ldw(q4), ldw(q4.add(16)));
        const M: u32 = 0x0F0F_0F0F;
        k_dot::<FTZ>(a & M, b & M, (a >> 4) & M, (b >> 4) & M, xb, bq8_offset, iqs, k_scales(wb, bq8_offset), ldw(wb))
    }

    /// `vec_dot_q5_K_q8_1`. block_q5_K: dm (0..4), scales (4..16), 32 high-bit bytes (16..48), 128 nibble bytes (48..176).
    #[inline(always)]
    pub unsafe fn dot_q5k<const FTZ: bool>(wb: *const u8, xb: *const u8, iqs: u32) -> f32 {
        let bq8_offset = 2 * ((iqs / 2) / 4);
        let o = 4 * ((iqs / 2) % 4);
        let ql = wb.add((48 + 16 * bq8_offset + o) as usize);
        let qh = wb.add((16 + o) as usize);
        let (l0, l1) = (ldw(ql), ldw(ql.add(16)));
        let (h0, h1) = (ldw(qh) >> bq8_offset, ldw(qh.add(16)) >> bq8_offset);
        const M: u32 = 0x0F0F_0F0F;
        const H: u32 = 0x1010_1010;
        let va0 = (l0 & M) | ((h0 << 4) & H);
        let vb0 = (l1 & M) | ((h1 << 4) & H);
        let va1 = ((l0 >> 4) & M) | (((h0 >> 1) << 4) & H);
        let vb1 = ((l1 >> 4) & M) | (((h1 >> 1) << 4) & H);
        k_dot::<FTZ>(va0, vb0, va1, vb1, xb, bq8_offset, iqs, k_scales(wb, bq8_offset), ldw(wb))
    }

    /// `tmp += vec_dot_q6_K_q8_1(...)`, the `+=` contracted: `fma(d, sumf, tmp)`.
    /// block_q6_K (210 bytes, 2-byte aligned): ql 0..128, qh 128..192, int8 scales 192..208, half d 208.
    #[inline(always)]
    pub unsafe fn acc_q6k<const FTZ: bool>(tmp: f32, b: *const u8, xb: *const u8, iqs: u32) -> f32 {
        let bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        let scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        let vh_shift = 2 * ((iqs % 16) / 8);
        let vl = ld2h(b.add((4 * iqs) as usize));
        let vh = (ld2h(b.add((128 + 4 * (8 * (iqs / 16) + iqs % 8)) as usize)) as i32) >> vh_shift;
        let u_off = (4 + 4 * (iqs % 8)) as usize;
        let mut sumf = 0f32;
        let mut i = 0u32;
        while i < 2 {
            unroll!();
            let bq8 = xb.add(((bq8_offset + 2 * i) * 36) as usize);
            let d8 = half_lo(ldw(bq8));
            let u = ldw(bq8.add(u_off));
            let sc = ldb(b.add((192 + scale_offset + 4 * i) as usize)) as u8 as i8 as i32;
            let vil = ((vl >> (4 * i)) as u32) & 0x0F0F_0F0F;
            let vih = (((vh >> (4 * i)) << 4) as u32) & 0x3030_3030;
            // __vsubss4(vil | vih, 0x20202020): bytes in [0, 63], so a per-byte x - 32 never saturates.
            let vi = ((vil | vih | 0x8080_8080).wrapping_sub(0x2020_2020)) ^ 0x8080_8080;
            sumf = fma::<FTZ>(d8, dp4a(vi, u, 0).wrapping_mul(sc) as f32, sumf);
            i += 1;
        }
        fma::<FTZ>(half_lo(ldh(b.add(208))), sumf, tmp)
    }

    /// `tmp += vec_dot_q1_0_q8_1(...)` (llama.cpp acecd56, fast-math): 32 sign bits -> +-1 bytes by
    /// `__byte_perm`s, four dp4a, `fma(mul(d, d8), (float)sumi, tmp)`. Block: half d + 16 bytes.
    #[inline(always)]
    pub unsafe fn acc_q1_0<const FTZ: bool>(tmp: f32, b: *const u8, xb: *const u8, iqs: u32) -> f32 {
        let bq8 = xb.add((iqs * 36) as usize);
        let mut sumi = 0i32;
        let mut j = 0u32;
        while j < 2 {
            unroll!();
            let q = (ldh(b.add((2 + 4 * iqs + 2 * j) as usize)) as u16 as i16) as i32 as u32;
            let n0 = byte_perm(0x11100100, 0x11100100, q);
            let n1 = byte_perm(0x11100100, 0x11100100, ((q as i32) >> 2) as u32);
            let s0 = byte_perm(0x01FF, 0x01FF, n0);
            let s1 = byte_perm(0x01FF, 0x01FF, n1);
            let s2 = byte_perm(0x01FF, 0x01FF, ((n0 as i32) >> 16) as u32);
            let s3 = byte_perm(0x01FF, 0x01FF, ((n1 as i32) >> 16) as u32);
            let yq = bq8.add((4 + 16 * j) as usize);
            sumi = dp4a(byte_perm(s0, s1, 0x5410), ldw(yq), sumi);
            sumi = dp4a(byte_perm(s0, s1, 0x7632), ldw(yq.add(4)), sumi);
            sumi = dp4a(byte_perm(s2, s3, 0x5410), ldw(yq.add(8)), sumi);
            sumi = dp4a(byte_perm(s2, s3, 0x7632), ldw(yq.add(12)), sumi);
            j += 1;
        }
        let d = half_lo(ldh(b));
        let d8 = half_lo(ldw(bq8));
        fma::<FTZ>(mul::<FTZ>(d, d8), sumi as f32, tmp)
    }

    /// llama.cpp `get_int_from_table_16(q4, table)`: (table bytes of the 4 low nibbles, of the 4 high).
    #[inline(always)]
    pub fn table16(q4: u32, t: [u32; 4]) -> (u32, u32) {
        let sel = 0x32103210 | ((q4 & 0x88888888) >> 1);
        let t0 = byte_perm(byte_perm(t[0], t[1], q4), byte_perm(t[2], t[3], q4), sel);
        let t1 = byte_perm(byte_perm(t[0], t[1], q4 >> 16), byte_perm(t[2], t[3], q4 >> 16), sel >> 16);
        (byte_perm(t0, t1, 0x6420), byte_perm(t0, t1, 0x7531))
    }

    /// `tmp += vec_dot_iq4_nl_q8_1(...)`: `fma(mul(d, d8), (float)sumi, tmp)`. Block: half d + 16 nibble bytes.
    #[inline(always)]
    pub unsafe fn acc_iq4_nl<const FTZ: bool>(tmp: f32, b: *const u8, xb: *const u8, iqs: u32) -> f32 {
        let mut sumi = 0i32;
        let mut l = 0u32;
        while l < 2 {
            unroll!();
            let (lo, hi) = table16(ld2h(b.add((2 + 4 * (iqs + l)) as usize)), IQ4_NL_TABLE);
            sumi = dp4a(lo, ldw(xb.add((4 * (1 + iqs + l)) as usize)), sumi);
            sumi = dp4a(hi, ldw(xb.add((4 * (5 + iqs + l)) as usize)), sumi);
            l += 1;
        }
        let d = half_lo(ldh(b));
        let d8 = half_lo(ldw(xb));
        fma::<FTZ>(mul::<FTZ>(d, d8), sumi as f32, tmp)
    }

    /// `ggml_cuda_e8m0_to_fp32` without the sm_120a-only cvt: 2^(e - 127), e = 255 NaN, e = 0 the
    /// denormal 2^-127. Every consumer is a mul.ftz, so this is the reference's bits (gated).
    #[inline(always)]
    pub fn e8m0(e: u32) -> f32 {
        if e == 255 {
            f32::from_bits(0x7FFF_FFFF)
        } else if e == 0 {
            f32::from_bits(0x0040_0000)
        } else {
            f32::from_bits(e << 23)
        }
    }

    /// `tmp += vec_dot_mxfp4_q8_1(...)`: `fma(mul(mul(e8m0(e), 0.5), d8), (float)sumi, tmp)`.
    /// Block: E8M0 byte + 16 nibble bytes (17 bytes, byte-addressed).
    #[inline(always)]
    pub unsafe fn acc_mxfp4<const FTZ: bool>(tmp: f32, b: *const u8, xb: *const u8, iqs: u32) -> f32 {
        let mut sumi = 0i32;
        let mut l = 0u32;
        while l < 2 {
            unroll!();
            let q = b.add((1 + 4 * (iqs + l)) as usize);
            let w = ldb(q) | (ldb(q.add(1)) << 8) | (ldb(q.add(2)) << 16) | (ldb(q.add(3)) << 24);
            let (lo, hi) = table16(w, MXFP4_TABLE);
            sumi = dp4a(lo, ldw(xb.add((4 * (1 + iqs + l)) as usize)), sumi);
            sumi = dp4a(hi, ldw(xb.add((4 * (5 + iqs + l)) as usize)), sumi);
            l += 1;
        }
        let d = mul::<FTZ>(mul::<FTZ>(e8m0(ldb(b)), 0.5), half_lo(ldw(xb)));
        fma::<FTZ>(d, sumi as f32, tmp)
    }

    /// `ggml_cuda_ue4m3_to_fp32` (FP8_AVAILABLE): the byte as E4M3 (0x7F / 0xFF -> 0), cvt to f16, to
    /// f32, then `/ 2` as fast-math `div.approx.ftz`.
    #[inline(always)]
    pub fn ue4m3(b: u32) -> f32 {
        let b = if b & 0x7F == 0x7F { 0u16 } else { b as u16 };
        let r: u32;
        unsafe { ptx_asm!("cvt.rn.f16x2.e4m3x2 %0, %1;", out("=r") r, in("h") b, options(register_only)); }
        let v = half_lo(r);
        let q: f32;
        unsafe { ptx_asm!("div.approx.ftz.f32 %0, %1, 0f40000000;", out("=f") q, in("f") v, options(register_only)); }
        q
    }

    /// `tmp += vec_dot_nvfp4_q8_1(...)`: per 16-value sub-block four dp4a, `sum = fma(mul(d_s, d8),
    /// (float)sumi, sum)` from +0, then `add(tmp, sum)`. Block: 4 UE4M3 scales + 32 nibble bytes.
    #[inline(always)]
    pub unsafe fn acc_nvfp4<const FTZ: bool>(tmp: f32, b: *const u8, xb: *const u8, iqs: u32) -> f32 {
        let mut sum = 0f32;
        let mut i = 0u32;
        while i < 2 {
            unroll!();
            let iqs0 = iqs + 2 * i;
            let is = iqs0 >> 1;
            let (v0x, v0y) = table16(ld2h(b.add((4 + 4 * iqs0) as usize)), MXFP4_TABLE);
            let (v1x, v1y) = table16(ld2h(b.add((8 + 4 * iqs0) as usize)), MXFP4_TABLE);
            let bq8 = xb.add(((is >> 1) * 36) as usize);
            let i8 = ((is & 1) << 2) as usize;
            let mut sumi = dp4a(v0x, ldw(bq8.add(4 * (1 + i8))), 0);
            sumi = dp4a(v0y, ldw(bq8.add(4 * (3 + i8))), sumi);
            sumi = dp4a(v1x, ldw(bq8.add(4 * (2 + i8))), sumi);
            sumi = dp4a(v1y, ldw(bq8.add(4 * (4 + i8))), sumi);
            let d = mul::<FTZ>(ue4m3(ldb(b.add(is as usize))), half_lo(ldw(bq8)));
            sum = fma::<FTZ>(d, sumi as f32, sum);
            i += 1;
        }
        add::<FTZ>(tmp, sum)
    }

    /// One k iteration of one thread: `tmp += vec_dot(block)` in the format's float program.
    #[inline(always)]
    pub unsafe fn step<const FMT: u32, const FTZ: bool>(tmp: f32, wb: *const u8, xb: *const u8, kqs: u32) -> f32 {
        if FMT == FMT_Q4K {
            add::<FTZ>(tmp, dot_q4k::<FTZ>(wb, xb, kqs))
        } else if FMT == FMT_Q5K {
            add::<FTZ>(tmp, dot_q5k::<FTZ>(wb, xb, kqs))
        } else if FMT == FMT_Q6K {
            acc_q6k::<FTZ>(tmp, wb, xb, kqs)
        } else if FMT == FMT_Q1_0 {
            acc_q1_0::<FTZ>(tmp, wb, xb, kqs)
        } else if FMT == FMT_IQ4_NL {
            acc_iq4_nl::<FTZ>(tmp, wb, xb, kqs)
        } else if FMT == FMT_MXFP4 {
            acc_mxfp4::<FTZ>(tmp, wb, xb, kqs)
        } else {
            acc_nvfp4::<FTZ>(tmp, wb, xb, kqs)
        }
    }

    /// xor butterfly over distances 16, 8, 4, 2, 1 (`warp_reduce_sum`).
    #[device]
    #[inline(always)]
    pub fn butterfly<const FTZ: bool>(mut t: f32) -> f32 {
        let mut mask = 16u32;
        while mask > 0 {
            t = add::<FTZ>(t, warp::shuffle_xor_f32_sync(0xffff_ffff, t, mask));
            mask >>= 1;
        }
        t
    }

    /// `mul_mat_vec_q_moe<type, 2, false>` over titan's slot map. NV = 1: llama.cpp's lane program
    /// (one partial per lane); NV = 4: each lane runs the 4 warps of the b1 kernel as virtual warps
    /// (partial v covers the quant blocks warp v of the b1 block would), added in warp order.
    /// Launch: grid (ceil(n / 2), topk), block (32, b), b <= 8 tokens.
    /// Layout: `ids[t * topk + k]` experts, `out[(t * topk + k) * n + row]`, input row t (input_dim1
    /// == 1) or t * topk + k, each k_padded / 32 Q8_1 blocks.
    #[device]
    #[inline(always)]
    pub unsafe fn moe_warp<const FMT: u32, const FTZ: bool, const NV: u32>(
        w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32,
        n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32,
    ) {
        let lane = thread::threadIdx_x();
        let token = thread::threadIdx_y();
        let task = token * topk + thread::blockIdx_y();
        let row0 = 2 * thread::blockIdx_x();
        let e = ldw(ids.add(task as usize) as *const u8);
        let slot = ldw(slot_map.add(e as usize) as *const u8);
        if slot == NOT_RESIDENT {
            return;
        }
        let input_row = if input_dim1 == 1 { token } else { task };
        let bpr = k / G::<FMT>::QK;
        let row_bytes = (bpr * G::<FMT>::BB) as usize;
        let xrow = x.add(input_row as usize * (k_padded / 32) as usize * 36);
        let w0 = w.add((slot as usize * n as usize + row0 as usize) * row_bytes);
        let w1 = w0.add(row_bytes);
        let ok1 = row0 + 1 < n;
        let kqs = G::<FMT>::VDR * (lane % G::<FMT>::TPB);
        let mut a0 = [0f32; 4];
        let mut a1 = [0f32; 4];
        let mut base = lane / G::<FMT>::TPB;
        while base < bpr {
            let mut v = 0u32;
            while v < NV {
                unroll!();
                let kbx = base + v * G::<FMT>::BPI1;
                if kbx < bpr {
                    let xb = xrow.add((kbx * G::<FMT>::Q8PB * 36) as usize);
                    let wo = (kbx * G::<FMT>::BB) as usize;
                    a0[v as usize] = step::<FMT, FTZ>(a0[v as usize], w0.add(wo), xb, kqs);
                    if ok1 {
                        a1[v as usize] = step::<FMT, FTZ>(a1[v as usize], w1.add(wo), xb, kqs);
                    }
                }
                v += 1;
            }
            base += NV * G::<FMT>::BPI1;
        }
        let mut t0 = a0[0];
        let mut t1 = a1[0];
        let mut v = 1u32;
        while v < NV {
            unroll!();
            t0 = add::<FTZ>(t0, a0[v as usize]);
            t1 = add::<FTZ>(t1, a1[v as usize]);
            v += 1;
        }
        t0 = butterfly::<FTZ>(t0);
        t1 = butterfly::<FTZ>(t1);
        let o = out.add(task as usize * n as usize + row0 as usize);
        if lane == 0 {
            *o = t0;
        }
        if lane == 1 && ok1 {
            *o.add(1) = t1;
        }
    }

    /// `mul_mat_vec_q<type, 1, false, small_k>` MUL_MAT_ID over titan's slot map: 4 warps, RPB rows
    /// per block (4 = small_k), thread tid = 32 * warp + lane reads quant blocks tid / (qi / vdr) +
    /// j * 4 * 32 / (qi / vdr); warps 1..3 hand partials to warp 0 in warp order, then the butterfly.
    /// Launch: grid (ceil(n / RPB), tasks), block (32, 4). Layout as `moe_warp` (task = blockIdx.y).
    #[device]
    #[inline(always)]
    pub unsafe fn b1_block<const FMT: u32, const FTZ: bool, const RPB: u32>(
        w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32,
        n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32,
    ) {
        // tmp_shared[3][RPB][32]
        static mut PARTIAL: SharedArray<f32, 384> = SharedArray::UNINIT;
        let lane = thread::threadIdx_x();
        let wy = thread::threadIdx_y();
        let task = thread::blockIdx_y();
        let row0 = RPB * thread::blockIdx_x();
        let e = ldw(ids.add(task as usize) as *const u8);
        let slot = ldw(slot_map.add(e as usize) as *const u8);
        // Uniform per block: the whole block leaves before the barrier.
        if slot == NOT_RESIDENT {
            return;
        }
        let token = task / topk;
        let input_row = if input_dim1 == 1 { token } else { task };
        let bpr = k / G::<FMT>::QK;
        let row_bytes = (bpr * G::<FMT>::BB) as usize;
        let xrow = x.add(input_row as usize * (k_padded / 32) as usize * 36);
        let wr = w.add((slot as usize * n as usize + row0 as usize) * row_bytes);
        let kqs = G::<FMT>::VDR * (lane % G::<FMT>::TPB);
        let mut acc = [0f32; 4];
        let mut kbx = (32 * wy + lane) / G::<FMT>::TPB;
        while kbx < bpr {
            let xb = xrow.add((kbx * G::<FMT>::Q8PB * 36) as usize);
            let wo = (kbx * G::<FMT>::BB) as usize;
            let mut i = 0u32;
            while i < RPB {
                unroll!();
                if RPB == 1 || row0 + i < n {
                    acc[i as usize] = step::<FMT, FTZ>(acc[i as usize], wr.add(i as usize * row_bytes + wo), xb, kqs);
                }
                i += 1;
            }
            kbx += 4 * G::<FMT>::BPI1;
        }
        let sh = SharedArray::as_raw_mut_ptr(&raw mut PARTIAL);
        if wy > 0 {
            let mut i = 0u32;
            while i < RPB {
                unroll!();
                *sh.add((((wy - 1) * RPB + i) * 32 + lane) as usize) = acc[i as usize];
                i += 1;
            }
        }
        thread::sync_threads();
        if wy > 0 {
            return;
        }
        let o = out.add(task as usize * n as usize + row0 as usize);
        let mut i = 0u32;
        while i < RPB {
            unroll!();
            let mut t = acc[i as usize];
            let mut l = 0u32;
            while l < 3 {
                unroll!();
                t = add::<FTZ>(t, *sh.add(((l * RPB + i) * 32 + lane) as usize));
                l += 1;
            }
            t = butterfly::<FTZ>(t);
            if lane == i && (RPB == 1 || row0 + i < n) {
                *o.add(i as usize) = t;
            }
            i += 1;
        }
    }

    // GENERATED KERNELS BEGIN (gen_kernels.py)
    /// q4k: MoE grid, each column reduced as titan's b1 GEMV (candle's program). Service kernel, b = 2..8.
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn q4k_moe_mmvq(w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) { moe_warp::<FMT_Q4K, false, 4>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1) }
    /// q4k: llama.cpp mul_mat_vec_q_moe<type, 2, false> float program (gated against it).
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn q4k_moe_mmvq_llama(w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) { moe_warp::<FMT_Q4K, true, 1>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1) }
    /// q4k: llama.cpp b1 MUL_MAT_ID grid, rpb 1 (candle's program). Service kernel, b = 1.
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn q4k_moe_b1(w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) { b1_block::<FMT_Q4K, false, 1>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1) }
    /// q4k: llama.cpp b1 MUL_MAT_ID grid, small_k (rpb 4) (candle's program). Service kernel, b = 1.
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn q4k_moe_b1_sk(w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) { b1_block::<FMT_Q4K, false, 4>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1) }
    /// q4k: llama.cpp mul_mat_vec_q<type, 1, false, false> float program (gated against it).
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn q4k_moe_b1_llama(w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) { b1_block::<FMT_Q4K, true, 1>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1) }
    /// q4k: llama.cpp mul_mat_vec_q<type, 1, false, true> float program (gated against it).
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn q4k_moe_b1_sk_llama(w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) { b1_block::<FMT_Q4K, true, 4>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1) }
    /// q5k: MoE grid, each column reduced as titan's b1 GEMV (candle's program). Service kernel, b = 2..8.
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn q5k_moe_mmvq(w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) { moe_warp::<FMT_Q5K, false, 4>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1) }
    /// q5k: llama.cpp mul_mat_vec_q_moe<type, 2, false> float program (gated against it).
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn q5k_moe_mmvq_llama(w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) { moe_warp::<FMT_Q5K, true, 1>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1) }
    /// q5k: llama.cpp b1 MUL_MAT_ID grid, rpb 1 (candle's program). Service kernel, b = 1.
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn q5k_moe_b1(w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) { b1_block::<FMT_Q5K, false, 1>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1) }
    /// q5k: llama.cpp b1 MUL_MAT_ID grid, small_k (rpb 4) (candle's program). Service kernel, b = 1.
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn q5k_moe_b1_sk(w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) { b1_block::<FMT_Q5K, false, 4>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1) }
    /// q5k: llama.cpp mul_mat_vec_q<type, 1, false, false> float program (gated against it).
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn q5k_moe_b1_llama(w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) { b1_block::<FMT_Q5K, true, 1>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1) }
    /// q5k: llama.cpp mul_mat_vec_q<type, 1, false, true> float program (gated against it).
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn q5k_moe_b1_sk_llama(w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) { b1_block::<FMT_Q5K, true, 4>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1) }
    /// q6k: MoE grid, each column reduced as titan's b1 GEMV (candle's program). Service kernel, b = 2..8.
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn q6k_moe_mmvq(w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) { moe_warp::<FMT_Q6K, false, 4>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1) }
    /// q6k: llama.cpp mul_mat_vec_q_moe<type, 2, false> float program (gated against it).
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn q6k_moe_mmvq_llama(w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) { moe_warp::<FMT_Q6K, true, 1>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1) }
    /// q6k: llama.cpp b1 MUL_MAT_ID grid, rpb 1 (candle's program). Service kernel, b = 1.
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn q6k_moe_b1(w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) { b1_block::<FMT_Q6K, false, 1>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1) }
    /// q6k: llama.cpp b1 MUL_MAT_ID grid, small_k (rpb 4) (candle's program). Service kernel, b = 1.
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn q6k_moe_b1_sk(w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) { b1_block::<FMT_Q6K, false, 4>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1) }
    /// q6k: llama.cpp mul_mat_vec_q<type, 1, false, false> float program (gated against it).
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn q6k_moe_b1_llama(w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) { b1_block::<FMT_Q6K, true, 1>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1) }
    /// q6k: llama.cpp mul_mat_vec_q<type, 1, false, true> float program (gated against it).
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn q6k_moe_b1_sk_llama(w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) { b1_block::<FMT_Q6K, true, 4>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1) }
    /// q1_0: MoE grid, each column reduced as titan's b1 GEMV (llama.cpp b1 program). Service kernel, b = 2..8.
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn q1_0_moe_mmvq(w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) { moe_warp::<FMT_Q1_0, true, 4>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1) }
    /// q1_0: llama.cpp mul_mat_vec_q_moe<type, 2, false> float program (gated against it).
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn q1_0_moe_mmvq_llama(w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) { moe_warp::<FMT_Q1_0, true, 1>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1) }
    /// q1_0: llama.cpp b1 MUL_MAT_ID grid, rpb 1 (llama.cpp b1 program). Service kernel, b = 1.
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn q1_0_moe_b1(w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) { b1_block::<FMT_Q1_0, true, 1>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1) }
    /// q1_0: llama.cpp b1 MUL_MAT_ID grid, small_k (rpb 4) (llama.cpp b1 program). Service kernel, b = 1.
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn q1_0_moe_b1_sk(w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) { b1_block::<FMT_Q1_0, true, 4>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1) }
    /// iq4_nl: MoE grid, each column reduced as titan's b1 GEMV (llama.cpp b1 program). Service kernel, b = 2..8.
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn iq4_nl_moe_mmvq(w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) { moe_warp::<FMT_IQ4_NL, true, 4>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1) }
    /// iq4_nl: llama.cpp mul_mat_vec_q_moe<type, 2, false> float program (gated against it).
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn iq4_nl_moe_mmvq_llama(w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) { moe_warp::<FMT_IQ4_NL, true, 1>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1) }
    /// iq4_nl: llama.cpp b1 MUL_MAT_ID grid, rpb 1 (llama.cpp b1 program). Service kernel, b = 1.
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn iq4_nl_moe_b1(w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) { b1_block::<FMT_IQ4_NL, true, 1>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1) }
    /// iq4_nl: llama.cpp b1 MUL_MAT_ID grid, small_k (rpb 4) (llama.cpp b1 program). Service kernel, b = 1.
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn iq4_nl_moe_b1_sk(w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) { b1_block::<FMT_IQ4_NL, true, 4>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1) }
    /// mxfp4: MoE grid, each column reduced as titan's b1 GEMV (llama.cpp b1 program). Service kernel, b = 2..8.
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn mxfp4_moe_mmvq(w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) { moe_warp::<FMT_MXFP4, true, 4>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1) }
    /// mxfp4: llama.cpp mul_mat_vec_q_moe<type, 2, false> float program (gated against it).
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn mxfp4_moe_mmvq_llama(w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) { moe_warp::<FMT_MXFP4, true, 1>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1) }
    /// mxfp4: llama.cpp b1 MUL_MAT_ID grid, rpb 1 (llama.cpp b1 program). Service kernel, b = 1.
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn mxfp4_moe_b1(w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) { b1_block::<FMT_MXFP4, true, 1>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1) }
    /// mxfp4: llama.cpp b1 MUL_MAT_ID grid, small_k (rpb 4) (llama.cpp b1 program). Service kernel, b = 1.
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn mxfp4_moe_b1_sk(w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) { b1_block::<FMT_MXFP4, true, 4>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1) }
    /// nvfp4: MoE grid, each column reduced as titan's b1 GEMV (llama.cpp b1 program). Service kernel, b = 2..8.
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn nvfp4_moe_mmvq(w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) { moe_warp::<FMT_NVFP4, true, 4>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1) }
    /// nvfp4: llama.cpp mul_mat_vec_q_moe<type, 2, false> float program (gated against it).
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn nvfp4_moe_mmvq_llama(w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) { moe_warp::<FMT_NVFP4, true, 1>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1) }
    /// nvfp4: llama.cpp b1 MUL_MAT_ID grid, rpb 1 (llama.cpp b1 program). Service kernel, b = 1.
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn nvfp4_moe_b1(w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) { b1_block::<FMT_NVFP4, true, 1>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1) }
    /// nvfp4: llama.cpp b1 MUL_MAT_ID grid, small_k (rpb 4) (llama.cpp b1 program). Service kernel, b = 1.
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn nvfp4_moe_b1_sk(w: *const u8, x: *const u8, ids: *const u32, slot_map: *const u32, out: *mut f32, n: u32, k: u32, k_padded: u32, topk: u32, input_dim1: u32) { b1_block::<FMT_NVFP4, true, 4>(w, x, ids, slot_map, out, n, k, k_padded, topk, input_dim1) }
    // GENERATED KERNELS END
}

fn main() {
    std::process::exit(if gate::run() { 0 } else { 1 });
}
