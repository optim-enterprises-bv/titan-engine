#![allow(unsafe_op_in_unsafe_fn)]
//! Row-exact batched GGUF matvec for titan-engine M6 (speculative verification in mistral.rs).
//!
//! `mmvq_gguf_<q>_<dst>_rows<b>`: grid (ceil(nrows / 2), 1, 1), block (32, 8, 1). Each of the b
//! activation columns is reduced exactly as the batch-1 MMVQ kernel of mistralrs-quant's
//! `mmvq_gguf.cu` (`mmvq_core_impl<ncols_dst = 1>`, 8 warps x 2 rows, fast-math) reduces it, so
//! every output row is bit-identical to a batch-1 launch on that row alone, while the weights are
//! read once for all b columns. The b_size 2..8 kernels split and round differently, and
//! verification must reproduce each decode step exactly.
//!
//! The helpers are those of ../mistralrs-quant-c (gated there against libmistralrsquant.a). The
//! reference here is the nvcc kernel of the M6 WIP launcher `launch_mmvq_gguf_*_plain_rows`
//! (reference/mistralrs-quant-rows/mmvq_gguf_rows.cubin: `*_plain_cuda1` with the blockIdx.y
//! offsets), gated kernel-vs-kernel with kdiff by `main()`. mistral.rs embeds mmvq_rows.ptx
//! (mistralrs-quant/src/gguf/mmvq_rows_oxide.ptx, see export_ptx.sh) and launches it from Rust.
#![allow(non_snake_case, clippy::missing_safety_doc, dead_code)]

use cuda_device::{SharedArray, bf16x2, convert, device, dotprod, f16x2, float, kernel, ptx_asm, thread, warp};
use cuda_host::cuda_module;

mod gate;

#[cuda_module]
pub mod kernels {
    use super::*;

    // ------------------------------------------------------------------------------------------
    // shared helpers (fast-math f32: every add / mul / fma is .ftz)

    #[inline(always)]
    pub fn h2f(bits: u32) -> f32 {
        convert::cvt_f32_f16x2_lo(bits & 0xFFFF)
    }
    #[inline(always)]
    pub fn bf2f(bits: u32) -> f32 {
        f32::from_bits(bits << 16)
    }
    // The f32 ops are emitted via ptx_asm! with a trailing comment holding every escaped whitespace
    // (PORTING.md rule 38): cuda-oxide's f32x2 feature scan otherwise searches to the end of the
    // LLVM text for each one (quadratic build time). The PTX instruction is the same.
    #[inline(always)]
    pub fn fma(a: f32, b: f32, c: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("fma.rn.ftz.f32 %0, %1, %2, %3; //\t\x0b\x0c\r\n", out("=f") r, in("f") a, in("f") b, in("f") c, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn mul(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("mul.rn.ftz.f32 %0, %1, %2; //\t\x0b\x0c\r\n", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn add(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("add.rn.ftz.f32 %0, %1, %2; //\t\x0b\x0c\r\n", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn sub(a: f32, b: f32) -> f32 {
        add(a, -b)
    }
    #[inline(always)]
    pub fn dp4a(a: u32, b: u32, c: i32) -> i32 {
        dotprod::dp4a_s32(a, b, c)
    }
    /// fmaxf under fast-math: `max.ftz.f32` [FMNMX.FTZ].
    #[inline(always)]
    pub fn fmaxf(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("max.ftz.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    /// fabsf under fast-math: `abs.ftz.f32` [FADD.FTZ |x|, -RZ]: denormals -> +0.
    #[inline(always)]
    pub fn fabsf(a: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("abs.ftz.f32 %0, %1;", out("=f") r, in("f") a, options(register_only)); }
        r
    }
    /// `x != 0.0f` under fast-math: `setp.neu.ftz.f32` (denormals compare equal to zero).
    #[inline(always)]
    pub fn neu0_ftz(a: f32) -> bool {
        let r: u32;
        unsafe { ptx_asm!("{ .reg .pred p; setp.neu.ftz.f32 p, %1, 0f00000000; selp.u32 %0, 1, 0, p; }", out("=r") r, in("f") a, options(register_only)); }
        r != 0
    }
    /// `cvt.rzi.s32.f32` (C float -> int conversion).
    #[inline(always)]
    pub fn f2i_rz(a: f32) -> i32 {
        let r: i32;
        unsafe { ptx_asm!("cvt.rzi.ftz.s32.f32 %0, %1;", out("=r") r, in("f") a, options(register_only)); }
        r
    }
    /// fast-math roundf: `cvt.rzi(add.rz.ftz(t, copysign(0.5, t)))` [FADD.FTZ.RZ; F2I.TRUNC.NTZ].
    #[inline(always)]
    pub fn roundf_i(t: f32) -> i32 {
        let half = f32::from_bits(0x3F00_0000 | (t.to_bits() & 0x8000_0000));
        f2i_rz(float::add_rz_ftz_f32(t, half))
    }
    #[inline(always)]
    pub fn f2h(x: f32) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("cvt.rn.f16.f32 %0, %1;", out("=h") r, in("f") x, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn f2bf(x: f32) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("cvt.rn.bf16.f32 %0, %1;", out("=h") r, in("f") x, options(register_only)); }
        r
    }

    // ------------------------------------------------------------------------------------------
    // fast-math activations (exact PTX of nvcc --use_fast_math; ptxas lowering in brackets),
    // shared by the fused-GLU GEMV and the GLU quantizer.

    /// `ex2.approx.ftz.f32` [MUFU.EX2].
    #[inline(always)]
    pub fn ex2(x: f32) -> f32 {
        float::ex2_approx_ftz_f32(x)
    }
    /// `rcp.approx.ftz.f32` [MUFU.RCP].
    #[inline(always)]
    pub fn rcp(x: f32) -> f32 {
        float::rcp_approx_ftz_f32(x)
    }
    /// fast-math `a / b` = `div.approx.ftz.f32` [MUFU.RCP b; FMUL.FTZ a, r].
    #[inline(always)]
    pub fn div_approx(a: f32, b: f32) -> f32 {
        mul(a, rcp(b))
    }
    pub const LOG2E: f32 = f32::from_bits(0x3fb8aa3b);
    pub const NEG_LOG2E: f32 = f32::from_bits(0xbfb8aa3b);
    pub const SQRT_2_OVER_PI: f32 = f32::from_bits(0x3f4c422a);
    pub const GELU_KAPPA: f32 = f32::from_bits(0x3d372713);

    /// `x / (1 + expf(-x))` [FMUL.FTZ x,-log2e; EX2; FADD.FTZ +1; RCP; FMUL.FTZ].
    #[inline(always)]
    pub fn silu(x: f32) -> f32 {
        div_approx(x, add(ex2(mul(x, NEG_LOG2E)), 1.0))
    }
    /// `0.5f * x * (1 + tanhf(k0 * (x + k1 * x*x*x)))` [FMUL x*x; FMUL x*(x*x); FFMA x3*k1+x;
    /// FMUL *k0; TANH; FADD +1; FMUL (0.5x)*(1+t)].
    #[inline(always)]
    pub fn gelu_tanh(x: f32) -> f32 {
        let x3 = mul(x, mul(x, x));
        let inner = mul(fma(x3, GELU_KAPPA, x), SQRT_2_OVER_PI);
        mul(mul(x, 0.5), add(float::tanh_approx_f32(inner), 1.0))
    }
    /// `cvt.rzi.f32.f32` [FRND.TRUNC].
    #[inline(always)]
    pub fn truncf(a: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("cvt.rzi.f32.f32 %0, %1;", out("=f") r, in("f") a, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn copysignf(mag: f32, sgn: f32) -> f32 {
        f32::from_bits((mag.to_bits() & 0x7fff_ffff) | (sgn.to_bits() & 0x8000_0000))
    }
    #[inline(always)]
    pub fn absf(a: f32) -> f32 {
        f32::from_bits(a.to_bits() & 0x7fff_ffff)
    }
    #[inline(always)]
    pub fn cf(b: u32) -> f32 {
        f32::from_bits(b)
    }

    /// libdevice `normcdff` as nvcc inlines it under --use_fast_math (FTZ reflect), transcribed
    /// instruction by instruction from the SASS (same program in mmvq fused GLU and ops.cu).
    #[inline(always)]
    pub fn normcdf_ftz(x: f32) -> f32 {
        let r8 = if absf(x) > 14.5 { copysignf(14.5, x) } else { x };
        let r3 = mul(r8, cf(0xbf3504f3));
        let p1 = !(r8 < -1.0);
        let r5 = fma(r8, cf(0xbf3504f3), -r3);
        let mut r6 = fma(r8, cf(0xb24fe77a), r5);
        let r4 = add(r3, r6);
        let a = absf(r4);
        let r7 = add(a, 4.0);
        let r5b = add(a, -4.0);
        let r10 = add(a, -0.0);
        let p2 = !(r4 < 0.0);
        let r12 = rcp(r7);
        if !p1 {
            let r3b = add(r3, -r4);
            r6 = add(r6, r3b);
        }
        let r7b = fma(a, 2.0, 1.0);
        let r9 = mul(r5b, r12);
        let r5c = mul(a, -r10);
        let mut r11 = add(r9, 1.0);
        let r13 = mul(r5c, LOG2E);
        r11 = fma(r11, -4.0, a);
        let r13 = truncf(r13);
        r11 = fma(a, -r9, r11);
        r11 = fma(r12, r11, r9);
        let mut q = fma(r11, cf(0x3a69a091), cf(0x3be6e05b));
        q = fma(r11, q, cf(0xbc81fb4b));
        let r16 = if absf(r13) > 126.0 { copysignf(126.0, r13) } else { r13 };
        q = fma(r11, q, cf(0x3d15373b));
        q = fma(r11, q, cf(0xbd887c5a));
        let big = a > cf(0x4120e148);
        let r14 = rcp(r7b);
        q = fma(r11, q, cf(0x3dc021d5));
        let r9b = fma(r16, cf(0xbf317218), r5c);
        let r5d = fma(a, -r10, -r5c);
        q = fma(r11, q, cf(0xbdced424));
        let r9c = fma(r16, cf(0x3102e308), r9b);
        let r16b = add(r16, cf(0x4b40007f));
        q = fma(r11, q, cf(0x3d8b74de));
        let r13b = mul(r9c, LOG2E);
        q = fma(r11, q, cf(0x3c7bf170));
        let r16c = f32::from_bits(r16b.to_bits() << 23);
        q = fma(r11, q, cf(0xbe0ef8d4));
        let r13c = ex2(r13b);
        let r11b = fma(r11, q, cf(0x3f9dd2c9));
        let r9d = mul(r11b, r14);
        let r12b = mul(r9d, -2.0);
        let r12c = fma(a, r12b, r11b);
        let r4m = if !p1 { mul(r4, -2.0) } else { r4 };
        let r16d = mul(r16c, r13c);
        let r12d = add(-r9d, r12c);
        let r5e = fma(r16d, r5d, r16d);
        let r12e = fma(r14, r12d, r9d);
        let mut r = mul(r12e, r5e);
        if big {
            r = 0.0;
        }
        if !p2 {
            r = add(-r, 2.0);
        }
        if !p1 {
            let t = mul(r, r4m);
            r = fma(r6, t, r);
        }
        mul(r, 0.5)
    }

    /// apply_glu_activation / mmq_glu_activation: 0 SiLU, 1 GELU (tanh), 2 ReLU, 3 GELU (erf),
    /// anything else SiLU.
    #[inline(always)]
    pub fn glu_act(x: f32, act: i32) -> f32 {
        match act {
            1 => gelu_tanh(x),
            2 => fmaxf(0.0, x),
            3 => mul(x, normcdf_ftz(x)),
            _ => silu(x),
        }
    }

    // ------------------------------------------------------------------------------------------
    // mmvq_gguf.cu

    #[inline(always)]
    pub unsafe fn ld8(p: *const u8, off: usize) -> u32 {
        *p.add(off) as u32
    }
    #[inline(always)]
    pub unsafe fn ld16(p: *const u8, off: usize) -> u32 {
        *(p.add(off) as *const u16) as u32
    }
    /// `get_int_from_uint8`: a 32-bit word from two 16-bit loads (2-byte aligned blocks).
    #[inline(always)]
    pub unsafe fn ld32u(p: *const u8, off: usize) -> u32 {
        ld16(p, off) | (ld16(p, off + 2) << 16)
    }
    /// `get_int_from_uint8_aligned`.
    #[inline(always)]
    pub unsafe fn ld32a(p: *const u8, off: usize) -> u32 {
        *(p.add(off) as *const u32)
    }
    /// Q8_1 block: quant word `i`.
    #[inline(always)]
    pub unsafe fn yq(yb: *const u8, i: usize) -> u32 {
        ld32a(yb, 4 + 4 * i)
    }
    /// Arithmetic shift right of an `int`.
    #[inline(always)]
    pub fn sar(x: u32, n: usize) -> u32 {
        ((x as i32) >> n) as u32
    }

    /// Per-byte signed saturating subtract (`__vsubss4`).
    #[inline(always)]
    pub fn vsubss4u(a: u32, b: u32) -> u32 {
        let mut r = 0u32;
        let mut k = 0;
        while k < 4 {
            let x = ((a >> (8 * k)) & 0xFF) as u8 as i8 as i32;
            let y = ((b >> (8 * k)) & 0xFF) as u8 as i8 as i32;
            let mut d = x - y;
            if d > 127 {
                d = 127;
            }
            if d < -128 {
                d = -128;
            }
            r |= ((d as u32) & 0xFF) << (8 * k);
            k += 1;
        }
        r
    }

    /// Output element of the GEMVs (`dst_t`).
    pub trait Dst: Copy {
        /// `(dst_t) v`.
        fn from_f(v: f32) -> Self;
        /// `(float) v`.
        fn to_f(self) -> f32;
        /// `dst_t a * dst_t b` (FMUL.FTZ / HMUL2 / HMUL2.BF16).
        fn tmul(a: Self, b: Self) -> Self;
    }
    impl Dst for f32 {
        #[inline(always)]
        fn from_f(v: f32) -> f32 {
            v
        }
        #[inline(always)]
        fn to_f(self) -> f32 {
            self
        }
        #[inline(always)]
        fn tmul(a: f32, b: f32) -> f32 {
            mul(a, b)
        }
    }
    /// f16 output.
    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct H(pub u16);
    /// bf16 output.
    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct B(pub u16);
    impl Dst for H {
        #[inline(always)]
        fn from_f(v: f32) -> H {
            H(f2h(v))
        }
        #[inline(always)]
        fn to_f(self) -> f32 {
            h2f(self.0 as u32)
        }
        #[inline(always)]
        fn tmul(a: H, b: H) -> H {
            H(f16x2::mul_f16x2(a.0 as u32, b.0 as u32) as u16)
        }
    }
    impl Dst for B {
        #[inline(always)]
        fn from_f(v: f32) -> B {
            B(f2bf(v))
        }
        #[inline(always)]
        fn to_f(self) -> f32 {
            bf2f(self.0 as u32)
        }
        #[inline(always)]
        fn tmul(a: B, b: B) -> B {
            B(bf16x2::mul_bf16x2(a.0 as u32, b.0 as u32) as u16)
        }
    }

    /// (qk, qi, vdr, block bytes). FMT: 0 q4_0, 1 q4_1, 2 q5_0, 3 q5_1, 4 q8_0, 5 q2_K, 6 q3_K,
    /// 7 q4_K, 8 q5_K, 9 q6_K.
    #[inline(always)]
    pub fn fmt_params<const FMT: u32>() -> (i32, i32, i32, i32) {
        match FMT {
            0 => (32, 4, 2, 18),
            1 => (32, 4, 2, 20),
            2 => (32, 4, 2, 22),
            3 => (32, 4, 2, 24),
            4 => (32, 8, 2, 34),
            5 => (256, 16, 1, 84),
            6 => (256, 16, 1, 110),
            7 => (256, 32, 2, 144),
            8 => (256, 32, 2, 176),
            _ => (256, 32, 1, 210),
        }
    }

    /// `tmp += vec_dot_<FMT>_q8_1(xb, yb, iqs)` with the reference's contraction of the `+=`.
    /// `xb`: the weight block, `yb`: the first Q8_1 block it pairs with.
    #[inline(always)]
    pub unsafe fn vdot<const FMT: u32>(tmp: f32, xb: *const u8, yb: *const u8, iqs: usize) -> f32 {
        match FMT {
            0 | 1 | 2 | 3 => {
                let mut sumi = 0i32;
                let mut i = 0;
                while i < 2 {
                    let u0 = yq(yb, iqs + i);
                    let u1 = yq(yb, iqs + i + 4);
                    let (vi0, vi1) = if FMT == 0 || FMT == 1 {
                        let v = if FMT == 0 { ld32u(xb, 2 + 4 * (iqs + i)) } else { ld32a(xb, 4 + 4 * (iqs + i)) };
                        (v & 0x0F0F0F0F, sar(v, 4) & 0x0F0F0F0F)
                    } else {
                        let (vl, qh) = if FMT == 2 {
                            (ld32u(xb, 6 + 4 * (iqs + i)), ld32u(xb, 2))
                        } else {
                            (ld32a(xb, 8 + 4 * (iqs + i)), ld32a(xb, 4))
                        };
                        let vh = sar(qh, 4 * (iqs + i));
                        let mut vi0 = vl & 0x0F0F0F0F;
                        vi0 |= (vh << 4) & 0x00000010;
                        vi0 |= (vh << 11) & 0x00001000;
                        vi0 |= (vh << 18) & 0x00100000;
                        vi0 |= (vh << 25) & 0x10000000;
                        let mut vi1 = sar(vl, 4) & 0x0F0F0F0F;
                        vi1 |= sar(vh, 12) & 0x00000010;
                        vi1 |= sar(vh, 5) & 0x00001000;
                        vi1 |= (vh << 2) & 0x00100000;
                        vi1 |= (vh << 9) & 0x10000000;
                        (vi0, vi1)
                    };
                    sumi = dp4a(vi0, u0, sumi);
                    sumi = dp4a(vi1, u1, sumi);
                    i += 1;
                }
                let ds = ld32a(yb, 0);
                let dsx = h2f(ds);
                let dsy = h2f(ds >> 16);
                let sf = sumi as f32;
                if FMT == 0 || FMT == 2 {
                    let c = if FMT == 0 { -4.0 } else { -8.0 };
                    fma(h2f(ld16(xb, 0)), fma(dsx, sf, mul(dsy, c)), tmp)
                } else {
                    let dm = ld32a(xb, 0);
                    let dd = mul(h2f(dm), dsx);
                    let ms = mul(mul(h2f(dm >> 16), dsy), 0.5);
                    add(fma(dd, sf, ms), tmp)
                }
            }
            4 => {
                let mut sumi = 0i32;
                let mut i = 0;
                while i < 2 {
                    sumi = dp4a(ld32u(xb, 2 + 4 * (iqs + i)), yq(yb, iqs + i), sumi);
                    i += 1;
                }
                fma(mul(sumi as f32, h2f(ld16(xb, 0))), h2f(ld16(yb, 0)), tmp)
            }
            5 => {
                let bq8_offset = 4 * (iqs / 8);
                let scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
                let v = ld32a(xb, 16 + 4 * iqs);
                let mut sumf_d = 0f32;
                let mut sumf_m = 0f32;
                let mut i = 0;
                while i < 4 {
                    let b8 = yb.add((bq8_offset + i) * 36);
                    let u = yq(b8, iqs % 8);
                    let d8 = h2f(ld16(b8, 0));
                    let sc = ld8(xb, scale_offset + 2 * i);
                    let vi = (v >> (2 * i)) & 0x03030303;
                    sumf_d = fma(d8, dp4a(vi, u, 0).wrapping_mul((sc & 0xF) as i32) as f32, sumf_d);
                    let mut m = sc >> 4;
                    m |= m << 8;
                    m |= m << 16;
                    sumf_m = fma(d8, dp4a(m, u, 0) as f32, sumf_m);
                    i += 1;
                }
                let dm = ld32a(xb, 80);
                add(fma(sumf_d, h2f(dm), -mul(sumf_m, h2f(dm >> 16))), tmp)
            }
            6 => {
                let bq8_offset = 4 * (iqs / 8);
                let scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
                let vl = ld32u(xb, 32 + 4 * iqs);
                let vh = sar(!ld32u(xb, 4 * (iqs % 8)), bq8_offset);
                let mut sumf = 0f32;
                let mut i = 0;
                while i < 4 {
                    let b8 = yb.add((bq8_offset + i) * 36);
                    let u = yq(b8, iqs % 8);
                    let d8 = h2f(ld16(b8, 0));
                    let isc = scale_offset + 2 * i;
                    let sc_low = (ld8(xb, 96 + isc % 8) >> (4 * (isc / 8))) & 0xF;
                    let sc_high = ((ld8(xb, 96 + 8 + isc % 4) >> (2 * (isc / 4))) & 3) << 4;
                    let sc = (sc_low | sc_high) as i32 - 32;
                    let vil = (vl >> (2 * i)) & 0x03030303;
                    let vih = (sar(vh, i) << 2) & 0x04040404;
                    let vi = vsubss4u(vil, vih);
                    sumf = fma(d8, dp4a(vi, u, 0).wrapping_mul(sc) as f32, sumf);
                    i += 1;
                }
                fma(h2f(ld16(xb, 108)), sumf, tmp)
            }
            7 | 8 => {
                let bq8_offset = 2 * ((iqs / 2) / 4);
                let (v0, v1) = if FMT == 7 {
                    let q4 = 16 + 16 * bq8_offset + 4 * ((iqs / 2) % 4);
                    let (a, b) = (ld32a(xb, q4), ld32a(xb, q4 + 16));
                    ([a & 0x0F0F0F0F, b & 0x0F0F0F0F], [sar(a, 4) & 0x0F0F0F0F, sar(b, 4) & 0x0F0F0F0F])
                } else {
                    let ql = 48 + 16 * bq8_offset + 4 * ((iqs / 2) % 4);
                    let qh = 16 + 4 * ((iqs / 2) % 4);
                    let (l0, l1) = (ld32a(xb, ql), ld32a(xb, ql + 16));
                    let h0 = sar(ld32a(xb, qh), bq8_offset);
                    let h1 = sar(ld32a(xb, qh + 16), bq8_offset);
                    (
                        [(l0 & 0x0F0F0F0F) | ((h0 << 4) & 0x10101010), (l1 & 0x0F0F0F0F) | ((h1 << 4) & 0x10101010)],
                        [
                            (sar(l0, 4) & 0x0F0F0F0F) | ((sar(h0, 1) << 4) & 0x10101010),
                            (sar(l1, 4) & 0x0F0F0F0F) | ((sar(h1, 1) << 4) & 0x10101010),
                        ],
                    )
                };
                let v = [v0, v1];
                let s16 = |k: usize| ld16(xb, 4 + 2 * k);
                let j = bq8_offset / 2;
                let (aux0, aux1) = if j < 2 {
                    (s16(j) & 0x3f3f, s16(j + 2) & 0x3f3f)
                } else {
                    (
                        (s16(j + 2) & 0x0f0f) | ((s16(j - 2) & 0xc0c0) >> 2),
                        ((s16(j + 2) >> 4) & 0x0f0f) | ((s16(j) & 0xc0c0) >> 2),
                    )
                };
                let sc = [aux0 & 0xFF, aux0 >> 8];
                let m = [aux1 & 0xFF, aux1 >> 8];
                let mut sumf_d = 0f32;
                let mut sumf_m = 0f32;
                let mut i = 0;
                while i < 2 {
                    let b8 = yb.add((bq8_offset + i) * 36);
                    let d8 = h2f(ld16(b8, 0));
                    let u0 = yq(b8, (iqs / 2) % 4);
                    let u1 = yq(b8, (iqs / 2) % 4 + 4);
                    let dot1 = dp4a(v[i][1], u1, dp4a(v[i][0], u0, 0));
                    let dot2 = dp4a(0x01010101, u1, dp4a(0x01010101, u0, 0));
                    sumf_d = fma(d8, dot1.wrapping_mul(sc[i] as i32) as f32, sumf_d);
                    sumf_m = fma(d8, dot2.wrapping_mul(m[i] as i32) as f32, sumf_m);
                    i += 1;
                }
                let dm = ld32a(xb, 0);
                add(fma(sumf_d, h2f(dm), -mul(sumf_m, h2f(dm >> 16))), tmp)
            }
            _ => {
                let bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
                let scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
                let vh_shift = 2 * ((iqs % 16) / 8);
                let vl = ld32u(xb, 4 * iqs);
                let vh = sar(ld32u(xb, 128 + 4 * (8 * (iqs / 16) + iqs % 8)), vh_shift);
                let mut sumf = 0f32;
                let mut i = 0;
                while i < 2 {
                    let b8 = yb.add((bq8_offset + 2 * i) * 36);
                    let u = yq(b8, iqs % 8);
                    let d8 = h2f(ld16(b8, 0));
                    let sc = *xb.add(192 + scale_offset + 4 * i) as i8 as i32;
                    let vil = sar(vl, 4 * i) & 0x0F0F0F0F;
                    let vih = (sar(vh, 4 * i) << 4) & 0x30303030;
                    let vi = vsubss4u(vil | vih, 0x20202020);
                    sumf = fma(d8, dp4a(vi, u, 0).wrapping_mul(sc) as f32, sumf);
                    i += 1;
                }
                fma(h2f(ld16(xb, 208)), sumf, tmp)
            }
        }
    }


    /// Plain GEMV geometry (MMVQ_NWARPS_SINGLE_COL_PLAIN = 8, rows 2 for every batch size).
    #[inline(always)]
    pub const fn plain_nwarps(nc: usize) -> i32 {
        if nc == 1 { 8 } else if nc <= 4 { 4 } else { 2 }
    }
    /// Fused-GLU geometry: 4 warps x 2 rows for one column, else 2 warps x (1 | 2) rows.
    #[inline(always)]
    pub const fn glu_nwarps(nc: usize) -> i32 {
        if nc == 1 { 4 } else { 2 }
    }
    #[inline(always)]
    pub const fn glu_rpb(nc: usize) -> i32 {
        if nc == 1 { 2 } else if nc <= 4 { 1 } else { 2 }
    }
    /// Fused-QKV geometry: (8 | 4 | 2) warps x 2 rows.
    #[inline(always)]
    pub const fn qkv_nwarps(nc: usize) -> i32 {
        if nc == 1 { 8 } else if nc <= 4 { 4 } else { 2 }
    }

    /// The k loop of every mmvq core: `tmp[j][i] += vec_dot(vx, &y[j*stride_col_y + kby],
    /// row*blocks_per_row + kbx, kqs)` for rows row0 + i (i < RPB, rows >= `nrows_guard` skipped
    /// when `GUARD`).
    #[inline(always)]
    pub unsafe fn kloop<const FMT: u32, const NC: usize, const GUARD: bool>(
        tmp: &mut [[f32; 2]; NC], vx: *const u8, vy: *const u8, ncols_x: i32, row0: i32, rpb: i32, nwarps: i32,
        stride_col_y: i32, nrows_guard: i32,
    ) {
        let (qk, qi, vdr, bs) = fmt_params::<FMT>();
        let tid = 32 * (thread::threadIdx_y() as i32) + thread::threadIdx_x() as i32;
        let bpr = ncols_x / qk;
        let bpi = vdr * nwarps * 32 / qi;
        let mut kbx = tid / (qi / vdr);
        while kbx < bpr {
            let kby = kbx.wrapping_mul(qk / 32);
            let kqs = (vdr * (tid % (qi / vdr))) as usize;
            let mut j = 0;
            while j < NC {
                let yb = vy.offset((j as i32).wrapping_mul(stride_col_y).wrapping_add(kby) as isize * 36);
                let mut i = 0;
                while i < rpb as usize {
                    let row = row0.wrapping_add(i as i32);
                    if !GUARD || row < nrows_guard {
                        let wk = row.wrapping_mul(bpr).wrapping_add(kbx);
                        let xb = vx.offset(wk as isize * bs as isize);
                        tmp[j][i] = vdot::<FMT>(tmp[j][i], xb, yb, kqs);
                    }
                    i += 1;
                }
                j += 1;
            }
            kbx += bpi;
        }
    }

    /// Warps 1.. store their partials to `tmp_shared[ty - 1][j][i][lane]`.
    #[inline(always)]
    pub unsafe fn store_partials<const NC: usize>(tmp: &[[f32; 2]; NC], sh: *mut f32, rpb: i32) {
        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as i32;
        if ty > 0 {
            let mut j = 0;
            while j < NC {
                let mut i = 0;
                while i < rpb as usize {
                    *sh.add((((ty - 1) as usize * NC + j) * rpb as usize + i) * 32 + tx) = tmp[j][i];
                    i += 1;
                }
                j += 1;
            }
        }
    }

    /// Warp 0 adds the partials of warps 1.. in warp order, then a butterfly over xor 16 .. 1.
    #[inline(always)]
    pub unsafe fn reduce_partials<const NC: usize>(tmp: &mut [[f32; 2]; NC], sh: *const f32, rpb: i32, nwarps: i32, j: usize) {
        let tx = thread::threadIdx_x() as usize;
        let mut i = 0;
        while i < rpb as usize {
            let mut l = 0;
            while l < nwarps - 1 {
                tmp[j][i] = add(tmp[j][i], *sh.add(((l as usize * NC + j) * rpb as usize + i) * 32 + tx));
                l += 1;
            }
            let mut mask = 16;
            while mask > 0 {
                tmp[j][i] = add(tmp[j][i], warp::shuffle_xor_f32_sync(0xffff_ffff, tmp[j][i], mask));
                mask >>= 1;
            }
            i += 1;
        }
    }

    /// `mmvq_core_impl<dst_t, qk, qi, block_q_t, vdr, vec_dot, NC>`: block (32, nwarps), 2 rows.
    #[device]
    #[inline(always)]
    pub unsafe fn mmvq<const FMT: u32, const NC: usize, D: Dst>(
        vx: *const u8, vy: *const u8, dst: *mut D, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32,
    ) {
        // tmp_shared[nwarps - 1][NC][2][32]: at most 3 * 4 * 2 * 32 floats.
        static mut TMP_SHARED: SharedArray<f32, 768> = SharedArray::UNINIT;
        let nwarps = plain_nwarps(NC);
        let rpb: i32 = 2;
        let tx = thread::threadIdx_x() as i32;
        let row0 = rpb.wrapping_mul(thread::blockIdx_x() as i32);
        let mut tmp = [[0f32; 2]; NC];
        kloop::<FMT, NC, false>(&mut tmp, vx, vy, ncols_x, row0, rpb, nwarps, stride_col_y, 0);
        let sh = SharedArray::as_raw_mut_ptr(&raw mut TMP_SHARED);
        store_partials::<NC>(&tmp, sh, rpb);
        thread::sync_threads();
        if thread::threadIdx_y() > 0 {
            return;
        }
        let mut j = 0;
        while j < NC {
            reduce_partials::<NC>(&mut tmp, sh, rpb, nwarps, j);
            let row = (row0 as u32).wrapping_add(tx as u32);
            if tx < rpb && row < nrows_x as u32 {
                let v = if tx == 0 { tmp[j][0] } else { tmp[j][1] };
                let o = ((j as i32).wrapping_mul(stride_col_dst) as u32).wrapping_add(row);
                *dst.add(o as usize) = D::from_f(v);
            }
            j += 1;
        }
    }


    /// `NC` activation columns, each reduced exactly as `mmvq_core_impl<ncols_dst = 1>` reduces a
    /// single column (8 warps x 2 rows, the same k split per thread, the same shared-memory handoff
    /// order and butterfly), so column j is bit-identical to a batch-1 launch on column j alone. The
    /// b_size 2..8 kernels of mmvq_gguf.cu use 4 or 2 warps and round differently. Unlike one
    /// batch-1 launch per column, the weight rows are read once for all NC columns.
    #[device]
    #[inline(always)]
    pub unsafe fn mmvq_rows<const FMT: u32, const NC: usize, D: Dst>(
        vx: *const u8, vy: *const u8, dst: *mut D, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32,
    ) {
        // tmp_shared[7][NC][2][32]: at most 7 * 8 * 2 * 32 floats.
        static mut TMP_ROWS: SharedArray<f32, 3584> = SharedArray::UNINIT;
        let nwarps: i32 = 8;
        let rpb: i32 = 2;
        let tx = thread::threadIdx_x() as i32;
        let row0 = rpb.wrapping_mul(thread::blockIdx_x() as i32);
        let mut tmp = [[0f32; 2]; NC];
        kloop::<FMT, NC, false>(&mut tmp, vx, vy, ncols_x, row0, rpb, nwarps, stride_col_y, 0);
        let sh = SharedArray::as_raw_mut_ptr(&raw mut TMP_ROWS);
        store_partials::<NC>(&tmp, sh, rpb);
        thread::sync_threads();
        if thread::threadIdx_y() > 0 {
            return;
        }
        let mut j = 0;
        while j < NC {
            reduce_partials::<NC>(&mut tmp, sh, rpb, nwarps, j);
            let row = (row0 as u32).wrapping_add(tx as u32);
            if tx < rpb && row < nrows_x as u32 {
                let v = if tx == 0 { tmp[j][0] } else { tmp[j][1] };
                let o = ((j as i32).wrapping_mul(stride_col_dst) as u32).wrapping_add(row);
                *dst.add(o as usize) = D::from_f(v);
            }
            j += 1;
        }
    }

    // GENERATED KERNELS BEGIN
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_rows1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<0, 1, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_rows2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<0, 2, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_rows3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<0, 3, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_rows4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<0, 4, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_rows5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<0, 5, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_rows6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<0, 6, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_rows7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<0, 7, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_rows8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<0, 8, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_rows1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<0, 1, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_rows2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<0, 2, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_rows3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<0, 3, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_rows4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<0, 4, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_rows5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<0, 5, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_rows6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<0, 6, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_rows7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<0, 7, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_rows8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<0, 8, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_rows1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<0, 1, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_rows2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<0, 2, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_rows3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<0, 3, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_rows4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<0, 4, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_rows5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<0, 5, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_rows6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<0, 6, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_rows7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<0, 7, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_rows8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<0, 8, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_rows1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<1, 1, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_rows2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<1, 2, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_rows3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<1, 3, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_rows4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<1, 4, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_rows5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<1, 5, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_rows6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<1, 6, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_rows7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<1, 7, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_rows8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<1, 8, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_rows1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<1, 1, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_rows2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<1, 2, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_rows3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<1, 3, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_rows4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<1, 4, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_rows5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<1, 5, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_rows6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<1, 6, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_rows7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<1, 7, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_rows8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<1, 8, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_rows1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<1, 1, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_rows2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<1, 2, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_rows3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<1, 3, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_rows4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<1, 4, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_rows5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<1, 5, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_rows6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<1, 6, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_rows7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<1, 7, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_rows8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<1, 8, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_rows1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<2, 1, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_rows2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<2, 2, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_rows3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<2, 3, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_rows4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<2, 4, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_rows5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<2, 5, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_rows6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<2, 6, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_rows7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<2, 7, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_rows8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<2, 8, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_rows1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<2, 1, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_rows2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<2, 2, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_rows3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<2, 3, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_rows4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<2, 4, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_rows5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<2, 5, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_rows6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<2, 6, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_rows7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<2, 7, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_rows8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<2, 8, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_rows1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<2, 1, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_rows2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<2, 2, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_rows3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<2, 3, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_rows4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<2, 4, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_rows5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<2, 5, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_rows6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<2, 6, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_rows7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<2, 7, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_rows8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<2, 8, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_rows1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<3, 1, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_rows2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<3, 2, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_rows3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<3, 3, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_rows4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<3, 4, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_rows5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<3, 5, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_rows6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<3, 6, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_rows7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<3, 7, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_rows8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<3, 8, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_rows1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<3, 1, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_rows2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<3, 2, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_rows3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<3, 3, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_rows4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<3, 4, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_rows5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<3, 5, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_rows6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<3, 6, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_rows7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<3, 7, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_rows8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<3, 8, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_rows1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<3, 1, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_rows2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<3, 2, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_rows3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<3, 3, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_rows4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<3, 4, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_rows5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<3, 5, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_rows6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<3, 6, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_rows7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<3, 7, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_rows8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<3, 8, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_rows1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<4, 1, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_rows2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<4, 2, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_rows3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<4, 3, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_rows4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<4, 4, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_rows5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<4, 5, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_rows6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<4, 6, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_rows7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<4, 7, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_rows8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<4, 8, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_rows1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<4, 1, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_rows2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<4, 2, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_rows3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<4, 3, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_rows4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<4, 4, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_rows5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<4, 5, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_rows6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<4, 6, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_rows7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<4, 7, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_rows8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<4, 8, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_rows1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<4, 1, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_rows2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<4, 2, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_rows3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<4, 3, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_rows4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<4, 4, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_rows5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<4, 5, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_rows6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<4, 6, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_rows7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<4, 7, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_rows8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<4, 8, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_rows1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<5, 1, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_rows2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<5, 2, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_rows3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<5, 3, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_rows4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<5, 4, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_rows5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<5, 5, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_rows6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<5, 6, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_rows7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<5, 7, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_rows8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<5, 8, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_rows1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<5, 1, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_rows2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<5, 2, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_rows3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<5, 3, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_rows4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<5, 4, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_rows5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<5, 5, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_rows6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<5, 6, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_rows7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<5, 7, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_rows8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<5, 8, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_rows1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<5, 1, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_rows2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<5, 2, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_rows3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<5, 3, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_rows4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<5, 4, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_rows5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<5, 5, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_rows6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<5, 6, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_rows7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<5, 7, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_rows8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<5, 8, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_rows1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<6, 1, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_rows2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<6, 2, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_rows3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<6, 3, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_rows4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<6, 4, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_rows5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<6, 5, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_rows6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<6, 6, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_rows7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<6, 7, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_rows8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<6, 8, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_rows1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<6, 1, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_rows2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<6, 2, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_rows3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<6, 3, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_rows4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<6, 4, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_rows5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<6, 5, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_rows6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<6, 6, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_rows7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<6, 7, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_rows8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<6, 8, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_rows1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<6, 1, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_rows2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<6, 2, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_rows3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<6, 3, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_rows4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<6, 4, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_rows5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<6, 5, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_rows6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<6, 6, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_rows7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<6, 7, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_rows8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<6, 8, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_rows1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<7, 1, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_rows2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<7, 2, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_rows3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<7, 3, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_rows4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<7, 4, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_rows5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<7, 5, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_rows6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<7, 6, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_rows7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<7, 7, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_rows8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<7, 8, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_rows1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<7, 1, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_rows2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<7, 2, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_rows3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<7, 3, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_rows4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<7, 4, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_rows5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<7, 5, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_rows6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<7, 6, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_rows7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<7, 7, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_rows8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<7, 8, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_rows1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<7, 1, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_rows2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<7, 2, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_rows3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<7, 3, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_rows4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<7, 4, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_rows5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<7, 5, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_rows6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<7, 6, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_rows7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<7, 7, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_rows8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<7, 8, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_rows1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<8, 1, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_rows2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<8, 2, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_rows3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<8, 3, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_rows4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<8, 4, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_rows5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<8, 5, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_rows6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<8, 6, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_rows7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<8, 7, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_rows8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<8, 8, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_rows1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<8, 1, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_rows2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<8, 2, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_rows3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<8, 3, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_rows4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<8, 4, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_rows5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<8, 5, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_rows6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<8, 6, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_rows7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<8, 7, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_rows8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<8, 8, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_rows1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<8, 1, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_rows2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<8, 2, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_rows3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<8, 3, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_rows4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<8, 4, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_rows5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<8, 5, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_rows6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<8, 6, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_rows7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<8, 7, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_rows8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<8, 8, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_rows1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<9, 1, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_rows2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<9, 2, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_rows3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<9, 3, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_rows4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<9, 4, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_rows5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<9, 5, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_rows6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<9, 6, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_rows7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<9, 7, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_rows8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<9, 8, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_rows1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<9, 1, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_rows2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<9, 2, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_rows3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<9, 3, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_rows4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<9, 4, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_rows5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<9, 5, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_rows6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<9, 6, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_rows7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<9, 7, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_rows8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<9, 8, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_rows1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<9, 1, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_rows2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<9, 2, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_rows3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<9, 3, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_rows4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<9, 4, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_rows5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<9, 5, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_rows6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<9, 6, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_rows7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<9, 7, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_rows8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq_rows::<9, 8, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    // GENERATED KERNELS END
}

fn main() {
    std::process::exit(if gate::run() { 0 } else { 1 });
}
