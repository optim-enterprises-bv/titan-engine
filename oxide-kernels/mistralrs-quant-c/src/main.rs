#![allow(unsafe_op_in_unsafe_fn)]
//! mistral.rs `mistralrs-quant` group C in cuda-oxide: the GGUF GEMV kernels of
//! `kernels/mmvq_gguf/mmvq_gguf.cu` (240 plain + 240 fused-GLU + 240 fused-QKV
//! `mmvq_gguf_<q>_<dst>_<kind>_cudaN` + 3 `mmvq_gguf_quantize_q8_1_<t>`) and the tiled MMQ kernels
//! of `kernels/mmq_gguf/` (mul_mat_q / stream-k fixup per type x mmq_x x need_check, and the
//! quantize_mmq_q8_1 activation quantizers), plus (src/launch.rs) pure-Rust twins of every
//! extern "C" host launcher, bit-identical to libmistralrsquant.a (nvcc -O3 --use_fast_math,
//! sm_120a). Checked launcher-vs-launcher by the gate in src/gate.rs.
//!
//! These are close copies of candle's mmvq_gguf / mmq_gguf (ported in ../candle-mmvq and
//! ../candle-mmq); what differs, read off the source diff and the SASS:
//! - fast-math: every f32 add / mul / fma is `.ftz`, fabsf is `abs.ftz` (FADD.FTZ |x|, -RZ),
//!   fmaxf is `max.ftz`, `amax / 127` is a multiply by the f32 reciprocal of 127, `x / d` is
//!   rcp.approx.ftz + mul.ftz, `amax == 0` is setp.neu.ftz, activations use ex2 / tanh / rcp
//!   approximations and libdevice's ftz normcdff.
//! - mmvq plain b_size 1: 8 warps x 2 rows per CUDA block (candle: 4 warps x 1 row).
//! - mmvq fused GLU: gate and up GEMVs in one kernel, `(dst_t) act((float)(dst_t) gate) * (dst_t) up`
//!   (HMUL2 / HMUL2.BF16 / FMUL.FTZ in the output type); fused QKV: three GEMVs, blocks packed
//!   along x (gridDim.y == 1) or one per blockIdx.y, rows past a matrix skipped.
//! - mmq: runtime output type (f32 / f16 / bf16 via mmq_store_dst), MoE path (ids_dst /
//!   expert_bounds), llama.cpp fastdiv (uint3 params), tile-efficiency grid, fixup grid
//!   (grid_sk.x, mmq_y / 32) x (32, 4) with one output row per thread; quantizers read
//!   f32 / f16 / bf16 and a fused GLU(gate) * up quantizer.
#![allow(non_snake_case, clippy::missing_safety_doc)]
mod gate;
mod gate_mmq;
pub mod launch;

use cuda_device::{DynamicSharedArray, SharedArray, bf16x2, convert, device, dotprod, f16x2, float, kernel, launch_bounds, ptx_asm, thread, warp};
use cuda_host::cuda_module;

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

    /// `mmvq_core_fused_glu_impl`: gate and up GEMVs, `(dst_t) act((float) gate) * up`.
    #[device]
    #[inline(always)]
    pub unsafe fn mmvq_glu<const FMT: u32, const NC: usize, D: Dst>(
        vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut D, ncols_x: i32, nrows_x: i32,
        stride_col_y: i32, stride_col_dst: i32, activation: i32,
    ) {
        // tmp_shared_{gate,up}[nwarps - 1][NC][rpb][32]: at most 1 * 8 * 2 * 32 floats.
        static mut SH_GATE: SharedArray<f32, 512> = SharedArray::UNINIT;
        static mut SH_UP: SharedArray<f32, 512> = SharedArray::UNINIT;
        let nwarps = glu_nwarps(NC);
        let rpb = glu_rpb(NC);
        let tx = thread::threadIdx_x() as i32;
        let row0 = rpb.wrapping_mul(thread::blockIdx_x() as i32);
        let (qk, qi, vdr, bs) = fmt_params::<FMT>();
        let tid = 32 * (thread::threadIdx_y() as i32) + tx;
        let bpr = ncols_x / qk;
        let bpi = vdr * nwarps * 32 / qi;
        let mut tg = [[0f32; 2]; NC];
        let mut tu = [[0f32; 2]; NC];
        let mut kbx = tid / (qi / vdr);
        while kbx < bpr {
            let kby = kbx.wrapping_mul(qk / 32);
            let kqs = (vdr * (tid % (qi / vdr))) as usize;
            let mut j = 0;
            while j < NC {
                let yb = vy.offset((j as i32).wrapping_mul(stride_col_y).wrapping_add(kby) as isize * 36);
                let mut i = 0;
                while i < rpb as usize {
                    let wk = row0.wrapping_add(i as i32).wrapping_mul(bpr).wrapping_add(kbx);
                    tg[j][i] = vdot::<FMT>(tg[j][i], vx_gate.offset(wk as isize * bs as isize), yb, kqs);
                    tu[j][i] = vdot::<FMT>(tu[j][i], vx_up.offset(wk as isize * bs as isize), yb, kqs);
                    i += 1;
                }
                j += 1;
            }
            kbx += bpi;
        }
        let shg = SharedArray::as_raw_mut_ptr(&raw mut SH_GATE);
        let shu = SharedArray::as_raw_mut_ptr(&raw mut SH_UP);
        store_partials::<NC>(&tg, shg, rpb);
        store_partials::<NC>(&tu, shu, rpb);
        thread::sync_threads();
        if thread::threadIdx_y() > 0 {
            return;
        }
        let mut j = 0;
        while j < NC {
            reduce_partials::<NC>(&mut tg, shg, rpb, nwarps, j);
            reduce_partials::<NC>(&mut tu, shu, rpb, nwarps, j);
            let row = (row0 as u32).wrapping_add(tx as u32);
            if tx < rpb && (rpb == 1 || row < nrows_x as u32) {
                let (g, u) = if tx == 0 { (tg[j][0], tu[j][0]) } else { (tg[j][1], tu[j][1]) };
                let gate_val = D::from_f(g);
                let up_val = D::from_f(u);
                let activated = D::from_f(glu_act(gate_val.to_f(), activation));
                let o = ((j as i32).wrapping_mul(stride_col_dst) as u32).wrapping_add(row);
                *dst.add(o as usize) = D::tmul(activated, up_val);
            }
            j += 1;
        }
    }

    /// `mmvq_core_fused_qkv_impl`: three GEMVs sharing y; dst index `j * nrows + row`.
    #[device]
    #[inline(always)]
    pub unsafe fn mmvq_qkv<const FMT: u32, const NC: usize, D: Dst>(
        vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut D, k_dst: *mut D, v_dst: *mut D,
        ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32,
    ) {
        static mut TMP_SHARED: SharedArray<f32, 768> = SharedArray::UNINIT;
        let nwarps = qkv_nwarps(NC);
        let rpb: i32 = 2;
        let (vx, dst, nrows_x, row_block);
        if thread::gridDim_y() == 1 {
            let q_blocks = nrows_q.wrapping_add(rpb - 1) / rpb;
            let k_blocks = nrows_k.wrapping_add(rpb - 1) / rpb;
            let block = thread::blockIdx_x() as i32;
            if block < q_blocks {
                row_block = block;
                vx = vx_q;
                dst = q_dst;
                nrows_x = nrows_q;
            } else if block < q_blocks.wrapping_add(k_blocks) {
                row_block = block.wrapping_sub(q_blocks);
                vx = vx_k;
                dst = k_dst;
                nrows_x = nrows_k;
            } else {
                row_block = block.wrapping_sub(q_blocks).wrapping_sub(k_blocks);
                vx = vx_v;
                dst = v_dst;
                nrows_x = nrows_v;
            }
        } else {
            row_block = thread::blockIdx_x() as i32;
            let by = thread::blockIdx_y();
            if by == 0 {
                vx = vx_q;
                dst = q_dst;
                nrows_x = nrows_q;
            } else if by == 1 {
                vx = vx_k;
                dst = k_dst;
                nrows_x = nrows_k;
            } else {
                vx = vx_v;
                dst = v_dst;
                nrows_x = nrows_v;
            }
        }
        let tx = thread::threadIdx_x() as i32;
        let row0 = rpb.wrapping_mul(row_block);
        if row0 >= nrows_x {
            return;
        }
        let mut tmp = [[0f32; 2]; NC];
        kloop::<FMT, NC, true>(&mut tmp, vx, vy, ncols_x, row0, rpb, nwarps, stride_col_y, nrows_x);
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
                let o = ((j as i32).wrapping_mul(nrows_x) as u32).wrapping_add(row);
                *dst.add(o as usize) = D::from_f(v);
            }
            j += 1;
        }
    }

    /// `mmvq_gguf_quantize_q8_1_*` (fast-math): one thread per padded element, block (256, 1),
    /// grid (ceil(kx_padded / 256), rows). `d = amax * (1/127)`, `q = (int8) roundf(x * rcp(d))`.
    #[device]
    #[inline(always)]
    pub unsafe fn quantize<T: Copy, L: Fn(*const T, isize) -> f32>(x: *const T, vy: *mut u8, kx: i32, kx_padded: i32, load: L) {
        let ix = thread::blockDim_x().wrapping_mul(thread::blockIdx_x()).wrapping_add(thread::threadIdx_x()) as i32;
        if ix >= kx_padded {
            return;
        }
        let iy = thread::blockDim_y().wrapping_mul(thread::blockIdx_y()).wrapping_add(thread::threadIdx_y()) as i32;
        let i_padded = iy.wrapping_mul(kx_padded).wrapping_add(ix);
        let ib = i_padded / 32;
        let iqs = i_padded % 32;
        let xi = if ix < kx { load(x, iy.wrapping_mul(kx).wrapping_add(ix) as isize) } else { 0.0 };
        let mut amax = fabsf(xi);
        let mut sum = xi;
        let mut mask = 16;
        while mask > 0 {
            amax = fmaxf(amax, warp::shuffle_xor_f32_sync(0xffff_ffff, amax, mask));
            sum = add(sum, warp::shuffle_xor_f32_sync(0xffff_ffff, sum, mask));
            mask >>= 1;
        }
        let d = mul(amax, f32::from_bits(0x3c01_0204));
        let q = if neu0_ftz(amax) { roundf_i(div_approx(xi, d)) } else { 0 };
        let y = vy.offset(ib as isize * 36);
        *y.offset(4 + iqs as isize) = q as u8;
        if iqs > 0 {
            return;
        }
        *(y as *mut u16) = f2h(d);
        *(y.add(2) as *mut u16) = f2h(sum);
    }

    #[kernel]
    pub unsafe fn mmvq_gguf_quantize_q8_1_bf16(x: *const u16, vy: *mut u8, kx: i32, kx_padded: i32) {
        quantize(x, vy, kx, kx_padded, |p, i| bf2f(*p.offset(i) as u32))
    }
    #[kernel]
    pub unsafe fn mmvq_gguf_quantize_q8_1_f16(x: *const u16, vy: *mut u8, kx: i32, kx_padded: i32) {
        quantize(x, vy, kx, kx_padded, |p, i| h2f(*p.offset(i) as u32))
    }
    #[kernel]
    pub unsafe fn mmvq_gguf_quantize_q8_1_f32(x: *const f32, vy: *mut u8, kx: i32, kx_padded: i32) {
        quantize(x, vy, kx, kx_padded, |p, i| *p.offset(i))
    }

    // ------------------------------------------------------------------------------------------
    // mmq_gguf (tiles, load_tiles, vec_dot: as candle-mmq, f32 epilogue ops now .ftz)

    // ggml_type ids.
    pub const Q4_0: u32 = 2;
    pub const Q4_1: u32 = 3;
    pub const Q5_0: u32 = 6;
    pub const Q5_1: u32 = 7;
    pub const Q8_0: u32 = 8;
    pub const Q2_K: u32 = 10;
    pub const Q3_K: u32 = 11;
    pub const Q4_K: u32 = 12;
    pub const Q5_K: u32 = 13;
    pub const Q6_K: u32 = 14;

    const WARP: i32 = 32;
    const NWARPS: i32 = 8;
    const MMQ_Y: i32 = 128;
    const TILE_NE_K: i32 = 32; // MMQ_TILE_NE_K
    const TILE_Y_K: i32 = 36; // MMQ_TILE_Y_K: 32 quant ints + 4 scale ints per column
    const ITER_K: i32 = 256;
    const SZ: i32 = 36; // sizeof(block_q8_1_mmq) / sizeof(int)

    #[inline(always)]
    pub const fn qk(t: u32) -> i32 {
        match t {
            Q2_K | Q3_K | Q4_K | Q5_K | Q6_K => 256,
            _ => 32,
        }
    }

    /// Block size in bytes of the x quant type.
    #[inline(always)]
    pub const fn block_bytes(t: u32) -> i32 {
        match t {
            Q4_0 => 18,
            Q4_1 => 20,
            Q5_0 => 22,
            Q5_1 => 24,
            Q8_0 => 34,
            Q2_K => 84,
            Q3_K => 110,
            Q4_K => 144,
            Q5_K => 176,
            _ => 210, // Q6_K
        }
    }

    /// mmq_get_mma_tile_x_k.
    #[inline(always)]
    pub const fn tile_x_k(t: u32) -> i32 {
        match t {
            Q2_K => 100,
            Q3_K => 84,
            _ => 76, // Q8_0, Q8_1, Q6_K
        }
    }

    #[inline(always)]
    const fn granularity(mmq_x: i32) -> i32 {
        if mmq_x >= 48 { 16 } else { 8 }
    }

    /// Per-mmq_x constants as associated consts (the unroll pass needs literal loop bounds).
    pub struct X<const MMQ_X: i32>;
    impl<const MMQ_X: i32> X<MMQ_X> {
        pub const NTX: i32 = granularity(MMQ_X) / 8;
        pub const JSTEP: i32 = Self::NTX * 8;
        pub const YLEN: i32 = MMQ_X * TILE_Y_K;
    }

    macro_rules! unroll {
        () => {
            cuda_device::thread::__unroll_config::<0>();
        };
    }

    #[inline(always)]
    fn tx() -> i32 {
        thread::threadIdx_x() as i32
    }
    #[inline(always)]
    fn ty() -> i32 {
        thread::threadIdx_y() as i32
    }
    #[inline(always)]
    fn imin(a: i32, b: i32) -> i32 {
        if a < b { a } else { b }
    }

    /// Shared memory: ids (mmq_x ints) | tile_y (padded to 256 ints) | tile_x.
    #[inline(always)]
    fn tile_y<const MMQ_X: i32>() -> *mut i32 {
        let s: *mut i32 = DynamicSharedArray::<i32>::get();
        s.wrapping_add(MMQ_X as usize)
    }
    #[inline(always)]
    fn tile_x<const MMQ_X: i32>() -> *mut i32 {
        let s: *mut i32 = DynamicSharedArray::<i32>::get();
        s.wrapping_add((MMQ_X + (MMQ_X * TILE_Y_K + 255) / 256 * 256) as usize)
    }

    /// `(const block_t *) x + kbx0 + i*stride`: both int offsets sign-extended separately.
    #[inline(always)]
    fn xblock(x: *const u8, bb: i32, kbx0: i32, i: i32, stride: i32) -> *const u8 {
        let off = (kbx0 as i64).wrapping_add(i.wrapping_mul(stride) as i64);
        x.wrapping_offset(off.wrapping_mul(bb as i64) as isize)
    }

    #[inline(always)]
    unsafe fn ld32(p: *const u8, byte_off: usize) -> i32 {
        *(p.add(byte_off) as *const i32)
    }
    /// get_int_b2: two 16-bit loads.
    #[inline(always)]
    unsafe fn ld32_b2(p: *const u8, byte_off: usize) -> i32 {
        let q = p.add(byte_off) as *const u16;
        (*q as i32) | ((*q.add(1) as i32) << 16)
    }

    #[inline(always)]
    fn h2lo(w: u32) -> f32 {
        convert::cvt_f32_f16x2_lo(w)
    }
    #[inline(always)]
    fn h2hi(w: u32) -> f32 {
        convert::cvt_f32_f16x2_lo(w >> 16)
    }

    // ---------------------------------------------------------------------------------------
    // Tensor-core primitives (exact instructions of mmq_mma.cuh on sm_80+).

    /// tile<16,8,int> via ldmatrix.sync.aligned.m8n8.x4.b16 (generic address, as the reference).
    #[inline(always)]
    unsafe fn ldmatrix_a(xs0: *const i32, stride: i32) -> [u32; 4] {
        let lane = tx();
        let p = xs0.wrapping_offset(((lane % 16) * stride + (lane / 16) * 4) as isize);
        let (a0, a1, a2, a3): (u32, u32, u32, u32);
        ptx_asm!(
            "ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];",
            out("=r") a0, out("=r") a1, out("=r") a2, out("=r") a3,
            in("l") p as u64,
        );
        [a0, a1, a2, a3]
    }

    /// tile<8,8,int> load_generic: x[l] = xs0[(lane/4)*stride + l*4 + lane%4].
    #[inline(always)]
    unsafe fn load_b(xs0: *const i32, stride: i32) -> [u32; 2] {
        let lane = tx();
        let base = (lane / 4) * stride + lane % 4;
        [*xs0.offset(base as isize) as u32, *xs0.offset((base + 4) as isize) as u32]
    }

    /// mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 with a zero accumulator.
    #[inline(always)]
    unsafe fn mma_s8(a: [u32; 4], b: [u32; 2]) -> [i32; 4] {
        let (d0, d1, d2, d3): (i32, i32, i32, i32);
        ptx_asm!(
            "{ .reg .s32 z; mov.s32 z, 0; mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {z, z, z, z}; }",
            out("=r") d0, out("=r") d1, out("=r") d2, out("=r") d3,
            in("r") a[0], in("r") a[1], in("r") a[2], in("r") a[3], in("r") b[0], in("r") b[1],
            options(register_only),
        );
        [d0, d1, d2, d3]
    }


    /// tile<16,4,int> via ldmatrix.sync.aligned.m8n8.x2.b16.
    #[inline(always)]
    unsafe fn ldmatrix_a2(xs0: *const i32, stride: i32) -> [u32; 2] {
        let lane = tx();
        let p = xs0.wrapping_offset(((lane % 16) * stride) as isize);
        let (a0, a1): (u32, u32);
        ptx_asm!(
            "ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];",
            out("=r") a0, out("=r") a1,
            in("l") p as u64,
        );
        [a0, a1]
    }

    /// tile<8,4,int> load_generic: x[0] = xs0[(lane/4)*stride + lane%4].
    #[inline(always)]
    unsafe fn load_b4(xs0: *const i32, stride: i32) -> u32 {
        let lane = tx();
        *xs0.offset(((lane / 4) * stride + lane % 4) as isize) as u32
    }

    /// mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 with a zero accumulator.
    #[inline(always)]
    unsafe fn mma_s8_k16(a: [u32; 2], b: u32) -> [i32; 4] {
        let (d0, d1, d2, d3): (i32, i32, i32, i32);
        ptx_asm!(
            "{ .reg .s32 z; mov.s32 z, 0; mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {z, z, z, z}; }",
            out("=r") d0, out("=r") d1, out("=r") d2, out("=r") d3,
            in("r") a[0], in("r") a[1], in("r") b,
            options(register_only),
        );
        [d0, d1, d2, d3]
    }

    /// __vsubss4: per-byte signed saturating subtraction.
    #[inline(always)]
    fn vsubss4(a: i32, b: i32) -> i32 {
        let mut r: u32 = 0;
        let mut k = 0;
        while k < 4 {
            let x = ((a >> (8 * k)) as i8) as i32;
            let y = ((b >> (8 * k)) as i8) as i32;
            let d = x - y;
            let d = if d > 127 { 127 } else if d < -128 { -128 } else { d };
            r |= ((d as u8) as u32) << (8 * k);
            k += 1;
        }
        r as i32
    }

    #[inline(always)]
    unsafe fn half_at(p: *const u8, off: usize) -> f32 {
        h2lo(*(p.add(off) as *const u16) as u32)
    }

    /// tile<16,8>::get_i(l) / get_j(l).
    #[inline(always)]
    fn c_i(l: i32) -> i32 {
        (l / 2) * 8 + tx() / 4
    }
    #[inline(always)]
    fn c_j(l: i32) -> i32 {
        (tx() % 4) * 2 + l % 2
    }

    // ---------------------------------------------------------------------------------------
    // load_tiles (mma layout).

    #[inline(always)]
    fn unpack_scales_q45_k(s: [i32; 3], ksc: i32) -> i32 {
        // Register selects instead of a dynamically indexed (local-memory) array.
        let pick = |k: i32| if k == 0 { s[0] } else if k == 1 { s[1] } else { s[2] };
        let a = pick((ksc % 2) + (ksc != 0) as i32) >> (4 * (ksc & (ksc / 2)));
        let b = pick(ksc / 2) >> (2 * (ksc % 2));
        (a & 0x0F0F_0F0F) | (b & 0x3030_3030)
    }

    #[inline(always)]
    unsafe fn load_tiles_q4_k<const MMQ_X: i32, const NC: bool>(x: *const u8, kbx0: i32, i_max: i32, stride: i32) {
        let x_qs = tile_x::<MMQ_X>();
        let x_dm = x_qs.wrapping_add(2 * TILE_NE_K as usize) as *mut u32;
        let txi = tx();
        let mut i0 = 0;
        while i0 < MMQ_Y {
            unroll!();
            let mut i = i0 + ty();
            if NC {
                i = imin(i, i_max);
            }
            let bxi = xblock(x, 144, kbx0, i, stride);
            let qs0 = ld32(bxi, 16 + 4 * txi as usize);
            let o = i * 76 + 16 * (txi / 8) + txi % 8;
            *x_qs.offset(o as isize) = (qs0 >> 0) & 0x0F0F_0F0F;
            *x_qs.offset((o + 8) as isize) = (qs0 >> 4) & 0x0F0F_0F0F;
            i0 += NWARPS;
        }
        // rows_per_warp = 16, one pass over 128 rows.
        let mut i = (ty() * 16 + tx() / 2) % MMQ_Y;
        if NC {
            i = imin(i, i_max);
        }
        let bxi = xblock(x, 144, kbx0, i, stride);
        let s = [ld32(bxi, 4), ld32(bxi, 8), ld32(bxi, 12)];
        let ksc = tx() % 2;
        let sc32 = unpack_scales_q45_k(s, ksc) as u32;
        let m32 = unpack_scales_q45_k(s, ksc + 2) as u32;
        let dm = f16x2::mul_f16x2(*(bxi as *const u32), 0xBC00_3C00);
        let mut l = 0;
        while l < 4 {
            unroll!();
            let sc = ((sc32 >> (8 * l)) & 0xFF) as f32;
            let m = ((m32 >> (8 * l)) & 0xFF) as f32;
            *x_dm.offset((i * 76 + 4 * ksc + l) as isize) = f16x2::mul_f16x2(dm, convert::cvt_f16x2_f32(sc, m));
            l += 1;
        }
    }


    /// load_tiles_q5_K (mma): 5-bit quants as bytes, scales as q4_K.
    #[inline(always)]
    unsafe fn load_tiles_q5_k<const MMQ_X: i32, const NC: bool>(x: *const u8, kbx0: i32, i_max: i32, stride: i32) {
        let x_qs = tile_x::<MMQ_X>();
        let x_dm = x_qs.wrapping_add(2 * TILE_NE_K as usize) as *mut u32;
        let txi = tx();
        let mut i0 = 0;
        while i0 < MMQ_Y {
            unroll!();
            let mut i = i0 + ty();
            if NC {
                i = imin(i, i_max);
            }
            let bxi = xblock(x, 176, kbx0, i, stride);
            let ky = 2 * txi;
            let ql = ld32(bxi, 48 + 4 * txi as usize);
            let ql0 = (ql >> 0) & 0x0F0F_0F0F;
            let ql1 = (ql >> 4) & 0x0F0F_0F0F;
            let qh = ld32(bxi, 16 + 4 * (txi % 8) as usize);
            let qh0 = ((qh >> (2 * (txi / 8) + 0)) << 4) & 0x1010_1010;
            let qh1 = ((qh >> (2 * (txi / 8) + 1)) << 4) & 0x1010_1010;
            let kq0 = ky - ky % 16 + txi % 8;
            let kq1 = kq0 + 8;
            *x_qs.offset((i * 76 + kq0) as isize) = ql0 | qh0;
            *x_qs.offset((i * 76 + kq1) as isize) = ql1 | qh1;
            i0 += NWARPS;
        }
        let mut i = (ty() * 16 + tx() / 2) % MMQ_Y;
        if NC {
            i = imin(i, i_max);
        }
        let bxi = xblock(x, 176, kbx0, i, stride);
        let s = [ld32(bxi, 4), ld32(bxi, 8), ld32(bxi, 12)];
        let ksc = tx() % 2;
        let sc32 = unpack_scales_q45_k(s, ksc) as u32;
        let m32 = unpack_scales_q45_k(s, ksc + 2) as u32;
        let dm = f16x2::mul_f16x2(*(bxi as *const u32), 0xBC00_3C00);
        let mut l = 0;
        while l < 4 {
            unroll!();
            let sc = ((sc32 >> (8 * l)) & 0xFF) as f32;
            let m = ((m32 >> (8 * l)) & 0xFF) as f32;
            *x_dm.offset((i * 76 + 4 * ksc + l) as isize) = f16x2::mul_f16x2(dm, convert::cvt_f16x2_f32(sc, m));
            l += 1;
        }
    }

    /// load_tiles_q6_K (mma): signed 6-bit quants, float d per row, 16 int8 scales per row.
    #[inline(always)]
    unsafe fn load_tiles_q6_k<const MMQ_X: i32, const NC: bool>(x: *const u8, kbx0: i32, i_max: i32, stride: i32) {
        let x_qs = tile_x::<MMQ_X>();
        let x_df = x_qs.wrapping_add(2 * TILE_NE_K as usize) as *mut f32;
        let x_sc = x_qs.wrapping_add(2 * TILE_NE_K as usize + 1);
        let txi = tx();
        let mut i0 = 0;
        while i0 < MMQ_Y {
            unroll!();
            let mut i = i0 + ty();
            if NC {
                i = imin(i, i_max);
            }
            let bxi = xblock(x, 210, kbx0, i, stride);
            let ql = ld32_b2(bxi, 4 * txi as usize);
            let ql0 = (ql >> 0) & 0x0F0F_0F0F;
            let ql1 = (ql >> 4) & 0x0F0F_0F0F;
            let qh = ld32_b2(bxi, 128 + 4 * (8 * (txi / 16) + txi % 8) as usize);
            let qh0 = ((qh >> ((txi & 0x08) >> 2)) << 4) & 0x3030_3030;
            let qh1 = (qh >> ((txi & 0x08) >> 2)) & 0x3030_3030;
            let kq0 = 2 * txi - txi % 16;
            let kq1 = kq0 + 16;
            *x_qs.offset((i * 76 + kq0) as isize) = vsubss4(ql0 | qh0, 0x2020_2020);
            *x_qs.offset((i * 76 + kq1) as isize) = vsubss4(ql1 | qh1, 0x2020_2020);
            i0 += NWARPS;
        }
        {
            let mut i = (ty() * WARP + tx()) % MMQ_Y;
            if NC {
                i = imin(i, i_max);
            }
            let bxi = xblock(x, 210, kbx0, i, stride);
            *x_df.offset((i * 76) as isize) = half_at(bxi, 208);
        }
        let mut i0 = 0;
        while i0 < MMQ_Y {
            unroll!();
            let mut i = (i0 + ty() * 8 + tx() / 4) % MMQ_Y;
            if NC {
                i = imin(i, i_max);
            }
            let bxi = xblock(x, 210, kbx0, i, stride);
            *x_sc.offset((i * 76 + tx() % 4) as isize) = ld32_b2(bxi, 192 + 4 * (tx() % 4) as usize);
            i0 += NWARPS * 8;
        }
    }

    /// load_tiles_q8_0 (mma).
    #[inline(always)]
    unsafe fn load_tiles_q8_0<const MMQ_X: i32, const NC: bool>(x: *const u8, kbx0: i32, i_max: i32, stride: i32) {
        let x_qs = tile_x::<MMQ_X>();
        let x_df = x_qs.wrapping_add(2 * TILE_NE_K as usize) as *mut f32;
        let txi = tx();
        let kbx = txi / 8;
        let kqsx = txi % 8;
        let mut i0 = 0;
        while i0 < MMQ_Y {
            unroll!();
            let mut i = i0 + ty();
            if NC {
                i = imin(i, i_max);
            }
            let bxi = xblock(x, 34, kbx0, i, stride).wrapping_offset((kbx * 34) as isize);
            *x_qs.offset((i * 76 + txi) as isize) = ld32_b2(bxi, 2 + 4 * kqsx as usize);
            *x_qs.offset((i * 76 + TILE_NE_K + txi) as isize) = ld32_b2(bxi, 4 * 34 + 2 + 4 * kqsx as usize);
            i0 += NWARPS;
        }
        let kbxd = tx() % 8;
        let mut i0 = 0;
        while i0 < MMQ_Y {
            unroll!();
            let mut i = i0 + ty() * 4 + tx() / 8;
            if NC {
                i = imin(i, i_max);
            }
            let bxi = xblock(x, 34, kbx0, i, stride).wrapping_offset((kbxd * 34) as isize);
            *x_df.offset((i * 76 + kbxd) as isize) = half_at(bxi, 0);
            i0 += NWARPS * 4;
        }
    }

    /// load_tiles_q4_0 / q4_1 / q5_0 / q5_1 (mma): 32-value blocks, 8 per 256-value row slice.
    #[inline(always)]
    unsafe fn load_tiles_q4q5<const T: u32, const MMQ_X: i32, const NC: bool>(x: *const u8, kbx0: i32, i_max: i32, stride: i32) {
        let bb = block_bytes(T);
        let x_qs = tile_x::<MMQ_X>();
        let x_d = x_qs.wrapping_add(2 * TILE_NE_K as usize);
        let txi = tx();
        let kbx = txi / 4;
        let kqsx = txi % 4;
        let mut i0 = 0;
        while i0 < MMQ_Y {
            unroll!();
            let mut i = i0 + ty();
            if NC {
                i = imin(i, i_max);
            }
            let bxi = xblock(x, bb, kbx0, i, stride).wrapping_offset((kbx * bb) as isize);
            let o = i * 76 + kbx * 8 + kqsx;
            if T == Q4_0 || T == Q4_1 {
                let qs0 = if T == Q4_0 { ld32_b2(bxi, 2 + 4 * kqsx as usize) } else { ld32(bxi, 4 + 4 * kqsx as usize) };
                if T == Q4_0 {
                    *x_qs.offset(o as isize) = vsubss4((qs0 >> 0) & 0x0F0F_0F0F, 0x0808_0808);
                    *x_qs.offset((o + 4) as isize) = vsubss4((qs0 >> 4) & 0x0F0F_0F0F, 0x0808_0808);
                } else {
                    *x_qs.offset(o as isize) = (qs0 >> 0) & 0x0F0F_0F0F;
                    *x_qs.offset((o + 4) as isize) = (qs0 >> 4) & 0x0F0F_0F0F;
                }
            } else {
                let (ql, qh) = if T == Q5_0 {
                    (ld32_b2(bxi, 6 + 4 * kqsx as usize), ld32_b2(bxi, 2) >> (4 * kqsx))
                } else {
                    (ld32(bxi, 8 + 4 * kqsx as usize), ld32(bxi, 4) >> (4 * kqsx))
                };
                let mut qs0 = (ql >> 0) & 0x0F0F_0F0F;
                qs0 |= (qh << 4) & 0x0000_0010;
                qs0 |= (qh << 11) & 0x0000_1000;
                qs0 |= (qh << 18) & 0x0010_0000;
                qs0 |= (qh << 25) & 0x1000_0000;
                let mut qs1 = (ql >> 4) & 0x0F0F_0F0F;
                qs1 |= (qh >> 12) & 0x0000_0010;
                qs1 |= (qh >> 5) & 0x0000_1000;
                qs1 |= (qh << 2) & 0x0010_0000;
                qs1 |= (qh << 9) & 0x1000_0000;
                if T == Q5_0 {
                    qs0 = vsubss4(qs0, 0x1010_1010);
                    qs1 = vsubss4(qs1, 0x1010_1010);
                }
                *x_qs.offset(o as isize) = qs0;
                *x_qs.offset((o + 4) as isize) = qs1;
            }
            i0 += NWARPS;
        }
        let kbxd = tx() % 8;
        let mut i0 = 0;
        while i0 < MMQ_Y {
            unroll!();
            let mut i = i0 + ty() * 4 + tx() / 8;
            if NC {
                i = imin(i, i_max);
            }
            let bxi = xblock(x, bb, kbx0, i, stride).wrapping_offset((kbxd * bb) as isize);
            if T == Q4_0 || T == Q5_0 {
                *(x_d.offset((i * 76 + kbxd) as isize) as *mut f32) = half_at(bxi, 0);
            } else {
                *(x_d.offset((i * 76 + kbxd) as isize) as *mut u32) = *(bxi as *const u32);
            }
            i0 += NWARPS * 4;
        }
    }

    /// load_tiles_q2_K (mma): 2-bit quants, half2 (d*sc, m*min) per 16 values.
    #[inline(always)]
    unsafe fn load_tiles_q2_k<const MMQ_X: i32, const NC: bool>(x: *const u8, kbx0: i32, i_max: i32, stride: i32) {
        let x_qs = tile_x::<MMQ_X>();
        let x_dm = x_qs.wrapping_add(2 * TILE_NE_K as usize) as *mut u32;
        let kqsx = tx() % 16;
        let mut i0 = 0;
        while i0 < MMQ_Y {
            unroll!();
            let mut i = i0 + ty() * 2 + tx() / 16;
            if NC {
                i = imin(i, i_max);
            }
            let bxi = xblock(x, 84, kbx0, i, stride);
            let x_ql_0 = ld32_b2(bxi, 16 + 4 * kqsx as usize);
            let mut l = 0;
            while l < 4 {
                unroll!();
                let k = (kqsx / 8) * 32 + l * 8 + kqsx % 8;
                *x_qs.offset((i * 100 + k) as isize) = (x_ql_0 >> (2 * l)) & 0x0303_0303;
                l += 1;
            }
            let sc_m = *bxi.add(kqsx as usize) as u32;
            let dm = *(bxi.add(80) as *const u32);
            *x_dm.offset((i * 100 + kqsx) as isize) = f16x2::mul_f16x2(dm, convert::cvt_f16x2_f32((sc_m & 0x0F) as f32, (sc_m >> 4) as f32));
            i0 += 2 * NWARPS;
        }
    }

    /// load_tiles_q3_K (mma): signed 3-bit quants, float d*sc per 16 values.
    #[inline(always)]
    unsafe fn load_tiles_q3_k<const MMQ_X: i32, const NC: bool>(x: *const u8, kbx0: i32, i_max: i32, stride: i32) {
        let x_qs = tile_x::<MMQ_X>();
        let x_df = x_qs.wrapping_add(2 * TILE_NE_K as usize) as *mut f32;
        let kqsx = tx() % 16;
        let mut i0 = 0;
        while i0 < MMQ_Y {
            unroll!();
            let mut i = i0 + ty() * 2 + tx() / 16;
            if NC {
                i = imin(i, i_max);
            }
            let bxi = xblock(x, 110, kbx0, i, stride);
            let x_ql_0 = ld32_b2(bxi, 32 + 4 * kqsx as usize);
            let x_qh_0 = ld32_b2(bxi, 4 * (kqsx % 8) as usize) >> (4 * (kqsx / 8));
            let mut l = 0;
            while l < 4 {
                unroll!();
                let k = (kqsx / 8) * 32 + l * 8 + kqsx % 8;
                let x_ql_k = (x_ql_0 >> (2 * l)) & 0x0303_0303;
                let x_qh_k = ((x_qh_0 >> l) << 2) & 0x0404_0404;
                *x_qs.offset((i * 84 + k) as isize) = vsubss4(x_ql_k | x_qh_k, 0x0404_0404);
                l += 1;
            }
            i0 += 2 * NWARPS;
        }
        let mut i0 = 0;
        while i0 < MMQ_Y {
            unroll!();
            let mut i = i0 + ty() * 8 + tx() / 4;
            if NC {
                i = imin(i, i_max);
            }
            let bxi = xblock(x, 110, kbx0, i, stride);
            let ksc = tx() % 4;
            let ksc_low = ksc % 2;
            let shift_low = 4 * (ksc / 2);
            let sc_low = (ld32_b2(bxi, 96 + 4 * ksc_low as usize) >> shift_low) & 0x0F0F_0F0F;
            let sc_high = ((ld32_b2(bxi, 96 + 8) >> (2 * ksc)) << 4) & 0x3030_3030;
            let sc = vsubss4(sc_low | sc_high, 0x2020_2020);
            let d = half_at(bxi, 108);
            let mut l = 0;
            while l < 4 {
                unroll!();
                *x_df.offset((i * 84 + 4 * ksc + l) as isize) = mul(d, ((sc >> (8 * l)) as i8) as f32);
                l += 1;
            }
            i0 += NWARPS * 8;
        }
    }

    #[inline(always)]
    unsafe fn load_tiles<const T: u32, const MMQ_X: i32, const NC: bool>(x: *const u8, kbx0: i32, i_max: i32, stride: i32) {
        match T {
            Q4_K => load_tiles_q4_k::<MMQ_X, NC>(x, kbx0, i_max, stride),
            Q5_K => load_tiles_q5_k::<MMQ_X, NC>(x, kbx0, i_max, stride),
            Q6_K => load_tiles_q6_k::<MMQ_X, NC>(x, kbx0, i_max, stride),
            Q8_0 => load_tiles_q8_0::<MMQ_X, NC>(x, kbx0, i_max, stride),
            Q2_K => load_tiles_q2_k::<MMQ_X, NC>(x, kbx0, i_max, stride),
            Q3_K => load_tiles_q3_k::<MMQ_X, NC>(x, kbx0, i_max, stride),
            _ => load_tiles_q4q5::<T, MMQ_X, NC>(x, kbx0, i_max, stride),
        }
    }

    // ---------------------------------------------------------------------------------------
    // vec_dot (mma).

    /// vec_dot_q8_1_q8_1_mma: x tile = 64 ints of 8-bit quants + 8 half2 (d*scale, -m*min) per row,
    /// y tile = per column 4 half2 (d, s) + 32 ints.
    #[inline(always)]
    unsafe fn vec_dot_q8_1_q8_1<const MMQ_X: i32>(sum: &mut [f32; 64], k00: i32) {
        let ntx = X::<MMQ_X>::NTX;
        let rows_per_warp = 2 * granularity(MMQ_X);
        let y = tile_y::<MMQ_X>().wrapping_offset(((ty() % ntx) * (8 * TILE_Y_K)) as isize) as *const i32;
        let x_qs = tile_x::<MMQ_X>() as *const i32;
        let x_dm = x_qs.wrapping_add(2 * TILE_NE_K as usize) as *const u32;
        let y_qs = y.wrapping_add(4);
        let y_dm = y as *const u32;
        let i0 = (ty() / ntx) * rows_per_warp;

        let mut a = [[[0u32; 4]; 4]; 2];
        let mut dma = [[[0u32; 4]; 2]; 2];
        let mut n = 0;
        while n < X::<MMQ_X>::NTX {
            unroll!();
            let mut k01 = 0;
            while k01 < TILE_NE_K {
                unroll!();
                let k0 = k00 + k01;
                a[n as usize][(k01 / 8) as usize] = ldmatrix_a(x_qs.wrapping_offset(((i0 + n * 16) * 76 + k0) as isize), 76);
                k01 += 8;
            }
            let mut l = 0;
            while l < 2 {
                unroll!();
                let i = i0 + n * 16 + c_i(2 * l);
                let mut k01 = 0;
                while k01 < TILE_NE_K {
                    unroll!();
                    let k0 = k00 + k01;
                    dma[n as usize][l as usize][(k01 / 8) as usize] = *x_dm.offset((i * 76 + k0 / 8) as isize);
                    k01 += 8;
                }
                l += 1;
            }
            n += 1;
        }

        let mut j0 = 0;
        while j0 < MMQ_X {
            unroll!();
            let mut k01 = 0;
            while k01 < TILE_NE_K {
                unroll!();
                let b = load_b(y_qs.wrapping_offset((j0 * TILE_Y_K + k01) as isize), TILE_Y_K);
                let mut dsb = [0u32; 2];
                let mut l = 0;
                while l < 2 {
                    unroll!();
                    let j = j0 + c_j(l);
                    dsb[l as usize] = *y_dm.offset((j * TILE_Y_K + k01 / 8) as isize);
                    l += 1;
                }
                let mut n = 0;
                while n < X::<MMQ_X>::NTX {
                    unroll!();
                    let c = mma_s8(a[n as usize][(k01 / 8) as usize], b);
                    let mut l = 0;
                    while l < 4 {
                        unroll!();
                        let da = dma[n as usize][(l / 2) as usize][(k01 / 8) as usize];
                        let db = dsb[(l % 2) as usize];
                        let idx = ((j0 / 8 + n) * 4 + l) as usize;
                        let s = fma(mul(h2lo(da), h2lo(db)), c[l as usize] as f32, sum[idx]);
                        sum[idx] = fma(h2hi(da), h2hi(db), s);
                        l += 1;
                    }
                    n += 1;
                }
                k01 += 8;
            }
            j0 += X::<MMQ_X>::JSTEP;
        }
    }


    /// vec_dot_q8_0_q8_1_mma: x = 8-bit quants + one float d per 32 values; y scale is the float
    /// d (D4) or the low half of (d, s) (DS4). sum += (C*dA)*dB.
    #[inline(always)]
    unsafe fn vec_dot_q8_0_q8_1<const MMQ_X: i32, const DS4: bool>(sum: &mut [f32; 64], k00: i32) {
        let ntx = X::<MMQ_X>::NTX;
        let rows_per_warp = 2 * granularity(MMQ_X);
        let y = tile_y::<MMQ_X>().wrapping_offset(((ty() % ntx) * (8 * TILE_Y_K)) as isize) as *const i32;
        let x_qs = tile_x::<MMQ_X>() as *const i32;
        let x_df = x_qs.wrapping_add(2 * TILE_NE_K as usize) as *const f32;
        let y_qs = y.wrapping_add(4);
        let i0 = (ty() / ntx) * rows_per_warp;

        let mut a = [[[0u32; 4]; 4]; 2];
        let mut da = [[[0f32; 4]; 2]; 2];
        let mut n = 0;
        while n < X::<MMQ_X>::NTX {
            unroll!();
            let mut k01 = 0;
            while k01 < TILE_NE_K {
                unroll!();
                let k0 = k00 + k01;
                a[n as usize][(k01 / 8) as usize] = ldmatrix_a(x_qs.wrapping_offset(((i0 + n * 16) * 76 + k0) as isize), 76);
                k01 += 8;
            }
            let mut l = 0;
            while l < 2 {
                unroll!();
                let i = i0 + n * 16 + c_i(2 * l);
                let mut k01 = 0;
                while k01 < TILE_NE_K {
                    unroll!();
                    let k0 = k00 + k01;
                    da[n as usize][l as usize][(k01 / 8) as usize] = *x_df.offset((i * 76 + k0 / 8) as isize);
                    k01 += 8;
                }
                l += 1;
            }
            n += 1;
        }

        let mut j0 = 0;
        while j0 < MMQ_X {
            unroll!();
            let mut k01 = 0;
            while k01 < TILE_NE_K {
                unroll!();
                let b = load_b(y_qs.wrapping_offset((j0 * TILE_Y_K + k01) as isize), TILE_Y_K);
                let mut db = [0f32; 2];
                let mut l = 0;
                while l < 2 {
                    unroll!();
                    let j = j0 + c_j(l);
                    let w = *y.offset((j * TILE_Y_K + k01 / 8) as isize) as u32;
                    db[l as usize] = if DS4 { h2lo(w) } else { f32::from_bits(w) };
                    l += 1;
                }
                let mut n = 0;
                while n < X::<MMQ_X>::NTX {
                    unroll!();
                    let c = mma_s8(a[n as usize][(k01 / 8) as usize], b);
                    let mut l = 0;
                    while l < 4 {
                        unroll!();
                        let idx = ((j0 / 8 + n) * 4 + l) as usize;
                        let t = mul(c[l as usize] as f32, da[n as usize][(l / 2) as usize][(k01 / 8) as usize]);
                        sum[idx] = fma(t, db[(l % 2) as usize], sum[idx]);
                        l += 1;
                    }
                    n += 1;
                }
                k01 += 8;
            }
            j0 += X::<MMQ_X>::JSTEP;
        }
    }

    /// vec_dot_q8_0_16_q8_1_mma (Q3_K): float scale per 16 values, m16n8k16 mma.
    #[inline(always)]
    unsafe fn vec_dot_q8_0_16<const MMQ_X: i32>(sum: &mut [f32; 64], k00: i32) {
        let ntx = X::<MMQ_X>::NTX;
        let y = tile_y::<MMQ_X>().wrapping_offset(((ty() % ntx) * (8 * TILE_Y_K)) as isize) as *const i32;
        let x_qs = tile_x::<MMQ_X>() as *const i32;
        let x_df = x_qs.wrapping_add(2 * TILE_NE_K as usize) as *const f32;
        let y_qs = y.wrapping_add(4);
        let y_df = y as *const f32;
        let i0 = (ty() / ntx) * (ntx * 16);

        let mut a = [[[0u32; 2]; 8]; 2];
        let mut da = [[[0f32; 8]; 2]; 2];
        let mut n = 0;
        while n < X::<MMQ_X>::NTX {
            unroll!();
            let mut k01 = 0;
            while k01 < TILE_NE_K {
                unroll!();
                let k0 = k00 + k01;
                let r = ldmatrix_a(x_qs.wrapping_offset(((i0 + n * 16) * 84 + k0) as isize), 84);
                a[n as usize][(k01 / 4) as usize] = [r[0], r[1]];
                a[n as usize][(k01 / 4 + 1) as usize] = [r[2], r[3]];
                k01 += 8;
            }
            let mut l = 0;
            while l < 2 {
                unroll!();
                let i = i0 + n * 16 + c_i(2 * l);
                let mut k01 = 0;
                while k01 < TILE_NE_K {
                    unroll!();
                    let k0 = k00 + k01;
                    da[n as usize][l as usize][(k01 / 4) as usize] = *x_df.offset((i * 84 + k0 / 4) as isize);
                    k01 += 4;
                }
                l += 1;
            }
            n += 1;
        }

        let mut j0 = 0;
        while j0 < MMQ_X {
            unroll!();
            let mut k01 = 0;
            while k01 < TILE_NE_K {
                unroll!();
                let b0 = load_b4(y_qs.wrapping_offset((j0 * TILE_Y_K + k01) as isize), TILE_Y_K);
                let b1 = load_b4(y_qs.wrapping_offset((j0 * TILE_Y_K + k01 + 4) as isize), TILE_Y_K);
                let mut db = [0f32; 2];
                let mut l = 0;
                while l < 2 {
                    unroll!();
                    let j = j0 + c_j(l);
                    db[l as usize] = *y_df.offset((j * TILE_Y_K + k01 / 8) as isize);
                    l += 1;
                }
                let mut n = 0;
                while n < X::<MMQ_X>::NTX {
                    unroll!();
                    let c0 = mma_s8_k16(a[n as usize][(k01 / 4) as usize], b0);
                    let c1 = mma_s8_k16(a[n as usize][(k01 / 4 + 1) as usize], b1);
                    let mut l = 0;
                    while l < 4 {
                        unroll!();
                        let idx = ((j0 / 8 + n) * 4 + l) as usize;
                        let d = &da[n as usize][(l / 2) as usize];
                        let t = fma(c0[l as usize] as f32, d[(k01 / 4) as usize], mul(c1[l as usize] as f32, d[(k01 / 4 + 1) as usize]));
                        sum[idx] = fma(db[(l % 2) as usize], t, sum[idx]);
                        l += 1;
                    }
                    n += 1;
                }
                k01 += 8;
            }
            j0 += X::<MMQ_X>::JSTEP;
        }
    }

    /// vec_dot_q2_K_q8_1_mma (Turing+ path), D2S6 y layout.
    #[inline(always)]
    unsafe fn vec_dot_q2_k<const MMQ_X: i32>(sum: &mut [f32; 64], k00: i32) {
        let ntx = X::<MMQ_X>::NTX;
        let y = tile_y::<MMQ_X>().wrapping_offset(((ty() % ntx) * (8 * TILE_Y_K)) as isize) as *const i32;
        let x_qs = tile_x::<MMQ_X>() as *const i32;
        let x_dm = x_qs.wrapping_add(2 * TILE_NE_K as usize) as *const u32;
        let y_qs = y.wrapping_add(4);
        let y_ds = y as *const u32;
        let i0 = (ty() / ntx) * (ntx * 16);

        let mut a = [[[0u32; 2]; 8]; 2];
        let mut da = [[[0f32; 8]; 2]; 2];
        let mut ma = [[[0f32; 8]; 2]; 2];
        let mut n = 0;
        while n < X::<MMQ_X>::NTX {
            unroll!();
            let mut k01 = 0;
            while k01 < TILE_NE_K {
                unroll!();
                let k0 = k00 + k01;
                let r = ldmatrix_a(x_qs.wrapping_offset(((i0 + n * 16) * 100 + k0) as isize), 100);
                a[n as usize][(k01 / 4) as usize] = [r[0], r[1]];
                a[n as usize][(k01 / 4 + 1) as usize] = [r[2], r[3]];
                k01 += 8;
            }
            n += 1;
        }
        let mut n = 0;
        while n < X::<MMQ_X>::NTX {
            unroll!();
            let mut l = 0;
            while l < 2 {
                unroll!();
                let i = i0 + n * 16 + c_i(2 * l);
                let mut k01 = 0;
                while k01 < TILE_NE_K {
                    unroll!();
                    let k0 = k00 + k01;
                    let dm = *x_dm.offset((i * 100 + k0 / 4) as isize);
                    da[n as usize][l as usize][(k01 / 4) as usize] = h2lo(dm);
                    ma[n as usize][l as usize][(k01 / 4) as usize] = h2hi(dm);
                    k01 += 4;
                }
                l += 1;
            }
            n += 1;
        }

        let mut j0 = 0;
        while j0 < MMQ_X {
            unroll!();
            // One call per j0 in MIR: the inlined body alone exceeds the unroll clone budget.
            q2k_j0::<MMQ_X>(sum, j0, y_qs, y_ds, &a, &da, &ma);
            j0 += X::<MMQ_X>::JSTEP;
        }
    }

    /// One j0 column group of vec_dot_q2_K (inlined by LLVM after the MIR unroll).
    #[inline(always)]
    unsafe fn q2k_j0<const MMQ_X: i32>(sum: &mut [f32; 64], j0: i32, y_qs: *const i32, y_ds: *const u32,
                                      a: &[[[u32; 2]; 8]; 2], da: &[[[f32; 8]; 2]; 2], ma: &[[[f32; 8]; 2]; 2]) {
        let mut db = [0u32; 2];
        let mut l = 0;
        while l < 2 {
            unroll!();
            let j = j0 + c_j(l);
            db[l as usize] = *y_ds.offset((j * TILE_Y_K) as isize);
            l += 1;
        }
        let mut k01 = 0;
        while k01 < TILE_NE_K {
            unroll!();
            let b0 = load_b4(y_qs.wrapping_offset((j0 * TILE_Y_K + k01) as isize), TILE_Y_K);
            let b1 = load_b4(y_qs.wrapping_offset((j0 * TILE_Y_K + k01 + 4) as isize), TILE_Y_K);
            let mut cm0 = [0i32; 4];
            let mut cm1 = [0i32; 4];
            if k01 >= TILE_NE_K * 3 / 4 {
                cm0 = mma_s8_k16([0x0101_0101, 0x0101_0101], b0);
                cm1 = mma_s8_k16([0x0101_0101, 0x0101_0101], b1);
            }
            let mut n = 0;
            while n < X::<MMQ_X>::NTX {
                unroll!();
                let cd0 = mma_s8_k16(a[n as usize][(k01 / 4) as usize], b0);
                let cd1 = mma_s8_k16(a[n as usize][(k01 / 4 + 1) as usize], b1);
                let mut l = 0;
                while l < 4 {
                    unroll!();
                    let idx = ((j0 / 8 + n) * 4 + l) as usize;
                    let d = &da[n as usize][(l / 2) as usize];
                    let m = &ma[n as usize][(l / 2) as usize];
                    let lu = l as usize;
                    let mut tmp = fma(cd0[lu] as f32, d[(k01 / 4) as usize], mul(cd1[lu] as f32, d[(k01 / 4 + 1) as usize]));
                    if k01 >= TILE_NE_K * 3 / 4 {
                        tmp = sub(tmp, fma(cm0[lu] as f32, m[(k01 / 4) as usize], mul(cm1[lu] as f32, m[(k01 / 4 + 1) as usize])));
                    }
                    let w = db[(l % 2) as usize];
                    let dbv = if k01 < TILE_NE_K / 2 { h2lo(w) } else { h2hi(w) };
                    sum[idx] = fma(tmp, dbv, sum[idx]);
                    l += 1;
                }
                n += 1;
            }
            k01 += 8;
        }
        let mut k01 = 0;
        while k01 < TILE_NE_K * 3 / 4 {
            unroll!();
            let mut sb = [0u32; 2];
            let mut l = 0;
            while l < 2 {
                unroll!();
                let j = j0 + c_j(l);
                sb[l as usize] = *y_ds.offset((j * TILE_Y_K + 1 + k01 / 8) as isize);
                l += 1;
            }
            let mut n = 0;
            while n < X::<MMQ_X>::NTX {
                unroll!();
                let mut l = 0;
                while l < 4 {
                    unroll!();
                    let idx = ((j0 / 8 + n) * 4 + l) as usize;
                    let m = &ma[n as usize][(l / 2) as usize];
                    let w = sb[(l % 2) as usize];
                    sum[idx] = fma(-m[(k01 / 4) as usize], h2lo(w), sum[idx]);
                    sum[idx] = fma(-m[(k01 / 4 + 1) as usize], h2hi(w), sum[idx]);
                    l += 1;
                }
                n += 1;
            }
            k01 += 8;
        }
    }

    /// vec_dot_q6_K_q8_1_mma (Turing+ path).
    #[inline(always)]
    unsafe fn vec_dot_q6_k<const MMQ_X: i32>(sum: &mut [f32; 64], k00: i32) {
        let ntx = X::<MMQ_X>::NTX;
        let y = tile_y::<MMQ_X>().wrapping_offset(((ty() % ntx) * (8 * TILE_Y_K)) as isize) as *const i32;
        let x_qs = tile_x::<MMQ_X>() as *const i32;
        let x_df = x_qs.wrapping_add(2 * TILE_NE_K as usize) as *const f32;
        let x_sc = x_qs.wrapping_add(2 * TILE_NE_K as usize + 1);
        let y_qs = y.wrapping_add(4);
        let y_df = y as *const f32;
        let i0 = (ty() / ntx) * (ntx * 16);

        let mut a = [[[0u32; 2]; 8]; 2];
        let mut sca = [[[0i32; 8]; 2]; 2];
        let mut da = [[0f32; 2]; 2];
        let mut n = 0;
        while n < X::<MMQ_X>::NTX {
            unroll!();
            let mut k01 = 0;
            while k01 < TILE_NE_K {
                unroll!();
                let k0 = k00 + k01;
                let base = x_qs.wrapping_offset(((i0 + n * 16) * 76 + k0) as isize);
                a[n as usize][(k01 / 4) as usize] = ldmatrix_a2(base, 76);
                a[n as usize][(k01 / 4 + 1) as usize] = ldmatrix_a2(base.wrapping_add(4), 76);
                k01 += 8;
            }
            let mut k01 = 0;
            while k01 < TILE_NE_K {
                unroll!();
                let k0 = k00 + k01;
                let mut l = 0;
                while l < 2 {
                    unroll!();
                    let i = i0 + n * 16 + c_i(2 * l);
                    let sc_packed = *x_sc.offset((i * 76 + k0 / 16) as isize);
                    let mut ksc = 0;
                    while ksc < 4 {
                        unroll!();
                        sca[n as usize][l as usize][(k01 / 4 + ksc) as usize] = ((sc_packed >> (8 * ksc)) as i8) as i32;
                        ksc += 1;
                    }
                    l += 1;
                }
                k01 += 16;
            }
            let mut l = 0;
            while l < 2 {
                unroll!();
                let i = i0 + n * 16 + c_i(2 * l);
                da[n as usize][l as usize] = *x_df.offset((i * 76) as isize);
                l += 1;
            }
            n += 1;
        }

        let mut j0 = 0;
        while j0 < MMQ_X {
            unroll!();
            let mut tmp = [[0f32; 4]; 2];
            let mut k01 = 0;
            while k01 < TILE_NE_K {
                unroll!();
                let b0 = load_b4(y_qs.wrapping_offset((j0 * TILE_Y_K + k01) as isize), TILE_Y_K);
                let b1 = load_b4(y_qs.wrapping_offset((j0 * TILE_Y_K + 4 + k01) as isize), TILE_Y_K);
                let mut db = [0f32; 2];
                let mut l = 0;
                while l < 2 {
                    unroll!();
                    let j = j0 + c_j(l);
                    db[l as usize] = *y_df.offset((j * TILE_Y_K + k01 / 8) as isize);
                    l += 1;
                }
                let mut n = 0;
                while n < X::<MMQ_X>::NTX {
                    unroll!();
                    let c0 = mma_s8_k16(a[n as usize][(k01 / 4) as usize], b0);
                    let c1 = mma_s8_k16(a[n as usize][(k01 / 4 + 1) as usize], b1);
                    let mut l = 0;
                    while l < 4 {
                        unroll!();
                        let s = &sca[n as usize][(l / 2) as usize];
                        let iv = c0[l as usize].wrapping_mul(s[(k01 / 4) as usize])
                            .wrapping_add(c1[l as usize].wrapping_mul(s[(k01 / 4 + 1) as usize]));
                        tmp[n as usize][l as usize] = fma(iv as f32, db[(l % 2) as usize], tmp[n as usize][l as usize]);
                        l += 1;
                    }
                    n += 1;
                }
                k01 += 8;
            }
            let mut n = 0;
            while n < X::<MMQ_X>::NTX {
                unroll!();
                let mut l = 0;
                while l < 4 {
                    unroll!();
                    let idx = ((j0 / 8 + n) * 4 + l) as usize;
                    sum[idx] = fma(tmp[n as usize][l as usize], da[n as usize][(l / 2) as usize], sum[idx]);
                    l += 1;
                }
                n += 1;
            }
            j0 += X::<MMQ_X>::JSTEP;
        }
    }

    #[inline(always)]
    unsafe fn vec_dot<const T: u32, const MMQ_X: i32>(sum: &mut [f32; 64], k00: i32) {
        match T {
            Q4_1 | Q5_1 | Q4_K | Q5_K => vec_dot_q8_1_q8_1::<MMQ_X>(sum, k00),
            Q4_0 => vec_dot_q8_0_q8_1::<MMQ_X, true>(sum, k00),
            Q5_0 | Q8_0 => vec_dot_q8_0_q8_1::<MMQ_X, false>(sum, k00),
            Q2_K => vec_dot_q2_k::<MMQ_X>(sum, k00),
            Q3_K => vec_dot_q8_0_16::<MMQ_X>(sum, k00),
            _ => vec_dot_q6_k::<MMQ_X>(sum, k00),
        }
    }


    // ---------------------------------------------------------------------------------------
    // mistralrs mmq_gguf.cuh: typed write back, MoE ids, fastdiv, stream-k kernel + fixup.

    /// ggml_type ids of the output (mmq_store_dst / mmq_load_dst / mmq_type_size).
    pub const T_F32: i32 = 0;
    pub const T_F16: i32 = 1;
    pub const T_BF16: i32 = 30;

    /// llama.cpp fastdiv values <mp, L, d> (a `uint3`, passed as three u32 kernel params).
    #[derive(Clone, Copy)]
    pub struct Fd {
        pub mp: u32,
        pub l: u32,
        pub d: u32,
    }
    #[inline(always)]
    fn umulhi(a: u32, b: u32) -> u32 {
        ((a as u64 * b as u64) >> 32) as u32
    }
    /// fastdiv: `(__umulhi(n, mp) + n) >> L` (32-bit wrapping add).
    #[inline(always)]
    fn fastdiv(n: u32, f: Fd) -> u32 {
        umulhi(n, f.mp).wrapping_add(n) >> f.l
    }
    #[inline(always)]
    fn fastmodulo(n: u32, f: Fd) -> u32 {
        n.wrapping_sub(fastdiv(n, f).wrapping_mul(f.d))
    }

    /// mmq_store_dst: `((dst_t *) dst)[idx] = value` for F32 / F16 / BF16, nothing otherwise.
    #[inline(always)]
    unsafe fn store_dst(dst: *mut u8, idx: i32, v: f32, type_dst: i32) {
        match type_dst {
            T_F32 => *(dst as *mut f32).offset(idx as isize) = v,
            T_F16 => *(dst as *mut u16).offset(idx as isize) = f2h(v),
            T_BF16 => *(dst as *mut u16).offset(idx as isize) = f2bf(v),
            _ => {}
        }
    }
    /// mmq_load_dst.
    #[inline(always)]
    unsafe fn load_dst(dst: *const u8, idx: i32, type_dst: i32) -> f32 {
        match type_dst {
            T_F32 => *(dst as *const f32).offset(idx as isize),
            T_F16 => h2f(*(dst as *const u16).offset(idx as isize) as u32),
            T_BF16 => bf2f(*(dst as *const u16).offset(idx as isize) as u32),
            _ => 0.0,
        }
    }
    /// mmq_type_size.
    #[inline(always)]
    fn type_size(t: i32) -> i32 {
        if t == T_F16 || t == T_BF16 { 2 } else { 4 }
    }

    /// The ids_dst_shared array at the start of the dynamic shared memory.
    #[inline(always)]
    fn ids_sh() -> *mut i32 {
        DynamicSharedArray::<i32>::get()
    }

    /// mmq_write_back_mma: `mmq_store_dst(dst, ids_dst[j]*stride + i, sum, type_dst)`.
    #[inline(always)]
    unsafe fn write_back<const MMQ_X: i32, const NC: bool>(
        sum: &[f32; 64], dst: *mut u8, type_dst: i32, stride: i32, i_max: i32, j_max: i32,
    ) {
        let ids = ids_sh();
        let ntx = X::<MMQ_X>::NTX;
        let i0 = (ty() / ntx) * (ntx * 16);
        let mut j0 = 0;
        while j0 < MMQ_X {
            unroll!();
            let mut n = 0;
            while n < X::<MMQ_X>::NTX {
                unroll!();
                let mut l = 0;
                while l < 4 {
                    unroll!();
                    let j = j0 + (ty() % ntx) * 8 + c_j(l);
                    let i = i0 + n * 16 + c_i(l);
                    if j <= j_max && !(NC && i > i_max) {
                        let idx = (*ids.offset(j as isize)).wrapping_mul(stride).wrapping_add(i);
                        store_dst(dst, idx, sum[((j0 / 8 + n) * 4 + l) as usize], type_dst);
                    }
                    l += 1;
                }
                n += 1;
            }
            j0 += X::<MMQ_X>::JSTEP;
        }
    }

    #[inline(always)]
    unsafe fn load_tile_y<const MMQ_X: i32>(by0: *const i32) {
        let ty_ = tile_y::<MMQ_X>();
        let mut l0 = 0;
        while l0 < X::<MMQ_X>::YLEN {
            unroll!();
            let l = l0 + ty() * WARP + tx();
            *ty_.offset(l as isize) = *by0.offset(l as isize);
            l0 += NWARPS * WARP;
        }
    }

    /// mul_mat_q_process_tile.
    #[inline(always)]
    unsafe fn process_tile<const T: u32, const MMQ_X: i32, const NC: bool, const FIXUP: bool>(
        x: *const u8, offset_x: i32, y: *const i32, dst: *mut u8, type_dst: i32, tmp_fixup: *mut f32,
        stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, tile_x_max_i: i32, tile_y_max_j: i32,
        kb0_start: i32, kb0_stop: i32,
    ) {
        let qk = qk(T);
        let blocks_per_iter = ITER_K / qk;
        let mut sum = [0f32; 64];
        let mut kb0 = kb0_start;
        while kb0 < kb0_stop {
            load_tiles::<T, MMQ_X, NC>(x, offset_x.wrapping_add(kb0), tile_x_max_i, stride_row_x);
            let c = kb0.wrapping_mul(qk) / 128;
            load_tile_y::<MMQ_X>(y.wrapping_offset(ncols_y.wrapping_mul(c).wrapping_mul(SZ) as isize));
            thread::sync_threads();
            vec_dot::<T, MMQ_X>(&mut sum, 0);
            thread::sync_threads();
            load_tile_y::<MMQ_X>(y.wrapping_offset(ncols_y.wrapping_mul(c.wrapping_mul(SZ).wrapping_add(SZ)) as isize));
            thread::sync_threads();
            vec_dot::<T, MMQ_X>(&mut sum, TILE_NE_K);
            thread::sync_threads();
            kb0 += blocks_per_iter;
        }
        if FIXUP {
            let t = tmp_fixup.wrapping_offset((thread::blockIdx_x() as i32).wrapping_mul(MMQ_X * MMQ_Y) as isize);
            write_back::<MMQ_X, NC>(&sum, t as *mut u8, T_F32, MMQ_Y, MMQ_Y, MMQ_X);
        } else {
            write_back::<MMQ_X, NC>(&sum, dst, type_dst, stride_col_dst, tile_x_max_i, tile_y_max_j);
        }
    }

    /// `ids_dst_shared[j] = f(j)` for j < mmq_x (256 threads, one pass: mmq_x <= 128).
    #[inline(always)]
    unsafe fn fill_ids<const MMQ_X: i32, F: Fn(i32) -> i32>(f: F) {
        let j = ty() * WARP + tx();
        if !(NWARPS * WARP > MMQ_X && j >= MMQ_X) {
            *ids_sh().offset(j as isize) = f(j);
        }
    }

    /// The tile of stream-k index `kbc` (k block continuous): (it, wt, zt, jt).
    #[inline(always)]
    fn tile_of(kbc: i32, bpn: Fd, ntx: Fd, ncy: Fd, nsy: Fd) -> (i32, i32, i32, i32) {
        let tmp = fastdiv(kbc as u32, bpn);
        let (d1, jt) = (fastdiv(tmp, ntx), fastmodulo(tmp, ntx));
        let (d2, zt) = (fastdiv(d1, ncy), fastmodulo(d1, ncy));
        let (it, wt) = (fastdiv(d2, nsy), fastmodulo(d2, nsy));
        (it as i32, wt as i32, zt as i32, jt as i32)
    }

    /// Stream-k `mul_mat_q` (the sm_120 path of mistralrs' mmq_gguf.cuh), dense or MoE (ids_dst).
    #[inline(always)]
    pub unsafe fn mul_mat_q<const T: u32, const MMQ_X: i32, const NC: bool>(
        x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32,
        bpn: Fd, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32,
        channel_ratio: Fd, ncy: Fd, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32,
        sample_ratio: Fd, nsy: Fd, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32,
        type_dst: i32, ntx: Fd,
    ) {
        let nty = (nrows_x.wrapping_add(MMQ_Y - 1) / MMQ_Y) as u32;
        fill_ids::<MMQ_X, _>(|j| j);
        thread::sync_threads();

        let qk = qk(T);
        let bpi = (ITER_K / qk) as u32;
        let total = nsy.d.wrapping_mul(ncy.d).wrapping_mul(ntx.d).wrapping_mul(nty).wrapping_mul(bpn.d);
        let gdim = thread::gridDim_x() as i64;
        let bid = thread::blockIdx_x() as i64;
        let mut kbc = (bid.wrapping_mul(total as i64) / gdim) as i32;
        let mut kbc_stop = ((bid + 1).wrapping_mul(total as i64) / gdim) as i32;
        kbc = (kbc as u32).wrapping_sub(fastmodulo(kbc as u32, bpn) % bpi) as i32;
        kbc_stop = (kbc_stop as u32).wrapping_sub(fastmodulo(kbc_stop as u32, bpn) % bpi) as i32;

        let umin = |a: u32, b: u32| if a < b { a } else { b };
        let mut kb0_start = fastmodulo(kbc as u32, bpn) as i32;
        let mut kb0_stop = umin(bpn.d, (kb0_start.wrapping_add(kbc_stop).wrapping_sub(kbc)) as u32) as i32;
        while kbc < kbc_stop && kb0_stop == bpn.d as i32 {
            let (it, wt, zt, jt) = tile_of(kbc, bpn, ntx, ncy, nsy);
            let mut col_low = 0;
            let mut col_diff = ncols_dst;
            let mut offset_y = wt.wrapping_mul(stride_sample_y).wrapping_add(zt.wrapping_mul(stride_channel_y));
            let mut offset_dst = wt.wrapping_mul(stride_sample_dst).wrapping_add(zt.wrapping_mul(stride_channel_dst))
                .wrapping_add(jt.wrapping_mul(MMQ_X).wrapping_mul(stride_col_dst));
            if !ids_dst.is_null() {
                col_low = *expert_bounds.offset(zt as isize);
                let col_high = *expert_bounds.offset(zt as isize + 1);
                col_diff = col_high.wrapping_sub(col_low);
                offset_y = 0;
                offset_dst = 0;
                if jt.wrapping_mul(MMQ_X) >= col_diff {
                    kbc = kbc.wrapping_add(bpn.d as i32);
                    kbc = (kbc as u32).wrapping_sub(fastmodulo(kbc as u32, bpn)) as i32;
                    kb0_start = 0;
                    kb0_stop = umin(bpn.d, kbc_stop.wrapping_sub(kbc) as u32) as i32;
                    continue;
                }
                thread::sync_threads();
                let base = col_low.wrapping_add(jt.wrapping_mul(MMQ_X));
                fill_ids::<MMQ_X, _>(|j| *ids_dst.offset(base.wrapping_add(j) as isize));
                thread::sync_threads();
            }
            offset_y = offset_y.wrapping_add(col_low.wrapping_add(jt.wrapping_mul(MMQ_X)).wrapping_mul(SZ));
            offset_dst = offset_dst.wrapping_add(it.wrapping_mul(MMQ_Y));
            let tile_x_max_i = nrows_x.wrapping_sub(it.wrapping_mul(MMQ_Y)).wrapping_sub(1);
            let tile_y_max_j = col_diff.wrapping_sub(jt.wrapping_mul(MMQ_X)).wrapping_sub(1);
            let offset_x = fastdiv(wt as u32, sample_ratio).wrapping_mul(stride_sample_x as u32)
                .wrapping_add(fastdiv(zt as u32, channel_ratio).wrapping_mul(stride_channel_x as u32))
                .wrapping_add(it.wrapping_mul(MMQ_Y).wrapping_mul(stride_row_x) as u32) as i32;
            process_tile::<T, MMQ_X, NC, false>(
                x, offset_x, y.wrapping_offset(offset_y as isize),
                dst.wrapping_offset(offset_dst.wrapping_mul(type_size(type_dst)) as isize), type_dst, tmp_fixup,
                stride_row_x, ncols_y, stride_col_dst, tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop,
            );
            kbc = kbc.wrapping_add(bpn.d as i32);
            kbc = (kbc as u32).wrapping_sub(fastmodulo(kbc as u32, bpn)) as i32;
            kb0_start = 0;
            kb0_stop = umin(bpn.d, kbc_stop.wrapping_sub(kbc) as u32) as i32;
        }
        if kbc >= kbc_stop {
            return;
        }
        let (it, wt, zt, jt) = tile_of(kbc, bpn, ntx, ncy, nsy);
        let mut col_low = 0;
        let mut col_diff = ncols_dst;
        let mut offset_y = wt.wrapping_mul(stride_sample_y).wrapping_add(zt.wrapping_mul(stride_channel_y));
        let mut offset_dst = wt.wrapping_mul(stride_sample_dst).wrapping_add(zt.wrapping_mul(stride_channel_dst))
            .wrapping_add(jt.wrapping_mul(MMQ_X).wrapping_mul(stride_col_dst));
        if !ids_dst.is_null() {
            col_low = *expert_bounds.offset(zt as isize);
            let col_high = *expert_bounds.offset(zt as isize + 1);
            col_diff = col_high.wrapping_sub(col_low);
            offset_y = 0;
            offset_dst = 0;
            if jt.wrapping_mul(MMQ_X) >= col_diff {
                return;
            }
            // The fixup buffer is always contiguous: reset the ids.
            thread::sync_threads();
            fill_ids::<MMQ_X, _>(|j| j);
            thread::sync_threads();
        }
        offset_y = offset_y.wrapping_add(col_low.wrapping_add(jt.wrapping_mul(MMQ_X)).wrapping_mul(SZ));
        offset_dst = offset_dst.wrapping_add(it.wrapping_mul(MMQ_Y));
        let tile_x_max_i = nrows_x.wrapping_sub(it.wrapping_mul(MMQ_Y)).wrapping_sub(1);
        let tile_y_max_j = col_diff.wrapping_sub(jt.wrapping_mul(MMQ_X)).wrapping_sub(1);
        let offset_x = fastdiv(wt as u32, sample_ratio).wrapping_mul(stride_sample_x as u32)
            .wrapping_add(fastdiv(zt as u32, channel_ratio).wrapping_mul(stride_channel_x as u32))
            .wrapping_add(it.wrapping_mul(MMQ_Y).wrapping_mul(stride_row_x) as u32) as i32;
        process_tile::<T, MMQ_X, NC, true>(
            x, offset_x, y.wrapping_offset(offset_y as isize),
            dst.wrapping_offset(offset_dst.wrapping_mul(type_size(type_dst)) as isize), type_dst, tmp_fixup,
            stride_row_x, ncols_y, stride_col_dst, tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop,
        );
    }

    /// mul_mat_q_stream_k_fixup: grid (nblocks_sk, mmq_y / 32), block (32, 4); thread (x, y) owns
    /// output row `blockIdx.y*32 + x` of the columns `j0 + y`. Depends on the type only through qk.
    #[inline(always)]
    pub unsafe fn stream_k_fixup<const QK: i32, const MMQ_X: i32, const NC: bool>(
        ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32,
        bpn: Fd, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy: Fd, stride_channel_dst: i32, nsy: Fd,
        stride_sample_dst: i32, ntx: Fd,
    ) {
        static mut IDS: SharedArray<i32, 128> = SharedArray::UNINIT;
        const FW: i32 = 4; // nwarps of the fixup kernel
        let bpi = (ITER_K / QK) as u32;
        let mut sum = [0f32; 32];
        let i = (thread::blockIdx_y() as i32) * WARP + tx();
        let nty = nrows_x.wrapping_add(MMQ_Y - 1) / MMQ_Y;
        let total = nsy.d.wrapping_mul(ncy.d).wrapping_mul(ntx.d).wrapping_mul(nty as u32).wrapping_mul(bpn.d);
        let gdim = thread::gridDim_x() as i64;
        let bidx0 = thread::blockIdx_x() as i32;
        let mut kbc0 = ((bidx0 as i64).wrapping_mul(total as i64) / gdim) as i32;
        let mut kbc0_stop = ((bidx0 as i64 + 1).wrapping_mul(total as i64) / gdim) as i32;
        kbc0 = (kbc0 as u32).wrapping_sub(fastmodulo(kbc0 as u32, bpn) % bpi) as i32;
        kbc0_stop = (kbc0_stop as u32).wrapping_sub(fastmodulo(kbc0_stop as u32, bpn) % bpi) as i32;

        let did_not_have_any_data = kbc0 == kbc0_stop;
        let wrote_beginning_of_tile = fastmodulo(kbc0 as u32, bpn) == 0;
        let did_not_write_last = fastdiv(kbc0 as u32, bpn) == fastdiv(kbc0_stop as u32, bpn) && fastmodulo(kbc0_stop as u32, bpn) != 0;
        if did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last {
            return;
        }

        let mut any_fixup = false;
        let mut bidx = bidx0 - 1;
        let mut kbc_stop = kbc0;
        loop {
            let mut kbc = ((bidx as i64).wrapping_mul(total as i64) / gdim) as i32;
            kbc = (kbc as u32).wrapping_sub(fastmodulo(kbc as u32, bpn) % bpi) as i32;
            if kbc == kbc_stop {
                bidx -= 1;
                kbc_stop = kbc;
                continue;
            }
            any_fixup = true;
            let base = tmp_last_tile.wrapping_offset(bidx.wrapping_mul(MMQ_X * MMQ_Y) as isize);
            let mut j0 = 0;
            while j0 < MMQ_X {
                unroll!();
                let j = j0 + ty();
                let s = (j0 / FW) as usize;
                sum[s] = add(sum[s], *base.offset((j * MMQ_Y + i) as isize));
                j0 += FW;
            }
            if fastmodulo(kbc as u32, bpn) == 0 || fastdiv(kbc as u32, bpn) < fastdiv(kbc0 as u32, bpn) {
                break;
            }
            bidx -= 1;
            kbc_stop = kbc;
        }
        if !any_fixup {
            return;
        }

        let (it, wt, zt, jt) = tile_of(kbc0, bpn, ntx, ncy, nsy);
        if ids_dst.is_null() {
            let offset_dst = wt.wrapping_mul(stride_sample_dst).wrapping_add(zt.wrapping_mul(stride_channel_dst))
                .wrapping_add(jt.wrapping_mul(MMQ_X).wrapping_mul(stride_col_dst)).wrapping_add(it.wrapping_mul(MMQ_Y));
            let i_max = nrows_x.wrapping_sub(it.wrapping_mul(MMQ_Y)).wrapping_sub(1);
            let j_max = ncols_dst.wrapping_sub(jt.wrapping_mul(MMQ_X)).wrapping_sub(1);
            if NC && i > i_max {
                return;
            }
            let mut j0 = 0;
            while j0 < MMQ_X {
                unroll!();
                let j = j0 + ty();
                if j > j_max {
                    return;
                }
                let idx = offset_dst.wrapping_add(j.wrapping_mul(stride_col_dst)).wrapping_add(i);
                store_dst(dst, idx, add(load_dst(dst, idx, type_dst), sum[(j0 / FW) as usize]), type_dst);
                j0 += FW;
            }
            return;
        }

        let ids = SharedArray::as_raw_mut_ptr(&raw mut IDS);
        let col_low = *expert_bounds.offset(zt as isize);
        let col_high = *expert_bounds.offset(zt as isize + 1);
        let col_diff = col_high.wrapping_sub(col_low);
        let mut j = ty() * WARP + tx();
        while j < MMQ_X {
            *ids.offset(j as isize) = *ids_dst.offset(col_low.wrapping_add(jt.wrapping_mul(MMQ_X)).wrapping_add(j) as isize);
            j += FW * WARP;
        }
        thread::sync_threads();
        let offset_dst = it.wrapping_mul(MMQ_Y);
        let i_max = nrows_x.wrapping_sub(it.wrapping_mul(MMQ_Y)).wrapping_sub(1);
        let j_max = col_diff.wrapping_sub(jt.wrapping_mul(MMQ_X)).wrapping_sub(1);
        if NC && i > i_max {
            return;
        }
        let mut j0 = 0;
        while j0 < MMQ_X {
            unroll!();
            let j = j0 + ty();
            if j > j_max {
                return;
            }
            let idx = offset_dst.wrapping_add((*ids.offset(j as isize)).wrapping_mul(stride_col_dst)).wrapping_add(i);
            store_dst(dst, idx, add(load_dst(dst, idx, type_dst), sum[(j0 / FW) as usize]), type_dst);
            j0 += FW;
        }
    }

    // ---------------------------------------------------------------------------------------
    // mmq_quantize.cu: quantize_mmq_q8_1<input_t, ds_layout> and quantize_mmq_q8_1_glu_f32.
    // LAYOUT 0 = D4, 1 = DS4, 2 = D2S6.

    /// The shared tail of both quantizers: warp amax / sum over the scale group, fast-math
    /// `d_inv = 127 * rcp(amax)`, `q = roundf(x * d_inv)`, `d = rcp(d_inv)`.
    #[inline(always)]
    unsafe fn quantize_tail<const LAYOUT: u32>(v: [f32; 4], vy: *mut u8, ib: i64, iqs: i64) {
        let vals_per_scale: u32 = if LAYOUT == 2 { 64 } else { 32 };
        let vals_per_sum: u32 = if LAYOUT == 2 { 16 } else { 32 };
        let [a, b, c, d] = v;
        let mut amax = fabsf(a);
        amax = fmaxf(amax, fabsf(b));
        amax = fmaxf(amax, fabsf(c));
        amax = fmaxf(amax, fabsf(d));
        let mut off = vals_per_scale / 8;
        while off > 0 {
            amax = fmaxf(amax, warp::shuffle_xor_f32_sync(0xFFFF_FFFF, amax, off));
            off >>= 1;
        }
        let mut sum = 0f32;
        if LAYOUT != 0 {
            sum = add(add(add(a, b), c), d);
            let mut off = vals_per_sum / 8;
            while off > 0 {
                sum = add(sum, warp::shuffle_xor_f32_sync(0xFFFF_FFFF, sum, off));
                off >>= 1;
            }
        }
        let d_inv = div_approx(127.0, amax);
        let q = (roundf_i(mul(a, d_inv)) as u32 & 0xFF)
            | ((roundf_i(mul(b, d_inv)) as u32 & 0xFF) << 8)
            | ((roundf_i(mul(c, d_inv)) as u32 & 0xFF) << 16)
            | ((roundf_i(mul(d, d_inv)) as u32 & 0xFF) << 24);
        let blk = vy.offset((ib * 144) as isize);
        *(blk.offset((16 + iqs) as isize) as *mut u32) = q;
        if LAYOUT == 2 {
            if iqs % 16 != 0 || iqs >= 96 {
                return;
            }
            *(blk.offset((2 * (2 + iqs / 16)) as isize) as *mut u16) = f2h(sum);
            if iqs % 64 != 0 {
                return;
            }
            *(blk.offset((2 * (iqs / 64)) as isize) as *mut u16) = f2h(rcp(d_inv));
            return;
        }
        if iqs % 32 != 0 {
            return;
        }
        let dd = rcp(d_inv);
        if LAYOUT == 1 {
            *(blk.offset((4 * (iqs / 32)) as isize) as *mut u32) = (f2h(dd) as u32) | ((f2h(sum) as u32) << 16);
        } else {
            *(blk.offset((4 * (iqs / 32)) as isize) as *mut f32) = dd;
        }
    }

    /// Input element types of the quantizer: 0 f32 (float4 loads), 1 f16, 2 bf16 (half2 loads).
    #[inline(always)]
    unsafe fn ld_in<const IT: u32>(x: *const u8, i: i64) -> f32 {
        match IT {
            0 => *(x as *const f32).offset(i as isize),
            1 => h2f(*(x as *const u16).offset(i as isize) as u32),
            _ => bf2f(*(x as *const u16).offset(i as isize) as u32),
        }
    }

    /// load_mmq4<input_t>: vector load when all four values are in range, else guarded scalars.
    #[inline(always)]
    unsafe fn load4<const IT: u32>(x: *const u8, base: i64, i0: i64, ne00: i64) -> [f32; 4] {
        if i0 + 3 < ne00 {
            if IT == 0 {
                let p = (x as *const f32).offset((base / 4 * 4) as isize);
                return [*p, *p.add(1), *p.add(2), *p.add(3)];
            }
            let p = (x as *const u16).offset((base / 2 * 2) as isize);
            let f = |k: usize| if IT == 1 { h2f(*p.add(k) as u32) } else { bf2f(*p.add(k) as u32) };
            return [f(0), f(1), f(2), f(3)];
        }
        let mut v = [0f32; 4];
        if i0 < ne00 {
            v[0] = ld_in::<IT>(x, base);
        }
        if i0 + 1 < ne00 {
            v[1] = ld_in::<IT>(x, base + 1);
        }
        if i0 + 2 < ne00 {
            v[2] = ld_in::<IT>(x, base + 2);
        }
        if i0 + 3 < ne00 {
            v[3] = ld_in::<IT>(x, base + 3);
        }
        v
    }

    #[inline(always)]
    pub unsafe fn quantize_mmq_q8_1<const IT: u32, const LAYOUT: u32>(
        x: *const u8, ids: *const i32, vy: *mut u8, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i32, ne2: i32,
    ) {
        let bdx = thread::blockDim_x() as i64;
        let i0 = (bdx * thread::blockIdx_y() as i64 + thread::threadIdx_x() as i64) * 4;
        if i0 >= ne0 {
            return;
        }
        let bz = thread::blockIdx_z() as i64;
        let i1 = thread::blockIdx_x() as i64;
        let i2 = bz % ne2 as i64;
        let i3 = bz / ne2 as i64;
        let i01 = if ids.is_null() { i1 } else { *ids.offset(i1 as isize) as i64 };
        let ib0 = bz * ((thread::gridDim_x() as i64) * (thread::gridDim_y() as i64) * bdx / 32);
        let ib = ib0 + (i0 / 128) * ne1 as i64 + i1;
        let iqs = i0 % 128;
        let base = i3 * s03 + i2 * s02 + i01 * s01 + i0;
        let v = load4::<IT>(x, base, i0, ne00);
        quantize_tail::<LAYOUT>(v, vy, ib, iqs);
    }

    #[inline(always)]
    pub unsafe fn quantize_mmq_q8_1_glu<const LAYOUT: u32>(
        gate: *const f32, up: *const f32, ids: *const i32, vy: *mut u8, ne00: i64, s01: i64, ne0: i64, ne1: i32, activation: i32,
    ) {
        let i0 = ((thread::blockDim_x() as i64) * thread::blockIdx_y() as i64 + thread::threadIdx_x() as i64) * 4;
        if i0 >= ne0 {
            return;
        }
        let i1 = thread::blockIdx_x() as i64;
        let i01 = if ids.is_null() { i1 } else { *ids.offset(i1 as isize) as i64 };
        let ib = (i0 / 128) * ne1 as i64 + i1;
        let iqs = i0 % 128;
        let base = i01 * s01 + i0;
        let mut v = [0f32; 4];
        if i0 < ne00 {
            v[0] = mul(glu_act(*gate.offset(base as isize), activation), *up.offset(base as isize));
        }
        if i0 + 1 < ne00 {
            v[1] = mul(glu_act(*gate.offset(base as isize + 1), activation), *up.offset(base as isize + 1));
        }
        if i0 + 2 < ne00 {
            v[2] = mul(glu_act(*gate.offset(base as isize + 2), activation), *up.offset(base as isize + 2));
        }
        if i0 + 3 < ne00 {
            v[3] = mul(glu_act(*gate.offset(base as isize + 3), activation), *up.offset(base as isize + 3));
        }
        quantize_tail::<LAYOUT>(v, vy, ib, iqs);
    }

    // GENERATED KERNELS BEGIN
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 1, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 2, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 3, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 4, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 5, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 6, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 7, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 8, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_fused_glu_cuda1(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<0, 1, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_fused_glu_cuda2(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<0, 2, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_fused_glu_cuda3(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<0, 3, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_fused_glu_cuda4(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<0, 4, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_fused_glu_cuda5(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<0, 5, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_fused_glu_cuda6(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<0, 6, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_fused_glu_cuda7(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<0, 7, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_fused_glu_cuda8(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<0, 8, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_fused_qkv_cuda1(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<0, 1, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_fused_qkv_cuda2(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<0, 2, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_fused_qkv_cuda3(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<0, 3, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_fused_qkv_cuda4(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<0, 4, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_fused_qkv_cuda5(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<0, 5, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_fused_qkv_cuda6(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<0, 6, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_fused_qkv_cuda7(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<0, 7, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_fused_qkv_cuda8(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<0, 8, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 1, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 2, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 3, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 4, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 5, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 6, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 7, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 8, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_fused_glu_cuda1(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<0, 1, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_fused_glu_cuda2(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<0, 2, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_fused_glu_cuda3(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<0, 3, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_fused_glu_cuda4(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<0, 4, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_fused_glu_cuda5(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<0, 5, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_fused_glu_cuda6(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<0, 6, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_fused_glu_cuda7(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<0, 7, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_fused_glu_cuda8(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<0, 8, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_fused_qkv_cuda1(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<0, 1, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_fused_qkv_cuda2(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<0, 2, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_fused_qkv_cuda3(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<0, 3, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_fused_qkv_cuda4(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<0, 4, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_fused_qkv_cuda5(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<0, 5, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_fused_qkv_cuda6(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<0, 6, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_fused_qkv_cuda7(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<0, 7, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_fused_qkv_cuda8(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<0, 8, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 1, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 2, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 3, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 4, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 5, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 6, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 7, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 8, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_fused_glu_cuda1(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<0, 1, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_fused_glu_cuda2(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<0, 2, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_fused_glu_cuda3(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<0, 3, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_fused_glu_cuda4(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<0, 4, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_fused_glu_cuda5(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<0, 5, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_fused_glu_cuda6(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<0, 6, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_fused_glu_cuda7(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<0, 7, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_fused_glu_cuda8(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<0, 8, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_fused_qkv_cuda1(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<0, 1, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_fused_qkv_cuda2(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<0, 2, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_fused_qkv_cuda3(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<0, 3, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_fused_qkv_cuda4(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<0, 4, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_fused_qkv_cuda5(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<0, 5, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_fused_qkv_cuda6(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<0, 6, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_fused_qkv_cuda7(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<0, 7, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_fused_qkv_cuda8(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<0, 8, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 1, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 2, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 3, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 4, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 5, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 6, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 7, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 8, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_fused_glu_cuda1(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<1, 1, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_fused_glu_cuda2(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<1, 2, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_fused_glu_cuda3(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<1, 3, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_fused_glu_cuda4(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<1, 4, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_fused_glu_cuda5(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<1, 5, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_fused_glu_cuda6(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<1, 6, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_fused_glu_cuda7(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<1, 7, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_fused_glu_cuda8(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<1, 8, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_fused_qkv_cuda1(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<1, 1, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_fused_qkv_cuda2(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<1, 2, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_fused_qkv_cuda3(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<1, 3, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_fused_qkv_cuda4(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<1, 4, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_fused_qkv_cuda5(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<1, 5, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_fused_qkv_cuda6(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<1, 6, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_fused_qkv_cuda7(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<1, 7, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_fused_qkv_cuda8(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<1, 8, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 1, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 2, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 3, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 4, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 5, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 6, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 7, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 8, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_fused_glu_cuda1(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<1, 1, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_fused_glu_cuda2(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<1, 2, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_fused_glu_cuda3(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<1, 3, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_fused_glu_cuda4(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<1, 4, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_fused_glu_cuda5(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<1, 5, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_fused_glu_cuda6(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<1, 6, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_fused_glu_cuda7(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<1, 7, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_fused_glu_cuda8(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<1, 8, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_fused_qkv_cuda1(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<1, 1, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_fused_qkv_cuda2(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<1, 2, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_fused_qkv_cuda3(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<1, 3, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_fused_qkv_cuda4(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<1, 4, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_fused_qkv_cuda5(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<1, 5, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_fused_qkv_cuda6(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<1, 6, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_fused_qkv_cuda7(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<1, 7, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_fused_qkv_cuda8(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<1, 8, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 1, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 2, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 3, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 4, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 5, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 6, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 7, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 8, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_fused_glu_cuda1(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<1, 1, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_fused_glu_cuda2(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<1, 2, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_fused_glu_cuda3(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<1, 3, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_fused_glu_cuda4(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<1, 4, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_fused_glu_cuda5(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<1, 5, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_fused_glu_cuda6(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<1, 6, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_fused_glu_cuda7(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<1, 7, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_fused_glu_cuda8(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<1, 8, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_fused_qkv_cuda1(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<1, 1, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_fused_qkv_cuda2(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<1, 2, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_fused_qkv_cuda3(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<1, 3, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_fused_qkv_cuda4(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<1, 4, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_fused_qkv_cuda5(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<1, 5, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_fused_qkv_cuda6(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<1, 6, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_fused_qkv_cuda7(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<1, 7, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_fused_qkv_cuda8(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<1, 8, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 1, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 2, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 3, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 4, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 5, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 6, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 7, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 8, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_fused_glu_cuda1(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<2, 1, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_fused_glu_cuda2(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<2, 2, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_fused_glu_cuda3(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<2, 3, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_fused_glu_cuda4(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<2, 4, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_fused_glu_cuda5(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<2, 5, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_fused_glu_cuda6(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<2, 6, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_fused_glu_cuda7(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<2, 7, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_fused_glu_cuda8(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<2, 8, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_fused_qkv_cuda1(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<2, 1, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_fused_qkv_cuda2(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<2, 2, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_fused_qkv_cuda3(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<2, 3, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_fused_qkv_cuda4(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<2, 4, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_fused_qkv_cuda5(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<2, 5, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_fused_qkv_cuda6(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<2, 6, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_fused_qkv_cuda7(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<2, 7, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_fused_qkv_cuda8(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<2, 8, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 1, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 2, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 3, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 4, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 5, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 6, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 7, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 8, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_fused_glu_cuda1(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<2, 1, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_fused_glu_cuda2(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<2, 2, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_fused_glu_cuda3(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<2, 3, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_fused_glu_cuda4(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<2, 4, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_fused_glu_cuda5(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<2, 5, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_fused_glu_cuda6(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<2, 6, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_fused_glu_cuda7(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<2, 7, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_fused_glu_cuda8(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<2, 8, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_fused_qkv_cuda1(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<2, 1, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_fused_qkv_cuda2(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<2, 2, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_fused_qkv_cuda3(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<2, 3, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_fused_qkv_cuda4(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<2, 4, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_fused_qkv_cuda5(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<2, 5, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_fused_qkv_cuda6(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<2, 6, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_fused_qkv_cuda7(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<2, 7, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_fused_qkv_cuda8(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<2, 8, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 1, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 2, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 3, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 4, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 5, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 6, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 7, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 8, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_fused_glu_cuda1(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<2, 1, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_fused_glu_cuda2(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<2, 2, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_fused_glu_cuda3(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<2, 3, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_fused_glu_cuda4(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<2, 4, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_fused_glu_cuda5(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<2, 5, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_fused_glu_cuda6(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<2, 6, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_fused_glu_cuda7(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<2, 7, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_fused_glu_cuda8(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<2, 8, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_fused_qkv_cuda1(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<2, 1, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_fused_qkv_cuda2(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<2, 2, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_fused_qkv_cuda3(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<2, 3, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_fused_qkv_cuda4(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<2, 4, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_fused_qkv_cuda5(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<2, 5, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_fused_qkv_cuda6(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<2, 6, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_fused_qkv_cuda7(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<2, 7, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_fused_qkv_cuda8(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<2, 8, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 1, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 2, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 3, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 4, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 5, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 6, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 7, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 8, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_fused_glu_cuda1(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<3, 1, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_fused_glu_cuda2(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<3, 2, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_fused_glu_cuda3(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<3, 3, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_fused_glu_cuda4(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<3, 4, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_fused_glu_cuda5(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<3, 5, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_fused_glu_cuda6(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<3, 6, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_fused_glu_cuda7(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<3, 7, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_fused_glu_cuda8(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<3, 8, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_fused_qkv_cuda1(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<3, 1, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_fused_qkv_cuda2(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<3, 2, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_fused_qkv_cuda3(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<3, 3, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_fused_qkv_cuda4(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<3, 4, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_fused_qkv_cuda5(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<3, 5, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_fused_qkv_cuda6(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<3, 6, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_fused_qkv_cuda7(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<3, 7, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_fused_qkv_cuda8(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<3, 8, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 1, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 2, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 3, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 4, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 5, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 6, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 7, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 8, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_fused_glu_cuda1(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<3, 1, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_fused_glu_cuda2(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<3, 2, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_fused_glu_cuda3(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<3, 3, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_fused_glu_cuda4(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<3, 4, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_fused_glu_cuda5(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<3, 5, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_fused_glu_cuda6(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<3, 6, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_fused_glu_cuda7(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<3, 7, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_fused_glu_cuda8(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<3, 8, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_fused_qkv_cuda1(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<3, 1, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_fused_qkv_cuda2(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<3, 2, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_fused_qkv_cuda3(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<3, 3, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_fused_qkv_cuda4(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<3, 4, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_fused_qkv_cuda5(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<3, 5, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_fused_qkv_cuda6(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<3, 6, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_fused_qkv_cuda7(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<3, 7, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_fused_qkv_cuda8(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<3, 8, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 1, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 2, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 3, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 4, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 5, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 6, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 7, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 8, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_fused_glu_cuda1(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<3, 1, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_fused_glu_cuda2(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<3, 2, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_fused_glu_cuda3(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<3, 3, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_fused_glu_cuda4(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<3, 4, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_fused_glu_cuda5(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<3, 5, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_fused_glu_cuda6(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<3, 6, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_fused_glu_cuda7(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<3, 7, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_fused_glu_cuda8(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<3, 8, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_fused_qkv_cuda1(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<3, 1, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_fused_qkv_cuda2(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<3, 2, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_fused_qkv_cuda3(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<3, 3, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_fused_qkv_cuda4(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<3, 4, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_fused_qkv_cuda5(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<3, 5, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_fused_qkv_cuda6(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<3, 6, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_fused_qkv_cuda7(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<3, 7, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_fused_qkv_cuda8(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<3, 8, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 1, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 2, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 3, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 4, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 5, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 6, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 7, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 8, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_fused_glu_cuda1(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<4, 1, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_fused_glu_cuda2(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<4, 2, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_fused_glu_cuda3(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<4, 3, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_fused_glu_cuda4(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<4, 4, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_fused_glu_cuda5(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<4, 5, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_fused_glu_cuda6(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<4, 6, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_fused_glu_cuda7(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<4, 7, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_fused_glu_cuda8(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<4, 8, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_fused_qkv_cuda1(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<4, 1, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_fused_qkv_cuda2(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<4, 2, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_fused_qkv_cuda3(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<4, 3, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_fused_qkv_cuda4(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<4, 4, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_fused_qkv_cuda5(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<4, 5, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_fused_qkv_cuda6(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<4, 6, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_fused_qkv_cuda7(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<4, 7, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_fused_qkv_cuda8(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<4, 8, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 1, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 2, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 3, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 4, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 5, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 6, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 7, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 8, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_fused_glu_cuda1(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<4, 1, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_fused_glu_cuda2(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<4, 2, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_fused_glu_cuda3(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<4, 3, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_fused_glu_cuda4(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<4, 4, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_fused_glu_cuda5(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<4, 5, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_fused_glu_cuda6(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<4, 6, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_fused_glu_cuda7(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<4, 7, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_fused_glu_cuda8(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<4, 8, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_fused_qkv_cuda1(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<4, 1, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_fused_qkv_cuda2(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<4, 2, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_fused_qkv_cuda3(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<4, 3, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_fused_qkv_cuda4(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<4, 4, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_fused_qkv_cuda5(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<4, 5, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_fused_qkv_cuda6(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<4, 6, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_fused_qkv_cuda7(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<4, 7, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_fused_qkv_cuda8(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<4, 8, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 1, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 2, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 3, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 4, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 5, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 6, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 7, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 8, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_fused_glu_cuda1(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<4, 1, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_fused_glu_cuda2(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<4, 2, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_fused_glu_cuda3(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<4, 3, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_fused_glu_cuda4(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<4, 4, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_fused_glu_cuda5(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<4, 5, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_fused_glu_cuda6(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<4, 6, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_fused_glu_cuda7(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<4, 7, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_fused_glu_cuda8(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<4, 8, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_fused_qkv_cuda1(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<4, 1, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_fused_qkv_cuda2(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<4, 2, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_fused_qkv_cuda3(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<4, 3, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_fused_qkv_cuda4(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<4, 4, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_fused_qkv_cuda5(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<4, 5, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_fused_qkv_cuda6(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<4, 6, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_fused_qkv_cuda7(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<4, 7, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_fused_qkv_cuda8(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<4, 8, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 1, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 2, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 3, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 4, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 5, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 6, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 7, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 8, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_fused_glu_cuda1(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<5, 1, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_fused_glu_cuda2(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<5, 2, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_fused_glu_cuda3(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<5, 3, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_fused_glu_cuda4(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<5, 4, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_fused_glu_cuda5(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<5, 5, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_fused_glu_cuda6(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<5, 6, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_fused_glu_cuda7(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<5, 7, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_fused_glu_cuda8(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<5, 8, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_fused_qkv_cuda1(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<5, 1, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_fused_qkv_cuda2(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<5, 2, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_fused_qkv_cuda3(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<5, 3, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_fused_qkv_cuda4(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<5, 4, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_fused_qkv_cuda5(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<5, 5, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_fused_qkv_cuda6(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<5, 6, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_fused_qkv_cuda7(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<5, 7, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_fused_qkv_cuda8(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<5, 8, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 1, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 2, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 3, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 4, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 5, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 6, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 7, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 8, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_fused_glu_cuda1(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<5, 1, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_fused_glu_cuda2(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<5, 2, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_fused_glu_cuda3(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<5, 3, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_fused_glu_cuda4(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<5, 4, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_fused_glu_cuda5(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<5, 5, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_fused_glu_cuda6(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<5, 6, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_fused_glu_cuda7(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<5, 7, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_fused_glu_cuda8(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<5, 8, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_fused_qkv_cuda1(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<5, 1, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_fused_qkv_cuda2(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<5, 2, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_fused_qkv_cuda3(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<5, 3, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_fused_qkv_cuda4(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<5, 4, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_fused_qkv_cuda5(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<5, 5, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_fused_qkv_cuda6(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<5, 6, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_fused_qkv_cuda7(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<5, 7, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_fused_qkv_cuda8(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<5, 8, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 1, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 2, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 3, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 4, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 5, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 6, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 7, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 8, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_fused_glu_cuda1(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<5, 1, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_fused_glu_cuda2(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<5, 2, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_fused_glu_cuda3(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<5, 3, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_fused_glu_cuda4(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<5, 4, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_fused_glu_cuda5(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<5, 5, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_fused_glu_cuda6(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<5, 6, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_fused_glu_cuda7(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<5, 7, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_fused_glu_cuda8(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<5, 8, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_fused_qkv_cuda1(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<5, 1, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_fused_qkv_cuda2(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<5, 2, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_fused_qkv_cuda3(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<5, 3, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_fused_qkv_cuda4(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<5, 4, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_fused_qkv_cuda5(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<5, 5, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_fused_qkv_cuda6(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<5, 6, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_fused_qkv_cuda7(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<5, 7, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_fused_qkv_cuda8(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<5, 8, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 1, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 2, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 3, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 4, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 5, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 6, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 7, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 8, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_fused_glu_cuda1(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<6, 1, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_fused_glu_cuda2(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<6, 2, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_fused_glu_cuda3(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<6, 3, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_fused_glu_cuda4(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<6, 4, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_fused_glu_cuda5(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<6, 5, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_fused_glu_cuda6(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<6, 6, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_fused_glu_cuda7(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<6, 7, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_fused_glu_cuda8(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<6, 8, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_fused_qkv_cuda1(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<6, 1, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_fused_qkv_cuda2(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<6, 2, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_fused_qkv_cuda3(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<6, 3, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_fused_qkv_cuda4(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<6, 4, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_fused_qkv_cuda5(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<6, 5, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_fused_qkv_cuda6(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<6, 6, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_fused_qkv_cuda7(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<6, 7, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_fused_qkv_cuda8(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<6, 8, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 1, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 2, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 3, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 4, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 5, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 6, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 7, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 8, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_fused_glu_cuda1(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<6, 1, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_fused_glu_cuda2(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<6, 2, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_fused_glu_cuda3(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<6, 3, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_fused_glu_cuda4(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<6, 4, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_fused_glu_cuda5(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<6, 5, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_fused_glu_cuda6(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<6, 6, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_fused_glu_cuda7(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<6, 7, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_fused_glu_cuda8(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<6, 8, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_fused_qkv_cuda1(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<6, 1, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_fused_qkv_cuda2(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<6, 2, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_fused_qkv_cuda3(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<6, 3, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_fused_qkv_cuda4(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<6, 4, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_fused_qkv_cuda5(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<6, 5, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_fused_qkv_cuda6(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<6, 6, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_fused_qkv_cuda7(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<6, 7, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_fused_qkv_cuda8(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<6, 8, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 1, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 2, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 3, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 4, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 5, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 6, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 7, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 8, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_fused_glu_cuda1(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<6, 1, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_fused_glu_cuda2(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<6, 2, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_fused_glu_cuda3(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<6, 3, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_fused_glu_cuda4(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<6, 4, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_fused_glu_cuda5(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<6, 5, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_fused_glu_cuda6(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<6, 6, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_fused_glu_cuda7(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<6, 7, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_fused_glu_cuda8(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<6, 8, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_fused_qkv_cuda1(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<6, 1, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_fused_qkv_cuda2(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<6, 2, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_fused_qkv_cuda3(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<6, 3, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_fused_qkv_cuda4(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<6, 4, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_fused_qkv_cuda5(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<6, 5, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_fused_qkv_cuda6(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<6, 6, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_fused_qkv_cuda7(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<6, 7, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_fused_qkv_cuda8(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<6, 8, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 1, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 2, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 3, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 4, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 5, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 6, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 7, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 8, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_fused_glu_cuda1(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<7, 1, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_fused_glu_cuda2(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<7, 2, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_fused_glu_cuda3(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<7, 3, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_fused_glu_cuda4(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<7, 4, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_fused_glu_cuda5(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<7, 5, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_fused_glu_cuda6(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<7, 6, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_fused_glu_cuda7(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<7, 7, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_fused_glu_cuda8(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<7, 8, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_fused_qkv_cuda1(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<7, 1, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_fused_qkv_cuda2(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<7, 2, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_fused_qkv_cuda3(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<7, 3, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_fused_qkv_cuda4(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<7, 4, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_fused_qkv_cuda5(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<7, 5, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_fused_qkv_cuda6(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<7, 6, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_fused_qkv_cuda7(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<7, 7, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_fused_qkv_cuda8(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<7, 8, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 1, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 2, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 3, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 4, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 5, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 6, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 7, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 8, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_fused_glu_cuda1(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<7, 1, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_fused_glu_cuda2(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<7, 2, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_fused_glu_cuda3(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<7, 3, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_fused_glu_cuda4(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<7, 4, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_fused_glu_cuda5(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<7, 5, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_fused_glu_cuda6(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<7, 6, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_fused_glu_cuda7(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<7, 7, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_fused_glu_cuda8(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<7, 8, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_fused_qkv_cuda1(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<7, 1, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_fused_qkv_cuda2(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<7, 2, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_fused_qkv_cuda3(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<7, 3, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_fused_qkv_cuda4(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<7, 4, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_fused_qkv_cuda5(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<7, 5, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_fused_qkv_cuda6(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<7, 6, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_fused_qkv_cuda7(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<7, 7, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_fused_qkv_cuda8(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<7, 8, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 1, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 2, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 3, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 4, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 5, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 6, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 7, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 8, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_fused_glu_cuda1(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<7, 1, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_fused_glu_cuda2(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<7, 2, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_fused_glu_cuda3(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<7, 3, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_fused_glu_cuda4(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<7, 4, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_fused_glu_cuda5(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<7, 5, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_fused_glu_cuda6(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<7, 6, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_fused_glu_cuda7(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<7, 7, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_fused_glu_cuda8(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<7, 8, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_fused_qkv_cuda1(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<7, 1, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_fused_qkv_cuda2(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<7, 2, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_fused_qkv_cuda3(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<7, 3, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_fused_qkv_cuda4(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<7, 4, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_fused_qkv_cuda5(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<7, 5, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_fused_qkv_cuda6(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<7, 6, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_fused_qkv_cuda7(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<7, 7, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_fused_qkv_cuda8(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<7, 8, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 1, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 2, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 3, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 4, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 5, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 6, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 7, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 8, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_fused_glu_cuda1(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<8, 1, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_fused_glu_cuda2(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<8, 2, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_fused_glu_cuda3(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<8, 3, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_fused_glu_cuda4(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<8, 4, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_fused_glu_cuda5(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<8, 5, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_fused_glu_cuda6(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<8, 6, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_fused_glu_cuda7(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<8, 7, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_fused_glu_cuda8(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<8, 8, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_fused_qkv_cuda1(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<8, 1, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_fused_qkv_cuda2(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<8, 2, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_fused_qkv_cuda3(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<8, 3, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_fused_qkv_cuda4(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<8, 4, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_fused_qkv_cuda5(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<8, 5, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_fused_qkv_cuda6(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<8, 6, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_fused_qkv_cuda7(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<8, 7, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_fused_qkv_cuda8(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<8, 8, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 1, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 2, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 3, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 4, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 5, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 6, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 7, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 8, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_fused_glu_cuda1(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<8, 1, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_fused_glu_cuda2(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<8, 2, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_fused_glu_cuda3(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<8, 3, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_fused_glu_cuda4(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<8, 4, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_fused_glu_cuda5(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<8, 5, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_fused_glu_cuda6(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<8, 6, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_fused_glu_cuda7(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<8, 7, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_fused_glu_cuda8(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<8, 8, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_fused_qkv_cuda1(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<8, 1, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_fused_qkv_cuda2(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<8, 2, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_fused_qkv_cuda3(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<8, 3, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_fused_qkv_cuda4(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<8, 4, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_fused_qkv_cuda5(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<8, 5, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_fused_qkv_cuda6(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<8, 6, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_fused_qkv_cuda7(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<8, 7, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_fused_qkv_cuda8(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<8, 8, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 1, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 2, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 3, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 4, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 5, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 6, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 7, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 8, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_fused_glu_cuda1(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<8, 1, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_fused_glu_cuda2(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<8, 2, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_fused_glu_cuda3(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<8, 3, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_fused_glu_cuda4(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<8, 4, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_fused_glu_cuda5(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<8, 5, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_fused_glu_cuda6(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<8, 6, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_fused_glu_cuda7(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<8, 7, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_fused_glu_cuda8(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<8, 8, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_fused_qkv_cuda1(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<8, 1, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_fused_qkv_cuda2(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<8, 2, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_fused_qkv_cuda3(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<8, 3, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_fused_qkv_cuda4(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<8, 4, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_fused_qkv_cuda5(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<8, 5, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_fused_qkv_cuda6(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<8, 6, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_fused_qkv_cuda7(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<8, 7, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_fused_qkv_cuda8(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<8, 8, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 1, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 2, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 3, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 4, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 5, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 6, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 7, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 8, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_fused_glu_cuda1(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<9, 1, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_fused_glu_cuda2(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<9, 2, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_fused_glu_cuda3(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<9, 3, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_fused_glu_cuda4(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<9, 4, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_fused_glu_cuda5(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<9, 5, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_fused_glu_cuda6(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<9, 6, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_fused_glu_cuda7(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<9, 7, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_fused_glu_cuda8(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<9, 8, B>(vx_gate, vx_up, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_fused_qkv_cuda1(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<9, 1, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_fused_qkv_cuda2(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<9, 2, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_fused_qkv_cuda3(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<9, 3, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_fused_qkv_cuda4(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<9, 4, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_fused_qkv_cuda5(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<9, 5, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_fused_qkv_cuda6(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<9, 6, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_fused_qkv_cuda7(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<9, 7, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_fused_qkv_cuda8(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<9, 8, B>(vx_q, vx_k, vx_v, vy, q_dst as *mut B, k_dst as *mut B, v_dst as *mut B, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 1, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 2, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 3, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 4, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 5, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 6, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 7, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 8, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_fused_glu_cuda1(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<9, 1, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_fused_glu_cuda2(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<9, 2, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_fused_glu_cuda3(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<9, 3, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_fused_glu_cuda4(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<9, 4, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_fused_glu_cuda5(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<9, 5, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_fused_glu_cuda6(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<9, 6, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_fused_glu_cuda7(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<9, 7, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_fused_glu_cuda8(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<9, 8, H>(vx_gate, vx_up, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_fused_qkv_cuda1(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<9, 1, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_fused_qkv_cuda2(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<9, 2, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_fused_qkv_cuda3(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<9, 3, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_fused_qkv_cuda4(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<9, 4, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_fused_qkv_cuda5(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<9, 5, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_fused_qkv_cuda6(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<9, 6, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_fused_qkv_cuda7(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<9, 7, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_fused_qkv_cuda8(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut u16, k_dst: *mut u16, v_dst: *mut u16, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<9, 8, H>(vx_q, vx_k, vx_v, vy, q_dst as *mut H, k_dst as *mut H, v_dst as *mut H, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 1, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 2, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 3, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 4, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 5, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 6, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 7, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 8, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_fused_glu_cuda1(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<9, 1, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_fused_glu_cuda2(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<9, 2, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_fused_glu_cuda3(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<9, 3, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_fused_glu_cuda4(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<9, 4, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_fused_glu_cuda5(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<9, 5, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_fused_glu_cuda6(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<9, 6, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_fused_glu_cuda7(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<9, 7, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_fused_glu_cuda8(vx_gate: *const u8, vx_up: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, activation: i32) { mmvq_glu::<9, 8, f32>(vx_gate, vx_up, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, activation) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_fused_qkv_cuda1(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<9, 1, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_fused_qkv_cuda2(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<9, 2, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_fused_qkv_cuda3(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<9, 3, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_fused_qkv_cuda4(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<9, 4, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_fused_qkv_cuda5(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<9, 5, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_fused_qkv_cuda6(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<9, 6, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_fused_qkv_cuda7(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<9, 7, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_fused_qkv_cuda8(vx_q: *const u8, vx_k: *const u8, vx_v: *const u8, vy: *const u8, q_dst: *mut f32, k_dst: *mut f32, v_dst: *mut f32, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32, stride_col_y: i32) { mmvq_qkv::<9, 8, f32>(vx_q, vx_k, vx_v, vy, q_dst, k_dst, v_dst, ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y) }
    #[kernel] pub unsafe fn quantize_mmq_q8_1_f32_d4(x: *const u8, ids: *const i32, vy: *mut u8, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i32, ne2: i32) { quantize_mmq_q8_1::<0, 0>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2) }
    #[kernel] pub unsafe fn quantize_mmq_q8_1_f16_d4(x: *const u8, ids: *const i32, vy: *mut u8, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i32, ne2: i32) { quantize_mmq_q8_1::<1, 0>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2) }
    #[kernel] pub unsafe fn quantize_mmq_q8_1_bf16_d4(x: *const u8, ids: *const i32, vy: *mut u8, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i32, ne2: i32) { quantize_mmq_q8_1::<2, 0>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2) }
    #[kernel] pub unsafe fn quantize_mmq_q8_1_glu_f32_d4(gate: *const f32, up: *const f32, ids: *const i32, vy: *mut u8, ne00: i64, s01: i64, ne0: i64, ne1: i32, activation: i32) { quantize_mmq_q8_1_glu::<0>(gate, up, ids, vy, ne00, s01, ne0, ne1, activation) }
    #[kernel] pub unsafe fn quantize_mmq_q8_1_f32_ds4(x: *const u8, ids: *const i32, vy: *mut u8, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i32, ne2: i32) { quantize_mmq_q8_1::<0, 1>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2) }
    #[kernel] pub unsafe fn quantize_mmq_q8_1_f16_ds4(x: *const u8, ids: *const i32, vy: *mut u8, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i32, ne2: i32) { quantize_mmq_q8_1::<1, 1>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2) }
    #[kernel] pub unsafe fn quantize_mmq_q8_1_bf16_ds4(x: *const u8, ids: *const i32, vy: *mut u8, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i32, ne2: i32) { quantize_mmq_q8_1::<2, 1>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2) }
    #[kernel] pub unsafe fn quantize_mmq_q8_1_glu_f32_ds4(gate: *const f32, up: *const f32, ids: *const i32, vy: *mut u8, ne00: i64, s01: i64, ne0: i64, ne1: i32, activation: i32) { quantize_mmq_q8_1_glu::<1>(gate, up, ids, vy, ne00, s01, ne0, ne1, activation) }
    #[kernel] pub unsafe fn quantize_mmq_q8_1_f32_d2s6(x: *const u8, ids: *const i32, vy: *mut u8, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i32, ne2: i32) { quantize_mmq_q8_1::<0, 2>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2) }
    #[kernel] pub unsafe fn quantize_mmq_q8_1_f16_d2s6(x: *const u8, ids: *const i32, vy: *mut u8, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i32, ne2: i32) { quantize_mmq_q8_1::<1, 2>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2) }
    #[kernel] pub unsafe fn quantize_mmq_q8_1_bf16_d2s6(x: *const u8, ids: *const i32, vy: *mut u8, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i32, ne2: i32) { quantize_mmq_q8_1::<2, 2>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2) }
    #[kernel] pub unsafe fn quantize_mmq_q8_1_glu_f32_d2s6(gate: *const f32, up: *const f32, ids: *const i32, vy: *mut u8, ne00: i64, s01: i64, ne0: i64, ne1: i32, activation: i32) { quantize_mmq_q8_1_glu::<2>(gate, up, ids, vy, ne00, s01, ne0, ne1, activation) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x8_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_0, 8, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x8_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_0, 8, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x16_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_0, 16, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x16_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_0, 16, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x24_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_0, 24, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x24_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_0, 24, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x32_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_0, 32, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x32_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_0, 32, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x40_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_0, 40, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x40_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_0, 40, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x48_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_0, 48, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x48_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_0, 48, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x64_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_0, 64, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x64_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_0, 64, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x80_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_0, 80, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x80_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_0, 80, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x96_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_0, 96, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x96_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_0, 96, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x112_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_0, 112, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x112_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_0, 112, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x128_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_0, 128, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x128_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_0, 128, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x8_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_1, 8, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x8_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_1, 8, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x16_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_1, 16, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x16_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_1, 16, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x24_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_1, 24, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x24_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_1, 24, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x32_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_1, 32, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x32_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_1, 32, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x40_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_1, 40, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x40_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_1, 40, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x48_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_1, 48, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x48_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_1, 48, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x64_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_1, 64, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x64_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_1, 64, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x80_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_1, 80, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x80_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_1, 80, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x96_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_1, 96, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x96_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_1, 96, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x112_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_1, 112, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x112_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_1, 112, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x128_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_1, 128, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x128_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_1, 128, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x8_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_0, 8, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x8_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_0, 8, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x16_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_0, 16, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x16_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_0, 16, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x24_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_0, 24, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x24_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_0, 24, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x32_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_0, 32, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x32_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_0, 32, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x40_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_0, 40, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x40_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_0, 40, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x48_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_0, 48, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x48_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_0, 48, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x64_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_0, 64, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x64_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_0, 64, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x80_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_0, 80, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x80_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_0, 80, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x96_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_0, 96, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x96_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_0, 96, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x112_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_0, 112, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x112_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_0, 112, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x128_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_0, 128, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x128_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_0, 128, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x8_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_1, 8, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x8_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_1, 8, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x16_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_1, 16, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x16_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_1, 16, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x24_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_1, 24, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x24_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_1, 24, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x32_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_1, 32, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x32_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_1, 32, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x40_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_1, 40, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x40_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_1, 40, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x48_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_1, 48, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x48_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_1, 48, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x64_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_1, 64, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x64_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_1, 64, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x80_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_1, 80, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x80_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_1, 80, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x96_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_1, 96, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x96_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_1, 96, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x112_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_1, 112, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x112_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_1, 112, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x128_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_1, 128, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x128_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_1, 128, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x8_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q8_0, 8, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x8_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q8_0, 8, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x16_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q8_0, 16, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x16_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q8_0, 16, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x24_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q8_0, 24, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x24_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q8_0, 24, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x32_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q8_0, 32, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x32_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q8_0, 32, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x40_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q8_0, 40, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x40_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q8_0, 40, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x48_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q8_0, 48, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x48_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q8_0, 48, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x64_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q8_0, 64, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x64_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q8_0, 64, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x80_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q8_0, 80, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x80_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q8_0, 80, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x96_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q8_0, 96, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x96_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q8_0, 96, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x112_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q8_0, 112, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x112_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q8_0, 112, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x128_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q8_0, 128, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x128_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q8_0, 128, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x8_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q2_K, 8, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x8_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q2_K, 8, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x16_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q2_K, 16, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x16_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q2_K, 16, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x24_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q2_K, 24, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x24_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q2_K, 24, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x32_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q2_K, 32, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x32_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q2_K, 32, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x40_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q2_K, 40, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x40_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q2_K, 40, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x48_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q2_K, 48, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x48_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q2_K, 48, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x64_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q2_K, 64, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x64_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q2_K, 64, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x80_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q2_K, 80, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x80_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q2_K, 80, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x96_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q2_K, 96, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x96_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q2_K, 96, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x112_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q2_K, 112, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x112_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q2_K, 112, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x128_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q2_K, 128, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x128_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q2_K, 128, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x8_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q3_K, 8, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x8_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q3_K, 8, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x16_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q3_K, 16, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x16_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q3_K, 16, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x24_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q3_K, 24, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x24_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q3_K, 24, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x32_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q3_K, 32, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x32_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q3_K, 32, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x40_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q3_K, 40, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x40_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q3_K, 40, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x48_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q3_K, 48, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x48_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q3_K, 48, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x64_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q3_K, 64, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x64_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q3_K, 64, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x80_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q3_K, 80, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x80_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q3_K, 80, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x96_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q3_K, 96, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x96_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q3_K, 96, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x112_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q3_K, 112, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x112_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q3_K, 112, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x128_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q3_K, 128, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x128_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q3_K, 128, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x8_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_K, 8, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x8_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_K, 8, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x16_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_K, 16, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x16_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_K, 16, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x24_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_K, 24, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x24_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_K, 24, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x32_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_K, 32, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x32_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_K, 32, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x40_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_K, 40, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x40_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_K, 40, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x48_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_K, 48, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x48_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_K, 48, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x64_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_K, 64, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x64_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_K, 64, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x80_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_K, 80, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x80_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_K, 80, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x96_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_K, 96, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x96_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_K, 96, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x112_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_K, 112, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x112_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_K, 112, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x128_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_K, 128, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x128_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q4_K, 128, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x8_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_K, 8, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x8_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_K, 8, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x16_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_K, 16, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x16_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_K, 16, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x24_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_K, 24, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x24_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_K, 24, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x32_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_K, 32, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x32_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_K, 32, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x40_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_K, 40, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x40_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_K, 40, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x48_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_K, 48, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x48_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_K, 48, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x64_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_K, 64, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x64_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_K, 64, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x80_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_K, 80, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x80_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_K, 80, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x96_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_K, 96, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x96_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_K, 96, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x112_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_K, 112, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x112_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_K, 112, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x128_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_K, 128, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x128_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q5_K, 128, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x8_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q6_K, 8, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x8_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q6_K, 8, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x16_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q6_K, 16, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x16_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q6_K, 16, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x24_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q6_K, 24, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x24_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q6_K, 24, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x32_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q6_K, 32, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x32_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q6_K, 32, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x40_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q6_K, 40, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x40_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q6_K, 40, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x48_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q6_K, 48, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x48_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q6_K, 48, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x64_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q6_K, 64, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x64_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q6_K, 64, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x80_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q6_K, 80, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x80_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q6_K, 80, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x96_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q6_K, 96, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x96_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q6_K, 96, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x112_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q6_K, 112, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x112_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q6_K, 112, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x128_nc0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q6_K, 128, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x128_nc1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, tmp_fixup: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, type_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Q6_K, 128, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, type_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk32_x8_nc0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<32, 8, false>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk32_x8_nc1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<32, 8, true>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk32_x16_nc0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<32, 16, false>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk32_x16_nc1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<32, 16, true>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk32_x24_nc0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<32, 24, false>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk32_x24_nc1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<32, 24, true>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk32_x32_nc0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<32, 32, false>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk32_x32_nc1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<32, 32, true>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk32_x40_nc0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<32, 40, false>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk32_x40_nc1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<32, 40, true>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk32_x48_nc0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<32, 48, false>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk32_x48_nc1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<32, 48, true>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk32_x64_nc0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<32, 64, false>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk32_x64_nc1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<32, 64, true>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk32_x80_nc0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<32, 80, false>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk32_x80_nc1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<32, 80, true>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk32_x96_nc0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<32, 96, false>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk32_x96_nc1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<32, 96, true>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk32_x112_nc0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<32, 112, false>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk32_x112_nc1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<32, 112, true>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk32_x128_nc0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<32, 128, false>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk32_x128_nc1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<32, 128, true>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk256_x8_nc0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<256, 8, false>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk256_x8_nc1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<256, 8, true>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk256_x16_nc0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<256, 16, false>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk256_x16_nc1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<256, 16, true>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk256_x24_nc0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<256, 24, false>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk256_x24_nc1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<256, 24, true>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk256_x32_nc0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<256, 32, false>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk256_x32_nc1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<256, 32, true>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk256_x40_nc0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<256, 40, false>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk256_x40_nc1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<256, 40, true>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk256_x48_nc0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<256, 48, false>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk256_x48_nc1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<256, 48, true>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk256_x64_nc0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<256, 64, false>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk256_x64_nc1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<256, 64, true>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk256_x80_nc0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<256, 80, false>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk256_x80_nc1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<256, 80, true>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk256_x96_nc0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<256, 96, false>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk256_x96_nc1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<256, 96, true>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk256_x112_nc0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<256, 112, false>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk256_x112_nc1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<256, 112, true>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk256_x128_nc0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<256, 128, false>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] pub unsafe fn mmq_fixup_qk256_x128_nc1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut u8, type_dst: i32, tmp_last_tile: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<256, 128, true>(ids_dst, expert_bounds, dst, type_dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    // GENERATED KERNELS END
}

fn main() {
    std::process::exit(gate::run());
}
