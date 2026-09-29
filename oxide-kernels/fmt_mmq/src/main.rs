//! llama.cpp MMQ (quantized-activation tile matmul, the prompt / prefill path) for GGML IQ4_NL
//! (type 20), MXFP4 (type 39) and NVFP4 (type 40) in cuda-oxide, ported from llama.cpp acecd56
//! under the reference's own parameter ABI and bit-identical to its nvcc build (nvcc 13.3, -O3
//! -use_fast_math, sm_120a; `ref/nvcc_*.sh` rebuild mmq-instance-<fmt>.cu / quantize.cu, and their
//! SASS is identical to the kernels in ~/ai/llama.cpp/build/bin/libggml-cuda.so, ref/sass_cmp.py).
//! The instances are the ones llama.cpp runs on this card (sm_120a, BLACKWELL_MMA_AVAILABLE):
//!
//! - IQ4_NL: mmq-config-ampere.cuh (256 threads, I = 128, SRAM layout Q8_0 = stride 76, K_vram 256,
//!   stream-k, J in {8,16,24,32,40,48,64,80,96,112,128} / fallback {8,16,32,64,128}), Turing+ MMA
//!   data layout: `ggml_cuda_mmq_load_tiles_iq4_nl` + `ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<D4>`
//!   (m16n8k32 s8 MMA, `sum = fma.ftz(mul.ftz(dA, (float) C), dB, sum)`) on activations from
//!   `quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4>` (block_q8_1_mmq, 128 values / 144 bytes).
//! - MXFP4 / NVFP4: mmq-config-blackwell.cuh (same thread / tile / J sets, SRAM layout FP4 = stride
//!   76, K_vram 512), native FP4: `ggml_cuda_mmq_load_tiles_{mxfp4_fp4,nvfp4_nvfp4}` +
//!   `ggml_cuda_mmq_vec_dot_fp4_fp4_mma` (`mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X
//!   ...ue8m0` resp. `kind::mxf4nvf4.block_scale.scale_vec::4X ...ue4m3`, m16n8k64, zero C, then
//!   `sum = add.ftz(C, sum)`), on activations quantized to FP4 by `quantize_mmq_mxfp4<false>`
//!   (E8M0 scale per 32 values) resp. `quantize_mmq_nvfp4<false, aligned>` (per-row f32 scale
//!   amax / 2688 into `y_scale`, UE4M3 scale per 16 values chosen from 5 candidates by squared
//!   error), block_fp4_mmq = 256 values / 144 bytes. NVFP4's write-back multiplies by
//!   `y_scale[j]`.
//! - every type: `mul_mat_q_stream_k_fixup<type, J, fallback>`.
//!
//! The quantizers take f32 input (the reference instances) or f16 / bf16 input widened exactly
//! first. `export_ptx.py` splits fmt_mmq.ptx into the per-format modules mistral.rs embeds
//! (mistralrs-quant/src/gguf/<fmt>_mmq_oxide.ptx).
#![allow(non_snake_case, clippy::missing_safety_doc, clippy::too_many_arguments)]
mod gate;

use cuda_device::{DynamicSharedArray, SharedArray, convert, kernel, launch_bounds, ptx_asm, thread, warp};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    macro_rules! unroll {
        () => {
            cuda_device::thread::__unroll_config::<0>();
        };
    }

    // ------------------------------------------------------------------------------------------
    // Scalar helpers (llama.cpp is -use_fast_math: every f32 op is .ftz).

    #[inline(always)]
    pub fn h2f(bits: u32) -> f32 {
        convert::cvt_f32_f16x2_lo(bits & 0xFFFF)
    }
    /// fast-math `a * b` [FMUL.FTZ].
    #[inline(always)]
    pub fn mul(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("mul.rn.ftz.f32 %0, %1, %2; //\t\x0b\x0c\r\n", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    /// `mul.ftz.f32` without `.rn`, as the reference PTX writes it (only where nothing can fuse).
    #[inline(always)]
    pub fn mulf(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("mul.ftz.f32 %0, %1, %2; //\t\x0b\x0c\r\n", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    /// fast-math `a + b` [FADD.FTZ].
    #[inline(always)]
    pub fn add(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("add.rn.ftz.f32 %0, %1, %2; //\t\x0b\x0c\r\n", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    /// fast-math contraction [FFMA.FTZ].
    #[inline(always)]
    pub fn fma(a: f32, b: f32, c: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("fma.rn.ftz.f32 %0, %1, %2, %3; //\t\x0b\x0c\r\n", out("=f") r, in("f") a, in("f") b, in("f") c, options(register_only)); }
        r
    }
    /// `|c| - |a| * b` as the SASS has it: FFMA.FTZ r, -b, |a|, |c| (one rounding).
    #[inline(always)]
    pub fn fnms_abs(a: f32, b: f32, c: f32) -> f32 {
        let r: f32;
        unsafe {
            ptx_asm!(
                "{ .reg .f32 x, y, z; abs.ftz.f32 x, %1; neg.f32 y, %2; abs.ftz.f32 z, %3; fma.rn.ftz.f32 %0, y, x, z; } //\t\x0b\x0c\r\n",
                out("=f") r, in("f") a, in("f") b, in("f") c, options(register_only)
            );
        }
        r
    }
    /// `__byte_perm(a, b, s)`: prmt with the selector masked to its 3-bit byte indices.
    #[inline(always)]
    pub fn byte_perm(a: u32, b: u32, s: u32) -> u32 {
        let r: u32;
        let s = s & 0x7777;
        unsafe { ptx_asm!("prmt.b32 %0, %1, %2, %3;", out("=r") r, in("r") a, in("r") b, in("r") s, options(register_only)); }
        r
    }
    /// llama.cpp fastdiv: `(__umulhi(n, mp) + n) >> L`.
    #[inline(always)]
    pub fn fastdiv(n: u32, mp: u32, l: u32) -> u32 {
        let hi = ((n as u64 * mp as u64) >> 32) as u32;
        hi.wrapping_add(n) >> l
    }
    /// fabsf [abs.ftz.f32].
    #[inline(always)]
    pub fn absf(x: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("abs.ftz.f32 %0, %1;", out("=f") r, in("f") x, options(register_only)); }
        r
    }
    /// fmaxf [max.ftz.f32].
    #[inline(always)]
    pub fn fmax(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("max.ftz.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    /// fast-math `a / b` [div.approx.ftz.f32].
    #[inline(always)]
    pub fn div_approx(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("div.approx.ftz.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    /// fast-math `1.0f / a` [rcp.approx.ftz.f32].
    #[inline(always)]
    pub fn rcp_approx(a: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("rcp.approx.ftz.f32 %0, %1;", out("=f") r, in("f") a, options(register_only)); }
        r
    }
    /// `__frcp_rn` [rcp.rn.ftz.f32].
    #[inline(always)]
    pub fn rcp_rn(a: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("rcp.rn.ftz.f32 %0, %1;", out("=f") r, in("f") a, options(register_only)); }
        r
    }
    /// `x > 0.0f` [setp.gt.ftz.f32].
    #[inline(always)]
    pub fn gt0(x: f32) -> bool {
        let r: u32;
        unsafe { ptx_asm!("{ .reg .pred p; setp.gt.ftz.f32 p, %1, 0f00000000; selp.u32 %0, 1, 0, p; }", out("=r") r, in("f") x, options(register_only)); }
        r != 0
    }
    /// `x == 0.0f` [setp.eq.ftz.f32].
    #[inline(always)]
    pub fn eq0(x: f32) -> bool {
        let r: u32;
        unsafe { ptx_asm!("{ .reg .pred p; setp.eq.ftz.f32 p, %1, 0f00000000; selp.u32 %0, 1, 0, p; }", out("=r") r, in("f") x, options(register_only)); }
        r != 0
    }
    /// `a < b` [setp.lt.ftz.f32].
    #[inline(always)]
    pub fn lt(a: f32, b: f32) -> bool {
        let r: u32;
        unsafe { ptx_asm!("{ .reg .pred p; setp.lt.ftz.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }", out("=r") r, in("f") a, in("f") b, options(register_only)); }
        r != 0
    }
    /// `(int) roundf(x * d_inv)` with the reference's unrounded `mul.ftz.f32`.
    #[inline(always)]
    pub fn round_mul(x: f32, d_inv: f32) -> i32 {
        let r: i32;
        unsafe {
            ptx_asm!(
                "{ .reg .f32 m, h, s, t; mul.ftz.f32 m, %1, %2; mov.b32 h, 0f3F000000; copysign.f32 h, m, h; add.rz.ftz.f32 s, m, h; cvt.rzi.f32.f32 t, s; cvt.rzi.ftz.s32.f32 %0, t; }",
                out("=r") r, in("f") x, in("f") d_inv, options(register_only)
            );
        }
        r
    }
    /// `compute_e8m0_scale(amax)`: log2f (lg2.approx.ftz), __float2int_rn (cvt.rni.ftz.s32),
    /// `clamp(e - 2 + 127, 0, 254)`; 0 unless amax > 0.
    #[inline(always)]
    pub fn e8m0_scale(amax: f32) -> u32 {
        let r: u32;
        unsafe {
            ptx_asm!(
                "{ .reg .pred p; .reg .f32 l; .reg .s32 e; setp.leu.ftz.f32 p, %1, 0f00000000; mov.u32 %0, 0; @!p lg2.approx.ftz.f32 l, %1; @!p cvt.rni.ftz.s32.f32 e, l; @!p max.s32 e, e, -125; @!p add.s32 e, e, 125; @!p min.u32 %0, e, 254; }",
                out("=r") r, in("f") amax, options(register_only)
            );
        }
        r
    }
    /// `ggml_cuda_e8m0_to_fp32` (CUDART >= 12.8): cvt.rn.bf16x2.ue8m0x2, bf16 -> f32 (exact).
    #[inline(always)]
    pub fn e8m0(e: u32) -> f32 {
        let r: u32;
        let h = e as u16;
        unsafe { ptx_asm!("cvt.rn.bf16x2.ue8m0x2 %0, %1;", out("=r") r, in("h") h, options(register_only)); }
        f32::from_bits((r & 0xFFFF) << 16)
    }
    /// cvt.rn.satfinite.e2m1x2.f32: the byte {hi nibble = hi, lo nibble = lo}.
    #[inline(always)]
    pub fn e2m1x2(hi: f32, lo: f32) -> u32 {
        let r: u16;
        unsafe {
            ptx_asm!(
                "{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %2; mov.b16 %0, {t, 0}; }",
                out("=h") r, in("f") hi, in("f") lo, options(register_only)
            );
        }
        (r & 0xFF) as u32
    }
    /// cvt.rn.f16x2.e2m1x2 of one byte: (f16 lo nibble) | (f16 hi nibble) << 16.
    #[inline(always)]
    pub fn e2m1x2_f16x2(b: u32) -> u32 {
        let r: u32;
        let h = (b & 0xFF) as u16;
        unsafe {
            ptx_asm!(
                "{ .reg .b8 t, z; mov.b16 {t, z}, %1; cvt.rn.f16x2.e2m1x2 %0, t; }",
                out("=r") r, in("h") h, options(register_only)
            );
        }
        r
    }
    /// `ggml_cuda_fp32_to_ue4m3`: 0 unless x > 0, else `__nv_fp8_e4m3(x)` (cvt.rn.satfinite).
    #[inline(always)]
    pub fn f2ue4m3(x: f32) -> u32 {
        let r: u16;
        unsafe {
            ptx_asm!(
                "{ .reg .pred p; .reg .b32 z; mov.b32 z, 0; mov.b16 %0, 0; setp.leu.ftz.f32 p, %1, 0f00000000; @!p cvt.rn.satfinite.e4m3x2.f32 %0, z, %1; }",
                out("=h") r, in("f") x, options(register_only)
            );
        }
        r as u32
    }
    /// `ggml_cuda_ue4m3_to_fp32` (FP8_AVAILABLE): 0x7F / 0xFF -> 0, cvt.rn.f16x2.e4m3x2, f16 -> f32,
    /// `/ 2` as div.approx.ftz.
    #[inline(always)]
    pub fn ue4m3(b: u32) -> f32 {
        let b = if b & 0x7F == 0x7F { 0u16 } else { (b & 0xFF) as u16 };
        let r: u32;
        unsafe { ptx_asm!("cvt.rn.f16x2.e4m3x2 %0, %1;", out("=r") r, in("h") b, options(register_only)); }
        div_approx(h2f(r & 0xFFFF), 2.0)
    }

    /// Read-only global loads as `ld.global.nc` (the reference's `const __restrict__` loads).
    #[inline(always)]
    unsafe fn ldg_u8(p: *const u8) -> u32 {
        let r: u32;
        ptx_asm!("ld.global.nc.u8 %0, [%1];", out("=r") r, in("l") p as u64, options(register_only));
        r
    }
    #[inline(always)]
    unsafe fn ldg_u16(p: *const u8) -> u32 {
        let r: u32;
        ptx_asm!("ld.global.nc.u16 %0, [%1];", out("=r") r, in("l") p as u64, options(register_only));
        r
    }
    #[inline(always)]
    unsafe fn ldg_b32(p: *const i32) -> i32 {
        let r: i32;
        ptx_asm!("ld.global.nc.b32 %0, [%1];", out("=r") r, in("l") p as u64, options(register_only));
        r
    }
    #[inline(always)]
    unsafe fn ldg_f32(p: *const f32) -> f32 {
        let r: f32;
        ptx_asm!("ld.global.nc.f32 %0, [%1];", out("=f") r, in("l") p as u64, options(register_only));
        r
    }

    // ------------------------------------------------------------------------------------------
    // Activation quantizers (quantize.cu). The f32 entries are the reference instances; f16 / bf16
    // inputs are widened exactly (bf16: 16-bit shift, f16: cvt.f32.f16), then the f32 code runs.

    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct H(pub u16);
    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct B(pub u16);

    pub trait Src: Copy {
        unsafe fn ld(p: *const Self) -> f32;
    }
    impl Src for f32 {
        #[inline(always)]
        unsafe fn ld(p: *const f32) -> f32 {
            ldg_f32(p)
        }
    }
    impl Src for H {
        #[inline(always)]
        unsafe fn ld(p: *const H) -> f32 {
            h2f(ldg_u16(p as *const u8))
        }
    }
    impl Src for B {
        #[inline(always)]
        unsafe fn ld(p: *const B) -> f32 {
            f32::from_bits(ldg_u16(p as *const u8) << 16)
        }
    }

    /// `quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4, scatter = false>`: 4 values per thread, grid
    /// (ne1, ceil(ne0 / 512), ne2*ne3), block 128. Params: x, ids, vy, ne00, s01, s02, s03, ne0,
    /// ne1, ne2, n_expert_used (unused).
    #[inline(always)]
    pub unsafe fn quantize_mmq_d4<S: Src>(
        x: *const S, ids: *const i32, vy: *mut u8, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i32, ne2: i32,
    ) {
        let t = thread::blockDim_x().wrapping_mul(thread::blockIdx_y()).wrapping_add(thread::threadIdx_x());
        let i0 = (t as u64 * 4) as i64;
        if ne0 <= i0 {
            return;
        }
        let z = thread::blockIdx_z();
        let i2 = z % (ne2 as u32);
        let i3 = z / (ne2 as u32);
        let bx = thread::blockIdx_x();
        let i01 = if ids.is_null() { bx } else { *ids.add(bx as usize) as u32 };
        let base = s03.wrapping_mul(i3 as i64).wrapping_add(s02.wrapping_mul(i2 as i64)).wrapping_add(s01.wrapping_mul(i01 as i64));
        let (x0, x1, x2, x3) = if i0 < ne00 {
            let e = base.wrapping_add(i0) / 4 * 4;
            let p = x.offset(e as isize);
            (S::ld(p), S::ld(p.add(1)), S::ld(p.add(2)), S::ld(p.add(3)))
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };
        let mut amax = fmax(fmax(fmax(absf(x0), absf(x1)), absf(x2)), absf(x3));
        let mut k = 0;
        while k < 3 {
            unroll!();
            amax = fmax(amax, warp::shuffle_xor_f32_sync(0xffff_ffff, amax, 4 >> k));
            k += 1;
        }
        let d_inv = div_approx(127.0, amax);
        let q0 = round_mul(x0, d_inv);
        let q1 = round_mul(x1, d_inv);
        let q2 = round_mul(x2, d_inv);
        let q3 = round_mul(x3, d_inv);
        let ib0 = (((thread::gridDim_x() as u64 * thread::gridDim_y() as u64).wrapping_mul(thread::blockDim_x() as u64)) >> 5) as i64;
        let ib = ib0.wrapping_mul(z as i64).wrapping_add(bx as i64).wrapping_add((((i0 as u64) >> 7) as i64).wrapping_mul(ne1 as i64));
        let yb = vy.offset(ib.wrapping_mul(144) as isize);
        let iqs = (i0 & 124) as usize;
        *(yb.add(16 + iqs) as *mut u32) =
            (q0 as u32 & 0xFF) | ((q1 as u32 & 0xFF) << 8) | ((q2 as u32 & 0xFF) << 16) | ((q3 as u32 & 0xFF) << 24);
        if (i0 & 28) != 0 {
            return;
        }
        *(yb.add(iqs >> 3) as *mut f32) = rcp_approx(d_inv);
    }

    /// `quantize_mmq_mxfp4<scatter = false>`: block (32, 8), grid (ne1, ceil(ne0 / 512), ne2*ne3);
    /// each warp quantizes 64 values (two E8M0 blocks of 32) into block_fp4_mmq, nibbles
    /// interleaved as MXFP4 stores them (a0 a16, a1 a17, ...).
    /// Params: x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2, n_expert_used (unused).
    #[inline(always)]
    pub unsafe fn quantize_mmq_mxfp4<S: Src>(
        x: *const S, ids: *const i32, vy: *mut u8, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i32, ne2: i32,
    ) {
        let warp_id = thread::threadIdx_y();
        let lane = thread::threadIdx_x();
        let ws = (thread::blockIdx_y().wrapping_mul(thread::blockDim_y()).wrapping_add(warp_id) as u64) * 64;
        if ne0 <= ws as i64 {
            return;
        }
        let quad = ((ws >> 6) & 3) as usize;
        let group_id = lane / 4;
        let base = group_id * 2;
        let z = thread::blockIdx_z();
        let i2 = z % (ne2 as u32);
        let i3 = z / (ne2 as u32);
        let bx = thread::blockIdx_x();
        let i01 = if ids.is_null() { bx } else { ldg_b32(ids.add(bx as usize)) as u32 };
        let base_pos = s02.wrapping_mul(i2 as i64).wrapping_add(s03.wrapping_mul(i3 as i64)).wrapping_add(s01.wrapping_mul(i01 as i64));
        let mut scales = [0u32; 2];
        let mut packed = [0u32; 2];
        let mut b = 0;
        while b < 2 {
            unroll!();
            let i0 = (ws + (b as u64) * 32 + lane as u64) as i64;
            let xi = if i0 < ne00 { S::ld(x.offset(base_pos.wrapping_add(i0) as isize)) } else { 0.0 };
            let mut amax = absf(xi);
            let mut k = 0;
            while k < 5 {
                unroll!();
                amax = fmax(amax, warp::shuffle_xor_f32_sync(0xffff_ffff, amax, 16 >> k));
                k += 1;
            }
            let e = e8m0_scale(amax);
            let inv_s = if eq0(amax) { 0.0 } else { rcp_rn(e8m0(e)) };
            let sv = mulf(xi, inv_s);
            let v0 = warp::shuffle_f32_sync(0xffff_ffff, sv, base);
            let v1 = warp::shuffle_f32_sync(0xffff_ffff, sv, base + 16);
            let v2 = warp::shuffle_f32_sync(0xffff_ffff, sv, base + 1);
            let v3 = warp::shuffle_f32_sync(0xffff_ffff, sv, base + 17);
            let p = e2m1x2(v1, v0) | (e2m1x2(v3, v2) << 8);
            if b == 0 {
                scales[0] = e;
                packed[0] = p;
            } else {
                scales[1] = e;
                packed[1] = p;
            }
            b += 1;
        }
        let ib = ((ne0 as u64 >> 8) as i64)
            .wrapping_mul(z as i64)
            .wrapping_add((ws >> 8) as i64)
            .wrapping_mul(ne1 as i64)
            .wrapping_add(bx as i64);
        let yb = vy.offset(ib.wrapping_mul(144) as isize);
        if lane % 4 == 0 {
            let q = yb.add(16 + (quad * 16 + group_id as usize) * 2) as *mut u16;
            *q = packed[0] as u16;
            *q.add(8) = packed[1] as u16;
        }
        if lane == 0 {
            *(yb.add(quad * 4) as *mut u32) = (scales[1] << 8) | scales[0];
        }
    }

    /// `nvfp4_native_scale_error(vals * inv_col_scale, ., inv_scale, scale)`: the sum of squared
    /// `|v| - |fp4(v * inv_scale)| * 2 * scale`, each term one FFMA as in the SASS.
    #[inline(always)]
    fn nv_err(v: &[f32; 16], inv_scale: f32, scale: f32) -> f32 {
        let s2 = add(scale, scale);
        let mut err = 0.0f32;
        let mut k = 0;
        while k < 16 {
            unroll!();
            let b0 = e2m1x2(mulf(inv_scale, v[k + 1]), mulf(inv_scale, v[k]));
            let b1 = e2m1x2(mulf(inv_scale, v[k + 3]), mulf(inv_scale, v[k + 2]));
            let h0 = e2m1x2_f16x2(b0);
            let h1 = e2m1x2_f16x2(b1);
            let e0 = fnms_abs(h2f(h0 & 0xFFFF), s2, v[k]);
            let e1 = fnms_abs(h2f(h0 >> 16), s2, v[k + 1]);
            let e2 = fnms_abs(h2f(h1 & 0xFFFF), s2, v[k + 2]);
            let e3 = fnms_abs(h2f(h1 >> 16), s2, v[k + 3]);
            err = fma(e0, e0, err);
            err = fma(e1, e1, err);
            err = fma(e2, e2, err);
            err = fma(e3, e3, err);
            k += 4;
        }
        err
    }

    /// `(0.5f / s)` if s > 0 else 0.
    #[inline(always)]
    fn half_inv(s: f32) -> f32 {
        if gt0(s) { div_approx(0.5, s) } else { 0.0 }
    }

    /// `quantize_mmq_nvfp4<scatter = false, use_aligned_float8>` (both variants give the same bytes
    /// for ne00 % 64 == 0): block 128, grid (ne1, ne2*ne3). Row amax (all 128 threads, then
    /// warp 0) -> `scale[row] = amax / 2688`; then one thread per 16-value sub-block.
    /// Params: x, ids, vy, scale, ne00, s01, s02, s03, ne0, ne1, ne2, n_expert_used (unused).
    #[inline(always)]
    pub unsafe fn quantize_mmq_nvfp4<S: Src>(
        x: *const S, ids: *const i32, vy: *mut u8, scale: *mut f32, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i64,
        ne2: i64,
    ) {
        static mut WARP_AMAX: SharedArray<f32, 4> = SharedArray::UNINIT;
        let wa = SharedArray::as_raw_mut_ptr(&raw mut WARP_AMAX);
        let by = thread::blockIdx_y() as u64;
        // i2 = by % ne2, i3 = by / ne2 with nvcc's 64-bit division bypass
        let (i3, i2) = if (ne2 as u64) >> 32 == 0 {
            let q = (by as u32) / (ne2 as u32);
            (q as i64, (by as u32).wrapping_sub(q.wrapping_mul(ne2 as u32)) as i64)
        } else {
            let q = (by as i64) / ne2;
            (q, (by as i64).wrapping_sub(q.wrapping_mul(ne2)))
        };
        let bx = thread::blockIdx_x();
        let i01 = if ids.is_null() { bx } else { ldg_b32(ids.add(bx as usize)) as u32 };
        let base_idx = s02.wrapping_mul(i2).wrapping_add(s03.wrapping_mul(i3)).wrapping_add(s01.wrapping_mul(i01 as i64));
        let x_row = x.offset(base_idx as isize);
        let tid = thread::threadIdx_x();
        let bdim = thread::blockDim_x() as i64;

        let mut amax = 0.0f32;
        let mut i0 = 8 * tid as i64;
        while i0 < ne00 {
            let p = x_row.offset(i0 as isize);
            let mut k = 0;
            while k < 8 {
                unroll!();
                amax = fmax(amax, absf(S::ld(p.add(k))));
                k += 1;
            }
            i0 += 8 * bdim;
        }
        let mut k = 0;
        while k < 5 {
            unroll!();
            amax = fmax(amax, warp::shuffle_xor_f32_sync(0xffff_ffff, amax, 16 >> k));
            k += 1;
        }
        let lane = tid % 32;
        let wid = tid / 32;
        if lane == 0 {
            *wa.add(wid as usize) = amax;
        }
        thread::sync_threads();
        if wid == 0 {
            let mut a = if tid < 4 { *wa.add(lane as usize) } else { 0.0 };
            let mut k = 0;
            while k < 5 {
                unroll!();
                a = fmax(a, warp::shuffle_xor_f32_sync(0xffff_ffff, a, 16 >> k));
                k += 1;
            }
            if lane == 0 {
                let rs = div_approx(a, 2688.0);
                *wa = rs;
                *scale.offset((ne1.wrapping_mul(by as i64)).wrapping_add(bx as i64) as isize) = rs;
            }
        }
        thread::sync_threads();

        let n_sub = (ne0 + 15) / 16;
        let row_scale = *wa;
        let inv_col = if gt0(row_scale) { rcp_approx(row_scale) } else { 0.0 };
        let bpc = (ne0 + 255) / 256;
        let mut isb = tid as i64;
        while isb < n_sub {
            let i0b = isb * 16;
            let mut vals = [0f32; 16];
            if i0b + 7 < ne00 {
                let p = x_row.offset(i0b as isize);
                let mut k = 0;
                while k < 8 {
                    unroll!();
                    vals[k] = S::ld(p.add(k));
                    k += 1;
                }
            }
            if i0b + 15 < ne00 {
                let p = x_row.offset(i0b as isize + 8);
                let mut k = 0;
                while k < 8 {
                    unroll!();
                    vals[8 + k] = S::ld(p.add(k));
                    k += 1;
                }
            }
            let mut v = [0f32; 16];
            let mut amax_sub = 0.0f32;
            let mut k = 0;
            while k < 16 {
                unroll!();
                v[k] = mulf(inv_col, vals[k]);
                amax_sub = fmax(amax_sub, absf(v[k]));
                k += 1;
            }
            let first = f2ue4m3(div_approx(amax_sub, 6.0));
            let first_i = (first & 0xFF) as i32;
            let mut code = first;
            let mut sscale = ue4m3(first);
            let mut best = nv_err(&v, half_inv(sscale), sscale);
            // test_offsets 0, -1, 1, -2, 2
            let mut t = 1;
            while t < 5 {
                unroll!();
                let off = if t == 1 { -1 } else if t == 2 { 1 } else if t == 3 { -2 } else { 2 };
                let tc = first_i + off;
                if !(tc < 0 || tc > 0x7e) {
                    let ts = ue4m3(tc as u32);
                    let cur = nv_err(&v, half_inv(ts), ts);
                    if lt(cur, best) {
                        best = cur;
                        code = tc as u32;
                        sscale = ts;
                    }
                }
                t += 1;
            }
            let s = mulf(inv_col, half_inv(sscale));
            let mut q0 = 0u32;
            let mut q1 = 0u32;
            let mut k = 0;
            while k < 4 {
                unroll!();
                q0 |= e2m1x2(mulf(vals[k + 8], s), mulf(vals[k], s)) << (8 * k);
                q1 |= e2m1x2(mulf(vals[k + 12], s), mulf(vals[k + 4], s)) << (8 * k);
                k += 1;
            }
            let ib = ((isb >> 4) + bpc.wrapping_mul(by as i64)).wrapping_mul(ne1).wrapping_add(bx as i64);
            let yb = vy.offset(ib.wrapping_mul(144) as isize);
            let sub = (isb & 15) as usize;
            let yq = yb.add(16 + 8 * sub) as *mut u32;
            *yq = q0;
            *yq.add(1) = q1;
            *yb.add(sub) = code as u8;
            isb += bdim;
        }
    }

    #[kernel]
    pub unsafe fn iq4_nl_quantize_mmq_d4_f32(x: *const f32, ids: *const i32, vy: *mut u8, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i32, ne2: i32, _n_expert_used: i32) {
        quantize_mmq_d4(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2)
    }
    #[kernel]
    pub unsafe fn iq4_nl_quantize_mmq_d4_f16(x: *const u16, ids: *const i32, vy: *mut u8, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i32, ne2: i32, _n_expert_used: i32) {
        quantize_mmq_d4(x as *const H, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2)
    }
    #[kernel]
    pub unsafe fn iq4_nl_quantize_mmq_d4_bf16(x: *const u16, ids: *const i32, vy: *mut u8, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i32, ne2: i32, _n_expert_used: i32) {
        quantize_mmq_d4(x as *const B, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2)
    }
    #[kernel]
    pub unsafe fn mxfp4_quantize_mmq_f32(x: *const f32, ids: *const i32, vy: *mut u8, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i32, ne2: i32, _n_expert_used: i32) {
        quantize_mmq_mxfp4(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2)
    }
    #[kernel]
    pub unsafe fn mxfp4_quantize_mmq_f16(x: *const u16, ids: *const i32, vy: *mut u8, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i32, ne2: i32, _n_expert_used: i32) {
        quantize_mmq_mxfp4(x as *const H, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2)
    }
    #[kernel]
    pub unsafe fn mxfp4_quantize_mmq_bf16(x: *const u16, ids: *const i32, vy: *mut u8, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i32, ne2: i32, _n_expert_used: i32) {
        quantize_mmq_mxfp4(x as *const B, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2)
    }
    #[kernel]
    pub unsafe fn nvfp4_quantize_mmq_f32(x: *const f32, ids: *const i32, vy: *mut u8, scale: *mut f32, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i64, ne2: i64, _n_expert_used: i32) {
        quantize_mmq_nvfp4(x, ids, vy, scale, ne00, s01, s02, s03, ne0, ne1, ne2)
    }
    #[kernel]
    pub unsafe fn nvfp4_quantize_mmq_f16(x: *const u16, ids: *const i32, vy: *mut u8, scale: *mut f32, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i64, ne2: i64, _n_expert_used: i32) {
        quantize_mmq_nvfp4(x as *const H, ids, vy, scale, ne00, s01, s02, s03, ne0, ne1, ne2)
    }
    #[kernel]
    pub unsafe fn nvfp4_quantize_mmq_bf16(x: *const u16, ids: *const i32, vy: *mut u8, scale: *mut f32, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i64, ne2: i64, _n_expert_used: i32) {
        quantize_mmq_nvfp4(x as *const B, ids, vy, scale, ne00, s01, s02, s03, ne0, ne1, ne2)
    }

    // ------------------------------------------------------------------------------------------
    // MMQ: `mul_mat_q<type, J, fallback>` + `mul_mat_q_stream_k_fixup` (mmq.cuh, MMA data layout;
    // 256 threads, I = 128, SRAM stride 76, stream-k). Shared memory: ids[J] | tile_y (J*36 ints
    // padded to 256) | tile_x (128 rows x 76 ints: 64 quant ints, then 8 scale words).

    const NWARPS: i32 = 8;
    const WARP: i32 = 32;
    const MMQ_I: i32 = 128;
    const TILE_NE_K: i32 = 32;
    const TILE_Y_K: i32 = 36;
    const SRAM: i32 = 76;
    const SZ: i32 = 36;

    /// Per-J constants as associated consts (the unroll pass needs literal loop bounds).
    pub struct Jc<const J: i32>;
    impl<const J: i32> Jc<J> {
        /// rows_per_warp / 16.
        pub const NTX: i32 = if J >= 48 && J % 16 == 0 { 2 } else { 1 };
        pub const JSTEP: i32 = Self::NTX * 8;
        pub const YLEN: i32 = J * TILE_Y_K;
        pub const YPAD: i32 = (J * TILE_Y_K + 255) / 256 * 256;
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
    fn smem() -> *mut i32 {
        DynamicSharedArray::<i32>::get()
    }

    #[derive(Clone, Copy)]
    pub struct Fd {
        pub mp: u32,
        pub l: u32,
        pub d: u32,
    }
    #[inline(always)]
    fn fdiv(n: u32, f: Fd) -> u32 {
        fastdiv(n, f.mp, f.l)
    }
    #[inline(always)]
    fn fmod(n: u32, f: Fd) -> u32 {
        n.wrapping_sub(fdiv(n, f).wrapping_mul(f.d))
    }

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
    /// `mma_block_scaled_fp4<MXFP4>`: m16n8k64 e2m1 x e2m1, one packed ue8m0 scale pair per
    /// thread for A and B, zero C.
    #[inline(always)]
    unsafe fn mma_mxf4(a: [u32; 4], b: [u32; 2], sa: u32, sb: u32) -> [f32; 4] {
        let (d0, d1, d2, d3): (f32, f32, f32, f32);
        ptx_asm!(
            "{ .reg .f32 z; mov.b32 z, 0f00000000; mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {z, z, z, z}, %10, {0, 0}, %11, {0, 0}; }",
            out("=f") d0, out("=f") d1, out("=f") d2, out("=f") d3,
            in("r") a[0], in("r") a[1], in("r") a[2], in("r") a[3], in("r") b[0], in("r") b[1], in("r") sa, in("r") sb,
        );
        [d0, d1, d2, d3]
    }
    /// `mma_block_scaled_fp4<NVFP4>`: m16n8k64 e2m1 x e2m1, four ue4m3 scales per thread for A
    /// and B, zero C.
    #[inline(always)]
    unsafe fn mma_nvf4(a: [u32; 4], b: [u32; 2], sa: u32, sb: u32) -> [f32; 4] {
        let (d0, d1, d2, d3): (f32, f32, f32, f32);
        ptx_asm!(
            "{ .reg .f32 z; mov.b32 z, 0f00000000; mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {z, z, z, z}, %10, {0, 0}, %11, {0, 0}; }",
            out("=f") d0, out("=f") d1, out("=f") d2, out("=f") d3,
            in("r") a[0], in("r") a[1], in("r") a[2], in("r") a[3], in("r") b[0], in("r") b[1], in("r") sa, in("r") sb,
        );
        [d0, d1, d2, d3]
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

    /// `(const block_t *) x + kbx0 + i*stride + kb`: each int offset sign-extended separately.
    #[inline(always)]
    fn xblock(x: *const u8, bytes: i32, kbx0: i32, i: i32, stride: i32, kb: i32) -> *const u8 {
        let off = (kbx0 as i64).wrapping_add(i.wrapping_mul(stride) as i64).wrapping_add(kb as i64);
        x.wrapping_offset(off.wrapping_mul(bytes as i64) as isize)
    }

    /// `kvalues_iq4nl` as the four little-endian words `get_int_from_table_16` reads.
    const IQ4NL_TABLE: [u32; 4] = [0xBFAD9881, 0xF6EADDCF, 0x26190D01, 0x71594535];

    /// `get_int_from_table_16(q4, kvalues_iq4nl)`: (bytes of the low nibbles, of the high nibbles).
    #[inline(always)]
    fn table16(q4: u32) -> (u32, u32) {
        let sel = 0x32103210 | ((q4 & 0x88888888) >> 1);
        let lo0 = byte_perm(IQ4NL_TABLE[0], IQ4NL_TABLE[1], q4);
        let hi0 = byte_perm(IQ4NL_TABLE[2], IQ4NL_TABLE[3], q4);
        let t0 = byte_perm(lo0, hi0, sel);
        let lo1 = byte_perm(IQ4NL_TABLE[0], IQ4NL_TABLE[1], q4 >> 16);
        let hi1 = byte_perm(IQ4NL_TABLE[2], IQ4NL_TABLE[3], q4 >> 16);
        let t1 = byte_perm(lo1, hi1, sel >> 16);
        (byte_perm(t0, t1, 0x6420), byte_perm(t0, t1, 0x7531))
    }

    /// A weight format's MMQ pieces.
    pub trait Fmt {
        /// values per weight block
        const QK: i32;
        /// weight block bytes
        const BYTES: i32;
        /// K_vram / QK: weight blocks per tile iteration
        const BPI: i32;
        /// values per activation block (block_q8_1_mmq 128, block_fp4_mmq 256)
        const NEB: i32;
        /// NVFP4: the write-back multiplies by the per-column activation scale
        const YSCALE: bool;
        unsafe fn load_tiles<const J: i32, const FB: bool>(x: *const u8, kbx0: i32, i_max: i32, stride: i32);
        unsafe fn vec_dot<const J: i32>(sum: &mut [f32; 64], k00: i32);
    }
    pub struct Iq4Nl;
    pub struct Mxfp4;
    pub struct Nvfp4;

    impl Fmt for Iq4Nl {
        const QK: i32 = 32;
        const BYTES: i32 = 18;
        const BPI: i32 = 8;
        const NEB: i32 = 128;
        const YSCALE: bool = false;
        /// ggml_cuda_mmq_load_tiles_iq4_nl (MMA layout): one row per warp and pass, lane ->
        /// (block lane/4, int lane%4); scales: 4 rows per warp and pass, 8 blocks per row.
        #[inline(always)]
        unsafe fn load_tiles<const J: i32, const FB: bool>(x: *const u8, kbx0: i32, i_max: i32, stride: i32) {
            let x_qs = smem().wrapping_add((J + Jc::<J>::YPAD) as usize);
            let x_df = x_qs.wrapping_add(2 * TILE_NE_K as usize) as *mut f32;
            let kbx = tx() / 4;
            let kqsx = tx() % 4;
            let k0 = kbx * 8 + kqsx;
            let mut i0 = 0;
            while i0 < MMQ_I {
                unroll!();
                let mut i = i0 + ty();
                if FB {
                    i = if i < i_max { i } else { i_max };
                }
                let q = xblock(x, Self::BYTES, kbx0, i, stride, kbx).wrapping_add(2 + 4 * kqsx as usize);
                let aux = ldg_u16(q) | (ldg_u16(q.wrapping_add(2)) << 16);
                let (v0, v1) = table16(aux);
                let row = x_qs.wrapping_offset((i * SRAM + k0) as isize);
                *row = v0 as i32;
                *row.add(4) = v1 as i32;
                i0 += NWARPS;
            }
            let kbxd = tx() % 8;
            let mut i0 = 0;
            while i0 < MMQ_I {
                unroll!();
                let mut i = i0 + ty() * 4 + tx() / 8;
                if FB {
                    i = if i < i_max { i } else { i_max };
                }
                let b = xblock(x, Self::BYTES, kbx0, i, stride, kbxd);
                *x_df.offset((i * SRAM + kbxd) as isize) = h2f(ldg_u16(b));
                i0 += 4 * NWARPS;
            }
        }
        #[inline(always)]
        unsafe fn vec_dot<const J: i32>(sum: &mut [f32; 64], k00: i32) {
            vec_dot_q8_0::<J>(sum, k00)
        }
    }

    impl Fmt for Mxfp4 {
        const QK: i32 = 32;
        const BYTES: i32 = 17;
        const BPI: i32 = 16;
        const NEB: i32 = 256;
        const YSCALE: bool = false;
        /// ggml_cuda_mmq_load_tiles_mxfp4_fp4: 16 blocks per row (one per lane), 2 rows per warp;
        /// the 16 nibble bytes are copied as is, even lanes pack the E8M0 bytes of blocks
        /// (kbx, kbx+1) into scale word kbx/2.
        #[inline(always)]
        unsafe fn load_tiles<const J: i32, const FB: bool>(x: *const u8, kbx0: i32, i_max: i32, stride: i32) {
            let x_qs = smem().wrapping_add((J + Jc::<J>::YPAD) as usize);
            let kbx = tx() % 16;
            let r = tx() / 16;
            let mut i0 = 0;
            while i0 < MMQ_I {
                unroll!();
                let mut i = i0 + ty() * 2 + r;
                if FB {
                    i = if i < i_max { i } else { i_max };
                }
                let b = xblock(x, Self::BYTES, kbx0, i, stride, kbx);
                let row = x_qs.wrapping_offset((i * SRAM) as isize);
                let dst = row.wrapping_offset((kbx * 4) as isize);
                let mut w = 0;
                while w < 4 {
                    unroll!();
                    let q = b.wrapping_add(1 + 4 * w);
                    *dst.add(w) = (ldg_u8(q) | (ldg_u8(q.wrapping_add(1)) << 8) | (ldg_u8(q.wrapping_add(2)) << 16) | (ldg_u8(q.wrapping_add(3)) << 24)) as i32;
                    w += 1;
                }
                if kbx % 2 == 0 {
                    let e = ldg_u8(b) | (ldg_u8(b.wrapping_add(17)) << 8);
                    *row.wrapping_offset((2 * TILE_NE_K + kbx / 2) as isize) = e as i32;
                }
                i0 += 2 * NWARPS;
            }
        }
        #[inline(always)]
        unsafe fn vec_dot<const J: i32>(sum: &mut [f32; 64], k00: i32) {
            vec_dot_fp4::<J, false>(sum, k00)
        }
    }

    impl Fmt for Nvfp4 {
        const QK: i32 = 64;
        const BYTES: i32 = 36;
        const BPI: i32 = 8;
        const NEB: i32 = 256;
        const YSCALE: bool = true;
        /// ggml_cuda_mmq_load_tiles_nvfp4_nvfp4: 8 blocks per row (one per lane), 4 rows per warp;
        /// the 8 nibble words are copied as is, the 4 UE4M3 bytes become scale word kbx.
        #[inline(always)]
        unsafe fn load_tiles<const J: i32, const FB: bool>(x: *const u8, kbx0: i32, i_max: i32, stride: i32) {
            let x_qs = smem().wrapping_add((J + Jc::<J>::YPAD) as usize);
            let kbx = tx() % 8;
            let r = tx() / 8;
            let mut i0 = 0;
            while i0 < MMQ_I {
                unroll!();
                let mut i = i0 + ty() * 4 + r;
                if FB {
                    i = if i < i_max { i } else { i_max };
                }
                let b = xblock(x, Self::BYTES, kbx0, i, stride, kbx) as *const i32;
                let row = x_qs.wrapping_offset((i * SRAM) as isize);
                let dst = row.wrapping_offset((8 * kbx) as isize);
                let mut w = 0;
                while w < 8 {
                    unroll!();
                    *dst.add(w) = ldg_b32(b.wrapping_add(1 + w));
                    w += 1;
                }
                *row.wrapping_offset((2 * TILE_NE_K + kbx) as isize) = ldg_b32(b);
                i0 += 4 * NWARPS;
            }
        }
        #[inline(always)]
        unsafe fn vec_dot<const J: i32>(sum: &mut [f32; 64], k00: i32) {
            vec_dot_fp4::<J, true>(sum, k00)
        }
    }

    /// ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<.., MMQ_Q8_1_DS_LAYOUT_D4> (NVIDIA branch):
    /// `sum += (dA * (float) C) * dB` as mul.ftz + fma.ftz.
    #[inline(always)]
    unsafe fn vec_dot_q8_0<const J: i32>(sum: &mut [f32; 64], k00: i32) {
        let ntx = Jc::<J>::NTX;
        let rows_per_warp = 16 * ntx;
        let y = smem().wrapping_add(J as usize).wrapping_offset(((ty() % ntx) * (8 * TILE_Y_K)) as isize) as *const i32;
        let x_qs = smem().wrapping_add((J + Jc::<J>::YPAD) as usize) as *const i32;
        let x_df = x_qs.wrapping_add(2 * TILE_NE_K as usize) as *const f32;
        let y_qs = y.wrapping_add(4);
        let y_df = y as *const f32;
        let i0 = (ty() / ntx) * rows_per_warp;

        let mut a = [[[0u32; 4]; 4]; 2];
        let mut da = [[[0f32; 4]; 2]; 2];
        let mut n = 0;
        while n < Jc::<J>::NTX {
            unroll!();
            let mut k01 = 0;
            while k01 < TILE_NE_K {
                unroll!();
                let k0 = k00 + k01;
                a[n as usize][(k01 / 8) as usize] = ldmatrix_a(x_qs.wrapping_offset(((i0 + n * 16) * SRAM + k0) as isize), SRAM);
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
                    da[n as usize][l as usize][(k01 / 8) as usize] = *x_df.offset((i * SRAM + k0 / 8) as isize);
                    k01 += 8;
                }
                l += 1;
            }
            n += 1;
        }

        let mut j0 = 0;
        while j0 < J {
            unroll!();
            let mut k01 = 0;
            while k01 < TILE_NE_K {
                unroll!();
                let b = load_b(y_qs.wrapping_offset((j0 * TILE_Y_K + k01) as isize), TILE_Y_K);
                let db0 = *y_df.offset(((j0 + c_j(0)) * TILE_Y_K + k01 / 8) as isize);
                let db1 = *y_df.offset(((j0 + c_j(1)) * TILE_Y_K + k01 / 8) as isize);
                let mut n = 0;
                while n < Jc::<J>::NTX {
                    unroll!();
                    let c = mma_s8(a[n as usize][(k01 / 8) as usize], b);
                    let mut l = 0;
                    while l < 4 {
                        unroll!();
                        let idx = ((j0 / 8 + n) * 4 + l) as usize;
                        let db = if l % 2 == 0 { db0 } else { db1 };
                        sum[idx] = fma(mul(da[n as usize][(l / 2) as usize][(k01 / 8) as usize], c[l as usize] as f32), db, sum[idx]);
                        l += 1;
                    }
                    n += 1;
                }
                k01 += 8;
            }
            j0 += Jc::<J>::JSTEP;
        }
    }

    /// ggml_cuda_mmq_vec_dot_fp4_fp4_mma: per 64-value fragment one block-scaled m16n8k64 MMA with
    /// zero C, then `sum += C` (add.ftz). A scale: row lane/4 + (lane%2)*8 of the 16-row tile;
    /// B scale: column lane/4.
    #[inline(always)]
    unsafe fn vec_dot_fp4<const J: i32, const NV: bool>(sum: &mut [f32; 64], k00: i32) {
        let ntx = Jc::<J>::NTX;
        let rows_per_warp = 16 * ntx;
        let y = smem().wrapping_add(J as usize).wrapping_offset(((ty() % ntx) * (8 * TILE_Y_K)) as isize) as *const i32;
        let x_qs = smem().wrapping_add((J + Jc::<J>::YPAD) as usize) as *const i32;
        let x_sc = x_qs.wrapping_add(2 * TILE_NE_K as usize);
        let y_qs = y.wrapping_add(4);
        let y_sc = y;
        let tidx_a = tx() / 4 + (tx() % 2) * 8;
        let tidx_b = tx() / 4;
        let i0 = (ty() / ntx) * rows_per_warp;

        let mut a = [[[0u32; 4]; 4]; 2];
        let mut sa = [[0u32; 4]; 2];
        let mut n = 0;
        while n < Jc::<J>::NTX {
            unroll!();
            let mut f = 0;
            while f < 4 {
                unroll!();
                let k0 = k00 + f * 8;
                a[n as usize][f as usize] = ldmatrix_a(x_qs.wrapping_offset(((i0 + n * 16) * SRAM + k0) as isize), SRAM);
                sa[n as usize][f as usize] = *x_sc.offset(((i0 + n * 16 + tidx_a) * SRAM + k0 / 8) as isize) as u32;
                f += 1;
            }
            n += 1;
        }

        let mut j0 = 0;
        while j0 < J {
            unroll!();
            let mut b = [[0u32; 2]; 4];
            let mut sb = [0u32; 4];
            let mut f = 0;
            while f < 4 {
                unroll!();
                b[f as usize] = load_b(y_qs.wrapping_offset((j0 * TILE_Y_K + f * 8) as isize), TILE_Y_K);
                sb[f as usize] = *y_sc.offset(((j0 + tidx_b) * TILE_Y_K + f) as isize) as u32;
                f += 1;
            }
            let mut n = 0;
            while n < Jc::<J>::NTX {
                unroll!();
                let mut f = 0;
                while f < 4 {
                    unroll!();
                    let c = if NV {
                        mma_nvf4(a[n as usize][f as usize], b[f as usize], sa[n as usize][f as usize], sb[f as usize])
                    } else {
                        mma_mxf4(a[n as usize][f as usize], b[f as usize], sa[n as usize][f as usize], sb[f as usize])
                    };
                    let mut l = 0;
                    while l < 4 {
                        unroll!();
                        let idx = ((j0 / 8 + n) * 4 + l) as usize;
                        sum[idx] = add(c[l as usize], sum[idx]);
                        l += 1;
                    }
                    f += 1;
                }
                n += 1;
            }
            j0 += Jc::<J>::JSTEP;
        }
    }

    /// ggml_cuda_mmq_write_back_mma: `dst[ids_dst[j]*stride + i] = [y_scale[j] *] sum[...]`.
    #[inline(always)]
    unsafe fn write_back<F: Fmt, const J: i32, const FB: bool>(
        sum: &[f32; 64], dst: *mut f32, y_scale: *const f32, stride: i32, i_max: i32, j_max: i32,
    ) {
        let ids = smem();
        let ntx = Jc::<J>::NTX;
        let i0 = (ty() / ntx) * (ntx * 16);
        let ys = F::YSCALE & !y_scale.is_null();
        let mut j0 = 0;
        while j0 < J {
            unroll!();
            let mut n = 0;
            while n < Jc::<J>::NTX {
                unroll!();
                let mut l = 0;
                while l < 4 {
                    unroll!();
                    let j = j0 + (ty() % ntx) * 8 + c_j(l);
                    let i = i0 + n * 16 + c_i(l);
                    if (j <= j_max) & !(FB & (i > i_max)) {
                        let v = sum[((j0 / 8 + n) * 4 + l) as usize];
                        let v = if ys { mul(ldg_f32(y_scale.offset(j as isize)), v) } else { v };
                        let o = (*ids.offset(j as isize)).wrapping_mul(stride).wrapping_add(i);
                        *dst.offset(o as isize) = v;
                    }
                    l += 1;
                }
                n += 1;
            }
            j0 += Jc::<J>::JSTEP;
        }
    }

    #[inline(always)]
    unsafe fn load_tile_y<const J: i32>(by0: *const i32) {
        let t = smem().wrapping_add(J as usize);
        let mut l0 = 0;
        while l0 < Jc::<J>::YLEN {
            unroll!();
            let l = l0 + ty() * WARP + tx();
            *t.offset(l as isize) = ldg_b32(by0.wrapping_offset(l as isize));
            l0 += NWARPS * WARP;
        }
    }

    /// mul_mat_q_process_tile.
    #[inline(always)]
    unsafe fn process_tile<F: Fmt, const J: i32, const FB: bool, const FIXUP: bool>(
        x: *const u8, offset_x: i32, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, stride_row_x: i32,
        ncols_y: i32, stride_col_dst: i32, tile_x_max_i: i32, tile_y_max_j: i32, kb0_start: i32, kb0_stop: i32,
    ) {
        let mut sum = [0f32; 64];
        let mut kb0 = kb0_start;
        while kb0 < kb0_stop {
            F::load_tiles::<J, FB>(x, offset_x.wrapping_add(kb0), tile_x_max_i, stride_row_x);
            let yb = kb0.wrapping_mul(F::QK) / F::NEB * SZ;
            load_tile_y::<J>(y.wrapping_offset(ncols_y.wrapping_mul(yb) as isize));
            thread::sync_threads();
            F::vec_dot::<J>(&mut sum, 0);
            thread::sync_threads();
            load_tile_y::<J>(y.wrapping_offset(ncols_y.wrapping_mul(yb.wrapping_add(SZ)) as isize));
            thread::sync_threads();
            F::vec_dot::<J>(&mut sum, TILE_NE_K);
            thread::sync_threads();
            kb0 += F::BPI;
        }
        if FIXUP {
            let t = tmp_fixup.wrapping_offset((thread::blockIdx_x() as i32).wrapping_mul(J * MMQ_I) as isize);
            write_back::<F, J, FB>(&sum, t, y_scale, MMQ_I, MMQ_I, J);
        } else {
            write_back::<F, J, FB>(&sum, dst, y_scale, stride_col_dst, tile_x_max_i, tile_y_max_j);
        }
    }

    /// `ids_dst_shared[j] = f(j)` for j < J (256 threads, one pass: J <= 128).
    #[inline(always)]
    unsafe fn fill_ids<const J: i32, G: Fn(i32) -> i32>(f: G) {
        let j = ty() * WARP + tx();
        if !(NWARPS * WARP > J && j >= J) {
            *smem().offset(j as isize) = f(j);
        }
    }

    /// The tile of stream-k index `kbc`: (it, wt, zt, jt).
    #[inline(always)]
    fn tile_of(kbc: i32, bpn: Fd, ntx: Fd, ncy: Fd, nsy: Fd) -> (i32, i32, i32, i32) {
        let tmp = fdiv(kbc as u32, bpn);
        let (d1, jt) = (fdiv(tmp, ntx), fmod(tmp, ntx));
        let (d2, zt) = (fdiv(d1, ncy), fmod(d1, ncy));
        let (it, wt) = (fdiv(d2, nsy), fmod(d2, nsy));
        (it as i32, wt as i32, zt as i32, jt as i32)
    }

    /// `y_scale + offset_y_scale` for NVFP4 (null for the others or when y_scale is null).
    #[inline(always)]
    fn y_scale_tile<F: Fmt>(y_scale: *const f32, off: i32) -> *const f32 {
        if F::YSCALE & !y_scale.is_null() { y_scale.wrapping_offset(off as isize) } else { core::ptr::null() }
    }

    /// Stream-k `mul_mat_q<type, J, fallback>` (dense, or MoE via ids_dst / expert_bounds).
    #[inline(always)]
    pub unsafe fn mul_mat_q<F: Fmt, const J: i32, const FB: bool>(
        x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32,
        y_scale: *const f32, bpn: Fd, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32,
        channel_ratio: Fd, ncy: Fd, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32,
        sample_ratio: Fd, nsy: Fd, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx: Fd,
    ) {
        let nty = (nrows_x.wrapping_add(MMQ_I - 1) / MMQ_I) as u32;
        fill_ids::<J, _>(|j| j);
        thread::sync_threads();

        let bpi = F::BPI as u32;
        let total = nsy.d.wrapping_mul(ncy.d).wrapping_mul(ntx.d).wrapping_mul(nty).wrapping_mul(bpn.d);
        let gdim = thread::gridDim_x() as i64;
        let bid = thread::blockIdx_x() as i64;
        let mut kbc = (bid.wrapping_mul(total as i64) / gdim) as i32;
        let mut kbc_stop = ((bid + 1).wrapping_mul(total as i64) / gdim) as i32;
        kbc = (kbc as u32).wrapping_sub(fmod(kbc as u32, bpn) % bpi) as i32;
        kbc_stop = (kbc_stop as u32).wrapping_sub(fmod(kbc_stop as u32, bpn) % bpi) as i32;

        let umin = |a: u32, b: u32| if a < b { a } else { b };
        let mut kb0_start = fmod(kbc as u32, bpn) as i32;
        let mut kb0_stop = umin(bpn.d, (kb0_start.wrapping_add(kbc_stop).wrapping_sub(kbc)) as u32) as i32;
        while kbc < kbc_stop && kb0_stop == bpn.d as i32 {
            let (it, wt, zt, jt) = tile_of(kbc, bpn, ntx, ncy, nsy);
            let mut col_low = 0;
            let mut col_diff = ncols_dst;
            let mut offset_y = wt.wrapping_mul(stride_sample_y).wrapping_add(zt.wrapping_mul(stride_channel_y));
            let mut offset_dst = wt
                .wrapping_mul(stride_sample_dst)
                .wrapping_add(zt.wrapping_mul(stride_channel_dst))
                .wrapping_add(jt.wrapping_mul(J).wrapping_mul(stride_col_dst));
            let mut offset_ys = wt.wrapping_mul(ncy.d as i32).wrapping_mul(ncols_y).wrapping_add(zt.wrapping_mul(ncols_y));
            if !ids_dst.is_null() {
                col_low = *expert_bounds.offset(zt as isize);
                let col_high = *expert_bounds.offset(zt as isize + 1);
                col_diff = col_high.wrapping_sub(col_low);
                offset_y = 0;
                offset_dst = 0;
                offset_ys = 0;
                if jt.wrapping_mul(J) >= col_diff {
                    kbc = kbc.wrapping_add(bpn.d as i32);
                    kbc = (kbc as u32).wrapping_sub(fmod(kbc as u32, bpn)) as i32;
                    kb0_start = 0;
                    kb0_stop = umin(bpn.d, kbc_stop.wrapping_sub(kbc) as u32) as i32;
                    continue;
                }
                thread::sync_threads();
                let base = col_low.wrapping_add(jt.wrapping_mul(J));
                fill_ids::<J, _>(|j| *ids_dst.offset(base.wrapping_add(j) as isize));
                thread::sync_threads();
            }
            offset_y = offset_y.wrapping_add(col_low.wrapping_add(jt.wrapping_mul(J)).wrapping_mul(SZ));
            offset_dst = offset_dst.wrapping_add(it.wrapping_mul(MMQ_I));
            offset_ys = offset_ys.wrapping_add(col_low.wrapping_add(jt.wrapping_mul(J)));
            let tile_x_max_i = nrows_x.wrapping_sub(it.wrapping_mul(MMQ_I)).wrapping_sub(1);
            let tile_y_max_j = col_diff.wrapping_sub(jt.wrapping_mul(J)).wrapping_sub(1);
            let offset_x = fdiv(wt as u32, sample_ratio)
                .wrapping_mul(stride_sample_x as u32)
                .wrapping_add(fdiv(zt as u32, channel_ratio).wrapping_mul(stride_channel_x as u32))
                .wrapping_add(it.wrapping_mul(MMQ_I).wrapping_mul(stride_row_x) as u32) as i32;
            process_tile::<F, J, FB, false>(
                x, offset_x, y.wrapping_offset(offset_y as isize), dst.wrapping_offset(offset_dst as isize), tmp_fixup,
                y_scale_tile::<F>(y_scale, offset_ys), stride_row_x, ncols_y, stride_col_dst, tile_x_max_i, tile_y_max_j,
                kb0_start, kb0_stop,
            );
            kbc = kbc.wrapping_add(bpn.d as i32);
            kbc = (kbc as u32).wrapping_sub(fmod(kbc as u32, bpn)) as i32;
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
        let mut offset_dst = wt
            .wrapping_mul(stride_sample_dst)
            .wrapping_add(zt.wrapping_mul(stride_channel_dst))
            .wrapping_add(jt.wrapping_mul(J).wrapping_mul(stride_col_dst));
        let mut offset_ys = wt.wrapping_mul(ncy.d as i32).wrapping_mul(ncols_y).wrapping_add(zt.wrapping_mul(ncols_y));
        if !ids_dst.is_null() {
            col_low = *expert_bounds.offset(zt as isize);
            let col_high = *expert_bounds.offset(zt as isize + 1);
            col_diff = col_high.wrapping_sub(col_low);
            offset_y = 0;
            offset_dst = 0;
            offset_ys = 0;
            if jt.wrapping_mul(J) >= col_diff {
                return;
            }
            // The fixup buffer is always contiguous: reset the ids.
            thread::sync_threads();
            fill_ids::<J, _>(|j| j);
            thread::sync_threads();
        }
        offset_y = offset_y.wrapping_add(col_low.wrapping_add(jt.wrapping_mul(J)).wrapping_mul(SZ));
        offset_dst = offset_dst.wrapping_add(it.wrapping_mul(MMQ_I));
        offset_ys = offset_ys.wrapping_add(col_low.wrapping_add(jt.wrapping_mul(J)));
        let tile_x_max_i = nrows_x.wrapping_sub(it.wrapping_mul(MMQ_I)).wrapping_sub(1);
        let tile_y_max_j = col_diff.wrapping_sub(jt.wrapping_mul(J)).wrapping_sub(1);
        let offset_x = fdiv(wt as u32, sample_ratio)
            .wrapping_mul(stride_sample_x as u32)
            .wrapping_add(fdiv(zt as u32, channel_ratio).wrapping_mul(stride_channel_x as u32))
            .wrapping_add(it.wrapping_mul(MMQ_I).wrapping_mul(stride_row_x) as u32) as i32;
        process_tile::<F, J, FB, true>(
            x, offset_x, y.wrapping_offset(offset_y as isize), dst.wrapping_offset(offset_dst as isize), tmp_fixup,
            y_scale_tile::<F>(y_scale, offset_ys), stride_row_x, ncols_y, stride_col_dst, tile_x_max_i, tile_y_max_j,
            kb0_start, kb0_stop,
        );
    }

    /// mul_mat_q_stream_k_fixup<type, J, fallback>: grid (nblocks_sk, I / 32), block (32, 4);
    /// thread (x, y) owns output row `blockIdx.y*32 + x` of the columns `j0 + y`.
    #[inline(always)]
    pub unsafe fn stream_k_fixup<F: Fmt, const J: i32, const FB: bool>(
        ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *const f32, bpn: Fd, nrows_x: i32,
        ncols_dst: i32, stride_col_dst: i32, ncy: Fd, stride_channel_dst: i32, nsy: Fd, stride_sample_dst: i32, ntx: Fd,
    ) {
        static mut IDS: SharedArray<i32, 128> = SharedArray::UNINIT;
        const FW: i32 = 4; // nwarps of the fixup kernel
        let bpi = F::BPI as u32;
        let mut sum = [0f32; 32];
        let i = (thread::blockIdx_y() as i32) * WARP + tx();
        let nty = nrows_x.wrapping_add(MMQ_I - 1) / MMQ_I;
        let total = nsy.d.wrapping_mul(ncy.d).wrapping_mul(ntx.d).wrapping_mul(nty as u32).wrapping_mul(bpn.d);
        let gdim = thread::gridDim_x() as i64;
        let bidx0 = thread::blockIdx_x() as i32;
        let mut kbc0 = ((bidx0 as i64).wrapping_mul(total as i64) / gdim) as i32;
        let mut kbc0_stop = ((bidx0 as i64 + 1).wrapping_mul(total as i64) / gdim) as i32;
        kbc0 = (kbc0 as u32).wrapping_sub(fmod(kbc0 as u32, bpn) % bpi) as i32;
        kbc0_stop = (kbc0_stop as u32).wrapping_sub(fmod(kbc0_stop as u32, bpn) % bpi) as i32;

        let did_not_have_any_data = kbc0 == kbc0_stop;
        let wrote_beginning_of_tile = fmod(kbc0 as u32, bpn) == 0;
        let did_not_write_last = fdiv(kbc0 as u32, bpn) == fdiv(kbc0_stop as u32, bpn) && fmod(kbc0_stop as u32, bpn) != 0;
        if did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last {
            return;
        }

        let mut any_fixup = false;
        let mut bidx = bidx0 - 1;
        let mut kbc_stop = kbc0;
        loop {
            let mut kbc = ((bidx as i64).wrapping_mul(total as i64) / gdim) as i32;
            kbc = (kbc as u32).wrapping_sub(fmod(kbc as u32, bpn) % bpi) as i32;
            if kbc == kbc_stop {
                bidx -= 1;
                kbc_stop = kbc;
                continue;
            }
            any_fixup = true;
            let base = tmp_last_tile.wrapping_offset(bidx.wrapping_mul(J * MMQ_I) as isize);
            let mut j0 = 0;
            while j0 < J {
                unroll!();
                let j = j0 + ty();
                let s = (j0 / FW) as usize;
                sum[s] = add(*base.offset((j * MMQ_I + i) as isize), sum[s]);
                j0 += FW;
            }
            if fmod(kbc as u32, bpn) == 0 || fdiv(kbc as u32, bpn) < fdiv(kbc0 as u32, bpn) {
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
            let offset_dst = wt
                .wrapping_mul(stride_sample_dst)
                .wrapping_add(zt.wrapping_mul(stride_channel_dst))
                .wrapping_add(jt.wrapping_mul(J).wrapping_mul(stride_col_dst))
                .wrapping_add(it.wrapping_mul(MMQ_I));
            let dst = dst.wrapping_offset(offset_dst as isize);
            let i_max = nrows_x.wrapping_sub(it.wrapping_mul(MMQ_I)).wrapping_sub(1);
            let j_max = ncols_dst.wrapping_sub(jt.wrapping_mul(J)).wrapping_sub(1);
            if FB && i > i_max {
                return;
            }
            let mut j0 = 0;
            while j0 < J {
                unroll!();
                let j = j0 + ty();
                if j > j_max {
                    return;
                }
                let p = dst.wrapping_offset(j.wrapping_mul(stride_col_dst).wrapping_add(i) as isize);
                *p = add(sum[(j0 / FW) as usize], *p);
                j0 += FW;
            }
            return;
        }

        let ids = SharedArray::as_raw_mut_ptr(&raw mut IDS);
        let col_low = *expert_bounds.offset(zt as isize);
        let col_high = *expert_bounds.offset(zt as isize + 1);
        let col_diff = col_high.wrapping_sub(col_low);
        let mut j = ty() * WARP + tx();
        while j < J {
            *ids.offset(j as isize) = *ids_dst.offset(col_low.wrapping_add(jt.wrapping_mul(J)).wrapping_add(j) as isize);
            j += FW * WARP;
        }
        thread::sync_threads();
        let dst = dst.wrapping_offset(it.wrapping_mul(MMQ_I) as isize);
        let i_max = nrows_x.wrapping_sub(it.wrapping_mul(MMQ_I)).wrapping_sub(1);
        let j_max = col_diff.wrapping_sub(jt.wrapping_mul(J)).wrapping_sub(1);
        if FB && i > i_max {
            return;
        }
        let mut j0 = 0;
        while j0 < J {
            unroll!();
            let j = j0 + ty();
            if j > j_max {
                return;
            }
            let p = dst.wrapping_offset((*ids.offset(j as isize)).wrapping_mul(stride_col_dst).wrapping_add(i) as isize);
            *p = add(sum[(j0 / FW) as usize], *p);
            j0 += FW;
        }
    }

    // One entry per reference instance (gen_kernels.py). mul_mat_q params: x, y, ids_dst,
    // expert_bounds, dst, tmp_fixup, y_scale, blocks_per_ne00 (uint3), nrows_x, ncols_dst,
    // stride_row_x, ncols_y, stride_col_dst, channel_ratio, nchannels_y (uint3),
    // stride_channel_{x,y,dst}, sample_ratio, nsamples_y (uint3), stride_sample_{x,y,dst}, ntx.
    // GENERATED KERNELS BEGIN
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn iq4_nl_mmq_j8_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Iq4Nl, 8, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn iq4_nl_mmq_j16_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Iq4Nl, 16, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn iq4_nl_mmq_j24_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Iq4Nl, 24, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn iq4_nl_mmq_j32_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Iq4Nl, 32, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn iq4_nl_mmq_j40_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Iq4Nl, 40, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn iq4_nl_mmq_j48_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Iq4Nl, 48, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn iq4_nl_mmq_j64_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Iq4Nl, 64, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn iq4_nl_mmq_j80_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Iq4Nl, 80, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn iq4_nl_mmq_j96_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Iq4Nl, 96, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn iq4_nl_mmq_j112_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Iq4Nl, 112, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn iq4_nl_mmq_j128_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Iq4Nl, 128, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn iq4_nl_mmq_j8_f1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Iq4Nl, 8, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn iq4_nl_mmq_j16_f1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Iq4Nl, 16, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn iq4_nl_mmq_j32_f1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Iq4Nl, 32, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn iq4_nl_mmq_j64_f1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Iq4Nl, 64, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn iq4_nl_mmq_j128_f1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Iq4Nl, 128, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn iq4_nl_mmq_fixup_j8_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Iq4Nl, 8, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn iq4_nl_mmq_fixup_j16_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Iq4Nl, 16, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn iq4_nl_mmq_fixup_j24_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Iq4Nl, 24, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn iq4_nl_mmq_fixup_j32_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Iq4Nl, 32, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn iq4_nl_mmq_fixup_j40_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Iq4Nl, 40, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn iq4_nl_mmq_fixup_j48_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Iq4Nl, 48, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn iq4_nl_mmq_fixup_j64_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Iq4Nl, 64, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn iq4_nl_mmq_fixup_j80_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Iq4Nl, 80, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn iq4_nl_mmq_fixup_j96_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Iq4Nl, 96, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn iq4_nl_mmq_fixup_j112_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Iq4Nl, 112, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn iq4_nl_mmq_fixup_j128_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Iq4Nl, 128, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn iq4_nl_mmq_fixup_j8_f1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Iq4Nl, 8, true>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn iq4_nl_mmq_fixup_j16_f1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Iq4Nl, 16, true>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn iq4_nl_mmq_fixup_j32_f1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Iq4Nl, 32, true>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn iq4_nl_mmq_fixup_j64_f1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Iq4Nl, 64, true>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn iq4_nl_mmq_fixup_j128_f1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Iq4Nl, 128, true>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mxfp4_mmq_j8_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Mxfp4, 8, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mxfp4_mmq_j16_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Mxfp4, 16, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mxfp4_mmq_j24_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Mxfp4, 24, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mxfp4_mmq_j32_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Mxfp4, 32, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mxfp4_mmq_j40_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Mxfp4, 40, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mxfp4_mmq_j48_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Mxfp4, 48, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mxfp4_mmq_j64_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Mxfp4, 64, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mxfp4_mmq_j80_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Mxfp4, 80, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mxfp4_mmq_j96_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Mxfp4, 96, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mxfp4_mmq_j112_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Mxfp4, 112, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mxfp4_mmq_j128_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Mxfp4, 128, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mxfp4_mmq_j8_f1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Mxfp4, 8, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mxfp4_mmq_j16_f1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Mxfp4, 16, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mxfp4_mmq_j32_f1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Mxfp4, 32, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mxfp4_mmq_j64_f1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Mxfp4, 64, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mxfp4_mmq_j128_f1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Mxfp4, 128, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn mxfp4_mmq_fixup_j8_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Mxfp4, 8, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn mxfp4_mmq_fixup_j16_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Mxfp4, 16, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn mxfp4_mmq_fixup_j24_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Mxfp4, 24, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn mxfp4_mmq_fixup_j32_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Mxfp4, 32, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn mxfp4_mmq_fixup_j40_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Mxfp4, 40, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn mxfp4_mmq_fixup_j48_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Mxfp4, 48, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn mxfp4_mmq_fixup_j64_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Mxfp4, 64, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn mxfp4_mmq_fixup_j80_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Mxfp4, 80, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn mxfp4_mmq_fixup_j96_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Mxfp4, 96, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn mxfp4_mmq_fixup_j112_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Mxfp4, 112, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn mxfp4_mmq_fixup_j128_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Mxfp4, 128, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn mxfp4_mmq_fixup_j8_f1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Mxfp4, 8, true>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn mxfp4_mmq_fixup_j16_f1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Mxfp4, 16, true>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn mxfp4_mmq_fixup_j32_f1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Mxfp4, 32, true>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn mxfp4_mmq_fixup_j64_f1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Mxfp4, 64, true>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn mxfp4_mmq_fixup_j128_f1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Mxfp4, 128, true>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn nvfp4_mmq_j8_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Nvfp4, 8, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn nvfp4_mmq_j16_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Nvfp4, 16, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn nvfp4_mmq_j24_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Nvfp4, 24, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn nvfp4_mmq_j32_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Nvfp4, 32, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn nvfp4_mmq_j40_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Nvfp4, 40, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn nvfp4_mmq_j48_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Nvfp4, 48, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn nvfp4_mmq_j64_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Nvfp4, 64, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn nvfp4_mmq_j80_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Nvfp4, 80, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn nvfp4_mmq_j96_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Nvfp4, 96, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn nvfp4_mmq_j112_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Nvfp4, 112, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn nvfp4_mmq_j128_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Nvfp4, 128, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn nvfp4_mmq_j8_f1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Nvfp4, 8, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn nvfp4_mmq_j16_f1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Nvfp4, 16, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn nvfp4_mmq_j32_f1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Nvfp4, 32, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn nvfp4_mmq_j64_f1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Nvfp4, 64, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn nvfp4_mmq_j128_f1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<Nvfp4, 128, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn nvfp4_mmq_fixup_j8_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Nvfp4, 8, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn nvfp4_mmq_fixup_j16_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Nvfp4, 16, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn nvfp4_mmq_fixup_j24_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Nvfp4, 24, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn nvfp4_mmq_fixup_j32_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Nvfp4, 32, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn nvfp4_mmq_fixup_j40_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Nvfp4, 40, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn nvfp4_mmq_fixup_j48_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Nvfp4, 48, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn nvfp4_mmq_fixup_j64_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Nvfp4, 64, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn nvfp4_mmq_fixup_j80_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Nvfp4, 80, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn nvfp4_mmq_fixup_j96_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Nvfp4, 96, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn nvfp4_mmq_fixup_j112_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Nvfp4, 112, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn nvfp4_mmq_fixup_j128_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Nvfp4, 128, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn nvfp4_mmq_fixup_j8_f1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Nvfp4, 8, true>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn nvfp4_mmq_fixup_j16_f1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Nvfp4, 16, true>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn nvfp4_mmq_fixup_j32_f1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Nvfp4, 32, true>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn nvfp4_mmq_fixup_j64_f1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Nvfp4, 64, true>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn nvfp4_mmq_fixup_j128_f1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<Nvfp4, 128, true>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    // GENERATED KERNELS END
}

fn main() {
    std::process::exit(if gate::run() { 0 } else { 1 });
}
