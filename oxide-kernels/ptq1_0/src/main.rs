//! PrismML PTQ1_0 (GGML type 143) CUDA kernels in cuda-oxide for titan-mistral, ported from
//! sudoingX/llama.cpp `bonsai2` v1.1 (ff41412: the PTQ1_0 mat-vec of PrismML-Eng/llama.cpp#218, PrismML's
//! PTQ1_0 MMQ tile loader, the prism Hadamard FWHT) under the reference's own parameter ABI, bit-identical to
//! its nvcc build (nvcc 13.3, g++-15, -O3 -use_fast_math, sm_120a; `ref/nvcc_ref.py` rebuilds the reference TUs
//! from the reference build's compile_commands.json):
//!
//! - `ptq1_0_quantize_pt_{f32,bf16}` = `quantize_q8_1<pt = true>` (quantize.cu; bf16 input widened exactly);
//! - `ptq1_0_fwht_{f32,bf16}_m{0,1,2}` = `fwht_cuda_block<1024, 256, float, has_signs>` (fwht.cu): m0 no signs,
//!   m1 signs before the transform (the fused MUL + FWHT of a folded weight's activation), m2 signs after it
//!   (the inverse on a latent embedding row, `ggml_mul(fwht(z), signs)`);
//! - `ptq1_0_fwht_q8pt_{f32,bf16}_m{0,1}`: fwht_cuda_block + quantize_q8_1<pt> in one launch (same values);
//! - `ptq1_0_mmvq_pt_c<n>[_bf16]` = `mul_mat_vec_ptq1_0_pt<n, 4, false, false>` (n = 1..4; `_bf16` rounds the
//!   f32 result once), `ptq1_0_mmvq_pt_glu_c<n>` = `mul_mat_vec_ptq1_0_pt<n, 4, true, true>` (gate + up + GLU);
//! - `ptq1_0_quantize_mmq_d4_{f32,bf16}` = `quantize_mmq_q8_1<D4, false, false, false>`;
//! - `ptq1_0_mmq_j<J>_f<fb>` / `ptq1_0_mmq_fixup_j<J>_f<fb>` = `mul_mat_q<PTQ1_0, J, fb>` (mmq-config-ampere.cuh,
//!   which sm_120 falls back to: 256 threads, I = 128, SRAM layout Q8_0, K_vram 256, stream-k) and
//!   `mul_mat_q_stream_k_fixup`: the q1_0 crate's MMQ core (acecd56 = bonsai2 off GB10) with PTQ1_0's loader.
//!
//! `export_ptx.py` splits ptq1_0.ptx into the modules mistral.rs embeds.
#![allow(non_snake_case, clippy::missing_safety_doc, clippy::too_many_arguments)]
mod gate;

use cuda_device::{DynamicSharedArray, SharedArray, convert, device, dotprod, kernel, launch_bounds, ptx_asm, thread, warp};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

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
    /// `__byte_perm(a, b, s)`: prmt with the selector masked to its 3-bit byte indices.
    #[inline(always)]
    pub fn byte_perm(a: u32, b: u32, s: u32) -> u32 {
        let r: u32;
        let s = s & 0x7777;
        unsafe { ptx_asm!("prmt.b32 %0, %1, %2, %3;", out("=r") r, in("r") a, in("r") b, in("r") s, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn dp4a(a: u32, b: u32, c: i32) -> i32 {
        dotprod::dp4a_s32(a, b, c)
    }
    /// llama.cpp fastdiv: `(__umulhi(n, mp) + n) >> L`.
    #[inline(always)]
    pub fn fastdiv(n: u32, mp: u32, l: u32) -> u32 {
        let hi = ((n as u64 * mp as u64) >> 32) as u32;
        hi.wrapping_add(n) >> l
    }

    /// Output element types.
    pub trait Dst: Copy {
        unsafe fn st(p: *mut Self, v: f32);
    }
    impl Dst for f32 {
        #[inline(always)]
        unsafe fn st(p: *mut f32, v: f32) {
            *p = v;
        }
    }
    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct H(pub u16);
    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct B(pub u16);
    impl Dst for H {
        #[inline(always)]
        unsafe fn st(p: *mut H, v: f32) {
            *(p as *mut u16) = f2h(v);
        }
    }
    impl Dst for B {
        #[inline(always)]
        unsafe fn st(p: *mut B, v: f32) {
            *(p as *mut u16) = f2bf(v);
        }
    }

    macro_rules! unroll {
        () => {
            cuda_device::thread::__unroll_config::<0>();
        };
    }

    // ------------------------------------------------------------------------------------------
    // Activation quantizers (llama.cpp quantize.cu, -use_fast_math): the f32 entries are the
    // reference instances; the f16 / bf16 entries read half inputs and convert them exactly
    // (bf16: 16-bit shift, f16: cvt.f32.f16), then run the same f32 code.

    /// Input element types for the quantizers.
    pub trait Src: Copy {
        unsafe fn ld(p: *const Self) -> f32;
    }
    impl Src for f32 {
        #[inline(always)]
        unsafe fn ld(p: *const f32) -> f32 {
            *p
        }
    }
    impl Src for H {
        #[inline(always)]
        unsafe fn ld(p: *const H) -> f32 {
            h2f(*(p as *const u16) as u32)
        }
    }
    impl Src for B {
        #[inline(always)]
        unsafe fn ld(p: *const B) -> f32 {
            f32::from_bits((*(p as *const u16) as u32) << 16)
        }
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
    /// `(int) roundf(x)` under fast-math: copysign(0.5, x) + add.rz.ftz + cvt.rzi + cvt.rzi.ftz.s32.
    #[inline(always)]
    pub fn roundi(x: f32) -> i32 {
        let r: i32;
        unsafe {
            ptx_asm!(
                "{ .reg .f32 h, s, t; mov.b32 h, 0f3F000000; copysign.f32 h, %1, h; add.rz.ftz.f32 s, %1, h; cvt.rzi.f32.f32 t, s; cvt.rzi.ftz.s32.f32 %0, t; }",
                out("=r") r, in("f") x, options(register_only)
            );
        }
        r
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

    /// `-x` [neg.ftz.f32] selected by `flip`, then `+ other` [add.ftz.f32]: one butterfly half as nvcc
    /// emits `(bit == 0) ? val + other : other - val` (fwht.cu).
    #[inline(always)]
    pub fn bfly(val: f32, other: f32, flip: bool) -> f32 {
        let r: f32;
        let f = flip as u32;
        unsafe {
            ptx_asm!(
                "{ .reg .f32 n, s; .reg .pred p; setp.ne.b32 p, %3, 0; neg.ftz.f32 n, %1; selp.f32 s, n, %1, p; add.ftz.f32 %0, s, %2; }",
                out("=f") r, in("f") val, in("f") other, in("r") f, options(register_only)
            );
        }
        r
    }
    /// fast-math `a + b` / `a - b` as the reference PTX writes them (no .rn: ptxas treats them alike).
    #[inline(always)]
    pub fn addf(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("add.ftz.f32 %0, %1, %2; //\t\x0b\x0c\r\n", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn subf(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("sub.ftz.f32 %0, %1, %2; //\t\x0b\x0c\r\n", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    /// fast-math `a * b` without .rn (the reference's scale / sign multiplies).
    #[inline(always)]
    pub fn mulf(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("mul.ftz.f32 %0, %1, %2; //\t\x0b\x0c\r\n", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }

    // ------------------------------------------------------------------------------------------
    // PT activation quantizer: `quantize_q8_1<pt = true>` (quantize.cu). The q8_1 float program of
    // quantize_q8_1 (abs/max.ftz and add.ftz butterflies over the 32-value warp, div.approx.ftz for
    // amax / 127 and x / d, fast-math roundf), stored in the planar-transposed layout of
    // mmvq-ptq1_0.cuh: per column (stride ne0 * 9 / 8 bytes) plane t = 0..7 holds byte e % 16 of
    // element kb * 128 + 16 * t + e % 16 at ((t * nblk + kb) * 16 + e % 16), plane 8 the half2 (d, sum)
    // of each 32-value sub-block at (ne0 & ~127) + kb * 16 + (e / 32) * 4.

    /// Store one quantized value (and, for the first lane of its 32-value group, the (d, sum) pair).
    #[inline(always)]
    pub unsafe fn pt_store(vy: *mut u8, row_cont: i64, ne0: i64, i0: i64, q: i32, d: f32, sum: f32) {
        let ycol = vy.offset(row_cont.wrapping_mul(ne0.wrapping_add(((ne0 as u64) >> 3) as i64)) as isize);
        let nblk = ((ne0 as u64) >> 7) as i64;
        let kb = ((i0 as u64) >> 7) as i64;
        let e = i0 & 127;
        let off = (((e >> 4).wrapping_mul(nblk).wrapping_add(kb)) << 4) | (i0 & 15);
        *ycol.offset(off as isize) = q as u8;
        if (row_cont.wrapping_mul(ne0).wrapping_add(i0) & 31) != 0 {
            return;
        }
        let ds = ycol.offset(((ne0 & !127) + (kb << 4) + ((e >> 5) << 2)) as isize) as *mut u32;
        *ds = (f2h(d) as u32) | ((f2h(sum) as u32) << 16);
    }

    /// Params: x, vy, ne00, s01, s02, s03, ne0, ne1, ne2 (uint3); grid (ceil(ne0/256), ne1, ne2*ne3), block 256.
    #[inline(always)]
    pub unsafe fn quantize_pt<S: Src>(
        x: *const S, vy: *mut u8, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: u32, ne2_mp: u32, ne2_l: u32, ne2_d: u32,
    ) {
        let i0 = ((thread::blockDim_x() as u64) * (thread::blockIdx_x() as u64)).wrapping_add(thread::threadIdx_x() as u64) as i64;
        if i0 >= ne0 {
            return;
        }
        let z = thread::blockIdx_z();
        let i3 = fastdiv(z, ne2_mp, ne2_l);
        let i2 = (z as i64).wrapping_sub((i3 as u64 * ne2_d as u64) as i64);
        let i3 = i3 as i64;
        let i1 = thread::blockIdx_y() as i64;
        let xi = if i0 < ne00 {
            let ix = s01.wrapping_mul(i1).wrapping_add(i0).wrapping_add(s03.wrapping_mul(i3)).wrapping_add(i2.wrapping_mul(s02));
            S::ld(x.offset(ix as isize))
        } else {
            0.0
        };
        let mut amax = absf(xi);
        let mut k = 0;
        while k < 5 {
            unroll!();
            amax = fmax(amax, warp::shuffle_xor_f32_sync(0xffff_ffff, amax, 16 >> k));
            k += 1;
        }
        let mut sum = xi;
        let mut k = 0;
        while k < 5 {
            unroll!();
            sum = add(sum, warp::shuffle_xor_f32_sync(0xffff_ffff, sum, 16 >> k));
            k += 1;
        }
        let d = div_approx(amax, 127.0);
        let q = if amax == 0.0 { 0 } else { roundi(div_approx(xi, d)) };
        let row_cont = ((ne1 as u64 * z as u64) as i64).wrapping_add(i1);
        pt_store(vy, row_cont, ne0, i0, q, d, sum);
    }

    #[kernel]
    pub unsafe fn ptq1_0_quantize_pt_f32(x: *const f32, vy: *mut u8, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: u32, ne2_mp: u32, ne2_l: u32, ne2_d: u32) {
        quantize_pt(x, vy, ne00, s01, s02, s03, ne0, ne1, ne2_mp, ne2_l, ne2_d)
    }
    #[kernel]
    pub unsafe fn ptq1_0_quantize_pt_bf16(x: *const u16, vy: *mut u8, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: u32, ne2_mp: u32, ne2_l: u32, ne2_d: u32) {
        quantize_pt(x as *const B, vy, ne00, s01, s02, s03, ne0, ne1, ne2_mp, ne2_l, ne2_d)
    }

    // ------------------------------------------------------------------------------------------
    // Hadamard: `fwht_cuda_block<1024, 256, float, has_signs>` (fwht.cu), one 1024-value row per block
    // of 256 threads; thread t holds values t + 256 j (j < 4). Load `mul.ftz(scale, x)` (then
    // `mul.ftz(., sign)` with pre-signs, the fused MUL + FWHT of ggml_cuda_op_fwht_signed), butterflies
    // over lanes (shuffles), over warps (shared memory) and over the 4 registers; `(bit == 0) ? v + o :
    // o - v` as neg.ftz + selp + add.ftz. MODE 0: no signs; 1: signs before the transform (activation
    // side of a folded weight); 2: signs after it (`ggml_mul(fwht(z), signs)`, the inverse applied to a
    // latent embedding row; a +-1 multiply is exact, so fusing it into the store changes no bit).

    /// The transform of this thread's 4 values of row `r` (in registers) through all 10 stages.
    #[inline(always)]
    pub unsafe fn fwht_1024(reg: &mut [f32; 4]) {
        static mut S: SharedArray<f32, 1024> = SharedArray::UNINIT;
        let s = SharedArray::as_raw_mut_ptr(&raw mut S);
        let tid = thread::threadIdx_x() as i32;
        let lane = tid & 31;
        let mut k = 0;
        while k < 5 {
            unroll!();
            let h = 1i32 << k;
            let mut j = 0;
            while j < 4 {
                unroll!();
                let o = warp::shuffle_xor_f32_sync(0xffff_ffff, reg[j], h as u32);
                reg[j] = bfly(reg[j], o, (lane & h) != 0);
                j += 1;
            }
            k += 1;
        }
        let mut k = 5;
        while k < 8 {
            unroll!();
            let h = 1i32 << k;
            let mut j = 0;
            while j < 4 {
                unroll!();
                *s.add(j * 256 + tid as usize) = reg[j];
                j += 1;
            }
            thread::sync_threads();
            let mut j = 0;
            while j < 4 {
                unroll!();
                let o = *s.add(j * 256 + (tid ^ h) as usize);
                reg[j] = bfly(reg[j], o, (tid & h) != 0);
                j += 1;
            }
            thread::sync_threads();
            k += 1;
        }
        // h = 256 (step 1), h = 512 (step 2)
        let (a0, a1, a2, a3) = (addf(reg[0], reg[1]), subf(reg[0], reg[1]), addf(reg[2], reg[3]), subf(reg[2], reg[3]));
        reg[0] = addf(a0, a2);
        reg[2] = subf(a0, a2);
        reg[1] = addf(a1, a3);
        reg[3] = subf(a1, a3);
    }

    /// Row r's 4 values of this thread after load + transform.
    #[inline(always)]
    pub unsafe fn fwht_row<S: Src, const MODE: u32>(src: *const S, r: i64, scale: f32, signs: *const f32, n_blk: i32) -> [f32; 4] {
        let tid = thread::threadIdx_x() as usize;
        let src = src.offset(r.wrapping_mul(1024) as isize);
        let srow = signs.wrapping_offset(((r as i32) % n_blk.max(1)) as isize * 1024);
        let mut reg = [0f32; 4];
        let mut j = 0;
        while j < 4 {
            unroll!();
            let mut v = mulf(scale, S::ld(src.add(j * 256 + tid)));
            if MODE == 1 {
                v = mulf(v, *srow.add(j * 256 + tid));
            }
            reg[j] = v;
            j += 1;
        }
        fwht_1024(&mut reg);
        reg
    }

    /// Params: src, dst (f32), n_rows (i64, rows of 1024), scale, signs, n_blk; grid n_rows, block 256.
    #[inline(always)]
    pub unsafe fn fwht<S: Src, const MODE: u32>(src: *const S, dst: *mut f32, n_rows: i64, scale: f32, signs: *const f32, n_blk: i32) {
        let r = thread::blockIdx_x() as i64;
        if r >= n_rows {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let reg = fwht_row::<S, MODE>(src, r, scale, signs, n_blk);
        let dst = dst.offset(r.wrapping_mul(1024) as isize);
        let srow = signs.wrapping_offset(((r as i32) % n_blk.max(1)) as isize * 1024);
        let mut j = 0;
        while j < 4 {
            unroll!();
            let v = if MODE == 2 { mulf(reg[j], *srow.add(j * 256 + tid)) } else { reg[j] };
            *dst.add(j * 256 + tid) = v;
            j += 1;
        }
    }

    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn ptq1_0_fwht_f32_m0(src: *const f32, dst: *mut f32, n_rows: i64, scale: f32, signs: *const f32, n_blk: i32) {
        fwht::<f32, 0>(src, dst, n_rows, scale, signs, n_blk)
    }
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn ptq1_0_fwht_f32_m1(src: *const f32, dst: *mut f32, n_rows: i64, scale: f32, signs: *const f32, n_blk: i32) {
        fwht::<f32, 1>(src, dst, n_rows, scale, signs, n_blk)
    }
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn ptq1_0_fwht_f32_m2(src: *const f32, dst: *mut f32, n_rows: i64, scale: f32, signs: *const f32, n_blk: i32) {
        fwht::<f32, 2>(src, dst, n_rows, scale, signs, n_blk)
    }
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn ptq1_0_fwht_bf16_m0(src: *const u16, dst: *mut f32, n_rows: i64, scale: f32, signs: *const f32, n_blk: i32) {
        fwht::<B, 0>(src as *const B, dst, n_rows, scale, signs, n_blk)
    }
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn ptq1_0_fwht_bf16_m1(src: *const u16, dst: *mut f32, n_rows: i64, scale: f32, signs: *const f32, n_blk: i32) {
        fwht::<B, 1>(src as *const B, dst, n_rows, scale, signs, n_blk)
    }
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn ptq1_0_fwht_bf16_m2(src: *const u16, dst: *mut f32, n_rows: i64, scale: f32, signs: *const f32, n_blk: i32) {
        fwht::<B, 2>(src as *const B, dst, n_rows, scale, signs, n_blk)
    }

    /// Fused `fwht_cuda_block` + `quantize_q8_1<pt = true>`: the transform of row r (block b = r % n_blk of
    /// token t = r / n_blk) quantized straight into token t's PT column (ne0 = n_blk * 1024 values, a
    /// multiple of 512, so no padding). Every value's float program is the two reference kernels':
    /// the 32-value group of a quantization block is one warp's lanes for one register j.
    /// Params: src, vy, n_rows, scale, signs, n_blk; grid n_rows, block 256.
    #[inline(always)]
    pub unsafe fn fwht_q8pt<S: Src, const MODE: u32>(src: *const S, vy: *mut u8, n_rows: i64, scale: f32, signs: *const f32, n_blk: i32) {
        let r = thread::blockIdx_x() as i64;
        if r >= n_rows {
            return;
        }
        let tid = thread::threadIdx_x() as i64;
        let reg = fwht_row::<S, MODE>(src, r, scale, signs, n_blk);
        let nb = n_blk as i64;
        let t = r / nb;
        let b = r - t * nb;
        let ne0 = nb * 1024;
        let mut j = 0;
        while j < 4 {
            unroll!();
            let xi = reg[j];
            let mut amax = absf(xi);
            let mut k = 0;
            while k < 5 {
                unroll!();
                amax = fmax(amax, warp::shuffle_xor_f32_sync(0xffff_ffff, amax, 16 >> k));
                k += 1;
            }
            let mut sum = xi;
            let mut k = 0;
            while k < 5 {
                unroll!();
                sum = add(sum, warp::shuffle_xor_f32_sync(0xffff_ffff, sum, 16 >> k));
                k += 1;
            }
            let d = div_approx(amax, 127.0);
            let q = if amax == 0.0 { 0 } else { roundi(div_approx(xi, d)) };
            pt_store(vy, t, ne0, b * 1024 + j as i64 * 256 + tid, q, d, sum);
            j += 1;
        }
    }

    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn ptq1_0_fwht_q8pt_f32_m0(src: *const f32, vy: *mut u8, n_rows: i64, scale: f32, signs: *const f32, n_blk: i32) {
        fwht_q8pt::<f32, 0>(src, vy, n_rows, scale, signs, n_blk)
    }
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn ptq1_0_fwht_q8pt_f32_m1(src: *const f32, vy: *mut u8, n_rows: i64, scale: f32, signs: *const f32, n_blk: i32) {
        fwht_q8pt::<f32, 1>(src, vy, n_rows, scale, signs, n_blk)
    }
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn ptq1_0_fwht_q8pt_bf16_m0(src: *const u16, vy: *mut u8, n_rows: i64, scale: f32, signs: *const f32, n_blk: i32) {
        fwht_q8pt::<B, 0>(src as *const B, vy, n_rows, scale, signs, n_blk)
    }
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn ptq1_0_fwht_q8pt_bf16_m1(src: *const u16, vy: *mut u8, n_rows: i64, scale: f32, signs: *const f32, n_blk: i32) {
        fwht_q8pt::<B, 1>(src as *const B, vy, n_rows, scale, signs, n_blk)
    }

    // ------------------------------------------------------------------------------------------
    // Decode / verify mat-vec: `mul_mat_vec_ptq1_0_pt<ncols, 4, has_fusion, has_gate>` (mmvq-ptq1_0.cuh,
    // sudoingX bonsai2 v1.1 = PrismML-Eng/llama.cpp#218). Work items (4-row group, K block) of a CTA of
    // `rows_per_cta` rows, 128 threads; per item and column: 5 + 5 + 2 trit steps of dp4a on the PT
    // activation planes, `acc = fma.rn.ftz(d8_k, (float) sumi_k, acc)` after each 32-value sub-block
    // (acc from +0), `mul.rn.ftz(d_w, acc)` to shared memory; then one warp per (row, column): lane-strided
    // add.ftz from +0 over the K blocks, butterfly xor 16..1. Per-column arithmetic is independent of
    // the column count (bit-identical rows for any batch 1..4). With the gate: the second weight's dots
    // in a second partials array, then `up * silu(gate)` (SWIGLU: g / (1 + ex2.approx(g * -log2 e)),
    // div.approx) or `up * gate` (other ops), as the reference's epilogue.

    /// Per-NC constants (unrolled loop bounds must be associated consts).
    pub struct Nc<const NC: usize>;
    impl<const NC: usize> Nc<NC> {
        pub const N: usize = NC;
    }

    #[inline(always)]
    unsafe fn ldv4(p: *const u8) -> [u32; 4] {
        let (a, b, c, d): (u32, u32, u32, u32);
        ptx_asm!("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];", out("=r") a, out("=r") b, out("=r") c, out("=r") d, in("l") p as u64, options(register_only));
        [a, b, c, d]
    }
    #[inline(always)]
    unsafe fn ld32(p: *const u8) -> u32 {
        let r: u32;
        ptx_asm!("ld.global.b32 %0, [%1];", out("=r") r, in("l") p as u64, options(register_only));
        r
    }
    #[inline(always)]
    unsafe fn ld16(p: *const u8) -> u32 {
        let r: u32;
        ptx_asm!("ld.global.u16 %0, [%1];", out("=r") r, in("l") p as u64, options(register_only));
        r
    }

    /// One base-3 digit of four bytes held as two 16-bit-lane words -> four weights (-1, 0, 1) as signed bytes.
    #[inline(always)]
    fn trit_step(vlo: &mut u32, vhi: &mut u32) -> u32 {
        let wlo = vlo.wrapping_mul(3);
        let whi = vhi.wrapping_mul(3);
        *vlo = wlo & 0x00FF00FF;
        *vhi = whi & 0x00FF00FF;
        byte_perm(wlo, whi, 0x7531).wrapping_add(0x7F7F7F7F) ^ 0x8080_8080
    }

    /// `acc[j][i] = fma(d8_k, (float) sumi[j][i], acc[j][i]); sumi = 0` for sub-block K.
    #[inline(always)]
    fn fold<const NC: usize, const K: usize>(acc: &mut [[f32; 4]; NC], sumi: &mut [[i32; 4]; NC], ds: &[[u32; 4]; NC]) {
        let mut j = 0;
        while j < Nc::<NC>::N {
            unroll!();
            let d8 = h2f(ds[j][K] & 0xFFFF);
            let mut i = 0;
            while i < 4 {
                unroll!();
                acc[j][i] = fma(d8, sumi[j][i] as f32, acc[j][i]);
                sumi[j][i] = 0;
                i += 1;
            }
            j += 1;
        }
    }

    /// ptq1_0_pt_block_dot<NC, 4>: the 4 rows' block `kbx` against the NC activation columns.
    #[inline(always)]
    unsafe fn pt_block_dot<const NC: usize>(bq: &[*const u8; 4], ycol: &[*const u8; NC], kbx: i32, nblk: i32, res: &mut [[f32; 4]; NC]) {
        let mut ds = [[0u32; 4]; NC];
        let mut j = 0;
        while j < Nc::<NC>::N {
            unroll!();
            ds[j] = ldv4(ycol[j].offset(((8 * nblk + kbx) as isize) * 16));
            j += 1;
        }
        let mut sumi = [[0i32; 4]; NC];
        let mut acc = [[0f32; 4]; NC];
        let mut vlo = [[0u32; 4]; 4];
        let mut vhi = [[0u32; 4]; 4];
        let mut i = 0;
        while i < 4 {
            unroll!();
            let mut g = 0;
            while g < 4 {
                unroll!();
                let packed = ld32(bq[i].add(4 * g));
                vlo[i][g] = byte_perm(packed, 0, 0x4140);
                vhi[i][g] = byte_perm(packed, 0, 0x4342);
                g += 1;
            }
            i += 1;
        }
        let mut t = 0;
        while t < 5 {
            unroll!();
            let mut u = [[0u32; 4]; NC];
            let mut j = 0;
            while j < Nc::<NC>::N {
                unroll!();
                u[j] = ldv4(ycol[j].offset(((t as i32 * nblk + kbx) as isize) * 16));
                j += 1;
            }
            let mut i = 0;
            while i < 4 {
                unroll!();
                let mut g = 0;
                while g < 4 {
                    unroll!();
                    let q = trit_step(&mut vlo[i][g], &mut vhi[i][g]);
                    let mut j = 0;
                    while j < Nc::<NC>::N {
                        unroll!();
                        sumi[j][i] = dp4a(q, u[j][g], sumi[j][i]);
                        j += 1;
                    }
                    g += 1;
                }
                i += 1;
            }
            if t == 1 {
                fold::<NC, 0>(&mut acc, &mut sumi, &ds);
            }
            if t == 3 {
                fold::<NC, 1>(&mut acc, &mut sumi, &ds);
            }
            t += 1;
        }
        // qs[16..23]: element 80 + 8 t + 4 g + b, words 20 + 2 t + g of planes 5, 6, 7
        let mut vlo2 = [[0u32; 2]; 4];
        let mut vhi2 = [[0u32; 2]; 4];
        let mut i = 0;
        while i < 4 {
            unroll!();
            let mut g = 0;
            while g < 2 {
                unroll!();
                let packed = ld32(bq[i].add(16 + 4 * g));
                vlo2[i][g] = byte_perm(packed, 0, 0x4140);
                vhi2[i][g] = byte_perm(packed, 0, 0x4342);
                g += 1;
            }
            i += 1;
        }
        let mut u2 = [[0u32; 4]; NC];
        let mut t = 0;
        while t < 5 {
            unroll!();
            if t % 2 == 0 {
                let mut j = 0;
                while j < Nc::<NC>::N {
                    unroll!();
                    u2[j] = ldv4(ycol[j].offset((((5 + t / 2) as i32 * nblk + kbx) as isize) * 16));
                    j += 1;
                }
            }
            let mut i = 0;
            while i < 4 {
                unroll!();
                let mut g = 0;
                while g < 2 {
                    unroll!();
                    let q = trit_step(&mut vlo2[i][g], &mut vhi2[i][g]);
                    let w = (20 + 2 * t + g) & 3;
                    let mut j = 0;
                    while j < Nc::<NC>::N {
                        unroll!();
                        sumi[j][i] = dp4a(q, u2[j][w], sumi[j][i]);
                        j += 1;
                    }
                    g += 1;
                }
                i += 1;
            }
            if t == 1 {
                fold::<NC, 2>(&mut acc, &mut sumi, &ds);
            }
            t += 1;
        }
        // qh: element 120 + 2 t + h, words 30, 31 (upper half of plane 7, still in u2)
        let mut i = 0;
        while i < 4 {
            unroll!();
            let h = ld16(bq[i].add(24));
            let mut v = (h & 0xFF) | ((h >> 8) << 16);
            let mut t = 0;
            while t < 2 {
                unroll!();
                let w0 = v.wrapping_mul(3);
                v = w0 & 0x00FF00FF;
                let w1 = v.wrapping_mul(3);
                v = w1 & 0x00FF00FF;
                let q = byte_perm(w0, w1, 0x7531).wrapping_add(0x7F7F7F7F) ^ 0x8080_8080;
                let mut j = 0;
                while j < Nc::<NC>::N {
                    unroll!();
                    sumi[j][i] = dp4a(q, u2[j][2 + t], sumi[j][i]);
                    j += 1;
                }
                t += 1;
            }
            i += 1;
        }
        fold::<NC, 3>(&mut acc, &mut sumi, &ds);
        let mut i = 0;
        while i < 4 {
            unroll!();
            let d = h2f(ld16(bq[i].add(26)));
            let mut j = 0;
            while j < Nc::<NC>::N {
                unroll!();
                res[j][i] = mul(d, acc[j][i]);
                j += 1;
            }
            i += 1;
        }
    }

    /// `up * silu(gate)` as the SWIGLU epilogue's PTX.
    #[inline(always)]
    pub fn swiglu_mul(up: f32, g: f32) -> f32 {
        let r: f32;
        unsafe {
            ptx_asm!(
                "{ .reg .f32 t, e, d, s; mul.ftz.f32 t, %2, 0fBFB8AA3B; ex2.approx.ftz.f32 e, t; add.ftz.f32 d, e, 0f3F800000; div.approx.ftz.f32 s, %2, d; mul.ftz.f32 %0, %1, s; }",
                out("=f") r, in("f") up, in("f") g, options(register_only)
            );
        }
        r
    }

    /// Params (reference ABI): vx, vy, fusion {x_bias, gate, gate_bias, x_scale, gate_scale: u64; glu_op: u32,
    /// pad}, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, rows_per_cta (i32), bpr (uint3).
    /// Grid ceil(nrows_x / rows_per_cta), block 128, dynamic smem NC * rows_per_cta * bpr * 4 (x2 with the gate).
    #[inline(always)]
    pub unsafe fn mmvq_pt<const NC: usize, const GLU: bool, D: Dst>(
        vx: *const u8, vy: *const u8, x_bias: *const f32, gate: *const u8, gate_bias: *const f32, glu_op: u32, dst: *mut D,
        ncols_x: i32, nrows_x: i32, stride_row_x: i32, stride_col_y: i32, stride_col_dst: i32, rows_per_cta: i32,
        bpr_mp: u32, bpr_l: u32,
    ) {
        let partials = DynamicSharedArray::<f32>::get();
        let bpr = ncols_x / 128;
        let nblk = ((ncols_x + 511) / 512) * 4;
        let row0 = rows_per_cta.wrapping_mul(thread::blockIdx_x() as i32);
        let tid = thread::threadIdx_x() as i32;
        let pgate = partials.wrapping_add((NC as i32 * rows_per_cta * bpr) as usize);

        let mut ycol = [vy; NC];
        let mut j = 0;
        while j < Nc::<NC>::N {
            unroll!();
            ycol[j] = vy.offset(((j as i32).wrapping_mul(stride_col_y) as isize) * 36);
            j += 1;
        }
        let n_items = (rows_per_cta / 4) * bpr;
        let mut idx = tid;
        while idx < n_items {
            let rg = fastdiv(idx as u32, bpr_mp, bpr_l) as i32;
            let kbx = idx - rg * bpr;
            let mut bq = [vx; 4];
            let mut i = 0;
            while i < 4 {
                unroll!();
                let mut row = row0 + rg * 4 + i as i32;
                row = if row < nrows_x { row } else { nrows_x - 1 };
                bq[i] = vx.offset((((row as i64) * (stride_row_x as i64) + kbx as i64) * 28) as isize);
                i += 1;
            }
            let mut dots = [[0f32; 4]; NC];
            pt_block_dot::<NC>(&bq, &ycol, kbx, nblk, &mut dots);
            let mut j = 0;
            while j < Nc::<NC>::N {
                unroll!();
                let mut i = 0;
                while i < 4 {
                    unroll!();
                    *partials.add((((j as i32) * rows_per_cta + rg * 4 + i as i32) * bpr + kbx) as usize) = dots[j][i];
                    i += 1;
                }
                j += 1;
            }
            if GLU {
                let mut bg = [gate; 4];
                let mut i = 0;
                while i < 4 {
                    unroll!();
                    let mut row = row0 + rg * 4 + i as i32;
                    row = if row < nrows_x { row } else { nrows_x - 1 };
                    bg[i] = gate.offset((((row as i64) * (stride_row_x as i64) + kbx as i64) * 28) as isize);
                    i += 1;
                }
                pt_block_dot::<NC>(&bg, &ycol, kbx, nblk, &mut dots);
                let mut j = 0;
                while j < Nc::<NC>::N {
                    unroll!();
                    let mut i = 0;
                    while i < 4 {
                        unroll!();
                        *pgate.add((((j as i32) * rows_per_cta + rg * 4 + i as i32) * bpr + kbx) as usize) = dots[j][i];
                        i += 1;
                    }
                    j += 1;
                }
            }
            idx += 128;
        }
        thread::sync_threads();

        let warp_id = tid / 32;
        let lane = tid % 32;
        let mut w = warp_id;
        while w < rows_per_cta * NC as i32 {
            let j = w / rows_per_cta;
            let r = w - j * rows_per_cta;
            let row = row0 + r;
            let mut sum = 0f32;
            let mut sum_gate = 0f32;
            let mut kbx = lane;
            while kbx < bpr {
                sum = add(sum, *partials.add(((j * rows_per_cta + r) * bpr + kbx) as usize));
                if GLU {
                    sum_gate = add(sum_gate, *pgate.add(((j * rows_per_cta + r) * bpr + kbx) as usize));
                }
                kbx += 32;
            }
            let mut k = 0;
            while k < 5 {
                unroll!();
                sum = add(sum, warp::shuffle_xor_f32_sync(0xffff_ffff, sum, 16 >> k));
                k += 1;
            }
            if GLU {
                let mut k = 0;
                while k < 5 {
                    unroll!();
                    sum_gate = add(sum_gate, warp::shuffle_xor_f32_sync(0xffff_ffff, sum_gate, 16 >> k));
                    k += 1;
                }
            }
            if lane == 0 && row < nrows_x {
                let o = j.wrapping_mul(stride_col_dst).wrapping_add(row);
                let mut result = sum;
                if GLU {
                    if !x_bias.is_null() {
                        result = addf(result, *x_bias.offset(o as isize));
                    }
                    let mut gv = sum_gate;
                    if !gate_bias.is_null() {
                        gv = addf(gv, *gate_bias.offset(o as isize));
                    }
                    result = if glu_op == 2 { swiglu_mul(result, gv) } else { mulf(result, gv) };
                }
                D::st(dst.offset(o as isize), result);
            }
            w += 4;
        }
    }

    // ------------------------------------------------------------------------------------------
    // MMQ quantizer: `quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4, false, false, false>` of the bonsai2 branch
    // (the acecd56 float program; the trailing gate / norm_weight / norm_scale pointers are unused).
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

    #[kernel]
    pub unsafe fn ptq1_0_quantize_mmq_d4_f32(x: *const f32, ids: *const i32, vy: *mut u8, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i32, ne2: i32, _n_expert_used: i32, _gate: u64, _norm_weight: u64, _norm_scale: u64) {
        quantize_mmq_d4(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2)
    }
    #[kernel]
    pub unsafe fn ptq1_0_quantize_mmq_d4_bf16(x: *const u16, ids: *const i32, vy: *mut u8, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i32, ne2: i32, _n_expert_used: i32, _gate: u64, _norm_weight: u64, _norm_scale: u64) {
        quantize_mmq_d4(x as *const B, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2)
    }

    // ------------------------------------------------------------------------------------------
    // MMQ: `mul_mat_q<GGML_TYPE_PTQ1_0, J, fallback>` + `mul_mat_q_stream_k_fixup` (mmq.cuh,
    // Turing+ MMA data layout; mmq-config-ampere.cuh: 256 threads, I = 128, SRAM layout Q8_0
    // (stride 76), K_vram = 256, stream-k). Shared memory: ids[J] | tile_y (J*36 ints padded to
    // 256) | tile_x (128 rows x 76 ints: 64 quant ints of two PTQ1_0 blocks, then 8 f32 scales).

    const NWARPS: i32 = 8;
    const WARP: i32 = 32;
    const MMQ_I: i32 = 128;
    const TILE_NE_K: i32 = 32;
    const TILE_Y_K: i32 = 36;
    const SRAM: i32 = 76;
    const SZ: i32 = 36;
    const BPI: i32 = 2; // MMQ_ITER_K / QK_PTQ1_0

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
    /// tile<16,8>::get_i(l) / get_j(l).
    #[inline(always)]
    fn c_i(l: i32) -> i32 {
        (l / 2) * 8 + tx() / 4
    }
    #[inline(always)]
    fn c_j(l: i32) -> i32 {
        (tx() % 4) * 2 + l % 2
    }

    /// Read-only global loads as `ld.global.nc` (the reference's `const __restrict__` loads).
    #[inline(always)]
    unsafe fn ldg_s16(p: *const i16) -> i32 {
        let r: i32;
        ptx_asm!("ld.global.nc.s16 %0, [%1];", out("=r") r, in("l") p as u64, options(register_only));
        r
    }
    #[inline(always)]
    unsafe fn ldg_u16(p: *const u16) -> u32 {
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

    /// `(const block_ptq1_0 *) x + kbx0 + i*stride + kb`: each int offset sign-extended separately.
    #[inline(always)]
    fn xblock(x: *const u8, kbx0: i32, i: i32, stride: i32, kb: i32) -> *const u8 {
        let off = (kbx0 as i64).wrapping_add(i.wrapping_mul(stride) as i64).wrapping_add(kb as i64);
        x.wrapping_offset(off.wrapping_mul(28) as isize)
    }

    /// ggml_cuda_mmq_load_tiles_ptq1_0 (MMA layout, branch-free unpack): 16 threads per row (8 per block);
    /// lane l < 4 decodes qs word l (elements 16 t + 4 l + b -> ints l + 4 t), lanes 4, 5 decode qs words 4, 5
    /// (ints 20 + 2 t + (l - 4)), lane 6 walks qh0 / qh1 (ints 30, 31), lane 7 idles; then the f16 scale into
    /// the 4 sub-block slots of each block. The tile holds exact integers, so only their placement matters.
    #[inline(always)]
    unsafe fn load_tiles<const J: i32, const FB: bool>(x: *const u8, kbx0: i32, i_max: i32, stride: i32) {
        let x_qs = smem().wrapping_add((J + Jc::<J>::YPAD) as usize);
        let x_df = x_qs.wrapping_add(2 * TILE_NE_K as usize) as *mut f32;
        let txi = tx() % 16;
        let kbx = txi / 8;
        let lane = txi % 8;
        let full = lane < 6;
        let dst_base = if lane < 4 { lane } else { 16 + lane };
        let dst_stride = if lane < 4 { 4 } else { 2 };
        let word = if lane < 7 { lane } else { 6 };
        let mut i0 = 0;
        while i0 < MMQ_I {
            unroll!();
            let mut i = i0 + ty() * 2 + tx() / 16;
            if FB {
                i = if i < i_max { i } else { i_max };
            }
            let bxi = xblock(x, kbx0, i, stride, kbx);
            let row = x_qs.wrapping_offset((i * SRAM + kbx * 32) as isize);
            let packed = ldg_b32(bxi.wrapping_add(4 * word as usize) as *const i32) as u32;
            let mut v_lo = byte_perm(packed, 0, 0x4140);
            let mut v_hi = byte_perm(packed, 0, 0x4342);
            v_hi = if full { v_hi } else { v_lo };
            let mut q = [0u32; 5];
            let mut t = 0;
            while t < 5 {
                unroll!();
                let w_lo = v_lo.wrapping_mul(3);
                let w_hi = v_hi.wrapping_mul(3);
                v_lo = w_lo & 0x00FF00FF;
                v_hi = w_hi & 0x00FF00FF;
                q[t] = byte_perm(w_lo, w_hi, 0x7531).wrapping_add(0x7F7F7F7F) ^ 0x8080_8080;
                if full {
                    *row.wrapping_offset((dst_base + t as i32 * dst_stride) as isize) = q[t] as i32;
                }
                t += 1;
            }
            if lane == 6 {
                *row.wrapping_add(30) = byte_perm(q[0], q[1], 0x5410) as i32;
                *row.wrapping_add(31) = byte_perm(q[2], q[3], 0x5410) as i32;
            }
            i0 += 2 * NWARPS;
        }
        let ksx = tx() % 8;
        let scale_block = ksx / 4;
        let mut i0 = 0;
        while i0 < MMQ_I {
            unroll!();
            let mut i = i0 + ty() * 4 + tx() / 8;
            if FB {
                i = if i < i_max { i } else { i_max };
            }
            let bxi = xblock(x, kbx0, i, stride, scale_block);
            *x_df.offset((i * SRAM + ksx) as isize) = h2f(ldg_u16(bxi.wrapping_add(26) as *const u16));
            i0 += 4 * NWARPS;
        }
    }

    /// ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<.., MMQ_Q8_1_DS_LAYOUT_D4> (NVIDIA branch):
    /// `sum += (dA * (float) C) * dB` as mul.ftz + fma.ftz.
    #[inline(always)]
    unsafe fn vec_dot<const J: i32>(sum: &mut [f32; 64], k00: i32) {
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

    /// ggml_cuda_mmq_write_back_mma: `dst[ids_dst[j]*stride + i] = sum[...]`.
    #[inline(always)]
    unsafe fn write_back<const J: i32, const FB: bool>(sum: &[f32; 64], dst: *mut f32, stride: i32, i_max: i32, j_max: i32) {
        let ids = smem();
        let ntx = Jc::<J>::NTX;
        let i0 = (ty() / ntx) * (ntx * 16);
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
                        let o = (*ids.offset(j as isize)).wrapping_mul(stride).wrapping_add(i);
                        *dst.offset(o as isize) = sum[((j0 / 8 + n) * 4 + l) as usize];
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
    unsafe fn process_tile<const J: i32, const FB: bool, const FIXUP: bool>(
        x: *const u8, offset_x: i32, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, stride_row_x: i32, ncols_y: i32,
        stride_col_dst: i32, tile_x_max_i: i32, tile_y_max_j: i32, kb0_start: i32, kb0_stop: i32,
    ) {
        let mut sum = [0f32; 64];
        let mut kb0 = kb0_start;
        while kb0 < kb0_stop {
            load_tiles::<J, FB>(x, offset_x.wrapping_add(kb0), tile_x_max_i, stride_row_x);
            load_tile_y::<J>(y.wrapping_offset(ncols_y.wrapping_mul(kb0.wrapping_mul(SZ)) as isize));
            thread::sync_threads();
            vec_dot::<J>(&mut sum, 0);
            thread::sync_threads();
            load_tile_y::<J>(y.wrapping_offset(ncols_y.wrapping_mul(kb0.wrapping_mul(SZ).wrapping_add(SZ)) as isize));
            thread::sync_threads();
            vec_dot::<J>(&mut sum, TILE_NE_K);
            thread::sync_threads();
            kb0 += BPI;
        }
        if FIXUP {
            let t = tmp_fixup.wrapping_offset((thread::blockIdx_x() as i32).wrapping_mul(J * MMQ_I) as isize);
            write_back::<J, FB>(&sum, t, MMQ_I, MMQ_I, J);
        } else {
            write_back::<J, FB>(&sum, dst, stride_col_dst, tile_x_max_i, tile_y_max_j);
        }
    }

    /// `ids_dst_shared[j] = f(j)` for j < J (256 threads, one pass: J <= 128).
    #[inline(always)]
    unsafe fn fill_ids<const J: i32, F: Fn(i32) -> i32>(f: F) {
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

    /// Stream-k `mul_mat_q<PTQ1_0, J, fallback>` (dense, or MoE via ids_dst / expert_bounds).
    #[inline(always)]
    pub unsafe fn mul_mat_q<const J: i32, const FB: bool>(
        x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32,
        bpn: Fd, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32,
        channel_ratio: Fd, ncy: Fd, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32,
        sample_ratio: Fd, nsy: Fd, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx: Fd,
    ) {
        let nty = (nrows_x.wrapping_add(MMQ_I - 1) / MMQ_I) as u32;
        fill_ids::<J, _>(|j| j);
        thread::sync_threads();

        let bpi = BPI as u32;
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
            if !ids_dst.is_null() {
                col_low = *expert_bounds.offset(zt as isize);
                let col_high = *expert_bounds.offset(zt as isize + 1);
                col_diff = col_high.wrapping_sub(col_low);
                offset_y = 0;
                offset_dst = 0;
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
            let tile_x_max_i = nrows_x.wrapping_sub(it.wrapping_mul(MMQ_I)).wrapping_sub(1);
            let tile_y_max_j = col_diff.wrapping_sub(jt.wrapping_mul(J)).wrapping_sub(1);
            let offset_x = fdiv(wt as u32, sample_ratio)
                .wrapping_mul(stride_sample_x as u32)
                .wrapping_add(fdiv(zt as u32, channel_ratio).wrapping_mul(stride_channel_x as u32))
                .wrapping_add(it.wrapping_mul(MMQ_I).wrapping_mul(stride_row_x) as u32) as i32;
            process_tile::<J, FB, false>(
                x, offset_x, y.wrapping_offset(offset_y as isize), dst.wrapping_offset(offset_dst as isize), tmp_fixup,
                stride_row_x, ncols_y, stride_col_dst, tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop,
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
        if !ids_dst.is_null() {
            col_low = *expert_bounds.offset(zt as isize);
            let col_high = *expert_bounds.offset(zt as isize + 1);
            col_diff = col_high.wrapping_sub(col_low);
            offset_y = 0;
            offset_dst = 0;
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
        let tile_x_max_i = nrows_x.wrapping_sub(it.wrapping_mul(MMQ_I)).wrapping_sub(1);
        let tile_y_max_j = col_diff.wrapping_sub(jt.wrapping_mul(J)).wrapping_sub(1);
        let offset_x = fdiv(wt as u32, sample_ratio)
            .wrapping_mul(stride_sample_x as u32)
            .wrapping_add(fdiv(zt as u32, channel_ratio).wrapping_mul(stride_channel_x as u32))
            .wrapping_add(it.wrapping_mul(MMQ_I).wrapping_mul(stride_row_x) as u32) as i32;
        process_tile::<J, FB, true>(
            x, offset_x, y.wrapping_offset(offset_y as isize), dst.wrapping_offset(offset_dst as isize), tmp_fixup,
            stride_row_x, ncols_y, stride_col_dst, tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop,
        );
    }

    /// mul_mat_q_stream_k_fixup<PTQ1_0, J, fallback>: grid (nblocks_sk, I / 32), block (32, 4);
    /// thread (x, y) owns output row `blockIdx.y*32 + x` of the columns `j0 + y`.
    #[inline(always)]
    pub unsafe fn stream_k_fixup<const J: i32, const FB: bool>(
        ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *const f32, bpn: Fd, nrows_x: i32,
        ncols_dst: i32, stride_col_dst: i32, ncy: Fd, stride_channel_dst: i32, nsy: Fd, stride_sample_dst: i32, ntx: Fd,
    ) {
        static mut IDS: SharedArray<i32, 128> = SharedArray::UNINIT;
        const FW: i32 = 4; // nwarps of the fixup kernel
        let bpi = BPI as u32;
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

    // One entry per reference instance (gen_kernels.py).
    // GENERATED KERNELS BEGIN
    #[kernel] #[launch_bounds(128, 4)] pub unsafe fn ptq1_0_mmvq_pt_c1(vx: *const u8, vy: *const u8, x_bias: *const f32, gate: *const u8, gate_bias: *const f32, _x_scale: u64, _gate_scale: u64, glu_op: u32, _pad: u32, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_row_x: i32, stride_col_y: i32, stride_col_dst: i32, rows_per_cta: i32, bpr_mp: u32, bpr_l: u32, _bpr_d: u32) { mmvq_pt::<1, false, f32>(vx, vy, x_bias, gate, gate_bias, glu_op, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, rows_per_cta, bpr_mp, bpr_l) }
    #[kernel] #[launch_bounds(128, 4)] pub unsafe fn ptq1_0_mmvq_pt_c1_bf16(vx: *const u8, vy: *const u8, x_bias: *const f32, gate: *const u8, gate_bias: *const f32, _x_scale: u64, _gate_scale: u64, glu_op: u32, _pad: u32, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_row_x: i32, stride_col_y: i32, stride_col_dst: i32, rows_per_cta: i32, bpr_mp: u32, bpr_l: u32, _bpr_d: u32) { mmvq_pt::<1, false, B>(vx, vy, x_bias, gate, gate_bias, glu_op, dst as *mut B, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, rows_per_cta, bpr_mp, bpr_l) }
    #[kernel] #[launch_bounds(128, 4)] pub unsafe fn ptq1_0_mmvq_pt_glu_c1(vx: *const u8, vy: *const u8, x_bias: *const f32, gate: *const u8, gate_bias: *const f32, _x_scale: u64, _gate_scale: u64, glu_op: u32, _pad: u32, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_row_x: i32, stride_col_y: i32, stride_col_dst: i32, rows_per_cta: i32, bpr_mp: u32, bpr_l: u32, _bpr_d: u32) { mmvq_pt::<1, true, f32>(vx, vy, x_bias, gate, gate_bias, glu_op, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, rows_per_cta, bpr_mp, bpr_l) }
    #[kernel] #[launch_bounds(128, 4)] pub unsafe fn ptq1_0_mmvq_pt_c2(vx: *const u8, vy: *const u8, x_bias: *const f32, gate: *const u8, gate_bias: *const f32, _x_scale: u64, _gate_scale: u64, glu_op: u32, _pad: u32, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_row_x: i32, stride_col_y: i32, stride_col_dst: i32, rows_per_cta: i32, bpr_mp: u32, bpr_l: u32, _bpr_d: u32) { mmvq_pt::<2, false, f32>(vx, vy, x_bias, gate, gate_bias, glu_op, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, rows_per_cta, bpr_mp, bpr_l) }
    #[kernel] #[launch_bounds(128, 4)] pub unsafe fn ptq1_0_mmvq_pt_c2_bf16(vx: *const u8, vy: *const u8, x_bias: *const f32, gate: *const u8, gate_bias: *const f32, _x_scale: u64, _gate_scale: u64, glu_op: u32, _pad: u32, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_row_x: i32, stride_col_y: i32, stride_col_dst: i32, rows_per_cta: i32, bpr_mp: u32, bpr_l: u32, _bpr_d: u32) { mmvq_pt::<2, false, B>(vx, vy, x_bias, gate, gate_bias, glu_op, dst as *mut B, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, rows_per_cta, bpr_mp, bpr_l) }
    #[kernel] #[launch_bounds(128, 4)] pub unsafe fn ptq1_0_mmvq_pt_glu_c2(vx: *const u8, vy: *const u8, x_bias: *const f32, gate: *const u8, gate_bias: *const f32, _x_scale: u64, _gate_scale: u64, glu_op: u32, _pad: u32, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_row_x: i32, stride_col_y: i32, stride_col_dst: i32, rows_per_cta: i32, bpr_mp: u32, bpr_l: u32, _bpr_d: u32) { mmvq_pt::<2, true, f32>(vx, vy, x_bias, gate, gate_bias, glu_op, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, rows_per_cta, bpr_mp, bpr_l) }
    #[kernel] #[launch_bounds(128, 3)] pub unsafe fn ptq1_0_mmvq_pt_c3(vx: *const u8, vy: *const u8, x_bias: *const f32, gate: *const u8, gate_bias: *const f32, _x_scale: u64, _gate_scale: u64, glu_op: u32, _pad: u32, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_row_x: i32, stride_col_y: i32, stride_col_dst: i32, rows_per_cta: i32, bpr_mp: u32, bpr_l: u32, _bpr_d: u32) { mmvq_pt::<3, false, f32>(vx, vy, x_bias, gate, gate_bias, glu_op, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, rows_per_cta, bpr_mp, bpr_l) }
    #[kernel] #[launch_bounds(128, 3)] pub unsafe fn ptq1_0_mmvq_pt_c3_bf16(vx: *const u8, vy: *const u8, x_bias: *const f32, gate: *const u8, gate_bias: *const f32, _x_scale: u64, _gate_scale: u64, glu_op: u32, _pad: u32, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_row_x: i32, stride_col_y: i32, stride_col_dst: i32, rows_per_cta: i32, bpr_mp: u32, bpr_l: u32, _bpr_d: u32) { mmvq_pt::<3, false, B>(vx, vy, x_bias, gate, gate_bias, glu_op, dst as *mut B, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, rows_per_cta, bpr_mp, bpr_l) }
    #[kernel] #[launch_bounds(128, 3)] pub unsafe fn ptq1_0_mmvq_pt_glu_c3(vx: *const u8, vy: *const u8, x_bias: *const f32, gate: *const u8, gate_bias: *const f32, _x_scale: u64, _gate_scale: u64, glu_op: u32, _pad: u32, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_row_x: i32, stride_col_y: i32, stride_col_dst: i32, rows_per_cta: i32, bpr_mp: u32, bpr_l: u32, _bpr_d: u32) { mmvq_pt::<3, true, f32>(vx, vy, x_bias, gate, gate_bias, glu_op, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, rows_per_cta, bpr_mp, bpr_l) }
    #[kernel] #[launch_bounds(128, 3)] pub unsafe fn ptq1_0_mmvq_pt_c4(vx: *const u8, vy: *const u8, x_bias: *const f32, gate: *const u8, gate_bias: *const f32, _x_scale: u64, _gate_scale: u64, glu_op: u32, _pad: u32, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_row_x: i32, stride_col_y: i32, stride_col_dst: i32, rows_per_cta: i32, bpr_mp: u32, bpr_l: u32, _bpr_d: u32) { mmvq_pt::<4, false, f32>(vx, vy, x_bias, gate, gate_bias, glu_op, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, rows_per_cta, bpr_mp, bpr_l) }
    #[kernel] #[launch_bounds(128, 3)] pub unsafe fn ptq1_0_mmvq_pt_c4_bf16(vx: *const u8, vy: *const u8, x_bias: *const f32, gate: *const u8, gate_bias: *const f32, _x_scale: u64, _gate_scale: u64, glu_op: u32, _pad: u32, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_row_x: i32, stride_col_y: i32, stride_col_dst: i32, rows_per_cta: i32, bpr_mp: u32, bpr_l: u32, _bpr_d: u32) { mmvq_pt::<4, false, B>(vx, vy, x_bias, gate, gate_bias, glu_op, dst as *mut B, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, rows_per_cta, bpr_mp, bpr_l) }
    #[kernel] #[launch_bounds(128, 3)] pub unsafe fn ptq1_0_mmvq_pt_glu_c4(vx: *const u8, vy: *const u8, x_bias: *const f32, gate: *const u8, gate_bias: *const f32, _x_scale: u64, _gate_scale: u64, glu_op: u32, _pad: u32, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_row_x: i32, stride_col_y: i32, stride_col_dst: i32, rows_per_cta: i32, bpr_mp: u32, bpr_l: u32, _bpr_d: u32) { mmvq_pt::<4, true, f32>(vx, vy, x_bias, gate, gate_bias, glu_op, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, rows_per_cta, bpr_mp, bpr_l) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn ptq1_0_mmq_j8_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, _y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<8, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn ptq1_0_mmq_j16_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, _y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<16, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn ptq1_0_mmq_j24_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, _y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<24, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn ptq1_0_mmq_j32_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, _y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<32, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn ptq1_0_mmq_j40_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, _y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<40, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn ptq1_0_mmq_j48_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, _y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<48, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn ptq1_0_mmq_j64_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, _y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<64, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn ptq1_0_mmq_j80_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, _y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<80, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn ptq1_0_mmq_j96_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, _y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<96, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn ptq1_0_mmq_j112_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, _y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<112, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn ptq1_0_mmq_j128_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, _y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<128, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn ptq1_0_mmq_j8_f1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, _y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<8, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn ptq1_0_mmq_j16_f1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, _y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<16, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn ptq1_0_mmq_j32_f1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, _y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<32, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn ptq1_0_mmq_j64_f1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, _y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<64, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn ptq1_0_mmq_j128_f1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, _y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<128, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn ptq1_0_mmq_fixup_j8_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<8, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn ptq1_0_mmq_fixup_j16_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<16, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn ptq1_0_mmq_fixup_j24_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<24, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn ptq1_0_mmq_fixup_j32_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<32, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn ptq1_0_mmq_fixup_j40_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<40, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn ptq1_0_mmq_fixup_j48_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<48, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn ptq1_0_mmq_fixup_j64_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<64, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn ptq1_0_mmq_fixup_j80_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<80, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn ptq1_0_mmq_fixup_j96_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<96, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn ptq1_0_mmq_fixup_j112_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<112, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn ptq1_0_mmq_fixup_j128_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<128, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn ptq1_0_mmq_fixup_j8_f1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<8, true>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn ptq1_0_mmq_fixup_j16_f1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<16, true>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn ptq1_0_mmq_fixup_j32_f1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<32, true>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn ptq1_0_mmq_fixup_j64_f1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<64, true>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn ptq1_0_mmq_fixup_j128_f1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<128, true>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    // GENERATED KERNELS END
}

fn main() {
    std::process::exit(if gate::run() { 0 } else { 1 });
}
