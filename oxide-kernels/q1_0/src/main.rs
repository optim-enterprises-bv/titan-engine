//! GGML Q1_0 (type 41) CUDA kernels in cuda-oxide, ported from llama.cpp acecd56 under the
//! reference's own parameter ABI and bit-identical to its nvcc build (nvcc 13.3, -O3
//! -use_fast_math, sm_120a; `ref/nvcc_*.sh` rebuild mmvq.cu / convert.cu, and their SASS is
//! identical to the kernels shipped in ~/ai/llama.cpp/build/bin/libggml-cuda.so, see ref/sass_cmp.py):
//!
//! - `q1_0_dequant_{f32,f16,bf16}` = `dequantize_block<128, 1, dequantize_q1_0, dst_t>`
//!   (convert.cu; grid (ceil(ne00/512), min(ne01, 65535), min(ne02*ne03, 65535)), block 256).
//!   `v = mul.ftz(d, (float)(2*bit - 1))`, then `cvt.rn.{f16,bf16}.f32` for the half outputs.
//! - `q1_0_mmvq_<n>` = `mul_mat_vec_q<GGML_TYPE_Q1_0, n, has_fusion=false, small_k, false>`
//!   (mmvq.cu, table MMVQ_PARAMETERS_GENERIC): block (32, nwarps), nwarps 4 for n <= 4 and 2 for
//!   n 5..8, 1 row per CUDA block for n == 1 (4 with small_k), 2 otherwise.
//!   vec_dot_q1_0_q8_1: 32 sign bits -> +-1 bytes via three `__byte_perm`s (prmt with the selector
//!   masked by 0x7777), four dp4a against the Q8_1 chunk, then (SASS)
//!   `tmp = fma.ftz(mul.ftz(d_x, d_y), (float)sumi, tmp)`. Block reduction: warps 1.. store to
//!   shared memory, warp 0 adds them in warp order (add.ftz), then a butterfly over xor 16..1.
//! - `q1_0_mmvq_<n>[_small_k]_{f16,bf16}`: the same instances writing the f32 result rounded once
//!   (`cvt.rn`), i.e. llama.cpp's mmvq followed by `convert_unary<float, half|bf16>`.
//! - `q1_0_quantize_q8_1_{f32,f16,bf16}` = `quantize_q8_1` (quantize.cu; the f16 / bf16 entries
//!   widen exactly, then run the reference's f32 code: abs/max.ftz butterflies, `div.approx.ftz`
//!   for `amax / 127` and `x / d`, fast-math roundf).
//! - `q1_0_quantize_mmq_d4_{f32,f16,bf16}` = `quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4, false>`.
//! - `q1_0_mmq_j<J>_f<fallback>` = `mul_mat_q<GGML_TYPE_Q1_0, J, fallback>` (mmq.cuh, Turing+ MMA
//!   layout, mmq-config-ampere.cuh: 256 threads, I = 128, SRAM stride 76, K_vram 256, stream-k;
//!   J in {8,16,24,32,40,48,64,80,96,112,128} without / {8,16,32,64,128} with fallback), incl. the
//!   MoE (ids_dst / expert_bounds) branches; `q1_0_mmq_fixup_j<J>_f<fb>` =
//!   `mul_mat_q_stream_k_fixup`. vec_dot is `ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<D4>`:
//!   `sum = fma.ftz(mul.ftz(dA, (float) C), dB, sum)` after each m16n8k32 s8 MMA.
//!
//! `export_ptx.py` splits q1_0.ptx into the two modules mistral.rs embeds
//! (mistralrs-quant/src/gguf/q1_0_{mmvq,mmq}_oxide.ptx).
//!
//! Parameter ABI: the reference's by-value structs are passed as their fields (same param-space
//! offsets): `ggml_cuda_mm_fusion_args_device` = 5 x u64 + u32 glu_op + f32 glu_limit (48 bytes,
//! offset 24), each `uint3` = 3 x u32. The gate launches both sides from one packed parameter
//! buffer (CU_LAUNCH_PARAM_BUFFER_POINTER), so the bytes are identical by construction.
//!
//! Not ported (not reached by candle / mistral.rs): the has_fusion=true mmvq instances (fused
//! gate/bias/GLU; mistral.rs' GGUF qwen3 / qwen35 models call gate and up separately),
//! `mul_mat_vec_q_moe` (multi-token MUL_MAT_ID) and the quantize_mmq scatter / DS4 / D2S6 layouts.
#![allow(non_snake_case, clippy::missing_safety_doc, clippy::too_many_arguments)]
mod gate;
mod gate2;

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

    /// `dequantize_block<128, 1, dequantize_q1_0, dst_t>`.
    #[inline(always)]
    pub unsafe fn dequant<D: Dst>(
        vx: *const u8, y: *mut D, ne00: i64, ne01: i64, ne0203: i64, ne02_mp: u32, ne02_l: u32, ne02_d: u32,
        s01: i64, s02: i64, s03: i64,
    ) {
        let i00 = 2 * ((thread::blockDim_x() as i64) * (thread::blockIdx_x() as i64) + thread::threadIdx_x() as i64);
        if i00 >= ne00 {
            return;
        }
        let mut i01 = thread::blockIdx_y() as i64;
        while i01 < ne01 {
            let mut i0203 = thread::blockIdx_z() as i64;
            while i0203 < ne0203 {
                let n = i0203 as u32;
                let q = fastdiv(n, ne02_mp, ne02_l);
                let i02 = n.wrapping_sub(q.wrapping_mul(ne02_d)) as i64;
                let i03 = q as i64;
                let ibx0 = i03.wrapping_mul(s03).wrapping_add(i02.wrapping_mul(s02)).wrapping_add(i01.wrapping_mul(s01));
                let ib = ibx0.wrapping_add(i00 / 128);
                let iqs = (i00 % 128) as i32;
                let iybs = i00 - i00 % 128;
                let xb = vx.offset((ib as isize).wrapping_mul(18));
                let d = h2f(*(xb as *const u16) as u32);
                let b0 = ((*xb.add(2 + (iqs / 8) as usize) as i32) >> (iqs % 8)) & 1;
                let b1 = ((*xb.add(2 + ((iqs + 1) / 8) as usize) as i32) >> ((iqs + 1) % 8)) & 1;
                let v0 = mul(d, (2 * b0 - 1) as f32);
                let v1 = mul(d, (2 * b1 - 1) as f32);
                let iy0 = (i0203.wrapping_mul(ne01).wrapping_add(i01)).wrapping_mul(ne00).wrapping_add(iybs).wrapping_add(iqs as i64);
                D::st(y.offset(iy0 as isize), v0);
                D::st(y.offset(iy0 as isize + 1), v1);
                i0203 += thread::gridDim_z() as i64;
            }
            i01 += thread::gridDim_y() as i64;
        }
    }

    #[kernel]
    pub unsafe fn q1_0_dequant_f32(
        vx: *const u8, y: *mut f32, ne00: i64, ne01: i64, ne0203: i64, ne02_mp: u32, ne02_l: u32, ne02_d: u32,
        s01: i64, s02: i64, s03: i64,
    ) {
        dequant(vx, y, ne00, ne01, ne0203, ne02_mp, ne02_l, ne02_d, s01, s02, s03)
    }
    #[kernel]
    pub unsafe fn q1_0_dequant_f16(
        vx: *const u8, y: *mut u16, ne00: i64, ne01: i64, ne0203: i64, ne02_mp: u32, ne02_l: u32, ne02_d: u32,
        s01: i64, s02: i64, s03: i64,
    ) {
        dequant(vx, y as *mut H, ne00, ne01, ne0203, ne02_mp, ne02_l, ne02_d, s01, s02, s03)
    }
    #[kernel]
    pub unsafe fn q1_0_dequant_bf16(
        vx: *const u8, y: *mut u16, ne00: i64, ne01: i64, ne0203: i64, ne02_mp: u32, ne02_l: u32, ne02_d: u32,
        s01: i64, s02: i64, s03: i64,
    ) {
        dequant(vx, y as *mut B, ne00, ne01, ne0203, ne02_mp, ne02_l, ne02_d, s01, s02, s03)
    }

    /// `tmp += vec_dot_q1_0_q8_1(vx, yb, kbx, iqs)` (the `+=` contracted as in the SASS).
    /// `yb`: the Q8_1 block aligned with Q1_0 block `kbx`; `iqs` selects the 32-value chunk.
    #[inline(always)]
    pub unsafe fn vdot(tmp: f32, vx: *const u8, yb: *const u8, kbx: i32, iqs: i32) -> f32 {
        let xb = vx.offset((kbx as isize).wrapping_mul(18));
        let qs = xb.add(2 + 4 * iqs as usize) as *const i16;
        let yc = yb.add(36 * iqs as usize);
        let yq = yc.add(4) as *const u32;
        let mut sumi = 0i32;
        let mut j = 0;
        while j < 2 {
            let q = *qs.add(j) as i32 as u32;
            let u0 = *yq.add(4 * j);
            let u1 = *yq.add(4 * j + 1);
            let u2 = *yq.add(4 * j + 2);
            let u3 = *yq.add(4 * j + 3);
            let n0 = byte_perm(0x11100100, 0x11100100, q);
            let n1 = byte_perm(0x11100100, 0x11100100, ((q as i32) >> 2) as u32);
            let s0 = byte_perm(0x01FF, 0x01FF, n0);
            let s1 = byte_perm(0x01FF, 0x01FF, n1);
            let s2 = byte_perm(0x01FF, 0x01FF, ((n0 as i32) >> 16) as u32);
            let s3 = byte_perm(0x01FF, 0x01FF, ((n1 as i32) >> 16) as u32);
            let v0 = byte_perm(s0, s1, 0x5410);
            let v1 = byte_perm(s0, s1, 0x7632);
            let v2 = byte_perm(s2, s3, 0x5410);
            let v3 = byte_perm(s2, s3, 0x7632);
            sumi = dp4a(v0, u0, sumi);
            sumi = dp4a(v1, u1, sumi);
            sumi = dp4a(v2, u2, sumi);
            sumi = dp4a(v3, u3, sumi);
            j += 1;
        }
        let d1 = h2f(*(xb as *const u16) as u32);
        let d8 = h2f(*(yc as *const u16) as u32);
        fma(mul(d1, d8), sumi as f32, tmp)
    }

    /// Compile-time loop bounds (`#[unroll]` needs literal trip counts after monomorphization;
    /// `tmp[j][i]` then stays in registers).
    pub struct K<const NC: usize, const SMALL_K: bool>;
    impl<const NC: usize, const SMALL_K: bool> K<NC, SMALL_K> {
        pub const NW: i32 = if NC <= 4 { 4 } else { 2 };
        pub const NW1: i32 = if NC <= 4 { 3 } else { 1 };
        pub const RPB: usize = if NC == 1 { if SMALL_K { 4 } else { 1 } } else { 2 };
    }

    /// `mul_mat_vec_q<Q1_0, NC, false, SMALL_K, false>` (MMVQ_PARAMETERS_GENERIC).
    #[device]
    pub unsafe fn mmvq<const NC: usize, const SMALL_K: bool, D: Dst>(
        vx: *const u8, vy: *const u8, ids: *const i32, dst: *mut D, ncols_x: u32,
        nchannels_y_mp: u32, nchannels_y_l: u32, nchannels_y_d: u32,
        stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32,
        channel_ratio_mp: u32, channel_ratio_l: u32,
        stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32,
        sample_ratio_mp: u32, sample_ratio_l: u32,
        stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32,
    ) {
        // tmp_shared[nwarps-1][NC][rows_per_block][32]: at most 768 floats.
        static mut TMP_SHARED: SharedArray<f32, 768> = SharedArray::UNINIT;
        const QK: i32 = 128;
        let nwarps: i32 = K::<NC, SMALL_K>::NW;
        let rpb: i32 = K::<NC, SMALL_K>::RPB as i32;
        let tx = thread::threadIdx_x() as i32;
        let ty = thread::threadIdx_y() as i32;
        let tid = 32 * ty + tx;
        let row0 = rpb.wrapping_mul(thread::blockIdx_x() as i32);
        let blocks_per_row_x = (ncols_x / QK as u32) as i32;
        let blocks_per_iter = nwarps * 32 / 4;

        let channel_dst = thread::blockIdx_y();
        let (channel_x, channel_y) = if NC == 1 && !ids.is_null() {
            let cx = *ids.add(channel_dst as usize) as u32;
            let q = fastdiv(channel_dst, nchannels_y_mp, nchannels_y_l);
            (cx, channel_dst.wrapping_sub(q.wrapping_mul(nchannels_y_d)))
        } else {
            (fastdiv(channel_dst, channel_ratio_mp, channel_ratio_l), channel_dst)
        };
        let sample_dst = thread::blockIdx_z();
        let sample_x = fastdiv(sample_dst, sample_ratio_mp, sample_ratio_l);
        let sample_y = sample_dst;

        let mut tmp = [[0f32; 4]; NC];
        let y = vy
            .add(sample_y.wrapping_mul(stride_sample_y) as usize * 36)
            .add(channel_y.wrapping_mul(stride_channel_y) as usize * 36);
        let kbx_offset = sample_x
            .wrapping_mul(stride_sample_x)
            .wrapping_add(channel_x.wrapping_mul(stride_channel_x))
            .wrapping_add((row0 as u32).wrapping_mul(stride_row_x));

        let mut kbx = tid / 4;
        while kbx < blocks_per_row_x {
            let kby = kbx.wrapping_mul(4);
            let kqs = tid % 4;
            let mut j = 0;
            #[unroll]
            while j < NC {
                let yi = (j as u32).wrapping_mul(stride_col_y).wrapping_add(kby as u32);
                let yb = y.add(yi as usize * 36);
                let mut i = 0;
                #[unroll]
                while i < K::<NC, SMALL_K>::RPB {
                    let xk = kbx_offset.wrapping_add((i as u32).wrapping_mul(stride_row_x)).wrapping_add(kbx as u32) as i32;
                    tmp[j][i] = vdot(tmp[j][i], vx, yb, xk, kqs);
                    i += 1;
                }
                j += 1;
            }
            kbx += blocks_per_iter;
        }

        let sh = SharedArray::as_raw_mut_ptr(&raw mut TMP_SHARED);
        let idx = |l: i32, j: usize, i: usize| ((l as usize * NC + j) * rpb as usize + i) * 32 + tx as usize;
        if ty > 0 {
            let mut j = 0;
            #[unroll]
            while j < NC {
                let mut i = 0;
                #[unroll]
                while i < K::<NC, SMALL_K>::RPB {
                    *sh.add(idx(ty - 1, j, i)) = tmp[j][i];
                    i += 1;
                }
                j += 1;
            }
        }
        thread::sync_threads();
        if ty > 0 {
            return;
        }

        let dst = dst.add(
            sample_dst
                .wrapping_mul(stride_sample_dst)
                .wrapping_add(channel_dst.wrapping_mul(stride_channel_dst))
                .wrapping_add(row0 as u32) as usize,
        );
        let mut j = 0;
        #[unroll]
        while j < NC {
            let mut i = 0;
            #[unroll]
            while i < K::<NC, SMALL_K>::RPB {
                let mut l = 0;
                #[unroll]
                while l < K::<NC, SMALL_K>::NW1 {
                    tmp[j][i] = add(tmp[j][i], *sh.add(idx(l, j, i)));
                    l += 1;
                }
                let mut k = 0;
                #[unroll]
                while k < 5 {
                    tmp[j][i] = add(tmp[j][i], warp::shuffle_xor_f32_sync(0xffff_ffff, tmp[j][i], 16 >> k));
                    k += 1;
                }
                if tx == i as i32 && (rpb == 1 || (row0 as u32).wrapping_add(i as u32) < stride_col_dst) {
                    let o = (j as u32).wrapping_mul(stride_col_dst).wrapping_add(i as u32);
                    D::st(dst.add(o as usize), tmp[j][i]);
                }
                i += 1;
            }
            j += 1;
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

    /// `quantize_q8_1` (quantize.cu): one thread per value, block_q8_1 rows of `ne0` values.
    /// Params: x, vy, ne00, s01, s02, s03, ne0, ne1, ne2 (uint3).
    #[inline(always)]
    pub unsafe fn quantize_q8_1<S: Src>(
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
        let i_cont = ((ne1 as u64 * z as u64) as i64).wrapping_add(i1).wrapping_mul(ne0).wrapping_add(i0);
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
        let ib = (i_cont as u64) >> 5;
        let iqs = (i_cont as u64) & 31;
        let yb = vy.add((ib as usize).wrapping_mul(36));
        *yb.add(4 + iqs as usize) = q as u8;
        if iqs != 0 {
            return;
        }
        *(yb as *mut u32) = (f2h(d) as u32) | ((f2h(sum) as u32) << 16);
    }

    #[kernel]
    pub unsafe fn q1_0_quantize_q8_1_f32(x: *const f32, vy: *mut u8, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: u32, ne2_mp: u32, ne2_l: u32, ne2_d: u32) {
        quantize_q8_1(x, vy, ne00, s01, s02, s03, ne0, ne1, ne2_mp, ne2_l, ne2_d)
    }
    #[kernel]
    pub unsafe fn q1_0_quantize_q8_1_f16(x: *const u16, vy: *mut u8, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: u32, ne2_mp: u32, ne2_l: u32, ne2_d: u32) {
        quantize_q8_1(x as *const H, vy, ne00, s01, s02, s03, ne0, ne1, ne2_mp, ne2_l, ne2_d)
    }
    #[kernel]
    pub unsafe fn q1_0_quantize_q8_1_bf16(x: *const u16, vy: *mut u8, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: u32, ne2_mp: u32, ne2_l: u32, ne2_d: u32) {
        quantize_q8_1(x as *const B, vy, ne00, s01, s02, s03, ne0, ne1, ne2_mp, ne2_l, ne2_d)
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

    #[kernel]
    pub unsafe fn q1_0_quantize_mmq_d4_f32(x: *const f32, ids: *const i32, vy: *mut u8, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i32, ne2: i32, _n_expert_used: i32) {
        quantize_mmq_d4(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2)
    }
    #[kernel]
    pub unsafe fn q1_0_quantize_mmq_d4_f16(x: *const u16, ids: *const i32, vy: *mut u8, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i32, ne2: i32, _n_expert_used: i32) {
        quantize_mmq_d4(x as *const H, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2)
    }
    #[kernel]
    pub unsafe fn q1_0_quantize_mmq_d4_bf16(x: *const u16, ids: *const i32, vy: *mut u8, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i32, ne2: i32, _n_expert_used: i32) {
        quantize_mmq_d4(x as *const B, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2)
    }

    // ------------------------------------------------------------------------------------------
    // MMQ: `mul_mat_q<GGML_TYPE_Q1_0, J, fallback>` + `mul_mat_q_stream_k_fixup` (mmq.cuh,
    // Turing+ MMA data layout; mmq-config-ampere.cuh: 256 threads, I = 128, SRAM layout Q8_0
    // (stride 76), K_vram = 256, stream-k). Shared memory: ids[J] | tile_y (J*36 ints padded to
    // 256) | tile_x (128 rows x 76 ints: 64 quant ints of two Q1_0 blocks, then 8 f32 scales).

    const NWARPS: i32 = 8;
    const WARP: i32 = 32;
    const MMQ_I: i32 = 128;
    const TILE_NE_K: i32 = 32;
    const TILE_Y_K: i32 = 36;
    const SRAM: i32 = 76;
    const SZ: i32 = 36;
    const BPI: i32 = 2; // MMQ_ITER_K / QK1_0

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

    /// `(const block_q1_0 *) x + kbx0 + i*stride + kb`: each int offset sign-extended separately.
    #[inline(always)]
    fn xblock(x: *const u8, kbx0: i32, i: i32, stride: i32, kb: i32) -> *const u8 {
        let off = (kbx0 as i64).wrapping_add(i.wrapping_mul(stride) as i64).wrapping_add(kb as i64);
        x.wrapping_offset(off.wrapping_mul(18) as isize)
    }

    /// ggml_cuda_mmq_load_tiles_q1_0 (MMA layout).
    #[inline(always)]
    unsafe fn load_tiles<const J: i32, const FB: bool>(x: *const u8, kbx0: i32, i_max: i32, stride: i32) {
        let x_qs = smem().wrapping_add((J + Jc::<J>::YPAD) as usize);
        let x_df = x_qs.wrapping_add(2 * TILE_NE_K as usize) as *mut f32;
        let txi = tx() % 8;
        let kbx = txi / 4;
        let kqsx = txi % 4;
        let dst_offset = kbx * 32 + kqsx * 8;
        let mut i0 = 0;
        while i0 < MMQ_I {
            unroll!();
            let mut i = i0 + ty() * 4 + tx() / 8;
            if FB {
                i = if i < i_max { i } else { i_max };
            }
            let qxi = xblock(x, kbx0, i, stride, kbx).wrapping_add(2 + 4 * kqsx as usize) as *const i16;
            let row = x_qs.wrapping_offset((i * SRAM + dst_offset) as isize);
            let mut j = 0;
            while j < 2 {
                unroll!();
                let q = ldg_s16(qxi.add(j)) as u32;
                let n0 = byte_perm(0x11100100, 0x11100100, q);
                let n1 = byte_perm(0x11100100, 0x11100100, ((q as i32) >> 2) as u32);
                let s0 = byte_perm(0x01FF, 0x01FF, n0);
                let s1 = byte_perm(0x01FF, 0x01FF, n1);
                let s2 = byte_perm(0x01FF, 0x01FF, ((n0 as i32) >> 16) as u32);
                let s3 = byte_perm(0x01FF, 0x01FF, ((n1 as i32) >> 16) as u32);
                *row.add(j * 4) = byte_perm(s0, s1, 0x5410) as i32;
                *row.add(j * 4 + 1) = byte_perm(s0, s1, 0x7632) as i32;
                *row.add(j * 4 + 2) = byte_perm(s2, s3, 0x5410) as i32;
                *row.add(j * 4 + 3) = byte_perm(s2, s3, 0x7632) as i32;
                j += 1;
            }
            i0 += 4 * NWARPS;
        }
        let ksx = tx() % 8;
        let scale_block = ksx / 4;
        let mut i0 = 0;
        while i0 < MMQ_I {
            unroll!();
            let mut i = i0 + ty();
            if FB {
                i = if i < i_max { i } else { i_max };
            }
            let bxi = xblock(x, kbx0, i, stride, scale_block);
            *x_df.offset((i * SRAM + ksx) as isize) = h2f(ldg_u16(bxi as *const u16));
            i0 += NWARPS;
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

    /// Stream-k `mul_mat_q<Q1_0, J, fallback>` (dense, or MoE via ids_dst / expert_bounds).
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

    /// mul_mat_q_stream_k_fixup<Q1_0, J, fallback>: grid (nblocks_sk, I / 32), block (32, 4);
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

    // One entry per reference instance. Params: vx, vy, ids, fusion{x_bias, gate, gate_bias,
    // x_scale, gate_scale: u64; glu_op: u32; glu_limit: f32}, dst, ncols_x, nchannels_y(uint3),
    // stride_row_x, stride_col_y, stride_col_dst, channel_ratio(uint3), stride_channel_{x,y,dst},
    // sample_ratio(uint3), stride_sample_{x,y,dst}, ids_stride.
    // GENERATED KERNELS BEGIN
    #[kernel] pub unsafe fn q1_0_mmvq_1_small_k(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<1, true, f32>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn q1_0_mmvq_1(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<1, false, f32>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn q1_0_mmvq_2(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<2, false, f32>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn q1_0_mmvq_3(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<3, false, f32>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn q1_0_mmvq_4(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<4, false, f32>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn q1_0_mmvq_5(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<5, false, f32>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn q1_0_mmvq_6(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<6, false, f32>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn q1_0_mmvq_7(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<7, false, f32>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn q1_0_mmvq_8(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<8, false, f32>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn q1_0_mmvq_1_small_k_f16(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut u16, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<1, true, H>(vx, vy, ids, dst as *mut H, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn q1_0_mmvq_1_f16(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut u16, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<1, false, H>(vx, vy, ids, dst as *mut H, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn q1_0_mmvq_2_f16(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut u16, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<2, false, H>(vx, vy, ids, dst as *mut H, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn q1_0_mmvq_3_f16(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut u16, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<3, false, H>(vx, vy, ids, dst as *mut H, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn q1_0_mmvq_4_f16(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut u16, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<4, false, H>(vx, vy, ids, dst as *mut H, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn q1_0_mmvq_5_f16(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut u16, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<5, false, H>(vx, vy, ids, dst as *mut H, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn q1_0_mmvq_6_f16(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut u16, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<6, false, H>(vx, vy, ids, dst as *mut H, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn q1_0_mmvq_7_f16(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut u16, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<7, false, H>(vx, vy, ids, dst as *mut H, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn q1_0_mmvq_8_f16(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut u16, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<8, false, H>(vx, vy, ids, dst as *mut H, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn q1_0_mmvq_1_small_k_bf16(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut u16, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<1, true, B>(vx, vy, ids, dst as *mut B, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn q1_0_mmvq_1_bf16(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut u16, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<1, false, B>(vx, vy, ids, dst as *mut B, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn q1_0_mmvq_2_bf16(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut u16, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<2, false, B>(vx, vy, ids, dst as *mut B, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn q1_0_mmvq_3_bf16(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut u16, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<3, false, B>(vx, vy, ids, dst as *mut B, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn q1_0_mmvq_4_bf16(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut u16, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<4, false, B>(vx, vy, ids, dst as *mut B, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn q1_0_mmvq_5_bf16(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut u16, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<5, false, B>(vx, vy, ids, dst as *mut B, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn q1_0_mmvq_6_bf16(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut u16, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<6, false, B>(vx, vy, ids, dst as *mut B, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn q1_0_mmvq_7_bf16(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut u16, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<7, false, B>(vx, vy, ids, dst as *mut B, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn q1_0_mmvq_8_bf16(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut u16, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<8, false, B>(vx, vy, ids, dst as *mut B, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn q1_0_mmq_j8_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, _y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<8, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn q1_0_mmq_j16_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, _y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<16, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn q1_0_mmq_j24_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, _y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<24, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn q1_0_mmq_j32_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, _y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<32, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn q1_0_mmq_j40_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, _y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<40, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn q1_0_mmq_j48_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, _y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<48, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn q1_0_mmq_j64_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, _y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<64, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn q1_0_mmq_j80_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, _y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<80, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn q1_0_mmq_j96_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, _y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<96, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn q1_0_mmq_j112_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, _y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<112, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn q1_0_mmq_j128_f0(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, _y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<128, false>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn q1_0_mmq_j8_f1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, _y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<8, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn q1_0_mmq_j16_f1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, _y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<16, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn q1_0_mmq_j32_f1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, _y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<32, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn q1_0_mmq_j64_f1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, _y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<64, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn q1_0_mmq_j128_f1(x: *const u8, y: *const i32, ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_fixup: *mut f32, _y_scale: *const f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, cr_mp: u32, cr_l: u32, cr_d: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_x: i32, stride_channel_y: i32, stride_channel_dst: i32, sr_mp: u32, sr_l: u32, sr_d: u32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_x: i32, stride_sample_y: i32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { mul_mat_q::<128, true>(x, y, ids_dst, expert_bounds, dst, tmp_fixup, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, Fd { mp: cr_mp, l: cr_l, d: cr_d }, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_x, stride_channel_y, stride_channel_dst, Fd { mp: sr_mp, l: sr_l, d: sr_d }, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_x, stride_sample_y, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn q1_0_mmq_fixup_j8_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<8, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn q1_0_mmq_fixup_j16_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<16, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn q1_0_mmq_fixup_j24_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<24, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn q1_0_mmq_fixup_j32_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<32, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn q1_0_mmq_fixup_j40_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<40, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn q1_0_mmq_fixup_j48_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<48, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn q1_0_mmq_fixup_j64_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<64, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn q1_0_mmq_fixup_j80_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<80, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn q1_0_mmq_fixup_j96_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<96, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn q1_0_mmq_fixup_j112_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<112, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn q1_0_mmq_fixup_j128_f0(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<128, false>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn q1_0_mmq_fixup_j8_f1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<8, true>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn q1_0_mmq_fixup_j16_f1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<16, true>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn q1_0_mmq_fixup_j32_f1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<32, true>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn q1_0_mmq_fixup_j64_f1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<64, true>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    #[kernel] #[launch_bounds(128, 1)] pub unsafe fn q1_0_mmq_fixup_j128_f1(ids_dst: *const i32, expert_bounds: *const i32, dst: *mut f32, tmp_last_tile: *mut f32, bpn_mp: u32, bpn_l: u32, bpn_d: u32, nrows_x: i32, ncols_dst: i32, stride_col_dst: i32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_channel_dst: i32, nsy_mp: u32, nsy_l: u32, nsy_d: u32, stride_sample_dst: i32, ntx_mp: u32, ntx_l: u32, ntx_d: u32) { stream_k_fixup::<128, true>(ids_dst, expert_bounds, dst, tmp_last_tile, Fd { mp: bpn_mp, l: bpn_l, d: bpn_d }, nrows_x, ncols_dst, stride_col_dst, Fd { mp: ncy_mp, l: ncy_l, d: ncy_d }, stride_channel_dst, Fd { mp: nsy_mp, l: nsy_l, d: nsy_d }, stride_sample_dst, Fd { mp: ntx_mp, l: ntx_l, d: ntx_d }) }
    // GENERATED KERNELS END
}

fn main() {
    std::process::exit(if gate::run() { 0 } else { 1 });
}
