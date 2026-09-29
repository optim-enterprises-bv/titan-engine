//! GGML NVFP4 (type 40) CUDA kernels in cuda-oxide, ported from llama.cpp acecd56 under the
//! reference's own parameter ABI and bit-identical to its nvcc build (nvcc 13.3, -O3
//! -use_fast_math, sm_120a; `ref/nvcc_*.sh` rebuild mmvq.cu / convert.cu, and their SASS is
//! identical to the kernels shipped in ~/ai/llama.cpp/build/bin/libggml-cuda.so, see ref/sass_cmp.py):
//!
//! - `nvfp4_dequant_{f32,f16,bf16}` = `dequantize_block_nvfp4<dst_t>` (convert.cu
//!   `dequantize_row_nvfp4_cuda`: grid k / 64, block 32, params (vx, y, ne)); each thread writes
//!   two values of one 16-value sub-block. `d = div.approx.ftz(cvt.f32.f16(cvt.rn.f16x2.e4m3x2(b)), 2)`
//!   with b = 0 when (b & 0x7F) == 0x7F (the CUDA path reads the UE4M3 byte as a signed E4M3),
//!   `v = mul.ftz(d, (float)kvalues_mxfp4[q])`, then `cvt.rn.{f16,bf16}.f32`.
//! - `nvfp4_mmvq_<n>` = `mul_mat_vec_q<GGML_TYPE_NVFP4, n, has_fusion=false, small_k, false>`
//!   (mmvq.cu, table MMVQ_PARAMETERS_GENERIC): block (32, nwarps), nwarps 4 for n <= 4 and 2 for
//!   n 5..8, 1 row per CUDA block for n == 1 (4 with small_k), 2 otherwise. qk 64, qi 8, vdr 4:
//!   two threads per block (two sub-blocks each), 16 * nwarps blocks per iteration.
//!   vec_dot_nvfp4_q8_1: per sub-block two nibble words -> table bytes, four dp4a, then (SASS)
//!   `sum = fma.ftz(mul.ftz(d_s, d_y), (float)sumi, sum)` from sum = +0 over the thread's two
//!   sub-blocks, and `tmp = add.ftz(tmp, sum)` (not contracted). Block reduction: warps 1.. store
//!   to shared memory, warp 0 adds them in warp order (add.ftz), then a butterfly over xor 16..1.
//!
//! Parameter ABI: the reference's by-value structs are passed as their fields (same param-space
//! offsets): `ggml_cuda_mm_fusion_args_device` = 5 x u64 + u32 glu_op + f32 glu_limit (48 bytes,
//! offset 24), each `uint3` = 3 x u32. The gate launches both sides from one packed parameter
//! buffer (CU_LAUNCH_PARAM_BUFFER_POINTER), so the bytes are identical by construction.
//!
//! Not ported (not reached by candle / mistral.rs): the has_fusion=true instances (fused
//! gate/bias/GLU), the halve_iters instance (DGX Spark table only) and `mul_mat_vec_q_moe`
//! (multi-token MUL_MAT_ID).
#![allow(non_snake_case, clippy::missing_safety_doc, clippy::too_many_arguments)]
mod gate;

use cuda_device::{SharedArray, convert, device, dotprod, kernel, ptx_asm, thread, warp};
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
    /// `mul.ftz.f32` without `.rn`, exactly as the reference PTX writes it: ptxas may fuse it
    /// (dequantize: `mul.ftz(mul.ftz(d, q), 0.5)` becomes one FMUL.FTZ.D2, which does not flush
    /// the halved result).
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

    /// `kvalues_mxfp4` as the four little-endian words `get_int_from_table_16` reads.
    pub const TABLE: [u32; 4] = [0x03020100, 0x0C080604, 0xFDFEFF00, 0xF4F8FAFC];
    pub const KV: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
    /// Block layout: 4 UE4M3 scale bytes, 32 nibble bytes.
    pub const BLOCK_BYTES: usize = 36;

    /// Read-only global loads as `ld.global.nc` (the reference's `const __restrict__` loads).
    #[inline(always)]
    pub unsafe fn ldg32(p: *const u8) -> u32 {
        let r: u32;
        ptx_asm!("ld.global.nc.b32 %0, [%1];", out("=r") r, in("l") p as u64, options(register_only));
        r
    }
    #[inline(always)]
    pub unsafe fn ldg16(p: *const u8) -> u32 {
        let r: u16;
        ptx_asm!("ld.global.nc.b16 %0, [%1];", out("=h") r, in("l") p as u64, options(register_only));
        r as u32
    }
    #[inline(always)]
    pub unsafe fn ldg8(p: *const u8) -> u8 {
        let r: u16;
        ptx_asm!("ld.global.nc.u8 %0, [%1];", out("=h") r, in("l") p as u64, options(register_only));
        r as u8
    }

    /// `ggml_cuda_ue4m3_to_fp32` (FP8_AVAILABLE): the byte as `__nv_fp8_e4m3` (0x7F / 0xFF -> 0),
    /// `cvt.rn.f16x2.e4m3x2`, f16 -> f32, then `/ 2` as fast-math `div.approx.ftz`.
    #[inline(always)]
    pub fn ue4m3(b: u8) -> f32 {
        let b = if b & 0x7F == 0x7F { 0u16 } else { b as u16 };
        let r: u32;
        unsafe { ptx_asm!("cvt.rn.f16x2.e4m3x2 %0, %1;", out("=r") r, in("h") b, options(register_only)); }
        let v = h2f(r & 0xFFFF);
        let q: f32;
        unsafe { ptx_asm!("div.approx.ftz.f32 %0, %1, 0f40000000;", out("=f") q, in("f") v, options(register_only)); }
        q
    }

    #[inline(always)]
    pub fn kv(i: u32) -> f32 {
        // select chain over the 16 entries keeps the table out of local memory
        let mut v = 0i32;
        let mut k = 0;
        while k < 16 {
            if i == k as u32 {
                v = KV[k] as i32;
            }
            k += 1;
        }
        v as f32
    }

    /// `dequantize_block_nvfp4<dst_t>`: one 64-value block per CUDA block of 32 threads.
    #[inline(always)]
    pub unsafe fn dequant<D: Dst>(vx: *const u8, yy: *mut D, ne: i64) {
        let i = thread::blockIdx_x() as i64;
        let tid = thread::threadIdx_x() as usize;
        let base = i * 64;
        if base >= ne {
            return;
        }
        let xb = vx.add(i as usize * BLOCK_BYTES);
        let sub = tid / 8;
        let j = tid % 8;
        let d = ue4m3(*xb.add(sub));
        let q = *xb.add(4 + sub * 8 + j) as u32;
        let y0 = base as usize + sub * 16 + j;
        D::st(yy.add(y0), mulf(d, kv(q & 0x0F)));
        D::st(yy.add(y0 + 8), mulf(d, kv(q >> 4)));
    }

    #[kernel]
    pub unsafe fn nvfp4_dequant_f32(vx: *const u8, y: *mut f32, ne: i64) {
        dequant(vx, y, ne)
    }
    #[kernel]
    pub unsafe fn nvfp4_dequant_f16(vx: *const u8, y: *mut u16, ne: i64) {
        dequant(vx, y as *mut H, ne)
    }
    #[kernel]
    pub unsafe fn nvfp4_dequant_bf16(vx: *const u8, y: *mut u16, ne: i64) {
        dequant(vx, y as *mut B, ne)
    }

    /// `get_int_from_table_16(q4, kvalues_mxfp4)`: (bytes of the low nibbles, of the high nibbles).
    #[inline(always)]
    pub fn table16(q4: u32) -> (u32, u32) {
        let sel = 0x32103210 | ((q4 & 0x88888888) >> 1);
        let lo0 = byte_perm(TABLE[0], TABLE[1], q4);
        let hi0 = byte_perm(TABLE[2], TABLE[3], q4);
        let t0 = byte_perm(lo0, hi0, sel);
        let lo1 = byte_perm(TABLE[0], TABLE[1], q4 >> 16);
        let hi1 = byte_perm(TABLE[2], TABLE[3], q4 >> 16);
        let t1 = byte_perm(lo1, hi1, sel >> 16);
        (byte_perm(t0, t1, 0x6420), byte_perm(t0, t1, 0x7531))
    }

    /// `tmp += vec_dot_nvfp4_q8_1(vx, yb, kbx, iqs)`: the dot's own sum starts at +0 and is added
    /// to `tmp` with add.ftz (as in the SASS). `yb`: the first of the two Q8_1 blocks aligned with
    /// NVFP4 block `kbx`; `iqs` is 0 or 4.
    #[inline(always)]
    pub unsafe fn vdot(tmp: f32, vx: *const u8, yb: *const u8, kbx: i32, iqs: i32) -> f32 {
        let xb = vx.offset((kbx as isize).wrapping_mul(BLOCK_BYTES as isize));
        let qs = xb.add(4);
        let mut sum = 0f32;
        let mut i = 0;
        while i < 2 {
            let iqs0 = (iqs + 2 * i) as usize;
            let is = iqs0 >> 1;
            let (v0x, v0y) = table16(ldg32(qs.add(4 * iqs0)));
            let (v1x, v1y) = table16(ldg32(qs.add(4 * iqs0 + 4)));
            let bq8 = yb.add((is >> 1) * 36);
            let q8 = bq8.add(4);
            let i8 = (is & 1) << 2;
            let mut sumi = dp4a(v0x, ldg32(q8.add(4 * i8)), 0);
            sumi = dp4a(v0y, ldg32(q8.add(4 * i8 + 8)), sumi);
            sumi = dp4a(v1x, ldg32(q8.add(4 * i8 + 4)), sumi);
            sumi = dp4a(v1y, ldg32(q8.add(4 * i8 + 12)), sumi);
            let d = mul(ue4m3(ldg8(xb.add(is))), h2f(ldg16(bq8)));
            sum = fma(d, sumi as f32, sum);
            i += 1;
        }
        add(tmp, sum)
    }

    /// Compile-time loop bounds (`#[unroll]` needs literal trip counts after monomorphization;
    /// `tmp[j][i]` then stays in registers).
    pub struct K<const NC: usize, const SMALL_K: bool>;
    impl<const NC: usize, const SMALL_K: bool> K<NC, SMALL_K> {
        pub const NW: i32 = if NC <= 4 { 4 } else { 2 };
        pub const NW1: i32 = if NC <= 4 { 3 } else { 1 };
        pub const RPB: usize = if NC == 1 { if SMALL_K { 4 } else { 1 } } else { 2 };
    }

    /// `mul_mat_vec_q<NVFP4, NC, false, SMALL_K, false>` (MMVQ_PARAMETERS_GENERIC).
    #[device]
    pub unsafe fn mmvq<const NC: usize, const SMALL_K: bool>(
        vx: *const u8, vy: *const u8, ids: *const i32, dst: *mut f32, ncols_x: u32,
        nchannels_y_mp: u32, nchannels_y_l: u32, nchannels_y_d: u32,
        stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32,
        channel_ratio_mp: u32, channel_ratio_l: u32,
        stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32,
        sample_ratio_mp: u32, sample_ratio_l: u32,
        stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32,
    ) {
        // tmp_shared[nwarps-1][NC][rows_per_block][32]: at most 768 floats.
        static mut TMP_SHARED: SharedArray<f32, 768> = SharedArray::UNINIT;
        const QK: i32 = 64;
        let nwarps: i32 = K::<NC, SMALL_K>::NW;
        let rpb: i32 = K::<NC, SMALL_K>::RPB as i32;
        let tx = thread::threadIdx_x() as i32;
        let ty = thread::threadIdx_y() as i32;
        let tid = 32 * ty + tx;
        let row0 = rpb.wrapping_mul(thread::blockIdx_x() as i32);
        let blocks_per_row_x = (ncols_x / QK as u32) as i32;
        let blocks_per_iter = 4 * nwarps * 32 / 8;

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

        let mut kbx = tid / 2;
        while kbx < blocks_per_row_x {
            let kby = kbx.wrapping_mul(2);
            let kqs = 4 * (tid % 2);
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
                    *dst.add(o as usize) = tmp[j][i];
                }
                i += 1;
            }
            j += 1;
        }
    }

    // One entry per reference instance. Params: vx, vy, ids, fusion{x_bias, gate, gate_bias,
    // x_scale, gate_scale: u64; glu_op: u32; glu_limit: f32}, dst, ncols_x, nchannels_y(uint3),
    // stride_row_x, stride_col_y, stride_col_dst, channel_ratio(uint3), stride_channel_{x,y,dst},
    // sample_ratio(uint3), stride_sample_{x,y,dst}, ids_stride.
    // GENERATED KERNELS BEGIN
    #[kernel] pub unsafe fn nvfp4_mmvq_1(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<1, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn nvfp4_mmvq_1_small_k(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<1, true>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn nvfp4_mmvq_2(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<2, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn nvfp4_mmvq_3(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<3, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn nvfp4_mmvq_4(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<4, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn nvfp4_mmvq_5(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<5, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn nvfp4_mmvq_6(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<6, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn nvfp4_mmvq_7(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<7, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn nvfp4_mmvq_8(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<8, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    // GENERATED KERNELS END
}

fn main() {
    std::process::exit(if gate::run() { 0 } else { 1 });
}
