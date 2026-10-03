//! GGML IQ4_XS (type 23) CUDA kernels in cuda-oxide, ported from llama.cpp acecd56 under the
//! reference's own parameter ABI and bit-identical to its nvcc build (nvcc 13.3, -O3
//! -use_fast_math, sm_120a; the cubins in ref/ are the same ones the iq4_nl port gates against,
//! and their SASS is identical to the kernels shipped in ~/ai/llama.cpp/build/bin/libggml-cuda.so):
//!
//! - `iq4_xs_dequant_{f32,f16,bf16}` = `dequantize_block_iq4_xs<dst_t>` (convert.cu
//!   `dequantize_row_iq4_xs_cuda`: grid ceil(k / 256), block 32). One block per 256 values: tid/8
//!   is the 32-value group, tid%8 the value-octet; the group scale is
//!   `((scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((scales_h >> 2*ib) & 3) << 4)` minus 32, so the
//!   table lookup is scaled by `d * (ls - 32)` rather than by `d` alone as in IQ4_NL.
//! - `iq4_xs_mmvq_<n>` = `mul_mat_vec_q<GGML_TYPE_IQ4_XS, n, has_fusion=false, small_k, false>`
//!   (mmvq.cu, table MMVQ_PARAMETERS_GENERIC): block (32, nwarps), nwarps 4 for n <= 4 and 2 for
//!   n 5..8, 1 row per CUDA block for n == 1 (4 with small_k), 2 otherwise. qi 32, vdr 4: eight
//!   threads per block, 4 * nwarps blocks per iteration, `kqs = 4 * (tid % 8)`.
//!   vec_dot_iq4_xs_q8_1: four `get_int_b4(bq4->qs, iqs + j)` words, each -> 8 table bytes via
//!   `get_int_from_table_16`, two dp4a per word against q8_1 word j and j+4 of the block at
//!   `iqs/4`, then the group scale, then `d_x * d_y * sumi`.
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

    /// `kvalues_iq4nl` as the four little-endian words `get_int_from_table_16` reads.
    pub const TABLE: [u32; 4] = [0xBFAD9881, 0xF6EADDCF, 0x26190D01, 0x71594535];
    pub const KV: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];
    /// IQ4_XS block layout: half d, u16 scales_h, 4 x u8 scales_l, 128 nibble bytes.
    pub const BLOCK_BYTES: usize = 136;
    /// Values per block (`QK_K`), 32-value groups per block, q8_1 blocks per block.
    pub const GROUPS: usize = 8;

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

    /// Raw 6-bit group scale from `scales_l` / `scales_h` (the reference subtracts 32 only in the
    /// vec-dot path, where it stays an integer multiply).
    #[inline(always)]
    pub unsafe fn group_scale6(x: *const u8, group: usize) -> u32 {
        let lo = *x.add(4 + group / 2) as u32;
        let hi = *(x.add(2) as *const u16) as u32;
        ((lo >> (4 * (group as u32 % 2))) & 0xF) | (((hi >> (2 * group as u32)) & 3) << 4)
    }

    /// `dequantize_block_iq4_xs<dst_t>`: one 256-value block per CUDA block of 32 threads.
    #[inline(always)]
    pub unsafe fn dequant<D: Dst, const MUT: u32>(vx: *const u8, yy: *mut D) {
        let i = thread::blockIdx_x() as u64;
        let tid = thread::threadIdx_x();
        let il = (tid / 8) as u64;
        let ib = (tid % 8) as u64;
        let x = vx.add((i * BLOCK_BYTES as u64) as usize);
        let y = yy.add((i * 256 + 32 * ib + 4 * il) as usize);
        let q4 = x.add(8 + 16 * ib as usize + 4 * il as usize);
        let scale = group_scale6(x, ib as usize) as i32 - 32;
        let d = mul(h2f(*(x as *const u16) as u32), scale as f32);
        let mut j = 0;
        while j < 4 {
            let q = *q4.add(j) as u32;
            // MUT 1 (gate mutant): low / high nibble swapped
            let (lo, hi) = if MUT == 1 { (q >> 4, q & 0xf) } else { (q & 0xf, q >> 4) };
            D::st(y.add(j), mul(d, kv(lo)));
            D::st(y.add(j + 16), mul(d, kv(hi)));
            j += 1;
        }
    }

    #[kernel]
    pub unsafe fn iq4_xs_dequant_f32(vx: *const u8, y: *mut f32) {
        dequant::<f32, 0>(vx, y)
    }
    /// Gate mutant (never launched by candle): nibbles swapped.
    #[kernel]
    pub unsafe fn iq4_xs_dequant_f32_mut(vx: *const u8, y: *mut f32) {
        dequant::<f32, 1>(vx, y)
    }
    #[kernel]
    pub unsafe fn iq4_xs_dequant_f16(vx: *const u8, y: *mut u16) {
        dequant::<H, 0>(vx, y as *mut H)
    }
    #[kernel]
    pub unsafe fn iq4_xs_dequant_bf16(vx: *const u8, y: *mut u16) {
        dequant::<B, 0>(vx, y as *mut B)
    }

    /// `get_int_from_table_16(q4, kvalues_iq4nl)`: (bytes of the low nibbles, of the high nibbles).
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

    /// `tmp += vec_dot_iq4_xs_q8_1(vx, yb, kbx, iqs)`: one 32-value group of block `kbx`,
    /// `iqs` = 4 * group. `yb` is the q8_1 block at `kbx * (QK_K / 32) + iqs / 4`. The reference
    /// compiles the accumulate-and-return as a single FFMA, so the product must not be rounded
    /// separately (`fma(d, sumi, acc)`), or the sum drifts a few ulp per block over K.
    #[inline(always)]
    /// Gate mutants (MUT != 0, never launched by candle): 1 = q8_1 words j / j+4 paired with the
    /// wrong nibble half, 2 = scales_h bits taken one bit too high, 3 = group bias 31 instead of 32.
    pub unsafe fn vdot_acc<const MUT: u32>(vx: *const u8, yb: *const u8, kbx: i32, iqs: i32, acc: f32) -> f32 {
        let xb = vx.offset((kbx as isize).wrapping_mul(BLOCK_BYTES as isize));
        let qs = xb.add(8 + (iqs as usize) * 4) as *const u32;
        let q8 = yb.add(4) as *const u32;
        let mut sumi = 0i32;
        let mut l = 0usize;
        while l < 4 {
            let (vx0, vy0) = if MUT == 1 { let (a, b) = table16(*qs.add(l)); (b, a) } else { table16(*qs.add(l)) };
            sumi = dp4a(vx0, *q8.add(l), sumi);
            sumi = dp4a(vy0, *q8.add(l + 4), sumi);
            l += 1;
        }
        // Reference order: the group bias is an integer multiply of the dp4a sum, and only
        // `d_x * d_y` is folded into a float -- associating it the other way drifts 1-2 ulp.
        // Reference: `sumi *= ls - 32` stays an integer multiply; only `d_x * d_y` is float.
        let g = (iqs as usize) / 4;
        let ls = if MUT == 2 {
            let lo = *xb.add(4 + g / 2) as u32;
            let hi = *(xb.add(2) as *const u16) as u32;
            (((lo >> (4 * (g as u32 % 2))) & 0xF) | (((hi >> (2 * g as u32 + 1)) & 3) << 4)) as i32 - 32
        } else {
            group_scale6(xb, g) as i32 - if MUT == 3 { 31 } else { 32 }
        };
        let sumi = sumi.wrapping_mul(ls);
        let d = mul(h2f(*(xb as *const u16) as u32), h2f(*(yb as *const u16) as u32));
        fma(d, sumi as f32, acc)
    }

    /// Compile-time loop bounds (`#[unroll]` needs literal trip counts after monomorphization;
    /// `tmp[j][i]` then stays in registers). `qi / vdr` = 8 threads per block, vdr = 4.
    pub struct K<const NC: usize, const SMALL_K: bool>;
    impl<const NC: usize, const SMALL_K: bool> K<NC, SMALL_K> {
        pub const NW: i32 = if NC <= 4 { 4 } else { 2 };
        pub const NW1: i32 = if NC <= 4 { 3 } else { 1 };
        pub const RPB: usize = if NC == 1 { if SMALL_K { 4 } else { 1 } } else { 2 };
        pub const BLOCKS_PER_ITER: i32 = 4 * Self::NW;
    }

    /// `mul_mat_vec_q<IQ4_XS, NC, false, SMALL_K, false>` (MMVQ_PARAMETERS_GENERIC).
    #[device]
    pub unsafe fn mmvq<const NC: usize, const SMALL_K: bool, const MUT: u32>(
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
        const QK: i32 = 256;
        const TPB: i32 = 8;
        let nwarps: i32 = K::<NC, SMALL_K>::NW;
        let rpb: i32 = K::<NC, SMALL_K>::RPB as i32;
        let tx = thread::threadIdx_x() as i32;
        let ty = thread::threadIdx_y() as i32;
        let tid = 32 * ty + tx;
        let row0 = rpb.wrapping_mul(thread::blockIdx_x() as i32);
        let blocks_per_row_x = (ncols_x / QK as u32) as i32;

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

        let mut kbx = tid / TPB;
        while kbx < blocks_per_row_x {
            // y block that aligns with kbx: kbx * (QK_K / QK8_1)
            let kby = 8 * kbx;
            let kqs = 4 * (tid % TPB);
            let mut j = 0;
            #[unroll]
            while j < NC {
                let yi = (j as u32).wrapping_mul(stride_col_y).wrapping_add((kby + kqs / 4) as u32);
                let yb = y.add(yi as usize * 36);
                let mut i = 0;
                #[unroll]
                while i < K::<NC, SMALL_K>::RPB {
                    let xk = kbx_offset.wrapping_add((i as u32).wrapping_mul(stride_row_x)).wrapping_add(kbx as u32) as i32;
                    tmp[j][i] = vdot_acc::<MUT>(vx, yb, xk, kqs, tmp[j][i]);
                    i += 1;
                }
                j += 1;
            }
            kbx += K::<NC, SMALL_K>::BLOCKS_PER_ITER;
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
    #[kernel] pub unsafe fn iq4_xs_mmvq_1(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<1, false, 0>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq4_xs_mmvq_1_small_k(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<1, true, 0>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq4_xs_mmvq_2(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<2, false, 0>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq4_xs_mmvq_3(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<3, false, 0>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq4_xs_mmvq_4(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<4, false, 0>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq4_xs_mmvq_5(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<5, false, 0>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq4_xs_mmvq_6(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<6, false, 0>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq4_xs_mmvq_7(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<7, false, 0>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq4_xs_mmvq_8(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<8, false, 0>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    // GENERATED KERNELS END
    // gate mutants (never launched by candle)
    #[kernel] pub unsafe fn iq4_xs_mmvq_1_mut1(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<1, false, 1>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq4_xs_mmvq_1_mut2(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<1, false, 2>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq4_xs_mmvq_1_mut3(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<1, false, 3>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
}

fn main() {
    std::process::exit(if gate::run() { 0 } else { 1 });
}
