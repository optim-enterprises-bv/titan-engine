//! GGML IQ2_S (type 22) CUDA kernels in cuda-oxide, ported from llama.cpp acecd56 under the
//! reference's own parameter ABI and bit-identical to its nvcc build (sm_120a; ref/*.cubin are the
//! same oracle the iq4_nl / iq4_xs / iq3_xxs ports gate against).
//!
//! Block layout (`block_iq2_s`, 82 bytes): `d` (half, offset 0), `qs[64]` (offset 2, grid codes in
//! the first 32 bytes / signs in the last 32), `qh[8]` (offset 66), `scales[8]` (offset 74).
//!
//! - `iq2_s_dequant_{f32,f16,bf16}` = `dequantize_block_iq2_s<dst_t>` (convert.cu, grid k/256,
//!   block 32). Per thread: `il = tid/8`, `ib = tid%8`; grid code
//!   `qs[4*ib+il] | ((qh[ib] << (8-2*il)) & 0x300)`, scale `d * (0.5 + ((scales[ib] >> 4*(il/2)) & 0xf)) * 0.25`,
//!   sign byte `qs[32 + 4*ib + il]` against `kmask_iq2xs[j]`, `j = 0..7`.
//! - `iq2_s_mmvq_<n>` = `mul_mat_vec_q<GGML_TYPE_IQ2_S, n, has_fusion=false, small_k, false>`
//!   (mmvq.cu GENERIC table: nwarps 4 for n<=4 / 2 for n 5..8, rows per block 1 for n==1 (nwarps
//!   with small_k), 2 otherwise; QR2_S 4, qi 16, vdr 2 -> `kqs = 2*(tid % 8)`).
//!   vec_dot_iq2_s_q8_1: `qs_packed` and `signs_packed` are `get_int_b2` reads at `qs + 4*(iqs/2)`
//!   and `qs + 32 + 4*(iqs/2)`; `qh` byte at `qh + iqs/2`; scales nibbles `ls0`/`ls1`. For
//!   l0 = 0,2,4,6 the grid index is `qs[l0/2] | ((qh << (8-l0)) & 0x300)` (note the shift uses l0,
//!   not 2*il as in dequantize) and the two grid words become `grid_pos[0]` / `grid_pos[1]`. l0 < 4
//!   accumulates sumi0, l0 >= 4 sumi1; final
//!   `sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0+sumi1)/2)/4`, then `d = block_d * q8_1[iqs/2].d`.
//!
//! Grid words come from the `.global` `IQ2S_GRID` table, as in iq3_s (see the note on `grid_lo`).
#![allow(non_snake_case, clippy::missing_safety_doc, clippy::too_many_arguments)]
mod gate;

use cuda_device::{SharedArray, convert, device, dotprod, kernel, ptx_asm, thread, warp};
use cuda_host::cuda_module;

/// `iq2s_grid[1024]` (ggml-common.h, u64) as little-endian (low, high) `u32` word pairs.
pub static IQ2S_GRID: [u32; 2048] = include!("iq2s_grid.rs");

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
        unsafe { ptx_asm!("mul.rn.ftz.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    /// fast-math `a + b` [FADD.FTZ].
    #[inline(always)]
    pub fn add(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("add.rn.ftz.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    /// fast-math contraction [FFMA.FTZ]: the caller's `tmp += vec_dot(...)` is one FFMA.
    #[inline(always)]
    pub fn fma(a: f32, b: f32, c: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("fma.rn.ftz.f32 %0, %1, %2, %3;", out("=f") r, in("f") a, in("f") b, in("f") c, options(register_only)); }
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
    #[inline(always)]
    pub fn dp4a(a: u32, b: u32, c: i32) -> i32 {
        dotprod::dp4a_s32(a, b, c)
    }
    /// `get_int_b2(x, i)`: two `u16` loads (the block is 82 bytes, so a `u32` load can be misaligned).
    #[inline(always)]
    pub unsafe fn get_int_b2(x: *const u8, i: i32) -> u32 {
        let p = x.add(4 * i as usize) as *const u16;
        (*p as u32) | ((*p.add(1) as u32) << 16)
    }
    /// `__vcmpne4(a, 0)`: 0xff in each byte that is non-zero (no cross-byte propagation).
    #[inline(always)]
    pub fn vcmpne4_zero(a: u32) -> u32 {
        let (x, y, z, w) = (a & 0xFF, (a >> 8) & 0xFF, (a >> 16) & 0xFF, (a >> 24) & 0xFF);
        (if x != 0 { 0xFFu32 } else { 0 })
            | (if y != 0 { 0xFF00u32 } else { 0 })
            | (if z != 0 { 0xFF_0000u32 } else { 0 })
            | (if w != 0 { 0xFF00_0000u32 } else { 0 })
    }
    /// `__vsub4(a, b)`: per-byte subtract. Each byte is independent and wraps; there is NO
    /// carry/borrow between bytes (that is what makes `(x ^ 0xff) - 0xff` a per-byte negation).
    #[inline(always)]
    pub fn vsub4(a: u32, b: u32) -> u32 {
        let (a0, a1, a2, a3) = (a & 0xFF, (a >> 8) & 0xFF, (a >> 16) & 0xFF, (a >> 24) & 0xFF);
        let (b0, b1, b2, b3) = (b & 0xFF, (b >> 8) & 0xFF, (b >> 16) & 0xFF, (b >> 24) & 0xFF);
        (a0.wrapping_sub(b0) & 0xFF)
            | ((a1.wrapping_sub(b1) & 0xFF) << 8)
            | ((a2.wrapping_sub(b2) & 0xFF) << 16)
            | ((a3.wrapping_sub(b3) & 0xFF) << 24)
    }
    /// llama.cpp fastdiv: `(__umulhi(n, mp) + n) >> L`.
    #[inline(always)]
    pub fn fastdiv(n: u32, mp: u32, l: u32) -> u32 {
        let hi = ((n as u64 * mp as u64) >> 32) as u32;
        hi.wrapping_add(n) >> l
    }

    /// Output element types (the tags are sized, or `yy.add(n)` would not advance).
    pub trait Dst: Copy {
        unsafe fn st(p: *mut Self, v: f32);
    }
    impl Dst for f32 {
        #[inline(always)]
        unsafe fn st(p: *mut f32, v: f32) {
            *p = v;
        }
    }
    #[repr(transparent)]
    #[derive(Clone, Copy)]
    pub struct H(pub u16);
    #[repr(transparent)]
    #[derive(Clone, Copy)]
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

    const BLOCK_BYTES: usize = 82; // sizeof(block_iq2_s)
    const QK: usize = 256;

    /// `dequantize_block_iq2_s<dst_t>`: one 256-value block per CUDA block of 32 threads.
    #[inline(always)]
    pub unsafe fn dequant<D: Dst>(vx: *const u8, yy: *mut D) {
        let i = thread::blockIdx_x() as usize;
        let x = vx.add(i * BLOCK_BYTES);
        let tid = thread::threadIdx_x() as usize;
        let il = tid / 8;
        let ib = tid % 8;
        let y = yy.add(i * QK + 32 * ib + 8 * il);
        // grid = iq2s_grid[qs[4*ib+il] | ((qh[ib] << (8-2*il)) & 0x300)]
        let gidx = *x.add(2 + 4 * ib + il) as u32 | (((*x.add(66 + ib) as u32) << (8 - 2 * il)) & 0x300);
        let d = mul(mul(h2f(*(x as *const u16) as u32), add(0.5, ((*x.add(74 + ib) >> (4 * (il / 2))) & 0xF) as f32)), 0.25);
        let signs = *x.add(2 + 32 + 4 * ib + il);
        // `iq2s_grid` entries are u64: `grid[j]` for j = 0..7 is the low word's bytes 0..3 followed
        // by the high word's bytes 0..3 (little-endian), so both words are needed here.
        let gl = grid_lo(gidx);
        let gh = grid_hi(gidx);
        let mut j = 0usize;
        while j < 8 {
            let s = if signs & KMASK[j] != 0 { -1.0f32 } else { 1.0f32 };
            let b = if j < 4 { (gl >> (8 * j)) & 0xFF } else { (gh >> (8 * (j - 4))) & 0xFF };
            D::st(y.add(j), mul(mul(d, b as f32), s));
            j += 1;
        }
    }

    #[kernel]
    pub unsafe fn iq2_s_dequant_f32(vx: *const u8, y: *mut f32) {
        dequant(vx, y)
    }
    #[kernel]
    pub unsafe fn iq2_s_dequant_f16(vx: *const u8, y: *mut u16) {
        dequant(vx, y as *mut H)
    }
    #[kernel]
    pub unsafe fn iq2_s_dequant_bf16(vx: *const u8, y: *mut u16) {
        dequant(vx, y as *mut B)
    }

    /// `vec_dot_iq2_s_q8_1`, accumulated into `acc` as one FFMA. `yb` = q8_1 block
    /// `kbx*(QK_K/32) + iqs/2`.
    #[inline(always)]
    pub unsafe fn vdot_acc(vx: *const u8, yb: *const u8, kbx: i32, iqs: i32, acc: f32) -> f32 {
        let xb = vx.offset((kbx as isize).wrapping_mul(BLOCK_BYTES as isize));
        let qs = xb.add(2); // block_iq2_s.qs
        let i2 = iqs / 2;
        // qs_packed / signs_packed: get_int_b2 at qs + 4*(iqs/2) and qs + 32 + 4*(iqs/2)
        let qp = get_int_b2(qs, i2);
        let sp = get_int_b2(qs.add(32), i2);
        let mut q8b = [0u8; 4];
        let mut s8 = [0u8; 4];
        let mut t = 0usize;
        while t < 4 {
            q8b[t] = (qp >> (8 * t)) as u8;
            s8[t] = (sp >> (8 * t)) as u8;
            t += 1;
        }
        let qh = *xb.add(66 + i2 as usize) as u32;
        let ls0 = (*xb.add(74 + i2 as usize) & 0x0F) as i32;
        let ls1 = (*xb.add(74 + i2 as usize) >> 4) as i32;
        let q8 = yb.add(4) as *const u32;
        let mut sumi0 = 0i32;
        let mut sumi1 = 0i32;
        let mut l0 = 0usize;
        while l0 < 8 {
            let gidx = q8b[l0 / 2] as u32 | ((qh << (8 - l0 as u32)) & 0x300);
            let gl = grid_lo(gidx);
            let gh = grid_hi(gidx);
            let sb = s8[l0 / 2] as u32;
            let s0 = vcmpne4_zero(((sb & 0x03) << 7) | ((sb & 0x0C) << 21));
            let s1 = vcmpne4_zero(((sb & 0x30) << 3) | ((sb & 0xC0) << 17));
            let g_l = vsub4(gl ^ s0, s0);
            let g_h = vsub4(gh ^ s1, s1);
            if l0 < 4 {
                sumi0 = dp4a(g_l, *q8.add(l0), sumi0);
                sumi0 = dp4a(g_h, *q8.add(l0 + 1), sumi0);
            } else {
                sumi1 = dp4a(g_l, *q8.add(l0), sumi1);
                sumi1 = dp4a(g_h, *q8.add(l0 + 1), sumi1);
            }
            l0 += 2;
        }
        let sumi = (sumi0.wrapping_mul(ls0).wrapping_add(sumi1.wrapping_mul(ls1)).wrapping_add((sumi0 + sumi1) / 2)) / 4;
        let d = mul(h2f(*(xb as *const u16) as u32), h2f(*(yb as *const u16) as u32));
        fma(d, sumi as f32, acc)
    }


    /// Compile-time loop bounds. qi/vdr = 8 threads per block, vdr = 2.
    pub struct K<const NC: usize, const SMALL_K: bool>;
    impl<const NC: usize, const SMALL_K: bool> K<NC, SMALL_K> {
        pub const NW: i32 = if NC <= 4 { 4 } else { 2 };
        pub const NW1: i32 = if NC <= 4 { 3 } else { 1 };
        pub const RPB: usize = if NC == 1 { if SMALL_K { 4 } else { 1 } } else { 2 };
        pub const BLOCKS_PER_ITER: i32 = 2 * Self::NW * 32 / 16;
    }

    /// `mul_mat_vec_q<IQ2_S, NC, false, SMALL_K, false>` (MMVQ_PARAMETERS_GENERIC).
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
        static mut TMP_SHARED: SharedArray<f32, 768> = SharedArray::UNINIT;
        const QKB: i32 = 256;
        const TPB: i32 = 8; // qi / vdr
        let nwarps = K::<NC, SMALL_K>::NW;
        let rpb = K::<NC, SMALL_K>::RPB as i32;
        let tx = thread::threadIdx_x() as i32;
        let ty = thread::threadIdx_y() as i32;
        let tid = 32 * ty + tx;
        let row0 = rpb.wrapping_mul(thread::blockIdx_x() as i32);
        let blocks_per_row_x = (ncols_x / QKB as u32) as i32;

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
            let kby = 8 * kbx;
            let kqs = 2 * (tid % TPB);
            let mut j = 0;
            #[unroll]
            while j < NC {
                let yi = (j as u32).wrapping_mul(stride_col_y).wrapping_add((kby + kqs / 2) as u32);
                let yb = y.add(yi as usize * 36);
                let mut i = 0;
                #[unroll]
                while i < K::<NC, SMALL_K>::RPB {
                    let xk = kbx_offset.wrapping_add((i as u32).wrapping_mul(stride_row_x)).wrapping_add(kbx as u32) as i32;
                    tmp[j][i] = vdot_acc(vx, yb, xk, kqs, tmp[j][i]);
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

    // GENERATED KERNELS BEGIN

    /// `iq2s_grid[i]` low word from the `.global` table (as iq3_s): the 1024-way compare chain this
    /// replaced made the dense decode ~18x slower than IQ3_S.
    #[inline(always)]
    pub fn grid_lo(i: u32) -> u32 {
        unsafe { *(&raw const IQ2S_GRID as *const u32).add(2 * (i & 1023) as usize) }
    }
    /// `iq2s_grid[i]` high word (see grid_lo).
    #[inline(always)]
    pub fn grid_hi(i: u32) -> u32 {
        unsafe { *(&raw const IQ2S_GRID as *const u32).add(2 * (i & 1023) as usize + 1) }
    }
    /// `kmask_iq2xs[8]` (indexed only by an unrolled constant, so a plain array stays in registers).
    pub const KMASK: [u8; 8] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80];
    #[kernel] pub unsafe fn iq2_s_mmvq_1(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<1, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq2_s_mmvq_1_small_k(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<1, true>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq2_s_mmvq_2(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<2, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq2_s_mmvq_3(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<3, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq2_s_mmvq_4(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<4, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq2_s_mmvq_5(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<5, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq2_s_mmvq_6(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<6, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq2_s_mmvq_7(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<7, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq2_s_mmvq_8(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<8, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    // GENERATED KERNELS END
}

fn main() {
    std::process::exit(if gate::run() { 0 } else { 1 });
}
