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
//! Grid words live as compare chains (see the note on `grid_lo`).
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

    /// `iq2s_grid[i]` low word: select chain (a runtime-indexed Rust array spills to local
    /// memory, and a host-side table never reaches the PTX at all).
    #[inline(always)]
    pub fn grid_lo(i: u32) -> u32 {
        let mut v: u32 = 0;
        if i == 0u32 { v = 0x08080808u32; }
        if i == 1u32 { v = 0x0808082bu32; }
        if i == 2u32 { v = 0x08081919u32; }
        if i == 3u32 { v = 0x08082b08u32; }
        if i == 4u32 { v = 0x08082b2bu32; }
        if i == 5u32 { v = 0x08190819u32; }
        if i == 6u32 { v = 0x08191908u32; }
        if i == 7u32 { v = 0x0819192bu32; }
        if i == 8u32 { v = 0x08192b19u32; }
        if i == 9u32 { v = 0x082b0808u32; }
        if i == 10u32 { v = 0x082b082bu32; }
        if i == 11u32 { v = 0x082b1919u32; }
        if i == 12u32 { v = 0x082b2b08u32; }
        if i == 13u32 { v = 0x19080819u32; }
        if i == 14u32 { v = 0x19081908u32; }
        if i == 15u32 { v = 0x1908192bu32; }
        if i == 16u32 { v = 0x19082b19u32; }
        if i == 17u32 { v = 0x19190808u32; }
        if i == 18u32 { v = 0x1919082bu32; }
        if i == 19u32 { v = 0x19191919u32; }
        if i == 20u32 { v = 0x19192b08u32; }
        if i == 21u32 { v = 0x192b0819u32; }
        if i == 22u32 { v = 0x192b1908u32; }
        if i == 23u32 { v = 0x192b192bu32; }
        if i == 24u32 { v = 0x192b2b19u32; }
        if i == 25u32 { v = 0x2b080808u32; }
        if i == 26u32 { v = 0x2b08082bu32; }
        if i == 27u32 { v = 0x2b081919u32; }
        if i == 28u32 { v = 0x2b082b08u32; }
        if i == 29u32 { v = 0x2b190819u32; }
        if i == 30u32 { v = 0x2b191908u32; }
        if i == 31u32 { v = 0x2b2b0808u32; }
        if i == 32u32 { v = 0x2b2b1919u32; }
        if i == 33u32 { v = 0x2b2b2b2bu32; }
        if i == 34u32 { v = 0x08080819u32; }
        if i == 35u32 { v = 0x08081908u32; }
        if i == 36u32 { v = 0x0808192bu32; }
        if i == 37u32 { v = 0x08082b19u32; }
        if i == 38u32 { v = 0x08190808u32; }
        if i == 39u32 { v = 0x0819082bu32; }
        if i == 40u32 { v = 0x08191919u32; }
        if i == 41u32 { v = 0x08192b08u32; }
        if i == 42u32 { v = 0x082b0819u32; }
        if i == 43u32 { v = 0x082b1908u32; }
        if i == 44u32 { v = 0x19080808u32; }
        if i == 45u32 { v = 0x1908082bu32; }
        if i == 46u32 { v = 0x19081919u32; }
        if i == 47u32 { v = 0x19082b08u32; }
        if i == 48u32 { v = 0x19190819u32; }
        if i == 49u32 { v = 0x19191908u32; }
        if i == 50u32 { v = 0x1919192bu32; }
        if i == 51u32 { v = 0x19192b19u32; }
        if i == 52u32 { v = 0x192b0808u32; }
        if i == 53u32 { v = 0x192b1919u32; }
        if i == 54u32 { v = 0x192b2b08u32; }
        if i == 55u32 { v = 0x2b080819u32; }
        if i == 56u32 { v = 0x2b081908u32; }
        if i == 57u32 { v = 0x2b190808u32; }
        if i == 58u32 { v = 0x2b19082bu32; }
        if i == 59u32 { v = 0x2b191919u32; }
        if i == 60u32 { v = 0x2b2b0819u32; }
        if i == 61u32 { v = 0x2b2b1908u32; }
        if i == 62u32 { v = 0x08080808u32; }
        if i == 63u32 { v = 0x0808082bu32; }
        if i == 64u32 { v = 0x08081919u32; }
        if i == 65u32 { v = 0x08082b08u32; }
        if i == 66u32 { v = 0x08190819u32; }
        if i == 67u32 { v = 0x08191908u32; }
        if i == 68u32 { v = 0x082b0808u32; }
        if i == 69u32 { v = 0x082b2b2bu32; }
        if i == 70u32 { v = 0x19080819u32; }
        if i == 71u32 { v = 0x19081908u32; }
        if i == 72u32 { v = 0x1908192bu32; }
        if i == 73u32 { v = 0x19082b19u32; }
        if i == 74u32 { v = 0x19190808u32; }
        if i == 75u32 { v = 0x19191919u32; }
        if i == 76u32 { v = 0x2b080808u32; }
        if i == 77u32 { v = 0x2b081919u32; }
        if i == 78u32 { v = 0x2b082b2bu32; }
        if i == 79u32 { v = 0x2b191908u32; }
        if i == 80u32 { v = 0x2b2b082bu32; }
        if i == 81u32 { v = 0x08080819u32; }
        if i == 82u32 { v = 0x08081908u32; }
        if i == 83u32 { v = 0x0808192bu32; }
        if i == 84u32 { v = 0x08082b19u32; }
        if i == 85u32 { v = 0x08190808u32; }
        if i == 86u32 { v = 0x0819082bu32; }
        if i == 87u32 { v = 0x08191919u32; }
        if i == 88u32 { v = 0x08192b08u32; }
        if i == 89u32 { v = 0x082b0819u32; }
        if i == 90u32 { v = 0x082b1908u32; }
        if i == 91u32 { v = 0x082b192bu32; }
        if i == 92u32 { v = 0x082b2b19u32; }
        if i == 93u32 { v = 0x19080808u32; }
        if i == 94u32 { v = 0x1908082bu32; }
        if i == 95u32 { v = 0x19081919u32; }
        if i == 96u32 { v = 0x19082b08u32; }
        if i == 97u32 { v = 0x19082b2bu32; }
        if i == 98u32 { v = 0x19190819u32; }
        if i == 99u32 { v = 0x19191908u32; }
        if i == 100u32 { v = 0x1919192bu32; }
        if i == 101u32 { v = 0x19192b19u32; }
        if i == 102u32 { v = 0x192b0808u32; }
        if i == 103u32 { v = 0x192b082bu32; }
        if i == 104u32 { v = 0x192b1919u32; }
        if i == 105u32 { v = 0x2b080819u32; }
        if i == 106u32 { v = 0x2b081908u32; }
        if i == 107u32 { v = 0x2b08192bu32; }
        if i == 108u32 { v = 0x2b082b19u32; }
        if i == 109u32 { v = 0x2b190808u32; }
        if i == 110u32 { v = 0x2b191919u32; }
        if i == 111u32 { v = 0x2b192b08u32; }
        if i == 112u32 { v = 0x2b2b0819u32; }
        if i == 113u32 { v = 0x2b2b1908u32; }
        if i == 114u32 { v = 0x08080808u32; }
        if i == 115u32 { v = 0x0808082bu32; }
        if i == 116u32 { v = 0x08081919u32; }
        if i == 117u32 { v = 0x08082b08u32; }
        if i == 118u32 { v = 0x08082b2bu32; }
        if i == 119u32 { v = 0x08190819u32; }
        if i == 120u32 { v = 0x08191908u32; }
        if i == 121u32 { v = 0x0819192bu32; }
        if i == 122u32 { v = 0x08192b19u32; }
        if i == 123u32 { v = 0x082b0808u32; }
        if i == 124u32 { v = 0x082b1919u32; }
        if i == 125u32 { v = 0x082b2b08u32; }
        if i == 126u32 { v = 0x19080819u32; }
        if i == 127u32 { v = 0x19081908u32; }
        if i == 128u32 { v = 0x1908192bu32; }
        if i == 129u32 { v = 0x19082b19u32; }
        if i == 130u32 { v = 0x19190808u32; }
        if i == 131u32 { v = 0x1919082bu32; }
        if i == 132u32 { v = 0x19191919u32; }
        if i == 133u32 { v = 0x19192b08u32; }
        if i == 134u32 { v = 0x192b0819u32; }
        if i == 135u32 { v = 0x192b1908u32; }
        if i == 136u32 { v = 0x2b080808u32; }
        if i == 137u32 { v = 0x2b08082bu32; }
        if i == 138u32 { v = 0x2b081919u32; }
        if i == 139u32 { v = 0x2b082b08u32; }
        if i == 140u32 { v = 0x2b190819u32; }
        if i == 141u32 { v = 0x2b191908u32; }
        if i == 142u32 { v = 0x2b2b0808u32; }
        if i == 143u32 { v = 0x08080819u32; }
        if i == 144u32 { v = 0x08081908u32; }
        if i == 145u32 { v = 0x0808192bu32; }
        if i == 146u32 { v = 0x08082b19u32; }
        if i == 147u32 { v = 0x08190808u32; }
        if i == 148u32 { v = 0x08191919u32; }
        if i == 149u32 { v = 0x19080808u32; }
        if i == 150u32 { v = 0x19081919u32; }
        if i == 151u32 { v = 0x19082b08u32; }
        if i == 152u32 { v = 0x19190819u32; }
        if i == 153u32 { v = 0x19191908u32; }
        if i == 154u32 { v = 0x192b0808u32; }
        if i == 155u32 { v = 0x2b080819u32; }
        if i == 156u32 { v = 0x2b081908u32; }
        if i == 157u32 { v = 0x2b190808u32; }
        if i == 158u32 { v = 0x08080808u32; }
        if i == 159u32 { v = 0x0808082bu32; }
        if i == 160u32 { v = 0x08081919u32; }
        if i == 161u32 { v = 0x08082b08u32; }
        if i == 162u32 { v = 0x08190819u32; }
        if i == 163u32 { v = 0x08191908u32; }
        if i == 164u32 { v = 0x0819192bu32; }
        if i == 165u32 { v = 0x08192b19u32; }
        if i == 166u32 { v = 0x082b0808u32; }
        if i == 167u32 { v = 0x082b1919u32; }
        if i == 168u32 { v = 0x082b2b2bu32; }
        if i == 169u32 { v = 0x19080819u32; }
        if i == 170u32 { v = 0x19081908u32; }
        if i == 171u32 { v = 0x1908192bu32; }
        if i == 172u32 { v = 0x19082b19u32; }
        if i == 173u32 { v = 0x19190808u32; }
        if i == 174u32 { v = 0x1919082bu32; }
        if i == 175u32 { v = 0x19191919u32; }
        if i == 176u32 { v = 0x19192b08u32; }
        if i == 177u32 { v = 0x192b0819u32; }
        if i == 178u32 { v = 0x192b1908u32; }
        if i == 179u32 { v = 0x2b080808u32; }
        if i == 180u32 { v = 0x2b081919u32; }
        if i == 181u32 { v = 0x2b191908u32; }
        if i == 182u32 { v = 0x2b2b2b2bu32; }
        if i == 183u32 { v = 0x08080819u32; }
        if i == 184u32 { v = 0x08081908u32; }
        if i == 185u32 { v = 0x08190808u32; }
        if i == 186u32 { v = 0x0819082bu32; }
        if i == 187u32 { v = 0x08191919u32; }
        if i == 188u32 { v = 0x08192b08u32; }
        if i == 189u32 { v = 0x082b0819u32; }
        if i == 190u32 { v = 0x19080808u32; }
        if i == 191u32 { v = 0x19081919u32; }
        if i == 192u32 { v = 0x19082b08u32; }
        if i == 193u32 { v = 0x19190819u32; }
        if i == 194u32 { v = 0x19191908u32; }
        if i == 195u32 { v = 0x192b0808u32; }
        if i == 196u32 { v = 0x2b080819u32; }
        if i == 197u32 { v = 0x2b190808u32; }
        if i == 198u32 { v = 0x08080808u32; }
        if i == 199u32 { v = 0x08190819u32; }
        if i == 200u32 { v = 0x08191908u32; }
        if i == 201u32 { v = 0x082b082bu32; }
        if i == 202u32 { v = 0x082b2b08u32; }
        if i == 203u32 { v = 0x082b2b2bu32; }
        if i == 204u32 { v = 0x19190808u32; }
        if i == 205u32 { v = 0x2b192b19u32; }
        if i == 206u32 { v = 0x08080819u32; }
        if i == 207u32 { v = 0x08081908u32; }
        if i == 208u32 { v = 0x0808192bu32; }
        if i == 209u32 { v = 0x08082b19u32; }
        if i == 210u32 { v = 0x08190808u32; }
        if i == 211u32 { v = 0x0819082bu32; }
        if i == 212u32 { v = 0x08191919u32; }
        if i == 213u32 { v = 0x08192b08u32; }
        if i == 214u32 { v = 0x082b0819u32; }
        if i == 215u32 { v = 0x082b1908u32; }
        if i == 216u32 { v = 0x082b192bu32; }
        if i == 217u32 { v = 0x19080808u32; }
        if i == 218u32 { v = 0x1908082bu32; }
        if i == 219u32 { v = 0x19081919u32; }
        if i == 220u32 { v = 0x19082b08u32; }
        if i == 221u32 { v = 0x19190819u32; }
        if i == 222u32 { v = 0x19191908u32; }
        if i == 223u32 { v = 0x1919192bu32; }
        if i == 224u32 { v = 0x19192b19u32; }
        if i == 225u32 { v = 0x192b0808u32; }
        if i == 226u32 { v = 0x192b082bu32; }
        if i == 227u32 { v = 0x192b1919u32; }
        if i == 228u32 { v = 0x192b2b08u32; }
        if i == 229u32 { v = 0x2b080819u32; }
        if i == 230u32 { v = 0x2b081908u32; }
        if i == 231u32 { v = 0x2b08192bu32; }
        if i == 232u32 { v = 0x2b190808u32; }
        if i == 233u32 { v = 0x2b191919u32; }
        if i == 234u32 { v = 0x2b192b08u32; }
        if i == 235u32 { v = 0x2b2b0819u32; }
        if i == 236u32 { v = 0x2b2b1908u32; }
        if i == 237u32 { v = 0x08080808u32; }
        if i == 238u32 { v = 0x0808082bu32; }
        if i == 239u32 { v = 0x08081919u32; }
        if i == 240u32 { v = 0x08082b08u32; }
        if i == 241u32 { v = 0x08082b2bu32; }
        if i == 242u32 { v = 0x08190819u32; }
        if i == 243u32 { v = 0x08191908u32; }
        if i == 244u32 { v = 0x0819192bu32; }
        if i == 245u32 { v = 0x08192b19u32; }
        if i == 246u32 { v = 0x082b0808u32; }
        if i == 247u32 { v = 0x082b082bu32; }
        if i == 248u32 { v = 0x082b1919u32; }
        if i == 249u32 { v = 0x082b2b08u32; }
        if i == 250u32 { v = 0x19080819u32; }
        if i == 251u32 { v = 0x19081908u32; }
        if i == 252u32 { v = 0x1908192bu32; }
        if i == 253u32 { v = 0x19082b19u32; }
        if i == 254u32 { v = 0x19190808u32; }
        if i == 255u32 { v = 0x1919082bu32; }
        if i == 256u32 { v = 0x19191919u32; }
        if i == 257u32 { v = 0x19192b08u32; }
        if i == 258u32 { v = 0x192b0819u32; }
        if i == 259u32 { v = 0x192b1908u32; }
        if i == 260u32 { v = 0x2b080808u32; }
        if i == 261u32 { v = 0x2b08082bu32; }
        if i == 262u32 { v = 0x2b081919u32; }
        if i == 263u32 { v = 0x2b082b08u32; }
        if i == 264u32 { v = 0x2b190819u32; }
        if i == 265u32 { v = 0x2b191908u32; }
        if i == 266u32 { v = 0x08080819u32; }
        if i == 267u32 { v = 0x08081908u32; }
        if i == 268u32 { v = 0x08082b19u32; }
        if i == 269u32 { v = 0x08190808u32; }
        if i == 270u32 { v = 0x08191919u32; }
        if i == 271u32 { v = 0x082b0819u32; }
        if i == 272u32 { v = 0x082b1908u32; }
        if i == 273u32 { v = 0x19080808u32; }
        if i == 274u32 { v = 0x19081919u32; }
        if i == 275u32 { v = 0x19190819u32; }
        if i == 276u32 { v = 0x19191908u32; }
        if i == 277u32 { v = 0x2b080819u32; }
        if i == 278u32 { v = 0x2b081908u32; }
        if i == 279u32 { v = 0x2b190808u32; }
        if i == 280u32 { v = 0x08080808u32; }
        if i == 281u32 { v = 0x0808082bu32; }
        if i == 282u32 { v = 0x08081919u32; }
        if i == 283u32 { v = 0x08082b08u32; }
        if i == 284u32 { v = 0x08190819u32; }
        if i == 285u32 { v = 0x08191908u32; }
        if i == 286u32 { v = 0x0819192bu32; }
        if i == 287u32 { v = 0x08192b19u32; }
        if i == 288u32 { v = 0x082b0808u32; }
        if i == 289u32 { v = 0x082b1919u32; }
        if i == 290u32 { v = 0x082b2b08u32; }
        if i == 291u32 { v = 0x19080819u32; }
        if i == 292u32 { v = 0x19081908u32; }
        if i == 293u32 { v = 0x1908192bu32; }
        if i == 294u32 { v = 0x19082b19u32; }
        if i == 295u32 { v = 0x19190808u32; }
        if i == 296u32 { v = 0x1919082bu32; }
        if i == 297u32 { v = 0x19191919u32; }
        if i == 298u32 { v = 0x19192b08u32; }
        if i == 299u32 { v = 0x192b0819u32; }
        if i == 300u32 { v = 0x192b1908u32; }
        if i == 301u32 { v = 0x2b080808u32; }
        if i == 302u32 { v = 0x2b08082bu32; }
        if i == 303u32 { v = 0x2b081919u32; }
        if i == 304u32 { v = 0x2b082b08u32; }
        if i == 305u32 { v = 0x2b190819u32; }
        if i == 306u32 { v = 0x2b191908u32; }
        if i == 307u32 { v = 0x2b2b0808u32; }
        if i == 308u32 { v = 0x08080819u32; }
        if i == 309u32 { v = 0x08081908u32; }
        if i == 310u32 { v = 0x0808192bu32; }
        if i == 311u32 { v = 0x08082b19u32; }
        if i == 312u32 { v = 0x08190808u32; }
        if i == 313u32 { v = 0x0819082bu32; }
        if i == 314u32 { v = 0x08191919u32; }
        if i == 315u32 { v = 0x08192b08u32; }
        if i == 316u32 { v = 0x082b0819u32; }
        if i == 317u32 { v = 0x082b1908u32; }
        if i == 318u32 { v = 0x19080808u32; }
        if i == 319u32 { v = 0x1908082bu32; }
        if i == 320u32 { v = 0x19081919u32; }
        if i == 321u32 { v = 0x19082b08u32; }
        if i == 322u32 { v = 0x19190819u32; }
        if i == 323u32 { v = 0x19191908u32; }
        if i == 324u32 { v = 0x192b0808u32; }
        if i == 325u32 { v = 0x2b080819u32; }
        if i == 326u32 { v = 0x2b081908u32; }
        if i == 327u32 { v = 0x2b190808u32; }
        if i == 328u32 { v = 0x08080808u32; }
        if i == 329u32 { v = 0x08081919u32; }
        if i == 330u32 { v = 0x08082b08u32; }
        if i == 331u32 { v = 0x08190819u32; }
        if i == 332u32 { v = 0x08191908u32; }
        if i == 333u32 { v = 0x082b0808u32; }
        if i == 334u32 { v = 0x19080819u32; }
        if i == 335u32 { v = 0x19081908u32; }
        if i == 336u32 { v = 0x19190808u32; }
        if i == 337u32 { v = 0x2b080808u32; }
        if i == 338u32 { v = 0x2b2b2b2bu32; }
        if i == 339u32 { v = 0x08080819u32; }
        if i == 340u32 { v = 0x08081908u32; }
        if i == 341u32 { v = 0x0808192bu32; }
        if i == 342u32 { v = 0x08082b19u32; }
        if i == 343u32 { v = 0x08190808u32; }
        if i == 344u32 { v = 0x08191919u32; }
        if i == 345u32 { v = 0x08192b08u32; }
        if i == 346u32 { v = 0x082b0819u32; }
        if i == 347u32 { v = 0x19080808u32; }
        if i == 348u32 { v = 0x1908082bu32; }
        if i == 349u32 { v = 0x19081919u32; }
        if i == 350u32 { v = 0x19082b08u32; }
        if i == 351u32 { v = 0x19190819u32; }
        if i == 352u32 { v = 0x19191908u32; }
        if i == 353u32 { v = 0x192b0808u32; }
        if i == 354u32 { v = 0x2b080819u32; }
        if i == 355u32 { v = 0x2b081908u32; }
        if i == 356u32 { v = 0x08080808u32; }
        if i == 357u32 { v = 0x0808082bu32; }
        if i == 358u32 { v = 0x08081919u32; }
        if i == 359u32 { v = 0x08082b08u32; }
        if i == 360u32 { v = 0x08190819u32; }
        if i == 361u32 { v = 0x08191908u32; }
        if i == 362u32 { v = 0x082b0808u32; }
        if i == 363u32 { v = 0x19080819u32; }
        if i == 364u32 { v = 0x19081908u32; }
        if i == 365u32 { v = 0x19190808u32; }
        if i == 366u32 { v = 0x192b2b19u32; }
        if i == 367u32 { v = 0x2b2b082bu32; }
        if i == 368u32 { v = 0x08081908u32; }
        if i == 369u32 { v = 0x08190808u32; }
        if i == 370u32 { v = 0x19080808u32; }
        if i == 371u32 { v = 0x1919192bu32; }
        if i == 372u32 { v = 0x08080808u32; }
        if i == 373u32 { v = 0x0808082bu32; }
        if i == 374u32 { v = 0x08081919u32; }
        if i == 375u32 { v = 0x08082b08u32; }
        if i == 376u32 { v = 0x08190819u32; }
        if i == 377u32 { v = 0x08191908u32; }
        if i == 378u32 { v = 0x0819192bu32; }
        if i == 379u32 { v = 0x08192b19u32; }
        if i == 380u32 { v = 0x082b0808u32; }
        if i == 381u32 { v = 0x082b1919u32; }
        if i == 382u32 { v = 0x082b2b2bu32; }
        if i == 383u32 { v = 0x19080819u32; }
        if i == 384u32 { v = 0x19081908u32; }
        if i == 385u32 { v = 0x19190808u32; }
        if i == 386u32 { v = 0x1919082bu32; }
        if i == 387u32 { v = 0x19191919u32; }
        if i == 388u32 { v = 0x192b1908u32; }
        if i == 389u32 { v = 0x2b080808u32; }
        if i == 390u32 { v = 0x2b082b2bu32; }
        if i == 391u32 { v = 0x2b191908u32; }
        if i == 392u32 { v = 0x2b2b2b2bu32; }
        if i == 393u32 { v = 0x08080819u32; }
        if i == 394u32 { v = 0x08081908u32; }
        if i == 395u32 { v = 0x08190808u32; }
        if i == 396u32 { v = 0x0819082bu32; }
        if i == 397u32 { v = 0x08191919u32; }
        if i == 398u32 { v = 0x082b0819u32; }
        if i == 399u32 { v = 0x19080808u32; }
        if i == 400u32 { v = 0x1908082bu32; }
        if i == 401u32 { v = 0x19081919u32; }
        if i == 402u32 { v = 0x19190819u32; }
        if i == 403u32 { v = 0x19191908u32; }
        if i == 404u32 { v = 0x192b0808u32; }
        if i == 405u32 { v = 0x2b080819u32; }
        if i == 406u32 { v = 0x2b081908u32; }
        if i == 407u32 { v = 0x2b190808u32; }
        if i == 408u32 { v = 0x08080808u32; }
        if i == 409u32 { v = 0x08082b2bu32; }
        if i == 410u32 { v = 0x082b082bu32; }
        if i == 411u32 { v = 0x082b2b08u32; }
        if i == 412u32 { v = 0x082b2b2bu32; }
        if i == 413u32 { v = 0x19081908u32; }
        if i == 414u32 { v = 0x19190808u32; }
        if i == 415u32 { v = 0x2b082b08u32; }
        if i == 416u32 { v = 0x2b082b2bu32; }
        if i == 417u32 { v = 0x2b2b2b08u32; }
        if i == 418u32 { v = 0x08080819u32; }
        if i == 419u32 { v = 0x08081908u32; }
        if i == 420u32 { v = 0x0808192bu32; }
        if i == 421u32 { v = 0x08082b19u32; }
        if i == 422u32 { v = 0x08190808u32; }
        if i == 423u32 { v = 0x08191919u32; }
        if i == 424u32 { v = 0x08192b08u32; }
        if i == 425u32 { v = 0x082b0819u32; }
        if i == 426u32 { v = 0x082b1908u32; }
        if i == 427u32 { v = 0x19080808u32; }
        if i == 428u32 { v = 0x1908082bu32; }
        if i == 429u32 { v = 0x19081919u32; }
        if i == 430u32 { v = 0x19082b08u32; }
        if i == 431u32 { v = 0x19190819u32; }
        if i == 432u32 { v = 0x19191908u32; }
        if i == 433u32 { v = 0x192b0808u32; }
        if i == 434u32 { v = 0x2b080819u32; }
        if i == 435u32 { v = 0x2b081908u32; }
        if i == 436u32 { v = 0x2b190808u32; }
        if i == 437u32 { v = 0x08080808u32; }
        if i == 438u32 { v = 0x08081919u32; }
        if i == 439u32 { v = 0x08082b08u32; }
        if i == 440u32 { v = 0x08190819u32; }
        if i == 441u32 { v = 0x08191908u32; }
        if i == 442u32 { v = 0x082b0808u32; }
        if i == 443u32 { v = 0x19080819u32; }
        if i == 444u32 { v = 0x19081908u32; }
        if i == 445u32 { v = 0x19190808u32; }
        if i == 446u32 { v = 0x192b192bu32; }
        if i == 447u32 { v = 0x2b080808u32; }
        if i == 448u32 { v = 0x08080819u32; }
        if i == 449u32 { v = 0x08081908u32; }
        if i == 450u32 { v = 0x08190808u32; }
        if i == 451u32 { v = 0x19080808u32; }
        if i == 452u32 { v = 0x19192b19u32; }
        if i == 453u32 { v = 0x08080808u32; }
        if i == 454u32 { v = 0x08081919u32; }
        if i == 455u32 { v = 0x08190819u32; }
        if i == 456u32 { v = 0x08191908u32; }
        if i == 457u32 { v = 0x19080819u32; }
        if i == 458u32 { v = 0x19081908u32; }
        if i == 459u32 { v = 0x19190808u32; }
        if i == 460u32 { v = 0x2b082b2bu32; }
        if i == 461u32 { v = 0x2b2b2b2bu32; }
        if i == 462u32 { v = 0x08080819u32; }
        if i == 463u32 { v = 0x08081908u32; }
        if i == 464u32 { v = 0x08190808u32; }
        if i == 465u32 { v = 0x2b191919u32; }
        if i == 466u32 { v = 0x08082b2bu32; }
        if i == 467u32 { v = 0x082b082bu32; }
        if i == 468u32 { v = 0x192b1908u32; }
        if i == 469u32 { v = 0x2b082b08u32; }
        if i == 470u32 { v = 0x2b082b2bu32; }
        if i == 471u32 { v = 0x08080819u32; }
        if i == 472u32 { v = 0x08081908u32; }
        if i == 473u32 { v = 0x0808192bu32; }
        if i == 474u32 { v = 0x08082b19u32; }
        if i == 475u32 { v = 0x08190808u32; }
        if i == 476u32 { v = 0x0819082bu32; }
        if i == 477u32 { v = 0x08191919u32; }
        if i == 478u32 { v = 0x08192b08u32; }
        if i == 479u32 { v = 0x08192b2bu32; }
        if i == 480u32 { v = 0x082b0819u32; }
        if i == 481u32 { v = 0x082b1908u32; }
        if i == 482u32 { v = 0x082b192bu32; }
        if i == 483u32 { v = 0x19080808u32; }
        if i == 484u32 { v = 0x1908082bu32; }
        if i == 485u32 { v = 0x19081919u32; }
        if i == 486u32 { v = 0x19082b08u32; }
        if i == 487u32 { v = 0x19082b2bu32; }
        if i == 488u32 { v = 0x19190819u32; }
        if i == 489u32 { v = 0x19191908u32; }
        if i == 490u32 { v = 0x1919192bu32; }
        if i == 491u32 { v = 0x19192b19u32; }
        if i == 492u32 { v = 0x192b0808u32; }
        if i == 493u32 { v = 0x192b082bu32; }
        if i == 494u32 { v = 0x192b1919u32; }
        if i == 495u32 { v = 0x2b080819u32; }
        if i == 496u32 { v = 0x2b081908u32; }
        if i == 497u32 { v = 0x2b190808u32; }
        if i == 498u32 { v = 0x2b191919u32; }
        if i == 499u32 { v = 0x2b192b08u32; }
        if i == 500u32 { v = 0x2b2b0819u32; }
        if i == 501u32 { v = 0x2b2b1908u32; }
        if i == 502u32 { v = 0x08080808u32; }
        if i == 503u32 { v = 0x0808082bu32; }
        if i == 504u32 { v = 0x08081919u32; }
        if i == 505u32 { v = 0x08082b08u32; }
        if i == 506u32 { v = 0x08190819u32; }
        if i == 507u32 { v = 0x08191908u32; }
        if i == 508u32 { v = 0x0819192bu32; }
        if i == 509u32 { v = 0x08192b19u32; }
        if i == 510u32 { v = 0x082b0808u32; }
        if i == 511u32 { v = 0x082b082bu32; }
        if i == 512u32 { v = 0x082b1919u32; }
        if i == 513u32 { v = 0x19080819u32; }
        if i == 514u32 { v = 0x19081908u32; }
        if i == 515u32 { v = 0x1908192bu32; }
        if i == 516u32 { v = 0x19082b19u32; }
        if i == 517u32 { v = 0x19190808u32; }
        if i == 518u32 { v = 0x1919082bu32; }
        if i == 519u32 { v = 0x19191919u32; }
        if i == 520u32 { v = 0x19192b08u32; }
        if i == 521u32 { v = 0x192b0819u32; }
        if i == 522u32 { v = 0x192b1908u32; }
        if i == 523u32 { v = 0x2b080808u32; }
        if i == 524u32 { v = 0x2b08082bu32; }
        if i == 525u32 { v = 0x2b081919u32; }
        if i == 526u32 { v = 0x2b082b08u32; }
        if i == 527u32 { v = 0x2b190819u32; }
        if i == 528u32 { v = 0x2b191908u32; }
        if i == 529u32 { v = 0x2b2b0808u32; }
        if i == 530u32 { v = 0x08080819u32; }
        if i == 531u32 { v = 0x08081908u32; }
        if i == 532u32 { v = 0x08190808u32; }
        if i == 533u32 { v = 0x0819082bu32; }
        if i == 534u32 { v = 0x08191919u32; }
        if i == 535u32 { v = 0x08192b08u32; }
        if i == 536u32 { v = 0x082b1908u32; }
        if i == 537u32 { v = 0x19080808u32; }
        if i == 538u32 { v = 0x19081919u32; }
        if i == 539u32 { v = 0x19082b08u32; }
        if i == 540u32 { v = 0x19190819u32; }
        if i == 541u32 { v = 0x19191908u32; }
        if i == 542u32 { v = 0x192b0808u32; }
        if i == 543u32 { v = 0x2b080819u32; }
        if i == 544u32 { v = 0x2b081908u32; }
        if i == 545u32 { v = 0x08080808u32; }
        if i == 546u32 { v = 0x0808082bu32; }
        if i == 547u32 { v = 0x08081919u32; }
        if i == 548u32 { v = 0x08082b08u32; }
        if i == 549u32 { v = 0x08082b2bu32; }
        if i == 550u32 { v = 0x08190819u32; }
        if i == 551u32 { v = 0x08191908u32; }
        if i == 552u32 { v = 0x0819192bu32; }
        if i == 553u32 { v = 0x08192b19u32; }
        if i == 554u32 { v = 0x082b0808u32; }
        if i == 555u32 { v = 0x082b082bu32; }
        if i == 556u32 { v = 0x082b1919u32; }
        if i == 557u32 { v = 0x082b2b08u32; }
        if i == 558u32 { v = 0x19080819u32; }
        if i == 559u32 { v = 0x19081908u32; }
        if i == 560u32 { v = 0x1908192bu32; }
        if i == 561u32 { v = 0x19082b19u32; }
        if i == 562u32 { v = 0x19190808u32; }
        if i == 563u32 { v = 0x1919082bu32; }
        if i == 564u32 { v = 0x19191919u32; }
        if i == 565u32 { v = 0x19192b08u32; }
        if i == 566u32 { v = 0x192b0819u32; }
        if i == 567u32 { v = 0x192b1908u32; }
        if i == 568u32 { v = 0x2b080808u32; }
        if i == 569u32 { v = 0x2b08082bu32; }
        if i == 570u32 { v = 0x2b081919u32; }
        if i == 571u32 { v = 0x2b082b08u32; }
        if i == 572u32 { v = 0x2b190819u32; }
        if i == 573u32 { v = 0x2b191908u32; }
        if i == 574u32 { v = 0x2b2b0808u32; }
        if i == 575u32 { v = 0x08080819u32; }
        if i == 576u32 { v = 0x08081908u32; }
        if i == 577u32 { v = 0x0808192bu32; }
        if i == 578u32 { v = 0x08082b19u32; }
        if i == 579u32 { v = 0x08190808u32; }
        if i == 580u32 { v = 0x0819082bu32; }
        if i == 581u32 { v = 0x08191919u32; }
        if i == 582u32 { v = 0x08192b08u32; }
        if i == 583u32 { v = 0x082b0819u32; }
        if i == 584u32 { v = 0x082b1908u32; }
        if i == 585u32 { v = 0x19080808u32; }
        if i == 586u32 { v = 0x1908082bu32; }
        if i == 587u32 { v = 0x19081919u32; }
        if i == 588u32 { v = 0x19082b08u32; }
        if i == 589u32 { v = 0x19190819u32; }
        if i == 590u32 { v = 0x19191908u32; }
        if i == 591u32 { v = 0x192b0808u32; }
        if i == 592u32 { v = 0x192b2b2bu32; }
        if i == 593u32 { v = 0x2b080819u32; }
        if i == 594u32 { v = 0x2b081908u32; }
        if i == 595u32 { v = 0x2b190808u32; }
        if i == 596u32 { v = 0x08080808u32; }
        if i == 597u32 { v = 0x0808082bu32; }
        if i == 598u32 { v = 0x08081919u32; }
        if i == 599u32 { v = 0x08082b08u32; }
        if i == 600u32 { v = 0x08190819u32; }
        if i == 601u32 { v = 0x08191908u32; }
        if i == 602u32 { v = 0x082b0808u32; }
        if i == 603u32 { v = 0x19080819u32; }
        if i == 604u32 { v = 0x19081908u32; }
        if i == 605u32 { v = 0x19190808u32; }
        if i == 606u32 { v = 0x2b080808u32; }
        if i == 607u32 { v = 0x2b2b1919u32; }
        if i == 608u32 { v = 0x08080819u32; }
        if i == 609u32 { v = 0x08081908u32; }
        if i == 610u32 { v = 0x08082b19u32; }
        if i == 611u32 { v = 0x08190808u32; }
        if i == 612u32 { v = 0x0819082bu32; }
        if i == 613u32 { v = 0x08191919u32; }
        if i == 614u32 { v = 0x08192b08u32; }
        if i == 615u32 { v = 0x082b0819u32; }
        if i == 616u32 { v = 0x082b1908u32; }
        if i == 617u32 { v = 0x19080808u32; }
        if i == 618u32 { v = 0x1908082bu32; }
        if i == 619u32 { v = 0x19081919u32; }
        if i == 620u32 { v = 0x19082b08u32; }
        if i == 621u32 { v = 0x19190819u32; }
        if i == 622u32 { v = 0x19191908u32; }
        if i == 623u32 { v = 0x192b0808u32; }
        if i == 624u32 { v = 0x2b081908u32; }
        if i == 625u32 { v = 0x2b190808u32; }
        if i == 626u32 { v = 0x08080808u32; }
        if i == 627u32 { v = 0x0808082bu32; }
        if i == 628u32 { v = 0x08081919u32; }
        if i == 629u32 { v = 0x08082b08u32; }
        if i == 630u32 { v = 0x08190819u32; }
        if i == 631u32 { v = 0x08191908u32; }
        if i == 632u32 { v = 0x082b0808u32; }
        if i == 633u32 { v = 0x19080819u32; }
        if i == 634u32 { v = 0x19081908u32; }
        if i == 635u32 { v = 0x19190808u32; }
        if i == 636u32 { v = 0x2b080808u32; }
        if i == 637u32 { v = 0x2b19192bu32; }
        if i == 638u32 { v = 0x08080819u32; }
        if i == 639u32 { v = 0x08081908u32; }
        if i == 640u32 { v = 0x08190808u32; }
        if i == 641u32 { v = 0x19080808u32; }
        if i == 642u32 { v = 0x08080808u32; }
        if i == 643u32 { v = 0x0808082bu32; }
        if i == 644u32 { v = 0x08081919u32; }
        if i == 645u32 { v = 0x08082b08u32; }
        if i == 646u32 { v = 0x08190819u32; }
        if i == 647u32 { v = 0x08191908u32; }
        if i == 648u32 { v = 0x0819192bu32; }
        if i == 649u32 { v = 0x08192b19u32; }
        if i == 650u32 { v = 0x082b0808u32; }
        if i == 651u32 { v = 0x082b082bu32; }
        if i == 652u32 { v = 0x082b1919u32; }
        if i == 653u32 { v = 0x082b2b08u32; }
        if i == 654u32 { v = 0x19080819u32; }
        if i == 655u32 { v = 0x19081908u32; }
        if i == 656u32 { v = 0x1908192bu32; }
        if i == 657u32 { v = 0x19082b19u32; }
        if i == 658u32 { v = 0x19190808u32; }
        if i == 659u32 { v = 0x1919082bu32; }
        if i == 660u32 { v = 0x19191919u32; }
        if i == 661u32 { v = 0x19192b08u32; }
        if i == 662u32 { v = 0x192b0819u32; }
        if i == 663u32 { v = 0x192b1908u32; }
        if i == 664u32 { v = 0x2b080808u32; }
        if i == 665u32 { v = 0x2b08082bu32; }
        if i == 666u32 { v = 0x2b081919u32; }
        if i == 667u32 { v = 0x2b082b08u32; }
        if i == 668u32 { v = 0x2b190819u32; }
        if i == 669u32 { v = 0x2b191908u32; }
        if i == 670u32 { v = 0x08080819u32; }
        if i == 671u32 { v = 0x08081908u32; }
        if i == 672u32 { v = 0x0808192bu32; }
        if i == 673u32 { v = 0x08082b19u32; }
        if i == 674u32 { v = 0x08190808u32; }
        if i == 675u32 { v = 0x0819082bu32; }
        if i == 676u32 { v = 0x08191919u32; }
        if i == 677u32 { v = 0x08192b08u32; }
        if i == 678u32 { v = 0x082b0819u32; }
        if i == 679u32 { v = 0x082b1908u32; }
        if i == 680u32 { v = 0x19080808u32; }
        if i == 681u32 { v = 0x1908082bu32; }
        if i == 682u32 { v = 0x19081919u32; }
        if i == 683u32 { v = 0x19082b08u32; }
        if i == 684u32 { v = 0x19190819u32; }
        if i == 685u32 { v = 0x19191908u32; }
        if i == 686u32 { v = 0x192b0808u32; }
        if i == 687u32 { v = 0x2b080819u32; }
        if i == 688u32 { v = 0x2b081908u32; }
        if i == 689u32 { v = 0x2b190808u32; }
        if i == 690u32 { v = 0x08080808u32; }
        if i == 691u32 { v = 0x08081919u32; }
        if i == 692u32 { v = 0x08082b08u32; }
        if i == 693u32 { v = 0x08190819u32; }
        if i == 694u32 { v = 0x08191908u32; }
        if i == 695u32 { v = 0x082b0808u32; }
        if i == 696u32 { v = 0x19080819u32; }
        if i == 697u32 { v = 0x19081908u32; }
        if i == 698u32 { v = 0x19190808u32; }
        if i == 699u32 { v = 0x192b2b19u32; }
        if i == 700u32 { v = 0x2b080808u32; }
        if i == 701u32 { v = 0x08080819u32; }
        if i == 702u32 { v = 0x08081908u32; }
        if i == 703u32 { v = 0x0808192bu32; }
        if i == 704u32 { v = 0x08082b19u32; }
        if i == 705u32 { v = 0x08190808u32; }
        if i == 706u32 { v = 0x0819082bu32; }
        if i == 707u32 { v = 0x08191919u32; }
        if i == 708u32 { v = 0x08192b08u32; }
        if i == 709u32 { v = 0x082b0819u32; }
        if i == 710u32 { v = 0x082b1908u32; }
        if i == 711u32 { v = 0x19080808u32; }
        if i == 712u32 { v = 0x1908082bu32; }
        if i == 713u32 { v = 0x19081919u32; }
        if i == 714u32 { v = 0x19082b08u32; }
        if i == 715u32 { v = 0x19190819u32; }
        if i == 716u32 { v = 0x19191908u32; }
        if i == 717u32 { v = 0x192b0808u32; }
        if i == 718u32 { v = 0x2b080819u32; }
        if i == 719u32 { v = 0x2b081908u32; }
        if i == 720u32 { v = 0x2b190808u32; }
        if i == 721u32 { v = 0x08080808u32; }
        if i == 722u32 { v = 0x0808082bu32; }
        if i == 723u32 { v = 0x08081919u32; }
        if i == 724u32 { v = 0x08082b08u32; }
        if i == 725u32 { v = 0x08190819u32; }
        if i == 726u32 { v = 0x08191908u32; }
        if i == 727u32 { v = 0x082b0808u32; }
        if i == 728u32 { v = 0x19080819u32; }
        if i == 729u32 { v = 0x19081908u32; }
        if i == 730u32 { v = 0x19190808u32; }
        if i == 731u32 { v = 0x2b080808u32; }
        if i == 732u32 { v = 0x08080819u32; }
        if i == 733u32 { v = 0x08081908u32; }
        if i == 734u32 { v = 0x08190808u32; }
        if i == 735u32 { v = 0x082b192bu32; }
        if i == 736u32 { v = 0x19080808u32; }
        if i == 737u32 { v = 0x08080808u32; }
        if i == 738u32 { v = 0x0808082bu32; }
        if i == 739u32 { v = 0x08081919u32; }
        if i == 740u32 { v = 0x08082b08u32; }
        if i == 741u32 { v = 0x08190819u32; }
        if i == 742u32 { v = 0x08191908u32; }
        if i == 743u32 { v = 0x082b0808u32; }
        if i == 744u32 { v = 0x19080819u32; }
        if i == 745u32 { v = 0x19081908u32; }
        if i == 746u32 { v = 0x19190808u32; }
        if i == 747u32 { v = 0x19192b2bu32; }
        if i == 748u32 { v = 0x2b080808u32; }
        if i == 749u32 { v = 0x08080819u32; }
        if i == 750u32 { v = 0x08081908u32; }
        if i == 751u32 { v = 0x08190808u32; }
        if i == 752u32 { v = 0x19080808u32; }
        if i == 753u32 { v = 0x08080808u32; }
        if i == 754u32 { v = 0x08192b19u32; }
        if i == 755u32 { v = 0x2b081919u32; }
        if i == 756u32 { v = 0x2b2b2b08u32; }
        if i == 757u32 { v = 0x08080819u32; }
        if i == 758u32 { v = 0x08081908u32; }
        if i == 759u32 { v = 0x0808192bu32; }
        if i == 760u32 { v = 0x08190808u32; }
        if i == 761u32 { v = 0x0819082bu32; }
        if i == 762u32 { v = 0x08191919u32; }
        if i == 763u32 { v = 0x08192b08u32; }
        if i == 764u32 { v = 0x082b0819u32; }
        if i == 765u32 { v = 0x082b1908u32; }
        if i == 766u32 { v = 0x19080808u32; }
        if i == 767u32 { v = 0x19081919u32; }
        if i == 768u32 { v = 0x19082b08u32; }
        if i == 769u32 { v = 0x19190819u32; }
        if i == 770u32 { v = 0x19191908u32; }
        if i == 771u32 { v = 0x192b0808u32; }
        if i == 772u32 { v = 0x2b081908u32; }
        if i == 773u32 { v = 0x2b190808u32; }
        if i == 774u32 { v = 0x08080808u32; }
        if i == 775u32 { v = 0x0808082bu32; }
        if i == 776u32 { v = 0x08081919u32; }
        if i == 777u32 { v = 0x08082b08u32; }
        if i == 778u32 { v = 0x08190819u32; }
        if i == 779u32 { v = 0x08191908u32; }
        if i == 780u32 { v = 0x082b0808u32; }
        if i == 781u32 { v = 0x19080819u32; }
        if i == 782u32 { v = 0x19081908u32; }
        if i == 783u32 { v = 0x19190808u32; }
        if i == 784u32 { v = 0x2b080808u32; }
        if i == 785u32 { v = 0x2b192b19u32; }
        if i == 786u32 { v = 0x08081908u32; }
        if i == 787u32 { v = 0x08190808u32; }
        if i == 788u32 { v = 0x19080808u32; }
        if i == 789u32 { v = 0x1919192bu32; }
        if i == 790u32 { v = 0x2b2b0819u32; }
        if i == 791u32 { v = 0x08080808u32; }
        if i == 792u32 { v = 0x08081919u32; }
        if i == 793u32 { v = 0x08082b08u32; }
        if i == 794u32 { v = 0x08190819u32; }
        if i == 795u32 { v = 0x08191908u32; }
        if i == 796u32 { v = 0x082b0808u32; }
        if i == 797u32 { v = 0x19080819u32; }
        if i == 798u32 { v = 0x19081908u32; }
        if i == 799u32 { v = 0x19190808u32; }
        if i == 800u32 { v = 0x2b080808u32; }
        if i == 801u32 { v = 0x08080819u32; }
        if i == 802u32 { v = 0x08081908u32; }
        if i == 803u32 { v = 0x08190808u32; }
        if i == 804u32 { v = 0x19080808u32; }
        if i == 805u32 { v = 0x19082b2bu32; }
        if i == 806u32 { v = 0x192b2b08u32; }
        if i == 807u32 { v = 0x2b19082bu32; }
        if i == 808u32 { v = 0x08080808u32; }
        if i == 809u32 { v = 0x2b191908u32; }
        if i == 810u32 { v = 0x08080819u32; }
        if i == 811u32 { v = 0x08081908u32; }
        if i == 812u32 { v = 0x08190808u32; }
        if i == 813u32 { v = 0x192b1919u32; }
        if i == 814u32 { v = 0x2b192b08u32; }
        if i == 815u32 { v = 0x08080808u32; }
        if i == 816u32 { v = 0x082b2b2bu32; }
        if i == 817u32 { v = 0x1908082bu32; }
        if i == 818u32 { v = 0x2b2b0819u32; }
        if i == 819u32 { v = 0x08080808u32; }
        if i == 820u32 { v = 0x0808082bu32; }
        if i == 821u32 { v = 0x08081919u32; }
        if i == 822u32 { v = 0x08082b08u32; }
        if i == 823u32 { v = 0x08190819u32; }
        if i == 824u32 { v = 0x08191908u32; }
        if i == 825u32 { v = 0x08192b19u32; }
        if i == 826u32 { v = 0x082b0808u32; }
        if i == 827u32 { v = 0x082b1919u32; }
        if i == 828u32 { v = 0x19080819u32; }
        if i == 829u32 { v = 0x19081908u32; }
        if i == 830u32 { v = 0x19190808u32; }
        if i == 831u32 { v = 0x1919082bu32; }
        if i == 832u32 { v = 0x19191919u32; }
        if i == 833u32 { v = 0x19192b08u32; }
        if i == 834u32 { v = 0x192b0819u32; }
        if i == 835u32 { v = 0x2b080808u32; }
        if i == 836u32 { v = 0x2b081919u32; }
        if i == 837u32 { v = 0x2b190819u32; }
        if i == 838u32 { v = 0x2b191908u32; }
        if i == 839u32 { v = 0x08080819u32; }
        if i == 840u32 { v = 0x08081908u32; }
        if i == 841u32 { v = 0x08082b19u32; }
        if i == 842u32 { v = 0x08190808u32; }
        if i == 843u32 { v = 0x0819082bu32; }
        if i == 844u32 { v = 0x08191919u32; }
        if i == 845u32 { v = 0x08192b08u32; }
        if i == 846u32 { v = 0x082b0819u32; }
        if i == 847u32 { v = 0x082b1908u32; }
        if i == 848u32 { v = 0x19080808u32; }
        if i == 849u32 { v = 0x1908082bu32; }
        if i == 850u32 { v = 0x19081919u32; }
        if i == 851u32 { v = 0x19082b08u32; }
        if i == 852u32 { v = 0x19190819u32; }
        if i == 853u32 { v = 0x19191908u32; }
        if i == 854u32 { v = 0x2b080819u32; }
        if i == 855u32 { v = 0x2b081908u32; }
        if i == 856u32 { v = 0x2b190808u32; }
        if i == 857u32 { v = 0x2b2b2b19u32; }
        if i == 858u32 { v = 0x08080808u32; }
        if i == 859u32 { v = 0x08081919u32; }
        if i == 860u32 { v = 0x08082b2bu32; }
        if i == 861u32 { v = 0x08190819u32; }
        if i == 862u32 { v = 0x08191908u32; }
        if i == 863u32 { v = 0x19080819u32; }
        if i == 864u32 { v = 0x19081908u32; }
        if i == 865u32 { v = 0x19190808u32; }
        if i == 866u32 { v = 0x08080819u32; }
        if i == 867u32 { v = 0x08081908u32; }
        if i == 868u32 { v = 0x0808192bu32; }
        if i == 869u32 { v = 0x08082b19u32; }
        if i == 870u32 { v = 0x08190808u32; }
        if i == 871u32 { v = 0x0819082bu32; }
        if i == 872u32 { v = 0x08191919u32; }
        if i == 873u32 { v = 0x08192b08u32; }
        if i == 874u32 { v = 0x082b0819u32; }
        if i == 875u32 { v = 0x19080808u32; }
        if i == 876u32 { v = 0x1908082bu32; }
        if i == 877u32 { v = 0x19081919u32; }
        if i == 878u32 { v = 0x19082b08u32; }
        if i == 879u32 { v = 0x19190819u32; }
        if i == 880u32 { v = 0x19191908u32; }
        if i == 881u32 { v = 0x192b0808u32; }
        if i == 882u32 { v = 0x2b080819u32; }
        if i == 883u32 { v = 0x2b081908u32; }
        if i == 884u32 { v = 0x2b190808u32; }
        if i == 885u32 { v = 0x08080808u32; }
        if i == 886u32 { v = 0x0808082bu32; }
        if i == 887u32 { v = 0x08081919u32; }
        if i == 888u32 { v = 0x08082b08u32; }
        if i == 889u32 { v = 0x08190819u32; }
        if i == 890u32 { v = 0x08191908u32; }
        if i == 891u32 { v = 0x082b0808u32; }
        if i == 892u32 { v = 0x19080819u32; }
        if i == 893u32 { v = 0x19081908u32; }
        if i == 894u32 { v = 0x19190808u32; }
        if i == 895u32 { v = 0x2b080808u32; }
        if i == 896u32 { v = 0x2b082b2bu32; }
        if i == 897u32 { v = 0x08080819u32; }
        if i == 898u32 { v = 0x08081908u32; }
        if i == 899u32 { v = 0x08190808u32; }
        if i == 900u32 { v = 0x082b2b19u32; }
        if i == 901u32 { v = 0x19080808u32; }
        if i == 902u32 { v = 0x08080808u32; }
        if i == 903u32 { v = 0x08081919u32; }
        if i == 904u32 { v = 0x08190819u32; }
        if i == 905u32 { v = 0x08191908u32; }
        if i == 906u32 { v = 0x19080819u32; }
        if i == 907u32 { v = 0x19081908u32; }
        if i == 908u32 { v = 0x19190808u32; }
        if i == 909u32 { v = 0x2b2b082bu32; }
        if i == 910u32 { v = 0x08080819u32; }
        if i == 911u32 { v = 0x08081908u32; }
        if i == 912u32 { v = 0x19080808u32; }
        if i == 913u32 { v = 0x192b1919u32; }
        if i == 914u32 { v = 0x082b082bu32; }
        if i == 915u32 { v = 0x19192b08u32; }
        if i == 916u32 { v = 0x19192b2bu32; }
        if i == 917u32 { v = 0x2b08082bu32; }
        if i == 918u32 { v = 0x2b2b082bu32; }
        if i == 919u32 { v = 0x08080819u32; }
        if i == 920u32 { v = 0x08081908u32; }
        if i == 921u32 { v = 0x08082b19u32; }
        if i == 922u32 { v = 0x08190808u32; }
        if i == 923u32 { v = 0x0819082bu32; }
        if i == 924u32 { v = 0x08191919u32; }
        if i == 925u32 { v = 0x08192b08u32; }
        if i == 926u32 { v = 0x082b1908u32; }
        if i == 927u32 { v = 0x19080808u32; }
        if i == 928u32 { v = 0x1908082bu32; }
        if i == 929u32 { v = 0x19081919u32; }
        if i == 930u32 { v = 0x19082b08u32; }
        if i == 931u32 { v = 0x19190819u32; }
        if i == 932u32 { v = 0x19191908u32; }
        if i == 933u32 { v = 0x192b0808u32; }
        if i == 934u32 { v = 0x2b080819u32; }
        if i == 935u32 { v = 0x2b081908u32; }
        if i == 936u32 { v = 0x2b190808u32; }
        if i == 937u32 { v = 0x08080808u32; }
        if i == 938u32 { v = 0x08081919u32; }
        if i == 939u32 { v = 0x08190819u32; }
        if i == 940u32 { v = 0x08191908u32; }
        if i == 941u32 { v = 0x19080819u32; }
        if i == 942u32 { v = 0x19081908u32; }
        if i == 943u32 { v = 0x19190808u32; }
        if i == 944u32 { v = 0x19192b2bu32; }
        if i == 945u32 { v = 0x08080819u32; }
        if i == 946u32 { v = 0x08081908u32; }
        if i == 947u32 { v = 0x08190808u32; }
        if i == 948u32 { v = 0x19080808u32; }
        if i == 949u32 { v = 0x2b2b192bu32; }
        if i == 950u32 { v = 0x08080808u32; }
        if i == 951u32 { v = 0x0808082bu32; }
        if i == 952u32 { v = 0x08081919u32; }
        if i == 953u32 { v = 0x08082b08u32; }
        if i == 954u32 { v = 0x08190819u32; }
        if i == 955u32 { v = 0x08191908u32; }
        if i == 956u32 { v = 0x082b0808u32; }
        if i == 957u32 { v = 0x19080819u32; }
        if i == 958u32 { v = 0x19081908u32; }
        if i == 959u32 { v = 0x19190808u32; }
        if i == 960u32 { v = 0x2b080808u32; }
        if i == 961u32 { v = 0x2b19192bu32; }
        if i == 962u32 { v = 0x08080819u32; }
        if i == 963u32 { v = 0x08081908u32; }
        if i == 964u32 { v = 0x08190808u32; }
        if i == 965u32 { v = 0x19080808u32; }
        if i == 966u32 { v = 0x2b192b08u32; }
        if i == 967u32 { v = 0x2b2b0819u32; }
        if i == 968u32 { v = 0x08080808u32; }
        if i == 969u32 { v = 0x1908192bu32; }
        if i == 970u32 { v = 0x192b1908u32; }
        if i == 971u32 { v = 0x08080819u32; }
        if i == 972u32 { v = 0x08081908u32; }
        if i == 973u32 { v = 0x08190808u32; }
        if i == 974u32 { v = 0x082b192bu32; }
        if i == 975u32 { v = 0x19080808u32; }
        if i == 976u32 { v = 0x2b2b2b19u32; }
        if i == 977u32 { v = 0x08080808u32; }
        if i == 978u32 { v = 0x19082b19u32; }
        if i == 979u32 { v = 0x1919082bu32; }
        if i == 980u32 { v = 0x2b190808u32; }
        if i == 981u32 { v = 0x08080808u32; }
        if i == 982u32 { v = 0x08081919u32; }
        if i == 983u32 { v = 0x08082b2bu32; }
        if i == 984u32 { v = 0x08191908u32; }
        if i == 985u32 { v = 0x082b082bu32; }
        if i == 986u32 { v = 0x082b2b2bu32; }
        if i == 987u32 { v = 0x19080819u32; }
        if i == 988u32 { v = 0x19081908u32; }
        if i == 989u32 { v = 0x19190808u32; }
        if i == 990u32 { v = 0x2b2b082bu32; }
        if i == 991u32 { v = 0x2b2b2b2bu32; }
        if i == 992u32 { v = 0x19080808u32; }
        if i == 993u32 { v = 0x192b1919u32; }
        if i == 994u32 { v = 0x0808082bu32; }
        if i == 995u32 { v = 0x08082b2bu32; }
        if i == 996u32 { v = 0x082b082bu32; }
        if i == 997u32 { v = 0x082b2b08u32; }
        if i == 998u32 { v = 0x082b2b2bu32; }
        if i == 999u32 { v = 0x2b08082bu32; }
        if i == 1000u32 { v = 0x2b082b08u32; }
        if i == 1001u32 { v = 0x2b082b2bu32; }
        if i == 1002u32 { v = 0x2b2b2b08u32; }
        if i == 1003u32 { v = 0x08080819u32; }
        if i == 1004u32 { v = 0x08081908u32; }
        if i == 1005u32 { v = 0x08190808u32; }
        if i == 1006u32 { v = 0x19080808u32; }
        if i == 1007u32 { v = 0x2b082b19u32; }
        if i == 1008u32 { v = 0x2b2b1908u32; }
        if i == 1009u32 { v = 0x08080808u32; }
        if i == 1010u32 { v = 0x08192b19u32; }
        if i == 1011u32 { v = 0x19190819u32; }
        if i == 1012u32 { v = 0x08082b2bu32; }
        if i == 1013u32 { v = 0x082b2b08u32; }
        if i == 1014u32 { v = 0x2b2b082bu32; }
        if i == 1015u32 { v = 0x19191908u32; }
        if i == 1016u32 { v = 0x2b08192bu32; }
        if i == 1017u32 { v = 0x08082b08u32; }
        if i == 1018u32 { v = 0x08082b2bu32; }
        if i == 1019u32 { v = 0x082b0808u32; }
        if i == 1020u32 { v = 0x082b082bu32; }
        if i == 1021u32 { v = 0x082b2b08u32; }
        if i == 1022u32 { v = 0x2b082b08u32; }
        if i == 1023u32 { v = 0x2b2b2b2bu32; }
        v
    }
    /// `iq2s_grid[i]` low word: select chain (a runtime-indexed Rust array spills to local
    /// memory, and a host-side table never reaches the PTX at all).
    #[inline(always)]
    pub fn grid_hi(i: u32) -> u32 {
        let mut v: u32 = 0;
        if i == 0u32 { v = 0x08080808u32; }
        if i == 1u32 { v = 0x08080808u32; }
        if i == 2u32 { v = 0x08080808u32; }
        if i == 3u32 { v = 0x08080808u32; }
        if i == 4u32 { v = 0x08080808u32; }
        if i == 5u32 { v = 0x08080808u32; }
        if i == 6u32 { v = 0x08080808u32; }
        if i == 7u32 { v = 0x08080808u32; }
        if i == 8u32 { v = 0x08080808u32; }
        if i == 9u32 { v = 0x08080808u32; }
        if i == 10u32 { v = 0x08080808u32; }
        if i == 11u32 { v = 0x08080808u32; }
        if i == 12u32 { v = 0x08080808u32; }
        if i == 13u32 { v = 0x08080808u32; }
        if i == 14u32 { v = 0x08080808u32; }
        if i == 15u32 { v = 0x08080808u32; }
        if i == 16u32 { v = 0x08080808u32; }
        if i == 17u32 { v = 0x08080808u32; }
        if i == 18u32 { v = 0x08080808u32; }
        if i == 19u32 { v = 0x08080808u32; }
        if i == 20u32 { v = 0x08080808u32; }
        if i == 21u32 { v = 0x08080808u32; }
        if i == 22u32 { v = 0x08080808u32; }
        if i == 23u32 { v = 0x08080808u32; }
        if i == 24u32 { v = 0x08080808u32; }
        if i == 25u32 { v = 0x08080808u32; }
        if i == 26u32 { v = 0x08080808u32; }
        if i == 27u32 { v = 0x08080808u32; }
        if i == 28u32 { v = 0x08080808u32; }
        if i == 29u32 { v = 0x08080808u32; }
        if i == 30u32 { v = 0x08080808u32; }
        if i == 31u32 { v = 0x08080808u32; }
        if i == 32u32 { v = 0x08080808u32; }
        if i == 33u32 { v = 0x08080808u32; }
        if i == 34u32 { v = 0x08080819u32; }
        if i == 35u32 { v = 0x08080819u32; }
        if i == 36u32 { v = 0x08080819u32; }
        if i == 37u32 { v = 0x08080819u32; }
        if i == 38u32 { v = 0x08080819u32; }
        if i == 39u32 { v = 0x08080819u32; }
        if i == 40u32 { v = 0x08080819u32; }
        if i == 41u32 { v = 0x08080819u32; }
        if i == 42u32 { v = 0x08080819u32; }
        if i == 43u32 { v = 0x08080819u32; }
        if i == 44u32 { v = 0x08080819u32; }
        if i == 45u32 { v = 0x08080819u32; }
        if i == 46u32 { v = 0x08080819u32; }
        if i == 47u32 { v = 0x08080819u32; }
        if i == 48u32 { v = 0x08080819u32; }
        if i == 49u32 { v = 0x08080819u32; }
        if i == 50u32 { v = 0x08080819u32; }
        if i == 51u32 { v = 0x08080819u32; }
        if i == 52u32 { v = 0x08080819u32; }
        if i == 53u32 { v = 0x08080819u32; }
        if i == 54u32 { v = 0x08080819u32; }
        if i == 55u32 { v = 0x08080819u32; }
        if i == 56u32 { v = 0x08080819u32; }
        if i == 57u32 { v = 0x08080819u32; }
        if i == 58u32 { v = 0x08080819u32; }
        if i == 59u32 { v = 0x08080819u32; }
        if i == 60u32 { v = 0x08080819u32; }
        if i == 61u32 { v = 0x08080819u32; }
        if i == 62u32 { v = 0x0808082bu32; }
        if i == 63u32 { v = 0x0808082bu32; }
        if i == 64u32 { v = 0x0808082bu32; }
        if i == 65u32 { v = 0x0808082bu32; }
        if i == 66u32 { v = 0x0808082bu32; }
        if i == 67u32 { v = 0x0808082bu32; }
        if i == 68u32 { v = 0x0808082bu32; }
        if i == 69u32 { v = 0x0808082bu32; }
        if i == 70u32 { v = 0x0808082bu32; }
        if i == 71u32 { v = 0x0808082bu32; }
        if i == 72u32 { v = 0x0808082bu32; }
        if i == 73u32 { v = 0x0808082bu32; }
        if i == 74u32 { v = 0x0808082bu32; }
        if i == 75u32 { v = 0x0808082bu32; }
        if i == 76u32 { v = 0x0808082bu32; }
        if i == 77u32 { v = 0x0808082bu32; }
        if i == 78u32 { v = 0x0808082bu32; }
        if i == 79u32 { v = 0x0808082bu32; }
        if i == 80u32 { v = 0x0808082bu32; }
        if i == 81u32 { v = 0x08081908u32; }
        if i == 82u32 { v = 0x08081908u32; }
        if i == 83u32 { v = 0x08081908u32; }
        if i == 84u32 { v = 0x08081908u32; }
        if i == 85u32 { v = 0x08081908u32; }
        if i == 86u32 { v = 0x08081908u32; }
        if i == 87u32 { v = 0x08081908u32; }
        if i == 88u32 { v = 0x08081908u32; }
        if i == 89u32 { v = 0x08081908u32; }
        if i == 90u32 { v = 0x08081908u32; }
        if i == 91u32 { v = 0x08081908u32; }
        if i == 92u32 { v = 0x08081908u32; }
        if i == 93u32 { v = 0x08081908u32; }
        if i == 94u32 { v = 0x08081908u32; }
        if i == 95u32 { v = 0x08081908u32; }
        if i == 96u32 { v = 0x08081908u32; }
        if i == 97u32 { v = 0x08081908u32; }
        if i == 98u32 { v = 0x08081908u32; }
        if i == 99u32 { v = 0x08081908u32; }
        if i == 100u32 { v = 0x08081908u32; }
        if i == 101u32 { v = 0x08081908u32; }
        if i == 102u32 { v = 0x08081908u32; }
        if i == 103u32 { v = 0x08081908u32; }
        if i == 104u32 { v = 0x08081908u32; }
        if i == 105u32 { v = 0x08081908u32; }
        if i == 106u32 { v = 0x08081908u32; }
        if i == 107u32 { v = 0x08081908u32; }
        if i == 108u32 { v = 0x08081908u32; }
        if i == 109u32 { v = 0x08081908u32; }
        if i == 110u32 { v = 0x08081908u32; }
        if i == 111u32 { v = 0x08081908u32; }
        if i == 112u32 { v = 0x08081908u32; }
        if i == 113u32 { v = 0x08081908u32; }
        if i == 114u32 { v = 0x08081919u32; }
        if i == 115u32 { v = 0x08081919u32; }
        if i == 116u32 { v = 0x08081919u32; }
        if i == 117u32 { v = 0x08081919u32; }
        if i == 118u32 { v = 0x08081919u32; }
        if i == 119u32 { v = 0x08081919u32; }
        if i == 120u32 { v = 0x08081919u32; }
        if i == 121u32 { v = 0x08081919u32; }
        if i == 122u32 { v = 0x08081919u32; }
        if i == 123u32 { v = 0x08081919u32; }
        if i == 124u32 { v = 0x08081919u32; }
        if i == 125u32 { v = 0x08081919u32; }
        if i == 126u32 { v = 0x08081919u32; }
        if i == 127u32 { v = 0x08081919u32; }
        if i == 128u32 { v = 0x08081919u32; }
        if i == 129u32 { v = 0x08081919u32; }
        if i == 130u32 { v = 0x08081919u32; }
        if i == 131u32 { v = 0x08081919u32; }
        if i == 132u32 { v = 0x08081919u32; }
        if i == 133u32 { v = 0x08081919u32; }
        if i == 134u32 { v = 0x08081919u32; }
        if i == 135u32 { v = 0x08081919u32; }
        if i == 136u32 { v = 0x08081919u32; }
        if i == 137u32 { v = 0x08081919u32; }
        if i == 138u32 { v = 0x08081919u32; }
        if i == 139u32 { v = 0x08081919u32; }
        if i == 140u32 { v = 0x08081919u32; }
        if i == 141u32 { v = 0x08081919u32; }
        if i == 142u32 { v = 0x08081919u32; }
        if i == 143u32 { v = 0x0808192bu32; }
        if i == 144u32 { v = 0x0808192bu32; }
        if i == 145u32 { v = 0x0808192bu32; }
        if i == 146u32 { v = 0x0808192bu32; }
        if i == 147u32 { v = 0x0808192bu32; }
        if i == 148u32 { v = 0x0808192bu32; }
        if i == 149u32 { v = 0x0808192bu32; }
        if i == 150u32 { v = 0x0808192bu32; }
        if i == 151u32 { v = 0x0808192bu32; }
        if i == 152u32 { v = 0x0808192bu32; }
        if i == 153u32 { v = 0x0808192bu32; }
        if i == 154u32 { v = 0x0808192bu32; }
        if i == 155u32 { v = 0x0808192bu32; }
        if i == 156u32 { v = 0x0808192bu32; }
        if i == 157u32 { v = 0x0808192bu32; }
        if i == 158u32 { v = 0x08082b08u32; }
        if i == 159u32 { v = 0x08082b08u32; }
        if i == 160u32 { v = 0x08082b08u32; }
        if i == 161u32 { v = 0x08082b08u32; }
        if i == 162u32 { v = 0x08082b08u32; }
        if i == 163u32 { v = 0x08082b08u32; }
        if i == 164u32 { v = 0x08082b08u32; }
        if i == 165u32 { v = 0x08082b08u32; }
        if i == 166u32 { v = 0x08082b08u32; }
        if i == 167u32 { v = 0x08082b08u32; }
        if i == 168u32 { v = 0x08082b08u32; }
        if i == 169u32 { v = 0x08082b08u32; }
        if i == 170u32 { v = 0x08082b08u32; }
        if i == 171u32 { v = 0x08082b08u32; }
        if i == 172u32 { v = 0x08082b08u32; }
        if i == 173u32 { v = 0x08082b08u32; }
        if i == 174u32 { v = 0x08082b08u32; }
        if i == 175u32 { v = 0x08082b08u32; }
        if i == 176u32 { v = 0x08082b08u32; }
        if i == 177u32 { v = 0x08082b08u32; }
        if i == 178u32 { v = 0x08082b08u32; }
        if i == 179u32 { v = 0x08082b08u32; }
        if i == 180u32 { v = 0x08082b08u32; }
        if i == 181u32 { v = 0x08082b08u32; }
        if i == 182u32 { v = 0x08082b08u32; }
        if i == 183u32 { v = 0x08082b19u32; }
        if i == 184u32 { v = 0x08082b19u32; }
        if i == 185u32 { v = 0x08082b19u32; }
        if i == 186u32 { v = 0x08082b19u32; }
        if i == 187u32 { v = 0x08082b19u32; }
        if i == 188u32 { v = 0x08082b19u32; }
        if i == 189u32 { v = 0x08082b19u32; }
        if i == 190u32 { v = 0x08082b19u32; }
        if i == 191u32 { v = 0x08082b19u32; }
        if i == 192u32 { v = 0x08082b19u32; }
        if i == 193u32 { v = 0x08082b19u32; }
        if i == 194u32 { v = 0x08082b19u32; }
        if i == 195u32 { v = 0x08082b19u32; }
        if i == 196u32 { v = 0x08082b19u32; }
        if i == 197u32 { v = 0x08082b19u32; }
        if i == 198u32 { v = 0x08082b2bu32; }
        if i == 199u32 { v = 0x08082b2bu32; }
        if i == 200u32 { v = 0x08082b2bu32; }
        if i == 201u32 { v = 0x08082b2bu32; }
        if i == 202u32 { v = 0x08082b2bu32; }
        if i == 203u32 { v = 0x08082b2bu32; }
        if i == 204u32 { v = 0x08082b2bu32; }
        if i == 205u32 { v = 0x08082b2bu32; }
        if i == 206u32 { v = 0x08190808u32; }
        if i == 207u32 { v = 0x08190808u32; }
        if i == 208u32 { v = 0x08190808u32; }
        if i == 209u32 { v = 0x08190808u32; }
        if i == 210u32 { v = 0x08190808u32; }
        if i == 211u32 { v = 0x08190808u32; }
        if i == 212u32 { v = 0x08190808u32; }
        if i == 213u32 { v = 0x08190808u32; }
        if i == 214u32 { v = 0x08190808u32; }
        if i == 215u32 { v = 0x08190808u32; }
        if i == 216u32 { v = 0x08190808u32; }
        if i == 217u32 { v = 0x08190808u32; }
        if i == 218u32 { v = 0x08190808u32; }
        if i == 219u32 { v = 0x08190808u32; }
        if i == 220u32 { v = 0x08190808u32; }
        if i == 221u32 { v = 0x08190808u32; }
        if i == 222u32 { v = 0x08190808u32; }
        if i == 223u32 { v = 0x08190808u32; }
        if i == 224u32 { v = 0x08190808u32; }
        if i == 225u32 { v = 0x08190808u32; }
        if i == 226u32 { v = 0x08190808u32; }
        if i == 227u32 { v = 0x08190808u32; }
        if i == 228u32 { v = 0x08190808u32; }
        if i == 229u32 { v = 0x08190808u32; }
        if i == 230u32 { v = 0x08190808u32; }
        if i == 231u32 { v = 0x08190808u32; }
        if i == 232u32 { v = 0x08190808u32; }
        if i == 233u32 { v = 0x08190808u32; }
        if i == 234u32 { v = 0x08190808u32; }
        if i == 235u32 { v = 0x08190808u32; }
        if i == 236u32 { v = 0x08190808u32; }
        if i == 237u32 { v = 0x08190819u32; }
        if i == 238u32 { v = 0x08190819u32; }
        if i == 239u32 { v = 0x08190819u32; }
        if i == 240u32 { v = 0x08190819u32; }
        if i == 241u32 { v = 0x08190819u32; }
        if i == 242u32 { v = 0x08190819u32; }
        if i == 243u32 { v = 0x08190819u32; }
        if i == 244u32 { v = 0x08190819u32; }
        if i == 245u32 { v = 0x08190819u32; }
        if i == 246u32 { v = 0x08190819u32; }
        if i == 247u32 { v = 0x08190819u32; }
        if i == 248u32 { v = 0x08190819u32; }
        if i == 249u32 { v = 0x08190819u32; }
        if i == 250u32 { v = 0x08190819u32; }
        if i == 251u32 { v = 0x08190819u32; }
        if i == 252u32 { v = 0x08190819u32; }
        if i == 253u32 { v = 0x08190819u32; }
        if i == 254u32 { v = 0x08190819u32; }
        if i == 255u32 { v = 0x08190819u32; }
        if i == 256u32 { v = 0x08190819u32; }
        if i == 257u32 { v = 0x08190819u32; }
        if i == 258u32 { v = 0x08190819u32; }
        if i == 259u32 { v = 0x08190819u32; }
        if i == 260u32 { v = 0x08190819u32; }
        if i == 261u32 { v = 0x08190819u32; }
        if i == 262u32 { v = 0x08190819u32; }
        if i == 263u32 { v = 0x08190819u32; }
        if i == 264u32 { v = 0x08190819u32; }
        if i == 265u32 { v = 0x08190819u32; }
        if i == 266u32 { v = 0x0819082bu32; }
        if i == 267u32 { v = 0x0819082bu32; }
        if i == 268u32 { v = 0x0819082bu32; }
        if i == 269u32 { v = 0x0819082bu32; }
        if i == 270u32 { v = 0x0819082bu32; }
        if i == 271u32 { v = 0x0819082bu32; }
        if i == 272u32 { v = 0x0819082bu32; }
        if i == 273u32 { v = 0x0819082bu32; }
        if i == 274u32 { v = 0x0819082bu32; }
        if i == 275u32 { v = 0x0819082bu32; }
        if i == 276u32 { v = 0x0819082bu32; }
        if i == 277u32 { v = 0x0819082bu32; }
        if i == 278u32 { v = 0x0819082bu32; }
        if i == 279u32 { v = 0x0819082bu32; }
        if i == 280u32 { v = 0x08191908u32; }
        if i == 281u32 { v = 0x08191908u32; }
        if i == 282u32 { v = 0x08191908u32; }
        if i == 283u32 { v = 0x08191908u32; }
        if i == 284u32 { v = 0x08191908u32; }
        if i == 285u32 { v = 0x08191908u32; }
        if i == 286u32 { v = 0x08191908u32; }
        if i == 287u32 { v = 0x08191908u32; }
        if i == 288u32 { v = 0x08191908u32; }
        if i == 289u32 { v = 0x08191908u32; }
        if i == 290u32 { v = 0x08191908u32; }
        if i == 291u32 { v = 0x08191908u32; }
        if i == 292u32 { v = 0x08191908u32; }
        if i == 293u32 { v = 0x08191908u32; }
        if i == 294u32 { v = 0x08191908u32; }
        if i == 295u32 { v = 0x08191908u32; }
        if i == 296u32 { v = 0x08191908u32; }
        if i == 297u32 { v = 0x08191908u32; }
        if i == 298u32 { v = 0x08191908u32; }
        if i == 299u32 { v = 0x08191908u32; }
        if i == 300u32 { v = 0x08191908u32; }
        if i == 301u32 { v = 0x08191908u32; }
        if i == 302u32 { v = 0x08191908u32; }
        if i == 303u32 { v = 0x08191908u32; }
        if i == 304u32 { v = 0x08191908u32; }
        if i == 305u32 { v = 0x08191908u32; }
        if i == 306u32 { v = 0x08191908u32; }
        if i == 307u32 { v = 0x08191908u32; }
        if i == 308u32 { v = 0x08191919u32; }
        if i == 309u32 { v = 0x08191919u32; }
        if i == 310u32 { v = 0x08191919u32; }
        if i == 311u32 { v = 0x08191919u32; }
        if i == 312u32 { v = 0x08191919u32; }
        if i == 313u32 { v = 0x08191919u32; }
        if i == 314u32 { v = 0x08191919u32; }
        if i == 315u32 { v = 0x08191919u32; }
        if i == 316u32 { v = 0x08191919u32; }
        if i == 317u32 { v = 0x08191919u32; }
        if i == 318u32 { v = 0x08191919u32; }
        if i == 319u32 { v = 0x08191919u32; }
        if i == 320u32 { v = 0x08191919u32; }
        if i == 321u32 { v = 0x08191919u32; }
        if i == 322u32 { v = 0x08191919u32; }
        if i == 323u32 { v = 0x08191919u32; }
        if i == 324u32 { v = 0x08191919u32; }
        if i == 325u32 { v = 0x08191919u32; }
        if i == 326u32 { v = 0x08191919u32; }
        if i == 327u32 { v = 0x08191919u32; }
        if i == 328u32 { v = 0x0819192bu32; }
        if i == 329u32 { v = 0x0819192bu32; }
        if i == 330u32 { v = 0x0819192bu32; }
        if i == 331u32 { v = 0x0819192bu32; }
        if i == 332u32 { v = 0x0819192bu32; }
        if i == 333u32 { v = 0x0819192bu32; }
        if i == 334u32 { v = 0x0819192bu32; }
        if i == 335u32 { v = 0x0819192bu32; }
        if i == 336u32 { v = 0x0819192bu32; }
        if i == 337u32 { v = 0x0819192bu32; }
        if i == 338u32 { v = 0x0819192bu32; }
        if i == 339u32 { v = 0x08192b08u32; }
        if i == 340u32 { v = 0x08192b08u32; }
        if i == 341u32 { v = 0x08192b08u32; }
        if i == 342u32 { v = 0x08192b08u32; }
        if i == 343u32 { v = 0x08192b08u32; }
        if i == 344u32 { v = 0x08192b08u32; }
        if i == 345u32 { v = 0x08192b08u32; }
        if i == 346u32 { v = 0x08192b08u32; }
        if i == 347u32 { v = 0x08192b08u32; }
        if i == 348u32 { v = 0x08192b08u32; }
        if i == 349u32 { v = 0x08192b08u32; }
        if i == 350u32 { v = 0x08192b08u32; }
        if i == 351u32 { v = 0x08192b08u32; }
        if i == 352u32 { v = 0x08192b08u32; }
        if i == 353u32 { v = 0x08192b08u32; }
        if i == 354u32 { v = 0x08192b08u32; }
        if i == 355u32 { v = 0x08192b08u32; }
        if i == 356u32 { v = 0x08192b19u32; }
        if i == 357u32 { v = 0x08192b19u32; }
        if i == 358u32 { v = 0x08192b19u32; }
        if i == 359u32 { v = 0x08192b19u32; }
        if i == 360u32 { v = 0x08192b19u32; }
        if i == 361u32 { v = 0x08192b19u32; }
        if i == 362u32 { v = 0x08192b19u32; }
        if i == 363u32 { v = 0x08192b19u32; }
        if i == 364u32 { v = 0x08192b19u32; }
        if i == 365u32 { v = 0x08192b19u32; }
        if i == 366u32 { v = 0x08192b19u32; }
        if i == 367u32 { v = 0x08192b19u32; }
        if i == 368u32 { v = 0x08192b2bu32; }
        if i == 369u32 { v = 0x08192b2bu32; }
        if i == 370u32 { v = 0x08192b2bu32; }
        if i == 371u32 { v = 0x08192b2bu32; }
        if i == 372u32 { v = 0x082b0808u32; }
        if i == 373u32 { v = 0x082b0808u32; }
        if i == 374u32 { v = 0x082b0808u32; }
        if i == 375u32 { v = 0x082b0808u32; }
        if i == 376u32 { v = 0x082b0808u32; }
        if i == 377u32 { v = 0x082b0808u32; }
        if i == 378u32 { v = 0x082b0808u32; }
        if i == 379u32 { v = 0x082b0808u32; }
        if i == 380u32 { v = 0x082b0808u32; }
        if i == 381u32 { v = 0x082b0808u32; }
        if i == 382u32 { v = 0x082b0808u32; }
        if i == 383u32 { v = 0x082b0808u32; }
        if i == 384u32 { v = 0x082b0808u32; }
        if i == 385u32 { v = 0x082b0808u32; }
        if i == 386u32 { v = 0x082b0808u32; }
        if i == 387u32 { v = 0x082b0808u32; }
        if i == 388u32 { v = 0x082b0808u32; }
        if i == 389u32 { v = 0x082b0808u32; }
        if i == 390u32 { v = 0x082b0808u32; }
        if i == 391u32 { v = 0x082b0808u32; }
        if i == 392u32 { v = 0x082b0808u32; }
        if i == 393u32 { v = 0x082b0819u32; }
        if i == 394u32 { v = 0x082b0819u32; }
        if i == 395u32 { v = 0x082b0819u32; }
        if i == 396u32 { v = 0x082b0819u32; }
        if i == 397u32 { v = 0x082b0819u32; }
        if i == 398u32 { v = 0x082b0819u32; }
        if i == 399u32 { v = 0x082b0819u32; }
        if i == 400u32 { v = 0x082b0819u32; }
        if i == 401u32 { v = 0x082b0819u32; }
        if i == 402u32 { v = 0x082b0819u32; }
        if i == 403u32 { v = 0x082b0819u32; }
        if i == 404u32 { v = 0x082b0819u32; }
        if i == 405u32 { v = 0x082b0819u32; }
        if i == 406u32 { v = 0x082b0819u32; }
        if i == 407u32 { v = 0x082b0819u32; }
        if i == 408u32 { v = 0x082b082bu32; }
        if i == 409u32 { v = 0x082b082bu32; }
        if i == 410u32 { v = 0x082b082bu32; }
        if i == 411u32 { v = 0x082b082bu32; }
        if i == 412u32 { v = 0x082b082bu32; }
        if i == 413u32 { v = 0x082b082bu32; }
        if i == 414u32 { v = 0x082b082bu32; }
        if i == 415u32 { v = 0x082b082bu32; }
        if i == 416u32 { v = 0x082b082bu32; }
        if i == 417u32 { v = 0x082b082bu32; }
        if i == 418u32 { v = 0x082b1908u32; }
        if i == 419u32 { v = 0x082b1908u32; }
        if i == 420u32 { v = 0x082b1908u32; }
        if i == 421u32 { v = 0x082b1908u32; }
        if i == 422u32 { v = 0x082b1908u32; }
        if i == 423u32 { v = 0x082b1908u32; }
        if i == 424u32 { v = 0x082b1908u32; }
        if i == 425u32 { v = 0x082b1908u32; }
        if i == 426u32 { v = 0x082b1908u32; }
        if i == 427u32 { v = 0x082b1908u32; }
        if i == 428u32 { v = 0x082b1908u32; }
        if i == 429u32 { v = 0x082b1908u32; }
        if i == 430u32 { v = 0x082b1908u32; }
        if i == 431u32 { v = 0x082b1908u32; }
        if i == 432u32 { v = 0x082b1908u32; }
        if i == 433u32 { v = 0x082b1908u32; }
        if i == 434u32 { v = 0x082b1908u32; }
        if i == 435u32 { v = 0x082b1908u32; }
        if i == 436u32 { v = 0x082b1908u32; }
        if i == 437u32 { v = 0x082b1919u32; }
        if i == 438u32 { v = 0x082b1919u32; }
        if i == 439u32 { v = 0x082b1919u32; }
        if i == 440u32 { v = 0x082b1919u32; }
        if i == 441u32 { v = 0x082b1919u32; }
        if i == 442u32 { v = 0x082b1919u32; }
        if i == 443u32 { v = 0x082b1919u32; }
        if i == 444u32 { v = 0x082b1919u32; }
        if i == 445u32 { v = 0x082b1919u32; }
        if i == 446u32 { v = 0x082b1919u32; }
        if i == 447u32 { v = 0x082b1919u32; }
        if i == 448u32 { v = 0x082b192bu32; }
        if i == 449u32 { v = 0x082b192bu32; }
        if i == 450u32 { v = 0x082b192bu32; }
        if i == 451u32 { v = 0x082b192bu32; }
        if i == 452u32 { v = 0x082b192bu32; }
        if i == 453u32 { v = 0x082b2b08u32; }
        if i == 454u32 { v = 0x082b2b08u32; }
        if i == 455u32 { v = 0x082b2b08u32; }
        if i == 456u32 { v = 0x082b2b08u32; }
        if i == 457u32 { v = 0x082b2b08u32; }
        if i == 458u32 { v = 0x082b2b08u32; }
        if i == 459u32 { v = 0x082b2b08u32; }
        if i == 460u32 { v = 0x082b2b08u32; }
        if i == 461u32 { v = 0x082b2b08u32; }
        if i == 462u32 { v = 0x082b2b19u32; }
        if i == 463u32 { v = 0x082b2b19u32; }
        if i == 464u32 { v = 0x082b2b19u32; }
        if i == 465u32 { v = 0x082b2b19u32; }
        if i == 466u32 { v = 0x082b2b2bu32; }
        if i == 467u32 { v = 0x082b2b2bu32; }
        if i == 468u32 { v = 0x082b2b2bu32; }
        if i == 469u32 { v = 0x082b2b2bu32; }
        if i == 470u32 { v = 0x082b2b2bu32; }
        if i == 471u32 { v = 0x19080808u32; }
        if i == 472u32 { v = 0x19080808u32; }
        if i == 473u32 { v = 0x19080808u32; }
        if i == 474u32 { v = 0x19080808u32; }
        if i == 475u32 { v = 0x19080808u32; }
        if i == 476u32 { v = 0x19080808u32; }
        if i == 477u32 { v = 0x19080808u32; }
        if i == 478u32 { v = 0x19080808u32; }
        if i == 479u32 { v = 0x19080808u32; }
        if i == 480u32 { v = 0x19080808u32; }
        if i == 481u32 { v = 0x19080808u32; }
        if i == 482u32 { v = 0x19080808u32; }
        if i == 483u32 { v = 0x19080808u32; }
        if i == 484u32 { v = 0x19080808u32; }
        if i == 485u32 { v = 0x19080808u32; }
        if i == 486u32 { v = 0x19080808u32; }
        if i == 487u32 { v = 0x19080808u32; }
        if i == 488u32 { v = 0x19080808u32; }
        if i == 489u32 { v = 0x19080808u32; }
        if i == 490u32 { v = 0x19080808u32; }
        if i == 491u32 { v = 0x19080808u32; }
        if i == 492u32 { v = 0x19080808u32; }
        if i == 493u32 { v = 0x19080808u32; }
        if i == 494u32 { v = 0x19080808u32; }
        if i == 495u32 { v = 0x19080808u32; }
        if i == 496u32 { v = 0x19080808u32; }
        if i == 497u32 { v = 0x19080808u32; }
        if i == 498u32 { v = 0x19080808u32; }
        if i == 499u32 { v = 0x19080808u32; }
        if i == 500u32 { v = 0x19080808u32; }
        if i == 501u32 { v = 0x19080808u32; }
        if i == 502u32 { v = 0x19080819u32; }
        if i == 503u32 { v = 0x19080819u32; }
        if i == 504u32 { v = 0x19080819u32; }
        if i == 505u32 { v = 0x19080819u32; }
        if i == 506u32 { v = 0x19080819u32; }
        if i == 507u32 { v = 0x19080819u32; }
        if i == 508u32 { v = 0x19080819u32; }
        if i == 509u32 { v = 0x19080819u32; }
        if i == 510u32 { v = 0x19080819u32; }
        if i == 511u32 { v = 0x19080819u32; }
        if i == 512u32 { v = 0x19080819u32; }
        if i == 513u32 { v = 0x19080819u32; }
        if i == 514u32 { v = 0x19080819u32; }
        if i == 515u32 { v = 0x19080819u32; }
        if i == 516u32 { v = 0x19080819u32; }
        if i == 517u32 { v = 0x19080819u32; }
        if i == 518u32 { v = 0x19080819u32; }
        if i == 519u32 { v = 0x19080819u32; }
        if i == 520u32 { v = 0x19080819u32; }
        if i == 521u32 { v = 0x19080819u32; }
        if i == 522u32 { v = 0x19080819u32; }
        if i == 523u32 { v = 0x19080819u32; }
        if i == 524u32 { v = 0x19080819u32; }
        if i == 525u32 { v = 0x19080819u32; }
        if i == 526u32 { v = 0x19080819u32; }
        if i == 527u32 { v = 0x19080819u32; }
        if i == 528u32 { v = 0x19080819u32; }
        if i == 529u32 { v = 0x19080819u32; }
        if i == 530u32 { v = 0x1908082bu32; }
        if i == 531u32 { v = 0x1908082bu32; }
        if i == 532u32 { v = 0x1908082bu32; }
        if i == 533u32 { v = 0x1908082bu32; }
        if i == 534u32 { v = 0x1908082bu32; }
        if i == 535u32 { v = 0x1908082bu32; }
        if i == 536u32 { v = 0x1908082bu32; }
        if i == 537u32 { v = 0x1908082bu32; }
        if i == 538u32 { v = 0x1908082bu32; }
        if i == 539u32 { v = 0x1908082bu32; }
        if i == 540u32 { v = 0x1908082bu32; }
        if i == 541u32 { v = 0x1908082bu32; }
        if i == 542u32 { v = 0x1908082bu32; }
        if i == 543u32 { v = 0x1908082bu32; }
        if i == 544u32 { v = 0x1908082bu32; }
        if i == 545u32 { v = 0x19081908u32; }
        if i == 546u32 { v = 0x19081908u32; }
        if i == 547u32 { v = 0x19081908u32; }
        if i == 548u32 { v = 0x19081908u32; }
        if i == 549u32 { v = 0x19081908u32; }
        if i == 550u32 { v = 0x19081908u32; }
        if i == 551u32 { v = 0x19081908u32; }
        if i == 552u32 { v = 0x19081908u32; }
        if i == 553u32 { v = 0x19081908u32; }
        if i == 554u32 { v = 0x19081908u32; }
        if i == 555u32 { v = 0x19081908u32; }
        if i == 556u32 { v = 0x19081908u32; }
        if i == 557u32 { v = 0x19081908u32; }
        if i == 558u32 { v = 0x19081908u32; }
        if i == 559u32 { v = 0x19081908u32; }
        if i == 560u32 { v = 0x19081908u32; }
        if i == 561u32 { v = 0x19081908u32; }
        if i == 562u32 { v = 0x19081908u32; }
        if i == 563u32 { v = 0x19081908u32; }
        if i == 564u32 { v = 0x19081908u32; }
        if i == 565u32 { v = 0x19081908u32; }
        if i == 566u32 { v = 0x19081908u32; }
        if i == 567u32 { v = 0x19081908u32; }
        if i == 568u32 { v = 0x19081908u32; }
        if i == 569u32 { v = 0x19081908u32; }
        if i == 570u32 { v = 0x19081908u32; }
        if i == 571u32 { v = 0x19081908u32; }
        if i == 572u32 { v = 0x19081908u32; }
        if i == 573u32 { v = 0x19081908u32; }
        if i == 574u32 { v = 0x19081908u32; }
        if i == 575u32 { v = 0x19081919u32; }
        if i == 576u32 { v = 0x19081919u32; }
        if i == 577u32 { v = 0x19081919u32; }
        if i == 578u32 { v = 0x19081919u32; }
        if i == 579u32 { v = 0x19081919u32; }
        if i == 580u32 { v = 0x19081919u32; }
        if i == 581u32 { v = 0x19081919u32; }
        if i == 582u32 { v = 0x19081919u32; }
        if i == 583u32 { v = 0x19081919u32; }
        if i == 584u32 { v = 0x19081919u32; }
        if i == 585u32 { v = 0x19081919u32; }
        if i == 586u32 { v = 0x19081919u32; }
        if i == 587u32 { v = 0x19081919u32; }
        if i == 588u32 { v = 0x19081919u32; }
        if i == 589u32 { v = 0x19081919u32; }
        if i == 590u32 { v = 0x19081919u32; }
        if i == 591u32 { v = 0x19081919u32; }
        if i == 592u32 { v = 0x19081919u32; }
        if i == 593u32 { v = 0x19081919u32; }
        if i == 594u32 { v = 0x19081919u32; }
        if i == 595u32 { v = 0x19081919u32; }
        if i == 596u32 { v = 0x1908192bu32; }
        if i == 597u32 { v = 0x1908192bu32; }
        if i == 598u32 { v = 0x1908192bu32; }
        if i == 599u32 { v = 0x1908192bu32; }
        if i == 600u32 { v = 0x1908192bu32; }
        if i == 601u32 { v = 0x1908192bu32; }
        if i == 602u32 { v = 0x1908192bu32; }
        if i == 603u32 { v = 0x1908192bu32; }
        if i == 604u32 { v = 0x1908192bu32; }
        if i == 605u32 { v = 0x1908192bu32; }
        if i == 606u32 { v = 0x1908192bu32; }
        if i == 607u32 { v = 0x1908192bu32; }
        if i == 608u32 { v = 0x19082b08u32; }
        if i == 609u32 { v = 0x19082b08u32; }
        if i == 610u32 { v = 0x19082b08u32; }
        if i == 611u32 { v = 0x19082b08u32; }
        if i == 612u32 { v = 0x19082b08u32; }
        if i == 613u32 { v = 0x19082b08u32; }
        if i == 614u32 { v = 0x19082b08u32; }
        if i == 615u32 { v = 0x19082b08u32; }
        if i == 616u32 { v = 0x19082b08u32; }
        if i == 617u32 { v = 0x19082b08u32; }
        if i == 618u32 { v = 0x19082b08u32; }
        if i == 619u32 { v = 0x19082b08u32; }
        if i == 620u32 { v = 0x19082b08u32; }
        if i == 621u32 { v = 0x19082b08u32; }
        if i == 622u32 { v = 0x19082b08u32; }
        if i == 623u32 { v = 0x19082b08u32; }
        if i == 624u32 { v = 0x19082b08u32; }
        if i == 625u32 { v = 0x19082b08u32; }
        if i == 626u32 { v = 0x19082b19u32; }
        if i == 627u32 { v = 0x19082b19u32; }
        if i == 628u32 { v = 0x19082b19u32; }
        if i == 629u32 { v = 0x19082b19u32; }
        if i == 630u32 { v = 0x19082b19u32; }
        if i == 631u32 { v = 0x19082b19u32; }
        if i == 632u32 { v = 0x19082b19u32; }
        if i == 633u32 { v = 0x19082b19u32; }
        if i == 634u32 { v = 0x19082b19u32; }
        if i == 635u32 { v = 0x19082b19u32; }
        if i == 636u32 { v = 0x19082b19u32; }
        if i == 637u32 { v = 0x19082b19u32; }
        if i == 638u32 { v = 0x19082b2bu32; }
        if i == 639u32 { v = 0x19082b2bu32; }
        if i == 640u32 { v = 0x19082b2bu32; }
        if i == 641u32 { v = 0x19082b2bu32; }
        if i == 642u32 { v = 0x19190808u32; }
        if i == 643u32 { v = 0x19190808u32; }
        if i == 644u32 { v = 0x19190808u32; }
        if i == 645u32 { v = 0x19190808u32; }
        if i == 646u32 { v = 0x19190808u32; }
        if i == 647u32 { v = 0x19190808u32; }
        if i == 648u32 { v = 0x19190808u32; }
        if i == 649u32 { v = 0x19190808u32; }
        if i == 650u32 { v = 0x19190808u32; }
        if i == 651u32 { v = 0x19190808u32; }
        if i == 652u32 { v = 0x19190808u32; }
        if i == 653u32 { v = 0x19190808u32; }
        if i == 654u32 { v = 0x19190808u32; }
        if i == 655u32 { v = 0x19190808u32; }
        if i == 656u32 { v = 0x19190808u32; }
        if i == 657u32 { v = 0x19190808u32; }
        if i == 658u32 { v = 0x19190808u32; }
        if i == 659u32 { v = 0x19190808u32; }
        if i == 660u32 { v = 0x19190808u32; }
        if i == 661u32 { v = 0x19190808u32; }
        if i == 662u32 { v = 0x19190808u32; }
        if i == 663u32 { v = 0x19190808u32; }
        if i == 664u32 { v = 0x19190808u32; }
        if i == 665u32 { v = 0x19190808u32; }
        if i == 666u32 { v = 0x19190808u32; }
        if i == 667u32 { v = 0x19190808u32; }
        if i == 668u32 { v = 0x19190808u32; }
        if i == 669u32 { v = 0x19190808u32; }
        if i == 670u32 { v = 0x19190819u32; }
        if i == 671u32 { v = 0x19190819u32; }
        if i == 672u32 { v = 0x19190819u32; }
        if i == 673u32 { v = 0x19190819u32; }
        if i == 674u32 { v = 0x19190819u32; }
        if i == 675u32 { v = 0x19190819u32; }
        if i == 676u32 { v = 0x19190819u32; }
        if i == 677u32 { v = 0x19190819u32; }
        if i == 678u32 { v = 0x19190819u32; }
        if i == 679u32 { v = 0x19190819u32; }
        if i == 680u32 { v = 0x19190819u32; }
        if i == 681u32 { v = 0x19190819u32; }
        if i == 682u32 { v = 0x19190819u32; }
        if i == 683u32 { v = 0x19190819u32; }
        if i == 684u32 { v = 0x19190819u32; }
        if i == 685u32 { v = 0x19190819u32; }
        if i == 686u32 { v = 0x19190819u32; }
        if i == 687u32 { v = 0x19190819u32; }
        if i == 688u32 { v = 0x19190819u32; }
        if i == 689u32 { v = 0x19190819u32; }
        if i == 690u32 { v = 0x1919082bu32; }
        if i == 691u32 { v = 0x1919082bu32; }
        if i == 692u32 { v = 0x1919082bu32; }
        if i == 693u32 { v = 0x1919082bu32; }
        if i == 694u32 { v = 0x1919082bu32; }
        if i == 695u32 { v = 0x1919082bu32; }
        if i == 696u32 { v = 0x1919082bu32; }
        if i == 697u32 { v = 0x1919082bu32; }
        if i == 698u32 { v = 0x1919082bu32; }
        if i == 699u32 { v = 0x1919082bu32; }
        if i == 700u32 { v = 0x1919082bu32; }
        if i == 701u32 { v = 0x19191908u32; }
        if i == 702u32 { v = 0x19191908u32; }
        if i == 703u32 { v = 0x19191908u32; }
        if i == 704u32 { v = 0x19191908u32; }
        if i == 705u32 { v = 0x19191908u32; }
        if i == 706u32 { v = 0x19191908u32; }
        if i == 707u32 { v = 0x19191908u32; }
        if i == 708u32 { v = 0x19191908u32; }
        if i == 709u32 { v = 0x19191908u32; }
        if i == 710u32 { v = 0x19191908u32; }
        if i == 711u32 { v = 0x19191908u32; }
        if i == 712u32 { v = 0x19191908u32; }
        if i == 713u32 { v = 0x19191908u32; }
        if i == 714u32 { v = 0x19191908u32; }
        if i == 715u32 { v = 0x19191908u32; }
        if i == 716u32 { v = 0x19191908u32; }
        if i == 717u32 { v = 0x19191908u32; }
        if i == 718u32 { v = 0x19191908u32; }
        if i == 719u32 { v = 0x19191908u32; }
        if i == 720u32 { v = 0x19191908u32; }
        if i == 721u32 { v = 0x19191919u32; }
        if i == 722u32 { v = 0x19191919u32; }
        if i == 723u32 { v = 0x19191919u32; }
        if i == 724u32 { v = 0x19191919u32; }
        if i == 725u32 { v = 0x19191919u32; }
        if i == 726u32 { v = 0x19191919u32; }
        if i == 727u32 { v = 0x19191919u32; }
        if i == 728u32 { v = 0x19191919u32; }
        if i == 729u32 { v = 0x19191919u32; }
        if i == 730u32 { v = 0x19191919u32; }
        if i == 731u32 { v = 0x19191919u32; }
        if i == 732u32 { v = 0x1919192bu32; }
        if i == 733u32 { v = 0x1919192bu32; }
        if i == 734u32 { v = 0x1919192bu32; }
        if i == 735u32 { v = 0x1919192bu32; }
        if i == 736u32 { v = 0x1919192bu32; }
        if i == 737u32 { v = 0x19192b08u32; }
        if i == 738u32 { v = 0x19192b08u32; }
        if i == 739u32 { v = 0x19192b08u32; }
        if i == 740u32 { v = 0x19192b08u32; }
        if i == 741u32 { v = 0x19192b08u32; }
        if i == 742u32 { v = 0x19192b08u32; }
        if i == 743u32 { v = 0x19192b08u32; }
        if i == 744u32 { v = 0x19192b08u32; }
        if i == 745u32 { v = 0x19192b08u32; }
        if i == 746u32 { v = 0x19192b08u32; }
        if i == 747u32 { v = 0x19192b08u32; }
        if i == 748u32 { v = 0x19192b08u32; }
        if i == 749u32 { v = 0x19192b19u32; }
        if i == 750u32 { v = 0x19192b19u32; }
        if i == 751u32 { v = 0x19192b19u32; }
        if i == 752u32 { v = 0x19192b19u32; }
        if i == 753u32 { v = 0x19192b2bu32; }
        if i == 754u32 { v = 0x19192b2bu32; }
        if i == 755u32 { v = 0x19192b2bu32; }
        if i == 756u32 { v = 0x19192b2bu32; }
        if i == 757u32 { v = 0x192b0808u32; }
        if i == 758u32 { v = 0x192b0808u32; }
        if i == 759u32 { v = 0x192b0808u32; }
        if i == 760u32 { v = 0x192b0808u32; }
        if i == 761u32 { v = 0x192b0808u32; }
        if i == 762u32 { v = 0x192b0808u32; }
        if i == 763u32 { v = 0x192b0808u32; }
        if i == 764u32 { v = 0x192b0808u32; }
        if i == 765u32 { v = 0x192b0808u32; }
        if i == 766u32 { v = 0x192b0808u32; }
        if i == 767u32 { v = 0x192b0808u32; }
        if i == 768u32 { v = 0x192b0808u32; }
        if i == 769u32 { v = 0x192b0808u32; }
        if i == 770u32 { v = 0x192b0808u32; }
        if i == 771u32 { v = 0x192b0808u32; }
        if i == 772u32 { v = 0x192b0808u32; }
        if i == 773u32 { v = 0x192b0808u32; }
        if i == 774u32 { v = 0x192b0819u32; }
        if i == 775u32 { v = 0x192b0819u32; }
        if i == 776u32 { v = 0x192b0819u32; }
        if i == 777u32 { v = 0x192b0819u32; }
        if i == 778u32 { v = 0x192b0819u32; }
        if i == 779u32 { v = 0x192b0819u32; }
        if i == 780u32 { v = 0x192b0819u32; }
        if i == 781u32 { v = 0x192b0819u32; }
        if i == 782u32 { v = 0x192b0819u32; }
        if i == 783u32 { v = 0x192b0819u32; }
        if i == 784u32 { v = 0x192b0819u32; }
        if i == 785u32 { v = 0x192b0819u32; }
        if i == 786u32 { v = 0x192b082bu32; }
        if i == 787u32 { v = 0x192b082bu32; }
        if i == 788u32 { v = 0x192b082bu32; }
        if i == 789u32 { v = 0x192b082bu32; }
        if i == 790u32 { v = 0x192b082bu32; }
        if i == 791u32 { v = 0x192b1908u32; }
        if i == 792u32 { v = 0x192b1908u32; }
        if i == 793u32 { v = 0x192b1908u32; }
        if i == 794u32 { v = 0x192b1908u32; }
        if i == 795u32 { v = 0x192b1908u32; }
        if i == 796u32 { v = 0x192b1908u32; }
        if i == 797u32 { v = 0x192b1908u32; }
        if i == 798u32 { v = 0x192b1908u32; }
        if i == 799u32 { v = 0x192b1908u32; }
        if i == 800u32 { v = 0x192b1908u32; }
        if i == 801u32 { v = 0x192b1919u32; }
        if i == 802u32 { v = 0x192b1919u32; }
        if i == 803u32 { v = 0x192b1919u32; }
        if i == 804u32 { v = 0x192b1919u32; }
        if i == 805u32 { v = 0x192b1919u32; }
        if i == 806u32 { v = 0x192b1919u32; }
        if i == 807u32 { v = 0x192b1919u32; }
        if i == 808u32 { v = 0x192b192bu32; }
        if i == 809u32 { v = 0x192b192bu32; }
        if i == 810u32 { v = 0x192b2b08u32; }
        if i == 811u32 { v = 0x192b2b08u32; }
        if i == 812u32 { v = 0x192b2b08u32; }
        if i == 813u32 { v = 0x192b2b08u32; }
        if i == 814u32 { v = 0x192b2b08u32; }
        if i == 815u32 { v = 0x192b2b19u32; }
        if i == 816u32 { v = 0x192b2b19u32; }
        if i == 817u32 { v = 0x192b2b2bu32; }
        if i == 818u32 { v = 0x192b2b2bu32; }
        if i == 819u32 { v = 0x2b080808u32; }
        if i == 820u32 { v = 0x2b080808u32; }
        if i == 821u32 { v = 0x2b080808u32; }
        if i == 822u32 { v = 0x2b080808u32; }
        if i == 823u32 { v = 0x2b080808u32; }
        if i == 824u32 { v = 0x2b080808u32; }
        if i == 825u32 { v = 0x2b080808u32; }
        if i == 826u32 { v = 0x2b080808u32; }
        if i == 827u32 { v = 0x2b080808u32; }
        if i == 828u32 { v = 0x2b080808u32; }
        if i == 829u32 { v = 0x2b080808u32; }
        if i == 830u32 { v = 0x2b080808u32; }
        if i == 831u32 { v = 0x2b080808u32; }
        if i == 832u32 { v = 0x2b080808u32; }
        if i == 833u32 { v = 0x2b080808u32; }
        if i == 834u32 { v = 0x2b080808u32; }
        if i == 835u32 { v = 0x2b080808u32; }
        if i == 836u32 { v = 0x2b080808u32; }
        if i == 837u32 { v = 0x2b080808u32; }
        if i == 838u32 { v = 0x2b080808u32; }
        if i == 839u32 { v = 0x2b080819u32; }
        if i == 840u32 { v = 0x2b080819u32; }
        if i == 841u32 { v = 0x2b080819u32; }
        if i == 842u32 { v = 0x2b080819u32; }
        if i == 843u32 { v = 0x2b080819u32; }
        if i == 844u32 { v = 0x2b080819u32; }
        if i == 845u32 { v = 0x2b080819u32; }
        if i == 846u32 { v = 0x2b080819u32; }
        if i == 847u32 { v = 0x2b080819u32; }
        if i == 848u32 { v = 0x2b080819u32; }
        if i == 849u32 { v = 0x2b080819u32; }
        if i == 850u32 { v = 0x2b080819u32; }
        if i == 851u32 { v = 0x2b080819u32; }
        if i == 852u32 { v = 0x2b080819u32; }
        if i == 853u32 { v = 0x2b080819u32; }
        if i == 854u32 { v = 0x2b080819u32; }
        if i == 855u32 { v = 0x2b080819u32; }
        if i == 856u32 { v = 0x2b080819u32; }
        if i == 857u32 { v = 0x2b080819u32; }
        if i == 858u32 { v = 0x2b08082bu32; }
        if i == 859u32 { v = 0x2b08082bu32; }
        if i == 860u32 { v = 0x2b08082bu32; }
        if i == 861u32 { v = 0x2b08082bu32; }
        if i == 862u32 { v = 0x2b08082bu32; }
        if i == 863u32 { v = 0x2b08082bu32; }
        if i == 864u32 { v = 0x2b08082bu32; }
        if i == 865u32 { v = 0x2b08082bu32; }
        if i == 866u32 { v = 0x2b081908u32; }
        if i == 867u32 { v = 0x2b081908u32; }
        if i == 868u32 { v = 0x2b081908u32; }
        if i == 869u32 { v = 0x2b081908u32; }
        if i == 870u32 { v = 0x2b081908u32; }
        if i == 871u32 { v = 0x2b081908u32; }
        if i == 872u32 { v = 0x2b081908u32; }
        if i == 873u32 { v = 0x2b081908u32; }
        if i == 874u32 { v = 0x2b081908u32; }
        if i == 875u32 { v = 0x2b081908u32; }
        if i == 876u32 { v = 0x2b081908u32; }
        if i == 877u32 { v = 0x2b081908u32; }
        if i == 878u32 { v = 0x2b081908u32; }
        if i == 879u32 { v = 0x2b081908u32; }
        if i == 880u32 { v = 0x2b081908u32; }
        if i == 881u32 { v = 0x2b081908u32; }
        if i == 882u32 { v = 0x2b081908u32; }
        if i == 883u32 { v = 0x2b081908u32; }
        if i == 884u32 { v = 0x2b081908u32; }
        if i == 885u32 { v = 0x2b081919u32; }
        if i == 886u32 { v = 0x2b081919u32; }
        if i == 887u32 { v = 0x2b081919u32; }
        if i == 888u32 { v = 0x2b081919u32; }
        if i == 889u32 { v = 0x2b081919u32; }
        if i == 890u32 { v = 0x2b081919u32; }
        if i == 891u32 { v = 0x2b081919u32; }
        if i == 892u32 { v = 0x2b081919u32; }
        if i == 893u32 { v = 0x2b081919u32; }
        if i == 894u32 { v = 0x2b081919u32; }
        if i == 895u32 { v = 0x2b081919u32; }
        if i == 896u32 { v = 0x2b081919u32; }
        if i == 897u32 { v = 0x2b08192bu32; }
        if i == 898u32 { v = 0x2b08192bu32; }
        if i == 899u32 { v = 0x2b08192bu32; }
        if i == 900u32 { v = 0x2b08192bu32; }
        if i == 901u32 { v = 0x2b08192bu32; }
        if i == 902u32 { v = 0x2b082b08u32; }
        if i == 903u32 { v = 0x2b082b08u32; }
        if i == 904u32 { v = 0x2b082b08u32; }
        if i == 905u32 { v = 0x2b082b08u32; }
        if i == 906u32 { v = 0x2b082b08u32; }
        if i == 907u32 { v = 0x2b082b08u32; }
        if i == 908u32 { v = 0x2b082b08u32; }
        if i == 909u32 { v = 0x2b082b08u32; }
        if i == 910u32 { v = 0x2b082b19u32; }
        if i == 911u32 { v = 0x2b082b19u32; }
        if i == 912u32 { v = 0x2b082b19u32; }
        if i == 913u32 { v = 0x2b082b19u32; }
        if i == 914u32 { v = 0x2b082b2bu32; }
        if i == 915u32 { v = 0x2b082b2bu32; }
        if i == 916u32 { v = 0x2b082b2bu32; }
        if i == 917u32 { v = 0x2b082b2bu32; }
        if i == 918u32 { v = 0x2b082b2bu32; }
        if i == 919u32 { v = 0x2b190808u32; }
        if i == 920u32 { v = 0x2b190808u32; }
        if i == 921u32 { v = 0x2b190808u32; }
        if i == 922u32 { v = 0x2b190808u32; }
        if i == 923u32 { v = 0x2b190808u32; }
        if i == 924u32 { v = 0x2b190808u32; }
        if i == 925u32 { v = 0x2b190808u32; }
        if i == 926u32 { v = 0x2b190808u32; }
        if i == 927u32 { v = 0x2b190808u32; }
        if i == 928u32 { v = 0x2b190808u32; }
        if i == 929u32 { v = 0x2b190808u32; }
        if i == 930u32 { v = 0x2b190808u32; }
        if i == 931u32 { v = 0x2b190808u32; }
        if i == 932u32 { v = 0x2b190808u32; }
        if i == 933u32 { v = 0x2b190808u32; }
        if i == 934u32 { v = 0x2b190808u32; }
        if i == 935u32 { v = 0x2b190808u32; }
        if i == 936u32 { v = 0x2b190808u32; }
        if i == 937u32 { v = 0x2b190819u32; }
        if i == 938u32 { v = 0x2b190819u32; }
        if i == 939u32 { v = 0x2b190819u32; }
        if i == 940u32 { v = 0x2b190819u32; }
        if i == 941u32 { v = 0x2b190819u32; }
        if i == 942u32 { v = 0x2b190819u32; }
        if i == 943u32 { v = 0x2b190819u32; }
        if i == 944u32 { v = 0x2b190819u32; }
        if i == 945u32 { v = 0x2b19082bu32; }
        if i == 946u32 { v = 0x2b19082bu32; }
        if i == 947u32 { v = 0x2b19082bu32; }
        if i == 948u32 { v = 0x2b19082bu32; }
        if i == 949u32 { v = 0x2b19082bu32; }
        if i == 950u32 { v = 0x2b191908u32; }
        if i == 951u32 { v = 0x2b191908u32; }
        if i == 952u32 { v = 0x2b191908u32; }
        if i == 953u32 { v = 0x2b191908u32; }
        if i == 954u32 { v = 0x2b191908u32; }
        if i == 955u32 { v = 0x2b191908u32; }
        if i == 956u32 { v = 0x2b191908u32; }
        if i == 957u32 { v = 0x2b191908u32; }
        if i == 958u32 { v = 0x2b191908u32; }
        if i == 959u32 { v = 0x2b191908u32; }
        if i == 960u32 { v = 0x2b191908u32; }
        if i == 961u32 { v = 0x2b191908u32; }
        if i == 962u32 { v = 0x2b191919u32; }
        if i == 963u32 { v = 0x2b191919u32; }
        if i == 964u32 { v = 0x2b191919u32; }
        if i == 965u32 { v = 0x2b191919u32; }
        if i == 966u32 { v = 0x2b191919u32; }
        if i == 967u32 { v = 0x2b191919u32; }
        if i == 968u32 { v = 0x2b19192bu32; }
        if i == 969u32 { v = 0x2b19192bu32; }
        if i == 970u32 { v = 0x2b19192bu32; }
        if i == 971u32 { v = 0x2b192b08u32; }
        if i == 972u32 { v = 0x2b192b08u32; }
        if i == 973u32 { v = 0x2b192b08u32; }
        if i == 974u32 { v = 0x2b192b08u32; }
        if i == 975u32 { v = 0x2b192b08u32; }
        if i == 976u32 { v = 0x2b192b08u32; }
        if i == 977u32 { v = 0x2b192b19u32; }
        if i == 978u32 { v = 0x2b192b19u32; }
        if i == 979u32 { v = 0x2b192b19u32; }
        if i == 980u32 { v = 0x2b192b2bu32; }
        if i == 981u32 { v = 0x2b2b0808u32; }
        if i == 982u32 { v = 0x2b2b0808u32; }
        if i == 983u32 { v = 0x2b2b0808u32; }
        if i == 984u32 { v = 0x2b2b0808u32; }
        if i == 985u32 { v = 0x2b2b0808u32; }
        if i == 986u32 { v = 0x2b2b0808u32; }
        if i == 987u32 { v = 0x2b2b0808u32; }
        if i == 988u32 { v = 0x2b2b0808u32; }
        if i == 989u32 { v = 0x2b2b0808u32; }
        if i == 990u32 { v = 0x2b2b0808u32; }
        if i == 991u32 { v = 0x2b2b0808u32; }
        if i == 992u32 { v = 0x2b2b0819u32; }
        if i == 993u32 { v = 0x2b2b0819u32; }
        if i == 994u32 { v = 0x2b2b082bu32; }
        if i == 995u32 { v = 0x2b2b082bu32; }
        if i == 996u32 { v = 0x2b2b082bu32; }
        if i == 997u32 { v = 0x2b2b082bu32; }
        if i == 998u32 { v = 0x2b2b082bu32; }
        if i == 999u32 { v = 0x2b2b082bu32; }
        if i == 1000u32 { v = 0x2b2b082bu32; }
        if i == 1001u32 { v = 0x2b2b082bu32; }
        if i == 1002u32 { v = 0x2b2b082bu32; }
        if i == 1003u32 { v = 0x2b2b1908u32; }
        if i == 1004u32 { v = 0x2b2b1908u32; }
        if i == 1005u32 { v = 0x2b2b1908u32; }
        if i == 1006u32 { v = 0x2b2b1908u32; }
        if i == 1007u32 { v = 0x2b2b1908u32; }
        if i == 1008u32 { v = 0x2b2b1908u32; }
        if i == 1009u32 { v = 0x2b2b1919u32; }
        if i == 1010u32 { v = 0x2b2b1919u32; }
        if i == 1011u32 { v = 0x2b2b192bu32; }
        if i == 1012u32 { v = 0x2b2b2b08u32; }
        if i == 1013u32 { v = 0x2b2b2b08u32; }
        if i == 1014u32 { v = 0x2b2b2b08u32; }
        if i == 1015u32 { v = 0x2b2b2b19u32; }
        if i == 1016u32 { v = 0x2b2b2b19u32; }
        if i == 1017u32 { v = 0x2b2b2b2bu32; }
        if i == 1018u32 { v = 0x2b2b2b2bu32; }
        if i == 1019u32 { v = 0x2b2b2b2bu32; }
        if i == 1020u32 { v = 0x2b2b2b2bu32; }
        if i == 1021u32 { v = 0x2b2b2b2bu32; }
        if i == 1022u32 { v = 0x2b2b2b2bu32; }
        if i == 1023u32 { v = 0x2b2b2b2bu32; }
        v
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
