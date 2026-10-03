//! GGML IQ2_XXS (type 16) CUDA kernels in cuda-oxide, ported from llama.cpp acecd56 under the
//! reference's own parameter ABI and checked bit-identical against its nvcc build (sm_120a; ref/*.cubin
//! are the same oracle the iq4_nl / iq4_xs / iq3_xxs / iq2_s ports gate against).
//!
//! Block layout (`block_iq2_xxs`, 66 bytes): `d` (half, offset 0) then `uint16_t qs[QK_K/8]` (offset 2,
//! 64 bytes). A `uint16_t` unit holds four grid indices; the top 3 bits of every fourth index are the
//! shared scale `ls`.
//!
//! - `iq2_xxs_dequant_{f32,f16,bf16}` = `dequantize_block_iq2_xxs<dst_t>` (convert.cu, grid k/256,
//!   block 32). Per thread `il = tid/8` (0..3), `ib = tid%8`; `q2 = qs + 4*ib`;
//!   `grid = iq2xxs_grid[((uint8_t*)q2)[il]]` (a u64 entry read as 8 bytes);
//!   `aux32 = q2[2] | (q2[3] << 16)`; `d = block_d * (0.5 + (aux32 >> 28)) * 0.25`;
//!   signs = `ksigns_iq2xs[(aux32 >> 7*il) & 127]` against `kmask_iq2xs[j]`, `j = 0..7`.
//! - `iq2_xxs_mmvq_<n>` = `mul_mat_vec_q<GGML_TYPE_IQ2_XXS, n, has_fusion=false, small_k, false>`
//!   (mmvq.cu GENERIC table: nwarps 4 for n<=4 / 2 for n 5..8, rows per block 1 for n==1 (nwarps with
//!   small_k), 2 otherwise; QR2_XXS 4, qi 16, vdr 2 -> `blocks_per_iter = 2*nwarps*32/16`,
//!   `kqs = 2*(tid % 8)`).
//!   vec_dot_iq2_xxs_q8_1: `q2 = get_int_b2(qs, iqs)`, `aux8 = bytes(q2)`,
//!   `aux32 = get_int_b2(qs, iqs + 1)`. For k0 = 0,2,4,6: `grid_pos` = the two u32 words of
//!   `iq2xxs_grid[aux8[k0/2]]`, `signs = unpack_ksigns(aux32 >> (7*k0/2))`,
//!   `signs0 = (signs & 0x08040201) != 0 per byte`, `grid0 = grid_pos.x ^ signs0 - signs0` (per byte),
//!   q8_1 words `k0` / `k0+1` of block `iqs/2`, accumulated with the SAME `sumi` for both.
//!   Finally `ls = (aux32 >> 27) | 1`, `sumi = sumi * ls / 8` in INTEGER, `d = block_d * q8_1.d`,
//!   accumulated as one FFMA by the caller.
//!
//! `iq2xxs_grid` is kept as two compare chains (low/high u32 word) because a runtime-indexed Rust
//! array spills to local memory and a host-side table never reaches the PTX (PORTING.md #43).
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
    /// `get_int_b2(x, i)`: two `u16` loads (a 66-byte block is only 2-byte aligned).
    #[inline(always)]
    pub unsafe fn get_int_b2(x: *const u8, i: i32) -> u32 {
        let p = x.add(4 * i as usize) as *const u16;
        (*p as u32) | ((*p.add(1) as u32) << 16)
    }
    /// `__vcmpne4(a, 0)`: 0xff in each byte that is non-zero. No cross-byte propagation.
    #[inline(always)]
    pub fn vcmpne4_zero(a: u32) -> u32 {
        let (x, y, z, w) = (a & 0xFF, (a >> 8) & 0xFF, (a >> 16) & 0xFF, (a >> 24) & 0xFF);
        (if x != 0 { 0xFFu32 } else { 0 })
            | (if y != 0 { 0xFF00u32 } else { 0 })
            | (if z != 0 { 0xFF_0000u32 } else { 0 })
            | (if w != 0 { 0xFF00_0000u32 } else { 0 })
    }
    /// `__vsub4(a, b)`: per-byte subtract, each byte independent and wrapping (no carry/borrow).
    /// That is what makes `(x ^ 0xff) - 0xff` a per-byte negation.
    #[inline(always)]
    pub fn vsub4(a: u32, b: u32) -> u32 {
        let (a0, a1, a2, a3) = (a & 0xFF, (a >> 8) & 0xFF, (a >> 16) & 0xFF, (a >> 24) & 0xFF);
        let (b0, b1, b2, b3) = (b & 0xFF, (b >> 8) & 0xFF, (b >> 16) & 0xFF, (b >> 24) & 0xFF);
        (a0.wrapping_sub(b0) & 0xFF)
            | ((a1.wrapping_sub(b1) & 0xFF) << 8)
            | ((a2.wrapping_sub(b2) & 0xFF) << 16)
            | ((a3.wrapping_sub(b3) & 0xFF) << 24)
    }
    /// `__popc(v)`: popcount of a 7-bit sign index.
    #[inline(always)]
    pub fn popc7(v: u32) -> u32 {
        let mut x = v & 0x7F;
        let mut n = 0u32;
        while x != 0 {
            n += x & 1;
            x >>= 1;
        }
        n
    }
    /// `unpack_ksigns(v)`: p = popc(v) & 1; s = v ^ (p << 7); broadcast s over all four bytes.
    /// The 8th sign bit is not stored -- it is the parity of the other seven.
    #[inline(always)]
    pub fn unpack_ksigns(v: u32) -> u32 {
        let p = popc7(v) & 1;
        let s = (v & 0x7F) ^ (p << 7);
        s.wrapping_mul(0x0101_0101)
    }
    /// llama.cpp fastdiv: `(__umulhi(n, mp) + n) >> L`.
    #[inline(always)]
    pub fn fastdiv(n: u32, mp: u32, l: u32) -> u32 {
        let hi = ((n as u64 * mp as u64) >> 32) as u32;
        hi.wrapping_add(n) >> l
    }

    /// Output element types (tags are sized, or `yy.add(n)` would not advance).
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

    const BLOCK_BYTES: usize = 66; // sizeof(block_iq2_xxs)
    const QK: usize = 256;

    /// `dequantize_block_iq2_xxs<dst_t>`: one 256-value block per CUDA block of 32 threads.
    #[inline(always)]
    pub unsafe fn dequant<D: Dst>(vx: *const u8, yy: *mut D) {
        let i = thread::blockIdx_x() as usize;
        let x = vx.add(i * BLOCK_BYTES);
        let tid = thread::threadIdx_x() as usize;
        let il = tid / 8;
        let ib = tid % 8;
        // The wrapper passes `yy + i*QK_K` into the device helper.
        let y = yy.add(i * QK + 32 * ib + 8 * il);
        let qs = x.add(2); // block_iq2_xxs.qs
        // uint16_t * q2 = qs + 4*ib; aux8 = (uint8_t *)q2
        let q2 = qs.add(8 * ib);
        // grid = iq2xxs_grid[aux8[il]]
        let gidx = *q2.add(il) as u32;
        // aux32 = q2[2] | (q2[3] << 16): q2 is a uint16_t*, so q2[2] sits at byte 8*ib + 4, i.e.
        // element (8*ib + 4)/4 = 2*ib + 1 of get_int_b2 (which advances 4 bytes per index).
        let aux32 = get_int_b2(qs, (2 * ib + 1) as i32);
        let d = mul(mul(h2f(*(x as *const u16) as u32), add(0.5, (aux32 >> 28) as f32)), 0.25);
        let signs = ksign((aux32 >> (7 * il as u32)) & 127);
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
    pub unsafe fn iq2_xxs_dequant_f32(vx: *const u8, y: *mut f32) {
        dequant(vx, y)
    }
    #[kernel]
    pub unsafe fn iq2_xxs_dequant_f16(vx: *const u8, y: *mut u16) {
        dequant(vx, y as *mut H)
    }
    #[kernel]
    pub unsafe fn iq2_xxs_dequant_bf16(vx: *const u8, y: *mut u16) {
        dequant(vx, y as *mut B)
    }

    /// `vec_dot_iq2_xxs_q8_1`, accumulated into `acc` as one FFMA.
    #[inline(always)]
    pub unsafe fn vdot_acc(vx: *const u8, yb: *const u8, kbx: i32, iqs: i32, acc: f32) -> f32 {
        let xb = vx.offset((kbx as isize).wrapping_mul(BLOCK_BYTES as isize));
        let qs = xb.add(2);
        let q2 = get_int_b2(qs, iqs);
        let aux32 = get_int_b2(qs, iqs + 1);
        let q8 = yb.add(4) as *const u32;
        let mut sumi = 0i32;
        let mut k0 = 0usize;
        while k0 < 8 {
            let a = (q2 >> (8 * (k0 / 2))) & 0xFF;
            let gl = grid_lo(a);
            let gh = grid_hi(a);
            let signs = unpack_ksigns(aux32 >> (7 * k0 as u32 / 2));
            let s0 = vcmpne4_zero(signs & 0x0804_0201);
            let s1 = vcmpne4_zero(signs & 0x8040_2010);
            let g0 = vsub4(gl ^ s0, s0);
            let g1 = vsub4(gh ^ s1, s1);
            sumi = dp4a(g0, *q8.add(k0), sumi);
            sumi = dp4a(g1, *q8.add(k0 + 1), sumi);
            k0 += 2;
        }
        // ls = (scale * 2 + 1); the shift of 27 covers the top 3 bits of the 4th index.
        let ls = (aux32 >> 27 | 1) as i32;
        let sumi2 = sumi.wrapping_mul(ls) / 8;
        let d = mul(h2f(*(xb as *const u16) as u32), h2f(*(yb as *const u16) as u32));
        fma(d, sumi2 as f32, acc)
    }

    /// Compile-time loop bounds. qi/vdr = 8 threads per block, vdr = 2.
    pub struct K<const NC: usize, const SMALL_K: bool>;
    impl<const NC: usize, const SMALL_K: bool> K<NC, SMALL_K> {
        pub const NW: i32 = if NC <= 4 { 4 } else { 2 };
        pub const NW1: i32 = if NC <= 4 { 3 } else { 1 };
        pub const RPB: usize = if NC == 1 { if SMALL_K { 4 } else { 1 } } else { 2 };
        pub const BLOCKS_PER_ITER: i32 = 2 * Self::NW * 32 / 16;
    }

    /// `mul_mat_vec_q<IQ2_XXS, NC, false, SMALL_K, false>` (MMVQ_PARAMETERS_GENERIC).
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
    /// `iq2xxs_grid[i]` low word, as a compare chain: a runtime-indexed Rust array spills to
    /// local memory and a host-side table never reaches the PTX (PORTING.md #43).
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
        if i == 7u32 { v = 0x082b0808u32; }
        if i == 8u32 { v = 0x082b082bu32; }
        if i == 9u32 { v = 0x082b2b08u32; }
        if i == 10u32 { v = 0x082b2b2bu32; }
        if i == 11u32 { v = 0x19080819u32; }
        if i == 12u32 { v = 0x19081908u32; }
        if i == 13u32 { v = 0x19190808u32; }
        if i == 14u32 { v = 0x19192b08u32; }
        if i == 15u32 { v = 0x192b0819u32; }
        if i == 16u32 { v = 0x192b1908u32; }
        if i == 17u32 { v = 0x2b080808u32; }
        if i == 18u32 { v = 0x2b08082bu32; }
        if i == 19u32 { v = 0x2b082b2bu32; }
        if i == 20u32 { v = 0x2b2b082bu32; }
        if i == 21u32 { v = 0x08080819u32; }
        if i == 22u32 { v = 0x08081908u32; }
        if i == 23u32 { v = 0x08190808u32; }
        if i == 24u32 { v = 0x08191919u32; }
        if i == 25u32 { v = 0x19080808u32; }
        if i == 26u32 { v = 0x2b081908u32; }
        if i == 27u32 { v = 0x2b192b08u32; }
        if i == 28u32 { v = 0x08080808u32; }
        if i == 29u32 { v = 0x0808082bu32; }
        if i == 30u32 { v = 0x082b082bu32; }
        if i == 31u32 { v = 0x2b08082bu32; }
        if i == 32u32 { v = 0x08080819u32; }
        if i == 33u32 { v = 0x08081908u32; }
        if i == 34u32 { v = 0x08190808u32; }
        if i == 35u32 { v = 0x082b0819u32; }
        if i == 36u32 { v = 0x082b1908u32; }
        if i == 37u32 { v = 0x19080808u32; }
        if i == 38u32 { v = 0x1908082bu32; }
        if i == 39u32 { v = 0x19082b08u32; }
        if i == 40u32 { v = 0x192b0808u32; }
        if i == 41u32 { v = 0x2b080819u32; }
        if i == 42u32 { v = 0x2b081908u32; }
        if i == 43u32 { v = 0x2b190808u32; }
        if i == 44u32 { v = 0x2b2b1908u32; }
        if i == 45u32 { v = 0x08080808u32; }
        if i == 46u32 { v = 0x0808082bu32; }
        if i == 47u32 { v = 0x08082b08u32; }
        if i == 48u32 { v = 0x082b0808u32; }
        if i == 49u32 { v = 0x1908192bu32; }
        if i == 50u32 { v = 0x192b2b19u32; }
        if i == 51u32 { v = 0x2b080808u32; }
        if i == 52u32 { v = 0x2b190819u32; }
        if i == 53u32 { v = 0x08082b19u32; }
        if i == 54u32 { v = 0x08190808u32; }
        if i == 55u32 { v = 0x19080808u32; }
        if i == 56u32 { v = 0x2b081908u32; }
        if i == 57u32 { v = 0x2b2b1908u32; }
        if i == 58u32 { v = 0x08080808u32; }
        if i == 59u32 { v = 0x08081919u32; }
        if i == 60u32 { v = 0x08082b08u32; }
        if i == 61u32 { v = 0x08191908u32; }
        if i == 62u32 { v = 0x082b2b08u32; }
        if i == 63u32 { v = 0x19080819u32; }
        if i == 64u32 { v = 0x19081908u32; }
        if i == 65u32 { v = 0x19190808u32; }
        if i == 66u32 { v = 0x1919082bu32; }
        if i == 67u32 { v = 0x2b082b08u32; }
        if i == 68u32 { v = 0x08081908u32; }
        if i == 69u32 { v = 0x19080808u32; }
        if i == 70u32 { v = 0x0808082bu32; }
        if i == 71u32 { v = 0x08191908u32; }
        if i == 72u32 { v = 0x08080819u32; }
        if i == 73u32 { v = 0x08081908u32; }
        if i == 74u32 { v = 0x08190808u32; }
        if i == 75u32 { v = 0x082b0819u32; }
        if i == 76u32 { v = 0x19080808u32; }
        if i == 77u32 { v = 0x192b0808u32; }
        if i == 78u32 { v = 0x2b081908u32; }
        if i == 79u32 { v = 0x2b190808u32; }
        if i == 80u32 { v = 0x2b191919u32; }
        if i == 81u32 { v = 0x08080808u32; }
        if i == 82u32 { v = 0x08082b08u32; }
        if i == 83u32 { v = 0x082b0808u32; }
        if i == 84u32 { v = 0x19190808u32; }
        if i == 85u32 { v = 0x19192b2bu32; }
        if i == 86u32 { v = 0x2b080808u32; }
        if i == 87u32 { v = 0x082b1908u32; }
        if i == 88u32 { v = 0x19081919u32; }
        if i == 89u32 { v = 0x08080808u32; }
        if i == 90u32 { v = 0x08082b08u32; }
        if i == 91u32 { v = 0x082b0808u32; }
        if i == 92u32 { v = 0x082b1919u32; }
        if i == 93u32 { v = 0x19082b19u32; }
        if i == 94u32 { v = 0x2b080808u32; }
        if i == 95u32 { v = 0x08192b08u32; }
        if i == 96u32 { v = 0x192b082bu32; }
        if i == 97u32 { v = 0x08080808u32; }
        if i == 98u32 { v = 0x0819192bu32; }
        if i == 99u32 { v = 0x08080819u32; }
        if i == 100u32 { v = 0x08081908u32; }
        if i == 101u32 { v = 0x08190808u32; }
        if i == 102u32 { v = 0x19080808u32; }
        if i == 103u32 { v = 0x2b080819u32; }
        if i == 104u32 { v = 0x08080808u32; }
        if i == 105u32 { v = 0x08081919u32; }
        if i == 106u32 { v = 0x2b2b0808u32; }
        if i == 107u32 { v = 0x19190819u32; }
        if i == 108u32 { v = 0x08080808u32; }
        if i == 109u32 { v = 0x0808082bu32; }
        if i == 110u32 { v = 0x08082b2bu32; }
        if i == 111u32 { v = 0x19081908u32; }
        if i == 112u32 { v = 0x192b0819u32; }
        if i == 113u32 { v = 0x2b080808u32; }
        if i == 114u32 { v = 0x2b08082bu32; }
        if i == 115u32 { v = 0x082b2b19u32; }
        if i == 116u32 { v = 0x19082b08u32; }
        if i == 117u32 { v = 0x08080808u32; }
        if i == 118u32 { v = 0x0808082bu32; }
        if i == 119u32 { v = 0x08080819u32; }
        if i == 120u32 { v = 0x08081908u32; }
        if i == 121u32 { v = 0x08190808u32; }
        if i == 122u32 { v = 0x19080808u32; }
        if i == 123u32 { v = 0x1919192bu32; }
        if i == 124u32 { v = 0x08080808u32; }
        if i == 125u32 { v = 0x19080819u32; }
        if i == 126u32 { v = 0x192b1908u32; }
        if i == 127u32 { v = 0x2b190808u32; }
        if i == 128u32 { v = 0x08082b08u32; }
        if i == 129u32 { v = 0x082b0808u32; }
        if i == 130u32 { v = 0x2b191908u32; }
        if i == 131u32 { v = 0x19081908u32; }
        if i == 132u32 { v = 0x08080819u32; }
        if i == 133u32 { v = 0x08081908u32; }
        if i == 134u32 { v = 0x08190808u32; }
        if i == 135u32 { v = 0x08192b08u32; }
        if i == 136u32 { v = 0x082b0819u32; }
        if i == 137u32 { v = 0x082b1908u32; }
        if i == 138u32 { v = 0x19080808u32; }
        if i == 139u32 { v = 0x19082b08u32; }
        if i == 140u32 { v = 0x1919192bu32; }
        if i == 141u32 { v = 0x192b0808u32; }
        if i == 142u32 { v = 0x2b080819u32; }
        if i == 143u32 { v = 0x2b081908u32; }
        if i == 144u32 { v = 0x2b190808u32; }
        if i == 145u32 { v = 0x08080808u32; }
        if i == 146u32 { v = 0x082b0808u32; }
        if i == 147u32 { v = 0x192b0819u32; }
        if i == 148u32 { v = 0x2b080808u32; }
        if i == 149u32 { v = 0x2b081919u32; }
        if i == 150u32 { v = 0x08080819u32; }
        if i == 151u32 { v = 0x08190808u32; }
        if i == 152u32 { v = 0x19082b08u32; }
        if i == 153u32 { v = 0x1919192bu32; }
        if i == 154u32 { v = 0x192b2b08u32; }
        if i == 155u32 { v = 0x08080808u32; }
        if i == 156u32 { v = 0x08082b08u32; }
        if i == 157u32 { v = 0x082b0808u32; }
        if i == 158u32 { v = 0x2b080808u32; }
        if i == 159u32 { v = 0x2b192b19u32; }
        if i == 160u32 { v = 0x0819082bu32; }
        if i == 161u32 { v = 0x082b1908u32; }
        if i == 162u32 { v = 0x08080808u32; }
        if i == 163u32 { v = 0x08080819u32; }
        if i == 164u32 { v = 0x08081908u32; }
        if i == 165u32 { v = 0x08190808u32; }
        if i == 166u32 { v = 0x19080808u32; }
        if i == 167u32 { v = 0x19081919u32; }
        if i == 168u32 { v = 0x08080808u32; }
        if i == 169u32 { v = 0x19192b08u32; }
        if i == 170u32 { v = 0x192b0819u32; }
        if i == 171u32 { v = 0x2b08082bu32; }
        if i == 172u32 { v = 0x19081919u32; }
        if i == 173u32 { v = 0x2b190808u32; }
        if i == 174u32 { v = 0x08080808u32; }
        if i == 175u32 { v = 0x08082b08u32; }
        if i == 176u32 { v = 0x08190819u32; }
        if i == 177u32 { v = 0x08192b19u32; }
        if i == 178u32 { v = 0x082b0808u32; }
        if i == 179u32 { v = 0x2b080808u32; }
        if i == 180u32 { v = 0x2b082b08u32; }
        if i == 181u32 { v = 0x08081908u32; }
        if i == 182u32 { v = 0x1908082bu32; }
        if i == 183u32 { v = 0x2b2b1908u32; }
        if i == 184u32 { v = 0x2b190819u32; }
        if i == 185u32 { v = 0x2b190808u32; }
        if i == 186u32 { v = 0x2b19082bu32; }
        if i == 187u32 { v = 0x08082b2bu32; }
        if i == 188u32 { v = 0x08080819u32; }
        if i == 189u32 { v = 0x19191908u32; }
        if i == 190u32 { v = 0x08080808u32; }
        if i == 191u32 { v = 0x08190819u32; }
        if i == 192u32 { v = 0x08192b19u32; }
        if i == 193u32 { v = 0x192b1908u32; }
        if i == 194u32 { v = 0x19080808u32; }
        if i == 195u32 { v = 0x08082b08u32; }
        if i == 196u32 { v = 0x08081908u32; }
        if i == 197u32 { v = 0x08190808u32; }
        if i == 198u32 { v = 0x19080808u32; }
        if i == 199u32 { v = 0x192b2b08u32; }
        if i == 200u32 { v = 0x08080808u32; }
        if i == 201u32 { v = 0x19191919u32; }
        if i == 202u32 { v = 0x08192b08u32; }
        if i == 203u32 { v = 0x192b0808u32; }
        if i == 204u32 { v = 0x08080808u32; }
        if i == 205u32 { v = 0x08081919u32; }
        if i == 206u32 { v = 0x08190808u32; }
        if i == 207u32 { v = 0x0819082bu32; }
        if i == 208u32 { v = 0x2b081908u32; }
        if i == 209u32 { v = 0x1908082bu32; }
        if i == 210u32 { v = 0x08080808u32; }
        if i == 211u32 { v = 0x0808082bu32; }
        if i == 212u32 { v = 0x08082b2bu32; }
        if i == 213u32 { v = 0x19080819u32; }
        if i == 214u32 { v = 0x2b08082bu32; }
        if i == 215u32 { v = 0x08081908u32; }
        if i == 216u32 { v = 0x08192b08u32; }
        if i == 217u32 { v = 0x19080808u32; }
        if i == 218u32 { v = 0x08190819u32; }
        if i == 219u32 { v = 0x08080819u32; }
        if i == 220u32 { v = 0x08081908u32; }
        if i == 221u32 { v = 0x08190808u32; }
        if i == 222u32 { v = 0x08191919u32; }
        if i == 223u32 { v = 0x19080808u32; }
        if i == 224u32 { v = 0x192b0808u32; }
        if i == 225u32 { v = 0x08080808u32; }
        if i == 226u32 { v = 0x1908192bu32; }
        if i == 227u32 { v = 0x2b191908u32; }
        if i == 228u32 { v = 0x08082b19u32; }
        if i == 229u32 { v = 0x19080808u32; }
        if i == 230u32 { v = 0x192b0808u32; }
        if i == 231u32 { v = 0x0808082bu32; }
        if i == 232u32 { v = 0x08081908u32; }
        if i == 233u32 { v = 0x08190819u32; }
        if i == 234u32 { v = 0x08081908u32; }
        if i == 235u32 { v = 0x08190808u32; }
        if i == 236u32 { v = 0x082b1908u32; }
        if i == 237u32 { v = 0x19080808u32; }
        if i == 238u32 { v = 0x2b2b0819u32; }
        if i == 239u32 { v = 0x0819192bu32; }
        if i == 240u32 { v = 0x2b080808u32; }
        if i == 241u32 { v = 0x19081919u32; }
        if i == 242u32 { v = 0x08080808u32; }
        if i == 243u32 { v = 0x082b082bu32; }
        if i == 244u32 { v = 0x19081908u32; }
        if i == 245u32 { v = 0x19190819u32; }
        if i == 246u32 { v = 0x2b080819u32; }
        if i == 247u32 { v = 0x082b0808u32; }
        if i == 248u32 { v = 0x0808082bu32; }
        if i == 249u32 { v = 0x19190808u32; }
        if i == 250u32 { v = 0x2b081919u32; }
        if i == 251u32 { v = 0x08082b19u32; }
        if i == 252u32 { v = 0x08080808u32; }
        if i == 253u32 { v = 0x08192b08u32; }
        if i == 254u32 { v = 0x19190808u32; }
        if i == 255u32 { v = 0x08081908u32; }
        v
    }
    /// `iq2xxs_grid[i]` high word.
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
        if i == 21u32 { v = 0x08080819u32; }
        if i == 22u32 { v = 0x08080819u32; }
        if i == 23u32 { v = 0x08080819u32; }
        if i == 24u32 { v = 0x08080819u32; }
        if i == 25u32 { v = 0x08080819u32; }
        if i == 26u32 { v = 0x08080819u32; }
        if i == 27u32 { v = 0x08080819u32; }
        if i == 28u32 { v = 0x0808082bu32; }
        if i == 29u32 { v = 0x0808082bu32; }
        if i == 30u32 { v = 0x0808082bu32; }
        if i == 31u32 { v = 0x0808082bu32; }
        if i == 32u32 { v = 0x08081908u32; }
        if i == 33u32 { v = 0x08081908u32; }
        if i == 34u32 { v = 0x08081908u32; }
        if i == 35u32 { v = 0x08081908u32; }
        if i == 36u32 { v = 0x08081908u32; }
        if i == 37u32 { v = 0x08081908u32; }
        if i == 38u32 { v = 0x08081908u32; }
        if i == 39u32 { v = 0x08081908u32; }
        if i == 40u32 { v = 0x08081908u32; }
        if i == 41u32 { v = 0x08081908u32; }
        if i == 42u32 { v = 0x08081908u32; }
        if i == 43u32 { v = 0x08081908u32; }
        if i == 44u32 { v = 0x08081908u32; }
        if i == 45u32 { v = 0x08081919u32; }
        if i == 46u32 { v = 0x08081919u32; }
        if i == 47u32 { v = 0x08081919u32; }
        if i == 48u32 { v = 0x08081919u32; }
        if i == 49u32 { v = 0x08081919u32; }
        if i == 50u32 { v = 0x08081919u32; }
        if i == 51u32 { v = 0x08081919u32; }
        if i == 52u32 { v = 0x08081919u32; }
        if i == 53u32 { v = 0x0808192bu32; }
        if i == 54u32 { v = 0x0808192bu32; }
        if i == 55u32 { v = 0x0808192bu32; }
        if i == 56u32 { v = 0x0808192bu32; }
        if i == 57u32 { v = 0x0808192bu32; }
        if i == 58u32 { v = 0x08082b08u32; }
        if i == 59u32 { v = 0x08082b08u32; }
        if i == 60u32 { v = 0x08082b08u32; }
        if i == 61u32 { v = 0x08082b08u32; }
        if i == 62u32 { v = 0x08082b08u32; }
        if i == 63u32 { v = 0x08082b08u32; }
        if i == 64u32 { v = 0x08082b08u32; }
        if i == 65u32 { v = 0x08082b08u32; }
        if i == 66u32 { v = 0x08082b08u32; }
        if i == 67u32 { v = 0x08082b08u32; }
        if i == 68u32 { v = 0x08082b19u32; }
        if i == 69u32 { v = 0x08082b19u32; }
        if i == 70u32 { v = 0x08082b2bu32; }
        if i == 71u32 { v = 0x08082b2bu32; }
        if i == 72u32 { v = 0x08190808u32; }
        if i == 73u32 { v = 0x08190808u32; }
        if i == 74u32 { v = 0x08190808u32; }
        if i == 75u32 { v = 0x08190808u32; }
        if i == 76u32 { v = 0x08190808u32; }
        if i == 77u32 { v = 0x08190808u32; }
        if i == 78u32 { v = 0x08190808u32; }
        if i == 79u32 { v = 0x08190808u32; }
        if i == 80u32 { v = 0x08190808u32; }
        if i == 81u32 { v = 0x08190819u32; }
        if i == 82u32 { v = 0x08190819u32; }
        if i == 83u32 { v = 0x08190819u32; }
        if i == 84u32 { v = 0x08190819u32; }
        if i == 85u32 { v = 0x08190819u32; }
        if i == 86u32 { v = 0x08190819u32; }
        if i == 87u32 { v = 0x0819082bu32; }
        if i == 88u32 { v = 0x0819082bu32; }
        if i == 89u32 { v = 0x08191908u32; }
        if i == 90u32 { v = 0x08191908u32; }
        if i == 91u32 { v = 0x08191908u32; }
        if i == 92u32 { v = 0x08191908u32; }
        if i == 93u32 { v = 0x08191908u32; }
        if i == 94u32 { v = 0x08191908u32; }
        if i == 95u32 { v = 0x08191919u32; }
        if i == 96u32 { v = 0x08191919u32; }
        if i == 97u32 { v = 0x0819192bu32; }
        if i == 98u32 { v = 0x0819192bu32; }
        if i == 99u32 { v = 0x08192b08u32; }
        if i == 100u32 { v = 0x08192b08u32; }
        if i == 101u32 { v = 0x08192b08u32; }
        if i == 102u32 { v = 0x08192b08u32; }
        if i == 103u32 { v = 0x08192b08u32; }
        if i == 104u32 { v = 0x08192b19u32; }
        if i == 105u32 { v = 0x08192b19u32; }
        if i == 106u32 { v = 0x08192b19u32; }
        if i == 107u32 { v = 0x08192b2bu32; }
        if i == 108u32 { v = 0x082b0808u32; }
        if i == 109u32 { v = 0x082b0808u32; }
        if i == 110u32 { v = 0x082b0808u32; }
        if i == 111u32 { v = 0x082b0808u32; }
        if i == 112u32 { v = 0x082b0808u32; }
        if i == 113u32 { v = 0x082b0808u32; }
        if i == 114u32 { v = 0x082b0808u32; }
        if i == 115u32 { v = 0x082b0819u32; }
        if i == 116u32 { v = 0x082b0819u32; }
        if i == 117u32 { v = 0x082b082bu32; }
        if i == 118u32 { v = 0x082b082bu32; }
        if i == 119u32 { v = 0x082b1908u32; }
        if i == 120u32 { v = 0x082b1908u32; }
        if i == 121u32 { v = 0x082b1908u32; }
        if i == 122u32 { v = 0x082b1908u32; }
        if i == 123u32 { v = 0x082b1908u32; }
        if i == 124u32 { v = 0x082b1919u32; }
        if i == 125u32 { v = 0x082b1919u32; }
        if i == 126u32 { v = 0x082b1919u32; }
        if i == 127u32 { v = 0x082b192bu32; }
        if i == 128u32 { v = 0x082b2b08u32; }
        if i == 129u32 { v = 0x082b2b08u32; }
        if i == 130u32 { v = 0x082b2b08u32; }
        if i == 131u32 { v = 0x082b2b2bu32; }
        if i == 132u32 { v = 0x19080808u32; }
        if i == 133u32 { v = 0x19080808u32; }
        if i == 134u32 { v = 0x19080808u32; }
        if i == 135u32 { v = 0x19080808u32; }
        if i == 136u32 { v = 0x19080808u32; }
        if i == 137u32 { v = 0x19080808u32; }
        if i == 138u32 { v = 0x19080808u32; }
        if i == 139u32 { v = 0x19080808u32; }
        if i == 140u32 { v = 0x19080808u32; }
        if i == 141u32 { v = 0x19080808u32; }
        if i == 142u32 { v = 0x19080808u32; }
        if i == 143u32 { v = 0x19080808u32; }
        if i == 144u32 { v = 0x19080808u32; }
        if i == 145u32 { v = 0x19080819u32; }
        if i == 146u32 { v = 0x19080819u32; }
        if i == 147u32 { v = 0x19080819u32; }
        if i == 148u32 { v = 0x19080819u32; }
        if i == 149u32 { v = 0x19080819u32; }
        if i == 150u32 { v = 0x1908082bu32; }
        if i == 151u32 { v = 0x1908082bu32; }
        if i == 152u32 { v = 0x1908082bu32; }
        if i == 153u32 { v = 0x1908082bu32; }
        if i == 154u32 { v = 0x1908082bu32; }
        if i == 155u32 { v = 0x19081908u32; }
        if i == 156u32 { v = 0x19081908u32; }
        if i == 157u32 { v = 0x19081908u32; }
        if i == 158u32 { v = 0x19081908u32; }
        if i == 159u32 { v = 0x19081908u32; }
        if i == 160u32 { v = 0x19081919u32; }
        if i == 161u32 { v = 0x19081919u32; }
        if i == 162u32 { v = 0x1908192bu32; }
        if i == 163u32 { v = 0x19082b08u32; }
        if i == 164u32 { v = 0x19082b08u32; }
        if i == 165u32 { v = 0x19082b08u32; }
        if i == 166u32 { v = 0x19082b08u32; }
        if i == 167u32 { v = 0x19082b08u32; }
        if i == 168u32 { v = 0x19082b19u32; }
        if i == 169u32 { v = 0x19082b19u32; }
        if i == 170u32 { v = 0x19082b19u32; }
        if i == 171u32 { v = 0x19082b19u32; }
        if i == 172u32 { v = 0x19082b2bu32; }
        if i == 173u32 { v = 0x19082b2bu32; }
        if i == 174u32 { v = 0x19190808u32; }
        if i == 175u32 { v = 0x19190808u32; }
        if i == 176u32 { v = 0x19190808u32; }
        if i == 177u32 { v = 0x19190808u32; }
        if i == 178u32 { v = 0x19190808u32; }
        if i == 179u32 { v = 0x19190808u32; }
        if i == 180u32 { v = 0x19190808u32; }
        if i == 181u32 { v = 0x19190819u32; }
        if i == 182u32 { v = 0x19190819u32; }
        if i == 183u32 { v = 0x19190819u32; }
        if i == 184u32 { v = 0x1919082bu32; }
        if i == 185u32 { v = 0x19191908u32; }
        if i == 186u32 { v = 0x19191908u32; }
        if i == 187u32 { v = 0x19191919u32; }
        if i == 188u32 { v = 0x1919192bu32; }
        if i == 189u32 { v = 0x1919192bu32; }
        if i == 190u32 { v = 0x19192b08u32; }
        if i == 191u32 { v = 0x19192b08u32; }
        if i == 192u32 { v = 0x19192b08u32; }
        if i == 193u32 { v = 0x19192b08u32; }
        if i == 194u32 { v = 0x19192b19u32; }
        if i == 195u32 { v = 0x19192b2bu32; }
        if i == 196u32 { v = 0x192b0808u32; }
        if i == 197u32 { v = 0x192b0808u32; }
        if i == 198u32 { v = 0x192b0808u32; }
        if i == 199u32 { v = 0x192b0808u32; }
        if i == 200u32 { v = 0x192b0819u32; }
        if i == 201u32 { v = 0x192b0819u32; }
        if i == 202u32 { v = 0x192b082bu32; }
        if i == 203u32 { v = 0x192b082bu32; }
        if i == 204u32 { v = 0x192b1908u32; }
        if i == 205u32 { v = 0x192b1908u32; }
        if i == 206u32 { v = 0x192b1919u32; }
        if i == 207u32 { v = 0x192b1919u32; }
        if i == 208u32 { v = 0x192b1919u32; }
        if i == 209u32 { v = 0x192b2b08u32; }
        if i == 210u32 { v = 0x2b080808u32; }
        if i == 211u32 { v = 0x2b080808u32; }
        if i == 212u32 { v = 0x2b080808u32; }
        if i == 213u32 { v = 0x2b080808u32; }
        if i == 214u32 { v = 0x2b080808u32; }
        if i == 215u32 { v = 0x2b080819u32; }
        if i == 216u32 { v = 0x2b080819u32; }
        if i == 217u32 { v = 0x2b080819u32; }
        if i == 218u32 { v = 0x2b08082bu32; }
        if i == 219u32 { v = 0x2b081908u32; }
        if i == 220u32 { v = 0x2b081908u32; }
        if i == 221u32 { v = 0x2b081908u32; }
        if i == 222u32 { v = 0x2b081908u32; }
        if i == 223u32 { v = 0x2b081908u32; }
        if i == 224u32 { v = 0x2b081908u32; }
        if i == 225u32 { v = 0x2b081919u32; }
        if i == 226u32 { v = 0x2b081919u32; }
        if i == 227u32 { v = 0x2b081919u32; }
        if i == 228u32 { v = 0x2b08192bu32; }
        if i == 229u32 { v = 0x2b08192bu32; }
        if i == 230u32 { v = 0x2b08192bu32; }
        if i == 231u32 { v = 0x2b082b08u32; }
        if i == 232u32 { v = 0x2b082b19u32; }
        if i == 233u32 { v = 0x2b082b2bu32; }
        if i == 234u32 { v = 0x2b190808u32; }
        if i == 235u32 { v = 0x2b190808u32; }
        if i == 236u32 { v = 0x2b190808u32; }
        if i == 237u32 { v = 0x2b190808u32; }
        if i == 238u32 { v = 0x2b190808u32; }
        if i == 239u32 { v = 0x2b190819u32; }
        if i == 240u32 { v = 0x2b190819u32; }
        if i == 241u32 { v = 0x2b19082bu32; }
        if i == 242u32 { v = 0x2b191908u32; }
        if i == 243u32 { v = 0x2b191908u32; }
        if i == 244u32 { v = 0x2b191908u32; }
        if i == 245u32 { v = 0x2b191919u32; }
        if i == 246u32 { v = 0x2b192b08u32; }
        if i == 247u32 { v = 0x2b192b19u32; }
        if i == 248u32 { v = 0x2b2b0808u32; }
        if i == 249u32 { v = 0x2b2b0808u32; }
        if i == 250u32 { v = 0x2b2b0808u32; }
        if i == 251u32 { v = 0x2b2b0819u32; }
        if i == 252u32 { v = 0x2b2b082bu32; }
        if i == 253u32 { v = 0x2b2b1908u32; }
        if i == 254u32 { v = 0x2b2b2b08u32; }
        if i == 255u32 { v = 0x2b2b2b19u32; }
        v
    }
    /// `ksigns_iq2xs[128]` (dequantize only; mmvq uses `unpack_ksigns`).
    #[inline(always)]
    pub fn ksign(i: u32) -> u8 {
        let mut v: u8 = 0;
        if i == 1u32 { v = 0x81u8; }
        if i == 2u32 { v = 0x82u8; }
        if i == 3u32 { v = 0x03u8; }
        if i == 4u32 { v = 0x84u8; }
        if i == 5u32 { v = 0x05u8; }
        if i == 6u32 { v = 0x06u8; }
        if i == 7u32 { v = 0x87u8; }
        if i == 8u32 { v = 0x88u8; }
        if i == 9u32 { v = 0x09u8; }
        if i == 10u32 { v = 0x0au8; }
        if i == 11u32 { v = 0x8bu8; }
        if i == 12u32 { v = 0x0cu8; }
        if i == 13u32 { v = 0x8du8; }
        if i == 14u32 { v = 0x8eu8; }
        if i == 15u32 { v = 0x0fu8; }
        if i == 16u32 { v = 0x90u8; }
        if i == 17u32 { v = 0x11u8; }
        if i == 18u32 { v = 0x12u8; }
        if i == 19u32 { v = 0x93u8; }
        if i == 20u32 { v = 0x14u8; }
        if i == 21u32 { v = 0x95u8; }
        if i == 22u32 { v = 0x96u8; }
        if i == 23u32 { v = 0x17u8; }
        if i == 24u32 { v = 0x18u8; }
        if i == 25u32 { v = 0x99u8; }
        if i == 26u32 { v = 0x9au8; }
        if i == 27u32 { v = 0x1bu8; }
        if i == 28u32 { v = 0x9cu8; }
        if i == 29u32 { v = 0x1du8; }
        if i == 30u32 { v = 0x1eu8; }
        if i == 31u32 { v = 0x9fu8; }
        if i == 32u32 { v = 0xa0u8; }
        if i == 33u32 { v = 0x21u8; }
        if i == 34u32 { v = 0x22u8; }
        if i == 35u32 { v = 0xa3u8; }
        if i == 36u32 { v = 0x24u8; }
        if i == 37u32 { v = 0xa5u8; }
        if i == 38u32 { v = 0xa6u8; }
        if i == 39u32 { v = 0x27u8; }
        if i == 40u32 { v = 0x28u8; }
        if i == 41u32 { v = 0xa9u8; }
        if i == 42u32 { v = 0xaau8; }
        if i == 43u32 { v = 0x2bu8; }
        if i == 44u32 { v = 0xacu8; }
        if i == 45u32 { v = 0x2du8; }
        if i == 46u32 { v = 0x2eu8; }
        if i == 47u32 { v = 0xafu8; }
        if i == 48u32 { v = 0x30u8; }
        if i == 49u32 { v = 0xb1u8; }
        if i == 50u32 { v = 0xb2u8; }
        if i == 51u32 { v = 0x33u8; }
        if i == 52u32 { v = 0xb4u8; }
        if i == 53u32 { v = 0x35u8; }
        if i == 54u32 { v = 0x36u8; }
        if i == 55u32 { v = 0xb7u8; }
        if i == 56u32 { v = 0xb8u8; }
        if i == 57u32 { v = 0x39u8; }
        if i == 58u32 { v = 0x3au8; }
        if i == 59u32 { v = 0xbbu8; }
        if i == 60u32 { v = 0x3cu8; }
        if i == 61u32 { v = 0xbdu8; }
        if i == 62u32 { v = 0xbeu8; }
        if i == 63u32 { v = 0x3fu8; }
        if i == 64u32 { v = 0xc0u8; }
        if i == 65u32 { v = 0x41u8; }
        if i == 66u32 { v = 0x42u8; }
        if i == 67u32 { v = 0xc3u8; }
        if i == 68u32 { v = 0x44u8; }
        if i == 69u32 { v = 0xc5u8; }
        if i == 70u32 { v = 0xc6u8; }
        if i == 71u32 { v = 0x47u8; }
        if i == 72u32 { v = 0x48u8; }
        if i == 73u32 { v = 0xc9u8; }
        if i == 74u32 { v = 0xcau8; }
        if i == 75u32 { v = 0x4bu8; }
        if i == 76u32 { v = 0xccu8; }
        if i == 77u32 { v = 0x4du8; }
        if i == 78u32 { v = 0x4eu8; }
        if i == 79u32 { v = 0xcfu8; }
        if i == 80u32 { v = 0x50u8; }
        if i == 81u32 { v = 0xd1u8; }
        if i == 82u32 { v = 0xd2u8; }
        if i == 83u32 { v = 0x53u8; }
        if i == 84u32 { v = 0xd4u8; }
        if i == 85u32 { v = 0x55u8; }
        if i == 86u32 { v = 0x56u8; }
        if i == 87u32 { v = 0xd7u8; }
        if i == 88u32 { v = 0xd8u8; }
        if i == 89u32 { v = 0x59u8; }
        if i == 90u32 { v = 0x5au8; }
        if i == 91u32 { v = 0xdbu8; }
        if i == 92u32 { v = 0x5cu8; }
        if i == 93u32 { v = 0xddu8; }
        if i == 94u32 { v = 0xdeu8; }
        if i == 95u32 { v = 0x5fu8; }
        if i == 96u32 { v = 0x60u8; }
        if i == 97u32 { v = 0xe1u8; }
        if i == 98u32 { v = 0xe2u8; }
        if i == 99u32 { v = 0x63u8; }
        if i == 100u32 { v = 0xe4u8; }
        if i == 101u32 { v = 0x65u8; }
        if i == 102u32 { v = 0x66u8; }
        if i == 103u32 { v = 0xe7u8; }
        if i == 104u32 { v = 0xe8u8; }
        if i == 105u32 { v = 0x69u8; }
        if i == 106u32 { v = 0x6au8; }
        if i == 107u32 { v = 0xebu8; }
        if i == 108u32 { v = 0x6cu8; }
        if i == 109u32 { v = 0xedu8; }
        if i == 110u32 { v = 0xeeu8; }
        if i == 111u32 { v = 0x6fu8; }
        if i == 112u32 { v = 0xf0u8; }
        if i == 113u32 { v = 0x71u8; }
        if i == 114u32 { v = 0x72u8; }
        if i == 115u32 { v = 0xf3u8; }
        if i == 116u32 { v = 0x74u8; }
        if i == 117u32 { v = 0xf5u8; }
        if i == 118u32 { v = 0xf6u8; }
        if i == 119u32 { v = 0x77u8; }
        if i == 120u32 { v = 0x78u8; }
        if i == 121u32 { v = 0xf9u8; }
        if i == 122u32 { v = 0xfau8; }
        if i == 123u32 { v = 0x7bu8; }
        if i == 124u32 { v = 0xfcu8; }
        if i == 125u32 { v = 0x7du8; }
        if i == 126u32 { v = 0x7eu8; }
        if i == 127u32 { v = 0xffu8; }
        v
    }
    /// `kmask_iq2xs[8]`, indexed only by unrolled constants so it stays in registers.
    pub const KMASK: [u8; 8] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80];
    #[kernel] pub unsafe fn iq2_xxs_mmvq_1(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<1, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq2_xxs_mmvq_1_small_k(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<1, true>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq2_xxs_mmvq_2(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<2, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq2_xxs_mmvq_3(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<3, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq2_xxs_mmvq_4(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<4, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq2_xxs_mmvq_5(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<5, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq2_xxs_mmvq_6(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<6, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq2_xxs_mmvq_7(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<7, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq2_xxs_mmvq_8(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<8, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    // GENERATED KERNELS END
}

fn main() {
    std::process::exit(if gate::run_gate() { 0 } else { 1 });
}
