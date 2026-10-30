//! GGML IQ3_XXS (type 18) CUDA kernels in cuda-oxide, ported from llama.cpp acecd56 under the
//! reference's own parameter ABI and bit-identical to its nvcc build (sm_120a; the cubins in ref/
//! are the ones the iq4_nl / iq4_xs ports gate against):
//!
//! - `iq3_xxs_dequant_{f32,f16,bf16}` = `dequantize_block_iq3_xxs<dst_t>` (convert.cu, grid
//!   ceil(k/256), block 32). `d = block_d * (0.5 + (aux>>28)) * 0.5` (a power of two, so the two
//!   fast-math multiplies are exact); values are `d * grid[j]` negated where the group's sign bit
//!   is set (`ksigns_iq2xs[(aux>>7*il) & 127]` vs `kmask_iq2xs[j]`, `j=0..7`).
//! - `iq3_xxs_mmvq_<n>` = `mul_mat_vec_q<GGML_TYPE_IQ3_XXS, n, has_fusion=false, small_k, false>`
//!   (mmvq.cu GENERIC table): block (32, nwarps), nwarps 4 for n <= 4 / 2 for n 5..8, rows per
//!   CUDA block 1 for n == 1 (nwarps with small_k) and 2 otherwise. qk 256, QR3_XXS 4, qi 16,
//!   vdr 2 -> `blocks_per_iter = vdr*nwarps*32/qi` = 8 (n <= 4) / 4 (n 5..8), and
//!   `kqs = vdr*(tid % (qi/vdr)) = 2*(tid % 8)`.
//!   vec_dot_iq3_xxs_q8_1: `q3_packed` = the two u32 at `qs + 4*iqs`; `aux32` = the u32 at
//!   `qs + 64 + 4*(iqs/2)`. For l0 = 0,2,4,6: grid entries `q3[l0]` / `q3[l0+1]`, sign selector
//!   `unpack_ksigns(aux32 >> (7*l0/2))`, `grid_l = __vsub4(grid ^ signs0, signs0)` against the
//!   q8_1 words `l0` and `l0+1` of block `iqs/2`; finally `sumi = (ls*sumi + sumi/2)/2` with
//!   `ls = aux32 >> 28`, and `d = block_d * q8_1[iqs/2].d`. The caller accumulates in one FFMA.
//!
//! The grid codebook is its 256 little-endian `u32` words in an initialized immutable `static` (a
//! `.global` array in the PTX, read with a runtime index like llama.cpp's `__device__` table; bytes via
//! `.to_le_bytes()`), as in iq3_s. A select chain per lookup made the dense decode compute-bound.
#![allow(non_snake_case, clippy::missing_safety_doc, clippy::too_many_arguments)]
mod gate;

use cuda_device::{SharedArray, convert, device, dotprod, kernel, ptx_asm, thread, warp};
use cuda_host::cuda_module;

/// `iq3xxs_grid[256]` (ggml-common.h).
pub static IQ3XXS_GRID: [u32; 256] = include!("iq3xxs_grid.rs");
/// `ksigns_iq2xs[128]` (ggml-common.h).
pub static KSIGNS_IQ2XS: [u8; 128] = [
    0x00, 0x81, 0x82, 0x03, 0x84, 0x05, 0x06, 0x87, 0x88, 0x09, 0x0a, 0x8b, 0x0c, 0x8d, 0x8e, 0x0f,
    0x90, 0x11, 0x12, 0x93, 0x14, 0x95, 0x96, 0x17, 0x18, 0x99, 0x9a, 0x1b, 0x9c, 0x1d, 0x1e, 0x9f,
    0xa0, 0x21, 0x22, 0xa3, 0x24, 0xa5, 0xa6, 0x27, 0x28, 0xa9, 0xaa, 0x2b, 0xac, 0x2d, 0x2e, 0xaf,
    0x30, 0xb1, 0xb2, 0x33, 0xb4, 0x35, 0x36, 0xb7, 0xb8, 0x39, 0x3a, 0xbb, 0x3c, 0xbd, 0xbe, 0x3f,
    0xc0, 0x41, 0x42, 0xc3, 0x44, 0xc5, 0xc6, 0x47, 0x48, 0xc9, 0xca, 0x4b, 0xcc, 0x4d, 0x4e, 0xcf,
    0x50, 0xd1, 0xd2, 0x53, 0xd4, 0x55, 0x56, 0xd7, 0xd8, 0x59, 0x5a, 0xdb, 0x5c, 0xdd, 0xde, 0x5f,
    0x60, 0xe1, 0xe2, 0x63, 0xe4, 0x65, 0x66, 0xe7, 0xe8, 0x69, 0x6a, 0xeb, 0x6c, 0xed, 0xee, 0x6f,
    0xf0, 0x71, 0x72, 0xf3, 0x74, 0xf5, 0xf6, 0x77, 0x78, 0xf9, 0xfa, 0x7b, 0xfc, 0x7d, 0x7e, 0xff,
];

#[cuda_module]
mod kernels {
    use super::*;

    /// `iq3xxs_grid[i]` from the `.global` table (as iq3_s reads its grid): the 256-way compare chain
    /// this replaced made the dense decode ~10x slower than IQ3_S (625 us vs 72 us for a 5120x17408
    /// row at m = 1).
    #[inline(always)]
    pub fn grid_word(i: u32) -> u32 {
        unsafe { *(&raw const IQ3XXS_GRID as *const u32).add((i & 255) as usize) }
    }
    /// `ksigns_iq2xs[i]` from the `.global` table (see grid_word).
    #[inline(always)]
    pub fn ksign(i: u32) -> u8 {
        unsafe { *(&raw const KSIGNS_IQ2XS as *const u8).add((i & 127) as usize) }
    }
    /// IQ3_XXS grid words: `iq3xxs_grid[256]` verbatim (values 1..127, 8 bytes per entry).
    /// The 8 bytes of an entry are read out with `to_le_bytes()`.
    /// `kmask_iq2xs[8]` is indexed only by `j`/`j+4` inside a fully unrolled loop, so it stays a
    /// plain array.
    pub const KMASK: [u8; 8] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80];

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
    /// `__popc`.
    #[inline(always)]
    pub fn popc(x: u32) -> u32 {
        let r: u32;
        unsafe { ptx_asm!("popc.b32 %0, %1;", out("=r") r, in("r") x, options(register_only)); }
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
    /// `__vcmpne4(a, 0)`: 0xff in each byte that is non-zero.
    #[inline(always)]
    pub fn vcmpne4_zero(a: u32) -> u32 {
        let (x, y, z, w) = (a & 0xFF, (a >> 8) & 0xFF, (a >> 16) & 0xFF, (a >> 24) & 0xFF);
        (if x != 0 { 0xFFu32 } else { 0 })
            | (if y != 0 { 0xFF00u32 } else { 0 })
            | (if z != 0 { 0xFF_0000u32 } else { 0 })
            | (if w != 0 { 0xFF00_0000u32 } else { 0 })
    }
    /// `__vsub4(a, b)`: the SIMD-video per-byte subtract. Each byte is independent and wraps --
    /// there is NO carry/borrow between bytes (that is what makes it usable as `(byte^0xff)-0xff`
    /// for a per-byte negation). Implementing it with borrow propagation breaks the cases where the
    /// selector byte is 0x10 and byte 0 is zero.
    #[inline(always)]
    pub fn vsub4(a: u32, b: u32) -> u32 {
        let (a0, a1, a2, a3) = (a & 0xFF, (a >> 8) & 0xFF, (a >> 16) & 0xFF, (a >> 24) & 0xFF);
        let (b0, b1, b2, b3) = (b & 0xFF, (b >> 8) & 0xFF, (b >> 16) & 0xFF, (b >> 24) & 0xFF);
        (a0.wrapping_sub(b0) & 0xFF)
            | ((a1.wrapping_sub(b1) & 0xFF) << 8)
            | ((a2.wrapping_sub(b2) & 0xFF) << 16)
            | ((a3.wrapping_sub(b3) & 0xFF) << 24)
    }
    /// `get_int_b2(x, i)`: two `u16` loads (the block is 98 bytes, so a `u32` load can be
    /// misaligned — the reference deliberately reads 16-bit halves).
    #[inline(always)]
    pub unsafe fn get_int_b2(x: *const u8, i: i32) -> u32 {
        let p = x.add(4 * i as usize) as *const u16;
        (*p as u32) | ((*p.add(1) as u32) << 16)
    }
    /// `unpack_ksigns(v)`: the 7-bit mask, with the 8th sign folded in as its parity.
    #[inline(always)]
    pub fn unpack_ksigns(v: u32) -> u8 {
        let v7 = (v & 0x7F) as u8;
        v7 ^ (((popc(v7 as u32) & 1) as u8) << 7)
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
    /// f16 and bf16 need separate zero-sized tags, or both kernels monomorphise to one function
    /// (and the `#[cuda_module]` scan would then only emit one of them).
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

    const BLOCK_BYTES: usize = 98; // sizeof(block_iq3_xxs)
    const QK: usize = 256;

    /// `dequantize_block_iq3_xxs<dst_t>`: one 256-value block per CUDA block of 32 threads.
    #[inline(always)]
    pub unsafe fn dequant<D: Dst>(vx: *const u8, yy: *mut D) {
        let i = thread::blockIdx_x() as usize;
        let x = vx.add(i * BLOCK_BYTES);
        let tid = thread::threadIdx_x() as usize;
        let il = tid / 8;
        let ib = tid % 8;
        // The reference __global__ wrapper passes `yy + i*QK_K` into the device helper.
        let y = yy.add(i * QK + 32 * ib + 8 * il);
        // `qs` is `qs[3*QK_K/8]` = 96 bytes at struct offset 2 (d is first).
        let q3 = x.add(2 + 8 * ib);
        // gas = (const uint16_t *)(qs + QK_K/4) + 2*ib = x + 2 + 64 + 4*ib.
        let aux32 = get_int_b2(x.add(2 + 64), ib as i32);
        // 0.5 + (aux32>>28) is a small integer; both multiplies are powers of two (exact).
        let d = mul(mul(h2f(*(x as *const u16) as u32), add(0.5, (aux32 >> 28) as f32)), 0.5);
        let signs = ksign((aux32 >> (7 * il as u32)) & 127);
        let w1 = grid_word(*q3.add(2 * il) as u32);
        let w2 = grid_word(*q3.add(2 * il + 1) as u32);
        let g1 = w1.to_le_bytes();
        let g2 = w2.to_le_bytes();
        let mut j = 0usize;
        while j < 4 {
            let s0 = if signs & KMASK[j] != 0 { -1.0f32 } else { 1.0f32 };
            let s1 = if signs & KMASK[j + 4] != 0 { -1.0f32 } else { 1.0f32 };
            D::st(y.add(j), mul(mul(d, g1[j] as f32), s0));
            D::st(y.add(j + 4), mul(mul(d, g2[j] as f32), s1));
            j += 1;
        }
    }

    #[kernel]
    pub unsafe fn iq3_xxs_dequant_f32(vx: *const u8, y: *mut f32) {
        dequant(vx, y)
    }
    #[kernel]
    pub unsafe fn iq3_xxs_dequant_f16(vx: *const u8, y: *mut u16) {
        dequant(vx, y as *mut H)
    }
    #[kernel]
    pub unsafe fn iq3_xxs_dequant_bf16(vx: *const u8, y: *mut u16) {
        dequant(vx, y as *mut B)
    }

    /// `vec_dot_iq3_xxs_q8_1(vx, yb, kbx, iqs)` accumulated into `acc` as one FFMA.
    /// `yb` is the q8_1 block at `kbx * (QK_K/32) + iqs/2`.
    #[inline(always)]
    pub unsafe fn vdot_acc(vx: *const u8, yb: *const u8, kbx: i32, iqs: i32, acc: f32) -> f32 {
        let xb = vx.offset((kbx as isize).wrapping_mul(BLOCK_BYTES as isize));
        let qs = xb.add(2); // block_iq3_xxs.qs
        // get_int_b2(qs, iqs): the u16 pair at qs + 4*iqs.
        let mut q3 = [0u8; 8];
        let mut t = 0usize;
        while t < 8 {
            q3[t] = *qs.add(4 * iqs as usize + t);
            t += 1;
        }
        let aux32 = get_int_b2(qs.add(64), iqs / 2);
        let q8 = yb.add(4) as *const u32;
        let mut sumi = 0i32;
        let mut l0 = 0usize;
        while l0 < 8 {
            let g0 = grid_word(q3[l0] as u32);
            let g1 = grid_word(q3[l0 + 1] as u32);
            let signs = unpack_ksigns(aux32 >> (7 * l0 as u32 / 2));
            // grid_l = __vsub4(grid ^ signs0, signs0) with signs0 = -1 per set sign bit =
            // wrap(byte + 1) = -byte, i.e. the sign is applied by negating that byte.
            let s0 = vcmpne4_zero((signs as u32 * 0x0101_0101) & 0x0804_0201);
            let gl = vsub4(g0 ^ s0, s0);
            let s1 = vcmpne4_zero((signs as u32 * 0x0101_0101) & 0x8040_2010);
            let gh = vsub4(g1 ^ s1, s1);
            sumi = dp4a(gl, *q8.add(l0), sumi);
            sumi = dp4a(gh, *q8.add(l0 + 1), sumi);
            l0 += 2;
        }
        // (ls*sumi + sumi/2)/2, integer; ls = aux32 >> 28.
        let ls = (aux32 >> 28) as i32;
        let sumi = (ls.wrapping_mul(sumi).wrapping_add(sumi / 2)) / 2;
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

    /// `mul_mat_vec_q<IQ3_XXS, NC, false, SMALL_K, false>` (MMVQ_PARAMETERS_GENERIC).
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
    #[kernel] pub unsafe fn iq3_xxs_mmvq_1(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<1, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq3_xxs_mmvq_1_small_k(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<1, true>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq3_xxs_mmvq_2(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<2, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq3_xxs_mmvq_3(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<3, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq3_xxs_mmvq_4(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<4, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq3_xxs_mmvq_5(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<5, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq3_xxs_mmvq_6(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<6, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq3_xxs_mmvq_7(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<7, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq3_xxs_mmvq_8(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<8, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    // GENERATED KERNELS END
}

fn main() {
    std::process::exit(if gate::run() { 0 } else { 1 });
}
