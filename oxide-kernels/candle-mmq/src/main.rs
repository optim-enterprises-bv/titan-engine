#![allow(unsafe_op_in_unsafe_fn)]
//! candle-kernels `mmq_gguf/` (llama.cpp's tiled quantized matmul, "MMQ") in cuda-oxide, plus
//! pure-Rust host launchers (src/launch.rs) with the exact `ffi.rs` signatures, bit-identical to
//! the statically linked C launchers + nvcc kernels in libmoe.a (sm_120a, -O3, no fast-math).
//!
//! What the reference does on sm_120 (read off the SASS, not only the source):
//! - tensor-core path: A tiles via `ldmatrix.x4` (LDSM.16.M88.4), B tiles by plain loads, one
//!   `mma.sync.m16n8k32.s32.s8.s8.s32` (IMMA.16832.S8.S8) per 16x8 output minitile per 32-value
//!   k step, accumulator reset to zero for every k step (integer results are exact);
//! - float epilogue per k step, in source order: `sum = fma(dA*dB, (float)C, sum)`
//!   (FMUL then FFMA), then `sum = fma(mA, sB, sum)` (Q8_1-style types);
//! - scales combined in half precision in load_tiles (HMUL2), converted with HADD2.F32;
//! - stream-k decomposition over nsm CUDA blocks; a block whose range ends inside an output tile
//!   writes its whole partial tile to tmp_fixup, and the fixup kernel adds the partials of the
//!   preceding blocks (in descending block order, starting from 0) onto dst.
mod launch;

use cuda_device::{DynamicSharedArray, convert, f16x2, float, kernel, launch_bounds, ptx_asm, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub mod kernels {
    use super::*;

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
    fn fma(a: f32, b: f32, c: f32) -> f32 {
        float::fma_rn_f32(a, b, c)
    }
    #[inline(always)]
    fn mul(a: f32, b: f32) -> f32 {
        float::mul_rn_f32(a, b)
    }
    #[inline(always)]
    fn sub(a: f32, b: f32) -> f32 {
        float::add_rn_f32(a, -b)
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
                        let s = float::fma_rn_f32(float::mul_rn_f32(h2lo(da), h2lo(db)), c[l as usize] as f32, sum[idx]);
                        sum[idx] = float::fma_rn_f32(h2hi(da), h2hi(db), s);
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
    // write back + tile processing + stream-k kernel.

    /// mmq_write_back_mma (ids_dst is the identity for dense matmuls).
    #[inline(always)]
    unsafe fn write_back<const MMQ_X: i32, const NC: bool>(sum: &[f32; 64], dst: *mut f32, stride: i32, i_max: i32, j_max: i32) {
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
                        *dst.offset(j.wrapping_mul(stride).wrapping_add(i) as isize) = sum[((j0 / 8 + n) * 4 + l) as usize];
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
        x: *const u8, offset_x: i32, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32,
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
            write_back::<MMQ_X, NC>(&sum, t, MMQ_Y, MMQ_Y, MMQ_X);
        } else {
            write_back::<MMQ_X, NC>(&sum, dst, stride_col_dst, tile_x_max_i, tile_y_max_j);
        }
    }

    /// Stream-k `mul_mat_q` for the dense case (ids_dst = expert_bounds = null, one channel,
    /// one sample: what the candle launchers always pass).
    #[inline(always)]
    pub unsafe fn mul_mat_q<const T: u32, const MMQ_X: i32, const NC: bool>(
        x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32,
        ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32,
    ) {
        let qk = qk(T);
        let ntx = ncols_max.wrapping_add(MMQ_X - 1) / MMQ_X;
        let nty = nrows_x.wrapping_add(MMQ_Y - 1) / MMQ_Y;
        let bpn: i64 = (ncols_x / qk) as i64;
        let bpi: i64 = (ITER_K / qk) as i64;
        let total = (ntx as i64).wrapping_mul(nty as i64).wrapping_mul(bpn);
        let bid = thread::blockIdx_x() as i64;
        let gdim = thread::gridDim_x() as i64;
        let mut kbc = bid.wrapping_mul(total) / gdim;
        let mut kbc_stop = (bid + 1).wrapping_mul(total) / gdim;
        kbc -= (kbc % bpn) % bpi;
        kbc_stop -= (kbc_stop % bpn) % bpi;

        let tile_bpn = (ntx as i64).wrapping_mul(bpn); // nsamples*nchannels*ntx*blocks_per_ne00
        let mut kb0_start = (kbc % bpn) as i32;
        let mut kb0_stop = {
            let v = (kb0_start as i64) + kbc_stop - kbc;
            (if bpn < v { bpn } else { v }) as i32
        };
        while kbc < kbc_stop && kb0_stop as i64 == bpn {
            let tmp = kbc as i32;
            let it = ((tmp as i64) / tile_bpn) as i32;
            let tmp = tmp.wrapping_sub(it.wrapping_mul(tile_bpn as i32));
            let jt = ((tmp as i64) / bpn) as i32;
            let offset_y = jt.wrapping_mul(MMQ_X).wrapping_mul(SZ);
            let offset_dst = jt.wrapping_mul(MMQ_X).wrapping_mul(stride_col_dst).wrapping_add(it.wrapping_mul(MMQ_Y));
            let tile_x_max_i = nrows_x - it.wrapping_mul(MMQ_Y) - 1;
            let tile_y_max_j = ncols_dst - jt.wrapping_mul(MMQ_X) - 1;
            let offset_x = it.wrapping_mul(MMQ_Y).wrapping_mul(stride_row_x);
            process_tile::<T, MMQ_X, NC, false>(
                x, offset_x, y.wrapping_offset(offset_y as isize), dst.wrapping_offset(offset_dst as isize), tmp_fixup,
                stride_row_x, ncols_y, stride_col_dst, tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop,
            );
            kbc += bpn;
            kbc -= kbc % bpn;
            kb0_start = 0;
            kb0_stop = {
                let v = kbc_stop - kbc;
                (if bpn < v { bpn } else { v }) as i32
            };
        }
        if kbc >= kbc_stop {
            return;
        }
        let tmp = kbc as i32;
        let it = ((tmp as i64) / tile_bpn) as i32;
        let tmp = tmp.wrapping_sub(it.wrapping_mul(tile_bpn as i32));
        let jt = ((tmp as i64) / bpn) as i32;
        let offset_y = jt.wrapping_mul(MMQ_X).wrapping_mul(SZ);
        let offset_dst = jt.wrapping_mul(MMQ_X).wrapping_mul(stride_col_dst).wrapping_add(it.wrapping_mul(MMQ_Y));
        let tile_x_max_i = nrows_x - it.wrapping_mul(MMQ_Y) - 1;
        let tile_y_max_j = ncols_dst - jt.wrapping_mul(MMQ_X) - 1;
        let offset_x = it.wrapping_mul(MMQ_Y).wrapping_mul(stride_row_x);
        process_tile::<T, MMQ_X, NC, true>(
            x, offset_x, y.wrapping_offset(offset_y as isize), dst.wrapping_offset(offset_dst as isize), tmp_fixup,
            stride_row_x, ncols_y, stride_col_dst, tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop,
        );
    }

    /// mul_mat_q_stream_k_fixup (dense case). Depends on the type only through qk.
    #[inline(always)]
    pub unsafe fn stream_k_fixup<const QK: i32, const MMQ_X: i32, const NC: bool>(
        dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32,
    ) {
        let bpi: i64 = (ITER_K / QK) as i64;
        let bpn: i64 = (ncols_x / QK) as i64;
        let mut sum = [0f32; 64];
        let ntx = ncols_max.wrapping_add(MMQ_X - 1) / MMQ_X;
        let nty = nrows_x.wrapping_add(MMQ_Y - 1) / MMQ_Y;
        let total = (ntx as i64).wrapping_mul(nty as i64).wrapping_mul(bpn);
        let gdim = thread::gridDim_x() as i64;
        let bidx0 = thread::blockIdx_x() as i64;
        let mut kbc0 = bidx0.wrapping_mul(total) / gdim;
        let mut kbc0_stop = (bidx0 + 1).wrapping_mul(total) / gdim;
        kbc0 -= (kbc0 % bpn) % bpi;
        kbc0_stop -= (kbc0_stop % bpn) % bpi;

        let did_not_have_any_data = kbc0 == kbc0_stop;
        let wrote_beginning_of_tile = kbc0 % bpn == 0;
        let did_not_write_last = kbc0 / bpn == kbc0_stop / bpn && kbc0_stop % bpn != 0;
        if did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last {
            return;
        }

        let mut any_fixup = false;
        let mut bidx = bidx0 - 1;
        let mut kbc_stop = kbc0;
        loop {
            let mut kbc = bidx.wrapping_mul(total) / gdim;
            kbc -= (kbc % bpn) % bpi;
            if kbc == kbc_stop {
                bidx -= 1;
                kbc_stop = kbc;
                continue;
            }
            any_fixup = true;
            let base = tmp_last_tile.wrapping_offset(bidx.wrapping_mul((MMQ_X * MMQ_Y) as i64) as isize);
            let mut j0 = 0;
            while j0 < MMQ_X {
                unroll!();
                let j = j0 + ty();
                let mut i0 = 0;
                while i0 < MMQ_Y {
                    unroll!();
                    let i = i0 + tx();
                    let idx = ((j0 / NWARPS) * (MMQ_Y / WARP) + i0 / WARP) as usize;
                    sum[idx] = float::add_rn_f32(sum[idx], *base.offset((j * MMQ_Y + i) as isize));
                    i0 += WARP;
                }
                j0 += NWARPS;
            }
            if kbc % bpn == 0 || kbc / bpn < kbc0 / bpn {
                break;
            }
            bidx -= 1;
            kbc_stop = kbc;
        }
        if !any_fixup {
            return;
        }

        let tmp = kbc0 as i32;
        let tile_bpn = (ntx as i64).wrapping_mul(bpn);
        let it = ((tmp as i64) / tile_bpn) as i32;
        let tmp = tmp.wrapping_sub(it.wrapping_mul(tile_bpn as i32));
        let jt = ((tmp as i64) / bpn) as i32;

        // size_t arithmetic, then truncated to int.
        let offset_dst = ((jt.wrapping_mul(MMQ_X) as i64 as u64).wrapping_mul(stride_col_dst)
            .wrapping_add(it.wrapping_mul(MMQ_Y) as i64 as u64)) as i32;
        let dst = dst.wrapping_offset(offset_dst as isize);
        let i_max = nrows_x - it.wrapping_mul(MMQ_Y) - 1;
        let j_max = ncols_dst - jt.wrapping_mul(MMQ_X) - 1;
        let mut j0 = 0;
        while j0 < MMQ_X {
            unroll!();
            let j = j0 + ty();
            if j > j_max {
                return;
            }
            let mut i0 = 0;
            while i0 < MMQ_Y {
                unroll!();
                let i = i0 + tx();
                if !(NC && i > i_max) {
                    let idx = ((j0 / NWARPS) * (MMQ_Y / WARP) + i0 / WARP) as usize;
                    let o = (j as i64 as u64).wrapping_mul(stride_col_dst).wrapping_add(i as i64 as u64);
                    let p = dst.wrapping_add(o as usize);
                    *p = float::add_rn_f32(*p, sum[idx]);
                }
                i0 += WARP;
            }
            j0 += NWARPS;
        }
    }


    // ---------------------------------------------------------------------------------------
    // quantize_mmq_q8_1<ds_layout>: f32 -> block_q8_1_mmq. LAYOUT 0 = D4, 1 = DS4, 2 = D2S6.

    /// Hardware `max.f32` (fmaxf: a NaN operand yields the other), `abs.f32`.
    #[inline(always)]
    fn fmaxf(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("max.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)) };
        r
    }
    #[inline(always)]
    fn fabsf(a: f32) -> f32 {
        f32::from_bits(a.to_bits() & 0x7FFF_FFFF)
    }
    /// nvcc roundf: trunc_rz(t + copysign(0.5, t)), then float -> int (cvt.rzi.s32, saturating).
    #[inline(always)]
    fn roundf_i8(t: f32) -> u32 {
        let half = f32::from_bits(0x3F00_0000 | (t.to_bits() & 0x8000_0000));
        (float::add_rz_f32(t, half) as i32 as u32) & 0xFF
    }

    #[inline(always)]
    pub unsafe fn quantize_mmq_q8_1<const LAYOUT: u32>(
        x: *const f32, ids: *const i32, vy: *mut u8, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i32, ne2: i32,
    ) {
        let vals_per_scale: u32 = if LAYOUT == 2 { 64 } else { 32 };
        let vals_per_sum: u32 = if LAYOUT == 2 { 16 } else { 32 };
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
        let (a, b, c, d) = if i0 < ne00 {
            let p = x.offset(((i3 * s03 + i2 * s02 + i01 * s01 + i0) / 4 * 4) as isize);
            (*p, *p.add(1), *p.add(2), *p.add(3))
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };
        let mut amax = fabsf(a);
        amax = fmaxf(amax, fabsf(b));
        amax = fmaxf(amax, fabsf(c));
        amax = fmaxf(amax, fabsf(d));
        let mut off = vals_per_scale / 8;
        while off > 0 {
            amax = fmaxf(amax, cuda_device::warp::shuffle_xor_f32_sync(0xFFFF_FFFF, amax, off));
            off >>= 1;
        }
        let mut sum = 0f32;
        if LAYOUT != 0 {
            sum = float::add_rn_f32(float::add_rn_f32(float::add_rn_f32(a, b), c), d);
            let mut off = vals_per_sum / 8;
            while off > 0 {
                sum = float::add_rn_f32(sum, cuda_device::warp::shuffle_xor_f32_sync(0xFFFF_FFFF, sum, off));
                off >>= 1;
            }
        }
        let d_inv = float::div_rn_f32(127.0, amax);
        let q = roundf_i8(float::mul_rn_f32(a, d_inv))
            | (roundf_i8(float::mul_rn_f32(b, d_inv)) << 8)
            | (roundf_i8(float::mul_rn_f32(c, d_inv)) << 16)
            | (roundf_i8(float::mul_rn_f32(d, d_inv)) << 24);
        let blk = vy.offset((ib * 144) as isize);
        *(blk.offset((16 + iqs) as isize) as *mut u32) = q;
        if LAYOUT == 2 {
            if iqs % 16 != 0 || iqs >= 96 {
                return;
            }
            *(blk.offset((2 * (2 + iqs / 16)) as isize) as *mut u16) = convert::cvt_f16x2_f32(sum, 0.0) as u16;
            if iqs % 64 != 0 {
                return;
            }
            let dd = float::rcp_rn_f32(d_inv);
            *(blk.offset((2 * (iqs / 64)) as isize) as *mut u16) = convert::cvt_f16x2_f32(dd, 0.0) as u16;
            return;
        }
        if iqs % 32 != 0 {
            return;
        }
        let dd = float::rcp_rn_f32(d_inv);
        if LAYOUT == 1 {
            *(blk.offset((4 * (iqs / 32)) as isize) as *mut u32) = convert::cvt_f16x2_f32(dd, sum);
        } else {
            *(blk.offset((4 * (iqs / 32)) as isize) as *mut f32) = dd;
        }
    }

    #[kernel] pub unsafe fn quantize_mmq_q8_1_d4(x: *const f32, ids: *const i32, vy: *mut u8, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i32, ne2: i32) { unsafe { quantize_mmq_q8_1::<0>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2) } }
    #[kernel] pub unsafe fn quantize_mmq_q8_1_ds4(x: *const f32, ids: *const i32, vy: *mut u8, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i32, ne2: i32) { unsafe { quantize_mmq_q8_1::<1>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2) } }
    #[kernel] pub unsafe fn quantize_mmq_q8_1_d2s6(x: *const f32, ids: *const i32, vy: *mut u8, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i32, ne2: i32) { unsafe { quantize_mmq_q8_1::<2>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2) } }

    // GENERATED KERNELS BEGIN (gen_kernels.py)
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x8_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_0, 8, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x8_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_0, 8, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x16_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_0, 16, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x16_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_0, 16, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x24_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_0, 24, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x24_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_0, 24, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x32_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_0, 32, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x32_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_0, 32, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x40_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_0, 40, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x40_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_0, 40, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x48_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_0, 48, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x48_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_0, 48, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x64_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_0, 64, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x64_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_0, 64, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x80_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_0, 80, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x80_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_0, 80, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x96_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_0, 96, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x96_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_0, 96, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x112_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_0, 112, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x112_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_0, 112, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x128_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_0, 128, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_0_x128_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_0, 128, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x8_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_1, 8, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x8_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_1, 8, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x16_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_1, 16, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x16_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_1, 16, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x24_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_1, 24, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x24_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_1, 24, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x32_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_1, 32, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x32_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_1, 32, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x40_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_1, 40, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x40_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_1, 40, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x48_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_1, 48, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x48_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_1, 48, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x64_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_1, 64, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x64_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_1, 64, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x80_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_1, 80, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x80_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_1, 80, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x96_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_1, 96, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x96_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_1, 96, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x112_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_1, 112, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x112_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_1, 112, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x128_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_1, 128, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_1_x128_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_1, 128, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x8_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_0, 8, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x8_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_0, 8, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x16_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_0, 16, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x16_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_0, 16, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x24_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_0, 24, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x24_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_0, 24, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x32_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_0, 32, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x32_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_0, 32, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x40_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_0, 40, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x40_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_0, 40, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x48_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_0, 48, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x48_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_0, 48, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x64_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_0, 64, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x64_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_0, 64, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x80_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_0, 80, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x80_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_0, 80, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x96_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_0, 96, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x96_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_0, 96, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x112_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_0, 112, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x112_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_0, 112, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x128_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_0, 128, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_0_x128_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_0, 128, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x8_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_1, 8, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x8_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_1, 8, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x16_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_1, 16, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x16_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_1, 16, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x24_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_1, 24, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x24_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_1, 24, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x32_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_1, 32, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x32_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_1, 32, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x40_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_1, 40, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x40_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_1, 40, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x48_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_1, 48, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x48_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_1, 48, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x64_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_1, 64, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x64_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_1, 64, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x80_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_1, 80, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x80_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_1, 80, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x96_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_1, 96, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x96_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_1, 96, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x112_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_1, 112, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x112_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_1, 112, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x128_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_1, 128, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_1_x128_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_1, 128, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x8_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q8_0, 8, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x8_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q8_0, 8, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x16_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q8_0, 16, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x16_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q8_0, 16, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x24_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q8_0, 24, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x24_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q8_0, 24, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x32_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q8_0, 32, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x32_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q8_0, 32, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x40_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q8_0, 40, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x40_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q8_0, 40, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x48_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q8_0, 48, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x48_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q8_0, 48, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x64_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q8_0, 64, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x64_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q8_0, 64, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x80_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q8_0, 80, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x80_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q8_0, 80, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x96_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q8_0, 96, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x96_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q8_0, 96, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x112_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q8_0, 112, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x112_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q8_0, 112, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x128_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q8_0, 128, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q8_0_x128_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q8_0, 128, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x8_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q2_K, 8, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x8_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q2_K, 8, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x16_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q2_K, 16, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x16_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q2_K, 16, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x24_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q2_K, 24, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x24_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q2_K, 24, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x32_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q2_K, 32, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x32_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q2_K, 32, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x40_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q2_K, 40, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x40_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q2_K, 40, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x48_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q2_K, 48, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x48_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q2_K, 48, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x64_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q2_K, 64, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x64_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q2_K, 64, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x80_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q2_K, 80, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x80_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q2_K, 80, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x96_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q2_K, 96, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x96_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q2_K, 96, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x112_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q2_K, 112, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x112_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q2_K, 112, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x128_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q2_K, 128, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q2_k_x128_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q2_K, 128, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x8_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q3_K, 8, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x8_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q3_K, 8, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x16_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q3_K, 16, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x16_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q3_K, 16, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x24_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q3_K, 24, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x24_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q3_K, 24, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x32_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q3_K, 32, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x32_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q3_K, 32, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x40_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q3_K, 40, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x40_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q3_K, 40, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x48_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q3_K, 48, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x48_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q3_K, 48, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x64_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q3_K, 64, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x64_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q3_K, 64, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x80_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q3_K, 80, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x80_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q3_K, 80, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x96_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q3_K, 96, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x96_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q3_K, 96, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x112_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q3_K, 112, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x112_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q3_K, 112, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x128_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q3_K, 128, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q3_k_x128_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q3_K, 128, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x8_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_K, 8, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x8_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_K, 8, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x16_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_K, 16, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x16_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_K, 16, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x24_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_K, 24, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x24_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_K, 24, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x32_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_K, 32, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x32_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_K, 32, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x40_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_K, 40, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x40_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_K, 40, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x48_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_K, 48, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x48_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_K, 48, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x64_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_K, 64, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x64_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_K, 64, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x80_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_K, 80, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x80_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_K, 80, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x96_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_K, 96, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x96_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_K, 96, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x112_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_K, 112, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x112_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_K, 112, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x128_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_K, 128, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q4_k_x128_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q4_K, 128, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x8_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_K, 8, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x8_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_K, 8, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x16_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_K, 16, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x16_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_K, 16, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x24_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_K, 24, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x24_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_K, 24, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x32_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_K, 32, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x32_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_K, 32, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x40_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_K, 40, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x40_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_K, 40, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x48_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_K, 48, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x48_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_K, 48, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x64_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_K, 64, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x64_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_K, 64, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x80_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_K, 80, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x80_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_K, 80, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x96_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_K, 96, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x96_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_K, 96, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x112_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_K, 112, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x112_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_K, 112, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x128_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_K, 128, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q5_k_x128_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q5_K, 128, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x8_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q6_K, 8, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x8_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q6_K, 8, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x16_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q6_K, 16, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x16_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q6_K, 16, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x24_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q6_K, 24, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x24_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q6_K, 24, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x32_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q6_K, 32, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x32_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q6_K, 32, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x40_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q6_K, 40, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x40_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q6_K, 40, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x48_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q6_K, 48, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x48_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q6_K, 48, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x64_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q6_K, 64, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x64_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q6_K, 64, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x80_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q6_K, 80, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x80_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q6_K, 80, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x96_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q6_K, 96, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x96_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q6_K, 96, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x112_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q6_K, 112, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x112_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q6_K, 112, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x128_nc0(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q6_K, 128, false>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_q6_k_x128_nc1(x: *const u8, y: *const i32, dst: *mut f32, tmp_fixup: *mut f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_row_x: i32, ncols_y: i32, stride_col_dst: i32, ncols_max: i32) { unsafe { mul_mat_q::<Q6_K, 128, true>(x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk32_x8_nc0(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<32, 8, false>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk32_x8_nc1(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<32, 8, true>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk32_x16_nc0(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<32, 16, false>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk32_x16_nc1(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<32, 16, true>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk32_x24_nc0(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<32, 24, false>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk32_x24_nc1(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<32, 24, true>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk32_x32_nc0(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<32, 32, false>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk32_x32_nc1(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<32, 32, true>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk32_x40_nc0(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<32, 40, false>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk32_x40_nc1(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<32, 40, true>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk32_x48_nc0(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<32, 48, false>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk32_x48_nc1(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<32, 48, true>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk32_x64_nc0(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<32, 64, false>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk32_x64_nc1(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<32, 64, true>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk32_x80_nc0(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<32, 80, false>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk32_x80_nc1(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<32, 80, true>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk32_x96_nc0(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<32, 96, false>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk32_x96_nc1(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<32, 96, true>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk32_x112_nc0(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<32, 112, false>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk32_x112_nc1(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<32, 112, true>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk32_x128_nc0(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<32, 128, false>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk32_x128_nc1(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<32, 128, true>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk256_x8_nc0(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<256, 8, false>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk256_x8_nc1(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<256, 8, true>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk256_x16_nc0(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<256, 16, false>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk256_x16_nc1(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<256, 16, true>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk256_x24_nc0(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<256, 24, false>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk256_x24_nc1(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<256, 24, true>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk256_x32_nc0(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<256, 32, false>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk256_x32_nc1(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<256, 32, true>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk256_x40_nc0(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<256, 40, false>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk256_x40_nc1(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<256, 40, true>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk256_x48_nc0(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<256, 48, false>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk256_x48_nc1(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<256, 48, true>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk256_x64_nc0(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<256, 64, false>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk256_x64_nc1(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<256, 64, true>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk256_x80_nc0(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<256, 80, false>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk256_x80_nc1(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<256, 80, true>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk256_x96_nc0(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<256, 96, false>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk256_x96_nc1(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<256, 96, true>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk256_x112_nc0(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<256, 112, false>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk256_x112_nc1(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<256, 112, true>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk256_x128_nc0(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<256, 128, false>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    #[kernel] #[launch_bounds(256, 1)] pub unsafe fn mmq_fixup_qk256_x128_nc1(dst: *mut f32, tmp_last_tile: *const f32, ncols_x: i32, nrows_x: i32, ncols_dst: i32, stride_col_dst: u64, ncols_max: i32) { unsafe { stream_k_fixup::<256, 128, true>(dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, ncols_max) } }
    // GENERATED KERNELS END
}

fn main() {
    std::process::exit(gate::run());
}

mod gate;
