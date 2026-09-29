//! candle-kernels `mmvq_gguf.cu` in cuda-oxide: the 240 `mmvq_gguf_<q>_<dst>_plain_cudaN` GEMV
//! kernels and the 3 `mmvq_gguf_quantize_q8_1_<t>` activation quantizers, under the reference's own
//! entry names and parameter ABI, plus (src/launch.rs) pure-Rust twins of the 33 extern "C" host
//! launchers `launch_mmvq_gguf_*`. Bit-identical to the nvcc build (SASS in libmoe.a /
//! mmvq_gguf.cubin; -O3, no fast-math): checked by the gate in src/gate.rs.
//!
//! The float program is read off the SASS (every contraction ptxas made is an explicit fma here):
//! - q4_0 / q5_0: `t = ds.y * -(4|8)` (FMUL), `r = fma(ds.x, sumi, t)`, `tmp = fma(d, r, tmp)`.
//! - q4_1 / q5_1: `r = fma(dm.x*ds.x, sumi, (dm.y*ds.y)*0.5)` (FMUL.D2), `tmp = r + tmp`.
//! - q8_0: `tmp = fma(sumi * d0, d1, tmp)` (left-to-right product, NOT d0*d1 first).
//! - q2_K / q4_K / q5_K: `sumf_{d,m} = fma(d8[i], float(int), sumf)`, then
//!   `r = fma(sumf_d, dm.x, -(sumf_m * dm.y))`, `tmp = r + tmp`.
//! - q3_K / q6_K: `sumf = fma(d8[i], float(int), sumf)`, `tmp = fma(d, sumf, tmp)`.
//! - block reduction: warps 1.. store to shared; warp 0 adds them in warp order, then a butterfly
//!   over xor 16, 8, 4, 2, 1; the row index into dst is 32-bit unsigned (C promotes to unsigned).
//! - quantize: hardware `max.f32` warp max, `d = amax / 127` and `x / d` as div.rn,
//!   roundf = `cvt.rzi(add.rz(t, copysign(0.5, t)))`. The add.rz-vs-add.rn difference is only
//!   observable for f32 input (t = +-0.49999997); an exhaustive search finds no bf16 / f16 input
//!   that reaches it, so the gate uses searched f32 activations for it.
//! - `(dm.y*ds.y)*0.5` (FMUL.D2) vs `dm.y*(ds.y*0.5)` cannot differ for f16 scales (no product is
//!   subnormal or overflows in f32), so that contraction choice is unobservable.
//!
//! Build: `cargo oxide build --arch sm_120`; `python3 gen_kernels.py` regenerates the 240 wrappers.
#![allow(non_snake_case, clippy::missing_safety_doc)]
mod gate;
pub mod launch;

use cuda_device::{SharedArray, convert, device, dotprod, float, kernel, ptx_asm, thread, warp};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(always)]
    pub fn h2f(bits: u32) -> f32 {
        convert::cvt_f32_f16x2_lo(bits & 0xFFFF)
    }
    #[inline(always)]
    pub fn fma(a: f32, b: f32, c: f32) -> f32 {
        float::fma_rn_f32(a, b, c)
    }
    #[inline(always)]
    pub fn mul(a: f32, b: f32) -> f32 {
        float::mul_rn_f32(a, b)
    }
    #[inline(always)]
    pub fn add(a: f32, b: f32) -> f32 {
        float::add_rn_f32(a, b)
    }
    #[inline(always)]
    pub fn dp4a(a: u32, b: u32, c: i32) -> i32 {
        dotprod::dp4a_s32(a, b, c)
    }
    /// Hardware `max.f32` (fmaxf), not compare+select.
    #[inline(always)]
    pub fn fmaxf(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("max.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    /// `cvt.rzi.s32.f32` (C float -> int conversion).
    #[inline(always)]
    pub fn f2i_rz(a: f32) -> i32 {
        let r: i32;
        unsafe { ptx_asm!("cvt.rzi.s32.f32 %0, %1;", out("=r") r, in("f") a, options(register_only)); }
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
    pub unsafe fn ld8(p: *const u8, off: usize) -> u32 {
        *p.add(off) as u32
    }
    #[inline(always)]
    pub unsafe fn ld16(p: *const u8, off: usize) -> u32 {
        *(p.add(off) as *const u16) as u32
    }
    /// `get_int_from_uint8`: a 32-bit word from two 16-bit loads (2-byte aligned blocks).
    #[inline(always)]
    pub unsafe fn ld32u(p: *const u8, off: usize) -> u32 {
        ld16(p, off) | (ld16(p, off + 2) << 16)
    }
    /// `get_int_from_uint8_aligned`.
    #[inline(always)]
    pub unsafe fn ld32a(p: *const u8, off: usize) -> u32 {
        *(p.add(off) as *const u32)
    }
    /// Q8_1 block: quant word `i`.
    #[inline(always)]
    pub unsafe fn yq(yb: *const u8, i: usize) -> u32 {
        ld32a(yb, 4 + 4 * i)
    }
    /// Arithmetic shift right of an `int`.
    #[inline(always)]
    pub fn sar(x: u32, n: usize) -> u32 {
        ((x as i32) >> n) as u32
    }

    /// Per-byte signed saturating subtract (`__vsubss4`).
    #[inline(always)]
    pub fn vsubss4(a: u32, b: u32) -> u32 {
        let mut r = 0u32;
        let mut k = 0;
        while k < 4 {
            let x = ((a >> (8 * k)) & 0xFF) as u8 as i8 as i32;
            let y = ((b >> (8 * k)) & 0xFF) as u8 as i8 as i32;
            let mut d = x - y;
            if d > 127 {
                d = 127;
            }
            if d < -128 {
                d = -128;
            }
            r |= ((d as u32) & 0xFF) << (8 * k);
            k += 1;
        }
        r
    }

    /// Output element of the GEMV.
    pub trait Dst: Copy {
        unsafe fn st(p: *mut Self, v: f32);
    }
    impl Dst for f32 {
        #[inline(always)]
        unsafe fn st(p: *mut f32, v: f32) {
            *p = v;
        }
    }
    /// f16 output.
    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct H(pub u16);
    /// bf16 output.
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

    /// (qk, qi, vdr, block bytes). FMT: 0 q4_0, 1 q4_1, 2 q5_0, 3 q5_1, 4 q8_0, 5 q2_K, 6 q3_K,
    /// 7 q4_K, 8 q5_K, 9 q6_K.
    #[inline(always)]
    pub fn fmt_params<const FMT: u32>() -> (i32, i32, i32, i32) {
        match FMT {
            0 => (32, 4, 2, 18),
            1 => (32, 4, 2, 20),
            2 => (32, 4, 2, 22),
            3 => (32, 4, 2, 24),
            4 => (32, 8, 2, 34),
            5 => (256, 16, 1, 84),
            6 => (256, 16, 1, 110),
            7 => (256, 32, 2, 144),
            8 => (256, 32, 2, 176),
            _ => (256, 32, 1, 210),
        }
    }

    /// `tmp += vec_dot_<FMT>_q8_1(xb, yb, iqs)` with the reference's contraction of the `+=`.
    /// `xb`: the weight block, `yb`: the first Q8_1 block it pairs with.
    #[inline(always)]
    pub unsafe fn vdot<const FMT: u32>(tmp: f32, xb: *const u8, yb: *const u8, iqs: usize) -> f32 {
        match FMT {
            0 | 1 | 2 | 3 => {
                let mut sumi = 0i32;
                let mut i = 0;
                while i < 2 {
                    let u0 = yq(yb, iqs + i);
                    let u1 = yq(yb, iqs + i + 4);
                    let (vi0, vi1) = if FMT == 0 || FMT == 1 {
                        let v = if FMT == 0 { ld32u(xb, 2 + 4 * (iqs + i)) } else { ld32a(xb, 4 + 4 * (iqs + i)) };
                        (v & 0x0F0F0F0F, sar(v, 4) & 0x0F0F0F0F)
                    } else {
                        let (vl, qh) = if FMT == 2 {
                            (ld32u(xb, 6 + 4 * (iqs + i)), ld32u(xb, 2))
                        } else {
                            (ld32a(xb, 8 + 4 * (iqs + i)), ld32a(xb, 4))
                        };
                        let vh = sar(qh, 4 * (iqs + i));
                        let mut vi0 = vl & 0x0F0F0F0F;
                        vi0 |= (vh << 4) & 0x00000010;
                        vi0 |= (vh << 11) & 0x00001000;
                        vi0 |= (vh << 18) & 0x00100000;
                        vi0 |= (vh << 25) & 0x10000000;
                        let mut vi1 = sar(vl, 4) & 0x0F0F0F0F;
                        vi1 |= sar(vh, 12) & 0x00000010;
                        vi1 |= sar(vh, 5) & 0x00001000;
                        vi1 |= (vh << 2) & 0x00100000;
                        vi1 |= (vh << 9) & 0x10000000;
                        (vi0, vi1)
                    };
                    sumi = dp4a(vi0, u0, sumi);
                    sumi = dp4a(vi1, u1, sumi);
                    i += 1;
                }
                let ds = ld32a(yb, 0);
                let dsx = h2f(ds);
                let dsy = h2f(ds >> 16);
                let sf = sumi as f32;
                if FMT == 0 || FMT == 2 {
                    let c = if FMT == 0 { -4.0 } else { -8.0 };
                    fma(h2f(ld16(xb, 0)), fma(dsx, sf, mul(dsy, c)), tmp)
                } else {
                    let dm = ld32a(xb, 0);
                    let dd = mul(h2f(dm), dsx);
                    let ms = mul(mul(h2f(dm >> 16), dsy), 0.5);
                    add(fma(dd, sf, ms), tmp)
                }
            }
            4 => {
                let mut sumi = 0i32;
                let mut i = 0;
                while i < 2 {
                    sumi = dp4a(ld32u(xb, 2 + 4 * (iqs + i)), yq(yb, iqs + i), sumi);
                    i += 1;
                }
                fma(mul(sumi as f32, h2f(ld16(xb, 0))), h2f(ld16(yb, 0)), tmp)
            }
            5 => {
                let bq8_offset = 4 * (iqs / 8);
                let scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
                let v = ld32a(xb, 16 + 4 * iqs);
                let mut sumf_d = 0f32;
                let mut sumf_m = 0f32;
                let mut i = 0;
                while i < 4 {
                    let b8 = yb.add((bq8_offset + i) * 36);
                    let u = yq(b8, iqs % 8);
                    let d8 = h2f(ld16(b8, 0));
                    let sc = ld8(xb, scale_offset + 2 * i);
                    let vi = (v >> (2 * i)) & 0x03030303;
                    sumf_d = fma(d8, dp4a(vi, u, 0).wrapping_mul((sc & 0xF) as i32) as f32, sumf_d);
                    let mut m = sc >> 4;
                    m |= m << 8;
                    m |= m << 16;
                    sumf_m = fma(d8, dp4a(m, u, 0) as f32, sumf_m);
                    i += 1;
                }
                let dm = ld32a(xb, 80);
                add(fma(sumf_d, h2f(dm), -mul(sumf_m, h2f(dm >> 16))), tmp)
            }
            6 => {
                let bq8_offset = 4 * (iqs / 8);
                let scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
                let vl = ld32u(xb, 32 + 4 * iqs);
                let vh = sar(!ld32u(xb, 4 * (iqs % 8)), bq8_offset);
                let mut sumf = 0f32;
                let mut i = 0;
                while i < 4 {
                    let b8 = yb.add((bq8_offset + i) * 36);
                    let u = yq(b8, iqs % 8);
                    let d8 = h2f(ld16(b8, 0));
                    let isc = scale_offset + 2 * i;
                    let sc_low = (ld8(xb, 96 + isc % 8) >> (4 * (isc / 8))) & 0xF;
                    let sc_high = ((ld8(xb, 96 + 8 + isc % 4) >> (2 * (isc / 4))) & 3) << 4;
                    let sc = (sc_low | sc_high) as i32 - 32;
                    let vil = (vl >> (2 * i)) & 0x03030303;
                    let vih = (sar(vh, i) << 2) & 0x04040404;
                    let vi = vsubss4(vil, vih);
                    sumf = fma(d8, dp4a(vi, u, 0).wrapping_mul(sc) as f32, sumf);
                    i += 1;
                }
                fma(h2f(ld16(xb, 108)), sumf, tmp)
            }
            7 | 8 => {
                let bq8_offset = 2 * ((iqs / 2) / 4);
                let (v0, v1) = if FMT == 7 {
                    let q4 = 16 + 16 * bq8_offset + 4 * ((iqs / 2) % 4);
                    let (a, b) = (ld32a(xb, q4), ld32a(xb, q4 + 16));
                    ([a & 0x0F0F0F0F, b & 0x0F0F0F0F], [sar(a, 4) & 0x0F0F0F0F, sar(b, 4) & 0x0F0F0F0F])
                } else {
                    let ql = 48 + 16 * bq8_offset + 4 * ((iqs / 2) % 4);
                    let qh = 16 + 4 * ((iqs / 2) % 4);
                    let (l0, l1) = (ld32a(xb, ql), ld32a(xb, ql + 16));
                    let h0 = sar(ld32a(xb, qh), bq8_offset);
                    let h1 = sar(ld32a(xb, qh + 16), bq8_offset);
                    (
                        [(l0 & 0x0F0F0F0F) | ((h0 << 4) & 0x10101010), (l1 & 0x0F0F0F0F) | ((h1 << 4) & 0x10101010)],
                        [
                            (sar(l0, 4) & 0x0F0F0F0F) | ((sar(h0, 1) << 4) & 0x10101010),
                            (sar(l1, 4) & 0x0F0F0F0F) | ((sar(h1, 1) << 4) & 0x10101010),
                        ],
                    )
                };
                let v = [v0, v1];
                let s16 = |k: usize| ld16(xb, 4 + 2 * k);
                let j = bq8_offset / 2;
                let (aux0, aux1) = if j < 2 {
                    (s16(j) & 0x3f3f, s16(j + 2) & 0x3f3f)
                } else {
                    (
                        (s16(j + 2) & 0x0f0f) | ((s16(j - 2) & 0xc0c0) >> 2),
                        ((s16(j + 2) >> 4) & 0x0f0f) | ((s16(j) & 0xc0c0) >> 2),
                    )
                };
                let sc = [aux0 & 0xFF, aux0 >> 8];
                let m = [aux1 & 0xFF, aux1 >> 8];
                let mut sumf_d = 0f32;
                let mut sumf_m = 0f32;
                let mut i = 0;
                while i < 2 {
                    let b8 = yb.add((bq8_offset + i) * 36);
                    let d8 = h2f(ld16(b8, 0));
                    let u0 = yq(b8, (iqs / 2) % 4);
                    let u1 = yq(b8, (iqs / 2) % 4 + 4);
                    let dot1 = dp4a(v[i][1], u1, dp4a(v[i][0], u0, 0));
                    let dot2 = dp4a(0x01010101, u1, dp4a(0x01010101, u0, 0));
                    sumf_d = fma(d8, dot1.wrapping_mul(sc[i] as i32) as f32, sumf_d);
                    sumf_m = fma(d8, dot2.wrapping_mul(m[i] as i32) as f32, sumf_m);
                    i += 1;
                }
                let dm = ld32a(xb, 0);
                add(fma(sumf_d, h2f(dm), -mul(sumf_m, h2f(dm >> 16))), tmp)
            }
            _ => {
                let bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
                let scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
                let vh_shift = 2 * ((iqs % 16) / 8);
                let vl = ld32u(xb, 4 * iqs);
                let vh = sar(ld32u(xb, 128 + 4 * (8 * (iqs / 16) + iqs % 8)), vh_shift);
                let mut sumf = 0f32;
                let mut i = 0;
                while i < 2 {
                    let b8 = yb.add((bq8_offset + 2 * i) * 36);
                    let u = yq(b8, iqs % 8);
                    let d8 = h2f(ld16(b8, 0));
                    let sc = *xb.add(192 + scale_offset + 4 * i) as i8 as i32;
                    let vil = sar(vl, 4 * i) & 0x0F0F0F0F;
                    let vih = (sar(vh, 4 * i) << 4) & 0x30303030;
                    let vi = vsubss4(vil | vih, 0x20202020);
                    sumf = fma(d8, dp4a(vi, u, 0).wrapping_mul(sc) as f32, sumf);
                    i += 1;
                }
                fma(h2f(ld16(xb, 208)), sumf, tmp)
            }
        }
    }

    /// `mmvq_core_impl<dst_t, qk, qi, block_q_t, vdr, vec_dot, NC>`: block (32, nwarps).
    #[device]
    #[inline(always)]
    pub unsafe fn mmvq<const FMT: u32, const NC: usize, D: Dst>(
        vx: *const u8, vy: *const u8, dst: *mut D, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32,
    ) {
        // tmp_shared[nwarps - 1][NC][rows_per_cuda_block][32]: at most 3 * 4 * 2 * 32 floats.
        static mut TMP_SHARED: SharedArray<f32, 768> = SharedArray::UNINIT;
        let nwarps: i32 = if NC <= 4 { 4 } else { 2 };
        let rpb: i32 = if NC == 1 { 1 } else { 2 };
        let (qk, qi, vdr, bs) = fmt_params::<FMT>();
        let tx = thread::threadIdx_x() as i32;
        let ty = thread::threadIdx_y() as i32;
        let tid = 32 * ty + tx;
        let row0 = rpb.wrapping_mul(thread::blockIdx_x() as i32);
        let bpr = ncols_x / qk;
        let bpi = vdr * nwarps * 32 / qi;
        let mut tmp = [[0f32; 2]; NC];
        let mut kbx = tid / (qi / vdr);
        while kbx < bpr {
            let kby = kbx.wrapping_mul(qk / 32);
            let kqs = (vdr * (tid % (qi / vdr))) as usize;
            let mut j = 0;
            while j < NC {
                let yb = vy.offset((j as i32).wrapping_mul(stride_col_y).wrapping_add(kby) as isize * 36);
                let mut i = 0;
                while i < rpb as usize {
                    let wk = (row0 + i as i32).wrapping_mul(bpr).wrapping_add(kbx);
                    let xb = vx.offset(wk as isize * bs as isize);
                    tmp[j][i] = vdot::<FMT>(tmp[j][i], xb, yb, kqs);
                    i += 1;
                }
                j += 1;
            }
            kbx += bpi;
        }
        let sh = SharedArray::as_raw_mut_ptr(&raw mut TMP_SHARED);
        let idx = |l: i32, j: usize, i: usize| ((l as usize * NC + j) * rpb as usize + i) * 32 + tx as usize;
        if ty > 0 {
            let mut j = 0;
            while j < NC {
                let mut i = 0;
                while i < rpb as usize {
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
        let mut j = 0;
        while j < NC {
            let mut i = 0;
            while i < rpb as usize {
                let mut l = 0;
                while l < nwarps - 1 {
                    tmp[j][i] = add(tmp[j][i], *sh.add(idx(l, j, i)));
                    l += 1;
                }
                let mut mask = 16;
                while mask > 0 {
                    tmp[j][i] = add(tmp[j][i], warp::shuffle_xor_f32_sync(0xffff_ffff, tmp[j][i], mask));
                    mask >>= 1;
                }
                i += 1;
            }
            let row = (row0 as u32).wrapping_add(tx as u32);
            if tx < rpb && (rpb == 1 || row < nrows_x as u32) {
                let v = if tx == 0 { tmp[j][0] } else { tmp[j][1] };
                let o = ((j as i32).wrapping_mul(stride_col_dst) as u32).wrapping_add(row);
                D::st(dst.add(o as usize), v);
            }
            j += 1;
        }
    }

    /// `mmvq_gguf_quantize_q8_1_*`: one thread per padded element, block (256, 1), grid
    /// (ceil(kx_padded / 256), rows). `d = amax / 127` (div.rn), `q = (int8)roundf(x / d)`.
    #[device]
    #[inline(always)]
    pub unsafe fn quantize<T: Copy, L: Fn(*const T, isize) -> f32>(x: *const T, vy: *mut u8, kx: i32, kx_padded: i32, load: L) {
        let ix = thread::blockDim_x().wrapping_mul(thread::blockIdx_x()).wrapping_add(thread::threadIdx_x()) as i32;
        if ix >= kx_padded {
            return;
        }
        let iy = thread::blockDim_y().wrapping_mul(thread::blockIdx_y()).wrapping_add(thread::threadIdx_y()) as i32;
        let i_padded = iy.wrapping_mul(kx_padded).wrapping_add(ix);
        let ib = i_padded / 32;
        let iqs = i_padded % 32;
        let xi = if ix < kx { load(x, iy.wrapping_mul(kx).wrapping_add(ix) as isize) } else { 0.0 };
        let mut amax = f32::from_bits(xi.to_bits() & 0x7FFF_FFFF);
        let mut sum = xi;
        let mut mask = 16;
        while mask > 0 {
            amax = fmaxf(amax, warp::shuffle_xor_f32_sync(0xffff_ffff, amax, mask));
            mask >>= 1;
        }
        let mut mask = 16;
        while mask > 0 {
            sum = add(sum, warp::shuffle_xor_f32_sync(0xffff_ffff, sum, mask));
            mask >>= 1;
        }
        let d = amax / 127.0;
        let q = if amax == 0.0 {
            0i32
        } else {
            let t = xi / d;
            let half = f32::from_bits(0x3F00_0000 | (t.to_bits() & 0x8000_0000));
            f2i_rz(float::add_rz_f32(t, half))
        };
        let y = vy.offset(ib as isize * 36);
        *y.offset(4 + iqs as isize) = q as u8;
        if iqs > 0 {
            return;
        }
        *(y as *mut u16) = f2h(d);
        *(y.add(2) as *mut u16) = f2h(sum);
    }

    #[kernel]
    pub unsafe fn mmvq_gguf_quantize_q8_1_bf16(x: *const u16, vy: *mut u8, kx: i32, kx_padded: i32) {
        quantize(x, vy, kx, kx_padded, |p, i| f32::from_bits((*p.offset(i) as u32) << 16))
    }
    #[kernel]
    pub unsafe fn mmvq_gguf_quantize_q8_1_f16(x: *const u16, vy: *mut u8, kx: i32, kx_padded: i32) {
        quantize(x, vy, kx, kx_padded, |p, i| h2f(*p.offset(i) as u32))
    }
    #[kernel]
    pub unsafe fn mmvq_gguf_quantize_q8_1_f32(x: *const f32, vy: *mut u8, kx: i32, kx_padded: i32) {
        quantize(x, vy, kx, kx_padded, |p, i| *p.offset(i))
    }

    // GENERATED KERNELS BEGIN
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 1, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 2, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 3, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 4, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 5, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 6, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 7, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_bf16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 8, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 1, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 2, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 3, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 4, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 5, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 6, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 7, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 8, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 1, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 2, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 3, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 4, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 5, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 6, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 7, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_0_f32_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<0, 8, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 1, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 2, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 3, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 4, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 5, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 6, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 7, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_bf16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 8, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 1, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 2, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 3, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 4, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 5, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 6, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 7, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 8, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 1, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 2, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 3, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 4, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 5, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 6, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 7, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_1_f32_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<1, 8, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 1, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 2, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 3, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 4, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 5, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 6, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 7, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_bf16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 8, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 1, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 2, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 3, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 4, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 5, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 6, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 7, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 8, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 1, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 2, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 3, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 4, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 5, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 6, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 7, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_0_f32_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<2, 8, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 1, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 2, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 3, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 4, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 5, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 6, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 7, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_bf16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 8, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 1, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 2, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 3, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 4, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 5, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 6, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 7, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 8, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 1, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 2, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 3, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 4, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 5, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 6, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 7, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_1_f32_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<3, 8, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 1, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 2, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 3, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 4, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 5, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 6, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 7, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_bf16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 8, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 1, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 2, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 3, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 4, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 5, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 6, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 7, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 8, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 1, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 2, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 3, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 4, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 5, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 6, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 7, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q8_0_f32_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<4, 8, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 1, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 2, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 3, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 4, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 5, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 6, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 7, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_bf16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 8, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 1, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 2, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 3, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 4, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 5, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 6, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 7, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 8, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 1, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 2, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 3, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 4, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 5, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 6, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 7, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q2_k_f32_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<5, 8, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 1, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 2, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 3, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 4, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 5, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 6, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 7, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_bf16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 8, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 1, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 2, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 3, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 4, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 5, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 6, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 7, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 8, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 1, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 2, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 3, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 4, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 5, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 6, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 7, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q3_k_f32_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<6, 8, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 1, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 2, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 3, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 4, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 5, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 6, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 7, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_bf16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 8, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 1, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 2, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 3, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 4, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 5, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 6, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 7, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 8, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 1, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 2, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 3, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 4, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 5, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 6, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 7, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q4_k_f32_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<7, 8, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 1, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 2, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 3, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 4, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 5, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 6, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 7, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_bf16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 8, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 1, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 2, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 3, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 4, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 5, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 6, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 7, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 8, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 1, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 2, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 3, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 4, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 5, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 6, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 7, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q5_k_f32_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<8, 8, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 1, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 2, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 3, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 4, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 5, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 6, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 7, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_bf16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 8, B>(vx, vy, dst as *mut B, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 1, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 2, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 3, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 4, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 5, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 6, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 7, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f16_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut u16, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 8, H>(vx, vy, dst as *mut H, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_plain_cuda1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 1, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_plain_cuda2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 2, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_plain_cuda3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 3, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_plain_cuda4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 4, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_plain_cuda5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 5, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_plain_cuda6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 6, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_plain_cuda7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 7, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    #[kernel] pub unsafe fn mmvq_gguf_q6_k_f32_plain_cuda8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32) { mmvq::<9, 8, f32>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst) }
    // GENERATED KERNELS END
}

fn main() {
    std::process::exit(if gate::run() { 0 } else { 1 });
}
