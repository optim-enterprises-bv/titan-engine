//! candle-kernels `quantized.cu` in cuda-oxide: same 129 entry names, same raw-pointer / by-value
//! `int` ABI, bit-identical output (checked by the host `main` against candle's own nvcc PTX).
//!
//! candle-kernels is built with plain `-O3` (no fast-math): IEEE ops, no `.ftz`. The reference PTX
//! uses unrounded `mul.f32` / `add.f32`, which ptxas contracts into FFMA where it sees fit, so the
//! float program here is read off the PTX/SASS and every fused multiply-add is written out
//! explicitly (`fma_rn_f32`); every separate FMUL/FADD is a plain `.rn` op (Rust never contracts).
//!
//! Integer work (nibble unpacking, dp4a dot products, `__vsubss4`) is exact, so only its value has
//! to match, not the instruction sequence.
//!
//! Families: dequantize_block_* (22), dequantize_mul_mat_vec_* (10), quantize_q8_1 (1),
//! mul_mat_vec_*_q8_1_cuda1..8 (80), indexed_moe_forward_* (6), mul_mat_q* (10).
mod gate;

use cuda_device::{SharedArray, convert, device, dotprod, float, kernel, ptx_asm, thread, warp};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    // ------------------------------------------------------------------------------------------
    // Scalar helpers.

    #[inline(always)]
    pub fn h2f(bits: u16) -> f32 {
        convert::cvt_f32_f16x2_lo(bits as u32)
    }

    #[inline(always)]
    pub fn f2h(x: f32) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("cvt.rn.f16.f32 %0, %1;", out("=h") r, in("f") x, options(register_only)); }
        r
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
    pub fn sub(a: f32, b: f32) -> f32 {
        float::add_rn_f32(a, -b)
    }

    #[inline(always)]
    pub unsafe fn ld8(p: *const u8, off: usize) -> u32 {
        *p.add(off) as u32
    }
    #[inline(always)]
    pub unsafe fn ld8s(p: *const u8, off: usize) -> i32 {
        *p.add(off) as i8 as i32
    }
    #[inline(always)]
    pub unsafe fn ld16(p: *const u8, off: usize) -> u32 {
        *(p.add(off) as *const u16) as u32
    }
    /// f16 at byte offset -> f32.
    #[inline(always)]
    pub unsafe fn ldh(p: *const u8, off: usize) -> f32 {
        h2f(*(p.add(off) as *const u16))
    }
    /// 32-bit word assembled from two 16-bit loads (`get_int_from_uint8`); same value as aligned.
    #[inline(always)]
    pub unsafe fn ld32(p: *const u8, off: usize) -> u32 {
        ld16(p, off) | (ld16(p, off + 2) << 16)
    }

    /// Read-only global loads as `ld.global.nc` (the reference's `const __restrict__` loads). Only
    /// for kernel inputs the kernel never writes; register_only lets LLVM schedule them freely.
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
    /// `st.shared.b32` at a 32-bit shared-window address (the tile stores of `load_tiles`, whose
    /// pointer argument would otherwise be generic across the #[device] call boundary).
    #[inline(always)]
    pub unsafe fn sts32(addr: u32, v: u32) {
        ptx_asm!("st.shared.b32 [%0], %1;", in("r") addr, in("r") v, clobber("memory"));
    }

    /// Destination element type of the dequantize kernels.
    pub trait Dst: Copy {
        unsafe fn st(p: *mut Self, i: usize, v: f32);
    }
    impl Dst for f32 {
        #[inline(always)]
        unsafe fn st(p: *mut f32, i: usize, v: f32) {
            *p.add(i) = v;
        }
    }
    impl Dst for u16 {
        #[inline(always)]
        unsafe fn st(p: *mut u16, i: usize, v: f32) {
            *p.add(i) = f2h(v);
        }
    }

    // ------------------------------------------------------------------------------------------
    // Family 1: dequantize_block_* (22 entries).

    #[inline(always)]
    unsafe fn deq_q4_0<T: Dst>(vx: *const u8, yy: *mut T, nb32: i32) {
        let i = thread::blockIdx_x() as i64;
        let tid = thread::threadIdx_x() as i64;
        let il = tid / 8;
        let ir = tid % 8;
        let ib = 8 * i + ir;
        if ib >= nb32 as i64 {
            return;
        }
        let y = yy.add((256 * i + 32 * ir + 4 * il) as usize);
        let x = vx.add(ib as usize * 18);
        let d = ldh(x, 0);
        let dm = mul(d, -8.0);
        let q = x.add(2 + 4 * il as usize);
        let mut l = 0;
        while l < 4 {
            let b = ld8(q, l);
            T::st(y, l, fma(d, (b & 0xF) as f32, dm));
            T::st(y, l + 16, fma(d, (b >> 4) as f32, dm));
            l += 1;
        }
    }

    #[inline(always)]
    unsafe fn deq_q4_1<T: Dst>(vx: *const u8, yy: *mut T, nb32: i32) {
        let i = thread::blockIdx_x() as i64;
        let tid = thread::threadIdx_x() as i64;
        let il = tid / 8;
        let ir = tid % 8;
        let ib = 8 * i + ir;
        if ib >= nb32 as i64 {
            return;
        }
        let y = yy.add((256 * i + 32 * ir + 4 * il) as usize);
        let x = vx.add(ib as usize * 20);
        let dx = ldh(x, 0);
        let dy = ldh(x, 2);
        let q = x.add(4 + 4 * il as usize);
        let mut l = 0;
        while l < 4 {
            let b = ld8(q, l);
            T::st(y, l, fma(dx, (b & 0xF) as f32, dy));
            T::st(y, l + 16, fma(dx, (b >> 4) as f32, dy));
            l += 1;
        }
    }

    /// `dequantize_block<32, 2, dequantize_q5_{0,1}>`: two values per thread.
    #[inline(always)]
    unsafe fn deq_q5<T: Dst, const ONE: bool>(vx: *const u8, y: *mut T, k: i32) {
        let i = (2u32.wrapping_mul(thread::blockDim_x().wrapping_mul(thread::blockIdx_x()).wrapping_add(thread::threadIdx_x()))) as i32;
        if i >= k {
            return;
        }
        let ib = (i / 32) as usize;
        let iqs = ((i % 32) / 2) as usize;
        let iybs = (i - i % 32) as usize;
        let (bs, qh_off, qs_off) = if ONE { (24, 4, 8) } else { (22, 2, 6) };
        let x = vx.add(ib * bs);
        let qh = ld16(x, qh_off) | (ld16(x, qh_off + 2) << 16);
        let xh_0 = ((qh >> iqs) << 4) & 0x10;
        let xh_1 = (qh >> (iqs + 12)) & 0x10;
        let b = ld8(x, qs_off + iqs);
        let v0 = ((b & 0xf) | xh_0) as i32;
        let v1 = ((b >> 4) | xh_1) as i32;
        let (r0, r1) = if ONE {
            let d = ldh(x, 0);
            let m = ldh(x, 2);
            (fma(v0 as f32, d, m), fma(v1 as f32, d, m))
        } else {
            let d = ldh(x, 0);
            (mul((v0 - 16) as f32, d), mul((v1 - 16) as f32, d))
        };
        T::st(y, iybs + iqs, r0);
        T::st(y, iybs + iqs + 16, r1);
    }

    #[inline(always)]
    unsafe fn deq_q8_0<T: Dst>(vx: *const u8, yy: *mut T, nb32: i32) {
        let i = thread::blockIdx_x() as i32;
        let tid = thread::threadIdx_x() as i32;
        let il = tid / 8;
        let ir = tid % 8;
        let ib = 8 * i + ir;
        if ib >= nb32 {
            return;
        }
        let y = yy.add((256 * i + 32 * ir + 8 * il) as usize);
        let x = vx.add(ib as usize * 34);
        let d = ldh(x, 0);
        let q = x.add(2 + 8 * il as usize);
        let mut l = 0;
        while l < 8 {
            T::st(y, l, mul(d, ld8s(q, l) as f32));
            l += 1;
        }
    }

    #[inline(always)]
    unsafe fn deq_q2_k<T: Dst>(vx: *const u8, yy: *mut T) {
        let i = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let x = vx.add(i * 84);
        let n = tid / 32;
        let l = tid - 32 * n;
        let is = 8 * n + l / 16;
        let q = ld8(x, 16 + 32 * n + l);
        let y = yy.add(i * 256 + 128 * n);
        let dall = ldh(x, 80);
        let dmin = ldh(x, 82);
        let mut k = 0;
        while k < 4 {
            let sc = ld8(x, is + 2 * k);
            let a = mul(dall, (sc & 0xF) as f32);
            let m = mul(dmin, (sc >> 4) as f32);
            T::st(y, l + 32 * k, fma(a, ((q >> (2 * k)) & 3) as f32, -m));
            k += 1;
        }
    }

    #[inline(always)]
    unsafe fn deq_q3_k<T: Dst>(vx: *const u8, yy: *mut T) {
        let i = thread::blockIdx_x() as usize;
        let t = thread::threadIdx_x() as usize;
        let x = vx.add(i * 110);
        let r = t / 4;
        let tid = r / 2;
        let is0 = r % 2;
        let l0 = 16 * is0 + 4 * (t % 4);
        let n = tid / 4;
        let j = tid - 4 * n;
        let m = (1u32 << (4 * n + j)) & 0xFF;
        let is = 8 * n + 2 * j + is0;
        let shift = 2 * j;
        let sc = |o: usize| ld8(x, 96 + o);
        let us = if is < 4 {
            (sc(is) & 0xF) | (((sc(is + 8) >> 0) & 3) << 4)
        } else if is < 8 {
            (sc(is) & 0xF) | (((sc(is + 4) >> 2) & 3) << 4)
        } else if is < 12 {
            (sc(is - 8) >> 4) | (((sc(is) >> 4) & 3) << 4)
        } else {
            (sc(is - 8) >> 4) | (((sc(is - 4) >> 6) & 3) << 4)
        };
        let us = us as u8 as i8 as i32;
        let d_all = ldh(x, 108);
        let dl = mul(d_all, (us - 32) as f32);
        let y = yy.add(i * 256 + 128 * n + 32 * j);
        let mut l = l0;
        while l < l0 + 4 {
            let q = ((ld8(x, 32 + 32 * n + l) >> shift) & 3) as i32;
            let h = if ld8(x, l) & m != 0 { 0 } else { 4 };
            T::st(y, l, mul(dl, (q - h) as f32));
            l += 1;
        }
    }

    /// `get_scale_min_k4` on the 12 scale bytes at `s`.
    #[inline(always)]
    pub unsafe fn scale_min_k4(j: usize, s: *const u8) -> (u32, u32) {
        if j < 4 {
            (ld8(s, j) & 63, ld8(s, j + 4) & 63)
        } else {
            ((ld8(s, j + 4) & 0xF) | ((ld8(s, j - 4) >> 6) << 4), (ld8(s, j + 4) >> 4) | ((ld8(s, j) >> 6) << 4))
        }
    }

    #[inline(always)]
    unsafe fn deq_q4_k<T: Dst>(vx: *const u8, yy: *mut T) {
        let i = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let x = vx.add(i * 144);
        let il = tid / 8;
        let ir = tid % 8;
        let is = 2 * il;
        let y = yy.add(i * 256 + 64 * il + 4 * ir);
        let dall = ldh(x, 0);
        let dmin = ldh(x, 2);
        let q = x.add(16 + 32 * il + 4 * ir);
        let (sc, m) = scale_min_k4(is, x.add(4));
        let d1 = mul(dall, sc as f32);
        let m1 = mul(dmin, m as f32);
        let (sc, m) = scale_min_k4(is + 1, x.add(4));
        let d2 = mul(dall, sc as f32);
        let m2 = mul(dmin, m as f32);
        let mut l = 0;
        while l < 4 {
            let b = ld8(q, l);
            T::st(y, l, fma(d1, (b & 0xF) as f32, -m1));
            T::st(y, l + 32, fma(d2, (b >> 4) as f32, -m2));
            l += 1;
        }
    }

    #[inline(always)]
    unsafe fn deq_q5_k<T: Dst>(vx: *const u8, yy: *mut T) {
        let i = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let x = vx.add(i * 176);
        let il = tid / 16;
        let ir = tid % 16;
        let is = 2 * il;
        let y = yy.add(i * 256 + 64 * il + 2 * ir);
        let dall = ldh(x, 0);
        let dmin = ldh(x, 2);
        let ql = x.add(48 + 32 * il + 2 * ir);
        let qh = x.add(16 + 2 * ir);
        let (sc, m) = scale_min_k4(is, x.add(4));
        let d1 = mul(dall, sc as f32);
        let m1 = mul(dmin, m as f32);
        let (sc, m) = scale_min_k4(is + 1, x.add(4));
        let d2 = mul(dall, sc as f32);
        let m2 = mul(dmin, m as f32);
        let hm = 1u32 << (2 * il);
        let hb = |k: usize, hm: u32| if ld8(qh, k) & hm != 0 { 16 } else { 0 };
        T::st(y, 0, fma(d1, ((ld8(ql, 0) & 0xF) + hb(0, hm)) as f32, -m1));
        T::st(y, 1, fma(d1, ((ld8(ql, 1) & 0xF) + hb(1, hm)) as f32, -m1));
        let hm = hm << 1;
        T::st(y, 32, fma(d2, ((ld8(ql, 0) >> 4) + hb(0, hm)) as f32, -m2));
        T::st(y, 33, fma(d2, ((ld8(ql, 1) >> 4) + hb(1, hm)) as f32, -m2));
    }

    #[inline(always)]
    unsafe fn deq_q6_k<T: Dst>(vx: *const u8, yy: *mut T) {
        let i = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let x = vx.add(i * 210);
        let ip = tid / 32;
        let il = tid - 32 * ip;
        let is = 8 * ip + il / 16;
        let y = yy.add(i * 256 + 128 * ip + il);
        let d = ldh(x, 208);
        let ql = x.add(64 * ip + il);
        let qh = ld8(x, 128 + 32 * ip + il);
        let sc = x.add(192 + is);
        let v = |lo: u32, sh: u32| (((lo | (((qh >> sh) & 3) << 4)) as u8 as i8) as i32 - 32) as f32;
        T::st(y, 0, mul(mul(d, ld8s(sc, 0) as f32), v(ld8(ql, 0) & 0xF, 0)));
        T::st(y, 32, mul(mul(d, ld8s(sc, 2) as f32), v(ld8(ql, 32) & 0xF, 2)));
        T::st(y, 64, mul(mul(d, ld8s(sc, 4) as f32), v(ld8(ql, 0) >> 4, 4)));
        T::st(y, 96, mul(mul(d, ld8s(sc, 6) as f32), v(ld8(ql, 32) >> 4, 6)));
    }

    #[inline(always)]
    unsafe fn deq_q8_k<T: Dst>(vx: *const u8, yy: *mut T) {
        let i = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let x = vx.add(i * 292);
        let il = tid / 8;
        let ir = tid % 8;
        let y = yy.add(i * 256 + 64 * il + 8 * ir);
        let q = x.add(4 + 64 * il + 8 * ir);
        let d = *(x as *const f32);
        let mut l = 0;
        while l < 8 {
            T::st(y, l, mul(ld8s(q, l) as f32, d));
            l += 1;
        }
    }

    #[kernel] pub unsafe fn dequantize_block_q2_K_f32(vx: *const u8, y: *mut f32) { deq_q2_k(vx, y) }
    #[kernel] pub unsafe fn dequantize_block_q2_K_f16(vx: *const u8, y: *mut u16) { deq_q2_k(vx, y) }
    #[kernel] pub unsafe fn dequantize_block_q3_K_f32(vx: *const u8, y: *mut f32) { deq_q3_k(vx, y) }
    #[kernel] pub unsafe fn dequantize_block_q3_K_f16(vx: *const u8, y: *mut u16) { deq_q3_k(vx, y) }
    #[kernel] pub unsafe fn dequantize_block_q4_K_f32(vx: *const u8, y: *mut f32) { deq_q4_k(vx, y) }
    #[kernel] pub unsafe fn dequantize_block_q4_K_f16(vx: *const u8, y: *mut u16) { deq_q4_k(vx, y) }
    #[kernel] pub unsafe fn dequantize_block_q5_K_f32(vx: *const u8, y: *mut f32) { deq_q5_k(vx, y) }
    #[kernel] pub unsafe fn dequantize_block_q5_K_f16(vx: *const u8, y: *mut u16) { deq_q5_k(vx, y) }
    #[kernel] pub unsafe fn dequantize_block_q6_K_f32(vx: *const u8, y: *mut f32) { deq_q6_k(vx, y) }
    #[kernel] pub unsafe fn dequantize_block_q6_K_f16(vx: *const u8, y: *mut u16) { deq_q6_k(vx, y) }
    #[kernel] pub unsafe fn dequantize_block_q8_K_f32(vx: *const u8, y: *mut f32) { deq_q8_k(vx, y) }
    #[kernel] pub unsafe fn dequantize_block_q8_K_f16(vx: *const u8, y: *mut u16) { deq_q8_k(vx, y) }
    #[kernel] pub unsafe fn dequantize_block_q4_0_f32(vx: *const u8, y: *mut f32, k: i32) { deq_q4_0(vx, y, k) }
    #[kernel] pub unsafe fn dequantize_block_q4_0_f16(vx: *const u8, y: *mut u16, k: i32) { deq_q4_0(vx, y, k) }
    #[kernel] pub unsafe fn dequantize_block_q4_1_f32(vx: *const u8, y: *mut f32, k: i32) { deq_q4_1(vx, y, k) }
    #[kernel] pub unsafe fn dequantize_block_q4_1_f16(vx: *const u8, y: *mut u16, k: i32) { deq_q4_1(vx, y, k) }
    #[kernel] pub unsafe fn dequantize_block_q5_0_f32(vx: *const u8, y: *mut f32, k: i32) { deq_q5::<f32, false>(vx, y, k) }
    #[kernel] pub unsafe fn dequantize_block_q5_0_f16(vx: *const u8, y: *mut u16, k: i32) { deq_q5::<u16, false>(vx, y, k) }
    #[kernel] pub unsafe fn dequantize_block_q5_1_f32(vx: *const u8, y: *mut f32, k: i32) { deq_q5::<f32, true>(vx, y, k) }
    #[kernel] pub unsafe fn dequantize_block_q5_1_f16(vx: *const u8, y: *mut u16, k: i32) { deq_q5::<u16, true>(vx, y, k) }
    #[kernel] pub unsafe fn dequantize_block_q8_0_f32(vx: *const u8, y: *mut f32, k: i32) { deq_q8_0(vx, y, k) }
    #[kernel] pub unsafe fn dequantize_block_q8_0_f16(vx: *const u8, y: *mut u16, k: i32) { deq_q8_0(vx, y, k) }

    // ------------------------------------------------------------------------------------------
    // Family 2: dequantize_mul_mat_vec_* (10 entries).
    //
    // A C sum of products `p0*q0 + p1*q1 + p2*q2 + ...` is contracted by NVVM into a chain of
    // fma. Which of the first two products becomes the fma and which the plain mul is LLVM's
    // choice and varies per expression (read off the PTX): `chain_a` fuses the first product,
    // `chain_b` the second.

    #[inline(always)]
    pub fn chain_a<const N: usize>(p: [f32; N], q: [f32; N]) -> f32 {
        let mut r = fma(p[0], q[0], mul(p[1], q[1]));
        let mut k = 2;
        while k < N {
            r = fma(p[k], q[k], r);
            k += 1;
        }
        r
    }
    #[inline(always)]
    pub fn chain_b<const N: usize>(p: [f32; N], q: [f32; N]) -> f32 {
        let mut r = fma(p[1], q[1], mul(p[0], q[0]));
        let mut k = 2;
        while k < N {
            r = fma(p[k], q[k], r);
            k += 1;
        }
        r
    }

    /// Butterfly `warp_reduce_sum`.
    #[inline(always)]
    pub fn warp_sum(mut x: f32) -> f32 {
        let mut mask = 16;
        while mask > 0 {
            x = add(x, warp::shuffle_xor_f32_sync(0xffff_ffff, x, mask));
            mask >>= 1;
        }
        x
    }

    /// `dequantize_q4_0/q4_1/q5_0/q5_1/q8_0` (the dfloat2 pair for quant index `iqs` of block `ib`).
    /// FMT: 0 = q4_0, 1 = q4_1, 2 = q5_0, 3 = q5_1, 4 = q8_0.
    #[inline(always)]
    pub unsafe fn dequant_pair<const FMT: u32>(vx: *const u8, ib: i32, iqs: i32) -> (f32, f32) {
        let iqs = iqs as usize;
        match FMT {
            0 => {
                let x = vx.add(ib as usize * 18);
                let d = ldh(x, 0);
                let b = ld8(x, 2 + iqs) as i32;
                (mul(((b & 0xF) - 8) as f32, d), mul(((b >> 4) - 8) as f32, d))
            }
            1 => {
                let x = vx.add(ib as usize * 20);
                let (d, m) = (ldh(x, 0), ldh(x, 2));
                let b = ld8(x, 4 + iqs);
                (fma((b & 0xF) as f32, d, m), fma((b >> 4) as f32, d, m))
            }
            2 | 3 => {
                let one = FMT == 3;
                let (bs, qh_off, qs_off) = if one { (24, 4, 8) } else { (22, 2, 6) };
                let x = vx.add(ib as usize * bs);
                let qh = ld16(x, qh_off) | (ld16(x, qh_off + 2) << 16);
                let xh_0 = ((qh >> iqs) << 4) & 0x10;
                let xh_1 = (qh >> (iqs + 12)) & 0x10;
                let b = ld8(x, qs_off + iqs);
                let v0 = ((b & 0xf) | xh_0) as i32;
                let v1 = ((b >> 4) | xh_1) as i32;
                let d = ldh(x, 0);
                if one {
                    let m = ldh(x, 2);
                    (fma(v0 as f32, d, m), fma(v1 as f32, d, m))
                } else {
                    (mul((v0 - 16) as f32, d), mul((v1 - 16) as f32, d))
                }
            }
            _ => {
                let x = vx.add(ib as usize * 34);
                let d = ldh(x, 0);
                (mul(ld8s(x, 2 + iqs) as f32, d), mul(ld8s(x, 3 + iqs) as f32, d))
            }
        }
    }

    /// Template `dequantize_mul_mat_vec<qk, qr, dequantize_kernel>` (GGML_CUDA_DMMV_X = 32).
    #[inline(always)]
    unsafe fn dmmv<const FMT: u32>(vx: *const u8, y: *const f32, dst: *mut f32, ncols: i32, nrows: i32) {
        let row = thread::blockIdx_x().wrapping_mul(thread::blockDim_y()).wrapping_add(thread::threadIdx_y()) as i32;
        if row >= nrows {
            return;
        }
        let tid = thread::threadIdx_x() as i32;
        let (qk, qr) = if FMT == 4 { (32, 1) } else { (32, 2) };
        let y_offset = if qr == 1 { 1 } else { qk / 2 };
        let mut tmp = 0f32;
        let mut i = 0i32;
        while i < ncols {
            let col = i + 2 * tid;
            let ib = (row.wrapping_mul(ncols) + col) / qk;
            let iqs = (col % qk) / qr;
            let iybs = col - col % qk;
            let (vx0, vx1) = dequant_pair::<FMT>(vx, ib, iqs);
            tmp = fma(vx0, *y.add((iybs + iqs) as usize), tmp);
            tmp = fma(vx1, *y.add((iybs + iqs + y_offset) as usize), tmp);
            i += 64;
        }
        tmp = warp_sum(tmp);
        if tid == 0 {
            *dst.add(row as usize) = tmp;
        }
    }

    #[kernel] pub unsafe fn dequantize_mul_mat_vec_q4_0_cuda(vx: *const u8, y: *const f32, dst: *mut f32, ncols: i32, nrows: i32) { dmmv::<0>(vx, y, dst, ncols, nrows) }
    #[kernel] pub unsafe fn dequantize_mul_mat_vec_q4_1_cuda(vx: *const u8, y: *const f32, dst: *mut f32, ncols: i32, nrows: i32) { dmmv::<1>(vx, y, dst, ncols, nrows) }
    #[kernel] pub unsafe fn dequantize_mul_mat_vec_q5_0_cuda(vx: *const u8, y: *const f32, dst: *mut f32, ncols: i32, nrows: i32) { dmmv::<2>(vx, y, dst, ncols, nrows) }
    #[kernel] pub unsafe fn dequantize_mul_mat_vec_q5_1_cuda(vx: *const u8, y: *const f32, dst: *mut f32, ncols: i32, nrows: i32) { dmmv::<3>(vx, y, dst, ncols, nrows) }
    #[kernel] pub unsafe fn dequantize_mul_mat_vec_q8_0_cuda(vx: *const u8, y: *const f32, dst: *mut f32, ncols: i32, nrows: i32) { dmmv::<4>(vx, y, dst, ncols, nrows) }

    #[kernel]
    pub unsafe fn dequantize_mul_mat_vec_q2_k(vx: *const u8, yy: *const f32, dst: *mut f32, ncols: i32, nrows: i32) {
        let row = thread::blockIdx_x().wrapping_mul(thread::blockDim_y()).wrapping_add(thread::threadIdx_y()) as i32;
        if row > nrows {
            return;
        }
        let nb = ncols / 256;
        let x0 = vx.offset(row.wrapping_mul(nb) as isize * 84);
        let tx = thread::threadIdx_x() as i32;
        let tid = tx / 2;
        let ix = tx % 2;
        let im = tid / 8;
        let inn = tid - 8 * im;
        let l0 = 2 * inn;
        let q_offset = (32 * im + l0) as usize;
        let s_offset = (8 * im) as usize;
        let y_offset = (128 * im + l0) as usize;
        let mut tmp = 0f32;
        let mut i = ix;
        while i < nb {
            let x = x0.add(i as usize * 84);
            let y = yy.add(i as usize * 256 + y_offset);
            let q = x.add(16 + q_offset);
            let dall = ldh(x, 80);
            let dmin = ldh(x, 82);
            let sc = |k: usize| ld8(x, s_offset + k);
            let d = |k: usize| (sc(k) & 0xF) as f32;
            let m = |k: usize| (sc(k) >> 4) as f32;
            let mut sum1 = 0f32;
            let mut sum2 = 0f32;
            let mut l = 0;
            while l < 2 {
                let yv = |o: usize| *y.add(l + o);
                let qa = ld8(q, l);
                let qb = ld8(q, l + 16);
                let s = chain_a(
                    [mul(yv(0), d(0)), mul(yv(32), d(2)), mul(yv(64), d(4)), mul(yv(96), d(6)),
                     mul(yv(16), d(1)), mul(yv(48), d(3)), mul(yv(80), d(5)), mul(yv(112), d(7))],
                    [(qa & 3) as f32, ((qa >> 2) & 3) as f32, ((qa >> 4) & 3) as f32, (qa >> 6) as f32,
                     (qb & 3) as f32, ((qb >> 2) & 3) as f32, ((qb >> 4) & 3) as f32, (qb >> 6) as f32],
                );
                sum1 = add(sum1, s);
                let t = chain_b(
                    [yv(32), yv(0), yv(64), yv(96), yv(16), yv(48), yv(80), yv(112)],
                    [m(2), m(0), m(4), m(6), m(1), m(3), m(5), m(7)],
                );
                sum2 = add(sum2, t);
                l += 1;
            }
            tmp = add(tmp, fma(dall, sum1, -mul(dmin, sum2)));
            i += 2;
        }
        tmp = warp_sum(tmp);
        if tx == 0 {
            *dst.offset(row as isize) = tmp;
        }
    }

    #[kernel]
    pub unsafe fn dequantize_mul_mat_vec_q3_k(vx: *const u8, yy: *const f32, dst: *mut f32, ncols: i32, nrows: i32) {
        let row = thread::blockIdx_x().wrapping_mul(thread::blockDim_y()).wrapping_add(thread::threadIdx_y()) as i32;
        if row > nrows {
            return;
        }
        let nb = ncols / 256;
        let x0 = vx.offset(row.wrapping_mul(nb) as isize * 110);
        let tx = thread::threadIdx_x() as i32;
        let tid = tx / 2;
        let ix = tx % 2;
        let im = tid / 8;
        let inn = tid - 8 * im;
        let m = 1u32 << (4 * im);
        let l0 = 2 * inn;
        let q_offset = (32 * im + l0) as usize;
        let y_offset = (128 * im + l0) as usize;
        let s_shift = (4 * im) as u32;
        let mut tmp = 0f32;
        let mut i = ix;
        while i < nb {
            let x = x0.add(i as usize * 110);
            let y = yy.add(i as usize * 256 + y_offset);
            let q = x.add(32 + q_offset);
            let h = x.add(l0 as usize);
            let a = |k: usize| ld16(x, 96 + 2 * k);
            let ut = [
                ((a(0) >> s_shift) & 0x0f0f) | (((a(4) >> s_shift) & 0x0303) << 4),
                ((a(1) >> s_shift) & 0x0f0f) | (((a(5) >> s_shift) & 0x0303) << 4),
                ((a(2) >> s_shift) & 0x0f0f) | (((a(4) >> (s_shift + 2)) & 0x0303) << 4),
                ((a(3) >> s_shift) & 0x0f0f) | (((a(5) >> (s_shift + 2)) & 0x0303) << 4),
            ];
            let s = |k: usize| ((((ut[k / 2] >> (8 * (k % 2))) & 0xFF) as u8 as i8 as i32) - 32) as f32;
            let d = ldh(x, 108);
            let mut sum = 0f32;
            let mut l = 0;
            while l < 2 {
                let yv = |o: usize| *y.add(l + o);
                let qv = |j: usize, sh: u32, mm: u32| {
                    let qq = ((ld8(q, j) >> sh) & 3) as i32;
                    let hh = if ld8(h, j) & mm != 0 { 0 } else { 4 };
                    (qq - hh) as f32
                };
                let c1 = chain_a(
                    [mul(yv(0), s(0)), mul(yv(32), s(2)), mul(yv(64), s(4)), mul(yv(96), s(6))],
                    [qv(l, 0, m), qv(l, 2, m << 1), qv(l, 4, m << 2), qv(l, 6, m << 3)],
                );
                sum = add(sum, c1);
                let c2 = chain_a(
                    [mul(yv(16), s(1)), mul(yv(48), s(3)), mul(yv(80), s(5)), mul(yv(112), s(7))],
                    [qv(l + 16, 0, m), qv(l + 16, 2, m << 1), qv(l + 16, 4, m << 2), qv(l + 16, 6, m << 3)],
                );
                sum = add(sum, c2);
                l += 1;
            }
            tmp = fma(d, sum, tmp);
            i += 2;
        }
        tmp = warp_sum(tmp);
        if tx == 0 {
            *dst.offset(row as isize) = tmp;
        }
    }

    /// The q4_K / q5_K `aux` scale words for half `im`: bytes sc[0..8].
    #[inline(always)]
    pub unsafe fn k4_aux(sc: *const u8, im: usize) -> [u32; 4] {
        let a = |k: usize| ld16(sc, 2 * k);
        [
            a(im) & 0x3f3f,
            a(im + 2) & 0x3f3f,
            ((a(im + 4) >> 0) & 0x0f0f) | ((a(im) & 0xc0c0) >> 2),
            ((a(im + 4) >> 4) & 0x0f0f) | ((a(im + 2) & 0xc0c0) >> 2),
        ]
    }
    #[inline(always)]
    pub fn aux_byte(aux: &[u32; 4], k: usize) -> u32 {
        (aux[k / 2] >> (8 * (k % 2))) & 0xFF
    }

    #[kernel]
    pub unsafe fn dequantize_mul_mat_vec_q4_k(vx: *const u8, yy: *const f32, dst: *mut f32, ncols: i32, nrows: i32) {
        let row = thread::blockIdx_x().wrapping_mul(thread::blockDim_y()).wrapping_add(thread::threadIdx_y()) as i32;
        if row > nrows {
            return;
        }
        let nb = ncols / 256;
        let x0 = vx.offset(row.wrapping_mul(nb) as isize * 144);
        let tx = thread::threadIdx_x() as i32;
        let tid = tx / 2;
        let ix = tx % 2;
        let il = tid / 4;
        let ir = tid - 4 * il;
        let im = il / 2;
        let inn = il % 2;
        let l0 = 4 * (2 * ir + inn);
        let q_offset = (32 * im + l0) as usize;
        let y_offset = (64 * im + l0) as usize;
        let mut tmp = 0f32;
        let mut i = ix;
        while i < nb {
            let x = x0.add(i as usize * 144);
            let y1 = yy.add(i as usize * 256 + y_offset);
            let y2 = y1.add(128);
            let dall = ldh(x, 0);
            let dmin = ldh(x, 2);
            let aux = k4_aux(x.add(4), im as usize);
            let sc = |k: usize| aux_byte(&aux, k) as f32;
            let q1 = *(x.add(16 + q_offset) as *const u32);
            let q2 = *(x.add(16 + q_offset + 64) as *const u32);
            let q32 = [q1 & 0x0f0f0f0f, q1 & 0xf0f0f0f0, q2 & 0x0f0f0f0f, q2 & 0xf0f0f0f0];
            let q4 = |k: usize| ((q32[k / 4] >> (8 * (k % 4))) & 0xFF) as f32;
            let (mut sx, mut sy, mut sz, mut sw) = (0f32, 0f32, 0f32, 0f32);
            let mut smin = 0f32;
            let mut l = 0;
            while l < 4 {
                sx = fma(*y1.add(l), q4(l), sx);
                sy = fma(*y1.add(l + 32), q4(l + 4), sy);
                sz = fma(*y2.add(l), q4(l + 8), sz);
                sw = fma(*y2.add(l + 32), q4(l + 12), sw);
                smin = add(smin, chain_b([*y1.add(l + 32), *y1.add(l), *y2.add(l), *y2.add(l + 32)], [sc(3), sc(2), sc(6), sc(7)]));
                l += 1;
            }
            let inner = fma(mul(sw, sc(5)), 0.0625, fma(sz, sc(4), fma(sx, sc(0), mul(mul(sy, sc(1)), 0.0625))));
            tmp = add(tmp, fma(dall, inner, -mul(dmin, smin)));
            i += 2;
        }
        tmp = warp_sum(tmp);
        if tid == 0 {
            *dst.offset(row as isize) = tmp;
        }
    }

    #[kernel]
    pub unsafe fn dequantize_mul_mat_vec_q5_k(vx: *const u8, yy: *const f32, dst: *mut f32, ncols: i32) {
        let row = thread::blockIdx_x() as i32;
        let nb = ncols / 256;
        let x0 = vx.offset(row.wrapping_mul(nb) as isize * 176);
        let tx = thread::threadIdx_x() as i32;
        let tid = tx / 2;
        let ix = tx % 2;
        let il = tid / 4;
        let ir = tid - 4 * il;
        let im = il / 2;
        let inn = il % 2;
        let l0 = 2 * (2 * ir + inn);
        let q_offset = (32 * im + l0) as usize;
        let y_offset = (64 * im + l0) as usize;
        let hm1 = (1u32 << (2 * im)) & 0xFF;
        let hm2 = (hm1 << 4) & 0xFF;
        let mut tmp = 0f32;
        let mut i = ix;
        while i < nb {
            let x = x0.add(i as usize * 176);
            let ql1 = x.add(48 + q_offset);
            let qh = x.add(16 + l0 as usize);
            let y1 = yy.add(i as usize * 256 + y_offset);
            let y2 = y1.add(128);
            let dall = ldh(x, 0);
            let dmin = ldh(x, 2);
            let aux = k4_aux(x.add(4), im as usize);
            let sc = |k: usize| aux_byte(&aux, k) as f32;
            let q1 = |k: usize| ld16(ql1, 2 * k);
            let q2 = |k: usize| ld16(ql1, 64 + 2 * k);
            let q16 = [
                q1(0) & 0x0f0f, q1(8) & 0x0f0f, (q1(0) >> 4) & 0x0f0f, (q1(8) >> 4) & 0x0f0f,
                q2(0) & 0x0f0f, q2(8) & 0x0f0f, (q2(0) >> 4) & 0x0f0f, (q2(8) >> 4) & 0x0f0f,
            ];
            let q4 = |k: usize| (q16[k / 2] >> (8 * (k % 2))) & 0xFF;
            let hb = |j: usize, mm: u32| if ld8(qh, j) & mm != 0 { 16 } else { 0 };
            let (mut sx, mut sy, mut sz, mut sw) = (0f32, 0f32, 0f32, 0f32);
            let mut smin = 0f32;
            let mut l = 0;
            while l < 2 {
                let v = |k: usize, j: usize, mm: u32| (q4(k) + hb(j, mm)) as f32;
                sx = add(sx, fma(*y1.add(l), v(l, l, hm1), mul(*y1.add(l + 16), v(l + 2, l + 16, hm1))));
                sy = add(sy, fma(*y1.add(l + 32), v(l + 4, l, hm1 << 1), mul(*y1.add(l + 48), v(l + 6, l + 16, hm1 << 1))));
                sz = add(sz, fma(*y2.add(l), v(l + 8, l, hm2), mul(*y2.add(l + 16), v(l + 10, l + 16, hm2))));
                sw = add(sw, fma(*y2.add(l + 32), v(l + 12, l, hm2 << 1), mul(*y2.add(l + 48), v(l + 14, l + 16, hm2 << 1))));
                smin = add(smin, chain_a(
                    [add(*y1.add(l), *y1.add(l + 16)), add(*y1.add(l + 32), *y1.add(l + 48)),
                     add(*y2.add(l), *y2.add(l + 16)), add(*y2.add(l + 32), *y2.add(l + 48))],
                    [sc(2), sc(3), sc(6), sc(7)],
                ));
                l += 1;
            }
            let inner = chain_b([sx, sy, sz, sw], [sc(0), sc(1), sc(4), sc(5)]);
            tmp = add(tmp, fma(dall, inner, -mul(dmin, smin)));
            i += 2;
        }
        tmp = warp_sum(tmp);
        if tx == 0 {
            *dst.offset(row as isize) = tmp;
        }
    }

    #[kernel]
    pub unsafe fn dequantize_mul_mat_vec_q6_k(vx: *const u8, yy: *const f32, dst: *mut f32, ncols: i32, nrows: i32) {
        let row = thread::blockIdx_x().wrapping_mul(thread::blockDim_y()).wrapping_add(thread::threadIdx_y()) as i32;
        if row > nrows {
            return;
        }
        let nb = ncols / 256;
        let x0 = vx.offset(row.wrapping_mul(nb) as isize * 210);
        let tx = thread::threadIdx_x() as i32;
        let tid = tx / 2;
        let ix = tx % 2;
        let im = tid / 8;
        let inn = tid - 8 * im;
        let l0 = 4 * inn;
        let is = inn / 4;
        let ql_offset = (64 * im + l0) as usize;
        let qh_offset = (32 * im + l0) as usize;
        let s_offset = (8 * im + is) as usize;
        let y_offset = (128 * im + l0) as usize;
        let mut tmp = 0f32;
        let mut i = ix;
        while i < nb {
            let x = x0.add(i as usize * 210);
            let y = yy.add(i as usize * 256 + y_offset);
            let ql = x.add(ql_offset);
            let qh = x.add(128 + qh_offset);
            let s = |k: usize| ld8s(x, 192 + s_offset + k) as f32;
            let d = ldh(x, 208);
            let mut sum = 0f32;
            let mut l = 0;
            while l < 4 {
                let h = ld8(qh, l);
                let v = |lo: u32, sh: u32| ((((lo | (((h >> sh) & 3) << 4)) as u8 as i8) as i32) - 32) as f32;
                let c = chain_a(
                    [mul(d, mul(*y.add(l), s(0))), mul(d, mul(*y.add(l + 32), s(2))),
                     mul(d, mul(*y.add(l + 64), s(4))), mul(d, mul(*y.add(l + 96), s(6)))],
                    [v(ld8(ql, l) & 0xF, 0), v(ld8(ql, l + 32) & 0xF, 2), v(ld8(ql, l) >> 4, 4), v(ld8(ql, l + 32) >> 4, 6)],
                );
                sum = add(sum, c);
                l += 1;
            }
            tmp = add(tmp, sum);
            i += 2;
        }
        tmp = warp_sum(tmp);
        if tid == 0 {
            *dst.offset(row as isize) = tmp;
        }
    }

    // ------------------------------------------------------------------------------------------
    // Family 3: quantize_q8_1 (1 entry). Launch: grid (kx_padded / 256, rows), block 256.

    #[kernel]
    pub unsafe fn quantize_q8_1(x: *const f32, vy: *mut u8, kx: i32, kx_padded: i32) {
        let ix = thread::blockDim_x().wrapping_mul(thread::blockIdx_x()).wrapping_add(thread::threadIdx_x()) as i32;
        if ix >= kx_padded {
            return;
        }
        let iy = thread::blockDim_y().wrapping_mul(thread::blockIdx_y()).wrapping_add(thread::threadIdx_y()) as i32;
        let i_padded = iy.wrapping_mul(kx_padded).wrapping_add(ix);
        let ib = i_padded / 32;
        let iqs = i_padded % 32;
        let xi = if ix < kx { *x.offset(iy.wrapping_mul(kx).wrapping_add(ix) as isize) } else { 0.0 };
        let mut amax = f32::from_bits(xi.to_bits() & 0x7FFF_FFFF); // abs.f32
        let mut sum = xi;
        let mut mask = 16;
        while mask > 0 {
            amax = amax.max(warp::shuffle_xor_f32_sync(0xffff_ffff, amax, mask));
            mask >>= 1;
        }
        sum = warp_sum(sum);
        let d = float::div_rn_f32(amax, 127.0);
        let q = if amax == 0.0 {
            0i32
        } else {
            // roundf: add.rz(t, copysign(0.5, t)), then truncate.
            let t = float::div_rn_f32(xi, d);
            let half = f32::from_bits(0x3F00_0000 | (t.to_bits() & 0x8000_0000));
            float::add_rz_f32(t, half) as i32
        };
        let y = vy.offset(ib as isize * 36);
        *y.offset(4 + iqs as isize) = q as u8;
        if iqs > 0 {
            return;
        }
        *(y as *mut u16) = f2h(d);
        *(y.add(2) as *mut u16) = f2h(sum);
    }

    // ------------------------------------------------------------------------------------------
    // Family 4: mul_mat_vec_*_q8_1_cudaN (80 entries) and indexed_moe_forward_* (6 entries),
    // both built on the MMVQ `vec_dot_*_q8_1`. FMT: 0 q4_0, 1 q4_1, 2 q5_0, 3 q5_1, 4 q8_0,
    // 5 q2_K, 6 q3_K, 7 q4_K, 8 q5_K, 9 q6_K.

    /// (qk, qi, vdr, block bytes) per format (MMVQ vdr).
    #[inline(always)]
    pub fn fmt_params<const FMT: u32>() -> (i32, i32, i32, usize) {
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
            11 => (256, 16, 2, 66), // iq2_xxs (GGML type 16)
            18 => (256, 16, 2, 98), // iq3_xxs: QR3_XXS=4 -> qi=QK_K/(4*QR)=16, vdr=2 (qi/vdr=8)
            23 => (256, 32, 4, 136), // iq4_xs (GGML type 23)
            17 => (256, 16, 2, 74), // iq2_xs (GGML type 17)
            22 => (256, 16, 2, 82), // iq2_s (GGML type 22)
            _ => (256, 32, 1, 210),
        }
    }

    /// `__vsubss4` (per-byte signed saturating subtract) at every call site in this file: all
    /// operands are masked so that no byte result leaves [-32, 31] (q3_K: [0,3] - {0,4}; q6_K and
    /// q3_K scales: [0,63] - 32; q5_0: [0,31] - 16), where saturation cannot trigger and the
    /// per-byte wrapping difference (SWAR, no carries across bytes) is the same value.
    #[inline(always)]
    pub fn vsubss4(a: u32, b: u32) -> u32 {
        ((a | 0x80808080).wrapping_sub(b & 0x7f7f7f7f)) ^ ((a ^ !b) & 0x80808080)
    }

    #[inline(always)]
    pub fn dp4a(a: u32, b: u32, c: i32) -> i32 {
        dotprod::dp4a_s32(a, b, c)
    }

    /// Q8_1 block `yb`: quant word `i` (4 int8), d (low half), s (high half).
    #[inline(always)]
    pub unsafe fn yq(yb: *const u8, i: usize) -> u32 {
        *(yb.add(4 + 4 * i) as *const u32)
    }

    /// `tmp += vec_dot_<FMT>_q8_1(xb, yb, iqs)` with the reference's contraction of the `+=`.
    #[inline(always)]

    // ---- IQ2_XXS (GGML type 16) -----------------------------------------------------------------
    // Table helpers and the sign/scale handling are lifted verbatim from the gated
    // oxide-kernels/iq2_xxs crate (bit-identical to llama.cpp acecd56).

    /// `__vsub4(a, b)`: per-byte subtract, each byte independent and wrapping (no borrow).
    #[inline(always)]
    pub fn vsub4_wrap(a: u32, b: u32) -> u32 {
        let (a0, a1, a2, a3) = (a & 0xFF, (a >> 8) & 0xFF, (a >> 16) & 0xFF, (a >> 24) & 0xFF);
        let (b0, b1, b2, b3) = (b & 0xFF, (b >> 8) & 0xFF, (b >> 16) & 0xFF, (b >> 24) & 0xFF);
        (a0.wrapping_sub(b0) & 0xFF)
            | ((a1.wrapping_sub(b1) & 0xFF) << 8)
            | ((a2.wrapping_sub(b2) & 0xFF) << 16)
            | ((a3.wrapping_sub(b3) & 0xFF) << 24)
    }

    /// `__vcmpne4(a, 0)`.
    #[inline(always)]
    pub fn vcmpne4_zero(a: u32) -> u32 {
        let (x, y, z, w) = (a & 0xFF, (a >> 8) & 0xFF, (a >> 16) & 0xFF, (a >> 24) & 0xFF);
        (if x != 0 { 0xFFu32 } else { 0 })
            | (if y != 0 { 0xFF00u32 } else { 0 })
            | (if z != 0 { 0xFF_0000u32 } else { 0 })
            | (if w != 0 { 0xFF00_0000u32 } else { 0 })
    }

    /// `get_int_b2(x, i)`: two `u16` loads.
    #[inline(always)]

    pub unsafe fn get_b4(x: *const u8, i: i32) -> u32 {
        // Assembled from two 16-bit reads: `qs` sits at an even (not 4-byte) offset
        // inside the i-quant blocks (xb + 2), so a plain u32 deref raises
        // CUDA_ERROR_MISALIGNED_ADDRESS.
        let p = x.add(4 * i as usize) as *const u16;
        (*p as u32) | ((*p.add(1) as u32) << 16)
    }

    pub fn byte_perm(a: u32, b: u32, s: u32) -> u32 {
        let r: u32;
        let s = s & 0x7777;
        unsafe { ptx_asm!("prmt.b32 %0, %1, %2, %3;", out("=r") r, in("r") a, in("r") b, in("r") s, options(register_only)); }
        r
    }

    pub const I4_TABLE: [u32; 4] = [0xBFAD9881, 0xF6EADDCF, 0x26190D01, 0x71594535];

    pub fn i4_table16(q4: u32) -> (u32, u32) {
            let sel = 0x32103210 | ((q4 & 0x88888888) >> 1);
            let lo0 = byte_perm(I4_TABLE[0], I4_TABLE[1], q4);
            let hi0 = byte_perm(I4_TABLE[2], I4_TABLE[3], q4);
            let t0 = byte_perm(lo0, hi0, sel);
            let lo1 = byte_perm(I4_TABLE[0], I4_TABLE[1], q4 >> 16);
            let hi1 = byte_perm(I4_TABLE[2], I4_TABLE[3], q4 >> 16);
            let t1 = byte_perm(lo1, hi1, sel >> 16);
            (byte_perm(t0, t1, 0x6420), byte_perm(t0, t1, 0x7531))
        }

    pub unsafe fn i4_group_scale6(x: *const u8, group: usize) -> u32 {
            let lo = *x.add(4 + group / 2) as u32;
            let hi = *(x.add(2) as *const u16) as u32;
            ((lo >> (4 * (group as u32 % 2))) & 0xF) | (((hi >> (2 * group as u32)) & 3) << 4)
        }

    pub fn i3_grid(i: u32) -> u32 {
            let mut v: u32 = 0;
            if i == 0u32 { v = 0x04040404u32 as u32; }
            if i == 1u32 { v = 0x04040414u32 as u32; }
            if i == 2u32 { v = 0x04040424u32 as u32; }
            if i == 3u32 { v = 0x04040c0cu32 as u32; }
            if i == 4u32 { v = 0x04040c1cu32 as u32; }
            if i == 5u32 { v = 0x04040c3eu32 as u32; }
            if i == 6u32 { v = 0x04041404u32 as u32; }
            if i == 7u32 { v = 0x04041414u32 as u32; }
            if i == 8u32 { v = 0x04041c0cu32 as u32; }
            if i == 9u32 { v = 0x04042414u32 as u32; }
            if i == 10u32 { v = 0x04043e1cu32 as u32; }
            if i == 11u32 { v = 0x04043e2cu32 as u32; }
            if i == 12u32 { v = 0x040c040cu32 as u32; }
            if i == 13u32 { v = 0x040c041cu32 as u32; }
            if i == 14u32 { v = 0x040c0c04u32 as u32; }
            if i == 15u32 { v = 0x040c0c14u32 as u32; }
            if i == 16u32 { v = 0x040c140cu32 as u32; }
            if i == 17u32 { v = 0x040c142cu32 as u32; }
            if i == 18u32 { v = 0x040c1c04u32 as u32; }
            if i == 19u32 { v = 0x040c1c14u32 as u32; }
            if i == 20u32 { v = 0x040c240cu32 as u32; }
            if i == 21u32 { v = 0x040c2c24u32 as u32; }
            if i == 22u32 { v = 0x040c3e04u32 as u32; }
            if i == 23u32 { v = 0x04140404u32 as u32; }
            if i == 24u32 { v = 0x04140414u32 as u32; }
            if i == 25u32 { v = 0x04140424u32 as u32; }
            if i == 26u32 { v = 0x04140c0cu32 as u32; }
            if i == 27u32 { v = 0x04141404u32 as u32; }
            if i == 28u32 { v = 0x04141414u32 as u32; }
            if i == 29u32 { v = 0x04141c0cu32 as u32; }
            if i == 30u32 { v = 0x04141c1cu32 as u32; }
            if i == 31u32 { v = 0x04141c3eu32 as u32; }
            if i == 32u32 { v = 0x04142c0cu32 as u32; }
            if i == 33u32 { v = 0x04142c3eu32 as u32; }
            if i == 34u32 { v = 0x04143e2cu32 as u32; }
            if i == 35u32 { v = 0x041c040cu32 as u32; }
            if i == 36u32 { v = 0x041c043eu32 as u32; }
            if i == 37u32 { v = 0x041c0c04u32 as u32; }
            if i == 38u32 { v = 0x041c0c14u32 as u32; }
            if i == 39u32 { v = 0x041c142cu32 as u32; }
            if i == 40u32 { v = 0x041c3e04u32 as u32; }
            if i == 41u32 { v = 0x04240c1cu32 as u32; }
            if i == 42u32 { v = 0x04241c3eu32 as u32; }
            if i == 43u32 { v = 0x04242424u32 as u32; }
            if i == 44u32 { v = 0x04242c3eu32 as u32; }
            if i == 45u32 { v = 0x04243e1cu32 as u32; }
            if i == 46u32 { v = 0x04243e2cu32 as u32; }
            if i == 47u32 { v = 0x042c040cu32 as u32; }
            if i == 48u32 { v = 0x042c043eu32 as u32; }
            if i == 49u32 { v = 0x042c1c14u32 as u32; }
            if i == 50u32 { v = 0x042c2c14u32 as u32; }
            if i == 51u32 { v = 0x04341c2cu32 as u32; }
            if i == 52u32 { v = 0x04343424u32 as u32; }
            if i == 53u32 { v = 0x043e0c04u32 as u32; }
            if i == 54u32 { v = 0x043e0c24u32 as u32; }
            if i == 55u32 { v = 0x043e0c34u32 as u32; }
            if i == 56u32 { v = 0x043e241cu32 as u32; }
            if i == 57u32 { v = 0x043e340cu32 as u32; }
            if i == 58u32 { v = 0x0c04040cu32 as u32; }
            if i == 59u32 { v = 0x0c04041cu32 as u32; }
            if i == 60u32 { v = 0x0c040c04u32 as u32; }
            if i == 61u32 { v = 0x0c040c14u32 as u32; }
            if i == 62u32 { v = 0x0c04140cu32 as u32; }
            if i == 63u32 { v = 0x0c04141cu32 as u32; }
            if i == 64u32 { v = 0x0c041c04u32 as u32; }
            if i == 65u32 { v = 0x0c041c14u32 as u32; }
            if i == 66u32 { v = 0x0c041c24u32 as u32; }
            if i == 67u32 { v = 0x0c04243eu32 as u32; }
            if i == 68u32 { v = 0x0c042c04u32 as u32; }
            if i == 69u32 { v = 0x0c0c0404u32 as u32; }
            if i == 70u32 { v = 0x0c0c0414u32 as u32; }
            if i == 71u32 { v = 0x0c0c0c0cu32 as u32; }
            if i == 72u32 { v = 0x0c0c1404u32 as u32; }
            if i == 73u32 { v = 0x0c0c1414u32 as u32; }
            if i == 74u32 { v = 0x0c14040cu32 as u32; }
            if i == 75u32 { v = 0x0c14041cu32 as u32; }
            if i == 76u32 { v = 0x0c140c04u32 as u32; }
            if i == 77u32 { v = 0x0c140c14u32 as u32; }
            if i == 78u32 { v = 0x0c14140cu32 as u32; }
            if i == 79u32 { v = 0x0c141c04u32 as u32; }
            if i == 80u32 { v = 0x0c143e14u32 as u32; }
            if i == 81u32 { v = 0x0c1c0404u32 as u32; }
            if i == 82u32 { v = 0x0c1c0414u32 as u32; }
            if i == 83u32 { v = 0x0c1c1404u32 as u32; }
            if i == 84u32 { v = 0x0c1c1c0cu32 as u32; }
            if i == 85u32 { v = 0x0c1c2434u32 as u32; }
            if i == 86u32 { v = 0x0c1c3434u32 as u32; }
            if i == 87u32 { v = 0x0c24040cu32 as u32; }
            if i == 88u32 { v = 0x0c24042cu32 as u32; }
            if i == 89u32 { v = 0x0c242c04u32 as u32; }
            if i == 90u32 { v = 0x0c2c1404u32 as u32; }
            if i == 91u32 { v = 0x0c2c1424u32 as u32; }
            if i == 92u32 { v = 0x0c2c2434u32 as u32; }
            if i == 93u32 { v = 0x0c2c3e0cu32 as u32; }
            if i == 94u32 { v = 0x0c34042cu32 as u32; }
            if i == 95u32 { v = 0x0c3e1414u32 as u32; }
            if i == 96u32 { v = 0x0c3e2404u32 as u32; }
            if i == 97u32 { v = 0x14040404u32 as u32; }
            if i == 98u32 { v = 0x14040414u32 as u32; }
            if i == 99u32 { v = 0x14040c0cu32 as u32; }
            if i == 100u32 { v = 0x14040c1cu32 as u32; }
            if i == 101u32 { v = 0x14041404u32 as u32; }
            if i == 102u32 { v = 0x14041414u32 as u32; }
            if i == 103u32 { v = 0x14041434u32 as u32; }
            if i == 104u32 { v = 0x14041c0cu32 as u32; }
            if i == 105u32 { v = 0x14042414u32 as u32; }
            if i == 106u32 { v = 0x140c040cu32 as u32; }
            if i == 107u32 { v = 0x140c041cu32 as u32; }
            if i == 108u32 { v = 0x140c042cu32 as u32; }
            if i == 109u32 { v = 0x140c0c04u32 as u32; }
            if i == 110u32 { v = 0x140c0c14u32 as u32; }
            if i == 111u32 { v = 0x140c140cu32 as u32; }
            if i == 112u32 { v = 0x140c1c04u32 as u32; }
            if i == 113u32 { v = 0x140c341cu32 as u32; }
            if i == 114u32 { v = 0x140c343eu32 as u32; }
            if i == 115u32 { v = 0x140c3e04u32 as u32; }
            if i == 116u32 { v = 0x14140404u32 as u32; }
            if i == 117u32 { v = 0x14140414u32 as u32; }
            if i == 118u32 { v = 0x14140c0cu32 as u32; }
            if i == 119u32 { v = 0x14140c3eu32 as u32; }
            if i == 120u32 { v = 0x14141404u32 as u32; }
            if i == 121u32 { v = 0x14141414u32 as u32; }
            if i == 122u32 { v = 0x14141c3eu32 as u32; }
            if i == 123u32 { v = 0x14142404u32 as u32; }
            if i == 124u32 { v = 0x14142c2cu32 as u32; }
            if i == 125u32 { v = 0x141c040cu32 as u32; }
            if i == 126u32 { v = 0x141c0c04u32 as u32; }
            if i == 127u32 { v = 0x141c0c24u32 as u32; }
            if i == 128u32 { v = 0x141c3e04u32 as u32; }
            if i == 129u32 { v = 0x141c3e24u32 as u32; }
            if i == 130u32 { v = 0x14241c2cu32 as u32; }
            if i == 131u32 { v = 0x14242c1cu32 as u32; }
            if i == 132u32 { v = 0x142c041cu32 as u32; }
            if i == 133u32 { v = 0x142c143eu32 as u32; }
            if i == 134u32 { v = 0x142c240cu32 as u32; }
            if i == 135u32 { v = 0x142c3e24u32 as u32; }
            if i == 136u32 { v = 0x143e040cu32 as u32; }
            if i == 137u32 { v = 0x143e041cu32 as u32; }
            if i == 138u32 { v = 0x143e0c34u32 as u32; }
            if i == 139u32 { v = 0x143e242cu32 as u32; }
            if i == 140u32 { v = 0x1c04040cu32 as u32; }
            if i == 141u32 { v = 0x1c040c04u32 as u32; }
            if i == 142u32 { v = 0x1c040c14u32 as u32; }
            if i == 143u32 { v = 0x1c04140cu32 as u32; }
            if i == 144u32 { v = 0x1c04141cu32 as u32; }
            if i == 145u32 { v = 0x1c042c04u32 as u32; }
            if i == 146u32 { v = 0x1c04342cu32 as u32; }
            if i == 147u32 { v = 0x1c043e14u32 as u32; }
            if i == 148u32 { v = 0x1c0c0404u32 as u32; }
            if i == 149u32 { v = 0x1c0c0414u32 as u32; }
            if i == 150u32 { v = 0x1c0c1404u32 as u32; }
            if i == 151u32 { v = 0x1c0c1c0cu32 as u32; }
            if i == 152u32 { v = 0x1c0c2424u32 as u32; }
            if i == 153u32 { v = 0x1c0c2434u32 as u32; }
            if i == 154u32 { v = 0x1c14040cu32 as u32; }
            if i == 155u32 { v = 0x1c14041cu32 as u32; }
            if i == 156u32 { v = 0x1c140c04u32 as u32; }
            if i == 157u32 { v = 0x1c14142cu32 as u32; }
            if i == 158u32 { v = 0x1c142c14u32 as u32; }
            if i == 159u32 { v = 0x1c143e14u32 as u32; }
            if i == 160u32 { v = 0x1c1c0c0cu32 as u32; }
            if i == 161u32 { v = 0x1c1c1c1cu32 as u32; }
            if i == 162u32 { v = 0x1c241c04u32 as u32; }
            if i == 163u32 { v = 0x1c24243eu32 as u32; }
            if i == 164u32 { v = 0x1c243e14u32 as u32; }
            if i == 165u32 { v = 0x1c2c0404u32 as u32; }
            if i == 166u32 { v = 0x1c2c0434u32 as u32; }
            if i == 167u32 { v = 0x1c2c1414u32 as u32; }
            if i == 168u32 { v = 0x1c2c2c2cu32 as u32; }
            if i == 169u32 { v = 0x1c340c24u32 as u32; }
            if i == 170u32 { v = 0x1c341c34u32 as u32; }
            if i == 171u32 { v = 0x1c34341cu32 as u32; }
            if i == 172u32 { v = 0x1c3e1c1cu32 as u32; }
            if i == 173u32 { v = 0x1c3e3404u32 as u32; }
            if i == 174u32 { v = 0x24040424u32 as u32; }
            if i == 175u32 { v = 0x24040c3eu32 as u32; }
            if i == 176u32 { v = 0x24041c2cu32 as u32; }
            if i == 177u32 { v = 0x24041c3eu32 as u32; }
            if i == 178u32 { v = 0x24042c1cu32 as u32; }
            if i == 179u32 { v = 0x24042c3eu32 as u32; }
            if i == 180u32 { v = 0x240c3e24u32 as u32; }
            if i == 181u32 { v = 0x24141404u32 as u32; }
            if i == 182u32 { v = 0x24141c3eu32 as u32; }
            if i == 183u32 { v = 0x24142404u32 as u32; }
            if i == 184u32 { v = 0x24143404u32 as u32; }
            if i == 185u32 { v = 0x24143434u32 as u32; }
            if i == 186u32 { v = 0x241c043eu32 as u32; }
            if i == 187u32 { v = 0x241c242cu32 as u32; }
            if i == 188u32 { v = 0x24240424u32 as u32; }
            if i == 189u32 { v = 0x24242c0cu32 as u32; }
            if i == 190u32 { v = 0x24243424u32 as u32; }
            if i == 191u32 { v = 0x242c142cu32 as u32; }
            if i == 192u32 { v = 0x242c241cu32 as u32; }
            if i == 193u32 { v = 0x242c3e04u32 as u32; }
            if i == 194u32 { v = 0x243e042cu32 as u32; }
            if i == 195u32 { v = 0x243e0c04u32 as u32; }
            if i == 196u32 { v = 0x243e0c14u32 as u32; }
            if i == 197u32 { v = 0x243e1c04u32 as u32; }
            if i == 198u32 { v = 0x2c040c14u32 as u32; }
            if i == 199u32 { v = 0x2c04240cu32 as u32; }
            if i == 200u32 { v = 0x2c043e04u32 as u32; }
            if i == 201u32 { v = 0x2c0c0404u32 as u32; }
            if i == 202u32 { v = 0x2c0c0434u32 as u32; }
            if i == 203u32 { v = 0x2c0c1434u32 as u32; }
            if i == 204u32 { v = 0x2c0c2c2cu32 as u32; }
            if i == 205u32 { v = 0x2c140c24u32 as u32; }
            if i == 206u32 { v = 0x2c141c14u32 as u32; }
            if i == 207u32 { v = 0x2c143e14u32 as u32; }
            if i == 208u32 { v = 0x2c1c0414u32 as u32; }
            if i == 209u32 { v = 0x2c1c2c1cu32 as u32; }
            if i == 210u32 { v = 0x2c240c04u32 as u32; }
            if i == 211u32 { v = 0x2c24141cu32 as u32; }
            if i == 212u32 { v = 0x2c24143eu32 as u32; }
            if i == 213u32 { v = 0x2c243e14u32 as u32; }
            if i == 214u32 { v = 0x2c2c0414u32 as u32; }
            if i == 215u32 { v = 0x2c2c1c0cu32 as u32; }
            if i == 216u32 { v = 0x2c342c04u32 as u32; }
            if i == 217u32 { v = 0x2c3e1424u32 as u32; }
            if i == 218u32 { v = 0x2c3e2414u32 as u32; }
            if i == 219u32 { v = 0x34041424u32 as u32; }
            if i == 220u32 { v = 0x34042424u32 as u32; }
            if i == 221u32 { v = 0x34042434u32 as u32; }
            if i == 222u32 { v = 0x34043424u32 as u32; }
            if i == 223u32 { v = 0x340c140cu32 as u32; }
            if i == 224u32 { v = 0x340c340cu32 as u32; }
            if i == 225u32 { v = 0x34140c3eu32 as u32; }
            if i == 226u32 { v = 0x34143424u32 as u32; }
            if i == 227u32 { v = 0x341c1c04u32 as u32; }
            if i == 228u32 { v = 0x341c1c34u32 as u32; }
            if i == 229u32 { v = 0x34242424u32 as u32; }
            if i == 230u32 { v = 0x342c042cu32 as u32; }
            if i == 231u32 { v = 0x342c2c14u32 as u32; }
            if i == 232u32 { v = 0x34341c1cu32 as u32; }
            if i == 233u32 { v = 0x343e041cu32 as u32; }
            if i == 234u32 { v = 0x343e140cu32 as u32; }
            if i == 235u32 { v = 0x3e04041cu32 as u32; }
            if i == 236u32 { v = 0x3e04042cu32 as u32; }
            if i == 237u32 { v = 0x3e04043eu32 as u32; }
            if i == 238u32 { v = 0x3e040c04u32 as u32; }
            if i == 239u32 { v = 0x3e041c14u32 as u32; }
            if i == 240u32 { v = 0x3e042c14u32 as u32; }
            if i == 241u32 { v = 0x3e0c1434u32 as u32; }
            if i == 242u32 { v = 0x3e0c2404u32 as u32; }
            if i == 243u32 { v = 0x3e140c14u32 as u32; }
            if i == 244u32 { v = 0x3e14242cu32 as u32; }
            if i == 245u32 { v = 0x3e142c14u32 as u32; }
            if i == 246u32 { v = 0x3e1c0404u32 as u32; }
            if i == 247u32 { v = 0x3e1c0c2cu32 as u32; }
            if i == 248u32 { v = 0x3e1c1c1cu32 as u32; }
            if i == 249u32 { v = 0x3e1c3404u32 as u32; }
            if i == 250u32 { v = 0x3e24140cu32 as u32; }
            if i == 251u32 { v = 0x3e24240cu32 as u32; }
            if i == 252u32 { v = 0x3e2c0404u32 as u32; }
            if i == 253u32 { v = 0x3e2c0414u32 as u32; }
            if i == 254u32 { v = 0x3e2c1424u32 as u32; }
            if i == 255u32 { v = 0x3e341c04u32 as u32; }
            v
        }

    pub unsafe fn get_b2(x: *const u8, i: i32) -> u32 {
        let p = x.add(4 * i as usize) as *const u16;
        (*p as u32) | ((*p.add(1) as u32) << 16)
    }

pub fn popc7(v: u32) -> u32 {
        let mut x = v & 0x7F;
        let mut n = 0u32;
        while x != 0 {
            n += x & 1;
            x >>= 1;
        }
        n
    }

pub fn unpack_ksigns(v: u32) -> u32 {
        let p = popc7(v) & 1;
        let s = (v & 0x7F) ^ (p << 7);
        s.wrapping_mul(0x0101_0101)
    }

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

pub fn s_grid_lo(i: u32) -> u32 {
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

pub fn s_grid_hi(i: u32) -> u32 {
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

pub fn xs_grid_lo(i: u32) -> u32 {
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
        if i == 23u32 { v = 0x2b080808u32; }
        if i == 24u32 { v = 0x2b08082bu32; }
        if i == 25u32 { v = 0x2b081919u32; }
        if i == 26u32 { v = 0x2b082b08u32; }
        if i == 27u32 { v = 0x2b190819u32; }
        if i == 28u32 { v = 0x2b191908u32; }
        if i == 29u32 { v = 0x2b192b19u32; }
        if i == 30u32 { v = 0x2b2b0808u32; }
        if i == 31u32 { v = 0x08080819u32; }
        if i == 32u32 { v = 0x08081908u32; }
        if i == 33u32 { v = 0x0808192bu32; }
        if i == 34u32 { v = 0x08082b19u32; }
        if i == 35u32 { v = 0x08190808u32; }
        if i == 36u32 { v = 0x0819082bu32; }
        if i == 37u32 { v = 0x08191919u32; }
        if i == 38u32 { v = 0x08192b08u32; }
        if i == 39u32 { v = 0x08192b2bu32; }
        if i == 40u32 { v = 0x082b0819u32; }
        if i == 41u32 { v = 0x082b1908u32; }
        if i == 42u32 { v = 0x19080808u32; }
        if i == 43u32 { v = 0x1908082bu32; }
        if i == 44u32 { v = 0x19081919u32; }
        if i == 45u32 { v = 0x19082b08u32; }
        if i == 46u32 { v = 0x19190819u32; }
        if i == 47u32 { v = 0x19191908u32; }
        if i == 48u32 { v = 0x192b0808u32; }
        if i == 49u32 { v = 0x192b2b08u32; }
        if i == 50u32 { v = 0x2b080819u32; }
        if i == 51u32 { v = 0x2b081908u32; }
        if i == 52u32 { v = 0x2b190808u32; }
        if i == 53u32 { v = 0x08080808u32; }
        if i == 54u32 { v = 0x0808082bu32; }
        if i == 55u32 { v = 0x08081919u32; }
        if i == 56u32 { v = 0x08082b08u32; }
        if i == 57u32 { v = 0x08190819u32; }
        if i == 58u32 { v = 0x08191908u32; }
        if i == 59u32 { v = 0x082b0808u32; }
        if i == 60u32 { v = 0x19080819u32; }
        if i == 61u32 { v = 0x19081908u32; }
        if i == 62u32 { v = 0x19190808u32; }
        if i == 63u32 { v = 0x19191919u32; }
        if i == 64u32 { v = 0x2b080808u32; }
        if i == 65u32 { v = 0x2b082b2bu32; }
        if i == 66u32 { v = 0x08080819u32; }
        if i == 67u32 { v = 0x08081908u32; }
        if i == 68u32 { v = 0x0808192bu32; }
        if i == 69u32 { v = 0x08082b19u32; }
        if i == 70u32 { v = 0x08190808u32; }
        if i == 71u32 { v = 0x0819082bu32; }
        if i == 72u32 { v = 0x08191919u32; }
        if i == 73u32 { v = 0x08192b08u32; }
        if i == 74u32 { v = 0x082b0819u32; }
        if i == 75u32 { v = 0x082b1908u32; }
        if i == 76u32 { v = 0x19080808u32; }
        if i == 77u32 { v = 0x1908082bu32; }
        if i == 78u32 { v = 0x19081919u32; }
        if i == 79u32 { v = 0x19082b08u32; }
        if i == 80u32 { v = 0x19190819u32; }
        if i == 81u32 { v = 0x19191908u32; }
        if i == 82u32 { v = 0x1919192bu32; }
        if i == 83u32 { v = 0x192b0808u32; }
        if i == 84u32 { v = 0x2b080819u32; }
        if i == 85u32 { v = 0x2b081908u32; }
        if i == 86u32 { v = 0x2b190808u32; }
        if i == 87u32 { v = 0x08080808u32; }
        if i == 88u32 { v = 0x0808082bu32; }
        if i == 89u32 { v = 0x08081919u32; }
        if i == 90u32 { v = 0x08082b08u32; }
        if i == 91u32 { v = 0x08190819u32; }
        if i == 92u32 { v = 0x08191908u32; }
        if i == 93u32 { v = 0x082b0808u32; }
        if i == 94u32 { v = 0x19080819u32; }
        if i == 95u32 { v = 0x19081908u32; }
        if i == 96u32 { v = 0x19190808u32; }
        if i == 97u32 { v = 0x192b0819u32; }
        if i == 98u32 { v = 0x2b080808u32; }
        if i == 99u32 { v = 0x08080819u32; }
        if i == 100u32 { v = 0x08081908u32; }
        if i == 101u32 { v = 0x08190808u32; }
        if i == 102u32 { v = 0x082b192bu32; }
        if i == 103u32 { v = 0x19080808u32; }
        if i == 104u32 { v = 0x1908082bu32; }
        if i == 105u32 { v = 0x2b081908u32; }
        if i == 106u32 { v = 0x08080808u32; }
        if i == 107u32 { v = 0x0808082bu32; }
        if i == 108u32 { v = 0x08081919u32; }
        if i == 109u32 { v = 0x08082b08u32; }
        if i == 110u32 { v = 0x08082b2bu32; }
        if i == 111u32 { v = 0x08190819u32; }
        if i == 112u32 { v = 0x08191908u32; }
        if i == 113u32 { v = 0x082b0808u32; }
        if i == 114u32 { v = 0x082b1919u32; }
        if i == 115u32 { v = 0x19080819u32; }
        if i == 116u32 { v = 0x19081908u32; }
        if i == 117u32 { v = 0x19190808u32; }
        if i == 118u32 { v = 0x19192b08u32; }
        if i == 119u32 { v = 0x2b080808u32; }
        if i == 120u32 { v = 0x2b2b0808u32; }
        if i == 121u32 { v = 0x2b2b2b2bu32; }
        if i == 122u32 { v = 0x08080819u32; }
        if i == 123u32 { v = 0x08081908u32; }
        if i == 124u32 { v = 0x08190808u32; }
        if i == 125u32 { v = 0x19080808u32; }
        if i == 126u32 { v = 0x2b080819u32; }
        if i == 127u32 { v = 0x2b082b19u32; }
        if i == 128u32 { v = 0x08080808u32; }
        if i == 129u32 { v = 0x082b0808u32; }
        if i == 130u32 { v = 0x082b2b08u32; }
        if i == 131u32 { v = 0x2b19192bu32; }
        if i == 132u32 { v = 0x2b2b0808u32; }
        if i == 133u32 { v = 0x08080819u32; }
        if i == 134u32 { v = 0x08081908u32; }
        if i == 135u32 { v = 0x0808192bu32; }
        if i == 136u32 { v = 0x08082b19u32; }
        if i == 137u32 { v = 0x08190808u32; }
        if i == 138u32 { v = 0x0819082bu32; }
        if i == 139u32 { v = 0x08191919u32; }
        if i == 140u32 { v = 0x08192b08u32; }
        if i == 141u32 { v = 0x082b0819u32; }
        if i == 142u32 { v = 0x082b1908u32; }
        if i == 143u32 { v = 0x19080808u32; }
        if i == 144u32 { v = 0x1908082bu32; }
        if i == 145u32 { v = 0x19081919u32; }
        if i == 146u32 { v = 0x19082b08u32; }
        if i == 147u32 { v = 0x19190819u32; }
        if i == 148u32 { v = 0x19191908u32; }
        if i == 149u32 { v = 0x192b0808u32; }
        if i == 150u32 { v = 0x192b2b2bu32; }
        if i == 151u32 { v = 0x2b080819u32; }
        if i == 152u32 { v = 0x2b081908u32; }
        if i == 153u32 { v = 0x2b190808u32; }
        if i == 154u32 { v = 0x08080808u32; }
        if i == 155u32 { v = 0x0808082bu32; }
        if i == 156u32 { v = 0x08081919u32; }
        if i == 157u32 { v = 0x08082b08u32; }
        if i == 158u32 { v = 0x08190819u32; }
        if i == 159u32 { v = 0x08191908u32; }
        if i == 160u32 { v = 0x082b0808u32; }
        if i == 161u32 { v = 0x19080819u32; }
        if i == 162u32 { v = 0x19081908u32; }
        if i == 163u32 { v = 0x19190808u32; }
        if i == 164u32 { v = 0x2b080808u32; }
        if i == 165u32 { v = 0x2b191908u32; }
        if i == 166u32 { v = 0x2b19192bu32; }
        if i == 167u32 { v = 0x08080819u32; }
        if i == 168u32 { v = 0x08081908u32; }
        if i == 169u32 { v = 0x0808192bu32; }
        if i == 170u32 { v = 0x08190808u32; }
        if i == 171u32 { v = 0x19080808u32; }
        if i == 172u32 { v = 0x192b0808u32; }
        if i == 173u32 { v = 0x08080808u32; }
        if i == 174u32 { v = 0x0808082bu32; }
        if i == 175u32 { v = 0x08081919u32; }
        if i == 176u32 { v = 0x08082b08u32; }
        if i == 177u32 { v = 0x08190819u32; }
        if i == 178u32 { v = 0x08191908u32; }
        if i == 179u32 { v = 0x082b0808u32; }
        if i == 180u32 { v = 0x19080819u32; }
        if i == 181u32 { v = 0x19081908u32; }
        if i == 182u32 { v = 0x19082b19u32; }
        if i == 183u32 { v = 0x19190808u32; }
        if i == 184u32 { v = 0x192b1908u32; }
        if i == 185u32 { v = 0x2b080808u32; }
        if i == 186u32 { v = 0x08080819u32; }
        if i == 187u32 { v = 0x08081908u32; }
        if i == 188u32 { v = 0x08190808u32; }
        if i == 189u32 { v = 0x19080808u32; }
        if i == 190u32 { v = 0x08080808u32; }
        if i == 191u32 { v = 0x08191908u32; }
        if i == 192u32 { v = 0x19082b19u32; }
        if i == 193u32 { v = 0x08080819u32; }
        if i == 194u32 { v = 0x08081908u32; }
        if i == 195u32 { v = 0x08190808u32; }
        if i == 196u32 { v = 0x0819082bu32; }
        if i == 197u32 { v = 0x19080808u32; }
        if i == 198u32 { v = 0x19191908u32; }
        if i == 199u32 { v = 0x2b08192bu32; }
        if i == 200u32 { v = 0x08080808u32; }
        if i == 201u32 { v = 0x08081919u32; }
        if i == 202u32 { v = 0x192b192bu32; }
        if i == 203u32 { v = 0x19190819u32; }
        if i == 204u32 { v = 0x2b2b2b19u32; }
        if i == 205u32 { v = 0x08080808u32; }
        if i == 206u32 { v = 0x0808082bu32; }
        if i == 207u32 { v = 0x08081919u32; }
        if i == 208u32 { v = 0x08082b08u32; }
        if i == 209u32 { v = 0x08082b2bu32; }
        if i == 210u32 { v = 0x08190819u32; }
        if i == 211u32 { v = 0x08191908u32; }
        if i == 212u32 { v = 0x082b0808u32; }
        if i == 213u32 { v = 0x19080819u32; }
        if i == 214u32 { v = 0x19081908u32; }
        if i == 215u32 { v = 0x19190808u32; }
        if i == 216u32 { v = 0x2b080808u32; }
        if i == 217u32 { v = 0x2b2b0808u32; }
        if i == 218u32 { v = 0x08080819u32; }
        if i == 219u32 { v = 0x08081908u32; }
        if i == 220u32 { v = 0x08190808u32; }
        if i == 221u32 { v = 0x19080808u32; }
        if i == 222u32 { v = 0x19082b08u32; }
        if i == 223u32 { v = 0x192b1919u32; }
        if i == 224u32 { v = 0x08080808u32; }
        if i == 225u32 { v = 0x082b082bu32; }
        if i == 226u32 { v = 0x2b080808u32; }
        if i == 227u32 { v = 0x2b2b2b08u32; }
        if i == 228u32 { v = 0x08080819u32; }
        if i == 229u32 { v = 0x08081908u32; }
        if i == 230u32 { v = 0x08190808u32; }
        if i == 231u32 { v = 0x082b2b19u32; }
        if i == 232u32 { v = 0x19080808u32; }
        if i == 233u32 { v = 0x08080808u32; }
        if i == 234u32 { v = 0x19080819u32; }
        if i == 235u32 { v = 0x1919082bu32; }
        if i == 236u32 { v = 0x2b192b19u32; }
        if i == 237u32 { v = 0x08080819u32; }
        if i == 238u32 { v = 0x08192b2bu32; }
        if i == 239u32 { v = 0x2b2b192bu32; }
        if i == 240u32 { v = 0x08080808u32; }
        if i == 241u32 { v = 0x08082b08u32; }
        if i == 242u32 { v = 0x08082b2bu32; }
        if i == 243u32 { v = 0x082b0808u32; }
        if i == 244u32 { v = 0x19191919u32; }
        if i == 245u32 { v = 0x2b082b08u32; }
        if i == 246u32 { v = 0x2b2b082bu32; }
        if i == 247u32 { v = 0x192b2b08u32; }
        if i == 248u32 { v = 0x2b190808u32; }
        if i == 249u32 { v = 0x08082b08u32; }
        if i == 250u32 { v = 0x082b0808u32; }
        if i == 251u32 { v = 0x2b08082bu32; }
        if i == 252u32 { v = 0x2b082b08u32; }
        if i == 253u32 { v = 0x2b082b2bu32; }
        if i == 254u32 { v = 0x08080819u32; }
        if i == 255u32 { v = 0x08081908u32; }
        if i == 256u32 { v = 0x0808192bu32; }
        if i == 257u32 { v = 0x08082b19u32; }
        if i == 258u32 { v = 0x08190808u32; }
        if i == 259u32 { v = 0x0819082bu32; }
        if i == 260u32 { v = 0x08191919u32; }
        if i == 261u32 { v = 0x08192b08u32; }
        if i == 262u32 { v = 0x082b0819u32; }
        if i == 263u32 { v = 0x082b1908u32; }
        if i == 264u32 { v = 0x19080808u32; }
        if i == 265u32 { v = 0x1908082bu32; }
        if i == 266u32 { v = 0x19081919u32; }
        if i == 267u32 { v = 0x19082b08u32; }
        if i == 268u32 { v = 0x19082b2bu32; }
        if i == 269u32 { v = 0x19190819u32; }
        if i == 270u32 { v = 0x19191908u32; }
        if i == 271u32 { v = 0x192b0808u32; }
        if i == 272u32 { v = 0x192b1919u32; }
        if i == 273u32 { v = 0x2b080819u32; }
        if i == 274u32 { v = 0x2b081908u32; }
        if i == 275u32 { v = 0x2b190808u32; }
        if i == 276u32 { v = 0x08080808u32; }
        if i == 277u32 { v = 0x0808082bu32; }
        if i == 278u32 { v = 0x08081919u32; }
        if i == 279u32 { v = 0x08082b08u32; }
        if i == 280u32 { v = 0x08190819u32; }
        if i == 281u32 { v = 0x08191908u32; }
        if i == 282u32 { v = 0x082b0808u32; }
        if i == 283u32 { v = 0x19080819u32; }
        if i == 284u32 { v = 0x19081908u32; }
        if i == 285u32 { v = 0x19190808u32; }
        if i == 286u32 { v = 0x2b080808u32; }
        if i == 287u32 { v = 0x2b081919u32; }
        if i == 288u32 { v = 0x2b2b082bu32; }
        if i == 289u32 { v = 0x08080819u32; }
        if i == 290u32 { v = 0x08081908u32; }
        if i == 291u32 { v = 0x08190808u32; }
        if i == 292u32 { v = 0x0819082bu32; }
        if i == 293u32 { v = 0x082b2b19u32; }
        if i == 294u32 { v = 0x19080808u32; }
        if i == 295u32 { v = 0x08080808u32; }
        if i == 296u32 { v = 0x0808082bu32; }
        if i == 297u32 { v = 0x08081919u32; }
        if i == 298u32 { v = 0x08082b08u32; }
        if i == 299u32 { v = 0x08190819u32; }
        if i == 300u32 { v = 0x08191908u32; }
        if i == 301u32 { v = 0x08192b19u32; }
        if i == 302u32 { v = 0x082b0808u32; }
        if i == 303u32 { v = 0x19080819u32; }
        if i == 304u32 { v = 0x19081908u32; }
        if i == 305u32 { v = 0x19190808u32; }
        if i == 306u32 { v = 0x2b080808u32; }
        if i == 307u32 { v = 0x2b191908u32; }
        if i == 308u32 { v = 0x08080819u32; }
        if i == 309u32 { v = 0x08081908u32; }
        if i == 310u32 { v = 0x08190808u32; }
        if i == 311u32 { v = 0x082b1908u32; }
        if i == 312u32 { v = 0x19080808u32; }
        if i == 313u32 { v = 0x2b192b2bu32; }
        if i == 314u32 { v = 0x08080808u32; }
        if i == 315u32 { v = 0x08082b2bu32; }
        if i == 316u32 { v = 0x19081908u32; }
        if i == 317u32 { v = 0x19190808u32; }
        if i == 318u32 { v = 0x08080819u32; }
        if i == 319u32 { v = 0x08081908u32; }
        if i == 320u32 { v = 0x08190808u32; }
        if i == 321u32 { v = 0x19080808u32; }
        if i == 322u32 { v = 0x19081919u32; }
        if i == 323u32 { v = 0x19191908u32; }
        if i == 324u32 { v = 0x192b082bu32; }
        if i == 325u32 { v = 0x08080808u32; }
        if i == 326u32 { v = 0x08190819u32; }
        if i == 327u32 { v = 0x19081908u32; }
        if i == 328u32 { v = 0x19190808u32; }
        if i == 329u32 { v = 0x192b2b19u32; }
        if i == 330u32 { v = 0x08081908u32; }
        if i == 331u32 { v = 0x08080808u32; }
        if i == 332u32 { v = 0x0808082bu32; }
        if i == 333u32 { v = 0x08081919u32; }
        if i == 334u32 { v = 0x08082b08u32; }
        if i == 335u32 { v = 0x08190819u32; }
        if i == 336u32 { v = 0x08191908u32; }
        if i == 337u32 { v = 0x082b0808u32; }
        if i == 338u32 { v = 0x082b2b08u32; }
        if i == 339u32 { v = 0x19080819u32; }
        if i == 340u32 { v = 0x19081908u32; }
        if i == 341u32 { v = 0x19190808u32; }
        if i == 342u32 { v = 0x2b080808u32; }
        if i == 343u32 { v = 0x08080819u32; }
        if i == 344u32 { v = 0x08081908u32; }
        if i == 345u32 { v = 0x08190808u32; }
        if i == 346u32 { v = 0x08191919u32; }
        if i == 347u32 { v = 0x19080808u32; }
        if i == 348u32 { v = 0x1908082bu32; }
        if i == 349u32 { v = 0x08080808u32; }
        if i == 350u32 { v = 0x19081908u32; }
        if i == 351u32 { v = 0x2b2b2b2bu32; }
        if i == 352u32 { v = 0x08080819u32; }
        if i == 353u32 { v = 0x08081908u32; }
        if i == 354u32 { v = 0x08190808u32; }
        if i == 355u32 { v = 0x082b0819u32; }
        if i == 356u32 { v = 0x19080808u32; }
        if i == 357u32 { v = 0x192b0808u32; }
        if i == 358u32 { v = 0x2b080819u32; }
        if i == 359u32 { v = 0x2b2b0819u32; }
        if i == 360u32 { v = 0x08080808u32; }
        if i == 361u32 { v = 0x08082b08u32; }
        if i == 362u32 { v = 0x2b080808u32; }
        if i == 363u32 { v = 0x2b082b08u32; }
        if i == 364u32 { v = 0x082b0819u32; }
        if i == 365u32 { v = 0x192b2b08u32; }
        if i == 366u32 { v = 0x2b2b0819u32; }
        if i == 367u32 { v = 0x08080808u32; }
        if i == 368u32 { v = 0x08191908u32; }
        if i == 369u32 { v = 0x19080819u32; }
        if i == 370u32 { v = 0x19190808u32; }
        if i == 371u32 { v = 0x2b192b19u32; }
        if i == 372u32 { v = 0x08192b2bu32; }
        if i == 373u32 { v = 0x19080808u32; }
        if i == 374u32 { v = 0x1908082bu32; }
        if i == 375u32 { v = 0x2b081919u32; }
        if i == 376u32 { v = 0x08080819u32; }
        if i == 377u32 { v = 0x08081908u32; }
        if i == 378u32 { v = 0x08190808u32; }
        if i == 379u32 { v = 0x19080808u32; }
        if i == 380u32 { v = 0x19191908u32; }
        if i == 381u32 { v = 0x192b082bu32; }
        if i == 382u32 { v = 0x2b08192bu32; }
        if i == 383u32 { v = 0x2b2b2b19u32; }
        if i == 384u32 { v = 0x08080808u32; }
        if i == 385u32 { v = 0x082b1908u32; }
        if i == 386u32 { v = 0x19082b2bu32; }
        if i == 387u32 { v = 0x2b19082bu32; }
        if i == 388u32 { v = 0x08080808u32; }
        if i == 389u32 { v = 0x0819192bu32; }
        if i == 390u32 { v = 0x08190808u32; }
        if i == 391u32 { v = 0x19080808u32; }
        if i == 392u32 { v = 0x19081919u32; }
        if i == 393u32 { v = 0x2b2b1908u32; }
        if i == 394u32 { v = 0x08080819u32; }
        if i == 395u32 { v = 0x192b2b2bu32; }
        if i == 396u32 { v = 0x082b1919u32; }
        if i == 397u32 { v = 0x0808192bu32; }
        if i == 398u32 { v = 0x19191908u32; }
        if i == 399u32 { v = 0x192b082bu32; }
        if i == 400u32 { v = 0x08080808u32; }
        if i == 401u32 { v = 0x0808082bu32; }
        if i == 402u32 { v = 0x08081919u32; }
        if i == 403u32 { v = 0x08082b08u32; }
        if i == 404u32 { v = 0x08190819u32; }
        if i == 405u32 { v = 0x08191908u32; }
        if i == 406u32 { v = 0x082b0808u32; }
        if i == 407u32 { v = 0x082b2b2bu32; }
        if i == 408u32 { v = 0x19080819u32; }
        if i == 409u32 { v = 0x19081908u32; }
        if i == 410u32 { v = 0x19190808u32; }
        if i == 411u32 { v = 0x2b080808u32; }
        if i == 412u32 { v = 0x2b08082bu32; }
        if i == 413u32 { v = 0x2b2b2b08u32; }
        if i == 414u32 { v = 0x2b2b2b2bu32; }
        if i == 415u32 { v = 0x08080819u32; }
        if i == 416u32 { v = 0x08081908u32; }
        if i == 417u32 { v = 0x0808192bu32; }
        if i == 418u32 { v = 0x08190808u32; }
        if i == 419u32 { v = 0x19080808u32; }
        if i == 420u32 { v = 0x19190819u32; }
        if i == 421u32 { v = 0x19192b19u32; }
        if i == 422u32 { v = 0x08080808u32; }
        if i == 423u32 { v = 0x082b0808u32; }
        if i == 424u32 { v = 0x2b080808u32; }
        if i == 425u32 { v = 0x2b08082bu32; }
        if i == 426u32 { v = 0x2b2b0808u32; }
        if i == 427u32 { v = 0x2b2b2b08u32; }
        if i == 428u32 { v = 0x08080819u32; }
        if i == 429u32 { v = 0x08081908u32; }
        if i == 430u32 { v = 0x08190808u32; }
        if i == 431u32 { v = 0x0819082bu32; }
        if i == 432u32 { v = 0x08191919u32; }
        if i == 433u32 { v = 0x19080808u32; }
        if i == 434u32 { v = 0x192b0808u32; }
        if i == 435u32 { v = 0x2b082b19u32; }
        if i == 436u32 { v = 0x08080808u32; }
        if i == 437u32 { v = 0x19081908u32; }
        if i == 438u32 { v = 0x2b2b1919u32; }
        if i == 439u32 { v = 0x08192b08u32; }
        if i == 440u32 { v = 0x192b2b2bu32; }
        if i == 441u32 { v = 0x08080808u32; }
        if i == 442u32 { v = 0x08082b08u32; }
        if i == 443u32 { v = 0x082b1919u32; }
        if i == 444u32 { v = 0x19192b2bu32; }
        if i == 445u32 { v = 0x2b080808u32; }
        if i == 446u32 { v = 0x2b08082bu32; }
        if i == 447u32 { v = 0x2b2b2b08u32; }
        if i == 448u32 { v = 0x0808192bu32; }
        if i == 449u32 { v = 0x082b082bu32; }
        if i == 450u32 { v = 0x2b080808u32; }
        if i == 451u32 { v = 0x2b082b08u32; }
        if i == 452u32 { v = 0x2b19192bu32; }
        if i == 453u32 { v = 0x2b2b2b08u32; }
        if i == 454u32 { v = 0x08080819u32; }
        if i == 455u32 { v = 0x08081908u32; }
        if i == 456u32 { v = 0x08190808u32; }
        if i == 457u32 { v = 0x19080808u32; }
        if i == 458u32 { v = 0x1919192bu32; }
        if i == 459u32 { v = 0x2b081908u32; }
        if i == 460u32 { v = 0x08080808u32; }
        if i == 461u32 { v = 0x082b082bu32; }
        if i == 462u32 { v = 0x192b1908u32; }
        if i == 463u32 { v = 0x1919192bu32; }
        if i == 464u32 { v = 0x2b082b19u32; }
        if i == 465u32 { v = 0x08080808u32; }
        if i == 466u32 { v = 0x08081919u32; }
        if i == 467u32 { v = 0x19081908u32; }
        if i == 468u32 { v = 0x19190808u32; }
        if i == 469u32 { v = 0x19192b08u32; }
        if i == 470u32 { v = 0x082b2b19u32; }
        if i == 471u32 { v = 0x2b190808u32; }
        if i == 472u32 { v = 0x2b19082bu32; }
        if i == 473u32 { v = 0x19080819u32; }
        if i == 474u32 { v = 0x19190819u32; }
        if i == 475u32 { v = 0x2b2b192bu32; }
        if i == 476u32 { v = 0x19082b19u32; }
        if i == 477u32 { v = 0x08191919u32; }
        if i == 478u32 { v = 0x192b0808u32; }
        if i == 479u32 { v = 0x08080808u32; }
        if i == 480u32 { v = 0x0808082bu32; }
        if i == 481u32 { v = 0x08082b08u32; }
        if i == 482u32 { v = 0x08082b2bu32; }
        if i == 483u32 { v = 0x082b0808u32; }
        if i == 484u32 { v = 0x082b2b2bu32; }
        if i == 485u32 { v = 0x2b2b0808u32; }
        if i == 486u32 { v = 0x19190819u32; }
        if i == 487u32 { v = 0x19192b19u32; }
        if i == 488u32 { v = 0x2b2b192bu32; }
        if i == 489u32 { v = 0x08080808u32; }
        if i == 490u32 { v = 0x0808082bu32; }
        if i == 491u32 { v = 0x08082b08u32; }
        if i == 492u32 { v = 0x082b2b2bu32; }
        if i == 493u32 { v = 0x2b080808u32; }
        if i == 494u32 { v = 0x2b2b0808u32; }
        if i == 495u32 { v = 0x19080808u32; }
        if i == 496u32 { v = 0x2b191919u32; }
        if i == 497u32 { v = 0x192b1919u32; }
        if i == 498u32 { v = 0x2b192b08u32; }
        if i == 499u32 { v = 0x08082b2bu32; }
        if i == 500u32 { v = 0x082b0808u32; }
        if i == 501u32 { v = 0x082b082bu32; }
        if i == 502u32 { v = 0x082b2b08u32; }
        if i == 503u32 { v = 0x2b2b0808u32; }
        if i == 504u32 { v = 0x2b2b2b08u32; }
        if i == 505u32 { v = 0x08081908u32; }
        if i == 506u32 { v = 0x2b081908u32; }
        if i == 507u32 { v = 0x2b08192bu32; }
        if i == 508u32 { v = 0x082b2b08u32; }
        if i == 509u32 { v = 0x082b2b2bu32; }
        if i == 510u32 { v = 0x2b190819u32; }
        if i == 511u32 { v = 0x2b2b2b2bu32; }
        v
    }

pub fn xs_grid_hi(i: u32) -> u32 {
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
        if i == 31u32 { v = 0x08080819u32; }
        if i == 32u32 { v = 0x08080819u32; }
        if i == 33u32 { v = 0x08080819u32; }
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
        if i == 53u32 { v = 0x0808082bu32; }
        if i == 54u32 { v = 0x0808082bu32; }
        if i == 55u32 { v = 0x0808082bu32; }
        if i == 56u32 { v = 0x0808082bu32; }
        if i == 57u32 { v = 0x0808082bu32; }
        if i == 58u32 { v = 0x0808082bu32; }
        if i == 59u32 { v = 0x0808082bu32; }
        if i == 60u32 { v = 0x0808082bu32; }
        if i == 61u32 { v = 0x0808082bu32; }
        if i == 62u32 { v = 0x0808082bu32; }
        if i == 63u32 { v = 0x0808082bu32; }
        if i == 64u32 { v = 0x0808082bu32; }
        if i == 65u32 { v = 0x0808082bu32; }
        if i == 66u32 { v = 0x08081908u32; }
        if i == 67u32 { v = 0x08081908u32; }
        if i == 68u32 { v = 0x08081908u32; }
        if i == 69u32 { v = 0x08081908u32; }
        if i == 70u32 { v = 0x08081908u32; }
        if i == 71u32 { v = 0x08081908u32; }
        if i == 72u32 { v = 0x08081908u32; }
        if i == 73u32 { v = 0x08081908u32; }
        if i == 74u32 { v = 0x08081908u32; }
        if i == 75u32 { v = 0x08081908u32; }
        if i == 76u32 { v = 0x08081908u32; }
        if i == 77u32 { v = 0x08081908u32; }
        if i == 78u32 { v = 0x08081908u32; }
        if i == 79u32 { v = 0x08081908u32; }
        if i == 80u32 { v = 0x08081908u32; }
        if i == 81u32 { v = 0x08081908u32; }
        if i == 82u32 { v = 0x08081908u32; }
        if i == 83u32 { v = 0x08081908u32; }
        if i == 84u32 { v = 0x08081908u32; }
        if i == 85u32 { v = 0x08081908u32; }
        if i == 86u32 { v = 0x08081908u32; }
        if i == 87u32 { v = 0x08081919u32; }
        if i == 88u32 { v = 0x08081919u32; }
        if i == 89u32 { v = 0x08081919u32; }
        if i == 90u32 { v = 0x08081919u32; }
        if i == 91u32 { v = 0x08081919u32; }
        if i == 92u32 { v = 0x08081919u32; }
        if i == 93u32 { v = 0x08081919u32; }
        if i == 94u32 { v = 0x08081919u32; }
        if i == 95u32 { v = 0x08081919u32; }
        if i == 96u32 { v = 0x08081919u32; }
        if i == 97u32 { v = 0x08081919u32; }
        if i == 98u32 { v = 0x08081919u32; }
        if i == 99u32 { v = 0x0808192bu32; }
        if i == 100u32 { v = 0x0808192bu32; }
        if i == 101u32 { v = 0x0808192bu32; }
        if i == 102u32 { v = 0x0808192bu32; }
        if i == 103u32 { v = 0x0808192bu32; }
        if i == 104u32 { v = 0x0808192bu32; }
        if i == 105u32 { v = 0x0808192bu32; }
        if i == 106u32 { v = 0x08082b08u32; }
        if i == 107u32 { v = 0x08082b08u32; }
        if i == 108u32 { v = 0x08082b08u32; }
        if i == 109u32 { v = 0x08082b08u32; }
        if i == 110u32 { v = 0x08082b08u32; }
        if i == 111u32 { v = 0x08082b08u32; }
        if i == 112u32 { v = 0x08082b08u32; }
        if i == 113u32 { v = 0x08082b08u32; }
        if i == 114u32 { v = 0x08082b08u32; }
        if i == 115u32 { v = 0x08082b08u32; }
        if i == 116u32 { v = 0x08082b08u32; }
        if i == 117u32 { v = 0x08082b08u32; }
        if i == 118u32 { v = 0x08082b08u32; }
        if i == 119u32 { v = 0x08082b08u32; }
        if i == 120u32 { v = 0x08082b08u32; }
        if i == 121u32 { v = 0x08082b08u32; }
        if i == 122u32 { v = 0x08082b19u32; }
        if i == 123u32 { v = 0x08082b19u32; }
        if i == 124u32 { v = 0x08082b19u32; }
        if i == 125u32 { v = 0x08082b19u32; }
        if i == 126u32 { v = 0x08082b19u32; }
        if i == 127u32 { v = 0x08082b19u32; }
        if i == 128u32 { v = 0x08082b2bu32; }
        if i == 129u32 { v = 0x08082b2bu32; }
        if i == 130u32 { v = 0x08082b2bu32; }
        if i == 131u32 { v = 0x08082b2bu32; }
        if i == 132u32 { v = 0x08082b2bu32; }
        if i == 133u32 { v = 0x08190808u32; }
        if i == 134u32 { v = 0x08190808u32; }
        if i == 135u32 { v = 0x08190808u32; }
        if i == 136u32 { v = 0x08190808u32; }
        if i == 137u32 { v = 0x08190808u32; }
        if i == 138u32 { v = 0x08190808u32; }
        if i == 139u32 { v = 0x08190808u32; }
        if i == 140u32 { v = 0x08190808u32; }
        if i == 141u32 { v = 0x08190808u32; }
        if i == 142u32 { v = 0x08190808u32; }
        if i == 143u32 { v = 0x08190808u32; }
        if i == 144u32 { v = 0x08190808u32; }
        if i == 145u32 { v = 0x08190808u32; }
        if i == 146u32 { v = 0x08190808u32; }
        if i == 147u32 { v = 0x08190808u32; }
        if i == 148u32 { v = 0x08190808u32; }
        if i == 149u32 { v = 0x08190808u32; }
        if i == 150u32 { v = 0x08190808u32; }
        if i == 151u32 { v = 0x08190808u32; }
        if i == 152u32 { v = 0x08190808u32; }
        if i == 153u32 { v = 0x08190808u32; }
        if i == 154u32 { v = 0x08190819u32; }
        if i == 155u32 { v = 0x08190819u32; }
        if i == 156u32 { v = 0x08190819u32; }
        if i == 157u32 { v = 0x08190819u32; }
        if i == 158u32 { v = 0x08190819u32; }
        if i == 159u32 { v = 0x08190819u32; }
        if i == 160u32 { v = 0x08190819u32; }
        if i == 161u32 { v = 0x08190819u32; }
        if i == 162u32 { v = 0x08190819u32; }
        if i == 163u32 { v = 0x08190819u32; }
        if i == 164u32 { v = 0x08190819u32; }
        if i == 165u32 { v = 0x08190819u32; }
        if i == 166u32 { v = 0x08190819u32; }
        if i == 167u32 { v = 0x0819082bu32; }
        if i == 168u32 { v = 0x0819082bu32; }
        if i == 169u32 { v = 0x0819082bu32; }
        if i == 170u32 { v = 0x0819082bu32; }
        if i == 171u32 { v = 0x0819082bu32; }
        if i == 172u32 { v = 0x0819082bu32; }
        if i == 173u32 { v = 0x08191908u32; }
        if i == 174u32 { v = 0x08191908u32; }
        if i == 175u32 { v = 0x08191908u32; }
        if i == 176u32 { v = 0x08191908u32; }
        if i == 177u32 { v = 0x08191908u32; }
        if i == 178u32 { v = 0x08191908u32; }
        if i == 179u32 { v = 0x08191908u32; }
        if i == 180u32 { v = 0x08191908u32; }
        if i == 181u32 { v = 0x08191908u32; }
        if i == 182u32 { v = 0x08191908u32; }
        if i == 183u32 { v = 0x08191908u32; }
        if i == 184u32 { v = 0x08191908u32; }
        if i == 185u32 { v = 0x08191908u32; }
        if i == 186u32 { v = 0x08191919u32; }
        if i == 187u32 { v = 0x08191919u32; }
        if i == 188u32 { v = 0x08191919u32; }
        if i == 189u32 { v = 0x08191919u32; }
        if i == 190u32 { v = 0x0819192bu32; }
        if i == 191u32 { v = 0x0819192bu32; }
        if i == 192u32 { v = 0x0819192bu32; }
        if i == 193u32 { v = 0x08192b08u32; }
        if i == 194u32 { v = 0x08192b08u32; }
        if i == 195u32 { v = 0x08192b08u32; }
        if i == 196u32 { v = 0x08192b08u32; }
        if i == 197u32 { v = 0x08192b08u32; }
        if i == 198u32 { v = 0x08192b08u32; }
        if i == 199u32 { v = 0x08192b08u32; }
        if i == 200u32 { v = 0x08192b19u32; }
        if i == 201u32 { v = 0x08192b19u32; }
        if i == 202u32 { v = 0x08192b19u32; }
        if i == 203u32 { v = 0x08192b2bu32; }
        if i == 204u32 { v = 0x08192b2bu32; }
        if i == 205u32 { v = 0x082b0808u32; }
        if i == 206u32 { v = 0x082b0808u32; }
        if i == 207u32 { v = 0x082b0808u32; }
        if i == 208u32 { v = 0x082b0808u32; }
        if i == 209u32 { v = 0x082b0808u32; }
        if i == 210u32 { v = 0x082b0808u32; }
        if i == 211u32 { v = 0x082b0808u32; }
        if i == 212u32 { v = 0x082b0808u32; }
        if i == 213u32 { v = 0x082b0808u32; }
        if i == 214u32 { v = 0x082b0808u32; }
        if i == 215u32 { v = 0x082b0808u32; }
        if i == 216u32 { v = 0x082b0808u32; }
        if i == 217u32 { v = 0x082b0808u32; }
        if i == 218u32 { v = 0x082b0819u32; }
        if i == 219u32 { v = 0x082b0819u32; }
        if i == 220u32 { v = 0x082b0819u32; }
        if i == 221u32 { v = 0x082b0819u32; }
        if i == 222u32 { v = 0x082b0819u32; }
        if i == 223u32 { v = 0x082b0819u32; }
        if i == 224u32 { v = 0x082b082bu32; }
        if i == 225u32 { v = 0x082b082bu32; }
        if i == 226u32 { v = 0x082b082bu32; }
        if i == 227u32 { v = 0x082b082bu32; }
        if i == 228u32 { v = 0x082b1908u32; }
        if i == 229u32 { v = 0x082b1908u32; }
        if i == 230u32 { v = 0x082b1908u32; }
        if i == 231u32 { v = 0x082b1908u32; }
        if i == 232u32 { v = 0x082b1908u32; }
        if i == 233u32 { v = 0x082b1919u32; }
        if i == 234u32 { v = 0x082b1919u32; }
        if i == 235u32 { v = 0x082b1919u32; }
        if i == 236u32 { v = 0x082b1919u32; }
        if i == 237u32 { v = 0x082b192bu32; }
        if i == 238u32 { v = 0x082b192bu32; }
        if i == 239u32 { v = 0x082b192bu32; }
        if i == 240u32 { v = 0x082b2b08u32; }
        if i == 241u32 { v = 0x082b2b08u32; }
        if i == 242u32 { v = 0x082b2b08u32; }
        if i == 243u32 { v = 0x082b2b08u32; }
        if i == 244u32 { v = 0x082b2b08u32; }
        if i == 245u32 { v = 0x082b2b08u32; }
        if i == 246u32 { v = 0x082b2b08u32; }
        if i == 247u32 { v = 0x082b2b19u32; }
        if i == 248u32 { v = 0x082b2b19u32; }
        if i == 249u32 { v = 0x082b2b2bu32; }
        if i == 250u32 { v = 0x082b2b2bu32; }
        if i == 251u32 { v = 0x082b2b2bu32; }
        if i == 252u32 { v = 0x082b2b2bu32; }
        if i == 253u32 { v = 0x082b2b2bu32; }
        if i == 254u32 { v = 0x19080808u32; }
        if i == 255u32 { v = 0x19080808u32; }
        if i == 256u32 { v = 0x19080808u32; }
        if i == 257u32 { v = 0x19080808u32; }
        if i == 258u32 { v = 0x19080808u32; }
        if i == 259u32 { v = 0x19080808u32; }
        if i == 260u32 { v = 0x19080808u32; }
        if i == 261u32 { v = 0x19080808u32; }
        if i == 262u32 { v = 0x19080808u32; }
        if i == 263u32 { v = 0x19080808u32; }
        if i == 264u32 { v = 0x19080808u32; }
        if i == 265u32 { v = 0x19080808u32; }
        if i == 266u32 { v = 0x19080808u32; }
        if i == 267u32 { v = 0x19080808u32; }
        if i == 268u32 { v = 0x19080808u32; }
        if i == 269u32 { v = 0x19080808u32; }
        if i == 270u32 { v = 0x19080808u32; }
        if i == 271u32 { v = 0x19080808u32; }
        if i == 272u32 { v = 0x19080808u32; }
        if i == 273u32 { v = 0x19080808u32; }
        if i == 274u32 { v = 0x19080808u32; }
        if i == 275u32 { v = 0x19080808u32; }
        if i == 276u32 { v = 0x19080819u32; }
        if i == 277u32 { v = 0x19080819u32; }
        if i == 278u32 { v = 0x19080819u32; }
        if i == 279u32 { v = 0x19080819u32; }
        if i == 280u32 { v = 0x19080819u32; }
        if i == 281u32 { v = 0x19080819u32; }
        if i == 282u32 { v = 0x19080819u32; }
        if i == 283u32 { v = 0x19080819u32; }
        if i == 284u32 { v = 0x19080819u32; }
        if i == 285u32 { v = 0x19080819u32; }
        if i == 286u32 { v = 0x19080819u32; }
        if i == 287u32 { v = 0x19080819u32; }
        if i == 288u32 { v = 0x19080819u32; }
        if i == 289u32 { v = 0x1908082bu32; }
        if i == 290u32 { v = 0x1908082bu32; }
        if i == 291u32 { v = 0x1908082bu32; }
        if i == 292u32 { v = 0x1908082bu32; }
        if i == 293u32 { v = 0x1908082bu32; }
        if i == 294u32 { v = 0x1908082bu32; }
        if i == 295u32 { v = 0x19081908u32; }
        if i == 296u32 { v = 0x19081908u32; }
        if i == 297u32 { v = 0x19081908u32; }
        if i == 298u32 { v = 0x19081908u32; }
        if i == 299u32 { v = 0x19081908u32; }
        if i == 300u32 { v = 0x19081908u32; }
        if i == 301u32 { v = 0x19081908u32; }
        if i == 302u32 { v = 0x19081908u32; }
        if i == 303u32 { v = 0x19081908u32; }
        if i == 304u32 { v = 0x19081908u32; }
        if i == 305u32 { v = 0x19081908u32; }
        if i == 306u32 { v = 0x19081908u32; }
        if i == 307u32 { v = 0x19081908u32; }
        if i == 308u32 { v = 0x19081919u32; }
        if i == 309u32 { v = 0x19081919u32; }
        if i == 310u32 { v = 0x19081919u32; }
        if i == 311u32 { v = 0x19081919u32; }
        if i == 312u32 { v = 0x19081919u32; }
        if i == 313u32 { v = 0x19081919u32; }
        if i == 314u32 { v = 0x1908192bu32; }
        if i == 315u32 { v = 0x1908192bu32; }
        if i == 316u32 { v = 0x1908192bu32; }
        if i == 317u32 { v = 0x1908192bu32; }
        if i == 318u32 { v = 0x19082b08u32; }
        if i == 319u32 { v = 0x19082b08u32; }
        if i == 320u32 { v = 0x19082b08u32; }
        if i == 321u32 { v = 0x19082b08u32; }
        if i == 322u32 { v = 0x19082b08u32; }
        if i == 323u32 { v = 0x19082b08u32; }
        if i == 324u32 { v = 0x19082b08u32; }
        if i == 325u32 { v = 0x19082b19u32; }
        if i == 326u32 { v = 0x19082b19u32; }
        if i == 327u32 { v = 0x19082b19u32; }
        if i == 328u32 { v = 0x19082b19u32; }
        if i == 329u32 { v = 0x19082b19u32; }
        if i == 330u32 { v = 0x19082b2bu32; }
        if i == 331u32 { v = 0x19190808u32; }
        if i == 332u32 { v = 0x19190808u32; }
        if i == 333u32 { v = 0x19190808u32; }
        if i == 334u32 { v = 0x19190808u32; }
        if i == 335u32 { v = 0x19190808u32; }
        if i == 336u32 { v = 0x19190808u32; }
        if i == 337u32 { v = 0x19190808u32; }
        if i == 338u32 { v = 0x19190808u32; }
        if i == 339u32 { v = 0x19190808u32; }
        if i == 340u32 { v = 0x19190808u32; }
        if i == 341u32 { v = 0x19190808u32; }
        if i == 342u32 { v = 0x19190808u32; }
        if i == 343u32 { v = 0x19190819u32; }
        if i == 344u32 { v = 0x19190819u32; }
        if i == 345u32 { v = 0x19190819u32; }
        if i == 346u32 { v = 0x19190819u32; }
        if i == 347u32 { v = 0x19190819u32; }
        if i == 348u32 { v = 0x19190819u32; }
        if i == 349u32 { v = 0x1919082bu32; }
        if i == 350u32 { v = 0x1919082bu32; }
        if i == 351u32 { v = 0x1919082bu32; }
        if i == 352u32 { v = 0x19191908u32; }
        if i == 353u32 { v = 0x19191908u32; }
        if i == 354u32 { v = 0x19191908u32; }
        if i == 355u32 { v = 0x19191908u32; }
        if i == 356u32 { v = 0x19191908u32; }
        if i == 357u32 { v = 0x19191908u32; }
        if i == 358u32 { v = 0x19191908u32; }
        if i == 359u32 { v = 0x19191908u32; }
        if i == 360u32 { v = 0x19191919u32; }
        if i == 361u32 { v = 0x19191919u32; }
        if i == 362u32 { v = 0x19191919u32; }
        if i == 363u32 { v = 0x19191919u32; }
        if i == 364u32 { v = 0x1919192bu32; }
        if i == 365u32 { v = 0x1919192bu32; }
        if i == 366u32 { v = 0x1919192bu32; }
        if i == 367u32 { v = 0x19192b08u32; }
        if i == 368u32 { v = 0x19192b08u32; }
        if i == 369u32 { v = 0x19192b08u32; }
        if i == 370u32 { v = 0x19192b08u32; }
        if i == 371u32 { v = 0x19192b08u32; }
        if i == 372u32 { v = 0x19192b19u32; }
        if i == 373u32 { v = 0x19192b19u32; }
        if i == 374u32 { v = 0x19192b19u32; }
        if i == 375u32 { v = 0x19192b2bu32; }
        if i == 376u32 { v = 0x192b0808u32; }
        if i == 377u32 { v = 0x192b0808u32; }
        if i == 378u32 { v = 0x192b0808u32; }
        if i == 379u32 { v = 0x192b0808u32; }
        if i == 380u32 { v = 0x192b0808u32; }
        if i == 381u32 { v = 0x192b0808u32; }
        if i == 382u32 { v = 0x192b0808u32; }
        if i == 383u32 { v = 0x192b0808u32; }
        if i == 384u32 { v = 0x192b0819u32; }
        if i == 385u32 { v = 0x192b082bu32; }
        if i == 386u32 { v = 0x192b082bu32; }
        if i == 387u32 { v = 0x192b082bu32; }
        if i == 388u32 { v = 0x192b1908u32; }
        if i == 389u32 { v = 0x192b1908u32; }
        if i == 390u32 { v = 0x192b1919u32; }
        if i == 391u32 { v = 0x192b1919u32; }
        if i == 392u32 { v = 0x192b1919u32; }
        if i == 393u32 { v = 0x192b1919u32; }
        if i == 394u32 { v = 0x192b2b08u32; }
        if i == 395u32 { v = 0x192b2b08u32; }
        if i == 396u32 { v = 0x192b2b19u32; }
        if i == 397u32 { v = 0x192b2b2bu32; }
        if i == 398u32 { v = 0x192b2b2bu32; }
        if i == 399u32 { v = 0x192b2b2bu32; }
        if i == 400u32 { v = 0x2b080808u32; }
        if i == 401u32 { v = 0x2b080808u32; }
        if i == 402u32 { v = 0x2b080808u32; }
        if i == 403u32 { v = 0x2b080808u32; }
        if i == 404u32 { v = 0x2b080808u32; }
        if i == 405u32 { v = 0x2b080808u32; }
        if i == 406u32 { v = 0x2b080808u32; }
        if i == 407u32 { v = 0x2b080808u32; }
        if i == 408u32 { v = 0x2b080808u32; }
        if i == 409u32 { v = 0x2b080808u32; }
        if i == 410u32 { v = 0x2b080808u32; }
        if i == 411u32 { v = 0x2b080808u32; }
        if i == 412u32 { v = 0x2b080808u32; }
        if i == 413u32 { v = 0x2b080808u32; }
        if i == 414u32 { v = 0x2b080808u32; }
        if i == 415u32 { v = 0x2b080819u32; }
        if i == 416u32 { v = 0x2b080819u32; }
        if i == 417u32 { v = 0x2b080819u32; }
        if i == 418u32 { v = 0x2b080819u32; }
        if i == 419u32 { v = 0x2b080819u32; }
        if i == 420u32 { v = 0x2b080819u32; }
        if i == 421u32 { v = 0x2b080819u32; }
        if i == 422u32 { v = 0x2b08082bu32; }
        if i == 423u32 { v = 0x2b08082bu32; }
        if i == 424u32 { v = 0x2b08082bu32; }
        if i == 425u32 { v = 0x2b08082bu32; }
        if i == 426u32 { v = 0x2b08082bu32; }
        if i == 427u32 { v = 0x2b08082bu32; }
        if i == 428u32 { v = 0x2b081908u32; }
        if i == 429u32 { v = 0x2b081908u32; }
        if i == 430u32 { v = 0x2b081908u32; }
        if i == 431u32 { v = 0x2b081908u32; }
        if i == 432u32 { v = 0x2b081908u32; }
        if i == 433u32 { v = 0x2b081908u32; }
        if i == 434u32 { v = 0x2b081908u32; }
        if i == 435u32 { v = 0x2b081908u32; }
        if i == 436u32 { v = 0x2b081919u32; }
        if i == 437u32 { v = 0x2b081919u32; }
        if i == 438u32 { v = 0x2b081919u32; }
        if i == 439u32 { v = 0x2b08192bu32; }
        if i == 440u32 { v = 0x2b08192bu32; }
        if i == 441u32 { v = 0x2b082b08u32; }
        if i == 442u32 { v = 0x2b082b08u32; }
        if i == 443u32 { v = 0x2b082b08u32; }
        if i == 444u32 { v = 0x2b082b08u32; }
        if i == 445u32 { v = 0x2b082b08u32; }
        if i == 446u32 { v = 0x2b082b08u32; }
        if i == 447u32 { v = 0x2b082b08u32; }
        if i == 448u32 { v = 0x2b082b19u32; }
        if i == 449u32 { v = 0x2b082b2bu32; }
        if i == 450u32 { v = 0x2b082b2bu32; }
        if i == 451u32 { v = 0x2b082b2bu32; }
        if i == 452u32 { v = 0x2b082b2bu32; }
        if i == 453u32 { v = 0x2b082b2bu32; }
        if i == 454u32 { v = 0x2b190808u32; }
        if i == 455u32 { v = 0x2b190808u32; }
        if i == 456u32 { v = 0x2b190808u32; }
        if i == 457u32 { v = 0x2b190808u32; }
        if i == 458u32 { v = 0x2b190808u32; }
        if i == 459u32 { v = 0x2b190808u32; }
        if i == 460u32 { v = 0x2b190819u32; }
        if i == 461u32 { v = 0x2b190819u32; }
        if i == 462u32 { v = 0x2b190819u32; }
        if i == 463u32 { v = 0x2b19082bu32; }
        if i == 464u32 { v = 0x2b19082bu32; }
        if i == 465u32 { v = 0x2b191908u32; }
        if i == 466u32 { v = 0x2b191908u32; }
        if i == 467u32 { v = 0x2b191908u32; }
        if i == 468u32 { v = 0x2b191908u32; }
        if i == 469u32 { v = 0x2b191908u32; }
        if i == 470u32 { v = 0x2b191919u32; }
        if i == 471u32 { v = 0x2b191919u32; }
        if i == 472u32 { v = 0x2b191919u32; }
        if i == 473u32 { v = 0x2b19192bu32; }
        if i == 474u32 { v = 0x2b192b08u32; }
        if i == 475u32 { v = 0x2b192b08u32; }
        if i == 476u32 { v = 0x2b192b19u32; }
        if i == 477u32 { v = 0x2b192b2bu32; }
        if i == 478u32 { v = 0x2b192b2bu32; }
        if i == 479u32 { v = 0x2b2b0808u32; }
        if i == 480u32 { v = 0x2b2b0808u32; }
        if i == 481u32 { v = 0x2b2b0808u32; }
        if i == 482u32 { v = 0x2b2b0808u32; }
        if i == 483u32 { v = 0x2b2b0808u32; }
        if i == 484u32 { v = 0x2b2b0808u32; }
        if i == 485u32 { v = 0x2b2b0808u32; }
        if i == 486u32 { v = 0x2b2b0819u32; }
        if i == 487u32 { v = 0x2b2b0819u32; }
        if i == 488u32 { v = 0x2b2b0819u32; }
        if i == 489u32 { v = 0x2b2b082bu32; }
        if i == 490u32 { v = 0x2b2b082bu32; }
        if i == 491u32 { v = 0x2b2b082bu32; }
        if i == 492u32 { v = 0x2b2b082bu32; }
        if i == 493u32 { v = 0x2b2b082bu32; }
        if i == 494u32 { v = 0x2b2b082bu32; }
        if i == 495u32 { v = 0x2b2b1908u32; }
        if i == 496u32 { v = 0x2b2b1908u32; }
        if i == 497u32 { v = 0x2b2b192bu32; }
        if i == 498u32 { v = 0x2b2b192bu32; }
        if i == 499u32 { v = 0x2b2b2b08u32; }
        if i == 500u32 { v = 0x2b2b2b08u32; }
        if i == 501u32 { v = 0x2b2b2b08u32; }
        if i == 502u32 { v = 0x2b2b2b08u32; }
        if i == 503u32 { v = 0x2b2b2b08u32; }
        if i == 504u32 { v = 0x2b2b2b08u32; }
        if i == 505u32 { v = 0x2b2b2b19u32; }
        if i == 506u32 { v = 0x2b2b2b19u32; }
        if i == 507u32 { v = 0x2b2b2b19u32; }
        if i == 508u32 { v = 0x2b2b2b2bu32; }
        if i == 509u32 { v = 0x2b2b2b2bu32; }
        if i == 510u32 { v = 0x2b2b2b2bu32; }
        if i == 511u32 { v = 0x2b2b2b2bu32; }
        v
    }

    pub unsafe fn mmvq_acc<const FMT: u32>(tmp: f32, xb: *const u8, yb: *const u8, iqs: usize) -> f32 {
        match FMT {
            18 => {
                // vec_dot_iq3_xxs_q8_1; kqs = 4*(tid % (qi/vdr)) => q8 block = kby + iqs/vdr.
                let qs = xb.add(2);
                let q3a = get_b4(qs, iqs as i32);
                let q3b = get_b4(qs, iqs as i32 + 1);
                let mut q3 = [0u8; 8];
                q3[0] = q3a as u8; q3[1] = (q3a >> 8) as u8; q3[2] = (q3a >> 16) as u8; q3[3] = (q3a >> 24) as u8;
                q3[4] = q3b as u8; q3[5] = (q3b >> 8) as u8; q3[6] = (q3b >> 16) as u8; q3[7] = (q3b >> 24) as u8;
                let aux32 = get_b4(qs.add(64), iqs as i32 / 2);
                let b8 = yb.add((iqs / 2) * 36);
                let q8 = b8.add(4) as *const u32;
                let mut sumi = 0i32;
                let mut l0 = 0usize;
                while l0 < 8 {
                    let g0 = i3_grid(q3[l0] as u32);
                    let g1 = i3_grid(q3[l0 + 1] as u32);
                    let signs = unpack_ksigns(aux32 >> (7 * l0 as u32 / 2));
                    // unpack_ksigns already broadcasts the byte across the word (s * 0x01010101);
                    // multiplying again here squares the byte and corrupts the sign selectors.
                    let s0 = vcmpne4_zero(signs & 0x0804_0201);
                    let g_l = vsub4_wrap(g0 ^ s0, s0);
                    let s1 = vcmpne4_zero(signs & 0x8040_2010);
                    let g_h = vsub4_wrap(g1 ^ s1, s1);
                    sumi = dp4a(g_l, *q8.add(l0), sumi);
                    sumi = dp4a(g_h, *q8.add(l0 + 1), sumi);
                    l0 += 2;
                }
                let ls = (aux32 >> 28) as i32;
                let sumi = (ls.wrapping_mul(sumi).wrapping_add(sumi / 2)) / 2;
                let d = mul(h2f(ld16(xb, 0) as u16), h2f(ld16(b8, 0) as u16));
                fma(d, sumi as f32, tmp)
            }
            23 => {
                // vec_dot_iq4_xs_q8_1; kqs = 4*(tid % (qi/vdr)) => q8 block = kby + iqs/vdr.
                let qs = xb.add(8) as *const u32;
                let b8 = yb.add((iqs / 4) * 36);
                let q8 = b8.add(4) as *const u32;
                let mut sumi = 0i32;
                let mut j = 0usize;
                while j < 4 {
                    let (v0, v1) = i4_table16(get_b4(xb, iqs as i32 + j as i32 + 2));
                    sumi = dp4a(v0, *q8.add(j), sumi);
                    sumi = dp4a(v1, *q8.add(j + 4), sumi);
                    j += 1;
                }
                let ls = i4_group_scale6(xb, iqs / 4) as i32 - 32;
                let sumi = sumi.wrapping_mul(ls);
                let d = mul(h2f(ld16(xb, 0) as u16), h2f(ld16(b8, 0) as u16));
                fma(d, sumi as f32, tmp)
            }

            17 => {
                // vec_dot_iq2_xs_q8_1; iqs is 2*(tid % 8) => q8 block iqs/2.
                let qs = xb.add(2);
                let q16 = qs.add(4 * iqs) as *const u16;
                let ls0 = (*xb.add(66 + (iqs / 2)) & 0x0F) as i32;
                let ls1 = (*xb.add(66 + (iqs / 2)) >> 4) as i32;
                let b8 = yb.add((iqs / 2) * 36);
                let q8 = b8.add(4) as *const u32;
                let mut sumi0 = 0i32;
                let mut sumi1 = 0i32;
                let mut l0 = 0usize;
                while l0 < 8 {
                    let w = *q16.add(l0 / 2) as u32;
                    let gl = xs_grid_lo(w & 0x1FF);
                    let gh = xs_grid_hi(w & 0x1FF);
                    let signs = unpack_ksigns(w >> 9);
                    let s0 = vcmpne4_zero(signs & 0x0804_0201);
                    let s1 = vcmpne4_zero(signs & 0x8040_2010);
                    let g0 = vsub4_wrap(gl ^ s0, s0);
                    let g1 = vsub4_wrap(gh ^ s1, s1);
                    if l0 < 4 {
                        sumi0 = dp4a(g0, *q8.add(l0), sumi0);
                        sumi0 = dp4a(g1, *q8.add(l0 + 1), sumi0);
                    } else {
                        sumi1 = dp4a(g0, *q8.add(l0), sumi1);
                        sumi1 = dp4a(g1, *q8.add(l0 + 1), sumi1);
                    }
                    l0 += 2;
                }
                let sumi = sumi0.wrapping_mul(ls0).wrapping_add(sumi1.wrapping_mul(ls1)).wrapping_add((sumi0 + sumi1) / 2) / 4;
                let d = mul(h2f(ld16(xb, 0) as u16), h2f(ld16(b8, 0) as u16));
                fma(d, sumi as f32, tmp)
            }
            22 => {
                // vec_dot_iq2_s_q8_1; iqs is 2*(tid % 8) => q8 block iqs/2.
                let i2 = iqs as i32 / 2;
                let qs = xb.add(2);
                let qp = get_b2(qs, i2);
                let sp = get_b2(qs.add(32), i2);
                let mut q8b = [0u8; 4];
                let mut s8 = [0u8; 4];
                let mut t = 0usize;
                while t < 4 { q8b[t] = (qp >> (8 * t)) as u8; s8[t] = (sp >> (8 * t)) as u8; t += 1; }
                let qh = *xb.add(66 + i2 as usize) as u32;
                let ls0 = (*xb.add(74 + i2 as usize) & 0x0F) as i32;
                let ls1 = (*xb.add(74 + i2 as usize) >> 4) as i32;
                let b8 = yb.add((iqs / 2) * 36);
                let q8 = b8.add(4) as *const u32;
                let mut sumi0 = 0i32;
                let mut sumi1 = 0i32;
                let mut l0 = 0usize;
                while l0 < 8 {
                    let gidx = q8b[l0 / 2] as u32 | ((qh << (8 - l0 as u32)) & 0x300);
                    let gl = s_grid_lo(gidx);
                    let gh = s_grid_hi(gidx);
                    let sb = s8[l0 / 2] as u32;
                    let s0 = vcmpne4_zero(((sb & 0x03) << 7) | ((sb & 0x0C) << 21));
                    let s1 = vcmpne4_zero(((sb & 0x30) << 3) | ((sb & 0xC0) << 17));
                    let g_l = vsub4_wrap(gl ^ s0, s0);
                    let g_h = vsub4_wrap(gh ^ s1, s1);
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
                let d = mul(h2f(ld16(xb, 0) as u16), h2f(ld16(b8, 0) as u16));
                fma(d, sumi as f32, tmp)
            }

            11 => {
                // vec_dot_iq2_xxs_q8_1; iqs is already 2*(tid % 8) => q8 word base iqs/2.
                // block_iq2_xxs starts with `half d`, so qs sits at byte 2: the reference
                // reads get_int_b2(bq2->qs, ...) = byte 2 + 4*k, not 4*k.
                let qs2 = xb.add(2);
                let q2 = get_b2(qs2, iqs as i32);
                let aux8 = q2.to_le_bytes();
                let aux32 = get_b2(qs2, iqs as i32 + 1);
                let mut sumi = 0i32;
                let mut k0 = 0usize;
                while k0 < 8 {
                    let idx = aux8[k0 / 2] as u32;
                    let gl = grid_lo(idx);
                    let gh = grid_hi(idx);
                    let signs = unpack_ksigns(aux32 >> (7 * k0 as u32 / 2));
                    let s0 = vcmpne4_zero(signs & 0x0804_0201);
                    let s1 = vcmpne4_zero(signs & 0x8040_2010);
                    let g0 = vsub4_wrap(gl ^ s0, s0);
                    let g1 = vsub4_wrap(gh ^ s1, s1);
                    let b8 = yb.add((iqs / 2) * 36);
                    sumi = dp4a(g0, yq(b8, k0), sumi);
                    sumi = dp4a(g1, yq(b8, k0 + 1), sumi);
                    k0 += 2;
                }
                let ls = (aux32 >> 27 | 1) as i32;
                let sumi2 = sumi.wrapping_mul(ls) / 8;
                let b8 = yb.add((iqs / 2) * 36);
                let d = mul(h2f(ld16(xb, 0) as u16), h2f(ld16(b8, 0) as u16));
                fma(d, sumi2 as f32, tmp)
            }

            0 | 1 | 2 | 3 => {
                // q4_0 / q4_1 / q5_0 / q5_1, vdr = 2, qi = 4.
                let mut sumi = 0i32;
                let mut i = 0;
                while i < 2 {
                    let u0 = yq(yb, iqs + i);
                    let u1 = yq(yb, iqs + i + 4);
                    let (vi0, vi1) = match FMT {
                        0 => {
                            let v = ld32(xb, 2 + 4 * (iqs + i));
                            (v & 0x0F0F0F0F, (v >> 4) & 0x0F0F0F0F)
                        }
                        1 => {
                            let v = *(xb.add(4 + 4 * (iqs + i)) as *const u32);
                            (v & 0x0F0F0F0F, (v >> 4) & 0x0F0F0F0F)
                        }
                        _ => {
                            let (vl, qh) = if FMT == 2 {
                                (ld32(xb, 6 + 4 * (iqs + i)), ld32(xb, 2))
                            } else {
                                (*(xb.add(8 + 4 * (iqs + i)) as *const u32), *(xb.add(4) as *const u32))
                            };
                            let vh = ((qh as i32) >> (4 * (iqs + i))) as u32;
                            let mut vi0 = vl & 0x0F0F0F0F;
                            vi0 |= (vh << 4) & 0x00000010;
                            vi0 |= (vh << 11) & 0x00001000;
                            vi0 |= (vh << 18) & 0x00100000;
                            vi0 |= (vh << 25) & 0x10000000;
                            let mut vi1 = (vl >> 4) & 0x0F0F0F0F;
                            vi1 |= (vh >> 12) & 0x00000010;
                            vi1 |= (vh >> 5) & 0x00001000;
                            vi1 |= (vh << 2) & 0x00100000;
                            vi1 |= (vh << 9) & 0x10000000;
                            (vi0, vi1)
                        }
                    };
                    sumi = dp4a(vi0, u0, sumi);
                    sumi = dp4a(vi1, u1, sumi);
                    i += 1;
                }
                let dsx = ldh(yb, 0);
                let dsy = ldh(yb, 2);
                let sf = sumi as f32;
                if FMT == 0 || FMT == 2 {
                    // d * (sumi * ds.x - (8|16 * vdr / qi) * ds.y)
                    let c = if FMT == 0 { 4.0 } else { 8.0 };
                    fma(ldh(xb, 0), fma(dsx, sf, -mul(dsy, c)), tmp)
                } else {
                    // sumi * (dm.x * ds.x) + (dm.y * ds.y) / 2
                    let dd = mul(ldh(xb, 0), dsx);
                    let ms = mul(ldh(xb, 2), dsy);
                    add(tmp, fma(dd, sf, mul(ms, 0.5)))
                }
            }
            4 => {
                // q8_0, vdr = 2: `d8_0*d8_1 * sumi` (d0*d1 first; the gate distinguishes the orders).
                let mut sumi = 0i32;
                let mut i = 0;
                while i < 2 {
                    sumi = dp4a(ld32(xb, 2 + 4 * (iqs + i)), yq(yb, iqs + i), sumi);
                    i += 1;
                }
                fma(mul(ldh(xb, 0), ldh(yb, 0)), sumi as f32, tmp)
            }
            5 => {
                // q2_K, vdr = 1.
                let bq8_offset = 4 * (iqs / 8);
                let scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
                let v = *(xb.add(16 + 4 * iqs) as *const u32);
                let mut sumf_d = 0f32;
                let mut sumf_m = 0f32;
                let mut i = 0;
                while i < 4 {
                    let b8 = yb.add((bq8_offset + i) * 36);
                    let u = yq(b8, iqs % 8);
                    let d8 = ldh(b8, 0);
                    let sc = ld8(xb, scale_offset + 2 * i);
                    let vi = (v >> (2 * i)) & 0x03030303;
                    sumf_d = fma(d8, (dp4a(vi, u, 0).wrapping_mul((sc & 0xF) as i32)) as f32, sumf_d);
                    let mut m = sc >> 4;
                    m |= m << 8;
                    m |= m << 16;
                    sumf_m = fma(d8, dp4a(m, u, 0) as f32, sumf_m);
                    i += 1;
                }
                add(tmp, fma(ldh(xb, 80), sumf_d, -mul(ldh(xb, 82), sumf_m)))
            }
            6 => {
                // q3_K, vdr = 1.
                let bq8_offset = 4 * (iqs / 8);
                let scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
                let vl = ld32(xb, 32 + 4 * iqs);
                let vh = ((!ld32(xb, 4 * (iqs % 8))) as i32 >> bq8_offset) as u32;
                let mut sumf = 0f32;
                let mut i = 0;
                while i < 4 {
                    let b8 = yb.add((bq8_offset + i) * 36);
                    let u = yq(b8, iqs % 8);
                    let d8 = ldh(b8, 0);
                    let isc = scale_offset + 2 * i;
                    let sc_low = (ld8(xb, 96 + isc % 8) >> (4 * (isc / 8))) & 0xF;
                    let sc_high = ((ld8(xb, 96 + 8 + isc % 4) >> (2 * (isc / 4))) & 3) << 4;
                    let sc = (sc_low | sc_high) as i32 - 32;
                    let vil = (vl >> (2 * i)) & 0x03030303;
                    let vih = ((vh >> i) << 2) & 0x04040404;
                    let vi = vsubss4(vil, vih);
                    sumf = fma(d8, (dp4a(vi, u, 0).wrapping_mul(sc)) as f32, sumf);
                    i += 1;
                }
                fma(ldh(xb, 108), sumf, tmp)
            }
            7 | 8 => {
                // q4_K / q5_K, vdr = 2, iqs even in 0..30.
                let bq8_offset = 2 * ((iqs / 2) / 4);
                let (v0, v1) = if FMT == 7 {
                    let q4 = xb.add(16 + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
                    let (a, b) = (*(q4 as *const u32), *(q4.add(16) as *const u32));
                    ([a & 0x0F0F0F0F, b & 0x0F0F0F0F], [(a >> 4) & 0x0F0F0F0F, (b >> 4) & 0x0F0F0F0F])
                } else {
                    let ql = xb.add(48 + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
                    let qh = xb.add(16 + 4 * ((iqs / 2) % 4));
                    let (l0, l1) = (*(ql as *const u32), *(ql.add(16) as *const u32));
                    let h0 = ((*(qh as *const u32)) as i32 >> bq8_offset) as u32;
                    let h1 = ((*(qh.add(16) as *const u32)) as i32 >> bq8_offset) as u32;
                    (
                        [(l0 & 0x0F0F0F0F) | ((h0 << 4) & 0x10101010), (l1 & 0x0F0F0F0F) | ((h1 << 4) & 0x10101010)],
                        [((l0 >> 4) & 0x0F0F0F0F) | (((h0 >> 1) << 4) & 0x10101010), ((l1 >> 4) & 0x0F0F0F0F) | (((h1 >> 1) << 4) & 0x10101010)],
                    )
                };
                let v = [v0, v1];
                let s16 = |k: usize| ld16(xb, 4 + 2 * k);
                let j = bq8_offset / 2;
                let (aux0, aux1) = if j < 2 {
                    (s16(j) & 0x3f3f, s16(j + 2) & 0x3f3f)
                } else {
                    (((s16(j + 2) >> 0) & 0x0f0f) | ((s16(j - 2) & 0xc0c0) >> 2), ((s16(j + 2) >> 4) & 0x0f0f) | ((s16(j) & 0xc0c0) >> 2))
                };
                let sc = [aux0 & 0xFF, aux0 >> 8];
                let m = [aux1 & 0xFF, aux1 >> 8];
                let mut sumf_d = 0f32;
                let mut sumf_m = 0f32;
                let mut i = 0;
                while i < 2 {
                    let b8 = yb.add((bq8_offset + i) * 36);
                    let d8 = ldh(b8, 0);
                    let u0 = yq(b8, (iqs / 2) % 4);
                    let u1 = yq(b8, (iqs / 2) % 4 + 4);
                    let dot1 = dp4a(v[i][1], u1, dp4a(v[i][0], u0, 0));
                    let dot2 = dp4a(0x01010101, u1, dp4a(0x01010101, u0, 0));
                    sumf_d = fma(d8, dot1.wrapping_mul(sc[i] as i32) as f32, sumf_d);
                    sumf_m = fma(d8, dot2.wrapping_mul(m[i] as i32) as f32, sumf_m);
                    i += 1;
                }
                add(tmp, fma(ldh(xb, 0), sumf_d, -mul(ldh(xb, 2), sumf_m)))
            }
            _ => {
                // q6_K, vdr = 1.
                let bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
                let scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
                let vh_shift = 2 * ((iqs % 16) / 8);
                let vl = ld32(xb, 4 * iqs);
                let vh = ((ld32(xb, 128 + 4 * (8 * (iqs / 16) + iqs % 8)) as i32) >> vh_shift) as u32;
                let mut sumf = 0f32;
                let mut i = 0;
                while i < 2 {
                    let b8 = yb.add((bq8_offset + 2 * i) * 36);
                    let u = yq(b8, iqs % 8);
                    let d8 = ldh(b8, 0);
                    let sc = ld8s(xb, 192 + scale_offset + 4 * i);
                    let vil = (vl >> (4 * i)) & 0x0F0F0F0F;
                    let vih = ((vh >> (4 * i)) << 4) & 0x30303030;
                    let vi = vsubss4(vil | vih, 0x20202020);
                    sumf = fma(d8, (dp4a(vi, u, 0).wrapping_mul(sc)) as f32, sumf);
                    i += 1;
                }
                fma(ldh(xb, 208), sumf, tmp)
            }
        }
    }

    /// Compile-time loop bounds (so `#[unroll]` sees literal trip counts after monomorphization;
    /// a bound read from a local or a struct field makes the unroll pass skip the loop).
    pub struct K<const FMT: u32, const NC: usize>;
    impl<const FMT: u32, const NC: usize> K<FMT, NC> {
        pub const RPB: usize = if NC == 1 { 1 } else { 2 };
        pub const NW1: i32 = if NC <= 4 { 3 } else { 1 };
        pub const MX: i32 = mmq_cfg(FMT).mmq_x;
        pub const MY: i32 = mmq_cfg(FMT).mmq_y;
        pub const QR: i32 = mmq_cfg(FMT).qr;
    }

    /// Template `mul_mat_vec_q<ncols_y, ...>`: block (32, nwarps), `rows_per_cuda_block` rows.
    #[device]
    #[inline(always)]
    pub unsafe fn mmvq<const FMT: u32, const NC: usize>(vx: *const u8, vy: *const u8, dst: *mut f32,
                                                        ncols_x: i32, _nrows_x: i32, nrows_y: i32, nrows_dst: i32) {
        static mut TMP_SHARED: SharedArray<f32, 768> = SharedArray::UNINIT;
        let nwarps: i32 = if NC <= 4 { 4 } else { 2 };
        let rpb: i32 = if NC == 1 { 1 } else { 2 };
        let (qk, qi, vdr, bs) = fmt_params::<FMT>();
        let tx = thread::threadIdx_x() as i32;
        let ty = thread::threadIdx_y() as i32;
        let tid = 32 * ty + tx;
        let row0 = rpb * thread::blockIdx_x() as i32;
        let bpr = ncols_x / qk;
        let bpc = nrows_y / 32;
        let bpi = vdr * nwarps * 32 / qi;
        let mut tmp = [[0f32; 2]; NC];
        let mut kbx = tid / (qi / vdr);
        while kbx < bpr {
            let kby = kbx * (qk / 32);
            let kqs = (vdr * (tid % (qi / vdr))) as usize;
            let mut j = 0;
            #[unroll]
            while j < NC {
                let mut i = 0;
                #[unroll]
                while i < K::<FMT, NC>::RPB {
                    let xb = vx.offset((kbx + (row0 + i as i32) * bpr) as isize * bs as isize);
                    let yb = vy.offset((j as i32 * bpc + kby) as isize * 36);
                    tmp[j][i] = mmvq_acc::<FMT>(tmp[j][i], xb, yb, kqs);
                    i += 1;
                }
                j += 1;
            }
            kbx += bpi;
        }
        let sh = SharedArray::as_raw_mut_ptr(&raw mut TMP_SHARED);
        // tmp_shared[nwarps-1][ncols_y][rows_per_cuda_block][WARP_SIZE]
        let idx = |l: i32, j: usize, i: usize| (((l as usize * NC + j) * rpb as usize + i) * 32 + tx as usize);
        if ty > 0 {
            let mut j = 0;
            #[unroll]
            while j < NC {
                let mut i = 0;
                #[unroll]
                while i < K::<FMT, NC>::RPB {
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
        #[unroll]
        while j < NC {
            let mut i = 0;
            #[unroll]
            while i < K::<FMT, NC>::RPB {
                let mut l = 0;
                #[unroll]
                while l < K::<FMT, NC>::NW1 {
                    tmp[j][i] = add(tmp[j][i], *sh.add(idx(l, j, i)));
                    l += 1;
                }
                tmp[j][i] = warp_sum(tmp[j][i]);
                i += 1;
            }
            if tx < rpb {
                let v = if tx == 0 { tmp[j][0] } else { tmp[j][1] };
                *dst.offset((j as i32 * nrows_dst + row0 + tx) as isize) = v;
            }
            j += 1;
        }
    }

    #[kernel] pub unsafe fn mul_mat_vec_q4_0_q8_1_cuda1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<0, 1>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q4_1_q8_1_cuda1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<1, 1>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q5_0_q8_1_cuda1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<2, 1>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q5_1_q8_1_cuda1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<3, 1>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q8_0_q8_1_cuda1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<4, 1>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q2_K_q8_1_cuda1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<5, 1>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q3_K_q8_1_cuda1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<6, 1>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q4_K_q8_1_cuda1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<7, 1>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q5_K_q8_1_cuda1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<8, 1>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q6_K_q8_1_cuda1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<9, 1>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q4_0_q8_1_cuda2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<0, 2>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q4_1_q8_1_cuda2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<1, 2>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q5_0_q8_1_cuda2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<2, 2>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q5_1_q8_1_cuda2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<3, 2>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q8_0_q8_1_cuda2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<4, 2>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q2_K_q8_1_cuda2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<5, 2>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q3_K_q8_1_cuda2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<6, 2>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q4_K_q8_1_cuda2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<7, 2>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q5_K_q8_1_cuda2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<8, 2>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q6_K_q8_1_cuda2(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<9, 2>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q4_0_q8_1_cuda3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<0, 3>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q4_1_q8_1_cuda3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<1, 3>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q5_0_q8_1_cuda3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<2, 3>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q5_1_q8_1_cuda3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<3, 3>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q8_0_q8_1_cuda3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<4, 3>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q2_K_q8_1_cuda3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<5, 3>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q3_K_q8_1_cuda3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<6, 3>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q4_K_q8_1_cuda3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<7, 3>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q5_K_q8_1_cuda3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<8, 3>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q6_K_q8_1_cuda3(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<9, 3>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q4_0_q8_1_cuda4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<0, 4>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q4_1_q8_1_cuda4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<1, 4>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q5_0_q8_1_cuda4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<2, 4>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q5_1_q8_1_cuda4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<3, 4>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q8_0_q8_1_cuda4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<4, 4>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q2_K_q8_1_cuda4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<5, 4>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q3_K_q8_1_cuda4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<6, 4>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q4_K_q8_1_cuda4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<7, 4>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q5_K_q8_1_cuda4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<8, 4>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q6_K_q8_1_cuda4(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<9, 4>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q4_0_q8_1_cuda5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<0, 5>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q4_1_q8_1_cuda5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<1, 5>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q5_0_q8_1_cuda5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<2, 5>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q5_1_q8_1_cuda5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<3, 5>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q8_0_q8_1_cuda5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<4, 5>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q2_K_q8_1_cuda5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<5, 5>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q3_K_q8_1_cuda5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<6, 5>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q4_K_q8_1_cuda5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<7, 5>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q5_K_q8_1_cuda5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<8, 5>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q6_K_q8_1_cuda5(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<9, 5>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q4_0_q8_1_cuda6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<0, 6>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q4_1_q8_1_cuda6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<1, 6>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q5_0_q8_1_cuda6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<2, 6>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q5_1_q8_1_cuda6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<3, 6>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q8_0_q8_1_cuda6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<4, 6>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q2_K_q8_1_cuda6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<5, 6>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q3_K_q8_1_cuda6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<6, 6>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q4_K_q8_1_cuda6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<7, 6>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q5_K_q8_1_cuda6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<8, 6>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q6_K_q8_1_cuda6(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<9, 6>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q4_0_q8_1_cuda7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<0, 7>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q4_1_q8_1_cuda7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<1, 7>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q5_0_q8_1_cuda7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<2, 7>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q5_1_q8_1_cuda7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<3, 7>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q8_0_q8_1_cuda7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<4, 7>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q2_K_q8_1_cuda7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<5, 7>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q3_K_q8_1_cuda7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<6, 7>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q4_K_q8_1_cuda7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<7, 7>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q5_K_q8_1_cuda7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<8, 7>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q6_K_q8_1_cuda7(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<9, 7>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q4_0_q8_1_cuda8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<0, 8>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q4_1_q8_1_cuda8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<1, 8>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q5_0_q8_1_cuda8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<2, 8>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q5_1_q8_1_cuda8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<3, 8>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q8_0_q8_1_cuda8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<4, 8>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q2_K_q8_1_cuda8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<5, 8>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q3_K_q8_1_cuda8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<6, 8>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q4_K_q8_1_cuda8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<7, 8>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q5_K_q8_1_cuda8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<8, 8>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_vec_q6_K_q8_1_cuda8(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, nrows_y: i32, nrows_dst: i32) { mmvq::<9, 8>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst) }

    /// `indexed_moe_forward<qk, qi, block_q_t, vdr, vec_dot>`: grid (n, batch, topk), block (32, 4).
    #[device]
    #[inline(always)]
    pub unsafe fn moe<const FMT: u32>(w: *const u8, xin: *const u8, indices: *const u32, out: *mut f32,
                                     n: i32, k: i32, _batch: i32, _topk: i32, k_padded: i32, input_dim1: i32) {
        static mut MOE_SHARED: SharedArray<f32, 96> = SharedArray::UNINIT;
        let current_batch = thread::blockIdx_y() as i32;
        let current_topk = thread::blockIdx_z() as i32;
        let task_id = current_batch.wrapping_mul(thread::gridDim_z() as i32).wrapping_add(current_topk);
        if (task_id as u32) >= thread::gridDim_y().wrapping_mul(thread::gridDim_z()) {
            return;
        }
        let input_idx = if input_dim1 == 1 { current_batch } else { task_id };
        let expert_id = *indices.offset(task_id as isize);
        let (qk, qi, vdr, bs) = fmt_params::<FMT>();
        // (size_t)(n * k) / QK_K * sizeof(block_q_t): QK_K = 256 for every format, q8_0 included.
        let weight_expert_stride = ((n.wrapping_mul(k) as i64 as u64) / 256).wrapping_mul(bs as u64);
        let input_task_stride = (k_padded as i64 as u64 / 32).wrapping_mul(36);
        let xw = w.wrapping_add((expert_id as u64).wrapping_mul(weight_expert_stride) as usize);
        let xq = xin.wrapping_add((input_idx as i64 as u64).wrapping_mul(input_task_stride) as usize);
        let outp = out.wrapping_add((task_id as i64 as u64).wrapping_mul(n as i64 as u64) as usize);
        let tx = thread::threadIdx_x() as i32;
        let ty = thread::threadIdx_y() as i32;
        let tid = 32 * ty + tx;
        let row0 = thread::blockIdx_x() as i32;
        if row0 >= n {
            return;
        }
        let bpr = k / qk;
        let bpi = vdr * 4 * 32 / qi;
        let mut tmp = 0f32;
        let mut kbx = tid / (qi / vdr);
        while kbx < bpr {
            let kby = kbx * (qk / 32);
            let kqs = (vdr * (tid % (qi / vdr))) as usize;
            let xb = xw.offset((kbx + row0 * bpr) as isize * bs as isize);
            let yb = xq.offset(kby as isize * 36);
            tmp = mmvq_acc::<FMT>(tmp, xb, yb, kqs);
            kbx += bpi;
        }
        let sh = SharedArray::as_raw_mut_ptr(&raw mut MOE_SHARED);
        if ty > 0 {
            *sh.add(((ty - 1) * 32 + tx) as usize) = tmp;
        }
        thread::sync_threads();
        if ty == 0 {
            let mut l = 0;
            while l < 3 {
                tmp = add(tmp, *sh.add((l * 32 + tx) as usize));
                l += 1;
            }
            tmp = warp_sum(tmp);
            if tx == 0 {
                *outp.offset(row0 as isize) = tmp;
            }
        }
    }

    /// Per-block isolation: one vec_dot value for a given (kbx, iqs), same pointer
    /// arithmetic as the real caller. Pairs with reference/iq_moe's iq_dbg_vdot_*.
    /// Piece-by-piece IQ2_XXS dump: mirrors iq_probe_iq2_xxs in reference/iq_moe so the two can
    /// be compared field by field (which component diverges, not just "the output").
    #[kernel] pub unsafe fn iq_probe_iq2_xxs(w: *const u8, x: *const u8, out: *mut f32, kbx: i32, iqs: i32) {
        if thread::threadIdx_x() == 0 {
            let xb = w.add((kbx as usize) * 66);
            let b8 = x.add((kbx as usize) * 8 * 36 + (iqs as usize / 2) * 36);
            let qs2 = xb.add(2);
            let q2 = get_b2(qs2, iqs);
            let aux8 = q2.to_le_bytes();
            let aux32 = get_b2(qs2, iqs + 1);
            let mut sumi = 0i32;
            let mut first_grid = 0u32;
            let mut k0 = 0usize;
            while k0 < 8 {
                let idx = aux8[k0 / 2] as u32;
                let gl = grid_lo(idx);
                let gh = grid_hi(idx);
                if k0 == 0 { first_grid = gl; }
                let signs = unpack_ksigns(aux32 >> (7 * k0 as u32 / 2));
                let s0 = vcmpne4_zero(signs & 0x0804_0201);
                let s1 = vcmpne4_zero(signs & 0x8040_2010);
                let g0 = vsub4_wrap(gl ^ s0, s0);
                let g1 = vsub4_wrap(gh ^ s1, s1);
                sumi = dp4a(g0, yq(b8, k0), sumi);
                sumi = dp4a(g1, yq(b8, k0 + 1), sumi);
                k0 += 2;
            }
            let ls = (aux32 >> 27 | 1) as i32;
            let d = mul(h2f(ld16(xb, 0) as u16), h2f(ld16(b8, 0) as u16));
            *out.add(0) = mul(d, (sumi.wrapping_mul(ls) / 8) as f32);
            *out.add(1) = sumi as f32;
            *out.add(2) = d;
            *out.add(3) = aux32 as f32;
            *out.add(4) = aux8[0] as f32;
            *out.add(5) = first_grid as f32;
            *out.add(6) = *b8.add(4) as f32;
            *out.add(7) = *b8.add(8) as f32;
            *out.add(8) = ld16(xb, 0) as f32;      // raw half bits of the weight scale
            *out.add(9) = ld16(b8, 0) as f32;      // raw bits of q8 d
            *out.add(10) = h2f(ld16(xb, 0) as u16);
            *out.add(11) = h2f(ld16(b8, 0) as u16);
            // k0 = 0 iteration internals
            let gh0 = grid_hi(aux8[0] as u32);
            *out.add(12) = gh0 as f32;
            let sg0 = unpack_ksigns(aux32);
            *out.add(13) = sg0 as f32;
            let vc = vcmpne4_zero(sg0 & 0x0804_0201);
            *out.add(14) = vc as f32;
            let gv = vsub4_wrap(grid_lo(aux8[0] as u32) ^ vc, vc);
            *out.add(15) = gv as f32;
            *out.add(16) = dp4a(gv, yq(b8, 0), 0) as f32;
        }
    }
    #[kernel] pub unsafe fn iq_dbg_vdot_iq2_xxs_q8_1(w: *const u8, x: *const u8, out: *mut f32, kbx: i32, iqs: i32) {
        if thread::threadIdx_x() == 0 {
            let wb = w.add((kbx as usize) * 66);
            let xb = x.add((kbx as usize) * 8 * 36);
            *out = mmvq_acc::<11>(0.0, wb, xb, iqs as usize);
        }
    }
    #[kernel] pub unsafe fn iq_dbg_vdot_iq2_xs_q8_1(w: *const u8, x: *const u8, out: *mut f32, kbx: i32, iqs: i32) {
        if thread::threadIdx_x() == 0 {
            let wb = w.add((kbx as usize) * 74);
            let xb = x.add((kbx as usize) * 8 * 36);
            *out = mmvq_acc::<17>(0.0, wb, xb, iqs as usize);
        }
    }
    #[kernel] pub unsafe fn iq_dbg_vdot_iq2_s_q8_1(w: *const u8, x: *const u8, out: *mut f32, kbx: i32, iqs: i32) {
        if thread::threadIdx_x() == 0 {
            let wb = w.add((kbx as usize) * 82);
            let xb = x.add((kbx as usize) * 8 * 36);
            *out = mmvq_acc::<22>(0.0, wb, xb, iqs as usize);
        }
    }
    #[kernel] pub unsafe fn iq_dbg_vdot_iq3_xxs_q8_1(w: *const u8, x: *const u8, out: *mut f32, kbx: i32, iqs: i32) {
        if thread::threadIdx_x() == 0 {
            let wb = w.add((kbx as usize) * 98);
            let xb = x.add((kbx as usize) * 8 * 36);
            *out = mmvq_acc::<18>(0.0, wb, xb, iqs as usize);
        }
    }
    #[kernel] pub unsafe fn iq_dbg_vdot_iq4_xs_q8_1(w: *const u8, x: *const u8, out: *mut f32, kbx: i32, iqs: i32) {
        if thread::threadIdx_x() == 0 {
            let wb = w.add((kbx as usize) * 136);
            let xb = x.add((kbx as usize) * 8 * 36);
            *out = mmvq_acc::<23>(0.0, wb, xb, iqs as usize);
        }
    }
    #[kernel] pub unsafe fn indexed_moe_forward_iq2_xxs_q8_1(w: *const u8, xin: *const u8, indices: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, k_padded: i32, input_dim1: i32) { moe::<11>(w, xin, indices, out, n, k, batch, topk, k_padded, input_dim1) }
    #[kernel] pub unsafe fn indexed_moe_forward_iq3_xxs_q8_1(w: *const u8, xin: *const u8, indices: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, k_padded: i32, input_dim1: i32) { moe::<18>(w, xin, indices, out, n, k, batch, topk, k_padded, input_dim1) }
    #[kernel] pub unsafe fn indexed_moe_forward_iq4_xs_q8_1(w: *const u8, xin: *const u8, indices: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, k_padded: i32, input_dim1: i32) { moe::<23>(w, xin, indices, out, n, k, batch, topk, k_padded, input_dim1) }
    #[kernel] pub unsafe fn indexed_moe_forward_iq2_xs_q8_1(w: *const u8, xin: *const u8, indices: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, k_padded: i32, input_dim1: i32) { moe::<17>(w, xin, indices, out, n, k, batch, topk, k_padded, input_dim1) }
    #[kernel] pub unsafe fn indexed_moe_forward_iq2_s_q8_1(w: *const u8, xin: *const u8, indices: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, k_padded: i32, input_dim1: i32) { moe::<22>(w, xin, indices, out, n, k, batch, topk, k_padded, input_dim1) }
    #[kernel] pub unsafe fn indexed_moe_forward_q2k_q8_1(w: *const u8, xin: *const u8, indices: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, k_padded: i32, input_dim1: i32) { moe::<5>(w, xin, indices, out, n, k, batch, topk, k_padded, input_dim1) }
    #[kernel] pub unsafe fn indexed_moe_forward_q3k_q8_1(w: *const u8, xin: *const u8, indices: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, k_padded: i32, input_dim1: i32) { moe::<6>(w, xin, indices, out, n, k, batch, topk, k_padded, input_dim1) }
    #[kernel] pub unsafe fn indexed_moe_forward_q4k_q8_1(w: *const u8, xin: *const u8, indices: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, k_padded: i32, input_dim1: i32) { moe::<7>(w, xin, indices, out, n, k, batch, topk, k_padded, input_dim1) }
    #[kernel] pub unsafe fn indexed_moe_forward_q5k_q8_1(w: *const u8, xin: *const u8, indices: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, k_padded: i32, input_dim1: i32) { moe::<8>(w, xin, indices, out, n, k, batch, topk, k_padded, input_dim1) }
    #[kernel] pub unsafe fn indexed_moe_forward_q6k_q8_1(w: *const u8, xin: *const u8, indices: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, k_padded: i32, input_dim1: i32) { moe::<9>(w, xin, indices, out, n, k, batch, topk, k_padded, input_dim1) }
    #[kernel] pub unsafe fn indexed_moe_forward_q8_0_q8_1(w: *const u8, xin: *const u8, indices: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, k_padded: i32, input_dim1: i32) { moe::<4>(w, xin, indices, out, n, k, batch, topk, k_padded, input_dim1) }

    // ------------------------------------------------------------------------------------------
    // Family 5: mul_mat_q* (10 entries): llama.cpp's tiled MMQ, non-tensor-core Ampere configs
    // (mmq_x, mmq_y, nwarps = 4), load_tiles with need_check = true. Block (32, 4),
    // grid (ceil(nrows_x / mmq_y), ceil(ncols_y / mmq_x)).

    /// Shared tile layout (u32 offsets) and template parameters of one MMQ format.
    #[derive(Clone, Copy)]
    pub struct MmqCfg {
        pub qk: i32, pub qr: i32, pub qi: i32, pub need_sum: bool,
        pub mmq_x: i32, pub mmq_y: i32, pub vdr: i32, pub bs: usize,
        pub x_ql: usize, pub x_dm: usize, pub x_qh: usize, pub x_sc: usize, pub y_qs: usize, pub y_ds: usize,
    }

    #[inline(always)]
    pub const fn mmq_cfg(fmt: u32) -> MmqCfg {
        // (qk, qr, qi, need_sum, mmq_x, mmq_y, vdr, bs, x_ql words, x_dm words, x_qh words, x_sc words)
        let (qk, qr, qi, need_sum, mmq_x, mmq_y, vdr, bs, nql, ndm, nqh, nsc) = match fmt {
            0 => (32, 2, 4, true, 64, 128, 4, 18, 128 * 33, 128 * 8 + 32, 0, 0),
            1 => (32, 2, 4, true, 64, 128, 4, 20, 128 * 33, 128 * 8 + 32, 0, 0),
            2 => (32, 2, 4, false, 128, 64, 4, 22, 64 * 65, 64 * 8 + 16, 0, 0),
            3 => (32, 2, 4, true, 128, 64, 4, 24, 64 * 65, 64 * 8 + 16, 0, 0),
            4 => (32, 1, 8, false, 128, 64, 8, 34, 64 * 33, 64 * 4 + 8, 0, 0),
            5 => (256, 4, 16, false, 64, 128, 2, 84, 128 * 33, 128 * 2 + 8, 0, 128 * 8 + 32),
            6 => (256, 4, 16, false, 128, 128, 2, 110, 128 * 33, 128 * 2 + 8, 128 * 16 + 64, 128 * 8 + 32),
            7 => (256, 2, 32, true, 64, 128, 8, 144, 128 * 33, 128 + 4, 0, 128 * 4 + 16),
            8 => (256, 2, 32, true, 64, 128, 8, 176, 128 * 65, 128 + 4, 0, 128 * 4 + 16),
            _ => (256, 2, 32, false, 64, 64, 8, 210, 64 * 65, 64 + 2, 0, 64 * 4 + 8),
        };
        let x_ql = 0;
        let x_dm = x_ql + nql;
        let x_qh = x_dm + ndm;
        let x_sc = x_qh + nqh;
        let y_qs = x_sc + nsc;
        let y_ds = y_qs + (mmq_x as usize) * 32;
        MmqCfg { qk, qr, qi, need_sum, mmq_x, mmq_y, vdr, bs, x_ql, x_dm, x_qh, x_sc, y_qs, y_ds }
    }

    #[inline(always)]
    fn imin(a: i32, b: i32) -> i32 {
        if a < b { a } else { b }
    }

    #[inline(always)]
    unsafe fn h2_lo(w: u32) -> f32 {
        h2f(w as u16)
    }
    #[inline(always)]
    unsafe fn h2_hi(w: u32) -> f32 {
        h2f((w >> 16) as u16)
    }
    /// Byte `b` of the u32 array at `p` (little-endian), as the C code's `(uint8_t *)&x_sc[...]`.
    #[inline(always)]
    unsafe fn sbyte(p: *const u32, b: usize) -> u32 {
        (*p.add(b / 4) >> (8 * (b % 4))) & 0xFF
    }

    /// `load_tiles_<fmt><mmq_y, nwarps = 4, need_check = true>`.
    #[device]
    #[inline(always)]
    pub unsafe fn load_tiles<const FMT: u32>(smb: u32, bx0: *const u8, i_offset: i32, i_max: i32, k: i32, bpr: i32) {
        let c = mmq_cfg(FMT);
        // 32-bit shared addresses (bytes) of the x tiles.
        let (x_ql, x_dm, x_qh, x_sc) = (smb + 4 * c.x_ql as u32, smb + 4 * c.x_dm as u32, smb + 4 * c.x_qh as u32, smb + 4 * c.x_sc as u32);
        let my = c.mmq_y;
        let blk = |i: i32, kb: i32| bx0.offset((i * bpr + kb) as isize * c.bs as isize);
        let st = |p: u32, idx: i32, v: u32| sts32(p.wrapping_add(4 * idx as u32), v);
        let aligned = |b: *const u8, off: i32| ldg32(b.offset(off as isize));
        let unal = |b: *const u8, off: i32| ldg16(b.offset(off as isize)) | (ldg16(b.offset(off as isize + 2)) << 16);
        let ldh = |b: *const u8, off: usize| h2f(ldg16(b.add(off)) as u16);
        match FMT {
            0 | 1 => {
                let (kbx, kqsx) = (k / 4, k % 4);
                let mut i0 = 0;
                #[unroll]
                while i0 < K::<FMT, 0>::MY {
                    let i = imin(i0 + i_offset, i_max);
                    let b = blk(i, kbx);
                    st(x_ql, i * 33 + k, if FMT == 0 { unal(b, 2 + 4 * kqsx) } else { aligned(b, 4 + 4 * kqsx) });
                    i0 += 4;
                }
                let kbxd = k % 8;
                let mut i0 = 0;
                #[unroll]
                while i0 < K::<FMT, 0>::MY {
                    let i = imin(i0 + i_offset * 4 + k / 8, i_max);
                    let b = blk(i, kbxd);
                    st(x_dm, i * 8 + i / 4 + kbxd, if FMT == 0 { ldh(b, 0).to_bits() } else { aligned(b, 0) });
                    i0 += 16;
                }
            }
            2 | 3 => {
                let (kbx, kqsx) = (k / 4, k % 4);
                let mut i0 = 0;
                #[unroll]
                while i0 < K::<FMT, 0>::MY {
                    let i = imin(i0 + i_offset, i_max);
                    let b = blk(i, kbx);
                    let (ql, qh) = if FMT == 2 { (unal(b, 6 + 4 * kqsx), unal(b, 2)) } else { (aligned(b, 8 + 4 * kqsx), aligned(b, 4)) };
                    let qh = ((qh as i32) >> (4 * (k % 4))) as u32;
                    let mut qs0 = ql & 0x0F0F0F0F;
                    qs0 |= (qh << 4) & 0x00000010;
                    qs0 |= (qh << 11) & 0x00001000;
                    qs0 |= (qh << 18) & 0x00100000;
                    qs0 |= (qh << 25) & 0x10000000;
                    let mut qs1 = (ql >> 4) & 0x0F0F0F0F;
                    qs1 |= (qh >> 12) & 0x00000010;
                    qs1 |= (qh >> 5) & 0x00001000;
                    qs1 |= (qh << 2) & 0x00100000;
                    qs1 |= (qh << 9) & 0x10000000;
                    if FMT == 2 {
                        qs0 = vsubss4(qs0, 0x10101010);
                        qs1 = vsubss4(qs1, 0x10101010);
                    }
                    st(x_ql, i * 65 + 2 * k, qs0);
                    st(x_ql, i * 65 + 2 * k + 1, qs1);
                    i0 += 4;
                }
                let kbxd = k % 8;
                let mut i0 = 0;
                #[unroll]
                while i0 < K::<FMT, 0>::MY {
                    let i = imin(i0 + i_offset * 4 + k / 8, i_max);
                    let b = blk(i, kbxd);
                    st(x_dm, i * 8 + i / 4 + kbxd, if FMT == 2 { ldh(b, 0).to_bits() } else { aligned(b, 0) });
                    i0 += 16;
                }
            }
            4 => {
                let (kbx, kqsx) = (k / 8, k % 8);
                let mut i0 = 0;
                #[unroll]
                while i0 < K::<FMT, 0>::MY {
                    let i = imin(i0 + i_offset, i_max);
                    st(x_ql, i * 33 + k, unal(blk(i, kbx), 2 + 4 * kqsx));
                    i0 += 4;
                }
                let kbxd = k % 4;
                let mut i0 = 0;
                #[unroll]
                while i0 < K::<FMT, 0>::MY {
                    let i = imin(i0 + i_offset * 8 + k / 4, i_max);
                    st(x_dm, i * 4 + i / 8 + kbxd, ldh(blk(i, kbxd), 0).to_bits());
                    i0 += 32;
                }
            }
            5 | 6 => {
                let (kbx, kqsx) = (k / 16, k % 16);
                let mut i0 = 0;
                #[unroll]
                while i0 < K::<FMT, 0>::MY {
                    let i = imin(i0 + i_offset, i_max);
                    let b = blk(i, kbx);
                    st(x_ql, i * 33 + k, if FMT == 5 { aligned(b, 16 + 4 * kqsx) } else { unal(b, 32 + 4 * kqsx) });
                    i0 += 4;
                }
                let kbxd = k % 2;
                let mut i0 = 0;
                #[unroll]
                while i0 < K::<FMT, 0>::MY {
                    let i = imin((i0 + i_offset * 16 + k / 2) % my, i_max);
                    let b = blk(i, kbxd);
                    st(x_dm, i * 2 + i / 16 + kbxd, if FMT == 5 { aligned(b, 80) } else { ldh(b, 108).to_bits() });
                    i0 += 64;
                }
                if FMT == 6 {
                    let mut i0 = 0;
                    #[unroll]
                    while i0 < K::<FMT, 0>::MY {
                        let i = imin(i0 + i_offset * 2 + k / 16, i_max);
                        let b = blk(i, (k % 16) / 8);
                        st(x_qh, i * 16 + i / 2 + k % 16, !unal(b, 4 * (k % 8)));
                        i0 += 8;
                    }
                }
                let mut i0 = 0;
                #[unroll]
                while i0 < K::<FMT, 0>::MY {
                    let i = imin(i0 + i_offset * 4 + k / 8, i_max);
                    let b = blk(i, (k % 8) / 4);
                    let v = if FMT == 5 {
                        aligned(b, 4 * (k % 4))
                    } else {
                        let ksc = k % 4;
                        let sc_low = (((unal(b, 96 + 4 * (ksc % 2)) as i32) >> (4 * (ksc / 2))) as u32) & 0x0F0F0F0F;
                        let sc_high = ((((unal(b, 96 + 8) as i32) >> (2 * ksc)) as u32) << 4) & 0x30303030;
                        vsubss4(sc_low | sc_high, 0x20202020)
                    };
                    st(x_sc, i * 8 + i / 4 + k % 8, v);
                    i0 += 16;
                }
            }
            7 | 8 | 9 => {
                let mut i0 = 0;
                #[unroll]
                while i0 < K::<FMT, 0>::MY {
                    let i = imin(i0 + i_offset, i_max);
                    let b = blk(i, 0);
                    if FMT == 7 {
                        st(x_ql, i * 33 + k, aligned(b, 16 + 4 * k));
                    } else if FMT == 8 {
                        let ky = 2 * k;
                        let ql = aligned(b, 48 + 4 * k);
                        let qh = aligned(b, 16 + 4 * (k % 8)) as i32;
                        let qh0 = (((qh >> (2 * (k / 8))) << 4) as u32) & 0x10101010;
                        let qh1 = (((qh >> (2 * (k / 8) + 1)) << 4) as u32) & 0x10101010;
                        let kq0 = ky - ky % 16 + k % 8;
                        st(x_ql, i * 65 + kq0, (ql & 0x0F0F0F0F) | qh0);
                        st(x_ql, i * 65 + kq0 + 8, ((ql >> 4) & 0x0F0F0F0F) | qh1);
                    } else {
                        let ky = 2 * k;
                        let ql = unal(b, 4 * k);
                        let qh = unal(b, 128 + 4 * (8 * (k / 16) + k % 8)) as i32;
                        let sh = 2 * ((k % 16) / 8);
                        let qh0 = (((qh >> sh) << 4) as u32) & 0x30303030;
                        let qh1 = ((qh >> sh) as u32) & 0x30303030;
                        let kq0 = ky - ky % 32 + k % 16;
                        st(x_ql, i * 65 + kq0, vsubss4((ql & 0x0F0F0F0F) | qh0, 0x20202020));
                        st(x_ql, i * 65 + kq0 + 16, vsubss4(((ql >> 4) & 0x0F0F0F0F) | qh1, 0x20202020));
                    }
                    i0 += 4;
                }
                let mut i0 = 0;
                #[unroll]
                while i0 < K::<FMT, 0>::MY {
                    let i = imin((i0 + i_offset * 32 + k) % my, i_max);
                    let b = blk(i, 0);
                    st(x_dm, i + i / 32, if FMT == 9 { ldh(b, 208).to_bits() } else { aligned(b, 0) });
                    i0 += 128;
                }
                let mut i0 = 0;
                #[unroll]
                while i0 < K::<FMT, 0>::MY {
                    let i = imin((i0 + i_offset * 8 + k / 4) % my, i_max);
                    let b = blk(i, 0);
                    let ksc = k % 4;
                    let v = if FMT == 9 {
                        unal(b, 192 + 4 * ksc)
                    } else {
                        let sw = |n: i32| aligned(b, 4 + 4 * n) as i32;
                        let lo = (sw((ksc % 2) + (ksc != 0) as i32) >> (4 * (ksc & (ksc / 2)))) as u32 & 0x0F0F0F0F;
                        let hi = (sw(ksc / 2) >> (2 * (ksc % 2))) as u32 & 0x30303030;
                        lo | hi
                    };
                    st(x_sc, i * 4 + i / 8 + ksc, v);
                    i0 += 32;
                }
            }
            _ => {}
        }
    }

    /// `sum += vec_dot_<fmt>_q8_1_mul_mat(tiles, i, j, k)` with the reference's contraction.
    #[inline(always)]
    unsafe fn mmq_acc<const FMT: u32>(s: f32, sm: *const u32, i: usize, j: usize, k: usize) -> f32 {
        let c = mmq_cfg(FMT);
        let (x_ql, x_dm, x_qh, x_sc, y_qs, y_ds) =
            (sm.add(c.x_ql), sm.add(c.x_dm), sm.add(c.x_qh), sm.add(c.x_sc), sm.add(c.y_qs), sm.add(c.y_ds));
        let f = |p: *const u32, idx: usize| f32::from_bits(*p.add(idx));
        match FMT {
            0 | 1 | 2 | 3 => {
                let kyqs = k % 4 + 8 * (k / 4);
                let mut u = [0u32; 8];
                let mut l = 0;
                while l < 4 {
                    u[2 * l] = *y_qs.add(j * 32 + (kyqs + l) % 32);
                    u[2 * l + 1] = *y_qs.add(j * 32 + (kyqs + l + 4) % 32);
                    l += 1;
                }
                let mut sumi = 0i32;
                if FMT <= 1 {
                    let mut l = 0;
                    while l < 4 {
                        let v = *x_ql.add(i * 33 + k + l);
                        sumi = dp4a(v & 0x0F0F0F0F, u[2 * l], sumi);
                        sumi = dp4a((v >> 4) & 0x0F0F0F0F, u[2 * l + 1], sumi);
                        l += 1;
                    }
                } else {
                    let mut l = 0;
                    while l < 8 {
                        sumi = dp4a(*x_ql.add(i * 65 + 2 * k + l), u[l], sumi);
                        l += 1;
                    }
                }
                let xi = i * 8 + i / 4 + k / 4;
                let yi = j * 4 + (2 * k / 8) % 4;
                let sf = sumi as f32;
                match FMT {
                    0 => {
                        let ds = *y_ds.add(yi);
                        fma(f(x_dm, xi), fma(h2_lo(ds), sf, -mul(h2_hi(ds), 8.0)), s)
                    }
                    2 => fma(mul(f(x_dm, xi), f(y_ds, yi)), sf, s),
                    _ => {
                        let dm = *x_dm.add(xi);
                        let ds = *y_ds.add(yi);
                        add(s, fma(mul(h2_lo(dm), h2_lo(ds)), sf, mul(h2_hi(dm), h2_hi(ds))))
                    }
                }
            }
            4 => {
                let mut sumi = 0i32;
                let mut l = 0;
                while l < 8 {
                    sumi = dp4a(*x_ql.add(i * 33 + k + l), *y_qs.add(j * 32 + k + l), sumi);
                    l += 1;
                }
                fma(mul(f(x_dm, i * 4 + i / 8 + k / 8), f(y_ds, j * 4 + k / 8)), sumi as f32, s)
            }
            5 | 6 => {
                let kbx = k / 16;
                let ky = (k % 16) * 4;
                let kqsx = i * 33 + kbx * 16 + 8 * (ky / 32) + ky % 8;
                let shift = 2 * ((ky % 32) / 8);
                let scw = x_sc.add(i * 8 + i / 4 + kbx * 4) as *const u32;
                let index_y = j * 32 + (4 * k) % 32;
                let mut v = [0u32; 8];
                let mut l = 0;
                while l < 8 {
                    let vll = (*x_ql.add(kqsx + l) >> shift) & 0x03030303;
                    v[l] = if FMT == 5 {
                        vll
                    } else {
                        let vh = (*x_qh.add(i * 16 + i / 2 + kbx * 8 + (ky + l) % 8) as i32) >> ((ky + l) / 8);
                        vsubss4(vll, ((vh << 2) as u32) & 0x04040404)
                    };
                    l += 1;
                }
                let d8 = f(y_ds, index_y / 8);
                if FMT == 5 {
                    let mut sumi_d = 0i32;
                    let mut sumi_m = 0i32;
                    let mut i0 = 0;
                    while i0 < 8 {
                        let sc = sbyte(scw, ky / 4 + i0 / 4);
                        let mut m = sc >> 4;
                        m |= m << 8;
                        m |= m << 16;
                        let mut sumi_d_sc = 0i32;
                        let mut ii = i0;
                        while ii < i0 + 4 {
                            let uu = *y_qs.add(index_y + ii);
                            sumi_d_sc = dp4a(v[ii], uu, sumi_d_sc);
                            sumi_m = dp4a(m, uu, sumi_m);
                            ii += 1;
                        }
                        sumi_d = sumi_d.wrapping_add(sumi_d_sc.wrapping_mul((sc & 0xF) as i32));
                        i0 += 4;
                    }
                    let dm = *x_dm.add(i * 2 + i / 16 + kbx);
                    fma(d8, fma(h2_lo(dm), sumi_d as f32, -mul(h2_hi(dm), sumi_m as f32)), s)
                } else {
                    let mut sumi = 0i32;
                    let mut i0 = 0;
                    while i0 < 8 {
                        let mut sumi_sc = 0i32;
                        let mut ii = i0;
                        while ii < i0 + 4 {
                            sumi_sc = dp4a(v[ii], *y_qs.add(index_y + ii), sumi_sc);
                            ii += 1;
                        }
                        let sc = sbyte(scw, ky / 4 + i0 / 4) as u8 as i8 as i32;
                        sumi = sumi.wrapping_add(sumi_sc.wrapping_mul(sc));
                        i0 += 4;
                    }
                    fma(mul(f(x_dm, i * 2 + i / 16 + kbx), d8), sumi as f32, s)
                }
            }
            7 | 8 => {
                let scw = x_sc.add(i * 4 + i / 8 + k / 16) as *const u32;
                let sco = 2 * ((k % 16) / 8);
                let index_y = j * 32 + (2 * k) % 32;
                let mut sumf_d = 0f32;
                let mut sumf_m = 0f32;
                let mut b = 0;
                while b < 2 {
                    let mut sumi_d = 0i32;
                    let mut jj = 0;
                    while jj < 8 {
                        let vv = if FMT == 7 {
                            (*x_ql.add(i * 33 + k + jj) >> (4 * b)) & 0x0F0F0F0F
                        } else {
                            *x_ql.add(i * 65 + 2 * k + b * 8 + jj)
                        };
                        sumi_d = dp4a(vv, *y_qs.add(index_y + b * 8 + jj), sumi_d);
                        jj += 1;
                    }
                    let ds = *y_ds.add(index_y / 8 + b);
                    let sc = sbyte(scw, sco + b) as i32;
                    let m = sbyte(scw, sco + b + 8);
                    sumf_d = fma(h2_lo(ds), sc.wrapping_mul(sumi_d) as f32, sumf_d);
                    sumf_m = fma(h2_hi(ds), m as f32, sumf_m);
                    b += 1;
                }
                let dm = *x_dm.add(i + i / 32);
                add(s, fma(h2_lo(dm), sumf_d, -mul(h2_hi(dm), sumf_m)))
            }
            _ => {
                let scw = x_sc.add(i * 4 + i / 8 + k / 8) as *const u32;
                let sc = |b: usize| sbyte(scw, b) as u8 as i8 as i32;
                let index_x = i * 65 + 2 * k;
                let index_y = j * 32 + (2 * k) % 32;
                let vv = |m: usize| *x_ql.add(index_x + m);
                let uu = |m: usize| *y_qs.add(index_y + m);
                let mut sumf_d = 0f32;
                let mut i0 = 0;
                while i0 < 8 {
                    let mut sx = 0i32;
                    let mut sy = 0i32;
                    let mut ii = i0;
                    while ii < i0 + 2 {
                        sx = dp4a(vv(2 * ii), uu(2 * ii), sx);
                        sx = dp4a(vv(2 * ii + 1), uu(2 * ii + 1), sx);
                        sy = dp4a(vv(2 * ii + 4), uu(2 * ii + 4), sy);
                        sy = dp4a(vv(2 * ii + 5), uu(2 * ii + 5), sy);
                        ii += 1;
                    }
                    let tot = sc(i0 / 2).wrapping_mul(sx).wrapping_add(sc(i0 / 2 + 1).wrapping_mul(sy));
                    sumf_d = fma(f(y_ds, index_y / 8 + i0 / 4), tot as f32, sumf_d);
                    i0 += 4;
                }
                fma(sumf_d, f(x_dm, i + i / 32), s)
            }
        }
    }

    /// Template `mul_mat_q<...>`. YI = mmq_y / 32, XJ = mmq_x / nwarps.
    #[device]
    #[inline(always)]
    pub unsafe fn mmq<const FMT: u32, const YI: usize, const XJ: usize>(
        vx: *const u8, vy: *const u8, dst: *mut f32,
        ncols_x: i32, nrows_x: i32, ncols_y: i32, nrows_y: i32, nrows_dst: i32,
    ) {
        static mut MMQ_SMEM: SharedArray<u32, 12264> = SharedArray::UNINIT;
        let sm = SharedArray::as_raw_mut_ptr(&raw mut MMQ_SMEM);
        let smb = cuda_device::shared::cvta_generic_to_shared_u32(sm as *const u8);
        let c = mmq_cfg(FMT);
        let tx = thread::threadIdx_x() as i32;
        let ty = thread::threadIdx_y() as i32;
        let bpr = ncols_x / c.qk;
        let bpc = nrows_y / 32;
        let bpw = 32 / c.qi;
        let row_dst_0 = thread::blockIdx_x() as i32 * c.mmq_y;
        let col_dst_0 = thread::blockIdx_y() as i32 * c.mmq_x;
        let (y_qs, y_ds) = (sm.add(c.y_qs), sm.add(c.y_ds));
        let mut sum = [[0f32; XJ]; YI];
        let mut ib0 = 0;
        while ib0 < bpr {
            let bx0 = vx.offset((row_dst_0 * bpr + ib0) as isize * c.bs as isize);
            load_tiles::<FMT>(smb, bx0, ty, nrows_x - row_dst_0 - 1, tx, bpr);
            let mut ir = 0;
            #[unroll]
            while ir < K::<FMT, 0>::QR {
                let kqs = ir * 32 + tx;
                let kbxd = kqs / 8;
                let mut i = 0;
                #[unroll]
                while i < K::<FMT, 0>::MX {
                    let col_y_eff = imin(col_dst_0 + ty + i, ncols_y - 1);
                    let by0 = vy.offset((col_y_eff * bpc + ib0 * (c.qk / 32) + kbxd) as isize * 36);
                    *y_qs.offset(((ty + i) * 32 + kqs % 32) as isize) = ldg32(by0.offset(4 + 4 * (tx % 8) as isize));
                    i += 4;
                }
                let mut ids0 = 0;
                #[unroll]
                while ids0 < K::<FMT, 0>::MX {
                    let ids = (ids0 + ty * 8 + tx / 4) % c.mmq_x;
                    let kby = tx % 4;
                    let col_y_eff = imin(col_dst_0 + ids, ncols_y - 1);
                    let src = vy.offset((col_y_eff * bpc + ib0 * (c.qk / 32) + ir * 4 + kby) as isize * 36);
                    let w = ldg32(src);
                    *y_ds.offset((ids * 4 + kby) as isize) = if c.need_sum { w } else { h2_lo(w).to_bits() };
                    ids0 += 32;
                }
                thread::sync_threads();
                let mut k = ir * 32 / c.qr;
                while k < (ir + 1) * 32 / c.qr {
                    let mut jj = 0;
                    #[unroll]
                    while jj < XJ {
                        let mut ii = 0;
                        #[unroll]
                        while ii < YI {
                            sum[ii][jj] = mmq_acc::<FMT>(sum[ii][jj], sm, tx as usize + 32 * ii, ty as usize + 4 * jj, k as usize);
                            ii += 1;
                        }
                        jj += 1;
                    }
                    k += c.vdr;
                }
                thread::sync_threads();
                ir += 1;
            }
            ib0 += bpw;
        }
        let mut jj = 0;
        #[unroll]
        while jj < XJ {
            let col_dst = col_dst_0 + 4 * jj as i32 + ty;
            if col_dst >= ncols_y {
                return;
            }
            let mut ii = 0;
            #[unroll]
            while ii < YI {
                let row_dst = row_dst_0 + tx + 32 * ii as i32;
                if row_dst < nrows_dst {
                    *dst.offset((col_dst * nrows_dst + row_dst) as isize) = sum[ii][jj];
                }
                ii += 1;
            }
            jj += 1;
        }
    }

    #[kernel] pub unsafe fn mul_mat_q4_0(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, ncols_y: i32, nrows_y: i32, nrows_dst: i32) { mmq::<0, 4, 16>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_q4_1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, ncols_y: i32, nrows_y: i32, nrows_dst: i32) { mmq::<1, 4, 16>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_q5_0(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, ncols_y: i32, nrows_y: i32, nrows_dst: i32) { mmq::<2, 2, 32>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_q5_1(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, ncols_y: i32, nrows_y: i32, nrows_dst: i32) { mmq::<3, 2, 32>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_q8_0(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, ncols_y: i32, nrows_y: i32, nrows_dst: i32) { mmq::<4, 2, 32>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_q2_K(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, ncols_y: i32, nrows_y: i32, nrows_dst: i32) { mmq::<5, 4, 16>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_q3_K(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, ncols_y: i32, nrows_y: i32, nrows_dst: i32) { mmq::<6, 4, 32>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_q4_K(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, ncols_y: i32, nrows_y: i32, nrows_dst: i32) { mmq::<7, 4, 16>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_q5_K(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, ncols_y: i32, nrows_y: i32, nrows_dst: i32) { mmq::<8, 4, 16>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst) }
    #[kernel] pub unsafe fn mul_mat_q6_K(vx: *const u8, vy: *const u8, dst: *mut f32, ncols_x: i32, nrows_x: i32, ncols_y: i32, nrows_y: i32, nrows_dst: i32) { mmq::<9, 2, 16>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst) }
}

fn main() {
    let ok = gate::run(std::env::args().skip(1).collect());
    std::process::exit(if ok { 0 } else { 1 });
}
