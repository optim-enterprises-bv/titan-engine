//! candle-kernels `conv.cu` in cuda-oxide: same 66 entry names, same raw-pointer ABI,
//! bit-identical output (checked by the host `main` against candle's own nvcc PTX).
//!
//! Semantics copied from the reference PTX/SASS, not guessed:
//! - the thread index is `blockIdx.x * blockDim.x + threadIdx.x` in 32 bits, then widened to size_t;
//!   all other index maths is size_t (u64) and wraps; every 64-bit `/` and `%` reproduces nvcc's
//!   div.u32 bypass (`(a | b) >> 32 == 0`), which is what the reference PTX does.
//! - conv / conv_transpose: loop order as in the source (1d: offset, c_in; 2d: w, h, c_in), one
//!   `fma.rn` per product (nvcc -fmad contracts `d += x * k`); half types accumulate in f32
//!   (cvt.f32.{bf16,f16} in, cvt.rn.{bf16,f16}.f32 out); u8/u32 accumulate in the element type (wraps).
//! - conv_transpose uses C `int` arithmetic: inp_x_stride is the low 32 bits of the size_t
//!   expression, `% stride` and `/ stride` are 64-bit (bypassed) on the sign-extended value.
//! - avg_pool2d: scale = (float)(1.0 / (double)(w_k * h_k)) (rcp.rn.f64, cvt.rn.f32.f64), a plain
//!   sum (no fma), then `d * scale`: f32 mul for f32/half/u8/u32 (ints via cvt.rn.f32 and back via
//!   cvt.rzi.u32.f32, truncated), f64 mul by the widened float scale for f64.
//! - max_pool2d: maxg = max.f32 / max.f64 (NaN-ignoring), max.NaN.{bf16,f16}, integer max; 0 if no tap.
//! - upsample_nearest2d: cvt.rn.f64.u64, mul.f64, cvt.rzi.u64.f64 (saturating), clamp to in-1.
//! - upsample_bilinear2d: f64 throughout, bools are any-non-zero bytes, 1/x is rcp.rn.f64,
//!   `s * (i + 0.5) - 0.5` and the lerps `a * (1 - w) + b * w` are contracted to fma.rn.f64 with the
//!   `b * w` product as the addend; result cvt.rn.{f32,bf16,f16}.f64 / cvt.rzi.u32.f64.
//! - col2im1d: `int` l_out_idx / l_in_idx / k0 exactly as in C; accumulation starts from +0 and
//!   stores after every add (add.bf16 / add.f16 / add.f32 / add.f64 / wrapping ints).
#![allow(unsafe_op_in_unsafe_fn)]
use cuda_device::{kernel, ptx_asm, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    // ---------------------------------------------------------------- integer helpers
    #[inline(always)]
    fn tid() -> usize {
        thread::blockIdx_x().wrapping_mul(thread::blockDim_x()).wrapping_add(thread::threadIdx_x()) as usize
    }
    #[inline(always)]
    fn div_u32(x: u32, y: u32) -> u32 {
        let r: u32;
        unsafe { ptx_asm!("div.u32 %0, %1, %2;", out("=r") r, in("r") x, in("r") y, options(register_only)); }
        r
    }
    #[inline(always)]
    fn rem_u32(x: u32, y: u32) -> u32 {
        let r: u32;
        unsafe { ptx_asm!("rem.u32 %0, %1, %2;", out("=r") r, in("r") x, in("r") y, options(register_only)); }
        r
    }
    #[inline(always)]
    fn div_u64(x: usize, y: usize) -> usize {
        let r: usize;
        unsafe { ptx_asm!("div.u64 %0, %1, %2;", out("=l") r, in("l") x, in("l") y, options(register_only)); }
        r
    }
    #[inline(always)]
    fn rem_u64(x: usize, y: usize) -> usize {
        let r: usize;
        unsafe { ptx_asm!("rem.u64 %0, %1, %2;", out("=l") r, in("l") x, in("l") y, options(register_only)); }
        r
    }
    /// size_t `/` with nvcc's 32-bit bypass.
    #[inline(always)]
    fn udiv(x: usize, y: usize) -> usize {
        if (x | y) >> 32 == 0 { div_u32(x as u32, y as u32) as usize } else { div_u64(x, y) }
    }
    /// size_t `%` with nvcc's 32-bit bypass.
    #[inline(always)]
    fn urem(x: usize, y: usize) -> usize {
        if (x | y) >> 32 == 0 { rem_u32(x as u32, y as u32) as usize } else { rem_u64(x, y) }
    }
    /// C conversion int -> size_t (sign extension).
    #[inline(always)]
    fn sx(x: i32) -> usize { x as i64 as u64 as usize }

    // ---------------------------------------------------------------- float helpers (exact PTX)
    #[inline(always)]
    pub fn bf16_f32(x: u16) -> f32 { let r: f32; unsafe { ptx_asm!("cvt.f32.bf16 %0, %1;", out("=f") r, in("h") x, options(register_only)); } r }
    #[inline(always)]
    pub fn f16_f32(x: u16) -> f32 { let r: f32; unsafe { ptx_asm!("cvt.f32.f16 %0, %1;", out("=f") r, in("h") x, options(register_only)); } r }
    #[inline(always)]
    pub fn f32_bf16(x: f32) -> u16 { let r: u16; unsafe { ptx_asm!("cvt.rn.bf16.f32 %0, %1;", out("=h") r, in("f") x, options(register_only)); } r }
    #[inline(always)]
    pub fn f32_f16(x: f32) -> u16 { let r: u16; unsafe { ptx_asm!("cvt.rn.f16.f32 %0, %1;", out("=h") r, in("f") x, options(register_only)); } r }
    #[inline(always)]
    pub fn f64_bf16(x: f64) -> u16 { let r: u16; unsafe { ptx_asm!("cvt.rn.bf16.f64 %0, %1;", out("=h") r, in("d") x, options(register_only)); } r }
    #[inline(always)]
    pub fn f64_f16(x: f64) -> u16 { let r: u16; unsafe { ptx_asm!("cvt.rn.f16.f64 %0, %1;", out("=h") r, in("d") x, options(register_only)); } r }
    #[inline(always)]
    pub fn f32_f64(x: f32) -> f64 { let r: f64; unsafe { ptx_asm!("cvt.f64.f32 %0, %1;", out("=d") r, in("f") x, options(register_only)); } r }
    #[inline(always)]
    pub fn f64_f32(x: f64) -> f32 { let r: f32; unsafe { ptx_asm!("cvt.rn.f32.f64 %0, %1;", out("=f") r, in("d") x, options(register_only)); } r }
    #[inline(always)]
    pub fn f32_u32(x: f32) -> u32 { let r: u32; unsafe { ptx_asm!("cvt.rzi.u32.f32 %0, %1;", out("=r") r, in("f") x, options(register_only)); } r }
    #[inline(always)]
    pub fn f64_u32(x: f64) -> u32 { let r: u32; unsafe { ptx_asm!("cvt.rzi.u32.f64 %0, %1;", out("=r") r, in("d") x, options(register_only)); } r }
    #[inline(always)]
    pub fn f64_u64(x: f64) -> usize { let r: usize; unsafe { ptx_asm!("cvt.rzi.u64.f64 %0, %1;", out("=l") r, in("d") x, options(register_only)); } r }
    #[inline(always)]
    pub fn u16_f32(x: u16) -> f32 { let r: f32; unsafe { ptx_asm!("cvt.rn.f32.u16 %0, %1;", out("=f") r, in("h") x, options(register_only)); } r }
    #[inline(always)]
    pub fn u32_f32(x: u32) -> f32 { let r: f32; unsafe { ptx_asm!("cvt.rn.f32.u32 %0, %1;", out("=f") r, in("r") x, options(register_only)); } r }
    #[inline(always)]
    pub fn u16_f64(x: u16) -> f64 { let r: f64; unsafe { ptx_asm!("cvt.rn.f64.u16 %0, %1;", out("=d") r, in("h") x, options(register_only)); } r }
    #[inline(always)]
    pub fn u32_f64(x: u32) -> f64 { let r: f64; unsafe { ptx_asm!("cvt.rn.f64.u32 %0, %1;", out("=d") r, in("r") x, options(register_only)); } r }
    #[inline(always)]
    pub fn u64_f64(x: usize) -> f64 { let r: f64; unsafe { ptx_asm!("cvt.rn.f64.u64 %0, %1;", out("=d") r, in("l") x, options(register_only)); } r }
    #[inline(always)]
    pub fn floor_f64(x: f64) -> f64 { let r: f64; unsafe { ptx_asm!("cvt.rmi.f64.f64 %0, %1;", out("=d") r, in("d") x, options(register_only)); } r }
    #[inline(always)]
    pub fn rcp_f64(x: f64) -> f64 { let r: f64; unsafe { ptx_asm!("rcp.rn.f64 %0, %1;", out("=d") r, in("d") x, options(register_only)); } r }
    #[inline(always)]
    pub fn div_f64(x: f64, y: f64) -> f64 { let r: f64; unsafe { ptx_asm!("div.rn.f64 %0, %1, %2;", out("=d") r, in("d") x, in("d") y, options(register_only)); } r }
    #[inline(always)]
    pub fn add_f64(x: f64, y: f64) -> f64 { let r: f64; unsafe { ptx_asm!("add.rn.f64 %0, %1, %2;", out("=d") r, in("d") x, in("d") y, options(register_only)); } r }
    #[inline(always)]
    pub fn sub_f64(x: f64, y: f64) -> f64 { let r: f64; unsafe { ptx_asm!("sub.rn.f64 %0, %1, %2;", out("=d") r, in("d") x, in("d") y, options(register_only)); } r }
    #[inline(always)]
    pub fn mul_f64(x: f64, y: f64) -> f64 { let r: f64; unsafe { ptx_asm!("mul.rn.f64 %0, %1, %2;", out("=d") r, in("d") x, in("d") y, options(register_only)); } r }
    #[inline(always)]
    pub fn fma_f64(x: f64, y: f64, z: f64) -> f64 { let r: f64; unsafe { ptx_asm!("fma.rn.f64 %0, %1, %2, %3;", out("=d") r, in("d") x, in("d") y, in("d") z, options(register_only)); } r }
    #[inline(always)]
    pub fn max_f64(x: f64, y: f64) -> f64 { let r: f64; unsafe { ptx_asm!("max.f64 %0, %1, %2;", out("=d") r, in("d") x, in("d") y, options(register_only)); } r }
    #[inline(always)]
    pub fn min_f64(x: f64, y: f64) -> f64 { let r: f64; unsafe { ptx_asm!("min.f64 %0, %1, %2;", out("=d") r, in("d") x, in("d") y, options(register_only)); } r }
    #[inline(always)]
    pub fn add_f32(x: f32, y: f32) -> f32 { let r: f32; unsafe { ptx_asm!("add.rn.f32 %0, %1, %2;", out("=f") r, in("f") x, in("f") y, options(register_only)); } r }
    #[inline(always)]
    pub fn mul_f32(x: f32, y: f32) -> f32 { let r: f32; unsafe { ptx_asm!("mul.rn.f32 %0, %1, %2;", out("=f") r, in("f") x, in("f") y, options(register_only)); } r }
    #[inline(always)]
    pub fn fma_f32(x: f32, y: f32, z: f32) -> f32 { let r: f32; unsafe { ptx_asm!("fma.rn.f32 %0, %1, %2, %3;", out("=f") r, in("f") x, in("f") y, in("f") z, options(register_only)); } r }
    #[inline(always)]
    pub fn max_f32(x: f32, y: f32) -> f32 { let r: f32; unsafe { ptx_asm!("max.f32 %0, %1, %2;", out("=f") r, in("f") x, in("f") y, options(register_only)); } r }
    #[inline(always)]
    pub fn max_bf16(x: u16, y: u16) -> u16 { let r: u16; unsafe { ptx_asm!("max.NaN.bf16 %0, %1, %2;", out("=h") r, in("h") x, in("h") y, options(register_only)); } r }
    #[inline(always)]
    pub fn max_f16(x: u16, y: u16) -> u16 { let r: u16; unsafe { ptx_asm!("max.NaN.f16 %0, %1, %2;", out("=h") r, in("h") x, in("h") y, options(register_only)); } r }
    #[inline(always)]
    pub fn add_bf16(x: u16, y: u16) -> u16 { let r: u16; unsafe { ptx_asm!("add.bf16 %0, %1, %2;", out("=h") r, in("h") x, in("h") y, options(register_only)); } r }
    #[inline(always)]
    pub fn add_f16(x: u16, y: u16) -> u16 { let r: u16; unsafe { ptx_asm!("add.f16 %0, %1, %2;", out("=h") r, in("h") x, in("h") y, options(register_only)); } r }

    // ---------------------------------------------------------------- f64 NaN payloads
    // f32/f16/bf16 arithmetic returns the canonical NaN, but f64 (DADD/DMUL/DFMA/DSETP.MAX) keeps a
    // NaN operand's payload (quieted), chosen by SASS operand POSITION: add/mul/max: b, then a;
    // fma: b, then c, then a. ptxas freely commutes these operands, so the payload of an f64 NaN
    // result depends on its register allocation. These helpers pin the reference's SASS operand
    // order: the arithmetic result is used only when no operand is NaN.
    const QUIET: u64 = 0x0008_0000_0000_0000;
    #[inline(always)]
    fn quiet(x: f64) -> f64 { f64::from_bits(x.to_bits() | QUIET) }
    /// `DADD a, b`
    #[inline(always)]
    pub fn sass_dadd(a: f64, b: f64) -> f64 {
        let r = add_f64(a, b);
        if b != b { quiet(b) } else if a != a { quiet(a) } else { r }
    }
    /// `DFMA a, b, c`
    #[inline(always)]
    pub fn sass_dfma(a: f64, b: f64, c: f64) -> f64 {
        let r = fma_f64(a, b, c);
        if b != b { quiet(b) } else if c != c { quiet(c) } else if a != a { quiet(a) } else { r }
    }
    /// `max.f64 a, b` (DSETP.MAX + selects): a NaN operand loses to a number; two NaNs give quiet(b).
    #[inline(always)]
    pub fn sass_dmax(a: f64, b: f64) -> f64 {
        let r = max_f64(a, b);
        if a != a && b != b { quiet(b) } else { r }
    }

    // ---------------------------------------------------------------- per-dtype pieces
    // conv multiply-accumulate: d + x * k
    #[inline(always)]
    pub fn mac_f32(x: f32, k: f32, d: f32) -> f32 { fma_f32(x, k, d) }
    #[inline(always)]
    pub fn mac_f64(x: f64, k: f64, d: f64) -> f64 { sass_dfma(x, k, d) }
    #[inline(always)]
    pub fn mac_u8(x: u8, k: u8, d: u8) -> u8 { d.wrapping_add(x.wrapping_mul(k)) }
    #[inline(always)]
    pub fn mac_u32(x: u32, k: u32, d: u32) -> u32 { d.wrapping_add(x.wrapping_mul(k)) }
    #[inline(always)]
    pub fn id<X>(x: X) -> X { x }
    #[inline(always)]
    pub fn add_u8(x: u8, y: u8) -> u8 { x.wrapping_add(y) }
    #[inline(always)]
    pub fn add_u32(x: u32, y: u32) -> u32 { x.wrapping_add(y) }
    #[inline(always)]
    pub fn max_u8(x: u8, y: u8) -> u8 { if x > y { x } else { y } }
    #[inline(always)]
    pub fn max_u32(x: u32, y: u32) -> u32 { if x > y { x } else { y } }
    // avg_pool final `d * scale`
    #[inline(always)]
    pub fn avg_bf16(d: f32, s: f32) -> u16 { f32_bf16(mul_f32(d, s)) }
    #[inline(always)]
    pub fn avg_f16(d: f32, s: f32) -> u16 { f32_f16(mul_f32(d, s)) }
    #[inline(always)]
    pub fn avg_f32(d: f32, s: f32) -> f32 { mul_f32(d, s) }
    #[inline(always)]
    pub fn avg_f64(d: f64, s: f32) -> f64 { mul_f64(d, f32_f64(s)) }
    #[inline(always)]
    pub fn avg_u8(d: u8, s: f32) -> u8 { f32_u32(mul_f32(u16_f32(d as u16), s)) as u8 }
    #[inline(always)]
    pub fn avg_u32(d: u32, s: f32) -> u32 { f32_u32(mul_f32(u32_f32(d), s)) }
    // bilinear element <-> double
    #[inline(always)]
    pub fn bf16_f64(x: u16) -> f64 { f32_f64(bf16_f32(x)) }
    #[inline(always)]
    pub fn f16_f64(x: u16) -> f64 { f32_f64(f16_f32(x)) }
    #[inline(always)]
    pub fn u8_f64(x: u8) -> f64 { u16_f64(x as u16) }
    #[inline(always)]
    pub fn f64_u8(x: f64) -> u8 { f64_u32(x) as u8 }

    // ---------------------------------------------------------------- the eleven templates
    #[inline(always)]
    pub unsafe fn conv1d<T: Copy, A: Copy, L: Fn(T) -> A, M: Fn(A, A, A) -> A, S: Fn(A) -> T>(
        l_out: usize, stride: usize, padding: usize, dilation: usize, info: *const usize,
        src: *const T, kernel: *const T, dst: *mut T, zero: A, load: L, mac: M, store: S,
    ) {
        let src_dims = info;
        let src_s = info.wrapping_add(3);
        let k_dims = info.wrapping_add(6);
        let k_s = info.wrapping_add(9);
        let dst_i = tid();
        let k_size = *k_dims.add(2);
        let c_out = *k_dims;
        let c_in = *src_dims.add(1);
        let l_in = *src_dims.add(2);
        if dst_i >= (*src_dims).wrapping_mul(c_out).wrapping_mul(l_out) {
            return;
        }
        let b_idx = udiv(dst_i, l_out.wrapping_mul(c_out));
        let dst_c_idx = urem(udiv(dst_i, l_out), c_out);
        let dst_l = urem(dst_i, l_out);
        let src_idx0 = b_idx.wrapping_mul(*src_s);
        let mut d = zero;
        let mut offset: usize = 0;
        while offset < k_size {
            let src_l = stride.wrapping_mul(dst_l).wrapping_add(offset).wrapping_mul(dilation);
            if !(src_l < padding || src_l >= padding.wrapping_add(l_in)) {
                let src_l = src_l.wrapping_sub(padding);
                let mut c: usize = 0;
                while c < c_in {
                    let src_idx = src_idx0.wrapping_add(c.wrapping_mul(*src_s.add(1))).wrapping_add(src_l.wrapping_mul(*src_s.add(2)));
                    let k_idx = dst_c_idx.wrapping_mul(*k_s).wrapping_add(c.wrapping_mul(*k_s.add(1))).wrapping_add(offset.wrapping_mul(*k_s.add(2)));
                    d = mac(load(*src.add(src_idx)), load(*kernel.add(k_idx)), d);
                    c += 1;
                }
            }
            offset += 1;
        }
        *dst.add(dst_i) = store(d);
    }

    #[inline(always)]
    pub unsafe fn conv2d<T: Copy, A: Copy, L: Fn(T) -> A, M: Fn(A, A, A) -> A, S: Fn(A) -> T>(
        w_out: usize, h_out: usize, stride: usize, padding: usize, dilation: usize, info: *const usize,
        src: *const T, kernel: *const T, dst: *mut T, zero: A, load: L, mac: M, store: S,
    ) {
        let dst_i = tid();
        let src_dims = info;
        let src_s = info.wrapping_add(4);
        let k_dims = info.wrapping_add(8);
        let k_s = info.wrapping_add(12);
        let h_k = *k_dims.add(2);
        let w_k = *k_dims.add(3);
        let c_out = *k_dims;
        let c_in = *src_dims.add(1);
        let h_in = *src_dims.add(2);
        let w_in = *src_dims.add(3);
        if dst_i >= (*src_dims).wrapping_mul(c_out).wrapping_mul(w_out).wrapping_mul(h_out) {
            return;
        }
        let b_idx = udiv(dst_i, w_out.wrapping_mul(h_out).wrapping_mul(c_out));
        let dst_c_idx = urem(udiv(dst_i, w_out.wrapping_mul(h_out)), c_out);
        let dst_h = urem(udiv(dst_i, w_out), h_out);
        let dst_w = urem(dst_i, w_out);
        let src_idx0 = b_idx.wrapping_mul(*src_s);
        let mut d = zero;
        let mut w_off: usize = 0;
        while w_off < w_k {
            let src_w = stride.wrapping_mul(dst_w).wrapping_add(w_off.wrapping_mul(dilation));
            if !(src_w < padding || src_w >= w_in.wrapping_add(padding)) {
                let src_w = src_w.wrapping_sub(padding);
                let mut h_off: usize = 0;
                while h_off < h_k {
                    let src_h = stride.wrapping_mul(dst_h).wrapping_add(h_off.wrapping_mul(dilation));
                    if !(src_h < padding || src_h >= h_in.wrapping_add(padding)) {
                        let src_h = src_h.wrapping_sub(padding);
                        let mut c: usize = 0;
                        while c < c_in {
                            let src_idx = src_idx0
                                .wrapping_add(c.wrapping_mul(*src_s.add(1)))
                                .wrapping_add(src_h.wrapping_mul(*src_s.add(2)))
                                .wrapping_add(src_w.wrapping_mul(*src_s.add(3)));
                            let k_idx = dst_c_idx
                                .wrapping_mul(*k_s)
                                .wrapping_add(c.wrapping_mul(*k_s.add(1)))
                                .wrapping_add(h_off.wrapping_mul(*k_s.add(2)))
                                .wrapping_add(w_off.wrapping_mul(*k_s.add(3)));
                            d = mac(load(*src.add(src_idx)), load(*kernel.add(k_idx)), d);
                            c += 1;
                        }
                    }
                    h_off += 1;
                }
            }
            w_off += 1;
        }
        *dst.add(dst_i) = store(d);
    }

    /// C: `int s = (int)(o + padding) - k * dilation; if (s < 0 || s % stride) skip; int i = s / stride;`
    /// Returns the (sign-extended) input position, or None when the tap is skipped.
    #[inline(always)]
    fn convt_pos(o: usize, padding: usize, k: i32, dilation: usize, stride: usize) -> Option<usize> {
        let s = o.wrapping_add(padding).wrapping_sub(sx(k).wrapping_mul(dilation)) as u32 as i32;
        if s < 0 || urem(sx(s), stride) != 0 {
            return None;
        }
        Some(sx(udiv(sx(s), stride) as u32 as i32))
    }

    #[inline(always)]
    pub unsafe fn conv_transpose1d<T: Copy, A: Copy, L: Fn(T) -> A, M: Fn(A, A, A) -> A, S: Fn(A) -> T>(
        l_out: usize, stride: usize, padding: usize, _out_padding: usize, dilation: usize, info: *const usize,
        src: *const T, kernel: *const T, dst: *mut T, zero: A, load: L, mac: M, store: S,
    ) {
        let dst_i = tid();
        let src_dims = info;
        let src_s = info.wrapping_add(3);
        let k_dims = info.wrapping_add(6);
        let k_s = info.wrapping_add(9);
        let l_k = *k_dims.add(2);
        let c_out = *k_dims.add(1);
        let c_in = *src_dims.add(1);
        let l_in = *src_dims.add(2);
        if dst_i >= (*src_dims).wrapping_mul(c_out).wrapping_mul(l_out) {
            return;
        }
        let b_idx = udiv(dst_i, l_out.wrapping_mul(c_out));
        let dst_c_idx = urem(udiv(dst_i, l_out), c_out);
        let out_x = urem(dst_i, l_out);
        let src_idx0 = b_idx.wrapping_mul(*src_s);
        let mut d = zero;
        let mut k_x: i32 = 0;
        while k_x < l_k as i32 {
            if let Some(inp_x) = convt_pos(out_x, padding, k_x, dilation, stride) {
                if inp_x < l_in {
                    let mut c: usize = 0;
                    while c < c_in {
                        let src_idx = src_idx0.wrapping_add(c.wrapping_mul(*src_s.add(1))).wrapping_add(inp_x.wrapping_mul(*src_s.add(2)));
                        let k_idx = c.wrapping_mul(*k_s).wrapping_add(dst_c_idx.wrapping_mul(*k_s.add(1))).wrapping_add(sx(k_x).wrapping_mul(*k_s.add(2)));
                        d = mac(load(*src.add(src_idx)), load(*kernel.add(k_idx)), d);
                        c += 1;
                    }
                }
            }
            k_x += 1;
        }
        *dst.add(dst_i) = store(d);
    }

    #[inline(always)]
    pub unsafe fn conv_transpose2d<T: Copy, A: Copy, L: Fn(T) -> A, M: Fn(A, A, A) -> A, S: Fn(A) -> T>(
        w_out: usize, h_out: usize, stride: usize, padding: usize, _out_padding: usize, dilation: usize, info: *const usize,
        src: *const T, kernel: *const T, dst: *mut T, zero: A, load: L, mac: M, store: S,
    ) {
        let dst_i = tid();
        let src_dims = info;
        let src_s = info.wrapping_add(4);
        let k_dims = info.wrapping_add(8);
        let k_s = info.wrapping_add(12);
        let h_k = *k_dims.add(2);
        let w_k = *k_dims.add(3);
        let c_out = *k_dims.add(1);
        let c_in = *src_dims.add(1);
        let h_in = *src_dims.add(2);
        let w_in = *src_dims.add(3);
        if dst_i >= (*src_dims).wrapping_mul(c_out).wrapping_mul(w_out).wrapping_mul(h_out) {
            return;
        }
        let b_idx = udiv(dst_i, w_out.wrapping_mul(h_out).wrapping_mul(c_out));
        let dst_c_idx = urem(udiv(dst_i, w_out.wrapping_mul(h_out)), c_out);
        let out_y = urem(udiv(dst_i, w_out), h_out);
        let out_x = urem(dst_i, w_out);
        let src_idx0 = b_idx.wrapping_mul(*src_s);
        let mut d = zero;
        let mut k_x: i32 = 0;
        while k_x < w_k as i32 {
            if let Some(inp_x) = convt_pos(out_x, padding, k_x, dilation, stride) {
                if inp_x < w_in {
                    let mut k_y: i32 = 0;
                    while k_y < h_k as i32 {
                        if let Some(inp_y) = convt_pos(out_y, padding, k_y, dilation, stride) {
                            if inp_y < h_in {
                                let mut c: usize = 0;
                                while c < c_in {
                                    let src_idx = src_idx0
                                        .wrapping_add(c.wrapping_mul(*src_s.add(1)))
                                        .wrapping_add(inp_y.wrapping_mul(*src_s.add(2)))
                                        .wrapping_add(inp_x.wrapping_mul(*src_s.add(3)));
                                    let k_idx = c
                                        .wrapping_mul(*k_s)
                                        .wrapping_add(dst_c_idx.wrapping_mul(*k_s.add(1)))
                                        .wrapping_add(sx(k_y).wrapping_mul(*k_s.add(2)))
                                        .wrapping_add(sx(k_x).wrapping_mul(*k_s.add(3)));
                                    d = mac(load(*src.add(src_idx)), load(*kernel.add(k_idx)), d);
                                    c += 1;
                                }
                            }
                        }
                        k_y += 1;
                    }
                }
            }
            k_x += 1;
        }
        *dst.add(dst_i) = store(d);
    }

    #[inline(always)]
    pub unsafe fn avg_pool2d<T: Copy, A: Copy, L: Fn(T) -> A, D: Fn(A, A) -> A, F: Fn(A, f32) -> T>(
        w_k: usize, h_k: usize, w_stride: usize, h_stride: usize, info: *const usize,
        src: *const T, dst: *mut T, zero: A, load: L, add: D, fin: F,
    ) {
        let dst_i = tid();
        let src_dims = info;
        let src_s = info.wrapping_add(4);
        let c = *src_dims.add(1);
        let w_in = *src_dims.add(2);
        let h_in = *src_dims.add(3);
        let w_out = udiv(w_in.wrapping_sub(w_k), w_stride).wrapping_add(1);
        let h_out = udiv(h_in.wrapping_sub(h_k), h_stride).wrapping_add(1);
        if dst_i >= (*src_dims).wrapping_mul(c).wrapping_mul(w_out).wrapping_mul(h_out) {
            return;
        }
        let b_idx = udiv(dst_i, w_out.wrapping_mul(h_out).wrapping_mul(c));
        let c_idx = urem(udiv(dst_i, w_out.wrapping_mul(h_out)), c);
        let dst_w = urem(udiv(dst_i, h_out), w_out);
        let dst_h = urem(dst_i, h_out);
        let src_idx0 = b_idx.wrapping_mul(*src_s);
        let scale = f64_f32(rcp_f64(u64_f64(w_k.wrapping_mul(h_k))));
        let mut d = zero;
        let mut w_off: usize = 0;
        while w_off < w_k {
            let src_w = w_stride.wrapping_mul(dst_w).wrapping_add(w_off);
            if src_w < w_in {
                let mut h_off: usize = 0;
                while h_off < h_k {
                    let src_h = h_stride.wrapping_mul(dst_h).wrapping_add(h_off);
                    if src_h < h_in {
                        let src_idx = src_idx0
                            .wrapping_add(c_idx.wrapping_mul(*src_s.add(1)))
                            .wrapping_add(src_w.wrapping_mul(*src_s.add(2)))
                            .wrapping_add(src_h.wrapping_mul(*src_s.add(3)));
                        d = add(d, load(*src.add(src_idx)));
                    }
                    h_off += 1;
                }
            }
            w_off += 1;
        }
        *dst.add(dst_i) = fin(d, scale);
    }

    #[inline(always)]
    pub unsafe fn max_pool2d<T: Copy, M: Fn(T, T) -> T>(
        w_k: usize, h_k: usize, w_stride: usize, h_stride: usize, info: *const usize,
        src: *const T, dst: *mut T, zero: T, max: M,
    ) {
        let dst_i = tid();
        let src_dims = info;
        let src_s = info.wrapping_add(4);
        let c = *src_dims.add(1);
        let w_in = *src_dims.add(2);
        let h_in = *src_dims.add(3);
        let w_out = udiv(w_in.wrapping_sub(w_k), w_stride).wrapping_add(1);
        let h_out = udiv(h_in.wrapping_sub(h_k), h_stride).wrapping_add(1);
        if dst_i >= (*src_dims).wrapping_mul(c).wrapping_mul(w_out).wrapping_mul(h_out) {
            return;
        }
        let b_idx = udiv(dst_i, w_out.wrapping_mul(h_out).wrapping_mul(c));
        let c_idx = urem(udiv(dst_i, w_out.wrapping_mul(h_out)), c);
        let dst_w = urem(udiv(dst_i, h_out), w_out);
        let dst_h = urem(dst_i, h_out);
        let src_idx0 = b_idx.wrapping_mul(*src_s);
        let mut d = zero;
        let mut set = false;
        let mut w_off: usize = 0;
        while w_off < w_k {
            let src_w = w_stride.wrapping_mul(dst_w).wrapping_add(w_off);
            if src_w < w_in {
                let mut h_off: usize = 0;
                while h_off < h_k {
                    let src_h = h_stride.wrapping_mul(dst_h).wrapping_add(h_off);
                    if src_h < h_in {
                        let src_idx = src_idx0
                            .wrapping_add(c_idx.wrapping_mul(*src_s.add(1)))
                            .wrapping_add(src_w.wrapping_mul(*src_s.add(2)))
                            .wrapping_add(src_h.wrapping_mul(*src_s.add(3)));
                        let v = *src.add(src_idx);
                        if set {
                            d = max(d, v);
                        } else {
                            d = v;
                            set = true;
                        }
                    }
                    h_off += 1;
                }
            }
            w_off += 1;
        }
        *dst.add(dst_i) = d;
    }

    #[inline(always)]
    pub unsafe fn upsample_nearest2d<T: Copy>(
        w_out: usize, h_out: usize, w_scale: f64, h_scale: f64, info: *const usize, src: *const T, dst: *mut T,
    ) {
        let dst_i = tid();
        let src_dims = info;
        let src_s = info.wrapping_add(4);
        let c = *src_dims.add(1);
        let w_in = *src_dims.add(2);
        let h_in = *src_dims.add(3);
        if dst_i >= (*src_dims).wrapping_mul(c).wrapping_mul(w_out).wrapping_mul(h_out) {
            return;
        }
        let b_idx = udiv(dst_i, w_out.wrapping_mul(h_out).wrapping_mul(c));
        let c_idx = urem(udiv(dst_i, w_out.wrapping_mul(h_out)), c);
        let dst_w = urem(udiv(dst_i, h_out), w_out);
        let dst_h = urem(dst_i, h_out);
        let mut src_w = f64_u64(mul_f64(u64_f64(dst_w), w_scale));
        let mut src_h = f64_u64(mul_f64(u64_f64(dst_h), h_scale));
        if src_w >= w_in {
            src_w = w_in.wrapping_sub(1);
        }
        if src_h >= h_in {
            src_h = h_in.wrapping_sub(1);
        }
        let src_i = b_idx
            .wrapping_mul(*src_s)
            .wrapping_add(c_idx.wrapping_mul(*src_s.add(1)))
            .wrapping_add(src_w.wrapping_mul(*src_s.add(2)))
            .wrapping_add(src_h.wrapping_mul(*src_s.add(3)));
        *dst.add(dst_i) = *src.add(src_i);
    }

    #[inline(always)]
    pub unsafe fn upsample_bilinear2d<T: Copy, I: Fn(T) -> f64, O: Fn(f64) -> T>(
        w_out: usize, h_out: usize, align_corners: u8, has_scale_h: u8, scale_h_factor: f64,
        has_scale_w: u8, scale_w_factor: f64, info: *const usize, src: *const T, dst: *mut T, to: I, from: O,
    ) {
        let dst_i = tid();
        let src_dims = info;
        let src_s = info.wrapping_add(4);
        let c = *src_dims.add(1);
        let h_in = *src_dims.add(2);
        let w_in = *src_dims.add(3);
        if dst_i >= (*src_dims).wrapping_mul(c).wrapping_mul(h_out).wrapping_mul(w_out) {
            return;
        }
        let b_idx = udiv(dst_i, h_out.wrapping_mul(w_out).wrapping_mul(c));
        let c_idx = urem(udiv(dst_i, h_out.wrapping_mul(w_out)), c);
        let dst_h = urem(udiv(dst_i, w_out), h_out);
        let dst_w = urem(dst_i, w_out);
        let align = align_corners != 0;
        let (h_scale, w_scale) = if align {
            (
                if h_out > 1 { div_f64(u64_f64(h_in.wrapping_sub(1)), u64_f64(h_out.wrapping_sub(1))) } else { 0.0 },
                if w_out > 1 { div_f64(u64_f64(w_in.wrapping_sub(1)), u64_f64(w_out.wrapping_sub(1))) } else { 0.0 },
            )
        } else {
            (
                if has_scale_h != 0 { rcp_f64(scale_h_factor) } else { div_f64(u64_f64(h_in), u64_f64(h_out)) },
                if has_scale_w != 0 { rcp_f64(scale_w_factor) } else { div_f64(u64_f64(w_in), u64_f64(w_out)) },
            )
        };
        let (src_h_fp, src_w_fp) = if align {
            (mul_f64(h_scale, u64_f64(dst_h)), mul_f64(w_scale, u64_f64(dst_w)))
        } else {
            (
                fma_f64(add_f64(u64_f64(dst_h), 0.5), h_scale, -0.5),
                fma_f64(add_f64(u64_f64(dst_w), 0.5), w_scale, -0.5),
            )
        };
        let src_h_fp = max_f64(src_h_fp, 0.0);
        let src_w_fp = max_f64(src_w_fp, 0.0);
        let h0 = f64_u64(floor_f64(src_h_fp));
        let w0 = f64_u64(floor_f64(src_w_fp));
        let h1 = { let a = h0.wrapping_add(1); let b = h_in.wrapping_sub(1); if a < b { a } else { b } };
        let w1 = { let a = w0.wrapping_add(1); let b = w_in.wrapping_sub(1); if a < b { a } else { b } };
        let weight_h = min_f64(max_f64(sub_f64(src_h_fp, u64_f64(h0)), 0.0), 1.0);
        let weight_w = min_f64(max_f64(sub_f64(src_w_fp, u64_f64(w0)), 0.0), 1.0);
        let base = b_idx.wrapping_mul(*src_s).wrapping_add(c_idx.wrapping_mul(*src_s.add(1)));
        let s2 = *src_s.add(2);
        let s3 = *src_s.add(3);
        let v00 = to(*src.add(base.wrapping_add(h0.wrapping_mul(s2)).wrapping_add(w0.wrapping_mul(s3))));
        let v10 = to(*src.add(base.wrapping_add(h0.wrapping_mul(s2)).wrapping_add(w1.wrapping_mul(s3))));
        let v01 = to(*src.add(base.wrapping_add(h1.wrapping_mul(s2)).wrapping_add(w0.wrapping_mul(s3))));
        let v11 = to(*src.add(base.wrapping_add(h1.wrapping_mul(s2)).wrapping_add(w1.wrapping_mul(s3))));
        let omw = sub_f64(1.0, weight_w);
        let v_top = sass_dfma(v00, omw, mul_f64(weight_w, v10));
        let v_bottom = sass_dfma(v01, omw, mul_f64(weight_w, v11));
        let value = sass_dfma(v_top, sub_f64(1.0, weight_h), mul_f64(weight_h, v_bottom));
        *dst.add(dst_i) = from(value);
    }

    #[inline(always)]
    pub unsafe fn im2col1d<T: Copy>(
        numel: usize, l_out: usize, l_k: usize, stride: usize, padding: usize, dilation: usize,
        info: *const usize, src: *const T, dst: *mut T, zero: T,
    ) {
        let thread_i = tid();
        if thread_i >= numel {
            return;
        }
        let src_dims = info;
        let src_s = info.wrapping_add(3);
        let c_in = *src_dims.add(1);
        let l_in = *src_dims.add(2);
        let dst_s1 = c_in;
        let dst_s0 = l_out.wrapping_mul(dst_s1);
        let mut tmp = thread_i;
        let b_idx = udiv(tmp, dst_s0);
        tmp = tmp.wrapping_sub(b_idx.wrapping_mul(dst_s0));
        let l_idx = udiv(tmp, dst_s1);
        tmp = tmp.wrapping_sub(l_idx.wrapping_mul(dst_s1));
        let c_idx = tmp;
        let mut l_k_idx: usize = 0;
        while l_k_idx < l_k {
            let src_l_idx = l_idx.wrapping_mul(stride).wrapping_add(l_k_idx.wrapping_mul(dilation));
            let dst_i = thread_i.wrapping_mul(l_k).wrapping_add(l_k_idx);
            if src_l_idx < padding || src_l_idx >= l_in.wrapping_add(padding) {
                *dst.add(dst_i) = zero;
            } else {
                let src_l_idx = src_l_idx.wrapping_sub(padding);
                let src_i = b_idx
                    .wrapping_mul(*src_s)
                    .wrapping_add(c_idx.wrapping_mul(*src_s.add(1)))
                    .wrapping_add(src_l_idx.wrapping_mul(*src_s.add(2)));
                *dst.add(dst_i) = *src.add(src_i);
            }
            l_k_idx += 1;
        }
    }

    #[inline(always)]
    pub unsafe fn col2im1d<T: Copy, D: Fn(T, T) -> T>(
        dst_el: usize, l_out: usize, l_in: usize, c_out: usize, k_size: usize, stride: usize,
        src: *const T, dst: *mut T, zero: T, add: D,
    ) {
        let dst_i = tid();
        if dst_i >= dst_el {
            return;
        }
        let dst_s0 = c_out.wrapping_mul(l_out);
        let dst_s1 = l_out;
        let src_s0 = c_out.wrapping_mul(k_size).wrapping_mul(l_in);
        let src_s1 = c_out.wrapping_mul(k_size);
        let src_s2 = k_size;
        let mut tmp = dst_i;
        let b_idx = udiv(tmp, dst_s0);
        tmp = tmp.wrapping_sub(b_idx.wrapping_mul(dst_s0));
        let c_idx = udiv(tmp, dst_s1);
        tmp = tmp.wrapping_sub(c_idx.wrapping_mul(dst_s1));
        let l_out_idx = tmp as u32 as i32;
        let mut acc = zero;
        *dst.add(dst_i) = acc;
        let mut l_in_idx = udiv(sx(l_out_idx), stride) as u32 as i32;
        let mut k0 = sx(l_out_idx).wrapping_sub(sx(l_in_idx).wrapping_mul(stride)) as u32 as i32;
        while sx(k0) < k_size && l_in_idx >= 0 {
            if sx(l_in_idx) < l_in {
                let src_i = b_idx
                    .wrapping_mul(src_s0)
                    .wrapping_add(sx(l_in_idx).wrapping_mul(src_s1))
                    .wrapping_add(c_idx.wrapping_mul(src_s2))
                    .wrapping_add(sx(k0));
                acc = add(acc, *src.add(src_i));
                *dst.add(dst_i) = acc;
            }
            k0 = sx(k0).wrapping_add(stride) as u32 as i32;
            l_in_idx = l_in_idx.wrapping_sub(1);
        }
    }

    #[inline(always)]
    pub unsafe fn im2col<T: Copy>(
        dst_numel: usize, h_out: usize, w_out: usize, h_k: usize, w_k: usize, stride: usize, padding: usize,
        dilation: usize, info: *const usize, src: *const T, dst: *mut T, zero: T,
    ) {
        let dst_i = tid();
        if dst_i >= dst_numel {
            return;
        }
        let src_dims = info;
        let src_s = info.wrapping_add(4);
        let c_in = *src_dims.add(1);
        let h_in = *src_dims.add(2);
        let w_in = *src_dims.add(3);
        let dst_s4 = w_k;
        let dst_s3 = h_k.wrapping_mul(dst_s4);
        let dst_s2 = c_in.wrapping_mul(dst_s3);
        let dst_s1 = w_out.wrapping_mul(dst_s2);
        let dst_s0 = h_out.wrapping_mul(dst_s1);
        let mut tmp = dst_i;
        let b_idx = udiv(tmp, dst_s0);
        tmp = tmp.wrapping_sub(b_idx.wrapping_mul(dst_s0));
        let h_idx = udiv(tmp, dst_s1);
        tmp = tmp.wrapping_sub(h_idx.wrapping_mul(dst_s1));
        let w_idx = udiv(tmp, dst_s2);
        tmp = tmp.wrapping_sub(w_idx.wrapping_mul(dst_s2));
        let c_idx = udiv(tmp, dst_s3);
        tmp = tmp.wrapping_sub(c_idx.wrapping_mul(dst_s3));
        let h_k_idx = udiv(tmp, dst_s4);
        tmp = tmp.wrapping_sub(h_k_idx.wrapping_mul(dst_s4));
        let w_k_idx = tmp;
        let src_h_idx = h_idx.wrapping_mul(stride).wrapping_add(h_k_idx.wrapping_mul(dilation));
        let src_w_idx = w_idx.wrapping_mul(stride).wrapping_add(w_k_idx.wrapping_mul(dilation));
        if src_h_idx < padding || src_h_idx >= h_in.wrapping_add(padding) {
            *dst.add(dst_i) = zero;
        } else if src_w_idx < padding || src_w_idx >= w_in.wrapping_add(padding) {
            *dst.add(dst_i) = zero;
        } else {
            let src_h_idx = src_h_idx.wrapping_sub(padding);
            let src_w_idx = src_w_idx.wrapping_sub(padding);
            let src_i = b_idx
                .wrapping_mul(*src_s)
                .wrapping_add(c_idx.wrapping_mul(*src_s.add(1)))
                .wrapping_add(src_h_idx.wrapping_mul(*src_s.add(2)))
                .wrapping_add(src_w_idx.wrapping_mul(*src_s.add(3)));
            *dst.add(dst_i) = *src.add(src_i);
        }
    }

    // GENERATED KERNELS BEGIN
    #[kernel]
    pub unsafe fn conv1d_bf16(_src_numel: usize, l_out: usize, stride: usize, padding: usize, dilation: usize, info: *const usize, src: *const u16, kernel: *const u16, dst: *mut u16) {
        unsafe { conv1d(l_out, stride, padding, dilation, info, src, kernel, dst, 0.0f32, bf16_f32, mac_f32, f32_bf16); }
    }

    #[kernel]
    pub unsafe fn conv2d_bf16(_src_numel: usize, w_out: usize, h_out: usize, stride: usize, padding: usize, dilation: usize, info: *const usize, src: *const u16, kernel: *const u16, dst: *mut u16) {
        unsafe { conv2d(w_out, h_out, stride, padding, dilation, info, src, kernel, dst, 0.0f32, bf16_f32, mac_f32, f32_bf16); }
    }

    #[kernel]
    pub unsafe fn conv_transpose1d_bf16(_src_numel: usize, l_out: usize, stride: usize, padding: usize, out_padding: usize, dilation: usize, info: *const usize, src: *const u16, kernel: *const u16, dst: *mut u16) {
        unsafe { conv_transpose1d(l_out, stride, padding, out_padding, dilation, info, src, kernel, dst, 0.0f32, bf16_f32, mac_f32, f32_bf16); }
    }

    #[kernel]
    pub unsafe fn conv_transpose2d_bf16(_src_numel: usize, w_out: usize, h_out: usize, stride: usize, padding: usize, out_padding: usize, dilation: usize, info: *const usize, src: *const u16, kernel: *const u16, dst: *mut u16) {
        unsafe { conv_transpose2d(w_out, h_out, stride, padding, out_padding, dilation, info, src, kernel, dst, 0.0f32, bf16_f32, mac_f32, f32_bf16); }
    }

    #[kernel]
    pub unsafe fn avg_pool2d_bf16(_src_numel: usize, w_k: usize, h_k: usize, w_stride: usize, h_stride: usize, info: *const usize, src: *const u16, dst: *mut u16) {
        unsafe { avg_pool2d(w_k, h_k, w_stride, h_stride, info, src, dst, 0.0f32, bf16_f32, add_f32, avg_bf16); }
    }

    #[kernel]
    pub unsafe fn max_pool2d_bf16(_src_numel: usize, w_k: usize, h_k: usize, w_stride: usize, h_stride: usize, info: *const usize, src: *const u16, dst: *mut u16) {
        unsafe { max_pool2d(w_k, h_k, w_stride, h_stride, info, src, dst, 0u16, max_bf16); }
    }

    #[kernel]
    pub unsafe fn upsample_nearest2d_bf16(w_out: usize, h_out: usize, w_scale: f64, h_scale: f64, info: *const usize, src: *const u16, dst: *mut u16) {
        unsafe { upsample_nearest2d(w_out, h_out, w_scale, h_scale, info, src, dst); }
    }

    #[kernel]
    pub unsafe fn upsample_bilinear2d_bf16(w_out: usize, h_out: usize, align_corners: u8, has_scale_h: u8, scale_h_factor: f64, has_scale_w: u8, scale_w_factor: f64, info: *const usize, src: *const u16, dst: *mut u16) {
        unsafe { upsample_bilinear2d(w_out, h_out, align_corners, has_scale_h, scale_h_factor, has_scale_w, scale_w_factor, info, src, dst, bf16_f64, f64_bf16); }
    }

    #[kernel]
    pub unsafe fn im2col_bf16(dst_numel: usize, h_out: usize, w_out: usize, h_k: usize, w_k: usize, stride: usize, padding: usize, dilation: usize, info: *const usize, src: *const u16, dst: *mut u16) {
        unsafe { im2col(dst_numel, h_out, w_out, h_k, w_k, stride, padding, dilation, info, src, dst, 0u16); }
    }

    #[kernel]
    pub unsafe fn im2col1d_bf16(dst_numel: usize, l_out: usize, l_k: usize, stride: usize, padding: usize, dilation: usize, info: *const usize, src: *const u16, dst: *mut u16) {
        unsafe { im2col1d(dst_numel, l_out, l_k, stride, padding, dilation, info, src, dst, 0u16); }
    }

    #[kernel]
    pub unsafe fn col2im1d_bf16(dst_el: usize, l_out: usize, l_in: usize, c_out: usize, k_size: usize, stride: usize, src: *const u16, dst: *mut u16) {
        unsafe { col2im1d(dst_el, l_out, l_in, c_out, k_size, stride, src, dst, 0u16, add_bf16); }
    }

    #[kernel]
    pub unsafe fn conv1d_f16(_src_numel: usize, l_out: usize, stride: usize, padding: usize, dilation: usize, info: *const usize, src: *const u16, kernel: *const u16, dst: *mut u16) {
        unsafe { conv1d(l_out, stride, padding, dilation, info, src, kernel, dst, 0.0f32, f16_f32, mac_f32, f32_f16); }
    }

    #[kernel]
    pub unsafe fn conv2d_f16(_src_numel: usize, w_out: usize, h_out: usize, stride: usize, padding: usize, dilation: usize, info: *const usize, src: *const u16, kernel: *const u16, dst: *mut u16) {
        unsafe { conv2d(w_out, h_out, stride, padding, dilation, info, src, kernel, dst, 0.0f32, f16_f32, mac_f32, f32_f16); }
    }

    #[kernel]
    pub unsafe fn conv_transpose1d_f16(_src_numel: usize, l_out: usize, stride: usize, padding: usize, out_padding: usize, dilation: usize, info: *const usize, src: *const u16, kernel: *const u16, dst: *mut u16) {
        unsafe { conv_transpose1d(l_out, stride, padding, out_padding, dilation, info, src, kernel, dst, 0.0f32, f16_f32, mac_f32, f32_f16); }
    }

    #[kernel]
    pub unsafe fn conv_transpose2d_f16(_src_numel: usize, w_out: usize, h_out: usize, stride: usize, padding: usize, out_padding: usize, dilation: usize, info: *const usize, src: *const u16, kernel: *const u16, dst: *mut u16) {
        unsafe { conv_transpose2d(w_out, h_out, stride, padding, out_padding, dilation, info, src, kernel, dst, 0.0f32, f16_f32, mac_f32, f32_f16); }
    }

    #[kernel]
    pub unsafe fn avg_pool2d_f16(_src_numel: usize, w_k: usize, h_k: usize, w_stride: usize, h_stride: usize, info: *const usize, src: *const u16, dst: *mut u16) {
        unsafe { avg_pool2d(w_k, h_k, w_stride, h_stride, info, src, dst, 0.0f32, f16_f32, add_f32, avg_f16); }
    }

    #[kernel]
    pub unsafe fn max_pool2d_f16(_src_numel: usize, w_k: usize, h_k: usize, w_stride: usize, h_stride: usize, info: *const usize, src: *const u16, dst: *mut u16) {
        unsafe { max_pool2d(w_k, h_k, w_stride, h_stride, info, src, dst, 0u16, max_f16); }
    }

    #[kernel]
    pub unsafe fn upsample_nearest2d_f16(w_out: usize, h_out: usize, w_scale: f64, h_scale: f64, info: *const usize, src: *const u16, dst: *mut u16) {
        unsafe { upsample_nearest2d(w_out, h_out, w_scale, h_scale, info, src, dst); }
    }

    #[kernel]
    pub unsafe fn upsample_bilinear2d_f16(w_out: usize, h_out: usize, align_corners: u8, has_scale_h: u8, scale_h_factor: f64, has_scale_w: u8, scale_w_factor: f64, info: *const usize, src: *const u16, dst: *mut u16) {
        unsafe { upsample_bilinear2d(w_out, h_out, align_corners, has_scale_h, scale_h_factor, has_scale_w, scale_w_factor, info, src, dst, f16_f64, f64_f16); }
    }

    #[kernel]
    pub unsafe fn im2col_f16(dst_numel: usize, h_out: usize, w_out: usize, h_k: usize, w_k: usize, stride: usize, padding: usize, dilation: usize, info: *const usize, src: *const u16, dst: *mut u16) {
        unsafe { im2col(dst_numel, h_out, w_out, h_k, w_k, stride, padding, dilation, info, src, dst, 0u16); }
    }

    #[kernel]
    pub unsafe fn im2col1d_f16(dst_numel: usize, l_out: usize, l_k: usize, stride: usize, padding: usize, dilation: usize, info: *const usize, src: *const u16, dst: *mut u16) {
        unsafe { im2col1d(dst_numel, l_out, l_k, stride, padding, dilation, info, src, dst, 0u16); }
    }

    #[kernel]
    pub unsafe fn col2im1d_f16(dst_el: usize, l_out: usize, l_in: usize, c_out: usize, k_size: usize, stride: usize, src: *const u16, dst: *mut u16) {
        unsafe { col2im1d(dst_el, l_out, l_in, c_out, k_size, stride, src, dst, 0u16, add_f16); }
    }

    #[kernel]
    pub unsafe fn conv1d_f32(_src_numel: usize, l_out: usize, stride: usize, padding: usize, dilation: usize, info: *const usize, src: *const f32, kernel: *const f32, dst: *mut f32) {
        unsafe { conv1d(l_out, stride, padding, dilation, info, src, kernel, dst, 0.0f32, id::<f32>, mac_f32, id::<f32>); }
    }

    #[kernel]
    pub unsafe fn conv1d_f64(_src_numel: usize, l_out: usize, stride: usize, padding: usize, dilation: usize, info: *const usize, src: *const f64, kernel: *const f64, dst: *mut f64) {
        unsafe { conv1d(l_out, stride, padding, dilation, info, src, kernel, dst, 0.0f64, id::<f64>, mac_f64, id::<f64>); }
    }

    #[kernel]
    pub unsafe fn conv1d_u8(_src_numel: usize, l_out: usize, stride: usize, padding: usize, dilation: usize, info: *const usize, src: *const u8, kernel: *const u8, dst: *mut u8) {
        unsafe { conv1d(l_out, stride, padding, dilation, info, src, kernel, dst, 0u8, id::<u8>, mac_u8, id::<u8>); }
    }

    #[kernel]
    pub unsafe fn conv1d_u32(_src_numel: usize, l_out: usize, stride: usize, padding: usize, dilation: usize, info: *const usize, src: *const u32, kernel: *const u32, dst: *mut u32) {
        unsafe { conv1d(l_out, stride, padding, dilation, info, src, kernel, dst, 0u32, id::<u32>, mac_u32, id::<u32>); }
    }

    #[kernel]
    pub unsafe fn conv2d_f32(_src_numel: usize, w_out: usize, h_out: usize, stride: usize, padding: usize, dilation: usize, info: *const usize, src: *const f32, kernel: *const f32, dst: *mut f32) {
        unsafe { conv2d(w_out, h_out, stride, padding, dilation, info, src, kernel, dst, 0.0f32, id::<f32>, mac_f32, id::<f32>); }
    }

    #[kernel]
    pub unsafe fn conv2d_f64(_src_numel: usize, w_out: usize, h_out: usize, stride: usize, padding: usize, dilation: usize, info: *const usize, src: *const f64, kernel: *const f64, dst: *mut f64) {
        unsafe { conv2d(w_out, h_out, stride, padding, dilation, info, src, kernel, dst, 0.0f64, id::<f64>, mac_f64, id::<f64>); }
    }

    #[kernel]
    pub unsafe fn conv2d_u8(_src_numel: usize, w_out: usize, h_out: usize, stride: usize, padding: usize, dilation: usize, info: *const usize, src: *const u8, kernel: *const u8, dst: *mut u8) {
        unsafe { conv2d(w_out, h_out, stride, padding, dilation, info, src, kernel, dst, 0u8, id::<u8>, mac_u8, id::<u8>); }
    }

    #[kernel]
    pub unsafe fn conv2d_u32(_src_numel: usize, w_out: usize, h_out: usize, stride: usize, padding: usize, dilation: usize, info: *const usize, src: *const u32, kernel: *const u32, dst: *mut u32) {
        unsafe { conv2d(w_out, h_out, stride, padding, dilation, info, src, kernel, dst, 0u32, id::<u32>, mac_u32, id::<u32>); }
    }

    #[kernel]
    pub unsafe fn conv_transpose1d_f32(_src_numel: usize, l_out: usize, stride: usize, padding: usize, out_padding: usize, dilation: usize, info: *const usize, src: *const f32, kernel: *const f32, dst: *mut f32) {
        unsafe { conv_transpose1d(l_out, stride, padding, out_padding, dilation, info, src, kernel, dst, 0.0f32, id::<f32>, mac_f32, id::<f32>); }
    }

    #[kernel]
    pub unsafe fn conv_transpose1d_f64(_src_numel: usize, l_out: usize, stride: usize, padding: usize, out_padding: usize, dilation: usize, info: *const usize, src: *const f64, kernel: *const f64, dst: *mut f64) {
        unsafe { conv_transpose1d(l_out, stride, padding, out_padding, dilation, info, src, kernel, dst, 0.0f64, id::<f64>, mac_f64, id::<f64>); }
    }

    #[kernel]
    pub unsafe fn conv_transpose1d_u8(_src_numel: usize, l_out: usize, stride: usize, padding: usize, out_padding: usize, dilation: usize, info: *const usize, src: *const u8, kernel: *const u8, dst: *mut u8) {
        unsafe { conv_transpose1d(l_out, stride, padding, out_padding, dilation, info, src, kernel, dst, 0u8, id::<u8>, mac_u8, id::<u8>); }
    }

    #[kernel]
    pub unsafe fn conv_transpose1d_u32(_src_numel: usize, l_out: usize, stride: usize, padding: usize, out_padding: usize, dilation: usize, info: *const usize, src: *const u32, kernel: *const u32, dst: *mut u32) {
        unsafe { conv_transpose1d(l_out, stride, padding, out_padding, dilation, info, src, kernel, dst, 0u32, id::<u32>, mac_u32, id::<u32>); }
    }

    #[kernel]
    pub unsafe fn conv_transpose2d_f32(_src_numel: usize, w_out: usize, h_out: usize, stride: usize, padding: usize, out_padding: usize, dilation: usize, info: *const usize, src: *const f32, kernel: *const f32, dst: *mut f32) {
        unsafe { conv_transpose2d(w_out, h_out, stride, padding, out_padding, dilation, info, src, kernel, dst, 0.0f32, id::<f32>, mac_f32, id::<f32>); }
    }

    #[kernel]
    pub unsafe fn conv_transpose2d_f64(_src_numel: usize, w_out: usize, h_out: usize, stride: usize, padding: usize, out_padding: usize, dilation: usize, info: *const usize, src: *const f64, kernel: *const f64, dst: *mut f64) {
        unsafe { conv_transpose2d(w_out, h_out, stride, padding, out_padding, dilation, info, src, kernel, dst, 0.0f64, id::<f64>, mac_f64, id::<f64>); }
    }

    #[kernel]
    pub unsafe fn conv_transpose2d_u8(_src_numel: usize, w_out: usize, h_out: usize, stride: usize, padding: usize, out_padding: usize, dilation: usize, info: *const usize, src: *const u8, kernel: *const u8, dst: *mut u8) {
        unsafe { conv_transpose2d(w_out, h_out, stride, padding, out_padding, dilation, info, src, kernel, dst, 0u8, id::<u8>, mac_u8, id::<u8>); }
    }

    #[kernel]
    pub unsafe fn conv_transpose2d_u32(_src_numel: usize, w_out: usize, h_out: usize, stride: usize, padding: usize, out_padding: usize, dilation: usize, info: *const usize, src: *const u32, kernel: *const u32, dst: *mut u32) {
        unsafe { conv_transpose2d(w_out, h_out, stride, padding, out_padding, dilation, info, src, kernel, dst, 0u32, id::<u32>, mac_u32, id::<u32>); }
    }

    #[kernel]
    pub unsafe fn avg_pool2d_f32(_src_numel: usize, w_k: usize, h_k: usize, w_stride: usize, h_stride: usize, info: *const usize, src: *const f32, dst: *mut f32) {
        unsafe { avg_pool2d(w_k, h_k, w_stride, h_stride, info, src, dst, 0.0f32, id::<f32>, add_f32, avg_f32); }
    }

    #[kernel]
    pub unsafe fn avg_pool2d_f64(_src_numel: usize, w_k: usize, h_k: usize, w_stride: usize, h_stride: usize, info: *const usize, src: *const f64, dst: *mut f64) {
        unsafe { avg_pool2d(w_k, h_k, w_stride, h_stride, info, src, dst, 0.0f64, id::<f64>, sass_dadd, avg_f64); }
    }

    #[kernel]
    pub unsafe fn avg_pool2d_u8(_src_numel: usize, w_k: usize, h_k: usize, w_stride: usize, h_stride: usize, info: *const usize, src: *const u8, dst: *mut u8) {
        unsafe { avg_pool2d(w_k, h_k, w_stride, h_stride, info, src, dst, 0u8, id::<u8>, add_u8, avg_u8); }
    }

    #[kernel]
    pub unsafe fn avg_pool2d_u32(_src_numel: usize, w_k: usize, h_k: usize, w_stride: usize, h_stride: usize, info: *const usize, src: *const u32, dst: *mut u32) {
        unsafe { avg_pool2d(w_k, h_k, w_stride, h_stride, info, src, dst, 0u32, id::<u32>, add_u32, avg_u32); }
    }

    #[kernel]
    pub unsafe fn max_pool2d_f32(_src_numel: usize, w_k: usize, h_k: usize, w_stride: usize, h_stride: usize, info: *const usize, src: *const f32, dst: *mut f32) {
        unsafe { max_pool2d(w_k, h_k, w_stride, h_stride, info, src, dst, 0.0f32, max_f32); }
    }

    #[kernel]
    pub unsafe fn max_pool2d_f64(_src_numel: usize, w_k: usize, h_k: usize, w_stride: usize, h_stride: usize, info: *const usize, src: *const f64, dst: *mut f64) {
        unsafe { max_pool2d(w_k, h_k, w_stride, h_stride, info, src, dst, 0.0f64, sass_dmax); }
    }

    #[kernel]
    pub unsafe fn max_pool2d_u8(_src_numel: usize, w_k: usize, h_k: usize, w_stride: usize, h_stride: usize, info: *const usize, src: *const u8, dst: *mut u8) {
        unsafe { max_pool2d(w_k, h_k, w_stride, h_stride, info, src, dst, 0u8, max_u8); }
    }

    #[kernel]
    pub unsafe fn max_pool2d_u32(_src_numel: usize, w_k: usize, h_k: usize, w_stride: usize, h_stride: usize, info: *const usize, src: *const u32, dst: *mut u32) {
        unsafe { max_pool2d(w_k, h_k, w_stride, h_stride, info, src, dst, 0u32, max_u32); }
    }

    #[kernel]
    pub unsafe fn upsample_nearest2d_f32(w_out: usize, h_out: usize, w_scale: f64, h_scale: f64, info: *const usize, src: *const f32, dst: *mut f32) {
        unsafe { upsample_nearest2d(w_out, h_out, w_scale, h_scale, info, src, dst); }
    }

    #[kernel]
    pub unsafe fn upsample_nearest2d_f64(w_out: usize, h_out: usize, w_scale: f64, h_scale: f64, info: *const usize, src: *const f64, dst: *mut f64) {
        unsafe { upsample_nearest2d(w_out, h_out, w_scale, h_scale, info, src, dst); }
    }

    #[kernel]
    pub unsafe fn upsample_nearest2d_u8(w_out: usize, h_out: usize, w_scale: f64, h_scale: f64, info: *const usize, src: *const u8, dst: *mut u8) {
        unsafe { upsample_nearest2d(w_out, h_out, w_scale, h_scale, info, src, dst); }
    }

    #[kernel]
    pub unsafe fn upsample_nearest2d_u32(w_out: usize, h_out: usize, w_scale: f64, h_scale: f64, info: *const usize, src: *const u32, dst: *mut u32) {
        unsafe { upsample_nearest2d(w_out, h_out, w_scale, h_scale, info, src, dst); }
    }

    #[kernel]
    pub unsafe fn upsample_bilinear2d_f32(w_out: usize, h_out: usize, align_corners: u8, has_scale_h: u8, scale_h_factor: f64, has_scale_w: u8, scale_w_factor: f64, info: *const usize, src: *const f32, dst: *mut f32) {
        unsafe { upsample_bilinear2d(w_out, h_out, align_corners, has_scale_h, scale_h_factor, has_scale_w, scale_w_factor, info, src, dst, f32_f64, f64_f32); }
    }

    #[kernel]
    pub unsafe fn upsample_bilinear2d_f64(w_out: usize, h_out: usize, align_corners: u8, has_scale_h: u8, scale_h_factor: f64, has_scale_w: u8, scale_w_factor: f64, info: *const usize, src: *const f64, dst: *mut f64) {
        unsafe { upsample_bilinear2d(w_out, h_out, align_corners, has_scale_h, scale_h_factor, has_scale_w, scale_w_factor, info, src, dst, id::<f64>, id::<f64>); }
    }

    #[kernel]
    pub unsafe fn upsample_bilinear2d_u8(w_out: usize, h_out: usize, align_corners: u8, has_scale_h: u8, scale_h_factor: f64, has_scale_w: u8, scale_w_factor: f64, info: *const usize, src: *const u8, dst: *mut u8) {
        unsafe { upsample_bilinear2d(w_out, h_out, align_corners, has_scale_h, scale_h_factor, has_scale_w, scale_w_factor, info, src, dst, u8_f64, f64_u8); }
    }

    #[kernel]
    pub unsafe fn upsample_bilinear2d_u32(w_out: usize, h_out: usize, align_corners: u8, has_scale_h: u8, scale_h_factor: f64, has_scale_w: u8, scale_w_factor: f64, info: *const usize, src: *const u32, dst: *mut u32) {
        unsafe { upsample_bilinear2d(w_out, h_out, align_corners, has_scale_h, scale_h_factor, has_scale_w, scale_w_factor, info, src, dst, u32_f64, f64_u32); }
    }

    #[kernel]
    pub unsafe fn im2col_f32(dst_numel: usize, h_out: usize, w_out: usize, h_k: usize, w_k: usize, stride: usize, padding: usize, dilation: usize, info: *const usize, src: *const f32, dst: *mut f32) {
        unsafe { im2col(dst_numel, h_out, w_out, h_k, w_k, stride, padding, dilation, info, src, dst, 0.0f32); }
    }

    #[kernel]
    pub unsafe fn im2col_f64(dst_numel: usize, h_out: usize, w_out: usize, h_k: usize, w_k: usize, stride: usize, padding: usize, dilation: usize, info: *const usize, src: *const f64, dst: *mut f64) {
        unsafe { im2col(dst_numel, h_out, w_out, h_k, w_k, stride, padding, dilation, info, src, dst, 0.0f64); }
    }

    #[kernel]
    pub unsafe fn im2col_u8(dst_numel: usize, h_out: usize, w_out: usize, h_k: usize, w_k: usize, stride: usize, padding: usize, dilation: usize, info: *const usize, src: *const u8, dst: *mut u8) {
        unsafe { im2col(dst_numel, h_out, w_out, h_k, w_k, stride, padding, dilation, info, src, dst, 0u8); }
    }

    #[kernel]
    pub unsafe fn im2col_u32(dst_numel: usize, h_out: usize, w_out: usize, h_k: usize, w_k: usize, stride: usize, padding: usize, dilation: usize, info: *const usize, src: *const u32, dst: *mut u32) {
        unsafe { im2col(dst_numel, h_out, w_out, h_k, w_k, stride, padding, dilation, info, src, dst, 0u32); }
    }

    #[kernel]
    pub unsafe fn im2col1d_f32(dst_numel: usize, l_out: usize, l_k: usize, stride: usize, padding: usize, dilation: usize, info: *const usize, src: *const f32, dst: *mut f32) {
        unsafe { im2col1d(dst_numel, l_out, l_k, stride, padding, dilation, info, src, dst, 0.0f32); }
    }

    #[kernel]
    pub unsafe fn im2col1d_f64(dst_numel: usize, l_out: usize, l_k: usize, stride: usize, padding: usize, dilation: usize, info: *const usize, src: *const f64, dst: *mut f64) {
        unsafe { im2col1d(dst_numel, l_out, l_k, stride, padding, dilation, info, src, dst, 0.0f64); }
    }

    #[kernel]
    pub unsafe fn im2col1d_u8(dst_numel: usize, l_out: usize, l_k: usize, stride: usize, padding: usize, dilation: usize, info: *const usize, src: *const u8, dst: *mut u8) {
        unsafe { im2col1d(dst_numel, l_out, l_k, stride, padding, dilation, info, src, dst, 0u8); }
    }

    #[kernel]
    pub unsafe fn im2col1d_u32(dst_numel: usize, l_out: usize, l_k: usize, stride: usize, padding: usize, dilation: usize, info: *const usize, src: *const u32, dst: *mut u32) {
        unsafe { im2col1d(dst_numel, l_out, l_k, stride, padding, dilation, info, src, dst, 0u32); }
    }

    #[kernel]
    pub unsafe fn col2im1d_f32(dst_el: usize, l_out: usize, l_in: usize, c_out: usize, k_size: usize, stride: usize, src: *const f32, dst: *mut f32) {
        unsafe { col2im1d(dst_el, l_out, l_in, c_out, k_size, stride, src, dst, 0.0f32, add_f32); }
    }

    #[kernel]
    pub unsafe fn col2im1d_f64(dst_el: usize, l_out: usize, l_in: usize, c_out: usize, k_size: usize, stride: usize, src: *const f64, dst: *mut f64) {
        unsafe { col2im1d(dst_el, l_out, l_in, c_out, k_size, stride, src, dst, 0.0f64, sass_dadd); }
    }

    #[kernel]
    pub unsafe fn col2im1d_u8(dst_el: usize, l_out: usize, l_in: usize, c_out: usize, k_size: usize, stride: usize, src: *const u8, dst: *mut u8) {
        unsafe { col2im1d(dst_el, l_out, l_in, c_out, k_size, stride, src, dst, 0u8, add_u8); }
    }

    #[kernel]
    pub unsafe fn col2im1d_u32(dst_el: usize, l_out: usize, l_in: usize, c_out: usize, k_size: usize, stride: usize, src: *const u32, dst: *mut u32) {
        unsafe { col2im1d(dst_el, l_out, l_in, c_out, k_size, stride, src, dst, 0u32, add_u32); }
    }
    // GENERATED KERNELS END
}

// ---------------------------------------------------------------------------------------------
// Differential gate against candle's nvcc-built conv.ptx.
use kdiff::{Arg, Harness, Rng, Tally, as_bytes, contiguous, layout_info};

fn esize(dt: &str) -> usize {
    match dt { "bf16" | "f16" => 2, "u8" => 1, "f32" | "u32" => 4, "f64" => 8, _ => panic!("{dt}") }
}

/// Element data. mode 0: mixed (mostly moderate values, some raw bit patterns and edge values);
/// mode 1: only edge values (+-0, nan, inf, denormals, ties) to probe NaN/zero semantics.
fn data(rng: &mut Rng, dt: &str, n: usize, mode: u32) -> Vec<u8> {
    let n = n.max(1);
    match dt {
        "bf16" | "f16" => {
            let edge: [u16; 10] = if dt == "bf16" {
                [0x0000, 0x8000, 0x7f80, 0xff80, 0x7fc0, 0x0001, 0x8001, 0x7f7f, 0x3f80, 0xbf80]
            } else {
                [0x0000, 0x8000, 0x7c00, 0xfc00, 0x7e00, 0x0001, 0x8001, 0x7bff, 0x3c00, 0xbc00]
            };
            let v: Vec<u16> = (0..n)
                .map(|_| {
                    let r = rng.next();
                    if mode == 1 {
                        return edge[(r % 10) as usize];
                    }
                    match r % 16 {
                        0 => edge[((r >> 8) % 10) as usize],
                        1 => (r >> 16) as u16,
                        _ => {
                            let sign = ((r >> 20) & 1) as u16;
                            if dt == "bf16" {
                                (sign << 15) | ((((r >> 24) % 12 + 121) as u16) << 7) | ((r >> 40) as u16 & 0x7f)
                            } else {
                                (sign << 15) | ((((r >> 24) % 12 + 9) as u16) << 10) | ((r >> 40) as u16 & 0x3ff)
                            }
                        }
                    }
                })
                .collect();
            as_bytes(&v)
        }
        "f32" => {
            if mode == 1 {
                let e = [0.0f32, -0.0, f32::NAN, f32::from_bits(0xffc0_0001), f32::INFINITY, f32::NEG_INFINITY, 1.0, -1.0, 1e-40, 3e38];
                as_bytes(&(0..n).map(|_| e[(rng.next() % 10) as usize]).collect::<Vec<_>>())
            } else {
                as_bytes(&rng.f32s(n))
            }
        }
        "f64" => {
            if mode == 1 {
                let e = [0.0f64, -0.0, f64::NAN, f64::from_bits(0xfff8_0000_0000_0001), f64::INFINITY, f64::NEG_INFINITY, 1.0, f64::from_bits(0x7ff0_0000_0000_0002), 1e-310, 1e308];
                as_bytes(&(0..n).map(|_| e[(rng.next() % 10) as usize]).collect::<Vec<_>>())
            } else {
                as_bytes(&rng.f64s(n))
            }
        }
        "u8" => {
            if mode == 1 { (0..n).map(|_| [0u8, 1, 255, 128][(rng.next() % 4) as usize]).collect() } else { rng.bytes(n) }
        }
        "u32" => {
            if mode == 1 {
                as_bytes(&(0..n).map(|_| [0u32, 1, u32::MAX, 0x8000_0000, 0x7fff_ffff][(rng.next() % 5) as usize]).collect::<Vec<_>>())
            } else {
                as_bytes(&(0..n).map(|_| { let r = rng.next(); if r % 2 == 0 { (r >> 32) as u32 } else { (r >> 40) as u32 % 1000 } }).collect::<Vec<_>>())
            }
        }
        _ => panic!("{dt}"),
    }
}

/// Stride variants for logical `dims`: contiguous, padded rows, column-major (permuted), broadcast dim 0.
fn layouts(dims: &[usize]) -> Vec<Vec<usize>> {
    let mut v = vec![contiguous(dims)];
    let mut padded = vec![1usize; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        padded[i] = padded[i + 1] * (dims[i + 1] + 1 + i);
    }
    v.push(padded);
    let mut colmajor = vec![1usize; dims.len()];
    for i in 1..dims.len() {
        colmajor[i] = colmajor[i - 1] * dims[i - 1];
    }
    v.push(colmajor);
    let mut bc = contiguous(dims);
    bc[0] = 0;
    v.push(bc);
    v
}
fn span(dims: &[usize], strides: &[usize]) -> usize {
    dims.iter().zip(strides).map(|(d, s)| (d.max(&1) - 1) * s).sum::<usize>() + 1
}
fn info(parts: &[(&[usize], &[usize])]) -> Vec<u8> {
    let mut v = vec![];
    for (d, s) in parts {
        v.extend(layout_info(d, s));
    }
    v
}

struct Gate {
    h: Harness,
    t: Tally,
    rng: Rng,
}

impl Gate {
    /// Launch with candle's config (1024-thread blocks) and two other block shapes that over-provision.
    fn run(&mut self, name: &str, label: &str, threads: usize, args: &[Arg], bufs: &[Vec<u8>], out: usize) {
        let n = threads.max(1) as u32;
        for (grid, block) in [(n.div_ceil(1024), 1024u32), (n.div_ceil(256) + 1, 256), (n.div_ceil(33) + 2, 33)] {
            let d = self.h.diff(name, (grid, 1, 1), (block, 1, 1), 0, args, bufs, &[out]);
            if d.differing > 0 && std::env::var("CONV_DEBUG").is_ok() && self.t.failures.len() < 3 {
                self.dump(name, (grid, block), args, bufs, out);
            }
            self.t.record(&format!("{name} {label} g{grid}x{block}"), &d);
        }
    }
}

impl Gate {
    /// Diagnostic: rerun one launch in both modules and print the differing 8-byte words.
    fn dump(&self, name: &str, (grid, block): (u32, u32), args: &[Arg], bufs: &[Vec<u8>], out: usize) {
        use kdiff::cuda_core::DeviceBuffer;
        let run = |m: &std::sync::Arc<kdiff::cuda_core::CudaModule>| -> Vec<u8> {
            let st = &self.h.stream;
            let dbufs: Vec<DeviceBuffer<u8>> = bufs.iter().map(|b| DeviceBuffer::from_host(st, b).unwrap()).collect();
            let mut vals: Vec<[u8; 8]> = args.iter().map(|a| match a {
                Arg::Buf(i) => dbufs[*i].cu_deviceptr().to_le_bytes(),
                Arg::U64(v) => v.to_le_bytes(),
                Arg::F64(v) => v.to_bits().to_le_bytes(),
                Arg::B8(v) => (*v as u64).to_le_bytes(),
                _ => unreachable!(),
            }).collect();
            let mut ptrs: Vec<*mut std::ffi::c_void> = vals.iter_mut().map(|v| v.as_mut_ptr() as *mut std::ffi::c_void).collect();
            let f = m.load_function(name).unwrap();
            unsafe { kdiff::cuda_core::simt::launch_kernel_on_stream(&f, (grid, 1, 1), (block, 1, 1), 0, st, &mut ptrs).unwrap(); }
            st.synchronize().unwrap();
            dbufs[out].to_host_vec(st).unwrap()
        };
        let (a, b) = (run(&self.h.reference), run(&self.h.oxide));
        for i in (0..a.len()).step_by(8) {
            let e = (i + 8).min(a.len());
            if a[i..e] != b[i..e] {
                println!("    word {}: ref {:02x?} oxide {:02x?}", i / 8, &a[i..e], &b[i..e]);
            }
        }
    }
}

fn rr(rng: &mut Rng, lo: usize, hi: usize) -> usize {
    lo + (rng.next() % (hi - lo + 1) as u64) as usize
}
const BIG: usize = (1 << 32) + 5; // forces the div.u64 / rem.u64 side of nvcc's bypass

impl Gate {
    fn u(x: usize) -> Arg { Arg::U64(x as u64) }

    /// conv1d / conv_transpose1d: (b, c_in, l_in, c_out, k, stride, padding, out_padding, dilation, l_out)
    fn conv1(&mut self, name: &str, dt: &str, tr: bool, c: [usize; 10], li: usize, mode: u32, threads: Option<usize>) {
        let [b, c_in, l_in, c_out, k, s, p, op_, dl, l_out] = c;
        let es = esize(dt);
        let sd = [b, c_in, l_in];
        let kd = if tr { [c_in, c_out, k] } else { [c_out, c_in, k] };
        let (ss, ks) = (layouts(&sd)[li % 4].clone(), layouts(&kd)[(li / 4) % 4].clone());
        let n = threads.unwrap_or(b * c_out * l_out);
        let bufs = vec![info(&[(&sd, &ss), (&kd, &ks)]), data(&mut self.rng, dt, span(&sd, &ss), mode),
                        data(&mut self.rng, dt, span(&kd, &ks), mode), self.rng.bytes(n * es)];
        let u = Self::u;
        let mut args = vec![u(b * c_in * l_in), u(l_out), u(s), u(p)];
        if tr { args.push(u(op_)); }
        args.extend([u(dl), Arg::Buf(0), Arg::Buf(1), Arg::Buf(2), Arg::Buf(3)]);
        self.run(name, &format!("{c:?} layout{li} mode{mode}"), n, &args, &bufs, 3);
    }

    /// conv2d / conv_transpose2d: (b, c_in, h, w, c_out, hk, wk, stride, padding, out_padding, dilation, h_out, w_out)
    fn conv2(&mut self, name: &str, dt: &str, tr: bool, c: [usize; 13], li: usize, mode: u32, threads: Option<usize>) {
        let [b, c_in, hh, ww, c_out, hk, wk, s, p, op_, dl, h_out, w_out] = c;
        let es = esize(dt);
        let sd = [b, c_in, hh, ww];
        let kd = if tr { [c_in, c_out, hk, wk] } else { [c_out, c_in, hk, wk] };
        let (ss, ks) = (layouts(&sd)[li % 4].clone(), layouts(&kd)[(li / 4) % 4].clone());
        let n = threads.unwrap_or(b * c_out * h_out * w_out);
        let bufs = vec![info(&[(&sd, &ss), (&kd, &ks)]), data(&mut self.rng, dt, span(&sd, &ss), mode),
                        data(&mut self.rng, dt, span(&kd, &ks), mode), self.rng.bytes(n * es)];
        let u = Self::u;
        let mut args = vec![u(b * c_in * hh * ww), u(w_out), u(h_out), u(s), u(p)];
        if tr { args.push(u(op_)); }
        args.extend([u(dl), Arg::Buf(0), Arg::Buf(1), Arg::Buf(2), Arg::Buf(3)]);
        self.run(name, &format!("{c:?} layout{li} mode{mode}"), n, &args, &bufs, 3);
    }

    /// avg/max pool: (b, c, w_in, h_in, w_k, h_k, w_stride, h_stride)
    fn pool(&mut self, name: &str, dt: &str, c: [usize; 8], li: usize, mode: u32, threads: Option<usize>) {
        let [b, ch, w_in, h_in, wk, hk, ws, hs] = c;
        let n = threads.unwrap_or_else(|| b * ch * ((w_in - wk) / ws + 1) * ((h_in - hk) / hs + 1));
        let sd = [b, ch, w_in, h_in];
        let ss = layouts(&sd)[li % 4].clone();
        let bufs = vec![info(&[(&sd, &ss)]), data(&mut self.rng, dt, span(&sd, &ss), mode), self.rng.bytes(n * esize(dt))];
        let u = Self::u;
        let args = [u(b * ch * w_in * h_in), u(wk), u(hk), u(ws), u(hs), Arg::Buf(0), Arg::Buf(1), Arg::Buf(2)];
        self.run(name, &format!("{c:?} layout{li} mode{mode}"), n, &args, &bufs, 2);
    }

    /// upsample_nearest2d: (b, c, w_in, h_in, w_out, h_out) + scales
    fn nearest(&mut self, name: &str, dt: &str, c: [usize; 6], sw: f64, sh: f64, li: usize, threads: Option<usize>) {
        let [b, ch, w_in, h_in, w_out, h_out] = c;
        let n = threads.unwrap_or(b * ch * w_out * h_out);
        let sd = [b, ch, w_in, h_in];
        let ss = layouts(&sd)[li % 4].clone();
        let bufs = vec![info(&[(&sd, &ss)]), data(&mut self.rng, dt, span(&sd, &ss), 0), self.rng.bytes(n * esize(dt))];
        let u = Self::u;
        let args = [u(w_out), u(h_out), Arg::F64(sw), Arg::F64(sh), Arg::Buf(0), Arg::Buf(1), Arg::Buf(2)];
        self.run(name, &format!("{c:?} scale({sw},{sh}) layout{li}"), n, &args, &bufs, 2);
    }

    /// upsample_bilinear2d: (b, c, h_in, w_in, h_out, w_out) + (align, has_h, fh, has_w, fw)
    fn bilinear(&mut self, name: &str, dt: &str, c: [usize; 6], m: (u8, u8, f64, u8, f64), li: usize, mode: u32, threads: Option<usize>) {
        let [b, ch, h_in, w_in, h_out, w_out] = c;
        let n = threads.unwrap_or(b * ch * w_out * h_out);
        let sd = [b, ch, h_in, w_in];
        let ss = layouts(&sd)[li % 4].clone();
        let bufs = vec![info(&[(&sd, &ss)]), data(&mut self.rng, dt, span(&sd, &ss), mode), self.rng.bytes(n * esize(dt))];
        let u = Self::u;
        let args = [u(w_out), u(h_out), Arg::B8(m.0), Arg::B8(m.1), Arg::F64(m.2), Arg::B8(m.3), Arg::F64(m.4), Arg::Buf(0), Arg::Buf(1), Arg::Buf(2)];
        self.run(name, &format!("{c:?} {m:?} layout{li} mode{mode}"), n, &args, &bufs, 2);
    }

    /// im2col1d: (b, c_in, l_in, l_k, stride, padding, dilation, l_out)
    fn im2col1(&mut self, name: &str, dt: &str, c: [usize; 8], li: usize, threads: Option<usize>) {
        let [b, c_in, l_in, lk, s, p, dl, l_out] = c;
        let n = threads.unwrap_or(b * l_out * c_in);
        let sd = [b, c_in, l_in];
        let ss = layouts(&sd)[li % 4].clone();
        let bufs = vec![info(&[(&sd, &ss)]), data(&mut self.rng, dt, span(&sd, &ss), 0), self.rng.bytes(n * lk * esize(dt))];
        let u = Self::u;
        let args = [u(n), u(l_out), u(lk), u(s), u(p), u(dl), Arg::Buf(0), Arg::Buf(1), Arg::Buf(2)];
        self.run(name, &format!("{c:?} layout{li}"), n, &args, &bufs, 2);
    }

    /// im2col: (b, c_in, h, w, hk, wk, stride, padding, dilation, h_out, w_out)
    fn im2col2(&mut self, name: &str, dt: &str, c: [usize; 11], li: usize, threads: Option<usize>) {
        let [b, c_in, hh, ww, hk, wk, s, p, dl, h_out, w_out] = c;
        let n = threads.unwrap_or(b * h_out * w_out * c_in * hk * wk);
        let sd = [b, c_in, hh, ww];
        let ss = layouts(&sd)[li % 4].clone();
        let bufs = vec![info(&[(&sd, &ss)]), data(&mut self.rng, dt, span(&sd, &ss), 0), self.rng.bytes(n * esize(dt))];
        let u = Self::u;
        let args = [u(n), u(h_out), u(w_out), u(hk), u(wk), u(s), u(p), u(dl), Arg::Buf(0), Arg::Buf(1), Arg::Buf(2)];
        self.run(name, &format!("{c:?} layout{li}"), n, &args, &bufs, 2);
    }

    /// col2im1d: (b, l_in, c_out, k_size, stride, l_out)
    fn col2im(&mut self, name: &str, dt: &str, c: [usize; 6], mode: u32, threads: Option<usize>) {
        let [b, l_in, c_out, k, s, l_out] = c;
        let n = threads.unwrap_or(b * c_out * l_out);
        let bufs = vec![data(&mut self.rng, dt, b * l_in * c_out * k, mode), self.rng.bytes(n * esize(dt))];
        let u = Self::u;
        let args = [u(n), u(l_out), u(l_in), u(c_out), u(k), u(s), Arg::Buf(0), Arg::Buf(1)];
        self.run(name, &format!("{c:?} mode{mode}"), n, &args, &bufs, 1);
    }
}

fn main() {
    let root = format!("{}/titan-engine/oxide-kernels", std::env::var("HOME").unwrap());
    let ref_ptx = format!("{root}/reference/candle/conv.ptx");
    let h = Harness::new(&ref_ptx, &format!("{root}/candle-conv/candle_conv.ptx"));
    let names: Vec<String> = std::fs::read_to_string(&ref_ptx)
        .unwrap()
        .lines()
        .filter_map(|l| l.strip_prefix(".visible .entry ").map(|s| s.trim_end_matches('(').to_string()))
        .collect();
    assert_eq!(names.len(), 66, "reference entry count");
    let mut g = Gate { h, t: Tally::default(), rng: Rng(0xC0417) };
    let mut missing = vec![];
    let ops = ["conv_transpose1d", "conv_transpose2d", "upsample_nearest2d", "upsample_bilinear2d", "avg_pool2d", "max_pool2d",
               "im2col1d", "col2im1d", "im2col", "conv1d", "conv2d"];
    let only = std::env::var("CONV_ONLY").ok();
    const RANDOM: usize = 40; // random shapes per entry, on top of the fixed edge cases
    for name in &names {
        if only.as_ref().is_some_and(|o| name != o) {
            continue;
        }
        if !g.h.has(true, name) {
            missing.push(name.clone());
            continue;
        }
        let op = *ops.iter().find(|o| name.starts_with(&format!("{o}_"))).unwrap();
        let dt = &name[op.len() + 1..];
        match op {
            "conv1d" | "conv_transpose1d" => {
                let tr = op == "conv_transpose1d";
                // (b, c_in, l_in, c_out, k, stride, padding, out_padding, dilation)
                let mut cases: Vec<[usize; 9]> = if tr {
                    vec![[1, 1, 1, 1, 1, 1, 0, 0, 1], [2, 3, 7, 4, 3, 2, 1, 1, 1], [1, 5, 9, 3, 4, 3, 0, 2, 2], [2, 2, 6, 2, 5, 1, 2, 0, 1],
                         [1, 40, 5, 2, 3, 2, 1, 1, 3], [3, 1, 4, 1, 2, 4, 0, 3, 1]]
                } else {
                    vec![[1, 1, 1, 1, 1, 1, 0, 0, 1], [2, 3, 10, 4, 3, 1, 1, 0, 1], [1, 5, 17, 3, 5, 2, 2, 0, 1], [3, 2, 9, 2, 2, 3, 0, 0, 2],
                         [2, 7, 33, 5, 4, 1, 3, 0, 3], [1, 64, 12, 2, 3, 1, 1, 0, 1], [2, 1, 5, 1, 5, 1, 4, 0, 1], [1, 3, 40, 3, 7, 4, 3, 0, 2]]
                };
                let fixed = cases.len();
                for _ in 0..RANDOM {
                    let r = &mut g.rng;
                    let (b, c_in, c_out, k, s, dl) = (rr(r, 1, 3), rr(r, 1, 9), rr(r, 1, 5), rr(r, 1, 6), rr(r, 1, 4), rr(r, 1, 3));
                    let l_in = rr(r, 1, 20);
                    if tr {
                        let op_ = rr(r, 0, s.max(dl) - 1);
                        let raw = (l_in - 1) * s + dl * (k - 1) + op_ + 1;
                        cases.push([b, c_in, l_in, c_out, k, s, rr(r, 0, (raw - 1) / 2), op_, dl]);
                    } else {
                        let p = rr(r, 0, 4);
                        let l_in = l_in.max((dl * (k - 1) + 1).saturating_sub(2 * p));
                        cases.push([b, c_in, l_in, c_out, k, s, p, 0, dl]);
                    }
                }
                for (ci, &[b, c_in, l_in, c_out, k, s, p, op_, dl]) in cases.iter().enumerate() {
                    let l_out = if tr { (l_in - 1) * s + dl * (k - 1) + op_ + 1 - 2 * p } else { (l_in + 2 * p - dl * (k - 1) - 1) / s + 1 };
                    let c = [b, c_in, l_in, c_out, k, s, p, op_, dl, l_out];
                    if ci < fixed {
                        for li in 0..16 {
                            g.conv1(name, dt, tr, c, li, (li == 0 || li == 5) as u32, None);
                        }
                        if !tr {
                            // l_out larger than the formula: every out-of-range tap is skipped by the padding test
                            g.conv1(name, dt, tr, [b, c_in, l_in, c_out, k, s, p, op_, dl, l_out + 2], 1, 0, None);
                        }
                    } else {
                        let (li, mode) = (rr(&mut g.rng, 0, 15), rr(&mut g.rng, 0, 1) as u32);
                        g.conv1(name, dt, tr, c, li, mode, None);
                    }
                }
                // 64-bit division side of the bypass: l_out >= 2^32 with a bounded thread count.
                g.conv1(name, dt, tr, [1, 2, 5, 1, 3, 1, 1, 0, 1, BIG], 0, 0, Some(300));
                g.conv1(name, dt, tr, [2, 3, 9, 2, 3, 2, 1, 1, 1, BIG + 7], 5, 1, Some(200));
                if tr {
                    // C int arithmetic: padding / dilation / stride past 32 bits truncate, >= 2^31 goes negative.
                    for (p, dl, s) in [((1usize << 32) + 2, 1, 2), (0x8000_0000, 1, 1), (0x7fff_fffe, 1, 1), (1, (1 << 32) + 1, 1), (1, 1, BIG), (3, 2, (1 << 32) + 2)] {
                        g.conv1(name, dt, tr, [2, 3, 6, 2, 3, s, p, 0, dl, 13], 0, 0, None);
                    }
                }
            }
            "conv2d" | "conv_transpose2d" => {
                let tr = op == "conv_transpose2d";
                // (b, c_in, h, w, c_out, hk, wk, stride, padding, out_padding, dilation)
                let mut cases: Vec<[usize; 11]> = if tr {
                    vec![[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1], [2, 3, 4, 5, 2, 3, 3, 2, 1, 1, 1], [1, 2, 5, 3, 3, 2, 4, 3, 0, 2, 2],
                         [1, 17, 3, 3, 2, 3, 3, 1, 1, 0, 1], [2, 1, 3, 6, 1, 5, 2, 2, 2, 1, 1]]
                } else {
                    vec![[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1], [2, 3, 6, 7, 4, 3, 3, 1, 1, 0, 1], [1, 5, 9, 8, 3, 2, 3, 2, 1, 0, 1],
                         [2, 2, 11, 10, 2, 3, 2, 3, 2, 0, 2], [1, 33, 5, 5, 2, 3, 3, 1, 1, 0, 1], [3, 1, 4, 9, 1, 1, 4, 1, 0, 0, 1],
                         [1, 4, 8, 8, 5, 5, 5, 2, 3, 0, 1]]
                };
                let fixed = cases.len();
                for _ in 0..RANDOM {
                    let r = &mut g.rng;
                    let (b, c_in, c_out, hk, wk, s, dl) = (rr(r, 1, 2), rr(r, 1, 6), rr(r, 1, 4), rr(r, 1, 4), rr(r, 1, 4), rr(r, 1, 3), rr(r, 1, 2));
                    let (hh, ww) = (rr(r, 1, 9), rr(r, 1, 9));
                    if tr {
                        let op_ = rr(r, 0, s.max(dl) - 1);
                        let raw = (hh.min(ww) - 1) * s + dl * (hk.min(wk) - 1) + op_ + 1;
                        cases.push([b, c_in, hh, ww, c_out, hk, wk, s, rr(r, 0, (raw - 1) / 2), op_, dl]);
                    } else {
                        let p = rr(r, 0, 3);
                        let hh = hh.max((dl * (hk - 1) + 1).saturating_sub(2 * p));
                        let ww = ww.max((dl * (wk - 1) + 1).saturating_sub(2 * p));
                        cases.push([b, c_in, hh, ww, c_out, hk, wk, s, p, 0, dl]);
                    }
                }
                for (ci, &[b, c_in, hh, ww, c_out, hk, wk, s, p, op_, dl]) in cases.iter().enumerate() {
                    let (h_out, w_out) = if tr {
                        ((hh - 1) * s + dl * (hk - 1) + op_ + 1 - 2 * p, (ww - 1) * s + dl * (wk - 1) + op_ + 1 - 2 * p)
                    } else {
                        ((hh + 2 * p - dl * (hk - 1) - 1) / s + 1, (ww + 2 * p - dl * (wk - 1) - 1) / s + 1)
                    };
                    let c = [b, c_in, hh, ww, c_out, hk, wk, s, p, op_, dl, h_out, w_out];
                    if ci < fixed {
                        for li in 0..16 {
                            g.conv2(name, dt, tr, c, li, (li == 0 || li == 5) as u32, None);
                        }
                        if !tr {
                            g.conv2(name, dt, tr, [b, c_in, hh, ww, c_out, hk, wk, s, p, op_, dl, h_out + 1, w_out + 2], 1, 0, None);
                        }
                    } else {
                        let (li, mode) = (rr(&mut g.rng, 0, 15), rr(&mut g.rng, 0, 1) as u32);
                        g.conv2(name, dt, tr, c, li, mode, None);
                    }
                }
                g.conv2(name, dt, tr, [1, 2, 4, 5, 1, 2, 3, 1, 1, 0, 1, 1, BIG], 0, 0, Some(300));
                g.conv2(name, dt, tr, [2, 2, 5, 4, 2, 3, 2, 1, 1, 0, 1, BIG, 1], 5, 1, Some(300));
                if tr {
                    for (p, dl, s) in [((1usize << 32) + 1, 1, 2), (0x8000_0000, 1, 1), (1, (1 << 32) + 1, 1), (1, 1, BIG)] {
                        g.conv2(name, dt, tr, [2, 3, 4, 5, 2, 3, 2, s, p, 0, dl, 7, 6], 0, 0, None);
                    }
                }
            }
            "avg_pool2d" | "max_pool2d" => {
                let mut cases: Vec<[usize; 8]> = vec![[1, 1, 1, 1, 1, 1, 1, 1], [2, 3, 8, 6, 2, 2, 2, 2], [1, 2, 9, 7, 3, 3, 1, 1], [3, 1, 10, 5, 3, 2, 3, 1],
                                                      [1, 4, 7, 11, 1, 4, 2, 3], [2, 5, 6, 6, 6, 6, 1, 1], [1, 3, 13, 9, 2, 3, 5, 4], [1, 1, 20, 20, 7, 5, 1, 2]];
                let fixed = cases.len();
                for _ in 0..RANDOM {
                    let r = &mut g.rng;
                    let (w_in, h_in) = (rr(r, 1, 15), rr(r, 1, 15));
                    cases.push([rr(r, 1, 3), rr(r, 1, 5), w_in, h_in, rr(r, 1, w_in), rr(r, 1, h_in), rr(r, 1, 4), rr(r, 1, 4)]);
                }
                // huge strides: w_out / h_out = 1 through the 64-bit division
                cases.push([2, 3, 5, 6, 2, 3, BIG, BIG + 1]);
                for (ci, &c) in cases.iter().enumerate() {
                    if ci < fixed {
                        for li in 0..4 {
                            for mode in 0..2 {
                                g.pool(name, dt, c, li, mode, None);
                            }
                        }
                    } else {
                        let (li, mode) = (rr(&mut g.rng, 0, 3), rr(&mut g.rng, 0, 1) as u32);
                        g.pool(name, dt, c, li, mode, None);
                    }
                }
                // stride 0 (never sent by candle) makes the bypass observable: (w_in - w_k) / 0 is
                // div.u32 -> 0xffffffff, so w_out = 2^32 (not 0 -> early return, as div.u64 would give).
                g.pool(name, dt, [2, 3, 5, 6, 2, 3, 0, 2], 0, 0, Some(700));
                g.pool(name, dt, [1, 2, 4, 4, 2, 2, 1, 0], 1, 0, Some(300));
            }
            "upsample_nearest2d" => {
                let mut cases: Vec<[usize; 6]> = vec![[1, 1, 1, 1, 1, 1], [2, 3, 4, 5, 8, 10], [1, 2, 7, 3, 5, 9], [3, 1, 6, 6, 3, 2], [1, 4, 5, 8, 13, 17]];
                let fixed = cases.len();
                for _ in 0..RANDOM {
                    let r = &mut g.rng;
                    cases.push([rr(r, 1, 3), rr(r, 1, 4), rr(r, 1, 9), rr(r, 1, 9), rr(r, 1, 17), rr(r, 1, 17)]);
                }
                for (ci, &c) in cases.iter().enumerate() {
                    let [_, _, w_in, h_in, w_out, h_out] = c;
                    let natural = (w_in as f64 / w_out as f64, h_in as f64 / h_out as f64);
                    if ci < fixed {
                        // candle's scales, then odd ones: > 1 (clamps), negative / nan (saturate to 0), tiny, inf.
                        let scales = [natural, (natural.0 * 1.37, 0.999), (-2.5, f64::NAN), (1e-300, 7.0), (f64::INFINITY, 0.3333)];
                        for li in 0..4 {
                            for (si, &(sw, sh)) in scales.iter().enumerate() {
                                if li > 0 && si > 1 { continue; }
                                g.nearest(name, dt, c, sw, sh, li, None);
                            }
                        }
                    } else {
                        let li = rr(&mut g.rng, 0, 3);
                        g.nearest(name, dt, c, natural.0, natural.1, li, None);
                    }
                }
                g.nearest(name, dt, [2, 3, 4, 5, 1, BIG], 1.0, 1e-9, 0, Some(300));
                g.nearest(name, dt, [2, 3, 4, 5, BIG, 1], 1e-9, 1.0, 2, Some(300));
            }
            "upsample_bilinear2d" => {
                let mut cases: Vec<[usize; 6]> = vec![[1, 1, 1, 1, 1, 1], [2, 3, 4, 5, 8, 10], [1, 2, 7, 3, 5, 9], [3, 1, 6, 6, 3, 2], [1, 4, 5, 8, 13, 17],
                                                      [1, 2, 9, 4, 1, 7], [2, 1, 3, 3, 3, 3]];
                let fixed = cases.len();
                for _ in 0..RANDOM {
                    let r = &mut g.rng;
                    cases.push([rr(r, 1, 3), rr(r, 1, 4), rr(r, 1, 9), rr(r, 1, 9), rr(r, 1, 17), rr(r, 1, 17)]);
                }
                for (ci, &c) in cases.iter().enumerate() {
                    let [_, _, h_in, w_in, h_out, w_out] = c;
                    let fh = h_out as f64 / h_in as f64;
                    let fw = w_out as f64 / w_in as f64;
                    // (align_corners, has_h, h_factor, has_w, w_factor); bools as any non-zero byte.
                    // Explicit factors keep the source position in range (as candle's do).
                    let modes: [(u8, u8, f64, u8, f64); 7] = [(0, 0, 0.0, 0, 0.0), (1, 0, 0.0, 0, 0.0), (0, 1, fh, 1, fw), (0, 1, fh * 1.25, 0, 0.0),
                                                              (0, 0, 0.0, 2, fw * 3.0 + 0.1), (7, 1, 0.5, 1, 0.5), (0, 255, fh, 0, 123.0)];
                    if ci < fixed {
                        for li in 0..4 {
                            for (mi, &m) in modes.iter().enumerate() {
                                for mode in 0..2u32 {
                                    if (li > 0 && mi > 2) || (mode == 1 && li > 0) { continue; }
                                    g.bilinear(name, dt, c, m, li, mode, None);
                                }
                            }
                        }
                    } else {
                        let (mi, li, mode) = (rr(&mut g.rng, 0, 6), rr(&mut g.rng, 0, 3), rr(&mut g.rng, 0, 1) as u32);
                        g.bilinear(name, dt, c, modes[mi], li, mode, None);
                    }
                }
                for m in [(0u8, 0u8, 0.0, 0u8, 0.0), (1, 0, 0.0, 0, 0.0)] {
                    g.bilinear(name, dt, [2, 3, 4, 5, 1, BIG], m, 0, 1, Some(300));
                    g.bilinear(name, dt, [2, 3, 4, 5, BIG, 1], m, 2, 0, Some(300));
                }
            }
            "im2col1d" => {
                let mut cases: Vec<[usize; 7]> = vec![[1, 1, 1, 1, 1, 0, 1], [2, 3, 10, 3, 1, 1, 1], [1, 5, 17, 5, 2, 2, 1], [3, 2, 9, 2, 3, 0, 2], [2, 4, 12, 4, 1, 3, 3]];
                let fixed = cases.len();
                for _ in 0..RANDOM {
                    let r = &mut g.rng;
                    let (lk, dl, p) = (rr(r, 1, 6), rr(r, 1, 3), rr(r, 0, 4));
                    let l_in = rr(r, 1, 20).max((dl * (lk - 1) + 1).saturating_sub(2 * p));
                    cases.push([rr(r, 1, 3), rr(r, 1, 6), l_in, lk, rr(r, 1, 4), p, dl]);
                }
                for (ci, &[b, c_in, l_in, lk, s, p, dl]) in cases.iter().enumerate() {
                    let l_out = (l_in + 2 * p - dl * (lk - 1) - 1) / s + 1;
                    let c = [b, c_in, l_in, lk, s, p, dl, l_out];
                    if ci < fixed {
                        for li in 0..4 {
                            g.im2col1(name, dt, c, li, None);
                        }
                    } else {
                        let li = rr(&mut g.rng, 0, 3);
                        g.im2col1(name, dt, c, li, None);
                    }
                }
                g.im2col1(name, dt, [1, 2, 7, 3, 1, 1, 1, BIG], 0, Some(300));
            }
            "im2col" => {
                let mut cases: Vec<[usize; 9]> = vec![[1, 1, 1, 1, 1, 1, 1, 0, 1], [2, 3, 6, 7, 3, 3, 1, 1, 1], [1, 2, 9, 8, 2, 3, 2, 1, 1], [2, 2, 11, 10, 3, 2, 3, 2, 2],
                                                      [1, 3, 5, 5, 5, 5, 1, 2, 1]];
                let fixed = cases.len();
                for _ in 0..RANDOM {
                    let r = &mut g.rng;
                    let (hk, wk, dl, p) = (rr(r, 1, 4), rr(r, 1, 4), rr(r, 1, 2), rr(r, 0, 3));
                    let hh = rr(r, 1, 9).max((dl * (hk - 1) + 1).saturating_sub(2 * p));
                    let ww = rr(r, 1, 9).max((dl * (wk - 1) + 1).saturating_sub(2 * p));
                    cases.push([rr(r, 1, 2), rr(r, 1, 4), hh, ww, hk, wk, rr(r, 1, 3), p, dl]);
                }
                for (ci, &[b, c_in, hh, ww, hk, wk, s, p, dl]) in cases.iter().enumerate() {
                    let h_out = (hh + 2 * p - dl * (hk - 1) - 1) / s + 1;
                    let w_out = (ww + 2 * p - dl * (wk - 1) - 1) / s + 1;
                    let c = [b, c_in, hh, ww, hk, wk, s, p, dl, h_out, w_out];
                    if ci < fixed {
                        for li in 0..4 {
                            g.im2col2(name, dt, c, li, None);
                        }
                    } else {
                        let li = rr(&mut g.rng, 0, 3);
                        g.im2col2(name, dt, c, li, None);
                    }
                }
                g.im2col2(name, dt, [1, 2, 4, 5, 2, 2, 1, 1, 1, 1, BIG], 0, Some(300));
                g.im2col2(name, dt, [1, 2, 4, 5, 2, 2, 1, 1, 1, BIG, 3], 2, Some(300));
            }
            "col2im1d" => {
                let mut cases: Vec<[usize; 5]> = vec![[1, 1, 1, 1, 1], [2, 5, 3, 4, 2], [1, 7, 2, 3, 1], [3, 4, 5, 2, 3], [1, 9, 1, 7, 2], [2, 6, 4, 5, 5], [1, 3, 2, 1, 1]];
                let fixed = cases.len();
                for _ in 0..RANDOM {
                    let r = &mut g.rng;
                    cases.push([rr(r, 1, 3), rr(r, 1, 10), rr(r, 1, 5), rr(r, 1, 6), rr(r, 1, 6)]);
                }
                for (ci, &[b, l_in, c_out, k, s]) in cases.iter().enumerate() {
                    let c = [b, l_in, c_out, k, s, (l_in - 1) * s + k];
                    if ci < fixed {
                        for mode in 0..2 {
                            g.col2im(name, dt, c, mode, None);
                        }
                    } else {
                        let mode = rr(&mut g.rng, 0, 1) as u32;
                        g.col2im(name, dt, c, mode, None);
                    }
                }
                // stride past 32 bits (l_in_idx = 0, k0 truncates), l_out past 32 bits
                g.col2im(name, dt, [2, 3, 2, 4, BIG, 9], 0, None);
                g.col2im(name, dt, [1, 3, 2, 4, 1, BIG], 1, Some(300));
            }
            _ => unreachable!(),
        }
    }
    for m in &missing {
        println!("  MISSING from oxide PTX: {m}");
    }
    let mut per: std::collections::BTreeMap<&str, usize> = Default::default();
    for f in &g.t.failures {
        *per.entry(f.split(' ').next().unwrap()).or_default() += 1;
    }
    for (k, v) in &per {
        println!("  failing entry {k}: {v} launches");
    }
    let ok = g.t.finish("conv") && missing.is_empty();
    println!("conv: {} of {} reference entries ported", names.len() - missing.len(), names.len());
    std::process::exit(if ok { 0 } else { 1 });
}
