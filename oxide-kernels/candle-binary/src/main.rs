//! candle-kernels `binary.cu` (+ `binary_op_macros.cuh`) in cuda-oxide: same 96 entry names, same
//! raw-pointer ABI, bit-identical output (checked by the host `main` against candle's nvcc PTX).
//!
//! Semantics copied from the reference PTX/SASS, not just the source:
//! - four layout branches in the macro's order (both contiguous / lhs only / rhs only / neither);
//!   `unsigned int` (u32) index accumulators with `size_t` (u64) dims/strides, `int d` dim counter;
//! - bf16/f16 add/sub/mul/max/min are native half PTX (`add.bf16`, `max.NaN.f16`, ...) -> same PTX here;
//! - bf16 `/` is cuda_bf16.hpp's `__fdividef`-style path: `div.approx.f32` with the 2^126 rescale;
//! - f16 `/` is cuda_fp16.hpp's `__hdiv`: `rcp.approx.ftz` estimate + one fma refinement when the
//!   f16 result is a tiny non-zero denormal;
//! - half comparisons are exact after widening to f32 (ordered compares, `!=` is unordered);
//! - fp8 e4m3 goes e4m3 -> f16 -> f32, f32 op (`max.f32`/`min.f32` for maxg/ming), satfinite rn back;
//! - integer `/` is the same PTX `div.u16` (u8, via C promotion), `div.u32`, `div.s64`: no trap on 0;
//!   i64 reproduces nvcc's div.u32 bypass when both operands fit in 32 bits (changes x / 0).
use cuda_device::{convert, kernel, ptx_asm, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// candle `is_contiguous` (cuda_utils.cuh).
    #[inline(always)]
    pub unsafe fn is_contiguous(num_dims: usize, dims: *const usize, strides: *const usize) -> bool {
        let mut acc: usize = 1;
        let mut d: u32 = 0;
        while (d as usize) < num_dims {
            let dim_idx = (num_dims - 1 - d as usize) as u32 as usize;
            let dim = *dims.add(dim_idx);
            if dim > 1 && acc != *strides.add(dim_idx) {
                return false;
            }
            acc = acc.wrapping_mul(dim);
            d += 1;
        }
        true
    }

    /// The macro's inner index loop for one operand: `for (int d = num_dims - 1; d >= 0; d--)`.
    #[inline(always)]
    pub unsafe fn strided_index(i: u32, num_dims: usize, dims: *const usize, strides: *const usize) -> u32 {
        let mut tmp_i = i;
        let mut idx: u32 = 0;
        let mut d: i32 = num_dims.wrapping_sub(1) as i32;
        while d >= 0 {
            let dim = *dims.add(d as isize as usize);
            let i_dim = (tmp_i as usize % dim) as u32;
            idx = (idx as usize).wrapping_add((i_dim as usize).wrapping_mul(*strides.add(d as isize as usize))) as u32;
            tmp_i = (tmp_i as usize / dim) as u32;
            d -= 1;
        }
        idx
    }

    /// Same loop, both operands at once (the macro's "neither contiguous" branch).
    #[inline(always)]
    pub unsafe fn strided_index2(i: u32, num_dims: usize, dims: *const usize, ls: *const usize, rs: *const usize) -> (u32, u32) {
        let mut tmp_i = i;
        let mut li: u32 = 0;
        let mut ri: u32 = 0;
        let mut d: i32 = num_dims.wrapping_sub(1) as i32;
        while d >= 0 {
            let dim = *dims.add(d as isize as usize);
            let i_dim = (tmp_i as usize % dim) as u32;
            li = (li as usize).wrapping_add((i_dim as usize).wrapping_mul(*ls.add(d as isize as usize))) as u32;
            ri = (ri as usize).wrapping_add((i_dim as usize).wrapping_mul(*rs.add(d as isize as usize))) as u32;
            tmp_i = (tmp_i as usize / dim) as u32;
            d -= 1;
        }
        (li, ri)
    }

    /// `BINARY_OP_OUT`: the shared body of every binary kernel.
    #[inline(always)]
    pub unsafe fn binary_loop<T: Copy, O: Copy, F: Fn(T, T) -> O>(
        numel: usize, num_dims: usize, info: *const usize, lhs: *const T, rhs: *const T, out: *mut O, op: F,
    ) {
        let dims = info;
        let lhs_strides = info.wrapping_add(num_dims);
        let rhs_strides = info.wrapping_add(2 * num_dims);
        let lhs_cont = info.is_null() || is_contiguous(num_dims, dims, lhs_strides);
        let rhs_cont = info.is_null() || is_contiguous(num_dims, dims, rhs_strides);
        let step = thread::blockDim_x().wrapping_mul(thread::gridDim_x());
        let mut i: u32 = thread::blockIdx_x().wrapping_mul(thread::blockDim_x()).wrapping_add(thread::threadIdx_x());
        if lhs_cont && rhs_cont {
            while (i as usize) < numel {
                *out.add(i as usize) = op(*lhs.add(i as usize), *rhs.add(i as usize));
                i = i.wrapping_add(step);
            }
        } else if lhs_cont {
            while (i as usize) < numel {
                let ri = strided_index(i, num_dims, dims, rhs_strides);
                *out.add(i as usize) = op(*lhs.add(i as usize), *rhs.add(ri as usize));
                i = i.wrapping_add(step);
            }
        } else if rhs_cont {
            while (i as usize) < numel {
                let li = strided_index(i, num_dims, dims, lhs_strides);
                *out.add(i as usize) = op(*lhs.add(li as usize), *rhs.add(i as usize));
                i = i.wrapping_add(step);
            }
        } else {
            while (i as usize) < numel {
                let (li, ri) = strided_index2(i, num_dims, dims, lhs_strides, rhs_strides);
                *out.add(i as usize) = op(*lhs.add(li as usize), *rhs.add(ri as usize));
                i = i.wrapping_add(step);
            }
        }
    }

    // ---- element ops -------------------------------------------------------------------------

    #[inline(always)]
    fn bf16_to_f32(x: u16) -> f32 {
        f32::from_bits((x as u32) << 16)
    }

    #[inline(always)]
    fn f16_to_f32(x: u16) -> f32 {
        convert::cvt_f32_f16x2_lo(x as u32)
    }

    #[inline(always)]
    fn e4m3_to_f32(b: u8) -> f32 {
        convert::cvt_f32_f16x2_lo(convert::cvt_rn_f16x2_e4m3x2(b as u16))
    }

    #[inline(always)]
    fn f32_to_e4m3(v: f32) -> u8 {
        convert::cvt_rn_satfinite_e4m3x2_f32(v, 0.0) as u8
    }

    #[inline(always)]
    fn e4m3_op<F: Fn(f32, f32) -> f32>(x: u8, y: u8, f: F) -> u8 {
        f32_to_e4m3(f(e4m3_to_f32(x), e4m3_to_f32(y)))
    }

    #[inline(always)]
    fn add_bf16(x: u16, y: u16) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("add.bf16 %0, %1, %2;", out("=h") r, in("h") x, in("h") y, options(register_only)); }
        r
    }

    #[inline(always)]
    fn sub_bf16(x: u16, y: u16) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("sub.bf16 %0, %1, %2;", out("=h") r, in("h") x, in("h") y, options(register_only)); }
        r
    }

    #[inline(always)]
    fn mul_bf16(x: u16, y: u16) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("mul.bf16 %0, %1, %2;", out("=h") r, in("h") x, in("h") y, options(register_only)); }
        r
    }

    #[inline(always)]
    fn max_bf16(x: u16, y: u16) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("max.NaN.bf16 %0, %1, %2;", out("=h") r, in("h") x, in("h") y, options(register_only)); }
        r
    }

    #[inline(always)]
    fn min_bf16(x: u16, y: u16) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("min.NaN.bf16 %0, %1, %2;", out("=h") r, in("h") x, in("h") y, options(register_only)); }
        r
    }

    #[inline(always)]
    fn add_f16(x: u16, y: u16) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("add.f16 %0, %1, %2;", out("=h") r, in("h") x, in("h") y, options(register_only)); }
        r
    }

    #[inline(always)]
    fn sub_f16(x: u16, y: u16) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("sub.f16 %0, %1, %2;", out("=h") r, in("h") x, in("h") y, options(register_only)); }
        r
    }

    #[inline(always)]
    fn mul_f16(x: u16, y: u16) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("mul.f16 %0, %1, %2;", out("=h") r, in("h") x, in("h") y, options(register_only)); }
        r
    }

    #[inline(always)]
    fn max_f16(x: u16, y: u16) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("max.NaN.f16 %0, %1, %2;", out("=h") r, in("h") x, in("h") y, options(register_only)); }
        r
    }

    #[inline(always)]
    fn min_f16(x: u16, y: u16) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("min.NaN.f16 %0, %1, %2;", out("=h") r, in("h") x, in("h") y, options(register_only)); }
        r
    }

    #[inline(always)]
    fn div_u16(x: u16, y: u16) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("div.u16 %0, %1, %2;", out("=h") r, in("h") x, in("h") y, options(register_only)); }
        r
    }

    #[inline(always)]
    fn max_f32(x: f32, y: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("max.f32 %0, %1, %2;", out("=f") r, in("f") x, in("f") y, options(register_only)); }
        r
    }

    #[inline(always)]
    fn min_f32(x: f32, y: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("min.f32 %0, %1, %2;", out("=f") r, in("f") x, in("f") y, options(register_only)); }
        r
    }

    #[inline(always)]
    fn div_f32(x: f32, y: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("div.rn.f32 %0, %1, %2;", out("=f") r, in("f") x, in("f") y, options(register_only)); }
        r
    }

    #[inline(always)]
    fn max_f64(x: f64, y: f64) -> f64 {
        let r: f64;
        unsafe { ptx_asm!("max.f64 %0, %1, %2;", out("=d") r, in("d") x, in("d") y, options(register_only)); }
        r
    }

    #[inline(always)]
    fn min_f64(x: f64, y: f64) -> f64 {
        let r: f64;
        unsafe { ptx_asm!("min.f64 %0, %1, %2;", out("=d") r, in("d") x, in("d") y, options(register_only)); }
        r
    }

    #[inline(always)]
    fn div_f64(x: f64, y: f64) -> f64 {
        let r: f64;
        unsafe { ptx_asm!("div.rn.f64 %0, %1, %2;", out("=d") r, in("d") x, in("d") y, options(register_only)); }
        r
    }

    #[inline(always)]
    fn div_u32(x: u32, y: u32) -> u32 {
        let r: u32;
        unsafe { ptx_asm!("div.u32 %0, %1, %2;", out("=r") r, in("r") x, in("r") y, options(register_only)); }
        r
    }

    #[inline(always)]
    fn div_s64(x: i64, y: i64) -> i64 {
        let r: i64;
        unsafe { ptx_asm!("div.s64 %0, %1, %2;", out("=l") r, in("l") x, in("l") y, options(register_only)); }
        r
    }

    /// nvcc's 64-bit division bypass (visible in the reference PTX): when both operands fit in
    /// 32 unsigned bits it uses `div.u32`. Observable: `x / 0` is 0x00000000_ffffffff there, not -1.
    #[inline(always)]
    fn div_i64(x: i64, y: i64) -> i64 {
        if ((x | y) as u64) & 0xffff_ffff_0000_0000 == 0 {
            div_u32(x as u32, y as u32) as u64 as i64
        } else {
            div_s64(x, y)
        }
    }

    /// cuda_bf16.hpp `__hdiv` on sm_80+: `__fdividef`-style approximate divide with the 2^126 rescale.
    #[inline(always)]
    fn div_bf16(x: u16, y: u16) -> u16 {
        let r: u16;
        unsafe {
            ptx_asm!(
                "{ .reg .f32 a, b, ab, bs, q, qs; .reg .pred p; cvt.f32.bf16 a, %1; cvt.f32.bf16 b, %2; abs.f32 ab, b; setp.ge.f32 p, ab, 0f7E800000; mul.f32 bs, b, 0f3E800000; selp.f32 b, bs, b, p; div.approx.f32 q, a, b; mul.f32 qs, q, 0f3E800000; selp.f32 q, qs, q, p; cvt.rn.bf16.f32 %0, q; }",
                out("=h") r, in("h") x, in("h") y, options(register_only),
            );
        }
        r
    }

    /// cuda_fp16.hpp `__hdiv`: rcp.approx.ftz estimate, one fma refinement for tiny non-zero results.
    #[inline(always)]
    fn div_f16(x: u16, y: u16) -> u16 {
        let r: u16;
        unsafe {
            ptx_asm!(
                "{ .reg .f32 a, b, rc, f, nb, e, f2, z; .reg .b16 h, ah, k, zh, h2; .reg .pred p, q; cvt.f32.f16 a, %1; cvt.f32.f16 b, %2; rcp.approx.ftz.f32 rc, b; mul.f32 f, a, rc; cvt.rn.f16.f32 h, f; abs.f16 ah, h; mov.b16 k, 143; setp.lt.f16 p, ah, k; mov.b32 z, 0; cvt.rn.f16.f32 zh, z; setp.lt.f16 q, zh, ah; and.pred p, p, q; neg.f32 nb, b; fma.rn.f32 e, nb, f, a; fma.rn.f32 f2, rc, e, f; cvt.rn.f16.f32 h2, f2; selp.b16 %0, h2, h, p; }",
                out("=h") r, in("h") x, in("h") y, options(register_only),
            );
        }
        r
    }

    #[kernel]
    pub unsafe fn badd_bf16(numel: usize, num_dims: usize, info: *const usize, lhs: *const u16, rhs: *const u16, out: *mut u16) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u16, y: u16| -> u16 { add_bf16(x, y) });
    }

    #[kernel]
    pub unsafe fn bdiv_bf16(numel: usize, num_dims: usize, info: *const usize, lhs: *const u16, rhs: *const u16, out: *mut u16) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u16, y: u16| -> u16 { div_bf16(x, y) });
    }

    #[kernel]
    pub unsafe fn bmul_bf16(numel: usize, num_dims: usize, info: *const usize, lhs: *const u16, rhs: *const u16, out: *mut u16) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u16, y: u16| -> u16 { mul_bf16(x, y) });
    }

    #[kernel]
    pub unsafe fn bsub_bf16(numel: usize, num_dims: usize, info: *const usize, lhs: *const u16, rhs: *const u16, out: *mut u16) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u16, y: u16| -> u16 { sub_bf16(x, y) });
    }

    #[kernel]
    pub unsafe fn bmaximum_bf16(numel: usize, num_dims: usize, info: *const usize, lhs: *const u16, rhs: *const u16, out: *mut u16) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u16, y: u16| -> u16 { max_bf16(x, y) });
    }

    #[kernel]
    pub unsafe fn bminimum_bf16(numel: usize, num_dims: usize, info: *const usize, lhs: *const u16, rhs: *const u16, out: *mut u16) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u16, y: u16| -> u16 { min_bf16(x, y) });
    }

    #[kernel]
    pub unsafe fn eq_bf16(numel: usize, num_dims: usize, info: *const usize, lhs: *const u16, rhs: *const u16, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u16, y: u16| -> u8 { (bf16_to_f32(x) == bf16_to_f32(y)) as u8 });
    }

    #[kernel]
    pub unsafe fn ne_bf16(numel: usize, num_dims: usize, info: *const usize, lhs: *const u16, rhs: *const u16, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u16, y: u16| -> u8 { (bf16_to_f32(x) != bf16_to_f32(y)) as u8 });
    }

    #[kernel]
    pub unsafe fn lt_bf16(numel: usize, num_dims: usize, info: *const usize, lhs: *const u16, rhs: *const u16, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u16, y: u16| -> u8 { (bf16_to_f32(x) < bf16_to_f32(y)) as u8 });
    }

    #[kernel]
    pub unsafe fn le_bf16(numel: usize, num_dims: usize, info: *const usize, lhs: *const u16, rhs: *const u16, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u16, y: u16| -> u8 { (bf16_to_f32(x) <= bf16_to_f32(y)) as u8 });
    }

    #[kernel]
    pub unsafe fn gt_bf16(numel: usize, num_dims: usize, info: *const usize, lhs: *const u16, rhs: *const u16, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u16, y: u16| -> u8 { (bf16_to_f32(x) > bf16_to_f32(y)) as u8 });
    }

    #[kernel]
    pub unsafe fn ge_bf16(numel: usize, num_dims: usize, info: *const usize, lhs: *const u16, rhs: *const u16, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u16, y: u16| -> u8 { (bf16_to_f32(x) >= bf16_to_f32(y)) as u8 });
    }

    #[kernel]
    pub unsafe fn badd_f8_e4m3(numel: usize, num_dims: usize, info: *const usize, lhs: *const u8, rhs: *const u8, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u8, y: u8| -> u8 { e4m3_op(x, y, |a, b| a + b) });
    }

    #[kernel]
    pub unsafe fn bdiv_f8_e4m3(numel: usize, num_dims: usize, info: *const usize, lhs: *const u8, rhs: *const u8, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u8, y: u8| -> u8 { e4m3_op(x, y, |a, b| div_f32(a, b)) });
    }

    #[kernel]
    pub unsafe fn bmul_f8_e4m3(numel: usize, num_dims: usize, info: *const usize, lhs: *const u8, rhs: *const u8, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u8, y: u8| -> u8 { e4m3_op(x, y, |a, b| a * b) });
    }

    #[kernel]
    pub unsafe fn bsub_f8_e4m3(numel: usize, num_dims: usize, info: *const usize, lhs: *const u8, rhs: *const u8, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u8, y: u8| -> u8 { e4m3_op(x, y, |a, b| a - b) });
    }

    #[kernel]
    pub unsafe fn bmaximum_f8_e4m3(numel: usize, num_dims: usize, info: *const usize, lhs: *const u8, rhs: *const u8, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u8, y: u8| -> u8 { e4m3_op(x, y, |a, b| max_f32(a, b)) });
    }

    #[kernel]
    pub unsafe fn bminimum_f8_e4m3(numel: usize, num_dims: usize, info: *const usize, lhs: *const u8, rhs: *const u8, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u8, y: u8| -> u8 { e4m3_op(x, y, |a, b| min_f32(a, b)) });
    }

    #[kernel]
    pub unsafe fn eq_f8_e4m3(numel: usize, num_dims: usize, info: *const usize, lhs: *const u8, rhs: *const u8, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u8, y: u8| -> u8 { (e4m3_to_f32(x) == e4m3_to_f32(y)) as u8 });
    }

    #[kernel]
    pub unsafe fn ne_f8_e4m3(numel: usize, num_dims: usize, info: *const usize, lhs: *const u8, rhs: *const u8, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u8, y: u8| -> u8 { (e4m3_to_f32(x) != e4m3_to_f32(y)) as u8 });
    }

    #[kernel]
    pub unsafe fn lt_f8_e4m3(numel: usize, num_dims: usize, info: *const usize, lhs: *const u8, rhs: *const u8, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u8, y: u8| -> u8 { (e4m3_to_f32(x) < e4m3_to_f32(y)) as u8 });
    }

    #[kernel]
    pub unsafe fn le_f8_e4m3(numel: usize, num_dims: usize, info: *const usize, lhs: *const u8, rhs: *const u8, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u8, y: u8| -> u8 { (e4m3_to_f32(x) <= e4m3_to_f32(y)) as u8 });
    }

    #[kernel]
    pub unsafe fn gt_f8_e4m3(numel: usize, num_dims: usize, info: *const usize, lhs: *const u8, rhs: *const u8, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u8, y: u8| -> u8 { (e4m3_to_f32(x) > e4m3_to_f32(y)) as u8 });
    }

    #[kernel]
    pub unsafe fn ge_f8_e4m3(numel: usize, num_dims: usize, info: *const usize, lhs: *const u8, rhs: *const u8, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u8, y: u8| -> u8 { (e4m3_to_f32(x) >= e4m3_to_f32(y)) as u8 });
    }

    #[kernel]
    pub unsafe fn badd_f16(numel: usize, num_dims: usize, info: *const usize, lhs: *const u16, rhs: *const u16, out: *mut u16) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u16, y: u16| -> u16 { add_f16(x, y) });
    }

    #[kernel]
    pub unsafe fn bdiv_f16(numel: usize, num_dims: usize, info: *const usize, lhs: *const u16, rhs: *const u16, out: *mut u16) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u16, y: u16| -> u16 { div_f16(x, y) });
    }

    #[kernel]
    pub unsafe fn bmul_f16(numel: usize, num_dims: usize, info: *const usize, lhs: *const u16, rhs: *const u16, out: *mut u16) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u16, y: u16| -> u16 { mul_f16(x, y) });
    }

    #[kernel]
    pub unsafe fn bsub_f16(numel: usize, num_dims: usize, info: *const usize, lhs: *const u16, rhs: *const u16, out: *mut u16) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u16, y: u16| -> u16 { sub_f16(x, y) });
    }

    #[kernel]
    pub unsafe fn bmaximum_f16(numel: usize, num_dims: usize, info: *const usize, lhs: *const u16, rhs: *const u16, out: *mut u16) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u16, y: u16| -> u16 { max_f16(x, y) });
    }

    #[kernel]
    pub unsafe fn bminimum_f16(numel: usize, num_dims: usize, info: *const usize, lhs: *const u16, rhs: *const u16, out: *mut u16) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u16, y: u16| -> u16 { min_f16(x, y) });
    }

    #[kernel]
    pub unsafe fn eq_f16(numel: usize, num_dims: usize, info: *const usize, lhs: *const u16, rhs: *const u16, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u16, y: u16| -> u8 { (f16_to_f32(x) == f16_to_f32(y)) as u8 });
    }

    #[kernel]
    pub unsafe fn ne_f16(numel: usize, num_dims: usize, info: *const usize, lhs: *const u16, rhs: *const u16, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u16, y: u16| -> u8 { (f16_to_f32(x) != f16_to_f32(y)) as u8 });
    }

    #[kernel]
    pub unsafe fn lt_f16(numel: usize, num_dims: usize, info: *const usize, lhs: *const u16, rhs: *const u16, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u16, y: u16| -> u8 { (f16_to_f32(x) < f16_to_f32(y)) as u8 });
    }

    #[kernel]
    pub unsafe fn le_f16(numel: usize, num_dims: usize, info: *const usize, lhs: *const u16, rhs: *const u16, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u16, y: u16| -> u8 { (f16_to_f32(x) <= f16_to_f32(y)) as u8 });
    }

    #[kernel]
    pub unsafe fn gt_f16(numel: usize, num_dims: usize, info: *const usize, lhs: *const u16, rhs: *const u16, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u16, y: u16| -> u8 { (f16_to_f32(x) > f16_to_f32(y)) as u8 });
    }

    #[kernel]
    pub unsafe fn ge_f16(numel: usize, num_dims: usize, info: *const usize, lhs: *const u16, rhs: *const u16, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u16, y: u16| -> u8 { (f16_to_f32(x) >= f16_to_f32(y)) as u8 });
    }

    #[kernel]
    pub unsafe fn badd_f32(numel: usize, num_dims: usize, info: *const usize, lhs: *const f32, rhs: *const f32, out: *mut f32) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: f32, y: f32| -> f32 { x + y });
    }

    #[kernel]
    pub unsafe fn bdiv_f32(numel: usize, num_dims: usize, info: *const usize, lhs: *const f32, rhs: *const f32, out: *mut f32) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: f32, y: f32| -> f32 { div_f32(x, y) });
    }

    #[kernel]
    pub unsafe fn bmul_f32(numel: usize, num_dims: usize, info: *const usize, lhs: *const f32, rhs: *const f32, out: *mut f32) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: f32, y: f32| -> f32 { x * y });
    }

    #[kernel]
    pub unsafe fn bsub_f32(numel: usize, num_dims: usize, info: *const usize, lhs: *const f32, rhs: *const f32, out: *mut f32) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: f32, y: f32| -> f32 { x - y });
    }

    #[kernel]
    pub unsafe fn bmaximum_f32(numel: usize, num_dims: usize, info: *const usize, lhs: *const f32, rhs: *const f32, out: *mut f32) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: f32, y: f32| -> f32 { max_f32(x, y) });
    }

    #[kernel]
    pub unsafe fn bminimum_f32(numel: usize, num_dims: usize, info: *const usize, lhs: *const f32, rhs: *const f32, out: *mut f32) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: f32, y: f32| -> f32 { min_f32(x, y) });
    }

    #[kernel]
    pub unsafe fn eq_f32(numel: usize, num_dims: usize, info: *const usize, lhs: *const f32, rhs: *const f32, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: f32, y: f32| -> u8 { (x == y) as u8 });
    }

    #[kernel]
    pub unsafe fn ne_f32(numel: usize, num_dims: usize, info: *const usize, lhs: *const f32, rhs: *const f32, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: f32, y: f32| -> u8 { (x != y) as u8 });
    }

    #[kernel]
    pub unsafe fn lt_f32(numel: usize, num_dims: usize, info: *const usize, lhs: *const f32, rhs: *const f32, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: f32, y: f32| -> u8 { (x < y) as u8 });
    }

    #[kernel]
    pub unsafe fn le_f32(numel: usize, num_dims: usize, info: *const usize, lhs: *const f32, rhs: *const f32, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: f32, y: f32| -> u8 { (x <= y) as u8 });
    }

    #[kernel]
    pub unsafe fn gt_f32(numel: usize, num_dims: usize, info: *const usize, lhs: *const f32, rhs: *const f32, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: f32, y: f32| -> u8 { (x > y) as u8 });
    }

    #[kernel]
    pub unsafe fn ge_f32(numel: usize, num_dims: usize, info: *const usize, lhs: *const f32, rhs: *const f32, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: f32, y: f32| -> u8 { (x >= y) as u8 });
    }

    #[kernel]
    pub unsafe fn badd_f64(numel: usize, num_dims: usize, info: *const usize, lhs: *const f64, rhs: *const f64, out: *mut f64) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: f64, y: f64| -> f64 { x + y });
    }

    #[kernel]
    pub unsafe fn bdiv_f64(numel: usize, num_dims: usize, info: *const usize, lhs: *const f64, rhs: *const f64, out: *mut f64) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: f64, y: f64| -> f64 { div_f64(x, y) });
    }

    #[kernel]
    pub unsafe fn bmul_f64(numel: usize, num_dims: usize, info: *const usize, lhs: *const f64, rhs: *const f64, out: *mut f64) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: f64, y: f64| -> f64 { x * y });
    }

    #[kernel]
    pub unsafe fn bsub_f64(numel: usize, num_dims: usize, info: *const usize, lhs: *const f64, rhs: *const f64, out: *mut f64) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: f64, y: f64| -> f64 { x - y });
    }

    #[kernel]
    pub unsafe fn bmaximum_f64(numel: usize, num_dims: usize, info: *const usize, lhs: *const f64, rhs: *const f64, out: *mut f64) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: f64, y: f64| -> f64 { max_f64(x, y) });
    }

    #[kernel]
    pub unsafe fn bminimum_f64(numel: usize, num_dims: usize, info: *const usize, lhs: *const f64, rhs: *const f64, out: *mut f64) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: f64, y: f64| -> f64 { min_f64(x, y) });
    }

    #[kernel]
    pub unsafe fn eq_f64(numel: usize, num_dims: usize, info: *const usize, lhs: *const f64, rhs: *const f64, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: f64, y: f64| -> u8 { (x == y) as u8 });
    }

    #[kernel]
    pub unsafe fn ne_f64(numel: usize, num_dims: usize, info: *const usize, lhs: *const f64, rhs: *const f64, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: f64, y: f64| -> u8 { (x != y) as u8 });
    }

    #[kernel]
    pub unsafe fn lt_f64(numel: usize, num_dims: usize, info: *const usize, lhs: *const f64, rhs: *const f64, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: f64, y: f64| -> u8 { (x < y) as u8 });
    }

    #[kernel]
    pub unsafe fn le_f64(numel: usize, num_dims: usize, info: *const usize, lhs: *const f64, rhs: *const f64, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: f64, y: f64| -> u8 { (x <= y) as u8 });
    }

    #[kernel]
    pub unsafe fn gt_f64(numel: usize, num_dims: usize, info: *const usize, lhs: *const f64, rhs: *const f64, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: f64, y: f64| -> u8 { (x > y) as u8 });
    }

    #[kernel]
    pub unsafe fn ge_f64(numel: usize, num_dims: usize, info: *const usize, lhs: *const f64, rhs: *const f64, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: f64, y: f64| -> u8 { (x >= y) as u8 });
    }

    #[kernel]
    pub unsafe fn badd_u8(numel: usize, num_dims: usize, info: *const usize, lhs: *const u8, rhs: *const u8, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u8, y: u8| -> u8 { x.wrapping_add(y) });
    }

    #[kernel]
    pub unsafe fn bdiv_u8(numel: usize, num_dims: usize, info: *const usize, lhs: *const u8, rhs: *const u8, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u8, y: u8| -> u8 { div_u16(x as u16, y as u16) as u8 });
    }

    #[kernel]
    pub unsafe fn bmul_u8(numel: usize, num_dims: usize, info: *const usize, lhs: *const u8, rhs: *const u8, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u8, y: u8| -> u8 { x.wrapping_mul(y) });
    }

    #[kernel]
    pub unsafe fn bsub_u8(numel: usize, num_dims: usize, info: *const usize, lhs: *const u8, rhs: *const u8, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u8, y: u8| -> u8 { x.wrapping_sub(y) });
    }

    #[kernel]
    pub unsafe fn bmaximum_u8(numel: usize, num_dims: usize, info: *const usize, lhs: *const u8, rhs: *const u8, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u8, y: u8| -> u8 { if x > y { x } else { y } });
    }

    #[kernel]
    pub unsafe fn bminimum_u8(numel: usize, num_dims: usize, info: *const usize, lhs: *const u8, rhs: *const u8, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u8, y: u8| -> u8 { if x < y { x } else { y } });
    }

    #[kernel]
    pub unsafe fn eq_u8(numel: usize, num_dims: usize, info: *const usize, lhs: *const u8, rhs: *const u8, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u8, y: u8| -> u8 { (x == y) as u8 });
    }

    #[kernel]
    pub unsafe fn ne_u8(numel: usize, num_dims: usize, info: *const usize, lhs: *const u8, rhs: *const u8, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u8, y: u8| -> u8 { (x != y) as u8 });
    }

    #[kernel]
    pub unsafe fn lt_u8(numel: usize, num_dims: usize, info: *const usize, lhs: *const u8, rhs: *const u8, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u8, y: u8| -> u8 { (x < y) as u8 });
    }

    #[kernel]
    pub unsafe fn le_u8(numel: usize, num_dims: usize, info: *const usize, lhs: *const u8, rhs: *const u8, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u8, y: u8| -> u8 { (x <= y) as u8 });
    }

    #[kernel]
    pub unsafe fn gt_u8(numel: usize, num_dims: usize, info: *const usize, lhs: *const u8, rhs: *const u8, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u8, y: u8| -> u8 { (x > y) as u8 });
    }

    #[kernel]
    pub unsafe fn ge_u8(numel: usize, num_dims: usize, info: *const usize, lhs: *const u8, rhs: *const u8, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u8, y: u8| -> u8 { (x >= y) as u8 });
    }

    #[kernel]
    pub unsafe fn badd_u32(numel: usize, num_dims: usize, info: *const usize, lhs: *const u32, rhs: *const u32, out: *mut u32) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u32, y: u32| -> u32 { x.wrapping_add(y) });
    }

    #[kernel]
    pub unsafe fn bdiv_u32(numel: usize, num_dims: usize, info: *const usize, lhs: *const u32, rhs: *const u32, out: *mut u32) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u32, y: u32| -> u32 { div_u32(x, y) });
    }

    #[kernel]
    pub unsafe fn bmul_u32(numel: usize, num_dims: usize, info: *const usize, lhs: *const u32, rhs: *const u32, out: *mut u32) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u32, y: u32| -> u32 { x.wrapping_mul(y) });
    }

    #[kernel]
    pub unsafe fn bsub_u32(numel: usize, num_dims: usize, info: *const usize, lhs: *const u32, rhs: *const u32, out: *mut u32) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u32, y: u32| -> u32 { x.wrapping_sub(y) });
    }

    #[kernel]
    pub unsafe fn bmaximum_u32(numel: usize, num_dims: usize, info: *const usize, lhs: *const u32, rhs: *const u32, out: *mut u32) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u32, y: u32| -> u32 { if x > y { x } else { y } });
    }

    #[kernel]
    pub unsafe fn bminimum_u32(numel: usize, num_dims: usize, info: *const usize, lhs: *const u32, rhs: *const u32, out: *mut u32) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u32, y: u32| -> u32 { if x < y { x } else { y } });
    }

    #[kernel]
    pub unsafe fn eq_u32(numel: usize, num_dims: usize, info: *const usize, lhs: *const u32, rhs: *const u32, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u32, y: u32| -> u8 { (x == y) as u8 });
    }

    #[kernel]
    pub unsafe fn ne_u32(numel: usize, num_dims: usize, info: *const usize, lhs: *const u32, rhs: *const u32, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u32, y: u32| -> u8 { (x != y) as u8 });
    }

    #[kernel]
    pub unsafe fn lt_u32(numel: usize, num_dims: usize, info: *const usize, lhs: *const u32, rhs: *const u32, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u32, y: u32| -> u8 { (x < y) as u8 });
    }

    #[kernel]
    pub unsafe fn le_u32(numel: usize, num_dims: usize, info: *const usize, lhs: *const u32, rhs: *const u32, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u32, y: u32| -> u8 { (x <= y) as u8 });
    }

    #[kernel]
    pub unsafe fn gt_u32(numel: usize, num_dims: usize, info: *const usize, lhs: *const u32, rhs: *const u32, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u32, y: u32| -> u8 { (x > y) as u8 });
    }

    #[kernel]
    pub unsafe fn ge_u32(numel: usize, num_dims: usize, info: *const usize, lhs: *const u32, rhs: *const u32, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: u32, y: u32| -> u8 { (x >= y) as u8 });
    }

    #[kernel]
    pub unsafe fn badd_i64(numel: usize, num_dims: usize, info: *const usize, lhs: *const i64, rhs: *const i64, out: *mut i64) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: i64, y: i64| -> i64 { x.wrapping_add(y) });
    }

    #[kernel]
    pub unsafe fn bdiv_i64(numel: usize, num_dims: usize, info: *const usize, lhs: *const i64, rhs: *const i64, out: *mut i64) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: i64, y: i64| -> i64 { div_i64(x, y) });
    }

    #[kernel]
    pub unsafe fn bmul_i64(numel: usize, num_dims: usize, info: *const usize, lhs: *const i64, rhs: *const i64, out: *mut i64) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: i64, y: i64| -> i64 { x.wrapping_mul(y) });
    }

    #[kernel]
    pub unsafe fn bsub_i64(numel: usize, num_dims: usize, info: *const usize, lhs: *const i64, rhs: *const i64, out: *mut i64) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: i64, y: i64| -> i64 { x.wrapping_sub(y) });
    }

    #[kernel]
    pub unsafe fn bmaximum_i64(numel: usize, num_dims: usize, info: *const usize, lhs: *const i64, rhs: *const i64, out: *mut i64) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: i64, y: i64| -> i64 { if x > y { x } else { y } });
    }

    #[kernel]
    pub unsafe fn bminimum_i64(numel: usize, num_dims: usize, info: *const usize, lhs: *const i64, rhs: *const i64, out: *mut i64) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: i64, y: i64| -> i64 { if x < y { x } else { y } });
    }

    #[kernel]
    pub unsafe fn eq_i64(numel: usize, num_dims: usize, info: *const usize, lhs: *const i64, rhs: *const i64, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: i64, y: i64| -> u8 { (x == y) as u8 });
    }

    #[kernel]
    pub unsafe fn ne_i64(numel: usize, num_dims: usize, info: *const usize, lhs: *const i64, rhs: *const i64, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: i64, y: i64| -> u8 { (x != y) as u8 });
    }

    #[kernel]
    pub unsafe fn lt_i64(numel: usize, num_dims: usize, info: *const usize, lhs: *const i64, rhs: *const i64, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: i64, y: i64| -> u8 { (x < y) as u8 });
    }

    #[kernel]
    pub unsafe fn le_i64(numel: usize, num_dims: usize, info: *const usize, lhs: *const i64, rhs: *const i64, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: i64, y: i64| -> u8 { (x <= y) as u8 });
    }

    #[kernel]
    pub unsafe fn gt_i64(numel: usize, num_dims: usize, info: *const usize, lhs: *const i64, rhs: *const i64, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: i64, y: i64| -> u8 { (x > y) as u8 });
    }

    #[kernel]
    pub unsafe fn ge_i64(numel: usize, num_dims: usize, info: *const usize, lhs: *const i64, rhs: *const i64, out: *mut u8) {
        binary_loop(numel, num_dims, info, lhs, rhs, out, |x: i64, y: i64| -> u8 { (x >= y) as u8 });
    }

}

// ---------------------------------------------------------------------------------------------
// Differential gate against candle's nvcc-built binary.ptx.
use kdiff::{Arg, Harness, Rng, Tally, as_bytes};

/// Element values for one dtype, as raw bytes: 3/8 from a pool of edge cases (so equal pairs,
/// NaN/inf/+-0/denormal operands and zero divisors all occur), the rest random bit patterns.
fn values(tname: &str, n: usize, rng: &mut Rng) -> Vec<u8> {
    let pick = |rng: &mut Rng| rng.next() % 8 < 3;
    match tname {
        "f32" => {
            let pool: [u32; 16] = [0, 0x8000_0000, 0x7f80_0000, 0xff80_0000, 0x7fc0_0000, 0xffc0_0000, 0x7f80_0001, 1,
                                   0x807f_ffff, 0x3f80_0000, 0xbf80_0000, 0x7f7f_ffff, 0x0080_0000, 0x4000_0000, 0x3f00_0000, 0xc0a0_0000];
            let r = rng.f32s(n);
            as_bytes(&r.iter().map(|v| if pick(rng) { pool[(rng.next() % 16) as usize] } else { v.to_bits() }).collect::<Vec<u32>>())
        }
        "f64" => {
            let pool: [u64; 14] = [0, 1 << 63, 0x7ff0 << 48, 0xfff0 << 48, 0x7ff8 << 48, 0xfff8 << 48, (0x7ff0 << 48) | 1, 1,
                                   0x800f_ffff_ffff_ffff, 0x3ff0 << 48, 0xbff0 << 48, 0x7fef_ffff_ffff_ffff, 0x0010 << 48, 0x4000 << 48];
            let r = rng.f64s(n);
            as_bytes(&r.iter().map(|v| if pick(rng) { pool[(rng.next() % 14) as usize] } else { v.to_bits() }).collect::<Vec<u64>>())
        }
        "f16" | "bf16" => {
            let pool: [u16; 16] = if tname == "f16" {
                [0, 0x8000, 0x7c00, 0xfc00, 0x7e00, 0xfe00, 0x7c01, 1, 0x83ff, 0x3c00, 0xbc00, 0x7bff, 0x0400, 0x008f, 0x0090, 0x4000]
            } else {
                [0, 0x8000, 0x7f80, 0xff80, 0x7fc0, 0xffc0, 0x7f81, 1, 0x807f, 0x3f80, 0xbf80, 0x7f7f, 0x0080, 0x7e80, 0x7f00, 0x4000]
            };
            let r = rng.b16s(n);
            as_bytes(&r.iter().map(|&v| if pick(rng) { pool[(rng.next() % 16) as usize] } else { v }).collect::<Vec<u16>>())
        }
        "f8_e4m3" | "u8" => {
            let pool: [u8; 8] = if tname == "u8" { [0, 1, 255, 2, 0x80, 7, 0, 1] } else { [0, 0x80, 0x7f, 0xff, 0x7e, 0x01, 0x38, 0xb8] };
            (0..n).map(|_| if pick(rng) { pool[(rng.next() % 8) as usize] } else { rng.next() as u8 }).collect()
        }
        "u32" => {
            let pool: [u32; 6] = [0, 1, u32::MAX, 2, 0x8000_0000, 7];
            as_bytes(&(0..n).map(|_| if pick(rng) { pool[(rng.next() % 6) as usize] } else { rng.next() as u32 }).collect::<Vec<u32>>())
        }
        "i64" => {
            let pool: [i64; 7] = [0, 1, -1, i64::MIN, i64::MAX, 2, -7];
            // small magnitudes too, so quotients are not almost always 0 / -1
            as_bytes(&(0..n).map(|_| match rng.next() % 8 {
                0..=2 => pool[(rng.next() % 7) as usize],
                3..=4 => (rng.next() as i64) >> (rng.next() % 64),
                _ => rng.next() as i64,
            }).collect::<Vec<i64>>())
        }
        _ => unreachable!(),
    }
}

fn span(dims: &[usize], strides: &[usize]) -> usize {
    dims.iter().zip(strides).map(|(d, s)| (d.max(&1) - 1) * s).sum::<usize>() + 1
}

fn main() {
    let root = format!("{}/titan-engine/oxide-kernels", std::env::var("HOME").unwrap());
    let h = Harness::new(&format!("{root}/reference/candle/binary.ptx"), &format!("{root}/candle-binary/candle_binary.ptx"));
    let mut t = Tally::default();
    let mut rng = Rng(0xB1_4A_27);
    let types: [(&str, usize); 8] = [("bf16", 2), ("f8_e4m3", 1), ("f16", 2), ("f32", 4), ("f64", 8), ("u8", 1), ("u32", 4), ("i64", 8)];
    let ops = ["badd", "bdiv", "bmul", "bsub", "bmaximum", "bminimum", "eq", "ne", "lt", "le", "gt", "ge"];
    // (dims, lhs strides, rhs strides, pass info?)
    type L = (Vec<usize>, Vec<usize>, Vec<usize>, bool);
    let layouts: Vec<L> = vec![
        (vec![1000], vec![1], vec![1], false),                                  // info == nullptr
        (vec![7, 13, 5], vec![65, 5, 1], vec![65, 5, 1], true),                 // both contiguous
        (vec![13, 7], vec![1, 13], vec![7, 1], true),                           // lhs transposed only
        (vec![13, 7], vec![7, 1], vec![1, 13], true),                           // rhs transposed only
        (vec![3, 4, 5], vec![40, 1, 8], vec![1, 3, 12], true),                  // both strided (permuted / transposed)
        (vec![6, 5], vec![5, 1], vec![0, 1], true),                             // rhs row broadcast
        (vec![6, 5], vec![1, 0], vec![5, 1], true),                             // lhs column broadcast
        (vec![4, 1, 9], vec![9, 9, 1], vec![9, 77, 1], true),                   // size-1 dim, odd stride: still contiguous
        (vec![3, 1, 4], vec![1, 5, 3], vec![0, 0, 1], true),                    // size-1 dim, both strided
        (vec![5, 7], vec![0, 0], vec![0, 0], true),                             // both scalar broadcast
        (vec![2, 3, 4, 5], vec![60, 20, 5, 1], vec![1, 2, 6, 24], true),        // 4-d, rhs reversed
        (vec![2, 3, 4, 5], vec![3, 1, 30, 6], vec![60, 20, 5, 1], true),        // 4-d, lhs permuted
        (vec![], vec![], vec![], true),                                         // num_dims == 0, numel 1
        (vec![1], vec![1], vec![1], true),                                      // single element
        (vec![257], vec![2], vec![3], true),                                    // odd 1-d, both strided
        (vec![1 << 18], vec![1], vec![1], false),                               // bulk value coverage
    ];
    for (tname, es) in types {
        for op in ops {
            let name = format!("{op}_{tname}");
            let oes = if op.starts_with('b') { es } else { 1 };
            for (li, (dims, ls, rs, with_info)) in layouts.iter().enumerate() {
                let numel: usize = dims.iter().product();
                let bulk = numel > 100_000;
                let grids: Vec<(u32, u32)> = if bulk { vec![(1024, 256), (7, 96)] } else {
                    vec![(4, 64), (((numel as u32) + 255) / 256, 256), (1, 32), (3, 33)]
                };
                let (lspan, rspan) = (span(dims, ls).max(numel), span(dims, rs).max(numel));
                let mut info: Vec<u64> = Vec::new();
                for v in [dims, ls, rs] { info.extend(v.iter().map(|&x| x as u64)); }
                for (grid, block) in grids {
                    let lhs = values(tname, lspan, &mut rng);
                    let rhs = values(tname, rspan, &mut rng);
                    let out = rng.bytes(numel * oes);
                    let info_arg = if *with_info { Arg::Buf(3) } else { Arg::Null };
                    let args = [Arg::U64(numel as u64), Arg::U64(dims.len() as u64), info_arg, Arg::Buf(0), Arg::Buf(1), Arg::Buf(2)];
                    let bufs = vec![lhs, rhs, out, as_bytes(&info)];
                    let d = h.diff(&name, (grid, 1, 1), (block, 1, 1), 0, &args, &bufs, &[2]);
                    t.record(&format!("{name} layout{li} g{grid}x{block}"), &d);
                }
            }
        }
    }
    // Half division at the edges of its special paths: quotients that land in the f16 denormal
    // range (where __hdiv refines with fma) and bf16 divisors around the 2^126 / 2^-126 rescales.
    for (name, lo_mask, hi_base, hi_mask) in [("bdiv_f16", 0x07ffu16, 0x3c00u16, 0x1fffu16), ("bdiv_bf16", 0x00ff, 0x7e00, 0x01ff), ("bdiv_bf16", 0x3fff, 0x0000, 0x00ff)] {
        for _ in 0..4 {
            let n = 1usize << 20;
            let lhs: Vec<u16> = (0..n).map(|_| { let r = rng.next(); (r as u16 & lo_mask) | (((r >> 20) as u16) & 0x8000) }).collect();
            let rhs: Vec<u16> = (0..n).map(|_| { let r = rng.next(); hi_base.wrapping_add(r as u16 & hi_mask) | (((r >> 20) as u16) & 0x8000) }).collect();
            let args = [Arg::U64(n as u64), Arg::U64(1), Arg::Null, Arg::Buf(0), Arg::Buf(1), Arg::Buf(2)];
            let bufs = vec![as_bytes(&lhs), as_bytes(&rhs), rng.bytes(n * 2)];
            let d = h.diff(name, (1024, 1, 1), (256, 1, 1), 0, &args, &bufs, &[2]);
            t.record(&format!("{name} edge-range"), &d);
        }
    }
    std::process::exit(if t.finish("binary") { 0 } else { 1 });
}
