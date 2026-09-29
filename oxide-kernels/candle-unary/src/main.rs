//! candle-kernels `unary.cu` in cuda-oxide: all 123 entries, same names, same raw-pointer ABI,
//! bit-identical output (checked by the host `main` against candle's own nvcc PTX).
//!
//! Semantics copied from the reference PTX/SASS, not just the source:
//! - f32/f64 maths calls the same libdevice `__nv_*` functions nvcc resolves `expf`, `tanhf`, ... to;
//! - nvcc contracts libdevice `expf`'s final multiply with the caller's `1 + e` / `e - 1`
//!   (sigmoid, silu, elu in f32 and fp8), so those use an explicit expf split + fma;
//! - f16/bf16 intrinsics (hexp, hlog, hsqrt, hsin/hcos fix-ups, __hdiv, mul/add/setp) are the
//!   cuda_fp16/cuda_bf16 headers' inline PTX, verbatim, so ptxas makes the same fusions
//!   (e.g. `mul.f16` + `add.f16` in gelu -> HFMA2);
//! - half constants are converted at run time from double / int exactly as the C++ casts do;
//! - fp8 e4m3 goes e4m3 -> f16 -> f32 and back with satfinite rn, except `recip`, which divides
//!   in double and uses cuda_fp8.hpp's software double -> e4m3 conversion.
use cuda_device::{convert, device, float, kernel, ptx_asm, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    // ---------------------------------------------------------------------------------------
    // Index helpers (cuda_utils.cuh), identical to candle-affine.

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

    /// candle `get_strided_index`: u32 accumulator, u64 intermediate products.
    #[inline(always)]
    pub unsafe fn get_strided_index(idx: u32, num_dims: usize, dims: *const usize, strides: *const usize) -> u32 {
        let mut idx = idx;
        let mut strided_i: u32 = 0;
        let mut d: u32 = 0;
        while (d as usize) < num_dims {
            let dim_idx = (num_dims - 1 - d as usize) as u32 as usize;
            let dim = *dims.add(dim_idx);
            let term = (idx as usize % dim).wrapping_mul(*strides.add(dim_idx));
            strided_i = (strided_i as usize).wrapping_add(term) as u32;
            idx = (idx as usize / dim) as u32;
            d += 1;
        }
        strided_i
    }

    /// The shared loop of every UNARY_OP / UNARY_OP1 kernel.
    #[inline(always)]
    pub unsafe fn unary_loop<T: Copy, F: Fn(T) -> T>(
        numel: usize, num_dims: usize, info: *const usize, inp: *const T, out: *mut T, op: F,
    ) {
        let dims = info;
        let strides = info.wrapping_add(num_dims);
        let step = thread::blockDim_x().wrapping_mul(thread::gridDim_x());
        let mut i: u32 = thread::blockIdx_x().wrapping_mul(thread::blockDim_x()).wrapping_add(thread::threadIdx_x());
        if info.is_null() || is_contiguous(num_dims, dims, strides) {
            while (i as usize) < numel {
                let x = if inp.is_null() { *out.add(i as usize) } else { *inp.add(i as usize) };
                *out.add(i as usize) = op(x);
                i = i.wrapping_add(step);
            }
        } else {
            while (i as usize) < numel {
                let s = get_strided_index(i, num_dims, dims, strides);
                let x = if inp.is_null() { *out.add(i as usize) } else { *inp.add(s as usize) };
                *out.add(i as usize) = op(x);
                i = i.wrapping_add(step);
            }
        }
    }

    // ---------------------------------------------------------------------------------------
    // libdevice: exactly the functions nvcc's math headers resolve to.
    #[device]
    unsafe extern "C" {
        fn __nv_expf(x: f32) -> f32;
        fn __nv_logf(x: f32) -> f32;
        fn __nv_sinf(x: f32) -> f32;
        fn __nv_cosf(x: f32) -> f32;
        fn __nv_tanhf(x: f32) -> f32;
        fn __nv_erff(x: f32) -> f32;
        fn __nv_ceilf(x: f32) -> f32;
        fn __nv_floorf(x: f32) -> f32;
        fn __nv_truncf(x: f32) -> f32;
        fn __nv_roundf(x: f32) -> f32;
        fn __nv_normcdff(x: f32) -> f32;
        fn __nv_fabsf(x: f32) -> f32;
        fn __nv_sqrtf(x: f32) -> f32;
        fn __nv_fmaxf(x: f32, y: f32) -> f32;
        fn __nv_exp(x: f64) -> f64;
        fn __nv_log(x: f64) -> f64;
        fn __nv_sin(x: f64) -> f64;
        fn __nv_cos(x: f64) -> f64;
        fn __nv_tanh(x: f64) -> f64;
        fn __nv_erf(x: f64) -> f64;
        fn __nv_ceil(x: f64) -> f64;
        fn __nv_floor(x: f64) -> f64;
        fn __nv_round(x: f64) -> f64;
        fn __nv_normcdf(x: f64) -> f64;
        fn __nv_fabs(x: f64) -> f64;
        fn __nv_sqrt(x: f64) -> f64;
        fn __nv_fmax(x: f64, y: f64) -> f64;
        fn __nv_pow(x: f64, y: f64) -> f64;
    }

    // ---------------------------------------------------------------------------------------
    // f32 / f64 building blocks.

    #[inline(always)]
    fn cvt_sat_f32(x: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("cvt.sat.f32.f32 %0, %1;", out("=f") r, in("f") x, options(register_only)); }
        r
    }

    /// libdevice `expf` split before its final multiply: returns (e, s) with expf(x) = e * s.
    /// nvcc contracts that final multiply with a following add (1 + expf(-x), expf(x) - 1)
    /// into one fma, so callers that add must use `fma(e, s, c)`.
    #[inline(always)]
    fn expf_parts(x: f32) -> (f32, f32) {
        let t = cvt_sat_f32(float::fma_rn_f32(x, f32::from_bits(0x3BBB989D), 0.5));
        let j = float::fma_rm_f32(t, f32::from_bits(0x437C0000), f32::from_bits(0x4B400001));
        let k = float::add_rn_f32(j, f32::from_bits(0xCB40007F));
        let r = float::fma_rn_f32(x, f32::from_bits(0x3FB8AA3B), -k);
        let r = float::fma_rn_f32(x, f32::from_bits(0x32A57060), r);
        let s = f32::from_bits(j.to_bits() << 23);
        (float::ex2_approx_ftz_f32(r), s)
    }

    // Non-`.rn` f32 ops, exactly as libdevice leaves them in nvcc's PTX: ptxas may contract these
    // (a `.rn` op never is), so where nvcc's PTX has them, so must ours.
    #[inline(always)]
    fn mul_f32(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("mul.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    #[inline(always)]
    fn add_f32(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("add.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    #[inline(always)]
    fn sub_f32(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("sub.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    #[inline(always)]
    fn rcp_approx_ftz(a: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("rcp.approx.ftz.f32 %0, %1;", out("=f") r, in("f") a, options(register_only)); }
        r
    }
    #[inline(always)]
    fn rni_f32(a: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("cvt.rni.f32.f32 %0, %1;", out("=f") r, in("f") a, options(register_only)); }
        r
    }
    #[inline(always)]
    fn rzi_s32_f32(a: f32) -> i32 {
        let r: i32;
        unsafe { ptx_asm!("cvt.rzi.s32.f32 %0, %1;", out("=r") r, in("f") a, options(register_only)); }
        r
    }

    /// libdevice `__nv_powf`, transcribed from nvcc's PTX. The libdevice call itself does not
    /// match: the oxide (LLVM NVPTX) pipeline contracts `t - q` with the multi-use `q = a * rcp`
    /// of the log2 kernel into an fma, nvcc's NVVM does not, and results differ by 1 ulp.
    #[inline(always)]
    fn powf(x: f32, y: f32) -> f32 {
        // exponent-dependent invariants (hoisted out of the loop by nvcc)
        let yh = unsafe { __nv_truncf(mul_f32(y, 0.5)) };
        let y_odd = sub_f32(y, add_f32(yh, yh)).abs(); // 1.0 iff y is an odd integer
        let ay = y.abs();
        let yfl = unsafe { __nv_floorf(y) };

        // log2(|x|) as a double-float (hi, lo)
        let ax = x.abs();
        let sub = ax < f32::from_bits(0x00800000);
        let m0 = if sub { mul_f32(ax, f32::from_bits(0x4B800000)) } else { ax };
        let e0: f32 = if sub { -24.0 } else { 0.0 };
        let eb = (m0.to_bits() as i32).wrapping_add(-1060439283) & (0xFF800000u32 as i32);
        let m = f32::from_bits((m0.to_bits() as i32).wrapping_sub(eb) as u32);
        let e = float::fma_rn_f32(eb as f32, f32::from_bits(0x34000000), e0);
        let a = add_f32(m, -1.0);
        let rcp = rcp_approx_ftz(add_f32(m, 1.0));
        let q = mul_f32(add_f32(a, a), rcp);
        let q2 = mul_f32(q, q);
        let t = sub_f32(a, q);
        let t = add_f32(t, t);
        let t = float::fma_rn_f32(-q, a, t);
        let ql = float::mul_rn_f32(rcp, t);
        let p = float::fma_rn_f32(q2, f32::from_bits(0x3A2C32E4), f32::from_bits(0x3B52E7DB));
        let p = float::fma_rn_f32(p, q2, f32::from_bits(0x3C93BB73));
        let p = float::fma_rn_f32(p, q2, f32::from_bits(0x3DF6384F));
        let p = float::mul_rn_f32(p, q2);
        let hi = float::fma_rn_f32(q, f32::from_bits(0x3FB8AA3B), e);
        let lo = sub_f32(e, hi);
        let lo = float::fma_rn_f32(q, f32::from_bits(0x3FB8AA3B), lo);
        let lo = float::fma_rn_f32(ql, f32::from_bits(0x3FB8AA3B), lo);
        let lo = float::fma_rn_f32(q, f32::from_bits(0x32A55E34), lo);
        let lo = float::fma_rn_f32(mul_f32(p, 3.0), ql, lo);
        let lo = float::fma_rn_f32(p, q, lo);
        let s = float::add_rn_f32(hi, lo);
        let l_lo = float::add_rn_f32(lo, -float::add_rn_f32(s, -hi));
        // y * log2(x), then exp2
        let r = float::mul_rn_f32(s, y);
        let r_lo = float::fma_rn_f32(s, y, -r);
        let r_lo = float::fma_rn_f32(l_lo, y, r_lo);
        let ri = rni_f32(r);
        let f = add_f32(sub_f32(r, ri), r_lo);
        let z = float::fma_rn_f32(f, f32::from_bits(0x391FCB8E), f32::from_bits(0x3AAF85ED));
        let z = float::fma_rn_f32(z, f, f32::from_bits(0x3C1D9856));
        let z = float::fma_rn_f32(z, f, f32::from_bits(0x3D6357BB));
        let z = float::fma_rn_f32(z, f, f32::from_bits(0x3E75FDEC));
        let z = float::fma_rn_f32(z, f, f32::from_bits(0x3F317218));
        let z = float::fma_rn_f32(z, f, 1.0);
        let n = rzi_s32_f32(ri);
        let bias: i32 = if ri > 0.0 { 0 } else { 0x83000000u32 as i32 };
        let z = mul_f32(z, f32::from_bits(bias.wrapping_add(0x7F000000) as u32));
        let z = mul_f32(z, f32::from_bits((n << 23).wrapping_sub(bias) as u32));
        let res = if r.abs() > 152.0 { if r < 0.0 { 0.0 } else { f32::INFINITY } } else { z };

        // special cases
        if y == 0.0 || x == 1.0 {
            return 1.0;
        }
        if ax.is_nan() || ay.is_nan() {
            return float::add_rn_f32(x, y);
        }
        if x == 0.0 || ax == f32::INFINITY {
            let v = add_f32(x, x).to_bits();
            let v = if y < 0.0 { v ^ 0x7F800000 } else { v };
            return f32::from_bits(if y_odd != 1.0 { v & 0x7FFFFFFF } else { v });
        }
        if ay == f32::INFINITY && x == -1.0 {
            return 1.0;
        }
        if !(x < 0.0) {
            return res;
        }
        let v = if y_odd != 1.0 { res } else { -res };
        if y != yfl {
            // A NaN constant does not survive the oxide pipeline bit-exact (0x7FFFFFFF is emitted as
            // 0x7FC00000), so materialise libdevice's NaN pattern in PTX.
            let nan: f32;
            unsafe { ptx_asm!("mov.b32 %0, 0x7FFFFFFF;", out("=f") nan, options(register_only)); }
            nan
        } else {
            v
        }
    }

    #[inline(always)]
    fn sign_f32(x: f32) -> f32 {
        let a: f32 = if x > 0.0 { 1.0 } else { 0.0 };
        let b: f32 = if x < 0.0 { 1.0 } else { 0.0 };
        float::add_rn_f32(a, -b)
    }
    #[inline(always)]
    fn sign_f64(x: f64) -> f64 {
        let a: f64 = if x > 0.0 { 1.0 } else { 0.0 };
        let b: f64 = if x < 0.0 { 1.0 } else { 0.0 };
        float::add_rn_f64(a, -b)
    }

    #[inline(always)]
    fn gelu_f32(x: f32) -> f32 {
        let x_sq = float::mul_rn_f32(x, x);
        let x_cube = float::mul_rn_f32(x, x_sq);
        let alpha = float::fma_rn_f32(x_cube, f32::from_bits(0x3D372713), x);
        let hx = float::mul_rn_f32(x, 0.5);
        let c = float::mul_rn_f32(alpha, f32::from_bits(0x3F4C422A));
        float::mul_rn_f32(hx, float::add_rn_f32(unsafe { __nv_tanhf(c) }, 1.0))
    }
    #[inline(always)]
    fn gelu_f64(x: f64) -> f64 {
        let x_sq = float::mul_rn_f64(x, x);
        let x_cube = float::mul_rn_f64(x, x_sq);
        let alpha = float::fma_rn_f64(x_cube, f64::from_bits(0x3FA6E4E26D4801F7), x);
        let c = float::mul_rn_f64(alpha, f64::from_bits(0x3FE9884533D43651));
        let t = float::add_rn_f64(unsafe { __nv_tanh(c) }, 1.0);
        float::mul_rn_f64(float::mul_rn_f64(x, 0.5), t)
    }
    #[inline(always)]
    fn gelu_erf_f32(x: f32) -> f32 { float::mul_rn_f32(x, unsafe { __nv_normcdff(x) }) }
    #[inline(always)]
    fn gelu_erf_f64(x: f64) -> f64 { float::mul_rn_f64(x, unsafe { __nv_normcdf(x) }) }

    #[inline(always)]
    fn elu_f32(x: f32, alpha: f32) -> f32 {
        if x > 0.0 {
            return x;
        }
        let (e, s) = expf_parts(x);
        float::mul_rn_f32(alpha, float::fma_rn_f32(e, s, -1.0))
    }
    #[inline(always)]
    fn elu_f64(x: f64, alpha: f64) -> f64 {
        if x > 0.0 {
            return x;
        }
        float::mul_rn_f64(alpha, float::add_rn_f64(unsafe { __nv_exp(x) }, -1.0))
    }
    #[inline(always)]
    fn silu_f32(x: f32) -> f32 {
        let (e, s) = expf_parts(-x);
        float::div_rn_f32(x, float::fma_rn_f32(e, s, 1.0))
    }
    #[inline(always)]
    fn silu_f64(x: f64) -> f64 {
        float::div_rn_f64(x, float::add_rn_f64(unsafe { __nv_exp(-x) }, 1.0))
    }
    #[inline(always)]
    fn sigmoid_f32(x: f32) -> f32 {
        let (e, s) = expf_parts(-x);
        float::rcp_rn_f32(float::fma_rn_f32(e, s, 1.0))
    }
    #[inline(always)]
    fn sigmoid_f64(x: f64) -> f64 {
        float::rcp_rn_f64(float::add_rn_f64(unsafe { __nv_exp(-x) }, 1.0))
    }

    // ---------------------------------------------------------------------------------------
    // f16 (cuda_fp16.hpp): the header's inline PTX, verbatim, so ptxas sees the same code.

    #[inline(always)]
    fn h2f(h: u16) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("{  cvt.f32.f16 %0, %1;}", out("=f") r, in("h") h, options(register_only)); }
        r
    }
    #[inline(always)]
    fn f2h(f: f32) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("{  cvt.rn.f16.f32 %0, %1;}", out("=h") r, in("f") f, options(register_only)); }
        r
    }
    #[inline(always)]
    fn d2h(f: f64) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("{  cvt.rn.f16.f64 %0, %1;}", out("=h") r, in("d") f, options(register_only)); }
        r
    }
    #[inline(always)]
    fn i2h(i: i32) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("cvt.rn.f16.s32 %0, %1;", out("=h") r, in("r") i, options(register_only)); }
        r
    }
    #[inline(always)]
    fn hneg(a: u16) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("{neg.f16 %0,%1;\n}", out("=h") r, in("h") a, options(register_only)); }
        r
    }
    #[inline(always)]
    fn habs(a: u16) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("{abs.f16 %0,%1;\n}", out("=h") r, in("h") a, options(register_only)); }
        r
    }
    #[inline(always)]
    fn hmul(a: u16, b: u16) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("{mul.f16 %0,%1,%2;\n}", out("=h") r, in("h") a, in("h") b, options(register_only)); }
        r
    }
    #[inline(always)]
    fn hadd(a: u16, b: u16) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("{add.f16 %0,%1,%2;\n}", out("=h") r, in("h") a, in("h") b, options(register_only)); }
        r
    }
    #[inline(always)]
    fn hsub(a: u16, b: u16) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("{sub.f16 %0,%1,%2;\n}", out("=h") r, in("h") a, in("h") b, options(register_only)); }
        r
    }
    #[inline(always)]
    fn hmax_nan(a: u16, b: u16) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("{max.NaN.f16 %0,%1,%2;\n}", out("=h") r, in("h") a, in("h") b, options(register_only)); }
        r
    }
    #[inline(always)]
    fn hgt(a: u16, b: u16) -> bool {
        let r: u16;
        unsafe {
            ptx_asm!("{ .reg .pred __$temp3;\n  setp.gt.f16  __$temp3, %1, %2;\n  selp.u16 %0, 1, 0, __$temp3;}",
                     out("=h") r, in("h") a, in("h") b, options(register_only));
        }
        r != 0
    }
    #[inline(always)]
    fn hlt(a: u16, b: u16) -> bool {
        let r: u16;
        unsafe {
            ptx_asm!("{ .reg .pred __$temp3;\n  setp.lt.f16  __$temp3, %1, %2;\n  selp.u16 %0, 1, 0, __$temp3;}",
                     out("=h") r, in("h") a, in("h") b, options(register_only));
        }
        r != 0
    }
    /// __hdiv (device path).
    #[inline(always)]
    fn hdiv(a: u16, b: u16) -> u16 {
        let fa = h2f(a);
        let fb = h2f(b);
        let rcp: f32;
        unsafe { ptx_asm!("{rcp.approx.ftz.f32 %0, %1;\n}", out("=f") rcp, in("f") fb, options(register_only)); }
        let mut fv = float::mul_rn_f32(rcp, fa);
        let mut v = f2h(fv);
        let abs = habs(v);
        if hlt(abs, 0x008F) && hlt(f2h(0.0), abs) {
            let err = float::fma_rn_f32(-fb, fv, fa);
            fv = float::fma_rn_f32(rcp, err, fv);
            v = f2h(fv);
        }
        v
    }
    #[inline(always)]
    fn hexp(a: u16) -> u16 {
        let r: u16;
        unsafe {
            ptx_asm!(
                "{.reg.b32         f, C, nZ;\n .reg.b16         h,r;\n  mov.b16         h,%1;\n  cvt.f32.f16     f,h;\n  mov.b32         C, 0x3fb8aa3bU;\n  mov.b32         nZ, 0x80000000U;\n  fma.rn.f32      f,f,C,nZ;\n  ex2.approx.ftz.f32  f,f;\n  cvt.rn.f16.f32      r,f;\n{.reg.b16 spc, ulp, p;\n  mov.b16 spc,0X1F79U;\n  mov.b16 ulp,0x9400U;\n  set.eq.f16.f16 p,h, spc;\n  fma.rn.f16 r,p,ulp,r;\n}\n{.reg.b16 spc, ulp, p;\n  mov.b16 spc,0X25CFU;\n  mov.b16 ulp,0x9400U;\n  set.eq.f16.f16 p,h, spc;\n  fma.rn.f16 r,p,ulp,r;\n}\n{.reg.b16 spc, ulp, p;\n  mov.b16 spc,0XC13BU;\n  mov.b16 ulp,0x0400U;\n  set.eq.f16.f16 p,h, spc;\n  fma.rn.f16 r,p,ulp,r;\n}\n{.reg.b16 spc, ulp, p;\n  mov.b16 spc,0XC1EFU;\n  mov.b16 ulp,0x0200U;\n  set.eq.f16.f16 p,h, spc;\n  fma.rn.f16 r,p,ulp,r;\n}\n  mov.b16         %0,r;\n}",
                out("=h") r, in("h") a, options(register_only));
        }
        r
    }
    #[inline(always)]
    fn hlog(a: u16) -> u16 {
        let r: u16;
        unsafe {
            ptx_asm!(
                "{.reg.b32         f, C;\n .reg.b16         r,h;\n  mov.b16         h,%1;\n  cvt.f32.f16     f,h;\n  lg2.approx.ftz.f32  f,f;\n  mov.b32         C, 0x3f317218U;\n  mul.f32         f,f,C;\n  cvt.rn.f16.f32      r,f;\n{.reg.b16 spc, ulp, p;\n  mov.b16 spc,0X160DU;\n  mov.b16 ulp,0x9C00U;\n  set.eq.f16.f16 p,h, spc;\n  fma.rn.f16 r,p,ulp,r;\n}\n{.reg.b16 spc, ulp, p;\n  mov.b16 spc,0X3BFEU;\n  mov.b16 ulp,0x8010U;\n  set.eq.f16.f16 p,h, spc;\n  fma.rn.f16 r,p,ulp,r;\n}\n{.reg.b16 spc, ulp, p;\n  mov.b16 spc,0X3C0BU;\n  mov.b16 ulp,0x8080U;\n  set.eq.f16.f16 p,h, spc;\n  fma.rn.f16 r,p,ulp,r;\n}\n{.reg.b16 spc, ulp, p;\n  mov.b16 spc,0X6051U;\n  mov.b16 ulp,0x1C00U;\n  set.eq.f16.f16 p,h, spc;\n  fma.rn.f16 r,p,ulp,r;\n}\n  mov.b16         %0,r;\n}",
                out("=h") r, in("h") a, options(register_only));
        }
        r
    }
    #[inline(always)]
    fn hsqrt(a: u16) -> u16 {
        let r: u16;
        unsafe {
            ptx_asm!(
                "{.reg.b32         f;\n .reg.b16         r;\n  mov.b16         r,%1;\n  cvt.f32.f16     f,r;\n  sqrt.approx.ftz.f32   f,f;\n  cvt.rn.f16.f32      r,f;\n  mov.b16         %0,r;\n}",
                out("=h") r, in("h") a, options(register_only));
        }
        r
    }
    /// __internal_trig_reduction_kernel + __internal_sin_cos_kernel (cuda_fp16.hpp).
    #[inline(always)]
    fn simpl_sincos(a: f32, cos: bool) -> f32 {
        let ar = float::fma_rn_f32(a, 0.636619772, 12582912.0);
        let q = ar.to_bits();
        let j = float::add_rn_f32(ar, -12582912.0);
        let t = float::fma_rn_f32(j, -1.5707962512969971e+000, a);
        let x = float::fma_rn_f32(j, -7.5497894158615964e-008, t);
        let i = if cos { (q & 3).wrapping_add(1) } else { q };
        let x2 = float::mul_rn_f32(x, x);
        let (a8, a6, a4, a2, a1, a0) = if (i & 1) != 0 {
            (2.44331571e-5f32, -1.38873163e-3f32, 4.16666457e-2f32, -5.00000000e-1f32, x2, 1.0f32)
        } else {
            (-1.95152959e-4f32, 8.33216087e-3f32, -1.66666546e-1f32, 0.0f32, x, x)
        };
        let mut z = float::fma_rn_f32(a8, x2, a6);
        z = float::fma_rn_f32(z, x2, a4);
        z = float::fma_rn_f32(z, x2, a2);
        z = float::fma_rn_f32(z, a1, a0);
        if (i & 2) != 0 {
            z = -z;
        }
        z
    }
    #[inline(always)]
    fn hsin(a: u16) -> u16 {
        let mut r = f2h(simpl_sincos(h2f(a), false));
        unsafe {
            ptx_asm!(
                "{\n\t  .reg.b16 i,r,t;     \n\t  mov.b16 r, %0;      \n\t  mov.b16 i, %1;      \n\t  and.b16 t, r, 0x8000U; \n\t  abs.f16 r, r;   \n\t  abs.f16 i, i;   \n\t{.reg.b16 spc, ulp, p;\n  mov.b16 spc,0X32B3U;\n  mov.b16 ulp,0x0800U;\n  set.eq.f16.f16 p,i, spc;\n  fma.rn.f16 r,p,ulp,r;\n}\n{.reg.b16 spc, ulp, p;\n  mov.b16 spc,0X5CB0U;\n  mov.b16 ulp,0x9000U;\n  set.eq.f16.f16 p,i, spc;\n  fma.rn.f16 r,p,ulp,r;\n}\n  or.b16  r,r,t;      \n\t  mov.b16 %0, r;      \n}\n",
                inout("+h") r, in("h") a, options(register_only));
        }
        r
    }
    #[inline(always)]
    fn hcos(a: u16) -> u16 {
        let mut r = f2h(simpl_sincos(h2f(a), true));
        unsafe {
            ptx_asm!(
                "{\n\t  .reg.b16 i,r;        \n\t  mov.b16 r, %0;       \n\t  mov.b16 i, %1;       \n\t  abs.f16 i, i;        \n\t{.reg.b16 spc, ulp, p;\n  mov.b16 spc,0X2B7CU;\n  mov.b16 ulp,0x1000U;\n  set.eq.f16.f16 p,i, spc;\n  fma.rn.f16 r,p,ulp,r;\n}\n  mov.b16 %0, r;       \n}\n",
                inout("+h") r, in("h") a, options(register_only));
        }
        r
    }

    // ---------------------------------------------------------------------------------------
    // bf16 (cuda_bf16.hpp).

    #[inline(always)]
    fn b2f(h: u16) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("{ cvt.f32.bf16 %0, %1;}", out("=f") r, in("h") h, options(register_only)); }
        r
    }
    #[inline(always)]
    fn f2b(f: f32) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("{  cvt.rn.bf16.f32 %0, %1;}", out("=h") r, in("f") f, options(register_only)); }
        r
    }
    #[inline(always)]
    fn d2b(f: f64) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("{  cvt.rn.bf16.f64 %0, %1;}", out("=h") r, in("d") f, options(register_only)); }
        r
    }
    #[inline(always)]
    fn i2b(i: i32) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("cvt.rn.bf16.s32 %0, %1;", out("=h") r, in("r") i, options(register_only)); }
        r
    }
    #[inline(always)]
    fn bneg(a: u16) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("{neg.bf16 %0,%1;\n}", out("=h") r, in("h") a, options(register_only)); }
        r
    }
    #[inline(always)]
    fn babs(a: u16) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("{abs.bf16 %0,%1;\n}", out("=h") r, in("h") a, options(register_only)); }
        r
    }
    #[inline(always)]
    fn bmul(a: u16, b: u16) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("{ mul.bf16 %0,%1,%2; }\n", out("=h") r, in("h") a, in("h") b, options(register_only)); }
        r
    }
    #[inline(always)]
    fn badd(a: u16, b: u16) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("{ add.bf16 %0,%1,%2; }\n", out("=h") r, in("h") a, in("h") b, options(register_only)); }
        r
    }
    #[inline(always)]
    fn bsub(a: u16, b: u16) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("{ sub.bf16 %0,%1,%2; }\n", out("=h") r, in("h") a, in("h") b, options(register_only)); }
        r
    }
    #[inline(always)]
    fn bmax_nan(a: u16, b: u16) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("{ max.NaN.bf16 %0,%1,%2;\n}", out("=h") r, in("h") a, in("h") b, options(register_only)); }
        r
    }
    #[inline(always)]
    fn bgt(a: u16, b: u16) -> bool {
        let r: u16;
        unsafe {
            ptx_asm!("{ .reg .pred __$temp3;\n  setp.gt.bf16  __$temp3, %1, %2;\n  selp.u16 %0, 1, 0, __$temp3;}",
                     out("=h") r, in("h") a, in("h") b, options(register_only));
        }
        r != 0
    }
    #[inline(always)]
    fn blt(a: u16, b: u16) -> bool {
        let r: u16;
        unsafe {
            ptx_asm!("{ .reg .pred __$temp3;\n  setp.lt.bf16  __$temp3, %1, %2;\n  selp.u16 %0, 1, 0, __$temp3;}",
                     out("=h") r, in("h") a, in("h") b, options(register_only));
        }
        r != 0
    }
    /// __internal_device_hdiv (bf16).
    #[inline(always)]
    fn bdiv(a: u16, b: u16) -> u16 {
        let a_f = b2f(a);
        let mut b_f = b2f(b);
        let b_big = b_f.abs() >= f32::from_bits(0x7E800000);
        if b_big {
            b_f = float::mul_rn_f32(b_f, 0.25);
        }
        let mut ans: f32;
        unsafe { ptx_asm!("{ div.approx.f32 %0, %1, %2; }", out("=f") ans, in("f") a_f, in("f") b_f, options(register_only)); }
        if b_big {
            ans = float::mul_rn_f32(ans, 0.25);
        }
        f2b(ans)
    }
    #[inline(always)]
    fn bexp(a: u16) -> u16 {
        let mut fa = float::mul_rn_f32(b2f(a), f32::from_bits(0x3FB8AA3C));
        unsafe { ptx_asm!("{ ex2.approx.f32 %0, %0; }", inout("+f") fa, options(register_only)); }
        f2b(fa)
    }
    #[inline(always)]
    fn blog(a: u16) -> u16 {
        let mut fa = b2f(a);
        unsafe { ptx_asm!("{ lg2.approx.f32 %0, %0; }", inout("+f") fa, options(register_only)); }
        f2b(float::mul_rn_f32(fa, f32::from_bits(0x3f317218)))
    }
    #[inline(always)]
    fn bsqrt(a: u16) -> u16 {
        let mut fa = b2f(a);
        unsafe { ptx_asm!("{ sqrt.approx.f32 %0, %0; }", inout("+f") fa, options(register_only)); }
        f2b(fa)
    }

    // ---------------------------------------------------------------------------------------
    // Generic half-type templates (unary.cu), parameterised by the type's primitive ops.

    pub trait HOps {
        fn mul(a: u16, b: u16) -> u16;
        fn add(a: u16, b: u16) -> u16;
        fn sub(a: u16, b: u16) -> u16;
        fn neg(a: u16) -> u16;
        fn div(a: u16, b: u16) -> u16;
        fn exp(a: u16) -> u16;
        fn gt(a: u16, b: u16) -> bool;
        fn lt(a: u16, b: u16) -> bool;
        fn to_f(a: u16) -> f32;
        fn from_f(a: f32) -> u16;
        fn from_d(a: f64) -> u16;
        fn from_i(a: i32) -> u16;
    }
    pub struct F16;
    pub struct BF16;
    impl HOps for F16 {
        #[inline(always)] fn mul(a: u16, b: u16) -> u16 { hmul(a, b) }
        #[inline(always)] fn add(a: u16, b: u16) -> u16 { hadd(a, b) }
        #[inline(always)] fn sub(a: u16, b: u16) -> u16 { hsub(a, b) }
        #[inline(always)] fn neg(a: u16) -> u16 { hneg(a) }
        #[inline(always)] fn div(a: u16, b: u16) -> u16 { hdiv(a, b) }
        #[inline(always)] fn exp(a: u16) -> u16 { hexp(a) }
        #[inline(always)] fn gt(a: u16, b: u16) -> bool { hgt(a, b) }
        #[inline(always)] fn lt(a: u16, b: u16) -> bool { hlt(a, b) }
        #[inline(always)] fn to_f(a: u16) -> f32 { h2f(a) }
        #[inline(always)] fn from_f(a: f32) -> u16 { f2h(a) }
        #[inline(always)] fn from_d(a: f64) -> u16 { d2h(a) }
        #[inline(always)] fn from_i(a: i32) -> u16 { i2h(a) }
    }
    impl HOps for BF16 {
        #[inline(always)] fn mul(a: u16, b: u16) -> u16 { bmul(a, b) }
        #[inline(always)] fn add(a: u16, b: u16) -> u16 { badd(a, b) }
        #[inline(always)] fn sub(a: u16, b: u16) -> u16 { bsub(a, b) }
        #[inline(always)] fn neg(a: u16) -> u16 { bneg(a) }
        #[inline(always)] fn div(a: u16, b: u16) -> u16 { bdiv(a, b) }
        #[inline(always)] fn exp(a: u16) -> u16 { bexp(a) }
        #[inline(always)] fn gt(a: u16, b: u16) -> bool { bgt(a, b) }
        #[inline(always)] fn lt(a: u16, b: u16) -> bool { blt(a, b) }
        #[inline(always)] fn to_f(a: u16) -> f32 { b2f(a) }
        #[inline(always)] fn from_f(a: f32) -> u16 { f2b(a) }
        #[inline(always)] fn from_d(a: f64) -> u16 { d2b(a) }
        #[inline(always)] fn from_i(a: i32) -> u16 { i2b(a) }
    }

    #[inline(always)]
    fn t_gelu<O: HOps>(x: u16) -> u16 {
        let c0447 = O::from_d(f64::from_bits(4586604931670606327));
        let c05 = O::from_d(0.5);
        let c1 = O::from_d(1.0);
        let csq = O::from_d(f64::from_bits(4605361924766709329));
        let x_sq = O::mul(x, x);
        let x_cube = O::mul(x_sq, x);
        let alpha = O::add(x, O::mul(c0447, x_cube));
        let hx = O::mul(c05, x);
        let a = O::mul(csq, alpha);
        let th = O::from_f(unsafe { __nv_tanhf(O::to_f(a)) });
        O::mul(hx, O::add(c1, th))
    }
    #[inline(always)]
    fn t_gelu_erf<O: HOps>(x: u16) -> u16 {
        O::mul(x, O::from_f(unsafe { __nv_normcdff(O::to_f(x)) }))
    }
    #[inline(always)]
    fn t_elu<O: HOps>(x: u16, alpha: u16) -> u16 {
        if O::gt(x, O::from_i(0)) {
            return x;
        }
        O::mul(alpha, O::sub(O::exp(x), O::from_i(1)))
    }
    #[inline(always)]
    fn t_silu<O: HOps>(x: u16) -> u16 {
        O::div(x, O::add(O::from_i(1), O::exp(O::neg(x))))
    }
    #[inline(always)]
    fn t_sigmoid<O: HOps>(x: u16) -> u16 {
        let one = O::from_d(1.0);
        O::div(one, O::add(O::from_i(1), O::exp(O::neg(x))))
    }
    #[inline(always)]
    fn t_sign<O: HOps>(x: u16) -> u16 {
        let z = O::from_i(0);
        let a = O::from_i(O::gt(x, z) as i32);
        let b = O::from_i(O::lt(x, z) as i32);
        O::sub(a, b)
    }
    #[inline(always)]
    fn t_f32<O: HOps, F: Fn(f32) -> f32>(x: u16, f: F) -> u16 {
        O::from_f(f(O::to_f(x)))
    }

    // ---------------------------------------------------------------------------------------
    // fp8 e4m3 (cuda_fp8.hpp): e4m3 -> f16 -> f32, and back with satfinite rn.

    #[inline(always)]
    fn e4m3_to_f32(b: u8) -> f32 {
        h2f(convert::cvt_rn_f16x2_e4m3x2(b as u16) as u16)
    }
    #[inline(always)]
    fn f32_to_e4m3(v: f32) -> u8 {
        convert::cvt_rn_satfinite_e4m3x2_f32(v, 0.0) as u8
    }
    /// __nv_cvt_double_to_fp8(x, __NV_SATFINITE, __NV_E4M3).
    #[inline(always)]
    fn f64_to_e4m3(x: f64) -> u8 {
        let xbits = x.to_bits();
        const HALF_ULP: u64 = 1u64 << (53 - 4 - 1);
        let sign = (((xbits >> 63) << 7) as u8) as u8;
        let exp = ((((xbits >> 52) as u16) & 0x7FF) as u32).wrapping_sub(1023).wrapping_add(7) as u8;
        let mut mantissa = ((xbits >> (53 - 4)) as u8) & 0x7;
        let absx = xbits & 0x7FFF_FFFF_FFFF_FFFF;
        let mut res: u8;
        if absx <= 0x3F50_0000_0000_0000 {
            res = 0;
        } else if absx > 0x7FF0_0000_0000_0000 {
            res = 0x7F;
        } else if absx > 0x407D_0000_0000_0000 {
            res = 0x7E;
        } else if absx >= 0x3F90_0000_0000_0000 {
            res = (((exp as u32) << 3) | mantissa as u32) as u8;
            let round = xbits & ((HALF_ULP << 1) - 1);
            if round > HALF_ULP || (round == HALF_ULP && (mantissa & 1) != 0) {
                res = res.wrapping_add(1);
            }
        } else {
            let shift = (1u32.wrapping_sub(exp as u32)) as u8;
            mantissa |= 1 << 3;
            res = ((mantissa as u32) >> (shift as u32)) as u8;
            let round = (xbits | (1u64 << 52)) & ((HALF_ULP << (shift as u64 + 1)) - 1);
            if round > (HALF_ULP << shift as u64) || (round == (HALF_ULP << shift as u64) && (res & 1) != 0) {
                res = res.wrapping_add(1);
            }
        }
        res | sign
    }
    #[kernel]
    pub unsafe fn ucopy_bf16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| x);
    }

    #[kernel]
    pub unsafe fn uneg_bf16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| bneg(x));
    }

    #[kernel]
    pub unsafe fn urecip_bf16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| BF16::div(BF16::from_d(1.0), x));
    }

    #[kernel]
    pub unsafe fn uexp_bf16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| bexp(x));
    }

    #[kernel]
    pub unsafe fn ulog_bf16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| blog(x));
    }

    #[kernel]
    pub unsafe fn usin_bf16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| t_f32::<BF16, _>(x, |f| unsafe { __nv_sinf(f) }));
    }

    #[kernel]
    pub unsafe fn ucos_bf16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| t_f32::<BF16, _>(x, |f| unsafe { __nv_cosf(f) }));
    }

    #[kernel]
    pub unsafe fn utanh_bf16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| t_f32::<BF16, _>(x, |f| unsafe { __nv_tanhf(f) }));
    }

    #[kernel]
    pub unsafe fn uerf_bf16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| t_f32::<BF16, _>(x, |f| unsafe { __nv_erff(f) }));
    }

    #[kernel]
    pub unsafe fn uceil_bf16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| t_f32::<BF16, _>(x, |f| unsafe { __nv_ceilf(f) }));
    }

    #[kernel]
    pub unsafe fn ufloor_bf16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| t_f32::<BF16, _>(x, |f| unsafe { __nv_floorf(f) }));
    }

    #[kernel]
    pub unsafe fn uround_bf16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| t_f32::<BF16, _>(x, |f| unsafe { __nv_roundf(f) }));
    }

    #[kernel]
    pub unsafe fn unormcdf_bf16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| t_f32::<BF16, _>(x, |f| unsafe { __nv_normcdff(f) }));
    }

    #[kernel]
    pub unsafe fn uabs_bf16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| babs(x));
    }

    #[kernel]
    pub unsafe fn usqr_bf16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| BF16::mul(x, x));
    }

    #[kernel]
    pub unsafe fn usqrt_bf16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| bsqrt(x));
    }

    #[kernel]
    pub unsafe fn ugelu_bf16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| t_gelu::<BF16>(x));
    }

    #[kernel]
    pub unsafe fn ugelu_erf_bf16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| t_gelu_erf::<BF16>(x));
    }

    #[kernel]
    pub unsafe fn urelu_bf16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| bmax_nan(x, BF16::from_d(0.0)));
    }

    #[kernel]
    pub unsafe fn uelu_bf16(numel: usize, num_dims: usize, info: *const usize, param: u16, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| t_elu::<BF16>(x, param));
    }

    #[kernel]
    pub unsafe fn usilu_bf16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| t_silu::<BF16>(x));
    }

    #[kernel]
    pub unsafe fn upowf_bf16(numel: usize, num_dims: usize, info: *const usize, param: u16, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| BF16::from_f(powf(BF16::to_f(x), BF16::to_f(param))));
    }

    #[kernel]
    pub unsafe fn usign_bf16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| t_sign::<BF16>(x));
    }

    #[kernel]
    pub unsafe fn usigmoid_bf16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| t_sigmoid::<BF16>(x));
    }

    #[kernel]
    pub unsafe fn ucopy_f8_e4m3(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u8) {
        unary_loop(numel, num_dims, info, inp, out, |x: u8| x);
    }

    #[kernel]
    pub unsafe fn uneg_fp8_e4m3(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u8) {
        unary_loop(numel, num_dims, info, inp, out, |x: u8| f32_to_e4m3(-e4m3_to_f32(x)));
    }

    #[kernel]
    pub unsafe fn urecip_fp8_e4m3(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u8) {
        unary_loop(numel, num_dims, info, inp, out, |x: u8| f64_to_e4m3(float::rcp_rn_f64(e4m3_to_f32(x) as f64)));
    }

    #[kernel]
    pub unsafe fn uexp_fp8_e4m3(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u8) {
        unary_loop(numel, num_dims, info, inp, out, |x: u8| f32_to_e4m3(unsafe { __nv_expf(e4m3_to_f32(x)) }));
    }

    #[kernel]
    pub unsafe fn ulog_fp8_e4m3(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u8) {
        unary_loop(numel, num_dims, info, inp, out, |x: u8| f32_to_e4m3(unsafe { __nv_logf(e4m3_to_f32(x)) }));
    }

    #[kernel]
    pub unsafe fn usin_fp8_e4m3(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u8) {
        unary_loop(numel, num_dims, info, inp, out, |x: u8| f32_to_e4m3(unsafe { __nv_sinf(e4m3_to_f32(x)) }));
    }

    #[kernel]
    pub unsafe fn ucos_fp8_e4m3(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u8) {
        unary_loop(numel, num_dims, info, inp, out, |x: u8| f32_to_e4m3(unsafe { __nv_cosf(e4m3_to_f32(x)) }));
    }

    #[kernel]
    pub unsafe fn utanh_fp8_e4m3(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u8) {
        unary_loop(numel, num_dims, info, inp, out, |x: u8| f32_to_e4m3(unsafe { __nv_tanhf(e4m3_to_f32(x)) }));
    }

    #[kernel]
    pub unsafe fn uerf_fp8_e4m3(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u8) {
        unary_loop(numel, num_dims, info, inp, out, |x: u8| f32_to_e4m3(unsafe { __nv_erff(e4m3_to_f32(x)) }));
    }

    #[kernel]
    pub unsafe fn uceil_fp8_e4m3(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u8) {
        unary_loop(numel, num_dims, info, inp, out, |x: u8| f32_to_e4m3(unsafe { __nv_ceilf(e4m3_to_f32(x)) }));
    }

    #[kernel]
    pub unsafe fn ufloor_fp8_e4m3(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u8) {
        unary_loop(numel, num_dims, info, inp, out, |x: u8| f32_to_e4m3(unsafe { __nv_floorf(e4m3_to_f32(x)) }));
    }

    #[kernel]
    pub unsafe fn uround_fp8_e4m3(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u8) {
        unary_loop(numel, num_dims, info, inp, out, |x: u8| f32_to_e4m3(unsafe { __nv_roundf(e4m3_to_f32(x)) }));
    }

    #[kernel]
    pub unsafe fn unormcdf_fp8_e4m3(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u8) {
        unary_loop(numel, num_dims, info, inp, out, |x: u8| f32_to_e4m3(unsafe { __nv_normcdff(e4m3_to_f32(x)) }));
    }

    #[kernel]
    pub unsafe fn uabs_fp8_e4m3(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u8) {
        unary_loop(numel, num_dims, info, inp, out, |x: u8| f32_to_e4m3(unsafe { __nv_fabsf(e4m3_to_f32(x)) }));
    }

    #[kernel]
    pub unsafe fn usqr_fp8_e4m3(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u8) {
        unary_loop(numel, num_dims, info, inp, out, |x: u8| { let f = e4m3_to_f32(x); f32_to_e4m3(float::mul_rn_f32(f, f)) });
    }

    #[kernel]
    pub unsafe fn usqrt_fp8_e4m3(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u8) {
        unary_loop(numel, num_dims, info, inp, out, |x: u8| f32_to_e4m3(unsafe { __nv_sqrtf(e4m3_to_f32(x)) }));
    }

    #[kernel]
    pub unsafe fn ugelu_fp8_e4m3(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u8) {
        unary_loop(numel, num_dims, info, inp, out, |x: u8| f32_to_e4m3(gelu_f32(e4m3_to_f32(x))));
    }

    #[kernel]
    pub unsafe fn ugelu_erf_fp8_e4m3(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u8) {
        unary_loop(numel, num_dims, info, inp, out, |x: u8| f32_to_e4m3(gelu_erf_f32(e4m3_to_f32(x))));
    }

    #[kernel]
    pub unsafe fn urelu_fp8_e4m3(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u8) {
        unary_loop(numel, num_dims, info, inp, out, |x: u8| f32_to_e4m3(unsafe { __nv_fmaxf(e4m3_to_f32(x), 0.0) }));
    }

    #[kernel]
    pub unsafe fn uelu_fp8_e4m3(numel: usize, num_dims: usize, info: *const usize, param: u8, inp: *const u8, out: *mut u8) {
        unary_loop(numel, num_dims, info, inp, out, |x: u8| f32_to_e4m3(elu_f32(e4m3_to_f32(x), e4m3_to_f32(param))));
    }

    #[kernel]
    pub unsafe fn usilu_fp8_e4m3(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u8) {
        unary_loop(numel, num_dims, info, inp, out, |x: u8| f32_to_e4m3(silu_f32(e4m3_to_f32(x))));
    }

    #[kernel]
    pub unsafe fn upowf_fp8_e4m3(numel: usize, num_dims: usize, info: *const usize, param: u8, inp: *const u8, out: *mut u8) {
        unary_loop(numel, num_dims, info, inp, out, |x: u8| f32_to_e4m3(powf(e4m3_to_f32(x), e4m3_to_f32(param))));
    }

    #[kernel]
    pub unsafe fn usign_fp8_e4m3(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u8) {
        unary_loop(numel, num_dims, info, inp, out, |x: u8| f32_to_e4m3(sign_f32(e4m3_to_f32(x))));
    }

    #[kernel]
    pub unsafe fn usigmoid_fp8_e4m3(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u8) {
        unary_loop(numel, num_dims, info, inp, out, |x: u8| f32_to_e4m3(sigmoid_f32(e4m3_to_f32(x))));
    }

    #[kernel]
    pub unsafe fn ucopy_f16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| x);
    }

    #[kernel]
    pub unsafe fn uneg_f16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| hneg(x));
    }

    #[kernel]
    pub unsafe fn urecip_f16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| F16::div(F16::from_d(1.0), x));
    }

    #[kernel]
    pub unsafe fn uexp_f16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| hexp(x));
    }

    #[kernel]
    pub unsafe fn ulog_f16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| hlog(x));
    }

    #[kernel]
    pub unsafe fn usin_f16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| hsin(x));
    }

    #[kernel]
    pub unsafe fn ucos_f16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| hcos(x));
    }

    #[kernel]
    pub unsafe fn utanh_f16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| t_f32::<F16, _>(x, |f| unsafe { __nv_tanhf(f) }));
    }

    #[kernel]
    pub unsafe fn uerf_f16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| t_f32::<F16, _>(x, |f| unsafe { __nv_erff(f) }));
    }

    #[kernel]
    pub unsafe fn uceil_f16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| t_f32::<F16, _>(x, |f| unsafe { __nv_ceilf(f) }));
    }

    #[kernel]
    pub unsafe fn ufloor_f16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| t_f32::<F16, _>(x, |f| unsafe { __nv_floorf(f) }));
    }

    #[kernel]
    pub unsafe fn uround_f16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| t_f32::<F16, _>(x, |f| unsafe { __nv_roundf(f) }));
    }

    #[kernel]
    pub unsafe fn unormcdf_f16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| t_f32::<F16, _>(x, |f| unsafe { __nv_normcdff(f) }));
    }

    #[kernel]
    pub unsafe fn uabs_f16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| habs(x));
    }

    #[kernel]
    pub unsafe fn usqr_f16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| F16::mul(x, x));
    }

    #[kernel]
    pub unsafe fn usqrt_f16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| hsqrt(x));
    }

    #[kernel]
    pub unsafe fn ugelu_f16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| t_gelu::<F16>(x));
    }

    #[kernel]
    pub unsafe fn ugelu_erf_f16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| t_gelu_erf::<F16>(x));
    }

    #[kernel]
    pub unsafe fn urelu_f16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| hmax_nan(x, F16::from_d(0.0)));
    }

    #[kernel]
    pub unsafe fn uelu_f16(numel: usize, num_dims: usize, info: *const usize, param: u16, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| t_elu::<F16>(x, param));
    }

    #[kernel]
    pub unsafe fn usilu_f16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| t_silu::<F16>(x));
    }

    #[kernel]
    pub unsafe fn upowf_f16(numel: usize, num_dims: usize, info: *const usize, param: u16, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| F16::from_f(powf(F16::to_f(x), F16::to_f(param))));
    }

    #[kernel]
    pub unsafe fn usign_f16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| t_sign::<F16>(x));
    }

    #[kernel]
    pub unsafe fn usigmoid_f16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        unary_loop(numel, num_dims, info, inp, out, |x: u16| t_sigmoid::<F16>(x));
    }

    #[kernel]
    pub unsafe fn ucopy_u8(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u8) {
        unary_loop(numel, num_dims, info, inp, out, |x: u8| x);
    }

    #[kernel]
    pub unsafe fn ucopy_u32(numel: usize, num_dims: usize, info: *const usize, inp: *const u32, out: *mut u32) {
        unary_loop(numel, num_dims, info, inp, out, |x: u32| x);
    }

    #[kernel]
    pub unsafe fn ucopy_i64(numel: usize, num_dims: usize, info: *const usize, inp: *const i64, out: *mut i64) {
        unary_loop(numel, num_dims, info, inp, out, |x: i64| x);
    }

    #[kernel]
    pub unsafe fn ucopy_f32(numel: usize, num_dims: usize, info: *const usize, inp: *const f32, out: *mut f32) {
        unary_loop(numel, num_dims, info, inp, out, |x: f32| x);
    }

    #[kernel]
    pub unsafe fn ucopy_f64(numel: usize, num_dims: usize, info: *const usize, inp: *const f64, out: *mut f64) {
        unary_loop(numel, num_dims, info, inp, out, |x: f64| x);
    }

    #[kernel]
    pub unsafe fn uneg_f32(numel: usize, num_dims: usize, info: *const usize, inp: *const f32, out: *mut f32) {
        unary_loop(numel, num_dims, info, inp, out, |x: f32| -x);
    }

    #[kernel]
    pub unsafe fn uneg_f64(numel: usize, num_dims: usize, info: *const usize, inp: *const f64, out: *mut f64) {
        unary_loop(numel, num_dims, info, inp, out, |x: f64| -x);
    }

    #[kernel]
    pub unsafe fn urecip_f32(numel: usize, num_dims: usize, info: *const usize, inp: *const f32, out: *mut f32) {
        unary_loop(numel, num_dims, info, inp, out, |x: f32| float::rcp_rn_f32(x));
    }

    #[kernel]
    pub unsafe fn urecip_f64(numel: usize, num_dims: usize, info: *const usize, inp: *const f64, out: *mut f64) {
        unary_loop(numel, num_dims, info, inp, out, |x: f64| float::rcp_rn_f64(x));
    }

    #[kernel]
    pub unsafe fn uexp_f32(numel: usize, num_dims: usize, info: *const usize, inp: *const f32, out: *mut f32) {
        unary_loop(numel, num_dims, info, inp, out, |x: f32| unsafe { __nv_expf(x) });
    }

    #[kernel]
    pub unsafe fn uexp_f64(numel: usize, num_dims: usize, info: *const usize, inp: *const f64, out: *mut f64) {
        unary_loop(numel, num_dims, info, inp, out, |x: f64| unsafe { __nv_exp(x) });
    }

    #[kernel]
    pub unsafe fn ulog_f32(numel: usize, num_dims: usize, info: *const usize, inp: *const f32, out: *mut f32) {
        unary_loop(numel, num_dims, info, inp, out, |x: f32| unsafe { __nv_logf(x) });
    }

    #[kernel]
    pub unsafe fn ulog_f64(numel: usize, num_dims: usize, info: *const usize, inp: *const f64, out: *mut f64) {
        unary_loop(numel, num_dims, info, inp, out, |x: f64| unsafe { __nv_log(x) });
    }

    #[kernel]
    pub unsafe fn usin_f32(numel: usize, num_dims: usize, info: *const usize, inp: *const f32, out: *mut f32) {
        unary_loop(numel, num_dims, info, inp, out, |x: f32| unsafe { __nv_sinf(x) });
    }

    #[kernel]
    pub unsafe fn usin_f64(numel: usize, num_dims: usize, info: *const usize, inp: *const f64, out: *mut f64) {
        unary_loop(numel, num_dims, info, inp, out, |x: f64| unsafe { __nv_sin(x) });
    }

    #[kernel]
    pub unsafe fn ucos_f32(numel: usize, num_dims: usize, info: *const usize, inp: *const f32, out: *mut f32) {
        unary_loop(numel, num_dims, info, inp, out, |x: f32| unsafe { __nv_cosf(x) });
    }

    #[kernel]
    pub unsafe fn ucos_f64(numel: usize, num_dims: usize, info: *const usize, inp: *const f64, out: *mut f64) {
        unary_loop(numel, num_dims, info, inp, out, |x: f64| unsafe { __nv_cos(x) });
    }

    #[kernel]
    pub unsafe fn utanh_f32(numel: usize, num_dims: usize, info: *const usize, inp: *const f32, out: *mut f32) {
        unary_loop(numel, num_dims, info, inp, out, |x: f32| unsafe { __nv_tanhf(x) });
    }

    #[kernel]
    pub unsafe fn utanh_f64(numel: usize, num_dims: usize, info: *const usize, inp: *const f64, out: *mut f64) {
        unary_loop(numel, num_dims, info, inp, out, |x: f64| unsafe { __nv_tanh(x) });
    }

    #[kernel]
    pub unsafe fn uerf_f32(numel: usize, num_dims: usize, info: *const usize, inp: *const f32, out: *mut f32) {
        unary_loop(numel, num_dims, info, inp, out, |x: f32| unsafe { __nv_erff(x) });
    }

    #[kernel]
    pub unsafe fn uerf_f64(numel: usize, num_dims: usize, info: *const usize, inp: *const f64, out: *mut f64) {
        unary_loop(numel, num_dims, info, inp, out, |x: f64| unsafe { __nv_erf(x) });
    }

    #[kernel]
    pub unsafe fn uceil_f32(numel: usize, num_dims: usize, info: *const usize, inp: *const f32, out: *mut f32) {
        unary_loop(numel, num_dims, info, inp, out, |x: f32| unsafe { __nv_ceilf(x) });
    }

    #[kernel]
    pub unsafe fn uceil_f64(numel: usize, num_dims: usize, info: *const usize, inp: *const f64, out: *mut f64) {
        unary_loop(numel, num_dims, info, inp, out, |x: f64| unsafe { __nv_ceil(x) });
    }

    #[kernel]
    pub unsafe fn ufloor_f32(numel: usize, num_dims: usize, info: *const usize, inp: *const f32, out: *mut f32) {
        unary_loop(numel, num_dims, info, inp, out, |x: f32| unsafe { __nv_floorf(x) });
    }

    #[kernel]
    pub unsafe fn ufloor_f64(numel: usize, num_dims: usize, info: *const usize, inp: *const f64, out: *mut f64) {
        unary_loop(numel, num_dims, info, inp, out, |x: f64| unsafe { __nv_floor(x) });
    }

    #[kernel]
    pub unsafe fn uround_f32(numel: usize, num_dims: usize, info: *const usize, inp: *const f32, out: *mut f32) {
        unary_loop(numel, num_dims, info, inp, out, |x: f32| unsafe { __nv_roundf(x) });
    }

    #[kernel]
    pub unsafe fn uround_f64(numel: usize, num_dims: usize, info: *const usize, inp: *const f64, out: *mut f64) {
        unary_loop(numel, num_dims, info, inp, out, |x: f64| unsafe { __nv_round(x) });
    }

    #[kernel]
    pub unsafe fn unormcdf_f32(numel: usize, num_dims: usize, info: *const usize, inp: *const f32, out: *mut f32) {
        unary_loop(numel, num_dims, info, inp, out, |x: f32| unsafe { __nv_normcdff(x) });
    }

    #[kernel]
    pub unsafe fn unormcdf_f64(numel: usize, num_dims: usize, info: *const usize, inp: *const f64, out: *mut f64) {
        unary_loop(numel, num_dims, info, inp, out, |x: f64| unsafe { __nv_normcdf(x) });
    }

    #[kernel]
    pub unsafe fn uabs_f32(numel: usize, num_dims: usize, info: *const usize, inp: *const f32, out: *mut f32) {
        unary_loop(numel, num_dims, info, inp, out, |x: f32| unsafe { __nv_fabsf(x) });
    }

    #[kernel]
    pub unsafe fn uabs_f64(numel: usize, num_dims: usize, info: *const usize, inp: *const f64, out: *mut f64) {
        unary_loop(numel, num_dims, info, inp, out, |x: f64| unsafe { __nv_fabs(x) });
    }

    #[kernel]
    pub unsafe fn usqr_f32(numel: usize, num_dims: usize, info: *const usize, inp: *const f32, out: *mut f32) {
        unary_loop(numel, num_dims, info, inp, out, |x: f32| float::mul_rn_f32(x, x));
    }

    #[kernel]
    pub unsafe fn usqr_f64(numel: usize, num_dims: usize, info: *const usize, inp: *const f64, out: *mut f64) {
        unary_loop(numel, num_dims, info, inp, out, |x: f64| float::mul_rn_f64(x, x));
    }

    #[kernel]
    pub unsafe fn usqrt_f32(numel: usize, num_dims: usize, info: *const usize, inp: *const f32, out: *mut f32) {
        unary_loop(numel, num_dims, info, inp, out, |x: f32| unsafe { __nv_sqrtf(x) });
    }

    #[kernel]
    pub unsafe fn usqrt_f64(numel: usize, num_dims: usize, info: *const usize, inp: *const f64, out: *mut f64) {
        unary_loop(numel, num_dims, info, inp, out, |x: f64| unsafe { __nv_sqrt(x) });
    }

    #[kernel]
    pub unsafe fn ugelu_f32(numel: usize, num_dims: usize, info: *const usize, inp: *const f32, out: *mut f32) {
        unary_loop(numel, num_dims, info, inp, out, |x: f32| gelu_f32(x));
    }

    #[kernel]
    pub unsafe fn ugelu_f64(numel: usize, num_dims: usize, info: *const usize, inp: *const f64, out: *mut f64) {
        unary_loop(numel, num_dims, info, inp, out, |x: f64| gelu_f64(x));
    }

    #[kernel]
    pub unsafe fn ugelu_erf_f32(numel: usize, num_dims: usize, info: *const usize, inp: *const f32, out: *mut f32) {
        unary_loop(numel, num_dims, info, inp, out, |x: f32| gelu_erf_f32(x));
    }

    #[kernel]
    pub unsafe fn ugelu_erf_f64(numel: usize, num_dims: usize, info: *const usize, inp: *const f64, out: *mut f64) {
        unary_loop(numel, num_dims, info, inp, out, |x: f64| gelu_erf_f64(x));
    }

    #[kernel]
    pub unsafe fn urelu_f32(numel: usize, num_dims: usize, info: *const usize, inp: *const f32, out: *mut f32) {
        unary_loop(numel, num_dims, info, inp, out, |x: f32| unsafe { __nv_fmaxf(x, 0.0) });
    }

    #[kernel]
    pub unsafe fn urelu_f64(numel: usize, num_dims: usize, info: *const usize, inp: *const f64, out: *mut f64) {
        unary_loop(numel, num_dims, info, inp, out, |x: f64| unsafe { __nv_fmax(x, 0.0) });
    }

    #[kernel]
    pub unsafe fn uelu_f32(numel: usize, num_dims: usize, info: *const usize, param: f32, inp: *const f32, out: *mut f32) {
        unary_loop(numel, num_dims, info, inp, out, |x: f32| elu_f32(x, param));
    }

    #[kernel]
    pub unsafe fn uelu_f64(numel: usize, num_dims: usize, info: *const usize, param: f64, inp: *const f64, out: *mut f64) {
        unary_loop(numel, num_dims, info, inp, out, |x: f64| elu_f64(x, param));
    }

    #[kernel]
    pub unsafe fn usilu_f32(numel: usize, num_dims: usize, info: *const usize, inp: *const f32, out: *mut f32) {
        unary_loop(numel, num_dims, info, inp, out, |x: f32| silu_f32(x));
    }

    #[kernel]
    pub unsafe fn usilu_f64(numel: usize, num_dims: usize, info: *const usize, inp: *const f64, out: *mut f64) {
        unary_loop(numel, num_dims, info, inp, out, |x: f64| silu_f64(x));
    }

    #[kernel]
    pub unsafe fn upowf_f32(numel: usize, num_dims: usize, info: *const usize, param: f32, inp: *const f32, out: *mut f32) {
        unary_loop(numel, num_dims, info, inp, out, |x: f32| powf(x, param));
    }

    #[kernel]
    pub unsafe fn upowf_f64(numel: usize, num_dims: usize, info: *const usize, param: f64, inp: *const f64, out: *mut f64) {
        unary_loop(numel, num_dims, info, inp, out, |x: f64| unsafe { __nv_pow(x, param) });
    }

    #[kernel]
    pub unsafe fn usign_f32(numel: usize, num_dims: usize, info: *const usize, inp: *const f32, out: *mut f32) {
        unary_loop(numel, num_dims, info, inp, out, |x: f32| sign_f32(x));
    }

    #[kernel]
    pub unsafe fn usign_f64(numel: usize, num_dims: usize, info: *const usize, inp: *const f64, out: *mut f64) {
        unary_loop(numel, num_dims, info, inp, out, |x: f64| sign_f64(x));
    }

    #[kernel]
    pub unsafe fn usigmoid_f32(numel: usize, num_dims: usize, info: *const usize, inp: *const f32, out: *mut f32) {
        unary_loop(numel, num_dims, info, inp, out, |x: f32| sigmoid_f32(x));
    }

    #[kernel]
    pub unsafe fn usigmoid_f64(numel: usize, num_dims: usize, info: *const usize, inp: *const f64, out: *mut f64) {
        unary_loop(numel, num_dims, info, inp, out, |x: f64| sigmoid_f64(x));
    }
}

// ---------------------------------------------------------------------------------------------
// Differential gate against candle's nvcc-built unary.ptx.
use kdiff::{Arg, Harness, Rng, Tally, as_bytes, contiguous, layout_info};

#[derive(Clone, Copy, PartialEq)]
enum Dt { F16, BF16, F8, F32, F64, U8, U32, I64 }

fn dtype(name: &str) -> Dt {
    for (suf, d) in [("_bf16", Dt::BF16), ("_f16", Dt::F16), ("_e4m3", Dt::F8), ("_f32", Dt::F32), ("_f64", Dt::F64),
                     ("_u8", Dt::U8), ("_u32", Dt::U32), ("_i64", Dt::I64)] {
        if name.ends_with(suf) { return d; }
    }
    panic!("unknown dtype for {name}")
}

fn esize(d: Dt) -> usize {
    match d { Dt::F16 | Dt::BF16 => 2, Dt::F8 | Dt::U8 => 1, Dt::F32 | Dt::U32 => 4, Dt::F64 | Dt::I64 => 8 }
}

/// Element data: float types mix normal ranges with every edge class; 16/8-bit types are raw patterns.
fn data(rng: &mut Rng, d: Dt, n: usize) -> Vec<u8> {
    match d {
        Dt::F32 => as_bytes(&rng.f32s(n)),
        Dt::F64 => as_bytes(&rng.f64s(n)),
        _ => rng.bytes(n * esize(d)),
    }
}

fn scalar(rng: &mut Rng, d: Dt) -> Arg {
    match d {
        Dt::F32 => Arg::F32(rng.f32s(1)[0]),
        Dt::F64 => Arg::F64(rng.f64s(1)[0]),
        Dt::F16 | Dt::BF16 => Arg::B16(rng.b16s(1)[0]),
        _ => Arg::B8(rng.next() as u8),
    }
}

/// Hand-picked parameters (elu alpha / powf exponent): 0, +-1, 0.5, 2, 3, -2.5, nan, inf, -0.
fn special_params(d: Dt) -> Vec<Arg> {
    match d {
        Dt::F32 => [0.0f32, 1.0, -1.0, 0.5, 2.0, 3.0, -2.5, f32::NAN, f32::INFINITY, -0.0].iter().map(|&v| Arg::F32(v)).collect(),
        Dt::F64 => [0.0f64, 1.0, -1.0, 0.5, 2.0, 3.0, -2.5, f64::NAN, f64::INFINITY, -0.0].iter().map(|&v| Arg::F64(v)).collect(),
        Dt::F16 => [0x0000u16, 0x3C00, 0xBC00, 0x3800, 0x4000, 0x4200, 0xC100, 0x7E00, 0x7C00, 0x8000].iter().map(|&v| Arg::B16(v)).collect(),
        Dt::BF16 => [0x0000u16, 0x3F80, 0xBF80, 0x3F00, 0x4000, 0x4040, 0xC020, 0x7FC0, 0x7F80, 0x8000].iter().map(|&v| Arg::B16(v)).collect(),
        _ => [0x00u8, 0x38, 0xB8, 0x30, 0x40, 0x44, 0xC2, 0x7F, 0x7E, 0x80].iter().map(|&v| Arg::B8(v)).collect(),
    }
}

/// Debug aid (`candle-unary --dump <entry> [param-bits]`): run an f32 entry on the mixed sweep
/// and print the first differing elements.
fn dump(h: &Harness, name: &str, param: Option<u32>) {
    use kdiff::cuda_core::DeviceBuffer;
    let mut rng = Rng(7);
    let x = rng.f32s(1 << 16);
    let n = x.len();
    let run = |m: &std::sync::Arc<kdiff::cuda_core::CudaModule>| -> Vec<u8> {
        let inp = DeviceBuffer::from_host(&h.stream, &as_bytes(&x)[..]).unwrap();
        let out = DeviceBuffer::from_host(&h.stream, &vec![0u8; n * 4][..]).unwrap();
        let mut vals: Vec<[u8; 8]> = vec![(n as u64).to_le_bytes(), 1u64.to_le_bytes(), 0u64.to_le_bytes()];
        if let Some(p) = param { vals.push((p as u64).to_le_bytes()); }
        vals.push(inp.cu_deviceptr().to_le_bytes());
        vals.push(out.cu_deviceptr().to_le_bytes());
        let mut ptrs: Vec<*mut std::ffi::c_void> = vals.iter_mut().map(|v| v.as_mut_ptr() as *mut _).collect();
        let f = m.load_function(name).unwrap();
        unsafe { kdiff::cuda_core::simt::launch_kernel_on_stream(&f, (64, 1, 1), (256, 1, 1), 0, &h.stream, &mut ptrs).unwrap(); }
        h.stream.synchronize().unwrap();
        out.to_host_vec(&h.stream).unwrap()
    };
    let a = run(&h.reference);
    let b = run(&h.oxide);
    let mut shown = 0;
    let mut total = 0;
    for i in 0..n {
        let ra = u32::from_le_bytes(a[i * 4..i * 4 + 4].try_into().unwrap());
        let rb = u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
        if ra != rb {
            total += 1;
            if shown < 20 {
                println!("x={:e} ({:#010x}) ref={:e} ({ra:#010x}) oxide={:e} ({rb:#010x})", x[i], x[i].to_bits(), f32::from_bits(ra), f32::from_bits(rb));
                shown += 1;
            }
        }
    }
    println!("{total} of {n} differ");
}

/// Optional exhaustive check (`candle-unary --exhaustive [filter]`): every f32 bit pattern through
/// every f32 entry (elu/powf with a few fixed parameters), in chunks of 2^24 elements.
fn exhaustive(h: &Harness, entries: &[String], only: Option<&str>) -> bool {
    let mut t = Tally::default();
    let chunk: u64 = 1 << 24;
    for name in entries.iter().filter(|n| n.ends_with("_f32")) {
        if let Some(o) = only { if !name.contains(o) { continue; } }
        let params: Vec<Option<Arg>> = if name.starts_with("uelu_") {
            vec![Some(Arg::F32(1.0)), Some(Arg::F32(-0.75))]
        } else if name.starts_with("upowf_") {
            vec![Some(Arg::F32(0.5)), Some(Arg::F32(3.0)), Some(Arg::F32(-2.5)), Some(Arg::F32(1.0e-3))]
        } else {
            vec![None]
        };
        let start = std::time::Instant::now();
        for p in &params {
            let mut failing = 0;
            for c in 0..(1u64 << 32) / chunk {
                let v: Vec<u32> = (c * chunk..(c + 1) * chunk).map(|b| b as u32).collect();
                let mut args = vec![Arg::U64(chunk), Arg::U64(1), Arg::Null];
                if let Some(p) = p.clone() { args.push(p); }
                args.push(Arg::Buf(0));
                args.push(Arg::Buf(1));
                let bufs = vec![as_bytes(&v), vec![0u8; (chunk * 4) as usize]];
                let dd = h.diff(name, (1024, 1, 1), (256, 1, 1), 0, &args, &bufs, &[1]);
                if dd.differing > 0 { failing += 1; }
                t.record(&format!("{name} exhaustive chunk {c}"), &dd);
            }
            println!("  {name} param={}: all 2^32 inputs, {failing} failing chunks ({:.1?})",
                     match p { Some(Arg::F32(v)) => format!("{v}"), _ => "-".into() }, start.elapsed());
        }
    }
    t.finish("unary exhaustive f32")
}

fn main() {
    let root = format!("{}/titan-engine/oxide-kernels", std::env::var("HOME").unwrap());
    let ref_ptx = format!("{root}/reference/candle/unary.ptx");
    let h = Harness::new(&ref_ptx, &format!("{root}/candle-unary/candle_unary.ptx"));
    let entries: Vec<String> = std::fs::read_to_string(&ref_ptx).unwrap().lines()
        .filter_map(|l| l.strip_prefix(".visible .entry ")).map(|l| l.trim_end_matches('(').to_string()).collect();
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 2 && args[1] == "--dump" {
        dump(&h, &args[2], args.get(3).map(|p| u32::from_str_radix(p.trim_start_matches("0x"), 16).unwrap()));
        return;
    }
    if args.len() > 1 && args[1] == "--exhaustive" {
        let ok = exhaustive(&h, &entries, args.get(2).map(|s| s.as_str()));
        std::process::exit(if ok { 0 } else { 1 });
    }
    let only = args.get(1).cloned();
    let mut t = Tally::default();
    let mut rng = Rng(0x0E1A_u64 << 20 | 0x5EED);
    let layouts: Vec<(Vec<usize>, Vec<usize>)> = vec![
        (vec![1000], vec![1]),
        (vec![7, 13, 5], contiguous(&[7, 13, 5])),
        (vec![13, 7], vec![1, 13]),                // transposed
        (vec![4, 1, 9], vec![9, 9, 1]),            // size-1 dim with odd stride
        (vec![6, 5], vec![0, 1]),                  // broadcast
        (vec![3, 4, 5], vec![40, 1, 8]),           // permuted
        (vec![2, 1, 3, 1], vec![1, 7, 2, 5]),      // permuted, several size-1 dims
    ];
    let mut missing = 0;
    for name in &entries {
        if let Some(o) = &only { if !name.contains(o.as_str()) { continue; } }
        if !h.has(true, name) {
            println!("  MISSING {name}");
            missing += 1;
            continue;
        }
        let d = dtype(name);
        let es = esize(d);
        let has_param = name.starts_with("uelu_") || name.starts_with("upowf_");
        // 1) layouts x grids x info/no-info x in-place
        for (li, (dims, strides)) in layouts.iter().enumerate() {
            let numel: usize = dims.iter().product();
            let src_len = dims.iter().zip(strides).map(|(d, s)| (d - 1) * s).sum::<usize>() + 1;
            for &(grid, block) in &[(4u32, 64u32), (((numel as u32) + 255) / 256, 256), (1, 32), (3, 7)] {
                for with_info in [true, false] {
                    if !with_info && li > 1 { continue; }
                    for inplace in [false, true] {
                        let inp = data(&mut rng, d, src_len.max(numel));
                        let out = data(&mut rng, d, numel);
                        let info = if with_info { Arg::Buf(2) } else { Arg::Null };
                        let inp_arg = if inplace { Arg::Null } else { Arg::Buf(0) };
                        let mut args = vec![Arg::U64(numel as u64), Arg::U64(dims.len() as u64), info];
                        if has_param { args.push(scalar(&mut rng, d)); }
                        args.push(inp_arg);
                        args.push(Arg::Buf(1));
                        let bufs = vec![inp, out, layout_info(dims, strides)];
                        let dd = h.diff(name, (grid, 1, 1), (block, 1, 1), 0, &args, &bufs, &[1]);
                        t.record(&format!("{name} layout{li} g{grid}x{block} info={with_info} inplace={inplace}"), &dd);
                    }
                }
            }
        }
        // 2) exhaustive (16/8-bit) or large random (32/64-bit) contiguous sweep, grid-stride loop
        let big: Vec<u8> = match d {
            Dt::F16 | Dt::BF16 => as_bytes(&(0..=u16::MAX).collect::<Vec<u16>>()),
            Dt::F8 | Dt::U8 => (0..=255u8).collect(),
            Dt::F32 => {
                let mut v = rng.f32s(1 << 16);
                // every exponent with a few mantissas, both signs
                for e in 0..256u32 { for m in [0u32, 1, 0x400000, 0x7FFFFF, 0x2A5A5A] { for s in [0u32, 1] {
                    v.push(f32::from_bits(s << 31 | e << 23 | m));
                } } }
                as_bytes(&v)
            }
            Dt::F64 => {
                let mut v = rng.f64s(1 << 14);
                for e in (0..2048u64).step_by(3) { for m in [0u64, 1, 1 << 51, (1 << 52) - 1] { for s in [0u64, 1] {
                    v.push(f64::from_bits(s << 63 | e << 52 | m));
                } } }
                as_bytes(&v)
            }
            _ => rng.bytes(4096 * es),
        };
        let numel = big.len() / es;
        let params: Vec<Option<Arg>> = if has_param {
            let mut p: Vec<Option<Arg>> = special_params(d).into_iter().map(Some).collect();
            for _ in 0..(if name.starts_with("upowf_") { 24 } else { 6 }) { p.push(Some(scalar(&mut rng, d))); }
            p
        } else { vec![None] };
        for p in params {
            for &(grid, block) in &[(16u32, 128u32), (1, 1024)] {
                let mut args = vec![Arg::U64(numel as u64), Arg::U64(1), Arg::Null];
                if let Some(p) = p.clone() { args.push(p); }
                args.push(Arg::Buf(0));
                args.push(Arg::Buf(1));
                let out = data(&mut rng, d, numel);
                let dd = h.diff(name, (grid, 1, 1), (block, 1, 1), 0, &args, &[big.clone(), out], &[1]);
                t.record(&format!("{name} sweep n={numel} g{grid}x{block}"), &dd);
            }
        }
    }
    if missing > 0 {
        println!("unary: {missing} reference entries missing from the oxide PTX");
    }
    let ok = t.finish("unary") && missing == 0;
    std::process::exit(if ok { 0 } else { 1 });
}
