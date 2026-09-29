#![allow(unsafe_op_in_unsafe_fn)]
//! candle-kernels `reduce.cu` in cuda-oxide: same 64 entry names, same by-value/raw-pointer ABI,
//! bit-identical output (checked by the host `main` against candle's own nvcc PTX).
//!
//! Semantics copied from the SASS, not just the source:
//! - fast_{sum,min,max,argmin,argmax}: one block per output, a per-thread running value (the
//!   reference keeps `shr[tid]` in a register until the tree), then the shared-memory tree
//!   `for s = blockDim/2; s > 0; s >>= 1 { sync; if tid < s: shr[tid] = op(shr[tid], shr[tid+s]) }`.
//!   f16/bf16 min/max are `min.NaN`/`max.NaN`; f16/bf16 sums accumulate in half precision.
//! - sum_*: atomicAdd scatter (`atom.add.{f32,f64,u32}`, `atom.add.noftz.{f16,bf16}`).
//! - softmax: warp-only butterfly (no shared memory); f32/f64 use libdevice exp, bf16 `hexp` is
//!   `ex2.approx.f32(x * 0x3FB8AA3C)`, f16 `hexp` is cuda_fp16's inline PTX (copied verbatim);
//!   `1. / tmp` is `rcp.rn`, the final scale is a half/float multiply.
//! - rmsnorm/layernorm: f32 accumulation (`fma(x, x, acc)`), butterfly, shared handoff when
//!   block_size > 32, `div.rn` by ncols, `rsqrt.approx.f32` (non-ftz).
//!   ptxas contracts `mean_sq - mean*mean` into `fma(-mean, mean, mean_sq)` and `lhs + b` /
//!   `lhs * a + b` into fmas; f64 variants compute in f32.
//! - rope*: `o1 = fma(c, x1, -(s * x2))`, `o2 = fma(s, x1, c * x2)` (FFMA/DFMA/HFMA2 in the SASS),
//!   with x1/x2 re-read after the first store (in-place aliasing).
use cuda_device::{SharedArray, bf16, bf16x2, convert, f16, f16x2, float, kernel, ptx_asm, thread, warp};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    const FULL: u32 = 0xffff_ffff;

    // ------------------------------------------------------------------ shared memory
    /// One block-shared scratch area for every kernel: fast_* use T[1024] at 0 and u32[1024] at
    /// 8192; rmsnorm/layernorm use float / float2 [32] at 0.
    #[inline(always)]
    unsafe fn smem() -> *mut u8 {
        static mut SMEM: SharedArray<u64, 1536> = SharedArray::UNINIT;
        SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *mut u8
    }

    // ------------------------------------------------------------------ index helpers
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

    #[inline(always)]
    pub unsafe fn get_strided_index(idx: u32, num_dims: usize, dims: *const usize, strides: *const usize) -> u32 {
        let mut idx = idx;
        let mut strided_i: u32 = 0;
        let mut d: u32 = 0;
        while (d as usize) < num_dims {
            let dim_idx = (num_dims.wrapping_sub(1).wrapping_sub(d as usize)) as u32 as usize;
            let dim = *dims.add(dim_idx);
            let term = (idx as usize % dim).wrapping_mul(*strides.add(dim_idx));
            strided_i = (strided_i as usize).wrapping_add(term) as u32;
            idx = (idx as usize / dim) as u32;
            d += 1;
        }
        strided_i
    }

    // ------------------------------------------------------------------ half helpers
    #[inline(always)]
    fn bf2f(x: u16) -> f32 {
        f32::from_bits((x as u32) << 16)
    }
    #[inline(always)]
    fn hf2f(x: u16) -> f32 {
        convert::cvt_f32_f16x2_lo(x as u32)
    }
    #[inline(always)]
    fn f2bf(v: f32) -> u16 {
        convert::cvt_bf16x2_f32(v, 0.0) as u16
    }
    #[inline(always)]
    fn f2hf(v: f32) -> u16 {
        convert::cvt_f16x2_f32(v, 0.0) as u16
    }
    #[inline(always)]
    fn bf_add(a: u16, b: u16) -> u16 {
        bf16x2::add_bf16x2(a as u32, b as u32) as u16
    }
    #[inline(always)]
    fn hf_add(a: u16, b: u16) -> u16 {
        f16x2::add_f16x2(a as u32, b as u32) as u16
    }
    #[inline(always)]
    fn bf_mul(a: u16, b: u16) -> u16 {
        bf16x2::mul_bf16x2(a as u32, b as u32) as u16
    }
    #[inline(always)]
    fn hf_mul(a: u16, b: u16) -> u16 {
        f16x2::mul_f16x2(a as u32, b as u32) as u16
    }
    #[inline(always)]
    fn bf_fma(a: u16, b: u16, c: u16) -> u16 {
        bf16x2::fma_bf16x2(a as u32, b as u32, c as u32) as u16
    }
    #[inline(always)]
    fn hf_fma(a: u16, b: u16, c: u16) -> u16 {
        f16x2::fma_f16x2(a as u32, b as u32, c as u32) as u16
    }
    #[inline(always)]
    fn bf_sub(a: u16, b: u16) -> u16 {
        bf16x2::sub_bf16x2(a as u32, b as u32) as u16
    }
    #[inline(always)]
    fn hf_sub(a: u16, b: u16) -> u16 {
        f16x2::sub_f16x2(a as u32, b as u32) as u16
    }

    /// cuda_bf16 `hexp`: ex2.approx.f32(float(x) * 0x3FB8AA3C) rounded to bf16.
    #[inline(always)]
    fn bf_exp(x: u16) -> u16 {
        f2bf(float::ex2_approx_f32(float::mul_rn_f32(bf2f(x), f32::from_bits(0x3FB8_AA3C))))
    }

    /// cuda_fp16 `hexp`, the reference's inline PTX verbatim.
    #[inline(always)]
    fn hf_exp(x: u16) -> u16 {
        let r: u16;
        unsafe {
            ptx_asm!(
                "{.reg.b32 f, C, nZ; .reg.b16 h,r; mov.b16 h,%1; cvt.f32.f16 f,h; \
                 mov.b32 C, 0x3fb8aa3bU; mov.b32 nZ, 0x80000000U; fma.rn.f32 f,f,C,nZ; \
                 ex2.approx.ftz.f32 f,f; cvt.rn.f16.f32 r,f; \
                 {.reg.b16 spc, ulp, p; mov.b16 spc,0X1F79U; mov.b16 ulp,0x9400U; set.eq.f16.f16 p,h, spc; fma.rn.f16 r,p,ulp,r;} \
                 {.reg.b16 spc, ulp, p; mov.b16 spc,0X25CFU; mov.b16 ulp,0x9400U; set.eq.f16.f16 p,h, spc; fma.rn.f16 r,p,ulp,r;} \
                 {.reg.b16 spc, ulp, p; mov.b16 spc,0XC13BU; mov.b16 ulp,0x0400U; set.eq.f16.f16 p,h, spc; fma.rn.f16 r,p,ulp,r;} \
                 {.reg.b16 spc, ulp, p; mov.b16 spc,0XC1EFU; mov.b16 ulp,0x0200U; set.eq.f16.f16 p,h, spc; fma.rn.f16 r,p,ulp,r;} \
                 mov.b16 %0,r;}",
                out("=h") r,
                in("h") x,
                options(register_only),
            );
        }
        r
    }

    // ------------------------------------------------------------------ PTX min/max
    // Rust's f32::min/max lower to setp/selp chains (order-dependent on +-0); the reference
    // uses the PTX min/max instructions, so emit exactly those.
    #[inline(always)]
    fn min32(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("min.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    #[inline(always)]
    fn max32(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("max.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    #[inline(always)]
    fn min64(a: f64, b: f64) -> f64 {
        let r: f64;
        unsafe { ptx_asm!("min.f64 %0, %1, %2;", out("=d") r, in("d") a, in("d") b, options(register_only)); }
        r
    }
    #[inline(always)]
    fn max64(a: f64, b: f64) -> f64 {
        let r: f64;
        unsafe { ptx_asm!("max.f64 %0, %1, %2;", out("=d") r, in("d") a, in("d") b, options(register_only)); }
        r
    }

    // ------------------------------------------------------------------ f64 NaN operand order
    // f32/f16/bf16 arithmetic returns a canonical NaN, but DADD/DMUL return the NaN of their
    // *second SASS operand* (quieted) when both inputs are NaN, and ptxas picks the SASS operand
    // order by its own heuristics (it differs between unrolled copies of one loop). So the
    // reference's choice is reproduced explicitly: `keep` is the operand whose NaN survives.
    #[inline(always)]
    fn quiet64(x: f64) -> f64 {
        f64::from_bits(x.to_bits() | 0x0008_0000_0000_0000)
    }
    #[inline(always)]
    fn add64_keep(other: f64, keep: f64) -> f64 {
        if other.is_nan() && keep.is_nan() { quiet64(keep) } else { other + keep }
    }
    #[inline(always)]
    fn mul64_keep(other: f64, keep: f64) -> f64 {
        if other.is_nan() && keep.is_nan() { quiet64(keep) } else { other * keep }
    }
    #[inline(always)]
    fn sub64_keep_x(x: f64, m: f64) -> f64 {
        if x.is_nan() && m.is_nan() { quiet64(x) } else { x - m }
    }

    // ------------------------------------------------------------------ fast reductions
    /// fast_sum / fast_min / fast_max. `op(shr[tid], x)`.
    #[inline(always)]
    pub unsafe fn fast_reduce<T: Copy, F: Fn(T, T) -> T>(
        src_numel: usize, per_block: usize, num_dims: usize, info: *const usize, src: *const T, dst: *mut T,
        init: T, op: F,
    ) {
        let dims = info;
        let strides = info.wrapping_add(num_dims);
        let shr = smem() as *mut T;
        let tid = thread::threadIdx_x() as usize;
        let dst_id = thread::blockIdx_x() as usize;
        let bdim = thread::blockDim_x() as usize;
        let start = dst_id.wrapping_mul(per_block);
        let stop = core::cmp::min(start.wrapping_add(per_block), src_numel);
        let mut idx = start.wrapping_add(tid);
        let mut acc = init;
        while idx < stop {
            let s = get_strided_index(idx as u32, num_dims, dims, strides);
            acc = op(acc, *src.add(s as usize));
            idx = idx.wrapping_add(bdim);
        }
        *shr.add(tid) = acc;
        let mut s = (thread::blockDim_x() / 2) as i32;
        while s > 0 {
            thread::sync_threads();
            if tid < s as usize {
                *shr.add(tid) = op(*shr.add(tid), *shr.add(tid + s as usize));
            }
            s >>= 1;
        }
        if tid == 0 {
            *dst.add(dst_id) = *shr;
        }
    }

    /// fast_argmin / fast_argmax. `better(x, cur)` is `x < cur` (argmin) or `x > cur` (argmax).
    #[inline(always)]
    pub unsafe fn fast_arg<T: Copy, F: Fn(T, T) -> bool>(
        src_numel: usize, per_block: usize, num_dims: usize, info: *const usize, src: *const T, dst: *mut u32,
        init: T, better: F,
    ) {
        let dims = info;
        let strides = info.wrapping_add(num_dims);
        let shr = smem() as *mut T;
        let shr_index = smem().add(8192) as *mut u32;
        let tid = thread::threadIdx_x() as usize;
        let dst_id = thread::blockIdx_x() as usize;
        let bdim = thread::blockDim_x() as usize;
        let start = dst_id.wrapping_mul(per_block);
        let stop = core::cmp::min(start.wrapping_add(per_block), src_numel);
        let mut idx = start.wrapping_add(tid);
        let mut v = init;
        let mut vi: u32 = 0xFFFF_FFFF;
        let mut not_set = true;
        while idx < stop {
            let s = get_strided_index(idx as u32, num_dims, dims, strides);
            let x = *src.add(s as usize);
            if not_set || better(x, v) {
                v = x;
                vi = (idx % *dims.add(num_dims.wrapping_sub(1))) as u32;
                not_set = false;
            }
            idx = idx.wrapping_add(bdim);
        }
        *shr.add(tid) = v;
        *shr_index.add(tid) = vi;
        let mut s = (thread::blockDim_x() / 2) as i32;
        while s > 0 {
            thread::sync_threads();
            if tid < s as usize {
                let o = tid + s as usize;
                if better(*shr.add(o), *shr.add(tid)) {
                    *shr.add(tid) = *shr.add(o);
                    *shr_index.add(tid) = *shr_index.add(o);
                }
            }
            s >>= 1;
        }
        if tid == 0 {
            *dst.add(dst_id) = *shr_index;
        }
    }

    // bf16
    #[kernel]
    pub unsafe fn fast_sum_bf16(n: usize, pb: usize, nd: usize, info: *const usize, src: *const u16, dst: *mut u16) {
        fast_reduce(n, pb, nd, info, src, dst, 0u16, bf_add);
    }
    #[kernel]
    pub unsafe fn fast_min_bf16(n: usize, pb: usize, nd: usize, info: *const usize, src: *const u16, dst: *mut u16) {
        fast_reduce(n, pb, nd, info, src, dst, 0x7F80u16, |a, b| bf16::min_nan_bf16(a, b));
    }
    #[kernel]
    pub unsafe fn fast_max_bf16(n: usize, pb: usize, nd: usize, info: *const usize, src: *const u16, dst: *mut u16) {
        fast_reduce(n, pb, nd, info, src, dst, 0xFF80u16, |a, b| bf16::max_nan_bf16(a, b));
    }
    #[kernel]
    pub unsafe fn fast_argmin_bf16(n: usize, pb: usize, nd: usize, info: *const usize, src: *const u16, dst: *mut u32) {
        fast_arg(n, pb, nd, info, src, dst, 0x7F80u16, |a, b| bf2f(a) < bf2f(b));
    }
    #[kernel]
    pub unsafe fn fast_argmax_bf16(n: usize, pb: usize, nd: usize, info: *const usize, src: *const u16, dst: *mut u32) {
        fast_arg(n, pb, nd, info, src, dst, 0xFF80u16, |a, b| bf2f(a) > bf2f(b));
    }
    // f16
    #[kernel]
    pub unsafe fn fast_sum_f16(n: usize, pb: usize, nd: usize, info: *const usize, src: *const u16, dst: *mut u16) {
        fast_reduce(n, pb, nd, info, src, dst, 0u16, hf_add);
    }
    #[kernel]
    pub unsafe fn fast_min_f16(n: usize, pb: usize, nd: usize, info: *const usize, src: *const u16, dst: *mut u16) {
        fast_reduce(n, pb, nd, info, src, dst, 0x7C00u16, |a, b| f16::min_nan_f16(a, b));
    }
    #[kernel]
    pub unsafe fn fast_max_f16(n: usize, pb: usize, nd: usize, info: *const usize, src: *const u16, dst: *mut u16) {
        fast_reduce(n, pb, nd, info, src, dst, 0xFC00u16, |a, b| f16::max_nan_f16(a, b));
    }
    #[kernel]
    pub unsafe fn fast_argmin_f16(n: usize, pb: usize, nd: usize, info: *const usize, src: *const u16, dst: *mut u32) {
        fast_arg(n, pb, nd, info, src, dst, 0x7C00u16, |a, b| hf2f(a) < hf2f(b));
    }
    #[kernel]
    pub unsafe fn fast_argmax_f16(n: usize, pb: usize, nd: usize, info: *const usize, src: *const u16, dst: *mut u32) {
        fast_arg(n, pb, nd, info, src, dst, 0xFC00u16, |a, b| hf2f(a) > hf2f(b));
    }
    // f32
    #[kernel]
    pub unsafe fn fast_sum_f32(n: usize, pb: usize, nd: usize, info: *const usize, src: *const f32, dst: *mut f32) {
        fast_reduce(n, pb, nd, info, src, dst, 0.0f32, |a, b| float::add_rn_f32(a, b));
    }
    #[kernel]
    pub unsafe fn fast_min_f32(n: usize, pb: usize, nd: usize, info: *const usize, src: *const f32, dst: *mut f32) {
        fast_reduce(n, pb, nd, info, src, dst, f32::INFINITY, min32);
    }
    #[kernel]
    pub unsafe fn fast_max_f32(n: usize, pb: usize, nd: usize, info: *const usize, src: *const f32, dst: *mut f32) {
        fast_reduce(n, pb, nd, info, src, dst, f32::NEG_INFINITY, max32);
    }
    #[kernel]
    pub unsafe fn fast_argmin_f32(n: usize, pb: usize, nd: usize, info: *const usize, src: *const f32, dst: *mut u32) {
        fast_arg(n, pb, nd, info, src, dst, f32::INFINITY, |a: f32, b: f32| a < b);
    }
    #[kernel]
    pub unsafe fn fast_argmax_f32(n: usize, pb: usize, nd: usize, info: *const usize, src: *const f32, dst: *mut u32) {
        fast_arg(n, pb, nd, info, src, dst, f32::NEG_INFINITY, |a: f32, b: f32| a > b);
    }
    // f64
    #[kernel]
    pub unsafe fn fast_sum_f64(n: usize, pb: usize, nd: usize, info: *const usize, src: *const f64, dst: *mut f64) {
        fast_reduce(n, pb, nd, info, src, dst, 0.0f64, |acc, x| add64_keep(x, acc));
    }
    #[kernel]
    pub unsafe fn fast_min_f64(n: usize, pb: usize, nd: usize, info: *const usize, src: *const f64, dst: *mut f64) {
        fast_reduce(n, pb, nd, info, src, dst, f64::INFINITY, min64);
    }
    #[kernel]
    pub unsafe fn fast_max_f64(n: usize, pb: usize, nd: usize, info: *const usize, src: *const f64, dst: *mut f64) {
        fast_reduce(n, pb, nd, info, src, dst, f64::NEG_INFINITY, max64);
    }
    #[kernel]
    pub unsafe fn fast_argmin_f64(n: usize, pb: usize, nd: usize, info: *const usize, src: *const f64, dst: *mut u32) {
        fast_arg(n, pb, nd, info, src, dst, f64::INFINITY, |a: f64, b: f64| a < b);
    }
    #[kernel]
    pub unsafe fn fast_argmax_f64(n: usize, pb: usize, nd: usize, info: *const usize, src: *const f64, dst: *mut u32) {
        fast_arg(n, pb, nd, info, src, dst, f64::NEG_INFINITY, |a: f64, b: f64| a > b);
    }
    // u32
    #[kernel]
    pub unsafe fn fast_sum_u32(n: usize, pb: usize, nd: usize, info: *const usize, src: *const u32, dst: *mut u32) {
        fast_reduce(n, pb, nd, info, src, dst, 0u32, |a: u32, b: u32| a.wrapping_add(b));
    }
    #[kernel]
    pub unsafe fn fast_min_u32(n: usize, pb: usize, nd: usize, info: *const usize, src: *const u32, dst: *mut u32) {
        fast_reduce(n, pb, nd, info, src, dst, u32::MAX, |a: u32, b: u32| a.min(b));
    }
    #[kernel]
    pub unsafe fn fast_max_u32(n: usize, pb: usize, nd: usize, info: *const usize, src: *const u32, dst: *mut u32) {
        fast_reduce(n, pb, nd, info, src, dst, 0u32, |a: u32, b: u32| a.max(b));
    }
    #[kernel]
    pub unsafe fn fast_argmin_u32(n: usize, pb: usize, nd: usize, info: *const usize, src: *const u32, dst: *mut u32) {
        fast_arg(n, pb, nd, info, src, dst, u32::MAX, |a: u32, b: u32| a < b);
    }
    #[kernel]
    pub unsafe fn fast_argmax_u32(n: usize, pb: usize, nd: usize, info: *const usize, src: *const u32, dst: *mut u32) {
        fast_arg(n, pb, nd, info, src, dst, 0u32, |a: u32, b: u32| a > b);
    }
    // i64
    #[kernel]
    pub unsafe fn fast_sum_i64(n: usize, pb: usize, nd: usize, info: *const usize, src: *const i64, dst: *mut i64) {
        fast_reduce(n, pb, nd, info, src, dst, 0i64, |a: i64, b: i64| a.wrapping_add(b));
    }
    #[kernel]
    pub unsafe fn fast_min_i64(n: usize, pb: usize, nd: usize, info: *const usize, src: *const i64, dst: *mut i64) {
        fast_reduce(n, pb, nd, info, src, dst, i64::MAX, |a: i64, b: i64| a.min(b));
    }
    #[kernel]
    pub unsafe fn fast_max_i64(n: usize, pb: usize, nd: usize, info: *const usize, src: *const i64, dst: *mut i64) {
        fast_reduce(n, pb, nd, info, src, dst, i64::MIN, |a: i64, b: i64| a.max(b));
    }
    #[kernel]
    pub unsafe fn fast_argmin_i64(n: usize, pb: usize, nd: usize, info: *const usize, src: *const i64, dst: *mut u32) {
        fast_arg(n, pb, nd, info, src, dst, i64::MAX, |a: i64, b: i64| a < b);
    }
    #[kernel]
    pub unsafe fn fast_argmax_i64(n: usize, pb: usize, nd: usize, info: *const usize, src: *const i64, dst: *mut u32) {
        fast_arg(n, pb, nd, info, src, dst, i64::MIN, |a: i64, b: i64| a > b);
    }
    // u8
    #[kernel]
    pub unsafe fn fast_sum_u8(n: usize, pb: usize, nd: usize, info: *const usize, src: *const u8, dst: *mut u8) {
        fast_reduce(n, pb, nd, info, src, dst, 0u8, |a: u8, b: u8| a.wrapping_add(b));
    }
    #[kernel]
    pub unsafe fn fast_min_u8(n: usize, pb: usize, nd: usize, info: *const usize, src: *const u8, dst: *mut u8) {
        fast_reduce(n, pb, nd, info, src, dst, u8::MAX, |a: u8, b: u8| a.min(b));
    }
    #[kernel]
    pub unsafe fn fast_max_u8(n: usize, pb: usize, nd: usize, info: *const usize, src: *const u8, dst: *mut u8) {
        fast_reduce(n, pb, nd, info, src, dst, 0u8, |a: u8, b: u8| a.max(b));
    }
    #[kernel]
    pub unsafe fn fast_argmin_u8(n: usize, pb: usize, nd: usize, info: *const usize, src: *const u8, dst: *mut u32) {
        fast_arg(n, pb, nd, info, src, dst, u8::MAX, |a: u8, b: u8| a < b);
    }
    #[kernel]
    pub unsafe fn fast_argmax_u8(n: usize, pb: usize, nd: usize, info: *const usize, src: *const u8, dst: *mut u32) {
        fast_arg(n, pb, nd, info, src, dst, 0u8, |a: u8, b: u8| a > b);
    }

    // ------------------------------------------------------------------ sum (atomic scatter)
    #[inline(always)]
    pub unsafe fn sum_op<T: Copy, A: Fn(*mut T, T)>(
        numel: usize, num_dims: usize, num_sum_dims: usize, info: *const usize, inp: *const T, out: *mut T, atomic_add: A,
    ) {
        let dims = info;
        let strides = info.wrapping_add(num_dims);
        let sum_dims_l = info.wrapping_add(2 * num_dims);
        let sum_dims_s = info.wrapping_add(2 * num_dims + num_sum_dims);
        let step = thread::blockDim_x().wrapping_mul(thread::gridDim_x());
        let mut i: u32 = thread::blockIdx_x().wrapping_mul(thread::blockDim_x()).wrapping_add(thread::threadIdx_x());
        let contiguous = is_contiguous(num_dims, dims, strides);
        while (i as usize) < numel {
            let src_i = if contiguous { i } else { get_strided_index(i, num_dims, dims, strides) };
            let mut dst_index = i as usize;
            let mut nd: u32 = 0;
            while (nd as usize) < num_sum_dims {
                let stride = *sum_dims_s.add(nd as usize);
                let pre = dst_index / stride;
                let post = dst_index % stride;
                dst_index = (pre / *sum_dims_l.add(nd as usize)).wrapping_mul(stride).wrapping_add(post);
                nd += 1;
            }
            atomic_add(out.wrapping_add(dst_index), *inp.add(src_i as usize));
            i = i.wrapping_add(step);
        }
    }

    #[inline(always)]
    unsafe fn atom_add_bf16(p: *mut u16, v: u16) {
        let _old: u16;
        ptx_asm!("atom.add.noftz.bf16 %0,[%1],%2;", out("=h") _old, in("l") p, in("h") v);
    }
    #[inline(always)]
    unsafe fn atom_add_f16(p: *mut u16, v: u16) {
        let _old: u16;
        ptx_asm!("atom.add.noftz.f16 %0,[%1],%2;", out("=h") _old, in("l") p, in("h") v);
    }
    #[inline(always)]
    unsafe fn atom_add_f32(p: *mut f32, v: f32) {
        let _old: f32;
        ptx_asm!("atom.global.add.f32 %0,[%1],%2;", out("=f") _old, in("l") p, in("f") v);
    }
    #[inline(always)]
    unsafe fn atom_add_f64(p: *mut f64, v: f64) {
        let _old: f64;
        ptx_asm!("atom.global.add.f64 %0,[%1],%2;", out("=d") _old, in("l") p, in("d") v);
    }
    #[inline(always)]
    unsafe fn atom_add_u32(p: *mut u32, v: u32) {
        let _old: u32;
        ptx_asm!("atom.global.add.u32 %0,[%1],%2;", out("=r") _old, in("l") p, in("r") v);
    }

    #[kernel]
    pub unsafe fn sum_bf16(numel: usize, nd: usize, nsd: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        sum_op(numel, nd, nsd, info, inp, out, |p, v| atom_add_bf16(p, v));
    }
    #[kernel]
    pub unsafe fn sum_f16(numel: usize, nd: usize, nsd: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        sum_op(numel, nd, nsd, info, inp, out, |p, v| atom_add_f16(p, v));
    }
    #[kernel]
    pub unsafe fn sum_f32(numel: usize, nd: usize, nsd: usize, info: *const usize, inp: *const f32, out: *mut f32) {
        sum_op(numel, nd, nsd, info, inp, out, |p, v| atom_add_f32(p, v));
    }
    #[kernel]
    pub unsafe fn sum_f64(numel: usize, nd: usize, nsd: usize, info: *const usize, inp: *const f64, out: *mut f64) {
        sum_op(numel, nd, nsd, info, inp, out, |p, v| atom_add_f64(p, v));
    }
    #[kernel]
    pub unsafe fn sum_u32(numel: usize, nd: usize, nsd: usize, info: *const usize, inp: *const u32, out: *mut u32) {
        sum_op(numel, nd, nsd, info, inp, out, |p, v| atom_add_u32(p, v));
    }

    // ------------------------------------------------------------------ warp sums
    #[inline(always)]
    fn warp_sum(mut x: f32) -> f32 {
        let mut m = 16;
        while m > 0 {
            x = float::add_rn_f32(x, warp::shuffle_xor_f32_sync(FULL, x, m));
            m >>= 1;
        }
        x
    }
    #[inline(always)]
    fn warp_sum2(mut x: f32, mut y: f32) -> (f32, f32) {
        let mut m = 16;
        while m > 0 {
            x = float::add_rn_f32(x, warp::shuffle_xor_f32_sync(FULL, x, m));
            y = float::add_rn_f32(y, warp::shuffle_xor_f32_sync(FULL, y, m));
            m >>= 1;
        }
        (x, y)
    }

    // ------------------------------------------------------------------ softmax
    /// `T` element, `A` accumulator. `expsub(x, max)` = expg(x - max) in T; `acc(tmp, val)` =
    /// tmp + ACC(val); `inv(tmp)` = the scale in the element domain; `scale(v, inv)` = v * inv.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn softmax<T: Copy, A: Copy, I: Copy>(
        x: *const T, dst: *mut T, ncols: i32, neg_inf: T, zero: A,
        max: impl Fn(T, T) -> T, shfl: impl Fn(T, u32) -> T, expsub: impl Fn(T, T) -> T,
        acc: impl Fn(A, T, bool) -> A, shfl_acc: impl Fn(A, u32) -> A, add_acc: impl Fn(A, A, bool) -> A,
        inv: impl Fn(A) -> I, scale: impl Fn(T, I) -> T,
    ) {
        let row = (thread::blockDim_x() as i32).wrapping_mul(thread::blockIdx_x() as i32).wrapping_add(thread::threadIdx_x() as i32);
        let block_size = thread::blockDim_y() as i32;
        let tid = thread::threadIdx_y() as i32;
        let base = row.wrapping_mul(ncols);

        let mut max_val = neg_inf;
        let mut col = tid;
        while col < ncols {
            max_val = max(max_val, *x.offset(base.wrapping_add(col) as isize));
            col = col.wrapping_add(block_size);
        }
        let mut m = 16;
        while m > 0 {
            max_val = max(max_val, shfl(max_val, m));
            m >>= 1;
        }

        // nvcc peels one iteration when the trip count is odd, then unrolls by two; the flag
        // tells `acc` which unrolled copy (and so which NaN-keeping operand order) runs.
        let trips: i64 = if tid < ncols { (ncols as i64 - tid as i64 + block_size as i64 - 1) / block_size as i64 } else { 0 };
        let mut tmp = zero;
        let mut col = tid;
        let mut j: i64 = 0;
        while col < ncols {
            let i = base.wrapping_add(col) as isize;
            let val = expsub(*x.offset(i), max_val);
            tmp = acc(tmp, val, (j + trips) % 2 == 0);
            *dst.offset(i) = val;
            col = col.wrapping_add(block_size);
            j += 1;
        }
        let mut m = 16;
        while m > 0 {
            tmp = add_acc(tmp, shfl_acc(tmp, m), m == 16);
            m >>= 1;
        }
        let inv_tmp = inv(tmp);
        let mut col = tid;
        while col < ncols {
            let i = base.wrapping_add(col) as isize;
            *dst.offset(i) = scale(*dst.offset(i), inv_tmp);
            col = col.wrapping_add(block_size);
        }
    }

    #[inline(always)]
    fn shfl16(v: u16, m: u32) -> u16 {
        let w = (v as u32) | ((v as u32) << 16);
        warp::shuffle_xor_sync(FULL, w, m) as u16
    }
    #[inline(always)]
    fn shfl32(v: f32, m: u32) -> f32 {
        warp::shuffle_xor_f32_sync(FULL, v, m)
    }
    #[inline(always)]
    fn shfl64(v: f64, m: u32) -> f64 {
        warp::shuffle_xor_f64_sync(FULL, v, m)
    }

    #[kernel]
    pub unsafe fn softmax_bf16(src: *const u16, dst: *mut u16, n_cols: i32) {
        softmax(
            src, dst, n_cols, 0xFF80u16, 0.0f32,
            |a, b| bf16::max_nan_bf16(a, b), shfl16, |a, m| bf_exp(bf_sub(a, m)),
            |t: f32, v: u16, _| float::add_rn_f32(t, bf2f(v)), shfl32, |a, b, _| float::add_rn_f32(a, b),
            |t: f32| f2bf(float::rcp_rn_f32(t)), bf_mul,
        );
    }
    #[kernel]
    pub unsafe fn softmax_f16(src: *const u16, dst: *mut u16, n_cols: i32) {
        softmax(
            src, dst, n_cols, 0xFC00u16, 0.0f32,
            |a, b| f16::max_nan_f16(a, b), shfl16, |a, m| hf_exp(hf_sub(a, m)),
            |t: f32, v: u16, _| float::add_rn_f32(t, hf2f(v)), shfl32, |a, b, _| float::add_rn_f32(a, b),
            |t: f32| f2hf(float::rcp_rn_f32(t)), hf_mul,
        );
    }
    #[kernel]
    pub unsafe fn softmax_f32(src: *const f32, dst: *mut f32, n_cols: i32) {
        softmax(
            src, dst, n_cols, f32::NEG_INFINITY, 0.0f32,
            max32, shfl32, |a: f32, m: f32| (a - m).exp(),
            |a, b, _| float::add_rn_f32(a, b), shfl32, |a, b, _| float::add_rn_f32(a, b),
            |a| float::rcp_rn_f32(a), |v: f32, s: f32| float::mul_rn_f32(s, v),
        );
    }
    #[kernel]
    pub unsafe fn softmax_f64(src: *const f64, dst: *mut f64, n_cols: i32) {
        softmax(
            src, dst, n_cols, f64::NEG_INFINITY, 0.0f64,
            max64, shfl64, |a: f64, m: f64| sub64_keep_x(a, m).exp(),
            // exp loop: first unrolled copy keeps tmp, second keeps the new value;
            // butterfly: the xor-16 step keeps the own value, later steps the shuffled one.
            |t: f64, v: f64, keep_tmp: bool| if keep_tmp { add64_keep(v, t) } else { add64_keep(t, v) },
            shfl64,
            |own: f64, sh: f64, first: bool| if first { add64_keep(sh, own) } else { add64_keep(own, sh) },
            |a| float::rcp_rn_f64(a), |v: f64, s: f64| mul64_keep(v, s),
        );
    }

    // ------------------------------------------------------------------ rmsnorm / layernorm
    #[inline(always)]
    pub unsafe fn rmsnorm<T: Copy>(
        x: *const T, dst: *mut T, alpha: *const T, ncols: i32, block_size: i32, eps: f32,
        ld: impl Fn(T) -> f32, st: impl Fn(f32) -> T,
    ) {
        let row = (thread::blockIdx_x() as i32).wrapping_mul(thread::blockDim_y() as i32).wrapping_add(thread::threadIdx_y() as i32);
        let tid = thread::threadIdx_x() as i32;
        let base = row.wrapping_mul(ncols);
        let mut tmp = 0.0f32;
        let mut col = tid;
        while col < ncols {
            let xi = ld(*x.offset(base.wrapping_add(col) as isize));
            tmp = float::fma_rn_f32(xi, xi, tmp);
            col = col.wrapping_add(block_size);
        }
        tmp = warp_sum(tmp);
        if block_size > 32 {
            let s_sum = smem() as *mut f32;
            let warp_id = thread::threadIdx_x() / 32;
            let lane_id = thread::threadIdx_x() % 32;
            if lane_id == 0 {
                *s_sum.add(warp_id as usize) = tmp;
            }
            thread::sync_threads();
            tmp = *s_sum.add(lane_id as usize);
            tmp = warp_sum(tmp);
        }
        let mean = float::div_rn_f32(tmp, ncols as f32);
        let scale = float::rsqrt_approx_f32(float::add_rn_f32(mean, eps));
        let mut col = tid;
        if alpha.is_null() {
            while col < ncols {
                let i = base.wrapping_add(col) as isize;
                *dst.offset(i) = st(float::mul_rn_f32(scale, ld(*x.offset(i))));
                col = col.wrapping_add(block_size);
            }
        } else {
            while col < ncols {
                let a = ld(*alpha.offset(col as isize));
                let i = base.wrapping_add(col) as isize;
                *dst.offset(i) = st(float::mul_rn_f32(float::mul_rn_f32(scale, ld(*x.offset(i))), a));
                col = col.wrapping_add(block_size);
            }
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn layernorm<T: Copy>(
        x: *const T, dst: *mut T, alpha: *const T, beta: *const T, ncols: i32, block_size: i32, eps: f32,
        ld: impl Fn(T) -> f32, st: impl Fn(f32) -> T,
    ) {
        let row = (thread::blockIdx_x() as i32).wrapping_mul(thread::blockDim_y() as i32).wrapping_add(thread::threadIdx_y() as i32);
        let tid = thread::threadIdx_x() as i32;
        let base = row.wrapping_mul(ncols);
        let mut sx = 0.0f32;
        let mut sy = 0.0f32;
        let mut col = tid;
        while col < ncols {
            let xi = ld(*x.offset(base.wrapping_add(col) as isize));
            sx = float::add_rn_f32(sx, xi);
            sy = float::fma_rn_f32(xi, xi, sy);
            col = col.wrapping_add(block_size);
        }
        (sx, sy) = warp_sum2(sx, sy);
        if block_size > 32 {
            let s_sum = smem() as *mut f32;
            let warp_id = thread::threadIdx_x() / 32;
            let lane_id = thread::threadIdx_x() % 32;
            if lane_id == 0 {
                *s_sum.add(2 * warp_id as usize) = sx;
                *s_sum.add(2 * warp_id as usize + 1) = sy;
            }
            thread::sync_threads();
            sx = *s_sum.add(2 * lane_id as usize);
            sy = *s_sum.add(2 * lane_id as usize + 1);
            (sx, sy) = warp_sum2(sx, sy);
        }
        let n = ncols as f32;
        let mean = float::div_rn_f32(sx, n);
        let var = float::fma_rn_f32(-mean, mean, float::div_rn_f32(sy, n));
        let inv_std = float::rsqrt_approx_f32(float::add_rn_f32(var, eps));
        let lhs = |i: isize| float::mul_rn_f32(inv_std, float::add_rn_f32(ld(*x.offset(i)), -mean));
        let mut col = tid;
        if alpha.is_null() && beta.is_null() {
            while col < ncols {
                let i = base.wrapping_add(col) as isize;
                *dst.offset(i) = st(lhs(i));
                col = col.wrapping_add(block_size);
            }
        } else if alpha.is_null() {
            while col < ncols {
                let b = ld(*beta.offset(col as isize));
                let i = base.wrapping_add(col) as isize;
                let d = float::add_rn_f32(ld(*x.offset(i)), -mean);
                *dst.offset(i) = st(float::fma_rn_f32(inv_std, d, b));
                col = col.wrapping_add(block_size);
            }
        } else if beta.is_null() {
            while col < ncols {
                let a = ld(*alpha.offset(col as isize));
                let i = base.wrapping_add(col) as isize;
                *dst.offset(i) = st(float::mul_rn_f32(a, lhs(i)));
                col = col.wrapping_add(block_size);
            }
        } else {
            while col < ncols {
                let a = ld(*alpha.offset(col as isize));
                let b = ld(*beta.offset(col as isize));
                let i = base.wrapping_add(col) as isize;
                *dst.offset(i) = st(float::fma_rn_f32(a, lhs(i), b));
                col = col.wrapping_add(block_size);
            }
        }
    }

    #[kernel]
    pub unsafe fn rmsnorm_bf16(src: *const u16, dst: *mut u16, alpha: *const u16, n_cols: i32, block_size: i32, eps: f32) {
        rmsnorm(src, dst, alpha, n_cols, block_size, eps, bf2f, f2bf);
    }
    #[kernel]
    pub unsafe fn rmsnorm_f16(src: *const u16, dst: *mut u16, alpha: *const u16, n_cols: i32, block_size: i32, eps: f32) {
        rmsnorm(src, dst, alpha, n_cols, block_size, eps, hf2f, f2hf);
    }
    #[kernel]
    pub unsafe fn rmsnorm_f32(src: *const f32, dst: *mut f32, alpha: *const f32, n_cols: i32, block_size: i32, eps: f32) {
        rmsnorm(src, dst, alpha, n_cols, block_size, eps, |v: f32| v, |v: f32| v);
    }
    #[kernel]
    pub unsafe fn rmsnorm_f64(src: *const f64, dst: *mut f64, alpha: *const f64, n_cols: i32, block_size: i32, eps: f32) {
        rmsnorm(src, dst, alpha, n_cols, block_size, eps, |v: f64| v as f32, |v: f32| v as f64);
    }
    #[kernel]
    pub unsafe fn layernorm_bf16(src: *const u16, dst: *mut u16, alpha: *const u16, beta: *const u16, n_cols: i32, block_size: i32, eps: f32) {
        layernorm(src, dst, alpha, beta, n_cols, block_size, eps, bf2f, f2bf);
    }
    #[kernel]
    pub unsafe fn layernorm_f16(src: *const u16, dst: *mut u16, alpha: *const u16, beta: *const u16, n_cols: i32, block_size: i32, eps: f32) {
        layernorm(src, dst, alpha, beta, n_cols, block_size, eps, hf2f, f2hf);
    }
    #[kernel]
    pub unsafe fn layernorm_f32(src: *const f32, dst: *mut f32, alpha: *const f32, beta: *const f32, n_cols: i32, block_size: i32, eps: f32) {
        layernorm(src, dst, alpha, beta, n_cols, block_size, eps, |v: f32| v, |v: f32| v);
    }
    #[kernel]
    pub unsafe fn layernorm_f64(src: *const f64, dst: *mut f64, alpha: *const f64, beta: *const f64, n_cols: i32, block_size: i32, eps: f32) {
        layernorm(src, dst, alpha, beta, n_cols, block_size, eps, |v: f64| v as f32, |v: f32| v as f64);
    }

    // ------------------------------------------------------------------ rope
    /// dst[i1] = c*x1 - s*x2 ; dst[i2] = s*x1 + c*x2, with nvcc/ptxas's contraction and the
    /// re-read of x1/x2 after the first store.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn rot<T: Copy>(
        src: *const T, dst: *mut T, i1: isize, i2: isize, c: T, s: T,
        mul: impl Fn(T, T) -> T, fma: impl Fn(T, T, T) -> T, neg: impl Fn(T) -> T,
    ) {
        let x1 = *src.offset(i1);
        let x2 = *src.offset(i2);
        *dst.offset(i1) = fma(c, x1, neg(mul(s, x2)));
        let x1 = *src.offset(i1);
        let x2 = *src.offset(i2);
        *dst.offset(i2) = fma(s, x1, mul(c, x2));
    }

    #[inline(always)]
    fn gidx() -> i32 {
        (thread::blockIdx_x() as i32).wrapping_mul(thread::blockDim_x() as i32).wrapping_add(thread::threadIdx_x() as i32)
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn ropei<T: Copy>(
        src: *const T, cos: *const T, sin: *const T, dst: *mut T, bh: u32, td: u32, stride_b: u32,
        mul: impl Fn(T, T) -> T, fma: impl Fn(T, T, T) -> T, neg: impl Fn(T) -> T,
    ) {
        let idx = gidx();
        let two = idx.wrapping_mul(2);
        if two as u32 >= bh.wrapping_mul(td) {
            return;
        }
        let mut rope_idx = (idx as u32) % (td / 2);
        if stride_b > 0 {
            let b_idx = (two as u32) / stride_b;
            rope_idx = rope_idx.wrapping_add(b_idx.wrapping_mul(td / 2));
        }
        let c = *cos.add(rope_idx as usize);
        let s = *sin.add(rope_idx as usize);
        let i1 = two as isize;
        rot(src, dst, i1, i1 + 1, c, s, mul, fma, neg);
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn rope<T: Copy>(
        src: *const T, cos: *const T, sin: *const T, dst: *mut T, bh: u32, td: u32, d: u32, stride_b: u32,
        mul: impl Fn(T, T) -> T, fma: impl Fn(T, T, T) -> T, neg: impl Fn(T) -> T,
    ) {
        let idx = gidx();
        let two = idx.wrapping_mul(2);
        if two as u32 >= bh.wrapping_mul(td) {
            return;
        }
        let u = idx as u32;
        let i_bh = u / (td / 2);
        let i_td = u.wrapping_sub((td / 2).wrapping_mul(i_bh));
        let i_t = i_td / (d / 2);
        let i_d = i_td.wrapping_sub((d / 2).wrapping_mul(i_t));
        let i1 = i_bh.wrapping_mul(td).wrapping_add(i_t.wrapping_mul(d)).wrapping_add(i_d);
        let i2 = i1.wrapping_add(d / 2);
        let mut i_cs = i_t.wrapping_mul(d / 2).wrapping_add(i_d);
        if stride_b > 0 {
            let b_idx = (two as u32) / stride_b;
            i_cs = i_cs.wrapping_add(b_idx.wrapping_mul(td / 2));
        }
        let c = *cos.add(i_cs as usize);
        let s = *sin.add(i_cs as usize);
        rot(src, dst, i1 as isize, i2 as isize, c, s, mul, fma, neg);
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn rope_thd<T: Copy>(
        src: *const T, cos: *const T, sin: *const T, dst: *mut T, b: u32, t: u32, h: u32, d: u32, stride_b: u32,
        mul: impl Fn(T, T) -> T, fma: impl Fn(T, T, T) -> T, neg: impl Fn(T) -> T,
    ) {
        let idx = gidx();
        let two = idx.wrapping_mul(2);
        if two as u32 >= b.wrapping_mul(t).wrapping_mul(h).wrapping_mul(d) {
            return;
        }
        let u = idx as u32;
        let i_bth = u / (d / 2);
        let i_d = u.wrapping_sub((d / 2).wrapping_mul(i_bth));
        let i_t = (i_bth / h) % t;
        let i1 = i_bth.wrapping_mul(d).wrapping_add(i_d);
        let i2 = i1.wrapping_add(d / 2);
        let mut i_cs = i_t.wrapping_mul(d / 2).wrapping_add(i_d);
        if stride_b > 0 {
            let b_idx = (two as u32) / stride_b;
            i_cs = i_cs.wrapping_add(b_idx.wrapping_mul(t.wrapping_mul(d) / 2));
        }
        let c = *cos.add(i_cs as usize);
        let s = *sin.add(i_cs as usize);
        rot(src, dst, i1 as isize, i2 as isize, c, s, mul, fma, neg);
    }

    #[inline(always)]
    fn neg16(x: u16) -> u16 {
        x ^ 0x8000
    }

    #[kernel]
    pub unsafe fn rope_i_bf16(src: *const u16, cos: *const u16, sin: *const u16, dst: *mut u16, bh: u32, td: u32, stride_b: u32) {
        ropei(src, cos, sin, dst, bh, td, stride_b, bf_mul, bf_fma, neg16);
    }
    #[kernel]
    pub unsafe fn rope_bf16(src: *const u16, cos: *const u16, sin: *const u16, dst: *mut u16, bh: u32, td: u32, d: u32, stride_b: u32) {
        rope(src, cos, sin, dst, bh, td, d, stride_b, bf_mul, bf_fma, neg16);
    }
    #[kernel]
    pub unsafe fn rope_thd_bf16(src: *const u16, cos: *const u16, sin: *const u16, dst: *mut u16, b: u32, t: u32, h: u32, d: u32, stride_b: u32) {
        rope_thd(src, cos, sin, dst, b, t, h, d, stride_b, bf_mul, bf_fma, neg16);
    }
    #[kernel]
    pub unsafe fn rope_i_f16(src: *const u16, cos: *const u16, sin: *const u16, dst: *mut u16, bh: u32, td: u32, stride_b: u32) {
        ropei(src, cos, sin, dst, bh, td, stride_b, hf_mul, hf_fma, neg16);
    }
    #[kernel]
    pub unsafe fn rope_f16(src: *const u16, cos: *const u16, sin: *const u16, dst: *mut u16, bh: u32, td: u32, d: u32, stride_b: u32) {
        rope(src, cos, sin, dst, bh, td, d, stride_b, hf_mul, hf_fma, neg16);
    }
    #[kernel]
    pub unsafe fn rope_thd_f16(src: *const u16, cos: *const u16, sin: *const u16, dst: *mut u16, b: u32, t: u32, h: u32, d: u32, stride_b: u32) {
        rope_thd(src, cos, sin, dst, b, t, h, d, stride_b, hf_mul, hf_fma, neg16);
    }
    #[kernel]
    pub unsafe fn rope_i_f32(src: *const f32, cos: *const f32, sin: *const f32, dst: *mut f32, bh: u32, td: u32, stride_b: u32) {
        ropei(src, cos, sin, dst, bh, td, stride_b, |a, b| float::mul_rn_f32(a, b), |a, b, c| float::fma_rn_f32(a, b, c), |v: f32| -v);
    }
    #[kernel]
    pub unsafe fn rope_f32(src: *const f32, cos: *const f32, sin: *const f32, dst: *mut f32, bh: u32, td: u32, d: u32, stride_b: u32) {
        rope(src, cos, sin, dst, bh, td, d, stride_b, |a, b| float::mul_rn_f32(a, b), |a, b, c| float::fma_rn_f32(a, b, c), |v: f32| -v);
    }
    #[kernel]
    pub unsafe fn rope_thd_f32(src: *const f32, cos: *const f32, sin: *const f32, dst: *mut f32, b: u32, t: u32, h: u32, d: u32, stride_b: u32) {
        rope_thd(src, cos, sin, dst, b, t, h, d, stride_b, |a, b| float::mul_rn_f32(a, b), |a, b, c| float::fma_rn_f32(a, b, c), |v: f32| -v);
    }
    #[kernel]
    pub unsafe fn rope_i_f64(src: *const f64, cos: *const f64, sin: *const f64, dst: *mut f64, bh: u32, td: u32, stride_b: u32) {
        ropei(src, cos, sin, dst, bh, td, stride_b, |a, b| float::mul_rn_f64(a, b), |a, b, c| float::fma_rn_f64(a, b, c), |v: f64| -v);
    }
    #[kernel]
    pub unsafe fn rope_f64(src: *const f64, cos: *const f64, sin: *const f64, dst: *mut f64, bh: u32, td: u32, d: u32, stride_b: u32) {
        rope(src, cos, sin, dst, bh, td, d, stride_b, |a, b| float::mul_rn_f64(a, b), |a, b, c| float::fma_rn_f64(a, b, c), |v: f64| -v);
    }
    #[kernel]
    pub unsafe fn rope_thd_f64(src: *const f64, cos: *const f64, sin: *const f64, dst: *mut f64, b: u32, t: u32, h: u32, d: u32, stride_b: u32) {
        rope_thd(src, cos, sin, dst, b, t, h, d, stride_b, |a, b| float::mul_rn_f64(a, b), |a, b, c| float::fma_rn_f64(a, b, c), |v: f64| -v);
    }
}

mod gate;

fn main() {
    std::process::exit(if gate::run() { 0 } else { 1 });
}
