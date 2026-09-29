//! candle-kernels `cast.cu` in cuda-oxide: same 58 entry names, same raw-pointer ABI,
//! bit-identical output (checked by the host `main` against candle's own nvcc PTX).
//!
//! Every float conversion is the exact PTX `cvt` the reference emits (read off its PTX per entry),
//! issued through `ptx_asm!`, so rounding mode, saturation and NaN handling are the hardware's own:
//! - float -> u8 is `cvt.rzi.u32.f{32,64}` (saturating to u32, NaN -> 0) then truncation to the
//!   low byte, NOT a saturating u8 conversion (C `(uint8_t)f` as nvcc lowers it);
//! - u32/f64 -> bf16/f16 are single-rounding `cvt.rn.{bf16,f16}.{u32,f64}` (no f32 double rounding);
//! - bf16 <-> f16, and every fp8 path, go through f32 (CAST_THROUGH_OP / F8E4M3_TO_FLOAT);
//! - into fp8 is `cvt.rn.satfinite.e4m3x2.f32` of `(float)x` (f64 -> f32 rounds first);
//! - u8 -> bf16/f16 go via s32, u8 -> f32/f64 via u16 (all exact);
//! - integer casts truncate / zero- / sign-extend as C does.
use cuda_device::{kernel, ptx_asm, thread};
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

    /// The shared loop of every cast kernel (`cast_`, `cast_fp8_`, `cast_fp8_into_`, `cast_through`).
    #[inline(always)]
    pub unsafe fn cast_loop<S: Copy, T: Copy, F: Fn(S) -> T>(
        numel: usize, num_dims: usize, info: *const usize, inp: *const S, out: *mut T, op: F,
    ) {
        let dims = info;
        let strides = info.wrapping_add(num_dims);
        let step = thread::blockDim_x().wrapping_mul(thread::gridDim_x());
        let mut i: u32 = thread::blockIdx_x().wrapping_mul(thread::blockDim_x()).wrapping_add(thread::threadIdx_x());
        if info.is_null() || is_contiguous(num_dims, dims, strides) {
            while (i as usize) < numel {
                *out.add(i as usize) = op(*inp.add(i as usize));
                i = i.wrapping_add(step);
            }
        } else {
            while (i as usize) < numel {
                let s = get_strided_index(i, num_dims, dims, strides);
                *out.add(i as usize) = op(*inp.add(s as usize));
                i = i.wrapping_add(step);
            }
        }
    }

    // ---- exact PTX conversions, one instruction each (as in the reference PTX) ----
    #[inline(always)]
    fn bf16_f32(x: u16) -> f32 { let r: f32; unsafe { ptx_asm!("cvt.f32.bf16 %0, %1;", out("=f") r, in("h") x, options(register_only)); } r }
    #[inline(always)]
    fn f16_f32(x: u16) -> f32 { let r: f32; unsafe { ptx_asm!("cvt.f32.f16 %0, %1;", out("=f") r, in("h") x, options(register_only)); } r }
    #[inline(always)]
    fn f32_bf16(x: f32) -> u16 { let r: u16; unsafe { ptx_asm!("cvt.rn.bf16.f32 %0, %1;", out("=h") r, in("f") x, options(register_only)); } r }
    #[inline(always)]
    fn f32_f16(x: f32) -> u16 { let r: u16; unsafe { ptx_asm!("cvt.rn.f16.f32 %0, %1;", out("=h") r, in("f") x, options(register_only)); } r }
    #[inline(always)]
    fn bf16_u32(x: u16) -> u32 { let r: u32; unsafe { ptx_asm!("cvt.rzi.u32.bf16 %0, %1;", out("=r") r, in("h") x, options(register_only)); } r }
    #[inline(always)]
    fn f16_u32(x: u16) -> u32 { let r: u32; unsafe { ptx_asm!("cvt.rzi.u32.f16 %0, %1;", out("=r") r, in("h") x, options(register_only)); } r }
    #[inline(always)]
    fn s32_bf16(x: i32) -> u16 { let r: u16; unsafe { ptx_asm!("cvt.rn.bf16.s32 %0, %1;", out("=h") r, in("r") x, options(register_only)); } r }
    #[inline(always)]
    fn u32_bf16(x: u32) -> u16 { let r: u16; unsafe { ptx_asm!("cvt.rn.bf16.u32 %0, %1;", out("=h") r, in("r") x, options(register_only)); } r }
    #[inline(always)]
    fn s32_f16(x: i32) -> u16 { let r: u16; unsafe { ptx_asm!("cvt.rn.f16.s32 %0, %1;", out("=h") r, in("r") x, options(register_only)); } r }
    #[inline(always)]
    fn u32_f16(x: u32) -> u16 { let r: u16; unsafe { ptx_asm!("cvt.rn.f16.u32 %0, %1;", out("=h") r, in("r") x, options(register_only)); } r }
    #[inline(always)]
    fn f64_bf16(x: f64) -> u16 { let r: u16; unsafe { ptx_asm!("cvt.rn.bf16.f64 %0, %1;", out("=h") r, in("d") x, options(register_only)); } r }
    #[inline(always)]
    fn f64_f16(x: f64) -> u16 { let r: u16; unsafe { ptx_asm!("cvt.rn.f16.f64 %0, %1;", out("=h") r, in("d") x, options(register_only)); } r }
    #[inline(always)]
    fn f32_f64(x: f32) -> f64 { let r: f64; unsafe { ptx_asm!("cvt.f64.f32 %0, %1;", out("=d") r, in("f") x, options(register_only)); } r }
    #[inline(always)]
    fn f64_f32(x: f64) -> f32 { let r: f32; unsafe { ptx_asm!("cvt.rn.f32.f64 %0, %1;", out("=f") r, in("d") x, options(register_only)); } r }
    #[inline(always)]
    fn f32_u32(x: f32) -> u32 { let r: u32; unsafe { ptx_asm!("cvt.rzi.u32.f32 %0, %1;", out("=r") r, in("f") x, options(register_only)); } r }
    #[inline(always)]
    fn f32_s32(x: f32) -> i32 { let r: i32; unsafe { ptx_asm!("cvt.rzi.s32.f32 %0, %1;", out("=r") r, in("f") x, options(register_only)); } r }
    #[inline(always)]
    fn f32_s64(x: f32) -> i64 { let r: i64; unsafe { ptx_asm!("cvt.rzi.s64.f32 %0, %1;", out("=l") r, in("f") x, options(register_only)); } r }
    #[inline(always)]
    fn f64_u32(x: f64) -> u32 { let r: u32; unsafe { ptx_asm!("cvt.rzi.u32.f64 %0, %1;", out("=r") r, in("d") x, options(register_only)); } r }
    #[inline(always)]
    fn f64_s64(x: f64) -> i64 { let r: i64; unsafe { ptx_asm!("cvt.rzi.s64.f64 %0, %1;", out("=l") r, in("d") x, options(register_only)); } r }
    #[inline(always)]
    fn u16_f32(x: u16) -> f32 { let r: f32; unsafe { ptx_asm!("cvt.rn.f32.u16 %0, %1;", out("=f") r, in("h") x, options(register_only)); } r }
    #[inline(always)]
    fn u32_f32(x: u32) -> f32 { let r: f32; unsafe { ptx_asm!("cvt.rn.f32.u32 %0, %1;", out("=f") r, in("r") x, options(register_only)); } r }
    #[inline(always)]
    fn s32_f32(x: i32) -> f32 { let r: f32; unsafe { ptx_asm!("cvt.rn.f32.s32 %0, %1;", out("=f") r, in("r") x, options(register_only)); } r }
    #[inline(always)]
    fn s64_f32(x: i64) -> f32 { let r: f32; unsafe { ptx_asm!("cvt.rn.f32.s64 %0, %1;", out("=f") r, in("l") x, options(register_only)); } r }
    #[inline(always)]
    fn u16_f64(x: u16) -> f64 { let r: f64; unsafe { ptx_asm!("cvt.rn.f64.u16 %0, %1;", out("=d") r, in("h") x, options(register_only)); } r }
    #[inline(always)]
    fn u32_f64(x: u32) -> f64 { let r: f64; unsafe { ptx_asm!("cvt.rn.f64.u32 %0, %1;", out("=d") r, in("r") x, options(register_only)); } r }
    #[inline(always)]
    fn s64_f64(x: i64) -> f64 { let r: f64; unsafe { ptx_asm!("cvt.rn.f64.s64 %0, %1;", out("=d") r, in("l") x, options(register_only)); } r }
    /// F8E4M3_TO_FLOAT: e4m3 -> f16 (`cvt.rn.f16x2.e4m3x2`, low half) -> f32.
    #[inline(always)]
    fn e4m3_f32(x: u8) -> f32 {
        let h: u32;
        unsafe { ptx_asm!("cvt.rn.f16x2.e4m3x2 %0, %1;", out("=r") h, in("h") x as u16, options(register_only)); }
        f16_f32(h as u16)
    }
    /// `__nv_fp8_e4m3(float)`: satfinite round-to-nearest, value in the low byte (high lane 0).
    #[inline(always)]
    fn f32_e4m3(x: f32) -> u8 {
        let r: u16;
        unsafe { ptx_asm!("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;", out("=h") r, in("f") 0.0f32, in("f") x, options(register_only)); }
        r as u8
    }

    #[kernel]
    pub unsafe fn cast_bf16_bf16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        cast_loop(numel, num_dims, info, inp, out, |x: u16| x);
    }

    #[kernel]
    pub unsafe fn cast_f8_e4m3_f8_e4m3(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u8) {
        cast_loop(numel, num_dims, info, inp, out, |x: u8| x);
    }

    #[kernel]
    pub unsafe fn cast_bf16_u32(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u32) {
        cast_loop(numel, num_dims, info, inp, out, |x: u16| bf16_u32(x));
    }

    #[kernel]
    pub unsafe fn cast_bf16_f32(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut f32) {
        cast_loop(numel, num_dims, info, inp, out, |x: u16| bf16_f32(x));
    }

    #[kernel]
    pub unsafe fn cast_bf16_f64(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut f64) {
        cast_loop(numel, num_dims, info, inp, out, |x: u16| f32_f64(bf16_f32(x)));
    }

    #[kernel]
    pub unsafe fn cast_u8_bf16(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u16) {
        cast_loop(numel, num_dims, info, inp, out, |x: u8| s32_bf16(x as i32));
    }

    #[kernel]
    pub unsafe fn cast_u32_bf16(numel: usize, num_dims: usize, info: *const usize, inp: *const u32, out: *mut u16) {
        cast_loop(numel, num_dims, info, inp, out, |x: u32| u32_bf16(x));
    }

    #[kernel]
    pub unsafe fn cast_f32_bf16(numel: usize, num_dims: usize, info: *const usize, inp: *const f32, out: *mut u16) {
        cast_loop(numel, num_dims, info, inp, out, |x: f32| f32_bf16(x));
    }

    #[kernel]
    pub unsafe fn cast_f64_bf16(numel: usize, num_dims: usize, info: *const usize, inp: *const f64, out: *mut u16) {
        cast_loop(numel, num_dims, info, inp, out, |x: f64| f64_bf16(x));
    }

    #[kernel]
    pub unsafe fn cast_bf16_u8(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u8) {
        cast_loop(numel, num_dims, info, inp, out, |x: u16| f32_u32(bf16_f32(x)) as u8);
    }

    #[kernel]
    pub unsafe fn cast_bf16_f16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        cast_loop(numel, num_dims, info, inp, out, |x: u16| f32_f16(bf16_f32(x)));
    }

    #[kernel]
    pub unsafe fn cast_f16_bf16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        cast_loop(numel, num_dims, info, inp, out, |x: u16| f32_bf16(f16_f32(x)));
    }

    #[kernel]
    pub unsafe fn cast_f8_e4m3_f32(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut f32) {
        cast_loop(numel, num_dims, info, inp, out, |x: u8| e4m3_f32(x));
    }

    #[kernel]
    pub unsafe fn cast_f32_f8_e4m3(numel: usize, num_dims: usize, info: *const usize, inp: *const f32, out: *mut u8) {
        cast_loop(numel, num_dims, info, inp, out, |x: f32| f32_e4m3(x));
    }

    #[kernel]
    pub unsafe fn cast_f8_e4m3_u8(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u8) {
        cast_loop(numel, num_dims, info, inp, out, |x: u8| f32_u32(e4m3_f32(x)) as u8);
    }

    #[kernel]
    pub unsafe fn cast_f8_e4m3_f16(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u16) {
        cast_loop(numel, num_dims, info, inp, out, |x: u8| f32_f16(e4m3_f32(x)));
    }

    #[kernel]
    pub unsafe fn cast_f8_e4m3_f64(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut f64) {
        cast_loop(numel, num_dims, info, inp, out, |x: u8| f32_f64(e4m3_f32(x)));
    }

    #[kernel]
    pub unsafe fn cast_f16_f8_e4m3(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u8) {
        cast_loop(numel, num_dims, info, inp, out, |x: u16| f32_e4m3(f16_f32(x)));
    }

    #[kernel]
    pub unsafe fn cast_f64_f8_e4m3(numel: usize, num_dims: usize, info: *const usize, inp: *const f64, out: *mut u8) {
        cast_loop(numel, num_dims, info, inp, out, |x: f64| f32_e4m3(f64_f32(x)));
    }

    #[kernel]
    pub unsafe fn cast_u8_f8_e4m3(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u8) {
        cast_loop(numel, num_dims, info, inp, out, |x: u8| f32_e4m3(u16_f32(x as u16)));
    }

    #[kernel]
    pub unsafe fn cast_i32_f8_e4m3(numel: usize, num_dims: usize, info: *const usize, inp: *const i32, out: *mut u8) {
        cast_loop(numel, num_dims, info, inp, out, |x: i32| f32_e4m3(s32_f32(x)));
    }

    #[kernel]
    pub unsafe fn cast_f8_e4m3_i32(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut i32) {
        cast_loop(numel, num_dims, info, inp, out, |x: u8| f32_s32(e4m3_f32(x)));
    }

    #[kernel]
    pub unsafe fn cast_f8_e4m3_bf16(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u16) {
        cast_loop(numel, num_dims, info, inp, out, |x: u8| f32_bf16(e4m3_f32(x)));
    }

    #[kernel]
    pub unsafe fn cast_bf16_f8_e4m3(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u8) {
        cast_loop(numel, num_dims, info, inp, out, |x: u16| f32_e4m3(bf16_f32(x)));
    }

    #[kernel]
    pub unsafe fn cast_f16_f16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16) {
        cast_loop(numel, num_dims, info, inp, out, |x: u16| x);
    }

    #[kernel]
    pub unsafe fn cast_f16_u8(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u8) {
        cast_loop(numel, num_dims, info, inp, out, |x: u16| f32_u32(f16_f32(x)) as u8);
    }

    #[kernel]
    pub unsafe fn cast_f16_u32(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u32) {
        cast_loop(numel, num_dims, info, inp, out, |x: u16| f16_u32(x));
    }

    #[kernel]
    pub unsafe fn cast_f16_f32(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut f32) {
        cast_loop(numel, num_dims, info, inp, out, |x: u16| f16_f32(x));
    }

    #[kernel]
    pub unsafe fn cast_f16_f64(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut f64) {
        cast_loop(numel, num_dims, info, inp, out, |x: u16| f32_f64(f16_f32(x)));
    }

    #[kernel]
    pub unsafe fn cast_u8_f16(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u16) {
        cast_loop(numel, num_dims, info, inp, out, |x: u8| s32_f16(x as i32));
    }

    #[kernel]
    pub unsafe fn cast_u32_f16(numel: usize, num_dims: usize, info: *const usize, inp: *const u32, out: *mut u16) {
        cast_loop(numel, num_dims, info, inp, out, |x: u32| u32_f16(x));
    }

    #[kernel]
    pub unsafe fn cast_f32_f16(numel: usize, num_dims: usize, info: *const usize, inp: *const f32, out: *mut u16) {
        cast_loop(numel, num_dims, info, inp, out, |x: f32| f32_f16(x));
    }

    #[kernel]
    pub unsafe fn cast_f64_f16(numel: usize, num_dims: usize, info: *const usize, inp: *const f64, out: *mut u16) {
        cast_loop(numel, num_dims, info, inp, out, |x: f64| f64_f16(x));
    }

    #[kernel]
    pub unsafe fn cast_u32_u32(numel: usize, num_dims: usize, info: *const usize, inp: *const u32, out: *mut u32) {
        cast_loop(numel, num_dims, info, inp, out, |x: u32| x);
    }

    #[kernel]
    pub unsafe fn cast_u32_u8(numel: usize, num_dims: usize, info: *const usize, inp: *const u32, out: *mut u8) {
        cast_loop(numel, num_dims, info, inp, out, |x: u32| x as u8);
    }

    #[kernel]
    pub unsafe fn cast_u32_i64(numel: usize, num_dims: usize, info: *const usize, inp: *const u32, out: *mut i64) {
        cast_loop(numel, num_dims, info, inp, out, |x: u32| x as i64);
    }

    #[kernel]
    pub unsafe fn cast_u32_f32(numel: usize, num_dims: usize, info: *const usize, inp: *const u32, out: *mut f32) {
        cast_loop(numel, num_dims, info, inp, out, |x: u32| u32_f32(x));
    }

    #[kernel]
    pub unsafe fn cast_u32_f64(numel: usize, num_dims: usize, info: *const usize, inp: *const u32, out: *mut f64) {
        cast_loop(numel, num_dims, info, inp, out, |x: u32| u32_f64(x));
    }

    #[kernel]
    pub unsafe fn cast_u8_u32(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u32) {
        cast_loop(numel, num_dims, info, inp, out, |x: u8| x as u32);
    }

    #[kernel]
    pub unsafe fn cast_u8_u8(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u8) {
        cast_loop(numel, num_dims, info, inp, out, |x: u8| x);
    }

    #[kernel]
    pub unsafe fn cast_u8_i64(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut i64) {
        cast_loop(numel, num_dims, info, inp, out, |x: u8| x as i64);
    }

    #[kernel]
    pub unsafe fn cast_u8_f32(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut f32) {
        cast_loop(numel, num_dims, info, inp, out, |x: u8| u16_f32(x as u16));
    }

    #[kernel]
    pub unsafe fn cast_u8_f64(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut f64) {
        cast_loop(numel, num_dims, info, inp, out, |x: u8| u16_f64(x as u16));
    }

    #[kernel]
    pub unsafe fn cast_i64_u32(numel: usize, num_dims: usize, info: *const usize, inp: *const i64, out: *mut u32) {
        cast_loop(numel, num_dims, info, inp, out, |x: i64| x as u32);
    }

    #[kernel]
    pub unsafe fn cast_i64_u8(numel: usize, num_dims: usize, info: *const usize, inp: *const i64, out: *mut u8) {
        cast_loop(numel, num_dims, info, inp, out, |x: i64| x as u8);
    }

    #[kernel]
    pub unsafe fn cast_i64_i64(numel: usize, num_dims: usize, info: *const usize, inp: *const i64, out: *mut i64) {
        cast_loop(numel, num_dims, info, inp, out, |x: i64| x);
    }

    #[kernel]
    pub unsafe fn cast_i64_f32(numel: usize, num_dims: usize, info: *const usize, inp: *const i64, out: *mut f32) {
        cast_loop(numel, num_dims, info, inp, out, |x: i64| s64_f32(x));
    }

    #[kernel]
    pub unsafe fn cast_i64_f64(numel: usize, num_dims: usize, info: *const usize, inp: *const i64, out: *mut f64) {
        cast_loop(numel, num_dims, info, inp, out, |x: i64| s64_f64(x));
    }

    #[kernel]
    pub unsafe fn cast_f32_u8(numel: usize, num_dims: usize, info: *const usize, inp: *const f32, out: *mut u8) {
        cast_loop(numel, num_dims, info, inp, out, |x: f32| f32_u32(x) as u8);
    }

    #[kernel]
    pub unsafe fn cast_f32_u32(numel: usize, num_dims: usize, info: *const usize, inp: *const f32, out: *mut u32) {
        cast_loop(numel, num_dims, info, inp, out, |x: f32| f32_u32(x));
    }

    #[kernel]
    pub unsafe fn cast_f32_i64(numel: usize, num_dims: usize, info: *const usize, inp: *const f32, out: *mut i64) {
        cast_loop(numel, num_dims, info, inp, out, |x: f32| f32_s64(x));
    }

    #[kernel]
    pub unsafe fn cast_f32_f32(numel: usize, num_dims: usize, info: *const usize, inp: *const f32, out: *mut f32) {
        cast_loop(numel, num_dims, info, inp, out, |x: f32| x);
    }

    #[kernel]
    pub unsafe fn cast_f32_f64(numel: usize, num_dims: usize, info: *const usize, inp: *const f32, out: *mut f64) {
        cast_loop(numel, num_dims, info, inp, out, |x: f32| f32_f64(x));
    }

    #[kernel]
    pub unsafe fn cast_f64_u8(numel: usize, num_dims: usize, info: *const usize, inp: *const f64, out: *mut u8) {
        cast_loop(numel, num_dims, info, inp, out, |x: f64| f64_u32(x) as u8);
    }

    #[kernel]
    pub unsafe fn cast_f64_u32(numel: usize, num_dims: usize, info: *const usize, inp: *const f64, out: *mut u32) {
        cast_loop(numel, num_dims, info, inp, out, |x: f64| f64_u32(x));
    }

    #[kernel]
    pub unsafe fn cast_f64_i64(numel: usize, num_dims: usize, info: *const usize, inp: *const f64, out: *mut i64) {
        cast_loop(numel, num_dims, info, inp, out, |x: f64| f64_s64(x));
    }

    #[kernel]
    pub unsafe fn cast_f64_f32(numel: usize, num_dims: usize, info: *const usize, inp: *const f64, out: *mut f32) {
        cast_loop(numel, num_dims, info, inp, out, |x: f64| f64_f32(x));
    }

    #[kernel]
    pub unsafe fn cast_f64_f64(numel: usize, num_dims: usize, info: *const usize, inp: *const f64, out: *mut f64) {
        cast_loop(numel, num_dims, info, inp, out, |x: f64| x);
    }

}

// ---------------------------------------------------------------------------------------------
// Differential gate against candle's nvcc-built cast.ptx.
use kdiff::{Arg, Harness, Rng, Tally, as_bytes, contiguous, layout_info};

const ENTRIES: [(&str, &str, usize); 58] = [
    ("cast_bf16_bf16", "bf16", 2),
    ("cast_f8_e4m3_f8_e4m3", "f8_e4m3", 1),
    ("cast_bf16_u32", "bf16", 4),
    ("cast_bf16_f32", "bf16", 4),
    ("cast_bf16_f64", "bf16", 8),
    ("cast_u8_bf16", "u8", 2),
    ("cast_u32_bf16", "u32", 2),
    ("cast_f32_bf16", "f32", 2),
    ("cast_f64_bf16", "f64", 2),
    ("cast_bf16_u8", "bf16", 1),
    ("cast_bf16_f16", "bf16", 2),
    ("cast_f16_bf16", "f16", 2),
    ("cast_f8_e4m3_f32", "f8_e4m3", 4),
    ("cast_f32_f8_e4m3", "f32", 1),
    ("cast_f8_e4m3_u8", "f8_e4m3", 1),
    ("cast_f8_e4m3_f16", "f8_e4m3", 2),
    ("cast_f8_e4m3_f64", "f8_e4m3", 8),
    ("cast_f16_f8_e4m3", "f16", 1),
    ("cast_f64_f8_e4m3", "f64", 1),
    ("cast_u8_f8_e4m3", "u8", 1),
    ("cast_i32_f8_e4m3", "i32", 1),
    ("cast_f8_e4m3_i32", "f8_e4m3", 4),
    ("cast_f8_e4m3_bf16", "f8_e4m3", 2),
    ("cast_bf16_f8_e4m3", "bf16", 1),
    ("cast_f16_f16", "f16", 2),
    ("cast_f16_u8", "f16", 1),
    ("cast_f16_u32", "f16", 4),
    ("cast_f16_f32", "f16", 4),
    ("cast_f16_f64", "f16", 8),
    ("cast_u8_f16", "u8", 2),
    ("cast_u32_f16", "u32", 2),
    ("cast_f32_f16", "f32", 2),
    ("cast_f64_f16", "f64", 2),
    ("cast_u32_u32", "u32", 4),
    ("cast_u32_u8", "u32", 1),
    ("cast_u32_i64", "u32", 8),
    ("cast_u32_f32", "u32", 4),
    ("cast_u32_f64", "u32", 8),
    ("cast_u8_u32", "u8", 4),
    ("cast_u8_u8", "u8", 1),
    ("cast_u8_i64", "u8", 8),
    ("cast_u8_f32", "u8", 4),
    ("cast_u8_f64", "u8", 8),
    ("cast_i64_u32", "i64", 4),
    ("cast_i64_u8", "i64", 1),
    ("cast_i64_i64", "i64", 8),
    ("cast_i64_f32", "i64", 4),
    ("cast_i64_f64", "i64", 8),
    ("cast_f32_u8", "f32", 1),
    ("cast_f32_u32", "f32", 4),
    ("cast_f32_i64", "f32", 8),
    ("cast_f32_f32", "f32", 4),
    ("cast_f32_f64", "f32", 8),
    ("cast_f64_u8", "f64", 1),
    ("cast_f64_u32", "f64", 4),
    ("cast_f64_i64", "f64", 8),
    ("cast_f64_f32", "f64", 4),
    ("cast_f64_f64", "f64", 8),
];

/// One f32 bit pattern from a mix of value classes that stress every conversion out of f32.
fn f32_bits(r: &mut Rng) -> u32 {
    let sign = ((r.next() & 1) as u32) << 31;
    let pm = |r: &mut Rng, b: u32| match r.next() % 3 { 0 => b, 1 => b.wrapping_add(1), _ => b.wrapping_sub(1) };
    match r.next() % 14 {
        0 => r.next() as u32,                                             // any pattern
        1 => r.f32s(1)[0].to_bits(),                                      // kdiff mix incl specials
        2 => { let b = ((r.next() as u32) << 16) | 0x8000; pm(r, b) }     // bf16 ties
        3 => { // f16 ties (normal and subnormal f16 range, and near overflow 65504..65536)
            let e = 103 + (r.next() % 42) as u32;                         // 2^-24 .. 2^17
            let b = sign | (e << 23) | ((r.next() as u32) & 0x7FE000) | 0x1000;
            pm(r, b)
        }
        4 => { // e4m3 ties / satfinite boundary (2^-10 .. 2^9)
            let e = 117 + (r.next() % 19) as u32;
            let b = sign | (e << 23) | ((r.next() as u32) & 0x700000) | 0x80000;
            pm(r, b)
        }
        5 => { // integer-ish around u8/u32/i32/i64 boundaries
            let bases: [f64; 12] = [0.0, 1.0, 255.0, 256.0, 257.0, 65535.0, 4294967295.0, 4294967296.0, 2147483648.0, 9.223372036854775807e18, 1.8446744073709552e19, 16777216.0];
            let v = bases[(r.next() % 12) as usize];
            let v = if r.next() & 1 == 1 { -v } else { v };
            pm(r, (v as f32).to_bits())
        }
        6 => { let v = (r.next() % 1000) as f32 / 4.0 - 20.0; v.to_bits() } // small, .25/.5/.75 fractions
        7 => 0x7F80_0000 | sign | (((r.next() as u32) & 0x7F_FFFF).max(1)), // NaN, quiet + signalling payloads
        8 => sign | ((r.next() as u32) & 0x7F_FFFF),                     // denormals / zero
        9 => [0u32, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7F7F_FFFF, 0xFF7F_FFFF, 0x7FC0_0000, 0x7F80_0001, 0x7FFF_FFFF, 0x0000_0001][(r.next() % 10) as usize],
        10 => { let v = (r.next() % 600) as f32; (v + [0.0f32, 0.5, 0.99, -0.5][(r.next() % 4) as usize]).to_bits() | sign }
        11 => { let e = 157 + (r.next() % 36) as u32; sign | (e << 23) | ((r.next() as u32) & 0x7F_FFFF) } // 2^30..2^65
        12 => { let b = 0x7F7F_0000 | ((r.next() as u32) & 0xFFFF); b | sign }                           // bf16 overflow edge
        _ => { let b = 0x43E0_0000 | ((r.next() as u32) & 0x3F_FFFF); b | sign }                         // 448..512 (fp8 max)
    }
}

/// One f64 bit pattern; also stresses single vs double rounding to f32/bf16/f16/e4m3.
fn f64_bits(r: &mut Rng) -> u64 {
    let pm = |r: &mut Rng, b: u64| match r.next() % 4 { 0 => b, 1 => b.wrapping_add(1), 2 => b.wrapping_sub(1), _ => b ^ (1 << (r.next() % 28)) };
    let widen = |b: u32| (f32::from_bits(b) as f64).to_bits();
    match r.next() % 12 {
        0 => r.next(),
        1 => r.f64s(1)[0].to_bits(),
        2 => { let b = widen(f32_bits(r)); if b & 0x7FF0_0000_0000_0000 == 0x7FF0_0000_0000_0000 { b } else { pm(r, b) } }
        3 => { let b = widen(f32_bits(r) & 0xFFFF_0000) | (1u64 << 44); pm(r, b) }          // bf16 ties in f64
        4 => { let b = widen(f32_bits(r) & 0xFFFF_E000) | (1u64 << 41); pm(r, b) }          // f16 ties
        5 => { let b = widen(f32_bits(r) & 0xFFF0_0000) | (1u64 << 48); pm(r, b) }          // e4m3 ties
        6 => { let b = widen(f32_bits(r)) | (1u64 << 28); pm(r, b) }                        // f32 ties
        7 => 0x7FF0_0000_0000_0000 | ((r.next() & 1) << 63) | ((r.next() & 0xF_FFFF_FFFF_FFFF).max(1) >> (r.next() % 52)).max(1), // NaN payloads (high/low bits only)
        8 => { let e = (r.next() % 2047) as u64; ((r.next() & 1) << 63) | (e << 52) | (r.next() & 0xF_FFFF_FFFF_FFFF) } // any exponent
        9 => { let e = 874 + (r.next() % 60) as u64; ((r.next() & 1) << 63) | (e << 52) | (r.next() & 0xF_FFFF_FFFF_FFFF) } // around f32/bf16 subnormal range
        10 => ((r.next() & 1) << 63) | (r.next() & 0xF_FFFF_FFFF_FFFF),                     // f64 denormals
        _ => { let e = 1150 + (r.next() % 30) as u64; ((r.next() & 1) << 63) | (e << 52) | (r.next() & 0xF_FFFF_FFFF_FFFF) } // ~2^127..2^156, f32 overflow
    }
}

/// Integer patterns that hit f32/bf16/f16 rounding ties (single vs double rounding) and ranges.
fn int_bits(r: &mut Rng) -> u64 {
    match r.next() % 6 {
        0 => r.next(),
        1 => r.next() % 70000,                                            // f16 overflow at 65520
        2 => { let k = 1 + r.next() % 62; let b = (r.next() >> k << k) | (1u64 << (k - 1)); match r.next() % 3 { 0 => b, 1 => b + 1, _ => b - 1 } }
        3 => { let k = 1 + r.next() % 31; let b = ((r.next() as u32) >> k << k) | (1u32 << (k - 1)); (match r.next() % 3 { 0 => b, 1 => b.wrapping_add(1), _ => b.wrapping_sub(1) }) as u64 }
        4 => (r.next() % 600).wrapping_sub(300),                        // small signed
        _ => [0u64, 1, u32::MAX as u64, i32::MIN as u32 as u64, i32::MAX as u64, i64::MIN as u64, i64::MAX as u64, u64::MAX, 65504, 65519, 65520, 16777217][(r.next() % 12) as usize],
    }
}

/// `n` source elements of class `src`, as raw bytes.
fn gen_src(r: &mut Rng, src: &str, n: usize) -> Vec<u8> {
    match src {
        "f32" => as_bytes(&(0..n).map(|_| f32_bits(r)).collect::<Vec<u32>>()),
        "f64" => as_bytes(&(0..n).map(|_| f64_bits(r)).collect::<Vec<u64>>()),
        "u32" | "i32" => as_bytes(&(0..n).map(|_| int_bits(r) as u32).collect::<Vec<u32>>()),
        "i64" => as_bytes(&(0..n).map(|_| int_bits(r)).collect::<Vec<u64>>()),
        "bf16" | "f16" => as_bytes(&r.b16s(n)),
        _ => r.bytes(n), // u8, f8_e4m3
    }
}

fn src_size(src: &str) -> usize {
    match src { "bf16" | "f16" => 2, "f32" | "u32" | "i32" => 4, "f64" | "i64" => 8, _ => 1 }
}

fn main() {
    let root = format!("{}/titan-engine/oxide-kernels", std::env::var("HOME").unwrap());
    let h = Harness::new(&format!("{root}/reference/candle/cast.ptx"), &format!("{root}/candle-cast/candle_cast.ptx"));
    let mut t = Tally::default();
    let mut rng = Rng(0xCA57);
    let layouts: Vec<(Vec<usize>, Vec<usize>)> = vec![
        (vec![4099], vec![1]),                     // odd contiguous
        (vec![7, 13, 5], contiguous(&[7, 13, 5])),
        (vec![13, 7], vec![1, 13]),                // transposed
        (vec![4, 1, 9], vec![9, 9, 1]),            // size-1 dim with odd stride
        (vec![6, 5], vec![0, 1]),                  // broadcast
        (vec![3, 4, 5], vec![40, 1, 8]),           // permuted
        (vec![1, 1], vec![3, 7]),                  // all size-1, non-unit strides (contiguous)
    ];
    for (name, src, dsz) in ENTRIES {
        let ssz = src_size(src);
        assert!(h.has(false, name), "{name} missing from reference");
        // Exhaustive: every bit pattern of an 8/16-bit source, contiguous.
        if ssz <= 2 {
            let n = 1usize << (8 * ssz);
            let inp: Vec<u8> = if ssz == 1 { (0..n).map(|v| v as u8).collect() } else { as_bytes(&(0..n).map(|v| v as u16).collect::<Vec<u16>>()) };
            for &(grid, block) in &[(((n as u32) + 255) / 256, 256), (3, 96)] {
                let out = rng.bytes(n * dsz);
                let args = [Arg::U64(n as u64), Arg::U64(1), Arg::Null, Arg::Buf(0), Arg::Buf(1)];
                let d = h.diff(name, (grid, 1, 1), (block, 1, 1), 0, &args, &[inp.clone(), out], &[1]);
                t.record(&format!("{name} exhaustive g{grid}x{block}"), &d);
            }
        }
        // Large random-class contiguous run (many value classes per launch).
        {
            let n = 1 << 16;
            let inp = gen_src(&mut rng, src, n);
            let out = rng.bytes(n * dsz);
            let args = [Arg::U64(n as u64), Arg::U64(1), Arg::Buf(2), Arg::Buf(0), Arg::Buf(1)];
            let d = h.diff(name, (40, 1, 1), (128, 1, 1), 0, &args, &[inp, out, layout_info(&[n], &[1])], &[1]);
            t.record(&format!("{name} random65536"), &d);
        }
        for (li, (dims, strides)) in layouts.iter().enumerate() {
            let numel: usize = dims.iter().product();
            let src_len = dims.iter().zip(strides).map(|(d, s)| (d - 1) * s).sum::<usize>() + 1;
            for &(grid, block) in &[(4u32, 64u32), (((numel as u32) + 255) / 256, 256), (1, 32), (1, 1)] {
                for with_info in [true, false] {
                    if !with_info && li > 1 { continue; }
                    let inp = gen_src(&mut rng, src, src_len.max(numel));
                    let out = rng.bytes(numel * dsz);
                    let info = if with_info { Arg::Buf(2) } else { Arg::Null };
                    let args = [Arg::U64(numel as u64), Arg::U64(dims.len() as u64), info, Arg::Buf(0), Arg::Buf(1)];
                    let bufs = vec![inp, out, layout_info(dims, strides)];
                    let d = h.diff(name, (grid, 1, 1), (block, 1, 1), 0, &args, &bufs, &[1]);
                    t.record(&format!("{name} layout{li} g{grid}x{block} info={with_info}"), &d);
                }
            }
        }
    }
    std::process::exit(if t.finish("cast") { 0 } else { 1 });
}
