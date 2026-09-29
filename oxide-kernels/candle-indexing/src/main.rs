//! candle-kernels `indexing.cu` in cuda-oxide: same 125 entry names, same raw-pointer ABI,
//! bit-identical output (checked by the host `main` against candle's own nvcc PTX).
//!
//! Semantics copied from the reference (source + PTX), not guessed:
//! - index_select: `unsigned int` dst/left/id/right/src indices; `src_i` is computed in u64 and
//!   TRUNCATED to u32 before the (optional) strided lookup; ids equal to the index type's max are
//!   "fill zero"; otherwise `assert(id < src_dim_size)` (an `__assertfail`, reproduced here).
//! - gather: u32 loop counter, all offsets in u64 (no truncation); max id -> zero; assert.
//! - index_add / scatter / scatter_add: one thread per (pre, post) pair, a sequential j loop, a plain
//!   (non-atomic) read-modify-write. Threads own disjoint destination rows, so colliding ids are
//!   deterministic (in j order). ids == max are skipped (`idx < max_value`), then assert.
//! - the fp8 index_add / scatter_add have NO sentinel and NO assert: `size_t idx = ids[j]`
//!   (sign-extended for signed index types); add is e4m3 -> f16 -> f32, f32 add, satfinite e4m3.
//! - signed index types widen by sign extension (C integer promotion then conversion to size_t).
//! - bf16/f16 `+=` is a single `add.bf16` / `add.f16` in the reference PTX; issued verbatim.
use cuda_device::{gpu_assert, kernel, ptx_asm, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// An index type: its candle `max_value<I>()` sentinel tests and its C widening to size_t.
    pub trait Idx: Copy {
        fn is_max(self) -> bool;
        fn lt_max(self) -> bool;
        fn widen(self) -> usize;
    }
    impl Idx for i64 {
        #[inline(always)] fn is_max(self) -> bool { self == i64::MAX }
        #[inline(always)] fn lt_max(self) -> bool { self < i64::MAX }
        #[inline(always)] fn widen(self) -> usize { self as u64 as usize }
    }
    impl Idx for u32 {
        #[inline(always)] fn is_max(self) -> bool { self == u32::MAX }
        #[inline(always)] fn lt_max(self) -> bool { self < u32::MAX }
        #[inline(always)] fn widen(self) -> usize { self as usize }
    }
    impl Idx for u8 {
        #[inline(always)] fn is_max(self) -> bool { self == u8::MAX }
        #[inline(always)] fn lt_max(self) -> bool { self < u8::MAX }
        #[inline(always)] fn widen(self) -> usize { self as usize }
    }
    impl Idx for i32 {
        #[inline(always)] fn is_max(self) -> bool { self == i32::MAX }
        #[inline(always)] fn lt_max(self) -> bool { self < i32::MAX }
        #[inline(always)] fn widen(self) -> usize { self as i64 as u64 as usize }
    }
    impl Idx for i16 {
        #[inline(always)] fn is_max(self) -> bool { self == i16::MAX }
        #[inline(always)] fn lt_max(self) -> bool { self < i16::MAX }
        #[inline(always)] fn widen(self) -> usize { self as i64 as u64 as usize }
    }

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

    #[inline(always)]
    fn start() -> u32 {
        thread::blockIdx_x().wrapping_mul(thread::blockDim_x()).wrapping_add(thread::threadIdx_x())
    }
    #[inline(always)]
    fn step() -> u32 {
        thread::blockDim_x().wrapping_mul(thread::gridDim_x())
    }

    // ---- element adds, exactly the reference's instructions ----
    #[inline(always)]
    pub fn add_bf16(x: u16, y: u16) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("add.bf16 %0, %1, %2;", out("=h") r, in("h") x, in("h") y, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn add_f16(x: u16, y: u16) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("add.f16 %0, %1, %2;", out("=h") r, in("h") x, in("h") y, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn add_f32(x: f32, y: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("add.f32 %0, %1, %2;", out("=f") r, in("f") x, in("f") y, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn add_f64(x: f64, y: f64) -> f64 {
        let r: f64;
        unsafe { ptx_asm!("add.f64 %0, %1, %2;", out("=d") r, in("d") x, in("d") y, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn add_u8(x: u8, y: u8) -> u8 { x.wrapping_add(y) }
    #[inline(always)]
    pub fn add_u32(x: u32, y: u32) -> u32 { x.wrapping_add(y) }
    #[inline(always)]
    pub fn add_i64(x: i64, y: i64) -> i64 { x.wrapping_add(y) }

    /// F8E4M3_TO_FLOAT: e4m3 -> f16 (`cvt.rn.f16x2.e4m3x2`, low half) -> f32.
    #[inline(always)]
    fn e4m3_f32(x: u8) -> f32 {
        let h: u32;
        let r: f32;
        unsafe {
            ptx_asm!("cvt.rn.f16x2.e4m3x2 %0, %1;", out("=r") h, in("h") x as u16, options(register_only));
            ptx_asm!("cvt.f32.f16 %0, %1;", out("=f") r, in("h") h as u16, options(register_only));
        }
        r
    }
    /// `__nv_fp8_e4m3(float)`: satfinite round-to-nearest, value in the low byte.
    #[inline(always)]
    fn f32_e4m3(x: f32) -> u8 {
        let r: u16;
        unsafe { ptx_asm!("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;", out("=h") r, in("f") 0.0f32, in("f") x, options(register_only)); }
        r as u8
    }
    #[inline(always)]
    pub fn add_e4m3(x: u8, y: u8) -> u8 { f32_e4m3(add_f32(e4m3_f32(x), e4m3_f32(y))) }

    // ---- the five loops ----
    #[inline(always)]
    pub unsafe fn index_select<T: Copy, I: Idx>(
        numel: usize, num_dims: usize, info: *const usize, ids: *const I, inp: *const T, out: *mut T,
        _left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize, zero: T,
    ) {
        let dims = info;
        let strides = info.wrapping_add(num_dims);
        let b = is_contiguous(num_dims, dims, strides);
        let step = step();
        let mut dst_i: u32 = start();
        while (dst_i as usize) < numel {
            let left_i = (dst_i as usize / ids_dim_size.wrapping_mul(right_size)) as u32;
            let id_i = (dst_i as usize / right_size % ids_dim_size) as u32;
            let right_i = (dst_i as usize % right_size) as u32;
            let id = *ids.add(id_i as usize);
            if id.is_max() {
                *out.add(dst_i as usize) = zero;
            } else {
                gpu_assert!(id.widen() < src_dim_size, "ids[id_i] < src_dim_size");
                let src_i = (left_i as usize)
                    .wrapping_mul(src_dim_size.wrapping_mul(right_size))
                    .wrapping_add(id.widen().wrapping_mul(right_size))
                    .wrapping_add(right_i as usize) as u32;
                let s = if b { src_i } else { get_strided_index(src_i, num_dims, dims, strides) };
                *out.add(dst_i as usize) = *inp.add(s as usize);
            }
            dst_i = dst_i.wrapping_add(step);
        }
    }

    #[inline(always)]
    pub unsafe fn gather<T: Copy, I: Idx>(
        numel: usize, ids: *const I, inp: *const T, out: *mut T,
        _left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize, zero: T,
    ) {
        let step = step();
        let mut i: u32 = start();
        while (i as usize) < numel {
            let post = i as usize % right_size;
            let idx = *ids.add(i as usize);
            if idx.is_max() {
                *out.add(i as usize) = zero;
            } else {
                gpu_assert!(idx.widen() < src_dim_size, "idx < src_dim_size");
                let pre = i as usize / right_size.wrapping_mul(ids_dim_size);
                let src_i = pre.wrapping_mul(src_dim_size).wrapping_add(idx.widen()).wrapping_mul(right_size).wrapping_add(post);
                *out.add(i as usize) = *inp.add(src_i);
            }
            i = i.wrapping_add(step);
        }
    }

    #[inline(always)]
    pub unsafe fn index_add<T: Copy, I: Idx, F: Fn(T, T) -> T>(
        ids: *const I, ids_dim_size: usize, inp: *const T, out: *mut T,
        left_size: usize, _src_dim_size: usize, dst_dim_size: usize, right_size: usize, add: F,
    ) {
        let numel = left_size.wrapping_mul(right_size);
        let step = step();
        let mut i: u32 = start();
        while (i as usize) < numel {
            let pre = i as usize / right_size;
            let post = i as usize % right_size;
            let mut j: u32 = 0;
            while (j as usize) < ids_dim_size {
                let idx = *ids.add(j as usize);
                let src_i = pre.wrapping_mul(ids_dim_size).wrapping_add(j as usize).wrapping_mul(right_size).wrapping_add(post);
                if idx.lt_max() {
                    gpu_assert!(idx.widen() < dst_dim_size, "idx < dst_dim_size");
                    let dst_i = pre.wrapping_mul(dst_dim_size).wrapping_add(idx.widen()).wrapping_mul(right_size).wrapping_add(post);
                    *out.add(dst_i) = add(*out.add(dst_i), *inp.add(src_i));
                }
                j = j.wrapping_add(1);
            }
            i = i.wrapping_add(step);
        }
    }

    #[inline(always)]
    pub unsafe fn index_add_f8<I: Idx>(
        ids: *const I, ids_dim_size: usize, inp: *const u8, out: *mut u8,
        left_size: usize, _src_dim_size: usize, dst_dim_size: usize, right_size: usize,
    ) {
        let numel = left_size.wrapping_mul(right_size);
        let step = step();
        let mut i: u32 = start();
        while (i as usize) < numel {
            let pre = i as usize / right_size;
            let post = i as usize % right_size;
            let mut j: u32 = 0;
            while (j as usize) < ids_dim_size {
                let idx = (*ids.add(j as usize)).widen();
                let src_i = pre.wrapping_mul(ids_dim_size).wrapping_add(j as usize).wrapping_mul(right_size).wrapping_add(post);
                let dst_i = pre.wrapping_mul(dst_dim_size).wrapping_add(idx).wrapping_mul(right_size).wrapping_add(post);
                *out.add(dst_i) = add_e4m3(*out.add(dst_i), *inp.add(src_i));
                j = j.wrapping_add(1);
            }
            i = i.wrapping_add(step);
        }
    }

    #[inline(always)]
    pub unsafe fn scatter<T: Copy, I: Idx>(
        ids: *const I, inp: *const T, out: *mut T,
        left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize,
    ) {
        let numel = left_size.wrapping_mul(right_size);
        let step = step();
        let mut i: u32 = start();
        while (i as usize) < numel {
            let pre = i as usize / right_size;
            let post = i as usize % right_size;
            let mut j: u32 = 0;
            while (j as usize) < src_dim_size {
                let src_i = pre.wrapping_mul(src_dim_size).wrapping_add(j as usize).wrapping_mul(right_size).wrapping_add(post);
                let idx = *ids.add(src_i);
                if idx.lt_max() {
                    gpu_assert!(idx.widen() < dst_dim_size, "idx < dst_dim_size");
                    let dst_i = pre.wrapping_mul(dst_dim_size).wrapping_add(idx.widen()).wrapping_mul(right_size).wrapping_add(post);
                    *out.add(dst_i) = *inp.add(src_i);
                }
                j = j.wrapping_add(1);
            }
            i = i.wrapping_add(step);
        }
    }

    #[inline(always)]
    pub unsafe fn scatter_add<T: Copy, I: Idx, F: Fn(T, T) -> T>(
        ids: *const I, inp: *const T, out: *mut T,
        left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize, add: F,
    ) {
        let numel = left_size.wrapping_mul(right_size);
        let step = step();
        let mut i: u32 = start();
        while (i as usize) < numel {
            let pre = i as usize / right_size;
            let post = i as usize % right_size;
            let mut j: u32 = 0;
            while (j as usize) < src_dim_size {
                let src_i = pre.wrapping_mul(src_dim_size).wrapping_add(j as usize).wrapping_mul(right_size).wrapping_add(post);
                let idx = *ids.add(src_i);
                if idx.lt_max() {
                    gpu_assert!(idx.widen() < dst_dim_size, "idx < dst_dim_size");
                    let dst_i = pre.wrapping_mul(dst_dim_size).wrapping_add(idx.widen()).wrapping_mul(right_size).wrapping_add(post);
                    *out.add(dst_i) = add(*out.add(dst_i), *inp.add(src_i));
                }
                j = j.wrapping_add(1);
            }
            i = i.wrapping_add(step);
        }
    }

    #[inline(always)]
    pub unsafe fn scatter_add_f8<I: Idx>(
        ids: *const I, inp: *const u8, out: *mut u8,
        left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize,
    ) {
        let numel = left_size.wrapping_mul(right_size);
        let step = step();
        let mut i: u32 = start();
        while (i as usize) < numel {
            let pre = i as usize / right_size;
            let post = i as usize % right_size;
            let mut j: u32 = 0;
            while (j as usize) < src_dim_size {
                let src_i = pre.wrapping_mul(src_dim_size).wrapping_add(j as usize).wrapping_mul(right_size).wrapping_add(post);
                let idx = (*ids.add(src_i)).widen();
                let dst_i = pre.wrapping_mul(dst_dim_size).wrapping_add(idx).wrapping_mul(right_size).wrapping_add(post);
                *out.add(dst_i) = add_e4m3(*out.add(dst_i), *inp.add(src_i));
                j = j.wrapping_add(1);
            }
            i = i.wrapping_add(step);
        }
    }

    // GENERATED KERNELS BEGIN
    #[kernel]
    pub unsafe fn is_i64_bf16(numel: usize, num_dims: usize, info: *const usize, ids: *const i64, inp: *const u16, out: *mut u16, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { index_select(numel, num_dims, info, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0u16); }
    }

    #[kernel]
    pub unsafe fn is_u32_bf16(numel: usize, num_dims: usize, info: *const usize, ids: *const u32, inp: *const u16, out: *mut u16, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { index_select(numel, num_dims, info, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0u16); }
    }

    #[kernel]
    pub unsafe fn is_u8_bf16(numel: usize, num_dims: usize, info: *const usize, ids: *const u8, inp: *const u16, out: *mut u16, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { index_select(numel, num_dims, info, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0u16); }
    }

    #[kernel]
    pub unsafe fn gather_i64_bf16(numel: usize, ids: *const i64, inp: *const u16, out: *mut u16, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { gather(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0u16); }
    }

    #[kernel]
    pub unsafe fn gather_u32_bf16(numel: usize, ids: *const u32, inp: *const u16, out: *mut u16, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { gather(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0u16); }
    }

    #[kernel]
    pub unsafe fn gather_u8_bf16(numel: usize, ids: *const u8, inp: *const u16, out: *mut u16, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { gather(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0u16); }
    }

    #[kernel]
    pub unsafe fn ia_i64_bf16(ids: *const i64, ids_dim_size: usize, inp: *const u16, out: *mut u16, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { index_add(ids, ids_dim_size, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_bf16); }
    }

    #[kernel]
    pub unsafe fn ia_u32_bf16(ids: *const u32, ids_dim_size: usize, inp: *const u16, out: *mut u16, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { index_add(ids, ids_dim_size, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_bf16); }
    }

    #[kernel]
    pub unsafe fn ia_u8_bf16(ids: *const u8, ids_dim_size: usize, inp: *const u16, out: *mut u16, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { index_add(ids, ids_dim_size, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_bf16); }
    }

    #[kernel]
    pub unsafe fn sa_i64_bf16(ids: *const i64, inp: *const u16, out: *mut u16, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter_add(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_bf16); }
    }

    #[kernel]
    pub unsafe fn sa_u32_bf16(ids: *const u32, inp: *const u16, out: *mut u16, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter_add(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_bf16); }
    }

    #[kernel]
    pub unsafe fn sa_u8_bf16(ids: *const u8, inp: *const u16, out: *mut u16, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter_add(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_bf16); }
    }

    #[kernel]
    pub unsafe fn s_i64_bf16(ids: *const i64, inp: *const u16, out: *mut u16, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size); }
    }

    #[kernel]
    pub unsafe fn s_u32_bf16(ids: *const u32, inp: *const u16, out: *mut u16, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size); }
    }

    #[kernel]
    pub unsafe fn s_u8_bf16(ids: *const u8, inp: *const u16, out: *mut u16, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size); }
    }

    #[kernel]
    pub unsafe fn is_i16_f8_e4m3(numel: usize, num_dims: usize, info: *const usize, ids: *const i16, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { index_select(numel, num_dims, info, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0u8); }
    }

    #[kernel]
    pub unsafe fn is_i32_f8_e4m3(numel: usize, num_dims: usize, info: *const usize, ids: *const i32, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { index_select(numel, num_dims, info, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0u8); }
    }

    #[kernel]
    pub unsafe fn is_i64_f8_e4m3(numel: usize, num_dims: usize, info: *const usize, ids: *const i64, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { index_select(numel, num_dims, info, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0u8); }
    }

    #[kernel]
    pub unsafe fn is_u32_f8_e4m3(numel: usize, num_dims: usize, info: *const usize, ids: *const u32, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { index_select(numel, num_dims, info, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0u8); }
    }

    #[kernel]
    pub unsafe fn is_u8_f8_e4m3(numel: usize, num_dims: usize, info: *const usize, ids: *const u8, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { index_select(numel, num_dims, info, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0u8); }
    }

    #[kernel]
    pub unsafe fn gather_i16_f8_e4m3(numel: usize, ids: *const i16, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { gather(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0u8); }
    }

    #[kernel]
    pub unsafe fn gather_i32_f8_e4m3(numel: usize, ids: *const i32, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { gather(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0u8); }
    }

    #[kernel]
    pub unsafe fn gather_i64_f8_e4m3(numel: usize, ids: *const i64, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { gather(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0u8); }
    }

    #[kernel]
    pub unsafe fn gather_u32_f8_e4m3(numel: usize, ids: *const u32, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { gather(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0u8); }
    }

    #[kernel]
    pub unsafe fn gather_u8_f8_e4m3(numel: usize, ids: *const u8, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { gather(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0u8); }
    }

    #[kernel]
    pub unsafe fn ia_i16_f8_e4m3(ids: *const i16, ids_dim_size: usize, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { index_add_f8(ids, ids_dim_size, inp, out, left_size, src_dim_size, dst_dim_size, right_size); }
    }

    #[kernel]
    pub unsafe fn ia_i32_f8_e4m3(ids: *const i32, ids_dim_size: usize, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { index_add_f8(ids, ids_dim_size, inp, out, left_size, src_dim_size, dst_dim_size, right_size); }
    }

    #[kernel]
    pub unsafe fn ia_i64_f8_e4m3(ids: *const i64, ids_dim_size: usize, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { index_add_f8(ids, ids_dim_size, inp, out, left_size, src_dim_size, dst_dim_size, right_size); }
    }

    #[kernel]
    pub unsafe fn ia_u32_f8_e4m3(ids: *const u32, ids_dim_size: usize, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { index_add_f8(ids, ids_dim_size, inp, out, left_size, src_dim_size, dst_dim_size, right_size); }
    }

    #[kernel]
    pub unsafe fn ia_u8_f8_e4m3(ids: *const u8, ids_dim_size: usize, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { index_add_f8(ids, ids_dim_size, inp, out, left_size, src_dim_size, dst_dim_size, right_size); }
    }

    #[kernel]
    pub unsafe fn sa_i16_f8_e4m3(ids: *const i16, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter_add_f8(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size); }
    }

    #[kernel]
    pub unsafe fn sa_i32_f8_e4m3(ids: *const i32, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter_add_f8(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size); }
    }

    #[kernel]
    pub unsafe fn sa_i64_f8_e4m3(ids: *const i64, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter_add_f8(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size); }
    }

    #[kernel]
    pub unsafe fn sa_u32_f8_e4m3(ids: *const u32, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter_add_f8(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size); }
    }

    #[kernel]
    pub unsafe fn sa_u8_f8_e4m3(ids: *const u8, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter_add_f8(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size); }
    }

    #[kernel]
    pub unsafe fn is_i64_f16(numel: usize, num_dims: usize, info: *const usize, ids: *const i64, inp: *const u16, out: *mut u16, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { index_select(numel, num_dims, info, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0u16); }
    }

    #[kernel]
    pub unsafe fn is_u32_f16(numel: usize, num_dims: usize, info: *const usize, ids: *const u32, inp: *const u16, out: *mut u16, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { index_select(numel, num_dims, info, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0u16); }
    }

    #[kernel]
    pub unsafe fn is_u8_f16(numel: usize, num_dims: usize, info: *const usize, ids: *const u8, inp: *const u16, out: *mut u16, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { index_select(numel, num_dims, info, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0u16); }
    }

    #[kernel]
    pub unsafe fn gather_i64_f16(numel: usize, ids: *const i64, inp: *const u16, out: *mut u16, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { gather(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0u16); }
    }

    #[kernel]
    pub unsafe fn gather_u32_f16(numel: usize, ids: *const u32, inp: *const u16, out: *mut u16, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { gather(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0u16); }
    }

    #[kernel]
    pub unsafe fn gather_u8_f16(numel: usize, ids: *const u8, inp: *const u16, out: *mut u16, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { gather(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0u16); }
    }

    #[kernel]
    pub unsafe fn ia_i64_f16(ids: *const i64, ids_dim_size: usize, inp: *const u16, out: *mut u16, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { index_add(ids, ids_dim_size, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_f16); }
    }

    #[kernel]
    pub unsafe fn ia_u32_f16(ids: *const u32, ids_dim_size: usize, inp: *const u16, out: *mut u16, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { index_add(ids, ids_dim_size, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_f16); }
    }

    #[kernel]
    pub unsafe fn ia_u8_f16(ids: *const u8, ids_dim_size: usize, inp: *const u16, out: *mut u16, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { index_add(ids, ids_dim_size, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_f16); }
    }

    #[kernel]
    pub unsafe fn sa_i64_f16(ids: *const i64, inp: *const u16, out: *mut u16, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter_add(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_f16); }
    }

    #[kernel]
    pub unsafe fn sa_u32_f16(ids: *const u32, inp: *const u16, out: *mut u16, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter_add(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_f16); }
    }

    #[kernel]
    pub unsafe fn sa_u8_f16(ids: *const u8, inp: *const u16, out: *mut u16, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter_add(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_f16); }
    }

    #[kernel]
    pub unsafe fn s_i64_f16(ids: *const i64, inp: *const u16, out: *mut u16, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size); }
    }

    #[kernel]
    pub unsafe fn s_u32_f16(ids: *const u32, inp: *const u16, out: *mut u16, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size); }
    }

    #[kernel]
    pub unsafe fn s_u8_f16(ids: *const u8, inp: *const u16, out: *mut u16, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size); }
    }

    #[kernel]
    pub unsafe fn is_i64_f32(numel: usize, num_dims: usize, info: *const usize, ids: *const i64, inp: *const f32, out: *mut f32, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { index_select(numel, num_dims, info, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0.0f32); }
    }

    #[kernel]
    pub unsafe fn is_i64_f64(numel: usize, num_dims: usize, info: *const usize, ids: *const i64, inp: *const f64, out: *mut f64, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { index_select(numel, num_dims, info, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0.0f64); }
    }

    #[kernel]
    pub unsafe fn is_i64_u8(numel: usize, num_dims: usize, info: *const usize, ids: *const i64, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { index_select(numel, num_dims, info, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0u8); }
    }

    #[kernel]
    pub unsafe fn is_i64_u32(numel: usize, num_dims: usize, info: *const usize, ids: *const i64, inp: *const u32, out: *mut u32, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { index_select(numel, num_dims, info, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0u32); }
    }

    #[kernel]
    pub unsafe fn is_i64_i64(numel: usize, num_dims: usize, info: *const usize, ids: *const i64, inp: *const i64, out: *mut i64, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { index_select(numel, num_dims, info, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0i64); }
    }

    #[kernel]
    pub unsafe fn is_u32_f32(numel: usize, num_dims: usize, info: *const usize, ids: *const u32, inp: *const f32, out: *mut f32, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { index_select(numel, num_dims, info, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0.0f32); }
    }

    #[kernel]
    pub unsafe fn is_u32_f64(numel: usize, num_dims: usize, info: *const usize, ids: *const u32, inp: *const f64, out: *mut f64, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { index_select(numel, num_dims, info, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0.0f64); }
    }

    #[kernel]
    pub unsafe fn is_u32_u8(numel: usize, num_dims: usize, info: *const usize, ids: *const u32, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { index_select(numel, num_dims, info, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0u8); }
    }

    #[kernel]
    pub unsafe fn is_u32_i64(numel: usize, num_dims: usize, info: *const usize, ids: *const u32, inp: *const i64, out: *mut i64, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { index_select(numel, num_dims, info, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0i64); }
    }

    #[kernel]
    pub unsafe fn is_u32_u32(numel: usize, num_dims: usize, info: *const usize, ids: *const u32, inp: *const u32, out: *mut u32, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { index_select(numel, num_dims, info, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0u32); }
    }

    #[kernel]
    pub unsafe fn is_u8_f32(numel: usize, num_dims: usize, info: *const usize, ids: *const u8, inp: *const f32, out: *mut f32, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { index_select(numel, num_dims, info, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0.0f32); }
    }

    #[kernel]
    pub unsafe fn is_u8_f64(numel: usize, num_dims: usize, info: *const usize, ids: *const u8, inp: *const f64, out: *mut f64, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { index_select(numel, num_dims, info, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0.0f64); }
    }

    #[kernel]
    pub unsafe fn is_u8_u8(numel: usize, num_dims: usize, info: *const usize, ids: *const u8, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { index_select(numel, num_dims, info, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0u8); }
    }

    #[kernel]
    pub unsafe fn is_u8_u32(numel: usize, num_dims: usize, info: *const usize, ids: *const u8, inp: *const u32, out: *mut u32, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { index_select(numel, num_dims, info, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0u32); }
    }

    #[kernel]
    pub unsafe fn is_u8_i64(numel: usize, num_dims: usize, info: *const usize, ids: *const u8, inp: *const i64, out: *mut i64, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { index_select(numel, num_dims, info, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0i64); }
    }

    #[kernel]
    pub unsafe fn gather_i64_f32(numel: usize, ids: *const i64, inp: *const f32, out: *mut f32, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { gather(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0.0f32); }
    }

    #[kernel]
    pub unsafe fn gather_i64_f64(numel: usize, ids: *const i64, inp: *const f64, out: *mut f64, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { gather(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0.0f64); }
    }

    #[kernel]
    pub unsafe fn gather_i64_u8(numel: usize, ids: *const i64, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { gather(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0u8); }
    }

    #[kernel]
    pub unsafe fn gather_i64_u32(numel: usize, ids: *const i64, inp: *const u32, out: *mut u32, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { gather(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0u32); }
    }

    #[kernel]
    pub unsafe fn gather_i64_i64(numel: usize, ids: *const i64, inp: *const i64, out: *mut i64, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { gather(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0i64); }
    }

    #[kernel]
    pub unsafe fn gather_u32_f32(numel: usize, ids: *const u32, inp: *const f32, out: *mut f32, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { gather(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0.0f32); }
    }

    #[kernel]
    pub unsafe fn gather_u32_f64(numel: usize, ids: *const u32, inp: *const f64, out: *mut f64, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { gather(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0.0f64); }
    }

    #[kernel]
    pub unsafe fn gather_u32_u8(numel: usize, ids: *const u32, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { gather(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0u8); }
    }

    #[kernel]
    pub unsafe fn gather_u32_i64(numel: usize, ids: *const u32, inp: *const i64, out: *mut i64, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { gather(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0i64); }
    }

    #[kernel]
    pub unsafe fn gather_u32_u32(numel: usize, ids: *const u32, inp: *const u32, out: *mut u32, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { gather(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0u32); }
    }

    #[kernel]
    pub unsafe fn gather_u8_f32(numel: usize, ids: *const u8, inp: *const f32, out: *mut f32, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { gather(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0.0f32); }
    }

    #[kernel]
    pub unsafe fn gather_u8_f64(numel: usize, ids: *const u8, inp: *const f64, out: *mut f64, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { gather(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0.0f64); }
    }

    #[kernel]
    pub unsafe fn gather_u8_u8(numel: usize, ids: *const u8, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { gather(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0u8); }
    }

    #[kernel]
    pub unsafe fn gather_u8_u32(numel: usize, ids: *const u8, inp: *const u32, out: *mut u32, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { gather(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0u32); }
    }

    #[kernel]
    pub unsafe fn gather_u8_i64(numel: usize, ids: *const u8, inp: *const i64, out: *mut i64, left_size: usize, src_dim_size: usize, ids_dim_size: usize, right_size: usize) {
        unsafe { gather(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size, 0i64); }
    }

    #[kernel]
    pub unsafe fn ia_i64_f32(ids: *const i64, ids_dim_size: usize, inp: *const f32, out: *mut f32, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { index_add(ids, ids_dim_size, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_f32); }
    }

    #[kernel]
    pub unsafe fn ia_i64_f64(ids: *const i64, ids_dim_size: usize, inp: *const f64, out: *mut f64, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { index_add(ids, ids_dim_size, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_f64); }
    }

    #[kernel]
    pub unsafe fn ia_i64_u8(ids: *const i64, ids_dim_size: usize, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { index_add(ids, ids_dim_size, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_u8); }
    }

    #[kernel]
    pub unsafe fn ia_i64_i64(ids: *const i64, ids_dim_size: usize, inp: *const i64, out: *mut i64, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { index_add(ids, ids_dim_size, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_i64); }
    }

    #[kernel]
    pub unsafe fn ia_i64_u32(ids: *const i64, ids_dim_size: usize, inp: *const u32, out: *mut u32, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { index_add(ids, ids_dim_size, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_u32); }
    }

    #[kernel]
    pub unsafe fn ia_u32_f32(ids: *const u32, ids_dim_size: usize, inp: *const f32, out: *mut f32, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { index_add(ids, ids_dim_size, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_f32); }
    }

    #[kernel]
    pub unsafe fn ia_u32_f64(ids: *const u32, ids_dim_size: usize, inp: *const f64, out: *mut f64, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { index_add(ids, ids_dim_size, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_f64); }
    }

    #[kernel]
    pub unsafe fn ia_u32_u8(ids: *const u32, ids_dim_size: usize, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { index_add(ids, ids_dim_size, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_u8); }
    }

    #[kernel]
    pub unsafe fn ia_u32_i64(ids: *const u32, ids_dim_size: usize, inp: *const i64, out: *mut i64, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { index_add(ids, ids_dim_size, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_i64); }
    }

    #[kernel]
    pub unsafe fn ia_u32_u32(ids: *const u32, ids_dim_size: usize, inp: *const u32, out: *mut u32, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { index_add(ids, ids_dim_size, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_u32); }
    }

    #[kernel]
    pub unsafe fn ia_u8_f32(ids: *const u8, ids_dim_size: usize, inp: *const f32, out: *mut f32, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { index_add(ids, ids_dim_size, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_f32); }
    }

    #[kernel]
    pub unsafe fn ia_u8_f64(ids: *const u8, ids_dim_size: usize, inp: *const f64, out: *mut f64, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { index_add(ids, ids_dim_size, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_f64); }
    }

    #[kernel]
    pub unsafe fn ia_u8_u8(ids: *const u8, ids_dim_size: usize, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { index_add(ids, ids_dim_size, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_u8); }
    }

    #[kernel]
    pub unsafe fn ia_u8_u32(ids: *const u8, ids_dim_size: usize, inp: *const u32, out: *mut u32, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { index_add(ids, ids_dim_size, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_u32); }
    }

    #[kernel]
    pub unsafe fn ia_u8_i64(ids: *const u8, ids_dim_size: usize, inp: *const i64, out: *mut i64, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { index_add(ids, ids_dim_size, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_i64); }
    }

    #[kernel]
    pub unsafe fn sa_i64_f32(ids: *const i64, inp: *const f32, out: *mut f32, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter_add(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_f32); }
    }

    #[kernel]
    pub unsafe fn sa_i64_f64(ids: *const i64, inp: *const f64, out: *mut f64, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter_add(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_f64); }
    }

    #[kernel]
    pub unsafe fn sa_i64_u8(ids: *const i64, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter_add(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_u8); }
    }

    #[kernel]
    pub unsafe fn sa_i64_i64(ids: *const i64, inp: *const i64, out: *mut i64, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter_add(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_i64); }
    }

    #[kernel]
    pub unsafe fn sa_i64_u32(ids: *const i64, inp: *const u32, out: *mut u32, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter_add(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_u32); }
    }

    #[kernel]
    pub unsafe fn sa_u32_f32(ids: *const u32, inp: *const f32, out: *mut f32, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter_add(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_f32); }
    }

    #[kernel]
    pub unsafe fn sa_u32_f64(ids: *const u32, inp: *const f64, out: *mut f64, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter_add(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_f64); }
    }

    #[kernel]
    pub unsafe fn sa_u32_u8(ids: *const u32, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter_add(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_u8); }
    }

    #[kernel]
    pub unsafe fn sa_u32_i64(ids: *const u32, inp: *const i64, out: *mut i64, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter_add(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_i64); }
    }

    #[kernel]
    pub unsafe fn sa_u32_u32(ids: *const u32, inp: *const u32, out: *mut u32, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter_add(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_u32); }
    }

    #[kernel]
    pub unsafe fn sa_u8_f32(ids: *const u8, inp: *const f32, out: *mut f32, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter_add(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_f32); }
    }

    #[kernel]
    pub unsafe fn sa_u8_f64(ids: *const u8, inp: *const f64, out: *mut f64, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter_add(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_f64); }
    }

    #[kernel]
    pub unsafe fn sa_u8_u8(ids: *const u8, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter_add(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_u8); }
    }

    #[kernel]
    pub unsafe fn sa_u8_u32(ids: *const u8, inp: *const u32, out: *mut u32, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter_add(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_u32); }
    }

    #[kernel]
    pub unsafe fn sa_u8_i64(ids: *const u8, inp: *const i64, out: *mut i64, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter_add(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size, add_i64); }
    }

    #[kernel]
    pub unsafe fn s_i64_f32(ids: *const i64, inp: *const f32, out: *mut f32, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size); }
    }

    #[kernel]
    pub unsafe fn s_i64_f64(ids: *const i64, inp: *const f64, out: *mut f64, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size); }
    }

    #[kernel]
    pub unsafe fn s_i64_u8(ids: *const i64, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size); }
    }

    #[kernel]
    pub unsafe fn s_i64_i64(ids: *const i64, inp: *const i64, out: *mut i64, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size); }
    }

    #[kernel]
    pub unsafe fn s_i64_u32(ids: *const i64, inp: *const u32, out: *mut u32, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size); }
    }

    #[kernel]
    pub unsafe fn s_u32_f32(ids: *const u32, inp: *const f32, out: *mut f32, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size); }
    }

    #[kernel]
    pub unsafe fn s_u32_f64(ids: *const u32, inp: *const f64, out: *mut f64, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size); }
    }

    #[kernel]
    pub unsafe fn s_u32_u8(ids: *const u32, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size); }
    }

    #[kernel]
    pub unsafe fn s_u32_i64(ids: *const u32, inp: *const i64, out: *mut i64, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size); }
    }

    #[kernel]
    pub unsafe fn s_u32_u32(ids: *const u32, inp: *const u32, out: *mut u32, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size); }
    }

    #[kernel]
    pub unsafe fn s_u8_f32(ids: *const u8, inp: *const f32, out: *mut f32, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size); }
    }

    #[kernel]
    pub unsafe fn s_u8_f64(ids: *const u8, inp: *const f64, out: *mut f64, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size); }
    }

    #[kernel]
    pub unsafe fn s_u8_u8(ids: *const u8, inp: *const u8, out: *mut u8, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size); }
    }

    #[kernel]
    pub unsafe fn s_u8_u32(ids: *const u8, inp: *const u32, out: *mut u32, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size); }
    }

    #[kernel]
    pub unsafe fn s_u8_i64(ids: *const u8, inp: *const i64, out: *mut i64, left_size: usize, src_dim_size: usize, dst_dim_size: usize, right_size: usize) {
        unsafe { scatter(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size); }
    }
    // GENERATED KERNELS END
}

// ---------------------------------------------------------------------------------------------
// Differential gate against candle's nvcc-built indexing.ptx.
use kdiff::{Arg, Harness, Rng, Tally, contiguous, layout_info};

fn vsize(vt: &str) -> usize {
    match vt { "bf16" | "f16" => 2, "f8_e4m3" | "u8" => 1, "f32" | "u32" => 4, "f64" | "i64" => 8, _ => panic!("{vt}") }
}
fn isize_of(it: &str) -> usize {
    match it { "u8" => 1, "i16" => 2, "u32" | "i32" => 4, "i64" => 8, _ => panic!("{it}") }
}
/// candle's `max_value<I>()` sentinel.
fn sentinel(it: &str) -> i64 {
    match it { "u8" => 0xFF, "i16" => 0x7FFF, "u32" => 0xFFFF_FFFF, "i32" => 0x7FFF_FFFF, "i64" => i64::MAX, _ => panic!() }
}
/// Largest non-negative value representable (+1) in the index type: the cap on generated ids.
fn repr_limit(it: &str) -> i64 {
    match it { "u8" => 0x100, "i16" => 0x8000, _ => i64::MAX }
}
fn encode_ids(it: &str, v: &[i64]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * isize_of(it));
    for &x in v {
        b.extend_from_slice(&x.to_le_bytes()[..isize_of(it)]);
    }
    b
}
/// ids in [0, range) (capped to what the type can hold), with boundary values and, when
/// `sent`, the sentinel mixed in.
fn gen_ids(rng: &mut Rng, it: &str, n: usize, range: usize, sent: bool) -> Vec<i64> {
    let range = (range as i64).min(repr_limit(it));
    (0..n)
        .map(|_| match rng.next() % 10 {
            0 if sent => sentinel(it),
            1 => range - 1,
            2 => 0,
            _ => (rng.next() % range as u64) as i64,
        })
        .collect()
}

fn main() {
    let root = format!("{}/titan-engine/oxide-kernels", std::env::var("HOME").unwrap());
    let ref_ptx = format!("{root}/reference/candle/indexing.ptx");
    let h = Harness::new(&ref_ptx, &format!("{root}/candle-indexing/candle_indexing.ptx"));
    let names: Vec<String> = std::fs::read_to_string(&ref_ptx)
        .unwrap()
        .lines()
        .filter_map(|l| l.strip_prefix(".visible .entry ").map(|s| s.trim_end_matches('(').to_string()))
        .collect();
    assert_eq!(names.len(), 125, "reference entry count");
    let mut t = Tally::default();
    let mut rng = Rng(0x1DE7);
    let mut missing = vec![];
    // (left, a, b, right): is/gather: src_dim=a, ids_dim=b; ia: dst_dim=a, ids_dim=b; s/sa: dst_dim=a, src_dim=b.
    let shapes: [(usize, usize, usize, usize); 9] = [
        (1, 1, 1, 1),
        (3, 5, 4, 7),
        (2, 7, 9, 1),
        (1, 300, 40, 3),   // u8 ids: 255 is in range but still the sentinel (skip / fill zero)
        (5, 17, 13, 33),
        (4, 6, 8, 65),
        (7, 2, 11, 5),     // dense collisions
        (1, 3, 1000, 1),   // long j loop, single thread of work for ia/s/sa
        (9, 1, 3, 2),      // every id hits row 0
    ];
    for name in &names {
        if !h.has(true, name) {
            missing.push(name.clone());
            continue;
        }
        let mut parts = name.splitn(3, '_');
        let op = parts.next().unwrap();
        let it = parts.next().unwrap();
        let vt = parts.next().unwrap();
        let (vs, f8) = (vsize(vt), vt == "f8_e4m3");
        let mut shp: Vec<(usize, usize, usize, usize)> = shapes.to_vec();
        if it == "i16" {
            shp.push((1, 32768, 64, 1)); // i16 max 32767 in range: sentinel for is/gather, a real index for f8 ia/sa
        }
        for &(left, a, b, right) in &shp {
            let work = if op == "is" || op == "gather" { left * b * right } else { left * right };
            let grids = [(4u32, 64u32), (((work as u32) + 255) / 256, 256), (1, 32), (1, 1), (3, 96)];
            // f8 index_add / scatter_add have no sentinel check: a sentinel is only generated where it is in range.
            let sent = !(f8 && (op == "ia" || op == "sa"));
            match op {
                "is" => {
                    let (src, ids_dim) = (a, b);
                    let numel = left * ids_dim * right;
                    let dims = vec![left, src, right];
                    let layouts: Vec<(Vec<usize>, Vec<usize>)> = vec![
                        (dims.clone(), contiguous(&dims)),
                        (vec![left * src * right], vec![1]),
                        (dims.clone(), vec![1, left, left * src]),
                        (dims.clone(), vec![src * (right + 2) + 5, right + 2, 1]),
                        (dims.clone(), vec![0, right, 1]),
                        (vec![left, src, 1, right], vec![src * right, right, 77, 1]),
                    ];
                    for (li, (d, s)) in layouts.iter().enumerate() {
                        let src_len = d.iter().zip(s).map(|(d, s)| (d - 1) * s).sum::<usize>() + 1;
                        for (gi, &(grid, block)) in grids.iter().enumerate() {
                            if li > 1 && gi > 2 { continue; }
                            let ids = encode_ids(it, &gen_ids(&mut rng, it, ids_dim, src, sent));
                            let bufs = vec![ids, rng.bytes(src_len * vs), rng.bytes(numel * vs), layout_info(d, s)];
                            let args = [Arg::U64(numel as u64), Arg::U64(d.len() as u64), Arg::Buf(3), Arg::Buf(0), Arg::Buf(1), Arg::Buf(2),
                                        Arg::U64(left as u64), Arg::U64(src as u64), Arg::U64(ids_dim as u64), Arg::U64(right as u64)];
                            let dd = h.diff(name, (grid, 1, 1), (block, 1, 1), 0, &args, &bufs, &[2]);
                            t.record(&format!("{name} shape{:?} layout{li} g{grid}x{block}", (left, a, b, right)), &dd);
                        }
                    }
                }
                "gather" => {
                    let (src, ids_dim) = (a, b);
                    let numel = left * ids_dim * right;
                    for &(grid, block) in &grids {
                        let ids = encode_ids(it, &gen_ids(&mut rng, it, numel, src, sent));
                        let bufs = vec![ids, rng.bytes(left * src * right * vs), rng.bytes(numel * vs)];
                        let args = [Arg::U64(numel as u64), Arg::Buf(0), Arg::Buf(1), Arg::Buf(2),
                                    Arg::U64(left as u64), Arg::U64(src as u64), Arg::U64(ids_dim as u64), Arg::U64(right as u64)];
                        let dd = h.diff(name, (grid, 1, 1), (block, 1, 1), 0, &args, &bufs, &[2]);
                        t.record(&format!("{name} shape{:?} g{grid}x{block}", (left, a, b, right)), &dd);
                    }
                }
                "ia" => {
                    let (dst, ids_dim) = (a, b);
                    for &(grid, block) in &grids {
                        let ids = encode_ids(it, &gen_ids(&mut rng, it, ids_dim, dst, sent));
                        let bufs = vec![ids, rng.bytes(left * ids_dim * right * vs), rng.bytes(left * dst * right * vs)];
                        let args = [Arg::Buf(0), Arg::U64(ids_dim as u64), Arg::Buf(1), Arg::Buf(2),
                                    Arg::U64(left as u64), Arg::U64(ids_dim as u64), Arg::U64(dst as u64), Arg::U64(right as u64)];
                        let dd = h.diff(name, (grid, 1, 1), (block, 1, 1), 0, &args, &bufs, &[2]);
                        t.record(&format!("{name} shape{:?} g{grid}x{block}", (left, a, b, right)), &dd);
                    }
                }
                "s" | "sa" => {
                    let (dst, src) = (a, b);
                    for &(grid, block) in &grids {
                        let ids = encode_ids(it, &gen_ids(&mut rng, it, left * src * right, dst, sent));
                        let bufs = vec![ids, rng.bytes(left * src * right * vs), rng.bytes(left * dst * right * vs)];
                        let args = [Arg::Buf(0), Arg::Buf(1), Arg::Buf(2),
                                    Arg::U64(left as u64), Arg::U64(src as u64), Arg::U64(dst as u64), Arg::U64(right as u64)];
                        let dd = h.diff(name, (grid, 1, 1), (block, 1, 1), 0, &args, &bufs, &[2]);
                        t.record(&format!("{name} shape{:?} g{grid}x{block}", (left, a, b, right)), &dd);
                    }
                }
                _ => panic!("{name}"),
            }
        }
        // index_select's src_i is truncated to u32: right_size = 2^32 + 1 wraps id*right to id.
        if op == "is" {
            let numel = 64usize;
            let right = (1usize << 32) + 1;
            for (d, s) in [(vec![130usize], vec![1usize]), (vec![65, 2], vec![1, 65])] {
                for idv in [0i64, 1, sentinel(it)] {
                    let ids = encode_ids(it, &[idv]);
                    let bufs = vec![ids, rng.bytes(130 * vs), rng.bytes(numel * vs), layout_info(&d, &s)];
                    let args = [Arg::U64(numel as u64), Arg::U64(d.len() as u64), Arg::Buf(3), Arg::Buf(0), Arg::Buf(1), Arg::Buf(2),
                                Arg::U64(1), Arg::U64(2), Arg::U64(1), Arg::U64(right as u64)];
                    let dd = h.diff(name, (2, 1, 1), (32, 1, 1), 0, &args, &bufs, &[2]);
                    t.record(&format!("{name} u32-truncation dims{d:?} id{idv}"), &dd);
                }
            }
        }
    }
    for m in &missing {
        println!("  MISSING from oxide PTX: {m}");
    }
    let ok = t.finish("indexing") && missing.is_empty();
    println!("indexing: {} of {} reference entries ported", names.len() - missing.len(), names.len());
    std::process::exit(if ok { 0 } else { 1 });
}
