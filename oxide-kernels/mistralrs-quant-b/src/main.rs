//! mistralrs-quant group B (quantization formats) in cuda-oxide: the kernels behind the extern "C"
//! launchers of mistralrs-quant's afq, fp8 (scalar / vector / blockwise), gptq, mxfp4, marlin and
//! cutlass-moe modules, plus (src/launch.rs) pure-Rust twins of those launchers. Bit-identical to
//! the nvcc build (-O3 --use_fast_math sm_120a, SASS in libmistralrsquant.a): checked by src/gate.rs,
//! which calls the REAL C launchers and the Rust twins on identical inputs.
//!
//! Fast-math program (read off the SASS): every f32 add/mul/fma is `.ftz`, `a / b` is
//! `rcp.approx.ftz(b) * a`, a constant divisor is a multiply by its f32 reciprocal, `fminf`/`fmaxf`
//! are `min.ftz`/`max.ftz`, and `(int)roundf(x)` is `cvt.rzi(add.rz.ftz(x, copysign(0.5, x)))`.
#![allow(non_snake_case, clippy::missing_safety_doc, clippy::too_many_arguments, unused_unsafe)]
mod gate;
mod gen_names;
#[path = "gen_marlin_names.rs"]
mod gen_names_marlin;
pub mod launch;

use cuda_device::{SharedArray, kernel, ptx_asm, thread, warp};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    // ------------------------------------------------------------------ fast-math f32 ops
    #[inline(always)]
    pub fn fma(a: f32, b: f32, c: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("fma.rn.ftz.f32 %0, %1, %2, %3;", out("=f") r, in("f") a, in("f") b, in("f") c, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn add(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("add.rn.ftz.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn sub(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("sub.rn.ftz.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn mul(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("mul.rn.ftz.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn rcp(a: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("rcp.approx.ftz.f32 %0, %1;", out("=f") r, in("f") a, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn fmin(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("min.ftz.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn fmax(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("max.ftz.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    /// `(int)roundf(x)` under fast-math: add.rz of copysign(0.5, x), then cvt.rzi.
    #[inline(always)]
    pub fn round_i(x: f32) -> i32 {
        let h = f32::from_bits(0x3f00_0000 | (x.to_bits() & 0x8000_0000));
        let t: f32;
        let r: i32;
        unsafe {
            ptx_asm!("add.rz.ftz.f32 %0, %1, %2;", out("=f") t, in("f") x, in("f") h, options(register_only));
            ptx_asm!("cvt.rzi.s32.f32 %0, %1;", out("=r") r, in("f") t, options(register_only));
        }
        r
    }
    #[inline(always)]
    pub fn u2f(q: u32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("cvt.rn.f32.u32 %0, %1;", out("=f") r, in("r") q, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn h2f(h: u16) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("cvt.f32.f16 %0, %1;", out("=f") r, in("h") h, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn bf2f(h: u16) -> f32 {
        f32::from_bits((h as u32) << 16)
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

    /// nvcc's signed 64-bit division: `div.u32` when both operands fit in 32 bits.
    #[inline(always)]
    pub fn sdiv64(a: i64, b: i64) -> i64 {
        if ((a | b) as u64) >> 32 == 0 {
            ((a as u32) / (b as u32)) as i64
        } else {
            a.wrapping_div(b)
        }
    }

    // ------------------------------------------------------------------ element types
    /// f16 element (bit pattern).
    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct H(pub u16);
    /// bf16 element (bit pattern).
    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct B(pub u16);

    pub trait El: Copy {
        unsafe fn ldf(p: *const Self, i: isize) -> f32;
        unsafe fn stf(p: *mut Self, i: isize, v: f32);
        /// AFQ `dequant_value<T>(q, scale, bias)` in the native type.
        unsafe fn deq(q: u32, p_s: *const Self, p_b: *const Self, i: isize, out: *mut Self, o: isize);
    }
    impl El for f32 {
        #[inline(always)]
        unsafe fn ldf(p: *const f32, i: isize) -> f32 {
            *p.offset(i)
        }
        #[inline(always)]
        unsafe fn stf(p: *mut f32, i: isize, v: f32) {
            *p.offset(i) = v;
        }
        #[inline(always)]
        unsafe fn deq(q: u32, s: *const f32, b: *const f32, i: isize, out: *mut f32, o: isize) {
            *out.offset(o) = fma(u2f(q), *s.offset(i), *b.offset(i));
        }
    }
    impl El for H {
        #[inline(always)]
        unsafe fn ldf(p: *const H, i: isize) -> f32 {
            h2f((*p.offset(i)).0)
        }
        #[inline(always)]
        unsafe fn stf(p: *mut H, i: isize, v: f32) {
            *(p.offset(i) as *mut u16) = f2h(v);
        }
        #[inline(always)]
        unsafe fn deq(q: u32, s: *const H, b: *const H, i: isize, out: *mut H, o: isize) {
            let (sv, bv) = ((*s.offset(i)).0, (*b.offset(i)).0);
            let r: u16;
            ptx_asm!("{ .reg .f16 t; cvt.rn.f16.u32 t, %1; fma.rn.f16 %0, t, %2, %3; }", out("=h") r, in("r") q, in("h") sv, in("h") bv, options(register_only));
            *(out.offset(o) as *mut u16) = r;
        }
    }
    impl El for B {
        #[inline(always)]
        unsafe fn ldf(p: *const B, i: isize) -> f32 {
            bf2f((*p.offset(i)).0)
        }
        #[inline(always)]
        unsafe fn stf(p: *mut B, i: isize, v: f32) {
            *(p.offset(i) as *mut u16) = f2bf(v);
        }
        #[inline(always)]
        unsafe fn deq(q: u32, s: *const B, b: *const B, i: isize, out: *mut B, o: isize) {
            let (sv, bv) = ((*s.offset(i)).0, (*b.offset(i)).0);
            let r: u16;
            ptx_asm!("{ .reg .b16 t; cvt.rn.bf16.u32 t, %1; fma.rn.bf16 %0, t, %2, %3; }", out("=h") r, in("r") q, in("h") sv, in("h") bv, options(register_only));
            *(out.offset(o) as *mut u16) = r;
        }
    }

    // ------------------------------------------------------------------ AFQ
    /// AFQ weight extraction. BITS 3 / 6: byte-packed (`extract_3bit` / `extract_6bit`) from the
    /// row's bytes; otherwise `(word[idx / (32/BITS)] >> (idx % (32/BITS)) * BITS) & mask`.
    #[inline(always)]
    pub unsafe fn afq_q<const BITS: i32>(row: *const u8, idx: i32) -> u32 {
        if BITS == 3 || BITS == 6 {
            let byte = idx.wrapping_mul(BITS) / 8;
            let bit = idx.wrapping_mul(BITS) % 8;
            let mask: u32 = (1 << BITS) - 1;
            let lim = 8 - BITS;
            let b0 = *row.offset(byte as isize) as u32;
            if bit <= lim {
                (b0 >> bit) & mask
            } else {
                let b1 = *row.offset(byte as isize + 1) as u32;
                ((b0 >> bit) | (b1 << (8 - bit))) & mask
            }
        } else {
            let vpu = 32 / BITS;
            let w = *(row as *const u32).offset((idx / vpu) as isize);
            (w >> ((idx % vpu) * BITS)) & ((1u32 << BITS) - 1)
        }
    }
    /// Bytes (3/6-bit) or u32 words per row of K packed values.
    #[inline(always)]
    pub fn afq_packed<const BITS: i32>(k: i32) -> i32 {
        if BITS == 3 || BITS == 6 { (k.wrapping_mul(BITS).wrapping_add(7)) / 8 } else { k.wrapping_mul(BITS) / 32 }
    }
    /// Row pointer: packed units (bytes for 3/6-bit, u32 otherwise) times `row`.
    #[inline(always)]
    pub unsafe fn afq_row<const BITS: i32>(w: *const u8, off: i64) -> *const u8 {
        if BITS == 3 || BITS == 6 { w.offset(off as isize) } else { w.offset(off as isize * 4) }
    }

    #[inline(always)]
    pub unsafe fn afq_dequant<T: El, const BITS: i32, const GS: i32>(w_q: *const u8, scales: *const T, biases: *const T, output: *mut T, rows: i32, cols: i32) {
        let packed_cols = afq_packed::<BITS>(cols);
        let groups_per_row = cols / GS;
        let tid = thread::blockIdx_x() as i64 * thread::blockDim_x() as i64 + thread::threadIdx_x() as i64;
        let total = rows as i64 * cols as i64;
        if tid >= total {
            return;
        }
        let q64 = sdiv64(tid, cols as i64);
        let row = q64 as i32;
        let col = tid.wrapping_sub(q64.wrapping_mul(cols as i64)) as i32;
        let q = afq_q::<BITS>(afq_row::<BITS>(w_q, row as i64 * packed_cols as i64), col);
        let g = row as i64 * groups_per_row as i64 + (col / GS) as i64;
        T::deq(q, scales, biases, g as isize, output, tid as isize);
    }

    #[inline(always)]
    pub unsafe fn afq_quantize<T: El, const BITS: i32, const GS: i32>(w: *const T, w_q: *mut u32, scales: *mut T, biases: *mut T, rows: i32, cols: i32) {
        let vpu = 32 / BITS;
        let max_q: i32 = (1 << BITS) - 1;
        let packed_cols = cols.wrapping_mul(BITS) / 32;
        let groups_per_row = cols / GS;
        let gt = thread::blockIdx_x() as i64 * thread::blockDim_x() as i64 + thread::threadIdx_x() as i64;
        let warp_id = gt / 32;
        let lane = (thread::threadIdx_x() % 32) as i32;
        let total_groups = rows as i64 * groups_per_row as i64;
        if warp_id >= total_groups {
            return;
        }
        let q64 = sdiv64(warp_id, groups_per_row as i64);
        let row = q64 as i32;
        let group_idx = warp_id.wrapping_sub(q64.wrapping_mul(groups_per_row as i64)) as i32;
        let group_start = group_idx.wrapping_mul(GS);
        let row_w = w.offset((row as i64 * cols as i64) as isize);

        let mut lmin = f32::MAX;
        let mut lmax = -f32::MAX;
        let mut i = lane;
        while i < GS {
            let col = group_start.wrapping_add(i);
            if col < cols {
                let v = T::ldf(row_w, col as isize);
                lmin = fmin(lmin, v);
                lmax = fmax(lmax, v);
            }
            i += 32;
        }
        let mut off = 16;
        while off > 0 {
            lmin = fmin(lmin, warp::shuffle_down_f32_sync(0xffff_ffff, lmin, off));
            off /= 2;
        }
        let mut off = 16;
        while off > 0 {
            lmax = fmax(lmax, warp::shuffle_down_f32_sync(0xffff_ffff, lmax, off));
            off /= 2;
        }
        let gmin = warp::shuffle_f32_sync(0xffff_ffff, lmin, 0);
        let gmax = warp::shuffle_f32_sync(0xffff_ffff, lmax, 0);
        let scale = fmax(sub(gmax, gmin), 1e-7);
        let bias = gmin;
        if lane == 0 {
            let inv_bins: f32 = match BITS { 2 => 1.0 / 3.0, 4 => 1.0 / 15.0, _ => 1.0 / 255.0 };
            let g = (row as i64 * groups_per_row as i64 + group_idx as i64) as isize;
            T::stf(scales, g, mul(scale, inv_bins));
            T::stf(biases, g, bias);
        }
        let inv_scale = mul(rcp(scale), max_q as f32);
        let mut i = lane;
        while i < GS {
            let col = group_start.wrapping_add(i);
            if col < cols {
                let v = T::ldf(row_w, col as isize);
                let mut q = round_i(mul(inv_scale, sub(v, bias)));
                if q < 0 {
                    q = 0;
                }
                if q > max_q {
                    q = max_q;
                }
                let shift = (col % vpu) * BITS;
                let p = w_q.offset((row as i64 * packed_cols as i64 + (col / vpu) as i64) as isize);
                let val = (q as u32) << shift;
                ptx_asm!("red.global.or.b32 [%0], %1;", in("l") p as u64, in("r") val, clobber("memory"));
            }
            i += 32;
        }
    }

    #[inline(always)]
    pub unsafe fn afq_qmv<T: El, const BITS: i32, const GS: i32>(x: *const T, w_q: *const u8, scales: *const T, biases: *const T, y: *mut T, M: i32, N: i32, K: i32) {
        let packed_k = afq_packed::<BITS>(K);
        let groups_per_row = K / GS;
        let warp_id = (thread::blockIdx_x().wrapping_mul(thread::blockDim_x()).wrapping_add(thread::threadIdx_x()) / 32) as i32;
        let lane = (thread::threadIdx_x() % 32) as i32;
        let total = M.wrapping_mul(N);
        if warp_id >= total {
            return;
        }
        let m = warp_id / N;
        let n = warp_id % N;
        let x_row = x.offset(m.wrapping_mul(K) as isize);
        let w_row = afq_row::<BITS>(w_q, n.wrapping_mul(packed_k) as i64);
        let s_row = scales.offset(n.wrapping_mul(groups_per_row) as isize);
        let b_row = biases.offset(n.wrapping_mul(groups_per_row) as isize);
        let mut acc = 0f32;
        let mut k = lane;
        while k < K {
            let q = afq_q::<BITS>(w_row, k);
            let g = (k / GS) as isize;
            let wv = fma(u2f(q), T::ldf(s_row, g), T::ldf(b_row, g));
            acc = fma(T::ldf(x_row, k as isize), wv, acc);
            k += 32;
        }
        let mut off = 16;
        while off > 0 {
            acc = add(acc, warp::shuffle_down_f32_sync(0xffff_ffff, acc, off));
            off /= 2;
        }
        if lane == 0 {
            T::stf(y, m.wrapping_mul(N).wrapping_add(n) as isize, acc);
        }
    }

    #[inline(always)]
    pub unsafe fn afq_qmm<T: El, const BITS: i32, const GS: i32>(x: *const T, w_q: *const u8, scales: *const T, biases: *const T, y: *mut T, M: i32, N: i32, K: i32) {
        static mut XT: SharedArray<f32, 1024> = SharedArray::UNINIT;
        static mut WT: SharedArray<f32, 1024> = SharedArray::UNINIT;
        let xt = SharedArray::as_raw_mut_ptr(&raw mut XT);
        let wt = SharedArray::as_raw_mut_ptr(&raw mut WT);
        let vpu = 32 / BITS;
        let packed_k = K.wrapping_mul(BITS) / 32;
        let groups_per_row = K / GS;
        let bm = (thread::blockIdx_y() as i32).wrapping_mul(32);
        let bn = (thread::blockIdx_x() as i32).wrapping_mul(32);
        let tx = thread::threadIdx_x() as i32;
        let ty = thread::threadIdx_y() as i32;
        let mut acc = 0f32;
        let mut bk = 0i32;
        while bk < K {
            let m_idx = bm + ty;
            let k_idx = bk + tx;
            let xv = if m_idx < M && k_idx < K { T::ldf(x, m_idx.wrapping_mul(K).wrapping_add(k_idx) as isize) } else { 0.0 };
            *xt.add((ty * 32 + tx) as usize) = xv;
            let n_idx = bn + ty;
            let wv = if n_idx < N && k_idx < K {
                let packed = *(w_q as *const u32).offset(n_idx.wrapping_mul(packed_k).wrapping_add(k_idx / vpu) as isize);
                let q = (packed >> ((k_idx % vpu) * BITS)) & ((1u32 << BITS) - 1);
                let g = n_idx.wrapping_mul(groups_per_row).wrapping_add(k_idx / GS) as isize;
                fma(u2f(q), T::ldf(scales, g), T::ldf(biases, g))
            } else {
                0.0
            };
            *wt.add((ty * 32 + tx) as usize) = wv;
            thread::sync_threads();
            let mut k = 0usize;
            while k < 32 {
                acc = fma(*xt.add(ty as usize * 32 + k), *wt.add(tx as usize * 32 + k), acc);
                k += 1;
            }
            thread::sync_threads();
            bk += 32;
        }
        let mo = bm + ty;
        let no = bn + tx;
        if mo < M && no < N {
            T::stf(y, mo.wrapping_mul(N).wrapping_add(no) as isize, acc);
        }
    }

    // ------------------------------------------------------------------ FP8 (e4m3)
    /// `__half2float(__nv_cvt_fp8_to_halfraw(b, E4M3))`.
    #[inline(always)]
    pub fn fp8_to_f32(b: u8) -> f32 {
        let h: u32;
        let bb = b as u16;
        unsafe { ptx_asm!("cvt.rn.f16x2.e4m3x2 %0, %1;", out("=r") h, in("h") bb, options(register_only)); }
        h2f(h as u16)
    }
    /// Clamp to +-448 (`if (v > 448) v = 448; if (v < -448) v = -448;` = NaN-propagating min/max),
    /// `__float2half`, then `__nv_cvt_halfraw_to_fp8(h, SATFINITE, E4M3)`.
    #[inline(always)]
    pub fn f32_to_fp8_clamped(v: f32) -> u8 {
        let c: f32;
        unsafe { ptx_asm!("{ .reg .f32 t; min.NaN.f32 t, %1, 0f43E00000; max.NaN.f32 %0, t, 0fC3E00000; }", out("=f") c, in("f") v, options(register_only)); }
        let h = f2h(c) as u32;
        let r: u16;
        unsafe { ptx_asm!("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;", out("=h") r, in("r") h, options(register_only)); }
        r as u8
    }
    /// `atomicMaxFloat(addr, fabsf(val))` on a shared float (fast-math: fabsf and the `>= 0` test flush
    /// denormals).
    #[inline(always)]
    pub unsafe fn atomic_absmax_shared(addr: *mut f32, val: f32) {
        let a: f32;
        ptx_asm!("abs.ftz.f32 %0, %1;", out("=f") a, in("f") val, options(register_only));
        let ge: u32;
        ptx_asm!("{ .reg .pred p; setp.ge.ftz.f32 p, %1, 0f00000000; selp.u32 %0, 1, 0, p; }", out("=r") ge, in("f") a, options(register_only));
        let bits = a.to_bits();
        let sa = cuda_device::shared::cvta_generic_to_shared_u32(addr as *const u8);
        if ge != 0 {
            ptx_asm!("red.shared.max.s32 [%0], %1;", in("r") sa, in("r") bits, clobber("memory"));
        } else {
            ptx_asm!("red.shared.min.u32 [%0], %1;", in("r") sa, in("r") bits, clobber("memory"));
        }
    }

    #[inline(always)]
    pub unsafe fn fp8_to_dtype<T: El>(input: *const u8, output: *mut T, n: usize) {
        let idx = thread::blockIdx_x().wrapping_mul(thread::blockDim_x()).wrapping_add(thread::threadIdx_x()) as usize;
        if idx < n {
            T::stf(output, idx as isize, fp8_to_f32(*input.add(idx)));
        }
    }
    #[inline(always)]
    pub unsafe fn dtype_to_fp8<T: El>(input: *const T, output: *mut u8, n: usize) {
        let idx = thread::blockIdx_x().wrapping_mul(thread::blockDim_x()).wrapping_add(thread::threadIdx_x()) as usize;
        if idx < n {
            *output.add(idx) = f32_to_fp8_clamped(T::ldf(input, idx as isize));
        }
    }

    #[inline(always)]
    pub unsafe fn dequant_fp8_vector<T: El>(weight: *const u8, scale: *const f32, output: *mut T, n: usize) {
        let vi = thread::blockIdx_x() as usize;
        let start = vi * 128;
        let mut i = thread::threadIdx_x() as usize;
        if !(i < 128 && start + i < n) {
            return;
        }
        let s = *scale.add(vi);
        while i < 128 && start + i < n {
            let g = start + i;
            T::stf(output, g as isize, mul(s, fp8_to_f32(*weight.add(g))));
            i += thread::blockDim_x() as usize;
        }
    }

    #[inline(always)]
    pub unsafe fn quant_fp8_vector<T: El>(input: *const T, weight: *mut u8, scale: *mut f32, n: usize) {
        static mut SH: SharedArray<f32, 2> = SharedArray::UNINIT;
        let sh = SharedArray::as_raw_mut_ptr(&raw mut SH);
        let vi = thread::blockIdx_x() as usize;
        let start = vi * 128;
        let tid = thread::threadIdx_x() as usize;
        if tid == 0 {
            *sh = 0.0;
        }
        thread::sync_threads();
        let mut i = tid;
        while i < 128 && start + i < n {
            atomic_absmax_shared(sh, T::ldf(input, (start + i) as isize));
            i += thread::blockDim_x() as usize;
        }
        thread::sync_threads();
        if tid == 0 {
            let vs = fp8_scale(*(sh as *const f32));
            *sh.add(1) = vs;
            *scale.add(vi) = vs;
        }
        thread::sync_threads();
        let r = rcp(*sh.add(1));
        let mut i = tid;
        while i < 128 && start + i < n {
            let g = start + i;
            *weight.add(g) = f32_to_fp8_clamped(mul(T::ldf(input, g as isize), r));
            i += thread::blockDim_x() as usize;
        }
    }
    /// `s = absmax / 448.0f; if (s < 1e-12f) s = 1e-12f;` (constant divisor: multiply by 1/448).
    #[inline(always)]
    pub fn fp8_scale(absmax: f32) -> f32 {
        let s = mul(absmax, 1.0 / 448.0);
        let r: f32;
        unsafe { ptx_asm!("max.NaN.f32 %0, %1, 0f2B8CBCCC;", out("=f") r, in("f") s, options(register_only)); }
        r
    }

    #[inline(always)]
    pub unsafe fn dequant_fp8_blockwise<T: El>(weight: *const u8, scale: *const f32, output: *mut T, h: i32, w: i32, row_stride: i32, scale_stride: i32, bsy: i32, bsx: i32) {
        static mut SH: SharedArray<f32, 1> = SharedArray::UNINIT;
        let sh = SharedArray::as_raw_mut_ptr(&raw mut SH);
        let gy = thread::blockIdx_y() as i32;
        let gx = thread::blockIdx_x() as i32;
        let sy = gy.wrapping_mul(bsy);
        let sx = gx.wrapping_mul(bsx);
        let (tx, ty) = (thread::threadIdx_x() as i32, thread::threadIdx_y() as i32);
        if tx == 0 && ty == 0 {
            *sh = *scale.offset(gy.wrapping_mul(scale_stride).wrapping_add(gx) as isize);
        }
        thread::sync_threads();
        let mut ly = ty;
        while ly < bsy {
            let mut lx = tx;
            while lx < bsx {
                let (wy, wx) = (sy.wrapping_add(ly), sx.wrapping_add(lx));
                if wy < h && wx < w {
                    let pos = wy.wrapping_mul(row_stride).wrapping_add(wx) as isize;
                    T::stf(output, pos, mul(fp8_to_f32(*weight.offset(pos)), *sh));
                }
                lx += thread::blockDim_x() as i32;
            }
            ly += thread::blockDim_y() as i32;
        }
    }

    #[inline(always)]
    pub unsafe fn quant_fp8_blockwise<T: El>(input: *const T, weight: *mut u8, scale: *mut f32, h: i32, w: i32, row_stride: i32, scale_stride: i32, bsy: i32, bsx: i32) {
        static mut SH: SharedArray<f32, 2> = SharedArray::UNINIT;
        let sh = SharedArray::as_raw_mut_ptr(&raw mut SH);
        let gy = thread::blockIdx_y() as i32;
        let gx = thread::blockIdx_x() as i32;
        let sy = gy.wrapping_mul(bsy);
        let sx = gx.wrapping_mul(bsx);
        let (tx, ty) = (thread::threadIdx_x() as i32, thread::threadIdx_y() as i32);
        if tx == 0 && ty == 0 {
            *sh = 0.0;
        }
        thread::sync_threads();
        let mut ly = ty;
        while ly < bsy {
            let mut lx = tx;
            while lx < bsx {
                let (wy, wx) = (sy.wrapping_add(ly), sx.wrapping_add(lx));
                if wy < h && wx < w {
                    let pos = wy.wrapping_mul(row_stride).wrapping_add(wx) as isize;
                    atomic_absmax_shared(sh, T::ldf(input, pos));
                }
                lx += thread::blockDim_x() as i32;
            }
            ly += thread::blockDim_y() as i32;
        }
        thread::sync_threads();
        if tx == 0 && ty == 0 {
            let bs = fp8_scale(*(sh as *const f32));
            *sh.add(1) = bs;
            *scale.offset(gy.wrapping_mul(scale_stride).wrapping_add(gx) as isize) = bs;
        }
        thread::sync_threads();
        let r = rcp(*sh.add(1));
        let mut ly = ty;
        while ly < bsy {
            let mut lx = tx;
            while lx < bsx {
                let (wy, wx) = (sy.wrapping_add(ly), sx.wrapping_add(lx));
                if wy < h && wx < w {
                    let pos = wy.wrapping_mul(row_stride).wrapping_add(wx) as isize;
                    *weight.offset(pos) = f32_to_fp8_clamped(mul(T::ldf(input, pos), r));
                }
                lx += thread::blockDim_x() as i32;
            }
            ly += thread::blockDim_y() as i32;
        }
    }

    /// `fp8_gemm::fp8_matmul_tiled<T, 32, 32, 32>`: shared rows padded to 36 floats.
    #[inline(always)]
    pub unsafe fn fp8_matmul_tiled<T: El>(input: *const T, weight: *const u8, wscale: *const f32, output: *mut T, M: i32, N: i32, K: i32, srs: i32, bsy: i32, bsx: i32) {
        static mut SI: SharedArray<f32, { 32 * 36 }, 16> = SharedArray::UNINIT;
        static mut SW: SharedArray<f32, { 32 * 36 }, 16> = SharedArray::UNINIT;
        let si = SharedArray::as_raw_mut_ptr(&raw mut SI);
        let sw = SharedArray::as_raw_mut_ptr(&raw mut SW);
        let bx = thread::blockIdx_x() as i32;
        let by = thread::blockIdx_y() as i32;
        let tx = thread::threadIdx_x() as i32;
        let ty = thread::threadIdx_y() as i32;
        let row = by.wrapping_mul(32).wrapping_add(ty);
        let col = bx.wrapping_mul(32).wrapping_add(tx);
        let mut acc = 0f32;
        let tid = ty * 32 + tx;
        let mut kt = 0i32;
        while kt < K {
            let mut i = tid;
            while i < 1024 {
                let (lm, lk) = (i / 32, i % 32);
                let gm = by.wrapping_mul(32).wrapping_add(lm);
                let gk = kt.wrapping_add(lk);
                let v = if gm < M && gk < K { T::ldf(input, gm.wrapping_mul(K).wrapping_add(gk) as isize) } else { 0.0 };
                *si.add((lm * 36 + lk) as usize) = v;
                i += 1024;
            }
            let mut i = tid;
            while i < 1024 {
                let (ln, lk) = (i / 32, i % 32);
                let gn = bx.wrapping_mul(32).wrapping_add(ln);
                let gk = kt.wrapping_add(lk);
                let v = if gn < N && gk < K {
                    let wv = fp8_to_f32(*weight.offset(gn.wrapping_mul(K).wrapping_add(gk) as isize));
                    let s = *wscale.offset(((gn / bsy).wrapping_mul(srs).wrapping_add(gk / bsx)) as isize);
                    mul(wv, s)
                } else {
                    0.0
                };
                *sw.add((ln * 36 + lk) as usize) = v;
                i += 1024;
            }
            thread::sync_threads();
            if row < M && col < N {
                let mut k = 0usize;
                while k < 32 {
                    acc = fma(*si.add(ty as usize * 36 + k), *sw.add(tx as usize * 36 + k), acc);
                    k += 1;
                }
            }
            thread::sync_threads();
            kt += 32;
        }
        if row < M && col < N {
            T::stf(output, row.wrapping_mul(N).wrapping_add(col) as isize, acc);
        }
    }

    /// `fp8_gemm::fp8_moe_gemm<T>`: one warp per output element.
    #[inline(always)]
    pub unsafe fn fp8_moe_gemm<T: El>(input: *const T, weights: *const u8, wscales: *const f32, indices: *const u32, output: *mut T, num_tokens: i32, topk: i32, num_experts: i32, N: i32, K: i32, srs: i32, bsy: i32, bsx: i32, has_topk: u8) {
        let warp_id = (thread::blockIdx_x().wrapping_mul(thread::blockDim_x()).wrapping_add(thread::threadIdx_x()) / 32) as i32;
        let lane = (thread::threadIdx_x() % 32) as i32;
        let n_idx = warp_id % N;
        let temp = warp_id / N;
        let slot = temp % topk;
        let token = temp / topk;
        if token >= num_tokens {
            return;
        }
        let expert = *indices.offset(token.wrapping_mul(topk).wrapping_add(slot) as isize);
        if expert >= num_experts as u32 {
            return;
        }
        let (Nu, Ku) = (N as i64 as u64, K as i64 as u64);
        let w_row = weights.add((expert as u64).wrapping_mul(Nu).wrapping_mul(Ku).wrapping_add((n_idx as i64 as u64).wrapping_mul(Ku)) as usize);
        let scale_n_dim = N.wrapping_add(bsy - 1) / bsy;
        let ses = scale_n_dim.wrapping_mul(srs);
        let escale = wscales.add((expert as u64).wrapping_mul(ses as i64 as u64) as usize);
        let in_row = if has_topk != 0 {
            input.add((token as i64 as u64).wrapping_mul(topk as i64 as u64).wrapping_mul(Ku).wrapping_add((slot as i64 as u64).wrapping_mul(Ku)) as usize)
        } else {
            input.add((token as i64 as u64).wrapping_mul(Ku) as usize)
        };
        let sro = (n_idx / bsy).wrapping_mul(srs);
        let mut acc = 0f32;
        let k_al = (K / 128) * 128;
        let mut kb = 0i32;
        while kb < k_al {
            let k = kb + lane * 4;
            let w4 = *(w_row.offset(k as isize) as *const u32);
            let i0 = T::ldf(in_row, k as isize);
            let i1 = T::ldf(in_row, k as isize + 1);
            let i2 = T::ldf(in_row, k as isize + 2);
            let i3 = T::ldf(in_row, k as isize + 3);
            let s = *escale.offset(sro.wrapping_add(k / bsx) as isize);
            let d = fma(i0, fp8_to_f32(w4 as u8), mul(i1, fp8_to_f32((w4 >> 8) as u8)));
            let d = fma(i2, fp8_to_f32((w4 >> 16) as u8), d);
            let d = fma(i3, fp8_to_f32((w4 >> 24) as u8), d);
            acc = fma(s, d, acc);
            kb += 128;
        }
        let mut k = k_al + lane;
        while k < K {
            let iv = T::ldf(in_row, k as isize);
            let wv = fp8_to_f32(*w_row.offset(k as isize));
            let s = *escale.offset(sro.wrapping_add(k / bsx) as isize);
            acc = fma(mul(s, iv), wv, acc);
            k += 32;
        }
        let mut off = 16;
        while off > 0 {
            acc = add(acc, warp::shuffle_down_f32_sync(0xffff_ffff, acc, off));
            off /= 2;
        }
        if lane == 0 {
            let oi = (token as i64 as u64).wrapping_mul(topk as i64 as u64).wrapping_mul(Nu).wrapping_add((slot as i64 as u64).wrapping_mul(Nu)).wrapping_add(n_idx as i64 as u64);
            T::stf(output, oi as isize, acc);
        }
    }

    // ------------------------------------------------------------------ f16 / f16x2 (u16 / u32 bits)
    #[inline(always)]
    pub fn hfma2(a: u32, b: u32, c: u32) -> u32 {
        let r: u32;
        unsafe { ptx_asm!("fma.rn.f16x2 %0, %1, %2, %3;", out("=r") r, in("r") a, in("r") b, in("r") c, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn hadd2(a: u32, b: u32) -> u32 {
        let r: u32;
        unsafe { ptx_asm!("add.rn.f16x2 %0, %1, %2;", out("=r") r, in("r") a, in("r") b, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn hmul2(a: u32, b: u32) -> u32 {
        let r: u32;
        unsafe { ptx_asm!("mul.rn.f16x2 %0, %1, %2;", out("=r") r, in("r") a, in("r") b, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn hfma(a: u16, b: u16, c: u16) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("fma.rn.f16 %0, %1, %2, %3;", out("=h") r, in("h") a, in("h") b, in("h") c, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn hadd(a: u16, b: u16) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("add.rn.f16 %0, %1, %2;", out("=h") r, in("h") a, in("h") b, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn hsub(a: u16, b: u16) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("sub.rn.f16 %0, %1, %2;", out("=h") r, in("h") a, in("h") b, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn hmul(a: u16, b: u16) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("mul.rn.f16 %0, %1, %2;", out("=h") r, in("h") a, in("h") b, options(register_only)); }
        r
    }
    /// `__int2half_rn`.
    #[inline(always)]
    pub fn i2h(x: i32) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("cvt.rn.f16.s32 %0, %1;", out("=h") r, in("r") x, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn h2(lo: u16, hi: u16) -> u32 {
        lo as u32 | ((hi as u32) << 16)
    }
    #[inline(always)]
    pub fn lo(x: u32) -> u16 {
        x as u16
    }
    #[inline(always)]
    pub fn hi(x: u32) -> u16 {
        (x >> 16) as u16
    }
    #[inline(always)]
    pub unsafe fn red_add_h2(p: *mut u32, v: u32) {
        ptx_asm!("red.global.add.noftz.f16x2 [%0], %1;", in("l") p as u64, in("r") v, clobber("memory"));
    }
    #[inline(always)]
    pub unsafe fn red_add_h(p: *mut u16, v: u16) {
        ptx_asm!("red.global.add.noftz.f16 [%0], %1;", in("l") p as u64, in("h") v, clobber("memory"));
    }

    // ------------------------------------------------------------------ GPTQ (exllama q_gemm.cu)
    /// `MatrixView_q{2,3,4,8}_row::item4(items, row, column)` (q8 keeps the upstream `* 2` shift).
    #[inline(always)]
    pub unsafe fn gq_item4<const BITS: i32>(d: *const u32, row: i32, col: i32, width: i32) -> [i32; 4] {
        let w = |i: i32| *d.offset(i as isize);
        let v: u32 = match BITS {
            2 => w(row.wrapping_mul(width) / 16 + col / 16) >> ((col & 0x0f) * 2),
            4 => w(row.wrapping_mul(width) / 8 + col / 8) >> ((col & 0x07) * 4),
            8 => w(row.wrapping_mul(width) / 4 + col / 4) >> ((col & 0x03) * 2),
            _ => {
                let shift = col & 0x1f;
                let base = row.wrapping_mul(width) / 32 * 3 + col * 3 / 32;
                if shift <= 4 {
                    w(base) >> (shift * 3)
                } else if shift == 8 {
                    (w(base) >> 24) | ((w(base + 1) & 0x0f) << 8)
                } else if shift <= 16 {
                    w(base) >> (shift * 3 - 32)
                } else if shift == 20 {
                    (w(base) >> 28) | ((w(base + 1) & 0xff) << 4)
                } else {
                    w(base) >> (shift * 3 - 64)
                }
            }
        };
        let (m, s) = match BITS { 2 => (3u32, 2), 4 => (0xf, 4), 8 => (0xff, 8), _ => (7, 3) };
        [(v & m) as i32, ((v >> s) & m) as i32, ((v >> (2 * s)) & m) as i32, ((v >> (3 * s)) & m) as i32]
    }
    /// `MatrixView_q{2,3,4,8}_row::item(row, column)`.
    #[inline(always)]
    pub unsafe fn gq_item<const BITS: i32>(d: *const u32, row: i32, col: i32, width: i32) -> i32 {
        let w = |i: i32| *d.offset(i as isize);
        (match BITS {
            2 => (w(row.wrapping_mul(width) / 16 + col / 16) >> ((col & 0x0f) * 2)) & 0x03,
            4 => (w(row.wrapping_mul(width) / 8 + col / 8) >> ((col & 0x07) * 4)) & 0x0f,
            8 => (w(row.wrapping_mul(width) / 4 + col / 4) >> ((col & 0x03) * 8)) & 0xff,
            _ => {
                let z_w = col * 3 / 32;
                let z_mod = col & 0x1f;
                let base = row.wrapping_mul(width).wrapping_mul(3) / 32;
                if z_mod == 10 {
                    (w(base + z_w) >> 30) | ((w(base + z_w + 1) << 2) & 0x4)
                } else if z_mod == 21 {
                    (w(base + z_w) >> 31) | ((w(base + z_w + 1) << 1) & 0x6)
                } else if z_mod < 10 {
                    (w(base + z_w) >> (z_mod * 3)) & 0x07
                } else if z_mod < 21 {
                    (w(base + z_w) >> (z_mod * 3 - 32)) & 0x07
                } else {
                    (w(base + z_w) >> (z_mod * 3 - 64)) & 0x07
                }
            }
        }) as i32
    }
    /// `dequant_4bit_8_prep_zero(zero)`: (z1, z16) as half2 splats.
    #[inline(always)]
    pub fn prep_zero4(zero: u32) -> (u32, u32) {
        let z1 = (0xe400 | zero) as u16;
        let z16 = hsub(i2h(-64), i2h(zero as i32));
        (h2(z1, z1), h2(z16, z16))
    }
    /// `dequant_4bit_8_gptq(q, dq, z1z16, y1y16, _, false)`.
    #[inline(always)]
    pub fn dq4(q: u32, z1: u32, z16: u32) -> [u32; 4] {
        let c0 = 0x6400_6400u32;
        let y16 = h2(0x2c00, 0x2c00);
        let qa = q >> 8;
        [
            hadd2((q & 0x000f_000f) | c0, z1),
            hfma2((q & 0x00f0_00f0) | c0, y16, z16),
            hadd2((qa & 0x000f_000f) | c0, z1),
            hfma2((qa & 0x00f0_00f0) | c0, y16, z16),
        ]
    }
    /// `dequant_2bit_16(q, dq, _, zero)`.
    #[inline(always)]
    pub fn dq2(q: u32, zero: u32) -> [u32; 8] {
        let c0 = 0x6400_6400u32;
        let (y4, y16, y64) = (h2(0x3400, 0x3400), h2(0x2c00, 0x2c00), h2(0x2400, 0x2400));
        let z1h = (0xe400 | zero) as u16;
        let z1 = h2(z1h, z1h);
        let zi = i2h(zero as i32);
        let (a, b, c) = (hsub(i2h(-256), zi), hsub(i2h(-64), zi), hsub(i2h(-16), zi));
        let (z4, z16, z64) = (h2(a, a), h2(b, b), h2(c, c));
        let qb = q >> 8;
        [
            hadd2((q & 0x0003_0003) | c0, z1),
            hfma2((q & 0x000c_000c) | c0, y4, z4),
            hfma2((q & 0x0030_0030) | c0, y16, z16),
            hfma2((q & 0x00c0_00c0) | c0, y64, z64),
            hadd2((qb & 0x0003_0003) | c0, z1),
            hfma2((qb & 0x000c_000c) | c0, y4, z4),
            hfma2((qb & 0x0030_0030) | c0, y16, z16),
            hfma2((qb & 0x00c0_00c0) | c0, y64, z64),
        ]
    }
    /// `dequant_3bit_32(q0, q1, q2, dq, _, zero)`.
    #[inline(always)]
    pub fn dq3(q0: u32, q1: u32, q2: u32, zero: u32) -> [u32; 16] {
        let c0 = 0x6400_6400u32;
        let (y8, y64) = (h2(0x3000, 0x3000), h2(0x2400, 0x2400));
        let z1h = (0xe400 | zero) as u16;
        let z1 = h2(z1h, z1h);
        let zi = i2h(zero as i32);
        let (a, b) = (hsub(i2h(-128), zi), hsub(i2h(-16), zi));
        let (z8, z64) = (h2(a, a), h2(b, b));
        let (mut qa, mut qb, mut qc) = (q0, q1, q2);
        let v0 = (qa & 0x0007_0007) | c0;
        let v1 = (qa & 0x0038_0038) | c0;
        qa >>= 6;
        let v2 = (qa & 0x0007_0007) | c0;
        let v3 = (qa & 0x0038_0038) | c0;
        let v4 = (qa & 0x01c0_01c0) | c0;
        qa >>= 9;
        qa &= 0x0001_0001;
        let v5 = (qb & 0x0007_0007) | c0;
        let v6 = (qb & 0x0038_0038) | c0;
        qb >>= 6;
        let v7 = (qb & 0x0007_0007) | c0;
        let v8 = (qb & 0x0038_0038) | c0;
        let v9 = (qb & 0x01c0_01c0) | c0;
        qb >>= 8;
        qb &= 0x0002_0002;
        let v10 = (qc & 0x0007_0007) | c0;
        let v11 = (qc & 0x0038_0038) | c0;
        qc >>= 6;
        let v12 = (qc & 0x0007_0007) | c0;
        let v13 = (qc & 0x0038_0038) | c0;
        let v14 = (qc & 0x01c0_01c0) | c0;
        qc >>= 7;
        qc &= 0x0004_0004;
        let v15 = (qa | qb | qc) | c0;
        [
            hadd2(v0, z1), hfma2(v1, y8, z8), hadd2(v2, z1), hfma2(v3, y8, z8), hfma2(v4, y64, z64),
            hadd2(v5, z1), hfma2(v6, y8, z8), hadd2(v7, z1), hfma2(v8, y8, z8), hfma2(v9, y64, z64),
            hadd2(v10, z1), hfma2(v11, y8, z8), hadd2(v12, z1), hfma2(v13, y8, z8), hfma2(v14, y64, z64),
            hadd2(v15, z1),
        ]
    }
    /// `dequant_8bit_8(q0, q1, dq, _, zero)`.
    #[inline(always)]
    pub fn dq8(q0: u32, q1: u32, zero: u32) -> [u32; 4] {
        let z = zero as i32;
        let e = |q: u32, i: u32| i2h(((q >> (i * 8)) & 0xff) as i32 - z);
        [h2(e(q0, 0), e(q0, 1)), h2(e(q0, 2), e(q0, 3)), h2(e(q1, 0), e(q1, 1)), h2(e(q1, 2), e(q1, 3))]
    }
    /// hfma2 chain over `N` half2 of dq against `a` (shared, half2-aligned), starting from +0.
    #[inline(always)]
    pub unsafe fn hdot<const N: usize>(dq: &[u32; N], a: *const u16) -> u32 {
        let a2 = a as *const u32;
        let mut r = 0u32;
        let mut i = 0;
        while i < N {
            r = hfma2(dq[i], *a2.add(i), r);
            i += 1;
        }
        r
    }

    #[inline(always)]
    pub unsafe fn gptq_gemm<const BITS: i32, const MC: usize>(
        a: *const u16, b_q_weight: *const u32, qzeros: *const u32, scales: *const u16, c: *mut u16, size_m: i32, size_n: i32, size_k: i32,
        groups: i32, perm: *const i32,
    ) {
        static mut BLOCK_A: SharedArray<u16, 1024, 16> = SharedArray::UNINIT;
        let block_a = SharedArray::as_raw_mut_ptr(&raw mut BLOCK_A);
        let _ = size_m;
        let t = thread::threadIdx_x() as i32;
        let offset_n = (thread::blockIdx_x() as i32).wrapping_mul(512);
        let offset_m = (thread::blockIdx_y() as i32).wrapping_mul(MC as i32);
        let offset_k = (thread::blockIdx_z() as i32).wrapping_mul(128);
        let end_k = if offset_k + 128 < size_k { offset_k + 128 } else { size_k };
        let n = offset_n + t * 4;
        if offset_k + t < end_k {
            let mut m = 0;
            while m < MC {
                let a_ptr = a.offset((offset_m + m as i32).wrapping_mul(size_k) as isize);
                let a0 = if !perm.is_null() { *a_ptr.offset(*perm.offset((offset_k + t) as isize) as isize) } else { *a_ptr.offset((offset_k + t) as isize) };
                *block_a.add(m * 128 + t as usize) = a0;
                m += 1;
            }
        }
        if n >= size_n {
            return;
        }
        if thread::blockIdx_z() == 0 {
            let mut m = 0;
            while m < MC {
                *(c.offset((offset_m + m as i32).wrapping_mul(size_n).wrapping_add(n) as isize) as *mut u64) = 0;
                m += 1;
            }
        }
        thread::sync_threads();
        let groupsize = size_k / groups;
        let mut group = offset_k / groupsize;
        let mut nextgroup = offset_k + groupsize;
        let qk = if BITS == 3 { offset_k / 32 * 3 } else { offset_k / (32 / BITS) };
        let mut b_ptr = b_q_weight.offset(qk.wrapping_mul(size_n).wrapping_add(n) as isize);
        let mut a_off = 0usize;
        let mut zeros = gq_item4::<BITS>(qzeros, group, n, size_n);
        let sp = scales.offset(group.wrapping_mul(size_n).wrapping_add(n) as isize);
        let mut sc = [*sp, *sp.add(1), *sp.add(2), *sp.add(3)];
        let mut z4 = [(0u32, 0u32); 4];
        if BITS == 4 {
            let mut v = 0;
            while v < 4 {
                z4[v] = prep_zero4(zeros[v] as u32 + 1);
                v += 1;
            }
        }
        let mut cf = [[0f32; 4]; MC];
        let mut ch = [[0u16; 4]; MC];
        let mut k = offset_k;
        while k < end_k {
            if k == nextgroup {
                group += 1;
                nextgroup += groupsize;
                zeros = gq_item4::<BITS>(qzeros, group, n, size_n);
                let sp = scales.offset(group.wrapping_mul(size_n).wrapping_add(n) as isize);
                sc = [*sp, *sp.add(1), *sp.add(2), *sp.add(3)];
                if BITS == 4 {
                    let mut v = 0;
                    while v < 4 {
                        z4[v] = prep_zero4(zeros[v] as u32 + 1);
                        v += 1;
                    }
                }
            }
            if BITS == 4 {
                let mut j = 0;
                while j < 4 {
                    let l = *(b_ptr as *const [u32; 4]);
                    let mut v = 0;
                    while v < 4 {
                        let dq = dq4(l[v], z4[v].0, z4[v].1);
                        let s = h2f(sc[v]);
                        let mut m = 0;
                        while m < MC {
                            let r = hdot::<4>(&dq, block_a.add(m * 128 + a_off));
                            let rf = add(h2f(lo(r)), h2f(hi(r)));
                            cf[m][v] = fma(rf, s, cf[m][v]);
                            m += 1;
                        }
                        v += 1;
                    }
                    b_ptr = b_ptr.offset(size_n as isize);
                    a_off += 8;
                    j += 1;
                }
                k += 32;
            } else if BITS == 2 {
                let l = *(b_ptr as *const [u32; 4]);
                let mut v = 0;
                while v < 4 {
                    let dq = dq2(l[v], zeros[v] as u32 + 1);
                    let mut m = 0;
                    while m < MC {
                        let r = hdot::<8>(&dq, block_a.add(m * 128 + a_off));
                        ch[m][v] = hfma(hadd(lo(r), hi(r)), sc[v], ch[m][v]);
                        m += 1;
                    }
                    v += 1;
                }
                b_ptr = b_ptr.offset(size_n as isize);
                a_off += 16;
                k += 16;
            } else if BITS == 3 {
                let l0 = *(b_ptr as *const [u32; 4]);
                b_ptr = b_ptr.offset(size_n as isize);
                let l1 = *(b_ptr as *const [u32; 4]);
                b_ptr = b_ptr.offset(size_n as isize);
                let l2 = *(b_ptr as *const [u32; 4]);
                b_ptr = b_ptr.offset(size_n as isize);
                let mut v = 0;
                while v < 4 {
                    let dq = dq3(l0[v], l1[v], l2[v], zeros[v] as u32 + 1);
                    let mut m = 0;
                    while m < MC {
                        let r = hdot::<16>(&dq, block_a.add(m * 128 + a_off));
                        ch[m][v] = hfma(hadd(lo(r), hi(r)), sc[v], ch[m][v]);
                        m += 1;
                    }
                    v += 1;
                }
                a_off += 32;
                k += 32;
            } else {
                let mut j = 0;
                while j < 4 {
                    let l0 = *(b_ptr as *const [u32; 4]);
                    b_ptr = b_ptr.offset(size_n as isize);
                    let l1 = *(b_ptr as *const [u32; 4]);
                    b_ptr = b_ptr.offset(size_n as isize);
                    let mut v = 0;
                    while v < 4 {
                        let dq = dq8(l0[v], l1[v], zeros[v] as u32 + 1);
                        let qs = h2f(sc[v]);
                        let mut m = 0;
                        while m < MC {
                            let ap = block_a.add(m * 128 + a_off);
                            let mut r = 0f32;
                            let mut i = 0;
                            while i < 4 {
                                r = fma(h2f(lo(dq[i])), h2f(*ap.add(2 * i)), r);
                                r = fma(h2f(hi(dq[i])), h2f(*ap.add(2 * i + 1)), r);
                                i += 1;
                            }
                            ch[m][v] = hadd(f2h(mul(r, qs)), ch[m][v]);
                            m += 1;
                        }
                        v += 1;
                    }
                    a_off += 8;
                    j += 1;
                }
                k += 32;
            }
        }
        let mut m = 0;
        while m < MC {
            let out = c.offset((offset_m + m as i32).wrapping_mul(size_n).wrapping_add(n) as isize) as *mut u32;
            let (r01, r23) = if BITS == 4 {
                (h2(f2h(cf[m][0]), f2h(cf[m][1])), h2(f2h(cf[m][2]), f2h(cf[m][3])))
            } else {
                (h2(ch[m][0], ch[m][1]), h2(ch[m][2], ch[m][3]))
            };
            red_add_h2(out, r01);
            red_add_h2(out.add(1), r23);
            m += 1;
        }
    }

    #[inline(always)]
    pub unsafe fn reconstruct_exllama<const BITS: i32>(
        b_q_weight: *const u32, b_q_perm: *const i32, qzeros: *const u32, scales: *const u16, size_k: i32, size_n: i32, groups: i32, b: *mut u16,
    ) {
        static mut PERM: SharedArray<i32, 128> = SharedArray::UNINIT;
        let perm = SharedArray::as_raw_mut_ptr(&raw mut PERM);
        let offset_k = 128i32.wrapping_mul(thread::blockIdx_y() as i32);
        let offset_n = (thread::blockIdx_x() as i32).wrapping_mul(512);
        let end_k = if offset_k + 128 < size_k { offset_k + 128 } else { size_k };
        let t = thread::threadIdx_x() as i32;
        if !b_q_perm.is_null() && offset_k + t < size_k {
            *perm.add(t as usize) = *b_q_perm.offset((offset_k + t) as isize);
        }
        let n = offset_n + t * 4;
        if n >= size_n {
            return;
        }
        let groupsize = size_k / groups;
        let mut group = offset_k / groupsize;
        let mut nextgroup = offset_k + groupsize;
        let qk = if BITS == 3 { offset_k / 32 * 3 } else { offset_k / (32 / BITS) };
        let mut b_ptr = b_q_weight.offset(qk.wrapping_mul(size_n).wrapping_add(n) as isize);
        let mut zeros = gq_item4::<BITS>(qzeros, group, n, size_n);
        let sp = scales.offset(group.wrapping_mul(size_n).wrapping_add(n) as isize);
        let mut sc = [h2(*sp, *sp), h2(*sp.add(1), *sp.add(1)), h2(*sp.add(2), *sp.add(2)), h2(*sp.add(3), *sp.add(3))];
        let mut z4 = [(0u32, 0u32); 4];
        if BITS == 4 {
            let mut v = 0;
            while v < 4 {
                z4[v] = prep_zero4(zeros[v] as u32 + 1);
                v += 1;
            }
        }
        thread::sync_threads();
        let mut k = offset_k;
        let mut lk = 0i32;
        let has_perm = !b_q_perm.is_null();
        // Writes one dequantized half2 group (4 columns x 2 rows).
        let put = |lk: &mut i32, d: [u32; 4], sc: &[u32; 4]| {
            let d = [hmul2(sc[0], d[0]), hmul2(sc[1], d[1]), hmul2(sc[2], d[2]), hmul2(sc[3], d[3])];
            let r0 = if has_perm { *perm.add(*lk as usize) } else { offset_k + *lk };
            *lk += 1;
            let p0 = b.offset(r0.wrapping_mul(size_n).wrapping_add(n) as isize) as *mut u32;
            *p0 = h2(lo(d[0]), lo(d[1]));
            *p0.add(1) = h2(lo(d[2]), lo(d[3]));
            let r1 = if has_perm { *perm.add(*lk as usize) } else { offset_k + *lk };
            *lk += 1;
            let p1 = b.offset(r1.wrapping_mul(size_n).wrapping_add(n) as isize) as *mut u32;
            *p1 = h2(hi(d[0]), hi(d[1]));
            *p1.add(1) = h2(hi(d[2]), hi(d[3]));
        };
        while k < end_k {
            if k == nextgroup {
                group += 1;
                nextgroup += groupsize;
                zeros = gq_item4::<BITS>(qzeros, group, n, size_n);
                let sp = scales.offset(group.wrapping_mul(size_n).wrapping_add(n) as isize);
                sc = [h2(*sp, *sp), h2(*sp.add(1), *sp.add(1)), h2(*sp.add(2), *sp.add(2)), h2(*sp.add(3), *sp.add(3))];
                if BITS == 4 {
                    let mut v = 0;
                    while v < 4 {
                        z4[v] = prep_zero4(zeros[v] as u32 + 1);
                        v += 1;
                    }
                }
            }
            if BITS == 4 {
                let mut p = 0;
                while p < 4 {
                    let l = *(b_ptr as *const [u32; 4]);
                    let d = [dq4(l[0], z4[0].0, z4[0].1), dq4(l[1], z4[1].0, z4[1].1), dq4(l[2], z4[2].0, z4[2].1), dq4(l[3], z4[3].0, z4[3].1)];
                    b_ptr = b_ptr.offset(size_n as isize);
                    let mut j = 0;
                    while j < 4 {
                        put(&mut lk, [d[0][j], d[1][j], d[2][j], d[3][j]], &sc);
                        j += 1;
                    }
                    p += 1;
                }
            } else if BITS == 2 {
                let mut p = 0;
                while p < 2 {
                    let l = *(b_ptr as *const [u32; 4]);
                    let d = [dq2(l[0], zeros[0] as u32 + 1), dq2(l[1], zeros[1] as u32 + 1), dq2(l[2], zeros[2] as u32 + 1), dq2(l[3], zeros[3] as u32 + 1)];
                    b_ptr = b_ptr.offset(size_n as isize);
                    let mut j = 0;
                    while j < 8 {
                        put(&mut lk, [d[0][j], d[1][j], d[2][j], d[3][j]], &sc);
                        j += 1;
                    }
                    p += 1;
                }
            } else if BITS == 3 {
                let l0 = *(b_ptr as *const [u32; 4]);
                b_ptr = b_ptr.offset(size_n as isize);
                let l1 = *(b_ptr as *const [u32; 4]);
                b_ptr = b_ptr.offset(size_n as isize);
                let l2 = *(b_ptr as *const [u32; 4]);
                b_ptr = b_ptr.offset(size_n as isize);
                let d = [
                    dq3(l0[0], l1[0], l2[0], zeros[0] as u32 + 1),
                    dq3(l0[1], l1[1], l2[1], zeros[1] as u32 + 1),
                    dq3(l0[2], l1[2], l2[2], zeros[2] as u32 + 1),
                    dq3(l0[3], l1[3], l2[3], zeros[3] as u32 + 1),
                ];
                let mut j = 0;
                while j < 16 {
                    put(&mut lk, [d[0][j], d[1][j], d[2][j], d[3][j]], &sc);
                    j += 1;
                }
            } else {
                let mut p = 0;
                while p < 4 {
                    let l0 = *(b_ptr as *const [u32; 4]);
                    b_ptr = b_ptr.offset(size_n as isize);
                    let l1 = *(b_ptr as *const [u32; 4]);
                    b_ptr = b_ptr.offset(size_n as isize);
                    let d = [dq8(l0[0], l1[0], zeros[0] as u32 + 1), dq8(l0[1], l1[1], zeros[1] as u32 + 1), dq8(l0[2], l1[2], zeros[2] as u32 + 1), dq8(l0[3], l1[3], zeros[3] as u32 + 1)];
                    let mut j = 0;
                    while j < 4 {
                        put(&mut lk, [d[0][j], d[1][j], d[2][j], d[3][j]], &sc);
                        j += 1;
                    }
                    p += 1;
                }
            }
            k += 32;
        }
    }

    #[inline(always)]
    pub unsafe fn gptq_alt<const BITS: i32>(
        vec: *const u32, mat: *const u32, mul_out: *mut u16, scales: *const u16, zeros: *const u32, g_idx: *const i32, batch: i32, height: i32,
        width: i32,
    ) {
        static mut BLOCKVEC: SharedArray<u32, 512> = SharedArray::UNINIT;
        let blockvec = SharedArray::as_raw_mut_ptr(&raw mut BLOCKVEC);
        let per = 32 / BITS; // values per u32
        let zero_width = width / per;
        let vec_height = height * (per / 2);
        let bx = thread::blockIdx_x() as i32;
        let bz = thread::blockIdx_z() as i32;
        let tx = thread::threadIdx_x() as i32;
        let b = (thread::blockIdx_y() as i32).wrapping_mul(8);
        let b_end = if 8 < batch - b { 8 } else { batch - b };
        let h = 128i32.wrapping_mul(bz) / per;
        let h_end = (if 128 / per < height - h { 128 / per } else { height - h }) * (per / 2);
        let w = 128i32.wrapping_mul(bx).wrapping_add(tx);
        if tx < h_end {
            let mut m = 0;
            while m < b_end {
                *blockvec.add((m * 64 + tx) as usize) = *vec.offset(((m + b).wrapping_mul(vec_height).wrapping_add(bz.wrapping_mul(64)).wrapping_add(tx)) as isize);
                m += 1;
            }
        }
        if bz == 0 {
            let mut m = 0;
            while m < b_end {
                *mul_out.offset((b + m).wrapping_mul(width).wrapping_add(w) as isize) = 0;
                m += 1;
            }
        }
        thread::sync_threads();
        let mut i = width.wrapping_mul(h).wrapping_add(w);
        let g_h = h * per;
        let mut k = 0;
        let z_w = w / per;
        let z_mod = (w % per) * BITS;
        let mut res = [0u16; 8];
        let mask = if BITS == 4 { 0xf } else { 0xff };
        let nsub = (per / 2) as usize; // half2 steps per u32
        while k < h_end {
            let tmp = *mat.offset(i as isize);
            let mut st = [0u32; 4];
            let mut zt = [0u32; 4];
            let mut tk = 0usize;
            while tk < nsub {
                let g = *g_idx.offset((g_h + (k + tk as i32) * 2) as isize);
                let g2 = *g_idx.offset((g_h + (k + tk as i32) * 2 + 1) as isize);
                let sf = *scales.offset(g.wrapping_mul(width).wrapping_add(w) as isize);
                let sf2 = *scales.offset(g2.wrapping_mul(width).wrapping_add(w) as isize);
                let z1 = ((*zeros.offset(g.wrapping_mul(zero_width).wrapping_add(z_w) as isize) >> z_mod) & mask) as i32;
                let z2 = ((*zeros.offset(g2.wrapping_mul(zero_width).wrapping_add(z_w) as isize) >> z_mod) & mask) as i32;
                st[tk] = h2(sf, sf2);
                zt[tk] = h2(hmul(sf, i2h(-z1 - 1)), hmul(sf2, i2h(-z2 - 1)));
                tk += 1;
            }
            let mut m = 0;
            while m < b_end {
                let mut res2 = 0u32;
                let mut tk = 0usize;
                while tk < nsub {
                    let v = if BITS == 4 {
                        let byte = (tmp >> (8 * tk)) & 0xff;
                        h2(i2h((byte & 0xf) as i32), i2h((byte >> 4) as i32))
                    } else {
                        h2(i2h(((tmp >> (16 * tk)) & 0xff) as i32), i2h(((tmp >> (16 * tk + 8)) & 0xff) as i32))
                    };
                    res2 = hfma2(hfma2(v, st[tk], zt[tk]), *blockvec.add((m * 64 + k + tk as i32) as usize), res2);
                    tk += 1;
                }
                res[m as usize] = hadd(res[m as usize], hadd(lo(res2), hi(res2)));
                m += 1;
            }
            i = i.wrapping_add(width);
            k += nsub as i32;
        }
        let mut m = 0;
        while m < b_end {
            red_add_h(mul_out.offset((b + m).wrapping_mul(width).wrapping_add(w) as isize), res[m as usize]);
            m += 1;
        }
    }

    #[inline(always)]
    pub unsafe fn reconstruct_gptq<const BITS: i32>(w: *const u32, w_scales: *const u16, w_zeros: *const u32, g_idx: *const i32, height: i32, width: i32, group: i32, out: *mut u16) {
        let _ = (height, group);
        let column = 128i32.wrapping_mul(thread::blockIdx_x() as i32).wrapping_add(thread::threadIdx_x() as i32);
        let by = thread::blockIdx_y() as i32;
        if column >= width {
            return;
        }
        if BITS == 3 {
            let row = by * 32;
            let w1 = *w.offset(((by * 3).wrapping_mul(width).wrapping_add(column)) as isize);
            let w2 = *w.offset(((by * 3 + 1).wrapping_mul(width).wrapping_add(column)) as isize);
            let w3 = *w.offset(((by * 3 + 2).wrapping_mul(width).wrapping_add(column)) as isize);
            let mut op = out.offset(row.wrapping_mul(width).wrapping_add(column) as isize);
            let mut i = 0i32;
            while i < 32 {
                let g = *g_idx.offset((row + i) as isize);
                let ws = *w_scales.offset(g.wrapping_mul(width).wrapping_add(column) as isize);
                let wz = (gq_item::<3>(w_zeros, g, column, width) as u32).wrapping_add(1);
                let wi: u32 = if i == 10 {
                    (w1 >> 30) | ((w2 << 2) & 0x4)
                } else if i == 21 {
                    (w2 >> 31) | ((w3 << 1) & 0x6)
                } else if i < 10 {
                    (w1 >> (i * 3)) & 0x7
                } else if i < 21 {
                    (w2 >> (i * 3 - 32)) & 0x7
                } else {
                    (w3 >> (i * 3 - 64)) & 0x7
                };
                *op = hmul(i2h(wi.wrapping_sub(wz) as i32), ws);
                op = op.offset(width as isize);
                i += 1;
            }
        } else {
            let row = by * 32 / BITS;
            let wr = *w.offset(by.wrapping_mul(width).wrapping_add(column) as isize);
            let mut op = out.offset(row.wrapping_mul(width).wrapping_add(column) as isize);
            let mut s = 0i32;
            while s < 32 {
                let g = *g_idx.offset((row + s / BITS) as isize);
                let ws = *w_scales.offset(g.wrapping_mul(width).wrapping_add(column) as isize);
                let wz = (gq_item::<BITS>(w_zeros, g, column, width) as u32).wrapping_add(1);
                let q = (wr >> s) & ((1u32 << BITS) - 1);
                *op = hmul(i2h(q.wrapping_sub(wz) as i32), ws);
                op = op.offset(width as isize);
                s += BITS;
            }
        }
    }

    /// `shuffle_{2,3,4,8}bit_kernel` (in-place; 8-bit is a no-op).
    #[inline(always)]
    pub unsafe fn gptq_shuffle<const BITS: i32>(b_q_weight: *mut u32, size_k: i32, size_n: i32) {
        let n = (thread::blockIdx_x() as i32).wrapping_mul(32).wrapping_add(thread::threadIdx_x() as i32);
        if n >= size_n || BITS == 8 {
            return;
        }
        let mut k = 0;
        let mut p = b_q_weight.offset(n as isize);
        while k < size_k {
            if BITS == 4 {
                let mut qa = *p;
                let mut qb = 0u32;
                let mut i = 0;
                while i < 4 {
                    let qa0 = qa & 0x0f;
                    let qa1 = (qa & 0xf0) >> 4;
                    qa >>= 8;
                    qb |= qa1 << (i * 4 + 16);
                    qb |= qa0 << (i * 4);
                    i += 1;
                }
                *p = qb;
                p = p.offset(size_n as isize);
                k += 8;
            } else if BITS == 2 {
                let mut qa = *p;
                let mut qb = 0u32;
                let mut i = 0;
                while i < 8 {
                    let qa0 = qa & 0x03;
                    let qa1 = (qa & 0x0c) >> 2;
                    qa >>= 4;
                    qb |= qa1 << (i * 2 + 16);
                    qb |= qa0 << (i * 2);
                    i += 1;
                }
                *p = qb;
                p = p.offset(size_n as isize);
                k += 16;
            } else {
                let s = size_n as isize;
                let mut qa = *p;
                let mut qb = *p.offset(s);
                let mut qc = *p.offset(2 * s);
                let qd = qc >> 26;
                qc <<= 4;
                qc |= qb >> 28;
                qb <<= 2;
                qb |= qa >> 30;
                let (mut za, mut zb, mut zc) = (0u32, 0u32, 0u32);
                let mut i = 0;
                while i < 5 {
                    za |= (qa & 0x07) << (i * 3);
                    za |= ((qa & 0x38) >> 3) << (i * 3 + 16);
                    qa >>= 6;
                    zb |= (qb & 0x07) << (i * 3);
                    zb |= ((qb & 0x38) >> 3) << (i * 3 + 16);
                    qb >>= 6;
                    zc |= (qc & 0x07) << (i * 3);
                    zc |= ((qc & 0x38) >> 3) << (i * 3 + 16);
                    qc >>= 6;
                    i += 1;
                }
                za |= (qd & 0x01) << 15;
                zb |= ((qd & 0x02) >> 1) << 15;
                zc |= ((qd & 0x04) >> 2) << 15;
                za |= ((qd & 0x08) >> 3) << 31;
                zb |= ((qd & 0x10) >> 4) << 31;
                zc |= ((qd & 0x20) >> 5) << 31;
                *p = za;
                *p.offset(s) = zb;
                *p.offset(2 * s) = zc;
                p = p.offset(3 * s);
                k += 32;
            }
        }
    }

    /// `make_sequential_{2,4,8}bit_kernel` (64-bit words: two columns at a time).
    #[inline(always)]
    pub unsafe fn make_sequential<const BITS: i32>(w: *const u32, w_new: *mut u32, q_perm: *const i32, w_width: i32) {
        let w2 = w as *const u64;
        let wn2 = w_new as *mut u64;
        let stride = w_width >> 1;
        let col = 32i32.wrapping_mul(thread::blockIdx_x() as i32).wrapping_add(thread::threadIdx_x() as i32);
        if col >= stride {
            return;
        }
        let per = 32 / BITS;
        let row = thread::blockIdx_y() as i32;
        let mut qi = row * per;
        let mut dst = 0u64;
        let mask: u64 = ((1u64 << BITS) - 1) | (((1u64 << BITS) - 1) << 32);
        let mut i = 0;
        while i < per {
            let src_row = *q_perm.offset(qi as isize);
            qi += 1;
            let sub = src_row & (per - 1);
            let mut src = *w2.offset(r_of::<BITS>(src_row).wrapping_mul(stride).wrapping_add(col) as isize);
            src >>= sub * BITS;
            src &= mask;
            src <<= i * BITS;
            dst |= src;
            i += 1;
        }
        *wn2.offset(row.wrapping_mul(stride).wrapping_add(col) as isize) = dst;
    }
    #[inline(always)]
    pub fn r_of<const BITS: i32>(src_row: i32) -> i32 {
        if BITS == 2 { src_row >> 4 } else if BITS == 4 { src_row >> 3 } else { src_row >> 2 }
    }

    #[inline(always)]
    pub unsafe fn make_sequential_3(w: *const u32, w_new: *mut u32, q_perm: *const i32, w_width: i32) {
        let col = 32i32.wrapping_mul(thread::blockIdx_x() as i32).wrapping_add(thread::threadIdx_x() as i32);
        if col >= w_width {
            return;
        }
        let by = thread::blockIdx_y() as i32;
        let w_new_row = by * 3;
        let mut qi = by << 5;
        let mut dst = [0u32; 3];
        let mut i = 0i32;
        while i < 32 {
            let src_row = *q_perm.offset(qi as isize);
            qi += 1;
            let mut z_w = (src_row / 32) * 3;
            let z_mod = src_row % 32;
            let mut z_bit = 0i32;
            if z_mod != 10 {
                if z_mod != 21 {
                    z_bit = z_mod;
                    if z_bit > 21 {
                        z_bit = z_bit * 3 - 64;
                        z_w += 2;
                    } else if z_bit > 10 {
                        z_bit = z_bit * 3 - 32;
                        z_w += 1;
                    } else {
                        z_bit *= 3;
                    }
                } else {
                    z_w += 1;
                }
            }
            let ld = |r: i32| *w.offset(r.wrapping_mul(w_width).wrapping_add(col) as isize);
            let src: u64 = if z_mod == 10 {
                ((ld(z_w) >> 30) | ((ld(z_w + 1) << 2) & 0x4)) as u64
            } else if z_mod == 21 {
                ((ld(z_w) >> 31) | ((ld(z_w + 1) << 1) & 0x6)) as u64
            } else {
                ((ld(z_w) as u64) >> z_bit) & 0x07
            };
            let mut zw = 0usize;
            let mut zb = 0i32;
            if i != 10 {
                if i != 21 {
                    zb = i;
                    if zb > 21 {
                        zb = zb * 3 - 64;
                        zw += 2;
                    } else if zb > 10 {
                        zb = zb * 3 - 32;
                        zw += 1;
                    } else {
                        zb *= 3;
                    }
                } else {
                    zw += 1;
                }
            }
            if i == 10 {
                dst[zw] |= ((src & 0x03) << 30) as u32;
                dst[zw + 1] |= ((src & 0x4) >> 2) as u32;
            } else if i == 21 {
                dst[zw] |= ((src & 0x01) << 31) as u32;
                dst[zw + 1] |= ((src & 0x6) >> 1) as u32;
            } else {
                dst[zw] |= (src << zb) as u32;
            }
            i += 1;
        }
        *w_new.offset((w_new_row.wrapping_mul(w_width).wrapping_add(col)) as isize) = dst[0];
        *w_new.offset(((w_new_row + 1).wrapping_mul(w_width).wrapping_add(col)) as isize) = dst[1];
        *w_new.offset(((w_new_row + 2).wrapping_mul(w_width).wrapping_add(col)) as isize) = dst[2];
    }

    // ------------------------------------------------------------------ MXFP4
    /// `__byte_perm(a, b, s)`: only the low 3 bits of each selector nibble count (PTX `prmt.b32`'s
    /// default mode would sign-replicate on bit 3, which `__byte_perm` does not do).
    #[inline(always)]
    pub fn prmt(a: u32, b: u32, s: u32) -> u32 {
        let r: u32;
        let s = s & 0x7777;
        unsafe { ptx_asm!("prmt.b32 %0, %1, %2, %3;", out("=r") r, in("r") a, in("r") b, in("r") s, options(register_only)); }
        r
    }
    /// `get_int_from_table_16(q4, LUT0..3)` with the x2-scaled FP4 table.
    #[inline(always)]
    pub fn fp4_table(q4: u32) -> (u32, u32) {
        let (t0, t1, t2, t3) = (0x0302_0100u32, 0x0C08_0604u32, 0xFDFE_FF00u32, 0xF4F8_FAFCu32);
        let lhs = 0x3210_3210 | ((q4 & 0x8888_8888) >> 1);
        let mut tmp = [0u32; 2];
        let mut i = 0;
        while i < 2 {
            let sh = 16 * i as u32;
            let q = ((q4 as i32) >> sh) as u32;
            let low = prmt(t0, t1, q);
            let high = prmt(t2, t3, q);
            tmp[i] = prmt(low, high, lhs >> sh);
            i += 1;
        }
        (prmt(tmp[0], tmp[1], 0x6420), prmt(tmp[0], tmp[1], 0x7531))
    }
    /// `(float)(int8_t)b`.
    #[inline(always)]
    pub fn s8f(x: u32, byte: u32) -> f32 {
        let r: f32;
        let v = ((x >> (8 * byte)) as u8 as i8) as i32;
        unsafe { ptx_asm!("cvt.rn.f32.s32 %0, %1;", out("=f") r, in("r") v, options(register_only)); }
        r
    }
    /// `dequant_store_8`: [x.b0, y.b0, x.b1, y.b1, x.b2, y.b2, x.b3, y.b3] * scale.
    #[inline(always)]
    pub fn fp4_dq8(q4: u32, scale: f32) -> [f32; 8] {
        let (x, y) = fp4_table(q4);
        [
            mul(s8f(x, 0), scale), mul(s8f(y, 0), scale), mul(s8f(x, 1), scale), mul(s8f(y, 1), scale),
            mul(s8f(x, 2), scale), mul(s8f(y, 2), scale), mul(s8f(x, 3), scale), mul(s8f(y, 3), scale),
        ]
    }
    /// `e8m0_to_float(e) * 0.5f`.
    #[inline(always)]
    pub fn e8m0_half(e: u8) -> f32 {
        mul(f32::from_bits((e as u32) << 23), 0.5)
    }
    #[inline(always)]
    pub unsafe fn ld_u4(p: *const u8) -> [u32; 4] {
        *(p as *const [u32; 4])
    }

    #[inline(always)]
    pub unsafe fn mxfp4_vecmat<T: El>(input: *const T, weight: *const u8, wscale: *const u8, bias: *const T, output: *mut T, M: i32, N: i32, K: i32, has_bias: u8) {
        static mut RED: SharedArray<f32, 32> = SharedArray::UNINIT;
        let red = SharedArray::as_raw_mut_ptr(&raw mut RED);
        let _ = M;
        let row = thread::blockIdx_y() as i32;
        let col_base = (thread::blockIdx_x() as i32).wrapping_mul(4);
        let tid = thread::threadIdx_x() as i32;
        let (warp_id, lane) = (tid / 32, tid % 32);
        let ss = K.wrapping_add(31) / 32;
        let k_half = K / 2;
        let mut acc = [0f32; 4];
        let nkb = K / 32;
        let mut blk = tid;
        while blk < nkb {
            let k_start = blk * 32;
            let in_ptr = input.add((row as i64 as u64).wrapping_mul(K as i64 as u64).wrapping_add(k_start as i64 as u64) as usize);
            let mut iv = [0f32; 32];
            let mut i = 0;
            while i < 32 {
                iv[i] = T::ldf(in_ptr, i as isize);
                i += 1;
            }
            let mut c = 0;
            while c < 4 {
                let col = col_base + c as i32;
                if col < N {
                    let w = ld_u4(weight.add((col as i64 as u64).wrapping_mul(k_half as i64 as u64).wrapping_add((k_start / 2) as i64 as u64) as usize));
                    let s = e8m0_half(*wscale.add((col as i64 as u64).wrapping_mul(ss as i64 as u64).wrapping_add(blk as i64 as u64) as usize));
                    let mut dot = 0f32;
                    let mut q = 0;
                    while q < 4 {
                        let wv = fp4_dq8(w[q], s);
                        let mut e = 0;
                        while e < 8 {
                            dot = fma(iv[q * 8 + e], wv[e], dot);
                            e += 1;
                        }
                        q += 1;
                    }
                    acc[c] = add(acc[c], dot);
                }
                c += 1;
            }
            blk += 256;
        }
        let mut c = 0;
        while c < 4 {
            let mut off = 16;
            while off > 0 {
                acc[c] = add(acc[c], warp::shuffle_down_f32_sync(0xffff_ffff, acc[c], off));
                off /= 2;
            }
            c += 1;
        }
        if lane == 0 {
            let mut c = 0;
            while c < 4 {
                *red.add(c * 8 + warp_id as usize) = acc[c];
                c += 1;
            }
        }
        thread::sync_threads();
        if warp_id == 0 {
            let mut c = 0;
            while c < 4 {
                let mut val = if lane < 8 { *red.add(c * 8 + lane as usize) } else { 0.0 };
                let mut off = 4;
                while off > 0 {
                    val = add(val, warp::shuffle_down_f32_sync(0xffff_ffff, val, off));
                    off /= 2;
                }
                if lane == 0 {
                    let col = col_base + c as i32;
                    if col < N {
                        if has_bias != 0 && !bias.is_null() {
                            val = add(val, T::ldf(bias, col as isize));
                        }
                        T::stf(output, (row as i64 as u64).wrapping_mul(N as i64 as u64).wrapping_add(col as i64 as u64) as isize, val);
                    }
                }
                c += 1;
            }
        }
    }

    /// Register-tiled 64x64x32 GEMM body shared by `mxfp4_matmul_tiled` and the grouped MoE kernel:
    /// `row_of(lm)` maps a tile row to an input row (None = zero), `out_of(lm)` to an output row.
    #[inline(always)]
    pub unsafe fn mxfp4_tile_gemm<T: El, R: Fn(i32) -> i32, O: Fn(i32) -> i32>(
        input: *const T, wt: *const u8, ws: *const u8, bias: *const T, output: *mut T, N: i32, K: i32, has_bias: u8, n_base: i32,
        si: *mut f32, sw: *mut f32, row_of: &R, out_of: &O,
    ) {
        let tx = thread::threadIdx_x() as i32;
        let ty = thread::threadIdx_y() as i32;
        let tid = ty * 16 + tx;
        let ss = K.wrapping_add(31) / 32;
        let mut acc = [[0f32; 4]; 4];
        let mut kt = 0i32;
        while kt < K {
            let mut idx = tid;
            while idx < 2048 {
                let (lm, lk) = (idx / 32, idx % 32);
                let r = row_of(lm);
                let gk = kt + lk;
                let v = if r >= 0 && gk < K { T::ldf(input, (r as i64 as u64).wrapping_mul(K as i64 as u64).wrapping_add(gk as i64 as u64) as isize) } else { 0.0 };
                *si.add((lm * 33 + lk) as usize) = v;
                idx += 256;
            }
            let mut ln = tid;
            while ln < 64 {
                let gn = n_base + ln;
                if gn < N {
                    let w = ld_u4(wt.add((gn as i64 as u64).wrapping_mul((K / 2) as i64 as u64).wrapping_add((kt / 2) as i64 as u64) as usize));
                    let s = e8m0_half(*ws.add((gn as i64 as u64).wrapping_mul(ss as i64 as u64).wrapping_add((kt / 32) as i64 as u64) as usize));
                    let mut q = 0;
                    while q < 4 {
                        let d = fp4_dq8(w[q], s);
                        let mut e = 0;
                        while e < 8 {
                            *sw.add((ln * 33) as usize + q * 8 + e) = d[e];
                            e += 1;
                        }
                        q += 1;
                    }
                } else {
                    let mut k = 0;
                    while k < 32 {
                        *sw.add((ln * 33 + k) as usize) = 0.0;
                        k += 1;
                    }
                }
                ln += 256;
            }
            thread::sync_threads();
            let mut k = 0usize;
            while k < 32 {
                let mut a = [0f32; 4];
                let mut b = [0f32; 4];
                let mut i = 0;
                while i < 4 {
                    a[i] = *si.add((ty as usize * 4 + i) * 33 + k);
                    b[i] = *sw.add((tx as usize * 4 + i) * 33 + k);
                    i += 1;
                }
                let mut i = 0;
                while i < 4 {
                    let mut j = 0;
                    while j < 4 {
                        acc[i][j] = fma(a[i], b[j], acc[i][j]);
                        j += 1;
                    }
                    i += 1;
                }
                k += 1;
            }
            thread::sync_threads();
            kt += 32;
        }
        let mut i = 0;
        while i < 4 {
            let orow = out_of(ty * 4 + i as i32);
            if orow >= 0 {
                let mut j = 0;
                while j < 4 {
                    let col = n_base + tx * 4 + j as i32;
                    if col < N {
                        let mut val = acc[i][j];
                        if has_bias != 0 && !bias.is_null() {
                            val = add(val, T::ldf(bias, col as isize));
                        }
                        T::stf(output, (orow as i64 as u64).wrapping_mul(N as i64 as u64).wrapping_add(col as i64 as u64) as isize, val);
                    }
                    j += 1;
                }
            }
            i += 1;
        }
    }

    #[inline(always)]
    pub unsafe fn mxfp4_matmul_tiled<T: El>(input: *const T, weight: *const u8, ws: *const u8, bias: *const T, output: *mut T, M: i32, N: i32, K: i32, has_bias: u8) {
        static mut SI: SharedArray<f32, { 64 * 33 }> = SharedArray::UNINIT;
        static mut SW: SharedArray<f32, { 64 * 33 }> = SharedArray::UNINIT;
        let si = SharedArray::as_raw_mut_ptr(&raw mut SI);
        let sw = SharedArray::as_raw_mut_ptr(&raw mut SW);
        let m_base = (thread::blockIdx_y() as i32).wrapping_mul(64);
        let n_base = (thread::blockIdx_x() as i32).wrapping_mul(64);
        let row_of = |lm: i32| if m_base + lm < M { m_base + lm } else { -1 };
        mxfp4_tile_gemm(input, weight, ws, bias, output, N, K, has_bias, n_base, si, sw, &row_of, &row_of);
    }

    /// Phase 1 of the grouped MoE kernels: shared token list of the block's expert (atomic order;
    /// the outputs do not depend on it). Returns the token count.
    #[inline(always)]
    pub unsafe fn moe_token_list(indices: *const u32, total_work: i32, expert: i32, cnt: *mut i32, list: *mut i32, tid: i32, nthreads: i32) -> i32 {
        if tid == 0 {
            *cnt = 0;
        }
        thread::sync_threads();
        let sc = cuda_device::shared::cvta_generic_to_shared_u32(cnt as *const u8);
        let mut i = tid;
        while i < total_work {
            if *indices.offset(i as isize) == expert as u32 {
                let pos: i32;
                ptx_asm!("atom.shared.add.u32 %0, [%1], 1;", out("=r") pos, in("r") sc, clobber("memory"));
                *list.offset(pos as isize) = i;
            }
            i += nthreads;
        }
        thread::sync_threads();
        *(cnt as *const i32)
    }

    #[inline(always)]
    pub unsafe fn mxfp4_moe_grouped_tiled<T: El>(
        input: *const T, weights: *const u8, wscales: *const u8, biases: *const T, indices: *const u32, output: *mut T, num_tokens: i32, topk: i32,
        num_experts: i32, N: i32, K: i32, has_bias: u8, has_topk: u8,
    ) {
        static mut SI: SharedArray<f32, { 64 * 33 }> = SharedArray::UNINIT;
        static mut SW: SharedArray<f32, { 64 * 33 }> = SharedArray::UNINIT;
        static mut CNT: SharedArray<i32, 1> = SharedArray::UNINIT;
        let si = SharedArray::as_raw_mut_ptr(&raw mut SI);
        let sw = SharedArray::as_raw_mut_ptr(&raw mut SW);
        let cnt = SharedArray::as_raw_mut_ptr(&raw mut CNT);
        // Generic pointer (a shared-space pointer cannot be captured by the tile closures).
        let list = cuda_device::DynamicSharedArray::<i32>::get() as usize as *mut i32;
        let _ = num_experts;
        let tid = thread::threadIdx_y() as i32 * 16 + thread::threadIdx_x() as i32;
        let expert = thread::blockIdx_y() as i32;
        let n_base = (thread::blockIdx_x() as i32).wrapping_mul(64);
        let ss = K.wrapping_add(31) / 32;
        let total = num_tokens.wrapping_mul(topk);
        let ew = weights.add((expert as i64 as u64).wrapping_mul(N as i64 as u64).wrapping_mul((K / 2) as i64 as u64) as usize);
        let es = wscales.add((expert as i64 as u64).wrapping_mul(N as i64 as u64).wrapping_mul(ss as i64 as u64) as usize);
        let m_exp = moe_token_list(indices, total, expert, cnt, list, tid, 256);
        if m_exp == 0 {
            return;
        }
        let eb = biases.add((expert as i64 as u64).wrapping_mul(N as i64 as u64) as usize);
        let bias = if biases.is_null() { biases } else { eb };
        let mut mt = 0;
        while mt < (m_exp + 63) / 64 {
            let row_of = |lm: i32| {
                let wp = mt * 64 + lm;
                if wp < m_exp {
                    let wi = *list.offset(wp as isize);
                    if has_topk != 0 { wi } else { wi / topk }
                } else {
                    -1
                }
            };
            let out_of = |lm: i32| {
                let wp = mt * 64 + lm;
                if wp < m_exp { *list.offset(wp as isize) } else { -1 }
            };
            mxfp4_tile_gemm(input, ew, es, bias, output, N, K, has_bias, n_base, si, sw, &row_of, &out_of);
            mt += 1;
        }
    }

    #[inline(always)]
    pub unsafe fn mxfp4_moe_gemm<T: El>(
        input: *const T, weights: *const u8, wscales: *const u8, biases: *const T, indices: *const u32, output: *mut T, num_tokens: i32, topk: i32,
        num_experts: i32, N: i32, K: i32, has_bias: u8, has_topk: u8,
    ) {
        let s_in = cuda_device::DynamicSharedArray::<f32>::get();
        let tid = thread::threadIdx_x() as i32;
        let bsz = thread::blockDim_x() as i32;
        let n_chunks = N.wrapping_add(7) / 8;
        let (warp_id, lane) = (tid / 32, tid % 32);
        let wrs = K / 2;
        let ss = K.wrapping_add(31) / 32;
        let bx = thread::blockIdx_x() as i32;
        let (token, s0, s1, n_base);
        if has_topk == 0 {
            n_base = (bx % n_chunks) * 8;
            token = bx / n_chunks;
            s0 = 0;
            s1 = topk;
        } else {
            n_base = (bx % n_chunks) * 8;
            let temp = bx / n_chunks;
            s0 = temp % topk;
            s1 = s0 + 1;
            token = temp / topk;
        }
        if token >= num_tokens {
            return;
        }
        let n_idx = n_base + warp_id;
        if n_idx >= N {
            return;
        }
        let in_row = if has_topk == 0 {
            input.add((token as i64 as u64).wrapping_mul(K as i64 as u64) as usize)
        } else {
            input.add((token as i64 as u64).wrapping_mul(topk as i64 as u64).wrapping_mul(K as i64 as u64).wrapping_add((s0 as i64 as u64).wrapping_mul(K as i64 as u64)) as usize)
        };
        let mut k = tid;
        while k < K {
            *s_in.offset((k + k / 32) as isize) = T::ldf(in_row, k as isize);
            k += bsz;
        }
        thread::sync_threads();
        let mut slot = s0;
        while slot < s1 {
            let expert = *indices.offset(token.wrapping_mul(topk).wrapping_add(slot) as isize);
            if expert < num_experts as u32 {
                let (Nu, wrsu, ssu) = (N as i64 as u64, wrs as i64 as u64, ss as i64 as u64);
                let w_row = weights.add((expert as u64).wrapping_mul(Nu).wrapping_mul(wrsu).wrapping_add((n_idx as i64 as u64).wrapping_mul(wrsu)) as usize);
                let s_row = wscales.add((expert as u64).wrapping_mul(Nu).wrapping_mul(ssu).wrapping_add((n_idx as i64 as u64).wrapping_mul(ssu)) as usize);
                let mut acc0 = 0f32;
                let mut acc1 = 0f32;
                let mut k = lane * 32;
                while k < K {
                    let s = e8m0_half(*s_row.offset((k / 32) as isize));
                    let w = ld_u4(w_row.offset((k / 2) as isize));
                    let inp = s_in.offset((k + k / 32) as isize);
                    let mut q = 0;
                    while q < 4 {
                        let (we, wo) = fp4_table(w[q]);
                        let mut b = 0;
                        while b < 4 {
                            acc0 = fma(*inp.add(q * 8 + 2 * b as usize), mul(s8f(we, b), s), acc0);
                            acc1 = fma(*inp.add(q * 8 + 2 * b as usize + 1), mul(s8f(wo, b), s), acc1);
                            b += 1;
                        }
                        q += 1;
                    }
                    k += 1024;
                }
                let mut acc = add(acc0, acc1);
                let mut off = 16;
                while off > 0 {
                    acc = add(acc, warp::shuffle_down_f32_sync(0xffff_ffff, acc, off));
                    off /= 2;
                }
                if lane == 0 {
                    if has_bias != 0 && !biases.is_null() {
                        let br = biases.add((expert as u64).wrapping_mul(Nu) as usize);
                        acc = add(acc, T::ldf(br, n_idx as isize));
                    }
                    let oi = (token as i64 as u64).wrapping_mul(topk as i64 as u64).wrapping_mul(Nu).wrapping_add((slot as i64 as u64).wrapping_mul(Nu)).wrapping_add(n_idx as i64 as u64);
                    T::stf(output, oi as isize, acc);
                }
            }
            slot += 1;
        }
    }

    // ---- WMMA (m16n16k16, f32 accumulate): the same PTX wmma ops nvcc emits.
    pub trait Wmma: El {
        /// Store f32 as this 16-bit type (bits).
        fn to16(v: f32) -> u16;
        /// c += a(16x16 row-major at `a`, ld 32) * b(16x16 col-major at `b`, ld 32); addresses in shared.
        unsafe fn mma(c: &mut [f32; 8], a: u32, b: u32);
    }
    impl Wmma for H {
        #[inline(always)]
        fn to16(v: f32) -> u16 {
            f2h(v)
        }
        #[inline(always)]
        unsafe fn mma(c: &mut [f32; 8], a: u32, b: u32) {
            let (a0, a1, a2, a3, a4, a5, a6, a7): (u32, u32, u32, u32, u32, u32, u32, u32);
            let (b0, b1, b2, b3, b4, b5, b6, b7): (u32, u32, u32, u32, u32, u32, u32, u32);
            ptx_asm!("wmma.load.a.sync.aligned.row.m16n16k16.shared.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], 32;",
                out("=r") a0, out("=r") a1, out("=r") a2, out("=r") a3, out("=r") a4, out("=r") a5, out("=r") a6, out("=r") a7, in("r") a);
            ptx_asm!("wmma.load.b.sync.aligned.col.m16n16k16.shared.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], 32;",
                out("=r") b0, out("=r") b1, out("=r") b2, out("=r") b3, out("=r") b4, out("=r") b5, out("=r") b6, out("=r") b7, in("r") b);
            ptx_asm!("wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19, %20, %21, %22, %23}, {%0, %1, %2, %3, %4, %5, %6, %7};",
                inout("+f") c[0], inout("+f") c[1], inout("+f") c[2], inout("+f") c[3], inout("+f") c[4], inout("+f") c[5], inout("+f") c[6], inout("+f") c[7],
                in("r") a0, in("r") a1, in("r") a2, in("r") a3, in("r") a4, in("r") a5, in("r") a6, in("r") a7,
                in("r") b0, in("r") b1, in("r") b2, in("r") b3, in("r") b4, in("r") b5, in("r") b6, in("r") b7);
        }
    }
    impl Wmma for B {
        #[inline(always)]
        fn to16(v: f32) -> u16 {
            f2bf(v)
        }
        #[inline(always)]
        unsafe fn mma(c: &mut [f32; 8], a: u32, b: u32) {
            let (a0, a1, a2, a3): (u32, u32, u32, u32);
            let (b0, b1, b2, b3): (u32, u32, u32, u32);
            ptx_asm!("wmma.load.a.sync.aligned.row.m16n16k16.shared.bf16 {%0, %1, %2, %3}, [%4], 32;",
                out("=r") a0, out("=r") a1, out("=r") a2, out("=r") a3, in("r") a);
            ptx_asm!("wmma.load.b.sync.aligned.col.m16n16k16.shared.bf16 {%0, %1, %2, %3}, [%4], 32;",
                out("=r") b0, out("=r") b1, out("=r") b2, out("=r") b3, in("r") b);
            ptx_asm!("wmma.mma.sync.aligned.row.col.m16n16k16.f32.bf16.bf16.f32 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15}, {%0, %1, %2, %3, %4, %5, %6, %7};",
                inout("+f") c[0], inout("+f") c[1], inout("+f") c[2], inout("+f") c[3], inout("+f") c[4], inout("+f") c[5], inout("+f") c[6], inout("+f") c[7],
                in("r") a0, in("r") a1, in("r") a2, in("r") a3, in("r") b0, in("r") b1, in("r") b2, in("r") b3);
        }
    }
    #[inline(always)]
    pub unsafe fn wmma_store(c: &[f32; 8], p: u32) {
        ptx_asm!("wmma.store.d.sync.aligned.row.m16n16k16.shared.f32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}, 64;",
            in("r") p, in("f") c[0], in("f") c[1], in("f") c[2], in("f") c[3], in("f") c[4], in("f") c[5], in("f") c[6], in("f") c[7], clobber("memory"));
    }

    /// WMMA 64x64x32 tile body over `smem` (A_sh 64x32 T, B_sh 64x32 T, C_sh 64x64 f32 at +8192).
    #[inline(always)]
    pub unsafe fn mxfp4_wmma_tile<T: Wmma, R: Fn(i32) -> i32, O: Fn(i32) -> i32>(
        input: *const T, wt: *const u8, ws: *const u8, bias: *const T, output: *mut T, N: i32, K: i32, has_bias: u8, n_base: i32, smem: *mut u8,
        row_of: &R, out_of: &O,
    ) {
        let a_sh = smem as *mut u16;
        let b_sh = smem.add(4096) as *mut u16;
        let c_sh = smem.add(8192) as *mut f32;
        let tid = thread::threadIdx_x() as i32;
        let warp = tid / 32;
        let (wm, wn) = (warp / 2, warp % 2);
        let ss = K.wrapping_add(31) / 32;
        let mut c = [[0f32; 8]; 2];
        let sa = cuda_device::shared::cvta_generic_to_shared_u32(a_sh as *const u8);
        let sb = cuda_device::shared::cvta_generic_to_shared_u32(b_sh as *const u8);
        let sc = cuda_device::shared::cvta_generic_to_shared_u32(c_sh as *const u8);
        let mut kb = 0i32;
        while kb < K {
            let mut i = tid;
            while i < 256 {
                let idx = i * 8;
                let (lm, lk) = (idx / 32, idx % 32);
                let r = row_of(lm);
                let gk = kb + lk;
                let dst = a_sh.add((lm * 32 + lk) as usize) as *mut [u32; 4];
                *dst = if r >= 0 && gk < K {
                    *(input.add((r as i64 as u64).wrapping_mul(K as i64 as u64).wrapping_add(gk as i64 as u64) as usize) as *const [u32; 4])
                } else {
                    [0; 4]
                };
                i += 256;
            }
            let mut ln = tid;
            while ln < 64 {
                let gn = n_base + ln;
                let dst = b_sh.add((ln * 32) as usize);
                if gn < N {
                    let w = ld_u4(wt.add((gn as i64 as u64).wrapping_mul((K / 2) as i64 as u64).wrapping_add((kb / 2) as i64 as u64) as usize));
                    let s = e8m0_half(*ws.add((gn as i64 as u64).wrapping_mul(ss as i64 as u64).wrapping_add((kb / 32) as i64 as u64) as usize));
                    let mut q = 0;
                    while q < 4 {
                        let d = fp4_dq8(w[q], s);
                        let mut e = 0;
                        while e < 8 {
                            *dst.add(q * 8 + e) = T::to16(d[e]);
                            e += 1;
                        }
                        q += 1;
                    }
                } else {
                    let mut k = 0;
                    while k < 32 {
                        *dst.add(k) = 0;
                        k += 1;
                    }
                }
                ln += 256;
            }
            thread::sync_threads();
            let mut ks = 0u32;
            while ks < 2 {
                let a_addr = sa + ((wm as u32 * 16 * 32 + ks * 16) * 2);
                let mut ns = 0u32;
                while ns < 2 {
                    let b_addr = sb + (((wn as u32 * 2 + ns) * 16 * 32 + ks * 16) * 2);
                    T::mma(&mut c[ns as usize], a_addr, b_addr);
                    ns += 1;
                }
                ks += 1;
            }
            thread::sync_threads();
            kb += 32;
        }
        let mut ns = 0u32;
        while ns < 2 {
            wmma_store(&c[ns as usize], sc + (wm as u32 * 16 * 64 + (wn as u32 * 2 + ns) * 16) * 4);
            ns += 1;
        }
        thread::sync_threads();
        let mut i = tid;
        while i < 4096 {
            let (lm, ln) = (i / 64, i % 64);
            let orow = out_of(lm);
            let gn = n_base + ln;
            if orow >= 0 && gn < N {
                let mut val = *c_sh.add(i as usize);
                if has_bias != 0 && !bias.is_null() {
                    val = add(val, T::ldf(bias, gn as isize));
                }
                *(output.add((orow as i64 as u64).wrapping_mul(N as i64 as u64).wrapping_add(gn as i64 as u64) as usize) as *mut u16) = T::to16(val);
            }
            i += 256;
        }
    }

    #[inline(always)]
    pub unsafe fn mxfp4_matmul_wmma<T: Wmma>(input: *const T, weight: *const u8, ws: *const u8, bias: *const T, output: *mut T, M: i32, N: i32, K: i32, has_bias: u8) {
        let smem = cuda_device::DynamicSharedArray::<u8, 16>::get_raw() as usize as *mut u8;
        let m_base = (thread::blockIdx_y() as i32).wrapping_mul(64);
        let n_base = (thread::blockIdx_x() as i32).wrapping_mul(64);
        let row_of = |lm: i32| if m_base + lm < M { m_base + lm } else { -1 };
        mxfp4_wmma_tile(input, weight, ws, bias, output, N, K, has_bias, n_base, smem, &row_of, &row_of);
    }

    #[inline(always)]
    pub unsafe fn mxfp4_moe_grouped_wmma<T: Wmma>(
        input: *const T, weights: *const u8, wscales: *const u8, biases: *const T, indices: *const u32, output: *mut T, num_tokens: i32, topk: i32,
        num_experts: i32, N: i32, K: i32, has_bias: u8, has_topk: u8,
    ) {
        static mut CNT: SharedArray<i32, 1> = SharedArray::UNINIT;
        let cnt = SharedArray::as_raw_mut_ptr(&raw mut CNT);
        let smem = cuda_device::DynamicSharedArray::<u8, 16>::get_raw() as usize as *mut u8;
        let list = smem as *mut i32;
        let _ = num_experts;
        let ss = K.wrapping_add(31) / 32;
        let expert = thread::blockIdx_y() as i32;
        let n_base = (thread::blockIdx_x() as i32).wrapping_mul(64);
        let tid = thread::threadIdx_x() as i32;
        let total = num_tokens.wrapping_mul(topk);
        let ew = weights.add((expert as i64 as u64).wrapping_mul(N as i64 as u64).wrapping_mul((K / 2) as i64 as u64) as usize);
        let es = wscales.add((expert as i64 as u64).wrapping_mul(N as i64 as u64).wrapping_mul(ss as i64 as u64) as usize);
        let tlb = (total.wrapping_mul(4).wrapping_add(15)) & !15;
        let m_exp = moe_token_list(indices, total, expert, cnt, list, tid, 256);
        if m_exp == 0 {
            return;
        }
        let eb = biases.add((expert as i64 as u64).wrapping_mul(N as i64 as u64) as usize);
        let bias = if biases.is_null() { biases } else { eb };
        let tile = smem.offset(tlb as isize);
        let mut mt = 0;
        while mt < (m_exp + 63) / 64 {
            let row_of = |lm: i32| {
                let wp = mt * 64 + lm;
                if wp < m_exp {
                    let wi = *list.offset(wp as isize);
                    if has_topk != 0 { wi } else { wi / topk }
                } else {
                    -1
                }
            };
            let out_of = |lm: i32| {
                let wp = mt * 64 + lm;
                if wp < m_exp { *list.offset(wp as isize) } else { -1 }
            };
            mxfp4_wmma_tile(input, ew, es, bias, output, N, K, has_bias, n_base, tile, &row_of, &out_of);
            thread::sync_threads();
            mt += 1;
        }
    }

    #[inline(always)]
    pub fn mma_bf16(c: &mut [f32; 4], a: &[u32; 4], b: &[u32; 2]) {
        unsafe {
            ptx_asm!("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};",
                inout("+f") c[0], inout("+f") c[1], inout("+f") c[2], inout("+f") c[3],
                in("r") a[0], in("r") a[1], in("r") a[2], in("r") a[3], in("r") b[0], in("r") b[1]);
        }
    }

    // ------------------------------------------------------------------ bitsandbytes (dequant.cu)
    /// `dDequantizeFP4Tree(val, absmax)` as compiled: `1.0f * absmax` is folded to `absmax`
    /// (no flush), every other code is `mul.ftz(c, absmax)`; `* -1.0f` is `add.ftz(-v, -0)`.
    #[inline(always)]
    pub fn bnb_fp4(val: u32, absmax: f32) -> f32 {
        let v = match val & 7 {
            7 => mul(0.25, absmax),
            6 => mul(0.16666667, absmax),
            5 => mul(0.5, absmax),
            4 => mul(0.33333333, absmax),
            3 => absmax,
            2 => mul(0.66666667, absmax),
            1 => mul(5.208333333e-03, absmax),
            _ => mul(0.0, absmax),
        };
        if val & 8 != 0 {
            let r: f32;
            unsafe { ptx_asm!("{ .reg .f32 t; neg.ftz.f32 t, %1; add.rn.ftz.f32 %0, t, 0f80000000; }", out("=f") r, in("f") v, options(register_only)); }
            r
        } else {
            v
        }
    }
    #[inline(always)]
    pub fn bnb_nf4(val: u32) -> f32 {
        [
            -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023,
            1.0f32,
        ][(val & 15) as usize]
    }
    /// `kDequantizeBlockwise<T, 512, 64, 8, DT>` (DT 0 int8 / 1 FP4 / 2 NF4): blocked cub load /
    /// store (thread t owns items t*8..t*8+8), out-of-range loads read 128, stores are guarded.
    #[inline(always)]
    pub unsafe fn bnb_dequant<T: El, const DT: u32>(code: *const f32, a: *const u8, absmax: *const f32, out: *mut T, blocksize: i32, n: i32) {
        let i = thread::blockIdx_x().wrapping_mul(512);
        let tid = thread::threadIdx_x();
        let (vl, vs): (u32, u32) = if DT > 0 {
            let l = (((n + 1) / 2) as u32).wrapping_sub(i);
            let s = (n as u32).wrapping_sub(i.wrapping_mul(2));
            (if l > 512 { 512 } else { l }, if s > 1024 { 1024 } else { s })
        } else {
            let l = (n as u32).wrapping_sub(i);
            (if l > 512 { 512 } else { l }, if l > 512 { 512 } else { l })
        };
        let lam = *absmax.add((i.wrapping_add(tid * 8) / blocksize as u32) as usize);
        let mut q = [128u32; 8];
        let mut j = 0;
        while j < 8 {
            if ((tid * 8 + j) as i32) < vl as i32 {
                q[j as usize] = *a.add((i + tid * 8 + j) as usize) as u32;
            }
            j += 1;
        }
        let per = if DT > 0 { 16 } else { 8 };
        let base = if DT > 0 { i * 2 } else { i };
        let mut j = 0u32;
        while j < per {
            let v = if DT == 0 {
                mul(*code.add(q[j as usize] as usize), lam)
            } else {
                let byte = q[(j / 2) as usize];
                let nib = if j % 2 == 0 { byte >> 4 } else { byte & 0x0f };
                if DT == 1 { bnb_fp4(nib, lam) } else { mul(bnb_nf4(nib), lam) }
            };
            if ((tid * per + j) as i32) < vs as i32 {
                T::stf(out, (base + tid * per + j) as isize, v);
            }
            j += 1;
        }
    }

    // ------------------------------------------------------------------ CUTLASS 2.x grouped GEMM (bf16)
    /// Per output element exactly what CUTLASS's Sm80 MmaMultistage computes: accumulators start
    /// at +0; K is walked in 32-wide tiles, the FIRST one being the residue tile (k < K % 32, or a
    /// full tile), zero-filled past its extent; each tile is two m16n8k16 bf16 mma (k 0..16,
    /// 16..32) in order; epilogue `mul.ftz(acc, alpha = 1)` then cvt.rn.bf16. Tiling/scheduling
    /// (device-only problem visitor) does not change any value.
    #[inline(always)]
    pub unsafe fn grouped_mm(
        a_ptrs: *const *const u16, b_ptrs: *const *const u16, d_ptrs: *const *mut u16, problem_sizes: *const i32, problem_count: i32,
        lda: *const i64, ldb: *const i64, ldd: *const i64,
    ) {
        let lane = (thread::threadIdx_x() % 32) as i64;
        let warp = (thread::threadIdx_x() / 32) as i64;
        let (g, q) = (lane / 4, lane % 4);
        let mut t = thread::blockIdx_x() as i64;
        let gstep = thread::gridDim_x() as i64;
        let mut p = 0i32;
        let mut first = 0i64; // first global tile index of problem p
        while p < problem_count {
            let m = *problem_sizes.offset((3 * p) as isize) as i64;
            let n = *problem_sizes.offset((3 * p + 1) as isize) as i64;
            let kd = *problem_sizes.offset((3 * p + 2) as isize) as i64;
            let tn = (n + 31) / 32;
            let tiles = ((m + 15) / 16) * tn;
            while t < first + tiles {
                let local = t - first;
                let m0 = (local / tn) * 16;
                let n0 = (local % tn) * 32 + warp * 8;
                let a = *a_ptrs.offset(p as isize);
                let b = *b_ptrs.offset(p as isize);
                let d = *d_ptrs.offset(p as isize);
                let (la, lb, ld) = (*lda.offset(p as isize), *ldb.offset(p as isize), *ldd.offset(p as isize));
                let ga = |row: i64, k: i64, valid: bool| -> u32 { if valid && row < m { *a.offset((row * la + k) as isize) as u32 } else { 0 } };
                let gb = |k: i64, col: i64, valid: bool| -> u32 { if valid && col < n { *b.offset((col * lb + k) as isize) as u32 } else { 0 } };
                let mut c = [0f32; 4];
                let ktiles = (kd + 31) / 32;
                let r = if kd % 32 == 0 { 32 } else { kd % 32 };
                let mut kt = 0i64;
                while kt < ktiles {
                    let (base, valid) = if kt == 0 { (0i64, r) } else { (r + 32 * (kt - 1), 32i64) };
                    let mut ch = 0i64;
                    while ch < 2 {
                        let s0 = ch * 16 + 2 * q;
                        let pa = |row: i64, s: i64| ga(row, base + s, s < valid) | (ga(row, base + s + 1, s + 1 < valid) << 16);
                        let pb = |col: i64, s: i64| gb(base + s, col, s < valid) | (gb(base + s + 1, col, s + 1 < valid) << 16);
                        let af = [pa(m0 + g, s0), pa(m0 + g + 8, s0), pa(m0 + g, s0 + 8), pa(m0 + g + 8, s0 + 8)];
                        let bf = [pb(n0 + g, s0), pb(n0 + g, s0 + 8)];
                        mma_bf16(&mut c, &af, &bf);
                        ch += 1;
                    }
                    kt += 1;
                }
                let mut e = 0;
                while e < 4 {
                    let row = m0 + g + if e >= 2 { 8 } else { 0 };
                    let col = n0 + 2 * q + (e % 2) as i64;
                    if row < m && col < n {
                        *d.offset((row * ld + col) as isize) = f2bf(mul(c[e], 1.0));
                    }
                    e += 1;
                }
                t += gstep;
            }
            first += tiles;
            p += 1;
        }
    }

    // GENERATED KERNELS BEGIN
    #[kernel] pub unsafe fn afq_dequantize_2bit_gs32_f32(w_q: *const u32, scales: *const f32, biases: *const f32, output: *mut f32, rows: i32, cols: i32) { afq_dequant::<f32, 2, 32>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_2bit_gs32_f32(x: *const f32, w_q: *const u32, scales: *const f32, biases: *const f32, y: *mut f32, m: i32, n: i32, kk: i32) { afq_qmv::<f32, 2, 32>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_2bit_gs64_f32(w_q: *const u32, scales: *const f32, biases: *const f32, output: *mut f32, rows: i32, cols: i32) { afq_dequant::<f32, 2, 64>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_2bit_gs64_f32(x: *const f32, w_q: *const u32, scales: *const f32, biases: *const f32, y: *mut f32, m: i32, n: i32, kk: i32) { afq_qmv::<f32, 2, 64>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_2bit_gs128_f32(w_q: *const u32, scales: *const f32, biases: *const f32, output: *mut f32, rows: i32, cols: i32) { afq_dequant::<f32, 2, 128>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_2bit_gs128_f32(x: *const f32, w_q: *const u32, scales: *const f32, biases: *const f32, y: *mut f32, m: i32, n: i32, kk: i32) { afq_qmv::<f32, 2, 128>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_3bit_gs32_f32(w_q: *const u8, scales: *const f32, biases: *const f32, output: *mut f32, rows: i32, cols: i32) { afq_dequant::<f32, 3, 32>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_3bit_gs32_f32(x: *const f32, w_q: *const u8, scales: *const f32, biases: *const f32, y: *mut f32, m: i32, n: i32, kk: i32) { afq_qmv::<f32, 3, 32>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_3bit_gs64_f32(w_q: *const u8, scales: *const f32, biases: *const f32, output: *mut f32, rows: i32, cols: i32) { afq_dequant::<f32, 3, 64>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_3bit_gs64_f32(x: *const f32, w_q: *const u8, scales: *const f32, biases: *const f32, y: *mut f32, m: i32, n: i32, kk: i32) { afq_qmv::<f32, 3, 64>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_3bit_gs128_f32(w_q: *const u8, scales: *const f32, biases: *const f32, output: *mut f32, rows: i32, cols: i32) { afq_dequant::<f32, 3, 128>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_3bit_gs128_f32(x: *const f32, w_q: *const u8, scales: *const f32, biases: *const f32, y: *mut f32, m: i32, n: i32, kk: i32) { afq_qmv::<f32, 3, 128>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_4bit_gs32_f32(w_q: *const u32, scales: *const f32, biases: *const f32, output: *mut f32, rows: i32, cols: i32) { afq_dequant::<f32, 4, 32>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_4bit_gs32_f32(x: *const f32, w_q: *const u32, scales: *const f32, biases: *const f32, y: *mut f32, m: i32, n: i32, kk: i32) { afq_qmv::<f32, 4, 32>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_4bit_gs64_f32(w_q: *const u32, scales: *const f32, biases: *const f32, output: *mut f32, rows: i32, cols: i32) { afq_dequant::<f32, 4, 64>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_4bit_gs64_f32(x: *const f32, w_q: *const u32, scales: *const f32, biases: *const f32, y: *mut f32, m: i32, n: i32, kk: i32) { afq_qmv::<f32, 4, 64>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_4bit_gs128_f32(w_q: *const u32, scales: *const f32, biases: *const f32, output: *mut f32, rows: i32, cols: i32) { afq_dequant::<f32, 4, 128>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_4bit_gs128_f32(x: *const f32, w_q: *const u32, scales: *const f32, biases: *const f32, y: *mut f32, m: i32, n: i32, kk: i32) { afq_qmv::<f32, 4, 128>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_6bit_gs32_f32(w_q: *const u8, scales: *const f32, biases: *const f32, output: *mut f32, rows: i32, cols: i32) { afq_dequant::<f32, 6, 32>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_6bit_gs32_f32(x: *const f32, w_q: *const u8, scales: *const f32, biases: *const f32, y: *mut f32, m: i32, n: i32, kk: i32) { afq_qmv::<f32, 6, 32>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_6bit_gs64_f32(w_q: *const u8, scales: *const f32, biases: *const f32, output: *mut f32, rows: i32, cols: i32) { afq_dequant::<f32, 6, 64>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_6bit_gs64_f32(x: *const f32, w_q: *const u8, scales: *const f32, biases: *const f32, y: *mut f32, m: i32, n: i32, kk: i32) { afq_qmv::<f32, 6, 64>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_6bit_gs128_f32(w_q: *const u8, scales: *const f32, biases: *const f32, output: *mut f32, rows: i32, cols: i32) { afq_dequant::<f32, 6, 128>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_6bit_gs128_f32(x: *const f32, w_q: *const u8, scales: *const f32, biases: *const f32, y: *mut f32, m: i32, n: i32, kk: i32) { afq_qmv::<f32, 6, 128>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_8bit_gs32_f32(w_q: *const u32, scales: *const f32, biases: *const f32, output: *mut f32, rows: i32, cols: i32) { afq_dequant::<f32, 8, 32>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_8bit_gs32_f32(x: *const f32, w_q: *const u32, scales: *const f32, biases: *const f32, y: *mut f32, m: i32, n: i32, kk: i32) { afq_qmv::<f32, 8, 32>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_8bit_gs64_f32(w_q: *const u32, scales: *const f32, biases: *const f32, output: *mut f32, rows: i32, cols: i32) { afq_dequant::<f32, 8, 64>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_8bit_gs64_f32(x: *const f32, w_q: *const u32, scales: *const f32, biases: *const f32, y: *mut f32, m: i32, n: i32, kk: i32) { afq_qmv::<f32, 8, 64>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_8bit_gs128_f32(w_q: *const u32, scales: *const f32, biases: *const f32, output: *mut f32, rows: i32, cols: i32) { afq_dequant::<f32, 8, 128>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_8bit_gs128_f32(x: *const f32, w_q: *const u32, scales: *const f32, biases: *const f32, y: *mut f32, m: i32, n: i32, kk: i32) { afq_qmv::<f32, 8, 128>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_quantize_2bit_gs32_f32(w: *const f32, w_q: *mut u32, scales: *mut f32, biases: *mut f32, rows: i32, cols: i32) { afq_quantize::<f32, 2, 32>(w, w_q, scales, biases, rows, cols) }
    #[kernel] pub unsafe fn afq_qmm_2bit_gs32_f32(x: *const f32, w_q: *const u32, scales: *const f32, biases: *const f32, y: *mut f32, m: i32, n: i32, kk: i32) { afq_qmm::<f32, 2, 32>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_quantize_2bit_gs64_f32(w: *const f32, w_q: *mut u32, scales: *mut f32, biases: *mut f32, rows: i32, cols: i32) { afq_quantize::<f32, 2, 64>(w, w_q, scales, biases, rows, cols) }
    #[kernel] pub unsafe fn afq_qmm_2bit_gs64_f32(x: *const f32, w_q: *const u32, scales: *const f32, biases: *const f32, y: *mut f32, m: i32, n: i32, kk: i32) { afq_qmm::<f32, 2, 64>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_quantize_2bit_gs128_f32(w: *const f32, w_q: *mut u32, scales: *mut f32, biases: *mut f32, rows: i32, cols: i32) { afq_quantize::<f32, 2, 128>(w, w_q, scales, biases, rows, cols) }
    #[kernel] pub unsafe fn afq_qmm_2bit_gs128_f32(x: *const f32, w_q: *const u32, scales: *const f32, biases: *const f32, y: *mut f32, m: i32, n: i32, kk: i32) { afq_qmm::<f32, 2, 128>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_quantize_4bit_gs32_f32(w: *const f32, w_q: *mut u32, scales: *mut f32, biases: *mut f32, rows: i32, cols: i32) { afq_quantize::<f32, 4, 32>(w, w_q, scales, biases, rows, cols) }
    #[kernel] pub unsafe fn afq_qmm_4bit_gs32_f32(x: *const f32, w_q: *const u32, scales: *const f32, biases: *const f32, y: *mut f32, m: i32, n: i32, kk: i32) { afq_qmm::<f32, 4, 32>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_quantize_4bit_gs64_f32(w: *const f32, w_q: *mut u32, scales: *mut f32, biases: *mut f32, rows: i32, cols: i32) { afq_quantize::<f32, 4, 64>(w, w_q, scales, biases, rows, cols) }
    #[kernel] pub unsafe fn afq_qmm_4bit_gs64_f32(x: *const f32, w_q: *const u32, scales: *const f32, biases: *const f32, y: *mut f32, m: i32, n: i32, kk: i32) { afq_qmm::<f32, 4, 64>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_quantize_4bit_gs128_f32(w: *const f32, w_q: *mut u32, scales: *mut f32, biases: *mut f32, rows: i32, cols: i32) { afq_quantize::<f32, 4, 128>(w, w_q, scales, biases, rows, cols) }
    #[kernel] pub unsafe fn afq_qmm_4bit_gs128_f32(x: *const f32, w_q: *const u32, scales: *const f32, biases: *const f32, y: *mut f32, m: i32, n: i32, kk: i32) { afq_qmm::<f32, 4, 128>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_quantize_8bit_gs32_f32(w: *const f32, w_q: *mut u32, scales: *mut f32, biases: *mut f32, rows: i32, cols: i32) { afq_quantize::<f32, 8, 32>(w, w_q, scales, biases, rows, cols) }
    #[kernel] pub unsafe fn afq_qmm_8bit_gs32_f32(x: *const f32, w_q: *const u32, scales: *const f32, biases: *const f32, y: *mut f32, m: i32, n: i32, kk: i32) { afq_qmm::<f32, 8, 32>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_quantize_8bit_gs64_f32(w: *const f32, w_q: *mut u32, scales: *mut f32, biases: *mut f32, rows: i32, cols: i32) { afq_quantize::<f32, 8, 64>(w, w_q, scales, biases, rows, cols) }
    #[kernel] pub unsafe fn afq_qmm_8bit_gs64_f32(x: *const f32, w_q: *const u32, scales: *const f32, biases: *const f32, y: *mut f32, m: i32, n: i32, kk: i32) { afq_qmm::<f32, 8, 64>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_quantize_8bit_gs128_f32(w: *const f32, w_q: *mut u32, scales: *mut f32, biases: *mut f32, rows: i32, cols: i32) { afq_quantize::<f32, 8, 128>(w, w_q, scales, biases, rows, cols) }
    #[kernel] pub unsafe fn afq_qmm_8bit_gs128_f32(x: *const f32, w_q: *const u32, scales: *const f32, biases: *const f32, y: *mut f32, m: i32, n: i32, kk: i32) { afq_qmm::<f32, 8, 128>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_2bit_gs32_f16(w_q: *const u32, scales: *const H, biases: *const H, output: *mut H, rows: i32, cols: i32) { afq_dequant::<H, 2, 32>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_2bit_gs32_f16(x: *const H, w_q: *const u32, scales: *const H, biases: *const H, y: *mut H, m: i32, n: i32, kk: i32) { afq_qmv::<H, 2, 32>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_2bit_gs64_f16(w_q: *const u32, scales: *const H, biases: *const H, output: *mut H, rows: i32, cols: i32) { afq_dequant::<H, 2, 64>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_2bit_gs64_f16(x: *const H, w_q: *const u32, scales: *const H, biases: *const H, y: *mut H, m: i32, n: i32, kk: i32) { afq_qmv::<H, 2, 64>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_2bit_gs128_f16(w_q: *const u32, scales: *const H, biases: *const H, output: *mut H, rows: i32, cols: i32) { afq_dequant::<H, 2, 128>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_2bit_gs128_f16(x: *const H, w_q: *const u32, scales: *const H, biases: *const H, y: *mut H, m: i32, n: i32, kk: i32) { afq_qmv::<H, 2, 128>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_3bit_gs32_f16(w_q: *const u8, scales: *const H, biases: *const H, output: *mut H, rows: i32, cols: i32) { afq_dequant::<H, 3, 32>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_3bit_gs32_f16(x: *const H, w_q: *const u8, scales: *const H, biases: *const H, y: *mut H, m: i32, n: i32, kk: i32) { afq_qmv::<H, 3, 32>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_3bit_gs64_f16(w_q: *const u8, scales: *const H, biases: *const H, output: *mut H, rows: i32, cols: i32) { afq_dequant::<H, 3, 64>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_3bit_gs64_f16(x: *const H, w_q: *const u8, scales: *const H, biases: *const H, y: *mut H, m: i32, n: i32, kk: i32) { afq_qmv::<H, 3, 64>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_3bit_gs128_f16(w_q: *const u8, scales: *const H, biases: *const H, output: *mut H, rows: i32, cols: i32) { afq_dequant::<H, 3, 128>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_3bit_gs128_f16(x: *const H, w_q: *const u8, scales: *const H, biases: *const H, y: *mut H, m: i32, n: i32, kk: i32) { afq_qmv::<H, 3, 128>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_4bit_gs32_f16(w_q: *const u32, scales: *const H, biases: *const H, output: *mut H, rows: i32, cols: i32) { afq_dequant::<H, 4, 32>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_4bit_gs32_f16(x: *const H, w_q: *const u32, scales: *const H, biases: *const H, y: *mut H, m: i32, n: i32, kk: i32) { afq_qmv::<H, 4, 32>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_4bit_gs64_f16(w_q: *const u32, scales: *const H, biases: *const H, output: *mut H, rows: i32, cols: i32) { afq_dequant::<H, 4, 64>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_4bit_gs64_f16(x: *const H, w_q: *const u32, scales: *const H, biases: *const H, y: *mut H, m: i32, n: i32, kk: i32) { afq_qmv::<H, 4, 64>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_4bit_gs128_f16(w_q: *const u32, scales: *const H, biases: *const H, output: *mut H, rows: i32, cols: i32) { afq_dequant::<H, 4, 128>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_4bit_gs128_f16(x: *const H, w_q: *const u32, scales: *const H, biases: *const H, y: *mut H, m: i32, n: i32, kk: i32) { afq_qmv::<H, 4, 128>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_6bit_gs32_f16(w_q: *const u8, scales: *const H, biases: *const H, output: *mut H, rows: i32, cols: i32) { afq_dequant::<H, 6, 32>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_6bit_gs32_f16(x: *const H, w_q: *const u8, scales: *const H, biases: *const H, y: *mut H, m: i32, n: i32, kk: i32) { afq_qmv::<H, 6, 32>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_6bit_gs64_f16(w_q: *const u8, scales: *const H, biases: *const H, output: *mut H, rows: i32, cols: i32) { afq_dequant::<H, 6, 64>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_6bit_gs64_f16(x: *const H, w_q: *const u8, scales: *const H, biases: *const H, y: *mut H, m: i32, n: i32, kk: i32) { afq_qmv::<H, 6, 64>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_6bit_gs128_f16(w_q: *const u8, scales: *const H, biases: *const H, output: *mut H, rows: i32, cols: i32) { afq_dequant::<H, 6, 128>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_6bit_gs128_f16(x: *const H, w_q: *const u8, scales: *const H, biases: *const H, y: *mut H, m: i32, n: i32, kk: i32) { afq_qmv::<H, 6, 128>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_8bit_gs32_f16(w_q: *const u32, scales: *const H, biases: *const H, output: *mut H, rows: i32, cols: i32) { afq_dequant::<H, 8, 32>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_8bit_gs32_f16(x: *const H, w_q: *const u32, scales: *const H, biases: *const H, y: *mut H, m: i32, n: i32, kk: i32) { afq_qmv::<H, 8, 32>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_8bit_gs64_f16(w_q: *const u32, scales: *const H, biases: *const H, output: *mut H, rows: i32, cols: i32) { afq_dequant::<H, 8, 64>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_8bit_gs64_f16(x: *const H, w_q: *const u32, scales: *const H, biases: *const H, y: *mut H, m: i32, n: i32, kk: i32) { afq_qmv::<H, 8, 64>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_8bit_gs128_f16(w_q: *const u32, scales: *const H, biases: *const H, output: *mut H, rows: i32, cols: i32) { afq_dequant::<H, 8, 128>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_8bit_gs128_f16(x: *const H, w_q: *const u32, scales: *const H, biases: *const H, y: *mut H, m: i32, n: i32, kk: i32) { afq_qmv::<H, 8, 128>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_quantize_2bit_gs32_f16(w: *const H, w_q: *mut u32, scales: *mut H, biases: *mut H, rows: i32, cols: i32) { afq_quantize::<H, 2, 32>(w, w_q, scales, biases, rows, cols) }
    #[kernel] pub unsafe fn afq_qmm_2bit_gs32_f16(x: *const H, w_q: *const u32, scales: *const H, biases: *const H, y: *mut H, m: i32, n: i32, kk: i32) { afq_qmm::<H, 2, 32>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_quantize_2bit_gs64_f16(w: *const H, w_q: *mut u32, scales: *mut H, biases: *mut H, rows: i32, cols: i32) { afq_quantize::<H, 2, 64>(w, w_q, scales, biases, rows, cols) }
    #[kernel] pub unsafe fn afq_qmm_2bit_gs64_f16(x: *const H, w_q: *const u32, scales: *const H, biases: *const H, y: *mut H, m: i32, n: i32, kk: i32) { afq_qmm::<H, 2, 64>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_quantize_2bit_gs128_f16(w: *const H, w_q: *mut u32, scales: *mut H, biases: *mut H, rows: i32, cols: i32) { afq_quantize::<H, 2, 128>(w, w_q, scales, biases, rows, cols) }
    #[kernel] pub unsafe fn afq_qmm_2bit_gs128_f16(x: *const H, w_q: *const u32, scales: *const H, biases: *const H, y: *mut H, m: i32, n: i32, kk: i32) { afq_qmm::<H, 2, 128>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_quantize_4bit_gs32_f16(w: *const H, w_q: *mut u32, scales: *mut H, biases: *mut H, rows: i32, cols: i32) { afq_quantize::<H, 4, 32>(w, w_q, scales, biases, rows, cols) }
    #[kernel] pub unsafe fn afq_qmm_4bit_gs32_f16(x: *const H, w_q: *const u32, scales: *const H, biases: *const H, y: *mut H, m: i32, n: i32, kk: i32) { afq_qmm::<H, 4, 32>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_quantize_4bit_gs64_f16(w: *const H, w_q: *mut u32, scales: *mut H, biases: *mut H, rows: i32, cols: i32) { afq_quantize::<H, 4, 64>(w, w_q, scales, biases, rows, cols) }
    #[kernel] pub unsafe fn afq_qmm_4bit_gs64_f16(x: *const H, w_q: *const u32, scales: *const H, biases: *const H, y: *mut H, m: i32, n: i32, kk: i32) { afq_qmm::<H, 4, 64>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_quantize_4bit_gs128_f16(w: *const H, w_q: *mut u32, scales: *mut H, biases: *mut H, rows: i32, cols: i32) { afq_quantize::<H, 4, 128>(w, w_q, scales, biases, rows, cols) }
    #[kernel] pub unsafe fn afq_qmm_4bit_gs128_f16(x: *const H, w_q: *const u32, scales: *const H, biases: *const H, y: *mut H, m: i32, n: i32, kk: i32) { afq_qmm::<H, 4, 128>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_quantize_8bit_gs32_f16(w: *const H, w_q: *mut u32, scales: *mut H, biases: *mut H, rows: i32, cols: i32) { afq_quantize::<H, 8, 32>(w, w_q, scales, biases, rows, cols) }
    #[kernel] pub unsafe fn afq_qmm_8bit_gs32_f16(x: *const H, w_q: *const u32, scales: *const H, biases: *const H, y: *mut H, m: i32, n: i32, kk: i32) { afq_qmm::<H, 8, 32>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_quantize_8bit_gs64_f16(w: *const H, w_q: *mut u32, scales: *mut H, biases: *mut H, rows: i32, cols: i32) { afq_quantize::<H, 8, 64>(w, w_q, scales, biases, rows, cols) }
    #[kernel] pub unsafe fn afq_qmm_8bit_gs64_f16(x: *const H, w_q: *const u32, scales: *const H, biases: *const H, y: *mut H, m: i32, n: i32, kk: i32) { afq_qmm::<H, 8, 64>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_quantize_8bit_gs128_f16(w: *const H, w_q: *mut u32, scales: *mut H, biases: *mut H, rows: i32, cols: i32) { afq_quantize::<H, 8, 128>(w, w_q, scales, biases, rows, cols) }
    #[kernel] pub unsafe fn afq_qmm_8bit_gs128_f16(x: *const H, w_q: *const u32, scales: *const H, biases: *const H, y: *mut H, m: i32, n: i32, kk: i32) { afq_qmm::<H, 8, 128>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_2bit_gs32_bf16(w_q: *const u32, scales: *const B, biases: *const B, output: *mut B, rows: i32, cols: i32) { afq_dequant::<B, 2, 32>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_2bit_gs32_bf16(x: *const B, w_q: *const u32, scales: *const B, biases: *const B, y: *mut B, m: i32, n: i32, kk: i32) { afq_qmv::<B, 2, 32>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_2bit_gs64_bf16(w_q: *const u32, scales: *const B, biases: *const B, output: *mut B, rows: i32, cols: i32) { afq_dequant::<B, 2, 64>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_2bit_gs64_bf16(x: *const B, w_q: *const u32, scales: *const B, biases: *const B, y: *mut B, m: i32, n: i32, kk: i32) { afq_qmv::<B, 2, 64>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_2bit_gs128_bf16(w_q: *const u32, scales: *const B, biases: *const B, output: *mut B, rows: i32, cols: i32) { afq_dequant::<B, 2, 128>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_2bit_gs128_bf16(x: *const B, w_q: *const u32, scales: *const B, biases: *const B, y: *mut B, m: i32, n: i32, kk: i32) { afq_qmv::<B, 2, 128>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_3bit_gs32_bf16(w_q: *const u8, scales: *const B, biases: *const B, output: *mut B, rows: i32, cols: i32) { afq_dequant::<B, 3, 32>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_3bit_gs32_bf16(x: *const B, w_q: *const u8, scales: *const B, biases: *const B, y: *mut B, m: i32, n: i32, kk: i32) { afq_qmv::<B, 3, 32>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_3bit_gs64_bf16(w_q: *const u8, scales: *const B, biases: *const B, output: *mut B, rows: i32, cols: i32) { afq_dequant::<B, 3, 64>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_3bit_gs64_bf16(x: *const B, w_q: *const u8, scales: *const B, biases: *const B, y: *mut B, m: i32, n: i32, kk: i32) { afq_qmv::<B, 3, 64>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_3bit_gs128_bf16(w_q: *const u8, scales: *const B, biases: *const B, output: *mut B, rows: i32, cols: i32) { afq_dequant::<B, 3, 128>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_3bit_gs128_bf16(x: *const B, w_q: *const u8, scales: *const B, biases: *const B, y: *mut B, m: i32, n: i32, kk: i32) { afq_qmv::<B, 3, 128>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_4bit_gs32_bf16(w_q: *const u32, scales: *const B, biases: *const B, output: *mut B, rows: i32, cols: i32) { afq_dequant::<B, 4, 32>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_4bit_gs32_bf16(x: *const B, w_q: *const u32, scales: *const B, biases: *const B, y: *mut B, m: i32, n: i32, kk: i32) { afq_qmv::<B, 4, 32>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_4bit_gs64_bf16(w_q: *const u32, scales: *const B, biases: *const B, output: *mut B, rows: i32, cols: i32) { afq_dequant::<B, 4, 64>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_4bit_gs64_bf16(x: *const B, w_q: *const u32, scales: *const B, biases: *const B, y: *mut B, m: i32, n: i32, kk: i32) { afq_qmv::<B, 4, 64>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_4bit_gs128_bf16(w_q: *const u32, scales: *const B, biases: *const B, output: *mut B, rows: i32, cols: i32) { afq_dequant::<B, 4, 128>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_4bit_gs128_bf16(x: *const B, w_q: *const u32, scales: *const B, biases: *const B, y: *mut B, m: i32, n: i32, kk: i32) { afq_qmv::<B, 4, 128>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_6bit_gs32_bf16(w_q: *const u8, scales: *const B, biases: *const B, output: *mut B, rows: i32, cols: i32) { afq_dequant::<B, 6, 32>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_6bit_gs32_bf16(x: *const B, w_q: *const u8, scales: *const B, biases: *const B, y: *mut B, m: i32, n: i32, kk: i32) { afq_qmv::<B, 6, 32>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_6bit_gs64_bf16(w_q: *const u8, scales: *const B, biases: *const B, output: *mut B, rows: i32, cols: i32) { afq_dequant::<B, 6, 64>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_6bit_gs64_bf16(x: *const B, w_q: *const u8, scales: *const B, biases: *const B, y: *mut B, m: i32, n: i32, kk: i32) { afq_qmv::<B, 6, 64>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_6bit_gs128_bf16(w_q: *const u8, scales: *const B, biases: *const B, output: *mut B, rows: i32, cols: i32) { afq_dequant::<B, 6, 128>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_6bit_gs128_bf16(x: *const B, w_q: *const u8, scales: *const B, biases: *const B, y: *mut B, m: i32, n: i32, kk: i32) { afq_qmv::<B, 6, 128>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_8bit_gs32_bf16(w_q: *const u32, scales: *const B, biases: *const B, output: *mut B, rows: i32, cols: i32) { afq_dequant::<B, 8, 32>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_8bit_gs32_bf16(x: *const B, w_q: *const u32, scales: *const B, biases: *const B, y: *mut B, m: i32, n: i32, kk: i32) { afq_qmv::<B, 8, 32>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_8bit_gs64_bf16(w_q: *const u32, scales: *const B, biases: *const B, output: *mut B, rows: i32, cols: i32) { afq_dequant::<B, 8, 64>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_8bit_gs64_bf16(x: *const B, w_q: *const u32, scales: *const B, biases: *const B, y: *mut B, m: i32, n: i32, kk: i32) { afq_qmv::<B, 8, 64>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_dequantize_8bit_gs128_bf16(w_q: *const u32, scales: *const B, biases: *const B, output: *mut B, rows: i32, cols: i32) { afq_dequant::<B, 8, 128>(w_q as *const u8, scales, biases, output, rows, cols) }
    #[kernel] pub unsafe fn afq_qmv_8bit_gs128_bf16(x: *const B, w_q: *const u32, scales: *const B, biases: *const B, y: *mut B, m: i32, n: i32, kk: i32) { afq_qmv::<B, 8, 128>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_quantize_2bit_gs32_bf16(w: *const B, w_q: *mut u32, scales: *mut B, biases: *mut B, rows: i32, cols: i32) { afq_quantize::<B, 2, 32>(w, w_q, scales, biases, rows, cols) }
    #[kernel] pub unsafe fn afq_qmm_2bit_gs32_bf16(x: *const B, w_q: *const u32, scales: *const B, biases: *const B, y: *mut B, m: i32, n: i32, kk: i32) { afq_qmm::<B, 2, 32>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_quantize_2bit_gs64_bf16(w: *const B, w_q: *mut u32, scales: *mut B, biases: *mut B, rows: i32, cols: i32) { afq_quantize::<B, 2, 64>(w, w_q, scales, biases, rows, cols) }
    #[kernel] pub unsafe fn afq_qmm_2bit_gs64_bf16(x: *const B, w_q: *const u32, scales: *const B, biases: *const B, y: *mut B, m: i32, n: i32, kk: i32) { afq_qmm::<B, 2, 64>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_quantize_2bit_gs128_bf16(w: *const B, w_q: *mut u32, scales: *mut B, biases: *mut B, rows: i32, cols: i32) { afq_quantize::<B, 2, 128>(w, w_q, scales, biases, rows, cols) }
    #[kernel] pub unsafe fn afq_qmm_2bit_gs128_bf16(x: *const B, w_q: *const u32, scales: *const B, biases: *const B, y: *mut B, m: i32, n: i32, kk: i32) { afq_qmm::<B, 2, 128>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_quantize_4bit_gs32_bf16(w: *const B, w_q: *mut u32, scales: *mut B, biases: *mut B, rows: i32, cols: i32) { afq_quantize::<B, 4, 32>(w, w_q, scales, biases, rows, cols) }
    #[kernel] pub unsafe fn afq_qmm_4bit_gs32_bf16(x: *const B, w_q: *const u32, scales: *const B, biases: *const B, y: *mut B, m: i32, n: i32, kk: i32) { afq_qmm::<B, 4, 32>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_quantize_4bit_gs64_bf16(w: *const B, w_q: *mut u32, scales: *mut B, biases: *mut B, rows: i32, cols: i32) { afq_quantize::<B, 4, 64>(w, w_q, scales, biases, rows, cols) }
    #[kernel] pub unsafe fn afq_qmm_4bit_gs64_bf16(x: *const B, w_q: *const u32, scales: *const B, biases: *const B, y: *mut B, m: i32, n: i32, kk: i32) { afq_qmm::<B, 4, 64>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_quantize_4bit_gs128_bf16(w: *const B, w_q: *mut u32, scales: *mut B, biases: *mut B, rows: i32, cols: i32) { afq_quantize::<B, 4, 128>(w, w_q, scales, biases, rows, cols) }
    #[kernel] pub unsafe fn afq_qmm_4bit_gs128_bf16(x: *const B, w_q: *const u32, scales: *const B, biases: *const B, y: *mut B, m: i32, n: i32, kk: i32) { afq_qmm::<B, 4, 128>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_quantize_8bit_gs32_bf16(w: *const B, w_q: *mut u32, scales: *mut B, biases: *mut B, rows: i32, cols: i32) { afq_quantize::<B, 8, 32>(w, w_q, scales, biases, rows, cols) }
    #[kernel] pub unsafe fn afq_qmm_8bit_gs32_bf16(x: *const B, w_q: *const u32, scales: *const B, biases: *const B, y: *mut B, m: i32, n: i32, kk: i32) { afq_qmm::<B, 8, 32>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_quantize_8bit_gs64_bf16(w: *const B, w_q: *mut u32, scales: *mut B, biases: *mut B, rows: i32, cols: i32) { afq_quantize::<B, 8, 64>(w, w_q, scales, biases, rows, cols) }
    #[kernel] pub unsafe fn afq_qmm_8bit_gs64_bf16(x: *const B, w_q: *const u32, scales: *const B, biases: *const B, y: *mut B, m: i32, n: i32, kk: i32) { afq_qmm::<B, 8, 64>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn afq_quantize_8bit_gs128_bf16(w: *const B, w_q: *mut u32, scales: *mut B, biases: *mut B, rows: i32, cols: i32) { afq_quantize::<B, 8, 128>(w, w_q, scales, biases, rows, cols) }
    #[kernel] pub unsafe fn afq_qmm_8bit_gs128_bf16(x: *const B, w_q: *const u32, scales: *const B, biases: *const B, y: *mut B, m: i32, n: i32, kk: i32) { afq_qmm::<B, 8, 128>(x, w_q as *const u8, scales, biases, y, m, n, kk) }
    #[kernel] pub unsafe fn fp8_to_f32(input: *const u8, output: *mut f32, n: usize) { fp8_to_dtype(input, output, n) }
    #[kernel] pub unsafe fn f32_to_fp8(input: *const f32, output: *mut u8, n: usize) { dtype_to_fp8(input, output, n) }
    #[kernel] pub unsafe fn dequant_fp8_vector_f32(weight: *const u8, scale: *const f32, output: *mut f32, n: usize) { dequant_fp8_vector(weight, scale, output, n) }
    #[kernel] pub unsafe fn quant_fp8_vector_f32(input: *const f32, weight: *mut u8, scale: *mut f32, n: usize) { quant_fp8_vector(input, weight, scale, n) }
    #[kernel] pub unsafe fn dequant_fp8_blockwise_f32(weight: *const u8, scale: *const f32, output: *mut f32, h: i32, w: i32, rs: i32, ss: i32, by: i32, bx: i32) { dequant_fp8_blockwise(weight, scale, output, h, w, rs, ss, by, bx) }
    #[kernel] pub unsafe fn quant_fp8_blockwise_f32(input: *const f32, weight: *mut u8, scale: *mut f32, h: i32, w: i32, rs: i32, ss: i32, by: i32, bx: i32) { quant_fp8_blockwise(input, weight, scale, h, w, rs, ss, by, bx) }
    #[kernel] pub unsafe fn fp8_to_f16(input: *const u8, output: *mut H, n: usize) { fp8_to_dtype(input, output, n) }
    #[kernel] pub unsafe fn f16_to_fp8(input: *const H, output: *mut u8, n: usize) { dtype_to_fp8(input, output, n) }
    #[kernel] pub unsafe fn dequant_fp8_vector_f16(weight: *const u8, scale: *const f32, output: *mut H, n: usize) { dequant_fp8_vector(weight, scale, output, n) }
    #[kernel] pub unsafe fn quant_fp8_vector_f16(input: *const H, weight: *mut u8, scale: *mut f32, n: usize) { quant_fp8_vector(input, weight, scale, n) }
    #[kernel] pub unsafe fn dequant_fp8_blockwise_f16(weight: *const u8, scale: *const f32, output: *mut H, h: i32, w: i32, rs: i32, ss: i32, by: i32, bx: i32) { dequant_fp8_blockwise(weight, scale, output, h, w, rs, ss, by, bx) }
    #[kernel] pub unsafe fn quant_fp8_blockwise_f16(input: *const H, weight: *mut u8, scale: *mut f32, h: i32, w: i32, rs: i32, ss: i32, by: i32, bx: i32) { quant_fp8_blockwise(input, weight, scale, h, w, rs, ss, by, bx) }
    #[kernel] pub unsafe fn fp8_to_bf16(input: *const u8, output: *mut B, n: usize) { fp8_to_dtype(input, output, n) }
    #[kernel] pub unsafe fn bf16_to_fp8(input: *const B, output: *mut u8, n: usize) { dtype_to_fp8(input, output, n) }
    #[kernel] pub unsafe fn dequant_fp8_vector_bf16(weight: *const u8, scale: *const f32, output: *mut B, n: usize) { dequant_fp8_vector(weight, scale, output, n) }
    #[kernel] pub unsafe fn quant_fp8_vector_bf16(input: *const B, weight: *mut u8, scale: *mut f32, n: usize) { quant_fp8_vector(input, weight, scale, n) }
    #[kernel] pub unsafe fn dequant_fp8_blockwise_bf16(weight: *const u8, scale: *const f32, output: *mut B, h: i32, w: i32, rs: i32, ss: i32, by: i32, bx: i32) { dequant_fp8_blockwise(weight, scale, output, h, w, rs, ss, by, bx) }
    #[kernel] pub unsafe fn quant_fp8_blockwise_bf16(input: *const B, weight: *mut u8, scale: *mut f32, h: i32, w: i32, rs: i32, ss: i32, by: i32, bx: i32) { quant_fp8_blockwise(input, weight, scale, h, w, rs, ss, by, bx) }
    #[kernel] pub unsafe fn fp8_matmul_f16(input: *const H, weight: *const u8, ws: *const f32, output: *mut H, m: i32, n: i32, kk: i32, srs: i32, by: i32, bx: i32) { fp8_matmul_tiled(input, weight, ws, output, m, n, kk, srs, by, bx) }
    #[kernel] pub unsafe fn fp8_moe_gemm_f16(input: *const H, weights: *const u8, ws: *const f32, indices: *const u32, output: *mut H, nt: i32, topk: i32, ne: i32, n: i32, kk: i32, srs: i32, by: i32, bx: i32, has_topk: u8) { fp8_moe_gemm(input, weights, ws, indices, output, nt, topk, ne, n, kk, srs, by, bx, has_topk) }
    #[kernel] pub unsafe fn fp8_matmul_bf16(input: *const B, weight: *const u8, ws: *const f32, output: *mut B, m: i32, n: i32, kk: i32, srs: i32, by: i32, bx: i32) { fp8_matmul_tiled(input, weight, ws, output, m, n, kk, srs, by, bx) }
    #[kernel] pub unsafe fn fp8_moe_gemm_bf16(input: *const B, weights: *const u8, ws: *const f32, indices: *const u32, output: *mut B, nt: i32, topk: i32, ne: i32, n: i32, kk: i32, srs: i32, by: i32, bx: i32, has_topk: u8) { fp8_moe_gemm(input, weights, ws, indices, output, nt, topk, ne, n, kk, srs, by, bx, has_topk) }
    #[kernel] pub unsafe fn gptq_gemm_2bit_m1(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, c: *mut u16, m: i32, n: i32, kk: i32, groups: i32, perm: *const i32) { gptq_gemm::<2, 1>(a, bq, qz, sc, c, m, n, kk, groups, perm) }
    #[kernel] pub unsafe fn gptq_gemm_2bit_m2(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, c: *mut u16, m: i32, n: i32, kk: i32, groups: i32, perm: *const i32) { gptq_gemm::<2, 2>(a, bq, qz, sc, c, m, n, kk, groups, perm) }
    #[kernel] pub unsafe fn gptq_gemm_2bit_m3(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, c: *mut u16, m: i32, n: i32, kk: i32, groups: i32, perm: *const i32) { gptq_gemm::<2, 3>(a, bq, qz, sc, c, m, n, kk, groups, perm) }
    #[kernel] pub unsafe fn gptq_gemm_2bit_m4(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, c: *mut u16, m: i32, n: i32, kk: i32, groups: i32, perm: *const i32) { gptq_gemm::<2, 4>(a, bq, qz, sc, c, m, n, kk, groups, perm) }
    #[kernel] pub unsafe fn gptq_gemm_2bit_m5(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, c: *mut u16, m: i32, n: i32, kk: i32, groups: i32, perm: *const i32) { gptq_gemm::<2, 5>(a, bq, qz, sc, c, m, n, kk, groups, perm) }
    #[kernel] pub unsafe fn gptq_gemm_2bit_m6(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, c: *mut u16, m: i32, n: i32, kk: i32, groups: i32, perm: *const i32) { gptq_gemm::<2, 6>(a, bq, qz, sc, c, m, n, kk, groups, perm) }
    #[kernel] pub unsafe fn gptq_gemm_2bit_m7(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, c: *mut u16, m: i32, n: i32, kk: i32, groups: i32, perm: *const i32) { gptq_gemm::<2, 7>(a, bq, qz, sc, c, m, n, kk, groups, perm) }
    #[kernel] pub unsafe fn gptq_gemm_2bit_m8(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, c: *mut u16, m: i32, n: i32, kk: i32, groups: i32, perm: *const i32) { gptq_gemm::<2, 8>(a, bq, qz, sc, c, m, n, kk, groups, perm) }
    #[kernel] pub unsafe fn reconstruct_exllama_2bit(bq: *const u32, perm: *const i32, qz: *const u32, sc: *const u16, size_k: i32, size_n: i32, groups: i32, out: *mut u16) { reconstruct_exllama::<2>(bq, perm, qz, sc, size_k, size_n, groups, out) }
    #[kernel] pub unsafe fn reconstruct_gptq_2bit(w: *const u32, sc: *const u16, qz: *const u32, g_idx: *const i32, height: i32, width: i32, group: i32, out: *mut u16) { reconstruct_gptq::<2>(w, sc, qz, g_idx, height, width, group, out) }
    #[kernel] pub unsafe fn gptq_shuffle_2bit(bq: *mut u32, size_k: i32, size_n: i32) { gptq_shuffle::<2>(bq, size_k, size_n) }
    #[kernel] pub unsafe fn make_sequential_2bit(w: *const u32, w_new: *mut u32, q_perm: *const i32, w_width: i32) { make_sequential::<2>(w, w_new, q_perm, w_width) }
    #[kernel] pub unsafe fn gptq_gemm_3bit_m1(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, c: *mut u16, m: i32, n: i32, kk: i32, groups: i32, perm: *const i32) { gptq_gemm::<3, 1>(a, bq, qz, sc, c, m, n, kk, groups, perm) }
    #[kernel] pub unsafe fn gptq_gemm_3bit_m2(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, c: *mut u16, m: i32, n: i32, kk: i32, groups: i32, perm: *const i32) { gptq_gemm::<3, 2>(a, bq, qz, sc, c, m, n, kk, groups, perm) }
    #[kernel] pub unsafe fn gptq_gemm_3bit_m3(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, c: *mut u16, m: i32, n: i32, kk: i32, groups: i32, perm: *const i32) { gptq_gemm::<3, 3>(a, bq, qz, sc, c, m, n, kk, groups, perm) }
    #[kernel] pub unsafe fn gptq_gemm_3bit_m4(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, c: *mut u16, m: i32, n: i32, kk: i32, groups: i32, perm: *const i32) { gptq_gemm::<3, 4>(a, bq, qz, sc, c, m, n, kk, groups, perm) }
    #[kernel] pub unsafe fn gptq_gemm_3bit_m5(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, c: *mut u16, m: i32, n: i32, kk: i32, groups: i32, perm: *const i32) { gptq_gemm::<3, 5>(a, bq, qz, sc, c, m, n, kk, groups, perm) }
    #[kernel] pub unsafe fn gptq_gemm_3bit_m6(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, c: *mut u16, m: i32, n: i32, kk: i32, groups: i32, perm: *const i32) { gptq_gemm::<3, 6>(a, bq, qz, sc, c, m, n, kk, groups, perm) }
    #[kernel] pub unsafe fn gptq_gemm_3bit_m7(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, c: *mut u16, m: i32, n: i32, kk: i32, groups: i32, perm: *const i32) { gptq_gemm::<3, 7>(a, bq, qz, sc, c, m, n, kk, groups, perm) }
    #[kernel] pub unsafe fn gptq_gemm_3bit_m8(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, c: *mut u16, m: i32, n: i32, kk: i32, groups: i32, perm: *const i32) { gptq_gemm::<3, 8>(a, bq, qz, sc, c, m, n, kk, groups, perm) }
    #[kernel] pub unsafe fn reconstruct_exllama_3bit(bq: *const u32, perm: *const i32, qz: *const u32, sc: *const u16, size_k: i32, size_n: i32, groups: i32, out: *mut u16) { reconstruct_exllama::<3>(bq, perm, qz, sc, size_k, size_n, groups, out) }
    #[kernel] pub unsafe fn reconstruct_gptq_3bit(w: *const u32, sc: *const u16, qz: *const u32, g_idx: *const i32, height: i32, width: i32, group: i32, out: *mut u16) { reconstruct_gptq::<3>(w, sc, qz, g_idx, height, width, group, out) }
    #[kernel] pub unsafe fn gptq_shuffle_3bit(bq: *mut u32, size_k: i32, size_n: i32) { gptq_shuffle::<3>(bq, size_k, size_n) }
    #[kernel] pub unsafe fn make_sequential_3bit(w: *const u32, w_new: *mut u32, q_perm: *const i32, w_width: i32) { make_sequential_3(w, w_new, q_perm, w_width) }
    #[kernel] pub unsafe fn gptq_gemm_4bit_m1(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, c: *mut u16, m: i32, n: i32, kk: i32, groups: i32, perm: *const i32) { gptq_gemm::<4, 1>(a, bq, qz, sc, c, m, n, kk, groups, perm) }
    #[kernel] pub unsafe fn gptq_gemm_4bit_m2(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, c: *mut u16, m: i32, n: i32, kk: i32, groups: i32, perm: *const i32) { gptq_gemm::<4, 2>(a, bq, qz, sc, c, m, n, kk, groups, perm) }
    #[kernel] pub unsafe fn gptq_gemm_4bit_m3(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, c: *mut u16, m: i32, n: i32, kk: i32, groups: i32, perm: *const i32) { gptq_gemm::<4, 3>(a, bq, qz, sc, c, m, n, kk, groups, perm) }
    #[kernel] pub unsafe fn gptq_gemm_4bit_m4(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, c: *mut u16, m: i32, n: i32, kk: i32, groups: i32, perm: *const i32) { gptq_gemm::<4, 4>(a, bq, qz, sc, c, m, n, kk, groups, perm) }
    #[kernel] pub unsafe fn gptq_gemm_4bit_m5(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, c: *mut u16, m: i32, n: i32, kk: i32, groups: i32, perm: *const i32) { gptq_gemm::<4, 5>(a, bq, qz, sc, c, m, n, kk, groups, perm) }
    #[kernel] pub unsafe fn gptq_gemm_4bit_m6(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, c: *mut u16, m: i32, n: i32, kk: i32, groups: i32, perm: *const i32) { gptq_gemm::<4, 6>(a, bq, qz, sc, c, m, n, kk, groups, perm) }
    #[kernel] pub unsafe fn gptq_gemm_4bit_m7(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, c: *mut u16, m: i32, n: i32, kk: i32, groups: i32, perm: *const i32) { gptq_gemm::<4, 7>(a, bq, qz, sc, c, m, n, kk, groups, perm) }
    #[kernel] pub unsafe fn gptq_gemm_4bit_m8(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, c: *mut u16, m: i32, n: i32, kk: i32, groups: i32, perm: *const i32) { gptq_gemm::<4, 8>(a, bq, qz, sc, c, m, n, kk, groups, perm) }
    #[kernel] pub unsafe fn reconstruct_exllama_4bit(bq: *const u32, perm: *const i32, qz: *const u32, sc: *const u16, size_k: i32, size_n: i32, groups: i32, out: *mut u16) { reconstruct_exllama::<4>(bq, perm, qz, sc, size_k, size_n, groups, out) }
    #[kernel] pub unsafe fn reconstruct_gptq_4bit(w: *const u32, sc: *const u16, qz: *const u32, g_idx: *const i32, height: i32, width: i32, group: i32, out: *mut u16) { reconstruct_gptq::<4>(w, sc, qz, g_idx, height, width, group, out) }
    #[kernel] pub unsafe fn gptq_shuffle_4bit(bq: *mut u32, size_k: i32, size_n: i32) { gptq_shuffle::<4>(bq, size_k, size_n) }
    #[kernel] pub unsafe fn make_sequential_4bit(w: *const u32, w_new: *mut u32, q_perm: *const i32, w_width: i32) { make_sequential::<4>(w, w_new, q_perm, w_width) }
    #[kernel] pub unsafe fn gptq_gemm_8bit_m1(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, c: *mut u16, m: i32, n: i32, kk: i32, groups: i32, perm: *const i32) { gptq_gemm::<8, 1>(a, bq, qz, sc, c, m, n, kk, groups, perm) }
    #[kernel] pub unsafe fn gptq_gemm_8bit_m2(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, c: *mut u16, m: i32, n: i32, kk: i32, groups: i32, perm: *const i32) { gptq_gemm::<8, 2>(a, bq, qz, sc, c, m, n, kk, groups, perm) }
    #[kernel] pub unsafe fn gptq_gemm_8bit_m3(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, c: *mut u16, m: i32, n: i32, kk: i32, groups: i32, perm: *const i32) { gptq_gemm::<8, 3>(a, bq, qz, sc, c, m, n, kk, groups, perm) }
    #[kernel] pub unsafe fn gptq_gemm_8bit_m4(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, c: *mut u16, m: i32, n: i32, kk: i32, groups: i32, perm: *const i32) { gptq_gemm::<8, 4>(a, bq, qz, sc, c, m, n, kk, groups, perm) }
    #[kernel] pub unsafe fn gptq_gemm_8bit_m5(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, c: *mut u16, m: i32, n: i32, kk: i32, groups: i32, perm: *const i32) { gptq_gemm::<8, 5>(a, bq, qz, sc, c, m, n, kk, groups, perm) }
    #[kernel] pub unsafe fn gptq_gemm_8bit_m6(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, c: *mut u16, m: i32, n: i32, kk: i32, groups: i32, perm: *const i32) { gptq_gemm::<8, 6>(a, bq, qz, sc, c, m, n, kk, groups, perm) }
    #[kernel] pub unsafe fn gptq_gemm_8bit_m7(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, c: *mut u16, m: i32, n: i32, kk: i32, groups: i32, perm: *const i32) { gptq_gemm::<8, 7>(a, bq, qz, sc, c, m, n, kk, groups, perm) }
    #[kernel] pub unsafe fn gptq_gemm_8bit_m8(a: *const u16, bq: *const u32, qz: *const u32, sc: *const u16, c: *mut u16, m: i32, n: i32, kk: i32, groups: i32, perm: *const i32) { gptq_gemm::<8, 8>(a, bq, qz, sc, c, m, n, kk, groups, perm) }
    #[kernel] pub unsafe fn reconstruct_exllama_8bit(bq: *const u32, perm: *const i32, qz: *const u32, sc: *const u16, size_k: i32, size_n: i32, groups: i32, out: *mut u16) { reconstruct_exllama::<8>(bq, perm, qz, sc, size_k, size_n, groups, out) }
    #[kernel] pub unsafe fn reconstruct_gptq_8bit(w: *const u32, sc: *const u16, qz: *const u32, g_idx: *const i32, height: i32, width: i32, group: i32, out: *mut u16) { reconstruct_gptq::<8>(w, sc, qz, g_idx, height, width, group, out) }
    #[kernel] pub unsafe fn gptq_shuffle_8bit(bq: *mut u32, size_k: i32, size_n: i32) { gptq_shuffle::<8>(bq, size_k, size_n) }
    #[kernel] pub unsafe fn make_sequential_8bit(w: *const u32, w_new: *mut u32, q_perm: *const i32, w_width: i32) { make_sequential::<8>(w, w_new, q_perm, w_width) }
    #[kernel] pub unsafe fn gemm_half_q_half_alt_4bit(vec: *const u32, mat: *const u32, mul_out: *mut u16, sc: *const u16, zeros: *const u32, g_idx: *const i32, batch: i32, height: i32, width: i32) { gptq_alt::<4>(vec, mat, mul_out, sc, zeros, g_idx, batch, height, width) }
    #[kernel] pub unsafe fn gemm_half_q_half_alt_8bit(vec: *const u32, mat: *const u32, mul_out: *mut u16, sc: *const u16, zeros: *const u32, g_idx: *const i32, batch: i32, height: i32, width: i32) { gptq_alt::<8>(vec, mat, mul_out, sc, zeros, g_idx, batch, height, width) }
    #[kernel] pub unsafe fn mxfp4_vecmat_f16(input: *const H, weight: *const u8, ws: *const u8, bias: *const H, output: *mut H, m: i32, n: i32, kk: i32, has_bias: u8) { mxfp4_vecmat(input, weight, ws, bias, output, m, n, kk, has_bias) }
    #[kernel] pub unsafe fn mxfp4_matmul_tiled_f16(input: *const H, weight: *const u8, ws: *const u8, bias: *const H, output: *mut H, m: i32, n: i32, kk: i32, has_bias: u8) { mxfp4_matmul_tiled(input, weight, ws, bias, output, m, n, kk, has_bias) }
    #[kernel] pub unsafe fn mxfp4_moe_gemm_f16(input: *const H, weights: *const u8, ws: *const u8, biases: *const H, indices: *const u32, output: *mut H, nt: i32, topk: i32, ne: i32, n: i32, kk: i32, has_bias: u8, has_topk: u8) { mxfp4_moe_gemm(input, weights, ws, biases, indices, output, nt, topk, ne, n, kk, has_bias, has_topk) }
    #[kernel] pub unsafe fn mxfp4_moe_grouped_tiled_f16(input: *const H, weights: *const u8, ws: *const u8, biases: *const H, indices: *const u32, output: *mut H, nt: i32, topk: i32, ne: i32, n: i32, kk: i32, has_bias: u8, has_topk: u8) { mxfp4_moe_grouped_tiled(input, weights, ws, biases, indices, output, nt, topk, ne, n, kk, has_bias, has_topk) }
    #[kernel] pub unsafe fn mxfp4_matmul_wmma_f16(input: *const H, weight: *const u8, ws: *const u8, bias: *const H, output: *mut H, m: i32, n: i32, kk: i32, has_bias: u8) { mxfp4_matmul_wmma(input, weight, ws, bias, output, m, n, kk, has_bias) }
    #[kernel] pub unsafe fn mxfp4_moe_grouped_wmma_f16(input: *const H, weights: *const u8, ws: *const u8, biases: *const H, indices: *const u32, output: *mut H, nt: i32, topk: i32, ne: i32, n: i32, kk: i32, has_bias: u8, has_topk: u8) { mxfp4_moe_grouped_wmma(input, weights, ws, biases, indices, output, nt, topk, ne, n, kk, has_bias, has_topk) }
    #[kernel] pub unsafe fn mxfp4_vecmat_bf16(input: *const B, weight: *const u8, ws: *const u8, bias: *const B, output: *mut B, m: i32, n: i32, kk: i32, has_bias: u8) { mxfp4_vecmat(input, weight, ws, bias, output, m, n, kk, has_bias) }
    #[kernel] pub unsafe fn mxfp4_matmul_tiled_bf16(input: *const B, weight: *const u8, ws: *const u8, bias: *const B, output: *mut B, m: i32, n: i32, kk: i32, has_bias: u8) { mxfp4_matmul_tiled(input, weight, ws, bias, output, m, n, kk, has_bias) }
    #[kernel] pub unsafe fn mxfp4_moe_gemm_bf16(input: *const B, weights: *const u8, ws: *const u8, biases: *const B, indices: *const u32, output: *mut B, nt: i32, topk: i32, ne: i32, n: i32, kk: i32, has_bias: u8, has_topk: u8) { mxfp4_moe_gemm(input, weights, ws, biases, indices, output, nt, topk, ne, n, kk, has_bias, has_topk) }
    #[kernel] pub unsafe fn mxfp4_moe_grouped_tiled_bf16(input: *const B, weights: *const u8, ws: *const u8, biases: *const B, indices: *const u32, output: *mut B, nt: i32, topk: i32, ne: i32, n: i32, kk: i32, has_bias: u8, has_topk: u8) { mxfp4_moe_grouped_tiled(input, weights, ws, biases, indices, output, nt, topk, ne, n, kk, has_bias, has_topk) }
    #[kernel] pub unsafe fn mxfp4_matmul_wmma_bf16(input: *const B, weight: *const u8, ws: *const u8, bias: *const B, output: *mut B, m: i32, n: i32, kk: i32, has_bias: u8) { mxfp4_matmul_wmma(input, weight, ws, bias, output, m, n, kk, has_bias) }
    #[kernel] pub unsafe fn mxfp4_moe_grouped_wmma_bf16(input: *const B, weights: *const u8, ws: *const u8, biases: *const B, indices: *const u32, output: *mut B, nt: i32, topk: i32, ne: i32, n: i32, kk: i32, has_bias: u8, has_topk: u8) { mxfp4_moe_grouped_wmma(input, weights, ws, biases, indices, output, nt, topk, ne, n, kk, has_bias, has_topk) }
    #[kernel] pub unsafe fn bnb_dequant_f32_int8(code: *const f32, a: *const u8, absmax: *const f32, out: *mut f32, blocksize: i32, n: i32) { bnb_dequant::<f32, 0>(code, a, absmax, out, blocksize, n) }
    #[kernel] pub unsafe fn bnb_dequant_f32_fp4(code: *const f32, a: *const u8, absmax: *const f32, out: *mut f32, blocksize: i32, n: i32) { bnb_dequant::<f32, 1>(code, a, absmax, out, blocksize, n) }
    #[kernel] pub unsafe fn bnb_dequant_f32_nf4(code: *const f32, a: *const u8, absmax: *const f32, out: *mut f32, blocksize: i32, n: i32) { bnb_dequant::<f32, 2>(code, a, absmax, out, blocksize, n) }
    #[kernel] pub unsafe fn bnb_dequant_f16_int8(code: *const f32, a: *const u8, absmax: *const f32, out: *mut H, blocksize: i32, n: i32) { bnb_dequant::<H, 0>(code, a, absmax, out, blocksize, n) }
    #[kernel] pub unsafe fn bnb_dequant_f16_fp4(code: *const f32, a: *const u8, absmax: *const f32, out: *mut H, blocksize: i32, n: i32) { bnb_dequant::<H, 1>(code, a, absmax, out, blocksize, n) }
    #[kernel] pub unsafe fn bnb_dequant_f16_nf4(code: *const f32, a: *const u8, absmax: *const f32, out: *mut H, blocksize: i32, n: i32) { bnb_dequant::<H, 2>(code, a, absmax, out, blocksize, n) }
    #[kernel] pub unsafe fn bnb_dequant_bf16_int8(code: *const f32, a: *const u8, absmax: *const f32, out: *mut B, blocksize: i32, n: i32) { bnb_dequant::<B, 0>(code, a, absmax, out, blocksize, n) }
    #[kernel] pub unsafe fn bnb_dequant_bf16_fp4(code: *const f32, a: *const u8, absmax: *const f32, out: *mut B, blocksize: i32, n: i32) { bnb_dequant::<B, 1>(code, a, absmax, out, blocksize, n) }
    #[kernel] pub unsafe fn bnb_dequant_bf16_nf4(code: *const f32, a: *const u8, absmax: *const f32, out: *mut B, blocksize: i32, n: i32) { bnb_dequant::<B, 2>(code, a, absmax, out, blocksize, n) }
    #[kernel] pub unsafe fn grouped_mm_2x_large(a: *const *const u16, b: *const *const u16, d: *const *mut u16, ps: *const i32, pc: i32, lda: *const i64, ldb: *const i64, ldd: *const i64) { grouped_mm(a, b, d, ps, pc, lda, ldb, ldd) }
    #[kernel] pub unsafe fn grouped_mm_2x_medium(a: *const *const u16, b: *const *const u16, d: *const *mut u16, ps: *const i32, pc: i32, lda: *const i64, ldb: *const i64, ldd: *const i64) { grouped_mm(a, b, d, ps, pc, lda, ldb, ldd) }
    #[kernel] pub unsafe fn grouped_mm_2x_small(a: *const *const u16, b: *const *const u16, d: *const *mut u16, ps: *const i32, pc: i32, lda: *const i64, ldb: *const i64, ldd: *const i64) { grouped_mm(a, b, d, ps, pc, lda, ldb, ldd) }
    // GENERATED KERNELS END
}

fn main() {
    std::process::exit(if gate::run() { 0 } else { 1 });
}
