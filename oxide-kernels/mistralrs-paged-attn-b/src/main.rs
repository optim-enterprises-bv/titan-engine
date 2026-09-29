//! mistralrs-paged-attn group 2 in cuda-oxide: the FlashInfer decode kernels of
//! `flashinfer_decode.cu` (BatchDecodeWithPagedKVCacheKernel for head dims 64/128/256/512, f16/bf16/f32,
//! GQA group sizes 1/2/3/4/8/16, sliding window x logits soft-cap, plus the split-KV
//! PersistentVariableLengthMergeStatesKernel and the reshape/gather cache helpers) and of
//! `flashinfer_mla_decode.cu` (BatchDecodeWithPagedKVCacheKernelMLA, ckv 512 / kpe 64), with
//! (src/launch.rs) pure-Rust twins of their extern "C" host launchers. Bit-identical to the nvcc build
//! (`-O3 --use_fast_math -DENABLE_FP8`, sm_120a; the SASS in libmistralrspagedattention.a is the spec,
//! regenerated PTX in ref/ reproduces it byte for byte), checked launcher-by-launcher by src/gate.rs.
//!
//! Every floating-point operation is written as the PTX instruction nvcc emits (`.ftz` everywhere,
//! `ex2.approx.ftz`, `lg2.approx.ftz`, `rcp.approx.ftz`, `tanh.approx`, `div.approx.ftz`), and every
//! contraction nvcc made is an explicit `fma.rn.ftz` at the same place, so ptxas sees the same data flow:
//! - q.k: `fma(q0, k0, 0)` then `fma(q_i, k_i, acc)`; butterfly `add(s, shfl_xor(s))`; `s * sm_scale_log2`.
//! - online softmax: `d = fma(o_scale, d, p0)`, then `d + p_j`; `o = fma(o_scale, o, p0 * v)` then
//!   `fma(p_j, v, o)` (nvcc contracts the `o * o_scale` of compute_qk into the first update).
//! - state merges: `d = fma(d_prev, e1, d_other * e2)`, `o = fma(o, e1, o_other * e2)`; a merge into the
//!   freshly initialised state (m = -5e4, d = 1, o = 0) is constant-folded by nvcc: `d = fma(d_other, e2, e1)`
//!   (or `e1 + e2` when d_other = 1) and `o = fma(e1, 0, o_other * e2)`.
//! - `math::inf` is 5e4, not infinity: masked logits are -5e4 and the output guard is `m != -5e4`.
#![allow(non_snake_case, clippy::missing_safety_doc, clippy::too_many_arguments)]
mod gate;
pub mod instances;
pub mod launch;

use cuda_device::{DynamicSharedArray, kernel, ptx_asm, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    // ------------------------------------------------------------------------------------------
    // exact nvcc PTX primitives (fast-math: everything .ftz)

    #[inline(always)]
    pub fn fmul(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("mul.ftz.f32 %0, %1, %2; //\t\x0b\x0c\r\n", out("=f") r, in("f") a, in("f") b, options(register_only)) };
        r
    }
    #[inline(always)]
    pub fn fadd(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("add.ftz.f32 %0, %1, %2; //\t\x0b\x0c\r\n", out("=f") r, in("f") a, in("f") b, options(register_only)) };
        r
    }
    #[inline(always)]
    pub fn fsub(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("sub.ftz.f32 %0, %1, %2; //\t\x0b\x0c\r\n", out("=f") r, in("f") a, in("f") b, options(register_only)) };
        r
    }
    #[inline(always)]
    pub fn ffma(a: f32, b: f32, c: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("fma.rn.ftz.f32 %0, %1, %2, %3; //\t\x0b\x0c\r\n", out("=f") r, in("f") a, in("f") b, in("f") c, options(register_only)) };
        r
    }
    #[inline(always)]
    pub fn fmax(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("max.ftz.f32 %0, %1, %2; //\t\x0b\x0c\r\n", out("=f") r, in("f") a, in("f") b, options(register_only)) };
        r
    }
    #[inline(always)]
    pub fn ex2(a: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("ex2.approx.ftz.f32 %0, %1; //\t\x0b\x0c\r\n", out("=f") r, in("f") a, options(register_only)) };
        r
    }
    #[inline(always)]
    pub fn lg2(a: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("lg2.approx.ftz.f32 %0, %1; //\t\x0b\x0c\r\n", out("=f") r, in("f") a, options(register_only)) };
        r
    }
    #[inline(always)]
    pub fn rcp(a: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("rcp.approx.ftz.f32 %0, %1; //\t\x0b\x0c\r\n", out("=f") r, in("f") a, options(register_only)) };
        r
    }
    #[inline(always)]
    pub fn tanh_approx(a: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("tanh.approx.f32 %0, %1; //\t\x0b\x0c\r\n", out("=f") r, in("f") a, options(register_only)) };
        r
    }
    /// `__fdividef` under fast-math.
    #[inline(always)]
    pub fn fdiv(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("div.approx.ftz.f32 %0, %1, %2; //\t\x0b\x0c\r\n", out("=f") r, in("f") a, in("f") b, options(register_only)) };
        r
    }
    /// `setp.eq.ftz.f32` as a bool.
    #[inline(always)]
    pub fn eq_ftz(a: f32, b: f32) -> bool {
        let r: u32;
        unsafe {
            ptx_asm!("{ .reg .pred p; setp.eq.ftz.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }", out("=r") r, in("f") a, in("f") b, options(register_only))
        };
        r != 0
    }
    /// `shfl.sync.bfly.b32 d, x, lane_mask, 0x1f, 0xffffffff` (math::shfl_xor_sync).
    #[inline(always)]
    pub fn shfl_xor(x: f32, m: u32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff; //\t\x0b\x0c\r\n", out("=f") r, in("f") x, in("r") m, options(register_only)) };
        r
    }
    #[inline(always)]
    pub fn h2f(h: u16) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("cvt.f32.f16 %0, %1; //\t\x0b\x0c\r\n", out("=f") r, in("h") h, options(register_only)) };
        r
    }
    #[inline(always)]
    pub fn bf2f(h: u16) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("cvt.f32.bf16 %0, %1; //\t\x0b\x0c\r\n", out("=f") r, in("h") h, options(register_only)) };
        r
    }
    /// `cvt.rn.f16x2.f32 d, hi, lo` (`__float22half2_rn`).
    #[inline(always)]
    pub fn f2h2(lo: f32, hi: f32) -> u32 {
        let r: u32;
        unsafe { ptx_asm!("cvt.rn.f16x2.f32 %0, %1, %2; //\t\x0b\x0c\r\n", out("=r") r, in("f") hi, in("f") lo, options(register_only)) };
        r
    }
    /// `cvt.rn.bf16x2.f32 d, hi, lo`.
    #[inline(always)]
    pub fn f2bf2(lo: f32, hi: f32) -> u32 {
        let r: u32;
        unsafe { ptx_asm!("cvt.rn.bf16x2.f32 %0, %1, %2; //\t\x0b\x0c\r\n", out("=r") r, in("f") hi, in("f") lo, options(register_only)) };
        r
    }
    /// `shr.u32` (shift counts >= 32 give 0, as in PTX).
    #[inline(always)]
    pub fn shr_u32(a: u32, s: u32) -> u32 {
        let r: u32;
        unsafe { ptx_asm!("shr.u32 %0, %1, %2; //\t\x0b\x0c\r\n", out("=r") r, in("r") a, in("r") s, options(register_only)) };
        r
    }
    #[inline(always)]
    pub fn sdiv(a: i32, b: i32) -> i32 {
        let r: i32;
        unsafe { ptx_asm!("div.s32 %0, %1, %2; //\t\x0b\x0c\r\n", out("=r") r, in("r") a, in("r") b, options(register_only)) };
        r
    }
    #[inline(always)]
    pub fn udiv(a: u32, b: u32) -> u32 {
        let r: u32;
        unsafe { ptx_asm!("div.u32 %0, %1, %2; //\t\x0b\x0c\r\n", out("=r") r, in("r") a, in("r") b, options(register_only)) };
        r
    }
    #[inline(always)]
    pub fn sdiv64_raw(a: i64, b: i64) -> i64 {
        let r: i64;
        unsafe { ptx_asm!("div.s64 %0, %1, %2; //\t\x0b\x0c\r\n", out("=l") r, in("l") a, in("l") b, options(register_only)) };
        r
    }
    /// nvcc's signed 64-bit `/` and `%`: `div.u32` when both operands fit in 32 unsigned bits.
    #[inline(always)]
    pub fn divrem64(a: i64, b: i64) -> (i64, i64) {
        if ((a | b) as u64) >> 32 == 0 {
            let q = udiv(a as u32, b as u32);
            let r = (a as u32).wrapping_sub(q.wrapping_mul(b as u32));
            (q as u64 as i64, r as u64 as i64)
        } else {
            let q = sdiv64_raw(a, b);
            (q, a.wrapping_sub(q.wrapping_mul(b)))
        }
    }

    /// `-math::inf`
    pub const NEG_INF: f32 = -5e4;
    pub const LOG2E: f32 = f32::from_bits(0x3FB8AA3B);

    // ------------------------------------------------------------------------------------------
    // shared memory (32-bit shared-window addresses) and cp.async

    #[inline(always)]
    pub unsafe fn lds32(a: u32) -> u32 {
        let r: u32;
        ptx_asm!("ld.shared.b32 %0, [%1]; //\t\x0b\x0c\r\n", out("=r") r, in("r") a, clobber("memory"));
        r
    }
    #[inline(always)]
    pub unsafe fn lds64(a: u32) -> u64 {
        let r: u64;
        ptx_asm!("ld.shared.b64 %0, [%1]; //\t\x0b\x0c\r\n", out("=l") r, in("r") a, clobber("memory"));
        r
    }
    #[inline(always)]
    pub unsafe fn lds_v2(a: u32) -> (u32, u32) {
        let x: u32;
        let y: u32;
        ptx_asm!("ld.shared.v2.b32 {%0, %1}, [%2]; //\t\x0b\x0c\r\n", out("=r") x, out("=r") y, in("r") a, clobber("memory"));
        (x, y)
    }
    #[inline(always)]
    pub unsafe fn lds_v4(a: u32) -> (u32, u32, u32, u32) {
        let x: u32;
        let y: u32;
        let z: u32;
        let w: u32;
        ptx_asm!("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4]; //\t\x0b\x0c\r\n", out("=r") x, out("=r") y, out("=r") z, out("=r") w, in("r") a, clobber("memory"));
        (x, y, z, w)
    }
    #[inline(always)]
    pub unsafe fn sts32(a: u32, v: u32) {
        ptx_asm!("st.shared.b32 [%0], %1; //\t\x0b\x0c\r\n", in("r") a, in("r") v, clobber("memory"));
    }
    #[inline(always)]
    pub unsafe fn sts64(a: u32, v: u64) {
        ptx_asm!("st.shared.b64 [%0], %1; //\t\x0b\x0c\r\n", in("r") a, in("l") v, clobber("memory"));
    }
    #[inline(always)]
    pub unsafe fn sts_v2(a: u32, x: u32, y: u32) {
        ptx_asm!("st.shared.v2.b32 [%0], {%1, %2}; //\t\x0b\x0c\r\n", in("r") a, in("r") x, in("r") y, clobber("memory"));
    }
    #[inline(always)]
    pub unsafe fn sts_v4(a: u32, x: u32, y: u32, z: u32, w: u32) {
        ptx_asm!("st.shared.v4.b32 [%0], {%1, %2, %3, %4}; //\t\x0b\x0c\r\n", in("r") a, in("r") x, in("r") y, in("r") z, in("r") w, clobber("memory"));
    }
    /// `pred_load_128b<kPrefetch, kNoFill>`: predicated 16-byte cp.async (shared keeps old data when false).
    #[inline(always)]
    pub unsafe fn cp16_nofill(sa: u32, g: *const u8, pred: bool) {
        let p = pred as u32;
        ptx_asm!(
            "{ .reg .pred p; setp.ne.b32 p, %0, 0; @p cp.async.cg.shared.global.L2::128B [%1], [%2], 16; }",
            in("r") p, in("r") sa, in("l") g, clobber("memory")
        );
    }
    /// `pred_load_128b<kPrefetch, kFillZero>`: 16-byte cp.async with src-size 16 or 0 (zero fill).
    #[inline(always)]
    pub unsafe fn cp16_zfill(sa: u32, g: *const u8, pred: bool) {
        let n: u32 = if pred { 16 } else { 0 };
        ptx_asm!("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2; //\t\x0b\x0c\r\n", in("r") sa, in("l") g, in("r") n, clobber("memory"));
    }
    /// `cp_async::pred_load<num_bits, kPrefetch, fill>` for `bytes` = num_bits / 8 (16, 32 or 64).
    #[inline(always)]
    pub unsafe fn pred_load<const ZFILL: bool>(sa: u32, g: *const u8, pred: bool, bytes: u32) {
        let mut c: u32 = 0;
        while c < 4 {
            cuda_device::thread::__unroll_config::<0>();
            if c * 16 < bytes {
                if ZFILL {
                    cp16_zfill(sa + c * 16, g.wrapping_add((c * 16) as usize), pred);
                } else {
                    cp16_nofill(sa + c * 16, g.wrapping_add((c * 16) as usize), pred);
                }
            }
            c += 1;
        }
    }
    #[inline(always)]
    pub unsafe fn commit_group() {
        ptx_asm!("cp.async.commit_group; //\t\x0b\x0c\r\n", clobber("memory"));
    }
    #[inline(always)]
    pub unsafe fn wait_group(n: u32) {
        match n {
            0 => ptx_asm!("cp.async.wait_group 0; //\t\x0b\x0c\r\n", clobber("memory")),
            1 => ptx_asm!("cp.async.wait_group 1; //\t\x0b\x0c\r\n", clobber("memory")),
            2 => ptx_asm!("cp.async.wait_group 2; //\t\x0b\x0c\r\n", clobber("memory")),
            _ => ptx_asm!("cp.async.wait_group 3; //\t\x0b\x0c\r\n", clobber("memory")),
        }
    }
    #[inline(always)]
    pub fn sync() {
        thread::sync_threads();
    }

    // ------------------------------------------------------------------------------------------
    // element types and vec_t cast_load / cast_store

    /// f16 element (bit pattern).
    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct H(pub u16);
    /// bf16 element (bit pattern).
    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct B(pub u16);

    pub trait El: Copy {
        /// sizeof(T)
        const SZ: u32;
        /// element `k` of a 32-bit word, widened to f32 (exact)
        fn w2f(w: u32, k: u32) -> f32;
        /// word holding (lo, hi) rounded to T (`__float22half2_rn` / bf16 / f32: lo only)
        fn f2w(lo: f32, hi: f32) -> u32;
    }
    impl El for f32 {
        const SZ: u32 = 4;
        #[inline(always)]
        fn w2f(w: u32, _k: u32) -> f32 {
            f32::from_bits(w)
        }
        #[inline(always)]
        fn f2w(lo: f32, _hi: f32) -> u32 {
            lo.to_bits()
        }
    }
    impl El for H {
        const SZ: u32 = 2;
        #[inline(always)]
        fn w2f(w: u32, k: u32) -> f32 {
            h2f(if k == 0 { w as u16 } else { (w >> 16) as u16 })
        }
        #[inline(always)]
        fn f2w(lo: f32, hi: f32) -> u32 {
            f2h2(lo, hi)
        }
    }
    impl El for B {
        const SZ: u32 = 2;
        #[inline(always)]
        fn w2f(w: u32, k: u32) -> f32 {
            bf2f(if k == 0 { w as u16 } else { (w >> 16) as u16 })
        }
        #[inline(always)]
        fn f2w(lo: f32, hi: f32) -> u32 {
            f2bf2(lo, hi)
        }
    }

    /// Raw words of a `bytes`-byte vector at shared address `a`.
    #[inline(always)]
    pub unsafe fn lds_words(a: u32, bytes: u32) -> [u32; 16] {
        let mut w = [0u32; 16];
        if bytes >= 16 {
            let mut c: usize = 0;
            while c < 4 {
                cuda_device::thread::__unroll_config::<0>();
                if (c as u32) * 16 < bytes {
                    let (x, y, z, u) = lds_v4(a + (c as u32) * 16);
                    w[c * 4] = x;
                    w[c * 4 + 1] = y;
                    w[c * 4 + 2] = z;
                    w[c * 4 + 3] = u;
                }
                c += 1;
            }
        } else if bytes == 8 {
            let (x, y) = lds_v2(a);
            w[0] = x;
            w[1] = y;
        } else {
            w[0] = lds32(a);
        }
        w
    }
    /// Store the first `bytes / 4` words at shared address `a`.
    #[inline(always)]
    pub unsafe fn sts_words(a: u32, w: &[u32; 16], bytes: u32) {
        if bytes >= 16 {
            let mut c: usize = 0;
            while c < 4 {
                cuda_device::thread::__unroll_config::<0>();
                if (c as u32) * 16 < bytes {
                    sts_v4(a + (c as u32) * 16, w[c * 4], w[c * 4 + 1], w[c * 4 + 2], w[c * 4 + 3]);
                }
                c += 1;
            }
        } else if bytes == 8 {
            sts_v2(a, w[0], w[1]);
        } else {
            sts32(a, w[0]);
        }
    }
    /// Raw words of a vector in global (generic) memory.
    #[inline(always)]
    pub unsafe fn ldg_words(p: *const u8, bytes: u32) -> [u32; 16] {
        let mut w = [0u32; 16];
        let q = p as *const u32;
        let mut i: usize = 0;
        while i < 16 {
            cuda_device::thread::__unroll_config::<0>();
            if (i as u32) * 4 < bytes {
                w[i] = *q.add(i);
            }
            i += 1;
        }
        w
    }
    #[inline(always)]
    pub unsafe fn stg_words(p: *mut u8, w: &[u32; 16], bytes: u32) {
        let q = p as *mut u32;
        let mut i: usize = 0;
        while i < 16 {
            cuda_device::thread::__unroll_config::<0>();
            if (i as u32) * 4 < bytes {
                *q.add(i) = w[i];
            }
            i += 1;
        }
    }
    /// vec_t<float, V>::cast_load from T words.
    #[inline(always)]
    pub fn words_to_f<T: El, const V: u32>(w: &[u32; 16]) -> [f32; 16] {
        let mut f = [0f32; 16];
        let epw = (4 / T::SZ) as usize;
        let mut i: usize = 0;
        while i < Cv::<V>::U {
            cuda_device::thread::__unroll_config::<0>();
            if true {
                f[i] = T::w2f(w[i / epw], (i % epw) as u32);
            }
            i += 1;
        }
        f
    }
    /// vec_t<float, V>::cast_store to T words.
    #[inline(always)]
    pub fn f_to_words<T: El, const V: u32>(f: &[f32; 16]) -> [u32; 16] {
        let mut w = [0u32; 16];
        let epw = (4 / T::SZ) as usize;
        let mut i: usize = 0;
        while i < Cv::<V>::U {
            cuda_device::thread::__unroll_config::<0>();
            if true && i % epw == 0 {
                w[i / epw] = T::f2w(f[i], if epw == 2 { f[i + 1] } else { 0.0 });
            }
            i += 1;
        }
        w
    }
    #[inline(always)]
    pub unsafe fn lds_vec<T: El, const V: u32>(a: u32) -> [f32; 16] {
        words_to_f::<T, V>(&lds_words(a, V * T::SZ))
    }
    #[inline(always)]
    pub unsafe fn ldg_vec<T: El, const V: u32>(p: *const T) -> [f32; 16] {
        words_to_f::<T, V>(&ldg_words(p as *const u8, V * T::SZ))
    }
    #[inline(always)]
    pub unsafe fn stg_vec<T: El, const V: u32>(p: *mut T, f: &[f32; 16]) {
        stg_words(p as *mut u8, &f_to_words::<T, V>(f), V * T::SZ)
    }
    #[inline(always)]
    pub unsafe fn sts_vec<T: El, const V: u32>(a: u32, f: &[f32; 16]) {
        sts_words(a, &f_to_words::<T, V>(f), V * T::SZ)
    }

    // ------------------------------------------------------------------------------------------
    // paged_kv_t helpers

    /// `uint_fastdiv::divmod`.
    #[inline(always)]
    pub fn divmod(n: u32, d: u32, m: u32, s: u32, a: u32) -> (u32, u32) {
        let q = if d == 1 {
            n
        } else {
            let hi = ((m as u64 * n as u64) >> 32) as u32;
            shr_u32(n.wrapping_mul(a).wrapping_add(hi), s)
        };
        (q, n.wrapping_sub(q.wrapping_mul(d)))
    }
    /// `paged_kv.get_length(batch_idx)`.
    #[inline(always)]
    pub unsafe fn get_length(indptr: *const i32, last_page_len: *const i32, b: u32, page_size: u32) -> u32 {
        let hi = *indptr.add(b.wrapping_add(1) as usize);
        let lo = *indptr.add(b as usize);
        if hi == lo {
            0
        } else {
            (hi.wrapping_sub(lo).wrapping_sub(1) as u32).wrapping_mul(page_size).wrapping_add(*last_page_len.add(b as usize) as u32)
        }
    }
    /// `protective_get_kv_offset` / `protective_get_offset_{ckv,kpe}`: page * stride_page + a + b.
    #[inline(always)]
    pub unsafe fn prot_off(page_iter: u32, last_indptr: i32, indices: *const i32, stride_page: u32, extra: u64) -> u64 {
        if (page_iter as i32) < last_indptr {
            let page = *indices.offset(page_iter as i32 as isize) as i64 as u64;
            page.wrapping_mul(stride_page as u64).wrapping_add(extra)
        } else {
            0
        }
    }

    // ------------------------------------------------------------------------------------------
    // BatchDecodeWithPagedKVCacheKernel<kNone, STAGES, TILE, VEC, BDX, BDY, BDZ,
    //   DefaultAttention<false, SW, SC, false>, BatchDecodeParams<T, T, T, int>>

    /// A const generic as a usize constant (a loop bound the unroll pass recognises).
    pub struct Cv<const N: u32>;
    impl<const N: u32> Cv<N> {
        pub const U: usize = N as usize;
    }
    /// Compile-time sizes of one decode instance.
    pub struct Dc<const VEC: u32, const BDX: u32, const BDY: u32, const BDZ: u32, const TILE: u32>;
    impl<const VEC: u32, const BDX: u32, const BDY: u32, const BDZ: u32, const TILE: u32> Dc<VEC, BDX, BDY, BDZ, TILE> {
        /// rows handled by compute_qk per thread (bdy * tile_size_per_bdx)
        pub const TS: usize = (BDY * TILE) as usize;
        pub const HD: u32 = BDX * VEC;
        pub const BDXL: u32 = if BDX >= 2 { BDX / 2 } else { 0 };
    }

    #[inline(always)]
    pub unsafe fn batch_decode<
        T: El,
        const STAGES: u32,
        const TILE: u32,
        const VEC: u32,
        const BDX: u32,
        const BDY: u32,
        const BDZ: u32,
        const SW: bool,
        const SC: bool,
    >(
        q: *const T, k_data: *const T, v_data: *const T, indices: *const i32, indptr: *const i32, last_page_len: *const i32, o: *mut T,
        lse: *mut f32, request_indices: *const i32, kv_tile_indices: *const i32, kv_chunk_size_ptr: *const i32, block_valid_mask: *const u8,
        num_qo_heads: u32, q_stride_n: i32, q_stride_h: i32, window_left_p: i32, logits_soft_cap: f32, sm_scale: f32, partition_kv: u32,
        fd_d: u32, fd_m: u32, fd_s: u32, fd_a: u32, batch_size: u32, stride_page: u32, stride_n: u32, stride_h: u32,
    ) {
        let smem = DynamicSharedArray::<u8, 16>::get_raw();
        let sbase = cuda_device::shared::cvta_generic_to_shared_u32(smem as *const u8);
        let hd: u32 = BDX * VEC;
        let sz = T::SZ;
        let vbytes = VEC * sz;
        let ts = Dc::<VEC, BDX, BDY, BDZ, TILE>::TS;
        let bx = thread::blockIdx_x();
        let by = thread::blockIdx_y();
        let tx = thread::threadIdx_x();
        let ty = thread::threadIdx_y();
        let tz = thread::threadIdx_z();
        let partition = partition_kv != 0;

        let batch_idx = *request_indices.add(bx as usize) as u32;
        let kv_tile_idx = *kv_tile_indices.add(bx as usize) as u32;
        let kv_head_idx = by;
        let qo_head_idx = kv_head_idx.wrapping_mul(BDY).wrapping_add(ty);
        if !block_valid_mask.is_null() && *block_valid_mask.add(bx as usize) == 0 {
            return;
        }
        let kv_chunk_size = *kv_chunk_size_ptr as u32;
        let kv_len = get_length(indptr, last_page_len, batch_idx, fd_d);
        let max_chunk_size = if partition { kv_chunk_size } else { kv_len };
        let chunk_start = if partition { kv_tile_idx.wrapping_mul(max_chunk_size) } else { 0 };
        let chunk_end = if partition { kv_tile_idx.wrapping_add(1).wrapping_mul(max_chunk_size).min(kv_len) } else { kv_len };
        let chunk_size = chunk_end.wrapping_sub(chunk_start);

        // DefaultAttention closure
        let (sm_scale_log2, soft_cap_pre_tanh_scale) =
            if SC { (fmul(LOG2E, logits_soft_cap), fmul(sm_scale, rcp(logits_soft_cap))) } else { (fmul(sm_scale, LOG2E), 0.0) };
        let window_left: u32 = if window_left_p >= 0 { window_left_p as u32 } else { kv_len };

        let stage_bytes = TILE * BDY * BDZ * hd * sz;
        let k_smem = sbase;
        let v_smem = sbase + STAGES * stage_bytes;
        let off_smem = sbase + 2 * STAGES * stage_bytes;

        let qoff = (batch_idx.wrapping_mul(q_stride_n as u32) as u64)
            .wrapping_add(qo_head_idx.wrapping_mul(q_stride_h as u32) as u64)
            .wrapping_add((tx * VEC) as u64);
        let q_vec = ldg_vec::<T, VEC>(q.wrapping_add(qoff as usize));

        let last_indptr = *indptr.add(batch_size as usize);
        let packed_page_iter_base = (*indptr.add(batch_idx as usize) as u32).wrapping_mul(fd_d).wrapping_add(chunk_start);
        let head_off = (kv_head_idx as u64).wrapping_mul(stride_h as u64);
        let fill_offsets = |base: u32| {
            let mut j: u32 = 0;
            while j < TILE {
                cuda_device::thread::__unroll_config::<0>();
                let e = ((j * BDZ + tz) * BDY + ty) * BDX + tx;
                let (pq, pr) = divmod(base.wrapping_add(e), fd_d, fd_m, fd_s, fd_a);
                let off = prot_off(pq, last_indptr, indices, stride_page, head_off.wrapping_add((pr as u64).wrapping_mul(stride_n as u64)));
                sts64(off_smem + e * 8, off);
                j += 1;
            }
        };
        fill_offsets(packed_page_iter_base);
        sync();

        let row_bytes = hd * sz;
        let mut stage_idx: u32 = 0;
        let mut it: u32 = 0;
        while it < STAGES {
            cuda_device::thread::__unroll_config::<0>();
            let mut kv_offset = [0u64; 4];
            let mut j: u32 = 0;
            while j < TILE {
                cuda_device::thread::__unroll_config::<0>();
                kv_offset[j as usize] = lds64(off_smem + (((it * BDZ + tz) * BDY + ty) * TILE + j) * 8).wrapping_add((tx * VEC) as u64);
                j += 1;
            }
            let mut j: u32 = 0;
            while j < TILE {
                cuda_device::thread::__unroll_config::<0>();
                let sa = k_smem + (((stage_idx * BDZ + tz) * BDY + ty) * TILE + j) * row_bytes + tx * vbytes;
                let g = (k_data as *const u8).wrapping_add((kv_offset[j as usize].wrapping_mul(sz as u64)) as usize);
                pred_load::<false>(sa, g, ((it * BDZ + tz) * BDY + ty) * TILE + j < chunk_size, vbytes);
                j += 1;
            }
            commit_group();
            let mut j: u32 = 0;
            while j < TILE {
                cuda_device::thread::__unroll_config::<0>();
                let sa = v_smem + (((stage_idx * BDZ + tz) * BDY + ty) * TILE + j) * row_bytes + tx * vbytes;
                let g = (v_data as *const u8).wrapping_add((kv_offset[j as usize].wrapping_mul(sz as u64)) as usize);
                pred_load::<true>(sa, g, ((it * BDZ + tz) * BDY + ty) * TILE + j < chunk_size, vbytes);
                j += 1;
            }
            commit_group();
            stage_idx = (stage_idx + 1) % STAGES;
            it += 1;
        }

        let mut st_o = [0f32; 16];
        let mut st_m: f32 = NEG_INF;
        let mut st_d: f32 = 1.0;
        let per_iter = TILE * BDY * BDZ;
        let n_iter = chunk_size.wrapping_add(per_iter - 1) / per_iter;
        let mut iter: u32 = 0;
        while iter < n_iter {
            if (iter + STAGES) % BDX == 0 {
                fill_offsets(packed_page_iter_base.wrapping_add((iter + STAGES).wrapping_mul(per_iter)));
            }
            // compute qk
            wait_group(2 * STAGES - 1);
            sync();
            let kst = k_smem + (stage_idx * BDZ + tz) * BDY * TILE * row_bytes;
            let kv_idx_base = chunk_start.wrapping_add(iter.wrapping_mul(per_iter));
            let iter_base = iter.wrapping_mul(per_iter);
            let mut s = [0f32; 16];
            let m_prev = st_m;
            let mut j: usize = 0;
            while j < Dc::<VEC, BDX, BDY, BDZ, TILE>::TS {
                cuda_device::thread::__unroll_config::<0>();
                if true {
                    let k_vec = lds_vec::<T, VEC>(kst + (j as u32 * BDX + tx) * vbytes);
                    let mut acc = ffma(q_vec[0], k_vec[0], 0.0);
                    let mut i: usize = 1;
                    while i < Cv::<VEC>::U {
                        cuda_device::thread::__unroll_config::<0>();
                        if true {
                            acc = ffma(q_vec[i], k_vec[i], acc);
                        }
                        i += 1;
                    }
                    let mut k: u32 = 0;
                    while k < 5 {
                        cuda_device::thread::__unroll_config::<0>();
                        let off = 16u32 >> k;
                        if off < BDX {
                            acc = fadd(acc, shfl_xor(acc, off));
                        }
                        k += 1;
                    }
                    let pos = kv_idx_base.wrapping_add(tz * ts as u32 + j as u32);
                    if SC {
                        acc = tanh_approx(fmul(acc, soft_cap_pre_tanh_scale));
                    }
                    acc = fmul(sm_scale_log2, acc);
                    let mask = if SW { pos.wrapping_add(1).wrapping_add(window_left) >= kv_len } else { true };
                    acc = if iter_base.wrapping_add(tz * ts as u32 + j as u32) < chunk_size && mask { acc } else { NEG_INF };
                    s[j] = acc;
                    st_m = fmax(st_m, acc);
                }
                j += 1;
            }
            let o_scale = ex2(fsub(m_prev, st_m));
            let mut j: usize = 0;
            while j < Dc::<VEC, BDX, BDY, BDZ, TILE>::TS {
                cuda_device::thread::__unroll_config::<0>();
                if true {
                    s[j] = ex2(fsub(s[j], st_m));
                    st_d = if j == 0 { ffma(o_scale, st_d, s[0]) } else { fadd(s[j], st_d) };
                }
                j += 1;
            }
            sync();

            let mut kv_offset = [0u64; 4];
            let slot = (iter + STAGES) % BDX;
            let mut j: u32 = 0;
            while j < TILE {
                cuda_device::thread::__unroll_config::<0>();
                kv_offset[j as usize] = lds64(off_smem + (((slot * BDZ + tz) * BDY + ty) * TILE + j) * 8).wrapping_add((tx * VEC) as u64);
                j += 1;
            }
            let mut j: u32 = 0;
            while j < TILE {
                cuda_device::thread::__unroll_config::<0>();
                let sa = k_smem + (((stage_idx * BDZ + tz) * BDY + ty) * TILE + j) * row_bytes + tx * vbytes;
                let g = (k_data as *const u8).wrapping_add((kv_offset[j as usize].wrapping_mul(sz as u64)) as usize);
                pred_load::<false>(sa, g, ((iter.wrapping_add(STAGES).wrapping_mul(BDZ) + tz) * BDY + ty).wrapping_mul(TILE) + j < chunk_size, vbytes);
                j += 1;
            }
            commit_group();

            // update m/d/o states
            wait_group(2 * STAGES - 1);
            sync();
            let vst = v_smem + (stage_idx * BDZ + tz) * BDY * TILE * row_bytes;
            let mut j: usize = 0;
            while j < Dc::<VEC, BDX, BDY, BDZ, TILE>::TS {
                cuda_device::thread::__unroll_config::<0>();
                if true {
                    let v_vec = lds_vec::<T, VEC>(vst + (j as u32 * BDX + tx) * vbytes);
                    let mut i: usize = 0;
                    while i < Cv::<VEC>::U {
                        cuda_device::thread::__unroll_config::<0>();
                        if true {
                            st_o[i] = if j == 0 { ffma(o_scale, st_o[i], fmul(s[0], v_vec[i])) } else { ffma(s[j], v_vec[i], st_o[i]) };
                        }
                        i += 1;
                    }
                }
                j += 1;
            }
            sync();

            let mut j: u32 = 0;
            while j < TILE {
                cuda_device::thread::__unroll_config::<0>();
                let sa = v_smem + (((stage_idx * BDZ + tz) * BDY + ty) * TILE + j) * row_bytes + tx * vbytes;
                let g = (v_data as *const u8).wrapping_add((kv_offset[j as usize].wrapping_mul(sz as u64)) as usize);
                pred_load::<true>(sa, g, ((iter.wrapping_add(STAGES).wrapping_mul(BDZ) + tz) * BDY + ty).wrapping_mul(TILE) + j < chunk_size, vbytes);
                j += 1;
            }
            commit_group();
            stage_idx = (stage_idx + 1) % STAGES;
            iter += 1;
        }
        wait_group(0);
        sync();

        // sync_state
        if BDZ > 1 {
            let fo = sbase + ((tz * BDY + ty) * hd + tx * VEC) * 4;
            sts_vec::<f32, VEC>(fo, &st_o);
            let md = off_smem + (tz * BDY + ty) * 8;
            sts_v2(md, st_m.to_bits(), st_d.to_bits());
            sync();
            let mut jz: u32 = 0;
            while jz < BDZ {
                cuda_device::thread::__unroll_config::<0>();
                let (mzb, dzb) = lds_v2(off_smem + (jz * BDY + ty) * 8);
                let (mz, dz) = (f32::from_bits(mzb), f32::from_bits(dzb));
                let oz = lds_vec::<f32, VEC>(sbase + ((jz * BDY + ty) * hd + tx * VEC) * 4);
                if jz == 0 {
                    let m = fmax(NEG_INF, mz);
                    let e1 = ex2(fsub(NEG_INF, m));
                    let e2 = ex2(fsub(mz, m));
                    st_d = ffma(dz, e2, e1);
                    let mut i: usize = 0;
                    while i < Cv::<VEC>::U {
                        cuda_device::thread::__unroll_config::<0>();
                        if true {
                            st_o[i] = ffma(e1, 0.0, fmul(oz[i], e2));
                        }
                        i += 1;
                    }
                    st_m = m;
                } else {
                    let m = fmax(st_m, mz);
                    let e1 = ex2(fsub(st_m, m));
                    let e2 = ex2(fsub(mz, m));
                    st_d = ffma(st_d, e1, fmul(dz, e2));
                    let mut i: usize = 0;
                    while i < Cv::<VEC>::U {
                        cuda_device::thread::__unroll_config::<0>();
                        if true {
                            st_o[i] = ffma(st_o[i], e1, fmul(oz[i], e2));
                        }
                        i += 1;
                    }
                    st_m = m;
                }
                jz += 1;
            }
        }
        // OutputTransform
        let mut i: usize = 0;
        while i < Cv::<VEC>::U {
            cuda_device::thread::__unroll_config::<0>();
            if true {
                let d_rcp = if !eq_ftz(st_m, NEG_INF) { rcp(st_d) } else { 0.0 };
                st_o[i] = fmul(st_o[i], d_rcp);
            }
            i += 1;
        }
        if tz == 0 {
            let row = bx.wrapping_mul(num_qo_heads).wrapping_add(qo_head_idx);
            let ooff = (row.wrapping_mul(hd) as u64).wrapping_add((tx * VEC) as u64);
            stg_vec::<T, VEC>(o.wrapping_add(ooff as usize), &st_o);
            if !lse.is_null() {
                *lse.add(row as usize) = fadd(st_m, lg2(st_d));
            }
        }
    }

    // ------------------------------------------------------------------------------------------
    // BatchDecodeWithPagedKVCacheKernelMLA<STAGES, 16, 2, 32, 8, 1, 2, DefaultAttention<0,0,0,0>,
    //   BatchDecodeParamsMLA<T, T, T, int>> (ckv 512, kpe 64; block (32, 8, 1))

    #[inline(always)]
    pub unsafe fn mla_decode<T: El, const STAGES: u32>(
        q_nope: *const T, q_pe: *const T, ckv_data: *const T, kpe_data: *const T, indices: *const i32, indptr: *const i32,
        last_page_len: *const i32, o: *mut T, lse: *mut f32, request_indices: *const i32, kv_tile_indices: *const i32,
        kv_chunk_size_ptr: *const i32, block_valid_mask: *const u8, num_qo_heads: u32, sm_scale_p: f32, partition_kv: u32, fd_d: u32,
        fd_m: u32, fd_s: u32, fd_a: u32, batch_size: u32, stride_page_ckv: u32, stride_page_kpe: u32, stride_n_ckv: u32, stride_n_kpe: u32,
    ) {
        const VC: u32 = 16;
        const VK: u32 = 2;
        const BDX: u32 = 32;
        const BDY: u32 = 8;
        const HDC: u32 = 512;
        const HDK: u32 = 64;
        const KV_ITER: u32 = 8;
        const TILE_QO: usize = 2;
        let smem = DynamicSharedArray::<u8, 16>::get_raw();
        let sbase = cuda_device::shared::cvta_generic_to_shared_u32(smem as *const u8);
        let sz = T::SZ;
        let sm_scale = fmul(sm_scale_p, LOG2E);
        let partition = partition_kv != 0;
        let batch_idx = thread::blockIdx_x();
        let tx = thread::threadIdx_x();
        let ty = thread::threadIdx_y();
        let tz = thread::threadIdx_z();
        let t_offset = (tz * BDY + ty) * BDX + tx;
        if !block_valid_mask.is_null() && *block_valid_mask.add(batch_idx as usize) == 0 {
            return;
        }
        let mapped = *request_indices.add(batch_idx as usize) as u32;
        let orig_seq_len = get_length(indptr, last_page_len, mapped, fd_d);
        let chunk_idx = *kv_tile_indices.add(batch_idx as usize) as u32;
        let kv_chunk_size = *kv_chunk_size_ptr as u32;
        let cur_chunk_start = if partition { chunk_idx.wrapping_mul(kv_chunk_size) } else { 0 };
        let cur_chunk_end = if partition { chunk_idx.wrapping_add(1).wrapping_mul(kv_chunk_size).min(orig_seq_len) } else { orig_seq_len };
        let cur_chunk_len = cur_chunk_end.wrapping_sub(cur_chunk_start);
        let packed_page_iter_base = (*indptr.add(mapped as usize) as u32).wrapping_mul(fd_d).wrapping_add(cur_chunk_start);
        let last_indptr = *indptr.add(batch_size as usize);

        let ckv_smem = sbase;
        let kpe_smem = sbase + STAGES * KV_ITER * HDC * sz;
        let ckv_off_smem = kpe_smem + STAGES * KV_ITER * HDK * sz;
        let kpe_off_smem = ckv_off_smem + BDX * BDY * 8;

        let mut q_nope_vec = [[0f32; 16]; TILE_QO];
        let mut q_pe_vec = [[0f32; 16]; TILE_QO];
        let mut qo_head_idx = [0u32; TILE_QO];
        let mut i: usize = 0;
        while i < TILE_QO {
            cuda_device::thread::__unroll_config::<0>();
            qo_head_idx[i] = (thread::blockIdx_y() * BDY + ty) * TILE_QO as u32 + i as u32;
            if qo_head_idx[i] < num_qo_heads {
                let row = mapped.wrapping_mul(num_qo_heads).wrapping_add(qo_head_idx[i]);
                q_nope_vec[i] = ldg_vec::<T, 16>(q_nope.wrapping_add((row.wrapping_mul(HDC) as u64).wrapping_add((tx * VC) as u64) as usize));
                q_pe_vec[i] = ldg_vec::<T, 2>(q_pe.wrapping_add((row.wrapping_mul(HDK) as u64).wrapping_add((tx * VK) as u64) as usize));
            }
            i += 1;
        }

        let fill = |n: u32| {
            let (pq, pr) = divmod(n, fd_d, fd_m, fd_s, fd_a);
            sts64(ckv_off_smem + t_offset * 8, prot_off(pq, last_indptr, indices, stride_page_ckv, (pr as u64).wrapping_mul(stride_n_ckv as u64)));
            sts64(kpe_off_smem + t_offset * 8, prot_off(pq, last_indptr, indices, stride_page_kpe, (pr as u64).wrapping_mul(stride_n_kpe as u64)));
        };
        fill(packed_page_iter_base.wrapping_add(t_offset));
        sync();

        let vbytes = VC * sz;
        let kpe_col = (tx / 8) * VC; // tx / tx_fold * vec_size_ckv
        let mut stage_idx: u32 = 0;
        let mut it: u32 = 0;
        while it < STAGES {
            cuda_device::thread::__unroll_config::<0>();
            let valid = it * KV_ITER + (tz * BDY + ty) < cur_chunk_len;
            let slot = (it + tz) * BDY + ty;
            let off = lds64(ckv_off_smem + slot * 8).wrapping_add((tx * VC) as u64);
            pred_load::<true>(
                ckv_smem + ((stage_idx * KV_ITER + tz * BDY + ty) * HDC + tx * VC) * sz,
                (ckv_data as *const u8).wrapping_add(off.wrapping_mul(sz as u64) as usize),
                valid,
                vbytes,
            );
            let off = lds64(kpe_off_smem + slot * 8).wrapping_add(kpe_col as u64);
            pred_load::<true>(
                kpe_smem + ((stage_idx * KV_ITER + tz * BDY + ty) * HDK + kpe_col) * sz,
                (kpe_data as *const u8).wrapping_add(off.wrapping_mul(sz as u64) as usize),
                valid,
                vbytes,
            );
            commit_group();
            stage_idx = (stage_idx + 1) % STAGES;
            it += 1;
        }

        let mut st_o = [[0f32; 16]; TILE_QO];
        let mut st_m = [NEG_INF; TILE_QO];
        let mut st_d = [1.0f32; TILE_QO];
        let n_iter = cur_chunk_len.wrapping_add(KV_ITER - 1) / KV_ITER;
        let mut iter: u32 = 0;
        while iter < n_iter {
            wait_group(STAGES - 1);
            sync();
            let cst = ckv_smem + (stage_idx * KV_ITER + tz * BDY) * HDC * sz;
            let kst = kpe_smem + (stage_idx * KV_ITER + tz * BDY) * HDK * sz;
            let iter_base = iter.wrapping_mul(KV_ITER);
            let mut h: usize = 0;
            while h < TILE_QO {
                cuda_device::thread::__unroll_config::<0>();
                let mut s = [0f32; 8];
                let m_prev = st_m[h];
                let mut m = m_prev;
                let mut j: usize = 0;
                while j < 8 {
                    cuda_device::thread::__unroll_config::<0>();
                    let c = lds_vec::<T, 16>(cst + (j as u32 * HDC + tx * VC) * sz);
                    let kp = lds_vec::<T, 2>(kst + (j as u32 * HDK + tx * VK) * sz);
                    let mut acc = ffma(q_nope_vec[h][0], c[0], 0.0);
                    let mut i: usize = 1;
                    while i < 16 {
                        cuda_device::thread::__unroll_config::<0>();
                        acc = ffma(q_nope_vec[h][i], c[i], acc);
                        i += 1;
                    }
                    acc = ffma(q_pe_vec[h][0], kp[0], acc);
                    acc = ffma(q_pe_vec[h][1], kp[1], acc);
                    acc = fmul(sm_scale, acc);
                    let mut k: u32 = 0;
                    while k < 5 {
                        cuda_device::thread::__unroll_config::<0>();
                        acc = fadd(acc, shfl_xor(acc, 16u32 >> k));
                        k += 1;
                    }
                    acc = if iter_base.wrapping_add(tz * 8 + j as u32) < cur_chunk_len { acc } else { NEG_INF };
                    s[j] = acc;
                    m = fmax(m, acc);
                    j += 1;
                }
                let o_scale = ex2(fsub(m_prev, m));
                let mut d = st_d[h];
                let mut j: usize = 0;
                while j < 8 {
                    cuda_device::thread::__unroll_config::<0>();
                    s[j] = ex2(fsub(s[j], m));
                    d = if j == 0 { ffma(o_scale, d, s[0]) } else { fadd(s[j], d) };
                    j += 1;
                }
                let mut j: usize = 0;
                while j < 8 {
                    cuda_device::thread::__unroll_config::<0>();
                    let v = lds_vec::<T, 16>(cst + (j as u32 * HDC + tx * VC) * sz);
                    let mut i: usize = 0;
                    while i < 16 {
                        cuda_device::thread::__unroll_config::<0>();
                        st_o[h][i] = if j == 0 { ffma(o_scale, st_o[h][i], fmul(s[0], v[i])) } else { ffma(s[j], v[i], st_o[h][i]) };
                        i += 1;
                    }
                    j += 1;
                }
                st_m[h] = m;
                st_d[h] = d;
                h += 1;
            }
            if (iter + STAGES) % BDX == 0 {
                fill(packed_page_iter_base.wrapping_add((iter + STAGES).wrapping_mul(KV_ITER)).wrapping_add(t_offset));
            }
            sync();
            let valid = (iter + STAGES).wrapping_mul(KV_ITER).wrapping_add(tz * BDY + ty) < cur_chunk_len;
            let slot = (((iter + STAGES) % BDX) + tz) * BDY + ty;
            let off = lds64(ckv_off_smem + slot * 8).wrapping_add((tx * VC) as u64);
            pred_load::<true>(
                ckv_smem + ((stage_idx * KV_ITER + tz * BDY + ty) * HDC + tx * VC) * sz,
                (ckv_data as *const u8).wrapping_add(off.wrapping_mul(sz as u64) as usize),
                valid,
                vbytes,
            );
            let off = lds64(kpe_off_smem + slot * 8).wrapping_add(kpe_col as u64);
            pred_load::<true>(
                kpe_smem + ((stage_idx * KV_ITER + tz * BDY + ty) * HDK + kpe_col) * sz,
                (kpe_data as *const u8).wrapping_add(off.wrapping_mul(sz as u64) as usize),
                valid,
                vbytes,
            );
            commit_group();
            stage_idx = (stage_idx + 1) % STAGES;
            iter += 1;
        }
        wait_group(0);
        sync();

        if tz == 0 {
            let mut h: usize = 0;
            while h < TILE_QO {
                cuda_device::thread::__unroll_config::<0>();
                if qo_head_idx[h] < num_qo_heads {
                    let mut i: usize = 0;
                    while i < 16 {
                        cuda_device::thread::__unroll_config::<0>();
                        let d_rcp = if !eq_ftz(st_m[h], NEG_INF) { rcp(st_d[h]) } else { 0.0 };
                        st_o[h][i] = fmul(st_o[h][i], d_rcp);
                        i += 1;
                    }
                    let row = batch_idx.wrapping_mul(num_qo_heads).wrapping_add(qo_head_idx[h]);
                    stg_vec::<T, 16>(o.wrapping_add((row.wrapping_mul(HDC) as u64).wrapping_add((tx * VC) as u64) as usize), &st_o[h]);
                    if !lse.is_null() {
                        *lse.add(row as usize) = fadd(st_m[h], lg2(st_d[h]));
                    }
                }
                h += 1;
            }
        }
    }

    // ------------------------------------------------------------------------------------------
    // PersistentVariableLengthMergeStatesKernel<VEC, BDX, BDY, 4, T, T, int>

    #[inline(always)]
    pub unsafe fn merge_states<T: El, const VEC: u32, const BDX: u32, const BDY: u32>(
        v: *const T, s: *const f32, indptr: *const i32, v_merged: *mut T, s_merged: *mut f32, max_seq_len: u32, seq_len_ptr: *const u32,
        num_heads: u32,
    ) {
        const STAGES: u32 = 4;
        let smem = DynamicSharedArray::<u8, 16>::get_raw();
        let sbase = cuda_device::shared::cvta_generic_to_shared_u32(smem as *const u8);
        let hd: u32 = VEC * BDX;
        let sz = T::SZ;
        let vbytes = VEC * sz;
        let tx = thread::threadIdx_x();
        let ty = thread::threadIdx_y();
        let num_ctas = thread::gridDim_x();
        let seq_len = if seq_len_ptr.is_null() { max_seq_len } else { *seq_len_ptr };
        let v_smem = sbase;
        let s_smem = sbase + STAGES * BDY * hd * sz;
        let mut i = thread::blockIdx_x();
        while i < seq_len.wrapping_mul(num_heads) {
            sync();
            let pos = udiv(i, num_heads);
            let head_idx = i.wrapping_sub(pos.wrapping_mul(num_heads));
            let ip = *indptr.add(pos as usize);
            let n = (*indptr.add(pos.wrapping_add(1) as usize)).wrapping_sub(ip) as u32;
            let out_row = pos.wrapping_mul(num_heads).wrapping_add(head_idx);
            let out_p = (v_merged as *mut u8).wrapping_add(((out_row.wrapping_mul(hd) as u64).wrapping_add((tx * VEC) as u64).wrapping_mul(sz as u64)) as usize);
            if n == 0 {
                stg_words(out_p, &[0u32; 16], vbytes);
                if !s_merged.is_null() {
                    *s_merged.add(out_row as usize) = NEG_INF;
                }
            } else if n == 1 {
                let in_row = (ip as u32).wrapping_mul(num_heads).wrapping_add(head_idx);
                let in_p = (v as *const u8).wrapping_add(((in_row.wrapping_mul(hd) as u64).wrapping_add((tx * VEC) as u64).wrapping_mul(sz as u64)) as usize);
                stg_words(out_p, &ldg_words(in_p, vbytes), vbytes);
                if !s_merged.is_null() {
                    *s_merged.add(out_row as usize) = *s.add(in_row as usize);
                }
            } else {
                let src = |r: u32| -> *const u8 {
                    let row = (ip as u32).wrapping_add(r).wrapping_mul(num_heads).wrapping_add(head_idx);
                    (v as *const u8).wrapping_add(((row.wrapping_mul(hd) as u64).wrapping_add((tx * VEC) as u64).wrapping_mul(sz as u64)) as usize)
                };
                let mut it: u32 = 0;
                while it < STAGES {
                    cuda_device::thread::__unroll_config::<0>();
                    pred_load::<false>(v_smem + ((it * BDY + ty) * hd + tx * VEC) * sz, src(it * BDY + ty), it * BDY + ty < n, vbytes);
                    commit_group();
                    it += 1;
                }
                let mut st_o = [0f32; 16];
                let mut st_m: f32 = NEG_INF;
                let mut st_d: f32 = 1.0;
                let n_iter = n.wrapping_add(BDY - 1) / BDY;
                let mut iter: u32 = 0;
                while iter < n_iter {
                    if iter % BDX == 0 {
                        let idx = iter.wrapping_mul(BDY).wrapping_add(ty * BDX + tx);
                        let sv = if idx < n {
                            *s.add((ip as u32).wrapping_add(idx).wrapping_mul(num_heads).wrapping_add(head_idx) as usize)
                        } else {
                            0.0
                        };
                        sts32(s_smem + (ty * BDX + tx) * 4, sv.to_bits());
                        sync();
                    }
                    wait_group(STAGES - 1);
                    sync();
                    let vv = lds_vec::<T, VEC>(v_smem + (((iter % STAGES) * BDY + ty) * hd + tx * VEC) * sz);
                    if iter.wrapping_mul(BDY).wrapping_add(ty) < n {
                        let sv = f32::from_bits(lds32(s_smem + ((iter % BDX) * BDY + ty) * 4));
                        let m = fmax(st_m, sv);
                        let e1 = ex2(fsub(st_m, m));
                        let e2 = ex2(fsub(sv, m));
                        st_d = ffma(st_d, e1, e2);
                        let mut k: usize = 0;
                        while k < Cv::<VEC>::U {
                            cuda_device::thread::__unroll_config::<0>();
                            if true {
                                st_o[k] = ffma(st_o[k], e1, fmul(vv[k], e2));
                            }
                            k += 1;
                        }
                        st_m = m;
                    }
                    sync();
                    let r = iter.wrapping_add(STAGES).wrapping_mul(BDY).wrapping_add(ty);
                    pred_load::<false>(v_smem + (((iter % STAGES) * BDY + ty) * hd + tx * VEC) * sz, src(r), r < n, vbytes);
                    commit_group();
                    iter += 1;
                }
                wait_group(0);
                sync();
                let mut k: usize = 0;
                while k < Cv::<VEC>::U {
                    cuda_device::thread::__unroll_config::<0>();
                    if true {
                        st_o[k] = fdiv(st_o[k], st_d);
                    }
                    k += 1;
                }
                // threadblock_sync_state
                sts_vec::<T, VEC>(v_smem + (ty * hd + tx * VEC) * sz, &st_o);
                sts32(s_smem + ty * 4, fadd(st_m, lg2(st_d)).to_bits());
                sync();
                let mut jy: u32 = 0;
                while jy < BDY {
                    cuda_device::thread::__unroll_config::<0>();
                    let sv = f32::from_bits(lds32(s_smem + jy * 4));
                    let vv = lds_vec::<T, VEC>(v_smem + (jy * hd + tx * VEC) * sz);
                    if jy == 0 {
                        let m = fmax(NEG_INF, sv);
                        let e1 = ex2(fsub(NEG_INF, m));
                        let e2 = ex2(fsub(sv, m));
                        st_d = fadd(e1, e2);
                        let mut k: usize = 0;
                        while k < Cv::<VEC>::U {
                            cuda_device::thread::__unroll_config::<0>();
                            if true {
                                st_o[k] = ffma(e1, 0.0, fmul(vv[k], e2));
                            }
                            k += 1;
                        }
                        st_m = m;
                    } else {
                        let m = fmax(st_m, sv);
                        let e1 = ex2(fsub(st_m, m));
                        let e2 = ex2(fsub(sv, m));
                        st_d = ffma(st_d, e1, e2);
                        let mut k: usize = 0;
                        while k < Cv::<VEC>::U {
                            cuda_device::thread::__unroll_config::<0>();
                            if true {
                                st_o[k] = ffma(st_o[k], e1, fmul(vv[k], e2));
                            }
                            k += 1;
                        }
                        st_m = m;
                    }
                    jy += 1;
                }
                let mut k: usize = 0;
                while k < Cv::<VEC>::U {
                    cuda_device::thread::__unroll_config::<0>();
                    if true {
                        st_o[k] = fdiv(st_o[k], st_d);
                    }
                    k += 1;
                }
                stg_vec::<T, VEC>(out_p as *mut T, &st_o);
                if !s_merged.is_null() {
                    *s_merged.add(out_row as usize) = fadd(st_m, lg2(st_d));
                }
            }
            i = i.wrapping_add(num_ctas);
        }
    }

    // ------------------------------------------------------------------------------------------
    // reshape_and_cache_flashinfer_kernel<T> / gather_kv_cache_flashinfer_kernel<T>

    #[inline(always)]
    pub unsafe fn reshape_fi<T: Copy>(
        key: *const T, value: *const T, key_cache: *mut T, value_cache: *mut T, slot_mapping: *const i64, num_heads: i32, head_size: i32,
        block_size: i32, key_stride: i32, value_stride: i32,
    ) {
        let token_idx = thread::blockIdx_x() as i32;
        let slot = *slot_mapping.offset(token_idx as isize);
        if slot < 0 {
            return;
        }
        let (block_idx, block_offset) = divrem64(slot, block_size as i64);
        let n = num_heads.wrapping_mul(head_size);
        let mut i = thread::threadIdx_x() as i32;
        while i < n {
            let head_idx = sdiv(i, head_size);
            let dim_idx = i.wrapping_sub(head_idx.wrapping_mul(head_size));
            let dst = block_idx
                .wrapping_mul(num_heads as i64)
                .wrapping_add(head_idx as i64)
                .wrapping_mul(block_size as i64)
                .wrapping_add(block_offset)
                .wrapping_mul(head_size as i64)
                .wrapping_add(dim_idx as i64);
            *key_cache.offset(dst as isize) = *key.offset(token_idx.wrapping_mul(key_stride).wrapping_add(i) as isize);
            *value_cache.offset(dst as isize) = *value.offset(token_idx.wrapping_mul(value_stride).wrapping_add(i) as isize);
            i = i.wrapping_add(thread::blockDim_x() as i32);
        }
    }

    #[inline(always)]
    pub unsafe fn gather_fi<T: Copy>(
        key_cache: *const T, value_cache: *const T, k_out: *mut T, v_out: *mut T, block_table: *const i32, cu_seq_lens: *const i32,
        num_tokens: i32, block_size: i32, block_table_stride: i32, num_kv_heads: i32, head_size: i32,
    ) {
        let token_id = thread::blockIdx_x() as i32;
        if token_id >= num_tokens {
            return;
        }
        let mut seq_id: i32 = 0;
        while *cu_seq_lens.offset(seq_id.wrapping_add(1) as isize) <= token_id {
            seq_id = seq_id.wrapping_add(1);
        }
        let seq_start = *cu_seq_lens.offset(seq_id as isize);
        let seq_offset = token_id.wrapping_sub(seq_start);
        let table_idx = sdiv(seq_offset, block_size);
        let slot = seq_offset.wrapping_sub(table_idx.wrapping_mul(block_size));
        let block_idx = *block_table.offset(seq_id.wrapping_mul(block_table_stride).wrapping_add(table_idx) as isize);
        let n = num_kv_heads.wrapping_mul(head_size);
        let mut i = thread::threadIdx_x() as i32;
        while i < n {
            let head_idx = sdiv(i, head_size);
            let dim_idx = i.wrapping_sub(head_idx.wrapping_mul(head_size));
            let cache_idx = (block_idx as i64)
                .wrapping_mul(num_kv_heads as i64)
                .wrapping_add(head_idx as i64)
                .wrapping_mul(block_size as i64)
                .wrapping_add(slot as i64)
                .wrapping_mul(head_size as i64)
                .wrapping_add(dim_idx as i64);
            let out_idx = (token_id as i64).wrapping_mul(n as i64).wrapping_add(i as i64);
            *k_out.offset(out_idx as isize) = *key_cache.offset(cache_idx as isize);
            *v_out.offset(out_idx as isize) = *value_cache.offset(cache_idx as isize);
            i = i.wrapping_add(thread::blockDim_x() as i32);
        }
    }

    // GENERATED KERNELS BEGIN
    #[kernel]
    pub unsafe fn _ZN20mistralrs_flashinfer33gather_kv_cache_flashinfer_kernelIfEEvPKT_S3_PS1_S4_PKiS6_iiiii(kc: *const f32, vc: *const f32, ko: *mut f32, vo: *mut f32, bt: *const i32, cu: *const i32, nt: i32, bs: i32, bts: i32, nkv: i32, hs: i32) {
        gather_fi::<f32>(kc, vc, ko, vo, bt, cu, nt, bs, bts, nkv, hs)
    }
    #[kernel]
    pub unsafe fn _ZN20mistralrs_flashinfer33gather_kv_cache_flashinfer_kernelI13__nv_bfloat16EEvPKT_S4_PS2_S5_PKiS7_iiiii(kc: *const u16, vc: *const u16, ko: *mut u16, vo: *mut u16, bt: *const i32, cu: *const i32, nt: i32, bs: i32, bts: i32, nkv: i32, hs: i32) {
        gather_fi::<u16>(kc, vc, ko, vo, bt, cu, nt, bs, bts, nkv, hs)
    }
    #[kernel]
    pub unsafe fn _ZN20mistralrs_flashinfer33gather_kv_cache_flashinfer_kernelI6__halfEEvPKT_S4_PS2_S5_PKiS7_iiiii(kc: *const u16, vc: *const u16, ko: *mut u16, vo: *mut u16, bt: *const i32, cu: *const i32, nt: i32, bs: i32, bts: i32, nkv: i32, hs: i32) {
        gather_fi::<u16>(kc, vc, ko, vo, bt, cu, nt, bs, bts, nkv, hs)
    }
    #[kernel]
    pub unsafe fn _ZN20mistralrs_flashinfer35reshape_and_cache_flashinfer_kernelIfEEvPKT_S3_PS1_S4_PKliiiii(k: *const f32, v: *const f32, kc: *mut f32, vc: *mut f32, slot: *const i64, nh: i32, hs: i32, bs: i32, ks: i32, vs: i32) {
        reshape_fi::<f32>(k, v, kc, vc, slot, nh, hs, bs, ks, vs)
    }
    #[kernel]
    pub unsafe fn _ZN20mistralrs_flashinfer35reshape_and_cache_flashinfer_kernelI13__nv_bfloat16EEvPKT_S4_PS2_S5_PKliiiii(k: *const u16, v: *const u16, kc: *mut u16, vc: *mut u16, slot: *const i64, nh: i32, hs: i32, bs: i32, ks: i32, vs: i32) {
        reshape_fi::<u16>(k, v, kc, vc, slot, nh, hs, bs, ks, vs)
    }
    #[kernel]
    pub unsafe fn _ZN20mistralrs_flashinfer35reshape_and_cache_flashinfer_kernelI6__halfEEvPKT_S4_PS2_S5_PKliiiii(k: *const u16, v: *const u16, kc: *mut u16, vc: *mut u16, slot: *const i64, nh: i32, hs: i32, bs: i32, ks: i32, vs: i32) {
        reshape_fi::<u16>(k, v, kc, vc, slot, nh, hs, bs, ks, vs)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 16, 32, 16, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 16, 32, 8, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 16, 32, 4, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 16, 32, 3, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 16, 32, 2, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 4, 16, 32, 1, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 16, 32, 16, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 16, 32, 8, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 16, 32, 4, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 16, 32, 3, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 16, 32, 2, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 4, 16, 32, 1, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 16, 32, 16, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 16, 32, 8, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 16, 32, 4, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 16, 32, 3, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 16, 32, 2, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 4, 16, 32, 1, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 16, 32, 16, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 16, 32, 8, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 16, 32, 4, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 16, 32, 3, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 16, 32, 2, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 4, 16, 32, 1, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 8, 32, 16, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 8, 32, 8, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 8, 32, 4, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 8, 32, 3, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 8, 32, 2, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 4, 8, 32, 1, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 8, 32, 16, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 8, 32, 8, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 8, 32, 4, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 8, 32, 3, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 8, 32, 2, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 4, 8, 32, 1, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 8, 32, 16, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 8, 32, 8, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 8, 32, 4, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 8, 32, 3, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 8, 32, 2, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 4, 8, 32, 1, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 8, 32, 16, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 8, 32, 8, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 8, 32, 4, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 8, 32, 3, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 8, 32, 2, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 4, 8, 32, 1, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 32, 16, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 32, 8, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 32, 4, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 32, 3, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 32, 2, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj4ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 4, 4, 32, 1, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 32, 16, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 32, 8, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 32, 4, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 32, 3, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 32, 2, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj4ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 4, 4, 32, 1, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 32, 16, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 32, 8, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 32, 4, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 32, 3, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 32, 2, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj4ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 4, 4, 32, 1, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 32, 16, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 32, 8, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 32, 4, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 32, 3, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 32, 2, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj4ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 4, 4, 32, 1, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 16, 16, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 16, 8, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 16, 4, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 16, 3, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 16, 2, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj4ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 4, 4, 16, 1, 8, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 16, 16, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 16, 8, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 16, 4, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 16, 3, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 16, 2, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj4ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 4, 4, 16, 1, 8, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 16, 16, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 16, 8, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 16, 4, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 16, 3, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 16, 2, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj4ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 4, 4, 16, 1, 8, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 16, 16, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 16, 8, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 16, 4, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 16, 3, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 1, 4, 16, 2, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer41PersistentVariableLengthMergeStatesKernelILj16ELj32ELj4ELj4EffiEEvPT3_PfPT5_PT4_S3_jPjj(v: *const f32, s: *const f32, indptr: *const i32, vm: *mut f32, smg: *mut f32, max_seq_len: u32, seq_len: *const u32, num_heads: u32) {
        merge_states::<f32, 16, 32, 4>(v, s, indptr, vm, smg, max_seq_len, seq_len, num_heads)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer41PersistentVariableLengthMergeStatesKernelILj8ELj32ELj4ELj4EffiEEvPT3_PfPT5_PT4_S3_jPjj(v: *const f32, s: *const f32, indptr: *const i32, vm: *mut f32, smg: *mut f32, max_seq_len: u32, seq_len: *const u32, num_heads: u32) {
        merge_states::<f32, 8, 32, 4>(v, s, indptr, vm, smg, max_seq_len, seq_len, num_heads)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer41PersistentVariableLengthMergeStatesKernelILj4ELj32ELj4ELj4EffiEEvPT3_PfPT5_PT4_S3_jPjj(v: *const f32, s: *const f32, indptr: *const i32, vm: *mut f32, smg: *mut f32, max_seq_len: u32, seq_len: *const u32, num_heads: u32) {
        merge_states::<f32, 4, 32, 4>(v, s, indptr, vm, smg, max_seq_len, seq_len, num_heads)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer41PersistentVariableLengthMergeStatesKernelILj4ELj16ELj8ELj4EffiEEvPT3_PfPT5_PT4_S3_jPjj(v: *const f32, s: *const f32, indptr: *const i32, vm: *mut f32, smg: *mut f32, max_seq_len: u32, seq_len: *const u32, num_heads: u32) {
        merge_states::<f32, 4, 16, 8>(v, s, indptr, vm, smg, max_seq_len, seq_len, num_heads)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj4ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, 2, 4, 4, 16, 1, 8, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 16, 32, 16, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 16, 32, 8, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 16, 32, 4, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 16, 32, 3, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 16, 32, 2, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 4, 16, 32, 1, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 16, 32, 16, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 16, 32, 8, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 16, 32, 4, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 16, 32, 3, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 16, 32, 2, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 4, 16, 32, 1, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 16, 32, 16, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 16, 32, 8, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 16, 32, 4, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 16, 32, 3, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 16, 32, 2, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 4, 16, 32, 1, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 16, 32, 16, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 16, 32, 8, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 16, 32, 4, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 16, 32, 3, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 16, 32, 2, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 4, 16, 32, 1, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 32, 16, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 32, 8, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 32, 4, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 32, 3, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 32, 2, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 4, 8, 32, 1, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 32, 16, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 32, 8, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 32, 4, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 32, 3, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 32, 2, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 4, 8, 32, 1, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 32, 16, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 32, 8, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 32, 4, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 32, 3, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 32, 2, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 4, 8, 32, 1, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 32, 16, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 32, 8, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 32, 4, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 32, 3, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 32, 2, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 4, 8, 32, 1, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 16, 16, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 16, 8, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 16, 4, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 16, 3, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 16, 2, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 4, 8, 16, 1, 8, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 16, 16, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 16, 8, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 16, 4, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 16, 3, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 16, 2, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 4, 8, 16, 1, 8, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 16, 16, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 16, 8, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 16, 4, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 16, 3, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 16, 2, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 4, 8, 16, 1, 8, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 16, 16, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 16, 8, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 16, 4, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 16, 3, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 16, 2, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 4, 8, 16, 1, 8, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 8, 16, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj8ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 8, 8, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj4ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 8, 4, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj3ELj5ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 8, 3, 5, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj2ELj8ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 8, 2, 8, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj8ELj1ELj16ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 4, 8, 8, 1, 16, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 8, 16, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj8ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 8, 8, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj4ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 8, 4, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj3ELj5ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 8, 3, 5, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj2ELj8ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 8, 2, 8, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj8ELj1ELj16ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 4, 8, 8, 1, 16, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 8, 16, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj8ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 8, 8, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj4ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 8, 4, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj3ELj5ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 8, 3, 5, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj2ELj8ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 8, 2, 8, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj8ELj1ELj16ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 4, 8, 8, 1, 16, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 8, 16, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj8ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 8, 8, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj4ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 8, 4, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj3ELj5ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 8, 3, 5, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj2ELj8ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 1, 8, 8, 2, 8, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer41PersistentVariableLengthMergeStatesKernelILj16ELj32ELj4ELj4E13__nv_bfloat16S1_iEEvPT3_PfPT5_PT4_S4_jPjj(v: *const B, s: *const f32, indptr: *const i32, vm: *mut B, smg: *mut f32, max_seq_len: u32, seq_len: *const u32, num_heads: u32) {
        merge_states::<B, 16, 32, 4>(v, s, indptr, vm, smg, max_seq_len, seq_len, num_heads)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer41PersistentVariableLengthMergeStatesKernelILj8ELj32ELj4ELj4E13__nv_bfloat16S1_iEEvPT3_PfPT5_PT4_S4_jPjj(v: *const B, s: *const f32, indptr: *const i32, vm: *mut B, smg: *mut f32, max_seq_len: u32, seq_len: *const u32, num_heads: u32) {
        merge_states::<B, 8, 32, 4>(v, s, indptr, vm, smg, max_seq_len, seq_len, num_heads)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer41PersistentVariableLengthMergeStatesKernelILj8ELj16ELj8ELj4E13__nv_bfloat16S1_iEEvPT3_PfPT5_PT4_S4_jPjj(v: *const B, s: *const f32, indptr: *const i32, vm: *mut B, smg: *mut f32, max_seq_len: u32, seq_len: *const u32, num_heads: u32) {
        merge_states::<B, 8, 16, 8>(v, s, indptr, vm, smg, max_seq_len, seq_len, num_heads)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer41PersistentVariableLengthMergeStatesKernelILj8ELj8ELj16ELj4E13__nv_bfloat16S1_iEEvPT3_PfPT5_PT4_S4_jPjj(v: *const B, s: *const f32, indptr: *const i32, vm: *mut B, smg: *mut f32, max_seq_len: u32, seq_len: *const u32, num_heads: u32) {
        merge_states::<B, 8, 8, 16>(v, s, indptr, vm, smg, max_seq_len, seq_len, num_heads)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj8ELj1ELj16ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, 2, 4, 8, 8, 1, 16, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 16, 32, 16, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 16, 32, 8, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 16, 32, 4, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 16, 32, 3, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 16, 32, 2, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 4, 16, 32, 1, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 16, 32, 16, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 16, 32, 8, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 16, 32, 4, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 16, 32, 3, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 16, 32, 2, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 4, 16, 32, 1, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 16, 32, 16, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 16, 32, 8, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 16, 32, 4, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 16, 32, 3, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 16, 32, 2, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 4, 16, 32, 1, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 16, 32, 16, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 16, 32, 8, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 16, 32, 4, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 16, 32, 3, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 16, 32, 2, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 4, 16, 32, 1, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 32, 16, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 32, 8, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 32, 4, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 32, 3, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 32, 2, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 4, 8, 32, 1, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 32, 16, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 32, 8, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 32, 4, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 32, 3, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 32, 2, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 4, 8, 32, 1, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 32, 16, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 32, 8, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 32, 4, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 32, 3, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 32, 2, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 4, 8, 32, 1, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 32, 16, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 32, 8, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 32, 4, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 32, 3, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 32, 2, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 4, 8, 32, 1, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 16, 16, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 16, 8, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 16, 4, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 16, 3, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 16, 2, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 4, 8, 16, 1, 8, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 16, 16, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 16, 8, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 16, 4, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 16, 3, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 16, 2, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 4, 8, 16, 1, 8, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 16, 16, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 16, 8, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 16, 4, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 16, 3, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 16, 2, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 4, 8, 16, 1, 8, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 16, 16, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 16, 8, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 16, 4, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 16, 3, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 16, 2, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 4, 8, 16, 1, 8, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 8, 16, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj8ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 8, 8, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj4ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 8, 4, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj3ELj5ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 8, 3, 5, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj2ELj8ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 8, 2, 8, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj8ELj1ELj16ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 4, 8, 8, 1, 16, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 8, 16, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj8ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 8, 8, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj4ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 8, 4, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj3ELj5ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 8, 3, 5, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj2ELj8ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 8, 2, 8, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj8ELj1ELj16ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 4, 8, 8, 1, 16, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 8, 16, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj8ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 8, 8, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj4ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 8, 4, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj3ELj5ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 8, 3, 5, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj2ELj8ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 8, 2, 8, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj8ELj1ELj16ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 4, 8, 8, 1, 16, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 8, 16, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj8ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 8, 8, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj4ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 8, 4, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj3ELj5ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 8, 3, 5, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj2ELj8ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 1, 8, 8, 2, 8, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer41PersistentVariableLengthMergeStatesKernelILj16ELj32ELj4ELj4E6__halfS1_iEEvPT3_PfPT5_PT4_S4_jPjj(v: *const H, s: *const f32, indptr: *const i32, vm: *mut H, smg: *mut f32, max_seq_len: u32, seq_len: *const u32, num_heads: u32) {
        merge_states::<H, 16, 32, 4>(v, s, indptr, vm, smg, max_seq_len, seq_len, num_heads)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer41PersistentVariableLengthMergeStatesKernelILj8ELj32ELj4ELj4E6__halfS1_iEEvPT3_PfPT5_PT4_S4_jPjj(v: *const H, s: *const f32, indptr: *const i32, vm: *mut H, smg: *mut f32, max_seq_len: u32, seq_len: *const u32, num_heads: u32) {
        merge_states::<H, 8, 32, 4>(v, s, indptr, vm, smg, max_seq_len, seq_len, num_heads)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer41PersistentVariableLengthMergeStatesKernelILj8ELj16ELj8ELj4E6__halfS1_iEEvPT3_PfPT5_PT4_S4_jPjj(v: *const H, s: *const f32, indptr: *const i32, vm: *mut H, smg: *mut f32, max_seq_len: u32, seq_len: *const u32, num_heads: u32) {
        merge_states::<H, 8, 16, 8>(v, s, indptr, vm, smg, max_seq_len, seq_len, num_heads)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer41PersistentVariableLengthMergeStatesKernelILj8ELj8ELj16ELj4E6__halfS1_iEEvPT3_PfPT5_PT4_S4_jPjj(v: *const H, s: *const f32, indptr: *const i32, vm: *mut H, smg: *mut f32, max_seq_len: u32, seq_len: *const u32, num_heads: u32) {
        merge_states::<H, 8, 8, 16>(v, s, indptr, vm, smg, max_seq_len, seq_len, num_heads)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj8ELj1ELj16ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, 2, 4, 8, 8, 1, 16, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer36BatchDecodeWithPagedKVCacheKernelMLAILj2ELj16ELj2ELj32ELj8ELj1ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_20BatchDecodeParamsMLAIfffiEEEEvT7_(qn: *const f32, qp: *const f32, ckv: *const f32, kpe: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, spc: u32, spk: u32, snc: u32, snk: u32) {
        mla_decode::<f32, 2>(qn, qp, ckv, kpe, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, sm, part, fd, fm, fs, fa, bs, spc, spk, snc, snk)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer36BatchDecodeWithPagedKVCacheKernelMLAILj2ELj16ELj2ELj32ELj8ELj1ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_20BatchDecodeParamsMLAI13__nv_bfloat16S4_S4_iEEEEvT7_(qn: *const B, qp: *const B, ckv: *const B, kpe: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, spc: u32, spk: u32, snc: u32, snk: u32) {
        mla_decode::<B, 2>(qn, qp, ckv, kpe, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, sm, part, fd, fm, fs, fa, bs, spc, spk, snc, snk)
    }
    #[kernel]
    pub unsafe fn _ZN10flashinfer36BatchDecodeWithPagedKVCacheKernelMLAILj2ELj16ELj2ELj32ELj8ELj1ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_20BatchDecodeParamsMLAI6__halfS4_S4_iEEEEvT7_(qn: *const H, qp: *const H, ckv: *const H, kpe: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, sm: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, spc: u32, spk: u32, snc: u32, snk: u32) {
        mla_decode::<H, 2>(qn, qp, ckv, kpe, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, sm, part, fd, fm, fs, fa, bs, spc, spk, snc, snk)
    }
    // GENERATED KERNELS END
}

fn main() {
    std::process::exit(if gate::run() { 0 } else { 1 });
}
