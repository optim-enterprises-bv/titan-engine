//! mistralrs-paged-attn group 2 in cuda-oxide (mistral.rs v0.9.4 sources, reference/mistralrs-paged-attn-094): the
//! FlashInfer decode kernels of `flashinfer_decode.cu` (BatchDecodeWithPagedKVCacheKernel for head dims 64/128/256/512,
//! f16/bf16/f32 queries over a same-dtype or an FP8 E4M3 KV cache (`batch_decode::<T, KV>`, KV = `E`: 16 fp8 per
//! thread, `cvt.rn.f16x2.e4m3x2` per element), GQA group sizes 1-8/16, sliding window x logits soft-cap, the v0.9.4
//! `v_scale` output factor),
//! plus the split-KV PersistentVariableLengthMergeStatesKernel and the reshape/gather cache helpers (all six
//! activation x cache pairs, FP8 E4M3 included) and of
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

    /// fp8 e4m3 element (bit pattern): the KV cache element of the v0.9.4 FP8 decode instances.
    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct E(pub u8);

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
    impl El for E {
        const SZ: u32 = 1;
        /// vec_t<float, V>::cast_load from __nv_fp8_e4m3: byte k of the word (little-endian),
        /// `cvt.rn.f16x2.e4m3x2` of the zero-extended byte, low half, `cvt.f32.f16`.
        #[inline(always)]
        fn w2f(w: u32, k: u32) -> f32 {
            e4m32f(((w >> (8 * k)) & 0xff) as u8)
        }
        /// never stored (the FP8 instances write DTypeO)
        #[inline(always)]
        fn f2w(_lo: f32, _hi: f32) -> u32 {
            0
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
        KV: El,
        const STAGES: u32,
        const TILE: u32,
        const VEC: u32,
        const BDX: u32,
        const BDY: u32,
        const BDZ: u32,
        const SW: bool,
        const SC: bool,
    >(
        q: *const T, k_data: *const KV, v_data: *const KV, indices: *const i32, indptr: *const i32, last_page_len: *const i32, o: *mut T,
        lse: *mut f32, request_indices: *const i32, kv_tile_indices: *const i32, kv_chunk_size_ptr: *const i32, block_valid_mask: *const u8,
        num_qo_heads: u32, q_stride_n: i32, q_stride_h: i32, window_left_p: i32, logits_soft_cap: f32, sm_scale: f32, v_scale: f32,
        partition_kv: u32,
        fd_d: u32, fd_m: u32, fd_s: u32, fd_a: u32, batch_size: u32, stride_page: u32, stride_n: u32, stride_h: u32,
    ) {
        let smem = DynamicSharedArray::<u8, 16>::get_raw();
        let sbase = cuda_device::shared::cvta_generic_to_shared_u32(smem as *const u8);
        let hd: u32 = BDX * VEC;
        // K/V element size (DTypeKV): T, or 1 for the FP8 E4M3 cache
        let sz = KV::SZ;
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
                    let k_vec = lds_vec::<KV, VEC>(kst + (j as u32 * BDX + tx) * vbytes);
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
                    let v_vec = lds_vec::<KV, VEC>(vst + (j as u32 * BDX + tx) * vbytes);
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
                // v0.9.4: OutputTransform's `output * d_rcp * v_scale` (BatchDecodeParams::v_scale)
                st_o[i] = fmul(fmul(st_o[i], d_rcp), v_scale);
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
    // v0.9.4 reshape_and_cache_flashinfer_kernel<DType, CacheType> / gather_kv_cache_flashinfer_kernel<CacheType, OutType>
    // (FP8 E4M3 KV cache with per-tensor k/v scales). Same-type pairs copy element bits; the FP8 pairs convert:
    // write `cvt.rn.satfinite.e4m3x2.f32 (0, (float)x / scale)` (div.approx.ftz: --use_fast_math), read
    // `cvt.rn.f16x2.e4m3x2` -> `cvt.f32.f16` -> `scale * x` (mul.ftz) -> `cvt.rn.{f16,bf16}.f32`.

    /// Scalar activation element: exact widening to f32 and round-to-nearest narrowing (the `static_cast`s).
    pub trait Act: Copy {
        fn to_f32(self) -> f32;
        fn from_f32(x: f32) -> Self;
        /// `static_cast<T>(0)` (all-zero bits for f16 / bf16 / f32)
        fn zero() -> Self;
    }
    impl Act for f32 {
        #[inline(always)]
        fn to_f32(self) -> f32 {
            self
        }
        #[inline(always)]
        fn from_f32(x: f32) -> Self {
            x
        }
        #[inline(always)]
        fn zero() -> Self {
            0.0
        }
    }
    impl Act for H {
        #[inline(always)]
        fn to_f32(self) -> f32 {
            h2f(self.0)
        }
        #[inline(always)]
        fn from_f32(x: f32) -> Self {
            let r: u16;
            unsafe { ptx_asm!("cvt.rn.f16.f32 %0, %1; //\t\x0b\x0c\r\n", out("=h") r, in("f") x, options(register_only)) };
            H(r)
        }
        #[inline(always)]
        fn zero() -> Self {
            H(0)
        }
    }
    impl Act for B {
        #[inline(always)]
        fn to_f32(self) -> f32 {
            bf2f(self.0)
        }
        #[inline(always)]
        fn from_f32(x: f32) -> Self {
            let r: u16;
            unsafe { ptx_asm!("cvt.rn.bf16.f32 %0, %1; //\t\x0b\x0c\r\n", out("=h") r, in("f") x, options(register_only)) };
            B(r)
        }
        #[inline(always)]
        fn zero() -> Self {
            B(0)
        }
    }
    /// f32 -> fp8 e4m3 byte: `__nv_fp8_e4m3(float)` = `cvt.rn.satfinite.e4m3x2.f32 d, 0, x` (low byte).
    #[inline(always)]
    pub fn f2e4m3(x: f32) -> u8 {
        let r: u16;
        unsafe { ptx_asm!("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2; //\t\x0b\x0c\r\n", out("=h") r, in("f") 0.0f32, in("f") x, options(register_only)) };
        r as u8
    }
    /// fp8 e4m3 byte -> f32: `cvt.rn.f16x2.e4m3x2` of the zero-extended byte, low half, `cvt.f32.f16`.
    #[inline(always)]
    pub fn e4m32f(b: u8) -> f32 {
        let r: u16;
        unsafe { ptx_asm!("{ .reg .b32 t; cvt.rn.f16x2.e4m3x2 t, %1; cvt.u16.u32 %0, t; }", out("=h") r, in("h") b as u16, options(register_only)) };
        h2f(r)
    }

    /// `C` is the cache element (`T` itself, or `u8` = __nv_fp8_e4m3 when `FP8`).
    #[inline(always)]
    pub unsafe fn reshape_fi<T: Act, C: Copy, const FP8: bool>(
        key: *const T, value: *const T, key_cache: *mut C, value_cache: *mut C, slot_mapping: *const i64, num_heads: i32, head_size: i32,
        block_size: i32, key_stride: i32, value_stride: i32, k_scale: f32, v_scale: f32,
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
            let k = *key.offset(token_idx.wrapping_mul(key_stride).wrapping_add(i) as isize);
            if FP8 {
                *(key_cache as *mut u8).offset(dst as isize) = f2e4m3(fdiv(k.to_f32(), k_scale));
            } else {
                *(key_cache as *mut T).offset(dst as isize) = k;
            }
            let v = *value.offset(token_idx.wrapping_mul(value_stride).wrapping_add(i) as isize);
            if FP8 {
                *(value_cache as *mut u8).offset(dst as isize) = f2e4m3(fdiv(v.to_f32(), v_scale));
            } else {
                *(value_cache as *mut T).offset(dst as isize) = v;
            }
            i = i.wrapping_add(thread::blockDim_x() as i32);
        }
    }

    /// `C` is the cache element (`O` itself, or `u8` = __nv_fp8_e4m3 when `FP8`).
    #[inline(always)]
    pub unsafe fn gather_fi<C: Copy, O: Act, const FP8: bool>(
        key_cache: *const C, value_cache: *const C, k_out: *mut O, v_out: *mut O, block_table: *const i32, cu_seq_lens: *const i32,
        num_tokens: i32, num_seqs: i32, block_size: i32, block_table_stride: i32, num_kv_heads: i32, head_size: i32, k_scale: f32,
        v_scale: f32,
    ) {
        let token_id = thread::blockIdx_x() as i32;
        if token_id >= num_tokens {
            return;
        }
        let n = num_kv_heads.wrapping_mul(head_size);
        if token_id >= *cu_seq_lens.offset(num_seqs as isize) {
            // tokens past the last sequence (padding) are zero-filled
            let mut i = thread::threadIdx_x() as i32;
            while i < n {
                let out_idx = (token_id as i64).wrapping_mul(n as i64).wrapping_add(i as i64);
                *k_out.offset(out_idx as isize) = O::zero();
                *v_out.offset(out_idx as isize) = O::zero();
                i = i.wrapping_add(thread::blockDim_x() as i32);
            }
            return;
        }
        let mut seq_id: i32 = 0;
        while seq_id.wrapping_add(1) < num_seqs && *cu_seq_lens.offset(seq_id.wrapping_add(1) as isize) <= token_id {
            seq_id = seq_id.wrapping_add(1);
        }
        let seq_start = *cu_seq_lens.offset(seq_id as isize);
        let seq_offset = token_id.wrapping_sub(seq_start);
        let table_idx = sdiv(seq_offset, block_size);
        let slot = seq_offset.wrapping_sub(table_idx.wrapping_mul(block_size));
        let block_idx = *block_table.offset(seq_id.wrapping_mul(block_table_stride).wrapping_add(table_idx) as isize);
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
            if FP8 {
                let kb = *(key_cache as *const u8).offset(cache_idx as isize);
                *k_out.offset(out_idx as isize) = O::from_f32(fmul(k_scale, e4m32f(kb)));
                let vb = *(value_cache as *const u8).offset(cache_idx as isize);
                *v_out.offset(out_idx as isize) = O::from_f32(fmul(v_scale, e4m32f(vb)));
            } else {
                *k_out.offset(out_idx as isize) = *(key_cache as *const O).offset(cache_idx as isize);
                *v_out.offset(out_idx as isize) = *(value_cache as *const O).offset(cache_idx as isize);
            }
            i = i.wrapping_add(thread::blockDim_x() as i32);
        }
    }

    // GENERATED KERNELS BEGIN
    #[kernel]
    pub unsafe fn _ZN20mistralrs_flashinfer33gather_kv_cache_flashinfer_kernelI13__nv_fp8_e4m3fEEvPKT_S4_PT0_S6_PKiS8_iiiiiiff(kc: *const u8, vc: *const u8, ko: *mut f32, vo: *mut f32, bt: *const i32, cu: *const i32, nt: i32, ns: i32, bs: i32, bts: i32, nkv: i32, hs: i32, ksc: f32, vsc: f32) {
        gather_fi::<u8, f32, true>(kc, vc, ko, vo, bt, cu, nt, ns, bs, bts, nkv, hs, ksc, vsc)
    }
    #[kernel]
    pub unsafe fn _ZN20mistralrs_flashinfer33gather_kv_cache_flashinfer_kernelI13__nv_fp8_e4m313__nv_bfloat16EEvPKT_S5_PT0_S7_PKiS9_iiiiiiff(kc: *const u8, vc: *const u8, ko: *mut B, vo: *mut B, bt: *const i32, cu: *const i32, nt: i32, ns: i32, bs: i32, bts: i32, nkv: i32, hs: i32, ksc: f32, vsc: f32) {
        gather_fi::<u8, B, true>(kc, vc, ko, vo, bt, cu, nt, ns, bs, bts, nkv, hs, ksc, vsc)
    }
    #[kernel]
    pub unsafe fn _ZN20mistralrs_flashinfer33gather_kv_cache_flashinfer_kernelI13__nv_fp8_e4m36__halfEEvPKT_S5_PT0_S7_PKiS9_iiiiiiff(kc: *const u8, vc: *const u8, ko: *mut H, vo: *mut H, bt: *const i32, cu: *const i32, nt: i32, ns: i32, bs: i32, bts: i32, nkv: i32, hs: i32, ksc: f32, vsc: f32) {
        gather_fi::<u8, H, true>(kc, vc, ko, vo, bt, cu, nt, ns, bs, bts, nkv, hs, ksc, vsc)
    }
    #[kernel]
    pub unsafe fn _ZN20mistralrs_flashinfer33gather_kv_cache_flashinfer_kernelIffEEvPKT_S3_PT0_S5_PKiS7_iiiiiiff(kc: *const f32, vc: *const f32, ko: *mut f32, vo: *mut f32, bt: *const i32, cu: *const i32, nt: i32, ns: i32, bs: i32, bts: i32, nkv: i32, hs: i32, ksc: f32, vsc: f32) {
        gather_fi::<f32, f32, false>(kc, vc, ko, vo, bt, cu, nt, ns, bs, bts, nkv, hs, ksc, vsc)
    }
    #[kernel]
    pub unsafe fn _ZN20mistralrs_flashinfer33gather_kv_cache_flashinfer_kernelI13__nv_bfloat16S1_EEvPKT_S4_PT0_S6_PKiS8_iiiiiiff(kc: *const B, vc: *const B, ko: *mut B, vo: *mut B, bt: *const i32, cu: *const i32, nt: i32, ns: i32, bs: i32, bts: i32, nkv: i32, hs: i32, ksc: f32, vsc: f32) {
        gather_fi::<B, B, false>(kc, vc, ko, vo, bt, cu, nt, ns, bs, bts, nkv, hs, ksc, vsc)
    }
    #[kernel]
    pub unsafe fn _ZN20mistralrs_flashinfer33gather_kv_cache_flashinfer_kernelI6__halfS1_EEvPKT_S4_PT0_S6_PKiS8_iiiiiiff(kc: *const H, vc: *const H, ko: *mut H, vo: *mut H, bt: *const i32, cu: *const i32, nt: i32, ns: i32, bs: i32, bts: i32, nkv: i32, hs: i32, ksc: f32, vsc: f32) {
        gather_fi::<H, H, false>(kc, vc, ko, vo, bt, cu, nt, ns, bs, bts, nkv, hs, ksc, vsc)
    }
    #[kernel]
    pub unsafe fn _ZN20mistralrs_flashinfer35reshape_and_cache_flashinfer_kernelIf13__nv_fp8_e4m3EEvPKT_S4_PT0_S6_PKliiiiiff(k: *const f32, v: *const f32, kc: *mut u8, vc: *mut u8, slot: *const i64, nh: i32, hs: i32, bs: i32, ks: i32, vs: i32, ksc: f32, vsc: f32) {
        reshape_fi::<f32, u8, true>(k, v, kc, vc, slot, nh, hs, bs, ks, vs, ksc, vsc)
    }
    #[kernel]
    pub unsafe fn _ZN20mistralrs_flashinfer35reshape_and_cache_flashinfer_kernelI13__nv_bfloat1613__nv_fp8_e4m3EEvPKT_S5_PT0_S7_PKliiiiiff(k: *const B, v: *const B, kc: *mut u8, vc: *mut u8, slot: *const i64, nh: i32, hs: i32, bs: i32, ks: i32, vs: i32, ksc: f32, vsc: f32) {
        reshape_fi::<B, u8, true>(k, v, kc, vc, slot, nh, hs, bs, ks, vs, ksc, vsc)
    }
    #[kernel]
    pub unsafe fn _ZN20mistralrs_flashinfer35reshape_and_cache_flashinfer_kernelI6__half13__nv_fp8_e4m3EEvPKT_S5_PT0_S7_PKliiiiiff(k: *const H, v: *const H, kc: *mut u8, vc: *mut u8, slot: *const i64, nh: i32, hs: i32, bs: i32, ks: i32, vs: i32, ksc: f32, vsc: f32) {
        reshape_fi::<H, u8, true>(k, v, kc, vc, slot, nh, hs, bs, ks, vs, ksc, vsc)
    }
    #[kernel]
    pub unsafe fn _ZN20mistralrs_flashinfer35reshape_and_cache_flashinfer_kernelIffEEvPKT_S3_PT0_S5_PKliiiiiff(k: *const f32, v: *const f32, kc: *mut f32, vc: *mut f32, slot: *const i64, nh: i32, hs: i32, bs: i32, ks: i32, vs: i32, ksc: f32, vsc: f32) {
        reshape_fi::<f32, f32, false>(k, v, kc, vc, slot, nh, hs, bs, ks, vs, ksc, vsc)
    }
    #[kernel]
    pub unsafe fn _ZN20mistralrs_flashinfer35reshape_and_cache_flashinfer_kernelI13__nv_bfloat16S1_EEvPKT_S4_PT0_S6_PKliiiiiff(k: *const B, v: *const B, kc: *mut B, vc: *mut B, slot: *const i64, nh: i32, hs: i32, bs: i32, ks: i32, vs: i32, ksc: f32, vsc: f32) {
        reshape_fi::<B, B, false>(k, v, kc, vc, slot, nh, hs, bs, ks, vs, ksc, vsc)
    }
    #[kernel]
    pub unsafe fn _ZN20mistralrs_flashinfer35reshape_and_cache_flashinfer_kernelI6__halfS1_EEvPKT_S4_PT0_S6_PKliiiiiff(k: *const H, v: *const H, kc: *mut H, vc: *mut H, slot: *const i64, nh: i32, hs: i32, bs: i32, ks: i32, vs: i32, ksc: f32, vsc: f32) {
        reshape_fi::<H, H, false>(k, v, kc, vc, slot, nh, hs, bs, ks, vs, ksc, vsc)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 32, 16, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd3(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 32, 8, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd5(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 32, 7, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd7(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 32, 6, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd9(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 32, 5, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd11(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 32, 4, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd13(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 32, 3, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd15(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 32, 2, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd17(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 2, 16, 32, 1, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd19(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 32, 16, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd21(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 32, 8, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd23(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 32, 7, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd25(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 32, 6, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd27(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 32, 5, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd29(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 32, 4, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd31(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 32, 3, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd33(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 32, 2, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd35(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 2, 16, 32, 1, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd37(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 32, 16, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd39(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 32, 8, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd41(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 32, 7, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd43(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 32, 6, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd45(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 32, 5, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd47(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 32, 4, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd49(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 32, 3, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd51(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 32, 2, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd53(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 2, 16, 32, 1, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd55(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 32, 16, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd57(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 32, 8, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd59(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 32, 7, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd61(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 32, 6, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd63(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 32, 5, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd65(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 32, 4, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd67(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 32, 3, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd69(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 32, 2, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd71(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 2, 16, 32, 1, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd73(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 16, 16, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd75(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 16, 8, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj7ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd77(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 16, 7, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj6ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd79(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 16, 6, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj5ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd81(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 16, 5, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd83(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 16, 4, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd85(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 16, 3, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd87(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 16, 2, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd89(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 2, 16, 16, 1, 8, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd91(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 16, 16, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd93(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 16, 8, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj7ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd95(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 16, 7, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj6ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd97(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 16, 6, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj5ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd99(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 16, 5, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd101(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 16, 4, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd103(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 16, 3, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd105(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 16, 2, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd107(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 2, 16, 16, 1, 8, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd109(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 16, 16, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd111(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 16, 8, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj7ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd113(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 16, 7, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj6ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd115(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 16, 6, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj5ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd117(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 16, 5, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd119(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 16, 4, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd121(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 16, 3, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd123(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 16, 2, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd125(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 2, 16, 16, 1, 8, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd127(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 16, 16, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd129(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 16, 8, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj7ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd131(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 16, 7, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj6ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd133(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 16, 6, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj5ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd135(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 16, 5, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd137(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 16, 4, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd139(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 16, 3, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd141(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 16, 2, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd143(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 2, 16, 16, 1, 8, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd145(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 8, 16, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj8ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd147(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 8, 8, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj7ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd149(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 8, 7, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj6ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd151(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 8, 6, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj5ELj3ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd153(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 8, 5, 3, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj4ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd155(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 8, 4, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj3ELj5ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd157(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 8, 3, 5, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj2ELj8ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd159(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 8, 2, 8, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj8ELj1ELj16ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd161(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 2, 16, 8, 1, 16, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd163(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 8, 16, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj8ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd165(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 8, 8, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj7ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd167(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 8, 7, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj6ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd169(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 8, 6, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj5ELj3ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd171(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 8, 5, 3, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj4ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd173(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 8, 4, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj3ELj5ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd175(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 8, 3, 5, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj2ELj8ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd177(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 8, 2, 8, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj8ELj1ELj16ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd179(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 2, 16, 8, 1, 16, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd181(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 8, 16, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj8ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd183(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 8, 8, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj7ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd185(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 8, 7, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj6ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd187(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 8, 6, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj5ELj3ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd189(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 8, 5, 3, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj4ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd191(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 8, 4, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj3ELj5ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd193(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 8, 3, 5, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj2ELj8ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd195(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 8, 2, 8, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj8ELj1ELj16ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd197(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 2, 16, 8, 1, 16, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd199(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 8, 16, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj8ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd201(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 8, 8, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj7ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd203(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 8, 7, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj6ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd205(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 8, 6, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj5ELj3ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd207(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 8, 5, 3, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj4ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd209(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 8, 4, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj3ELj5ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd211(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 8, 3, 5, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj2ELj8ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd213(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 8, 2, 8, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj8ELj1ELj16ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd215(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 2, 16, 8, 1, 16, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj16ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd217(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 4, 16, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj8ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd219(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 4, 8, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj7ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd221(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 4, 7, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj6ELj5ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd223(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 4, 6, 5, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj5ELj6ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd225(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 4, 5, 6, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj4ELj8ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd227(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 4, 4, 8, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj3ELj10ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd229(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 4, 3, 10, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj2ELj16ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd231(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 4, 2, 16, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj4ELj1ELj32ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd233(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 2, 16, 4, 1, 32, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj16ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd235(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 4, 16, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj8ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd237(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 4, 8, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj7ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd239(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 4, 7, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj6ELj5ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd241(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 4, 6, 5, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj5ELj6ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd243(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 4, 5, 6, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj4ELj8ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd245(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 4, 4, 8, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj3ELj10ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd247(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 4, 3, 10, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj2ELj16ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd249(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 4, 2, 16, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj4ELj1ELj32ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd251(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 2, 16, 4, 1, 32, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj16ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd253(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 4, 16, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj8ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd255(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 4, 8, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj7ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd257(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 4, 7, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj6ELj5ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd259(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 4, 6, 5, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj5ELj6ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd261(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 4, 5, 6, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj4ELj8ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd263(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 4, 4, 8, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj3ELj10ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd265(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 4, 3, 10, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj2ELj16ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd267(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 4, 2, 16, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj4ELj1ELj32ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd269(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 2, 16, 4, 1, 32, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj16ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd271(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 4, 16, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj8ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd273(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 4, 8, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj7ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd275(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 4, 7, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj6ELj5ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd277(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 4, 6, 5, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj5ELj6ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd279(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 4, 5, 6, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj4ELj8ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd281(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 4, 4, 8, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj3ELj10ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd283(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 4, 3, 10, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj2ELj16ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd285(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 1, 16, 4, 2, 16, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj4ELj1ELj32ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIf13__nv_fp8_e4m3fiEEEEvT7_
    #[kernel]
    pub unsafe fn bd287(q: *const f32, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, E, 2, 2, 16, 4, 1, 32, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd289(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 32, 16, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd291(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 32, 8, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd293(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 32, 7, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd295(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 32, 6, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd297(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 32, 5, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd299(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 32, 4, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd301(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 32, 3, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd303(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 32, 2, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd305(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 2, 16, 32, 1, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd307(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 32, 16, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd309(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 32, 8, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd311(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 32, 7, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd313(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 32, 6, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd315(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 32, 5, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd317(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 32, 4, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd319(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 32, 3, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd321(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 32, 2, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd323(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 2, 16, 32, 1, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd325(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 32, 16, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd327(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 32, 8, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd329(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 32, 7, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd331(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 32, 6, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd333(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 32, 5, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd335(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 32, 4, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd337(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 32, 3, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd339(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 32, 2, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd341(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 2, 16, 32, 1, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd343(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 32, 16, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd345(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 32, 8, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd347(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 32, 7, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd349(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 32, 6, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd351(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 32, 5, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd353(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 32, 4, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd355(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 32, 3, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd357(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 32, 2, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd359(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 2, 16, 32, 1, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd361(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 16, 16, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd363(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 16, 8, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj7ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd365(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 16, 7, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj6ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd367(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 16, 6, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj5ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd369(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 16, 5, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd371(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 16, 4, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd373(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 16, 3, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd375(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 16, 2, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd377(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 2, 16, 16, 1, 8, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd379(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 16, 16, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd381(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 16, 8, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj7ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd383(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 16, 7, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj6ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd385(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 16, 6, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj5ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd387(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 16, 5, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd389(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 16, 4, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd391(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 16, 3, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd393(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 16, 2, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd395(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 2, 16, 16, 1, 8, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd397(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 16, 16, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd399(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 16, 8, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj7ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd401(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 16, 7, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj6ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd403(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 16, 6, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj5ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd405(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 16, 5, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd407(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 16, 4, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd409(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 16, 3, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd411(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 16, 2, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd413(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 2, 16, 16, 1, 8, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd415(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 16, 16, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd417(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 16, 8, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj7ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd419(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 16, 7, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj6ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd421(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 16, 6, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj5ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd423(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 16, 5, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd425(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 16, 4, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd427(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 16, 3, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd429(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 16, 2, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd431(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 2, 16, 16, 1, 8, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd433(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 8, 16, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj8ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd435(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 8, 8, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj7ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd437(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 8, 7, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj6ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd439(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 8, 6, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj5ELj3ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd441(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 8, 5, 3, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj4ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd443(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 8, 4, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj3ELj5ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd445(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 8, 3, 5, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj2ELj8ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd447(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 8, 2, 8, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj8ELj1ELj16ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd449(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 2, 16, 8, 1, 16, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd451(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 8, 16, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj8ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd453(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 8, 8, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj7ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd455(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 8, 7, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj6ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd457(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 8, 6, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj5ELj3ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd459(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 8, 5, 3, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj4ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd461(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 8, 4, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj3ELj5ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd463(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 8, 3, 5, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj2ELj8ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd465(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 8, 2, 8, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj8ELj1ELj16ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd467(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 2, 16, 8, 1, 16, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd469(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 8, 16, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj8ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd471(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 8, 8, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj7ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd473(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 8, 7, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj6ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd475(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 8, 6, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj5ELj3ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd477(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 8, 5, 3, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj4ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd479(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 8, 4, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj3ELj5ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd481(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 8, 3, 5, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj2ELj8ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd483(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 8, 2, 8, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj8ELj1ELj16ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd485(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 2, 16, 8, 1, 16, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd487(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 8, 16, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj8ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd489(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 8, 8, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj7ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd491(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 8, 7, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj6ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd493(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 8, 6, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj5ELj3ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd495(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 8, 5, 3, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj4ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd497(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 8, 4, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj3ELj5ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd499(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 8, 3, 5, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj2ELj8ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd501(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 8, 2, 8, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj8ELj1ELj16ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd503(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 2, 16, 8, 1, 16, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj16ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd505(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 4, 16, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj8ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd507(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 4, 8, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj7ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd509(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 4, 7, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj6ELj5ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd511(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 4, 6, 5, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj5ELj6ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd513(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 4, 5, 6, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj4ELj8ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd515(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 4, 4, 8, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj3ELj10ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd517(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 4, 3, 10, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj2ELj16ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd519(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 4, 2, 16, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj4ELj1ELj32ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd521(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 2, 16, 4, 1, 32, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj16ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd523(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 4, 16, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj8ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd525(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 4, 8, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj7ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd527(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 4, 7, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj6ELj5ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd529(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 4, 6, 5, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj5ELj6ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd531(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 4, 5, 6, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj4ELj8ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd533(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 4, 4, 8, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj3ELj10ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd535(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 4, 3, 10, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj2ELj16ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd537(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 4, 2, 16, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj4ELj1ELj32ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd539(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 2, 16, 4, 1, 32, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj16ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd541(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 4, 16, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj8ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd543(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 4, 8, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj7ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd545(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 4, 7, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj6ELj5ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd547(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 4, 6, 5, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj5ELj6ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd549(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 4, 5, 6, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj4ELj8ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd551(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 4, 4, 8, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj3ELj10ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd553(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 4, 3, 10, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj2ELj16ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd555(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 4, 2, 16, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj4ELj1ELj32ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd557(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 2, 16, 4, 1, 32, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj16ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd559(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 4, 16, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj8ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd561(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 4, 8, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj7ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd563(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 4, 7, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj6ELj5ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd565(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 4, 6, 5, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj5ELj6ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd567(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 4, 5, 6, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj4ELj8ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd569(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 4, 4, 8, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj3ELj10ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd571(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 4, 3, 10, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj2ELj16ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd573(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 1, 16, 4, 2, 16, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj4ELj1ELj32ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat1613__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd575(q: *const B, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, E, 2, 2, 16, 4, 1, 32, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd577(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 32, 16, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd579(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 32, 8, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd581(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 32, 7, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd583(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 32, 6, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd585(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 32, 5, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd587(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 32, 4, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd589(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 32, 3, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd591(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 32, 2, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd593(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 2, 16, 32, 1, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd595(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 32, 16, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd597(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 32, 8, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd599(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 32, 7, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd601(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 32, 6, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd603(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 32, 5, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd605(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 32, 4, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd607(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 32, 3, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd609(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 32, 2, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd611(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 2, 16, 32, 1, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd613(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 32, 16, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd615(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 32, 8, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd617(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 32, 7, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd619(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 32, 6, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd621(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 32, 5, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd623(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 32, 4, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd625(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 32, 3, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd627(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 32, 2, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd629(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 2, 16, 32, 1, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd631(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 32, 16, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd633(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 32, 8, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd635(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 32, 7, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd637(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 32, 6, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd639(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 32, 5, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd641(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 32, 4, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd643(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 32, 3, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd645(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 32, 2, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd647(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 2, 16, 32, 1, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd649(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 16, 16, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd651(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 16, 8, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj7ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd653(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 16, 7, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj6ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd655(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 16, 6, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj5ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd657(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 16, 5, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd659(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 16, 4, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd661(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 16, 3, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd663(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 16, 2, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd665(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 2, 16, 16, 1, 8, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd667(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 16, 16, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd669(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 16, 8, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj7ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd671(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 16, 7, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj6ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd673(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 16, 6, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj5ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd675(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 16, 5, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd677(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 16, 4, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd679(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 16, 3, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd681(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 16, 2, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd683(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 2, 16, 16, 1, 8, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd685(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 16, 16, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd687(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 16, 8, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj7ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd689(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 16, 7, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj6ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd691(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 16, 6, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj5ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd693(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 16, 5, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd695(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 16, 4, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd697(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 16, 3, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd699(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 16, 2, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd701(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 2, 16, 16, 1, 8, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd703(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 16, 16, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd705(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 16, 8, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj7ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd707(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 16, 7, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj6ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd709(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 16, 6, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj5ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd711(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 16, 5, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd713(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 16, 4, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd715(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 16, 3, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd717(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 16, 2, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd719(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 2, 16, 16, 1, 8, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd721(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 8, 16, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj8ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd723(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 8, 8, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj7ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd725(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 8, 7, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj6ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd727(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 8, 6, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj5ELj3ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd729(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 8, 5, 3, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj4ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd731(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 8, 4, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj3ELj5ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd733(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 8, 3, 5, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj2ELj8ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd735(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 8, 2, 8, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj8ELj1ELj16ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd737(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 2, 16, 8, 1, 16, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd739(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 8, 16, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj8ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd741(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 8, 8, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj7ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd743(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 8, 7, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj6ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd745(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 8, 6, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj5ELj3ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd747(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 8, 5, 3, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj4ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd749(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 8, 4, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj3ELj5ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd751(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 8, 3, 5, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj2ELj8ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd753(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 8, 2, 8, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj8ELj1ELj16ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd755(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 2, 16, 8, 1, 16, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd757(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 8, 16, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj8ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd759(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 8, 8, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj7ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd761(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 8, 7, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj6ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd763(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 8, 6, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj5ELj3ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd765(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 8, 5, 3, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj4ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd767(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 8, 4, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj3ELj5ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd769(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 8, 3, 5, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj2ELj8ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd771(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 8, 2, 8, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj8ELj1ELj16ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd773(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 2, 16, 8, 1, 16, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd775(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 8, 16, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj8ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd777(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 8, 8, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj7ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd779(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 8, 7, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj6ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd781(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 8, 6, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj5ELj3ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd783(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 8, 5, 3, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj4ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd785(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 8, 4, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj3ELj5ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd787(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 8, 3, 5, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj8ELj2ELj8ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd789(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 8, 2, 8, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj8ELj1ELj16ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd791(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 2, 16, 8, 1, 16, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj16ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd793(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 4, 16, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj8ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd795(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 4, 8, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj7ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd797(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 4, 7, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj6ELj5ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd799(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 4, 6, 5, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj5ELj6ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd801(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 4, 5, 6, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj4ELj8ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd803(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 4, 4, 8, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj3ELj10ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd805(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 4, 3, 10, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj2ELj16ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd807(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 4, 2, 16, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj4ELj1ELj32ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd809(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 2, 16, 4, 1, 32, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj16ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd811(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 4, 16, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj8ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd813(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 4, 8, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj7ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd815(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 4, 7, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj6ELj5ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd817(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 4, 6, 5, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj5ELj6ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd819(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 4, 5, 6, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj4ELj8ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd821(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 4, 4, 8, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj3ELj10ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd823(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 4, 3, 10, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj2ELj16ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd825(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 4, 2, 16, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj4ELj1ELj32ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd827(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 2, 16, 4, 1, 32, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj16ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd829(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 4, 16, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj8ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd831(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 4, 8, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj7ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd833(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 4, 7, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj6ELj5ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd835(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 4, 6, 5, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj5ELj6ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd837(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 4, 5, 6, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj4ELj8ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd839(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 4, 4, 8, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj3ELj10ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd841(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 4, 3, 10, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj2ELj16ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd843(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 4, 2, 16, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj4ELj1ELj32ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd845(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 2, 16, 4, 1, 32, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj16ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd847(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 4, 16, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj8ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd849(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 4, 8, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj7ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd851(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 4, 7, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj6ELj5ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd853(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 4, 6, 5, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj5ELj6ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd855(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 4, 5, 6, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj4ELj8ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd857(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 4, 4, 8, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj3ELj10ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd859(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 4, 3, 10, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj4ELj2ELj16ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd861(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 1, 16, 4, 2, 16, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj2ELj16ELj4ELj1ELj32ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd863(q: *const H, k: *const E, v: *const E, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, E, 2, 2, 16, 4, 1, 32, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd865(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 16, 32, 16, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd867(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 16, 32, 8, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd869(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 16, 32, 7, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd871(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 16, 32, 6, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd873(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 16, 32, 5, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd875(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 16, 32, 4, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd877(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 16, 32, 3, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd879(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 16, 32, 2, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd881(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 4, 16, 32, 1, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd883(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 16, 32, 16, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd885(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 16, 32, 8, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd887(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 16, 32, 7, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd889(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 16, 32, 6, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd891(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 16, 32, 5, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd893(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 16, 32, 4, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd895(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 16, 32, 3, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd897(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 16, 32, 2, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd899(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 4, 16, 32, 1, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd901(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 16, 32, 16, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd903(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 16, 32, 8, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd905(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 16, 32, 7, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd907(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 16, 32, 6, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd909(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 16, 32, 5, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd911(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 16, 32, 4, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd913(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 16, 32, 3, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd915(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 16, 32, 2, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd917(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 4, 16, 32, 1, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd919(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 16, 32, 16, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd921(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 16, 32, 8, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd923(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 16, 32, 7, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd925(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 16, 32, 6, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd927(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 16, 32, 5, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd929(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 16, 32, 4, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd931(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 16, 32, 3, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd933(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 16, 32, 2, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd935(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 4, 16, 32, 1, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd937(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 8, 32, 16, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd939(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 8, 32, 8, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd941(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 8, 32, 7, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd943(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 8, 32, 6, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd945(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 8, 32, 5, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd947(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 8, 32, 4, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd949(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 8, 32, 3, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd951(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 8, 32, 2, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd953(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 4, 8, 32, 1, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd955(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 8, 32, 16, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd957(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 8, 32, 8, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd959(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 8, 32, 7, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd961(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 8, 32, 6, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd963(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 8, 32, 5, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd965(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 8, 32, 4, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd967(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 8, 32, 3, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd969(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 8, 32, 2, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd971(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 4, 8, 32, 1, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd973(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 8, 32, 16, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd975(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 8, 32, 8, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd977(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 8, 32, 7, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd979(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 8, 32, 6, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd981(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 8, 32, 5, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd983(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 8, 32, 4, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd985(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 8, 32, 3, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd987(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 8, 32, 2, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd989(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 4, 8, 32, 1, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd991(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 8, 32, 16, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd993(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 8, 32, 8, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd995(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 8, 32, 7, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd997(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 8, 32, 6, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd999(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 8, 32, 5, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1001(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 8, 32, 4, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1003(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 8, 32, 3, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1005(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 8, 32, 2, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1007(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 4, 8, 32, 1, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1009(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 32, 16, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1011(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 32, 8, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1013(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 32, 7, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1015(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 32, 6, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1017(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 32, 5, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1019(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 32, 4, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1021(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 32, 3, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1023(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 32, 2, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj4ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1025(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 4, 4, 32, 1, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1027(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 32, 16, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1029(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 32, 8, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1031(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 32, 7, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1033(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 32, 6, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1035(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 32, 5, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1037(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 32, 4, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1039(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 32, 3, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1041(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 32, 2, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj4ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1043(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 4, 4, 32, 1, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1045(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 32, 16, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1047(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 32, 8, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1049(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 32, 7, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1051(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 32, 6, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1053(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 32, 5, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1055(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 32, 4, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1057(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 32, 3, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1059(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 32, 2, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj4ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1061(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 4, 4, 32, 1, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1063(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 32, 16, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1065(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 32, 8, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1067(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 32, 7, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1069(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 32, 6, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1071(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 32, 5, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1073(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 32, 4, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1075(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 32, 3, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1077(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 32, 2, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj4ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1079(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 4, 4, 32, 1, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1081(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 16, 16, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1083(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 16, 8, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj7ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1085(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 16, 7, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj6ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1087(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 16, 6, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj5ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1089(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 16, 5, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1091(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 16, 4, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1093(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 16, 3, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1095(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 16, 2, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj4ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1097(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 4, 4, 16, 1, 8, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1099(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 16, 16, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1101(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 16, 8, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj7ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1103(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 16, 7, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj6ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1105(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 16, 6, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj5ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1107(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 16, 5, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1109(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 16, 4, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1111(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 16, 3, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1113(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 16, 2, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj4ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1115(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 4, 4, 16, 1, 8, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1117(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 16, 16, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1119(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 16, 8, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj7ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1121(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 16, 7, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj6ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1123(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 16, 6, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj5ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1125(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 16, 5, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1127(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 16, 4, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1129(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 16, 3, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1131(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 16, 2, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj4ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1133(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 4, 4, 16, 1, 8, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1135(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 16, 16, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1137(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 16, 8, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj7ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1139(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 16, 7, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj6ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1141(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 16, 6, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj5ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1143(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 16, 5, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1145(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 16, 4, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1147(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 16, 3, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj4ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1149(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 1, 4, 16, 2, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
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
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj4ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsIfffiEEEEvT7_
    #[kernel]
    pub unsafe fn bd1151(q: *const f32, k: *const f32, v: *const f32, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut f32, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<f32, f32, 2, 4, 4, 16, 1, 8, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1153(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 16, 32, 16, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1155(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 16, 32, 8, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1157(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 16, 32, 7, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1159(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 16, 32, 6, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1161(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 16, 32, 5, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1163(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 16, 32, 4, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1165(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 16, 32, 3, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1167(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 16, 32, 2, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1169(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 4, 16, 32, 1, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1171(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 16, 32, 16, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1173(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 16, 32, 8, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1175(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 16, 32, 7, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1177(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 16, 32, 6, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1179(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 16, 32, 5, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1181(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 16, 32, 4, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1183(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 16, 32, 3, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1185(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 16, 32, 2, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1187(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 4, 16, 32, 1, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1189(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 16, 32, 16, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1191(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 16, 32, 8, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1193(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 16, 32, 7, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1195(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 16, 32, 6, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1197(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 16, 32, 5, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1199(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 16, 32, 4, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1201(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 16, 32, 3, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1203(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 16, 32, 2, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1205(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 4, 16, 32, 1, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1207(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 16, 32, 16, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1209(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 16, 32, 8, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1211(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 16, 32, 7, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1213(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 16, 32, 6, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1215(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 16, 32, 5, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1217(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 16, 32, 4, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1219(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 16, 32, 3, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1221(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 16, 32, 2, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1223(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 4, 16, 32, 1, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1225(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 32, 16, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1227(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 32, 8, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1229(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 32, 7, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1231(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 32, 6, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1233(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 32, 5, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1235(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 32, 4, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1237(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 32, 3, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1239(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 32, 2, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1241(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 4, 8, 32, 1, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1243(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 32, 16, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1245(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 32, 8, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1247(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 32, 7, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1249(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 32, 6, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1251(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 32, 5, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1253(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 32, 4, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1255(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 32, 3, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1257(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 32, 2, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1259(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 4, 8, 32, 1, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1261(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 32, 16, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1263(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 32, 8, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1265(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 32, 7, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1267(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 32, 6, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1269(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 32, 5, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1271(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 32, 4, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1273(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 32, 3, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1275(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 32, 2, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1277(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 4, 8, 32, 1, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1279(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 32, 16, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1281(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 32, 8, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1283(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 32, 7, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1285(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 32, 6, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1287(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 32, 5, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1289(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 32, 4, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1291(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 32, 3, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1293(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 32, 2, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1295(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 4, 8, 32, 1, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1297(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 16, 16, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1299(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 16, 8, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj7ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1301(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 16, 7, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj6ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1303(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 16, 6, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj5ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1305(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 16, 5, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1307(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 16, 4, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1309(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 16, 3, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1311(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 16, 2, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1313(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 4, 8, 16, 1, 8, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1315(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 16, 16, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1317(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 16, 8, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj7ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1319(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 16, 7, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj6ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1321(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 16, 6, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj5ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1323(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 16, 5, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1325(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 16, 4, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1327(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 16, 3, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1329(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 16, 2, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1331(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 4, 8, 16, 1, 8, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1333(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 16, 16, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1335(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 16, 8, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj7ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1337(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 16, 7, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj6ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1339(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 16, 6, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj5ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1341(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 16, 5, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1343(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 16, 4, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1345(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 16, 3, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1347(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 16, 2, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1349(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 4, 8, 16, 1, 8, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1351(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 16, 16, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1353(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 16, 8, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj7ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1355(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 16, 7, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj6ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1357(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 16, 6, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj5ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1359(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 16, 5, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1361(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 16, 4, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1363(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 16, 3, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1365(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 16, 2, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1367(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 4, 8, 16, 1, 8, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1369(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 8, 16, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj8ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1371(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 8, 8, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj7ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1373(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 8, 7, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj6ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1375(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 8, 6, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj5ELj3ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1377(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 8, 5, 3, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj4ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1379(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 8, 4, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj3ELj5ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1381(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 8, 3, 5, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj2ELj8ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1383(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 8, 2, 8, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj8ELj1ELj16ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1385(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 4, 8, 8, 1, 16, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1387(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 8, 16, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj8ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1389(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 8, 8, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj7ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1391(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 8, 7, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj6ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1393(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 8, 6, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj5ELj3ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1395(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 8, 5, 3, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj4ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1397(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 8, 4, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj3ELj5ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1399(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 8, 3, 5, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj2ELj8ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1401(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 8, 2, 8, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj8ELj1ELj16ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1403(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 4, 8, 8, 1, 16, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1405(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 8, 16, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj8ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1407(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 8, 8, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj7ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1409(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 8, 7, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj6ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1411(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 8, 6, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj5ELj3ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1413(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 8, 5, 3, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj4ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1415(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 8, 4, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj3ELj5ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1417(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 8, 3, 5, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj2ELj8ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1419(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 8, 2, 8, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj8ELj1ELj16ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1421(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 4, 8, 8, 1, 16, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1423(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 8, 16, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj8ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1425(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 8, 8, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj7ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1427(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 8, 7, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj6ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1429(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 8, 6, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj5ELj3ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1431(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 8, 5, 3, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj4ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1433(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 8, 4, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj3ELj5ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1435(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 8, 3, 5, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj2ELj8ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1437(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 1, 8, 8, 2, 8, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
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
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj8ELj1ELj16ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI13__nv_bfloat16S5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1439(q: *const B, k: *const B, v: *const B, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut B, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<B, B, 2, 4, 8, 8, 1, 16, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1441(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 16, 32, 16, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1443(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 16, 32, 8, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1445(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 16, 32, 7, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1447(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 16, 32, 6, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1449(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 16, 32, 5, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1451(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 16, 32, 4, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1453(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 16, 32, 3, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1455(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 16, 32, 2, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1457(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 4, 16, 32, 1, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1459(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 16, 32, 16, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1461(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 16, 32, 8, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1463(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 16, 32, 7, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1465(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 16, 32, 6, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1467(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 16, 32, 5, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1469(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 16, 32, 4, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1471(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 16, 32, 3, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1473(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 16, 32, 2, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1475(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 4, 16, 32, 1, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1477(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 16, 32, 16, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1479(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 16, 32, 8, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1481(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 16, 32, 7, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1483(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 16, 32, 6, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1485(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 16, 32, 5, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1487(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 16, 32, 4, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1489(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 16, 32, 3, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1491(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 16, 32, 2, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1493(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 4, 16, 32, 1, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1495(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 16, 32, 16, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1497(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 16, 32, 8, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1499(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 16, 32, 7, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1501(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 16, 32, 6, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1503(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 16, 32, 5, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1505(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 16, 32, 4, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1507(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 16, 32, 3, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj16ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1509(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 16, 32, 2, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj16ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1511(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 4, 16, 32, 1, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1513(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 32, 16, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1515(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 32, 8, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1517(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 32, 7, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1519(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 32, 6, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1521(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 32, 5, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1523(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 32, 4, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1525(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 32, 3, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1527(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 32, 2, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1529(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 4, 8, 32, 1, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1531(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 32, 16, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1533(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 32, 8, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1535(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 32, 7, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1537(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 32, 6, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1539(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 32, 5, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1541(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 32, 4, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1543(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 32, 3, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1545(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 32, 2, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1547(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 4, 8, 32, 1, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1549(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 32, 16, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1551(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 32, 8, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1553(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 32, 7, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1555(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 32, 6, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1557(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 32, 5, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1559(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 32, 4, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1561(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 32, 3, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1563(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 32, 2, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1565(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 4, 8, 32, 1, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1567(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 32, 16, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1569(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 32, 8, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj7ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1571(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 32, 7, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj6ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1573(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 32, 6, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj5ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1575(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 32, 5, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1577(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 32, 4, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj3ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1579(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 32, 3, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj32ELj2ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1581(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 32, 2, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj32ELj1ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1583(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 4, 8, 32, 1, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1585(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 16, 16, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1587(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 16, 8, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj7ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1589(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 16, 7, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj6ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1591(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 16, 6, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj5ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1593(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 16, 5, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1595(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 16, 4, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1597(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 16, 3, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1599(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 16, 2, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1601(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 4, 8, 16, 1, 8, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1603(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 16, 16, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1605(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 16, 8, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj7ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1607(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 16, 7, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj6ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1609(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 16, 6, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj5ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1611(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 16, 5, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1613(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 16, 4, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1615(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 16, 3, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1617(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 16, 2, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1619(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 4, 8, 16, 1, 8, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1621(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 16, 16, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1623(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 16, 8, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj7ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1625(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 16, 7, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj6ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1627(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 16, 6, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj5ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1629(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 16, 5, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1631(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 16, 4, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1633(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 16, 3, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1635(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 16, 2, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1637(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 4, 8, 16, 1, 8, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1639(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 16, 16, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1641(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 16, 8, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj7ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1643(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 16, 7, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj6ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1645(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 16, 6, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj5ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1647(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 16, 5, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1649(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 16, 4, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj3ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1651(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 16, 3, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj16ELj2ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1653(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 16, 2, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj16ELj1ELj8ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1655(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 4, 8, 16, 1, 8, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1657(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 8, 16, 1, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj8ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1659(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 8, 8, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj7ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1661(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 8, 7, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj6ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1663(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 8, 6, 2, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj5ELj3ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1665(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 8, 5, 3, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj4ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1667(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 8, 4, 4, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj3ELj5ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1669(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 8, 3, 5, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj2ELj8ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1671(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 8, 2, 8, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj8ELj1ELj16ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1673(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 4, 8, 8, 1, 16, false, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj16ELj1ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1675(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 8, 16, 1, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj8ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1677(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 8, 8, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj7ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1679(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 8, 7, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj6ELj2ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1681(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 8, 6, 2, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj5ELj3ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1683(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 8, 5, 3, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj4ELj4ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1685(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 8, 4, 4, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj3ELj5ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1687(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 8, 3, 5, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj2ELj8ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1689(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 8, 2, 8, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj8ELj1ELj16ENS_16DefaultAttentionILb0ELb0ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1691(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 4, 8, 8, 1, 16, false, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1693(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 8, 16, 1, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj8ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1695(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 8, 8, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj7ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1697(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 8, 7, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj6ELj2ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1699(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 8, 6, 2, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj5ELj3ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1701(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 8, 5, 3, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj4ELj4ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1703(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 8, 4, 4, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj3ELj5ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1705(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 8, 3, 5, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj2ELj8ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1707(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 8, 2, 8, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj8ELj1ELj16ENS_16DefaultAttentionILb0ELb1ELb0ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1709(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 4, 8, 8, 1, 16, true, false>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj16ELj1ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1711(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 8, 16, 1, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj8ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1713(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 8, 8, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj7ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1715(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 8, 7, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj6ELj2ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1717(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 8, 6, 2, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj5ELj3ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1719(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 8, 5, 3, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj4ELj4ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1721(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 8, 4, 4, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj3ELj5ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1723(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 8, 3, 5, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
    }
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj8ELj8ELj2ELj8ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1725(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 1, 8, 8, 2, 8, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
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
    /// _ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj4ELj8ELj8ELj1ELj16ENS_16DefaultAttentionILb0ELb1ELb1ELb0EEENS_17BatchDecodeParamsI6__halfS5_S5_iEEEEvT7_
    #[kernel]
    pub unsafe fn bd1727(q: *const H, k: *const H, v: *const H, idx: *const i32, indptr: *const i32, last: *const i32, o: *mut H, lse: *mut f32, req: *const i32, tiles: *const i32, chunk: *const i32, mask: *const u8, nqh: u32, qsn: i32, qsh: i32, wl: i32, cap: f32, sm: f32, vs: f32, part: u32, fd: u32, fm: u32, fs: u32, fa: u32, bs: u32, sp: u32, sn: u32, sh: u32) {
        batch_decode::<H, H, 2, 4, 8, 8, 1, 16, true, true>(q, k, v, idx, indptr, last, o, lse, req, tiles, chunk, mask, nqh, qsn, qsh, wl, cap, sm, vs, part, fd, fm, fs, fa, bs, sp, sn, sh)
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
