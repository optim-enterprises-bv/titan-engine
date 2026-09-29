//! mistralrs-core's CUDA library (`libmistralrscuda.a`: gdn, ssm, graph, moe_gemv, moe_gemm,
//! moe_gemm_wmma, attention_prep, sort) in cuda-oxide, plus (src/launch.rs) pure-Rust twins of its
//! extern "C" host launchers. Bit-identical to the nvcc build (`-O3 --use_fast_math`, sm_120a,
//! `--default-stream per-thread`; the SASS of libmistralrscuda.a is the spec), checked
//! launcher-by-launcher by the gate in src/gate.rs.
//!
//! Kernel entry names are the reference's mangled names, parameter lists are the same.
//!
//! Fast-math lowerings read off the SASS (every one is written out explicitly here):
//! - f32 add/sub/mul/fma are `.ftz` (FADD.FTZ / FMUL.FTZ / FFMA.FTZ); `a + b*c` contracts to FFMA.
//! - `expf(x)` = `ex2.approx.ftz(x * log2e)`; `logf(x)` = `lg2.approx.ftz(x) * ln2`;
//!   `1/x` = `rcp.approx.ftz`; `a / b` = `a * rcp(b)` (contracted into an FFMA when added to);
//!   `rsqrtf` = `rsqrt.approx.ftz`; `fminf`/`fmaxf` = `min.ftz`/`max.ftz`.
//! - `log1pf` is libdevice's ftz variant, transcribed from the SASS (`log1p_ftz`).
//! - f32 comparisons are `setp.*.ftz` (a denormal compares as zero).
#![allow(non_snake_case, clippy::missing_safety_doc, clippy::too_many_arguments)]
#![allow(unsafe_op_in_unsafe_fn)]
mod gate;
pub mod launch;

use cuda_device::{DynamicSharedArray, SharedArray, bf16x2, convert, f16x2, float, kernel, ptx_asm, thread, warp};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    const FULL: u32 = 0xffff_ffff;

    // ------------------------------------------------------------------------------------------
    // fast-math primitives

    #[inline(always)]
    pub fn addf(a: f32, b: f32) -> f32 {
        float::add_rn_ftz_f32(a, b)
    }
    #[inline(always)]
    pub fn subf(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("sub.rn.ftz.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)) };
        r
    }
    #[inline(always)]
    pub fn mulf(a: f32, b: f32) -> f32 {
        float::mul_rn_ftz_f32(a, b)
    }
    #[inline(always)]
    pub fn fmaf(a: f32, b: f32, c: f32) -> f32 {
        float::fma_rn_ftz_f32(a, b, c)
    }
    #[inline(always)]
    pub fn ex2(x: f32) -> f32 {
        float::ex2_approx_ftz_f32(x)
    }
    #[inline(always)]
    pub fn lg2(x: f32) -> f32 {
        float::lg2_approx_ftz_f32(x)
    }
    #[inline(always)]
    pub fn rcp(x: f32) -> f32 {
        float::rcp_approx_ftz_f32(x)
    }
    #[inline(always)]
    pub fn rsqrt(x: f32) -> f32 {
        float::rsqrt_approx_ftz_f32(x)
    }
    #[inline(always)]
    pub fn maxf(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("max.ftz.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)) };
        r
    }
    #[inline(always)]
    pub fn minf(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("min.ftz.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)) };
        r
    }
    pub const LOG2E: f32 = f32::from_bits(0x3fb8aa3b);
    pub const LN2: f32 = f32::from_bits(0x3f317218);
    /// fast `expf(x)`.
    #[inline(always)]
    pub fn expf(x: f32) -> f32 {
        ex2(mulf(x, LOG2E))
    }
    /// fast `logf(x)`.
    #[inline(always)]
    pub fn logf(x: f32) -> f32 {
        mulf(lg2(x), LN2)
    }
    /// `a > b` as `setp.gt.ftz.f32`.
    #[inline(always)]
    pub fn gt(a: f32, b: f32) -> bool {
        let r: u32;
        unsafe {
            ptx_asm!("{ .reg .pred p; setp.gt.ftz.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }", out("=r") r, in("f") a, in("f") b, options(register_only))
        };
        r != 0
    }
    /// `a >= b` as `setp.ge.ftz.f32`.
    #[inline(always)]
    pub fn ge(a: f32, b: f32) -> bool {
        let r: u32;
        unsafe {
            ptx_asm!("{ .reg .pred p; setp.ge.ftz.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }", out("=r") r, in("f") a, in("f") b, options(register_only))
        };
        r != 0
    }
    /// libdevice `__nv_log1pf` (ftz flavour), transcribed from the reference SASS.
    #[inline(always)]
    pub fn log1p_ftz(u: f32) -> f32 {
        let r: f32;
        unsafe {
            ptx_asm!(
                "{ .reg .f32 t, m, s, e, p; .reg .b32 bt, be, bs, bm; .reg .pred p0, p1, p2;\n\
                 add.rz.ftz.f32 t, %1, 0f3F800000;\n\
                 mov.b32 bt, t;\n\
                 sub.s32 be, bt, 0x3f400000;\n\
                 and.b32 be, be, 0xff800000;\n\
                 sub.s32 bs, 0x40800000, be;\n\
                 mov.b32 s, bs;\n\
                 mov.b32 bm, %1;\n\
                 sub.s32 bm, bm, be;\n\
                 mov.b32 m, bm;\n\
                 cvt.rn.f32.s32 e, be;\n\
                 fma.rn.ftz.f32 s, s, 0f3E800000, 0fBF800000;\n\
                 mul.rn.ftz.f32 e, e, 0f34000000;\n\
                 add.rn.ftz.f32 m, m, s;\n\
                 fma.rn.ftz.f32 p, m, 0fBD39BF78, 0f3DD80012;\n\
                 fma.rn.ftz.f32 p, m, p, 0fBE0778E0;\n\
                 fma.rn.ftz.f32 p, m, p, 0f3E146475;\n\
                 fma.rn.ftz.f32 p, m, p, 0fBE2A68DD;\n\
                 fma.rn.ftz.f32 p, m, p, 0f3E4CAF9E;\n\
                 fma.rn.ftz.f32 p, m, p, 0fBE800042;\n\
                 fma.rn.ftz.f32 p, m, p, 0f3EAAAAE6;\n\
                 fma.rn.ftz.f32 p, m, p, 0fBF000000;\n\
                 mul.rn.ftz.f32 p, m, p;\n\
                 fma.rn.ftz.f32 p, m, p, m;\n\
                 fma.rn.ftz.f32 %0, e, 0f3F317218, p;\n\
                 mov.b32 bm, %1;\n\
                 setp.ge.u32 p0, bm, 0x7f800000;\n\
                 setp.gt.s32 p2, bm, -1082130432;\n\
                 and.pred p2, p2, p0;\n\
                 @p2 fma.rn.ftz.f32 %0, %1, 0f7F800000, 0f7F800000;\n\
                 setp.neu.ftz.f32 p1, %1, 0f00000000;\n\
                 not.pred p1, p1;\n\
                 and.pred p1, p1, p0;\n\
                 @p1 mov.f32 %0, 0f80000000;\n\
                 }",
                out("=f") r,
                in("f") u,
                options(register_only)
            )
        };
        r
    }

    // ------------------------------------------------------------------------------------------
    // half types

    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct H(pub u16);
    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct B(pub u16);
    pub trait Half: Copy {
        fn f(self) -> f32;
        fn t(v: f32) -> Self;
        fn zero() -> Self;
    }
    impl Half for H {
        #[inline(always)]
        fn f(self) -> f32 {
            convert::cvt_f32_f16x2_lo(self.0 as u32)
        }
        #[inline(always)]
        fn t(v: f32) -> H {
            H(convert::cvt_f16x2_f32(v, 0.0) as u16)
        }
        #[inline(always)]
        fn zero() -> H {
            H(0)
        }
    }
    impl Half for B {
        #[inline(always)]
        fn f(self) -> f32 {
            f32::from_bits((self.0 as u32) << 16)
        }
        #[inline(always)]
        fn t(v: f32) -> B {
            B(convert::cvt_bf16x2_f32(v, 0.0) as u16)
        }
        #[inline(always)]
        fn zero() -> B {
            B(0)
        }
    }

    #[inline(always)]
    fn off<T>(p: *const T, i: i32) -> *const T {
        p.wrapping_offset(i as isize)
    }
    #[inline(always)]
    fn offm<T>(p: *mut T, i: i32) -> *mut T {
        p.wrapping_offset(i as isize)
    }
    #[inline(always)]
    fn tid_x() -> i32 {
        thread::threadIdx_x() as i32
    }
    #[inline(always)]
    fn bid_x() -> i32 {
        thread::blockIdx_x() as i32
    }
    #[inline(always)]
    fn bid_y() -> i32 {
        thread::blockIdx_y() as i32
    }
    #[inline(always)]
    fn bdim_x() -> i32 {
        thread::blockDim_x() as i32
    }

    // ==========================================================================================
    // gdn.cu
    // ==========================================================================================

    /// Cooperative-load trip count `ceil(KD / 64)` as an associated const (unrollable bound).
    pub struct GK<const KD: usize>;
    impl<const KD: usize> GK<KD> {
        pub const NLD: usize = KD.div_ceil(64);
    }

    /// gated_delta_rule_recurrence_kernel_tiled<BK, 64> (KD = BK) and _fallback<64, 256> (KD = 0:
    /// runtime k_dim, dynamic shared k_buf/q_buf, local-memory state).
    #[inline(always)]
    unsafe fn gdr_tiled<const KD: usize>(
        q: *const f32, k: *const f32, v: *const f32, g: *const f32, beta: *const f32, state: *mut f32, output: *mut f32,
        seq_len: i32, k_dim_rt: i32, v_dim: i32,
    ) {
        const BV: i32 = 64;
        static mut KB: SharedArray<f32, 128> = SharedArray::UNINIT;
        static mut QB: SharedArray<f32, 128> = SharedArray::UNINIT;
        let kd: i32 = if KD == 0 { k_dim_rt } else { KD as i32 };
        let v_tile = bid_x();
        let bh = bid_y();
        let tid = tid_x();
        let v_idx = v_tile.wrapping_mul(BV).wrapping_add(tid);
        if v_idx >= v_dim {
            return;
        }
        let (k_buf, q_buf): (*mut f32, *mut f32) = if KD == 0 {
            let sh = DynamicSharedArray::<f32>::get();
            (sh, sh.wrapping_offset(kd as isize))
        } else {
            (SharedArray::as_raw_mut_ptr(&raw mut KB), SharedArray::as_raw_mut_ptr(&raw mut QB))
        };
        let q_bh = off(q, bh.wrapping_mul(seq_len).wrapping_mul(kd));
        let k_bh = off(k, bh.wrapping_mul(seq_len).wrapping_mul(kd));
        let v_bh = off(v, bh.wrapping_mul(seq_len).wrapping_mul(v_dim));
        let g_bh = off(g, bh.wrapping_mul(seq_len));
        let beta_bh = off(beta, bh.wrapping_mul(seq_len));
        let state_bh = offm(state, bh.wrapping_mul(kd).wrapping_mul(v_dim));
        let out_bh = offm(output, bh.wrapping_mul(seq_len).wrapping_mul(v_dim));

        if KD != 0 {
            // Compile-time KD: the state column is a [f32; KD] indexed only by constants inside
            // fully unrolled loops, so it lives in registers (the reference's `float s[BK]`).
            let mut s = [0f32; KD];
            let mut j = 0usize;
            while j < KD {
                cuda_device::thread::__unroll_config::<0>();
                s[j] = *off(state_bh, (j as i32).wrapping_mul(v_dim).wrapping_add(v_idx));
                j += 1;
            }
            let mut t = 0;
            while t < seq_len {
                // `for (j = tid; j < BK; j += BV)` as a constant-count guarded loop (same
                // iterations for any tid >= 0), fully unrolled like the reference's.
                let mut c = 0usize;
                while c < GK::<KD>::NLD {
                    cuda_device::thread::__unroll_config::<0>();
                    let j = tid.wrapping_add((c as i32) * BV);
                    if j < KD as i32 {
                        *k_buf.offset(j as isize) = *off(k_bh, t.wrapping_mul(kd).wrapping_add(j));
                    }
                    c += 1;
                }
                thread::sync_threads();
                let decay = expf(*off(g_bh, t));
                let beta_t = *off(beta_bh, t);
                let v_t = *off(v_bh, t.wrapping_mul(v_dim).wrapping_add(v_idx));
                let mut kv_mem = 0f32;
                let mut j = 0usize;
                while j < KD {
                    cuda_device::thread::__unroll_config::<0>();
                    s[j] = mulf(s[j], decay);
                    kv_mem = fmaf(s[j], *k_buf.add(j), kv_mem);
                    j += 1;
                }
                let delta = mulf(subf(v_t, kv_mem), beta_t);
                // `for (j = tid; j < BK; j += BV)` as a constant-count guarded loop (same
                // iterations for any tid >= 0), fully unrolled like the reference's.
                let mut c = 0usize;
                while c < GK::<KD>::NLD {
                    cuda_device::thread::__unroll_config::<0>();
                    let j = tid.wrapping_add((c as i32) * BV);
                    if j < KD as i32 {
                        *q_buf.offset(j as isize) = *off(q_bh, t.wrapping_mul(kd).wrapping_add(j));
                    }
                    c += 1;
                }
                thread::sync_threads();
                let mut y = 0f32;
                let mut j = 0usize;
                while j < KD {
                    cuda_device::thread::__unroll_config::<0>();
                    s[j] = fmaf(*k_buf.add(j), delta, s[j]);
                    y = fmaf(s[j], *q_buf.add(j), y);
                    j += 1;
                }
                *offm(out_bh, t.wrapping_mul(v_dim).wrapping_add(v_idx)) = y;
                thread::sync_threads();
                t += 1;
            }
            let mut j = 0usize;
            while j < KD {
                cuda_device::thread::__unroll_config::<0>();
                *offm(state_bh, (j as i32).wrapping_mul(v_dim).wrapping_add(v_idx)) = s[j];
                j += 1;
            }
            return;
        }
        let mut s = [0f32; 256];
        let n = kd;
        let mut j = 0;
        while j < n {
            s[j as usize] = *off(state_bh, j.wrapping_mul(v_dim).wrapping_add(v_idx));
            j += 1;
        }
        let mut t = 0;
        while t < seq_len {
            let mut j = tid;
            while j < n {
                *k_buf.offset(j as isize) = *off(k_bh, t.wrapping_mul(kd).wrapping_add(j));
                j += BV;
            }
            thread::sync_threads();
            let decay = expf(*off(g_bh, t));
            let beta_t = *off(beta_bh, t);
            let v_t = *off(v_bh, t.wrapping_mul(v_dim).wrapping_add(v_idx));
            let mut kv_mem = 0f32;
            let mut j = 0;
            while j < n {
                s[j as usize] = mulf(s[j as usize], decay);
                kv_mem = fmaf(s[j as usize], *k_buf.offset(j as isize), kv_mem);
                j += 1;
            }
            let delta = mulf(subf(v_t, kv_mem), beta_t);
            let mut j = tid;
            while j < n {
                *q_buf.offset(j as isize) = *off(q_bh, t.wrapping_mul(kd).wrapping_add(j));
                j += BV;
            }
            thread::sync_threads();
            let mut y = 0f32;
            let mut j = 0;
            while j < n {
                s[j as usize] = fmaf(*k_buf.offset(j as isize), delta, s[j as usize]);
                y = fmaf(s[j as usize], *q_buf.offset(j as isize), y);
                j += 1;
            }
            *offm(out_bh, t.wrapping_mul(v_dim).wrapping_add(v_idx)) = y;
            thread::sync_threads();
            t += 1;
        }
        let mut j = 0;
        while j < n {
            *offm(state_bh, j.wrapping_mul(v_dim).wrapping_add(v_idx)) = s[j as usize];
            j += 1;
        }
    }

    #[kernel]
    pub unsafe fn _Z40gated_delta_rule_recurrence_kernel_tiledILi64ELi64EEvPKfS1_S1_S1_S1_PfS2_ii(
        q: *const f32, k: *const f32, v: *const f32, g: *const f32, beta: *const f32, state: *mut f32, output: *mut f32,
        seq_len: i32, v_dim: i32,
    ) {
        gdr_tiled::<64>(q, k, v, g, beta, state, output, seq_len, 64, v_dim)
    }
    #[kernel]
    pub unsafe fn _Z40gated_delta_rule_recurrence_kernel_tiledILi128ELi64EEvPKfS1_S1_S1_S1_PfS2_ii(
        q: *const f32, k: *const f32, v: *const f32, g: *const f32, beta: *const f32, state: *mut f32, output: *mut f32,
        seq_len: i32, v_dim: i32,
    ) {
        gdr_tiled::<128>(q, k, v, g, beta, state, output, seq_len, 128, v_dim)
    }
    #[kernel]
    pub unsafe fn _Z43gated_delta_rule_recurrence_kernel_fallbackILi64ELi256EEvPKfS1_S1_S1_S1_PfS2_iii(
        q: *const f32, k: *const f32, v: *const f32, g: *const f32, beta: *const f32, state: *mut f32, output: *mut f32,
        seq_len: i32, k_dim: i32, v_dim: i32,
    ) {
        gdr_tiled::<0>(q, k, v, g, beta, state, output, seq_len, k_dim, v_dim)
    }

    /// `gdn_warp_sum<32>`: shfl.down butterfly then broadcast lane 0.
    #[inline(always)]
    fn gdn_warp_sum(mut x: f32) -> f32 {
        let mut o = 16;
        while o > 0 {
            x = addf(x, warp::shuffle_down_f32_sync(FULL, x, o));
            o >>= 1;
        }
        warp::shuffle_f32_sync(FULL, x, 0)
    }

    /// gated_delta_rule_recurrence_kernel_warp<BK, 4>.
    #[inline(always)]
    unsafe fn gdr_warp<const BK: usize>(
        q: *const f32, k: *const f32, v: *const f32, g: *const f32, beta: *const f32, state: *mut f32, output: *mut f32,
        seq_len: i32, v_dim: i32,
    ) {
        const NW: i32 = 4;
        let rpl = BK / 32;
        let bk = BK as i32;
        let lane = tid_x();
        let wid = thread::threadIdx_y() as i32;
        let v_idx = bid_x().wrapping_mul(NW).wrapping_add(wid);
        let bh = bid_y();
        if v_idx >= v_dim {
            return;
        }
        let q_bh = off(q, bh.wrapping_mul(seq_len).wrapping_mul(bk));
        let k_bh = off(k, bh.wrapping_mul(seq_len).wrapping_mul(bk));
        let v_bh = off(v, bh.wrapping_mul(seq_len).wrapping_mul(v_dim));
        let g_bh = off(g, bh.wrapping_mul(seq_len));
        let beta_bh = off(beta, bh.wrapping_mul(seq_len));
        let state_bh = offm(state, bh.wrapping_mul(bk).wrapping_mul(v_dim));
        let out_bh = offm(output, bh.wrapping_mul(seq_len).wrapping_mul(v_dim));
        let mut s = [0f32; 4];
        let mut r = 0;
        while r < rpl {
            let row = (r as i32).wrapping_mul(32).wrapping_add(lane);
            s[r] = *off(state_bh, row.wrapping_mul(v_dim).wrapping_add(v_idx));
            r += 1;
        }
        let mut t = 0;
        while t < seq_len {
            let q_t = off(q_bh, t.wrapping_mul(bk));
            let k_t = off(k_bh, t.wrapping_mul(bk));
            let mut k_reg = [0f32; 4];
            let mut q_reg = [0f32; 4];
            let mut kv_partial = 0f32;
            let mut r = 0;
            while r < rpl {
                let row = (r as i32).wrapping_mul(32).wrapping_add(lane);
                let kv = *off(k_t, row);
                k_reg[r] = kv;
                q_reg[r] = *off(q_t, row);
                kv_partial = fmaf(s[r], kv, kv_partial);
                r += 1;
            }
            let decay = expf(*off(g_bh, t));
            let kv_col = gdn_warp_sum(kv_partial);
            let delta = mulf(fmaf(-decay, kv_col, *off(v_bh, t.wrapping_mul(v_dim).wrapping_add(v_idx))), *off(beta_bh, t));
            let mut y_partial = 0f32;
            let mut r = 0;
            while r < rpl {
                s[r] = fmaf(k_reg[r], delta, mulf(decay, s[r]));
                y_partial = fmaf(s[r], q_reg[r], y_partial);
                r += 1;
            }
            let y_col = gdn_warp_sum(y_partial);
            if lane == 0 {
                *offm(out_bh, t.wrapping_mul(v_dim).wrapping_add(v_idx)) = y_col;
            }
            t += 1;
        }
        let mut r = 0;
        while r < rpl {
            let row = (r as i32).wrapping_mul(32).wrapping_add(lane);
            *offm(state_bh, row.wrapping_mul(v_dim).wrapping_add(v_idx)) = s[r];
            r += 1;
        }
    }

    #[kernel]
    pub unsafe fn _Z39gated_delta_rule_recurrence_kernel_warpILi64ELi4EEvPKfS1_S1_S1_S1_PfS2_ii(
        q: *const f32, k: *const f32, v: *const f32, g: *const f32, beta: *const f32, state: *mut f32, output: *mut f32,
        seq_len: i32, v_dim: i32,
    ) {
        gdr_warp::<64>(q, k, v, g, beta, state, output, seq_len, v_dim)
    }
    #[kernel]
    pub unsafe fn _Z39gated_delta_rule_recurrence_kernel_warpILi128ELi4EEvPKfS1_S1_S1_S1_PfS2_ii(
        q: *const f32, k: *const f32, v: *const f32, g: *const f32, beta: *const f32, state: *mut f32, output: *mut f32,
        seq_len: i32, v_dim: i32,
    ) {
        gdr_warp::<128>(q, k, v, g, beta, state, output, seq_len, v_dim)
    }

    /// chunked_gated_delta_rule_kernel<64, BK, 64>.
    #[inline(always)]
    unsafe fn gdr_chunked<const BK: usize>(
        q: *const f32, k: *const f32, v: *const f32, g: *const f32, beta: *const f32, state: *mut f32, output: *mut f32,
        seq_len: i32, v_dim: i32,
    ) {
        const BT: i32 = 64;
        const BV: i32 = 64;
        let bk = BK as i32;
        let v_tile = bid_x();
        let bh = bid_y();
        let tid = tid_x();
        let v_idx = v_tile.wrapping_mul(BV).wrapping_add(tid);
        if v_idx >= v_dim {
            return;
        }
        let num_chunks = seq_len.wrapping_add(BT - 1) / BT;
        let q_bh = off(q, bh.wrapping_mul(seq_len).wrapping_mul(bk));
        let k_bh = off(k, bh.wrapping_mul(seq_len).wrapping_mul(bk));
        let v_bh = off(v, bh.wrapping_mul(seq_len).wrapping_mul(v_dim));
        let g_bh = off(g, bh.wrapping_mul(seq_len));
        let beta_bh = off(beta, bh.wrapping_mul(seq_len));
        let state_bh = offm(state, bh.wrapping_mul(bk).wrapping_mul(v_dim));
        let out_bh = offm(output, bh.wrapping_mul(seq_len).wrapping_mul(v_dim));

        let smem = DynamicSharedArray::<f32>::get();
        let k_chunk = smem;
        let kk_dot = smem.wrapping_offset((BT * bk) as isize);
        let gcum = smem.wrapping_offset((BT * bk + BT * BT) as isize);
        let beta_s = gcum.wrapping_offset(BT as isize);
        let q_buf = beta_s.wrapping_offset(BT as isize);

        let mut s = [0f32; BK];
        let mut j = 0;
        while j < BK {
            s[j] = *off(state_bh, (j as i32).wrapping_mul(v_dim).wrapping_add(v_idx));
            j += 1;
        }
        let mut delta = [0f32; 64];
        let mut c = 0;
        while c < num_chunks {
            let chunk_start = c * BT;
            let chunk_len = if BT < seq_len - chunk_start { BT } else { seq_len - chunk_start };
            // phase 1
            let mut t = 0;
            while t < chunk_len {
                let mut j = tid;
                while j < bk {
                    *k_chunk.offset((t * bk + j) as isize) = *off(k_bh, (chunk_start + t).wrapping_mul(bk).wrapping_add(j));
                    j += BV;
                }
                t += 1;
            }
            if tid < chunk_len {
                *beta_s.offset(tid as isize) = *off(beta_bh, chunk_start + tid);
                *gcum.offset(tid as isize) = *off(g_bh, chunk_start + tid);
            }
            thread::sync_threads();
            // phase 1b: Hillis-Steele prefix sum
            let mut stride = 1;
            while stride < BT {
                let mut prev = 0f32;
                if tid < chunk_len && tid >= stride {
                    prev = *gcum.offset((tid - stride) as isize);
                }
                thread::sync_threads();
                if tid < chunk_len && tid >= stride {
                    *gcum.offset(tid as isize) = addf(*gcum.offset(tid as isize), prev);
                }
                thread::sync_threads();
                stride <<= 1;
            }
            // phase 2
            let mut idx = tid;
            while idx < chunk_len * chunk_len {
                let i = idx / chunk_len;
                let j = idx % chunk_len;
                if j < i {
                    let mut dot = 0f32;
                    let mut d = 0;
                    while d < bk {
                        dot = fmaf(*k_chunk.offset((i * bk + d) as isize), *k_chunk.offset((j * bk + d) as isize), dot);
                        d += 1;
                    }
                    *kk_dot.offset((i * BT + j) as isize) = dot;
                }
                idx += BV;
            }
            thread::sync_threads();
            // phase 3: forward substitution
            let mut i = 0;
            while i < chunk_len {
                let v_i = *off(v_bh, (chunk_start + i).wrapping_mul(v_dim).wrapping_add(v_idx));
                let gi = *gcum.offset(i as isize);
                let decay_i = expf(gi);
                let beta_i = *beta_s.offset(i as isize);
                let mut kv_mem = 0f32;
                let mut d = 0;
                while d < BK {
                    kv_mem = fmaf(mulf(s[d], decay_i), *k_chunk.offset((i * bk + d as i32) as isize), kv_mem);
                    d += 1;
                }
                // nvcc contracts `beta_i * (v_i - kv_mem)` into the first subtraction:
                // rhs = fma(v_i - kv_mem, beta_i, -(delta[0] * a_i0)), then fma(-delta[j], a_ij, rhs).
                let dv = subf(v_i, kv_mem);
                let rhs = if i == 0 {
                    mulf(beta_i, dv)
                } else {
                    let a0 = mulf(mulf(beta_i, *kk_dot.offset((i * BT) as isize)), expf(subf(gi, *gcum)));
                    let mut rhs = fmaf(dv, beta_i, -mulf(delta[0], a0));
                    let mut j = 1;
                    while j < i {
                        let a_ij = mulf(mulf(beta_i, *kk_dot.offset((i * BT + j) as isize)), expf(subf(gi, *gcum.offset(j as isize))));
                        rhs = fmaf(-delta[j as usize], a_ij, rhs);
                        j += 1;
                    }
                    rhs
                };
                delta[i as usize] = rhs;
                i += 1;
            }
            // phase 4: output
            let mut i = 0;
            while i < chunk_len {
                let mut j = tid;
                while j < bk {
                    *q_buf.offset(j as isize) = *off(q_bh, (chunk_start + i).wrapping_mul(bk).wrapping_add(j));
                    j += BV;
                }
                thread::sync_threads();
                let gi = *gcum.offset(i as isize);
                let decay_i = expf(gi);
                let mut o_val = 0f32;
                let mut d = 0;
                while d < BK {
                    o_val = fmaf(*q_buf.offset(d as isize), mulf(s[d], decay_i), o_val);
                    d += 1;
                }
                let mut j = 0;
                while j <= i {
                    let mut qk = 0f32;
                    let mut d = 0;
                    while d < bk {
                        qk = fmaf(*q_buf.offset(d as isize), *k_chunk.offset((j * bk + d) as isize), qk);
                        d += 1;
                    }
                    o_val = fmaf(mulf(qk, delta[j as usize]), expf(subf(gi, *gcum.offset(j as isize))), o_val);
                    j += 1;
                }
                *offm(out_bh, (chunk_start + i).wrapping_mul(v_dim).wrapping_add(v_idx)) = o_val;
                thread::sync_threads();
                i += 1;
            }
            // phase 5: state update
            let g_total = *gcum.offset((chunk_len - 1) as isize);
            let mut d = 0;
            while d < BK {
                // `s[d] * expf(g_total)` is contracted into the first addition.
                let term0 = mulf(mulf(*k_chunk.offset(d as isize), delta[0]), expf(subf(g_total, *gcum)));
                let mut s_new = fmaf(s[d], expf(g_total), term0);
                let mut t = 1;
                while t < chunk_len {
                    s_new = fmaf(
                        mulf(*k_chunk.offset((t * bk + d as i32) as isize), delta[t as usize]),
                        expf(subf(g_total, *gcum.offset(t as isize))),
                        s_new,
                    );
                    t += 1;
                }
                s[d] = s_new;
                d += 1;
            }
            thread::sync_threads();
            c += 1;
        }
        let mut j = 0;
        while j < BK {
            *offm(state_bh, (j as i32).wrapping_mul(v_dim).wrapping_add(v_idx)) = s[j];
            j += 1;
        }
    }

    #[kernel]
    pub unsafe fn _Z31chunked_gated_delta_rule_kernelILi64ELi64ELi64EEvPKfS1_S1_S1_S1_PfS2_ii(
        q: *const f32, k: *const f32, v: *const f32, g: *const f32, beta: *const f32, state: *mut f32, output: *mut f32,
        seq_len: i32, v_dim: i32,
    ) {
        gdr_chunked::<64>(q, k, v, g, beta, state, output, seq_len, v_dim)
    }
    #[kernel]
    pub unsafe fn _Z31chunked_gated_delta_rule_kernelILi64ELi128ELi64EEvPKfS1_S1_S1_S1_PfS2_ii(
        q: *const f32, k: *const f32, v: *const f32, g: *const f32, beta: *const f32, state: *mut f32, output: *mut f32,
        seq_len: i32, v_dim: i32,
    ) {
        gdr_chunked::<128>(q, k, v, g, beta, state, output, seq_len, v_dim)
    }

    /// sigmoid-weighted `acc * (1 / (1 + expf(-acc)))`.
    #[inline(always)]
    fn silu_rcp(acc: f32) -> f32 {
        mulf(acc, rcp(addf(ex2(mulf(acc, -LOG2E)), 1.0)))
    }

    #[inline(always)]
    unsafe fn conv_update<T: Half>(x: *const T, weight: *const T, conv_state: *mut T, output: *mut T, batch_size: i32, conv_dim: i32, kernel_size: i32) {
        let ch = bid_x().wrapping_mul(bdim_x()).wrapping_add(tid_x());
        let b = bid_y();
        if ch >= conv_dim || b >= batch_size {
            return;
        }
        let cs = offm(conv_state, b.wrapping_mul(conv_dim).wrapping_add(ch).wrapping_mul(kernel_size));
        let w = off(weight, ch.wrapping_mul(kernel_size));
        let mut i = 0;
        while i < kernel_size - 1 {
            *offm(cs, i) = *offm(cs, i + 1);
            i += 1;
        }
        *offm(cs, kernel_size - 1) = *off(x, b.wrapping_mul(conv_dim).wrapping_add(ch));
        let mut acc = 0f32;
        let mut i = 0;
        while i < kernel_size {
            acc = fmaf((*offm(cs, i)).f(), (*off(w, i)).f(), acc);
            i += 1;
        }
        *offm(output, b.wrapping_mul(conv_dim).wrapping_add(ch)) = T::t(silu_rcp(acc));
    }
    #[kernel]
    pub unsafe fn _Z27causal_conv1d_update_kernelI13__nv_bfloat16EvPKT_S3_PS1_S4_iii(
        x: *const B, weight: *const B, conv_state: *mut B, output: *mut B, batch_size: i32, conv_dim: i32, kernel_size: i32,
    ) {
        conv_update::<B>(x, weight, conv_state, output, batch_size, conv_dim, kernel_size)
    }
    #[kernel]
    pub unsafe fn _Z27causal_conv1d_update_kernelI6__halfEvPKT_S3_PS1_S4_iii(
        x: *const H, weight: *const H, conv_state: *mut H, output: *mut H, batch_size: i32, conv_dim: i32, kernel_size: i32,
    ) {
        conv_update::<H>(x, weight, conv_state, output, batch_size, conv_dim, kernel_size)
    }

    #[inline(always)]
    unsafe fn conv_full<T: Half>(x: *const T, weight: *const T, output: *mut T, batch_size: i32, conv_dim: i32, seq_len: i32, kernel_size: i32) {
        let ch = bid_x().wrapping_mul(bdim_x()).wrapping_add(tid_x());
        let pos = bid_y();
        let b = thread::blockIdx_z() as i32;
        if ch >= conv_dim || pos >= seq_len || b >= batch_size {
            return;
        }
        let x_bch = off(x, b.wrapping_mul(conv_dim).wrapping_add(ch).wrapping_mul(seq_len));
        let w = off(weight, ch.wrapping_mul(kernel_size));
        let mut acc = 0f32;
        let mut i = 0;
        while i < kernel_size {
            let src_pos = pos - (kernel_size - 1) + i;
            let xv = if src_pos >= 0 { (*off(x_bch, src_pos)).f() } else { 0.0 };
            acc = fmaf(xv, (*off(w, i)).f(), acc);
            i += 1;
        }
        *offm(output, b.wrapping_mul(conv_dim).wrapping_add(ch).wrapping_mul(seq_len).wrapping_add(pos)) = T::t(silu_rcp(acc));
    }
    #[kernel]
    pub unsafe fn _Z25causal_conv1d_full_kernelI13__nv_bfloat16EvPKT_S3_PS1_iiii(
        x: *const B, weight: *const B, output: *mut B, batch_size: i32, conv_dim: i32, seq_len: i32, kernel_size: i32,
    ) {
        conv_full::<B>(x, weight, output, batch_size, conv_dim, seq_len, kernel_size)
    }
    #[kernel]
    pub unsafe fn _Z25causal_conv1d_full_kernelI6__halfEvPKT_S3_PS1_iiii(
        x: *const H, weight: *const H, output: *mut H, batch_size: i32, conv_dim: i32, seq_len: i32, kernel_size: i32,
    ) {
        conv_full::<H>(x, weight, output, batch_size, conv_dim, seq_len, kernel_size)
    }

    #[inline(always)]
    unsafe fn save_conv_state<T: Half>(x: *const T, cs_out: *mut T, batch_size: i32, conv_dim: i32, seq_len: i32, kernel_size: i32) {
        let ch = bid_x().wrapping_mul(bdim_x()).wrapping_add(tid_x());
        let b = bid_y();
        if ch >= conv_dim || b >= batch_size {
            return;
        }
        let x_bch = off(x, b.wrapping_mul(conv_dim).wrapping_add(ch).wrapping_mul(seq_len));
        let cs = offm(cs_out, b.wrapping_mul(conv_dim).wrapping_add(ch).wrapping_mul(kernel_size));
        let pad = kernel_size - seq_len;
        let mut i = 0;
        while i < kernel_size {
            *offm(cs, i) = if i < pad { T::zero() } else { *off(x_bch, seq_len - kernel_size + i) };
            i += 1;
        }
    }
    #[kernel]
    pub unsafe fn _Z22save_conv_state_kernelI13__nv_bfloat16EvPKT_PS1_iiii(x: *const B, cs: *mut B, bs: i32, cd: i32, sl: i32, ks: i32) {
        save_conv_state::<B>(x, cs, bs, cd, sl, ks)
    }
    #[kernel]
    pub unsafe fn _Z22save_conv_state_kernelI6__halfEvPKT_PS1_iiii(x: *const H, cs: *mut H, bs: i32, cd: i32, sl: i32, ks: i32) {
        save_conv_state::<H>(x, cs, bs, cd, sl, ks)
    }

    /// softplus of gdn prepare/decode: a > 20 ? a : (a > 0 ? a + log1pf(expf(-a)) : log1pf(expf(a))).
    #[inline(always)]
    fn gdn_softplus(a_val: f32) -> f32 {
        if gt(a_val, 20.0) {
            a_val
        } else if gt(a_val, 0.0) {
            addf(a_val, log1p_ftz(ex2(mulf(a_val, -LOG2E))))
        } else {
            log1p_ftz(expf(a_val))
        }
    }

    #[inline(always)]
    unsafe fn gdn_prepare<T: Half>(
        mixed_qkv: *const T, b: *const T, a: *const T, a_log: *const f32, dt_bias: *const f32, q_out: *mut f32, k_out: *mut f32,
        v_out: *mut f32, g_out: *mut f32, beta_out: *mut f32, batch_size: i32, seq_len: i32, num_k_heads: i32, num_v_heads: i32,
        head_k_dim: i32, head_v_dim: i32,
    ) {
        static mut RED_Q: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut RED_K: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut MUL: SharedArray<f32, 2> = SharedArray::UNINIT;
        let red_q = SharedArray::as_raw_mut_ptr(&raw mut RED_Q);
        let red_k = SharedArray::as_raw_mut_ptr(&raw mut RED_K);
        let mul = SharedArray::as_raw_mut_ptr(&raw mut MUL);
        let token_head = bid_x();
        let hv = token_head % num_v_heads;
        let token = token_head / num_v_heads;
        let t = token % seq_len;
        let bidx = token / seq_len;
        let tid = tid_x();
        if bidx >= batch_size {
            return;
        }
        let v_per_group = num_v_heads / num_k_heads;
        let hk = hv / v_per_group;
        let key_dim = num_k_heads.wrapping_mul(head_k_dim);
        let value_dim = num_v_heads.wrapping_mul(head_v_dim);
        let conv_dim = key_dim.wrapping_mul(2).wrapping_add(value_dim);
        let bh = bidx.wrapping_mul(num_v_heads).wrapping_add(hv);
        let tok = bidx.wrapping_mul(seq_len).wrapping_add(t);
        let row = off(mixed_qkv, tok.wrapping_mul(conv_dim));
        let b_row = off(b, tok.wrapping_mul(num_v_heads));
        let a_row = off(a, tok.wrapping_mul(num_v_heads));
        let bd = bdim_x();
        let mut q_sum = 0f32;
        let mut k_sum = 0f32;
        let mut d = tid;
        while d < head_k_dim {
            let qv = (*off(row, hk.wrapping_mul(head_k_dim).wrapping_add(d))).f();
            let kv = (*off(row, key_dim.wrapping_add(hk.wrapping_mul(head_k_dim)).wrapping_add(d))).f();
            q_sum = fmaf(qv, qv, q_sum);
            k_sum = fmaf(kv, kv, k_sum);
            d = d.wrapping_add(bd);
        }
        *red_q.offset(tid as isize) = q_sum;
        *red_k.offset(tid as isize) = k_sum;
        thread::sync_threads();
        let mut stride = (thread::blockDim_x() >> 1) as i32;
        while stride > 0 {
            if tid < stride {
                *red_q.offset(tid as isize) = addf(*red_q.offset((tid + stride) as isize), *red_q.offset(tid as isize));
                *red_k.offset(tid as isize) = addf(*red_k.offset((tid + stride) as isize), *red_k.offset(tid as isize));
            }
            thread::sync_threads();
            stride >>= 1;
        }
        if tid == 0 {
            *mul = mulf(rsqrt(addf(*red_q, 1.0e-6)), rsqrt(head_k_dim as f32));
            *mul.offset(1) = rsqrt(addf(*red_k, 1.0e-6));
            let b_val = (*off(b_row, hv)).f();
            let a_val = addf((*off(a_row, hv)).f(), *off(dt_bias, hv));
            let sp = gdn_softplus(a_val);
            *offm(beta_out, bh.wrapping_mul(seq_len).wrapping_add(t)) = rcp(addf(ex2(mulf(b_val, -LOG2E)), 1.0));
            *offm(g_out, bh.wrapping_mul(seq_len).wrapping_add(t)) = mulf(expf(*off(a_log, hv)), -sp);
        }
        thread::sync_threads();
        let q_mul = *mul;
        let k_mul = *mul.offset(1);
        let q_dst = offm(q_out, bh.wrapping_mul(seq_len).wrapping_add(t).wrapping_mul(head_k_dim));
        let k_dst = offm(k_out, bh.wrapping_mul(seq_len).wrapping_add(t).wrapping_mul(head_k_dim));
        let v_dst = offm(v_out, bh.wrapping_mul(seq_len).wrapping_add(t).wrapping_mul(head_v_dim));
        let mut d = tid;
        while d < head_k_dim {
            let qv = (*off(row, hk.wrapping_mul(head_k_dim).wrapping_add(d))).f();
            let kv = (*off(row, key_dim.wrapping_add(hk.wrapping_mul(head_k_dim)).wrapping_add(d))).f();
            *offm(q_dst, d) = mulf(qv, q_mul);
            *offm(k_dst, d) = mulf(kv, k_mul);
            d = d.wrapping_add(bd);
        }
        let mut d = tid;
        while d < head_v_dim {
            *offm(v_dst, d) = (*off(row, key_dim.wrapping_mul(2).wrapping_add(hv.wrapping_mul(head_v_dim)).wrapping_add(d))).f();
            d = d.wrapping_add(bd);
        }
    }
    #[kernel]
    pub unsafe fn _Z29gdn_prepare_recurrence_kernelI13__nv_bfloat16EvPKT_S3_S3_PKfS5_PfS6_S6_S6_S6_iiiiii(
        mixed_qkv: *const B, b: *const B, a: *const B, a_log: *const f32, dt_bias: *const f32, q_out: *mut f32, k_out: *mut f32,
        v_out: *mut f32, g_out: *mut f32, beta_out: *mut f32, batch_size: i32, seq_len: i32, num_k_heads: i32, num_v_heads: i32,
        head_k_dim: i32, head_v_dim: i32,
    ) {
        gdn_prepare::<B>(mixed_qkv, b, a, a_log, dt_bias, q_out, k_out, v_out, g_out, beta_out, batch_size, seq_len, num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    }
    #[kernel]
    pub unsafe fn _Z29gdn_prepare_recurrence_kernelI6__halfEvPKT_S3_S3_PKfS5_PfS6_S6_S6_S6_iiiiii(
        mixed_qkv: *const H, b: *const H, a: *const H, a_log: *const f32, dt_bias: *const f32, q_out: *mut f32, k_out: *mut f32,
        v_out: *mut f32, g_out: *mut f32, beta_out: *mut f32, batch_size: i32, seq_len: i32, num_k_heads: i32, num_v_heads: i32,
        head_k_dim: i32, head_v_dim: i32,
    ) {
        gdn_prepare::<H>(mixed_qkv, b, a, a_log, dt_bias, q_out, k_out, v_out, g_out, beta_out, batch_size, seq_len, num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    }

    /// gdn_decode_recurrence_kernel<T, BK, 64> (KD = BK) and _fallback<T, 64, 256> (KD = 0).
    #[inline(always)]
    unsafe fn gdn_decode<T: Half, const KD: usize>(
        mixed_qkv: *const T, b: *const T, a: *const T, a_log: *const f32, dt_bias: *const f32, state: *mut f32, output: *mut f32,
        batch_size: i32, num_k_heads: i32, num_v_heads: i32, head_k_dim_rt: i32, head_v_dim: i32,
    ) {
        const BV: i32 = 64;
        static mut RQ: SharedArray<f32, 64> = SharedArray::UNINIT;
        static mut RK: SharedArray<f32, 64> = SharedArray::UNINIT;
        static mut QB: SharedArray<f32, 128> = SharedArray::UNINIT;
        static mut KB: SharedArray<f32, 128> = SharedArray::UNINIT;
        static mut SC: SharedArray<f32, 4> = SharedArray::UNINIT;
        let kd: i32 = if KD == 0 { head_k_dim_rt } else { KD as i32 };
        let (red_q, red_k, q_buf, k_buf) = if KD == 0 {
            let sh = DynamicSharedArray::<f32>::get();
            let rk = sh.wrapping_offset(BV as isize);
            let qb = rk.wrapping_offset(BV as isize);
            (sh, rk, qb, qb.wrapping_offset(kd as isize))
        } else {
            (
                SharedArray::as_raw_mut_ptr(&raw mut RQ),
                SharedArray::as_raw_mut_ptr(&raw mut RK),
                SharedArray::as_raw_mut_ptr(&raw mut QB),
                SharedArray::as_raw_mut_ptr(&raw mut KB),
            )
        };
        let sc = SharedArray::as_raw_mut_ptr(&raw mut SC); // beta_t, decay_t, q_mul, k_mul
        let v_tile = bid_x();
        let bh = bid_y();
        let tid = tid_x();
        let v_idx = v_tile.wrapping_mul(BV).wrapping_add(tid);
        let bidx = bh / num_v_heads;
        let hv = bh.wrapping_sub(bidx.wrapping_mul(num_v_heads));
        if bidx >= batch_size {
            return;
        }
        let v_per_group = num_v_heads / num_k_heads;
        let hk = hv / v_per_group;
        let key_dim = num_k_heads.wrapping_mul(kd);
        let value_dim = num_v_heads.wrapping_mul(head_v_dim);
        let conv_dim = key_dim.wrapping_mul(2).wrapping_add(value_dim);
        let row = off(mixed_qkv, bidx.wrapping_mul(conv_dim));
        let b_row = off(b, bidx.wrapping_mul(num_v_heads));
        let a_row = off(a, bidx.wrapping_mul(num_v_heads));
        let state_bh = offm(state, bh.wrapping_mul(kd).wrapping_mul(head_v_dim));
        let out_bh = offm(output, bh.wrapping_mul(head_v_dim));

        let mut q_sum = 0f32;
        let mut k_sum = 0f32;
        let mut d = tid;
        while d < kd {
            let qv = (*off(row, hk.wrapping_mul(kd).wrapping_add(d))).f();
            let kv = (*off(row, key_dim.wrapping_add(hk.wrapping_mul(kd)).wrapping_add(d))).f();
            q_sum = fmaf(qv, qv, q_sum);
            k_sum = fmaf(kv, kv, k_sum);
            d += BV;
        }
        *red_q.offset(tid as isize) = q_sum;
        *red_k.offset(tid as isize) = k_sum;
        thread::sync_threads();
        let mut stride = BV >> 1;
        while stride > 0 {
            if tid < stride {
                *red_q.offset(tid as isize) = addf(*red_q.offset((tid + stride) as isize), *red_q.offset(tid as isize));
                *red_k.offset(tid as isize) = addf(*red_k.offset((tid + stride) as isize), *red_k.offset(tid as isize));
            }
            thread::sync_threads();
            stride >>= 1;
        }
        if tid == 0 {
            *sc.offset(2) = mulf(rsqrt(addf(*red_q, 1.0e-6)), rsqrt(kd as f32));
            *sc.offset(3) = rsqrt(addf(*red_k, 1.0e-6));
            let b_val = (*off(b_row, hv)).f();
            let a_val = addf((*off(a_row, hv)).f(), *off(dt_bias, hv));
            let sp = gdn_softplus(a_val);
            *sc = rcp(addf(ex2(mulf(b_val, -LOG2E)), 1.0));
            *sc.offset(1) = expf(mulf(expf(*off(a_log, hv)), -sp));
        }
        thread::sync_threads();
        let q_mul = *sc.offset(2);
        let k_mul = *sc.offset(3);
        let mut d = tid;
        while d < kd {
            *q_buf.offset(d as isize) = mulf((*off(row, hk.wrapping_mul(kd).wrapping_add(d))).f(), q_mul);
            *k_buf.offset(d as isize) = mulf((*off(row, key_dim.wrapping_add(hk.wrapping_mul(kd)).wrapping_add(d))).f(), k_mul);
            d += BV;
        }
        thread::sync_threads();
        if v_idx >= head_v_dim {
            return;
        }
        let beta_t = *sc;
        let decay_t = *sc.offset(1);
        if KD != 0 {
            // Compile-time KD: fully unrolled loops over a [f32; KD] indexed by constants, so the
            // state column lives in registers (as in the reference's `float s[BK]`).
            let mut s = [0f32; KD];
            let mut j = 0usize;
            while j < KD {
                cuda_device::thread::__unroll_config::<0>();
                s[j] = mulf(*offm(state_bh, (j as i32).wrapping_mul(head_v_dim).wrapping_add(v_idx)), decay_t);
                j += 1;
            }
            let v_t = (*off(row, key_dim.wrapping_mul(2).wrapping_add(hv.wrapping_mul(head_v_dim)).wrapping_add(v_idx))).f();
            let mut kv_mem = 0f32;
            let mut j = 0usize;
            while j < KD {
                cuda_device::thread::__unroll_config::<0>();
                kv_mem = fmaf(s[j], *k_buf.add(j), kv_mem);
                j += 1;
            }
            let delta = mulf(subf(v_t, kv_mem), beta_t);
            let mut y = 0f32;
            let mut j = 0usize;
            while j < KD {
                cuda_device::thread::__unroll_config::<0>();
                s[j] = fmaf(*k_buf.add(j), delta, s[j]);
                y = fmaf(s[j], *q_buf.add(j), y);
                j += 1;
            }
            let mut j = 0usize;
            while j < KD {
                cuda_device::thread::__unroll_config::<0>();
                *offm(state_bh, (j as i32).wrapping_mul(head_v_dim).wrapping_add(v_idx)) = s[j];
                j += 1;
            }
            *offm(out_bh, v_idx) = y;
            return;
        }
        let n = kd;
        let mut s = [0f32; 256];
        let mut j = 0;
        while j < n {
            s[j as usize] = mulf(*offm(state_bh, j.wrapping_mul(head_v_dim).wrapping_add(v_idx)), decay_t);
            j += 1;
        }
        let v_t = (*off(row, key_dim.wrapping_mul(2).wrapping_add(hv.wrapping_mul(head_v_dim)).wrapping_add(v_idx))).f();
        let mut kv_mem = 0f32;
        let mut j = 0;
        while j < n {
            kv_mem = fmaf(s[j as usize], *k_buf.offset(j as isize), kv_mem);
            j += 1;
        }
        let delta = mulf(subf(v_t, kv_mem), beta_t);
        let mut y = 0f32;
        let mut j = 0;
        while j < n {
            s[j as usize] = fmaf(*k_buf.offset(j as isize), delta, s[j as usize]);
            y = fmaf(s[j as usize], *q_buf.offset(j as isize), y);
            j += 1;
        }
        let mut j = 0;
        while j < n {
            *offm(state_bh, j.wrapping_mul(head_v_dim).wrapping_add(v_idx)) = s[j as usize];
            j += 1;
        }
        *offm(out_bh, v_idx) = y;
    }

    #[kernel]
    pub unsafe fn _Z28gdn_decode_recurrence_kernelI13__nv_bfloat16Li64ELi64EEvPKT_S3_S3_PKfS5_PfS6_iiii(
        m: *const B, b: *const B, a: *const B, a_log: *const f32, dt_bias: *const f32, state: *mut f32, output: *mut f32,
        batch_size: i32, nk: i32, nv: i32, hvd: i32,
    ) {
        gdn_decode::<B, 64>(m, b, a, a_log, dt_bias, state, output, batch_size, nk, nv, 64, hvd)
    }
    #[kernel]
    pub unsafe fn _Z28gdn_decode_recurrence_kernelI6__halfLi64ELi64EEvPKT_S3_S3_PKfS5_PfS6_iiii(
        m: *const H, b: *const H, a: *const H, a_log: *const f32, dt_bias: *const f32, state: *mut f32, output: *mut f32,
        batch_size: i32, nk: i32, nv: i32, hvd: i32,
    ) {
        gdn_decode::<H, 64>(m, b, a, a_log, dt_bias, state, output, batch_size, nk, nv, 64, hvd)
    }
    #[kernel]
    pub unsafe fn _Z28gdn_decode_recurrence_kernelI13__nv_bfloat16Li128ELi64EEvPKT_S3_S3_PKfS5_PfS6_iiii(
        m: *const B, b: *const B, a: *const B, a_log: *const f32, dt_bias: *const f32, state: *mut f32, output: *mut f32,
        batch_size: i32, nk: i32, nv: i32, hvd: i32,
    ) {
        gdn_decode::<B, 128>(m, b, a, a_log, dt_bias, state, output, batch_size, nk, nv, 128, hvd)
    }
    #[kernel]
    pub unsafe fn _Z28gdn_decode_recurrence_kernelI6__halfLi128ELi64EEvPKT_S3_S3_PKfS5_PfS6_iiii(
        m: *const H, b: *const H, a: *const H, a_log: *const f32, dt_bias: *const f32, state: *mut f32, output: *mut f32,
        batch_size: i32, nk: i32, nv: i32, hvd: i32,
    ) {
        gdn_decode::<H, 128>(m, b, a, a_log, dt_bias, state, output, batch_size, nk, nv, 128, hvd)
    }
    #[kernel]
    pub unsafe fn _Z37gdn_decode_recurrence_kernel_fallbackI13__nv_bfloat16Li64ELi256EEvPKT_S3_S3_PKfS5_PfS6_iiiii(
        m: *const B, b: *const B, a: *const B, a_log: *const f32, dt_bias: *const f32, state: *mut f32, output: *mut f32,
        batch_size: i32, nk: i32, nv: i32, hkd: i32, hvd: i32,
    ) {
        gdn_decode::<B, 0>(m, b, a, a_log, dt_bias, state, output, batch_size, nk, nv, hkd, hvd)
    }
    #[kernel]
    pub unsafe fn _Z37gdn_decode_recurrence_kernel_fallbackI6__halfLi64ELi256EEvPKT_S3_S3_PKfS5_PfS6_iiiii(
        m: *const H, b: *const H, a: *const H, a_log: *const f32, dt_bias: *const f32, state: *mut f32, output: *mut f32,
        batch_size: i32, nk: i32, nv: i32, hkd: i32, hvd: i32,
    ) {
        gdn_decode::<H, 0>(m, b, a, a_log, dt_bias, state, output, batch_size, nk, nv, hkd, hvd)
    }

    /// gdn_silu (NaN passthrough, +-inf -> max(0, x), else the two-sided form).
    #[inline(always)]
    fn gdn_silu(x: f32) -> f32 {
        if x != x {
            return x;
        }
        if x == f32::INFINITY || x == f32::NEG_INFINITY {
            return maxf(0.0, x);
        }
        if ge(x, 0.0) {
            return mulf(x, rcp(addf(ex2(mulf(x, -LOG2E)), 1.0)));
        }
        let ex = expf(x);
        mulf(mulf(x, ex), rcp(addf(ex, 1.0)))
    }

    #[inline(always)]
    unsafe fn rmsnorm_gated<T: Half>(x: *const T, gate: *const T, weight: *const T, output: *mut T, rows: i32, hidden_dim: i32, eps: f32) {
        static mut SM: SharedArray<f32, 256> = SharedArray::UNINIT;
        let smem = SharedArray::as_raw_mut_ptr(&raw mut SM);
        let row = bid_x();
        let tid = tid_x();
        if row >= rows {
            return;
        }
        let x_row = off(x, row.wrapping_mul(hidden_dim));
        let gate_row = off(gate, row.wrapping_mul(hidden_dim));
        let out_row = offm(output, row.wrapping_mul(hidden_dim));
        let bd = bdim_x();
        let mut sum = 0f32;
        let mut i = tid;
        while i < hidden_dim {
            let xv = (*off(x_row, i)).f();
            sum = fmaf(xv, xv, sum);
            i = i.wrapping_add(bd);
        }
        *smem.offset(tid as isize) = sum;
        thread::sync_threads();
        let mut stride = (thread::blockDim_x() >> 1) as i32;
        while stride > 0 {
            if tid < stride {
                *smem.offset(tid as isize) = addf(*smem.offset((tid + stride) as isize), *smem.offset(tid as isize));
            }
            thread::sync_threads();
            stride >>= 1;
        }
        let inv_rms = rsqrt(fmaf(*smem, rcp(hidden_dim as f32), eps));
        let mut i = tid;
        while i < hidden_dim {
            let gv = (*off(gate_row, i)).f();
            let o = mulf(mulf(mulf(inv_rms, (*off(x_row, i)).f()), (*off(weight, i)).f()), gdn_silu(gv));
            *offm(out_row, i) = T::t(o);
            i = i.wrapping_add(bd);
        }
    }
    #[kernel]
    pub unsafe fn _Z24gdn_rmsnorm_gated_kernelI13__nv_bfloat16EvPKT_S3_S3_PS1_iif(
        x: *const B, gate: *const B, weight: *const B, output: *mut B, rows: i32, hidden_dim: i32, eps: f32,
    ) {
        rmsnorm_gated::<B>(x, gate, weight, output, rows, hidden_dim, eps)
    }
    #[kernel]
    pub unsafe fn _Z24gdn_rmsnorm_gated_kernelI6__halfEvPKT_S3_S3_PS1_iif(
        x: *const H, gate: *const H, weight: *const H, output: *mut H, rows: i32, hidden_dim: i32, eps: f32,
    ) {
        rmsnorm_gated::<H>(x, gate, weight, output, rows, hidden_dim, eps)
    }

    #[inline(always)]
    unsafe fn gdn_gating<T: Half>(b: *const T, a: *const T, a_log: *const f32, dt_bias: *const f32, beta_out: *mut T, g_out: *mut T, total: i32, num_heads: i32) {
        let idx = bid_x().wrapping_mul(bdim_x()).wrapping_add(tid_x());
        if idx >= total {
            return;
        }
        let h = idx % num_heads;
        let b_val = (*off(b, idx)).f();
        let beta = rcp(addf(ex2(mulf(b_val, -LOG2E)), 1.0));
        let a_val = (*off(a, idx)).f();
        let sp_in = addf(a_val, *off(dt_bias, h));
        let l = lg2(addf(expf(sp_in), 1.0));
        let g_val = mulf(mulf(l, -LN2), expf(*off(a_log, h)));
        *offm(beta_out, idx) = T::t(beta);
        *offm(g_out, idx) = T::t(g_val);
    }
    #[kernel]
    pub unsafe fn _Z23fused_gdn_gating_kernelI13__nv_bfloat16EvPKT_S3_PKfS5_PS1_S6_ii(
        b: *const B, a: *const B, a_log: *const f32, dt_bias: *const f32, beta_out: *mut B, g_out: *mut B, total: i32, num_heads: i32,
    ) {
        gdn_gating::<B>(b, a, a_log, dt_bias, beta_out, g_out, total, num_heads)
    }
    #[kernel]
    pub unsafe fn _Z23fused_gdn_gating_kernelI6__halfEvPKT_S3_PKfS5_PS1_S6_ii(
        b: *const H, a: *const H, a_log: *const f32, dt_bias: *const f32, beta_out: *mut H, g_out: *mut H, total: i32, num_heads: i32,
    ) {
        gdn_gating::<H>(b, a, a_log, dt_bias, beta_out, g_out, total, num_heads)
    }

    // ==========================================================================================
    // ssm.cu
    // ==========================================================================================

    #[inline(always)]
    unsafe fn ssm_scan<const CF: usize>(
        x: *const f32, dt: *const f32, a: *const f32, bm: *const f32, cm: *const f32, dm: *const f32, dt_bias: *const f32,
        state: *mut f32, y: *mut f32, n_heads: i32, head_dim: i32, d_state: i32, seq_len: i32, dt_min: f32, dt_max: f32,
    ) {
        let batch_idx = bid_y();
        let warp_idx = bid_x();
        let lane = tid_x();
        let head_idx = warp_idx / head_dim;
        let head_off = warp_idx % head_dim;
        if head_idx >= n_heads {
            return;
        }
        let d_inner = n_heads.wrapping_mul(head_dim);
        let a_val = *off(a, head_idx);
        let d_val = *off(dm, head_idx);
        let dtb = *off(dt_bias, head_idx);
        let state_base = batch_idx
            .wrapping_mul(n_heads)
            .wrapping_mul(head_dim)
            .wrapping_mul(d_state)
            .wrapping_add(head_idx.wrapping_mul(head_dim).wrapping_mul(d_state))
            .wrapping_add(head_off.wrapping_mul(d_state));
        let mut rs = [0f32; CF];
        let mut j = 0;
        while j < CF {
            rs[j] = *offm(state, state_base.wrapping_add(lane.wrapping_mul(CF as i32)).wrapping_add(j as i32));
            j += 1;
        }
        let mut t = 0;
        while t < seq_len {
            let mut dtv = *off(dt, batch_idx.wrapping_mul(seq_len).wrapping_mul(n_heads).wrapping_add(t.wrapping_mul(n_heads)).wrapping_add(head_idx));
            dtv = addf(dtv, dtb);
            if !gt(dtv, 20.0) {
                dtv = logf(addf(expf(dtv), 1.0));
            }
            dtv = minf(maxf(dtv, dt_min), dt_max);
            let da = expf(mulf(a_val, dtv));
            let x_val = *off(
                x,
                batch_idx.wrapping_mul(seq_len).wrapping_mul(d_inner).wrapping_add(t.wrapping_mul(d_inner)).wrapping_add(head_idx.wrapping_mul(head_dim)).wrapping_add(head_off),
            );
            let x_dt = mulf(dtv, x_val);
            let mut sum = 0f32;
            let bc_base = batch_idx
                .wrapping_mul(seq_len)
                .wrapping_mul(n_heads)
                .wrapping_mul(d_state)
                .wrapping_add(t.wrapping_mul(n_heads).wrapping_mul(d_state))
                .wrapping_add(head_idx.wrapping_mul(d_state));
            let mut j = 0;
            while j < CF {
                let so = lane.wrapping_mul(CF as i32).wrapping_add(j as i32);
                let b_val = *off(bm, bc_base.wrapping_add(so));
                let c_val = *off(cm, bc_base.wrapping_add(so));
                // ptxas contracts differently per instance: CF=1 keeps b*x_dt rounded.
                rs[j] = if CF == 1 { fmaf(da, rs[j], mulf(x_dt, b_val)) } else { fmaf(x_dt, b_val, mulf(rs[j], da)) };
                sum = fmaf(c_val, rs[j], sum);
                j += 1;
            }
            let mut o = 16;
            while o > 0 {
                sum = addf(sum, warp::shuffle_xor_f32_sync(FULL, sum, o));
                o >>= 1;
            }
            if lane == 0 {
                *offm(
                    y,
                    batch_idx.wrapping_mul(seq_len).wrapping_mul(d_inner).wrapping_add(t.wrapping_mul(d_inner)).wrapping_add(head_idx.wrapping_mul(head_dim)).wrapping_add(head_off),
                ) = fmaf(d_val, x_val, sum);
            }
            t += 1;
        }
        let mut j = 0;
        while j < CF {
            *offm(state, state_base.wrapping_add(lane.wrapping_mul(CF as i32)).wrapping_add(j as i32)) = rs[j];
            j += 1;
        }
    }
    #[kernel]
    pub unsafe fn _Z15ssm_scan_kernelILi1EEvPKfS1_S1_S1_S1_S1_S1_PfS2_iiiiff(
        x: *const f32, dt: *const f32, a: *const f32, bm: *const f32, cm: *const f32, dm: *const f32, dt_bias: *const f32,
        state: *mut f32, y: *mut f32, n_heads: i32, head_dim: i32, d_state: i32, seq_len: i32, dt_min: f32, dt_max: f32,
    ) {
        ssm_scan::<1>(x, dt, a, bm, cm, dm, dt_bias, state, y, n_heads, head_dim, d_state, seq_len, dt_min, dt_max)
    }
    #[kernel]
    pub unsafe fn _Z15ssm_scan_kernelILi2EEvPKfS1_S1_S1_S1_S1_S1_PfS2_iiiiff(
        x: *const f32, dt: *const f32, a: *const f32, bm: *const f32, cm: *const f32, dm: *const f32, dt_bias: *const f32,
        state: *mut f32, y: *mut f32, n_heads: i32, head_dim: i32, d_state: i32, seq_len: i32, dt_min: f32, dt_max: f32,
    ) {
        ssm_scan::<2>(x, dt, a, bm, cm, dm, dt_bias, state, y, n_heads, head_dim, d_state, seq_len, dt_min, dt_max)
    }
    #[kernel]
    pub unsafe fn _Z15ssm_scan_kernelILi4EEvPKfS1_S1_S1_S1_S1_S1_PfS2_iiiiff(
        x: *const f32, dt: *const f32, a: *const f32, bm: *const f32, cm: *const f32, dm: *const f32, dt_bias: *const f32,
        state: *mut f32, y: *mut f32, n_heads: i32, head_dim: i32, d_state: i32, seq_len: i32, dt_min: f32, dt_max: f32,
    ) {
        ssm_scan::<4>(x, dt, a, bm, cm, dm, dt_bias, state, y, n_heads, head_dim, d_state, seq_len, dt_min, dt_max)
    }
    #[kernel]
    pub unsafe fn _Z15ssm_scan_kernelILi8EEvPKfS1_S1_S1_S1_S1_S1_PfS2_iiiiff(
        x: *const f32, dt: *const f32, a: *const f32, bm: *const f32, cm: *const f32, dm: *const f32, dt_bias: *const f32,
        state: *mut f32, y: *mut f32, n_heads: i32, head_dim: i32, d_state: i32, seq_len: i32, dt_min: f32, dt_max: f32,
    ) {
        ssm_scan::<8>(x, dt, a, bm, cm, dm, dt_bias, state, y, n_heads, head_dim, d_state, seq_len, dt_min, dt_max)
    }

    // ==========================================================================================
    // moe_utils.h / moe_gemv.cu / moe_gemm.cu / moe_gemm_wmma.cu
    // ==========================================================================================

    /// `vllm::from_float`: bf16 is `__float2bfloat16`; f16 is `static_cast<half>(float_to_half(v))`
    /// where float_to_half returns the *bits* as uint16_t, so the stored half is the value of those
    /// bits as an integer (SASS: F2FP.F16 then I2F.F16.U16). Reproduced as-is.
    #[inline(always)]
    fn vllm_from_float<const BF16: bool>(v: f32) -> u16 {
        if BF16 {
            convert::cvt_bf16x2_f32(v, 0.0) as u16
        } else {
            let bits = convert::cvt_f16x2_f32(v, 0.0) as u16;
            let r: u16;
            unsafe { ptx_asm!("cvt.rn.f16.u16 %0, %1;", out("=h") r, in("h") bits, options(register_only)) };
            r
        }
    }
    /// `vllm::to_float`: bf16 is `__bfloat162float`; f16 is `half_to_float(static_cast<uint16_t>(u))`,
    /// and `static_cast<uint16_t>(half)` converts the half's *value* (cvt.rzi.u16.f16, saturating,
    /// NaN -> 0) whose integer is then reinterpreted as half bits (SASS: F2I.U16.F16.TRUNC then
    /// HADD2.F32). Reproduced as-is.
    #[inline(always)]
    fn vllm_to_float<const BF16: bool>(b: u16) -> f32 {
        if BF16 {
            B(b).f()
        } else {
            let u: u16;
            unsafe { ptx_asm!("cvt.rzi.u16.f16 %0, %1;", out("=h") u, in("h") b, options(register_only)) };
            H(u).f()
        }
    }

    #[kernel]
    pub unsafe fn _Z30count_tokens_per_expert_kernelPKiPii(expert_ids: *const i32, counts: *mut i32, size_m: i32) {
        let i = bid_x().wrapping_mul(bdim_x()).wrapping_add(tid_x());
        if i < size_m {
            let e = *off(expert_ids, i);
            let p = offm(counts, e) as u64;
            ptx_asm!("red.global.add.s32 [%0], %1;", in("l") p, in("r") 1i32, clobber("memory"));
        }
    }

    /// Replaces thrust::inclusive_scan(counts -> offsets + 1) + `offsets[0] = 0` (cub DeviceScan
    /// kernels in the reference): one block of 1024 threads, any `n`, wrapping i32 (one correct answer).
    #[kernel]
    pub unsafe fn moe_expert_offsets_scan(counts: *const i32, offsets: *mut i32, n: i32) {
        static mut PART: SharedArray<i32, 1024> = SharedArray::UNINIT;
        let part = SharedArray::as_raw_mut_ptr(&raw mut PART);
        let tid = tid_x();
        let chunk = (n + 1023) / 1024;
        let start = tid * chunk;
        let end = if start + chunk < n { start + chunk } else { n };
        let mut s = 0i32;
        let mut i = start;
        while i < end {
            s = s.wrapping_add(*counts.add(i as usize));
            i += 1;
        }
        *part.add(tid as usize) = s;
        thread::sync_threads();
        let mut o = 1;
        while o < 1024 {
            let t = if tid >= o { *part.add((tid - o) as usize) } else { 0 };
            thread::sync_threads();
            if tid >= o {
                *part.add(tid as usize) = (*part.add(tid as usize)).wrapping_add(t);
            }
            thread::sync_threads();
            o <<= 1;
        }
        let mut run = if tid > 0 { *part.add(tid as usize - 1) } else { 0 };
        let mut i = start;
        while i < end {
            run = run.wrapping_add(*counts.add(i as usize));
            *offsets.add(i as usize + 1) = run;
            i += 1;
        }
        if tid == 0 {
            *offsets = 0;
        }
    }

    /// moe_gemv_kernel<T, 256>: one block per (row, token); 16-byte vector loads.
    #[inline(always)]
    unsafe fn moe_gemv_k<const BF16: bool>(
        input: *const u16, weights: *const u16, sorted: *const i32, expert_ids: *const i32, tw: *const f32, output: *mut u16,
        num_experts: i32, topk: i32, m: i32, n: i32, k: i32,
    ) {
        static mut SM: SharedArray<f32, 8> = SharedArray::UNINIT;
        let smem = SharedArray::as_raw_mut_ptr(&raw mut SM);
        let row = bid_x();
        let token_idx = bid_y();
        if token_idx >= m || row >= n {
            return;
        }
        let token_id = *off(sorted, token_idx);
        let expert = *off(expert_ids, token_idx);
        if expert < 0 || expert >= num_experts {
            return;
        }
        let input_idx = token_id / (if tw.is_null() { topk } else { 1 });
        let input_row = input.wrapping_add((input_idx as i64 as u64).wrapping_mul(k as i64 as u64) as usize);
        let weight_row = weights.wrapping_add(
            (expert as i64 as u64).wrapping_mul(n as i64 as u64).wrapping_mul(k as i64 as u64).wrapping_add((row as i64 as u64).wrapping_mul(k as i64 as u64)) as usize,
        );
        let tid = tid_x();
        let k_vec = k / 8;
        let in_vec = input_row as *const [u32; 4];
        let w_vec = weight_row as *const [u32; 4];
        let mut sum = 0f32;
        let mut kk = tid;
        while kk < k_vec {
            let iv = *in_vec.offset(kk as isize);
            let wv = *w_vec.offset(kk as isize);
            let mut i = 0;
            while i < 4 {
                if BF16 {
                    let (a, w) = (iv[i], wv[i]);
                    sum = fmaf(B(a as u16).f(), B(w as u16).f(), sum);
                    sum = fmaf(B((a >> 16) as u16).f(), B((w >> 16) as u16).f(), sum);
                } else {
                    let prod = f16x2::mul_f16x2(iv[i], wv[i]);
                    sum = addf(sum, addf(H(prod as u16).f(), H((prod >> 16) as u16).f()));
                }
                i += 1;
            }
            kk += 256;
        }
        let mut kk = k_vec * 8 + tid;
        while kk < k {
            sum = fmaf(vllm_to_float::<BF16>(*off(input_row, kk)), vllm_to_float::<BF16>(*off(weight_row, kk)), sum);
            kk += 256;
        }
        let mut o = 16;
        while o > 0 {
            sum = addf(sum, warp::shuffle_xor_f32_sync(FULL, sum, o));
            o >>= 1;
        }
        let warp_id = tid / 32;
        let lane = tid % 32;
        if lane == 0 {
            *smem.offset(warp_id as isize) = sum;
        }
        thread::sync_threads();
        if warp_id == 0 {
            sum = if lane < 8 { *smem.offset(lane as isize) } else { 0.0 };
            let mut o = 4;
            while o > 0 {
                sum = addf(sum, warp::shuffle_xor_f32_sync(FULL, sum, o));
                o >>= 1;
            }
            if lane == 0 {
                if !tw.is_null() {
                    sum = mulf(sum, *off(tw, token_id));
                }
                *output.wrapping_add((token_id as i64 as u64).wrapping_mul(n as i64 as u64).wrapping_add(row as i64 as u64) as usize) = vllm_from_float::<BF16>(sum);
            }
        }
    }
    #[kernel]
    pub unsafe fn _Z15moe_gemv_kernelI13__nv_bfloat16Li256EEvPKT_S3_PKiS5_PKfPS1_iiiii(
        input: *const u16, weights: *const u16, sorted: *const i32, expert_ids: *const i32, tw: *const f32, output: *mut u16,
        num_experts: i32, topk: i32, m: i32, n: i32, k: i32,
    ) {
        moe_gemv_k::<true>(input, weights, sorted, expert_ids, tw, output, num_experts, topk, m, n, k)
    }
    #[kernel]
    pub unsafe fn _Z15moe_gemv_kernelI6__halfLi256EEvPKT_S3_PKiS5_PKfPS1_iiiii(
        input: *const u16, weights: *const u16, sorted: *const i32, expert_ids: *const i32, tw: *const f32, output: *mut u16,
        num_experts: i32, topk: i32, m: i32, n: i32, k: i32,
    ) {
        moe_gemv_k::<false>(input, weights, sorted, expert_ids, tw, output, num_experts, topk, m, n, k)
    }

    /// moe_gemm_vectorized_kernel<T, T2, 64, 64, 8>: block (64, 8), grid (ceil(N/64), M).
    #[inline(always)]
    unsafe fn moe_gemm_vec<const BF16: bool>(
        input: *const u16, weights: *const u16, sorted: *const i32, expert_ids: *const i32, tw: *const f32, output: *mut u16,
        num_experts: i32, topk: i32, m: i32, n_dim: i32, k: i32,
    ) {
        static mut S_IN: SharedArray<[u32; 4], 8> = SharedArray::UNINIT;
        static mut S_W: SharedArray<[u32; 4], 512> = SharedArray::UNINIT;
        let s_in = SharedArray::as_raw_mut_ptr(&raw mut S_IN);
        let s_w = SharedArray::as_raw_mut_ptr(&raw mut S_W);
        let token_idx = bid_y();
        if token_idx >= m {
            return;
        }
        let n_tile_start = bid_x() * 64;
        let tid_n = tid_x();
        let tid_k = thread::threadIdx_y() as i32;
        let n = n_tile_start + tid_n;
        if n >= n_dim {
            return;
        }
        let token_id = *off(sorted, token_idx);
        let expert = *off(expert_ids, token_idx);
        if expert < 0 || expert >= num_experts {
            return;
        }
        let kk = k as i64 as u64;
        let input_row = input.wrapping_add(((token_id / (if tw.is_null() { topk } else { 1 })) as i64 as u64).wrapping_mul(kk) as usize);
        let weight_row = weights
            .wrapping_add((expert as i64 as u64).wrapping_mul(n_dim as i64 as u64).wrapping_mul(kk) as usize)
            .wrapping_add((n as i64 as u64).wrapping_mul(kk) as usize);
        let in_vec = input_row as *const [u32; 4];
        let w_vec = weight_row as *const [u32; 4];
        let mut acc: u32 = 0;
        let k_vec_dim = k / 8;
        let bdx = bdim_x();
        let loaders = bdx * thread::blockDim_y() as i32;
        let mut ks = 0;
        while ks < k_vec_dim {
            let mut i = tid_k * bdx + tid_n;
            while i < 8 {
                *s_in.offset(i as isize) = if ks + i < k_vec_dim { *in_vec.offset((ks + i) as isize) } else { [0; 4] };
                i += loaders;
            }
            let mut kv = tid_k;
            while kv < 8 {
                *s_w.offset((tid_n * 8 + kv) as isize) = if ks + kv < k_vec_dim { *w_vec.offset((ks + kv) as isize) } else { [0; 4] };
                kv += 8;
            }
            thread::sync_threads();
            let iv = s_in as *const u32;
            let wv = s_w.offset((tid_n * 8) as isize) as *const u32;
            let mut j = 0;
            while j < 32 {
                acc = if BF16 {
                    bf16x2::fma_bf16x2(*iv.offset(j), *wv.offset(j), acc)
                } else {
                    f16x2::fma_f16x2(*iv.offset(j), *wv.offset(j), acc)
                };
                j += 1;
            }
            ks += 8;
        }
        thread::sync_threads();
        let hsum = if BF16 { bf16x2::add_bf16x2(acc, acc >> 16) } else { f16x2::add_f16x2(acc, acc >> 16) } as u16;
        let o = output.wrapping_offset(token_id.wrapping_mul(n_dim).wrapping_add(n) as isize);
        if !tw.is_null() {
            *o = vllm_from_float::<BF16>(mulf(vllm_to_float::<BF16>(hsum), *off(tw, token_id)));
        } else {
            *o = hsum;
        }
    }
    #[kernel]
    pub unsafe fn _Z26moe_gemm_vectorized_kernelI13__nv_bfloat1614__nv_bfloat162Li64ELi64ELi8EEvPKT_S4_PKiS6_PKfPS2_iiiii(
        input: *const u16, weights: *const u16, sorted: *const i32, expert_ids: *const i32, tw: *const f32, output: *mut u16,
        num_experts: i32, topk: i32, m: i32, n: i32, k: i32,
    ) {
        moe_gemm_vec::<true>(input, weights, sorted, expert_ids, tw, output, num_experts, topk, m, n, k)
    }
    #[kernel]
    pub unsafe fn _Z26moe_gemm_vectorized_kernelI6__half7__half2Li64ELi64ELi8EEvPKT_S4_PKiS6_PKfPS2_iiiii(
        input: *const u16, weights: *const u16, sorted: *const i32, expert_ids: *const i32, tw: *const f32, output: *mut u16,
        num_experts: i32, topk: i32, m: i32, n: i32, k: i32,
    ) {
        moe_gemm_vec::<false>(input, weights, sorted, expert_ids, tw, output, num_experts, topk, m, n, k)
    }

    /// wmma m16n16k16 load.a (row) + load.b (col) + mma into the f32 accumulator `c`.
    #[inline(always)]
    pub unsafe fn wmma_step<const BF16: bool>(c: &mut [f32; 8], a_ptr: *const u16, b_ptr: *const u16, ldm: u32) {
        let (a0, a1, a2, a3, a4, a5, a6, a7): (u32, u32, u32, u32, u32, u32, u32, u32);
        let (b0, b1, b2, b3, b4, b5, b6, b7): (u32, u32, u32, u32, u32, u32, u32, u32);
        let ap = a_ptr as u64;
        let bp = b_ptr as u64;
        let (mut c0, mut c1, mut c2, mut c3, mut c4, mut c5, mut c6, mut c7) = (c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
        if !BF16 {
            ptx_asm!("wmma.load.a.sync.aligned.row.m16n16k16.f16 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;",
                out("=r") a0, out("=r") a1, out("=r") a2, out("=r") a3, out("=r") a4, out("=r") a5, out("=r") a6, out("=r") a7,
                in("l") ap, in("r") ldm, clobber("memory"));
            ptx_asm!("wmma.load.b.sync.aligned.col.m16n16k16.f16 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;",
                out("=r") b0, out("=r") b1, out("=r") b2, out("=r") b3, out("=r") b4, out("=r") b5, out("=r") b6, out("=r") b7,
                in("l") bp, in("r") ldm, clobber("memory"));
            ptx_asm!("wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19,%20,%21,%22,%23}, {%0,%1,%2,%3,%4,%5,%6,%7};",
                inout("+f") c0, inout("+f") c1, inout("+f") c2, inout("+f") c3, inout("+f") c4, inout("+f") c5, inout("+f") c6, inout("+f") c7,
                in("r") a0, in("r") a1, in("r") a2, in("r") a3, in("r") a4, in("r") a5, in("r") a6, in("r") a7,
                in("r") b0, in("r") b1, in("r") b2, in("r") b3, in("r") b4, in("r") b5, in("r") b6, in("r") b7);
        } else {
            ptx_asm!("wmma.load.a.sync.aligned.row.m16n16k16.bf16 {%0,%1,%2,%3}, [%4], %5;",
                out("=r") a0, out("=r") a1, out("=r") a2, out("=r") a3, in("l") ap, in("r") ldm, clobber("memory"));
            ptx_asm!("wmma.load.b.sync.aligned.col.m16n16k16.bf16 {%0,%1,%2,%3}, [%4], %5;",
                out("=r") b0, out("=r") b1, out("=r") b2, out("=r") b3, in("l") bp, in("r") ldm, clobber("memory"));
            ptx_asm!("wmma.mma.sync.aligned.row.col.m16n16k16.f32.bf16.bf16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, {%0,%1,%2,%3,%4,%5,%6,%7};",
                inout("+f") c0, inout("+f") c1, inout("+f") c2, inout("+f") c3, inout("+f") c4, inout("+f") c5, inout("+f") c6, inout("+f") c7,
                in("r") a0, in("r") a1, in("r") a2, in("r") a3,
                in("r") b0, in("r") b1, in("r") b2, in("r") b3);
        }
        *c = [c0, c1, c2, c3, c4, c5, c6, c7];
    }

    #[inline(always)]
    pub unsafe fn wmma_store(p: *mut f32, c: &[f32; 8], ldm: u32) {
        let pp = p as u64;
        ptx_asm!("wmma.store.d.sync.aligned.row.m16n16k16.f32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8}, %9;",
            in("l") pp, in("f") c[0], in("f") c[1], in("f") c[2], in("f") c[3], in("f") c[4], in("f") c[5], in("f") c[6], in("f") c[7],
            in("r") ldm, clobber("memory"));
    }

    #[inline(always)]
    pub unsafe fn copy16(dst: *mut u8, src: *const u8) {
        *(dst as *mut [u32; 4]) = *(src as *const [u32; 4]);
    }
    #[inline(always)]
    pub unsafe fn zero16(dst: *mut u8) {
        *(dst as *mut [u32; 4]) = [0; 4];
    }

    /// moe_gemm_grouped_kernel<T>: 2x2 warps of m16n16k16, block tile 32x32x16, block 128, grid
    /// (num_experts, ceil(n/32)), dynamic shared A 1024 + B 1024 + C 4096.
    #[inline(always)]
    unsafe fn grouped<const BF16: bool>(
        input: *const u16, weights: *const u16, sorted: *const i32, offsets: *const i32, tw: *const f32, output: *mut u16,
        num_experts: i32, topk: i32, size_m: i32, size_n: i32, size_k: i32,
    ) {
        let expert_id = bid_x();
        let n_tile = bid_y();
        if expert_id < 0 || expert_id >= num_experts {
            return;
        }
        let seg_start = *offsets.add(expert_id as usize);
        let seg_end = *offsets.add(expert_id as usize + 1);
        let rows = seg_end.wrapping_sub(seg_start);
        if rows == 0 {
            return;
        }
        let n_base = n_tile * 32;
        if n_base >= size_n {
            return;
        }
        let expert_w = weights.wrapping_add((expert_id as u64).wrapping_mul(size_n as i64 as u64).wrapping_mul(size_k as i64 as u64) as usize);
        let smem = DynamicSharedArray::<u8>::get_raw();
        let a_sh = smem as *mut u16;
        let b_sh = smem.add(1024) as *mut u16;
        let c_sh = smem.add(2048) as *mut f32;
        let tid = tid_x();
        let warp_id = tid / 32;
        let warp_m = warp_id / 2;
        let warp_n = warp_id % 2;
        let mut m_base = 0i32;
        while m_base < rows {
            let mut c = [0f32; 8];
            let mut k_base = 0i32;
            while k_base < size_k {
                let mut i = tid;
                while i < 64 {
                    let idx = i * 8;
                    let n_local = idx / 16;
                    let k_local = idx % 16;
                    let n_global = n_base + n_local;
                    let k_global = k_base + k_local;
                    let dst = b_sh.add((n_local * 16 + k_local) as usize) as *mut u8;
                    if n_global < size_n && k_global < size_k {
                        let o = (n_global as i64 as u64).wrapping_mul(size_k as i64 as u64).wrapping_add(k_global as i64 as u64);
                        copy16(dst, expert_w.wrapping_add(o as usize) as *const u8);
                    } else {
                        zero16(dst);
                    }
                    i += 128;
                }
                let mut i = tid;
                while i < 64 {
                    let idx = i * 8;
                    let m_local = idx / 16;
                    let k_local = idx % 16;
                    let m_seg = m_base + m_local;
                    let k_global = k_base + k_local;
                    let dst = a_sh.add((m_local * 16 + k_local) as usize) as *mut u8;
                    if m_seg < rows && k_global < size_k {
                        let token_index = *off(sorted, seg_start + m_seg);
                        let input_index = token_index / (if tw.is_null() { topk } else { 1 });
                        let o = (input_index as i64 as u64).wrapping_mul(size_k as i64 as u64).wrapping_add(k_global as i64 as u64);
                        copy16(dst, input.wrapping_add(o as usize) as *const u8);
                    } else {
                        zero16(dst);
                    }
                    i += 128;
                }
                thread::sync_threads();
                wmma_step::<BF16>(&mut c, a_sh.add((warp_m * 16 * 16) as usize), b_sh.add((warp_n * 16 * 16) as usize), 16);
                thread::sync_threads();
                k_base += 16;
            }
            wmma_store(c_sh.add((warp_m * 16 * 32 + warp_n * 16) as usize), &c, 32);
            thread::sync_threads();
            let mut i = tid;
            while i < 1024 {
                let m_local = i / 32;
                let n_local = i % 32;
                let m_seg = m_base + m_local;
                let n_global = n_base + n_local;
                if m_seg < rows && n_global < size_n {
                    let tp = seg_start + m_seg;
                    if tp < size_m {
                        let token_index = *off(sorted, tp);
                        let mut val = *c_sh.add((m_local * 32 + n_local) as usize);
                        if !tw.is_null() {
                            val = mulf(val, *off(tw, token_index));
                        }
                        let o = (token_index as i64 as u64).wrapping_mul(size_n as i64 as u64).wrapping_add(n_global as i64 as u64);
                        *output.wrapping_add(o as usize) = vllm_from_float::<BF16>(val);
                    }
                }
                i += 128;
            }
            m_base += 32;
        }
    }
    #[kernel]
    pub unsafe fn _Z23moe_gemm_grouped_kernelI13__nv_bfloat16EvPKT_S3_PKiS5_PKfPS1_iiiii(
        input: *const u16, weights: *const u16, sorted: *const i32, offsets: *const i32, tw: *const f32, output: *mut u16,
        num_experts: i32, topk: i32, size_m: i32, size_n: i32, size_k: i32,
    ) {
        grouped::<true>(input, weights, sorted, offsets, tw, output, num_experts, topk, size_m, size_n, size_k)
    }
    #[kernel]
    pub unsafe fn _Z23moe_gemm_grouped_kernelI6__halfEvPKT_S3_PKiS5_PKfPS1_iiiii(
        input: *const u16, weights: *const u16, sorted: *const i32, offsets: *const i32, tw: *const f32, output: *mut u16,
        num_experts: i32, topk: i32, size_m: i32, size_n: i32, size_k: i32,
    ) {
        grouped::<false>(input, weights, sorted, offsets, tw, output, num_experts, topk, size_m, size_n, size_k)
    }

    // ==========================================================================================
    // attention_prep.cu
    // ==========================================================================================

    pub trait Ap: Copy {
        fn af(self) -> f32;
        fn at(v: f32) -> Self;
    }
    impl Ap for f32 {
        #[inline(always)]
        fn af(self) -> f32 {
            self
        }
        #[inline(always)]
        fn at(v: f32) -> f32 {
            v
        }
    }
    impl Ap for H {
        #[inline(always)]
        fn af(self) -> f32 {
            Half::f(self)
        }
        #[inline(always)]
        fn at(v: f32) -> H {
            <H as Half>::t(v)
        }
    }
    impl Ap for B {
        #[inline(always)]
        fn af(self) -> f32 {
            Half::f(self)
        }
        #[inline(always)]
        fn at(v: f32) -> B {
            <B as Half>::t(v)
        }
    }

    /// Sum of squares over a (strided) row, tree-reduced in shared memory; returns inv_rms.
    #[inline(always)]
    unsafe fn ap_inv_rms<T: Ap>(src: *const T, src_base: i64, sd: i64, head_dim: i32, eps: f32, reduce: *mut f32) -> f32 {
        let tid = tid_x();
        let bd = bdim_x();
        let mut sum = 0f32;
        let mut col = tid;
        while col < head_dim {
            let v = (*src.wrapping_offset(src_base.wrapping_add((col as i64).wrapping_mul(sd)) as isize)).af();
            sum = fmaf(v, v, sum);
            col = col.wrapping_add(bd);
        }
        *reduce.offset(tid as isize) = sum;
        thread::sync_threads();
        let mut stride = bd / 2;
        while stride > 0 {
            if tid < stride {
                *reduce.offset(tid as isize) = addf(*reduce.offset(tid as isize), *reduce.offset((tid + stride) as isize));
            }
            thread::sync_threads();
            stride >>= 1;
        }
        rsqrt(fmaf(*reduce, rcp(head_dim as f32), eps))
    }

    #[inline(always)]
    unsafe fn write_norm_rope_row<T: Ap, const NEOX: bool>(
        src: *const T, weight: *const T, cos: *const T, sin: *const T, dst: *mut T, src_base: i64, sd: i64, head_dim: i32,
        rot_dim: i32, eps: f32, reduce: *mut f32,
    ) {
        let tid = tid_x();
        let bd = bdim_x();
        let inv_rms = ap_inv_rms(src, src_base, sd, head_dim, eps, reduce);
        let ld = |i: i32| (*src.wrapping_offset(src_base.wrapping_add((i as i64).wrapping_mul(sd)) as isize)).af();
        let mut r = tid;
        while r < rot_dim {
            let (xi, yi) = if NEOX { (r, rot_dim + r) } else { (2 * r, 2 * r + 1) };
            let x = mulf(mulf(inv_rms, ld(xi)), (*off(weight, xi)).af());
            let y = mulf(mulf(inv_rms, ld(yi)), (*off(weight, yi)).af());
            let c = (*off(cos, r)).af();
            let sn = (*off(sin, r)).af();
            *offm(dst, xi) = T::at(fmaf(x, c, -mulf(y, sn)));
            *offm(dst, yi) = T::at(fmaf(y, c, mulf(x, sn)));
            r = r.wrapping_add(bd);
        }
        let mut col = rot_dim.wrapping_mul(2).wrapping_add(tid);
        while col < head_dim {
            *offm(dst, col) = T::at(mulf(mulf(inv_rms, ld(col)), (*off(weight, col)).af()));
            col = col.wrapping_add(bd);
        }
    }

    #[inline(always)]
    unsafe fn write_norm_row<T: Ap>(src: *const T, weight: *const T, dst: *mut T, src_base: i64, sd: i64, head_dim: i32, eps: f32, reduce: *mut f32) {
        let tid = tid_x();
        let bd = bdim_x();
        let inv_rms = ap_inv_rms(src, src_base, sd, head_dim, eps, reduce);
        let mut col = tid;
        while col < head_dim {
            let v = (*src.wrapping_offset(src_base.wrapping_add((col as i64).wrapping_mul(sd)) as isize)).af();
            *offm(dst, col) = T::at(mulf(mulf(inv_rms, v), (*off(weight, col)).af()));
            col = col.wrapping_add(bd);
        }
    }

    /// qk_rms_norm_rope_kernel (POS = false, cos row from cos_batch_stride) and
    /// qk_rms_norm_rope_positions_kernel (POS = true).
    #[inline(always)]
    unsafe fn qk_norm_rope<T: Ap, const NEOX: bool, const POS: bool>(
        q: *const T, k: *const T, qw: *const T, kw: *const T, cos: *const T, sin: *const T, positions: *const u32, q_out: *mut T,
        k_out: *mut T, qsb: i64, qsh: i64, qss: i64, qsd: i64, ksb: i64, ksh: i64, kss: i64, ksd: i64, batch: i32, q_heads: i32,
        k_heads: i32, seq_len: i32, head_dim: i32, rot_dim: i32, cos_batch_stride: i32, q_eps: f32, k_eps: f32,
    ) {
        static mut RED: SharedArray<f32, 1024> = SharedArray::UNINIT;
        let reduce = SharedArray::as_raw_mut_ptr(&raw mut RED);
        let q_rows = batch.wrapping_mul(q_heads).wrapping_mul(seq_len);
        let row = bid_x();
        let is_q = row < q_rows;
        let local_row = if is_q { row } else { row.wrapping_sub(q_rows) };
        let heads = if is_q { q_heads } else { k_heads };
        let seq = local_row % seq_len;
        let tmp = local_row / seq_len;
        let head = tmp % heads;
        let batch_idx = tmp / heads;
        let cos_row: i64 = if POS {
            *off(positions, batch_idx.wrapping_mul(seq_len).wrapping_add(seq)) as i64
        } else if cos_batch_stride == 0 {
            seq as i64
        } else {
            batch_idx.wrapping_mul(cos_batch_stride).wrapping_add(seq) as i64
        };
        let cp = cos.wrapping_offset(cos_row.wrapping_mul(rot_dim as i64) as isize);
        let sp = sin.wrapping_offset(cos_row.wrapping_mul(rot_dim as i64) as isize);
        if is_q {
            let base = (batch_idx as i64).wrapping_mul(qsb).wrapping_add((head as i64).wrapping_mul(qsh)).wrapping_add((seq as i64).wrapping_mul(qss));
            let dst = q_out.wrapping_offset((row as i64).wrapping_mul(head_dim as i64) as isize);
            write_norm_rope_row::<T, NEOX>(q, qw, cp, sp, dst, base, qsd, head_dim, rot_dim, q_eps, reduce);
        } else {
            let base = (batch_idx as i64).wrapping_mul(ksb).wrapping_add((head as i64).wrapping_mul(ksh)).wrapping_add((seq as i64).wrapping_mul(kss));
            let dst = k_out.wrapping_offset((local_row as i64).wrapping_mul(head_dim as i64) as isize);
            write_norm_rope_row::<T, NEOX>(k, kw, cp, sp, dst, base, ksd, head_dim, rot_dim, k_eps, reduce);
        }
    }

    #[inline(always)]
    unsafe fn qkv_norm_rope<T: Ap, const NEOX: bool>(
        q: *const T, k: *const T, v: *const T, qw: *const T, kw: *const T, vw: *const T, cos: *const T, sin: *const T,
        positions: *const u32, q_out: *mut T, k_out: *mut T, v_out: *mut T, qsb: i64, qsh: i64, qss: i64, qsd: i64, ksb: i64,
        ksh: i64, kss: i64, ksd: i64, vsb: i64, vsh: i64, vss: i64, vsd: i64, batch: i32, q_heads: i32, k_heads: i32, seq_len: i32,
        head_dim: i32, rot_dim: i32, q_eps: f32, k_eps: f32, v_eps: f32,
    ) {
        static mut RED: SharedArray<f32, 1024> = SharedArray::UNINIT;
        let reduce = SharedArray::as_raw_mut_ptr(&raw mut RED);
        let q_rows = batch.wrapping_mul(q_heads).wrapping_mul(seq_len);
        let kv_rows = batch.wrapping_mul(k_heads).wrapping_mul(seq_len);
        let row = bid_x();
        if row < q_rows {
            let seq = row % seq_len;
            let tmp = row / seq_len;
            let head = tmp % q_heads;
            let batch_idx = tmp / q_heads;
            let pos = *off(positions, batch_idx.wrapping_mul(seq_len).wrapping_add(seq)) as i64;
            let cp = cos.wrapping_offset(pos.wrapping_mul(rot_dim as i64) as isize);
            let sp = sin.wrapping_offset(pos.wrapping_mul(rot_dim as i64) as isize);
            let base = (batch_idx as i64).wrapping_mul(qsb).wrapping_add((head as i64).wrapping_mul(qsh)).wrapping_add((seq as i64).wrapping_mul(qss));
            let dst = q_out.wrapping_offset((row as i64).wrapping_mul(head_dim as i64) as isize);
            write_norm_rope_row::<T, NEOX>(q, qw, cp, sp, dst, base, qsd, head_dim, rot_dim, q_eps, reduce);
            return;
        }
        let kv_row = row.wrapping_sub(q_rows);
        let is_k = kv_row < kv_rows;
        let local_row = if is_k { kv_row } else { kv_row.wrapping_sub(kv_rows) };
        let seq = local_row % seq_len;
        let tmp = local_row / seq_len;
        let head = tmp % k_heads;
        let batch_idx = tmp / k_heads;
        if is_k {
            let pos = *off(positions, batch_idx.wrapping_mul(seq_len).wrapping_add(seq)) as i64;
            let cp = cos.wrapping_offset(pos.wrapping_mul(rot_dim as i64) as isize);
            let sp = sin.wrapping_offset(pos.wrapping_mul(rot_dim as i64) as isize);
            let base = (batch_idx as i64).wrapping_mul(ksb).wrapping_add((head as i64).wrapping_mul(ksh)).wrapping_add((seq as i64).wrapping_mul(kss));
            let dst = k_out.wrapping_offset((local_row as i64).wrapping_mul(head_dim as i64) as isize);
            write_norm_rope_row::<T, NEOX>(k, kw, cp, sp, dst, base, ksd, head_dim, rot_dim, k_eps, reduce);
        } else {
            let base = (batch_idx as i64).wrapping_mul(vsb).wrapping_add((head as i64).wrapping_mul(vsh)).wrapping_add((seq as i64).wrapping_mul(vss));
            let dst = v_out.wrapping_offset((local_row as i64).wrapping_mul(head_dim as i64) as isize);
            write_norm_row::<T>(v, vw, dst, base, vsd, head_dim, v_eps, reduce);
        }
    }
    // GENERATED AP KERNELS BEGIN
    #[kernel]
    pub unsafe fn _Z23qk_rms_norm_rope_kernelIfLb0EEvPKT_S2_S2_S2_S2_S2_PS0_S3_lllllllliiiiiiiff(
        q: *const f32, k: *const f32, qw: *const f32, kw: *const f32, cos: *const f32, sin: *const f32, q_out: *mut f32, k_out: *mut f32,
        qsb: i64, qsh: i64, qss: i64, qsd: i64, ksb: i64, ksh: i64, kss: i64, ksd: i64, batch: i32, q_heads: i32, k_heads: i32,
        seq_len: i32, head_dim: i32, rot_dim: i32, cbs: i32, q_eps: f32, k_eps: f32,
    ) {
        qk_norm_rope::<f32, false, false>(q, k, qw, kw, cos, sin, core::ptr::null(), q_out, k_out, qsb, qsh, qss, qsd, ksb, ksh, kss, ksd, batch, q_heads, k_heads, seq_len, head_dim, rot_dim, cbs, q_eps, k_eps)
    }
    #[kernel]
    pub unsafe fn _Z33qk_rms_norm_rope_positions_kernelIfLb0EEvPKT_S2_S2_S2_S2_S2_PKjPS0_S5_lllllllliiiiiiff(
        q: *const f32, k: *const f32, qw: *const f32, kw: *const f32, cos: *const f32, sin: *const f32, positions: *const u32,
        q_out: *mut f32, k_out: *mut f32, qsb: i64, qsh: i64, qss: i64, qsd: i64, ksb: i64, ksh: i64, kss: i64, ksd: i64, batch: i32,
        q_heads: i32, k_heads: i32, seq_len: i32, head_dim: i32, rot_dim: i32, q_eps: f32, k_eps: f32,
    ) {
        qk_norm_rope::<f32, false, true>(q, k, qw, kw, cos, sin, positions, q_out, k_out, qsb, qsh, qss, qsd, ksb, ksh, kss, ksd, batch, q_heads, k_heads, seq_len, head_dim, rot_dim, 0, q_eps, k_eps)
    }
    #[kernel]
    pub unsafe fn _Z34qkv_rms_norm_rope_positions_kernelIfLb0EEvPKT_S2_S2_S2_S2_S2_S2_S2_PKjPS0_S5_S5_lllllllllllliiiiiifff(
        q: *const f32, k: *const f32, v: *const f32, qw: *const f32, kw: *const f32, vw: *const f32, cos: *const f32, sin: *const f32,
        positions: *const u32, q_out: *mut f32, k_out: *mut f32, v_out: *mut f32, qsb: i64, qsh: i64, qss: i64, qsd: i64, ksb: i64,
        ksh: i64, kss: i64, ksd: i64, vsb: i64, vsh: i64, vss: i64, vsd: i64, batch: i32, q_heads: i32, k_heads: i32, seq_len: i32,
        head_dim: i32, rot_dim: i32, q_eps: f32, k_eps: f32, v_eps: f32,
    ) {
        qkv_norm_rope::<f32, false>(q, k, v, qw, kw, vw, cos, sin, positions, q_out, k_out, v_out, qsb, qsh, qss, qsd, ksb, ksh, kss, ksd, vsb, vsh, vss, vsd, batch, q_heads, k_heads, seq_len, head_dim, rot_dim, q_eps, k_eps, v_eps)
    }
    #[kernel]
    pub unsafe fn _Z23qk_rms_norm_rope_kernelIfLb1EEvPKT_S2_S2_S2_S2_S2_PS0_S3_lllllllliiiiiiiff(
        q: *const f32, k: *const f32, qw: *const f32, kw: *const f32, cos: *const f32, sin: *const f32, q_out: *mut f32, k_out: *mut f32,
        qsb: i64, qsh: i64, qss: i64, qsd: i64, ksb: i64, ksh: i64, kss: i64, ksd: i64, batch: i32, q_heads: i32, k_heads: i32,
        seq_len: i32, head_dim: i32, rot_dim: i32, cbs: i32, q_eps: f32, k_eps: f32,
    ) {
        qk_norm_rope::<f32, true, false>(q, k, qw, kw, cos, sin, core::ptr::null(), q_out, k_out, qsb, qsh, qss, qsd, ksb, ksh, kss, ksd, batch, q_heads, k_heads, seq_len, head_dim, rot_dim, cbs, q_eps, k_eps)
    }
    #[kernel]
    pub unsafe fn _Z33qk_rms_norm_rope_positions_kernelIfLb1EEvPKT_S2_S2_S2_S2_S2_PKjPS0_S5_lllllllliiiiiiff(
        q: *const f32, k: *const f32, qw: *const f32, kw: *const f32, cos: *const f32, sin: *const f32, positions: *const u32,
        q_out: *mut f32, k_out: *mut f32, qsb: i64, qsh: i64, qss: i64, qsd: i64, ksb: i64, ksh: i64, kss: i64, ksd: i64, batch: i32,
        q_heads: i32, k_heads: i32, seq_len: i32, head_dim: i32, rot_dim: i32, q_eps: f32, k_eps: f32,
    ) {
        qk_norm_rope::<f32, true, true>(q, k, qw, kw, cos, sin, positions, q_out, k_out, qsb, qsh, qss, qsd, ksb, ksh, kss, ksd, batch, q_heads, k_heads, seq_len, head_dim, rot_dim, 0, q_eps, k_eps)
    }
    #[kernel]
    pub unsafe fn _Z34qkv_rms_norm_rope_positions_kernelIfLb1EEvPKT_S2_S2_S2_S2_S2_S2_S2_PKjPS0_S5_S5_lllllllllllliiiiiifff(
        q: *const f32, k: *const f32, v: *const f32, qw: *const f32, kw: *const f32, vw: *const f32, cos: *const f32, sin: *const f32,
        positions: *const u32, q_out: *mut f32, k_out: *mut f32, v_out: *mut f32, qsb: i64, qsh: i64, qss: i64, qsd: i64, ksb: i64,
        ksh: i64, kss: i64, ksd: i64, vsb: i64, vsh: i64, vss: i64, vsd: i64, batch: i32, q_heads: i32, k_heads: i32, seq_len: i32,
        head_dim: i32, rot_dim: i32, q_eps: f32, k_eps: f32, v_eps: f32,
    ) {
        qkv_norm_rope::<f32, true>(q, k, v, qw, kw, vw, cos, sin, positions, q_out, k_out, v_out, qsb, qsh, qss, qsd, ksb, ksh, kss, ksd, vsb, vsh, vss, vsd, batch, q_heads, k_heads, seq_len, head_dim, rot_dim, q_eps, k_eps, v_eps)
    }
    #[kernel]
    pub unsafe fn _Z23qk_rms_norm_rope_kernelI13__nv_bfloat16Lb0EEvPKT_S3_S3_S3_S3_S3_PS1_S4_lllllllliiiiiiiff(
        q: *const B, k: *const B, qw: *const B, kw: *const B, cos: *const B, sin: *const B, q_out: *mut B, k_out: *mut B,
        qsb: i64, qsh: i64, qss: i64, qsd: i64, ksb: i64, ksh: i64, kss: i64, ksd: i64, batch: i32, q_heads: i32, k_heads: i32,
        seq_len: i32, head_dim: i32, rot_dim: i32, cbs: i32, q_eps: f32, k_eps: f32,
    ) {
        qk_norm_rope::<B, false, false>(q, k, qw, kw, cos, sin, core::ptr::null(), q_out, k_out, qsb, qsh, qss, qsd, ksb, ksh, kss, ksd, batch, q_heads, k_heads, seq_len, head_dim, rot_dim, cbs, q_eps, k_eps)
    }
    #[kernel]
    pub unsafe fn _Z33qk_rms_norm_rope_positions_kernelI13__nv_bfloat16Lb0EEvPKT_S3_S3_S3_S3_S3_PKjPS1_S6_lllllllliiiiiiff(
        q: *const B, k: *const B, qw: *const B, kw: *const B, cos: *const B, sin: *const B, positions: *const u32,
        q_out: *mut B, k_out: *mut B, qsb: i64, qsh: i64, qss: i64, qsd: i64, ksb: i64, ksh: i64, kss: i64, ksd: i64, batch: i32,
        q_heads: i32, k_heads: i32, seq_len: i32, head_dim: i32, rot_dim: i32, q_eps: f32, k_eps: f32,
    ) {
        qk_norm_rope::<B, false, true>(q, k, qw, kw, cos, sin, positions, q_out, k_out, qsb, qsh, qss, qsd, ksb, ksh, kss, ksd, batch, q_heads, k_heads, seq_len, head_dim, rot_dim, 0, q_eps, k_eps)
    }
    #[kernel]
    pub unsafe fn _Z34qkv_rms_norm_rope_positions_kernelI13__nv_bfloat16Lb0EEvPKT_S3_S3_S3_S3_S3_S3_S3_PKjPS1_S6_S6_lllllllllllliiiiiifff(
        q: *const B, k: *const B, v: *const B, qw: *const B, kw: *const B, vw: *const B, cos: *const B, sin: *const B,
        positions: *const u32, q_out: *mut B, k_out: *mut B, v_out: *mut B, qsb: i64, qsh: i64, qss: i64, qsd: i64, ksb: i64,
        ksh: i64, kss: i64, ksd: i64, vsb: i64, vsh: i64, vss: i64, vsd: i64, batch: i32, q_heads: i32, k_heads: i32, seq_len: i32,
        head_dim: i32, rot_dim: i32, q_eps: f32, k_eps: f32, v_eps: f32,
    ) {
        qkv_norm_rope::<B, false>(q, k, v, qw, kw, vw, cos, sin, positions, q_out, k_out, v_out, qsb, qsh, qss, qsd, ksb, ksh, kss, ksd, vsb, vsh, vss, vsd, batch, q_heads, k_heads, seq_len, head_dim, rot_dim, q_eps, k_eps, v_eps)
    }
    #[kernel]
    pub unsafe fn _Z23qk_rms_norm_rope_kernelI13__nv_bfloat16Lb1EEvPKT_S3_S3_S3_S3_S3_PS1_S4_lllllllliiiiiiiff(
        q: *const B, k: *const B, qw: *const B, kw: *const B, cos: *const B, sin: *const B, q_out: *mut B, k_out: *mut B,
        qsb: i64, qsh: i64, qss: i64, qsd: i64, ksb: i64, ksh: i64, kss: i64, ksd: i64, batch: i32, q_heads: i32, k_heads: i32,
        seq_len: i32, head_dim: i32, rot_dim: i32, cbs: i32, q_eps: f32, k_eps: f32,
    ) {
        qk_norm_rope::<B, true, false>(q, k, qw, kw, cos, sin, core::ptr::null(), q_out, k_out, qsb, qsh, qss, qsd, ksb, ksh, kss, ksd, batch, q_heads, k_heads, seq_len, head_dim, rot_dim, cbs, q_eps, k_eps)
    }
    #[kernel]
    pub unsafe fn _Z33qk_rms_norm_rope_positions_kernelI13__nv_bfloat16Lb1EEvPKT_S3_S3_S3_S3_S3_PKjPS1_S6_lllllllliiiiiiff(
        q: *const B, k: *const B, qw: *const B, kw: *const B, cos: *const B, sin: *const B, positions: *const u32,
        q_out: *mut B, k_out: *mut B, qsb: i64, qsh: i64, qss: i64, qsd: i64, ksb: i64, ksh: i64, kss: i64, ksd: i64, batch: i32,
        q_heads: i32, k_heads: i32, seq_len: i32, head_dim: i32, rot_dim: i32, q_eps: f32, k_eps: f32,
    ) {
        qk_norm_rope::<B, true, true>(q, k, qw, kw, cos, sin, positions, q_out, k_out, qsb, qsh, qss, qsd, ksb, ksh, kss, ksd, batch, q_heads, k_heads, seq_len, head_dim, rot_dim, 0, q_eps, k_eps)
    }
    #[kernel]
    pub unsafe fn _Z34qkv_rms_norm_rope_positions_kernelI13__nv_bfloat16Lb1EEvPKT_S3_S3_S3_S3_S3_S3_S3_PKjPS1_S6_S6_lllllllllllliiiiiifff(
        q: *const B, k: *const B, v: *const B, qw: *const B, kw: *const B, vw: *const B, cos: *const B, sin: *const B,
        positions: *const u32, q_out: *mut B, k_out: *mut B, v_out: *mut B, qsb: i64, qsh: i64, qss: i64, qsd: i64, ksb: i64,
        ksh: i64, kss: i64, ksd: i64, vsb: i64, vsh: i64, vss: i64, vsd: i64, batch: i32, q_heads: i32, k_heads: i32, seq_len: i32,
        head_dim: i32, rot_dim: i32, q_eps: f32, k_eps: f32, v_eps: f32,
    ) {
        qkv_norm_rope::<B, true>(q, k, v, qw, kw, vw, cos, sin, positions, q_out, k_out, v_out, qsb, qsh, qss, qsd, ksb, ksh, kss, ksd, vsb, vsh, vss, vsd, batch, q_heads, k_heads, seq_len, head_dim, rot_dim, q_eps, k_eps, v_eps)
    }
    #[kernel]
    pub unsafe fn _Z23qk_rms_norm_rope_kernelI6__halfLb0EEvPKT_S3_S3_S3_S3_S3_PS1_S4_lllllllliiiiiiiff(
        q: *const H, k: *const H, qw: *const H, kw: *const H, cos: *const H, sin: *const H, q_out: *mut H, k_out: *mut H,
        qsb: i64, qsh: i64, qss: i64, qsd: i64, ksb: i64, ksh: i64, kss: i64, ksd: i64, batch: i32, q_heads: i32, k_heads: i32,
        seq_len: i32, head_dim: i32, rot_dim: i32, cbs: i32, q_eps: f32, k_eps: f32,
    ) {
        qk_norm_rope::<H, false, false>(q, k, qw, kw, cos, sin, core::ptr::null(), q_out, k_out, qsb, qsh, qss, qsd, ksb, ksh, kss, ksd, batch, q_heads, k_heads, seq_len, head_dim, rot_dim, cbs, q_eps, k_eps)
    }
    #[kernel]
    pub unsafe fn _Z33qk_rms_norm_rope_positions_kernelI6__halfLb0EEvPKT_S3_S3_S3_S3_S3_PKjPS1_S6_lllllllliiiiiiff(
        q: *const H, k: *const H, qw: *const H, kw: *const H, cos: *const H, sin: *const H, positions: *const u32,
        q_out: *mut H, k_out: *mut H, qsb: i64, qsh: i64, qss: i64, qsd: i64, ksb: i64, ksh: i64, kss: i64, ksd: i64, batch: i32,
        q_heads: i32, k_heads: i32, seq_len: i32, head_dim: i32, rot_dim: i32, q_eps: f32, k_eps: f32,
    ) {
        qk_norm_rope::<H, false, true>(q, k, qw, kw, cos, sin, positions, q_out, k_out, qsb, qsh, qss, qsd, ksb, ksh, kss, ksd, batch, q_heads, k_heads, seq_len, head_dim, rot_dim, 0, q_eps, k_eps)
    }
    #[kernel]
    pub unsafe fn _Z34qkv_rms_norm_rope_positions_kernelI6__halfLb0EEvPKT_S3_S3_S3_S3_S3_S3_S3_PKjPS1_S6_S6_lllllllllllliiiiiifff(
        q: *const H, k: *const H, v: *const H, qw: *const H, kw: *const H, vw: *const H, cos: *const H, sin: *const H,
        positions: *const u32, q_out: *mut H, k_out: *mut H, v_out: *mut H, qsb: i64, qsh: i64, qss: i64, qsd: i64, ksb: i64,
        ksh: i64, kss: i64, ksd: i64, vsb: i64, vsh: i64, vss: i64, vsd: i64, batch: i32, q_heads: i32, k_heads: i32, seq_len: i32,
        head_dim: i32, rot_dim: i32, q_eps: f32, k_eps: f32, v_eps: f32,
    ) {
        qkv_norm_rope::<H, false>(q, k, v, qw, kw, vw, cos, sin, positions, q_out, k_out, v_out, qsb, qsh, qss, qsd, ksb, ksh, kss, ksd, vsb, vsh, vss, vsd, batch, q_heads, k_heads, seq_len, head_dim, rot_dim, q_eps, k_eps, v_eps)
    }
    #[kernel]
    pub unsafe fn _Z23qk_rms_norm_rope_kernelI6__halfLb1EEvPKT_S3_S3_S3_S3_S3_PS1_S4_lllllllliiiiiiiff(
        q: *const H, k: *const H, qw: *const H, kw: *const H, cos: *const H, sin: *const H, q_out: *mut H, k_out: *mut H,
        qsb: i64, qsh: i64, qss: i64, qsd: i64, ksb: i64, ksh: i64, kss: i64, ksd: i64, batch: i32, q_heads: i32, k_heads: i32,
        seq_len: i32, head_dim: i32, rot_dim: i32, cbs: i32, q_eps: f32, k_eps: f32,
    ) {
        qk_norm_rope::<H, true, false>(q, k, qw, kw, cos, sin, core::ptr::null(), q_out, k_out, qsb, qsh, qss, qsd, ksb, ksh, kss, ksd, batch, q_heads, k_heads, seq_len, head_dim, rot_dim, cbs, q_eps, k_eps)
    }
    #[kernel]
    pub unsafe fn _Z33qk_rms_norm_rope_positions_kernelI6__halfLb1EEvPKT_S3_S3_S3_S3_S3_PKjPS1_S6_lllllllliiiiiiff(
        q: *const H, k: *const H, qw: *const H, kw: *const H, cos: *const H, sin: *const H, positions: *const u32,
        q_out: *mut H, k_out: *mut H, qsb: i64, qsh: i64, qss: i64, qsd: i64, ksb: i64, ksh: i64, kss: i64, ksd: i64, batch: i32,
        q_heads: i32, k_heads: i32, seq_len: i32, head_dim: i32, rot_dim: i32, q_eps: f32, k_eps: f32,
    ) {
        qk_norm_rope::<H, true, true>(q, k, qw, kw, cos, sin, positions, q_out, k_out, qsb, qsh, qss, qsd, ksb, ksh, kss, ksd, batch, q_heads, k_heads, seq_len, head_dim, rot_dim, 0, q_eps, k_eps)
    }
    #[kernel]
    pub unsafe fn _Z34qkv_rms_norm_rope_positions_kernelI6__halfLb1EEvPKT_S3_S3_S3_S3_S3_S3_S3_PKjPS1_S6_S6_lllllllllllliiiiiifff(
        q: *const H, k: *const H, v: *const H, qw: *const H, kw: *const H, vw: *const H, cos: *const H, sin: *const H,
        positions: *const u32, q_out: *mut H, k_out: *mut H, v_out: *mut H, qsb: i64, qsh: i64, qss: i64, qsd: i64, ksb: i64,
        ksh: i64, kss: i64, ksd: i64, vsb: i64, vsh: i64, vss: i64, vsd: i64, batch: i32, q_heads: i32, k_heads: i32, seq_len: i32,
        head_dim: i32, rot_dim: i32, q_eps: f32, k_eps: f32, v_eps: f32,
    ) {
        qkv_norm_rope::<H, true>(q, k, v, qw, kw, vw, cos, sin, positions, q_out, k_out, v_out, qsb, qsh, qss, qsd, ksb, ksh, kss, ksd, vsb, vsh, vss, vsd, batch, q_heads, k_heads, seq_len, head_dim, rot_dim, q_eps, k_eps, v_eps)
    }
    // GENERATED AP KERNELS END

    // ==========================================================================================
    // sort.cu
    // ==========================================================================================

    /// `a == b` as `setp.eq.ftz.f32`.
    #[inline(always)]
    pub fn eq_ftz(a: f32, b: f32) -> bool {
        let r: u32;
        unsafe {
            ptx_asm!("{ .reg .pred p; setp.eq.ftz.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }", out("=r") r, in("f") a, in("f") b, options(register_only))
        };
        r != 0
    }
    /// `a < b` as `setp.lt.ftz.f32`.
    #[inline(always)]
    pub fn lt_ftz(a: f32, b: f32) -> bool {
        let r: u32;
        unsafe {
            ptx_asm!("{ .reg .pred p; setp.lt.ftz.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }", out("=r") r, in("f") a, in("f") b, options(register_only))
        };
        r != 0
    }
    #[inline(always)]
    fn hcmp<const BF16: bool, const GT: bool>(a: u16, b: u16) -> bool {
        let r: u32;
        unsafe {
            match (BF16, GT) {
                (false, true) => ptx_asm!("{ .reg .pred p; setp.gt.f16 p, %1, %2; selp.u32 %0, 1, 0, p; }", out("=r") r, in("h") a, in("h") b, options(register_only)),
                (false, false) => ptx_asm!("{ .reg .pred p; setp.lt.f16 p, %1, %2; selp.u32 %0, 1, 0, p; }", out("=r") r, in("h") a, in("h") b, options(register_only)),
                (true, true) => ptx_asm!("{ .reg .pred p; setp.gt.bf16 p, %1, %2; selp.u32 %0, 1, 0, p; }", out("=r") r, in("h") a, in("h") b, options(register_only)),
                (true, false) => ptx_asm!("{ .reg .pred p; setp.lt.bf16 p, %1, %2; selp.u32 %0, 1, 0, p; }", out("=r") r, in("h") a, in("h") b, options(register_only)),
            }
        };
        r != 0
    }

    /// Element types of the sort kernels: `>` / `<` exactly as the reference compares them
    /// (f32: FSETP.*.FTZ, f16/bf16: HSETP2, f64: DSETP, integers).
    pub trait Sk: Copy {
        fn sgt(a: Self, b: Self) -> bool;
        fn slt(a: Self, b: Self) -> bool;
    }
    impl Sk for f32 {
        #[inline(always)]
        fn sgt(a: f32, b: f32) -> bool {
            gt(a, b)
        }
        #[inline(always)]
        fn slt(a: f32, b: f32) -> bool {
            lt_ftz(a, b)
        }
    }
    impl Sk for f64 {
        #[inline(always)]
        fn sgt(a: f64, b: f64) -> bool {
            a > b
        }
        #[inline(always)]
        fn slt(a: f64, b: f64) -> bool {
            a < b
        }
    }
    impl Sk for u8 {
        #[inline(always)]
        fn sgt(a: u8, b: u8) -> bool {
            a > b
        }
        #[inline(always)]
        fn slt(a: u8, b: u8) -> bool {
            a < b
        }
    }
    impl Sk for u32 {
        #[inline(always)]
        fn sgt(a: u32, b: u32) -> bool {
            a > b
        }
        #[inline(always)]
        fn slt(a: u32, b: u32) -> bool {
            a < b
        }
    }
    impl Sk for i64 {
        #[inline(always)]
        fn sgt(a: i64, b: i64) -> bool {
            a > b
        }
        #[inline(always)]
        fn slt(a: i64, b: i64) -> bool {
            a < b
        }
    }
    impl Sk for H {
        #[inline(always)]
        fn sgt(a: H, b: H) -> bool {
            hcmp::<false, true>(a.0, b.0)
        }
        #[inline(always)]
        fn slt(a: H, b: H) -> bool {
            hcmp::<false, false>(a.0, b.0)
        }
    }
    impl Sk for B {
        #[inline(always)]
        fn sgt(a: B, b: B) -> bool {
            hcmp::<true, true>(a.0, b.0)
        }
        #[inline(always)]
        fn slt(a: B, b: B) -> bool {
            hcmp::<true, false>(a.0, b.0)
        }
    }

    #[inline(always)]
    unsafe fn bitonic<T: Sk, const ASC: bool>(arr: *mut T, dst: *mut u32, j: i32, k: i32) {
        let i = thread::threadIdx_x().wrapping_add(thread::blockDim_x().wrapping_mul(thread::blockIdx_x()));
        let ij = i ^ (j as u32);
        if ij > i {
            let (pi, pj) = (arr.add(i as usize), arr.add(ij as usize));
            let first = if ASC { (i & k as u32) == 0 } else { (i & k as u32) != 0 };
            let sw = if first { T::sgt(*pi, *pj) } else { T::slt(*pi, *pj) };
            if sw {
                let t = *pi;
                *pi = *pj;
                *pj = t;
                let (di, dj) = (dst.add(i as usize), dst.add(ij as usize));
                let t = *di;
                *di = *dj;
                *dj = t;
            }
        }
        thread::sync_threads();
    }
    // GENERATED SORT KERNELS BEGIN
    #[kernel]
    pub unsafe fn _Z26rms_norm_strided_4d_kernelI13__nv_bfloat16EvPKT_S3_PS1_lllliiiif(x: *const B, w: *const B, d: *mut B, sb: i64, sh: i64, ss: i64, sd: i64, b: i32, h: i32, s: i32, hd: i32, eps: f32) {
        rms_strided_4d::<B>(x, w, d, sb, sh, ss, sd, b, h, s, hd, eps)
    }
    #[kernel]
    pub unsafe fn _Z26rms_norm_strided_4d_kernelI6__halfEvPKT_S3_PS1_lllliiiif(x: *const H, w: *const H, d: *mut H, sb: i64, sh: i64, ss: i64, sd: i64, b: i32, h: i32, s: i32, hd: i32, eps: f32) {
        rms_strided_4d::<H>(x, w, d, sb, sh, ss, sd, b, h, s, hd, eps)
    }
    #[kernel]
    pub unsafe fn _Z26rms_norm_strided_4d_kernelIfEvPKT_S2_PS0_lllliiiif(x: *const f32, w: *const f32, d: *mut f32, sb: i64, sh: i64, ss: i64, sd: i64, b: i32, h: i32, s: i32, hd: i32, eps: f32) {
        rms_strided_4d::<f32>(x, w, d, sb, sh, ss, sd, b, h, s, hd, eps)
    }
    #[kernel]
    pub unsafe fn _Z38rms_norm_residual_then_rms_norm_kernelI13__nv_bfloat16EvPKT_S3_S3_S3_S3_PS1_S4_iff(x: *const B, r: *const B, rw: *const B, s: *const B, nw: *const B, rd: *mut B, nd: *mut B, ncols: i32, re: f32, ne: f32) {
        rms_residual_then::<B, false>(x, r, rw, s, nw, rd, nd, ncols, re, ne)
    }
    #[kernel]
    pub unsafe fn _Z43rms_norm_residual_then_rms_norm_vec8_kernelI13__nv_bfloat16EvPKT_S3_S3_S3_S3_PS1_S4_iff(x: *const B, r: *const B, rw: *const B, s: *const B, nw: *const B, rd: *mut B, nd: *mut B, ncols: i32, re: f32, ne: f32) {
        rms_residual_then::<B, true>(x, r, rw, s, nw, rd, nd, ncols, re, ne)
    }
    #[kernel]
    pub unsafe fn _Z38rms_norm_residual_then_rms_norm_kernelI6__halfEvPKT_S3_S3_S3_S3_PS1_S4_iff(x: *const H, r: *const H, rw: *const H, s: *const H, nw: *const H, rd: *mut H, nd: *mut H, ncols: i32, re: f32, ne: f32) {
        rms_residual_then::<H, false>(x, r, rw, s, nw, rd, nd, ncols, re, ne)
    }
    #[kernel]
    pub unsafe fn _Z43rms_norm_residual_then_rms_norm_vec8_kernelI6__halfEvPKT_S3_S3_S3_S3_PS1_S4_iff(x: *const H, r: *const H, rw: *const H, s: *const H, nw: *const H, rd: *mut H, nd: *mut H, ncols: i32, re: f32, ne: f32) {
        rms_residual_then::<H, true>(x, r, rw, s, nw, rd, nd, ncols, re, ne)
    }
    #[kernel]
    pub unsafe fn _Z38rms_norm_residual_then_rms_norm_kernelIfEvPKT_S2_S2_S2_S2_PS0_S3_iff(x: *const f32, r: *const f32, rw: *const f32, s: *const f32, nw: *const f32, rd: *mut f32, nd: *mut f32, ncols: i32, re: f32, ne: f32) {
        rms_residual_then::<f32, false>(x, r, rw, s, nw, rd, nd, ncols, re, ne)
    }
    #[kernel]
    pub unsafe fn _Z24rms_norm_residual_kernelI13__nv_bfloat16EvPKT_S3_S3_S3_PS1_if(x: *const B, r: *const B, w: *const B, s: *const B, d: *mut B, ncols: i32, eps: f32) {
        rms_residual::<B, false>(x, r, w, s, d, ncols, eps)
    }
    #[kernel]
    pub unsafe fn _Z29rms_norm_residual_vec8_kernelI13__nv_bfloat16EvPKT_S3_S3_S3_PS1_if(x: *const B, r: *const B, w: *const B, s: *const B, d: *mut B, ncols: i32, eps: f32) {
        rms_residual_vec8::<B>(x, r, w, s, d, ncols, eps)
    }
    #[kernel]
    pub unsafe fn _Z24rms_norm_residual_kernelI6__halfEvPKT_S3_S3_S3_PS1_if(x: *const H, r: *const H, w: *const H, s: *const H, d: *mut H, ncols: i32, eps: f32) {
        rms_residual::<H, false>(x, r, w, s, d, ncols, eps)
    }
    #[kernel]
    pub unsafe fn _Z29rms_norm_residual_vec8_kernelI6__halfEvPKT_S3_S3_S3_PS1_if(x: *const H, r: *const H, w: *const H, s: *const H, d: *mut H, ncols: i32, eps: f32) {
        rms_residual_vec8::<H>(x, r, w, s, d, ncols, eps)
    }
    #[kernel]
    pub unsafe fn _Z24rms_norm_residual_kernelIfEvPKT_S2_S2_S2_PS0_if(x: *const f32, r: *const f32, w: *const f32, s: *const f32, d: *mut f32, ncols: i32, eps: f32) {
        rms_residual::<f32, false>(x, r, w, s, d, ncols, eps)
    }
    #[kernel]
    pub unsafe fn _Z19bitonic_sort_kernelIlLb0EEvPT_Pjii(arr: *mut i64, dst: *mut u32, j: i32, k: i32) {
        bitonic::<i64, false>(arr, dst, j, k)
    }
    #[kernel]
    pub unsafe fn _Z19bitonic_sort_kernelIjLb0EEvPT_Pjii(arr: *mut u32, dst: *mut u32, j: i32, k: i32) {
        bitonic::<u32, false>(arr, dst, j, k)
    }
    #[kernel]
    pub unsafe fn _Z19bitonic_sort_kernelIhLb0EEvPT_Pjii(arr: *mut u8, dst: *mut u32, j: i32, k: i32) {
        bitonic::<u8, false>(arr, dst, j, k)
    }
    #[kernel]
    pub unsafe fn _Z19bitonic_sort_kernelIdLb0EEvPT_Pjii(arr: *mut f64, dst: *mut u32, j: i32, k: i32) {
        bitonic::<f64, false>(arr, dst, j, k)
    }
    #[kernel]
    pub unsafe fn _Z19bitonic_sort_kernelIfLb0EEvPT_Pjii(arr: *mut f32, dst: *mut u32, j: i32, k: i32) {
        bitonic::<f32, false>(arr, dst, j, k)
    }
    #[kernel]
    pub unsafe fn _Z19bitonic_sort_kernelIlLb1EEvPT_Pjii(arr: *mut i64, dst: *mut u32, j: i32, k: i32) {
        bitonic::<i64, true>(arr, dst, j, k)
    }
    #[kernel]
    pub unsafe fn _Z19bitonic_sort_kernelIjLb1EEvPT_Pjii(arr: *mut u32, dst: *mut u32, j: i32, k: i32) {
        bitonic::<u32, true>(arr, dst, j, k)
    }
    #[kernel]
    pub unsafe fn _Z19bitonic_sort_kernelIhLb1EEvPT_Pjii(arr: *mut u8, dst: *mut u32, j: i32, k: i32) {
        bitonic::<u8, true>(arr, dst, j, k)
    }
    #[kernel]
    pub unsafe fn _Z19bitonic_sort_kernelIdLb1EEvPT_Pjii(arr: *mut f64, dst: *mut u32, j: i32, k: i32) {
        bitonic::<f64, true>(arr, dst, j, k)
    }
    #[kernel]
    pub unsafe fn _Z19bitonic_sort_kernelIfLb1EEvPT_Pjii(arr: *mut f32, dst: *mut u32, j: i32, k: i32) {
        bitonic::<f32, true>(arr, dst, j, k)
    }
    #[kernel]
    pub unsafe fn _Z19bitonic_sort_kernelI6__halfLb0EEvPT_Pjii(arr: *mut H, dst: *mut u32, j: i32, k: i32) {
        bitonic::<H, false>(arr, dst, j, k)
    }
    #[kernel]
    pub unsafe fn _Z19bitonic_sort_kernelI6__halfLb1EEvPT_Pjii(arr: *mut H, dst: *mut u32, j: i32, k: i32) {
        bitonic::<H, true>(arr, dst, j, k)
    }
    #[kernel]
    pub unsafe fn _Z19bitonic_sort_kernelI13__nv_bfloat16Lb0EEvPT_Pjii(arr: *mut B, dst: *mut u32, j: i32, k: i32) {
        bitonic::<B, false>(arr, dst, j, k)
    }
    #[kernel]
    pub unsafe fn _Z19bitonic_sort_kernelI13__nv_bfloat16Lb1EEvPT_Pjii(arr: *mut B, dst: *mut u32, j: i32, k: i32) {
        bitonic::<B, true>(arr, dst, j, k)
    }
    // GENERATED SORT KERNELS END

    // ------------------------------------------------------------------ sampling helpers
    #[kernel]
    pub unsafe fn _Z15copy_f32_kernelPKfPfi(x: *const f32, dst: *mut f32, n: i32) {
        let idx = bid_x().wrapping_mul(bdim_x()).wrapping_add(tid_x());
        if idx >= n {
            return;
        }
        *offm(dst, idx) = *off(x, idx);
    }
    #[kernel]
    pub unsafe fn _Z33apply_sparse_penalties_f32_kernelPfPKjPKfiifff(
        logits: *mut f32, token_ids: *const u32, counts: *const f32, n: i32, n_tokens: i32, freq: f32, presence: f32, rep: f32,
    ) {
        let idx = bid_x().wrapping_mul(bdim_x()).wrapping_add(tid_x());
        if idx >= n_tokens {
            return;
        }
        let tok = *off(token_ids, idx);
        if tok >= n as u32 {
            return;
        }
        let count = *off(counts, idx);
        // `count <= 0` (FSETP.GTU.FTZ: NaN proceeds, a denormal count is zero)
        if !(gt(count, 0.0) || count != count) {
            return;
        }
        let p = logits.add(tok as usize);
        let mut value = subf(*p, fmaf(count, freq, presence));
        let rep_ne_one = !eq_ftz(rep, 1.0);
        if rep_ne_one {
            value = if gt(value, 0.0) { mulf(value, rcp(rep)) } else { mulf(value, rep) };
        }
        *p = value;
    }
    #[kernel]
    pub unsafe fn _Z35apply_sparse_logits_bias_f32_kernelPfPKjPKfii(logits: *mut f32, token_ids: *const u32, biases: *const f32, n: i32, n_tokens: i32) {
        let idx = bid_x().wrapping_mul(bdim_x()).wrapping_add(tid_x());
        if idx >= n_tokens {
            return;
        }
        let tok = *off(token_ids, idx);
        if tok >= n as u32 {
            return;
        }
        let p = logits.add(tok as usize);
        *p = addf(*p, *off(biases, idx));
    }
    #[kernel]
    pub unsafe fn _Z28apply_causal_mask_f32_kernelPfiiiii(scores: *mut f32, batch_heads: i32, q_len: i32, kv_len: i32, q_offset: i32, prefix_len: i32) {
        let idx = bid_x().wrapping_mul(bdim_x()).wrapping_add(tid_x());
        let total = batch_heads.wrapping_mul(q_len).wrapping_mul(kv_len);
        if idx >= total {
            return;
        }
        let kv_idx = idx % kv_len;
        let q_idx = (idx / kv_len) % q_len;
        let q_pos = prefix_len.wrapping_add(q_offset).wrapping_add(q_idx);
        if kv_idx > q_pos {
            *offm(scores, idx) = f32::NEG_INFINITY;
        }
    }

    // ------------------------------------------------------------------ fused RMSNorm (+ residual)
    /// `rms_block_sum`: shfl.down per warp, warp sums in shared memory, warp 0 reduces again,
    /// thread 0 publishes to warp_sums[0].
    #[inline(always)]
    unsafe fn rms_block_sum(mut value: f32, ws: *mut f32) -> f32 {
        let tid = tid_x();
        let lane = tid & 31;
        let warp = tid >> 5;
        let mut o = 16;
        while o > 0 {
            value = addf(value, warp::shuffle_down_f32_sync(FULL, value, o));
            o >>= 1;
        }
        if lane == 0 {
            *ws.offset(warp as isize) = value;
        }
        thread::sync_threads();
        let num_warps = (bdim_x() + 31) >> 5;
        value = if tid < num_warps { *ws.offset(lane as isize) } else { 0.0 };
        if warp == 0 {
            let mut o = 16;
            while o > 0 {
                value = addf(value, warp::shuffle_down_f32_sync(FULL, value, o));
                o >>= 1;
            }
        }
        if tid == 0 {
            *ws = value;
        }
        thread::sync_threads();
        *ws
    }

    /// rms_norm_residual[_vec8]_kernel (VEC8: 16-byte vectors, only f16/bf16).
    #[inline(always)]
    unsafe fn rms_residual<T: Ap, const VEC8: bool>(x: *const T, residual: *const T, weight: *const T, scale: *const T, dst: *mut T, ncols: i32, eps: f32) {
        static mut RED: SharedArray<f32, 32> = SharedArray::UNINIT;
        let ws = SharedArray::as_raw_mut_ptr(&raw mut RED);
        let row = bid_x();
        let tid = tid_x();
        let bd = bdim_x();
        let row_offset = row.wrapping_mul(ncols);
        let scale_value = if scale.is_null() { 1.0 } else { (*scale).af() };
        let mut sum = 0f32;
        let mut col = tid;
        while col < ncols {
            let v = (*off(x, row_offset + col)).af();
            sum = fmaf(v, v, sum);
            col += bd;
        }
        let _ = VEC8;
        let inv_rms = rsqrt(fmaf(rms_block_sum(sum, ws), rcp(ncols as f32), eps));
        let mut col = tid;
        while col < ncols {
            let normed_part = mulf(inv_rms, (*off(x, row_offset + col)).af());
            let value = mulf(fmaf(normed_part, (*off(weight, col)).af(), (*off(residual, row_offset + col)).af()), scale_value);
            *offm(dst, row_offset + col) = T::at(value);
            col += bd;
        }
    }

    /// The vec8 kernels: each thread handles 8 consecutive columns per step (same arithmetic
    /// per element; the sum of squares accumulates the 8 values into a fresh partial first).
    #[inline(always)]
    unsafe fn rms_residual_vec8<T: Ap>(x: *const T, residual: *const T, weight: *const T, scale: *const T, dst: *mut T, ncols: i32, eps: f32) {
        static mut RED: SharedArray<f32, 32> = SharedArray::UNINIT;
        let ws = SharedArray::as_raw_mut_ptr(&raw mut RED);
        let row = bid_x();
        let tid = tid_x();
        let bd = bdim_x();
        let vec_cols = ncols / 8;
        let row_offset = row.wrapping_mul(vec_cols);
        let scale_value = if scale.is_null() { 1.0 } else { (*scale).af() };
        let mut sum = 0f32;
        let mut col = tid;
        while col < vec_cols {
            let base = (row_offset + col) * 8;
            let mut part = 0f32;
            let mut i = 0;
            while i < 8 {
                let v = (*off(x, base + i)).af();
                part = fmaf(v, v, part);
                i += 1;
            }
            sum = addf(sum, part);
            col += bd;
        }
        let inv_rms = rsqrt(fmaf(rms_block_sum(sum, ws), rcp(ncols as f32), eps));
        let mut col = tid;
        while col < vec_cols {
            let base = (row_offset + col) * 8;
            let mut i = 0;
            while i < 8 {
                let normed_part = mulf(inv_rms, (*off(x, base + i)).af());
                let value = mulf(fmaf(normed_part, (*off(weight, col * 8 + i)).af(), (*off(residual, base + i)).af()), scale_value);
                *offm(dst, base + i) = T::at(value);
                i += 1;
            }
            col += bd;
        }
    }

    #[inline(always)]
    unsafe fn rms_residual_then<T: Ap, const VEC8: bool>(
        x: *const T, residual: *const T, rw: *const T, scale: *const T, nw: *const T, rdst: *mut T, ndst: *mut T, ncols: i32, reps: f32, neps: f32,
    ) {
        static mut RED: SharedArray<f32, 32> = SharedArray::UNINIT;
        let ws = SharedArray::as_raw_mut_ptr(&raw mut RED);
        let row = bid_x();
        let tid = tid_x();
        let bd = bdim_x();
        let (step, cols) = if VEC8 { (8, ncols / 8) } else { (1, ncols) };
        let row_offset = row.wrapping_mul(cols);
        let scale_value = if scale.is_null() { 1.0 } else { (*scale).af() };
        let mut sum = 0f32;
        let mut col = tid;
        while col < cols {
            let base = (row_offset + col) * step;
            if VEC8 {
                let mut part = 0f32;
                let mut i = 0;
                while i < 8 {
                    let v = (*off(x, base + i)).af();
                    part = fmaf(v, v, part);
                    i += 1;
                }
                sum = addf(sum, part);
            } else {
                let v = (*off(x, base)).af();
                sum = fmaf(v, v, sum);
            }
            col += bd;
        }
        let inv_rms = rsqrt(fmaf(rms_block_sum(sum, ws), rcp(ncols as f32), reps));
        let mut rsum = 0f32;
        let mut col = tid;
        while col < cols {
            let base = (row_offset + col) * step;
            let mut i = 0;
            while i < step {
                let normed_part = mulf(inv_rms, (*off(x, base + i)).af());
                let value = mulf(fmaf(normed_part, (*off(rw, col * step + i)).af(), (*off(residual, base + i)).af()), scale_value);
                *offm(rdst, base + i) = T::at(value);
                rsum = fmaf(value, value, rsum);
                i += 1;
            }
            col += bd;
        }
        let ninv = rsqrt(fmaf(rms_block_sum(rsum, ws), rcp(ncols as f32), neps));
        let mut col = tid;
        while col < cols {
            let base = (row_offset + col) * step;
            let mut i = 0;
            while i < step {
                let v = mulf(mulf(ninv, (*offm(rdst, base + i)).af()), (*off(nw, col * step + i)).af());
                *offm(ndst, base + i) = T::at(v);
                i += 1;
            }
            col += bd;
        }
    }

    #[inline(always)]
    unsafe fn rms_strided_4d<T: Ap>(x: *const T, weight: *const T, dst: *mut T, sb: i64, sh: i64, ss: i64, sd: i64, _batch: i32, heads: i32, seq_len: i32, head_dim: i32, eps: f32) {
        static mut RED: SharedArray<f32, 32> = SharedArray::UNINIT;
        let ws = SharedArray::as_raw_mut_ptr(&raw mut RED);
        let row = bid_x();
        let tid = tid_x();
        let bd = bdim_x();
        let seq = row % seq_len;
        let tmp = row / seq_len;
        let head = tmp % heads;
        let b = tmp / heads;
        let base = (b as i64).wrapping_mul(sb).wrapping_add((head as i64).wrapping_mul(sh)).wrapping_add((seq as i64).wrapping_mul(ss));
        let dst_base = (row as i64).wrapping_mul(head_dim as i64);
        let ld = |c: i32| (*x.wrapping_offset(base.wrapping_add((c as i64).wrapping_mul(sd)) as isize)).af();
        let mut sum = 0f32;
        let mut col = tid;
        while col < head_dim {
            let v = ld(col);
            sum = fmaf(v, v, sum);
            col += bd;
        }
        let inv_rms = rsqrt(fmaf(rms_block_sum(sum, ws), rcp(head_dim as f32), eps));
        let mut col = tid;
        while col < head_dim {
            let v = mulf(mulf(inv_rms, ld(col)), (*off(weight, col)).af());
            *dst.wrapping_offset(dst_base.wrapping_add(col as i64) as isize) = T::at(v);
            col += bd;
        }
    }

    // ------------------------------------------------------------------ topk (small k)
    pub trait Tk: Ap + Sk {
        fn neg_inf() -> Self;
        fn zero_v() -> Self;
        fn to_bits32(self) -> u32;
        fn from_bits32(b: u32) -> Self;
    }
    impl Tk for f32 {
        #[inline(always)]
        fn neg_inf() -> f32 {
            f32::NEG_INFINITY
        }
        #[inline(always)]
        fn zero_v() -> f32 {
            0.0
        }
        #[inline(always)]
        fn to_bits32(self) -> u32 {
            self.to_bits()
        }
        #[inline(always)]
        fn from_bits32(b: u32) -> f32 {
            f32::from_bits(b)
        }
    }
    impl Tk for H {
        #[inline(always)]
        fn neg_inf() -> H {
            H(0xfc00)
        }
        #[inline(always)]
        fn zero_v() -> H {
            H(0)
        }
        #[inline(always)]
        fn to_bits32(self) -> u32 {
            self.0 as u32
        }
        #[inline(always)]
        fn from_bits32(b: u32) -> H {
            H(b as u16)
        }
    }
    impl Tk for B {
        #[inline(always)]
        fn neg_inf() -> B {
            B(0xff80)
        }
        #[inline(always)]
        fn zero_v() -> B {
            B(0)
        }
        #[inline(always)]
        fn to_bits32(self) -> u32 {
            self.0 as u32
        }
        #[inline(always)]
        fn from_bits32(b: u32) -> B {
            B(b as u16)
        }
    }

    /// warp_reduce_max_with_idx<T>: shfl.down, take the other lane's pair iff other > val.
    #[inline(always)]
    fn warp_max_idx<T: Tk>(mut val: T, mut idx: i32) -> (T, i32) {
        let mut o = 16;
        while o > 0 {
            let ov = T::from_bits32(warp::shuffle_down_sync(FULL, val.to_bits32(), o));
            let oi = warp::shuffle_down_sync(FULL, idx as u32, o) as i32;
            if T::sgt(ov, val) {
                val = ov;
                idx = oi;
            }
            o >>= 1;
        }
        (val, idx)
    }

    #[inline(always)]
    unsafe fn topk_small<T: Tk>(input: *const T, values_out: *mut T, indices_out: *mut u32, nrows: i32, ncols: i32, k: i32) {
        static mut WM: SharedArray<u32, 32> = SharedArray::UNINIT;
        static mut WI: SharedArray<i32, 32> = SharedArray::UNINIT;
        let wm = SharedArray::as_raw_mut_ptr(&raw mut WM);
        let wi = SharedArray::as_raw_mut_ptr(&raw mut WI);
        let row = bid_x();
        if row >= nrows {
            return;
        }
        let row_in = off(input, row.wrapping_mul(ncols));
        let row_values = offm(values_out, row.wrapping_mul(k));
        let row_indices = offm(indices_out, row.wrapping_mul(k));
        let tid = tid_x();
        let bs = bdim_x();
        let smem = DynamicSharedArray::<u8>::get_raw();
        let s_data = smem as *mut T;
        let s_used = s_data.wrapping_offset(ncols as isize) as *mut u8;
        let mut i = tid;
        while i < ncols {
            *s_data.offset(i as isize) = *off(row_in, i);
            *s_used.offset(i as isize) = 0;
            i += bs;
        }
        thread::sync_threads();
        let mut ki = 0;
        while ki < k {
            let mut local_max = T::neg_inf();
            let mut local_idx = -1i32;
            let mut i = tid;
            while i < ncols {
                let cand = (*s_data.offset(i as isize)).af();
                if *s_used.offset(i as isize) == 0 && cand == cand && gt(cand, local_max.af()) {
                    local_max = *s_data.offset(i as isize);
                    local_idx = i;
                }
                i += bs;
            }
            let (wmax, widx) = warp_max_idx(local_max, local_idx);
            let warp_id = tid / 32;
            let lane = tid % 32;
            let num_warps = (bs + 31) / 32;
            if lane == 0 {
                *wm.offset(warp_id as isize) = wmax.to_bits32();
                *wi.offset(warp_id as isize) = widx;
            }
            thread::sync_threads();
            if tid < 32 {
                let v = if tid < num_warps { T::from_bits32(*wm.offset(tid as isize)) } else { T::neg_inf() };
                let ix = if tid < num_warps { *wi.offset(tid as isize) } else { -1 };
                let (mut fmax, mut fidx) = warp_max_idx(v, ix);
                if tid == 0 {
                    if fidx < 0 {
                        fidx = 0;
                        fmax = T::zero_v();
                    }
                    *offm(row_values, ki) = fmax;
                    *offm(row_indices, ki) = fidx as u32;
                    *s_used.offset(fidx as isize) = 1;
                }
            }
            thread::sync_threads();
            ki += 1;
        }
    }
    #[kernel]
    pub unsafe fn _Z11topk_kernelIfEvPKT_PS0_Pjiii(input: *const f32, v: *mut f32, ix: *mut u32, nrows: i32, ncols: i32, k: i32) {
        topk_small::<f32>(input, v, ix, nrows, ncols, k)
    }
    #[kernel]
    pub unsafe fn _Z11topk_kernelI13__nv_bfloat16EvPKT_PS1_Pjiii(input: *const B, v: *mut B, ix: *mut u32, nrows: i32, ncols: i32, k: i32) {
        topk_small::<B>(input, v, ix, nrows, ncols, k)
    }
    #[kernel]
    pub unsafe fn _Z11topk_kernelI6__halfEvPKT_PS1_Pjiii(input: *const H, v: *mut H, ix: *mut u32, nrows: i32, ncols: i32, k: i32) {
        topk_small::<H>(input, v, ix, nrows, ncols, k)
    }

    // ------------------------------------------------------------------ MoE router top-k
    #[inline(always)]
    fn router_sum(mut v: f32) -> f32 {
        let mut o = 16;
        while o > 0 {
            v = addf(v, warp::shuffle_xor_f32_sync(FULL, v, o));
            o >>= 1;
        }
        v
    }
    #[inline(always)]
    fn router_max(mut v: f32) -> f32 {
        let mut o = 16;
        while o > 0 {
            v = maxf(v, warp::shuffle_xor_f32_sync(FULL, v, o));
            o >>= 1;
        }
        v
    }
    /// Values per lane as an associated const, so the `#[unroll]`-style loops below see a
    /// compile-time trip count (every per-lane array is then indexed by constants only and SROA
    /// keeps it in registers; slots >= VPT are dead and vanish).
    pub struct RK<const NE: usize>;
    impl<const NE: usize> RK<NE> {
        pub const VPT: usize = if NE > 32 { NE / 32 } else { 1 };
    }
    #[inline(always)]
    fn router_softmax<const NE: usize>(vals: &mut [f32; 18], limit: i32, lane: i32, use_limit: bool) {
        let mut mx = f32::NEG_INFINITY;
        let mut i = 0usize;
        while i < RK::<NE>::VPT {
            cuda_device::thread::__unroll_config::<0>();
            let idx = lane + (i as i32) * 32;
            if !use_limit || idx < limit {
                mx = maxf(mx, vals[i]);
            }
            i += 1;
        }
        mx = router_max(mx);
        let mut sum = 0f32;
        let mut i = 0usize;
        while i < RK::<NE>::VPT {
            cuda_device::thread::__unroll_config::<0>();
            let idx = lane + (i as i32) * 32;
            if !use_limit || idx < limit {
                vals[i] = expf(subf(vals[i], mx));
                sum = addf(sum, vals[i]);
            } else {
                vals[i] = 0.0;
            }
            i += 1;
        }
        sum = router_sum(sum);
        let inv = rcp(sum);
        let mut i = 0usize;
        while i < RK::<NE>::VPT {
            cuda_device::thread::__unroll_config::<0>();
            let idx = lane + (i as i32) * 32;
            if !use_limit || idx < limit {
                vals[i] = mulf(vals[i], inv);
            }
            i += 1;
        }
    }

    #[inline(always)]
    unsafe fn moe_router<T: Ap, const NE: usize, const HB: bool, const HS: bool>(
        logits: *const T, weights: *mut f32, ids: *mut u32, bias: *const f32, escale: *const f32, n_rows: i32, top_k: i32, score_mode: i32,
        weight_mode: i32, renormalize: bool, clamp_logits: bool, clamp_min: f32, clamp_max: f32, norm_min: f32, output_scale: f32,
    ) {
        let ne = NE as i32;
        let lane = tid_x();
        let row = bid_x().wrapping_mul(thread::blockDim_y() as i32).wrapping_add(thread::threadIdx_y() as i32);
        if row >= n_rows {
            return;
        }
        let logits = off(logits, row.wrapping_mul(ne));
        let weights = offm(weights, row.wrapping_mul(top_k));
        let ids = offm(ids, row.wrapping_mul(top_k));
        let mut raw = [0f32; 18];
        let mut score = [0f32; 18];
        let mut sel = [0f32; 18];
        let mut ow = [0f32; 18];
        let mut oid = [0u32; 18];
        let mut i = 0usize;
        while i < RK::<NE>::VPT {
            cuda_device::thread::__unroll_config::<0>();
            let expert = lane + (i as i32) * 32;
            let mut value = if NE % 32 == 0 || expert < ne { (*off(logits, expert)).af() } else { f32::NEG_INFINITY };
            if clamp_logits && expert < ne {
                value = minf(maxf(value, clamp_min), clamp_max);
            }
            if value != value {
                value = f32::NEG_INFINITY;
            }
            raw[i] = value;
            score[i] = value;
            sel[i] = value;
            i += 1;
        }
        if score_mode == 1 {
            router_softmax::<NE>(&mut score, ne, lane, false);
        } else if score_mode == 2 {
            let mut i = 0usize;
            while i < RK::<NE>::VPT {
                cuda_device::thread::__unroll_config::<0>();
                score[i] = rcp(addf(ex2(mulf(score[i], -LOG2E)), 1.0));
                i += 1;
            }
        }
        let mut i = 0usize;
        while i < RK::<NE>::VPT {
            cuda_device::thread::__unroll_config::<0>();
            let expert = lane + (i as i32) * 32;
            sel[i] = score[i];
            if HB && expert < ne {
                sel[i] = addf(sel[i], *off(bias, expert));
            }
            if sel[i] != sel[i] {
                sel[i] = f32::NEG_INFINITY;
            }
            i += 1;
        }
        let mut k_idx = 0;
        while k_idx < top_k {
            let mut bsel = sel[0];
            let mut bsc = score[0];
            let mut braw = raw[0];
            let mut bex = lane;
            let mut i = 1usize;
            while i < RK::<NE>::VPT {
                cuda_device::thread::__unroll_config::<0>();
                let expert = lane + (i as i32) * 32;
                if (NE % 32 == 0 || expert < ne) & gt(sel[i], bsel) {
                    bsel = sel[i];
                    bsc = score[i];
                    braw = raw[i];
                    bex = expert;
                }
                i += 1;
            }
            let mut m = 16;
            while m > 0 {
                let os = warp::shuffle_xor_f32_sync(FULL, bsel, m);
                let osc = warp::shuffle_xor_f32_sync(FULL, bsc, m);
                let orw = warp::shuffle_xor_f32_sync(FULL, braw, m);
                let oex = warp::shuffle_xor_sync(FULL, bex as u32, m) as i32;
                // Non-short-circuit `|`/`&`: the ptx_asm! predicates are not speculated, so `||`
                // becomes a branch per butterfly step; eager evaluation gives selects.
                if gt(os, bsel) | (eq_ftz(os, bsel) & (oex < bex)) {
                    bsel = os;
                    bsc = osc;
                    braw = orw;
                    bex = oex;
                }
                m >>= 1;
            }
            let mut out = bsc;
            if weight_mode == 1 {
                out = braw;
            } else if weight_mode == 2 {
                out = rcp(addf(ex2(mulf(braw, -LOG2E)), 1.0));
            }
            // Runtime-indexed writes (`ow[k_idx / 32]`, `sel[bex / 32]`) as select chains over
            // constant indices, so the arrays stay in registers.
            let wslot = if (k_idx & 31) == lane { (k_idx / 32) as usize } else { usize::MAX };
            let sslot = if (bex & 31) == lane { (bex / 32) as usize } else { usize::MAX };
            let mut i = 0usize;
            while i < RK::<NE>::VPT {
                cuda_device::thread::__unroll_config::<0>();
                if wslot == i {
                    ow[i] = out;
                    oid[i] = bex as u32;
                }
                if sslot == i {
                    sel[i] = f32::NEG_INFINITY;
                }
                i += 1;
            }
            k_idx += 1;
        }
        if weight_mode == 1 {
            router_softmax::<NE>(&mut ow, top_k, lane, true);
        }
        if renormalize {
            let mut sum = 0f32;
            let mut i = 0usize;
            while i < RK::<NE>::VPT {
                cuda_device::thread::__unroll_config::<0>();
                if lane + (i as i32) * 32 < top_k {
                    sum = addf(sum, ow[i]);
                }
                i += 1;
            }
            sum = router_sum(sum);
            sum = maxf(sum, norm_min);
            let inv = rcp(sum);
            let mut i = 0usize;
            while i < RK::<NE>::VPT {
                cuda_device::thread::__unroll_config::<0>();
                ow[i] = mulf(ow[i], inv);
                i += 1;
            }
        }
        let mut i = 0usize;
        while i < RK::<NE>::VPT {
            cuda_device::thread::__unroll_config::<0>();
            let idx = lane + (i as i32) * 32;
            if idx < top_k {
                let mut sc = output_scale;
                if HS {
                    sc = mulf(sc, *escale.add(oid[i] as usize));
                }
                *offm(weights, idx) = mulf(ow[i], sc);
                *offm(ids, idx) = oid[i];
            }
            i += 1;
        }
    }
    // GENERATED ROUTER KERNELS BEGIN
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi576ELb0ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 576, false, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi512ELb0ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 512, false, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi256ELb0ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 256, false, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi128ELb0ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 128, false, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi64ELb0ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 64, false, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi32ELb0ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 32, false, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi16ELb0ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 16, false, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi8ELb0ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 8, false, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi4ELb0ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 4, false, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi2ELb0ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 2, false, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi1ELb0ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 1, false, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi576ELb0ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 576, false, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi512ELb0ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 512, false, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi256ELb0ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 256, false, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi128ELb0ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 128, false, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi64ELb0ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 64, false, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi32ELb0ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 32, false, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi16ELb0ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 16, false, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi8ELb0ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 8, false, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi4ELb0ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 4, false, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi2ELb0ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 2, false, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi1ELb0ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 1, false, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi576ELb1ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 576, true, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi512ELb1ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 512, true, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi256ELb1ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 256, true, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi128ELb1ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 128, true, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi64ELb1ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 64, true, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi32ELb1ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 32, true, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi16ELb1ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 16, true, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi8ELb1ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 8, true, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi4ELb1ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 4, true, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi2ELb1ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 2, true, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi1ELb1ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 1, true, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi576ELb1ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 576, true, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi512ELb1ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 512, true, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi256ELb1ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 256, true, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi128ELb1ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 128, true, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi64ELb1ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 64, true, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi32ELb1ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 32, true, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi16ELb1ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 16, true, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi8ELb1ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 8, true, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi4ELb1ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 4, true, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi2ELb1ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 2, true, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI6__halfLi1ELb1ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const H, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<H, 1, true, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li576ELb0ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 576, false, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li512ELb0ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 512, false, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li256ELb0ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 256, false, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li128ELb0ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 128, false, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li64ELb0ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 64, false, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li32ELb0ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 32, false, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li16ELb0ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 16, false, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li8ELb0ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 8, false, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li4ELb0ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 4, false, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li2ELb0ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 2, false, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li1ELb0ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 1, false, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li576ELb0ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 576, false, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li512ELb0ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 512, false, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li256ELb0ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 256, false, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li128ELb0ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 128, false, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li64ELb0ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 64, false, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li32ELb0ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 32, false, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li16ELb0ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 16, false, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li8ELb0ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 8, false, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li4ELb0ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 4, false, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li2ELb0ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 2, false, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li1ELb0ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 1, false, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li576ELb1ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 576, true, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li512ELb1ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 512, true, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li256ELb1ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 256, true, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li128ELb1ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 128, true, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li64ELb1ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 64, true, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li32ELb1ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 32, true, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li16ELb1ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 16, true, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li8ELb1ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 8, true, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li4ELb1ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 4, true, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li2ELb1ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 2, true, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li1ELb1ELb0EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 1, true, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li576ELb1ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 576, true, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li512ELb1ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 512, true, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li256ELb1ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 256, true, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li128ELb1ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 128, true, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li64ELb1ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 64, true, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li32ELb1ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 32, true, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li16ELb1ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 16, true, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li8ELb1ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 8, true, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li4ELb1ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 4, true, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li2ELb1ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 2, true, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelI13__nv_bfloat16Li1ELb1ELb1EEvPKT_PfPjPKfS7_iiiibbffff(l: *const B, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<B, 1, true, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi576ELb0ELb0EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 576, false, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi512ELb0ELb0EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 512, false, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi256ELb0ELb0EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 256, false, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi128ELb0ELb0EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 128, false, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi64ELb0ELb0EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 64, false, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi32ELb0ELb0EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 32, false, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi16ELb0ELb0EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 16, false, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi8ELb0ELb0EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 8, false, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi4ELb0ELb0EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 4, false, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi2ELb0ELb0EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 2, false, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi1ELb0ELb0EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 1, false, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi576ELb0ELb1EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 576, false, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi512ELb0ELb1EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 512, false, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi256ELb0ELb1EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 256, false, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi128ELb0ELb1EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 128, false, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi64ELb0ELb1EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 64, false, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi32ELb0ELb1EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 32, false, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi16ELb0ELb1EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 16, false, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi8ELb0ELb1EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 8, false, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi4ELb0ELb1EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 4, false, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi2ELb0ELb1EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 2, false, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi1ELb0ELb1EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 1, false, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi576ELb1ELb0EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 576, true, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi512ELb1ELb0EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 512, true, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi256ELb1ELb0EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 256, true, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi128ELb1ELb0EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 128, true, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi64ELb1ELb0EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 64, true, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi32ELb1ELb0EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 32, true, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi16ELb1ELb0EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 16, true, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi8ELb1ELb0EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 8, true, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi4ELb1ELb0EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 4, true, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi2ELb1ELb0EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 2, true, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi1ELb1ELb0EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 1, true, false>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi576ELb1ELb1EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 576, true, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi512ELb1ELb1EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 512, true, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi256ELb1ELb1EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 256, true, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi128ELb1ELb1EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 128, true, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi64ELb1ELb1EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 64, true, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi32ELb1ELb1EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 32, true, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi16ELb1ELb1EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 16, true, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi8ELb1ELb1EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 8, true, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi4ELb1ELb1EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 4, true, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi2ELb1ELb1EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 2, true, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    #[kernel]
    pub unsafe fn _Z22moe_router_topk_kernelIfLi1ELb1ELb1EEvPKT_PfPjPKfS6_iiiibbffff(l: *const f32, w: *mut f32, ids: *mut u32, b: *const f32, es: *const f32, nr: i32, tk: i32, sm: i32, wm: i32, rn: u8, cl: u8, cmin: f32, cmax: f32, nmin: f32, os: f32) {
        moe_router::<f32, 1, true, true>(l, w, ids, b, es, nr, tk, sm, wm, rn != 0, cl != 0, cmin, cmax, nmin, os)
    }
    // GENERATED ROUTER KERNELS END

    // ------------------------------------------------------------------ large-vocabulary top-k
    #[inline(always)]
    unsafe fn block_reduce_sum(mut val: f32, ws: *mut f32) -> f32 {
        let tid = tid_x();
        let warp_id = tid / 32;
        let lane = tid % 32;
        let num_warps = (bdim_x() + 31) / 32;
        let mut o = 16;
        while o > 0 {
            val = addf(val, warp::shuffle_down_f32_sync(FULL, val, o));
            o >>= 1;
        }
        if lane == 0 {
            *ws.offset(warp_id as isize) = val;
        }
        thread::sync_threads();
        val = if tid < num_warps { *ws.offset(tid as isize) } else { 0.0 };
        if warp_id == 0 {
            let mut o = 16;
            while o > 0 {
                val = addf(val, warp::shuffle_down_f32_sync(FULL, val, o));
                o >>= 1;
            }
        }
        val
    }

    /// Block-wide argmax of (val, idx) pairs: warp_max_idx, warp results in shared memory, warp 0
    /// again. Returns the result in thread 0 (other threads: unspecified).
    #[inline(always)]
    unsafe fn block_argmax(v: f32, ix: i32, wm: *mut u32, wi: *mut i32) -> (f32, i32) {
        let tid = tid_x();
        let (wmax, widx) = warp_max_idx::<f32>(v, ix);
        let warp_id = tid / 32;
        let lane = tid % 32;
        let num_warps = (bdim_x() + 31) / 32;
        if lane == 0 {
            *wm.offset(warp_id as isize) = wmax.to_bits();
            *wi.offset(warp_id as isize) = widx;
        }
        thread::sync_threads();
        let mut r = (0f32, 0i32);
        if tid < 32 {
            let val = if tid < num_warps { f32::from_bits(*wm.offset(tid as isize)) } else { f32::NEG_INFINITY };
            let idx = if tid < num_warps { *wi.offset(tid as isize) } else { -1 };
            r = warp_max_idx::<f32>(val, idx);
        }
        r
    }

    #[kernel]
    pub unsafe fn _Z21topk_large_stage1_f32PKfPfPjS1_S1_iiif(
        input: *const f32, block_values: *mut f32, block_indices: *mut u32, block_maxes: *mut f32, block_sums: *mut f32, ncols: i32, k: i32,
        chunk_size: i32, inv_t: f32,
    ) {
        static mut WM: SharedArray<u32, 32> = SharedArray::UNINIT;
        static mut WI: SharedArray<i32, 32> = SharedArray::UNINIT;
        static mut WS: SharedArray<f32, 32> = SharedArray::UNINIT;
        let wm = SharedArray::as_raw_mut_ptr(&raw mut WM);
        let wi = SharedArray::as_raw_mut_ptr(&raw mut WI);
        let ws = SharedArray::as_raw_mut_ptr(&raw mut WS);
        let chunk = bid_x();
        let start = chunk.wrapping_mul(chunk_size);
        let e = start.wrapping_add(chunk_size);
        let end = if e < ncols { e } else { ncols };
        let width = if end - start > 0 { end - start } else { 0 };
        let tid = tid_x();
        let bs = bdim_x();
        let s_used = DynamicSharedArray::<u8>::get_raw();
        let mut i = tid;
        while i < chunk_size {
            *s_used.offset(i as isize) = 0;
            i += bs;
        }
        thread::sync_threads();
        let mut ki = 0;
        while ki < k {
            let mut lmax = f32::NEG_INFINITY;
            let mut lidx = -1i32;
            let mut l = tid;
            while l < width {
                let c = *off(input, start + l);
                if *s_used.offset(l as isize) == 0 && c == c && gt(c, lmax) {
                    lmax = c;
                    lidx = start + l;
                }
                l += bs;
            }
            let (fmax, fidx) = block_argmax(lmax, lidx, wm, wi);
            if tid == 0 {
                *offm(block_values, chunk.wrapping_mul(k).wrapping_add(ki)) = fmax;
                *offm(block_indices, chunk.wrapping_mul(k).wrapping_add(ki)) = if fidx >= 0 { fidx as u32 } else { 0 };
                if fidx >= start && fidx < end {
                    *s_used.offset((fidx - start) as isize) = 1;
                }
            }
            thread::sync_threads();
            ki += 1;
        }
        let block_max = if width > 0 { mulf(*offm(block_values, chunk.wrapping_mul(k)), inv_t) } else { f32::NEG_INFINITY };
        let mut lsum = 0f32;
        if block_max != f32::NEG_INFINITY {
            let mut l = tid;
            while l < width {
                let c = *off(input, start + l);
                if c == c {
                    lsum = addf(lsum, ex2(mulf(fmaf(c, inv_t, -block_max), LOG2E)));
                }
                l += bs;
            }
        }
        let bsum = block_reduce_sum(lsum, ws);
        if tid == 0 {
            *offm(block_maxes, chunk) = block_max;
            *offm(block_sums, chunk) = bsum;
        }
    }

    /// topk_large_stage2_f32 (PACKED = false) and topk_large_stage2_f32_packed.
    #[inline(always)]
    unsafe fn topk_large_stage2<const PACKED: bool>(
        block_values: *const f32, block_indices: *const u32, block_maxes: *const f32, block_sums: *const f32, values_out: *mut f32,
        indices_out: *mut u32, info_out: *mut f32, nblocks: i32, k: i32,
    ) {
        static mut WM: SharedArray<u32, 32> = SharedArray::UNINIT;
        static mut WI: SharedArray<i32, 32> = SharedArray::UNINIT;
        static mut WS: SharedArray<f32, 32> = SharedArray::UNINIT;
        static mut GM: SharedArray<f32, 1> = SharedArray::UNINIT;
        let wm = SharedArray::as_raw_mut_ptr(&raw mut WM);
        let wi = SharedArray::as_raw_mut_ptr(&raw mut WI);
        let ws = SharedArray::as_raw_mut_ptr(&raw mut WS);
        let gm = SharedArray::as_raw_mut_ptr(&raw mut GM);
        let tid = tid_x();
        let bs = bdim_x();
        let n_cand = nblocks.wrapping_mul(k);
        let s_used = DynamicSharedArray::<u8>::get_raw();
        let mut i = tid;
        while i < n_cand {
            *s_used.offset(i as isize) = 0;
            i += bs;
        }
        thread::sync_threads();
        let mut lgm = f32::NEG_INFINITY;
        let mut b = tid;
        while b < nblocks {
            lgm = maxf(lgm, *off(block_maxes, b));
            b += bs;
        }
        let (gmax, _) = block_argmax(lgm, tid, wm, wi);
        if tid == 0 {
            *gm = gmax;
        }
        thread::sync_threads();
        let g = *gm;
        let mut ld = 0f32;
        if g != f32::NEG_INFINITY {
            let mut b = tid;
            while b < nblocks {
                ld = fmaf(*off(block_sums, b), expf(subf(*off(block_maxes, b), g)), ld);
                b += bs;
            }
        }
        let denom = block_reduce_sum(ld, ws);
        if tid == 0 {
            if PACKED {
                *offm(values_out, k.wrapping_mul(2)) = denom;
                *offm(values_out, k.wrapping_mul(2).wrapping_add(1)) = g;
            } else {
                *info_out = denom;
                *info_out.add(1) = g;
            }
        }
        thread::sync_threads();
        let mut ki = 0;
        while ki < k {
            let mut lmax = f32::NEG_INFINITY;
            let mut lpos = -1i32;
            let mut pos = tid;
            while pos < n_cand {
                let c = *off(block_values, pos);
                if *s_used.offset(pos as isize) == 0 && c == c && gt(c, lmax) {
                    lmax = c;
                    lpos = pos;
                }
                pos += bs;
            }
            let (fmax, fpos) = block_argmax(lmax, lpos, wm, wi);
            if tid == 0 {
                if PACKED {
                    *offm(values_out, ki) = fmax;
                    *offm(values_out, k.wrapping_add(ki)) = if fpos >= 0 { cvt_u32_f32(*off(block_indices, fpos)) } else { 0.0 };
                } else {
                    *offm(values_out, ki) = fmax;
                    *offm(indices_out, ki) = if fpos >= 0 { *off(block_indices, fpos) } else { 0 };
                }
                if fpos >= 0 {
                    *s_used.offset(fpos as isize) = 1;
                }
            }
            thread::sync_threads();
            ki += 1;
        }
    }
    /// `static_cast<float>(uint32_t)` (cvt.rn.f32.u32).
    #[inline(always)]
    fn cvt_u32_f32(u: u32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("cvt.rn.f32.u32 %0, %1;", out("=f") r, in("r") u, options(register_only)) };
        r
    }
    #[kernel]
    pub unsafe fn _Z21topk_large_stage2_f32PKfPKjS0_S0_PfPjS3_ii(
        block_values: *const f32, block_indices: *const u32, block_maxes: *const f32, block_sums: *const f32, values_out: *mut f32,
        indices_out: *mut u32, info_out: *mut f32, nblocks: i32, k: i32,
    ) {
        topk_large_stage2::<false>(block_values, block_indices, block_maxes, block_sums, values_out, indices_out, info_out, nblocks, k)
    }
    #[kernel]
    pub unsafe fn _Z28topk_large_stage2_f32_packedPKfPKjS0_S0_Pfii(
        block_values: *const f32, block_indices: *const u32, block_maxes: *const f32, block_sums: *const f32, packed_out: *mut f32, nblocks: i32, k: i32,
    ) {
        topk_large_stage2::<true>(block_values, block_indices, block_maxes, block_sums, packed_out, core::ptr::null_mut(), core::ptr::null_mut(), nblocks, k)
    }
    #[kernel]
    pub unsafe fn _Z21top1_large_stage1_f32PKfPfPjii(input: *const f32, block_values: *mut f32, block_indices: *mut u32, ncols: i32, chunk_size: i32) {
        static mut WM: SharedArray<u32, 32> = SharedArray::UNINIT;
        static mut WI: SharedArray<i32, 32> = SharedArray::UNINIT;
        let wm = SharedArray::as_raw_mut_ptr(&raw mut WM);
        let wi = SharedArray::as_raw_mut_ptr(&raw mut WI);
        let chunk = bid_x();
        let start = chunk.wrapping_mul(chunk_size);
        let e = start.wrapping_add(chunk_size);
        let end = if e < ncols { e } else { ncols };
        let tid = tid_x();
        let bs = bdim_x();
        let mut lmax = f32::NEG_INFINITY;
        let mut lidx = -1i32;
        let mut idx = start + tid;
        while idx < end {
            let c = *off(input, idx);
            if c == c && gt(c, lmax) {
                lmax = c;
                lidx = idx;
            }
            idx += bs;
        }
        let (fmax, fidx) = block_argmax(lmax, lidx, wm, wi);
        if tid == 0 {
            *offm(block_values, chunk) = fmax;
            *offm(block_indices, chunk) = if fidx >= 0 { fidx as u32 } else { 0 };
        }
    }
    #[kernel]
    pub unsafe fn _Z28top1_large_stage2_f32_packedPKfPKjPfi(block_values: *const f32, block_indices: *const u32, packed_out: *mut f32, nblocks: i32) {
        static mut WM: SharedArray<u32, 32> = SharedArray::UNINIT;
        static mut WI: SharedArray<i32, 32> = SharedArray::UNINIT;
        let wm = SharedArray::as_raw_mut_ptr(&raw mut WM);
        let wi = SharedArray::as_raw_mut_ptr(&raw mut WI);
        let tid = tid_x();
        let bs = bdim_x();
        let mut lmax = f32::NEG_INFINITY;
        let mut lpos = -1i32;
        let mut pos = tid;
        while pos < nblocks {
            let c = *off(block_values, pos);
            if c == c && gt(c, lmax) {
                lmax = c;
                lpos = pos;
            }
            pos += bs;
        }
        let (fmax, fpos) = block_argmax(lmax, lpos, wm, wi);
        if tid == 0 {
            *packed_out = fmax;
            *packed_out.add(1) = if fpos >= 0 { cvt_u32_f32(*off(block_indices, fpos)) } else { 0.0 };
        }
    }
}

fn main() {
    std::process::exit(if gate::run() { 0 } else { 1 });
}
