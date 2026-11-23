//! titan-fused: fused decode-step kernels for titan-engine (cuda-oxide).
//!
//! `titan_moe_epilogue_f32` replaces the end of a qwen35moe MoE layer, eight launches per layer in the decode graph:
//! `bmul` (routed outputs x routing weights), `fast_sum` (over the top-k), `usigmoid` (shared-expert gate logit),
//! `bmul` (shared expert x gate), `badd` (routed + shared), `badd` (+ the attention residual). It runs the same f32
//! operations in the same order as those candle kernels:
//!   - fast_sum with block size b = next_pow2(k): thread t starts from 0.0 and adds its one element (t < k), then the
//!     shared-memory tree s = b/2, b/4, .., 1 adds slot t + s into slot t;
//!   - sigmoid as candle-unary's `sigmoid_f32` (libdevice expf split, rcp.rn);
//!   - every mul / add `.rn` (candle-kernels are built without fast-math, and separate kernels never contract).
//! mistral.rs embeds titan_fused.ptx (mistralrs-core/src/models/titan_fused_oxide.ptx) and checks it against the
//! unfused ops on live data with TITAN_FUSED_CHECK=1 (graphs off).
#![allow(non_snake_case, clippy::missing_safety_doc, dead_code)]

use cuda_device::{float, kernel, ptx_asm, thread, warp, SharedArray};
use cuda_host::cuda_module;

#[cuda_module]
pub mod kernels {
    use super::*;

    pub const KMAX: usize = 16;

    #[inline(always)]
    fn cvt_sat_f32(x: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("cvt.sat.f32.f32 %0, %1;", out("=f") r, in("f") x, options(register_only)); }
        r
    }

    /// candle-unary's libdevice `expf` split before its final multiply: expf(x) = e * s.
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

    /// candle-unary `sigmoid_f32`.
    #[inline(always)]
    fn sigmoid_f32(x: f32) -> f32 {
        let (e, s) = expf_parts(-x);
        float::rcp_rn_f32(float::fma_rn_f32(e, s, 1.0))
    }

    #[inline(always)]
    fn max_f32(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("max.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    #[inline(always)]
    fn min_f32(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("min.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }

    /// `x` rounded to the nearest integer, ties to even (`cvt.rni`, as `f32::round_ties_even`).
    #[inline(always)]
    fn rint_f32(x: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("cvt.rni.f32.f32 %0, %1;", out("=f") r, in("f") x, options(register_only)); }
        r
    }

    /// e^x from IEEE operations only (no MUFU approximations), so the CPU twin (`titan_cpu::exp_portable`) computes the
    /// same bits: n = rint(x log2 e), r = x - n ln2 (two fma), Taylor to r^7 by Horner fma, times 2^n. x clamped to
    /// [-87, 88] (2^n stays a normal number).
    #[inline(always)]
    pub fn exp_portable(x: f32) -> f32 {
        let x = min_f32(max_f32(x, -87.0), 88.0);
        let n = rint_f32(float::mul_rn_f32(x, f32::from_bits(0x3fb8aa3b)));
        let r = float::fma_rn_f32(-n, f32::from_bits(0x3f317200), x);
        let r = float::fma_rn_f32(-n, f32::from_bits(0x35bfbe8e), r);
        let mut p = f32::from_bits(0x39500d01); // 1/5040
        p = float::fma_rn_f32(p, r, f32::from_bits(0x3ab60b61)); // 1/720
        p = float::fma_rn_f32(p, r, f32::from_bits(0x3c088889)); // 1/120
        p = float::fma_rn_f32(p, r, f32::from_bits(0x3d2aaaab)); // 1/24
        p = float::fma_rn_f32(p, r, f32::from_bits(0x3e2aaaab)); // 1/6
        p = float::fma_rn_f32(p, r, 0.5);
        p = float::fma_rn_f32(p, r, 1.0);
        p = float::fma_rn_f32(p, r, 1.0);
        let s = f32::from_bits((((n as i32) + 127) as u32) << 23);
        float::mul_rn_f32(p, s)
    }

    /// SwiGLU of the tiered experts: out[i] = (g / (1 + e^-g)) * u, IEEE ops and `exp_portable`, bit-identical to the
    /// CPU twin's `titan_cpu::silu_mul`. grid ceil(n / 256), block 256.
    #[kernel]
    pub unsafe fn titan_swiglu_f32(g: *const f32, u: *const f32, out: *mut f32, n: u32) {
        let i = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if i >= n {
            return;
        }
        let x = *g.add(i as usize);
        let e = exp_portable(-x);
        let s = float::div_rn_f32(x, float::add_rn_f32(1.0, e));
        *out.add(i as usize) = float::mul_rn_f32(s, *u.add(i as usize));
    }

    /// candle-reduce's `warp_sum`: shfl.bfly 16, 8, 4, 2, 1, add.rn.
    #[inline(always)]
    fn warp_sum(mut x: f32) -> f32 {
        let mut m = 16;
        while m > 0 {
            x = float::add_rn_f32(x, warp::shuffle_xor_f32_sync(0xffff_ffff, x, m));
            m >>= 1;
        }
        x
    }

    /// Residual add + RMSNorm of the decode trunk in one pass: s = f32(y) + r (badd_f32's add.f32), written to `sum`;
    /// RMSNorm of s with candle-reduce `rmsnorm_f32`'s exact order (thread partials by fma.rn over a stride of
    /// `block_size`, warp_sum, the 32 warp sums through shared memory and warp_sum again when block_size > 32,
    /// div.rn by ncols, rsqrt.approx(mean + eps), out = (scale * s) * alpha with mul.rn) written to `out`, and its
    /// cvt.rn bf16 narrowing to `outb` (the matvec input, `narrow_matvec`). One launch for cast_bf16_f32 (y), badd_f32,
    /// rmsnorm_f32 and cast_f32_bf16. grid = rows, block = block_size (candle: 1024 for ncols >= 1024, else 32).
    #[inline(always)]
    unsafe fn add_rmsnorm<L: Fn(isize) -> f32>(
        ly: L, r: *const f32, alpha: *const f32, sum: *mut f32, out: *mut f32, outb: *mut u16, ncols: i32, block_size: i32, eps: f32,
    ) {
        static mut S_SUM: SharedArray<f32, 32> = SharedArray::UNINIT;
        let row = thread::blockIdx_x() as i32;
        let tid = thread::threadIdx_x() as i32;
        let base = row.wrapping_mul(ncols);
        let mut tmp = 0.0f32;
        let mut col = tid;
        while col < ncols {
            let i = base.wrapping_add(col) as isize;
            let x = ly(i) + *r.offset(i);
            *sum.offset(i) = x;
            tmp = float::fma_rn_f32(x, x, tmp);
            col = col.wrapping_add(block_size);
        }
        tmp = warp_sum(tmp);
        if block_size > 32 {
            let s_sum = SharedArray::as_raw_mut_ptr(&raw mut S_SUM);
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
        while col < ncols {
            let i = base.wrapping_add(col) as isize;
            let x = ly(i) + *r.offset(i);
            let v = float::mul_rn_f32(float::mul_rn_f32(scale, x), *alpha.offset(col as isize));
            *out.offset(i) = v;
            if !outb.is_null() {
                let b: u16;
                ptx_asm!("cvt.rn.bf16.f32 %0, %1;", out("=h") b, in("f") v, options(register_only));
                *outb.offset(i) = b;
            }
            col = col.wrapping_add(block_size);
        }
    }

    /// `add_rmsnorm` with a bf16 `y` (a narrowed projection's output, widened exactly).
    #[kernel]
    pub unsafe fn titan_add_rmsnorm_bf16y(
        y: *const u16, r: *const f32, alpha: *const f32, sum: *mut f32, out: *mut f32, outb: *mut u16, ncols: i32, block_size: i32, eps: f32,
    ) {
        add_rmsnorm(|i| f32::from_bits((*y.offset(i) as u32) << 16), r, alpha, sum, out, outb, ncols, block_size, eps)
    }

    /// `add_rmsnorm` with an f32 `y`.
    #[kernel]
    pub unsafe fn titan_add_rmsnorm_f32y(
        y: *const f32, r: *const f32, alpha: *const f32, sum: *mut f32, out: *mut f32, outb: *mut u16, ncols: i32, block_size: i32, eps: f32,
    ) {
        add_rmsnorm(|i| *y.offset(i), r, alpha, sum, out, outb, ncols, block_size, eps)
    }

    /// Dense SwiGLU of a decode step, bf16 in and out: out[i] = bf16(mulf(silu(g[i]), u[i])) with g, u widened exactly.
    /// One launch for what was cast_bf16_f32 (g), cast_bf16_f32 (u), fused_glu4_f32k / fused_glu_f32k (silu, act 0)
    /// and cast_f32_bf16 (the narrowing before the down projection), with the same instructions: silu(x) =
    /// x * rcp.approx.ftz(ex2.approx.ftz(x * -log2e) + 1) (mul/add .rn.ftz), the product mul.rn.ftz, cvt.rn.bf16.f32.
    /// grid ceil(n / 256), block 256.
    #[kernel]
    pub unsafe fn titan_glu_silu_bf16(g: *const u16, u: *const u16, out: *mut u16, n: u32) {
        let i = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if i >= n {
            return;
        }
        let gf = f32::from_bits((*g.add(i as usize) as u32) << 16);
        let uf = f32::from_bits((*u.add(i as usize) as u32) << 16);
        let e = float::ex2_approx_ftz_f32(float::mul_rn_ftz_f32(gf, f32::from_bits(0xbfb8aa3b)));
        let s = float::mul_rn_ftz_f32(gf, float::rcp_approx_ftz_f32(float::add_rn_ftz_f32(e, 1.0)));
        let h = float::mul_rn_ftz_f32(s, uf);
        let r: u16;
        ptx_asm!("cvt.rn.bf16.f32 %0, %1;", out("=h") r, in("f") h, options(register_only));
        *out.add(i as usize) = r;
    }

    /// out[r][h] = (sum_k routed[r][k][h] * w[r][k] + shared[r][h] * sigmoid(glogit[r])) + x[r][h]
    /// grid (ceil(hidden / 256), rows), block 256; k <= 16.
    #[kernel]
    pub unsafe fn titan_moe_epilogue_f32(
        routed: *const f32, w: *const f32, shared: *const f32, glogit: *const f32, x: *const f32, out: *mut f32,
        hidden: u32, k: u32,
    ) {
        let h = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if h >= hidden {
            return;
        }
        let r = thread::blockIdx_y();
        let (hd, kk) = (hidden as usize, k as usize);
        let base = r as usize * kk * hd + h as usize;
        // fast_sum's slots: block size next_pow2(k), slot t = 0.0 + element t (t < k), else 0.0
        let mut b = 1usize;
        while b < kk {
            b *= 2;
        }
        let mut s = [0f32; KMAX];
        let mut t = 0usize;
        while t < kk {
            let p = float::mul_rn_f32(*routed.add(base + t * hd), *w.add(r as usize * kk + t));
            s[t] = float::add_rn_f32(0.0, p);
            t += 1;
        }
        let mut step = b / 2;
        while step > 0 {
            let mut i = 0usize;
            while i < step {
                s[i] = float::add_rn_f32(s[i], s[i + step]);
                i += 1;
            }
            step /= 2;
        }
        let o = r as usize * hd + h as usize;
        let g = sigmoid_f32(*glogit.add(r as usize));
        let sh = float::mul_rn_f32(*shared.add(o), g);
        let c = float::add_rn_f32(s[0], sh);
        *out.add(o) = float::add_rn_f32(c, *x.add(o));
    }
}

fn main() {
    println!("titan-fused: kernels only (gated in titan with TITAN_FUSED_CHECK=1)");
}
