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

use cuda_device::{float, kernel, ptx_asm, thread};
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
