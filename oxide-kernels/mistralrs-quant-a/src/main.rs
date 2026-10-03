//! mistralrs-quant group A in cuda-oxide: the kernels of gemv, ops, indexed_moe, moe_grouped, hqq,
//! rotary, bitsandbytes dequant, cutlass moe_data, moe_align, hqq_bitpack and gelu_tanh_and_mul,
//! plus (src/launch.rs) pure-Rust twins of their extern "C" host launchers. Bit-identical to the
//! nvcc build (`-O3 --use_fast_math`, sm_120a; the SASS in libmistralrsquant.a is the spec),
//! checked launcher-by-launcher by the gate in src/gate.rs.
//!
//! Fast-math lowerings read off the SASS (every one is written out explicitly here):
//! - f32 add/sub/mul/fma are `.ftz` (FADD.FTZ / FMUL.FTZ / FFMA.FTZ); f16/bf16 packed ops never flush.
//! - `expf(x)` = `ex2.approx.ftz(x * 1.4427)` (FMUL.FTZ + MUFU.EX2); `expf(-x)` folds the sign into
//!   the constant (x * -1.4427).
//! - `a / b` = `rcp.approx.ftz(b)` then `a * r` (MUFU.RCP + FMUL.FTZ) unless noted.
//! - `tanhf` = `tanh.approx.f32` (MUFU.TANH).
#![allow(non_snake_case, clippy::missing_safety_doc, clippy::too_many_arguments)]
mod gate;
pub mod launch;

use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32, DeviceAtomicI32, DeviceAtomicU32};
use cuda_device::{DynamicSharedArray, SharedArray, bf16x2, convert, dotprod, f16x2, float, kernel, ptx_asm, thread, warp};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    // ------------------------------------------------------------------------------------------
    // shared helpers

    /// `blockIdx.x * blockDim.x + threadIdx.x` in 32-bit unsigned arithmetic (as C computes it).
    #[inline(always)]
    pub fn gid_x() -> u32 {
        thread::blockIdx_x().wrapping_mul(thread::blockDim_x()).wrapping_add(thread::threadIdx_x())
    }
    #[inline(always)]
    pub fn h2f(bits: u16) -> f32 {
        convert::cvt_f32_f16x2_lo(bits as u32)
    }
    #[inline(always)]
    pub fn bf2f(bits: u16) -> f32 {
        f32::from_bits((bits as u32) << 16)
    }
    /// `__float2half` (cvt.rn.f16.f32).
    #[inline(always)]
    pub fn f2h(v: f32) -> u16 {
        convert::cvt_f16x2_f32(v, 0.0) as u16
    }
    /// `__float2bfloat16` (cvt.rn.bf16.f32).
    #[inline(always)]
    pub fn f2bf(v: f32) -> u16 {
        convert::cvt_bf16x2_f32(v, 0.0) as u16
    }
    #[inline(always)]
    pub fn addf(a: f32, b: f32) -> f32 {
        float::add_rn_ftz_f32(a, b)
    }
    #[inline(always)]
    pub fn subf(a: f32, b: f32) -> f32 {
        float::add_rn_ftz_f32(a, -b)
    }
    #[inline(always)]
    pub fn mulf(a: f32, b: f32) -> f32 {
        float::mul_rn_ftz_f32(a, b)
    }
    #[inline(always)]
    pub fn fmaf(a: f32, b: f32, c: f32) -> f32 {
        float::fma_rn_ftz_f32(a, b, c)
    }

    // ------------------------------------------------------------------------------------------
    // hqq_bitpack.cu: integer packing (size_t indices, 32-bit thread id).

    #[inline(always)]
    unsafe fn pack_u8<const PER: usize, const BITS: u32>(input: *const u8, output: *mut u8, n: usize, width: usize) {
        let tid = gid_x() as usize;
        let step = n / PER;
        if tid < step.wrapping_mul(width) {
            let row = tid / width;
            let col = tid % width;
            if row < step {
                let mut packed: u8 = 0;
                let mut i = 0;
                while i < PER {
                    let r = row.wrapping_add(i.wrapping_mul(step));
                    if r < n {
                        let v = *input.add(r.wrapping_mul(width).wrapping_add(col)) & ((1u8 << BITS) - 1);
                        packed |= ((v as u32) << (8 - BITS as usize * (i + 1))) as u8;
                    }
                    i += 1;
                }
                *output.add(row.wrapping_mul(width).wrapping_add(col)) = packed;
            }
        }
    }

    #[kernel]
    pub unsafe fn pack_1bit_kernel(input: *const u8, output: *mut u8, n: usize, width: usize) {
        pack_u8::<8, 1>(input, output, n, width)
    }
    #[kernel]
    pub unsafe fn pack_2bit_kernel(input: *const u8, output: *mut u8, n: usize, width: usize) {
        pack_u8::<4, 2>(input, output, n, width)
    }
    #[kernel]
    pub unsafe fn pack_4bit_kernel(input: *const u8, output: *mut u8, n: usize, width: usize) {
        pack_u8::<2, 4>(input, output, n, width)
    }
    #[kernel]
    pub unsafe fn pack_3bit_kernel(input: *const u32, output: *mut i32, n: usize, width: usize) {
        let tid = gid_x() as usize;
        let step = n / 10;
        if tid < step.wrapping_mul(width) {
            let row = tid / width;
            let col = tid % width;
            if row < step {
                let mut packed: i32 = 0;
                let mut i = 0usize;
                while i < 10 {
                    let r = row.wrapping_add(i.wrapping_mul(step));
                    if r < n {
                        let v = (*input.add(r.wrapping_mul(width).wrapping_add(col)) & 7) as i32;
                        packed |= v << (27 - 3 * i as u32);
                    }
                    i += 1;
                }
                *output.add(row.wrapping_mul(width).wrapping_add(col)) = packed;
            }
        }
    }
    #[kernel]
    pub unsafe fn pack_8bit_kernel(input: *const u8, output: *mut u8, n: usize) {
        let tid = gid_x() as usize;
        if tid < n {
            *output.add(tid) = *input.add(tid);
        }
    }

    // ------------------------------------------------------------------------------------------
    // hqq.cu: W_r[i + c*n] = ((T)(chunk c of Wq[i]) - zero[j]) * scale[j], j = i % w (int maths).

    /// One dtype of the hqq dequantizers: (T)(small int) exactly, then `(q - z) * s` in T.
    pub trait Hqq: Copy {
        fn deq(q: u32, z: Self, s: Self) -> Self;
    }
    impl Hqq for f32 {
        #[inline(always)]
        fn deq(q: u32, z: f32, s: f32) -> f32 {
            mulf(subf(q as f32, z), s)
        }
    }
    #[derive(Clone, Copy)]
    pub struct H(pub u16);
    #[derive(Clone, Copy)]
    pub struct B(pub u16);
    impl Hqq for H {
        #[inline(always)]
        fn deq(q: u32, z: H, s: H) -> H {
            let qh = f2h(q as f32) as u32; // exact: q < 256
            H(f16x2::mul_f16x2(f16x2::sub_f16x2(qh, z.0 as u32), s.0 as u32) as u16)
        }
    }
    impl Hqq for B {
        #[inline(always)]
        fn deq(q: u32, z: B, s: B) -> B {
            let qb = f2bf(q as f32) as u32; // exact: q < 256
            B(bf16x2::mul_bf16x2(bf16x2::sub_bf16x2(qb, z.0 as u32), s.0 as u32) as u16)
        }
    }

    /// `CHUNKS` values of `BITS` bits per packed word, most significant first.
    #[inline(always)]
    unsafe fn hqq_deq<T: Hqq, W: Copy + Into<u32>, const CHUNKS: i32, const BITS: u32, const TOP: u32>(
        wq: *const W, scale: *const T, zero: *const T, out: *mut T, h: i32, w: i32,
    ) {
        let i = gid_x() as i32;
        let n = h.wrapping_mul(w);
        if i >= n {
            return;
        }
        let j = i.wrapping_rem(w);
        let word: u32 = (*wq.offset(i as isize)).into();
        let (z, s) = (*zero.offset(j as isize), *scale.offset(j as isize));
        let mut c = 0;
        while c < CHUNKS {
            let q = (word >> (TOP - BITS * c as u32)) & ((1u32 << BITS) - 1);
            *out.offset(i.wrapping_add(n.wrapping_mul(c)) as isize) = T::deq(q, z, s);
            c += 1;
        }
    }

    #[kernel] pub unsafe fn hqq_8bit_f32(wq: *const u8, s: *const f32, z: *const f32, o: *mut f32, h: i32, w: i32) { hqq_deq::<f32, u8, 1, 8, 0>(wq, s, z, o, h, w) }
    #[kernel] pub unsafe fn hqq_8bit_f16(wq: *const u8, s: *const H, z: *const H, o: *mut H, h: i32, w: i32) { hqq_deq::<H, u8, 1, 8, 0>(wq, s, z, o, h, w) }
    #[kernel] pub unsafe fn hqq_8bit_bf16(wq: *const u8, s: *const B, z: *const B, o: *mut B, h: i32, w: i32) { hqq_deq::<B, u8, 1, 8, 0>(wq, s, z, o, h, w) }
    #[kernel] pub unsafe fn hqq_4bit_f32(wq: *const u8, s: *const f32, z: *const f32, o: *mut f32, h: i32, w: i32) { hqq_deq::<f32, u8, 2, 4, 4>(wq, s, z, o, h, w) }
    #[kernel] pub unsafe fn hqq_4bit_f16(wq: *const u8, s: *const H, z: *const H, o: *mut H, h: i32, w: i32) { hqq_deq::<H, u8, 2, 4, 4>(wq, s, z, o, h, w) }
    #[kernel] pub unsafe fn hqq_4bit_bf16(wq: *const u8, s: *const B, z: *const B, o: *mut B, h: i32, w: i32) { hqq_deq::<B, u8, 2, 4, 4>(wq, s, z, o, h, w) }
    #[kernel] pub unsafe fn hqq_2bit_f32(wq: *const u8, s: *const f32, z: *const f32, o: *mut f32, h: i32, w: i32) { hqq_deq::<f32, u8, 4, 2, 6>(wq, s, z, o, h, w) }
    #[kernel] pub unsafe fn hqq_2bit_f16(wq: *const u8, s: *const H, z: *const H, o: *mut H, h: i32, w: i32) { hqq_deq::<H, u8, 4, 2, 6>(wq, s, z, o, h, w) }
    #[kernel] pub unsafe fn hqq_2bit_bf16(wq: *const u8, s: *const B, z: *const B, o: *mut B, h: i32, w: i32) { hqq_deq::<B, u8, 4, 2, 6>(wq, s, z, o, h, w) }
    #[kernel] pub unsafe fn hqq_1bit_f32(wq: *const u8, s: *const f32, z: *const f32, o: *mut f32, h: i32, w: i32) { hqq_deq::<f32, u8, 8, 1, 7>(wq, s, z, o, h, w) }
    #[kernel] pub unsafe fn hqq_1bit_f16(wq: *const u8, s: *const H, z: *const H, o: *mut H, h: i32, w: i32) { hqq_deq::<H, u8, 8, 1, 7>(wq, s, z, o, h, w) }
    #[kernel] pub unsafe fn hqq_1bit_bf16(wq: *const u8, s: *const B, z: *const B, o: *mut B, h: i32, w: i32) { hqq_deq::<B, u8, 8, 1, 7>(wq, s, z, o, h, w) }
    #[kernel] pub unsafe fn hqq_3bit_f32(wq: *const u32, s: *const f32, z: *const f32, o: *mut f32, h: i32, w: i32) { hqq_deq::<f32, u32, 10, 3, 27>(wq, s, z, o, h, w) }
    #[kernel] pub unsafe fn hqq_3bit_f16(wq: *const u32, s: *const H, z: *const H, o: *mut H, h: i32, w: i32) { hqq_deq::<H, u32, 10, 3, 27>(wq, s, z, o, h, w) }
    #[kernel] pub unsafe fn hqq_3bit_bf16(wq: *const u32, s: *const B, z: *const B, o: *mut B, h: i32, w: i32) { hqq_deq::<B, u32, 10, 3, 27>(wq, s, z, o, h, w) }

    // ------------------------------------------------------------------------------------------
    // bitsandbytes dequant.cu: kDequantizeBlockwise<T, 512, 64, 8, DATA_TYPE>. The cub warp-transpose
    // load/store only moves data: thread t owns tile items t*8+j (loads past `valid_load` read 128)
    // and writes outputs t*NV+k while below `valid_store`.

    /// `neg.ftz.f32` (ptxas: FADD.FTZ -x, -RZ): flushes, and a NaN comes back canonical.
    #[inline(always)]
    pub fn neg_ftz(x: f32) -> f32 {
        let r: f32;
        unsafe { cuda_device::ptx_asm!("neg.ftz.f32 %0, %1;", out("=f") r, in("f") x, options(register_only)); }
        r
    }

    /// dDequantizeFP4Tree(v, absmax): `c * absmax * sign`; nvcc folds `1.0f * absmax` to absmax
    /// (no flush) and `x * -1.0f` to neg.ftz.
    #[inline(always)]
    pub fn fp4(v: u32, absmax: f32) -> f32 {
        let c: f32 = if v & 4 != 0 {
            if v & 2 != 0 { if v & 1 != 0 { 0.25 } else { 0.16666667 } } else if v & 1 != 0 { 0.5 } else { 0.33333333 }
        } else if v & 2 != 0 {
            if v & 1 != 0 { 1.0 } else { 0.66666667 }
        } else if v & 1 != 0 {
            5.208333333e-03
        } else {
            0.0
        };
        let base = if v & 3 == 3 && v & 4 == 0 { absmax } else { mulf(c, absmax) };
        if v & 8 != 0 { neg_ftz(base) } else { base }
    }

    #[inline(always)]
    pub fn nf4(v: u32) -> f32 {
        const T: [f32; 16] = [
            -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
            -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
            0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
        ];
        T[(v & 15) as usize]
    }

    pub trait Out: Copy {
        fn from_f32(v: f32) -> Self;
    }
    impl Out for f32 {
        #[inline(always)]
        fn from_f32(v: f32) -> f32 { v }
    }
    impl Out for H {
        #[inline(always)]
        fn from_f32(v: f32) -> H { H(f2h(v)) }
    }
    impl Out for B {
        #[inline(always)]
        fn from_f32(v: f32) -> B { B(f2bf(v)) }
    }

    #[inline(always)]
    unsafe fn bnb_dequant<T: Out, const DT: u32>(code: *const f32, a: *const u8, absmax: *const f32, out: *mut T, blocksize: i32, n: i32) {
        const TILE: u32 = 512;
        let n_load = thread::gridDim_x().wrapping_mul(TILE);
        let tid = thread::threadIdx_x();
        let mut i = thread::blockIdx_x().wrapping_mul(TILE);
        while i < n_load {
            let (valid_load, valid_store): (i32, i32) = if DT > 0 {
                let h = (n.wrapping_add(1) / 2) as u32;
                let vl = if h.wrapping_sub(i) > TILE { TILE } else { h.wrapping_sub(i) };
                let rem = (n as u32).wrapping_sub(i.wrapping_mul(2));
                (vl as i32, if rem > 2 * TILE { 2 * TILE } else { rem } as i32)
            } else {
                let rem = (n as u32).wrapping_sub(i);
                let v = if rem > TILE { TILE } else { rem } as i32;
                (v, v)
            };
            let am = *absmax.add((i.wrapping_add(tid.wrapping_mul(8)) / blocksize as u32) as usize);
            let obase = if DT > 0 { i.wrapping_mul(2) } else { i };
            let mut j = 0u32;
            while j < 8 {
                let item = (tid * 8 + j) as i32;
                let q: u32 = if item < valid_load { *a.add(i.wrapping_add(item as u32) as usize) as u32 } else { 128 };
                if DT == 0 {
                    if item < valid_store {
                        *out.add(obase.wrapping_add(item as u32) as usize) = T::from_f32(mulf(*code.add(q as usize), am));
                    }
                } else {
                    let (hi, lo) = if DT == 1 { (fp4(q >> 4, am), fp4(q & 15, am)) } else { (mulf(nf4(q >> 4), am), mulf(nf4(q & 15), am)) };
                    let o = (tid * 16 + 2 * j) as i32;
                    if o < valid_store {
                        *out.add(obase.wrapping_add(o as u32) as usize) = T::from_f32(hi);
                    }
                    if o + 1 < valid_store {
                        *out.add(obase.wrapping_add(o as u32 + 1) as usize) = T::from_f32(lo);
                    }
                }
                j += 1;
            }
            i = i.wrapping_add(thread::gridDim_x().wrapping_mul(TILE));
        }
    }

    #[kernel] pub unsafe fn bnb_f32_int8(c: *const f32, a: *const u8, am: *const f32, o: *mut f32, bs: i32, n: i32) { bnb_dequant::<f32, 0>(c, a, am, o, bs, n) }
    #[kernel] pub unsafe fn bnb_f32_fp4(c: *const f32, a: *const u8, am: *const f32, o: *mut f32, bs: i32, n: i32) { bnb_dequant::<f32, 1>(c, a, am, o, bs, n) }
    #[kernel] pub unsafe fn bnb_f32_nf4(c: *const f32, a: *const u8, am: *const f32, o: *mut f32, bs: i32, n: i32) { bnb_dequant::<f32, 2>(c, a, am, o, bs, n) }
    #[kernel] pub unsafe fn bnb_f16_int8(c: *const f32, a: *const u8, am: *const f32, o: *mut H, bs: i32, n: i32) { bnb_dequant::<H, 0>(c, a, am, o, bs, n) }
    #[kernel] pub unsafe fn bnb_f16_fp4(c: *const f32, a: *const u8, am: *const f32, o: *mut H, bs: i32, n: i32) { bnb_dequant::<H, 1>(c, a, am, o, bs, n) }
    #[kernel] pub unsafe fn bnb_f16_nf4(c: *const f32, a: *const u8, am: *const f32, o: *mut H, bs: i32, n: i32) { bnb_dequant::<H, 2>(c, a, am, o, bs, n) }
    #[kernel] pub unsafe fn bnb_bf16_int8(c: *const f32, a: *const u8, am: *const f32, o: *mut B, bs: i32, n: i32) { bnb_dequant::<B, 0>(c, a, am, o, bs, n) }
    #[kernel] pub unsafe fn bnb_bf16_fp4(c: *const f32, a: *const u8, am: *const f32, o: *mut B, bs: i32, n: i32) { bnb_dequant::<B, 1>(c, a, am, o, bs, n) }
    #[kernel] pub unsafe fn bnb_bf16_nf4(c: *const f32, a: *const u8, am: *const f32, o: *mut B, bs: i32, n: i32) { bnb_dequant::<B, 2>(c, a, am, o, bs, n) }

    // ------------------------------------------------------------------------------------------
    // fast-math primitives (exact PTX of nvcc --use_fast_math; ptxas lowering in brackets)

    /// `ex2.approx.ftz.f32` [MUFU.EX2].
    #[inline(always)]
    pub fn ex2(x: f32) -> f32 {
        float::ex2_approx_ftz_f32(x)
    }
    /// `rcp.approx.ftz.f32` [MUFU.RCP].
    #[inline(always)]
    pub fn rcp(x: f32) -> f32 {
        float::rcp_approx_ftz_f32(x)
    }
    /// `tanh.approx.f32` [MUFU.TANH].
    #[inline(always)]
    pub fn tanh_approx(x: f32) -> f32 {
        float::tanh_approx_f32(x)
    }
    /// fast-math `a / b` = `div.approx.ftz.f32` [MUFU.RCP b; FMUL.FTZ a, r].
    #[inline(always)]
    pub fn div_approx(a: f32, b: f32) -> f32 {
        mulf(a, rcp(b))
    }
    pub const LOG2E: f32 = f32::from_bits(0x3fb8aa3b);
    pub const NEG_LOG2E: f32 = f32::from_bits(0xbfb8aa3b);
    pub const SQRT_2_OVER_PI: f32 = f32::from_bits(0x3f4c422a);
    pub const GELU_KAPPA: f32 = f32::from_bits(0x3d372713);

    /// `x / (1 + expf(-x))` [FMUL.FTZ x,-log2e; EX2; FADD.FTZ +1; RCP; FMUL.FTZ].
    #[inline(always)]
    pub fn silu(x: f32) -> f32 {
        div_approx(x, addf(ex2(mulf(x, NEG_LOG2E)), 1.0))
    }
    /// `0.5f * x * (1 + tanhf(k0 * (x + k1 * x*x*x)))` [FMUL x*x; FMUL x*(x*x); FFMA x3*k1+x;
    /// FMUL *k0; TANH; FADD +1; FMUL (0.5x)*(1+t)].
    #[inline(always)]
    pub fn gelu_tanh(x: f32) -> f32 {
        let x3 = mulf(x, mulf(x, x));
        let inner = mulf(fmaf(x3, GELU_KAPPA, x), SQRT_2_OVER_PI);
        mulf(mulf(x, 0.5), addf(tanh_approx(inner), 1.0))
    }

    // ------------------------------------------------------------------------------------------
    // moe/gelu_tanh_and_mul.cu

    pub trait Half: Copy {
        fn to_f32(self) -> f32;
        fn from_f(v: f32) -> Self;
    }
    impl Half for H {
        #[inline(always)]
        fn to_f32(self) -> f32 { h2f(self.0) }
        #[inline(always)]
        fn from_f(v: f32) -> H { H(f2h(v)) }
    }
    impl Half for B {
        #[inline(always)]
        fn to_f32(self) -> f32 { bf2f(self.0) }
        #[inline(always)]
        fn from_f(v: f32) -> B { B(f2bf(v)) }
    }

    #[inline(always)]
    unsafe fn act_and_mul<T: Half, const ACT: u32>(out: *mut T, input: *const T, d: i32) {
        let bid = thread::blockIdx_x() as u64;
        let x_ptr = input.wrapping_add((bid * 2).wrapping_mul(d as i64 as u64) as usize);
        let y_ptr = x_ptr.wrapping_offset(d as isize);
        let out_ptr = out.wrapping_add(bid.wrapping_mul(d as i64 as u64) as usize);
        let mut idx = thread::threadIdx_x() as i32;
        while idx < d {
            let gate = (*x_ptr.offset(idx as isize)).to_f32();
            let up = (*y_ptr.offset(idx as isize)).to_f32();
            let act = if ACT == 0 { gelu_tanh(gate) } else { silu(gate) };
            *out_ptr.offset(idx as isize) = T::from_f(mulf(up, act));
            idx = idx.wrapping_add(thread::blockDim_x() as i32);
        }
    }
    #[kernel] pub unsafe fn gelu_tanh_and_mul_f16(out: *mut H, input: *const H, d: i32) { act_and_mul::<H, 0>(out, input, d) }
    #[kernel] pub unsafe fn gelu_tanh_and_mul_bf16(out: *mut B, input: *const B, d: i32) { act_and_mul::<B, 0>(out, input, d) }
    #[kernel] pub unsafe fn silu_and_mul_bf16(out: *mut B, input: *const B, d: i32) { act_and_mul::<B, 1>(out, input, d) }

    #[kernel]
    pub unsafe fn moe_sum_bf16(out: *mut B, input: *const B, num_tokens: i32, hidden: i32, topk: i32) {
        let token = thread::blockIdx_x() as i32;
        let col = thread::blockIdx_y().wrapping_mul(thread::blockDim_x()).wrapping_add(thread::threadIdx_x()) as i32;
        if token >= num_tokens || col >= hidden {
            return;
        }
        let base = (token as i64 as u64).wrapping_mul(topk as i64 as u64).wrapping_mul(hidden as i64 as u64).wrapping_add(col as i64 as u64);
        let mut acc = 0.0f32;
        let mut slot = 0i32;
        while slot < topk {
            acc = addf(acc, (*input.add(base.wrapping_add((slot as i64 as u64).wrapping_mul(hidden as i64 as u64)) as usize)).to_f32());
            slot += 1;
        }
        *out.add((token as i64 as u64).wrapping_mul(hidden as i64 as u64).wrapping_add(col as i64 as u64) as usize) = <B as Half>::from_f(acc);
    }

    // ------------------------------------------------------------------------------------------
    // rotary.cu (vLLM): x' = fma(x, cos, -(y*sin)) in the element type; y' = fma(y, cos, x*sin) for
    // f16/bf16 (one HMUL2 of (y,x)*sin, then two HFMA2) but fma(x, sin, y*cos) for f32.

    pub trait Rot: Copy {
        fn rot(x: Self, y: Self, c: Self, s: Self) -> (Self, Self);
    }
    impl Rot for f32 {
        #[inline(always)]
        fn rot(x: f32, y: f32, c: f32, s: f32) -> (f32, f32) {
            // ptxas contracts differently from the half path: y' = fma(x, sin, y*cos).
            let (ys, yc) = (mulf(y, s), mulf(y, c));
            (fmaf(x, c, -ys), fmaf(x, s, yc))
        }
    }
    impl Rot for H {
        #[inline(always)]
        fn rot(x: H, y: H, c: H, s: H) -> (H, H) {
            let (x, y, c, s) = (x.0 as u32, y.0 as u32, c.0 as u32, s.0 as u32);
            let p = f16x2::mul_f16x2(y | (x << 16), s | (s << 16));
            let nx = f16x2::fma_f16x2(x, c, (p & 0xFFFF) ^ 0x8000);
            let ny = f16x2::fma_f16x2(y, c, p >> 16);
            (H(nx as u16), H(ny as u16))
        }
    }
    impl Rot for B {
        #[inline(always)]
        fn rot(x: B, y: B, c: B, s: B) -> (B, B) {
            let (x, y, c, s) = (x.0 as u32, y.0 as u32, c.0 as u32, s.0 as u32);
            let p = bf16x2::mul_bf16x2(y | (x << 16), s | (s << 16));
            let nx = bf16x2::fma_bf16x2(x, c, (p & 0xFFFF) ^ 0x8000);
            let ny = bf16x2::fma_bf16x2(y, c, p >> 16);
            (B(nx as u16), B(ny as u16))
        }
    }

    #[inline(always)]
    unsafe fn apply_rotary<T: Rot, const NEOX: bool>(arr: *mut T, cos_ptr: *const T, sin_ptr: *const T, rot_offset: i32, rot_dim: i32) {
        let (xi, yi, ci) = if NEOX {
            (rot_offset, rot_dim.wrapping_add(rot_offset), rot_offset)
        } else {
            (2 * rot_offset, 2 * rot_offset + 1, rot_offset)
        };
        let (c, s) = (*cos_ptr.offset(ci as isize), *sin_ptr.offset(ci as isize));
        let (x, y) = (*arr.offset(xi as isize), *arr.offset(yi as isize));
        let (nx, ny) = T::rot(x, y, c, s);
        *arr.offset(xi as isize) = nx;
        *arr.offset(yi as isize) = ny;
    }

    #[inline(always)]
    unsafe fn rotary_heads<T: Rot, const NEOX: bool>(
        arr: *mut T, cos_ptr: *const T, sin_ptr: *const T, token_idx: i32, stride: i64, heads: i32, rot_dim: i32, head_size: i32,
    ) {
        let n = heads.wrapping_mul(rot_dim);
        let mut i = thread::threadIdx_x() as i32;
        while i < n {
            let head_idx = i.wrapping_div(rot_dim);
            let token_head = (token_idx as i64).wrapping_mul(stride).wrapping_add(head_idx.wrapping_mul(head_size) as i64);
            let rot_offset = i.wrapping_rem(rot_dim);
            apply_rotary::<T, NEOX>(arr.wrapping_offset(token_head as isize), cos_ptr, sin_ptr, rot_offset, rot_dim);
            i = i.wrapping_add(thread::blockDim_x() as i32);
        }
    }

    #[inline(always)]
    unsafe fn rotary<T: Rot, const NEOX: bool>(
        query: *mut T, key: *mut T, cos_cache: *const T, sin_cache: *const T, rot_dim: i32, query_stride: i64, key_stride: i64,
        num_heads: i32, num_kv_heads: i32, head_size: i32,
    ) {
        let token_idx = thread::blockIdx_x() as i32;
        let off = token_idx.wrapping_mul(rot_dim) as isize;
        let (cp, sp) = (cos_cache.wrapping_offset(off), sin_cache.wrapping_offset(off));
        rotary_heads::<T, NEOX>(query, cp, sp, token_idx, query_stride, num_heads, rot_dim, head_size);
        rotary_heads::<T, NEOX>(key, cp, sp, token_idx, key_stride, num_kv_heads, rot_dim, head_size);
    }

    #[inline(always)]
    unsafe fn rotary_pos<T: Rot, const NEOX: bool>(
        query: *mut T, key: *mut T, cos_cache: *const T, sin_cache: *const T, positions: *const u32, rot_dim: i32, _seq_len: i32,
        query_stride: i64, key_stride: i64, num_heads: i32, num_kv_heads: i32, head_size: i32,
    ) {
        let token_idx = thread::blockIdx_x() as i32;
        let position = *positions.offset(token_idx as isize);
        let off = position.wrapping_mul(rot_dim as u32) as usize;
        let (cp, sp) = (cos_cache.wrapping_add(off), sin_cache.wrapping_add(off));
        rotary_heads::<T, NEOX>(query, cp, sp, token_idx, query_stride, num_heads, rot_dim, head_size);
        rotary_heads::<T, NEOX>(key, cp, sp, token_idx, key_stride, num_kv_heads, rot_dim, head_size);
    }

    #[kernel] pub unsafe fn rotary_f16_neox(q: *mut H, k: *mut H, c: *const H, s: *const H, rd: i32, qs: i64, ks: i64, nh: i32, nkv: i32, hs: i32) { rotary::<H, true>(q, k, c, s, rd, qs, ks, nh, nkv, hs) }
    #[kernel] pub unsafe fn rotary_bf16_neox(q: *mut B, k: *mut B, c: *const B, s: *const B, rd: i32, qs: i64, ks: i64, nh: i32, nkv: i32, hs: i32) { rotary::<B, true>(q, k, c, s, rd, qs, ks, nh, nkv, hs) }
    #[kernel] pub unsafe fn rotary_f32_neox(q: *mut f32, k: *mut f32, c: *const f32, s: *const f32, rd: i32, qs: i64, ks: i64, nh: i32, nkv: i32, hs: i32) { rotary::<f32, true>(q, k, c, s, rd, qs, ks, nh, nkv, hs) }
    #[kernel] pub unsafe fn rotary_f16_gptj(q: *mut H, k: *mut H, c: *const H, s: *const H, rd: i32, qs: i64, ks: i64, nh: i32, nkv: i32, hs: i32) { rotary::<H, false>(q, k, c, s, rd, qs, ks, nh, nkv, hs) }
    #[kernel] pub unsafe fn rotary_bf16_gptj(q: *mut B, k: *mut B, c: *const B, s: *const B, rd: i32, qs: i64, ks: i64, nh: i32, nkv: i32, hs: i32) { rotary::<B, false>(q, k, c, s, rd, qs, ks, nh, nkv, hs) }
    #[kernel] pub unsafe fn rotary_f32_gptj(q: *mut f32, k: *mut f32, c: *const f32, s: *const f32, rd: i32, qs: i64, ks: i64, nh: i32, nkv: i32, hs: i32) { rotary::<f32, false>(q, k, c, s, rd, qs, ks, nh, nkv, hs) }
    #[kernel] pub unsafe fn rotary_pos_f16_neox(q: *mut H, k: *mut H, c: *const H, s: *const H, p: *const u32, rd: i32, sl: i32, qs: i64, ks: i64, nh: i32, nkv: i32, hs: i32) { rotary_pos::<H, true>(q, k, c, s, p, rd, sl, qs, ks, nh, nkv, hs) }
    #[kernel] pub unsafe fn rotary_pos_bf16_neox(q: *mut B, k: *mut B, c: *const B, s: *const B, p: *const u32, rd: i32, sl: i32, qs: i64, ks: i64, nh: i32, nkv: i32, hs: i32) { rotary_pos::<B, true>(q, k, c, s, p, rd, sl, qs, ks, nh, nkv, hs) }
    #[kernel] pub unsafe fn rotary_pos_f32_neox(q: *mut f32, k: *mut f32, c: *const f32, s: *const f32, p: *const u32, rd: i32, sl: i32, qs: i64, ks: i64, nh: i32, nkv: i32, hs: i32) { rotary_pos::<f32, true>(q, k, c, s, p, rd, sl, qs, ks, nh, nkv, hs) }
    #[kernel] pub unsafe fn rotary_pos_f16_gptj(q: *mut H, k: *mut H, c: *const H, s: *const H, p: *const u32, rd: i32, sl: i32, qs: i64, ks: i64, nh: i32, nkv: i32, hs: i32) { rotary_pos::<H, false>(q, k, c, s, p, rd, sl, qs, ks, nh, nkv, hs) }
    #[kernel] pub unsafe fn rotary_pos_bf16_gptj(q: *mut B, k: *mut B, c: *const B, s: *const B, p: *const u32, rd: i32, sl: i32, qs: i64, ks: i64, nh: i32, nkv: i32, hs: i32) { rotary_pos::<B, false>(q, k, c, s, p, rd, sl, qs, ks, nh, nkv, hs) }
    #[kernel] pub unsafe fn rotary_pos_f32_gptj(q: *mut f32, k: *mut f32, c: *const f32, s: *const f32, p: *const u32, rd: i32, sl: i32, qs: i64, ks: i64, nh: i32, nkv: i32, hs: i32) { rotary_pos::<f32, false>(q, k, c, s, p, rd, sl, qs, ks, nh, nkv, hs) }

    // ------------------------------------------------------------------------------------------
    // moe/moe_align.cu (vLLM)

    #[inline(always)]
    pub unsafe fn atomic_add_i32(p: *mut i32, v: i32) -> i32 {
        DeviceAtomicI32::from_ptr(p).fetch_add(v, AtomicOrdering::Relaxed)
    }

    /// Block-wide exclusive sum (blockDim a multiple of 32; e.g. cub::BlockScan<int, 1024>). Integer:
    /// any exact scan order gives the same result.
    #[inline(always)]
    pub unsafe fn block_exclusive_sum(v: i32, ws: *mut i32) -> i32 {
        let tid = thread::threadIdx_x();
        let lane = tid % 32;
        let wid = tid / 32;
        let mut inc = v;
        let mut d = 1;
        while d < 32 {
            let o = warp::shuffle_up_sync(0xffff_ffff, inc as u32, d) as i32;
            if lane >= d {
                inc = inc.wrapping_add(o);
            }
            d <<= 1;
        }
        if lane == 31 {
            *ws.add(wid as usize) = inc;
        }
        thread::sync_threads();
        if wid == 0 {
            let t = *ws.add(lane as usize);
            let mut ti = t;
            let mut d = 1;
            while d < 32 {
                let o = warp::shuffle_up_sync(0xffff_ffff, ti as u32, d) as i32;
                if lane >= d {
                    ti = ti.wrapping_add(o);
                }
                d <<= 1;
            }
            *ws.add(lane as usize) = ti.wrapping_sub(t); // exclusive warp prefix
        }
        thread::sync_threads();
        let r = (*ws.add(wid as usize)).wrapping_add(inc).wrapping_sub(v);
        thread::sync_threads();
        r
    }

    #[kernel]
    pub unsafe fn moe_align_block_size_kernel(
        topk_ids: *const i32, sorted_token_ids: *mut i32, expert_ids: *mut i32, total_tokens_post_pad: *mut i32,
        num_experts: i32, padded_num_experts: i32, experts_per_warp: i32, block_size: i32, numel: usize, cumsum: *mut i32,
        max_num_tokens_padded: i32,
    ) {
        static mut WS: SharedArray<i32, 32> = SharedArray::UNINIT;
        let ws = SharedArray::as_raw_mut_ptr(&raw mut WS);
        let counts = DynamicSharedArray::<i32>::get();
        let tid = thread::threadIdx_x();
        let bdim = thread::blockDim_x();
        let max_num_m_blocks = max_num_tokens_padded.wrapping_add(block_size).wrapping_sub(1).wrapping_div(block_size);
        if thread::blockIdx_x() % 2 != 0 {
            let mut it = tid as usize;
            while it < max_num_tokens_padded as isize as usize {
                *sorted_token_ids.add(it) = numel as i32;
                it = it.wrapping_add(bdim as usize);
            }
            return;
        }
        let warp_id = (tid / 32) as i32;
        let start = warp_id.wrapping_mul(experts_per_warp);
        let mut i = 0;
        while i < experts_per_warp {
            if start.wrapping_add(i) < padded_num_experts {
                *counts.offset(warp_id.wrapping_mul(experts_per_warp).wrapping_add(i) as isize) = 0;
            }
            i += 1;
        }
        thread::sync_threads();
        let mut i = tid as usize;
        while i < numel {
            let e = *topk_ids.add(i);
            if e < num_experts {
                let wi = e.wrapping_div(experts_per_warp);
                let off = e.wrapping_rem(experts_per_warp);
                atomic_add_i32(counts.offset(wi.wrapping_mul(experts_per_warp).wrapping_add(off) as isize), 1);
            }
            i = i.wrapping_add(bdim as usize);
        }
        thread::sync_threads();
        let expert_id = tid as i32;
        let mut expert_count = 0i32;
        if expert_id < num_experts {
            let wi = expert_id.wrapping_div(experts_per_warp);
            let off = expert_id.wrapping_rem(experts_per_warp);
            expert_count = *counts.offset(wi.wrapping_mul(experts_per_warp).wrapping_add(off) as isize);
            expert_count = expert_count.wrapping_add(block_size).wrapping_sub(1).wrapping_div(block_size).wrapping_mul(block_size);
        }
        let cumsum_val = block_exclusive_sum(expert_count, ws);
        if expert_id <= num_experts {
            *cumsum.offset(expert_id as isize) = cumsum_val;
        }
        if expert_id == num_experts {
            *total_tokens_post_pad = cumsum_val;
        }
        thread::sync_threads();
        if (tid as i32) < num_experts {
            let mut i = *cumsum.add(tid as usize);
            while i < *cumsum.add(tid as usize + 1) {
                *expert_ids.offset(i.wrapping_div(block_size) as isize) = tid as i32;
                i = i.wrapping_add(block_size);
            }
        }
        let fill_start = ((*cumsum.offset(num_experts as isize)).wrapping_div(block_size) as u32).wrapping_add(tid) as usize;
        let mut i = fill_start;
        while i < max_num_m_blocks as isize as usize {
            *expert_ids.add(i) = -1;
            i = i.wrapping_add(bdim as usize);
        }
    }

    #[kernel]
    pub unsafe fn count_and_sort_expert_tokens_kernel(topk_ids: *const i32, sorted_token_ids: *mut i32, cumsum_buffer: *mut i32, numel: usize, num_experts: i32) {
        let tid = thread::blockIdx_y().wrapping_mul(thread::blockDim_x()).wrapping_add(thread::threadIdx_x()) as usize;
        let stride = thread::blockDim_x().wrapping_mul(thread::gridDim_y()) as usize;
        let mut i = tid;
        while i < numel {
            let e = *topk_ids.add(i);
            if e < num_experts {
                let rank = atomic_add_i32(cumsum_buffer.offset(e as isize), 1);
                *sorted_token_ids.offset(rank as isize) = i as i32;
            }
            i = i.wrapping_add(stride);
        }
    }

    #[kernel]
    pub unsafe fn hunyuan_moe_capacity_mask_kernel(ids: *const u32, weights: *const f32, masked: *mut f32, n_tokens: i32, n_experts: i32, top_k: i32, capacity: i32) {
        let expert = thread::blockIdx_x() as i32;
        let priority = thread::blockIdx_y() as i32;
        if expert >= n_experts || priority >= top_k || thread::threadIdx_x() != 0 {
            return;
        }
        let mut accepted = 0i32;
        let mut pp = 0;
        while pp < priority {
            let mut t = 0;
            while t < n_tokens {
                if *ids.offset(t.wrapping_mul(top_k).wrapping_add(pp) as isize) == expert as u32 {
                    accepted += 1;
                }
                t += 1;
            }
            pp += 1;
        }
        let mut t = 0;
        while t < n_tokens {
            let route = t.wrapping_mul(top_k).wrapping_add(priority);
            if *ids.offset(route as isize) == expert as u32 && accepted < capacity {
                *masked.offset(route as isize) = *weights.offset(route as isize);
                accepted += 1;
            }
            t += 1;
        }
    }

    #[kernel]
    pub unsafe fn fill_f32_kernel(dst: *mut f32, value: f32, n: i32) {
        let idx = gid_x() as i32;
        if idx >= n {
            return;
        }
        *dst.offset(idx as isize) = value;
    }

    // ------------------------------------------------------------------------------------------
    // cutlass_moe/moe_data.cu (vLLM data preparation for the CUTLASS grouped GEMM)

    #[kernel]
    pub unsafe fn compute_problem_sizes_kernel(
        topk_ids: *const i32, ps1: *mut i32, ps2: *mut i32, atomic_buffer: *mut i32, topk_length: i32, n: i32, k: i32, is_gated: u8,
    ) {
        let expert_id = thread::blockIdx_x() as i32;
        let n1 = if is_gated != 0 { 2i32.wrapping_mul(n) } else { n };
        let mut occ = 0i32;
        let mut i = thread::threadIdx_x() as i32;
        while i < topk_length {
            occ += (*topk_ids.offset(i as isize) == expert_id) as i32;
            i = i.wrapping_add(512);
        }
        atomic_add_i32(atomic_buffer.offset(expert_id as isize), occ);
        thread::sync_threads();
        if thread::threadIdx_x() == 0 {
            let fin = *atomic_buffer.offset(expert_id as isize);
            let e3 = expert_id.wrapping_mul(3) as isize;
            *ps1.offset(e3) = fin;
            *ps1.offset(e3 + 1) = n1;
            *ps1.offset(e3 + 2) = k;
            *ps2.offset(e3) = fin;
            *ps2.offset(e3 + 1) = k;
            *ps2.offset(e3 + 2) = n;
        }
    }

    #[kernel]
    pub unsafe fn compute_expert_offsets_kernel(ps1: *const i32, expert_offsets: *mut i32, atomic_buffer: *mut i32, num_experts: i32) {
        let mut tot = 0i32;
        *expert_offsets = 0;
        let mut i = 0i32;
        while i < num_experts {
            *atomic_buffer.offset(i as isize) = tot;
            tot = tot.wrapping_add(*ps1.offset(i.wrapping_mul(3) as isize));
            *expert_offsets.offset(i as isize + 1) = tot;
            i += 1;
        }
    }

    #[kernel]
    pub unsafe fn compute_arg_sorts_kernel(topk_ids: *const i32, inp_perm: *mut i32, out_perm: *mut i32, atomic_buffer: *mut i32, topk_length: i32, topk: i32) {
        let e = thread::blockIdx_x() as i32;
        let mut i = thread::threadIdx_x() as i32;
        while i < topk_length {
            if *topk_ids.offset(i as isize) == e {
                let start = atomic_add_i32(atomic_buffer.offset(e as isize), 1);
                *inp_perm.offset(start as isize) = i.wrapping_div(topk);
                *out_perm.offset(i as isize) = start;
            }
            i = i.wrapping_add(512);
        }
    }

    #[kernel]
    pub unsafe fn gather_rows_bf16_kernel(dst: *mut u16, src: *const u16, map: *const i32, num_rows: i32, k: i32) {
        let row = thread::blockIdx_x() as i32;
        if row >= num_rows {
            return;
        }
        let src_row = src.wrapping_offset(((*map.offset(row as isize)) as i64).wrapping_mul(k as i64) as isize);
        let dst_row = dst.wrapping_offset((row as i64).wrapping_mul(k as i64) as isize);
        let mut i = thread::threadIdx_x() as i32;
        while i < k {
            *dst_row.offset(i as isize) = *src_row.offset(i as isize);
            i = i.wrapping_add(256);
        }
    }

    #[kernel]
    pub unsafe fn gather_weighted_bf16_kernel(out: *mut B, inp: *const B, out_perm: *const i32, weights: *const f32, num_rows: i32, k: i32) {
        let row = thread::blockIdx_x() as i32;
        if row >= num_rows {
            return;
        }
        let w = *weights.offset(row as isize);
        let in_row = inp.wrapping_offset(((*out_perm.offset(row as isize)) as i64).wrapping_mul(k as i64) as isize);
        let out_row = out.wrapping_offset((row as i64).wrapping_mul(k as i64) as isize);
        let mut i = thread::threadIdx_x() as i32;
        while i < k {
            *out_row.offset(i as isize) = <B as Half>::from_f(mulf(w, (*in_row.offset(i as isize)).to_f32()));
            i = i.wrapping_add(256);
        }
    }

    #[kernel]
    pub unsafe fn get_group_starts_bf16_kernel(
        expert_offsets: *const i32, a_base: u64, b_base: u64, d_base: u64, a_ptrs: *mut u64, b_ptrs: *mut u64, d_ptrs: *mut u64,
        lda: *mut i64, ldb: *mut i64, ldd: *mut i64, n: i64, k: i64,
    ) {
        let e = thread::threadIdx_x() as i32;
        let off = *expert_offsets.offset(e as isize) as i64;
        let ei = e as isize;
        *a_ptrs.offset(ei) = a_base.wrapping_add((off.wrapping_mul(k) as u64).wrapping_mul(2));
        *b_ptrs.offset(ei) = b_base.wrapping_add(((e as i64).wrapping_mul(n).wrapping_mul(k) as u64).wrapping_mul(2));
        *d_ptrs.offset(ei) = d_base.wrapping_add((off.wrapping_mul(n) as u64).wrapping_mul(2));
        *lda.offset(ei) = k;
        *ldb.offset(ei) = k;
        *ldd.offset(ei) = n;
    }

    // ------------------------------------------------------------------------------------------
    // ops/ops.cu

    /// `min.ftz.f32` / `max.ftz.f32` [FMNMX.FTZ]: fminf/fmaxf under fast-math.
    #[inline(always)]
    pub fn fminf(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { cuda_device::ptx_asm!("min.ftz.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn fmaxf(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { cuda_device::ptx_asm!("max.ftz.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    /// `cvt.rzi.f32.f32` [FRND.TRUNC].
    #[inline(always)]
    pub fn truncf(a: f32) -> f32 {
        let r: f32;
        unsafe { cuda_device::ptx_asm!("cvt.rzi.f32.f32 %0, %1;", out("=f") r, in("f") a, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn copysignf(mag: f32, sgn: f32) -> f32 {
        f32::from_bits((mag.to_bits() & 0x7fff_ffff) | (sgn.to_bits() & 0x8000_0000))
    }
    #[inline(always)]
    pub fn absf(a: f32) -> f32 {
        f32::from_bits(a.to_bits() & 0x7fff_ffff)
    }
    #[inline(always)]
    pub fn cf(b: u32) -> f32 {
        f32::from_bits(b)
    }

    /// libdevice `normcdff` as nvcc inlines it under --use_fast_math (FTZ reflect), transcribed
    /// instruction by instruction from the SASS of fused_glu_kernel<float> (GLU_GELU_ERF).
    #[inline(always)]
    pub fn normcdf_ftz(x: f32) -> f32 {
        let r8 = if absf(x) > 14.5 { copysignf(14.5, x) } else { x };
        let r3 = mulf(r8, cf(0xbf3504f3));
        let p1 = !(r8 < -1.0);
        let r5 = fmaf(r8, cf(0xbf3504f3), -r3);
        let mut r6 = fmaf(r8, cf(0xb24fe77a), r5);
        let r4 = addf(r3, r6);
        let a = absf(r4);
        let r7 = addf(a, 4.0);
        let r5b = addf(a, -4.0);
        let r10 = addf(a, -0.0);
        let p2 = !(r4 < 0.0);
        let r12 = rcp(r7);
        if !p1 {
            let r3b = addf(r3, -r4);
            r6 = addf(r6, r3b);
        }
        let r7b = fmaf(a, 2.0, 1.0);
        let r9 = mulf(r5b, r12);
        let r5c = mulf(a, -r10);
        let mut r11 = addf(r9, 1.0);
        let r13 = mulf(r5c, LOG2E);
        r11 = fmaf(r11, -4.0, a);
        let r13 = truncf(r13);
        r11 = fmaf(a, -r9, r11);
        r11 = fmaf(r12, r11, r9);
        let mut q = fmaf(r11, cf(0x3a69a091), cf(0x3be6e05b));
        q = fmaf(r11, q, cf(0xbc81fb4b));
        let r16 = if absf(r13) > 126.0 { copysignf(126.0, r13) } else { r13 };
        q = fmaf(r11, q, cf(0x3d15373b));
        q = fmaf(r11, q, cf(0xbd887c5a));
        let big = a > cf(0x4120e148);
        let r14 = rcp(r7b);
        q = fmaf(r11, q, cf(0x3dc021d5));
        let r9b = fmaf(r16, cf(0xbf317218), r5c);
        let r5d = fmaf(a, -r10, -r5c);
        q = fmaf(r11, q, cf(0xbdced424));
        let r9c = fmaf(r16, cf(0x3102e308), r9b);
        let r16b = addf(r16, cf(0x4b40007f));
        q = fmaf(r11, q, cf(0x3d8b74de));
        let r13b = mulf(r9c, LOG2E);
        q = fmaf(r11, q, cf(0x3c7bf170));
        let r16c = f32::from_bits(r16b.to_bits() << 23);
        q = fmaf(r11, q, cf(0xbe0ef8d4));
        let r13c = ex2(r13b);
        let r11b = fmaf(r11, q, cf(0x3f9dd2c9));
        let r9d = mulf(r11b, r14);
        let r12b = mulf(r9d, -2.0);
        let r12c = fmaf(a, r12b, r11b);
        let r4m = if !p1 { mulf(r4, -2.0) } else { r4 };
        let r16d = mulf(r16c, r13c);
        let r12d = addf(-r9d, r12c);
        let r5e = fmaf(r16d, r5d, r16d);
        let r12e = fmaf(r14, r12d, r9d);
        let mut r = mulf(r12e, r5e);
        if big {
            r = 0.0;
        }
        if !p2 {
            r = addf(-r, 2.0);
        }
        if !p1 {
            let t = mulf(r, r4m);
            r = fmaf(r6, t, r);
        }
        mulf(r, 0.5)
    }

    /// apply_glu_activation: 0 SiLU, 1 GELU (tanh), 2 ReLU, 3 GELU (erf), anything else SiLU.
    #[inline(always)]
    pub fn glu_act(x: f32, act: i32) -> f32 {
        match act {
            1 => gelu_tanh(x),
            2 => fmaxf(0.0, x),
            3 => mulf(x, normcdf_ftz(x)),
            _ => silu(x),
        }
    }

    /// A value type of ops.cu: f32 or a 16-bit float (loaded via exact cvt, stored via cvt.rn).
    pub trait Val: Copy {
        fn f(self) -> f32;
        fn t(v: f32) -> Self;
        /// `T activated * T b` in T (FMUL.FTZ / HMUL2 / HMUL2.BF16).
        fn tmul(a: Self, b: Self) -> Self;
    }
    impl Val for f32 {
        #[inline(always)] fn f(self) -> f32 { self }
        #[inline(always)] fn t(v: f32) -> f32 { v }
        #[inline(always)] fn tmul(a: f32, b: f32) -> f32 { mulf(a, b) }
    }
    impl Val for H {
        #[inline(always)] fn f(self) -> f32 { h2f(self.0) }
        #[inline(always)] fn t(v: f32) -> H { H(f2h(v)) }
        #[inline(always)] fn tmul(a: H, b: H) -> H { H(f16x2::mul_f16x2(a.0 as u32, b.0 as u32) as u16) }
    }
    impl Val for B {
        #[inline(always)] fn f(self) -> f32 { bf2f(self.0) }
        #[inline(always)] fn t(v: f32) -> B { B(f2bf(v)) }
        #[inline(always)] fn tmul(a: B, b: B) -> B { B(bf16x2::mul_bf16x2(a.0 as u32, b.0 as u32) as u16) }
    }

    /// gptoss SwiGLU on one element: min.ftz / max.ftz clamps, sigmoid = rcp(1 + ex2(-a*g*log2e)).
    #[inline(always)]
    pub fn swiglu(g: f32, u: f32, alpha: f32, limit: f32) -> f32 {
        let gc = fminf(g, limit);
        let uc = fmaxf(fminf(u, limit), -limit);
        let sig = rcp(addf(ex2(mulf(mulf(gc, alpha), NEG_LOG2E)), 1.0));
        mulf(mulf(gc, sig), addf(uc, 1.0))
    }

    #[inline(always)]
    unsafe fn gptoss_swiglu<T: Val>(gate: *const T, up: *const T, out: *mut T, n: u32, alpha: f32, limit: f32) {
        let idx = gid_x() as i32;
        if idx as u32 >= n {
            return;
        }
        let r = swiglu((*gate.offset(idx as isize)).f(), (*up.offset(idx as isize)).f(), alpha, limit);
        *out.offset(idx as isize) = T::t(r);
    }
    /// The vec4 kernels: four consecutive elements per thread, same per-element program.
    #[inline(always)]
    unsafe fn gptoss_swiglu4<T: Val>(gate: *const T, up: *const T, out: *mut T, n4: u32, alpha: f32, limit: f32) {
        let idx = gid_x() as i32;
        if idx as u32 >= n4 {
            return;
        }
        let base = (idx as isize) * 4;
        let mut i = 0;
        while i < 4 {
            let r = swiglu((*gate.offset(base + i)).f(), (*up.offset(base + i)).f(), alpha, limit);
            *out.offset(base + i) = T::t(r);
            i += 1;
        }
    }
    #[kernel] pub unsafe fn gptoss_swiglu_f32k(g: *const f32, u: *const f32, o: *mut f32, n: u32, a: f32, l: f32) { gptoss_swiglu(g, u, o, n, a, l) }
    #[kernel] pub unsafe fn gptoss_swiglu_f16k(g: *const H, u: *const H, o: *mut H, n: u32, a: f32, l: f32) { gptoss_swiglu(g, u, o, n, a, l) }
    #[kernel] pub unsafe fn gptoss_swiglu_bf16k(g: *const B, u: *const B, o: *mut B, n: u32, a: f32, l: f32) { gptoss_swiglu(g, u, o, n, a, l) }
    #[kernel] pub unsafe fn gptoss_swiglu4_f32k(g: *const f32, u: *const f32, o: *mut f32, n: u32, a: f32, l: f32) { gptoss_swiglu4(g, u, o, n, a, l) }
    #[kernel] pub unsafe fn gptoss_swiglu4_f16k(g: *const H, u: *const H, o: *mut H, n: u32, a: f32, l: f32) { gptoss_swiglu4(g, u, o, n, a, l) }
    #[kernel] pub unsafe fn gptoss_swiglu4_bf16k(g: *const B, u: *const B, o: *mut B, n: u32, a: f32, l: f32) { gptoss_swiglu4(g, u, o, n, a, l) }

    #[inline(always)]
    unsafe fn gptoss_swiglu_il<T: Val>(gate_up: *const T, out: *mut T, n: u32, isz: u32, alpha: f32, limit: f32) {
        let idx = gid_x() as i32;
        let total = n.wrapping_mul(isz);
        if idx as u32 >= total {
            return;
        }
        let nn = (idx as u32) / isz;
        let i = (idx as u32) % isz;
        let base = (nn.wrapping_mul(isz).wrapping_add(i).wrapping_mul(2)) as i32;
        let r = swiglu((*gate_up.offset(base as isize)).f(), (*gate_up.offset(base as isize + 1)).f(), alpha, limit);
        *out.offset(idx as isize) = T::t(r);
    }
    #[kernel] pub unsafe fn gptoss_swiglu_il_f32k(g: *const f32, o: *mut f32, n: u32, i: u32, a: f32, l: f32) { gptoss_swiglu_il(g, o, n, i, a, l) }
    #[kernel] pub unsafe fn gptoss_swiglu_il_f16k(g: *const H, o: *mut H, n: u32, i: u32, a: f32, l: f32) { gptoss_swiglu_il(g, o, n, i, a, l) }
    #[kernel] pub unsafe fn gptoss_swiglu_il_bf16k(g: *const B, o: *mut B, n: u32, i: u32, a: f32, l: f32) { gptoss_swiglu_il(g, o, n, i, a, l) }

    #[inline(always)]
    unsafe fn fused_glu<T: Val>(a: *const T, b: *const T, out: *mut T, n: u32, act: i32) {
        let idx = gid_x() as i32;
        if idx as u32 >= n {
            return;
        }
        let activated = T::t(glu_act((*a.offset(idx as isize)).f(), act));
        *out.offset(idx as isize) = T::tmul(activated, *b.offset(idx as isize));
    }
    #[inline(always)]
    unsafe fn fused_glu4<T: Val>(a: *const T, b: *const T, out: *mut T, n4: u32, act: i32) {
        let idx = gid_x() as i32;
        if idx as u32 >= n4 {
            return;
        }
        let base = (idx as isize) * 4;
        let mut i = 0;
        while i < 4 {
            let activated = T::t(glu_act((*a.offset(base + i)).f(), act));
            *out.offset(base + i) = T::tmul(activated, *b.offset(base + i));
            i += 1;
        }
    }
    #[kernel] pub unsafe fn fused_glu_f32k(a: *const f32, b: *const f32, o: *mut f32, n: u32, act: i32) { fused_glu(a, b, o, n, act) }
    #[kernel] pub unsafe fn fused_glu_f16k(a: *const H, b: *const H, o: *mut H, n: u32, act: i32) { fused_glu(a, b, o, n, act) }
    #[kernel] pub unsafe fn fused_glu_bf16k(a: *const B, b: *const B, o: *mut B, n: u32, act: i32) { fused_glu(a, b, o, n, act) }
    #[kernel] pub unsafe fn fused_glu4_f32k(a: *const f32, b: *const f32, o: *mut f32, n: u32, act: i32) { fused_glu4(a, b, o, n, act) }
    #[kernel] pub unsafe fn fused_glu4_f16k(a: *const H, b: *const H, o: *mut H, n: u32, act: i32) { fused_glu4(a, b, o, n, act) }
    #[kernel] pub unsafe fn fused_glu4_bf16k(a: *const B, b: *const B, o: *mut B, n: u32, act: i32) { fused_glu4(a, b, o, n, act) }

    /// softcap: tanhf(x / cap) * cap = tanh.approx(x * rcp(cap)) * cap.
    #[inline(always)]
    unsafe fn softcap<T: Val>(input: *const T, out: *mut f32, n: u32, cap: f32) {
        let idx = gid_x() as i32;
        if idx as u32 >= n {
            return;
        }
        let v = (*input.offset(idx as isize)).f();
        *out.offset(idx as isize) = mulf(tanh_approx(div_approx(v, cap)), cap);
    }
    #[kernel] pub unsafe fn softcap_f32k(i: *const f32, o: *mut f32, n: u32, cap: f32) { softcap(i, o, n, cap) }
    #[kernel] pub unsafe fn softcap_f16k(i: *const H, o: *mut f32, n: u32, cap: f32) { softcap(i, o, n, cap) }
    #[kernel] pub unsafe fn softcap_bf16k(i: *const B, o: *mut f32, n: u32, cap: f32) { softcap(i, o, n, cap) }

    /// warp_reduce_{max,sum} of ops.cu: shfl.down 16, 8, 4, 2, 1 (lanes past the end keep their value).
    #[inline(always)]
    pub fn warp_down_max(mut v: f32) -> f32 {
        let mut off = 16;
        while off > 0 {
            v = fmaxf(v, warp::shuffle_down_f32_sync(0xffff_ffff, v, off));
            off /= 2;
        }
        v
    }
    #[inline(always)]
    pub fn warp_down_sum(mut v: f32) -> f32 {
        let mut off = 16;
        while off > 0 {
            v = addf(v, warp::shuffle_down_f32_sync(0xffff_ffff, v, off));
            off /= 2;
        }
        v
    }

    #[inline(always)]
    unsafe fn softmax_with_sinks<T: Val>(
        logits: *const T, sinks: *const T, mask: *const T, output: *mut T, batch: i32, heads: i32, q_len: i32, k_len: i32,
    ) {
        static mut RED: SharedArray<f32, 34> = SharedArray::UNINIT;
        let red = SharedArray::as_raw_mut_ptr(&raw mut RED);
        let row = thread::blockIdx_x() as i32;
        let total = batch.wrapping_mul(heads).wrapping_mul(q_len);
        if row >= total {
            return;
        }
        let b = row.wrapping_div(heads.wrapping_mul(q_len));
        let h = row.wrapping_div(q_len).wrapping_rem(heads);
        let q = row.wrapping_rem(q_len);
        let off = b.wrapping_mul(heads).wrapping_add(h).wrapping_mul(q_len).wrapping_add(q).wrapping_mul(k_len);
        let rl = logits.wrapping_offset(off as isize);
        let ro = output.wrapping_offset(off as isize);
        let sink = (*sinks.offset(h as isize)).f();
        let moff = b.wrapping_mul(q_len).wrapping_add(q).wrapping_mul(k_len);
        let rm = if mask.is_null() { mask } else { mask.wrapping_offset(moff as isize) };
        let tid = thread::threadIdx_x() as i32;
        let bs = thread::blockDim_x() as i32;
        let load = |k: i32| -> f32 {
            let mut v = (*rl.offset(k as isize)).f();
            if !rm.is_null() {
                v = addf(v, (*rm.offset(k as isize)).f());
            }
            v
        };
        let mut lmax = f32::NEG_INFINITY;
        let mut k = tid;
        while k < k_len {
            lmax = fmaxf(lmax, load(k));
            k += bs;
        }
        if tid == 0 {
            lmax = fmaxf(lmax, sink);
        }
        lmax = warp_down_max(lmax);
        let nw = (bs + 31) / 32;
        if bs > 32 {
            if tid % 32 == 0 {
                *red.add((tid / 32) as usize) = lmax;
            }
            thread::sync_threads();
            if tid < 32 {
                lmax = if tid < nw { *red.add(tid as usize) } else { f32::NEG_INFINITY };
                lmax = warp_down_max(lmax);
            }
        }
        if tid == 0 {
            *red.add(32) = lmax;
        }
        thread::sync_threads();
        let row_max = *red.add(32);
        let mut lsum = 0.0f32;
        let mut k = tid;
        while k < k_len {
            lsum = addf(lsum, ex2(mulf(subf(load(k), row_max), LOG2E)));
            k += bs;
        }
        if tid == 0 {
            lsum = addf(lsum, ex2(mulf(subf(sink, row_max), LOG2E)));
        }
        lsum = warp_down_sum(lsum);
        if bs > 32 {
            // warp_sums is a separate __shared__ array in the reference; the max pass is finished
            // with red[0..32) (every thread read row_max after the barrier), but a warp may still
            // be reading red[tid] for the max: sync first.
            thread::sync_threads();
            if tid % 32 == 0 {
                *red.add((tid / 32) as usize) = lsum;
            }
            thread::sync_threads();
            if tid < 32 {
                lsum = if tid < nw { *red.add(tid as usize) } else { 0.0 };
                lsum = warp_down_sum(lsum);
            }
        }
        if tid == 0 {
            *red.add(33) = lsum;
        }
        thread::sync_threads();
        let inv = rcp(*red.add(33));
        let mut k = tid;
        while k < k_len {
            *ro.offset(k as isize) = T::t(mulf(ex2(mulf(subf(load(k), row_max), LOG2E)), inv));
            k += bs;
        }
    }
    #[kernel] pub unsafe fn softmax_with_sinks_f32k(l: *const f32, s: *const f32, m: *const f32, o: *mut f32, b: i32, h: i32, q: i32, k: i32, _scale: f32) { softmax_with_sinks(l, s, m, o, b, h, q, k) }
    #[kernel] pub unsafe fn softmax_with_sinks_f16k(l: *const H, s: *const H, m: *const H, o: *mut H, b: i32, h: i32, q: i32, k: i32, _scale: f32) { softmax_with_sinks(l, s, m, o, b, h, q, k) }
    #[kernel] pub unsafe fn softmax_with_sinks_bf16k(l: *const B, s: *const B, m: *const B, o: *mut B, b: i32, h: i32, q: i32, k: i32, _scale: f32) { softmax_with_sinks(l, s, m, o, b, h, q, k) }

    // bitwise / leftshift (integer; shift counts >= the width give 0 as PTX shl does)
    pub trait Bits: Copy {
        fn and(a: Self, b: Self) -> Self;
        fn or(a: Self, b: Self) -> Self;
        fn xor(a: Self, b: Self) -> Self;
        fn shl(a: Self, k: i32) -> Self;
    }
    macro_rules! bits32 {
        ($($t:ty),*) => {$(
            impl Bits for $t {
                #[inline(always)] fn and(a: $t, b: $t) -> $t { a & b }
                #[inline(always)] fn or(a: $t, b: $t) -> $t { a | b }
                #[inline(always)] fn xor(a: $t, b: $t) -> $t { a ^ b }
                #[inline(always)] fn shl(a: $t, k: i32) -> $t {
                    // C promotes to int; PTX shl.b32 clamps the count (k as unsigned >= 32 -> 0).
                    let v = a as u32;
                    (if (k as u32) >= 32 { 0 } else { v << k }) as $t
                }
            }
        )*};
    }
    bits32!(u8, u32, i32);
    impl Bits for i64 {
        #[inline(always)] fn and(a: i64, b: i64) -> i64 { a & b }
        #[inline(always)] fn or(a: i64, b: i64) -> i64 { a | b }
        #[inline(always)] fn xor(a: i64, b: i64) -> i64 { a ^ b }
        #[inline(always)] fn shl(a: i64, k: i32) -> i64 { if (k as u32) >= 64 { 0 } else { a << k } }
    }
    #[inline(always)]
    unsafe fn bitwise<T: Bits, const OP: u32>(a: *const T, b: *const T, o: *mut T, n: u32) {
        let idx = gid_x() as i32;
        if (idx as u32) < n {
            let (x, y) = (*a.offset(idx as isize), *b.offset(idx as isize));
            *o.offset(idx as isize) = match OP { 0 => T::and(x, y), 1 => T::or(x, y), _ => T::xor(x, y) };
        }
    }
    #[inline(always)]
    unsafe fn leftshift<T: Bits>(a: *const T, o: *mut T, n: u32, k: i32) {
        let idx = gid_x() as i32;
        if (idx as u32) < n {
            *o.offset(idx as isize) = T::shl(*a.offset(idx as isize), k);
        }
    }
    #[kernel] pub unsafe fn bitwise_and_u8k(a: *const u8, b: *const u8, o: *mut u8, n: u32) { bitwise::<u8, 0>(a, b, o, n) }
    #[kernel] pub unsafe fn bitwise_or_u8k(a: *const u8, b: *const u8, o: *mut u8, n: u32) { bitwise::<u8, 1>(a, b, o, n) }
    #[kernel] pub unsafe fn bitwise_xor_u8k(a: *const u8, b: *const u8, o: *mut u8, n: u32) { bitwise::<u8, 2>(a, b, o, n) }
    #[kernel] pub unsafe fn bitwise_and_u32k(a: *const u32, b: *const u32, o: *mut u32, n: u32) { bitwise::<u32, 0>(a, b, o, n) }
    #[kernel] pub unsafe fn bitwise_or_u32k(a: *const u32, b: *const u32, o: *mut u32, n: u32) { bitwise::<u32, 1>(a, b, o, n) }
    #[kernel] pub unsafe fn bitwise_xor_u32k(a: *const u32, b: *const u32, o: *mut u32, n: u32) { bitwise::<u32, 2>(a, b, o, n) }
    #[kernel] pub unsafe fn bitwise_and_i32k(a: *const i32, b: *const i32, o: *mut i32, n: u32) { bitwise::<i32, 0>(a, b, o, n) }
    #[kernel] pub unsafe fn bitwise_or_i32k(a: *const i32, b: *const i32, o: *mut i32, n: u32) { bitwise::<i32, 1>(a, b, o, n) }
    #[kernel] pub unsafe fn bitwise_xor_i32k(a: *const i32, b: *const i32, o: *mut i32, n: u32) { bitwise::<i32, 2>(a, b, o, n) }
    #[kernel] pub unsafe fn bitwise_and_i64k(a: *const i64, b: *const i64, o: *mut i64, n: u32) { bitwise::<i64, 0>(a, b, o, n) }
    #[kernel] pub unsafe fn bitwise_or_i64k(a: *const i64, b: *const i64, o: *mut i64, n: u32) { bitwise::<i64, 1>(a, b, o, n) }
    #[kernel] pub unsafe fn bitwise_xor_i64k(a: *const i64, b: *const i64, o: *mut i64, n: u32) { bitwise::<i64, 2>(a, b, o, n) }
    #[kernel] pub unsafe fn leftshift_u8k(a: *const u8, o: *mut u8, n: u32, k: i32) { leftshift(a, o, n, k) }
    #[kernel] pub unsafe fn leftshift_u32k(a: *const u32, o: *mut u32, n: u32, k: i32) { leftshift(a, o, n, k) }
    #[kernel] pub unsafe fn leftshift_i32k(a: *const i32, o: *mut i32, n: u32, k: i32) { leftshift(a, o, n, k) }
    #[kernel] pub unsafe fn leftshift_i64k(a: *const i64, o: *mut i64, n: u32, k: i32) { leftshift(a, o, n, k) }

    // count_nonzero / nonzero (cub DeviceReduce::Sum / DeviceSelect::Flagged in the reference; the
    // results are integers, so any exact algorithm matches). NonZeroOp: `a != T(0)`; for f32 that
    // is setp.neu.ftz (denormals count as zero, NaN as non-zero), for f64 setp.neu.
    pub trait Nz: Copy {
        fn nz(self) -> bool;
    }
    impl Nz for f32 {
        #[inline(always)] fn nz(self) -> bool { self.to_bits() & 0x7f80_0000 != 0 }
    }
    impl Nz for f64 {
        #[inline(always)] fn nz(self) -> bool { self != 0.0 }
    }
    macro_rules! nz_int { ($($t:ty),*) => {$( impl Nz for $t { #[inline(always)] fn nz(self) -> bool { self != 0 } } )*}; }
    nz_int!(u8, u32, i16, i32, i64);

    pub const NZ_TILE: u32 = 2048; // 256 threads x 8 consecutive elements

    #[inline(always)]
    unsafe fn count_nz<T: Nz>(d_in: *const T, n: u32, count: *mut u32) {
        let mut c = 0u32;
        let mut i = gid_x();
        let step = thread::blockDim_x().wrapping_mul(thread::gridDim_x());
        while i < n {
            c += (*d_in.add(i as usize)).nz() as u32;
            i = i.wrapping_add(step);
        }
        let mut off = 16;
        while off > 0 {
            c = c.wrapping_add(warp::shuffle_down_sync(0xffff_ffff, c, off));
            off /= 2;
        }
        if thread::threadIdx_x() % 32 == 0 && c != 0 {
            DeviceAtomicU32::from_ptr(count).fetch_add(c, AtomicOrdering::Relaxed);
        }
    }
    /// Per-tile counts of non-zero elements.
    #[inline(always)]
    unsafe fn tile_count_nz<T: Nz>(d_in: *const T, n: u32, counts: *mut i32) {
        static mut WS: SharedArray<i32, 32> = SharedArray::UNINIT;
        let ws = SharedArray::as_raw_mut_ptr(&raw mut WS);
        let base = thread::blockIdx_x() * NZ_TILE + thread::threadIdx_x() * 8;
        let mut c = 0i32;
        let mut j = 0;
        while j < 8 {
            let i = base + j;
            if i < n && (*d_in.add(i as usize)).nz() {
                c += 1;
            }
            j += 1;
        }
        let ex = block_exclusive_sum(c, ws);
        if thread::threadIdx_x() == thread::blockDim_x() - 1 {
            *counts.add(thread::blockIdx_x() as usize) = ex + c;
        }
    }
    /// Scatter the indices of non-zero elements in order: out[offset[tile] + rank] = index.
    #[inline(always)]
    unsafe fn tile_select_nz<T: Nz>(d_in: *const T, n: u32, offsets: *const i32, out: *mut u32, cap: u32) {
        static mut WS: SharedArray<i32, 32> = SharedArray::UNINIT;
        let ws = SharedArray::as_raw_mut_ptr(&raw mut WS);
        let base = thread::blockIdx_x() * NZ_TILE + thread::threadIdx_x() * 8;
        let mut c = 0i32;
        let mut j = 0;
        while j < 8 {
            let i = base + j;
            if i < n && (*d_in.add(i as usize)).nz() {
                c += 1;
            }
            j += 1;
        }
        let mut pos = (*offsets.add(thread::blockIdx_x() as usize) + block_exclusive_sum(c, ws)) as u32;
        let mut j = 0;
        while j < 8 {
            let i = base + j;
            if i < n && (*d_in.add(i as usize)).nz() {
                if pos < cap {
                    *out.add(pos as usize) = i;
                }
                pos += 1;
            }
            j += 1;
        }
    }
    /// Exclusive scan of the tile counts (one 1024-thread block, running carry).
    #[kernel]
    pub unsafe fn nz_scan_tiles(counts: *const i32, offsets: *mut i32, ntiles: u32) {
        static mut WS: SharedArray<i32, 32> = SharedArray::UNINIT;
        static mut CARRY: SharedArray<i32, 1> = SharedArray::UNINIT;
        let ws = SharedArray::as_raw_mut_ptr(&raw mut WS);
        let carry = SharedArray::as_raw_mut_ptr(&raw mut CARRY);
        let tid = thread::threadIdx_x();
        if tid == 0 {
            *carry = 0;
        }
        thread::sync_threads();
        let mut base = 0u32;
        while base < ntiles {
            let i = base + tid;
            let v = if i < ntiles { *counts.add(i as usize) } else { 0 };
            let ex = block_exclusive_sum(v, ws);
            let c = *carry;
            if i < ntiles {
                *offsets.add(i as usize) = c + ex;
            }
            thread::sync_threads();
            if tid == thread::blockDim_x() - 1 {
                *carry = c + ex + v;
            }
            thread::sync_threads();
            base += thread::blockDim_x();
        }
    }
    /// ops.cu transform_indices (int temp_index, unsigned % and / by dims[i]).
    #[kernel]
    pub unsafe fn transform_indices(temp: *const u32, num_nonzero: u32, dims: *const u32, num_dims: u32, d_out: *mut u32) {
        let idx = gid_x() as i32;
        if (idx as u32) < num_nonzero {
            let mut t = *temp.offset(idx as isize) as i32;
            let mut i = num_dims as i32 - 1;
            while i >= 0 {
                let d = *dims.offset(i as isize);
                let o = (idx as u32).wrapping_mul(num_dims).wrapping_add(i as u32);
                *d_out.add(o as usize) = rem_u32(t as u32, d);
                if d != 0 {
                    t = ((t as u32) / d) as i32;
                }
                i -= 1;
            }
        }
    }
    /// `rem.u32` (defined for a zero divisor the way the hardware sequence computes it).
    #[inline(always)]
    pub fn rem_u32(a: u32, b: u32) -> u32 {
        let r: u32;
        unsafe { cuda_device::ptx_asm!("rem.u32 %0, %1, %2;", out("=r") r, in("r") a, in("r") b, options(register_only)); }
        r
    }
    #[kernel] pub unsafe fn count_nz_f32(d: *const f32, n: u32, c: *mut u32) { count_nz(d, n, c) }
    #[kernel] pub unsafe fn count_nz_f64(d: *const f64, n: u32, c: *mut u32) { count_nz(d, n, c) }
    #[kernel] pub unsafe fn count_nz_u8(d: *const u8, n: u32, c: *mut u32) { count_nz(d, n, c) }
    #[kernel] pub unsafe fn count_nz_u32(d: *const u32, n: u32, c: *mut u32) { count_nz(d, n, c) }
    #[kernel] pub unsafe fn count_nz_i16(d: *const i16, n: u32, c: *mut u32) { count_nz(d, n, c) }
    #[kernel] pub unsafe fn count_nz_i32(d: *const i32, n: u32, c: *mut u32) { count_nz(d, n, c) }
    #[kernel] pub unsafe fn count_nz_i64(d: *const i64, n: u32, c: *mut u32) { count_nz(d, n, c) }
    #[kernel] pub unsafe fn tile_count_nz_f32(d: *const f32, n: u32, c: *mut i32) { tile_count_nz(d, n, c) }
    #[kernel] pub unsafe fn tile_count_nz_f64(d: *const f64, n: u32, c: *mut i32) { tile_count_nz(d, n, c) }
    #[kernel] pub unsafe fn tile_count_nz_u8(d: *const u8, n: u32, c: *mut i32) { tile_count_nz(d, n, c) }
    #[kernel] pub unsafe fn tile_count_nz_u32(d: *const u32, n: u32, c: *mut i32) { tile_count_nz(d, n, c) }
    #[kernel] pub unsafe fn tile_count_nz_i16(d: *const i16, n: u32, c: *mut i32) { tile_count_nz(d, n, c) }
    #[kernel] pub unsafe fn tile_count_nz_i32(d: *const i32, n: u32, c: *mut i32) { tile_count_nz(d, n, c) }
    #[kernel] pub unsafe fn tile_count_nz_i64(d: *const i64, n: u32, c: *mut i32) { tile_count_nz(d, n, c) }
    #[kernel] pub unsafe fn tile_select_nz_f32(d: *const f32, n: u32, off: *const i32, o: *mut u32, cap: u32) { tile_select_nz(d, n, off, o, cap) }
    #[kernel] pub unsafe fn tile_select_nz_f64(d: *const f64, n: u32, off: *const i32, o: *mut u32, cap: u32) { tile_select_nz(d, n, off, o, cap) }
    #[kernel] pub unsafe fn tile_select_nz_u8(d: *const u8, n: u32, off: *const i32, o: *mut u32, cap: u32) { tile_select_nz(d, n, off, o, cap) }
    #[kernel] pub unsafe fn tile_select_nz_u32(d: *const u32, n: u32, off: *const i32, o: *mut u32, cap: u32) { tile_select_nz(d, n, off, o, cap) }
    #[kernel] pub unsafe fn tile_select_nz_i16(d: *const i16, n: u32, off: *const i32, o: *mut u32, cap: u32) { tile_select_nz(d, n, off, o, cap) }
    #[kernel] pub unsafe fn tile_select_nz_i32(d: *const i32, n: u32, off: *const i32, o: *mut u32, cap: u32) { tile_select_nz(d, n, off, o, cap) }
    #[kernel] pub unsafe fn tile_select_nz_i64(d: *const i64, n: u32, off: *const i32, o: *mut u32, cap: u32) { tile_select_nz(d, n, off, o, cap) }

    // ------------------------------------------------------------------------------------------
    // gemv/gemv.cu: Y[b, row] = sum_k A[row, k] * X[b, k] (+ bias[row]). Per thread fma.ftz over
    // pairs (a.x then a.y), odd-K tail on thread 0, xor butterfly, then (BLOCK > 32) warp partials
    // through shared memory into warp 0 and a second butterfly; thread 0 adds the bias (or +0.0).

    /// The xor butterfly of gemv.cu / indexed_moe.cu: 16, 8, 4, 2, 1 with add.ftz.
    #[inline(always)]
    pub fn warp_xor_sum(mut v: f32) -> f32 {
        let mut off = 16;
        while off > 0 {
            v = addf(v, warp::shuffle_xor_f32_sync(0xffff_ffff, v, off));
            off /= 2;
        }
        v
    }

    #[inline(always)]
    unsafe fn gemv<T: Val, const BLOCK: usize, const BATCH: usize>(a: *const T, x: *const T, bias: *const T, y: *mut T, m: i32, k: i32, has_bias: u8) {
        static mut WSUM: SharedArray<f32, 64> = SharedArray::UNINIT; // [NUM_WARPS][BATCH] <= 8 x 8
        let wsum = SharedArray::as_raw_mut_ptr(&raw mut WSUM);
        let row = thread::blockIdx_x() as i32;
        if row >= m {
            return;
        }
        let tid = thread::threadIdx_x() as i32;
        let k2 = k / 2;
        let arow = a.wrapping_offset(row.wrapping_mul(k) as isize);
        let mut acc = [0.0f32; BATCH];
        let mut col2 = tid;
        while col2 < k2 {
            let (a0, a1) = ((*arow.offset(2 * col2 as isize)).f(), (*arow.offset(2 * col2 as isize + 1)).f());
            let mut b = 0;
            while b < BATCH {
                let xr = x.wrapping_offset((b as i32).wrapping_mul(k) as isize);
                let (x0, x1) = ((*xr.offset(2 * col2 as isize)).f(), (*xr.offset(2 * col2 as isize + 1)).f());
                acc[b] = fmaf(a0, x0, acc[b]);
                acc[b] = fmaf(a1, x1, acc[b]);
                b += 1;
            }
            col2 += BLOCK as i32;
        }
        if k % 2 != 0 && tid == 0 {
            let last = k - 1;
            let al = (*arow.offset(last as isize)).f();
            let mut b = 0;
            while b < BATCH {
                let xl = (*x.wrapping_offset((b as i32).wrapping_mul(k).wrapping_add(last) as isize)).f();
                acc[b] = fmaf(al, xl, acc[b]);
                b += 1;
            }
        }
        let mut b = 0;
        while b < BATCH {
            acc[b] = warp_xor_sum(acc[b]);
            b += 1;
        }
        if BLOCK > 32 {
            let (wid, lane) = (tid / 32, tid % 32);
            if lane == 0 {
                let mut b = 0;
                while b < BATCH {
                    *wsum.add(wid as usize * BATCH + b) = acc[b];
                    b += 1;
                }
            }
            thread::sync_threads();
            if wid == 0 {
                let mut b = 0;
                while b < BATCH {
                    acc[b] = if (lane as usize) < BLOCK / 32 { *wsum.add(lane as usize * BATCH + b) } else { 0.0 };
                    acc[b] = warp_xor_sum(acc[b]);
                    b += 1;
                }
            }
        }
        if tid == 0 {
            let bias_val = if has_bias != 0 { (*bias.offset(row as isize)).f() } else { 0.0 };
            let mut b = 0;
            while b < BATCH {
                *y.offset((b as i32).wrapping_mul(m).wrapping_add(row) as isize) = T::t(addf(acc[b], bias_val));
                b += 1;
            }
        }
    }
    // GENERATED GEMV KERNELS BEGIN
    #[kernel] pub unsafe fn gemv_f32_32_1(a: *const f32, x: *const f32, bias: *const f32, y: *mut f32, m: i32, k: i32, has_bias: u8) { gemv::<f32, 32, 1>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f32_32_2(a: *const f32, x: *const f32, bias: *const f32, y: *mut f32, m: i32, k: i32, has_bias: u8) { gemv::<f32, 32, 2>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f32_32_3(a: *const f32, x: *const f32, bias: *const f32, y: *mut f32, m: i32, k: i32, has_bias: u8) { gemv::<f32, 32, 3>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f32_32_4(a: *const f32, x: *const f32, bias: *const f32, y: *mut f32, m: i32, k: i32, has_bias: u8) { gemv::<f32, 32, 4>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f32_32_5(a: *const f32, x: *const f32, bias: *const f32, y: *mut f32, m: i32, k: i32, has_bias: u8) { gemv::<f32, 32, 5>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f32_32_6(a: *const f32, x: *const f32, bias: *const f32, y: *mut f32, m: i32, k: i32, has_bias: u8) { gemv::<f32, 32, 6>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f32_32_7(a: *const f32, x: *const f32, bias: *const f32, y: *mut f32, m: i32, k: i32, has_bias: u8) { gemv::<f32, 32, 7>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f32_32_8(a: *const f32, x: *const f32, bias: *const f32, y: *mut f32, m: i32, k: i32, has_bias: u8) { gemv::<f32, 32, 8>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f32_64_1(a: *const f32, x: *const f32, bias: *const f32, y: *mut f32, m: i32, k: i32, has_bias: u8) { gemv::<f32, 64, 1>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f32_64_2(a: *const f32, x: *const f32, bias: *const f32, y: *mut f32, m: i32, k: i32, has_bias: u8) { gemv::<f32, 64, 2>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f32_64_3(a: *const f32, x: *const f32, bias: *const f32, y: *mut f32, m: i32, k: i32, has_bias: u8) { gemv::<f32, 64, 3>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f32_64_4(a: *const f32, x: *const f32, bias: *const f32, y: *mut f32, m: i32, k: i32, has_bias: u8) { gemv::<f32, 64, 4>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f32_64_5(a: *const f32, x: *const f32, bias: *const f32, y: *mut f32, m: i32, k: i32, has_bias: u8) { gemv::<f32, 64, 5>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f32_64_6(a: *const f32, x: *const f32, bias: *const f32, y: *mut f32, m: i32, k: i32, has_bias: u8) { gemv::<f32, 64, 6>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f32_64_7(a: *const f32, x: *const f32, bias: *const f32, y: *mut f32, m: i32, k: i32, has_bias: u8) { gemv::<f32, 64, 7>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f32_64_8(a: *const f32, x: *const f32, bias: *const f32, y: *mut f32, m: i32, k: i32, has_bias: u8) { gemv::<f32, 64, 8>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f32_128_1(a: *const f32, x: *const f32, bias: *const f32, y: *mut f32, m: i32, k: i32, has_bias: u8) { gemv::<f32, 128, 1>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f32_128_2(a: *const f32, x: *const f32, bias: *const f32, y: *mut f32, m: i32, k: i32, has_bias: u8) { gemv::<f32, 128, 2>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f32_128_3(a: *const f32, x: *const f32, bias: *const f32, y: *mut f32, m: i32, k: i32, has_bias: u8) { gemv::<f32, 128, 3>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f32_128_4(a: *const f32, x: *const f32, bias: *const f32, y: *mut f32, m: i32, k: i32, has_bias: u8) { gemv::<f32, 128, 4>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f32_128_5(a: *const f32, x: *const f32, bias: *const f32, y: *mut f32, m: i32, k: i32, has_bias: u8) { gemv::<f32, 128, 5>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f32_128_6(a: *const f32, x: *const f32, bias: *const f32, y: *mut f32, m: i32, k: i32, has_bias: u8) { gemv::<f32, 128, 6>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f32_128_7(a: *const f32, x: *const f32, bias: *const f32, y: *mut f32, m: i32, k: i32, has_bias: u8) { gemv::<f32, 128, 7>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f32_128_8(a: *const f32, x: *const f32, bias: *const f32, y: *mut f32, m: i32, k: i32, has_bias: u8) { gemv::<f32, 128, 8>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f32_256_1(a: *const f32, x: *const f32, bias: *const f32, y: *mut f32, m: i32, k: i32, has_bias: u8) { gemv::<f32, 256, 1>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f32_256_2(a: *const f32, x: *const f32, bias: *const f32, y: *mut f32, m: i32, k: i32, has_bias: u8) { gemv::<f32, 256, 2>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f32_256_3(a: *const f32, x: *const f32, bias: *const f32, y: *mut f32, m: i32, k: i32, has_bias: u8) { gemv::<f32, 256, 3>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f32_256_4(a: *const f32, x: *const f32, bias: *const f32, y: *mut f32, m: i32, k: i32, has_bias: u8) { gemv::<f32, 256, 4>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f32_256_5(a: *const f32, x: *const f32, bias: *const f32, y: *mut f32, m: i32, k: i32, has_bias: u8) { gemv::<f32, 256, 5>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f32_256_6(a: *const f32, x: *const f32, bias: *const f32, y: *mut f32, m: i32, k: i32, has_bias: u8) { gemv::<f32, 256, 6>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f32_256_7(a: *const f32, x: *const f32, bias: *const f32, y: *mut f32, m: i32, k: i32, has_bias: u8) { gemv::<f32, 256, 7>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f32_256_8(a: *const f32, x: *const f32, bias: *const f32, y: *mut f32, m: i32, k: i32, has_bias: u8) { gemv::<f32, 256, 8>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f16_32_1(a: *const H, x: *const H, bias: *const H, y: *mut H, m: i32, k: i32, has_bias: u8) { gemv::<H, 32, 1>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f16_32_2(a: *const H, x: *const H, bias: *const H, y: *mut H, m: i32, k: i32, has_bias: u8) { gemv::<H, 32, 2>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f16_32_3(a: *const H, x: *const H, bias: *const H, y: *mut H, m: i32, k: i32, has_bias: u8) { gemv::<H, 32, 3>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f16_32_4(a: *const H, x: *const H, bias: *const H, y: *mut H, m: i32, k: i32, has_bias: u8) { gemv::<H, 32, 4>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f16_32_5(a: *const H, x: *const H, bias: *const H, y: *mut H, m: i32, k: i32, has_bias: u8) { gemv::<H, 32, 5>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f16_32_6(a: *const H, x: *const H, bias: *const H, y: *mut H, m: i32, k: i32, has_bias: u8) { gemv::<H, 32, 6>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f16_32_7(a: *const H, x: *const H, bias: *const H, y: *mut H, m: i32, k: i32, has_bias: u8) { gemv::<H, 32, 7>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f16_32_8(a: *const H, x: *const H, bias: *const H, y: *mut H, m: i32, k: i32, has_bias: u8) { gemv::<H, 32, 8>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f16_64_1(a: *const H, x: *const H, bias: *const H, y: *mut H, m: i32, k: i32, has_bias: u8) { gemv::<H, 64, 1>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f16_64_2(a: *const H, x: *const H, bias: *const H, y: *mut H, m: i32, k: i32, has_bias: u8) { gemv::<H, 64, 2>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f16_64_3(a: *const H, x: *const H, bias: *const H, y: *mut H, m: i32, k: i32, has_bias: u8) { gemv::<H, 64, 3>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f16_64_4(a: *const H, x: *const H, bias: *const H, y: *mut H, m: i32, k: i32, has_bias: u8) { gemv::<H, 64, 4>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f16_64_5(a: *const H, x: *const H, bias: *const H, y: *mut H, m: i32, k: i32, has_bias: u8) { gemv::<H, 64, 5>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f16_64_6(a: *const H, x: *const H, bias: *const H, y: *mut H, m: i32, k: i32, has_bias: u8) { gemv::<H, 64, 6>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f16_64_7(a: *const H, x: *const H, bias: *const H, y: *mut H, m: i32, k: i32, has_bias: u8) { gemv::<H, 64, 7>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f16_64_8(a: *const H, x: *const H, bias: *const H, y: *mut H, m: i32, k: i32, has_bias: u8) { gemv::<H, 64, 8>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f16_128_1(a: *const H, x: *const H, bias: *const H, y: *mut H, m: i32, k: i32, has_bias: u8) { gemv::<H, 128, 1>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f16_128_2(a: *const H, x: *const H, bias: *const H, y: *mut H, m: i32, k: i32, has_bias: u8) { gemv::<H, 128, 2>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f16_128_3(a: *const H, x: *const H, bias: *const H, y: *mut H, m: i32, k: i32, has_bias: u8) { gemv::<H, 128, 3>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f16_128_4(a: *const H, x: *const H, bias: *const H, y: *mut H, m: i32, k: i32, has_bias: u8) { gemv::<H, 128, 4>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f16_128_5(a: *const H, x: *const H, bias: *const H, y: *mut H, m: i32, k: i32, has_bias: u8) { gemv::<H, 128, 5>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f16_128_6(a: *const H, x: *const H, bias: *const H, y: *mut H, m: i32, k: i32, has_bias: u8) { gemv::<H, 128, 6>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f16_128_7(a: *const H, x: *const H, bias: *const H, y: *mut H, m: i32, k: i32, has_bias: u8) { gemv::<H, 128, 7>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f16_128_8(a: *const H, x: *const H, bias: *const H, y: *mut H, m: i32, k: i32, has_bias: u8) { gemv::<H, 128, 8>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f16_256_1(a: *const H, x: *const H, bias: *const H, y: *mut H, m: i32, k: i32, has_bias: u8) { gemv::<H, 256, 1>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f16_256_2(a: *const H, x: *const H, bias: *const H, y: *mut H, m: i32, k: i32, has_bias: u8) { gemv::<H, 256, 2>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f16_256_3(a: *const H, x: *const H, bias: *const H, y: *mut H, m: i32, k: i32, has_bias: u8) { gemv::<H, 256, 3>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f16_256_4(a: *const H, x: *const H, bias: *const H, y: *mut H, m: i32, k: i32, has_bias: u8) { gemv::<H, 256, 4>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f16_256_5(a: *const H, x: *const H, bias: *const H, y: *mut H, m: i32, k: i32, has_bias: u8) { gemv::<H, 256, 5>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f16_256_6(a: *const H, x: *const H, bias: *const H, y: *mut H, m: i32, k: i32, has_bias: u8) { gemv::<H, 256, 6>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f16_256_7(a: *const H, x: *const H, bias: *const H, y: *mut H, m: i32, k: i32, has_bias: u8) { gemv::<H, 256, 7>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_f16_256_8(a: *const H, x: *const H, bias: *const H, y: *mut H, m: i32, k: i32, has_bias: u8) { gemv::<H, 256, 8>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_bf16_32_1(a: *const B, x: *const B, bias: *const B, y: *mut B, m: i32, k: i32, has_bias: u8) { gemv::<B, 32, 1>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_bf16_32_2(a: *const B, x: *const B, bias: *const B, y: *mut B, m: i32, k: i32, has_bias: u8) { gemv::<B, 32, 2>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_bf16_32_3(a: *const B, x: *const B, bias: *const B, y: *mut B, m: i32, k: i32, has_bias: u8) { gemv::<B, 32, 3>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_bf16_32_4(a: *const B, x: *const B, bias: *const B, y: *mut B, m: i32, k: i32, has_bias: u8) { gemv::<B, 32, 4>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_bf16_32_5(a: *const B, x: *const B, bias: *const B, y: *mut B, m: i32, k: i32, has_bias: u8) { gemv::<B, 32, 5>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_bf16_32_6(a: *const B, x: *const B, bias: *const B, y: *mut B, m: i32, k: i32, has_bias: u8) { gemv::<B, 32, 6>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_bf16_32_7(a: *const B, x: *const B, bias: *const B, y: *mut B, m: i32, k: i32, has_bias: u8) { gemv::<B, 32, 7>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_bf16_32_8(a: *const B, x: *const B, bias: *const B, y: *mut B, m: i32, k: i32, has_bias: u8) { gemv::<B, 32, 8>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_bf16_64_1(a: *const B, x: *const B, bias: *const B, y: *mut B, m: i32, k: i32, has_bias: u8) { gemv::<B, 64, 1>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_bf16_64_2(a: *const B, x: *const B, bias: *const B, y: *mut B, m: i32, k: i32, has_bias: u8) { gemv::<B, 64, 2>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_bf16_64_3(a: *const B, x: *const B, bias: *const B, y: *mut B, m: i32, k: i32, has_bias: u8) { gemv::<B, 64, 3>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_bf16_64_4(a: *const B, x: *const B, bias: *const B, y: *mut B, m: i32, k: i32, has_bias: u8) { gemv::<B, 64, 4>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_bf16_64_5(a: *const B, x: *const B, bias: *const B, y: *mut B, m: i32, k: i32, has_bias: u8) { gemv::<B, 64, 5>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_bf16_64_6(a: *const B, x: *const B, bias: *const B, y: *mut B, m: i32, k: i32, has_bias: u8) { gemv::<B, 64, 6>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_bf16_64_7(a: *const B, x: *const B, bias: *const B, y: *mut B, m: i32, k: i32, has_bias: u8) { gemv::<B, 64, 7>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_bf16_64_8(a: *const B, x: *const B, bias: *const B, y: *mut B, m: i32, k: i32, has_bias: u8) { gemv::<B, 64, 8>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_bf16_128_1(a: *const B, x: *const B, bias: *const B, y: *mut B, m: i32, k: i32, has_bias: u8) { gemv::<B, 128, 1>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_bf16_128_2(a: *const B, x: *const B, bias: *const B, y: *mut B, m: i32, k: i32, has_bias: u8) { gemv::<B, 128, 2>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_bf16_128_3(a: *const B, x: *const B, bias: *const B, y: *mut B, m: i32, k: i32, has_bias: u8) { gemv::<B, 128, 3>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_bf16_128_4(a: *const B, x: *const B, bias: *const B, y: *mut B, m: i32, k: i32, has_bias: u8) { gemv::<B, 128, 4>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_bf16_128_5(a: *const B, x: *const B, bias: *const B, y: *mut B, m: i32, k: i32, has_bias: u8) { gemv::<B, 128, 5>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_bf16_128_6(a: *const B, x: *const B, bias: *const B, y: *mut B, m: i32, k: i32, has_bias: u8) { gemv::<B, 128, 6>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_bf16_128_7(a: *const B, x: *const B, bias: *const B, y: *mut B, m: i32, k: i32, has_bias: u8) { gemv::<B, 128, 7>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_bf16_128_8(a: *const B, x: *const B, bias: *const B, y: *mut B, m: i32, k: i32, has_bias: u8) { gemv::<B, 128, 8>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_bf16_256_1(a: *const B, x: *const B, bias: *const B, y: *mut B, m: i32, k: i32, has_bias: u8) { gemv::<B, 256, 1>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_bf16_256_2(a: *const B, x: *const B, bias: *const B, y: *mut B, m: i32, k: i32, has_bias: u8) { gemv::<B, 256, 2>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_bf16_256_3(a: *const B, x: *const B, bias: *const B, y: *mut B, m: i32, k: i32, has_bias: u8) { gemv::<B, 256, 3>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_bf16_256_4(a: *const B, x: *const B, bias: *const B, y: *mut B, m: i32, k: i32, has_bias: u8) { gemv::<B, 256, 4>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_bf16_256_5(a: *const B, x: *const B, bias: *const B, y: *mut B, m: i32, k: i32, has_bias: u8) { gemv::<B, 256, 5>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_bf16_256_6(a: *const B, x: *const B, bias: *const B, y: *mut B, m: i32, k: i32, has_bias: u8) { gemv::<B, 256, 6>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_bf16_256_7(a: *const B, x: *const B, bias: *const B, y: *mut B, m: i32, k: i32, has_bias: u8) { gemv::<B, 256, 7>(a, x, bias, y, m, k, has_bias) }
    #[kernel] pub unsafe fn gemv_bf16_256_8(a: *const B, x: *const B, bias: *const B, y: *mut B, m: i32, k: i32, has_bias: u8) { gemv::<B, 256, 8>(a, x, bias, y, m, k, has_bias) }
    // GENERATED GEMV KERNELS END

    // ------------------------------------------------------------------------------------------
    // GGUF x Q8_1 dot products (llama.cpp mmvq vec_dot_*), shared by indexed_moe.cu and
    // moe_grouped.cu. Integer parts from candle-mmvq; the float program is the fast-math one
    // (every op .ftz), read off indexed_moe.cubin:
    // - q4_0 / q5_0: r = fma(ds.x, sumi, ds.y * -(4|8)); tmp = fma(d, r, tmp)
    // - q4_1 / q5_1: r = fma(dm.x*ds.x, sumi, (dm.y*ds.y)*0.5 [FMUL.D2]); tmp = r + tmp
    // - q8_0: tmp = fma(sumi * d0, d1, tmp)
    // - q8_1: r = fma(w.y, v.y, (w.x*v.x) * sumi); tmp = r + tmp
    // - q2_K / q4_K / q5_K: sumf_{d,m} = fma(d8[i], float(int), sumf) from 0,
    //   r = fma(sumf_d, dm.x, -(sumf_m * dm.y)); tmp = r + tmp
    // - q3_K / q6_K: sumf = fma(d8[i], float(int), sumf) from 0; tmp = fma(d, sumf, tmp)
    // FMT: 0 q4_0, 1 q4_1, 2 q5_0, 3 q5_1, 4 q8_0, 5 q2_K, 6 q3_K, 7 q4_K, 8 q5_K, 9 q6_K, 10 q8_1.

    #[inline(always)]
    pub fn dp4a(a: u32, b: u32, c: i32) -> i32 {
        dotprod::dp4a_s32(a, b, c)
    }
    #[inline(always)]
    pub unsafe fn ld8(p: *const u8, off: usize) -> u32 {
        *p.add(off) as u32
    }
    #[inline(always)]
    pub unsafe fn ld16(p: *const u8, off: usize) -> u32 {
        *(p.add(off) as *const u16) as u32
    }
    /// `get_int_from_uint8`: a 32-bit word from two 16-bit loads (2-byte aligned blocks).
    #[inline(always)]
    pub unsafe fn ld32u(p: *const u8, off: usize) -> u32 {
        ld16(p, off) | (ld16(p, off + 2) << 16)
    }
    /// `get_int_from_uint8_aligned`.
    #[inline(always)]
    pub unsafe fn ld32a(p: *const u8, off: usize) -> u32 {
        *(p.add(off) as *const u32)
    }
    /// Q8_1 block: quant word `i`.
    #[inline(always)]
    pub unsafe fn yq(yb: *const u8, i: usize) -> u32 {
        ld32a(yb, 4 + 4 * i)
    }
    #[inline(always)]
    pub fn sar(x: u32, n: usize) -> u32 {
        ((x as i32) >> n) as u32
    }
    #[inline(always)]
    pub fn h2f32(bits: u32) -> f32 {
        convert::cvt_f32_f16x2_lo(bits & 0xFFFF)
    }
    /// Per-byte signed saturating subtract (`__vsubss4`).
    #[inline(always)]
    pub fn vsubss4(a: u32, b: u32) -> u32 {
        let mut r = 0u32;
        let mut k = 0;
        while k < 4 {
            let x = ((a >> (8 * k)) & 0xFF) as u8 as i8 as i32;
            let y = ((b >> (8 * k)) & 0xFF) as u8 as i8 as i32;
            let mut d = x - y;
            if d > 127 {
                d = 127;
            }
            if d < -128 {
                d = -128;
            }
            r |= ((d as u32) & 0xFF) << (8 * k);
            k += 1;
        }
        r
    }

    /// (qk, qi, vdr, block bytes).
    #[inline(always)]
    pub fn fmt_params<const FMT: u32>() -> (i32, i32, i32, i32) {
        match FMT {

            0 => (32, 4, 2, 18),
            1 => (32, 4, 2, 20),
            2 => (32, 4, 2, 22),
            3 => (32, 4, 2, 24),
            4 => (32, 8, 2, 34),
            5 => (256, 16, 1, 84),
            6 => (256, 16, 1, 110),
            7 => (256, 32, 2, 144),
            8 => (256, 32, 2, 176),
            9 => (256, 32, 1, 210),
            11 => (256, 16, 2, 66), // iq2_xxs (GGML type 16)
            18 => (256, 16, 2, 98), // iq3_xxs: QR3_XXS=4 -> qi=QK_K/(4*QR)=16, vdr=2 (qi/vdr=8)
            23 => (256, 32, 4, 136), // iq4_xs (GGML type 23)
            20 => (32, 4, 2, 18), // iq4_nl (GGML type 20): QI4_NL = 4, VDR_IQ4_NL_Q8_1_MMVQ = 2
            17 => (256, 16, 2, 74), // iq2_xs (GGML type 17)
            22 => (256, 16, 2, 82), // iq2_s (GGML type 22)
            _ => (32, 8, 2, 36),
        }
    }


    // ==========================================================================================
    // IQ2_XXS (GGML type 16): fused MoE for the low-bit "UD" artifacts. Table helpers and the
    // sign/scale handling are lifted from the gated oxide-kernels/iq2_xxs crate, which is
    // bit-identical to llama.cpp acecd56; only the call convention differs.
    // ==========================================================================================

    /// `__vsub4(a, b)`: per-byte subtract, each byte independent and wrapping. No borrow.
    /// (NOT `vsubss4` -- that one saturates.)
    #[inline(always)]
    pub fn vsub4_wrap(a: u32, b: u32) -> u32 {
        let (a0, a1, a2, a3) = (a & 0xFF, (a >> 8) & 0xFF, (a >> 16) & 0xFF, (a >> 24) & 0xFF);
        let (b0, b1, b2, b3) = (b & 0xFF, (b >> 8) & 0xFF, (b >> 16) & 0xFF, (b >> 24) & 0xFF);
        (a0.wrapping_sub(b0) & 0xFF)
            | ((a1.wrapping_sub(b1) & 0xFF) << 8)
            | ((a2.wrapping_sub(b2) & 0xFF) << 16)
            | ((a3.wrapping_sub(b3) & 0xFF) << 24)
    }

    /// `__vcmpne4(a, 0)`: 0xff in every byte that is non-zero.
    #[inline(always)]
    pub fn vcmpne4_zero(a: u32) -> u32 {
        let (x, y, z, w) = (a & 0xFF, (a >> 8) & 0xFF, (a >> 16) & 0xFF, (a >> 24) & 0xFF);
        (if x != 0 { 0xFFu32 } else { 0 })
            | (if y != 0 { 0xFF00u32 } else { 0 })
            | (if z != 0 { 0xFF_0000u32 } else { 0 })
            | (if w != 0 { 0xFF00_0000u32 } else { 0 })
    }

    /// `get_int_b2(x, i)`: two `u16` loads (a 66-byte block is only 2-byte aligned).
    #[inline(always)]

    pub unsafe fn get_b4(x: *const u8, i: i32) -> u32 {
        // Assembled from two 16-bit reads: `qs` sits at an even (not 4-byte) offset
        // inside the i-quant blocks (xb + 2), so a plain u32 deref raises
        // CUDA_ERROR_MISALIGNED_ADDRESS.
        let p = x.add(4 * i as usize) as *const u16;
        (*p as u32) | ((*p.add(1) as u32) << 16)
    }

    pub fn byte_perm(a: u32, b: u32, s: u32) -> u32 {
        let r: u32;
        let s = s & 0x7777;
        unsafe { ptx_asm!("prmt.b32 %0, %1, %2, %3;", out("=r") r, in("r") a, in("r") b, in("r") s, options(register_only)); }
        r
    }

    pub const I4_TABLE: [u32; 4] = [0xBFAD9881, 0xF6EADDCF, 0x26190D01, 0x71594535];

    pub fn i4_table16(q4: u32) -> (u32, u32) {
            let sel = 0x32103210 | ((q4 & 0x88888888) >> 1);
            let lo0 = byte_perm(I4_TABLE[0], I4_TABLE[1], q4);
            let hi0 = byte_perm(I4_TABLE[2], I4_TABLE[3], q4);
            let t0 = byte_perm(lo0, hi0, sel);
            let lo1 = byte_perm(I4_TABLE[0], I4_TABLE[1], q4 >> 16);
            let hi1 = byte_perm(I4_TABLE[2], I4_TABLE[3], q4 >> 16);
            let t1 = byte_perm(lo1, hi1, sel >> 16);
            (byte_perm(t0, t1, 0x6420), byte_perm(t0, t1, 0x7531))
        }

    pub unsafe fn i4_group_scale6(x: *const u8, group: usize) -> u32 {
            let lo = *x.add(4 + group / 2) as u32;
            let hi = *(x.add(2) as *const u16) as u32;
            ((lo >> (4 * (group as u32 % 2))) & 0xF) | (((hi >> (2 * group as u32)) & 3) << 4)
        }

    pub fn i3_grid(i: u32) -> u32 {
            let mut v: u32 = 0;
            if i == 0u32 { v = 0x04040404u32 as u32; }
            if i == 1u32 { v = 0x04040414u32 as u32; }
            if i == 2u32 { v = 0x04040424u32 as u32; }
            if i == 3u32 { v = 0x04040c0cu32 as u32; }
            if i == 4u32 { v = 0x04040c1cu32 as u32; }
            if i == 5u32 { v = 0x04040c3eu32 as u32; }
            if i == 6u32 { v = 0x04041404u32 as u32; }
            if i == 7u32 { v = 0x04041414u32 as u32; }
            if i == 8u32 { v = 0x04041c0cu32 as u32; }
            if i == 9u32 { v = 0x04042414u32 as u32; }
            if i == 10u32 { v = 0x04043e1cu32 as u32; }
            if i == 11u32 { v = 0x04043e2cu32 as u32; }
            if i == 12u32 { v = 0x040c040cu32 as u32; }
            if i == 13u32 { v = 0x040c041cu32 as u32; }
            if i == 14u32 { v = 0x040c0c04u32 as u32; }
            if i == 15u32 { v = 0x040c0c14u32 as u32; }
            if i == 16u32 { v = 0x040c140cu32 as u32; }
            if i == 17u32 { v = 0x040c142cu32 as u32; }
            if i == 18u32 { v = 0x040c1c04u32 as u32; }
            if i == 19u32 { v = 0x040c1c14u32 as u32; }
            if i == 20u32 { v = 0x040c240cu32 as u32; }
            if i == 21u32 { v = 0x040c2c24u32 as u32; }
            if i == 22u32 { v = 0x040c3e04u32 as u32; }
            if i == 23u32 { v = 0x04140404u32 as u32; }
            if i == 24u32 { v = 0x04140414u32 as u32; }
            if i == 25u32 { v = 0x04140424u32 as u32; }
            if i == 26u32 { v = 0x04140c0cu32 as u32; }
            if i == 27u32 { v = 0x04141404u32 as u32; }
            if i == 28u32 { v = 0x04141414u32 as u32; }
            if i == 29u32 { v = 0x04141c0cu32 as u32; }
            if i == 30u32 { v = 0x04141c1cu32 as u32; }
            if i == 31u32 { v = 0x04141c3eu32 as u32; }
            if i == 32u32 { v = 0x04142c0cu32 as u32; }
            if i == 33u32 { v = 0x04142c3eu32 as u32; }
            if i == 34u32 { v = 0x04143e2cu32 as u32; }
            if i == 35u32 { v = 0x041c040cu32 as u32; }
            if i == 36u32 { v = 0x041c043eu32 as u32; }
            if i == 37u32 { v = 0x041c0c04u32 as u32; }
            if i == 38u32 { v = 0x041c0c14u32 as u32; }
            if i == 39u32 { v = 0x041c142cu32 as u32; }
            if i == 40u32 { v = 0x041c3e04u32 as u32; }
            if i == 41u32 { v = 0x04240c1cu32 as u32; }
            if i == 42u32 { v = 0x04241c3eu32 as u32; }
            if i == 43u32 { v = 0x04242424u32 as u32; }
            if i == 44u32 { v = 0x04242c3eu32 as u32; }
            if i == 45u32 { v = 0x04243e1cu32 as u32; }
            if i == 46u32 { v = 0x04243e2cu32 as u32; }
            if i == 47u32 { v = 0x042c040cu32 as u32; }
            if i == 48u32 { v = 0x042c043eu32 as u32; }
            if i == 49u32 { v = 0x042c1c14u32 as u32; }
            if i == 50u32 { v = 0x042c2c14u32 as u32; }
            if i == 51u32 { v = 0x04341c2cu32 as u32; }
            if i == 52u32 { v = 0x04343424u32 as u32; }
            if i == 53u32 { v = 0x043e0c04u32 as u32; }
            if i == 54u32 { v = 0x043e0c24u32 as u32; }
            if i == 55u32 { v = 0x043e0c34u32 as u32; }
            if i == 56u32 { v = 0x043e241cu32 as u32; }
            if i == 57u32 { v = 0x043e340cu32 as u32; }
            if i == 58u32 { v = 0x0c04040cu32 as u32; }
            if i == 59u32 { v = 0x0c04041cu32 as u32; }
            if i == 60u32 { v = 0x0c040c04u32 as u32; }
            if i == 61u32 { v = 0x0c040c14u32 as u32; }
            if i == 62u32 { v = 0x0c04140cu32 as u32; }
            if i == 63u32 { v = 0x0c04141cu32 as u32; }
            if i == 64u32 { v = 0x0c041c04u32 as u32; }
            if i == 65u32 { v = 0x0c041c14u32 as u32; }
            if i == 66u32 { v = 0x0c041c24u32 as u32; }
            if i == 67u32 { v = 0x0c04243eu32 as u32; }
            if i == 68u32 { v = 0x0c042c04u32 as u32; }
            if i == 69u32 { v = 0x0c0c0404u32 as u32; }
            if i == 70u32 { v = 0x0c0c0414u32 as u32; }
            if i == 71u32 { v = 0x0c0c0c0cu32 as u32; }
            if i == 72u32 { v = 0x0c0c1404u32 as u32; }
            if i == 73u32 { v = 0x0c0c1414u32 as u32; }
            if i == 74u32 { v = 0x0c14040cu32 as u32; }
            if i == 75u32 { v = 0x0c14041cu32 as u32; }
            if i == 76u32 { v = 0x0c140c04u32 as u32; }
            if i == 77u32 { v = 0x0c140c14u32 as u32; }
            if i == 78u32 { v = 0x0c14140cu32 as u32; }
            if i == 79u32 { v = 0x0c141c04u32 as u32; }
            if i == 80u32 { v = 0x0c143e14u32 as u32; }
            if i == 81u32 { v = 0x0c1c0404u32 as u32; }
            if i == 82u32 { v = 0x0c1c0414u32 as u32; }
            if i == 83u32 { v = 0x0c1c1404u32 as u32; }
            if i == 84u32 { v = 0x0c1c1c0cu32 as u32; }
            if i == 85u32 { v = 0x0c1c2434u32 as u32; }
            if i == 86u32 { v = 0x0c1c3434u32 as u32; }
            if i == 87u32 { v = 0x0c24040cu32 as u32; }
            if i == 88u32 { v = 0x0c24042cu32 as u32; }
            if i == 89u32 { v = 0x0c242c04u32 as u32; }
            if i == 90u32 { v = 0x0c2c1404u32 as u32; }
            if i == 91u32 { v = 0x0c2c1424u32 as u32; }
            if i == 92u32 { v = 0x0c2c2434u32 as u32; }
            if i == 93u32 { v = 0x0c2c3e0cu32 as u32; }
            if i == 94u32 { v = 0x0c34042cu32 as u32; }
            if i == 95u32 { v = 0x0c3e1414u32 as u32; }
            if i == 96u32 { v = 0x0c3e2404u32 as u32; }
            if i == 97u32 { v = 0x14040404u32 as u32; }
            if i == 98u32 { v = 0x14040414u32 as u32; }
            if i == 99u32 { v = 0x14040c0cu32 as u32; }
            if i == 100u32 { v = 0x14040c1cu32 as u32; }
            if i == 101u32 { v = 0x14041404u32 as u32; }
            if i == 102u32 { v = 0x14041414u32 as u32; }
            if i == 103u32 { v = 0x14041434u32 as u32; }
            if i == 104u32 { v = 0x14041c0cu32 as u32; }
            if i == 105u32 { v = 0x14042414u32 as u32; }
            if i == 106u32 { v = 0x140c040cu32 as u32; }
            if i == 107u32 { v = 0x140c041cu32 as u32; }
            if i == 108u32 { v = 0x140c042cu32 as u32; }
            if i == 109u32 { v = 0x140c0c04u32 as u32; }
            if i == 110u32 { v = 0x140c0c14u32 as u32; }
            if i == 111u32 { v = 0x140c140cu32 as u32; }
            if i == 112u32 { v = 0x140c1c04u32 as u32; }
            if i == 113u32 { v = 0x140c341cu32 as u32; }
            if i == 114u32 { v = 0x140c343eu32 as u32; }
            if i == 115u32 { v = 0x140c3e04u32 as u32; }
            if i == 116u32 { v = 0x14140404u32 as u32; }
            if i == 117u32 { v = 0x14140414u32 as u32; }
            if i == 118u32 { v = 0x14140c0cu32 as u32; }
            if i == 119u32 { v = 0x14140c3eu32 as u32; }
            if i == 120u32 { v = 0x14141404u32 as u32; }
            if i == 121u32 { v = 0x14141414u32 as u32; }
            if i == 122u32 { v = 0x14141c3eu32 as u32; }
            if i == 123u32 { v = 0x14142404u32 as u32; }
            if i == 124u32 { v = 0x14142c2cu32 as u32; }
            if i == 125u32 { v = 0x141c040cu32 as u32; }
            if i == 126u32 { v = 0x141c0c04u32 as u32; }
            if i == 127u32 { v = 0x141c0c24u32 as u32; }
            if i == 128u32 { v = 0x141c3e04u32 as u32; }
            if i == 129u32 { v = 0x141c3e24u32 as u32; }
            if i == 130u32 { v = 0x14241c2cu32 as u32; }
            if i == 131u32 { v = 0x14242c1cu32 as u32; }
            if i == 132u32 { v = 0x142c041cu32 as u32; }
            if i == 133u32 { v = 0x142c143eu32 as u32; }
            if i == 134u32 { v = 0x142c240cu32 as u32; }
            if i == 135u32 { v = 0x142c3e24u32 as u32; }
            if i == 136u32 { v = 0x143e040cu32 as u32; }
            if i == 137u32 { v = 0x143e041cu32 as u32; }
            if i == 138u32 { v = 0x143e0c34u32 as u32; }
            if i == 139u32 { v = 0x143e242cu32 as u32; }
            if i == 140u32 { v = 0x1c04040cu32 as u32; }
            if i == 141u32 { v = 0x1c040c04u32 as u32; }
            if i == 142u32 { v = 0x1c040c14u32 as u32; }
            if i == 143u32 { v = 0x1c04140cu32 as u32; }
            if i == 144u32 { v = 0x1c04141cu32 as u32; }
            if i == 145u32 { v = 0x1c042c04u32 as u32; }
            if i == 146u32 { v = 0x1c04342cu32 as u32; }
            if i == 147u32 { v = 0x1c043e14u32 as u32; }
            if i == 148u32 { v = 0x1c0c0404u32 as u32; }
            if i == 149u32 { v = 0x1c0c0414u32 as u32; }
            if i == 150u32 { v = 0x1c0c1404u32 as u32; }
            if i == 151u32 { v = 0x1c0c1c0cu32 as u32; }
            if i == 152u32 { v = 0x1c0c2424u32 as u32; }
            if i == 153u32 { v = 0x1c0c2434u32 as u32; }
            if i == 154u32 { v = 0x1c14040cu32 as u32; }
            if i == 155u32 { v = 0x1c14041cu32 as u32; }
            if i == 156u32 { v = 0x1c140c04u32 as u32; }
            if i == 157u32 { v = 0x1c14142cu32 as u32; }
            if i == 158u32 { v = 0x1c142c14u32 as u32; }
            if i == 159u32 { v = 0x1c143e14u32 as u32; }
            if i == 160u32 { v = 0x1c1c0c0cu32 as u32; }
            if i == 161u32 { v = 0x1c1c1c1cu32 as u32; }
            if i == 162u32 { v = 0x1c241c04u32 as u32; }
            if i == 163u32 { v = 0x1c24243eu32 as u32; }
            if i == 164u32 { v = 0x1c243e14u32 as u32; }
            if i == 165u32 { v = 0x1c2c0404u32 as u32; }
            if i == 166u32 { v = 0x1c2c0434u32 as u32; }
            if i == 167u32 { v = 0x1c2c1414u32 as u32; }
            if i == 168u32 { v = 0x1c2c2c2cu32 as u32; }
            if i == 169u32 { v = 0x1c340c24u32 as u32; }
            if i == 170u32 { v = 0x1c341c34u32 as u32; }
            if i == 171u32 { v = 0x1c34341cu32 as u32; }
            if i == 172u32 { v = 0x1c3e1c1cu32 as u32; }
            if i == 173u32 { v = 0x1c3e3404u32 as u32; }
            if i == 174u32 { v = 0x24040424u32 as u32; }
            if i == 175u32 { v = 0x24040c3eu32 as u32; }
            if i == 176u32 { v = 0x24041c2cu32 as u32; }
            if i == 177u32 { v = 0x24041c3eu32 as u32; }
            if i == 178u32 { v = 0x24042c1cu32 as u32; }
            if i == 179u32 { v = 0x24042c3eu32 as u32; }
            if i == 180u32 { v = 0x240c3e24u32 as u32; }
            if i == 181u32 { v = 0x24141404u32 as u32; }
            if i == 182u32 { v = 0x24141c3eu32 as u32; }
            if i == 183u32 { v = 0x24142404u32 as u32; }
            if i == 184u32 { v = 0x24143404u32 as u32; }
            if i == 185u32 { v = 0x24143434u32 as u32; }
            if i == 186u32 { v = 0x241c043eu32 as u32; }
            if i == 187u32 { v = 0x241c242cu32 as u32; }
            if i == 188u32 { v = 0x24240424u32 as u32; }
            if i == 189u32 { v = 0x24242c0cu32 as u32; }
            if i == 190u32 { v = 0x24243424u32 as u32; }
            if i == 191u32 { v = 0x242c142cu32 as u32; }
            if i == 192u32 { v = 0x242c241cu32 as u32; }
            if i == 193u32 { v = 0x242c3e04u32 as u32; }
            if i == 194u32 { v = 0x243e042cu32 as u32; }
            if i == 195u32 { v = 0x243e0c04u32 as u32; }
            if i == 196u32 { v = 0x243e0c14u32 as u32; }
            if i == 197u32 { v = 0x243e1c04u32 as u32; }
            if i == 198u32 { v = 0x2c040c14u32 as u32; }
            if i == 199u32 { v = 0x2c04240cu32 as u32; }
            if i == 200u32 { v = 0x2c043e04u32 as u32; }
            if i == 201u32 { v = 0x2c0c0404u32 as u32; }
            if i == 202u32 { v = 0x2c0c0434u32 as u32; }
            if i == 203u32 { v = 0x2c0c1434u32 as u32; }
            if i == 204u32 { v = 0x2c0c2c2cu32 as u32; }
            if i == 205u32 { v = 0x2c140c24u32 as u32; }
            if i == 206u32 { v = 0x2c141c14u32 as u32; }
            if i == 207u32 { v = 0x2c143e14u32 as u32; }
            if i == 208u32 { v = 0x2c1c0414u32 as u32; }
            if i == 209u32 { v = 0x2c1c2c1cu32 as u32; }
            if i == 210u32 { v = 0x2c240c04u32 as u32; }
            if i == 211u32 { v = 0x2c24141cu32 as u32; }
            if i == 212u32 { v = 0x2c24143eu32 as u32; }
            if i == 213u32 { v = 0x2c243e14u32 as u32; }
            if i == 214u32 { v = 0x2c2c0414u32 as u32; }
            if i == 215u32 { v = 0x2c2c1c0cu32 as u32; }
            if i == 216u32 { v = 0x2c342c04u32 as u32; }
            if i == 217u32 { v = 0x2c3e1424u32 as u32; }
            if i == 218u32 { v = 0x2c3e2414u32 as u32; }
            if i == 219u32 { v = 0x34041424u32 as u32; }
            if i == 220u32 { v = 0x34042424u32 as u32; }
            if i == 221u32 { v = 0x34042434u32 as u32; }
            if i == 222u32 { v = 0x34043424u32 as u32; }
            if i == 223u32 { v = 0x340c140cu32 as u32; }
            if i == 224u32 { v = 0x340c340cu32 as u32; }
            if i == 225u32 { v = 0x34140c3eu32 as u32; }
            if i == 226u32 { v = 0x34143424u32 as u32; }
            if i == 227u32 { v = 0x341c1c04u32 as u32; }
            if i == 228u32 { v = 0x341c1c34u32 as u32; }
            if i == 229u32 { v = 0x34242424u32 as u32; }
            if i == 230u32 { v = 0x342c042cu32 as u32; }
            if i == 231u32 { v = 0x342c2c14u32 as u32; }
            if i == 232u32 { v = 0x34341c1cu32 as u32; }
            if i == 233u32 { v = 0x343e041cu32 as u32; }
            if i == 234u32 { v = 0x343e140cu32 as u32; }
            if i == 235u32 { v = 0x3e04041cu32 as u32; }
            if i == 236u32 { v = 0x3e04042cu32 as u32; }
            if i == 237u32 { v = 0x3e04043eu32 as u32; }
            if i == 238u32 { v = 0x3e040c04u32 as u32; }
            if i == 239u32 { v = 0x3e041c14u32 as u32; }
            if i == 240u32 { v = 0x3e042c14u32 as u32; }
            if i == 241u32 { v = 0x3e0c1434u32 as u32; }
            if i == 242u32 { v = 0x3e0c2404u32 as u32; }
            if i == 243u32 { v = 0x3e140c14u32 as u32; }
            if i == 244u32 { v = 0x3e14242cu32 as u32; }
            if i == 245u32 { v = 0x3e142c14u32 as u32; }
            if i == 246u32 { v = 0x3e1c0404u32 as u32; }
            if i == 247u32 { v = 0x3e1c0c2cu32 as u32; }
            if i == 248u32 { v = 0x3e1c1c1cu32 as u32; }
            if i == 249u32 { v = 0x3e1c3404u32 as u32; }
            if i == 250u32 { v = 0x3e24140cu32 as u32; }
            if i == 251u32 { v = 0x3e24240cu32 as u32; }
            if i == 252u32 { v = 0x3e2c0404u32 as u32; }
            if i == 253u32 { v = 0x3e2c0414u32 as u32; }
            if i == 254u32 { v = 0x3e2c1424u32 as u32; }
            if i == 255u32 { v = 0x3e341c04u32 as u32; }
            v
        }

    pub unsafe fn get_b2(x: *const u8, i: i32) -> u32 {
        let p = x.add(4 * i as usize) as *const u16;
        (*p as u32) | ((*p.add(1) as u32) << 16)
    }

pub fn popc7(v: u32) -> u32 {
        let mut x = v & 0x7F;
        let mut n = 0u32;
        while x != 0 {
            n += x & 1;
            x >>= 1;
        }
        n
    }

pub fn unpack_ksigns(v: u32) -> u32 {
        let p = popc7(v) & 1;
        let s = (v & 0x7F) ^ (p << 7);
        s.wrapping_mul(0x0101_0101)
    }

pub fn grid_lo(i: u32) -> u32 {
        let mut v: u32 = 0;
        if i == 0u32 { v = 0x08080808u32; }
        if i == 1u32 { v = 0x0808082bu32; }
        if i == 2u32 { v = 0x08081919u32; }
        if i == 3u32 { v = 0x08082b08u32; }
        if i == 4u32 { v = 0x08082b2bu32; }
        if i == 5u32 { v = 0x08190819u32; }
        if i == 6u32 { v = 0x08191908u32; }
        if i == 7u32 { v = 0x082b0808u32; }
        if i == 8u32 { v = 0x082b082bu32; }
        if i == 9u32 { v = 0x082b2b08u32; }
        if i == 10u32 { v = 0x082b2b2bu32; }
        if i == 11u32 { v = 0x19080819u32; }
        if i == 12u32 { v = 0x19081908u32; }
        if i == 13u32 { v = 0x19190808u32; }
        if i == 14u32 { v = 0x19192b08u32; }
        if i == 15u32 { v = 0x192b0819u32; }
        if i == 16u32 { v = 0x192b1908u32; }
        if i == 17u32 { v = 0x2b080808u32; }
        if i == 18u32 { v = 0x2b08082bu32; }
        if i == 19u32 { v = 0x2b082b2bu32; }
        if i == 20u32 { v = 0x2b2b082bu32; }
        if i == 21u32 { v = 0x08080819u32; }
        if i == 22u32 { v = 0x08081908u32; }
        if i == 23u32 { v = 0x08190808u32; }
        if i == 24u32 { v = 0x08191919u32; }
        if i == 25u32 { v = 0x19080808u32; }
        if i == 26u32 { v = 0x2b081908u32; }
        if i == 27u32 { v = 0x2b192b08u32; }
        if i == 28u32 { v = 0x08080808u32; }
        if i == 29u32 { v = 0x0808082bu32; }
        if i == 30u32 { v = 0x082b082bu32; }
        if i == 31u32 { v = 0x2b08082bu32; }
        if i == 32u32 { v = 0x08080819u32; }
        if i == 33u32 { v = 0x08081908u32; }
        if i == 34u32 { v = 0x08190808u32; }
        if i == 35u32 { v = 0x082b0819u32; }
        if i == 36u32 { v = 0x082b1908u32; }
        if i == 37u32 { v = 0x19080808u32; }
        if i == 38u32 { v = 0x1908082bu32; }
        if i == 39u32 { v = 0x19082b08u32; }
        if i == 40u32 { v = 0x192b0808u32; }
        if i == 41u32 { v = 0x2b080819u32; }
        if i == 42u32 { v = 0x2b081908u32; }
        if i == 43u32 { v = 0x2b190808u32; }
        if i == 44u32 { v = 0x2b2b1908u32; }
        if i == 45u32 { v = 0x08080808u32; }
        if i == 46u32 { v = 0x0808082bu32; }
        if i == 47u32 { v = 0x08082b08u32; }
        if i == 48u32 { v = 0x082b0808u32; }
        if i == 49u32 { v = 0x1908192bu32; }
        if i == 50u32 { v = 0x192b2b19u32; }
        if i == 51u32 { v = 0x2b080808u32; }
        if i == 52u32 { v = 0x2b190819u32; }
        if i == 53u32 { v = 0x08082b19u32; }
        if i == 54u32 { v = 0x08190808u32; }
        if i == 55u32 { v = 0x19080808u32; }
        if i == 56u32 { v = 0x2b081908u32; }
        if i == 57u32 { v = 0x2b2b1908u32; }
        if i == 58u32 { v = 0x08080808u32; }
        if i == 59u32 { v = 0x08081919u32; }
        if i == 60u32 { v = 0x08082b08u32; }
        if i == 61u32 { v = 0x08191908u32; }
        if i == 62u32 { v = 0x082b2b08u32; }
        if i == 63u32 { v = 0x19080819u32; }
        if i == 64u32 { v = 0x19081908u32; }
        if i == 65u32 { v = 0x19190808u32; }
        if i == 66u32 { v = 0x1919082bu32; }
        if i == 67u32 { v = 0x2b082b08u32; }
        if i == 68u32 { v = 0x08081908u32; }
        if i == 69u32 { v = 0x19080808u32; }
        if i == 70u32 { v = 0x0808082bu32; }
        if i == 71u32 { v = 0x08191908u32; }
        if i == 72u32 { v = 0x08080819u32; }
        if i == 73u32 { v = 0x08081908u32; }
        if i == 74u32 { v = 0x08190808u32; }
        if i == 75u32 { v = 0x082b0819u32; }
        if i == 76u32 { v = 0x19080808u32; }
        if i == 77u32 { v = 0x192b0808u32; }
        if i == 78u32 { v = 0x2b081908u32; }
        if i == 79u32 { v = 0x2b190808u32; }
        if i == 80u32 { v = 0x2b191919u32; }
        if i == 81u32 { v = 0x08080808u32; }
        if i == 82u32 { v = 0x08082b08u32; }
        if i == 83u32 { v = 0x082b0808u32; }
        if i == 84u32 { v = 0x19190808u32; }
        if i == 85u32 { v = 0x19192b2bu32; }
        if i == 86u32 { v = 0x2b080808u32; }
        if i == 87u32 { v = 0x082b1908u32; }
        if i == 88u32 { v = 0x19081919u32; }
        if i == 89u32 { v = 0x08080808u32; }
        if i == 90u32 { v = 0x08082b08u32; }
        if i == 91u32 { v = 0x082b0808u32; }
        if i == 92u32 { v = 0x082b1919u32; }
        if i == 93u32 { v = 0x19082b19u32; }
        if i == 94u32 { v = 0x2b080808u32; }
        if i == 95u32 { v = 0x08192b08u32; }
        if i == 96u32 { v = 0x192b082bu32; }
        if i == 97u32 { v = 0x08080808u32; }
        if i == 98u32 { v = 0x0819192bu32; }
        if i == 99u32 { v = 0x08080819u32; }
        if i == 100u32 { v = 0x08081908u32; }
        if i == 101u32 { v = 0x08190808u32; }
        if i == 102u32 { v = 0x19080808u32; }
        if i == 103u32 { v = 0x2b080819u32; }
        if i == 104u32 { v = 0x08080808u32; }
        if i == 105u32 { v = 0x08081919u32; }
        if i == 106u32 { v = 0x2b2b0808u32; }
        if i == 107u32 { v = 0x19190819u32; }
        if i == 108u32 { v = 0x08080808u32; }
        if i == 109u32 { v = 0x0808082bu32; }
        if i == 110u32 { v = 0x08082b2bu32; }
        if i == 111u32 { v = 0x19081908u32; }
        if i == 112u32 { v = 0x192b0819u32; }
        if i == 113u32 { v = 0x2b080808u32; }
        if i == 114u32 { v = 0x2b08082bu32; }
        if i == 115u32 { v = 0x082b2b19u32; }
        if i == 116u32 { v = 0x19082b08u32; }
        if i == 117u32 { v = 0x08080808u32; }
        if i == 118u32 { v = 0x0808082bu32; }
        if i == 119u32 { v = 0x08080819u32; }
        if i == 120u32 { v = 0x08081908u32; }
        if i == 121u32 { v = 0x08190808u32; }
        if i == 122u32 { v = 0x19080808u32; }
        if i == 123u32 { v = 0x1919192bu32; }
        if i == 124u32 { v = 0x08080808u32; }
        if i == 125u32 { v = 0x19080819u32; }
        if i == 126u32 { v = 0x192b1908u32; }
        if i == 127u32 { v = 0x2b190808u32; }
        if i == 128u32 { v = 0x08082b08u32; }
        if i == 129u32 { v = 0x082b0808u32; }
        if i == 130u32 { v = 0x2b191908u32; }
        if i == 131u32 { v = 0x19081908u32; }
        if i == 132u32 { v = 0x08080819u32; }
        if i == 133u32 { v = 0x08081908u32; }
        if i == 134u32 { v = 0x08190808u32; }
        if i == 135u32 { v = 0x08192b08u32; }
        if i == 136u32 { v = 0x082b0819u32; }
        if i == 137u32 { v = 0x082b1908u32; }
        if i == 138u32 { v = 0x19080808u32; }
        if i == 139u32 { v = 0x19082b08u32; }
        if i == 140u32 { v = 0x1919192bu32; }
        if i == 141u32 { v = 0x192b0808u32; }
        if i == 142u32 { v = 0x2b080819u32; }
        if i == 143u32 { v = 0x2b081908u32; }
        if i == 144u32 { v = 0x2b190808u32; }
        if i == 145u32 { v = 0x08080808u32; }
        if i == 146u32 { v = 0x082b0808u32; }
        if i == 147u32 { v = 0x192b0819u32; }
        if i == 148u32 { v = 0x2b080808u32; }
        if i == 149u32 { v = 0x2b081919u32; }
        if i == 150u32 { v = 0x08080819u32; }
        if i == 151u32 { v = 0x08190808u32; }
        if i == 152u32 { v = 0x19082b08u32; }
        if i == 153u32 { v = 0x1919192bu32; }
        if i == 154u32 { v = 0x192b2b08u32; }
        if i == 155u32 { v = 0x08080808u32; }
        if i == 156u32 { v = 0x08082b08u32; }
        if i == 157u32 { v = 0x082b0808u32; }
        if i == 158u32 { v = 0x2b080808u32; }
        if i == 159u32 { v = 0x2b192b19u32; }
        if i == 160u32 { v = 0x0819082bu32; }
        if i == 161u32 { v = 0x082b1908u32; }
        if i == 162u32 { v = 0x08080808u32; }
        if i == 163u32 { v = 0x08080819u32; }
        if i == 164u32 { v = 0x08081908u32; }
        if i == 165u32 { v = 0x08190808u32; }
        if i == 166u32 { v = 0x19080808u32; }
        if i == 167u32 { v = 0x19081919u32; }
        if i == 168u32 { v = 0x08080808u32; }
        if i == 169u32 { v = 0x19192b08u32; }
        if i == 170u32 { v = 0x192b0819u32; }
        if i == 171u32 { v = 0x2b08082bu32; }
        if i == 172u32 { v = 0x19081919u32; }
        if i == 173u32 { v = 0x2b190808u32; }
        if i == 174u32 { v = 0x08080808u32; }
        if i == 175u32 { v = 0x08082b08u32; }
        if i == 176u32 { v = 0x08190819u32; }
        if i == 177u32 { v = 0x08192b19u32; }
        if i == 178u32 { v = 0x082b0808u32; }
        if i == 179u32 { v = 0x2b080808u32; }
        if i == 180u32 { v = 0x2b082b08u32; }
        if i == 181u32 { v = 0x08081908u32; }
        if i == 182u32 { v = 0x1908082bu32; }
        if i == 183u32 { v = 0x2b2b1908u32; }
        if i == 184u32 { v = 0x2b190819u32; }
        if i == 185u32 { v = 0x2b190808u32; }
        if i == 186u32 { v = 0x2b19082bu32; }
        if i == 187u32 { v = 0x08082b2bu32; }
        if i == 188u32 { v = 0x08080819u32; }
        if i == 189u32 { v = 0x19191908u32; }
        if i == 190u32 { v = 0x08080808u32; }
        if i == 191u32 { v = 0x08190819u32; }
        if i == 192u32 { v = 0x08192b19u32; }
        if i == 193u32 { v = 0x192b1908u32; }
        if i == 194u32 { v = 0x19080808u32; }
        if i == 195u32 { v = 0x08082b08u32; }
        if i == 196u32 { v = 0x08081908u32; }
        if i == 197u32 { v = 0x08190808u32; }
        if i == 198u32 { v = 0x19080808u32; }
        if i == 199u32 { v = 0x192b2b08u32; }
        if i == 200u32 { v = 0x08080808u32; }
        if i == 201u32 { v = 0x19191919u32; }
        if i == 202u32 { v = 0x08192b08u32; }
        if i == 203u32 { v = 0x192b0808u32; }
        if i == 204u32 { v = 0x08080808u32; }
        if i == 205u32 { v = 0x08081919u32; }
        if i == 206u32 { v = 0x08190808u32; }
        if i == 207u32 { v = 0x0819082bu32; }
        if i == 208u32 { v = 0x2b081908u32; }
        if i == 209u32 { v = 0x1908082bu32; }
        if i == 210u32 { v = 0x08080808u32; }
        if i == 211u32 { v = 0x0808082bu32; }
        if i == 212u32 { v = 0x08082b2bu32; }
        if i == 213u32 { v = 0x19080819u32; }
        if i == 214u32 { v = 0x2b08082bu32; }
        if i == 215u32 { v = 0x08081908u32; }
        if i == 216u32 { v = 0x08192b08u32; }
        if i == 217u32 { v = 0x19080808u32; }
        if i == 218u32 { v = 0x08190819u32; }
        if i == 219u32 { v = 0x08080819u32; }
        if i == 220u32 { v = 0x08081908u32; }
        if i == 221u32 { v = 0x08190808u32; }
        if i == 222u32 { v = 0x08191919u32; }
        if i == 223u32 { v = 0x19080808u32; }
        if i == 224u32 { v = 0x192b0808u32; }
        if i == 225u32 { v = 0x08080808u32; }
        if i == 226u32 { v = 0x1908192bu32; }
        if i == 227u32 { v = 0x2b191908u32; }
        if i == 228u32 { v = 0x08082b19u32; }
        if i == 229u32 { v = 0x19080808u32; }
        if i == 230u32 { v = 0x192b0808u32; }
        if i == 231u32 { v = 0x0808082bu32; }
        if i == 232u32 { v = 0x08081908u32; }
        if i == 233u32 { v = 0x08190819u32; }
        if i == 234u32 { v = 0x08081908u32; }
        if i == 235u32 { v = 0x08190808u32; }
        if i == 236u32 { v = 0x082b1908u32; }
        if i == 237u32 { v = 0x19080808u32; }
        if i == 238u32 { v = 0x2b2b0819u32; }
        if i == 239u32 { v = 0x0819192bu32; }
        if i == 240u32 { v = 0x2b080808u32; }
        if i == 241u32 { v = 0x19081919u32; }
        if i == 242u32 { v = 0x08080808u32; }
        if i == 243u32 { v = 0x082b082bu32; }
        if i == 244u32 { v = 0x19081908u32; }
        if i == 245u32 { v = 0x19190819u32; }
        if i == 246u32 { v = 0x2b080819u32; }
        if i == 247u32 { v = 0x082b0808u32; }
        if i == 248u32 { v = 0x0808082bu32; }
        if i == 249u32 { v = 0x19190808u32; }
        if i == 250u32 { v = 0x2b081919u32; }
        if i == 251u32 { v = 0x08082b19u32; }
        if i == 252u32 { v = 0x08080808u32; }
        if i == 253u32 { v = 0x08192b08u32; }
        if i == 254u32 { v = 0x19190808u32; }
        if i == 255u32 { v = 0x08081908u32; }
        v
    }

pub fn grid_hi(i: u32) -> u32 {
        let mut v: u32 = 0;
        if i == 0u32 { v = 0x08080808u32; }
        if i == 1u32 { v = 0x08080808u32; }
        if i == 2u32 { v = 0x08080808u32; }
        if i == 3u32 { v = 0x08080808u32; }
        if i == 4u32 { v = 0x08080808u32; }
        if i == 5u32 { v = 0x08080808u32; }
        if i == 6u32 { v = 0x08080808u32; }
        if i == 7u32 { v = 0x08080808u32; }
        if i == 8u32 { v = 0x08080808u32; }
        if i == 9u32 { v = 0x08080808u32; }
        if i == 10u32 { v = 0x08080808u32; }
        if i == 11u32 { v = 0x08080808u32; }
        if i == 12u32 { v = 0x08080808u32; }
        if i == 13u32 { v = 0x08080808u32; }
        if i == 14u32 { v = 0x08080808u32; }
        if i == 15u32 { v = 0x08080808u32; }
        if i == 16u32 { v = 0x08080808u32; }
        if i == 17u32 { v = 0x08080808u32; }
        if i == 18u32 { v = 0x08080808u32; }
        if i == 19u32 { v = 0x08080808u32; }
        if i == 20u32 { v = 0x08080808u32; }
        if i == 21u32 { v = 0x08080819u32; }
        if i == 22u32 { v = 0x08080819u32; }
        if i == 23u32 { v = 0x08080819u32; }
        if i == 24u32 { v = 0x08080819u32; }
        if i == 25u32 { v = 0x08080819u32; }
        if i == 26u32 { v = 0x08080819u32; }
        if i == 27u32 { v = 0x08080819u32; }
        if i == 28u32 { v = 0x0808082bu32; }
        if i == 29u32 { v = 0x0808082bu32; }
        if i == 30u32 { v = 0x0808082bu32; }
        if i == 31u32 { v = 0x0808082bu32; }
        if i == 32u32 { v = 0x08081908u32; }
        if i == 33u32 { v = 0x08081908u32; }
        if i == 34u32 { v = 0x08081908u32; }
        if i == 35u32 { v = 0x08081908u32; }
        if i == 36u32 { v = 0x08081908u32; }
        if i == 37u32 { v = 0x08081908u32; }
        if i == 38u32 { v = 0x08081908u32; }
        if i == 39u32 { v = 0x08081908u32; }
        if i == 40u32 { v = 0x08081908u32; }
        if i == 41u32 { v = 0x08081908u32; }
        if i == 42u32 { v = 0x08081908u32; }
        if i == 43u32 { v = 0x08081908u32; }
        if i == 44u32 { v = 0x08081908u32; }
        if i == 45u32 { v = 0x08081919u32; }
        if i == 46u32 { v = 0x08081919u32; }
        if i == 47u32 { v = 0x08081919u32; }
        if i == 48u32 { v = 0x08081919u32; }
        if i == 49u32 { v = 0x08081919u32; }
        if i == 50u32 { v = 0x08081919u32; }
        if i == 51u32 { v = 0x08081919u32; }
        if i == 52u32 { v = 0x08081919u32; }
        if i == 53u32 { v = 0x0808192bu32; }
        if i == 54u32 { v = 0x0808192bu32; }
        if i == 55u32 { v = 0x0808192bu32; }
        if i == 56u32 { v = 0x0808192bu32; }
        if i == 57u32 { v = 0x0808192bu32; }
        if i == 58u32 { v = 0x08082b08u32; }
        if i == 59u32 { v = 0x08082b08u32; }
        if i == 60u32 { v = 0x08082b08u32; }
        if i == 61u32 { v = 0x08082b08u32; }
        if i == 62u32 { v = 0x08082b08u32; }
        if i == 63u32 { v = 0x08082b08u32; }
        if i == 64u32 { v = 0x08082b08u32; }
        if i == 65u32 { v = 0x08082b08u32; }
        if i == 66u32 { v = 0x08082b08u32; }
        if i == 67u32 { v = 0x08082b08u32; }
        if i == 68u32 { v = 0x08082b19u32; }
        if i == 69u32 { v = 0x08082b19u32; }
        if i == 70u32 { v = 0x08082b2bu32; }
        if i == 71u32 { v = 0x08082b2bu32; }
        if i == 72u32 { v = 0x08190808u32; }
        if i == 73u32 { v = 0x08190808u32; }
        if i == 74u32 { v = 0x08190808u32; }
        if i == 75u32 { v = 0x08190808u32; }
        if i == 76u32 { v = 0x08190808u32; }
        if i == 77u32 { v = 0x08190808u32; }
        if i == 78u32 { v = 0x08190808u32; }
        if i == 79u32 { v = 0x08190808u32; }
        if i == 80u32 { v = 0x08190808u32; }
        if i == 81u32 { v = 0x08190819u32; }
        if i == 82u32 { v = 0x08190819u32; }
        if i == 83u32 { v = 0x08190819u32; }
        if i == 84u32 { v = 0x08190819u32; }
        if i == 85u32 { v = 0x08190819u32; }
        if i == 86u32 { v = 0x08190819u32; }
        if i == 87u32 { v = 0x0819082bu32; }
        if i == 88u32 { v = 0x0819082bu32; }
        if i == 89u32 { v = 0x08191908u32; }
        if i == 90u32 { v = 0x08191908u32; }
        if i == 91u32 { v = 0x08191908u32; }
        if i == 92u32 { v = 0x08191908u32; }
        if i == 93u32 { v = 0x08191908u32; }
        if i == 94u32 { v = 0x08191908u32; }
        if i == 95u32 { v = 0x08191919u32; }
        if i == 96u32 { v = 0x08191919u32; }
        if i == 97u32 { v = 0x0819192bu32; }
        if i == 98u32 { v = 0x0819192bu32; }
        if i == 99u32 { v = 0x08192b08u32; }
        if i == 100u32 { v = 0x08192b08u32; }
        if i == 101u32 { v = 0x08192b08u32; }
        if i == 102u32 { v = 0x08192b08u32; }
        if i == 103u32 { v = 0x08192b08u32; }
        if i == 104u32 { v = 0x08192b19u32; }
        if i == 105u32 { v = 0x08192b19u32; }
        if i == 106u32 { v = 0x08192b19u32; }
        if i == 107u32 { v = 0x08192b2bu32; }
        if i == 108u32 { v = 0x082b0808u32; }
        if i == 109u32 { v = 0x082b0808u32; }
        if i == 110u32 { v = 0x082b0808u32; }
        if i == 111u32 { v = 0x082b0808u32; }
        if i == 112u32 { v = 0x082b0808u32; }
        if i == 113u32 { v = 0x082b0808u32; }
        if i == 114u32 { v = 0x082b0808u32; }
        if i == 115u32 { v = 0x082b0819u32; }
        if i == 116u32 { v = 0x082b0819u32; }
        if i == 117u32 { v = 0x082b082bu32; }
        if i == 118u32 { v = 0x082b082bu32; }
        if i == 119u32 { v = 0x082b1908u32; }
        if i == 120u32 { v = 0x082b1908u32; }
        if i == 121u32 { v = 0x082b1908u32; }
        if i == 122u32 { v = 0x082b1908u32; }
        if i == 123u32 { v = 0x082b1908u32; }
        if i == 124u32 { v = 0x082b1919u32; }
        if i == 125u32 { v = 0x082b1919u32; }
        if i == 126u32 { v = 0x082b1919u32; }
        if i == 127u32 { v = 0x082b192bu32; }
        if i == 128u32 { v = 0x082b2b08u32; }
        if i == 129u32 { v = 0x082b2b08u32; }
        if i == 130u32 { v = 0x082b2b08u32; }
        if i == 131u32 { v = 0x082b2b2bu32; }
        if i == 132u32 { v = 0x19080808u32; }
        if i == 133u32 { v = 0x19080808u32; }
        if i == 134u32 { v = 0x19080808u32; }
        if i == 135u32 { v = 0x19080808u32; }
        if i == 136u32 { v = 0x19080808u32; }
        if i == 137u32 { v = 0x19080808u32; }
        if i == 138u32 { v = 0x19080808u32; }
        if i == 139u32 { v = 0x19080808u32; }
        if i == 140u32 { v = 0x19080808u32; }
        if i == 141u32 { v = 0x19080808u32; }
        if i == 142u32 { v = 0x19080808u32; }
        if i == 143u32 { v = 0x19080808u32; }
        if i == 144u32 { v = 0x19080808u32; }
        if i == 145u32 { v = 0x19080819u32; }
        if i == 146u32 { v = 0x19080819u32; }
        if i == 147u32 { v = 0x19080819u32; }
        if i == 148u32 { v = 0x19080819u32; }
        if i == 149u32 { v = 0x19080819u32; }
        if i == 150u32 { v = 0x1908082bu32; }
        if i == 151u32 { v = 0x1908082bu32; }
        if i == 152u32 { v = 0x1908082bu32; }
        if i == 153u32 { v = 0x1908082bu32; }
        if i == 154u32 { v = 0x1908082bu32; }
        if i == 155u32 { v = 0x19081908u32; }
        if i == 156u32 { v = 0x19081908u32; }
        if i == 157u32 { v = 0x19081908u32; }
        if i == 158u32 { v = 0x19081908u32; }
        if i == 159u32 { v = 0x19081908u32; }
        if i == 160u32 { v = 0x19081919u32; }
        if i == 161u32 { v = 0x19081919u32; }
        if i == 162u32 { v = 0x1908192bu32; }
        if i == 163u32 { v = 0x19082b08u32; }
        if i == 164u32 { v = 0x19082b08u32; }
        if i == 165u32 { v = 0x19082b08u32; }
        if i == 166u32 { v = 0x19082b08u32; }
        if i == 167u32 { v = 0x19082b08u32; }
        if i == 168u32 { v = 0x19082b19u32; }
        if i == 169u32 { v = 0x19082b19u32; }
        if i == 170u32 { v = 0x19082b19u32; }
        if i == 171u32 { v = 0x19082b19u32; }
        if i == 172u32 { v = 0x19082b2bu32; }
        if i == 173u32 { v = 0x19082b2bu32; }
        if i == 174u32 { v = 0x19190808u32; }
        if i == 175u32 { v = 0x19190808u32; }
        if i == 176u32 { v = 0x19190808u32; }
        if i == 177u32 { v = 0x19190808u32; }
        if i == 178u32 { v = 0x19190808u32; }
        if i == 179u32 { v = 0x19190808u32; }
        if i == 180u32 { v = 0x19190808u32; }
        if i == 181u32 { v = 0x19190819u32; }
        if i == 182u32 { v = 0x19190819u32; }
        if i == 183u32 { v = 0x19190819u32; }
        if i == 184u32 { v = 0x1919082bu32; }
        if i == 185u32 { v = 0x19191908u32; }
        if i == 186u32 { v = 0x19191908u32; }
        if i == 187u32 { v = 0x19191919u32; }
        if i == 188u32 { v = 0x1919192bu32; }
        if i == 189u32 { v = 0x1919192bu32; }
        if i == 190u32 { v = 0x19192b08u32; }
        if i == 191u32 { v = 0x19192b08u32; }
        if i == 192u32 { v = 0x19192b08u32; }
        if i == 193u32 { v = 0x19192b08u32; }
        if i == 194u32 { v = 0x19192b19u32; }
        if i == 195u32 { v = 0x19192b2bu32; }
        if i == 196u32 { v = 0x192b0808u32; }
        if i == 197u32 { v = 0x192b0808u32; }
        if i == 198u32 { v = 0x192b0808u32; }
        if i == 199u32 { v = 0x192b0808u32; }
        if i == 200u32 { v = 0x192b0819u32; }
        if i == 201u32 { v = 0x192b0819u32; }
        if i == 202u32 { v = 0x192b082bu32; }
        if i == 203u32 { v = 0x192b082bu32; }
        if i == 204u32 { v = 0x192b1908u32; }
        if i == 205u32 { v = 0x192b1908u32; }
        if i == 206u32 { v = 0x192b1919u32; }
        if i == 207u32 { v = 0x192b1919u32; }
        if i == 208u32 { v = 0x192b1919u32; }
        if i == 209u32 { v = 0x192b2b08u32; }
        if i == 210u32 { v = 0x2b080808u32; }
        if i == 211u32 { v = 0x2b080808u32; }
        if i == 212u32 { v = 0x2b080808u32; }
        if i == 213u32 { v = 0x2b080808u32; }
        if i == 214u32 { v = 0x2b080808u32; }
        if i == 215u32 { v = 0x2b080819u32; }
        if i == 216u32 { v = 0x2b080819u32; }
        if i == 217u32 { v = 0x2b080819u32; }
        if i == 218u32 { v = 0x2b08082bu32; }
        if i == 219u32 { v = 0x2b081908u32; }
        if i == 220u32 { v = 0x2b081908u32; }
        if i == 221u32 { v = 0x2b081908u32; }
        if i == 222u32 { v = 0x2b081908u32; }
        if i == 223u32 { v = 0x2b081908u32; }
        if i == 224u32 { v = 0x2b081908u32; }
        if i == 225u32 { v = 0x2b081919u32; }
        if i == 226u32 { v = 0x2b081919u32; }
        if i == 227u32 { v = 0x2b081919u32; }
        if i == 228u32 { v = 0x2b08192bu32; }
        if i == 229u32 { v = 0x2b08192bu32; }
        if i == 230u32 { v = 0x2b08192bu32; }
        if i == 231u32 { v = 0x2b082b08u32; }
        if i == 232u32 { v = 0x2b082b19u32; }
        if i == 233u32 { v = 0x2b082b2bu32; }
        if i == 234u32 { v = 0x2b190808u32; }
        if i == 235u32 { v = 0x2b190808u32; }
        if i == 236u32 { v = 0x2b190808u32; }
        if i == 237u32 { v = 0x2b190808u32; }
        if i == 238u32 { v = 0x2b190808u32; }
        if i == 239u32 { v = 0x2b190819u32; }
        if i == 240u32 { v = 0x2b190819u32; }
        if i == 241u32 { v = 0x2b19082bu32; }
        if i == 242u32 { v = 0x2b191908u32; }
        if i == 243u32 { v = 0x2b191908u32; }
        if i == 244u32 { v = 0x2b191908u32; }
        if i == 245u32 { v = 0x2b191919u32; }
        if i == 246u32 { v = 0x2b192b08u32; }
        if i == 247u32 { v = 0x2b192b19u32; }
        if i == 248u32 { v = 0x2b2b0808u32; }
        if i == 249u32 { v = 0x2b2b0808u32; }
        if i == 250u32 { v = 0x2b2b0808u32; }
        if i == 251u32 { v = 0x2b2b0819u32; }
        if i == 252u32 { v = 0x2b2b082bu32; }
        if i == 253u32 { v = 0x2b2b1908u32; }
        if i == 254u32 { v = 0x2b2b2b08u32; }
        if i == 255u32 { v = 0x2b2b2b19u32; }
        v
    }
pub fn s_grid_lo(i: u32) -> u32 {
        let mut v: u32 = 0;
        if i == 0u32 { v = 0x08080808u32; }
        if i == 1u32 { v = 0x0808082bu32; }
        if i == 2u32 { v = 0x08081919u32; }
        if i == 3u32 { v = 0x08082b08u32; }
        if i == 4u32 { v = 0x08082b2bu32; }
        if i == 5u32 { v = 0x08190819u32; }
        if i == 6u32 { v = 0x08191908u32; }
        if i == 7u32 { v = 0x0819192bu32; }
        if i == 8u32 { v = 0x08192b19u32; }
        if i == 9u32 { v = 0x082b0808u32; }
        if i == 10u32 { v = 0x082b082bu32; }
        if i == 11u32 { v = 0x082b1919u32; }
        if i == 12u32 { v = 0x082b2b08u32; }
        if i == 13u32 { v = 0x19080819u32; }
        if i == 14u32 { v = 0x19081908u32; }
        if i == 15u32 { v = 0x1908192bu32; }
        if i == 16u32 { v = 0x19082b19u32; }
        if i == 17u32 { v = 0x19190808u32; }
        if i == 18u32 { v = 0x1919082bu32; }
        if i == 19u32 { v = 0x19191919u32; }
        if i == 20u32 { v = 0x19192b08u32; }
        if i == 21u32 { v = 0x192b0819u32; }
        if i == 22u32 { v = 0x192b1908u32; }
        if i == 23u32 { v = 0x192b192bu32; }
        if i == 24u32 { v = 0x192b2b19u32; }
        if i == 25u32 { v = 0x2b080808u32; }
        if i == 26u32 { v = 0x2b08082bu32; }
        if i == 27u32 { v = 0x2b081919u32; }
        if i == 28u32 { v = 0x2b082b08u32; }
        if i == 29u32 { v = 0x2b190819u32; }
        if i == 30u32 { v = 0x2b191908u32; }
        if i == 31u32 { v = 0x2b2b0808u32; }
        if i == 32u32 { v = 0x2b2b1919u32; }
        if i == 33u32 { v = 0x2b2b2b2bu32; }
        if i == 34u32 { v = 0x08080819u32; }
        if i == 35u32 { v = 0x08081908u32; }
        if i == 36u32 { v = 0x0808192bu32; }
        if i == 37u32 { v = 0x08082b19u32; }
        if i == 38u32 { v = 0x08190808u32; }
        if i == 39u32 { v = 0x0819082bu32; }
        if i == 40u32 { v = 0x08191919u32; }
        if i == 41u32 { v = 0x08192b08u32; }
        if i == 42u32 { v = 0x082b0819u32; }
        if i == 43u32 { v = 0x082b1908u32; }
        if i == 44u32 { v = 0x19080808u32; }
        if i == 45u32 { v = 0x1908082bu32; }
        if i == 46u32 { v = 0x19081919u32; }
        if i == 47u32 { v = 0x19082b08u32; }
        if i == 48u32 { v = 0x19190819u32; }
        if i == 49u32 { v = 0x19191908u32; }
        if i == 50u32 { v = 0x1919192bu32; }
        if i == 51u32 { v = 0x19192b19u32; }
        if i == 52u32 { v = 0x192b0808u32; }
        if i == 53u32 { v = 0x192b1919u32; }
        if i == 54u32 { v = 0x192b2b08u32; }
        if i == 55u32 { v = 0x2b080819u32; }
        if i == 56u32 { v = 0x2b081908u32; }
        if i == 57u32 { v = 0x2b190808u32; }
        if i == 58u32 { v = 0x2b19082bu32; }
        if i == 59u32 { v = 0x2b191919u32; }
        if i == 60u32 { v = 0x2b2b0819u32; }
        if i == 61u32 { v = 0x2b2b1908u32; }
        if i == 62u32 { v = 0x08080808u32; }
        if i == 63u32 { v = 0x0808082bu32; }
        if i == 64u32 { v = 0x08081919u32; }
        if i == 65u32 { v = 0x08082b08u32; }
        if i == 66u32 { v = 0x08190819u32; }
        if i == 67u32 { v = 0x08191908u32; }
        if i == 68u32 { v = 0x082b0808u32; }
        if i == 69u32 { v = 0x082b2b2bu32; }
        if i == 70u32 { v = 0x19080819u32; }
        if i == 71u32 { v = 0x19081908u32; }
        if i == 72u32 { v = 0x1908192bu32; }
        if i == 73u32 { v = 0x19082b19u32; }
        if i == 74u32 { v = 0x19190808u32; }
        if i == 75u32 { v = 0x19191919u32; }
        if i == 76u32 { v = 0x2b080808u32; }
        if i == 77u32 { v = 0x2b081919u32; }
        if i == 78u32 { v = 0x2b082b2bu32; }
        if i == 79u32 { v = 0x2b191908u32; }
        if i == 80u32 { v = 0x2b2b082bu32; }
        if i == 81u32 { v = 0x08080819u32; }
        if i == 82u32 { v = 0x08081908u32; }
        if i == 83u32 { v = 0x0808192bu32; }
        if i == 84u32 { v = 0x08082b19u32; }
        if i == 85u32 { v = 0x08190808u32; }
        if i == 86u32 { v = 0x0819082bu32; }
        if i == 87u32 { v = 0x08191919u32; }
        if i == 88u32 { v = 0x08192b08u32; }
        if i == 89u32 { v = 0x082b0819u32; }
        if i == 90u32 { v = 0x082b1908u32; }
        if i == 91u32 { v = 0x082b192bu32; }
        if i == 92u32 { v = 0x082b2b19u32; }
        if i == 93u32 { v = 0x19080808u32; }
        if i == 94u32 { v = 0x1908082bu32; }
        if i == 95u32 { v = 0x19081919u32; }
        if i == 96u32 { v = 0x19082b08u32; }
        if i == 97u32 { v = 0x19082b2bu32; }
        if i == 98u32 { v = 0x19190819u32; }
        if i == 99u32 { v = 0x19191908u32; }
        if i == 100u32 { v = 0x1919192bu32; }
        if i == 101u32 { v = 0x19192b19u32; }
        if i == 102u32 { v = 0x192b0808u32; }
        if i == 103u32 { v = 0x192b082bu32; }
        if i == 104u32 { v = 0x192b1919u32; }
        if i == 105u32 { v = 0x2b080819u32; }
        if i == 106u32 { v = 0x2b081908u32; }
        if i == 107u32 { v = 0x2b08192bu32; }
        if i == 108u32 { v = 0x2b082b19u32; }
        if i == 109u32 { v = 0x2b190808u32; }
        if i == 110u32 { v = 0x2b191919u32; }
        if i == 111u32 { v = 0x2b192b08u32; }
        if i == 112u32 { v = 0x2b2b0819u32; }
        if i == 113u32 { v = 0x2b2b1908u32; }
        if i == 114u32 { v = 0x08080808u32; }
        if i == 115u32 { v = 0x0808082bu32; }
        if i == 116u32 { v = 0x08081919u32; }
        if i == 117u32 { v = 0x08082b08u32; }
        if i == 118u32 { v = 0x08082b2bu32; }
        if i == 119u32 { v = 0x08190819u32; }
        if i == 120u32 { v = 0x08191908u32; }
        if i == 121u32 { v = 0x0819192bu32; }
        if i == 122u32 { v = 0x08192b19u32; }
        if i == 123u32 { v = 0x082b0808u32; }
        if i == 124u32 { v = 0x082b1919u32; }
        if i == 125u32 { v = 0x082b2b08u32; }
        if i == 126u32 { v = 0x19080819u32; }
        if i == 127u32 { v = 0x19081908u32; }
        if i == 128u32 { v = 0x1908192bu32; }
        if i == 129u32 { v = 0x19082b19u32; }
        if i == 130u32 { v = 0x19190808u32; }
        if i == 131u32 { v = 0x1919082bu32; }
        if i == 132u32 { v = 0x19191919u32; }
        if i == 133u32 { v = 0x19192b08u32; }
        if i == 134u32 { v = 0x192b0819u32; }
        if i == 135u32 { v = 0x192b1908u32; }
        if i == 136u32 { v = 0x2b080808u32; }
        if i == 137u32 { v = 0x2b08082bu32; }
        if i == 138u32 { v = 0x2b081919u32; }
        if i == 139u32 { v = 0x2b082b08u32; }
        if i == 140u32 { v = 0x2b190819u32; }
        if i == 141u32 { v = 0x2b191908u32; }
        if i == 142u32 { v = 0x2b2b0808u32; }
        if i == 143u32 { v = 0x08080819u32; }
        if i == 144u32 { v = 0x08081908u32; }
        if i == 145u32 { v = 0x0808192bu32; }
        if i == 146u32 { v = 0x08082b19u32; }
        if i == 147u32 { v = 0x08190808u32; }
        if i == 148u32 { v = 0x08191919u32; }
        if i == 149u32 { v = 0x19080808u32; }
        if i == 150u32 { v = 0x19081919u32; }
        if i == 151u32 { v = 0x19082b08u32; }
        if i == 152u32 { v = 0x19190819u32; }
        if i == 153u32 { v = 0x19191908u32; }
        if i == 154u32 { v = 0x192b0808u32; }
        if i == 155u32 { v = 0x2b080819u32; }
        if i == 156u32 { v = 0x2b081908u32; }
        if i == 157u32 { v = 0x2b190808u32; }
        if i == 158u32 { v = 0x08080808u32; }
        if i == 159u32 { v = 0x0808082bu32; }
        if i == 160u32 { v = 0x08081919u32; }
        if i == 161u32 { v = 0x08082b08u32; }
        if i == 162u32 { v = 0x08190819u32; }
        if i == 163u32 { v = 0x08191908u32; }
        if i == 164u32 { v = 0x0819192bu32; }
        if i == 165u32 { v = 0x08192b19u32; }
        if i == 166u32 { v = 0x082b0808u32; }
        if i == 167u32 { v = 0x082b1919u32; }
        if i == 168u32 { v = 0x082b2b2bu32; }
        if i == 169u32 { v = 0x19080819u32; }
        if i == 170u32 { v = 0x19081908u32; }
        if i == 171u32 { v = 0x1908192bu32; }
        if i == 172u32 { v = 0x19082b19u32; }
        if i == 173u32 { v = 0x19190808u32; }
        if i == 174u32 { v = 0x1919082bu32; }
        if i == 175u32 { v = 0x19191919u32; }
        if i == 176u32 { v = 0x19192b08u32; }
        if i == 177u32 { v = 0x192b0819u32; }
        if i == 178u32 { v = 0x192b1908u32; }
        if i == 179u32 { v = 0x2b080808u32; }
        if i == 180u32 { v = 0x2b081919u32; }
        if i == 181u32 { v = 0x2b191908u32; }
        if i == 182u32 { v = 0x2b2b2b2bu32; }
        if i == 183u32 { v = 0x08080819u32; }
        if i == 184u32 { v = 0x08081908u32; }
        if i == 185u32 { v = 0x08190808u32; }
        if i == 186u32 { v = 0x0819082bu32; }
        if i == 187u32 { v = 0x08191919u32; }
        if i == 188u32 { v = 0x08192b08u32; }
        if i == 189u32 { v = 0x082b0819u32; }
        if i == 190u32 { v = 0x19080808u32; }
        if i == 191u32 { v = 0x19081919u32; }
        if i == 192u32 { v = 0x19082b08u32; }
        if i == 193u32 { v = 0x19190819u32; }
        if i == 194u32 { v = 0x19191908u32; }
        if i == 195u32 { v = 0x192b0808u32; }
        if i == 196u32 { v = 0x2b080819u32; }
        if i == 197u32 { v = 0x2b190808u32; }
        if i == 198u32 { v = 0x08080808u32; }
        if i == 199u32 { v = 0x08190819u32; }
        if i == 200u32 { v = 0x08191908u32; }
        if i == 201u32 { v = 0x082b082bu32; }
        if i == 202u32 { v = 0x082b2b08u32; }
        if i == 203u32 { v = 0x082b2b2bu32; }
        if i == 204u32 { v = 0x19190808u32; }
        if i == 205u32 { v = 0x2b192b19u32; }
        if i == 206u32 { v = 0x08080819u32; }
        if i == 207u32 { v = 0x08081908u32; }
        if i == 208u32 { v = 0x0808192bu32; }
        if i == 209u32 { v = 0x08082b19u32; }
        if i == 210u32 { v = 0x08190808u32; }
        if i == 211u32 { v = 0x0819082bu32; }
        if i == 212u32 { v = 0x08191919u32; }
        if i == 213u32 { v = 0x08192b08u32; }
        if i == 214u32 { v = 0x082b0819u32; }
        if i == 215u32 { v = 0x082b1908u32; }
        if i == 216u32 { v = 0x082b192bu32; }
        if i == 217u32 { v = 0x19080808u32; }
        if i == 218u32 { v = 0x1908082bu32; }
        if i == 219u32 { v = 0x19081919u32; }
        if i == 220u32 { v = 0x19082b08u32; }
        if i == 221u32 { v = 0x19190819u32; }
        if i == 222u32 { v = 0x19191908u32; }
        if i == 223u32 { v = 0x1919192bu32; }
        if i == 224u32 { v = 0x19192b19u32; }
        if i == 225u32 { v = 0x192b0808u32; }
        if i == 226u32 { v = 0x192b082bu32; }
        if i == 227u32 { v = 0x192b1919u32; }
        if i == 228u32 { v = 0x192b2b08u32; }
        if i == 229u32 { v = 0x2b080819u32; }
        if i == 230u32 { v = 0x2b081908u32; }
        if i == 231u32 { v = 0x2b08192bu32; }
        if i == 232u32 { v = 0x2b190808u32; }
        if i == 233u32 { v = 0x2b191919u32; }
        if i == 234u32 { v = 0x2b192b08u32; }
        if i == 235u32 { v = 0x2b2b0819u32; }
        if i == 236u32 { v = 0x2b2b1908u32; }
        if i == 237u32 { v = 0x08080808u32; }
        if i == 238u32 { v = 0x0808082bu32; }
        if i == 239u32 { v = 0x08081919u32; }
        if i == 240u32 { v = 0x08082b08u32; }
        if i == 241u32 { v = 0x08082b2bu32; }
        if i == 242u32 { v = 0x08190819u32; }
        if i == 243u32 { v = 0x08191908u32; }
        if i == 244u32 { v = 0x0819192bu32; }
        if i == 245u32 { v = 0x08192b19u32; }
        if i == 246u32 { v = 0x082b0808u32; }
        if i == 247u32 { v = 0x082b082bu32; }
        if i == 248u32 { v = 0x082b1919u32; }
        if i == 249u32 { v = 0x082b2b08u32; }
        if i == 250u32 { v = 0x19080819u32; }
        if i == 251u32 { v = 0x19081908u32; }
        if i == 252u32 { v = 0x1908192bu32; }
        if i == 253u32 { v = 0x19082b19u32; }
        if i == 254u32 { v = 0x19190808u32; }
        if i == 255u32 { v = 0x1919082bu32; }
        if i == 256u32 { v = 0x19191919u32; }
        if i == 257u32 { v = 0x19192b08u32; }
        if i == 258u32 { v = 0x192b0819u32; }
        if i == 259u32 { v = 0x192b1908u32; }
        if i == 260u32 { v = 0x2b080808u32; }
        if i == 261u32 { v = 0x2b08082bu32; }
        if i == 262u32 { v = 0x2b081919u32; }
        if i == 263u32 { v = 0x2b082b08u32; }
        if i == 264u32 { v = 0x2b190819u32; }
        if i == 265u32 { v = 0x2b191908u32; }
        if i == 266u32 { v = 0x08080819u32; }
        if i == 267u32 { v = 0x08081908u32; }
        if i == 268u32 { v = 0x08082b19u32; }
        if i == 269u32 { v = 0x08190808u32; }
        if i == 270u32 { v = 0x08191919u32; }
        if i == 271u32 { v = 0x082b0819u32; }
        if i == 272u32 { v = 0x082b1908u32; }
        if i == 273u32 { v = 0x19080808u32; }
        if i == 274u32 { v = 0x19081919u32; }
        if i == 275u32 { v = 0x19190819u32; }
        if i == 276u32 { v = 0x19191908u32; }
        if i == 277u32 { v = 0x2b080819u32; }
        if i == 278u32 { v = 0x2b081908u32; }
        if i == 279u32 { v = 0x2b190808u32; }
        if i == 280u32 { v = 0x08080808u32; }
        if i == 281u32 { v = 0x0808082bu32; }
        if i == 282u32 { v = 0x08081919u32; }
        if i == 283u32 { v = 0x08082b08u32; }
        if i == 284u32 { v = 0x08190819u32; }
        if i == 285u32 { v = 0x08191908u32; }
        if i == 286u32 { v = 0x0819192bu32; }
        if i == 287u32 { v = 0x08192b19u32; }
        if i == 288u32 { v = 0x082b0808u32; }
        if i == 289u32 { v = 0x082b1919u32; }
        if i == 290u32 { v = 0x082b2b08u32; }
        if i == 291u32 { v = 0x19080819u32; }
        if i == 292u32 { v = 0x19081908u32; }
        if i == 293u32 { v = 0x1908192bu32; }
        if i == 294u32 { v = 0x19082b19u32; }
        if i == 295u32 { v = 0x19190808u32; }
        if i == 296u32 { v = 0x1919082bu32; }
        if i == 297u32 { v = 0x19191919u32; }
        if i == 298u32 { v = 0x19192b08u32; }
        if i == 299u32 { v = 0x192b0819u32; }
        if i == 300u32 { v = 0x192b1908u32; }
        if i == 301u32 { v = 0x2b080808u32; }
        if i == 302u32 { v = 0x2b08082bu32; }
        if i == 303u32 { v = 0x2b081919u32; }
        if i == 304u32 { v = 0x2b082b08u32; }
        if i == 305u32 { v = 0x2b190819u32; }
        if i == 306u32 { v = 0x2b191908u32; }
        if i == 307u32 { v = 0x2b2b0808u32; }
        if i == 308u32 { v = 0x08080819u32; }
        if i == 309u32 { v = 0x08081908u32; }
        if i == 310u32 { v = 0x0808192bu32; }
        if i == 311u32 { v = 0x08082b19u32; }
        if i == 312u32 { v = 0x08190808u32; }
        if i == 313u32 { v = 0x0819082bu32; }
        if i == 314u32 { v = 0x08191919u32; }
        if i == 315u32 { v = 0x08192b08u32; }
        if i == 316u32 { v = 0x082b0819u32; }
        if i == 317u32 { v = 0x082b1908u32; }
        if i == 318u32 { v = 0x19080808u32; }
        if i == 319u32 { v = 0x1908082bu32; }
        if i == 320u32 { v = 0x19081919u32; }
        if i == 321u32 { v = 0x19082b08u32; }
        if i == 322u32 { v = 0x19190819u32; }
        if i == 323u32 { v = 0x19191908u32; }
        if i == 324u32 { v = 0x192b0808u32; }
        if i == 325u32 { v = 0x2b080819u32; }
        if i == 326u32 { v = 0x2b081908u32; }
        if i == 327u32 { v = 0x2b190808u32; }
        if i == 328u32 { v = 0x08080808u32; }
        if i == 329u32 { v = 0x08081919u32; }
        if i == 330u32 { v = 0x08082b08u32; }
        if i == 331u32 { v = 0x08190819u32; }
        if i == 332u32 { v = 0x08191908u32; }
        if i == 333u32 { v = 0x082b0808u32; }
        if i == 334u32 { v = 0x19080819u32; }
        if i == 335u32 { v = 0x19081908u32; }
        if i == 336u32 { v = 0x19190808u32; }
        if i == 337u32 { v = 0x2b080808u32; }
        if i == 338u32 { v = 0x2b2b2b2bu32; }
        if i == 339u32 { v = 0x08080819u32; }
        if i == 340u32 { v = 0x08081908u32; }
        if i == 341u32 { v = 0x0808192bu32; }
        if i == 342u32 { v = 0x08082b19u32; }
        if i == 343u32 { v = 0x08190808u32; }
        if i == 344u32 { v = 0x08191919u32; }
        if i == 345u32 { v = 0x08192b08u32; }
        if i == 346u32 { v = 0x082b0819u32; }
        if i == 347u32 { v = 0x19080808u32; }
        if i == 348u32 { v = 0x1908082bu32; }
        if i == 349u32 { v = 0x19081919u32; }
        if i == 350u32 { v = 0x19082b08u32; }
        if i == 351u32 { v = 0x19190819u32; }
        if i == 352u32 { v = 0x19191908u32; }
        if i == 353u32 { v = 0x192b0808u32; }
        if i == 354u32 { v = 0x2b080819u32; }
        if i == 355u32 { v = 0x2b081908u32; }
        if i == 356u32 { v = 0x08080808u32; }
        if i == 357u32 { v = 0x0808082bu32; }
        if i == 358u32 { v = 0x08081919u32; }
        if i == 359u32 { v = 0x08082b08u32; }
        if i == 360u32 { v = 0x08190819u32; }
        if i == 361u32 { v = 0x08191908u32; }
        if i == 362u32 { v = 0x082b0808u32; }
        if i == 363u32 { v = 0x19080819u32; }
        if i == 364u32 { v = 0x19081908u32; }
        if i == 365u32 { v = 0x19190808u32; }
        if i == 366u32 { v = 0x192b2b19u32; }
        if i == 367u32 { v = 0x2b2b082bu32; }
        if i == 368u32 { v = 0x08081908u32; }
        if i == 369u32 { v = 0x08190808u32; }
        if i == 370u32 { v = 0x19080808u32; }
        if i == 371u32 { v = 0x1919192bu32; }
        if i == 372u32 { v = 0x08080808u32; }
        if i == 373u32 { v = 0x0808082bu32; }
        if i == 374u32 { v = 0x08081919u32; }
        if i == 375u32 { v = 0x08082b08u32; }
        if i == 376u32 { v = 0x08190819u32; }
        if i == 377u32 { v = 0x08191908u32; }
        if i == 378u32 { v = 0x0819192bu32; }
        if i == 379u32 { v = 0x08192b19u32; }
        if i == 380u32 { v = 0x082b0808u32; }
        if i == 381u32 { v = 0x082b1919u32; }
        if i == 382u32 { v = 0x082b2b2bu32; }
        if i == 383u32 { v = 0x19080819u32; }
        if i == 384u32 { v = 0x19081908u32; }
        if i == 385u32 { v = 0x19190808u32; }
        if i == 386u32 { v = 0x1919082bu32; }
        if i == 387u32 { v = 0x19191919u32; }
        if i == 388u32 { v = 0x192b1908u32; }
        if i == 389u32 { v = 0x2b080808u32; }
        if i == 390u32 { v = 0x2b082b2bu32; }
        if i == 391u32 { v = 0x2b191908u32; }
        if i == 392u32 { v = 0x2b2b2b2bu32; }
        if i == 393u32 { v = 0x08080819u32; }
        if i == 394u32 { v = 0x08081908u32; }
        if i == 395u32 { v = 0x08190808u32; }
        if i == 396u32 { v = 0x0819082bu32; }
        if i == 397u32 { v = 0x08191919u32; }
        if i == 398u32 { v = 0x082b0819u32; }
        if i == 399u32 { v = 0x19080808u32; }
        if i == 400u32 { v = 0x1908082bu32; }
        if i == 401u32 { v = 0x19081919u32; }
        if i == 402u32 { v = 0x19190819u32; }
        if i == 403u32 { v = 0x19191908u32; }
        if i == 404u32 { v = 0x192b0808u32; }
        if i == 405u32 { v = 0x2b080819u32; }
        if i == 406u32 { v = 0x2b081908u32; }
        if i == 407u32 { v = 0x2b190808u32; }
        if i == 408u32 { v = 0x08080808u32; }
        if i == 409u32 { v = 0x08082b2bu32; }
        if i == 410u32 { v = 0x082b082bu32; }
        if i == 411u32 { v = 0x082b2b08u32; }
        if i == 412u32 { v = 0x082b2b2bu32; }
        if i == 413u32 { v = 0x19081908u32; }
        if i == 414u32 { v = 0x19190808u32; }
        if i == 415u32 { v = 0x2b082b08u32; }
        if i == 416u32 { v = 0x2b082b2bu32; }
        if i == 417u32 { v = 0x2b2b2b08u32; }
        if i == 418u32 { v = 0x08080819u32; }
        if i == 419u32 { v = 0x08081908u32; }
        if i == 420u32 { v = 0x0808192bu32; }
        if i == 421u32 { v = 0x08082b19u32; }
        if i == 422u32 { v = 0x08190808u32; }
        if i == 423u32 { v = 0x08191919u32; }
        if i == 424u32 { v = 0x08192b08u32; }
        if i == 425u32 { v = 0x082b0819u32; }
        if i == 426u32 { v = 0x082b1908u32; }
        if i == 427u32 { v = 0x19080808u32; }
        if i == 428u32 { v = 0x1908082bu32; }
        if i == 429u32 { v = 0x19081919u32; }
        if i == 430u32 { v = 0x19082b08u32; }
        if i == 431u32 { v = 0x19190819u32; }
        if i == 432u32 { v = 0x19191908u32; }
        if i == 433u32 { v = 0x192b0808u32; }
        if i == 434u32 { v = 0x2b080819u32; }
        if i == 435u32 { v = 0x2b081908u32; }
        if i == 436u32 { v = 0x2b190808u32; }
        if i == 437u32 { v = 0x08080808u32; }
        if i == 438u32 { v = 0x08081919u32; }
        if i == 439u32 { v = 0x08082b08u32; }
        if i == 440u32 { v = 0x08190819u32; }
        if i == 441u32 { v = 0x08191908u32; }
        if i == 442u32 { v = 0x082b0808u32; }
        if i == 443u32 { v = 0x19080819u32; }
        if i == 444u32 { v = 0x19081908u32; }
        if i == 445u32 { v = 0x19190808u32; }
        if i == 446u32 { v = 0x192b192bu32; }
        if i == 447u32 { v = 0x2b080808u32; }
        if i == 448u32 { v = 0x08080819u32; }
        if i == 449u32 { v = 0x08081908u32; }
        if i == 450u32 { v = 0x08190808u32; }
        if i == 451u32 { v = 0x19080808u32; }
        if i == 452u32 { v = 0x19192b19u32; }
        if i == 453u32 { v = 0x08080808u32; }
        if i == 454u32 { v = 0x08081919u32; }
        if i == 455u32 { v = 0x08190819u32; }
        if i == 456u32 { v = 0x08191908u32; }
        if i == 457u32 { v = 0x19080819u32; }
        if i == 458u32 { v = 0x19081908u32; }
        if i == 459u32 { v = 0x19190808u32; }
        if i == 460u32 { v = 0x2b082b2bu32; }
        if i == 461u32 { v = 0x2b2b2b2bu32; }
        if i == 462u32 { v = 0x08080819u32; }
        if i == 463u32 { v = 0x08081908u32; }
        if i == 464u32 { v = 0x08190808u32; }
        if i == 465u32 { v = 0x2b191919u32; }
        if i == 466u32 { v = 0x08082b2bu32; }
        if i == 467u32 { v = 0x082b082bu32; }
        if i == 468u32 { v = 0x192b1908u32; }
        if i == 469u32 { v = 0x2b082b08u32; }
        if i == 470u32 { v = 0x2b082b2bu32; }
        if i == 471u32 { v = 0x08080819u32; }
        if i == 472u32 { v = 0x08081908u32; }
        if i == 473u32 { v = 0x0808192bu32; }
        if i == 474u32 { v = 0x08082b19u32; }
        if i == 475u32 { v = 0x08190808u32; }
        if i == 476u32 { v = 0x0819082bu32; }
        if i == 477u32 { v = 0x08191919u32; }
        if i == 478u32 { v = 0x08192b08u32; }
        if i == 479u32 { v = 0x08192b2bu32; }
        if i == 480u32 { v = 0x082b0819u32; }
        if i == 481u32 { v = 0x082b1908u32; }
        if i == 482u32 { v = 0x082b192bu32; }
        if i == 483u32 { v = 0x19080808u32; }
        if i == 484u32 { v = 0x1908082bu32; }
        if i == 485u32 { v = 0x19081919u32; }
        if i == 486u32 { v = 0x19082b08u32; }
        if i == 487u32 { v = 0x19082b2bu32; }
        if i == 488u32 { v = 0x19190819u32; }
        if i == 489u32 { v = 0x19191908u32; }
        if i == 490u32 { v = 0x1919192bu32; }
        if i == 491u32 { v = 0x19192b19u32; }
        if i == 492u32 { v = 0x192b0808u32; }
        if i == 493u32 { v = 0x192b082bu32; }
        if i == 494u32 { v = 0x192b1919u32; }
        if i == 495u32 { v = 0x2b080819u32; }
        if i == 496u32 { v = 0x2b081908u32; }
        if i == 497u32 { v = 0x2b190808u32; }
        if i == 498u32 { v = 0x2b191919u32; }
        if i == 499u32 { v = 0x2b192b08u32; }
        if i == 500u32 { v = 0x2b2b0819u32; }
        if i == 501u32 { v = 0x2b2b1908u32; }
        if i == 502u32 { v = 0x08080808u32; }
        if i == 503u32 { v = 0x0808082bu32; }
        if i == 504u32 { v = 0x08081919u32; }
        if i == 505u32 { v = 0x08082b08u32; }
        if i == 506u32 { v = 0x08190819u32; }
        if i == 507u32 { v = 0x08191908u32; }
        if i == 508u32 { v = 0x0819192bu32; }
        if i == 509u32 { v = 0x08192b19u32; }
        if i == 510u32 { v = 0x082b0808u32; }
        if i == 511u32 { v = 0x082b082bu32; }
        if i == 512u32 { v = 0x082b1919u32; }
        if i == 513u32 { v = 0x19080819u32; }
        if i == 514u32 { v = 0x19081908u32; }
        if i == 515u32 { v = 0x1908192bu32; }
        if i == 516u32 { v = 0x19082b19u32; }
        if i == 517u32 { v = 0x19190808u32; }
        if i == 518u32 { v = 0x1919082bu32; }
        if i == 519u32 { v = 0x19191919u32; }
        if i == 520u32 { v = 0x19192b08u32; }
        if i == 521u32 { v = 0x192b0819u32; }
        if i == 522u32 { v = 0x192b1908u32; }
        if i == 523u32 { v = 0x2b080808u32; }
        if i == 524u32 { v = 0x2b08082bu32; }
        if i == 525u32 { v = 0x2b081919u32; }
        if i == 526u32 { v = 0x2b082b08u32; }
        if i == 527u32 { v = 0x2b190819u32; }
        if i == 528u32 { v = 0x2b191908u32; }
        if i == 529u32 { v = 0x2b2b0808u32; }
        if i == 530u32 { v = 0x08080819u32; }
        if i == 531u32 { v = 0x08081908u32; }
        if i == 532u32 { v = 0x08190808u32; }
        if i == 533u32 { v = 0x0819082bu32; }
        if i == 534u32 { v = 0x08191919u32; }
        if i == 535u32 { v = 0x08192b08u32; }
        if i == 536u32 { v = 0x082b1908u32; }
        if i == 537u32 { v = 0x19080808u32; }
        if i == 538u32 { v = 0x19081919u32; }
        if i == 539u32 { v = 0x19082b08u32; }
        if i == 540u32 { v = 0x19190819u32; }
        if i == 541u32 { v = 0x19191908u32; }
        if i == 542u32 { v = 0x192b0808u32; }
        if i == 543u32 { v = 0x2b080819u32; }
        if i == 544u32 { v = 0x2b081908u32; }
        if i == 545u32 { v = 0x08080808u32; }
        if i == 546u32 { v = 0x0808082bu32; }
        if i == 547u32 { v = 0x08081919u32; }
        if i == 548u32 { v = 0x08082b08u32; }
        if i == 549u32 { v = 0x08082b2bu32; }
        if i == 550u32 { v = 0x08190819u32; }
        if i == 551u32 { v = 0x08191908u32; }
        if i == 552u32 { v = 0x0819192bu32; }
        if i == 553u32 { v = 0x08192b19u32; }
        if i == 554u32 { v = 0x082b0808u32; }
        if i == 555u32 { v = 0x082b082bu32; }
        if i == 556u32 { v = 0x082b1919u32; }
        if i == 557u32 { v = 0x082b2b08u32; }
        if i == 558u32 { v = 0x19080819u32; }
        if i == 559u32 { v = 0x19081908u32; }
        if i == 560u32 { v = 0x1908192bu32; }
        if i == 561u32 { v = 0x19082b19u32; }
        if i == 562u32 { v = 0x19190808u32; }
        if i == 563u32 { v = 0x1919082bu32; }
        if i == 564u32 { v = 0x19191919u32; }
        if i == 565u32 { v = 0x19192b08u32; }
        if i == 566u32 { v = 0x192b0819u32; }
        if i == 567u32 { v = 0x192b1908u32; }
        if i == 568u32 { v = 0x2b080808u32; }
        if i == 569u32 { v = 0x2b08082bu32; }
        if i == 570u32 { v = 0x2b081919u32; }
        if i == 571u32 { v = 0x2b082b08u32; }
        if i == 572u32 { v = 0x2b190819u32; }
        if i == 573u32 { v = 0x2b191908u32; }
        if i == 574u32 { v = 0x2b2b0808u32; }
        if i == 575u32 { v = 0x08080819u32; }
        if i == 576u32 { v = 0x08081908u32; }
        if i == 577u32 { v = 0x0808192bu32; }
        if i == 578u32 { v = 0x08082b19u32; }
        if i == 579u32 { v = 0x08190808u32; }
        if i == 580u32 { v = 0x0819082bu32; }
        if i == 581u32 { v = 0x08191919u32; }
        if i == 582u32 { v = 0x08192b08u32; }
        if i == 583u32 { v = 0x082b0819u32; }
        if i == 584u32 { v = 0x082b1908u32; }
        if i == 585u32 { v = 0x19080808u32; }
        if i == 586u32 { v = 0x1908082bu32; }
        if i == 587u32 { v = 0x19081919u32; }
        if i == 588u32 { v = 0x19082b08u32; }
        if i == 589u32 { v = 0x19190819u32; }
        if i == 590u32 { v = 0x19191908u32; }
        if i == 591u32 { v = 0x192b0808u32; }
        if i == 592u32 { v = 0x192b2b2bu32; }
        if i == 593u32 { v = 0x2b080819u32; }
        if i == 594u32 { v = 0x2b081908u32; }
        if i == 595u32 { v = 0x2b190808u32; }
        if i == 596u32 { v = 0x08080808u32; }
        if i == 597u32 { v = 0x0808082bu32; }
        if i == 598u32 { v = 0x08081919u32; }
        if i == 599u32 { v = 0x08082b08u32; }
        if i == 600u32 { v = 0x08190819u32; }
        if i == 601u32 { v = 0x08191908u32; }
        if i == 602u32 { v = 0x082b0808u32; }
        if i == 603u32 { v = 0x19080819u32; }
        if i == 604u32 { v = 0x19081908u32; }
        if i == 605u32 { v = 0x19190808u32; }
        if i == 606u32 { v = 0x2b080808u32; }
        if i == 607u32 { v = 0x2b2b1919u32; }
        if i == 608u32 { v = 0x08080819u32; }
        if i == 609u32 { v = 0x08081908u32; }
        if i == 610u32 { v = 0x08082b19u32; }
        if i == 611u32 { v = 0x08190808u32; }
        if i == 612u32 { v = 0x0819082bu32; }
        if i == 613u32 { v = 0x08191919u32; }
        if i == 614u32 { v = 0x08192b08u32; }
        if i == 615u32 { v = 0x082b0819u32; }
        if i == 616u32 { v = 0x082b1908u32; }
        if i == 617u32 { v = 0x19080808u32; }
        if i == 618u32 { v = 0x1908082bu32; }
        if i == 619u32 { v = 0x19081919u32; }
        if i == 620u32 { v = 0x19082b08u32; }
        if i == 621u32 { v = 0x19190819u32; }
        if i == 622u32 { v = 0x19191908u32; }
        if i == 623u32 { v = 0x192b0808u32; }
        if i == 624u32 { v = 0x2b081908u32; }
        if i == 625u32 { v = 0x2b190808u32; }
        if i == 626u32 { v = 0x08080808u32; }
        if i == 627u32 { v = 0x0808082bu32; }
        if i == 628u32 { v = 0x08081919u32; }
        if i == 629u32 { v = 0x08082b08u32; }
        if i == 630u32 { v = 0x08190819u32; }
        if i == 631u32 { v = 0x08191908u32; }
        if i == 632u32 { v = 0x082b0808u32; }
        if i == 633u32 { v = 0x19080819u32; }
        if i == 634u32 { v = 0x19081908u32; }
        if i == 635u32 { v = 0x19190808u32; }
        if i == 636u32 { v = 0x2b080808u32; }
        if i == 637u32 { v = 0x2b19192bu32; }
        if i == 638u32 { v = 0x08080819u32; }
        if i == 639u32 { v = 0x08081908u32; }
        if i == 640u32 { v = 0x08190808u32; }
        if i == 641u32 { v = 0x19080808u32; }
        if i == 642u32 { v = 0x08080808u32; }
        if i == 643u32 { v = 0x0808082bu32; }
        if i == 644u32 { v = 0x08081919u32; }
        if i == 645u32 { v = 0x08082b08u32; }
        if i == 646u32 { v = 0x08190819u32; }
        if i == 647u32 { v = 0x08191908u32; }
        if i == 648u32 { v = 0x0819192bu32; }
        if i == 649u32 { v = 0x08192b19u32; }
        if i == 650u32 { v = 0x082b0808u32; }
        if i == 651u32 { v = 0x082b082bu32; }
        if i == 652u32 { v = 0x082b1919u32; }
        if i == 653u32 { v = 0x082b2b08u32; }
        if i == 654u32 { v = 0x19080819u32; }
        if i == 655u32 { v = 0x19081908u32; }
        if i == 656u32 { v = 0x1908192bu32; }
        if i == 657u32 { v = 0x19082b19u32; }
        if i == 658u32 { v = 0x19190808u32; }
        if i == 659u32 { v = 0x1919082bu32; }
        if i == 660u32 { v = 0x19191919u32; }
        if i == 661u32 { v = 0x19192b08u32; }
        if i == 662u32 { v = 0x192b0819u32; }
        if i == 663u32 { v = 0x192b1908u32; }
        if i == 664u32 { v = 0x2b080808u32; }
        if i == 665u32 { v = 0x2b08082bu32; }
        if i == 666u32 { v = 0x2b081919u32; }
        if i == 667u32 { v = 0x2b082b08u32; }
        if i == 668u32 { v = 0x2b190819u32; }
        if i == 669u32 { v = 0x2b191908u32; }
        if i == 670u32 { v = 0x08080819u32; }
        if i == 671u32 { v = 0x08081908u32; }
        if i == 672u32 { v = 0x0808192bu32; }
        if i == 673u32 { v = 0x08082b19u32; }
        if i == 674u32 { v = 0x08190808u32; }
        if i == 675u32 { v = 0x0819082bu32; }
        if i == 676u32 { v = 0x08191919u32; }
        if i == 677u32 { v = 0x08192b08u32; }
        if i == 678u32 { v = 0x082b0819u32; }
        if i == 679u32 { v = 0x082b1908u32; }
        if i == 680u32 { v = 0x19080808u32; }
        if i == 681u32 { v = 0x1908082bu32; }
        if i == 682u32 { v = 0x19081919u32; }
        if i == 683u32 { v = 0x19082b08u32; }
        if i == 684u32 { v = 0x19190819u32; }
        if i == 685u32 { v = 0x19191908u32; }
        if i == 686u32 { v = 0x192b0808u32; }
        if i == 687u32 { v = 0x2b080819u32; }
        if i == 688u32 { v = 0x2b081908u32; }
        if i == 689u32 { v = 0x2b190808u32; }
        if i == 690u32 { v = 0x08080808u32; }
        if i == 691u32 { v = 0x08081919u32; }
        if i == 692u32 { v = 0x08082b08u32; }
        if i == 693u32 { v = 0x08190819u32; }
        if i == 694u32 { v = 0x08191908u32; }
        if i == 695u32 { v = 0x082b0808u32; }
        if i == 696u32 { v = 0x19080819u32; }
        if i == 697u32 { v = 0x19081908u32; }
        if i == 698u32 { v = 0x19190808u32; }
        if i == 699u32 { v = 0x192b2b19u32; }
        if i == 700u32 { v = 0x2b080808u32; }
        if i == 701u32 { v = 0x08080819u32; }
        if i == 702u32 { v = 0x08081908u32; }
        if i == 703u32 { v = 0x0808192bu32; }
        if i == 704u32 { v = 0x08082b19u32; }
        if i == 705u32 { v = 0x08190808u32; }
        if i == 706u32 { v = 0x0819082bu32; }
        if i == 707u32 { v = 0x08191919u32; }
        if i == 708u32 { v = 0x08192b08u32; }
        if i == 709u32 { v = 0x082b0819u32; }
        if i == 710u32 { v = 0x082b1908u32; }
        if i == 711u32 { v = 0x19080808u32; }
        if i == 712u32 { v = 0x1908082bu32; }
        if i == 713u32 { v = 0x19081919u32; }
        if i == 714u32 { v = 0x19082b08u32; }
        if i == 715u32 { v = 0x19190819u32; }
        if i == 716u32 { v = 0x19191908u32; }
        if i == 717u32 { v = 0x192b0808u32; }
        if i == 718u32 { v = 0x2b080819u32; }
        if i == 719u32 { v = 0x2b081908u32; }
        if i == 720u32 { v = 0x2b190808u32; }
        if i == 721u32 { v = 0x08080808u32; }
        if i == 722u32 { v = 0x0808082bu32; }
        if i == 723u32 { v = 0x08081919u32; }
        if i == 724u32 { v = 0x08082b08u32; }
        if i == 725u32 { v = 0x08190819u32; }
        if i == 726u32 { v = 0x08191908u32; }
        if i == 727u32 { v = 0x082b0808u32; }
        if i == 728u32 { v = 0x19080819u32; }
        if i == 729u32 { v = 0x19081908u32; }
        if i == 730u32 { v = 0x19190808u32; }
        if i == 731u32 { v = 0x2b080808u32; }
        if i == 732u32 { v = 0x08080819u32; }
        if i == 733u32 { v = 0x08081908u32; }
        if i == 734u32 { v = 0x08190808u32; }
        if i == 735u32 { v = 0x082b192bu32; }
        if i == 736u32 { v = 0x19080808u32; }
        if i == 737u32 { v = 0x08080808u32; }
        if i == 738u32 { v = 0x0808082bu32; }
        if i == 739u32 { v = 0x08081919u32; }
        if i == 740u32 { v = 0x08082b08u32; }
        if i == 741u32 { v = 0x08190819u32; }
        if i == 742u32 { v = 0x08191908u32; }
        if i == 743u32 { v = 0x082b0808u32; }
        if i == 744u32 { v = 0x19080819u32; }
        if i == 745u32 { v = 0x19081908u32; }
        if i == 746u32 { v = 0x19190808u32; }
        if i == 747u32 { v = 0x19192b2bu32; }
        if i == 748u32 { v = 0x2b080808u32; }
        if i == 749u32 { v = 0x08080819u32; }
        if i == 750u32 { v = 0x08081908u32; }
        if i == 751u32 { v = 0x08190808u32; }
        if i == 752u32 { v = 0x19080808u32; }
        if i == 753u32 { v = 0x08080808u32; }
        if i == 754u32 { v = 0x08192b19u32; }
        if i == 755u32 { v = 0x2b081919u32; }
        if i == 756u32 { v = 0x2b2b2b08u32; }
        if i == 757u32 { v = 0x08080819u32; }
        if i == 758u32 { v = 0x08081908u32; }
        if i == 759u32 { v = 0x0808192bu32; }
        if i == 760u32 { v = 0x08190808u32; }
        if i == 761u32 { v = 0x0819082bu32; }
        if i == 762u32 { v = 0x08191919u32; }
        if i == 763u32 { v = 0x08192b08u32; }
        if i == 764u32 { v = 0x082b0819u32; }
        if i == 765u32 { v = 0x082b1908u32; }
        if i == 766u32 { v = 0x19080808u32; }
        if i == 767u32 { v = 0x19081919u32; }
        if i == 768u32 { v = 0x19082b08u32; }
        if i == 769u32 { v = 0x19190819u32; }
        if i == 770u32 { v = 0x19191908u32; }
        if i == 771u32 { v = 0x192b0808u32; }
        if i == 772u32 { v = 0x2b081908u32; }
        if i == 773u32 { v = 0x2b190808u32; }
        if i == 774u32 { v = 0x08080808u32; }
        if i == 775u32 { v = 0x0808082bu32; }
        if i == 776u32 { v = 0x08081919u32; }
        if i == 777u32 { v = 0x08082b08u32; }
        if i == 778u32 { v = 0x08190819u32; }
        if i == 779u32 { v = 0x08191908u32; }
        if i == 780u32 { v = 0x082b0808u32; }
        if i == 781u32 { v = 0x19080819u32; }
        if i == 782u32 { v = 0x19081908u32; }
        if i == 783u32 { v = 0x19190808u32; }
        if i == 784u32 { v = 0x2b080808u32; }
        if i == 785u32 { v = 0x2b192b19u32; }
        if i == 786u32 { v = 0x08081908u32; }
        if i == 787u32 { v = 0x08190808u32; }
        if i == 788u32 { v = 0x19080808u32; }
        if i == 789u32 { v = 0x1919192bu32; }
        if i == 790u32 { v = 0x2b2b0819u32; }
        if i == 791u32 { v = 0x08080808u32; }
        if i == 792u32 { v = 0x08081919u32; }
        if i == 793u32 { v = 0x08082b08u32; }
        if i == 794u32 { v = 0x08190819u32; }
        if i == 795u32 { v = 0x08191908u32; }
        if i == 796u32 { v = 0x082b0808u32; }
        if i == 797u32 { v = 0x19080819u32; }
        if i == 798u32 { v = 0x19081908u32; }
        if i == 799u32 { v = 0x19190808u32; }
        if i == 800u32 { v = 0x2b080808u32; }
        if i == 801u32 { v = 0x08080819u32; }
        if i == 802u32 { v = 0x08081908u32; }
        if i == 803u32 { v = 0x08190808u32; }
        if i == 804u32 { v = 0x19080808u32; }
        if i == 805u32 { v = 0x19082b2bu32; }
        if i == 806u32 { v = 0x192b2b08u32; }
        if i == 807u32 { v = 0x2b19082bu32; }
        if i == 808u32 { v = 0x08080808u32; }
        if i == 809u32 { v = 0x2b191908u32; }
        if i == 810u32 { v = 0x08080819u32; }
        if i == 811u32 { v = 0x08081908u32; }
        if i == 812u32 { v = 0x08190808u32; }
        if i == 813u32 { v = 0x192b1919u32; }
        if i == 814u32 { v = 0x2b192b08u32; }
        if i == 815u32 { v = 0x08080808u32; }
        if i == 816u32 { v = 0x082b2b2bu32; }
        if i == 817u32 { v = 0x1908082bu32; }
        if i == 818u32 { v = 0x2b2b0819u32; }
        if i == 819u32 { v = 0x08080808u32; }
        if i == 820u32 { v = 0x0808082bu32; }
        if i == 821u32 { v = 0x08081919u32; }
        if i == 822u32 { v = 0x08082b08u32; }
        if i == 823u32 { v = 0x08190819u32; }
        if i == 824u32 { v = 0x08191908u32; }
        if i == 825u32 { v = 0x08192b19u32; }
        if i == 826u32 { v = 0x082b0808u32; }
        if i == 827u32 { v = 0x082b1919u32; }
        if i == 828u32 { v = 0x19080819u32; }
        if i == 829u32 { v = 0x19081908u32; }
        if i == 830u32 { v = 0x19190808u32; }
        if i == 831u32 { v = 0x1919082bu32; }
        if i == 832u32 { v = 0x19191919u32; }
        if i == 833u32 { v = 0x19192b08u32; }
        if i == 834u32 { v = 0x192b0819u32; }
        if i == 835u32 { v = 0x2b080808u32; }
        if i == 836u32 { v = 0x2b081919u32; }
        if i == 837u32 { v = 0x2b190819u32; }
        if i == 838u32 { v = 0x2b191908u32; }
        if i == 839u32 { v = 0x08080819u32; }
        if i == 840u32 { v = 0x08081908u32; }
        if i == 841u32 { v = 0x08082b19u32; }
        if i == 842u32 { v = 0x08190808u32; }
        if i == 843u32 { v = 0x0819082bu32; }
        if i == 844u32 { v = 0x08191919u32; }
        if i == 845u32 { v = 0x08192b08u32; }
        if i == 846u32 { v = 0x082b0819u32; }
        if i == 847u32 { v = 0x082b1908u32; }
        if i == 848u32 { v = 0x19080808u32; }
        if i == 849u32 { v = 0x1908082bu32; }
        if i == 850u32 { v = 0x19081919u32; }
        if i == 851u32 { v = 0x19082b08u32; }
        if i == 852u32 { v = 0x19190819u32; }
        if i == 853u32 { v = 0x19191908u32; }
        if i == 854u32 { v = 0x2b080819u32; }
        if i == 855u32 { v = 0x2b081908u32; }
        if i == 856u32 { v = 0x2b190808u32; }
        if i == 857u32 { v = 0x2b2b2b19u32; }
        if i == 858u32 { v = 0x08080808u32; }
        if i == 859u32 { v = 0x08081919u32; }
        if i == 860u32 { v = 0x08082b2bu32; }
        if i == 861u32 { v = 0x08190819u32; }
        if i == 862u32 { v = 0x08191908u32; }
        if i == 863u32 { v = 0x19080819u32; }
        if i == 864u32 { v = 0x19081908u32; }
        if i == 865u32 { v = 0x19190808u32; }
        if i == 866u32 { v = 0x08080819u32; }
        if i == 867u32 { v = 0x08081908u32; }
        if i == 868u32 { v = 0x0808192bu32; }
        if i == 869u32 { v = 0x08082b19u32; }
        if i == 870u32 { v = 0x08190808u32; }
        if i == 871u32 { v = 0x0819082bu32; }
        if i == 872u32 { v = 0x08191919u32; }
        if i == 873u32 { v = 0x08192b08u32; }
        if i == 874u32 { v = 0x082b0819u32; }
        if i == 875u32 { v = 0x19080808u32; }
        if i == 876u32 { v = 0x1908082bu32; }
        if i == 877u32 { v = 0x19081919u32; }
        if i == 878u32 { v = 0x19082b08u32; }
        if i == 879u32 { v = 0x19190819u32; }
        if i == 880u32 { v = 0x19191908u32; }
        if i == 881u32 { v = 0x192b0808u32; }
        if i == 882u32 { v = 0x2b080819u32; }
        if i == 883u32 { v = 0x2b081908u32; }
        if i == 884u32 { v = 0x2b190808u32; }
        if i == 885u32 { v = 0x08080808u32; }
        if i == 886u32 { v = 0x0808082bu32; }
        if i == 887u32 { v = 0x08081919u32; }
        if i == 888u32 { v = 0x08082b08u32; }
        if i == 889u32 { v = 0x08190819u32; }
        if i == 890u32 { v = 0x08191908u32; }
        if i == 891u32 { v = 0x082b0808u32; }
        if i == 892u32 { v = 0x19080819u32; }
        if i == 893u32 { v = 0x19081908u32; }
        if i == 894u32 { v = 0x19190808u32; }
        if i == 895u32 { v = 0x2b080808u32; }
        if i == 896u32 { v = 0x2b082b2bu32; }
        if i == 897u32 { v = 0x08080819u32; }
        if i == 898u32 { v = 0x08081908u32; }
        if i == 899u32 { v = 0x08190808u32; }
        if i == 900u32 { v = 0x082b2b19u32; }
        if i == 901u32 { v = 0x19080808u32; }
        if i == 902u32 { v = 0x08080808u32; }
        if i == 903u32 { v = 0x08081919u32; }
        if i == 904u32 { v = 0x08190819u32; }
        if i == 905u32 { v = 0x08191908u32; }
        if i == 906u32 { v = 0x19080819u32; }
        if i == 907u32 { v = 0x19081908u32; }
        if i == 908u32 { v = 0x19190808u32; }
        if i == 909u32 { v = 0x2b2b082bu32; }
        if i == 910u32 { v = 0x08080819u32; }
        if i == 911u32 { v = 0x08081908u32; }
        if i == 912u32 { v = 0x19080808u32; }
        if i == 913u32 { v = 0x192b1919u32; }
        if i == 914u32 { v = 0x082b082bu32; }
        if i == 915u32 { v = 0x19192b08u32; }
        if i == 916u32 { v = 0x19192b2bu32; }
        if i == 917u32 { v = 0x2b08082bu32; }
        if i == 918u32 { v = 0x2b2b082bu32; }
        if i == 919u32 { v = 0x08080819u32; }
        if i == 920u32 { v = 0x08081908u32; }
        if i == 921u32 { v = 0x08082b19u32; }
        if i == 922u32 { v = 0x08190808u32; }
        if i == 923u32 { v = 0x0819082bu32; }
        if i == 924u32 { v = 0x08191919u32; }
        if i == 925u32 { v = 0x08192b08u32; }
        if i == 926u32 { v = 0x082b1908u32; }
        if i == 927u32 { v = 0x19080808u32; }
        if i == 928u32 { v = 0x1908082bu32; }
        if i == 929u32 { v = 0x19081919u32; }
        if i == 930u32 { v = 0x19082b08u32; }
        if i == 931u32 { v = 0x19190819u32; }
        if i == 932u32 { v = 0x19191908u32; }
        if i == 933u32 { v = 0x192b0808u32; }
        if i == 934u32 { v = 0x2b080819u32; }
        if i == 935u32 { v = 0x2b081908u32; }
        if i == 936u32 { v = 0x2b190808u32; }
        if i == 937u32 { v = 0x08080808u32; }
        if i == 938u32 { v = 0x08081919u32; }
        if i == 939u32 { v = 0x08190819u32; }
        if i == 940u32 { v = 0x08191908u32; }
        if i == 941u32 { v = 0x19080819u32; }
        if i == 942u32 { v = 0x19081908u32; }
        if i == 943u32 { v = 0x19190808u32; }
        if i == 944u32 { v = 0x19192b2bu32; }
        if i == 945u32 { v = 0x08080819u32; }
        if i == 946u32 { v = 0x08081908u32; }
        if i == 947u32 { v = 0x08190808u32; }
        if i == 948u32 { v = 0x19080808u32; }
        if i == 949u32 { v = 0x2b2b192bu32; }
        if i == 950u32 { v = 0x08080808u32; }
        if i == 951u32 { v = 0x0808082bu32; }
        if i == 952u32 { v = 0x08081919u32; }
        if i == 953u32 { v = 0x08082b08u32; }
        if i == 954u32 { v = 0x08190819u32; }
        if i == 955u32 { v = 0x08191908u32; }
        if i == 956u32 { v = 0x082b0808u32; }
        if i == 957u32 { v = 0x19080819u32; }
        if i == 958u32 { v = 0x19081908u32; }
        if i == 959u32 { v = 0x19190808u32; }
        if i == 960u32 { v = 0x2b080808u32; }
        if i == 961u32 { v = 0x2b19192bu32; }
        if i == 962u32 { v = 0x08080819u32; }
        if i == 963u32 { v = 0x08081908u32; }
        if i == 964u32 { v = 0x08190808u32; }
        if i == 965u32 { v = 0x19080808u32; }
        if i == 966u32 { v = 0x2b192b08u32; }
        if i == 967u32 { v = 0x2b2b0819u32; }
        if i == 968u32 { v = 0x08080808u32; }
        if i == 969u32 { v = 0x1908192bu32; }
        if i == 970u32 { v = 0x192b1908u32; }
        if i == 971u32 { v = 0x08080819u32; }
        if i == 972u32 { v = 0x08081908u32; }
        if i == 973u32 { v = 0x08190808u32; }
        if i == 974u32 { v = 0x082b192bu32; }
        if i == 975u32 { v = 0x19080808u32; }
        if i == 976u32 { v = 0x2b2b2b19u32; }
        if i == 977u32 { v = 0x08080808u32; }
        if i == 978u32 { v = 0x19082b19u32; }
        if i == 979u32 { v = 0x1919082bu32; }
        if i == 980u32 { v = 0x2b190808u32; }
        if i == 981u32 { v = 0x08080808u32; }
        if i == 982u32 { v = 0x08081919u32; }
        if i == 983u32 { v = 0x08082b2bu32; }
        if i == 984u32 { v = 0x08191908u32; }
        if i == 985u32 { v = 0x082b082bu32; }
        if i == 986u32 { v = 0x082b2b2bu32; }
        if i == 987u32 { v = 0x19080819u32; }
        if i == 988u32 { v = 0x19081908u32; }
        if i == 989u32 { v = 0x19190808u32; }
        if i == 990u32 { v = 0x2b2b082bu32; }
        if i == 991u32 { v = 0x2b2b2b2bu32; }
        if i == 992u32 { v = 0x19080808u32; }
        if i == 993u32 { v = 0x192b1919u32; }
        if i == 994u32 { v = 0x0808082bu32; }
        if i == 995u32 { v = 0x08082b2bu32; }
        if i == 996u32 { v = 0x082b082bu32; }
        if i == 997u32 { v = 0x082b2b08u32; }
        if i == 998u32 { v = 0x082b2b2bu32; }
        if i == 999u32 { v = 0x2b08082bu32; }
        if i == 1000u32 { v = 0x2b082b08u32; }
        if i == 1001u32 { v = 0x2b082b2bu32; }
        if i == 1002u32 { v = 0x2b2b2b08u32; }
        if i == 1003u32 { v = 0x08080819u32; }
        if i == 1004u32 { v = 0x08081908u32; }
        if i == 1005u32 { v = 0x08190808u32; }
        if i == 1006u32 { v = 0x19080808u32; }
        if i == 1007u32 { v = 0x2b082b19u32; }
        if i == 1008u32 { v = 0x2b2b1908u32; }
        if i == 1009u32 { v = 0x08080808u32; }
        if i == 1010u32 { v = 0x08192b19u32; }
        if i == 1011u32 { v = 0x19190819u32; }
        if i == 1012u32 { v = 0x08082b2bu32; }
        if i == 1013u32 { v = 0x082b2b08u32; }
        if i == 1014u32 { v = 0x2b2b082bu32; }
        if i == 1015u32 { v = 0x19191908u32; }
        if i == 1016u32 { v = 0x2b08192bu32; }
        if i == 1017u32 { v = 0x08082b08u32; }
        if i == 1018u32 { v = 0x08082b2bu32; }
        if i == 1019u32 { v = 0x082b0808u32; }
        if i == 1020u32 { v = 0x082b082bu32; }
        if i == 1021u32 { v = 0x082b2b08u32; }
        if i == 1022u32 { v = 0x2b082b08u32; }
        if i == 1023u32 { v = 0x2b2b2b2bu32; }
        v
    }

pub fn s_grid_hi(i: u32) -> u32 {
        let mut v: u32 = 0;
        if i == 0u32 { v = 0x08080808u32; }
        if i == 1u32 { v = 0x08080808u32; }
        if i == 2u32 { v = 0x08080808u32; }
        if i == 3u32 { v = 0x08080808u32; }
        if i == 4u32 { v = 0x08080808u32; }
        if i == 5u32 { v = 0x08080808u32; }
        if i == 6u32 { v = 0x08080808u32; }
        if i == 7u32 { v = 0x08080808u32; }
        if i == 8u32 { v = 0x08080808u32; }
        if i == 9u32 { v = 0x08080808u32; }
        if i == 10u32 { v = 0x08080808u32; }
        if i == 11u32 { v = 0x08080808u32; }
        if i == 12u32 { v = 0x08080808u32; }
        if i == 13u32 { v = 0x08080808u32; }
        if i == 14u32 { v = 0x08080808u32; }
        if i == 15u32 { v = 0x08080808u32; }
        if i == 16u32 { v = 0x08080808u32; }
        if i == 17u32 { v = 0x08080808u32; }
        if i == 18u32 { v = 0x08080808u32; }
        if i == 19u32 { v = 0x08080808u32; }
        if i == 20u32 { v = 0x08080808u32; }
        if i == 21u32 { v = 0x08080808u32; }
        if i == 22u32 { v = 0x08080808u32; }
        if i == 23u32 { v = 0x08080808u32; }
        if i == 24u32 { v = 0x08080808u32; }
        if i == 25u32 { v = 0x08080808u32; }
        if i == 26u32 { v = 0x08080808u32; }
        if i == 27u32 { v = 0x08080808u32; }
        if i == 28u32 { v = 0x08080808u32; }
        if i == 29u32 { v = 0x08080808u32; }
        if i == 30u32 { v = 0x08080808u32; }
        if i == 31u32 { v = 0x08080808u32; }
        if i == 32u32 { v = 0x08080808u32; }
        if i == 33u32 { v = 0x08080808u32; }
        if i == 34u32 { v = 0x08080819u32; }
        if i == 35u32 { v = 0x08080819u32; }
        if i == 36u32 { v = 0x08080819u32; }
        if i == 37u32 { v = 0x08080819u32; }
        if i == 38u32 { v = 0x08080819u32; }
        if i == 39u32 { v = 0x08080819u32; }
        if i == 40u32 { v = 0x08080819u32; }
        if i == 41u32 { v = 0x08080819u32; }
        if i == 42u32 { v = 0x08080819u32; }
        if i == 43u32 { v = 0x08080819u32; }
        if i == 44u32 { v = 0x08080819u32; }
        if i == 45u32 { v = 0x08080819u32; }
        if i == 46u32 { v = 0x08080819u32; }
        if i == 47u32 { v = 0x08080819u32; }
        if i == 48u32 { v = 0x08080819u32; }
        if i == 49u32 { v = 0x08080819u32; }
        if i == 50u32 { v = 0x08080819u32; }
        if i == 51u32 { v = 0x08080819u32; }
        if i == 52u32 { v = 0x08080819u32; }
        if i == 53u32 { v = 0x08080819u32; }
        if i == 54u32 { v = 0x08080819u32; }
        if i == 55u32 { v = 0x08080819u32; }
        if i == 56u32 { v = 0x08080819u32; }
        if i == 57u32 { v = 0x08080819u32; }
        if i == 58u32 { v = 0x08080819u32; }
        if i == 59u32 { v = 0x08080819u32; }
        if i == 60u32 { v = 0x08080819u32; }
        if i == 61u32 { v = 0x08080819u32; }
        if i == 62u32 { v = 0x0808082bu32; }
        if i == 63u32 { v = 0x0808082bu32; }
        if i == 64u32 { v = 0x0808082bu32; }
        if i == 65u32 { v = 0x0808082bu32; }
        if i == 66u32 { v = 0x0808082bu32; }
        if i == 67u32 { v = 0x0808082bu32; }
        if i == 68u32 { v = 0x0808082bu32; }
        if i == 69u32 { v = 0x0808082bu32; }
        if i == 70u32 { v = 0x0808082bu32; }
        if i == 71u32 { v = 0x0808082bu32; }
        if i == 72u32 { v = 0x0808082bu32; }
        if i == 73u32 { v = 0x0808082bu32; }
        if i == 74u32 { v = 0x0808082bu32; }
        if i == 75u32 { v = 0x0808082bu32; }
        if i == 76u32 { v = 0x0808082bu32; }
        if i == 77u32 { v = 0x0808082bu32; }
        if i == 78u32 { v = 0x0808082bu32; }
        if i == 79u32 { v = 0x0808082bu32; }
        if i == 80u32 { v = 0x0808082bu32; }
        if i == 81u32 { v = 0x08081908u32; }
        if i == 82u32 { v = 0x08081908u32; }
        if i == 83u32 { v = 0x08081908u32; }
        if i == 84u32 { v = 0x08081908u32; }
        if i == 85u32 { v = 0x08081908u32; }
        if i == 86u32 { v = 0x08081908u32; }
        if i == 87u32 { v = 0x08081908u32; }
        if i == 88u32 { v = 0x08081908u32; }
        if i == 89u32 { v = 0x08081908u32; }
        if i == 90u32 { v = 0x08081908u32; }
        if i == 91u32 { v = 0x08081908u32; }
        if i == 92u32 { v = 0x08081908u32; }
        if i == 93u32 { v = 0x08081908u32; }
        if i == 94u32 { v = 0x08081908u32; }
        if i == 95u32 { v = 0x08081908u32; }
        if i == 96u32 { v = 0x08081908u32; }
        if i == 97u32 { v = 0x08081908u32; }
        if i == 98u32 { v = 0x08081908u32; }
        if i == 99u32 { v = 0x08081908u32; }
        if i == 100u32 { v = 0x08081908u32; }
        if i == 101u32 { v = 0x08081908u32; }
        if i == 102u32 { v = 0x08081908u32; }
        if i == 103u32 { v = 0x08081908u32; }
        if i == 104u32 { v = 0x08081908u32; }
        if i == 105u32 { v = 0x08081908u32; }
        if i == 106u32 { v = 0x08081908u32; }
        if i == 107u32 { v = 0x08081908u32; }
        if i == 108u32 { v = 0x08081908u32; }
        if i == 109u32 { v = 0x08081908u32; }
        if i == 110u32 { v = 0x08081908u32; }
        if i == 111u32 { v = 0x08081908u32; }
        if i == 112u32 { v = 0x08081908u32; }
        if i == 113u32 { v = 0x08081908u32; }
        if i == 114u32 { v = 0x08081919u32; }
        if i == 115u32 { v = 0x08081919u32; }
        if i == 116u32 { v = 0x08081919u32; }
        if i == 117u32 { v = 0x08081919u32; }
        if i == 118u32 { v = 0x08081919u32; }
        if i == 119u32 { v = 0x08081919u32; }
        if i == 120u32 { v = 0x08081919u32; }
        if i == 121u32 { v = 0x08081919u32; }
        if i == 122u32 { v = 0x08081919u32; }
        if i == 123u32 { v = 0x08081919u32; }
        if i == 124u32 { v = 0x08081919u32; }
        if i == 125u32 { v = 0x08081919u32; }
        if i == 126u32 { v = 0x08081919u32; }
        if i == 127u32 { v = 0x08081919u32; }
        if i == 128u32 { v = 0x08081919u32; }
        if i == 129u32 { v = 0x08081919u32; }
        if i == 130u32 { v = 0x08081919u32; }
        if i == 131u32 { v = 0x08081919u32; }
        if i == 132u32 { v = 0x08081919u32; }
        if i == 133u32 { v = 0x08081919u32; }
        if i == 134u32 { v = 0x08081919u32; }
        if i == 135u32 { v = 0x08081919u32; }
        if i == 136u32 { v = 0x08081919u32; }
        if i == 137u32 { v = 0x08081919u32; }
        if i == 138u32 { v = 0x08081919u32; }
        if i == 139u32 { v = 0x08081919u32; }
        if i == 140u32 { v = 0x08081919u32; }
        if i == 141u32 { v = 0x08081919u32; }
        if i == 142u32 { v = 0x08081919u32; }
        if i == 143u32 { v = 0x0808192bu32; }
        if i == 144u32 { v = 0x0808192bu32; }
        if i == 145u32 { v = 0x0808192bu32; }
        if i == 146u32 { v = 0x0808192bu32; }
        if i == 147u32 { v = 0x0808192bu32; }
        if i == 148u32 { v = 0x0808192bu32; }
        if i == 149u32 { v = 0x0808192bu32; }
        if i == 150u32 { v = 0x0808192bu32; }
        if i == 151u32 { v = 0x0808192bu32; }
        if i == 152u32 { v = 0x0808192bu32; }
        if i == 153u32 { v = 0x0808192bu32; }
        if i == 154u32 { v = 0x0808192bu32; }
        if i == 155u32 { v = 0x0808192bu32; }
        if i == 156u32 { v = 0x0808192bu32; }
        if i == 157u32 { v = 0x0808192bu32; }
        if i == 158u32 { v = 0x08082b08u32; }
        if i == 159u32 { v = 0x08082b08u32; }
        if i == 160u32 { v = 0x08082b08u32; }
        if i == 161u32 { v = 0x08082b08u32; }
        if i == 162u32 { v = 0x08082b08u32; }
        if i == 163u32 { v = 0x08082b08u32; }
        if i == 164u32 { v = 0x08082b08u32; }
        if i == 165u32 { v = 0x08082b08u32; }
        if i == 166u32 { v = 0x08082b08u32; }
        if i == 167u32 { v = 0x08082b08u32; }
        if i == 168u32 { v = 0x08082b08u32; }
        if i == 169u32 { v = 0x08082b08u32; }
        if i == 170u32 { v = 0x08082b08u32; }
        if i == 171u32 { v = 0x08082b08u32; }
        if i == 172u32 { v = 0x08082b08u32; }
        if i == 173u32 { v = 0x08082b08u32; }
        if i == 174u32 { v = 0x08082b08u32; }
        if i == 175u32 { v = 0x08082b08u32; }
        if i == 176u32 { v = 0x08082b08u32; }
        if i == 177u32 { v = 0x08082b08u32; }
        if i == 178u32 { v = 0x08082b08u32; }
        if i == 179u32 { v = 0x08082b08u32; }
        if i == 180u32 { v = 0x08082b08u32; }
        if i == 181u32 { v = 0x08082b08u32; }
        if i == 182u32 { v = 0x08082b08u32; }
        if i == 183u32 { v = 0x08082b19u32; }
        if i == 184u32 { v = 0x08082b19u32; }
        if i == 185u32 { v = 0x08082b19u32; }
        if i == 186u32 { v = 0x08082b19u32; }
        if i == 187u32 { v = 0x08082b19u32; }
        if i == 188u32 { v = 0x08082b19u32; }
        if i == 189u32 { v = 0x08082b19u32; }
        if i == 190u32 { v = 0x08082b19u32; }
        if i == 191u32 { v = 0x08082b19u32; }
        if i == 192u32 { v = 0x08082b19u32; }
        if i == 193u32 { v = 0x08082b19u32; }
        if i == 194u32 { v = 0x08082b19u32; }
        if i == 195u32 { v = 0x08082b19u32; }
        if i == 196u32 { v = 0x08082b19u32; }
        if i == 197u32 { v = 0x08082b19u32; }
        if i == 198u32 { v = 0x08082b2bu32; }
        if i == 199u32 { v = 0x08082b2bu32; }
        if i == 200u32 { v = 0x08082b2bu32; }
        if i == 201u32 { v = 0x08082b2bu32; }
        if i == 202u32 { v = 0x08082b2bu32; }
        if i == 203u32 { v = 0x08082b2bu32; }
        if i == 204u32 { v = 0x08082b2bu32; }
        if i == 205u32 { v = 0x08082b2bu32; }
        if i == 206u32 { v = 0x08190808u32; }
        if i == 207u32 { v = 0x08190808u32; }
        if i == 208u32 { v = 0x08190808u32; }
        if i == 209u32 { v = 0x08190808u32; }
        if i == 210u32 { v = 0x08190808u32; }
        if i == 211u32 { v = 0x08190808u32; }
        if i == 212u32 { v = 0x08190808u32; }
        if i == 213u32 { v = 0x08190808u32; }
        if i == 214u32 { v = 0x08190808u32; }
        if i == 215u32 { v = 0x08190808u32; }
        if i == 216u32 { v = 0x08190808u32; }
        if i == 217u32 { v = 0x08190808u32; }
        if i == 218u32 { v = 0x08190808u32; }
        if i == 219u32 { v = 0x08190808u32; }
        if i == 220u32 { v = 0x08190808u32; }
        if i == 221u32 { v = 0x08190808u32; }
        if i == 222u32 { v = 0x08190808u32; }
        if i == 223u32 { v = 0x08190808u32; }
        if i == 224u32 { v = 0x08190808u32; }
        if i == 225u32 { v = 0x08190808u32; }
        if i == 226u32 { v = 0x08190808u32; }
        if i == 227u32 { v = 0x08190808u32; }
        if i == 228u32 { v = 0x08190808u32; }
        if i == 229u32 { v = 0x08190808u32; }
        if i == 230u32 { v = 0x08190808u32; }
        if i == 231u32 { v = 0x08190808u32; }
        if i == 232u32 { v = 0x08190808u32; }
        if i == 233u32 { v = 0x08190808u32; }
        if i == 234u32 { v = 0x08190808u32; }
        if i == 235u32 { v = 0x08190808u32; }
        if i == 236u32 { v = 0x08190808u32; }
        if i == 237u32 { v = 0x08190819u32; }
        if i == 238u32 { v = 0x08190819u32; }
        if i == 239u32 { v = 0x08190819u32; }
        if i == 240u32 { v = 0x08190819u32; }
        if i == 241u32 { v = 0x08190819u32; }
        if i == 242u32 { v = 0x08190819u32; }
        if i == 243u32 { v = 0x08190819u32; }
        if i == 244u32 { v = 0x08190819u32; }
        if i == 245u32 { v = 0x08190819u32; }
        if i == 246u32 { v = 0x08190819u32; }
        if i == 247u32 { v = 0x08190819u32; }
        if i == 248u32 { v = 0x08190819u32; }
        if i == 249u32 { v = 0x08190819u32; }
        if i == 250u32 { v = 0x08190819u32; }
        if i == 251u32 { v = 0x08190819u32; }
        if i == 252u32 { v = 0x08190819u32; }
        if i == 253u32 { v = 0x08190819u32; }
        if i == 254u32 { v = 0x08190819u32; }
        if i == 255u32 { v = 0x08190819u32; }
        if i == 256u32 { v = 0x08190819u32; }
        if i == 257u32 { v = 0x08190819u32; }
        if i == 258u32 { v = 0x08190819u32; }
        if i == 259u32 { v = 0x08190819u32; }
        if i == 260u32 { v = 0x08190819u32; }
        if i == 261u32 { v = 0x08190819u32; }
        if i == 262u32 { v = 0x08190819u32; }
        if i == 263u32 { v = 0x08190819u32; }
        if i == 264u32 { v = 0x08190819u32; }
        if i == 265u32 { v = 0x08190819u32; }
        if i == 266u32 { v = 0x0819082bu32; }
        if i == 267u32 { v = 0x0819082bu32; }
        if i == 268u32 { v = 0x0819082bu32; }
        if i == 269u32 { v = 0x0819082bu32; }
        if i == 270u32 { v = 0x0819082bu32; }
        if i == 271u32 { v = 0x0819082bu32; }
        if i == 272u32 { v = 0x0819082bu32; }
        if i == 273u32 { v = 0x0819082bu32; }
        if i == 274u32 { v = 0x0819082bu32; }
        if i == 275u32 { v = 0x0819082bu32; }
        if i == 276u32 { v = 0x0819082bu32; }
        if i == 277u32 { v = 0x0819082bu32; }
        if i == 278u32 { v = 0x0819082bu32; }
        if i == 279u32 { v = 0x0819082bu32; }
        if i == 280u32 { v = 0x08191908u32; }
        if i == 281u32 { v = 0x08191908u32; }
        if i == 282u32 { v = 0x08191908u32; }
        if i == 283u32 { v = 0x08191908u32; }
        if i == 284u32 { v = 0x08191908u32; }
        if i == 285u32 { v = 0x08191908u32; }
        if i == 286u32 { v = 0x08191908u32; }
        if i == 287u32 { v = 0x08191908u32; }
        if i == 288u32 { v = 0x08191908u32; }
        if i == 289u32 { v = 0x08191908u32; }
        if i == 290u32 { v = 0x08191908u32; }
        if i == 291u32 { v = 0x08191908u32; }
        if i == 292u32 { v = 0x08191908u32; }
        if i == 293u32 { v = 0x08191908u32; }
        if i == 294u32 { v = 0x08191908u32; }
        if i == 295u32 { v = 0x08191908u32; }
        if i == 296u32 { v = 0x08191908u32; }
        if i == 297u32 { v = 0x08191908u32; }
        if i == 298u32 { v = 0x08191908u32; }
        if i == 299u32 { v = 0x08191908u32; }
        if i == 300u32 { v = 0x08191908u32; }
        if i == 301u32 { v = 0x08191908u32; }
        if i == 302u32 { v = 0x08191908u32; }
        if i == 303u32 { v = 0x08191908u32; }
        if i == 304u32 { v = 0x08191908u32; }
        if i == 305u32 { v = 0x08191908u32; }
        if i == 306u32 { v = 0x08191908u32; }
        if i == 307u32 { v = 0x08191908u32; }
        if i == 308u32 { v = 0x08191919u32; }
        if i == 309u32 { v = 0x08191919u32; }
        if i == 310u32 { v = 0x08191919u32; }
        if i == 311u32 { v = 0x08191919u32; }
        if i == 312u32 { v = 0x08191919u32; }
        if i == 313u32 { v = 0x08191919u32; }
        if i == 314u32 { v = 0x08191919u32; }
        if i == 315u32 { v = 0x08191919u32; }
        if i == 316u32 { v = 0x08191919u32; }
        if i == 317u32 { v = 0x08191919u32; }
        if i == 318u32 { v = 0x08191919u32; }
        if i == 319u32 { v = 0x08191919u32; }
        if i == 320u32 { v = 0x08191919u32; }
        if i == 321u32 { v = 0x08191919u32; }
        if i == 322u32 { v = 0x08191919u32; }
        if i == 323u32 { v = 0x08191919u32; }
        if i == 324u32 { v = 0x08191919u32; }
        if i == 325u32 { v = 0x08191919u32; }
        if i == 326u32 { v = 0x08191919u32; }
        if i == 327u32 { v = 0x08191919u32; }
        if i == 328u32 { v = 0x0819192bu32; }
        if i == 329u32 { v = 0x0819192bu32; }
        if i == 330u32 { v = 0x0819192bu32; }
        if i == 331u32 { v = 0x0819192bu32; }
        if i == 332u32 { v = 0x0819192bu32; }
        if i == 333u32 { v = 0x0819192bu32; }
        if i == 334u32 { v = 0x0819192bu32; }
        if i == 335u32 { v = 0x0819192bu32; }
        if i == 336u32 { v = 0x0819192bu32; }
        if i == 337u32 { v = 0x0819192bu32; }
        if i == 338u32 { v = 0x0819192bu32; }
        if i == 339u32 { v = 0x08192b08u32; }
        if i == 340u32 { v = 0x08192b08u32; }
        if i == 341u32 { v = 0x08192b08u32; }
        if i == 342u32 { v = 0x08192b08u32; }
        if i == 343u32 { v = 0x08192b08u32; }
        if i == 344u32 { v = 0x08192b08u32; }
        if i == 345u32 { v = 0x08192b08u32; }
        if i == 346u32 { v = 0x08192b08u32; }
        if i == 347u32 { v = 0x08192b08u32; }
        if i == 348u32 { v = 0x08192b08u32; }
        if i == 349u32 { v = 0x08192b08u32; }
        if i == 350u32 { v = 0x08192b08u32; }
        if i == 351u32 { v = 0x08192b08u32; }
        if i == 352u32 { v = 0x08192b08u32; }
        if i == 353u32 { v = 0x08192b08u32; }
        if i == 354u32 { v = 0x08192b08u32; }
        if i == 355u32 { v = 0x08192b08u32; }
        if i == 356u32 { v = 0x08192b19u32; }
        if i == 357u32 { v = 0x08192b19u32; }
        if i == 358u32 { v = 0x08192b19u32; }
        if i == 359u32 { v = 0x08192b19u32; }
        if i == 360u32 { v = 0x08192b19u32; }
        if i == 361u32 { v = 0x08192b19u32; }
        if i == 362u32 { v = 0x08192b19u32; }
        if i == 363u32 { v = 0x08192b19u32; }
        if i == 364u32 { v = 0x08192b19u32; }
        if i == 365u32 { v = 0x08192b19u32; }
        if i == 366u32 { v = 0x08192b19u32; }
        if i == 367u32 { v = 0x08192b19u32; }
        if i == 368u32 { v = 0x08192b2bu32; }
        if i == 369u32 { v = 0x08192b2bu32; }
        if i == 370u32 { v = 0x08192b2bu32; }
        if i == 371u32 { v = 0x08192b2bu32; }
        if i == 372u32 { v = 0x082b0808u32; }
        if i == 373u32 { v = 0x082b0808u32; }
        if i == 374u32 { v = 0x082b0808u32; }
        if i == 375u32 { v = 0x082b0808u32; }
        if i == 376u32 { v = 0x082b0808u32; }
        if i == 377u32 { v = 0x082b0808u32; }
        if i == 378u32 { v = 0x082b0808u32; }
        if i == 379u32 { v = 0x082b0808u32; }
        if i == 380u32 { v = 0x082b0808u32; }
        if i == 381u32 { v = 0x082b0808u32; }
        if i == 382u32 { v = 0x082b0808u32; }
        if i == 383u32 { v = 0x082b0808u32; }
        if i == 384u32 { v = 0x082b0808u32; }
        if i == 385u32 { v = 0x082b0808u32; }
        if i == 386u32 { v = 0x082b0808u32; }
        if i == 387u32 { v = 0x082b0808u32; }
        if i == 388u32 { v = 0x082b0808u32; }
        if i == 389u32 { v = 0x082b0808u32; }
        if i == 390u32 { v = 0x082b0808u32; }
        if i == 391u32 { v = 0x082b0808u32; }
        if i == 392u32 { v = 0x082b0808u32; }
        if i == 393u32 { v = 0x082b0819u32; }
        if i == 394u32 { v = 0x082b0819u32; }
        if i == 395u32 { v = 0x082b0819u32; }
        if i == 396u32 { v = 0x082b0819u32; }
        if i == 397u32 { v = 0x082b0819u32; }
        if i == 398u32 { v = 0x082b0819u32; }
        if i == 399u32 { v = 0x082b0819u32; }
        if i == 400u32 { v = 0x082b0819u32; }
        if i == 401u32 { v = 0x082b0819u32; }
        if i == 402u32 { v = 0x082b0819u32; }
        if i == 403u32 { v = 0x082b0819u32; }
        if i == 404u32 { v = 0x082b0819u32; }
        if i == 405u32 { v = 0x082b0819u32; }
        if i == 406u32 { v = 0x082b0819u32; }
        if i == 407u32 { v = 0x082b0819u32; }
        if i == 408u32 { v = 0x082b082bu32; }
        if i == 409u32 { v = 0x082b082bu32; }
        if i == 410u32 { v = 0x082b082bu32; }
        if i == 411u32 { v = 0x082b082bu32; }
        if i == 412u32 { v = 0x082b082bu32; }
        if i == 413u32 { v = 0x082b082bu32; }
        if i == 414u32 { v = 0x082b082bu32; }
        if i == 415u32 { v = 0x082b082bu32; }
        if i == 416u32 { v = 0x082b082bu32; }
        if i == 417u32 { v = 0x082b082bu32; }
        if i == 418u32 { v = 0x082b1908u32; }
        if i == 419u32 { v = 0x082b1908u32; }
        if i == 420u32 { v = 0x082b1908u32; }
        if i == 421u32 { v = 0x082b1908u32; }
        if i == 422u32 { v = 0x082b1908u32; }
        if i == 423u32 { v = 0x082b1908u32; }
        if i == 424u32 { v = 0x082b1908u32; }
        if i == 425u32 { v = 0x082b1908u32; }
        if i == 426u32 { v = 0x082b1908u32; }
        if i == 427u32 { v = 0x082b1908u32; }
        if i == 428u32 { v = 0x082b1908u32; }
        if i == 429u32 { v = 0x082b1908u32; }
        if i == 430u32 { v = 0x082b1908u32; }
        if i == 431u32 { v = 0x082b1908u32; }
        if i == 432u32 { v = 0x082b1908u32; }
        if i == 433u32 { v = 0x082b1908u32; }
        if i == 434u32 { v = 0x082b1908u32; }
        if i == 435u32 { v = 0x082b1908u32; }
        if i == 436u32 { v = 0x082b1908u32; }
        if i == 437u32 { v = 0x082b1919u32; }
        if i == 438u32 { v = 0x082b1919u32; }
        if i == 439u32 { v = 0x082b1919u32; }
        if i == 440u32 { v = 0x082b1919u32; }
        if i == 441u32 { v = 0x082b1919u32; }
        if i == 442u32 { v = 0x082b1919u32; }
        if i == 443u32 { v = 0x082b1919u32; }
        if i == 444u32 { v = 0x082b1919u32; }
        if i == 445u32 { v = 0x082b1919u32; }
        if i == 446u32 { v = 0x082b1919u32; }
        if i == 447u32 { v = 0x082b1919u32; }
        if i == 448u32 { v = 0x082b192bu32; }
        if i == 449u32 { v = 0x082b192bu32; }
        if i == 450u32 { v = 0x082b192bu32; }
        if i == 451u32 { v = 0x082b192bu32; }
        if i == 452u32 { v = 0x082b192bu32; }
        if i == 453u32 { v = 0x082b2b08u32; }
        if i == 454u32 { v = 0x082b2b08u32; }
        if i == 455u32 { v = 0x082b2b08u32; }
        if i == 456u32 { v = 0x082b2b08u32; }
        if i == 457u32 { v = 0x082b2b08u32; }
        if i == 458u32 { v = 0x082b2b08u32; }
        if i == 459u32 { v = 0x082b2b08u32; }
        if i == 460u32 { v = 0x082b2b08u32; }
        if i == 461u32 { v = 0x082b2b08u32; }
        if i == 462u32 { v = 0x082b2b19u32; }
        if i == 463u32 { v = 0x082b2b19u32; }
        if i == 464u32 { v = 0x082b2b19u32; }
        if i == 465u32 { v = 0x082b2b19u32; }
        if i == 466u32 { v = 0x082b2b2bu32; }
        if i == 467u32 { v = 0x082b2b2bu32; }
        if i == 468u32 { v = 0x082b2b2bu32; }
        if i == 469u32 { v = 0x082b2b2bu32; }
        if i == 470u32 { v = 0x082b2b2bu32; }
        if i == 471u32 { v = 0x19080808u32; }
        if i == 472u32 { v = 0x19080808u32; }
        if i == 473u32 { v = 0x19080808u32; }
        if i == 474u32 { v = 0x19080808u32; }
        if i == 475u32 { v = 0x19080808u32; }
        if i == 476u32 { v = 0x19080808u32; }
        if i == 477u32 { v = 0x19080808u32; }
        if i == 478u32 { v = 0x19080808u32; }
        if i == 479u32 { v = 0x19080808u32; }
        if i == 480u32 { v = 0x19080808u32; }
        if i == 481u32 { v = 0x19080808u32; }
        if i == 482u32 { v = 0x19080808u32; }
        if i == 483u32 { v = 0x19080808u32; }
        if i == 484u32 { v = 0x19080808u32; }
        if i == 485u32 { v = 0x19080808u32; }
        if i == 486u32 { v = 0x19080808u32; }
        if i == 487u32 { v = 0x19080808u32; }
        if i == 488u32 { v = 0x19080808u32; }
        if i == 489u32 { v = 0x19080808u32; }
        if i == 490u32 { v = 0x19080808u32; }
        if i == 491u32 { v = 0x19080808u32; }
        if i == 492u32 { v = 0x19080808u32; }
        if i == 493u32 { v = 0x19080808u32; }
        if i == 494u32 { v = 0x19080808u32; }
        if i == 495u32 { v = 0x19080808u32; }
        if i == 496u32 { v = 0x19080808u32; }
        if i == 497u32 { v = 0x19080808u32; }
        if i == 498u32 { v = 0x19080808u32; }
        if i == 499u32 { v = 0x19080808u32; }
        if i == 500u32 { v = 0x19080808u32; }
        if i == 501u32 { v = 0x19080808u32; }
        if i == 502u32 { v = 0x19080819u32; }
        if i == 503u32 { v = 0x19080819u32; }
        if i == 504u32 { v = 0x19080819u32; }
        if i == 505u32 { v = 0x19080819u32; }
        if i == 506u32 { v = 0x19080819u32; }
        if i == 507u32 { v = 0x19080819u32; }
        if i == 508u32 { v = 0x19080819u32; }
        if i == 509u32 { v = 0x19080819u32; }
        if i == 510u32 { v = 0x19080819u32; }
        if i == 511u32 { v = 0x19080819u32; }
        if i == 512u32 { v = 0x19080819u32; }
        if i == 513u32 { v = 0x19080819u32; }
        if i == 514u32 { v = 0x19080819u32; }
        if i == 515u32 { v = 0x19080819u32; }
        if i == 516u32 { v = 0x19080819u32; }
        if i == 517u32 { v = 0x19080819u32; }
        if i == 518u32 { v = 0x19080819u32; }
        if i == 519u32 { v = 0x19080819u32; }
        if i == 520u32 { v = 0x19080819u32; }
        if i == 521u32 { v = 0x19080819u32; }
        if i == 522u32 { v = 0x19080819u32; }
        if i == 523u32 { v = 0x19080819u32; }
        if i == 524u32 { v = 0x19080819u32; }
        if i == 525u32 { v = 0x19080819u32; }
        if i == 526u32 { v = 0x19080819u32; }
        if i == 527u32 { v = 0x19080819u32; }
        if i == 528u32 { v = 0x19080819u32; }
        if i == 529u32 { v = 0x19080819u32; }
        if i == 530u32 { v = 0x1908082bu32; }
        if i == 531u32 { v = 0x1908082bu32; }
        if i == 532u32 { v = 0x1908082bu32; }
        if i == 533u32 { v = 0x1908082bu32; }
        if i == 534u32 { v = 0x1908082bu32; }
        if i == 535u32 { v = 0x1908082bu32; }
        if i == 536u32 { v = 0x1908082bu32; }
        if i == 537u32 { v = 0x1908082bu32; }
        if i == 538u32 { v = 0x1908082bu32; }
        if i == 539u32 { v = 0x1908082bu32; }
        if i == 540u32 { v = 0x1908082bu32; }
        if i == 541u32 { v = 0x1908082bu32; }
        if i == 542u32 { v = 0x1908082bu32; }
        if i == 543u32 { v = 0x1908082bu32; }
        if i == 544u32 { v = 0x1908082bu32; }
        if i == 545u32 { v = 0x19081908u32; }
        if i == 546u32 { v = 0x19081908u32; }
        if i == 547u32 { v = 0x19081908u32; }
        if i == 548u32 { v = 0x19081908u32; }
        if i == 549u32 { v = 0x19081908u32; }
        if i == 550u32 { v = 0x19081908u32; }
        if i == 551u32 { v = 0x19081908u32; }
        if i == 552u32 { v = 0x19081908u32; }
        if i == 553u32 { v = 0x19081908u32; }
        if i == 554u32 { v = 0x19081908u32; }
        if i == 555u32 { v = 0x19081908u32; }
        if i == 556u32 { v = 0x19081908u32; }
        if i == 557u32 { v = 0x19081908u32; }
        if i == 558u32 { v = 0x19081908u32; }
        if i == 559u32 { v = 0x19081908u32; }
        if i == 560u32 { v = 0x19081908u32; }
        if i == 561u32 { v = 0x19081908u32; }
        if i == 562u32 { v = 0x19081908u32; }
        if i == 563u32 { v = 0x19081908u32; }
        if i == 564u32 { v = 0x19081908u32; }
        if i == 565u32 { v = 0x19081908u32; }
        if i == 566u32 { v = 0x19081908u32; }
        if i == 567u32 { v = 0x19081908u32; }
        if i == 568u32 { v = 0x19081908u32; }
        if i == 569u32 { v = 0x19081908u32; }
        if i == 570u32 { v = 0x19081908u32; }
        if i == 571u32 { v = 0x19081908u32; }
        if i == 572u32 { v = 0x19081908u32; }
        if i == 573u32 { v = 0x19081908u32; }
        if i == 574u32 { v = 0x19081908u32; }
        if i == 575u32 { v = 0x19081919u32; }
        if i == 576u32 { v = 0x19081919u32; }
        if i == 577u32 { v = 0x19081919u32; }
        if i == 578u32 { v = 0x19081919u32; }
        if i == 579u32 { v = 0x19081919u32; }
        if i == 580u32 { v = 0x19081919u32; }
        if i == 581u32 { v = 0x19081919u32; }
        if i == 582u32 { v = 0x19081919u32; }
        if i == 583u32 { v = 0x19081919u32; }
        if i == 584u32 { v = 0x19081919u32; }
        if i == 585u32 { v = 0x19081919u32; }
        if i == 586u32 { v = 0x19081919u32; }
        if i == 587u32 { v = 0x19081919u32; }
        if i == 588u32 { v = 0x19081919u32; }
        if i == 589u32 { v = 0x19081919u32; }
        if i == 590u32 { v = 0x19081919u32; }
        if i == 591u32 { v = 0x19081919u32; }
        if i == 592u32 { v = 0x19081919u32; }
        if i == 593u32 { v = 0x19081919u32; }
        if i == 594u32 { v = 0x19081919u32; }
        if i == 595u32 { v = 0x19081919u32; }
        if i == 596u32 { v = 0x1908192bu32; }
        if i == 597u32 { v = 0x1908192bu32; }
        if i == 598u32 { v = 0x1908192bu32; }
        if i == 599u32 { v = 0x1908192bu32; }
        if i == 600u32 { v = 0x1908192bu32; }
        if i == 601u32 { v = 0x1908192bu32; }
        if i == 602u32 { v = 0x1908192bu32; }
        if i == 603u32 { v = 0x1908192bu32; }
        if i == 604u32 { v = 0x1908192bu32; }
        if i == 605u32 { v = 0x1908192bu32; }
        if i == 606u32 { v = 0x1908192bu32; }
        if i == 607u32 { v = 0x1908192bu32; }
        if i == 608u32 { v = 0x19082b08u32; }
        if i == 609u32 { v = 0x19082b08u32; }
        if i == 610u32 { v = 0x19082b08u32; }
        if i == 611u32 { v = 0x19082b08u32; }
        if i == 612u32 { v = 0x19082b08u32; }
        if i == 613u32 { v = 0x19082b08u32; }
        if i == 614u32 { v = 0x19082b08u32; }
        if i == 615u32 { v = 0x19082b08u32; }
        if i == 616u32 { v = 0x19082b08u32; }
        if i == 617u32 { v = 0x19082b08u32; }
        if i == 618u32 { v = 0x19082b08u32; }
        if i == 619u32 { v = 0x19082b08u32; }
        if i == 620u32 { v = 0x19082b08u32; }
        if i == 621u32 { v = 0x19082b08u32; }
        if i == 622u32 { v = 0x19082b08u32; }
        if i == 623u32 { v = 0x19082b08u32; }
        if i == 624u32 { v = 0x19082b08u32; }
        if i == 625u32 { v = 0x19082b08u32; }
        if i == 626u32 { v = 0x19082b19u32; }
        if i == 627u32 { v = 0x19082b19u32; }
        if i == 628u32 { v = 0x19082b19u32; }
        if i == 629u32 { v = 0x19082b19u32; }
        if i == 630u32 { v = 0x19082b19u32; }
        if i == 631u32 { v = 0x19082b19u32; }
        if i == 632u32 { v = 0x19082b19u32; }
        if i == 633u32 { v = 0x19082b19u32; }
        if i == 634u32 { v = 0x19082b19u32; }
        if i == 635u32 { v = 0x19082b19u32; }
        if i == 636u32 { v = 0x19082b19u32; }
        if i == 637u32 { v = 0x19082b19u32; }
        if i == 638u32 { v = 0x19082b2bu32; }
        if i == 639u32 { v = 0x19082b2bu32; }
        if i == 640u32 { v = 0x19082b2bu32; }
        if i == 641u32 { v = 0x19082b2bu32; }
        if i == 642u32 { v = 0x19190808u32; }
        if i == 643u32 { v = 0x19190808u32; }
        if i == 644u32 { v = 0x19190808u32; }
        if i == 645u32 { v = 0x19190808u32; }
        if i == 646u32 { v = 0x19190808u32; }
        if i == 647u32 { v = 0x19190808u32; }
        if i == 648u32 { v = 0x19190808u32; }
        if i == 649u32 { v = 0x19190808u32; }
        if i == 650u32 { v = 0x19190808u32; }
        if i == 651u32 { v = 0x19190808u32; }
        if i == 652u32 { v = 0x19190808u32; }
        if i == 653u32 { v = 0x19190808u32; }
        if i == 654u32 { v = 0x19190808u32; }
        if i == 655u32 { v = 0x19190808u32; }
        if i == 656u32 { v = 0x19190808u32; }
        if i == 657u32 { v = 0x19190808u32; }
        if i == 658u32 { v = 0x19190808u32; }
        if i == 659u32 { v = 0x19190808u32; }
        if i == 660u32 { v = 0x19190808u32; }
        if i == 661u32 { v = 0x19190808u32; }
        if i == 662u32 { v = 0x19190808u32; }
        if i == 663u32 { v = 0x19190808u32; }
        if i == 664u32 { v = 0x19190808u32; }
        if i == 665u32 { v = 0x19190808u32; }
        if i == 666u32 { v = 0x19190808u32; }
        if i == 667u32 { v = 0x19190808u32; }
        if i == 668u32 { v = 0x19190808u32; }
        if i == 669u32 { v = 0x19190808u32; }
        if i == 670u32 { v = 0x19190819u32; }
        if i == 671u32 { v = 0x19190819u32; }
        if i == 672u32 { v = 0x19190819u32; }
        if i == 673u32 { v = 0x19190819u32; }
        if i == 674u32 { v = 0x19190819u32; }
        if i == 675u32 { v = 0x19190819u32; }
        if i == 676u32 { v = 0x19190819u32; }
        if i == 677u32 { v = 0x19190819u32; }
        if i == 678u32 { v = 0x19190819u32; }
        if i == 679u32 { v = 0x19190819u32; }
        if i == 680u32 { v = 0x19190819u32; }
        if i == 681u32 { v = 0x19190819u32; }
        if i == 682u32 { v = 0x19190819u32; }
        if i == 683u32 { v = 0x19190819u32; }
        if i == 684u32 { v = 0x19190819u32; }
        if i == 685u32 { v = 0x19190819u32; }
        if i == 686u32 { v = 0x19190819u32; }
        if i == 687u32 { v = 0x19190819u32; }
        if i == 688u32 { v = 0x19190819u32; }
        if i == 689u32 { v = 0x19190819u32; }
        if i == 690u32 { v = 0x1919082bu32; }
        if i == 691u32 { v = 0x1919082bu32; }
        if i == 692u32 { v = 0x1919082bu32; }
        if i == 693u32 { v = 0x1919082bu32; }
        if i == 694u32 { v = 0x1919082bu32; }
        if i == 695u32 { v = 0x1919082bu32; }
        if i == 696u32 { v = 0x1919082bu32; }
        if i == 697u32 { v = 0x1919082bu32; }
        if i == 698u32 { v = 0x1919082bu32; }
        if i == 699u32 { v = 0x1919082bu32; }
        if i == 700u32 { v = 0x1919082bu32; }
        if i == 701u32 { v = 0x19191908u32; }
        if i == 702u32 { v = 0x19191908u32; }
        if i == 703u32 { v = 0x19191908u32; }
        if i == 704u32 { v = 0x19191908u32; }
        if i == 705u32 { v = 0x19191908u32; }
        if i == 706u32 { v = 0x19191908u32; }
        if i == 707u32 { v = 0x19191908u32; }
        if i == 708u32 { v = 0x19191908u32; }
        if i == 709u32 { v = 0x19191908u32; }
        if i == 710u32 { v = 0x19191908u32; }
        if i == 711u32 { v = 0x19191908u32; }
        if i == 712u32 { v = 0x19191908u32; }
        if i == 713u32 { v = 0x19191908u32; }
        if i == 714u32 { v = 0x19191908u32; }
        if i == 715u32 { v = 0x19191908u32; }
        if i == 716u32 { v = 0x19191908u32; }
        if i == 717u32 { v = 0x19191908u32; }
        if i == 718u32 { v = 0x19191908u32; }
        if i == 719u32 { v = 0x19191908u32; }
        if i == 720u32 { v = 0x19191908u32; }
        if i == 721u32 { v = 0x19191919u32; }
        if i == 722u32 { v = 0x19191919u32; }
        if i == 723u32 { v = 0x19191919u32; }
        if i == 724u32 { v = 0x19191919u32; }
        if i == 725u32 { v = 0x19191919u32; }
        if i == 726u32 { v = 0x19191919u32; }
        if i == 727u32 { v = 0x19191919u32; }
        if i == 728u32 { v = 0x19191919u32; }
        if i == 729u32 { v = 0x19191919u32; }
        if i == 730u32 { v = 0x19191919u32; }
        if i == 731u32 { v = 0x19191919u32; }
        if i == 732u32 { v = 0x1919192bu32; }
        if i == 733u32 { v = 0x1919192bu32; }
        if i == 734u32 { v = 0x1919192bu32; }
        if i == 735u32 { v = 0x1919192bu32; }
        if i == 736u32 { v = 0x1919192bu32; }
        if i == 737u32 { v = 0x19192b08u32; }
        if i == 738u32 { v = 0x19192b08u32; }
        if i == 739u32 { v = 0x19192b08u32; }
        if i == 740u32 { v = 0x19192b08u32; }
        if i == 741u32 { v = 0x19192b08u32; }
        if i == 742u32 { v = 0x19192b08u32; }
        if i == 743u32 { v = 0x19192b08u32; }
        if i == 744u32 { v = 0x19192b08u32; }
        if i == 745u32 { v = 0x19192b08u32; }
        if i == 746u32 { v = 0x19192b08u32; }
        if i == 747u32 { v = 0x19192b08u32; }
        if i == 748u32 { v = 0x19192b08u32; }
        if i == 749u32 { v = 0x19192b19u32; }
        if i == 750u32 { v = 0x19192b19u32; }
        if i == 751u32 { v = 0x19192b19u32; }
        if i == 752u32 { v = 0x19192b19u32; }
        if i == 753u32 { v = 0x19192b2bu32; }
        if i == 754u32 { v = 0x19192b2bu32; }
        if i == 755u32 { v = 0x19192b2bu32; }
        if i == 756u32 { v = 0x19192b2bu32; }
        if i == 757u32 { v = 0x192b0808u32; }
        if i == 758u32 { v = 0x192b0808u32; }
        if i == 759u32 { v = 0x192b0808u32; }
        if i == 760u32 { v = 0x192b0808u32; }
        if i == 761u32 { v = 0x192b0808u32; }
        if i == 762u32 { v = 0x192b0808u32; }
        if i == 763u32 { v = 0x192b0808u32; }
        if i == 764u32 { v = 0x192b0808u32; }
        if i == 765u32 { v = 0x192b0808u32; }
        if i == 766u32 { v = 0x192b0808u32; }
        if i == 767u32 { v = 0x192b0808u32; }
        if i == 768u32 { v = 0x192b0808u32; }
        if i == 769u32 { v = 0x192b0808u32; }
        if i == 770u32 { v = 0x192b0808u32; }
        if i == 771u32 { v = 0x192b0808u32; }
        if i == 772u32 { v = 0x192b0808u32; }
        if i == 773u32 { v = 0x192b0808u32; }
        if i == 774u32 { v = 0x192b0819u32; }
        if i == 775u32 { v = 0x192b0819u32; }
        if i == 776u32 { v = 0x192b0819u32; }
        if i == 777u32 { v = 0x192b0819u32; }
        if i == 778u32 { v = 0x192b0819u32; }
        if i == 779u32 { v = 0x192b0819u32; }
        if i == 780u32 { v = 0x192b0819u32; }
        if i == 781u32 { v = 0x192b0819u32; }
        if i == 782u32 { v = 0x192b0819u32; }
        if i == 783u32 { v = 0x192b0819u32; }
        if i == 784u32 { v = 0x192b0819u32; }
        if i == 785u32 { v = 0x192b0819u32; }
        if i == 786u32 { v = 0x192b082bu32; }
        if i == 787u32 { v = 0x192b082bu32; }
        if i == 788u32 { v = 0x192b082bu32; }
        if i == 789u32 { v = 0x192b082bu32; }
        if i == 790u32 { v = 0x192b082bu32; }
        if i == 791u32 { v = 0x192b1908u32; }
        if i == 792u32 { v = 0x192b1908u32; }
        if i == 793u32 { v = 0x192b1908u32; }
        if i == 794u32 { v = 0x192b1908u32; }
        if i == 795u32 { v = 0x192b1908u32; }
        if i == 796u32 { v = 0x192b1908u32; }
        if i == 797u32 { v = 0x192b1908u32; }
        if i == 798u32 { v = 0x192b1908u32; }
        if i == 799u32 { v = 0x192b1908u32; }
        if i == 800u32 { v = 0x192b1908u32; }
        if i == 801u32 { v = 0x192b1919u32; }
        if i == 802u32 { v = 0x192b1919u32; }
        if i == 803u32 { v = 0x192b1919u32; }
        if i == 804u32 { v = 0x192b1919u32; }
        if i == 805u32 { v = 0x192b1919u32; }
        if i == 806u32 { v = 0x192b1919u32; }
        if i == 807u32 { v = 0x192b1919u32; }
        if i == 808u32 { v = 0x192b192bu32; }
        if i == 809u32 { v = 0x192b192bu32; }
        if i == 810u32 { v = 0x192b2b08u32; }
        if i == 811u32 { v = 0x192b2b08u32; }
        if i == 812u32 { v = 0x192b2b08u32; }
        if i == 813u32 { v = 0x192b2b08u32; }
        if i == 814u32 { v = 0x192b2b08u32; }
        if i == 815u32 { v = 0x192b2b19u32; }
        if i == 816u32 { v = 0x192b2b19u32; }
        if i == 817u32 { v = 0x192b2b2bu32; }
        if i == 818u32 { v = 0x192b2b2bu32; }
        if i == 819u32 { v = 0x2b080808u32; }
        if i == 820u32 { v = 0x2b080808u32; }
        if i == 821u32 { v = 0x2b080808u32; }
        if i == 822u32 { v = 0x2b080808u32; }
        if i == 823u32 { v = 0x2b080808u32; }
        if i == 824u32 { v = 0x2b080808u32; }
        if i == 825u32 { v = 0x2b080808u32; }
        if i == 826u32 { v = 0x2b080808u32; }
        if i == 827u32 { v = 0x2b080808u32; }
        if i == 828u32 { v = 0x2b080808u32; }
        if i == 829u32 { v = 0x2b080808u32; }
        if i == 830u32 { v = 0x2b080808u32; }
        if i == 831u32 { v = 0x2b080808u32; }
        if i == 832u32 { v = 0x2b080808u32; }
        if i == 833u32 { v = 0x2b080808u32; }
        if i == 834u32 { v = 0x2b080808u32; }
        if i == 835u32 { v = 0x2b080808u32; }
        if i == 836u32 { v = 0x2b080808u32; }
        if i == 837u32 { v = 0x2b080808u32; }
        if i == 838u32 { v = 0x2b080808u32; }
        if i == 839u32 { v = 0x2b080819u32; }
        if i == 840u32 { v = 0x2b080819u32; }
        if i == 841u32 { v = 0x2b080819u32; }
        if i == 842u32 { v = 0x2b080819u32; }
        if i == 843u32 { v = 0x2b080819u32; }
        if i == 844u32 { v = 0x2b080819u32; }
        if i == 845u32 { v = 0x2b080819u32; }
        if i == 846u32 { v = 0x2b080819u32; }
        if i == 847u32 { v = 0x2b080819u32; }
        if i == 848u32 { v = 0x2b080819u32; }
        if i == 849u32 { v = 0x2b080819u32; }
        if i == 850u32 { v = 0x2b080819u32; }
        if i == 851u32 { v = 0x2b080819u32; }
        if i == 852u32 { v = 0x2b080819u32; }
        if i == 853u32 { v = 0x2b080819u32; }
        if i == 854u32 { v = 0x2b080819u32; }
        if i == 855u32 { v = 0x2b080819u32; }
        if i == 856u32 { v = 0x2b080819u32; }
        if i == 857u32 { v = 0x2b080819u32; }
        if i == 858u32 { v = 0x2b08082bu32; }
        if i == 859u32 { v = 0x2b08082bu32; }
        if i == 860u32 { v = 0x2b08082bu32; }
        if i == 861u32 { v = 0x2b08082bu32; }
        if i == 862u32 { v = 0x2b08082bu32; }
        if i == 863u32 { v = 0x2b08082bu32; }
        if i == 864u32 { v = 0x2b08082bu32; }
        if i == 865u32 { v = 0x2b08082bu32; }
        if i == 866u32 { v = 0x2b081908u32; }
        if i == 867u32 { v = 0x2b081908u32; }
        if i == 868u32 { v = 0x2b081908u32; }
        if i == 869u32 { v = 0x2b081908u32; }
        if i == 870u32 { v = 0x2b081908u32; }
        if i == 871u32 { v = 0x2b081908u32; }
        if i == 872u32 { v = 0x2b081908u32; }
        if i == 873u32 { v = 0x2b081908u32; }
        if i == 874u32 { v = 0x2b081908u32; }
        if i == 875u32 { v = 0x2b081908u32; }
        if i == 876u32 { v = 0x2b081908u32; }
        if i == 877u32 { v = 0x2b081908u32; }
        if i == 878u32 { v = 0x2b081908u32; }
        if i == 879u32 { v = 0x2b081908u32; }
        if i == 880u32 { v = 0x2b081908u32; }
        if i == 881u32 { v = 0x2b081908u32; }
        if i == 882u32 { v = 0x2b081908u32; }
        if i == 883u32 { v = 0x2b081908u32; }
        if i == 884u32 { v = 0x2b081908u32; }
        if i == 885u32 { v = 0x2b081919u32; }
        if i == 886u32 { v = 0x2b081919u32; }
        if i == 887u32 { v = 0x2b081919u32; }
        if i == 888u32 { v = 0x2b081919u32; }
        if i == 889u32 { v = 0x2b081919u32; }
        if i == 890u32 { v = 0x2b081919u32; }
        if i == 891u32 { v = 0x2b081919u32; }
        if i == 892u32 { v = 0x2b081919u32; }
        if i == 893u32 { v = 0x2b081919u32; }
        if i == 894u32 { v = 0x2b081919u32; }
        if i == 895u32 { v = 0x2b081919u32; }
        if i == 896u32 { v = 0x2b081919u32; }
        if i == 897u32 { v = 0x2b08192bu32; }
        if i == 898u32 { v = 0x2b08192bu32; }
        if i == 899u32 { v = 0x2b08192bu32; }
        if i == 900u32 { v = 0x2b08192bu32; }
        if i == 901u32 { v = 0x2b08192bu32; }
        if i == 902u32 { v = 0x2b082b08u32; }
        if i == 903u32 { v = 0x2b082b08u32; }
        if i == 904u32 { v = 0x2b082b08u32; }
        if i == 905u32 { v = 0x2b082b08u32; }
        if i == 906u32 { v = 0x2b082b08u32; }
        if i == 907u32 { v = 0x2b082b08u32; }
        if i == 908u32 { v = 0x2b082b08u32; }
        if i == 909u32 { v = 0x2b082b08u32; }
        if i == 910u32 { v = 0x2b082b19u32; }
        if i == 911u32 { v = 0x2b082b19u32; }
        if i == 912u32 { v = 0x2b082b19u32; }
        if i == 913u32 { v = 0x2b082b19u32; }
        if i == 914u32 { v = 0x2b082b2bu32; }
        if i == 915u32 { v = 0x2b082b2bu32; }
        if i == 916u32 { v = 0x2b082b2bu32; }
        if i == 917u32 { v = 0x2b082b2bu32; }
        if i == 918u32 { v = 0x2b082b2bu32; }
        if i == 919u32 { v = 0x2b190808u32; }
        if i == 920u32 { v = 0x2b190808u32; }
        if i == 921u32 { v = 0x2b190808u32; }
        if i == 922u32 { v = 0x2b190808u32; }
        if i == 923u32 { v = 0x2b190808u32; }
        if i == 924u32 { v = 0x2b190808u32; }
        if i == 925u32 { v = 0x2b190808u32; }
        if i == 926u32 { v = 0x2b190808u32; }
        if i == 927u32 { v = 0x2b190808u32; }
        if i == 928u32 { v = 0x2b190808u32; }
        if i == 929u32 { v = 0x2b190808u32; }
        if i == 930u32 { v = 0x2b190808u32; }
        if i == 931u32 { v = 0x2b190808u32; }
        if i == 932u32 { v = 0x2b190808u32; }
        if i == 933u32 { v = 0x2b190808u32; }
        if i == 934u32 { v = 0x2b190808u32; }
        if i == 935u32 { v = 0x2b190808u32; }
        if i == 936u32 { v = 0x2b190808u32; }
        if i == 937u32 { v = 0x2b190819u32; }
        if i == 938u32 { v = 0x2b190819u32; }
        if i == 939u32 { v = 0x2b190819u32; }
        if i == 940u32 { v = 0x2b190819u32; }
        if i == 941u32 { v = 0x2b190819u32; }
        if i == 942u32 { v = 0x2b190819u32; }
        if i == 943u32 { v = 0x2b190819u32; }
        if i == 944u32 { v = 0x2b190819u32; }
        if i == 945u32 { v = 0x2b19082bu32; }
        if i == 946u32 { v = 0x2b19082bu32; }
        if i == 947u32 { v = 0x2b19082bu32; }
        if i == 948u32 { v = 0x2b19082bu32; }
        if i == 949u32 { v = 0x2b19082bu32; }
        if i == 950u32 { v = 0x2b191908u32; }
        if i == 951u32 { v = 0x2b191908u32; }
        if i == 952u32 { v = 0x2b191908u32; }
        if i == 953u32 { v = 0x2b191908u32; }
        if i == 954u32 { v = 0x2b191908u32; }
        if i == 955u32 { v = 0x2b191908u32; }
        if i == 956u32 { v = 0x2b191908u32; }
        if i == 957u32 { v = 0x2b191908u32; }
        if i == 958u32 { v = 0x2b191908u32; }
        if i == 959u32 { v = 0x2b191908u32; }
        if i == 960u32 { v = 0x2b191908u32; }
        if i == 961u32 { v = 0x2b191908u32; }
        if i == 962u32 { v = 0x2b191919u32; }
        if i == 963u32 { v = 0x2b191919u32; }
        if i == 964u32 { v = 0x2b191919u32; }
        if i == 965u32 { v = 0x2b191919u32; }
        if i == 966u32 { v = 0x2b191919u32; }
        if i == 967u32 { v = 0x2b191919u32; }
        if i == 968u32 { v = 0x2b19192bu32; }
        if i == 969u32 { v = 0x2b19192bu32; }
        if i == 970u32 { v = 0x2b19192bu32; }
        if i == 971u32 { v = 0x2b192b08u32; }
        if i == 972u32 { v = 0x2b192b08u32; }
        if i == 973u32 { v = 0x2b192b08u32; }
        if i == 974u32 { v = 0x2b192b08u32; }
        if i == 975u32 { v = 0x2b192b08u32; }
        if i == 976u32 { v = 0x2b192b08u32; }
        if i == 977u32 { v = 0x2b192b19u32; }
        if i == 978u32 { v = 0x2b192b19u32; }
        if i == 979u32 { v = 0x2b192b19u32; }
        if i == 980u32 { v = 0x2b192b2bu32; }
        if i == 981u32 { v = 0x2b2b0808u32; }
        if i == 982u32 { v = 0x2b2b0808u32; }
        if i == 983u32 { v = 0x2b2b0808u32; }
        if i == 984u32 { v = 0x2b2b0808u32; }
        if i == 985u32 { v = 0x2b2b0808u32; }
        if i == 986u32 { v = 0x2b2b0808u32; }
        if i == 987u32 { v = 0x2b2b0808u32; }
        if i == 988u32 { v = 0x2b2b0808u32; }
        if i == 989u32 { v = 0x2b2b0808u32; }
        if i == 990u32 { v = 0x2b2b0808u32; }
        if i == 991u32 { v = 0x2b2b0808u32; }
        if i == 992u32 { v = 0x2b2b0819u32; }
        if i == 993u32 { v = 0x2b2b0819u32; }
        if i == 994u32 { v = 0x2b2b082bu32; }
        if i == 995u32 { v = 0x2b2b082bu32; }
        if i == 996u32 { v = 0x2b2b082bu32; }
        if i == 997u32 { v = 0x2b2b082bu32; }
        if i == 998u32 { v = 0x2b2b082bu32; }
        if i == 999u32 { v = 0x2b2b082bu32; }
        if i == 1000u32 { v = 0x2b2b082bu32; }
        if i == 1001u32 { v = 0x2b2b082bu32; }
        if i == 1002u32 { v = 0x2b2b082bu32; }
        if i == 1003u32 { v = 0x2b2b1908u32; }
        if i == 1004u32 { v = 0x2b2b1908u32; }
        if i == 1005u32 { v = 0x2b2b1908u32; }
        if i == 1006u32 { v = 0x2b2b1908u32; }
        if i == 1007u32 { v = 0x2b2b1908u32; }
        if i == 1008u32 { v = 0x2b2b1908u32; }
        if i == 1009u32 { v = 0x2b2b1919u32; }
        if i == 1010u32 { v = 0x2b2b1919u32; }
        if i == 1011u32 { v = 0x2b2b192bu32; }
        if i == 1012u32 { v = 0x2b2b2b08u32; }
        if i == 1013u32 { v = 0x2b2b2b08u32; }
        if i == 1014u32 { v = 0x2b2b2b08u32; }
        if i == 1015u32 { v = 0x2b2b2b19u32; }
        if i == 1016u32 { v = 0x2b2b2b19u32; }
        if i == 1017u32 { v = 0x2b2b2b2bu32; }
        if i == 1018u32 { v = 0x2b2b2b2bu32; }
        if i == 1019u32 { v = 0x2b2b2b2bu32; }
        if i == 1020u32 { v = 0x2b2b2b2bu32; }
        if i == 1021u32 { v = 0x2b2b2b2bu32; }
        if i == 1022u32 { v = 0x2b2b2b2bu32; }
        if i == 1023u32 { v = 0x2b2b2b2bu32; }
        v
    }

pub fn xs_grid_lo(i: u32) -> u32 {
        let mut v: u32 = 0;
        if i == 0u32 { v = 0x08080808u32; }
        if i == 1u32 { v = 0x0808082bu32; }
        if i == 2u32 { v = 0x08081919u32; }
        if i == 3u32 { v = 0x08082b08u32; }
        if i == 4u32 { v = 0x08082b2bu32; }
        if i == 5u32 { v = 0x08190819u32; }
        if i == 6u32 { v = 0x08191908u32; }
        if i == 7u32 { v = 0x0819192bu32; }
        if i == 8u32 { v = 0x08192b19u32; }
        if i == 9u32 { v = 0x082b0808u32; }
        if i == 10u32 { v = 0x082b082bu32; }
        if i == 11u32 { v = 0x082b1919u32; }
        if i == 12u32 { v = 0x082b2b08u32; }
        if i == 13u32 { v = 0x19080819u32; }
        if i == 14u32 { v = 0x19081908u32; }
        if i == 15u32 { v = 0x1908192bu32; }
        if i == 16u32 { v = 0x19082b19u32; }
        if i == 17u32 { v = 0x19190808u32; }
        if i == 18u32 { v = 0x1919082bu32; }
        if i == 19u32 { v = 0x19191919u32; }
        if i == 20u32 { v = 0x19192b08u32; }
        if i == 21u32 { v = 0x192b0819u32; }
        if i == 22u32 { v = 0x192b1908u32; }
        if i == 23u32 { v = 0x2b080808u32; }
        if i == 24u32 { v = 0x2b08082bu32; }
        if i == 25u32 { v = 0x2b081919u32; }
        if i == 26u32 { v = 0x2b082b08u32; }
        if i == 27u32 { v = 0x2b190819u32; }
        if i == 28u32 { v = 0x2b191908u32; }
        if i == 29u32 { v = 0x2b192b19u32; }
        if i == 30u32 { v = 0x2b2b0808u32; }
        if i == 31u32 { v = 0x08080819u32; }
        if i == 32u32 { v = 0x08081908u32; }
        if i == 33u32 { v = 0x0808192bu32; }
        if i == 34u32 { v = 0x08082b19u32; }
        if i == 35u32 { v = 0x08190808u32; }
        if i == 36u32 { v = 0x0819082bu32; }
        if i == 37u32 { v = 0x08191919u32; }
        if i == 38u32 { v = 0x08192b08u32; }
        if i == 39u32 { v = 0x08192b2bu32; }
        if i == 40u32 { v = 0x082b0819u32; }
        if i == 41u32 { v = 0x082b1908u32; }
        if i == 42u32 { v = 0x19080808u32; }
        if i == 43u32 { v = 0x1908082bu32; }
        if i == 44u32 { v = 0x19081919u32; }
        if i == 45u32 { v = 0x19082b08u32; }
        if i == 46u32 { v = 0x19190819u32; }
        if i == 47u32 { v = 0x19191908u32; }
        if i == 48u32 { v = 0x192b0808u32; }
        if i == 49u32 { v = 0x192b2b08u32; }
        if i == 50u32 { v = 0x2b080819u32; }
        if i == 51u32 { v = 0x2b081908u32; }
        if i == 52u32 { v = 0x2b190808u32; }
        if i == 53u32 { v = 0x08080808u32; }
        if i == 54u32 { v = 0x0808082bu32; }
        if i == 55u32 { v = 0x08081919u32; }
        if i == 56u32 { v = 0x08082b08u32; }
        if i == 57u32 { v = 0x08190819u32; }
        if i == 58u32 { v = 0x08191908u32; }
        if i == 59u32 { v = 0x082b0808u32; }
        if i == 60u32 { v = 0x19080819u32; }
        if i == 61u32 { v = 0x19081908u32; }
        if i == 62u32 { v = 0x19190808u32; }
        if i == 63u32 { v = 0x19191919u32; }
        if i == 64u32 { v = 0x2b080808u32; }
        if i == 65u32 { v = 0x2b082b2bu32; }
        if i == 66u32 { v = 0x08080819u32; }
        if i == 67u32 { v = 0x08081908u32; }
        if i == 68u32 { v = 0x0808192bu32; }
        if i == 69u32 { v = 0x08082b19u32; }
        if i == 70u32 { v = 0x08190808u32; }
        if i == 71u32 { v = 0x0819082bu32; }
        if i == 72u32 { v = 0x08191919u32; }
        if i == 73u32 { v = 0x08192b08u32; }
        if i == 74u32 { v = 0x082b0819u32; }
        if i == 75u32 { v = 0x082b1908u32; }
        if i == 76u32 { v = 0x19080808u32; }
        if i == 77u32 { v = 0x1908082bu32; }
        if i == 78u32 { v = 0x19081919u32; }
        if i == 79u32 { v = 0x19082b08u32; }
        if i == 80u32 { v = 0x19190819u32; }
        if i == 81u32 { v = 0x19191908u32; }
        if i == 82u32 { v = 0x1919192bu32; }
        if i == 83u32 { v = 0x192b0808u32; }
        if i == 84u32 { v = 0x2b080819u32; }
        if i == 85u32 { v = 0x2b081908u32; }
        if i == 86u32 { v = 0x2b190808u32; }
        if i == 87u32 { v = 0x08080808u32; }
        if i == 88u32 { v = 0x0808082bu32; }
        if i == 89u32 { v = 0x08081919u32; }
        if i == 90u32 { v = 0x08082b08u32; }
        if i == 91u32 { v = 0x08190819u32; }
        if i == 92u32 { v = 0x08191908u32; }
        if i == 93u32 { v = 0x082b0808u32; }
        if i == 94u32 { v = 0x19080819u32; }
        if i == 95u32 { v = 0x19081908u32; }
        if i == 96u32 { v = 0x19190808u32; }
        if i == 97u32 { v = 0x192b0819u32; }
        if i == 98u32 { v = 0x2b080808u32; }
        if i == 99u32 { v = 0x08080819u32; }
        if i == 100u32 { v = 0x08081908u32; }
        if i == 101u32 { v = 0x08190808u32; }
        if i == 102u32 { v = 0x082b192bu32; }
        if i == 103u32 { v = 0x19080808u32; }
        if i == 104u32 { v = 0x1908082bu32; }
        if i == 105u32 { v = 0x2b081908u32; }
        if i == 106u32 { v = 0x08080808u32; }
        if i == 107u32 { v = 0x0808082bu32; }
        if i == 108u32 { v = 0x08081919u32; }
        if i == 109u32 { v = 0x08082b08u32; }
        if i == 110u32 { v = 0x08082b2bu32; }
        if i == 111u32 { v = 0x08190819u32; }
        if i == 112u32 { v = 0x08191908u32; }
        if i == 113u32 { v = 0x082b0808u32; }
        if i == 114u32 { v = 0x082b1919u32; }
        if i == 115u32 { v = 0x19080819u32; }
        if i == 116u32 { v = 0x19081908u32; }
        if i == 117u32 { v = 0x19190808u32; }
        if i == 118u32 { v = 0x19192b08u32; }
        if i == 119u32 { v = 0x2b080808u32; }
        if i == 120u32 { v = 0x2b2b0808u32; }
        if i == 121u32 { v = 0x2b2b2b2bu32; }
        if i == 122u32 { v = 0x08080819u32; }
        if i == 123u32 { v = 0x08081908u32; }
        if i == 124u32 { v = 0x08190808u32; }
        if i == 125u32 { v = 0x19080808u32; }
        if i == 126u32 { v = 0x2b080819u32; }
        if i == 127u32 { v = 0x2b082b19u32; }
        if i == 128u32 { v = 0x08080808u32; }
        if i == 129u32 { v = 0x082b0808u32; }
        if i == 130u32 { v = 0x082b2b08u32; }
        if i == 131u32 { v = 0x2b19192bu32; }
        if i == 132u32 { v = 0x2b2b0808u32; }
        if i == 133u32 { v = 0x08080819u32; }
        if i == 134u32 { v = 0x08081908u32; }
        if i == 135u32 { v = 0x0808192bu32; }
        if i == 136u32 { v = 0x08082b19u32; }
        if i == 137u32 { v = 0x08190808u32; }
        if i == 138u32 { v = 0x0819082bu32; }
        if i == 139u32 { v = 0x08191919u32; }
        if i == 140u32 { v = 0x08192b08u32; }
        if i == 141u32 { v = 0x082b0819u32; }
        if i == 142u32 { v = 0x082b1908u32; }
        if i == 143u32 { v = 0x19080808u32; }
        if i == 144u32 { v = 0x1908082bu32; }
        if i == 145u32 { v = 0x19081919u32; }
        if i == 146u32 { v = 0x19082b08u32; }
        if i == 147u32 { v = 0x19190819u32; }
        if i == 148u32 { v = 0x19191908u32; }
        if i == 149u32 { v = 0x192b0808u32; }
        if i == 150u32 { v = 0x192b2b2bu32; }
        if i == 151u32 { v = 0x2b080819u32; }
        if i == 152u32 { v = 0x2b081908u32; }
        if i == 153u32 { v = 0x2b190808u32; }
        if i == 154u32 { v = 0x08080808u32; }
        if i == 155u32 { v = 0x0808082bu32; }
        if i == 156u32 { v = 0x08081919u32; }
        if i == 157u32 { v = 0x08082b08u32; }
        if i == 158u32 { v = 0x08190819u32; }
        if i == 159u32 { v = 0x08191908u32; }
        if i == 160u32 { v = 0x082b0808u32; }
        if i == 161u32 { v = 0x19080819u32; }
        if i == 162u32 { v = 0x19081908u32; }
        if i == 163u32 { v = 0x19190808u32; }
        if i == 164u32 { v = 0x2b080808u32; }
        if i == 165u32 { v = 0x2b191908u32; }
        if i == 166u32 { v = 0x2b19192bu32; }
        if i == 167u32 { v = 0x08080819u32; }
        if i == 168u32 { v = 0x08081908u32; }
        if i == 169u32 { v = 0x0808192bu32; }
        if i == 170u32 { v = 0x08190808u32; }
        if i == 171u32 { v = 0x19080808u32; }
        if i == 172u32 { v = 0x192b0808u32; }
        if i == 173u32 { v = 0x08080808u32; }
        if i == 174u32 { v = 0x0808082bu32; }
        if i == 175u32 { v = 0x08081919u32; }
        if i == 176u32 { v = 0x08082b08u32; }
        if i == 177u32 { v = 0x08190819u32; }
        if i == 178u32 { v = 0x08191908u32; }
        if i == 179u32 { v = 0x082b0808u32; }
        if i == 180u32 { v = 0x19080819u32; }
        if i == 181u32 { v = 0x19081908u32; }
        if i == 182u32 { v = 0x19082b19u32; }
        if i == 183u32 { v = 0x19190808u32; }
        if i == 184u32 { v = 0x192b1908u32; }
        if i == 185u32 { v = 0x2b080808u32; }
        if i == 186u32 { v = 0x08080819u32; }
        if i == 187u32 { v = 0x08081908u32; }
        if i == 188u32 { v = 0x08190808u32; }
        if i == 189u32 { v = 0x19080808u32; }
        if i == 190u32 { v = 0x08080808u32; }
        if i == 191u32 { v = 0x08191908u32; }
        if i == 192u32 { v = 0x19082b19u32; }
        if i == 193u32 { v = 0x08080819u32; }
        if i == 194u32 { v = 0x08081908u32; }
        if i == 195u32 { v = 0x08190808u32; }
        if i == 196u32 { v = 0x0819082bu32; }
        if i == 197u32 { v = 0x19080808u32; }
        if i == 198u32 { v = 0x19191908u32; }
        if i == 199u32 { v = 0x2b08192bu32; }
        if i == 200u32 { v = 0x08080808u32; }
        if i == 201u32 { v = 0x08081919u32; }
        if i == 202u32 { v = 0x192b192bu32; }
        if i == 203u32 { v = 0x19190819u32; }
        if i == 204u32 { v = 0x2b2b2b19u32; }
        if i == 205u32 { v = 0x08080808u32; }
        if i == 206u32 { v = 0x0808082bu32; }
        if i == 207u32 { v = 0x08081919u32; }
        if i == 208u32 { v = 0x08082b08u32; }
        if i == 209u32 { v = 0x08082b2bu32; }
        if i == 210u32 { v = 0x08190819u32; }
        if i == 211u32 { v = 0x08191908u32; }
        if i == 212u32 { v = 0x082b0808u32; }
        if i == 213u32 { v = 0x19080819u32; }
        if i == 214u32 { v = 0x19081908u32; }
        if i == 215u32 { v = 0x19190808u32; }
        if i == 216u32 { v = 0x2b080808u32; }
        if i == 217u32 { v = 0x2b2b0808u32; }
        if i == 218u32 { v = 0x08080819u32; }
        if i == 219u32 { v = 0x08081908u32; }
        if i == 220u32 { v = 0x08190808u32; }
        if i == 221u32 { v = 0x19080808u32; }
        if i == 222u32 { v = 0x19082b08u32; }
        if i == 223u32 { v = 0x192b1919u32; }
        if i == 224u32 { v = 0x08080808u32; }
        if i == 225u32 { v = 0x082b082bu32; }
        if i == 226u32 { v = 0x2b080808u32; }
        if i == 227u32 { v = 0x2b2b2b08u32; }
        if i == 228u32 { v = 0x08080819u32; }
        if i == 229u32 { v = 0x08081908u32; }
        if i == 230u32 { v = 0x08190808u32; }
        if i == 231u32 { v = 0x082b2b19u32; }
        if i == 232u32 { v = 0x19080808u32; }
        if i == 233u32 { v = 0x08080808u32; }
        if i == 234u32 { v = 0x19080819u32; }
        if i == 235u32 { v = 0x1919082bu32; }
        if i == 236u32 { v = 0x2b192b19u32; }
        if i == 237u32 { v = 0x08080819u32; }
        if i == 238u32 { v = 0x08192b2bu32; }
        if i == 239u32 { v = 0x2b2b192bu32; }
        if i == 240u32 { v = 0x08080808u32; }
        if i == 241u32 { v = 0x08082b08u32; }
        if i == 242u32 { v = 0x08082b2bu32; }
        if i == 243u32 { v = 0x082b0808u32; }
        if i == 244u32 { v = 0x19191919u32; }
        if i == 245u32 { v = 0x2b082b08u32; }
        if i == 246u32 { v = 0x2b2b082bu32; }
        if i == 247u32 { v = 0x192b2b08u32; }
        if i == 248u32 { v = 0x2b190808u32; }
        if i == 249u32 { v = 0x08082b08u32; }
        if i == 250u32 { v = 0x082b0808u32; }
        if i == 251u32 { v = 0x2b08082bu32; }
        if i == 252u32 { v = 0x2b082b08u32; }
        if i == 253u32 { v = 0x2b082b2bu32; }
        if i == 254u32 { v = 0x08080819u32; }
        if i == 255u32 { v = 0x08081908u32; }
        if i == 256u32 { v = 0x0808192bu32; }
        if i == 257u32 { v = 0x08082b19u32; }
        if i == 258u32 { v = 0x08190808u32; }
        if i == 259u32 { v = 0x0819082bu32; }
        if i == 260u32 { v = 0x08191919u32; }
        if i == 261u32 { v = 0x08192b08u32; }
        if i == 262u32 { v = 0x082b0819u32; }
        if i == 263u32 { v = 0x082b1908u32; }
        if i == 264u32 { v = 0x19080808u32; }
        if i == 265u32 { v = 0x1908082bu32; }
        if i == 266u32 { v = 0x19081919u32; }
        if i == 267u32 { v = 0x19082b08u32; }
        if i == 268u32 { v = 0x19082b2bu32; }
        if i == 269u32 { v = 0x19190819u32; }
        if i == 270u32 { v = 0x19191908u32; }
        if i == 271u32 { v = 0x192b0808u32; }
        if i == 272u32 { v = 0x192b1919u32; }
        if i == 273u32 { v = 0x2b080819u32; }
        if i == 274u32 { v = 0x2b081908u32; }
        if i == 275u32 { v = 0x2b190808u32; }
        if i == 276u32 { v = 0x08080808u32; }
        if i == 277u32 { v = 0x0808082bu32; }
        if i == 278u32 { v = 0x08081919u32; }
        if i == 279u32 { v = 0x08082b08u32; }
        if i == 280u32 { v = 0x08190819u32; }
        if i == 281u32 { v = 0x08191908u32; }
        if i == 282u32 { v = 0x082b0808u32; }
        if i == 283u32 { v = 0x19080819u32; }
        if i == 284u32 { v = 0x19081908u32; }
        if i == 285u32 { v = 0x19190808u32; }
        if i == 286u32 { v = 0x2b080808u32; }
        if i == 287u32 { v = 0x2b081919u32; }
        if i == 288u32 { v = 0x2b2b082bu32; }
        if i == 289u32 { v = 0x08080819u32; }
        if i == 290u32 { v = 0x08081908u32; }
        if i == 291u32 { v = 0x08190808u32; }
        if i == 292u32 { v = 0x0819082bu32; }
        if i == 293u32 { v = 0x082b2b19u32; }
        if i == 294u32 { v = 0x19080808u32; }
        if i == 295u32 { v = 0x08080808u32; }
        if i == 296u32 { v = 0x0808082bu32; }
        if i == 297u32 { v = 0x08081919u32; }
        if i == 298u32 { v = 0x08082b08u32; }
        if i == 299u32 { v = 0x08190819u32; }
        if i == 300u32 { v = 0x08191908u32; }
        if i == 301u32 { v = 0x08192b19u32; }
        if i == 302u32 { v = 0x082b0808u32; }
        if i == 303u32 { v = 0x19080819u32; }
        if i == 304u32 { v = 0x19081908u32; }
        if i == 305u32 { v = 0x19190808u32; }
        if i == 306u32 { v = 0x2b080808u32; }
        if i == 307u32 { v = 0x2b191908u32; }
        if i == 308u32 { v = 0x08080819u32; }
        if i == 309u32 { v = 0x08081908u32; }
        if i == 310u32 { v = 0x08190808u32; }
        if i == 311u32 { v = 0x082b1908u32; }
        if i == 312u32 { v = 0x19080808u32; }
        if i == 313u32 { v = 0x2b192b2bu32; }
        if i == 314u32 { v = 0x08080808u32; }
        if i == 315u32 { v = 0x08082b2bu32; }
        if i == 316u32 { v = 0x19081908u32; }
        if i == 317u32 { v = 0x19190808u32; }
        if i == 318u32 { v = 0x08080819u32; }
        if i == 319u32 { v = 0x08081908u32; }
        if i == 320u32 { v = 0x08190808u32; }
        if i == 321u32 { v = 0x19080808u32; }
        if i == 322u32 { v = 0x19081919u32; }
        if i == 323u32 { v = 0x19191908u32; }
        if i == 324u32 { v = 0x192b082bu32; }
        if i == 325u32 { v = 0x08080808u32; }
        if i == 326u32 { v = 0x08190819u32; }
        if i == 327u32 { v = 0x19081908u32; }
        if i == 328u32 { v = 0x19190808u32; }
        if i == 329u32 { v = 0x192b2b19u32; }
        if i == 330u32 { v = 0x08081908u32; }
        if i == 331u32 { v = 0x08080808u32; }
        if i == 332u32 { v = 0x0808082bu32; }
        if i == 333u32 { v = 0x08081919u32; }
        if i == 334u32 { v = 0x08082b08u32; }
        if i == 335u32 { v = 0x08190819u32; }
        if i == 336u32 { v = 0x08191908u32; }
        if i == 337u32 { v = 0x082b0808u32; }
        if i == 338u32 { v = 0x082b2b08u32; }
        if i == 339u32 { v = 0x19080819u32; }
        if i == 340u32 { v = 0x19081908u32; }
        if i == 341u32 { v = 0x19190808u32; }
        if i == 342u32 { v = 0x2b080808u32; }
        if i == 343u32 { v = 0x08080819u32; }
        if i == 344u32 { v = 0x08081908u32; }
        if i == 345u32 { v = 0x08190808u32; }
        if i == 346u32 { v = 0x08191919u32; }
        if i == 347u32 { v = 0x19080808u32; }
        if i == 348u32 { v = 0x1908082bu32; }
        if i == 349u32 { v = 0x08080808u32; }
        if i == 350u32 { v = 0x19081908u32; }
        if i == 351u32 { v = 0x2b2b2b2bu32; }
        if i == 352u32 { v = 0x08080819u32; }
        if i == 353u32 { v = 0x08081908u32; }
        if i == 354u32 { v = 0x08190808u32; }
        if i == 355u32 { v = 0x082b0819u32; }
        if i == 356u32 { v = 0x19080808u32; }
        if i == 357u32 { v = 0x192b0808u32; }
        if i == 358u32 { v = 0x2b080819u32; }
        if i == 359u32 { v = 0x2b2b0819u32; }
        if i == 360u32 { v = 0x08080808u32; }
        if i == 361u32 { v = 0x08082b08u32; }
        if i == 362u32 { v = 0x2b080808u32; }
        if i == 363u32 { v = 0x2b082b08u32; }
        if i == 364u32 { v = 0x082b0819u32; }
        if i == 365u32 { v = 0x192b2b08u32; }
        if i == 366u32 { v = 0x2b2b0819u32; }
        if i == 367u32 { v = 0x08080808u32; }
        if i == 368u32 { v = 0x08191908u32; }
        if i == 369u32 { v = 0x19080819u32; }
        if i == 370u32 { v = 0x19190808u32; }
        if i == 371u32 { v = 0x2b192b19u32; }
        if i == 372u32 { v = 0x08192b2bu32; }
        if i == 373u32 { v = 0x19080808u32; }
        if i == 374u32 { v = 0x1908082bu32; }
        if i == 375u32 { v = 0x2b081919u32; }
        if i == 376u32 { v = 0x08080819u32; }
        if i == 377u32 { v = 0x08081908u32; }
        if i == 378u32 { v = 0x08190808u32; }
        if i == 379u32 { v = 0x19080808u32; }
        if i == 380u32 { v = 0x19191908u32; }
        if i == 381u32 { v = 0x192b082bu32; }
        if i == 382u32 { v = 0x2b08192bu32; }
        if i == 383u32 { v = 0x2b2b2b19u32; }
        if i == 384u32 { v = 0x08080808u32; }
        if i == 385u32 { v = 0x082b1908u32; }
        if i == 386u32 { v = 0x19082b2bu32; }
        if i == 387u32 { v = 0x2b19082bu32; }
        if i == 388u32 { v = 0x08080808u32; }
        if i == 389u32 { v = 0x0819192bu32; }
        if i == 390u32 { v = 0x08190808u32; }
        if i == 391u32 { v = 0x19080808u32; }
        if i == 392u32 { v = 0x19081919u32; }
        if i == 393u32 { v = 0x2b2b1908u32; }
        if i == 394u32 { v = 0x08080819u32; }
        if i == 395u32 { v = 0x192b2b2bu32; }
        if i == 396u32 { v = 0x082b1919u32; }
        if i == 397u32 { v = 0x0808192bu32; }
        if i == 398u32 { v = 0x19191908u32; }
        if i == 399u32 { v = 0x192b082bu32; }
        if i == 400u32 { v = 0x08080808u32; }
        if i == 401u32 { v = 0x0808082bu32; }
        if i == 402u32 { v = 0x08081919u32; }
        if i == 403u32 { v = 0x08082b08u32; }
        if i == 404u32 { v = 0x08190819u32; }
        if i == 405u32 { v = 0x08191908u32; }
        if i == 406u32 { v = 0x082b0808u32; }
        if i == 407u32 { v = 0x082b2b2bu32; }
        if i == 408u32 { v = 0x19080819u32; }
        if i == 409u32 { v = 0x19081908u32; }
        if i == 410u32 { v = 0x19190808u32; }
        if i == 411u32 { v = 0x2b080808u32; }
        if i == 412u32 { v = 0x2b08082bu32; }
        if i == 413u32 { v = 0x2b2b2b08u32; }
        if i == 414u32 { v = 0x2b2b2b2bu32; }
        if i == 415u32 { v = 0x08080819u32; }
        if i == 416u32 { v = 0x08081908u32; }
        if i == 417u32 { v = 0x0808192bu32; }
        if i == 418u32 { v = 0x08190808u32; }
        if i == 419u32 { v = 0x19080808u32; }
        if i == 420u32 { v = 0x19190819u32; }
        if i == 421u32 { v = 0x19192b19u32; }
        if i == 422u32 { v = 0x08080808u32; }
        if i == 423u32 { v = 0x082b0808u32; }
        if i == 424u32 { v = 0x2b080808u32; }
        if i == 425u32 { v = 0x2b08082bu32; }
        if i == 426u32 { v = 0x2b2b0808u32; }
        if i == 427u32 { v = 0x2b2b2b08u32; }
        if i == 428u32 { v = 0x08080819u32; }
        if i == 429u32 { v = 0x08081908u32; }
        if i == 430u32 { v = 0x08190808u32; }
        if i == 431u32 { v = 0x0819082bu32; }
        if i == 432u32 { v = 0x08191919u32; }
        if i == 433u32 { v = 0x19080808u32; }
        if i == 434u32 { v = 0x192b0808u32; }
        if i == 435u32 { v = 0x2b082b19u32; }
        if i == 436u32 { v = 0x08080808u32; }
        if i == 437u32 { v = 0x19081908u32; }
        if i == 438u32 { v = 0x2b2b1919u32; }
        if i == 439u32 { v = 0x08192b08u32; }
        if i == 440u32 { v = 0x192b2b2bu32; }
        if i == 441u32 { v = 0x08080808u32; }
        if i == 442u32 { v = 0x08082b08u32; }
        if i == 443u32 { v = 0x082b1919u32; }
        if i == 444u32 { v = 0x19192b2bu32; }
        if i == 445u32 { v = 0x2b080808u32; }
        if i == 446u32 { v = 0x2b08082bu32; }
        if i == 447u32 { v = 0x2b2b2b08u32; }
        if i == 448u32 { v = 0x0808192bu32; }
        if i == 449u32 { v = 0x082b082bu32; }
        if i == 450u32 { v = 0x2b080808u32; }
        if i == 451u32 { v = 0x2b082b08u32; }
        if i == 452u32 { v = 0x2b19192bu32; }
        if i == 453u32 { v = 0x2b2b2b08u32; }
        if i == 454u32 { v = 0x08080819u32; }
        if i == 455u32 { v = 0x08081908u32; }
        if i == 456u32 { v = 0x08190808u32; }
        if i == 457u32 { v = 0x19080808u32; }
        if i == 458u32 { v = 0x1919192bu32; }
        if i == 459u32 { v = 0x2b081908u32; }
        if i == 460u32 { v = 0x08080808u32; }
        if i == 461u32 { v = 0x082b082bu32; }
        if i == 462u32 { v = 0x192b1908u32; }
        if i == 463u32 { v = 0x1919192bu32; }
        if i == 464u32 { v = 0x2b082b19u32; }
        if i == 465u32 { v = 0x08080808u32; }
        if i == 466u32 { v = 0x08081919u32; }
        if i == 467u32 { v = 0x19081908u32; }
        if i == 468u32 { v = 0x19190808u32; }
        if i == 469u32 { v = 0x19192b08u32; }
        if i == 470u32 { v = 0x082b2b19u32; }
        if i == 471u32 { v = 0x2b190808u32; }
        if i == 472u32 { v = 0x2b19082bu32; }
        if i == 473u32 { v = 0x19080819u32; }
        if i == 474u32 { v = 0x19190819u32; }
        if i == 475u32 { v = 0x2b2b192bu32; }
        if i == 476u32 { v = 0x19082b19u32; }
        if i == 477u32 { v = 0x08191919u32; }
        if i == 478u32 { v = 0x192b0808u32; }
        if i == 479u32 { v = 0x08080808u32; }
        if i == 480u32 { v = 0x0808082bu32; }
        if i == 481u32 { v = 0x08082b08u32; }
        if i == 482u32 { v = 0x08082b2bu32; }
        if i == 483u32 { v = 0x082b0808u32; }
        if i == 484u32 { v = 0x082b2b2bu32; }
        if i == 485u32 { v = 0x2b2b0808u32; }
        if i == 486u32 { v = 0x19190819u32; }
        if i == 487u32 { v = 0x19192b19u32; }
        if i == 488u32 { v = 0x2b2b192bu32; }
        if i == 489u32 { v = 0x08080808u32; }
        if i == 490u32 { v = 0x0808082bu32; }
        if i == 491u32 { v = 0x08082b08u32; }
        if i == 492u32 { v = 0x082b2b2bu32; }
        if i == 493u32 { v = 0x2b080808u32; }
        if i == 494u32 { v = 0x2b2b0808u32; }
        if i == 495u32 { v = 0x19080808u32; }
        if i == 496u32 { v = 0x2b191919u32; }
        if i == 497u32 { v = 0x192b1919u32; }
        if i == 498u32 { v = 0x2b192b08u32; }
        if i == 499u32 { v = 0x08082b2bu32; }
        if i == 500u32 { v = 0x082b0808u32; }
        if i == 501u32 { v = 0x082b082bu32; }
        if i == 502u32 { v = 0x082b2b08u32; }
        if i == 503u32 { v = 0x2b2b0808u32; }
        if i == 504u32 { v = 0x2b2b2b08u32; }
        if i == 505u32 { v = 0x08081908u32; }
        if i == 506u32 { v = 0x2b081908u32; }
        if i == 507u32 { v = 0x2b08192bu32; }
        if i == 508u32 { v = 0x082b2b08u32; }
        if i == 509u32 { v = 0x082b2b2bu32; }
        if i == 510u32 { v = 0x2b190819u32; }
        if i == 511u32 { v = 0x2b2b2b2bu32; }
        v
    }

pub fn xs_grid_hi(i: u32) -> u32 {
        let mut v: u32 = 0;
        if i == 0u32 { v = 0x08080808u32; }
        if i == 1u32 { v = 0x08080808u32; }
        if i == 2u32 { v = 0x08080808u32; }
        if i == 3u32 { v = 0x08080808u32; }
        if i == 4u32 { v = 0x08080808u32; }
        if i == 5u32 { v = 0x08080808u32; }
        if i == 6u32 { v = 0x08080808u32; }
        if i == 7u32 { v = 0x08080808u32; }
        if i == 8u32 { v = 0x08080808u32; }
        if i == 9u32 { v = 0x08080808u32; }
        if i == 10u32 { v = 0x08080808u32; }
        if i == 11u32 { v = 0x08080808u32; }
        if i == 12u32 { v = 0x08080808u32; }
        if i == 13u32 { v = 0x08080808u32; }
        if i == 14u32 { v = 0x08080808u32; }
        if i == 15u32 { v = 0x08080808u32; }
        if i == 16u32 { v = 0x08080808u32; }
        if i == 17u32 { v = 0x08080808u32; }
        if i == 18u32 { v = 0x08080808u32; }
        if i == 19u32 { v = 0x08080808u32; }
        if i == 20u32 { v = 0x08080808u32; }
        if i == 21u32 { v = 0x08080808u32; }
        if i == 22u32 { v = 0x08080808u32; }
        if i == 23u32 { v = 0x08080808u32; }
        if i == 24u32 { v = 0x08080808u32; }
        if i == 25u32 { v = 0x08080808u32; }
        if i == 26u32 { v = 0x08080808u32; }
        if i == 27u32 { v = 0x08080808u32; }
        if i == 28u32 { v = 0x08080808u32; }
        if i == 29u32 { v = 0x08080808u32; }
        if i == 30u32 { v = 0x08080808u32; }
        if i == 31u32 { v = 0x08080819u32; }
        if i == 32u32 { v = 0x08080819u32; }
        if i == 33u32 { v = 0x08080819u32; }
        if i == 34u32 { v = 0x08080819u32; }
        if i == 35u32 { v = 0x08080819u32; }
        if i == 36u32 { v = 0x08080819u32; }
        if i == 37u32 { v = 0x08080819u32; }
        if i == 38u32 { v = 0x08080819u32; }
        if i == 39u32 { v = 0x08080819u32; }
        if i == 40u32 { v = 0x08080819u32; }
        if i == 41u32 { v = 0x08080819u32; }
        if i == 42u32 { v = 0x08080819u32; }
        if i == 43u32 { v = 0x08080819u32; }
        if i == 44u32 { v = 0x08080819u32; }
        if i == 45u32 { v = 0x08080819u32; }
        if i == 46u32 { v = 0x08080819u32; }
        if i == 47u32 { v = 0x08080819u32; }
        if i == 48u32 { v = 0x08080819u32; }
        if i == 49u32 { v = 0x08080819u32; }
        if i == 50u32 { v = 0x08080819u32; }
        if i == 51u32 { v = 0x08080819u32; }
        if i == 52u32 { v = 0x08080819u32; }
        if i == 53u32 { v = 0x0808082bu32; }
        if i == 54u32 { v = 0x0808082bu32; }
        if i == 55u32 { v = 0x0808082bu32; }
        if i == 56u32 { v = 0x0808082bu32; }
        if i == 57u32 { v = 0x0808082bu32; }
        if i == 58u32 { v = 0x0808082bu32; }
        if i == 59u32 { v = 0x0808082bu32; }
        if i == 60u32 { v = 0x0808082bu32; }
        if i == 61u32 { v = 0x0808082bu32; }
        if i == 62u32 { v = 0x0808082bu32; }
        if i == 63u32 { v = 0x0808082bu32; }
        if i == 64u32 { v = 0x0808082bu32; }
        if i == 65u32 { v = 0x0808082bu32; }
        if i == 66u32 { v = 0x08081908u32; }
        if i == 67u32 { v = 0x08081908u32; }
        if i == 68u32 { v = 0x08081908u32; }
        if i == 69u32 { v = 0x08081908u32; }
        if i == 70u32 { v = 0x08081908u32; }
        if i == 71u32 { v = 0x08081908u32; }
        if i == 72u32 { v = 0x08081908u32; }
        if i == 73u32 { v = 0x08081908u32; }
        if i == 74u32 { v = 0x08081908u32; }
        if i == 75u32 { v = 0x08081908u32; }
        if i == 76u32 { v = 0x08081908u32; }
        if i == 77u32 { v = 0x08081908u32; }
        if i == 78u32 { v = 0x08081908u32; }
        if i == 79u32 { v = 0x08081908u32; }
        if i == 80u32 { v = 0x08081908u32; }
        if i == 81u32 { v = 0x08081908u32; }
        if i == 82u32 { v = 0x08081908u32; }
        if i == 83u32 { v = 0x08081908u32; }
        if i == 84u32 { v = 0x08081908u32; }
        if i == 85u32 { v = 0x08081908u32; }
        if i == 86u32 { v = 0x08081908u32; }
        if i == 87u32 { v = 0x08081919u32; }
        if i == 88u32 { v = 0x08081919u32; }
        if i == 89u32 { v = 0x08081919u32; }
        if i == 90u32 { v = 0x08081919u32; }
        if i == 91u32 { v = 0x08081919u32; }
        if i == 92u32 { v = 0x08081919u32; }
        if i == 93u32 { v = 0x08081919u32; }
        if i == 94u32 { v = 0x08081919u32; }
        if i == 95u32 { v = 0x08081919u32; }
        if i == 96u32 { v = 0x08081919u32; }
        if i == 97u32 { v = 0x08081919u32; }
        if i == 98u32 { v = 0x08081919u32; }
        if i == 99u32 { v = 0x0808192bu32; }
        if i == 100u32 { v = 0x0808192bu32; }
        if i == 101u32 { v = 0x0808192bu32; }
        if i == 102u32 { v = 0x0808192bu32; }
        if i == 103u32 { v = 0x0808192bu32; }
        if i == 104u32 { v = 0x0808192bu32; }
        if i == 105u32 { v = 0x0808192bu32; }
        if i == 106u32 { v = 0x08082b08u32; }
        if i == 107u32 { v = 0x08082b08u32; }
        if i == 108u32 { v = 0x08082b08u32; }
        if i == 109u32 { v = 0x08082b08u32; }
        if i == 110u32 { v = 0x08082b08u32; }
        if i == 111u32 { v = 0x08082b08u32; }
        if i == 112u32 { v = 0x08082b08u32; }
        if i == 113u32 { v = 0x08082b08u32; }
        if i == 114u32 { v = 0x08082b08u32; }
        if i == 115u32 { v = 0x08082b08u32; }
        if i == 116u32 { v = 0x08082b08u32; }
        if i == 117u32 { v = 0x08082b08u32; }
        if i == 118u32 { v = 0x08082b08u32; }
        if i == 119u32 { v = 0x08082b08u32; }
        if i == 120u32 { v = 0x08082b08u32; }
        if i == 121u32 { v = 0x08082b08u32; }
        if i == 122u32 { v = 0x08082b19u32; }
        if i == 123u32 { v = 0x08082b19u32; }
        if i == 124u32 { v = 0x08082b19u32; }
        if i == 125u32 { v = 0x08082b19u32; }
        if i == 126u32 { v = 0x08082b19u32; }
        if i == 127u32 { v = 0x08082b19u32; }
        if i == 128u32 { v = 0x08082b2bu32; }
        if i == 129u32 { v = 0x08082b2bu32; }
        if i == 130u32 { v = 0x08082b2bu32; }
        if i == 131u32 { v = 0x08082b2bu32; }
        if i == 132u32 { v = 0x08082b2bu32; }
        if i == 133u32 { v = 0x08190808u32; }
        if i == 134u32 { v = 0x08190808u32; }
        if i == 135u32 { v = 0x08190808u32; }
        if i == 136u32 { v = 0x08190808u32; }
        if i == 137u32 { v = 0x08190808u32; }
        if i == 138u32 { v = 0x08190808u32; }
        if i == 139u32 { v = 0x08190808u32; }
        if i == 140u32 { v = 0x08190808u32; }
        if i == 141u32 { v = 0x08190808u32; }
        if i == 142u32 { v = 0x08190808u32; }
        if i == 143u32 { v = 0x08190808u32; }
        if i == 144u32 { v = 0x08190808u32; }
        if i == 145u32 { v = 0x08190808u32; }
        if i == 146u32 { v = 0x08190808u32; }
        if i == 147u32 { v = 0x08190808u32; }
        if i == 148u32 { v = 0x08190808u32; }
        if i == 149u32 { v = 0x08190808u32; }
        if i == 150u32 { v = 0x08190808u32; }
        if i == 151u32 { v = 0x08190808u32; }
        if i == 152u32 { v = 0x08190808u32; }
        if i == 153u32 { v = 0x08190808u32; }
        if i == 154u32 { v = 0x08190819u32; }
        if i == 155u32 { v = 0x08190819u32; }
        if i == 156u32 { v = 0x08190819u32; }
        if i == 157u32 { v = 0x08190819u32; }
        if i == 158u32 { v = 0x08190819u32; }
        if i == 159u32 { v = 0x08190819u32; }
        if i == 160u32 { v = 0x08190819u32; }
        if i == 161u32 { v = 0x08190819u32; }
        if i == 162u32 { v = 0x08190819u32; }
        if i == 163u32 { v = 0x08190819u32; }
        if i == 164u32 { v = 0x08190819u32; }
        if i == 165u32 { v = 0x08190819u32; }
        if i == 166u32 { v = 0x08190819u32; }
        if i == 167u32 { v = 0x0819082bu32; }
        if i == 168u32 { v = 0x0819082bu32; }
        if i == 169u32 { v = 0x0819082bu32; }
        if i == 170u32 { v = 0x0819082bu32; }
        if i == 171u32 { v = 0x0819082bu32; }
        if i == 172u32 { v = 0x0819082bu32; }
        if i == 173u32 { v = 0x08191908u32; }
        if i == 174u32 { v = 0x08191908u32; }
        if i == 175u32 { v = 0x08191908u32; }
        if i == 176u32 { v = 0x08191908u32; }
        if i == 177u32 { v = 0x08191908u32; }
        if i == 178u32 { v = 0x08191908u32; }
        if i == 179u32 { v = 0x08191908u32; }
        if i == 180u32 { v = 0x08191908u32; }
        if i == 181u32 { v = 0x08191908u32; }
        if i == 182u32 { v = 0x08191908u32; }
        if i == 183u32 { v = 0x08191908u32; }
        if i == 184u32 { v = 0x08191908u32; }
        if i == 185u32 { v = 0x08191908u32; }
        if i == 186u32 { v = 0x08191919u32; }
        if i == 187u32 { v = 0x08191919u32; }
        if i == 188u32 { v = 0x08191919u32; }
        if i == 189u32 { v = 0x08191919u32; }
        if i == 190u32 { v = 0x0819192bu32; }
        if i == 191u32 { v = 0x0819192bu32; }
        if i == 192u32 { v = 0x0819192bu32; }
        if i == 193u32 { v = 0x08192b08u32; }
        if i == 194u32 { v = 0x08192b08u32; }
        if i == 195u32 { v = 0x08192b08u32; }
        if i == 196u32 { v = 0x08192b08u32; }
        if i == 197u32 { v = 0x08192b08u32; }
        if i == 198u32 { v = 0x08192b08u32; }
        if i == 199u32 { v = 0x08192b08u32; }
        if i == 200u32 { v = 0x08192b19u32; }
        if i == 201u32 { v = 0x08192b19u32; }
        if i == 202u32 { v = 0x08192b19u32; }
        if i == 203u32 { v = 0x08192b2bu32; }
        if i == 204u32 { v = 0x08192b2bu32; }
        if i == 205u32 { v = 0x082b0808u32; }
        if i == 206u32 { v = 0x082b0808u32; }
        if i == 207u32 { v = 0x082b0808u32; }
        if i == 208u32 { v = 0x082b0808u32; }
        if i == 209u32 { v = 0x082b0808u32; }
        if i == 210u32 { v = 0x082b0808u32; }
        if i == 211u32 { v = 0x082b0808u32; }
        if i == 212u32 { v = 0x082b0808u32; }
        if i == 213u32 { v = 0x082b0808u32; }
        if i == 214u32 { v = 0x082b0808u32; }
        if i == 215u32 { v = 0x082b0808u32; }
        if i == 216u32 { v = 0x082b0808u32; }
        if i == 217u32 { v = 0x082b0808u32; }
        if i == 218u32 { v = 0x082b0819u32; }
        if i == 219u32 { v = 0x082b0819u32; }
        if i == 220u32 { v = 0x082b0819u32; }
        if i == 221u32 { v = 0x082b0819u32; }
        if i == 222u32 { v = 0x082b0819u32; }
        if i == 223u32 { v = 0x082b0819u32; }
        if i == 224u32 { v = 0x082b082bu32; }
        if i == 225u32 { v = 0x082b082bu32; }
        if i == 226u32 { v = 0x082b082bu32; }
        if i == 227u32 { v = 0x082b082bu32; }
        if i == 228u32 { v = 0x082b1908u32; }
        if i == 229u32 { v = 0x082b1908u32; }
        if i == 230u32 { v = 0x082b1908u32; }
        if i == 231u32 { v = 0x082b1908u32; }
        if i == 232u32 { v = 0x082b1908u32; }
        if i == 233u32 { v = 0x082b1919u32; }
        if i == 234u32 { v = 0x082b1919u32; }
        if i == 235u32 { v = 0x082b1919u32; }
        if i == 236u32 { v = 0x082b1919u32; }
        if i == 237u32 { v = 0x082b192bu32; }
        if i == 238u32 { v = 0x082b192bu32; }
        if i == 239u32 { v = 0x082b192bu32; }
        if i == 240u32 { v = 0x082b2b08u32; }
        if i == 241u32 { v = 0x082b2b08u32; }
        if i == 242u32 { v = 0x082b2b08u32; }
        if i == 243u32 { v = 0x082b2b08u32; }
        if i == 244u32 { v = 0x082b2b08u32; }
        if i == 245u32 { v = 0x082b2b08u32; }
        if i == 246u32 { v = 0x082b2b08u32; }
        if i == 247u32 { v = 0x082b2b19u32; }
        if i == 248u32 { v = 0x082b2b19u32; }
        if i == 249u32 { v = 0x082b2b2bu32; }
        if i == 250u32 { v = 0x082b2b2bu32; }
        if i == 251u32 { v = 0x082b2b2bu32; }
        if i == 252u32 { v = 0x082b2b2bu32; }
        if i == 253u32 { v = 0x082b2b2bu32; }
        if i == 254u32 { v = 0x19080808u32; }
        if i == 255u32 { v = 0x19080808u32; }
        if i == 256u32 { v = 0x19080808u32; }
        if i == 257u32 { v = 0x19080808u32; }
        if i == 258u32 { v = 0x19080808u32; }
        if i == 259u32 { v = 0x19080808u32; }
        if i == 260u32 { v = 0x19080808u32; }
        if i == 261u32 { v = 0x19080808u32; }
        if i == 262u32 { v = 0x19080808u32; }
        if i == 263u32 { v = 0x19080808u32; }
        if i == 264u32 { v = 0x19080808u32; }
        if i == 265u32 { v = 0x19080808u32; }
        if i == 266u32 { v = 0x19080808u32; }
        if i == 267u32 { v = 0x19080808u32; }
        if i == 268u32 { v = 0x19080808u32; }
        if i == 269u32 { v = 0x19080808u32; }
        if i == 270u32 { v = 0x19080808u32; }
        if i == 271u32 { v = 0x19080808u32; }
        if i == 272u32 { v = 0x19080808u32; }
        if i == 273u32 { v = 0x19080808u32; }
        if i == 274u32 { v = 0x19080808u32; }
        if i == 275u32 { v = 0x19080808u32; }
        if i == 276u32 { v = 0x19080819u32; }
        if i == 277u32 { v = 0x19080819u32; }
        if i == 278u32 { v = 0x19080819u32; }
        if i == 279u32 { v = 0x19080819u32; }
        if i == 280u32 { v = 0x19080819u32; }
        if i == 281u32 { v = 0x19080819u32; }
        if i == 282u32 { v = 0x19080819u32; }
        if i == 283u32 { v = 0x19080819u32; }
        if i == 284u32 { v = 0x19080819u32; }
        if i == 285u32 { v = 0x19080819u32; }
        if i == 286u32 { v = 0x19080819u32; }
        if i == 287u32 { v = 0x19080819u32; }
        if i == 288u32 { v = 0x19080819u32; }
        if i == 289u32 { v = 0x1908082bu32; }
        if i == 290u32 { v = 0x1908082bu32; }
        if i == 291u32 { v = 0x1908082bu32; }
        if i == 292u32 { v = 0x1908082bu32; }
        if i == 293u32 { v = 0x1908082bu32; }
        if i == 294u32 { v = 0x1908082bu32; }
        if i == 295u32 { v = 0x19081908u32; }
        if i == 296u32 { v = 0x19081908u32; }
        if i == 297u32 { v = 0x19081908u32; }
        if i == 298u32 { v = 0x19081908u32; }
        if i == 299u32 { v = 0x19081908u32; }
        if i == 300u32 { v = 0x19081908u32; }
        if i == 301u32 { v = 0x19081908u32; }
        if i == 302u32 { v = 0x19081908u32; }
        if i == 303u32 { v = 0x19081908u32; }
        if i == 304u32 { v = 0x19081908u32; }
        if i == 305u32 { v = 0x19081908u32; }
        if i == 306u32 { v = 0x19081908u32; }
        if i == 307u32 { v = 0x19081908u32; }
        if i == 308u32 { v = 0x19081919u32; }
        if i == 309u32 { v = 0x19081919u32; }
        if i == 310u32 { v = 0x19081919u32; }
        if i == 311u32 { v = 0x19081919u32; }
        if i == 312u32 { v = 0x19081919u32; }
        if i == 313u32 { v = 0x19081919u32; }
        if i == 314u32 { v = 0x1908192bu32; }
        if i == 315u32 { v = 0x1908192bu32; }
        if i == 316u32 { v = 0x1908192bu32; }
        if i == 317u32 { v = 0x1908192bu32; }
        if i == 318u32 { v = 0x19082b08u32; }
        if i == 319u32 { v = 0x19082b08u32; }
        if i == 320u32 { v = 0x19082b08u32; }
        if i == 321u32 { v = 0x19082b08u32; }
        if i == 322u32 { v = 0x19082b08u32; }
        if i == 323u32 { v = 0x19082b08u32; }
        if i == 324u32 { v = 0x19082b08u32; }
        if i == 325u32 { v = 0x19082b19u32; }
        if i == 326u32 { v = 0x19082b19u32; }
        if i == 327u32 { v = 0x19082b19u32; }
        if i == 328u32 { v = 0x19082b19u32; }
        if i == 329u32 { v = 0x19082b19u32; }
        if i == 330u32 { v = 0x19082b2bu32; }
        if i == 331u32 { v = 0x19190808u32; }
        if i == 332u32 { v = 0x19190808u32; }
        if i == 333u32 { v = 0x19190808u32; }
        if i == 334u32 { v = 0x19190808u32; }
        if i == 335u32 { v = 0x19190808u32; }
        if i == 336u32 { v = 0x19190808u32; }
        if i == 337u32 { v = 0x19190808u32; }
        if i == 338u32 { v = 0x19190808u32; }
        if i == 339u32 { v = 0x19190808u32; }
        if i == 340u32 { v = 0x19190808u32; }
        if i == 341u32 { v = 0x19190808u32; }
        if i == 342u32 { v = 0x19190808u32; }
        if i == 343u32 { v = 0x19190819u32; }
        if i == 344u32 { v = 0x19190819u32; }
        if i == 345u32 { v = 0x19190819u32; }
        if i == 346u32 { v = 0x19190819u32; }
        if i == 347u32 { v = 0x19190819u32; }
        if i == 348u32 { v = 0x19190819u32; }
        if i == 349u32 { v = 0x1919082bu32; }
        if i == 350u32 { v = 0x1919082bu32; }
        if i == 351u32 { v = 0x1919082bu32; }
        if i == 352u32 { v = 0x19191908u32; }
        if i == 353u32 { v = 0x19191908u32; }
        if i == 354u32 { v = 0x19191908u32; }
        if i == 355u32 { v = 0x19191908u32; }
        if i == 356u32 { v = 0x19191908u32; }
        if i == 357u32 { v = 0x19191908u32; }
        if i == 358u32 { v = 0x19191908u32; }
        if i == 359u32 { v = 0x19191908u32; }
        if i == 360u32 { v = 0x19191919u32; }
        if i == 361u32 { v = 0x19191919u32; }
        if i == 362u32 { v = 0x19191919u32; }
        if i == 363u32 { v = 0x19191919u32; }
        if i == 364u32 { v = 0x1919192bu32; }
        if i == 365u32 { v = 0x1919192bu32; }
        if i == 366u32 { v = 0x1919192bu32; }
        if i == 367u32 { v = 0x19192b08u32; }
        if i == 368u32 { v = 0x19192b08u32; }
        if i == 369u32 { v = 0x19192b08u32; }
        if i == 370u32 { v = 0x19192b08u32; }
        if i == 371u32 { v = 0x19192b08u32; }
        if i == 372u32 { v = 0x19192b19u32; }
        if i == 373u32 { v = 0x19192b19u32; }
        if i == 374u32 { v = 0x19192b19u32; }
        if i == 375u32 { v = 0x19192b2bu32; }
        if i == 376u32 { v = 0x192b0808u32; }
        if i == 377u32 { v = 0x192b0808u32; }
        if i == 378u32 { v = 0x192b0808u32; }
        if i == 379u32 { v = 0x192b0808u32; }
        if i == 380u32 { v = 0x192b0808u32; }
        if i == 381u32 { v = 0x192b0808u32; }
        if i == 382u32 { v = 0x192b0808u32; }
        if i == 383u32 { v = 0x192b0808u32; }
        if i == 384u32 { v = 0x192b0819u32; }
        if i == 385u32 { v = 0x192b082bu32; }
        if i == 386u32 { v = 0x192b082bu32; }
        if i == 387u32 { v = 0x192b082bu32; }
        if i == 388u32 { v = 0x192b1908u32; }
        if i == 389u32 { v = 0x192b1908u32; }
        if i == 390u32 { v = 0x192b1919u32; }
        if i == 391u32 { v = 0x192b1919u32; }
        if i == 392u32 { v = 0x192b1919u32; }
        if i == 393u32 { v = 0x192b1919u32; }
        if i == 394u32 { v = 0x192b2b08u32; }
        if i == 395u32 { v = 0x192b2b08u32; }
        if i == 396u32 { v = 0x192b2b19u32; }
        if i == 397u32 { v = 0x192b2b2bu32; }
        if i == 398u32 { v = 0x192b2b2bu32; }
        if i == 399u32 { v = 0x192b2b2bu32; }
        if i == 400u32 { v = 0x2b080808u32; }
        if i == 401u32 { v = 0x2b080808u32; }
        if i == 402u32 { v = 0x2b080808u32; }
        if i == 403u32 { v = 0x2b080808u32; }
        if i == 404u32 { v = 0x2b080808u32; }
        if i == 405u32 { v = 0x2b080808u32; }
        if i == 406u32 { v = 0x2b080808u32; }
        if i == 407u32 { v = 0x2b080808u32; }
        if i == 408u32 { v = 0x2b080808u32; }
        if i == 409u32 { v = 0x2b080808u32; }
        if i == 410u32 { v = 0x2b080808u32; }
        if i == 411u32 { v = 0x2b080808u32; }
        if i == 412u32 { v = 0x2b080808u32; }
        if i == 413u32 { v = 0x2b080808u32; }
        if i == 414u32 { v = 0x2b080808u32; }
        if i == 415u32 { v = 0x2b080819u32; }
        if i == 416u32 { v = 0x2b080819u32; }
        if i == 417u32 { v = 0x2b080819u32; }
        if i == 418u32 { v = 0x2b080819u32; }
        if i == 419u32 { v = 0x2b080819u32; }
        if i == 420u32 { v = 0x2b080819u32; }
        if i == 421u32 { v = 0x2b080819u32; }
        if i == 422u32 { v = 0x2b08082bu32; }
        if i == 423u32 { v = 0x2b08082bu32; }
        if i == 424u32 { v = 0x2b08082bu32; }
        if i == 425u32 { v = 0x2b08082bu32; }
        if i == 426u32 { v = 0x2b08082bu32; }
        if i == 427u32 { v = 0x2b08082bu32; }
        if i == 428u32 { v = 0x2b081908u32; }
        if i == 429u32 { v = 0x2b081908u32; }
        if i == 430u32 { v = 0x2b081908u32; }
        if i == 431u32 { v = 0x2b081908u32; }
        if i == 432u32 { v = 0x2b081908u32; }
        if i == 433u32 { v = 0x2b081908u32; }
        if i == 434u32 { v = 0x2b081908u32; }
        if i == 435u32 { v = 0x2b081908u32; }
        if i == 436u32 { v = 0x2b081919u32; }
        if i == 437u32 { v = 0x2b081919u32; }
        if i == 438u32 { v = 0x2b081919u32; }
        if i == 439u32 { v = 0x2b08192bu32; }
        if i == 440u32 { v = 0x2b08192bu32; }
        if i == 441u32 { v = 0x2b082b08u32; }
        if i == 442u32 { v = 0x2b082b08u32; }
        if i == 443u32 { v = 0x2b082b08u32; }
        if i == 444u32 { v = 0x2b082b08u32; }
        if i == 445u32 { v = 0x2b082b08u32; }
        if i == 446u32 { v = 0x2b082b08u32; }
        if i == 447u32 { v = 0x2b082b08u32; }
        if i == 448u32 { v = 0x2b082b19u32; }
        if i == 449u32 { v = 0x2b082b2bu32; }
        if i == 450u32 { v = 0x2b082b2bu32; }
        if i == 451u32 { v = 0x2b082b2bu32; }
        if i == 452u32 { v = 0x2b082b2bu32; }
        if i == 453u32 { v = 0x2b082b2bu32; }
        if i == 454u32 { v = 0x2b190808u32; }
        if i == 455u32 { v = 0x2b190808u32; }
        if i == 456u32 { v = 0x2b190808u32; }
        if i == 457u32 { v = 0x2b190808u32; }
        if i == 458u32 { v = 0x2b190808u32; }
        if i == 459u32 { v = 0x2b190808u32; }
        if i == 460u32 { v = 0x2b190819u32; }
        if i == 461u32 { v = 0x2b190819u32; }
        if i == 462u32 { v = 0x2b190819u32; }
        if i == 463u32 { v = 0x2b19082bu32; }
        if i == 464u32 { v = 0x2b19082bu32; }
        if i == 465u32 { v = 0x2b191908u32; }
        if i == 466u32 { v = 0x2b191908u32; }
        if i == 467u32 { v = 0x2b191908u32; }
        if i == 468u32 { v = 0x2b191908u32; }
        if i == 469u32 { v = 0x2b191908u32; }
        if i == 470u32 { v = 0x2b191919u32; }
        if i == 471u32 { v = 0x2b191919u32; }
        if i == 472u32 { v = 0x2b191919u32; }
        if i == 473u32 { v = 0x2b19192bu32; }
        if i == 474u32 { v = 0x2b192b08u32; }
        if i == 475u32 { v = 0x2b192b08u32; }
        if i == 476u32 { v = 0x2b192b19u32; }
        if i == 477u32 { v = 0x2b192b2bu32; }
        if i == 478u32 { v = 0x2b192b2bu32; }
        if i == 479u32 { v = 0x2b2b0808u32; }
        if i == 480u32 { v = 0x2b2b0808u32; }
        if i == 481u32 { v = 0x2b2b0808u32; }
        if i == 482u32 { v = 0x2b2b0808u32; }
        if i == 483u32 { v = 0x2b2b0808u32; }
        if i == 484u32 { v = 0x2b2b0808u32; }
        if i == 485u32 { v = 0x2b2b0808u32; }
        if i == 486u32 { v = 0x2b2b0819u32; }
        if i == 487u32 { v = 0x2b2b0819u32; }
        if i == 488u32 { v = 0x2b2b0819u32; }
        if i == 489u32 { v = 0x2b2b082bu32; }
        if i == 490u32 { v = 0x2b2b082bu32; }
        if i == 491u32 { v = 0x2b2b082bu32; }
        if i == 492u32 { v = 0x2b2b082bu32; }
        if i == 493u32 { v = 0x2b2b082bu32; }
        if i == 494u32 { v = 0x2b2b082bu32; }
        if i == 495u32 { v = 0x2b2b1908u32; }
        if i == 496u32 { v = 0x2b2b1908u32; }
        if i == 497u32 { v = 0x2b2b192bu32; }
        if i == 498u32 { v = 0x2b2b192bu32; }
        if i == 499u32 { v = 0x2b2b2b08u32; }
        if i == 500u32 { v = 0x2b2b2b08u32; }
        if i == 501u32 { v = 0x2b2b2b08u32; }
        if i == 502u32 { v = 0x2b2b2b08u32; }
        if i == 503u32 { v = 0x2b2b2b08u32; }
        if i == 504u32 { v = 0x2b2b2b08u32; }
        if i == 505u32 { v = 0x2b2b2b19u32; }
        if i == 506u32 { v = 0x2b2b2b19u32; }
        if i == 507u32 { v = 0x2b2b2b19u32; }
        if i == 508u32 { v = 0x2b2b2b2bu32; }
        if i == 509u32 { v = 0x2b2b2b2bu32; }
        if i == 510u32 { v = 0x2b2b2b2bu32; }
        if i == 511u32 { v = 0x2b2b2b2bu32; }
        v
    }

    /// `tmp += vec_dot_<FMT>_q8_1(xb, yb, iqs)` with the reference's contraction of the `+=`.
    /// GROUPED selects moe_grouped.cu's association for q4_1/q5_1 (`sumi * dm.x * ds.x + ...`).
    #[inline(always)]
    pub unsafe fn vdot<const FMT: u32, const GROUPED: bool>(tmp: f32, xb: *const u8, yb: *const u8, iqs: usize) -> f32 {
        match FMT {
            18 => {
                // vec_dot_iq3_xxs_q8_1; kqs = 4*(tid % (qi/vdr)) => q8 block = kby + iqs/vdr.
                let qs = xb.add(2);
                let q3a = get_b4(qs, iqs as i32);
                let q3b = get_b4(qs, iqs as i32 + 1);
                let mut q3 = [0u8; 8];
                q3[0] = q3a as u8; q3[1] = (q3a >> 8) as u8; q3[2] = (q3a >> 16) as u8; q3[3] = (q3a >> 24) as u8;
                q3[4] = q3b as u8; q3[5] = (q3b >> 8) as u8; q3[6] = (q3b >> 16) as u8; q3[7] = (q3b >> 24) as u8;
                let aux32 = get_b4(qs.add(64), iqs as i32 / 2);
                let b8 = yb.add((iqs / 2) * 36);
                let q8 = b8.add(4) as *const u32;
                let mut sumi = 0i32;
                let mut l0 = 0usize;
                while l0 < 8 {
                    let g0 = i3_grid(q3[l0] as u32);
                    let g1 = i3_grid(q3[l0 + 1] as u32);
                    let signs = unpack_ksigns(aux32 >> (7 * l0 as u32 / 2));
                    // unpack_ksigns already broadcasts the byte across the word (s * 0x01010101);
                    // multiplying again here squares the byte and corrupts the sign selectors.
                    let s0 = vcmpne4_zero(signs & 0x0804_0201);
                    let g_l = vsub4_wrap(g0 ^ s0, s0);
                    let s1 = vcmpne4_zero(signs & 0x8040_2010);
                    let g_h = vsub4_wrap(g1 ^ s1, s1);
                    sumi = dp4a(g_l, *q8.add(l0), sumi);
                    sumi = dp4a(g_h, *q8.add(l0 + 1), sumi);
                    l0 += 2;
                }
                let ls = (aux32 >> 28) as i32;
                let sumi = (ls.wrapping_mul(sumi).wrapping_add(sumi / 2)) / 2;
                let d = mulf(h2f32(ld16(xb, 0)), h2f32(ld16(b8, 0)));
                fmaf(d, sumi as f32, tmp)
            }
            20 => {
                // vec_dot_iq4_nl_q8_1 (vecdotq.cuh): one q8_1 block per iq4_nl block (QK 32);
                // q8 = bq8_1->qs + iqs, aux = get_int_b2(bq4->qs, iqs + l), v = table16(aux) against q8[l], q8[l + 4].
                let qs = xb.add(2);
                let q8 = yb.add(4) as *const u32;
                let mut sumi = 0i32;
                let mut l = 0usize;
                while l < 2 {
                    let (v0, v1) = i4_table16(get_b2(qs, (iqs + l) as i32));
                    sumi = dp4a(v0, *q8.add(iqs + l), sumi);
                    sumi = dp4a(v1, *q8.add(iqs + l + 4), sumi);
                    l += 1;
                }
                let d = mulf(h2f32(ld16(xb, 0)), h2f32(ld16(yb, 0)));
                fmaf(d, sumi as f32, tmp)
            }
            23 => {
                // vec_dot_iq4_xs_q8_1; kqs = 4*(tid % (qi/vdr)) => q8 block = kby + iqs/vdr.
                let qs = xb.add(8) as *const u32;
                let b8 = yb.add((iqs / 4) * 36);
                let q8 = b8.add(4) as *const u32;
                let mut sumi = 0i32;
                let mut j = 0usize;
                while j < 4 {
                    let (v0, v1) = i4_table16(get_b4(xb, iqs as i32 + j as i32 + 2));
                    sumi = dp4a(v0, *q8.add(j), sumi);
                    sumi = dp4a(v1, *q8.add(j + 4), sumi);
                    j += 1;
                }
                let ls = i4_group_scale6(xb, iqs / 4) as i32 - 32;
                let sumi = sumi.wrapping_mul(ls);
                let d = mulf(h2f32(ld16(xb, 0)), h2f32(ld16(b8, 0)));
                fmaf(d, sumi as f32, tmp)
            }

            17 => {
                // vec_dot_iq2_xs_q8_1; iqs is 2*(tid % 8) => q8 block iqs/2.
                let qs = xb.add(2);
                let q16 = qs.add(4 * iqs) as *const u16;
                let ls0 = (*xb.add(66 + (iqs / 2)) & 0x0F) as i32;
                let ls1 = (*xb.add(66 + (iqs / 2)) >> 4) as i32;
                let b8 = yb.add((iqs / 2) * 36);
                let q8 = b8.add(4) as *const u32;
                let mut sumi0 = 0i32;
                let mut sumi1 = 0i32;
                let mut l0 = 0usize;
                while l0 < 8 {
                    let w = *q16.add(l0 / 2) as u32;
                    let gl = xs_grid_lo(w & 0x1FF);
                    let gh = xs_grid_hi(w & 0x1FF);
                    let signs = unpack_ksigns(w >> 9);
                    let s0 = vcmpne4_zero(signs & 0x0804_0201);
                    let s1 = vcmpne4_zero(signs & 0x8040_2010);
                    let g0 = vsub4_wrap(gl ^ s0, s0);
                    let g1 = vsub4_wrap(gh ^ s1, s1);
                    if l0 < 4 {
                        sumi0 = dp4a(g0, *q8.add(l0), sumi0);
                        sumi0 = dp4a(g1, *q8.add(l0 + 1), sumi0);
                    } else {
                        sumi1 = dp4a(g0, *q8.add(l0), sumi1);
                        sumi1 = dp4a(g1, *q8.add(l0 + 1), sumi1);
                    }
                    l0 += 2;
                }
                let sumi = sumi0.wrapping_mul(ls0).wrapping_add(sumi1.wrapping_mul(ls1)).wrapping_add((sumi0 + sumi1) / 2) / 4;
                let d = mulf(h2f32(ld16(xb, 0)), h2f32(ld16(b8, 0)));
                fmaf(d, sumi as f32, tmp)
            }
            22 => {
                // vec_dot_iq2_s_q8_1; iqs is 2*(tid % 8) => q8 block iqs/2.
                let i2 = iqs as i32 / 2;
                let qs = xb.add(2);
                let qp = get_b2(qs, i2);
                let sp = get_b2(qs.add(32), i2);
                let mut q8b = [0u8; 4];
                let mut s8 = [0u8; 4];
                let mut t = 0usize;
                while t < 4 { q8b[t] = (qp >> (8 * t)) as u8; s8[t] = (sp >> (8 * t)) as u8; t += 1; }
                let qh = *xb.add(66 + i2 as usize) as u32;
                let ls0 = (*xb.add(74 + i2 as usize) & 0x0F) as i32;
                let ls1 = (*xb.add(74 + i2 as usize) >> 4) as i32;
                let b8 = yb.add((iqs / 2) * 36);
                let q8 = b8.add(4) as *const u32;
                let mut sumi0 = 0i32;
                let mut sumi1 = 0i32;
                let mut l0 = 0usize;
                while l0 < 8 {
                    let gidx = q8b[l0 / 2] as u32 | ((qh << (8 - l0 as u32)) & 0x300);
                    let gl = s_grid_lo(gidx);
                    let gh = s_grid_hi(gidx);
                    let sb = s8[l0 / 2] as u32;
                    let s0 = vcmpne4_zero(((sb & 0x03) << 7) | ((sb & 0x0C) << 21));
                    let s1 = vcmpne4_zero(((sb & 0x30) << 3) | ((sb & 0xC0) << 17));
                    let g_l = vsub4_wrap(gl ^ s0, s0);
                    let g_h = vsub4_wrap(gh ^ s1, s1);
                    if l0 < 4 {
                        sumi0 = dp4a(g_l, *q8.add(l0), sumi0);
                        sumi0 = dp4a(g_h, *q8.add(l0 + 1), sumi0);
                    } else {
                        sumi1 = dp4a(g_l, *q8.add(l0), sumi1);
                        sumi1 = dp4a(g_h, *q8.add(l0 + 1), sumi1);
                    }
                    l0 += 2;
                }
                let sumi = (sumi0.wrapping_mul(ls0).wrapping_add(sumi1.wrapping_mul(ls1)).wrapping_add((sumi0 + sumi1) / 2)) / 4;
                let d = mulf(h2f32(ld16(xb, 0)), h2f32(ld16(b8, 0)));
                fmaf(d, sumi as f32, tmp)
            }

            11 => {
                // vec_dot_iq2_xxs_q8_1. iqs is already 2*(tid % 8) => q8 word base iqs/2.
                // block_iq2_xxs starts with `half d`, so qs sits at byte 2: the reference
                // reads get_int_b2(bq2->qs, ...) = byte 2 + 4*k, not 4*k.
                let qs2 = xb.add(2);
                let q2 = get_b2(qs2, iqs as i32);
                let aux8 = q2.to_le_bytes();
                let aux32 = get_b2(qs2, iqs as i32 + 1);
                let mut sumi = 0i32;
                let mut k0 = 0usize;
                while k0 < 8 {
                    let idx = aux8[k0 / 2] as usize;
                    let gl = grid_lo(idx as u32);
                    let gh = grid_hi(idx as u32);
                    let signs = unpack_ksigns(aux32 >> (7 * k0 as u32 / 2));
                    let s0 = vcmpne4_zero(signs & 0x0804_0201);
                    let s1 = vcmpne4_zero(signs & 0x8040_2010);
                    let g0 = vsub4_wrap(gl ^ s0, s0);
                    let g1 = vsub4_wrap(gh ^ s1, s1);
                    let b8 = yb.add((iqs / 2) * 36);
                    sumi = dp4a(g0, yq(b8, k0), sumi);
                    sumi = dp4a(g1, yq(b8, k0 + 1), sumi);
                    k0 += 2;
                }
                let ls = (aux32 >> 27 | 1) as i32;
                let sumi2 = sumi.wrapping_mul(ls) / 8;
                let b8 = yb.add((iqs / 2) * 36);
                let d = mulf(h2f32(ld16(xb, 0)), h2f32(ld16(b8, 0)));
                fmaf(d, sumi2 as f32, tmp)
            }
            0 | 1 | 2 | 3 => {
                let mut sumi = 0i32;
                let mut i = 0;
                while i < 2 {
                    let u0 = yq(yb, iqs + i);
                    let u1 = yq(yb, iqs + i + 4);
                    let (vi0, vi1) = if FMT == 0 || FMT == 1 {
                        let v = if FMT == 0 { ld32u(xb, 2 + 4 * (iqs + i)) } else { ld32a(xb, 4 + 4 * (iqs + i)) };
                        (v & 0x0F0F0F0F, sar(v, 4) & 0x0F0F0F0F)
                    } else {
                        let (vl, qh) = if FMT == 2 {
                            (ld32u(xb, 6 + 4 * (iqs + i)), ld32u(xb, 2))
                        } else {
                            (ld32a(xb, 8 + 4 * (iqs + i)), ld32a(xb, 4))
                        };
                        let vh = sar(qh, 4 * (iqs + i));
                        let mut vi0 = vl & 0x0F0F0F0F;
                        vi0 |= (vh << 4) & 0x00000010;
                        vi0 |= (vh << 11) & 0x00001000;
                        vi0 |= (vh << 18) & 0x00100000;
                        vi0 |= (vh << 25) & 0x10000000;
                        let mut vi1 = sar(vl, 4) & 0x0F0F0F0F;
                        vi1 |= sar(vh, 12) & 0x00000010;
                        vi1 |= sar(vh, 5) & 0x00001000;
                        vi1 |= (vh << 2) & 0x00100000;
                        vi1 |= (vh << 9) & 0x10000000;
                        (vi0, vi1)
                    };
                    sumi = dp4a(vi0, u0, sumi);
                    sumi = dp4a(vi1, u1, sumi);
                    i += 1;
                }
                let ds = ld32a(yb, 0);
                let dsx = h2f32(ds);
                let dsy = h2f32(ds >> 16);
                let sf = sumi as f32;
                if FMT == 0 || FMT == 2 {
                    let c = if FMT == 0 { -4.0 } else { -8.0 };
                    fmaf(h2f32(ld16(xb, 0)), fmaf(dsx, sf, mulf(dsy, c)), tmp)
                } else {
                    let dm = ld32a(xb, 0);
                    let ms = mulf(mulf(h2f32(dm >> 16), dsy), 0.5);
                    if GROUPED {
                        addf(fmaf(mulf(sf, h2f32(dm)), dsx, ms), tmp)
                    } else {
                        let dd = mulf(h2f32(dm), dsx);
                        addf(fmaf(dd, sf, ms), tmp)
                    }
                }
            }
            4 => {
                let mut sumi = 0i32;
                let mut i = 0;
                while i < 2 {
                    sumi = dp4a(ld32u(xb, 2 + 4 * (iqs + i)), yq(yb, iqs + i), sumi);
                    i += 1;
                }
                fmaf(mulf(sumi as f32, h2f32(ld16(xb, 0))), h2f32(ld16(yb, 0)), tmp)
            }
            5 => {
                let bq8_offset = 4 * (iqs / 8);
                let scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
                let v = ld32a(xb, 16 + 4 * iqs);
                let mut sumf_d = 0f32;
                let mut sumf_m = 0f32;
                let mut i = 0;
                while i < 4 {
                    let b8 = yb.add((bq8_offset + i) * 36);
                    let u = yq(b8, iqs % 8);
                    let d8 = h2f32(ld16(b8, 0));
                    let sc = ld8(xb, scale_offset + 2 * i);
                    let vi = (v >> (2 * i)) & 0x03030303;
                    sumf_d = fmaf(d8, dp4a(vi, u, 0).wrapping_mul((sc & 0xF) as i32) as f32, sumf_d);
                    let mut m = sc >> 4;
                    m |= m << 8;
                    m |= m << 16;
                    sumf_m = fmaf(d8, dp4a(m, u, 0) as f32, sumf_m);
                    i += 1;
                }
                let dm = ld32a(xb, 80);
                addf(fmaf(sumf_d, h2f32(dm), -mulf(sumf_m, h2f32(dm >> 16))), tmp)
            }
            6 => {
                let bq8_offset = 4 * (iqs / 8);
                let scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
                let vl = ld32u(xb, 32 + 4 * iqs);
                let vh = sar(!ld32u(xb, 4 * (iqs % 8)), bq8_offset);
                let mut sumf = 0f32;
                let mut i = 0;
                while i < 4 {
                    let b8 = yb.add((bq8_offset + i) * 36);
                    let u = yq(b8, iqs % 8);
                    let d8 = h2f32(ld16(b8, 0));
                    let isc = scale_offset + 2 * i;
                    let sc_low = (ld8(xb, 96 + isc % 8) >> (4 * (isc / 8))) & 0xF;
                    let sc_high = ((ld8(xb, 96 + 8 + isc % 4) >> (2 * (isc / 4))) & 3) << 4;
                    let sc = (sc_low | sc_high) as i32 - 32;
                    let vil = (vl >> (2 * i)) & 0x03030303;
                    let vih = (sar(vh, i) << 2) & 0x04040404;
                    let vi = vsubss4(vil, vih);
                    sumf = fmaf(d8, dp4a(vi, u, 0).wrapping_mul(sc) as f32, sumf);
                    i += 1;
                }
                fmaf(h2f32(ld16(xb, 108)), sumf, tmp)
            }
            7 | 8 => {
                let bq8_offset = 2 * ((iqs / 2) / 4);
                let (v0, v1) = if FMT == 7 {
                    let q4 = 16 + 16 * bq8_offset + 4 * ((iqs / 2) % 4);
                    let (a, b) = (ld32a(xb, q4), ld32a(xb, q4 + 16));
                    ([a & 0x0F0F0F0F, b & 0x0F0F0F0F], [sar(a, 4) & 0x0F0F0F0F, sar(b, 4) & 0x0F0F0F0F])
                } else {
                    let ql = 48 + 16 * bq8_offset + 4 * ((iqs / 2) % 4);
                    let qh = 16 + 4 * ((iqs / 2) % 4);
                    let (l0, l1) = (ld32a(xb, ql), ld32a(xb, ql + 16));
                    let h0 = sar(ld32a(xb, qh), bq8_offset);
                    let h1 = sar(ld32a(xb, qh + 16), bq8_offset);
                    (
                        [(l0 & 0x0F0F0F0F) | ((h0 << 4) & 0x10101010), (l1 & 0x0F0F0F0F) | ((h1 << 4) & 0x10101010)],
                        [
                            (sar(l0, 4) & 0x0F0F0F0F) | ((sar(h0, 1) << 4) & 0x10101010),
                            (sar(l1, 4) & 0x0F0F0F0F) | ((sar(h1, 1) << 4) & 0x10101010),
                        ],
                    )
                };
                let v = [v0, v1];
                let s16 = |k: usize| ld16(xb, 4 + 2 * k);
                let j = bq8_offset / 2;
                let (aux0, aux1) = if j < 2 {
                    (s16(j) & 0x3f3f, s16(j + 2) & 0x3f3f)
                } else {
                    (
                        (s16(j + 2) & 0x0f0f) | ((s16(j - 2) & 0xc0c0) >> 2),
                        ((s16(j + 2) >> 4) & 0x0f0f) | ((s16(j) & 0xc0c0) >> 2),
                    )
                };
                let sc = [aux0 & 0xFF, aux0 >> 8];
                let m = [aux1 & 0xFF, aux1 >> 8];
                let mut sumf_d = 0f32;
                let mut sumf_m = 0f32;
                let mut i = 0;
                while i < 2 {
                    let b8 = yb.add((bq8_offset + i) * 36);
                    let d8 = h2f32(ld16(b8, 0));
                    let u0 = yq(b8, (iqs / 2) % 4);
                    let u1 = yq(b8, (iqs / 2) % 4 + 4);
                    let dot1 = dp4a(v[i][1], u1, dp4a(v[i][0], u0, 0));
                    let dot2 = dp4a(0x01010101, u1, dp4a(0x01010101, u0, 0));
                    sumf_d = fmaf(d8, dot1.wrapping_mul(sc[i] as i32) as f32, sumf_d);
                    sumf_m = fmaf(d8, dot2.wrapping_mul(m[i] as i32) as f32, sumf_m);
                    i += 1;
                }
                let dm = ld32a(xb, 0);
                addf(fmaf(sumf_d, h2f32(dm), -mulf(sumf_m, h2f32(dm >> 16))), tmp)
            }
            9 => {
                let bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
                let scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
                let vh_shift = 2 * ((iqs % 16) / 8);
                let vl = ld32u(xb, 4 * iqs);
                let vh = sar(ld32u(xb, 128 + 4 * (8 * (iqs / 16) + iqs % 8)), vh_shift);
                let mut sumf = 0f32;
                let mut i = 0;
                while i < 2 {
                    let b8 = yb.add((bq8_offset + 2 * i) * 36);
                    let u = yq(b8, iqs % 8);
                    let d8 = h2f32(ld16(b8, 0));
                    let sc = *xb.add(192 + scale_offset + 4 * i) as i8 as i32;
                    let vil = sar(vl, 4 * i) & 0x0F0F0F0F;
                    let vih = (sar(vh, 4 * i) << 4) & 0x30303030;
                    let vi = vsubss4(vil | vih, 0x20202020);
                    sumf = fmaf(d8, dp4a(vi, u, 0).wrapping_mul(sc) as f32, sumf);
                    i += 1;
                }
                fmaf(h2f32(ld16(xb, 208)), sumf, tmp)
            }
            _ => {
                // q8_1 weights x q8_1 activations
                let mut sumi = 0i32;
                let mut i = 0;
                while i < 2 {
                    sumi = dp4a(ld32a(xb, 4 + 4 * (iqs + i)), yq(yb, iqs + i), sumi);
                    i += 1;
                }
                let (dw, dv) = (ld32a(xb, 0), ld32a(yb, 0));
                let p = mulf(mulf(h2f32(dw), h2f32(dv)), sumi as f32);
                addf(fmaf(h2f32(dw >> 16), h2f32(dv >> 16), p), tmp)
            }
        }
    }

    // ------------------------------------------------------------------------------------------
    // indexed_moe/indexed_moe.cu

    /// `abs.ftz.f32` [FADD.FTZ |x|, -RZ].
    #[inline(always)]
    pub fn abs_ftz(a: f32) -> f32 {
        let r: f32;
        unsafe { cuda_device::ptx_asm!("abs.ftz.f32 %0, %1;", out("=f") r, in("f") a, options(register_only)); }
        r
    }
    /// `add.rz.ftz.f32`.
    #[inline(always)]
    pub fn add_rz_ftz(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { cuda_device::ptx_asm!("add.rz.ftz.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    /// `cvt.rzi.s32.f32`.
    #[inline(always)]
    pub fn f2i_rz(a: f32) -> i32 {
        let r: i32;
        unsafe { cuda_device::ptx_asm!("cvt.rzi.s32.f32 %0, %1;", out("=r") r, in("f") a, options(register_only)); }
        r
    }

    /// quantize_q8_1{,_bf16,_f16}: per 32 values a warp max of |x| and a warp sum (xor butterflies,
    /// .ftz), d = amax * (1/127) [FMUL], q = roundf(x * rcp(d)) [copysign(0.5) + add.rz + cvt.rzi].
    #[inline(always)]
    unsafe fn quantize_q8_1<T: Val>(x: *const T, vy: *mut u8, kx: i32, kx_padded: i32) {
        let ix = thread::blockDim_x().wrapping_mul(thread::blockIdx_x()).wrapping_add(thread::threadIdx_x()) as i32;
        if ix >= kx_padded {
            return;
        }
        let iy = thread::blockDim_y().wrapping_mul(thread::blockIdx_y()).wrapping_add(thread::threadIdx_y()) as i32;
        let i_padded = iy.wrapping_mul(kx_padded).wrapping_add(ix);
        let ib = i_padded / 32;
        let iqs = i_padded % 32;
        let xi = if ix < kx { (*x.offset(iy.wrapping_mul(kx).wrapping_add(ix) as isize)).f() } else { 0.0 };
        let mut amax = abs_ftz(xi);
        let mut sum = xi;
        let mut m = 16;
        while m > 0 {
            amax = fmaxf(amax, warp::shuffle_xor_f32_sync(0xffff_ffff, amax, m));
            m >>= 1;
        }
        let mut m = 16;
        while m > 0 {
            sum = addf(sum, warp::shuffle_xor_f32_sync(0xffff_ffff, sum, m));
            m >>= 1;
        }
        let d = mulf(amax, cf(0x3c010204));
        let q: i32 = if amax != 0.0 {
            let t = mulf(rcp(d), xi);
            f2i_rz(add_rz_ftz(t, copysignf(0.5, t)))
        } else {
            0
        };
        let yb = vy.wrapping_offset(ib as isize * 36);
        *yb.offset(4 + iqs as isize) = q as u8;
        if iqs > 0 {
            return;
        }
        *(yb as *mut u16) = f2h(d);
        *(yb.add(2) as *mut u16) = f2h(sum);
    }
    #[kernel] pub unsafe fn quantize_q8_1(x: *const f32, vy: *mut u8, kx: i32, kp: i32) { quantize_q8_1::<f32>(x, vy, kx, kp) }
    #[kernel] pub unsafe fn quantize_q8_1_bf16(x: *const B, vy: *mut u8, kx: i32, kp: i32) { quantize_q8_1::<B>(x, vy, kx, kp) }
    #[kernel] pub unsafe fn quantize_q8_1_f16(x: *const H, vy: *mut u8, kx: i32, kp: i32) { quantize_q8_1::<H>(x, vy, kx, kp) }

    /// indexed_moe_forward: grid (n, batch, topk), block (32, 4); one output row per CUDA block.
    #[inline(always)]
    unsafe fn indexed_moe_forward<const FMT: u32>(
        all_w: *const u8, all_x: *const u8, indices: *const u32, all_out: *mut f32, n: i32, k: i32, _batch: i32, _topk: i32,
        k_padded: i32, input_dim1: i32,
    ) {
        static mut TMP: SharedArray<f32, 96> = SharedArray::UNINIT;
        let tmp_sh = SharedArray::as_raw_mut_ptr(&raw mut TMP);
        let (qk, qi, vdr, bs) = fmt_params::<FMT>();
        let cur_batch = thread::blockIdx_y() as i32;
        let cur_topk = thread::blockIdx_z() as i32;
        let task_id = cur_batch.wrapping_mul(thread::gridDim_z() as i32).wrapping_add(cur_topk);
        if task_id >= (thread::gridDim_y() as i32).wrapping_mul(thread::gridDim_z() as i32) {
            return;
        }
        let input_idx = if input_dim1 == 1 { cur_batch } else { task_id };
        let expert_id = *indices.offset(task_id as isize);
        let wbpr = (k as i64 as u64).wrapping_add(qk as u64 - 1) / qk as u64;
        let w_stride = (n as i64 as u64).wrapping_mul(wbpr).wrapping_mul(bs as u64);
        let x_stride = (k_padded as i64 as u64) / 32 * 36;
        let x = all_x.wrapping_add((input_idx as i64 as u64).wrapping_mul(x_stride) as usize);
        let w = all_w.wrapping_add((expert_id as u64).wrapping_mul(w_stride) as usize);
        let out = all_out.wrapping_add((task_id as i64 as u64).wrapping_mul(n as i64 as u64) as usize);
        let tx = thread::threadIdx_x() as i32;
        let ty = thread::threadIdx_y() as i32;
        let tid = 32 * ty + tx;
        let row0 = thread::blockIdx_x() as i32;
        if row0 >= n {
            return;
        }
        let bpr_x = k.wrapping_add(qk - 1) / qk;
        let bpi = vdr * 4 * 32 / qi;
        let mut tmp = 0.0f32;
        let mut kbx = tid / (qi / vdr);
        while kbx < bpr_x {
            let kby = kbx * (qk / 32);
            let kqs = (vdr * (tid % (qi / vdr))) as usize;
            let wb = w.wrapping_offset(kbx.wrapping_add(row0.wrapping_mul(bpr_x)) as isize * bs as isize);
            tmp = vdot::<FMT, false>(tmp, wb, x.wrapping_offset(kby as isize * 36), kqs);
            kbx += bpi;
        }
        if ty > 0 {
            *tmp_sh.add(((ty - 1) * 32 + tx) as usize) = tmp;
        }
        thread::sync_threads();
        if ty == 0 {
            let mut l = 0;
            while l < 3 {
                tmp = addf(tmp, *tmp_sh.add((l * 32 + tx) as usize));
                l += 1;
            }
            tmp = warp_xor_sum(tmp);
            if tx == 0 {
                *out.offset(row0 as isize) = tmp;
            }
        }
    }

    /// moe_gemv_fused_gate_up: grid (ceil(n / 8), topk, batch), block (32, 4); each warp 2 rows.
    #[inline(always)]
    unsafe fn fused_gate_up<const FMT: u32>(
        gate_w: *const u8, up_w: *const u8, all_x: *const u8, indices: *const u32, all_out: *mut f32, n: i32, k: i32, _batch: i32,
        topk: i32, k_padded: i32, act_type: i32,
    ) {
        let (qk, qi, vdr, bs) = fmt_params::<FMT>();
        let warp_id = thread::threadIdx_y() as i32;
        let tx = thread::threadIdx_x() as i32;
        let row0 = (8 * thread::blockIdx_x() as i32).wrapping_add(warp_id * 2);
        let cur_topk = thread::blockIdx_y() as i32;
        let cur_batch = thread::blockIdx_z() as i32;
        if row0 >= n {
            return;
        }
        let task_id = cur_batch.wrapping_mul(topk).wrapping_add(cur_topk);
        let expert_id = *indices.offset(task_id as isize);
        let bpr = (k as i64 as u64).wrapping_add(qk as u64 - 1) / qk as u64;
        let w_stride = (n as i64 as u64).wrapping_mul(bpr).wrapping_mul(bs as u64);
        let x_stride = (k_padded as i64 as u64) / 32 * 36;
        let x = all_x.wrapping_add((cur_batch as i64 as u64).wrapping_mul(x_stride) as usize);
        let gw = gate_w.wrapping_add((expert_id as u64).wrapping_mul(w_stride) as usize);
        let uw = up_w.wrapping_add((expert_id as u64).wrapping_mul(w_stride) as usize);
        let bpi = vdr * 32 / qi;
        let bpr_x = bpr as i32;
        let out = all_out.wrapping_add((task_id as i64 as u64).wrapping_mul(n as i64 as u64) as usize);
        let mut r = 0;
        while r < 2 && row0 + r < n {
            let row = row0 + r;
            let mut g_sum = 0.0f32;
            let mut u_sum = 0.0f32;
            let mut kbx = tx / (qi / vdr);
            while kbx < bpr_x {
                let kby = kbx * (qk / 32);
                let kqs = (vdr * (tx % (qi / vdr))) as usize;
                let off = (kbx as i64 as u64).wrapping_add((row as i64 as u64).wrapping_mul(bpr)).wrapping_mul(bs as u64) as usize;
                let xb = x.wrapping_offset(kby as isize * 36);
                g_sum = vdot::<FMT, false>(g_sum, gw.wrapping_add(off), xb, kqs);
                u_sum = vdot::<FMT, false>(u_sum, uw.wrapping_add(off), xb, kqs);
                kbx += bpi;
            }
            g_sum = warp_xor_sum(g_sum);
            u_sum = warp_xor_sum(u_sum);
            if tx == 0 {
                let activated = if act_type == 0 { gelu_tanh(g_sum) } else { silu(g_sum) };
                *out.offset(row as isize) = mulf(u_sum, activated);
            }
            r += 1;
        }
    }

    /// moe_gemv_down_aggregate: grid (ceil(n / 4), 1, batch), block (32, 4); each warp one row and every
    /// top-k slot of its token in slot order: out[batch, row] = ((out + p_0) + p_1) + ... with
    /// p_s = dot_s * topk_weight_s and .ftz adds, i.e. the sum the reference's float atomics
    /// (red.add.f32 is .ftz) form when they arrive in slot order. The atomics' arrival order was the
    /// hardware's, so the reference's result varied run to run; this one is fixed (redcell).
    #[inline(always)]
    unsafe fn down_aggregate<const FMT: u32>(
        all_w: *const u8, all_x: *const u8, indices: *const u32, topk_weights: *const f32, all_out: *mut f32, n: i32, k: i32,
        _batch: i32, topk: i32, k_padded: i32,
    ) {
        let (qk, qi, vdr, bs) = fmt_params::<FMT>();
        let warp_id = thread::threadIdx_y() as i32;
        let tx = thread::threadIdx_x() as i32;
        let row = (4 * thread::blockIdx_x() as i32).wrapping_add(warp_id);
        let cur_batch = thread::blockIdx_z() as i32;
        if row >= n {
            return;
        }
        let bpr = (k as i64 as u64).wrapping_add(qk as u64 - 1) / qk as u64;
        let w_stride = (n as i64 as u64).wrapping_mul(bpr).wrapping_mul(bs as u64);
        let x_stride = (k_padded as i64 as u64) / 32 * 36;
        let bpi = vdr * 32 / qi;
        let bpr_x = bpr as i32;
        let out = all_out.wrapping_add((cur_batch as i64 as u64).wrapping_mul(n as i64 as u64) as usize);
        let mut acc = *out.offset(row as isize);
        let mut slot = 0;
        while slot < topk {
            let task_id = cur_batch.wrapping_mul(topk).wrapping_add(slot);
            let expert_id = *indices.offset(task_id as isize);
            let tw = *topk_weights.offset(task_id as isize);
            let x = all_x.wrapping_add((task_id as i64 as u64).wrapping_mul(x_stride) as usize);
            let w = all_w.wrapping_add((expert_id as u64).wrapping_mul(w_stride) as usize);
            let mut tmp = 0.0f32;
            let mut kbx = tx / (qi / vdr);
            while kbx < bpr_x {
                let kby = kbx * (qk / 32);
                let kqs = (vdr * (tx % (qi / vdr))) as usize;
                let off = (kbx as i64 as u64).wrapping_add((row as i64 as u64).wrapping_mul(bpr)).wrapping_mul(bs as u64) as usize;
                tmp = vdot::<FMT, false>(tmp, w.wrapping_add(off), x.wrapping_offset(kby as isize * 36), kqs);
                kbx += bpi;
            }
            tmp = warp_xor_sum(tmp);
            acc = addf(acc, mulf(tmp, tw));
            slot += 1;
        }
        if tx == 0 {
            *out.offset(row as isize) = acc;
        }
    }
    // GENERATED MOE KERNELS BEGIN
    #[kernel] pub unsafe fn indexed_moe_forward_q4_0_q8_1(w: *const u8, x: *const u8, ids: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32, d1: i32) { indexed_moe_forward::<0>(w, x, ids, out, n, k, batch, topk, kp, d1) }
    #[kernel] pub unsafe fn moe_gemv_fused_gate_up_q4_0_q8_1(gw: *const u8, uw: *const u8, x: *const u8, ids: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32, act: i32) { fused_gate_up::<0>(gw, uw, x, ids, out, n, k, batch, topk, kp, act) }
    #[kernel] pub unsafe fn moe_gemv_down_aggregate_q4_0_q8_1(w: *const u8, x: *const u8, ids: *const u32, tw: *const f32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32) { down_aggregate::<0>(w, x, ids, tw, out, n, k, batch, topk, kp) }
    #[kernel] pub unsafe fn indexed_moe_forward_q4_1_q8_1(w: *const u8, x: *const u8, ids: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32, d1: i32) { indexed_moe_forward::<1>(w, x, ids, out, n, k, batch, topk, kp, d1) }
    #[kernel] pub unsafe fn moe_gemv_fused_gate_up_q4_1_q8_1(gw: *const u8, uw: *const u8, x: *const u8, ids: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32, act: i32) { fused_gate_up::<1>(gw, uw, x, ids, out, n, k, batch, topk, kp, act) }
    #[kernel] pub unsafe fn moe_gemv_down_aggregate_q4_1_q8_1(w: *const u8, x: *const u8, ids: *const u32, tw: *const f32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32) { down_aggregate::<1>(w, x, ids, tw, out, n, k, batch, topk, kp) }
    #[kernel] pub unsafe fn indexed_moe_forward_q5_0_q8_1(w: *const u8, x: *const u8, ids: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32, d1: i32) { indexed_moe_forward::<2>(w, x, ids, out, n, k, batch, topk, kp, d1) }
    #[kernel] pub unsafe fn moe_gemv_fused_gate_up_q5_0_q8_1(gw: *const u8, uw: *const u8, x: *const u8, ids: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32, act: i32) { fused_gate_up::<2>(gw, uw, x, ids, out, n, k, batch, topk, kp, act) }
    #[kernel] pub unsafe fn moe_gemv_down_aggregate_q5_0_q8_1(w: *const u8, x: *const u8, ids: *const u32, tw: *const f32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32) { down_aggregate::<2>(w, x, ids, tw, out, n, k, batch, topk, kp) }
    #[kernel] pub unsafe fn indexed_moe_forward_q5_1_q8_1(w: *const u8, x: *const u8, ids: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32, d1: i32) { indexed_moe_forward::<3>(w, x, ids, out, n, k, batch, topk, kp, d1) }
    #[kernel] pub unsafe fn moe_gemv_fused_gate_up_q5_1_q8_1(gw: *const u8, uw: *const u8, x: *const u8, ids: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32, act: i32) { fused_gate_up::<3>(gw, uw, x, ids, out, n, k, batch, topk, kp, act) }
    #[kernel] pub unsafe fn moe_gemv_down_aggregate_q5_1_q8_1(w: *const u8, x: *const u8, ids: *const u32, tw: *const f32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32) { down_aggregate::<3>(w, x, ids, tw, out, n, k, batch, topk, kp) }
    #[kernel] pub unsafe fn indexed_moe_forward_q8_0_q8_1(w: *const u8, x: *const u8, ids: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32, d1: i32) { indexed_moe_forward::<4>(w, x, ids, out, n, k, batch, topk, kp, d1) }
    #[kernel] pub unsafe fn moe_gemv_fused_gate_up_q8_0_q8_1(gw: *const u8, uw: *const u8, x: *const u8, ids: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32, act: i32) { fused_gate_up::<4>(gw, uw, x, ids, out, n, k, batch, topk, kp, act) }
    #[kernel] pub unsafe fn moe_gemv_down_aggregate_q8_0_q8_1(w: *const u8, x: *const u8, ids: *const u32, tw: *const f32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32) { down_aggregate::<4>(w, x, ids, tw, out, n, k, batch, topk, kp) }
    #[kernel] pub unsafe fn indexed_moe_forward_iq3_xxs_q8_1(w: *const u8, x: *const u8, ids: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32, d1: i32) { indexed_moe_forward::<18>(w, x, ids, out, n, k, batch, topk, kp, d1) }
    #[kernel] pub unsafe fn indexed_moe_forward_iq4_xs_q8_1(w: *const u8, x: *const u8, ids: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32, d1: i32) { indexed_moe_forward::<23>(w, x, ids, out, n, k, batch, topk, kp, d1) }
    #[kernel] pub unsafe fn indexed_moe_forward_iq4_nl_q8_1(w: *const u8, x: *const u8, ids: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32, d1: i32) { indexed_moe_forward::<20>(w, x, ids, out, n, k, batch, topk, kp, d1) }
    #[kernel] pub unsafe fn indexed_moe_forward_iq2_xs_q8_1(w: *const u8, x: *const u8, ids: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32, d1: i32) { indexed_moe_forward::<17>(w, x, ids, out, n, k, batch, topk, kp, d1) }
    #[kernel] pub unsafe fn indexed_moe_forward_iq2_s_q8_1(w: *const u8, x: *const u8, ids: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32, d1: i32) { indexed_moe_forward::<22>(w, x, ids, out, n, k, batch, topk, kp, d1) }
    #[kernel] pub unsafe fn indexed_moe_forward_q2k_q8_1(w: *const u8, x: *const u8, ids: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32, d1: i32) { indexed_moe_forward::<5>(w, x, ids, out, n, k, batch, topk, kp, d1) }
    #[kernel] pub unsafe fn moe_gemv_fused_gate_up_q2k_q8_1(gw: *const u8, uw: *const u8, x: *const u8, ids: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32, act: i32) { fused_gate_up::<5>(gw, uw, x, ids, out, n, k, batch, topk, kp, act) }
    #[kernel] pub unsafe fn moe_gemv_down_aggregate_q2k_q8_1(w: *const u8, x: *const u8, ids: *const u32, tw: *const f32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32) { down_aggregate::<5>(w, x, ids, tw, out, n, k, batch, topk, kp) }
    #[kernel] pub unsafe fn indexed_moe_forward_q3k_q8_1(w: *const u8, x: *const u8, ids: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32, d1: i32) { indexed_moe_forward::<6>(w, x, ids, out, n, k, batch, topk, kp, d1) }
    #[kernel] pub unsafe fn moe_gemv_fused_gate_up_q3k_q8_1(gw: *const u8, uw: *const u8, x: *const u8, ids: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32, act: i32) { fused_gate_up::<6>(gw, uw, x, ids, out, n, k, batch, topk, kp, act) }
    #[kernel] pub unsafe fn moe_gemv_down_aggregate_q3k_q8_1(w: *const u8, x: *const u8, ids: *const u32, tw: *const f32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32) { down_aggregate::<6>(w, x, ids, tw, out, n, k, batch, topk, kp) }
    #[kernel] pub unsafe fn indexed_moe_forward_q4k_q8_1(w: *const u8, x: *const u8, ids: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32, d1: i32) { indexed_moe_forward::<7>(w, x, ids, out, n, k, batch, topk, kp, d1) }
    #[kernel] pub unsafe fn moe_gemv_fused_gate_up_q4k_q8_1(gw: *const u8, uw: *const u8, x: *const u8, ids: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32, act: i32) { fused_gate_up::<7>(gw, uw, x, ids, out, n, k, batch, topk, kp, act) }
    #[kernel] pub unsafe fn moe_gemv_down_aggregate_q4k_q8_1(w: *const u8, x: *const u8, ids: *const u32, tw: *const f32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32) { down_aggregate::<7>(w, x, ids, tw, out, n, k, batch, topk, kp) }
    #[kernel] pub unsafe fn indexed_moe_forward_q5k_q8_1(w: *const u8, x: *const u8, ids: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32, d1: i32) { indexed_moe_forward::<8>(w, x, ids, out, n, k, batch, topk, kp, d1) }
    #[kernel] pub unsafe fn moe_gemv_fused_gate_up_q5k_q8_1(gw: *const u8, uw: *const u8, x: *const u8, ids: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32, act: i32) { fused_gate_up::<8>(gw, uw, x, ids, out, n, k, batch, topk, kp, act) }
    #[kernel] pub unsafe fn moe_gemv_down_aggregate_q5k_q8_1(w: *const u8, x: *const u8, ids: *const u32, tw: *const f32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32) { down_aggregate::<8>(w, x, ids, tw, out, n, k, batch, topk, kp) }
    #[kernel] pub unsafe fn indexed_moe_forward_q6k_q8_1(w: *const u8, x: *const u8, ids: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32, d1: i32) { indexed_moe_forward::<9>(w, x, ids, out, n, k, batch, topk, kp, d1) }
    #[kernel] pub unsafe fn moe_gemv_fused_gate_up_q6k_q8_1(gw: *const u8, uw: *const u8, x: *const u8, ids: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32, act: i32) { fused_gate_up::<9>(gw, uw, x, ids, out, n, k, batch, topk, kp, act) }
    #[kernel] pub unsafe fn moe_gemv_down_aggregate_q6k_q8_1(w: *const u8, x: *const u8, ids: *const u32, tw: *const f32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32) { down_aggregate::<9>(w, x, ids, tw, out, n, k, batch, topk, kp) }
    #[kernel] pub unsafe fn moe_gemv_fused_gate_up_iq3_xxs_q8_1(gw: *const u8, uw: *const u8, x: *const u8, ids: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32, act: i32) { fused_gate_up::<18>(gw, uw, x, ids, out, n, k, batch, topk, kp, act) }
    #[kernel] pub unsafe fn moe_gemv_fused_gate_up_iq4_xs_q8_1(gw: *const u8, uw: *const u8, x: *const u8, ids: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32, act: i32) { fused_gate_up::<23>(gw, uw, x, ids, out, n, k, batch, topk, kp, act) }
    #[kernel] pub unsafe fn moe_gemv_fused_gate_up_iq4_nl_q8_1(gw: *const u8, uw: *const u8, x: *const u8, ids: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32, act: i32) { fused_gate_up::<20>(gw, uw, x, ids, out, n, k, batch, topk, kp, act) }
    #[kernel] pub unsafe fn moe_gemv_fused_gate_up_iq2_xs_q8_1(gw: *const u8, uw: *const u8, x: *const u8, ids: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32, act: i32) { fused_gate_up::<17>(gw, uw, x, ids, out, n, k, batch, topk, kp, act) }
    #[kernel] pub unsafe fn moe_gemv_down_aggregate_iq3_xxs_q8_1(w: *const u8, x: *const u8, ids: *const u32, tw: *const f32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32) { down_aggregate::<18>(w, x, ids, tw, out, n, k, batch, topk, kp) }
    #[kernel] pub unsafe fn moe_gemv_down_aggregate_iq4_xs_q8_1(w: *const u8, x: *const u8, ids: *const u32, tw: *const f32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32) { down_aggregate::<23>(w, x, ids, tw, out, n, k, batch, topk, kp) }
    #[kernel] pub unsafe fn moe_gemv_down_aggregate_iq4_nl_q8_1(w: *const u8, x: *const u8, ids: *const u32, tw: *const f32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32) { down_aggregate::<20>(w, x, ids, tw, out, n, k, batch, topk, kp) }
    #[kernel] pub unsafe fn moe_gemv_down_aggregate_iq2_xs_q8_1(w: *const u8, x: *const u8, ids: *const u32, tw: *const f32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32) { down_aggregate::<17>(w, x, ids, tw, out, n, k, batch, topk, kp) }
    #[kernel] pub unsafe fn moe_gemv_fused_gate_up_iq2_s_q8_1(gw: *const u8, uw: *const u8, x: *const u8, ids: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32, act: i32) { fused_gate_up::<22>(gw, uw, x, ids, out, n, k, batch, topk, kp, act) }
    #[kernel] pub unsafe fn moe_gemv_down_aggregate_iq2_s_q8_1(w: *const u8, x: *const u8, ids: *const u32, tw: *const f32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32) { down_aggregate::<22>(w, x, ids, tw, out, n, k, batch, topk, kp) }
    #[kernel] pub unsafe fn moe_gemv_fused_gate_up_iq2_xxs_q8_1(gw: *const u8, uw: *const u8, x: *const u8, ids: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32, act: i32) { fused_gate_up::<11>(gw, uw, x, ids, out, n, k, batch, topk, kp, act) }
    #[kernel] pub unsafe fn moe_gemv_down_aggregate_iq2_xxs_q8_1(w: *const u8, x: *const u8, ids: *const u32, tw: *const f32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32) { down_aggregate::<11>(w, x, ids, tw, out, n, k, batch, topk, kp) }
    #[kernel] pub unsafe fn indexed_moe_forward_iq2_xxs_q8_1(w: *const u8, x: *const u8, ids: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32, d1: i32) { indexed_moe_forward::<11>(w, x, ids, out, n, k, batch, topk, kp, d1) }
    #[kernel] pub unsafe fn indexed_moe_forward_q8_1_q8_1(w: *const u8, x: *const u8, ids: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32, d1: i32) { indexed_moe_forward::<10>(w, x, ids, out, n, k, batch, topk, kp, d1) }
    #[kernel] pub unsafe fn moe_gemv_fused_gate_up_q8_1_q8_1(gw: *const u8, uw: *const u8, x: *const u8, ids: *const u32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32, act: i32) { fused_gate_up::<10>(gw, uw, x, ids, out, n, k, batch, topk, kp, act) }
    #[kernel] pub unsafe fn moe_gemv_down_aggregate_q8_1_q8_1(w: *const u8, x: *const u8, ids: *const u32, tw: *const f32, out: *mut f32, n: i32, k: i32, batch: i32, topk: i32, kp: i32) { down_aggregate::<10>(w, x, ids, tw, out, n, k, batch, topk, kp) }
    // GENERATED MOE KERNELS END

    // ------------------------------------------------------------------------------------------
    // moe_grouped/moe_grouped.cu. The reference stages byte tiles through shared memory; the
    // arithmetic per (row, token) is a plain sequential sum over weight blocks and sub-blocks, so the
    // port reads global memory directly with the same thread -> (rows, tokens) assignment:
    // grid (ceil(N/64), E), block (32, 8), thread (tx, ty) owns rows n_base + {tx, 32+tx} and tile
    // tokens ty*8 .. ty*8+7 of every 64-token chunk.

    #[inline(always)]
    unsafe fn grouped_store(out: *mut f32, tw: *const f32, ti: i32, flat: i32, topk: i32, n: i32, n_row: i32, acc: f32) {
        if !tw.is_null() {
            let o = ((flat.wrapping_div(topk) as i64 as u64).wrapping_mul(n as i64 as u64)).wrapping_add(n_row as i64 as u64);
            DeviceAtomicF32::from_ptr(out.add(o as usize)).fetch_add(mulf(acc, *tw.offset(flat as isize)), AtomicOrdering::Relaxed);
        } else {
            let o = ((ti as i64 as u64).wrapping_mul(n as i64 as u64)).wrapping_add(n_row as i64 as u64);
            *out.add(o as usize) = acc;
        }
    }

    #[inline(always)]
    unsafe fn input_row_of(ti: i32, flat: i32, topk: i32, d1: i32) -> i32 {
        if d1 == 0 { ti } else if d1 == 1 { flat.wrapping_div(topk) } else { flat }
    }

    #[inline(always)]
    unsafe fn moe_grouped<const FMT: u32>(
        all_w: *const u8, all_x: *const u8, bounds: *const i32, sorted: *const i32, tw: *const f32, out: *mut f32, n: i32, k: i32,
        k_padded: i32, num_experts: i32, topk: i32, d1: i32,
    ) {
        let (qk, qi, vdr, bs) = fmt_params::<FMT>();
        let n_base = (thread::blockIdx_x() as i32).wrapping_mul(64);
        let expert = thread::blockIdx_y() as i32;
        if n_base >= n || expert >= num_experts {
            return;
        }
        let t_start = *bounds.offset(expert as isize);
        let t_end = *bounds.offset(expert as isize + 1);
        if t_start >= t_end {
            return;
        }
        let bprw = k / qk;
        let bprx = k_padded / 32;
        let xbpw = qk / 32;
        let ew_off = (expert as i64 as u64).wrapping_mul(n as i64 as u64).wrapping_mul(bprw as i64 as u64).wrapping_mul(bs as u64);
        let (tx, ty) = (thread::threadIdx_x() as i32, thread::threadIdx_y() as i32);
        let mut t_base = t_start;
        while t_base < t_end {
            let m_count = if t_end - t_base < 64 { t_end - t_base } else { 64 };
            let mut jl = 0;
            while jl < 8 {
                let j = ty * 8 + jl;
                if j < m_count {
                    let ti = t_base + j;
                    let flat = *sorted.offset(ti as isize);
                    let in_row = input_row_of(ti, flat, topk, d1);
                    let xrow = all_x.wrapping_add((in_row as i64 as u64).wrapping_mul(bprx as i64 as u64).wrapping_mul(36) as usize);
                    let mut il = 0;
                    while il < 2 {
                        let n_row = n_base + il * 32 + tx;
                        if n_row < n {
                            let wrow = all_w.wrapping_add(ew_off.wrapping_add((n_row as i64 as u64).wrapping_mul(bprw as i64 as u64).wrapping_mul(bs as u64)) as usize);
                            let mut acc = 0.0f32;
                            let mut kb = 0;
                            while kb < bprw {
                                let wb = wrow.wrapping_add((kb as usize) * bs as usize);
                                let xb = xrow.wrapping_add((kb * xbpw) as usize * 36);
                                let mut iqs = 0;
                                while iqs < qi {
                                    acc = vdot::<FMT, true>(acc, wb, xb, iqs as usize);
                                    iqs += vdr;
                                }
                                kb += 1;
                            }
                            grouped_store(out, tw, ti, flat, topk, n, n_row, acc);
                        }
                        il += 1;
                    }
                }
                jl += 1;
            }
            t_base += 64;
        }
    }
    // GENERATED GROUPED KERNELS BEGIN
    #[kernel] pub unsafe fn moe_grouped_gemm_q4_0(w: *const u8, x: *const u8, eb: *const i32, st: *const i32, tw: *const f32, out: *mut f32, n: i32, k: i32, kp: i32, ne: i32, topk: i32, d1: i32) { moe_grouped::<0>(w, x, eb, st, tw, out, n, k, kp, ne, topk, d1) }
    #[kernel] pub unsafe fn moe_grouped_gemm_q4_1(w: *const u8, x: *const u8, eb: *const i32, st: *const i32, tw: *const f32, out: *mut f32, n: i32, k: i32, kp: i32, ne: i32, topk: i32, d1: i32) { moe_grouped::<1>(w, x, eb, st, tw, out, n, k, kp, ne, topk, d1) }
    #[kernel] pub unsafe fn moe_grouped_gemm_q5_0(w: *const u8, x: *const u8, eb: *const i32, st: *const i32, tw: *const f32, out: *mut f32, n: i32, k: i32, kp: i32, ne: i32, topk: i32, d1: i32) { moe_grouped::<2>(w, x, eb, st, tw, out, n, k, kp, ne, topk, d1) }
    #[kernel] pub unsafe fn moe_grouped_gemm_q5_1(w: *const u8, x: *const u8, eb: *const i32, st: *const i32, tw: *const f32, out: *mut f32, n: i32, k: i32, kp: i32, ne: i32, topk: i32, d1: i32) { moe_grouped::<3>(w, x, eb, st, tw, out, n, k, kp, ne, topk, d1) }
    #[kernel] pub unsafe fn moe_grouped_gemm_q2k(w: *const u8, x: *const u8, eb: *const i32, st: *const i32, tw: *const f32, out: *mut f32, n: i32, k: i32, kp: i32, ne: i32, topk: i32, d1: i32) { moe_grouped::<5>(w, x, eb, st, tw, out, n, k, kp, ne, topk, d1) }
    #[kernel] pub unsafe fn moe_grouped_gemm_q3k(w: *const u8, x: *const u8, eb: *const i32, st: *const i32, tw: *const f32, out: *mut f32, n: i32, k: i32, kp: i32, ne: i32, topk: i32, d1: i32) { moe_grouped::<6>(w, x, eb, st, tw, out, n, k, kp, ne, topk, d1) }
    #[kernel] pub unsafe fn moe_grouped_gemm_q4k(w: *const u8, x: *const u8, eb: *const i32, st: *const i32, tw: *const f32, out: *mut f32, n: i32, k: i32, kp: i32, ne: i32, topk: i32, d1: i32) { moe_grouped::<7>(w, x, eb, st, tw, out, n, k, kp, ne, topk, d1) }
    #[kernel] pub unsafe fn moe_grouped_gemm_q5k(w: *const u8, x: *const u8, eb: *const i32, st: *const i32, tw: *const f32, out: *mut f32, n: i32, k: i32, kp: i32, ne: i32, topk: i32, d1: i32) { moe_grouped::<8>(w, x, eb, st, tw, out, n, k, kp, ne, topk, d1) }
    #[kernel] pub unsafe fn moe_grouped_gemm_q6k(w: *const u8, x: *const u8, eb: *const i32, st: *const i32, tw: *const f32, out: *mut f32, n: i32, k: i32, kp: i32, ne: i32, topk: i32, d1: i32) { moe_grouped::<9>(w, x, eb, st, tw, out, n, k, kp, ne, topk, d1) }
    #[kernel] pub unsafe fn moe_grouped_gemm_q8_1(w: *const u8, x: *const u8, eb: *const i32, st: *const i32, tw: *const f32, out: *mut f32, n: i32, k: i32, kp: i32, ne: i32, topk: i32, d1: i32) { moe_grouped::<10>(w, x, eb, st, tw, out, n, k, kp, ne, topk, d1) }
    // GENERATED GROUPED KERNELS END

    /// moe_grouped_gemm_q8_0 (the dp4a K-tile kernel): acc = fma(d_w * d_x, sumi, acc) per 32-value
    /// block in 256-value K tiles. In a partial last tile the reference still runs all 8 block slots:
    /// the missing ones have sumi = 0 but read the d_w / d_x left in shared memory by the previous
    /// tile (so a NaN/inf scale product there still reaches the sum, and +-0 can flip a -0).
    #[kernel]
    pub unsafe fn moe_grouped_gemm_q8_0(
        all_w: *const u8, all_x: *const u8, bounds: *const i32, sorted: *const i32, tw: *const f32, out: *mut f32, n: i32, k: i32,
        k_padded: i32, num_experts: i32, topk: i32, d1: i32,
    ) {
        let n_base = (thread::blockIdx_x() as i32).wrapping_mul(64);
        let expert = thread::blockIdx_y() as i32;
        if n_base >= n || expert >= num_experts {
            return;
        }
        let t_start = *bounds.offset(expert as isize);
        let t_end = *bounds.offset(expert as isize + 1);
        if t_start >= t_end {
            return;
        }
        let bprw = k / 32;
        let bprx = k_padded / 32;
        let ew_off = (expert as i64 as u64).wrapping_mul(n as i64 as u64).wrapping_mul(bprw as i64 as u64);
        let (tx, ty) = (thread::threadIdx_x() as i32, thread::threadIdx_y() as i32);
        let mut t_base = t_start;
        while t_base < t_end {
            let m_count = if t_end - t_base < 64 { t_end - t_base } else { 64 };
            let mut jl = 0;
            while jl < 8 {
                let j = ty * 8 + jl;
                if j < m_count {
                    let ti = t_base + j;
                    let flat = *sorted.offset(ti as isize);
                    let in_row = input_row_of(ti, flat, topk, d1);
                    let xblk = |qb: i32| all_x.wrapping_add(((in_row as i64 as u64).wrapping_mul(bprx as i64 as u64).wrapping_add(qb as i64 as u64)).wrapping_mul(36) as usize);
                    let mut il = 0;
                    while il < 2 {
                        let n_row = n_base + il * 32 + tx;
                        if n_row < n {
                            let wblk = |qb: i32| all_w.wrapping_add((ew_off.wrapping_add((n_row as i64 as u64).wrapping_mul(bprw as i64 as u64)).wrapping_add(qb as i64 as u64)).wrapping_mul(34) as usize);
                            let d_w_of = |qb: i32| if qb < bprw { h2f32(ld16(wblk(qb), 0)) } else { 0.0 };
                            let d_x_of = |qb: i32| if qb < bprx { h2f32(ld16(xblk(qb), 0)) } else { 0.0 };
                            let mut acc = 0.0f32;
                            let mut k_base = 0;
                            while k_base < k {
                                let kta = if k - k_base < 256 { k - k_base } else { 256 };
                                let ktq8 = kta / 32;
                                let mut kb = 0;
                                while kb < 8 {
                                    let qb = k_base / 32 + kb;
                                    if kb < ktq8 {
                                        let (wb, xb) = (wblk(qb), xblk(qb));
                                        let mut sumi = 0i32;
                                        let mut kk = 0;
                                        while kk < 8 {
                                            sumi = dp4a(ld32u(wb, 2 + 4 * kk), ld32a(xb, 4 + 4 * kk), sumi);
                                            kk += 1;
                                        }
                                        acc = fmaf(mulf(d_w_of(qb), d_x_of(qb)), sumi as f32, acc);
                                    } else if k_base >= 256 {
                                        // stale slot: the previous tile's scales, sumi == 0
                                        acc = fmaf(mulf(d_w_of(qb - 8), d_x_of(qb - 8)), 0.0, acc);
                                    }
                                    kb += 1;
                                }
                                k_base += 256;
                            }
                            grouped_store(out, tw, ti, flat, topk, n, n_row, acc);
                        }
                        il += 1;
                    }
                }
                jl += 1;
            }
            t_base += 64;
        }
    }

    #[kernel]
    pub unsafe fn moe_dispatch_count_kernel(topk_ids: *const i32, counts: *mut i32, total: i32) {
        let idx = gid_x() as i32;
        if idx < total {
            atomic_add_i32(counts.offset(*topk_ids.offset(idx as isize) as isize), 1);
        }
    }
    #[kernel]
    pub unsafe fn moe_dispatch_prefix_sum_kernel(counts: *const i32, bounds: *mut i32, num_experts: i32) {
        if thread::threadIdx_x() == 0 && thread::blockIdx_x() == 0 {
            *bounds = 0;
            let mut i = 0;
            while i < num_experts {
                *bounds.offset(i as isize + 1) = (*bounds.offset(i as isize)).wrapping_add(*counts.offset(i as isize));
                i += 1;
            }
        }
    }
    #[kernel]
    pub unsafe fn moe_dispatch_scatter_kernel(topk_ids: *const i32, cursors: *mut i32, sorted_token_ids: *mut i32, sorted_source_ids: *mut i32, total: i32, topk: i32) {
        let idx = gid_x() as i32;
        if idx < total {
            let e = *topk_ids.offset(idx as isize);
            let pos = atomic_add_i32(cursors.offset(e as isize), 1);
            *sorted_token_ids.offset(pos as isize) = idx;
            if !sorted_source_ids.is_null() {
                *sorted_source_ids.offset(pos as isize) = idx.wrapping_div(topk);
            }
        }
    }

    /// Stable twin of moe_dispatch_scatter_kernel (redcell): grid (num_experts), block (32). The warp of
    /// expert e walks the assignments in index order and hands out positions cursors[e].. in that order
    /// (a warp prefix sum of the matches), so sorted_token_ids is ascending within every expert. The
    /// atomic scatter's order changed from run to run, and with it which MMQ tile (and stream-k split)
    /// a token landed in, i.e. the bits of the grouped prefill.
    #[kernel]
    pub unsafe fn moe_dispatch_scatter_stable_kernel(topk_ids: *const i32, cursors: *mut i32, sorted_token_ids: *mut i32, sorted_source_ids: *mut i32, total: i32, topk: i32) {
        let e = thread::blockIdx_x() as i32;
        let lane = thread::threadIdx_x() as i32;
        let mut pos = *cursors.offset(e as isize);
        let mut base = 0i32;
        while base < total {
            let idx = base.wrapping_add(lane);
            let hit = if idx < total && *topk_ids.offset(idx as isize) == e { 1i32 } else { 0i32 };
            let mut inc = hit;
            let mut d = 1u32;
            while d < 32 {
                let o = warp::shuffle_up_sync(0xffff_ffff, inc as u32, d) as i32;
                if lane as u32 >= d {
                    inc = inc.wrapping_add(o);
                }
                d <<= 1;
            }
            if hit != 0 {
                let p = pos.wrapping_add(inc).wrapping_sub(1);
                *sorted_token_ids.offset(p as isize) = idx;
                if !sorted_source_ids.is_null() {
                    *sorted_source_ids.offset(p as isize) = idx.wrapping_div(topk);
                }
            }
            pos = pos.wrapping_add(warp::shuffle_sync(0xffff_ffff, inc as u32, 31) as i32);
            base = base.wrapping_add(32);
        }
        // leave the cursor where the atomic scatter leaves it (expert_bounds[e + 1])
        if lane == 0 {
            *cursors.offset(e as isize) = pos;
        }
    }

    #[inline(always)]
    unsafe fn weighted_reduce<T: Val>(inputs: *const f32, topk_weights: *const f32, outputs: *mut T, num_tokens: i32, hidden: i32, topk: i32) {
        let token = thread::blockIdx_x() as i32;
        let h = thread::blockIdx_y().wrapping_mul(thread::blockDim_x()).wrapping_add(thread::threadIdx_x()) as i32;
        if token >= num_tokens {
            return;
        }
        let weights = DynamicSharedArray::<f32>::get();
        let mut slot = thread::threadIdx_x() as i32;
        while slot < topk {
            *weights.offset(slot as isize) = *topk_weights.offset(token.wrapping_mul(topk).wrapping_add(slot) as isize);
            slot += thread::blockDim_x() as i32;
        }
        thread::sync_threads();
        if h >= hidden {
            return;
        }
        let base = (token as i64 as u64).wrapping_mul(topk as i64 as u64).wrapping_mul(hidden as i64 as u64).wrapping_add(h as i64 as u64);
        let mut acc = 0.0f32;
        let mut slot = 0;
        while slot < topk {
            let v = *inputs.add(base.wrapping_add((slot as i64 as u64).wrapping_mul(hidden as i64 as u64)) as usize);
            acc = fmaf(v, *weights.offset(slot as isize), acc);
            slot += 1;
        }
        *outputs.add((token as i64 as u64).wrapping_mul(hidden as i64 as u64).wrapping_add(h as i64 as u64) as usize) = T::t(acc);
    }
    #[kernel] pub unsafe fn moe_weighted_reduce_flat_f32(i: *const f32, w: *const f32, o: *mut f32, t: i32, h: i32, k: i32) { weighted_reduce(i, w, o, t, h, k) }
    #[kernel] pub unsafe fn moe_weighted_reduce_flat_bf16(i: *const f32, w: *const f32, o: *mut B, t: i32, h: i32, k: i32) { weighted_reduce(i, w, o, t, h, k) }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() == 4 && a[1] == "--exit-case" {
        gate::exit_case(a[2] == "rust", a[3].parse().unwrap());
        std::process::exit(0);
    }
    std::process::exit(if gate::run() { 0 } else { 1 });
}
