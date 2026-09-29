//! mistralrs-paged-attn group 1 in cuda-oxide: the kernels of reshape_and_cache, gather_kv_cache,
//! update_kvscales, copy_blocks, concat_and_cache_mla, gather_mla_cache, paged_attention v1/v2
//! (f32/f16/bf16, auto and fp8-e4m3 caches) and flash_attn_sinks (+varlen), plus (src/launch.rs)
//! pure-Rust twins of their extern "C" host launchers. Bit-identical to the nvcc build
//! (`-O3 --use_fast_math -DENABLE_FP8`, sm_120a; the SASS in libmistralrspagedattention.a is the
//! spec), checked launcher-by-launcher by the gate in src/gate.rs.
//!
//! Every floating-point operation is written out as the exact PTX instruction nvcc emits (same
//! spelling, including the non-`.rn` `mul.ftz`/`add.ftz`/`mul.f16x2`/`add.f16x2` that ptxas may fuse),
//! with the same data flow, so ptxas makes the same fusion decisions:
//! - f32 add/sub/mul are `.ftz`; C `a*b + c` contractions are explicit `fma.rn.ftz.f32` exactly where
//!   nvcc's PTX has them (e.g. qk: `fma(q0, k0, q1*k1)`, then `fma(q_i, k_i, acc)`).
//! - `__expf`/`expf(x)` = `ex2.approx.ftz(x * 1.4427)`; `a / b` and `__fdividef` = `div.approx.ftz.f32`
//!   (MUFU.RCP + FMUL.FTZ); `fmaxf` = `max.ftz.f32`; `fabsf` = `abs.ftz.f32`; float compares are
//!   `.ftz` (a denormal alibi slope counts as zero).
//! - fp8 e4m3: encode `cvt.rn.satfinite.e4m3x2.f32` of `x / scale`; decode `cvt.rn.f16x2.e4m3x2`, then
//!   `half * scale` in f32 and back to the target type (for f32 targets reached through a half2
//!   vector conversion the product is rounded to half first, as vLLM's quant_utils does).
#![allow(non_snake_case, clippy::missing_safety_doc, clippy::too_many_arguments)]
mod gate;
pub mod launch;

use cuda_device::atomic::{AtomicOrdering, SystemAtomicI32};
use cuda_device::{DynamicSharedArray, SharedArray, kernel, ptx_asm, thread, warp};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    // ------------------------------------------------------------------------------------------
    // exact nvcc PTX primitives

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
        unsafe { ptx_asm!("max.ftz.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)) };
        r
    }
    #[inline(always)]
    pub fn fabs(a: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("abs.ftz.f32 %0, %1;", out("=f") r, in("f") a, options(register_only)) };
        r
    }
    #[inline(always)]
    pub fn fdiv(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("div.approx.ftz.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)) };
        r
    }
    #[inline(always)]
    pub fn ex2(a: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("ex2.approx.ftz.f32 %0, %1;", out("=f") r, in("f") a, options(register_only)) };
        r
    }
    /// fast-math `__expf(x)` / `expf(x)`.
    #[inline(always)]
    pub fn fexp(a: f32) -> f32 {
        ex2(fmul(a, f32::from_bits(0x3FB8AA3B)))
    }
    #[inline(always)]
    pub fn tanh_approx(a: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("tanh.approx.f32 %0, %1;", out("=f") r, in("f") a, options(register_only)) };
        r
    }
    /// `setp.gt.ftz.f32` as a bool.
    #[inline(always)]
    pub fn gt_ftz(a: f32, b: f32) -> bool {
        let r: u32;
        unsafe {
            ptx_asm!("{ .reg .pred p; setp.gt.ftz.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }", out("=r") r, in("f") a, in("f") b, options(register_only))
        };
        r != 0
    }
    /// `setp.le.ftz.f32` as a bool.
    #[inline(always)]
    pub fn le_ftz(a: f32, b: f32) -> bool {
        let r: u32;
        unsafe {
            ptx_asm!("{ .reg .pred p; setp.le.ftz.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }", out("=r") r, in("f") a, in("f") b, options(register_only))
        };
        r != 0
    }
    /// `setp.neu.ftz.f32` as a bool.
    #[inline(always)]
    pub fn neu_ftz(a: f32, b: f32) -> bool {
        let r: u32;
        unsafe {
            ptx_asm!("{ .reg .pred p; setp.neu.ftz.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }", out("=r") r, in("f") a, in("f") b, options(register_only))
        };
        r != 0
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
    /// `cvt.rn.f32.u32`.
    #[inline(always)]
    pub fn u2f(a: u32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("cvt.rn.f32.u32 %0, %1;", out("=f") r, in("r") a, options(register_only)) };
        r
    }
    #[inline(always)]
    pub fn h2f(h: u16) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("cvt.f32.f16 %0, %1;", out("=f") r, in("h") h, options(register_only)) };
        r
    }
    #[inline(always)]
    pub fn f2h(f: f32) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("cvt.rn.f16.f32 %0, %1;", out("=h") r, in("f") f, options(register_only)) };
        r
    }
    #[inline(always)]
    pub fn bf2f(h: u16) -> f32 {
        f32::from_bits((h as u32) << 16)
    }
    #[inline(always)]
    pub fn f2bf(f: f32) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("cvt.rn.bf16.f32 %0, %1;", out("=h") r, in("f") f, options(register_only)) };
        r
    }
    /// `cvt.rn.f16x2.f32 d, hi, lo`.
    #[inline(always)]
    pub fn f2h2(lo: f32, hi: f32) -> u32 {
        let r: u32;
        unsafe { ptx_asm!("cvt.rn.f16x2.f32 %0, %1, %2;", out("=r") r, in("f") hi, in("f") lo, options(register_only)) };
        r
    }
    /// `cvt.rn.bf16x2.f32 d, hi, lo`.
    #[inline(always)]
    pub fn f2bf2(lo: f32, hi: f32) -> u32 {
        let r: u32;
        unsafe { ptx_asm!("cvt.rn.bf16x2.f32 %0, %1, %2;", out("=r") r, in("f") hi, in("f") lo, options(register_only)) };
        r
    }
    /// `mul.f16x2` (not `.rn`: ptxas may fuse it with a following `add.f16x2`).
    #[inline(always)]
    pub fn hmul2(a: u32, b: u32) -> u32 {
        let r: u32;
        unsafe { ptx_asm!("mul.f16x2 %0, %1, %2; //\t\x0b\x0c\r\n", out("=r") r, in("r") a, in("r") b, options(register_only)) };
        r
    }
    #[inline(always)]
    pub fn hadd2(a: u32, b: u32) -> u32 {
        let r: u32;
        unsafe { ptx_asm!("add.f16x2 %0, %1, %2; //\t\x0b\x0c\r\n", out("=r") r, in("r") a, in("r") b, options(register_only)) };
        r
    }
    /// `mul.bf16x2` (`__hmul2` for bf16).
    #[inline(always)]
    pub fn bmul2(a: u32, b: u32) -> u32 {
        let r: u32;
        unsafe { ptx_asm!("mul.bf16x2 %0, %1, %2; //\t\x0b\x0c\r\n", out("=r") r, in("r") a, in("r") b, options(register_only)) };
        r
    }
    /// fp8 e4m3 byte -> f16 bits (`__nv_cvt_fp8_to_halfraw`).
    #[inline(always)]
    pub fn fp8_to_h(b: u8) -> u16 {
        let r: u16;
        unsafe {
            ptx_asm!("{ .reg .b32 t; cvt.rn.f16x2.e4m3x2 t, %1; cvt.u16.u32 %0, t; }", out("=h") r, in("h") b as u16, options(register_only))
        };
        r
    }
    /// f32 -> fp8 e4m3 (`__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3)`).
    #[inline(always)]
    pub fn f_to_fp8(v: f32) -> u8 {
        let r: u16;
        unsafe {
            ptx_asm!("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;", out("=h") r, in("f") 0.0f32, in("f") v, options(register_only))
        };
        r as u8
    }

    // integer division exactly as the PTX (no Rust panics on 0 / overflow)
    #[inline(always)]
    pub fn sdiv(a: i32, b: i32) -> i32 {
        let r: i32;
        unsafe { ptx_asm!("div.s32 %0, %1, %2;", out("=r") r, in("r") a, in("r") b, options(register_only)) };
        r
    }
    /// C `a % b` for int as nvcc emits it (a - (a/b)*b).
    #[inline(always)]
    pub fn srem(a: i32, b: i32) -> i32 {
        a.wrapping_sub(sdiv(a, b).wrapping_mul(b))
    }
    #[inline(always)]
    pub fn udiv(a: u32, b: u32) -> u32 {
        let r: u32;
        unsafe { ptx_asm!("div.u32 %0, %1, %2;", out("=r") r, in("r") a, in("r") b, options(register_only)) };
        r
    }
    #[inline(always)]
    pub fn sdiv64_raw(a: i64, b: i64) -> i64 {
        let r: i64;
        unsafe { ptx_asm!("div.s64 %0, %1, %2;", out("=l") r, in("l") a, in("l") b, options(register_only)) };
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

    #[inline(always)]
    pub fn tid() -> i32 {
        thread::threadIdx_x() as i32
    }
    #[inline(always)]
    pub fn ntid() -> i32 {
        thread::blockDim_x() as i32
    }

    // ------------------------------------------------------------------------------------------
    // element types

    /// f16 element (bit pattern).
    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct H(pub u16);
    /// bf16 element (bit pattern).
    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct B(pub u16);

    pub trait El: Copy {
        /// exact widening to f32
        fn f(self) -> f32;
        /// `from_float` / `__float2half` / `__float2bfloat16` (f32: identity)
        fn from_f(v: f32) -> Self;
        /// reshape_and_cache fp8 store: `scaled_convert<uint8_t, T>(x, scale)`
        fn to_fp8(self, scale: f32) -> u8 {
            f_to_fp8(fdiv(self.f(), scale))
        }
        /// gather_kv_cache fp8 load: `scaled_convert<T, uint8_t>(fp8, scale)`
        fn from_fp8(b: u8, scale: f32) -> Self;
    }
    impl El for f32 {
        #[inline(always)]
        fn f(self) -> f32 {
            self
        }
        #[inline(always)]
        fn from_f(v: f32) -> f32 {
            v
        }
        #[inline(always)]
        fn from_fp8(b: u8, scale: f32) -> f32 {
            fmul(scale, h2f(fp8_to_h(b)))
        }
    }
    impl El for H {
        #[inline(always)]
        fn f(self) -> f32 {
            h2f(self.0)
        }
        #[inline(always)]
        fn from_f(v: f32) -> H {
            H(f2h(v))
        }
        #[inline(always)]
        fn from_fp8(b: u8, scale: f32) -> H {
            H(f2h(fmul(scale, h2f(fp8_to_h(b)))))
        }
    }
    impl El for B {
        #[inline(always)]
        fn f(self) -> f32 {
            bf2f(self.0)
        }
        #[inline(always)]
        fn from_f(v: f32) -> B {
            B(f2bf(v))
        }
        #[inline(always)]
        fn from_fp8(b: u8, scale: f32) -> B {
            B(f2bf(fmul(scale, h2f(fp8_to_h(b)))))
        }
    }

    // ------------------------------------------------------------------------------------------
    // reshape_and_cache_kernel.cu. Grid (num_tokens), block min(num_heads*head_size, 512).

    #[inline(always)]
    unsafe fn reshape_and_cache<T: El, C: Copy, const FP8: bool>(
        key: *const T, value: *const T, key_cache: *mut C, value_cache: *mut C, slot_mapping: *const i64, key_stride: i32,
        value_stride: i32, num_heads: i32, head_size: i32, block_size: i32, x: i32, k_scale: *const f32, v_scale: *const f32,
    ) {
        let token_idx = thread::blockIdx_x() as i64;
        let slot_idx = *slot_mapping.offset(token_idx as isize);
        if slot_idx < 0 {
            return;
        }
        let (block_idx, block_offset) = divrem64(slot_idx, block_size as i64);
        let n = num_heads.wrapping_mul(head_size);
        let hs_x = sdiv(head_size, x);
        let mut i = tid();
        while i < n {
            let src_key_idx = token_idx.wrapping_mul(key_stride as i64).wrapping_add(i as i64);
            let src_value_idx = token_idx.wrapping_mul(value_stride as i64).wrapping_add(i as i64);
            let head_idx = sdiv(i, head_size);
            let head_offset = i.wrapping_sub(head_idx.wrapping_mul(head_size));
            let x_idx = sdiv(head_offset, x);
            let x_offset = head_offset.wrapping_sub(x_idx.wrapping_mul(x));
            let tgt_key_idx = block_idx
                .wrapping_mul(num_heads as i64)
                .wrapping_mul(block_size as i64)
                .wrapping_mul(hs_x as i64)
                .wrapping_add(block_offset)
                .wrapping_mul(x as i64)
                .wrapping_add(head_idx.wrapping_mul(x.wrapping_mul(block_size).wrapping_mul(hs_x)) as i64)
                .wrapping_add(x_idx.wrapping_mul(x.wrapping_mul(block_size)) as i64)
                .wrapping_add(x_offset as i64);
            let tgt_value_idx = block_idx
                .wrapping_mul(num_heads as i64)
                .wrapping_mul(block_size as i64)
                .wrapping_mul(head_size as i64)
                .wrapping_add(block_offset)
                .wrapping_add(head_idx.wrapping_mul(block_size.wrapping_mul(head_size)) as i64)
                .wrapping_add((head_offset as i64).wrapping_mul(block_size as i64));
            let tk = *key.offset(src_key_idx as isize);
            let tv = *value.offset(src_value_idx as isize);
            if FP8 {
                *(key_cache as *mut u8).offset(tgt_key_idx as isize) = tk.to_fp8(*k_scale);
                *(value_cache as *mut u8).offset(tgt_value_idx as isize) = tv.to_fp8(*v_scale);
            } else {
                *(key_cache as *mut T).offset(tgt_key_idx as isize) = tk;
                *(value_cache as *mut T).offset(tgt_value_idx as isize) = tv;
            }
            i = i.wrapping_add(ntid());
        }
    }
    #[kernel] pub unsafe fn rac_f32(k: *const f32, v: *const f32, kc: *mut f32, vc: *mut f32, s: *const i64, ks: i32, vs: i32, nh: i32, hs: i32, bs: i32, x: i32, ksc: *const f32, vsc: *const f32) { reshape_and_cache::<f32, f32, false>(k, v, kc, vc, s, ks, vs, nh, hs, bs, x, ksc, vsc) }
    #[kernel] pub unsafe fn rac_f16(k: *const H, v: *const H, kc: *mut H, vc: *mut H, s: *const i64, ks: i32, vs: i32, nh: i32, hs: i32, bs: i32, x: i32, ksc: *const f32, vsc: *const f32) { reshape_and_cache::<H, H, false>(k, v, kc, vc, s, ks, vs, nh, hs, bs, x, ksc, vsc) }
    #[kernel] pub unsafe fn rac_bf16(k: *const B, v: *const B, kc: *mut B, vc: *mut B, s: *const i64, ks: i32, vs: i32, nh: i32, hs: i32, bs: i32, x: i32, ksc: *const f32, vsc: *const f32) { reshape_and_cache::<B, B, false>(k, v, kc, vc, s, ks, vs, nh, hs, bs, x, ksc, vsc) }
    #[kernel] pub unsafe fn rac_f32_fp8(k: *const f32, v: *const f32, kc: *mut u8, vc: *mut u8, s: *const i64, ks: i32, vs: i32, nh: i32, hs: i32, bs: i32, x: i32, ksc: *const f32, vsc: *const f32) { reshape_and_cache::<f32, u8, true>(k, v, kc, vc, s, ks, vs, nh, hs, bs, x, ksc, vsc) }
    #[kernel] pub unsafe fn rac_f16_fp8(k: *const H, v: *const H, kc: *mut u8, vc: *mut u8, s: *const i64, ks: i32, vs: i32, nh: i32, hs: i32, bs: i32, x: i32, ksc: *const f32, vsc: *const f32) { reshape_and_cache::<H, u8, true>(k, v, kc, vc, s, ks, vs, nh, hs, bs, x, ksc, vsc) }
    #[kernel] pub unsafe fn rac_bf16_fp8(k: *const B, v: *const B, kc: *mut u8, vc: *mut u8, s: *const i64, ks: i32, vs: i32, nh: i32, hs: i32, bs: i32, x: i32, ksc: *const f32, vsc: *const f32) { reshape_and_cache::<B, u8, true>(k, v, kc, vc, s, ks, vs, nh, hs, bs, x, ksc, vsc) }

    // ------------------------------------------------------------------------------------------
    // gather_kv_cache_kernel.cu. Grid (num_tokens), block min(num_kv_heads*head_size, 512).

    #[inline(always)]
    unsafe fn gather_kv_cache<C: Copy, T: El, const FP8: bool>(
        key_cache: *const C, value_cache: *const C, k_out: *mut T, v_out: *mut T, k_scale: *const f32, v_scale: *const f32,
        block_table: *const i32, cu_seq_lens: *const i32, num_tokens: i32, num_seqs: i32, block_size: i32, block_table_stride: i32,
        num_kv_heads: i32, head_size: i32, x: i32,
    ) {
        let token_id = thread::blockIdx_x() as i32;
        if token_id >= num_tokens {
            return;
        }
        let mut lo: i32 = 0;
        let mut hi: i32 = num_seqs;
        while lo < hi {
            let mid = ((lo.wrapping_add(hi).wrapping_add(1) as u32) >> 1) as i32;
            if *cu_seq_lens.add(mid as u32 as usize) <= token_id {
                lo = mid;
            } else {
                hi = mid.wrapping_sub(1);
            }
        }
        let batch_id = lo;
        let batch_offset = token_id.wrapping_sub(*cu_seq_lens.add(batch_id as u32 as usize));
        let block_table_id = sdiv(batch_offset, block_size);
        let slot = batch_offset.wrapping_sub(block_table_id.wrapping_mul(block_size));
        let block_id = *block_table.offset(batch_id.wrapping_mul(block_table_stride).wrapping_add(block_table_id) as isize);
        let n = num_kv_heads.wrapping_mul(head_size);
        let out_base = (token_id as i64).wrapping_mul(num_kv_heads as i64).wrapping_mul(head_size as i64);
        let hs_x = sdiv(head_size, x);
        let k_block_stride = (num_kv_heads as i64).wrapping_mul(hs_x as i64).wrapping_mul(block_size as i64).wrapping_mul(x as i64);
        let k_head_stride = (hs_x as i64).wrapping_mul(block_size as i64).wrapping_mul(x as i64);
        let v_block_stride = (num_kv_heads as i64).wrapping_mul(head_size as i64).wrapping_mul(block_size as i64);
        let v_head_stride = (head_size as i64).wrapping_mul(block_size as i64);
        let mut i = tid();
        while i < n {
            let head_idx = sdiv(i, head_size);
            let d = i.wrapping_sub(head_idx.wrapping_mul(head_size));
            let x_idx = sdiv(d, x);
            let x_offset = d.wrapping_sub(x_idx.wrapping_mul(x));
            let k_src_idx = (block_id as i64)
                .wrapping_mul(k_block_stride)
                .wrapping_add((head_idx as i64).wrapping_mul(k_head_stride))
                .wrapping_add(x_idx.wrapping_mul(block_size).wrapping_mul(x) as i64)
                .wrapping_add(slot.wrapping_mul(x) as i64)
                .wrapping_add(x_offset as i64);
            let v_src_idx = (block_id as i64)
                .wrapping_mul(v_block_stride)
                .wrapping_add((head_idx as i64).wrapping_mul(v_head_stride))
                .wrapping_add((d as i64).wrapping_mul(block_size as i64))
                .wrapping_add(slot as i64);
            let o = out_base.wrapping_add(i as i64) as isize;
            if FP8 {
                *k_out.offset(o) = T::from_fp8(*(key_cache as *const u8).offset(k_src_idx as isize), *k_scale);
                *v_out.offset(o) = T::from_fp8(*(value_cache as *const u8).offset(v_src_idx as isize), *v_scale);
            } else {
                *k_out.offset(o) = *(key_cache as *const T).offset(k_src_idx as isize);
                *v_out.offset(o) = *(value_cache as *const T).offset(v_src_idx as isize);
            }
            i = i.wrapping_add(ntid());
        }
    }
    #[kernel] pub unsafe fn gkv_f32(kc: *const f32, vc: *const f32, ko: *mut f32, vo: *mut f32, ks: *const f32, vs: *const f32, bt: *const i32, cu: *const i32, nt: i32, ns: i32, bs: i32, bts: i32, nkv: i32, hs: i32, x: i32) { gather_kv_cache::<f32, f32, false>(kc, vc, ko, vo, ks, vs, bt, cu, nt, ns, bs, bts, nkv, hs, x) }
    #[kernel] pub unsafe fn gkv_f16(kc: *const H, vc: *const H, ko: *mut H, vo: *mut H, ks: *const f32, vs: *const f32, bt: *const i32, cu: *const i32, nt: i32, ns: i32, bs: i32, bts: i32, nkv: i32, hs: i32, x: i32) { gather_kv_cache::<H, H, false>(kc, vc, ko, vo, ks, vs, bt, cu, nt, ns, bs, bts, nkv, hs, x) }
    #[kernel] pub unsafe fn gkv_bf16(kc: *const B, vc: *const B, ko: *mut B, vo: *mut B, ks: *const f32, vs: *const f32, bt: *const i32, cu: *const i32, nt: i32, ns: i32, bs: i32, bts: i32, nkv: i32, hs: i32, x: i32) { gather_kv_cache::<B, B, false>(kc, vc, ko, vo, ks, vs, bt, cu, nt, ns, bs, bts, nkv, hs, x) }
    #[kernel] pub unsafe fn gkv_fp8_f32(kc: *const u8, vc: *const u8, ko: *mut f32, vo: *mut f32, ks: *const f32, vs: *const f32, bt: *const i32, cu: *const i32, nt: i32, ns: i32, bs: i32, bts: i32, nkv: i32, hs: i32, x: i32) { gather_kv_cache::<u8, f32, true>(kc, vc, ko, vo, ks, vs, bt, cu, nt, ns, bs, bts, nkv, hs, x) }
    #[kernel] pub unsafe fn gkv_fp8_f16(kc: *const u8, vc: *const u8, ko: *mut H, vo: *mut H, ks: *const f32, vs: *const f32, bt: *const i32, cu: *const i32, nt: i32, ns: i32, bs: i32, bts: i32, nkv: i32, hs: i32, x: i32) { gather_kv_cache::<u8, H, true>(kc, vc, ko, vo, ks, vs, bt, cu, nt, ns, bs, bts, nkv, hs, x) }
    #[kernel] pub unsafe fn gkv_fp8_bf16(kc: *const u8, vc: *const u8, ko: *mut B, vo: *mut B, ks: *const f32, vs: *const f32, bt: *const i32, cu: *const i32, nt: i32, ns: i32, bs: i32, bts: i32, nkv: i32, hs: i32, x: i32) { gather_kv_cache::<u8, B, true>(kc, vc, ko, vo, ks, vs, bt, cu, nt, ns, bs, bts, nkv, hs, x) }

    // ------------------------------------------------------------------------------------------
    // update_kvscales.cu: compute_and_update_scales_kernel<T>. Grid min(max(ceil(n/512),1),65535),
    // block 512, dynamic shared 2*512 floats.

    #[inline(always)]
    unsafe fn update_scales<T: El>(k: *const T, v: *const T, num_elements: i64, k_scales: *mut f32, v_scales: *mut f32) {
        let sdata = DynamicSharedArray::<f32>::get();
        let t = thread::threadIdx_x();
        let bdim = thread::blockDim_x();
        let s_k = sdata;
        let s_v = sdata.add(bdim as usize);
        let mut idx = (thread::blockIdx_x() as u64).wrapping_mul(bdim as u64).wrapping_add(t as u64) as i64;
        let stride = (bdim as u64).wrapping_mul(thread::gridDim_x() as u64) as i64;
        let mut mk = 0.0f32;
        let mut mv = 0.0f32;
        while idx < num_elements {
            let avk = fabs((*k.offset(idx as isize)).f());
            let avv = fabs((*v.offset(idx as isize)).f());
            if gt_ftz(avk, mk) {
                mk = avk;
            }
            if gt_ftz(avv, mv) {
                mv = avv;
            }
            idx = idx.wrapping_add(stride);
        }
        *s_k.add(t as usize) = mk;
        *s_v.add(t as usize) = mv;
        thread::sync_threads();
        let mut s = bdim >> 1;
        while s > 0 {
            if t < s {
                let ok = *s_k.add((t + s) as usize);
                if gt_ftz(ok, *s_k.add(t as usize)) {
                    *s_k.add(t as usize) = ok;
                }
                let ov = *s_v.add((t + s) as usize);
                if gt_ftz(ov, *s_v.add(t as usize)) {
                    *s_v.add(t as usize) = ov;
                }
            }
            thread::sync_threads();
            s >>= 1;
        }
        if t == 0 {
            let ck = fdiv(*s_k, 240.0);
            let cv = fdiv(*s_v, 240.0);
            if gt_ftz(ck, 0.0) {
                atomic_max_float(k_scales, ck);
            }
            if gt_ftz(cv, 0.0) {
                atomic_max_float(v_scales, cv);
            }
        }
    }
    /// atomicMaxFloat: CAS loop on the int bits (`atom.relaxed.sys.global.cas.b32`).
    #[inline(always)]
    unsafe fn atomic_max_float(address: *mut f32, val: f32) {
        let a = SystemAtomicI32::from_ptr(address as *mut i32);
        let mut old_i = *(address as *const i32);
        while gt_ftz(val, f32::from_bits(old_i as u32)) {
            let assumed = old_i;
            match a.compare_exchange(assumed, val.to_bits() as i32, AtomicOrdering::Relaxed, AtomicOrdering::Relaxed) {
                Ok(_) => return,
                Err(prev) => old_i = prev,
            }
        }
    }
    #[kernel] pub unsafe fn kvscales_f32(k: *const f32, v: *const f32, n: i64, ks: *mut f32, vs: *mut f32) { update_scales::<f32>(k, v, n, ks, vs) }
    #[kernel] pub unsafe fn kvscales_f16(k: *const H, v: *const H, n: i64, ks: *mut f32, vs: *mut f32) { update_scales::<H>(k, v, n, ks, vs) }
    #[kernel] pub unsafe fn kvscales_bf16(k: *const B, v: *const B, n: i64, ks: *mut f32, vs: *mut f32) { update_scales::<B>(k, v, n, ks, vs) }

    // ------------------------------------------------------------------------------------------
    // copy_blocks_kernel.cu: grid (num_layers, num_pairs), block min(max(key, value numel), 1024).

    #[inline(always)]
    unsafe fn copy_blocks<T: Copy>(key_cache_ptrs: *const i64, value_cache_ptrs: *const i64, block_mapping: *const i64, nk: i32, nv: i32) {
        let layer = thread::blockIdx_x() as usize;
        let pair = thread::blockIdx_y() as i32;
        let kc = *key_cache_ptrs.add(layer) as *mut T;
        let vc = *value_cache_ptrs.add(layer) as *mut T;
        let src = *block_mapping.offset((2 * pair) as isize);
        let dst = *block_mapping.offset((2 * pair + 1) as isize);
        let sk = src.wrapping_mul(nk as i64);
        let dk = dst.wrapping_mul(nk as i64);
        let mut i = tid();
        while i < nk {
            *kc.offset(dk.wrapping_add(i as i64) as isize) = *kc.offset(sk.wrapping_add(i as i64) as isize);
            i = i.wrapping_add(ntid());
        }
        let sv = src.wrapping_mul(nv as i64);
        let dv = dst.wrapping_mul(nv as i64);
        let mut i = tid();
        while i < nv {
            *vc.offset(dv.wrapping_add(i as i64) as isize) = *vc.offset(sv.wrapping_add(i as i64) as isize);
            i = i.wrapping_add(ntid());
        }
    }
    #[kernel] pub unsafe fn copy_blocks_kernel_f32(k: *const i64, v: *const i64, m: *const i64, nk: i32, nv: i32) { copy_blocks::<f32>(k, v, m, nk, nv) }
    #[kernel] pub unsafe fn copy_blocks_kernel_f16(k: *const i64, v: *const i64, m: *const i64, nk: i32, nv: i32) { copy_blocks::<i16>(k, v, m, nk, nv) }
    #[kernel] pub unsafe fn copy_blocks_kernel_bf16(k: *const i64, v: *const i64, m: *const i64, nk: i32, nv: i32) { copy_blocks::<i16>(k, v, m, nk, nv) }

    // ------------------------------------------------------------------------------------------
    // concat_and_cache_mla_kernel.cu: grid (num_tokens), block min(max(rank, kpe), 512).

    #[inline(always)]
    unsafe fn concat_and_cache_mla<T: Copy>(
        ckv: *const T, k_pe: *const T, ckv_cache: *mut T, kpe_cache: *mut T, slot_mapping: *const i64, ckv_stride: i32, kpe_stride: i32,
        kv_lora_rank: i32, kpe_head_dim: i32, block_size: i32,
    ) {
        let token_idx = thread::blockIdx_x() as i64;
        let slot_idx = *slot_mapping.offset(token_idx as isize);
        if slot_idx < 0 {
            return;
        }
        let (block_idx, block_offset) = divrem64(slot_idx, block_size as i64);
        let ckv_dst = block_idx.wrapping_mul(block_size as i64).wrapping_add(block_offset).wrapping_mul(kv_lora_rank as i64);
        let mut i = tid();
        while i < kv_lora_rank {
            let src = token_idx.wrapping_mul(ckv_stride as i64).wrapping_add(i as i64);
            *ckv_cache.offset(ckv_dst.wrapping_add(i as i64) as isize) = *ckv.offset(src as isize);
            i = i.wrapping_add(ntid());
        }
        let kpe_dst = block_idx.wrapping_mul(block_size as i64).wrapping_add(block_offset).wrapping_mul(kpe_head_dim as i64);
        let mut i = tid();
        while i < kpe_head_dim {
            let src = token_idx.wrapping_mul(kpe_stride as i64).wrapping_add(i as i64);
            *kpe_cache.offset(kpe_dst.wrapping_add(i as i64) as isize) = *k_pe.offset(src as isize);
            i = i.wrapping_add(ntid());
        }
    }
    #[kernel] pub unsafe fn ccmla_f32(c: *const f32, k: *const f32, cc: *mut f32, kc: *mut f32, s: *const i64, cs: i32, ks: i32, r: i32, d: i32, bs: i32) { concat_and_cache_mla::<f32>(c, k, cc, kc, s, cs, ks, r, d, bs) }
    #[kernel] pub unsafe fn ccmla_f16(c: *const u16, k: *const u16, cc: *mut u16, kc: *mut u16, s: *const i64, cs: i32, ks: i32, r: i32, d: i32, bs: i32) { concat_and_cache_mla::<u16>(c, k, cc, kc, s, cs, ks, r, d, bs) }
    #[kernel] pub unsafe fn ccmla_bf16(c: *const u16, k: *const u16, cc: *mut u16, kc: *mut u16, s: *const i64, cs: i32, ks: i32, r: i32, d: i32, bs: i32) { concat_and_cache_mla::<u16>(c, k, cc, kc, s, cs, ks, r, d, bs) }

    // ------------------------------------------------------------------------------------------
    // gather_mla_cache_kernel.cu: grid (num_tokens), block 256.

    #[inline(always)]
    unsafe fn gather_mla_cache<T: Copy>(
        ckv_cache: *const T, kpe_cache: *const T, ckv_out: *mut T, kpe_out: *mut T, block_table: *const i32, cu_seq_lens: *const i32,
        token_to_seq: *const i32, num_tokens: i32, block_size: i32, block_table_stride: i32, kv_lora_rank: i32, kpe_head_dim: i32,
    ) {
        let token_id = thread::blockIdx_x() as i32;
        if token_id >= num_tokens {
            return;
        }
        let batch_id = *token_to_seq.offset(token_id as isize);
        let batch_start = *cu_seq_lens.offset(batch_id as isize);
        let batch_offset = token_id.wrapping_sub(batch_start);
        let block_table_id = sdiv(batch_offset, block_size);
        let slot = batch_offset.wrapping_sub(block_table_id.wrapping_mul(block_size));
        let block_id = *block_table.offset(batch_id.wrapping_mul(block_table_stride).wrapping_add(block_table_id) as isize);
        let cache_base = (block_id as i64).wrapping_mul(block_size as i64).wrapping_add(slot as i64);
        let ckv_offset = cache_base.wrapping_mul(kv_lora_rank as i64);
        let kpe_offset = cache_base.wrapping_mul(kpe_head_dim as i64);
        let out_ckv = (token_id as i64).wrapping_mul(kv_lora_rank as i64);
        let out_kpe = (token_id as i64).wrapping_mul(kpe_head_dim as i64);
        let mut i = tid();
        while i < kv_lora_rank {
            *ckv_out.offset(out_ckv.wrapping_add(i as i64) as isize) = *ckv_cache.offset(ckv_offset.wrapping_add(i as i64) as isize);
            i = i.wrapping_add(ntid());
        }
        let mut i = tid();
        while i < kpe_head_dim {
            *kpe_out.offset(out_kpe.wrapping_add(i as i64) as isize) = *kpe_cache.offset(kpe_offset.wrapping_add(i as i64) as isize);
            i = i.wrapping_add(ntid());
        }
    }
    #[kernel] pub unsafe fn gmla_f32(c: *const f32, k: *const f32, co: *mut f32, ko: *mut f32, bt: *const i32, cu: *const i32, ts: *const i32, nt: i32, bs: i32, bts: i32, r: i32, d: i32) { gather_mla_cache::<f32>(c, k, co, ko, bt, cu, ts, nt, bs, bts, r, d) }
    #[kernel] pub unsafe fn gmla_f16(c: *const u16, k: *const u16, co: *mut u16, ko: *mut u16, bt: *const i32, cu: *const i32, ts: *const i32, nt: i32, bs: i32, bts: i32, r: i32, d: i32) { gather_mla_cache::<u16>(c, k, co, ko, bt, cu, ts, nt, bs, bts, r, d) }
    #[kernel] pub unsafe fn gmla_bf16(c: *const u16, k: *const u16, co: *mut u16, ko: *mut u16, bt: *const i32, cu: *const i32, ts: *const i32, nt: i32, bs: i32, bts: i32, r: i32, d: i32) { gather_mla_cache::<u16>(c, k, co, ko, bt, cu, ts, nt, bs, bts, r, d) }


    // ------------------------------------------------------------------------------------------
    // pagedattention.cuh: paged_attention_kernel<scalar_t, cache_t, kv_dt, HEAD_SIZE, BLOCK_SIZE,
    // NUM_THREADS = 128, PARTITION_SIZE> (v1: PARTITION_SIZE 0; v2: 512) and the v2 reduce kernel.

    pub trait Sc: El {
        /// sizeof(scalar_t)
        const SIZE: usize;
        /// V_VEC_SIZE = min(16 / sizeof(scalar_t), BLOCK_SIZE) (BLOCK_SIZE >= 8 here)
        const VV: usize;
        const ZERO: Self;
        /// A K element of an fp8 cache as the float the qk dot sees, when the K vector has `vec`
        /// elements (`scaled_convert<K_vec, Cache_K_vec>`).
        fn k_fp8(b: u8, scale: f32, vec: usize) -> f32;
        /// A V element of an fp8 cache (`scaled_convert<V_vec, Cache_V_vec>`).
        fn v_fp8(b: u8, scale: f32) -> Self;
        /// `dot(logits_vec, v_vec)` with `L_vec` from `from_float(Float_L_vec)`.
        fn v_dot(l: &[f32; 8], v: &[Self; 8]) -> f32;
    }
    impl Sc for f32 {
        const SIZE: usize = 4;
        const VV: usize = 4;
        const ZERO: f32 = 0.0;
        #[inline(always)]
        fn k_fp8(b: u8, scale: f32, vec: usize) -> f32 {
            let p = fmul(scale, h2f(fp8_to_h(b)));
            // float <- uint8: no rounding; float2/float4 go through half2 (quant_utils).
            if vec == 1 { p } else { h2f(f2h(p)) }
        }
        #[inline(always)]
        fn v_fp8(b: u8, scale: f32) -> f32 {
            h2f(f2h(fmul(scale, h2f(fp8_to_h(b)))))
        }
        #[inline(always)]
        fn v_dot(l: &[f32; 8], v: &[f32; 8]) -> f32 {
            let a = ffma(l[0], v[0], fmul(l[1], v[1]));
            let a = ffma(l[2], v[2], a);
            ffma(l[3], v[3], a)
        }
    }
    impl Sc for H {
        const SIZE: usize = 2;
        const VV: usize = 8;
        const ZERO: H = H(0);
        #[inline(always)]
        fn k_fp8(b: u8, scale: f32, _vec: usize) -> f32 {
            h2f(f2h(fmul(scale, h2f(fp8_to_h(b)))))
        }
        #[inline(always)]
        fn v_fp8(b: u8, scale: f32) -> H {
            H(f2h(fmul(scale, h2f(fp8_to_h(b)))))
        }
        #[inline(always)]
        fn v_dot(l: &[f32; 8], v: &[H; 8]) -> f32 {
            let w = |i: usize| (v[2 * i].0 as u32) | ((v[2 * i + 1].0 as u32) << 16);
            let p0 = hmul2(f2h2(l[0], l[1]), w(0));
            let p1 = hmul2(f2h2(l[2], l[3]), w(1));
            let p2 = hmul2(f2h2(l[4], l[5]), w(2));
            let p3 = hmul2(f2h2(l[6], l[7]), w(3));
            let s = hadd2(hadd2(hadd2(p0, p1), p2), p3);
            fadd(h2f(s as u16), h2f((s >> 16) as u16))
        }
    }
    impl Sc for B {
        const SIZE: usize = 2;
        const VV: usize = 8;
        const ZERO: B = B(0);
        #[inline(always)]
        fn k_fp8(b: u8, scale: f32, _vec: usize) -> f32 {
            bf2f(f2bf(fmul(scale, h2f(fp8_to_h(b)))))
        }
        #[inline(always)]
        fn v_fp8(b: u8, scale: f32) -> B {
            B(f2bf(fmul(scale, h2f(fp8_to_h(b)))))
        }
        #[inline(always)]
        fn v_dot(l: &[f32; 8], v: &[B; 8]) -> f32 {
            let w = |i: usize| (v[2 * i].0 as u32) | ((v[2 * i + 1].0 as u32) << 16);
            let s = |p: u32| fadd(bf2f(p as u16), bf2f((p >> 16) as u16));
            let s0 = s(bmul2(f2bf2(l[0], l[1]), w(0)));
            let s1 = s(bmul2(f2bf2(l[2], l[3]), w(1)));
            let s2 = s(bmul2(f2bf2(l[4], l[5]), w(2)));
            let s3 = s(bmul2(f2bf2(l[6], l[7]), w(3)));
            fadd(fadd(fadd(s0, s1), s2), s3)
        }
    }

    /// `q_vecs` static shared array of exactly HEAD_SIZE * sizeof(scalar_t) bytes (as nvcc sizes it).
    macro_rules! q_smem {
        ($bytes:expr; $($n:literal)*) => {
            match $bytes {
                $( $n => { static mut Q: SharedArray<u8, $n, 16> = SharedArray::UNINIT; SharedArray::as_raw_mut_ptr(&raw mut Q) } )*
                _ => core::ptr::null_mut(),
            }
        };
    }

    #[inline(always)]
    pub fn imin(a: i32, b: i32) -> i32 {
        if a < b { a } else { b }
    }

    /// block_sum<NUM_WARPS = 4>(red_smem, sum)
    #[inline(always)]
    unsafe fn block_sum4(red: *mut f32, mut sum: f32) -> f32 {
        let warp = (thread::threadIdx_x() / 32) as usize;
        let lane = thread::threadIdx_x() % 32;
        let mut m = 16;
        while m >= 1 {
            sum = fadd(sum, warp::shuffle_xor_f32(sum, m));
            m /= 2;
        }
        if lane == 0 {
            *red.add(warp) = sum;
        }
        thread::sync_threads();
        if lane < 4 {
            sum = *red.add(lane as usize);
        }
        sum = fadd(sum, warp::shuffle_xor_f32(sum, 2));
        sum = fadd(sum, warp::shuffle_xor_f32(sum, 1));
        warp::shuffle_f32(sum, 0)
    }

    #[inline(always)]
    unsafe fn paged_attention<S: Sc, const FP8: bool, const HEAD: usize, const BLOCK: usize, const PART: usize, const NRT: usize>(
        exp_sums: *mut f32, max_logits: *mut f32, out: *mut S, q: *const S, k_cache: *const u8, v_cache: *const u8, num_kv_heads: i32,
        scale: f32, softcapping: f32, block_tables: *const u32, context_lens: *const u32, max_num_blocks_per_seq: i32,
        alibi_slopes: *const f32, q_stride: i32, kv_block_stride: i32, kv_head_stride: i32, k_scale: *const f32, v_scale: *const f32,
        sinks: *const f32,
    ) {
        let tgs: usize = if 32 / BLOCK > 1 { 32 / BLOCK } else { 1 };
        let ntg: usize = 128 / tgs;
        let vec: usize = if 16 / (tgs * S::SIZE) > 1 { 16 / (tgs * S::SIZE) } else { 1 };
        let nvt: usize = HEAD / tgs / vec;
        let x: usize = if FP8 { 16 } else { 16 / S::SIZE };
        let vv: usize = S::VV;
        let nvpr: usize = BLOCK / vv;
        let nrpi: usize = 32 / nvpr;

        let seq_idx = thread::blockIdx_y() as i32;
        let partition_idx = thread::blockIdx_z() as i32;
        let max_num_partitions = thread::gridDim_z() as i32;
        let context_len = *context_lens.add(seq_idx as usize);
        if PART > 0 && (partition_idx.wrapping_mul(PART as i32) as u32) >= context_len {
            return;
        }
        let num_context_blocks = (context_len.wrapping_add(BLOCK as u32 - 1) / BLOCK as u32) as i32;
        let nbpp = if PART > 0 { (PART / BLOCK) as i32 } else { num_context_blocks };
        let start_block_idx = if PART > 0 { partition_idx.wrapping_mul(nbpp) } else { 0 };
        let end_block_idx = imin(start_block_idx.wrapping_add(nbpp), num_context_blocks);
        let num_blocks = end_block_idx.wrapping_sub(start_block_idx);
        let start_token_idx = start_block_idx.wrapping_mul(BLOCK as i32);
        let e = start_token_idx.wrapping_add(num_blocks.wrapping_mul(BLOCK as i32)) as u32;
        let end_token_idx = (if e < context_len { e } else { context_len }) as i32;
        let num_tokens = end_token_idx.wrapping_sub(start_token_idx);

        let thread_idx = thread::threadIdx_x() as i32;
        let warp_idx = thread_idx / 32;
        let lane = thread_idx % 32;
        let head_idx = thread::blockIdx_x() as i32;
        let num_heads = thread::gridDim_x() as i32;
        let num_queries_per_kv = sdiv(num_heads, num_kv_heads);
        let kv_head_idx = sdiv(head_idx, num_queries_per_kv);
        let alibi_slope = if alibi_slopes.is_null() { 0.0f32 } else { *alibi_slopes.add(head_idx as usize) };

        let thread_group_idx = thread_idx as usize / tgs;
        let thread_group_offset = thread_idx as usize % tgs;

        // q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD] (raw scalar_t elements)
        let q_s = q_smem!(HEAD * S::SIZE; 128 160 192 224 256 320 384 448 512 768 1024 2048) as *mut S;
        let q_ptr = q.offset((seq_idx as i64).wrapping_mul(q_stride as i64) as isize).add(head_idx as usize * HEAD);
        let mut i = thread_group_idx;
        while i < nvt {
            let vec_idx = thread_group_offset + i * tgs;
            let mut l = 0;
            while l < vec {
                *q_s.add((thread_group_offset * nvt + i) * vec + l) = *q_ptr.add(vec_idx * vec + l);
                l += 1;
            }
            i += ntg;
        }
        thread::sync_threads();

        let logits = DynamicSharedArray::<f32>::get();
        static mut RED: SharedArray<f32, 8> = SharedArray::UNINIT;
        let red = SharedArray::as_raw_mut_ptr(&raw mut RED);
        let mut qk_max = f32::from_bits(0xff7fffff);

        let block_table = block_tables.offset(max_num_blocks_per_seq.wrapping_mul(seq_idx) as isize);
        let kc = k_cache as *const S;
        let qrow = q_s.add(thread_group_offset * nvt * vec);
        let mut block_idx = start_block_idx.wrapping_add(warp_idx);
        while block_idx < end_block_idx {
            let pbn = *block_table.offset(block_idx as isize) as i64;
            let pbo = (thread_group_idx % BLOCK) as i32;
            let token_idx = block_idx.wrapping_mul(BLOCK as i32).wrapping_add(pbo);
            let k_off = pbn
                .wrapping_mul(kv_block_stride as i64)
                .wrapping_add((kv_head_idx as i64).wrapping_mul(kv_head_stride as i64))
                .wrapping_add((pbo as i64) * x as i64);
            // element j*vec + l of this thread's K vectors
            let kel = |j: usize, l: usize| -> f32 {
                let vi = (thread_group_offset + j * tgs) * vec;
                let o = k_off + ((vi / x) * BLOCK * x + vi % x + l) as i64;
                if FP8 { S::k_fp8(*k_cache.offset(o as isize), *k_scale, vec) } else { (*kc.offset(o as isize)).f() }
            };
            let qel = |j: usize, l: usize| -> f32 { (*qrow.add(j * vec + l)).f() };
            let mut acc = [0.0f32; 8];
            let mut l = 0;
            while l < vec {
                acc[l] = ffma(qel(0, l), kel(0, l), fmul(qel(1, l), kel(1, l)));
                l += 1;
            }
            let mut j = 2;
            while j < nvt {
                let mut l = 0;
                while l < vec {
                    acc[l] = ffma(qel(j, l), kel(j, l), acc[l]);
                    l += 1;
                }
                j += 1;
            }
            let mut qk = acc[0];
            let mut l = 1;
            while l < vec {
                qk = fadd(qk, acc[l]);
                l += 1;
            }
            let mut m = tgs / 2;
            while m >= 1 {
                qk = fadd(qk, warp::shuffle_xor_f32(qk, m as u32));
                m /= 2;
            }
            let mut qk = fmul(scale, qk);
            if !eq_ftz(softcapping, 1.0) {
                qk = fmul(softcapping, tanh_approx(fdiv(qk, softcapping)));
            }
            let bias = if neu_ftz(alibi_slope, 0.0) {
                fmul(alibi_slope, u2f((token_idx as u32).wrapping_sub(context_len).wrapping_add(1)))
            } else {
                0.0
            };
            let qk = fadd(bias, qk);
            if thread_group_offset == 0 {
                let mask = (token_idx as u32) >= context_len;
                *logits.offset(token_idx.wrapping_sub(start_token_idx) as isize) = if mask { 0.0 } else { qk };
                qk_max = if mask { qk_max } else { fmax(qk_max, qk) };
            }
            block_idx = block_idx.wrapping_add(4);
        }

        let mut m = 16;
        while m >= tgs as u32 {
            qk_max = fmax(qk_max, warp::shuffle_xor_f32(qk_max, m));
            m /= 2;
        }
        if lane == 0 {
            *red.add(warp_idx as usize) = qk_max;
        }
        thread::sync_threads();
        qk_max = if lane < 4 { *red.add(lane as usize) } else { f32::from_bits(0xff7fffff) };
        qk_max = fmax(qk_max, warp::shuffle_xor_f32(qk_max, 2));
        qk_max = fmax(qk_max, warp::shuffle_xor_f32(qk_max, 1));
        qk_max = warp::shuffle_f32(qk_max, 0);
        if PART == 0 && !sinks.is_null() {
            qk_max = fmax(qk_max, *sinks.add(head_idx as usize));
        }

        let mut exp_sum = 0.0f32;
        let mut i = thread_idx;
        while i < num_tokens {
            let val = fexp(fsub(*logits.offset(i as isize), qk_max));
            *logits.offset(i as isize) = val;
            exp_sum = fadd(exp_sum, val);
            i += 128;
        }
        exp_sum = block_sum4(red.add(4), exp_sum);
        if PART == 0 && !sinks.is_null() {
            exp_sum = fadd(fexp(fsub(*sinks.add(head_idx as usize), qk_max)), exp_sum);
        }
        let inv_sum = fdiv(1.0, fadd(exp_sum, f32::from_bits(0x358637BD)));
        let mut i = thread_idx;
        while i < num_tokens {
            *logits.offset(i as isize) = fmul(inv_sum, *logits.offset(i as isize));
            i += 128;
        }
        thread::sync_threads();

        if PART > 0 && thread_idx == 0 {
            let o = seq_idx
                .wrapping_mul(num_heads)
                .wrapping_mul(max_num_partitions)
                .wrapping_add(head_idx.wrapping_mul(max_num_partitions))
                .wrapping_add(partition_idx);
            *max_logits.offset(o as isize) = qk_max;
            *exp_sums.offset(o as isize) = exp_sum;
        }

        let mut accs = [0.0f32; NRT];
        let vc = v_cache as *const S;
        let mut block_idx = start_block_idx.wrapping_add(warp_idx);
        while block_idx < end_block_idx {
            let pbn = *block_table.offset(block_idx as isize) as i64;
            let pbo = ((lane as usize % nvpr) * vv) as i32;
            let token_idx = block_idx.wrapping_mul(BLOCK as i32).wrapping_add(pbo);
            let mut lv = [0.0f32; 8];
            let lp = logits.offset(token_idx.wrapping_sub(start_token_idx) as isize);
            let mut j = 0;
            while j < vv {
                lv[j] = *lp.add(j);
                j += 1;
            }
            let v_off = pbn.wrapping_mul(kv_block_stride as i64).wrapping_add((kv_head_idx as i64).wrapping_mul(kv_head_stride as i64));
            let mut r = 0;
            while r < NRT {
                let row = lane as usize / nvpr + r * nrpi;
                if row < HEAD {
                    let o = v_off + (row * BLOCK) as i64 + pbo as i64;
                    let mut v = [S::ZERO; 8];
                    let mut j = 0;
                    while j < vv {
                        v[j] = if FP8 { S::v_fp8(*v_cache.offset(o as isize + j as isize), *v_scale) } else { *vc.offset(o as isize + j as isize) };
                        j += 1;
                    }
                    if block_idx == num_context_blocks.wrapping_sub(1) {
                        let mut j = 0;
                        while j < vv {
                            if !((token_idx.wrapping_add(j as i32) as u32) < context_len) {
                                v[j] = S::ZERO;
                            }
                            j += 1;
                        }
                    }
                    accs[r] = fadd(accs[r], S::v_dot(&lv, &v));
                }
                r += 1;
            }
            block_idx = block_idx.wrapping_add(4);
        }

        let mut r = 0;
        while r < NRT {
            let mut acc = accs[r];
            let mut m = nvpr / 2;
            while m >= 1 {
                acc = fadd(acc, warp::shuffle_xor_f32(acc, m as u32));
                m /= 2;
            }
            accs[r] = acc;
            r += 1;
        }
        thread::sync_threads();

        let out_smem = logits;
        let lane_ok = lane as usize % nvpr == 0;
        let mut nw = 4;
        while nw > 1 {
            let mid = nw / 2;
            if warp_idx >= mid && warp_idx < nw {
                let dst = out_smem.add((warp_idx - mid) as usize * HEAD);
                let mut r = 0;
                while r < NRT {
                    let row = lane as usize / nvpr + r * nrpi;
                    if row < HEAD && lane_ok {
                        *dst.add(row) = accs[r];
                    }
                    r += 1;
                }
            }
            thread::sync_threads();
            if warp_idx < mid {
                let src = out_smem.add(warp_idx as usize * HEAD);
                let mut r = 0;
                while r < NRT {
                    let row = lane as usize / nvpr + r * nrpi;
                    if row < HEAD && lane_ok {
                        accs[r] = fadd(*src.add(row), accs[r]);
                    }
                    r += 1;
                }
            }
            thread::sync_threads();
            nw /= 2;
        }

        if warp_idx == 0 {
            let o = seq_idx
                .wrapping_mul(num_heads)
                .wrapping_mul(max_num_partitions)
                .wrapping_mul(HEAD as i32)
                .wrapping_add(head_idx.wrapping_mul(max_num_partitions).wrapping_mul(HEAD as i32))
                .wrapping_add(partition_idx.wrapping_mul(HEAD as i32));
            let out_ptr = out.offset(o as isize);
            let mut r = 0;
            while r < NRT {
                let row = lane as usize / nvpr + r * nrpi;
                if row < HEAD && lane_ok {
                    *out_ptr.add(row) = S::from_f(accs[r]);
                }
                r += 1;
            }
        }
    }

    #[inline(always)]
    unsafe fn paged_attention_v2_reduce<S: Sc, const HEAD: usize>(
        out: *mut S, exp_sums: *const f32, max_logits: *const f32, tmp_out: *const S, context_lens: *const u32, max_num_partitions: i32,
        sinks: *const f32,
    ) {
        let num_heads = thread::gridDim_x() as i32;
        let head_idx = thread::blockIdx_x() as i32;
        let seq_idx = thread::blockIdx_y() as i32;
        let context_len = *context_lens.add(seq_idx as usize);
        let num_partitions = (context_len.wrapping_add(511) / 512) as i32;
        let t = thread::threadIdx_x() as i32;
        let bd = thread::blockDim_x() as i32;
        if num_partitions == 1 && sinks.is_null() {
            let out_ptr = out.offset(seq_idx.wrapping_mul(num_heads).wrapping_mul(HEAD as i32).wrapping_add(head_idx.wrapping_mul(HEAD as i32)) as isize);
            let tmp_ptr = tmp_out.offset(
                seq_idx
                    .wrapping_mul(num_heads)
                    .wrapping_mul(max_num_partitions)
                    .wrapping_mul(HEAD as i32)
                    .wrapping_add(head_idx.wrapping_mul(max_num_partitions).wrapping_mul(HEAD as i32)) as isize,
            );
            let mut i = t;
            while i < HEAD as i32 {
                *out_ptr.offset(i as isize) = *tmp_ptr.offset(i as isize);
                i = i.wrapping_add(bd);
            }
            return;
        }
        let warp_idx = t / 32;
        let lane = t % 32;
        let smem = DynamicSharedArray::<f32>::get();
        static mut RED: SharedArray<f32, 8> = SharedArray::UNINIT;
        let red = SharedArray::as_raw_mut_ptr(&raw mut RED);
        let shared_max_logits = smem;
        let base = seq_idx.wrapping_mul(num_heads).wrapping_mul(max_num_partitions).wrapping_add(head_idx.wrapping_mul(max_num_partitions));
        let max_logits_ptr = max_logits.offset(base as isize);
        let mut max_logit = f32::from_bits(0xff7fffff);
        let mut i = t;
        while i < num_partitions {
            let l = *max_logits_ptr.offset(i as isize);
            *shared_max_logits.offset(i as isize) = l;
            max_logit = fmax(max_logit, l);
            i = i.wrapping_add(bd);
        }
        thread::sync_threads();
        let mut m = 16;
        while m >= 1 {
            max_logit = fmax(max_logit, warp::shuffle_xor_f32(max_logit, m));
            m /= 2;
        }
        if lane == 0 {
            *red.add(warp_idx as usize) = max_logit;
        }
        thread::sync_threads();
        max_logit = if lane < 4 { *red.add(lane as usize) } else { f32::from_bits(0xff7fffff) };
        max_logit = fmax(max_logit, warp::shuffle_xor_f32(max_logit, 2));
        max_logit = fmax(max_logit, warp::shuffle_xor_f32(max_logit, 1));
        max_logit = warp::shuffle_f32(max_logit, 0);
        if !sinks.is_null() {
            max_logit = fmax(max_logit, *sinks.add(head_idx as usize));
        }
        let shared_exp_sums = smem.offset(num_partitions as isize);
        let exp_sums_ptr = exp_sums.offset(base as isize);
        let mut global_exp_sum = 0.0f32;
        let mut i = t;
        while i < num_partitions {
            let l = *shared_max_logits.offset(i as isize);
            let r = fmul(*exp_sums_ptr.offset(i as isize), fexp(fsub(l, max_logit)));
            global_exp_sum = fadd(global_exp_sum, r);
            *shared_exp_sums.offset(i as isize) = r;
            i = i.wrapping_add(bd);
        }
        thread::sync_threads();
        global_exp_sum = block_sum4(red.add(4), global_exp_sum);
        if !sinks.is_null() {
            global_exp_sum = fadd(fexp(fsub(*sinks.add(head_idx as usize), max_logit)), global_exp_sum);
        }
        let inv = fdiv(1.0, fadd(global_exp_sum, f32::from_bits(0x358637BD)));
        let tmp_ptr = tmp_out.offset(
            seq_idx
                .wrapping_mul(num_heads)
                .wrapping_mul(max_num_partitions)
                .wrapping_mul(HEAD as i32)
                .wrapping_add(head_idx.wrapping_mul(max_num_partitions).wrapping_mul(HEAD as i32)) as isize,
        );
        let out_ptr = out.offset(seq_idx.wrapping_mul(num_heads).wrapping_mul(HEAD as i32).wrapping_add(head_idx.wrapping_mul(HEAD as i32)) as isize);
        let mut i = t;
        while i < HEAD as i32 {
            let mut acc = 0.0f32;
            let mut j = 0;
            while j < num_partitions {
                let tv = (*tmp_ptr.offset(j.wrapping_mul(HEAD as i32).wrapping_add(i) as isize)).f();
                acc = ffma(inv, fmul(tv, *shared_exp_sums.offset(j as isize)), acc);
                j += 1;
            }
            *out_ptr.offset(i as isize) = S::from_f(acc);
            i += 128;
        }
    }


    // ------------------------------------------------------------------------------------------
    // flash_attn_sinks.cu: flash_attn_sinks{,_varlen}_kernel<scalar_t, HEAD_DIM, BR = 8, BC>.
    // Grid (num_heads, batch, cdiv(q_len, 8)), block 256.

    macro_rules! kv_smem {
        ($floats:expr; $($n:literal)*) => {
            match $floats {
                $( $n => {
                    static mut KS: SharedArray<f32, $n, 16> = SharedArray::UNINIT;
                    static mut VS: SharedArray<f32, $n, 16> = SharedArray::UNINIT;
                    (SharedArray::as_raw_mut_ptr(&raw mut KS), SharedArray::as_raw_mut_ptr(&raw mut VS))
                } )*
                _ => (core::ptr::null_mut(), core::ptr::null_mut()),
            }
        };
    }

    #[inline(always)]
    pub fn imax(a: i32, b: i32) -> i32 {
        if a > b { a } else { b }
    }

    /// `VARLEN`: q_len_or_max = max_q_len, kv_len ignored, cu_seqlens given.
    #[inline(always)]
    unsafe fn flash_attn_sinks<S: El, const HEAD: usize, const BC: usize, const EPT: usize, const VARLEN: bool>(
        Q: *const S, K: *const S, V: *const S, O: *mut S, sinks: *const f32, cu_q: *const u32, cu_k: *const u32, scale: f32, q_len_arg: i32,
        kv_len_arg: i32, num_heads: i32, num_kv_heads: i32, window_size: i32,
    ) {
        let d_pad = EPT * 32;
        let (k_smem, v_smem) = kv_smem!(BC * d_pad; 3072 4096);
        let warp_id = (thread::threadIdx_x() / 32) as i32;
        let lane_id = (thread::threadIdx_x() % 32) as i32;
        let head_idx = thread::blockIdx_x() as i32;
        let batch_idx = thread::blockIdx_y() as i32;
        let q_tile_idx = thread::blockIdx_z() as i32;
        let gqa_ratio = sdiv(num_heads, num_kv_heads);
        let kv_head_idx = sdiv(head_idx, gqa_ratio);
        let (q_len, kv_len, kv_start) = if VARLEN {
            let b = batch_idx as usize;
            ((*cu_q.add(b + 1)).wrapping_sub(*cu_q.add(b)) as i32, (*cu_k.add(b + 1)).wrapping_sub(*cu_k.add(b)) as i32, *cu_k.add(b) as i32)
        } else {
            (q_len_arg, kv_len_arg, 0)
        };
        let q_rows = if VARLEN { q_len_arg } else { q_len };
        let q_row = q_tile_idx.wrapping_mul(8).wrapping_add(warp_id);
        let valid = q_row < q_len;
        let kv_offset = kv_len.wrapping_sub(q_len);
        let q_offset = batch_idx.wrapping_mul(num_heads).wrapping_add(head_idx).wrapping_mul(q_rows).wrapping_add(q_row).wrapping_mul(HEAD as i32);
        let kv_base = batch_idx.wrapping_mul(num_kv_heads).wrapping_add(kv_head_idx).wrapping_mul(kv_len).wrapping_mul(HEAD as i32);

        let mut q_reg = [0.0f32; EPT];
        let mut i = 0;
        while i < EPT {
            let d = (i * 32) as i32 + lane_id;
            q_reg[i] = if valid && d < HEAD as i32 { fmul(scale, (*Q.offset(q_offset.wrapping_add(d) as isize)).f()) } else { 0.0 };
            i += 1;
        }
        let mut o_acc = [0.0f32; EPT];
        let neg_max = f32::from_bits(0xff7fffff);
        let mut m_i = neg_max;
        let mut l_i = 0.0f32;

        let block_q_start = q_tile_idx.wrapping_mul(8);
        let block_q_end = imin(block_q_start.wrapping_add(8), q_len);
        let block_kv_start = if window_size > 0 { imax(0, block_q_start.wrapping_add(kv_offset).wrapping_sub(window_size).wrapping_add(1)) } else { 0 };
        let block_kv_end = imin(kv_len, block_q_end.wrapping_add(kv_offset));
        let my_kv_start = if window_size > 0 && (VARLEN || valid) { imax(0, q_row.wrapping_add(kv_offset).wrapping_sub(window_size).wrapping_add(1)) } else { 0 };
        let my_kv_end = if VARLEN || valid { q_row.wrapping_add(kv_offset).wrapping_add(1) } else { 0 };

        let mut scores = [0.0f32; BC];
        let t = thread::threadIdx_x() as i32;
        let mut tile_start = block_kv_start;
        while tile_start < block_kv_end {
            let tile_end = imin(tile_start.wrapping_add(BC as i32), block_kv_end);
            let tile_len = tile_end.wrapping_sub(tile_start);
            let src = |p: *const S, j: i32, d: i32| -> f32 {
                let o = if VARLEN {
                    kv_start.wrapping_add(tile_start).wrapping_add(j).wrapping_mul(num_kv_heads).wrapping_add(kv_head_idx).wrapping_mul(HEAD as i32).wrapping_add(d)
                } else {
                    kv_base.wrapping_add(tile_start.wrapping_add(j).wrapping_mul(HEAD as i32)).wrapping_add(d)
                };
                (*p.offset(o as isize)).f()
            };
            let mut idx = t;
            while idx < (BC * d_pad) as i32 {
                let kj = idx / d_pad as i32;
                let kd = idx % d_pad as i32;
                *k_smem.add(idx as usize) = if kj < tile_len && kd < HEAD as i32 { src(K, kj, kd) } else { 0.0 };
                idx += 256;
            }
            let mut idx = t;
            while idx < (BC * d_pad) as i32 {
                let vj = idx / d_pad as i32;
                let vd = idx % d_pad as i32;
                *v_smem.add(idx as usize) = if vj < tile_len && vd < HEAD as i32 { src(V, vj, vd) } else { 0.0 };
                idx += 256;
            }
            thread::sync_threads();

            if valid {
                let mut tile_max = neg_max;
                let mut j = 0i32;
                while j < tile_len {
                    let kv_pos = tile_start.wrapping_add(j);
                    if kv_pos >= my_kv_end {
                        let mut jj = j;
                        while jj < tile_len {
                            scores[jj as usize] = neg_max;
                            jj += 1;
                        }
                        break;
                    }
                    if kv_pos < my_kv_start {
                        scores[j as usize] = neg_max;
                        j += 1;
                        continue;
                    }
                    let mut dot = 0.0f32;
                    let mut i = 0;
                    while i < EPT {
                        dot = ffma(q_reg[i], *k_smem.add(j as usize * d_pad + i * 32 + lane_id as usize), dot);
                        i += 1;
                    }
                    let mut m = 16;
                    while m > 0 {
                        dot = fadd(dot, warp::shuffle_xor_f32(dot, m));
                        m >>= 1;
                    }
                    scores[j as usize] = dot;
                    tile_max = fmax(tile_max, dot);
                    j += 1;
                }
                if gt_ftz(tile_max, neg_max) {
                    let m_new = fmax(m_i, tile_max);
                    let rescale = fexp(fsub(m_i, m_new));
                    let mut i = 0;
                    while i < EPT {
                        o_acc[i] = fmul(rescale, o_acc[i]);
                        i += 1;
                    }
                    l_i = fmul(l_i, rescale);
                    m_i = m_new;
                    let mut j = 0i32;
                    while j < tile_len {
                        let s = scores[j as usize];
                        if le_ftz(s, neg_max) {
                            j += 1;
                            continue;
                        }
                        let p = fexp(fsub(s, m_i));
                        l_i = fadd(l_i, p);
                        let mut i = 0;
                        while i < EPT {
                            o_acc[i] = ffma(p, *v_smem.add(j as usize * d_pad + i * 32 + lane_id as usize), o_acc[i]);
                            i += 1;
                        }
                        j += 1;
                    }
                }
            }
            thread::sync_threads();
            tile_start = tile_start.wrapping_add(BC as i32);
        }
        if !valid {
            return;
        }
        if !sinks.is_null() {
            let sink_val = *sinks.add(head_idx as usize);
            let m_new = fmax(m_i, sink_val);
            let rescale = fexp(fsub(m_i, m_new));
            let mut i = 0;
            while i < EPT {
                o_acc[i] = fmul(rescale, o_acc[i]);
                i += 1;
            }
            l_i = ffma(l_i, rescale, fexp(fsub(sink_val, m_new)));
        }
        let inv_l = if gt_ftz(l_i, 0.0) { rcp_approx(l_i) } else { 0.0 };
        let mut i = 0;
        while i < EPT {
            let d = (i * 32) as i32 + lane_id;
            if d < HEAD as i32 {
                *O.offset(q_offset.wrapping_add(d) as isize) = S::from_f(fmul(inv_l, o_acc[i]));
            }
            i += 1;
        }
    }
    #[inline(always)]
    pub fn rcp_approx(a: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("rcp.approx.ftz.f32 %0, %1;", out("=f") r, in("f") a, options(register_only)) };
        r
    }

    // GENERATED flash_attn_sinks kernels BEGIN
    #[kernel] pub unsafe fn fas_f32_64(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, sc: f32, ql: i32, kl: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<f32, 64, 64, 2, false>(q as _, k as _, v as _, o as _, sk, core::ptr::null(), core::ptr::null(), sc, ql, kl, nh, nkv, w) }
    #[kernel] pub unsafe fn fasv_f32_64(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, cq: *const u32, ck: *const u32, sc: f32, mq: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<f32, 64, 64, 2, true>(q as _, k as _, v as _, o as _, sk, cq, ck, sc, mq, 0, nh, nkv, w) }
    #[kernel] pub unsafe fn fas_f32_80(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, sc: f32, ql: i32, kl: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<f32, 80, 32, 3, false>(q as _, k as _, v as _, o as _, sk, core::ptr::null(), core::ptr::null(), sc, ql, kl, nh, nkv, w) }
    #[kernel] pub unsafe fn fasv_f32_80(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, cq: *const u32, ck: *const u32, sc: f32, mq: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<f32, 80, 32, 3, true>(q as _, k as _, v as _, o as _, sk, cq, ck, sc, mq, 0, nh, nkv, w) }
    #[kernel] pub unsafe fn fas_f32_96(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, sc: f32, ql: i32, kl: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<f32, 96, 32, 3, false>(q as _, k as _, v as _, o as _, sk, core::ptr::null(), core::ptr::null(), sc, ql, kl, nh, nkv, w) }
    #[kernel] pub unsafe fn fasv_f32_96(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, cq: *const u32, ck: *const u32, sc: f32, mq: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<f32, 96, 32, 3, true>(q as _, k as _, v as _, o as _, sk, cq, ck, sc, mq, 0, nh, nkv, w) }
    #[kernel] pub unsafe fn fas_f32_112(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, sc: f32, ql: i32, kl: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<f32, 112, 32, 4, false>(q as _, k as _, v as _, o as _, sk, core::ptr::null(), core::ptr::null(), sc, ql, kl, nh, nkv, w) }
    #[kernel] pub unsafe fn fasv_f32_112(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, cq: *const u32, ck: *const u32, sc: f32, mq: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<f32, 112, 32, 4, true>(q as _, k as _, v as _, o as _, sk, cq, ck, sc, mq, 0, nh, nkv, w) }
    #[kernel] pub unsafe fn fas_f32_128(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, sc: f32, ql: i32, kl: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<f32, 128, 32, 4, false>(q as _, k as _, v as _, o as _, sk, core::ptr::null(), core::ptr::null(), sc, ql, kl, nh, nkv, w) }
    #[kernel] pub unsafe fn fasv_f32_128(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, cq: *const u32, ck: *const u32, sc: f32, mq: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<f32, 128, 32, 4, true>(q as _, k as _, v as _, o as _, sk, cq, ck, sc, mq, 0, nh, nkv, w) }
    #[kernel] pub unsafe fn fas_f32_192(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, sc: f32, ql: i32, kl: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<f32, 192, 16, 6, false>(q as _, k as _, v as _, o as _, sk, core::ptr::null(), core::ptr::null(), sc, ql, kl, nh, nkv, w) }
    #[kernel] pub unsafe fn fasv_f32_192(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, cq: *const u32, ck: *const u32, sc: f32, mq: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<f32, 192, 16, 6, true>(q as _, k as _, v as _, o as _, sk, cq, ck, sc, mq, 0, nh, nkv, w) }
    #[kernel] pub unsafe fn fas_f32_256(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, sc: f32, ql: i32, kl: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<f32, 256, 16, 8, false>(q as _, k as _, v as _, o as _, sk, core::ptr::null(), core::ptr::null(), sc, ql, kl, nh, nkv, w) }
    #[kernel] pub unsafe fn fasv_f32_256(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, cq: *const u32, ck: *const u32, sc: f32, mq: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<f32, 256, 16, 8, true>(q as _, k as _, v as _, o as _, sk, cq, ck, sc, mq, 0, nh, nkv, w) }
    #[kernel] pub unsafe fn fas_f16_64(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, sc: f32, ql: i32, kl: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<H, 64, 64, 2, false>(q as _, k as _, v as _, o as _, sk, core::ptr::null(), core::ptr::null(), sc, ql, kl, nh, nkv, w) }
    #[kernel] pub unsafe fn fasv_f16_64(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, cq: *const u32, ck: *const u32, sc: f32, mq: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<H, 64, 64, 2, true>(q as _, k as _, v as _, o as _, sk, cq, ck, sc, mq, 0, nh, nkv, w) }
    #[kernel] pub unsafe fn fas_f16_80(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, sc: f32, ql: i32, kl: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<H, 80, 32, 3, false>(q as _, k as _, v as _, o as _, sk, core::ptr::null(), core::ptr::null(), sc, ql, kl, nh, nkv, w) }
    #[kernel] pub unsafe fn fasv_f16_80(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, cq: *const u32, ck: *const u32, sc: f32, mq: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<H, 80, 32, 3, true>(q as _, k as _, v as _, o as _, sk, cq, ck, sc, mq, 0, nh, nkv, w) }
    #[kernel] pub unsafe fn fas_f16_96(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, sc: f32, ql: i32, kl: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<H, 96, 32, 3, false>(q as _, k as _, v as _, o as _, sk, core::ptr::null(), core::ptr::null(), sc, ql, kl, nh, nkv, w) }
    #[kernel] pub unsafe fn fasv_f16_96(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, cq: *const u32, ck: *const u32, sc: f32, mq: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<H, 96, 32, 3, true>(q as _, k as _, v as _, o as _, sk, cq, ck, sc, mq, 0, nh, nkv, w) }
    #[kernel] pub unsafe fn fas_f16_112(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, sc: f32, ql: i32, kl: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<H, 112, 32, 4, false>(q as _, k as _, v as _, o as _, sk, core::ptr::null(), core::ptr::null(), sc, ql, kl, nh, nkv, w) }
    #[kernel] pub unsafe fn fasv_f16_112(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, cq: *const u32, ck: *const u32, sc: f32, mq: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<H, 112, 32, 4, true>(q as _, k as _, v as _, o as _, sk, cq, ck, sc, mq, 0, nh, nkv, w) }
    #[kernel] pub unsafe fn fas_f16_128(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, sc: f32, ql: i32, kl: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<H, 128, 32, 4, false>(q as _, k as _, v as _, o as _, sk, core::ptr::null(), core::ptr::null(), sc, ql, kl, nh, nkv, w) }
    #[kernel] pub unsafe fn fasv_f16_128(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, cq: *const u32, ck: *const u32, sc: f32, mq: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<H, 128, 32, 4, true>(q as _, k as _, v as _, o as _, sk, cq, ck, sc, mq, 0, nh, nkv, w) }
    #[kernel] pub unsafe fn fas_f16_192(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, sc: f32, ql: i32, kl: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<H, 192, 16, 6, false>(q as _, k as _, v as _, o as _, sk, core::ptr::null(), core::ptr::null(), sc, ql, kl, nh, nkv, w) }
    #[kernel] pub unsafe fn fasv_f16_192(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, cq: *const u32, ck: *const u32, sc: f32, mq: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<H, 192, 16, 6, true>(q as _, k as _, v as _, o as _, sk, cq, ck, sc, mq, 0, nh, nkv, w) }
    #[kernel] pub unsafe fn fas_f16_256(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, sc: f32, ql: i32, kl: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<H, 256, 16, 8, false>(q as _, k as _, v as _, o as _, sk, core::ptr::null(), core::ptr::null(), sc, ql, kl, nh, nkv, w) }
    #[kernel] pub unsafe fn fasv_f16_256(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, cq: *const u32, ck: *const u32, sc: f32, mq: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<H, 256, 16, 8, true>(q as _, k as _, v as _, o as _, sk, cq, ck, sc, mq, 0, nh, nkv, w) }
    #[kernel] pub unsafe fn fas_bf16_64(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, sc: f32, ql: i32, kl: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<B, 64, 64, 2, false>(q as _, k as _, v as _, o as _, sk, core::ptr::null(), core::ptr::null(), sc, ql, kl, nh, nkv, w) }
    #[kernel] pub unsafe fn fasv_bf16_64(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, cq: *const u32, ck: *const u32, sc: f32, mq: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<B, 64, 64, 2, true>(q as _, k as _, v as _, o as _, sk, cq, ck, sc, mq, 0, nh, nkv, w) }
    #[kernel] pub unsafe fn fas_bf16_80(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, sc: f32, ql: i32, kl: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<B, 80, 32, 3, false>(q as _, k as _, v as _, o as _, sk, core::ptr::null(), core::ptr::null(), sc, ql, kl, nh, nkv, w) }
    #[kernel] pub unsafe fn fasv_bf16_80(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, cq: *const u32, ck: *const u32, sc: f32, mq: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<B, 80, 32, 3, true>(q as _, k as _, v as _, o as _, sk, cq, ck, sc, mq, 0, nh, nkv, w) }
    #[kernel] pub unsafe fn fas_bf16_96(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, sc: f32, ql: i32, kl: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<B, 96, 32, 3, false>(q as _, k as _, v as _, o as _, sk, core::ptr::null(), core::ptr::null(), sc, ql, kl, nh, nkv, w) }
    #[kernel] pub unsafe fn fasv_bf16_96(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, cq: *const u32, ck: *const u32, sc: f32, mq: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<B, 96, 32, 3, true>(q as _, k as _, v as _, o as _, sk, cq, ck, sc, mq, 0, nh, nkv, w) }
    #[kernel] pub unsafe fn fas_bf16_112(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, sc: f32, ql: i32, kl: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<B, 112, 32, 4, false>(q as _, k as _, v as _, o as _, sk, core::ptr::null(), core::ptr::null(), sc, ql, kl, nh, nkv, w) }
    #[kernel] pub unsafe fn fasv_bf16_112(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, cq: *const u32, ck: *const u32, sc: f32, mq: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<B, 112, 32, 4, true>(q as _, k as _, v as _, o as _, sk, cq, ck, sc, mq, 0, nh, nkv, w) }
    #[kernel] pub unsafe fn fas_bf16_128(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, sc: f32, ql: i32, kl: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<B, 128, 32, 4, false>(q as _, k as _, v as _, o as _, sk, core::ptr::null(), core::ptr::null(), sc, ql, kl, nh, nkv, w) }
    #[kernel] pub unsafe fn fasv_bf16_128(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, cq: *const u32, ck: *const u32, sc: f32, mq: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<B, 128, 32, 4, true>(q as _, k as _, v as _, o as _, sk, cq, ck, sc, mq, 0, nh, nkv, w) }
    #[kernel] pub unsafe fn fas_bf16_192(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, sc: f32, ql: i32, kl: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<B, 192, 16, 6, false>(q as _, k as _, v as _, o as _, sk, core::ptr::null(), core::ptr::null(), sc, ql, kl, nh, nkv, w) }
    #[kernel] pub unsafe fn fasv_bf16_192(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, cq: *const u32, ck: *const u32, sc: f32, mq: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<B, 192, 16, 6, true>(q as _, k as _, v as _, o as _, sk, cq, ck, sc, mq, 0, nh, nkv, w) }
    #[kernel] pub unsafe fn fas_bf16_256(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, sc: f32, ql: i32, kl: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<B, 256, 16, 8, false>(q as _, k as _, v as _, o as _, sk, core::ptr::null(), core::ptr::null(), sc, ql, kl, nh, nkv, w) }
    #[kernel] pub unsafe fn fasv_bf16_256(q: *const u8, k: *const u8, v: *const u8, o: *mut u8, sk: *const f32, cq: *const u32, ck: *const u32, sc: f32, mq: i32, nh: i32, nkv: i32, w: i32) { flash_attn_sinks::<B, 256, 16, 8, true>(q as _, k as _, v as _, o as _, sk, cq, ck, sc, mq, 0, nh, nkv, w) }
    // GENERATED FA END
    // GENERATED paged attention kernels (gen_kernels.py) BEGIN
    #[kernel] pub unsafe fn pa1_f32_a_64_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 64, 8, 0, 4>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_a_64_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 64, 8, 512, 4>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_a_80_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 80, 8, 0, 5>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_a_80_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 80, 8, 512, 5>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_a_96_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 96, 8, 0, 6>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_a_96_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 96, 8, 512, 6>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_a_112_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 112, 8, 0, 7>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_a_112_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 112, 8, 512, 7>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_a_128_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 128, 8, 0, 8>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_a_128_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 128, 8, 512, 8>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_a_192_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 192, 8, 0, 12>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_a_192_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 192, 8, 512, 12>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_a_256_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 256, 8, 0, 16>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_a_256_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 256, 8, 512, 16>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_a_512_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 512, 8, 0, 32>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_a_512_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 512, 8, 512, 32>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_a_64_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 64, 16, 0, 8>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_a_64_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 64, 16, 512, 8>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_a_80_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 80, 16, 0, 10>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_a_80_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 80, 16, 512, 10>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_a_96_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 96, 16, 0, 12>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_a_96_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 96, 16, 512, 12>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_a_112_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 112, 16, 0, 14>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_a_112_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 112, 16, 512, 14>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_a_128_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 128, 16, 0, 16>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_a_128_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 128, 16, 512, 16>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_a_192_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 192, 16, 0, 24>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_a_192_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 192, 16, 512, 24>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_a_256_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 256, 16, 0, 32>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_a_256_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 256, 16, 512, 32>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_a_512_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 512, 16, 0, 64>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_a_512_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 512, 16, 512, 64>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_a_64_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 64, 32, 0, 16>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_a_64_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 64, 32, 512, 16>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_a_80_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 80, 32, 0, 20>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_a_80_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 80, 32, 512, 20>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_a_96_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 96, 32, 0, 24>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_a_96_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 96, 32, 512, 24>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_a_112_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 112, 32, 0, 28>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_a_112_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 112, 32, 512, 28>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_a_128_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 128, 32, 0, 32>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_a_128_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 128, 32, 512, 32>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_a_192_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 192, 32, 0, 48>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_a_192_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 192, 32, 512, 48>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_a_256_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 256, 32, 0, 64>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_a_256_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 256, 32, 512, 64>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_a_512_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 512, 32, 0, 128>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_a_512_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, false, 512, 32, 512, 128>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_e_64_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 64, 8, 0, 4>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_e_64_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 64, 8, 512, 4>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_e_80_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 80, 8, 0, 5>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_e_80_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 80, 8, 512, 5>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_e_96_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 96, 8, 0, 6>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_e_96_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 96, 8, 512, 6>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_e_112_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 112, 8, 0, 7>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_e_112_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 112, 8, 512, 7>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_e_128_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 128, 8, 0, 8>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_e_128_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 128, 8, 512, 8>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_e_192_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 192, 8, 0, 12>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_e_192_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 192, 8, 512, 12>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_e_256_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 256, 8, 0, 16>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_e_256_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 256, 8, 512, 16>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_e_512_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 512, 8, 0, 32>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_e_512_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 512, 8, 512, 32>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_e_64_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 64, 16, 0, 8>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_e_64_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 64, 16, 512, 8>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_e_80_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 80, 16, 0, 10>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_e_80_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 80, 16, 512, 10>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_e_96_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 96, 16, 0, 12>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_e_96_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 96, 16, 512, 12>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_e_112_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 112, 16, 0, 14>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_e_112_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 112, 16, 512, 14>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_e_128_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 128, 16, 0, 16>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_e_128_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 128, 16, 512, 16>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_e_192_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 192, 16, 0, 24>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_e_192_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 192, 16, 512, 24>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_e_256_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 256, 16, 0, 32>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_e_256_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 256, 16, 512, 32>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_e_512_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 512, 16, 0, 64>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_e_512_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 512, 16, 512, 64>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_e_64_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 64, 32, 0, 16>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_e_64_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 64, 32, 512, 16>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_e_80_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 80, 32, 0, 20>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_e_80_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 80, 32, 512, 20>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_e_96_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 96, 32, 0, 24>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_e_96_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 96, 32, 512, 24>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_e_112_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 112, 32, 0, 28>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_e_112_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 112, 32, 512, 28>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_e_128_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 128, 32, 0, 32>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_e_128_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 128, 32, 512, 32>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_e_192_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 192, 32, 0, 48>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_e_192_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 192, 32, 512, 48>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_e_256_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 256, 32, 0, 64>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_e_256_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 256, 32, 512, 64>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f32_e_512_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 512, 32, 0, 128>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f32_e_512_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<f32, true, 512, 32, 512, 128>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2r_f32_64(o: *mut u8, es: *const f32, ml: *const f32, t: *const u8, cl: *const u32, mp: i32, sk: *const f32) { paged_attention_v2_reduce::<f32, 64>(o as _, es, ml, t as _, cl, mp, sk) }
    #[kernel] pub unsafe fn pa2r_f32_80(o: *mut u8, es: *const f32, ml: *const f32, t: *const u8, cl: *const u32, mp: i32, sk: *const f32) { paged_attention_v2_reduce::<f32, 80>(o as _, es, ml, t as _, cl, mp, sk) }
    #[kernel] pub unsafe fn pa2r_f32_96(o: *mut u8, es: *const f32, ml: *const f32, t: *const u8, cl: *const u32, mp: i32, sk: *const f32) { paged_attention_v2_reduce::<f32, 96>(o as _, es, ml, t as _, cl, mp, sk) }
    #[kernel] pub unsafe fn pa2r_f32_112(o: *mut u8, es: *const f32, ml: *const f32, t: *const u8, cl: *const u32, mp: i32, sk: *const f32) { paged_attention_v2_reduce::<f32, 112>(o as _, es, ml, t as _, cl, mp, sk) }
    #[kernel] pub unsafe fn pa2r_f32_128(o: *mut u8, es: *const f32, ml: *const f32, t: *const u8, cl: *const u32, mp: i32, sk: *const f32) { paged_attention_v2_reduce::<f32, 128>(o as _, es, ml, t as _, cl, mp, sk) }
    #[kernel] pub unsafe fn pa2r_f32_192(o: *mut u8, es: *const f32, ml: *const f32, t: *const u8, cl: *const u32, mp: i32, sk: *const f32) { paged_attention_v2_reduce::<f32, 192>(o as _, es, ml, t as _, cl, mp, sk) }
    #[kernel] pub unsafe fn pa2r_f32_256(o: *mut u8, es: *const f32, ml: *const f32, t: *const u8, cl: *const u32, mp: i32, sk: *const f32) { paged_attention_v2_reduce::<f32, 256>(o as _, es, ml, t as _, cl, mp, sk) }
    #[kernel] pub unsafe fn pa2r_f32_512(o: *mut u8, es: *const f32, ml: *const f32, t: *const u8, cl: *const u32, mp: i32, sk: *const f32) { paged_attention_v2_reduce::<f32, 512>(o as _, es, ml, t as _, cl, mp, sk) }
    #[kernel] pub unsafe fn pa1_f16_a_64_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 64, 8, 0, 2>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_a_64_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 64, 8, 512, 2>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_a_80_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 80, 8, 0, 3>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_a_80_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 80, 8, 512, 3>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_a_96_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 96, 8, 0, 3>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_a_96_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 96, 8, 512, 3>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_a_112_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 112, 8, 0, 4>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_a_112_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 112, 8, 512, 4>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_a_128_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 128, 8, 0, 4>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_a_128_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 128, 8, 512, 4>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_a_192_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 192, 8, 0, 6>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_a_192_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 192, 8, 512, 6>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_a_256_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 256, 8, 0, 8>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_a_256_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 256, 8, 512, 8>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_a_512_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 512, 8, 0, 16>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_a_512_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 512, 8, 512, 16>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_a_64_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 64, 16, 0, 4>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_a_64_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 64, 16, 512, 4>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_a_80_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 80, 16, 0, 5>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_a_80_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 80, 16, 512, 5>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_a_96_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 96, 16, 0, 6>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_a_96_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 96, 16, 512, 6>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_a_112_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 112, 16, 0, 7>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_a_112_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 112, 16, 512, 7>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_a_128_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 128, 16, 0, 8>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_a_128_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 128, 16, 512, 8>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_a_192_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 192, 16, 0, 12>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_a_192_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 192, 16, 512, 12>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_a_256_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 256, 16, 0, 16>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_a_256_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 256, 16, 512, 16>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_a_512_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 512, 16, 0, 32>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_a_512_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 512, 16, 512, 32>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_a_64_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 64, 32, 0, 8>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_a_64_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 64, 32, 512, 8>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_a_80_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 80, 32, 0, 10>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_a_80_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 80, 32, 512, 10>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_a_96_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 96, 32, 0, 12>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_a_96_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 96, 32, 512, 12>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_a_112_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 112, 32, 0, 14>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_a_112_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 112, 32, 512, 14>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_a_128_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 128, 32, 0, 16>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_a_128_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 128, 32, 512, 16>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_a_192_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 192, 32, 0, 24>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_a_192_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 192, 32, 512, 24>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_a_256_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 256, 32, 0, 32>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_a_256_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 256, 32, 512, 32>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_a_512_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 512, 32, 0, 64>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_a_512_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, false, 512, 32, 512, 64>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_e_64_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 64, 8, 0, 2>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_e_64_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 64, 8, 512, 2>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_e_80_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 80, 8, 0, 3>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_e_80_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 80, 8, 512, 3>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_e_96_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 96, 8, 0, 3>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_e_96_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 96, 8, 512, 3>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_e_112_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 112, 8, 0, 4>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_e_112_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 112, 8, 512, 4>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_e_128_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 128, 8, 0, 4>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_e_128_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 128, 8, 512, 4>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_e_192_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 192, 8, 0, 6>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_e_192_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 192, 8, 512, 6>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_e_256_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 256, 8, 0, 8>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_e_256_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 256, 8, 512, 8>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_e_512_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 512, 8, 0, 16>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_e_512_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 512, 8, 512, 16>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_e_64_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 64, 16, 0, 4>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_e_64_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 64, 16, 512, 4>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_e_80_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 80, 16, 0, 5>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_e_80_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 80, 16, 512, 5>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_e_96_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 96, 16, 0, 6>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_e_96_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 96, 16, 512, 6>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_e_112_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 112, 16, 0, 7>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_e_112_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 112, 16, 512, 7>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_e_128_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 128, 16, 0, 8>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_e_128_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 128, 16, 512, 8>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_e_192_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 192, 16, 0, 12>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_e_192_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 192, 16, 512, 12>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_e_256_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 256, 16, 0, 16>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_e_256_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 256, 16, 512, 16>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_e_512_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 512, 16, 0, 32>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_e_512_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 512, 16, 512, 32>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_e_64_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 64, 32, 0, 8>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_e_64_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 64, 32, 512, 8>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_e_80_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 80, 32, 0, 10>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_e_80_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 80, 32, 512, 10>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_e_96_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 96, 32, 0, 12>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_e_96_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 96, 32, 512, 12>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_e_112_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 112, 32, 0, 14>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_e_112_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 112, 32, 512, 14>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_e_128_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 128, 32, 0, 16>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_e_128_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 128, 32, 512, 16>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_e_192_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 192, 32, 0, 24>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_e_192_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 192, 32, 512, 24>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_e_256_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 256, 32, 0, 32>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_e_256_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 256, 32, 512, 32>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_f16_e_512_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 512, 32, 0, 64>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_f16_e_512_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<H, true, 512, 32, 512, 64>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2r_f16_64(o: *mut u8, es: *const f32, ml: *const f32, t: *const u8, cl: *const u32, mp: i32, sk: *const f32) { paged_attention_v2_reduce::<H, 64>(o as _, es, ml, t as _, cl, mp, sk) }
    #[kernel] pub unsafe fn pa2r_f16_80(o: *mut u8, es: *const f32, ml: *const f32, t: *const u8, cl: *const u32, mp: i32, sk: *const f32) { paged_attention_v2_reduce::<H, 80>(o as _, es, ml, t as _, cl, mp, sk) }
    #[kernel] pub unsafe fn pa2r_f16_96(o: *mut u8, es: *const f32, ml: *const f32, t: *const u8, cl: *const u32, mp: i32, sk: *const f32) { paged_attention_v2_reduce::<H, 96>(o as _, es, ml, t as _, cl, mp, sk) }
    #[kernel] pub unsafe fn pa2r_f16_112(o: *mut u8, es: *const f32, ml: *const f32, t: *const u8, cl: *const u32, mp: i32, sk: *const f32) { paged_attention_v2_reduce::<H, 112>(o as _, es, ml, t as _, cl, mp, sk) }
    #[kernel] pub unsafe fn pa2r_f16_128(o: *mut u8, es: *const f32, ml: *const f32, t: *const u8, cl: *const u32, mp: i32, sk: *const f32) { paged_attention_v2_reduce::<H, 128>(o as _, es, ml, t as _, cl, mp, sk) }
    #[kernel] pub unsafe fn pa2r_f16_192(o: *mut u8, es: *const f32, ml: *const f32, t: *const u8, cl: *const u32, mp: i32, sk: *const f32) { paged_attention_v2_reduce::<H, 192>(o as _, es, ml, t as _, cl, mp, sk) }
    #[kernel] pub unsafe fn pa2r_f16_256(o: *mut u8, es: *const f32, ml: *const f32, t: *const u8, cl: *const u32, mp: i32, sk: *const f32) { paged_attention_v2_reduce::<H, 256>(o as _, es, ml, t as _, cl, mp, sk) }
    #[kernel] pub unsafe fn pa2r_f16_512(o: *mut u8, es: *const f32, ml: *const f32, t: *const u8, cl: *const u32, mp: i32, sk: *const f32) { paged_attention_v2_reduce::<H, 512>(o as _, es, ml, t as _, cl, mp, sk) }
    #[kernel] pub unsafe fn pa1_bf16_a_64_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 64, 8, 0, 2>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_a_64_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 64, 8, 512, 2>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_a_80_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 80, 8, 0, 3>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_a_80_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 80, 8, 512, 3>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_a_96_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 96, 8, 0, 3>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_a_96_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 96, 8, 512, 3>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_a_112_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 112, 8, 0, 4>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_a_112_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 112, 8, 512, 4>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_a_128_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 128, 8, 0, 4>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_a_128_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 128, 8, 512, 4>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_a_192_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 192, 8, 0, 6>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_a_192_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 192, 8, 512, 6>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_a_256_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 256, 8, 0, 8>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_a_256_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 256, 8, 512, 8>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_a_512_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 512, 8, 0, 16>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_a_512_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 512, 8, 512, 16>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_a_64_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 64, 16, 0, 4>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_a_64_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 64, 16, 512, 4>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_a_80_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 80, 16, 0, 5>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_a_80_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 80, 16, 512, 5>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_a_96_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 96, 16, 0, 6>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_a_96_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 96, 16, 512, 6>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_a_112_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 112, 16, 0, 7>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_a_112_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 112, 16, 512, 7>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_a_128_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 128, 16, 0, 8>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_a_128_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 128, 16, 512, 8>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_a_192_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 192, 16, 0, 12>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_a_192_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 192, 16, 512, 12>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_a_256_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 256, 16, 0, 16>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_a_256_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 256, 16, 512, 16>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_a_512_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 512, 16, 0, 32>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_a_512_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 512, 16, 512, 32>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_a_64_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 64, 32, 0, 8>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_a_64_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 64, 32, 512, 8>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_a_80_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 80, 32, 0, 10>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_a_80_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 80, 32, 512, 10>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_a_96_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 96, 32, 0, 12>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_a_96_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 96, 32, 512, 12>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_a_112_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 112, 32, 0, 14>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_a_112_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 112, 32, 512, 14>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_a_128_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 128, 32, 0, 16>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_a_128_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 128, 32, 512, 16>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_a_192_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 192, 32, 0, 24>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_a_192_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 192, 32, 512, 24>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_a_256_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 256, 32, 0, 32>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_a_256_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 256, 32, 512, 32>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_a_512_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 512, 32, 0, 64>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_a_512_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, false, 512, 32, 512, 64>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_e_64_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 64, 8, 0, 2>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_e_64_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 64, 8, 512, 2>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_e_80_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 80, 8, 0, 3>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_e_80_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 80, 8, 512, 3>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_e_96_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 96, 8, 0, 3>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_e_96_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 96, 8, 512, 3>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_e_112_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 112, 8, 0, 4>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_e_112_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 112, 8, 512, 4>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_e_128_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 128, 8, 0, 4>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_e_128_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 128, 8, 512, 4>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_e_192_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 192, 8, 0, 6>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_e_192_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 192, 8, 512, 6>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_e_256_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 256, 8, 0, 8>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_e_256_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 256, 8, 512, 8>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_e_512_8(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 512, 8, 0, 16>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_e_512_8(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 512, 8, 512, 16>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_e_64_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 64, 16, 0, 4>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_e_64_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 64, 16, 512, 4>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_e_80_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 80, 16, 0, 5>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_e_80_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 80, 16, 512, 5>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_e_96_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 96, 16, 0, 6>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_e_96_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 96, 16, 512, 6>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_e_112_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 112, 16, 0, 7>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_e_112_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 112, 16, 512, 7>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_e_128_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 128, 16, 0, 8>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_e_128_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 128, 16, 512, 8>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_e_192_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 192, 16, 0, 12>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_e_192_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 192, 16, 512, 12>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_e_256_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 256, 16, 0, 16>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_e_256_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 256, 16, 512, 16>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_e_512_16(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 512, 16, 0, 32>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_e_512_16(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 512, 16, 512, 32>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_e_64_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 64, 32, 0, 8>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_e_64_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 64, 32, 512, 8>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_e_80_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 80, 32, 0, 10>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_e_80_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 80, 32, 512, 10>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_e_96_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 96, 32, 0, 12>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_e_96_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 96, 32, 512, 12>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_e_112_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 112, 32, 0, 14>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_e_112_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 112, 32, 512, 14>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_e_128_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 128, 32, 0, 16>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_e_128_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 128, 32, 512, 16>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_e_192_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 192, 32, 0, 24>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_e_192_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 192, 32, 512, 24>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_e_256_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 256, 32, 0, 32>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_e_256_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 256, 32, 512, 32>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa1_bf16_e_512_32(o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 512, 32, 0, 64>(core::ptr::null_mut(), core::ptr::null_mut(), o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2_bf16_e_512_32(es: *mut f32, ml: *mut f32, o: *mut u8, q: *const u8, kc: *const u8, vc: *const u8, nkv: i32, sc: f32, cap: f32, bt: *const u32, cl: *const u32, mb: i32, al: *const f32, qs: i32, kbs: i32, khs: i32, ks: *const f32, vs: *const f32, sk: *const f32) { paged_attention::<B, true, 512, 32, 512, 64>(es, ml, o as _, q as _, kc, vc, nkv, sc, cap, bt, cl, mb, al, qs, kbs, khs, ks, vs, sk) }
    #[kernel] pub unsafe fn pa2r_bf16_64(o: *mut u8, es: *const f32, ml: *const f32, t: *const u8, cl: *const u32, mp: i32, sk: *const f32) { paged_attention_v2_reduce::<B, 64>(o as _, es, ml, t as _, cl, mp, sk) }
    #[kernel] pub unsafe fn pa2r_bf16_80(o: *mut u8, es: *const f32, ml: *const f32, t: *const u8, cl: *const u32, mp: i32, sk: *const f32) { paged_attention_v2_reduce::<B, 80>(o as _, es, ml, t as _, cl, mp, sk) }
    #[kernel] pub unsafe fn pa2r_bf16_96(o: *mut u8, es: *const f32, ml: *const f32, t: *const u8, cl: *const u32, mp: i32, sk: *const f32) { paged_attention_v2_reduce::<B, 96>(o as _, es, ml, t as _, cl, mp, sk) }
    #[kernel] pub unsafe fn pa2r_bf16_112(o: *mut u8, es: *const f32, ml: *const f32, t: *const u8, cl: *const u32, mp: i32, sk: *const f32) { paged_attention_v2_reduce::<B, 112>(o as _, es, ml, t as _, cl, mp, sk) }
    #[kernel] pub unsafe fn pa2r_bf16_128(o: *mut u8, es: *const f32, ml: *const f32, t: *const u8, cl: *const u32, mp: i32, sk: *const f32) { paged_attention_v2_reduce::<B, 128>(o as _, es, ml, t as _, cl, mp, sk) }
    #[kernel] pub unsafe fn pa2r_bf16_192(o: *mut u8, es: *const f32, ml: *const f32, t: *const u8, cl: *const u32, mp: i32, sk: *const f32) { paged_attention_v2_reduce::<B, 192>(o as _, es, ml, t as _, cl, mp, sk) }
    #[kernel] pub unsafe fn pa2r_bf16_256(o: *mut u8, es: *const f32, ml: *const f32, t: *const u8, cl: *const u32, mp: i32, sk: *const f32) { paged_attention_v2_reduce::<B, 256>(o as _, es, ml, t as _, cl, mp, sk) }
    #[kernel] pub unsafe fn pa2r_bf16_512(o: *mut u8, es: *const f32, ml: *const f32, t: *const u8, cl: *const u32, mp: i32, sk: *const f32) { paged_attention_v2_reduce::<B, 512>(o as _, es, ml, t as _, cl, mp, sk) }
    // GENERATED END
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() == 4 && a[1] == "--exit-case" {
        gate::exit_case(a[2] == "rust", a[3].parse().unwrap());
        std::process::exit(0);
    }
    std::process::exit(if gate::run() { 0 } else { 1 });
}
