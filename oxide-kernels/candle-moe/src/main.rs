#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::too_many_arguments)]
//! candle-kernels MoE (`moe/moe_wmma.cu`, `moe/moe_gguf.cu`, `moe/moe_wmma_gguf.cu`) in cuda-oxide,
//! plus pure-Rust host launchers (`launch.rs`) with the exact `ffi.rs` signatures of
//! `moe_gemm_wmma`, `moe_gemm_gguf` and `moe_gemm_gguf_prefill`.
//!
//! The reference exists only as SASS (candle builds these with nvcc -O3, no fast-math, sm_120a,
//! `--default-stream per-thread`), so every float program below is read off that SASS:
//!
//! - candle bug reproduced: moe_gemm_wmma's f16 epilogue stores `half((float)bits)` where `bits` is
//!   the rounded f16 bit pattern (`static_cast<half>(uint16_t)` in moe_utils.cuh `from_float`), so
//!   its f16 output is not the product. bf16 output is correct.
//! - WMMA: the reference's `nvcuda::wmma` calls are emitted as the same PTX `wmma.*` instructions
//!   (`ptx_asm!`), so ptxas lowers them to the same `HMMA.16816.F32[.BF16]` sequence (m8n32k16 is two
//!   HMMAs per k-step in both). Epilogue: FMUL by the top-k weight, then `cvt.rn.{f16,bf16}.f32`.
//! - GGUF decode vec-dots (llama.cpp mmvq): per sub-block `sumf = fma(d8, float(dp4a*sc), sumf)`;
//!   K-quants with mins end in `fma(sumf_d, dm.x, -(sumf_m * dm.y))` added to the accumulator
//!   (FADD); Q3_K/Q6_K end in `acc = fma(d, sumf, acc)`; Q8_0 is `acc = fma(d_w*d_y, float(sumi), acc)`.
//!   Then an xor butterfly and one FMUL by the top-k weight.
//! - GGUF prefill dequantisation uses `cuda_fp16` intrinsics whose PTX (`mul.f16`, `sub.f16`, no
//!   `.rn`) ptxas contracts: `__hsub(__hmul(a,b), c)` -> HFMA2 `fma(a, b, -c)`; for Q2_K
//!   `dall*x - dmin*y` -> `fma(dall, x, -(dmin*y))`. Written here with `.rn` (uncontractable) ops.
//!   bf16 destinations convert f16 -> f32 -> `cvt.rn.bf16.f32`.
//! - quantize_q8_1: `max.f32` butterfly (fmaxf), IEEE `div.rn`, roundf = `add.rz(t, copysign(.5,t))`,
//!   `cvt.rzi.s32.f32` truncated to the low byte.
//! - The thrust/cub scans and `expert_prefix_sum_kernel` are replaced by one single-block exclusive
//!   scan with identical integer output.
mod launch;

use cuda_device::{DynamicSharedArray, SharedArray, convert, dotprod, float, kernel, ptx_asm, thread, warp};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    const FULL: u32 = 0xffff_ffff;

    // ------------------------------------------------------------------ scalar helpers
    #[inline(always)]
    pub fn h2f(bits: u32) -> f32 {
        convert::cvt_f32_f16x2_lo(bits & 0xffff)
    }
    #[inline(always)]
    pub fn f2h(x: f32) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("cvt.rn.f16.f32 %0, %1;", out("=h") r, in("f") x, options(register_only)) };
        r
    }
    #[inline(always)]
    pub fn f2bf(x: f32) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("cvt.rn.bf16.f32 %0, %1;", out("=h") r, in("f") x, options(register_only)) };
        r
    }
    #[inline(always)]
    pub fn hf2f(h: u16) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("cvt.f32.f16 %0, %1;", out("=f") r, in("h") h, options(register_only)) };
        r
    }
    /// `__ushort2half_rn` (`cvt.rn.f16.u16`).
    #[inline(always)]
    pub fn u16_to_h(x: u16) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("cvt.rn.f16.u16 %0, %1;", out("=h") r, in("h") x, options(register_only)) };
        r
    }
    /// `__int2half_rn`.
    #[inline(always)]
    pub fn i2h(i: i32) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("cvt.rn.f16.s32 %0, %1;", out("=h") r, in("r") i, options(register_only)) };
        r
    }
    /// A standalone HMUL2 (`.rn` so ptxas never contracts it).
    #[inline(always)]
    pub fn hmul(a: u16, b: u16) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("mul.rn.f16 %0, %1, %2;", out("=h") r, in("h") a, in("h") b, options(register_only)) };
        r
    }
    #[inline(always)]
    pub fn hfma(a: u16, b: u16, c: u16) -> u16 {
        let r: u16;
        unsafe {
            ptx_asm!("fma.rn.f16 %0, %1, %2, %3;", out("=h") r, in("h") a, in("h") b, in("h") c, options(register_only))
        };
        r
    }
    #[inline(always)]
    pub fn f2i_rz(x: f32) -> i32 {
        let r: i32;
        unsafe { ptx_asm!("cvt.rzi.s32.f32 %0, %1;", out("=r") r, in("f") x, options(register_only)) };
        r
    }
    #[inline(always)]
    pub fn fmax(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("max.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)) };
        r
    }
    #[inline(always)]
    pub fn fma(a: f32, b: f32, c: f32) -> f32 {
        float::fma_rn_f32(a, b, c)
    }
    #[inline(always)]
    pub fn mul(a: f32, b: f32) -> f32 {
        float::mul_rn_f32(a, b)
    }
    #[inline(always)]
    pub fn add(a: f32, b: f32) -> f32 {
        float::add_rn_f32(a, b)
    }
    #[inline(always)]
    pub fn dp4a(a: u32, b: u32, c: i32) -> i32 {
        dotprod::dp4a_s32(a, b, c)
    }
    /// `__vsubss4`: per-byte signed saturating subtraction.
    #[inline(always)]
    pub fn vsubss4(a: u32, b: u32) -> u32 {
        let mut r = 0u32;
        let mut i = 0;
        while i < 4 {
            let x = ((a >> (8 * i)) as u8 as i8 as i32) - ((b >> (8 * i)) as u8 as i8 as i32);
            let x = if x < -128 { -128 } else if x > 127 { 127 } else { x };
            r |= (x as u8 as u32) << (8 * i);
            i += 1;
        }
        r
    }

    #[inline(always)]
    pub unsafe fn ld8(p: *const u8, o: usize) -> u32 {
        *p.add(o) as u32
    }
    #[inline(always)]
    pub unsafe fn ld16(p: *const u8, o: usize) -> u32 {
        *(p.add(o) as *const u16) as u32
    }
    /// `get_int_from_uint8`: two 16-bit loads.
    #[inline(always)]
    pub unsafe fn ld32u(p: *const u8, o: usize) -> u32 {
        ld16(p, o) | (ld16(p, o + 2) << 16)
    }
    #[inline(always)]
    pub unsafe fn ld32a(p: *const u8, o: usize) -> u32 {
        *(p.add(o) as *const u32)
    }
    #[inline(always)]
    pub unsafe fn copy16(dst: *mut u8, src: *const u8) {
        *(dst as *mut [u32; 4]) = *(src as *const [u32; 4]);
    }
    #[inline(always)]
    pub unsafe fn zero16(dst: *mut u8) {
        *(dst as *mut [u32; 4]) = [0; 4];
    }

    #[inline(always)]
    pub fn tid_x() -> i32 {
        thread::threadIdx_x() as i32
    }
    #[inline(always)]
    pub fn tid_y() -> i32 {
        thread::threadIdx_y() as i32
    }

    // ------------------------------------------------------------------ WMMA (PTX wmma.*)
    // Fragments are carried as 8 x b32 (a/b) and 8 x f32 (c). Unused registers stay 0.

    /// load.a (row-major) + load.b (col-major) + mma, accumulating into `c`.
    /// SHAPE: 0 = m16n16k16, 1 = m8n32k16.
    #[inline(always)]
    pub unsafe fn wmma_step<const BF16: bool, const SHAPE: u32>(c: &mut [f32; 8], a_ptr: *const u16, b_ptr: *const u16, ldm: u32) {
        let (a0, a1, a2, a3, a4, a5, a6, a7): (u32, u32, u32, u32, u32, u32, u32, u32);
        let (b0, b1, b2, b3, b4, b5, b6, b7): (u32, u32, u32, u32, u32, u32, u32, u32);
        let ap = a_ptr as u64;
        let bp = b_ptr as u64;
        let (mut c0, mut c1, mut c2, mut c3, mut c4, mut c5, mut c6, mut c7) = (c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
        if !BF16 {
            if SHAPE == 0 {
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
                ptx_asm!("wmma.load.a.sync.aligned.row.m8n32k16.f16 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;",
                    out("=r") a0, out("=r") a1, out("=r") a2, out("=r") a3, out("=r") a4, out("=r") a5, out("=r") a6, out("=r") a7,
                    in("l") ap, in("r") ldm, clobber("memory"));
                ptx_asm!("wmma.load.b.sync.aligned.col.m8n32k16.f16 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;",
                    out("=r") b0, out("=r") b1, out("=r") b2, out("=r") b3, out("=r") b4, out("=r") b5, out("=r") b6, out("=r") b7,
                    in("l") bp, in("r") ldm, clobber("memory"));
                ptx_asm!("wmma.mma.sync.aligned.row.col.m8n32k16.f32.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19,%20,%21,%22,%23}, {%0,%1,%2,%3,%4,%5,%6,%7};",
                    inout("+f") c0, inout("+f") c1, inout("+f") c2, inout("+f") c3, inout("+f") c4, inout("+f") c5, inout("+f") c6, inout("+f") c7,
                    in("r") a0, in("r") a1, in("r") a2, in("r") a3, in("r") a4, in("r") a5, in("r") a6, in("r") a7,
                    in("r") b0, in("r") b1, in("r") b2, in("r") b3, in("r") b4, in("r") b5, in("r") b6, in("r") b7);
            }
        } else if SHAPE == 0 {
            ptx_asm!("wmma.load.a.sync.aligned.row.m16n16k16.bf16 {%0,%1,%2,%3}, [%4], %5;",
                out("=r") a0, out("=r") a1, out("=r") a2, out("=r") a3, in("l") ap, in("r") ldm, clobber("memory"));
            ptx_asm!("wmma.load.b.sync.aligned.col.m16n16k16.bf16 {%0,%1,%2,%3}, [%4], %5;",
                out("=r") b0, out("=r") b1, out("=r") b2, out("=r") b3, in("l") bp, in("r") ldm, clobber("memory"));
            ptx_asm!("wmma.mma.sync.aligned.row.col.m16n16k16.f32.bf16.bf16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, {%0,%1,%2,%3,%4,%5,%6,%7};",
                inout("+f") c0, inout("+f") c1, inout("+f") c2, inout("+f") c3, inout("+f") c4, inout("+f") c5, inout("+f") c6, inout("+f") c7,
                in("r") a0, in("r") a1, in("r") a2, in("r") a3,
                in("r") b0, in("r") b1, in("r") b2, in("r") b3);
        } else {
            ptx_asm!("wmma.load.a.sync.aligned.row.m8n32k16.bf16 {%0,%1}, [%2], %3;",
                out("=r") a0, out("=r") a1, in("l") ap, in("r") ldm, clobber("memory"));
            ptx_asm!("wmma.load.b.sync.aligned.col.m8n32k16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;",
                out("=r") b0, out("=r") b1, out("=r") b2, out("=r") b3, out("=r") b4, out("=r") b5, out("=r") b6, out("=r") b7,
                in("l") bp, in("r") ldm, clobber("memory"));
            ptx_asm!("wmma.mma.sync.aligned.row.col.m8n32k16.f32.bf16.bf16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13,%14,%15,%16,%17}, {%0,%1,%2,%3,%4,%5,%6,%7};",
                inout("+f") c0, inout("+f") c1, inout("+f") c2, inout("+f") c3, inout("+f") c4, inout("+f") c5, inout("+f") c6, inout("+f") c7,
                in("r") a0, in("r") a1,
                in("r") b0, in("r") b1, in("r") b2, in("r") b3, in("r") b4, in("r") b5, in("r") b6, in("r") b7);
        }
        *c = [c0, c1, c2, c3, c4, c5, c6, c7];
    }

    /// store_matrix_sync(ptr, c, ldm, mem_row_major) for the f32 accumulator.
    #[inline(always)]
    pub unsafe fn wmma_store<const SHAPE: u32>(p: *mut f32, c: &[f32; 8], ldm: u32) {
        let pp = p as u64;
        if SHAPE == 0 {
            ptx_asm!("wmma.store.d.sync.aligned.row.m16n16k16.f32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8}, %9;",
                in("l") pp, in("f") c[0], in("f") c[1], in("f") c[2], in("f") c[3], in("f") c[4], in("f") c[5], in("f") c[6], in("f") c[7],
                in("r") ldm, clobber("memory"));
        } else {
            ptx_asm!("wmma.store.d.sync.aligned.row.m8n32k16.f32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8}, %9;",
                in("l") pp, in("f") c[0], in("f") c[1], in("f") c[2], in("f") c[3], in("f") c[4], in("f") c[5], in("f") c[6], in("f") c[7],
                in("r") ldm, clobber("memory"));
        }
    }

    // ------------------------------------------------------------------ routing helpers
    #[kernel]
    pub unsafe fn moe_count_tokens(expert_ids: *const i32, counts: *mut i32, size_m: i32) {
        let i = (thread::blockIdx_x() as i32).wrapping_mul(thread::blockDim_x() as i32).wrapping_add(tid_x());
        if i < size_m {
            let e = *expert_ids.add(i as usize);
            let p = counts.offset(e as isize) as u64;
            ptx_asm!("red.global.add.s32 [%0], %1;", in("l") p, in("r") 1i32, clobber("memory"));
        }
    }

    /// Exclusive scan `offsets[0] = 0, offsets[i + 1] = counts[0] + .. + counts[i]` (wrapping i32),
    /// one block of 1024 threads, any `n`.
    #[kernel]
    pub unsafe fn moe_expert_offsets(counts: *const i32, offsets: *mut i32, n: i32) {
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
        let mut off = 1;
        while off < 1024 {
            let t = if tid >= off { *part.add((tid - off) as usize) } else { 0 };
            thread::sync_threads();
            if tid >= off {
                *part.add(tid as usize) = (*part.add(tid as usize)).wrapping_add(t);
            }
            thread::sync_threads();
            off <<= 1;
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

    // ------------------------------------------------------------------ moe_wmma.cu
    /// `vllm_rs::moe_gemm_grouped_kernel<T, WM, WN, WARPS_N>`: SHAPE 0 = <16,16,2> (prefill),
    /// 1 = <8,32,1> (decode). Block 128, grid (num_experts, ceil(n/32)).
    #[inline(always)]
    unsafe fn grouped<const BF16: bool, const SHAPE: u32>(
        input: *const u16, weights: *const u16, sorted: *const i32, offsets: *const i32, tw: *const f32, output: *mut u16,
        num_experts: i32, topk: i32, size_m: i32, size_n: i32, size_k: i32,
    ) {
        let (wm, wn, warps_n) = if SHAPE == 0 { (16i32, 16i32, 2i32) } else { (8, 32, 1) };
        let expert_id = thread::blockIdx_x() as i32;
        let n_tile = thread::blockIdx_y() as i32;
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
        let expert_w = weights.add((expert_id as u64).wrapping_mul(size_n as i64 as u64).wrapping_mul(size_k as i64 as u64) as usize);
        let smem = DynamicSharedArray::<u8>::get_raw();
        let a_sh = smem as *mut u16;
        let b_sh = smem.add(1024) as *mut u16;
        let c_sh = smem.add(2048) as *mut f32;
        let tid = tid_x();
        let warp_id = tid / 32;
        let warp_m = warp_id / warps_n;
        let warp_n = warp_id % warps_n;
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
                        let off = (n_global as i64 as u64).wrapping_mul(size_k as i64 as u64).wrapping_add(k_global as i64 as u64);
                        copy16(dst, expert_w.add(off as usize) as *const u8);
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
                        let token_index = *sorted.add((seg_start + m_seg) as usize);
                        let input_index = token_index / (if tw.is_null() { topk } else { 1 });
                        let off = (input_index as i64 as u64).wrapping_mul(size_k as i64 as u64).wrapping_add(k_global as i64 as u64);
                        copy16(dst, input.add(off as usize) as *const u8);
                    } else {
                        zero16(dst);
                    }
                    i += 128;
                }
                thread::sync_threads();
                wmma_step::<BF16, SHAPE>(
                    &mut c,
                    a_sh.add((warp_m * wm * 16) as usize),
                    b_sh.add((warp_n * wn * 16) as usize),
                    16,
                );
                thread::sync_threads();
                k_base += 16;
            }
            wmma_store::<SHAPE>(c_sh.add((warp_m * wm * 32 + warp_n * wn) as usize), &c, 32);
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
                        let token_index = *sorted.add(tp as usize);
                        let mut val = *c_sh.add((m_local * 32 + n_local) as usize);
                        if !tw.is_null() {
                            val = mul(val, *tw.add(token_index as usize));
                        }
                        let o = (token_index as i64 as u64).wrapping_mul(size_n as i64 as u64).wrapping_add(n_global as i64 as u64);
                        // f16: candle's `from_float(half&, float)` is `static_cast<half>(float_to_half(v))`
                        // with float_to_half returning the *bits* as uint16_t, so the stored value is
                        // the half nearest to the integer value of those bits (SASS: F2FP.F16 then
                        // I2F.F16.U16). Reproduced as-is.
                        *output.add(o as usize) = if BF16 { f2bf(val) } else { u16_to_h(f2h(val)) };
                    }
                }
                i += 128;
            }
            m_base += 32;
        }
    }

    #[kernel]
    pub unsafe fn moe_wmma_f16_prefill(input: *const u16, weights: *const u16, sorted: *const i32, offsets: *const i32, tw: *const f32,
                                       output: *mut u16, num_experts: i32, topk: i32, size_m: i32, size_n: i32, size_k: i32) {
        grouped::<false, 0>(input, weights, sorted, offsets, tw, output, num_experts, topk, size_m, size_n, size_k)
    }
    #[kernel]
    pub unsafe fn moe_wmma_f16_decode(input: *const u16, weights: *const u16, sorted: *const i32, offsets: *const i32, tw: *const f32,
                                      output: *mut u16, num_experts: i32, topk: i32, size_m: i32, size_n: i32, size_k: i32) {
        grouped::<false, 1>(input, weights, sorted, offsets, tw, output, num_experts, topk, size_m, size_n, size_k)
    }
    #[kernel]
    pub unsafe fn moe_wmma_bf16_prefill(input: *const u16, weights: *const u16, sorted: *const i32, offsets: *const i32, tw: *const f32,
                                        output: *mut u16, num_experts: i32, topk: i32, size_m: i32, size_n: i32, size_k: i32) {
        grouped::<true, 0>(input, weights, sorted, offsets, tw, output, num_experts, topk, size_m, size_n, size_k)
    }
    #[kernel]
    pub unsafe fn moe_wmma_bf16_decode(input: *const u16, weights: *const u16, sorted: *const i32, offsets: *const i32, tw: *const f32,
                                       output: *mut u16, num_experts: i32, topk: i32, size_m: i32, size_n: i32, size_k: i32) {
        grouped::<true, 1>(input, weights, sorted, offsets, tw, output, num_experts, topk, size_m, size_n, size_k)
    }

    // ------------------------------------------------------------------ gguf.cuh quantize_q8_1
    /// Grid (ceil(kx_padded/256), rows), block 256. `y` rows of kx_padded/32 blocks of 36 bytes.
    #[kernel]
    pub unsafe fn moe_quantize_q8_1(x: *const f32, y: *mut u8, kx: i32, kx_padded: i32) {
        let ix = (thread::blockDim_x() as i32).wrapping_mul(thread::blockIdx_x() as i32).wrapping_add(tid_x());
        if ix >= kx_padded {
            return;
        }
        let iy = (thread::blockDim_y() as i32).wrapping_mul(thread::blockIdx_y() as i32).wrapping_add(tid_y());
        let i_padded = iy.wrapping_mul(kx_padded).wrapping_add(ix);
        let ib = i_padded / 32;
        let iqs = i_padded % 32;
        let xi = if ix < kx { *x.offset(iy.wrapping_mul(kx).wrapping_add(ix) as isize) } else { 0.0 };
        let mut amax = f32::from_bits(xi.to_bits() & 0x7fff_ffff);
        let mut sum = xi;
        let mut mask = 16;
        while mask > 0 {
            amax = fmax(warp::shuffle_xor_f32_sync(FULL, amax, mask), amax);
            sum = add(warp::shuffle_xor_f32_sync(FULL, sum, mask), sum);
            mask >>= 1;
        }
        let d = amax / 127.0;
        let q = if amax == 0.0 {
            0i32
        } else {
            let t = xi / d;
            let half = f32::from_bits(0x3f00_0000 | (t.to_bits() & 0x8000_0000));
            f2i_rz(float::add_rz_f32(t, half))
        };
        let blk = y.offset(ib as isize * 36);
        *blk.offset(4 + iqs as isize) = q as u8;
        if iqs > 0 {
            return;
        }
        *(blk as *mut u16) = f2h(d);
        *(blk.add(2) as *mut u16) = f2h(sum);
    }

    // ------------------------------------------------------------------ moe_gguf.cu vec-dots
    // `w` = the weight block (in shared memory), `y` = its first q8_1 block (global), `iqs` = kqs.
    // Each returns the updated accumulator (the reference's `acc += vec_dot(...)` after ptxas).

    #[inline(always)]
    unsafe fn step_q8_0(acc: f32, w: *const u8, y: *const u8, iqs: i32) -> f32 {
        let mut sumi = 0i32;
        let mut i = 0;
        while i < 2 {
            let v = ld32u(w, 2 + 4 * (iqs + i) as usize);
            let u = ld32a(y, 4 + 4 * (iqs + i) as usize);
            sumi = dp4a(v, u, sumi);
            i += 1;
        }
        let d = mul(h2f(ld16(w, 0)), h2f(ld16(y, 0)));
        fma(d, sumi as f32, acc)
    }

    #[inline(always)]
    unsafe fn step_q2_k(acc: f32, w: *const u8, y: *const u8, iqs: i32) -> f32 {
        let bq8_offset = 4 * (iqs / 8);
        let scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
        let v = ld32a(w, 16 + 4 * iqs as usize);
        let mut sd = 0f32;
        let mut sm = 0f32;
        let mut i = 0;
        while i < 4 {
            let yb = y.add(36 * (bq8_offset + i) as usize);
            let u = ld32a(yb, 4 + 4 * (iqs % 8) as usize);
            let d8 = h2f(ld16(yb, 0));
            let sc = ld8(w, (scale_offset + 2 * i) as usize);
            let vi = (v >> (2 * i)) & 0x0303_0303;
            sd = fma(d8, (dp4a(vi, u, 0) * (sc & 0xF) as i32) as f32, sd);
            let mut m = sc >> 4;
            m |= m << 8;
            m |= m << 16;
            sm = fma(d8, dp4a(m, u, 0) as f32, sm);
            i += 1;
        }
        let dm = ld32a(w, 80);
        let p = mul(sm, h2f(dm >> 16));
        add(acc, fma(sd, h2f(dm), -p))
    }

    #[inline(always)]
    unsafe fn step_q3_k(acc: f32, w: *const u8, y: *const u8, iqs: i32) -> f32 {
        let bq8_offset = 4 * (iqs / 8);
        let scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
        let d = h2f(ld16(w, 108));
        let vl = ld32u(w, 32 + 4 * iqs as usize) as i32;
        let vh = (!(ld32u(w, 4 * (iqs % 8) as usize) as i32)) >> bq8_offset;
        let mut sumf = 0f32;
        let mut i = 0;
        while i < 4 {
            let yb = y.add(36 * (bq8_offset + i) as usize);
            let u = ld32a(yb, 4 + 4 * (iqs % 8) as usize);
            let d8 = h2f(ld16(yb, 0));
            let isc = scale_offset + 2 * i;
            let sc_low = (ld8(w, 96 + (isc % 8) as usize) >> (4 * (isc / 8))) & 0xF;
            let sc_high = ((ld8(w, 96 + 8 + (isc % 4) as usize) >> (2 * (isc / 4))) & 3) << 4;
            let sc = (sc_low | sc_high) as i32 - 32;
            let vil = ((vl >> (2 * i)) & 0x0303_0303) as u32;
            let vih = (((vh >> i) << 2) & 0x0404_0404) as u32;
            let vi = vsubss4(vil, vih);
            sumf = fma(d8, (dp4a(vi, u, 0) * sc) as f32, sumf);
            i += 1;
        }
        fma(d, sumf, acc)
    }

    /// Q4_K / Q5_K scale and min bytes for this thread's sub-block pair.
    #[inline(always)]
    unsafe fn k_scales(w: *const u8, bq8_offset: i32) -> [u32; 4] {
        let s16 = |j: i32| -> u32 { ld16(w, 4 + 2 * j as usize) };
        let j = bq8_offset / 2;
        let (aux0, aux1) = if j < 2 {
            (s16(j) & 0x3f3f, s16(j + 2) & 0x3f3f)
        } else {
            (
                (s16(j + 2) & 0x0f0f) | ((s16(j - 2) & 0xc0c0) >> 2),
                ((s16(j + 2) >> 4) & 0x0f0f) | ((s16(j) & 0xc0c0) >> 2),
            )
        };
        [aux0 & 0xff, aux0 >> 8, aux1 & 0xff, aux1 >> 8]
    }

    /// Shared tail of the Q4_K / Q5_K vec-dots: v[i] = (v0i, v1i) per sub-block.
    #[inline(always)]
    unsafe fn k_tail(acc: f32, v: [[u32; 2]; 2], w: *const u8, y: *const u8, bq8_offset: i32, iqs: i32) -> f32 {
        let s = k_scales(w, bq8_offset);
        let mut sd = 0f32;
        let mut sm = 0f32;
        let mut i = 0;
        while i < 2 {
            let yb = y.add(36 * (bq8_offset + i) as usize);
            let d8 = h2f(ld16(yb, 0));
            let u0 = ld32a(yb, 4 + 4 * ((iqs / 2) % 4) as usize);
            let u1 = ld32a(yb, 4 + 4 * ((iqs / 2) % 4) as usize + 16);
            let dot1 = dp4a(v[i as usize][1], u1, dp4a(v[i as usize][0], u0, 0));
            let dot2 = dp4a(0x0101_0101, u1, dp4a(0x0101_0101, u0, 0));
            sd = fma(d8, (dot1 * s[i as usize] as i32) as f32, sd);
            sm = fma(d8, (dot2 * s[2 + i as usize] as i32) as f32, sm);
            i += 1;
        }
        let dm = ld32a(w, 0);
        let p = mul(sm, h2f(dm >> 16));
        add(acc, fma(sd, h2f(dm), -p))
    }

    #[inline(always)]
    unsafe fn step_q4_k(acc: f32, w: *const u8, y: *const u8, iqs: i32) -> f32 {
        let bq8_offset = 2 * ((iqs / 2) / 4);
        let q = 16 + 16 * bq8_offset as usize + 4 * ((iqs / 2) % 4) as usize;
        let (a, b) = (ld32a(w, q), ld32a(w, q + 16));
        let v = [[a & 0x0f0f_0f0f, b & 0x0f0f_0f0f], [(a >> 4) & 0x0f0f_0f0f, (b >> 4) & 0x0f0f_0f0f]];
        k_tail(acc, v, w, y, bq8_offset, iqs)
    }

    #[inline(always)]
    unsafe fn step_q5_k(acc: f32, w: *const u8, y: *const u8, iqs: i32) -> f32 {
        let bq8_offset = 2 * ((iqs / 2) / 4);
        let ql = 48 + 16 * bq8_offset as usize + 4 * ((iqs / 2) % 4) as usize;
        let qh = 16 + 4 * ((iqs / 2) % 4) as usize;
        let (l0, l1) = (ld32a(w, ql), ld32a(w, ql + 16));
        let h0 = ((ld32a(w, qh) as i32) >> bq8_offset) as u32;
        let h1 = ((ld32a(w, qh + 16) as i32) >> bq8_offset) as u32;
        let mut v = [[0u32; 2]; 2];
        let mut i = 0;
        while i < 2 {
            v[i][0] = ((l0 >> (4 * i)) & 0x0f0f_0f0f) | (((h0 >> i) << 4) & 0x1010_1010);
            v[i][1] = ((l1 >> (4 * i)) & 0x0f0f_0f0f) | (((h1 >> i) << 4) & 0x1010_1010);
            i += 1;
        }
        k_tail(acc, v, w, y, bq8_offset, iqs)
    }

    #[inline(always)]
    unsafe fn step_q6_k(acc: f32, w: *const u8, y: *const u8, iqs: i32) -> f32 {
        let bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        let scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        let vh_shift = 2 * ((iqs % 16) / 8);
        let vl = ld32u(w, 4 * iqs as usize) as i32;
        let vh = (ld32u(w, 128 + 4 * (8 * (iqs / 16) + iqs % 8) as usize) as i32) >> vh_shift;
        let d = h2f(ld16(w, 208));
        let mut sumf = 0f32;
        let mut i = 0;
        while i < 2 {
            let yb = y.add(36 * (bq8_offset + 2 * i) as usize);
            let u = ld32a(yb, 4 + 4 * (iqs % 8) as usize);
            let d8 = h2f(ld16(yb, 0));
            let sc = ld8(w, 192 + (scale_offset + 4 * i) as usize) as u8 as i8 as i32;
            let vil = ((vl >> (4 * i)) & 0x0f0f_0f0f) as u32;
            let vih = (((vh >> (4 * i)) << 4) & 0x3030_3030) as u32;
            let vi = vsubss4(vil | vih, 0x2020_2020);
            sumf = fma(d8, (dp4a(vi, u, 0) * sc) as f32, sumf);
            i += 1;
        }
        fma(d, sumf, acc)
    }

    /// `vllm_rs::moe_gemm_gguf_kernel`. TY: 0 Q8_0, 1 Q4_K, 2 Q2_K, 3 Q3_K, 4 Q5_K, 5 Q6_K.
    /// Grid (ceil(n/4), size_m), block (32, 4), dynamic shared size_k/qk*BS*4 + 1024.
    #[inline(always)]
    unsafe fn gguf_decode<const TY: u32>(
        weights: *const u8, y: *const u8, sorted: *const i32, expert_ids: *const i32, tw: *const f32, out: *mut f32,
        num_experts: i32, topk: i32, size_m: i32, size_n: i32, size_k: i32, k_padded: i32,
    ) {
        let (qk, qi, bs, vdr): (i32, i32, usize, i32) = match TY {
            0 => (32, 8, 34, 2),
            1 => (256, 32, 144, 2),
            2 => (256, 16, 84, 1),
            3 => (256, 16, 110, 1),
            4 => (256, 32, 176, 2),
            _ => (256, 32, 210, 1),
        };
        let lane = tid_x();
        let wrap_id = tid_y();
        let nwraps = thread::blockDim_y() as i32;
        let row = (thread::blockIdx_x() as i32).wrapping_mul(nwraps).wrapping_add(wrap_id);
        let m_idx = thread::blockIdx_y() as i32;
        if row >= size_n || m_idx >= size_m {
            return;
        }
        let w_stride = (size_n.wrapping_mul(size_k) as i64 as u64) / qk as u64 * bs as u64;
        let y_stride = (k_padded as i64 as u64) / 32 * 36;
        let token_id = *sorted.add(m_idx as usize);
        let expert = *expert_ids.add(m_idx as usize);
        if expert < 0 || expert >= num_experts {
            return;
        }
        let scale = if !tw.is_null() { *tw.offset(token_id as isize) } else { 1.0 };
        let w_expert = weights.add((expert as u64).wrapping_mul(w_stride) as usize);
        let input_index = if !tw.is_null() { token_id } else { token_id / topk };
        let y_ptr = y.add((input_index as i64 as u64).wrapping_mul(y_stride) as usize);
        let bpr = size_k / qk;
        let bpi = vdr * 32 / qi;
        let sh = DynamicSharedArray::<u8>::get_raw();
        let mut i = lane;
        while i < bpr {
            let dst = sh.offset((wrap_id * bpr + i) as isize * bs as isize);
            let src = w_expert.offset((row.wrapping_mul(bpr).wrapping_add(i)) as isize * bs as isize);
            let mut b = 0;
            while b < bs {
                *(dst.add(b) as *mut u16) = *(src.add(b) as *const u16);
                b += 2;
            }
            i += 32;
        }
        thread::sync_threads();
        let mut acc = 0f32;
        let mut kbx = lane / (qi / vdr);
        while kbx < bpr {
            let kby = kbx * (qk / 32);
            let kqs = vdr * (lane % (qi / vdr));
            let wb = sh.offset((wrap_id * bpr + kbx) as isize * bs as isize) as *const u8;
            let yb = y_ptr.offset(kby as isize * 36);
            acc = match TY {
                0 => step_q8_0(acc, wb, yb, kqs),
                1 => step_q4_k(acc, wb, yb, kqs),
                2 => step_q2_k(acc, wb, yb, kqs),
                3 => step_q3_k(acc, wb, yb, kqs),
                4 => step_q5_k(acc, wb, yb, kqs),
                _ => step_q6_k(acc, wb, yb, kqs),
            };
            kbx += bpi;
        }
        let mut mask = 16;
        while mask > 0 {
            acc = add(warp::shuffle_xor_f32_sync(FULL, acc, mask), acc);
            mask >>= 1;
        }
        let v = mul(acc, scale);
        if lane == 0 {
            let o = (token_id as i64 as u64).wrapping_mul(size_n as i64 as u64).wrapping_add(row as i64 as u64);
            *out.add(o as usize) = v;
        }
    }

    #[kernel]
    pub unsafe fn moe_gguf_q8_0(w: *const u8, y: *const u8, sorted: *const i32, eids: *const i32, tw: *const f32, out: *mut f32,
                                ne: i32, topk: i32, m: i32, n: i32, k: i32, kp: i32) {
        gguf_decode::<0>(w, y, sorted, eids, tw, out, ne, topk, m, n, k, kp)
    }
    #[kernel]
    pub unsafe fn moe_gguf_q4_k(w: *const u8, y: *const u8, sorted: *const i32, eids: *const i32, tw: *const f32, out: *mut f32,
                                ne: i32, topk: i32, m: i32, n: i32, k: i32, kp: i32) {
        gguf_decode::<1>(w, y, sorted, eids, tw, out, ne, topk, m, n, k, kp)
    }
    #[kernel]
    pub unsafe fn moe_gguf_q2_k(w: *const u8, y: *const u8, sorted: *const i32, eids: *const i32, tw: *const f32, out: *mut f32,
                                ne: i32, topk: i32, m: i32, n: i32, k: i32, kp: i32) {
        gguf_decode::<2>(w, y, sorted, eids, tw, out, ne, topk, m, n, k, kp)
    }
    #[kernel]
    pub unsafe fn moe_gguf_q3_k(w: *const u8, y: *const u8, sorted: *const i32, eids: *const i32, tw: *const f32, out: *mut f32,
                                ne: i32, topk: i32, m: i32, n: i32, k: i32, kp: i32) {
        gguf_decode::<3>(w, y, sorted, eids, tw, out, ne, topk, m, n, k, kp)
    }
    #[kernel]
    pub unsafe fn moe_gguf_q5_k(w: *const u8, y: *const u8, sorted: *const i32, eids: *const i32, tw: *const f32, out: *mut f32,
                                ne: i32, topk: i32, m: i32, n: i32, k: i32, kp: i32) {
        gguf_decode::<4>(w, y, sorted, eids, tw, out, ne, topk, m, n, k, kp)
    }
    #[kernel]
    pub unsafe fn moe_gguf_q6_k(w: *const u8, y: *const u8, sorted: *const i32, eids: *const i32, tw: *const f32, out: *mut f32,
                                ne: i32, topk: i32, m: i32, n: i32, k: i32, kp: i32) {
        gguf_decode::<5>(w, y, sorted, eids, tw, out, ne, topk, m, n, k, kp)
    }

    // ------------------------------------------------------------------ moe_wmma_gguf.cu
    /// Store one dequantised f16 value to a T (f16 or bf16) destination.
    #[inline(always)]
    unsafe fn st_h<const BF16: bool>(p: *mut u16, i: usize, h: u16) {
        *p.add(i) = if BF16 { f2bf(hf2f(h)) } else { h };
    }

    /// `get_scale_min_k4`.
    #[inline(always)]
    unsafe fn scale_min_k4(j: i32, q: *const u8) -> (u32, u32) {
        let j = j as usize;
        if j < 4 {
            (ld8(q, j) & 63, ld8(q, j + 4) & 63)
        } else {
            ((ld8(q, j + 4) & 0xF) | ((ld8(q, j - 4) >> 6) << 4), (ld8(q, j + 4) >> 4) | ((ld8(q, j) >> 6) << 4))
        }
    }

    /// `dequantize_block_warp` for the reachable (type, block) pairs. `tx` = threadIdx.x.
    #[inline(always)]
    unsafe fn dequant<const BF16: bool, const TY: u32>(yy: *mut u16, x: *const u8, tx: i32) {
        match TY {
            0 => {
                // Q8_0, 32 lanes: `T((float)qs[lane] * __half2float(d))`.
                if tx < 32 {
                    let d = h2f(ld16(x, 0));
                    let v = mul((*x.add(2 + tx as usize) as i8) as f32, d);
                    *yy.add(tx as usize) = if BF16 { f2bf(v) } else { f2h(v) };
                }
            }
            1 => {
                // Q4_K, 32 lanes.
                let il = tx / 8;
                let ir = tx % 8;
                let is = 2 * il;
                let y = yy.add((64 * il + 4 * ir) as usize);
                let dm = ld32a(x, 0);
                let (dall, dmin) = (dm as u16, (dm >> 16) as u16);
                let q = x.add(16 + (32 * il + 4 * ir) as usize);
                let (sc, m) = scale_min_k4(is, x.add(4));
                let d1 = hmul(dall, i2h(sc as i32));
                let m1 = hmul(dmin, i2h(m as i32));
                let (sc, m) = scale_min_k4(is + 1, x.add(4));
                let d2 = hmul(dall, i2h(sc as i32));
                let m2 = hmul(dmin, i2h(m as i32));
                let mut l = 0;
                while l < 4 {
                    let b = ld8(q, l);
                    st_h::<BF16>(y, l, hfma(d1, i2h((b & 0xF) as i32), m1 ^ 0x8000));
                    st_h::<BF16>(y, l + 32, hfma(d2, i2h((b >> 4) as i32), m2 ^ 0x8000));
                    l += 1;
                }
            }
            2 => {
                // Q2_K, 64 lanes.
                let n = tx / 32;
                let l = tx - 32 * n;
                let is = (8 * n + l / 16) as usize;
                let q = ld8(x, 16 + (32 * n + l) as usize);
                let y = yy.add((128 * n) as usize);
                let dm = ld32a(x, 80);
                let (dall, dmin) = (dm as u16, (dm >> 16) as u16);
                let mut j = 0;
                while j < 4 {
                    let sc = ld8(x, is + 2 * j);
                    let a = i2h(((sc & 0xF) * ((q >> (2 * j)) & 3)) as i32);
                    let p = hmul(dmin, i2h((sc >> 4) as i32));
                    st_h::<BF16>(y, l as usize + 32 * j, hfma(dall, a, p ^ 0x8000));
                    j += 1;
                }
            }
            3 => {
                // Q3_K, 64 lanes.
                let r = tx / 4;
                let t = r / 2;
                let is0 = r % 2;
                let l0 = 16 * is0 + 4 * (tx % 4);
                let n = t / 4;
                let j = t - 4 * n;
                let m = 1u32 << (4 * n + j);
                let is = 8 * n + 2 * j + is0;
                let shift = 2 * j;
                let s = |i: i32| ld8(x, 96 + i as usize);
                let us = if is < 4 {
                    (s(is) & 0xF) | ((s(is + 8) & 3) << 4)
                } else if is < 8 {
                    (s(is) & 0xF) | (((s(is + 4) >> 2) & 3) << 4)
                } else if is < 12 {
                    (s(is - 8) >> 4) | (((s(is) >> 4) & 3) << 4)
                } else {
                    (s(is - 8) >> 4) | (((s(is - 4) >> 6) & 3) << 4)
                };
                let d_all = ld16(x, 108) as u16;
                let dl = hmul(d_all, i2h(us as u8 as i8 as i32 - 32));
                let y = yy.add((128 * n + 32 * j) as usize);
                let q = x.add(32 + (32 * n) as usize);
                let mut l = l0;
                while l < l0 + 4 {
                    let qv = ((ld8(q, l as usize) >> shift) & 3) as i32;
                    let h = if ld8(x, l as usize) & m != 0 { 0 } else { 4 };
                    st_h::<BF16>(y, l as usize, hmul(dl, i2h(qv - h)));
                    l += 1;
                }
            }
            4 => {
                // Q5_K, 64 lanes.
                let il = tx / 16;
                let ir = tx % 16;
                let is = 2 * il;
                let y = yy.add((64 * il + 2 * ir) as usize);
                let dm = ld32a(x, 0);
                let (dall, dmin) = (dm as u16, (dm >> 16) as u16);
                let ql = x.add(48 + (32 * il + 2 * ir) as usize);
                let qh = x.add(16 + (2 * ir) as usize);
                let (sc, m) = scale_min_k4(is, x.add(4));
                let d1 = hmul(dall, i2h(sc as i32));
                let m1 = hmul(dmin, i2h(m as i32));
                let (sc, m) = scale_min_k4(is + 1, x.add(4));
                let d2 = hmul(dall, i2h(sc as i32));
                let m2 = hmul(dmin, i2h(m as i32));
                let hm = 1u32 << (2 * il);
                let hb = |i: usize, hm: u32| if ld8(qh, i) & hm != 0 { 16 } else { 0 };
                st_h::<BF16>(y, 0, hfma(d1, i2h(((ld8(ql, 0) & 0xF) + hb(0, hm)) as i32), m1 ^ 0x8000));
                st_h::<BF16>(y, 1, hfma(d1, i2h(((ld8(ql, 1) & 0xF) + hb(1, hm)) as i32), m1 ^ 0x8000));
                let hm = hm << 1;
                st_h::<BF16>(y, 32, hfma(d2, i2h(((ld8(ql, 0) >> 4) + hb(0, hm)) as i32), m2 ^ 0x8000));
                st_h::<BF16>(y, 33, hfma(d2, i2h(((ld8(ql, 1) >> 4) + hb(1, hm)) as i32), m2 ^ 0x8000));
            }
            _ => {
                // Q6_K, 64 lanes.
                let ip = tx / 32;
                let il = tx - 32 * ip;
                let is = (8 * ip + il / 16) as usize;
                let y = yy.add((128 * ip + il) as usize);
                let d = ld16(x, 208) as u16;
                let ql = x.add((64 * ip + il) as usize);
                let qh = ld8(x, 128 + (32 * ip + il) as usize);
                let sc = |i: usize| ld8(x, 192 + is + i) as u8 as i8 as i32;
                let v = |lo: u32, sh: u32| (((lo | (((qh >> sh) & 3) << 4)) as u8 as i8) as i32) - 32;
                st_h::<BF16>(y, 0, hmul(d, i2h(sc(0) * v(ld8(ql, 0) & 0xF, 0))));
                st_h::<BF16>(y, 32, hmul(d, i2h(sc(2) * v(ld8(ql, 32) & 0xF, 2))));
                st_h::<BF16>(y, 64, hmul(d, i2h(sc(4) * v(ld8(ql, 0) >> 4, 4))));
                st_h::<BF16>(y, 96, hmul(d, i2h(sc(6) * v(ld8(ql, 32) >> 4, 6))));
            }
        }
    }

    /// `moe_gemm_gguf_prefill_kernel<T, qk, block_q_t, wrap_size>`: grid (num_experts, ceil(n/32)),
    /// block (wrap, 4); wrap = 32 for Q8_0 / Q4_K, 64 otherwise.
    #[inline(always)]
    unsafe fn gguf_prefill<const BF16: bool, const TY: u32>(
        input: *const u16, weights: *const u8, sorted: *const i32, offsets: *const i32, tw: *const f32, output: *mut f32,
        num_experts: i32, topk: i32, size_m: i32, size_n: i32, size_k: i32,
    ) {
        let (qk, bs, wrap): (i32, usize, i32) = match TY {
            0 => (32, 34, 32),
            1 => (256, 144, 32),
            2 => (256, 84, 64),
            3 => (256, 110, 64),
            4 => (256, 176, 64),
            _ => (256, 210, 64),
        };
        let expert_id = thread::blockIdx_x() as i32;
        let n_tile = thread::blockIdx_y() as i32;
        if expert_id < 0 || expert_id >= num_experts {
            return;
        }
        let seg_start = *offsets.add(expert_id as usize);
        let seg_end = *offsets.add(expert_id as usize + 1);
        let rows = seg_end.wrapping_sub(seg_start);
        if rows == 0 {
            return;
        }
        let block_threads = 4 * wrap;
        let n_base = n_tile * 32;
        if n_base >= size_n {
            return;
        }
        let row_stride = ((size_k / qk) as i64 as u64).wrapping_mul(bs as u64);
        let expert_w = weights.add((expert_id as u64).wrapping_mul(size_n as i64 as u64).wrapping_mul(row_stride) as usize);
        let smem = DynamicSharedArray::<u8>::get_raw();
        let a_bytes = 32 * qk as usize * 2;
        let a_sh = smem as *mut u16;
        let b_sh = smem.add(a_bytes) as *mut u16;
        let bq_sh = smem.add(2 * a_bytes);
        let c_sh = smem.add(2 * a_bytes + 32 * bs) as *mut f32;
        let lane = tid_x();
        let warp_id = tid_y();
        let thread_id = warp_id * wrap + lane;
        let warp_m = warp_id / 2;
        let warp_n = warp_id % 2;
        let vec_a = (32 * qk / 8) as u64;
        let mut m_base = 0i32;
        while m_base < rows {
            let mut c = [0f32; 8];
            let mut k_base = 0i32;
            while k_base < size_k {
                let mut i = thread_id as u64;
                while i < vec_a {
                    let idx = i * 8;
                    let m_local = idx / qk as u64;
                    let k_local = idx % qk as u64;
                    let m_seg = (m_base as u64).wrapping_add(m_local) as i32;
                    let k_global = (k_base as u64).wrapping_add(k_local) as i32;
                    let dst = a_sh.add((m_local * qk as u64 + k_local) as usize) as *mut u8;
                    if m_seg < rows && k_global < size_k {
                        let token_index = *sorted.add((seg_start + m_seg) as usize);
                        let input_index = token_index / (if tw.is_null() { topk } else { 1 });
                        let off = (input_index as i64 as u64).wrapping_mul(size_k as i64 as u64).wrapping_add(k_global as i64 as u64);
                        copy16(dst, input.add(off as usize) as *const u8);
                    } else {
                        zero16(dst);
                    }
                    i += block_threads as u64;
                }
                let k_off = ((k_base / qk) as i64 as u64).wrapping_mul(bs as u64);
                let mut r = 0;
                while r < 8 {
                    let n_local = warp_id * 8 + r;
                    let n_global = n_base + n_local;
                    if n_local < 32 && n_global < size_n {
                        let dst = bq_sh.add(n_local as usize * bs);
                        let src = expert_w.add((n_global as i64 as u64).wrapping_mul(row_stride).wrapping_add(k_off) as usize);
                        let mut b = 2 * lane as usize;
                        while b < bs {
                            *(dst.add(b) as *mut u16) = *(src.add(b) as *const u16);
                            b += 2 * wrap as usize;
                        }
                    }
                    r += 1;
                }
                thread::sync_threads();
                let mut r = 0;
                while r < 8 {
                    let n_local = warp_id * 8 + r;
                    let n_global = n_base + n_local;
                    if n_local < 32 && n_global < size_n {
                        dequant::<BF16, TY>(b_sh.add((n_local * qk) as usize), bq_sh.add(n_local as usize * bs), lane);
                    }
                    r += 1;
                }
                thread::sync_threads();
                let mut k_tile = 0;
                while k_tile < qk {
                    wmma_step::<BF16, 0>(
                        &mut c,
                        a_sh.add((warp_m * 16 * qk + k_tile) as usize),
                        b_sh.add((warp_n * 16 * qk + k_tile) as usize),
                        qk as u32,
                    );
                    k_tile += 16;
                }
                k_base += qk;
            }
            wmma_store::<0>(c_sh.add((warp_m * 16 * 32 + warp_n * 16) as usize), &c, 32);
            thread::sync_threads();
            let mut i = thread_id;
            while i < 1024 {
                let m_local = i / 32;
                let n_local = i % 32;
                let m_seg = m_base + m_local;
                let n_global = n_base + n_local;
                if m_seg < rows && n_global < size_n {
                    let tp = seg_start + m_seg;
                    if tp < size_m {
                        let token_index = *sorted.add(tp as usize);
                        let mut val = *c_sh.add((m_local * 32 + n_local) as usize);
                        if !tw.is_null() {
                            val = mul(val, *tw.add(token_index as usize));
                        }
                        let o = (token_index as i64 as u64).wrapping_mul(size_n as i64 as u64).wrapping_add(n_global as i64 as u64);
                        *output.add(o as usize) = val;
                    }
                }
                i += block_threads;
            }
            m_base += 32;
        }
    }

    // GENERATED-BY-HAND: 12 prefill instances (T x gguf type).
    #[kernel]
    pub unsafe fn moe_gguf_prefill_f16_q8_0(i: *const u16, w: *const u8, s: *const i32, o: *const i32, tw: *const f32, out: *mut f32,
                                            ne: i32, topk: i32, m: i32, n: i32, k: i32) {
        gguf_prefill::<false, 0>(i, w, s, o, tw, out, ne, topk, m, n, k)
    }
    #[kernel]
    pub unsafe fn moe_gguf_prefill_f16_q4_k(i: *const u16, w: *const u8, s: *const i32, o: *const i32, tw: *const f32, out: *mut f32,
                                            ne: i32, topk: i32, m: i32, n: i32, k: i32) {
        gguf_prefill::<false, 1>(i, w, s, o, tw, out, ne, topk, m, n, k)
    }
    #[kernel]
    pub unsafe fn moe_gguf_prefill_f16_q2_k(i: *const u16, w: *const u8, s: *const i32, o: *const i32, tw: *const f32, out: *mut f32,
                                            ne: i32, topk: i32, m: i32, n: i32, k: i32) {
        gguf_prefill::<false, 2>(i, w, s, o, tw, out, ne, topk, m, n, k)
    }
    #[kernel]
    pub unsafe fn moe_gguf_prefill_f16_q3_k(i: *const u16, w: *const u8, s: *const i32, o: *const i32, tw: *const f32, out: *mut f32,
                                            ne: i32, topk: i32, m: i32, n: i32, k: i32) {
        gguf_prefill::<false, 3>(i, w, s, o, tw, out, ne, topk, m, n, k)
    }
    #[kernel]
    pub unsafe fn moe_gguf_prefill_f16_q5_k(i: *const u16, w: *const u8, s: *const i32, o: *const i32, tw: *const f32, out: *mut f32,
                                            ne: i32, topk: i32, m: i32, n: i32, k: i32) {
        gguf_prefill::<false, 4>(i, w, s, o, tw, out, ne, topk, m, n, k)
    }
    #[kernel]
    pub unsafe fn moe_gguf_prefill_f16_q6_k(i: *const u16, w: *const u8, s: *const i32, o: *const i32, tw: *const f32, out: *mut f32,
                                            ne: i32, topk: i32, m: i32, n: i32, k: i32) {
        gguf_prefill::<false, 5>(i, w, s, o, tw, out, ne, topk, m, n, k)
    }
    #[kernel]
    pub unsafe fn moe_gguf_prefill_bf16_q8_0(i: *const u16, w: *const u8, s: *const i32, o: *const i32, tw: *const f32, out: *mut f32,
                                             ne: i32, topk: i32, m: i32, n: i32, k: i32) {
        gguf_prefill::<true, 0>(i, w, s, o, tw, out, ne, topk, m, n, k)
    }
    #[kernel]
    pub unsafe fn moe_gguf_prefill_bf16_q4_k(i: *const u16, w: *const u8, s: *const i32, o: *const i32, tw: *const f32, out: *mut f32,
                                             ne: i32, topk: i32, m: i32, n: i32, k: i32) {
        gguf_prefill::<true, 1>(i, w, s, o, tw, out, ne, topk, m, n, k)
    }
    #[kernel]
    pub unsafe fn moe_gguf_prefill_bf16_q2_k(i: *const u16, w: *const u8, s: *const i32, o: *const i32, tw: *const f32, out: *mut f32,
                                             ne: i32, topk: i32, m: i32, n: i32, k: i32) {
        gguf_prefill::<true, 2>(i, w, s, o, tw, out, ne, topk, m, n, k)
    }
    #[kernel]
    pub unsafe fn moe_gguf_prefill_bf16_q3_k(i: *const u16, w: *const u8, s: *const i32, o: *const i32, tw: *const f32, out: *mut f32,
                                             ne: i32, topk: i32, m: i32, n: i32, k: i32) {
        gguf_prefill::<true, 3>(i, w, s, o, tw, out, ne, topk, m, n, k)
    }
    #[kernel]
    pub unsafe fn moe_gguf_prefill_bf16_q5_k(i: *const u16, w: *const u8, s: *const i32, o: *const i32, tw: *const f32, out: *mut f32,
                                             ne: i32, topk: i32, m: i32, n: i32, k: i32) {
        gguf_prefill::<true, 4>(i, w, s, o, tw, out, ne, topk, m, n, k)
    }
    #[kernel]
    pub unsafe fn moe_gguf_prefill_bf16_q6_k(i: *const u16, w: *const u8, s: *const i32, o: *const i32, tw: *const f32, out: *mut f32,
                                             ne: i32, topk: i32, m: i32, n: i32, k: i32) {
        gguf_prefill::<true, 5>(i, w, s, o, tw, out, ne, topk, m, n, k)
    }
}

mod gate;

fn main() {
    std::process::exit(if gate::run() { 0 } else { 1 });
}
