//! Blockwise E4M3 W8A16 GEMM in cuda-oxide (titan-engine), the oxide port of mistralrs-quant's
//! cuTile `fp8_w8a16_block_bf16` (src/cutile/fp8_w8a16.rs): y[M,N] = sum_k x[m,k] * e4m3(w[n,k]) * scale(n,k),
//! bf16 in/out, f32 accumulate, per-[bs,bs] weight-block f32 scale. Weight is [N,K] (y = x @ wT).
//!
//! Milestone 1: correct scalar accumulation (one thread per output element). Not tensor-core yet; this pins the
//! ABI, the E4M3/bf16 conversions and the block-scale indexing, verified numerically against a CPU f32 reference.
//! `main` builds random inputs, launches, and compares (bf16 tolerance); exits non-zero on failure.
#![allow(non_snake_case, clippy::missing_safety_doc, clippy::too_many_arguments)]

use cuda_device::{convert, kernel, ptx_asm, thread, SharedArray};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// bf16 bit-pattern -> f32 (exact 16-bit shift).
    #[inline(always)]
    fn bf16_f32(x: u16) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("cvt.f32.bf16 %0, %1;", out("=f") r, in("h") x, options(register_only)); }
        r
    }
    /// f32 -> bf16 bit-pattern, round-to-nearest-even.
    #[inline(always)]
    fn f32_bf16(x: f32) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("cvt.rn.bf16.f32 %0, %1;", out("=h") r, in("f") x, options(register_only)); }
        r
    }
    /// one E4M3 byte -> f32 (via e4m3x2 -> f16x2, low lane).
    #[inline(always)]
    fn e4m3_f32(b: u8) -> f32 {
        convert::cvt_f32_f16x2_lo(convert::cvt_rn_f16x2_e4m3x2(b as u16))
    }
    /// pack two bf16 bit-patterns into one u32 (lo in low 16 bits).
    #[inline(always)]
    fn pack2(lo: u16, hi: u16) -> u32 {
        (lo as u32) | ((hi as u32) << 16)
    }
    /// m16n8k16 bf16 MMA: D = A*B + C. A=[u32;4] (8 bf16/lane), B=[u32;2] (4 bf16/lane), C/D=[f32;4].
    #[inline(always)]
    fn mma_bf16(a: [u32; 4], b: [u32; 2], c: [f32; 4]) -> [f32; 4] {
        let (d0, d1, d2, d3): (f32, f32, f32, f32);
        unsafe {
            ptx_asm!(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};",
                out("=f") d0, out("=f") d1, out("=f") d2, out("=f") d3,
                in("r") a[0], in("r") a[1], in("r") a[2], in("r") a[3], in("r") b[0], in("r") b[1],
                in("f") c[0], in("f") c[1], in("f") c[2], in("f") c[3],
                options(register_only),
            );
        }
        [d0, d1, d2, d3]
    }

    /// pack four E4M3 bytes into one u32 (b0 in the low byte = lowest k).
    #[inline(always)]
    fn pack4(b0: u8, b1: u8, b2: u8, b3: u8) -> u32 {
        (b0 as u32) | ((b1 as u32) << 8) | ((b2 as u32) << 16) | ((b3 as u32) << 24)
    }
    /// m16n8k32 E4M3 MMA: D = A*B + C in f32. A=[u32;4] (16 e4m3/lane), B=[u32;2] (8 e4m3/lane), C/D=[f32;4].
    #[inline(always)]
    fn mma_fp8(a: [u32; 4], b: [u32; 2], c: [f32; 4]) -> [f32; 4] {
        let (d0, d1, d2, d3): (f32, f32, f32, f32);
        unsafe {
            ptx_asm!(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};",
                out("=f") d0, out("=f") d1, out("=f") d2, out("=f") d3,
                in("r") a[0], in("r") a[1], in("r") a[2], in("r") a[3], in("r") b[0], in("r") b[1],
                in("f") c[0], in("f") c[1], in("f") c[2], in("f") c[3],
                options(register_only),
            );
        }
        [d0, d1, d2, d3]
    }

    /// Tensor-core blockwise FP8 W8A16 GEMM: one warp per 16x8 output tile. Operand fragments are loaded manually
    /// by the standard m16n8k16 layout (A row-major m*k, B col-major k*n), weight converted E4M3->bf16 on load,
    /// the per-[bs,bs] block scale applied per accumulator register (each maps to its own (m,n)).
    #[kernel]
    pub unsafe fn fp8_w8a16_block_bf16_mma(
        y: *mut u16,
        x: *const u16,
        w: *const u8,
        ws: *const f32,
        m_dim: i32,
        n_dim: i32,
        k_dim: i32,
        bs: i32,
    ) {
        let (m_dim, n_dim, k_dim, bs) = (m_dim as usize, n_dim as usize, k_dim as usize, bs as usize);
        let ntn = (n_dim + 7) / 8;
        let bid = thread::blockIdx_x() as usize;
        let bm = (bid / ntn) * 16; // tile origin row (m)
        let bn = (bid % ntn) * 8; // tile origin col (n)
        let lane = thread::threadIdx_x() as usize; // 0..32
        let group = lane / 4;
        let tig = lane % 4;
        let nkb = (k_dim + bs - 1) / bs;

        let mut acc = [0f32; 4];
        let mut k0 = 0usize;
        while k0 < k_dim {
            // A fragment: a[r] packs (m_row, k_col) pairs per the m16k16 operand layout.
            let a_at = |mrow: usize, kc: usize| -> u16 {
                if bm + mrow < m_dim && k0 + kc < k_dim {
                    *x.add((bm + mrow) * k_dim + (k0 + kc))
                } else {
                    0
                }
            };
            let a = [
                pack2(a_at(group, 2 * tig), a_at(group, 2 * tig + 1)),
                pack2(a_at(group + 8, 2 * tig), a_at(group + 8, 2 * tig + 1)),
                pack2(a_at(group, 2 * tig + 8), a_at(group, 2 * tig + 8 + 1)),
                pack2(a_at(group + 8, 2 * tig + 8), a_at(group + 8, 2 * tig + 8 + 1)),
            ];
            // B fragment: b[r] packs (k_row, n_col=group) pairs, weight E4M3 -> bf16.
            let b_at = |kc: usize, ncol: usize| -> u16 {
                if bn + ncol < n_dim && k0 + kc < k_dim {
                    f32_bf16(e4m3_f32(*w.add((bn + ncol) * k_dim + (k0 + kc))))
                } else {
                    0
                }
            };
            let b = [
                pack2(b_at(2 * tig, group), b_at(2 * tig + 1, group)),
                pack2(b_at(2 * tig + 8, group), b_at(2 * tig + 8 + 1, group)),
            ];
            let part = mma_bf16(a, b, [0f32; 4]);
            // per-register block scale: register j -> (row=group+8*(j>>1), col=2*tig+(j&1)); n = bn+col, k-block = k0/bs.
            let kblk = k0 / bs;
            let mut j = 0usize;
            while j < 4 {
                let col = 2 * tig + (j & 1);
                let nj = bn + col;
                let sc = if nj < n_dim {
                    *ws.add((nj / bs) * nkb + kblk)
                } else {
                    0.0
                };
                acc[j] += part[j] * sc;
                j += 1;
            }
            k0 += 16;
        }
        // Epilogue: write each accumulator register to its (m,n).
        let mut j = 0usize;
        while j < 4 {
            let row = group + 8 * (j >> 1);
            let col = 2 * tig + (j & 1);
            let m = bm + row;
            let n = bn + col;
            if m < m_dim && n < n_dim {
                *y.add(m * n_dim + n) = f32_bf16(acc[j]);
            }
            j += 1;
        }
    }

    /// Routed grouped FP8 MoE GEMM (W8A16): the expert GEMM behind a fused MoE. Tokens are grouped by expert via
    /// the Triton `sorted_token_ids`/`expert_ids` layout; each warp computes a 16x8 tile for one expert's [N,K] FP8
    /// slab (converted to bf16, 128x128 block scale), gathers the A rows by token, multiplies each routed row by its
    /// top-k weight, and stores to the routed row. One output row per (token,expert) routing; the caller sums over k.
    #[kernel]
    pub unsafe fn fp8_moe_w8a16_mma(
        out: *mut u16,                 // [em_rows, N] bf16 (per routed row)
        x: *const u16,                 // [num_tokens, K] bf16
        w: *const u8,                  // [E, N, K] e4m3
        ws: *const f32,                // [E, N/bs, K/bs] block scales
        sorted_token_ids: *const i32,  // [em_rows], values in 0..num_tokens*top_k, padding >= num_valid
        expert_ids: *const i32,        // [em_rows/16] expert per 16-row block
        topk_weights: *const f32,      // [num_tokens*top_k]
        num_valid: i32,                // number of valid routed rows (rest are padding)
        n_dim: i32,
        k_dim: i32,
        bs: i32,
        top_k: i32,
    ) {
        let (n_dim, k_dim, bs, top_k) = (n_dim as usize, k_dim as usize, bs as usize, top_k as usize);
        let num_valid = num_valid as usize;
        let ntn = (n_dim + 7) / 8;
        let bid = thread::blockIdx_x() as usize;
        let mblk = bid / ntn; // which 16-row routed block
        let bn = (bid % ntn) * 8;
        let lane = thread::threadIdx_x() as usize;
        let group = lane / 4;
        let tig = lane % 4;
        let nkb = (k_dim + bs - 1) / bs;

        let expert = *expert_ids.add(mblk);
        if expert < 0 {
            return; // fully-padded block
        }
        let expert = expert as usize;
        let w_slab = expert * n_dim * k_dim;
        let ws_slab = expert * ((n_dim + bs - 1) / bs) * nkb;

        // Resolve the 16 routed rows of this block: sorted id -> token, valid flag, top-k weight.
        let srow = mblk * 16;
        let row_token = |r: usize| -> (usize, bool, f32) {
            let sid = *sorted_token_ids.add(srow + r);
            if (sid as usize) < num_valid {
                let token = (sid as usize) / top_k;
                (token, true, *topk_weights.add(sid as usize))
            } else {
                (0, false, 0.0)
            }
        };

        let mut acc = [0f32; 4];
        let mut k0 = 0usize;
        while k0 < k_dim {
            let a_at = |mrow: usize, kc: usize| -> u16 {
                let (token, valid, _) = row_token(mrow);
                if valid && k0 + kc < k_dim {
                    *x.add(token * k_dim + (k0 + kc))
                } else {
                    0
                }
            };
            let a = [
                pack2(a_at(group, 2 * tig), a_at(group, 2 * tig + 1)),
                pack2(a_at(group + 8, 2 * tig), a_at(group + 8, 2 * tig + 1)),
                pack2(a_at(group, 2 * tig + 8), a_at(group, 2 * tig + 8 + 1)),
                pack2(a_at(group + 8, 2 * tig + 8), a_at(group + 8, 2 * tig + 8 + 1)),
            ];
            let b_at = |kc: usize, ncol: usize| -> u16 {
                if bn + ncol < n_dim && k0 + kc < k_dim {
                    f32_bf16(e4m3_f32(*w.add(w_slab + (bn + ncol) * k_dim + (k0 + kc))))
                } else {
                    0
                }
            };
            let b = [
                pack2(b_at(2 * tig, group), b_at(2 * tig + 1, group)),
                pack2(b_at(2 * tig + 8, group), b_at(2 * tig + 8 + 1, group)),
            ];
            let part = mma_bf16(a, b, [0f32; 4]);
            let kblk = k0 / bs;
            let mut j = 0usize;
            while j < 4 {
                let col = 2 * tig + (j & 1);
                let nj = bn + col;
                let sc = if nj < n_dim { *ws.add(ws_slab + (nj / bs) * nkb + kblk) } else { 0.0 };
                acc[j] += part[j] * sc;
                j += 1;
            }
            k0 += 16;
        }
        let mut j = 0usize;
        while j < 4 {
            let row = group + 8 * (j >> 1);
            let col = 2 * tig + (j & 1);
            let (_, valid, tw) = row_token(row);
            let n = bn + col;
            if valid && n < n_dim {
                *out.add((srow + row) * n_dim + n) = f32_bf16(acc[j] * tw);
            }
            j += 1;
        }
    }

    /// Tensor-core routed FP8 W8A8 grouped MoE GEMM: FP8 activations and FP8 weights, native m16n8k32 e4m3 MMA.
    /// Per-token 1x128 activation scale `xs[token, k/128]` and per-[128,128] weight-block scale `ws[e, n/128, k/128]`;
    /// both share a 128-wide k-block, so four k=32 FP8 MMAs accumulate raw, then the combined scale is applied per block.
    /// Routing layout matches `fp8_moe_w8a16_mma` (sorted_token_ids / expert_ids / topk_weights). Out bf16 per routed row.
    #[kernel]
    pub unsafe fn fp8_moe_w8a8_mma(
        out: *mut u16,                 // [em_rows, N] bf16 (per routed row)
        x: *const u8,                  // [num_tokens, K] e4m3
        xs: *const f32,                // [num_tokens, K/128] per-token activation block scales
        w: *const u8,                  // [E, N, K] e4m3
        ws: *const f32,                // [E, N/128, K/128] weight block scales
        sorted_token_ids: *const i32,  // [em_rows], values in 0..num_tokens*top_k, padding >= num_valid
        expert_ids: *const i32,        // [em_rows/16] expert per 16-row block
        topk_weights: *const f32,      // [num_tokens*top_k]
        num_valid: i32,                // number of valid routed rows (rest are padding)
        n_dim: i32,
        k_dim: i32,
        bs: i32,                       // weight/activation scale block (128)
        top_k: i32,
    ) {
        let (n_dim, k_dim, bs, top_k) = (n_dim as usize, k_dim as usize, bs as usize, top_k as usize);
        let num_valid = num_valid as usize;
        let ntn = (n_dim + 7) / 8;
        let bid = thread::blockIdx_x() as usize;
        let mblk = bid / ntn; // which 16-row routed block
        let bn = (bid % ntn) * 8;
        let lane = thread::threadIdx_x() as usize;
        let group = lane / 4;
        let tig = lane % 4;
        let nkb = (k_dim + bs - 1) / bs;

        let expert = *expert_ids.add(mblk);
        if expert < 0 {
            return; // fully-padded block
        }
        let expert = expert as usize;
        let w_slab = expert * n_dim * k_dim;
        let ws_slab = expert * ((n_dim + bs - 1) / bs) * nkb;

        let srow = mblk * 16;
        let row_token = |r: usize| -> (usize, bool, f32) {
            let sid = *sorted_token_ids.add(srow + r);
            if (sid as usize) < num_valid {
                let token = (sid as usize) / top_k;
                (token, true, *topk_weights.add(sid as usize))
            } else {
                (0, false, 0.0)
            }
        };

        let mut acc = [0f32; 4];
        let mut kb = 0usize;
        while kb < k_dim {
            // Accumulate the four k=32 FP8 MMAs within this 128-wide scale block, raw (no scale yet).
            let mut part = [0f32; 4];
            let mut kk = 0usize;
            while kk < bs {
                let kbase = kb + kk;
                // A fragment: m16n8k32 e4m3 layout. Two k-chunks (kbase+tig*4, kbase+tig*4+16), rows group / group+8.
                let a_at = |mrow: usize, kcol: usize| -> u8 {
                    let (token, valid, _) = row_token(mrow);
                    if valid && kcol < k_dim {
                        *x.add(token * k_dim + kcol)
                    } else {
                        0
                    }
                };
                let c0 = kbase + tig * 4;
                let c1 = kbase + tig * 4 + 16;
                let a = [
                    pack4(a_at(group, c0), a_at(group, c0 + 1), a_at(group, c0 + 2), a_at(group, c0 + 3)),
                    pack4(a_at(group + 8, c0), a_at(group + 8, c0 + 1), a_at(group + 8, c0 + 2), a_at(group + 8, c0 + 3)),
                    pack4(a_at(group, c1), a_at(group, c1 + 1), a_at(group, c1 + 2), a_at(group, c1 + 3)),
                    pack4(a_at(group + 8, c1), a_at(group + 8, c1 + 1), a_at(group + 8, c1 + 2), a_at(group + 8, c1 + 3)),
                ];
                // B fragment: w[n,k] row-major, this lane holds n=group, same two k-chunks.
                let b_at = |kcol: usize, ncol: usize| -> u8 {
                    if bn + ncol < n_dim && kcol < k_dim {
                        *w.add(w_slab + (bn + ncol) * k_dim + kcol)
                    } else {
                        0
                    }
                };
                let b = [
                    pack4(b_at(c0, group), b_at(c0 + 1, group), b_at(c0 + 2, group), b_at(c0 + 3, group)),
                    pack4(b_at(c1, group), b_at(c1 + 1, group), b_at(c1 + 2, group), b_at(c1 + 3, group)),
                ];
                part = mma_fp8(a, b, part);
                kk += 32;
            }
            // Apply the per-block combined scale (per-token activation * weight block) to each accumulator register.
            let kblk = kb / bs;
            let mut j = 0usize;
            while j < 4 {
                let row = group + 8 * (j >> 1);
                let col = 2 * tig + (j & 1);
                let (token, valid, _) = row_token(row);
                let nj = bn + col;
                let asc = if valid { *xs.add(token * nkb + kblk) } else { 0.0 };
                let wsc = if nj < n_dim { *ws.add(ws_slab + (nj / bs) * nkb + kblk) } else { 0.0 };
                acc[j] += part[j] * asc * wsc;
                j += 1;
            }
            kb += bs;
        }
        let mut j = 0usize;
        while j < 4 {
            let row = group + 8 * (j >> 1);
            let col = 2 * tig + (j & 1);
            let (_, valid, tw) = row_token(row);
            let n = bn + col;
            if valid && n < n_dim {
                *out.add((srow + row) * n_dim + n) = f32_bf16(acc[j] * tw);
            }
            j += 1;
        }
    }

    /// Count routed rows per expert (expert_ids is the sorted-by-expert expert per routed row).
    #[kernel]
    pub unsafe fn moe_count(expert_ids: *const i32, counts: *mut i32, size_m: i32) {
        let i = (thread::blockIdx_x() as i32)
            .wrapping_mul(thread::blockDim_x() as i32)
            .wrapping_add(thread::threadIdx_x() as i32);
        if i < size_m {
            let e = *expert_ids.add(i as usize);
            let p = counts.offset(e as isize) as u64;
            ptx_asm!("red.global.add.s32 [%0], %1;", in("l") p, in("r") 1i32, clobber("memory"));
        }
    }

    /// Exclusive scan counts[0..n] -> offsets[0..=n] (offsets[0]=0), one block of 1024 threads.
    #[kernel]
    pub unsafe fn moe_scan(counts: *const i32, offsets: *mut i32, n: i32) {
        static mut PART: SharedArray<i32, 1024> = SharedArray::UNINIT;
        let part = SharedArray::as_raw_mut_ptr(&raw mut PART);
        let tid = thread::threadIdx_x() as i32;
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

    /// FP8 W8A16 grouped MoE GEMM in the (sorted, offsets, tw) convention of candle-moe `grouped`.
    /// One warp per (expert, 8-wide N tile); loops the expert's row segment in 16-row MMA tiles.
    /// `input` bf16 [*, K], `w` e4m3 [E, N, K], `ws` f32 block scales [E, N/bs, K/bs]. Output bf16
    /// [size_m, N] indexed by the sorted value (flat routed index). tw null (gate_up) divides the
    /// sorted value by topk to index `input`; tw set (down) indexes directly and scales by tw.
    #[kernel]
    pub unsafe fn fp8_moe_grouped(
        out: *mut u16,
        input: *const u16,
        w: *const u8,
        ws: *const f32,
        sorted: *const i32,
        offsets: *const i32,
        tw: *const f32,
        num_experts: i32,
        topk: i32,
        has_tw: i32,
        size_n: i32,
        size_k: i32,
        bs: i32,
    ) {
        let expert = thread::blockIdx_x() as i32;
        if expert >= num_experts {
            return;
        }
        let bn_i = (thread::blockIdx_y() as i32) * 8;
        if bn_i >= size_n {
            return;
        }
        let seg_start = *offsets.add(expert as usize);
        let seg_end = *offsets.add(expert as usize + 1);
        let rows = seg_end - seg_start;
        if rows <= 0 {
            return;
        }

        let (n_dim, k_dim, bs) = (size_n as usize, size_k as usize, bs as usize);
        let bn = bn_i as usize;
        let lane = thread::threadIdx_x() as usize;
        let group = lane / 4;
        let tig = lane % 4;
        let nkb = (k_dim + bs - 1) / bs;
        let w_slab = (expert as usize) * n_dim * k_dim;
        let ws_slab = (expert as usize) * ((n_dim + bs - 1) / bs) * nkb;
        let div = if has_tw == 0 { topk } else { 1 };

        let mut m_base = 0i32;
        while m_base < rows {
            let row_src = |r: usize| -> (usize, bool, usize) {
                let mseg = m_base + r as i32;
                if mseg < rows {
                    let tokidx = *sorted.add((seg_start + mseg) as usize);
                    ((tokidx / div) as usize, true, tokidx as usize)
                } else {
                    (0, false, 0)
                }
            };
            let mut acc = [0f32; 4];
            let mut k0 = 0usize;
            while k0 < k_dim {
                let a_at = |mrow: usize, kc: usize| -> u16 {
                    let (iidx, valid, _) = row_src(mrow);
                    if valid && k0 + kc < k_dim {
                        *input.add(iidx * k_dim + (k0 + kc))
                    } else {
                        0
                    }
                };
                let a = [
                    pack2(a_at(group, 2 * tig), a_at(group, 2 * tig + 1)),
                    pack2(a_at(group + 8, 2 * tig), a_at(group + 8, 2 * tig + 1)),
                    pack2(a_at(group, 2 * tig + 8), a_at(group, 2 * tig + 8 + 1)),
                    pack2(a_at(group + 8, 2 * tig + 8), a_at(group + 8, 2 * tig + 8 + 1)),
                ];
                let b_at = |kc: usize, ncol: usize| -> u16 {
                    if bn + ncol < n_dim && k0 + kc < k_dim {
                        f32_bf16(e4m3_f32(*w.add(w_slab + (bn + ncol) * k_dim + (k0 + kc))))
                    } else {
                        0
                    }
                };
                let b = [
                    pack2(b_at(2 * tig, group), b_at(2 * tig + 1, group)),
                    pack2(b_at(2 * tig + 8, group), b_at(2 * tig + 8 + 1, group)),
                ];
                let part = mma_bf16(a, b, [0f32; 4]);
                let kblk = k0 / bs;
                let mut j = 0usize;
                while j < 4 {
                    let col = 2 * tig + (j & 1);
                    let nj = bn + col;
                    let sc = if nj < n_dim { *ws.add(ws_slab + (nj / bs) * nkb + kblk) } else { 0.0 };
                    acc[j] += part[j] * sc;
                    j += 1;
                }
                k0 += 16;
            }
            let mut j = 0usize;
            while j < 4 {
                let row = group + 8 * (j >> 1);
                let col = 2 * tig + (j & 1);
                let (_, valid, tokidx) = row_src(row);
                let n = bn + col;
                if valid && n < n_dim {
                    let mut v = acc[j];
                    if has_tw != 0 {
                        v *= *tw.add(tokidx);
                    }
                    *out.add(tokidx * n_dim + n) = f32_bf16(v);
                }
                j += 1;
            }
            m_base += 16;
        }
    }

    #[kernel]
    pub unsafe fn fp8_w8a16_block_bf16(
        y: *mut u16,
        x: *const u16,
        w: *const u8,
        ws: *const f32,
        m_dim: i32,
        n_dim: i32,
        k_dim: i32,
        bs: i32,
    ) {
        let gid = thread::blockIdx_x()
            .wrapping_mul(thread::blockDim_x())
            .wrapping_add(thread::threadIdx_x()) as usize;
        let (m_dim, n_dim, k_dim, bs) = (m_dim as usize, n_dim as usize, k_dim as usize, bs as usize);
        if gid >= m_dim.wrapping_mul(n_dim) {
            return;
        }
        let m = gid / n_dim;
        let n = gid % n_dim;
        let nkb = (k_dim + bs - 1) / bs; // k-blocks per row of the scale grid
        let nblk = n / bs;
        let mut acc: f32 = 0.0;
        let mut k = 0usize;
        while k < k_dim {
            let xf = bf16_f32(*x.add(m * k_dim + k));
            let wf = e4m3_f32(*w.add(n * k_dim + k));
            let sc = *ws.add(nblk * nkb + k / bs);
            acc += xf * wf * sc;
            k += 1;
        }
        *y.add(m * n_dim + n) = f32_bf16(acc);
    }
}

// ---------------------------------------------------------------------------
// Host side: numerical self-test against a CPU f32 reference.
// ---------------------------------------------------------------------------
use cuda_core::{simt::launch_kernel_on_stream, CudaContext, DeviceBuffer};
use std::ffi::c_void;

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
fn f32_to_bf16(x: f32) -> u16 {
    // round-to-nearest-even, matching cvt.rn.bf16.f32 for finite inputs.
    let bits = x.to_bits();
    let round_bias = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(round_bias)) >> 16) as u16
}
fn e4m3_to_f32(b: u8) -> f32 {
    let s = (b >> 7) & 1;
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 0x7) as f32;
    let sign = if s == 1 { -1.0 } else { 1.0 };
    let v = if e == 0 {
        (m / 8.0) * 2f32.powi(-6) // subnormal: 2^(1-bias) * m/8, bias 7
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 7)
    };
    sign * v
}

struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    /// a "nice" bf16 in [-2, 2)
    fn bf16(&mut self) -> u16 {
        let f = (self.next_u32() as f32 / u32::MAX as f32) * 4.0 - 2.0;
        f32_to_bf16(f)
    }
    /// a safe E4M3 byte (avoid 0x7F/0xFF NaN)
    fn e4m3(&mut self) -> u8 {
        loop {
            let b = (self.next_u32() & 0xFF) as u8;
            if (b & 0x7F) != 0x7F {
                return b;
            }
        }
    }
    /// a small-magnitude E4M3 byte (exponent in [5,8] -> |v| in ~[0.25, 3.75]), like real MoE weights.
    fn e4m3_small(&mut self) -> u8 {
        let r = self.next_u32();
        let sign = ((r >> 15) & 1) as u8;
        let exp = 5 + (r & 3) as u8; // 5..8
        let man = ((r >> 4) & 7) as u8;
        (sign << 7) | (exp << 3) | man
    }
    fn scale(&mut self) -> f32 {
        0.25 + (self.next_u32() as f32 / u32::MAX as f32) * 1.5
    }
    /// small bf16 activation in [-0.2, 0.2], keeps a 2-GEMM FFN well-conditioned (no cancellation blowup).
    fn bf16_small(&mut self) -> u16 {
        let f = (self.next_u32() as f32 / u32::MAX as f32) * 0.4 - 0.2;
        f32_to_bf16(f)
    }
    /// mild block scale in [0.5, 0.9].
    fn scale_mild(&mut self) -> f32 {
        0.5 + (self.next_u32() as f32 / u32::MAX as f32) * 0.4
    }
}

fn main() {
    let shapes = [
        (37usize, 130usize, 256usize, 128usize), // partial tiles + n-block crossing + 2 k-blocks
        (1, 4096, 4096, 128),                     // decode: single row, realistic FFN width
        (256, 512, 512, 128),                     // square-ish, many tiles
        (16, 8, 128, 128),                        // exactly one warp tile, one k-block
        (128, 384, 1024, 128),                    // tall, 8 k-blocks, 3 n-blocks
    ];
    let ctx = CudaContext::new(0).expect("cuda context");
    let stream = ctx.default_stream();
    let ptx_path = concat!(env!("CARGO_MANIFEST_DIR"), "/fp8_blockwise.ptx");
    let module = ctx
        .load_module_from_ptx_src(&std::fs::read_to_string(ptx_path).expect(ptx_path))
        .expect("ptx load");
    let f = module.load_function("fp8_w8a16_block_bf16_mma").expect("entry");
    let mut all_ok = true;
    for (m, n, k, bs) in shapes {
        if !run_shape(&ctx, &stream, &f, m, n, k, bs) {
            all_ok = false;
        }
    }
    let f_moe = module.load_function("fp8_moe_w8a16_mma").expect("moe entry");
    for (t, n, k, e, top_k) in [(40usize, 128usize, 256usize, 8usize, 2usize), (96, 256, 512, 16, 4)] {
        if !run_moe(&stream, &f_moe, t, n, k, e, top_k, 128) {
            all_ok = false;
        }
    }
    let f_moe8 = module.load_function("fp8_moe_w8a8_mma").expect("moe w8a8 entry");
    for (t, n, k, e, top_k) in [(40usize, 128usize, 256usize, 8usize, 2usize), (96, 256, 512, 16, 4)] {
        if !run_moe_w8a8(&stream, &f_moe8, t, n, k, e, top_k, 128) {
            all_ok = false;
        }
    }
    let f_cnt = module.load_function("moe_count").expect("moe_count entry");
    let f_scan = module.load_function("moe_scan").expect("moe_scan entry");
    let f_grp = module.load_function("fp8_moe_grouped").expect("fp8_moe_grouped entry");
    for (t, n, k, e, top_k, htw) in [
        (24usize, 256usize, 384usize, 16usize, 4usize, false),
        (24, 256, 384, 16, 4, true),
        (64, 768, 128, 16, 4, false),
    ] {
        if !run_grouped_gemm(&stream, &f_cnt, &f_scan, &f_grp, t, n, k, e, top_k, htw, 128) {
            all_ok = false;
        }
    }
    for (t, h, i, e, top_k) in [(24usize, 128usize, 256usize, 8usize, 2usize), (64, 256, 384, 16, 4)] {
        if !run_moe_ffn(&stream, &f_cnt, &f_scan, &f_grp, t, h, i, e, top_k, 128) {
            all_ok = false;
        }
    }
    if !all_ok {
        std::process::exit(1);
    }
}

/// Routed FP8 MoE grouped GEMM test: random top-k routing, Triton-style sorted layout, per-routed-row compare.
fn run_moe(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    f: &cuda_core::CudaFunction,
    t: usize,
    n: usize,
    k: usize,
    e: usize,
    top_k: usize,
    bs: usize,
) -> bool {
    let nkb = k.div_ceil(bs);
    let nmb = n.div_ceil(bs);
    let mut rng = Rng(0xa5a5_1234_dead_beef);

    let x: Vec<u16> = (0..t * k).map(|_| rng.bf16()).collect();
    let w: Vec<u8> = (0..e * n * k).map(|_| rng.e4m3()).collect();
    let ws: Vec<f32> = (0..e * nmb * nkb).map(|_| rng.scale()).collect();

    // random top_k distinct experts per token + weights
    let mut topk_ids = vec![0i32; t * top_k];
    let mut topk_w = vec![0f32; t * top_k];
    for ti in 0..t {
        let mut chosen = Vec::new();
        while chosen.len() < top_k {
            let c = (rng.next_u32() as usize) % e;
            if !chosen.contains(&c) {
                chosen.push(c);
            }
        }
        for s in 0..top_k {
            topk_ids[ti * top_k + s] = chosen[s] as i32;
            topk_w[ti * top_k + s] = 0.1 + (rng.next_u32() as f32 / u32::MAX as f32);
        }
    }

    // moe_align: group routed sids by expert, pad each group to a multiple of 16.
    let num_valid = t * top_k;
    let mut per_expert: Vec<Vec<i32>> = vec![Vec::new(); e];
    for ti in 0..t {
        for s in 0..top_k {
            let sid = (ti * top_k + s) as i32;
            per_expert[topk_ids[ti * top_k + s] as usize].push(sid);
        }
    }
    let mut sorted_token_ids: Vec<i32> = Vec::new();
    let mut expert_ids: Vec<i32> = Vec::new();
    let pad_sid = num_valid as i32; // >= num_valid marks padding
    for (ex, rows) in per_expert.iter().enumerate() {
        if rows.is_empty() {
            continue;
        }
        let blocks = rows.len().div_ceil(16);
        for b in 0..blocks {
            expert_ids.push(ex as i32);
            for i in 0..16 {
                let idx = b * 16 + i;
                sorted_token_ids.push(if idx < rows.len() { rows[idx] } else { pad_sid });
            }
        }
    }
    let em_rows = sorted_token_ids.len();

    // CPU reference per valid routed row.
    let mut ref_out = vec![0u16; em_rows * n];
    for (blk, &expert) in expert_ids.iter().enumerate() {
        let expert = expert as usize;
        for r in 0..16 {
            let srow = blk * 16 + r;
            let sid = sorted_token_ids[srow];
            if (sid as usize) >= num_valid {
                continue;
            }
            let token = (sid as usize) / top_k;
            let tw = topk_w[sid as usize];
            for ni in 0..n {
                let mut acc = 0f32;
                for ki in 0..k {
                    acc += bf16_to_f32(x[token * k + ki])
                        * e4m3_to_f32(w[expert * n * k + ni * k + ki])
                        * ws[expert * nmb * nkb + (ni / bs) * nkb + ki / bs];
                }
                ref_out[srow * n + ni] = f32_to_bf16(acc * tw);
            }
        }
    }

    // GPU launch.
    let u16b = |s: &[u16]| s.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    let i32b = |s: &[i32]| s.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    let f32b = |s: &[f32]| s.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    let dout = DeviceBuffer::from_host(stream, &vec![0u8; em_rows * n * 2]).unwrap();
    let dx = DeviceBuffer::from_host(stream, &u16b(&x)).unwrap();
    let dw = DeviceBuffer::from_host(stream, &w.to_vec()).unwrap();
    let dws = DeviceBuffer::from_host(stream, &f32b(&ws)).unwrap();
    let dsid = DeviceBuffer::from_host(stream, &i32b(&sorted_token_ids)).unwrap();
    let deid = DeviceBuffer::from_host(stream, &i32b(&expert_ids)).unwrap();
    let dtw = DeviceBuffer::from_host(stream, &f32b(&topk_w)).unwrap();

    let mut vals: Vec<[u8; 8]> = vec![
        dout.cu_deviceptr().to_le_bytes(),
        dx.cu_deviceptr().to_le_bytes(),
        dw.cu_deviceptr().to_le_bytes(),
        dws.cu_deviceptr().to_le_bytes(),
        dsid.cu_deviceptr().to_le_bytes(),
        deid.cu_deviceptr().to_le_bytes(),
        dtw.cu_deviceptr().to_le_bytes(),
        (num_valid as u32 as u64).to_le_bytes(),
        (n as u32 as u64).to_le_bytes(),
        (k as u32 as u64).to_le_bytes(),
        (bs as u32 as u64).to_le_bytes(),
        (top_k as u32 as u64).to_le_bytes(),
    ];
    let mut ptrs: Vec<*mut c_void> = vals.iter_mut().map(|v| v.as_mut_ptr() as *mut c_void).collect();
    let grid = (em_rows / 16) as u32 * n.div_ceil(8) as u32;
    unsafe {
        launch_kernel_on_stream(f, (grid, 1, 1), (32, 1, 1), 0, stream, &mut ptrs).expect("moe launch");
    }
    stream.synchronize().expect("sync");
    let ob = dout.to_host_vec(stream).unwrap();
    let gpu: Vec<u16> = ob.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();

    let atol = 0.02 * (k as f32).sqrt();
    let mut fails = 0;
    let mut worst = 0f32;
    for i in 0..em_rows * n {
        let (fa, fb) = (bf16_to_f32(ref_out[i]), bf16_to_f32(gpu[i]));
        let ab = (fa - fb).abs();
        worst = worst.max(ab);
        if ab > atol + 0.02 * fa.abs() {
            fails += 1;
        }
    }
    println!(
        "fp8_moe_w8a16_mma: T={t} N={n} K={k} E={e} top_k={top_k}, {em_rows} routed rows, worst_abs={worst:.5}, {fails} failing -> {}",
        if fails == 0 { "PASS" } else { "FAIL" }
    );
    fails == 0
}

/// Routed FP8 W8A8 MoE grouped GEMM test: FP8 activations (per-token 1x128 scale) + FP8 weights (128x128 scale).
fn run_moe_w8a8(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    f: &cuda_core::CudaFunction,
    t: usize,
    n: usize,
    k: usize,
    e: usize,
    top_k: usize,
    bs: usize,
) -> bool {
    let nkb = k.div_ceil(bs);
    let nmb = n.div_ceil(bs);
    let mut rng = Rng(0x1357_9bdf_2468_ace0);

    // FP8 activations with a per-token, per-128-k scale; FP8 weights with a per-128x128 scale.
    let x: Vec<u8> = (0..t * k).map(|_| rng.e4m3()).collect();
    let xs: Vec<f32> = (0..t * nkb).map(|_| rng.scale()).collect();
    let w: Vec<u8> = (0..e * n * k).map(|_| rng.e4m3()).collect();
    let ws: Vec<f32> = (0..e * nmb * nkb).map(|_| rng.scale()).collect();

    let mut topk_ids = vec![0i32; t * top_k];
    let mut topk_w = vec![0f32; t * top_k];
    for ti in 0..t {
        let mut chosen = Vec::new();
        while chosen.len() < top_k {
            let c = (rng.next_u32() as usize) % e;
            if !chosen.contains(&c) {
                chosen.push(c);
            }
        }
        for s in 0..top_k {
            topk_ids[ti * top_k + s] = chosen[s] as i32;
            topk_w[ti * top_k + s] = 0.1 + (rng.next_u32() as f32 / u32::MAX as f32);
        }
    }

    let num_valid = t * top_k;
    let mut per_expert: Vec<Vec<i32>> = vec![Vec::new(); e];
    for ti in 0..t {
        for s in 0..top_k {
            let sid = (ti * top_k + s) as i32;
            per_expert[topk_ids[ti * top_k + s] as usize].push(sid);
        }
    }
    let mut sorted_token_ids: Vec<i32> = Vec::new();
    let mut expert_ids: Vec<i32> = Vec::new();
    let pad_sid = num_valid as i32;
    for (ex, rows) in per_expert.iter().enumerate() {
        if rows.is_empty() {
            continue;
        }
        for b in 0..rows.len().div_ceil(16) {
            expert_ids.push(ex as i32);
            for i in 0..16 {
                let idx = b * 16 + i;
                sorted_token_ids.push(if idx < rows.len() { rows[idx] } else { pad_sid });
            }
        }
    }
    let em_rows = sorted_token_ids.len();

    // CPU reference: both operands FP8, scaled per 128-block; accumulate in f32.
    let mut ref_out = vec![0u16; em_rows * n];
    for (blk, &expert) in expert_ids.iter().enumerate() {
        let expert = expert as usize;
        for r in 0..16 {
            let srow = blk * 16 + r;
            let sid = sorted_token_ids[srow];
            if (sid as usize) >= num_valid {
                continue;
            }
            let token = (sid as usize) / top_k;
            let tw = topk_w[sid as usize];
            for ni in 0..n {
                let mut acc = 0f32;
                for ki in 0..k {
                    acc += e4m3_to_f32(x[token * k + ki])
                        * e4m3_to_f32(w[expert * n * k + ni * k + ki])
                        * xs[token * nkb + ki / bs]
                        * ws[expert * nmb * nkb + (ni / bs) * nkb + ki / bs];
                }
                ref_out[srow * n + ni] = f32_to_bf16(acc * tw);
            }
        }
    }

    let i32b = |s: &[i32]| s.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    let f32b = |s: &[f32]| s.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    let dout = DeviceBuffer::from_host(stream, &vec![0u8; em_rows * n * 2]).unwrap();
    let dx = DeviceBuffer::from_host(stream, &x.to_vec()).unwrap();
    let dxs = DeviceBuffer::from_host(stream, &f32b(&xs)).unwrap();
    let dw = DeviceBuffer::from_host(stream, &w.to_vec()).unwrap();
    let dws = DeviceBuffer::from_host(stream, &f32b(&ws)).unwrap();
    let dsid = DeviceBuffer::from_host(stream, &i32b(&sorted_token_ids)).unwrap();
    let deid = DeviceBuffer::from_host(stream, &i32b(&expert_ids)).unwrap();
    let dtw = DeviceBuffer::from_host(stream, &f32b(&topk_w)).unwrap();

    let mut vals: Vec<[u8; 8]> = vec![
        dout.cu_deviceptr().to_le_bytes(),
        dx.cu_deviceptr().to_le_bytes(),
        dxs.cu_deviceptr().to_le_bytes(),
        dw.cu_deviceptr().to_le_bytes(),
        dws.cu_deviceptr().to_le_bytes(),
        dsid.cu_deviceptr().to_le_bytes(),
        deid.cu_deviceptr().to_le_bytes(),
        dtw.cu_deviceptr().to_le_bytes(),
        (num_valid as u32 as u64).to_le_bytes(),
        (n as u32 as u64).to_le_bytes(),
        (k as u32 as u64).to_le_bytes(),
        (bs as u32 as u64).to_le_bytes(),
        (top_k as u32 as u64).to_le_bytes(),
    ];
    let mut ptrs: Vec<*mut c_void> = vals.iter_mut().map(|v| v.as_mut_ptr() as *mut c_void).collect();
    let grid = (em_rows / 16) as u32 * n.div_ceil(8) as u32;
    unsafe {
        launch_kernel_on_stream(f, (grid, 1, 1), (32, 1, 1), 0, stream, &mut ptrs).expect("moe w8a8 launch");
    }
    stream.synchronize().expect("sync");
    let ob = dout.to_host_vec(stream).unwrap();
    let gpu: Vec<u16> = ob.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();

    let atol = 0.02 * (k as f32).sqrt();
    let mut fails = 0;
    let mut worst = 0f32;
    for i in 0..em_rows * n {
        let (fa, fb) = (bf16_to_f32(ref_out[i]), bf16_to_f32(gpu[i]));
        let ab = (fa - fb).abs();
        worst = worst.max(ab);
        if ab > atol + 0.02 * fa.abs() {
            fails += 1;
        }
    }
    println!(
        "fp8_moe_w8a8_mma:  T={t} N={n} K={k} E={e} top_k={top_k}, {em_rows} routed rows, worst_abs={worst:.5}, {fails} failing -> {}",
        if fails == 0 { "PASS" } else { "FAIL" }
    );
    fails == 0
}

/// Direct single-projection test of fp8_moe_grouped (+ moe_count/moe_scan align) vs a bf16-faithful
/// per-routed-row reference. Isolates the grouped kernel from the FFN's silu/sum compounding.
#[allow(clippy::too_many_arguments)]
fn run_grouped_gemm(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    f_cnt: &cuda_core::CudaFunction,
    f_scan: &cuda_core::CudaFunction,
    f_grp: &cuda_core::CudaFunction,
    t: usize,
    n: usize,
    k: usize,
    e: usize,
    top_k: usize,
    has_tw: bool,
    bs: usize,
) -> bool {
    let nkb = k.div_ceil(bs);
    let nmb = n.div_ceil(bs);
    let mut rng = Rng(0xbeef_0bad_1234_5678);
    let num_valid = t * top_k;
    let in_rows = if has_tw { num_valid } else { t };

    let input: Vec<u16> = (0..in_rows * k).map(|_| rng.bf16()).collect();
    let w: Vec<u8> = (0..e * n * k).map(|_| rng.e4m3_small()).collect();
    let ws: Vec<f32> = (0..e * nmb * nkb).map(|_| rng.scale()).collect();

    let mut topk_ids = vec![0usize; num_valid];
    let mut topk_w = vec![0f32; num_valid];
    for ti in 0..t {
        let mut chosen = Vec::new();
        while chosen.len() < top_k {
            let c = (rng.next_u32() as usize) % e;
            if !chosen.contains(&c) {
                chosen.push(c);
            }
        }
        for s in 0..top_k {
            topk_ids[ti * top_k + s] = chosen[s];
            topk_w[ti * top_k + s] = 0.1 + (rng.next_u32() as f32 / u32::MAX as f32);
        }
    }

    let mut order: Vec<usize> = (0..num_valid).collect();
    order.sort_by_key(|&sid| topk_ids[sid]);
    let sorted: Vec<i32> = order.iter().map(|&sid| sid as i32).collect();
    let expert_ids: Vec<i32> = order.iter().map(|&sid| topk_ids[sid] as i32).collect();

    let i32b = |s: &[i32]| s.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    let f32b = |s: &[f32]| s.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    let u16b = |s: &[u16]| s.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    let d_eid = DeviceBuffer::from_host(stream, &i32b(&expert_ids)).unwrap();
    let d_counts = DeviceBuffer::from_host(stream, &vec![0u8; e * 4]).unwrap();
    let d_off = DeviceBuffer::from_host(stream, &vec![0u8; (e + 1) * 4]).unwrap();
    let mut ca: Vec<[u8; 8]> = vec![
        d_eid.cu_deviceptr().to_le_bytes(),
        d_counts.cu_deviceptr().to_le_bytes(),
        (num_valid as u32 as u64).to_le_bytes(),
    ];
    let mut cp: Vec<*mut c_void> = ca.iter_mut().map(|v| v.as_mut_ptr() as *mut c_void).collect();
    unsafe {
        launch_kernel_on_stream(f_cnt, (((num_valid + 255) / 256).max(1) as u32, 1, 1), (256, 1, 1), 0, stream, &mut cp).unwrap();
    }
    let mut sa: Vec<[u8; 8]> = vec![
        d_counts.cu_deviceptr().to_le_bytes(),
        d_off.cu_deviceptr().to_le_bytes(),
        (e as u32 as u64).to_le_bytes(),
    ];
    let mut sp: Vec<*mut c_void> = sa.iter_mut().map(|v| v.as_mut_ptr() as *mut c_void).collect();
    unsafe {
        launch_kernel_on_stream(f_scan, (1, 1, 1), (1024, 1, 1), 0, stream, &mut sp).unwrap();
    }

    let d_sorted = DeviceBuffer::from_host(stream, &i32b(&sorted)).unwrap();
    let d_in = DeviceBuffer::from_host(stream, &u16b(&input)).unwrap();
    let d_w = DeviceBuffer::from_host(stream, &w).unwrap();
    let d_ws = DeviceBuffer::from_host(stream, &f32b(&ws)).unwrap();
    let d_tw = DeviceBuffer::from_host(stream, &f32b(&topk_w)).unwrap();
    let d_out = DeviceBuffer::from_host(stream, &vec![0u8; num_valid * n * 2]).unwrap();
    let htw_i = if has_tw { 1i32 } else { 0 };
    let mut ga: Vec<[u8; 8]> = vec![
        d_out.cu_deviceptr().to_le_bytes(),
        d_in.cu_deviceptr().to_le_bytes(),
        d_w.cu_deviceptr().to_le_bytes(),
        d_ws.cu_deviceptr().to_le_bytes(),
        d_sorted.cu_deviceptr().to_le_bytes(),
        d_off.cu_deviceptr().to_le_bytes(),
        d_tw.cu_deviceptr().to_le_bytes(),
        (e as u32 as u64).to_le_bytes(),
        (top_k as u32 as u64).to_le_bytes(),
        (htw_i as u32 as u64).to_le_bytes(),
        (n as u32 as u64).to_le_bytes(),
        (k as u32 as u64).to_le_bytes(),
        (bs as u32 as u64).to_le_bytes(),
    ];
    let mut gp: Vec<*mut c_void> = ga.iter_mut().map(|v| v.as_mut_ptr() as *mut c_void).collect();
    unsafe {
        launch_kernel_on_stream(f_grp, (e as u32, n.div_ceil(8) as u32, 1), (32, 1, 1), 0, stream, &mut gp).unwrap();
    }
    stream.synchronize().unwrap();
    let ob = d_out.to_host_vec(stream).unwrap();
    let gpu: Vec<u16> = ob.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();

    // bf16-faithful reference, indexed by sid.
    let mut refout = vec![0u16; num_valid * n];
    for sid in 0..num_valid {
        let ex = topk_ids[sid];
        let iidx = if has_tw { sid } else { sid / top_k };
        for ni in 0..n {
            let mut acc = 0f32;
            for ki in 0..k {
                acc += bf16_to_f32(input[iidx * k + ki])
                    * e4m3_to_f32(w[ex * n * k + ni * k + ki])
                    * ws[ex * nmb * nkb + (ni / bs) * nkb + ki / bs];
            }
            if has_tw {
                acc *= topk_w[sid];
            }
            refout[sid * n + ni] = f32_to_bf16(acc);
        }
    }

    let atol = 0.02 * (k as f32).sqrt();
    let mut fails = 0;
    let mut worst = 0f32;
    for idx in 0..num_valid * n {
        let (fa, fb) = (bf16_to_f32(refout[idx]), bf16_to_f32(gpu[idx]));
        let ab = (fa - fb).abs();
        worst = worst.max(ab);
        if ab > atol + 0.02 * fa.abs() {
            fails += 1;
        }
    }
    println!(
        "fp8_moe_grouped:   T={t} N={n} K={k} E={e} top_k={top_k} has_tw={has_tw}, {num_valid} rows, worst_abs={worst:.5}, {fails} failing -> {}",
        if fails == 0 { "PASS" } else { "FAIL" }
    );
    fails == 0
}

/// Full FP8 W8A16 MoE FFN: count+scan align, grouped gate_up, host GLU, grouped down (tw), host sum.
/// Mirrors the engine orchestration (candle argsort -> moe_gemm x2 -> act_and_mul -> moe_sum).
#[allow(clippy::too_many_arguments)]
fn run_moe_ffn(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    f_cnt: &cuda_core::CudaFunction,
    f_scan: &cuda_core::CudaFunction,
    f_grp: &cuda_core::CudaFunction,
    t: usize,
    h: usize,   // hidden
    i: usize,   // intermediate
    e: usize,
    top_k: usize,
    bs: usize,
) -> bool {
    let two_i = 2 * i;
    let mut rng = Rng(0x0f0f_1234_5678_9abc);
    let silu = |x: f32| x / (1.0 + (-x).exp());

    // Weights: gate_up [E, 2I, H] e4m3 + scales [E, 2I/bs, H/bs]; down [E, H, I] e4m3 + scales.
    let gu_nkb = h.div_ceil(bs);
    let gu_nmb = two_i.div_ceil(bs);
    let dn_nkb = i.div_ceil(bs);
    let dn_nmb = h.div_ceil(bs);
    let gu_w: Vec<u8> = (0..e * two_i * h).map(|_| rng.e4m3_small()).collect();
    let gu_s: Vec<f32> = (0..e * gu_nmb * gu_nkb).map(|_| rng.scale_mild()).collect();
    let dn_w: Vec<u8> = (0..e * h * i).map(|_| rng.e4m3_small()).collect();
    let dn_s: Vec<f32> = (0..e * dn_nmb * dn_nkb).map(|_| rng.scale_mild()).collect();
    let xs: Vec<u16> = (0..t * h).map(|_| rng.bf16_small()).collect();

    // Router: distinct top_k experts per token + weights.
    let mut topk_ids = vec![0usize; t * top_k];
    let mut topk_w = vec![0f32; t * top_k];
    for ti in 0..t {
        let mut chosen = Vec::new();
        while chosen.len() < top_k {
            let c = (rng.next_u32() as usize) % e;
            if !chosen.contains(&c) {
                chosen.push(c);
            }
        }
        for s in 0..top_k {
            topk_ids[ti * top_k + s] = chosen[s];
            topk_w[ti * top_k + s] = 0.1 + (rng.next_u32() as f32 / u32::MAX as f32);
        }
    }

    // Align by argsort: group the num_valid routed indices by expert (stable).
    let num_valid = t * top_k;
    let mut order: Vec<usize> = (0..num_valid).collect();
    order.sort_by_key(|&sid| topk_ids[sid]);
    let sorted: Vec<i32> = order.iter().map(|&sid| sid as i32).collect();
    let expert_ids: Vec<i32> = order.iter().map(|&sid| topk_ids[sid] as i32).collect();
    let tw_by_sid: Vec<f32> = topk_w.clone(); // tw indexed by sid (down reads tw[tokidx=sid])

    // Device buffers for align.
    let i32b = |s: &[i32]| s.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    let f32b = |s: &[f32]| s.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    let u16b = |s: &[u16]| s.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    let d_eid = DeviceBuffer::from_host(stream, &i32b(&expert_ids)).unwrap();
    let d_counts = DeviceBuffer::from_host(stream, &vec![0u8; e * 4]).unwrap();
    let d_off = DeviceBuffer::from_host(stream, &vec![0u8; (e + 1) * 4]).unwrap();

    // moe_count then moe_scan -> offsets.
    let mut cnt_args: Vec<[u8; 8]> = vec![
        d_eid.cu_deviceptr().to_le_bytes(),
        d_counts.cu_deviceptr().to_le_bytes(),
        (num_valid as u32 as u64).to_le_bytes(),
    ];
    let mut cnt_ptrs: Vec<*mut c_void> = cnt_args.iter_mut().map(|v| v.as_mut_ptr() as *mut c_void).collect();
    let grid_cnt = ((num_valid + 255) / 256) as u32;
    unsafe {
        launch_kernel_on_stream(f_cnt, (grid_cnt.max(1), 1, 1), (256, 1, 1), 0, stream, &mut cnt_ptrs).expect("count");
    }
    let mut scan_args: Vec<[u8; 8]> = vec![
        d_counts.cu_deviceptr().to_le_bytes(),
        d_off.cu_deviceptr().to_le_bytes(),
        (e as u32 as u64).to_le_bytes(),
    ];
    let mut scan_ptrs: Vec<*mut c_void> = scan_args.iter_mut().map(|v| v.as_mut_ptr() as *mut c_void).collect();
    unsafe {
        launch_kernel_on_stream(f_scan, (1, 1, 1), (1024, 1, 1), 0, stream, &mut scan_ptrs).expect("scan");
    }
    stream.synchronize().expect("sync");

    // Grouped launch helper.
    let d_sorted = DeviceBuffer::from_host(stream, &i32b(&sorted)).unwrap();
    let run_grouped = |out_buf: &DeviceBuffer<u8>, input: &DeviceBuffer<u8>, w: &DeviceBuffer<u8>, ws: &DeviceBuffer<u8>,
                       tw: &DeviceBuffer<u8>, has_tw: i32, n: usize, k: usize| {
        let mut args: Vec<[u8; 8]> = vec![
            out_buf.cu_deviceptr().to_le_bytes(),
            input.cu_deviceptr().to_le_bytes(),
            w.cu_deviceptr().to_le_bytes(),
            ws.cu_deviceptr().to_le_bytes(),
            d_sorted.cu_deviceptr().to_le_bytes(),
            d_off.cu_deviceptr().to_le_bytes(),
            tw.cu_deviceptr().to_le_bytes(),
            (e as u32 as u64).to_le_bytes(),
            (top_k as u32 as u64).to_le_bytes(),
            (has_tw as u32 as u64).to_le_bytes(),
            (n as u32 as u64).to_le_bytes(),
            (k as u32 as u64).to_le_bytes(),
            (bs as u32 as u64).to_le_bytes(),
        ];
        let mut ptrs: Vec<*mut c_void> = args.iter_mut().map(|v| v.as_mut_ptr() as *mut c_void).collect();
        let grid = (e as u32, n.div_ceil(8) as u32, 1);
        unsafe {
            launch_kernel_on_stream(f_grp, grid, (32, 1, 1), 0, stream, &mut ptrs).expect("grouped");
        }
    };

    // GEMM1: gate_up (tw null). out [num_valid, 2I] in flat-routed order.
    let d_xs = DeviceBuffer::from_host(stream, &u16b(&xs)).unwrap();
    let d_guw = DeviceBuffer::from_host(stream, &gu_w).unwrap();
    let d_gus = DeviceBuffer::from_host(stream, &f32b(&gu_s)).unwrap();
    let d_tw = DeviceBuffer::from_host(stream, &f32b(&tw_by_sid)).unwrap();
    let d_c1 = DeviceBuffer::from_host(stream, &vec![0u8; num_valid * two_i * 2]).unwrap();
    run_grouped(&d_c1, &d_xs, &d_guw, &d_gus, &d_tw, 0, two_i, h);
    stream.synchronize().expect("sync");
    let c1b = d_c1.to_host_vec(stream).unwrap();
    let c1: Vec<u16> = c1b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();

    // Host GLU: silu(gate) * up -> [num_valid, I] bf16.
    let mut glu = vec![0u16; num_valid * i];
    for r in 0..num_valid {
        for c in 0..i {
            let g = bf16_to_f32(c1[r * two_i + c]);
            let u = bf16_to_f32(c1[r * two_i + i + c]);
            glu[r * i + c] = f32_to_bf16(silu(g) * u);
        }
    }

    // GEMM2: down (tw set). out [num_valid, H].
    let d_glu = DeviceBuffer::from_host(stream, &u16b(&glu)).unwrap();
    let d_dnw = DeviceBuffer::from_host(stream, &dn_w).unwrap();
    let d_dns = DeviceBuffer::from_host(stream, &f32b(&dn_s)).unwrap();
    let d_c2 = DeviceBuffer::from_host(stream, &vec![0u8; num_valid * h * 2]).unwrap();
    run_grouped(&d_c2, &d_glu, &d_dnw, &d_dns, &d_tw, 1, h, i);
    stream.synchronize().expect("sync");
    let c2b = d_c2.to_host_vec(stream).unwrap();
    let c2: Vec<u16> = c2b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();

    // Host sum over top_k per token -> [T, H].
    let mut gpu_out = vec![0f32; t * h];
    for ti in 0..t {
        for s in 0..top_k {
            let sid = ti * top_k + s;
            for c in 0..h {
                gpu_out[ti * h + c] += bf16_to_f32(c2[sid * h + c]);
            }
        }
    }

    // CPU reference FFN.
    let mut ref_out = vec![0f32; t * h];
    for ti in 0..t {
        for s in 0..top_k {
            let sid = ti * top_k + s;
            let ex = topk_ids[sid];
            let tw = topk_w[sid];
            // gate_up: [2I] = xs[ti] @ gu[ex]^T (block scaled), stored bf16 (matches GPU c1).
            let mut gu_row = vec![0f32; two_i];
            for ni in 0..two_i {
                let mut acc = 0f32;
                for ki in 0..h {
                    acc += bf16_to_f32(xs[ti * h + ki])
                        * e4m3_to_f32(gu_w[ex * two_i * h + ni * h + ki])
                        * gu_s[ex * gu_nmb * gu_nkb + (ni / bs) * gu_nkb + ki / bs];
                }
                gu_row[ni] = bf16_to_f32(f32_to_bf16(acc));
            }
            // GLU: silu(gate) * up, stored bf16.
            let mut gact = vec![0f32; i];
            for c in 0..i {
                gact[c] = bf16_to_f32(f32_to_bf16(silu(gu_row[c]) * gu_row[i + c]));
            }
            // down: [H] = glu @ dn[ex]^T (block scaled) * tw, stored bf16, accumulated into token.
            for ni in 0..h {
                let mut acc = 0f32;
                for ki in 0..i {
                    acc += gact[ki]
                        * e4m3_to_f32(dn_w[ex * h * i + ni * i + ki])
                        * dn_s[ex * dn_nmb * dn_nkb + (ni / bs) * dn_nkb + ki / bs];
                }
                ref_out[ti * h + ni] += bf16_to_f32(f32_to_bf16(acc * tw));
            }
        }
    }

    // Compare (bf16 round-trip through two GEMMs + glu; GEMM noise floor + relative).
    // Two chained bf16 GEMMs + silu: noise floor is the final GEMM's f32 accumulation noise (0.1*sqrt(inner),
    // the same family as the single-GEMM tests) times the top_k reduction, plus a relative term.
    let atol = 0.1 * (i as f32).sqrt() * (top_k as f32).sqrt();
    let mut fails = 0;
    let mut worst = 0f32;
    for idx in 0..t * h {
        let (fa, fb) = (ref_out[idx], gpu_out[idx]);
        let ab = (fa - fb).abs();
        worst = worst.max(ab);
        if ab > atol + 0.03 * fa.abs() {
            fails += 1;
        }
    }
    println!(
        "fp8_moe_ffn:       T={t} H={h} I={i} E={e} top_k={top_k}, {num_valid} routed rows, worst_abs={worst:.5}, {fails} failing -> {}",
        if fails == 0 { "PASS" } else { "FAIL" }
    );
    fails == 0
}

fn run_shape(
    ctx: &CudaContext,
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    f: &cuda_core::CudaFunction,
    m: usize,
    n: usize,
    k: usize,
    bs: usize,
) -> bool {
    let _ = ctx;
    let nmb = (n + bs - 1) / bs;
    let nkb = (k + bs - 1) / bs;
    let mut rng = Rng(0x1234_5678_9abc_def0);

    let x: Vec<u16> = (0..m * k).map(|_| rng.bf16()).collect();
    let w: Vec<u8> = (0..n * k).map(|_| rng.e4m3()).collect();
    let ws: Vec<f32> = (0..nmb * nkb).map(|_| rng.scale()).collect();

    // CPU reference.
    let mut y_ref = vec![0u16; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0f32;
            for ki in 0..k {
                let xf = bf16_to_f32(x[mi * k + ki]);
                let wf = e4m3_to_f32(w[ni * k + ki]);
                let sc = ws[(ni / bs) * nkb + ki / bs];
                acc += xf * wf * sc;
            }
            y_ref[mi * n + ni] = f32_to_bf16(acc);
        }
    }

    // GPU launch (ctx/stream/f passed in).
    let as_bytes = |s: &[u8]| s.to_vec();
    let u16_bytes = |s: &[u16]| s.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    let f32_bytes = |s: &[f32]| s.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();

    let dy = DeviceBuffer::from_host(stream, &vec![0u8; m * n * 2]).unwrap();
    let dx = DeviceBuffer::from_host(stream, &u16_bytes(&x)).unwrap();
    let dw = DeviceBuffer::from_host(stream, &as_bytes(&w)).unwrap();
    let dws = DeviceBuffer::from_host(stream, &f32_bytes(&ws)).unwrap();

    let mut vals: Vec<[u8; 8]> = vec![
        dy.cu_deviceptr().to_le_bytes(),
        dx.cu_deviceptr().to_le_bytes(),
        dw.cu_deviceptr().to_le_bytes(),
        dws.cu_deviceptr().to_le_bytes(),
        (m as u32 as u64).to_le_bytes(),
        (n as u32 as u64).to_le_bytes(),
        (k as u32 as u64).to_le_bytes(),
        (bs as u32 as u64).to_le_bytes(),
    ];
    let mut ptrs: Vec<*mut c_void> = vals.iter_mut().map(|v| v.as_mut_ptr() as *mut c_void).collect();

    // warp (32 lanes) per 16x8 output tile.
    let ntm = m.div_ceil(16) as u32;
    let ntn = n.div_ceil(8) as u32;
    let grid = ntm * ntn;
    unsafe {
        launch_kernel_on_stream(f, (grid, 1, 1), (32, 1, 1), 0, stream, &mut ptrs).expect("launch");
    }
    stream.synchronize().expect("sync");
    let y_bytes = dy.to_host_vec(stream).unwrap();
    let y_gpu: Vec<u16> = y_bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();

    // Compare vs the sequential-f32 reference. The tensor core accumulates in a different order, so near a
    // cancellation the bf16 result can differ by the f32 noise floor ~ sqrt(K)*eps*|partials|. A real layout bug
    // corrupts many elements systematically; so use an absolute floor that scales with K plus a relative term.
    let atol = 0.01 * (k as f32).sqrt();
    let mut worst_ulp = 0i32;
    let mut worst_abs = 0f32;
    let mut fails = 0usize;
    for i in 0..m * n {
        let (a, b) = (y_ref[i], y_gpu[i]);
        worst_ulp = worst_ulp.max((a as i32 - b as i32).abs());
        let (fa, fb) = (bf16_to_f32(a), bf16_to_f32(b));
        let abs = (fa - fb).abs();
        worst_abs = worst_abs.max(abs);
        if abs > atol + 0.02 * fa.abs() {
            if fails < 8 {
                eprintln!("  mismatch [{i}]: ref={fa} gpu={fb} abs={abs:.5} (atol={atol:.4})");
            }
            fails += 1;
        }
    }
    println!(
        "fp8_w8a16_block_bf16_mma: {m}x{n}x{k} bs={bs}, {} elems, worst_ulp={worst_ulp}, worst_abs={worst_abs:.5}, {fails} failing -> {}",
        m * n,
        if fails == 0 { "PASS" } else { "FAIL" }
    );
    fails == 0
}
