#![allow(unsafe_op_in_unsafe_fn)]
//! Split-K flash-decoding attention for titan-mistral's long-context decode (Qwen3.6-35B-A3B:
//! head_dim 256, bf16 KV, GQA). One launch per attention layer and decode step covers all query rows
//! of the step (1 decode row, or the MTP verify rows); each KV head is read once for its
//! `n_rep` query heads x rows.
//!
//! `flash_decode_split_g{8,16,24}`: grid (nchunks, kv_heads), block 256. Block (c, h) handles keys
//! `[c * chunk, min((c + 1) * chunk, kv_total))` of KV head h for the G = n_rep * n_rows queries of that
//! head, in tiles of 128 keys, with an online softmax per tile, and writes the un-normalised partial
//! `VKQ` (f32, 256 per query) plus (max, sum) per query and chunk. `flash_decode_combine`: grid
//! (n_heads, n_rows), block 256, merges a row's chunks in chunk order and writes bf16.
//!
//! Numerics follow llama.cpp's fattn-vec (acecd56, MIT) for bf16 K/V on NVIDIA: Q converted to f32
//! and pre-multiplied by the softmax scale, f32 K.Q dots, a running max shifted by
//! FATTN_KQ_MAX_OFFSET, fast-math `expf` (ex2.approx(x * log2e)), f32 V accumulation, and
//! flash_attn_combine_results' merge (`sum_c exp(m_c - M) * VKQ_c / sum_c exp(m_c - M) * s_c`).
//! What differs from fattn-vec is the work split (GQA-packed queries, 2 threads per key, fixed
//! tiles) and so the summation order.
//!
//! Row exactness (the MTP invariant): query row r sees keys `< kv_total - n_rows + r + 1`. Tiles and
//! chunks start at fixed multiples of 128 / `chunk` from key 0, masked keys contribute exact zeros,
//! a tile with no valid key for a query leaves its state untouched (scale exactly 1), and every
//! reduction runs in a fixed order over one query's own values. So row r of an n-row launch is
//! bit-identical to a 1-row launch with kv_total = its own key count, whatever G instance runs it
//! (all float ops are explicit PTX: nothing is contracted differently per instance). `gate.rs` checks
//! this bit for bit, plus accuracy against an f64 reference.
//!
//! mistral.rs embeds flash_decode.ptx (mistralrs-core/src/attention/flash_decode_oxide.ptx, see
//! export_ptx.sh) and launches it from Rust (attention/flash_decode.rs), in both the nvcc and the
//! nvcc-free build.
#![allow(non_snake_case, clippy::missing_safety_doc, dead_code)]

use cuda_device::{SharedArray, device, float, kernel, ptx_asm, thread, warp};
use cuda_host::cuda_module;

mod gate;

#[cuda_module]
pub mod kernels {
    use super::*;

    pub const D: i32 = 256;
    /// Keys per tile: 2 threads per key, 256 threads.
    pub const TILE: i32 = 128;
    /// Largest query count per block (n_rep * n_rows): 8 x 3 for Qwen3.6 MTP=2.
    pub const GMAX: usize = 24;
    // Shared memory, in f32 words: Q [GMAX][256], P [TILE][G], RED [8][GMAX], M, SCALE, SUM, LEN [GMAX].
    pub const OFF_Q: u32 = 0;
    pub const OFF_P: u32 = 6144;
    pub const OFF_RED: u32 = 6144 + 3072;
    pub const OFF_M: u32 = OFF_RED + 192;
    pub const OFF_SC: u32 = OFF_M + 24;
    pub const OFF_SUM: u32 = OFF_SC + 24;
    pub const OFF_LEN: u32 = OFF_SUM + 24;
    pub const SMEM_WORDS: usize = 9504;

    pub const LOG2E: f32 = f32::from_bits(0x3fb8aa3b);
    /// llama.cpp FATTN_KQ_MAX_OFFSET = 3.0f*0.6931f (f32 product).
    pub const KQ_MAX_OFFSET: f32 = 3.0f32 * 0.6931f32;
    /// -FLT_MAX/2.0f, llama.cpp's initial running max.
    pub const NEG_HALF_MAX: f32 = f32::from_bits(0xfeffffff);
    pub const NEG_INF: f32 = f32::from_bits(0xff800000);

    // ------------------------------------------------------------------------------------------
    // f32 ops as explicit PTX (fast-math .ftz like llama.cpp's build; nothing is left for ptxas to
    // contract, so every G instance rounds identically). The trailing comment keeps cuda-oxide's
    // f32x2 scan cheap (PORTING.md rule 38).

    #[inline(always)]
    pub fn fma(a: f32, b: f32, c: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("fma.rn.ftz.f32 %0, %1, %2, %3; //\t\x0b\x0c\r\n", out("=f") r, in("f") a, in("f") b, in("f") c, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn mul(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("mul.rn.ftz.f32 %0, %1, %2; //\t\x0b\x0c\r\n", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn add(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("add.rn.ftz.f32 %0, %1, %2; //\t\x0b\x0c\r\n", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn sub(a: f32, b: f32) -> f32 {
        add(a, -b)
    }
    /// fmaxf under fast-math: `max.ftz.f32`.
    #[inline(always)]
    pub fn fmaxf(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("max.ftz.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    /// IEEE division (the combine's final normalisation).
    #[inline(always)]
    pub fn div_rn(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("div.rn.ftz.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    /// fast-math `expf(x)`: `ex2.approx.ftz(x * log2e)` (PORTING.md rule 28).
    #[inline(always)]
    pub fn expf(x: f32) -> f32 {
        float::ex2_approx_ftz_f32(mul(x, LOG2E))
    }
    #[inline(always)]
    pub fn bf_lo(w: u32) -> f32 {
        f32::from_bits(w << 16)
    }
    #[inline(always)]
    pub fn bf_hi(w: u32) -> f32 {
        f32::from_bits(w & 0xffff_0000)
    }
    #[inline(always)]
    pub fn f2bf(x: f32) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("cvt.rn.bf16.f32 %0, %1;", out("=h") r, in("f") x, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn imin(a: i32, b: i32) -> i32 {
        if a < b { a } else { b }
    }

    // Memory access. Global K/V are read-only for the kernel (`ld.global.nc`); shared memory is
    // addressed through 32-bit shared-window addresses. None are register_only: they stay in their
    // control flow (no speculative out-of-range loads) and on their side of each barrier.

    #[inline(always)]
    pub unsafe fn ldg_v4(p: *const u16) -> (u32, u32, u32, u32) {
        let (a, b, c, d): (u32, u32, u32, u32);
        ptx_asm!("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];", out("=r") a, out("=r") b, out("=r") c, out("=r") d, in("l") p as u64);
        (a, b, c, d)
    }
    #[inline(always)]
    pub unsafe fn ldg_bf(p: *const u16) -> f32 {
        let r: u16;
        ptx_asm!("ld.global.nc.b16 %0, [%1];", out("=h") r, in("l") p as u64);
        f32::from_bits((r as u32) << 16)
    }
    #[inline(always)]
    pub unsafe fn lds(addr: u32) -> f32 {
        let r: f32;
        ptx_asm!("ld.shared.f32 %0, [%1];", out("=f") r, in("r") addr);
        r
    }
    #[inline(always)]
    pub unsafe fn lds_v4(addr: u32) -> (f32, f32, f32, f32) {
        let (a, b, c, d): (f32, f32, f32, f32);
        ptx_asm!("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];", out("=f") a, out("=f") b, out("=f") c, out("=f") d, in("r") addr);
        (a, b, c, d)
    }
    #[inline(always)]
    pub unsafe fn sts(addr: u32, v: f32) {
        ptx_asm!("st.shared.f32 [%0], %1;", in("r") addr, in("f") v, clobber("memory"));
    }
    #[inline(always)]
    pub unsafe fn sts_v4(addr: u32, a: f32, b: f32, c: f32, d: f32) {
        ptx_asm!("st.shared.v4.f32 [%0], {%1, %2, %3, %4};", in("r") addr, in("f") a, in("f") b, in("f") c, in("f") d, clobber("memory"));
    }
    #[inline(always)]
    pub fn w(off: u32) -> u32 {
        off * 4
    }

    /// The split kernel for G = n_rep * n_rows queries per KV head (G a multiple of 8, <= 24; queries
    /// past n_rep * n_rows are zero and never written).
    ///
    /// q: bf16, query head `hd` of row r at `q + hd * q_hs + r * q_rs` (256 contiguous).
    /// k, v: bf16, key `t` of KV head h at `k + h * k_hs + t * 256`; 16-byte aligned rows.
    /// part: f32 [n_rows * n_heads][nchunks][256]; meta: f32 [n_rows * n_heads][nchunks][2] = (max, sum),
    /// query index `r * n_heads + hd`.
    #[device]
    #[inline(always)]
    pub unsafe fn split<const G: usize>(
        q: *const u16, k: *const u16, v: *const u16, part: *mut f32, meta: *mut f32, scale: f32,
        n_rep: i32, n_heads: i32, kv_total: i32, n_rows: i32, chunk: i32, nchunks: i32,
        q_hs: i32, q_rs: i32, k_hs: u64, v_hs: u64,
    ) {
        static mut SMEM: SharedArray<f32, 9504, 16> = SharedArray::UNINIT;
        let sm = SharedArray::as_raw_mut_ptr(&raw mut SMEM);
        let sb = cuda_device::shared::cvta_generic_to_shared_u32(sm as *const u8);
        let tid = thread::threadIdx_x() as i32;
        let lane = tid & 31;
        let wid = tid >> 5;
        let half = tid & 1;
        let c = thread::blockIdx_x() as i32;
        let kvh = thread::blockIdx_y() as i32;
        let gact = n_rep * n_rows;
        let c0 = c * chunk;
        let c1 = imin(c0 + chunk, kv_total);
        // row 0 (the shortest) sees keys < len0; later rows one more each
        let len0 = kv_total - n_rows + 1;

        // Q (scaled, f32) into shared memory; thread tid = dimension d.
        let mut jj = 0;
        #[unroll]
        while jj < G {
            let ji = jj as i32;
            let mut val = 0f32;
            if ji < gact {
                let r = ji / n_rep;
                let hd = kvh * n_rep + ji % n_rep;
                let off = (hd as i64 * q_hs as i64 + r as i64 * q_rs as i64 + tid as i64) as usize;
                val = mul(f32::from_bits((*q.add(off) as u32) << 16), scale);
            }
            sts(sb + w(OFF_Q + (jj as u32) * 256 + tid as u32), val);
            jj += 1;
        }
        if tid < G as i32 {
            sts(sb + w(OFF_M + tid as u32), NEG_HALF_MAX);
            sts(sb + w(OFF_SUM + tid as u32), 0.0);
            let len = if tid < gact { kv_total - n_rows + tid / n_rep + 1 } else { 0 };
            sts(sb + w(OFF_LEN + tid as u32), f32::from_bits(len as u32));
        }
        let mut acc = [0f32; G];
        thread::sync_threads();

        let mut t0 = c0;
        while t0 < c1 {
            // ---- phase A: scores. Key t0 + tid/2; thread `half` takes 16-byte chunks 2i + half.
            let key = t0 + (tid >> 1);
            let mut s = [0f32; G];
            if key < c1 {
                let kp = k.add((kvh as u64 * k_hs + key as u64 * 256) as usize);
                let mut i = 0;
                #[unroll]
                while i < 16 {
                    let cc = (2 * i + half) as u32;
                    let (a0, a1, a2, a3) = ldg_v4(kp.add((cc * 8) as usize));
                    let kf = [bf_lo(a0), bf_hi(a0), bf_lo(a1), bf_hi(a1), bf_lo(a2), bf_hi(a2), bf_lo(a3), bf_hi(a3)];
                    let qa = sb + w(OFF_Q + cc * 8);
                    let mut j = 0;
                    #[unroll]
                    while j < G {
                        let (q0, q1, q2, q3) = lds_v4(qa + (j as u32) * 1024);
                        let (q4, q5, q6, q7) = lds_v4(qa + (j as u32) * 1024 + 16);
                        let mut t = s[j];
                        t = fma(kf[0], q0, t);
                        t = fma(kf[1], q1, t);
                        t = fma(kf[2], q2, t);
                        t = fma(kf[3], q3, t);
                        t = fma(kf[4], q4, t);
                        t = fma(kf[5], q5, t);
                        t = fma(kf[6], q6, t);
                        t = fma(kf[7], q7, t);
                        s[j] = t;
                        j += 1;
                    }
                    i += 1;
                }
            }
            // pair sum (commutative: both threads hold the same value), mask, tile max per query
            let mut j = 0;
            #[unroll]
            while j < G {
                let o = warp::shuffle_xor_f32_sync(0xffff_ffff, s[j], 1);
                let tot = add(s[j], o);
                let len = lds(sb + w(OFF_LEN + j as u32)).to_bits() as i32;
                s[j] = if key < len { tot } else { NEG_INF };
                let mut m = s[j];
                m = fmaxf(m, warp::shuffle_xor_f32_sync(0xffff_ffff, m, 16));
                m = fmaxf(m, warp::shuffle_xor_f32_sync(0xffff_ffff, m, 8));
                m = fmaxf(m, warp::shuffle_xor_f32_sync(0xffff_ffff, m, 4));
                m = fmaxf(m, warp::shuffle_xor_f32_sync(0xffff_ffff, m, 2));
                if lane == 0 {
                    sts(sb + w(OFF_RED + (wid as u32) * (GMAX as u32) + j as u32), m);
                }
                j += 1;
            }
            thread::sync_threads();
            if tid < G as i32 {
                let jt = tid as u32;
                let mut tm = lds(sb + w(OFF_RED + jt));
                let mut ww = 1u32;
                while ww < 8 {
                    tm = fmaxf(tm, lds(sb + w(OFF_RED + ww * (GMAX as u32) + jt)));
                    ww += 1;
                }
                let mo = lds(sb + w(OFF_M + jt));
                let mn = fmaxf(mo, add(tm, KQ_MAX_OFFSET));
                let sc = if mn == mo { 1.0 } else { expf(sub(mo, mn)) };
                sts(sb + w(OFF_SC + jt), sc);
                sts(sb + w(OFF_M + jt), mn);
            }
            thread::sync_threads();
            // probabilities; half 0 stores P[key][j]; tile sum per query (half 1 adds zeros)
            let mut j = 0;
            #[unroll]
            while j < G {
                let mn = lds(sb + w(OFF_M + j as u32));
                s[j] = expf(sub(s[j], mn));
                j += 1;
            }
            if half == 0 {
                let pa = sb + w(OFF_P + ((tid >> 1) as u32) * (G as u32));
                let mut j = 0;
                #[unroll]
                while j < G {
                    sts_v4(pa + (j as u32) * 4, s[j], s[j + 1], s[j + 2], s[j + 3]);
                    j += 4;
                }
            }
            let mut j = 0;
            #[unroll]
            while j < G {
                let mut p = if half == 0 { s[j] } else { 0.0 };
                p = add(p, warp::shuffle_xor_f32_sync(0xffff_ffff, p, 16));
                p = add(p, warp::shuffle_xor_f32_sync(0xffff_ffff, p, 8));
                p = add(p, warp::shuffle_xor_f32_sync(0xffff_ffff, p, 4));
                p = add(p, warp::shuffle_xor_f32_sync(0xffff_ffff, p, 2));
                p = add(p, warp::shuffle_xor_f32_sync(0xffff_ffff, p, 1));
                if lane == 0 {
                    sts(sb + w(OFF_RED + (wid as u32) * (GMAX as u32) + j as u32), p);
                }
                j += 1;
            }
            thread::sync_threads();
            if tid < G as i32 {
                let jt = tid as u32;
                let mut ts = lds(sb + w(OFF_RED + jt));
                let mut ww = 1u32;
                while ww < 8 {
                    ts = add(ts, lds(sb + w(OFF_RED + ww * (GMAX as u32) + jt)));
                    ww += 1;
                }
                let so = lds(sb + w(OFF_SUM + jt));
                sts(sb + w(OFF_SUM + jt), fma(so, lds(sb + w(OFF_SC + jt)), ts));
            }

            // ---- phase B: V. Thread tid owns dimension tid of every query.
            let mut j = 0;
            #[unroll]
            while j < G {
                acc[j] = mul(acc[j], lds(sb + w(OFF_SC + j as u32)));
                j += 1;
            }
            let nk = imin(TILE, c1 - t0);
            let mut nfull = len0 - t0;
            if nfull < 0 {
                nfull = 0;
            }
            if nfull > nk {
                nfull = nk;
            }
            let vp = v.add((kvh as u64 * v_hs + t0 as u64 * 256 + tid as u64) as usize);
            let mut kk = 0;
            while kk + 4 <= nfull {
                let v0 = ldg_bf(vp.add((kk * 256) as usize));
                let v1 = ldg_bf(vp.add(((kk + 1) * 256) as usize));
                let v2 = ldg_bf(vp.add(((kk + 2) * 256) as usize));
                let v3 = ldg_bf(vp.add(((kk + 3) * 256) as usize));
                let vs = [v0, v1, v2, v3];
                let mut u = 0;
                #[unroll]
                while u < 4 {
                    let pa = sb + w(OFF_P + ((kk + u as i32) as u32) * (G as u32));
                    let mut j = 0;
                    #[unroll]
                    while j < G {
                        let (p0, p1, p2, p3) = lds_v4(pa + (j as u32) * 4);
                        acc[j] = fma(p0, vs[u], acc[j]);
                        acc[j + 1] = fma(p1, vs[u], acc[j + 1]);
                        acc[j + 2] = fma(p2, vs[u], acc[j + 2]);
                        acc[j + 3] = fma(p3, vs[u], acc[j + 3]);
                        j += 4;
                    }
                    u += 1;
                }
                kk += 4;
            }
            while kk < nfull {
                let vv = ldg_bf(vp.add((kk * 256) as usize));
                let pa = sb + w(OFF_P + (kk as u32) * (G as u32));
                let mut j = 0;
                #[unroll]
                while j < G {
                    let (p0, p1, p2, p3) = lds_v4(pa + (j as u32) * 4);
                    acc[j] = fma(p0, vv, acc[j]);
                    acc[j + 1] = fma(p1, vv, acc[j + 1]);
                    acc[j + 2] = fma(p2, vv, acc[j + 2]);
                    acc[j + 3] = fma(p3, vv, acc[j + 3]);
                    j += 4;
                }
                kk += 1;
            }
            // the last n_rows - 1 keys of the sequence: only some rows see them
            while kk < nk {
                let key = t0 + kk;
                let vv = ldg_bf(vp.add((kk * 256) as usize));
                let pa = sb + w(OFF_P + (kk as u32) * (G as u32));
                let mut j = 0;
                #[unroll]
                while j < G {
                    let p = lds(pa + (j as u32) * 4);
                    let len = lds(sb + w(OFF_LEN + j as u32)).to_bits() as i32;
                    if key < len {
                        acc[j] = fma(p, vv, acc[j]);
                    }
                    j += 1;
                }
                kk += 1;
            }
            thread::sync_threads();
            t0 += TILE;
        }

        // partial VKQ and (max, sum) of this chunk
        let mut j = 0;
        #[unroll]
        while j < G {
            let ji = j as i32;
            if ji < gact {
                let r = ji / n_rep;
                let hd = kvh * n_rep + ji % n_rep;
                let qi = (r * n_heads + hd) as u64;
                *part.add(((qi * nchunks as u64 + c as u64) * 256 + tid as u64) as usize) = acc[j];
            }
            j += 1;
        }
        if tid < gact {
            let r = tid / n_rep;
            let hd = kvh * n_rep + tid % n_rep;
            let qi = (r * n_heads + hd) as u64;
            let mo = meta.add(((qi * nchunks as u64 + c as u64) * 2) as usize);
            *mo = lds(sb + w(OFF_M + tid as u32));
            *mo.add(1) = lds(sb + w(OFF_SUM + tid as u32));
        }
    }

    #[kernel]
    pub unsafe fn flash_decode_split_g8(
        q: *const u16, k: *const u16, v: *const u16, part: *mut f32, meta: *mut f32, scale: f32,
        n_rep: i32, n_heads: i32, kv_total: i32, n_rows: i32, chunk: i32, nchunks: i32,
        q_hs: i32, q_rs: i32, k_hs: u64, v_hs: u64,
    ) {
        split::<8>(q, k, v, part, meta, scale, n_rep, n_heads, kv_total, n_rows, chunk, nchunks, q_hs, q_rs, k_hs, v_hs)
    }
    #[kernel]
    pub unsafe fn flash_decode_split_g16(
        q: *const u16, k: *const u16, v: *const u16, part: *mut f32, meta: *mut f32, scale: f32,
        n_rep: i32, n_heads: i32, kv_total: i32, n_rows: i32, chunk: i32, nchunks: i32,
        q_hs: i32, q_rs: i32, k_hs: u64, v_hs: u64,
    ) {
        split::<16>(q, k, v, part, meta, scale, n_rep, n_heads, kv_total, n_rows, chunk, nchunks, q_hs, q_rs, k_hs, v_hs)
    }
    #[kernel]
    pub unsafe fn flash_decode_split_g24(
        q: *const u16, k: *const u16, v: *const u16, part: *mut f32, meta: *mut f32, scale: f32,
        n_rep: i32, n_heads: i32, kv_total: i32, n_rows: i32, chunk: i32, nchunks: i32,
        q_hs: i32, q_rs: i32, k_hs: u64, v_hs: u64,
    ) {
        split::<24>(q, k, v, part, meta, scale, n_rep, n_heads, kv_total, n_rows, chunk, nchunks, q_hs, q_rs, k_hs, v_hs)
    }

    /// Merge row r's chunks `0..ceil(len_r / chunk)` in chunk order (flash_attn_combine_results) and
    /// write bf16 `out[r * out_rs + head * out_hs + d]`. grid (n_heads, n_rows), block 256.
    #[kernel]
    pub unsafe fn flash_decode_combine(
        part: *const f32, meta: *const f32, out: *mut u16, n_heads: i32, n_rows: i32, kv_total: i32,
        chunk: i32, nchunks: i32, out_rs: i32, out_hs: i32,
    ) {
        let d = thread::threadIdx_x() as i32;
        let hd = thread::blockIdx_x() as i32;
        let r = thread::blockIdx_y() as i32;
        let qi = (r * n_heads + hd) as u64;
        let len = kv_total - n_rows + r + 1;
        let nc = (len + chunk - 1) / chunk;
        let mp = meta.add((qi * nchunks as u64 * 2) as usize);
        let pp = part.add((qi * nchunks as u64 * 256 + d as u64) as usize);
        let mut mx = *mp;
        let mut c = 1;
        while c < nc {
            mx = fmaxf(mx, *mp.add((2 * c) as usize));
            c += 1;
        }
        let mut num = 0f32;
        let mut den = 0f32;
        let mut c = 0;
        while c < nc {
            let sc = expf(sub(*mp.add((2 * c) as usize), mx));
            num = fma(sc, *pp.add((c * 256) as usize), num);
            den = fma(sc, *mp.add((2 * c + 1) as usize), den);
            c += 1;
        }
        *out.add((r as i64 * out_rs as i64 + hd as i64 * out_hs as i64 + d as i64) as usize) = f2bf(div_rn(num, den));
    }
}

fn main() {
    std::process::exit(if gate::run() { 0 } else { 1 });
}
