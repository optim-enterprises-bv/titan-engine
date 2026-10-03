#![allow(unsafe_op_in_unsafe_fn)]
//! Tensor-core flash attention for titan-mistral's prompt chunks (Qwen3.6-35B-A3B: head_dim 256, bf16 Q/K/V,
//! 16 query heads on 2 KV heads, causal with the chunk's keys appended after `past` cached ones).
//!
//! `flash_prefill`: grid (ceil(s / 8), kv_heads, nsplit), block 128 (4 warps). A block takes 8 query positions of
//! one KV head for its 8 query heads (64 rows, GQA-packed: K and V are read once for all of them); warp w holds
//! positions 2w (rows 0..7 = heads) and 2w + 1 (rows 8..15) as one m16 tile. The block's keys
//! `[0, min(kv_len, past + p0 + 8))` are cut into `nsplit` ranges of whole 32-key tiles (split-KV, so that short
//! chunks still fill the GPU). Per tile: K and V into shared memory (16-byte chunks XOR-swizzled by row, so
//! fragment loads and ldmatrix are conflict-free), S = Q K^T by `mma.m16n8k16` bf16 -> f32 (Q fragments straight
//! from global memory, L1-resident), an online softmax in base 2 (scores times scale * log2 e, `ex2.approx`), P
//! rounded to bf16 and O += P V by mma with V fragments from `ldmatrix.trans`. Query row i sees keys
//! `< past + i + 1`; keys past the range are zero-filled and masked, so nothing beyond it is read.
//! `nsplit == 1` writes bf16 O / l; otherwise un-normalised f32 O plus (m, l) per row go to `part`/`ml` and
//! `flash_prefill_combine` (grid (s, heads), block 256) merges the splits in order.
//!
//! Numerics are llama.cpp fattn-mma's (acecd56, MIT): f32 scores and accumulators, bf16-rounded probabilities,
//! f32 row sums of the unrounded ones. mistral.rs embeds flash_prefill.ptx (export_ptx.sh) and launches it from
//! Rust (mistralrs-core/src/attention/flash_prefill.rs).
#![allow(non_snake_case, clippy::missing_safety_doc, dead_code)]

use cuda_device::{SharedArray, float, kernel, launch_bounds, ptx_asm, thread, warp};
use cuda_host::cuda_module;

mod gate;
mod gate512;

#[cuda_module]
pub mod kernels {
    use super::*;

    pub const D: i32 = 256;
    /// Keys per tile.
    pub const BC: i32 = 32;
    /// Query positions per block (x 8 heads = 64 rows, 16 per warp).
    pub const POS: i32 = 8;
    pub const N_REP: i32 = 8;
    /// Bytes per K / V row in shared memory (no padding: the 16-byte chunks are swizzled).
    pub const ROW_BYTES: u32 = 512;
    /// V tile offset in shared memory, bytes.
    pub const V_OFF: u32 = 32 * 512;
    pub const LOG2E: f32 = f32::from_bits(0x3fb8aa3b);
    pub const NEG_INF: f32 = f32::from_bits(0xff800000);

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
    #[inline(always)]
    pub fn fmaxf(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("max.ftz.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn div_rn(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("div.rn.ftz.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn ex2(x: f32) -> f32 {
        float::ex2_approx_ftz_f32(x)
    }
    /// Two f32 to packed bf16x2, `lo` in the low half.
    #[inline(always)]
    pub fn pack_bf2(lo: f32, hi: f32) -> u32 {
        let r: u32;
        unsafe { ptx_asm!("cvt.rn.bf16x2.f32 %0, %1, %2;", out("=r") r, in("f") hi, in("f") lo, options(register_only)); }
        r
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

    #[inline(always)]
    pub unsafe fn ldg_v4(p: *const u16) -> (u32, u32, u32, u32) {
        let (a, b, c, d): (u32, u32, u32, u32);
        ptx_asm!("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];", out("=r") a, out("=r") b, out("=r") c, out("=r") d, in("l") p as u64);
        (a, b, c, d)
    }
    #[inline(always)]
    pub unsafe fn ldg_u32(p: *const u16) -> u32 {
        let r: u32;
        ptx_asm!("ld.global.nc.u32 %0, [%1];", out("=r") r, in("l") p as u64);
        r
    }
    #[inline(always)]
    pub unsafe fn lds_u32(addr: u32) -> u32 {
        let r: u32;
        ptx_asm!("ld.shared.u32 %0, [%1];", out("=r") r, in("r") addr);
        r
    }
    #[inline(always)]
    pub unsafe fn sts_v4(addr: u32, a: u32, b: u32, c: u32, d: u32) {
        ptx_asm!("st.shared.v4.u32 [%0], {%1, %2, %3, %4};", in("r") addr, in("r") a, in("r") b, in("r") c, in("r") d, clobber("memory"));
    }
    #[inline(always)]
    pub unsafe fn ldsm_x4_trans(addr: u32) -> (u32, u32, u32, u32) {
        let (a, b, c, d): (u32, u32, u32, u32);
        ptx_asm!("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];", out("=r") a, out("=r") b, out("=r") c, out("=r") d, in("r") addr);
        (a, b, c, d)
    }
    #[inline(always)]
    pub unsafe fn stg_u32(p: *mut u16, v: u32) {
        ptx_asm!("st.global.u32 [%0], %1;", in("l") p as u64, in("r") v, clobber("memory"));
    }
    #[inline(always)]
    pub unsafe fn stg_v2f(p: *mut f32, a: f32, b: f32) {
        ptx_asm!("st.global.v2.f32 [%0], {%1, %2};", in("l") p as u64, in("f") a, in("f") b, clobber("memory"));
    }

    /// D += A B, m16n8k16, bf16 inputs, f32 accumulators.
    #[inline(always)]
    pub fn mma(c: [f32; 4], a0: u32, a1: u32, a2: u32, a3: u32, b0: u32, b1: u32) -> [f32; 4] {
        let (d0, d1, d2, d3): (f32, f32, f32, f32);
        unsafe {
            ptx_asm!(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};",
                out("=f") d0, out("=f") d1, out("=f") d2, out("=f") d3,
                in("r") a0, in("r") a1, in("r") a2, in("r") a3, in("r") b0, in("r") b1,
                in("f") c[0], in("f") c[1], in("f") c[2], in("f") c[3],
                options(register_only),
            );
        }
        [d0, d1, d2, d3]
    }

    /// D += A B, m16n8k16, f16 inputs, f32 accumulators.
    #[inline(always)]
    pub fn mma_f16(c: [f32; 4], a0: u32, a1: u32, a2: u32, a3: u32, b0: u32, b1: u32) -> [f32; 4] {
        let (d0, d1, d2, d3): (f32, f32, f32, f32);
        unsafe {
            ptx_asm!(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};",
                out("=f") d0, out("=f") d1, out("=f") d2, out("=f") d3,
                in("r") a0, in("r") a1, in("r") a2, in("r") a3, in("r") b0, in("r") b1,
                in("f") c[0], in("f") c[1], in("f") c[2], in("f") c[3],
                options(register_only),
            );
        }
        [d0, d1, d2, d3]
    }
    /// Two f32 to packed f16x2, `lo` in the low half.
    #[inline(always)]
    pub fn pack_h2(lo: f32, hi: f32) -> u32 {
        let r: u32;
        unsafe { ptx_asm!("cvt.rn.f16x2.f32 %0, %1, %2;", out("=r") r, in("f") hi, in("f") lo, options(register_only)); }
        r
    }
    /// `mma` on bf16 (F16 = false) or f16 (F16 = true) Q / K / V / P.
    #[inline(always)]
    pub fn mma_t<const F16: bool>(c: [f32; 4], a0: u32, a1: u32, a2: u32, a3: u32, b0: u32, b1: u32) -> [f32; 4] {
        if F16 { mma_f16(c, a0, a1, a2, a3, b0, b1) } else { mma(c, a0, a1, a2, a3, b0, b1) }
    }
    /// P fragment packing in the input type.
    #[inline(always)]
    pub fn pack_p<const F16: bool>(lo: f32, hi: f32) -> u32 {
        if F16 { pack_h2(lo, hi) } else { pack_bf2(lo, hi) }
    }

    /// Byte offset of (row r, 16-byte chunk ch) in a swizzled 512-byte-row tile.
    #[inline(always)]
    pub fn swz(r: u32, ch: u32) -> u32 {
        r * ROW_BYTES + ((ch ^ (r & 7)) << 4)
    }

    /// q: bf16, head hd position p at `q + hd * q_hs + p * 256`. k, v: bf16, key t of KV head h at
    /// `k + h * k_hs + t * 256` (16-byte aligned rows). out: bf16, `out + hd * o_hs + p * 256` (nsplit == 1).
    /// part: f32 [nsplit][n_heads][s][256], ml: f32 [nsplit][n_heads][s][2] = (max, sum), base-2 scores.
    #[kernel]
    #[launch_bounds(128, 2)]
    pub unsafe fn flash_prefill_v1(
        q: *const u16, k: *const u16, v: *const u16, out: *mut u16, part: *mut f32, ml: *mut f32,
        scale_log2: f32, s: i32, past: i32, kv_len: i32, n_heads: i32, nsplit: i32,
        q_hs: i32, o_hs: i32, k_hs: u64, v_hs: u64,
    ) {
        static mut SMEM: SharedArray<u16, 16384, 16> = SharedArray::UNINIT;
        let sm = SharedArray::as_raw_mut_ptr(&raw mut SMEM);
        let sk = cuda_device::shared::cvta_generic_to_shared_u32(sm as *const u8);
        let sv = sk + V_OFF;
        let tid = thread::threadIdx_x() as i32;
        let lane = tid & 31;
        let wid = tid >> 5;
        let g = lane >> 2;
        let t = lane & 3;
        let pb = thread::blockIdx_x() as i32;
        let kvh = thread::blockIdx_y() as i32;
        let split = thread::blockIdx_z() as i32;
        let p0 = pb * POS;
        let pos_a = p0 + 2 * wid;
        let pos_b = pos_a + 1;
        let hd = kvh * N_REP + g;
        // keys row i may see: < past + i + 1
        let lim_a = past + pos_a + 1;
        let lim_b = lim_a + 1;
        let kend = imin(kv_len, past + p0 + POS);
        let ntiles = (kend + BC - 1) / BC;
        let tps = (ntiles + nsplit - 1) / nsplit;
        let tb = split * tps;
        let te = imin(ntiles, tb + tps);
        let k1 = imin(kend, te * BC);

        let zero_q = 0u32;
        let qa = q.add((hd as i64 * q_hs as i64 + pos_a as i64 * 256) as usize);
        let qb = q.add((hd as i64 * q_hs as i64 + pos_b as i64 * 256) as usize);
        let va = pos_a < s;
        let vb = pos_b < s;
        let kbase = k.add((kvh as u64 * k_hs) as usize);
        let vbase = v.add((kvh as u64 * v_hs) as usize);

        let mut o = [[0f32; 4]; 32];
        let mut m_a = NEG_INF;
        let mut m_b = NEG_INF;
        let mut l_a = 0f32;
        let mut l_b = 0f32;

        let mut tile = tb;
        while tile < te {
            let t0 = tile * BC;
            thread::sync_threads();
            // K and V tile: 32 rows x 32 chunks each; thread takes chunks tid + 128 j
            let mut j = 0;
            #[unroll]
            while j < 8 {
                let i = tid + 128 * j;
                let r = i >> 5;
                let ch = i & 31;
                let key = t0 + r;
                let (mut a0, mut a1, mut a2, mut a3) = (0u32, 0u32, 0u32, 0u32);
                let (mut b0, mut b1, mut b2, mut b3) = (0u32, 0u32, 0u32, 0u32);
                if key < k1 {
                    let off = (key as u64 * 256 + ch as u64 * 8) as usize;
                    (a0, a1, a2, a3) = ldg_v4(kbase.add(off));
                    (b0, b1, b2, b3) = ldg_v4(vbase.add(off));
                }
                let so = swz(r as u32, ch as u32);
                sts_v4(sk + so, a0, a1, a2, a3);
                sts_v4(sv + so, b0, b1, b2, b3);
                j += 1;
            }
            thread::sync_threads();

            // S = Q K^T for the warp's 16 rows x 32 keys (4 n-tiles of 8 keys)
            let mut sc = [[0f32; 4]; 4];
            let mut kk = 0;
            #[unroll]
            while kk < 16 {
                let col = (16 * kk + 2 * t) as usize;
                let a0 = if va { ldg_u32(qa.add(col)) } else { zero_q };
                let a1 = if vb { ldg_u32(qb.add(col)) } else { zero_q };
                let a2 = if va { ldg_u32(qa.add(col + 8)) } else { zero_q };
                let a3 = if vb { ldg_u32(qb.add(col + 8)) } else { zero_q };
                let mut nj = 0;
                #[unroll]
                while nj < 4 {
                    let r = (8 * nj as i32 + g) as u32;
                    let b0 = lds_u32(sk + swz(r, (2 * kk) as u32) + 4 * t as u32);
                    let b1 = lds_u32(sk + swz(r, (2 * kk + 1) as u32) + 4 * t as u32);
                    sc[nj] = mma(sc[nj], a0, a1, a2, a3, b0, b1);
                    nj += 1;
                }
                kk += 1;
            }

            // scale, causal mask, tile row maxima
            let ea = imin(lim_a, k1);
            let eb = imin(lim_b, k1);
            let mut tm_a = NEG_INF;
            let mut tm_b = NEG_INF;
            let mut nj = 0;
            #[unroll]
            while nj < 4 {
                let key = t0 + 8 * nj as i32 + 2 * t;
                sc[nj][0] = if key < ea { mul(sc[nj][0], scale_log2) } else { NEG_INF };
                sc[nj][1] = if key + 1 < ea { mul(sc[nj][1], scale_log2) } else { NEG_INF };
                sc[nj][2] = if key < eb { mul(sc[nj][2], scale_log2) } else { NEG_INF };
                sc[nj][3] = if key + 1 < eb { mul(sc[nj][3], scale_log2) } else { NEG_INF };
                tm_a = fmaxf(tm_a, fmaxf(sc[nj][0], sc[nj][1]));
                tm_b = fmaxf(tm_b, fmaxf(sc[nj][2], sc[nj][3]));
                nj += 1;
            }
            tm_a = fmaxf(tm_a, warp::shuffle_xor_f32_sync(0xffff_ffff, tm_a, 1));
            tm_a = fmaxf(tm_a, warp::shuffle_xor_f32_sync(0xffff_ffff, tm_a, 2));
            tm_b = fmaxf(tm_b, warp::shuffle_xor_f32_sync(0xffff_ffff, tm_b, 1));
            tm_b = fmaxf(tm_b, warp::shuffle_xor_f32_sync(0xffff_ffff, tm_b, 2));
            let mn_a = fmaxf(m_a, tm_a);
            let mn_b = fmaxf(m_b, tm_b);
            let mu_a = if mn_a == NEG_INF { 0.0 } else { mn_a };
            let mu_b = if mn_b == NEG_INF { 0.0 } else { mn_b };
            let cr_a = ex2(sub(m_a, mu_a));
            let cr_b = ex2(sub(m_b, mu_b));
            m_a = mn_a;
            m_b = mn_b;

            // probabilities (f32 row sums), bf16 A fragments of P: k-step ks covers n-tiles 2ks, 2ks + 1
            let mut ps_a = 0f32;
            let mut ps_b = 0f32;
            let mut nj = 0;
            #[unroll]
            while nj < 4 {
                sc[nj][0] = ex2(sub(sc[nj][0], mu_a));
                sc[nj][1] = ex2(sub(sc[nj][1], mu_a));
                sc[nj][2] = ex2(sub(sc[nj][2], mu_b));
                sc[nj][3] = ex2(sub(sc[nj][3], mu_b));
                ps_a = add(ps_a, add(sc[nj][0], sc[nj][1]));
                ps_b = add(ps_b, add(sc[nj][2], sc[nj][3]));
                nj += 1;
            }
            l_a = fma(l_a, cr_a, ps_a);
            l_b = fma(l_b, cr_b, ps_b);
            let mut dn = 0;
            #[unroll]
            while dn < 32 {
                o[dn][0] = mul(o[dn][0], cr_a);
                o[dn][1] = mul(o[dn][1], cr_a);
                o[dn][2] = mul(o[dn][2], cr_b);
                o[dn][3] = mul(o[dn][3], cr_b);
                dn += 1;
            }

            // O += P V: per k-step of 16 keys, per pair of 8-dim n-tiles one ldmatrix.x4.trans
            let mi = (lane >> 3) as u32;
            let ri = (lane & 7) as u32;
            let mut ks = 0;
            #[unroll]
            while ks < 2 {
                let pa0 = pack_bf2(sc[2 * ks][0], sc[2 * ks][1]);
                let pa1 = pack_bf2(sc[2 * ks][2], sc[2 * ks][3]);
                let pa2 = pack_bf2(sc[2 * ks + 1][0], sc[2 * ks + 1][1]);
                let pa3 = pack_bf2(sc[2 * ks + 1][2], sc[2 * ks + 1][3]);
                let vr = 16 * ks as u32 + (mi & 1) * 8 + ri;
                let mut np = 0;
                #[unroll]
                while np < 16 {
                    let ch = 2 * np as u32 + (mi >> 1);
                    let (b0, b1, b2, b3) = ldsm_x4_trans(sv + swz(vr, ch));
                    o[2 * np] = mma(o[2 * np], pa0, pa1, pa2, pa3, b0, b1);
                    o[2 * np + 1] = mma(o[2 * np + 1], pa0, pa1, pa2, pa3, b2, b3);
                    np += 1;
                }
                ks += 1;
            }
            tile += 1;
        }

        l_a = add(l_a, warp::shuffle_xor_f32_sync(0xffff_ffff, l_a, 1));
        l_a = add(l_a, warp::shuffle_xor_f32_sync(0xffff_ffff, l_a, 2));
        l_b = add(l_b, warp::shuffle_xor_f32_sync(0xffff_ffff, l_b, 1));
        l_b = add(l_b, warp::shuffle_xor_f32_sync(0xffff_ffff, l_b, 2));

        if nsplit == 1 {
            let oa = out.add((hd as i64 * o_hs as i64 + pos_a as i64 * 256) as usize);
            let ob = out.add((hd as i64 * o_hs as i64 + pos_b as i64 * 256) as usize);
            let mut dn = 0;
            #[unroll]
            while dn < 32 {
                let col = (8 * dn as i32 + 2 * t) as usize;
                if va {
                    stg_u32(oa.add(col), pack_bf2(div_rn(o[dn][0], l_a), div_rn(o[dn][1], l_a)));
                }
                if vb {
                    stg_u32(ob.add(col), pack_bf2(div_rn(o[dn][2], l_b), div_rn(o[dn][3], l_b)));
                }
                dn += 1;
            }
        } else {
            let ra = ((split as i64 * n_heads as i64 + hd as i64) * s as i64 + pos_a as i64) as usize;
            let rb = ra + 1;
            let mut dn = 0;
            #[unroll]
            while dn < 32 {
                let col = (8 * dn as i32 + 2 * t) as usize;
                if va {
                    stg_v2f(part.add(ra * 256 + col), o[dn][0], o[dn][1]);
                }
                if vb {
                    stg_v2f(part.add(rb * 256 + col), o[dn][2], o[dn][3]);
                }
                dn += 1;
            }
            if t == 0 {
                if va {
                    stg_v2f(ml.add(ra * 2), m_a, l_a);
                }
                if vb {
                    stg_v2f(ml.add(rb * 2), m_b, l_b);
                }
            }
        }
    }

    macro_rules! unroll {
        () => {
            cuda_device::thread::__unroll_config::<0>();
        };
    }

    /// 16-byte cp.async (L2 only); `bytes` 0 zero-fills without reading `src`.
    #[inline(always)]
    pub unsafe fn cp_async16(dst: u32, src: *const u16, bytes: u32) {
        ptx_asm!("cp.async.cg.shared.global [%0], [%1], 16, %2;", in("r") dst, in("l") src as u64, in("r") bytes, clobber("memory"));
    }
    #[inline(always)]
    pub unsafe fn cp_commit() {
        ptx_asm!("cp.async.commit_group;", clobber("memory"));
    }
    #[inline(always)]
    pub unsafe fn cp_wait_all() {
        ptx_asm!("cp.async.wait_group 0;", clobber("memory"));
    }
    #[inline(always)]
    pub unsafe fn ldsm_x4(addr: u32) -> (u32, u32, u32, u32) {
        let (a, b, c, d): (u32, u32, u32, u32);
        ptx_asm!("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];", out("=r") a, out("=r") b, out("=r") c, out("=r") d, in("r") addr, clobber("memory"));
        (a, b, c, d)
    }

    /// One 32-key K or V tile (32 rows x 32 16-byte chunks, swizzled) into shared memory at `dst` by cp.async,
    /// keys `>= k1` zero-filled without a read. `NW` warps share the 1024 chunks.
    #[inline(always)]
    pub unsafe fn load_tile<const NW: i32>(dst: u32, base: *const u16, t0: i32, k1: i32, tid: i32) {
        let mut j = 0;
        while j < 32 / NW {
            unroll!();
            let i = tid + 32 * NW * j;
            let r = i >> 5;
            let ch = i & 31;
            let key = t0 + r;
            let ok = key < k1;
            let src = if ok { base.add((key as u64 * 256 + ch as u64 * 8) as usize) } else { base };
            cp_async16(dst + swz(r as u32, ch as u32), src, if ok { 16 } else { 0 });
            j += 1;
        }
    }

    /// `flash_prefill_v1` restructured after llama.cpp fattn-mma's pipeline (acecd56, MIT): Q fragments in
    /// registers (`QREG`) or from L1, K fragments by `ldmatrix.x4`, and K / V tiles by cp.async so that the V tile
    /// loads while S = Q K^T runs and the next K tile loads while O += P V runs (one K and one V buffer, 32 KiB).
    /// `NW` warps per block take `2 NW` positions (grid x = ceil(s / (2 NW)), block 32 NW). Every product and sum
    /// is v1's in v1's order, so with the same split ranges the output is bit-identical to v1.
    #[inline(always)]
    pub unsafe fn body<const NW: i32, const QREG: bool, const F16: bool, const REP: i32, const O32: bool>(
        sk: u32, q: *const u16, k: *const u16, v: *const u16, out: *mut u16, part: *mut f32, ml: *mut f32,
        scale_log2: f32, s: i32, past: i32, kv_len: i32, n_heads: i32, nsplit: i32,
        q_hs: i32, o_hs: i32, k_hs: u64, v_hs: u64, win: i32,
    ) {
        let sv = sk + V_OFF;
        let tid = thread::threadIdx_x() as i32;
        let lane = tid & 31;
        let wid = tid >> 5;
        let g = lane >> 2;
        let t = lane & 3;
        let pb = thread::blockIdx_x() as i32;
        let kvh = thread::blockIdx_y() as i32;
        let split = thread::blockIdx_z() as i32;
        // REP query heads per KV head x 16 / REP positions per warp tile: row r = (head r % REP, position r / REP);
        // a lane's rows g and g + 8 share the head and are 8 / REP positions apart (REP 8: v1's 8 heads x 2 positions)
        let ppw = 16 / REP;
        let p0 = pb * ppw * NW;
        let pos_a = p0 + ppw * wid + g / REP;
        let pos_b = pos_a + 8 / REP;
        let hd = kvh * REP + g % REP;
        let lim_a = past + pos_a + 1;
        let lim_b = past + pos_b + 1;
        // sliding window (win > 0): row i also needs keys >= past + i + 1 - win; tiles below the first row's skipped
        let lo_a = if win > 0 { lim_a - win } else { 0 };
        let lo_b = if win > 0 { lim_b - win } else { 0 };
        let kbeg = if win > 0 && past + p0 + 1 > win { past + p0 + 1 - win } else { 0 };
        let tfirst = kbeg / BC;
        let kend = imin(kv_len, past + p0 + ppw * NW);
        let ntiles = (kend + BC - 1) / BC - tfirst;
        let tps = (ntiles + nsplit - 1) / nsplit;
        let tb = tfirst + split * tps;
        let te = imin(tfirst + ntiles, tb + tps);
        let k1 = imin(kend, te * BC);

        let qa = q.add((hd as i64 * q_hs as i64 + pos_a as i64 * 256) as usize);
        let qb = q.add((hd as i64 * q_hs as i64 + pos_b as i64 * 256) as usize);
        let va = pos_a < s;
        let vb = pos_b < s;
        let kbase = k.add((kvh as u64 * k_hs) as usize);
        let vbase = v.add((kvh as u64 * v_hs) as usize);

        let mut qf = [[0u32; 4]; 16];
        if QREG {
            let mut kk = 0;
            while kk < 16 {
                unroll!();
                let col = (16 * kk + 2 * t) as usize;
                qf[kk as usize][0] = if va { ldg_u32(qa.add(col)) } else { 0 };
                qf[kk as usize][1] = if vb { ldg_u32(qb.add(col)) } else { 0 };
                qf[kk as usize][2] = if va { ldg_u32(qa.add(col + 8)) } else { 0 };
                qf[kk as usize][3] = if vb { ldg_u32(qb.add(col + 8)) } else { 0 };
                kk += 1;
            }
        }

        let mut o = [[0f32; 4]; 32];
        let mut m_a = NEG_INF;
        let mut m_b = NEG_INF;
        let mut l_a = 0f32;
        let mut l_b = 0f32;
        let mi = (lane >> 3) as u32;
        let ri = (lane & 7) as u32;

        if tb < te {
            load_tile::<NW>(sk, kbase, tb * BC, k1, tid);
        }
        cp_commit();
        let mut tile = tb;
        while tile < te {
            let t0 = tile * BC;
            // K(tile) landed; every warp is past the previous O += P V, so the V buffer is free
            cp_wait_all();
            thread::sync_threads();
            load_tile::<NW>(sv, vbase, t0, k1, tid);
            cp_commit();

            // S = Q K^T: per k-step one ldmatrix.x4 gives the B fragments of two 8-key n-tiles
            let mut sc = [[0f32; 4]; 4];
            let mut kk = 0;
            while kk < 16 {
                unroll!();
                let (a0, a1, a2, a3) = if QREG {
                    (qf[kk as usize][0], qf[kk as usize][1], qf[kk as usize][2], qf[kk as usize][3])
                } else {
                    let col = (16 * kk + 2 * t) as usize;
                    (
                        if va { ldg_u32(qa.add(col)) } else { 0 },
                        if vb { ldg_u32(qb.add(col)) } else { 0 },
                        if va { ldg_u32(qa.add(col + 8)) } else { 0 },
                        if vb { ldg_u32(qb.add(col + 8)) } else { 0 },
                    )
                };
                let mut np = 0;
                while np < 2 {
                    unroll!();
                    let r = 16 * np as u32 + (mi >> 1) * 8 + ri;
                    let ch = 2 * kk as u32 + (mi & 1);
                    let (b0, b1, b2, b3) = ldsm_x4(sk + swz(r, ch));
                    sc[2 * np] = mma_t::<F16>(sc[2 * np], a0, a1, a2, a3, b0, b1);
                    sc[2 * np + 1] = mma_t::<F16>(sc[2 * np + 1], a0, a1, a2, a3, b2, b3);
                    np += 1;
                }
                kk += 1;
            }

            // scale, causal mask, tile row maxima (as v1)
            let ea = imin(lim_a, k1);
            let eb = imin(lim_b, k1);
            let mut tm_a = NEG_INF;
            let mut tm_b = NEG_INF;
            let mut nj = 0;
            while nj < 4 {
                unroll!();
                let key = t0 + 8 * nj as i32 + 2 * t;
                sc[nj][0] = if key < ea && key >= lo_a { mul(sc[nj][0], scale_log2) } else { NEG_INF };
                sc[nj][1] = if key + 1 < ea && key + 1 >= lo_a { mul(sc[nj][1], scale_log2) } else { NEG_INF };
                sc[nj][2] = if key < eb && key >= lo_b { mul(sc[nj][2], scale_log2) } else { NEG_INF };
                sc[nj][3] = if key + 1 < eb && key + 1 >= lo_b { mul(sc[nj][3], scale_log2) } else { NEG_INF };
                tm_a = fmaxf(tm_a, fmaxf(sc[nj][0], sc[nj][1]));
                tm_b = fmaxf(tm_b, fmaxf(sc[nj][2], sc[nj][3]));
                nj += 1;
            }
            tm_a = fmaxf(tm_a, warp::shuffle_xor_f32_sync(0xffff_ffff, tm_a, 1));
            tm_a = fmaxf(tm_a, warp::shuffle_xor_f32_sync(0xffff_ffff, tm_a, 2));
            tm_b = fmaxf(tm_b, warp::shuffle_xor_f32_sync(0xffff_ffff, tm_b, 1));
            tm_b = fmaxf(tm_b, warp::shuffle_xor_f32_sync(0xffff_ffff, tm_b, 2));
            let mn_a = fmaxf(m_a, tm_a);
            let mn_b = fmaxf(m_b, tm_b);
            let mu_a = if mn_a == NEG_INF { 0.0 } else { mn_a };
            let mu_b = if mn_b == NEG_INF { 0.0 } else { mn_b };
            let cr_a = ex2(sub(m_a, mu_a));
            let cr_b = ex2(sub(m_b, mu_b));
            m_a = mn_a;
            m_b = mn_b;

            let mut ps_a = 0f32;
            let mut ps_b = 0f32;
            let mut nj = 0;
            while nj < 4 {
                unroll!();
                sc[nj][0] = ex2(sub(sc[nj][0], mu_a));
                sc[nj][1] = ex2(sub(sc[nj][1], mu_a));
                sc[nj][2] = ex2(sub(sc[nj][2], mu_b));
                sc[nj][3] = ex2(sub(sc[nj][3], mu_b));
                ps_a = add(ps_a, add(sc[nj][0], sc[nj][1]));
                ps_b = add(ps_b, add(sc[nj][2], sc[nj][3]));
                nj += 1;
            }
            l_a = fma(l_a, cr_a, ps_a);
            l_b = fma(l_b, cr_b, ps_b);
            let mut dn = 0;
            while dn < 32 {
                unroll!();
                o[dn][0] = mul(o[dn][0], cr_a);
                o[dn][1] = mul(o[dn][1], cr_a);
                o[dn][2] = mul(o[dn][2], cr_b);
                o[dn][3] = mul(o[dn][3], cr_b);
                dn += 1;
            }

            // V(tile) landed; every warp is past S = Q K^T, so the K buffer is free for the next tile
            cp_wait_all();
            thread::sync_threads();
            if tile + 1 < te {
                load_tile::<NW>(sk, kbase, t0 + BC, k1, tid);
            }
            cp_commit();

            let mut ks = 0;
            while ks < 2 {
                unroll!();
                let pa0 = pack_p::<F16>(sc[2 * ks][0], sc[2 * ks][1]);
                let pa1 = pack_p::<F16>(sc[2 * ks][2], sc[2 * ks][3]);
                let pa2 = pack_p::<F16>(sc[2 * ks + 1][0], sc[2 * ks + 1][1]);
                let pa3 = pack_p::<F16>(sc[2 * ks + 1][2], sc[2 * ks + 1][3]);
                let vr = 16 * ks as u32 + (mi & 1) * 8 + ri;
                let mut np = 0;
                while np < 16 {
                    unroll!();
                    let ch = 2 * np as u32 + (mi >> 1);
                    let (b0, b1, b2, b3) = ldsm_x4_trans(sv + swz(vr, ch));
                    o[2 * np] = mma_t::<F16>(o[2 * np], pa0, pa1, pa2, pa3, b0, b1);
                    o[2 * np + 1] = mma_t::<F16>(o[2 * np + 1], pa0, pa1, pa2, pa3, b2, b3);
                    np += 1;
                }
                ks += 1;
            }
            tile += 1;
        }
        cp_wait_all();

        l_a = add(l_a, warp::shuffle_xor_f32_sync(0xffff_ffff, l_a, 1));
        l_a = add(l_a, warp::shuffle_xor_f32_sync(0xffff_ffff, l_a, 2));
        l_b = add(l_b, warp::shuffle_xor_f32_sync(0xffff_ffff, l_b, 1));
        l_b = add(l_b, warp::shuffle_xor_f32_sync(0xffff_ffff, l_b, 2));

        if nsplit == 1 {
            let oa = out.add((hd as i64 * o_hs as i64 + pos_a as i64 * 256) as usize);
            let ob = out.add((hd as i64 * o_hs as i64 + pos_b as i64 * 256) as usize);
            let mut dn = 0;
            while dn < 32 {
                unroll!();
                let col = (8 * dn as i32 + 2 * t) as usize;
                if O32 {
                    if va {
                        stg_v2f((out as *mut f32).add((hd as i64 * o_hs as i64 + pos_a as i64 * 256) as usize + col), div_rn(o[dn][0], l_a), div_rn(o[dn][1], l_a));
                    }
                    if vb {
                        stg_v2f((out as *mut f32).add((hd as i64 * o_hs as i64 + pos_b as i64 * 256) as usize + col), div_rn(o[dn][2], l_b), div_rn(o[dn][3], l_b));
                    }
                } else {
                    if va {
                        stg_u32(oa.add(col), pack_bf2(div_rn(o[dn][0], l_a), div_rn(o[dn][1], l_a)));
                    }
                    if vb {
                        stg_u32(ob.add(col), pack_bf2(div_rn(o[dn][2], l_b), div_rn(o[dn][3], l_b)));
                    }
                }
                dn += 1;
            }
        } else {
            let ra = ((split as i64 * n_heads as i64 + hd as i64) * s as i64 + pos_a as i64) as usize;
            let rb = ((split as i64 * n_heads as i64 + hd as i64) * s as i64 + pos_b as i64) as usize;
            let mut dn = 0;
            while dn < 32 {
                unroll!();
                let col = (8 * dn as i32 + 2 * t) as usize;
                if va {
                    stg_v2f(part.add(ra * 256 + col), o[dn][0], o[dn][1]);
                }
                if vb {
                    stg_v2f(part.add(rb * 256 + col), o[dn][2], o[dn][3]);
                }
                dn += 1;
            }
            if t == 0 {
                if va {
                    stg_v2f(ml.add(ra * 2), m_a, l_a);
                }
                if vb {
                    stg_v2f(ml.add(rb * 2), m_b, l_b);
                }
            }
        }
    }

    /// Q in registers, 4 warps (8 positions) per block: grid (ceil(s / 8), kv_heads, nsplit), block 128.
    #[kernel]
    #[launch_bounds(128, 2)]
    pub unsafe fn flash_prefill(
        q: *const u16, k: *const u16, v: *const u16, out: *mut u16, part: *mut f32, ml: *mut f32,
        scale_log2: f32, s: i32, past: i32, kv_len: i32, n_heads: i32, nsplit: i32,
        q_hs: i32, o_hs: i32, k_hs: u64, v_hs: u64, win: i32,
    ) {
        static mut SMEM: SharedArray<u16, 16384, 16> = SharedArray::UNINIT;
        let sk = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        body::<4, true, false, 8, false>(sk, q, k, v, out, part, ml, scale_log2, s, past, kv_len, n_heads, nsplit, q_hs, o_hs, k_hs, v_hs, win);
    }

    /// Q in registers, 8 warps (16 positions) per block: grid (ceil(s / 16), kv_heads, nsplit), block 256.
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn flash_prefill_w8(
        q: *const u16, k: *const u16, v: *const u16, out: *mut u16, part: *mut f32, ml: *mut f32,
        scale_log2: f32, s: i32, past: i32, kv_len: i32, n_heads: i32, nsplit: i32,
        q_hs: i32, o_hs: i32, k_hs: u64, v_hs: u64, win: i32,
    ) {
        static mut SMEM: SharedArray<u16, 16384, 16> = SharedArray::UNINIT;
        let sk = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        body::<8, true, false, 8, false>(sk, q, k, v, out, part, ml, scale_log2, s, past, kv_len, n_heads, nsplit, q_hs, o_hs, k_hs, v_hs, win);
    }

    /// Q fragments from L1 each tile (fewer registers), 4 warps.
    #[kernel]
    #[launch_bounds(128, 2)]
    pub unsafe fn flash_prefill_qg(
        q: *const u16, k: *const u16, v: *const u16, out: *mut u16, part: *mut f32, ml: *mut f32,
        scale_log2: f32, s: i32, past: i32, kv_len: i32, n_heads: i32, nsplit: i32,
        q_hs: i32, o_hs: i32, k_hs: u64, v_hs: u64, win: i32,
    ) {
        static mut SMEM: SharedArray<u16, 16384, 16> = SharedArray::UNINIT;
        let sk = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        body::<4, false, false, 8, false>(sk, q, k, v, out, part, ml, scale_log2, s, past, kv_len, n_heads, nsplit, q_hs, o_hs, k_hs, v_hs, win);
    }

    /// Q fragments from L1 each tile, 8 warps.
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn flash_prefill_qg_w8(
        q: *const u16, k: *const u16, v: *const u16, out: *mut u16, part: *mut f32, ml: *mut f32,
        scale_log2: f32, s: i32, past: i32, kv_len: i32, n_heads: i32, nsplit: i32,
        q_hs: i32, o_hs: i32, k_hs: u64, v_hs: u64, win: i32,
    ) {
        static mut SMEM: SharedArray<u16, 16384, 16> = SharedArray::UNINIT;
        let sk = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        body::<8, false, false, 8, false>(sk, q, k, v, out, part, ml, scale_log2, s, past, kv_len, n_heads, nsplit, q_hs, o_hs, k_hs, v_hs, win);
    }

    /// Merge the splits of row (pos, head) in split order: `sum_i 2^(m_i - M) O_i / sum_i 2^(m_i - M) l_i`, bf16 to
    /// `out + head * o_hs + pos * 256`. grid (s, n_heads), block 256.
    #[kernel]
    pub unsafe fn flash_prefill_combine(
        part: *const f32, ml: *const f32, out: *mut u16, s: i32, n_heads: i32, nsplit: i32, o_hs: i32,
    ) {
        let d = thread::threadIdx_x() as i64;
        let pos = thread::blockIdx_x() as i64;
        let hd = thread::blockIdx_y() as i64;
        let row = |i: i32| ((i as i64 * n_heads as i64 + hd) * s as i64 + pos) as usize;
        let mut mx = NEG_INF;
        let mut i = 0;
        while i < nsplit {
            mx = fmaxf(mx, *ml.add(row(i) * 2));
            i += 1;
        }
        let mut num = 0f32;
        let mut den = 0f32;
        let mut i = 0;
        while i < nsplit {
            let m = *ml.add(row(i) * 2);
            if m != NEG_INF {
                let w = ex2(sub(m, mx));
                num = fma(w, *part.add(row(i) * 256 + d as usize), num);
                den = fma(w, *ml.add(row(i) * 2 + 1), den);
            }
            i += 1;
        }
        *out.add((hd * o_hs as i64 + pos * 256 + d) as usize) = f2bf(div_rn(num, den));
    }

    // ---- head dim 512 (gemma4 full-attention layers) ----

    /// Bytes per K row in shared memory at head dim 512 (64 swizzled 16-byte chunks).
    pub const ROW_BYTES_512: u32 = 1024;
    /// V tile offset in the 512 kernels' shared memory: after the 32 x 1024-byte K tile.
    pub const V_OFF_512: u32 = 32 * 1024;

    /// Byte offset of (row r, 16-byte chunk ch) in a swizzled 1024-byte-row K tile.
    #[inline(always)]
    pub fn swz512(r: u32, ch: u32) -> u32 {
        r * ROW_BYTES_512 + ((ch ^ (r & 7)) << 4)
    }

    /// One 32-key K tile at head dim 512 (32 rows x 64 chunks, swizzled) by cp.async, keys `>= k1` zero-filled.
    #[inline(always)]
    pub unsafe fn load_k512<const NW: i32>(dst: u32, base: *const u16, t0: i32, k1: i32, tid: i32) {
        let mut j = 0;
        while j < 64 / NW {
            unroll!();
            let i = tid + 32 * NW * j;
            let r = i >> 6;
            let ch = i & 63;
            let key = t0 + r;
            let ok = key < k1;
            let src = if ok { base.add((key as u64 * 512 + ch as u64 * 8) as usize) } else { base };
            cp_async16(dst + swz512(r as u32, ch as u32), src, if ok { 16 } else { 0 });
            j += 1;
        }
    }

    /// One 32-key V tile, the 256 dims `[vo, vo + 256)` of head dim 512 (32 rows x 32 chunks, the 256 kernels'
    /// swizzled 512-byte rows), keys `>= k1` zero-filled.
    #[inline(always)]
    pub unsafe fn load_v512<const NW: i32>(dst: u32, base: *const u16, t0: i32, k1: i32, tid: i32) {
        let mut j = 0;
        while j < 32 / NW {
            unroll!();
            let i = tid + 32 * NW * j;
            let r = i >> 5;
            let ch = i & 31;
            let key = t0 + r;
            let ok = key < k1;
            let src = if ok { base.add((key as u64 * 512 + ch as u64 * 8) as usize) } else { base };
            cp_async16(dst + swz(r as u32, ch as u32), src, if ok { 16 } else { 0 });
            j += 1;
        }
    }

    /// `body` at head dim 512: a block computes one half (`[256 h, 256 h + 256)`, h = blockIdx.z & 1) of the
    /// output dims of its 8 heads x 2 NW positions, S = Q K^T over all 512 dims (each half recomputes S in the same
    /// order, so both halves see the same probabilities, maxima and sums). The 128 f32 O accumulators per thread are
    /// the 256 kernels'; Q fragments come from L1 each tile (512 dims of Q do not fit in registers next to O).
    /// Shared memory: K tile 32 x 1024 B + V half-tile 32 x 512 B = 48 KiB. Grid (ceil(s / (2 NW)), n_heads / 8,
    /// 2 nsplit), block 32 NW; query head group y (heads 8y .. 8y + 7) reads KV head y / gq.
    /// q: head hd position p at `q + hd * q_hs + p * 512`; k, v: key t of KV head h at `k + h * k_hs + t * 512`;
    /// out: `out + hd * o_hs + p * 512` (nsplit == 1); part: f32 [nsplit][n_heads][s][512], ml as the 256 kernels'.
    #[inline(always)]
    pub unsafe fn body512<const NW: i32, const F16: bool, const O32: bool>(
        sk: u32, q: *const u16, k: *const u16, v: *const u16, out: *mut u16, part: *mut f32, ml: *mut f32,
        scale_log2: f32, s: i32, past: i32, kv_len: i32, n_heads: i32, nsplit: i32,
        q_hs: i32, o_hs: i32, k_hs: u64, v_hs: u64, win: i32, gq: i32,
    ) {
        let sv = sk + V_OFF_512;
        let tid = thread::threadIdx_x() as i32;
        let lane = tid & 31;
        let wid = tid >> 5;
        let g = lane >> 2;
        let t = lane & 3;
        let pb = thread::blockIdx_x() as i32;
        let grp = thread::blockIdx_y() as i32;
        let zz = thread::blockIdx_z() as i32;
        let split = zz >> 1;
        let vo = (zz & 1) * 256;
        let kvh = grp / gq;
        let p0 = pb * 2 * NW;
        let pos_a = p0 + 2 * wid;
        let pos_b = pos_a + 1;
        let hd = grp * N_REP + g;
        let lim_a = past + pos_a + 1;
        let lim_b = lim_a + 1;
        let lo_a = if win > 0 { lim_a - win } else { 0 };
        let lo_b = if win > 0 { lim_b - win } else { 0 };
        let kbeg = if win > 0 && past + p0 + 1 > win { past + p0 + 1 - win } else { 0 };
        let tfirst = kbeg / BC;
        let kend = imin(kv_len, past + p0 + 2 * NW);
        let ntiles = (kend + BC - 1) / BC - tfirst;
        let tps = (ntiles + nsplit - 1) / nsplit;
        let tb = tfirst + split * tps;
        let te = imin(tfirst + ntiles, tb + tps);
        let k1 = imin(kend, te * BC);

        let qa = q.add((hd as i64 * q_hs as i64 + pos_a as i64 * 512) as usize);
        let qb = q.add((hd as i64 * q_hs as i64 + pos_b as i64 * 512) as usize);
        let va = pos_a < s;
        let vb = pos_b < s;
        let kbase = k.add((kvh as u64 * k_hs) as usize);
        let vbase = v.add((kvh as u64 * v_hs) as usize + vo as usize);

        let mut o = [[0f32; 4]; 32];
        let mut m_a = NEG_INF;
        let mut m_b = NEG_INF;
        let mut l_a = 0f32;
        let mut l_b = 0f32;
        let mi = (lane >> 3) as u32;
        let ri = (lane & 7) as u32;

        if tb < te {
            load_k512::<NW>(sk, kbase, tb * BC, k1, tid);
        }
        cp_commit();
        let mut tile = tb;
        while tile < te {
            let t0 = tile * BC;
            cp_wait_all();
            thread::sync_threads();
            load_v512::<NW>(sv, vbase, t0, k1, tid);
            cp_commit();

            let mut sc = [[0f32; 4]; 4];
            let mut kk = 0;
            while kk < 32 {
                unroll!();
                let col = (16 * kk + 2 * t) as usize;
                let a0 = if va { ldg_u32(qa.add(col)) } else { 0 };
                let a1 = if vb { ldg_u32(qb.add(col)) } else { 0 };
                let a2 = if va { ldg_u32(qa.add(col + 8)) } else { 0 };
                let a3 = if vb { ldg_u32(qb.add(col + 8)) } else { 0 };
                let mut np = 0;
                while np < 2 {
                    unroll!();
                    let r = 16 * np as u32 + (mi >> 1) * 8 + ri;
                    let ch = 2 * kk as u32 + (mi & 1);
                    let (b0, b1, b2, b3) = ldsm_x4(sk + swz512(r, ch));
                    sc[2 * np] = mma_t::<F16>(sc[2 * np], a0, a1, a2, a3, b0, b1);
                    sc[2 * np + 1] = mma_t::<F16>(sc[2 * np + 1], a0, a1, a2, a3, b2, b3);
                    np += 1;
                }
                kk += 1;
            }

            let ea = imin(lim_a, k1);
            let eb = imin(lim_b, k1);
            let mut tm_a = NEG_INF;
            let mut tm_b = NEG_INF;
            let mut nj = 0;
            while nj < 4 {
                unroll!();
                let key = t0 + 8 * nj as i32 + 2 * t;
                sc[nj][0] = if key < ea && key >= lo_a { mul(sc[nj][0], scale_log2) } else { NEG_INF };
                sc[nj][1] = if key + 1 < ea && key + 1 >= lo_a { mul(sc[nj][1], scale_log2) } else { NEG_INF };
                sc[nj][2] = if key < eb && key >= lo_b { mul(sc[nj][2], scale_log2) } else { NEG_INF };
                sc[nj][3] = if key + 1 < eb && key + 1 >= lo_b { mul(sc[nj][3], scale_log2) } else { NEG_INF };
                tm_a = fmaxf(tm_a, fmaxf(sc[nj][0], sc[nj][1]));
                tm_b = fmaxf(tm_b, fmaxf(sc[nj][2], sc[nj][3]));
                nj += 1;
            }
            tm_a = fmaxf(tm_a, warp::shuffle_xor_f32_sync(0xffff_ffff, tm_a, 1));
            tm_a = fmaxf(tm_a, warp::shuffle_xor_f32_sync(0xffff_ffff, tm_a, 2));
            tm_b = fmaxf(tm_b, warp::shuffle_xor_f32_sync(0xffff_ffff, tm_b, 1));
            tm_b = fmaxf(tm_b, warp::shuffle_xor_f32_sync(0xffff_ffff, tm_b, 2));
            let mn_a = fmaxf(m_a, tm_a);
            let mn_b = fmaxf(m_b, tm_b);
            let mu_a = if mn_a == NEG_INF { 0.0 } else { mn_a };
            let mu_b = if mn_b == NEG_INF { 0.0 } else { mn_b };
            let cr_a = ex2(sub(m_a, mu_a));
            let cr_b = ex2(sub(m_b, mu_b));
            m_a = mn_a;
            m_b = mn_b;

            let mut ps_a = 0f32;
            let mut ps_b = 0f32;
            let mut nj = 0;
            while nj < 4 {
                unroll!();
                sc[nj][0] = ex2(sub(sc[nj][0], mu_a));
                sc[nj][1] = ex2(sub(sc[nj][1], mu_a));
                sc[nj][2] = ex2(sub(sc[nj][2], mu_b));
                sc[nj][3] = ex2(sub(sc[nj][3], mu_b));
                ps_a = add(ps_a, add(sc[nj][0], sc[nj][1]));
                ps_b = add(ps_b, add(sc[nj][2], sc[nj][3]));
                nj += 1;
            }
            l_a = fma(l_a, cr_a, ps_a);
            l_b = fma(l_b, cr_b, ps_b);
            let mut dn = 0;
            while dn < 32 {
                unroll!();
                o[dn][0] = mul(o[dn][0], cr_a);
                o[dn][1] = mul(o[dn][1], cr_a);
                o[dn][2] = mul(o[dn][2], cr_b);
                o[dn][3] = mul(o[dn][3], cr_b);
                dn += 1;
            }

            cp_wait_all();
            thread::sync_threads();
            if tile + 1 < te {
                load_k512::<NW>(sk, kbase, t0 + BC, k1, tid);
            }
            cp_commit();

            let mut ks = 0;
            while ks < 2 {
                unroll!();
                let pa0 = pack_p::<F16>(sc[2 * ks][0], sc[2 * ks][1]);
                let pa1 = pack_p::<F16>(sc[2 * ks][2], sc[2 * ks][3]);
                let pa2 = pack_p::<F16>(sc[2 * ks + 1][0], sc[2 * ks + 1][1]);
                let pa3 = pack_p::<F16>(sc[2 * ks + 1][2], sc[2 * ks + 1][3]);
                let vr = 16 * ks as u32 + (mi & 1) * 8 + ri;
                let mut np = 0;
                while np < 16 {
                    unroll!();
                    let ch = 2 * np as u32 + (mi >> 1);
                    let (b0, b1, b2, b3) = ldsm_x4_trans(sv + swz(vr, ch));
                    o[2 * np] = mma_t::<F16>(o[2 * np], pa0, pa1, pa2, pa3, b0, b1);
                    o[2 * np + 1] = mma_t::<F16>(o[2 * np + 1], pa0, pa1, pa2, pa3, b2, b3);
                    np += 1;
                }
                ks += 1;
            }
            tile += 1;
        }
        cp_wait_all();

        l_a = add(l_a, warp::shuffle_xor_f32_sync(0xffff_ffff, l_a, 1));
        l_a = add(l_a, warp::shuffle_xor_f32_sync(0xffff_ffff, l_a, 2));
        l_b = add(l_b, warp::shuffle_xor_f32_sync(0xffff_ffff, l_b, 1));
        l_b = add(l_b, warp::shuffle_xor_f32_sync(0xffff_ffff, l_b, 2));

        if nsplit == 1 {
            let oa = out.add((hd as i64 * o_hs as i64 + pos_a as i64 * 512 + vo as i64) as usize);
            let ob = out.add((hd as i64 * o_hs as i64 + pos_b as i64 * 512 + vo as i64) as usize);
            let mut dn = 0;
            while dn < 32 {
                unroll!();
                let col = (8 * dn as i32 + 2 * t) as usize;
                if O32 {
                    if va {
                        stg_v2f((out as *mut f32).add((hd as i64 * o_hs as i64 + pos_a as i64 * 512 + vo as i64) as usize + col), div_rn(o[dn][0], l_a), div_rn(o[dn][1], l_a));
                    }
                    if vb {
                        stg_v2f((out as *mut f32).add((hd as i64 * o_hs as i64 + pos_b as i64 * 512 + vo as i64) as usize + col), div_rn(o[dn][2], l_b), div_rn(o[dn][3], l_b));
                    }
                } else {
                    if va {
                        stg_u32(oa.add(col), pack_bf2(div_rn(o[dn][0], l_a), div_rn(o[dn][1], l_a)));
                    }
                    if vb {
                        stg_u32(ob.add(col), pack_bf2(div_rn(o[dn][2], l_b), div_rn(o[dn][3], l_b)));
                    }
                }
                dn += 1;
            }
        } else {
            let ra = ((split as i64 * n_heads as i64 + hd as i64) * s as i64 + pos_a as i64) as usize;
            let rb = ra + 1;
            let mut dn = 0;
            while dn < 32 {
                unroll!();
                let col = (vo + 8 * dn as i32 + 2 * t) as usize;
                if va {
                    stg_v2f(part.add(ra * 512 + col), o[dn][0], o[dn][1]);
                }
                if vb {
                    stg_v2f(part.add(rb * 512 + col), o[dn][2], o[dn][3]);
                }
                dn += 1;
            }
            // both halves hold the same (m, l); half 0 writes them
            if t == 0 && vo == 0 {
                if va {
                    stg_v2f(ml.add(ra * 2), m_a, l_a);
                }
                if vb {
                    stg_v2f(ml.add(rb * 2), m_b, l_b);
                }
            }
        }
    }

    /// Head dim 512, 4 warps (8 positions) per block: grid (ceil(s / 8), n_heads / 8, 2 nsplit), block 128.
    #[kernel]
    #[launch_bounds(128, 2)]
    pub unsafe fn flash_prefill_d512(
        q: *const u16, k: *const u16, v: *const u16, out: *mut u16, part: *mut f32, ml: *mut f32,
        scale_log2: f32, s: i32, past: i32, kv_len: i32, n_heads: i32, nsplit: i32,
        q_hs: i32, o_hs: i32, k_hs: u64, v_hs: u64, win: i32, gq: i32,
    ) {
        static mut SMEM: SharedArray<u16, 24576, 16> = SharedArray::UNINIT;
        let sk = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        body512::<4, false, false>(sk, q, k, v, out, part, ml, scale_log2, s, past, kv_len, n_heads, nsplit, q_hs, o_hs, k_hs, v_hs, win, gq);
    }

    /// Head dim 512, 8 warps (16 positions) per block: grid (ceil(s / 16), n_heads / 8, 2 nsplit), block 256.
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn flash_prefill_d512_w8(
        q: *const u16, k: *const u16, v: *const u16, out: *mut u16, part: *mut f32, ml: *mut f32,
        scale_log2: f32, s: i32, past: i32, kv_len: i32, n_heads: i32, nsplit: i32,
        q_hs: i32, o_hs: i32, k_hs: u64, v_hs: u64, win: i32, gq: i32,
    ) {
        static mut SMEM: SharedArray<u16, 24576, 16> = SharedArray::UNINIT;
        let sk = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        body512::<8, false, false>(sk, q, k, v, out, part, ml, scale_log2, s, past, kv_len, n_heads, nsplit, q_hs, o_hs, k_hs, v_hs, win, gq);
    }

    /// `flash_prefill_w8` on f16 Q / K / V (f16 probabilities, as llama.cpp's fattn-mma on an f16 KV cache); bf16 out.
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn flash_prefill_w8_f16(
        q: *const u16, k: *const u16, v: *const u16, out: *mut u16, part: *mut f32, ml: *mut f32,
        scale_log2: f32, s: i32, past: i32, kv_len: i32, n_heads: i32, nsplit: i32,
        q_hs: i32, o_hs: i32, k_hs: u64, v_hs: u64, win: i32,
    ) {
        static mut SMEM: SharedArray<u16, 16384, 16> = SharedArray::UNINIT;
        let sk = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        body::<8, true, true, 8, false>(sk, q, k, v, out, part, ml, scale_log2, s, past, kv_len, n_heads, nsplit, q_hs, o_hs, k_hs, v_hs, win);
    }

    /// `flash_prefill_d512` on f16 Q / K / V; bf16 out.
    #[kernel]
    #[launch_bounds(128, 2)]
    pub unsafe fn flash_prefill_d512_f16(
        q: *const u16, k: *const u16, v: *const u16, out: *mut u16, part: *mut f32, ml: *mut f32,
        scale_log2: f32, s: i32, past: i32, kv_len: i32, n_heads: i32, nsplit: i32,
        q_hs: i32, o_hs: i32, k_hs: u64, v_hs: u64, win: i32, gq: i32,
    ) {
        static mut SMEM: SharedArray<u16, 24576, 16> = SharedArray::UNINIT;
        let sk = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        body512::<4, true, false>(sk, q, k, v, out, part, ml, scale_log2, s, past, kv_len, n_heads, nsplit, q_hs, o_hs, k_hs, v_hs, win, gq);
    }

    /// `flash_prefill_d512_w8` on f16 Q / K / V; bf16 out.
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn flash_prefill_d512_w8_f16(
        q: *const u16, k: *const u16, v: *const u16, out: *mut u16, part: *mut f32, ml: *mut f32,
        scale_log2: f32, s: i32, past: i32, kv_len: i32, n_heads: i32, nsplit: i32,
        q_hs: i32, o_hs: i32, k_hs: u64, v_hs: u64, win: i32, gq: i32,
    ) {
        static mut SMEM: SharedArray<u16, 24576, 16> = SharedArray::UNINIT;
        let sk = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        body512::<8, true, false>(sk, q, k, v, out, part, ml, scale_log2, s, past, kv_len, n_heads, nsplit, q_hs, o_hs, k_hs, v_hs, win, gq);
    }

    /// Head dim 256, 2 query heads per KV head (gemma4 sliding layers) x 8 positions per warp tile, 8 warps (64 positions per block): grid (ceil(s / 64), kv_heads, nsplit); bf16 in / out.
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn flash_prefill_r2_w8(
        q: *const u16, k: *const u16, v: *const u16, out: *mut u16, part: *mut f32, ml: *mut f32,
        scale_log2: f32, s: i32, past: i32, kv_len: i32, n_heads: i32, nsplit: i32,
        q_hs: i32, o_hs: i32, k_hs: u64, v_hs: u64, win: i32,
    ) {
        static mut SMEM: SharedArray<u16, 16384, 16> = SharedArray::UNINIT;
        let sk = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        body::<8, true, false, 2, false>(sk, q, k, v, out, part, ml, scale_log2, s, past, kv_len, n_heads, nsplit, q_hs, o_hs, k_hs, v_hs, win);
    }

    /// `flash_prefill_r2_w8` on f16 Q / K / V with f32 output (llama.cpp layout: f16 KV, f32 activations).
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn flash_prefill_r2_w8_f16o32(
        q: *const u16, k: *const u16, v: *const u16, out: *mut u16, part: *mut f32, ml: *mut f32,
        scale_log2: f32, s: i32, past: i32, kv_len: i32, n_heads: i32, nsplit: i32,
        q_hs: i32, o_hs: i32, k_hs: u64, v_hs: u64, win: i32,
    ) {
        static mut SMEM: SharedArray<u16, 16384, 16> = SharedArray::UNINIT;
        let sk = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        body::<8, true, true, 2, true>(sk, q, k, v, out, part, ml, scale_log2, s, past, kv_len, n_heads, nsplit, q_hs, o_hs, k_hs, v_hs, win);
    }

    /// `flash_prefill_w8_f16` with f32 output.
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn flash_prefill_w8_f16o32(
        q: *const u16, k: *const u16, v: *const u16, out: *mut u16, part: *mut f32, ml: *mut f32,
        scale_log2: f32, s: i32, past: i32, kv_len: i32, n_heads: i32, nsplit: i32,
        q_hs: i32, o_hs: i32, k_hs: u64, v_hs: u64, win: i32,
    ) {
        static mut SMEM: SharedArray<u16, 16384, 16> = SharedArray::UNINIT;
        let sk = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        body::<8, true, true, 8, true>(sk, q, k, v, out, part, ml, scale_log2, s, past, kv_len, n_heads, nsplit, q_hs, o_hs, k_hs, v_hs, win);
    }

    /// `flash_prefill_d512_f16` with f32 output.
    #[kernel]
    #[launch_bounds(128, 2)]
    pub unsafe fn flash_prefill_d512_f16o32(
        q: *const u16, k: *const u16, v: *const u16, out: *mut u16, part: *mut f32, ml: *mut f32,
        scale_log2: f32, s: i32, past: i32, kv_len: i32, n_heads: i32, nsplit: i32,
        q_hs: i32, o_hs: i32, k_hs: u64, v_hs: u64, win: i32, gq: i32,
    ) {
        static mut SMEM: SharedArray<u16, 24576, 16> = SharedArray::UNINIT;
        let sk = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        body512::<4, true, true>(sk, q, k, v, out, part, ml, scale_log2, s, past, kv_len, n_heads, nsplit, q_hs, o_hs, k_hs, v_hs, win, gq);
    }

    /// `flash_prefill_combine` writing f32 (the o32 kernels' split merge).
    #[kernel]
    pub unsafe fn flash_prefill_combine_f32(
        part: *const f32, ml: *const f32, out: *mut f32, s: i32, n_heads: i32, nsplit: i32, o_hs: i32,
    ) {
        let d = thread::threadIdx_x() as i64;
        let pos = thread::blockIdx_x() as i64;
        let hd = thread::blockIdx_y() as i64;
        let row = |i: i32| ((i as i64 * n_heads as i64 + hd) * s as i64 + pos) as usize;
        let mut mx = NEG_INF;
        let mut i = 0;
        while i < nsplit {
            mx = fmaxf(mx, *ml.add(row(i) * 2));
            i += 1;
        }
        let mut num = 0f32;
        let mut den = 0f32;
        let mut i = 0;
        while i < nsplit {
            let m = *ml.add(row(i) * 2);
            if m != NEG_INF {
                let w = ex2(sub(m, mx));
                num = fma(w, *part.add(row(i) * 256 + d as usize), num);
                den = fma(w, *ml.add(row(i) * 2 + 1), den);
            }
            i += 1;
        }
        *out.add((hd * o_hs as i64 + pos * 256 + d) as usize) = div_rn(num, den);
    }

    /// `flash_prefill_combine512` writing f32.
    #[kernel]
    pub unsafe fn flash_prefill_combine512_f32(
        part: *const f32, ml: *const f32, out: *mut f32, s: i32, n_heads: i32, nsplit: i32, o_hs: i32,
    ) {
        let d0 = thread::threadIdx_x() as i64;
        let pos = thread::blockIdx_x() as i64;
        let hd = thread::blockIdx_y() as i64;
        let row = |i: i32| ((i as i64 * n_heads as i64 + hd) * s as i64 + pos) as usize;
        let mut mx = NEG_INF;
        let mut i = 0;
        while i < nsplit {
            mx = fmaxf(mx, *ml.add(row(i) * 2));
            i += 1;
        }
        let mut j = 0;
        while j < 2 {
            let d = d0 + 256 * j as i64;
            let mut num = 0f32;
            let mut den = 0f32;
            let mut i = 0;
            while i < nsplit {
                let m = *ml.add(row(i) * 2);
                if m != NEG_INF {
                    let w = ex2(sub(m, mx));
                    num = fma(w, *part.add(row(i) * 512 + d as usize), num);
                    den = fma(w, *ml.add(row(i) * 2 + 1), den);
                }
                i += 1;
            }
            *out.add((hd * o_hs as i64 + pos * 512 + d) as usize) = div_rn(num, den);
            j += 1;
        }
    }

    /// `flash_prefill_combine` at head dim 512: grid (s, n_heads), block 256, two dims per thread.
    #[kernel]
    pub unsafe fn flash_prefill_combine512(
        part: *const f32, ml: *const f32, out: *mut u16, s: i32, n_heads: i32, nsplit: i32, o_hs: i32,
    ) {
        let d0 = thread::threadIdx_x() as i64;
        let pos = thread::blockIdx_x() as i64;
        let hd = thread::blockIdx_y() as i64;
        let row = |i: i32| ((i as i64 * n_heads as i64 + hd) * s as i64 + pos) as usize;
        let mut mx = NEG_INF;
        let mut i = 0;
        while i < nsplit {
            mx = fmaxf(mx, *ml.add(row(i) * 2));
            i += 1;
        }
        let mut j = 0;
        while j < 2 {
            let d = d0 + 256 * j as i64;
            let mut num = 0f32;
            let mut den = 0f32;
            let mut i = 0;
            while i < nsplit {
                let m = *ml.add(row(i) * 2);
                if m != NEG_INF {
                    let w = ex2(sub(m, mx));
                    num = fma(w, *part.add(row(i) * 512 + d as usize), num);
                    den = fma(w, *ml.add(row(i) * 2 + 1), den);
                }
                i += 1;
            }
            *out.add((hd * o_hs as i64 + pos * 512 + d) as usize) = f2bf(div_rn(num, den));
            j += 1;
        }
    }

    // ---- head dim 128 (qwen3 dense: 40 query heads on 8 KV heads, zero-padded to 8 per KV head) ----

    /// Bytes per K / V row in shared memory at head dim 128 (16 swizzled 16-byte chunks).
    pub const ROW_BYTES_128: u32 = 256;
    /// V tile offset in the 128 kernels' shared memory: after the 32 x 256-byte K tile.
    pub const V_OFF_128: u32 = 32 * 256;

    /// Byte offset of (row r, 16-byte chunk ch < 16) in a swizzled 256-byte-row tile.
    #[inline(always)]
    pub fn swz128(r: u32, ch: u32) -> u32 {
        r * ROW_BYTES_128 + ((ch ^ (r & 7)) << 4)
    }

    /// One 32-key K or V tile at head dim 128 (32 rows x 16 chunks, swizzled) by cp.async, keys `>= k1`
    /// zero-filled without a read. `NW` warps share the 512 chunks.
    #[inline(always)]
    pub unsafe fn load_tile128<const NW: i32>(dst: u32, base: *const u16, t0: i32, k1: i32, tid: i32) {
        let mut j = 0;
        while j < 16 / NW {
            unroll!();
            let i = tid + 32 * NW * j;
            let r = i >> 4;
            let ch = i & 15;
            let key = t0 + r;
            let ok = key < k1;
            let src = if ok { base.add((key as u64 * 128 + ch as u64 * 8) as usize) } else { base };
            cp_async16(dst + swz128(r as u32, ch as u32), src, if ok { 16 } else { 0 });
            j += 1;
        }
    }

    /// `body` at head dim 128: the same pipeline, products and sums in the same order, over 8 k-steps of 16 dims
    /// (S = Q K^T) and 16 n-tiles of 8 dims (O += P V); Q in registers, 8 query heads per KV head (REP 8).
    #[inline(always)]
    pub unsafe fn body128<const NW: i32, const F16: bool, const O32: bool>(
        sk: u32, q: *const u16, k: *const u16, v: *const u16, out: *mut u16, part: *mut f32, ml: *mut f32,
        scale_log2: f32, s: i32, past: i32, kv_len: i32, n_heads: i32, nsplit: i32,
        q_hs: i32, o_hs: i32, k_hs: u64, v_hs: u64, win: i32,
    ) {
        let sv = sk + V_OFF_128;
        let tid = thread::threadIdx_x() as i32;
        let lane = tid & 31;
        let wid = tid >> 5;
        let g = lane >> 2;
        let t = lane & 3;
        let pb = thread::blockIdx_x() as i32;
        let kvh = thread::blockIdx_y() as i32;
        let split = thread::blockIdx_z() as i32;
        // 8 query heads x 2 positions per warp tile: row r = (head r % 8, position r / 8)
        let p0 = pb * 2 * NW;
        let pos_a = p0 + 2 * wid;
        let pos_b = pos_a + 1;
        let hd = kvh * 8 + g;
        let lim_a = past + pos_a + 1;
        let lim_b = past + pos_b + 1;
        let lo_a = if win > 0 { lim_a - win } else { 0 };
        let lo_b = if win > 0 { lim_b - win } else { 0 };
        let kbeg = if win > 0 && past + p0 + 1 > win { past + p0 + 1 - win } else { 0 };
        let tfirst = kbeg / BC;
        let kend = imin(kv_len, past + p0 + 2 * NW);
        let ntiles = (kend + BC - 1) / BC - tfirst;
        let tps = (ntiles + nsplit - 1) / nsplit;
        let tb = tfirst + split * tps;
        let te = imin(tfirst + ntiles, tb + tps);
        let k1 = imin(kend, te * BC);

        let qa = q.add((hd as i64 * q_hs as i64 + pos_a as i64 * 128) as usize);
        let qb = q.add((hd as i64 * q_hs as i64 + pos_b as i64 * 128) as usize);
        let va = pos_a < s;
        let vb = pos_b < s;
        let kbase = k.add((kvh as u64 * k_hs) as usize);
        let vbase = v.add((kvh as u64 * v_hs) as usize);

        let mut qf = [[0u32; 4]; 8];
        let mut kk = 0;
        while kk < 8 {
            unroll!();
            let col = (16 * kk + 2 * t) as usize;
            qf[kk as usize][0] = if va { ldg_u32(qa.add(col)) } else { 0 };
            qf[kk as usize][1] = if vb { ldg_u32(qb.add(col)) } else { 0 };
            qf[kk as usize][2] = if va { ldg_u32(qa.add(col + 8)) } else { 0 };
            qf[kk as usize][3] = if vb { ldg_u32(qb.add(col + 8)) } else { 0 };
            kk += 1;
        }

        let mut o = [[0f32; 4]; 16];
        let mut m_a = NEG_INF;
        let mut m_b = NEG_INF;
        let mut l_a = 0f32;
        let mut l_b = 0f32;
        let mi = (lane >> 3) as u32;
        let ri = (lane & 7) as u32;

        if tb < te {
            load_tile128::<NW>(sk, kbase, tb * BC, k1, tid);
        }
        cp_commit();
        let mut tile = tb;
        while tile < te {
            let t0 = tile * BC;
            cp_wait_all();
            thread::sync_threads();
            load_tile128::<NW>(sv, vbase, t0, k1, tid);
            cp_commit();

            let mut sc = [[0f32; 4]; 4];
            let mut kk = 0;
            while kk < 8 {
                unroll!();
                let (a0, a1, a2, a3) = (qf[kk as usize][0], qf[kk as usize][1], qf[kk as usize][2], qf[kk as usize][3]);
                let mut np = 0;
                while np < 2 {
                    unroll!();
                    let r = 16 * np as u32 + (mi >> 1) * 8 + ri;
                    let ch = 2 * kk as u32 + (mi & 1);
                    let (b0, b1, b2, b3) = ldsm_x4(sk + swz128(r, ch));
                    sc[2 * np] = mma_t::<F16>(sc[2 * np], a0, a1, a2, a3, b0, b1);
                    sc[2 * np + 1] = mma_t::<F16>(sc[2 * np + 1], a0, a1, a2, a3, b2, b3);
                    np += 1;
                }
                kk += 1;
            }

            let ea = imin(lim_a, k1);
            let eb = imin(lim_b, k1);
            let mut tm_a = NEG_INF;
            let mut tm_b = NEG_INF;
            let mut nj = 0;
            while nj < 4 {
                unroll!();
                let key = t0 + 8 * nj as i32 + 2 * t;
                sc[nj][0] = if key < ea && key >= lo_a { mul(sc[nj][0], scale_log2) } else { NEG_INF };
                sc[nj][1] = if key + 1 < ea && key + 1 >= lo_a { mul(sc[nj][1], scale_log2) } else { NEG_INF };
                sc[nj][2] = if key < eb && key >= lo_b { mul(sc[nj][2], scale_log2) } else { NEG_INF };
                sc[nj][3] = if key + 1 < eb && key + 1 >= lo_b { mul(sc[nj][3], scale_log2) } else { NEG_INF };
                tm_a = fmaxf(tm_a, fmaxf(sc[nj][0], sc[nj][1]));
                tm_b = fmaxf(tm_b, fmaxf(sc[nj][2], sc[nj][3]));
                nj += 1;
            }
            tm_a = fmaxf(tm_a, warp::shuffle_xor_f32_sync(0xffff_ffff, tm_a, 1));
            tm_a = fmaxf(tm_a, warp::shuffle_xor_f32_sync(0xffff_ffff, tm_a, 2));
            tm_b = fmaxf(tm_b, warp::shuffle_xor_f32_sync(0xffff_ffff, tm_b, 1));
            tm_b = fmaxf(tm_b, warp::shuffle_xor_f32_sync(0xffff_ffff, tm_b, 2));
            let mn_a = fmaxf(m_a, tm_a);
            let mn_b = fmaxf(m_b, tm_b);
            let mu_a = if mn_a == NEG_INF { 0.0 } else { mn_a };
            let mu_b = if mn_b == NEG_INF { 0.0 } else { mn_b };
            let cr_a = ex2(sub(m_a, mu_a));
            let cr_b = ex2(sub(m_b, mu_b));
            m_a = mn_a;
            m_b = mn_b;

            let mut ps_a = 0f32;
            let mut ps_b = 0f32;
            let mut nj = 0;
            while nj < 4 {
                unroll!();
                sc[nj][0] = ex2(sub(sc[nj][0], mu_a));
                sc[nj][1] = ex2(sub(sc[nj][1], mu_a));
                sc[nj][2] = ex2(sub(sc[nj][2], mu_b));
                sc[nj][3] = ex2(sub(sc[nj][3], mu_b));
                ps_a = add(ps_a, add(sc[nj][0], sc[nj][1]));
                ps_b = add(ps_b, add(sc[nj][2], sc[nj][3]));
                nj += 1;
            }
            l_a = fma(l_a, cr_a, ps_a);
            l_b = fma(l_b, cr_b, ps_b);
            let mut dn = 0;
            while dn < 16 {
                unroll!();
                o[dn][0] = mul(o[dn][0], cr_a);
                o[dn][1] = mul(o[dn][1], cr_a);
                o[dn][2] = mul(o[dn][2], cr_b);
                o[dn][3] = mul(o[dn][3], cr_b);
                dn += 1;
            }

            cp_wait_all();
            thread::sync_threads();
            if tile + 1 < te {
                load_tile128::<NW>(sk, kbase, t0 + BC, k1, tid);
            }
            cp_commit();

            let mut ks = 0;
            while ks < 2 {
                unroll!();
                let pa0 = pack_p::<F16>(sc[2 * ks][0], sc[2 * ks][1]);
                let pa1 = pack_p::<F16>(sc[2 * ks][2], sc[2 * ks][3]);
                let pa2 = pack_p::<F16>(sc[2 * ks + 1][0], sc[2 * ks + 1][1]);
                let pa3 = pack_p::<F16>(sc[2 * ks + 1][2], sc[2 * ks + 1][3]);
                let vr = 16 * ks as u32 + (mi & 1) * 8 + ri;
                let mut np = 0;
                while np < 8 {
                    unroll!();
                    let ch = 2 * np as u32 + (mi >> 1);
                    let (b0, b1, b2, b3) = ldsm_x4_trans(sv + swz128(vr, ch));
                    o[2 * np] = mma_t::<F16>(o[2 * np], pa0, pa1, pa2, pa3, b0, b1);
                    o[2 * np + 1] = mma_t::<F16>(o[2 * np + 1], pa0, pa1, pa2, pa3, b2, b3);
                    np += 1;
                }
                ks += 1;
            }
            tile += 1;
        }
        cp_wait_all();

        l_a = add(l_a, warp::shuffle_xor_f32_sync(0xffff_ffff, l_a, 1));
        l_a = add(l_a, warp::shuffle_xor_f32_sync(0xffff_ffff, l_a, 2));
        l_b = add(l_b, warp::shuffle_xor_f32_sync(0xffff_ffff, l_b, 1));
        l_b = add(l_b, warp::shuffle_xor_f32_sync(0xffff_ffff, l_b, 2));

        if nsplit == 1 {
            let oa = out.add((hd as i64 * o_hs as i64 + pos_a as i64 * 128) as usize);
            let ob = out.add((hd as i64 * o_hs as i64 + pos_b as i64 * 128) as usize);
            let mut dn = 0;
            while dn < 16 {
                unroll!();
                let col = (8 * dn as i32 + 2 * t) as usize;
                if O32 {
                    if va {
                        stg_v2f((out as *mut f32).add((hd as i64 * o_hs as i64 + pos_a as i64 * 128) as usize + col), div_rn(o[dn][0], l_a), div_rn(o[dn][1], l_a));
                    }
                    if vb {
                        stg_v2f((out as *mut f32).add((hd as i64 * o_hs as i64 + pos_b as i64 * 128) as usize + col), div_rn(o[dn][2], l_b), div_rn(o[dn][3], l_b));
                    }
                } else {
                    if va {
                        stg_u32(oa.add(col), pack_bf2(div_rn(o[dn][0], l_a), div_rn(o[dn][1], l_a)));
                    }
                    if vb {
                        stg_u32(ob.add(col), pack_bf2(div_rn(o[dn][2], l_b), div_rn(o[dn][3], l_b)));
                    }
                }
                dn += 1;
            }
        } else {
            let ra = ((split as i64 * n_heads as i64 + hd as i64) * s as i64 + pos_a as i64) as usize;
            let rb = ((split as i64 * n_heads as i64 + hd as i64) * s as i64 + pos_b as i64) as usize;
            let mut dn = 0;
            while dn < 16 {
                unroll!();
                let col = (8 * dn as i32 + 2 * t) as usize;
                if va {
                    stg_v2f(part.add(ra * 128 + col), o[dn][0], o[dn][1]);
                }
                if vb {
                    stg_v2f(part.add(rb * 128 + col), o[dn][2], o[dn][3]);
                }
                dn += 1;
            }
            if t == 0 {
                if va {
                    stg_v2f(ml.add(ra * 2), m_a, l_a);
                }
                if vb {
                    stg_v2f(ml.add(rb * 2), m_b, l_b);
                }
            }
        }
    }

    /// Head dim 128, bf16 in / out, 8 warps (16 positions) per block: grid (ceil(s / 16), kv_heads, nsplit), block 256.
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn flash_prefill_d128_w8(
        q: *const u16, k: *const u16, v: *const u16, out: *mut u16, part: *mut f32, ml: *mut f32,
        scale_log2: f32, s: i32, past: i32, kv_len: i32, n_heads: i32, nsplit: i32,
        q_hs: i32, o_hs: i32, k_hs: u64, v_hs: u64, win: i32,
    ) {
        static mut SMEM: SharedArray<u16, 8192, 16> = SharedArray::UNINIT;
        let sk = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        body128::<8, false, false>(sk, q, k, v, out, part, ml, scale_log2, s, past, kv_len, n_heads, nsplit, q_hs, o_hs, k_hs, v_hs, win);
    }

    /// `flash_prefill_d128_w8` on f16 Q / K / V (f16 probabilities) with f32 output.
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn flash_prefill_d128_w8_f16o32(
        q: *const u16, k: *const u16, v: *const u16, out: *mut u16, part: *mut f32, ml: *mut f32,
        scale_log2: f32, s: i32, past: i32, kv_len: i32, n_heads: i32, nsplit: i32,
        q_hs: i32, o_hs: i32, k_hs: u64, v_hs: u64, win: i32,
    ) {
        static mut SMEM: SharedArray<u16, 8192, 16> = SharedArray::UNINIT;
        let sk = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        body128::<8, true, true>(sk, q, k, v, out, part, ml, scale_log2, s, past, kv_len, n_heads, nsplit, q_hs, o_hs, k_hs, v_hs, win);
    }

    /// `flash_prefill_combine` at head dim 128 (bf16 out): grid (s, n_heads), block 128.
    #[kernel]
    pub unsafe fn flash_prefill_combine128(
        part: *const f32, ml: *const f32, out: *mut u16, s: i32, n_heads: i32, nsplit: i32, o_hs: i32,
    ) {
        let d = thread::threadIdx_x() as i64;
        let pos = thread::blockIdx_x() as i64;
        let hd = thread::blockIdx_y() as i64;
        let row = |i: i32| ((i as i64 * n_heads as i64 + hd) * s as i64 + pos) as usize;
        let mut mx = NEG_INF;
        let mut i = 0;
        while i < nsplit {
            mx = fmaxf(mx, *ml.add(row(i) * 2));
            i += 1;
        }
        let mut num = 0f32;
        let mut den = 0f32;
        let mut i = 0;
        while i < nsplit {
            let m = *ml.add(row(i) * 2);
            if m != NEG_INF {
                let w = ex2(sub(m, mx));
                num = fma(w, *part.add(row(i) * 128 + d as usize), num);
                den = fma(w, *ml.add(row(i) * 2 + 1), den);
            }
            i += 1;
        }
        *out.add((hd * o_hs as i64 + pos * 128 + d) as usize) = f2bf(div_rn(num, den));
    }

    /// `flash_prefill_combine128` writing f32.
    #[kernel]
    pub unsafe fn flash_prefill_combine128_f32(
        part: *const f32, ml: *const f32, out: *mut f32, s: i32, n_heads: i32, nsplit: i32, o_hs: i32,
    ) {
        let d = thread::threadIdx_x() as i64;
        let pos = thread::blockIdx_x() as i64;
        let hd = thread::blockIdx_y() as i64;
        let row = |i: i32| ((i as i64 * n_heads as i64 + hd) * s as i64 + pos) as usize;
        let mut mx = NEG_INF;
        let mut i = 0;
        while i < nsplit {
            mx = fmaxf(mx, *ml.add(row(i) * 2));
            i += 1;
        }
        let mut num = 0f32;
        let mut den = 0f32;
        let mut i = 0;
        while i < nsplit {
            let m = *ml.add(row(i) * 2);
            if m != NEG_INF {
                let w = ex2(sub(m, mx));
                num = fma(w, *part.add(row(i) * 128 + d as usize), num);
                den = fma(w, *ml.add(row(i) * 2 + 1), den);
            }
            i += 1;
        }
        *out.add((hd * o_hs as i64 + pos * 128 + d) as usize) = div_rn(num, den);
    }
}

fn main() {
    std::process::exit(if gate::run() { 0 } else { 1 });
}
