//! CPU twin of the expert GEMV, bit-identical to the GPU kernels' candle float program.
//!
//! Everything integer (dp4a, scale unpacking) is exact anywhere. The floats match
//! because x86 `fma` and IEEE mul/add are the GPU's `fma.rn`/`mul.rn`/`add.rn`
//! (candle's build does not flush denormals, and neither does the CPU by default),
//! and because the sum is formed in the GPU's exact order: 128 per-thread partials,
//! warps 1..3 added into warp 0 in warp order, then the xor butterfly 16, 8, 4, 2, 1.
//! Tiering can therefore move an expert between the GPU and the CPU without
//! changing a single output bit. A NaN result is returned as the GPU's canonical
//! NaN (0x7FFFFFFF): x86 propagates operand payloads, the GPU does not.
//!
//! Q1_0 twins llama.cpp's fast-math program (`q1_0_q8_1_moe_gemv`: mul.ftz, fma.ftz, add.ftz) with
//! plain IEEE ops: every product of two f16 scales is exact and a multiple of 2^-48, so every value
//! the dot and the reduction produce is 0 or a multiple of 2^-71 -- never an f32 denormal, so the
//! flushes never fire. IQ4_NL (`iq4_nl_q8_1_moe_gemv`, the same fast-math program shape) twins
//! with plain IEEE ops for the same reason. MXFP4's E8M0 scales reach the denormal range, so its
//! twin flushes exactly where the GPU's `.ftz` ops do (inputs and results, sign kept). NVFP4's
//! scales are E4M3 / 2 (>= 2^-10) times an f16, so its values stay normal: plain IEEE again.

const Q8_BYTES: usize = 36;
const THREADS: usize = 128;
/// Q4_K / Q5_K: qi / vdr threads per K-quant block, blocks per iteration, vdr.
const THREADS_PER_BLOCK: usize = 16;
const BLOCKS_PER_ITER: usize = 8;
/// Q6_K (qi 32, vdr 1).
const Q6K_THREADS_PER_BLOCK: usize = 32;
const Q6K_BLOCKS_PER_ITER: usize = 4;
/// Q1_0 (llama.cpp mul_mat_vec_q, qi 4, vdr 1): threads per block, blocks per iteration.
const Q1_0_THREADS_PER_BLOCK: usize = 4;
const Q1_0_BLOCKS_PER_ITER: usize = 32;
/// IQ4_NL (llama.cpp mul_mat_vec_q, qi 4, vdr 2).
const IQ4NL_THREADS_PER_BLOCK: usize = 2;
const IQ4NL_BLOCKS_PER_ITER: usize = 64;
/// kvalues_iq4nl (ggml-common.h).
const KVALUES_IQ4NL: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];
/// kvalues_mxfp4 (ggml-common.h kvalues_fp4): doubled E2M1 values.
const KVALUES_MXFP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// The GPU's `.ftz`: a subnormal operand or result is a zero of the same sign.
#[inline(always)]
fn ftz(v: f32) -> f32 {
    if v.is_subnormal() { f32::from_bits(v.to_bits() & 0x8000_0000) } else { v }
}
#[inline(always)]
fn mul_ftz(a: f32, b: f32) -> f32 {
    ftz(ftz(a) * ftz(b))
}
#[inline(always)]
fn add_ftz(a: f32, b: f32) -> f32 {
    ftz(ftz(a) + ftz(b))
}
#[inline(always)]
fn fma_ftz(a: f32, b: f32, c: f32) -> f32 {
    ftz(ftz(a).mul_add(ftz(b), ftz(c)))
}

/// `cvt.rn.bf16x2.ue8m0x2` then bf16 -> f32: 2^(e - 127), e = 255 NaN (e = 0 gives the
/// denormal 2^-127, which every consumer flushes).
#[inline(always)]
fn e8m0_to_f32(e: u8) -> f32 {
    match e {
        255 => f32::NAN,
        0 => f32::from_bits(0x0040_0000),
        _ => f32::from_bits((e as u32) << 23),
    }
}
/// block_q6_K: ql 0..128, qh 128..192, int8 scales 192..208, half d 208..210.
const Q6K_QH: usize = 128;
const Q6K_SCALES: usize = 192;
const Q6K_D: usize = 208;

/// The GPU's NaN: every f32 op there returns 0x7FFFFFFF for a NaN result.
#[inline(always)]
fn canonical(v: f32) -> f32 {
    if v.is_nan() { f32::from_bits(0x7FFF_FFFF) } else { v }
}

/// IEEE binary16 to f32, exact (every f16 is an f32).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let man = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            // subnormal: normalise
            let mut e = 113u32;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3FF) << 13)
        }
    } else if exp == 31 {
        sign | 0x7F80_0000 | (man << 13)
    } else {
        sign | ((exp + 112) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

#[inline(always)]
fn word(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}

/// `__dp4a` (signed x signed bytes, plus accumulator).
#[inline(always)]
fn dp4a(a: u32, b: u32, c: i32) -> i32 {
    let mut s = c;
    for i in 0..4 {
        s += ((a >> (8 * i)) as u8 as i8 as i32) * ((b >> (8 * i)) as u8 as i8 as i32);
    }
    s
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Format {
    Q4K,
    Q5K,
    Q6K,
    /// GGML type 41: half d, 128 sign bits (bit 1 -> +d, bit 0 -> -d).
    Q1_0,
    /// GGML type 20: half d, 32 4-bit indices into kvalues_iq4nl.
    IQ4NL,
    /// GGML type 39: E8M0 e, 32 4-bit indices into kvalues_mxfp4.
    MXFP4,
    /// GGML type 40: 4 UE4M3 sub-block scales, 64 4-bit indices into kvalues_mxfp4.
    NVFP4,
}

impl Format {
    pub fn block_bytes(self) -> usize {
        match self {
            Format::Q4K => 144,
            Format::Q5K => 176,
            Format::Q6K => 210,
            Format::Q1_0 => 18,
            Format::IQ4NL => 18,
            Format::MXFP4 => 17,
            Format::NVFP4 => 36,
        }
    }

    /// Values per block.
    pub fn block_values(self) -> usize {
        match self {
            Format::Q1_0 => 128,
            Format::IQ4NL | Format::MXFP4 => 32,
            Format::NVFP4 => 64,
            _ => 256,
        }
    }

    /// (threads per K-quant block, K-quant blocks per iteration, vdr) of the GPU kernel.
    fn geometry(self) -> (usize, usize, usize) {
        match self {
            Format::Q6K => (Q6K_THREADS_PER_BLOCK, Q6K_BLOCKS_PER_ITER, 1),
            Format::Q1_0 => (Q1_0_THREADS_PER_BLOCK, Q1_0_BLOCKS_PER_ITER, 1),
            Format::IQ4NL | Format::MXFP4 => (IQ4NL_THREADS_PER_BLOCK, IQ4NL_BLOCKS_PER_ITER, 2),
            Format::NVFP4 => (IQ4NL_THREADS_PER_BLOCK, IQ4NL_BLOCKS_PER_ITER, 4),
            _ => (THREADS_PER_BLOCK, BLOCKS_PER_ITER, 2),
        }
    }
}

/// Q6_K's per-thread unpack: (ql word, qh word >> vh_shift, bq8_offset, scale_offset).
#[inline(always)]
fn q6k_words(blk: &[u8], iqs: usize) -> (u32, i32, usize, usize) {
    let bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    let scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    let vh_shift = 2 * ((iqs % 16) / 8);
    let vl = word(blk, 4 * iqs);
    let vh = (word(blk, Q6K_QH + 4 * (8 * (iqs / 16) + iqs % 8)) as i32) >> vh_shift;
    (vl, vh, bq8_offset, scale_offset)
}

/// `(vl >> 4i) & 0x0F | ((vh >> 4i) << 4) & 0x30` per byte: the 6-bit quant, before the -32.
#[inline(always)]
fn q6k_quant(vl: u32, vh: i32, i: usize) -> u32 {
    ((vl >> (4 * i)) & 0x0F0F_0F0F) | ((((vh >> (4 * i)) << 4) as u32) & 0x3030_3030)
}

/// `tmp += vec_dot_q6_K_q8_1`, contracted to `fma(d, sumf, tmp)` as on the GPU.
fn acc_q6k(tmp: f32, blk: &[u8], xq: &[u8], iqs: usize) -> f32 {
    let (vl, vh, bq8_offset, scale_offset) = q6k_words(blk, iqs);
    let mut sumf = 0f32;
    for i in 0..2 {
        let q8 = &xq[(bq8_offset + 2 * i) * Q8_BYTES..];
        let d8 = f16_to_f32(u16::from_le_bytes([q8[0], q8[1]]));
        let u = word(q8, 4 + 4 * (iqs % 8));
        let sc = blk[Q6K_SCALES + scale_offset + 4 * i] as i8 as i32;
        // __vsubss4(q, 0x20202020): bytes in [0, 63], no saturation, no borrow past the 128 offset.
        let vi = ((q6k_quant(vl, vh, i) | 0x8080_8080).wrapping_sub(0x2020_2020)) ^ 0x8080_8080;
        sumf = d8.mul_add(dp4a(vi, u, 0).wrapping_mul(sc) as f32, sumf);
    }
    let d = f16_to_f32(u16::from_le_bytes([blk[Q6K_D], blk[Q6K_D + 1]]));
    d.mul_add(sumf, tmp)
}

fn k_scales(blk: &[u8], bq8_offset: usize) -> [u32; 4] {
    let s16 = |i: usize| -> u32 { u16::from_le_bytes([blk[4 + 2 * i], blk[4 + 2 * i + 1]]) as u32 };
    let j = bq8_offset / 2;
    let (a0, a1) = if j < 2 {
        (s16(j) & 0x3f3f, s16(j + 2) & 0x3f3f)
    } else {
        (
            (s16(j + 2) & 0x0f0f) | ((s16(j - 2) & 0xc0c0) >> 2),
            ((s16(j + 2) >> 4) & 0x0f0f) | ((s16(j) & 0xc0c0) >> 2),
        )
    };
    [a0 & 0xFF, a0 >> 8, a1 & 0xFF, a1 >> 8]
}

/// One thread's contribution for one K-quant block (`vec_dot_q{4,5}_K_q8_1`).
fn dot(fmt: Format, blk: &[u8], xq: &[u8], iqs: usize) -> f32 {
    let bq8_offset = 2 * ((iqs / 2) / 4);
    let v: [[u32; 2]; 2] = match fmt {
        Format::Q4K => {
            let q4 = 16 + 16 * bq8_offset + 4 * ((iqs / 2) % 4);
            let (a, b) = (word(blk, q4), word(blk, q4 + 16));
            [[a & 0x0F0F_0F0F, b & 0x0F0F_0F0F], [(a >> 4) & 0x0F0F_0F0F, (b >> 4) & 0x0F0F_0F0F]]
        }
        Format::Q5K => {
            let ql = 48 + 16 * bq8_offset + 4 * ((iqs / 2) % 4);
            let qh = 16 + 4 * ((iqs / 2) % 4);
            let (l0, l1) = (word(blk, ql), word(blk, ql + 16));
            let (h0, h1) = (word(blk, qh) >> bq8_offset, word(blk, qh + 16) >> bq8_offset);
            let mut v = [[0u32; 2]; 2];
            for i in 0..2 {
                v[i][0] = ((l0 >> (4 * i)) & 0x0F0F_0F0F) | (((h0 >> i) << 4) & 0x1010_1010);
                v[i][1] = ((l1 >> (4 * i)) & 0x0F0F_0F0F) | (((h1 >> i) << 4) & 0x1010_1010);
            }
            v
        }
        Format::Q6K | Format::Q1_0 | Format::IQ4NL | Format::MXFP4 | Format::NVFP4 => {
            unreachable!("Q6_K / Q1_0 / IQ4_NL / MXFP4 / NVFP4 accumulate through acc_*")
        }
    };
    let s = k_scales(blk, bq8_offset);
    let (mut sumf_d, mut sumf_m) = (0f32, 0f32);
    for i in 0..2 {
        let q8 = &xq[(bq8_offset + i) * Q8_BYTES..];
        let d8 = f16_to_f32(u16::from_le_bytes([q8[0], q8[1]]));
        let u0 = word(q8, 4 + 4 * ((iqs / 2) % 4));
        let u1 = word(q8, 4 + 4 * ((iqs / 2) % 4) + 16);
        let dot1 = dp4a(v[i][1], u1, dp4a(v[i][0], u0, 0));
        let dot2 = dp4a(0x0101_0101, u1, dp4a(0x0101_0101, u0, 0));
        sumf_d = d8.mul_add((dot1 * s[i] as i32) as f32, sumf_d);
        sumf_m = d8.mul_add((dot2 * s[2 + i] as i32) as f32, sumf_m);
    }
    let dm_d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
    let dm_m = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
    let p = sumf_m * dm_m;
    sumf_d.mul_add(dm_d, -p)
}

/// One output row of one expert: `w_row` is the row's K-quant blocks, `xq` the Q8_1 row.
pub fn row(fmt: Format, w_row: &[u8], xq: &[u8], k: usize) -> f32 {
    row_impl(fmt, w_row, xq, k, false)
}

fn row_impl(fmt: Format, w_row: &[u8], xq: &[u8], k: usize, simd: bool) -> f32 {
    let bpr = k / fmt.block_values();
    let q8_per_block = fmt.block_values() / 32;
    let bb = fmt.block_bytes();
    let (tpb, bpi, vdr) = fmt.geometry();
    let mut tmp = [0f32; THREADS];
    for (tid, t) in tmp.iter_mut().enumerate() {
        let mut kbx = tid / tpb;
        let kqs = vdr * (tid % tpb);
        while kbx < bpr {
            let (blk, x) = (&w_row[kbx * bb..(kbx + 1) * bb], &xq[kbx * q8_per_block * Q8_BYTES..]);
            *t = match fmt {
                Format::Q6K => acc_q6k(*t, blk, x, kqs),
                Format::Q1_0 => acc_q1_0(*t, blk, x, kqs, simd),
                Format::IQ4NL => acc_iq4nl(*t, blk, x, kqs),
                Format::MXFP4 => acc_mxfp4(*t, blk, x, kqs),
                Format::NVFP4 => acc_nvfp4(*t, blk, x, kqs),
                _ => *t + dot(fmt, blk, x, kqs),
            };
            kbx += bpi;
        }
    }
    if fmt == Format::MXFP4 { reduce_ftz(&tmp) } else { reduce(&tmp) }
}

/// `reduce` with the GPU's add.ftz (only MXFP4 can produce subnormal partial sums).
fn reduce_ftz(tmp: &[f32; THREADS]) -> f32 {
    let mut lanes = [0f32; 32];
    for (l, v) in lanes.iter_mut().enumerate() {
        *v = add_ftz(tmp[l], tmp[32 + l]);
        *v = add_ftz(*v, tmp[64 + l]);
        *v = add_ftz(*v, tmp[96 + l]);
    }
    let mut mask = 16;
    while mask > 0 {
        let prev = lanes;
        for l in 0..32 {
            lanes[l] = add_ftz(prev[l], prev[l ^ mask]);
        }
        mask >>= 1;
    }
    canonical(lanes[0])
}

/// The GPU's cross-thread sum: warps 1..3 into warp 0 in order, then the xor butterfly.
fn reduce(tmp: &[f32; THREADS]) -> f32 {
    let mut lanes = [0f32; 32];
    for (l, v) in lanes.iter_mut().enumerate() {
        *v = tmp[l] + tmp[32 + l];
        *v += tmp[64 + l];
        *v += tmp[96 + l];
    }
    let mut mask = 16;
    while mask > 0 {
        let prev = lanes;
        for l in 0..32 {
            lanes[l] = prev[l] + prev[l ^ mask];
        }
        mask >>= 1;
    }
    canonical(lanes[0])
}

/// `tmp += vec_dot_q1_0_q8_1`: `fma(d * d8, sumi, tmp)`, sumi = sum over the 32-value chunk `iqs`
/// of (bit ? q : -q) = 2 * (sum of q where bit) - (sum of q), exact (the GPU's dp4a against +-1).
fn acc_q1_0(tmp: f32, blk: &[u8], xq: &[u8], iqs: usize, simd: bool) -> f32 {
    let q8 = &xq[iqs * Q8_BYTES..(iqs + 1) * Q8_BYTES];
    let bits = word(blk, 2 + 4 * iqs);
    let sumi = if simd {
        // SAFETY: rows8's caller checked avx2.
        unsafe { chunk_sumi_avx2(bits, &q8[4..36]) }
    } else {
        let mut s = 0i32;
        for j in 0..32 {
            let q = q8[4 + j] as i8 as i32;
            s += if (bits >> j) & 1 != 0 { q } else { -q };
        }
        s
    };
    let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
    let d8 = f16_to_f32(u16::from_le_bytes([q8[0], q8[1]]));
    (d * d8).mul_add(sumi as f32, tmp)
}

/// `tmp += vec_dot_iq4_nl_q8_1`: `fma(d * d8, sumi, tmp)` over the 8 + 8 values of 4-byte chunks
/// `iqs` and `iqs + 1` (low nibbles pair with q8 bytes 4w.., high nibbles with 16 + 4w..), exact.
fn acc_iq4nl(tmp: f32, blk: &[u8], xq: &[u8], iqs: usize) -> f32 {
    let q8 = &xq[..Q8_BYTES];
    let mut sumi = 0i32;
    for w in iqs..iqs + 2 {
        for b in 0..4 {
            let q = blk[2 + 4 * w + b];
            sumi += KVALUES_IQ4NL[(q & 0xf) as usize] as i32 * q8[4 + 4 * w + b] as i8 as i32;
            sumi += KVALUES_IQ4NL[(q >> 4) as usize] as i32 * q8[4 + 16 + 4 * w + b] as i8 as i32;
        }
    }
    let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
    let d8 = f16_to_f32(u16::from_le_bytes([q8[0], q8[1]]));
    (d * d8).mul_add(sumi as f32, tmp)
}

/// `tmp += vec_dot_mxfp4_q8_1`: `fma.ftz(mul.ftz(mul.ftz(2^(e-127), 0.5), d8), sumi, tmp)`.
fn acc_mxfp4(tmp: f32, blk: &[u8], xq: &[u8], iqs: usize) -> f32 {
    let q8 = &xq[..Q8_BYTES];
    let mut sumi = 0i32;
    for w in iqs..iqs + 2 {
        for b in 0..4 {
            let q = blk[1 + 4 * w + b];
            sumi += KVALUES_MXFP4[(q & 0xf) as usize] as i32 * q8[4 + 4 * w + b] as i8 as i32;
            sumi += KVALUES_MXFP4[(q >> 4) as usize] as i32 * q8[4 + 16 + 4 * w + b] as i8 as i32;
        }
    }
    let d8 = f16_to_f32(u16::from_le_bytes([q8[0], q8[1]]));
    let d = mul_ftz(mul_ftz(e8m0_to_f32(blk[0]), 0.5), d8);
    fma_ftz(d, sumi as f32, tmp)
}

/// `ggml_cuda_ue4m3_to_fp32` on the GPU: the byte as a signed E4M3 (0x7F / 0xFF -> +0), halved
/// (`div.approx.ftz` by 2 is exact here). Exact.
fn e4m3_half(b: u8) -> f32 {
    if b & 0x7F == 0x7F {
        return 0.0;
    }
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 7) as f32;
    let v = if e == 0 { m * 2f32.powi(-9) } else { (1.0 + m / 8.0) * 2f32.powi(e - 7) };
    let v = if b & 0x80 != 0 { -v } else { v };
    v * 0.5
}

/// `tmp += vec_dot_nvfp4_q8_1`: for the thread's two sub-blocks `sum = fma(d_s * d8, sumi, sum)`
/// from +0, then `tmp + sum`. Sub-block `is` pairs with Q8_1 block `is / 2`, values `16 (is % 2)..`.
fn acc_nvfp4(tmp: f32, blk: &[u8], xq: &[u8], iqs: usize) -> f32 {
    let mut sum = 0f32;
    for is in [iqs / 2, iqs / 2 + 1] {
        let q8 = &xq[(is / 2) * Q8_BYTES..(is / 2 + 1) * Q8_BYTES];
        let off = 4 + 16 * (is % 2);
        let mut sumi = 0i32;
        for j in 0..8 {
            let q = blk[4 + 8 * is + j];
            sumi += KVALUES_MXFP4[(q & 0xf) as usize] as i32 * q8[off + j] as i8 as i32;
            sumi += KVALUES_MXFP4[(q >> 4) as usize] as i32 * q8[off + 8 + j] as i8 as i32;
        }
        let d8 = f16_to_f32(u16::from_le_bytes([q8[0], q8[1]]));
        sum = (e4m3_half(blk[is]) * d8).mul_add(sumi as f32, sum);
    }
    tmp + sum
}

/// `chunk_sumi` with AVX2: the bits become 0/1 bytes, maddubs (unsigned 0/1 x signed q, no pair
/// can saturate) + madd give the selected sum and the total exactly.
#[target_feature(enable = "avx2")]
unsafe fn chunk_sumi_avx2(bits: u32, q: &[u8]) -> i32 {
    use std::arch::x86_64::*;
    let byte_shuf = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    let bit_masks = _mm256_setr_epi8(1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128);
    let ones8 = _mm256_set1_epi8(1);
    let ones16 = _mm256_set1_epi16(1);
    let qy = _mm256_loadu_si256(q.as_ptr() as *const __m256i);
    let set = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(bits as i32), byte_shuf), bit_masks), bit_masks);
    let sel = _mm256_and_si256(set, ones8);
    let s = _mm256_madd_epi16(_mm256_maddubs_epi16(sel, qy), ones16);
    let t = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, qy), ones16);
    let v = _mm256_sub_epi32(_mm256_add_epi32(s, s), t);
    let h = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    let h = _mm_add_epi32(h, _mm_shuffle_epi32(h, 0b01_00_11_10));
    let h = _mm_add_epi32(h, _mm_shuffle_epi32(h, 0b10_11_00_01));
    _mm_cvtsi128_si32(h)
}

/// Eight rows of one expert at once, lane i = row i, each lane running `row`'s exact
/// operation sequence (AVX2 maddubs+madd is an exact dp4a for these operands, vfmadd
/// is the same correctly rounded fma), so every lane is bit-identical to `row`.
/// SAFETY: caller checked avx2 and fma; `w_rows` are 8 row slices of `k/256` blocks each.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn rows8(fmt: Format, w_rows: &[&[u8]; 8], xq: &[u8], k: usize) -> [f32; 8] {
    use std::arch::x86_64::*;
    let bpr = k / 256;
    let bb = fmt.block_bytes();
    let ones_b = _mm256_set1_epi8(1);
    let ones_w = _mm256_set1_epi16(1);
    let dp4a_v = |a: __m256i, b: __m256i| -> __m256i { _mm256_madd_epi16(_mm256_maddubs_epi16(a, b), ones_w) };
    let lanes_u32 = |f: &dyn Fn(usize) -> u32| -> __m256i {
        _mm256_setr_epi32(f(0) as i32, f(1) as i32, f(2) as i32, f(3) as i32, f(4) as i32, f(5) as i32, f(6) as i32, f(7) as i32)
    };
    if matches!(fmt, Format::Q1_0 | Format::IQ4NL | Format::MXFP4 | Format::NVFP4) {
        // The float work is one fma per 32 weights; the AVX2 part is Q1_0's integer chunk sum.
        return std::array::from_fn(|r| row_impl(fmt, w_rows[r], xq, k, true));
    }
    let mut tmp = [_mm256_setzero_ps(); THREADS];
    if fmt == Format::Q6K {
        for (tid, t) in tmp.iter_mut().enumerate() {
            let mut kbx = tid / Q6K_THREADS_PER_BLOCK;
            let iqs = tid % Q6K_THREADS_PER_BLOCK;
            while kbx < bpr {
                let blk = |r: usize| &w_rows[r][kbx * bb..(kbx + 1) * bb];
                let words: [(u32, i32, usize, usize); 8] = std::array::from_fn(|r| q6k_words(blk(r), iqs));
                let (bq8_offset, scale_offset) = (words[0].2, words[0].3);
                let mut sumf = _mm256_setzero_ps();
                for i in 0..2 {
                    let q8 = &xq[kbx * 8 * Q8_BYTES + (bq8_offset + 2 * i) * Q8_BYTES..];
                    let d8 = _mm256_set1_ps(f16_to_f32(u16::from_le_bytes([q8[0], q8[1]])));
                    let u = word(q8, 4 + 4 * (iqs % 8));
                    // dp4a(q - 32, u) = dp4a(q, u) - 32 * sum(u): q in [0, 63] is maddubs's unsigned
                    // operand and no pair sum saturates (2 * 63 * 128 < 2^15), so this is exact.
                    let usum = dp4a(0x0101_0101, u, 0);
                    let q = lanes_u32(&|r| q6k_quant(words[r].0, words[r].1, i));
                    let dot = _mm256_sub_epi32(dp4a_v(q, _mm256_set1_epi32(u as i32)), _mm256_set1_epi32(32 * usum));
                    let sc = lanes_u32(&|r| blk(r)[Q6K_SCALES + scale_offset + 4 * i] as i8 as i32 as u32);
                    let x = _mm256_cvtepi32_ps(_mm256_mullo_epi32(dot, sc));
                    sumf = _mm256_fmadd_ps(d8, x, sumf);
                }
                let d: [f32; 8] = std::array::from_fn(|r| f16_to_f32(u16::from_le_bytes([blk(r)[Q6K_D], blk(r)[Q6K_D + 1]])));
                *t = _mm256_fmadd_ps(_mm256_loadu_ps(d.as_ptr()), sumf, *t);
                kbx += Q6K_BLOCKS_PER_ITER;
            }
        }
        return reduce8(&tmp);
    }
    for (tid, t) in tmp.iter_mut().enumerate() {
        let mut kbx = tid / THREADS_PER_BLOCK;
        let iqs = 2 * (tid % THREADS_PER_BLOCK);
        let bq8_offset = 2 * ((iqs / 2) / 4);
        while kbx < bpr {
            let blk = |r: usize| &w_rows[r][kbx * bb..(kbx + 1) * bb];
            let scales: [[u32; 4]; 8] = std::array::from_fn(|r| k_scales(blk(r), bq8_offset));
            let (v0, v1): ([u32; 8], [u32; 8]) = match fmt {
                Format::Q4K => {
                    let q4 = 16 + 16 * bq8_offset + 4 * ((iqs / 2) % 4);
                    (std::array::from_fn(|r| word(blk(r), q4)), std::array::from_fn(|r| word(blk(r), q4 + 16)))
                }
                Format::Q5K => {
                    let ql = 48 + 16 * bq8_offset + 4 * ((iqs / 2) % 4);
                    (std::array::from_fn(|r| word(blk(r), ql)), std::array::from_fn(|r| word(blk(r), ql + 16)))
                }
                Format::Q6K | Format::Q1_0 | Format::IQ4NL | Format::MXFP4 | Format::NVFP4 => unreachable!(),
            };
            let (h0, h1): ([u32; 8], [u32; 8]) = match fmt {
                Format::Q4K | Format::Q6K | Format::Q1_0 | Format::IQ4NL | Format::MXFP4 | Format::NVFP4 => ([0; 8], [0; 8]),
                Format::Q5K => {
                    let qh = 16 + 4 * ((iqs / 2) % 4);
                    (std::array::from_fn(|r| word(blk(r), qh) >> bq8_offset), std::array::from_fn(|r| word(blk(r), qh + 16) >> bq8_offset))
                }
            };
            let mut sumf_d = _mm256_setzero_ps();
            let mut sumf_m = _mm256_setzero_ps();
            for i in 0..2 {
                let q8 = &xq[kbx * 8 * Q8_BYTES + (bq8_offset + i) * Q8_BYTES..];
                let d8 = _mm256_set1_ps(f16_to_f32(u16::from_le_bytes([q8[0], q8[1]])));
                let u0 = _mm256_set1_epi32(word(q8, 4 + 4 * ((iqs / 2) % 4)) as i32);
                let u1 = _mm256_set1_epi32(word(q8, 4 + 4 * ((iqs / 2) % 4) + 16) as i32);
                let vi0 = lanes_u32(&|r| ((v0[r] >> (4 * i)) & 0x0F0F_0F0F) | (((h0[r] >> i) << 4) & 0x1010_1010));
                let vi1 = lanes_u32(&|r| ((v1[r] >> (4 * i)) & 0x0F0F_0F0F) | (((h1[r] >> i) << 4) & 0x1010_1010));
                let dot1 = _mm256_add_epi32(dp4a_v(vi1, u1), dp4a_v(vi0, u0));
                let dot2 = _mm256_add_epi32(dp4a_v(ones_b, u1), dp4a_v(ones_b, u0));
                let sc = lanes_u32(&|r| scales[r][i]);
                let mm = lanes_u32(&|r| scales[r][2 + i]);
                let x_d = _mm256_cvtepi32_ps(_mm256_mullo_epi32(dot1, sc));
                let x_m = _mm256_cvtepi32_ps(_mm256_mullo_epi32(dot2, mm));
                sumf_d = _mm256_fmadd_ps(d8, x_d, sumf_d);
                sumf_m = _mm256_fmadd_ps(d8, x_m, sumf_m);
            }
            let dm_d = _mm256_setr_ps(
                f16_to_f32(u16::from_le_bytes([blk(0)[0], blk(0)[1]])), f16_to_f32(u16::from_le_bytes([blk(1)[0], blk(1)[1]])),
                f16_to_f32(u16::from_le_bytes([blk(2)[0], blk(2)[1]])), f16_to_f32(u16::from_le_bytes([blk(3)[0], blk(3)[1]])),
                f16_to_f32(u16::from_le_bytes([blk(4)[0], blk(4)[1]])), f16_to_f32(u16::from_le_bytes([blk(5)[0], blk(5)[1]])),
                f16_to_f32(u16::from_le_bytes([blk(6)[0], blk(6)[1]])), f16_to_f32(u16::from_le_bytes([blk(7)[0], blk(7)[1]])));
            let dm_m = _mm256_setr_ps(
                f16_to_f32(u16::from_le_bytes([blk(0)[2], blk(0)[3]])), f16_to_f32(u16::from_le_bytes([blk(1)[2], blk(1)[3]])),
                f16_to_f32(u16::from_le_bytes([blk(2)[2], blk(2)[3]])), f16_to_f32(u16::from_le_bytes([blk(3)[2], blk(3)[3]])),
                f16_to_f32(u16::from_le_bytes([blk(4)[2], blk(4)[3]])), f16_to_f32(u16::from_le_bytes([blk(5)[2], blk(5)[3]])),
                f16_to_f32(u16::from_le_bytes([blk(6)[2], blk(6)[3]])), f16_to_f32(u16::from_le_bytes([blk(7)[2], blk(7)[3]])));
            let p = _mm256_mul_ps(sumf_m, dm_m);
            // -p by sign flip, exactly as the scalar path's negation (0 - p would turn -0 into +0).
            let r = _mm256_fmadd_ps(sumf_d, dm_d, _mm256_xor_ps(p, _mm256_set1_ps(-0.0)));
            *t = _mm256_add_ps(*t, r);
            kbx += BLOCKS_PER_ITER;
        }
    }
    reduce8(&tmp)
}

/// `row` for every row of `w_rows` (Q4_K / Q5_K / Q6_K), vectorised across the GPU threads instead of
/// across rows: lane i of accumulator v is thread 8v + i, which runs `row`'s exact per-thread operation
/// sequence (AVX2 maddubs + madd is an exact dp4a for these operands, vfmadd the same correctly rounded
/// fma), then `reduce`'s order. The weight words of one K-quant block are contiguous across the
/// threads that read them, so they load as vectors instead of per-lane gathers, and the Q8_1 side
/// (per-block scales, byte sums) is prepared once for all rows.
/// SAFETY: caller checked avx2 and fma; every row holds `k / 256` blocks; `out.len() == w_rows.len()`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn rows_lanes(fmt: Format, w_rows: &[&[u8]], xq: &[u8], k: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let bpr = k / 256;
    let bb = fmt.block_bytes();
    let ones_b = _mm256_set1_epi8(1);
    let ones_w = _mm256_set1_epi16(1);
    let dp4a_v = |a: __m256i, b: __m256i| -> __m256i { _mm256_madd_epi16(_mm256_maddubs_epi16(a, b), ones_w) };
    let ld = |p: &[u8], o: usize| -> __m256i { _mm256_loadu_si256(p[o..o + 32].as_ptr() as *const __m256i) };
    // lanes 0-3 from byte offset `lo`, lanes 4-7 from `hi` (16 bytes each)
    let ld2 = |p: &[u8], lo: usize, hi: usize| -> __m256i {
        _mm256_loadu2_m128i(p[hi..hi + 16].as_ptr() as *const __m128i, p[lo..lo + 16].as_ptr() as *const __m128i)
    };
    let nq8 = k / 32;
    let d8: Vec<f32> = (0..nq8).map(|b| f16_to_f32(u16::from_le_bytes([xq[b * Q8_BYTES], xq[b * Q8_BYTES + 1]]))).collect();
    let neg0 = _mm256_set1_ps(-0.0);
    if fmt == Format::Q6K {
        // per Q8_1 block: its 8 words and 32 * (byte sum of each word) (dp4a(q - 32, u) = dp4a(q, u) - 32 * sum(u))
        let u: Vec<__m256i> = (0..nq8).map(|b| ld(xq, b * Q8_BYTES + 4)).collect();
        let usum32: Vec<__m256i> = u.iter().map(|&v| _mm256_slli_epi32(dp4a_v(ones_b, v), 5)).collect();
        for (row, o) in w_rows.iter().zip(out.iter_mut()) {
            let mut acc = [_mm256_setzero_ps(); 16];
            for kbx in 0..bpr {
                let blk = &row[kbx * bb..(kbx + 1) * bb];
                let d = _mm256_set1_ps(f16_to_f32(u16::from_le_bytes([blk[Q6K_D], blk[Q6K_D + 1]])));
                for q in 0..4 {
                    // threads iqs = 8q + lane: bq8_offset 4(q/2) + q%2, scale_offset 8(q/2) + 2(q%2) + lane/4
                    let bq8 = kbx * 8 + 4 * (q / 2) + q % 2;
                    let so = Q6K_SCALES + 8 * (q / 2) + 2 * (q % 2);
                    let vl = ld(blk, 32 * q);
                    let vh = _mm256_srai_epi32::<0>(ld(blk, Q6K_QH + 32 * (q / 2)));
                    let vh = if q % 2 == 1 { _mm256_srai_epi32::<2>(vh) } else { vh };
                    let mut sumf = _mm256_setzero_ps();
                    for i in 0..2 {
                        let b = bq8 + 2 * i;
                        let (vl_i, vh_i) = if i == 0 { (vl, vh) } else { (_mm256_srli_epi32::<4>(vl), _mm256_srai_epi32::<4>(vh)) };
                        let quant = _mm256_or_si256(
                            _mm256_and_si256(vl_i, _mm256_set1_epi32(0x0F0F_0F0F)),
                            _mm256_and_si256(_mm256_slli_epi32::<4>(vh_i), _mm256_set1_epi32(0x3030_3030)),
                        );
                        let dot = _mm256_sub_epi32(dp4a_v(quant, u[b]), usum32[b]);
                        let (s0, s1) = (blk[so + 4 * i] as i8 as i32, blk[so + 1 + 4 * i] as i8 as i32);
                        let sc = _mm256_setr_epi32(s0, s0, s0, s0, s1, s1, s1, s1);
                        let x = _mm256_cvtepi32_ps(_mm256_mullo_epi32(dot, sc));
                        sumf = _mm256_fmadd_ps(_mm256_set1_ps(d8[b]), x, sumf);
                    }
                    let v = 4 * (kbx % Q6K_BLOCKS_PER_ITER) + q;
                    acc[v] = _mm256_fmadd_ps(d, sumf, acc[v]);
                }
            }
            *o = reduce_lanes(&acc);
        }
        return;
    }
    // Q4_K / Q5_K: threads j = 0..15 of a block, half h = j / 8, group g = j / 4 (bq8_offset 2g).
    // Per (block, half, i): the Q8_1 words u0 / u1 of each lane, the byte sums (dot2) and d8.
    let n = bpr * 4;
    let mut u0 = Vec::with_capacity(n);
    let mut u1 = Vec::with_capacity(n);
    let mut dot2 = Vec::with_capacity(n);
    let mut d8v = Vec::with_capacity(n);
    for kbx in 0..bpr {
        for h in 0..2 {
            for i in 0..2 {
                let (b_lo, b_hi) = (kbx * 8 + 4 * h + i, kbx * 8 + 4 * h + 2 + i);
                let a = ld2(xq, b_lo * Q8_BYTES + 4, b_hi * Q8_BYTES + 4);
                let c = ld2(xq, b_lo * Q8_BYTES + 20, b_hi * Q8_BYTES + 20);
                dot2.push(_mm256_add_epi32(dp4a_v(ones_b, c), dp4a_v(ones_b, a)));
                u0.push(a);
                u1.push(c);
                let (dl, dh) = (d8[b_lo], d8[b_hi]);
                d8v.push(_mm256_setr_ps(dl, dl, dl, dl, dh, dh, dh, dh));
            }
        }
    }
    let q5 = fmt == Format::Q5K;
    let (lo_off, shift_h) = if q5 { (48, [_mm256_setr_epi32(0, 0, 0, 0, 2, 2, 2, 2), _mm256_setr_epi32(4, 4, 4, 4, 6, 6, 6, 6)]) } else { (16, [_mm256_setzero_si256(); 2]) };
    for (row, o) in w_rows.iter().zip(out.iter_mut()) {
        let mut acc = [_mm256_setzero_ps(); 16];
        for kbx in 0..bpr {
            let blk = &row[kbx * bb..(kbx + 1) * bb];
            let dm_d = _mm256_set1_ps(f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])));
            let dm_m = _mm256_set1_ps(f16_to_f32(u16::from_le_bytes([blk[2], blk[3]])));
            let s: [[u32; 4]; 4] = std::array::from_fn(|g| k_scales(blk, 2 * g));
            let (qh0, qh1) = if q5 { (_mm256_broadcastsi128_si256(_mm_loadu_si128(blk[16..32].as_ptr() as *const __m128i)), _mm256_broadcastsi128_si256(_mm_loadu_si128(blk[32..48].as_ptr() as *const __m128i))) } else { (_mm256_setzero_si256(), _mm256_setzero_si256()) };
            for h in 0..2 {
                let (g0, g1) = (2 * h, 2 * h + 1);
                let v0 = ld2(blk, lo_off + 32 * g0, lo_off + 32 * g1);
                let v1 = ld2(blk, lo_off + 16 + 32 * g0, lo_off + 16 + 32 * g1);
                let (h0, h1) = (_mm256_srlv_epi32(qh0, shift_h[h]), _mm256_srlv_epi32(qh1, shift_h[h]));
                let mut sumf_d = _mm256_setzero_ps();
                let mut sumf_m = _mm256_setzero_ps();
                for i in 0..2 {
                    let idx = (kbx * 2 + h) * 2 + i;
                    let (mut vi0, mut vi1) = if i == 0 { (v0, v1) } else { (_mm256_srli_epi32::<4>(v0), _mm256_srli_epi32::<4>(v1)) };
                    vi0 = _mm256_and_si256(vi0, _mm256_set1_epi32(0x0F0F_0F0F));
                    vi1 = _mm256_and_si256(vi1, _mm256_set1_epi32(0x0F0F_0F0F));
                    if q5 {
                        let (hh0, hh1) = if i == 0 { (h0, h1) } else { (_mm256_srli_epi32::<1>(h0), _mm256_srli_epi32::<1>(h1)) };
                        vi0 = _mm256_or_si256(vi0, _mm256_and_si256(_mm256_slli_epi32::<4>(hh0), _mm256_set1_epi32(0x1010_1010)));
                        vi1 = _mm256_or_si256(vi1, _mm256_and_si256(_mm256_slli_epi32::<4>(hh1), _mm256_set1_epi32(0x1010_1010)));
                    }
                    let dot1 = _mm256_add_epi32(dp4a_v(vi1, u1[idx]), dp4a_v(vi0, u0[idx]));
                    let (a, b) = (s[g0][i] as i32, s[g1][i] as i32);
                    let sc = _mm256_setr_epi32(a, a, a, a, b, b, b, b);
                    let (a, b) = (s[g0][2 + i] as i32, s[g1][2 + i] as i32);
                    let mm = _mm256_setr_epi32(a, a, a, a, b, b, b, b);
                    let x_d = _mm256_cvtepi32_ps(_mm256_mullo_epi32(dot1, sc));
                    let x_m = _mm256_cvtepi32_ps(_mm256_mullo_epi32(dot2[idx], mm));
                    sumf_d = _mm256_fmadd_ps(d8v[idx], x_d, sumf_d);
                    sumf_m = _mm256_fmadd_ps(d8v[idx], x_m, sumf_m);
                }
                let p = _mm256_mul_ps(sumf_m, dm_m);
                let r = _mm256_fmadd_ps(sumf_d, dm_d, _mm256_xor_ps(p, neg0));
                let v = 2 * (kbx % BLOCKS_PER_ITER) + h;
                acc[v] = _mm256_add_ps(acc[v], r);
            }
        }
        *o = reduce_lanes(&acc);
    }
}

/// `reduce` on the 128 thread partials of `rows_lanes` (thread 8v + i in lane i of `acc[v]`).
#[target_feature(enable = "avx2,fma")]
unsafe fn reduce_lanes(acc: &[std::arch::x86_64::__m256; 16]) -> f32 {
    use std::arch::x86_64::*;
    // lane l = 8g + i of the warp: tmp[l] + tmp[32 + l] + tmp[64 + l] + tmp[96 + l]
    let l: [__m256; 4] = std::array::from_fn(|g| _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(acc[g], acc[4 + g]), acc[8 + g]), acc[12 + g]));
    // xor 16 (g ^ 2), xor 8 (g ^ 1): every group then holds the same sums
    let m0 = _mm256_add_ps(l[0], l[2]);
    let m1 = _mm256_add_ps(l[1], l[3]);
    let x = _mm256_add_ps(m0, m1);
    let x = _mm256_add_ps(x, _mm256_permute2f128_ps::<1>(x, x));
    let x = _mm256_add_ps(x, _mm256_permute_ps::<0b01_00_11_10>(x));
    let x = _mm256_add_ps(x, _mm256_permute_ps::<0b10_11_00_01>(x));
    canonical(_mm256_cvtss_f32(x))
}

/// `row`'s cross-thread sum on 8 rows at once: warps 1..3 into warp 0 in order, then the butterfly.
#[target_feature(enable = "avx2,fma")]
unsafe fn reduce8(tmp: &[std::arch::x86_64::__m256; THREADS]) -> [f32; 8] {
    use std::arch::x86_64::*;
    let mut lanes = [_mm256_setzero_ps(); 32];
    for (l, v) in lanes.iter_mut().enumerate() {
        *v = _mm256_add_ps(tmp[l], tmp[32 + l]);
        *v = _mm256_add_ps(*v, tmp[64 + l]);
        *v = _mm256_add_ps(*v, tmp[96 + l]);
    }
    let mut mask = 16;
    while mask > 0 {
        let prev = lanes;
        for l in 0..32 {
            lanes[l] = _mm256_add_ps(prev[l], prev[l ^ mask]);
        }
        mask >>= 1;
    }
    let mut out = [0f32; 8];
    _mm256_storeu_ps(out.as_mut_ptr(), lanes[0]);
    out.map(canonical)
}
