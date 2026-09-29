//! GGML NVFP4 (type 40): 64 weights per block in 4 sub-blocks of 16, one UE4M3 scale per
//! sub-block and 64 FP4 (E2M1) values as 4-bit indices into `kvalues_mxfp4` (the doubled E2M1
//! values; `ggml_ue4m3_to_fp32` returns the scale halved). Bit-identical ports of llama.cpp
//! (acecd56): `dequantize_row_nvfp4`, `quantize_row_nvfp4_ref` (what the CPU `from_float` calls),
//! `ggml_vec_dot_nvfp4_q8_0` (AVX2 path) and `ggml_vec_dot_nvfp4_q8_0_generic` (gcc -O3
//! -march=native contracts `sumf += dy * d * sumi` into one fma).
//!
//! llama.cpp dots NVFP4 against two consecutive Q8_0 blocks; the vec-dot type is [`BlockQ8_0x2`],
//! byte-identical to that Q8_0 row. The CPU reads a UE4M3 byte ignoring its top bit, the CUDA
//! program as a signed E4M3 (both map 0x7F to 0): llama.cpp's CPU and GPU differ for bytes >= 0x80,
//! which the quantizer never writes.

use super::k_quants::{BlockQ8_0, GgmlType};
use super::GgmlDType;
use half::f16;

pub const QK_NVFP4: usize = 64;
pub const QK_NVFP4_SUB: usize = 16;

use super::mxfp4::KVALUES_MXFP4;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockNVFP4 {
    pub(crate) d: [u8; QK_NVFP4 / QK_NVFP4_SUB],
    pub(crate) qs: [u8; QK_NVFP4 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockNVFP4>() == 36);

/// Two Q8_0 blocks (64 values): the vec-dot partner of [`BlockNVFP4`].
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8_0x2 {
    pub(crate) b: [BlockQ8_0; 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ8_0x2>() == 2 * 34);

fn as_q8_0(ys: &[BlockQ8_0x2]) -> &[BlockQ8_0] {
    unsafe { std::slice::from_raw_parts(ys.as_ptr() as *const BlockQ8_0, ys.len() * 2) }
}

fn as_q8_0_mut(ys: &mut [BlockQ8_0x2]) -> &mut [BlockQ8_0] {
    unsafe { std::slice::from_raw_parts_mut(ys.as_mut_ptr() as *mut BlockQ8_0, ys.len() * 2) }
}

impl GgmlType for BlockQ8_0x2 {
    // Only ever used as a vec-dot scratch type; its bytes are a Q8_0 row.
    const DTYPE: GgmlDType = GgmlDType::Q8_0;
    const BLCK_SIZE: usize = QK_NVFP4;
    type VecDotType = BlockQ8_0x2;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        BlockQ8_0::to_float(as_q8_0(xs), ys)
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        BlockQ8_0::from_float(xs, as_q8_0_mut(ys))
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        BlockQ8_0::vec_dot(n, as_q8_0(xs), as_q8_0(ys))
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        BlockQ8_0::vec_dot_unopt(n, as_q8_0(xs), as_q8_0(ys))
    }
}

/// ggml-impl.h `ggml_ue4m3_to_fp32` (exact: every value is m * 2^k): the UE4M3 value halved,
/// 0 and 0x7F map to 0, bit 7 is ignored.
pub fn ue4m3_to_fp32(x: u8) -> f32 {
    if x == 0 || x == 0x7F {
        return 0.0;
    }
    let exp = ((x >> 3) & 0xF) as i32;
    let man = (x & 0x7) as f32;
    let raw = if exp == 0 { man * 2f32.powi(-9) } else { (1.0 + man / 8.0) * 2f32.powi(exp - 7) };
    raw * 0.5
}

/// ggml-impl.h `ggml_fp32_to_ue4m3`.
fn fp32_to_ue4m3(x: f32) -> u8 {
    if !(x > 0.0) {
        return 0;
    }
    let x = if x > 448.0 { 448.0 } else { x };
    let bits = x.to_bits();
    let fp32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let fp32_man = ((bits >> 20) & 0x7) as i32;
    let mut ue4m3_exp = fp32_exp + 7;
    if ue4m3_exp <= 0 {
        // subnormal: value = man * 2^-9, man = round(x * 2^9)
        let mut man = (x * 512.0 + 0.5) as i32;
        if man > 7 {
            man = 7;
        }
        if man < 1 {
            return 0;
        }
        return man as u8;
    }
    if ue4m3_exp >= 15 {
        return 0x7E;
    }
    let round_bit = ((bits >> 19) & 1) as i32;
    let mut ue4m3_man = fp32_man + round_bit;
    if ue4m3_man > 7 {
        ue4m3_man = 0;
        ue4m3_exp += 1;
        if ue4m3_exp >= 15 {
            return 0x7E;
        }
    }
    ((ue4m3_exp << 3) | ue4m3_man) as u8
}

/// ggml-quants.c `best_index_mxfp4`.
fn best_index_mxfp4(x: f32, e: f32) -> u8 {
    let mut best_index = 0u8;
    let mut best_err = (KVALUES_MXFP4[0] as f32 * e - x).abs();
    for (i, &kv) in KVALUES_MXFP4.iter().enumerate().skip(1) {
        let err = (kv as f32 * e - x).abs();
        if err < best_err {
            best_index = i as u8;
            best_err = err;
        }
    }
    best_index
}

/// `sumi_lo + sumi_hi` of sub-block `s` against its 16 Q8_0 values (exact).
#[inline(always)]
fn sub_sumi(x: &BlockNVFP4, y: &[BlockQ8_0], s: usize) -> i32 {
    let yb = &y[s / 2];
    let off = (s % 2) * QK_NVFP4_SUB;
    let (mut lo, mut hi) = (0i32, 0i32);
    for j in 0..QK_NVFP4_SUB / 2 {
        let qv = x.qs[s * (QK_NVFP4_SUB / 2) + j];
        lo += yb.qs[off + j] as i32 * KVALUES_MXFP4[(qv & 0xf) as usize] as i32;
        hi += yb.qs[off + j + QK_NVFP4_SUB / 2] as i32 * KVALUES_MXFP4[(qv >> 4) as usize] as i32;
    }
    lo + hi
}

impl GgmlType for BlockNVFP4 {
    const DTYPE: GgmlDType = GgmlDType::NVFP4;
    const BLCK_SIZE: usize = QK_NVFP4;
    type VecDotType = BlockQ8_0x2;

    // dequantize_row_nvfp4
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(k.is_multiple_of(QK_NVFP4), "dequantize_row_nvfp4: {k} % {QK_NVFP4}");
        for (x, ys) in xs.iter().zip(ys.chunks_exact_mut(QK_NVFP4)) {
            for s in 0..4 {
                let d = ue4m3_to_fp32(x.d[s]);
                let yb = &mut ys[s * QK_NVFP4_SUB..(s + 1) * QK_NVFP4_SUB];
                for j in 0..QK_NVFP4_SUB / 2 {
                    let q = x.qs[s * (QK_NVFP4_SUB / 2) + j];
                    yb[j] = KVALUES_MXFP4[(q & 0x0f) as usize] as f32 * d;
                    yb[j + QK_NVFP4_SUB / 2] = KVALUES_MXFP4[(q >> 4) as usize] as f32 * d;
                }
            }
        }
    }

    // quantize_row_nvfp4_ref
    fn from_float(xs: &[f32], ys: &mut [Self]) {
        let k = xs.len();
        debug_assert!(k.is_multiple_of(QK_NVFP4), "quantize_row_nvfp4: {k} % {QK_NVFP4}");
        debug_assert_eq!(ys.len(), k / QK_NVFP4);
        for (y, x) in ys.iter_mut().zip(xs.chunks_exact(QK_NVFP4)) {
            for s in 0..4 {
                let xb = &x[s * QK_NVFP4_SUB..(s + 1) * QK_NVFP4_SUB];
                let mut amax = 0f32;
                for &v in xb {
                    if amax < v.abs() {
                        amax = v.abs();
                    }
                }
                let ue = fp32_to_ue4m3(amax / 6.0);
                y.d[s] = ue;
                let d = ue4m3_to_fp32(ue);
                for j in 0..QK_NVFP4_SUB / 2 {
                    let x0 = best_index_mxfp4(xb[j], d);
                    let x1 = best_index_mxfp4(xb[QK_NVFP4_SUB / 2 + j], d);
                    y.qs[s * (QK_NVFP4_SUB / 2) + j] = x0 | (x1 << 4);
                }
            }
        }
    }

    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        #[cfg(target_feature = "avx2")]
        return unsafe { vec_dot_nvfp4_q8_0_avx2(n, xs, as_q8_0(ys)) };

        Self::vec_dot_unopt(n, xs, ys)
    }

    // ggml_vec_dot_nvfp4_q8_0_generic
    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(n.is_multiple_of(QK_NVFP4), "vec_dot_nvfp4_q8_0: {n} % {QK_NVFP4}");
        let nb = n / QK_NVFP4;
        let y = as_q8_0(ys);
        let mut sumf = 0f32;
        for (ib, x) in xs.iter().enumerate().take(nb) {
            for s in 0..4 {
                let d = ue4m3_to_fp32(x.d[s]);
                let dy = y[2 * ib + s / 2].d.to_f32();
                sumf = (dy * d).mul_add(sub_sumi(x, &y[2 * ib..], s) as f32, sumf);
            }
        }
        sumf
    }
}

// ggml_vec_dot_nvfp4_q8_0, `#if defined(__AVX2__)` branch.
#[cfg(target_feature = "avx2")]
#[inline(always)]
unsafe fn vec_dot_nvfp4_q8_0_avx2(n: usize, xs: &[BlockNVFP4], ys: &[BlockQ8_0]) -> f32 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    debug_assert!(n.is_multiple_of(QK_NVFP4), "vec_dot_nvfp4_q8_0: {n} % {QK_NVFP4}");
    let nb = n / QK_NVFP4;
    debug_assert!(xs.len() >= nb && ys.len() >= 2 * nb);

    let values128 = _mm_loadu_si128(KVALUES_MXFP4.as_ptr() as *const __m128i);
    let m4b = _mm_set1_epi8(0x0f);
    let mone = _mm256_set1_epi16(1);
    // mul_add_epi8: maddubs(sign(x, x), sign(y, x))
    let mul_add = |x: __m256i, y: __m256i| _mm256_maddubs_epi16(_mm256_sign_epi8(x, x), _mm256_sign_epi8(y, x));
    let mut accum = _mm256_setzero_ps();
    for ib in 0..nb {
        let x = &xs[ib];
        let q4bits_01 = _mm_loadu_si128(x.qs.as_ptr() as *const __m128i);
        let q4bits_23 = _mm_loadu_si128(x.qs.as_ptr().add(16) as *const __m128i);
        let q8_01 = _mm256_loadu_si256(ys[2 * ib].qs.as_ptr() as *const __m256i);
        let q8_23 = _mm256_loadu_si256(ys[2 * ib + 1].qs.as_ptr() as *const __m256i);
        let q4_01_lo = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_01, m4b));
        let q4_01_hi = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_01, 4), m4b));
        let q4_23_lo = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_23, m4b));
        let q4_23_hi = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_23, 4), m4b));
        let q4_01 = _mm256_set_m128i(_mm_unpackhi_epi64(q4_01_lo, q4_01_hi), _mm_unpacklo_epi64(q4_01_lo, q4_01_hi));
        let q4_23 = _mm256_set_m128i(_mm_unpackhi_epi64(q4_23_lo, q4_23_hi), _mm_unpacklo_epi64(q4_23_lo, q4_23_hi));
        let p_1 = _mm256_madd_epi16(mul_add(q4_01, q8_01), mone);
        let p_2 = _mm256_madd_epi16(mul_add(q4_23, q8_23), mone);
        let dy0 = ys[2 * ib].d.to_f32();
        let dy1 = ys[2 * ib + 1].d.to_f32();
        let s0 = ue4m3_to_fp32(x.d[0]) * dy0;
        let s1 = ue4m3_to_fp32(x.d[1]) * dy0;
        let s2 = ue4m3_to_fp32(x.d[2]) * dy1;
        let s3 = ue4m3_to_fp32(x.d[3]) * dy1;
        let scales01 = _mm256_set_m128(_mm_set1_ps(s1), _mm_set1_ps(s0));
        let scales23 = _mm256_set_m128(_mm_set1_ps(s3), _mm_set1_ps(s2));
        accum = _mm256_fmadd_ps(scales01, _mm256_cvtepi32_ps(p_1), accum);
        accum = _mm256_fmadd_ps(scales23, _mm256_cvtepi32_ps(p_2), accum);
    }
    super::avx::hsum_float_8(accum)
}

#[cfg(test)]
mod tests {
    //! Oracle: llama.cpp acecd56 itself (oxide-kernels/nvfp4/ref/cpu_ref.c, linked against
    //! libggml-base/libggml-cpu built with -O3 -march=native). Same seeded inputs as below.
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn f16(&mut self, i: usize, rate: u64) -> f16 {
            const SP: [u16; 8] = [0x0000, 0x8000, 0x7C00, 0xFC00, 0x7E00, 0x0001, 0x83FF, 0x7BFF];
            if self.next() % rate == 0 {
                return f16::from_bits(SP[i % 8]);
            }
            let v = (0x2000 + self.next() % 0x2800) as u16;
            f16::from_bits(v | (((self.next() & 1) as u16) << 15))
        }
    }

    fn fnv(b: &[u8]) -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        for &x in b {
            h ^= x as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    fn bytes<T>(v: &[T]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
    }

    const NB1: usize = 512;
    const NQ: usize = 96;
    const DEQUANT_FNV: u64 = 0x8de1ccda2eae37cd;
    const QUANT_FNV: u64 = 0x502d07aef420a841;
    /// (AVX2 vec_dot, generic vec_dot) result bits.
    const VECDOT: [(u32, u32); 400] = [
        (0x4447f75c, 0x4447f75c), (0x4a0d01fc, 0x4a0e46f1), (0x4a7a272f, 0x4a757004), (0xc9ba6ef4, 0xc9ba75af), (0x49de9168, 0x49de90b8), (0xc9613d60, 0xc9613d5f),
        (0x47a346ea, 0x477a8d3e), (0xc90a4895, 0xc90a4895), (0x491b560e, 0x491b5615), (0x4937248b, 0x4937248c), (0x48ba6510, 0x48c8f282), (0xc915e67d, 0xc913c160),
        (0xc7588488, 0xc758845d), (0xc8936d4c, 0xc89a268d), (0x4900bb06, 0x490078c9), (0x48a38d0c, 0x48b2a885), (0xca0415f4, 0xca0415e7), (0x44ca8e6d, 0x44ca8e6e),
        (0xc935f9c4, 0xc9693ce2), (0x48d0d6ab, 0x48d0d89b), (0x48469149, 0x4846914b), (0xca3d2376, 0xca3d2377), (0xcdf8c14b, 0xcdf8c14b), (0x46ea375e, 0x46ea375e),
        (0x48d5d4d0, 0x48d5d009), (0x47b2fa62, 0x47b2fa68), (0x4986e262, 0x49841d5e), (0x49690342, 0x4968997e), (0x4988dbde, 0x4988c193), (0xc8d4c1e8, 0xc8b28df3),
        (0x49e2b65e, 0x49d987ea), (0x47c9244a, 0x47c9244e), (0xc8318f30, 0xc839c902), (0x489e000e, 0x48978f0a), (0xffc00000, 0xffc00000), (0xc92ad1c0, 0xc92ad1be),
        (0xc8f9222a, 0xc8f98b45), (0x49a190de, 0x49a1b44d), (0xc6a120c6, 0xc6a120c4), (0xc7e78820, 0xc7e6a2e8), (0x48b2c25a, 0x48b2beb4), (0x4996e34d, 0x4996e2f9),
        (0xc8acea64, 0xc8acea60), (0x47a8bc7b, 0x47a60aaa), (0xc815e3cf, 0xc7df1791), (0x488f4840, 0x488f4840), (0x4956139a, 0x4956083e), (0x49a68d9e, 0x49a68e1f),
        (0x494beeea, 0x4944cb7b), (0x49a18bd6, 0x49a1813b), (0x499e9880, 0x499e9881), (0x47f5ab08, 0x47f5ab07), (0xc6f5c392, 0xc6f5c390), (0x471b67e0, 0x471ac64e),
        (0x4a0d9a18, 0x4a0a5ee9), (0x4855c1f7, 0x47cdc8b1), (0x4b5e3478, 0x4b5e901c), (0xc72f8260, 0xc72fdb07), (0x49e73ab7, 0x49dfb3bc), (0x4a24d042, 0x4a24d044),
        (0x49080958, 0x4908047a), (0x482a38f4, 0x482a2aca), (0xca7ab7aa, 0xca7ab7aa), (0x48b31c77, 0x48accc7f), (0xc99d64da, 0xc99d64d8), (0x4a0c8aed, 0x4a0d0034),
        (0xc9ff1400, 0xc9ff6d54), (0xcdcf90e4, 0xcdcf89b2), (0xc6b89dd8, 0xc6e21436), (0x47affa78, 0x47b7ac31), (0xc7f36802, 0xc898ad59), (0x4a2a3a02, 0x4a3afdee),
        (0xc8d7af78, 0xc8d7bc18), (0x48d4ba9c, 0x48d4ba9b), (0x47699de8, 0x47699dee), (0xc98fadfa, 0xc98fae03), (0xc92c969e, 0xc92c96d1), (0x483da9c0, 0x483da9bf),
        (0x4a35d5f4, 0x4a35d600), (0xc97fb799, 0xc97fb79d), (0xc8ac0ed0, 0xc8aa3e58), (0xffc00000, 0xff800000), (0xc891cd7a, 0xc891cd7a), (0xc8226d28, 0xc823c26a),
        (0xc8b79bfc, 0xc8b7ebbc), (0x4905684a, 0x49056849), (0xc7e67155, 0xc7e63769), (0xc949c9e1, 0xc949ca14), (0x47986adf, 0x479d3d6d), (0xca1e049d, 0xca1b5113),
        (0xc7b09c8e, 0xc7af5d6b), (0x49ac4fde, 0x49abea25), (0x473874c0, 0x47366926), (0xc605c268, 0xc605c268), (0x46152936, 0x46159e11), (0xc77b779e, 0xc77b779c),
        (0xc9bb6e98, 0xc9bb73f9), (0x49616816, 0x49616666), (0x481b9e52, 0x480b0955), (0x48c19970, 0x48c19970), (0xc7829270, 0xc7fd9ced), (0x482fbf7f, 0x482fbf7e),
        (0xc9da54be, 0xc9c8e7c3), (0xc97697f8, 0xc973cc37), (0x48087af8, 0x48083c37), (0x4a92e8b2, 0x4a92e924), (0xc9c938b6, 0xc9cce0ca), (0xffc00000, 0xffc00000),
        (0x49539063, 0x4953a167), (0xc8c8f743, 0xc8dd9cc1), (0xc9e73079, 0xca125a5b), (0x49638dbf, 0x495efd9f), (0xc955cda6, 0xc955db5e), (0x49a56f77, 0x49a571ee),
        (0x4a2f7b45, 0x4a2d9aec), (0x46bf5892, 0x46bf588d), (0x48d2ed28, 0x48dfb128), (0xc9048802, 0xc90489bd), (0xc937f2ba, 0xc937f2bf), (0x49af367e, 0x49af352c),
        (0x49255ccf, 0x4925fd9c), (0xc6006ae9, 0xc6006aeb), (0x4904fde2, 0x4904fdce), (0x49af2924, 0x49ae14ee), (0x4a09ce76, 0x4a09cd9f), (0xc8ad6b02, 0xc8a8439f),
        (0x491e21d6, 0x491e21d2), (0x49deb8ee, 0x49de43fc), (0x48d78006, 0x48d78007), (0xc8d67d53, 0xc8d490e2), (0x49e991ec, 0x49e7d5b9), (0xc9d11cd1, 0xc9d1a5b1),
        (0x490aadb8, 0x490ad12e), (0xc898e1cc, 0xc898ac82), (0xca358ef6, 0xca3537d4), (0x47c75919, 0x47c246d6), (0x491e667a, 0x491e38e5), (0xc84293fe, 0xc841f545),
        (0x4584b420, 0x47ab7350), (0xc7d0c018, 0xc7d0c018), (0x44ef2968, 0x44ef2968), (0xc88c39cd, 0xc89c782e), (0xc7e45ad8, 0xc808d136), (0x4813368c, 0x4813368c),
        (0xc837116e, 0xc8b9e078), (0x493a7e9f, 0x49563564), (0x4aba7ef6, 0x4aba7ef6), (0xca061152, 0xca0617ac), (0xc8deadd8, 0xc8deadd9), (0x49e11881, 0x49d52433),
        (0xffc00000, 0xff800000), (0x49654066, 0x4965405c), (0x4800f0b0, 0x4800f5de), (0x4933ef26, 0x4933f4c4), (0xc89dc7b4, 0xc89dc99e), (0xc94d7594, 0xc94d7592),
        (0xc6a77880, 0xc6a8bd3c), (0xc7c600a0, 0xc916f616), (0xca26aa9d, 0xca26ab1d), (0x48939a11, 0x489298d5), (0x455161e1, 0x455161e0), (0x4987f310, 0x49890b99),
        (0x47cce892, 0x47cdd22b), (0xc9c2e5fc, 0xc97a63f4), (0x45f0924a, 0x45f0924a), (0x481ae42c, 0x481ae42e), (0x495a9482, 0x495d9b7b), (0xca9f5cef, 0xca9f5e9e),
        (0x48c399b2, 0x48c38f87), (0x47f58013, 0x47f6eb81), (0xca16ab44, 0xca16afe1), (0xc98c5a94, 0xc98c5a9d), (0xc99eb250, 0xc99eb555), (0xffc00000, 0xff800000),
        (0xc920ea0e, 0xc922bd81), (0xc997bbf6, 0xc9937bbb), (0xc9115eb5, 0xc9115eb4), (0xc976225e, 0xc9264a20), (0x490c8172, 0x490af3dd), (0xca7b2c82, 0xca7b2c83),
        (0x49864b36, 0x49862542), (0x496b8f6a, 0x495f9d49), (0xc743d019, 0xc743d018), (0x48662cf2, 0x48662ce9), (0x4a468c3f, 0x4a474e08), (0xc98a4672, 0xc99c4b4c),
        (0xc91b5ca3, 0xc91afa78), (0xcabe8663, 0xcaf93227), (0xc65f40c8, 0xc65faf56), (0x48167d4d, 0x48167d4d), (0xffc00000, 0xff800000), (0xc8834d41, 0xc8834d41),
        (0x4939ea52, 0x4939ea50), (0xc619b450, 0xc619b450), (0x491f920d, 0x491f8d87), (0xc79b8938, 0xc7b8a8ef), (0xc79c894e, 0xc79e968d), (0xc8957b4e, 0xc8952820),
        (0xc8eb6c3c, 0xc8eb7a97), (0x481f9bd4, 0x4792b7a7), (0x47cef668, 0x47cef667), (0xca5c69ad, 0xca5c69ad), (0xcab048a1, 0xcab0492d), (0xc7dd14c0, 0xc739a206),
        (0xca9cc59a, 0xca9c477d), (0x491ea1e7, 0x496a275c), (0xc974c39e, 0xc972b1a1), (0x49efdb66, 0x49f0d83f), (0xffc00000, 0x7f800000), (0xc930e334, 0xc9305565),
        (0xc88d3c7e, 0xc88d3c7e), (0x4a70440e, 0x4a70440c), (0x499ef034, 0x499f8a7b), (0x4987f6ba, 0x4981bebc), (0x4910980e, 0x49109808), (0xcbc98301, 0xcbc9529a),
        (0xc6cf99a4, 0xc6cf99aa), (0x489a1436, 0x47fa7c54), (0x48084b2a, 0x48084b29), (0x49dc5494, 0x49e035c1), (0xc9462d53, 0xc9462d75), (0x48bbed4a, 0x48bbed4a),
        (0xc8cdf1ee, 0xc9092236), (0xc7b21167, 0xc7b21165), (0xc41baf40, 0xc41bafe7), (0xc8a34b7a, 0xc882ab32), (0xc7c27c7a, 0xc7c27c7a), (0xca74f19e, 0xca79edd5),
        (0xc8e90758, 0xc8e90758), (0x4824c5d8, 0x4824c5d9), (0xc78d7344, 0xc78d7984), (0xc92a6ef2, 0xc92a6ef2), (0xc9c00277, 0xc9c51b4b), (0xc8af353f, 0xc8b1f3dc),
        (0xc92f8463, 0xc92e5992), (0xc833b4cc, 0xc835c11d), (0x48b7a1f4, 0x48b7a1f7), (0xc8bfbcac, 0xc8bfbcab), (0x493f322a, 0x493b5de0), (0xc89a63d8, 0xc89a63d8),
        (0x4999d1fa, 0x4994551e), (0xc368acf8, 0xc368acf9), (0x4a92458f, 0x4a8f8cef), (0xc9e146ff, 0xc9ddd3ab), (0xc9362703, 0xc93dc386), (0x49a3f09d, 0x49adb6ff),
        (0xd00dd53d, 0xd00dd52d), (0xc9a4b660, 0xc9a4b6d6), (0xc7e7a3a7, 0xc7e737d8), (0xc9169afb, 0xc9169af9), (0xc9c0fdfe, 0xc9c0f43a), (0xc981d736, 0xc981d7bb),
        (0x48f39b82, 0x48f39b7f), (0xffc00000, 0xffc00000), (0xc95536e2, 0xc95536e2), (0xc8616ff2, 0xc8616ff1), (0xc86a9cd3, 0xc86a9cd4), (0x49664d14, 0x49664d64),
        (0xc9aacc9a, 0xc9aacc99), (0x4a9f67cd, 0x4a9f5bd6), (0xc7e0a148, 0xc7e0a12d), (0x4a62ae14, 0x4a62ae14), (0xc9dd08b5, 0xca1ef2f1), (0xc90257d4, 0xc9028d9f),
        (0x498e9728, 0x498e34fe), (0x49ba14ed, 0x49b3403c), (0x509771c8, 0x509771c5), (0xc95029ee, 0xc9502e60), (0xc9baf84c, 0xc9baf1ee), (0xc8a21bac, 0xc8a21bd9),
        (0xc90aa770, 0xc90aa76f), (0x491984b0, 0x491984b0), (0x48f838e0, 0x48f838e0), (0xc9305a57, 0xc932e066), (0x49beccc4, 0x49beccd4), (0x476d3b80, 0x4774e3e9),
        (0xc827386d, 0xc7bacd19), (0x48a5859b, 0x48abe15a), (0x453017a0, 0x45301760), (0x4981a96a, 0x4978b335), (0xc856d443, 0xc83ac645), (0xc8c2601e, 0xc8c180af),
        (0xffc00000, 0xff800000), (0xcaa8538e, 0xcaa2e19c), (0x48d1a978, 0x48d7e57f), (0x49a13540, 0x49a13545), (0x4e1266d3, 0x4e1266db), (0xcae7ba50, 0xcae713aa),
        (0xc8d2a500, 0xc8f411c8), (0x4794a042, 0x4794a041), (0xc9eb7332, 0xc9e39a58), (0xc841678a, 0xc8419653), (0xc692fc93, 0xc692fc96), (0x49c28786, 0x49c2713b),
        (0xc88c6fcc, 0xc88c9bb7), (0xc94b7be9, 0xc94b7be8), (0xc839988d, 0xc839988b), (0x47d1352d, 0x47d123f8), (0x4998b175, 0x4998aad2), (0x49fc80e2, 0x49fc44d0),
        (0x490e1552, 0x49142fc8), (0xc8997aa8, 0xc8994dfb), (0x4961728a, 0x4961728a), (0xca374fdc, 0xca3f8652), (0x48e5bb64, 0x48e5bafe), (0x49a0df84, 0x49267c47),
        (0x43f56ade, 0x43f56add), (0xc71d9740, 0xc6f96036), (0x4938b696, 0x4938a375), (0x4a16ea1b, 0x4a16ecfd), (0x48cc4a0b, 0x48caeb0d), (0x4a5f4772, 0x4a5f47bb),
        (0x48dad5de, 0x488e4bb0), (0xc992f41e, 0xc992f41d), (0x48e1d21d, 0x492338a4), (0xc600e1ce, 0xc61f30b2), (0xca2d7c8a, 0xca2e137c), (0xc8c534ae, 0xc8c534ae),
        (0xc790bef9, 0xc790bf57), (0xc82734e8, 0xc82734e9), (0xca161160, 0xc9dfc1ee), (0x489b6f30, 0x4884ef79), (0x499ee142, 0x498e5210), (0xc65be5cf, 0xc65be5cf),
        (0xc8f95bb2, 0xc8f96a0c), (0xc90ec54b, 0xc9172953), (0xc9831d8e, 0xc9823326), (0x47ed9280, 0x47ee7360), (0x480d2d60, 0x480d2d61), (0x46656a94, 0x466bcc96),
        (0xc9fb05f5, 0xc9ff30c7), (0x47706680, 0x478092d8), (0x4859d9a0, 0x4856642a), (0x4a3097e2, 0x4a3114d6), (0x492704e8, 0x48b04650), (0x48ee4ab2, 0x48ed2060),
        (0xc845b06a, 0xc84718c7), (0x49baac90, 0x49bbc76e), (0xd00c6d51, 0xd00c6d48), (0xc8a3297a, 0xc8a33230), (0x4a2b8ac9, 0x4a2b1d59), (0x483e6680, 0x483e6682),
        (0xcc828204, 0xcc8281f4), (0x4a3f58c0, 0x4a24410e), (0x49c7f835, 0x49c8e71b), (0xca4c311a, 0xca4c8891), (0xffc00000, 0xff800000), (0x493002d0, 0x493a653f),
        (0xc98214dc, 0xc997d4ef), (0xc8a9317c, 0xc8ab2c07), (0xc9bef56c, 0xc9befea7), (0xc77ea46e, 0xc77d6e06), (0xc8b34244, 0xc8b3d9d8), (0x4626b5a2, 0x4626b5a0),
        (0xc8f38f08, 0xc93697e2), (0xffc00000, 0xffc00000), (0x479b3cac, 0xc6511290), (0xca31ccc0, 0xca322418), (0xc9f82924, 0xc9f99125), (0xffc00000, 0x7f800000),
        (0x48650258, 0x48650257), (0xc9b967ba, 0xc9b96240), (0x491cf1ae, 0x4917d2ab), (0x461ea409, 0x461ea40b), (0xc99a3057, 0xc99b8779), (0x48e40973, 0x48e40972),
        (0x485f6e08, 0x485f6e0e), (0xc93a51f3, 0xc93a5003), (0x46edc254, 0x46edd3ae), (0x4a0efec7, 0x4a102477), (0xc97854d9, 0xc9773feb), (0xc78883a8, 0xc5c0ca7c),
        (0xc8bd448c, 0xc8bc8590), (0x497de696, 0x497de696), (0xffc00000, 0xffc00000), (0xc578d000, 0xc57f62e0), (0x4cfbb9d6, 0x4cfcae07), (0x47116cbf, 0x47116cc2),
        (0xca163706, 0xca16d6a5), (0xffc00000, 0x7f800000), (0xc943c0d8, 0xc942e20e), (0xc8d32c00, 0xc8d32c74), (0x498d7306, 0x498d8a30), (0x49c2dd57, 0x49d3f6df),
        (0x4874c67b, 0x488d8ac2), (0x46f85ee8, 0x46f85ee9), (0x50268364, 0x50268364), (0xc978bad6, 0xc978bad6), (0x490730da, 0x48fec258), (0x49038b00, 0x49038aff),
        (0xc9a78cb7, 0xc9a7904b), (0x49b6df83, 0x49b5fdcd), (0x47671b33, 0x47671b34), (0x4918b03f, 0x4903e8c7), (0x49c4eec0, 0x49c4ed9d), (0xc63318df, 0xc63318e0),
        (0xffc00000, 0xff800000), (0x4a229029, 0x4a2282b3), (0xc7207cb3, 0xc7207cb1), (0x4786a5e8, 0x476e5394),
    ];

    #[test]
    fn nvfp4_bit_identical_to_llama_cpp() {
        let mut r = Rng(0x9E3779B97F4A7C15);
        let mut x = vec![BlockNVFP4::zeros(); NB1];
        for b in x.iter_mut() {
            for d in b.d.iter_mut() {
                *d = r.next() as u8;
            }
            for q in b.qs.iter_mut() {
                *q = r.next() as u8;
            }
        }
        let mut y = vec![BlockQ8_0x2::zeros(); NB1];
        for (i, b) in as_q8_0_mut(&mut y).iter_mut().enumerate() {
            b.d = r.f16(i, 512);
            for q in b.qs.iter_mut() {
                *q = r.next() as u8 as i8;
            }
        }

        let mut deq = vec![0f32; NB1 * QK_NVFP4];
        BlockNVFP4::to_float(&x, &mut deq);
        assert_eq!(fnv(bytes(&deq)), DEQUANT_FNV, "dequantize_row_nvfp4");

        let mut bad = Vec::new();
        for (t, &(avx, generic)) in VECDOT.iter().enumerate() {
            let nb = 1 + (r.next() % 32) as usize;
            let xo = (r.next() % (NB1 - nb + 1) as u64) as usize;
            let yo = (r.next() % (NB1 - nb + 1) as u64) as usize;
            let n = nb * QK_NVFP4;
            let got = BlockNVFP4::vec_dot(n, &x[xo..xo + nb], &y[yo..yo + nb]).to_bits();
            let got_g = BlockNVFP4::vec_dot_unopt(n, &x[xo..xo + nb], &y[yo..yo + nb]).to_bits();
            #[cfg(target_feature = "avx2")]
            if got != avx {
                bad.push(format!("{t} avx2: {got:08x} want {avx:08x}"));
            }
            #[cfg(not(target_feature = "avx2"))]
            let _ = (got, avx);
            if got_g != generic {
                bad.push(format!("{t} generic: {got_g:08x} want {generic:08x}"));
            }
        }
        assert!(bad.is_empty(), "{} vec_dot mismatches: {:?}", bad.len(), &bad[..bad.len().min(10)]);

        let mut xs = vec![0f32; NQ * QK_NVFP4];
        for v in xs.iter_mut() {
            let rr = r.next();
            *v = if rr % 23 == 0 {
                if rr & 64 != 0 { -0.0 } else { 0.0 }
            } else {
                ((r.next() >> 11) as f64 / 9007199254740992.0 * 16.0 - 8.0) as f32
                    * (1 + (rr % 5) as i32) as f32
                    * 0.01
            };
        }
        for b in (5..NQ).step_by(17) {
            for j in 0..QK_NVFP4 {
                xs[b * QK_NVFP4 + j] = if j & 1 != 0 { -0.0 } else { 0.0 };
            }
        }
        for j in 0..QK_NVFP4 {
            xs[9 * QK_NVFP4 + j] *= 1e-40;
            xs[10 * QK_NVFP4 + j] *= 3e4;
            xs[11 * QK_NVFP4 + j] *= 1e-37;
            xs[12 * QK_NVFP4 + j] *= 1e36;
            xs[13 * QK_NVFP4 + j] = (j as f32 - 16.0) * 0.25;
        }
        let mut q = vec![BlockNVFP4::zeros(); NQ];
        BlockNVFP4::from_float(&xs, &mut q);
        assert_eq!(fnv(bytes(&q)), QUANT_FNV, "quantize_row_nvfp4_ref");
    }
}
