//! GGML MXFP4 (type 39): 32 weights per block, one E8M0 shared exponent `e` and 32 FP4 (E2M1)
//! values as 4-bit indices into `kvalues_mxfp4` (the doubled E2M1 values; the scale is
//! `GGML_E8M0_TO_FP32_HALF(e)` = 2^(e - 128)). Bit-identical ports of llama.cpp (acecd56):
//! `dequantize_row_mxfp4`, `quantize_row_mxfp4_ref` (what the CPU `from_float` calls),
//! `ggml_vec_dot_mxfp4_q8_0` (AVX2 path; its scalar tail is contracted to one fma) and
//! `ggml_vec_dot_mxfp4_q8_0_generic` (gcc -O3 -march=native keeps that one as mul + add).
//! The vec-dot partner is plain Q8_0 (same 32-value block). The CPU keeps denormal scales
//! (e < 2), the CUDA program flushes them: llama.cpp's own CPU and GPU differ there.

use super::k_quants::{BlockQ8_0, GgmlType};
use super::GgmlDType;
use half::f16;

pub const QK_MXFP4: usize = 32;

/// `kvalues_mxfp4` (ggml-common.h `kvalues_fp4`): E2M1 values times two.
pub const KVALUES_MXFP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockMXFP4 {
    pub(crate) e: u8,
    pub(crate) qs: [u8; QK_MXFP4 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockMXFP4>() == 17);

/// ggml-impl.h `ggml_e8m0_to_fp32_half`: 2^(x - 128), denormal for x < 2.
#[inline(always)]
pub fn e8m0_to_fp32_half(x: u8) -> f32 {
    let bits = if x < 2 { 0x0020_0000u32 << x } else { ((x as u32) - 1) << 23 };
    f32::from_bits(bits)
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

/// gcc's `(uint8_t)float` on x86-64: cvttss2si to int32 (0x80000000 for NaN / out of range),
/// then the low byte.
#[inline(always)]
fn cvt_u8(v: f32) -> u8 {
    let i = if v.is_nan() || v >= 2147483648.0 || v < -2147483648.0 { i32::MIN } else { v as i32 };
    i as u8
}

impl GgmlType for BlockMXFP4 {
    const DTYPE: GgmlDType = GgmlDType::MXFP4;
    const BLCK_SIZE: usize = QK_MXFP4;
    type VecDotType = BlockQ8_0;

    // dequantize_row_mxfp4
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(k.is_multiple_of(QK_MXFP4), "dequantize_row_mxfp4: {k} % {QK_MXFP4}");
        for (x, ys) in xs.iter().zip(ys.chunks_exact_mut(QK_MXFP4)) {
            let d = e8m0_to_fp32_half(x.e);
            for j in 0..QK_MXFP4 / 2 {
                ys[j] = KVALUES_MXFP4[(x.qs[j] & 0x0f) as usize] as f32 * d;
                ys[j + QK_MXFP4 / 2] = KVALUES_MXFP4[(x.qs[j] >> 4) as usize] as f32 * d;
            }
        }
    }

    // quantize_row_mxfp4_ref
    fn from_float(xs: &[f32], ys: &mut [Self]) {
        let k = xs.len();
        debug_assert!(k.is_multiple_of(QK_MXFP4), "quantize_row_mxfp4: {k} % {QK_MXFP4}");
        debug_assert_eq!(ys.len(), k / QK_MXFP4);
        for (y, xb) in ys.iter_mut().zip(xs.chunks_exact(QK_MXFP4)) {
            let mut amax = 0f32;
            for &v in xb {
                if amax < v.abs() {
                    amax = v.abs();
                }
            }
            // (uint8_t)(floorf(log2f(amax)) - 2 + 127), glibc log2f
            let e = if amax > 0.0 { cvt_u8(amax.log2().floor() - 2.0 + 127.0) } else { 0 };
            let d = e8m0_to_fp32_half(e);
            y.e = e;
            for j in 0..QK_MXFP4 / 2 {
                let x0 = best_index_mxfp4(xb[j], d);
                let x1 = best_index_mxfp4(xb[QK_MXFP4 / 2 + j], d);
                y.qs[j] = x0 | (x1 << 4);
            }
        }
    }

    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        #[cfg(target_feature = "avx2")]
        return unsafe { vec_dot_mxfp4_q8_0_avx2(n, xs, ys) };

        Self::vec_dot_unopt(n, xs, ys)
    }

    // ggml_vec_dot_mxfp4_q8_0_generic (separate mul and add)
    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(n.is_multiple_of(QK_MXFP4), "vec_dot_mxfp4_q8_0: {n} % {QK_MXFP4}");
        let nb = n / QK_MXFP4;
        let mut sumf = 0f32;
        for (x, y) in xs.iter().zip(ys).take(nb) {
            let d = y.d.to_f32() * e8m0_to_fp32_half(x.e);
            sumf += d * block_sumi(x, y) as f32;
        }
        sumf
    }
}

/// `sumi1 + sumi2` of the scalar loops (exact).
#[inline(always)]
fn block_sumi(x: &BlockMXFP4, y: &BlockQ8_0) -> i32 {
    let (mut s1, mut s2) = (0i32, 0i32);
    for j in 0..QK_MXFP4 / 2 {
        s1 += y.qs[j] as i32 * KVALUES_MXFP4[(x.qs[j] & 0xf) as usize] as i32;
        s2 += y.qs[j + QK_MXFP4 / 2] as i32 * KVALUES_MXFP4[(x.qs[j] >> 4) as usize] as i32;
    }
    s1 + s2
}

// ggml_vec_dot_mxfp4_q8_0, `#if defined __AVX2__` branch, then the scalar tail (one fma).
#[cfg(target_feature = "avx2")]
#[inline(always)]
unsafe fn vec_dot_mxfp4_q8_0_avx2(n: usize, xs: &[BlockMXFP4], ys: &[BlockQ8_0]) -> f32 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    debug_assert!(n.is_multiple_of(QK_MXFP4), "vec_dot_mxfp4_q8_0: {n} % {QK_MXFP4}");
    let nb = n / QK_MXFP4;
    debug_assert!(xs.len() >= nb && ys.len() >= nb);

    let values128 = _mm_loadu_si128(KVALUES_MXFP4.as_ptr() as *const __m128i);
    let m4b = _mm_set1_epi8(0x0f);
    let mone = _mm256_set1_epi16(1);
    // mul_add_epi8: maddubs(sign(x, x), sign(y, x))
    let p = |x: &BlockMXFP4, y: &BlockQ8_0| -> __m256 {
        let q4bits = _mm_loadu_si128(x.qs.as_ptr() as *const __m128i);
        let q8b = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);
        let q4b = _mm256_set_m128i(
            _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits, 4), m4b)),
            _mm_shuffle_epi8(values128, _mm_and_si128(q4bits, m4b)),
        );
        let p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(q4b, q4b), _mm256_sign_epi8(q8b, q4b));
        _mm256_cvtepi32_ps(_mm256_madd_epi16(p16, mone))
    };
    let mut accum1 = _mm256_setzero_ps();
    let mut accum2 = _mm256_setzero_ps();
    let mut ib = 0;
    while ib + 1 < nb {
        let (x0, x1, y0, y1) = (&xs[ib], &xs[ib + 1], &ys[ib], &ys[ib + 1]);
        accum1 = _mm256_fmadd_ps(_mm256_set1_ps(y0.d.to_f32() * e8m0_to_fp32_half(x0.e)), p(x0, y0), accum1);
        accum2 = _mm256_fmadd_ps(_mm256_set1_ps(y1.d.to_f32() * e8m0_to_fp32_half(x1.e)), p(x1, y1), accum2);
        ib += 2;
    }
    let mut sumf = super::avx::hsum_float_8(_mm256_add_ps(accum1, accum2));
    while ib < nb {
        let d = ys[ib].d.to_f32() * e8m0_to_fp32_half(xs[ib].e);
        sumf = d.mul_add(block_sumi(&xs[ib], &ys[ib]) as f32, sumf);
        ib += 1;
    }
    sumf
}

#[cfg(test)]
mod tests {
    //! Oracle: llama.cpp acecd56 itself (oxide-kernels/mxfp4/ref/cpu_ref.c, linked against
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
    const DEQUANT_FNV: u64 = 0x2535278fc1816d47;
    const QUANT_FNV: u64 = 0xaf4df096c2fc82f9;
    /// (AVX2 vec_dot, generic vec_dot) result bits.
    const VECDOT: [(u32, u32); 400] = [
        (0x7bccf8fe, 0x7bccf900), (0xf6cd2b32, 0xf6cd2b34), (0xffc00000, 0xff800000), (0xfcff14d0, 0xfcff14cf), (0xffc00000, 0xffc00000), (0xffc00000, 0xff800000),
        (0xffc00000, 0xff800000), (0xffc00000, 0xffc00000), (0xffc00000, 0xffc00000), (0xffc00000, 0x7f800000), (0xffc00000, 0x7f800000), (0x786a04e0, 0x786a04df),
        (0xfe8ca9f7, 0xfe8ca9fa), (0x7d4593bb, 0x7d4593bb), (0xff800000, 0xff800000), (0x72ee87f1, 0x72ee87f1), (0xffc00000, 0xffc00000), (0xffc00000, 0x7f800000),
        (0x7e32d2c0, 0x7e32d2c0), (0x7600d708, 0x7600d708), (0xf109bf60, 0xf11ef35c), (0xffc00000, 0x7f800000), (0x7eb233e1, 0x7eb23321), (0xffc00000, 0x7f800000),
        (0x7042427c, 0x7042427c), (0xffc00000, 0x7f800000), (0xffc00000, 0xff800000), (0xf7568908, 0xf756890c), (0x7cb09d42, 0x7cb09d42), (0xfbbf3d41, 0xfbbf3d40),
        (0x6fc9711c, 0x6fc9711a), (0xffc00000, 0xff800000), (0xffc00000, 0x7f800000), (0xffc00000, 0x7f800000), (0xf0fa0a00, 0xf0fa0980), (0xffc00000, 0xff800000),
        (0xffc00000, 0xff800000), (0xffc00000, 0xffc00000), (0xff800000, 0xff76f1fc), (0xffc00000, 0xffc00000), (0xff800000, 0xff19034a), (0xff014dca, 0xff014dca),
        (0x7d4928c5, 0x7d2cf4c5), (0x7795cd9c, 0x7795cd9a), (0xffc00000, 0xffc00000), (0xffc00000, 0xff800000), (0xffc00000, 0xff800000), (0xfbc34ec0, 0xfbc34ec0),
        (0xdfb4a186, 0xdfb4a186), (0xffc00000, 0xff800000), (0xffc00000, 0xff800000), (0xff563b7b, 0xff563b7b), (0xffc00000, 0x7f800000), (0xffc00000, 0x7f800000),
        (0xffc00000, 0x7f800000), (0xffc00000, 0x7f800000), (0x77a63f4e, 0x77a63f4f), (0xffc00000, 0xff800000), (0xffc00000, 0xff800000), (0xffc00000, 0x7f800000),
        (0xffc00000, 0xffc00000), (0xffc00000, 0xffc00000), (0xffc00000, 0xff800000), (0x7b5d31a2, 0x7b5d31a2), (0xffc00000, 0x7f800000), (0xf8ac0b26, 0xf8ac0c03),
        (0xffc00000, 0xffc00000), (0x7dab0e3c, 0x7dab0e3d), (0xffc00000, 0xff800000), (0x6fb095d1, 0x6fb095d2), (0x7653faa0, 0x7653fa90), (0xffc00000, 0xffc00000),
        (0x7c7c744e, 0x7c7c744e), (0x7ddfd28a, 0x7ddfd28b), (0xff800000, 0xff800000), (0xffc00000, 0x7f800000), (0xff800000, 0xff1ba5e0), (0x7e2beb86, 0x7e2beb87),
        (0xffc00000, 0xff800000), (0xffc00000, 0x7f800000), (0x7b6d9a06, 0x7b6d9a06), (0xffc00000, 0xffc00000), (0xffc00000, 0xff800000), (0x7819131c, 0x7819131c),
        (0xffc00000, 0xffc00000), (0xffc00000, 0xffc00000), (0xffc00000, 0x7f800000), (0xffc00000, 0xffc00000), (0xffc00000, 0xff800000), (0xffc00000, 0xff800000),
        (0x7f0fcd67, 0x7f0fcd68), (0xffc00000, 0x7f800000), (0xffc00000, 0xff800000), (0xffc00000, 0xff800000), (0xffc00000, 0xff800000), (0xffc00000, 0xff800000),
        (0xffc00000, 0xffc00000), (0x7f800000, 0x7f800000), (0xffc00000, 0xffc00000), (0x7a5ae620, 0x7a5ae620), (0xfded2b30, 0xfded2b32), (0x7d92c5fc, 0x7d92c5fb),
        (0xffc00000, 0xffc00000), (0xffc00000, 0xff800000), (0xffc00000, 0xff800000), (0xffc00000, 0xff800000), (0xd73f57c0, 0xd73f57c0), (0xffc00000, 0x7f800000),
        (0x7558bd00, 0x7558bd00), (0xffc00000, 0xffc00000), (0xffc00000, 0xffc00000), (0xffc00000, 0xffc00000), (0x5be14dd0, 0x5be14dd0), (0xfbdf8684, 0xfbdf8683),
        (0x73b69cda, 0x73b69cda), (0xff800000, 0xff4a29f2), (0xffc00000, 0xff800000), (0x7e559300, 0x7e559300), (0xffc00000, 0xff800000), (0xffc00000, 0x7f800000),
        (0x7e1349e0, 0x7e1349e1), (0x7ee1efc1, 0x7ee1efbf), (0xffc00000, 0x7f800000), (0xffc00000, 0xffc00000), (0x7eab2650, 0x7eab264f), (0x6262bd00, 0x6262bd00),
        (0x7074d1b9, 0x7074d1b9), (0xffc00000, 0x7f800000), (0xffc00000, 0xffc00000), (0xffc00000, 0xff800000), (0xffc00000, 0xffc00000), (0xfd05ecc4, 0xfd05ecc4),
        (0xffc00000, 0xffc00000), (0xfc9fd356, 0xfc9fd356), (0xffc00000, 0x7f800000), (0x41037d68, 0x41037d67), (0xffc00000, 0xff800000), (0xfe930fee, 0xfe930fed),
        (0xffc00000, 0x7f800000), (0x775c835c, 0x775a44dc), (0xffc00000, 0xff800000), (0xffc00000, 0xffc00000), (0xe38715ca, 0xe38715c9), (0xffc00000, 0xff800000),
        (0xfb9ee899, 0xfb9ee899), (0xfd41d28c, 0xfd41d28c), (0xffc00000, 0x7f800000), (0xffc00000, 0x7ed3afc1), (0xffc00000, 0xff800000), (0xe8c0f800, 0xe8c0f800),
        (0x7a2dd888, 0x7a2dd890), (0xffc00000, 0xffc00000), (0x7e8ab01a, 0x7e8ab018), (0xffc00000, 0x7f800000), (0xffc00000, 0xff800000), (0xffc00000, 0x7f800000),
        (0xffc00000, 0xff800000), (0xffc00000, 0x7f800000), (0x7e0a2b8b, 0x7e0a16df), (0xfe56b140, 0xfe56b13f), (0xffc00000, 0xff800000), (0x7f800000, 0x7f800000),
        (0xffc00000, 0xff800000), (0xfedfbb1c, 0xfedfbd36), (0xf6309ea3, 0xf6309ea3), (0xffc00000, 0xffc00000), (0x7e2c190c, 0x7e08310d), (0xfaa157f4, 0xfaa157ea),
        (0xffc00000, 0xff800000), (0xfd0cfaed, 0xfd0cfaec), (0x7ecb6301, 0x7ecb6301), (0xffc00000, 0xfedeaa2d), (0xffc00000, 0x7f800000), (0xffc00000, 0xff800000),
        (0xffc00000, 0xff800000), (0xffc00000, 0x7f800000), (0xfccbf5f2, 0xfccbf5f2), (0xffc00000, 0x7f800000), (0xffc00000, 0x7f800000), (0x3825aa46, 0x3825aa46),
        (0xffc00000, 0xff800000), (0xffc00000, 0xff800000), (0xffc00000, 0xffc00000), (0xffc00000, 0xffc00000), (0xff17bc06, 0xff17bc05), (0xffc00000, 0x7f800000),
        (0xfc430b4e, 0xfc430b4e), (0x7e7530ac, 0x7e7530ac), (0xfc911b23, 0xfc911b23), (0xfdb9515c, 0xfdb9515b), (0x7cea5652, 0x7cea4cf3), (0xf356b420, 0xf356b43b),
        (0xffc00000, 0xffc00000), (0xffc00000, 0xff800000), (0x54854a26, 0x54854a26), (0xffc00000, 0x7f800000), (0xda0b96d9, 0xda0b96d9), (0xffc00000, 0xffc00000),
        (0xffc00000, 0x7f800000), (0xffc00000, 0xff800000), (0xffc00000, 0xff800000), (0x77bceda8, 0x77bceda3), (0x7b8a2907, 0x7b8a1166), (0xffc00000, 0xffc00000),
        (0x7dc53474, 0x7dc4e7d2), (0xffc00000, 0xff800000), (0xffc00000, 0x7f800000), (0xffc00000, 0xffc00000), (0xffc00000, 0xffc00000), (0xffc00000, 0xff800000),
        (0x7f800000, 0x7f800000), (0xffc00000, 0x7f800000), (0xffc00000, 0xffc00000), (0xfaa76fc0, 0xfaa76c84), (0xffc00000, 0xffc00000), (0x18de8de4, 0x18de8de4),
        (0xffc00000, 0xffc00000), (0xffc00000, 0xff800000), (0x7ef4abb4, 0x7ef4a51e), (0x5252c54c, 0x5252c54b), (0xffc00000, 0xff800000), (0xffc00000, 0xffc00000),
        (0xffc00000, 0xff800000), (0xffc00000, 0xffc00000), (0xffc00000, 0xff800000), (0xca6eef70, 0xca6eef70), (0xffc00000, 0xffc00000), (0x758fdc68, 0x758fdc67),
        (0xffc00000, 0xffc00000), (0xffc00000, 0xff800000), (0xffc00000, 0x7f800000), (0x7a22a050, 0x7a218171), (0xffc00000, 0x7f800000), (0x7639e30a, 0x7639e30a),
        (0x7e6098f9, 0x7e6098fa), (0xffc00000, 0xffc00000), (0x7e9cc570, 0x7e9cc56f), (0xffc00000, 0x7f800000), (0xfd508bdd, 0xfd508ed4), (0xffc00000, 0x7f800000),
        (0xffc00000, 0x7f800000), (0xffc00000, 0xffc00000), (0xea0bcdfa, 0xea0bcdfb), (0x7f800000, 0x7f800000), (0xfea51e10, 0xfea51e11), (0xffc00000, 0x7f800000),
        (0xffc00000, 0x7f800000), (0x6f92d0e5, 0x6f92d0e6), (0x7a527a82, 0x7a527a83), (0xffc00000, 0x7f800000), (0x7d18e608, 0x7d18e3dc), (0xffc00000, 0xffc00000),
        (0xffc00000, 0x7f800000), (0xffc00000, 0x7f800000), (0xf8902420, 0xf906a610), (0xffc00000, 0xff800000), (0xff800000, 0xff800000), (0xffc00000, 0xffc00000),
        (0x7e2d921e, 0xff16ab7b), (0xffc00000, 0xff800000), (0x7b796f24, 0x7b796f24), (0xffc00000, 0xffc00000), (0xffc00000, 0x7e8de2f3), (0xffc00000, 0xff800000),
        (0xffc00000, 0xffc00000), (0xffc00000, 0x7f800000), (0xffc00000, 0xffc00000), (0xffc00000, 0x7f800000), (0xff800000, 0xff800000), (0x7f800000, 0x7f800000),
        (0x793e4520, 0x793e4520), (0x7d949fa8, 0x7d949fa5), (0xffc00000, 0x7f800000), (0xeb224fc2, 0xeb224fc2), (0xf7a0927e, 0xf7a0927d), (0x7b8f5000, 0x7b8f5016),
        (0xfc887ab8, 0xfc887ab7), (0xfc991388, 0xfc991330), (0xffc00000, 0x7f800000), (0xffc00000, 0x7f800000), (0xffc00000, 0xff800000), (0xffc00000, 0xffc00000),
        (0xffc00000, 0x7f800000), (0xffc00000, 0xffc00000), (0xfb5ae5d2, 0xfb5ae5d2), (0x7492a9f5, 0x7492a9f3), (0xfc1003fa, 0xfc100d09), (0xffc00000, 0xff800000),
        (0xffc00000, 0xff800000), (0x7f800000, 0x7f193bd2), (0xffc00000, 0xff800000), (0x60170ec1, 0x60170ec2), (0xffc00000, 0xffc00000), (0xfec49ca9, 0xfec49ca9),
        (0xfcefa258, 0xfcefa243), (0xffc00000, 0xffc00000), (0xffc00000, 0xffc00000), (0xffc00000, 0x7f800000), (0xffc00000, 0x7f800000), (0x7de22e9c, 0x7de22e97),
        (0xff0eec74, 0xff0eec74), (0xffc00000, 0xffc00000), (0xffc00000, 0x7f800000), (0xffc00000, 0x7f800000), (0xff800000, 0xff800000), (0xffc00000, 0x7f800000),
        (0x5be457c0, 0x5be457c0), (0x74985f80, 0x74985f80), (0xffc00000, 0xff800000), (0xffc00000, 0x7f800000), (0xff014878, 0xff014879), (0xffc00000, 0xffc00000),
        (0xffc00000, 0x7f800000), (0xffc00000, 0xff800000), (0xfcd44a3c, 0xfcd44a3b), (0x7f800000, 0x7f800000), (0x7e07c8bf, 0x7e2274bf), (0xffc00000, 0x7f800000),
        (0xffc00000, 0x7f800000), (0xfe2c984d, 0xfe2c984c), (0x4b56bf60, 0x4b56bf60), (0xfb614946, 0xfb614946), (0x7d842512, 0x7d8e43a7), (0xffc00000, 0xff800000),
        (0x7ba1971e, 0x7bb71f1e), (0x7f800000, 0x7f800000), (0xffc00000, 0x7f800000), (0xffc00000, 0xffc00000), (0xffc00000, 0x7f800000), (0xffc00000, 0xff800000),
        (0xfcbdaade, 0xfcbdab9b), (0x7e593998, 0x7e593998), (0xffc00000, 0x7f800000), (0x71ee6af4, 0x71ee6c92), (0xffc00000, 0xffc00000), (0x7e021164, 0x7de03aca),
        (0xffc00000, 0xffc00000), (0xffc00000, 0xff800000), (0xffc00000, 0xffc00000), (0xffc00000, 0xffc00000), (0x1a952a24, 0x1a952a24), (0xffc00000, 0xff800000),
        (0xffc00000, 0xffc00000), (0x7ce0ac04, 0x7ce0ac04), (0xffc00000, 0xff800000), (0xfd5c58a1, 0xfd5c58a0), (0xffc00000, 0xffc00000), (0xffc00000, 0xff800000),
        (0xfb8bc954, 0xfbe2a554), (0xffc00000, 0xff800000), (0xffc00000, 0xff800000), (0xffc00000, 0xffc00000), (0xffc00000, 0xff800000), (0xffc00000, 0xffc00000),
        (0xfd566fd0, 0xfd566fd1), (0x75c83070, 0x75c83070), (0xffc00000, 0x7f800000), (0xfc1f4f6e, 0xfd00f3dc), (0x7c2fa596, 0x7c2fa654), (0xffc00000, 0xff800000),
        (0xffc00000, 0xffc00000), (0xffc00000, 0xff800000), (0xff800000, 0xff800000), (0xffc00000, 0x7f800000), (0xffc00000, 0xffc00000), (0x7d2b65d6, 0x7d2b0716),
        (0xffc00000, 0xff800000), (0xff800000, 0xff800000), (0xfd3625bc, 0xfd4293bc), (0xffc00000, 0xffc00000), (0xffc00000, 0xff800000), (0xff800000, 0xff800000),
        (0xffc00000, 0xff800000), (0xff800000, 0xff800000), (0xffc00000, 0xffc00000), (0x7970d2c7, 0x7970d2c6), (0xfd4c9360, 0xfd4c9361), (0x7a8863ac, 0x7a8863ab),
        (0xffc00000, 0xffc00000), (0xffc00000, 0x7f800000), (0xffc00000, 0xffc00000), (0x75d0eb10, 0x75d0eb0f), (0xffc00000, 0x7f800000), (0x7dd66168, 0x7dd66169),
        (0xfa1ce9b0, 0xfa1ce9af), (0x789e6899, 0x789e6898), (0xfdd3cf68, 0xfdd3cf68), (0xffc00000, 0xff800000), (0x59bae460, 0x59bae460), (0xfda3f8e6, 0xfda3f8e6),
        (0x7daa7877, 0x7daa7829), (0xfc991a3c, 0xfc991a39), (0xffc00000, 0xffc00000), (0xf300aecb, 0xf300aeca), (0xffc00000, 0x7f800000), (0xffc00000, 0x7f800000),
        (0xf0df7902, 0xf0df7903), (0xffc00000, 0xffc00000), (0xff800000, 0xff800000), (0xffc00000, 0x7f800000),
    ];

    #[test]
    fn mxfp4_bit_identical_to_llama_cpp() {
        let mut r = Rng(0x9E3779B97F4A7C15);
        let mut x = vec![BlockMXFP4::zeros(); NB1];
        for b in x.iter_mut() {
            b.e = r.next() as u8;
            for q in b.qs.iter_mut() {
                *q = r.next() as u8;
            }
        }
        let mut y = vec![BlockQ8_0::zeros(); NB1];
        for (i, b) in y.iter_mut().enumerate() {
            b.d = r.f16(i, 512);
            for q in b.qs.iter_mut() {
                *q = r.next() as u8 as i8;
            }
        }

        let mut deq = vec![0f32; NB1 * QK_MXFP4];
        BlockMXFP4::to_float(&x, &mut deq);
        assert_eq!(fnv(bytes(&deq)), DEQUANT_FNV, "dequantize_row_mxfp4");

        let mut bad = Vec::new();
        for (t, &(avx, generic)) in VECDOT.iter().enumerate() {
            let nb = 1 + (r.next() % 64) as usize;
            let xo = (r.next() % (NB1 - nb + 1) as u64) as usize;
            let yo = (r.next() % (NB1 - nb + 1) as u64) as usize;
            let n = nb * QK_MXFP4;
            let got = BlockMXFP4::vec_dot(n, &x[xo..xo + nb], &y[yo..yo + nb]).to_bits();
            let got_g = BlockMXFP4::vec_dot_unopt(n, &x[xo..xo + nb], &y[yo..yo + nb]).to_bits();
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

        let mut xs = vec![0f32; NQ * QK_MXFP4];
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
            for j in 0..QK_MXFP4 {
                xs[b * QK_MXFP4 + j] = if j & 1 != 0 { -0.0 } else { 0.0 };
            }
        }
        for j in 0..QK_MXFP4 {
            xs[9 * QK_MXFP4 + j] *= 1e-40;
            xs[10 * QK_MXFP4 + j] *= 3e4;
            xs[11 * QK_MXFP4 + j] *= 1e-37;
            xs[12 * QK_MXFP4 + j] *= 1e36;
            xs[13 * QK_MXFP4 + j] = (j as f32 - 16.0) * 0.25;
        }
        let mut q = vec![BlockMXFP4::zeros(); NQ];
        BlockMXFP4::from_float(&xs, &mut q);
        assert_eq!(fnv(bytes(&q)), QUANT_FNV, "quantize_row_mxfp4_ref");
    }
}
