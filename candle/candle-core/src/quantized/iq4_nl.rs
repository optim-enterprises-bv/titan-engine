//! GGML IQ4_NL (type 20): 32 weights per block, one f16 scale `d` and 32 4-bit indices into the
//! non-linear table `kvalues_iq4nl` (value = d * table[index]). Bit-identical ports of llama.cpp
//! (acecd56): `dequantize_row_iq4_nl`, `quantize_row_iq4_nl_ref` (what the CPU `from_float`
//! calls), `ggml_vec_dot_iq4_nl_q8_0` (AVX2 path) and `ggml_vec_dot_iq4_nl_q8_0_generic`
//! (gcc -O3 -march=native contracts `sumf += d * sumi` into one fma, as does the AVX2 tail).
//! The vec-dot partner is plain Q8_0 (same 32-value block).

use super::k_quants::{BlockQ8_0, GgmlType};
use super::GgmlDType;
use half::f16;

pub const QK4_NL: usize = 32;

/// `kvalues_iq4nl` (ggml-common.h).
pub const KVALUES_IQ4NL: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockIQ4nl {
    pub(crate) d: f16,
    pub(crate) qs: [u8; QK4_NL / 2],
}
const _: () = assert!(std::mem::size_of::<BlockIQ4nl>() == 18);

/// ggml-quants.c `best_index_int8`.
fn best_index_int8(val: &[i8; 16], x: f32) -> usize {
    if x <= val[0] as f32 {
        return 0;
    }
    if x >= val[15] as f32 {
        return 15;
    }
    let (mut ml, mut mu) = (0usize, 15usize);
    while mu - ml > 1 {
        let mav = (ml + mu) / 2;
        if x < val[mav] as f32 {
            mu = mav
        } else {
            ml = mav
        }
    }
    if x - (val[mu - 1] as f32) < (val[mu] as f32) - x {
        mu - 1
    } else {
        mu
    }
}

impl GgmlType for BlockIQ4nl {
    const DTYPE: GgmlDType = GgmlDType::IQ4NL;
    const BLCK_SIZE: usize = QK4_NL;
    type VecDotType = BlockQ8_0;

    // dequantize_row_iq4_nl
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(k.is_multiple_of(QK4_NL), "dequantize_row_iq4_nl: {k} % {QK4_NL}");
        for (x, ys) in xs.iter().zip(ys.chunks_exact_mut(QK4_NL)) {
            let d = x.d.to_f32();
            for j in 0..QK4_NL / 2 {
                ys[j] = d * KVALUES_IQ4NL[(x.qs[j] & 0xf) as usize] as f32;
                ys[j + QK4_NL / 2] = d * KVALUES_IQ4NL[(x.qs[j] >> 4) as usize] as f32;
            }
        }
    }

    // quantize_row_iq4_nl_ref: quantize_row_iq4_nl_impl(QK4_NL, 32, ..., quant_weights = NULL,
    // ntry = -1) per block. `L` is one scratch array for the whole row, so a block whose max |x|
    // is below GROUP_MAX_EPS keeps the previous block's indices (and d = 0).
    fn from_float(xs: &[f32], ys: &mut [Self]) {
        const GROUP_MAX_EPS: f32 = 1e-15;
        let k = xs.len();
        debug_assert!(k.is_multiple_of(QK4_NL), "quantize_row_iq4_nl: {k} % {QK4_NL}");
        debug_assert_eq!(ys.len(), k / QK4_NL);
        let values = &KVALUES_IQ4NL;
        let mut l = [0u8; QK4_NL];
        for (y, xb) in ys.iter_mut().zip(xs.chunks_exact(QK4_NL)) {
            let mut weight = [0f32; QK4_NL];
            for (w, &x) in weight.iter_mut().zip(xb) {
                *w = x * x;
            }
            let (mut amax, mut max) = (0f32, 0f32);
            for &x in xb {
                let ax = x.abs();
                if ax > amax {
                    amax = ax;
                    max = x;
                }
            }
            let scale = if amax < GROUP_MAX_EPS {
                0f32
            } else {
                let d = max / values[0] as f32;
                let id = 1.0 / d;
                let (mut sumqx, mut sumq2) = (0f32, 0f32);
                for j in 0..QK4_NL {
                    let al = id * xb[j];
                    let li = best_index_int8(values, al);
                    l[j] = li as u8;
                    let q = values[li] as f32;
                    let w = weight[j];
                    sumqx += w * q * xb[j];
                    sumq2 += w * q * q;
                }
                if sumq2 > 0.0 {
                    sumqx / sumq2
                } else {
                    0.0
                }
            };
            y.d = f16::from_f32(scale);
            for j in 0..16 {
                y.qs[j] = l[j] | (l[16 + j] << 4);
            }
        }
    }

    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        #[cfg(target_feature = "avx2")]
        return unsafe { vec_dot_iq4_nl_q8_0_avx2(n, xs, ys) };

        Self::vec_dot_unopt(n, xs, ys)
    }

    // ggml_vec_dot_iq4_nl_q8_0_generic
    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(n.is_multiple_of(QK4_NL), "vec_dot_iq4_nl_q8_0: {n} % {QK4_NL}");
        let nb = n / QK4_NL;
        let mut sumf = 0f32;
        for (x, y) in xs.iter().zip(ys).take(nb) {
            let d = y.d.to_f32() * x.d.to_f32();
            sumf = d.mul_add(block_sumi(x, y) as f32, sumf);
        }
        sumf
    }
}

/// `sumi1 + sumi2` of the scalar loops (exact).
#[inline(always)]
fn block_sumi(x: &BlockIQ4nl, y: &BlockQ8_0) -> i32 {
    let (mut s1, mut s2) = (0i32, 0i32);
    for j in 0..QK4_NL / 2 {
        s1 += y.qs[j] as i32 * KVALUES_IQ4NL[(x.qs[j] & 0xf) as usize] as i32;
        s2 += y.qs[j + QK4_NL / 2] as i32 * KVALUES_IQ4NL[(x.qs[j] >> 4) as usize] as i32;
    }
    s1 + s2
}

// ggml_vec_dot_iq4_nl_q8_0, `#if defined __AVX2__` branch, then the scalar tail.
#[cfg(target_feature = "avx2")]
#[inline(always)]
unsafe fn vec_dot_iq4_nl_q8_0_avx2(n: usize, xs: &[BlockIQ4nl], ys: &[BlockQ8_0]) -> f32 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    debug_assert!(n.is_multiple_of(QK4_NL), "vec_dot_iq4_nl_q8_0: {n} % {QK4_NL}");
    let nb = n / QK4_NL;
    debug_assert!(xs.len() >= nb && ys.len() >= nb);

    let values128 = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
    let m4b = _mm_set1_epi8(0x0f);
    let mone = _mm256_set1_epi16(1);
    // mul_add_epi8: maddubs(sign(x, x), sign(y, x)) (y = -128 with x < 0 wraps, as in llama.cpp)
    let p = |x: &BlockIQ4nl, y: &BlockQ8_0| -> __m256 {
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
        accum1 = _mm256_fmadd_ps(_mm256_set1_ps(y0.d.to_f32() * x0.d.to_f32()), p(x0, y0), accum1);
        accum2 = _mm256_fmadd_ps(_mm256_set1_ps(y1.d.to_f32() * x1.d.to_f32()), p(x1, y1), accum2);
        ib += 2;
    }
    let mut sumf = super::avx::hsum_float_8(_mm256_add_ps(accum1, accum2));
    while ib < nb {
        let d = ys[ib].d.to_f32() * xs[ib].d.to_f32();
        sumf = d.mul_add(block_sumi(&xs[ib], &ys[ib]) as f32, sumf);
        ib += 1;
    }
    sumf
}

#[cfg(test)]
mod tests {
    //! Oracle: llama.cpp acecd56 itself (oxide-kernels/iq4_nl/ref/cpu_ref.c, linked against
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
    const DEQUANT_FNV: u64 = 0xf0a873dc19ca8261;
    const QUANT_FNV: u64 = 0xc9ec7ea4d911f0c7;
    /// (AVX2 vec_dot, generic vec_dot) result bits.
    const VECDOT: [(u32, u32); 400] = [
        (0xffc00000, 0x7f800000), (0x47fb86fa, 0x47f88f48), (0xc9665d69, 0xc94dbdf2), (0xffc00000, 0xff800000), (0xffc00000, 0xff800000), (0x7fc00000, 0x7fc00000),
        (0x48e2b637, 0x48e2b637), (0xffc00000, 0x7f800000), (0xc8253d61, 0xc863336e), (0xc975d87f, 0xc9761cd6), (0xcc24213e, 0xcc2433d8), (0xc8050b99, 0xc80163a6),
        (0xc8eebfc6, 0xc907460c), (0xffc00000, 0xff800000), (0x487ff81f, 0x487f3d7e), (0x7fc00000, 0x7fc00000), (0x7fc00000, 0x7fc00000), (0xc7d22f25, 0xc7bc8896),
        (0xffc00000, 0x7f800000), (0x491aa780, 0x491a7666), (0x48b6aa66, 0x48b6eb0a), (0xc7198398, 0xc7198399), (0x7fc00000, 0x7fc00000), (0xc913f1da, 0xc90e2726),
        (0xc9db09fb, 0xc9d90aab), (0xd01dc5fe, 0xd01dc5fe), (0x4903ced9, 0x490401b1), (0x4986455a, 0x4988c8a5), (0x46d37311, 0x46d4ec99), (0xcc6eabc3, 0xcc6e5a2b),
        (0x4ab15db1, 0x4ab13df8), (0x49d7da51, 0x49d73f85), (0x7fc00000, 0x7fc00000), (0xffc00000, 0xff800000), (0x497503f1, 0x4960824f), (0xcc2cf9a4, 0xcc2ccf2b),
        (0xffc00000, 0xffc00000), (0xffc00000, 0xff800000), (0xffc00000, 0xff800000), (0xc680a15e, 0xc680a15f), (0x7fc00000, 0x7fc00000), (0x488e8f60, 0x488e8feb),
        (0xc719c3cf, 0xc76a0e8e), (0x482d8afc, 0x482d8afd), (0xffc00000, 0x7f800000), (0x47433368, 0x47433368), (0x4d943d27, 0x4c59198a), (0x445b98a0, 0x45c6addc),
        (0xffc00000, 0xff800000), (0x47112c2a, 0x4712b47c), (0x48fa5d27, 0x48fa5d26), (0xffc00000, 0xff800000), (0xc966cd18, 0xc963e4ad), (0xffc00000, 0x7f800000),
        (0x467564e5, 0x467564e4), (0xc9533112, 0xc9532a27), (0xc898178f, 0xc888f709), (0xc9ccac95, 0xc9ccac95), (0xc9167626, 0xc917c693), (0xc762a78c, 0xc762a78b),
        (0xffc00000, 0x7f800000), (0xcf5021a6, 0xcf5021af), (0xffc00000, 0x7f800000), (0x491a19b2, 0x49499109), (0xc8d4153b, 0xc8d40dd4), (0x4d03cd4c, 0x4d03cd60),
        (0x49991b66, 0x49a11d94), (0xc8079ac9, 0xc8176b2e), (0xffc00000, 0x7f800000), (0x485c9506, 0x485c929e), (0x472632c0, 0x472632c2), (0xffc00000, 0x7f800000),
        (0xc7c5fcc9, 0xc7c5fcc9), (0xc8bf0d7c, 0xc8c0f125), (0x494f554d, 0x494b9c9d), (0xffc00000, 0xff800000), (0x49000b7a, 0x4900113b), (0xc592c514, 0xc592c513),
        (0xffc00000, 0x7f800000), (0xc5d92d94, 0xc5d92d94), (0xffc00000, 0xff800000), (0x7fc00000, 0x7fc00000), (0xffc00000, 0x7f800000), (0x464fe036, 0x464fe034),
        (0x49115cf4, 0x49115cf6), (0xc5c59e48, 0xc5c59e49), (0x4f85193c, 0x4f85193f), (0xffc00000, 0x7f800000), (0xc8a4e4ac, 0xc8a565aa), (0xffc00000, 0x7f800000),
        (0xc35112b8, 0xc351129f), (0x4907f136, 0x49093ae2), (0x4dbe0e46, 0x4dbe0e44), (0xffc00000, 0x7f800000), (0xffc00000, 0xff800000), (0xc8ddda8a, 0xc9265ce6),
        (0x4a079d04, 0x49f1ee28), (0xc804549d, 0xc8366784), (0xffc00000, 0xffc00000), (0x480e6a56, 0x480e6a55), (0xc647ce28, 0xc648b72a), (0xc98b04de, 0xc98ad8bc),
        (0xffc00000, 0xffc00000), (0xc67741df, 0xc67741dc), (0xffc00000, 0x7f800000), (0x4845c9ad, 0x4845f7cf), (0xc64f5800, 0xc64f5800), (0x4804704f, 0x47f8d699),
        (0xc775aa3f, 0xc775aa43), (0xc7b1d778, 0xc7af8a3b), (0xffc00000, 0xffc00000), (0xc88a132c, 0xc8895fd8), (0x4822e2f3, 0x4822e2f3), (0x48cee826, 0x48d7f457),
        (0xc4b53180, 0xc4b53168), (0xcf4f2947, 0xcf4f2af7), (0xffc00000, 0xffc00000), (0x49ab638b, 0x498b8f08), (0xffc00000, 0x7f800000), (0xc5d2e868, 0xc5d2e887),
        (0x490595a4, 0x490544b5), (0xc68aa068, 0xc68aa068), (0x49d0eb27, 0x49d09132), (0xffc00000, 0xff800000), (0xc8b8dbef, 0xc8b3d8a8), (0x48a7f9f0, 0x48a7f157),
        (0xc6ac47f0, 0xc6ae92b8), (0x481f8d98, 0x481f8d9a), (0xc8864667, 0xc88610fb), (0xc6766803, 0xc6766803), (0xc84517ba, 0xc84b262c), (0x7fc00000, 0x7fc00000),
        (0xc62f0470, 0xc684d3ef), (0x474b7e59, 0x474b7e59), (0xc7e7df92, 0xc7d7aace), (0xffc00000, 0xffc00000), (0xc6b24ae2, 0xc6b25153), (0x45de2f40, 0x45de2f41),
        (0x7fc00000, 0x7fc00000), (0x44e1dcd6, 0x44e1dcd5), (0xffc00000, 0xffc00000), (0x48029edc, 0x4808661d), (0x49821783, 0x4982186c), (0xc8fbc41f, 0xc8f752e4),
        (0x49957486, 0x498d4247), (0x4995050c, 0x4995050a), (0xffc00000, 0xff800000), (0xc43b32dc, 0xc43b32ea), (0xc3dd49dd, 0xc3dd49dd), (0xc87f8b35, 0xc87f8b38),
        (0xffc00000, 0xffc00000), (0xffc00000, 0xff800000), (0xc90cacfe, 0xc90cba0e), (0x4f0b237c, 0x4f0b2326), (0xc697fa58, 0xc697fa58), (0xc388c57c, 0xc388c57c),
        (0xc3fedf98, 0xc3fedf97), (0x46756a32, 0x46173b82), (0x4ee07a19, 0x4ee07a18), (0xd02427de, 0xd02426d3), (0x4940555d, 0x493c6a72), (0xc700cad5, 0xc700bf5b),
        (0xc8fc3587, 0xc90e2a3b), (0x483026ba, 0x483026b9), (0x7fc00000, 0x7fc00000), (0xffc00000, 0x7f800000), (0x4767971c, 0x4767971a), (0xc889ed3d, 0xc88a1f9f),
        (0x7fc00000, 0x7fc00000), (0xc9287d84, 0xc91ee2fe), (0x498b01d0, 0x498b041b), (0xc94bacf4, 0xc948064a), (0xce0c56ec, 0xce0c6a09), (0x7fc00000, 0x7fc00000),
        (0xc9a1db00, 0xc914b0f1), (0x45c72379, 0x45c72379), (0xffc00000, 0x7f800000), (0xc64088ea, 0xc64088ea), (0xc600f5b0, 0xc600f582), (0x7fc00000, 0x7fc00000),
        (0x4d00cd32, 0x4d00fc7c), (0x4715105e, 0x47051834), (0xc824a082, 0xc824a082), (0x4e8fc2b5, 0x4e8fc2ce), (0xc11a7f06, 0xc11a7f06), (0xca387f6b, 0xca3849e9),
        (0xc858addd, 0xc867dcd6), (0x7fc00000, 0x7fc00000), (0xc88525e2, 0xc8853ed1), (0x465d1f30, 0x465d1f2d), (0xc8d7df2e, 0xc8d7bdff), (0xc76e6020, 0xc772d9fb),
        (0xc808cfe7, 0xc808cfe7), (0xc8b33926, 0xc8b3b278), (0xc892bdeb, 0xc892bdea), (0xffc00000, 0x7f800000), (0xffc00000, 0x7f800000), (0xffc00000, 0xff800000),
        (0xcf044dff, 0xcf044e19), (0xffc00000, 0xffc00000), (0x4c038a88, 0x4c0393f7), (0xcc95fb6c, 0xcc95fb6d), (0xffc00000, 0x7f800000), (0xc9aaf8b6, 0xc9abfbbd),
        (0x4873c359, 0x4873d14f), (0x7fc00000, 0x7fc00000), (0xffc00000, 0xff800000), (0xc6a331f0, 0xc6a331f1), (0xc73c0c94, 0xc737cb51), (0xffc00000, 0xffc00000),
        (0xc898a526, 0xc898a528), (0x7fc00000, 0x7fc00000), (0x7fc00000, 0x7fc00000), (0xc87c39c4, 0xc89d7672), (0xc95bbf41, 0xc958d6d6), (0xce635f87, 0xce636047),
        (0xce8b7cc6, 0xce8b7cc3), (0x48ab23d3, 0x48b648c4), (0x47e7d5a8, 0x47e96a2f), (0xffc00000, 0x7f800000), (0x7fc00000, 0x7fc00000), (0x7fc00000, 0x7fc00000),
        (0xccbdd177, 0xccbdd166), (0xffc00000, 0x7f800000), (0xc828e363, 0xc828e365), (0x4f025777, 0x4f025777), (0xffc00000, 0xffc00000), (0xc7ca8053, 0xc798128d),
        (0xc8b3ecf7, 0xc8b3ecf6), (0xc84012a0, 0xc83fc0c4), (0x48622195, 0x4862219b), (0x484add72, 0x488064cc), (0xffc00000, 0xffc00000), (0x4723820c, 0x47244bd0),
        (0xcc7e455a, 0xcc7e50d4), (0xffc00000, 0x7f800000), (0x4687d63d, 0x4687d63d), (0xffc00000, 0xff800000), (0x7f800000, 0x7f800000), (0x4d5fae6a, 0xcda41a03),
        (0xc967348f, 0xc9673491), (0x47b187b1, 0x47b1907f), (0x7fc00000, 0x7fc00000), (0xc98a543d, 0xc98a6836), (0xc8381b5c, 0xc82c2e19), (0xffc00000, 0x7f800000),
        (0x7fc00000, 0x7fc00000), (0x48823633, 0x48823634), (0xc6f3ad6c, 0xc6f677b4), (0xc8480a56, 0xc7323c00), (0xcca52f28, 0xcca5307d), (0xca12237c, 0xca129b4a),
        (0x47893174, 0x47893174), (0xffc00000, 0x7f800000), (0xffc00000, 0xffc00000), (0xc6a2a9be, 0xc6a2a9b8), (0xcb56499d, 0xca8c7f8c), (0xc7aba86c, 0xc807e71c),
        (0x4993c519, 0x4994b4a4), (0xffc00000, 0xff800000), (0xc92a2993, 0xc92a17c8), (0x491a6aa5, 0x491980a2), (0x7fc00000, 0x7fc00000), (0xc89948f6, 0xc89a79f1),
        (0x4e249fad, 0x4e249c78), (0x48df8f17, 0x48de6537), (0xc69689e7, 0xc69689e6), (0x42f3ce2b, 0xc36f2fbb), (0xffc00000, 0x7f800000), (0xc9a8b74a, 0xc9a8b747),
        (0x7fc00000, 0x7fc00000), (0xc7f0b11c, 0xc7c0605f), (0x48aeb8e2, 0x48ad1d42), (0xffc00000, 0xffc00000), (0xffc00000, 0x7f800000), (0xcccd4fcd, 0xcccd435a),
        (0x50751922, 0x50751921), (0xffc00000, 0xff800000), (0xc796fa0c, 0xc796ae0b), (0x4c8d2d05, 0xccc864a6), (0xc66d0d65, 0xc66d0d66), (0xc635bbc8, 0xc635bbf0),
        (0x48ab3aa4, 0x48ab6eb0), (0x48509310, 0x486f0fdb), (0x4e37bbec, 0x4e37bbeb), (0x4ca0b679, 0x4ca0b6cd), (0xffc00000, 0x7f800000), (0x469a5f58, 0x46c6ce0f),
        (0xc785bdde, 0xc72a2787), (0xffc00000, 0xff800000), (0xffc00000, 0xff800000), (0xc8dbfff4, 0xc8dbe5c3), (0x7fc00000, 0x7fc00000), (0xccb7fc08, 0xccb7f613),
        (0xc8dfca3f, 0xc8dfca40), (0x7fc00000, 0x7fc00000), (0xc9636ec4, 0xc9632085), (0xffc00000, 0x7f800000), (0xc80f42ea, 0xc82441ad), (0xffc00000, 0xff800000),
        (0xcea66f42, 0xcea66f17), (0xc8b6b451, 0xc8c9e3a1), (0xc75a901c, 0xc75a901c), (0x48b4b6d0, 0x48b52eca), (0x7fc00000, 0x7fc00000), (0x4cb8033c, 0x4cb80b37),
        (0xc89b98ae, 0xc89b98ae), (0x7fc00000, 0x7fc00000), (0x4977096b, 0x4975c21e), (0x497ca64e, 0x49800fef), (0x487c755d, 0x487c6e8f), (0x7fc00000, 0x7fc00000),
        (0x7fc00000, 0x7fc00000), (0xc9a2f298, 0xc9a2fd20), (0xce574058, 0xce574091), (0xc71a6530, 0x4880a292), (0x7fc00000, 0x7fc00000), (0xc952ac78, 0xc96d7826),
        (0xc836f0af, 0xc836f89f), (0x47b5973c, 0x47b37d49), (0x488df636, 0x488fb873), (0x4c69f0e8, 0x4c6a45d0), (0xc61f2d65, 0xc68b9a9f), (0x47b136ee, 0x47b136ef),
        (0x4c241b13, 0x4c2423c9), (0xffc00000, 0x7f800000), (0xffc00000, 0xffc00000), (0x4787ffc5, 0x47d87c76), (0xc511a600, 0xc78c4759), (0x48d2af32, 0x48aa8e86),
        (0x49fe57d1, 0x49fcc256), (0x46bd8453, 0x46bd8453), (0xca6612b7, 0xca661db6), (0x488a1f96, 0x48acd27f), (0xc895e41c, 0xc8968faa), (0xffc00000, 0xff800000),
        (0xffc00000, 0xff800000), (0xffc00000, 0xffc00000), (0xffc00000, 0x7f800000), (0xffc00000, 0xff800000), (0xffc00000, 0x7f800000), (0x7fc00000, 0x7fc00000),
        (0xccac6cd8, 0xccac6cd7), (0x4846a93f, 0x4846a93e), (0x4888c10d, 0x488879e3), (0x7fc00000, 0x7fc00000), (0xffc00000, 0x7f800000), (0x47c86a25, 0x47c86a27),
        (0x476b4fed, 0x476b4ff5), (0x4683972a, 0x4683972a), (0x43e39640, 0x43e39640), (0xcbcfbab2, 0xcbcfbab7), (0x7fc00000, 0x7fc00000), (0xffc00000, 0xff800000),
        (0x7fc00000, 0x7fc00000), (0xffc00000, 0x7f800000), (0x48fade61, 0x48fadcb2), (0x7fc00000, 0x7fc00000), (0x485bbd58, 0x485c7ddb), (0xc5d3b8fc, 0xc5d620e6),
        (0x7fc00000, 0x7fc00000), (0xc79ca80d, 0xc797df26), (0xff800000, 0xff800000), (0x48592272, 0x48592271), (0xffc00000, 0xffc00000), (0x4ae91e3c, 0x4ae810c5),
        (0x7fc00000, 0x7fc00000), (0x50075d68, 0x50075cf2), (0x49827da7, 0x49827da7), (0x7fc00000, 0x7fc00000), (0xffc00000, 0x7f800000), (0x5002388c, 0x5002388d),
        (0x47ed700e, 0xc8973325), (0xffc00000, 0x7f800000), (0xffc00000, 0x7f800000), (0x453c8ae8, 0x453c8ae8), (0xffc00000, 0xff800000), (0xffc00000, 0x7f800000),
        (0xffc00000, 0xffc00000), (0x4908bdd2, 0x4908bdd3), (0xc704ff19, 0xc704ff19), (0x4cdaa6ac, 0x4cdaa516), (0x46c6f6c0, 0x46c6f6b7), (0x45730e46, 0x45730e46),
        (0x47a1eabe, 0x47a336ca), (0x48b35b1d, 0x48d56dd5), (0xffc00000, 0x7f800000), (0x4917e8d7, 0x4917e8d3), (0xffc00000, 0xff800000), (0x4766a2b0, 0x475cd657),
        (0x7fc00000, 0x7fc00000), (0xc6c0e3fe, 0xc690389a), (0x7fc00000, 0x7fc00000), (0x47bb2904, 0x47c7369f), (0x7fc00000, 0x7fc00000), (0x47c9ceda, 0x47c9cedb),
        (0xffc00000, 0xffc00000), (0x47c1c6ef, 0x47c1cc10), (0xc7ae41ce, 0xc79be7a2), (0xc7cd61d6, 0xc7ce1ba2),
    ];

    #[test]
    fn iq4_nl_bit_identical_to_llama_cpp() {
        let mut r = Rng(0x9E3779B97F4A7C15);
        let mut x = vec![BlockIQ4nl::zeros(); NB1];
        for (i, b) in x.iter_mut().enumerate() {
            b.d = r.f16(i, 32);
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

        let mut deq = vec![0f32; NB1 * QK4_NL];
        BlockIQ4nl::to_float(&x, &mut deq);
        assert_eq!(fnv(bytes(&deq)), DEQUANT_FNV, "dequantize_row_iq4_nl");

        let mut bad = Vec::new();
        for (t, &(avx, generic)) in VECDOT.iter().enumerate() {
            let nb = 1 + (r.next() % 64) as usize;
            let xo = (r.next() % (NB1 - nb + 1) as u64) as usize;
            let yo = (r.next() % (NB1 - nb + 1) as u64) as usize;
            let n = nb * QK4_NL;
            let got = BlockIQ4nl::vec_dot(n, &x[xo..xo + nb], &y[yo..yo + nb]).to_bits();
            let got_g = BlockIQ4nl::vec_dot_unopt(n, &x[xo..xo + nb], &y[yo..yo + nb]).to_bits();
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

        let mut xs = vec![0f32; NQ * QK4_NL];
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
            for j in 0..QK4_NL {
                xs[b * QK4_NL + j] = if j & 1 != 0 { -0.0 } else { 0.0 };
            }
        }
        for j in 0..QK4_NL {
            xs[9 * QK4_NL + j] *= 1e-17;
            xs[10 * QK4_NL + j] *= 3e4;
        }
        let mut q = vec![BlockIQ4nl::zeros(); NQ];
        BlockIQ4nl::from_float(&xs, &mut q);
        assert_eq!(fnv(bytes(&q)), QUANT_FNV, "quantize_row_iq4_nl_ref");
    }
}
