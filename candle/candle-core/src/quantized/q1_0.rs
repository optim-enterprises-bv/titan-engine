//! GGML Q1_0 (type 41): 128 weights per block, one f16 scale `d` and 128 sign bits
//! (bit 1 -> +d, bit 0 -> -d). Bit-identical ports of llama.cpp (acecd56):
//! `dequantize_row_q1_0`, `quantize_row_q1_0_ref`, `ggml_vec_dot_q1_0_q8_0` (AVX2 path) and
//! `ggml_vec_dot_q1_0_q8_0_generic` (separate mul/add, as gcc -O3 -march=native emits it).
//!
//! llama.cpp dots Q1_0 against four consecutive Q8_0 blocks. candle's generic matmul needs the
//! vec-dot type to share the 128 block size, so the activations use [`BlockQ8_0x4`]: four
//! `block_q8_0` back to back, byte-identical to llama.cpp's Q8_0 row.

use super::k_quants::{BlockQ8_0, GgmlType};
use super::GgmlDType;
use half::f16;

pub const QK1_0: usize = 128;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ1_0 {
    pub(crate) d: f16,
    pub(crate) qs: [u8; QK1_0 / 8],
}
const _: () = assert!(std::mem::size_of::<BlockQ1_0>() == 18);

/// Four Q8_0 blocks (128 values): the vec-dot partner of [`BlockQ1_0`].
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8_0x4 {
    pub(crate) b: [BlockQ8_0; 4],
}
const _: () = assert!(std::mem::size_of::<BlockQ8_0x4>() == 4 * 34);

fn as_q8_0(ys: &[BlockQ8_0x4]) -> &[BlockQ8_0] {
    unsafe { std::slice::from_raw_parts(ys.as_ptr() as *const BlockQ8_0, ys.len() * 4) }
}

fn as_q8_0_mut(ys: &mut [BlockQ8_0x4]) -> &mut [BlockQ8_0] {
    unsafe { std::slice::from_raw_parts_mut(ys.as_mut_ptr() as *mut BlockQ8_0, ys.len() * 4) }
}

impl GgmlType for BlockQ8_0x4 {
    // Only ever used as a vec-dot scratch type; its bytes are a Q8_0 row.
    const DTYPE: GgmlDType = GgmlDType::Q8_0;
    const BLCK_SIZE: usize = QK1_0;
    type VecDotType = BlockQ8_0x4;

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

impl GgmlType for BlockQ1_0 {
    const DTYPE: GgmlDType = GgmlDType::Q1_0;
    const BLCK_SIZE: usize = QK1_0;
    type VecDotType = BlockQ8_0x4;

    // dequantize_row_q1_0
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(k.is_multiple_of(QK1_0), "dequantize_row_q1_0: {k} % {QK1_0}");
        for (x, ys) in xs.iter().zip(ys.chunks_exact_mut(QK1_0)) {
            let d = x.d.to_f32();
            let neg_d = -d;
            for (j, y) in ys.iter_mut().enumerate() {
                let bit = (x.qs[j / 8] >> (j % 8)) & 1;
                *y = if bit != 0 { d } else { neg_d };
            }
        }
    }

    // quantize_row_q1_0_ref
    fn from_float(xs: &[f32], ys: &mut [Self]) {
        let k = xs.len();
        debug_assert!(k.is_multiple_of(QK1_0), "quantize_row_q1_0: {k} % {QK1_0}");
        debug_assert_eq!(ys.len(), k / QK1_0);
        for (y, xs) in ys.iter_mut().zip(xs.chunks_exact(QK1_0)) {
            let mut sum_abs = 0f32;
            for &x in xs {
                sum_abs += x.abs();
            }
            let d = sum_abs / QK1_0 as f32;
            y.d = f16::from_f32(d);
            y.qs = [0u8; QK1_0 / 8];
            for (j, &x) in xs.iter().enumerate() {
                if x >= 0.0 {
                    y.qs[j / 8] |= 1 << (j % 8);
                }
            }
        }
    }

    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        #[cfg(target_feature = "avx2")]
        return unsafe { vec_dot_q1_0_q8_0_avx2(n, xs, as_q8_0(ys)) };

        Self::vec_dot_unopt(n, xs, ys)
    }

    // ggml_vec_dot_q1_0_q8_0_generic
    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(n.is_multiple_of(QK1_0), "vec_dot_q1_0_q8_0: {n} % {QK1_0}");
        let nb = n / QK1_0;
        let y = as_q8_0(ys);
        let mut sumf = 0f32;
        for (i, x) in xs.iter().enumerate().take(nb) {
            let d0 = x.d.to_f32();
            let mut sumi = 0f32;
            for k in 0..4 {
                let yb = &y[i * 4 + k];
                let d1 = yb.d.to_f32();
                let mut sumi_block = 0i32;
                for b in 0..4 {
                    let mask = x.qs[k * 4 + b];
                    let qy = &yb.qs[b * 8..b * 8 + 8];
                    let mut s = 0i32;
                    for (l, &q) in qy.iter().enumerate() {
                        let q = q as i32;
                        s += if mask & (1 << l) != 0 { q } else { -q };
                    }
                    sumi_block += s;
                }
                // gcc -O3 -march=native keeps these as separate vmulss/vaddss (checked in the
                // ref harness disassembly), so no fma here.
                sumi += d1 * sumi_block as f32;
            }
            sumf += d0 * sumi;
        }
        sumf
    }
}

// ggml_vec_dot_q1_0_q8_0, `#if defined(__AVX2__)` branch.
#[cfg(target_feature = "avx2")]
#[inline(always)]
unsafe fn vec_dot_q1_0_q8_0_avx2(n: usize, xs: &[BlockQ1_0], ys: &[BlockQ8_0]) -> f32 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    debug_assert!(n.is_multiple_of(QK1_0), "vec_dot_q1_0_q8_0: {n} % {QK1_0}");
    let nb = n / QK1_0;
    debug_assert!(xs.len() >= nb && ys.len() >= nb * 4);

    let ones_8 = _mm256_set1_epi8(1);
    let ones_16 = _mm256_set1_epi16(1);
    let byte_shuf = _mm256_setr_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
        3,
    );
    let bit_masks = _mm256_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1,
        2, 4, 8, 16, 32, 64, -128,
    );
    let zero = _mm256_setzero_si256();
    let mut acc = _mm256_setzero_ps();

    let chunk = |bits: u32, y: &BlockQ8_0| -> __m256 {
        let qy = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);
        let sm = _mm256_cmpeq_epi8(
            _mm256_and_si256(
                _mm256_shuffle_epi8(_mm256_set1_epi32(bits as i32), byte_shuf),
                bit_masks,
            ),
            zero,
        );
        let sy = _mm256_sub_epi8(_mm256_xor_si256(qy, sm), sm);
        let s32 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_8, sy), ones_16);
        _mm256_cvtepi32_ps(s32)
    };

    for ib in 0..nb {
        let x = &xs[ib];
        let d0 = x.d.to_f32();
        let q = |k: usize| u32::from_le_bytes([x.qs[4 * k], x.qs[4 * k + 1], x.qs[4 * k + 2], x.qs[4 * k + 3]]);
        let y = &ys[ib * 4..ib * 4 + 4];
        let mut acc_block = _mm256_mul_ps(_mm256_set1_ps(y[0].d.to_f32()), chunk(q(0), &y[0]));
        for k in 1..4 {
            acc_block = _mm256_fmadd_ps(_mm256_set1_ps(y[k].d.to_f32()), chunk(q(k), &y[k]), acc_block);
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d0), acc_block, acc);
    }
    super::avx::hsum_float_8(acc)
}

#[cfg(test)]
mod tests {
    //! Oracle: llama.cpp acecd56 itself (oxide-kernels/q1_0/ref/cpu_ref.c, linked against
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
    const DEQUANT_FNV: u64 = 0x270f64585f2698a5;
    const QUANT_FNV: u64 = 0x2ce9bd399e39025e;
    /// (AVX2 ggml_vec_dot_q1_0_q8_0, ggml_vec_dot_q1_0_q8_0_generic) result bits.
    const VECDOT: [(u32, u32); 400] = [
        (0xc5108b80, 0xc50629b0), (0x45e00bcc, 0x461a70b4), (0xc69915a2, 0xc699760b), (0x436c9694, 0x4467b14a), (0x459c5f8d, 0x459ce797), (0x45c36e0c, 0x45d147dd),
        (0xffc00000, 0x7f800000), (0xc4bde388, 0xc4bde388), (0xc4c1628b, 0xc4c96f56), (0xc510fc42, 0xc5104205), (0xffc00000, 0x7f800000), (0x46ea350d, 0x46e90fbd),
        (0xc6222d22, 0xc6149136), (0xc6198f62, 0xc6146afb), (0x457bb833, 0x45861094), (0x468264fb, 0x468264fa), (0x4315251e, 0x4315251d), (0xc5fd8280, 0xc6076043),
        (0x453cb8df, 0x456ac12e), (0x7fc00000, 0x7fc00000), (0x45be5308, 0x45d16c8f), (0xffc00000, 0xffc00000), (0xffc00000, 0x7f800000), (0x4523f0d0, 0x464dfa50),
        (0xc4bc8d25, 0xc4db33bd), (0xc45c8fdb, 0xc48d2c3d), (0x7fc00000, 0x7fc00000), (0x4d2ec105, 0x4d1f5338), (0xffc00000, 0x7f800000), (0x4173f840, 0xc13eb500),
        (0xc37d77e4, 0xc37d77e3), (0xc4d041fc, 0xc2252100), (0x7fc00000, 0x7fc00000), (0xcc23e7b4, 0xcc23e72b), (0x46a0a985, 0x46a18658), (0x471a5618, 0x47240b47),
        (0xc58d360a, 0xc57f3dd2), (0xffc00000, 0x7f800000), (0xc3c32b2c, 0xc3c0c118), (0xc4ccb0b9, 0xc4d57db1), (0x463e8f68, 0x4624af25), (0xc6a31628, 0xc69ff333),
        (0x7fc00000, 0x7fc00000), (0xc5311c52, 0xc5311c52), (0xc5e4bb55, 0xc5e4d224), (0xc5a1de90, 0xc5957faa), (0xc4da43d1, 0xc4da43d2), (0x45111944, 0x45124448),
        (0x453c5d44, 0x46012b58), (0xc4f3cb74, 0xc4efe684), (0x4576341c, 0x4577ae54), (0xc6c4883c, 0xc6ba6089), (0xc38599a5, 0xc38599a5), (0x46453d2e, 0x464a1224),
        (0x43249d32, 0x43249d32), (0x44b9608e, 0x44b9608e), (0x44f2feb6, 0x44a3bae1), (0x47629d5a, 0x4765a0a4), (0x7fc00000, 0x7fc00000), (0xc42778f3, 0xc48d6ecf),
        (0x45002ba0, 0x45002ba0), (0xcc469cf7, 0xcc46a68d), (0xccd35fd8, 0xccd35fe8), (0x44377d31, 0x4403ebb2), (0xcc00a876, 0xcc00a88c), (0xc39eba28, 0x45696b81),
        (0x448d18b8, 0x44b5fdb7), (0xc5c3a3ca, 0xc5cdfea8), (0xffc00000, 0xff800000), (0x4543c014, 0x453d6fab), (0x4517bfc8, 0x44ec3b8b), (0xc44e8a20, 0xc398bf18),
        (0xffc00000, 0xffc00000), (0x4562ec2f, 0x45504136), (0x4380b176, 0x439247b6), (0xc5cfafd0, 0xc5cf9bd6), (0x45fc7e40, 0x45fc7e3e), (0xffc00000, 0x7f800000),
        (0x4427ef68, 0x4421e582), (0xc263b114, 0xc267cd50), (0xc5dc3e1e, 0xc5ebe811), (0xffc00000, 0xffc00000), (0x468045fe, 0x4638ceae), (0x4684416a, 0x46871364),
        (0xc5c09a33, 0xc5f844c2), (0xffc00000, 0x7f800000), (0x450d7990, 0x44ff4d8b), (0xffc00000, 0x7f800000), (0xc43a2346, 0xc4362e44), (0x470cd17d, 0x470ca39c),
        (0x45f3e4f0, 0x4605606e), (0xc636234e, 0xc637f4a9), (0x3fc94fd0, 0x3fc94fce), (0xc61b844f, 0xc61a29d2), (0xc70c612e, 0xc70c71b1), (0xc5e404a3, 0xc6079c3a),
        (0x4c46c383, 0xca850d35), (0xc6be0128, 0xc6bf01f2), (0xc5dd5fe3, 0xc5eda4be), (0x409ff000, 0x4125d170), (0xc64d36bc, 0xc64b0452), (0x445afc34, 0x45f9bc86),
        (0x7fc00000, 0x7fc00000), (0xc64f28e6, 0xc64f38e8), (0x4230191c, 0x411c54e4), (0xc5baccdc, 0xc5d9b794), (0xc443dee0, 0xc43bed40), (0xc5220e0e, 0xc51ce8c1),
        (0x4623ee36, 0x460d1a89), (0xffc00000, 0x7f800000), (0x7fc00000, 0x7fc00000), (0xc6ab378a, 0xc6ab16a2), (0x439343e0, 0x43982eec), (0xc63e92a1, 0xc63f139e),
        (0x45727a07, 0x45727a06), (0xc4585c41, 0xc441e490), (0x7fc00000, 0x7fc00000), (0x461dc538, 0x4584e9be), (0x45eba2fd, 0x45ed9d3b), (0x425357d8, 0x42a247d4),
        (0x4459365c, 0x445770f5), (0xbf1c4000, 0x4443f7f5), (0xc731a638, 0xc72ba902), (0x44685664, 0x439ae814), (0xc62ee16e, 0xc630df93), (0x4593f625, 0x4594f3f3),
        (0x460a6805, 0x460e4229), (0xc6a5fd92, 0xc6a8e849), (0x461b6143, 0x462ab262), (0x4601e5c0, 0x46022d36), (0x4d957db4, 0x4d957dc5), (0x454fa31a, 0x451dcf44),
        (0x7fc00000, 0x7fc00000), (0xc13bd62a, 0xc13bd629), (0xc6f21d50, 0xc6f84964), (0xffc00000, 0xff800000), (0x7fc00000, 0x7fc00000), (0x7fc00000, 0x7fc00000),
        (0x449ef768, 0x45f20643), (0x4a052a18, 0xcaa0ee82), (0x46970a0b, 0x467c44fa), (0xc6bb9940, 0xc6c26782), (0x450223fc, 0x450223fb), (0xc2e01982, 0xc2e01982),
        (0xffc00000, 0xffc00000), (0x4737d12a, 0x473d595b), (0x46ab08b0, 0x4687d9ee), (0xc70f0609, 0xc70bc8ba), (0x45a14b56, 0x45a7d181), (0xc3c26063, 0xc3c26063),
        (0x4733dec9, 0x471c7c3e), (0xc6b7eeb4, 0xc6b3ff95), (0x472b058c, 0x472ab13b), (0xc3026458, 0xc2cf619a), (0xc69d8415, 0xc68592af), (0xffc00000, 0xff800000),
        (0x4480763e, 0x4480763f), (0xc50ebb4e, 0xc52b6295), (0xffc00000, 0x7f800000), (0xc516f45e, 0xc5872312), (0xc589f53d, 0xc58dffe0), (0xc5755a3e, 0xc4e74688),
        (0xc3dcf384, 0xc3dcf384), (0x46656206, 0x46656206), (0x44fb770e, 0x446ffbee), (0x4690d02b, 0x469304cb), (0x46366b52, 0x46178f18), (0xc1bea7c0, 0x41eb61e0),
        (0xffc00000, 0x7f800000), (0x7fc00000, 0x7fc00000), (0xccab74e2, 0xcd2e4f89), (0xc4c4ec00, 0xc50181d0), (0x46cd0fec, 0x46ce5494), (0xc5192c0c, 0xc5192c0c),
        (0x447626e0, 0x447ed24c), (0xc669e737, 0xc669e8b2), (0xc6df6813, 0xc6d72944), (0x42ebe180, 0x428c601c), (0xc36d6f0c, 0xc36ed8bd), (0xc6fc2b20, 0xc6fb7ba3),
        (0xffc00000, 0x7f800000), (0xc651080a, 0xc63ac1a4), (0xc6d7b753, 0xc6c979fd), (0xc2f9714e, 0xc2e73892), (0xc6161c58, 0xc61313ae), (0x44497db8, 0x44497dbb),
        (0x7fc00000, 0x7fc00000), (0xffc00000, 0xffc00000), (0x451db342, 0x44eaf8df), (0xc2d15b22, 0xc2d15b22), (0x45b6780e, 0x458e413d), (0x462f34a3, 0x4635deda),
        (0xc68cafa6, 0xc68cafa6), (0x449fc3a4, 0x449fc3a4), (0xc5a572e8, 0xc5a572e7), (0xffc00000, 0xff800000), (0xffc00000, 0x7f800000), (0xc4a5e84e, 0xc4a5f148),
        (0xc49f1450, 0xc49f1450), (0x463f2936, 0x463fb706), (0x449535cf, 0x449535cf), (0x43702240, 0x43f5ecfb), (0xc4a20140, 0xc49c0aa5), (0xc6101e08, 0xc615a372),
        (0x420243e6, 0x4202fb61), (0x45c24e8c, 0x45fd088c), (0xc32561ba, 0xc32561ba), (0x4601d487, 0x4606a642), (0x45d2beaa, 0x45c6c65e), (0xc6bcc56b, 0xc6bcdc30),
        (0xc6c0d934, 0xc6c0d934), (0xc61354d9, 0xc60f5c66), (0x4322b268, 0x4347bedc), (0xcc936a6f, 0xcc936a39), (0x46392ffe, 0x463e50d2), (0x7fc00000, 0x7fc00000),
        (0x46527f8e, 0x4699bf98), (0x4aa88590, 0x4aa88588), (0xc54845a5, 0xc4b62e6e), (0x44e1a109, 0x44db1677), (0xcce52712, 0xcce52712), (0xc60a6d4a, 0xc6111150),
        (0x7fc00000, 0x7fc00000), (0xc623f6d3, 0xc61f0f28), (0x44b1b25c, 0x44af9b18), (0xc5a80098, 0xc5ba55fb), (0xc5aacb02, 0xc57f8cbd), (0x7fc00000, 0x7fc00000),
        (0xc70387ee, 0xc7064d78), (0xc613fd2a, 0xc60f2676), (0x450a940a, 0x450a940a), (0x43d070e0, 0x43aea5d9), (0x46161730, 0x46161730), (0xc610454f, 0xc60ff0e0),
        (0xc669a34c, 0xc66a85b4), (0x4620425d, 0x4620169c), (0x468aebbc, 0x4692bc2a), (0x7fc00000, 0x7fc00000), (0x4573ecd8, 0x4573ecd7), (0xc61d0a4a, 0xc61ea757),
        (0xc4b4650a, 0xc4b3ef78), (0xc3e1d85c, 0xc3fcef4d), (0xffc00000, 0x7f800000), (0xc4480a9f, 0xc49ad881), (0x458f5b31, 0x458f5b30), (0x4690f1ea, 0x46e45af6),
        (0x46b917f1, 0x46bb6770), (0xc5c152c8, 0xc5c1ff1b), (0xffc00000, 0x7f800000), (0x7fc00000, 0x7fc00000), (0xffc00000, 0xff800000), (0x46194c48, 0x4628a229),
        (0xc5754a36, 0xc5794901), (0xc6330fad, 0xc633ecae), (0xc6235c85, 0xc6235c84), (0xc488cb20, 0xc484503c), (0xffc00000, 0x7f800000), (0x45ecc6c4, 0x45ecb701),
        (0xc6c91b09, 0xc6cc6917), (0xc3691ec0, 0xc3977523), (0x45884e8c, 0x45812e5c), (0xc5b35b81, 0xc472a28e), (0x7fc00000, 0x7fc00000), (0xffc00000, 0xffc00000),
        (0x4587be38, 0x45d27258), (0xc53ee04d, 0xc53ee04b), (0x4693b8e1, 0x4693d83d), (0xc4d27be4, 0x41891800), (0x4406f18d, 0x4406f18e), (0xffc00000, 0xffc00000),
        (0x4c0935fa, 0x4c0935f9), (0xc5cb67aa, 0xc6125780), (0xffc00000, 0x7f800000), (0xc56ee4b0, 0xc5391dd3), (0xc5488cb0, 0xc5347c2f), (0xc7242352, 0xc70637cb),
        (0xc53b6994, 0xc53b6996), (0x44572852, 0x44572852), (0x46d1103a, 0x46d112d1), (0x43c8abea, 0x43c8abea), (0x45021282, 0xc34e64c0), (0x46ac8a00, 0x46a4b197),
        (0x463c98d9, 0x4611741f), (0x4c36d3ee, 0x4c36d4af), (0x472da81e, 0x472db23d), (0x7fc00000, 0x7fc00000), (0x4407eb82, 0x44093d88), (0x4c2f338a, 0x4c2f33cd),
        (0xc58483bc, 0xc599f3c1), (0xc4bac7a8, 0xc4bac7aa), (0xc5e3da83, 0xc5e81c64), (0xc4491ee0, 0xc44cc6be), (0xc1841087, 0xc1a27264), (0x4477f980, 0xc49ae3b7),
        (0xc38d8a58, 0xc38d8a59), (0x4657356e, 0x465e6492), (0xffc00000, 0xff800000), (0x46ed24ec, 0x46e9d612), (0x445140f2, 0xc308fd04), (0x4674a4d1, 0x468d974e),
        (0x44cb6340, 0x44cb94b9), (0xc63a78e8, 0xc639c0b1), (0xc6a65f1c, 0xc6b1312f), (0xffc00000, 0x7f800000), (0xc60edf2d, 0xc6357c2c), (0xffc00000, 0xffc00000),
        (0x46b50c72, 0x46a45add), (0xc0930330, 0xc0930340), (0x43194a74, 0x431979ec), (0xc627db3e, 0xc627fccc), (0x44e1bd28, 0x44e1bd26), (0x4bee7b46, 0x4bee7bb0),
        (0x43f7a960, 0x441caf03), (0xc71a586a, 0xc71a1d7a), (0x432ec434, 0xc29da8ce), (0xffc00000, 0xff800000), (0x44f117f8, 0x44bc9a7a), (0x3fe61d14, 0x3fe61cce),
        (0x7fc00000, 0x7fc00000), (0xc524a7e5, 0xc52a99a7), (0xc319d860, 0xc495ffb8), (0x451e3664, 0x452a5605), (0xffc00000, 0xffc00000), (0x468e1514, 0x4690dc58),
        (0x43552a1a, 0x43552a18), (0x46bd4018, 0x46cfea9f), (0x4c1344a2, 0x4c13446b), (0xc690f5d0, 0xc6944926), (0xc53946fe, 0xc549fa94), (0x434654ad, 0x434654ae),
        (0x4d2f5726, 0x4d2f5729), (0xc6166a4c, 0xc613e9c4), (0xc60558e9, 0xc605fb8c), (0x443c844e, 0x441ef6c9), (0xc52cf0a0, 0xc52cf0a1), (0x468c054c, 0x46a3be6a),
        (0x450938d8, 0x45044a26), (0xc4fd328d, 0xc55281c3), (0xc5ccfaa8, 0xc5b879a4), (0x45334355, 0x45854db6), (0x450ce4e1, 0x450d89af), (0xca905971, 0xca905b62),
        (0x44c7feba, 0x44c7fda3), (0xc60257dc, 0xc60146c8), (0x45a9bc31, 0x44a091f0), (0x4b0ad850, 0x4b0ad870), (0x4697a79e, 0x46988c08), (0xffc00000, 0x7f800000),
        (0x7fc00000, 0x7fc00000), (0xc626bc26, 0xc626cf49), (0x464f138e, 0x464d340b), (0x4719e6ed, 0x47207999), (0x7fc00000, 0x7fc00000), (0x4479ebf8, 0x4479ebfa),
        (0xc4ab1d42, 0xc4adadce), (0xffc00000, 0x7f800000), (0x46b28d2a, 0x46b2f3dd), (0xc666196a, 0xc6552542), (0x451cf3ee, 0x45927d43), (0x46682cf0, 0x46508ca5),
        (0x46a7012e, 0x46c022c4), (0xc41e3858, 0xc41c12a2), (0xc5c9037e, 0xc5ba288a), (0x43c45cbd, 0x43c4bc9e), (0xc5c97590, 0xc5a93416), (0xc679eb00, 0xc6808788),
        (0x45d6895e, 0x45ddff79), (0xc5a423a4, 0xc5c81791), (0xffc00000, 0x7f800000), (0xc6a16443, 0xc697a38a), (0x465f95e2, 0x4657432a), (0xffc00000, 0xffc00000),
        (0x460ad8b7, 0x464a9667), (0xc5d2cc3c, 0xc5cf60f4), (0xffc00000, 0xff800000), (0xc619febf, 0xc62cb968), (0x46769658, 0x46801b50), (0xc5908d90, 0xc4cb5c9d),
        (0xc5cd5470, 0xc615579e), (0x44f8c9c0, 0xc407fdf5), (0xc5e79b02, 0xc616f2be), (0xffc00000, 0xff800000), (0xcd044698, 0xcd044697), (0x43a35200, 0xc28920b7),
        (0xcc1888bb, 0xcc1888c1), (0xc56da8be, 0xc56dd624), (0xffc00000, 0x7f800000), (0xc6dfb9f4, 0xc6d4c884), (0xffc00000, 0x7f800000), (0x464ec90e, 0x4649d5f1),
        (0xc6106c42, 0xc618df04), (0xc6c6e3be, 0xc67f0b4b), (0xffc00000, 0x7f800000), (0x43659558, 0x4519463a), (0xc33f56e0, 0xc33f56e1), (0xc590eae0, 0xc59d8d7d),
        (0xc5ba181a, 0xc5bab083), (0x45713e98, 0x45625fb3), (0xc6ade20e, 0xc6ae6ddb), (0x469b7ade, 0x46baa656),
    ];

    #[test]
    fn q1_0_bit_identical_to_llama_cpp() {
        let mut r = Rng(0x9E3779B97F4A7C15);
        let mut x = vec![BlockQ1_0::zeros(); NB1];
        for (i, b) in x.iter_mut().enumerate() {
            b.d = r.f16(i, 32);
            for q in b.qs.iter_mut() {
                *q = r.next() as u8;
            }
        }
        let mut y = vec![BlockQ8_0x4::zeros(); NB1];
        for (i, b) in as_q8_0_mut(&mut y).iter_mut().enumerate() {
            b.d = r.f16(i, 512);
            for q in b.qs.iter_mut() {
                *q = r.next() as u8 as i8;
            }
        }

        let mut deq = vec![0f32; NB1 * QK1_0];
        BlockQ1_0::to_float(&x, &mut deq);
        assert_eq!(fnv(bytes(&deq)), DEQUANT_FNV, "dequantize_row_q1_0");

        let mut bad = Vec::new();
        for (t, &(avx, generic)) in VECDOT.iter().enumerate() {
            let nb = 1 + (r.next() % 32) as usize;
            let xo = (r.next() % (NB1 - nb + 1) as u64) as usize;
            let yo = (r.next() % (NB1 - nb + 1) as u64) as usize;
            let n = nb * QK1_0;
            let got = BlockQ1_0::vec_dot(n, &x[xo..xo + nb], &y[yo..yo + nb]).to_bits();
            let got_g = BlockQ1_0::vec_dot_unopt(n, &x[xo..xo + nb], &y[yo..yo + nb]).to_bits();
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

        let mut xs = vec![0f32; 96 * QK1_0];
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
        let mut q = vec![BlockQ1_0::zeros(); 96];
        BlockQ1_0::from_float(&xs, &mut q);
        assert_eq!(fnv(bytes(&q)), QUANT_FNV, "quantize_row_q1_0_ref");
    }
}
