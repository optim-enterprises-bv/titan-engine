import os

p = "~/titan-engine/candle/candle-core/src/quantized/iquant.rs"
s = open(p).read()

MAC = '''/// Scalar CPU fallback for the dot product. The model path never uses this: on CUDA these types
/// go through the cuda-oxide MMVQ kernels (oxide-kernels/iq2_xxs, iq3_xxs, iq2_s), which are gated
/// bit-identical to llama.cpp's `vec_dot_iq*_q8_1`. This implementation dequantizes and dots in
/// f32, so it is *not* bit-identical to that reference and exists only so the trait is complete
/// for CPU tensors.
macro_rules! iquant_trait {
    ($t:ty, $dt:expr, $f:ident) => {
        impl GgmlType for $t {
            const DTYPE: GgmlDType = $dt;
            const BLCK_SIZE: usize = QK_K;
            type VecDotType = BlockQ8_1;

            fn to_float(xs: &[Self], ys: &mut [f32]) {
                debug_assert!(
                    ys.len() % QK_K == 0,
                    "dequantize: {} not a multiple of {QK_K}",
                    ys.len()
                );
                for (x, y) in xs.iter().zip(ys.chunks_exact_mut(QK_K)) {
                    $f(x, y);
                }
            }

            fn from_float(_xs: &[f32], _ys: &mut [Self]) {
                unimplemented!(
                    "quantizing to this i-quant is not supported; use a pre-quantized GGUF"
                )
            }

            fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
                Self::vec_dot_unopt(n, xs, ys)
            }

            fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
                debug_assert!(n % QK_K == 0, "vec_dot: {n} not a multiple of {QK_K}");
                let nb = n / QK_K;
                let mut xbuf = vec![0f32; QK_K];
                let mut sumf = 0f32;
                for (i, x) in xs.iter().take(nb).enumerate() {
                    $f(x, &mut xbuf);
                    let mut acc = 0f32;
                    for s in 0..8 {
                        let yb = &ys[i * 8 + s];
                        let d = yb.d.to_f32();
                        for j in 0..32 {
                            acc += xbuf[s * 32 + j] * (d * yb.qs[j] as f32);
                        }
                    }
                    sumf += acc;
                }
                sumf
            }
        }
    };
}

'''

head, sep, _ = s.partition("macro_rules! iquant_trait {")
assert sep, "macro marker not found"
s = head + MAC + """iquant_trait!(BlockIQ2xxs, GgmlDType::IQ2XXS, iq2xxs_block);
iquant_trait!(BlockIQ3xxs, GgmlDType::IQ3XXS, iq3xxs_block);
iquant_trait!(BlockIQ2s, GgmlDType::IQ2S, iq2s_block);
"""
open(p, "w").write(s)
print("trait macro replaced;", os.path.getsize(p), "bytes")
