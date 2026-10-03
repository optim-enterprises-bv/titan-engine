import os

Q = "~/titan-engine/candle/candle-core/src/quantized/"

# --- mod.rs: metal bail + cuda load_quantized -------------------------------
p = Q + "mod.rs"
s = open(p).read()
a = """                GgmlDType::MXFP4 => crate::bail!("MXFP4 is not supported on metal"),
                GgmlDType::NVFP4 => crate::bail!("NVFP4 is not supported on metal"),"""
b = """                GgmlDType::MXFP4 => crate::bail!("MXFP4 is not supported on metal"),
                GgmlDType::NVFP4 => crate::bail!("NVFP4 is not supported on metal"),
                GgmlDType::IQ2XXS | GgmlDType::IQ3XXS | GgmlDType::IQ2S => {
                    crate::bail!("i-quant lookup-table types are not supported on metal")
                }"""
assert a in s, "metal arm"
s = s.replace(a, b, 1)

a = """                GgmlDType::NVFP4 => cuda::load_quantized(d, as_t_slice::<BlockNVFP4>(data)),
                GgmlDType::BF16 => cuda::load_quantized(d, as_t_slice::<bf16>(data)),"""
b = """                GgmlDType::NVFP4 => cuda::load_quantized(d, as_t_slice::<BlockNVFP4>(data)),
                GgmlDType::IQ2XXS => cuda::load_quantized(d, as_t_slice::<BlockIQ2xxs>(data)),
                GgmlDType::IQ3XXS => cuda::load_quantized(d, as_t_slice::<BlockIQ3xxs>(data)),
                GgmlDType::IQ2S => cuda::load_quantized(d, as_t_slice::<BlockIQ2s>(data)),
                GgmlDType::BF16 => cuda::load_quantized(d, as_t_slice::<bf16>(data)),"""
assert a in s, "cuda load arm"
s = s.replace(a, b, 1)
open(p, "w").write(s)
print("mod.rs patched")

# --- cuda.rs: the f32 dequant fallback + the fast-kernel allow list ----------
p = Q + "cuda.rs"
s = open(p).read()
a = """            GgmlDType::NVFP4 => deq::<crate::quantized::BlockNVFP4>(&buffer, block_len, &mut out),
        }"""
b = """            GgmlDType::NVFP4 => deq::<crate::quantized::BlockNVFP4>(&buffer, block_len, &mut out),
            GgmlDType::IQ2XXS => deq::<crate::quantized::BlockIQ2xxs>(&buffer, block_len, &mut out),
            GgmlDType::IQ3XXS => deq::<crate::quantized::BlockIQ3xxs>(&buffer, block_len, &mut out),
            GgmlDType::IQ2S => deq::<crate::quantized::BlockIQ2s>(&buffer, block_len, &mut out),
        }"""
assert a in s, "cuda deq arm"
s = s.replace(a, b, 1)
open(p, "w").write(s)
print("cuda.rs patched")
