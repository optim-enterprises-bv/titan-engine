import os
Q = "~/titan-engine/candle/candle-core/src/quantized/"

# ---- mod.rs: IQ4XS / IQ2XS variants + every exhaustive dispatch -------------
p = Q + "mod.rs"
s = open(p).read()
reps = [
 ("""    /// GGML type 22: 256 weights, f16 scale + 256 grid-codebook indices and 8 group scales (IQ2_S).
    IQ2S,
}""",
  """    /// GGML type 22: 256 weights, f16 scale + 256 grid-codebook indices and 8 group scales (IQ2_S).
    IQ2S,
    /// GGML type 23: 256 weights, f16 scale + 6-bit group scales + 128 nibble bytes (IQ4_XS).
    IQ4XS,
    /// GGML type 17: 256 weights, f16 scale + 512-entry 9-bit grid codes + 4-bit group scales.
    IQ2XS,
}"""),
 ("""            22 => Self::IQ2S,
            _ => crate::bail!("unknown dtype for tensor {u}"),""",
  """            22 => Self::IQ2S,
            23 => Self::IQ4XS,
            17 => Self::IQ2XS,
            _ => crate::bail!("unknown dtype for tensor {u}"),"""),
 ("""            Self::IQ2S => 22,""", """            Self::IQ2S => 22,
            Self::IQ4XS => 23,
            Self::IQ2XS => 17,"""),
 ("""            Self::IQ2S => Box::new(vec![BlockIQ2s::zeros(); elem_count / QK_K]),""",
  """            Self::IQ2S => Box::new(vec![BlockIQ2s::zeros(); elem_count / QK_K]),
            Self::IQ4XS => Box::new(vec![BlockIQ4xs::zeros(); elem_count / QK_K]),
            Self::IQ2XS => Box::new(vec![BlockIQ2xs::zeros(); elem_count / QK_K]),"""),
 ("""            Self::IQ2S => Box::new(as_t_slice::<BlockIQ2s>(data).to_vec()),""",
  """            Self::IQ2S => Box::new(as_t_slice::<BlockIQ2s>(data).to_vec()),
            Self::IQ4XS => Box::new(as_t_slice::<BlockIQ4xs>(data).to_vec()),
            Self::IQ2XS => Box::new(as_t_slice::<BlockIQ2xs>(data).to_vec()),"""),
 ("""            Self::IQ2S => std::mem::size_of::<BlockIQ2s>(),""",
  """            Self::IQ2S => std::mem::size_of::<BlockIQ2s>(),
            Self::IQ4XS => std::mem::size_of::<BlockIQ4xs>(),
            Self::IQ2XS => std::mem::size_of::<BlockIQ2xs>(),"""),
 ("""            Self::IQ2XXS | Self::IQ3XXS | Self::IQ2S => QK_K,""",
  """            Self::IQ2XXS | Self::IQ3XXS | Self::IQ2S | Self::IQ4XS | Self::IQ2XS => QK_K,"""),
 ("""pub use iquant::{BlockIQ2s, BlockIQ2xxs, BlockIQ3xxs, QK_K as QK_K_IQ};""",
  """pub use iquant::{BlockIQ2s, BlockIQ2xs, BlockIQ2xxs, BlockIQ3xxs, BlockIQ4xs, QK_K as QK_K_IQ};"""),
 ("""                GgmlDType::IQ2XXS | GgmlDType::IQ3XXS | GgmlDType::IQ2S => {
                    crate::bail!("i-quant lookup-table types are not supported on metal")
                }""",
  """                GgmlDType::IQ2XXS
                | GgmlDType::IQ3XXS
                | GgmlDType::IQ2S
                | GgmlDType::IQ4XS
                | GgmlDType::IQ2XS => {
                    crate::bail!("i-quant lookup-table types are not supported on metal")
                }"""),
 ("""                GgmlDType::IQ2S => cuda::load_quantized(d, as_t_slice::<BlockIQ2s>(data)),""",
  """                GgmlDType::IQ2S => cuda::load_quantized(d, as_t_slice::<BlockIQ2s>(data)),
                GgmlDType::IQ4XS => cuda::load_quantized(d, as_t_slice::<BlockIQ4xs>(data)),
                GgmlDType::IQ2XS => cuda::load_quantized(d, as_t_slice::<BlockIQ2xs>(data)),"""),
]
for a, b in reps:
    assert a in s, a[:70]
    s = s.replace(a, b, 1)
open(p, "w").write(s)
print("mod.rs patched")

# ---- cuda.rs: dequant arms ---------------------------------------------
p = Q + "cuda.rs"
s = open(p).read()
a = """            GgmlDType::IQ2S => deq::<crate::quantized::BlockIQ2s>(&buffer, block_len, &mut out),"""
b = """            GgmlDType::IQ2S => deq::<crate::quantized::BlockIQ2s>(&buffer, block_len, &mut out),
            GgmlDType::IQ4XS => deq::<crate::quantized::BlockIQ4xs>(&buffer, block_len, &mut out),
            GgmlDType::IQ2XS => deq::<crate::quantized::BlockIQ2xs>(&buffer, block_len, &mut out),"""
assert a in s
s = s.replace(a, b, 1)
open(p, "w").write(s)
print("cuda.rs deq patched")

# ---- mistralrs-quant: both dtype maps ----------------------------------
p = "~/titan-engine/mistral.rs/mistralrs-quant/src/gguf/mod.rs"
s = open(p).read()
s = s.replace("""        GgmlDType::IQ2S => 22,""", """        GgmlDType::IQ2S => 22,
        GgmlDType::IQ4XS => 23,
        GgmlDType::IQ2XS => 17,""", 1)
s = s.replace("""        22 => Ok(GgmlDType::IQ2S),""", """        22 => Ok(GgmlDType::IQ2S),
        23 => Ok(GgmlDType::IQ4XS),
        17 => Ok(GgmlDType::IQ2XS),""", 1)
open(p, "w").write(s)
print("gguf/mod.rs patched")

p = "~/titan-engine/mistral.rs/mistralrs-quant/src/lib.rs"
s = open(p).read()
s = s.replace("""            | GgmlDType::IQ2S => {
                candle_core::bail!("Expected valid GGML ISQ type.")
            }""", """            | GgmlDType::IQ2S
            | GgmlDType::IQ4XS
            | GgmlDType::IQ2XS => {
                candle_core::bail!("Expected valid GGML ISQ type.")
            }""", 1)
open(p, "w").write(s)
print("lib.rs patched")
