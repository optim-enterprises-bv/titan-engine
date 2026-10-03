//! GGML IQ3_XXS (type 18) CUDA kernels in cuda-oxide, ported from llama.cpp acecd56 under the
//! reference's own parameter ABI and bit-identical to its nvcc build (sm_120a; the cubins in ref/
//! are the ones the iq4_nl / iq4_xs ports gate against):
//!
//! - `iq3_xxs_dequant_{f32,f16,bf16}` = `dequantize_block_iq3_xxs<dst_t>` (convert.cu, grid
//!   ceil(k/256), block 32). `d = block_d * (0.5 + (aux>>28)) * 0.5` (a power of two, so the two
//!   fast-math multiplies are exact); values are `d * grid[j]` negated where the group's sign bit
//!   is set (`ksigns_iq2xs[(aux>>7*il) & 127]` vs `kmask_iq2xs[j]`, `j=0..7`).
//! - `iq3_xxs_mmvq_<n>` = `mul_mat_vec_q<GGML_TYPE_IQ3_XXS, n, has_fusion=false, small_k, false>`
//!   (mmvq.cu GENERIC table): block (32, nwarps), nwarps 4 for n <= 4 / 2 for n 5..8, rows per
//!   CUDA block 1 for n == 1 (nwarps with small_k) and 2 otherwise. qk 256, QR3_XXS 4, qi 16,
//!   vdr 2 -> `blocks_per_iter = vdr*nwarps*32/qi` = 8 (n <= 4) / 4 (n 5..8), and
//!   `kqs = vdr*(tid % (qi/vdr)) = 2*(tid % 8)`.
//!   vec_dot_iq3_xxs_q8_1: `q3_packed` = the two u32 at `qs + 4*iqs`; `aux32` = the u32 at
//!   `qs + 64 + 4*(iqs/2)`. For l0 = 0,2,4,6: grid entries `q3[l0]` / `q3[l0+1]`, sign selector
//!   `unpack_ksigns(aux32 >> (7*l0/2))`, `grid_l = __vsub4(grid ^ signs0, signs0)` against the
//!   q8_1 words `l0` and `l0+1` of block `iqs/2`; finally `sumi = (ls*sumi + sumi/2)/2` with
//!   `ls = aux32 >> 28`, and `d = block_d * q8_1[iqs/2].d`. The caller accumulates in one FFMA.
//!
//! The grid codebook is kept as its 256 little-endian `u32` words read through `.to_le_bytes()`
//! rather than as a `u8` array: Rust lowers a runtime-indexed array read to local-memory LDL, while
//! a word read becomes a select chain (see PORTING.md #43 / #175).
#![allow(non_snake_case, clippy::missing_safety_doc, clippy::too_many_arguments)]
mod gate;

use cuda_device::{SharedArray, convert, device, dotprod, kernel, ptx_asm, thread, warp};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// `grid_word`: explicit compare chain; a runtime-indexed Rust array would spill to local
    /// memory (PORTING.md #43) and a host-side table never reaches the PTX at all.
    #[inline(always)]
    pub fn grid_word(i: u32) -> u32 {
        let mut v: u32 = 0;
        if i == 0u32 { v = 0x04040404u32 as u32; }
        if i == 1u32 { v = 0x04040414u32 as u32; }
        if i == 2u32 { v = 0x04040424u32 as u32; }
        if i == 3u32 { v = 0x04040c0cu32 as u32; }
        if i == 4u32 { v = 0x04040c1cu32 as u32; }
        if i == 5u32 { v = 0x04040c3eu32 as u32; }
        if i == 6u32 { v = 0x04041404u32 as u32; }
        if i == 7u32 { v = 0x04041414u32 as u32; }
        if i == 8u32 { v = 0x04041c0cu32 as u32; }
        if i == 9u32 { v = 0x04042414u32 as u32; }
        if i == 10u32 { v = 0x04043e1cu32 as u32; }
        if i == 11u32 { v = 0x04043e2cu32 as u32; }
        if i == 12u32 { v = 0x040c040cu32 as u32; }
        if i == 13u32 { v = 0x040c041cu32 as u32; }
        if i == 14u32 { v = 0x040c0c04u32 as u32; }
        if i == 15u32 { v = 0x040c0c14u32 as u32; }
        if i == 16u32 { v = 0x040c140cu32 as u32; }
        if i == 17u32 { v = 0x040c142cu32 as u32; }
        if i == 18u32 { v = 0x040c1c04u32 as u32; }
        if i == 19u32 { v = 0x040c1c14u32 as u32; }
        if i == 20u32 { v = 0x040c240cu32 as u32; }
        if i == 21u32 { v = 0x040c2c24u32 as u32; }
        if i == 22u32 { v = 0x040c3e04u32 as u32; }
        if i == 23u32 { v = 0x04140404u32 as u32; }
        if i == 24u32 { v = 0x04140414u32 as u32; }
        if i == 25u32 { v = 0x04140424u32 as u32; }
        if i == 26u32 { v = 0x04140c0cu32 as u32; }
        if i == 27u32 { v = 0x04141404u32 as u32; }
        if i == 28u32 { v = 0x04141414u32 as u32; }
        if i == 29u32 { v = 0x04141c0cu32 as u32; }
        if i == 30u32 { v = 0x04141c1cu32 as u32; }
        if i == 31u32 { v = 0x04141c3eu32 as u32; }
        if i == 32u32 { v = 0x04142c0cu32 as u32; }
        if i == 33u32 { v = 0x04142c3eu32 as u32; }
        if i == 34u32 { v = 0x04143e2cu32 as u32; }
        if i == 35u32 { v = 0x041c040cu32 as u32; }
        if i == 36u32 { v = 0x041c043eu32 as u32; }
        if i == 37u32 { v = 0x041c0c04u32 as u32; }
        if i == 38u32 { v = 0x041c0c14u32 as u32; }
        if i == 39u32 { v = 0x041c142cu32 as u32; }
        if i == 40u32 { v = 0x041c3e04u32 as u32; }
        if i == 41u32 { v = 0x04240c1cu32 as u32; }
        if i == 42u32 { v = 0x04241c3eu32 as u32; }
        if i == 43u32 { v = 0x04242424u32 as u32; }
        if i == 44u32 { v = 0x04242c3eu32 as u32; }
        if i == 45u32 { v = 0x04243e1cu32 as u32; }
        if i == 46u32 { v = 0x04243e2cu32 as u32; }
        if i == 47u32 { v = 0x042c040cu32 as u32; }
        if i == 48u32 { v = 0x042c043eu32 as u32; }
        if i == 49u32 { v = 0x042c1c14u32 as u32; }
        if i == 50u32 { v = 0x042c2c14u32 as u32; }
        if i == 51u32 { v = 0x04341c2cu32 as u32; }
        if i == 52u32 { v = 0x04343424u32 as u32; }
        if i == 53u32 { v = 0x043e0c04u32 as u32; }
        if i == 54u32 { v = 0x043e0c24u32 as u32; }
        if i == 55u32 { v = 0x043e0c34u32 as u32; }
        if i == 56u32 { v = 0x043e241cu32 as u32; }
        if i == 57u32 { v = 0x043e340cu32 as u32; }
        if i == 58u32 { v = 0x0c04040cu32 as u32; }
        if i == 59u32 { v = 0x0c04041cu32 as u32; }
        if i == 60u32 { v = 0x0c040c04u32 as u32; }
        if i == 61u32 { v = 0x0c040c14u32 as u32; }
        if i == 62u32 { v = 0x0c04140cu32 as u32; }
        if i == 63u32 { v = 0x0c04141cu32 as u32; }
        if i == 64u32 { v = 0x0c041c04u32 as u32; }
        if i == 65u32 { v = 0x0c041c14u32 as u32; }
        if i == 66u32 { v = 0x0c041c24u32 as u32; }
        if i == 67u32 { v = 0x0c04243eu32 as u32; }
        if i == 68u32 { v = 0x0c042c04u32 as u32; }
        if i == 69u32 { v = 0x0c0c0404u32 as u32; }
        if i == 70u32 { v = 0x0c0c0414u32 as u32; }
        if i == 71u32 { v = 0x0c0c0c0cu32 as u32; }
        if i == 72u32 { v = 0x0c0c1404u32 as u32; }
        if i == 73u32 { v = 0x0c0c1414u32 as u32; }
        if i == 74u32 { v = 0x0c14040cu32 as u32; }
        if i == 75u32 { v = 0x0c14041cu32 as u32; }
        if i == 76u32 { v = 0x0c140c04u32 as u32; }
        if i == 77u32 { v = 0x0c140c14u32 as u32; }
        if i == 78u32 { v = 0x0c14140cu32 as u32; }
        if i == 79u32 { v = 0x0c141c04u32 as u32; }
        if i == 80u32 { v = 0x0c143e14u32 as u32; }
        if i == 81u32 { v = 0x0c1c0404u32 as u32; }
        if i == 82u32 { v = 0x0c1c0414u32 as u32; }
        if i == 83u32 { v = 0x0c1c1404u32 as u32; }
        if i == 84u32 { v = 0x0c1c1c0cu32 as u32; }
        if i == 85u32 { v = 0x0c1c2434u32 as u32; }
        if i == 86u32 { v = 0x0c1c3434u32 as u32; }
        if i == 87u32 { v = 0x0c24040cu32 as u32; }
        if i == 88u32 { v = 0x0c24042cu32 as u32; }
        if i == 89u32 { v = 0x0c242c04u32 as u32; }
        if i == 90u32 { v = 0x0c2c1404u32 as u32; }
        if i == 91u32 { v = 0x0c2c1424u32 as u32; }
        if i == 92u32 { v = 0x0c2c2434u32 as u32; }
        if i == 93u32 { v = 0x0c2c3e0cu32 as u32; }
        if i == 94u32 { v = 0x0c34042cu32 as u32; }
        if i == 95u32 { v = 0x0c3e1414u32 as u32; }
        if i == 96u32 { v = 0x0c3e2404u32 as u32; }
        if i == 97u32 { v = 0x14040404u32 as u32; }
        if i == 98u32 { v = 0x14040414u32 as u32; }
        if i == 99u32 { v = 0x14040c0cu32 as u32; }
        if i == 100u32 { v = 0x14040c1cu32 as u32; }
        if i == 101u32 { v = 0x14041404u32 as u32; }
        if i == 102u32 { v = 0x14041414u32 as u32; }
        if i == 103u32 { v = 0x14041434u32 as u32; }
        if i == 104u32 { v = 0x14041c0cu32 as u32; }
        if i == 105u32 { v = 0x14042414u32 as u32; }
        if i == 106u32 { v = 0x140c040cu32 as u32; }
        if i == 107u32 { v = 0x140c041cu32 as u32; }
        if i == 108u32 { v = 0x140c042cu32 as u32; }
        if i == 109u32 { v = 0x140c0c04u32 as u32; }
        if i == 110u32 { v = 0x140c0c14u32 as u32; }
        if i == 111u32 { v = 0x140c140cu32 as u32; }
        if i == 112u32 { v = 0x140c1c04u32 as u32; }
        if i == 113u32 { v = 0x140c341cu32 as u32; }
        if i == 114u32 { v = 0x140c343eu32 as u32; }
        if i == 115u32 { v = 0x140c3e04u32 as u32; }
        if i == 116u32 { v = 0x14140404u32 as u32; }
        if i == 117u32 { v = 0x14140414u32 as u32; }
        if i == 118u32 { v = 0x14140c0cu32 as u32; }
        if i == 119u32 { v = 0x14140c3eu32 as u32; }
        if i == 120u32 { v = 0x14141404u32 as u32; }
        if i == 121u32 { v = 0x14141414u32 as u32; }
        if i == 122u32 { v = 0x14141c3eu32 as u32; }
        if i == 123u32 { v = 0x14142404u32 as u32; }
        if i == 124u32 { v = 0x14142c2cu32 as u32; }
        if i == 125u32 { v = 0x141c040cu32 as u32; }
        if i == 126u32 { v = 0x141c0c04u32 as u32; }
        if i == 127u32 { v = 0x141c0c24u32 as u32; }
        if i == 128u32 { v = 0x141c3e04u32 as u32; }
        if i == 129u32 { v = 0x141c3e24u32 as u32; }
        if i == 130u32 { v = 0x14241c2cu32 as u32; }
        if i == 131u32 { v = 0x14242c1cu32 as u32; }
        if i == 132u32 { v = 0x142c041cu32 as u32; }
        if i == 133u32 { v = 0x142c143eu32 as u32; }
        if i == 134u32 { v = 0x142c240cu32 as u32; }
        if i == 135u32 { v = 0x142c3e24u32 as u32; }
        if i == 136u32 { v = 0x143e040cu32 as u32; }
        if i == 137u32 { v = 0x143e041cu32 as u32; }
        if i == 138u32 { v = 0x143e0c34u32 as u32; }
        if i == 139u32 { v = 0x143e242cu32 as u32; }
        if i == 140u32 { v = 0x1c04040cu32 as u32; }
        if i == 141u32 { v = 0x1c040c04u32 as u32; }
        if i == 142u32 { v = 0x1c040c14u32 as u32; }
        if i == 143u32 { v = 0x1c04140cu32 as u32; }
        if i == 144u32 { v = 0x1c04141cu32 as u32; }
        if i == 145u32 { v = 0x1c042c04u32 as u32; }
        if i == 146u32 { v = 0x1c04342cu32 as u32; }
        if i == 147u32 { v = 0x1c043e14u32 as u32; }
        if i == 148u32 { v = 0x1c0c0404u32 as u32; }
        if i == 149u32 { v = 0x1c0c0414u32 as u32; }
        if i == 150u32 { v = 0x1c0c1404u32 as u32; }
        if i == 151u32 { v = 0x1c0c1c0cu32 as u32; }
        if i == 152u32 { v = 0x1c0c2424u32 as u32; }
        if i == 153u32 { v = 0x1c0c2434u32 as u32; }
        if i == 154u32 { v = 0x1c14040cu32 as u32; }
        if i == 155u32 { v = 0x1c14041cu32 as u32; }
        if i == 156u32 { v = 0x1c140c04u32 as u32; }
        if i == 157u32 { v = 0x1c14142cu32 as u32; }
        if i == 158u32 { v = 0x1c142c14u32 as u32; }
        if i == 159u32 { v = 0x1c143e14u32 as u32; }
        if i == 160u32 { v = 0x1c1c0c0cu32 as u32; }
        if i == 161u32 { v = 0x1c1c1c1cu32 as u32; }
        if i == 162u32 { v = 0x1c241c04u32 as u32; }
        if i == 163u32 { v = 0x1c24243eu32 as u32; }
        if i == 164u32 { v = 0x1c243e14u32 as u32; }
        if i == 165u32 { v = 0x1c2c0404u32 as u32; }
        if i == 166u32 { v = 0x1c2c0434u32 as u32; }
        if i == 167u32 { v = 0x1c2c1414u32 as u32; }
        if i == 168u32 { v = 0x1c2c2c2cu32 as u32; }
        if i == 169u32 { v = 0x1c340c24u32 as u32; }
        if i == 170u32 { v = 0x1c341c34u32 as u32; }
        if i == 171u32 { v = 0x1c34341cu32 as u32; }
        if i == 172u32 { v = 0x1c3e1c1cu32 as u32; }
        if i == 173u32 { v = 0x1c3e3404u32 as u32; }
        if i == 174u32 { v = 0x24040424u32 as u32; }
        if i == 175u32 { v = 0x24040c3eu32 as u32; }
        if i == 176u32 { v = 0x24041c2cu32 as u32; }
        if i == 177u32 { v = 0x24041c3eu32 as u32; }
        if i == 178u32 { v = 0x24042c1cu32 as u32; }
        if i == 179u32 { v = 0x24042c3eu32 as u32; }
        if i == 180u32 { v = 0x240c3e24u32 as u32; }
        if i == 181u32 { v = 0x24141404u32 as u32; }
        if i == 182u32 { v = 0x24141c3eu32 as u32; }
        if i == 183u32 { v = 0x24142404u32 as u32; }
        if i == 184u32 { v = 0x24143404u32 as u32; }
        if i == 185u32 { v = 0x24143434u32 as u32; }
        if i == 186u32 { v = 0x241c043eu32 as u32; }
        if i == 187u32 { v = 0x241c242cu32 as u32; }
        if i == 188u32 { v = 0x24240424u32 as u32; }
        if i == 189u32 { v = 0x24242c0cu32 as u32; }
        if i == 190u32 { v = 0x24243424u32 as u32; }
        if i == 191u32 { v = 0x242c142cu32 as u32; }
        if i == 192u32 { v = 0x242c241cu32 as u32; }
        if i == 193u32 { v = 0x242c3e04u32 as u32; }
        if i == 194u32 { v = 0x243e042cu32 as u32; }
        if i == 195u32 { v = 0x243e0c04u32 as u32; }
        if i == 196u32 { v = 0x243e0c14u32 as u32; }
        if i == 197u32 { v = 0x243e1c04u32 as u32; }
        if i == 198u32 { v = 0x2c040c14u32 as u32; }
        if i == 199u32 { v = 0x2c04240cu32 as u32; }
        if i == 200u32 { v = 0x2c043e04u32 as u32; }
        if i == 201u32 { v = 0x2c0c0404u32 as u32; }
        if i == 202u32 { v = 0x2c0c0434u32 as u32; }
        if i == 203u32 { v = 0x2c0c1434u32 as u32; }
        if i == 204u32 { v = 0x2c0c2c2cu32 as u32; }
        if i == 205u32 { v = 0x2c140c24u32 as u32; }
        if i == 206u32 { v = 0x2c141c14u32 as u32; }
        if i == 207u32 { v = 0x2c143e14u32 as u32; }
        if i == 208u32 { v = 0x2c1c0414u32 as u32; }
        if i == 209u32 { v = 0x2c1c2c1cu32 as u32; }
        if i == 210u32 { v = 0x2c240c04u32 as u32; }
        if i == 211u32 { v = 0x2c24141cu32 as u32; }
        if i == 212u32 { v = 0x2c24143eu32 as u32; }
        if i == 213u32 { v = 0x2c243e14u32 as u32; }
        if i == 214u32 { v = 0x2c2c0414u32 as u32; }
        if i == 215u32 { v = 0x2c2c1c0cu32 as u32; }
        if i == 216u32 { v = 0x2c342c04u32 as u32; }
        if i == 217u32 { v = 0x2c3e1424u32 as u32; }
        if i == 218u32 { v = 0x2c3e2414u32 as u32; }
        if i == 219u32 { v = 0x34041424u32 as u32; }
        if i == 220u32 { v = 0x34042424u32 as u32; }
        if i == 221u32 { v = 0x34042434u32 as u32; }
        if i == 222u32 { v = 0x34043424u32 as u32; }
        if i == 223u32 { v = 0x340c140cu32 as u32; }
        if i == 224u32 { v = 0x340c340cu32 as u32; }
        if i == 225u32 { v = 0x34140c3eu32 as u32; }
        if i == 226u32 { v = 0x34143424u32 as u32; }
        if i == 227u32 { v = 0x341c1c04u32 as u32; }
        if i == 228u32 { v = 0x341c1c34u32 as u32; }
        if i == 229u32 { v = 0x34242424u32 as u32; }
        if i == 230u32 { v = 0x342c042cu32 as u32; }
        if i == 231u32 { v = 0x342c2c14u32 as u32; }
        if i == 232u32 { v = 0x34341c1cu32 as u32; }
        if i == 233u32 { v = 0x343e041cu32 as u32; }
        if i == 234u32 { v = 0x343e140cu32 as u32; }
        if i == 235u32 { v = 0x3e04041cu32 as u32; }
        if i == 236u32 { v = 0x3e04042cu32 as u32; }
        if i == 237u32 { v = 0x3e04043eu32 as u32; }
        if i == 238u32 { v = 0x3e040c04u32 as u32; }
        if i == 239u32 { v = 0x3e041c14u32 as u32; }
        if i == 240u32 { v = 0x3e042c14u32 as u32; }
        if i == 241u32 { v = 0x3e0c1434u32 as u32; }
        if i == 242u32 { v = 0x3e0c2404u32 as u32; }
        if i == 243u32 { v = 0x3e140c14u32 as u32; }
        if i == 244u32 { v = 0x3e14242cu32 as u32; }
        if i == 245u32 { v = 0x3e142c14u32 as u32; }
        if i == 246u32 { v = 0x3e1c0404u32 as u32; }
        if i == 247u32 { v = 0x3e1c0c2cu32 as u32; }
        if i == 248u32 { v = 0x3e1c1c1cu32 as u32; }
        if i == 249u32 { v = 0x3e1c3404u32 as u32; }
        if i == 250u32 { v = 0x3e24140cu32 as u32; }
        if i == 251u32 { v = 0x3e24240cu32 as u32; }
        if i == 252u32 { v = 0x3e2c0404u32 as u32; }
        if i == 253u32 { v = 0x3e2c0414u32 as u32; }
        if i == 254u32 { v = 0x3e2c1424u32 as u32; }
        if i == 255u32 { v = 0x3e341c04u32 as u32; }
        v
    }
    /// `ksign`: explicit compare chain; a runtime-indexed Rust array would spill to local
    /// memory (PORTING.md #43) and a host-side table never reaches the PTX at all.
    #[inline(always)]
    pub fn ksign(i: u32) -> u8 {
        let mut v: u8 = 0;
        if i == 0u32 { v = 0x00000000u32 as u8; }
        if i == 1u32 { v = 0x00000081u32 as u8; }
        if i == 2u32 { v = 0x00000082u32 as u8; }
        if i == 3u32 { v = 0x00000003u32 as u8; }
        if i == 4u32 { v = 0x00000084u32 as u8; }
        if i == 5u32 { v = 0x00000005u32 as u8; }
        if i == 6u32 { v = 0x00000006u32 as u8; }
        if i == 7u32 { v = 0x00000087u32 as u8; }
        if i == 8u32 { v = 0x00000088u32 as u8; }
        if i == 9u32 { v = 0x00000009u32 as u8; }
        if i == 10u32 { v = 0x0000000au32 as u8; }
        if i == 11u32 { v = 0x0000008bu32 as u8; }
        if i == 12u32 { v = 0x0000000cu32 as u8; }
        if i == 13u32 { v = 0x0000008du32 as u8; }
        if i == 14u32 { v = 0x0000008eu32 as u8; }
        if i == 15u32 { v = 0x0000000fu32 as u8; }
        if i == 16u32 { v = 0x00000090u32 as u8; }
        if i == 17u32 { v = 0x00000011u32 as u8; }
        if i == 18u32 { v = 0x00000012u32 as u8; }
        if i == 19u32 { v = 0x00000093u32 as u8; }
        if i == 20u32 { v = 0x00000014u32 as u8; }
        if i == 21u32 { v = 0x00000095u32 as u8; }
        if i == 22u32 { v = 0x00000096u32 as u8; }
        if i == 23u32 { v = 0x00000017u32 as u8; }
        if i == 24u32 { v = 0x00000018u32 as u8; }
        if i == 25u32 { v = 0x00000099u32 as u8; }
        if i == 26u32 { v = 0x0000009au32 as u8; }
        if i == 27u32 { v = 0x0000001bu32 as u8; }
        if i == 28u32 { v = 0x0000009cu32 as u8; }
        if i == 29u32 { v = 0x0000001du32 as u8; }
        if i == 30u32 { v = 0x0000001eu32 as u8; }
        if i == 31u32 { v = 0x0000009fu32 as u8; }
        if i == 32u32 { v = 0x000000a0u32 as u8; }
        if i == 33u32 { v = 0x00000021u32 as u8; }
        if i == 34u32 { v = 0x00000022u32 as u8; }
        if i == 35u32 { v = 0x000000a3u32 as u8; }
        if i == 36u32 { v = 0x00000024u32 as u8; }
        if i == 37u32 { v = 0x000000a5u32 as u8; }
        if i == 38u32 { v = 0x000000a6u32 as u8; }
        if i == 39u32 { v = 0x00000027u32 as u8; }
        if i == 40u32 { v = 0x00000028u32 as u8; }
        if i == 41u32 { v = 0x000000a9u32 as u8; }
        if i == 42u32 { v = 0x000000aau32 as u8; }
        if i == 43u32 { v = 0x0000002bu32 as u8; }
        if i == 44u32 { v = 0x000000acu32 as u8; }
        if i == 45u32 { v = 0x0000002du32 as u8; }
        if i == 46u32 { v = 0x0000002eu32 as u8; }
        if i == 47u32 { v = 0x000000afu32 as u8; }
        if i == 48u32 { v = 0x00000030u32 as u8; }
        if i == 49u32 { v = 0x000000b1u32 as u8; }
        if i == 50u32 { v = 0x000000b2u32 as u8; }
        if i == 51u32 { v = 0x00000033u32 as u8; }
        if i == 52u32 { v = 0x000000b4u32 as u8; }
        if i == 53u32 { v = 0x00000035u32 as u8; }
        if i == 54u32 { v = 0x00000036u32 as u8; }
        if i == 55u32 { v = 0x000000b7u32 as u8; }
        if i == 56u32 { v = 0x000000b8u32 as u8; }
        if i == 57u32 { v = 0x00000039u32 as u8; }
        if i == 58u32 { v = 0x0000003au32 as u8; }
        if i == 59u32 { v = 0x000000bbu32 as u8; }
        if i == 60u32 { v = 0x0000003cu32 as u8; }
        if i == 61u32 { v = 0x000000bdu32 as u8; }
        if i == 62u32 { v = 0x000000beu32 as u8; }
        if i == 63u32 { v = 0x0000003fu32 as u8; }
        if i == 64u32 { v = 0x000000c0u32 as u8; }
        if i == 65u32 { v = 0x00000041u32 as u8; }
        if i == 66u32 { v = 0x00000042u32 as u8; }
        if i == 67u32 { v = 0x000000c3u32 as u8; }
        if i == 68u32 { v = 0x00000044u32 as u8; }
        if i == 69u32 { v = 0x000000c5u32 as u8; }
        if i == 70u32 { v = 0x000000c6u32 as u8; }
        if i == 71u32 { v = 0x00000047u32 as u8; }
        if i == 72u32 { v = 0x00000048u32 as u8; }
        if i == 73u32 { v = 0x000000c9u32 as u8; }
        if i == 74u32 { v = 0x000000cau32 as u8; }
        if i == 75u32 { v = 0x0000004bu32 as u8; }
        if i == 76u32 { v = 0x000000ccu32 as u8; }
        if i == 77u32 { v = 0x0000004du32 as u8; }
        if i == 78u32 { v = 0x0000004eu32 as u8; }
        if i == 79u32 { v = 0x000000cfu32 as u8; }
        if i == 80u32 { v = 0x00000050u32 as u8; }
        if i == 81u32 { v = 0x000000d1u32 as u8; }
        if i == 82u32 { v = 0x000000d2u32 as u8; }
        if i == 83u32 { v = 0x00000053u32 as u8; }
        if i == 84u32 { v = 0x000000d4u32 as u8; }
        if i == 85u32 { v = 0x00000055u32 as u8; }
        if i == 86u32 { v = 0x00000056u32 as u8; }
        if i == 87u32 { v = 0x000000d7u32 as u8; }
        if i == 88u32 { v = 0x000000d8u32 as u8; }
        if i == 89u32 { v = 0x00000059u32 as u8; }
        if i == 90u32 { v = 0x0000005au32 as u8; }
        if i == 91u32 { v = 0x000000dbu32 as u8; }
        if i == 92u32 { v = 0x0000005cu32 as u8; }
        if i == 93u32 { v = 0x000000ddu32 as u8; }
        if i == 94u32 { v = 0x000000deu32 as u8; }
        if i == 95u32 { v = 0x0000005fu32 as u8; }
        if i == 96u32 { v = 0x00000060u32 as u8; }
        if i == 97u32 { v = 0x000000e1u32 as u8; }
        if i == 98u32 { v = 0x000000e2u32 as u8; }
        if i == 99u32 { v = 0x00000063u32 as u8; }
        if i == 100u32 { v = 0x000000e4u32 as u8; }
        if i == 101u32 { v = 0x00000065u32 as u8; }
        if i == 102u32 { v = 0x00000066u32 as u8; }
        if i == 103u32 { v = 0x000000e7u32 as u8; }
        if i == 104u32 { v = 0x000000e8u32 as u8; }
        if i == 105u32 { v = 0x00000069u32 as u8; }
        if i == 106u32 { v = 0x0000006au32 as u8; }
        if i == 107u32 { v = 0x000000ebu32 as u8; }
        if i == 108u32 { v = 0x0000006cu32 as u8; }
        if i == 109u32 { v = 0x000000edu32 as u8; }
        if i == 110u32 { v = 0x000000eeu32 as u8; }
        if i == 111u32 { v = 0x0000006fu32 as u8; }
        if i == 112u32 { v = 0x000000f0u32 as u8; }
        if i == 113u32 { v = 0x00000071u32 as u8; }
        if i == 114u32 { v = 0x00000072u32 as u8; }
        if i == 115u32 { v = 0x000000f3u32 as u8; }
        if i == 116u32 { v = 0x00000074u32 as u8; }
        if i == 117u32 { v = 0x000000f5u32 as u8; }
        if i == 118u32 { v = 0x000000f6u32 as u8; }
        if i == 119u32 { v = 0x00000077u32 as u8; }
        if i == 120u32 { v = 0x00000078u32 as u8; }
        if i == 121u32 { v = 0x000000f9u32 as u8; }
        if i == 122u32 { v = 0x000000fau32 as u8; }
        if i == 123u32 { v = 0x0000007bu32 as u8; }
        if i == 124u32 { v = 0x000000fcu32 as u8; }
        if i == 125u32 { v = 0x0000007du32 as u8; }
        if i == 126u32 { v = 0x0000007eu32 as u8; }
        if i == 127u32 { v = 0x000000ffu32 as u8; }
        v
    }
    /// IQ3_XXS grid words: `iq3xxs_grid[256]` verbatim (values 1..127, 8 bytes per entry).
    /// The 8 bytes of an entry are read out with `to_le_bytes()`.
    /// `kmask_iq2xs[8]` is indexed only by `j`/`j+4` inside a fully unrolled loop, so it stays a
    /// plain array.
    pub const KMASK: [u8; 8] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80];

    #[inline(always)]
    pub fn h2f(bits: u32) -> f32 {
        convert::cvt_f32_f16x2_lo(bits & 0xFFFF)
    }
    /// fast-math `a * b` [FMUL.FTZ].
    #[inline(always)]
    pub fn mul(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("mul.rn.ftz.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    /// fast-math `a + b` [FADD.FTZ].
    #[inline(always)]
    pub fn add(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("add.rn.ftz.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    /// fast-math contraction [FFMA.FTZ]: the caller's `tmp += vec_dot(...)` is one FFMA.
    #[inline(always)]
    pub fn fma(a: f32, b: f32, c: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("fma.rn.ftz.f32 %0, %1, %2, %3;", out("=f") r, in("f") a, in("f") b, in("f") c, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn f2h(x: f32) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("cvt.rn.f16.f32 %0, %1;", out("=h") r, in("f") x, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn f2bf(x: f32) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("cvt.rn.bf16.f32 %0, %1;", out("=h") r, in("f") x, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn dp4a(a: u32, b: u32, c: i32) -> i32 {
        dotprod::dp4a_s32(a, b, c)
    }
    /// `__popc`.
    #[inline(always)]
    pub fn popc(x: u32) -> u32 {
        let r: u32;
        unsafe { ptx_asm!("popc.b32 %0, %1;", out("=r") r, in("r") x, options(register_only)); }
        r
    }
    /// `__byte_perm(a, b, s)`: prmt with the selector masked to its 3-bit byte indices.
    #[inline(always)]
    pub fn byte_perm(a: u32, b: u32, s: u32) -> u32 {
        let r: u32;
        let s = s & 0x7777;
        unsafe { ptx_asm!("prmt.b32 %0, %1, %2, %3;", out("=r") r, in("r") a, in("r") b, in("r") s, options(register_only)); }
        r
    }
    /// `__vcmpne4(a, 0)`: 0xff in each byte that is non-zero.
    #[inline(always)]
    pub fn vcmpne4_zero(a: u32) -> u32 {
        let (x, y, z, w) = (a & 0xFF, (a >> 8) & 0xFF, (a >> 16) & 0xFF, (a >> 24) & 0xFF);
        (if x != 0 { 0xFFu32 } else { 0 })
            | (if y != 0 { 0xFF00u32 } else { 0 })
            | (if z != 0 { 0xFF_0000u32 } else { 0 })
            | (if w != 0 { 0xFF00_0000u32 } else { 0 })
    }
    /// `__vsub4(a, b)`: the SIMD-video per-byte subtract. Each byte is independent and wraps --
    /// there is NO carry/borrow between bytes (that is what makes it usable as `(byte^0xff)-0xff`
    /// for a per-byte negation). Implementing it with borrow propagation breaks the cases where the
    /// selector byte is 0x10 and byte 0 is zero.
    #[inline(always)]
    pub fn vsub4(a: u32, b: u32) -> u32 {
        let (a0, a1, a2, a3) = (a & 0xFF, (a >> 8) & 0xFF, (a >> 16) & 0xFF, (a >> 24) & 0xFF);
        let (b0, b1, b2, b3) = (b & 0xFF, (b >> 8) & 0xFF, (b >> 16) & 0xFF, (b >> 24) & 0xFF);
        (a0.wrapping_sub(b0) & 0xFF)
            | ((a1.wrapping_sub(b1) & 0xFF) << 8)
            | ((a2.wrapping_sub(b2) & 0xFF) << 16)
            | ((a3.wrapping_sub(b3) & 0xFF) << 24)
    }
    /// `get_int_b2(x, i)`: two `u16` loads (the block is 98 bytes, so a `u32` load can be
    /// misaligned — the reference deliberately reads 16-bit halves).
    #[inline(always)]
    pub unsafe fn get_int_b2(x: *const u8, i: i32) -> u32 {
        let p = x.add(4 * i as usize) as *const u16;
        (*p as u32) | ((*p.add(1) as u32) << 16)
    }
    /// `unpack_ksigns(v)`: the 7-bit mask, with the 8th sign folded in as its parity.
    #[inline(always)]
    pub fn unpack_ksigns(v: u32) -> u8 {
        let v7 = (v & 0x7F) as u8;
        v7 ^ (((popc(v7 as u32) & 1) as u8) << 7)
    }
    /// llama.cpp fastdiv: `(__umulhi(n, mp) + n) >> L`.
    #[inline(always)]
    pub fn fastdiv(n: u32, mp: u32, l: u32) -> u32 {
        let hi = ((n as u64 * mp as u64) >> 32) as u32;
        hi.wrapping_add(n) >> l
    }

    /// Output element types.
    pub trait Dst: Copy {
        unsafe fn st(p: *mut Self, v: f32);
    }
    impl Dst for f32 {
        #[inline(always)]
        unsafe fn st(p: *mut f32, v: f32) {
            *p = v;
        }
    }
    /// f16 and bf16 need separate zero-sized tags, or both kernels monomorphise to one function
    /// (and the `#[cuda_module]` scan would then only emit one of them).
    #[repr(transparent)]
    #[derive(Clone, Copy)]
    pub struct H(pub u16);
    #[repr(transparent)]
    #[derive(Clone, Copy)]
    pub struct B(pub u16);
    impl Dst for H {
        #[inline(always)]
        unsafe fn st(p: *mut H, v: f32) {
            *(p as *mut u16) = f2h(v);
        }
    }
    impl Dst for B {
        #[inline(always)]
        unsafe fn st(p: *mut B, v: f32) {
            *(p as *mut u16) = f2bf(v);
        }
    }

    const BLOCK_BYTES: usize = 98; // sizeof(block_iq3_xxs)
    const QK: usize = 256;

    /// `dequantize_block_iq3_xxs<dst_t>`: one 256-value block per CUDA block of 32 threads.
    #[inline(always)]
    pub unsafe fn dequant<D: Dst>(vx: *const u8, yy: *mut D) {
        let i = thread::blockIdx_x() as usize;
        let x = vx.add(i * BLOCK_BYTES);
        let tid = thread::threadIdx_x() as usize;
        let il = tid / 8;
        let ib = tid % 8;
        // The reference __global__ wrapper passes `yy + i*QK_K` into the device helper.
        let y = yy.add(i * QK + 32 * ib + 8 * il);
        // `qs` is `qs[3*QK_K/8]` = 96 bytes at struct offset 2 (d is first).
        let q3 = x.add(2 + 8 * ib);
        // gas = (const uint16_t *)(qs + QK_K/4) + 2*ib = x + 2 + 64 + 4*ib.
        let aux32 = get_int_b2(x.add(2 + 64), ib as i32);
        // 0.5 + (aux32>>28) is a small integer; both multiplies are powers of two (exact).
        let d = mul(mul(h2f(*(x as *const u16) as u32), add(0.5, (aux32 >> 28) as f32)), 0.5);
        let signs = ksign((aux32 >> (7 * il as u32)) & 127);
        let w1 = grid_word(*q3.add(2 * il) as u32);
        let w2 = grid_word(*q3.add(2 * il + 1) as u32);
        let g1 = w1.to_le_bytes();
        let g2 = w2.to_le_bytes();
        let mut j = 0usize;
        while j < 4 {
            let s0 = if signs & KMASK[j] != 0 { -1.0f32 } else { 1.0f32 };
            let s1 = if signs & KMASK[j + 4] != 0 { -1.0f32 } else { 1.0f32 };
            D::st(y.add(j), mul(mul(d, g1[j] as f32), s0));
            D::st(y.add(j + 4), mul(mul(d, g2[j] as f32), s1));
            j += 1;
        }
    }

    #[kernel]
    pub unsafe fn iq3_xxs_dequant_f32(vx: *const u8, y: *mut f32) {
        dequant(vx, y)
    }
    #[kernel]
    pub unsafe fn iq3_xxs_dequant_f16(vx: *const u8, y: *mut u16) {
        dequant(vx, y as *mut H)
    }
    #[kernel]
    pub unsafe fn iq3_xxs_dequant_bf16(vx: *const u8, y: *mut u16) {
        dequant(vx, y as *mut B)
    }

    /// `vec_dot_iq3_xxs_q8_1(vx, yb, kbx, iqs)` accumulated into `acc` as one FFMA.
    /// `yb` is the q8_1 block at `kbx * (QK_K/32) + iqs/2`.
    #[inline(always)]
    pub unsafe fn vdot_acc(vx: *const u8, yb: *const u8, kbx: i32, iqs: i32, acc: f32) -> f32 {
        let xb = vx.offset((kbx as isize).wrapping_mul(BLOCK_BYTES as isize));
        let qs = xb.add(2); // block_iq3_xxs.qs
        // get_int_b2(qs, iqs): the u16 pair at qs + 4*iqs.
        let mut q3 = [0u8; 8];
        let mut t = 0usize;
        while t < 8 {
            q3[t] = *qs.add(4 * iqs as usize + t);
            t += 1;
        }
        let aux32 = get_int_b2(qs.add(64), iqs / 2);
        let q8 = yb.add(4) as *const u32;
        let mut sumi = 0i32;
        let mut l0 = 0usize;
        while l0 < 8 {
            let g0 = grid_word(q3[l0] as u32);
            let g1 = grid_word(q3[l0 + 1] as u32);
            let signs = unpack_ksigns(aux32 >> (7 * l0 as u32 / 2));
            // grid_l = __vsub4(grid ^ signs0, signs0) with signs0 = -1 per set sign bit =
            // wrap(byte + 1) = -byte, i.e. the sign is applied by negating that byte.
            let s0 = vcmpne4_zero((signs as u32 * 0x0101_0101) & 0x0804_0201);
            let gl = vsub4(g0 ^ s0, s0);
            let s1 = vcmpne4_zero((signs as u32 * 0x0101_0101) & 0x8040_2010);
            let gh = vsub4(g1 ^ s1, s1);
            sumi = dp4a(gl, *q8.add(l0), sumi);
            sumi = dp4a(gh, *q8.add(l0 + 1), sumi);
            l0 += 2;
        }
        // (ls*sumi + sumi/2)/2, integer; ls = aux32 >> 28.
        let ls = (aux32 >> 28) as i32;
        let sumi = (ls.wrapping_mul(sumi).wrapping_add(sumi / 2)) / 2;
        let d = mul(h2f(*(xb as *const u16) as u32), h2f(*(yb as *const u16) as u32));
        fma(d, sumi as f32, acc)
    }

    /// Compile-time loop bounds. qi/vdr = 8 threads per block, vdr = 2.
    pub struct K<const NC: usize, const SMALL_K: bool>;
    impl<const NC: usize, const SMALL_K: bool> K<NC, SMALL_K> {
        pub const NW: i32 = if NC <= 4 { 4 } else { 2 };
        pub const NW1: i32 = if NC <= 4 { 3 } else { 1 };
        pub const RPB: usize = if NC == 1 { if SMALL_K { 4 } else { 1 } } else { 2 };
        pub const BLOCKS_PER_ITER: i32 = 2 * Self::NW * 32 / 16;
    }

    /// `mul_mat_vec_q<IQ3_XXS, NC, false, SMALL_K, false>` (MMVQ_PARAMETERS_GENERIC).
    #[device]
    pub unsafe fn mmvq<const NC: usize, const SMALL_K: bool>(
        vx: *const u8, vy: *const u8, ids: *const i32, dst: *mut f32, ncols_x: u32,
        nchannels_y_mp: u32, nchannels_y_l: u32, nchannels_y_d: u32,
        stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32,
        channel_ratio_mp: u32, channel_ratio_l: u32,
        stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32,
        sample_ratio_mp: u32, sample_ratio_l: u32,
        stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32,
    ) {
        static mut TMP_SHARED: SharedArray<f32, 768> = SharedArray::UNINIT;
        const QKB: i32 = 256;
        const TPB: i32 = 8; // qi / vdr
        let nwarps = K::<NC, SMALL_K>::NW;
        let rpb = K::<NC, SMALL_K>::RPB as i32;
        let tx = thread::threadIdx_x() as i32;
        let ty = thread::threadIdx_y() as i32;
        let tid = 32 * ty + tx;
        let row0 = rpb.wrapping_mul(thread::blockIdx_x() as i32);
        let blocks_per_row_x = (ncols_x / QKB as u32) as i32;

        let channel_dst = thread::blockIdx_y();
        let (channel_x, channel_y) = if NC == 1 && !ids.is_null() {
            let cx = *ids.add(channel_dst as usize) as u32;
            let q = fastdiv(channel_dst, nchannels_y_mp, nchannels_y_l);
            (cx, channel_dst.wrapping_sub(q.wrapping_mul(nchannels_y_d)))
        } else {
            (fastdiv(channel_dst, channel_ratio_mp, channel_ratio_l), channel_dst)
        };
        let sample_dst = thread::blockIdx_z();
        let sample_x = fastdiv(sample_dst, sample_ratio_mp, sample_ratio_l);
        let sample_y = sample_dst;

        let mut tmp = [[0f32; 4]; NC];
        let y = vy
            .add(sample_y.wrapping_mul(stride_sample_y) as usize * 36)
            .add(channel_y.wrapping_mul(stride_channel_y) as usize * 36);
        let kbx_offset = sample_x
            .wrapping_mul(stride_sample_x)
            .wrapping_add(channel_x.wrapping_mul(stride_channel_x))
            .wrapping_add((row0 as u32).wrapping_mul(stride_row_x));

        let mut kbx = tid / TPB;
        while kbx < blocks_per_row_x {
            let kby = 8 * kbx;
            let kqs = 2 * (tid % TPB);
            let mut j = 0;
            #[unroll]
            while j < NC {
                let yi = (j as u32).wrapping_mul(stride_col_y).wrapping_add((kby + kqs / 2) as u32);
                let yb = y.add(yi as usize * 36);
                let mut i = 0;
                #[unroll]
                while i < K::<NC, SMALL_K>::RPB {
                    let xk = kbx_offset.wrapping_add((i as u32).wrapping_mul(stride_row_x)).wrapping_add(kbx as u32) as i32;
                    tmp[j][i] = vdot_acc(vx, yb, xk, kqs, tmp[j][i]);
                    i += 1;
                }
                j += 1;
            }
            kbx += K::<NC, SMALL_K>::BLOCKS_PER_ITER;
        }

        let sh = SharedArray::as_raw_mut_ptr(&raw mut TMP_SHARED);
        let idx = |l: i32, j: usize, i: usize| ((l as usize * NC + j) * rpb as usize + i) * 32 + tx as usize;
        if ty > 0 {
            let mut j = 0;
            #[unroll]
            while j < NC {
                let mut i = 0;
                #[unroll]
                while i < K::<NC, SMALL_K>::RPB {
                    *sh.add(idx(ty - 1, j, i)) = tmp[j][i];
                    i += 1;
                }
                j += 1;
            }
        }
        thread::sync_threads();
        if ty > 0 {
            return;
        }

        let dst = dst.add(
            sample_dst
                .wrapping_mul(stride_sample_dst)
                .wrapping_add(channel_dst.wrapping_mul(stride_channel_dst))
                .wrapping_add(row0 as u32) as usize,
        );
        let mut j = 0;
        #[unroll]
        while j < NC {
            let mut i = 0;
            #[unroll]
            while i < K::<NC, SMALL_K>::RPB {
                let mut l = 0;
                #[unroll]
                while l < K::<NC, SMALL_K>::NW1 {
                    tmp[j][i] = add(tmp[j][i], *sh.add(idx(l, j, i)));
                    l += 1;
                }
                let mut k = 0;
                #[unroll]
                while k < 5 {
                    tmp[j][i] = add(tmp[j][i], warp::shuffle_xor_f32_sync(0xffff_ffff, tmp[j][i], 16 >> k));
                    k += 1;
                }
                if tx == i as i32 && (rpb == 1 || (row0 as u32).wrapping_add(i as u32) < stride_col_dst) {
                    let o = (j as u32).wrapping_mul(stride_col_dst).wrapping_add(i as u32);
                    *dst.add(o as usize) = tmp[j][i];
                }
                i += 1;
            }
            j += 1;
        }
    }

    // GENERATED KERNELS BEGIN
    #[kernel] pub unsafe fn iq3_xxs_mmvq_1(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<1, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq3_xxs_mmvq_1_small_k(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<1, true>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq3_xxs_mmvq_2(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<2, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq3_xxs_mmvq_3(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<3, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq3_xxs_mmvq_4(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<4, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq3_xxs_mmvq_5(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<5, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq3_xxs_mmvq_6(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<6, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq3_xxs_mmvq_7(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<7, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    #[kernel] pub unsafe fn iq3_xxs_mmvq_8(vx: *const u8, vy: *const u8, ids: *const i32, _f0: u64, _f1: u64, _f2: u64, _f3: u64, _f4: u64, _glu_op: u32, _glu_limit: f32, dst: *mut f32, ncols_x: u32, ncy_mp: u32, ncy_l: u32, ncy_d: u32, stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32, cr_mp: u32, cr_l: u32, _cr_d: u32, stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32, sr_mp: u32, sr_l: u32, _sr_d: u32, stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, _ids_stride: u32) { mmvq::<8, false>(vx, vy, ids, dst, ncols_x, ncy_mp, ncy_l, ncy_d, stride_row_x, stride_col_y, stride_col_dst, cr_mp, cr_l, stride_channel_x, stride_channel_y, stride_channel_dst, sr_mp, sr_l, stride_sample_x, stride_sample_y, stride_sample_dst) }
    // GENERATED KERNELS END
}

fn main() {
    std::process::exit(if gate::run() { 0 } else { 1 });
}
