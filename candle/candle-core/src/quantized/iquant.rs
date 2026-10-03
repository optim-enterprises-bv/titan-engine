//! GGML i-quant lookup-table families used by Unsloth "UD" quants: IQ2_XXS (type 16), IQ3_XXS
//! (type 18) and IQ2_S (type 22). All three are 256-value (QK_K) blocks with a f16 scale and a
//! grid codebook; the codebooks are the same tables the cuda-oxide ports in
//! `titan-engine/oxide-kernels/{iq2_xxs,iq3_xxs,iq2_s}` index.
//!
//! Only the *dequantize* path is implemented here: it is what the GGUF loader and candle's
//! `dequantize` need. The CUDA kernel path (`fast_mm*`) uses the oxide PTX modules instead, so
//! `from_float` / `vec_dot` are deliberately unimplemented.

use super::k_quants::{BlockQ8_1, GgmlType};
use super::GgmlDType;
use half::f16;

pub const QK_K: usize = 256;

/// `kmask_iq2xs[8]` (ggml-common.h).
const KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// `ksigns_iq2xs[128]`.
const KSIGNS_IQ2XS: [u8; 128] = [
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15, 144, 17, 18, 147, 20, 149,
    150, 23, 24, 153, 154, 27, 156, 29, 30, 159, 160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170,
    43, 172, 45, 46, 175, 48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207, 80, 209, 210, 83, 212,
    85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95, 96, 225, 226, 99, 228, 101, 102, 231, 232,
    105, 106, 235, 108, 237, 238, 111, 240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123,
    252, 125, 126, 255,
];

/// `iq2xxs_grid[256]` (ggml-common.h).
pub const IQ2XXS_GRID: [u64; 256] = include!("data_iq2xxs_grid.rs");
/// `iq2s_grid[1024]`.
pub const IQ2S_GRID: [u64; 1024] = include!("data_iq2s_grid.rs");
/// `iq3xxs_grid[256]`.
pub const IQ3XXS_GRID: [u64; 256] = include!("data_iq3xxs_grid.rs");

/// Read one grid entry's eight bytes as `[u8; 8]` (little-endian, matching the CUDA kernels).
#[inline]
fn grid_bytes(g: u64) -> [u8; 8] {
    g.to_le_bytes()
}

/// `dequantize_row_iq2_xxs` for one block, writing QK_K values. Group ib32 is 8 bytes of `qs`
/// (ggml: two u32): bytes 0-3 index the grid, the little-endian u32 in bytes 4-7 holds four 7-bit
/// sign indices and the 4-bit scale in its top bits.
fn iq2xxs_block(x: &BlockIQ2xxs, ys: &mut [f32]) {
    let d = x.d.to_f32();
    for ib32 in 0..QK_K / 32 {
        let q = &x.qs[8 * ib32..8 * ib32 + 8];
        let aux1 = u32::from_le_bytes([q[4], q[5], q[6], q[7]]);
        let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
        for l in 0..4usize {
            let grid = grid_bytes(IQ2XXS_GRID[q[l] as usize]);
            let signs = KSIGNS_IQ2XS[((aux1 >> (7 * l as u32)) & 127) as usize];
            for j in 0..8 {
                let s = if signs & KMASK_IQ2XS[j] != 0 { -1.0f32 } else { 1.0f32 };
                ys[32 * ib32 + 8 * l + j] = db * grid[j] as f32 * s;
            }
        }
    }
}

/// `dequantize_row_iq3_xxs` for one block.
fn iq3xxs_block(x: &BlockIQ3xxs, ys: &mut [f32]) {
    const GRID_BYTES: usize = 4;
    let d = x.d.to_f32();
    for tid in 0..32usize {
        let (il, ib) = (tid / 8, tid % 8);
        let qs_off = 8 * ib;
        // scales_and_signs = qs + QK_K/4: one u32 per 32-value group.
        let ss = QK_K / 4 + 4 * ib;
        let aux32 = u32::from_le_bytes([x.qs[ss], x.qs[ss + 1], x.qs[ss + 2], x.qs[ss + 3]]);
        let dd = d * (0.5 + ((aux32 >> 28) as f32)) * 0.5;
        let signs = KSIGNS_IQ2XS[((aux32 >> (7 * il as u32)) & 127) as usize];
        for j in 0..4 {
            let g1 = grid_bytes(IQ3XXS_GRID[x.qs[qs_off + 2 * il] as usize]);
            let g2 = grid_bytes(IQ3XXS_GRID[x.qs[qs_off + 2 * il + 1] as usize]);
            let _ = GRID_BYTES;
            let s0 = if signs & KMASK_IQ2XS[j] != 0 { -1.0f32 } else { 1.0f32 };
            let s1 = if signs & KMASK_IQ2XS[j + 4] != 0 { -1.0f32 } else { 1.0f32 };
            ys[32 * ib + 8 * il + j] = dd * g1[j] as f32 * s0;
            ys[32 * ib + 8 * il + j + 4] = dd * g2[j] as f32 * s1;
        }
    }
}

/// `dequantize_row_iq2_s` for one block.
fn iq2s_block(x: &BlockIQ2s, ys: &mut [f32]) {
    let d = x.d.to_f32();
    for tid in 0..32usize {
        let (il, ib) = (tid / 8, tid % 8);
        let idx = (x.qs[4 * ib + il] as usize) | ((((x.qh[ib] as usize) << (8 - 2 * il)) & 0x300) as usize);
        let grid = grid_bytes(IQ2S_GRID[idx]);
        let dd = d * (0.5 + (((x.scales[ib] >> (4 * (il / 2))) & 0xF) as f32)) * 0.25;
        let signs = x.qs[32 + 4 * ib + il];
        for j in 0..8 {
            let s = if signs & KMASK_IQ2XS[j] != 0 { -1.0f32 } else { 1.0f32 };
            ys[32 * ib + 8 * il + j] = dd * grid[j] as f32 * s;
        }
    }
}

/// GGML type 16: 256 weights, f16 scale + 32 packed 16-bit units (4 grid indices + shared scale).
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockIQ2xxs {
    pub(crate) d: f16,
    pub(crate) qs: [u8; QK_K / 4],
}
const _: () = assert!(std::mem::size_of::<BlockIQ2xxs>() == 2 + QK_K / 4);

/// GGML type 18: 256 weights, f16 scale, 64 bytes of grid indices, 8 bytes of packed group
/// scales/signs in a trailing u32 per group.
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockIQ3xxs {
    pub(crate) d: f16,
    pub(crate) qs: [u8; 96],
}
const _: () = assert!(std::mem::size_of::<BlockIQ3xxs>() == 98);

/// GGML type 22: 256 weights, f16 scale, 64 bytes of grid codes/signs, 8 high bits, 8 scales.
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockIQ2s {
    pub(crate) d: f16,
    pub(crate) qs: [u8; QK_K / 4],
    pub(crate) qh: [u8; QK_K / 32],
    pub(crate) scales: [u8; QK_K / 32],
}
const _: () = assert!(std::mem::size_of::<BlockIQ2s>() == 2 + QK_K / 4 + 2 * (QK_K / 32));

/// Scalar CPU fallback for the dot product. The model path never uses this: on CUDA these types
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

// ---------------------------------------------------------------------------------------------
// IQ3_S (type 21): d (f16), qs[64] (low 8 bits of 9-bit grid indices), qh[8] (their high bits),
// signs[32] (one sign bit per value), scales[4] (4-bit group scales, `1 + 2*s`). 110 bytes.
// The CUDA path is the gated oxide-kernels/iq3_s port (dequantize + mmvq, cuda.rs IQ3_S_FMT).
// ---------------------------------------------------------------------------------------------

/// `iq3s_grid[512]` (ggml-common.h), generated by oxide-kernels/iq3_s/gen_tables.py.
pub const IQ3S_GRID: [u32; 512] = include!("data_iq3s_grid.rs");

/// GGML type 21: 256 weights in 8 groups of 32.
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockIQ3s {
    pub(crate) d: f16,
    pub(crate) qs: [u8; QK_K / 4],
    pub(crate) qh: [u8; QK_K / 32],
    pub(crate) signs: [u8; QK_K / 8],
    pub(crate) scales: [u8; QK_K / 64],
}
const _: () = assert!(std::mem::size_of::<BlockIQ3s>() == 110);

/// `dequantize_row_iq3_s` (ggml-quants.c) for one block, in its operation order:
/// `db = d * (1 + 2*s)` (f32 times the converted int), `y = db * grid[j] * (sign ? -1 : 1)`.
fn iq3s_block(x: &BlockIQ3s, ys: &mut [f32]) {
    let d = x.d.to_f32();
    for ib32 in (0..QK_K / 32).step_by(2) {
        let db = [
            d * (1 + 2 * (x.scales[ib32 / 2] & 0xf) as i32) as f32,
            d * (1 + 2 * (x.scales[ib32 / 2] >> 4) as i32) as f32,
        ];
        for h in 0..2 {
            let ib = ib32 + h;
            let qs = &x.qs[8 * ib..8 * ib + 8];
            let qh = x.qh[ib] as usize;
            let signs = &x.signs[4 * ib..4 * ib + 4];
            for l in 0..4 {
                let g1 = IQ3S_GRID[qs[2 * l] as usize | ((qh << (8 - 2 * l)) & 256)].to_le_bytes();
                let g2 = IQ3S_GRID[qs[2 * l + 1] as usize | ((qh << (7 - 2 * l)) & 256)].to_le_bytes();
                for j in 0..4 {
                    let s1 = if signs[l] & KMASK_IQ2XS[j] != 0 { -1.0f32 } else { 1.0f32 };
                    let s2 = if signs[l] & KMASK_IQ2XS[j + 4] != 0 { -1.0f32 } else { 1.0f32 };
                    ys[32 * ib + 8 * l + j] = db[h] * g1[j] as f32 * s1;
                    ys[32 * ib + 8 * l + j + 4] = db[h] * g2[j] as f32 * s2;
                }
            }
        }
    }
}

iquant_trait!(BlockIQ3s, GgmlDType::IQ3S, iq3s_block);
iquant_trait!(BlockIQ2xxs, GgmlDType::IQ2XXS, iq2xxs_block);
iquant_trait!(BlockIQ3xxs, GgmlDType::IQ3XXS, iq3xxs_block);
iquant_trait!(BlockIQ2s, GgmlDType::IQ2S, iq2s_block);

// ---------------------------------------------------------------------------------------------
// IQ4_XS (type 23) and IQ2_XS (type 17). Both use QK_K = 256 and the same GgmlType shape as
// above. Their CUDA path is the gated oxide kernel (oxide-kernels/iq4_xs, oxide-kernels/iq2_xs);
// these structs exist so the loader can size and slice the tensors.
// ---------------------------------------------------------------------------------------------

/// IQ4_XS: `d` (f16) + `scales_h` (u16, 2 bits per group) + 4 x `scales_l` (u8, low nibbles) +
/// 128 nibble bytes packing 256 4-bit values.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockIQ4xs {
    pub d: f16,
    pub scales_h: u16,
    pub scales_l: [u8; 4],
    pub qs: [u8; 128],
}

#[cfg(feature = "cuda")]
unsafe impl cudarc::driver::DeviceRepr for BlockIQ4xs {}
#[cfg(feature = "cuda")]
unsafe impl cudarc::driver::ValidAsZeroBits for BlockIQ4xs {}

impl BlockIQ4xs {
    pub const BLCK_SIZE: usize = QK_K;
    pub fn zeros() -> Self {
        Self { d: f16::from_f32(0.), scales_h: 0, scales_l: [0; 4], qs: [0; 128] }
    }

    /// 6-bit group scale: `((scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((scales_h >> 2*ib) & 3) << 4)`.
    fn group_scale6(&self, ib: usize) -> u32 {
        let lo = ((self.scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) as u32;
        lo | ((((self.scales_h >> (2 * ib)) & 3) as u32) << 4)
    }

    /// `dequantize_row_iq4_xs` for one block: group ib (32 values) uses `dl = d * (ls - 32)` and
    /// the 16 bytes `qs[16*ib..]`; low nibbles give values 0-15, high nibbles 16-31.
    pub fn to_float(&self, y: &mut [f32]) {
        let d = self.d.to_f32();
        for ib in 0..QK_K / 32 {
            let dl = d * (self.group_scale6(ib) as i32 - 32) as f32;
            let qs = &self.qs[16 * ib..16 * ib + 16];
            for j in 0..16 {
                y[32 * ib + j] = dl * KV4XS[(qs[j] & 0xf) as usize];
                y[32 * ib + j + 16] = dl * KV4XS[(qs[j] >> 4) as usize];
            }
        }
    }
}
const KV4XS: [f32; 16] = {
    let mut a = [0f32; 16];
    let k: [i8; 16] = [
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    ];
    let mut i = 0;
    while i < 16 {
        a[i] = k[i] as f32;
        i += 1;
    }
    a
};

/// IQ2_XS: `d` (f16) + 32 u16 grid codes + 8 u8 4-bit group scales.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockIQ2xs {
    pub d: f16,
    pub qs: [u16; 32],
    pub scales: [u8; 8],
}

#[cfg(feature = "cuda")]
unsafe impl cudarc::driver::DeviceRepr for BlockIQ2xs {}
#[cfg(feature = "cuda")]
unsafe impl cudarc::driver::ValidAsZeroBits for BlockIQ2xs {}

impl BlockIQ2xs {
    pub const BLCK_SIZE: usize = QK_K;
    pub fn zeros() -> Self {
        Self { d: f16::from_f32(0.), qs: [0; 32], scales: [0; 8] }
    }
}

impl GgmlType for BlockIQ4xs {
    const DTYPE: GgmlDType = GgmlDType::IQ4XS;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8_1;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (x, y) in xs.iter().zip(ys.chunks_exact_mut(QK_K)) {
            x.to_float(y);
        }
    }
    fn from_float(_xs: &[f32], _ys: &mut [Self]) {
        unimplemented!("quantizing to IQ4_XS is not supported; use a pre-quantized GGUF")
    }
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }
    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        let nb = n / QK_K;
        let mut xbuf = vec![0f32; QK_K];
        let mut sumf = 0f32;
        for (i, x) in xs.iter().take(nb).enumerate() {
            x.to_float(&mut xbuf);
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

impl GgmlType for BlockIQ2xs {
    const DTYPE: GgmlDType = GgmlDType::IQ2XS;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8_1;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (x, y) in xs.iter().zip(ys.chunks_exact_mut(QK_K)) {
            iq2xs_block(x, y);
        }
    }
    fn from_float(_xs: &[f32], _ys: &mut [Self]) {
        unimplemented!("quantizing to IQ2_XS is not supported; use a pre-quantized GGUF")
    }
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }
    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        let nb = n / QK_K;
        let mut xbuf = vec![0f32; QK_K];
        let mut sumf = 0f32;
        for (i, x) in xs.iter().take(nb).enumerate() {
            iq2xs_block(x, &mut xbuf);
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

/// `dequantize_row_iq2_xs`: each u16 code is a 9-bit index into the 512-entry grid plus a 7-bit
/// index into `ksigns_iq2xs`; each 32-value group has two 4-bit scales (`d * (0.5 + s) * 0.25`),
/// the low nibble for its first 16 values and the high nibble for the last 16.
fn iq2xs_block(x: &BlockIQ2xs, y: &mut [f32]) {
    let d = x.d.to_f32();
    for ib32 in 0..QK_K / 32 {
        let db = [
            d * (0.5 + (x.scales[ib32] & 0xf) as f32) * 0.25,
            d * (0.5 + (x.scales[ib32] >> 4) as f32) * 0.25,
        ];
        for l in 0..4usize {
            let q = x.qs[4 * ib32 + l];
            let grid = IQ2XS_GRID[(q & 511) as usize].to_le_bytes();
            let signs = KSIGNS_IQ2XS[(q >> 9) as usize];
            for j in 0..8 {
                let s = if signs & KMASK_IQ2XS[j] != 0 { -1.0f32 } else { 1.0f32 };
                y[32 * ib32 + 8 * l + j] = db[l / 2] * grid[j] as f32 * s;
            }
        }
    }
}

/// `iq2xs_grid`: 512 u64 entries, 8 bytes each (values 0..=127, so all bytes are unsigned).
pub const IQ2XS_GRID: [u64; 512] = include!("data_iq2xs_grid.rs");

