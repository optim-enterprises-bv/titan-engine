import re, os

p = "~/titan-engine/candle/candle-core/src/quantized/iquant.rs"
s = open(p).read()

# ---- IQ4_XS (type 23) and IQ2_XS (type 17) block definitions + dequant ----
EXTRA = r'''
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

unsafe impl cudarc::driver::DeviceRepr for BlockIQ4xs {}
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

    /// Row of 256 i-quant values -> 256 f32. The value is a nibble into a 16-entry table of
    /// *signed* 4-bit values, scaled by `(group_scale6 - 32)` and the block scale.
    pub fn to_float(&self, y: &mut [f32]) {
        const KVALUES: [i8; 16] = [
            -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
        ];
        let d = self.d.to_f32();
        for j in 0..QK_K {
            let g = j / 32;
            let ls = self.group_scale6(g) as i32 - 32;
            let byte = self.qs[j / 2];
            let nib = if j % 2 == 0 { byte & 0xf } else { byte >> 4 };
            y[j] = d * (ls as f32) * KV4XS[nib as usize];
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

unsafe impl cudarc::driver::DeviceRepr for BlockIQ2xs {}
unsafe impl cudarc::driver::ValidAsZeroBits for BlockIQ2xs {}

impl BlockIQ2xs {
    pub const BLCK_SIZE: usize = QK_K;
    pub fn zeros() -> Self {
        Self { d: f16::from_f32(0.), qs: [0; 32], scales: [0; 8] }
    }
}
'''

s = s.rstrip() + "\n" + EXTRA

TRAITS = '''
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

/// `dequantize_row_iq2_xs`: each u16 code carries two 9-bit grid indices (512-entry table) plus
/// sign bits; the group scale is a 4-bit nibble.
fn iq2xs_block(x: &BlockIQ2xs, y: &mut [f32]) {
    for ib in 0..8usize {
        let ls = (x.scales[ib / 2] >> (4 * (ib % 2))) & 0xf;
        let d = x.d.to_f32() * (((ls as i32 + 1) >> 1) as f32) * if ls & 1 != 0 { 1.0 } else { -1.0 };
        let dl = d;
        for l in 0..4usize {
            let q = x.qs[ib * 4 + l] as u32;
            let idx0 = (q & 0x1ff) as usize;
            let idx1 = ((q >> 9) & 0x1ff) as usize;
            let g0 = IQ2XS_GRID[idx0];
            let g1 = IQ2XS_GRID[idx1];
            for b in 0..4usize {
                y[ib * 32 + l * 8 + b] = dl * g0[b] as f32;
                y[ib * 32 + l * 8 + 4 + b] = dl * g1[b] as f32;
            }
        }
    }
}

/// `iq2xs_grid`: 512 u64 entries, 8 bytes each (values 0..=127, so all bytes are unsigned).
pub const IQ2XS_GRID: [u64; 512] = include!("data_iq2xs_grid.rs");

'''

s = s + TRAITS

# grid bytes helper used by the IQ2_XXS dot above, if it is still referenced
if "grid_bytes(" in s and "fn grid_bytes(" not in s:
    s += '''
/// Bytes of a little-endian u64 grid entry.
fn grid_bytes(v: u64) -> [u8; 8] {
    v.to_le_bytes()
}
'''
open(p, "w").write(s)
print("iquant.rs now", os.path.getsize(p), "bytes")
