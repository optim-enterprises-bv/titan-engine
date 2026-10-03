//! PrismML PTQ1_0 (GGML type 143, sudoingX/llama.cpp bonsai2 v1.1 / PrismML-Eng/llama.cpp): ternary
//! weights at group size 128, 1.75 bpw. `qs[24]` carries 120 trits (5 per byte, base 3, most
//! significant first after the `ceil(q * 256 / 243)` scaling), `qh[2]` the last 8 (4 per byte), then
//! one f16 scale `d`: w = (trit - 1) * d.
//!
//! Element order (llama.cpp `dequantize_row_ptq1_0`): e < 80 -> byte `qs[e & 15]`, digit `e >> 4`;
//! 80 <= e < 120 -> byte `qs[16 + ((e - 80) & 7)]`, digit `(e - 80) >> 3`; e >= 120 -> byte
//! `qh[(e - 120) & 1]`, digit `(e - 120) >> 1`. Digit n of byte b: `q = b * 3^n (mod 256)`,
//! trit = `(q * 3) >> 8`.
//!
//! The CUDA path (titan-engine/oxide-kernels/ptq1_0, called from mistralrs-quant) never uses these
//! CPU routines except `to_float` (host embedding lookups and the generic dequantize fallback).

use super::k_quants::GgmlType;
use super::q1_0::BlockQ8_0x4;
use super::GgmlDType;
use half::f16;

pub const QK_PTQ1_0: usize = 128;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockPTQ1_0 {
    pub(crate) qs: [u8; 24],
    pub(crate) qh: [u8; 2],
    pub(crate) d: f16,
}
const _: () = assert!(std::mem::size_of::<BlockPTQ1_0>() == 28);

const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];

impl BlockPTQ1_0 {
    /// The 128 trits of the block as -1, 0, 1, in element order.
    #[inline]
    pub fn trits(&self) -> [i8; QK_PTQ1_0] {
        let mut t = [0i8; QK_PTQ1_0];
        for (e, v) in t.iter_mut().enumerate() {
            let (b, n) = if e < 80 {
                (self.qs[e & 15], e >> 4)
            } else if e < 120 {
                (self.qs[16 + ((e - 80) & 7)], (e - 80) >> 3)
            } else {
                (self.qh[(e - 120) & 1], (e - 120) >> 1)
            };
            let q = b.wrapping_mul(POW3[n]);
            *v = (((q as u16) * 3) >> 8) as i8 - 1;
        }
        t
    }
}

impl GgmlType for BlockPTQ1_0 {
    const DTYPE: GgmlDType = GgmlDType::PTQ1_0;
    const BLCK_SIZE: usize = QK_PTQ1_0;
    type VecDotType = BlockQ8_0x4;

    // dequantize_row_ptq1_0: (float)(xi - 1) * d
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        debug_assert!(ys.len().is_multiple_of(QK_PTQ1_0));
        for (x, ys) in xs.iter().zip(ys.chunks_exact_mut(QK_PTQ1_0)) {
            let d = x.d.to_f32();
            for (y, t) in ys.iter_mut().zip(x.trits()) {
                *y = t as f32 * d;
            }
        }
    }

    // quantize_row_ptq1_0_ref
    fn from_float(xs: &[f32], ys: &mut [Self]) {
        debug_assert!(xs.len().is_multiple_of(QK_PTQ1_0));
        for (y, x) in ys.iter_mut().zip(xs.chunks_exact(QK_PTQ1_0)) {
            let amax = x.iter().fold(0f32, |a, v| a.max(v.abs()));
            let id = if amax != 0.0 { 1.0 / amax } else { 0.0 };
            y.d = f16::from_f32(amax);
            let t = |v: f32| ((v * id).round() as i32 + 1) as u8;
            let mut off = 0usize;
            let mut j = 0usize;
            for &c in &[16usize, 8] {
                for m in 0..c {
                    let mut q: u8 = 0;
                    for n in 0..5 {
                        q = q.wrapping_mul(3).wrapping_add(t(x[off + m + n * c]));
                    }
                    y.qs[j + m] = ((q as u16 * 256 + 242) / 243) as u8;
                }
                off += 5 * c;
                j += c;
            }
            for h in 0..2 {
                let mut q: u8 = 0;
                for m in 0..4 {
                    q = q.wrapping_mul(3).wrapping_add(t(x[off + h + m * 2]));
                }
                q = q.wrapping_mul(3);
                y.qh[h] = ((q as u16 * 256 + 242) / 243) as u8;
            }
        }
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    // Integer sums per 32-element sub-block, `sumf += d_x * sum_k d8_k * sumi_k`.
    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(n.is_multiple_of(QK_PTQ1_0));
        let nb = n / QK_PTQ1_0;
        let mut sumf = 0f32;
        for (x, y) in xs.iter().zip(ys.iter()).take(nb) {
            let t = x.trits();
            let mut acc = 0f32;
            for k in 0..4 {
                let yb = &y.b[k];
                let mut sumi = 0i32;
                for l in 0..32 {
                    sumi += t[k * 32 + l] as i32 * yb.qs[l] as i32;
                }
                acc += yb.d.to_f32() * sumi as f32;
            }
            sumf += x.d.to_f32() * acc;
        }
        sumf
    }
}
