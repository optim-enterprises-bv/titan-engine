//! Launcher-level differential gate: every extern "C" launcher of group C is called twice on
//! identical inputs in the same (primary) context and on the same stream -- once the REAL C
//! launcher from libmistralrsquant.a (nvcc kernels), once its pure-Rust twin in `crate::launch`
//! (oxide kernels) -- and every output byte is compared, including untouched gaps of strided
//! outputs and the stream-k tmp_fixup buffer. Afterwards every kernel instance of the reference
//! cubins must have been launched through its Rust twin (or be in the documented unreachable set).
//!
//! Env: MRQC_ONLY=mmvq,quantize,mmq_quantize,q4_k,... restricts the families run;
//! MRQC_SELF=1 runs the C launcher on both sides (checks the gate itself).
use crate::launch as rs;
use cuda_core::sys;
use std::collections::{BTreeMap, HashSet};
use std::ffi::c_void;

mod cref {
    use std::ffi::c_void;
    macro_rules! decl {
        ($($plain:ident, $glu:ident, $qkv:ident;)*) => { unsafe extern "C" { $(
            pub fn $plain(vx: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32, nrows_x: i32,
                          stride_col_y: i32, stride_col_dst: i32, b_size: i32, stream: *mut c_void);
            pub fn $glu(vx_gate: *const c_void, vx_up: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32,
                        nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, b_size: i32, activation: i32, stream: *mut c_void);
            pub fn $qkv(vx_q: *const c_void, vx_k: *const c_void, vx_v: *const c_void, vy: *const c_void, q_dst: *mut c_void,
                        k_dst: *mut c_void, v_dst: *mut c_void, ncols_x: i32, nrows_q: i32, nrows_k: i32, nrows_v: i32,
                        stride_col_y: i32, b_size: i32, stream: *mut c_void);
        )* } };
    }
    decl! {
        launch_mmvq_gguf_q4_0_bf16_plain, launch_mmvq_gguf_q4_0_bf16_fused_glu, launch_mmvq_gguf_q4_0_bf16_fused_qkv;
        launch_mmvq_gguf_q4_1_bf16_plain, launch_mmvq_gguf_q4_1_bf16_fused_glu, launch_mmvq_gguf_q4_1_bf16_fused_qkv;
        launch_mmvq_gguf_q5_0_bf16_plain, launch_mmvq_gguf_q5_0_bf16_fused_glu, launch_mmvq_gguf_q5_0_bf16_fused_qkv;
        launch_mmvq_gguf_q5_1_bf16_plain, launch_mmvq_gguf_q5_1_bf16_fused_glu, launch_mmvq_gguf_q5_1_bf16_fused_qkv;
        launch_mmvq_gguf_q8_0_bf16_plain, launch_mmvq_gguf_q8_0_bf16_fused_glu, launch_mmvq_gguf_q8_0_bf16_fused_qkv;
        launch_mmvq_gguf_q2_k_bf16_plain, launch_mmvq_gguf_q2_k_bf16_fused_glu, launch_mmvq_gguf_q2_k_bf16_fused_qkv;
        launch_mmvq_gguf_q3_k_bf16_plain, launch_mmvq_gguf_q3_k_bf16_fused_glu, launch_mmvq_gguf_q3_k_bf16_fused_qkv;
        launch_mmvq_gguf_q4_k_bf16_plain, launch_mmvq_gguf_q4_k_bf16_fused_glu, launch_mmvq_gguf_q4_k_bf16_fused_qkv;
        launch_mmvq_gguf_q5_k_bf16_plain, launch_mmvq_gguf_q5_k_bf16_fused_glu, launch_mmvq_gguf_q5_k_bf16_fused_qkv;
        launch_mmvq_gguf_q6_k_bf16_plain, launch_mmvq_gguf_q6_k_bf16_fused_glu, launch_mmvq_gguf_q6_k_bf16_fused_qkv;
        launch_mmvq_gguf_q4_0_f16_plain, launch_mmvq_gguf_q4_0_f16_fused_glu, launch_mmvq_gguf_q4_0_f16_fused_qkv;
        launch_mmvq_gguf_q4_1_f16_plain, launch_mmvq_gguf_q4_1_f16_fused_glu, launch_mmvq_gguf_q4_1_f16_fused_qkv;
        launch_mmvq_gguf_q5_0_f16_plain, launch_mmvq_gguf_q5_0_f16_fused_glu, launch_mmvq_gguf_q5_0_f16_fused_qkv;
        launch_mmvq_gguf_q5_1_f16_plain, launch_mmvq_gguf_q5_1_f16_fused_glu, launch_mmvq_gguf_q5_1_f16_fused_qkv;
        launch_mmvq_gguf_q8_0_f16_plain, launch_mmvq_gguf_q8_0_f16_fused_glu, launch_mmvq_gguf_q8_0_f16_fused_qkv;
        launch_mmvq_gguf_q2_k_f16_plain, launch_mmvq_gguf_q2_k_f16_fused_glu, launch_mmvq_gguf_q2_k_f16_fused_qkv;
        launch_mmvq_gguf_q3_k_f16_plain, launch_mmvq_gguf_q3_k_f16_fused_glu, launch_mmvq_gguf_q3_k_f16_fused_qkv;
        launch_mmvq_gguf_q4_k_f16_plain, launch_mmvq_gguf_q4_k_f16_fused_glu, launch_mmvq_gguf_q4_k_f16_fused_qkv;
        launch_mmvq_gguf_q5_k_f16_plain, launch_mmvq_gguf_q5_k_f16_fused_glu, launch_mmvq_gguf_q5_k_f16_fused_qkv;
        launch_mmvq_gguf_q6_k_f16_plain, launch_mmvq_gguf_q6_k_f16_fused_glu, launch_mmvq_gguf_q6_k_f16_fused_qkv;
        launch_mmvq_gguf_q4_0_f32_plain, launch_mmvq_gguf_q4_0_f32_fused_glu, launch_mmvq_gguf_q4_0_f32_fused_qkv;
        launch_mmvq_gguf_q4_1_f32_plain, launch_mmvq_gguf_q4_1_f32_fused_glu, launch_mmvq_gguf_q4_1_f32_fused_qkv;
        launch_mmvq_gguf_q5_0_f32_plain, launch_mmvq_gguf_q5_0_f32_fused_glu, launch_mmvq_gguf_q5_0_f32_fused_qkv;
        launch_mmvq_gguf_q5_1_f32_plain, launch_mmvq_gguf_q5_1_f32_fused_glu, launch_mmvq_gguf_q5_1_f32_fused_qkv;
        launch_mmvq_gguf_q8_0_f32_plain, launch_mmvq_gguf_q8_0_f32_fused_glu, launch_mmvq_gguf_q8_0_f32_fused_qkv;
        launch_mmvq_gguf_q2_k_f32_plain, launch_mmvq_gguf_q2_k_f32_fused_glu, launch_mmvq_gguf_q2_k_f32_fused_qkv;
        launch_mmvq_gguf_q3_k_f32_plain, launch_mmvq_gguf_q3_k_f32_fused_glu, launch_mmvq_gguf_q3_k_f32_fused_qkv;
        launch_mmvq_gguf_q4_k_f32_plain, launch_mmvq_gguf_q4_k_f32_fused_glu, launch_mmvq_gguf_q4_k_f32_fused_qkv;
        launch_mmvq_gguf_q5_k_f32_plain, launch_mmvq_gguf_q5_k_f32_fused_glu, launch_mmvq_gguf_q5_k_f32_fused_qkv;
        launch_mmvq_gguf_q6_k_f32_plain, launch_mmvq_gguf_q6_k_f32_fused_glu, launch_mmvq_gguf_q6_k_f32_fused_qkv;
    }
    unsafe extern "C" {
        pub fn launch_mmvq_gguf_quantize_q8_1_bf16(x: *const c_void, vy: *mut c_void, kx: i32, kx_padded: i32, num_rows: i32, stream: *mut c_void);
        pub fn launch_mmvq_gguf_quantize_q8_1_f16(x: *const c_void, vy: *mut c_void, kx: i32, kx_padded: i32, num_rows: i32, stream: *mut c_void);
        pub fn launch_mmvq_gguf_quantize_q8_1_f32(x: *const c_void, vy: *mut c_void, kx: i32, kx_padded: i32, num_rows: i32, stream: *mut c_void);
        pub fn cudaSetDevice(d: i32) -> i32;
        pub fn cudaFree(p: *mut c_void) -> i32;
        pub fn cudaGetLastError() -> i32;
    }
}

type Plain = unsafe extern "C" fn(*const c_void, *const c_void, *mut c_void, i32, i32, i32, i32, i32, *mut c_void);
type Glu = unsafe extern "C" fn(*const c_void, *const c_void, *const c_void, *mut c_void, i32, i32, i32, i32, i32, i32, *mut c_void);
type Qkv = unsafe extern "C" fn(
    *const c_void, *const c_void, *const c_void, *const c_void, *mut c_void, *mut c_void, *mut c_void, i32, i32, i32, i32, i32, i32,
    *mut c_void,
);
type Quant = unsafe extern "C" fn(*const c_void, *mut c_void, i32, i32, i32, *mut c_void);

/// (C, Rust) launcher triples per [dst][fmt].
struct Mmvq {
    plain: (Plain, Plain),
    glu: (Glu, Glu),
    qkv: (Qkv, Qkv),
}

macro_rules! m {
    ($p:ident, $g:ident, $k:ident) => {
        Mmvq { plain: (cref::$p, rs::$p), glu: (cref::$g, rs::$g), qkv: (cref::$k, rs::$k) }
    };
}

fn mmvq_fns(di: usize, fi: usize) -> Mmvq {
    let t: [[Mmvq; 10]; 3] = [
        [
            m!(launch_mmvq_gguf_q4_0_bf16_plain, launch_mmvq_gguf_q4_0_bf16_fused_glu, launch_mmvq_gguf_q4_0_bf16_fused_qkv),
            m!(launch_mmvq_gguf_q4_1_bf16_plain, launch_mmvq_gguf_q4_1_bf16_fused_glu, launch_mmvq_gguf_q4_1_bf16_fused_qkv),
            m!(launch_mmvq_gguf_q5_0_bf16_plain, launch_mmvq_gguf_q5_0_bf16_fused_glu, launch_mmvq_gguf_q5_0_bf16_fused_qkv),
            m!(launch_mmvq_gguf_q5_1_bf16_plain, launch_mmvq_gguf_q5_1_bf16_fused_glu, launch_mmvq_gguf_q5_1_bf16_fused_qkv),
            m!(launch_mmvq_gguf_q8_0_bf16_plain, launch_mmvq_gguf_q8_0_bf16_fused_glu, launch_mmvq_gguf_q8_0_bf16_fused_qkv),
            m!(launch_mmvq_gguf_q2_k_bf16_plain, launch_mmvq_gguf_q2_k_bf16_fused_glu, launch_mmvq_gguf_q2_k_bf16_fused_qkv),
            m!(launch_mmvq_gguf_q3_k_bf16_plain, launch_mmvq_gguf_q3_k_bf16_fused_glu, launch_mmvq_gguf_q3_k_bf16_fused_qkv),
            m!(launch_mmvq_gguf_q4_k_bf16_plain, launch_mmvq_gguf_q4_k_bf16_fused_glu, launch_mmvq_gguf_q4_k_bf16_fused_qkv),
            m!(launch_mmvq_gguf_q5_k_bf16_plain, launch_mmvq_gguf_q5_k_bf16_fused_glu, launch_mmvq_gguf_q5_k_bf16_fused_qkv),
            m!(launch_mmvq_gguf_q6_k_bf16_plain, launch_mmvq_gguf_q6_k_bf16_fused_glu, launch_mmvq_gguf_q6_k_bf16_fused_qkv),
        ],
        [
            m!(launch_mmvq_gguf_q4_0_f16_plain, launch_mmvq_gguf_q4_0_f16_fused_glu, launch_mmvq_gguf_q4_0_f16_fused_qkv),
            m!(launch_mmvq_gguf_q4_1_f16_plain, launch_mmvq_gguf_q4_1_f16_fused_glu, launch_mmvq_gguf_q4_1_f16_fused_qkv),
            m!(launch_mmvq_gguf_q5_0_f16_plain, launch_mmvq_gguf_q5_0_f16_fused_glu, launch_mmvq_gguf_q5_0_f16_fused_qkv),
            m!(launch_mmvq_gguf_q5_1_f16_plain, launch_mmvq_gguf_q5_1_f16_fused_glu, launch_mmvq_gguf_q5_1_f16_fused_qkv),
            m!(launch_mmvq_gguf_q8_0_f16_plain, launch_mmvq_gguf_q8_0_f16_fused_glu, launch_mmvq_gguf_q8_0_f16_fused_qkv),
            m!(launch_mmvq_gguf_q2_k_f16_plain, launch_mmvq_gguf_q2_k_f16_fused_glu, launch_mmvq_gguf_q2_k_f16_fused_qkv),
            m!(launch_mmvq_gguf_q3_k_f16_plain, launch_mmvq_gguf_q3_k_f16_fused_glu, launch_mmvq_gguf_q3_k_f16_fused_qkv),
            m!(launch_mmvq_gguf_q4_k_f16_plain, launch_mmvq_gguf_q4_k_f16_fused_glu, launch_mmvq_gguf_q4_k_f16_fused_qkv),
            m!(launch_mmvq_gguf_q5_k_f16_plain, launch_mmvq_gguf_q5_k_f16_fused_glu, launch_mmvq_gguf_q5_k_f16_fused_qkv),
            m!(launch_mmvq_gguf_q6_k_f16_plain, launch_mmvq_gguf_q6_k_f16_fused_glu, launch_mmvq_gguf_q6_k_f16_fused_qkv),
        ],
        [
            m!(launch_mmvq_gguf_q4_0_f32_plain, launch_mmvq_gguf_q4_0_f32_fused_glu, launch_mmvq_gguf_q4_0_f32_fused_qkv),
            m!(launch_mmvq_gguf_q4_1_f32_plain, launch_mmvq_gguf_q4_1_f32_fused_glu, launch_mmvq_gguf_q4_1_f32_fused_qkv),
            m!(launch_mmvq_gguf_q5_0_f32_plain, launch_mmvq_gguf_q5_0_f32_fused_glu, launch_mmvq_gguf_q5_0_f32_fused_qkv),
            m!(launch_mmvq_gguf_q5_1_f32_plain, launch_mmvq_gguf_q5_1_f32_fused_glu, launch_mmvq_gguf_q5_1_f32_fused_qkv),
            m!(launch_mmvq_gguf_q8_0_f32_plain, launch_mmvq_gguf_q8_0_f32_fused_glu, launch_mmvq_gguf_q8_0_f32_fused_qkv),
            m!(launch_mmvq_gguf_q2_k_f32_plain, launch_mmvq_gguf_q2_k_f32_fused_glu, launch_mmvq_gguf_q2_k_f32_fused_qkv),
            m!(launch_mmvq_gguf_q3_k_f32_plain, launch_mmvq_gguf_q3_k_f32_fused_glu, launch_mmvq_gguf_q3_k_f32_fused_qkv),
            m!(launch_mmvq_gguf_q4_k_f32_plain, launch_mmvq_gguf_q4_k_f32_fused_glu, launch_mmvq_gguf_q4_k_f32_fused_qkv),
            m!(launch_mmvq_gguf_q5_k_f32_plain, launch_mmvq_gguf_q5_k_f32_fused_glu, launch_mmvq_gguf_q5_k_f32_fused_qkv),
            m!(launch_mmvq_gguf_q6_k_f32_plain, launch_mmvq_gguf_q6_k_f32_fused_glu, launch_mmvq_gguf_q6_k_f32_fused_qkv),
        ],
    ];
    let [a, b, c] = t;
    [a, b, c].into_iter().nth(di).unwrap().into_iter().nth(fi).unwrap()
}

fn quant_pair(t: usize) -> (Quant, Quant) {
    [
        (cref::launch_mmvq_gguf_quantize_q8_1_bf16 as Quant, rs::launch_mmvq_gguf_quantize_q8_1_bf16 as Quant),
        (cref::launch_mmvq_gguf_quantize_q8_1_f16, rs::launch_mmvq_gguf_quantize_q8_1_f16),
        (cref::launch_mmvq_gguf_quantize_q8_1_f32, rs::launch_mmvq_gguf_quantize_q8_1_f32),
    ][t]
}

/// A GGUF weight format: name, block bytes, values per block, f16 scale-field offsets.
pub struct Fmt {
    pub name: &'static str,
    pub bs: usize,
    pub qk: usize,
    pub f16s: &'static [usize],
}
pub const FMTS: [Fmt; 10] = [
    Fmt { name: "q4_0", bs: 18, qk: 32, f16s: &[0] },
    Fmt { name: "q4_1", bs: 20, qk: 32, f16s: &[0, 2] },
    Fmt { name: "q5_0", bs: 22, qk: 32, f16s: &[0] },
    Fmt { name: "q5_1", bs: 24, qk: 32, f16s: &[0, 2] },
    Fmt { name: "q8_0", bs: 34, qk: 32, f16s: &[0] },
    Fmt { name: "q2_k", bs: 84, qk: 256, f16s: &[80, 82] },
    Fmt { name: "q3_k", bs: 110, qk: 256, f16s: &[108] },
    Fmt { name: "q4_k", bs: 144, qk: 256, f16s: &[0, 2] },
    Fmt { name: "q5_k", bs: 176, qk: 256, f16s: &[0, 2] },
    Fmt { name: "q6_k", bs: 210, qk: 256, f16s: &[208] },
];
const DSTS: [(&str, usize); 3] = [("bf16", 2), ("f16", 2), ("f32", 4)];

// ------------------------------------------------------------------------------------------------
// test data

pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    pub fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
    /// An f16 scale: mostly normal magnitudes 2^-12..2^3, 1 in `srate` a special value or raw bits.
    pub fn f16_scale(&mut self, srate: u64) -> u16 {
        let r = self.next();
        match if r % srate == 0 { (r >> 40) % 2 } else { 2 } {
            0 => [0x0000, 0x8000, 0x7c00, 0xfc00, 0x7e00, 0x0001, 0x83ff, 0x7bff, 0x0400, 0xfe01, 0x7c01, 0x03ff][((r >> 8) % 12) as usize],
            1 => (r >> 16) as u16,
            _ => {
                let sign = ((r >> 8) & 1) as u16;
                let exp = 3 + ((r >> 9) % 16) as u16;
                (sign << 15) | (exp << 10) | ((r >> 16) & 0x3ff) as u16
            }
        }
    }
    /// A plain activation value, sometimes an edge case (nan, inf, denormal, huge, zero).
    pub fn f32v(&mut self, srate: u64) -> f32 {
        let r = self.next();
        if r % srate == 0 {
            [0.0, -0.0, 1e-40, -3e-39, 65504.0, 1e5, -2e6, f32::INFINITY, f32::NAN, -f32::INFINITY, 1.5e-38][((r >> 20) % 11) as usize]
        } else {
            ((r >> 11) as f64 / (1u64 << 53) as f64 * 16.0 - 8.0) as f32
        }
    }
}

/// `n` random weight blocks with f16 scale fields from `f16_scale`, plus trailing pad bytes.
fn blocks(rng: &mut Rng, f: &Fmt, n: usize, pad: usize, srate: u64) -> Vec<u8> {
    let mut b = rng.bytes(n * f.bs + pad);
    for i in 0..n {
        for &o in f.f16s {
            let v = rng.f16_scale(srate);
            b[i * f.bs + o..i * f.bs + o + 2].copy_from_slice(&v.to_le_bytes());
        }
    }
    b
}
/// `n` random Q8_1 blocks (36 bytes): random int8 quants, `ds` from `f16_scale`.
fn q8_1(rng: &mut Rng, n: usize, srate: u64) -> Vec<u8> {
    let mut b = rng.bytes(n * 36);
    for i in 0..n {
        let (d, s) = (rng.f16_scale(srate), rng.f16_scale(srate));
        b[i * 36..i * 36 + 2].copy_from_slice(&d.to_le_bytes());
        b[i * 36 + 2..i * 36 + 4].copy_from_slice(&s.to_le_bytes());
    }
    b
}

pub fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    if e == 0 {
        return f32::from_bits(((m as f32) * (2.0f32).powi(-24)).to_bits() | sign);
    }
    if e == 31 {
        return f32::from_bits(sign | 0x7f80_0000 | (m << 13));
    }
    f32::from_bits(sign | ((e + 112) << 23) | (m << 13))
}
/// f32 -> f16 bits (truncating; only used to make test data).
pub fn f32_to_f16_trunc(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    if x.is_nan() {
        return sign | 0x7e00;
    }
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = (b & 0x7fffff) | 0x800000;
        return sign | (m >> (14 - e)) as u16;
    }
    sign | ((e as u16) << 10) | ((b >> 13) & 0x3ff) as u16
}

/// Encode f32 values as element type `t` (0 bf16, 1 f16, 2 f32); `raw` 1-in-N raw 16-bit patterns.
pub fn encode(rng: &mut Rng, t: usize, v: &[f32], raw: u64) -> Vec<u8> {
    match t {
        2 => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        _ => v
            .iter()
            .flat_map(|&x| {
                let r = rng.next();
                let h = if raw > 0 && r % raw == 0 {
                    r as u16
                } else if t == 0 {
                    (x.to_bits() >> 16) as u16
                } else {
                    f32_to_f16_trunc(x)
                };
                h.to_le_bytes()
            })
            .collect(),
    }
}

/// roundf(t) as nvcc emits it (`trunc(add.rz(t, copysign(0.5, t)))`) and with add.rn instead.
fn round_rz_rn(t: f32) -> (i64, i64) {
    let h = if t.is_sign_negative() { -0.5f32 } else { 0.5 };
    let exact = t as f64 + h as f64;
    let mut rz = exact as f32;
    if (rz as f64).abs() > exact.abs() {
        rz = f32::from_bits(rz.to_bits() - 1);
    }
    (rz.trunc() as i64, (t + h).trunc() as i64)
}

/// Host model of the fast-math quotient `x * rcp.approx(amax * (1/127))`: rcp.approx is not
/// modelled exactly, so the search only proposes candidates near rounding boundaries (x / d close
/// to k + 0.5); both sides see the same values, so an imprecise model only costs coverage.
fn tie_values(t: usize, n: usize, rng: &mut Rng) -> Vec<f32> {
    let to_f = |bits: u32| -> f32 {
        match t {
            0 => f32::from_bits(bits << 16),
            1 => f16_bits_to_f32(bits as u16),
            _ => f32::from_bits(bits),
        }
    };
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let abits = match t {
            0 => 0x0080 + (rng.next() % 0x7e00) as u32,
            1 => 0x0001 + (rng.next() % 0x7bfe) as u32,
            _ => 0x0080_0000 + (rng.next() % 0x7e00_0000) as u32,
        };
        let amax = to_f(abits);
        let d = amax * (1.0f32 / 127.0);
        if !(d > 0.0) || !d.is_finite() {
            continue;
        }
        out.push(amax);
        for l in 1..32 {
            let k = (rng.next() % 12) as f32;
            let target = (k + 0.5) * d;
            let base = match t {
                0 => target.to_bits() >> 16,
                1 => f32_to_f16_trunc(target) as u32,
                _ => target.to_bits(),
            };
            let dx = (rng.next() % 5) as i32 - 2;
            let mut x = to_f((base as i32 + dx) as u32);
            if !(x.abs() < amax) {
                x = 0.0;
            }
            let _ = round_rz_rn(x / d);
            out.push(if l % 2 == 0 { -x } else { x });
        }
    }
    out.truncate(n);
    out
}

// ------------------------------------------------------------------------------------------------
// raw device memory

pub struct Buf {
    pub ptr: u64,
    pub len: usize,
}
impl Buf {
    pub fn new(data: &[u8]) -> Buf {
        let mut ptr = 0u64;
        unsafe {
            check(sys::cuMemAlloc_v2(&mut ptr, data.len().max(1)), "cuMemAlloc");
            if !data.is_empty() {
                check(sys::cuMemcpyHtoD_v2(ptr, data.as_ptr() as *const c_void, data.len()), "HtoD");
            }
            // The launch stream may be non-blocking: wait for the copy before any kernel reads it.
            check(sys::cuCtxSynchronize(), "sync after HtoD");
        }
        Buf { ptr, len: data.len() }
    }
    pub fn read(&self) -> Vec<u8> {
        let mut v = vec![0u8; self.len];
        unsafe { check(sys::cuMemcpyDtoH_v2(v.as_mut_ptr() as *mut c_void, self.ptr, self.len), "DtoH") };
        v
    }
    pub fn p(&self) -> *mut c_void {
        self.ptr as *mut c_void
    }
}
impl Drop for Buf {
    fn drop(&mut self) {
        unsafe { sys::cuMemFree_v2(self.ptr) };
    }
}
pub fn check(r: sys::CUresult, what: &str) {
    if r != sys::cudaError_enum_CUDA_SUCCESS {
        panic!("{what}: CUDA error {r}");
    }
}
pub fn sync(what: &str) {
    let r = unsafe { sys::cuCtxSynchronize() };
    if r != sys::cudaError_enum_CUDA_SUCCESS {
        panic!("{what}: CUDA error {r} (runtime last error {})", unsafe { cref::cudaGetLastError() });
    }
}

// ------------------------------------------------------------------------------------------------
// tally

#[derive(Default)]
pub struct Tally {
    pub calls: usize,
    pub bytes: usize,
    pub failures: Vec<String>,
    /// family -> (calls, failures, cases where the C side changed its output)
    pub fam: BTreeMap<String, (usize, usize, usize)>,
}
impl Tally {
    pub fn cmp(&mut self, fam: &str, label: &str, init: &[u8], a: &[u8], b: &[u8]) {
        self.bytes += a.len();
        let e = self.fam.entry(fam.to_string()).or_default();
        if a != init {
            e.2 += 1;
        }
        let n = a.iter().zip(b).filter(|(x, y)| x != y).count();
        if n > 0 || a.len() != b.len() {
            e.1 += 1;
            let first = a.iter().zip(b).position(|(x, y)| x != y).unwrap_or(0);
            let fw = (first / 4 * 4).min(a.len().saturating_sub(4));
            let w = |v: &[u8]| if v.len() >= fw + 4 { u32::from_le_bytes([v[fw], v[fw + 1], v[fw + 2], v[fw + 3]]) } else { 0 };
            self.failures.push(format!("{label}: {n} of {} bytes differ (first at byte {first}: C {:#010x} Rust {:#010x})", a.len(), w(a), w(b)));
        }
    }
    pub fn calls(&mut self, fam: &str, n: usize) {
        self.calls += n;
        self.fam.entry(fam.to_string()).or_default().0 += n;
    }
}

pub struct G {
    pub rng: Rng,
    pub t: Tally,
    pub stream: *mut c_void,
    pub self_test: bool,
}

impl G {
    /// The stream for case `k`: the null (per-thread default) stream or the created one.
    pub fn s(&self, k: usize) -> *mut c_void {
        if k % 2 == 0 { std::ptr::null_mut() } else { self.stream }
    }

    // ---- mmvq --------------------------------------------------------------------------------

    fn srate_for(f: &Fmt, bpr: usize) -> u64 {
        (3 * bpr * (f.f16s.len() + 2 * f.qk / 32)).max(16) as u64
    }

    /// Plain launcher, identical weights / q8_1 / pre-filled dst for both launchers.
    #[allow(clippy::too_many_arguments)]
    fn plain(&mut self, fi: usize, di: usize, ncols: i32, nrows: i32, scy: i32, scd: i32, b: i32, k: usize) {
        let f = &FMTS[fi];
        let bpr = (ncols.max(0) as usize) / f.qk;
        let sr = Self::srate_for(f, bpr);
        // One spare row of blocks: 2-row CUDA blocks read row `nrows` when nrows is odd.
        let w = blocks(&mut self.rng, f, (nrows.max(0) as usize + 1) * bpr.max(1), 64, sr);
        let ncol_y = b.clamp(1, 8) as usize;
        let y = q8_1(&mut self.rng, (ncol_y - 1) * scy as usize + bpr * f.qk / 32 + 8, sr);
        let es = DSTS[di].1;
        let d0 = self.rng.bytes(((ncol_y - 1) * scd.max(0) as usize + nrows.max(0) as usize + 3) * es);
        let fns = mmvq_fns(di, fi);
        let (wd, yd) = (Buf::new(&w), Buf::new(&y));
        let s = self.s(k);
        let mut outs = vec![];
        for (side, f_) in [fns.plain.0, if self.self_test { fns.plain.0 } else { fns.plain.1 }].into_iter().enumerate() {
            let d = Buf::new(&d0);
            unsafe { f_(wd.p(), yd.p(), d.p(), ncols, nrows, scy, scd, b, s) };
            sync(&format!("plain side {side}"));
            outs.push(d.read());
        }
        self.t.calls("mmvq_plain", 2);
        let label = format!("plain {}_{} ncols={ncols} nrows={nrows} scy={scy} scd={scd} b={b} s={}", f.name, DSTS[di].0, k % 2);
        self.t.cmp("mmvq_plain", &label, &d0, &outs[0], &outs[1]);
    }

    #[allow(clippy::too_many_arguments)]
    fn glu(&mut self, fi: usize, di: usize, ncols: i32, nrows: i32, scy: i32, scd: i32, b: i32, act: i32, k: usize) {
        let f = &FMTS[fi];
        let bpr = (ncols.max(0) as usize) / f.qk;
        let sr = Self::srate_for(f, bpr);
        let wg = blocks(&mut self.rng, f, (nrows.max(0) as usize + 1) * bpr.max(1), 64, sr);
        let wu = blocks(&mut self.rng, f, (nrows.max(0) as usize + 1) * bpr.max(1), 64, sr);
        let ncol_y = b.clamp(1, 8) as usize;
        let y = q8_1(&mut self.rng, (ncol_y - 1) * scy as usize + bpr * f.qk / 32 + 8, sr);
        let es = DSTS[di].1;
        let d0 = self.rng.bytes(((ncol_y - 1) * scd.max(0) as usize + nrows.max(0) as usize + 3) * es);
        let fns = mmvq_fns(di, fi);
        let (gd, ud, yd) = (Buf::new(&wg), Buf::new(&wu), Buf::new(&y));
        let s = self.s(k);
        let mut outs = vec![];
        for f_ in [fns.glu.0, if self.self_test { fns.glu.0 } else { fns.glu.1 }] {
            let d = Buf::new(&d0);
            unsafe { f_(gd.p(), ud.p(), yd.p(), d.p(), ncols, nrows, scy, scd, b, act, s) };
            sync("glu");
            outs.push(d.read());
        }
        self.t.calls("mmvq_glu", 2);
        let label = format!("glu {}_{} ncols={ncols} nrows={nrows} scy={scy} scd={scd} b={b} act={act} s={}", f.name, DSTS[di].0, k % 2);
        self.t.cmp("mmvq_glu", &label, &d0, &outs[0], &outs[1]);
    }

    #[allow(clippy::too_many_arguments)]
    fn qkv(&mut self, fi: usize, di: usize, ncols: i32, nr: [i32; 3], scy: i32, b: i32, k: usize) {
        let f = &FMTS[fi];
        let bpr = (ncols.max(0) as usize) / f.qk;
        let sr = Self::srate_for(f, bpr);
        let ws: Vec<Vec<u8>> = (0..3).map(|m| blocks(&mut self.rng, f, (nr[m].max(0) as usize + 1) * bpr.max(1), 64, sr)).collect();
        let ncol_y = b.clamp(1, 8) as usize;
        let y = q8_1(&mut self.rng, (ncol_y - 1) * scy as usize + bpr * f.qk / 32 + 8, sr);
        let es = DSTS[di].1;
        let d0: Vec<Vec<u8>> = (0..3).map(|m| self.rng.bytes((ncol_y * nr[m].max(0) as usize + 3) * es)).collect();
        let fns = mmvq_fns(di, fi);
        let wd: Vec<Buf> = ws.iter().map(|w| Buf::new(w)).collect();
        let yd = Buf::new(&y);
        let s = self.s(k);
        let mut outs = vec![];
        for f_ in [fns.qkv.0, if self.self_test { fns.qkv.0 } else { fns.qkv.1 }] {
            let d: Vec<Buf> = d0.iter().map(|x| Buf::new(x)).collect();
            unsafe { f_(wd[0].p(), wd[1].p(), wd[2].p(), yd.p(), d[0].p(), d[1].p(), d[2].p(), ncols, nr[0], nr[1], nr[2], scy, b, s) };
            sync("qkv");
            outs.push(d.iter().map(|x| x.read()).collect::<Vec<_>>());
        }
        self.t.calls("mmvq_qkv", 2);
        for m in 0..3 {
            let label = format!("qkv {}_{} ncols={ncols} nrows={nr:?} scy={scy} b={b} s={} out={m}", f.name, DSTS[di].0, k % 2);
            self.t.cmp("mmvq_qkv", &label, &d0[m], &outs[0][m], &outs[1][m]);
        }
    }

    /// One quantize-launcher case. style: 0 random + raw patterns, 1 zeros, 2 ties at d = 1,
    /// 3 denormals, 4 plain, 5 rounding-boundary search.
    fn quant(&mut self, t: usize, kx: i32, kxp: i32, rows: i32, style: usize, k: usize) {
        let n = (rows * kx.max(1)) as usize;
        let v: Vec<f32> = match style {
            1 => vec![0.0; n],
            2 => (0..n).map(|i| if i % 32 == 0 { 127.0 } else { ((i % 64) as f32 - 32.0) * 0.5 }).collect(),
            3 => (0..n).map(|i| (i as f32 - 300.0) * 1e-38).collect(),
            5 => tie_values(t, n, &mut self.rng),
            4 => (0..n).map(|_| self.rng.f32v(1 << 40)).collect(),
            // whole 32-blocks of f32 denormals (bf16 / f32 only): abs.ftz / max.ftz flush them to 0
            6 => (0..n).map(|_| { let r = self.rng.next(); f32::from_bits((r as u32 & 0x807f_ffff) | if t == 0 { 0x0001_0000 } else { 1 }) }).collect(),
            _ => (0..n).map(|_| self.rng.f32v(13)).collect(),
        };
        let x = encode(&mut self.rng, t, &v, if style == 0 { 12 } else { 0 });
        let y0 = self.rng.bytes((rows * kxp) as usize / 32 * 36 + 36);
        let (cf, of) = quant_pair(t);
        let xd = Buf::new(&x);
        let s = self.s(k);
        let mut outs = vec![];
        for f_ in [cf, if self.self_test { cf } else { of }] {
            let y = Buf::new(&y0);
            unsafe { f_(xd.p(), y.p(), kx, kxp, rows, s) };
            sync("quantize");
            outs.push(y.read());
        }
        self.t.calls("mmvq_quantize", 2);
        let label = format!("quantize_q8_1_{} kx={kx} kxp={kxp} rows={rows} style={style} s={}", ["bf16", "f16", "f32"][t], k % 2);
        self.t.cmp("mmvq_quantize", &label, &y0, &outs[0], &outs[1]);
    }

    /// fast_mmvq's call sequence: quantize then GEMV (plain / glu / qkv), each side with its own
    /// launchers end to end.
    fn pipeline(&mut self, fi: usize, di: usize, kk: usize, nrows: usize, b: usize, kind: usize, k: usize) {
        let f = &FMTS[fi];
        let kp = kk.div_ceil(512) * 512;
        let bpr = kk / f.qk;
        let sr = Self::srate_for(f, bpr);
        let w: Vec<Vec<u8>> = (0..3).map(|_| blocks(&mut self.rng, f, (nrows + 1) * bpr, 64, sr)).collect();
        let v: Vec<f32> = (0..b * kk).map(|_| self.rng.f32v(3 * kk as u64)).collect();
        let x = encode(&mut self.rng, di, &v, 0);
        let es = DSTS[di].1;
        let d0 = self.rng.bytes(b * nrows * es);
        let (cq, oq) = quant_pair(di);
        let fns = mmvq_fns(di, fi);
        let wd: Vec<Buf> = w.iter().map(|w| Buf::new(w)).collect();
        let xd = Buf::new(&x);
        let s = self.s(k);
        let (ki, kpi, bi, ni, scy) = (kk as i32, kp as i32, b as i32, nrows as i32, (kp / 32) as i32);
        let mut outs = vec![];
        for side in 0..2 {
            let rust = side == 1 && !self.self_test;
            let y = Buf::new(&vec![0u8; b * kp / 32 * 36]);
            let d: Vec<Buf> = (0..3).map(|_| Buf::new(&d0)).collect();
            unsafe {
                (if rust { oq } else { cq })(xd.p(), y.p(), ki, kpi, bi, s);
                match kind {
                    0 => (if rust { fns.plain.1 } else { fns.plain.0 })(wd[0].p(), y.p(), d[0].p(), ki, ni, scy, ni, bi, s),
                    1 => (if rust { fns.glu.1 } else { fns.glu.0 })(wd[0].p(), wd[1].p(), y.p(), d[0].p(), ki, ni, scy, ni, bi, (k % 4) as i32, s),
                    _ => (if rust { fns.qkv.1 } else { fns.qkv.0 })(
                        wd[0].p(), wd[1].p(), wd[2].p(), y.p(), d[0].p(), d[1].p(), d[2].p(), ki, ni, ni, ni / 2, scy, bi, s,
                    ),
                }
            }
            sync("pipeline");
            outs.push(d.iter().map(|x| x.read()).collect::<Vec<_>>());
        }
        self.t.calls("mmvq_pipeline", 4);
        for m in 0..3 {
            let label = format!("pipeline kind={kind} {}_{} k={kk} nrows={nrows} b={b} s={} out={m}", f.name, DSTS[di].0, k % 2);
            self.t.cmp("mmvq_pipeline", &label, &d0, &outs[0][m], &outs[1][m]);
        }
    }
}

fn mmvq_suite(g: &mut G) {
    let mut k = 0usize;
    for fi in 0..FMTS.len() {
        let qk = FMTS[fi].qk as i32;
        for di in 0..3 {
            for b in 0..=9 {
                // (ncols, nrows, extra y stride, extra dst stride)
                let shapes: [(i32, i32, i32, i32); 4] =
                    [(qk * 3, 7, 0, 0), (qk * 17 + qk / 2 + 3, 33, 5, 3), (qk * 2, 1, 16, 1), (if qk == 32 { 4096 } else { 4608 }, 131, 16, 0)];
                for &(ncols, nrows, ey, ed) in shapes.iter() {
                    let scy = (ncols + 511) / 512 * 512 / 32 + ey;
                    let scd = nrows + ed;
                    k += 1;
                    g.plain(fi, di, ncols, nrows, scy, scd, b, k);
                    k += 1;
                    g.glu(fi, di, ncols, nrows, scy, scd, b, (k % 6) as i32 - 1, k);
                    // q / k / v row counts: unequal (packed grid) and near-equal (grid.y = 3)
                    let nr = if k % 2 == 0 { [nrows, nrows / 3 + 1, nrows / 5 + 1] } else { [nrows, nrows + 1, nrows.max(2) - 1] };
                    k += 1;
                    g.qkv(fi, di, ncols, nr, scy, b, k);
                }
            }
            // larger row counts at the batch sizes that switch geometry, every activation
            for &b in &[1, 2, 4, 5, 8] {
                k += 1;
                g.plain(fi, di, qk * 8, 1025, qk * 8 / 32 + 3, 1030, b, k);
                for act in 0..4 {
                    k += 1;
                    g.glu(fi, di, qk * 8, 257, qk * 8 / 32 + 3, 260, b, act, k);
                }
                k += 1;
                g.qkv(fi, di, qk * 8, [1024, 256, 256], qk * 8 / 32 + 3, b, k);
                k += 1;
                g.qkv(fi, di, qk * 8, [300, 280, 290], qk * 8 / 32 + 3, b, k);
            }
            // ncols smaller than one block (no loop iterations), one block, empty grids
            for &b in &[1, 3, 6] {
                k += 1;
                g.plain(fi, di, qk - 1, 5, 4, 9, b, k);
                k += 1;
                g.glu(fi, di, qk, 6, 9, 6, b, 3, k);
                k += 1;
                g.qkv(fi, di, qk, [6, 2, 0], 9, b, k);
            }
            for &b in &[1, 5] {
                k += 1;
                g.plain(fi, di, qk * 2, 0, 8, 4, b, k);
                k += 1;
                g.glu(fi, di, qk * 2, 0, 8, 4, b, 0, k);
            }
            for &(kk, n, b, kind) in &[(2048usize, 96usize, 1usize, 0usize), (1536, 61, 3, 1), (4096, 40, 8, 2), (2560, 17, 5, 0), (2048, 64, 1, 1), (1024, 33, 1, 2)] {
                k += 1;
                g.pipeline(fi, di, kk, n, b, kind, k);
            }
        }
    }
}

fn mmvq_quant_suite(g: &mut G) {
    let mut k = 0;
    for t in 0..3 {
        for &(kx, kxp) in &[(512, 512), (1000, 1024), (1, 512), (511, 512), (2048, 2560), (4096, 4096), (0, 512), (96, 96), (70, 96), (3000, 3072)] {
            for &rows in &[1, 2, 5, 8, 13] {
                for style in 0..7 {
                    if style > 0 && rows > 2 {
                        continue;
                    }
                    k += 1;
                    g.quant(t, kx, kxp, rows, style, k);
                }
            }
        }
        for _ in 0..40 {
            k += 1;
            g.quant(t, 4096, 4096, 4, 5, k);
        }
    }
}

// ------------------------------------------------------------------------------------------------
// reference instance coverage

/// Every kernel entry of the reference cubins, as the key `crate::launch` records for its twin.
fn reference_instances() -> Vec<(String, String)> {
    let root = format!("{}/titan-engine/oxide-kernels", std::env::var("HOME").unwrap());
    let mut files = vec!["mmvq_gguf".to_string(), "mmq_quantize".to_string()];
    for t in rs::types() {
        files.push(format!("mmq_instance_{t}"));
    }
    let mut out = vec![];
    for f in files {
        let o = std::process::Command::new(format!("{root}/tools/cuobjdump"))
            .arg("-symbols")
            .arg(format!("{root}/reference/mistralrs-quant/{f}.cubin"))
            .output()
            .expect("cuobjdump -symbols");
        for line in String::from_utf8_lossy(&o.stdout).lines() {
            if !line.contains("STO_ENTRY") {
                continue;
            }
            let name = line.split_whitespace().last().unwrap().to_string();
            out.push((f.clone(), name));
        }
    }
    out
}

/// The launch key of a reference entry name (mangled for the MMQ templates).
pub fn key_of(name: &str) -> String {
    let tname = |id: u32| match id {
        2 => "q4_0",
        3 => "q4_1",
        6 => "q5_0",
        7 => "q5_1",
        8 => "q8_0",
        10 => "q2_k",
        11 => "q3_k",
        12 => "q4_k",
        13 => "q5_k",
        14 => "q6_k",
        _ => "?",
    };
    let num_after = |s: &str, pat: &str| -> u32 {
        let i = s.find(pat).unwrap() + pat.len();
        s[i..].chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap()
    };
    let layout = |s: &str| ["d4", "ds4", "d2s6"][num_after(s, "mmq_q8_1_ds_layout") as usize];
    if let Some(rest) = name.strip_prefix("_Z9mul_mat_qI") {
        return format!("mmq_{}_x{}_nc{}", tname(num_after(rest, "ggml_type")), num_after(rest, "ELi"), num_after(rest, "ELb"));
    }
    if let Some(rest) = name.strip_prefix("_Z24mul_mat_q_stream_k_fixupI") {
        return format!("mmq_fixup_{}_x{}_nc{}", tname(num_after(rest, "ggml_type")), num_after(rest, "ELi"), num_after(rest, "ELb"));
    }
    if name.starts_with("_Z25quantize_mmq_q8_1_glu_f32") {
        return format!("quantize_mmq_q8_1_glu_f32_{}", layout(name));
    }
    if let Some(rest) = name.strip_prefix("_Z17quantize_mmq_q8_1I") {
        let t = if rest.starts_with("13__nv_bfloat16") {
            "bf16"
        } else if rest.starts_with("6__half") {
            "f16"
        } else {
            "f32"
        };
        return format!("quantize_mmq_q8_1_{t}_{}", layout(name));
    }
    name.to_string()
}

/// MMQ instances no launcher can reach with cc >= 1200 (sm_120): tile widths that are not a
/// multiple of the mma granularity (16 from 48 on) are never selected, and their mul_mat_q
/// bodies are NO_DEVICE_CODE traps.
fn unreachable_key(key: &str) -> bool {
    ["_x56_", "_x72_", "_x88_", "_x104_", "_x120_"].iter().any(|x| key.contains(x))
}

fn coverage(families_run: &HashSet<String>, t: &mut Tally) {
    let launched = rs::LAUNCHED.lock().unwrap().clone().unwrap_or_default();
    let mut per: BTreeMap<String, (usize, usize, usize, Vec<String>)> = BTreeMap::new();
    for (file, name) in reference_instances() {
        let key = key_of(&name);
        let fam = if file.starts_with("mmq_instance_") { file.trim_start_matches("mmq_instance_").to_string() } else { file.clone() };
        let e = per.entry(fam).or_default();
        e.0 += 1;
        if launched.contains(&key) {
            e.1 += 1;
        } else if unreachable_key(&key) {
            e.2 += 1;
        } else {
            e.3.push(key);
        }
    }
    for (fam, (n, hit, unr, missing)) in per {
        let run = families_run.contains(&fam);
        println!("  instances {fam}: {n} in cubin, {hit} launched via Rust twin, {unr} unreachable on sm_120, {} missing{}", missing.len(),
                 if run { "" } else { " (family not run)" });
        if run && !missing.is_empty() {
            t.failures.push(format!("{fam}: {} reference instances never launched: {:?}", missing.len(), &missing[..missing.len().min(8)]));
        }
    }
}

pub fn run() -> i32 {
    let only: Option<Vec<String>> = std::env::var("MRQC_ONLY").ok().map(|s| s.split(',').map(|x| x.to_string()).collect());
    let want = |f: &str| only.as_ref().is_none_or(|o| o.iter().any(|n| n == f));
    unsafe {
        check(cref::cudaSetDevice(0) as u32, "cudaSetDevice");
        check(cref::cudaFree(std::ptr::null_mut()) as u32, "cudaFree(0) runtime init");
    }
    // The runtime made the primary context current; use it (the same one mistral.rs uses).
    let ctx = cuda_core::CudaContext::new(0).expect("primary context");
    ctx.bind_to_thread().expect("bind");
    let stream_owner = ctx.new_stream().expect("stream");
    let mut g = G { rng: Rng(0x3317_9A1D_C0DE), t: Tally::default(), stream: stream_owner.cu_stream() as *mut c_void, self_test: std::env::var("MRQC_SELF").is_ok() };
    let mut run: HashSet<String> = HashSet::new();

    if want("mmvq") {
        mmvq_suite(&mut g);
    }
    if want("quantize") {
        mmvq_quant_suite(&mut g);
    }
    if want("mmvq") && want("quantize") {
        run.insert("mmvq_gguf".into());
    }
    crate::gate_mmq::run(&mut g, &only, &mut run);

    for (fam, (calls, fails, wrote)) in &g.t.fam {
        println!("  {fam}: {calls} launcher calls, {fails} failing cases (C changed its output in {wrote} cases)");
    }
    coverage(&run, &mut g.t);
    for f in g.t.failures.iter().take(40) {
        println!("  FAIL {f}");
    }
    if let Ok(p) = std::env::var("MRQC_FAILLOG") {
        std::fs::write(&p, g.t.failures.join("\n")).ok();
    }
    let ok = g.t.failures.is_empty();
    println!("mistralrs-quant-c: {} launcher calls, {} bytes compared, {} failing -> {}", g.t.calls, g.t.bytes, g.t.failures.len(),
             if ok { "PASS" } else { "FAIL" });
    drop(stream_owner);
    if ok { 0 } else { 1 }
}
