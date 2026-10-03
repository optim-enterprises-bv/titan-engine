//! Differential gate: every oxide entry against the reference instance in the rebuilt llama.cpp
//! cubins (ref/mmvq.cubin, ref/convert.cubin; SASS identical to libggml-cuda.so). Both sides get
//! the same packed parameter buffer (CU_LAUNCH_PARAM_BUFFER_POINTER), private copies of the same
//! input bytes and a pre-filled output buffer; every output byte is compared.
use cuda_core::sys as cu;
use kdiff::{Rng, Tally};
use std::ffi::c_void;

const REF_MMVQ: &str = "ref/mmvq.cubin";
const REF_CONVERT: &str = "ref/convert.cubin";
const OXIDE: &str = "iq3_xxs.ptx";

fn mmvq_ref_name(nc: usize, small_k: bool) -> String {
    format!(
        "_Z13mul_mat_vec_qIL9ggml_type18ELi{nc}ELb0ELb{}ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj",
        small_k as u32
    )
}
fn mmvq_ox_name(nc: usize, small_k: bool) -> String {
    if small_k { "iq3_xxs_mmvq_1_small_k".into() } else { format!("iq3_xxs_mmvq_{nc}") }
}
const DEQ_REF: [&str; 3] = [
    "_Z24dequantize_block_iq3_xxsIfEvPKvPT_",
    "_Z24dequantize_block_iq3_xxsI6__halfEvPKvPT_",
    "_Z24dequantize_block_iq3_xxsI13__nv_bfloat16EvPKvPT_",
];
const DEQ_OX: [&str; 3] = ["iq3_xxs_dequant_f32", "iq3_xxs_dequant_f16", "iq3_xxs_dequant_bf16"];
/// Bytes per IQ3_XXS block (half d, 72 grid-code bytes, 24 bytes of group scales + sign masks) and
/// values per block (QK_K). 8 q8_1 blocks per 256-value X block.
const BB: usize = 98;
const QK: u32 = 256;

/// llama.cpp `init_fastdiv_values`: <mp, L, d>.
pub fn fastdiv_values(d: u64) -> [u32; 3] {
    assert!(d != 0 && d <= u32::MAX as u64);
    let d = d as u32;
    let mut l = 0u32;
    while l < 32 && (1u64 << l) < d as u64 {
        l += 1;
    }
    let mp = (((1u64 << 32) * ((1u64 << l) - d as u64)) / d as u64 + 1) as u32;
    [mp, l, d]
}

macro_rules! ck {
    ($e:expr) => {{
        let r = $e;
        assert_eq!(r, cu::cudaError_enum_CUDA_SUCCESS, "{}", stringify!($e));
    }};
}

struct Mods {
    mmvq: cu::CUmodule,
    convert: cu::CUmodule,
    oxide: cu::CUmodule,
}

fn load(path: &str) -> cu::CUmodule {
    let mut img = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    img.push(0);
    let mut m: cu::CUmodule = std::ptr::null_mut();
    unsafe { ck!(cu::cuModuleLoadData(&mut m, img.as_ptr() as *const c_void)) };
    m
}

fn func(m: cu::CUmodule, name: &str) -> cu::CUfunction {
    let mut f: cu::CUfunction = std::ptr::null_mut();
    let c = std::ffi::CString::new(name).unwrap();
    let r = unsafe { cu::cuModuleGetFunction(&mut f, m, c.as_ptr()) };
    assert_eq!(r, cu::cudaError_enum_CUDA_SUCCESS, "missing function {name}");
    f
}

/// Packed kernel parameters, laid out with C alignment.
#[derive(Default, Clone)]
pub struct Params(pub Vec<u8>);
impl Params {
    fn put<T: Copy>(&mut self, v: T) -> &mut Self {
        let a = std::mem::align_of::<T>();
        while self.0.len() % a != 0 {
            self.0.push(0);
        }
        let b = unsafe { std::slice::from_raw_parts(&v as *const T as *const u8, std::mem::size_of::<T>()) };
        self.0.extend_from_slice(b);
        self
    }
}

/// Device copies of `bufs`, parameter bytes with `Ptr(i)` placeholders patched, launch, sync,
/// copy back `outs`.
fn launch(f: cu::CUfunction, grid: (u32, u32, u32), block: (u32, u32, u32), params: &Params, ptr_slots: &[(usize, Option<usize>)], bufs: &[Vec<u8>], outs: &[usize]) -> Vec<Vec<u8>> {
    unsafe {
        let mut dptrs = Vec::new();
        for b in bufs {
            let mut p: cu::CUdeviceptr = 0;
            ck!(cu::cuMemAlloc_v2(&mut p, b.len().max(1)));
            ck!(cu::cuMemcpyHtoD_v2(p, b.as_ptr() as *const c_void, b.len()));
            dptrs.push(p);
        }
        let mut pb = params.0.clone();
        for &(off, bi) in ptr_slots {
            let v: u64 = bi.map(|i| dptrs[i]).unwrap_or(0);
            pb[off..off + 8].copy_from_slice(&v.to_le_bytes());
        }
        let mut size = pb.len();
        let mut extra: [*mut c_void; 5] = [
            1 as *mut c_void, // CU_LAUNCH_PARAM_BUFFER_POINTER
            pb.as_mut_ptr() as *mut c_void,
            2 as *mut c_void, // CU_LAUNCH_PARAM_BUFFER_SIZE
            &mut size as *mut usize as *mut c_void,
            std::ptr::null_mut(), // CU_LAUNCH_PARAM_END
        ];
        ck!(cu::cuLaunchKernel(f, grid.0, grid.1, grid.2, block.0, block.1, block.2, 0, std::ptr::null_mut(), std::ptr::null_mut(), extra.as_mut_ptr()));
        ck!(cu::cuCtxSynchronize());
        let mut res = Vec::new();
        for &o in outs {
            let mut h = vec![0u8; bufs[o].len()];
            ck!(cu::cuMemcpyDtoH_v2(h.as_mut_ptr() as *mut c_void, dptrs[o], h.len()));
            res.push(h);
        }
        for p in dptrs {
            ck!(cu::cuMemFree_v2(p));
        }
        res
    }
}

/// Largest / mean distance in representable floats between two f32 buffers.
fn ulp(a: &[u8], b: &[u8]) -> (u32, f64) {
    let f = |v: &[u8], i: usize| f32::from_le_bytes([v[i], v[i + 1], v[i + 2], v[i + 3]]);
    let (mut mx, mut acc, mut n) = (0u32, 0f64, 0u64);
    for i in (0..a.len().min(b.len())).step_by(4) {
        let (x, y) = (f(a, i), f(b, i));
        let d = match (x.is_finite(), y.is_finite()) {
            (true, true) => (x.to_bits() as i64 - y.to_bits() as i64).unsigned_abs() as u32,
            (false, false) => 0,
            _ => u32::MAX,
        };
        mx = mx.max(d);
        acc += d as f64;
        n += 1;
    }
    (mx, if n == 0 { 0.0 } else { acc / n as f64 })
}


/// Tables for the host arbiter, copied from the device chain.
pub static GRID: [u32; 256] = [0x4040404,0x4040414,0x4040424,0x4040c0c,0x4040c1c,0x4040c3e,0x4041404,0x4041414,0x4041c0c,0x4042414,0x4043e1c,0x4043e2c,0x40c040c,0x40c041c,0x40c0c04,0x40c0c14,0x40c140c,0x40c142c,0x40c1c04,0x40c1c14,0x40c240c,0x40c2c24,0x40c3e04,0x4140404,0x4140414,0x4140424,0x4140c0c,0x4141404,0x4141414,0x4141c0c,0x4141c1c,0x4141c3e,0x4142c0c,0x4142c3e,0x4143e2c,0x41c040c,0x41c043e,0x41c0c04,0x41c0c14,0x41c142c,0x41c3e04,0x4240c1c,0x4241c3e,0x4242424,0x4242c3e,0x4243e1c,0x4243e2c,0x42c040c,0x42c043e,0x42c1c14,0x42c2c14,0x4341c2c,0x4343424,0x43e0c04,0x43e0c24,0x43e0c34,0x43e241c,0x43e340c,0xc04040c,0xc04041c,0xc040c04,0xc040c14,0xc04140c,0xc04141c,0xc041c04,0xc041c14,0xc041c24,0xc04243e,0xc042c04,0xc0c0404,0xc0c0414,0xc0c0c0c,0xc0c1404,0xc0c1414,0xc14040c,0xc14041c,0xc140c04,0xc140c14,0xc14140c,0xc141c04,0xc143e14,0xc1c0404,0xc1c0414,0xc1c1404,0xc1c1c0c,0xc1c2434,0xc1c3434,0xc24040c,0xc24042c,0xc242c04,0xc2c1404,0xc2c1424,0xc2c2434,0xc2c3e0c,0xc34042c,0xc3e1414,0xc3e2404,0x14040404,0x14040414,0x14040c0c,0x14040c1c,0x14041404,0x14041414,0x14041434,0x14041c0c,0x14042414,0x140c040c,0x140c041c,0x140c042c,0x140c0c04,0x140c0c14,0x140c140c,0x140c1c04,0x140c341c,0x140c343e,0x140c3e04,0x14140404,0x14140414,0x14140c0c,0x14140c3e,0x14141404,0x14141414,0x14141c3e,0x14142404,0x14142c2c,0x141c040c,0x141c0c04,0x141c0c24,0x141c3e04,0x141c3e24,0x14241c2c,0x14242c1c,0x142c041c,0x142c143e,0x142c240c,0x142c3e24,0x143e040c,0x143e041c,0x143e0c34,0x143e242c,0x1c04040c,0x1c040c04,0x1c040c14,0x1c04140c,0x1c04141c,0x1c042c04,0x1c04342c,0x1c043e14,0x1c0c0404,0x1c0c0414,0x1c0c1404,0x1c0c1c0c,0x1c0c2424,0x1c0c2434,0x1c14040c,0x1c14041c,0x1c140c04,0x1c14142c,0x1c142c14,0x1c143e14,0x1c1c0c0c,0x1c1c1c1c,0x1c241c04,0x1c24243e,0x1c243e14,0x1c2c0404,0x1c2c0434,0x1c2c1414,0x1c2c2c2c,0x1c340c24,0x1c341c34,0x1c34341c,0x1c3e1c1c,0x1c3e3404,0x24040424,0x24040c3e,0x24041c2c,0x24041c3e,0x24042c1c,0x24042c3e,0x240c3e24,0x24141404,0x24141c3e,0x24142404,0x24143404,0x24143434,0x241c043e,0x241c242c,0x24240424,0x24242c0c,0x24243424,0x242c142c,0x242c241c,0x242c3e04,0x243e042c,0x243e0c04,0x243e0c14,0x243e1c04,0x2c040c14,0x2c04240c,0x2c043e04,0x2c0c0404,0x2c0c0434,0x2c0c1434,0x2c0c2c2c,0x2c140c24,0x2c141c14,0x2c143e14,0x2c1c0414,0x2c1c2c1c,0x2c240c04,0x2c24141c,0x2c24143e,0x2c243e14,0x2c2c0414,0x2c2c1c0c,0x2c342c04,0x2c3e1424,0x2c3e2414,0x34041424,0x34042424,0x34042434,0x34043424,0x340c140c,0x340c340c,0x34140c3e,0x34143424,0x341c1c04,0x341c1c34,0x34242424,0x342c042c,0x342c2c14,0x34341c1c,0x343e041c,0x343e140c,0x3e04041c,0x3e04042c,0x3e04043e,0x3e040c04,0x3e041c14,0x3e042c14,0x3e0c1434,0x3e0c2404,0x3e140c14,0x3e14242c,0x3e142c14,0x3e1c0404,0x3e1c0c2c,0x3e1c1c1c,0x3e1c3404,0x3e24140c,0x3e24240c,0x3e2c0404,0x3e2c0414,0x3e2c1424,0x3e341c04];
pub static KSIGNS: [u8; 128] = [0,129,130,3,132,5,6,135,136,9,10,139,12,141,142,15,144,17,18,147,20,149,150,23,24,153,154,27,156,29,30,159,160,33,34,163,36,165,166,39,40,169,170,43,172,45,46,175,48,177,178,51,180,53,54,183,184,57,58,187,60,189,190,63,192,65,66,195,68,197,198,71,72,201,202,75,204,77,78,207,80,209,210,83,212,85,86,215,216,89,90,219,92,221,222,95,96,225,226,99,228,101,102,231,232,105,106,235,108,237,238,111,240,113,114,243,116,245,246,119,120,249,250,123,252,125,126,255];
pub static MASK: [u8; 8] = [1,2,4,8,16,32,64,128];

/// Host transcription of llama.cpp's `dequantize_iq3_xxs` (dequantize.cuh) for arbitering the cubin.
pub fn cpu_deq_iq3_xxs(x: &[u8], nvals: usize, out_bytes: &mut [u8], es: usize) {
    let q = |v: u32| -> f32 { f32::from_bits(v) };
    let nb = nvals / 256;
    for blk in 0..nb {
        let xb = &x[blk * 98..blk * 98 + 98];
        let d0 = f16_to_f32(u16::from_le_bytes([xb[0], xb[1]]));
        for tid in 0..32usize {
            let il = tid / 8;
            let ib = tid % 8;
            let q3 = &xb[2 + 8 * ib..];
            let gas = 2 + 64 + 4 * ib;
            let aux32 = u16::from_le_bytes([xb[gas], xb[gas + 1]]) as u32
                | ((u16::from_le_bytes([xb[gas + 2], xb[gas + 3]]) as u32) << 16);
            let d = d0 * (0.5 + ((aux32 >> 28) as f32)) * 0.5;
            let signs = KSIGNS[((aux32 >> (7 * il as u32)) & 127) as usize];
            for j in 0..4 {
                let g1 = GRID[q3[2 * il] as usize];
                let g2 = GRID[q3[2 * il + 1] as usize];
                let b1 = ((g1 >> (8 * j)) & 0xFF) as f32;
                let b2 = ((g2 >> (8 * j)) & 0xFF) as f32;
                let s0 = if signs & MASK[j] != 0 { -1.0 } else { 1.0 };
                let s1 = if signs & MASK[j + 4] != 0 { -1.0 } else { 1.0 };
                let v1 = d * b1 * s0;
                let v2 = d * b2 * s1;
                let yb = (blk * 256 + 32 * ib + 8 * il + j) * es;
                let yb2 = (blk * 256 + 32 * ib + 8 * il + j + 4) * es;
                for (off, v) in [(yb, v1), (yb2, v2)] {
                    if es == 4 {
                        out_bytes[off..off + 4].copy_from_slice(&v.to_le_bytes());
                    } else if es == 2 {
                        out_bytes[off..off + 2].copy_from_slice(&f16_bits_of(v).to_le_bytes());
                    }
                }
            }
        }
    }
}
pub fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1F) as u32;
    let m = (h & 0x3FF) as u32;
    let bits = if e == 0 {
        if m == 0 { s << 31 } else {
            let mut m = m; let mut e = 127 - 15 + 1;
            while m & 0x400 == 0 { m <<= 1; e -= 1; }
            (s << 31) | ((e as u32) << 23) | ((m & 0x3FF) << 13)
        }
    } else if e == 31 { (s << 31) | 0x7F80_0000 | (m << 13) }
    else { (s << 31) | ((e + 127 - 15) << 23) | (m << 13) };
    f32::from_bits(bits)
}
pub fn f16_bits_of(v: f32) -> u16 {
    // round-to-nearest-even f32 -> f16 (matches cvt.rn.f16.f32 for the values here)
    let b = v.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    let m = b & 0x7F_FFFF;
    if e >= 31 { return s | 0x7C00; }
    if e <= 0 {
        if e < -10 { return s; }
        let m = (m | 0x80_0000) >> (14 - e);
        let mut r = (m >> 13) as u16;
        if m & 0x1FFF > 0x1000 || (m & 0x1FFF == 0x1000 && r & 1 == 1) { r += 1; }
        return s | r;
    }
    let mut r = ((e as u32) << 10 | (m >> 13)) as u16;
    if m & 0x1FFF > 0x1000 || (m & 0x1FFF == 0x1000 && r & 1 == 1) { r += 1; }
    s | r
}

fn compare(t: &mut Tally, label: &str, a: &[Vec<u8>], b: &[Vec<u8>], f32_buf: Option<usize>) {
    let mut d = kdiff::Diff { bytes: 0, differing: 0, first: None };
    for (k, (x, y)) in a.iter().zip(b).enumerate() {
        d.bytes += x.len();
        for i in 0..x.len() {
            if x[i] != y[i] {
                d.differing += 1;
                if d.first.is_none() {
                    d.first = Some((k, i, x[i], y[i]));
                }
            }
        }
    }
    if let Some(i) = f32_buf {
        let (mx, mean) = ulp(&a[i], &b[i]);
        t.record(&format!("{label} [ulp max={mx} mean={mean:.2}]"), &d);
    } else {
        t.record(label, &d);
    }
}

/// Random f16 scale bits: mostly normal magnitudes, `1/rate` specials (0, -0, inf, nan, denormal, max).
fn f16_bits(r: &mut Rng, rate: u64) -> u16 {
    const SP: [u16; 8] = [0x0000, 0x8000, 0x7C00, 0xFC00, 0x7E00, 0x0001, 0x83FF, 0x7BFF];
    if r.next() % rate == 0 {
        return SP[(r.next() % 8) as usize];
    }
    (0x1800 + (r.next() % 0x3000) as u16) | (((r.next() & 1) as u16) << 15)
}

fn x_blocks(r: &mut Rng, n: usize, rate: u64) -> Vec<u8> {
    let mut v = Vec::with_capacity(n * BB);
    for _ in 0..n {
        v.extend_from_slice(&f16_bits(r, rate).to_le_bytes());
        for _ in 0..72 {
            v.push(r.next() as u8);
        }
        for _ in 0..24 {
            v.push(r.next() as u8);
        }
    }
    v
}

fn q8_1_blocks(r: &mut Rng, n: usize, rate: u64) -> Vec<u8> {
    let mut v = Vec::with_capacity(n * 36);
    for _ in 0..n {
        v.extend_from_slice(&f16_bits(r, rate).to_le_bytes());
        v.extend_from_slice(&f16_bits(r, rate).to_le_bytes());
        for _ in 0..32 {
            v.push(r.next() as u8);
        }
    }
    v
}

struct MmvqCase {
    nc: usize,
    small_k: bool,
    bpr: u32,
    nrows: u32,
    pad_row: u32,
    pad_dst: u32,
    nch_dst: u32,
    nch_x: u32,
    nsamp_dst: u32,
    nsamp_x: u32,
    ids: bool,
}

fn mmvq_case(m: &Mods, t: &mut Tally, r: &mut Rng, c: &MmvqCase, rate: u64) {
    let nwarps: u32 = if c.nc <= 4 { 4 } else { 2 };
    let rpb: u32 = if c.nc == 1 { if c.small_k { nwarps } else { 1 } } else { 2 };
    let stride_row_x = c.bpr + c.pad_row;
    let stride_channel_x = (c.nrows + rpb) * stride_row_x;
    let stride_sample_x = c.nch_x * stride_channel_x;
    let nch_y = if c.ids { c.nch_x.max(2) } else { c.nch_dst };
    // IQ3_XXS blocks hold QK (256) values, so the q8_1 side has QK/32 blocks per X block.
    let yf: u32 = QK / 32;
    let stride_col_y = yf * (c.bpr + c.pad_row);
    let stride_channel_y = c.nc as u32 * stride_col_y + 3;
    let stride_sample_y = nch_y * stride_channel_y;
    let stride_col_dst = c.nrows + c.pad_dst;
    let stride_channel_dst = c.nc as u32 * stride_col_dst + 1;
    let stride_sample_dst = c.nch_dst * stride_channel_dst;

    let nsamp_x = c.nsamp_x;
    let x = x_blocks(r, (nsamp_x * stride_sample_x + 4) as usize, rate);
    let y = q8_1_blocks(r, (c.nsamp_dst * stride_sample_y + 8) as usize, rate);
    let dst = r.bytes((c.nsamp_dst * stride_sample_dst) as usize * 4 + 64);
    let ids: Vec<u8> = (0..c.nch_dst).flat_map(|_| ((r.next() % c.nch_x as u64) as i32).to_le_bytes()).collect();

    let ncy = if c.ids { fastdiv_values(nch_y as u64) } else { [0, 0, 0] };
    let cr = if c.ids { [0, 0, 0] } else { fastdiv_values((c.nch_dst / c.nch_x) as u64) };
    let sr = fastdiv_values((c.nsamp_dst / nsamp_x) as u64);

    let mut p = Params::default();
    let mut slots = vec![];
    slots.push((p.0.len(), Some(0)));
    p.put(0u64);
    slots.push((p.0.len(), Some(1)));
    p.put(0u64);
    slots.push((p.0.len(), if c.ids { Some(3) } else { None }));
    p.put(0u64);
    // fusion args (all null: has_fusion = false instances never read them)
    for _ in 0..5 {
        p.put(0u64);
    }
    p.put(0u32).put(0f32);
    slots.push((p.0.len(), Some(2)));
    p.put(0u64);
    p.put(c.bpr * QK);
    p.put(ncy[0]).put(ncy[1]).put(ncy[2]);
    p.put(stride_row_x).put(stride_col_y).put(stride_col_dst);
    p.put(cr[0]).put(cr[1]).put(cr[2]);
    p.put(stride_channel_x).put(stride_channel_y).put(stride_channel_dst);
    p.put(sr[0]).put(sr[1]).put(sr[2]);
    p.put(stride_sample_x).put(stride_sample_y).put(stride_sample_dst);
    p.put(0u32);
    assert_eq!(p.0.len(), 160);

    let grid = (c.nrows.div_ceil(rpb), c.nch_dst, c.nsamp_dst);
    let block = (32, nwarps, 1);
    let (yb, xb) = (y.len() / 36, x.len() / BB);
    println!("CASE mmvq nc={} small_k={} bpr={} nrows={} pad={}/{} ch={}/{} samp={}/{} ids={} yblocks={} xblocks={}",
        c.nc, c.small_k, c.bpr, c.nrows, c.pad_row, c.pad_dst, c.nch_dst, c.nch_x, c.nsamp_dst, nsamp_x, c.ids, yb, xb);
    let bufs = vec![x, y, dst, ids];
    let a = launch(func(m.mmvq, &mmvq_ref_name(c.nc, c.small_k)), grid, block, &p, &slots, &bufs, &[2]);
    let b = launch(func(m.oxide, &mmvq_ox_name(c.nc, c.small_k)), grid, block, &p, &slots, &bufs, &[2]);
    let label = format!(
        "mmvq nc={} small_k={} bpr={} nrows={} pads={}/{} ch={}/{} samp={}/{} ids={}",
        c.nc, c.small_k, c.bpr, c.nrows, c.pad_row, c.pad_dst, c.nch_dst, c.nch_x, c.nsamp_dst, nsamp_x, c.ids
    );
    compare(t, &label, &a, &b, Some(0));
}

/// `dequantize_row_iq3_xxs_cuda`: grid ceil(k / 256), block 32, params (vx, y). Both sides get
/// the same input bytes (including the tail past `k` that the reference reads) and output fill.
fn deq_case(m: &Mods, t: &mut Tally, r: &mut Rng, k: usize, nvals: usize, rate: u64) {
    let es = [4usize, 2, 2][k];
    let groups = nvals.div_ceil(256);
    let x = x_blocks(r, groups * 8, rate);
    let y = r.bytes(groups * 256 * es + 16);
    let mut p = Params::default();
    let slots = vec![(0usize, Some(0usize)), (8, Some(1))];
    p.put(0u64).put(0u64);
    let grid = (groups as u32, 1, 1);
    println!("CASE deq k={k} nvals={nvals} rate={rate} groups={groups} xbytes={} ybytes={}", x.len(), y.len());
    let bufs = vec![x.clone(), y];
    let a = launch(func(m.convert, DEQ_REF[k]), grid, (32, 1, 1), &p, &slots, &bufs, &[1]);
    let b = launch(func(m.oxide, DEQ_OX[k]), grid, (32, 1, 1), &p, &slots, &bufs, &[1]);
    // host arbiter: which side, if either, matches a straight transcription of the reference?
    let mut cpu = vec![0u8; bufs[1].len()];
    cpu_deq_iq3_xxs(&bufs[0], groups * 256, &mut cpu, es);
    if std::env::var("IQ3_ARB").is_ok() {
        let (rm, ru) = (ulp(&a[0], &cpu), ulp(&b[0], &cpu));
        println!("  ARB k={k} nvals={nvals} ref-vs-cpu ulp max={} oxide-vs-cpu ulp max={}", rm.0, ru.0);
    }
    compare(t, &format!("{} k={nvals}", DEQ_OX[k]), &a, &b, Some(0));
}

pub fn run_gate() -> bool {
    let dir = env!("CARGO_MANIFEST_DIR");
    let _ctx = kdiff::cuda_core::CudaContext::new(0).expect("cuda context");
    unsafe {
        let mut c: cu::CUcontext = std::ptr::null_mut();
        ck!(cu::cuCtxGetCurrent(&mut c));
        if c.is_null() {
            let mut dev = 0;
            ck!(cu::cuDeviceGet(&mut dev, 0));
            ck!(cu::cuDevicePrimaryCtxRetain(&mut c, dev));
            ck!(cu::cuCtxSetCurrent(c));
        }
    }
    let m = Mods {
        mmvq: load(&format!("{dir}/{REF_MMVQ}")),
        convert: load(&format!("{dir}/{REF_CONVERT}")),
        oxide: load(&format!("{dir}/{OXIDE}")),
    };
    let mut t = Tally::default();
    let mut r = Rng(0x0DD5_EED5_1234_5678);

    // dequantize: whole 256-value groups and a ragged tail group, clean and specials-heavy scales.
    for k in 0..3 {
        for &n in &[32usize, 256, 640, 4096, 12288, 32 * 1001] {
            deq_case(&m, &mut t, &mut r, k, n, 16);
            deq_case(&m, &mut t, &mut r, k, n, 3);
        }
    }

    // mmvq
    let mut cases = vec![];
    for nc in 1..=8usize {
        for &small_k in if nc == 1 { &[false, true][..] } else { &[false][..] } {
            for &bpr in &[1u32, 3, 7, 31, 32, 33, 63, 64, 65, 127, 128, 129, 385] {
                for &nrows in &[1u32, 2, 5, 37] {
                    let pad = (bpr + nrows) % 3;
                    cases.push(MmvqCase { nc, small_k, bpr, nrows, pad_row: pad, pad_dst: 0, nch_dst: 1, nch_x: 1, nsamp_dst: 1, nsamp_x: 1, ids: false });
                }
            }
            for (i, &(nch_dst, nch_x, nsamp_dst, nsamp_x)) in [(4u32, 2u32, 2u32, 1u32), (3, 3, 2, 2), (6, 3, 1, 1)].iter().enumerate() {
                cases.push(MmvqCase { nc, small_k, bpr: 9 + i as u32, nrows: 13, pad_row: i as u32, pad_dst: 2 * (i as u32 % 2), nch_dst, nch_x, nsamp_dst, nsamp_x, ids: false });
                if nc == 1 {
                    cases.push(MmvqCase { nc, small_k, bpr: 5 + i as u32, nrows: 11, pad_row: 1, pad_dst: 0, nch_dst, nch_x, nsamp_dst, nsamp_x, ids: true });
                }
            }
            // model shapes (Qwen3-14B dense: 5120 -> 1024 / 5120 / 17408, 17408 -> 5120; qwen3 30B experts 2048 -> 768)
            for &(bpr, nrows) in &[(160u32, 1024u32), (160, 5120), (544, 5120), (64, 768)] {
                cases.push(MmvqCase { nc, small_k, bpr, nrows, pad_row: 0, pad_dst: 0, nch_dst: 1, nch_x: 1, nsamp_dst: 1, nsamp_x: 1, ids: false });
            }
        }
    }
    for c in &cases {
        mmvq_case(&m, &mut t, &mut r, c, 512);
    }
    // specials-heavy pass (NaN / inf / zero / denormal scales)
    for c in cases.iter().filter(|c| c.nrows <= 37) {
        mmvq_case(&m, &mut t, &mut r, c, 3);
    }
    t.finish("iq3_xxs (dequantize x3, mmvq x9) vs llama.cpp acecd56 cubins")
}

/// `--bench`: reference vs oxide mmvq (nc = 1) on Qwen3-14B decode shapes, 200 launches each.
fn bench() {
    let dir = env!("CARGO_MANIFEST_DIR");
    let _ctx = kdiff::cuda_core::CudaContext::new(0).expect("cuda context");
    let m = Mods { mmvq: load(&format!("{dir}/{REF_MMVQ}")), convert: load(&format!("{dir}/{REF_CONVERT}")), oxide: load(&format!("{dir}/{OXIDE}")) };
    let _ = m.convert;
    let mut r = Rng(7);
    for &(bpr, nrows) in &[(160u32, 17408u32), (544, 5120), (160, 5120), (160, 151936)] {
        let x = x_blocks(&mut r, (bpr * (nrows + 1)) as usize, 512);
        let y = q8_1_blocks(&mut r, (bpr + 8) as usize, 512);
        unsafe {
            let mut dx: cu::CUdeviceptr = 0;
            let mut dy: cu::CUdeviceptr = 0;
            let mut dd: cu::CUdeviceptr = 0;
            ck!(cu::cuMemAlloc_v2(&mut dx, x.len()));
            ck!(cu::cuMemAlloc_v2(&mut dy, y.len()));
            ck!(cu::cuMemAlloc_v2(&mut dd, nrows as usize * 4));
            ck!(cu::cuMemcpyHtoD_v2(dx, x.as_ptr() as *const c_void, x.len()));
            ck!(cu::cuMemcpyHtoD_v2(dy, y.as_ptr() as *const c_void, y.len()));
            let one = fastdiv_values(1);
            let mut p = Params::default();
            p.put(dx).put(dy).put(0u64);
            for _ in 0..5 { p.put(0u64); }
            p.put(0u32).put(0f32).put(dd).put(bpr * QK).put(0u32).put(0u32).put(0u32);
            p.put(bpr).put(bpr).put(nrows).put(one[0]).put(one[1]).put(one[2]).put(0u32).put(0u32).put(0u32);
            p.put(one[0]).put(one[1]).put(one[2]).put(0u32).put(0u32).put(0u32).put(0u32);
            let mut ms = [0f32; 2];
            for (k, f) in [func(m.mmvq, &mmvq_ref_name(1, false)), func(m.oxide, "iq3_xxs_mmvq_1")].into_iter().enumerate() {
                let mut pb = p.0.clone();
                let mut size = pb.len();
                let mut extra: [*mut c_void; 5] = [1 as *mut c_void, pb.as_mut_ptr() as *mut c_void, 2 as *mut c_void, &mut size as *mut usize as *mut c_void, std::ptr::null_mut()];
                let (mut e0, mut e1): (cu::CUevent, cu::CUevent) = (std::ptr::null_mut(), std::ptr::null_mut());
                ck!(cu::cuEventCreate(&mut e0, 0));
                ck!(cu::cuEventCreate(&mut e1, 0));
                for it in 0..220 {
                    if it == 20 { ck!(cu::cuEventRecord(e0, std::ptr::null_mut())); }
                    ck!(cu::cuLaunchKernel(f, nrows, 1, 1, 32, 4, 1, 0, std::ptr::null_mut(), std::ptr::null_mut(), extra.as_mut_ptr()));
                }
                ck!(cu::cuEventRecord(e1, std::ptr::null_mut()));
                ck!(cu::cuEventSynchronize(e1));
                ck!(cu::cuEventElapsedTime_v2(&mut ms[k], e0, e1));
            }
            let gb = (x.len() as f64 * nrows as f64 / (nrows + 1) as f64) / 1e9;
            println!("mmvq nc=1 K={} N={nrows}: ref {:.2} us ({:.0} GB/s), oxide {:.2} us ({:.0} GB/s)", bpr * QK,
                ms[0] * 5.0, gb / (ms[0] as f64 / 200e3), ms[1] * 5.0, gb / (ms[1] as f64 / 200e3));
            cu::cuMemFree_v2(dx); cu::cuMemFree_v2(dy); cu::cuMemFree_v2(dd);
        }
    }
}

pub fn run() -> bool {
    if std::env::args().any(|a| a == "--bench") {
        bench();
        return true;
    }
    run_gate()
}
