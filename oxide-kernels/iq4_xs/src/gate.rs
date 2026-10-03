//! Differential gate: every oxide entry against the reference instance in the rebuilt llama.cpp
//! cubins (ref/mmvq.cubin, ref/convert.cubin, copied from oxide-kernels/iq4_nl/ref or rebuilt by
//! ref/nvcc_{mmvq,convert}.sh; SASS identical to libggml-cuda.so). Both sides get
//! the same packed parameter buffer (CU_LAUNCH_PARAM_BUFFER_POINTER), private copies of the same
//! input bytes and a pre-filled output buffer; every output byte is compared.
use cuda_core::sys as cu;
use kdiff::{Rng, Tally};
use std::ffi::c_void;

const REF_MMVQ: &str = "ref/mmvq.cubin";
const REF_CONVERT: &str = "ref/convert.cubin";
const OXIDE: &str = "iq4_xs.ptx";

fn mmvq_ref_name(nc: usize, small_k: bool) -> String {
    format!(
        "_Z13mul_mat_vec_qIL9ggml_type23ELi{nc}ELb0ELb{}ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj",
        small_k as u32
    )
}
fn mmvq_ox_name(nc: usize, small_k: bool) -> String {
    if small_k { "iq4_xs_mmvq_1_small_k".into() } else { format!("iq4_xs_mmvq_{nc}") }
}
const DEQ_REF: [&str; 3] = [
    "_Z23dequantize_block_iq4_xsIfEvPKvPT_",
    "_Z23dequantize_block_iq4_xsI6__halfEvPKvPT_",
    "_Z23dequantize_block_iq4_xsI13__nv_bfloat16EvPKvPT_",
];
const DEQ_OX: [&str; 3] = ["iq4_xs_dequant_f32", "iq4_xs_dequant_f16", "iq4_xs_dequant_bf16"];
/// Bytes per IQ4_XS block (half d, u16 scales_h, 4 x u8 scales_l, 128 nibble bytes) and values per
/// block (QK_K). 8 blocks per 256-value dequantize group, 8 q8_1 blocks per X block.
const BB: usize = 136;
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

fn compare(t: &mut Tally, label: &str, a: &[Vec<u8>], b: &[Vec<u8>], f32_buf: Option<usize>) -> usize {
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
    let n = d.differing;
    if let Some(i) = f32_buf {
        let (mx, mean) = ulp(&a[i], &b[i]);
        t.record(&format!("{label} [ulp max={mx} mean={mean:.2}]"), &d);
    } else {
        t.record(label, &d);
    }
    n
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
        v.extend_from_slice(&(r.next() as u16).to_le_bytes());
        for _ in 0..4 {
            v.push(r.next() as u8);
        }
        for _ in 0..128 {
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
    mmvq_case_named(m, t, r, c, rate, &mmvq_ox_name(c.nc, c.small_k));
}

fn mmvq_case_named(m: &Mods, t: &mut Tally, r: &mut Rng, c: &MmvqCase, rate: u64, ox_name: &str) -> usize {
    let nwarps: u32 = if c.nc <= 4 { 4 } else { 2 };
    let rpb: u32 = if c.nc == 1 { if c.small_k { nwarps } else { 1 } } else { 2 };
    let stride_row_x = c.bpr + c.pad_row;
    let stride_channel_x = (c.nrows + rpb) * stride_row_x;
    let stride_sample_x = c.nch_x * stride_channel_x;
    let nch_y = if c.ids { c.nch_x.max(2) } else { c.nch_dst };
    // IQ4_XS blocks hold QK (256) values, so the q8_1 side has QK/32 blocks per X block.
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
    let b = launch(func(m.oxide, ox_name), grid, block, &p, &slots, &bufs, &[2]);
    let label = format!(
        "mmvq nc={} small_k={} bpr={} nrows={} pads={}/{} ch={}/{} samp={}/{} ids={}",
        c.nc, c.small_k, c.bpr, c.nrows, c.pad_row, c.pad_dst, c.nch_dst, c.nch_x, c.nsamp_dst, nsamp_x, c.ids
    );
    compare(t, &format!("{ox_name}: {label}"), &a, &b, Some(0))
}

/// `dequantize_row_iq4_xs_cuda`: grid ceil(k / 256), block 32, params (vx, y). Both sides get
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
    let bufs = vec![x, y];
    let a = launch(func(m.convert, DEQ_REF[k]), grid, (32, 1, 1), &p, &slots, &bufs, &[1]);
    let b = launch(func(m.oxide, DEQ_OX[k]), grid, (32, 1, 1), &p, &slots, &bufs, &[1]);
    compare(t, &format!("{} k={nvals}", DEQ_OX[k]), &a, &b, Some(0));
}

unsafe extern "C" {
    fn dlopen(filename: *const std::ffi::c_char, flag: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const std::ffi::c_char) -> *mut c_void;
}
const LIBGGML_BASE: &str = "~/ai/llama.cpp/build/bin/libggml-base.so";

/// CPU oracle: llama.cpp's own `dequantize_row_iq4_xs` (libggml-base, acecd56) on random blocks with
/// finite scales, against the oxide f32 dequantize. The blocks and llama.cpp's output are written to
/// ref/cpu_blocks.bin / ref/cpu_deq.bin so candle's CPU dequantize (iquant.rs) is gated against the
/// same bytes by candle's `cuda_iq4_xs` test.
fn cpu_oracle(m: &Mods, t: &mut Tally, r: &mut Rng) {
    let nb = 4096usize;
    let x = x_blocks(r, nb, u64::MAX);
    let deq: unsafe extern "C" fn(*const u8, *mut f32, i64) = unsafe {
        let lib = std::ffi::CString::new(LIBGGML_BASE).unwrap();
        let h = dlopen(lib.as_ptr(), 2 /* RTLD_NOW */);
        assert!(!h.is_null(), "dlopen {LIBGGML_BASE}");
        let sym = std::ffi::CString::new("dequantize_row_iq4_xs").unwrap();
        let f = dlsym(h, sym.as_ptr());
        assert!(!f.is_null(), "dlsym dequantize_row_iq4_xs");
        std::mem::transmute(f)
    };
    let mut cpu = vec![0f32; nb * 256];
    unsafe { deq(x.as_ptr(), cpu.as_mut_ptr(), (nb * 256) as i64) };
    let cpu_bytes: Vec<u8> = cpu.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut p = Params::default();
    p.put(0u64).put(0u64);
    let slots = vec![(0usize, Some(0usize)), (8, Some(1))];
    let bufs = vec![x.clone(), vec![0u8; nb * 256 * 4]];
    let b = launch(func(m.oxide, DEQ_OX[0]), (nb as u32, 1, 1), (32, 1, 1), &p, &slots, &bufs, &[1]);
    compare(t, "cpu oracle: llama.cpp dequantize_row_iq4_xs vs iq4_xs_dequant_f32 (4096 blocks)", &[cpu_bytes.clone()], &b, None);
    let dir = env!("CARGO_MANIFEST_DIR");
    std::fs::write(format!("{dir}/ref/cpu_blocks.bin"), &x).unwrap();
    std::fs::write(format!("{dir}/ref/cpu_deq.bin"), &cpu_bytes).unwrap();
}

/// Mutation check: deliberately wrong variants of the port (built into the same PTX) must FAIL
/// against the reference, or the gate could not tell a right port from a plausible wrong one.
///   mut 1: q8_1 words j / j+4 paired with the wrong nibble half
///   mut 2: scales_h bits taken one bit too high
///   mut 3: group bias 31 instead of 32
///   deq mut: low / high nibbles swapped
fn mutations(m: &Mods, t: &mut Tally, r: &mut Rng) {
    let mut detected = 0;
    for k in 1..=3u32 {
        let mut tm = Tally::default();
        let mut caught = false;
        for &(bpr, nrows) in &[(4u32, 37u32), (20, 64), (68, 16)] {
            let c = MmvqCase { nc: 1, small_k: false, bpr, nrows, pad_row: 0, pad_dst: 0, nch_dst: 1, nch_x: 1, nsamp_dst: 1, nsamp_x: 1, ids: false };
            if mmvq_case_named(m, &mut tm, r, &c, 512, &format!("iq4_xs_mmvq_1_mut{k}")) > 0 { caught = true; }
        }
        println!("  mutation mmvq mut{k}: {}", if caught { "DETECTED (fails vs reference, as it must)" } else { "NOT DETECTED" });
        if caught { detected += 1; }
    }
    let mut tm = Tally::default();
    let x = x_blocks(r, 64, 512);
    let y = r.bytes(64 * 256 * 4 + 16);
    let mut p = Params::default();
    p.put(0u64).put(0u64);
    let slots = vec![(0usize, Some(0usize)), (8, Some(1))];
    let bufs = vec![x, y];
    let a = launch(func(m.convert, DEQ_REF[0]), (64, 1, 1), (32, 1, 1), &p, &slots, &bufs, &[1]);
    let b = launch(func(m.oxide, "iq4_xs_dequant_f32_mut"), (64, 1, 1), (32, 1, 1), &p, &slots, &bufs, &[1]);
    let caught = compare(&mut tm, "deq mut", &a, &b, None) > 0;
    println!("  mutation dequant: {}", if caught { "DETECTED (fails vs reference, as it must)" } else { "NOT DETECTED" });
    if caught { detected += 1; }
    let d = kdiff::Diff { bytes: 4, differing: if detected == 4 { 0 } else { 1 }, first: None };
    t.record(&format!("mutation sensitivity ({detected}/4 mutants detected)"), &d);
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
            // model shapes in IQ4_XS blocks (256 values): Qwen3.5-27B dense (OrcaSAQ-2) 5120 -> 6144 /
            // 12288 / 17408 / 1024 / 48, 17408 / 6144 / 10240 -> 5120; Qwen3.5-9B 4096 -> 12288 / 32,
            // 12288 -> 4096; qwen3 30B experts 2048 -> 768
            for &(bpr, nrows) in &[(20u32, 6144u32), (20, 12288), (20, 17408), (20, 1024), (20, 48), (68, 5120), (24, 5120), (40, 5120), (16, 12288), (16, 32), (48, 4096), (8, 768)] {
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
    cpu_oracle(&m, &mut t, &mut r);
    mutations(&m, &mut t, &mut r);
    t.finish("iq4_xs (dequantize x3, mmvq x9) vs llama.cpp acecd56 cubins")
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
            for (k, f) in [func(m.mmvq, &mmvq_ref_name(1, false)), func(m.oxide, "iq4_xs_mmvq_1")].into_iter().enumerate() {
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
