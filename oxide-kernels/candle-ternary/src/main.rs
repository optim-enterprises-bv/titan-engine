//! candle-kernels `ternary.cu` in cuda-oxide: same 26 `where_<idx>_<dtype>` entry names, same
//! raw-pointer ABI, bit-identical output (checked by the host `main` against candle's nvcc PTX).
//!
//! Semantics copied from the reference:
//! - fast path only when ids, t AND f layouts are all `is_contiguous` (a size-1 dim may carry any stride);
//! - otherwise three independent `get_strided_index` calls (u32 accumulator, u64 products, truncated);
//! - the condition is `ids[..] != 0` over the full width of the index type (an i64 with only high
//!   bits set is true); the selected value is copied as raw bits (NaN payloads, -0 preserved).
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// A condition type: C truthiness (`x != 0`).
    pub trait Cond: Copy {
        fn truthy(self) -> bool;
    }
    impl Cond for i64 { #[inline(always)] fn truthy(self) -> bool { self != 0 } }
    impl Cond for u32 { #[inline(always)] fn truthy(self) -> bool { self != 0 } }
    impl Cond for u8 { #[inline(always)] fn truthy(self) -> bool { self != 0 } }
    impl Cond for i16 { #[inline(always)] fn truthy(self) -> bool { self != 0 } }
    impl Cond for i32 { #[inline(always)] fn truthy(self) -> bool { self != 0 } }

    /// candle `is_contiguous` (cuda_utils.cuh).
    #[inline(always)]
    pub unsafe fn is_contiguous(num_dims: usize, dims: *const usize, strides: *const usize) -> bool {
        let mut acc: usize = 1;
        let mut d: u32 = 0;
        while (d as usize) < num_dims {
            let dim_idx = (num_dims - 1 - d as usize) as u32 as usize;
            let dim = *dims.add(dim_idx);
            if dim > 1 && acc != *strides.add(dim_idx) {
                return false;
            }
            acc = acc.wrapping_mul(dim);
            d += 1;
        }
        true
    }

    /// candle `get_strided_index`: u32 accumulator, u64 intermediate products.
    #[inline(always)]
    pub unsafe fn get_strided_index(idx: u32, num_dims: usize, dims: *const usize, strides: *const usize) -> u32 {
        let mut idx = idx;
        let mut strided_i: u32 = 0;
        let mut d: u32 = 0;
        while (d as usize) < num_dims {
            let dim_idx = (num_dims - 1 - d as usize) as u32 as usize;
            let dim = *dims.add(dim_idx);
            let term = (idx as usize % dim).wrapping_mul(*strides.add(dim_idx));
            strided_i = (strided_i as usize).wrapping_add(term) as u32;
            idx = (idx as usize / dim) as u32;
            d += 1;
        }
        strided_i
    }

    #[inline(always)]
    pub unsafe fn where_op<I: Cond, T: Copy>(
        numel: usize, num_dims: usize, info: *const usize, ids: *const I, t: *const T, f: *const T, out: *mut T,
    ) {
        let dims = info;
        let strides = info.wrapping_add(num_dims);
        let strides_t = info.wrapping_add(2 * num_dims);
        let strides_f = info.wrapping_add(3 * num_dims);
        let step = thread::blockDim_x().wrapping_mul(thread::gridDim_x());
        let mut i: u32 = thread::blockIdx_x().wrapping_mul(thread::blockDim_x()).wrapping_add(thread::threadIdx_x());
        if is_contiguous(num_dims, dims, strides)
            && is_contiguous(num_dims, dims, strides_f)
            && is_contiguous(num_dims, dims, strides_t)
        {
            while (i as usize) < numel {
                let k = i as usize;
                *out.add(k) = if (*ids.add(k)).truthy() { *t.add(k) } else { *f.add(k) };
                i = i.wrapping_add(step);
            }
        } else {
            while (i as usize) < numel {
                let s = get_strided_index(i, num_dims, dims, strides) as usize;
                let st = get_strided_index(i, num_dims, dims, strides_t) as usize;
                let sf = get_strided_index(i, num_dims, dims, strides_f) as usize;
                *out.add(i as usize) = if (*ids.add(s)).truthy() { *t.add(st) } else { *f.add(sf) };
                i = i.wrapping_add(step);
            }
        }
    }

    // GENERATED KERNELS BEGIN
    #[kernel]
    pub unsafe fn where_i64_bf16(numel: usize, num_dims: usize, info: *const usize, ids: *const i64, t: *const u16, f: *const u16, out: *mut u16) {
        unsafe { where_op(numel, num_dims, info, ids, t, f, out) }
    }

    #[kernel]
    pub unsafe fn where_u32_bf16(numel: usize, num_dims: usize, info: *const usize, ids: *const u32, t: *const u16, f: *const u16, out: *mut u16) {
        unsafe { where_op(numel, num_dims, info, ids, t, f, out) }
    }

    #[kernel]
    pub unsafe fn where_u8_bf16(numel: usize, num_dims: usize, info: *const usize, ids: *const u8, t: *const u16, f: *const u16, out: *mut u16) {
        unsafe { where_op(numel, num_dims, info, ids, t, f, out) }
    }

    #[kernel]
    pub unsafe fn where_i16_fp8_e4m3(numel: usize, num_dims: usize, info: *const usize, ids: *const i16, t: *const u8, f: *const u8, out: *mut u8) {
        unsafe { where_op(numel, num_dims, info, ids, t, f, out) }
    }

    #[kernel]
    pub unsafe fn where_i32_fp8_e4m3(numel: usize, num_dims: usize, info: *const usize, ids: *const i32, t: *const u8, f: *const u8, out: *mut u8) {
        unsafe { where_op(numel, num_dims, info, ids, t, f, out) }
    }

    #[kernel]
    pub unsafe fn where_i64_fp8_e4m3(numel: usize, num_dims: usize, info: *const usize, ids: *const i64, t: *const u8, f: *const u8, out: *mut u8) {
        unsafe { where_op(numel, num_dims, info, ids, t, f, out) }
    }

    #[kernel]
    pub unsafe fn where_u32_fp8_e4m3(numel: usize, num_dims: usize, info: *const usize, ids: *const u32, t: *const u8, f: *const u8, out: *mut u8) {
        unsafe { where_op(numel, num_dims, info, ids, t, f, out) }
    }

    #[kernel]
    pub unsafe fn where_u8_fp8_e4m3(numel: usize, num_dims: usize, info: *const usize, ids: *const u8, t: *const u8, f: *const u8, out: *mut u8) {
        unsafe { where_op(numel, num_dims, info, ids, t, f, out) }
    }

    #[kernel]
    pub unsafe fn where_i64_f16(numel: usize, num_dims: usize, info: *const usize, ids: *const i64, t: *const u16, f: *const u16, out: *mut u16) {
        unsafe { where_op(numel, num_dims, info, ids, t, f, out) }
    }

    #[kernel]
    pub unsafe fn where_u32_f16(numel: usize, num_dims: usize, info: *const usize, ids: *const u32, t: *const u16, f: *const u16, out: *mut u16) {
        unsafe { where_op(numel, num_dims, info, ids, t, f, out) }
    }

    #[kernel]
    pub unsafe fn where_u8_f16(numel: usize, num_dims: usize, info: *const usize, ids: *const u8, t: *const u16, f: *const u16, out: *mut u16) {
        unsafe { where_op(numel, num_dims, info, ids, t, f, out) }
    }

    #[kernel]
    pub unsafe fn where_i64_f32(numel: usize, num_dims: usize, info: *const usize, ids: *const i64, t: *const f32, f: *const f32, out: *mut f32) {
        unsafe { where_op(numel, num_dims, info, ids, t, f, out) }
    }

    #[kernel]
    pub unsafe fn where_i64_f64(numel: usize, num_dims: usize, info: *const usize, ids: *const i64, t: *const f64, f: *const f64, out: *mut f64) {
        unsafe { where_op(numel, num_dims, info, ids, t, f, out) }
    }

    #[kernel]
    pub unsafe fn where_i64_u8(numel: usize, num_dims: usize, info: *const usize, ids: *const i64, t: *const u8, f: *const u8, out: *mut u8) {
        unsafe { where_op(numel, num_dims, info, ids, t, f, out) }
    }

    #[kernel]
    pub unsafe fn where_i64_u32(numel: usize, num_dims: usize, info: *const usize, ids: *const i64, t: *const u32, f: *const u32, out: *mut u32) {
        unsafe { where_op(numel, num_dims, info, ids, t, f, out) }
    }

    #[kernel]
    pub unsafe fn where_i64_i64(numel: usize, num_dims: usize, info: *const usize, ids: *const i64, t: *const i64, f: *const i64, out: *mut i64) {
        unsafe { where_op(numel, num_dims, info, ids, t, f, out) }
    }

    #[kernel]
    pub unsafe fn where_u32_f32(numel: usize, num_dims: usize, info: *const usize, ids: *const u32, t: *const f32, f: *const f32, out: *mut f32) {
        unsafe { where_op(numel, num_dims, info, ids, t, f, out) }
    }

    #[kernel]
    pub unsafe fn where_u32_f64(numel: usize, num_dims: usize, info: *const usize, ids: *const u32, t: *const f64, f: *const f64, out: *mut f64) {
        unsafe { where_op(numel, num_dims, info, ids, t, f, out) }
    }

    #[kernel]
    pub unsafe fn where_u32_u8(numel: usize, num_dims: usize, info: *const usize, ids: *const u32, t: *const u8, f: *const u8, out: *mut u8) {
        unsafe { where_op(numel, num_dims, info, ids, t, f, out) }
    }

    #[kernel]
    pub unsafe fn where_u32_u32(numel: usize, num_dims: usize, info: *const usize, ids: *const u32, t: *const u32, f: *const u32, out: *mut u32) {
        unsafe { where_op(numel, num_dims, info, ids, t, f, out) }
    }

    #[kernel]
    pub unsafe fn where_u32_i64(numel: usize, num_dims: usize, info: *const usize, ids: *const u32, t: *const i64, f: *const i64, out: *mut i64) {
        unsafe { where_op(numel, num_dims, info, ids, t, f, out) }
    }

    #[kernel]
    pub unsafe fn where_u8_f32(numel: usize, num_dims: usize, info: *const usize, ids: *const u8, t: *const f32, f: *const f32, out: *mut f32) {
        unsafe { where_op(numel, num_dims, info, ids, t, f, out) }
    }

    #[kernel]
    pub unsafe fn where_u8_f64(numel: usize, num_dims: usize, info: *const usize, ids: *const u8, t: *const f64, f: *const f64, out: *mut f64) {
        unsafe { where_op(numel, num_dims, info, ids, t, f, out) }
    }

    #[kernel]
    pub unsafe fn where_u8_u8(numel: usize, num_dims: usize, info: *const usize, ids: *const u8, t: *const u8, f: *const u8, out: *mut u8) {
        unsafe { where_op(numel, num_dims, info, ids, t, f, out) }
    }

    #[kernel]
    pub unsafe fn where_u8_u32(numel: usize, num_dims: usize, info: *const usize, ids: *const u8, t: *const u32, f: *const u32, out: *mut u32) {
        unsafe { where_op(numel, num_dims, info, ids, t, f, out) }
    }

    #[kernel]
    pub unsafe fn where_u8_i64(numel: usize, num_dims: usize, info: *const usize, ids: *const u8, t: *const i64, f: *const i64, out: *mut i64) {
        unsafe { where_op(numel, num_dims, info, ids, t, f, out) }
    }
    // GENERATED KERNELS END
}

// ---------------------------------------------------------------------------------------------
// Differential gate against candle's nvcc-built ternary.ptx.
use kdiff::{Arg, Harness, Rng, Tally, as_bytes, contiguous};

fn vsize(vt: &str) -> usize {
    match vt { "bf16" | "f16" => 2, "fp8_e4m3" | "u8" => 1, "f32" | "u32" => 4, "f64" | "i64" => 8, _ => panic!("{vt}") }
}
fn isize_of(it: &str) -> usize {
    match it { "u8" => 1, "i16" => 2, "u32" | "i32" => 4, "i64" => 8, _ => panic!("{it}") }
}
/// Conditions: zeros, ones, random, and values that are non-zero only in the top byte of the type.
fn gen_cond(rng: &mut Rng, it: &str, n: usize) -> Vec<u8> {
    let w = isize_of(it);
    let mut b = Vec::with_capacity(n * w);
    for _ in 0..n {
        let v: u64 = match rng.next() % 5 {
            0 | 1 => 0,
            2 => 1,
            3 => 1u64 << (8 * w - 1 - (rng.next() % 8) as usize), // high bits only
            _ => rng.next(),
        };
        b.extend_from_slice(&v.to_le_bytes()[..w]);
    }
    b
}
/// Smallest buffer length (elements) that covers every strided index, u32-truncated as the kernel does.
fn span(dims: &[usize], strides: &[usize]) -> usize {
    let numel: usize = dims.iter().product();
    let mut m = 0usize;
    for i in 0..numel {
        let (mut idx, mut s) = (i as u32, 0u32);
        for d in (0..dims.len()).rev() {
            s = (s as u64).wrapping_add((idx as u64 % dims[d] as u64).wrapping_mul(strides[d] as u64)) as u32;
            idx = (idx as u64 / dims[d] as u64) as u32;
        }
        m = m.max(s as usize);
    }
    m + 1
}

fn main() {
    let root = format!("{}/titan-engine/oxide-kernels", std::env::var("HOME").unwrap());
    let ref_ptx = format!("{root}/reference/candle/ternary.ptx");
    let h = Harness::new(&ref_ptx, &format!("{root}/candle-ternary/candle_ternary.ptx"));
    let names: Vec<String> = std::fs::read_to_string(&ref_ptx)
        .unwrap()
        .lines()
        .filter_map(|l| l.strip_prefix(".visible .entry ").map(|s| s.trim_end_matches('(').to_string()))
        .collect();
    assert_eq!(names.len(), 26, "reference entry count");
    let mut t = Tally::default();
    let mut rng = Rng(0x7E27);
    let mut missing = vec![];

    // (dims, strides ids, strides t, strides f)
    type L = (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>);
    let c = |d: &[usize]| contiguous(d);
    let big = (1usize << 32) + 3; // u32 truncation: behaves like stride 3
    let layouts: Vec<L> = vec![
        (vec![1000], vec![1], vec![1], vec![1]),
        (vec![7, 13, 5], c(&[7, 13, 5]), c(&[7, 13, 5]), c(&[7, 13, 5])),
        (vec![4, 1, 9], vec![9, 1234, 1], vec![9, 7, 1], vec![9, 0, 1]),         // size-1 dims: still contiguous
        (vec![13, 7], vec![1, 13], c(&[13, 7]), c(&[13, 7])),                     // only ids transposed
        (vec![13, 7], c(&[13, 7]), vec![1, 13], c(&[13, 7])),                     // only t transposed
        (vec![13, 7], c(&[13, 7]), c(&[13, 7]), vec![0, 1]),                      // only f broadcast
        (vec![6, 5], vec![0, 1], vec![1, 0], vec![0, 0]),                         // all broadcast
        (vec![3, 4, 5], vec![40, 1, 8], vec![1, 3, 12], vec![20, 5, 1]),          // permuted, independent
        (vec![2, 3], vec![big, 1], vec![3, big - 2], vec![big, big - 2]),         // u32 truncation of the index
        (vec![], vec![], vec![], vec![]),                                         // scalar (num_dims 0)
        (vec![0, 5], vec![5, 1], vec![1, 0], vec![5, 1]),                         // numel 0
        (vec![1], vec![1], vec![1], vec![1]),
        (vec![2, 1, 3, 1, 4], vec![12, 99, 4, 5, 1], vec![1, 0, 2, 0, 6], c(&[2, 1, 3, 1, 4])),
    ];
    for name in &names {
        if !h.has(true, name) {
            missing.push(name.clone());
            continue;
        }
        let rest = name.strip_prefix("where_").unwrap();
        let (it, vt) = rest.split_once('_').unwrap();
        let vs = vsize(vt);
        for (li, (dims, si, st, sf)) in layouts.iter().enumerate() {
            let numel: usize = dims.iter().product();
            let (ni, nt, nf) = (span(dims, si), span(dims, st), span(dims, sf));
            let grids = [(4u32, 64u32), (((numel as u32) + 255) / 256, 256), (1, 32), (1, 1), (3, 96)];
            for &(grid, block) in &grids {
                let grid = grid.max(1);
                let mut info: Vec<u64> = vec![];
                for v in [dims, si, st, sf] {
                    info.extend(v.iter().map(|&x| x as u64));
                }
                if info.is_empty() {
                    info.push(0);
                }
                let bufs = vec![
                    gen_cond(&mut rng, it, ni),
                    rng.bytes(nt * vs),
                    rng.bytes(nf * vs),
                    rng.bytes(numel.max(1) * vs),
                    as_bytes(&info),
                ];
                let args = [Arg::U64(numel as u64), Arg::U64(dims.len() as u64), Arg::Buf(4), Arg::Buf(0), Arg::Buf(1), Arg::Buf(2), Arg::Buf(3)];
                let d = h.diff(name, (grid, 1, 1), (block, 1, 1), 0, &args, &bufs, &[3]);
                t.record(&format!("{name} layout{li} g{grid}x{block}"), &d);
            }
        }
    }
    for m in &missing {
        println!("  MISSING from oxide PTX: {m}");
    }
    let ok = t.finish("ternary") && missing.is_empty();
    println!("ternary: {} of {} reference entries ported", names.len() - missing.len(), names.len());
    std::process::exit(if ok { 0 } else { 1 });
}
