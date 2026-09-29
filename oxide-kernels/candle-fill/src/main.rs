//! candle-kernels `fill.cu` in cuda-oxide: same 24 entry names (fill_*, copy2d_*, const_set_*),
//! same by-value/raw-pointer ABI, bit-identical output (checked by the host `main` against
//! candle's own nvcc PTX).
//!
//! Semantics copied from the reference:
//! - fill: grid-stride loop with a `unsigned int` counter compared against the u64 numel;
//! - copy2d: ONE element per thread (no grid-stride loop); all arithmetic is u32 and wraps,
//!   including the `d1 * d2` bound and the `idx1 * s + idx2` offsets (zero-extended afterwards);
//! - const_set: `info == nullptr || is_contiguous` -> dense writes, else scattered writes to
//!   `get_strided_index(i)` (u32 accumulator, u64 products, truncated);
//! - values are stored as raw bits (f16/bf16 are u16 patterns, fp8 a u8 pattern).
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

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
    fn start() -> u32 {
        thread::blockIdx_x().wrapping_mul(thread::blockDim_x()).wrapping_add(thread::threadIdx_x())
    }
    #[inline(always)]
    fn step() -> u32 {
        thread::blockDim_x().wrapping_mul(thread::gridDim_x())
    }

    #[inline(always)]
    pub unsafe fn fill_with<T: Copy>(buf: *mut T, value: T, numel: usize) {
        let step = step();
        let mut i = start();
        while (i as usize) < numel {
            *buf.add(i as usize) = value;
            i = i.wrapping_add(step);
        }
    }

    #[inline(always)]
    pub unsafe fn copy2d<T: Copy>(src: *const T, dst: *mut T, d1: u32, d2: u32, src_s: u32, dst_s: u32) {
        let idx = start();
        if idx >= d1.wrapping_mul(d2) {
            return;
        }
        let idx1 = idx / d2;
        let idx2 = idx.wrapping_sub(d2.wrapping_mul(idx1));
        let di = idx1.wrapping_mul(dst_s).wrapping_add(idx2);
        let si = idx1.wrapping_mul(src_s).wrapping_add(idx2);
        *dst.add(di as usize) = *src.add(si as usize);
    }

    #[inline(always)]
    pub unsafe fn const_set<T: Copy>(numel: usize, num_dims: usize, info: *const usize, inp: T, out: *mut T) {
        let dims = info;
        let strides = info.wrapping_add(num_dims);
        let step = step();
        let mut i = start();
        if info.is_null() || is_contiguous(num_dims, dims, strides) {
            while (i as usize) < numel {
                *out.add(i as usize) = inp;
                i = i.wrapping_add(step);
            }
        } else {
            while (i as usize) < numel {
                let s = get_strided_index(i, num_dims, dims, strides);
                *out.add(s as usize) = inp;
                i = i.wrapping_add(step);
            }
        }
    }

    #[kernel] pub unsafe fn fill_u8(buf: *mut u8, value: u8, numel: usize) { unsafe { fill_with(buf, value, numel) } }
    #[kernel] pub unsafe fn fill_u32(buf: *mut u32, value: u32, numel: usize) { unsafe { fill_with(buf, value, numel) } }
    #[kernel] pub unsafe fn fill_i64(buf: *mut i64, value: i64, numel: usize) { unsafe { fill_with(buf, value, numel) } }
    #[kernel] pub unsafe fn fill_f32(buf: *mut f32, value: f32, numel: usize) { unsafe { fill_with(buf, value, numel) } }
    #[kernel] pub unsafe fn fill_f64(buf: *mut f64, value: f64, numel: usize) { unsafe { fill_with(buf, value, numel) } }
    #[kernel] pub unsafe fn fill_f16(buf: *mut u16, value: u16, numel: usize) { unsafe { fill_with(buf, value, numel) } }
    #[kernel] pub unsafe fn fill_bf16(buf: *mut u16, value: u16, numel: usize) { unsafe { fill_with(buf, value, numel) } }
    #[kernel] pub unsafe fn fill_f8_e4m3(buf: *mut u8, value: u8, numel: usize) { unsafe { fill_with(buf, value, numel) } }

    #[kernel] pub unsafe fn copy2d_f32(src: *const f32, dst: *mut f32, d1: u32, d2: u32, src_s: u32, dst_s: u32) { unsafe { copy2d(src, dst, d1, d2, src_s, dst_s) } }
    #[kernel] pub unsafe fn copy2d_f64(src: *const f64, dst: *mut f64, d1: u32, d2: u32, src_s: u32, dst_s: u32) { unsafe { copy2d(src, dst, d1, d2, src_s, dst_s) } }
    #[kernel] pub unsafe fn copy2d_u8(src: *const u8, dst: *mut u8, d1: u32, d2: u32, src_s: u32, dst_s: u32) { unsafe { copy2d(src, dst, d1, d2, src_s, dst_s) } }
    #[kernel] pub unsafe fn copy2d_u32(src: *const u32, dst: *mut u32, d1: u32, d2: u32, src_s: u32, dst_s: u32) { unsafe { copy2d(src, dst, d1, d2, src_s, dst_s) } }
    #[kernel] pub unsafe fn copy2d_i64(src: *const i64, dst: *mut i64, d1: u32, d2: u32, src_s: u32, dst_s: u32) { unsafe { copy2d(src, dst, d1, d2, src_s, dst_s) } }
    #[kernel] pub unsafe fn copy2d_f16(src: *const u16, dst: *mut u16, d1: u32, d2: u32, src_s: u32, dst_s: u32) { unsafe { copy2d(src, dst, d1, d2, src_s, dst_s) } }
    #[kernel] pub unsafe fn copy2d_bf16(src: *const u16, dst: *mut u16, d1: u32, d2: u32, src_s: u32, dst_s: u32) { unsafe { copy2d(src, dst, d1, d2, src_s, dst_s) } }
    #[kernel] pub unsafe fn copy2d_f8_e4m3(src: *const u8, dst: *mut u8, d1: u32, d2: u32, src_s: u32, dst_s: u32) { unsafe { copy2d(src, dst, d1, d2, src_s, dst_s) } }

    #[kernel] pub unsafe fn const_set_f32(numel: usize, num_dims: usize, info: *const usize, inp: f32, out: *mut f32) { unsafe { const_set(numel, num_dims, info, inp, out) } }
    #[kernel] pub unsafe fn const_set_f64(numel: usize, num_dims: usize, info: *const usize, inp: f64, out: *mut f64) { unsafe { const_set(numel, num_dims, info, inp, out) } }
    #[kernel] pub unsafe fn const_set_u8(numel: usize, num_dims: usize, info: *const usize, inp: u8, out: *mut u8) { unsafe { const_set(numel, num_dims, info, inp, out) } }
    #[kernel] pub unsafe fn const_set_u32(numel: usize, num_dims: usize, info: *const usize, inp: u32, out: *mut u32) { unsafe { const_set(numel, num_dims, info, inp, out) } }
    #[kernel] pub unsafe fn const_set_i64(numel: usize, num_dims: usize, info: *const usize, inp: i64, out: *mut i64) { unsafe { const_set(numel, num_dims, info, inp, out) } }
    #[kernel] pub unsafe fn const_set_f16(numel: usize, num_dims: usize, info: *const usize, inp: u16, out: *mut u16) { unsafe { const_set(numel, num_dims, info, inp, out) } }
    #[kernel] pub unsafe fn const_set_bf16(numel: usize, num_dims: usize, info: *const usize, inp: u16, out: *mut u16) { unsafe { const_set(numel, num_dims, info, inp, out) } }
    #[kernel] pub unsafe fn const_set_f8_e4m3(numel: usize, num_dims: usize, info: *const usize, inp: u8, out: *mut u8) { unsafe { const_set(numel, num_dims, info, inp, out) } }
}

// ---------------------------------------------------------------------------------------------
// Differential gate against candle's nvcc-built fill.ptx.
use kdiff::{Arg, Harness, Rng, Tally, contiguous, layout_info};

fn vsize(vt: &str) -> usize {
    match vt { "bf16" | "f16" => 2, "f8_e4m3" | "u8" => 1, "f32" | "u32" => 4, "f64" | "i64" => 8, _ => panic!("{vt}") }
}
/// A by-value scalar of type `vt`, drawn from raw bit patterns (NaN payloads, -0, denormals).
fn scalar(rng: &mut Rng, vt: &str) -> Arg {
    let r = rng.next();
    match vt {
        "f32" => Arg::F32(if r % 3 == 0 { rng.f32s(1)[0] } else { f32::from_bits(r as u32) }),
        "f64" => Arg::F64(if r % 3 == 0 { rng.f64s(1)[0] } else { f64::from_bits(r) }),
        "bf16" | "f16" => Arg::B16(r as u16),
        "f8_e4m3" | "u8" => Arg::B8(r as u8),
        "u32" => Arg::U32(r as u32),
        "i64" => Arg::I64(r as i64),
        _ => panic!("{vt}"),
    }
}
/// Largest u32-truncated strided index + 1.
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
    let ref_ptx = format!("{root}/reference/candle/fill.ptx");
    let h = Harness::new(&ref_ptx, &format!("{root}/candle-fill/candle_fill.ptx"));
    let names: Vec<String> = std::fs::read_to_string(&ref_ptx)
        .unwrap()
        .lines()
        .filter_map(|l| l.strip_prefix(".visible .entry ").map(|s| s.trim_end_matches('(').to_string()))
        .collect();
    assert_eq!(names.len(), 24, "reference entry count");
    let mut t = Tally::default();
    let mut rng = Rng(0xF111);
    let mut missing = vec![];
    let grids_for = |work: usize| [(4u32, 64u32), ((((work as u32) + 255) / 256).max(1), 256), (1, 32), (1, 1), (3, 96)];

    for name in &names {
        if !h.has(true, name) {
            missing.push(name.clone());
            continue;
        }
        let (op, vt) = if let Some(v) = name.strip_prefix("fill_") {
            ("fill", v)
        } else if let Some(v) = name.strip_prefix("copy2d_") {
            ("copy2d", v)
        } else {
            ("const_set", name.strip_prefix("const_set_").unwrap())
        };
        let vs = vsize(vt);
        match op {
            "fill" => {
                for numel in [0usize, 1, 2, 31, 1000, 4099] {
                    for (grid, block) in grids_for(numel) {
                        // buffer longer than numel: the tail must stay untouched
                        let bufs = vec![rng.bytes((numel + 7) * vs)];
                        let args = [Arg::Buf(0), scalar(&mut rng, vt), Arg::U64(numel as u64)];
                        let d = h.diff(name, (grid, 1, 1), (block, 1, 1), 0, &args, &bufs, &[0]);
                        t.record(&format!("{name} numel{numel} g{grid}x{block}"), &d);
                    }
                }
            }
            "copy2d" => {
                // (d1, d2, src_s, dst_s)
                let cases: [(u32, u32, u32, u32); 10] = [
                    (0, 5, 5, 5), (5, 0, 5, 5), (1, 1, 1, 1), (1, 17, 17, 17), (13, 7, 7, 7),
                    (13, 7, 11, 9), (7, 13, 13, 20), (9, 5, 0, 5), (65537, 65536, 1, 2), // d1*d2 wraps to 65536: threads past it must not write
                    (3, 700, 1000, 701),
                ];
                for &(d1, d2, ss, ds) in &cases {
                    let work = d1.wrapping_mul(d2) as usize;
                    let rows = if work == 0 { 0 } else { (work - 1) / d2 as usize + 1 };
                    let src_len = if rows == 0 { 1 } else { (rows - 1) * ss as usize + d2 as usize };
                    let dst_len = if rows == 0 { 1 } else { (rows - 1) * ds as usize + d2 as usize + 3 };
                    // copy2d has no grid-stride loop: always launch enough threads (plus extra to test the bound).
                    for (grid, block) in [(((work as u32) + 255) / 256 + 1, 256u32), (((work as u32) + 63) / 64 + 2, 64), ((work as u32) + 1, 1)] {
                        if grid as u64 * block as u64 > 1 << 20 && block == 1 { continue; }
                        let bufs = vec![rng.bytes(src_len * vs), rng.bytes(dst_len * vs)];
                        let args = [Arg::Buf(0), Arg::Buf(1), Arg::U32(d1), Arg::U32(d2), Arg::U32(ss), Arg::U32(ds)];
                        let d = h.diff(name, (grid, 1, 1), (block, 1, 1), 0, &args, &bufs, &[1]);
                        t.record(&format!("{name} ({d1},{d2},{ss},{ds}) g{grid}x{block}"), &d);
                    }
                }
            }
            _ => {
                let big = (1usize << 32) + 3; // u32 truncation: behaves like stride 3
                let layouts: Vec<(Vec<usize>, Vec<usize>)> = vec![
                    (vec![1000], vec![1]),
                    (vec![7, 13, 5], contiguous(&[7, 13, 5])),
                    (vec![13, 7], vec![1, 13]),         // transposed
                    (vec![4, 1, 9], vec![9, 1234, 1]),  // size-1 dim, odd stride: contiguous
                    (vec![6, 5], vec![0, 1]),           // broadcast (repeated writes of the same value)
                    (vec![3, 4, 5], vec![40, 1, 8]),    // permuted
                    (vec![5, 3], vec![11, 2]),          // gaps: holes must stay untouched
                    (vec![2, 3], vec![big, 1]),         // u32 truncation of the index
                    (vec![], vec![]),                   // scalar
                    (vec![0, 4], vec![4, 1]),           // numel 0
                    (vec![1], vec![1]),
                ];
                for (li, (dims, strides)) in layouts.iter().enumerate() {
                    let numel: usize = dims.iter().product();
                    let out_len = span(dims, strides).max(numel) + 5;
                    for (gi, (grid, block)) in grids_for(numel).into_iter().enumerate() {
                        for with_info in [true, false] {
                            if !with_info && gi > 1 { continue; }
                            let mut info = layout_info(dims, strides);
                            if info.is_empty() { info = vec![0u8; 8]; }
                            let bufs = vec![rng.bytes(out_len * vs), info];
                            let info_arg = if with_info { Arg::Buf(1) } else { Arg::Null };
                            let args = [Arg::U64(numel as u64), Arg::U64(dims.len() as u64), info_arg, scalar(&mut rng, vt), Arg::Buf(0)];
                            let d = h.diff(name, (grid, 1, 1), (block, 1, 1), 0, &args, &bufs, &[0]);
                            t.record(&format!("{name} layout{li} g{grid}x{block} info={with_info}"), &d);
                        }
                    }
                }
            }
        }
    }
    for m in &missing {
        println!("  MISSING from oxide PTX: {m}");
    }
    let ok = t.finish("fill") && missing.is_empty();
    println!("fill: {} of {} reference entries ported", names.len() - missing.len(), names.len());
    std::process::exit(if ok { 0 } else { 1 });
}
