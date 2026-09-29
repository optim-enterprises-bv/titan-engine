//! candle-kernels `affine.cu` in cuda-oxide: same entry names, same by-value/raw-pointer ABI,
//! bit-identical output (checked by the host `main` against candle's own nvcc PTX).
//!
//! Semantics copied from the SASS, not just the source:
//! - index maths is `unsigned int` (u32) with `size_t` (u64) dims/strides, truncating as C does;
//! - f32/f64 `x * mul + add` is contracted by nvcc (-fmad=true) to one fma;
//! - f16/bf16 are contracted by ptxas to one HFMA2 / HFMA2.BF16, so they use a fused half fma;
//! - fp8 e4m3 goes e4m3 -> f16 -> f32, f32 fma, then satfinite e4m3 with round-to-nearest;
//! - integer types wrap (C promotion then truncation).
use cuda_device::{convert, float, kernel, thread};
use cuda_device::{bf16x2, f16x2};
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
    fn e4m3_to_f32(b: u8) -> f32 {
        convert::cvt_f32_f16x2_lo(convert::cvt_rn_f16x2_e4m3x2(b as u16))
    }

    #[inline(always)]
    fn f32_to_e4m3(v: f32) -> u8 {
        convert::cvt_rn_satfinite_e4m3x2_f32(v, 0.0) as u8
    }

    /// The shared loop of every affine kernel; `op` maps (x, mul, add) to the output.
    #[inline(always)]
    pub unsafe fn affine_loop<T: Copy, F: Fn(T, T, T) -> T>(
        numel: usize, num_dims: usize, info: *const usize, inp: *const T, out: *mut T, mul: T, add: T, op: F,
    ) {
        let dims = info;
        let strides = info.wrapping_add(num_dims);
        let step = thread::blockDim_x().wrapping_mul(thread::gridDim_x());
        let mut i: u32 = thread::blockIdx_x().wrapping_mul(thread::blockDim_x()).wrapping_add(thread::threadIdx_x());
        if info.is_null() || is_contiguous(num_dims, dims, strides) {
            while (i as usize) < numel {
                let x = if inp.is_null() { *out.add(i as usize) } else { *inp.add(i as usize) };
                *out.add(i as usize) = op(x, mul, add);
                i = i.wrapping_add(step);
            }
        } else {
            while (i as usize) < numel {
                let s = get_strided_index(i, num_dims, dims, strides);
                let x = if inp.is_null() { *out.add(i as usize) } else { *inp.add(s as usize) };
                *out.add(i as usize) = op(x, mul, add);
                i = i.wrapping_add(step);
            }
        }
    }

    #[kernel]
    pub unsafe fn affine_bf16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16, mul: u16, add: u16) {
        affine_loop(numel, num_dims, info, inp, out, mul, add, |x, m, a| bf16x2::fma_bf16x2(x as u32, m as u32, a as u32) as u16);
    }

    #[kernel]
    pub unsafe fn affine_f8_e4m3(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u8, mul: u8, add: u8) {
        affine_loop(numel, num_dims, info, inp, out, mul, add, |x, m, a| f32_to_e4m3(float::fma_rn_f32(e4m3_to_f32(x), e4m3_to_f32(m), e4m3_to_f32(a))));
    }

    #[kernel]
    pub unsafe fn affine_f16(numel: usize, num_dims: usize, info: *const usize, inp: *const u16, out: *mut u16, mul: u16, add: u16) {
        affine_loop(numel, num_dims, info, inp, out, mul, add, |x, m, a| f16x2::fma_f16x2(x as u32, m as u32, a as u32) as u16);
    }

    #[kernel]
    pub unsafe fn affine_f32(numel: usize, num_dims: usize, info: *const usize, inp: *const f32, out: *mut f32, mul: f32, add: f32) {
        affine_loop(numel, num_dims, info, inp, out, mul, add, |x, m, a| float::fma_rn_f32(x, m, a));
    }

    #[kernel]
    pub unsafe fn affine_f64(numel: usize, num_dims: usize, info: *const usize, inp: *const f64, out: *mut f64, mul: f64, add: f64) {
        affine_loop(numel, num_dims, info, inp, out, mul, add, |x, m, a| float::fma_rn_f64(x, m, a));
    }

    #[kernel]
    pub unsafe fn affine_u8(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u8, mul: u8, add: u8) {
        affine_loop(numel, num_dims, info, inp, out, mul, add, |x: u8, m, a| x.wrapping_mul(m).wrapping_add(a));
    }

    #[kernel]
    pub unsafe fn affine_u32(numel: usize, num_dims: usize, info: *const usize, inp: *const u32, out: *mut u32, mul: u32, add: u32) {
        affine_loop(numel, num_dims, info, inp, out, mul, add, |x: u32, m, a| x.wrapping_mul(m).wrapping_add(a));
    }

    #[kernel]
    pub unsafe fn affine_i16(numel: usize, num_dims: usize, info: *const usize, inp: *const i16, out: *mut i16, mul: i16, add: i16) {
        affine_loop(numel, num_dims, info, inp, out, mul, add, |x: i16, m, a| x.wrapping_mul(m).wrapping_add(a));
    }

    #[kernel]
    pub unsafe fn affine_i32(numel: usize, num_dims: usize, info: *const usize, inp: *const i32, out: *mut i32, mul: i32, add: i32) {
        affine_loop(numel, num_dims, info, inp, out, mul, add, |x: i32, m, a| x.wrapping_mul(m).wrapping_add(a));
    }

    #[kernel]
    pub unsafe fn affine_i64(numel: usize, num_dims: usize, info: *const usize, inp: *const i64, out: *mut i64, mul: i64, add: i64) {
        affine_loop(numel, num_dims, info, inp, out, mul, add, |x: i64, m, a| x.wrapping_mul(m).wrapping_add(a));
    }

}

// ---------------------------------------------------------------------------------------------
// Differential gate against candle's nvcc-built affine.ptx.
use kdiff::{Arg, Harness, Rng, Tally, as_bytes, contiguous, layout_info};

fn main() {
    let root = format!("{}/titan-engine/oxide-kernels", std::env::var("HOME").unwrap());
    let h = Harness::new(&format!("{root}/reference/candle/affine.ptx"), &format!("{root}/candle-affine/candle_affine.ptx"));
    let mut t = Tally::default();
    let mut rng = Rng(0xAFF1);
    // (name, element size, scalar generator)
    let types: [(&str, usize); 10] = [
        ("bf16", 2), ("f8_e4m3", 1), ("f16", 2), ("f32", 4), ("f64", 8), ("u8", 1), ("u32", 4), ("i16", 2), ("i32", 4), ("i64", 8),
    ];
    // layouts: (dims, strides as a view over a larger source buffer)
    let layouts: Vec<(Vec<usize>, Vec<usize>)> = vec![
        (vec![1000], vec![1]),
        (vec![7, 13, 5], contiguous(&[7, 13, 5])),
        (vec![13, 7], vec![1, 13]),                // transposed
        (vec![4, 1, 9], vec![9, 9, 1]),            // size-1 dim with odd stride
        (vec![6, 5], vec![0, 1]),                  // broadcast
        (vec![3, 4, 5], vec![40, 1, 8]),           // permuted
    ];
    for (tname, es) in types {
        let name = format!("affine_{tname}");
        for (li, (dims, strides)) in layouts.iter().enumerate() {
            let numel: usize = dims.iter().product();
            let src_len = dims.iter().zip(strides).map(|(d, s)| (d - 1) * s).sum::<usize>() + 1;
            for &(grid, block) in &[(4u32, 64u32), (((numel as u32) + 255) / 256, 256), (1, 32)] {
                for with_info in [true, false] {
                    if !with_info && li > 1 { continue; }
                    for inplace in [false, true] {
                        let inp = rng.bytes(src_len.max(numel) * es);
                        let out = rng.bytes(numel * es);
                        let scal = |r: &mut Rng| -> Arg {
                            match tname {
                                "f32" => Arg::F32(r.f32s(1)[0]),
                                "f64" => Arg::F64(r.f64s(1)[0]),
                                "bf16" | "f16" | "i16" => Arg::B16(r.b16s(1)[0]),
                                "f8_e4m3" | "u8" => Arg::B8(r.next() as u8),
                                "u32" | "i32" => Arg::U32(r.next() as u32),
                                _ => Arg::U64(r.next()),
                            }
                        };
                        let (m, a) = (scal(&mut rng), scal(&mut rng));
                        let info = if with_info { Arg::Buf(2) } else { Arg::Null };
                        let inp_arg = if inplace { Arg::Null } else { Arg::Buf(0) };
                        let args = [Arg::U64(numel as u64), Arg::U64(dims.len() as u64), info, inp_arg, Arg::Buf(1), m, a];
                        let bufs = vec![inp, out, layout_info(dims, strides)];
                        let d = h.diff(&name, (grid, 1, 1), (block, 1, 1), 0, &args, &bufs, &[1]);
                        t.record(&format!("{name} layout{li} g{grid}x{block} info={with_info} inplace={inplace}"), &d);
                    }
                }
            }
        }
    }
    let _ = as_bytes::<u8>(&[]);
    std::process::exit(if t.finish("affine") { 0 } else { 1 });
}
