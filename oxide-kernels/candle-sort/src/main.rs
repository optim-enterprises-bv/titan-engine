//! candle-kernels `sort.cu` (llama.cpp bitonic argsort) in cuda-oxide: same 14 entry names
//! (asort_asc_* / asort_desc_*), same ABI (x, dst: *u32, ncols: i32, ncols_pad: i32), bit-identical
//! output (checked by the host `main` against candle's own nvcc PTX).
//!
//! Semantics copied from the reference:
//! - one block per row, `extern __shared__ int dst_row[ncols_pad]` (dynamic shared memory);
//! - all index arithmetic is signed 32-bit `int` (row * ncols, col, ixj), columns strided by blockDim;
//! - compare-and-swap exactly as the C source: padding indices (>= ncols) sink to the end; the
//!   value compare is a strict `>` / `<` (ties and NaN never swap: every NaN compare is false);
//!   f16/bf16 compare as their exact f32 values (so +0 == -0 is a tie);
//! - one `__syncthreads` after the init and after every j pass;
//! - the written index type is the shared `int` stored as u32.
use cuda_device::{DynamicSharedArray, convert, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// The C `>` / `<` on the element type.
    pub trait Key: Copy {
        fn gt(self, o: Self) -> bool;
        fn lt(self, o: Self) -> bool;
    }
    macro_rules! key_prim {
        ($($t:ty),*) => {$(
            impl Key for $t {
                #[inline(always)] fn gt(self, o: Self) -> bool { self > o }
                #[inline(always)] fn lt(self, o: Self) -> bool { self < o }
            }
        )*};
    }
    key_prim!(f32, f64, u8, u32, i64);

    /// f16 bit pattern (`__half` compares as its exact f32 value).
    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct H(pub u16);
    impl Key for H {
        #[inline(always)] fn gt(self, o: Self) -> bool { convert::cvt_f32_f16x2_lo(self.0 as u32) > convert::cvt_f32_f16x2_lo(o.0 as u32) }
        #[inline(always)] fn lt(self, o: Self) -> bool { convert::cvt_f32_f16x2_lo(self.0 as u32) < convert::cvt_f32_f16x2_lo(o.0 as u32) }
    }
    /// bf16 bit pattern (exact widening to f32 by a 16-bit shift).
    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct B(pub u16);
    impl Key for B {
        #[inline(always)] fn gt(self, o: Self) -> bool { f32::from_bits((self.0 as u32) << 16) > f32::from_bits((o.0 as u32) << 16) }
        #[inline(always)] fn lt(self, o: Self) -> bool { f32::from_bits((self.0 as u32) << 16) < f32::from_bits((o.0 as u32) << 16) }
    }

    /// `k_argsort<order>`; `ASC` selects the order.
    #[inline(always)]
    pub unsafe fn argsort<T: Key, const ASC: bool>(x: *const T, dst: *mut u32, ncols: i32, ncols_pad: i32) {
        let dst_row: *mut i32 = DynamicSharedArray::<i32>::get();
        let row = thread::blockIdx_x() as i32;
        let x_row = x.offset(row.wrapping_mul(ncols) as isize);
        let tid = thread::threadIdx_x() as i32;
        let bdim = thread::blockDim_x() as i32;

        let mut col = tid;
        while col < ncols_pad {
            *dst_row.offset(col as isize) = col;
            col = col.wrapping_add(bdim);
        }
        thread::sync_threads();

        let mut k: i32 = 2;
        while k <= ncols_pad {
            let mut j = k / 2;
            while j > 0 {
                let mut col = tid;
                while col < ncols_pad {
                    let ixj = col ^ j;
                    if ixj > col {
                        let a = *dst_row.offset(col as isize);
                        let b = *dst_row.offset(ixj as isize);
                        let swap = if (col & k) == 0 {
                            a >= ncols
                                || (b < ncols && {
                                    let (va, vb) = (*x_row.offset(a as isize), *x_row.offset(b as isize));
                                    if ASC { va.gt(vb) } else { va.lt(vb) }
                                })
                        } else {
                            b >= ncols
                                || (a < ncols && {
                                    let (va, vb) = (*x_row.offset(a as isize), *x_row.offset(b as isize));
                                    if ASC { va.lt(vb) } else { va.gt(vb) }
                                })
                        };
                        if swap {
                            *dst_row.offset(col as isize) = b;
                            *dst_row.offset(ixj as isize) = a;
                        }
                    }
                    col = col.wrapping_add(bdim);
                }
                thread::sync_threads();
                j /= 2;
            }
            k = k.wrapping_mul(2);
        }

        let mut col = tid;
        while col < ncols {
            *dst.offset(row.wrapping_mul(ncols).wrapping_add(col) as isize) = *dst_row.offset(col as isize) as u32;
            col = col.wrapping_add(bdim);
        }
    }

    #[kernel] pub unsafe fn asort_asc_bf16(x: *const u16, dst: *mut u32, ncols: i32, ncols_pad: i32) { unsafe { argsort::<B, true>(x as *const B, dst, ncols, ncols_pad) } }
    #[kernel] pub unsafe fn asort_desc_bf16(x: *const u16, dst: *mut u32, ncols: i32, ncols_pad: i32) { unsafe { argsort::<B, false>(x as *const B, dst, ncols, ncols_pad) } }
    #[kernel] pub unsafe fn asort_asc_f16(x: *const u16, dst: *mut u32, ncols: i32, ncols_pad: i32) { unsafe { argsort::<H, true>(x as *const H, dst, ncols, ncols_pad) } }
    #[kernel] pub unsafe fn asort_desc_f16(x: *const u16, dst: *mut u32, ncols: i32, ncols_pad: i32) { unsafe { argsort::<H, false>(x as *const H, dst, ncols, ncols_pad) } }
    #[kernel] pub unsafe fn asort_asc_f32(x: *const f32, dst: *mut u32, ncols: i32, ncols_pad: i32) { unsafe { argsort::<f32, true>(x, dst, ncols, ncols_pad) } }
    #[kernel] pub unsafe fn asort_desc_f32(x: *const f32, dst: *mut u32, ncols: i32, ncols_pad: i32) { unsafe { argsort::<f32, false>(x, dst, ncols, ncols_pad) } }
    #[kernel] pub unsafe fn asort_asc_f64(x: *const f64, dst: *mut u32, ncols: i32, ncols_pad: i32) { unsafe { argsort::<f64, true>(x, dst, ncols, ncols_pad) } }
    #[kernel] pub unsafe fn asort_desc_f64(x: *const f64, dst: *mut u32, ncols: i32, ncols_pad: i32) { unsafe { argsort::<f64, false>(x, dst, ncols, ncols_pad) } }
    #[kernel] pub unsafe fn asort_asc_u8(x: *const u8, dst: *mut u32, ncols: i32, ncols_pad: i32) { unsafe { argsort::<u8, true>(x, dst, ncols, ncols_pad) } }
    #[kernel] pub unsafe fn asort_desc_u8(x: *const u8, dst: *mut u32, ncols: i32, ncols_pad: i32) { unsafe { argsort::<u8, false>(x, dst, ncols, ncols_pad) } }
    #[kernel] pub unsafe fn asort_asc_u32(x: *const u32, dst: *mut u32, ncols: i32, ncols_pad: i32) { unsafe { argsort::<u32, true>(x, dst, ncols, ncols_pad) } }
    #[kernel] pub unsafe fn asort_desc_u32(x: *const u32, dst: *mut u32, ncols: i32, ncols_pad: i32) { unsafe { argsort::<u32, false>(x, dst, ncols, ncols_pad) } }
    #[kernel] pub unsafe fn asort_asc_i64(x: *const i64, dst: *mut u32, ncols: i32, ncols_pad: i32) { unsafe { argsort::<i64, true>(x, dst, ncols, ncols_pad) } }
    #[kernel] pub unsafe fn asort_desc_i64(x: *const i64, dst: *mut u32, ncols: i32, ncols_pad: i32) { unsafe { argsort::<i64, false>(x, dst, ncols, ncols_pad) } }
}

// ---------------------------------------------------------------------------------------------
// Differential gate against candle's nvcc-built sort.ptx.
use kdiff::{Arg, Harness, Rng, Tally, as_bytes};

/// Row data with many ties, signed zeros, NaNs (both signs, several payloads) and infinities.
fn gen_data(rng: &mut Rng, vt: &str, n: usize, tie_heavy: bool) -> Vec<u8> {
    let pick = |rng: &mut Rng| rng.next() % if tie_heavy { 2 } else { 4 } != 0;
    match vt {
        "f32" => {
            let pool = [0.0f32, -0.0, 1.0, -1.0, f32::NAN, -f32::NAN, f32::from_bits(0x7fc0_1234), f32::INFINITY, f32::NEG_INFINITY, 2.5, 1e-40];
            let v: Vec<f32> = (0..n).map(|_| if pick(rng) { pool[(rng.next() % pool.len() as u64) as usize] } else { rng.f32s(1)[0] }).collect();
            as_bytes(&v)
        }
        "f64" => {
            let pool = [0.0f64, -0.0, 1.0, -1.0, f64::NAN, -f64::NAN, f64::from_bits(0x7ff0_0000_0000_0001), f64::INFINITY, f64::NEG_INFINITY, 2.5, 1e-310];
            let v: Vec<f64> = (0..n).map(|_| if pick(rng) { pool[(rng.next() % pool.len() as u64) as usize] } else { rng.f64s(1)[0] }).collect();
            as_bytes(&v)
        }
        "f16" | "bf16" => {
            let pool: [u16; 10] = if vt == "f16" {
                [0x0000, 0x8000, 0x3c00, 0xbc00, 0x7e00, 0xfe00, 0x7c01, 0x7c00, 0xfc00, 0x0001]
            } else {
                [0x0000, 0x8000, 0x3f80, 0xbf80, 0x7fc0, 0xffc0, 0x7f81, 0x7f80, 0xff80, 0x0001]
            };
            let v: Vec<u16> = (0..n).map(|_| if pick(rng) { pool[(rng.next() % pool.len() as u64) as usize] } else { rng.next() as u16 }).collect();
            as_bytes(&v)
        }
        "u8" => (0..n).map(|_| if pick(rng) { (rng.next() % 5) as u8 } else { rng.next() as u8 }).collect(),
        "u32" => {
            let v: Vec<u32> = (0..n).map(|_| if pick(rng) { [0u32, 1, 7, u32::MAX, 0x8000_0000][(rng.next() % 5) as usize] } else { rng.next() as u32 }).collect();
            as_bytes(&v)
        }
        "i64" => {
            let v: Vec<i64> = (0..n).map(|_| if pick(rng) { [0i64, 1, -1, i64::MIN, i64::MAX][(rng.next() % 5) as usize] } else { rng.next() as i64 }).collect();
            as_bytes(&v)
        }
        _ => panic!("{vt}"),
    }
}

fn main() {
    let root = format!("{}/titan-engine/oxide-kernels", std::env::var("HOME").unwrap());
    let ref_ptx = format!("{root}/reference/candle/sort.ptx");
    let h = Harness::new(&ref_ptx, &format!("{root}/candle-sort/candle_sort.ptx"));
    let names: Vec<String> = std::fs::read_to_string(&ref_ptx)
        .unwrap()
        .lines()
        .filter_map(|l| l.strip_prefix(".visible .entry ").map(|s| s.trim_end_matches('(').to_string()))
        .collect();
    assert_eq!(names.len(), 14, "reference entry count");
    let mut t = Tally::default();
    let mut rng = Rng(0x5027);
    let mut missing = vec![];
    let ncols_list = [1usize, 2, 3, 4, 5, 7, 8, 16, 17, 31, 32, 33, 64, 100, 127, 128, 255, 256, 500, 1000, 1023, 1024, 1025, 2000, 4096, 5000, 8192];
    for name in &names {
        if !h.has(true, name) {
            missing.push(name.clone());
            continue;
        }
        let vt = name.rsplit('_').next().unwrap();
        for &ncols in &ncols_list {
            let ncols_pad = ncols.next_power_of_two();
            let nrows = if ncols >= 2000 { 2 } else { 3 };
            let shared = (ncols_pad * 4) as u32;
            // candle's launch (block = min(ncols_pad, 1024)), plus smaller blocks (several columns per thread)
            let mut blocks = vec![ncols_pad.min(1024) as u32];
            for b in [32u32, 96] {
                if (b as usize) < ncols_pad { blocks.push(b); }
            }
            for block in blocks {
                for tie_heavy in [true, false] {
                    let es = match vt { "u8" => 1, "f16" | "bf16" => 2, "f32" | "u32" => 4, _ => 8 };
                    let x = gen_data(&mut rng, vt, nrows * ncols, tie_heavy);
                    assert_eq!(x.len(), nrows * ncols * es);
                    let bufs = vec![x, rng.bytes((nrows * ncols + 3) * 4)];
                    let args = [Arg::Buf(0), Arg::Buf(1), Arg::I32(ncols as i32), Arg::I32(ncols_pad as i32)];
                    let d = h.diff(name, (nrows as u32, 1, 1), (block, 1, 1), shared, &args, &bufs, &[1]);
                    t.record(&format!("{name} ncols{ncols} block{block} ties={tie_heavy}"), &d);
                }
            }
        }
    }
    for m in &missing {
        println!("  MISSING from oxide PTX: {m}");
    }
    let ok = t.finish("sort") && missing.is_empty();
    println!("sort: {} of {} reference entries ported", names.len() - missing.len(), names.len());
    std::process::exit(if ok { 0 } else { 1 });
}
