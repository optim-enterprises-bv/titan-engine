//! Compiles candle's own iquant.rs (via #[path], not a copy) against stand-ins for the two items it
//! imports (k_quants::{GgmlType, BlockQ8_1}, GgmlDType) and compares every type's `to_float` bit for
//! bit with llama.cpp's dequantize_row_* output (ggml_deq.c dumps in ../ref, 1048576 values each).
//! usage: iqcheck REFDIR
#![allow(dead_code, unused_imports)]
mod quantized {
    pub mod k_quants {
        use half::f16;
        pub trait GgmlType: Sized + Clone + Send + Sync {
            const DTYPE: super::GgmlDType;
            const BLCK_SIZE: usize;
            type VecDotType;
            fn to_float(xs: &[Self], ys: &mut [f32]);
            fn from_float(xs: &[f32], ys: &mut [Self]);
            fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32;
            fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32;
        }
        #[derive(Clone)]
        pub struct BlockQ8_1 {
            pub d: f16,
            pub s: f16,
            pub qs: [i8; 32],
        }
    }
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum GgmlDType { IQ2XXS, IQ3XXS, IQ2S, IQ4XS, IQ2XS, IQ3S }
    #[path = "~/titan-engine/top-emb/candle/candle-core/src/quantized/iquant.rs"]
    pub mod iquant;
}
use quantized::iquant::*;
use quantized::k_quants::GgmlType;

fn check<T: GgmlType>(dir: &str, id: u32, name: &str) -> bool {
    let blocks = std::fs::read(format!("{dir}/{id}.blocks")).unwrap();
    let want = std::fs::read(format!("{dir}/{id}.f32")).unwrap();
    let sz = std::mem::size_of::<T>();
    assert_eq!(blocks.len() % sz, 0, "{name}: block size {sz}");
    let nb = blocks.len() / sz;
    let mut xs: Vec<T> = Vec::with_capacity(nb);
    unsafe {
        std::ptr::copy_nonoverlapping(blocks.as_ptr(), xs.as_mut_ptr() as *mut u8, blocks.len());
        xs.set_len(nb);
    }
    let n = want.len() / 4;
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut ys = vec![0f32; n];
        T::to_float(&xs, &mut ys);
        ys
    }));
    let ys = match r {
        Ok(ys) => ys,
        Err(e) => {
            let msg = e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()));
            println!("{name:8} (type {id:2}): {nb} blocks, {n} values: PANIC {msg:?} -> FAIL");
            return false;
        }
    };
    let (mut bad, mut nan, mut first) = (0usize, 0usize, None);
    for (i, (y, w)) in ys.iter().zip(want.chunks_exact(4)).enumerate() {
        let w = f32::from_le_bytes([w[0], w[1], w[2], w[3]]);
        if y.is_nan() && w.is_nan() {
            nan += 1;
        } else if y.to_bits() != w.to_bits() {
            bad += 1;
            first.get_or_insert((i, *y, w));
        }
    }
    println!("{name:8} (type {id:2}): {nb} blocks, {n} values, {nan} NaN in both, {bad} differing{} -> {}",
        first.map(|(i, y, w)| format!(" (first at {i}: candle {y} llama.cpp {w})")).unwrap_or_default(),
        if bad == 0 { "PASS" } else { "FAIL" });
    bad == 0
}

fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    let dir = std::env::args().nth(1).expect("REFDIR");
    let ok = [
        check::<BlockIQ2xxs>(&dir, 16, "IQ2_XXS"),
        check::<BlockIQ2xs>(&dir, 17, "IQ2_XS"),
        check::<BlockIQ3xxs>(&dir, 18, "IQ3_XXS"),
        check::<BlockIQ3s>(&dir, 21, "IQ3_S"),
        check::<BlockIQ2s>(&dir, 22, "IQ2_S"),
        check::<BlockIQ4xs>(&dir, 23, "IQ4_XS"),
    ];
    std::process::exit(if ok.iter().all(|&b| b) { 0 } else { 1 });
}
