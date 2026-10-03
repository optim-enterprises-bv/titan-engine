#!/usr/bin/env python3
"""Build a standalone checker around candle's IQ3_S CPU dequantize, copied verbatim from
top-iq3s/candle/candle-core/src/quantized/iquant.rs (IQ3S_GRID .. end of fn iq3s_block), and
compare it bit-for-bit with llama.cpp's dequantize_row_iq3_s output written by the
oxide-kernels/iq3_s gate (ref/cpu_blocks.bin, ref/cpu_deq.bin)."""
import os, re
Q = "~/titan-engine/top-iq3s/candle/candle-core/src/quantized"
src = open(f"{Q}/iquant.rs").read()
a = src.index("pub const IQ3S_GRID")
b = src.index("iquant_trait!(BlockIQ3s")
body = src[a:b].replace('include!("data_iq3s_grid.rs")', f'include!("{Q}/data_iq3s_grid.rs")')
kmask = re.search(r"const KMASK_IQ2XS: \[u8; 8\] = [^;]+;", src).group(0)
prog = '''#![allow(dead_code, non_camel_case_types)]
const QK_K: usize = 256;
/// minimal `half::f16` stand-in: exact f16 -> f32 widening.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct f16(u16);
impl f16 {
    fn to_f32(self) -> f32 {
        let h = self.0;
        let s = ((h >> 15) as u32) << 31;
        let e = ((h >> 10) & 0x1f) as u32;
        let m = (h & 0x3ff) as u32;
        f32::from_bits(if e == 0 {
            if m == 0 { s } else {
                let (mut m, mut e) = (m, 113u32);
                while m & 0x400 == 0 { m <<= 1; e -= 1; }
                s | (e << 23) | ((m & 0x3ff) << 13)
            }
        } else if e == 31 { s | 0x7f80_0000 | (m << 13) } else { s | ((e + 112) << 23) | (m << 13) })
    }
}
''' + kmask + "\n" + body + '''
fn main() {
    let dir = "~/titan-engine/oxide-iq3s/iq3_s/ref";
    let blocks = std::fs::read(format!("{dir}/cpu_blocks.bin")).expect("cpu_blocks.bin (run the iq3_s gate first)");
    let want = std::fs::read(format!("{dir}/cpu_deq.bin")).expect("cpu_deq.bin");
    let nb = blocks.len() / 110;
    let xs: &[BlockIQ3s] = unsafe { std::slice::from_raw_parts(blocks.as_ptr() as *const BlockIQ3s, nb) };
    let mut ys = vec![0f32; nb * 256];
    for (x, y) in xs.iter().zip(ys.chunks_exact_mut(256)) { iq3s_block(x, y); }
    let got: Vec<u8> = ys.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(got.len(), want.len());
    let bad = got.chunks(4).zip(want.chunks(4)).filter(|(a, b)| a != b).count();
    println!("candle iquant.rs iq3s_block vs llama.cpp dequantize_row_iq3_s: {nb} blocks, {} values, {bad} differing -> {}",
             nb * 256, if bad == 0 { "PASS" } else { "FAIL" });
    std::process::exit(if bad == 0 { 0 } else { 1 });
}
'''
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "check.rs"), "w").write(prog)
print("wrote check.rs")
