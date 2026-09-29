//! Differential gate against candle's nvcc-built reduce.ptx.
use kdiff::{Arg, Harness, Rng, Tally, as_bytes, contiguous, layout_info};

#[derive(Clone, Copy, PartialEq)]
enum Ty {
    Bf16,
    F16,
    F32,
    F64,
    U32,
    I64,
    U8,
}

impl Ty {
    fn size(self) -> usize {
        match self {
            Ty::Bf16 | Ty::F16 => 2,
            Ty::F32 | Ty::U32 => 4,
            Ty::F64 | Ty::I64 => 8,
            Ty::U8 => 1,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Ty::Bf16 => "bf16",
            Ty::F16 => "f16",
            Ty::F32 => "f32",
            Ty::F64 => "f64",
            Ty::U32 => "u32",
            Ty::I64 => "i64",
            Ty::U8 => "u8",
        }
    }
}

/// Data modes.
#[derive(Clone, Copy, Debug)]
enum Mode {
    /// Moderate finite values of varied magnitude.
    Clean,
    /// Clean plus NaN / inf / +-0 / denormals / raw bit patterns.
    Edge,
    /// A handful of distinct values (argmin/argmax ties, +-0).
    Ties,
    /// Small integers in [-4, 4] (exact under any summation order: atomics).
    SmallInt,
}

fn f16_bits(r: &mut Rng) -> u16 {
    // sign | exponent 8..=17 (2^-7 .. 4) | random mantissa
    let x = r.next();
    (((x & 1) as u16) << 15) | ((8 + (x >> 1) % 10) as u16) << 10 | ((x >> 8) as u16 & 0x3FF)
}

fn f32_to_f16_small_int(v: i32) -> u16 {
    // exact small integers |v| <= 2048
    if v == 0 {
        return 0;
    }
    let s = if v < 0 { 0x8000u16 } else { 0 };
    let a = v.unsigned_abs();
    let e = 31 - a.leading_zeros(); // floor(log2)
    let mant = ((a << (10 - e)) & 0x3FF) as u16;
    s | (((e + 15) as u16) << 10) | mant
}

fn data(r: &mut Rng, t: Ty, n: usize, mode: Mode) -> Vec<u8> {
    let small = |r: &mut Rng| (r.next() % 9) as i32 - 4;
    match t {
        Ty::F32 => {
            let v: Vec<f32> = match mode {
                Mode::Clean => (0..n).map(|_| ((r.next() >> 11) as f64 / (1u64 << 53) as f64 * 16.0 - 8.0) as f32).collect(),
                Mode::Edge => r.f32s(n),
                Mode::Ties => (0..n).map(|_| [0.0f32, -0.0, 1.0, -1.0, 2.0, 0.5][(r.next() % 6) as usize]).collect(),
                Mode::SmallInt => (0..n).map(|_| small(r) as f32).collect(),
            };
            as_bytes(&v)
        }
        Ty::F64 => {
            let v: Vec<f64> = match mode {
                Mode::Clean => (0..n).map(|_| (r.next() >> 11) as f64 / (1u64 << 53) as f64 * 16.0 - 8.0).collect(),
                Mode::Edge => r.f64s(n).into_iter().map(|v| {
                    // kdiff's f64s has one NaN payload; DADD/DMUL/DFMA pick between two NaN
                    // operands by position, so vary sign, payload and signalling-ness.
                    if r.next() % 24 == 0 {
                        f64::from_bits([0x7ff8_0000_0000_0000u64, 0xfff8_0000_0000_0000, 0x7ff0_0000_0000_0001, 0xfff4_0000_0000_0123,
                            0x7ffc_0000_0000_abcd, 0xfff0_dead_beef_0001][(r.next() % 6) as usize])
                    } else { v }
                }).collect(),
                Mode::Ties => (0..n).map(|_| [0.0f64, -0.0, 1.0, -1.0, 2.0, 0.5][(r.next() % 6) as usize]).collect(),
                Mode::SmallInt => (0..n).map(|_| small(r) as f64).collect(),
            };
            as_bytes(&v)
        }
        Ty::Bf16 => {
            let v: Vec<u16> = match mode {
                Mode::Clean => (0..n).map(|_| (((r.next() >> 11) as f64 / (1u64 << 53) as f64 * 16.0 - 8.0) as f32).to_bits().wrapping_shr(16) as u16).collect(),
                Mode::Edge => r.f32s(n).iter().map(|x| (x.to_bits() >> 16) as u16).zip(r.b16s(n)).enumerate()
                    .map(|(i, (a, b))| if i % 13 == 5 { b } else { a }).collect(),
                Mode::Ties => (0..n).map(|_| [0x0000u16, 0x8000, 0x3F80, 0xBF80, 0x4000, 0x3F00][(r.next() % 6) as usize]).collect(),
                Mode::SmallInt => (0..n).map(|_| ((small(r) as f32).to_bits() >> 16) as u16).collect(),
            };
            as_bytes(&v)
        }
        Ty::F16 => {
            let v: Vec<u16> = match mode {
                Mode::Clean => (0..n).map(|_| f16_bits(r)).collect(),
                Mode::Edge => (0..n).map(|i| match r.next() % 12 {
                    0 => [0x0000u16, 0x8000, 0x7C00, 0xFC00, 0x7E00, 0x0001, 0x83FF, 0x7BFF, 0xFBFF][i % 9],
                    1 => r.next() as u16,
                    _ => f16_bits(r),
                }).collect(),
                Mode::Ties => (0..n).map(|_| [0x0000u16, 0x8000, 0x3C00, 0xBC00, 0x4000, 0x3800][(r.next() % 6) as usize]).collect(),
                Mode::SmallInt => (0..n).map(|_| f32_to_f16_small_int(small(r))).collect(),
            };
            as_bytes(&v)
        }
        Ty::U32 => {
            let v: Vec<u32> = match mode {
                Mode::Ties | Mode::SmallInt => (0..n).map(|_| (r.next() % 3) as u32).collect(),
                _ => (0..n).map(|_| r.next() as u32).collect(),
            };
            as_bytes(&v)
        }
        Ty::I64 => {
            let v: Vec<i64> = match mode {
                Mode::Ties | Mode::SmallInt => (0..n).map(|_| (r.next() % 3) as i64 - 1).collect(),
                Mode::Edge => (0..n).map(|i| match i % 7 { 0 => i64::MIN, 1 => i64::MAX, _ => r.next() as i64 }).collect(),
                Mode::Clean => (0..n).map(|_| r.next() as i64).collect(),
            };
            as_bytes(&v)
        }
        Ty::U8 => match mode {
            Mode::Ties | Mode::SmallInt => (0..n).map(|_| (r.next() % 3) as u8).collect(),
            _ => r.bytes(n),
        },
    }
}

fn src_len(dims: &[usize], strides: &[usize]) -> usize {
    if dims.contains(&0) {
        return 1;
    }
    dims.iter().zip(strides).map(|(d, s)| (d - 1) * s).sum::<usize>() + 1
}

const ALL: [Ty; 7] = [Ty::Bf16, Ty::F16, Ty::F32, Ty::F64, Ty::U32, Ty::I64, Ty::U8];
const FLOATS: [Ty; 4] = [Ty::Bf16, Ty::F16, Ty::F32, Ty::F64];

fn fast_ops(h: &Harness, t: &mut Tally, rng: &mut Rng) {
    // (dims, strides, dst_el): the reduced dims are last (candle's FastReduce layout).
    let mut layouts: Vec<(Vec<usize>, Vec<usize>, usize)> = vec![];
    for r in [1usize, 2, 5, 31, 32, 33, 100, 1000, 1024, 1025, 2500] {
        layouts.push((vec![r], vec![1], 1));
        layouts.push((vec![3, r], contiguous(&[3, r]), 3));
        layouts.push((vec![5, r], vec![1, 5], 5)); // reduce over a transposed dim
    }
    layouts.push((vec![4, 6, 7], contiguous(&[4, 6, 7]), 4)); // two reduced dims
    layouts.push((vec![4, 6, 7], vec![1, 28, 4], 4)); // permuted
    layouts.push((vec![3, 1, 9], vec![9, 9, 1], 3)); // size-1 dim
    layouts.push((vec![6, 50], vec![0, 1], 6)); // broadcast output rows
    for ty in ALL {
        for op in ["sum", "min", "max", "argmin", "argmax"] {
            let name = format!("fast_{op}_{}", ty.name());
            let arg = op.starts_with("arg");
            for (li, (dims, strides, dst_el)) in layouts.iter().enumerate() {
                let numel: usize = dims.iter().product();
                let per_block = numel / dst_el;
                let candle_block = per_block.min(1024).next_power_of_two() as u32;
                let mut blocks = vec![candle_block];
                if li % 3 == 0 {
                    blocks.push(32);
                    blocks.push(1024);
                    blocks.push(96); // not a power of two: the tree drops lanes identically
                }
                for &block in &blocks {
                    for mode in [Mode::Edge, Mode::Ties, Mode::Clean] {
                        let src = data(rng, ty, src_len(dims, strides), mode);
                        let out_es = if arg { 4 } else { ty.size() };
                        let dst = rng.bytes(dst_el * out_es);
                        let args = [Arg::U64(numel as u64), Arg::U64(per_block as u64), Arg::U64(dims.len() as u64), Arg::Buf(0), Arg::Buf(1), Arg::Buf(2)];
                        let bufs = vec![layout_info(dims, strides), src, dst];
                        let d = h.diff(&name, (*dst_el as u32, 1, 1), (block, 1, 1), 0, &args, &bufs, &[2]);
                        t.record(&format!("{name} layout{li} {dims:?}/{strides:?} block{block} {mode:?}"), &d);
                    }
                }
            }
        }
    }
}

fn sum_ops(h: &Harness, t: &mut Tally, rng: &mut Rng) {
    // (dims, strides, sum_dims_l, sum_dims_s, out_len)
    let cases: Vec<(Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>, usize)> = vec![
        (vec![4, 6, 5], contiguous(&[4, 6, 5]), vec![6], vec![5], 20),
        (vec![4, 6, 5], contiguous(&[4, 6, 5]), vec![4, 5], vec![30, 1], 6),
        (vec![4, 6, 5], vec![1, 4, 24], vec![6], vec![5], 20), // non-contiguous
        (vec![7, 30], contiguous(&[7, 30]), vec![30], vec![1], 7),
        (vec![30, 7], vec![1, 30], vec![30], vec![7], 7), // transposed, reduce dim 0
        (vec![1000], vec![1], vec![1000], vec![1], 1),
        (vec![3, 1, 9], vec![9, 9, 1], vec![1], vec![9], 27), // size-1 sum dim: one add per output
        (vec![300], vec![1], vec![], vec![], 300),            // no sum dims: one add per output
        (vec![20, 15], vec![1, 20], vec![], vec![], 300),     // strided, one add per output
    ];
    for ty in [Ty::Bf16, Ty::F16, Ty::F32, Ty::F64, Ty::U32] {
        let name = format!("sum_{}", ty.name());
        for (ci, (dims, strides, l, s, out_len)) in cases.iter().enumerate() {
            let numel: usize = dims.iter().product();
            let one_add = numel == *out_len;
            // Float atomics are order-dependent: only exact data unless every output gets one add.
            // bf16 holds integers exactly only up to 256: longer reductions of [-4, 4] can round
            // differently under a different (legal) atomic order, even reference vs reference.
            if ty == Ty::Bf16 && numel / out_len > 64 {
                continue;
            }
            let modes: &[Mode] = if one_add || ty == Ty::U32 { &[Mode::Edge, Mode::SmallInt] } else { &[Mode::SmallInt] };
            let mut info: Vec<usize> = dims.clone();
            info.extend(strides);
            info.extend(l);
            info.extend(s);
            let info_b = as_bytes(&info.iter().map(|&x| x as u64).collect::<Vec<u64>>());
            for &(grid, block) in &[(((numel as u32) + 255) / 256, 256u32), (2, 64), (1, 32)] {
                for &mode in modes {
                    let inp = data(rng, ty, src_len(dims, strides), mode);
                    let out = if matches!(mode, Mode::SmallInt) { data(rng, ty, *out_len, Mode::SmallInt) } else { data(rng, ty, *out_len, Mode::Edge) };
                    let args = [Arg::U64(numel as u64), Arg::U64(dims.len() as u64), Arg::U64(l.len() as u64), Arg::Buf(0), Arg::Buf(1), Arg::Buf(2)];
                    let d = h.diff(&name, (grid, 1, 1), (block, 1, 1), 0, &args, &[info_b.clone(), inp, out], &[2]);
                    t.record(&format!("{name} case{ci} g{grid}x{block} {mode:?}"), &d);
                }
            }
        }
    }
}

fn softmax_ops(h: &Harness, t: &mut Tally, rng: &mut Rng) {
    for ty in FLOATS {
        let name = format!("softmax_{}", ty.name());
        for ncols in [1usize, 3, 31, 32, 33, 100, 257, 1000, 4097] {
            let rows = 6usize;
            for &(grid, block) in &[((rows as u32, 1, 1), (1u32, 32u32, 1u32)), ((rows as u32 / 2, 1, 1), (2, 16, 1))] {
                for mode in [Mode::Clean, Mode::Edge, Mode::Ties] {
                    for inplace in [false, true] {
                        let src = data(rng, ty, rows * ncols, mode);
                        let dst = rng.bytes(rows * ncols * ty.size());
                        let (bufs, args, outs) = if inplace {
                            (vec![src], [Arg::Buf(0), Arg::Buf(0), Arg::I32(ncols as i32)], vec![0])
                        } else {
                            (vec![src, dst], [Arg::Buf(0), Arg::Buf(1), Arg::I32(ncols as i32)], vec![1])
                        };
                        let d = h.diff(&name, grid, block, 0, &args, &bufs, &outs);
                        t.record(&format!("{name} ncols{ncols} block{block:?} {mode:?} inplace={inplace}"), &d);
                    }
                }
            }
        }
    }
}

/// exp coverage: rows of [+0, v] make the kernel evaluate exp(v - 0) = exp(v) for every
/// non-positive v: all 32768 negative f16/bf16 patterns (incl. the hexp fix-up points 0xC13B,
/// 0xC1EF, denormals, -inf, NaNs), and 2^16 random negative f32/f64 patterns.
fn softmax_exp_sweep(h: &Harness, t: &mut Tally, rng: &mut Rng) {
    for ty in FLOATS {
        let name = format!("softmax_{}", ty.name());
        let rows = 32768usize.max(if ty.size() > 2 { 65536 } else { 0 });
        let mut row_vals: Vec<u64> = vec![];
        for r in 0..rows {
            row_vals.push(match ty {
                Ty::Bf16 | Ty::F16 => 0x8000 | r as u64,
                Ty::F32 => 0x8000_0000 | (rng.next() as u32 as u64),
                _ => 0x8000_0000_0000_0000 | rng.next(),
            });
        }
        let mut src: Vec<u8> = vec![];
        for v in &row_vals {
            let b = v.to_le_bytes();
            src.extend(std::iter::repeat_n(0u8, ty.size()));
            src.extend(&b[..ty.size()]);
        }
        let dst = rng.bytes(src.len());
        let d = h.diff(&name, (rows as u32, 1, 1), (1, 32, 1), 0, &[Arg::Buf(0), Arg::Buf(1), Arg::I32(2)], &[src, dst], &[1]);
        t.record(&format!("{name} exp sweep"), &d);
    }
}

fn norm_ops(h: &Harness, t: &mut Tally, rng: &mut Rng) {
    for ty in FLOATS {
        for ln in [false, true] {
            let name = format!("{}_{}", if ln { "layernorm" } else { "rmsnorm" }, ty.name());
            for ncols in [1usize, 7, 32, 33, 100, 1023, 1024, 1500, 4096, 5000] {
                let rows = 4usize;
                let bs: u32 = if ncols < 1024 { 32 } else { 1024 };
                // (grid, block, block_size param)
                let mut shapes = vec![((rows as u32, 1, 1), (bs, 1, 1), bs)];
                if bs == 32 {
                    shapes.push(((rows as u32 / 2, 1, 1), (32, 2, 1), 32)); // two rows per block
                    shapes.push(((rows as u32, 1, 1), (1024, 1, 1), 1024)); // large block on a short row
                }
                for &(grid, block, bsz) in &shapes {
                    for mode in [Mode::Clean, Mode::Edge] {
                        for with_a in [false, true] {
                            for with_b in if ln { vec![false, true] } else { vec![false] } {
                                let eps = if with_a { 1e-5f32 } else { 1e-6f32 };
                                let src = data(rng, ty, rows * ncols, mode);
                                let alpha = data(rng, ty, ncols, Mode::Clean);
                                let beta = data(rng, ty, ncols, Mode::Clean);
                                let dst = rng.bytes(rows * ncols * ty.size());
                                let a = if with_a { Arg::Buf(2) } else { Arg::Null };
                                let b = if with_b { Arg::Buf(3) } else { Arg::Null };
                                let args: Vec<Arg> = if ln {
                                    vec![Arg::Buf(0), Arg::Buf(1), a, b, Arg::I32(ncols as i32), Arg::I32(bsz as i32), Arg::F32(eps)]
                                } else {
                                    vec![Arg::Buf(0), Arg::Buf(1), a, Arg::I32(ncols as i32), Arg::I32(bsz as i32), Arg::F32(eps)]
                                };
                                let d = h.diff(&name, grid, block, 0, &args, &[src, dst, alpha, beta], &[1]);
                                t.record(&format!("{name} ncols{ncols} block{block:?} bs{bsz} {mode:?} a={with_a} b={with_b}"), &d);
                            }
                        }
                    }
                }
            }
        }
    }
}

fn rope_ops(h: &Harness, t: &mut Tally, rng: &mut Rng) {
    // (b, h, t, d)
    let shapes = [(1usize, 1usize, 1usize, 2usize), (2, 3, 5, 8), (1, 4, 7, 64), (2, 2, 33, 128), (3, 1, 2, 6)];
    for ty in FLOATS {
        for kind in ["rope_i", "rope", "rope_thd"] {
            let name = format!("{kind}_{}", ty.name());
            for &(b, hh, tt, d) in &shapes {
                let el = b * hh * tt * d;
                for batched in [false, true] {
                    let stride_b = if batched { (hh * tt * d) as u32 } else { 0 };
                    let cs_len = if batched { b * tt * d / 2 } else { tt * d / 2 };
                    let n = (el / 2) as u32;
                    for &block in &[1024u32, 64] {
                        let grid = n.div_ceil(block);
                        for mode in [Mode::Clean, Mode::Edge] {
                            for inplace in [false, true] {
                                let src = data(rng, ty, el, mode);
                                let cos = data(rng, ty, cs_len, mode);
                                let sin = data(rng, ty, cs_len, mode);
                                let dst = rng.bytes(el * ty.size());
                                let dsti = if inplace { 0 } else { 3 };
                                let mut args = vec![Arg::Buf(0), Arg::Buf(1), Arg::Buf(2), Arg::Buf(dsti)];
                                match kind {
                                    "rope_i" => args.extend([Arg::U32((b * hh) as u32), Arg::U32((tt * d) as u32), Arg::U32(stride_b)]),
                                    "rope" => args.extend([Arg::U32((b * hh) as u32), Arg::U32((tt * d) as u32), Arg::U32(d as u32), Arg::U32(stride_b)]),
                                    _ => args.extend([Arg::U32(b as u32), Arg::U32(tt as u32), Arg::U32(hh as u32), Arg::U32(d as u32), Arg::U32(stride_b)]),
                                }
                                let dd = h.diff(&name, (grid, 1, 1), (block, 1, 1), 0, &args, &[src, cos, sin, dst], &[dsti]);
                                t.record(&format!("{name} {:?} batched={batched} block{block} {mode:?} inplace={inplace}", (b, hh, tt, d)), &dd);
                            }
                        }
                    }
                }
            }
        }
    }
}

pub fn run() -> bool {
    let root = format!("{}/titan-engine/oxide-kernels", std::env::var("HOME").unwrap());
    let reference = format!("{root}/reference/candle/reduce.ptx");
    let h = Harness::new(&reference, &format!("{root}/candle-reduce/candle_reduce.ptx"));
    // Every reference entry must exist in the port.
    let ptx = std::fs::read_to_string(&reference).unwrap();
    let entries: Vec<String> = ptx
        .lines()
        .filter_map(|l| l.strip_prefix(".visible .entry "))
        .map(|l| l.trim_end_matches('(').to_string())
        .collect();
    let missing: Vec<&String> = entries.iter().filter(|e| !h.has(true, e)).collect();
    println!("reduce: {} reference entries, {} present in the port", entries.len(), entries.len() - missing.len());
    for m in &missing {
        println!("  MISSING {m}");
    }
    let mut t = Tally::default();
    let mut rng = Rng(0x5EDC_0DE5);
    let only = std::env::var("ONLY").unwrap_or_default();
    if only.is_empty() || only == "fast" {
        fast_ops(&h, &mut t, &mut rng);
    }
    if only.is_empty() || only == "sum" {
        sum_ops(&h, &mut t, &mut rng);
    }
    if only.is_empty() || only == "softmax" {
        softmax_ops(&h, &mut t, &mut rng);
        softmax_exp_sweep(&h, &mut t, &mut rng);
    }
    if only.is_empty() || only == "norm" {
        norm_ops(&h, &mut t, &mut rng);
    }
    if only.is_empty() || only == "rope" {
        rope_ops(&h, &mut t, &mut rng);
    }
    let mut per: std::collections::BTreeMap<&str, usize> = Default::default();
    for f in &t.failures {
        *per.entry(f.split(' ').next().unwrap()).or_default() += 1;
    }
    for (e, n) in &per {
        println!("  failing launches for {e}: {n}");
    }
    t.finish("reduce") && missing.is_empty()
}
