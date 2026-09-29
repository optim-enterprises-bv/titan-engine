//! Differential gate against candle's nvcc-built quantized.ptx (host side).
use kdiff::{Arg, Harness, Rng, Tally, as_bytes};

/// A GGML block format: name, bytes per block, values per block, byte offsets of f16 scale
/// fields and of f32 scale fields inside a block.
pub struct QType {
    pub name: &'static str,
    pub bs: usize,
    pub qk: usize,
    pub f16s: &'static [usize],
    pub f32s: &'static [usize],
}

pub const Q4_0: QType = QType { name: "q4_0", bs: 18, qk: 32, f16s: &[0], f32s: &[] };
pub const Q4_1: QType = QType { name: "q4_1", bs: 20, qk: 32, f16s: &[0, 2], f32s: &[] };
pub const Q5_0: QType = QType { name: "q5_0", bs: 22, qk: 32, f16s: &[0], f32s: &[] };
pub const Q5_1: QType = QType { name: "q5_1", bs: 24, qk: 32, f16s: &[0, 2], f32s: &[] };
pub const Q8_0: QType = QType { name: "q8_0", bs: 34, qk: 32, f16s: &[0], f32s: &[] };
pub const Q8_1: QType = QType { name: "q8_1", bs: 36, qk: 32, f16s: &[0, 2], f32s: &[] };
pub const Q2_K: QType = QType { name: "q2_K", bs: 84, qk: 256, f16s: &[80, 82], f32s: &[] };
pub const Q3_K: QType = QType { name: "q3_K", bs: 110, qk: 256, f16s: &[108], f32s: &[] };
pub const Q4_K: QType = QType { name: "q4_K", bs: 144, qk: 256, f16s: &[0, 2], f32s: &[] };
pub const Q5_K: QType = QType { name: "q5_K", bs: 176, qk: 256, f16s: &[0, 2], f32s: &[] };
pub const Q6_K: QType = QType { name: "q6_K", bs: 210, qk: 256, f16s: &[208], f32s: &[] };
pub const Q8_K: QType = QType { name: "q8_K", bs: 292, qk: 256, f16s: &[], f32s: &[0] };

pub struct G {
    pub h: Harness,
    pub t: Tally,
    pub rng: Rng,
    pub per_entry: std::collections::BTreeMap<String, usize>,
    /// 1 in `srate` scale fields / activations is a special value (nan, inf, denormal, ...).
    pub srate: u64,
    pub fails: std::collections::BTreeMap<String, usize>,
}

impl G {
    /// An f16 scale: mostly normal magnitudes, plus every special class.
    pub fn f16_scale(&mut self) -> u16 {
        let r = self.rng.next();
        let sel = if r % self.srate == 0 { (r >> 40) % 2 } else { 2 };
        match sel {
            0 => [0x0000, 0x8000, 0x7c00, 0xfc00, 0x7e00, 0x0001, 0x83ff, 0x7bff, 0x0400, 0xfe01][((r >> 8) % 10) as usize],
            1 => (r >> 16) as u16,
            _ => {
                let sign = ((r >> 8) & 1) as u16;
                let exp = 3 + ((r >> 9) % 16) as u16; // 2^-12 .. 2^3
                let mant = ((r >> 16) & 0x3ff) as u16;
                (sign << 15) | (exp << 10) | mant
            }
        }
    }

    /// `n` random blocks of `q` with scale fields replaced by `f16_scale` / edge f32 values,
    /// plus `pad` trailing random bytes (the reference may read past the last block).
    pub fn blocks(&mut self, q: &QType, n: usize, pad: usize) -> Vec<u8> {
        let mut b = self.rng.bytes(n * q.bs + pad);
        for i in 0..n {
            for &o in q.f16s {
                let v = self.f16_scale();
                b[i * q.bs + o..i * q.bs + o + 2].copy_from_slice(&v.to_le_bytes());
            }
            for &o in q.f32s {
                let v = if self.rng.next() % 8 == 0 { self.rng.f32s(1)[0] } else { (self.f16_scale() as f32 - 30000.0) * 1e-6 };
                b[i * q.bs + o..i * q.bs + o + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        b
    }

    pub fn f32buf(&mut self, n: usize) -> Vec<u8> {
        as_bytes(&self.acts(n))
    }

    /// Activations: uniform in [-8, 8) with 1 in `srate` special values.
    pub fn acts(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|i| {
            let r = self.rng.next();
            if r % self.srate == 0 {
                [0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 1e-40, -1e-42, 3e38, -3e38][((r >> 20) as usize + i) % 9]
            } else {
                ((r >> 11) as f64 / (1u64 << 53) as f64 * 16.0 - 8.0) as f32
            }
        }).collect()
    }

    /// Launch `name` on the reference module only and return the f32 output buffer `out`
    /// (to check that a test's outputs are mostly finite and non-zero, i.e. meaningful).
    #[allow(clippy::too_many_arguments)]
    pub fn probe(&self, name: &str, grid: (u32, u32, u32), block: (u32, u32, u32), args: &[Arg], bufs: &[Vec<u8>], out: usize) -> Vec<f32> {
        use kdiff::cuda_core::DeviceBuffer;
        let st = &self.h.stream;
        let d: Vec<DeviceBuffer<u8>> = bufs.iter().map(|b| DeviceBuffer::from_host(st, b).unwrap()).collect();
        let mut vals: Vec<[u8; 8]> = args.iter().map(|a| match a {
            Arg::Buf(i) => d[*i].cu_deviceptr().to_le_bytes(),
            Arg::I32(v) => (*v as u32 as u64).to_le_bytes(),
            _ => panic!("probe: unsupported arg"),
        }).collect();
        let mut ptrs: Vec<*mut std::ffi::c_void> = vals.iter_mut().map(|v| v.as_mut_ptr() as *mut std::ffi::c_void).collect();
        let f = self.h.reference.load_function(name).unwrap();
        unsafe { kdiff::cuda_core::simt::launch_kernel_on_stream(&f, grid, block, 0, st, &mut ptrs).unwrap(); }
        st.synchronize().unwrap();
        let b = d[out].to_host_vec(st).unwrap();
        b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run(&mut self, name: &str, label: &str, grid: (u32, u32, u32), block: (u32, u32, u32), shared: u32,
               args: &[Arg], bufs: &[Vec<u8>], outs: &[usize]) {
        let d = self.h.diff(name, grid, block, shared, args, bufs, outs);
        *self.per_entry.entry(name.to_string()).or_default() += 1;
        if d.differing > 0 {
            *self.fails.entry(name.to_string()).or_default() += 1;
        }
        self.t.record(&format!("{name} {label} g{grid:?} b{block:?}"), &d);
    }
}

fn reference_entries(path: &str) -> Vec<String> {
    std::fs::read_to_string(path).unwrap().lines()
        .filter_map(|l| l.strip_prefix(".visible .entry "))
        .map(|l| l.trim_end_matches('(').to_string())
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Family 1: dequantize_block_*.
fn dequantize(g: &mut G) {
    // candle's launch: (kernel, block_dim, grid) per format; K formats take no count.
    let k_types: [(&QType, u32); 6] = [(&Q2_K, 64), (&Q3_K, 64), (&Q4_K, 32), (&Q5_K, 64), (&Q6_K, 64), (&Q8_K, 32)];
    for (q, bd) in k_types {
        for &nb in &[1usize, 3, 17, 64] {
            let x = g.blocks(q, nb, 64);
            let n = nb * 256;
            for (suffix, es) in [("f32", 4usize), ("f16", 2)] {
                let name = format!("dequantize_block_{}_{suffix}", q.name);
                let out = g.rng.bytes(n * es + 256);
                g.run(&name, &format!("nb={nb}"), (nb as u32, 1, 1), (bd, 1, 1), 0,
                      &[Arg::Buf(0), Arg::Buf(1)], &[x.clone(), out], &[1]);
            }
        }
    }
    for q in [&Q4_0, &Q4_1, &Q8_0, &Q5_0, &Q5_1] {
        for &n in &[32usize, 96, 256, 32 * 37, 4096, 32 * 129] {
            let nb32 = n / 32;
            let x = g.blocks(q, nb32, 64);
            for (suffix, es) in [("f32", 4usize), ("f16", 2)] {
                let name = format!("dequantize_block_{}_{suffix}", q.name);
                let is5 = q.name.starts_with("q5");
                // candle: q5_x -> block 256, grid ceil(n/512), k = n; others block 32, grid ceil(n/256), k = n/32.
                let cfgs: Vec<(u32, u32, i32)> = if is5 {
                    vec![(n.div_ceil(512) as u32, 256, n as i32), (n.div_ceil(128) as u32, 64, n as i32), (1, 256, n as i32)]
                } else {
                    vec![(n.div_ceil(256) as u32, 32, nb32 as i32), (n.div_ceil(256) as u32 + 2, 32, nb32 as i32)]
                };
                for (grid, bd, k) in cfgs {
                    let out = g.rng.bytes(n * es + 256);
                    g.run(&name, &format!("n={n}"), (grid, 1, 1), (bd, 1, 1), 0,
                          &[Arg::Buf(0), Arg::Buf(1), Arg::I32(k)], &[x.clone(), out], &[1]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Family 2: dequantize_mul_mat_vec_*.
fn dmmv(g: &mut G) {
    let generic: [(&QType, &str); 5] = [(&Q4_0, "q4_0"), (&Q4_1, "q4_1"), (&Q5_0, "q5_0"), (&Q5_1, "q5_1"), (&Q8_0, "q8_0")];
    let ktypes: [(&QType, &str); 5] = [(&Q2_K, "q2_k"), (&Q3_K, "q3_k"), (&Q4_K, "q4_k"), (&Q5_K, "q5_k"), (&Q6_K, "q6_k")];
    for (q, n) in generic.iter().map(|(q, n)| (*q, format!("dequantize_mul_mat_vec_{n}_cuda")))
        .chain(ktypes.iter().map(|(q, n)| (*q, format!("dequantize_mul_mat_vec_{n}"))))
    {
        let colsets: &[usize] = if q.qk == 256 { &[256, 512, 768, 2048, 300] } else { &[64, 256, 96, 1024, 32 * 67] };
        for &ncols in colsets {
            for &nrows in &[1usize, 7, 33] {
                let nblk = ncols.div_ceil(q.qk) * (nrows + 1);
                let x = g.blocks(q, nblk, 256);
                let y = g.f32buf(ncols + 1024);
                let q5k = n.ends_with("q5_k");
                // candle: grid (nrows), block (32, 1). Odd: 2 rows per block (row may reach nrows).
                let mut cfgs = vec![((nrows as u32, 1, 1), (32u32, 1u32, 1u32))];
                if !q5k {
                    cfgs.push(((nrows.div_ceil(2) as u32, 1, 1), (32, 2, 1)));
                }
                for (grid, block) in cfgs {
                    let dst = g.rng.bytes((nrows + 8) * 4);
                    let mut args = vec![Arg::Buf(0), Arg::Buf(1), Arg::Buf(2), Arg::I32(ncols as i32)];
                    if !q5k {
                        args.push(Arg::I32(nrows as i32));
                    }
                    g.run(&n, &format!("ncols={ncols} nrows={nrows}"), grid, block, 0, &args, &[x.clone(), y.clone(), dst], &[2]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Family 3: quantize_q8_1.
fn quantize(g: &mut G) {
    g.srate = 64;
    for &(rows, kx) in &[(1usize, 32usize), (1, 4096), (3, 1000), (7, 257), (4, 2048), (2, 5000), (5, 512)] {
        let kxp = kx.div_ceil(512) * 512;
        let mut x = g.acts(rows * kx);
        if rows >= 3 {
            // Edge rows: all zero (amax == 0), tiny values, exact .5 ties, huge values (f16 overflow).
            for v in &mut x[0..kx] { *v = 0.0; }
            for (i, v) in x[kx..2 * kx].iter_mut().enumerate() { *v = (i as f32 - 300.0) * 1e-30; }
            for (i, v) in x[2 * kx..3 * kx].iter_mut().enumerate() { *v = ((i % 64) as f32 - 32.0) * 0.5; }
        }
        if rows >= 4 {
            for (i, v) in x[3 * kx..4 * kx].iter_mut().enumerate() { *v = if i % 3 == 0 { -0.0 } else { (i as f32) * 1e4 }; }
        }
        if rows >= 2 {
            // roundf targets: every 32-group holds +-127 (amax = 127, d = 1, t = x exactly) and values
            // where round-half-away (add.rz of copysign(0.5)) and other roundings differ.
            let ties = [0.49999997f32, 0.5, 0.50000006, 1.4999999, 1.5, 2.5, 126.5, 125.49999, 3.5, 0.0, -0.0, 1e-30];
            let row = if rows >= 5 { 4 } else { rows - 1 };
            for (i, v) in x[row * kx..(row + 1) * kx].iter_mut().enumerate() {
                let t = ties[(i / 2) % ties.len()];
                *v = match i % 32 { 0 => if (i / 32) % 2 == 0 { 127.0 } else { -127.0 }, _ => if i % 2 == 0 { t } else { -t } };
            }
        }
        let y = g.rng.bytes(rows * kxp / 32 * 36);
        for (grid, block) in [((kxp as u32 / 256, rows as u32, 1), (256u32, 1u32, 1u32)), ((kxp as u32 / 64, rows as u32, 1), (64, 1, 1))] {
            g.run("quantize_q8_1", &format!("rows={rows} kx={kx}"), grid, block, 0,
                  &[Arg::Buf(0), Arg::Buf(1), Arg::I32(kx as i32), Arg::I32(kxp as i32)], &[as_bytes(&x), y.clone()], &[1]);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Family 4: mul_mat_vec_*_q8_1_cudaN and indexed_moe_forward_*.
pub const MMVQ_TYPES: [(&QType, &str); 10] = [(&Q4_0, "q4_0"), (&Q4_1, "q4_1"), (&Q5_0, "q5_0"), (&Q5_1, "q5_1"), (&Q8_0, "q8_0"),
    (&Q2_K, "q2_K"), (&Q3_K, "q3_K"), (&Q4_K, "q4_K"), (&Q5_K, "q5_K"), (&Q6_K, "q6_K")];

fn mmvq(g: &mut G) {
    for nc in 1..=8usize {
        for (q, qn) in MMVQ_TYPES {
            let name = format!("mul_mat_vec_{qn}_q8_1_cuda{nc}");
            let colsets: &[usize] = if q.qk == 256 { &[256, 768, 4096] } else { &[64, 32 * 37, 1024] };
            for &ncols in colsets {
                for &nrows in &[1usize, 2, 7, 40] {
                    let ncols_padded = ncols.div_ceil(512) * 512;
                    let x = g.blocks(q, (ncols / q.qk) * (nrows + 1), 256);
                    let y = g.blocks(&Q8_1, nc * ncols_padded / 32, 64);
                    let (nwarps, rpb) = (if nc <= 4 { 4u32 } else { 2 }, if nc == 1 { 1 } else { 2 });
                    let grid = (nrows as u32).div_ceil(rpb);
                    // nrows_dst: exact, or one more for odd rows with 2 rows per block (the last
                    // block writes row `nrows`; exact would race with the next column's row 0).
                    let nrows_dst = if rpb == 2 && nrows % 2 == 1 { nrows + 1 } else { nrows };
                    for extra in [0usize, 3] {
                        let nd = nrows_dst + extra;
                        let dst = g.rng.bytes((nc * nd + 8) * 4);
                        g.run(&name, &format!("ncols={ncols} nrows={nrows} nrows_dst={nd}"), (grid, 1, 1), (32, nwarps, 1), 0,
                              &[Arg::Buf(0), Arg::Buf(1), Arg::Buf(2), Arg::I32(ncols as i32), Arg::I32(nrows as i32),
                                Arg::I32(ncols_padded as i32), Arg::I32(nd as i32)],
                              &[x.clone(), y.clone(), dst], &[2]);
                    }
                }
            }
        }
    }
}

fn moe(g: &mut G) {
    let types: [(&QType, &str); 6] = [(&Q2_K, "q2k"), (&Q3_K, "q3k"), (&Q4_K, "q4k"), (&Q5_K, "q5k"), (&Q6_K, "q6k"), (&Q8_0, "q8_0")];
    for (q, qn) in types {
        let name = format!("indexed_moe_forward_{qn}_q8_1");
        for &(n, k) in &[(1usize, 256usize), (5, 768), (64, 2048), (3, 512)] {
            let experts = 4usize;
            let stride = (n * k) / 256 * q.bs; // the kernel's expert stride (QK_K even for q8_0)
            let wlen = (experts - 1) * stride + n * (k / q.qk) * q.bs;
            let w = g.blocks(q, wlen.div_ceil(q.bs), 256);
            let k_padded = k.div_ceil(512) * 512;
            for &(batch, topk, dim1_is_one) in &[(1usize, 4usize, true), (3, 2, true), (2, 3, false), (1, 1, false)] {
                let rows = if dim1_is_one { batch } else { batch * topk };
                let xin = g.blocks(&Q8_1, rows * k_padded / 32, 64);
                let ids: Vec<u32> = (0..batch * topk).map(|_| (g.rng.next() % experts as u64) as u32).collect();
                let out = g.rng.bytes(batch * topk * n * 4 + 32);
                g.run(&name, &format!("n={n} k={k} batch={batch} topk={topk} dim1_one={dim1_is_one}"),
                      (n as u32, batch as u32, topk as u32), (32, 4, 1), 0,
                      &[Arg::Buf(0), Arg::Buf(1), Arg::Buf(2), Arg::Buf(3), Arg::I32(n as i32), Arg::I32(k as i32),
                        Arg::I32(batch as i32), Arg::I32(topk as i32), Arg::I32(k_padded as i32),
                        Arg::I32(if dim1_is_one { 1 } else { topk as i32 })],
                      &[w.clone(), xin, as_bytes(&ids), out], &[3]);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Family 5: mul_mat_q*.
fn mmq(g: &mut G) {
    // (format, name, mmq_x, mmq_y) as candle launches them.
    let types: [(&QType, &str, usize, usize); 10] = [(&Q4_0, "q4_0", 64, 128), (&Q4_1, "q4_1", 64, 128), (&Q5_0, "q5_0", 128, 64),
        (&Q5_1, "q5_1", 128, 64), (&Q8_0, "q8_0", 128, 64), (&Q2_K, "q2_K", 64, 128), (&Q3_K, "q3_K", 128, 128),
        (&Q4_K, "q4_K", 64, 128), (&Q5_K, "q5_K", 64, 128), (&Q6_K, "q6_K", 64, 64)];
    for (q, qn, mmq_x, mmq_y) in types {
        let name = format!("mul_mat_{qn}");
        let ks: &[usize] = if q.qk == 256 { &[256, 1024, 4096] } else { &[256, 96, 1024, 4096] };
        for &k in ks {
            for &(x_rows, y_cols) in &[(1usize, 1usize), (50, 3), (135, 70), (64, 128), (129, 130)] {
                let k_padded = k.div_ceil(512) * 512;
                let bpr = k / q.qk;
                let x = g.blocks(q, x_rows * bpr + 16, 512);
                let y = g.blocks(&Q8_1, y_cols * k_padded / 32 + 16, 64);
                let dst = g.rng.bytes(x_rows * y_cols * 4 + 64);
                let grid = (x_rows.div_ceil(mmq_y) as u32, y_cols.div_ceil(mmq_x) as u32, 1);
                if std::env::var("QSTATS").is_ok() && x_rows == 135 {
                    let a = [Arg::Buf(0), Arg::Buf(1), Arg::Buf(2), Arg::I32(k as i32), Arg::I32(x_rows as i32),
                             Arg::I32(y_cols as i32), Arg::I32(k_padded as i32), Arg::I32(x_rows as i32)];
                    let o = g.probe(&name, grid, (32, 4, 1), &a, &[x.clone(), y.clone(), dst.clone()], 2);
                    let o = &o[..x_rows * y_cols];
                    let fin = o.iter().filter(|v| v.is_finite() && **v != 0.0).count();
                    println!("  stats {name} k={k}: {fin}/{} finite non-zero, sample {:?}", o.len(), &o[..3]);
                }
                g.run(&name, &format!("k={k} x_rows={x_rows} y_cols={y_cols}"), grid, (32, 4, 1), 0,
                      &[Arg::Buf(0), Arg::Buf(1), Arg::Buf(2), Arg::I32(k as i32), Arg::I32(x_rows as i32),
                        Arg::I32(y_cols as i32), Arg::I32(k_padded as i32), Arg::I32(x_rows as i32)],
                      &[x, y, dst], &[2]);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Optional timing (not part of the gate): `candle-quantized bench`.
fn bench(g: &mut G) {
    use kdiff::cuda_core::DeviceBuffer;
    g.srate = 1024;
    let st = g.h.stream.clone();
    let time = |m: &std::sync::Arc<kdiff::cuda_core::CudaModule>, name: &str, grid: (u32, u32, u32), block: (u32, u32, u32), args: &[Arg], d: &[DeviceBuffer<u8>]| -> f64 {
        let mut vals: Vec<[u8; 8]> = args.iter().map(|a| match a {
            Arg::Buf(i) => d[*i].cu_deviceptr().to_le_bytes(),
            Arg::I32(v) => (*v as u32 as u64).to_le_bytes(),
            _ => unreachable!(),
        }).collect();
        let mut ptrs: Vec<*mut std::ffi::c_void> = vals.iter_mut().map(|v| v.as_mut_ptr() as *mut std::ffi::c_void).collect();
        let f = m.load_function(name).unwrap();
        let reps = 50;
        for r in 0..reps + 5 {
            if r == 5 { st.synchronize().unwrap(); }
            unsafe { kdiff::cuda_core::simt::launch_kernel_on_stream(&f, grid, block, 0, &st, &mut ptrs).unwrap(); }
        }
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            unsafe { kdiff::cuda_core::simt::launch_kernel_on_stream(&f, grid, block, 0, &st, &mut ptrs).unwrap(); }
        }
        st.synchronize().unwrap();
        t0.elapsed().as_secs_f64() * 1e6 / reps as f64
    };
    let (rows, k, cols) = (1024usize, 4096usize, 256usize);
    let kp = k.div_ceil(512) * 512;
    for (q, qn, mx, my) in [(&Q4_0, "q4_0", 64usize, 128usize), (&Q4_1, "q4_1", 64, 128), (&Q5_0, "q5_0", 128, 64), (&Q5_1, "q5_1", 128, 64),
        (&Q8_0, "q8_0", 128, 64), (&Q2_K, "q2_K", 64, 128), (&Q3_K, "q3_K", 128, 128), (&Q4_K, "q4_K", 64, 128), (&Q5_K, "q5_K", 64, 128), (&Q6_K, "q6_K", 64, 64)] {
        let x = g.blocks(q, rows * k / q.qk, 64);
        let y = g.blocks(&Q8_1, cols * kp / 32, 64);
        let d: Vec<DeviceBuffer<u8>> = [x, y, vec![0u8; rows * cols * 4]].iter().map(|b| DeviceBuffer::from_host(&st, b).unwrap()).collect();
        let args = [Arg::Buf(0), Arg::Buf(1), Arg::Buf(2), Arg::I32(k as i32), Arg::I32(rows as i32), Arg::I32(cols as i32), Arg::I32(kp as i32), Arg::I32(rows as i32)];
        let grid = (rows.div_ceil(my) as u32, cols.div_ceil(mx) as u32, 1);
        let name = format!("mul_mat_{qn}");
        let (a, b) = (time(&g.h.reference, &name, grid, (32, 4, 1), &args, &d), time(&g.h.oxide, &name, grid, (32, 4, 1), &args, &d));
        println!("bench {name} {rows}x{k}x{cols}: reference {a:.1} us, oxide {b:.1} us ({:.2}x)", b / a);
        for nc in [1usize, 8] {
            let name = format!("mul_mat_vec_{qn}_q8_1_cuda{nc}");
            let (nw, rpb) = (if nc <= 4 { 4 } else { 2 }, if nc == 1 { 1 } else { 2 });
            let args = [Arg::Buf(0), Arg::Buf(1), Arg::Buf(2), Arg::I32(k as i32), Arg::I32(rows as i32), Arg::I32(kp as i32), Arg::I32(rows as i32)];
            let grid = ((rows as u32).div_ceil(rpb), 1, 1);
            let (a, b) = (time(&g.h.reference, &name, grid, (32, nw, 1), &args, &d), time(&g.h.oxide, &name, grid, (32, nw, 1), &args, &d));
            println!("bench {name} {rows}x{k}: reference {a:.1} us, oxide {b:.1} us ({:.2}x)", b / a);
        }
    }
}

pub fn run(args: Vec<String>) -> bool {
    let root = format!("{}/titan-engine/oxide-kernels", std::env::var("HOME").unwrap());
    let refp = format!("{root}/reference/candle/quantized.ptx");
    let h = Harness::new(&refp, &format!("{root}/candle-quantized/candle_quantized.ptx"));
    let mut g = G { h, t: Tally::default(), rng: Rng(0x9A17), per_entry: Default::default(), srate: 16, fails: Default::default() };
    if args.iter().any(|a| a == "bench") {
        bench(&mut g);
        return true;
    }
    let all = args.is_empty();
    let want = |f: &str| all || args.iter().any(|a| a == f);
    // Dot-product families run twice: rare special values (outputs mostly finite, so rounding
    // differences show) and frequent ones (nan / inf / denormal propagation).
    if want("dequant") { g.srate = 16; dequantize(&mut g); g.srate = 3; dequantize(&mut g); }
    if want("dmmv") { for r in [1024, 8] { g.srate = r; dmmv(&mut g); } }
    if want("quantize") { quantize(&mut g); }
    if want("mmvq") { for r in [1024, 8] { g.srate = r; mmvq(&mut g); } }
    if want("moe") { for r in [1024, 8] { g.srate = r; moe(&mut g); } }
    if want("mmq") { for r in [1024, 8] { g.srate = r; mmq(&mut g); } }

    // Coverage: every reference entry must exist in the port and have been gated.
    let entries = reference_entries(&refp);
    let mut missing = vec![];
    let mut untested = vec![];
    for e in &entries {
        if !g.h.has(true, e) {
            missing.push(e.clone());
        } else if !g.per_entry.contains_key(e) {
            untested.push(e.clone());
        }
    }
    for (e, n) in &g.fails {
        println!("  failing entry {e}: {n} of {} launches", g.per_entry[e]);
    }
    let ok = g.t.finish("quantized");
    println!("entries: {} in reference, {} ported, {} gated this run", entries.len(), entries.len() - missing.len(), g.per_entry.len());
    if !missing.is_empty() {
        println!("  not ported ({}): {}", missing.len(), missing.join(" "));
    }
    if !untested.is_empty() {
        println!("  ported but not gated this run ({}): {}", untested.len(), untested.join(" "));
    }
    ok && (!all || (missing.is_empty() && untested.is_empty()))
}
