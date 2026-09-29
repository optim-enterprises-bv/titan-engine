//! Kernel-level gate: every `mmvq_gguf_<q>_<dst>_plain_rows` entry against the nvcc kernel the
//! M6 WIP launcher ran (`mmvq_gguf_<q>_<dst>_plain_cuda1` of mmvq_gguf_rows.cubin, which offsets y
//! and dst by blockIdx.y), same grid (ceil(nrows / 2), b, 1) and block (32, 8, 1), identical
//! argument bytes and buffers; every output byte is compared, including the untouched gaps
//! between the strided output columns (dst starts as random bytes).
//!
//! Env: ROWS_REF=<cubin> overrides the reference (mutation check: the original mmvq_gguf.cubin,
//! whose cuda1 kernels ignore blockIdx.y, must FAIL for b > 1); ROWS_TIME=1 also times both sides.
use kdiff::{Arg, Harness, Rng, Tally};

/// (name, qk, block bytes, byte offsets of f16 scale fields in a block)
const FMTS: [(&str, usize, usize, &[usize]); 10] = [
    ("q4_0", 32, 18, &[0]),
    ("q4_1", 32, 20, &[0, 2]),
    ("q5_0", 32, 22, &[0]),
    ("q5_1", 32, 24, &[0, 2]),
    ("q8_0", 32, 34, &[0]),
    ("q2_k", 256, 84, &[80, 82]),
    ("q3_k", 256, 110, &[108]),
    ("q4_k", 256, 144, &[0, 2]),
    ("q5_k", 256, 176, &[0, 2]),
    ("q6_k", 256, 210, &[208]),
];
const DSTS: [(&str, usize); 3] = [("bf16", 2), ("f16", 2), ("f32", 4)];

/// A random f16 of moderate magnitude (2^-10 .. 2^-3), or (1 in 64) any bit pattern.
fn f16_scale(r: &mut Rng) -> u16 {
    let v = r.next();
    if v % 64 == 0 {
        return (v >> 8) as u16;
    }
    let sign = ((v >> 8) & 1) as u16;
    let exp = 5 + ((v >> 9) % 7) as u16;
    let man = ((v >> 16) & 0x3ff) as u16;
    (sign << 15) | (exp << 10) | man
}

fn weights(r: &mut Rng, f: &(&str, usize, usize, &[usize]), blocks: usize) -> Vec<u8> {
    let mut w = r.bytes(blocks * f.2);
    for b in 0..blocks {
        for &o in f.3 {
            let s = f16_scale(r).to_le_bytes();
            w[b * f.2 + o] = s[0];
            w[b * f.2 + o + 1] = s[1];
        }
    }
    w
}

/// Q8_1 blocks: (d, s) f16 then 32 random int8.
fn q8_1(r: &mut Rng, blocks: usize) -> Vec<u8> {
    let mut y = r.bytes(blocks * 36);
    for b in 0..blocks {
        for o in [0, 2] {
            let s = f16_scale(r).to_le_bytes();
            y[b * 36 + o] = s[0];
            y[b * 36 + o + 1] = s[1];
        }
    }
    y
}

/// Launch `f` on fresh device copies of `bufs` with the 7 mmvq args, return buffer 2 (dst).
/// `col` offsets y by `col * scy` blocks and dst by `col * scd` elements (one batch-1 launch per column).
#[allow(clippy::too_many_arguments)]
fn launch(h: &Harness, f: &kdiff::cuda_core::CudaFunction, grid: (u32, u32, u32), bufs: &[Vec<u8>], ncols: usize, nrows: usize,
          scy: usize, scd: usize, es: usize, cols: &[usize]) -> Vec<u8> {
    use kdiff::cuda_core::DeviceBuffer;
    let dev: Vec<DeviceBuffer<u8>> = bufs.iter().map(|b| DeviceBuffer::from_host(&h.stream, b).unwrap()).collect();
    for &c in cols {
        let mut vals: Vec<[u8; 8]> = vec![
            dev[0].cu_deviceptr().to_le_bytes(),
            (dev[1].cu_deviceptr() + (c * scy * 36) as u64).to_le_bytes(),
            (dev[2].cu_deviceptr() + (c * scd * es) as u64).to_le_bytes(),
            (ncols as u64).to_le_bytes(),
            (nrows as u64).to_le_bytes(),
            (scy as u64).to_le_bytes(),
            (scd as u64).to_le_bytes(),
        ];
        let mut ptrs: Vec<*mut std::ffi::c_void> = vals.iter_mut().map(|v| v.as_mut_ptr() as *mut std::ffi::c_void).collect();
        unsafe { kdiff::cuda_core::simt::launch_kernel_on_stream(f, grid, (32, 8, 1), 0, &h.stream, &mut ptrs).unwrap() };
    }
    h.stream.synchronize().unwrap();
    dev[2].to_host_vec(&h.stream).unwrap()
}

fn compare(t: &mut Tally, label: &str, a: &[u8], b: &[u8]) {
    let mut d = kdiff::Diff { bytes: a.len(), differing: 0, first: None };
    for i in 0..a.len() {
        if a[i] != b[i] {
            d.differing += 1;
            if d.first.is_none() {
                d.first = Some((2, i, a[i], b[i]));
            }
        }
    }
    t.record(label, &d);
}

pub fn run() -> bool {
    let root = format!("{}/titan-engine/oxide-kernels", std::env::var("HOME").unwrap());
    // (a) the M6 WIP nvcc launcher's kernel (cuda1 + blockIdx.y offsets, grid.y = b)
    let rows_ref = std::env::var("ROWS_REF").unwrap_or(format!("{root}/reference/mistralrs-quant-rows/mmvq_gguf_rows.cubin"));
    let h = Harness::from_files(&rows_ref, &format!("{root}/mmvq-rows/mmvq_rows.ptx"));
    // (b) the unmodified nvcc batch-1 kernel decode runs, one launch per column
    let plain_ref = h.ctx.load_module_from_file(&format!("{root}/reference/mistralrs-quant/mmvq_gguf.cubin")).unwrap();
    let mut t = Tally::default();
    let mut r = Rng(0x0A7E_5EED);
    let mut finite = (0usize, 0usize);
    for f in FMTS.iter() {
        for (dn, es) in DSTS.iter() {
            let ref_name = format!("mmvq_gguf_{}_{dn}_plain_cuda1", f.0);
            let fa = h.reference.load_function(&ref_name).unwrap();
            let fb = plain_ref.load_function(&ref_name).unwrap();
            // (ncols, nrows, extra y blocks per column, extra dst elements per column)
            let shapes: [(usize, usize, usize, usize); 4] = [
                (f.1 * 9, 45, 2, 5),
                (f.1 * 3, 1, 0, 0),
                (2048, 257, 0, 3),
                (2048, 512, 0, 0),
            ];
            for (si, &(ncols, nrows, gy, gd)) in shapes.iter().enumerate() {
                for b in 1..=8usize {
                    if si == 1 && b > 3 {
                        continue;
                    }
                    let ox_name = format!("mmvq_gguf_{}_{dn}_rows{b}", f.0);
                    let fo = h.oxide.load_function(&ox_name).unwrap();
                    let bpr = ncols / f.1;
                    let scy = ncols / 32 + gy;
                    let scd = nrows + gd;
                    let bufs = [
                        weights(&mut r, f, nrows * bpr),
                        q8_1(&mut r, (b - 1) * scy + ncols / 32),
                        r.bytes(((b - 1) * scd + nrows) * es),
                    ];
                    let nb = (nrows as u32).div_ceil(2);
                    let label = format!("{ox_name} ncols {ncols} nrows {nrows}");
                    let ox = launch(&h, &fo, (nb, 1, 1), &bufs, ncols, nrows, scy, scd, *es, &[0]);
                    let ra = launch(&h, &fa, (nb, b as u32, 1), &bufs, ncols, nrows, scy, scd, *es, &[0]);
                    let cols: Vec<usize> = (0..b).collect();
                    let rb = launch(&h, &fb, (nb, 1, 1), &bufs, ncols, nrows, scy, scd, *es, &cols);
                    compare(&mut t, &format!("{label} vs nvcc rows launcher"), &ra, &ox);
                    compare(&mut t, &format!("{label} vs nvcc batch-1 per column"), &rb, &ox);
                    if *dn == "f32" {
                        for c in 0..b {
                            for i in 0..nrows {
                                let o = (c * scd + i) * 4;
                                let v = f32::from_le_bytes(ox[o..o + 4].try_into().unwrap());
                                finite.1 += 1;
                                finite.0 += v.is_finite() as usize;
                            }
                        }
                    }
                }
            }
        }
    }
    println!("f32 outputs: {} of {} finite", finite.0, finite.1);
    if std::env::var("ROWS_TIME").is_ok_and(|v| v == "1") {
        time(&h, &plain_ref, &mut r);
    }
    t.finish("mmvq-rows kernel-level kdiff vs nvcc") && finite.0 * 10 > finite.1 * 9
}

/// Wall time (200 repetitions, stream-synchronised) on the shapes of Qwen3.6-35B-A3B verification
/// (hidden 2048): b nvcc batch-1 launches (one per row) vs one oxide rows<b> launch.
fn time(h: &Harness, plain_ref: &std::sync::Arc<kdiff::cuda_core::CudaModule>, r: &mut Rng) {
    use kdiff::cuda_core::DeviceBuffer;
    for (fi, ncols, nrows) in [(7usize, 2048usize, 4096usize), (9, 2048, 4096), (4, 2048, 512), (4, 4096, 2048), (9, 2048, 248320)] {
        let f = &FMTS[fi];
        for b in [1usize, 2, 3, 5] {
            let bufs: Vec<DeviceBuffer<u8>> = [weights(r, f, nrows * ncols / f.1), q8_1(r, b * ncols / 32), vec![0u8; b * nrows * 4]]
                .iter()
                .map(|v| DeviceBuffer::from_host(&h.stream, v).unwrap())
                .collect();
            let mut res = Vec::new();
            for side in 0..2 {
                let func = if side == 0 {
                    plain_ref.load_function(&format!("mmvq_gguf_{}_f32_plain_cuda1", f.0)).unwrap()
                } else {
                    h.oxide.load_function(&format!("mmvq_gguf_{}_f32_rows{b}", f.0)).unwrap()
                };
                let cols = if side == 0 { b } else { 1 };
                let grid = ((nrows as u32).div_ceil(2), 1, 1);
                let run = || {
                    for c in 0..cols {
                        let mut vals: Vec<[u8; 8]> = vec![
                            bufs[0].cu_deviceptr().to_le_bytes(),
                            (bufs[1].cu_deviceptr() + (c * ncols / 32 * 36) as u64).to_le_bytes(),
                            (bufs[2].cu_deviceptr() + (c * nrows * 4) as u64).to_le_bytes(),
                            (ncols as u64).to_le_bytes(),
                            (nrows as u64).to_le_bytes(),
                            ((ncols / 32) as u64).to_le_bytes(),
                            (nrows as u64).to_le_bytes(),
                        ];
                        let mut ptrs: Vec<*mut std::ffi::c_void> = vals.iter_mut().map(|v| v.as_mut_ptr() as *mut std::ffi::c_void).collect();
                        unsafe { kdiff::cuda_core::simt::launch_kernel_on_stream(&func, grid, (32, 8, 1), 0, &h.stream, &mut ptrs).unwrap() };
                    }
                };
                for _ in 0..20 {
                    run();
                }
                h.stream.synchronize().unwrap();
                let t0 = std::time::Instant::now();
                for _ in 0..200 {
                    run();
                }
                h.stream.synchronize().unwrap();
                res.push(t0.elapsed().as_secs_f64() * 1e6 / 200.0);
            }
            println!("  time {} f32 {ncols}x{nrows} b {b}: nvcc batch-1 x{b} {:.2} us, oxide rows{b} {:.2} us ({:.2}x)", f.0, res[0], res[1], res[1] / res[0]);
        }
    }
}
