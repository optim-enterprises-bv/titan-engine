//! m4/emb gate (a): candle's row-only `QTensor::embedding` against `dequantize` (whole table) +
//! `index_select`, bit for bit, for every GgmlDType on CPU and CUDA tables; plus every dtype's CPU
//! `dequantize` against llama.cpp's own CPU dequantize_row_* (m4/emb/ref, from cpucheck/ggml_deq.c),
//! and the CUDA dequantize against the same reference (informational). Then timings.
//! usage: embcheck REFDIR
use candle_core::quantized::{GgmlDType, QStorage, QTensor};
use candle_core::{DType, Device, Result, Tensor};
use std::borrow::Cow;
use std::time::Instant;

const ALL: [GgmlDType; 26] = [
    GgmlDType::F32, GgmlDType::F16, GgmlDType::BF16, GgmlDType::Q4_0, GgmlDType::Q4_1, GgmlDType::Q5_0,
    GgmlDType::Q5_1, GgmlDType::Q8_0, GgmlDType::Q8_1, GgmlDType::Q2K, GgmlDType::Q3K, GgmlDType::Q4K,
    GgmlDType::Q5K, GgmlDType::Q6K, GgmlDType::Q8K, GgmlDType::Q1_0, GgmlDType::IQ4NL, GgmlDType::MXFP4,
    GgmlDType::PTQ1_0, GgmlDType::NVFP4, GgmlDType::IQ2XXS, GgmlDType::IQ3XXS, GgmlDType::IQ2S,
    GgmlDType::IQ4XS, GgmlDType::IQ2XS, GgmlDType::IQ3S,
];

fn ggml_id(dt: GgmlDType) -> u32 {
    use GgmlDType::*;
    match dt {
        F32 => 0, F16 => 1, Q4_0 => 2, Q4_1 => 3, Q5_0 => 6, Q5_1 => 7, Q8_0 => 8, Q8_1 => 9, Q2K => 10,
        Q3K => 11, Q4K => 12, Q5K => 13, Q6K => 14, Q8K => 15, IQ2XXS => 16, IQ2XS => 17, IQ3XXS => 18,
        IQ4NL => 20, IQ3S => 21, IQ2S => 22, IQ4XS => 23, BF16 => 30, MXFP4 => 39, NVFP4 => 40, Q1_0 => 41,
        PTQ1_0 => 143,
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| (self.next() >> 56) as u8).collect()
    }
}

fn qtensor(bytes: &[u8], dt: GgmlDType, shape: (usize, usize), dev: &Device) -> Result<QTensor> {
    QTensor::new(QStorage::from_data(Cow::Borrowed(bytes), dev, dt)?, shape)
}

fn bits(t: &Tensor) -> Result<Vec<u32>> {
    Ok(t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?.iter().map(|v| v.to_bits()).collect())
}

fn catch<T>(f: impl FnOnce() -> Result<T>) -> std::result::Result<T, String> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(format!("error: {e}")),
        Err(p) => Err(format!(
            "panic: {}",
            p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
        )),
    }
}

/// returns (values compared, differing) for row-only vs full-then-gather
fn rows_vs_full(dt: GgmlDType, rows: usize, cols: usize, dev: &Device, ids_dev: &Device, seed: u64, shift: u32) -> Result<(usize, usize)> {
    let mut rng = Rng(seed);
    let nbytes = rows * cols / dt.block_size() * dt.type_size();
    let table = rng.bytes(nbytes);
    let qt = qtensor(&table, dt, (rows, cols), dev)?;
    // ids: both ends, a consecutive run, duplicates, random; shape (3, 41)
    let mut ids: Vec<u32> = vec![0, rows as u32 - 1, 5, 6, 7, 8, 9, 9, 9, 0];
    while ids.len() < 123 {
        ids.push((rng.next() % rows as u64) as u32);
    }
    let ids_t = Tensor::from_vec(ids.clone(), (3, 41), ids_dev)?;
    let got = qt.embedding(&ids_t)?;
    assert_eq!(got.dims(), &[3, 41, cols]);
    // shift != 0: negative control, the reference uses different rows
    let ref_ids: Vec<u32> = ids.iter().map(|&i| (i + shift) % rows as u32).collect();
    let ref_ids = Tensor::from_vec(ref_ids, 123, dev)?;
    let want = qt.dequantize(dev)?.index_select(&ref_ids, 0)?.reshape((3, 41, cols))?;
    let (g, w) = (bits(&got)?, bits(&want)?);
    Ok((g.len(), g.iter().zip(&w).filter(|(a, b)| a != b).count()))
}

fn vs_llama(dt: GgmlDType, dir: &str, dev: &Device) -> std::result::Result<String, String> {
    let id = ggml_id(dt);
    let Ok(blocks) = std::fs::read(format!("{dir}/{id}.blocks")) else { return Err("no reference".into()) };
    let want = std::fs::read(format!("{dir}/{id}.f32")).map_err(|e| e.to_string())?;
    let n = want.len() / 4;
    let got = catch(|| bits(&qtensor(&blocks, dt, (1, n), dev)?.dequantize(dev)?))?;
    let (mut bad, mut nan) = (0, 0);
    for (g, w) in got.iter().zip(want.chunks_exact(4)) {
        let w = u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
        if f32::from_bits(*g).is_nan() && f32::from_bits(w).is_nan() {
            nan += 1;
        } else if *g != w {
            bad += 1;
        }
    }
    Ok(format!("{n} values, {nan} NaN in both, {bad} differing -> {}", if bad == 0 { "PASS" } else { "FAIL" }))
}

fn time_embed(qt: &QTensor, dev: &Device, n_ids: usize, iters: usize) -> Result<(f64, f64)> {
    let rows = qt.shape().dims2()?.0 as u64;
    let ids: Vec<u32> = (0..n_ids as u64).map(|i| ((i * 2654435761) % rows) as u32).collect();
    let ids = Tensor::from_vec(ids, (1, n_ids), dev)?;
    let sync = |t: &Tensor| -> Result<()> { t.sum_all()?.to_vec0::<f32>().map(|_| ()) };
    sync(&qt.embedding(&ids)?)?;
    let t0 = Instant::now();
    for _ in 0..iters {
        sync(&qt.embedding(&ids)?)?;
    }
    let row = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
    let flat = ids.flatten_all()?;
    let full = |qt: &QTensor| -> Result<Tensor> { qt.dequantize(dev)?.index_select(&flat, 0) };
    sync(&full(qt)?)?;
    let t0 = Instant::now();
    for _ in 0..iters {
        sync(&full(qt)?)?;
    }
    Ok((row, t0.elapsed().as_secs_f64() * 1e3 / iters as f64))
}

fn main() -> Result<()> {
    std::panic::set_hook(Box::new(|_| {}));
    let dir = std::env::args().nth(1).expect("REFDIR");
    let cuda = Device::new_cuda(0)?;
    let mut fails = 0;

    println!("== CPU dequantize (the routine the CPU row lookup uses) vs llama.cpp dequantize_row_* (1048576 random values each)");
    for dt in &ALL {
        match vs_llama(*dt, &dir, &Device::Cpu) {
            Ok(s) => {
                if s.ends_with("FAIL") { fails += 1; }
                println!("  cpu  {dt:?}: {s}");
            }
            Err(e) => println!("  cpu  {dt:?}: not compared ({e})"),
        }
    }
    println!("== CUDA dequantize vs llama.cpp CPU dequantize (informational: CUDA kernels need not match the CPU reference)");
    for dt in &ALL {
        match vs_llama(*dt, &dir, &cuda) {
            Ok(s) => println!("  cuda {dt:?}: {s}"),
            Err(e) => println!("  cuda {dt:?}: not compared ({e})"),
        }
    }

    println!("== row-only embedding vs dequantize(whole table) + index_select, bit for bit, 123 ids (3 x 41) per case");
    let devs = [("cpu ", Device::Cpu, Device::Cpu), ("cuda", cuda.clone(), cuda.clone()), ("cuda/host-ids", cuda.clone(), Device::Cpu)];
    for (k, dt) in ALL.iter().enumerate() {
        let bs = dt.block_size();
        // a wide row and the narrowest row of whole blocks that is not a multiple of 256 values (ragged groups)
        let mut colss = vec![1024usize];
        let narrow = if bs >= 256 { 512 } else if bs == 1 { 96 } else { bs * 3 };
        colss.push(narrow);
        for (name, dev, ids_dev) in &devs {
            for &cols in &colss {
                let seed = 0x5EED_0000 + (k as u64) * 16 + cols as u64;
                let r = catch(|| rows_vs_full(*dt, 257, cols, dev, ids_dev, seed, 0));
                let ctl = catch(|| rows_vs_full(*dt, 257, cols, dev, ids_dev, seed, 1));
                let line = match (&r, &ctl) {
                    (Ok((n, 0)), Ok((_, c))) if *c > 0 => format!("{n} values, 0 differing (control: {c} differ) -> PASS"),
                    (Ok((n, d)), Ok((_, c))) => { fails += 1; format!("{n} values, {d} differing (control {c}) -> FAIL") }
                    (Err(e), _) => {
                        // a dtype whose whole-table dequantize also fails cannot be compared
                        let full = catch(|| { let mut g = Rng(seed); let b = g.bytes(257 * cols / bs * dt.type_size()); qtensor(&b, *dt, (257, cols), dev)?.dequantize(dev).map(|_| ()) });
                        match full {
                            Err(fe) => format!("not comparable: row-only {e}; whole-table dequantize {fe}"),
                            Ok(()) => { fails += 1; format!("row-only {e} but whole-table dequantize works -> FAIL") }
                        }
                    }
                    (Ok(_), Err(e)) => { fails += 1; format!("control failed: {e} -> FAIL") }
                };
                println!("  {name:13} {dt:?} [257 x {cols}]: {line}");
            }
        }
    }
    // empty ids
    for (name, dev) in [("cpu", Device::Cpu), ("cuda", cuda.clone())] {
        let r = catch(|| {
            let qt = qtensor(&Rng(7).bytes(4 * 1024 / 256 * 210), GgmlDType::Q6K, (4, 1024), &dev)?;
            let e = qt.embedding(&Tensor::zeros((1, 0), DType::U32, &dev)?)?;
            Ok(format!("{:?} {:?}", e.dims(), e.dtype()))
        });
        println!("  {name} empty ids -> {r:?}");
    }

    println!("== timings: row-only vs whole-table dequantize + index_select, ms per call (synchronised)");
    for (label, rows, cols) in [("Spark-X2.5 token_embd Q6_K", 131072usize, 2560usize), ("gemma4-12b token_embd Q6_K", 262144, 3840)] {
        let bytes = Rng(11).bytes(rows * cols / 256 * 210);
        for dev in [cuda.clone(), Device::Cpu] {
            let qt = qtensor(&bytes, GgmlDType::Q6K, (rows, cols), &dev)?;
            for (n, it) in [(1usize, 20usize), (512, 5)] {
                let (row, full) = time_embed(&qt, &dev, n, if dev.is_cpu() { 2 } else { it })?;
                println!("  {label} [{rows} x {cols}] {:4} {n:4} ids: row-only {row:8.3} ms, whole table {full:9.3} ms", if dev.is_cpu() { "cpu" } else { "cuda" });
            }
        }
    }
    println!("embcheck: {} -> {}", if fails == 0 { "all comparisons equal" } else { "FAILURES" }, if fails == 0 { "PASS" } else { "FAIL" });
    std::process::exit(if fails == 0 { 0 } else { 1 });
}
