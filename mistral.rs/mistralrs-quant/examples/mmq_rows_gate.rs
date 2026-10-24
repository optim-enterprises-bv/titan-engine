//! Gate for `mmq_rows`: every output of the MMQ-exact kernels against the MMQ launcher on the same quantized
//! activations, bit for bit, over the model's matmul shapes, 1..=8 columns and BF16 / F32 outputs. Then the
//! per-launch time of both by column count. `cargo run --release --example mmq_rows_gate --features <cuda|oxide>`
//! (`MMQ_ROWS_GATE=bench` skips the gate, `=gate` the timing).
use candle_core::{quantized::GgmlDType, quantized::QTensor, DType, Device, Result, Tensor};
use mistralrs_quant::{mmq_plain, mmq_rows};

/// (rows, k) of the model's matmuls, plus a partial last tile and a one-block-per-tile (grid = tiles) launch.
const SHAPES: [(usize, usize); 9] = [
    (1024, 5120),
    (6144, 5120),
    (5120, 6144),
    (10240, 5120),
    (12288, 5120),
    (17408, 5120),
    (5120, 17408),
    (1000, 5120),
    (15360, 512),
];
/// The model's shapes the timing covers (the first `TIMED` of `SHAPES`).
const TIMED: usize = 7;
const FORMATS: [GgmlDType; 2] = [GgmlDType::Q4K, GgmlDType::Q5K];
const ITERS: usize = 50;

fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
        })
        .collect()
}

/// Activations whose second half nearly cancels the first, so accumulation-order slips show.
fn rows(b: usize, k: usize, seed: u64, dev: &Device) -> Result<Tensor> {
    let mut v = noise(b * k, seed);
    for r in 0..b {
        for i in k / 2..k {
            v[r * k + i] = -v[r * k + i - k / 2] * 0.999;
        }
    }
    Tensor::from_vec(v, (b, k), &Device::Cpu)?.to_device(dev)
}

fn bits(t: &Tensor) -> Result<Vec<u32>> {
    Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?.into_iter().map(f32::to_bits).collect())
}

fn gate(dev: &Device) -> Result<bool> {
    let (mut checked, mut failed) = (0usize, 0usize);
    for (si, &(n, k)) in SHAPES.iter().enumerate() {
        let w32 = Tensor::from_vec(noise(n * k, si as u64 + 1), (n, k), &Device::Cpu)?;
        for dtype in FORMATS {
            let w = QTensor::quantize_onto(&w32, dtype, dev)?;
            for b in 1..=mmq_rows::MAX_COLS {
                for xt in [DType::BF16, DType::F32] {
                    let x = rows(b, k, (si * 31 + b) as u64, dev)?.to_dtype(xt)?;
                    mmq_rows::set_max_cols(0);
                    let want = bits(&mmq_plain(&w, &x)?)?;
                    mmq_rows::set_max_cols(mmq_rows::MAX_COLS);
                    let got = bits(&mmq_plain(&w, &x)?)?;
                    let bad = want.iter().zip(&got).filter(|(a, b)| a != b).count();
                    if bad > 0 {
                        failed += 1;
                        let i = want.iter().zip(&got).position(|(a, b)| a != b).unwrap();
                        println!("FAIL {dtype:?} {n}x{k} cols {b} {xt:?}: {bad} of {} differ, first [{i}] {:08x} vs {:08x}",
                                 want.len(), want[i], got[i]);
                    }
                    checked += want.len();
                }
            }
        }
    }
    mmq_rows::set_max_cols(usize::MAX);
    println!("mmq_rows gate: {checked} outputs, {failed} failing cases -> {}", if failed == 0 { "PASS" } else { "FAIL" });
    Ok(failed == 0)
}

/// Bytes of distinct weight copies each timing cycles through, to keep the 64 MiB L2 cold as decoding does.
const COLD_BYTES: usize = 160 << 20;

fn time_us(dev: &Device, ws: &[QTensor], x: &Tensor, f: &dyn Fn(&QTensor, &Tensor) -> Result<Tensor>) -> Result<f64> {
    f(&ws[0], x)?;
    dev.synchronize()?;
    let t = std::time::Instant::now();
    for i in 0..ITERS {
        f(&ws[i % ws.len()], x)?;
    }
    dev.synchronize()?;
    Ok(t.elapsed().as_secs_f64() * 1e6 / ITERS as f64)
}

fn bench(dev: &Device) -> Result<()> {
    for &(n, k) in &SHAPES[..TIMED] {
        let w32 = Tensor::from_vec(noise(n * k, 7), (n, k), &Device::Cpu)?;
        for dtype in FORMATS {
            let w = QTensor::quantize_onto(&w32, dtype, dev)?;
            let copies = COLD_BYTES.div_ceil(w.storage_size_in_bytes()).max(2);
            let data = w.data()?;
            let ws = (0..copies)
                .map(|_| QTensor::new(candle_core::quantized::QStorage::from_data(data.clone(), dev, dtype)?, (n, k)))
                .collect::<Result<Vec<_>>>()?;
            let mut line = format!("{dtype:?} {n}x{k}:");
            for b in 1..=mmq_rows::MAX_COLS {
                let x = rows(b, k, 3, dev)?.to_dtype(DType::BF16)?;
                mmq_rows::set_max_cols(0);
                let mmq = time_us(dev, &ws, &x, &|w, x| mmq_plain(w, x))?;
                mmq_rows::set_max_cols(mmq_rows::MAX_COLS);
                let exact = time_us(dev, &ws, &x, &|w, x| mmq_plain(w, x))?;
                let rows_us = time_us(dev, &ws, &x, &|w, x| mistralrs_quant::mmvq_rows::plain_rows(w, x))?;
                line += &format!(" {b}:{mmq:.0}/{exact:.0}/{rows_us:.0}");
            }
            println!("{line}  (us MMQ/exact/rows, L2 cold)");
        }
    }
    mmq_rows::set_max_cols(usize::MAX);
    Ok(())
}

fn main() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let mode = std::env::var("MMQ_ROWS_GATE").unwrap_or_default();
    let ok = mode == "bench" || gate(&dev)?;
    if mode != "gate" {
        bench(&dev)?;
    }
    std::process::exit(if ok { 0 } else { 1 });
}
