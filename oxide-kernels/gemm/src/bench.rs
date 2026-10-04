//! GEMM_BENCH=<shapes file>: per inventoried shape, the planned oxide kernel vs cuBLAS (cublasGemmStridedBatchedEx
//! with 32F compute: candle's call; cuBLASLt calls are timed through the same entry, without their bias epilogue),
//! best of 3 rounds of `reps` back-to-back calls each (wall clock / reps, synchronised). GEMV shapes also time the
//! other GEMV variant (warp per column vs block per column) to tune the plan's rule.
//!
//! Shapes file: `model count dt batch m n k a_s0 a_s1 b_s0 b_s1 sa sb alpha beta has_bias tag...` (m4/noblas/mkshapes.py).
use crate::gate::{Cublas, H, Rng};
use crate::plan::{self, Dt, Force, Problem};
use kdiff::cuda_core::DeviceBuffer;
use std::collections::BTreeMap;

fn time<F: FnMut()>(h: &H, mut f: F) -> f64 {
    f();
    h.st.synchronize().unwrap();
    // reps for ~5 ms per round
    let t = std::time::Instant::now();
    f();
    h.st.synchronize().unwrap();
    let one = t.elapsed().as_secs_f64();
    let reps = ((5e-3 / one.max(1e-6)) as usize).clamp(3, 2000);
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t = std::time::Instant::now();
        for _ in 0..reps {
            f();
        }
        h.st.synchronize().unwrap();
        best = best.min(t.elapsed().as_secs_f64() / reps as f64);
    }
    best * 1e6
}

pub fn bench(h: &H, path: &str) -> bool {
    let st = &h.st;
    let cub = Cublas::new(st);
    if cub.is_none() {
        println!("cuBLAS not loadable: timing the oxide kernels only");
    }
    let mut r = Rng(0xBE4C);
    // per model: (sum count x ours, sum count x cublas)
    let mut tot: BTreeMap<String, (f64, f64, usize)> = BTreeMap::new();
    let mut worst: Vec<(f64, String)> = Vec::new();
    println!("{:<20} {:>6} {:<58} {:<26} {:>9} {:>9} {:>6} {:>9}", "model", "count", "shape", "kernel", "oxide_us", "cublas_us", "ratio", "alt_us");
    for line in std::fs::read_to_string(path).unwrap().lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 16 || f[0].starts_with('#') {
            continue;
        }
        let model = f[0].to_string();
        let count: usize = f[1].parse().unwrap();
        let dt = match f[2] {
            "f32" => Dt::F32,
            "f16" => Dt::F16,
            _ => Dt::Bf16,
        };
        let v: Vec<i64> = f[3..13].iter().map(|x| x.parse().unwrap()).collect();
        let (batch, m, n, k, a_s0, a_s1, b_s0, b_s1, sa, sb) = (v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9]);
        let alpha: f32 = f[13].parse().unwrap();
        let beta: f32 = f[14].parse().unwrap();
        let has_bias = f[15] == "1";
        let tag = f[16..].join(" ");
        let es = dt.size() as usize;
        let ext = |s0: i64, d0: i64, s1: i64, d1: i64, sz: i64| ((batch - 1) * sz + (d0 - 1) * s0 + (d1 - 1) * s1 + 1) as usize;
        let a_len = ext(a_s0, m, a_s1, k, sa);
        let b_len = ext(b_s0, k, b_s1, n, sb);
        let d_len = (batch * m * n) as usize;
        let fill = |r: &mut Rng, len: usize| -> Vec<u8> {
            let mut out = Vec::with_capacity(len * es);
            for _ in 0..len {
                let x = r.unif();
                match dt {
                    Dt::F32 => out.extend_from_slice(&x.to_bits().to_le_bytes()),
                    Dt::F16 => out.extend_from_slice(&crate::gate::f2h(x).to_le_bytes()),
                    Dt::Bf16 => out.extend_from_slice(&crate::gate::f2bf(x).to_le_bytes()),
                }
            }
            out
        };
        let da = DeviceBuffer::from_host(st, &fill(&mut r, a_len)).unwrap();
        let db = DeviceBuffer::from_host(st, &fill(&mut r, b_len)).unwrap();
        let dd = DeviceBuffer::from_host(st, &fill(&mut r, d_len)).unwrap();
        let dbias = DeviceBuffer::from_host(st, &fill(&mut r, n as usize)).unwrap();
        let p = Problem {
            dt,
            batch,
            m,
            n,
            k,
            a_s0,
            a_s1,
            b_s0,
            b_s1,
            d_s0: n,
            d_s1: 1,
            c_s0: n,
            c_s1: 1,
            bias_s0: 0,
            bias_s1: 1,
            sa,
            sb,
            sd: m * n,
            sc: m * n,
            alpha,
            beta,
            has_c: beta != 0.0,
            has_bias,
            a_addr: da.cu_deviceptr(),
            b_addr: db.cu_deviceptr(),
            d_addr: dd.cu_deviceptr(),
        };
        let l = match plan::plan(&p, h.sms, Force::default()) {
            Ok(l) => l,
            Err(e) => {
                println!("{model:<20} {count:>6} {tag:<58} plan error {e}");
                continue;
            }
        };
        let ws = DeviceBuffer::<u8>::from_host(st, &vec![0u8; l.ws_floats.max(1) * 4]).unwrap();
        let c = if p.has_c { p.d_addr } else { 0 };
        let bias = if has_bias { dbias.cu_deviceptr() } else { 0 };
        let ours = time(h, || h.run(&l, c, bias, ws.cu_deviceptr()));
        let alt = if l.kernel.starts_with("gemv_k") {
            let fam = if l.kernel.starts_with("gemv_ks") { 1 } else { 2 };
            let la = plan::plan(&p, h.sms, Force { family: fam, ..Default::default() }).unwrap();
            format!("{:.1}", time(h, || h.run(&la, c, bias, ws.cu_deviceptr())))
        } else {
            "-".to_string()
        };
        let cb = cub.as_ref().and_then(|cb| {
            let s = cb.gemm(&p, p.has_c)?;
            if s != 0 {
                return None;
            }
            Some(time(h, || {
                cb.gemm(&p, p.has_c);
            }))
        });
        let (cbs, ratio) = match cb {
            Some(t) => (format!("{t:.1}"), format!("{:.2}", ours / t)),
            None => ("-".into(), "-".into()),
        };
        let shape = format!("{} b{batch} {m}x{n}x{k} a({a_s0},{a_s1}) b({b_s0},{b_s1}) s({sa},{sb}){}", dt.name(), if has_bias { " bias" } else { "" });
        println!("{model:<20} {count:>6} {shape:<58} {:<26} {ours:>9.1} {cbs:>9} {ratio:>6} {alt:>9}  {tag}", format!("{}{}", l.kernel, if l.nsplit > 1 { format!("/s{}", l.nsplit) } else { String::new() }));
        let e = tot.entry(model.clone()).or_insert((0.0, 0.0, 0));
        e.0 += count as f64 * ours;
        if let Some(t) = cb {
            e.1 += count as f64 * t;
            worst.push((ours / t, format!("{model} {shape} x{count}: {ours:.1} vs {t:.1} us")));
        } else {
            e.2 += 1;
        }
    }
    println!("\nper model (count-weighted, inventory runs): oxide ms vs cuBLAS ms");
    for (m, (o, c, nc)) in &tot {
        println!("  {m:<22} oxide {:>9.3} ms  cuBLAS {:>9.3} ms  ratio {:.3}{}", o / 1e3, c / 1e3, if *c > 0.0 { o / c } else { 0.0 }, if *nc > 0 { format!(" ({nc} shapes without a cuBLAS time)") } else { String::new() });
    }
    worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!("\nslowest relative to cuBLAS:");
    for (rt, s) in worst.iter().take(25) {
        println!("  {rt:.2}x  {s}");
    }
    true
}
