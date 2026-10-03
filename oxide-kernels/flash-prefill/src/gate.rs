//! Gate for the flash-prefill kernels (new kernels: there is no nvcc twin to kdiff against).
//!
//! (1) accuracy: sampled query rows (all heads) against an f64 causal softmax(Q K^T * scale) V over the row's own
//!     keys `< past + i + 1`, with the kernel's bf16 inputs. Bound |out - ref| <= 2^-7 |ref| + 2^-9 (V in [-1, 1];
//!     P is rounded to bf16 as in llama.cpp's fattn-mma, the output to bf16).
//! (2) splits: nsplit 1 vs 2 / 3 / 8 on the same data stay within the same bound of each other.
//! (3) poison: cache rows at and past kv_len are NaN, and query rows past s never read; a read shows as NaN.
//! (4) mutation: shifting `past` by one (a wrong causal limit) must break the accuracy check.
//! (7) head dim 512 and the 256 kernels against the base PTX: gate512.rs.
//! (5) FP_TIME=1: time at the 35B chunk shapes (512 rows, 16 heads, 2 KV heads, 256).
use kdiff::cuda_core::{CudaFunction, CudaStream, DeviceBuffer};
use kdiff::Rng;

const D: usize = 256;
const H: usize = 16;
const KVH: usize = 2;

fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let r = b.wrapping_add(0x7fff + ((b >> 16) & 1));
    (r >> 16) as u16
}
fn bf(x: u16) -> f64 {
    f32::from_bits((x as u32) << 16) as f64
}
fn b8(x: u64) -> [u8; 8] {
    x.to_le_bytes()
}

struct Case {
    s: usize,
    past: usize,
    cap: usize,
    qmul: f32,
}

struct Data {
    q: Vec<u16>, // [head][s][D]
    k: Vec<u16>, // [kvh][cap][D]
    v: Vec<u16>,
}

fn make(r: &mut Rng, c: &Case) -> Data {
    let mut unif = |a: f32| -> u16 {
        let u = (r.next() >> 40) as f32 / (1u64 << 24) as f32;
        bf16_bits((2.0 * u - 1.0) * a)
    };
    let q: Vec<u16> = (0..H * c.s * D).map(|_| unif(2.0 * c.qmul)).collect();
    let l = c.past + c.s;
    let mut k = vec![0x7fc1u16; KVH * c.cap * D];
    let mut v = vec![0x7fc1u16; KVH * c.cap * D];
    for h in 0..KVH {
        for t in 0..l {
            for d in 0..D {
                k[(h * c.cap + t) * D + d] = unif(2.0);
                v[(h * c.cap + t) * D + d] = unif(1.0);
            }
        }
    }
    Data { q, k, v }
}

struct Fns {
    main: CudaFunction,
    comb: CudaFunction,
    /// warps per block of `main` (2 x warps positions per block)
    nw: usize,
}

/// One flash-prefill call; returns bf16 out [head][s][D] and the time per call (us) over `reps`.
#[allow(clippy::too_many_arguments)]
fn call(st: &CudaStream, f: &Fns, c: &Case, dq: &DeviceBuffer<u8>, dk: &DeviceBuffer<u8>, dv: &DeviceBuffer<u8>,
        past: usize, nsplit: usize, reps: usize, win: usize) -> (Vec<u16>, f64) {
    let s = c.s;
    let kv_len = past + s;
    let out = DeviceBuffer::from_host(st, &vec![0xffu8; H * s * D * 2]).unwrap();
    let part = DeviceBuffer::from_host(st, &vec![0xffu8; nsplit.max(2) * H * s * D * 4]).unwrap();
    let ml = DeviceBuffer::from_host(st, &vec![0xffu8; nsplit.max(2) * H * s * 2 * 4]).unwrap();
    let scale = 1.0f32 / 16.0 * std::f32::consts::LOG2_E;
    let launch = || {
        let mut a: Vec<[u8; 8]> = vec![
            b8(dq.cu_deviceptr()), b8(dk.cu_deviceptr()), b8(dv.cu_deviceptr()), b8(out.cu_deviceptr()),
            b8(part.cu_deviceptr()), b8(ml.cu_deviceptr()), b8(scale.to_bits() as u64), b8(s as u64), b8(past as u64),
            b8(kv_len as u64), b8(H as u64), b8(nsplit as u64), b8((s * D) as u64), b8((s * D) as u64),
            b8((c.cap * D) as u64), b8((c.cap * D) as u64), b8(win as u64),
        ];
        let mut p: Vec<*mut std::ffi::c_void> = a.iter_mut().map(|v| v.as_mut_ptr() as *mut std::ffi::c_void).collect();
        unsafe {
            kdiff::cuda_core::simt::launch_kernel_on_stream(&f.main, (s.div_ceil(2 * f.nw) as u32, KVH as u32, nsplit as u32), ((32 * f.nw) as u32, 1, 1), 0, st, &mut p)
                .unwrap()
        };
        if nsplit > 1 {
            let mut b: Vec<[u8; 8]> = vec![
                b8(part.cu_deviceptr()), b8(ml.cu_deviceptr()), b8(out.cu_deviceptr()), b8(s as u64), b8(H as u64),
                b8(nsplit as u64), b8((s * D) as u64),
            ];
            let mut p: Vec<*mut std::ffi::c_void> = b.iter_mut().map(|v| v.as_mut_ptr() as *mut std::ffi::c_void).collect();
            unsafe {
                kdiff::cuda_core::simt::launch_kernel_on_stream(&f.comb, (s as u32, H as u32, 1), (256, 1, 1), 0, st, &mut p).unwrap()
            };
        }
    };
    launch();
    st.synchronize().unwrap();
    let mut us = 0.0;
    if reps > 0 {
        for _ in 0..5 {
            launch();
        }
        st.synchronize().unwrap();
        let t = std::time::Instant::now();
        for _ in 0..reps {
            launch();
        }
        st.synchronize().unwrap();
        us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
    }
    let bytes = out.to_host_vec(st).unwrap();
    (bytes.chunks(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect(), us)
}

/// f64 reference for query position i, all heads: [H][D].
fn reference(c: &Case, dt: &Data, past: usize, i: usize, win: usize) -> Vec<f64> {
    let n_rep = H / KVH;
    let len = past + i + 1;
    let lo = if win > 0 { len.saturating_sub(win) } else { 0 };
    let mut out = vec![0f64; H * D];
    for h in 0..H {
        let kvh = h / n_rep;
        let qv: Vec<f64> = (0..D).map(|d| bf(dt.q[(h * c.s + i) * D + d]) / 16.0).collect();
        let mut sc = vec![0f64; len];
        let mut mx = f64::NEG_INFINITY;
        for (t, x) in sc.iter_mut().enumerate() {
            if t < lo {
                *x = f64::NEG_INFINITY;
                continue;
            }
            let kr = &dt.k[(kvh * c.cap + t) * D..(kvh * c.cap + t + 1) * D];
            *x = (0..D).map(|d| qv[d] * bf(kr[d])).sum();
            mx = mx.max(*x);
        }
        let mut den = 0.0;
        for x in sc.iter_mut() {
            *x = (*x - mx).exp();
            den += *x;
        }
        for (t, p) in sc.iter().enumerate().skip(lo) {
            let vr = &dt.v[(kvh * c.cap + t) * D..(kvh * c.cap + t + 1) * D];
            for d in 0..D {
                out[h * D + d] += p * bf(vr[d]);
            }
        }
        for d in 0..D {
            out[h * D + d] /= den;
        }
    }
    out
}

fn tol(r: f64) -> f64 {
    r.abs() / 128.0 + 1.0 / 512.0
}

pub fn run() -> bool {
    let root = env!("CARGO_MANIFEST_DIR");
    let ctx = kdiff::cuda_core::CudaContext::new(0).expect("cuda context");
    let st = ctx.default_stream();
    let m = ctx.load_module_from_file(&format!("{root}/flash_prefill.ptx")).expect("load flash_prefill.ptx");
    // the variants: (kernel, warps per block); [0] is the default, the last is the previous kernel (v1)
    let variants: Vec<(&str, usize)> =
        vec![("flash_prefill", 4), ("flash_prefill_w8", 8), ("flash_prefill_qg", 4), ("flash_prefill_qg_w8", 8), ("flash_prefill_v1", 4)];
    let fns: Vec<Fns> = variants
        .iter()
        .map(|&(n, nw)| Fns { main: m.load_function(n).unwrap(), comb: m.load_function("flash_prefill_combine").unwrap(), nw })
        .collect();
    for (&(n, _), f) in variants.iter().zip(&fns) {
        println!(
            "{n}: {} registers, {} B local (spills), {} B static shared",
            f.main.num_registers().unwrap_or(0),
            f.main.local_size_bytes().unwrap_or(0),
            f.main.static_shared_memory_bytes().unwrap_or(0)
        );
    }
    let v1 = fns.len() - 1;
    let (mut n_bit, mut n_bit_diff, mut n_v1, mut worst_v1) = (0usize, 0usize, 0usize, 0f64);
    let mut rng = Rng(0xF1A5_4F11);
    let mut ok = true;
    let (mut n_acc, mut worst, mut sum_err) = (0usize, 0f64, 0f64);
    let (mut n_split, mut worst_split) = (0usize, 0f64);
    let (mut n_mut, mut n_mut_caught) = (0usize, 0usize);
    let quick = std::env::var("FP_QUICK").is_ok_and(|v| v == "1");
    // (s, past): first chunks, odd sizes, tails, long contexts
    let mut cases: Vec<(usize, usize)> = vec![(8, 0), (13, 0), (64, 0), (512, 0), (200, 1), (511, 37), (512, 512), (64, 4096), (100, 3000), (512, 4608)];
    if !quick {
        cases.extend([(512, 13000), (233, 27500)]);
    }
    for &(s, past) in &cases {
        let c = Case { s, past, cap: past + s + 40, qmul: if (s + past) % 2 == 0 { 1.0 } else { 3.0 } };
        let dt = make(&mut rng, &c);
        let dq = DeviceBuffer::from_host(&st, &kdiff::as_bytes(&dt.q)).unwrap();
        let dk = DeviceBuffer::from_host(&st, &kdiff::as_bytes(&dt.k)).unwrap();
        let dv = DeviceBuffer::from_host(&st, &kdiff::as_bytes(&dt.v)).unwrap();
        // sampled rows: every row of small chunks, else the first 9, the last 9 and 12 spread ones
        let rows: Vec<usize> = if s <= 64 {
            (0..s).collect()
        } else {
            let mut r: Vec<usize> = (0..9).chain(s - 9..s).chain((1..13).map(|j| j * s / 13)).collect();
            r.sort_unstable();
            r.dedup();
            r
        };
        let refs: Vec<Vec<f64>> = rows.iter().map(|&i| reference(&c, &dt, past, i, 0)).collect();
        let mut outs = Vec::new();
        let (o_v1, _) = call(&st, &fns[v1], &c, &dq, &dk, &dv, past, 1, 0, 0);
        for (vi, f) in fns.iter().enumerate() {
        let mut vouts = Vec::new();
        for &ns in &[1usize, 2, 3, 8] {
            let (o, _) = call(&st, f, &c, &dq, &dk, &dv, past, ns, 0, 0);
            if ns == 1 && vi != v1 {
                // (6) against the previous kernel: bit-identical with one split, else within the tolerance
                for (a, b) in o.iter().zip(o_v1.iter()) {
                    n_bit += 1;
                    n_bit_diff += (a != b) as usize;
                }
            }
            if vi != v1 {
                for (a, b) in o.iter().zip(o_v1.iter()) {
                    let (a, b) = (bf(*a), bf(*b));
                    let e = (a - b).abs() / tol(b);
                    worst_v1 = worst_v1.max(if e.is_nan() { f64::INFINITY } else { e });
                    n_v1 += 1;
                    if !(e <= 1.0) {
                        ok = false;
                    }
                }
            }
            let mut case_worst = 0f64;
            for (ri, &i) in rows.iter().enumerate() {
                for h in 0..H {
                    for d in 0..D {
                        let got = bf(o[(h * s + i) * D + d]);
                        let want = refs[ri][h * D + d];
                        let err = (got - want).abs();
                        let e = err / tol(want);
                        if !(e <= 1.0) {
                            if ok {
                                println!("ACCURACY FAIL s {s} past {past} nsplit {ns} row {i} head {h} dim {d}: got {got} ref {want}");
                            }
                            ok = false;
                        }
                        case_worst = case_worst.max(if e.is_nan() { f64::INFINITY } else { e });
                        sum_err += err;
                        n_acc += 1;
                    }
                }
            }
            worst = worst.max(case_worst);
            println!("  {} s {s:4} past {past:5} nsplit {ns}: worst err/tol {case_worst:.3}", variants[vi].0);
            vouts.push(o);
        }
        outs.extend(vouts);
        }
        for o in &outs[1..] {
            for (a, b) in outs[0].iter().zip(o.iter()) {
                let (a, b) = (bf(*a), bf(*b));
                let e = (a - b).abs() / tol(a);
                worst_split = worst_split.max(if e.is_nan() { f64::INFINITY } else { e });
                n_split += 1;
                if !(e <= 1.0) {
                    ok = false;
                }
            }
        }
        // (4) mutation: the kernel run with past - 1 (keys one short) against the true reference
        if past > 0 {
            let (o, _) = call(&st, &fns[0], &c, &dq, &dk, &dv, past - 1, 1, 0, 0);
            n_mut += 1;
            let mut caught = false;
            for (ri, &i) in rows.iter().enumerate() {
                for j in 0..H * D {
                    let h = j / D;
                    let d = j % D;
                    let got = bf(o[(h * s + i) * D + d]);
                    if !((got - refs[ri][j]).abs() <= tol(refs[ri][j]) / 8.0) {
                        caught = true;
                    }
                }
            }
            n_mut_caught += caught as usize;
        }
    }
    // (7) sliding window: row i sees keys in [past + i + 1 - win, past + i]; all non-v1 variants, splits 1/2/3/8,
    // plus a mutation (win + 1 must break it)
    let (mut n_win, mut worst_win, mut n_wmut, mut n_wmut_caught) = (0usize, 0f64, 0usize, 0usize);
    for &(s, past, win) in &[(512usize, 0usize, 512usize), (700, 0, 512), (512, 4608, 512), (64, 1000, 512), (300, 50, 100), (33, 0, 16), (1000, 0, 40)] {
        let c = Case { s, past, cap: past + s + 40, qmul: 2.0 };
        let dt = make(&mut rng, &c);
        let dq = DeviceBuffer::from_host(&st, &kdiff::as_bytes(&dt.q)).unwrap();
        let dk = DeviceBuffer::from_host(&st, &kdiff::as_bytes(&dt.k)).unwrap();
        let dv = DeviceBuffer::from_host(&st, &kdiff::as_bytes(&dt.v)).unwrap();
        let rows: Vec<usize> = if s <= 64 {
            (0..s).collect()
        } else {
            let mut r: Vec<usize> = (0..9).chain(s - 9..s).chain((1..13).map(|j| j * s / 13)).collect();
            r.sort_unstable();
            r.dedup();
            r
        };
        let refs: Vec<Vec<f64>> = rows.iter().map(|&i| reference(&c, &dt, past, i, win)).collect();
        for (vi, f) in fns.iter().enumerate().filter(|(vi, _)| *vi != v1) {
            for &ns in &[1usize, 2, 3, 8] {
                let (o, _) = call(&st, f, &c, &dq, &dk, &dv, past, ns, 0, win);
                let mut case_worst = 0f64;
                for (ri, &i) in rows.iter().enumerate() {
                    for j in 0..H * D {
                        let (h, d) = (j / D, j % D);
                        let got = bf(o[(h * s + i) * D + d]);
                        let want = refs[ri][j];
                        let e = (got - want).abs() / tol(want);
                        if !(e <= 1.0) {
                            if ok {
                                println!("WINDOW FAIL s {s} past {past} win {win} nsplit {ns} row {i} head {h} dim {d}: got {got} ref {want}");
                            }
                            ok = false;
                        }
                        case_worst = case_worst.max(if e.is_nan() { f64::INFINITY } else { e });
                        n_win += 1;
                    }
                }
                worst_win = worst_win.max(case_worst);
                println!("  {} s {s:4} past {past:5} win {win:3} nsplit {ns}: worst err/tol {case_worst:.3}", variants[vi].0);
            }
        }
        if past + s <= win {
            continue;
        }
        let (o, _) = call(&st, &fns[1], &c, &dq, &dk, &dv, past, 1, 0, win + 1);
        n_wmut += 1;
        let caught = rows.iter().enumerate().any(|(ri, &i)| {
            (0..H * D).any(|j| !((bf(o[((j / D) * s + i) * D + j % D]) - refs[ri][j]).abs() <= tol(refs[ri][j]) / 8.0))
        });
        n_wmut_caught += caught as usize;
    }
    println!("window: {n_win} outputs, worst err/tol {worst_win:.3}; mutation (win + 1): {n_wmut_caught} of {n_wmut} caught");
    if n_wmut_caught < n_wmut {
        println!("WINDOW MUTATION CHECK WEAK");
        ok = false;
    }
    println!("accuracy: {n_acc} outputs, worst err/tol {worst:.3}, mean |err| {:.2e}", sum_err / n_acc.max(1) as f64);
    println!("splits: {n_split} outputs vs nsplit 1, worst {worst_split:.3} of tol");
    println!("mutation (past - 1): {n_mut_caught} of {n_mut} caught");
    println!("vs v1 (nsplit 1): {n_bit_diff} of {n_bit} outputs differ in any bit; all splits: {n_v1} outputs, worst {worst_v1:.3} of tol");
    if n_mut_caught < n_mut {
        println!("MUTATION CHECK WEAK");
        ok = false;
    }
    if std::env::var("FP_TIME").is_ok_and(|v| v == "1") {
        for &(s, past) in &[(512usize, 0usize), (512, 3584), (512, 12800), (512, 27136), (512, 40000), (2048, 26624)] {
            let c = Case { s, past, cap: past + s + 64, qmul: 1.0 };
            let dt = make(&mut rng, &c);
            let dq = DeviceBuffer::from_host(&st, &kdiff::as_bytes(&dt.q)).unwrap();
            let dk = DeviceBuffer::from_host(&st, &kdiff::as_bytes(&dt.k)).unwrap();
            let dv = DeviceBuffer::from_host(&st, &kdiff::as_bytes(&dt.v)).unwrap();
            for (vi, f) in fns.iter().enumerate() {
                for &ns in &[1usize, 2, 4, 8] {
                    let (_, us) = call(&st, f, &c, &dq, &dk, &dv, past, ns, 20, 0);
                    let flop = 4.0 * (H * s * D) as f64 * (past as f64 + s as f64 / 2.0);
                    println!("  time {:20} s {s} past {past:5} nsplit {ns}: {us:8.1} us ({:.1} TFLOP/s)", variants[vi].0, flop / (us * 1e6));
                }
            }
        }
    }
    println!("flash-prefill gate (head dim 256): {}", if ok { "PASS" } else { "FAIL" });
    let ok512 = crate::gate512::run(&ctx, &m);
    let ok = ok && ok512;
    println!("flash-prefill gate: {}", if ok { "PASS" } else { "FAIL" });
    ok
}
