//! Gate for the flash-decode kernels (there is no nvcc twin to kdiff against: the kernel is new).
//!
//! (1) accuracy: every output against an f64 softmax(Q K^T * scale) V over the row's own keys, with
//!     the kernel's own bf16 inputs; bound |out - ref| <= 2^-8 |ref| + 2^-12 max|V| (bf16 output
//!     rounding is 2^-9 relative).
//! (2) row exactness, bit for bit: row r of an n-row launch (the MTP verify rows) against a 1-row
//!     launch of that row alone with kv_total = its own key count (the decode step), across chunk
//!     sizes and G instances. Keys past kv_total are NaN in the cache, so any read past the end
//!     poisons the output.
//! (3) mutation checks: the same comparison against the neighbouring key count, and across chunk
//!     sizes, must find differences (the bit-exact check is not vacuous).
//! (4) FD_TIME=1: time split + combine at the 35B shapes (16 heads, 2 KV heads, 256).
use kdiff::cuda_core::{CudaFunction, CudaStream, DeviceBuffer};
use kdiff::Rng;

const D: usize = 256;

fn bf16_bits(x: f32) -> u16 {
    // round to nearest even
    let b = x.to_bits();
    let r = b.wrapping_add(0x7fff + ((b >> 16) & 1));
    (r >> 16) as u16
}
fn bf(x: u16) -> f64 {
    f32::from_bits((x as u32) << 16) as f64
}

struct Case {
    n_heads: usize,
    n_kv: usize,
    l: usize,     // kv_total
    rows: usize,  // query rows; row r sees l - rows + r + 1 keys
    cap: usize,   // cache capacity (>= l); keys l..cap are NaN
    qmul: f32,    // Q magnitude (peakiness of the softmax)
}

struct Data {
    q: Vec<u16>,  // [head][row][D]
    k: Vec<u16>,  // [kvh][cap][D]
    v: Vec<u16>,
}

fn make(r: &mut Rng, c: &Case) -> Data {
    let mut unif = |a: f32| -> u16 {
        let u = (r.next() >> 40) as f32 / (1u64 << 24) as f32;
        bf16_bits((2.0 * u - 1.0) * a)
    };
    let q: Vec<u16> = (0..c.n_heads * c.rows * D).map(|_| unif(2.0 * c.qmul)).collect();
    let mut k = vec![0x7fc1u16; c.n_kv * c.cap * D];
    let mut v = vec![0x7fc1u16; c.n_kv * c.cap * D];
    for h in 0..c.n_kv {
        for t in 0..c.l {
            for d in 0..D {
                k[(h * c.cap + t) * D + d] = unif(2.0);
                v[(h * c.cap + t) * D + d] = unif(1.0);
            }
        }
    }
    Data { q, k, v }
}

fn b8(x: u64) -> [u8; 8] {
    x.to_le_bytes()
}

struct Fns {
    g8: CudaFunction,
    g16: CudaFunction,
    g24: CudaFunction,
    comb: CudaFunction,
}

/// One flash-decode call: split + combine for `rows` rows starting at query row `r0` of `dq`,
/// kv_total `l`, writing bf16 out [rows][n_heads][D]. Returns the host copy of out.
#[allow(clippy::too_many_arguments)]
fn call(st: &CudaStream, f: &Fns, c: &Case, dq: &DeviceBuffer<u8>, dk: &DeviceBuffer<u8>, dv: &DeviceBuffer<u8>,
       r0: usize, rows: usize, l: usize, chunk: usize, reps: usize) -> (Vec<u16>, f64) {
    let n_rep = c.n_heads / c.n_kv;
    let g = n_rep * rows;
    let func = if g <= 8 { &f.g8 } else if g <= 16 { &f.g16 } else { &f.g24 };
    assert!(g <= 24);
    let nchunks = l.div_ceil(chunk);
    let nq = rows * c.n_heads;
    let part = DeviceBuffer::from_host(st, &vec![0xffu8; nq * nchunks * D * 4]).unwrap();
    let meta = DeviceBuffer::from_host(st, &vec![0xffu8; nq * nchunks * 2 * 4]).unwrap();
    let out = DeviceBuffer::from_host(st, &vec![0xffu8; nq * D * 2]).unwrap();
    let qp = dq.cu_deviceptr() + (r0 * D * 2) as u64;
    let scale = 1.0f32 / 16.0;
    let launch = || {
        let mut a: Vec<[u8; 8]> = vec![
            b8(qp), b8(dk.cu_deviceptr()), b8(dv.cu_deviceptr()), b8(part.cu_deviceptr()), b8(meta.cu_deviceptr()),
            b8(scale.to_bits() as u64), b8(n_rep as u64), b8(c.n_heads as u64), b8(l as u64), b8(rows as u64),
            b8(chunk as u64), b8(nchunks as u64), b8((c.rows * D) as u64), b8(D as u64),
            b8((c.cap * D) as u64), b8((c.cap * D) as u64),
        ];
        let mut p: Vec<*mut std::ffi::c_void> = a.iter_mut().map(|v| v.as_mut_ptr() as *mut std::ffi::c_void).collect();
        unsafe { kdiff::cuda_core::simt::launch_kernel_on_stream(func, (nchunks as u32, c.n_kv as u32, 1), (256, 1, 1), 0, st, &mut p).unwrap() };
        let mut b: Vec<[u8; 8]> = vec![
            b8(part.cu_deviceptr()), b8(meta.cu_deviceptr()), b8(out.cu_deviceptr()), b8(c.n_heads as u64), b8(rows as u64),
            b8(l as u64), b8(chunk as u64), b8(nchunks as u64), b8((c.n_heads * D) as u64), b8(D as u64),
        ];
        let mut p: Vec<*mut std::ffi::c_void> = b.iter_mut().map(|v| v.as_mut_ptr() as *mut std::ffi::c_void).collect();
        unsafe { kdiff::cuda_core::simt::launch_kernel_on_stream(&f.comb, (c.n_heads as u32, rows as u32, 1), (256, 1, 1), 0, st, &mut p).unwrap() };
    };
    launch();
    st.synchronize().unwrap();
    let mut us = 0.0;
    if reps > 0 {
        for _ in 0..10 {
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

/// f64 reference for query row r (0-based within the case's rows), all heads: [n_heads][D].
fn reference(c: &Case, dt: &Data, r: usize) -> Vec<f64> {
    let n_rep = c.n_heads / c.n_kv;
    let len = c.l - c.rows + r + 1;
    let mut out = vec![0f64; c.n_heads * D];
    for h in 0..c.n_heads {
        let kvh = h / n_rep;
        let qv: Vec<f64> = (0..D).map(|d| bf(dt.q[(h * c.rows + r) * D + d]) / 16.0).collect();
        let mut s = vec![0f64; len];
        let mut mx = f64::NEG_INFINITY;
        for (t, st) in s.iter_mut().enumerate() {
            let kr = &dt.k[(kvh * c.cap + t) * D..(kvh * c.cap + t + 1) * D];
            *st = (0..D).map(|d| qv[d] * bf(kr[d])).sum();
            mx = mx.max(*st);
        }
        let mut den = 0.0;
        for st in s.iter_mut() {
            *st = (*st - mx).exp();
            den += *st;
        }
        for (t, p) in s.iter().enumerate() {
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

pub fn run() -> bool {
    let root = format!("{}/titan-engine/oxide-kernels/flash-decode", std::env::var("HOME").unwrap());
    let ctx = kdiff::cuda_core::CudaContext::new(0).expect("cuda context");
    let st = ctx.default_stream();
    let m = ctx.load_module_from_file(&format!("{root}/flash_decode.ptx")).expect("load flash_decode.ptx");
    let f = Fns {
        g8: m.load_function("flash_decode_split_g8").unwrap(),
        g16: m.load_function("flash_decode_split_g16").unwrap(),
        g24: m.load_function("flash_decode_split_g24").unwrap(),
        comb: m.load_function("flash_decode_combine").unwrap(),
    };
    let mut rng = Rng(0xF1A5_4DEC);
    let mut ok = true;
    let (mut n_acc, mut worst, mut n_exact, mut n_exact_fail, mut n_mut, mut n_mut_caught) = (0usize, 0f64, 0usize, 0usize, 0usize, 0usize);
    let quick = std::env::var("FD_QUICK").is_ok_and(|v| v == "1");
    let mut lens: Vec<usize> = vec![1, 2, 3, 4, 127, 128, 129, 130, 255, 256, 257, 511, 513, 1000, 4096, 4099];
    if !quick {
        lens.extend([13000, 28001]);
    }
    for &(n_heads, n_kv) in &[(16usize, 2usize), (16, 4), (8, 1)] {
        for &l in &lens {
            let n_rep = n_heads / n_kv;
            let max_rows = (24 / n_rep).clamp(1, 3);
            for rows in 1..=max_rows {
                if rows > l {
                    continue;
                }
                if (n_heads, n_kv) != (16, 2) && l > 4099 {
                    continue;
                }
                let qmul = if l % 2 == 0 { 1.0 } else { 3.0 };
                let c = Case { n_heads, n_kv, l, rows, cap: l + 37, qmul };
                let dt = make(&mut rng, &c);
                let dq = DeviceBuffer::from_host(&st, &kdiff::as_bytes(&dt.q)).unwrap();
                let dk = DeviceBuffer::from_host(&st, &kdiff::as_bytes(&dt.k)).unwrap();
                let dv = DeviceBuffer::from_host(&st, &kdiff::as_bytes(&dt.v)).unwrap();
                let chunks: &[usize] = if l > 2000 { &[256, 512] } else { &[128, 256, 512] };
                let mut outs = Vec::new();
                for &chunk in chunks {
                    let (o, _) = call(&st, &f, &c, &dq, &dk, &dv, 0, rows, l, chunk, 0);
                    // (1) accuracy
                    for r in 0..rows {
                        let rf = reference(&c, &dt, r);
                        let vmax = 1.0;
                        for i in 0..n_heads * D {
                            let got = bf(o[r * n_heads * D + i]);
                            let err = (got - rf[i]).abs();
                            let tol = rf[i].abs() / 256.0 + vmax / 4096.0;
                            worst = worst.max(err / tol);
                            n_acc += 1;
                            if !(err <= tol) {
                                if ok {
                                    println!("ACCURACY FAIL heads {n_heads}/{n_kv} L {l} rows {rows} chunk {chunk} row {r} i {i}: got {got} ref {}", rf[i]);
                                }
                                ok = false;
                            }
                        }
                    }
                    // (2) row exactness vs 1-row launches, (3) mutation: neighbouring key count
                    for r in 0..rows {
                        let own = l - rows + r + 1;
                        let (o1, _) = call(&st, &f, &c, &dq, &dk, &dv, r, 1, own, chunk, 0);
                        let row = &o[r * n_heads * D..(r + 1) * n_heads * D];
                        n_exact += 1;
                        if row != &o1[..] {
                            n_exact_fail += 1;
                            ok = false;
                            let nd = row.iter().zip(o1.iter()).filter(|(a, b)| a != b).count();
                            println!("ROW-EXACT FAIL heads {n_heads}/{n_kv} L {l} rows {rows} chunk {chunk} row {r}: {nd} of {} differ", row.len());
                        }
                        if own > 1 && rows > 1 {
                            let (om, _) = call(&st, &f, &c, &dq, &dk, &dv, r, 1, own - 1, chunk, 0);
                            n_mut += 1;
                            if row != &om[..] {
                                n_mut_caught += 1;
                            }
                        }
                    }
                    outs.push(o);
                }
                if rows == 1 && n_heads == 16 && l > 256 {
                    let nd = outs[0].iter().zip(outs.last().unwrap().iter()).filter(|(a, b)| a != b).count();
                    println!("  L {l}: chunk {} vs {} differ in {nd} of {} outputs (summation order)", chunks[0], chunks[chunks.len() - 1], outs[0].len());
                }
            }
        }
    }
    println!("accuracy: {n_acc} outputs, worst err/tol {worst:.3}");
    println!("row exactness: {n_exact} rows compared bit for bit, {n_exact_fail} failing");
    println!("mutation (neighbouring key count): {n_mut_caught} of {n_mut} caught");
    if n_mut_caught * 10 < n_mut * 9 {
        println!("MUTATION CHECK WEAK");
        ok = false;
    }
    if std::env::var("FD_TIME").is_ok_and(|v| v == "1") {
        for &l in &[2100usize, 4096, 13000, 28000, 60000] {
            let c = Case { n_heads: 16, n_kv: 2, l, rows: 3, cap: l + 64, qmul: 1.0 };
            let dt = make(&mut rng, &c);
            let dq = DeviceBuffer::from_host(&st, &kdiff::as_bytes(&dt.q)).unwrap();
            let dk = DeviceBuffer::from_host(&st, &kdiff::as_bytes(&dt.k)).unwrap();
            let dv = DeviceBuffer::from_host(&st, &kdiff::as_bytes(&dt.v)).unwrap();
            for &rows in &[1usize, 3] {
                for &chunk in &[128usize, 256, 512, 1024] {
                    let (_, us) = call(&st, &f, &c, &dq, &dk, &dv, 0, rows, l, chunk, 200);
                    let gb = (2 * l * D * 2 * 2) as f64 / (us * 1e3);
                    println!("  time L {l} rows {rows} chunk {chunk}: {us:.1} us ({gb:.0} GB/s of K+V)");
                }
            }
        }
    }
    println!("flash-decode gate: {}", if ok { "PASS" } else { "FAIL" });
    ok
}
