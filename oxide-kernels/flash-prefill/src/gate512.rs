//! Gate for the head-dim-512 kernels (`flash_prefill_d512`, `flash_prefill_d512_w8`, `flash_prefill_combine512`),
//! gemma4's full-attention layers (16 query heads on 1 KV head, gemma4-12b; on 2 KV heads, REDCELL-26B), their f16
//! twins and the f16 head-dim-256 `flash_prefill_w8_f16` (gemma4's sliding layers on f16 Q / K / V), and the check
//! that the bf16 head-dim-256 kernels are unchanged.
//!
//! (1) accuracy: sampled query rows (all heads, all 512 dims) against an f64 causal softmax(Q K^T * scale) V over
//!     the row's keys, the kernel's bf16 (f16) inputs; bound |out - ref| <= 2^-7 |ref| + 2^-9 (the 256 gate's).
//! (2) splits: nsplit 2 / 3 / 8 against nsplit 1 within the same bound.
//! (3) poison: cache rows at and past kv_len are NaN; query rows past s are never read.
//! (4) mutations that must break (1): past - 1 (causal limit), gq of the other layout (wrong KV head), win + 1.
//! (5) the 256 kernels of this PTX against the base PTX (FP_BASE_PTX, default the integ worktree's committed
//!     flash_prefill.ptx): every variant, nsplit 1 / 2 / 3 / 8, windowed and not: 0 differing bits required.
//! (6) FP_TIME=1: time at gemma4-12b's ~4k / ~8k full-layer shapes (all rows of one prompt pass).
use kdiff::cuda_core::{CudaFunction, CudaModule, CudaStream, DeviceBuffer};
use kdiff::Rng;
use std::sync::Arc;


fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let r = b.wrapping_add(0x7fff + ((b >> 16) & 1));
    (r >> 16) as u16
}
fn bf(x: u16) -> f64 {
    f32::from_bits((x as u32) << 16) as f64
}
/// f32 to f16 bits, round to nearest even (normal range and subnormals; the gate's values are |x| <= 6)
fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let a = x.abs();
    if a < 6.103_515_6e-5 {
        // subnormal: units of 2^-24
        let q = (a as f64 * 16777216.0).round_ties_even() as u16;
        return sign | q;
    }
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let m = b & 0x7f_ffff;
    let mut h = ((e as u32) << 10) | (m >> 13);
    let rest = m & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    sign | h as u16
}
fn hf(x: u16) -> f64 {
    let sign = if x & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((x >> 10) & 0x1f) as i32;
    let m = (x & 0x3ff) as f64;
    if e == 0x1f {
        return if m == 0.0 { sign * f64::INFINITY } else { f64::NAN };
    }
    if e == 0 {
        return sign * m * 2f64.powi(-24);
    }
    sign * (1.0 + m / 1024.0) * 2f64.powi(e - 15)
}
fn b8(x: u64) -> [u8; 8] {
    x.to_le_bytes()
}
fn tol(r: f64) -> f64 {
    r.abs() / 128.0 + 1.0 / 512.0
}

struct Case {
    d: usize,
    /// f16 inputs (else bf16)
    f16: bool,
    s: usize,
    past: usize,
    cap: usize,
    h: usize,
    kvh: usize,
    qmul: f32,
}

struct Data {
    q: Vec<u16>, // [head][s][D]
    k: Vec<u16>, // [kvh][cap][D]
    v: Vec<u16>,
}

fn make(r: &mut Rng, c: &Case) -> Data {
    let d = c.d;
    let f16 = c.f16;
    let mut unif = |a: f32| -> u16 {
        let u = (r.next() >> 40) as f32 / (1u64 << 24) as f32;
        if f16 { f16_bits((2.0 * u - 1.0) * a) } else { bf16_bits((2.0 * u - 1.0) * a) }
    };
    let q: Vec<u16> = (0..c.h * c.s * d).map(|_| unif(2.0 * c.qmul)).collect();
    let l = c.past + c.s;
    let nan = if f16 { 0x7e01u16 } else { 0x7fc1u16 };
    let mut k = vec![nan; c.kvh * c.cap * d];
    let mut v = vec![nan; c.kvh * c.cap * d];
    for h in 0..c.kvh {
        for t in 0..l {
            for x in 0..d {
                k[(h * c.cap + t) * d + x] = unif(2.0);
                v[(h * c.cap + t) * d + x] = unif(1.0);
            }
        }
    }
    Data { q, k, v }
}

/// softmax scale: 1 / sqrt(D) (gemma4 itself uses 1.0 on normed q / k; the kernel takes any scale)
fn scale(d: usize) -> f64 {
    1.0 / (d as f64).sqrt()
}

/// f64 reference for query position i, all heads: [h][d].
fn reference(c: &Case, dt: &Data, past: usize, i: usize, win: usize) -> Vec<f64> {
    let d_ = c.d;
    let cv = |x: u16| if c.f16 { hf(x) } else { bf(x) };
    let n_rep = c.h / c.kvh;
    let len = past + i + 1;
    let lo = if win > 0 { len.saturating_sub(win) } else { 0 };
    let sc_ = scale(d_);
    let mut out = vec![0f64; c.h * d_];
    for h in 0..c.h {
        let kvh = h / n_rep;
        let qv: Vec<f64> = (0..d_).map(|d| cv(dt.q[(h * c.s + i) * d_ + d]) * sc_).collect();
        let mut sc = vec![f64::NEG_INFINITY; len];
        let mut mx = f64::NEG_INFINITY;
        for (t, x) in sc.iter_mut().enumerate().skip(lo) {
            let kr = &dt.k[(kvh * c.cap + t) * d_..(kvh * c.cap + t + 1) * d_];
            *x = (0..d_).map(|d| qv[d] * cv(kr[d])).sum();
            mx = mx.max(*x);
        }
        let mut den = 0.0;
        for x in sc.iter_mut() {
            *x = (*x - mx).exp();
            den += *x;
        }
        for (t, p) in sc.iter().enumerate().skip(lo) {
            let vr = &dt.v[(kvh * c.cap + t) * d_..(kvh * c.cap + t + 1) * d_];
            for d in 0..d_ {
                out[h * d_ + d] += p * cv(vr[d]);
            }
        }
        for d in 0..d_ {
            out[h * d_ + d] /= den;
        }
    }
    out
}

struct Bufs {
    q: DeviceBuffer<u8>,
    k: DeviceBuffer<u8>,
    v: DeviceBuffer<u8>,
}

fn upload(st: &Arc<CudaStream>, dt: &Data) -> Bufs {
    Bufs {
        q: DeviceBuffer::from_host(st, &kdiff::as_bytes(&dt.q)).unwrap(),
        k: DeviceBuffer::from_host(st, &kdiff::as_bytes(&dt.k)).unwrap(),
        v: DeviceBuffer::from_host(st, &kdiff::as_bytes(&dt.v)).unwrap(),
    }
}

/// One call: a head-dim-512 kernel (grid y = n_heads / 8 query-head groups, `gq` per KV head, z = 2 nsplit,
/// combine512) when c.d == 512, else a head-dim-256 kernel (grid y = KV heads, z = nsplit, combine); `nw` warps.
/// Returns bf16 out [head][s][d] and us per call over `reps`.
#[allow(clippy::too_many_arguments)]
fn call512(st: &Arc<CudaStream>, main: &CudaFunction, comb: &CudaFunction, ks: Ks, c: &Case, b: &Bufs, past: usize,
           nsplit: usize, reps: usize, win: usize, gq: usize) -> (Vec<f64>, f64) {
    let nw = ks.nw;
    let esz = if ks.o32 { 4 } else { 2 };
    let (s, h, d) = (c.s, c.h, c.d);
    let kv_len = past + s;
    let out = DeviceBuffer::from_host(st, &vec![0xffu8; h * s * d * esz]).unwrap();
    let part = DeviceBuffer::from_host(st, &vec![0xffu8; nsplit.max(2) * h * s * d * 4]).unwrap();
    let ml = DeviceBuffer::from_host(st, &vec![0xffu8; nsplit.max(2) * h * s * 2 * 4]).unwrap();
    let sl2 = (scale(d) as f32) * std::f32::consts::LOG2_E;
    let (gy, gz) = if d == 512 { ((h / 8) as u32, (2 * nsplit) as u32) } else { (c.kvh as u32, nsplit as u32) };
    let launch = || {
        let mut a: Vec<[u8; 8]> = vec![
            b8(b.q.cu_deviceptr()), b8(b.k.cu_deviceptr()), b8(b.v.cu_deviceptr()), b8(out.cu_deviceptr()),
            b8(part.cu_deviceptr()), b8(ml.cu_deviceptr()), b8(sl2.to_bits() as u64), b8(s as u64), b8(past as u64),
            b8(kv_len as u64), b8(h as u64), b8(nsplit as u64), b8((s * d) as u64), b8((s * d) as u64),
            b8((c.cap * d) as u64), b8((c.cap * d) as u64), b8(win as u64),
        ];
        if d == 512 {
            a.push(b8(gq as u64));
        }
        let mut p: Vec<*mut std::ffi::c_void> = a.iter_mut().map(|v| v.as_mut_ptr() as *mut std::ffi::c_void).collect();
        unsafe {
            kdiff::cuda_core::simt::launch_kernel_on_stream(main, (s.div_ceil((16 / ks.krep) * nw) as u32, gy, gz), ((32 * nw) as u32, 1, 1), 0, st, &mut p)
                .unwrap()
        };
        if nsplit > 1 {
            let mut a: Vec<[u8; 8]> = vec![
                b8(part.cu_deviceptr()), b8(ml.cu_deviceptr()), b8(out.cu_deviceptr()), b8(s as u64), b8(h as u64),
                b8(nsplit as u64), b8((s * d) as u64),
            ];
            let mut p: Vec<*mut std::ffi::c_void> = a.iter_mut().map(|v| v.as_mut_ptr() as *mut std::ffi::c_void).collect();
            let cb = if d == 128 { 128 } else { 256 };
            unsafe { kdiff::cuda_core::simt::launch_kernel_on_stream(comb, (s as u32, h as u32, 1), (cb, 1, 1), 0, st, &mut p).unwrap() };
        }
    };
    launch();
    st.synchronize().unwrap();
    let mut us = 0.0;
    if reps > 0 {
        for _ in 0..3 {
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
    let o: Vec<f64> = if ks.o32 {
        bytes.chunks(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64).collect()
    } else {
        bytes.chunks(2).map(|b| bf(u16::from_le_bytes([b[0], b[1]]))).collect()
    };
    (o, us)
}

/// Kernel launch shape: warps per block, query heads per KV head the warp tile packs (8, or 2 for the r2 kernels),
/// f32 output (the o32 kernels, with the f32 combines).
#[derive(Clone, Copy)]
struct Ks {
    nw: usize,
    krep: usize,
    o32: bool,
}

fn sample_rows(s: usize) -> Vec<usize> {
    if s <= 64 {
        (0..s).collect()
    } else {
        let mut r: Vec<usize> = (0..9).chain(s - 9..s).chain((1..13).map(|j| j * s / 13)).collect();
        r.sort_unstable();
        r.dedup();
        r
    }
}

/// worst err / tol of `o` against `refs` on `rows` (NaN counts as infinite)
fn worst(o: &[f64], refs: &[Vec<f64>], rows: &[usize], c: &Case) -> f64 {
    let mut w = 0f64;
    for (ri, &i) in rows.iter().enumerate() {
        for j in 0..c.h * c.d {
            let got = o[((j / c.d) * c.s + i) * c.d + j % c.d];
            let e = (got - refs[ri][j]).abs() / tol(refs[ri][j]);
            w = w.max(if e.is_nan() { f64::INFINITY } else { e });
        }
    }
    w
}

/// Accuracy (all splits, every variant), splits / variants consistency, mutations, for kernels `vars` at head dim
/// `d` on the cases (s, past, kv heads, win) with 16 query heads; prints and returns pass.
fn suite(st: &Arc<CudaStream>, m: &Arc<CudaModule>, label: &str, d: usize, f16: bool, vars: &[(&str, Ks)],
         cases: &[(usize, usize, usize, usize)], rng: &mut Rng) -> bool {
    let mut ok = true;
    let mains: Vec<CudaFunction> = vars.iter().map(|&(n, _)| m.load_function(n).unwrap()).collect();
    let comb_for = |ks: Ks| {
        m.load_function(match (d, ks.o32) {
            (512, false) => "flash_prefill_combine512",
            (512, true) => "flash_prefill_combine512_f32",
            (128, false) => "flash_prefill_combine128",
            (128, true) => "flash_prefill_combine128_f32",
            (_, false) => "flash_prefill_combine",
            (_, true) => "flash_prefill_combine_f32",
        })
        .unwrap()
    };
    let combs: Vec<CudaFunction> = vars.iter().map(|&(_, ks)| comb_for(ks)).collect();
    for (&(n, _), f) in vars.iter().zip(&mains) {
        println!(
            "{n}: {} registers, {} B local (spills), {} B static shared",
            f.num_registers().unwrap_or(0),
            f.local_size_bytes().unwrap_or(0),
            f.static_shared_memory_bytes().unwrap_or(0)
        );
    }
    let (mut n_acc, mut worst_acc) = (0usize, 0f64);
    let (mut n_split, mut worst_split) = (0usize, 0f64);
    let (mut n_var, mut n_var_diff) = (0usize, 0usize);
    let (mut n_mut, mut n_mut_caught) = (0usize, 0usize);
    for &(s, past, kvh, win) in cases {
        let c = Case { d, f16, s, past, cap: past + s + 40, h: 16, kvh, qmul: if (s + past) % 2 == 0 { 1.0 } else { 3.0 } };
        let gq = (c.h / c.kvh / 8).max(1);
        let dt = make(rng, &c);
        let b = upload(st, &dt);
        let rows = sample_rows(s);
        let refs: Vec<Vec<f64>> = rows.iter().map(|&i| reference(&c, &dt, past, i, win)).collect();
        let mut first: Option<Vec<f64>> = None;
        let mut ns1: Vec<Vec<f64>> = Vec::new();
        for (vi, f) in mains.iter().enumerate() {
            for &ns in &[1usize, 2, 3, 8] {
                let (o, _) = call512(st, f, &combs[vi], vars[vi].1, &c, &b, past, ns, 0, win, gq);
                let w = worst(&o, &refs, &rows, &c);
                n_acc += rows.len() * c.h * d;
                worst_acc = worst_acc.max(w);
                if !(w <= 1.0) {
                    println!("ACCURACY FAIL {} s {s} past {past} kvh {kvh} win {win} nsplit {ns}: worst {w:.3}", vars[vi].0);
                    ok = false;
                }
                println!("  {} s {s:4} past {past:5} kvh {kvh} win {win:4} nsplit {ns}: worst err/tol {w:.3}", vars[vi].0);
                if ns == 1 {
                    ns1.push(o.clone());
                }
                match &first {
                    None => first = Some(o),
                    Some(f0) => {
                        for (a, b) in f0.iter().zip(o.iter()) {
                            let e = (a - b).abs() / tol(*a);
                            worst_split = worst_split.max(if e.is_nan() { f64::INFINITY } else { e });
                            n_split += 1;
                            if !(e <= 1.0) {
                                ok = false;
                            }
                        }
                    }
                }
            }
        }
        for o in &ns1[1..] {
            for (a, b) in ns1[0].iter().zip(o.iter()) {
                n_var += 1;
                n_var_diff += (a.to_bits() != b.to_bits()) as usize;
            }
        }
        let caught = |o: &[f64]| worst(o, &refs, &rows, &c) > 1.0 / 8.0;
        // past - 1 drops one key of past + i + 1: below the tolerance by design for a lone decode row over thousands
        // of keys (the f32-output kernels have no bf16 output noise to push it over), so only for s >= 8
        if past > 0 && s >= 8 {
            let (o, _) = call512(st, &mains[0], &combs[0], vars[0].1, &c, &b, past - 1, 1, 0, win, gq);
            n_mut += 1;
            n_mut_caught += caught(&o) as usize;
        }
        if d == 512 && kvh == 2 {
            let l = mains.len() - 1;
            let (o, _) = call512(st, &mains[l], &combs[l], vars[l].1, &c, &b, past, 1, 0, win, 2);
            n_mut += 1;
            n_mut_caught += caught(&o) as usize;
        }
        if win > 0 && past + s > win {
            let (o, _) = call512(st, &mains[0], &combs[0], vars[0].1, &c, &b, past, 2, 0, win + 1, gq);
            n_mut += 1;
            n_mut_caught += caught(&o) as usize;
        }
    }
    println!("{label} accuracy: {n_acc} outputs, worst err/tol {worst_acc:.3}");
    println!("{label} splits / variants: {n_split} outputs vs the first variant at nsplit 1, worst {worst_split:.3} of tol");
    println!("{label} variants at nsplit 1: {n_var_diff} of {n_var} outputs differ in any bit from the first (informational; bf16 vs f32 outputs always differ)");
    println!("{label} mutations (past - 1, wrong KV head, win + 1): {n_mut_caught} of {n_mut} caught");
    if n_mut_caught < n_mut || n_mut == 0 {
        println!("{label} MUTATION CHECK WEAK");
        ok = false;
    }
    ok
}

pub fn run(ctx: &Arc<kdiff::cuda_core::CudaContext>, m: &Arc<CudaModule>) -> bool {
    let st = ctx.default_stream();
    let mut ok = true;
    let quick = std::env::var("FP_QUICK").is_ok_and(|v| v == "1");
    let mut rng = Rng(0x0D51_2F11);
    let k = |nw: usize, krep: usize, o32: bool| Ks { nw, krep, o32 };
    // (s, past, kv heads, win), 16 query heads; at 512 kvh 1 = gemma4-12b (gq 2), kvh 2 = REDCELL (gq 1)
    let mut c512: Vec<(usize, usize, usize, usize)> = vec![
        (8, 0, 1, 0), (13, 0, 2, 0), (64, 0, 1, 0), (512, 0, 2, 0), (200, 1, 1, 0), (511, 37, 1, 0), (512, 512, 2, 0),
        (64, 4096, 1, 0), (100, 3000, 2, 0), (512, 4608, 1, 0), (300, 50, 2, 100), (700, 0, 1, 512), (33, 0, 1, 16),
        // decode rows (bf16 decode runs on these kernels): one query over the whole cache
        (1, 0, 1, 0), (1, 3000, 2, 0), (1, 5000, 1, 0),
    ];
    if !quick {
        c512.extend([(512, 13000, 1, 0), (233, 7500, 2, 0), (1, 9000, 1, 0)]);
    }
    ok &= suite(&st, m, "d512 bf16", 512, false, &[("flash_prefill_d512", k(4, 8, false)), ("flash_prefill_d512_w8", k(8, 8, false))], &c512, &mut rng);
    let c512h: Vec<(usize, usize, usize, usize)> =
        vec![(1, 0, 1, 0), (1, 3000, 2, 0), (13, 0, 2, 0), (512, 0, 1, 0), (511, 37, 1, 0), (100, 3000, 2, 0), (512, 4608, 1, 0), (300, 50, 2, 100)];
    ok &= suite(&st, m, "d512 f16", 512, true, &[("flash_prefill_d512_f16", k(4, 8, false)), ("flash_prefill_d512_w8_f16", k(8, 8, false))], &c512h, &mut rng);
    ok &= suite(&st, m, "d512 f16 o32", 512, true, &[("flash_prefill_d512_f16o32", k(4, 8, true))], &c512h, &mut rng);
    // head dim 256, 8 query heads per KV head (kvh 2), f16
    let c256h: Vec<(usize, usize, usize, usize)> = vec![
        (13, 0, 2, 0), (512, 0, 2, 0), (511, 37, 2, 0), (100, 3000, 2, 0), (512, 4608, 2, 1024), (1500, 0, 2, 1024),
        (300, 50, 2, 100), (1000, 0, 2, 40),
    ];
    ok &= suite(&st, m, "d256 f16", 256, true, &[("flash_prefill_w8_f16", k(8, 8, false)), ("flash_prefill_w8_f16o32", k(8, 8, true))], &c256h, &mut rng);
    // head dim 256, 2 query heads per KV head (kvh 8: gemma4 sliding layers), window 1024 and others, decode rows
    let c256r2: Vec<(usize, usize, usize, usize)> = vec![
        (1, 0, 8, 0), (1, 2000, 8, 1024), (7, 0, 8, 0), (13, 0, 8, 0), (64, 0, 8, 0), (511, 37, 8, 0), (100, 3000, 8, 0),
        (512, 4608, 8, 1024), (1500, 0, 8, 1024), (300, 50, 8, 100), (1000, 0, 8, 40), (65, 1023, 8, 1024),
    ];
    ok &= suite(&st, m, "d256 r2 bf16", 256, false, &[("flash_prefill_r2_w8", k(8, 2, false))], &c256r2, &mut rng);
    ok &= suite(&st, m, "d256 r2 f16 o32", 256, true, &[("flash_prefill_r2_w8_f16o32", k(8, 2, true))], &c256r2, &mut rng);
    // head dim 128 (qwen3: 5 query heads per KV head, zero-padded to the kernel's 8), 8 per KV head, kvh 2
    let mut c128: Vec<(usize, usize, usize, usize)> = vec![
        (1, 0, 2, 0), (1, 3000, 2, 0), (8, 0, 2, 0), (13, 0, 2, 0), (64, 0, 2, 0), (512, 0, 2, 0), (200, 1, 2, 0),
        (511, 37, 2, 0), (512, 512, 2, 0), (100, 3000, 2, 0), (512, 4608, 2, 0), (1500, 0, 2, 0), (300, 50, 2, 100),
        (33, 0, 2, 16),
    ];
    if !quick {
        c128.extend([(512, 13000, 2, 0), (233, 27500, 2, 0), (4096, 12000, 2, 0)]);
    }
    ok &= suite(&st, m, "d128 bf16", 128, false, &[("flash_prefill_d128_w8", k(8, 8, false))], &c128, &mut rng);
    ok &= suite(&st, m, "d128 f16 o32", 128, true, &[("flash_prefill_d128_w8_f16o32", k(8, 8, true))], &c128, &mut rng);

    // the existing kernels against the base PTX (the integ2 committed one), bit for bit
    let base_path = std::env::var("FP_BASE_PTX")
        .unwrap_or_else(|_| format!("{}/titan-engine/oxide-integ2/flash-prefill/flash_prefill.ptx", std::env::var("HOME").unwrap()));
    let base = ctx.load_module_from_file(&base_path).expect("load base flash_prefill.ptx");
    let (mut n_bits, mut n_bits_diff, mut n_calls) = (0usize, 0usize, 0usize);
    // (kernel, ks, head dim, f16)
    let old: Vec<(&str, Ks, usize, bool)> = vec![
        ("flash_prefill", k(4, 8, false), 256, false), ("flash_prefill_w8", k(8, 8, false), 256, false),
        ("flash_prefill_qg", k(4, 8, false), 256, false), ("flash_prefill_qg_w8", k(8, 8, false), 256, false),
        ("flash_prefill_v1", k(4, 8, false), 256, false), ("flash_prefill_w8_f16", k(8, 8, false), 256, true),
        ("flash_prefill_d512", k(4, 8, false), 512, false), ("flash_prefill_d512_w8", k(8, 8, false), 512, false),
        ("flash_prefill_d512_f16", k(4, 8, false), 512, true), ("flash_prefill_d512_w8_f16", k(8, 8, false), 512, true),
        ("flash_prefill_r2_w8", k(8, 2, false), 256, false), ("flash_prefill_r2_w8_f16o32", k(8, 2, true), 256, true),
        ("flash_prefill_w8_f16o32", k(8, 8, true), 256, true), ("flash_prefill_d512_f16o32", k(4, 8, true), 512, true),
    ];
    for &(s, past, win) in &[(13usize, 0usize, 0usize), (512, 0, 0), (511, 37, 0), (100, 3000, 0), (512, 4608, 0), (512, 4608, 512), (300, 50, 100), (1000, 0, 40)] {
        for &(n, ks, d, f16) in &old {
            if n == "flash_prefill_v1" && win > 0 {
                continue;
            }
            if d == 512 && s > 512 {
                continue;
            }
            let kvh = if d == 512 { 1 } else if ks.krep == 2 { 8 } else { 2 };
            let c = Case { d, f16, s, past, cap: past + s + 40, h: 16, kvh, qmul: 2.0 };
            let dt = make(&mut rng, &c);
            let b = upload(&st, &dt);
            let cn = match (d == 512, ks.o32) {
                (true, false) => "flash_prefill_combine512",
                (true, true) => "flash_prefill_combine512_f32",
                (false, false) => "flash_prefill_combine",
                (false, true) => "flash_prefill_combine_f32",
            };
            let (fnew, fbase) = (m.load_function(n).unwrap(), base.load_function(n).unwrap());
            let (cnew, cbase) = (m.load_function(cn).unwrap(), base.load_function(cn).unwrap());
            let gq = if d == 512 { 2 } else { 1 };
            for &ns in &[1usize, 2, 3, 8] {
                let (a, _) = call512(&st, &fnew, &cnew, ks, &c, &b, past, ns, 0, win, gq);
                let (o, _) = call512(&st, &fbase, &cbase, ks, &c, &b, past, ns, 0, win, gq);
                n_calls += 1;
                for (x, y) in a.iter().zip(o.iter()) {
                    n_bits += 1;
                    n_bits_diff += (x.to_bits() != y.to_bits()) as usize;
                }
            }
        }
    }
    println!("existing kernels (256 bf16 x5, 256 f16, 512 bf16 x2, 512 f16 x2, r2 bf16 / f16 o32, 256 f16 o32, 512 f16 o32) vs base PTX {base_path}: {n_calls} calls, {n_bits_diff} of {n_bits} outputs differ in any bit");
    if n_bits_diff > 0 || n_bits == 0 {
        ok = false;
    }

    if std::env::var("FP_TIME").is_ok_and(|v| v == "1") {
        // gemma4-12b one-pass prompt shapes: full layers (16 heads, 1 KV head, 512), sliding (16 heads, 8 KV, 256, win 1024)
        let tv: Vec<(&str, Ks, usize, bool, usize, usize)> = vec![
            ("flash_prefill_d512", k(4, 8, false), 512, false, 1, 0), ("flash_prefill_d512_f16o32", k(4, 8, true), 512, true, 1, 0),
            ("flash_prefill_w8", k(8, 8, false), 256, false, 2, 1024), ("flash_prefill_r2_w8", k(8, 2, false), 256, false, 8, 1024),
            ("flash_prefill_r2_w8_f16o32", k(8, 2, true), 256, true, 8, 1024),
        ];
        for &s in &[4096usize, 8192] {
            for &(n, ks, d, f16, kvh, win) in &tv {
                // the padded w8 path runs 8 heads per KV head: time it on 16 heads / 2 KV heads (the padded shape of 4 real)
                let c = Case { d, f16, s, past: 0, cap: s + 64, h: 16, kvh, qmul: 1.0 };
                let dt = make(&mut rng, &c);
                let b = upload(&st, &dt);
                let f = m.load_function(n).unwrap();
                let comb = m.load_function(match (d, ks.o32) { (512, false) => "flash_prefill_combine512", (512, true) => "flash_prefill_combine512_f32", (_, false) => "flash_prefill_combine", _ => "flash_prefill_combine_f32" }).unwrap();
                for &ns in &[1usize, 2] {
                    let (_, us) = call512(&st, &f, &comb, ks, &c, &b, 0, ns, 5, win, (c.h / kvh / 8).max(1));
                    println!("  time {n:28} s {s} kvh {kvh} win {win} nsplit {ns}: {us:9.1} us");
                }
            }
        }
    }
    println!("flash-prefill d512 / f16 gate: {}", if ok { "PASS" } else { "FAIL" });
    ok
}
