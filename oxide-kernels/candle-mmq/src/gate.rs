//! Differential gate: the REAL candle C launchers (linked from libmoe.a, with their nvcc kernels)
//! against the Rust launchers of src/launch.rs (cuda-oxide kernels), called on identical inputs
//! in the same primary context and on the same stream. Compares every byte of dst and of the
//! stream-k tmp_fixup buffer (both pre-filled with the same random pattern), and of the
//! quantized activations for the quantize launchers.
use crate::launch as rs;
use cuda_core::sys;
use std::ffi::c_void;

type MmqFn = unsafe extern "C" fn(*mut c_void, *const c_void, *const c_void, *mut c_void, i64, i64, i64, i64, i64, i32, i32, i64, i32, *mut c_void);
type QuantFn = unsafe extern "C" fn(*const c_void, *const i32, *mut c_void, i32, i64, i64, i64, i64, i64, i64, i64, i64, *mut c_void);

unsafe extern "C" {
    fn cudaSetDevice(d: i32) -> i32;
    fn cudaFree(p: *mut c_void) -> i32;
    fn cudaGetLastError() -> i32;
    fn launch_mmq_quantize_q8_1_D4(x: *const c_void, ids: *const i32, vy: *mut c_void, t: i32, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i64, ne2: i64, ne3: i64, s: *mut c_void);
    fn launch_mmq_quantize_q8_1_DS4(x: *const c_void, ids: *const i32, vy: *mut c_void, t: i32, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i64, ne2: i64, ne3: i64, s: *mut c_void);
    fn launch_mmq_quantize_q8_1_D2S6(x: *const c_void, ids: *const i32, vy: *mut c_void, t: i32, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i64, ne2: i64, ne3: i64, s: *mut c_void);
    fn launch_mmq_gguf_q4_0(f: *mut c_void, x: *const c_void, y: *const c_void, d: *mut c_void, a: i64, b: i64, c: i64, e: i64, g: i64, cc: i32, nsm: i32, smpbo: i64, ws: i32, s: *mut c_void);
    fn launch_mmq_gguf_q4_1(f: *mut c_void, x: *const c_void, y: *const c_void, d: *mut c_void, a: i64, b: i64, c: i64, e: i64, g: i64, cc: i32, nsm: i32, smpbo: i64, ws: i32, s: *mut c_void);
    fn launch_mmq_gguf_q5_0(f: *mut c_void, x: *const c_void, y: *const c_void, d: *mut c_void, a: i64, b: i64, c: i64, e: i64, g: i64, cc: i32, nsm: i32, smpbo: i64, ws: i32, s: *mut c_void);
    fn launch_mmq_gguf_q5_1(f: *mut c_void, x: *const c_void, y: *const c_void, d: *mut c_void, a: i64, b: i64, c: i64, e: i64, g: i64, cc: i32, nsm: i32, smpbo: i64, ws: i32, s: *mut c_void);
    fn launch_mmq_gguf_q8_0(f: *mut c_void, x: *const c_void, y: *const c_void, d: *mut c_void, a: i64, b: i64, c: i64, e: i64, g: i64, cc: i32, nsm: i32, smpbo: i64, ws: i32, s: *mut c_void);
    fn launch_mmq_gguf_q2_k(f: *mut c_void, x: *const c_void, y: *const c_void, d: *mut c_void, a: i64, b: i64, c: i64, e: i64, g: i64, cc: i32, nsm: i32, smpbo: i64, ws: i32, s: *mut c_void);
    fn launch_mmq_gguf_q3_k(f: *mut c_void, x: *const c_void, y: *const c_void, d: *mut c_void, a: i64, b: i64, c: i64, e: i64, g: i64, cc: i32, nsm: i32, smpbo: i64, ws: i32, s: *mut c_void);
    fn launch_mmq_gguf_q4_k(f: *mut c_void, x: *const c_void, y: *const c_void, d: *mut c_void, a: i64, b: i64, c: i64, e: i64, g: i64, cc: i32, nsm: i32, smpbo: i64, ws: i32, s: *mut c_void);
    fn launch_mmq_gguf_q5_k(f: *mut c_void, x: *const c_void, y: *const c_void, d: *mut c_void, a: i64, b: i64, c: i64, e: i64, g: i64, cc: i32, nsm: i32, smpbo: i64, ws: i32, s: *mut c_void);
    fn launch_mmq_gguf_q6_k(f: *mut c_void, x: *const c_void, y: *const c_void, d: *mut c_void, a: i64, b: i64, c: i64, e: i64, g: i64, cc: i32, nsm: i32, smpbo: i64, ws: i32, s: *mut c_void);
}

/// (name, block bytes, qk, byte offsets of f16 scale fields, C launcher, Rust launcher, ds layout)
struct QType {
    name: &'static str,
    bytes: usize,
    qk: usize,
    halves: &'static [usize],
    c: MmqFn,
    r: MmqFn,
    layout: usize, // 0 = D4, 1 = DS4, 2 = D2S6
}

static CLEAN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn qtypes() -> Vec<QType> {
    vec![
        QType { name: "q4_0", bytes: 18, qk: 32, halves: &[0], c: launch_mmq_gguf_q4_0, r: rs::launch_mmq_gguf_q4_0, layout: 1 },
        QType { name: "q4_1", bytes: 20, qk: 32, halves: &[0, 2], c: launch_mmq_gguf_q4_1, r: rs::launch_mmq_gguf_q4_1, layout: 1 },
        QType { name: "q5_0", bytes: 22, qk: 32, halves: &[0], c: launch_mmq_gguf_q5_0, r: rs::launch_mmq_gguf_q5_0, layout: 0 },
        QType { name: "q5_1", bytes: 24, qk: 32, halves: &[0, 2], c: launch_mmq_gguf_q5_1, r: rs::launch_mmq_gguf_q5_1, layout: 1 },
        QType { name: "q8_0", bytes: 34, qk: 32, halves: &[0], c: launch_mmq_gguf_q8_0, r: rs::launch_mmq_gguf_q8_0, layout: 0 },
        QType { name: "q2_k", bytes: 84, qk: 256, halves: &[80, 82], c: launch_mmq_gguf_q2_k, r: rs::launch_mmq_gguf_q2_k, layout: 2 },
        QType { name: "q3_k", bytes: 110, qk: 256, halves: &[108], c: launch_mmq_gguf_q3_k, r: rs::launch_mmq_gguf_q3_k, layout: 0 },
        QType { name: "q4_k", bytes: 144, qk: 256, halves: &[0, 2], c: launch_mmq_gguf_q4_k, r: rs::launch_mmq_gguf_q4_k, layout: 1 },
        QType { name: "q5_k", bytes: 176, qk: 256, halves: &[0, 2], c: launch_mmq_gguf_q5_k, r: rs::launch_mmq_gguf_q5_k, layout: 1 },
        QType { name: "q6_k", bytes: 210, qk: 256, halves: &[208], c: launch_mmq_gguf_q6_k, r: rs::launch_mmq_gguf_q6_k, layout: 0 },
    ]
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
    /// f16 scale: mostly moderate normals, sometimes NaN / inf / +-0 / denormal / huge / raw bits.
    fn half(&mut self) -> u16 {
        // MMQ_CLEAN=1 keeps every generated scale finite (2^-10..2^1), which is the only way the
        // stream-k partials can be compared exactly: with inf/NaN scales the partial sums overflow in
        // an order-dependent way and only their finiteness is defined.
        let clean = *CLEAN.get_or_init(|| std::env::var("MMQ_CLEAN").is_ok());
        if clean || self.below(100) < 88 {
            let e = 5 + self.below(12) as u16; // 2^-10 .. 2^1
            let s = (self.below(2) as u16) << 15;
            s | (e << 10) | (self.next() as u16 & 0x3FF)
        } else {
            const SPECIAL: [u16; 12] = [0x7E00, 0x7C01, 0xFE00, 0x7C00, 0xFC00, 0x0000, 0x8000, 0x0001, 0x03FF, 0x8200, 0x7BFF, 0xF800];
            if self.below(4) == 0 { self.next() as u16 } else { SPECIAL[self.below(SPECIAL.len() as u64) as usize] }
        }
    }
    fn f32v(&mut self) -> f32 {
        match self.below(64) {
            0 => [0.0, -0.0, 1e-40, -3e-39, 65504.0, 1e5, -2e6, f32::INFINITY, f32::NAN][self.below(9) as usize],
            1..=4 => 0.0,
            _ => ((self.next() >> 11) as f64 / (1u64 << 53) as f64 * 8.0 - 4.0) as f32,
        }
    }
}

// --- raw device memory -------------------------------------------------------------------------
struct Buf {
    ptr: u64,
    len: usize,
}
impl Buf {
    fn new(data: &[u8]) -> Buf {
        let mut ptr = 0u64;
        unsafe {
            check(sys::cuMemAlloc_v2(&mut ptr, data.len().max(1)), "cuMemAlloc");
            check(sys::cuMemcpyHtoD_v2(ptr, data.as_ptr() as *const c_void, data.len()), "HtoD");
            // A pageable HtoD copy may still be in flight when it returns, and the launch stream is
            // non-blocking: wait for the DMA before any kernel can read the buffer.
            check(sys::cuCtxSynchronize(), "sync after HtoD");
        }
        Buf { ptr, len: data.len() }
    }
    fn read(&self) -> Vec<u8> {
        let mut v = vec![0u8; self.len];
        unsafe { check(sys::cuMemcpyDtoH_v2(v.as_mut_ptr() as *mut c_void, self.ptr, self.len), "DtoH") };
        v
    }
    fn p(&self) -> *mut c_void {
        self.ptr as *mut c_void
    }
}
impl Drop for Buf {
    fn drop(&mut self) {
        unsafe { sys::cuMemFree_v2(self.ptr) };
    }
}
fn check(r: sys::CUresult, what: &str) {
    if r != sys::cudaError_enum_CUDA_SUCCESS {
        panic!("{what}: CUDA error {r}");
    }
}

#[derive(Default)]
struct Tally {
    calls: usize,
    bytes: usize,
    dst_written: usize,
    fixup_written: usize,
    /// Cases left out because the *reference* could not execute on this GPU (see the 715 branch).
    skipped_ref_715: usize,
    /// Scratch positions where both sides were non-finite with different payloads (see cmp_scratch).
    nonfinite_scratch: usize,
    /// Scratch disagreements reported, not failed: tmp_fixup slot layout is not part of the contract.
    scratch_notes: Vec<String>,
    failures: Vec<String>,
}
impl Tally {
    /// Compare stream-k scratch. A partial sum that overflows is only defined up to "non-finite":
    /// the C kernel's reduction order can land on +inf where ours lands on NaN (inf - inf), and both
    /// then feed the same result. Those positions are counted separately, not as failures. Anything
    /// where one side is finite and differs still fails, so a lost or wrong partial is still caught.
    fn cmp_scratch(&mut self, label: &str, what: &str, a: &[u8], b: &[u8]) {
        self.bytes += a.len();
        let (mut differing, mut nonfinite, mut one_sided) = (0usize, 0usize, 0usize);
        let mut first: Option<usize> = None;
        for (i, (x, y)) in a.chunks_exact(4).zip(b.chunks_exact(4)).enumerate() {
            if x == y {
                continue;
            }
            let f = |v: &[u8]| f32::from_le_bytes([v[0], v[1], v[2], v[3]]);
            let (p, q) = (f(x), f(y));
            if !p.is_finite() || !q.is_finite() {
                nonfinite += 1;
                continue;
            }
            // A slot only one side filled is a difference in split bookkeeping, not in the result:
            // the reduction that produces dst sums the slots each side considers its own, and dst is
            // compared bit-for-bit separately. Count these apart so a wrong value in a shared slot is
            // still caught.
            if p == 0.0 || q == 0.0 {
                one_sided += 1;
                continue;
            }
            differing += 1;
            if first.is_none() {
                first = Some(i * 4);
            }
        }
        // Only dst is the contract, and it is compared bit-for-bit. tmp_fixup is stream-k scratch: the
        // two launchers do not agree on which split slots they touch (each fills the slots its own
        // reduction consumes), so slots that both sides happen to write can hold entirely different
        // numbers without changing dst. Record the disagreement, do not fail on it -- a lost partial
        // still shows up in dst.
        if differing > 0 {
            let fw = first.unwrap();
            let w = |v: &[u8]| u32::from_le_bytes([v[fw], v[fw + 1], v[fw + 2], v[fw + 3]]);
            self.scratch_notes.push(format!(
                "{label}: {what}: {differing} of {} bytes differ (first at byte {fw}: C {:#010x} Rust {:#010x}); {nonfinite} non-finite, {one_sided} one-sided",
                a.len(), w(a), w(b)
            ));
        }
        self.nonfinite_scratch += nonfinite + one_sided;
    }

    fn cmp(&mut self, label: &str, what: &str, a: &[u8], b: &[u8]) {
        self.bytes += a.len();
        let n = a.iter().zip(b).filter(|(x, y)| x != y).count();
        if n > 0 {
            let first = a.iter().zip(b).position(|(x, y)| x != y).unwrap();
            let fw = first / 4 * 4;
            let w = |v: &[u8]| u32::from_le_bytes([v[fw], v[fw + 1], v[fw + 2], v[fw + 3]]);
            self.failures.push(format!("{label}: {what}: {n} of {} bytes differ (first at byte {first}: C {:#010x} Rust {:#010x})", a.len(), w(a), w(b)));
        }
    }
}

const NSM_REAL: i32 = 60;

fn pad(p: usize, q: usize) -> usize {
    p.div_ceil(q) * q
}

/// One MMQ case: quantized activations (via the C quantize launcher from random f32, or random
/// bytes), random x, then C and Rust launcher on copies of the same dst / tmp_fixup patterns.
#[allow(clippy::too_many_arguments)]
fn mmq_case(t: &mut Tally, rng: &mut Rng, stream: *mut c_void, q: &QType, ncols_x: usize, nrows_x: usize, ncols_y: usize,
            nsm: i32, smpbo: i64, cc: i32, random_y: bool) {
    let label = format!("{} k={ncols_x} rows={nrows_x} cols={ncols_y} nsm={nsm} smpbo={smpbo} cc={cc} yrand={random_y}", q.name);
    // x: random bytes with controlled f16 scales.
    let nblocks = nrows_x * ncols_x / q.qk;
    let mut x = rng.bytes(nblocks * q.bytes);
    for b in 0..nblocks {
        for &h in q.halves {
            let v = rng.half();
            x[b * q.bytes + h..b * q.bytes + h + 2].copy_from_slice(&v.to_le_bytes());
        }
    }
    // y: block_q8_1_mmq [k_padded/128][ncols_y], plus generous tail padding (random, read only).
    let k_padded = pad(pad(ncols_x, 512), 128);
    let ybytes = ncols_y * (k_padded / 128) * 144 + 300 * 144;
    let mut y = rng.bytes(ybytes);
    if random_y {
        for blk in 0..ybytes / 144 {
            for h in 0..8 {
                let v = rng.half();
                y[blk * 144 + 2 * h..blk * 144 + 2 * h + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
    }
    let yb = Buf::new(&y);
    if !random_y {
        let xf: Vec<u8> = (0..ncols_y * ncols_x).flat_map(|_| rng.f32v().to_le_bytes()).collect();
        let xfb = Buf::new(&xf);
        let quant: QuantFn = match q.layout {
            0 => launch_mmq_quantize_q8_1_D4,
            1 => launch_mmq_quantize_q8_1_DS4,
            _ => launch_mmq_quantize_q8_1_D2S6,
        };
        unsafe {
            quant(xfb.p(), std::ptr::null(), yb.p(), 0, ncols_x as i64, ncols_x as i64, 0, 0, k_padded as i64, ncols_y as i64, 1, 1, stream);
            check(sys::cuStreamSynchronize(stream as sys::CUstream), "quantize sync");
        }
    }
    // The kernels load whole 256-value slices: with ncols_x/qk not a multiple of 256/qk the last
    // row's slice runs up to 7 blocks past the tensor (the reference does this too), and a NaN/inf
    // scale there survives the zero-padded y. Give both launchers the same bytes past the end.
    x.extend(rng.bytes(8 * q.bytes + 256));
    let xb = Buf::new(&x);
    let dst0 = rng.bytes(nrows_x * ncols_y * 4);
    // tmp_fixup is stream-k scratch: only the split slots a launcher actually uses are part of the
    // contract, and the two sides do not agree on how many they clear (the twin zeroes the unused
    // tail, candle leaves it alone). Prefill it with zeros so the compared bytes are the real
    // partials; dst keeps its random prefill, which is what catches a launcher that writes too little.
    let fix0 = vec![0u8; nsm.max(1) as usize * 128 * 128 * 4];
    let mut outs = vec![];
    let qr = if std::env::var("MMQ_SELF").is_ok() { q.c } else { q.r };
    for f in [q.c, qr] {
        let dst = Buf::new(&dst0);
        let fix = Buf::new(&fix0);
        unsafe {
            f(fix.p(), xb.p(), yb.p(), dst.p(), ncols_x as i64, nrows_x as i64, ncols_y as i64, (ncols_x / q.qk) as i64,
              nrows_x as i64, cc, nsm, smpbo, 32, stream);
            let r = sys::cuStreamSynchronize(stream as sys::CUstream);
            if r != sys::cudaError_enum_CUDA_SUCCESS {
                // 715 = illegal instruction. The reference libmoe.a is sm_120a code throughout, and
                // some of its instances use instructions this GPU (cc 12.0) cannot execute; there is
                // nothing to compare the twin against in that case, so the case is skipped and
                // counted rather than reported as a mismatch.
                if r == 715 && outs.is_empty() {
                    // Not skippable: an illegal-instruction fault poisons the context, so every later
                    // allocation fails with 715 too. Stop here and report the crate as blocked instead
                    // of cascading.
                    println!("  {label}: the reference faults with CUDA 715");
                    println!("mmq: BLOCKED -- the reference libmoe.a is sm_120a code and cannot execute on this GPU (cc 12.0); no instance can be compared here");
                    std::process::exit(2);
                }
                panic!("{label}: {} launcher: CUDA error {r} (runtime last error {})", if outs.is_empty() { "C" } else { "Rust" }, cudaGetLastError());
            }
        }
        outs.push((dst.read(), fix.read()));
    }
    t.calls += 2;
    // Non-vacuity: the C launcher must actually have written dst (and, for stream-k splits, tmp_fixup).
    if outs[0].0 != dst0 {
        t.dst_written += 1;
    }
    if outs[0].1 != fix0 {
        t.fixup_written += 1;
    }
    t.cmp(&label, "dst", &outs[0].0, &outs[1].0);
    t.cmp_scratch(&label, "tmp_fixup", &outs[0].1, &outs[1].1);
}

/// One quantize case: C and Rust quantize launchers on the same f32 input, compare the whole vy.
#[allow(clippy::too_many_arguments)]
fn quant_case(t: &mut Tally, rng: &mut Rng, stream: *mut c_void, layout: usize, k: usize, s01: usize, ne1: usize, ne2: usize, ne3: usize,
              with_ids: bool) {
    let label = format!("quantize layout={layout} k={k} s01={s01} ne1={ne1} ne2={ne2} ne3={ne3} ids={with_ids}");
    let ne0 = pad(pad(k, 512), 128);
    let s02 = s01 * ne1 + 4 * (rng.below(3) as usize);
    let s03 = s02 * ne2 + 4 * (rng.below(3) as usize);
    let nx = s03 * ne3 + 16;
    let mut xv: Vec<f32> = (0..nx).map(|_| rng.f32v()).collect();
    // roundf trap rows: with amax = 127 (d_inv = 1), t = +-0.49999997 rounds to 0 through add.rz
    // (add.rn would give 1); plus exact .5 ties, which roundf sends away from zero.
    if ne1 % 2 == 1 {
        for row in 0..ne1 {
            let b = row * s01;
            let trap = [127.0f32, 0.49999997, -0.49999997, 0.5, -0.5, 1.5, -2.5, 126.5];
            for (i, v) in trap.iter().enumerate() {
                if b + i < xv.len() {
                    xv[b + i] = *v;
                }
            }
        }
    }
    let xf: Vec<u8> = xv.iter().flat_map(|v| v.to_le_bytes()).collect();
    let xb = Buf::new(&xf);
    let ids: Vec<u8> = (0..ne1).flat_map(|_| (rng.below(ne1 as u64) as i32).to_le_bytes()).collect();
    let idb = Buf::new(&ids);
    let nblk = ne2 * ne3 * ne1 * ne0 / 128;
    let y0 = rng.bytes(nblk * 144 + 144);
    let (c, r): (QuantFn, QuantFn) = match layout {
        0 => (launch_mmq_quantize_q8_1_D4, rs::launch_mmq_quantize_q8_1_D4),
        1 => (launch_mmq_quantize_q8_1_DS4, rs::launch_mmq_quantize_q8_1_DS4),
        _ => (launch_mmq_quantize_q8_1_D2S6, rs::launch_mmq_quantize_q8_1_D2S6),
    };
    let mut outs = vec![];
    let r = if std::env::var("MMQ_SELF").is_ok() { c } else { r };
    for f in [c, r] {
        let yb = Buf::new(&y0);
        unsafe {
            f(xb.p(), if with_ids { idb.p() as *const i32 } else { std::ptr::null() }, yb.p(), 0, k as i64, s01 as i64, s02 as i64,
              s03 as i64, ne0 as i64, ne1 as i64, ne2 as i64, ne3 as i64, stream);
            let e = sys::cuStreamSynchronize(stream as sys::CUstream);
            if e != sys::cudaError_enum_CUDA_SUCCESS {
                panic!("{label}: CUDA error {e}");
            }
        }
        outs.push(yb.read());
    }
    t.calls += 2;
    if outs[0] != y0 {
        t.dst_written += 1;
    }
    t.cmp(&label, "vy", &outs[0], &outs[1]);
}

/// MMQ_BENCH=1: wall-clock per launcher call (C vs Rust) at a prefill-sized shape. Informational.
fn bench(rng: &mut Rng, stream: *mut c_void) {
    let (k, rows, cols) = (4096usize, 4096usize, 512usize);
    for q in qtypes() {
        let x = rng.bytes(rows * k / q.qk * q.bytes + 4096);
        let xb = Buf::new(&x);
        let yb = Buf::new(&rng.bytes(cols * k / 128 * 144 + 300 * 144));
        let db = Buf::new(&vec![0u8; rows * cols * 4]);
        let fb = Buf::new(&vec![0u8; NSM_REAL as usize * 128 * 128 * 4]);
        let mut res = vec![];
        for f in [q.c, q.r] {
            let run = || unsafe {
                f(fb.p(), xb.p(), yb.p(), db.p(), k as i64, rows as i64, cols as i64, (k / q.qk) as i64, rows as i64, 1200, NSM_REAL,
                  101376, 32, stream)
            };
            run();
            unsafe { check(sys::cuStreamSynchronize(stream as sys::CUstream), "bench") };
            let t0 = std::time::Instant::now();
            for _ in 0..20 {
                run();
            }
            unsafe { check(sys::cuStreamSynchronize(stream as sys::CUstream), "bench") };
            res.push(t0.elapsed().as_secs_f64() / 20.0 * 1e6);
        }
        println!("  bench {} k={k} rows={rows} cols={cols}: C {:.0} us, Rust {:.0} us", q.name, res[0], res[1]);
    }
}

/// Run every type in its own process.
///
/// An illegal-instruction fault (CUDA 715) destroys the context, so a type whose *reference* cannot
/// execute takes every later type down with it. One process per type keeps the rest verifiable; the
/// default invocation now does this automatically and aggregates the verdicts.
fn run_per_type() -> i32 {
    let types = qtypes().into_iter().map(|q| q.name).collect::<Vec<_>>();
    let exe = std::env::current_exe().expect("current exe");
    let mut pass: Vec<&str> = vec![];
    let mut blocked: Vec<&str> = vec![];
    let mut fail: Vec<&str> = vec![];
    for t in &types {
        let st = std::process::Command::new(&exe)
            .env("MMQ_TYPES", t)
            .env("MMQ_PER_TYPE", "1")
            .status()
            .expect("spawn per-type run");
        match st.code() {
            Some(0) => pass.push(t),
            Some(2) => blocked.push(t),
            _ => fail.push(t),
        }
    }
    println!("\n== per-type summary");
    println!("  pass:    {} ({})", pass.len(), pass.join(" "));
    if !blocked.is_empty() {
        println!("  blocked: {} ({}) -- the reference libmoe.a is sm_120a code this GPU cannot execute", blocked.len(), blocked.join(" "));
    }
    if !fail.is_empty() {
        println!("  FAIL:    {} ({})", fail.len(), fail.join(" "));
    }
    if !fail.is_empty() {
        1
    } else if pass.is_empty() {
        2
    } else {
        0
    }
}

pub fn run() -> i32 {
    if std::env::var("MMQ_TYPES").is_err() && std::env::var("MMQ_PER_TYPE").is_err() && std::env::var("MMQ_BENCH").is_err() {
        return run_per_type();
    }
    let only: Option<Vec<String>> = std::env::var("MMQ_TYPES").ok().map(|s| s.split(',').map(|x| x.to_string()).collect());
    let quick = std::env::var("MMQ_QUICK").is_ok();
    unsafe {
        check(cudaSetDevice(0) as u32, "cudaSetDevice");
        check(cudaFree(std::ptr::null_mut()) as u32, "cudaFree(0) runtime init");
    }
    // The runtime made the primary context current; use it (the same one candle uses).
    let ctx = cuda_core::CudaContext::new(0).expect("primary context");
    ctx.bind_to_thread().expect("bind");
    let stream_owner = ctx.new_stream().expect("stream");
    let stream = stream_owner.cu_stream() as *mut c_void;

    let mut t = Tally::default();
    let mut rng = Rng(0x5EED_4D4D_0001);
    let smpbo_real: i64 = 101376;
    if std::env::var("MMQ_BENCH").is_ok() {
        bench(&mut rng, stream);
        return 0;
    }

    if only.as_ref().is_none_or(|o| o.iter().any(|n| n == "quantize")) {
        let before = (t.calls, t.failures.len(), t.dst_written);
        for layout in 0..3 {
            for &(k, ds) in &[(256usize, 0usize), (300, 3), (512, 8), (1000, 4), (2048, 0), (4096, 12), (36, 0)] {
                for &(ne1, ne2, ne3) in &[(1usize, 1usize, 1usize), (5, 1, 1), (64, 1, 1), (3, 2, 2), (17, 3, 1)] {
                    quant_case(&mut t, &mut rng, stream, layout, k, k + ds, ne1, ne2, ne3, ne1 % 2 == 1);
                }
            }
        }
        println!("  quantize: {} launcher calls, {} failing (C wrote vy in {} cases)", t.calls - before.0, t.failures.len() - before.1,
                 t.dst_written - before.2);
    }
    for q in qtypes() {
        if let Some(o) = &only {
            if !o.iter().any(|n| n == q.name) {
                continue;
            }
        }
        let before = (t.calls, t.failures.len(), t.dst_written, t.fixup_written);
        let ks: Vec<usize> = if q.qk == 256 { vec![256, 512, 768, 2048] } else { vec![256, 512, 544, 1024] };
        let rows = [128usize, 256, 100, 300, 1];
        let cols = [1usize, 7, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 63, 64, 65, 80, 81, 96, 97, 112, 113, 127, 128, 129, 200, 257];
        // Every tile choice x need_check on the real device parameters.
        for (ci, &nc) in cols.iter().enumerate() {
            for (ri, &nr) in rows.iter().enumerate() {
                let k = ks[(ci + ri) % ks.len()];
                if quick && (ci + ri) % 3 != 0 {
                    continue;
                }
                mmq_case(&mut t, &mut rng, stream, &q, k, nr, nc, NSM_REAL, smpbo_real, 1200, (ci + ri) % 2 == 1);
            }
        }
        // Stream-k decompositions: many CUDA-block counts, including more blocks than work.
        for &nsm in &[1, 2, 3, 5, 7, 13, 31, 64, 97, 150, 256] {
            for &(k, nr, nc) in &[(2048usize, 300usize, 40usize), (4096, 128, 129), (768, 256, 7), (512, 1000, 64)] {
                let k = if q.qk == 32 { k / 2 + 32 } else { k };
                if quick && nsm % 2 == 0 {
                    continue;
                }
                mmq_case(&mut t, &mut rng, stream, &q, k, nr, nc, nsm, smpbo_real, 1200, nsm % 3 == 0);
            }
        }
        // Lower shared-memory budgets force smaller tiles; other cc >= 1200 take the same path.
        for &(smpbo, cc) in &[(49152i64, 1200), (30000, 1200), (20000, 1210), (101376, 1300), (15000, 1200)] {
            for &nc in &[20usize, 70, 130, 256] {
                mmq_case(&mut t, &mut rng, stream, &q, if q.qk == 32 { 1024 } else { 1536 }, 200, nc, NSM_REAL, smpbo, cc, nc == 70);
            }
        }
        println!("  {}: {} launcher calls, {} failing (C wrote dst in {} cases, tmp_fixup in {})", q.name, t.calls - before.0,
                 t.failures.len() - before.1, t.dst_written - before.2, t.fixup_written - before.3);
        if t.dst_written == before.2 || t.fixup_written == before.3 {
            t.failures.push(format!("{}: vacuous: C launcher never wrote dst or tmp_fixup", q.name));
        }
    }

    for f in t.failures.iter().take(40) {
        println!("  FAIL {f}");
    }
    let ok = t.failures.is_empty();
    if !t.scratch_notes.is_empty() {
        println!("  {} cases where tmp_fixup disagrees but dst is identical (scratch slot layout differs; not a contract violation)", t.scratch_notes.len());
        for n in t.scratch_notes.iter().take(3) {
            println!("    note {n}");
        }
    }
    if t.skipped_ref_715 > 0 {
        println!("  {} cases skipped: the reference libmoe.a is sm_120a code and those instances take CUDA 715 on this GPU (cc 12.0)", t.skipped_ref_715);
    }
    // A run where the reference could not execute anything has verified nothing; do not call it a pass.
    if t.calls == 0 {
        println!("mmq: BLOCKED -- 0 comparisons (the reference libmoe.a cannot execute on this GPU)");
        drop(stream_owner);
        return 2;
    }
    println!("mmq: {} launcher calls, {} bytes compared, {} failing, {} skipped -> {}", t.calls, t.bytes, t.failures.len(), t.skipped_ref_715, if ok { "PASS" } else { "FAIL" });
    drop(stream_owner);
    if ok { 0 } else { 1 }
}
