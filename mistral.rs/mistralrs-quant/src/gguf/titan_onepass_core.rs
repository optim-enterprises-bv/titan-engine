//! The host arithmetic of `TITAN_CPU_ONEPASS` (see titan_onepass.rs), free of CUDA so it can be tested on the CPU.
//! Batch invariance (MTP verify rows must equal the decode step they verify): every task's output is computed from
//! its own input row alone, over row chunks whose bounds depend only on `n` and the chunk size, never on how many
//! tasks share the pass, how they group by expert, or which worker runs a chunk.

#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use super::titan_cpu;

/// Bytes of one q8_1 block (32 values).
pub const Q8_1_BLOCK_BYTES: usize = 36;
/// q8_1 rows are padded to a multiple of this many values.
pub const MATRIX_ROW_PADDING: usize = 512;

/// Bytes of one q8_1 row of `k` values (padded to MATRIX_ROW_PADDING).
pub fn q8_row_bytes(k: usize) -> usize {
    k.div_ceil(MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING / 32 * Q8_1_BLOCK_BYTES
}

/// One projection of a layer's experts, as the CPU computes it.
pub trait ExpertRows: Sync {
    fn n(&self) -> usize;
    fn k(&self) -> usize;
    /// Rows `[r0, r1)` of expert `e` against one q8_1 input row.
    fn rows_into(&self, e: usize, r0: usize, r1: usize, xq_row: &[u8], out: &mut [f32]);
}

/// The missed tasks of one layer, grouped by expert: output row `i` belongs to `tasks[i]`.
pub struct Misses {
    /// (expert, first index into `tasks`, count)
    pub groups: Vec<(u32, usize, usize)>,
    pub tasks: Vec<usize>,
}

impl Misses {
    /// Tasks whose expert `miss(e)` says is not on the GPU.
    pub fn new(ids: &[u32], miss: impl Fn(u32) -> bool) -> Self {
        let mut by: Vec<(u32, usize)> = Vec::new();
        for (t, &e) in ids.iter().enumerate() {
            if miss(e) {
                by.push((e, t));
            }
        }
        by.sort_unstable();
        let mut groups = Vec::new();
        let mut tasks = Vec::with_capacity(by.len());
        for (i, &(e, t)) in by.iter().enumerate() {
            if i == 0 || by[i - 1].0 != e {
                groups.push((e, i, 0));
            }
            groups.last_mut().expect("group").2 += 1;
            tasks.push(t);
        }
        Self { groups, tasks }
    }

    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }
}

/// Raw output pointer shared by the pool's items, each writing a disjoint range.
#[derive(Clone, Copy)]
pub struct Out(pub *mut f32);
unsafe impl Send for Out {}
unsafe impl Sync for Out {}

impl Out {
    /// SAFETY: callers write disjoint ranges while the buffer outlives the pass.
    pub unsafe fn range(self, lo: usize, len: usize) -> &'static mut [f32] {
        std::slice::from_raw_parts_mut(self.0.add(lo), len)
    }
}

/// Runs `f(0..n)` in parallel and returns when every call has finished.
pub type ParFor<'a> = &'a (dyn Fn(usize, &(dyn Fn(usize) + Sync)) + Sync);

/// Gate and up rows of every missed task, items (expert, projection, row chunk): row `i` of `g` / `u` (stride `ng`)
/// belongs to `misses.tasks[i]`, whose q8_1 input is `x(task)`.
#[allow(clippy::too_many_arguments)]
pub fn gate_up_rows<'a, P: ExpertRows>(
    gate: &P,
    up: &P,
    misses: &Misses,
    x: &(dyn Fn(usize) -> &'a [u8] + Sync),
    g: Out,
    u: Out,
    chunk: usize,
    par: ParFor<'_>,
) {
    let ng = gate.n();
    let chunks = ng.div_ceil(chunk);
    par(misses.groups.len() * 2 * chunks, &|w| {
        let (gi, rest) = (w / (2 * chunks), w % (2 * chunks));
        let (p, c) = (rest / chunks, rest % chunks);
        let (e, i0, cnt) = misses.groups[gi];
        let (r0, r1) = (c * chunk, ((c + 1) * chunk).min(ng));
        let (proj, dst) = if p == 0 { (gate, g) } else { (up, u) };
        for i in i0..i0 + cnt {
            let o = unsafe { dst.range(i * ng + r0, r1 - r0) };
            proj.rows_into(e as usize, r0, r1, x(misses.tasks[i]), o);
        }
    });
}

/// Down rows of every missed task, items (expert, row chunk): row `i` of `out` (stride `nd`) from the q8_1 activation
/// `a(i)`.
pub fn down_rows<'a, P: ExpertRows>(down: &P, misses: &Misses, a: &(dyn Fn(usize) -> &'a [u8] + Sync), out: Out, chunk: usize, par: ParFor<'_>) {
    let nd = down.n();
    let chunks = nd.div_ceil(chunk);
    par(misses.groups.len() * chunks, &|w| {
        let (gi, c) = (w / chunks, w % chunks);
        let (e, i0, cnt) = misses.groups[gi];
        let (r0, r1) = (c * chunk, ((c + 1) * chunk).min(nd));
        for i in i0..i0 + cnt {
            let o = unsafe { out.range(i * nd + r0, r1 - r0) };
            down.rows_into(e as usize, r0, r1, a(i), o);
        }
    });
}

/// SwiGLU of one task's gate and up rows, requantized to q8_1 for the down projection.
pub fn activate(g: &[f32], u: &[f32], act: &mut [f32], qa: &mut [u8]) {
    titan_cpu::silu_mul(g, u, act);
    titan_cpu::quantize_q8_1(act, qa);
}

/// One layer's misses on the host: `x(task)` is the task's gate/up q8_1 input row, `out` receives the down rows,
/// `[misses.tasks.len(), down.n()]`. `per_expert`: each missed expert's whole pass on one worker (mode 2), else
/// row chunks over the pool with the SwiGLU on the caller (mode 1). Both give the same bits.
#[allow(clippy::too_many_arguments)]
pub fn onepass<'a, P: ExpertRows>(
    gate: &P,
    up: &P,
    down: &P,
    misses: &Misses,
    x: &(dyn Fn(usize) -> &'a [u8] + Sync),
    out: &mut [f32],
    chunk: usize,
    per_expert: bool,
    par: ParFor<'_>,
) {
    let m = misses.tasks.len();
    if m == 0 {
        return;
    }
    let (ng, nd) = (gate.n(), down.n());
    let qa_bytes = q8_row_bytes(down.k());
    let outp = Out(out.as_mut_ptr());
    if per_expert {
        par(misses.groups.len(), &|gi| {
            let (e, i0, cnt) = misses.groups[gi];
            let e = e as usize;
            let (mut g, mut u, mut act) = (vec![0f32; ng], vec![0f32; ng], vec![0f32; ng]);
            let mut qa = vec![0u8; qa_bytes];
            for i in i0..i0 + cnt {
                let xr = x(misses.tasks[i]);
                // the same chunk bounds as mode 1, so a row's reduction is the same
                for r0 in (0..ng).step_by(chunk) {
                    let r1 = (r0 + chunk).min(ng);
                    gate.rows_into(e, r0, r1, xr, &mut g[r0..r1]);
                    up.rows_into(e, r0, r1, xr, &mut u[r0..r1]);
                }
                activate(&g, &u, &mut act, &mut qa);
                for r0 in (0..nd).step_by(chunk) {
                    let r1 = (r0 + chunk).min(nd);
                    let o = unsafe { outp.range(i * nd + r0, r1 - r0) };
                    down.rows_into(e, r0, r1, &qa, o);
                }
            }
        });
        return;
    }
    let mut g = vec![0f32; m * ng];
    let mut u = vec![0f32; m * ng];
    gate_up_rows(gate, up, misses, x, Out(g.as_mut_ptr()), Out(u.as_mut_ptr()), chunk, par);
    let mut qa = vec![0u8; m * qa_bytes];
    let mut act = vec![0f32; ng];
    for i in 0..m {
        activate(&g[i * ng..(i + 1) * ng], &u[i * ng..(i + 1) * ng], &mut act, &mut qa[i * qa_bytes..(i + 1) * qa_bytes]);
    }
    down_rows(down, misses, &|i| &qa[i * qa_bytes..(i + 1) * qa_bytes], outp, chunk, par);
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use titan_cpu::Format;

    struct HostProj {
        fmt: Format,
        n: usize,
        k: usize,
        w: Vec<u8>,
    }

    impl ExpertRows for HostProj {
        fn n(&self) -> usize {
            self.n
        }
        fn k(&self) -> usize {
            self.k
        }
        fn rows_into(&self, e: usize, r0: usize, r1: usize, xq_row: &[u8], out: &mut [f32]) {
            let rb = self.k / self.fmt.block_values() * self.fmt.block_bytes();
            let ex = &self.w[e * self.n * rb..(e + 1) * self.n * rb];
            let rows: Vec<&[u8]> = (r0..r1).map(|r| &ex[r * rb..(r + 1) * rb]).collect();
            unsafe { titan_cpu::rows_lanes(self.fmt, &rows, xq_row, self.k, out) };
        }
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn f(&mut self) -> f32 {
            (self.next() % 2_000_001) as f32 / 1e6 - 1.0
        }
    }

    fn weights(r: &mut Rng, fmt: Format, experts: usize, n: usize, k: usize) -> Vec<u8> {
        let bb = fmt.block_bytes();
        let blocks = experts * n * k / fmt.block_values();
        let mut w: Vec<u8> = (0..blocks * bb).map(|_| r.next() as u8).collect();
        let offs: &[usize] = if fmt == Format::Q6K { &[208] } else { &[0, 2] };
        for b in 0..blocks {
            for &o in offs {
                let v = r.next();
                let h = ((((v >> 8) & 1) << 15) | ((5 + (v >> 9) % 9) << 10) | ((v >> 16) & 0x3ff)) as u16;
                w[b * bb + o..b * bb + o + 2].copy_from_slice(&h.to_le_bytes());
            }
        }
        w
    }

    fn pool(threads: usize) -> impl Fn(usize, &(dyn Fn(usize) + Sync)) + Sync {
        move |n: usize, f: &(dyn Fn(usize) + Sync)| {
            let next = std::sync::atomic::AtomicUsize::new(0);
            std::thread::scope(|s| {
                for _ in 0..threads {
                    s.spawn(|| loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= n {
                            return;
                        }
                        f(i);
                    });
                }
            });
        }
    }

    /// A token's CPU-expert output is bit-identical whether it is computed alone (decode) or among 2..8 tokens (MTP
    /// verify rows), in mode 1 and mode 2, on any worker count: the MTP == MTP-off guarantee.
    #[test]
    fn onepass_batch_invariant() {
        if !(std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")) {
            return;
        }
        let mut r = Rng(0x5eed_1234_abcd);
        let (experts, topk, ng, k) = (12usize, 8usize, 512usize, 2048usize);
        for down_fmt in [Format::Q5K, Format::Q6K] {
            let gate = HostProj { fmt: Format::Q4K, n: ng, k, w: weights(&mut r, Format::Q4K, experts, ng, k) };
            let up = HostProj { fmt: Format::Q4K, n: ng, k, w: weights(&mut r, Format::Q4K, experts, ng, k) };
            let down = HostProj { fmt: down_fmt, n: k, k: ng, w: weights(&mut r, down_fmt, experts, k, ng) };
            let xb = q8_row_bytes(k);
            let tokens = 8;
            let xq: Vec<Vec<u8>> = (0..tokens)
                .map(|_| {
                    let x: Vec<f32> = (0..k).map(|_| r.f()).collect();
                    let mut q = vec![0u8; xb];
                    titan_cpu::quantize_q8_1(&x, &mut q);
                    q
                })
                .collect();
            // routing: tokens share experts so verify batches group several tokens per expert
            let ids: Vec<u32> = (0..tokens * topk).map(|_| (r.next() % experts as u64) as u32).collect();
            let resident = |e: u32| e % 3 == 0;
            // reference: every token alone (batch 1), serial
            let serial = pool(1);
            let alone: Vec<Vec<f32>> = (0..tokens)
                .map(|t| {
                    let ids1 = &ids[t * topk..(t + 1) * topk];
                    let mis = Misses::new(ids1, |e| !resident(e));
                    let mut out = vec![0f32; mis.tasks.len() * k];
                    onepass(&gate, &up, &down, &mis, &|_| &xq[t][..], &mut out, 32, false, &serial);
                    let mut full = vec![f32::NAN; topk * k];
                    for (i, &task) in mis.tasks.iter().enumerate() {
                        full[task * k..(task + 1) * k].copy_from_slice(&out[i * k..(i + 1) * k]);
                    }
                    full
                })
                .collect();
            let mut checked = 0usize;
            for batch in [1usize, 2, 3, 5, 8] {
                for threads in [1usize, 3, 7] {
                    for per_expert in [false, true] {
                        let p = pool(threads);
                        for start in 0..=tokens - batch {
                            let ids_b = &ids[start * topk..(start + batch) * topk];
                            let mis = Misses::new(ids_b, |e| !resident(e));
                            let mut out = vec![0f32; mis.tasks.len() * k];
                            onepass(&gate, &up, &down, &mis, &|task| &xq[start + task / topk][..], &mut out, 32, per_expert, &p);
                            for (i, &task) in mis.tasks.iter().enumerate() {
                                let (tok, slot) = (start + task / topk, task % topk);
                                let want = &alone[tok][slot * k..(slot + 1) * k];
                                let got = &out[i * k..(i + 1) * k];
                                assert!(
                                    want.iter().zip(got).all(|(a, b)| a.to_bits() == b.to_bits()),
                                    "{down_fmt:?}: token {tok} slot {slot} differs at batch {batch}, {threads} threads, per_expert {per_expert}"
                                );
                                checked += 1;
                            }
                        }
                    }
                }
            }
            assert!(checked > 1000, "only {checked} rows checked");
        }
    }
}
