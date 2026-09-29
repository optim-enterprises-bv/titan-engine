//! `TITAN_CPU_ONEPASS`: a MoE layer's CPU misses as one host pass. For each missed (token, expert) task the CPU runs
//! gate/up, SwiGLU (candle's formula with the host's `exp`), the q8_1 requantization (the GPU kernel's layout and
//! rounding) and down, with no GPU round trip between the projections. Only the SiLU's `exp` differs from the GPU,
//! so an expert's output may move by a rounding step when it moves between the GPU and the CPU (the policy since
//! 2026-09-29: statistical gates, not bit-identity). The arithmetic is in `titan_onepass_core` and is batch invariant,
//! so MTP verify rows stay equal to MTP-off decode. Mode 1 splits each projection's rows over the pool; mode 2 gives
//! each missed expert (all its tokens) to one core for the whole pass (same bits).

use candle_core::{Result, Storage, Tensor};
use rayon::prelude::*;

pub(super) use super::super::titan_onepass_core::{q8_row_bytes, Misses, Out};
use super::super::titan_onepass_core::{self as core, ExpertRows};
use super::{miss_pool, rows_per_chunk, titan_cpu, titan_pfs, Cache, Format, Policy, TieredExperts, NOT_RESIDENT};
use crate::utils::slice_ptr;

/// `TITAN_CPU_ONEPASS`: 0 off, 1 row-split passes, 2 one core per missed expert.
pub(crate) fn mode() -> u32 {
    static M: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *M.get_or_init(|| std::env::var("TITAN_CPU_ONEPASS").ok().and_then(|v| v.parse().ok()).unwrap_or(0))
}

fn par_for(n: usize, f: &(dyn Fn(usize) + Sync)) {
    match miss_pool() {
        Some(pool) => pool.run(n, f),
        None => (0..n).into_par_iter().for_each(f),
    }
}

impl ExpertRows for TieredExperts {
    fn n(&self) -> usize {
        self.n
    }
    fn k(&self) -> usize {
        self.k
    }
    fn rows_into(&self, e: usize, r0: usize, r1: usize, xq_row: &[u8], out: &mut [f32]) {
        let fmt = self.format();
        let row_bytes = self.expert_bytes / self.n;
        let expert = self.host_expert(e);
        let row_of = |r: usize| &expert[r * row_bytes..][..row_bytes];
        let simd = std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma");
        if simd && matches!(fmt, Format::Q4K | Format::Q5K | Format::Q6K | Format::IQ4NL | Format::MXFP4 | Format::NVFP4) {
            let rows: Vec<&[u8]> = (r0..r1).map(row_of).collect();
            unsafe { titan_cpu::rows_lanes(fmt, &rows, xq_row, self.k, out) };
            return;
        }
        let mut r = r0;
        if simd {
            while r + 8 <= r1 {
                let rows: [&[u8]; 8] = std::array::from_fn(|i| row_of(r + i));
                out[r - r0..r - r0 + 8].copy_from_slice(&unsafe { titan_cpu::rows8(fmt, &rows, xq_row, self.k) });
                r += 8;
            }
        }
        for (o, r) in out[r - r0..].iter_mut().zip(r..r1) {
            *o = titan_cpu::row(fmt, row_of(r), xq_row, self.k);
        }
    }
}

impl TieredExperts {
    /// Gate and up rows of every missed task (two-phase doorbell): row `i` of `g` / `u` belongs to `misses.tasks[i]`.
    pub(super) fn gate_up_rows<'a>(gate: &TieredExperts, up: &TieredExperts, misses: &Misses, x: &(dyn Fn(usize) -> &'a [u8] + Sync), g: Out, u: Out, _ng: usize) {
        core::gate_up_rows(gate, up, misses, x, g, u, rows_per_chunk(), &par_for);
    }

    /// Down rows of every missed task from the q8_1 activation `a(i)` (two-phase doorbell).
    pub(super) fn down_rows<'a>(down: &TieredExperts, misses: &Misses, a: &(dyn Fn(usize) -> &'a [u8] + Sync), out: Out, _nd: usize) {
        core::down_rows(down, misses, a, out, rows_per_chunk(), &par_for);
    }

    /// Whether gate, up and down may take the one-pass path: static placement with the same experts resident in all
    /// three (so a task misses all three or none), no fingerprint slots, the down input is the gate/up output.
    pub(crate) fn onepass_ok(gate: &TieredExperts, up: &TieredExperts, down: &TieredExperts) -> bool {
        let projs = [gate, up, down];
        if projs.iter().any(|p| p.policy != Policy::Static || p.flex > 0) || gate.n != up.n || down.k != gate.n || gate.k != up.k {
            return false;
        }
        let maps: Vec<std::sync::MutexGuard<'_, Cache>> = projs.iter().map(|p| p.cache.lock().unwrap()).collect();
        maps[0].map == maps[1].map && maps[0].map == maps[2].map
    }

    /// Whether every expert of this projection is on the GPU.
    pub(crate) fn all_resident(&self) -> bool {
        self.cache.lock().unwrap().owner.len() == self.num_experts
    }

    /// The CPU side of one layer's misses: `xq` holds the gate/up q8_1 input rows (`xq_row_bytes` each; task t reads
    /// row t / topk when `input_dim1` is 1), `out` receives the down rows, `[misses.tasks.len(), down.n]`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn onepass(
        gate: &TieredExperts,
        up: &TieredExperts,
        down: &TieredExperts,
        misses: &Misses,
        xq: &[u8],
        xq_row_bytes: usize,
        input_dim1: usize,
        topk: usize,
        out: &mut [f32],
    ) {
        if misses.is_empty() {
            return;
        }
        let t_mon = std::time::Instant::now();
        let x = |t: usize| {
            let r = if input_dim1 == 1 { t / topk } else { t };
            &xq[r * xq_row_bytes..(r + 1) * xq_row_bytes]
        };
        core::onepass(gate, up, down, misses, &x, out, rows_per_chunk(), mode() == 2, &par_for);
        crate::titan_monitor::tiered_miss_ns(t_mon.elapsed().as_nanos() as u64);
    }
}

impl TieredExperts {
    /// `TITAN_CPU_ONEPASS` without the doorbell: one routing-id sync (plus the input fetch when something misses),
    /// every GPU launch of the layer ahead of the CPU pass, then the down rows uploaded over the GPU's zeros.
    #[allow(clippy::type_complexity)]
    pub(super) fn forward_onepass_sync(
        gate: &TieredExperts,
        up: &TieredExperts,
        down: &TieredExperts,
        xs: &Tensor,
        indices: &Tensor,
        preds: &[([&TieredExperts; 3], usize, Tensor)],
        act: &dyn Fn(&Tensor, &Tensor) -> Result<Tensor>,
    ) -> Result<Tensor> {
        let (batch, topk) = indices.dims2()?;
        if gate.pfs.is_some() {
            titan_pfs::idle(&gate.dev, batch)?;
        }
        let qx = gate.quantize_input(xs)?;
        let ids_host = if preds.is_empty() {
            Self::routing_to_host(indices)?
        } else {
            let mut all = vec![indices.flatten_all()?];
            for (_, _, p) in preds {
                all.push(p.flatten_all()?);
            }
            let mut v = Self::routing_to_host(&Tensor::cat(&all, 0)?)?;
            let mut rest = v.split_off(indices.elem_count());
            for (projs, depth, p) in preds {
                let tail = rest.split_off(p.elem_count());
                Self::lookahead(projs, *depth, &rest);
                rest = tail;
            }
            v
        };
        for p in [gate, up, down] {
            p.prefetch(&ids_host);
        }
        let misses = {
            let mut c = gate.cache.lock().unwrap();
            if batch == 1 {
                gate.score_predictions(&c.map, &ids_host);
            }
            let m = Misses::new(&ids_host, |e| c.map[e as usize] == NOT_RESIDENT);
            gate.touch_and_admit(&mut c, &ids_host, batch)?;
            m
        };
        let xq_host = if misses.is_empty() { Vec::new() } else { gate.dev.clone_dtoh(&qx.xq)? };
        let ids = indices.contiguous()?;
        let (st, lay) = ids.storage_and_layout();
        let Storage::Cuda(c) = &*st else {
            candle_core::bail!("titan tiered experts: ids not on CUDA");
        };
        let (ids_ptr, g) = slice_ptr(c.as_cuda_slice::<u32>()?, lay.start_offset());
        let og = gate.gemv_launch(&gate.cache.lock().unwrap(), &qx, ids_ptr, topk)?;
        let ou = up.gemv_launch(&up.cache.lock().unwrap(), &qx, ids_ptr, topk)?;
        let a = act(&gate.wrap(og, batch, topk), &up.wrap(ou, batch, topk))?;
        let qa = down.quantize_input(&a)?;
        let mut od = down.gemv_launch(&down.cache.lock().unwrap(), &qa, ids_ptr, topk)?;
        drop(g);
        if !misses.is_empty() {
            let nd = down.n;
            let mut rows = vec![0f32; misses.tasks.len() * nd];
            Self::onepass(gate, up, down, &misses, &xq_host, qx.q8_row_bytes, qx.input_dim1, topk, &mut rows);
            down.upload_rows(&mut od, misses.tasks.iter().enumerate().map(|(i, &t)| (t, 0, rows[i * nd..(i + 1) * nd].to_vec())))?;
        }
        Ok(down.wrap(od, batch, topk))
    }
}
