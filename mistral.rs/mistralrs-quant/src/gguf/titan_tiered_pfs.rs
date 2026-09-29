//! `TieredExperts` methods of prefill expert streaming (`titan_pfs`): a streamed projection runs the slot-map
//! GEMV over the resident slots and then over the staging buffer, or llama.cpp's grouped MMQ over both.

use candle_core::{
    cuda::{cudarc::driver::{LaunchConfig, PushKernelArg}, WrapErr},
    Result, Storage, Tensor,
};

use super::{titan_pfs, QInput, TieredExperts, MATRIX_ROW_PADDING, MODULE, NOT_RESIDENT, PTX};
use crate::utils::slice_ptr;

/// Pointers and shape of one slot-map GEMV launch (see `launch_gemv`).
struct GemvArgs {
    ids_ptr: u64,
    xq_ptr: u64,
    xq_len: u64,
    out_ptr: u64,
    tasks: usize,
    topk: usize,
    input_dim1: usize,
}

impl TieredExperts {
    /// The KV room (tokens) of the prompt pass about to run: long contexts get a smaller staging ring.
    pub fn set_prefill_context(tokens: usize) {
        titan_pfs::set_context(tokens);
    }

    /// Whether a forward of `rows` tokens streams this tensor's non-resident experts (`titan_pfs`).
    pub fn streams(&self, rows: usize) -> bool {
        self.pfs.is_some() && rows >= titan_pfs::min_rows()
    }

    /// `forward_q` for a prefill chunk (`streams`): the slot-map GEMV over the resident slots, then the
    /// same kernel over the staging buffer the non-resident experts were copied into. Experts left out of
    /// the stream (`TITAN_PFS_SHARE` < 1) go to the CPU twin. Bit-identical to `forward_q`; `None` when
    /// no staging buffer could be had.
    pub fn forward_streamed(&self, q: &QInput, ids: &Tensor, ids_host: &[u32]) -> Result<Option<Tensor>> {
        let Some(src) = &self.pfs else {
            return Ok(None);
        };
        let (batch, input_dim1, k) = (q.batch, q.input_dim1, q.k);
        let (id_batch, topk) = ids.dims2()?;
        if k != self.k || id_batch != batch || ids_host.len() != batch * topk {
            candle_core::bail!("titan tiered experts: input [{batch}, {input_dim1}, {k}] / ids {:?} vs k={}", ids.dims(), self.k);
        }
        let Some(st) = titan_pfs::acquire(&self.dev, src, batch * topk)? else {
            return Ok(None);
        };
        let ids = ids.contiguous()?;
        let (ids_storage, ids_layout) = ids.storage_and_layout();
        let Storage::Cuda(ids_cuda) = &*ids_storage else {
            candle_core::bail!("titan tiered experts: ids not on CUDA");
        };
        let ids_slice = ids_cuda.as_cuda_slice::<u32>()?;
        let tasks = batch * topk;
        let mut out = self.dev.alloc_zeros::<f32>(tasks * self.n)?;
        let cache = self.cache.lock().unwrap();
        {
            let (ids_ptr, g1) = slice_ptr(ids_slice, ids_layout.start_offset());
            let (xq_ptr, g2) = slice_ptr(&q.xq, 0);
            let (w_ptr, g3) = slice_ptr(&cache.slots, 0);
            let (map_ptr, g4) = slice_ptr(&cache.slot_map, 0);
            let (stage_ptr, g5) = slice_ptr(&src.stage_map, 0);
            let (out_ptr, g6) = slice_ptr(&out, 0);
            let a = GemvArgs { ids_ptr, xq_ptr, xq_len: (q.xq.len() / 4) as u64, out_ptr, tasks, topk, input_dim1 };
            self.launch_gemv(&a, w_ptr, (cache.slots.len() / 4) as u64, map_ptr)?;
            self.launch_gemv(&a, st.ptr, (st.len / 4) as u64, stage_ptr)?;
            drop((g1, g2, g3, g4, g5, g6));
        }
        let routed = if self.proj == "gate" {
            let mut seen = vec![false; self.num_experts];
            let r = ids_host
                .iter()
                .filter(|&&e| src.stage_index[e as usize] != NOT_RESIDENT && !std::mem::replace(&mut seen[e as usize], true))
                .count();
            Some((r, src.staged))
        } else {
            None
        };
        titan_pfs::release(&self.dev, st, routed)?;
        if src.staged + cache.owner.len() < self.num_experts {
            let served: Vec<u32> = (0..self.num_experts)
                .map(|e| if cache.map[e] == NOT_RESIDENT && src.stage_index[e] == NOT_RESIDENT { NOT_RESIDENT } else { 0 })
                .collect();
            self.cpu_misses(&mut out, &served, ids_host, &q.xq, q.host.as_deref(), input_dim1, topk, q.q8_row_bytes)?;
        }
        Ok(Some(self.wrap(out, batch, topk)))
    }

    /// Whether streamed forwards of this tensor run the grouped MMQ (`TITAN_PFS_KERNEL=mmq`, K-quants).
    pub fn streams_mmq(&self) -> bool {
        titan_pfs::mmq() && crate::gguf::fast_mmq::supports(self.dtype)
    }

    /// `forward_streamed` through llama.cpp's grouped MMQ (`streams_mmq`): the dispatch sorts the tasks by
    /// expert over the resident slots and the staging buffer, then one MMQ launch per buffer. `x` is
    /// `[rows, k]` f32 and task t reads row `t / topk`, or row t with `per_task`. Returns `[tasks, 1, n]`
    /// in task order; `None` when no staging buffer could be had.
    pub fn forward_streamed_mmq(&self, x: &Tensor, ids_host: &[u32], topk: usize, per_task: bool) -> Result<Option<Tensor>> {
        let mut out = None;
        let ok = self.forward_streamed_mmq_pieces(x, ids_host, topk, per_task, ids_host.len(), |_, y| {
            out = Some(y);
            Ok(())
        })?;
        Ok(if ok { out } else { None })
    }

    /// `forward_streamed_mmq` in pieces of `piece_tasks` tasks (a multiple of `topk`) against one staged copy
    /// of the experts: acquire once, one dispatch and MMQ pass per piece, release once. Each piece's
    /// `[piece, 1, n]` output goes to `each(first_task, out)` before the next piece runs, so a big prompt chunk
    /// never holds all its tasks' outputs at once. `false` when no staging buffer could be had.
    pub fn forward_streamed_mmq_pieces(
        &self,
        x: &Tensor,
        ids_host: &[u32],
        topk: usize,
        per_task: bool,
        piece_tasks: usize,
        mut each: impl FnMut(usize, Tensor) -> Result<()>,
    ) -> Result<bool> {
        let Some(src) = &self.pfs else {
            return Ok(false);
        };
        let tasks = ids_host.len();
        let (rows, k) = x.dims2()?;
        let want_rows = if per_task { tasks } else { tasks / topk.max(1) };
        if k != self.k || rows != want_rows || piece_tasks == 0 || (!per_task && piece_tasks % topk.max(1) != 0) {
            candle_core::bail!(
                "titan tiered experts: mmq input {:?} for {tasks} tasks (topk {topk}, pieces of {piece_tasks}) vs k={}",
                x.dims(),
                self.k
            );
        }
        let st = if src.staged > 0 {
            let Some(st) = titan_pfs::acquire(&self.dev, src, tasks)? else {
                return Ok(false);
            };
            Some(st)
        } else {
            None
        };
        let cache = self.cache.lock().unwrap();
        let resident = cache.owner.len();
        let slots = resident + src.staged;
        // dispatch slot: the resident slot, else resident + staging index; None = CPU twin
        let slot_of = |e: usize| {
            if cache.map[e] != NOT_RESIDENT {
                Some(cache.map[e] as usize)
            } else if src.stage_index[e] != NOT_RESIDENT {
                Some(resident + src.stage_index[e] as usize)
            } else {
                None
            }
        };
        let routed = (self.proj == "gate").then(|| {
            let mut used = vec![false; slots];
            for &e in ids_host {
                if let Some(v) = slot_of(e as usize) {
                    used[v] = true;
                }
            }
            (used[resident..].iter().filter(|&&u| u).count(), src.staged)
        });
        let mut t0 = 0;
        while t0 < tasks {
            let t1 = (t0 + piece_tasks).min(tasks);
            let ids = &ids_host[t0..t1];
            let ptasks = t1 - t0;
            let xp = if tasks == ptasks {
                x.clone()
            } else if per_task {
                x.narrow(0, t0, ptasks)?
            } else {
                x.narrow(0, t0 / topk, ptasks / topk)?
            };
            let mut counts = vec![0u32; slots];
            for &e in ids {
                if let Some(v) = slot_of(e as usize) {
                    counts[v] += 1;
                }
            }
            let mut bounds = vec![0u32; slots + 1];
            for v in 0..slots {
                bounds[v + 1] = bounds[v] + counts[v];
            }
            let total = bounds[slots] as usize;
            let mut next = bounds.clone();
            let (mut ids_src, mut ids_dst) = (vec![0u32; total], vec![0u32; total]);
            for (t, &e) in ids.iter().enumerate() {
                if let Some(v) = slot_of(e as usize) {
                    let j = next[v] as usize;
                    next[v] += 1;
                    ids_src[j] = (if per_task { t } else { t / topk }) as u32;
                    ids_dst[j] = t as u32;
                }
            }
            let out = self.dev.alloc_zeros::<f32>(ptasks * self.n)?;
            if total > 0 {
                let most = |r: std::ops::Range<usize>| counts[r].iter().copied().max().unwrap_or(0) as usize;
                let parts = [
                    crate::gguf::fast_mmq::GroupedPart { weight: cache.slots_ptr, first: 0, count: resident, ncols_max: most(0..resident) },
                    crate::gguf::fast_mmq::GroupedPart {
                        weight: st.as_ref().map_or(0, |st| st.ptr),
                        first: resident,
                        count: src.staged,
                        ncols_max: most(resident..slots),
                    },
                ];
                let ids_src = self.dev.clone_htod(&ids_src)?;
                let ids_dst = self.dev.clone_htod(&ids_dst)?;
                let bounds = self.dev.clone_htod(&bounds)?;
                crate::gguf::fast_mmq::grouped_parts(self.dtype, self.n, &xp, &ids_src, &ids_dst, &bounds, total, &parts, &out, &self.dev)?;
            }
            let mut out = out;
            if total < ptasks {
                let served: Vec<u32> = (0..self.num_experts).map(|e| if slot_of(e).is_some() { 0 } else { NOT_RESIDENT }).collect();
                let prows = xp.dim(0)?;
                let x3 = if per_task { xp.reshape((ptasks / topk, topk, k))? } else { xp.reshape((prows, 1, k))? };
                let q = self.quantize_input(&x3)?;
                self.cpu_misses(&mut out, &served, ids, &q.xq, None, q.input_dim1, topk, q.q8_row_bytes)?;
            }
            each(t0, self.wrap(out, ptasks, 1))?;
            t0 = t1;
        }
        drop(cache);
        if let Some(st) = st {
            titan_pfs::release(&self.dev, st, routed)?;
        }
        Ok(true)
    }

    /// One slot-map GEMV launch over the weights at `w_ptr` through `map_ptr` (expert -> slot there).
    fn launch_gemv(&self, a: &GemvArgs, w_ptr: u64, w_len: u64, map_ptr: u64) -> Result<()> {
        let (name, warp_rows) = self.gemv_kernel();
        let k_padded = self.k.div_ceil(MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;
        let (kx, kp) = (self.k as u32, k_padded as u32);
        let (ids_len, map_len) = (a.tasks as u64, self.num_experts as u64);
        let (n, topk_u, dim1) = (self.n as u32, a.topk as u32, a.input_dim1 as u32);
        let gemv = self.dev.get_or_load_custom_func(name, MODULE, PTX)?;
        let mut b = gemv.builder();
        b.arg(&w_ptr).arg(&w_len).arg(&a.xq_ptr).arg(&a.xq_len).arg(&a.ids_ptr).arg(&ids_len)
            .arg(&map_ptr).arg(&map_len).arg(&a.out_ptr).arg(&n).arg(&kx).arg(&kp).arg(&topk_u).arg(&dim1);
        unsafe {
            let blocks = if warp_rows { self.n.div_ceil(4) * a.tasks } else { self.n * a.tasks };
            b.launch(LaunchConfig { grid_dim: (blocks as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 })
        }
        .w()?;
        Ok(())
    }
}
