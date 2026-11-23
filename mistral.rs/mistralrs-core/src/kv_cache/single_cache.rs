use candle_core::{Result, Tensor};

use super::NormalCache;

const HOST_GROW_DIV: usize = 4;

#[derive(Debug, Clone)]
pub struct SingleCacheSnapshot {
    pub current_seq_len: usize,
}

#[derive(Debug, Clone)]
pub struct SingleCache {
    // all_data is an option on a Tensor, this makes it possible to only create the actual tensor
    // on the first call where the batch size is easily known.
    // Also this makes it safe to clone a KvCache that has been reset (as in it will not share
    // its internal state with the cloned instance).
    pub all_data: Option<Tensor>,
    pub dim: usize,
    pub current_seq_len: usize,
    pub capacity_seq_len: usize,
    pub max_seq_len: usize,
    /// TITAN_KV_SPILL split attention: positions from `at` on live in a host buffer, the first `at` stay in `all_data`.
    pub host_split: Option<HostSplit>,
}

/// The host part of a split cache: positions `at..` in `data` (CPU, room for `cap` positions).
#[derive(Debug, Clone)]
pub struct HostSplit {
    pub at: usize,
    pub data: Option<Tensor>,
    pub cap: usize,
}

impl SingleCache {
    /// From here on, positions `at..` go to host memory (`at` >= the current length is rounded down to it).
    pub fn split_at(&mut self, at: usize) {
        if self.host_split.is_none() {
            let at = at.max(self.current_seq_len).min(self.capacity_seq_len);
            self.host_split = Some(HostSplit { at, data: None, cap: 0 });
        }
    }

    /// The host part's current positions (`None` when unsplit or empty).
    pub fn host_data(&self) -> Result<Option<Tensor>> {
        match &self.host_split {
            Some(h) if self.current_seq_len > h.at => match &h.data {
                Some(d) => Ok(Some(d.narrow(self.dim, 0, self.current_seq_len - h.at)?)),
                None => Ok(None),
            },
            _ => Ok(None),
        }
    }

    pub fn new(dim: usize, max_seq_len: usize, capacity_seq_len: usize) -> Self {
        Self {
            host_split: None,
            all_data: None,
            dim,
            current_seq_len: 0,
            max_seq_len,
            capacity_seq_len,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn all_data(&self) -> Option<&Tensor> {
        self.all_data.as_ref()
    }

    pub fn current_data(&self) -> Result<Option<Tensor>> {
        // a split cache: the device part only (`host_data` has the rest)
        let len = match &self.host_split {
            Some(h) => self.current_seq_len.min(h.at),
            None => self.current_seq_len,
        };
        let data = match self.all_data.as_ref() {
            None => None,
            Some(d) => Some(d.narrow(self.dim, 0, len)?),
        };
        Ok(data)
    }

    pub fn snapshot(&self) -> SingleCacheSnapshot {
        SingleCacheSnapshot {
            current_seq_len: self.current_seq_len,
        }
    }

    pub fn can_append_from_snapshot(
        &self,
        snapshot: &SingleCacheSnapshot,
        append_len: usize,
    ) -> bool {
        snapshot.current_seq_len == self.current_seq_len
            && snapshot
                .current_seq_len
                .checked_add(append_len)
                .is_some_and(|len| len <= self.max_seq_len)
    }

    pub fn rollback_to(&mut self, keep_len: usize) -> candle_core::Result<()> {
        self.set_len(keep_len)
    }

    /// An empty cache gives its buffer back and takes room for `len` positions at the next append.
    pub fn shrink_empty_to(&mut self, len: usize) {
        if self.current_seq_len == 0 && len < self.capacity_seq_len {
            self.all_data = None;
            self.capacity_seq_len = len;
        }
    }

    pub fn reset(&mut self) {
        self.current_seq_len = 0;
        self.host_split = None;
        // a preallocated or dynamic CUDA buffer (TITAN_KV_PREALLOC / TITAN_KV_DYNAMIC) is kept: only its first
        // `current_seq_len` rows are ever read, and reallocating it fragments the pool
        if (self.capacity_seq_len == self.max_seq_len || super::cuda_kv_dynamic())
            && super::cuda_kv_prealloc()
            && self.all_data.as_ref().is_some_and(|d| d.device().is_cuda())
        {
            return;
        }
        self.all_data = None;
        // a reset cache starts small again: a capacity grown (or reserved) for an earlier, longer
        // sequence would otherwise be allocated in full by the next sequence's first append
        self.capacity_seq_len = self.capacity_seq_len.min(NormalCache::CACHE_GROW_SIZE);
    }

    pub fn try_set_len(&self, len: usize) -> candle_core::Result<()> {
        let room = match &self.host_split {
            Some(h) => h.at + h.cap,
            None => self.capacity_seq_len,
        };
        if len > room {
            candle_core::bail!(
                "kv-cache: requested length ({}) exceeds current capacity ({})",
                len,
                room
            );
        }
        Ok(())
    }

    pub fn set_len(&mut self, len: usize) -> candle_core::Result<()> {
        self.try_set_len(len)?;
        self.current_seq_len = len;
        Ok(())
    }

    /// Room for `len` positions (at most `max_seq_len`) in one allocation now, instead of one
    /// reallocation per `CACHE_GROW_SIZE` step as appends arrive (a chunked prompt knows its length up
    /// front; the outgrown buffers fragment the device pool).
    pub fn reserve(&mut self, len: usize) -> Result<()> {
        if self.host_split.is_some() {
            return Ok(());
        }
        let len = len.min(self.max_seq_len);
        if len <= self.capacity_seq_len {
            return Ok(());
        }
        if let Some(ad) = &self.all_data {
            let mut shape = ad.dims().to_vec();
            shape[self.dim] = len;
            let grown = kv_zeros(shape, ad.dtype(), ad.device())?;
            grown.slice_set(ad, self.dim, 0)?;
            self.all_data = Some(grown);
        }
        self.capacity_seq_len = len;
        Ok(())
    }

    pub fn append(&mut self, src: &Tensor) -> Result<()> {
        if let Some(at) = self.host_split.as_ref().map(|h| h.at) {
            return self.append_split(src, at);
        }
        self.append_unsplit(src)
    }

    /// A split cache's append: rows below `at` into the device buffer, the rest into the host buffer (grown by a
    /// quarter at a time).
    fn append_split(&mut self, src: &Tensor, at: usize) -> Result<()> {
        let n = src.dim(self.dim)?;
        let dev_rows = at.saturating_sub(self.current_seq_len).min(n);
        if dev_rows > 0 {
            if self.all_data.is_none() {
                // split before its first append (a fresh long prompt): the device part's buffer, `at` positions
                let mut shape = src.dims().to_vec();
                shape[self.dim] = at;
                self.all_data = Some(kv_zeros(shape, src.dtype(), src.device())?);
                self.capacity_seq_len = at;
            }
            let ad = self.all_data.as_mut().expect("split cache has its device buffer");
            ad.slice_set(&src.narrow(self.dim, 0, dev_rows)?.contiguous()?, self.dim, self.current_seq_len)?;
        }
        let host_rows = n - dev_rows;
        if host_rows > 0 {
            let start = self.current_seq_len + dev_rows - at;
            let rows = src.narrow(self.dim, dev_rows, host_rows)?.to_device(&candle_core::Device::Cpu)?;
            let h = self.host_split.as_mut().expect("split");
            let need = start + host_rows;
            if need > h.cap || h.data.is_none() {
                let cap = need.max(h.cap + h.cap / HOST_GROW_DIV).next_multiple_of(NormalCache::CACHE_GROW_SIZE);
                let mut shape = rows.dims().to_vec();
                shape[self.dim] = cap;
                let grown = Tensor::zeros(shape, rows.dtype(), &candle_core::Device::Cpu)?;
                if let Some(old) = &h.data {
                    grown.slice_set(&old.narrow(self.dim, 0, start)?.contiguous()?, self.dim, 0)?;
                }
                h.data = Some(grown);
                h.cap = cap;
            }
            h.data.as_ref().expect("grown").slice_set(&rows.contiguous()?, self.dim, start)?;
        }
        self.current_seq_len += n;
        Ok(())
    }

    fn append_unsplit(&mut self, src: &Tensor) -> Result<()> {
        let seq_len = src.dim(self.dim)?;
        // This doesn't seem very idiomatic but because the creation can fail, it's tricky to use
        // self.all_data.get_or_insert_with.
        if self.all_data.is_none() {
            if src.device().is_cuda() && super::cuda_kv_prealloc() {
                self.capacity_seq_len = if super::cuda_kv_dynamic() {
                    self.capacity_seq_len.max(super::cuda_kv_base()).min(self.max_seq_len)
                } else {
                    self.max_seq_len
                };
            }
            let mut shape = src.dims().to_vec();
            shape[self.dim] = self.capacity_seq_len;
            let ad = kv_zeros(shape, src.dtype(), src.device())?;
            self.all_data = Some(ad);
        };

        // Host-RAM KV (TITAN_KV_HOST) stores F8E4M3 on the CPU; a buffer cloned in as BF16 (a batched
        // template from clone_in_cache) must take the appended dtype/device before the slice_set. The
        // appended prefix is what carries the real KV, so reconciling the whole buffer is lossless here.
        if let Some(ad) = &self.all_data {
            if ad.dtype() != src.dtype() || !ad.device().same_device(src.device()) {
                self.all_data = Some(ad.to_dtype(src.dtype())?.to_device(src.device())?);
            }
        }

        // Expand kv cache
        if self.current_seq_len + seq_len > self.capacity_seq_len {
            let diff = self.current_seq_len + seq_len - self.capacity_seq_len;
            // a host-RAM cache can reach GiBs: grow by a quarter so the copies stay amortised O(len)
            let step = if src.device().is_cpu() {
                (self.capacity_seq_len / HOST_GROW_DIV)
                    .next_multiple_of(NormalCache::CACHE_GROW_SIZE)
                    .max(NormalCache::CACHE_GROW_SIZE)
            } else if super::cuda_kv_dynamic() {
                // a dynamic CUDA cache normally grew ahead of this (`ensure_kv_room`): if not, a whole bucket at once
                crate::models::quantized_qwen35_moe::KV_RESERVE_STEP
            } else {
                NormalCache::CACHE_GROW_SIZE
            };
            let n_blocks_needed = diff.div_ceil(step);
            self.capacity_seq_len = (self.capacity_seq_len + n_blocks_needed * step)
                .min(self.max_seq_len)
                .max(self.current_seq_len + seq_len);
            if self.capacity_seq_len > self.max_seq_len {
                candle_core::bail!(
                    "kv-cache: requested capacity ({}) above max seq len ({})",
                    self.capacity_seq_len,
                    self.max_seq_len
                )
            }
            let mut shape = src.dims().to_vec();
            shape[self.dim] = self.capacity_seq_len;
            let ad = kv_zeros(shape, src.dtype(), src.device())?;
            ad.slice_set(self.all_data.as_ref().unwrap(), self.dim, 0)?;
            self.all_data = Some(ad);
        }

        let ad = self.all_data.as_mut().unwrap();

        ad.slice_set(src, self.dim, self.current_seq_len)?;
        self.current_seq_len += seq_len;
        Ok(())
    }
}

/// A KV buffer: from titan's long-lived CUDA pool (`titan_static_pool`) when the cache is a titan one
/// (TITAN_KV_PREALLOC): resizing it must not leave holes among the step temporaries of the default pool.
fn kv_zeros(shape: Vec<usize>, dtype: candle_core::DType, device: &candle_core::Device) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if let (candle_core::Device::Cuda(dev), true) = (device, super::cuda_kv_prealloc()) {
        use candle_core::cuda::CudaStorage;
        use candle_core::DType;
        let n: usize = shape.iter().product();
        let pool = mistralrs_quant::titan_static_pool::alloc_zeros::<u8>;
        let st = match dtype {
            DType::F8E4M3 => CudaStorage::wrap_cuda_slice(mistralrs_quant::titan_static_pool::alloc_zeros::<float8::F8E4M3>(dev, n)?, dev.clone()),
            DType::BF16 => CudaStorage::wrap_cuda_slice(mistralrs_quant::titan_static_pool::alloc_zeros::<half::bf16>(dev, n)?, dev.clone()),
            DType::F16 => CudaStorage::wrap_cuda_slice(mistralrs_quant::titan_static_pool::alloc_zeros::<half::f16>(dev, n)?, dev.clone()),
            DType::F32 => CudaStorage::wrap_cuda_slice(mistralrs_quant::titan_static_pool::alloc_zeros::<f32>(dev, n)?, dev.clone()),
            _ => {
                let _ = pool;
                return Tensor::zeros(shape, dtype, device);
            }
        };
        return Ok(Tensor::from((candle_core::Storage::Cuda(st), candle_core::Shape::from(shape))));
    }
    Tensor::zeros(shape, dtype, device)
}
