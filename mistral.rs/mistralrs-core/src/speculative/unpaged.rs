//! Speculative verification without PagedAttention (titan-engine M6, GGUF qwen35moe + MTP).
//!
//! The target model's own caches hold the verification rows after the forward: a model-provided
//! [`SpeculativeRollback`] keeps the accepted prefix (attention KV truncation plus, for hybrid
//! models, the recurrent state saved after the last accepted row), and the per-sequence KV copies
//! made by the cache manager's `clone_out_cache` are trimmed to the same length.

use std::sync::Arc;

use candle_core::{Device, Result};

use crate::device_map::DeviceMapper;
use crate::pipeline::text_models_inputs_processor::InputMetadata;
use crate::sequence::Sequence;

use super::cache::{SpeculativeCacheAccess, SpeculativeCacheGuard};
use super::proposer::SpeculativeKvCache;

/// Model side of an unpaged rollback.
pub trait SpeculativeRollback: Send + Sync {
    /// Keep the first `keep_len` positions of the sequence that was just verified.
    fn keep(&self, keep_len: usize) -> Result<()>;
}

pub struct UnpagedSpeculativeCacheAccess {
    rollback: Arc<dyn SpeculativeRollback>,
}

impl UnpagedSpeculativeCacheAccess {
    pub fn new(rollback: Arc<dyn SpeculativeRollback>) -> Self {
        Self { rollback }
    }
}

pub struct UnpagedSpeculativeCacheGuard {
    rollback: Arc<dyn SpeculativeRollback>,
    reserved_len: usize,
}

impl SpeculativeCacheGuard for UnpagedSpeculativeCacheGuard {
    fn commit(&mut self) -> Result<()> {
        self.rollback.keep(self.reserved_len)
    }

    fn rollback_to(&mut self, keep_len: usize) -> Result<()> {
        self.rollback.keep(keep_len.min(self.reserved_len))
    }
}

impl SpeculativeCacheAccess for UnpagedSpeculativeCacheAccess {
    type Guard = UnpagedSpeculativeCacheGuard;

    fn begin(
        &self,
        seq_id: usize,
        base_len: usize,
        verify_len: usize,
    ) -> Result<Option<Self::Guard>> {
        Ok(Some(self.guard_for_reserved(seq_id, base_len, verify_len)))
    }

    fn guard_for_reserved(&self, _seq_id: usize, base_len: usize, verify_len: usize) -> Self::Guard {
        UnpagedSpeculativeCacheGuard {
            rollback: self.rollback.clone(),
            reserved_len: base_len + verify_len,
        }
    }

    fn make_verify_input_metadata(
        &self,
        _verify_tokens: &[u32],
        _seq_id: usize,
        _base_len: usize,
        _device: &Device,
        _mapper: &dyn DeviceMapper,
    ) -> Result<InputMetadata> {
        candle_core::bail!(
            "unpaged speculation verifies staged tokens through the regular decode inputs"
        )
    }

    fn proposer_cache(&self, _sequences: &[&Sequence]) -> Result<SpeculativeKvCache<'_>> {
        Ok(SpeculativeKvCache::Unpaged)
    }

    fn finish_verification(
        &self,
        guard: &mut Self::Guard,
        seq: &mut Sequence,
        keep_len: usize,
        accepted_all: bool,
    ) -> Result<()> {
        if accepted_all {
            guard.commit()?;
        } else {
            guard.rollback_to(keep_len)?;
        }
        // `clone_out_cache` already copied the untrimmed attention caches into the sequence.
        for kv in seq.normal_cache().iter_mut().flatten() {
            if kv.current_seq_len() > keep_len {
                kv.set_len(keep_len)?;
            }
        }
        Ok(())
    }
}
