use candle_core::{DType, Result, Tensor};
use mistralrs_quant::{Comm, QuantMethod, ShardedVarBuilder};
use std::sync::Arc;

use crate::device_map::DeviceMapper;
use crate::pipeline::RecurrentBatchKind;

use super::backend;
use super::cache::GdnLayerCache;
use super::config::{GdnConfig, GdnDims};
use super::norm::RmsNormGated;
use super::projection::{GdnInputProjection, GdnProjection};
use super::weights::{GdnInputProjectionKind, GdnWeightLoadCtx, GdnWeights};

/// One decode row's inputs to the GDN state update: its causal-conv input and gates, `(batch, 1, ..)`.
#[derive(Clone, Debug)]
pub struct DecodeRowInputs {
    pub mixed_qkv: Tensor,
    pub b: Tensor,
    pub a: Tensor,
}

impl DecodeRowInputs {
    /// Batch row `i` of these inputs.
    pub fn batch_row(&self, i: usize) -> Result<Self> {
        Ok(Self { mixed_qkv: self.mixed_qkv.narrow(0, i, 1)?, b: self.b.narrow(0, i, 1)?, a: self.a.narrow(0, i, 1)? })
    }
}

/// A GDN layer's state update alone: advances a state by one decode row with the decode kernels, the same
/// conv and recurrence `forward_decode_rows` ran for that row, so replaying a row onto the state before it
/// gives exactly the state after it.
#[derive(Clone, Debug)]
pub struct GdnStateStep {
    conv1d_weight: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    dims: GdnDims,
    dtype: DType,
}

impl GdnStateStep {
    pub fn apply(&self, cache: &mut GdnLayerCache, row: &DecodeRowInputs) -> Result<()> {
        let mixed =
            backend::causal_conv1d(&row.mixed_qkv, &self.conv1d_weight, &self.dims, cache, RecurrentBatchKind::Decode)?;
        backend::apply_recurrence_from_convolved(
            &mixed,
            &row.b,
            &row.a,
            &self.a_log,
            &self.dt_bias,
            &self.dims,
            row.mixed_qkv.dim(0)?,
            1,
            cache,
            self.dtype,
        )?;
        Ok(())
    }
}

pub struct GatedDeltaNet {
    pub input_proj: GdnInputProjection,
    pub conv1d_weight: Tensor,
    pub dt_bias: Tensor,
    pub a_log: Tensor,
    pub norm: RmsNormGated,
    pub out_proj: Arc<dyn QuantMethod>,
    /// GGUF files whose `ssm_out` quant blocks span several V heads keep the converter's (tiled) head
    /// order in the weight; the output is gathered into that order before `out_proj` instead.
    pub out_proj_input_perm: Option<Tensor>,
    dims: GdnDims,
}

impl GatedDeltaNet {
    pub fn load(
        vb: ShardedVarBuilder,
        cfg: &dyn GdnConfig,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        comm: &Arc<Comm>,
        input_projection_kind: GdnInputProjectionKind,
    ) -> Result<Self> {
        let dims = GdnDims::new(cfg);
        let weights = GdnWeights::load(
            vb,
            GdnWeightLoadCtx {
                cfg,
                dims: &dims,
                mapper,
                layer_idx,
                loading_isq,
                comm,
                input_projection_kind,
            },
        )?;
        Ok(Self {
            input_proj: weights.input_proj,
            conv1d_weight: weights.conv1d_weight,
            dt_bias: weights.dt_bias,
            a_log: weights.a_log,
            norm: weights.norm,
            out_proj: weights.out_proj,
            out_proj_input_perm: None,
            dims,
        })
    }

    /// Build a GDN layer from already-loaded weights (GGUF path), in HF layout:
    /// grouped V-head order, `conv1d_weight` shaped `(conv_dim, 1, kernel)` and
    /// `a_log` = `A_log` (not the `-exp(A_log)` that GGUF stores as `ssm_a`).
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        cfg: &dyn GdnConfig,
        in_proj_qkv: Arc<dyn QuantMethod>,
        in_proj_z: Arc<dyn QuantMethod>,
        in_proj_b: Arc<dyn QuantMethod>,
        in_proj_a: Arc<dyn QuantMethod>,
        conv1d_weight: Tensor,
        dt_bias: Tensor,
        a_log: Tensor,
        norm_weight: Tensor,
        out_proj: Arc<dyn QuantMethod>,
    ) -> Self {
        Self {
            input_proj: GdnInputProjection::Split {
                in_proj_qkv,
                in_proj_z,
                in_proj_b,
                in_proj_a,
            },
            conv1d_weight,
            dt_bias,
            a_log,
            norm: RmsNormGated::from_weight(norm_weight, cfg.rms_norm_eps()),
            out_proj,
            out_proj_input_perm: None,
            dims: GdnDims::new(cfg),
        }
    }

    /// See `out_proj_input_perm`.
    pub fn with_out_proj_input_perm(mut self, index: Tensor) -> Self {
        self.out_proj_input_perm = Some(index);
        self
    }

    /// The state update of `forward_decode_rows` on `dtype` inputs, for replaying its rows.
    pub fn state_step(&self, dtype: DType) -> GdnStateStep {
        GdnStateStep {
            conv1d_weight: self.conv1d_weight.clone(),
            a_log: self.a_log.clone(),
            dt_bias: self.dt_bias.clone(),
            dims: self.dims,
            dtype,
        }
    }

    pub fn forward(
        &self,
        x: &Tensor,
        cache: &mut GdnLayerCache,
        batch_kind: RecurrentBatchKind,
    ) -> Result<Tensor> {
        self.forward_with_conv_carry(x, cache, batch_kind, false)
    }

    /// `forward`, where `conv_carry` marks a prompt chunk that continues the one before it: the
    /// full causal conv (which zero-pads its window) then sees the last `kernel - 1` inputs of the
    /// previous chunk, kept in `cache.conv_state`. They are prepended to the chunk and their outputs
    /// dropped, so every output is the same window sum, in the same order, as an unchunked prompt's.
    /// The recurrence already starts from `cache.recurrent_state`.
    pub fn forward_with_conv_carry(
        &self,
        x: &Tensor,
        cache: &mut GdnLayerCache,
        batch_kind: RecurrentBatchKind,
        conv_carry: bool,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;
        let dtype = x.dtype();

        let projected = self.project(x, batch_size, seq_len)?;
        let mixed_qkv = projected.conv_input();
        let carry = self.dims.conv_kernel_size.saturating_sub(1);
        let mixed_qkv = if conv_carry && carry > 0 && matches!(batch_kind, RecurrentBatchKind::Prefill) {
            let state_len = cache.conv_state.dim(2)?;
            let prev = cache
                .conv_state
                .narrow(2, state_len - carry, carry)?
                .transpose(1, 2)?
                .to_dtype(mixed_qkv.dtype())?;
            let ext = Tensor::cat(&[prev, mixed_qkv], 1)?;
            backend::causal_conv1d(&ext, &self.conv1d_weight, &self.dims, cache, batch_kind)?
                .narrow(1, carry, seq_len)?
        } else {
            backend::causal_conv1d(
                &mixed_qkv,
                &self.conv1d_weight,
                &self.dims,
                cache,
                batch_kind,
            )?
        };
        let y = backend::apply_recurrence_from_convolved(
            &mixed_qkv,
            &projected.b,
            &projected.a,
            &self.a_log,
            &self.dt_bias,
            &self.dims,
            batch_size,
            seq_len,
            cache,
            dtype,
        )?;

        self.finish_forward(y, projected.z, batch_size, seq_len, dtype)
    }

    /// Decode steps for the `seq_len` rows of each sequence of `x` in one call, each row bit-identical to
    /// `forward(row, cache, Decode)` of its sequence alone: the projections go through `lin` (a row-exact batched
    /// matmul), the causal conv and the recurrence run row by row with the decode kernels, and
    /// `on_row(r, cache, inputs)` sees the state after row `r` and that row's inputs (speculative verification
    /// keeps one or the other for its rollback; `state_step` replays a row from its inputs).
    pub fn forward_decode_rows(
        &self,
        x: &Tensor,
        cache: &mut GdnLayerCache,
        mm: &super::projection::DecodeMatmuls,
        on_row: &mut dyn FnMut(usize, &GdnLayerCache, DecodeRowInputs) -> Result<()>,
    ) -> Result<Tensor> {
        let lin = &mm.lin;
        let (batch_size, seq_len, _) = x.dims3()?;
        let dtype = x.dtype();
        let d = &self.dims;
        // (conv input, z, per-row (b, a)): b and a feed the recurrence one row at a time
        let (mixed_qkv, z, ba) = match &self.input_proj {
            GdnInputProjection::Split {
                in_proj_qkv,
                in_proj_z,
                in_proj_b,
                in_proj_a,
            } => {
                // F32-activation weights cast each row in and out: cast the rows in once
                let f32_in = x.dtype() != DType::F32
                    && [in_proj_b, in_proj_a].iter().all(|w| w.quantized_act_type() == Some(DType::F32));
                let xs = if f32_in { x.to_dtype(DType::F32)? } else { x.clone() };
                let [qkv, z] = <[Tensor; 2]>::try_from((mm.many)(&[in_proj_qkv, in_proj_z], x)?)
                    .map_err(|_| candle_core::Error::msg("GDN decode rows: two input projections expected"))?;
                // every row as its decode step's kernel computes it alone
                let [b, a] = <[Tensor; 2]>::try_from((mm.lone)(&[in_proj_b, in_proj_a], &xs)?)
                    .map_err(|_| candle_core::Error::msg("GDN decode rows: two gate projections expected"))?;
                let gate = |y: Tensor, r: usize| -> Result<Tensor> {
                    let y = y.reshape((batch_size, seq_len, d.num_v_heads))?.narrow(1, r, 1)?;
                    if f32_in { y.to_dtype(dtype) } else { Ok(y) }
                };
                (
                    qkv.reshape((batch_size, seq_len, d.conv_dim))?,
                    z.reshape((batch_size, seq_len, d.num_v_heads, d.head_v_dim))?,
                    (0..seq_len).map(|r| Ok((gate(b.clone(), r)?, gate(a.clone(), r)?))).collect::<Result<Vec<_>>>()?,
                )
            }
            GdnInputProjection::Grouped { .. } => {
                let p = self.input_proj.forward_with(x, d, batch_size, seq_len, lin)?;
                let ba = (0..seq_len)
                    .map(|r| Ok((p.b.narrow(1, r, 1)?, p.a.narrow(1, r, 1)?)))
                    .collect::<Result<Vec<_>>>()?;
                (p.conv_input(), p.z, ba)
            }
        };
        let mut ys = Vec::with_capacity(seq_len);
        for r in 0..seq_len {
            let row = DecodeRowInputs { mixed_qkv: mixed_qkv.narrow(1, r, 1)?, b: ba[r].0.clone(), a: ba[r].1.clone() };
            let mixed = backend::causal_conv1d(
                &row.mixed_qkv,
                &self.conv1d_weight,
                &self.dims,
                cache,
                RecurrentBatchKind::Decode,
            )?;
            ys.push(backend::apply_recurrence_from_convolved(
                &mixed,
                &ba[r].0,
                &ba[r].1,
                &self.a_log,
                &self.dt_bias,
                &self.dims,
                batch_size,
                1,
                cache,
                dtype,
            )?);
            on_row(r, cache, row)?;
        }
        let y = Tensor::cat(&ys, 1)?;
        self.finish_forward_with(y, z, batch_size, seq_len, lin)
    }

    fn project(&self, x: &Tensor, batch_size: usize, seq_len: usize) -> Result<GdnProjection> {
        self.input_proj.forward(x, &self.dims, batch_size, seq_len)
    }

    fn finish_forward(
        &self,
        y: Tensor,
        z: Tensor,
        batch_size: usize,
        seq_len: usize,
        _dtype: DType,
    ) -> Result<Tensor> {
        self.finish_forward_with(y, z, batch_size, seq_len, &|w, x| w.forward(x))
    }

    fn finish_forward_with(
        &self,
        y: Tensor,
        z: Tensor,
        batch_size: usize,
        seq_len: usize,
        lin: &super::projection::LinearFn<'_>,
    ) -> Result<Tensor> {
        let z_shape = z.shape().clone();
        let y = y.reshape(((), self.dims.head_v_dim))?;
        let z = z.reshape(((), self.dims.head_v_dim))?;
        let y = self.norm.forward(&y, &z)?;
        let y = y.reshape(z_shape)?;
        let y = y.reshape((batch_size, seq_len, self.dims.value_dim))?;
        let y = match &self.out_proj_input_perm {
            Some(index) => y.index_select(index, 2)?,
            None => y,
        };
        lin(&self.out_proj, &y)
    }
}
