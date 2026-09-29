use super::config::GdnDims;
use candle_core::{Result, Tensor, D};
use mistralrs_quant::QuantMethod;
use std::sync::Arc;

pub enum GdnInputProjection {
    Grouped {
        in_proj_qkvz: Arc<dyn QuantMethod>,
        in_proj_ba: Arc<dyn QuantMethod>,
    },
    Split {
        in_proj_qkv: Arc<dyn QuantMethod>,
        in_proj_z: Arc<dyn QuantMethod>,
        in_proj_b: Arc<dyn QuantMethod>,
        in_proj_a: Arc<dyn QuantMethod>,
    },
}

/// How a projection weight is applied to activations (plain `forward`, or a row-exact batched matmul).
pub type LinearFn<'a> = dyn Fn(&Arc<dyn QuantMethod>, &Tensor) -> Result<Tensor> + 'a;

impl GdnInputProjection {
    pub fn forward(
        &self,
        x: &Tensor,
        dims: &GdnDims,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<GdnProjection> {
        self.forward_with(x, dims, batch_size, seq_len, &|w, x| w.forward(x))
    }

    pub fn forward_with(
        &self,
        x: &Tensor,
        dims: &GdnDims,
        batch_size: usize,
        seq_len: usize,
        lin: &LinearFn<'_>,
    ) -> Result<GdnProjection> {
        match self {
            Self::Grouped {
                in_proj_qkvz,
                in_proj_ba,
            } => GdnProjection::from_grouped(
                lin(in_proj_qkvz, x)?,
                lin(in_proj_ba, x)?,
                dims,
                batch_size,
                seq_len,
            ),
            Self::Split {
                in_proj_qkv,
                in_proj_z,
                in_proj_b,
                in_proj_a,
            } => GdnProjection::from_split(
                lin(in_proj_qkv, x)?,
                lin(in_proj_z, x)?,
                lin(in_proj_b, x)?,
                lin(in_proj_a, x)?,
                dims,
                batch_size,
                seq_len,
            ),
        }
    }
}

pub struct GdnProjection {
    /// The causal-conv input `[batch, seq, 2 * key_dim + value_dim]`: q | k | v per token.
    pub qkv: Tensor,
    pub z: Tensor,
    pub b: Tensor,
    pub a: Tensor,
}

impl GdnProjection {
    pub fn from_grouped(
        mixed_qkvz: Tensor,
        mixed_ba: Tensor,
        dims: &GdnDims,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Self> {
        let group_size_qkvz = 2 * dims.head_k_dim + 2 * dims.v_per_group * dims.head_v_dim;
        let mixed_qkvz =
            mixed_qkvz.reshape((batch_size, seq_len, dims.num_k_heads, group_size_qkvz))?;
        let mixed_ba =
            mixed_ba.reshape((batch_size, seq_len, dims.num_k_heads, 2 * dims.v_per_group))?;

        let mut offset = 0;
        let q = mixed_qkvz.narrow(D::Minus1, offset, dims.head_k_dim)?;
        offset += dims.head_k_dim;
        let k = mixed_qkvz.narrow(D::Minus1, offset, dims.head_k_dim)?;
        offset += dims.head_k_dim;
        let v = mixed_qkvz.narrow(D::Minus1, offset, dims.v_per_group * dims.head_v_dim)?;
        offset += dims.v_per_group * dims.head_v_dim;
        let z = mixed_qkvz.narrow(D::Minus1, offset, dims.v_per_group * dims.head_v_dim)?;

        let b = mixed_ba.narrow(D::Minus1, 0, dims.v_per_group)?;
        let a = mixed_ba.narrow(D::Minus1, dims.v_per_group, dims.v_per_group)?;

        let qkv = Tensor::cat(
            &[
                &q.reshape((batch_size, seq_len, dims.key_dim))?,
                &k.reshape((batch_size, seq_len, dims.key_dim))?,
                &v.reshape((batch_size, seq_len, dims.value_dim))?,
            ],
            D::Minus1,
        )?;
        Ok(Self {
            qkv,
            z: z.reshape((batch_size, seq_len, dims.num_v_heads, dims.head_v_dim))?,
            b: b.reshape((batch_size, seq_len, dims.num_v_heads))?,
            a: a.reshape((batch_size, seq_len, dims.num_v_heads))?,
        })
    }

    pub fn from_split(
        mixed_qkv: Tensor,
        mixed_z: Tensor,
        mixed_b: Tensor,
        mixed_a: Tensor,
        dims: &GdnDims,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Self> {
        // already q | k | v per token: no split and re-concatenation
        Ok(Self {
            qkv: mixed_qkv.reshape((batch_size, seq_len, 2 * dims.key_dim + dims.value_dim))?,
            z: mixed_z.reshape((batch_size, seq_len, dims.num_v_heads, dims.head_v_dim))?,
            b: mixed_b.reshape((batch_size, seq_len, dims.num_v_heads))?,
            a: mixed_a.reshape((batch_size, seq_len, dims.num_v_heads))?,
        })
    }

    pub fn conv_input(&self) -> Tensor {
        self.qkv.clone()
    }
}
