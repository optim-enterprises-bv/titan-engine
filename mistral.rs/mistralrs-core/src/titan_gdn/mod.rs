#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! titan fork-local Gated Delta Net: the GDN implementation the titan hybrid GGUF models
//! (`quantized_qwen35_moe`: qwen35moe, qwen35, qwen3next) run on, kept at the fork's ktrace
//! version (84b53bf kernels + the MTP-verify / chunked-prefill changes). Upstream v0.9.4 rewrote
//! `crate::gdn` around in-place pooled-state kernels whose launchers are not ported to cuda-oxide,
//! so upstream's own models keep `crate::gdn` and the titan models use this copy.

mod backend;
mod cache;
mod config;
pub(crate) mod cuda_gdn;
mod ffi;
mod layer;
mod norm;
mod projection;
mod weights;

pub use cache::GdnLayerCache;
pub use config::GdnConfig;
pub use layer::GatedDeltaNet;
pub use weights::GdnInputProjectionKind;
