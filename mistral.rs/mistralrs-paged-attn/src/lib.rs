#[cfg(all(feature = "cuda", not(feature = "oxide")))]
compile_error!("titan-engine is oxide-only: build with the `oxide` feature (the crate default). The cuBLAS/nvcc path is not supported here.");

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KvCacheScales {
    pub k: f32,
    pub v: f32,
}

pub const DEFAULT_FP8_KV_CACHE_SCALES: KvCacheScales = KvCacheScales { k: 1.0, v: 1.0 };

#[cfg(feature = "oxide")]
use titan_oxide_ffi as _;
#[cfg(all(feature = "cuda", target_family = "unix"))]
mod cuda;
#[cfg(all(feature = "cuda", target_family = "unix"))]
pub use cuda::*;

#[cfg(feature = "metal")]
mod metal;
#[cfg(feature = "metal")]
pub use metal::*;
