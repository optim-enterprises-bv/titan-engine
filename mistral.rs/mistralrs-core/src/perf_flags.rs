const CUDA_GRAPHS_ENV: &str = "MISTRALRS_CUDA_GRAPHS";
const FLASHINFER_DECODE_ENV: &str = "MISTRALRS_FLASHINFER_DECODE";

// Per model through titan_cfg (`[models.titan.env]` applies; on/off/1/0/true/false/yes/no), recomputed when
// the settings generation changes. A process-wide OnceLock over std::env ignored the per-model value.
static CUDA_GRAPHS_ENABLED: mistralrs_quant::titan_cfg::GenCell<bool> = mistralrs_quant::titan_cfg::GenCell::new();
static FLASHINFER_DECODE_ENABLED: mistralrs_quant::titan_cfg::GenCell<bool> =
    mistralrs_quant::titan_cfg::GenCell::new();

fn env_flag(name: &str, default: bool) -> bool {
    mistralrs_quant::titan_cfg::flag(name, default)
}

pub(crate) fn cuda_graphs_enabled() -> bool {
    *CUDA_GRAPHS_ENABLED.get_or_init(|| env_flag(CUDA_GRAPHS_ENV, true))
}

pub(crate) fn flashinfer_decode_enabled() -> bool {
    *FLASHINFER_DECODE_ENABLED.get_or_init(|| env_flag(FLASHINFER_DECODE_ENV, true))
}
