//! titan-engine: admission check (`TITAN_ADMIT`). Before a request is scheduled, work out what it will
//! need and refuse it up front instead of letting it fail partway through prefill or decode.
//!
//! - Context: prompt + `max_tokens` must fit in `--max-seq-len`. Stock mistral.rs only refuses a prompt
//!   that alone is over it, and a reply then stops at the limit without saying so.
//! - Device memory: the KV room the request will hold (whole `KV_RESERVE_STEP` buckets covering
//!   prompt + `max_tokens`) plus the prompt pass's working set must fit in what the device has free
//!   (driver free + memory the async pool holds unused). The KV bytes per token come from the
//!   loader's own KV sizing, so this is only checked on titan GGUF loads with `GPU_FRACTION=auto`.
//!
//! `TITAN_ADMIT=1` refuses; `TITAN_ADMIT=log` only logs each verdict (for calibrating the working-set
//! terms against real runs, e.g. with `TITAN_PREFILL_MEMLOG=1`); unset or `0` is off. Working set:
//! `TITAN_ADMIT_BASE_MIB` (default 256) + `TITAN_ADMIT_CTX_KIB` (default 10, the PFS ring's measured
//! long-context attention temporaries) per KV token + `TITAN_ADMIT_CHUNK_KIB` (default 256: 32 KiB per
//! routed (token, expert) task at top-8, as the PFS ring counts it) per prefill-chunk token. The
//! optional PFS staging ring is not counted: without room it falls back to the CPU twin.
//! Nothing here runs on the step path, and it never changes the output of a request it admits.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::models::quantized_qwen35_moe::KV_RESERVE_STEP;

static KV_BYTES_PER_TOKEN: AtomicUsize = AtomicUsize::new(0);

/// Device KV bytes per token of one sequence, from the loader's own KV sizing (`gguf_titan`).
pub(crate) fn set_kv_bytes_per_token(bytes: usize) {
    KV_BYTES_PER_TOKEN.store(bytes, Ordering::Relaxed);
}

fn kv_bytes_per_token() -> usize {
    KV_BYTES_PER_TOKEN.load(Ordering::Relaxed)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Off,
    Log,
    Enforce,
}

pub(crate) fn mode() -> Mode {
    static M: std::sync::OnceLock<Mode> = std::sync::OnceLock::new();
    *M.get_or_init(|| parse_mode(std::env::var("TITAN_ADMIT").ok().as_deref()))
}

fn parse_mode(v: Option<&str>) -> Mode {
    match v {
        Some("1") => Mode::Enforce,
        Some("log") => Mode::Log,
        _ => Mode::Off,
    }
}

/// Working-set terms of a prompt pass, in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WorkingSet {
    pub base: usize,
    pub per_kv_token: usize,
    pub per_chunk_token: usize,
}

impl WorkingSet {
    pub(crate) fn from_env() -> Self {
        let get = |name: &str, default: usize| {
            std::env::var(name).ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(default)
        };
        Self {
            base: get("TITAN_ADMIT_BASE_MIB", 256) << 20,
            per_kv_token: get("TITAN_ADMIT_CTX_KIB", 10) << 10,
            per_chunk_token: get("TITAN_ADMIT_CHUNK_KIB", 256) << 10,
        }
    }
}

/// What one request asks of the context and the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Need {
    /// prompt + the reply room counted against the context (at least 1 token).
    pub context: usize,
    /// KV room reserved on the device, in whole buckets, capped at `max_seq_len`.
    pub kv_tokens: usize,
    pub kv_bytes: usize,
    pub work_bytes: usize,
}

impl Need {
    pub(crate) fn device_bytes(&self) -> usize {
        self.kv_bytes + self.work_bytes
    }
}

/// What the loader and the env fix for every request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Limits {
    pub max_seq_len: usize,
    pub kv_per_token: usize,
    /// Prefill chunk (tokens), 0 for an unchunked prompt pass.
    pub chunk: usize,
    pub ws: WorkingSet,
}

/// `max_new`: the request's `max_tokens`; unset counts one token, as the prompt check always did
/// (the reply then runs until it stops or reaches `max_seq_len`, whose KV the load plan already keeps).
pub(crate) fn need(prompt: usize, max_new: Option<usize>, l: &Limits) -> Need {
    let context = prompt + max_new.unwrap_or(1).max(1);
    let kv_tokens = context.next_multiple_of(KV_RESERVE_STEP).min(l.max_seq_len.max(1));
    let chunk = if l.chunk == 0 { prompt } else { l.chunk.min(prompt) };
    Need {
        context,
        kv_tokens,
        kv_bytes: kv_tokens * l.kv_per_token,
        work_bytes: l.ws.base + kv_tokens * l.ws.per_kv_token + chunk * l.ws.per_chunk_token,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Verdict {
    Fits,
    OverContext { prompt: usize, reply: usize, max_seq_len: usize },
    OverMemory { need: usize, free: usize },
}

/// `free`: device bytes available now, `None` when that cannot be queried (the memory check is skipped).
pub(crate) fn judge(prompt: usize, max_new: Option<usize>, max_seq_len: usize, n: &Need, free: Option<usize>) -> Verdict {
    if n.context > max_seq_len {
        return Verdict::OverContext { prompt, reply: max_new.unwrap_or(1).max(1), max_seq_len };
    }
    match free {
        Some(free) if n.device_bytes() > free => Verdict::OverMemory { need: n.device_bytes(), free },
        _ => Verdict::Fits,
    }
}

impl Verdict {
    /// The error the client sees (OpenAI's context-length wording for the context case).
    pub(crate) fn message(&self) -> Option<String> {
        match self {
            Verdict::Fits => None,
            Verdict::OverContext { prompt, reply, max_seq_len } => Some(format!(
                "This model's maximum context length is {max_seq_len} tokens. However, you requested {} tokens ({prompt} in the messages, {reply} in the completion). Please reduce the length of the messages or completion.",
                prompt + reply
            )),
            Verdict::OverMemory { need, free } => Some(format!(
                "Not enough device memory for this request: it needs about {} MiB (KV cache and prompt working set) and {} MiB is free. Please reduce the length of the messages or completion.",
                need >> 20,
                free >> 20
            )),
        }
    }
}

/// Device bytes available to a new request: driver free + what the async pool holds but does not use.
#[cfg(feature = "cuda")]
pub(crate) fn free_device_bytes(device: &candle_core::Device) -> Option<usize> {
    use crate::utils::memory_usage::{DeviceMemory, MemoryUsage};
    let DeviceMemory::Discrete { free, .. } = MemoryUsage.query(device).ok()? else {
        return None;
    };
    let cached = MemoryUsage.query_cuda_memory_pool(device).ok().flatten().map_or(0, |p| p.cached());
    Some(free + cached)
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn free_device_bytes(_device: &candle_core::Device) -> Option<usize> {
    None
}

/// The whole check for one request. Returns the refusal message, or `None` to admit.
pub(crate) fn check(
    id: usize,
    prompt: usize,
    max_new: Option<usize>,
    max_seq_len: usize,
    device: &candle_core::Device,
) -> Option<String> {
    let mode = mode();
    if mode == Mode::Off {
        return None;
    }
    let kv_per_token = kv_bytes_per_token();
    let limits = Limits {
        max_seq_len,
        kv_per_token,
        chunk: crate::models::quantized_qwen35_moe::prefill_chunk_size(),
        ws: WorkingSet::from_env(),
    };
    let n = need(prompt, max_new, &limits);
    // without a KV size from the loader the memory estimate would be meaningless: context only
    let free = if kv_per_token > 0 && device.is_cuda() { free_device_bytes(device) } else { None };
    let verdict = judge(prompt, max_new, max_seq_len, &n, free);
    tracing::info!(
        "titan admit {id}: prompt {prompt}, max_tokens {max_new:?}, kv {} tokens {} MiB, work {} MiB, free {} -> {verdict:?}{}",
        n.kv_tokens,
        n.kv_bytes >> 20,
        n.work_bytes >> 20,
        free.map_or("?".to_string(), |f| format!("{} MiB", f >> 20)),
        if mode == Mode::Log && verdict != Verdict::Fits { " (log only, admitted)" } else { "" }
    );
    match mode {
        Mode::Enforce => verdict.message(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: usize = 1 << 20;
    const WS: WorkingSet = WorkingSet { base: 256 * MIB, per_kv_token: 10 << 10, per_chunk_token: 256 << 10 };

    fn lim(max_seq_len: usize, kv_per_token: usize, chunk: usize) -> Limits {
        Limits { max_seq_len, kv_per_token, chunk, ws: WS }
    }

    #[test]
    fn modes() {
        assert_eq!(parse_mode(None), Mode::Off);
        assert_eq!(parse_mode(Some("0")), Mode::Off);
        assert_eq!(parse_mode(Some("1")), Mode::Enforce);
        assert_eq!(parse_mode(Some("log")), Mode::Log);
    }

    #[test]
    fn exactly_fits_the_context() {
        let n = need(65_000, Some(536), &lim(65_536, 1, 512));
        assert_eq!(n.context, 65_536);
        assert_eq!(n.kv_tokens, 65_536);
        assert_eq!(judge(65_000, Some(536), 65_536, &n, None), Verdict::Fits);
    }

    #[test]
    fn one_token_over_the_context() {
        let n = need(65_000, Some(537), &lim(65_536, 1, 512));
        let v = judge(65_000, Some(537), 65_536, &n, None);
        assert_eq!(v, Verdict::OverContext { prompt: 65_000, reply: 537, max_seq_len: 65_536 });
        assert!(v.message().unwrap().contains("65537 tokens (65000 in the messages, 537 in the completion)"));
    }

    #[test]
    fn unset_max_tokens_counts_one() {
        let n = need(65_535, None, &lim(65_536, 1, 512));
        assert_eq!(n.context, 65_536);
        assert_eq!(judge(65_535, None, 65_536, &n, None), Verdict::Fits);
        let n = need(65_536, None, &lim(65_536, 1, 512));
        assert!(matches!(judge(65_536, None, 65_536, &n, None), Verdict::OverContext { reply: 1, .. }));
        // max_tokens 0 is treated like unset: the pass still produces one token
        assert_eq!(need(100, Some(0), &lim(65_536, 1, 512)).context, 101);
    }

    #[test]
    fn kv_in_whole_buckets_capped_at_max_seq_len() {
        assert_eq!(need(100, Some(100), &lim(65_536, 1, 512)).kv_tokens, 8192);
        assert_eq!(need(8191, Some(1), &lim(65_536, 1, 512)).kv_tokens, 8192);
        assert_eq!(need(8192, Some(1), &lim(65_536, 1, 512)).kv_tokens, 16_384);
        assert_eq!(need(60_000, Some(1000), &lim(65_536, 1, 512)).kv_tokens, 65_536);
        assert_eq!(need(100, Some(100), &lim(4096, 1, 512)).kv_tokens, 4096);
    }

    #[test]
    fn working_set_terms() {
        let n = need(13_000, Some(3000), &lim(65_536, 20 << 10, 512));
        assert_eq!(n.kv_tokens, 16_384);
        assert_eq!(n.kv_bytes, 16_384 * (20 << 10));
        assert_eq!(n.work_bytes, 256 * MIB + 16_384 * (10 << 10) + 512 * (256 << 10));
        // a prompt shorter than a chunk only pays for its own rows; unchunked pays for all of them
        assert_eq!(need(100, Some(1), &lim(65_536, 0, 512)).work_bytes, 256 * MIB + 8192 * (10 << 10) + 100 * (256 << 10));
        assert_eq!(need(2000, Some(1), &lim(65_536, 0, 0)).work_bytes, 256 * MIB + 8192 * (10 << 10) + 2000 * (256 << 10));
    }

    #[test]
    fn memory_boundary() {
        let n = need(13_000, Some(3000), &lim(65_536, 20 << 10, 512));
        let b = n.device_bytes();
        assert_eq!(judge(13_000, Some(3000), 65_536, &n, Some(b)), Verdict::Fits);
        assert_eq!(judge(13_000, Some(3000), 65_536, &n, Some(b - 1)), Verdict::OverMemory { need: b, free: b - 1 });
        // unknown free memory: context check only
        assert_eq!(judge(13_000, Some(3000), 65_536, &n, None), Verdict::Fits);
    }

    #[test]
    fn context_is_checked_before_memory() {
        let n = need(70_000, Some(10), &lim(65_536, 20 << 10, 512));
        assert!(matches!(judge(70_000, Some(10), 65_536, &n, Some(0)), Verdict::OverContext { .. }));
    }
}
