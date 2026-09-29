//! Process-wide counters for the live monitor page (`/monitor`, `GET /v1/titan/stats`).
//!
//! Writers are the engine, the titan model and the tiered experts; the reader is the server's stats
//! endpoint. Every per-step write is a relaxed atomic. The only lock (`RECENT`) is taken at request
//! boundaries (prompt start, first token, finish), never per decode step.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// Requests waiting to be scheduled (scheduler waiting queue + pending admission).
static QUEUED: AtomicU64 = AtomicU64::new(0);
/// Sequences the scheduler is running.
static RUNNING: AtomicU64 = AtomicU64::new(0);
static MAX_SEQ_LEN: AtomicU64 = AtomicU64::new(0);

static PREFILL_ACTIVE: AtomicBool = AtomicBool::new(false);
/// Prompt tokens in the KV cache so far (reused ones included) / the whole prompt / reused.
static PREFILL_DONE: AtomicU64 = AtomicU64::new(0);
static PREFILL_TOTAL: AtomicU64 = AtomicU64::new(0);
static PREFILL_REUSED: AtomicU64 = AtomicU64::new(0);
static PREFILL_START_MS: AtomicU64 = AtomicU64::new(0);

/// The sequence the "current" figures describe (id + 1; 0 = none).
static CUR_SEQ: AtomicU64 = AtomicU64::new(0);
static CUR_OUT: AtomicU64 = AtomicU64::new(0);
static CUR_DECODE_NS: AtomicU64 = AtomicU64::new(0);
static CUR_CTX: AtomicU64 = AtomicU64::new(0);
static CUR_PROMPT_TPS_BITS: AtomicU64 = AtomicU64::new(0);

static TOTAL_DECODE_TOKENS: AtomicU64 = AtomicU64::new(0);
static TOTAL_DECODE_NS: AtomicU64 = AtomicU64::new(0);
static TOTAL_PROMPT_TOKENS: AtomicU64 = AtomicU64::new(0);
static TOTAL_PROMPT_NS: AtomicU64 = AtomicU64::new(0);
static TOTAL_REQUESTS: AtomicU64 = AtomicU64::new(0);
static LAST_ERROR_MS: AtomicU64 = AtomicU64::new(0);

static MTP_STEPS: AtomicU64 = AtomicU64::new(0);
static MTP_DRAFTED: AtomicU64 = AtomicU64::new(0);
static MTP_ACCEPTED: AtomicU64 = AtomicU64::new(0);

static PC_LOOKUPS: AtomicU64 = AtomicU64::new(0);
static PC_HITS: AtomicU64 = AtomicU64::new(0);
static PC_ENTRIES: AtomicU64 = AtomicU64::new(0);
static PC_BYTES: AtomicU64 = AtomicU64::new(0);
static PC_LAST_REUSED: AtomicU64 = AtomicU64::new(0);

static TIERED_EXPERTS_GPU: AtomicU64 = AtomicU64::new(0);
static TIERED_EXPERTS_TOTAL: AtomicU64 = AtomicU64::new(0);
static TIERED_GPU_BYTES: AtomicU64 = AtomicU64::new(0);
static TIERED_HOST_BYTES: AtomicU64 = AtomicU64::new(0);
/// Wall time of CPU miss passes (the spin pool's busy time), split by prompt / decode.
static MISS_NS_PREFILL: AtomicU64 = AtomicU64::new(0);
static MISS_NS_DECODE: AtomicU64 = AtomicU64::new(0);
/// Host-to-device bytes the tiered experts copy (miss rows, expert admissions).
static UPLOAD_BYTES: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Default)]
pub struct RequestRecord {
    pub id: u64,
    /// Arrival (ms since the epoch).
    pub start_ms: u64,
    /// "prefill", "decode", "done", "length", "cancelled", "error".
    pub status: &'static str,
    pub prompt: u64,
    pub reused: u64,
    pub output: u64,
    pub decode_ns: u64,
    pub prompt_ns: u64,
    pub ttft_ms: Option<u64>,
    pub end_ms: Option<u64>,
}

const RECENT_MAX: usize = 20;
static RECENT: Mutex<VecDeque<RequestRecord>> = Mutex::new(VecDeque::new());

fn with_record(id: u64, f: impl FnOnce(&mut RequestRecord)) {
    if let Ok(mut r) = RECENT.lock() {
        if let Some(rec) = r.iter_mut().rev().find(|x| x.id == id) {
            f(rec);
        }
    }
}

/// The context budget shown as "context fill" (the titan loaders set the planned `--max-seq-len`).
pub fn set_max_seq_len(n: usize) {
    MAX_SEQ_LEN.store(n as u64, Relaxed);
}

/// The model's own limit, unless a loader already set the planned one.
pub fn set_max_seq_len_if_unset(n: usize) {
    let _ = MAX_SEQ_LEN.compare_exchange(0, n as u64, Relaxed, Relaxed);
}

/// Once per engine loop iteration.
pub fn set_sched(queued: usize, running: usize) {
    QUEUED.store(queued as u64, Relaxed);
    RUNNING.store(running as u64, Relaxed);
}

/// A sequence's prompt pass starts. `reused`: prompt tokens resumed from the prefix cache.
pub fn prompt_start(id: usize, prompt: usize, reused: usize, arrival_ms: u128) {
    let key = id as u64 + 1;
    if PREFILL_ACTIVE.load(Relaxed) && CUR_SEQ.load(Relaxed) == key {
        return; // a later chunk of the same prompt (paged chunking)
    }
    CUR_SEQ.store(key, Relaxed);
    CUR_OUT.store(0, Relaxed);
    CUR_DECODE_NS.store(0, Relaxed);
    CUR_CTX.store(prompt as u64, Relaxed);
    CUR_PROMPT_TPS_BITS.store(0, Relaxed);
    PREFILL_TOTAL.store(prompt as u64, Relaxed);
    PREFILL_REUSED.store(reused as u64, Relaxed);
    PREFILL_DONE.store(reused as u64, Relaxed);
    PREFILL_START_MS.store(now_ms(), Relaxed);
    PREFILL_ACTIVE.store(true, Relaxed);
    PC_LAST_REUSED.store(reused as u64, Relaxed);
    TOTAL_REQUESTS.fetch_add(1, Relaxed);
    if let Ok(mut r) = RECENT.lock() {
        if r.iter().any(|x| x.id == key) {
            return;
        }
        if r.len() == RECENT_MAX {
            r.pop_front();
        }
        r.push_back(RequestRecord {
            id: key,
            start_ms: arrival_ms as u64,
            status: "prefill",
            prompt: prompt as u64,
            reused: reused as u64,
            ..Default::default()
        });
    }
}

/// Prompt tokens now in the cache (called between prompt chunks by the model).
pub fn prefill_progress(done: usize) {
    PREFILL_DONE.store(done as u64, Relaxed);
}

/// The prompt pass finished. `new_tokens`: prompt tokens actually run; `output`: tokens it sampled.
pub fn prompt_end(id: usize, new_tokens: usize, output: usize, ns: u64, arrival_ms: u128) {
    let key = id as u64 + 1;
    TOTAL_PROMPT_TOKENS.fetch_add(new_tokens as u64, Relaxed);
    TOTAL_PROMPT_NS.fetch_add(ns, Relaxed);
    if CUR_SEQ.load(Relaxed) == key {
        PREFILL_DONE.store(PREFILL_TOTAL.load(Relaxed), Relaxed);
        PREFILL_ACTIVE.store(false, Relaxed);
        CUR_OUT.store(output as u64, Relaxed);
        let tps = if ns > 0 {
            new_tokens as f64 / (ns as f64 * 1e-9)
        } else {
            0.0
        };
        CUR_PROMPT_TPS_BITS.store(tps.to_bits(), Relaxed);
    }
    let now = now_ms();
    with_record(key, |r| {
        if r.status == "prefill" {
            r.status = "decode";
        }
        r.prompt_ns += ns;
        r.output = output as u64;
        r.ttft_ms
            .get_or_insert(now.saturating_sub(arrival_ms as u64));
    });
}

/// One decode step of a sequence finished: `output` tokens generated so far, `ctx` tokens in context.
pub fn decode_step(id: usize, output: usize, ctx: usize, ns: u64) {
    let key = id as u64 + 1;
    let added = if CUR_SEQ.load(Relaxed) == key {
        CUR_DECODE_NS.fetch_add(ns, Relaxed);
        CUR_CTX.store(ctx as u64, Relaxed);
        (output as u64).saturating_sub(CUR_OUT.swap(output as u64, Relaxed))
    } else {
        1
    };
    TOTAL_DECODE_TOKENS.fetch_add(added, Relaxed);
    TOTAL_DECODE_NS.fetch_add(ns, Relaxed);
}

/// A sequence ended (`status`: "done", "length", "cancelled", "error"); called again after the last
/// step's timing is in, which refreshes the counts.
pub fn finish(id: usize, status: &'static str, output: usize, arrival_ms: u128) {
    let key = id as u64 + 1;
    let now = now_ms();
    if status == "error" {
        LAST_ERROR_MS.store(now, Relaxed);
    }
    let cur = CUR_SEQ.load(Relaxed) == key;
    if cur && PREFILL_ACTIVE.load(Relaxed) {
        PREFILL_ACTIVE.store(false, Relaxed);
    }
    let decode_ns = if cur { CUR_DECODE_NS.load(Relaxed) } else { 0 };
    with_record(key, |r| {
        if r.status != "error" {
            r.status = status;
        }
        r.output = output as u64;
        if cur {
            r.decode_ns = decode_ns;
        }
        r.end_ms = Some(now.max(arrival_ms as u64));
    });
}

pub fn mtp_step(drafted: u64, accepted: u64) {
    MTP_STEPS.fetch_add(1, Relaxed);
    MTP_DRAFTED.fetch_add(drafted, Relaxed);
    MTP_ACCEPTED.fetch_add(accepted, Relaxed);
}

pub fn prefix_lookup(hit: bool) {
    PC_LOOKUPS.fetch_add(1, Relaxed);
    if hit {
        PC_HITS.fetch_add(1, Relaxed);
    }
}

pub fn prefix_size(entries: usize, bytes: usize) {
    PC_ENTRIES.store(entries as u64, Relaxed);
    PC_BYTES.store(bytes as u64, Relaxed);
}

/// One tiered tensor built: `experts` of which `resident` on the GPU (count only for one projection
/// per layer, `count_experts`), and the bytes each side holds.
pub fn tiered_placement(
    count_experts: bool,
    experts: usize,
    resident: usize,
    gpu_bytes: usize,
    host_bytes: usize,
) {
    if count_experts {
        TIERED_EXPERTS_TOTAL.fetch_add(experts as u64, Relaxed);
        TIERED_EXPERTS_GPU.fetch_add(resident as u64, Relaxed);
    }
    TIERED_GPU_BYTES.fetch_add(gpu_bytes as u64, Relaxed);
    TIERED_HOST_BYTES.fetch_add(host_bytes as u64, Relaxed);
}

pub fn tiered_miss_ns(ns: u64) {
    if PREFILL_ACTIVE.load(Relaxed) {
        MISS_NS_PREFILL.fetch_add(ns, Relaxed);
    } else {
        MISS_NS_DECODE.fetch_add(ns, Relaxed);
    }
}

pub fn tiered_upload_bytes(bytes: usize) {
    UPLOAD_BYTES.fetch_add(bytes as u64, Relaxed);
}

/// Everything above, read at one moment (relaxed: fields may be a step apart).
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub queued: u64,
    pub running: u64,
    pub max_seq_len: u64,
    pub prefill_active: bool,
    pub prefill_done: u64,
    pub prefill_total: u64,
    pub prefill_reused: u64,
    pub prefill_start_ms: u64,
    pub cur_seq: u64,
    pub cur_out: u64,
    pub cur_decode_ns: u64,
    pub cur_ctx: u64,
    pub cur_prompt_tps: f64,
    pub total_decode_tokens: u64,
    pub total_decode_ns: u64,
    pub total_prompt_tokens: u64,
    pub total_prompt_ns: u64,
    pub total_requests: u64,
    pub last_error_ms: u64,
    pub mtp_steps: u64,
    pub mtp_drafted: u64,
    pub mtp_accepted: u64,
    pub pc_lookups: u64,
    pub pc_hits: u64,
    pub pc_entries: u64,
    pub pc_bytes: u64,
    pub pc_last_reused: u64,
    pub tiered_experts_gpu: u64,
    pub tiered_experts_total: u64,
    pub tiered_gpu_bytes: u64,
    pub tiered_host_bytes: u64,
    pub miss_ns_prefill: u64,
    pub miss_ns_decode: u64,
    pub upload_bytes: u64,
    /// Tiered hit counters: [decode hits, decode misses, verify-row hits, verify-row misses, expert uploads].
    pub tiered_hits: [u64; 5],
}

pub fn snapshot() -> Snapshot {
    Snapshot {
        queued: QUEUED.load(Relaxed),
        running: RUNNING.load(Relaxed),
        max_seq_len: MAX_SEQ_LEN.load(Relaxed),
        prefill_active: PREFILL_ACTIVE.load(Relaxed),
        prefill_done: PREFILL_DONE.load(Relaxed),
        prefill_total: PREFILL_TOTAL.load(Relaxed),
        prefill_reused: PREFILL_REUSED.load(Relaxed),
        prefill_start_ms: PREFILL_START_MS.load(Relaxed),
        cur_seq: CUR_SEQ.load(Relaxed),
        cur_out: CUR_OUT.load(Relaxed),
        cur_decode_ns: CUR_DECODE_NS.load(Relaxed),
        cur_ctx: CUR_CTX.load(Relaxed),
        cur_prompt_tps: f64::from_bits(CUR_PROMPT_TPS_BITS.load(Relaxed)),
        total_decode_tokens: TOTAL_DECODE_TOKENS.load(Relaxed),
        total_decode_ns: TOTAL_DECODE_NS.load(Relaxed),
        total_prompt_tokens: TOTAL_PROMPT_TOKENS.load(Relaxed),
        total_prompt_ns: TOTAL_PROMPT_NS.load(Relaxed),
        total_requests: TOTAL_REQUESTS.load(Relaxed),
        last_error_ms: LAST_ERROR_MS.load(Relaxed),
        mtp_steps: MTP_STEPS.load(Relaxed),
        mtp_drafted: MTP_DRAFTED.load(Relaxed),
        mtp_accepted: MTP_ACCEPTED.load(Relaxed),
        pc_lookups: PC_LOOKUPS.load(Relaxed),
        pc_hits: PC_HITS.load(Relaxed),
        pc_entries: PC_ENTRIES.load(Relaxed),
        pc_bytes: PC_BYTES.load(Relaxed),
        pc_last_reused: PC_LAST_REUSED.load(Relaxed),
        tiered_experts_gpu: TIERED_EXPERTS_GPU.load(Relaxed),
        tiered_experts_total: TIERED_EXPERTS_TOTAL.load(Relaxed),
        tiered_gpu_bytes: TIERED_GPU_BYTES.load(Relaxed),
        tiered_host_bytes: TIERED_HOST_BYTES.load(Relaxed),
        miss_ns_prefill: MISS_NS_PREFILL.load(Relaxed),
        miss_ns_decode: MISS_NS_DECODE.load(Relaxed),
        upload_bytes: UPLOAD_BYTES.load(Relaxed),
        tiered_hits: tiered_hits(),
    }
}

#[cfg(feature = "cuda")]
fn tiered_hits() -> [u64; 5] {
    crate::gguf::titan_tiered::monitor_hits()
}

#[cfg(not(feature = "cuda"))]
fn tiered_hits() -> [u64; 5] {
    [0; 5]
}

/// The last `RECENT_MAX` requests, oldest first.
pub fn recent() -> Vec<RequestRecord> {
    RECENT
        .lock()
        .map(|r| r.iter().cloned().collect())
        .unwrap_or_default()
}
