//! Live monitor: `GET /monitor` (one self-contained page) and `GET /v1/titan/stats` (its JSON).
//!
//! Engine figures are relaxed atomics (`mistralrs_core::titan_monitor`) read per request. System
//! figures (NVML, /proc) come from a 1 Hz sampler thread that runs only while the endpoint is polled.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::Json;
use mistralrs_core::titan_monitor::{self as mon, Snapshot};
use serde_json::{json, Value};

use crate::titan_sys::{read_proc, GpuSample, Nvml, ProcCounters};
use crate::types::ExtractedMistralRsState;

const PAGE: &str = include_str!("titan_monitor.html");
const SAMPLE_PERIOD: Duration = Duration::from_secs(1);
/// The sampler stops this long after the last poll.
const IDLE_STOP: Duration = Duration::from_secs(10);
/// Samples kept for the rolling rates (5 s at 1 Hz).
const WINDOW: usize = 6;
/// A sample older than this is not reported (the sampler was asleep).
const STALE: Duration = Duration::from_secs(3);
/// The model state reads "error" for this long after a request failed.
const ERROR_HOLD_MS: u64 = 5_000;
const MIB: f64 = 1024.0 * 1024.0;

struct Tick {
    at: Instant,
    engine: Snapshot,
    procs: ProcCounters,
}

#[derive(Default)]
struct Latest {
    at: Option<Instant>,
    gpu: Option<GpuSample>,
    rates: Value,
}

struct Sampler {
    epoch: Instant,
    last_poll_ms: AtomicU64,
    running: AtomicBool,
    latest: Mutex<Latest>,
    nvml: OnceLock<Result<Nvml, String>>,
}

fn sampler() -> &'static Sampler {
    static S: OnceLock<Sampler> = OnceLock::new();
    S.get_or_init(|| Sampler {
        epoch: Instant::now(),
        last_poll_ms: AtomicU64::new(0),
        running: AtomicBool::new(false),
        latest: Mutex::new(Latest::default()),
        nvml: OnceLock::new(),
    })
}

impl Sampler {
    fn now_ms(&self) -> u64 {
        u64::try_from(self.epoch.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    fn touch(&'static self) {
        self.last_poll_ms.store(self.now_ms(), Relaxed);
        if !self.running.swap(true, Relaxed) {
            let spawned = std::thread::Builder::new()
                .name("titan-monitor".into())
                .spawn(move || self.run());
            if spawned.is_err() {
                self.running.store(false, Relaxed);
            }
        }
    }

    fn idle(&self) -> bool {
        self.now_ms()
            .saturating_sub(self.last_poll_ms.load(Relaxed))
            > IDLE_STOP.as_millis() as u64
    }

    fn run(&'static self) {
        let mut ticks: VecDeque<Tick> = VecDeque::with_capacity(WINDOW + 1);
        loop {
            let t0 = Instant::now();
            let gpu = match self.nvml.get_or_init(Nvml::load) {
                Ok(n) => Some(n.sample()),
                Err(_) => None,
            };
            ticks.push_back(Tick {
                at: Instant::now(),
                engine: mon::snapshot(),
                procs: read_proc(),
            });
            if ticks.len() > WINDOW {
                ticks.pop_front();
            }
            let rates = rates(&ticks);
            if let Ok(mut l) = self.latest.lock() {
                *l = Latest {
                    at: Some(Instant::now()),
                    gpu,
                    rates,
                };
            }
            if self.idle() {
                self.running.store(false, Relaxed);
                // a poll that raced the stop restarts the thread itself
                if self.idle() || self.running.swap(true, Relaxed) {
                    return;
                }
            }
            std::thread::sleep(SAMPLE_PERIOD.saturating_sub(t0.elapsed()));
        }
    }
}

fn per_s(d: u64, dt: f64) -> f64 {
    if dt > 0.0 {
        d as f64 / dt
    } else {
        0.0
    }
}

fn ratio(num: u64, den: u64) -> Option<f64> {
    (den > 0).then(|| num as f64 / den as f64)
}

/// Rates over the last second (`*_1s`) and the whole window (rolling).
fn rates(ticks: &VecDeque<Tick>) -> Value {
    let (Some(last), Some(first)) = (ticks.back(), ticks.front()) else {
        return Value::Null;
    };
    if ticks.len() < 2 {
        return Value::Null;
    }
    let prev = &ticks[ticks.len() - 2];
    let dt1 = last.at.duration_since(prev.at).as_secs_f64();
    let dtw = last.at.duration_since(first.at).as_secs_f64();
    let (e1, e0, ew) = (&last.engine, &prev.engine, &first.engine);
    let (p1, p0) = (&last.procs, &prev.procs);
    let d = |a: u64, b: u64| a.saturating_sub(b);

    let cpu_busy = d(p1.cpu.0, p0.cpu.0);
    let cpu_total = d(p1.cpu.1, p0.cpu.1);
    // decode steps and MTP verify rows together: with MTP every decode step is a verify batch
    let hits_w = d(
        e1.tiered_hits[0] + e1.tiered_hits[2],
        ew.tiered_hits[0] + ew.tiered_hits[2],
    );
    let miss_w = d(
        e1.tiered_hits[1] + e1.tiered_hits[3],
        ew.tiered_hits[1] + ew.tiered_hits[3],
    );
    let dec_tok_w = d(e1.total_decode_tokens, ew.total_decode_tokens);
    let miss_ns_w = d(e1.miss_ns_decode, ew.miss_ns_decode);
    let mtp_steps_w = d(e1.mtp_steps, ew.mtp_steps);
    let mtp_acc_w = d(e1.mtp_accepted, ew.mtp_accepted);
    let mtp_dr_w = d(e1.mtp_drafted, ew.mtp_drafted);
    let miss_ns_1 = d(
        e1.miss_ns_decode + e1.miss_ns_prefill,
        e0.miss_ns_decode + e0.miss_ns_prefill,
    );
    // prompt chunks land every ~2 s, so the prompt rate uses the window's ticks of the same request
    let prompt_tps = if e1.prefill_active || e0.prefill_active {
        ticks
            .iter()
            .find(|t| t.engine.cur_seq == e1.cur_seq)
            .filter(|t| t.at < last.at)
            .map_or(0.0, |t| {
                per_s(
                    d(e1.prefill_done, t.engine.prefill_done),
                    last.at.duration_since(t.at).as_secs_f64(),
                )
            })
    } else {
        0.0
    };
    json!({
        "window_s": dtw,
        "decode_tps_1s": per_s(d(e1.total_decode_tokens, e0.total_decode_tokens), dt1),
        "prompt_tps_rolling": prompt_tps,
        "cpu_pct": ratio(cpu_busy, cpu_total).map(|r| 100.0 * r),
        "miss_pool_busy_pct": 100.0 * (miss_ns_1 as f64 * 1e-9 / dt1.max(1e-9)).min(1.0),
        "disk_read_mbs": per_s(d(p1.disk_read_bytes, p0.disk_read_bytes), dt1) / MIB,
        "disk_write_mbs": per_s(d(p1.disk_write_bytes, p0.disk_write_bytes), dt1) / MIB,
        "proc_read_mbs": per_s(d(p1.proc_read_bytes, p0.proc_read_bytes), dt1) / MIB,
        "upload_mbs": per_s(d(e1.upload_bytes, e0.upload_bytes), dt1) / MIB,
        "hit_rate_rolling": ratio(hits_w, hits_w + miss_w),
        "cpu_miss_ms_per_token": ratio(miss_ns_w, dec_tok_w).map(|ns| ns / 1e6),
        "mtp_tokens_per_step_rolling": ratio(mtp_acc_w, mtp_steps_w).map(|a| 1.0 + a),
        "mtp_acceptance_rolling": ratio(mtp_acc_w, mtp_dr_w),
    })
}

fn model_state(s: &Snapshot, now_ms: u64) -> &'static str {
    if s.last_error_ms > 0 && now_ms.saturating_sub(s.last_error_ms) < ERROR_HOLD_MS {
        "error"
    } else if s.prefill_active {
        "prefill"
    } else if s.running > 0 {
        "generating"
    } else if s.queued > 0 {
        "queued"
    } else {
        "idle"
    }
}

fn epoch_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

fn gpu_json(gpu: Option<&GpuSample>, nvml: Option<&Result<Nvml, String>>) -> Value {
    let (Some(g), Some(Ok(n))) = (gpu, nvml) else {
        let reason = match nvml {
            Some(Err(e)) => e.clone(),
            _ => "no sample yet".to_string(),
        };
        return json!({ "available": false, "reason": reason });
    };
    json!({
        "available": true,
        "name": n.name,
        "util_pct": g.util_pct,
        "mem_util_pct": g.mem_util_pct,
        "vram_used_mib": g.vram_used.map(|b| b as f64 / MIB),
        "vram_total_mib": g.vram_total.map(|b| b as f64 / MIB),
        "temp_c": g.temp_c,
        "power_w": g.power_w,
        "power_limit_w": g.power_limit_w,
        "clock_graphics_mhz": g.clock_gr_mhz,
        "clock_sm_mhz": g.clock_sm_mhz,
        "clock_sm_max_mhz": g.clock_sm_max_mhz,
        "clock_mem_mhz": g.clock_mem_mhz,
        "pstate": g.pstate,
    })
}

fn build_stats(model: &str) -> Value {
    let smp = sampler();
    smp.touch();
    let s = mon::snapshot();
    let now = epoch_ms();
    let (fresh, gpu, rates) = match smp.latest.lock() {
        Ok(l) => {
            let fresh = l.at.is_some_and(|t| t.elapsed() < STALE);
            (
                fresh,
                if fresh { l.gpu.clone() } else { None },
                if fresh { l.rates.clone() } else { Value::Null },
            )
        }
        Err(_) => (false, None, Value::Null),
    };
    let procs = read_proc();
    let r = |k: &str| rates.get(k).cloned().unwrap_or(Value::Null);
    let nvml = smp.nvml.get();

    let cur_tps = if s.cur_decode_ns > 0 {
        Some(s.cur_out.saturating_sub(1) as f64 / (s.cur_decode_ns as f64 * 1e-9))
    } else {
        None
    };
    let recent = mon::recent();
    let requests: Vec<Value> = recent
        .iter()
        .rev()
        .map(|q| {
            let tps = (q.decode_ns > 0)
                .then(|| q.output.saturating_sub(1) as f64 / (q.decode_ns as f64 * 1e-9));
            let new_prompt = q.prompt.saturating_sub(q.reused);
            let prompt_tps =
                (q.prompt_ns > 0).then(|| new_prompt as f64 / (q.prompt_ns as f64 * 1e-9));
            json!({
                "id": q.id - 1,
                "start_ms": q.start_ms,
                "status": q.status,
                "prompt": q.prompt,
                "reused": q.reused,
                "output": q.output,
                "tps": tps,
                "prompt_tps": prompt_tps,
                "duration_s": q.end_ms.unwrap_or(now).saturating_sub(q.start_ms) as f64 / 1e3,
                "ttft_s": q.ttft_ms.map(|t| t as f64 / 1e3),
            })
        })
        .collect();
    let last_done = recent
        .iter()
        .rev()
        .find(|q| q.end_ms.is_some() && q.decode_ns > 0);
    let last_tps =
        last_done.map(|q| q.output.saturating_sub(1) as f64 / (q.decode_ns as f64 * 1e-9));
    let prefill_elapsed_s = s
        .prefill_active
        .then(|| now.saturating_sub(s.prefill_start_ms) as f64 / 1e3);
    let prompt_tps_current = prefill_elapsed_s
        .filter(|&e| e > 0.0)
        .map(|e| s.prefill_done.saturating_sub(s.prefill_reused) as f64 / e);
    let h = s.tiered_hits;
    let ctx_tokens = if s.prefill_active {
        s.prefill_done
    } else {
        s.cur_ctx
    };

    json!({
        "time_ms": now,
        "sampler": { "active": smp.running.load(Relaxed), "fresh": fresh, "period_ms": SAMPLE_PERIOD.as_millis() as u64 },
        "model": {
            "name": model,
            "state": model_state(&s, now),
            "queued": s.queued,
            "running": s.running,
            "requests_total": s.total_requests,
            "prefill": {
                "done": s.prefill_done,
                "total": s.prefill_total,
                "reused": s.prefill_reused,
                "elapsed_s": prefill_elapsed_s,
            },
            "last_error_ms": (s.last_error_ms > 0).then_some(s.last_error_ms),
        },
        "speed": {
            "decode_tps_current": cur_tps,
            "decode_tps_last": last_tps,
            "decode_tps_1s": r("decode_tps_1s"),
            "prompt_tps_current": prompt_tps_current,
            "prompt_tps_last": (s.cur_prompt_tps > 0.0).then_some(s.cur_prompt_tps),
            "prompt_tps_rolling": r("prompt_tps_rolling"),
            "decode_tps_avg": ratio(s.total_decode_tokens, s.total_decode_ns).map(|x| x * 1e9),
            "prompt_tps_avg": ratio(s.total_prompt_tokens, s.total_prompt_ns).map(|x| x * 1e9),
        },
        "context": { "tokens": ctx_tokens, "max": s.max_seq_len },
        "gpu": gpu_json(gpu.as_ref(), nvml),
        "pcie": {
            "available": gpu.is_some(),
            "gen": gpu.as_ref().and_then(|g| g.link_gen),
            "width": gpu.as_ref().and_then(|g| g.link_width),
            "max_gen": gpu.as_ref().and_then(|g| g.max_link_gen),
            "max_width": gpu.as_ref().and_then(|g| g.max_link_width),
            "tx_mbs": gpu.as_ref().and_then(|g| g.pcie_tx_kbs).map(|k| f64::from(k) / 1024.0),
            "rx_mbs": gpu.as_ref().and_then(|g| g.pcie_rx_kbs).map(|k| f64::from(k) / 1024.0),
            "engine_upload_mbs": r("upload_mbs"),
        },
        "cpu": {
            "util_pct": r("cpu_pct"),
            "cores": std::thread::available_parallelism().map_or(0, |n| n.get()),
            "miss_pool_busy_pct": r("miss_pool_busy_pct"),
        },
        "ram": {
            "used_mib": procs.mem_total.saturating_sub(procs.mem_available) as f64 / MIB,
            "total_mib": procs.mem_total as f64 / MIB,
            "rss_mib": procs.rss as f64 / MIB,
            "rss_anon_mib": procs.rss_anon as f64 / MIB,
            "rss_file_mib": procs.rss_file as f64 / MIB,
        },
        "disk": {
            "devices": procs.disk_names,
            "read_mbs": r("disk_read_mbs"),
            "write_mbs": r("disk_write_mbs"),
            "process_read_mbs": r("proc_read_mbs"),
        },
        "titan": {
            "tiered": s.tiered_experts_total > 0,
            "experts_gpu": s.tiered_experts_gpu,
            "experts_total": s.tiered_experts_total,
            "experts_gpu_mib": s.tiered_gpu_bytes as f64 / MIB,
            "experts_host_mib": s.tiered_host_bytes as f64 / MIB,
            "hit_rate_rolling": r("hit_rate_rolling"),
            "hit_rate_total": ratio(h[0] + h[2], h[0] + h[1] + h[2] + h[3]),
            "single_row_hit_rate_total": ratio(h[0], h[0] + h[1]),
            "verify_hit_rate_total": ratio(h[2], h[2] + h[3]),
            "expert_uploads": h[4],
            "cpu_miss_ms_per_token": r("cpu_miss_ms_per_token"),
            "cpu_miss_s_total": (s.miss_ns_decode + s.miss_ns_prefill) as f64 * 1e-9,
            "mtp": {
                "enabled": s.mtp_steps > 0,
                "steps": s.mtp_steps,
                "tokens_per_step_rolling": r("mtp_tokens_per_step_rolling"),
                "tokens_per_step_total": ratio(s.mtp_accepted, s.mtp_steps).map(|a| 1.0 + a),
                "acceptance_rolling": r("mtp_acceptance_rolling"),
                "acceptance_total": ratio(s.mtp_accepted, s.mtp_drafted),
            },
            "prefix_cache": {
                "lookups": s.pc_lookups,
                "hits": s.pc_hits,
                "hit_rate": ratio(s.pc_hits, s.pc_lookups),
                "entries": s.pc_entries,
                "mib": s.pc_bytes as f64 / MIB,
                "last_reused": s.pc_last_reused,
            },
        },
        "requests": requests,
    })
}

fn model_name(state: &ExtractedMistralRsState) -> String {
    static NAME: OnceLock<String> = OnceLock::new();
    NAME.get_or_init(|| {
        let id = state
            .get_default_model_id()
            .ok()
            .flatten()
            .or_else(|| state.list_models().ok().and_then(|m| m.into_iter().next()));
        id.unwrap_or_default()
    })
    .clone()
}

pub async fn titan_stats(state: ExtractedMistralRsState) -> Json<Value> {
    let model = model_name(&state);
    Json(build_stats(&model))
}

pub async fn monitor_page() -> Response {
    (
        [
            (header::CONTENT_TYPE, "text/html; charset=utf-8"),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        PAGE,
    )
        .into_response()
}
