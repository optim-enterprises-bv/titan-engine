//! titan-engine: `TITAN_FAULT_LOG=1` counts the process's major page faults (reads from disk) per
//! forward, so a decode step that paid for a disk read shows up instead of hiding in the average. Each
//! forward is charged the faults from its start to the next forward's start: the doorbell's CPU experts
//! answer after `forward` returns, and their faults belong to the step they served. Decode forwards
//! (at most `DECODE_ROWS` rows) and prompt passes are counted apart; a summary is logged every
//! `LOG_EVERY` decode forwards. One `getrusage` per forward when on, nothing when off.

use std::sync::Mutex;

/// Forwards of at most this many rows are decode steps (batch-1 decode and MTP verification).
const DECODE_ROWS: usize = 8;
const LOG_EVERY: u64 = 200;
/// Bounds of the per-step fault histogram: 0, 1, 2-7, 8-63, 64+.
const BUCKETS: [u64; 4] = [1, 2, 8, 64];

pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TITAN_FAULT_LOG").is_ok_and(|v| v == "1"))
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub(crate) struct Counts {
    pub forwards: u64,
    pub faults: u64,
    pub with_fault: u64,
    pub max: u64,
    pub hist: [u64; BUCKETS.len() + 1],
}

impl Counts {
    fn add(&mut self, faults: u64) {
        self.forwards += 1;
        self.faults += faults;
        self.with_fault += u64::from(faults > 0);
        self.max = self.max.max(faults);
        self.hist[BUCKETS.iter().take_while(|&&b| faults >= b).count()] += 1;
    }

    fn report(&self) -> String {
        format!(
            "{} forwards, {} with a major fault ({:.2}%), {} faults, max {}/forward, hist [0: {}, 1: {}, 2-7: {}, 8-63: {}, 64+: {}]",
            self.forwards,
            self.with_fault,
            100.0 * self.with_fault as f64 / self.forwards.max(1) as f64,
            self.faults,
            self.max,
            self.hist[0],
            self.hist[1],
            self.hist[2],
            self.hist[3],
            self.hist[4]
        )
    }
}

#[derive(Default)]
struct State {
    /// Fault count and row count at the start of the forward in flight.
    open: Option<(u64, usize)>,
    decode: Counts,
    prompt: Counts,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn major_faults() -> u64 {
    let mut u = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: getrusage only writes the struct it is given
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, u.as_mut_ptr()) } != 0 {
        return 0;
    }
    // SAFETY: zero-initialised and filled by a successful getrusage
    unsafe { u.assume_init() }.ru_majflt as u64
}

/// Call at the start of every forward with its row count.
pub(crate) fn forward_start(rows: usize) {
    if !enabled() {
        return;
    }
    let now = major_faults();
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(State::default);
    if let Some((start, prev_rows)) = s.open.replace((now, rows)) {
        let faults = now.saturating_sub(start);
        if prev_rows <= DECODE_ROWS {
            s.decode.add(faults);
            if s.decode.forwards.is_multiple_of(LOG_EVERY) {
                tracing::info!("titan faults: decode {}; prompt {}", s.decode.report(), s.prompt.report());
            }
        } else {
            s.prompt.add(faults);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_buckets() {
        let mut c = Counts::default();
        for f in [0, 0, 1, 2, 7, 8, 63, 64, 1000] {
            c.add(f);
        }
        assert_eq!(c.forwards, 9);
        assert_eq!(c.with_fault, 7);
        assert_eq!(c.faults, 1 + 2 + 7 + 8 + 63 + 64 + 1000);
        assert_eq!(c.max, 1000);
        assert_eq!(c.hist, [2, 1, 2, 2, 2]);
    }

    #[test]
    fn reads_rusage() {
        let a = major_faults();
        assert!(major_faults() >= a);
    }
}
