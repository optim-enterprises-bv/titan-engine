//! Context lookup drafting (`TITAN_LOOKUP`, on by default next to MTP): the tokens that followed the latest earlier
//! occurrence of a sequence's last tokens, verified like an MTP draft. An agent re-emits much of what its context
//! holds (file contents, edits, tool output); there most of such a draft is accepted, several tokens per step.

/// Default `TITAN_LOOKUP_MIN`: shortest suffix that has to recur before its continuation is proposed.
const DEFAULT_MIN_NGRAM: usize = 3;
/// Longest suffix compared: among matches, the longest one wins, then the latest.
const MAX_NGRAM: usize = 8;

struct Cfg {
    on: bool,
    min_ngram: usize,
}

fn cfg() -> &'static Cfg {
    static C: mistralrs_quant::titan_cfg::GenCell<Cfg> = mistralrs_quant::titan_cfg::GenCell::new();
    C.get_or_init(|| {
        let on = mistralrs_quant::titan_cfg::var("TITAN_LOOKUP").map(|v| v != "0").unwrap_or(true);
        let min_ngram = mistralrs_quant::titan_cfg::var("TITAN_LOOKUP_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_MIN_NGRAM)
            .clamp(1, MAX_NGRAM);
        if on {
            tracing::info!("titan lookup: context lookup drafts from {min_ngram}-token matches");
        }
        Cfg { on, min_ngram }
    })
}

pub(crate) fn enabled() -> bool {
    cfg().on
}

/// Up to `max_len` tokens that followed the best earlier match of the end of `toks`, empty when nothing recurs.
pub(crate) fn draft(toks: &[u32], max_len: usize) -> Vec<u32> {
    let c = cfg();
    let len = toks.len();
    if !c.on || max_len == 0 || len <= c.min_ngram {
        return Vec::new();
    }
    let min = c.min_ngram;
    let tail = &toks[len - min..];
    let (mut best_end, mut best_n) = (0, 0);
    // `end`: one past a candidate match, where its continuation starts; the suffix itself (end == len) is excluded
    for end in (min..len).rev() {
        if toks[end - 1] != tail[min - 1] || toks[end - min..end] != *tail {
            continue;
        }
        let mut n = min;
        while n < MAX_NGRAM && n < end && toks[end - n - 1] == toks[len - n - 1] {
            n += 1;
        }
        if n > best_n {
            (best_end, best_n) = (end, n);
            if n == MAX_NGRAM {
                break;
            }
        }
    }
    if best_n == 0 {
        return Vec::new();
    }
    toks[best_end..(best_end + max_len).min(len)].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proposes_the_continuation_of_the_longest_latest_match() {
        let toks = [1, 2, 3, 4, 5, 9, 9, 2, 3, 4, 6, 7, 8, 1, 2, 3, 4];
        // [2, 3, 4] recurs at 1 and 7; the match at 1 extends to [1, 2, 3, 4]
        assert_eq!(draft(&toks, 3), vec![5, 9, 9]);
        assert_eq!(draft(&[5, 6, 7, 8], 4), Vec::<u32>::new());
        // a continuation running into the end is cut there
        assert_eq!(draft(&[1, 2, 3, 1, 2, 3], 8), vec![1, 2, 3]);
    }
}
