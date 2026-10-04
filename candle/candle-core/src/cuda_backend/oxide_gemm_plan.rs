// Copied from titan-engine/oxide-kernels/gemm/src/plan.rs by its export.sh: edit there, not here.
//! Host-side kernel choice for the titan GEMM kernels (pure Rust; candle's cuda_backend/oxide_gemm.rs carries a
//! verbatim copy, so the gate tests exactly the plan the engine runs).
//!
//! The problem is the kernels' row-major one (see the crate doc). `plan` rewrites it before choosing:
//!   * batch folding: a B shared by every batch (sb == 0) whose A / D (/ C) batches are row-stacked becomes one
//!     batch of batch x m rows (the cuBLASLt GQA decode calls: n_rep query heads on one KV head); the mirror case
//!     (A shared, column-stacked B / D) folds into columns;
//!   * transposition: m > 8 >= n is solved as D^T = B^T A^T (a GEMV over the m columns);
//! then picks GEMV (m <= 8), tensor cores (f16 / bf16 with 16-byte aligned rows) or SIMT, a tile size and a
//! split-K factor that gives the GPU at least about two blocks per SM.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dt {
    F32,
    F16,
    Bf16,
}

impl Dt {
    pub fn size(self) -> i64 {
        match self {
            Dt::F32 => 4,
            _ => 2,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Dt::F32 => "f32",
            Dt::F16 => "f16",
            Dt::Bf16 => "bf16",
        }
    }
}

/// D(i, j) = alpha sum_k A(i, k) B(k, j) + beta C(i, j) + bias(i, j) per batch z (all strides in elements).
#[derive(Clone, Copy, Debug)]
pub struct Problem {
    pub dt: Dt,
    pub batch: i64,
    pub m: i64,
    pub n: i64,
    pub k: i64,
    pub a_s0: i64,
    pub a_s1: i64,
    pub b_s0: i64,
    pub b_s1: i64,
    pub d_s0: i64,
    pub d_s1: i64,
    pub c_s0: i64,
    pub c_s1: i64,
    pub bias_s0: i64,
    pub bias_s1: i64,
    pub sa: i64,
    pub sb: i64,
    pub sd: i64,
    pub sc: i64,
    pub alpha: f32,
    pub beta: f32,
    pub has_c: bool,
    pub has_bias: bool,
    /// a, b, d base addresses (bytes), for the alignment rules only
    pub a_addr: u64,
    pub b_addr: u64,
    pub d_addr: u64,
}

#[derive(Clone, Debug)]
pub struct Launch {
    pub kernel: String,
    pub grid: (u32, u32, u32),
    /// the problem the kernel sees (after folding / transposition)
    pub p: Problem,
    pub nsplit: i32,
    /// split-K workspace (f32 elements), 0 without split
    pub ws_floats: usize,
    /// split-K merge kernel and its grid
    pub reduce: Option<(String, (u32, u32, u32))>,
}

/// Overrides for the gate and the bench: force a kernel family / tile / split.
#[derive(Clone, Copy, Debug, Default)]
pub struct Force {
    /// 0: plan; 1: GEMV warp-per-column; 2: GEMV block-per-column; 3: tensor cores; 4: SIMT; 5: f64 reference
    pub family: u8,
    /// tensor-core BM or SIMT tile (0: plan)
    pub tile: i64,
    /// split-K factor (0: plan)
    pub nsplit: i64,
}

fn cdiv(a: i64, b: i64) -> i64 {
    (a + b - 1) / b
}

/// Transposed problem: D^T = B^T A^T.
pub fn transpose(p: &Problem) -> Problem {
    Problem {
        m: p.n,
        n: p.m,
        a_s0: p.b_s1,
        a_s1: p.b_s0,
        b_s0: p.a_s1,
        b_s1: p.a_s0,
        d_s0: p.d_s1,
        d_s1: p.d_s0,
        c_s0: p.c_s1,
        c_s1: p.c_s0,
        bias_s0: p.bias_s1,
        bias_s1: p.bias_s0,
        sa: p.sb,
        sb: p.sa,
        a_addr: p.b_addr,
        b_addr: p.a_addr,
        ..*p
    }
}

/// Folds a broadcast operand's batch into rows (B shared) or columns (A shared) when the other operand and the
/// outputs stack exactly.
pub fn fold(p: &Problem) -> Problem {
    if p.batch <= 1 {
        return *p;
    }
    let c_rows = !p.has_c || p.sc == p.m * p.c_s0;
    let bias_rows = !p.has_bias || p.bias_s0 == 0;
    if p.sb == 0 && p.sa == p.m * p.a_s0 && p.sd == p.m * p.d_s0 && c_rows && bias_rows {
        return Problem { batch: 1, m: p.batch * p.m, sa: 0, sd: 0, sc: 0, ..*p };
    }
    let c_cols = !p.has_c || p.sc == p.n * p.c_s1;
    let bias_cols = !p.has_bias || p.bias_s1 == 0;
    if p.sa == 0 && p.sb == p.n * p.b_s1 && p.sd == p.n * p.d_s1 && c_cols && bias_cols {
        return Problem { batch: 1, n: p.batch * p.n, sb: 0, sd: 0, sc: 0, ..*p };
    }
    *p
}

pub fn plan(p0: &Problem, sms: i64, f: Force) -> Result<Launch, String> {
    if f.family == 5 {
        let p = *p0;
        if p.batch < 1 || p.batch > 65535 || p.m > 65535 {
            return Err(format!("titan gemm: reference grid out of range (batch {}, m {})", p.batch, p.m));
        }
        return Ok(Launch { kernel: format!("gemm_ref_{}", p.dt.name()), grid: (cdiv(p.n, 256) as u32, p.m as u32, p.batch as u32), p, nsplit: 1, ws_floats: 0, reduce: None });
    }
    let mut p = fold(p0);
    if p.m > 8 && p.n <= 8 && f.family != 3 && f.family != 4 {
        p = transpose(&p);
    }
    let es = p.dt.size();
    for (v, what) in [(p.m, "m"), (p.n, "n"), (p.k, "k")] {
        if v > i32::MAX as i64 || v < 0 {
            return Err(format!("titan gemm: {what} = {v} out of range"));
        }
    }
    if p.batch < 1 || p.batch > 65535 {
        return Err(format!("titan gemm: batch {} out of range", p.batch));
    }
    let dt = p.dt.name();
    let one = |l: Launch| Ok(l);
    // ---- GEMV
    let gemv = f.family == 1 || f.family == 2 || (f.family == 0 && p.m <= 8);
    if gemv && p.m <= 8 {
        let r = if p.m <= 1 { 1 } else if p.m <= 2 { 2 } else if p.m <= 4 { 4 } else { 8 };
        if p.b_s0 == 1 || p.k == 1 {
            let e = 16 / es;
            let vec = p.b_s0 == 1
                && p.a_s1 == 1
                && p.k % e == 0
                && p.b_s1 % e == 0
                && (p.m == 1 || p.a_s0 % e == 0)
                && p.sa % e == 0
                && p.sb % e == 0
                && p.a_addr % 16 == 0
                && p.b_addr % 16 == 0;
            // a block per column when the columns alone cannot fill the GPU and each has a long k
            let per_block = match f.family {
                1 => false,
                2 => true,
                _ => p.n * p.batch < 32 * sms && p.k >= 512,
            };
            let v = if vec { "_v" } else { "" };
            return if per_block {
                one(Launch { kernel: format!("gemv_ks{r}_{dt}{v}"), grid: (p.n as u32, 1, p.batch as u32), p, nsplit: 1, ws_floats: 0, reduce: None })
            } else {
                one(Launch { kernel: format!("gemv_k{r}_{dt}{v}"), grid: (cdiv(p.n, 8) as u32, 1, p.batch as u32), p, nsplit: 1, ws_floats: 0, reduce: None })
            };
        }
        if p.b_s1 == 1 && f.family == 0 {
            return one(Launch { kernel: format!("gemv_n{r}_{dt}"), grid: (cdiv(p.n, 32) as u32, 1, p.batch as u32), p, nsplit: 1, ws_floats: 0, reduce: None });
        }
    }
    // ---- tensor cores
    let a_kc = p.a_s1 == 1 && p.k % 8 == 0 && (p.m == 1 || p.a_s0 % 8 == 0);
    let a_mc = p.a_s0 == 1 && p.m % 8 == 0 && (p.k == 1 || p.a_s1 % 8 == 0);
    let b_kc = p.b_s0 == 1 && p.k % 8 == 0 && (p.n == 1 || p.b_s1 % 8 == 0);
    let b_nc = p.b_s1 == 1 && p.n % 8 == 0 && (p.k == 1 || p.b_s0 % 8 == 0);
    let aligned = p.sa % 8 == 0 && p.sb % 8 == 0 && p.a_addr % 16 == 0 && p.b_addr % 16 == 0;
    let tc_ok = p.dt != Dt::F32 && aligned && (a_kc || a_mc) && (b_kc || b_nc);
    let split = |blocks: i64, tiles: i64, min_tiles: i64| -> i64 {
        if f.nsplit > 0 {
            return f.nsplit.min(tiles.max(1));
        }
        if blocks >= sms || tiles < 2 * min_tiles {
            return 1;
        }
        let want = cdiv(2 * sms, blocks);
        want.min(tiles / min_tiles).clamp(1, 16)
    };
    if (f.family == 0 && tc_ok) || (f.family == 3 && tc_ok) {
        let gx = cdiv(p.n, 128);
        let bm = if f.family == 3 && matches!(f.tile, 32 | 64 | 128) {
            f.tile
        } else if gx * cdiv(p.m, 128) * p.batch >= sms {
            128
        } else if gx * cdiv(p.m, 64) * p.batch >= sms || p.m > 32 {
            64
        } else {
            32
        };
        let gy = cdiv(p.m, bm);
        if gy > 65535 {
            return Err(format!("titan gemm: m = {} too large for the tensor-core grid", p.m));
        }
        let ns = split(gx * gy * p.batch, cdiv(p.k, 32), 4).min(65535 / p.batch);
        let lay = format!("{}{}", if a_kc { "k" } else { "m" }, if b_kc { "k" } else { "n" });
        return Ok(with_split(Launch {
            kernel: format!("tc_{dt}_{lay}_{bm}"),
            grid: (gx as u32, gy as u32, (p.batch * ns) as u32),
            p,
            nsplit: ns as i32,
            ws_floats: 0,
            reduce: None,
        }));
    }
    // ---- SIMT (any type, any strides)
    let at = p.a_s1 != 1 && p.a_s0 == 1;
    let bt = p.b_s1 == 1 && p.b_s0 != 1;
    // the 128 tile runs about twice as efficiently as the 64 one: keep it (split-K fills the GPU) unless a side is <= 64
    let t = if f.family == 4 && matches!(f.tile, 64 | 128) { f.tile } else if p.m.min(p.n) > 64 { 128 } else { 64 };
    let gx = cdiv(p.n, t);
    let gy = cdiv(p.m, t);
    if gy > 65535 {
        return Err(format!("titan gemm: m = {} too large for the SIMT grid", p.m));
    }
    let ns = split(gx * gy * p.batch, cdiv(p.k, 8), 16).min(65535 / p.batch);
    let lay = format!("{}{}", if at { "m" } else { "k" }, if bt { "n" } else { "k" });
    Ok(with_split(Launch {
        kernel: format!("simt_{dt}_{lay}_{t}"),
        grid: (gx as u32, gy as u32, (p.batch * ns) as u32),
        p,
        nsplit: ns as i32,
        ws_floats: 0,
        reduce: None,
    }))
}

fn with_split(mut l: Launch) -> Launch {
    if l.nsplit > 1 {
        let p = &l.p;
        l.ws_floats = (l.nsplit as i64 * p.batch * p.m * p.n) as usize;
        l.reduce = Some((format!("splitk_reduce_{}", p.dt.name()), (cdiv(p.m * p.n, 256) as u32, 1, p.batch as u32)));
    }
    l
}
