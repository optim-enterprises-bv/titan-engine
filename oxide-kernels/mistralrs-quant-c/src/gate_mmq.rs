//! MMQ part of the gate: the quantize_mmq launchers (3 layouts x f32 / f16 / bf16 input, and the
//! fused GLU f32 quantizers) and launch_mmq_gguf_<type> / launch_mmq_gguf_<type>_moe, the REAL C
//! launchers against the Rust twins on the same inputs, same context and stream. Compares every
//! byte of dst / vy and of the stream-k tmp_fixup buffer (both pre-filled with the same pattern).
use crate::gate::{Buf, G, Rng, encode, sync};
use crate::launch as rs;
use std::collections::HashSet;
use std::ffi::c_void;

type QuantFn = unsafe extern "C" fn(*const c_void, *const i32, *mut c_void, i32, i64, i64, i64, i64, i64, i64, i64, i64, *mut c_void);
type GluFn = unsafe extern "C" fn(*const f32, *const f32, *const i32, *mut c_void, i64, i64, i64, i64, i32, *mut c_void);
type MmqFn =
    unsafe extern "C" fn(*mut c_void, *const c_void, *const c_void, *mut c_void, i64, i64, i64, i64, i64, i32, i32, i64, i32, i32, *mut c_void);
type MoeFn = unsafe extern "C" fn(
    *mut c_void, *const c_void, *const c_void, *const i32, *const i32, *mut c_void, i64, i64, i64, i64, i64, i64, i64, i32, i32, i64, i32,
    *mut c_void,
);

mod cref {
    use std::ffi::c_void;
    macro_rules! decl {
        ($($dense:ident, $moe:ident;)*) => { unsafe extern "C" { $(
            pub fn $dense(f: *mut c_void, x: *const c_void, y: *const c_void, d: *mut c_void, a: i64, b: i64, c: i64, e: i64, g: i64,
                          cc: i32, nsm: i32, smpbo: i64, ws: i32, type_dst: i32, s: *mut c_void);
            pub fn $moe(f: *mut c_void, x: *const c_void, y: *const c_void, ids_dst: *const i32, eb: *const i32, d: *mut c_void,
                        ncols_x: i64, nrows_x: i64, ncols_dst: i64, stride_row_x: i64, stride_col_dst: i64, num_experts: i64,
                        ncols_max: i64, cc: i32, nsm: i32, smpbo: i64, ws: i32, s: *mut c_void);
        )* } };
    }
    decl! {
        launch_mmq_gguf_q4_0, launch_mmq_gguf_q4_0_moe;
        launch_mmq_gguf_q4_1, launch_mmq_gguf_q4_1_moe;
        launch_mmq_gguf_q5_0, launch_mmq_gguf_q5_0_moe;
        launch_mmq_gguf_q5_1, launch_mmq_gguf_q5_1_moe;
        launch_mmq_gguf_q8_0, launch_mmq_gguf_q8_0_moe;
        launch_mmq_gguf_q2_k, launch_mmq_gguf_q2_k_moe;
        launch_mmq_gguf_q3_k, launch_mmq_gguf_q3_k_moe;
        launch_mmq_gguf_q4_k, launch_mmq_gguf_q4_k_moe;
        launch_mmq_gguf_q5_k, launch_mmq_gguf_q5_k_moe;
        launch_mmq_gguf_q6_k, launch_mmq_gguf_q6_k_moe;
    }
    unsafe extern "C" {
        pub fn launch_mmq_quantize_q8_1_D4(x: *const c_void, ids: *const i32, vy: *mut c_void, t: i32, ne00: i64, s01: i64, s02: i64,
                                           s03: i64, ne0: i64, ne1: i64, ne2: i64, ne3: i64, s: *mut c_void);
        pub fn launch_mmq_quantize_q8_1_DS4(x: *const c_void, ids: *const i32, vy: *mut c_void, t: i32, ne00: i64, s01: i64, s02: i64,
                                            s03: i64, ne0: i64, ne1: i64, ne2: i64, ne3: i64, s: *mut c_void);
        pub fn launch_mmq_quantize_q8_1_D2S6(x: *const c_void, ids: *const i32, vy: *mut c_void, t: i32, ne00: i64, s01: i64, s02: i64,
                                             s03: i64, ne0: i64, ne1: i64, ne2: i64, ne3: i64, s: *mut c_void);
        pub fn launch_mmq_quantize_glu_q8_1_D4_f32(g: *const f32, u: *const f32, ids: *const i32, vy: *mut c_void, ne00: i64, s01: i64,
                                                   ne0: i64, ne1: i64, act: i32, s: *mut c_void);
        pub fn launch_mmq_quantize_glu_q8_1_DS4_f32(g: *const f32, u: *const f32, ids: *const i32, vy: *mut c_void, ne00: i64, s01: i64,
                                                    ne0: i64, ne1: i64, act: i32, s: *mut c_void);
        pub fn launch_mmq_quantize_glu_q8_1_D2S6_f32(g: *const f32, u: *const f32, ids: *const i32, vy: *mut c_void, ne00: i64, s01: i64,
                                                     ne0: i64, ne1: i64, act: i32, s: *mut c_void);
        pub fn launch_mmq_quantize_glu_q8_1_D4(g: *const c_void, u: *const c_void, ids: *const i32, vy: *mut c_void, type_x: i32, ne00: i64,
                                               s01: i64, ne0: i64, ne1: i64, act: i32, s: *mut c_void);
        pub fn launch_mmq_quantize_glu_q8_1_DS4(g: *const c_void, u: *const c_void, ids: *const i32, vy: *mut c_void, type_x: i32, ne00: i64,
                                                s01: i64, ne0: i64, ne1: i64, act: i32, s: *mut c_void);
        pub fn launch_mmq_quantize_glu_q8_1_D2S6(g: *const c_void, u: *const c_void, ids: *const i32, vy: *mut c_void, type_x: i32, ne00: i64,
                                                 s01: i64, ne0: i64, ne1: i64, act: i32, s: *mut c_void);
    }
}

/// The void-pointer (`<input_t>`-templated) GLU quantize launchers: `type_x` 0 f32, 1 f16, 30 bf16.
type GluTypedFn =
    unsafe extern "C" fn(*const c_void, *const c_void, *const i32, *mut c_void, i32, i64, i64, i64, i64, i32, *mut c_void);

fn glu_typed_pair(layout: usize) -> (GluTypedFn, GluTypedFn) {
    [
        (cref::launch_mmq_quantize_glu_q8_1_D4 as GluTypedFn, rs::launch_mmq_quantize_glu_q8_1_D4 as GluTypedFn),
        (cref::launch_mmq_quantize_glu_q8_1_DS4, rs::launch_mmq_quantize_glu_q8_1_DS4),
        (cref::launch_mmq_quantize_glu_q8_1_D2S6, rs::launch_mmq_quantize_glu_q8_1_D2S6),
    ][layout]
}

fn quant_pair(layout: usize) -> (QuantFn, QuantFn) {
    [
        (cref::launch_mmq_quantize_q8_1_D4 as QuantFn, rs::launch_mmq_quantize_q8_1_D4 as QuantFn),
        (cref::launch_mmq_quantize_q8_1_DS4, rs::launch_mmq_quantize_q8_1_DS4),
        (cref::launch_mmq_quantize_q8_1_D2S6, rs::launch_mmq_quantize_q8_1_D2S6),
    ][layout]
}
fn glu_pair(layout: usize) -> (GluFn, GluFn) {
    [
        (cref::launch_mmq_quantize_glu_q8_1_D4_f32 as GluFn, rs::launch_mmq_quantize_glu_q8_1_D4_f32 as GluFn),
        (cref::launch_mmq_quantize_glu_q8_1_DS4_f32, rs::launch_mmq_quantize_glu_q8_1_DS4_f32),
        (cref::launch_mmq_quantize_glu_q8_1_D2S6_f32, rs::launch_mmq_quantize_glu_q8_1_D2S6_f32),
    ][layout]
}

/// (name, block bytes, qk, f16 scale-field offsets, C / Rust dense, C / Rust moe, ds layout)
struct QType {
    name: &'static str,
    bytes: usize,
    qk: usize,
    halves: &'static [usize],
    dense: (MmqFn, MmqFn),
    moe: (MoeFn, MoeFn),
    layout: usize, // 0 = D4, 1 = DS4, 2 = D2S6
}

macro_rules! q {
    ($n:literal, $b:expr, $qk:expr, $h:expr, $d:ident, $m:ident, $l:expr) => {
        QType { name: $n, bytes: $b, qk: $qk, halves: $h, dense: (cref::$d, rs::$d), moe: (cref::$m, rs::$m), layout: $l }
    };
}

fn qtypes() -> Vec<QType> {
    // q4_k first: the order the families are run in.
    vec![
        q!("q4_k", 144, 256, &[0, 2], launch_mmq_gguf_q4_k, launch_mmq_gguf_q4_k_moe, 1),
        q!("q4_0", 18, 32, &[0], launch_mmq_gguf_q4_0, launch_mmq_gguf_q4_0_moe, 1),
        q!("q4_1", 20, 32, &[0, 2], launch_mmq_gguf_q4_1, launch_mmq_gguf_q4_1_moe, 1),
        q!("q5_0", 22, 32, &[0], launch_mmq_gguf_q5_0, launch_mmq_gguf_q5_0_moe, 0),
        q!("q5_1", 24, 32, &[0, 2], launch_mmq_gguf_q5_1, launch_mmq_gguf_q5_1_moe, 1),
        q!("q8_0", 34, 32, &[0], launch_mmq_gguf_q8_0, launch_mmq_gguf_q8_0_moe, 0),
        q!("q2_k", 84, 256, &[80, 82], launch_mmq_gguf_q2_k, launch_mmq_gguf_q2_k_moe, 2),
        q!("q3_k", 110, 256, &[108], launch_mmq_gguf_q3_k, launch_mmq_gguf_q3_k_moe, 0),
        q!("q5_k", 176, 256, &[0, 2], launch_mmq_gguf_q5_k, launch_mmq_gguf_q5_k_moe, 1),
        q!("q6_k", 210, 256, &[208], launch_mmq_gguf_q6_k, launch_mmq_gguf_q6_k_moe, 0),
    ]
}

const NSM_REAL: i32 = 60;
const SMPBO_REAL: i64 = 101376;

fn pad(p: usize, q: usize) -> usize {
    p.div_ceil(q) * q
}

/// f16 scale for mmq data: mostly moderate normals, sometimes NaN / inf / +-0 / denormal / raw bits.
fn mmq_half(rng: &mut Rng) -> u16 {
    if rng.below(100) < 88 {
        let e = 5 + rng.below(12) as u16; // 2^-10 .. 2^1
        let s = (rng.below(2) as u16) << 15;
        s | (e << 10) | (rng.next() as u16 & 0x3FF)
    } else {
        const SPECIAL: [u16; 12] = [0x7E00, 0x7C01, 0xFE00, 0x7C00, 0xFC00, 0x0000, 0x8000, 0x0001, 0x03FF, 0x8200, 0x7BFF, 0xF800];
        if rng.below(4) == 0 { rng.next() as u16 } else { SPECIAL[rng.below(SPECIAL.len() as u64) as usize] }
    }
}
fn mmq_f32(rng: &mut Rng) -> f32 {
    match rng.below(64) {
        0 => [0.0, -0.0, 1e-40, -3e-39, 65504.0, 1e5, -2e6, f32::INFINITY, f32::NAN][rng.below(9) as usize],
        1..=4 => 0.0,
        _ => ((rng.next() >> 11) as f64 / (1u64 << 53) as f64 * 8.0 - 4.0) as f32,
    }
}

/// x weights: `nblocks` random blocks with controlled f16 scales, plus identical tail bytes: the
/// kernels load whole 256-value slices, so the last row's slice runs up to 7 blocks past the tensor.
fn weights(rng: &mut Rng, q: &QType, nblocks: usize) -> Vec<u8> {
    let mut x = rng.bytes(nblocks * q.bytes);
    for b in 0..nblocks {
        for &h in q.halves {
            let v = mmq_half(rng);
            x[b * q.bytes + h..b * q.bytes + h + 2].copy_from_slice(&v.to_le_bytes());
        }
    }
    x.extend(rng.bytes(8 * q.bytes + 256));
    x
}

/// block_q8_1_mmq activations for `ncols` columns of `k` values: through the C quantize launcher
/// from random f32 (`random_y` false) or random bytes with controlled scales; generous tail.
fn activations(g: &mut G, q: &QType, k: usize, ncols: usize, random_y: bool, stream: *mut c_void) -> Buf {
    let k_padded = pad(pad(k, 512), 128);
    let ybytes = ncols * (k_padded / 128) * 144 + 300 * 144;
    let mut y = g.rng.bytes(ybytes);
    if random_y {
        for blk in 0..ybytes / 144 {
            for h in 0..8 {
                let v = mmq_half(&mut g.rng);
                y[blk * 144 + 2 * h..blk * 144 + 2 * h + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
    }
    let yb = Buf::new(&y);
    if !random_y {
        let xf: Vec<u8> = (0..ncols * k).flat_map(|_| mmq_f32(&mut g.rng).to_le_bytes()).collect();
        let xfb = Buf::new(&xf);
        unsafe { quant_pair(q.layout).0(xfb.p(), std::ptr::null(), yb.p(), 0, k as i64, k as i64, 0, 0, k_padded as i64, ncols as i64, 1, 1, stream) };
        sync("activation quantize");
    }
    yb
}

/// One dense MMQ case.
#[allow(clippy::too_many_arguments)]
fn mmq_case(g: &mut G, q: &QType, k: usize, nrows: usize, ncols: usize, nsm: i32, smpbo: i64, cc: i32, random_y: bool, type_dst: i32, sk: usize) {
    let stream = g.s(sk);
    let label = format!("{} k={k} rows={nrows} cols={ncols} nsm={nsm} smpbo={smpbo} cc={cc} yrand={random_y} type_dst={type_dst} s={}", q.name, sk % 2);
    let x = weights(&mut g.rng, q, nrows * k / q.qk);
    let yb = activations(g, q, k, ncols, random_y, stream);
    let xb = Buf::new(&x);
    let es = if type_dst == 1 || type_dst == 30 { 2 } else { 4 };
    let dst0 = g.rng.bytes(nrows * ncols * es + 64);
    let fix0 = g.rng.bytes(nsm.max(1) as usize * 128 * 128 * 4);
    let mut outs = vec![];
    for f in [q.dense.0, if g.self_test { q.dense.0 } else { q.dense.1 }, q.dense.0] {
        let dst = Buf::new(&dst0);
        let fix = Buf::new(&fix0);
        unsafe {
            f(fix.p(), xb.p(), yb.p(), dst.p(), k as i64, nrows as i64, ncols as i64, (k / q.qk) as i64, nrows as i64, cc, nsm, smpbo, 32,
              type_dst, stream)
        };
        sync(&label);
        outs.push((dst.read(), fix.read()));
    }
    let fam = format!("mmq_{}", q.name);
    let geo = Geo { qk: q.qk, k, nrows, ncols_max: ncols, nch: 1, plan: if cc >= 1200 { rs::plan(q.name, ncols as i64, nrows as i64, 1, cc, nsm, smpbo) } else { None } };
    g.t.calls(&fam, 2);
    g.t.cmp(&fam, &format!("{label}: dst"), &dst0, &outs[0].0, &outs[1].0);
    fixup_cmp(g, &fam, &label, &fix0, &outs[0].1, &outs[1].1, &outs[2].1, &geo);
    g.t.cmp(&fam, &format!("{label}: dst (C rerun)"), &dst0, &outs[0].0, &outs[2].0);
}

/// tmp_fixup: the reference itself is not deterministic in slots the fixup kernel never reads
/// (two C runs on identical inputs disagree), so Rust is compared on the bytes where both C runs
/// agree; the masked byte count is reported. dst is compared in full (and C vs C, too).
/// Geometry of a stream-k launch, for masking: (qk, k, nrows, ncols_max, nchannels, plan).
pub struct Geo {
    pub qk: usize,
    pub k: usize,
    pub nrows: usize,
    pub ncols_max: usize,
    pub nch: usize,
    pub plan: Option<(i32, u32)>,
}

/// Rows past the tensor in the partial tile a CUDA block writes to tmp_fixup hold values that
/// depend on uninitialised state in the reference (two C runs on identical inputs disagree there);
/// the fixup kernel never reads them. Returns, per 4-byte word, whether it is such a row.
fn beyond_rows(geo: &Geo, words: usize) -> Vec<bool> {
    let mut m = vec![false; words];
    let Some((mmq_x, grid)) = geo.plan else { return m };
    let bpn = (geo.k / geo.qk) as i64;
    let bpi = (256 / geo.qk) as i64;
    let ntx = (geo.ncols_max as i64 + mmq_x as i64 - 1) / mmq_x as i64;
    let nty = (geo.nrows as i64 + 127) / 128;
    let total = ntx * nty * geo.nch as i64 * bpn;
    let slot = mmq_x as usize * 128;
    for b in 0..grid as i64 {
        let mut stop = (b + 1) * total / grid as i64;
        stop -= (stop % bpn) % bpi;
        if stop == 0 {
            continue;
        }
        let tile = (stop - 1) / bpn;
        let it = tile / ntx / geo.nch as i64;
        let valid = geo.nrows as i64 - it * 128;
        for j in 0..mmq_x as usize {
            for i in 0..128usize {
                let w = b as usize * slot + j * 128 + i;
                if (i as i64) >= valid && w < words {
                    m[w] = true;
                }
            }
        }
    }
    m
}

fn fixup_cmp(g: &mut G, fam: &str, label: &str, init: &[u8], c1: &[u8], r: &[u8], c2: &[u8], geo: &Geo) {
    let mut a = c1.to_vec();
    let mut b = r.to_vec();
    let beyond = beyond_rows(geo, a.len() / 4);
    let mut masked = 0;
    let mut other_nd = 0;
    for i in 0..a.len() {
        if beyond[i / 4] {
            a[i] = 0;
            b[i] = 0;
            masked += 1;
        } else if c1[i] != c2[i] {
            other_nd += 1;
        }
    }
    if other_nd > 0 {
        g.t.failures.push(format!("{label}: reference tmp_fixup differs between two C runs in {other_nd} bytes of real rows"));
    }
    if std::env::var("MRQC_FIXDBG").is_ok() && a != b {
        let mut seen = std::collections::BTreeMap::new();
        for w in 0..a.len() / 4 {
            let (x, y) = (&a[4 * w..4 * w + 4], &b[4 * w..4 * w + 4]);
            let cdiff = c1[4 * w..4 * w + 4] != c2[4 * w..4 * w + 4];
            if x != y || cdiff {
                // slot of 128*128 floats max; report (slot by mmq_x=?) raw word index
                *seen.entry((w, x != y, cdiff)).or_insert(0usize) += 1;
            }
        }
        let nd = seen.keys().filter(|k| k.2).count();
        let (lo, hi) = (seen.keys().next().map(|k| k.0), seen.keys().last().map(|k| k.0));
        println!("FIXDBG c-vs-c words {nd}, range {lo:?}..{hi:?}");
        println!("FIXDBG {label}: {:?}", seen.keys().filter(|k| k.1).take(12).collect::<Vec<_>>());
    }
    if masked > 0 {
        g.t.fam.entry("zz_tmp_fixup_bytes_masked_(rows_past_tensor,_count_in_calls_column)".into()).or_default().0 += masked;
    }
    g.t.cmp(&format!("{fam}_fixupbuf"), &format!("{label}: tmp_fixup"), init, &a, &b);
}

/// One MoE MMQ case: `ne` experts with the given token counts (a sorted compact layout: expert e
/// owns y columns bounds[e]..bounds[e+1]); ids_dst a random permutation of the assignment rows.
#[allow(clippy::too_many_arguments)]
fn moe_case(g: &mut G, q: &QType, k: usize, nrows: usize, counts: &[usize], extra_max: usize, nsm: i32, random_y: bool, sk: usize, packed: bool) {
    let stream = g.s(sk);
    let ne = counts.len();
    let total: usize = counts.iter().sum();
    let ncols_max = counts.iter().copied().max().unwrap_or(0) + extra_max;
    // packed: dst column stride 2 * nrows (fast_mmq::grouped_pair_packed writes gate and up into one buffer)
    let sc = if packed { 2 * nrows } else { nrows };
    let label = format!("{} moe k={k} rows={nrows} counts={counts:?} ncols_max={ncols_max} nsm={nsm} yrand={random_y} stride_col_dst={sc} s={}", q.name, sk % 2);
    let x = weights(&mut g.rng, q, ne * nrows * k / q.qk);
    let yb = activations(g, q, k, total.max(1), random_y, stream);
    let xb = Buf::new(&x);
    let mut bounds = vec![0i32];
    for &c in counts {
        bounds.push(bounds.last().unwrap() + c as i32);
    }
    let mut perm: Vec<i32> = (0..total as i32).collect();
    for i in (1..perm.len()).rev() {
        let j = g.rng.below(i as u64 + 1) as usize;
        perm.swap(i, j);
    }
    // The kernels read ids_dst up to mmq_x (128) entries past the last expert's rows.
    for _ in 0..256 {
        perm.push(g.rng.below(total.max(1) as u64) as i32);
    }
    let ids = Buf::new(&perm.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    let eb = Buf::new(&bounds.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    let dst0 = g.rng.bytes(sc * total * 4 + 64);
    let fix0 = g.rng.bytes(nsm.max(1) as usize * 128 * 128 * 4);
    if packed {
        // The C launcher is not meant for stride_col_dst != nrows (it bounds the rows by it and fills the gap rows
        // from past the weight rows), so the reference is the C result at stride nrows, scattered into the packed
        // layout on the host: real rows bit-identical, gap rows untouched. Same tiles / stream-k split (nrows_x).
        let dst_c0 = g.rng.bytes(nrows * total * 4 + 64);
        let (dc, fc) = (Buf::new(&dst_c0), Buf::new(&fix0));
        let (dr, fr) = (Buf::new(&dst0), Buf::new(&fix0));
        let rust = if g.self_test { q.moe.0 } else { q.moe.1 };
        unsafe {
            (q.moe.0)(fc.p(), xb.p(), yb.p(), ids.p() as *const i32, eb.p() as *const i32, dc.p(), k as i64, nrows as i64, total as i64,
                    (k / q.qk) as i64, nrows as i64, ne as i64, ncols_max as i64, 1200, nsm, SMPBO_REAL, 32, stream);
            rust(fr.p(), xb.p(), yb.p(), ids.p() as *const i32, eb.p() as *const i32, dr.p(), k as i64, nrows as i64, total as i64,
                 (k / q.qk) as i64, sc as i64, ne as i64, ncols_max as i64, 1200, nsm, SMPBO_REAL, 32, stream);
        }
        sync(&label);
        let (c, r) = (dc.read(), dr.read());
        let mut want = dst0.clone();
        for col in 0..total {
            want[col * sc * 4..col * sc * 4 + nrows * 4].copy_from_slice(&c[col * nrows * 4..(col + 1) * nrows * 4]);
        }
        let fam = format!("mmq_{}", q.name);
        g.t.calls(&format!("{fam}_moe_packed"), 2);
        g.t.cmp(&format!("{fam}_moe_packed"), &format!("{label}: dst vs C at stride nrows, scattered"), &dst0, &want, &r);
        return;
    }
    let mut outs = vec![];
    for f in [q.moe.0, if g.self_test { q.moe.0 } else { q.moe.1 }, q.moe.0] {
        let dst = Buf::new(&dst0);
        let fix = Buf::new(&fix0);
        unsafe {
            f(fix.p(), xb.p(), yb.p(), ids.p() as *const i32, eb.p() as *const i32, dst.p(), k as i64, nrows as i64, total as i64,
              (k / q.qk) as i64, sc as i64, ne as i64, ncols_max as i64, 1200, nsm, SMPBO_REAL, 32, stream)
        };
        sync(&label);
        outs.push((dst.read(), fix.read()));
    }
    let fam = format!("mmq_{}", q.name);
    let geo = Geo { qk: q.qk, k, nrows, ncols_max, nch: ne, plan: rs::plan(q.name, ncols_max as i64, nrows as i64, ne as i64, 1200, nsm, SMPBO_REAL) };
    g.t.calls(&format!("{fam}_moe"), 2);
    g.t.cmp(&format!("{fam}_moe"), &format!("{label}: dst"), &dst0, &outs[0].0, &outs[1].0);
    fixup_cmp(g, &fam, &label, &fix0, &outs[0].1, &outs[1].1, &outs[2].1, &geo);
    g.t.cmp(&fam, &format!("{label}: dst (C rerun)"), &dst0, &outs[0].0, &outs[2].0);
}

/// One quantize case: C and Rust quantize launchers on the same input, compare the whole vy.
#[allow(clippy::too_many_arguments)]
fn quant_case(g: &mut G, layout: usize, type_x: i32, k: usize, s01: usize, ne1: usize, ne2: usize, ne3: usize, with_ids: bool, style: usize, sk: usize) {
    let stream = g.s(sk);
    let label = format!("quantize layout={layout} type_x={type_x} k={k} s01={s01} ne1={ne1} ne2={ne2} ne3={ne3} ids={with_ids} style={style}");
    let ne0 = pad(pad(k, 512), 128);
    let s02 = s01 * ne1 + 4 * (g.rng.below(3) as usize);
    let s03 = s02 * ne2 + 4 * (g.rng.below(3) as usize);
    let nx = s03 * ne3 + 16;
    let mut xv: Vec<f32> = (0..nx).map(|_| mmq_f32(&mut g.rng)).collect();
    if style == 1 {
        // whole groups of f32 denormals (abs.ftz / max.ftz flush them to 0)
        for v in xv.iter_mut() {
            *v = f32::from_bits((g.rng.next() as u32 & 0x807f_ffff) | if type_x == 30 { 0x0001_0000 } else { 1 });
        }
    }
    // roundf trap rows: with amax = 127 (d_inv ~ 1), t near +-0.5 rounds through add.rz; exact .5
    // ties go away from zero.
    if ne1 % 2 == 1 && style == 0 {
        for row in 0..ne1 {
            let b = row * s01;
            let trap = [127.0f32, 0.49999997, -0.49999997, 0.5, -0.5, 1.5, -2.5, 126.5];
            for (i, v) in trap.iter().enumerate() {
                if b + i < xv.len() {
                    xv[b + i] = *v;
                }
            }
        }
    }
    let t = match type_x {
        0 => 2,
        1 => 1,
        _ => 0,
    };
    let x = encode(&mut g.rng, t, &xv, if style == 0 { 16 } else { 0 });
    let xb = Buf::new(&x);
    let ids: Vec<u8> = (0..ne1).flat_map(|_| (g.rng.below(ne1 as u64) as i32).to_le_bytes()).collect();
    let idb = Buf::new(&ids);
    let nblk = ne2 * ne3 * ne1 * ne0 / 128;
    let y0 = g.rng.bytes(nblk * 144 + 144);
    let (c, r) = quant_pair(layout);
    let mut outs = vec![];
    for f in [c, if g.self_test { c } else { r }] {
        let yb = Buf::new(&y0);
        unsafe {
            f(xb.p(), if with_ids { idb.p() as *const i32 } else { std::ptr::null() }, yb.p(), type_x, k as i64, s01 as i64, s02 as i64,
              s03 as i64, ne0 as i64, ne1 as i64, ne2 as i64, ne3 as i64, stream)
        };
        sync(&label);
        outs.push(yb.read());
    }
    g.t.calls("mmq_quantize", 2);
    g.t.cmp("mmq_quantize", &label, &y0, &outs[0], &outs[1]);
}

/// One fused GLU quantize case (f32 gate / up).
#[allow(clippy::too_many_arguments)]
fn glu_case(g: &mut G, layout: usize, k: usize, s01: usize, ne1: usize, with_ids: bool, act: i32, sk: usize) {
    let stream = g.s(sk);
    let label = format!("quantize_glu layout={layout} k={k} s01={s01} ne1={ne1} ids={with_ids} act={act}");
    let ne0 = pad(pad(k, 512), 128);
    let nx = s01 * ne1 + 16;
    let gv: Vec<f32> = (0..nx).map(|_| mmq_f32(&mut g.rng) * 2.0).collect();
    let uv: Vec<f32> = (0..nx).map(|_| mmq_f32(&mut g.rng)).collect();
    let gb = Buf::new(&gv.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    let ub = Buf::new(&uv.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    let ids: Vec<u8> = (0..ne1).flat_map(|_| (g.rng.below(ne1 as u64) as i32).to_le_bytes()).collect();
    let idb = Buf::new(&ids);
    let y0 = g.rng.bytes(ne1 * ne0 / 128 * 144 + 144);
    let (c, r) = glu_pair(layout);
    let mut outs = vec![];
    for f in [c, if g.self_test { c } else { r }] {
        let yb = Buf::new(&y0);
        unsafe {
            f(gb.p() as *const f32, ub.p() as *const f32, if with_ids { idb.p() as *const i32 } else { std::ptr::null() }, yb.p(), k as i64,
              s01 as i64, ne0 as i64, ne1 as i64, act, stream)
        };
        sync(&label);
        outs.push(yb.read());
    }
    g.t.calls("mmq_quantize_glu", 2);
    g.t.cmp("mmq_quantize_glu", &label, &y0, &outs[0], &outs[1]);
}

/// One fused GLU quantize case through the `<input_t>`-templated (void-pointer) launcher:
/// `type_x` 0 f32, 1 f16, 30 bf16. gate / up are encoded in the target type, so the kernel's
/// `(input_t)act(g) * (input_t)u` rounding is what is compared.
#[allow(clippy::too_many_arguments)]
fn glu_typed_case(g: &mut G, layout: usize, type_x: i32, k: usize, s01: usize, ne1: usize, with_ids: bool, act: i32, sk: usize) {
    let stream = g.s(sk);
    let label = format!("quantize_glu_t layout={layout} type_x={type_x} k={k} s01={s01} ne1={ne1} ids={with_ids} act={act}");
    let ne0 = pad(pad(k, 512), 128);
    let nx = s01 * ne1 + 16;
    let gv: Vec<f32> = (0..nx).map(|_| mmq_f32(&mut g.rng) * 2.0).collect();
    let uv: Vec<f32> = (0..nx).map(|_| mmq_f32(&mut g.rng)).collect();
    let t = match type_x {
        0 => 2,
        1 => 1,
        _ => 0,
    };
    let gb = Buf::new(&encode(&mut g.rng, t, &gv, 16));
    let ub = Buf::new(&encode(&mut g.rng, t, &uv, 16));
    let ids: Vec<u8> = (0..ne1).flat_map(|_| (g.rng.below(ne1 as u64) as i32).to_le_bytes()).collect();
    let idb = Buf::new(&ids);
    let y0 = g.rng.bytes(ne1 * ne0 / 128 * 144 + 144);
    let (c, r) = glu_typed_pair(layout);
    let mut outs = vec![];
    for f in [c, if g.self_test { c } else { r }] {
        let yb = Buf::new(&y0);
        unsafe {
            f(gb.p(), ub.p(), if with_ids { idb.p() as *const i32 } else { std::ptr::null() }, yb.p(), type_x, k as i64,
              s01 as i64, ne0 as i64, ne1 as i64, act, stream)
        };
        sync(&label);
        outs.push(yb.read());
    }
    g.t.calls("mmq_quantize_glu_t", 2);
    g.t.cmp("mmq_quantize_glu_t", &label, &y0, &outs[0], &outs[1]);
}

pub fn run(g: &mut G, only: &Option<Vec<String>>, run: &mut HashSet<String>) {
    let want = |f: &str| only.as_ref().is_none_or(|o| o.iter().any(|n| n == f));
    let quick = std::env::var("MRQC_QUICK").is_ok();
    let mut sk = 0usize;

    if want("mmq_quantize") {
        for layout in 0..3 {
            for &type_x in &[0, 1, 30, 7] {
                for &(k, ds) in &[(256usize, 0usize), (300, 3), (512, 8), (1000, 4), (2048, 0), (4096, 12), (36, 0), (130, 1), (3, 0)] {
                    for &(ne1, ne2, ne3) in &[(1usize, 1usize, 1usize), (5, 1, 1), (64, 1, 1), (3, 2, 2), (17, 3, 1)] {
                        for style in 0..2 {
                            if style == 1 && ne1 > 5 {
                                continue;
                            }
                            sk += 1;
                            quant_case(g, layout, type_x, k, k + ds, ne1, ne2, ne3, ne1 % 2 == 1, style, sk);
                        }
                    }
                }
            }
            for act in -1..5 {
                for &(k, ds) in &[(256usize, 0usize), (300, 3), (1000, 4), (4096, 12), (130, 1), (3, 0)] {
                    for &ne1 in &[1usize, 5, 64] {
                        sk += 1;
                        glu_case(g, layout, k, k + ds, ne1, ne1 % 2 == 1, act, sk);
                    }
                }
            }

            for &type_x in &[0, 1, 30] {
                for act in -1..5 {
                    for &(k, ds) in &[(256usize, 0usize), (300, 3), (1000, 4), (4096, 12), (130, 1), (3, 0)] {
                        for &ne1 in &[1usize, 5, 64] {
                            sk += 1;
                            glu_typed_case(g, layout, type_x, k, k + ds, ne1, ne1 % 2 == 1, act, sk);
                        }
                    }
                }
            }
        }
        run.insert("mmq_quantize".into());
    }

    for q in qtypes() {
        if !want(q.name) {
            continue;
        }
        let ks: Vec<usize> = if q.qk == 256 { vec![256, 512, 768, 2048] } else { vec![256, 512, 544, 1024] };
        let rows = [128usize, 256, 100, 300, 1];
        let cols = [1usize, 7, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 63, 64, 65, 80, 81, 96, 97, 112, 113, 127, 128, 129, 200, 257];
        let tds = [0, 1, 30, 0, 5];
        // Every tile choice x need_check on the real device parameters, every output type.
        for (ci, &nc) in cols.iter().enumerate() {
            for (ri, &nr) in rows.iter().enumerate() {
                let k = ks[(ci + ri) % ks.len()];
                if quick && (ci + ri) % 3 != 0 {
                    continue;
                }
                sk += 1;
                mmq_case(g, &q, k, nr, nc, NSM_REAL, SMPBO_REAL, 1200, (ci + ri) % 2 == 1, tds[(ci * 7 + ri) % tds.len()], sk);
            }
        }
        // Stream-k decompositions: many CUDA-block counts (incl. more blocks than work), all out types.
        for &nsm in &[1, 2, 3, 5, 7, 13, 31, 64, 97, 150, 256] {
            for (si, &(k, nr, nc)) in [(2048usize, 300usize, 40usize), (4096, 128, 129), (768, 256, 7), (512, 1000, 64)].iter().enumerate() {
                let k = if q.qk == 32 { k / 2 + 32 } else { k };
                if quick && nsm % 2 == 0 {
                    continue;
                }
                sk += 1;
                mmq_case(g, &q, k, nr, nc, nsm, SMPBO_REAL, 1200, nsm % 3 == 0, [0, 1, 30][(si + nsm as usize) % 3], sk);
            }
        }
        // Tile efficiency >= 90%: one CUDA block per output tile, no fixup.
        for &(k, nr, nc, nsm) in &[(1024usize, 1280usize, 128usize, 10i32), (512, 512, 256, 8), (768, 384, 96, 3)] {
            sk += 1;
            mmq_case(g, &q, if q.qk == 32 { k / 2 } else { k }, nr, nc, nsm, SMPBO_REAL, 1200, false, 1, sk);
        }
        // Lower shared-memory budgets force smaller tiles; other cc >= 1200 take the same path.
        for &(smpbo, cc) in &[(49152i64, 1200), (30000, 1200), (20000, 1210), (101376, 1300), (15000, 1200)] {
            for &nc in &[20usize, 70, 130, 256] {
                sk += 1;
                mmq_case(g, &q, if q.qk == 32 { 1024 } else { 1536 }, 200, nc, NSM_REAL, smpbo, cc, nc == 70, [0, 1, 30, 0][sk % 4], sk);
            }
        }
        // MoE: empty experts, experts larger than a tile, ncols_max above the real maximum.
        let moe_shapes: [(&[usize], usize); 8] = [
            (&[5, 0, 17, 1], 0),
            (&[40, 33, 0, 0, 12], 0),
            (&[130, 2, 64], 0),
            (&[1, 1, 1, 1, 1, 1, 1, 1], 0),
            (&[9, 70], 30),
            (&[0, 0, 3], 0),
            (&[200, 150, 7, 90], 0),
            (&[16, 16, 16, 16], 5),
        ];
        for (mi, &(counts, extra)) in moe_shapes.iter().enumerate() {
            for &nsm in &[NSM_REAL, 7, 1, 150] {
                let k = if q.qk == 32 { [512usize, 256, 544][mi % 3] } else { [512usize, 256, 768][mi % 3] };
                let nr = [128usize, 100, 256, 300][(mi + nsm as usize) % 4];
                if quick && nsm != NSM_REAL {
                    continue;
                }
                sk += 1;
                moe_case(g, &q, k, nr, counts, extra, nsm, (mi + nsm as usize) % 2 == 1, sk, false);
                // redcell2: the packed gate/up layout (dst column stride 2 * nrows) on every shape at the real SM count
                if nsm == NSM_REAL {
                    sk += 1;
                    moe_case(g, &q, k, nr, counts, extra, nsm, mi % 2 == 0, sk, true);
                }
            }
        }
        run.insert(q.name.to_string());
    }
}
