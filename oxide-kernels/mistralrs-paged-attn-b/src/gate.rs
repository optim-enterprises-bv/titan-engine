//! Launcher-level differential gate for mistralrs-paged-attn group 2 (FlashInfer decode + MLA decode):
//! every extern "C" launcher is called twice on identical inputs in the same (primary) context -- once
//! the REAL C launcher from libmistralrspagedattention.a (nvcc kernels, -O3 --use_fast_math), once the
//! pure-Rust twin in `crate::launch` (oxide kernels) -- and every byte of every buffer the call can see
//! is compared (outputs, split-KV workspaces, inputs, and a random guard region after each buffer), plus
//! the launcher's return code. On a mismatch the C launcher is re-run on fresh copies to flag a
//! reference that disagrees with itself (a race). At the end every reference kernel instance that is
//! reachable on this device must have been launched through its Rust twin.
//!
//! Env: MPB_ONLY=mla,decode,reshape,gather,errors (families); MPB_HD=64,128 (decode head dims);
//! MPB_QUICK=1 (fewer decode cases).
use crate::instances::{INSTANCES, Kind};
use crate::launch as ox;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use kdiff::{Rng, Tally, as_bytes};
use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::Arc;

/// The C launchers (libmistralrspagedattention.a).
pub mod cref {
    use std::ffi::c_void;
    unsafe extern "C" {
        pub fn cudaGetLastError() -> i32;
        pub fn flashinfer_decode(
            q: *mut c_void, key_cache: *mut c_void, value_cache: *mut c_void, kv_indptr: *const i32, kv_indices: *const i32,
            kv_last_page_len: *const i32, request_indices: *const i32, kv_tile_indices: *const i32, o_indptr: *const i32,
            kv_chunk_size_ptr: *const i32, block_valid_mask: *const u8, o: *mut c_void, tmp_v: *mut c_void, tmp_s: *mut c_void,
            batch_size: i32, padded_batch_size: i32, num_qo_heads: i32, num_kv_heads: i32, head_size: i32, page_size: i32,
            q_stride_n: i32, q_stride_h: i32, sm_scale: f32, window_left: i32, logits_soft_cap: f32, dtype: u32, stream: *mut c_void,
        ) -> i32;
        pub fn flashinfer_mla_decode(
            q_nope: *mut c_void, q_pe: *mut c_void, ckv_cache: *mut c_void, kpe_cache: *mut c_void, kv_indptr: *const i32,
            kv_indices: *const i32, kv_last_page_len: *const i32, o: *mut c_void, batch_size: i32, num_qo_heads: i32, page_size: i32,
            sm_scale: f32, window_left: i32, logits_soft_cap: f32, rope_scale: f32, rope_theta: f32, request_indices: *const i32,
            kv_tile_indices: *const i32, o_indptr: *const i32, kv_chunk_size_ptr: *const i32, dtype: u32, stream: *mut c_void,
        ) -> i32;
        pub fn reshape_and_cache_flashinfer(
            key: *mut c_void, value: *mut c_void, key_cache: *mut c_void, value_cache: *mut c_void, slot_mapping: *mut i64,
            num_tokens: i32, num_heads: i32, head_size: i32, block_size: i32, key_stride: i32, value_stride: i32, dtype: u32,
            stream: *mut c_void,
        );
        pub fn gather_kv_cache_flashinfer(
            key_cache: *mut c_void, value_cache: *mut c_void, k_out: *mut c_void, v_out: *mut c_void, block_table: *const i32,
            cu_seq_lens: *const i32, num_tokens: i32, num_seqs: i32, block_size: i32, block_table_stride: i32, num_kv_heads: i32,
            head_size: i32, dtype: u32, stream: *mut c_void,
        );
    }
}

/// Bytes of random guard data appended to every buffer (compared like the data).
const GUARD: usize = 64;

pub struct G {
    pub ctx: Arc<CudaContext>,
    pub streams: Vec<Arc<CudaStream>>,
    pub rng: Rng,
    pub t: Tally,
    pub calls: usize,
    pub races: usize,
    pub skipped: usize,
}

/// Device copies of a case's buffers for one side (C or Rust).
pub struct Side {
    pub bufs: Vec<DeviceBuffer<u8>>,
}
impl Side {
    pub fn p(&self, i: usize) -> *mut c_void {
        self.bufs[i].cu_deviceptr() as *mut c_void
    }
}

impl G {
    /// Stream `i`: 0 = null (the per-thread default stream of the reference), 1 = a created stream.
    pub fn stream(&self, i: usize) -> *mut c_void {
        if i == 0 { std::ptr::null_mut() } else { self.streams[1].cu_stream() as *mut c_void }
    }

    fn upload(&self, bufs: &[Vec<u8>]) -> Side {
        Side { bufs: bufs.iter().map(|b| DeviceBuffer::from_host(&self.streams[0], b).unwrap()).collect() }
    }
    fn download(&self, s: &Side) -> Vec<Vec<u8>> {
        s.bufs.iter().map(|x| x.to_host_vec(&self.streams[0]).unwrap()).collect()
    }

    /// Run `f` (returning the launcher's status, or 0 for void launchers) for the C side, then the Rust
    /// side, each on private copies of `bufs` (each followed by a random guard), and compare every byte.
    pub fn case(&mut self, label: &str, bufs: Vec<Vec<u8>>, f: impl Fn(bool, &Side) -> i32) {
        if std::env::var("MPB_VERBOSE").is_ok() {
            println!("  case {label}");
        }
        let bufs: Vec<Vec<u8>> = bufs
            .into_iter()
            .map(|mut b| {
                b.extend(self.rng.bytes(GUARD));
                b
            })
            .collect();
        let (a, b) = (self.upload(&bufs), self.upload(&bufs));
        self.ctx.synchronize().unwrap();
        unsafe { cref::cudaGetLastError() };
        let ra = f(false, &a);
        if std::env::var("MPB_VERBOSE").is_ok() { self.ctx.synchronize().unwrap(); println!("    C side done"); }
        self.ctx.synchronize().unwrap_or_else(|e| panic!("{label}: C side: {e:?}"));
        unsafe { cref::cudaGetLastError() };
        let rb = f(true, &b);
        self.ctx.synchronize().unwrap_or_else(|e| panic!("{label}: Rust side: {e:?}"));
        let (x, y) = (self.download(&a), self.download(&b));
        let mut d = kdiff::Diff { bytes: 4, differing: 0, first: None };
        if ra != rb {
            d.differing += 4;
            d.first = Some((usize::MAX, 0, ra as u8, rb as u8));
        }
        for (k, (xk, yk)) in x.iter().zip(&y).enumerate() {
            d.bytes += xk.len();
            for i in 0..xk.len() {
                if xk[i] != yk[i] {
                    d.differing += 1;
                    if d.first.is_none() {
                        d.first = Some((k, i, xk[i], yk[i]));
                    }
                }
            }
        }
        if d.differing > 0 {
            // race detection: does the reference agree with itself?
            let c = self.upload(&bufs);
            self.ctx.synchronize().unwrap();
            let rc = f(false, &c);
            self.ctx.synchronize().unwrap();
            let z = self.download(&c);
            if rc != ra || z != x {
                self.races += 1;
                println!("  RACE {label}: the C launcher disagrees with itself on identical inputs");
            }
            let (k, i, r, o) = d.first.unwrap();
            if k != usize::MAX {
                println!("  first diff {label}: buf {k} byte {i}: ref {r:#04x} oxide {o:#04x} (status {ra}/{rb})");
            } else {
                println!("  status diff {label}: ref {ra} oxide {rb}");
            }
        }
        self.calls += 2;
        self.t.record(label, &d);
    }

    /// `n` values of dtype `t` (0 f16, 1 bf16, 2 f32): random sign, magnitudes 2^-7 .. 4 with full
    /// random mantissas and a few zeros; `special` in 1/1000 replaced by NaN / +-inf / +-0 / denormals /
    /// huge values / raw bit patterns.
    pub fn vals(&mut self, t: u32, n: usize, special: u64) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::with_capacity(n * esize(t));
        for _ in 0..n {
            let r = self.rng.next();
            let sp = (r >> 40) % 1000 < special;
            let sign = (r >> 63) as u32;
            let e_off = ((r >> 20) % 10) as i32 - 7; // 2^-7 .. 2^2
            let zero = (r >> 32) % 64 == 0;
            match t {
                2 => {
                    let bits = if sp {
                        [0x7fc0_0000u32, 0xffc0_0001, 0x7f80_0000, 0xff80_0000, 0x8000_0000, 0x0000_0001, 0x8040_0000, 0x7f7f_ffff, 0xfeff_0000, self.rng.next() as u32][(r % 10) as usize]
                    } else if zero {
                        sign << 31
                    } else {
                        (sign << 31) | (((127 + e_off) as u32) << 23) | ((self.rng.next() as u32) & 0x7f_ffff)
                    };
                    out.extend(bits.to_le_bytes());
                }
                1 => {
                    let bits: u16 = if sp {
                        [0x7fc0u16, 0xffc1, 0x7f80, 0xff80, 0x8000, 0x0001, 0x8040, 0x7f7f, 0xfeff, self.rng.next() as u16][(r % 10) as usize]
                    } else if zero {
                        (sign << 15) as u16
                    } else {
                        ((sign << 15) | (((127 + e_off) as u32) << 7) | ((self.rng.next() as u32) & 0x7f)) as u16
                    };
                    out.extend(bits.to_le_bytes());
                }
                _ => {
                    let bits: u16 = if sp {
                        [0x7e00u16, 0xfe01, 0x7c00, 0xfc00, 0x8000, 0x0001, 0x8200, 0x7bff, 0xfbff, self.rng.next() as u16][(r % 10) as usize]
                    } else if zero {
                        (sign << 15) as u16
                    } else {
                        ((sign << 15) | (((15 + e_off) as u32) << 10) | ((self.rng.next() as u32) & 0x3ff)) as u16
                    };
                    out.extend(bits.to_le_bytes());
                }
            }
        }
        out
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.rng.next() % n.max(1) as u64) as usize
    }
}

pub fn esize(dtype: u32) -> usize {
    if dtype == 2 { 4 } else { 2 }
}
pub fn i32s(v: &[i32]) -> Vec<u8> {
    as_bytes(v)
}

/// FlashInfer paged-KV CSR (make_paged_kv_tensors): pages drawn at random from a pool of `num_pages`.
pub struct Paged {
    pub indptr: Vec<i32>,
    pub indices: Vec<i32>,
    pub last: Vec<i32>,
    pub num_pages: usize,
}
pub fn make_paged(g: &mut G, lens: &[usize], page_size: usize, pad: usize) -> Paged {
    let blocks: Vec<usize> = lens.iter().map(|l| l.div_ceil(page_size)).collect();
    let total: usize = blocks.iter().sum();
    let num_pages = total + 1 + g.below(3);
    let mut pool: Vec<i32> = (0..num_pages as i32).collect();
    for i in (1..pool.len()).rev() {
        let j = g.below(i + 1);
        pool.swap(i, j);
    }
    let mut indptr = vec![0i32];
    let mut indices = Vec::new();
    let mut last = Vec::new();
    let mut k = 0;
    for (b, &l) in blocks.iter().zip(lens) {
        for _ in 0..*b {
            indices.push(pool[k]);
            k += 1;
        }
        indptr.push(indices.len() as i32);
        last.push(if *b == 0 { 0 } else { (l - (b - 1) * page_size) as i32 });
    }
    indices.resize(indices.len() + pad, 0);
    Paged { indptr, indices, last, num_pages }
}

/// make_paged_kv_decode_tensors: request_indices, kv_tile_indices, o_indptr, kv_chunk_size, mask, padded.
pub struct Plan {
    pub req: Vec<i32>,
    pub tiles: Vec<i32>,
    pub o_indptr: Vec<i32>,
    pub chunk: i32,
    pub mask: Vec<u8>,
}
pub fn make_plan(lens: &[usize], page_size: usize, split_pages: Option<usize>, pad: usize) -> Plan {
    let chunk_pages = split_pages.unwrap_or(usize::MAX).max(1);
    let mut req = Vec::new();
    let mut tiles = Vec::new();
    let mut o_indptr = vec![0i32];
    for (b, l) in lens.iter().enumerate() {
        let nb = l.div_ceil(page_size);
        let nc = nb.max(1).div_ceil(chunk_pages);
        for t in 0..nc {
            req.push(b as i32);
            tiles.push(t as i32);
        }
        o_indptr.push(req.len() as i32);
    }
    let valid = req.len();
    req.resize(valid + pad, 0);
    tiles.resize(valid + pad, 0);
    let mut mask = vec![1u8; valid];
    mask.resize(valid + pad, 0);
    Plan { req, tiles, o_indptr, chunk: (split_pages.unwrap_or(1) * page_size) as i32, mask }
}

// ================================================================================================
// MLA decode

struct MlaCase {
    dtype: u32,
    lens: Vec<usize>,
    heads: usize,
    page_size: usize,
    special: u64,
    sm_scale: f32,
    stream: usize,
    /// kv_chunk_size / kv_tile_indices contents (read by the kernel; only used when partitioned)
    chunk: i32,
}

fn mla_case(g: &mut G, c: &MlaCase) {
    let b = c.lens.len();
    let es = esize(c.dtype);
    let pad = g.below(3);
    let pg = make_paged(g, &c.lens, c.page_size, pad);
    let q_nope = g.vals(c.dtype, b * c.heads * 512, c.special);
    let q_pe = g.vals(c.dtype, b * c.heads * 64, c.special);
    let ckv = g.vals(c.dtype, pg.num_pages * c.page_size * 512, c.special);
    let kpe = g.vals(c.dtype, pg.num_pages * c.page_size * 64, c.special);
    let out = g.rng.bytes(b * c.heads * 512 * es);
    let req: Vec<i32> = (0..b as i32).collect();
    let tiles: Vec<i32> = (0..b).map(|_| g.below(3) as i32).collect();
    let o_indptr: Vec<i32> = (0..=b as i32).collect();
    let bufs = vec![
        q_nope,
        q_pe,
        ckv,
        kpe,
        i32s(&pg.indptr),
        i32s(&pg.indices),
        i32s(&pg.last),
        out,
        i32s(&req),
        i32s(&tiles),
        i32s(&o_indptr),
        i32s(&[c.chunk]),
    ];
    let label = format!("mla dt{} heads{} ps{} lens{:?} sp{}", c.dtype, c.heads, c.page_size, c.lens, c.special);
    let (heads, ps, sm, dt) = (c.heads as i32, c.page_size as i32, c.sm_scale, c.dtype);
    let st = g.stream(c.stream);
    g.case(&label, bufs, |rust, s| unsafe {
        let f = if rust { ox::flashinfer_mla_decode } else { cref::flashinfer_mla_decode };
        f(s.p(0), s.p(1), s.p(2), s.p(3), s.p(4) as _, s.p(5) as _, s.p(6) as _, s.p(7), b as i32, heads, ps, sm, -1, 0.0, 1.0, 1.0, s.p(8) as _, s.p(9) as _, s.p(10) as _, s.p(11) as _, dt, st)
    });
}

fn mla(g: &mut G) {
    for dtype in [0u32, 1, 2] {
        for &ps in &[1usize, 3, 7, 16, 64] {
            for &heads in &[1usize, 16, 20, 128] {
                let lens: Vec<usize> = match (ps, heads) {
                    (1, _) => vec![0, 1, 7, 8, 9, 257],
                    (3, _) => vec![3, 5, 6, 24, 25],
                    (7, _) => vec![7, 8, 13, 14, 15, 700],
                    (16, 128) => vec![16, 300],
                    (16, _) => vec![1, 15, 16, 17, 255, 256, 257],
                    (_, 128) => vec![64, 65],
                    _ => vec![63, 64, 65, 520, 2 * 64 * 32 + 5],
                };
                let special = if heads == 16 && ps == 16 { 30 } else { 0 };
                let sm = 1.0 / ((512.0f32 + 64.0).sqrt());
                mla_case(g, &MlaCase { dtype, lens, heads, page_size: ps, special, sm_scale: sm, stream: (heads % 2), chunk: 64 });
            }
        }
        // odd sm_scale values, denormal / special heavy data, per-thread default stream
        mla_case(g, &MlaCase { dtype, lens: vec![33, 70], heads: 16, page_size: 16, special: 200, sm_scale: 0.1, stream: 0, chunk: 32 });
        mla_case(g, &MlaCase { dtype, lens: vec![40], heads: 17, page_size: 8, special: 0, sm_scale: 3.0, stream: 1, chunk: 32 });
        mla_case(g, &MlaCase { dtype, lens: vec![40, 3], heads: 32, page_size: 8, special: 5, sm_scale: 1e-3, stream: 1, chunk: 32 });
    }
    // error path: unsupported dtype; empty batch (grid 0)
    mla_case_err(g, 3);
    mla_case(g, &MlaCase { dtype: 1, lens: vec![], heads: 16, page_size: 16, special: 0, sm_scale: 0.1, stream: 1, chunk: 32 });
}

fn mla_case_err(g: &mut G, dtype: u32) {
    let bufs = vec![vec![0u8; 64]; 12];
    let st = g.stream(1);
    g.case(&format!("mla bad dtype {dtype}"), bufs, |rust, s| unsafe {
        let f = if rust { ox::flashinfer_mla_decode } else { cref::flashinfer_mla_decode };
        f(s.p(0), s.p(1), s.p(2), s.p(3), s.p(4) as _, s.p(5) as _, s.p(6) as _, s.p(7), 1, 16, 16, 0.1, -1, 0.0, 1.0, 1.0, s.p(8) as _, s.p(9) as _, s.p(10) as _, s.p(11) as _, dtype, st)
    });
}

// ================================================================================================
// batch decode

#[derive(Clone)]
pub struct DecCase {
    pub dtype: u32,
    pub hd: usize,
    pub lens: Vec<usize>,
    pub nkv: usize,
    pub group: usize,
    pub page_size: usize,
    /// Some(split_pages): split-KV (tmp_v / tmp_s workspaces) when it produces more tiles than requests
    pub split: Option<usize>,
    pub pad: usize,
    pub window_left: i32,
    pub cap: f32,
    pub sm_scale: f32,
    pub special: u64,
    /// q rows are this many heads apart (>= num_qo_heads: a slice of a fused qkv tensor)
    pub q_row_heads: usize,
    pub null_mask: bool,
    pub force_tmp: Option<bool>,
    pub stream: usize,
}

pub fn dec_case(g: &mut G, c: &DecCase) {
    if [1usize, 2, 3, 4, 8, 16].contains(&c.group) && [64usize, 128, 256, 512].contains(&c.hd) && c.dtype <= 2 {
        let n = ox::decode_kernel_name(c.dtype, c.hd as u32, c.group as u32, c.window_left >= 0, c.cap > 0.0);
        if let Some(n) = n {
            if !ox::available(n) {
                g.skipped += 1;
                return;
            }
        }
    }
    let b = c.lens.len();
    let es = esize(c.dtype);
    let nqo = c.nkv * c.group;
    let pad = g.below(3);
    let pg = make_paged(g, &c.lens, c.page_size, pad);
    let plan = make_plan(&c.lens, c.page_size, c.split, c.pad);
    let padded = plan.req.len();
    let split_kv = padded > b;
    let use_tmp = c.force_tmp.unwrap_or(split_kv);
    let q_rows = c.q_row_heads.max(nqo);
    let q = g.vals(c.dtype, b.max(1) * q_rows * c.hd, c.special);
    let kc = g.vals(c.dtype, pg.num_pages * c.nkv * c.page_size * c.hd, c.special);
    let vc = g.vals(c.dtype, pg.num_pages * c.nkv * c.page_size * c.hd, c.special);
    let out = g.rng.bytes(b * nqo * c.hd * es);
    let tmp_v = g.rng.bytes(padded.max(1) * nqo * c.hd * es);
    let tmp_s = g.rng.bytes(padded.max(1) * nqo * 4);
    let bufs = vec![
        q,
        kc,
        vc,
        i32s(&pg.indptr),
        i32s(&pg.indices),
        i32s(&pg.last),
        i32s(&plan.req),
        i32s(&plan.tiles),
        i32s(&plan.o_indptr),
        i32s(&[plan.chunk]),
        plan.mask.clone(),
        out,
        tmp_v,
        tmp_s,
    ];
    let label = format!(
        "decode dt{} hd{} g{} nkv{} ps{} lens{:?} split{:?} pad{} wl{} cap{} sp{} qrow{} tmp{}",
        c.dtype, c.hd, c.group, c.nkv, c.page_size, c.lens, c.split, c.pad, c.window_left, c.cap, c.special, q_rows, use_tmp
    );
    let st = g.stream(c.stream);
    let (dt, hd, nkv, ps, wl, cap, sm, null_mask) = (c.dtype, c.hd as i32, c.nkv as i32, c.page_size as i32, c.window_left, c.cap, c.sm_scale, c.null_mask);
    let (qsn, qsh) = ((q_rows * c.hd) as i32, c.hd as i32);
    g.case(&label, bufs, |rust, s| unsafe {
        let f = if rust { ox::flashinfer_decode } else { cref::flashinfer_decode };
        let mask = if null_mask { std::ptr::null() } else { s.p(10) as *const u8 };
        let (tv, ts) = if use_tmp { (s.p(12), s.p(13)) } else { (std::ptr::null_mut(), std::ptr::null_mut()) };
        f(
            s.p(0), s.p(1), s.p(2), s.p(3) as _, s.p(4) as _, s.p(5) as _, s.p(6) as _, s.p(7) as _, s.p(8) as _, s.p(9) as _, mask, s.p(11), tv, ts,
            b as i32, padded as i32, nqo as i32, nkv, hd, ps, qsn, qsh, sm, wl, cap, dt, st,
        )
    });
}

/// Sequence lengths around page / chunk / refill boundaries for one decode configuration.
fn lens_for(g: &mut G, ps: usize, per_iter: usize, long: bool) -> Vec<usize> {
    let mut v = vec![1, ps, ps + 1, per_iter.max(1) * 3 + 1];
    if long {
        v.push(per_iter * 40 + ps - 1);
    }
    v.push(g.below(200) + 2);
    v
}

fn decode(g: &mut G, hds: &[usize], quick: bool) {
    for dtype in [0u32, 1, 2] {
        let es = esize(dtype);
        for &hd in hds {
            let vec = (16 / es).max(hd / 32);
            let bdx = hd / vec;
            for &group in &[1usize, 2, 3, 4, 8, 16] {
                let nt = 128.max(bdx * group);
                let bdz = nt / (bdx * group);
                let tile = if group == 1 { 4 } else { 1 };
                let per_iter = tile * group * bdz;
                for (sw, sc) in [(false, false), (true, false), (false, true), (true, true)] {
                    let sm = 1.0 / (hd as f32).sqrt();
                    let ps = [16usize, 1, 7, 64, 21, 3][(group + sw as usize * 2 + sc as usize + dtype as usize) % 6];
                    let nkv = if group == 16 { 1 } else { 1 + g.below(2) };
                    let long = !quick && (group == 1 || group == 8) && sw == sc;
                    let lens = lens_for(g, ps, per_iter, long);
                    let wl = if sw { [0, 5, 17, 300][g.below(4)] } else { -1 };
                    let cap = if sc { [30.0f32, 50.0, 1.5][g.below(3)] } else { 0.0 };
                    let base = DecCase {
                        dtype, hd, lens, nkv, group, page_size: ps, split: None, pad: 0, window_left: wl, cap, sm_scale: sm, special: 0,
                        q_row_heads: 0, null_mask: false, force_tmp: None, stream: 1,
                    };
                    // non-split, contiguous q
                    dec_case(g, &base);
                    // split-KV with padding tiles (workspaces + merge), fused-qkv q strides
                    let mut c = base.clone();
                    c.split = Some([1usize, 2, 3][g.below(3)]);
                    c.pad = 2;
                    c.q_row_heads = group * nkv + 2 * nkv;
                    c.special = if quick { 0 } else { 3 };
                    c.stream = 0;
                    dec_case(g, &c);
                    if !quick {
                        // non-split with a null block_valid_mask, special values
                        let mut c = base.clone();
                        c.null_mask = true;
                        c.special = 20;
                        c.lens = vec![ps * 2 + 1, 0, 7];
                        dec_case(g, &c);
                    }
                }
            }
        }
    }
}

/// Extra decode cases: window / soft-cap edge values, split plans with single-chunk requests, empty
/// sequences, masked padded tiles without workspaces, dense specials.
fn decode_edges(g: &mut G) {
    for dtype in [0u32, 1, 2] {
        let mk = |hd: usize, group: usize| DecCase {
            dtype, hd, lens: vec![5, 40, 0, 17], nkv: 2, group, page_size: 4, split: None, pad: 0, window_left: -1, cap: 0.0,
            sm_scale: 0.125, special: 0, q_row_heads: 0, null_mask: false, force_tmp: None, stream: 1,
        };
        for wl in [0, 1, 4, 16, 39, 40, 41, 1000, i32::MAX] {
            let mut c = mk(128, 4);
            c.window_left = wl;
            dec_case(g, &c);
        }
        for cap in [1e-3f32, 0.5, 1e30, f32::INFINITY, f32::NAN, -1.0, f32::from_bits(1)] {
            let mut c = mk(64, 2);
            c.cap = cap;
            dec_case(g, &c);
        }
        for sm in [0.0f32, -0.5, 1e10, f32::NAN] {
            let mut c = mk(256, 1);
            c.sm_scale = sm;
            dec_case(g, &c);
        }
        // split where some requests fit in one chunk (merge copy path) and empty requests
        let mut c = mk(128, 8);
        c.lens = vec![0, 3, 4, 5, 33, 1];
        c.split = Some(1);
        c.pad = 3;
        dec_case(g, &c);
        // masked padded tiles without workspaces
        let mut c = mk(64, 3);
        c.pad = 4;
        c.force_tmp = Some(false);
        dec_case(g, &c);
        // workspaces with padded_batch == batch (single chunks: every merge is a copy)
        let mut c = mk(512, 2);
        c.force_tmp = Some(true);
        dec_case(g, &c);
        // dense specials
        let mut c = mk(128, 16);
        c.nkv = 1;
        c.special = 300;
        c.split = Some(2);
        dec_case(g, &c);
    }
}

fn decode_errors(g: &mut G) {
    let base = DecCase {
        dtype: 1, hd: 128, lens: vec![5, 9], nkv: 2, group: 2, page_size: 4, split: None, pad: 0, window_left: -1, cap: 0.0,
        sm_scale: 0.1, special: 0, q_row_heads: 0, null_mask: false, force_tmp: None, stream: 1,
    };
    // unsupported GQA group (exception -> 999)
    let mut c = base.clone();
    c.group = 5;
    dec_case(g, &c);
    // f32 head dim 512, group 1 / 16: 135168 bytes of dynamic shared memory (> device limit)
    for group in [1usize, 16] {
        let mut c = base.clone();
        c.dtype = 2;
        c.hd = 512;
        c.group = group;
        c.nkv = 1;
        dec_case(g, &c);
    }
    // empty batch (grid 0 -> invalid configuration)
    let mut c = base.clone();
    c.lens = vec![];
    dec_case(g, &c);
    // bad dtype / head size
    for (dt, hd) in [(3u32, 128i32), (0, 96)] {
        let bufs = vec![vec![0u8; 64]; 14];
        let st = g.stream(1);
        g.case(&format!("decode bad dtype {dt} hd {hd}"), bufs, |rust, s| unsafe {
            let f = if rust { ox::flashinfer_decode } else { cref::flashinfer_decode };
            f(
                s.p(0), s.p(1), s.p(2), s.p(3) as _, s.p(4) as _, s.p(5) as _, s.p(6) as _, s.p(7) as _, s.p(8) as _, s.p(9) as _, s.p(10) as _,
                s.p(11), std::ptr::null_mut(), std::ptr::null_mut(), 1, 1, 2, 1, hd, 16, 256, 128, 0.1, -1, 0.0, dt, st,
            )
        });
    }
}

// ================================================================================================
// reshape_and_cache_flashinfer / gather_kv_cache_flashinfer

fn reshape(g: &mut G) {
    for dtype in [0u32, 1, 2] {
        for &(nt, nh, hs, bs) in &[(5usize, 2usize, 64usize, 16usize), (7, 4, 128, 1), (3, 8, 256, 5), (1, 1, 512, 64), (9, 3, 96, 3)] {
            let es = esize(dtype);
            let num_blocks = nt.div_ceil(bs) + 2;
            let stride_extra = [0usize, 64][nt % 2];
            let ks = nh * hs + stride_extra;
            let vs = nh * hs + 2 * stride_extra;
            let key = g.rng.bytes(nt * ks * es);
            let value = g.rng.bytes(nt * vs * es);
            let kc = g.rng.bytes(num_blocks * nh * bs * hs * es);
            let vcache = g.rng.bytes(num_blocks * nh * bs * hs * es);
            let mut slots: Vec<i64> = (0..(num_blocks * bs) as i64).collect();
            for i in (1..slots.len()).rev() {
                let j = g.below(i + 1);
                slots.swap(i, j);
            }
            slots.truncate(nt);
            if nt > 2 {
                slots[1] = -1; // padding token
            }
            let bufs = vec![key, value, kc, vcache, as_bytes(&slots)];
            let st = g.stream(nt % 2);
            g.case(&format!("reshape_fi dt{dtype} nt{nt} nh{nh} hs{hs} bs{bs}"), bufs, |rust, s| unsafe {
                let f = if rust { ox::reshape_and_cache_flashinfer } else { cref::reshape_and_cache_flashinfer };
                f(s.p(0), s.p(1), s.p(2), s.p(3), s.p(4) as _, nt as i32, nh as i32, hs as i32, bs as i32, ks as i32, vs as i32, dtype, st);
                0
            });
        }
    }
}

fn gather(g: &mut G) {
    for dtype in [0u32, 1, 2] {
        for &(nkv, hs, bs) in &[(2usize, 64usize, 16usize), (1, 128, 1), (4, 256, 5), (1, 512, 64), (3, 96, 3)] {
            let es = esize(dtype);
            let lens = [bs + 1, 1, 2 * bs, 3];
            let nseq = lens.len();
            let stride = lens.iter().map(|l| l.div_ceil(bs)).max().unwrap() + 1;
            let num_blocks = nseq * stride;
            let mut table: Vec<i32> = (0..num_blocks as i32).collect();
            for i in (1..table.len()).rev() {
                let j = g.below(i + 1);
                table.swap(i, j);
            }
            let mut cu = vec![0i32];
            for l in lens {
                cu.push(cu.last().unwrap() + l as i32);
            }
            let nt = *cu.last().unwrap() as usize;
            let kc = g.rng.bytes(num_blocks * nkv * bs * hs * es);
            let vcache = g.rng.bytes(num_blocks * nkv * bs * hs * es);
            let ko = g.rng.bytes(nt * nkv * hs * es);
            let vo = g.rng.bytes(nt * nkv * hs * es);
            let bufs = vec![kc, vcache, ko, vo, i32s(&table), i32s(&cu)];
            let st = g.stream(nkv % 2);
            g.case(&format!("gather_fi dt{dtype} nkv{nkv} hs{hs} bs{bs} nt{nt}"), bufs, |rust, s| unsafe {
                let f = if rust { ox::gather_kv_cache_flashinfer } else { cref::gather_kv_cache_flashinfer };
                f(s.p(0), s.p(1), s.p(2), s.p(3), s.p(4) as _, s.p(5) as _, nt as i32, nseq as i32, bs as i32, stride as i32, nkv as i32, hs as i32, dtype, st);
                0
            });
        }
        // unsupported dtype: no launch
    }
    let bufs = vec![vec![0u8; 64]; 6];
    let st = g.stream(1);
    g.case("gather_fi bad dtype", bufs, |rust, s| unsafe {
        let f = if rust { ox::gather_kv_cache_flashinfer } else { cref::gather_kv_cache_flashinfer };
        f(s.p(0), s.p(1), s.p(2), s.p(3), s.p(4) as _, s.p(5) as _, 1, 1, 16, 1, 1, 64, 7, st);
        0
    });
}

// ================================================================================================

/// Every reference instance reachable on this device must have been launched through its twin.
fn coverage(cc_major: i32, max_smem: u32) -> (usize, usize, Vec<String>) {
    let launched = ox::LAUNCHED.lock().unwrap().clone().unwrap_or_default();
    let stages = if cc_major >= 8 { 2 } else { 1 };
    let mut reach = 0;
    let mut hit = 0;
    let mut notes = Vec::new();
    let mut missing = Vec::new();
    let mut unreach: std::collections::BTreeMap<String, usize> = Default::default();
    for i in INSTANCES {
        let why: Option<String> = match i.kind {
            Kind::Decode | Kind::Mla if i.stages != stages => Some(format!("NUM_STAGES_SMEM={} (compute capability major {cc_major} selects {stages})", i.stages)),
            Kind::MergeMla => Some("MLA-cubin merge instance (run_mla_decode never passes workspaces; same template as the decode cubin's)".into()),
            Kind::Decode => {
                let es = if i.dtype == 2 { 4 } else { 2 };
                let hd = i.vec * i.bdx;
                let nt = i.bdx * i.bdy * i.bdz;
                let smem = 2 * i.stages * i.tile * i.bdy * i.bdz * hd * es + (i.tile * nt * 8).max(2 * i.bdy * i.bdz * 4);
                if smem > max_smem { Some(format!("needs {smem} bytes of dynamic shared memory (device max {max_smem}): both launchers return cudaErrorInvalidValue")) } else { None }
            }
            _ => None,
        };
        match why {
            Some(w) => *unreach.entry(w).or_default() += 1,
            None => {
                reach += 1;
                if launched.contains(i.name) {
                    hit += 1;
                } else {
                    missing.push(i.name.to_string());
                }
            }
        }
    }
    for (w, n) in unreach {
        notes.push(format!("{n} instances not reachable here: {w}"));
    }
    for m in missing.iter().take(20) {
        notes.push(format!("NOT LAUNCHED: {m}"));
    }
    (reach, hit, notes)
}

pub fn run() -> bool {
    let ctx = CudaContext::new(0).expect("cuda context");
    ctx.bind_to_thread().unwrap();
    let streams = vec![ctx.default_stream(), ctx.new_stream().unwrap()];
    let mut g = G { ctx: ctx.clone(), streams, rng: Rng(0x5EED_B0B2_2026), t: Tally::default(), calls: 0, races: 0, skipped: 0 };
    let only = std::env::var("MPB_ONLY").ok();
    let want = |f: &str| only.as_deref().map(|o| o.split(',').any(|x| x == f)).unwrap_or(true);
    let hds: Vec<usize> = std::env::var("MPB_HD")
        .ok()
        .map(|s| s.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or(vec![64, 128, 256, 512]);
    let quick = std::env::var("MPB_QUICK").is_ok();

    let all = only.is_none() && hds.len() == 4;
    let mut fam = |name: &str, g: &mut G, f: &dyn Fn(&mut G)| {
        if want(name) {
            let (c0, f0) = (g.calls, g.t.failures.len());
            f(g);
            println!("  family {name}: {} launcher calls, {} failing", g.calls - c0, g.t.failures.len() - f0);
        }
    };
    fam("mla", &mut g, &mla);
    fam("reshape", &mut g, &reshape);
    fam("gather", &mut g, &gather);
    fam("decode", &mut g, &|g: &mut G| decode(g, &hds, quick));
    fam("edges", &mut g, &decode_edges);
    fam("errors", &mut g, &decode_errors);
    if want("kernels") {
        let rm = load_ref();
        fam("kernels", &mut g, &|g: &mut G| kernels(g, &rm));
    }

    for f in g.t.failures.iter().take(40) {
        println!("  FAIL {f}");
    }
    let (cc, max_smem) = unsafe {
        let dev = 0;
        let mut major = 0;
        let mut ms = 0;
        cuda_core::sys::cuDeviceGetAttribute(&mut major, cuda_core::sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
        cuda_core::sys::cuDeviceGetAttribute(&mut ms, cuda_core::sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev);
        (major, ms as u32)
    };
    let (reach, hit, notes) = coverage(cc, max_smem);
    for n in &notes {
        println!("  {n}");
    }
    println!("  coverage: {hit} of {reach} reachable reference instances launched through their Rust twins ({} total)", INSTANCES.len());
    if g.skipped > 0 {
        println!("  {} kernel-level cases skipped: kernel not in this build (the NUM_STAGES_SMEM=1 instances are gated by a separate build: STAGES1=1 ONLY='ModeE0ELj1E|KernelMLAILj' python3 gen_kernels.py, then MPB_ONLY=kernels; see gate_stages1.log)", g.skipped);
    }
    if g.races > 0 {
        println!("  {} reference self-disagreements (races) observed", g.races);
    }
    let cover_ok = !all || hit == reach;
    let ok = g.t.failures.is_empty() && g.calls > 0 && cover_ok;
    println!(
        "mistralrs-paged-attn-b: {} launcher calls, {} bytes compared, {} failing -> {}",
        g.calls,
        g.t.bytes,
        g.t.failures.len() + if cover_ok { 0 } else { reach - hit },
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}

// ================================================================================================
// Kernel-level gate: every decode / MLA instance (including the NUM_STAGES_SMEM=1 ones no launcher
// reaches on this device) launched from the reference cubin with the exact by-value Params struct and
// from this crate with the same inputs, both with and without partition_kv (lse output).

struct RefMod {
    dec: cuda_core::sys::CUmodule,
    mla: cuda_core::sys::CUmodule,
}

fn load_ref() -> RefMod {
    use cuda_core::sys as cu;
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../reference/mistralrs-paged-attn");
    let load = |f: &str| unsafe {
        let mut img = std::fs::read(format!("{dir}/{f}")).unwrap();
        img.push(0);
        let mut m: cu::CUmodule = std::ptr::null_mut();
        assert_eq!(cu::cuModuleLoadData(&mut m, img.as_ptr() as *const c_void), cu::cudaError_enum_CUDA_SUCCESS, "load {f}");
        m
    };
    RefMod { dec: load("flashinfer_decode.cubin"), mla: load("flashinfer_mla_decode.cubin") }
}

unsafe fn ref_launch(m: cuda_core::sys::CUmodule, name: &str, grid: (u32, u32, u32), block: (u32, u32, u32), smem: u32, params: &mut [u8; 216], stream: *mut c_void) -> i32 {
    use cuda_core::sys as cu;
    unsafe {
        let mut f: cu::CUfunction = std::ptr::null_mut();
        let c = std::ffi::CString::new(name).unwrap();
        assert_eq!(cu::cuModuleGetFunction(&mut f, m, c.as_ptr()), cu::cudaError_enum_CUDA_SUCCESS);
        let r = cu::cuFuncSetAttribute(f, cu::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32);
        if r != cu::cudaError_enum_CUDA_SUCCESS {
            return r as i32;
        }
        let mut p = [params.as_mut_ptr() as *mut c_void];
        let s = if stream.is_null() { 0x2 as cu::CUstream } else { stream as cu::CUstream };
        cu::cuLaunchKernel(f, grid.0, grid.1, grid.2, block.0, block.1, block.2, smem, s, p.as_mut_ptr(), std::ptr::null_mut()) as i32
    }
}

fn put<T: Copy>(b: &mut [u8; 216], off: usize, v: T) {
    let n = std::mem::size_of::<T>();
    let src = unsafe { std::slice::from_raw_parts(&v as *const T as *const u8, n) };
    b[off..off + n].copy_from_slice(src);
}

fn kernels(g: &mut G, rm: &RefMod) {
    for inst in INSTANCES {
        match inst.kind {
            Kind::Decode => {
                if !ox::available(inst.name) {
                    g.skipped += 1;
                    continue;
                }
                for part in [false, true] {
                    kern_decode(g, rm, inst, part);
                }
            }
            Kind::Mla => {
                if !ox::available(inst.name) {
                    g.skipped += 1;
                    continue;
                }
                for part in [false, true] {
                    kern_mla(g, rm, inst, part);
                }
            }
            _ => {}
        }
    }
}

fn kern_decode(g: &mut G, rm: &RefMod, i: &crate::instances::Inst, part: bool) {
    let es = esize(i.dtype);
    let hd = (i.vec * i.bdx) as usize;
    let nt = i.bdx * i.bdy * i.bdz;
    let smem = 2 * i.stages * i.tile * i.bdy * i.bdz * hd as u32 * es as u32 + (i.tile * nt * 8).max(2 * i.bdy * i.bdz * 4);
    let group = i.bdy as usize;
    let nkv = 2usize;
    let nqo = nkv * group;
    let ps = [5usize, 16, 1, 7, 21][g.below(5)];
    let per_iter = (i.tile * i.bdy * i.bdz) as usize;
    let lens = vec![per_iter * 3 + 1, 0, ps + 1, g.below(3 * per_iter) + 1];
    let b = lens.len();
    let pad = g.below(2);
    let pg = make_paged(g, &lens, ps, pad);
    let plan = make_plan(&lens, ps, if part { Some(1 + g.below(2)) } else { None }, 1);
    let padded = plan.req.len();
    let q_rows = nqo + 1;
    let q = g.vals(i.dtype, b * q_rows * hd, 2);
    let kc = g.vals(i.dtype, pg.num_pages * nkv * ps * hd, 2);
    let vc = g.vals(i.dtype, pg.num_pages * nkv * ps * hd, 2);
    let out = g.rng.bytes(padded * nqo * hd * es);
    let lse = g.rng.bytes(padded * nqo * 4);
    let wl = [3i32, 40, -1][g.below(3)];
    let cap = 20.0f32;
    let sm = 1.0 / (hd as f32).sqrt();
    let bufs = vec![q, kc, vc, i32s(&pg.indptr), i32s(&pg.indices), i32s(&pg.last), i32s(&plan.req), i32s(&plan.tiles), i32s(&plan.o_indptr), i32s(&[plan.chunk]), plan.mask.clone(), out, lse];
    let label = format!("kernel {} part{part} ps{ps} lens{lens:?}", &i.name[..i.name.len().min(110)]);
    let (fd, fm, fs, fa) = ox::fastdiv(ps as u32);
    let st = g.stream(1);
    let grid = (padded as u32, nkv as u32, 1);
    let block = (i.bdx, i.bdy, i.bdz);
    let name = i.name;
    let dm = rm.dec;
    let (qsn, qsh) = ((q_rows * hd) as i32, hd as i32);
    g.case(&label, bufs, |rust, s| unsafe {
        let lse_p = if part { s.p(12) } else { std::ptr::null_mut() };
        let stride_page = (nkv * ps * hd) as u32;
        if rust {
            let r = ox::set_max_dynamic_smem(st, name, smem);
            if r != 0 {
                return r;
            }
            let a = ox::Args::new()
                .p(s.p(0)).p(s.p(1)).p(s.p(2)).p(s.p(4)).p(s.p(3)).p(s.p(5)).p(s.p(11)).p(lse_p).p(s.p(6)).p(s.p(7)).p(s.p(9)).p(s.p(10))
                .u(nqo as u32).i(qsn).i(qsh).i(wl).f(cap).f(sm).u(part as u32).u(fd).u(fm).u(fs).u(fa).u(b as u32)
                .u(stride_page).u(hd as u32).u((ps * hd) as u32);
            ox::launch(st, name, grid, block, smem, a)
        } else {
            let mut pb = [0u8; 216];
            put(&mut pb, 0, s.p(0) as u64);
            put(&mut pb, 16, fd);
            put(&mut pb, 20, fm);
            put(&mut pb, 24, fs);
            put(&mut pb, 28, fa);
            put(&mut pb, 32, nkv as u32);
            put(&mut pb, 36, hd as u32);
            put(&mut pb, 40, b as u32);
            put(&mut pb, 44, stride_page);
            put(&mut pb, 48, hd as u32);
            put(&mut pb, 52, (ps * hd) as u32);
            put(&mut pb, 56, s.p(1) as u64);
            put(&mut pb, 64, s.p(2) as u64);
            put(&mut pb, 72, s.p(4) as u64);
            put(&mut pb, 80, s.p(3) as u64);
            put(&mut pb, 88, s.p(5) as u64);
            put(&mut pb, 104, s.p(11) as u64);
            put(&mut pb, 112, lse_p as u64);
            put(&mut pb, 128, padded as u32);
            put(&mut pb, 132, nqo as u32);
            put(&mut pb, 136, qsn);
            put(&mut pb, 140, qsh);
            put(&mut pb, 144, wl);
            put(&mut pb, 148, cap);
            put(&mut pb, 152, sm);
            put(&mut pb, 156, 1.0f32);
            put(&mut pb, 160, 1.0f32);
            put(&mut pb, 168, s.p(6) as u64);
            put(&mut pb, 176, s.p(7) as u64);
            put(&mut pb, 184, s.p(8) as u64);
            put(&mut pb, 192, s.p(9) as u64);
            put(&mut pb, 200, s.p(10) as u64);
            put(&mut pb, 208, part as u8);
            ref_launch(dm, name, grid, block, smem, &mut pb, st)
        }
    });
}

fn kern_mla(g: &mut G, rm: &RefMod, i: &crate::instances::Inst, part: bool) {
    let es = esize(i.dtype);
    let smem = i.stages * 8 * (512 + 64) * es as u32 + 4096;
    let heads = [16usize, 20][g.below(2)];
    let ps = [16usize, 3, 7][g.below(3)];
    let lens = vec![70usize, 0, 9, 17];
    let b = lens.len();
    let pad = g.below(2);
    let pg = make_paged(g, &lens, ps, pad);
    let q_nope = g.vals(i.dtype, b * heads * 512, 2);
    let q_pe = g.vals(i.dtype, b * heads * 64, 2);
    let ckv = g.vals(i.dtype, pg.num_pages * ps * 512, 2);
    let kpe = g.vals(i.dtype, pg.num_pages * ps * 64, 2);
    let out = g.rng.bytes(b * heads * 512 * es);
    let lse = g.rng.bytes(b * heads * 4);
    let req: Vec<i32> = (0..b as i32).collect();
    // chunk 24: request 0 (70 tokens) has tiles 0..=2; the others one tile (a tile past the end of a
    // request makes cur_chunk_len wrap and the reference loop ~2^29 times, as the planner never does)
    let tiles: Vec<i32> = vec![g.below(3) as i32, 0, 0, 0];
    let mask = vec![1u8, 1, 0, 1];
    let bufs = vec![q_nope, q_pe, ckv, kpe, i32s(&pg.indptr), i32s(&pg.indices), i32s(&pg.last), out, lse, i32s(&req), i32s(&tiles), i32s(&[24]), mask];
    let label = format!("kernel {} part{part} heads{heads} ps{ps}", &i.name[..i.name.len().min(110)]);
    let (fd, fm, fs, fa) = ox::fastdiv(ps as u32);
    let st = g.stream(1);
    let grid = (b as u32, (heads as u32).div_ceil(16), 1);
    let name = i.name;
    let mm = rm.mla;
    let sm = 0.07f32;
    g.case(&label, bufs, |rust, s| unsafe {
        let lse_p = if part { s.p(8) } else { std::ptr::null_mut() };
        let mask_p = if part { s.p(12) } else { std::ptr::null_mut() };
        if rust {
            let r = ox::set_max_dynamic_smem(st, name, smem);
            if r != 0 {
                return r;
            }
            let a = ox::Args::new()
                .p(s.p(0)).p(s.p(1)).p(s.p(2)).p(s.p(3)).p(s.p(5)).p(s.p(4)).p(s.p(6)).p(s.p(7)).p(lse_p).p(s.p(9)).p(s.p(10)).p(s.p(11)).p(mask_p)
                .u(heads as u32).f(sm).u(part as u32).u(fd).u(fm).u(fs).u(fa).u(b as u32).u((ps * 512) as u32).u((ps * 64) as u32).u(512).u(64);
            ox::launch(st, name, grid, (32, 8, 1), smem, a)
        } else {
            let mut pb = [0u8; 216];
            put(&mut pb, 0, s.p(0) as u64);
            put(&mut pb, 8, s.p(1) as u64);
            put(&mut pb, 16, s.p(7) as u64);
            put(&mut pb, 24, lse_p as u64);
            put(&mut pb, 32, sm);
            put(&mut pb, 48, fd);
            put(&mut pb, 52, fm);
            put(&mut pb, 56, fs);
            put(&mut pb, 60, fa);
            put(&mut pb, 64, 512u32);
            put(&mut pb, 68, 64u32);
            put(&mut pb, 72, b as u32);
            put(&mut pb, 76, (ps * 512) as u32);
            put(&mut pb, 80, (ps * 64) as u32);
            put(&mut pb, 84, 512u32);
            put(&mut pb, 88, 64u32);
            put(&mut pb, 96, s.p(2) as u64);
            put(&mut pb, 104, s.p(3) as u64);
            put(&mut pb, 112, s.p(5) as u64);
            put(&mut pb, 120, s.p(4) as u64);
            put(&mut pb, 128, s.p(6) as u64);
            put(&mut pb, 144, b as u32);
            put(&mut pb, 148, heads as u32);
            put(&mut pb, 152, -1i32);
            put(&mut pb, 160, 1.0f32);
            put(&mut pb, 164, 1.0f32);
            put(&mut pb, 168, s.p(9) as u64);
            put(&mut pb, 176, s.p(10) as u64);
            put(&mut pb, 192, s.p(11) as u64);
            put(&mut pb, 200, mask_p as u64);
            put(&mut pb, 208, part as u8);
            ref_launch(mm, name, grid, (32, 8, 1), smem, &mut pb, st)
        }
    });
}
