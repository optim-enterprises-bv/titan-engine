//! Differential gates and microbench for mmvq_moe.
//!
//! (A) `*_moe_mmvq_llama` vs llama.cpp acecd56 `mul_mat_vec_q_moe<type, 2, false>`
//!     (../iq4_nl/ref/mmvq.cubin, SASS identical to libggml-cuda.so), b = 1..8, many shapes.
//! (B) `*_moe_b1[_sk][_llama]` vs llama.cpp `mul_mat_vec_q<type, 1, false, small_k>` (MUL_MAT_ID,
//!     one launch per token as ggml does).
//! (C) the service kernels `*_moe_mmvq` (b = 1..8) and `*_moe_b1[_sk]` vs the deployed expert
//!     kernels (titan_kernels_oxide.ptx `*_q8_1_moe_gemv[_w]`, what titan_cpu.rs twins), with
//!     non-resident experts (slot map NOT_RESIDENT) and a sentinel-filled output: every byte.
//! (D) sensitivity: the llama MoE program must DIFFER from the deployed one where llama.cpp's MoE
//!     kernel reduces in another order (Q6_K k=512, IQ4_NL k=2048), so the gates see reduction order.
//! Oxide kernels read a permuted slot buffer through the slot map in every gate.
//! `--bench`: kernel times (CUDA events, L2 flushed by a 256 MiB memset before every launch) on the
//! 35B expert shapes, 256 experts, top-8 distinct random experts per token.
use cuda_core::sys as cu;
use std::ffi::c_void;

macro_rules! ck {
    ($e:expr) => {{
        let r = $e;
        assert_eq!(r, cu::cudaError_enum_CUDA_SUCCESS, "{}", stringify!($e));
    }};
}

const NOT_RESIDENT: u32 = u32::MAX;

pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Fmt {
    Q4K,
    Q5K,
    Q6K,
    Q1_0,
    IQ4NL,
    MXFP4,
    NVFP4,
}
pub const ALL: [Fmt; 7] = [Fmt::Q4K, Fmt::Q5K, Fmt::Q6K, Fmt::Q1_0, Fmt::IQ4NL, Fmt::MXFP4, Fmt::NVFP4];

impl Fmt {
    pub fn name(self) -> &'static str {
        match self {
            Fmt::Q4K => "q4k",
            Fmt::Q5K => "q5k",
            Fmt::Q6K => "q6k",
            Fmt::Q1_0 => "q1_0",
            Fmt::IQ4NL => "iq4_nl",
            Fmt::MXFP4 => "mxfp4",
            Fmt::NVFP4 => "nvfp4",
        }
    }
    pub fn qk(self) -> usize {
        match self {
            Fmt::Q4K | Fmt::Q5K | Fmt::Q6K => 256,
            Fmt::Q1_0 => 128,
            Fmt::NVFP4 => 64,
            _ => 32,
        }
    }
    pub fn bb(self) -> usize {
        match self {
            Fmt::Q4K => 144,
            Fmt::Q5K => 176,
            Fmt::Q6K => 210,
            Fmt::MXFP4 => 17,
            Fmt::NVFP4 => 36,
            _ => 18,
        }
    }
    pub fn ggml(self) -> u32 {
        match self {
            Fmt::Q4K => 12,
            Fmt::Q5K => 13,
            Fmt::Q6K => 14,
            Fmt::Q1_0 => 41,
            Fmt::IQ4NL => 20,
            Fmt::MXFP4 => 39,
            Fmt::NVFP4 => 40,
        }
    }
    /// qi / vdr
    pub fn tpb(self) -> usize {
        match self {
            Fmt::Q4K | Fmt::Q5K => 16,
            Fmt::Q6K => 32,
            Fmt::Q1_0 => 4,
            _ => 2,
        }
    }
    pub fn kquant(self) -> bool {
        matches!(self, Fmt::Q4K | Fmt::Q5K | Fmt::Q6K)
    }
    /// llama.cpp `should_use_small_k` for ncols_dst 1 on NVIDIA Turing+ (4 warps).
    pub fn small_k(self, k: usize) -> bool {
        k / self.qk() < 4 * (32 / self.tpb())
    }
    pub fn rpb_b1(self, k: usize) -> usize {
        if self.small_k(k) { 4 } else { 1 }
    }
    /// The deployed expert GEMV (titan_tiered.rs `gemv_kernel`) and whether it is the warp-rows variant.
    pub fn old_kernel(self, k: usize) -> (String, bool) {
        let warp_rows = k <= 512 && matches!(self, Fmt::Q4K | Fmt::Q5K);
        let base = match self {
            Fmt::Q1_0 => "q1_0",
            Fmt::IQ4NL => "iq4_nl",
            f => f.name(),
        };
        (format!("{base}_q8_1_moe_gemv{}", if warp_rows { "_w" } else { "" }), warp_rows)
    }
    /// Byte offsets of the block's f16 scales (NVFP4 / MXFP4 handled separately).
    fn half_offsets(self) -> &'static [usize] {
        match self {
            Fmt::Q4K | Fmt::Q5K => &[0, 2],
            Fmt::Q6K => &[208],
            Fmt::Q1_0 | Fmt::IQ4NL => &[0],
            _ => &[],
        }
    }
}

fn f16_normal(r: &mut Rng, emin: i32, emax: i32) -> u16 {
    let e = emin + r.below((emax - emin + 1) as usize) as i32;
    let sign = ((r.next() & 1) as u16) << 15;
    sign | (((e + 15) as u16) << 10) | (r.next() as u16 & 0x3FF)
}

/// Mostly random finite f16s of any exponent, plus subnormals, signed zeros, infinities and NaNs.
fn f16_adversarial(r: &mut Rng) -> u16 {
    let p = r.next() % 1000;
    let sign = ((r.next() & 1) as u16) << 15;
    let man = (r.next() % 0x3FF) as u16 + 1;
    sign | match p {
        0..=2 => 0x7C00 | man,
        3..=5 => 0x7C00,
        6..=199 => man,
        200..=249 => 0,
        _ => (((r.next() % 30) as u16 + 1) << 10) | (man - 1),
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Mode {
    Normal,
    Adversarial,
}

pub fn gen_weights(fmt: Fmt, blocks: usize, mode: Mode, r: &mut Rng) -> Vec<u8> {
    let bb = fmt.bb();
    let mut b: Vec<u8> = (0..blocks * bb).map(|_| r.next() as u8).collect();
    for blk in b.chunks_exact_mut(bb) {
        match fmt {
            Fmt::MXFP4 => {
                blk[0] = match mode {
                    Mode::Normal => 115 + r.below(16) as u8,
                    Mode::Adversarial => match r.next() % 100 {
                        0 => 255,
                        1..=40 => (r.next() % 12) as u8,
                        41..=70 => 120 + (r.next() % 16) as u8,
                        _ => (r.next() % 255) as u8,
                    },
                }
            }
            Fmt::NVFP4 => {
                if mode == Mode::Normal {
                    for s in &mut blk[0..4] {
                        *s = 0x28 + r.below(0x20) as u8;
                    }
                }
            }
            _ => {
                for &o in fmt.half_offsets() {
                    let h = match mode {
                        Mode::Normal => f16_normal(r, -12, -3),
                        Mode::Adversarial => f16_adversarial(r),
                    };
                    blk[o..o + 2].copy_from_slice(&h.to_le_bytes());
                }
            }
        }
    }
    b
}

/// Q8_1 blocks: random int8 quants (-128 included), ds = half2(d, s).
pub fn gen_q8(blocks: usize, mode: Mode, r: &mut Rng) -> Vec<u8> {
    let mut v: Vec<u8> = (0..blocks * 36).map(|_| r.next() as u8).collect();
    for blk in v.chunks_exact_mut(36) {
        let d = match mode {
            Mode::Normal => f16_normal(r, -10, -4),
            Mode::Adversarial => f16_adversarial(r),
        };
        let s = f16_normal(r, -4, 4);
        blk[0..2].copy_from_slice(&d.to_le_bytes());
        blk[2..4].copy_from_slice(&s.to_le_bytes());
    }
    v
}

pub struct Buf {
    pub p: u64,
    pub len: usize,
}
impl Buf {
    pub fn up(b: &[u8]) -> Buf {
        let mut p: cu::CUdeviceptr = 0;
        unsafe {
            ck!(cu::cuMemAlloc_v2(&mut p, b.len().max(4)));
            if !b.is_empty() {
                ck!(cu::cuMemcpyHtoD_v2(p, b.as_ptr() as *const c_void, b.len()));
            }
        }
        Buf { p: p as u64, len: b.len() }
    }
    pub fn filled(len: usize, byte: u8) -> Buf {
        let mut p: cu::CUdeviceptr = 0;
        unsafe {
            ck!(cu::cuMemAlloc_v2(&mut p, len.max(4)));
            ck!(cu::cuMemsetD8_v2(p, byte, len));
        }
        Buf { p: p as u64, len }
    }
    pub fn down(&self) -> Vec<u8> {
        let mut h = vec![0u8; self.len];
        unsafe {
            ck!(cu::cuCtxSynchronize());
            ck!(cu::cuMemcpyDtoH_v2(h.as_mut_ptr() as *mut c_void, self.p as cu::CUdeviceptr, self.len));
        }
        h
    }
}
impl Drop for Buf {
    fn drop(&mut self) {
        unsafe {
            cu::cuMemFree_v2(self.p as cu::CUdeviceptr);
        }
    }
}

pub fn load(path: &str) -> cu::CUmodule {
    let mut img = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    img.push(0);
    let mut m: cu::CUmodule = std::ptr::null_mut();
    unsafe { ck!(cu::cuModuleLoadData(&mut m, img.as_ptr() as *const c_void)) };
    m
}

pub fn func(m: cu::CUmodule, name: &str) -> cu::CUfunction {
    let mut f: cu::CUfunction = std::ptr::null_mut();
    let c = std::ffi::CString::new(name).unwrap();
    let r = unsafe { cu::cuModuleGetFunction(&mut f, m, c.as_ptr()) };
    assert_eq!(r, cu::cudaError_enum_CUDA_SUCCESS, "missing function {name}");
    f
}

unsafe fn launch(f: cu::CUfunction, grid: (u32, u32, u32), block: (u32, u32, u32), args: &mut [*mut c_void]) {
    ck!(cu::cuLaunchKernel(f, grid.0, grid.1, grid.2, block.0, block.1, block.2, 0, std::ptr::null_mut(), args.as_mut_ptr(), std::ptr::null_mut()));
}

unsafe fn launch_packed(f: cu::CUfunction, grid: (u32, u32, u32), block: (u32, u32, u32), params: &mut [u8]) {
    let mut size = params.len();
    let mut extra: [*mut c_void; 5] =
        [1 as *mut c_void, params.as_mut_ptr() as *mut c_void, 2 as *mut c_void, &mut size as *mut usize as *mut c_void, std::ptr::null_mut()];
    ck!(cu::cuLaunchKernel(f, grid.0, grid.1, grid.2, block.0, block.1, block.2, 0, std::ptr::null_mut(), std::ptr::null_mut(), extra.as_mut_ptr()));
}

#[derive(Default)]
struct Params(Vec<u8>);
impl Params {
    fn put<T: Copy>(&mut self, v: T) -> &mut Self {
        let a = std::mem::align_of::<T>();
        while self.0.len() % a != 0 {
            self.0.push(0);
        }
        self.0.extend_from_slice(unsafe { std::slice::from_raw_parts(&v as *const T as *const u8, std::mem::size_of::<T>()) });
        self
    }
}

/// llama.cpp `init_fastdiv_values`: <mp, L, d>.
fn fastdiv_values(d: u32) -> [u32; 3] {
    let mut l = 0u32;
    while l < 32 && (1u64 << l) < d as u64 {
        l += 1;
    }
    [(((1u64 << 32) * ((1u64 << l) - d as u64)) / d as u64 + 1) as u32, l, d]
}

/// The titan ABI of every mmvq_moe kernel.
#[derive(Clone, Copy)]
pub struct Call {
    pub w: u64,
    pub x: u64,
    pub ids: u64,
    pub map: u64,
    pub out: u64,
    pub n: u32,
    pub k: u32,
    pub kp: u32,
    pub topk: u32,
    pub dim1: u32,
}

pub unsafe fn launch_titan(f: cu::CUfunction, grid: (u32, u32, u32), block: (u32, u32, u32), c: &Call) {
    let mut p = [c.w, c.x, c.ids, c.map, c.out];
    let mut u = [c.n, c.k, c.kp, c.topk, c.dim1];
    let pp = p.as_mut_ptr();
    let up = u.as_mut_ptr();
    let mut args: [*mut c_void; 10] = [
        pp as _, pp.add(1) as _, pp.add(2) as _, pp.add(3) as _, pp.add(4) as _,
        up as _, up.add(1) as _, up.add(2) as _, up.add(3) as _, up.add(4) as _,
    ];
    launch(f, grid, block, &mut args);
}

/// The deployed kernels' ABI: slices as (pointer, element count) pairs, 1-D grid of 128-thread blocks.
pub unsafe fn launch_old(f: cu::CUfunction, warp_rows: bool, c: &Call, w_words: u64, x_words: u64, tasks: usize, map_len: u64) {
    let mut p = [c.w, w_words, c.x, x_words, c.ids, tasks as u64, c.map, map_len, c.out];
    let mut u = [c.n, c.k, c.kp, c.topk, c.dim1];
    let pp = p.as_mut_ptr();
    let up = u.as_mut_ptr();
    let mut args: [*mut c_void; 14] = [
        pp as _, pp.add(1) as _, pp.add(2) as _, pp.add(3) as _, pp.add(4) as _, pp.add(5) as _, pp.add(6) as _, pp.add(7) as _, pp.add(8) as _,
        up as _, up.add(1) as _, up.add(2) as _, up.add(3) as _, up.add(4) as _,
    ];
    let blocks = if warp_rows { (c.n as usize).div_ceil(4) * tasks } else { c.n as usize * tasks };
    launch(f, (blocks as u32, 1, 1), (128, 1, 1), &mut args);
}

fn moe_ref_name(t: u32) -> String {
    format!("_Z17mul_mat_vec_q_moeIL9ggml_type{t}ELi2ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjjjjjjj")
}
fn b1_ref_name(t: u32, small_k: bool) -> String {
    format!(
        "_Z13mul_mat_vec_qIL9ggml_type{t}ELi1ELb0ELb{}ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj",
        small_k as u32
    )
}

pub struct Mods {
    pub ox: cu::CUmodule,
    pub llama: cu::CUmodule,
    pub old: cu::CUmodule,
}

#[derive(Default)]
pub struct Tally {
    rows: std::collections::BTreeMap<String, [usize; 3]>,
    pub ok: bool,
}
impl Tally {
    fn rec(&mut self, label: &str, words: usize, bad: usize) {
        let e = self.rows.entry(label.to_string()).or_default();
        e[0] += 1;
        e[1] += words;
        e[2] += bad;
    }
}

fn diff_words(a: &[u8], b: &[u8]) -> usize {
    a.chunks_exact(4).zip(b.chunks_exact(4)).filter(|(x, y)| x != y).count()
}

/// Weights for a case: `experts` experts of [n][k / qk] blocks, from a tensor file or generated.
fn weights(fmt: Fmt, k: usize, n: usize, experts: usize, file: Option<&str>, mode: Mode, r: &mut Rng) -> Vec<u8> {
    let eb = n * (k / fmt.qk()) * fmt.bb();
    match file {
        Some(f) => {
            use std::io::Read;
            let mut v = Vec::new();
            std::fs::File::open(f).unwrap_or_else(|e| panic!("{f}: {e} (run extract_q35.py)")).take((experts * eb) as u64).read_to_end(&mut v).unwrap();
            assert_eq!(v.len(), experts * eb, "{f}");
            v
        }
        None => gen_weights(fmt, experts * n * (k / fmt.qk()), mode, r),
    }
}

/// Slot buffer: resident expert e at slot perm[e] (NOT_RESIDENT experts are absent), padded.
fn slotted(bytes: &[u8], eb: usize, map: &[u32]) -> Vec<u8> {
    let slots = map.iter().filter(|&&s| s != NOT_RESIDENT).count();
    let mut v = vec![0u8; slots * eb + 4 * eb.max(64)];
    for (e, &s) in map.iter().enumerate() {
        if s != NOT_RESIDENT {
            v[s as usize * eb..(s as usize + 1) * eb].copy_from_slice(&bytes[e * eb..(e + 1) * eb]);
        }
    }
    v
}

fn slot_map(experts: usize, resident_pct: usize, r: &mut Rng) -> Vec<u32> {
    let mut res: Vec<usize> = (0..experts).filter(|_| r.below(100) < resident_pct).collect();
    if res.is_empty() {
        res.push(0);
    }
    for i in (1..res.len()).rev() {
        res.swap(i, r.below(i + 1));
    }
    let mut map = vec![NOT_RESIDENT; experts];
    for (s, &e) in res.iter().enumerate() {
        map[e] = s as u32;
    }
    map
}

struct Shape {
    fmt: Fmt,
    k: usize,
    n: usize,
    experts: usize,
    file: Option<String>,
    mode: Mode,
}

fn shapes() -> Vec<Shape> {
    let data = concat!(env!("CARGO_MANIFEST_DIR"), "/data");
    let mut v = Vec::new();
    let mut add = |fmt, k, n, experts, file: Option<String>, mode| v.push(Shape { fmt, k, n, experts, file, mode });
    // the 35B service tensors (first 32 experts of layer 0 / 1)
    add(Fmt::Q4K, 2048, 512, 32, Some(format!("{data}/q35.blk0.ffn_gate_exps.q4k.k2048n512")), Mode::Normal);
    add(Fmt::Q5K, 512, 2048, 32, Some(format!("{data}/q35.blk0.ffn_down_exps.q5k.k512n2048")), Mode::Normal);
    add(Fmt::Q6K, 512, 2048, 32, Some(format!("{data}/q35.blk1.ffn_down_exps.q6k.k512n2048")), Mode::Normal);
    for fmt in ALL {
        let ks: &[(usize, usize)] = match fmt {
            // 2048 / 512: the 35B shapes (Q5_K gate/up is layer 1's); 768: a partial second
            // iteration; 4096 x 33: several iterations, odd rows; 6144: Q6_K lanes chain 6+ blocks.
            Fmt::Q4K | Fmt::Q5K | Fmt::Q6K => &[(2048, 512), (512, 2048), (768, 64), (4096, 33), (256, 7), (6144, 16)],
            Fmt::Q1_0 => &[(2048, 512), (512, 64), (384, 33), (12288, 16), (4096, 17)],
            Fmt::NVFP4 => &[(2048, 512), (512, 64), (8256, 33), (12288, 16), (192, 9)],
            _ => &[(2048, 512), (512, 64), (4128, 33), (12288, 16), (96, 9)],
        };
        for (i, &(k, n)) in ks.iter().enumerate() {
            add(fmt, k, n, 12, None, if i % 2 == 0 { Mode::Normal } else { Mode::Adversarial });
        }
        add(fmt, ks[0].0, 64, 12, None, Mode::Adversarial);
    }
    v
}

/// (topk, input_dim1): gate/up (one input row per token) and down (one per task) at top-8, plus
/// top-3, top-1 and top-10.
const CONFIGS: [(usize, u32); 5] = [(8, 1), (8, 0), (3, 1), (1, 0), (10, 1)];

struct Case {
    ids: Vec<u32>,
    xq: Vec<u8>,
    tasks: usize,
}

fn case(sh: &Shape, batch: usize, topk: usize, dim1: u32, r: &mut Rng) -> Case {
    let kp = sh.k.div_ceil(512) * 512;
    let tasks = batch * topk;
    let in_rows = if dim1 == 1 { batch } else { tasks };
    let ids: Vec<u32> = (0..tasks).map(|_| r.below(sh.experts) as u32).collect();
    let xq = gen_q8(in_rows * kp / 32, sh.mode, r);
    Case { ids, xq, tasks }
}

fn as_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// (A) + (D): the llama-program MoE kernel vs llama.cpp's.
fn gate_moe_llama(m: &Mods, t: &mut Tally, sh: &Shape, r: &mut Rng) {
    let fmt = sh.fmt;
    let (k, n) = (sh.k, sh.n);
    let kp = k.div_ceil(512) * 512;
    let bpr = k / fmt.qk();
    let eb = n * bpr * fmt.bb();
    let bytes = weights(fmt, k, n, sh.experts, sh.file.as_deref(), sh.mode, r);
    let mut contig = bytes.clone();
    contig.extend(std::iter::repeat_n(0u8, 4 * eb.max(64)));
    let map = slot_map(sh.experts, 100, r);
    let w_ref = Buf::up(&contig);
    let w_ox = Buf::up(&slotted(&bytes, eb, &map));
    let map_d = Buf::up(&as_bytes(&map));
    let f_ref = func(m.llama, &moe_ref_name(fmt.ggml()));
    let f_ox = func(m.ox, &format!("{}_moe_mmvq_llama", fmt.name()));
    let label = format!("A {:?} llama mul_mat_vec_q_moe", fmt);
    for batch in 1..=8usize {
        for &(topk, d1) in &CONFIGS {
            let dim1 = if d1 == 1 { 1 } else { topk as u32 };
            let c = case(sh, batch, topk, dim1, r);
            let ids = Buf::up(&as_bytes(&c.ids));
            let x = Buf::up(&c.xq);
            let out_ref = Buf::filled(c.tasks * n * 4, 0xA5);
            let out_ox = Buf::filled(c.tasks * n * 4, 0xA5);
            let q8 = (kp / 32) as u32;
            unsafe {
                let ncy = fastdiv_values(if dim1 == 1 { 1 } else { topk as u32 });
                let mut p = Params::default();
                p.put(w_ref.p).put(x.p).put(ids.p);
                for _ in 0..5 {
                    p.put(0u64);
                }
                p.put(0u32).put(0f32).put(out_ref.p);
                let (scy, sch) = if dim1 == 1 { (q8, 0) } else { (topk as u32 * q8, q8) };
                for v in [k as u32, ncy[0], ncy[1], ncy[2], n as u32, bpr as u32, scy, (topk * n) as u32, (n * bpr) as u32, sch, n as u32, batch as u32, topk as u32] {
                    p.put(v);
                }
                assert_eq!(p.0.len(), 132);
                launch_packed(f_ref, (n.div_ceil(2) as u32, topk as u32, 1), (32, batch as u32, 1), &mut p.0);
                let call = Call { w: w_ox.p, x: x.p, ids: ids.p, map: map_d.p, out: out_ox.p, n: n as u32, k: k as u32, kp: kp as u32, topk: topk as u32, dim1: if dim1 == 1 { 1 } else { 0 } };
                launch_titan(f_ox, (n.div_ceil(2) as u32, topk as u32, 1), (32, batch as u32, 1), &call);
            }
            let (a, b) = (out_ref.down(), out_ox.down());
            let bad = diff_words(&a, &b);
            let untouched = a.chunks_exact(4).filter(|w| *w == [0xA5; 4]).count();
            t.rec(&label, a.len() / 4, bad);
            if bad > 0 || untouched > 0 {
                println!("  MISMATCH {label} k={k} n={n} b={batch} topk={topk} dim1={dim1} mode={:?}: {bad} words differ, {untouched} untouched", sh.mode);
                t.ok = false;
            }
        }
    }
}

/// (B): the b1 grid kernels vs llama.cpp's `mul_mat_vec_q<type, 1>` launched per token.
fn gate_b1_llama(m: &Mods, t: &mut Tally, sh: &Shape, r: &mut Rng) {
    let fmt = sh.fmt;
    let (k, n) = (sh.k, sh.n);
    let kp = k.div_ceil(512) * 512;
    let bpr = k / fmt.qk();
    let eb = n * bpr * fmt.bb();
    let sk = fmt.small_k(k);
    let rpb = fmt.rpb_b1(k);
    let bytes = weights(fmt, k, n, sh.experts, sh.file.as_deref(), sh.mode, r);
    let mut contig = bytes.clone();
    contig.extend(std::iter::repeat_n(0u8, 4 * eb.max(64)));
    let map = slot_map(sh.experts, 100, r);
    let w_ref = Buf::up(&contig);
    let w_ox = Buf::up(&slotted(&bytes, eb, &map));
    let map_d = Buf::up(&as_bytes(&map));
    let f_ref = func(m.llama, &b1_ref_name(fmt.ggml(), sk));
    let ox_name = format!("{}_moe_b1{}{}", fmt.name(), if sk { "_sk" } else { "" }, if fmt.kquant() { "_llama" } else { "" });
    let f_ox = func(m.ox, &ox_name);
    let label = format!("B {:?} llama mul_mat_vec_q<1> ({ox_name})", fmt);
    for &(batch, topk, d1) in &[(1usize, 8usize, 1u32), (1, 8, 0), (3, 8, 1), (3, 8, 0), (2, 10, 1), (4, 1, 0)] {
        let dim1 = if d1 == 1 { 1 } else { topk as u32 };
        let c = case(sh, batch, topk, dim1, r);
        let ids = Buf::up(&as_bytes(&c.ids));
        let x = Buf::up(&c.xq);
        let out_ref = Buf::filled(c.tasks * n * 4, 0x5A);
        let out_ox = Buf::filled(c.tasks * n * 4, 0x5A);
        let q8 = kp / 32;
        unsafe {
            let one = fastdiv_values(1);
            let ncy = fastdiv_values(if dim1 == 1 { 1 } else { topk as u32 });
            for b in 0..batch {
                let mut p = Params::default();
                let y_row = if dim1 == 1 { b } else { b * topk };
                p.put(w_ref.p).put(x.p + (y_row * q8 * 36) as u64).put(ids.p + (b * topk * 4) as u64);
                for _ in 0..5 {
                    p.put(0u64);
                }
                p.put(0u32).put(0f32).put(out_ref.p + (b * topk * n * 4) as u64);
                for v in [k as u32, ncy[0], ncy[1], ncy[2], bpr as u32, q8 as u32, n as u32, 0, 0, 0, (n * bpr) as u32, q8 as u32, n as u32, one[0], one[1], one[2], 0, 0, 0, 0] {
                    p.put(v);
                }
                assert_eq!(p.0.len(), 160);
                launch_packed(f_ref, (n.div_ceil(rpb) as u32, topk as u32, 1), (32, 4, 1), &mut p.0);
            }
            let call = Call { w: w_ox.p, x: x.p, ids: ids.p, map: map_d.p, out: out_ox.p, n: n as u32, k: k as u32, kp: kp as u32, topk: topk as u32, dim1: if dim1 == 1 { 1 } else { 0 } };
            launch_titan(f_ox, (n.div_ceil(rpb) as u32, c.tasks as u32, 1), (32, 4, 1), &call);
        }
        let (a, b) = (out_ref.down(), out_ox.down());
        let bad = diff_words(&a, &b);
        let untouched = a.chunks_exact(4).filter(|w| *w == [0x5A; 4]).count();
        t.rec(&label, a.len() / 4, bad);
        if bad > 0 || untouched > 0 {
            println!("  MISMATCH {label} k={k} n={n} b={batch} topk={topk} dim1={dim1} mode={:?}: {bad} words differ, {untouched} untouched", sh.mode);
            t.ok = false;
        }
    }
}

/// (C) + (D): the service kernels vs the deployed ones, 70% of experts resident.
fn gate_service(m: &Mods, t: &mut Tally, sens: &mut std::collections::BTreeMap<String, [usize; 2]>, sh: &Shape, r: &mut Rng) {
    let fmt = sh.fmt;
    let (k, n) = (sh.k, sh.n);
    let kp = k.div_ceil(512) * 512;
    let bpr = k / fmt.qk();
    let eb = n * bpr * fmt.bb();
    let rpb = fmt.rpb_b1(k);
    let bytes = weights(fmt, k, n, sh.experts, sh.file.as_deref(), sh.mode, r);
    let map = slot_map(sh.experts, 70, r);
    let sb = slotted(&bytes, eb, &map);
    let w_words = (sb.len() / 4) as u64;
    let w = Buf::up(&sb);
    let map_d = Buf::up(&as_bytes(&map));
    let (old_name, warp_rows) = fmt.old_kernel(k);
    let f_old = func(m.old, &old_name);
    let f_moe = func(m.ox, &format!("{}_moe_mmvq", fmt.name()));
    let b1_name = format!("{}_moe_b1{}", fmt.name(), if fmt.small_k(k) { "_sk" } else { "" });
    let f_b1 = func(m.ox, &b1_name);
    let f_ll = func(m.ox, &format!("{}_moe_mmvq_llama", fmt.name()));
    for batch in 1..=8usize {
        for &(topk, d1) in &CONFIGS {
            let dim1 = if d1 == 1 { 1 } else { topk as u32 };
            let c = case(sh, batch, topk, dim1, r);
            let ids = Buf::up(&as_bytes(&c.ids));
            let x = Buf::up(&c.xq);
            let sentinel = 0x3C + (batch as u8);
            let outs: Vec<Buf> = (0..4).map(|_| Buf::filled(c.tasks * n * 4, sentinel)).collect();
            let call = |o: &Buf| Call { w: w.p, x: x.p, ids: ids.p, map: map_d.p, out: o.p, n: n as u32, k: k as u32, kp: kp as u32, topk: topk as u32, dim1: if dim1 == 1 { 1 } else { 0 } };
            unsafe {
                launch_old(f_old, warp_rows, &call(&outs[0]), w_words, (c.xq.len() / 4) as u64, c.tasks, map.len() as u64);
                launch_titan(f_moe, (n.div_ceil(2) as u32, topk as u32, 1), (32, batch as u32, 1), &call(&outs[1]));
                launch_titan(f_b1, (n.div_ceil(rpb) as u32, c.tasks as u32, 1), (32, 4, 1), &call(&outs[2]));
                launch_titan(f_ll, (n.div_ceil(2) as u32, topk as u32, 1), (32, batch as u32, 1), &call(&outs[3]));
            }
            let a = outs[0].down();
            let resident = c.ids.iter().filter(|&&e| map[e as usize] != NOT_RESIDENT).count();
            let written = a.chunks_exact(4).filter(|w| *w != [sentinel; 4]).count();
            // non-resident tasks keep the sentinel, resident ones are written (sanity of the reference)
            if written != resident * n {
                println!("  NOTE old kernel wrote {written} words, {} resident outputs (NaN/sentinel collisions are possible in adversarial data)", resident * n);
            }
            for (j, name) in [(1usize, format!("{}_moe_mmvq", fmt.name())), (2, b1_name.clone())] {
                let b = outs[j].down();
                let bad = diff_words(&a, &b);
                let label = format!("C {:?} {name} vs deployed {old_name}", fmt);
                t.rec(&label, a.len() / 4, bad);
                if bad > 0 {
                    println!("  MISMATCH {label} k={k} n={n} b={batch} topk={topk} dim1={dim1} mode={:?}: {bad} words differ", sh.mode);
                    t.ok = false;
                }
            }
            let b = outs[3].down();
            let e = sens.entry(format!("{:?} k={k} {:?}", fmt, sh.mode)).or_default();
            e[0] += resident * n;
            e[1] += diff_words(&a, &b);
        }
    }
}

pub fn setup() -> Mods {
    let dir = env!("CARGO_MANIFEST_DIR");
    let _ctx = cuda_core::CudaContext::new(0).expect("cuda context");
    std::mem::forget(_ctx);
    unsafe {
        let mut c: cu::CUcontext = std::ptr::null_mut();
        ck!(cu::cuCtxGetCurrent(&mut c));
        if c.is_null() {
            let mut dev = 0;
            ck!(cu::cuDeviceGet(&mut dev, 0));
            ck!(cu::cuDevicePrimaryCtxRetain(&mut c, dev));
            ck!(cu::cuCtxSetCurrent(c));
        }
    }
    let home = std::env::var("HOME").unwrap();
    let llama = std::env::var("LLAMA_CUBIN").unwrap_or_else(|_| format!("{dir}/../iq4_nl/ref/mmvq.cubin"));
    let old = std::env::var("OLD_PTX").unwrap_or_else(|_| format!("{home}/titan-engine/mr-mmvqmoe/mistralrs-quant/src/gguf/titan_kernels_oxide.ptx"));
    let ox = std::env::var("OXIDE_PTX").unwrap_or_else(|_| format!("{dir}/mmvq_moe.ptx"));
    println!("oxide {ox}\nllama.cpp {llama}\ndeployed {old}");
    Mods { ox: load(&ox), llama: load(&llama), old: load(&old) }
}

pub fn run() -> bool {
    let m = setup();
    if std::env::args().any(|a| a == "--bench") {
        crate::gate::bench(&m);
        return true;
    }
    let only = std::env::var("FMT").ok();
    let mut t = Tally { ok: true, ..Default::default() };
    let mut sens: std::collections::BTreeMap<String, [usize; 2]> = Default::default();
    let mut r = Rng(0x9E37_79B9_7F4A_7C15);
    let t0 = std::time::Instant::now();
    for sh in shapes() {
        if only.as_deref().is_some_and(|o| o != sh.fmt.name()) {
            continue;
        }
        println!("{:?} k={} n={} experts={} {} {:?}", sh.fmt, sh.k, sh.n, sh.experts, sh.file.as_deref().map(|f| f.rsplit('/').next().unwrap()).unwrap_or("synthetic"), sh.mode);
        gate_moe_llama(&m, &mut t, &sh, &mut r);
        gate_b1_llama(&m, &mut t, &sh, &mut r);
        gate_service(&m, &mut t, &mut sens, &sh, &mut r);
    }
    println!("\n== gate summary ({:.1}s)", t0.elapsed().as_secs_f64());
    let mut launches = 0;
    let mut words = 0;
    for (label, [c, w, bad]) in &t.rows {
        println!("{label}: {c} comparisons, {w} output words, {bad} differ");
        launches += c;
        words += w;
    }
    println!("{launches} comparisons, {words} output words ({} MB) compared", words * 4 / 1_000_000);
    // (D) the gates must see reduction order: llama.cpp's MoE program differs from the deployed one
    // where each lane chains fused `tmp += d * sumf` over several quant blocks.
    println!("\n== sensitivity: llama.cpp MoE program vs deployed b1 program (resident outputs that differ)");
    let mut sens_ok = true;
    for (key, [tot, bad]) in &sens {
        println!("{key}: {bad} / {tot}");
    }
    for must in ["Q6K k=512 Normal", "IQ4NL k=2048 Normal"] {
        let bad = sens.get(must).map(|v| v[1]).unwrap_or(0);
        if only.is_none() && bad == 0 {
            println!("SENSITIVITY FAIL: {must} shows no difference");
            sens_ok = false;
        }
    }
    let ok = t.ok && sens_ok;
    println!("{}", if ok { "MMVQ-MOE GATE: PASS" } else { "MMVQ-MOE GATE: FAIL" });
    ok
}

/// Median of the event-timed kernel (µs), L2 flushed before every launch.
unsafe fn time_us(flush: &Buf, reps: usize, mut f: impl FnMut()) -> f64 {
    let mut e0: cu::CUevent = std::ptr::null_mut();
    let mut e1: cu::CUevent = std::ptr::null_mut();
    ck!(cu::cuEventCreate(&mut e0, 0));
    ck!(cu::cuEventCreate(&mut e1, 0));
    let mut v = Vec::new();
    for i in 0..reps + 3 {
        ck!(cu::cuMemsetD8Async(flush.p as cu::CUdeviceptr, i as u8, flush.len, std::ptr::null_mut()));
        ck!(cu::cuEventRecord(e0, std::ptr::null_mut()));
        f();
        ck!(cu::cuEventRecord(e1, std::ptr::null_mut()));
        ck!(cu::cuEventSynchronize(e1));
        let mut ms = 0f32;
        ck!(cu::cu_event_elapsed_time(&mut ms, e0, e1));
        if i >= 3 {
            v.push(ms as f64 * 1000.0);
        }
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

pub fn bench(m: &Mods) {
    let flush = Buf::filled(256 << 20, 0);
    let reps: usize = std::env::var("REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(50);
    let mut r = Rng(0xBE7C_4000_0000_0001);
    println!("| op | b | deployed µs | new µs | new/deployed | alt µs (b1: MoE grid; b>1: b1 grid) | llama.cpp µs | new/llama | oxide llama-program MoE µs |");
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|");
    for &(fmt, k, n, dim1, tag) in &[
        (Fmt::Q4K, 2048usize, 512usize, 1u32, "q35.exp_gate_up.q4_K.k2048n512"),
        (Fmt::Q5K, 512, 2048, 0, "q35.exp_down.q5_K.k512n2048"),
        (Fmt::Q6K, 512, 2048, 0, "q35.exp_down.q6_K.k512n2048"),
    ] {
        let experts = 256;
        let topk = 8;
        let kp = k.div_ceil(512) * 512;
        let bpr = k / fmt.qk();
        let eb = n * bpr * fmt.bb();
        let mut bytes = gen_weights(fmt, experts * n * bpr, Mode::Normal, &mut r);
        bytes.extend(std::iter::repeat_n(0u8, 4 * eb));
        let w = Buf::up(&bytes);
        let map: Vec<u32> = (0..experts as u32).collect();
        let map_d = Buf::up(&as_bytes(&map));
        let (old_name, warp_rows) = fmt.old_kernel(k);
        let f_old = func(m.old, &old_name);
        let f_moe = func(m.ox, &format!("{}_moe_mmvq", fmt.name()));
        let sk = fmt.small_k(k);
        let rpb = fmt.rpb_b1(k);
        let f_b1 = func(m.ox, &format!("{}_moe_b1{}", fmt.name(), if sk { "_sk" } else { "" }));
        let f_ll = func(m.ox, &format!("{}_moe_mmvq_llama", fmt.name()));
        let f_ref_moe = func(m.llama, &moe_ref_name(fmt.ggml()));
        let f_ref_b1 = func(m.llama, &b1_ref_name(fmt.ggml(), sk));
        for &batch in &[1usize, 2, 3, 4, 8] {
            let tasks = batch * topk;
            // top-8 distinct experts per token
            let mut ids = Vec::with_capacity(tasks);
            for _ in 0..batch {
                let mut chosen: Vec<u32> = Vec::new();
                while chosen.len() < topk {
                    let e = r.below(experts) as u32;
                    if !chosen.contains(&e) {
                        chosen.push(e);
                    }
                }
                ids.extend(chosen);
            }
            let in_rows = if dim1 == 1 { batch } else { tasks };
            let xq = gen_q8(in_rows * kp / 32, Mode::Normal, &mut r);
            let ids_d = Buf::up(&as_bytes(&ids));
            let x = Buf::up(&xq);
            let out = Buf::filled(tasks * n * 4, 0);
            let call = Call { w: w.p, x: x.p, ids: ids_d.p, map: map_d.p, out: out.p, n: n as u32, k: k as u32, kp: kp as u32, topk: topk as u32, dim1 };
            let w_words = (bytes.len() / 4) as u64;
            unsafe {
                let t_old = time_us(&flush, reps, || launch_old(f_old, warp_rows, &call, w_words, (xq.len() / 4) as u64, tasks, experts as u64));
                let t_moe = time_us(&flush, reps, || launch_titan(f_moe, (n.div_ceil(2) as u32, topk as u32, 1), (32, batch as u32, 1), &call));
                let t_b1 = time_us(&flush, reps, || launch_titan(f_b1, (n.div_ceil(rpb) as u32, tasks as u32, 1), (32, 4, 1), &call));
                let t_ll = time_us(&flush, reps, || launch_titan(f_ll, (n.div_ceil(2) as u32, topk as u32, 1), (32, batch as u32, 1), &call));
                let q8 = (kp / 32) as u32;
                let t_ref = if batch == 1 {
                    let one = fastdiv_values(1);
                    let ncy = fastdiv_values(if dim1 == 1 { 1 } else { topk as u32 });
                    let mut p = Params::default();
                    p.put(w.p).put(x.p).put(ids_d.p);
                    for _ in 0..5 {
                        p.put(0u64);
                    }
                    p.put(0u32).put(0f32).put(out.p);
                    for v in [k as u32, ncy[0], ncy[1], ncy[2], bpr as u32, q8, n as u32, 0, 0, 0, (n * bpr) as u32, q8, n as u32, one[0], one[1], one[2], 0, 0, 0, 0] {
                        p.put(v);
                    }
                    time_us(&flush, reps, || launch_packed(f_ref_b1, (n.div_ceil(rpb) as u32, topk as u32, 1), (32, 4, 1), &mut p.0))
                } else {
                    let ncy = fastdiv_values(if dim1 == 1 { 1 } else { topk as u32 });
                    let (scy, sch) = if dim1 == 1 { (q8, 0) } else { (topk as u32 * q8, q8) };
                    let mut p = Params::default();
                    p.put(w.p).put(x.p).put(ids_d.p);
                    for _ in 0..5 {
                        p.put(0u64);
                    }
                    p.put(0u32).put(0f32).put(out.p);
                    for v in [k as u32, ncy[0], ncy[1], ncy[2], n as u32, bpr as u32, scy, (topk * n) as u32, (n * bpr) as u32, sch, n as u32, batch as u32, topk as u32] {
                        p.put(v);
                    }
                    time_us(&flush, reps, || launch_packed(f_ref_moe, (n.div_ceil(2) as u32, topk as u32, 1), (32, batch as u32, 1), &mut p.0))
                };
                let (t_new, t_alt) = if batch == 1 { (t_b1, t_moe) } else { (t_moe, t_b1) };
                println!("| {tag} | {batch} | {t_old:.1} | {t_new:.1} | {:.2} | {t_alt:.1} | {t_ref:.1} | {:.2} | {t_ll:.1} |", t_new / t_old, t_new / t_ref);
            }
        }
    }
}
