#!/usr/bin/env python3
"""Generate titan's warp-row matvec kernels (`<fmt>_mmvq_w<n>`, `_w1_bf16o`, `_w1_small_k[_bf16o]`) into an
IQ-format crate, plus their gate checks and the `--bench` timing.

The crate supplies (inside its `#[cuda_module]`):
  - `pub const GRID_W: u32` -- words of its lookup table to copy into shared memory once per CUDA block (0: none)
  - `pub fn grid_w(e: u32) -> u32` -- word `e` of that table
  - `pub unsafe fn vdot_w<const MUT: u32>(gt: *const u32, vx: *const u8, yb: *const u8, kbx: i32, iqs: i32,
    acc: f32) -> f32` -- the reference `vdot_acc` with the table read from `gt` (shared) when GRID_W > 0

Re-running replaces the previously generated blocks (between the WARP-ROW markers).

usage: gen_warp_row.py <crate dir> <prefix>
"""
import re
import sys

crate, P = sys.argv[1], sys.argv[2]
main_rs = f"{crate}/src/main.rs"
gate_rs = f"{crate}/src/gate.rs"
s = open(main_rs).read()

BEGIN, END = "    // WARP-ROW BEGIN (gen_warp_row.py)\n", "    // WARP-ROW END\n"
if BEGIN in s:
    s = s[: s.index(BEGIN)] + s[s.index(END) + len(END):]

mm = s[s.index("pub unsafe fn mmvq<"):]
kqm = int(re.search(r"let kqs = (\d+) \* \(tid % TPB\);", mm).group(1))
assert re.search(rf"\(kby \+ kqs / {kqm}\) as u32", mm), "y index is not kby + kqs/KQM"
assert "const TPB: i32 = 8;" in mm
assert re.search(r"pub const BLOCKS_PER_ITER: i32 = (4 \* Self::NW|2 \* Self::NW \* 32 / 16);", s)

proto = re.search(rf"    #\[kernel\] pub unsafe fn {P}_mmvq_1\((.*?)\) \{{ mmvq::<1, false,( 0,)? false>\((.*?)\) \}}\n", s)
params, args = proto.group(1), proto.group(3)
assert params.endswith("_ids_stride: u32")
wparams = params[: -len("_ids_stride: u32")] + "nvb: u32"
wargs = args + ", nvb"

T = r'''
    /// Warps per CUDA block of the warp-row kernels.
    pub const WPB: usize = 4;

    /// Titan's warp-row matvec: one warp computes what one CUDA block of `mmvq` computes (`RPB` rows of one column
    /// group), `WPB` warps per CUDA block, so the block-wide costs (lookup-table copy, barrier) are shared by `WPB`
    /// row groups and the cross-warp reduction happens in registers instead of through shared memory.
    ///
    /// Bit-identical to `mmvq`: lane `l` keeps one accumulator per reference warp `v` and walks exactly the blocks
    /// reference thread `(v, l)` walks, in the same order (`kbx = 4v + l/8 + 4*NW*t`, `kqs = KQM*(l%8)`), then
    /// sums them `((a0 + a1) + a2) + a3` as the reference adds its shared-memory partials, then the same
    /// xor-shuffle tree. A `GRID_W`-word lookup table is read from a shared-memory copy: a warp's 32 scattered
    /// lookups cost a few bank-conflict replays there, against an L1 wavefront per distinct 128-byte line from
    /// global.
    ///
    /// `nvb` (the reference ABI's otherwise unused `ids_stride` slot) = number of row groups, `ceil(nrows/RPB)`;
    /// launch `(ceil(nvb / WPB), channels, samples)` blocks of `(32, WPB)`.
    #[device]
    pub unsafe fn mmvq_w<const NC: usize, const SMALL_K: bool, const MUT: u32, const OB: bool>(
        vx: *const u8, vy: *const u8, ids: *const i32, dst: *mut f32, ncols_x: u32,
        nchannels_y_mp: u32, nchannels_y_l: u32, nchannels_y_d: u32,
        stride_row_x: u32, stride_col_y: u32, stride_col_dst: u32,
        channel_ratio_mp: u32, channel_ratio_l: u32,
        stride_channel_x: u32, stride_channel_y: u32, stride_channel_dst: u32,
        sample_ratio_mp: u32, sample_ratio_l: u32,
        stride_sample_x: u32, stride_sample_y: u32, stride_sample_dst: u32, nvb: u32,
    ) {
        static mut GRID_SHARED: SharedArray<u32, { if GRID_W == 0 { 4 } else { GRID_W as usize } }, 16> = SharedArray::UNINIT;
        const QKB: i32 = 256;
        const TPB: i32 = 8; // qi / vdr
        let nw = K::<NC, SMALL_K>::NW;
        let rpb = K::<NC, SMALL_K>::RPB as u32;
        let tx = thread::threadIdx_x() as i32;
        let ty = thread::threadIdx_y() as i32;
        let sh = SharedArray::as_raw_mut_ptr(&raw mut GRID_SHARED) as *mut u32;
        if GRID_W > 0 {
            let mut e = (32 * ty + tx) as u32;
            while e < GRID_W {
                *sh.add(e as usize) = grid_w(e);
                e += 32 * WPB as u32;
            }
            thread::sync_threads();
        }
        let gt = sh as *const u32;
        let vb = thread::blockIdx_x().wrapping_mul(WPB as u32).wrapping_add(ty as u32);
        if vb >= nvb {
            return;
        }
        let row0 = rpb.wrapping_mul(vb);
        let blocks_per_row_x = (ncols_x / QKB as u32) as i32;

        let channel_dst = thread::blockIdx_y();
        let (channel_x, channel_y) = if NC == 1 && !ids.is_null() {
            let cx = *ids.add(channel_dst as usize) as u32;
            let q = fastdiv(channel_dst, nchannels_y_mp, nchannels_y_l);
            (cx, channel_dst.wrapping_sub(q.wrapping_mul(nchannels_y_d)))
        } else {
            (fastdiv(channel_dst, channel_ratio_mp, channel_ratio_l), channel_dst)
        };
        let sample_dst = thread::blockIdx_z();
        let sample_x = fastdiv(sample_dst, sample_ratio_mp, sample_ratio_l);
        let sample_y = sample_dst;

        // acc[j][i][v]: column j, row i of the group, reference warp v
        let mut acc = [[[0f32; 4]; 4]; NC];
        let y = vy
            .add(sample_y.wrapping_mul(stride_sample_y) as usize * 36)
            .add(channel_y.wrapping_mul(stride_channel_y) as usize * 36);
        let kbx_offset = sample_x
            .wrapping_mul(stride_sample_x)
            .wrapping_add(channel_x.wrapping_mul(stride_channel_x))
            .wrapping_add(row0.wrapping_mul(stride_row_x));

        let kqs = KQM * (tx % TPB);
        let mut kb0 = 0i32;
        while kb0 < blocks_per_row_x {
            let mut v = 0usize;
            #[unroll]
            while v < 4 {
                let kbx = kb0 + 4 * v as i32 + tx / TPB;
                if (v as i32) < nw && kbx < blocks_per_row_x {
                    let kby = 8 * kbx;
                    let mut j = 0;
                    #[unroll]
                    while j < NC {
                        let yi = (j as u32).wrapping_mul(stride_col_y).wrapping_add((kby + kqs / KQM) as u32);
                        let yb = y.add(yi as usize * 36);
                        let mut i = 0;
                        #[unroll]
                        while i < K::<NC, SMALL_K>::RPB {
                            let xk = kbx_offset.wrapping_add((i as u32).wrapping_mul(stride_row_x)).wrapping_add(kbx as u32) as i32;
                            acc[j][i][v] = vdot_w::<MUT>(gt, vx, yb, xk, kqs, acc[j][i][v]);
                            i += 1;
                        }
                        j += 1;
                    }
                }
                v += 1;
            }
            kb0 += K::<NC, SMALL_K>::BLOCKS_PER_ITER;
        }

        let dbase = sample_dst
            .wrapping_mul(stride_sample_dst)
            .wrapping_add(channel_dst.wrapping_mul(stride_channel_dst))
            .wrapping_add(row0) as usize;
        let dst16 = (dst as *mut u16).add(dbase);
        let dst = dst.add(dbase);
        let mut j = 0;
        #[unroll]
        while j < NC {
            let mut i = 0;
            #[unroll]
            while i < K::<NC, SMALL_K>::RPB {
                let mut t = acc[j][i][0];
                let mut l = 1;
                #[unroll]
                while l < 4 {
                    if (l as i32) < nw {
                        t = add(t, acc[j][i][l]);
                    }
                    l += 1;
                }
                let mut k = 0;
                #[unroll]
                while k < 5 {
                    t = add(t, warp::shuffle_xor_f32_sync(0xffff_ffff, t, 16 >> k));
                    k += 1;
                }
                if tx == i as i32 && (rpb == 1 || row0.wrapping_add(i as u32) < stride_col_dst) {
                    let o = (j as u32).wrapping_mul(stride_col_dst).wrapping_add(i as u32);
                    if OB {
                        // bf16 out: rounded as candle's cast_f32_bf16 rounds
                        *dst16.add(o as usize) = f2bf(t);
                    } else {
                        *dst.add(o as usize) = t;
                    }
                }
                i += 1;
            }
            j += 1;
        }
    }
'''.replace("KQM", str(kqm))

ents = []
def ent(name, nc, sk, ob):
    ents.append(f"    #[kernel] pub unsafe fn {name}({wparams}) {{ mmvq_w::<{nc}, {sk}, 0, {ob}>({wargs}) }}\n")
ent(f"{P}_mmvq_w1", 1, "false", "false")
ent(f"{P}_mmvq_w1_bf16o", 1, "false", "true")
ent(f"{P}_mmvq_w1_small_k", 1, "true", "false")
ent(f"{P}_mmvq_w1_small_k_bf16o", 1, "true", "true")
for n in range(2, 9):
    ent(f"{P}_mmvq_w{n}", n, "false", "false")

block = BEGIN + T + "".join(ents) + END
anchor = s.index("    // GENERATED KERNELS BEGIN")
anchor = s.rindex("\n", 0, s.rindex("\n", 0, anchor)) + 1  # before the comment line preceding the marker
s = s[:anchor] + block + s[anchor:]
open(main_rs, "w").write(s)

# ---- gate: warp-row + bf16 checks in every non-mutant mmvq case; the bench times ref / oxide / warp-row
g = open(gate_rs).read()
GB, GE = "    // WARP-ROW GATE BEGIN (gen_warp_row.py)\n", "    // WARP-ROW GATE END\n"
if GB in g:
    i0, i1 = g.index(GB), g.index(GE) + len(GE)
    # undo: the generated block plus the `let bad = ` / trailing `bad` wrapping around the original compare
    pre = g[:i0]
    k = pre.rindex("    let bad = ")
    stmt = pre[k + len("    let bad = "):]
    post = g[i1:]
    if post.startswith("    bad\n"):
        g = pre[:k] + "    " + stmt.rstrip("\n").rstrip(";") + "\n" + post[len("    bad\n"):]
    else:
        g = pre[:k] + "    " + stmt + post
# the mmvq case function: the one whose body launches `m.oxide` with the mmvq name and ends in a compare
fm = re.search(r"\nfn (mmvq_case_named)\(m: &Mods, t: &mut Tally, r: &mut Rng, c: &MmvqCase, rate: u64(, ox_name: &str)?\)( -> usize)? \{\n", g)
if fm is None:
    fm = re.search(r"\nfn (mmvq_case)\(m: &Mods, t: &mut Tally, r: &mut Rng, c: &MmvqCase, rate: u64(, ox_name: &str)?\)( -> usize)? \{\n", g)
fs = fm.end()
fe = g.index("\n}\n", fs) + 1
body = g[fs:fe]
oxname = "ox_name" if fm.group(2) else "&mmvq_ox_name(c.nc, c.small_k)"
cm = re.search(r"    (compare\(t, [^;\n]*\))(;?)\n$", body)
assert cm, "final compare not found"
five = re.search(r"\nfn compare\(t: &mut Tally, label: &str, a: &\[Vec<u8>\], b: &\[Vec<u8>\], f32_buf", g) is not None
extra = ", None" if five else ""
wgate = GB + '''    if !OXNAME.contains("mut") {
        // titan's warp-row kernels (one warp per reference CUDA block, 4 per block): same bits as the reference
        let nvb = c.nrows.div_ceil(rpb);
        let mut pw = Params(p.0.clone());
        let n = pw.0.len();
        pw.0[n - 4..].copy_from_slice(&nvb.to_le_bytes());
        let wgrid = (nvb.div_ceil(4), c.nch_dst, c.nsamp_dst);
        let wname = OXNAME.replacen("PFX_mmvq_", "PFX_mmvq_w", 1);
        let w = launch(func(m.oxide, &wname), wgrid, (32, 4, 1), &pw, &slots, &bufs, &[2]);
        let _ = compare(t, &format!("{wname}: {label}"), &a, &wEXTRA);
        if c.nc == 1 {
            // bf16 out: the reference's f32 sums rounded to nearest even (NaN stays NaN), the rest untouched
            let mut fb = bufs.clone();
            fb[2] = vec![0xFF; bufs[2].len()];
            let rf = launch(func(m.mmvq, &mmvq_ref_name(c.nc, c.small_k)), grid, block, &p, &slots, &fb, &[2]);
            let ob = launch(func(m.oxide, &format!("{wname}_bf16o")), wgrid, (32, 4, 1), &pw, &slots, &fb, &[2]);
            let (rf, ob) = (&rf[0], &ob[0]);
            let mut errs = 0usize;
            for k in 0..rf.len() / 4 {
                let f = u32::from_le_bytes(rf[4 * k..4 * k + 4].try_into().unwrap());
                let h = u16::from_le_bytes(ob[2 * k..2 * k + 2].try_into().unwrap());
                let ok = if f == 0xFFFF_FFFF {
                    h == 0xFFFF
                } else if f32::from_bits(f).is_nan() {
                    (h & 0x7F80) == 0x7F80 && (h & 0x7F) != 0
                } else {
                    h as u32 == (f.wrapping_add(0x7FFF + ((f >> 16) & 1))) >> 16
                };
                if !ok { errs += 1; }
            }
            for k in rf.len() / 4..ob.len() / 2 {
                if u16::from_le_bytes(ob[2 * k..2 * k + 2].try_into().unwrap()) != 0xFFFF { errs += 1; }
            }
            t.record(&format!("{wname}_bf16o: {label}"), &kdiff::Diff { bytes: ob.len(), differing: errs, first: if errs > 0 { Some((0, 0, 0, 0)) } else { None } });
        }
    }
''' + GE
wgate = wgate.replace("PFX", P).replace("OXNAME", oxname).replace("EXTRA", extra)
if cm.group(2) == ";":
    nb = body[: cm.start()] + "    let bad = " + cm.group(1) + ";\n" + wgate + "    let _ = bad;\n"
else:
    nb = body[: cm.start()] + "    let bad = " + cm.group(1) + ";\n" + wgate + "    bad\n"
g = g[:fs] + nb + g[fe:]

b0 = g.index("fn bench() {")
b0 = g.rindex("\n", 0, g.rindex("\n", 0, b0)) + 1  # include the doc comment line
b1 = g.index("\n}\n", g.index("fn bench() {")) + 3
bench = r'''/// `--bench`: reference vs oxide vs warp-row mmvq (nc = 1) on Qwen decode shapes. ROT=1 cycles through weight
/// copies totalling >= 512 MiB (each launch streams from DRAM, as in decode); ITERS (default 200) launches each;
/// CARVE sets the preferred shared-memory carveout; ONLY=k times one of them.
fn bench() {
    let dir = env!("CARGO_MANIFEST_DIR");
    let _ctx = kdiff::cuda_core::CudaContext::new(0).expect("cuda context");
    let m = Mods { mmvq: load(&format!("{dir}/{REF_MMVQ}")), convert: load(&format!("{dir}/{REF_CONVERT}")), oxide: load(&format!("{dir}/{OXIDE}")) };
    let _ = m.convert;
    let mut r = Rng(7);
    let iters: usize = std::env::var("ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
    let rot = std::env::var("ROT").is_ok_and(|v| v == "1");
    for &(bpr, nrows) in &[(20u32, 17408u32), (68, 5120), (20, 5120), (20, 10240), (24, 5120)] {
        let x = x_blocks(&mut r, (bpr * (nrows + 1)) as usize, 512);
        let y = q8_1_blocks(&mut r, (8 * bpr + 8) as usize, 512);
        unsafe {
            let ncopy = if rot { (512usize << 20).div_ceil(x.len()).max(2) } else { 1 };
            let mut copies = vec![];
            for _ in 0..ncopy {
                let mut d: cu::CUdeviceptr = 0;
                ck!(cu::cuMemAlloc_v2(&mut d, x.len()));
                ck!(cu::cuMemcpyHtoD_v2(d, x.as_ptr() as *const c_void, x.len()));
                copies.push(d);
            }
            let (mut dy, mut dd): (cu::CUdeviceptr, cu::CUdeviceptr) = (0, 0);
            ck!(cu::cuMemAlloc_v2(&mut dy, y.len()));
            ck!(cu::cuMemAlloc_v2(&mut dd, nrows as usize * 4));
            ck!(cu::cuMemcpyHtoD_v2(dy, y.as_ptr() as *const c_void, y.len()));
            let one = fastdiv_values(1);
            let mut p = Params::default();
            p.put(copies[0]).put(dy).put(0u64);
            for _ in 0..5 { p.put(0u64); }
            p.put(0u32).put(0f32).put(dd).put(bpr * QK).put(0u32).put(0u32).put(0u32);
            p.put(bpr).put(8 * bpr).put(nrows).put(one[0]).put(one[1]).put(one[2]).put(0u32).put(0u32).put(0u32);
            p.put(one[0]).put(one[1]).put(one[2]).put(0u32).put(0u32).put(0u32).put(0u32);
            // the kernels are timed interleaved, in rounds of 100 launches each, so drift in clocks (the decode power
            // cap settles over ~100 ms) hits all of them alike; timing one after another favoured whichever ran first
            let mut ms = [0f32; 3];
            let fs = [func(m.mmvq, &mmvq_ref_name(1, false)), func(m.oxide, "PFX_mmvq_1"), func(m.oxide, "PFX_mmvq_w1")];
            let shm: u32 = std::env::var("SHM").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
            let only = std::env::var("ONLY").ok().and_then(|v| v.parse::<usize>().ok());
            let mut pbs = vec![];
            for (k, &f) in fs.iter().enumerate() {
                if let Some(c) = std::env::var("CARVE").ok().and_then(|v| v.parse::<i32>().ok()) {
                    ck!(cu::cuFuncSetAttribute(f, cu::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, c));
                }
                let mut pb = p.0.clone();
                if k == 2 {
                    let n = pb.len();
                    pb[n - 4..].copy_from_slice(&nrows.to_le_bytes());
                }
                pbs.push(pb);
            }
            let (mut e0, mut e1): (cu::CUevent, cu::CUevent) = (std::ptr::null_mut(), std::ptr::null_mut());
            ck!(cu::cuEventCreate(&mut e0, 0));
            ck!(cu::cuEventCreate(&mut e1, 0));
            let mut it = 0usize;
            for round in 0..(iters / 100).max(1) + 1 {
                for (k, &f) in fs.iter().enumerate() {
                    if only.is_some_and(|o| o != k) { continue; }
                    let pb = &mut pbs[k];
                    let mut size = pb.len();
                    let mut extra: [*mut c_void; 5] = [1 as *mut c_void, pb.as_mut_ptr() as *mut c_void, 2 as *mut c_void, &mut size as *mut usize as *mut c_void, std::ptr::null_mut()];
                    let gx = if k == 2 { nrows.div_ceil(4) } else { nrows };
                    ck!(cu::cuEventRecord(e0, std::ptr::null_mut()));
                    for _ in 0..100 {
                        (*pb)[0..8].copy_from_slice(&copies[it % copies.len()].to_le_bytes());
                        it += 1;
                        ck!(cu::cuLaunchKernel(f, gx, 1, 1, 32, 4, 1, if k > 0 { shm } else { 0 }, std::ptr::null_mut(), std::ptr::null_mut(), extra.as_mut_ptr()));
                    }
                    ck!(cu::cuEventRecord(e1, std::ptr::null_mut()));
                    ck!(cu::cuEventSynchronize(e1));
                    let mut t = 0f32;
                    ck!(cu::cuEventElapsedTime_v2(&mut t, e0, e1));
                    if round > 0 { ms[k] += t; } // round 0 warms up
                }
            }
            for v in ms.iter_mut() { *v *= 1e3 / (100 * (iters / 100).max(1)) as f32; }
            let gb = (x.len() as f64 * nrows as f64 / (nrows + 1) as f64) / 1e9;
            let gbs = |us: f32| gb / (us as f64 / 1e6);
            println!("mmvq nc=1 K={} N={nrows}: ref {:.2} us ({:.0} GB/s), oxide {:.2} ({:.0}), warp-row {:.2} ({:.0})", bpr * QK,
                ms[0], gbs(ms[0]), ms[1], gbs(ms[1]), ms[2], gbs(ms[2]));
            for d in copies { cu::cuMemFree_v2(d); }
            cu::cuMemFree_v2(dy);
            cu::cuMemFree_v2(dd);
        }
    }
}
'''.replace("PFX", P)
g = g[:b0] + bench + g[b1:]
open(gate_rs, "w").write(g)
print(f"{P}: KQM={kqm}, {len(ents)} warp-row entries")
