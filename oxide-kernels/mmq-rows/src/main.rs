#![allow(unsafe_op_in_unsafe_fn)]
//! MMQ-exact GGUF matmul for a few activation columns (titan-engine speculative verification).
//!
//! `mmq_rows_<q>_nc<n>`: every output is bit-identical to the stream-k `mul_mat_q` x8 tile of
//! ../mistralrs-quant-c (sm_120 MMA path, the one mistral.rs runs for 1..=8 columns) and its
//! fixup. Per output, MMQ is a serial f32 chain over the 32-value sub-blocks of each stream-k
//! segment (the int8 tensor-core dots are exact integers, so only their f32 consumption matters),
//! the tile's last segment stored to dst and the earlier ones summed last-to-first and added to it.
//! This kernel runs one thread per (weight row, segment) with all columns in registers, so a
//! 2-column verification reads the weights once at matvec speed instead of filling an 8-wide MMA
//! tile, and needs no second launch.
//!
//! grid (ceil(nrows / 32), 1, 1), block (32, S, 1): lane = weight row within a 32-row group (all in
//! one 128-row MMQ tile), warp y = segment y of that tile, S = the most segments any tile has.
#![allow(non_snake_case, clippy::missing_safety_doc, dead_code)]
use cuda_device::{DynamicSharedArray, convert, dotprod, f16x2, kernel, ptx_asm, thread, warp};
use cuda_host::cuda_module;

#[cuda_module]
pub mod kernels {
    use super::*;

    const T_F32: i32 = 0;
    const T_F16: i32 = 1;
    const T_BF16: i32 = 30;
    /// MMQ's mmq_y: weight rows per tile.
    const MMQ_Y: i32 = 128;
    const WARP: i32 = 32;
    /// block_q8_1_mmq: 4 (d, s) half2 or 4 f32 d, then 128 int8.
    const YB: usize = 144;
    /// 16-byte chunks per Q4_K / Q5_K block.
    const CHUNKS_Q4: usize = 9;
    const CHUNKS_Q5: usize = 11;
    /// Staged words per row (odd, so a warp reading one word of each of its 32 rows hits 32 banks) and k blocks
    /// staged per round; kept in step with mistralrs-quant's mmq_rows.rs.
    const RS_Q4: usize = 37;
    const RS_Q5: usize = 45;
    const KB_PER_STAGE: usize = 2;

    macro_rules! unroll {
        () => {
            cuda_device::thread::__unroll_config::<0>();
        };
    }

    // fast-math f32 (every add / mul / fma is .ftz), as mistralrs-quant-c

    #[inline(always)]
    pub fn h2lo(w: u32) -> f32 {
        convert::cvt_f32_f16x2_lo(w)
    }
    #[inline(always)]
    pub fn h2hi(w: u32) -> f32 {
        convert::cvt_f32_f16x2_lo(w >> 16)
    }
    #[inline(always)]
    pub fn bf2f(bits: u32) -> f32 {
        f32::from_bits(bits << 16)
    }
    // trailing escaped-whitespace comment: see PORTING.md rule 38
    #[inline(always)]
    pub fn fma(a: f32, b: f32, c: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("fma.rn.ftz.f32 %0, %1, %2, %3; //\t\x0b\x0c\r\n", out("=f") r, in("f") a, in("f") b, in("f") c, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn mul(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("mul.rn.ftz.f32 %0, %1, %2; //\t\x0b\x0c\r\n", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn add(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("add.rn.ftz.f32 %0, %1, %2; //\t\x0b\x0c\r\n", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn dp4a(a: u32, b: u32, c: i32) -> i32 {
        dotprod::dp4a_s32(a, b, c)
    }
    #[inline(always)]
    pub fn f2h(x: f32) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("cvt.rn.f16.f32 %0, %1;", out("=h") r, in("f") x, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn f2bf(x: f32) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("cvt.rn.bf16.f32 %0, %1;", out("=h") r, in("f") x, options(register_only)); }
        r
    }
    // Weights and activations are read-only for the launch: non-coherent loads.
    #[inline(always)]
    pub unsafe fn ld32(p: *const u8) -> u32 {
        let r: u32;
        ptx_asm!("ld.global.nc.u32 %0, [%1];", out("=r") r, in("l") p as u64, options(register_only));
        r
    }
    #[inline(always)]
    pub unsafe fn ld128(p: *const u8) -> [u32; 4] {
        let (a, b, c, d): (u32, u32, u32, u32);
        ptx_asm!("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];", out("=r") a, out("=r") b, out("=r") c, out("=r") d,
                 in("l") p as u64, options(register_only));
        [a, b, c, d]
    }
    /// 8 words from a 16-byte aligned address.
    #[inline(always)]
    pub unsafe fn ld8w(p: *const u8) -> [u32; 8] {
        let a = ld128(p);
        let b = ld128(p.add(16));
        [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
    }
    #[inline(always)]
    unsafe fn lds(addr: u32) -> u32 {
        let r: u32;
        ptx_asm!("ld.shared.u32 %0, [%1];", out("=r") r, in("r") addr);
        r
    }
    #[inline(always)]
    unsafe fn sts(addr: u32, v: u32) {
        ptx_asm!("st.shared.u32 [%0], %1;", in("r") addr, in("r") v, clobber("memory"));
    }
    /// Word `i` of a staged row at shared address `ra`.
    #[inline(always)]
    unsafe fn xw(ra: u32, i: usize) -> u32 {
        lds(ra + 4 * i as u32)
    }

    #[inline(always)]
    pub const fn row_words(fmt: u32) -> usize {
        if fmt == 7 { RS_Q4 } else { RS_Q5 }
    }

    /// Copy k blocks kb .. kb + n (n <= KB_PER_STAGE) of the warp's 32 rows into consecutive staging buffers from
    /// `sa`, every load issued before the first store. Rows past `nrows` stage row `row0`; their results are dropped.
    #[inline(always)]
    unsafe fn stage<const FMT: u32>(sa: u32, x: *const u8, row0: i32, nrows: i32, row_bytes: i64, kb: i32, n: i32, lane: i32) {
        let bb: i64 = if FMT == 7 { 144 } else { 176 };
        let cpr: i32 = if FMT == 7 { CHUNKS_Q4 as i32 } else { CHUNKS_Q5 as i32 };
        let rs = row_words(FMT) as u32;
        let mut v = [[0u32; 4]; KB_PER_STAGE * CHUNKS_Q5];
        let mut q = 0;
        while q < KB_PER_STAGE {
            unroll!();
            let mut it = 0;
            while it < CHUNKS_Q5 {
                unroll!();
                if (FMT == 8 || it < CHUNKS_Q4) & ((q as i32) < n) {
                    let c = it as i32 * WARP + lane;
                    let (r, o) = (c / cpr, c % cpr);
                    let row = if row0 + r < nrows { row0 + r } else { row0 };
                    let off = (row as i64).wrapping_mul(row_bytes) + (kb + q as i32) as i64 * bb + 16 * o as i64;
                    v[q * CHUNKS_Q5 + it] = ld128(x.offset(off as isize));
                }
                it += 1;
            }
            q += 1;
        }
        let mut q = 0;
        while q < KB_PER_STAGE {
            unroll!();
            let mut it = 0;
            while it < CHUNKS_Q5 {
                unroll!();
                if (FMT == 8 || it < CHUNKS_Q4) & ((q as i32) < n) {
                    let c = it as i32 * WARP + lane;
                    let (r, o) = (c / cpr, c % cpr);
                    let w = sa + 4 * ((q as u32 * WARP as u32 + r as u32) * rs + 4 * o as u32);
                    let t = v[q * CHUNKS_Q5 + it];
                    sts(w, t[0]);
                    sts(w + 4, t[1]);
                    sts(w + 8, t[2]);
                    sts(w + 12, t[3]);
                }
                it += 1;
            }
            q += 1;
        }
    }

    #[inline(always)]
    unsafe fn store_dst(dst: *mut u8, idx: i32, v: f32, type_dst: i32) {
        match type_dst {
            T_F32 => *(dst as *mut f32).offset(idx as isize) = v,
            T_F16 => *(dst as *mut u16).offset(idx as isize) = f2h(v),
            T_BF16 => *(dst as *mut u16).offset(idx as isize) = f2bf(v),
            _ => {}
        }
    }
    #[inline(always)]
    unsafe fn load_dst(dst: *const u8, idx: i32, type_dst: i32) -> f32 {
        match type_dst {
            T_F32 => *(dst as *const f32).offset(idx as isize),
            T_F16 => convert::cvt_f32_f16x2_lo(*(dst as *const u16).offset(idx as isize) as u32),
            T_BF16 => bf2f(*(dst as *const u16).offset(idx as isize) as u32),
            _ => 0.0,
        }
    }

    /// Stream-k block b's first k block (`b * total / grid`; one k block per MMQ iteration for qk 256).
    #[inline(always)]
    fn kstart(b: i32, total: i32, grid: i32) -> i32 {
        ((b as i64).wrapping_mul(total as i64) / grid as i64) as i32
    }

    /// Segment `s` of `tile`: the k blocks (within the row) of the s-th stream-k block that touches it, or (0, 0).
    #[inline(always)]
    fn segment(tile: i32, s: i32, bpn: i32, total: i32, grid: i32) -> (i32, i32) {
        let a = tile.wrapping_mul(bpn);
        let e = a + bpn;
        let mut b0 = ((a as i64).wrapping_mul(grid as i64) / total as i64) as i32;
        while b0 + 1 < grid && kstart(b0 + 1, total, grid) <= a {
            b0 += 1;
        }
        while b0 > 0 && kstart(b0, total, grid) > a {
            b0 -= 1;
        }
        let b = b0 + s;
        if b >= grid {
            return (0, 0);
        }
        let lo = kstart(b, total, grid);
        let hi = kstart(b + 1, total, grid);
        let lo = if lo > a { lo } else { a };
        let hi = if hi < e { hi } else { e };
        if lo >= hi { (0, 0) } else { (lo - a, hi - a) }
    }

    /// unpack_scales_q45_K.
    #[inline(always)]
    fn unpack_scales_q45_k(s: [u32; 3], ksc: i32) -> u32 {
        let pick = |k: i32| if k == 0 { s[0] } else if k == 1 { s[1] } else { s[2] };
        let a = ((pick((ksc % 2) + (ksc != 0) as i32) as i32) >> (4 * (ksc & (ksc / 2)))) as u32;
        let b = ((pick(ksc / 2) as i32) >> (2 * (ksc % 2))) as u32;
        (a & 0x0F0F_0F0F) | (b & 0x3030_3030)
    }

    /// x_dm of load_tiles_q4_K / q5_K for the 8 sub-blocks: (d * sc, -dmin * m) as f16x2.
    #[inline(always)]
    unsafe fn q45_dm(ra: u32) -> [u32; 8] {
        let h = [xw(ra, 0), xw(ra, 1), xw(ra, 2), xw(ra, 3)];
        let s = [h[1], h[2], h[3]];
        let dm = f16x2::mul_f16x2(h[0], 0xBC00_3C00);
        let mut da = [0u32; 8];
        let mut ksc = 0;
        while ksc < 2 {
            unroll!();
            let sc32 = unpack_scales_q45_k(s, ksc as i32);
            let m32 = unpack_scales_q45_k(s, ksc as i32 + 2);
            let mut l = 0;
            while l < 4 {
                unroll!();
                let sc = ((sc32 >> (8 * l)) & 0xFF) as f32;
                let m = ((m32 >> (8 * l)) & 0xFF) as f32;
                da[4 * ksc + l] = f16x2::mul_f16x2(dm, convert::cvt_f16x2_f32(sc, m));
                l += 1;
            }
            ksc += 1;
        }
        da
    }

    /// vec_dot_q8_1_q8_1_mma's update for sub-block `sb` (32 values in `v`) of one column.
    #[inline(always)]
    unsafe fn ds4_step(acc: f32, da: u32, v: &[u32; 8], yc: *const u8, sb: usize) -> f32 {
        let q = sb % 4;
        let ds = ld32(yc.add(4 * q));
        let yw = ld8w(yc.add(16 + 32 * q));
        let mut c = 0i32;
        let mut k = 0;
        while k < 8 {
            unroll!();
            c = dp4a(v[k], yw[k], c);
            k += 1;
        }
        let s = fma(mul(h2lo(da), h2lo(ds)), c as f32, acc);
        fma(h2hi(da), h2hi(ds), s)
    }

    /// One k block of Q4_K (FMT 7) or Q5_K (FMT 8) for every column.
    #[inline(always)]
    unsafe fn kblock_q45<const FMT: u32, const NC: usize>(acc: &mut [f32; NC], ra: u32, y: *const u8, ncols_y: i32, kb: i32) {
        let da = q45_dm(ra);
        let mut qh = [0u32; 8];
        if FMT == 8 {
            let mut k = 0;
            while k < 8 {
                unroll!();
                qh[k] = xw(ra, 4 + k);
                k += 1;
            }
        }
        let qs0 = if FMT == 8 { 12 } else { 4 };
        let mut c = 0;
        while c < 4 {
            unroll!();
            let mut w = [0u32; 8];
            let mut k = 0;
            while k < 8 {
                unroll!();
                w[k] = xw(ra, qs0 + 8 * c + k);
                k += 1;
            }
            let mut t = 0;
            while t < 2 {
                unroll!();
                let sb = 2 * c + t;
                let mut v = [0u32; 8];
                let mut k = 0;
                while k < 8 {
                    unroll!();
                    v[k] = (w[k] >> (4 * t)) & 0x0F0F_0F0F;
                    if FMT == 8 {
                        v[k] |= ((qh[k] >> sb) << 4) & 0x1010_1010;
                    }
                    k += 1;
                }
                let chunk = 2 * kb + (sb / 4) as i32;
                let mut j = 0;
                while j < NC {
                    unroll!();
                    let yc = y.offset((ncols_y.wrapping_mul(chunk).wrapping_add(j as i32) as isize) * YB as isize);
                    acc[j] = ds4_step(acc[j], da[sb], &v, yc, sb);
                    j += 1;
                }
                t += 1;
            }
            c += 1;
        }
    }

    /// FMT: 7 Q4_K, 8 Q5_K. `bpn` k blocks per row, `stride_row_x` blocks between rows, `total` =
    /// ceil(nrows / 128) * bpn and `grid` the stream-k block count of the MMQ launch this reproduces.
    #[inline(always)]
    pub unsafe fn mmq_rows<const FMT: u32, const NC: usize>(
        x: *const u8, y: *const u8, dst: *mut u8, nrows: i32, bpn: i32, stride_row_x: i32, ncols_y: i32, total: i32,
        grid: i32, type_dst: i32,
    ) {
        let lane = thread::threadIdx_x() as i32;
        let s = thread::threadIdx_y() as i32;
        let nseg = thread::blockDim_y() as i32;
        let row0 = (thread::blockIdx_x() as i32) * WARP;
        let tile = row0 / MMQ_Y;
        let row = row0 + lane;
        let bb: i32 = if FMT == 7 { 144 } else { 176 };
        let (lo, hi) = segment(tile, s, bpn, total, grid);
        let mut acc = [0f32; NC];
        // dynamic shared: [segment][column][lane] partials, then each warp's staged rows
        let part: *mut f32 = DynamicSharedArray::<f32>::get();
        let row_bytes = (stride_row_x as i64).wrapping_mul(bb as i64);
        let sa = cuda_device::shared::cvta_generic_to_shared_u32(part as *const u8)
            + 4 * (nseg as u32 * NC as u32 * WARP as u32 + (s * WARP) as u32 * (KB_PER_STAGE * row_words(FMT)) as u32);
        let ra = sa + 4 * lane as u32 * row_words(FMT) as u32;
        let buf = 4 * WARP as u32 * row_words(FMT) as u32;
        let mut kb = lo;
        while kb < hi {
            let n = if hi - kb < KB_PER_STAGE as i32 { hi - kb } else { KB_PER_STAGE as i32 };
            warp::sync_mask(0xFFFF_FFFF);
            stage::<FMT>(sa, x, row0, nrows, row_bytes, kb, n, lane);
            warp::sync_mask(0xFFFF_FFFF);
            kblock_q45::<FMT, NC>(&mut acc, ra, y, ncols_y, kb);
            if n > 1 {
                kblock_q45::<FMT, NC>(&mut acc, ra + buf, y, ncols_y, kb + 1);
            }
            kb += n;
        }
        let mut j = 0;
        while j < NC {
            unroll!();
            *part.add(((s as usize) * NC + j) * WARP as usize + lane as usize) = acc[j];
            j += 1;
        }
        thread::sync_threads();
        if s != 0 || row >= nrows {
            return;
        }
        // the tile's last segment is MMQ's direct store, the earlier ones its fixup (summed last-to-first)
        let mut last = nseg - 1;
        while last > 0 && segment(tile, last, bpn, total, grid).1 == 0 {
            last -= 1;
        }
        let mut j = 0;
        while j < NC {
            unroll!();
            let idx = (j as i32).wrapping_mul(nrows).wrapping_add(row);
            store_dst(dst, idx, *part.add(((last as usize) * NC + j) * WARP as usize + lane as usize), type_dst);
            if last > 0 {
                let mut sum = 0f32;
                let mut q = last - 1;
                while q >= 0 {
                    if segment(tile, q, bpn, total, grid).1 != 0 {
                        sum = add(sum, *part.add(((q as usize) * NC + j) * WARP as usize + lane as usize));
                    }
                    q -= 1;
                }
                store_dst(dst, idx, add(load_dst(dst, idx, type_dst), sum), type_dst);
            }
            j += 1;
        }
    }

    // GENERATED KERNELS BEGIN
    #[kernel] pub unsafe fn mmq_rows_q4_k_nc1(x: *const u8, y: *const u8, dst: *mut u8, nrows: i32, bpn: i32, stride_row_x: i32, ncols_y: i32, total: i32, grid: i32, type_dst: i32) { mmq_rows::<7, 1>(x, y, dst, nrows, bpn, stride_row_x, ncols_y, total, grid, type_dst) }
    #[kernel] pub unsafe fn mmq_rows_q4_k_nc2(x: *const u8, y: *const u8, dst: *mut u8, nrows: i32, bpn: i32, stride_row_x: i32, ncols_y: i32, total: i32, grid: i32, type_dst: i32) { mmq_rows::<7, 2>(x, y, dst, nrows, bpn, stride_row_x, ncols_y, total, grid, type_dst) }
    #[kernel] pub unsafe fn mmq_rows_q4_k_nc3(x: *const u8, y: *const u8, dst: *mut u8, nrows: i32, bpn: i32, stride_row_x: i32, ncols_y: i32, total: i32, grid: i32, type_dst: i32) { mmq_rows::<7, 3>(x, y, dst, nrows, bpn, stride_row_x, ncols_y, total, grid, type_dst) }
    #[kernel] pub unsafe fn mmq_rows_q4_k_nc4(x: *const u8, y: *const u8, dst: *mut u8, nrows: i32, bpn: i32, stride_row_x: i32, ncols_y: i32, total: i32, grid: i32, type_dst: i32) { mmq_rows::<7, 4>(x, y, dst, nrows, bpn, stride_row_x, ncols_y, total, grid, type_dst) }
    #[kernel] pub unsafe fn mmq_rows_q4_k_nc5(x: *const u8, y: *const u8, dst: *mut u8, nrows: i32, bpn: i32, stride_row_x: i32, ncols_y: i32, total: i32, grid: i32, type_dst: i32) { mmq_rows::<7, 5>(x, y, dst, nrows, bpn, stride_row_x, ncols_y, total, grid, type_dst) }
    #[kernel] pub unsafe fn mmq_rows_q4_k_nc6(x: *const u8, y: *const u8, dst: *mut u8, nrows: i32, bpn: i32, stride_row_x: i32, ncols_y: i32, total: i32, grid: i32, type_dst: i32) { mmq_rows::<7, 6>(x, y, dst, nrows, bpn, stride_row_x, ncols_y, total, grid, type_dst) }
    #[kernel] pub unsafe fn mmq_rows_q4_k_nc7(x: *const u8, y: *const u8, dst: *mut u8, nrows: i32, bpn: i32, stride_row_x: i32, ncols_y: i32, total: i32, grid: i32, type_dst: i32) { mmq_rows::<7, 7>(x, y, dst, nrows, bpn, stride_row_x, ncols_y, total, grid, type_dst) }
    #[kernel] pub unsafe fn mmq_rows_q4_k_nc8(x: *const u8, y: *const u8, dst: *mut u8, nrows: i32, bpn: i32, stride_row_x: i32, ncols_y: i32, total: i32, grid: i32, type_dst: i32) { mmq_rows::<7, 8>(x, y, dst, nrows, bpn, stride_row_x, ncols_y, total, grid, type_dst) }
    #[kernel] pub unsafe fn mmq_rows_q5_k_nc1(x: *const u8, y: *const u8, dst: *mut u8, nrows: i32, bpn: i32, stride_row_x: i32, ncols_y: i32, total: i32, grid: i32, type_dst: i32) { mmq_rows::<8, 1>(x, y, dst, nrows, bpn, stride_row_x, ncols_y, total, grid, type_dst) }
    #[kernel] pub unsafe fn mmq_rows_q5_k_nc2(x: *const u8, y: *const u8, dst: *mut u8, nrows: i32, bpn: i32, stride_row_x: i32, ncols_y: i32, total: i32, grid: i32, type_dst: i32) { mmq_rows::<8, 2>(x, y, dst, nrows, bpn, stride_row_x, ncols_y, total, grid, type_dst) }
    #[kernel] pub unsafe fn mmq_rows_q5_k_nc3(x: *const u8, y: *const u8, dst: *mut u8, nrows: i32, bpn: i32, stride_row_x: i32, ncols_y: i32, total: i32, grid: i32, type_dst: i32) { mmq_rows::<8, 3>(x, y, dst, nrows, bpn, stride_row_x, ncols_y, total, grid, type_dst) }
    #[kernel] pub unsafe fn mmq_rows_q5_k_nc4(x: *const u8, y: *const u8, dst: *mut u8, nrows: i32, bpn: i32, stride_row_x: i32, ncols_y: i32, total: i32, grid: i32, type_dst: i32) { mmq_rows::<8, 4>(x, y, dst, nrows, bpn, stride_row_x, ncols_y, total, grid, type_dst) }
    #[kernel] pub unsafe fn mmq_rows_q5_k_nc5(x: *const u8, y: *const u8, dst: *mut u8, nrows: i32, bpn: i32, stride_row_x: i32, ncols_y: i32, total: i32, grid: i32, type_dst: i32) { mmq_rows::<8, 5>(x, y, dst, nrows, bpn, stride_row_x, ncols_y, total, grid, type_dst) }
    #[kernel] pub unsafe fn mmq_rows_q5_k_nc6(x: *const u8, y: *const u8, dst: *mut u8, nrows: i32, bpn: i32, stride_row_x: i32, ncols_y: i32, total: i32, grid: i32, type_dst: i32) { mmq_rows::<8, 6>(x, y, dst, nrows, bpn, stride_row_x, ncols_y, total, grid, type_dst) }
    #[kernel] pub unsafe fn mmq_rows_q5_k_nc7(x: *const u8, y: *const u8, dst: *mut u8, nrows: i32, bpn: i32, stride_row_x: i32, ncols_y: i32, total: i32, grid: i32, type_dst: i32) { mmq_rows::<8, 7>(x, y, dst, nrows, bpn, stride_row_x, ncols_y, total, grid, type_dst) }
    #[kernel] pub unsafe fn mmq_rows_q5_k_nc8(x: *const u8, y: *const u8, dst: *mut u8, nrows: i32, bpn: i32, stride_row_x: i32, ncols_y: i32, total: i32, grid: i32, type_dst: i32) { mmq_rows::<8, 8>(x, y, dst, nrows, bpn, stride_row_x, ncols_y, total, grid, type_dst) }
    // GENERATED KERNELS END
}

fn main() {}
