#![allow(unsafe_op_in_unsafe_fn)]
//! titan noblas: cuda-oxide float GEMM / GEMV and Philox RNG kernels that replace cuBLAS, cuBLASLt and cuRAND in
//! titan's nvcc-free (`oxide`) build (candle's CUDA matmul, mistralrs-quant's cuBLASLt wrapper, candle's rand_*).
//!
//! One problem description for every kernel (row-major semantics, any strides, batch in grid z):
//!   D(i, j) = alpha * sum_k A(i, k) B(k, j) + beta * C(i, j) + bias(i, j),  i < m, j < n, batch z
//!   A(i, k) = a[z sa + i a_s0 + k a_s1], B(k, j) = b[z sb + k b_s0 + j b_s1], D(i, j) = d[z sd + i d_s0 + j d_s1],
//!   C(i, j) = c[z sc + i c_s0 + j c_s1] (read only when beta != 0 and c is not null; c may alias d),
//!   bias(i, j) = bias[i bias_s0 + j bias_s1] (when not null). Products and sums in f32, D rounded once (RN).
//! Element types: f32 / f16 / bf16 (all operands one type). Kernels:
//!   tc_*    f16 / bf16 tensor cores: mma.sync m16n8k16 (f32 accumulate), BM x 128 x 32 tiles (BM 128 / 64 / 32),
//!           8 warps, 3-stage cp.async pipeline, XOR-swizzled shared tiles read by ldmatrix (.trans for an A that
//!           is contiguous along m or a B contiguous along n). Needs 16-byte aligned rows: the contiguous dimension
//!           and the other stride multiples of 8 elements, 16-byte aligned base and batch strides.
//!   simt_*  any type and any strides (the f32 path and the fallback): 128 x 128 x 8 tiles, 8 x 8 outputs per
//!           thread (FFMA), register-staged double-buffered shared tiles.
//!   gemv_k* m <= R (1 / 2 / 4 / 8) rows, B contiguous along k: one warp per column j (16-byte loads when
//!           aligned, `_v`), warp-shuffle reduction.
//!   gemv_n* m <= R rows, B contiguous along n: one lane per column, 8 warps split k, shared-memory reduction.
//!   philox_* counter-based Philox4x32-10 uniform (0, 1] and Box-Muller normal fills (f32 / f64), seeded and
//!           stateless: element e of a call with (seed, offset) is a pure function of (seed, offset, e).
//! The host-side choice of kernel (mirrored by candle's cuda_backend/oxide_gemm.rs) is `gate::plan`.
#![allow(non_snake_case, clippy::missing_safety_doc, dead_code, unconditional_panic)]

use cuda_device::{SharedArray, kernel, launch_bounds, ptx_asm, thread, warp};
use cuda_host::cuda_module;

mod bench;
mod gate;
mod plan;

#[cuda_module]
pub mod kernels {
    use super::*;

    macro_rules! unroll {
        () => {
            cuda_device::thread::__unroll_config::<0>();
        };
    }

    /// Straight-line unrolling (cuda-oxide's #[unroll] skips loops it cannot prove constant, which left the
    /// accumulator arrays in local memory): copies `i` = 0 .. 7 of the body, each guarded by `i < bound` (a constant
    /// after monomorphisation, so the dead copies fold away).
    macro_rules! rep8 {
        ($i:ident: $t:ty, $bound:expr, $b:block) => {{
            { let $i: $t = 0; if $i < $bound $b }
            { let $i: $t = 1; if $i < $bound $b }
            { let $i: $t = 2; if $i < $bound $b }
            { let $i: $t = 3; if $i < $bound $b }
            { let $i: $t = 4; if $i < $bound $b }
            { let $i: $t = 5; if $i < $bound $b }
            { let $i: $t = 6; if $i < $bound $b }
            { let $i: $t = 7; if $i < $bound $b }
        }};
    }

    pub const DT_F32: u32 = 0;
    pub const DT_F16: u32 = 1;
    pub const DT_BF16: u32 = 2;
    /// k per pipeline stage (tensor-core kernels).
    pub const BK: i32 = 32;
    /// n per block (tensor-core kernels).
    pub const BN: i32 = 128;
    pub const STAGES: i32 = 3;

    // ---------------------------------------------------------------- scalar helpers
    #[inline(always)]
    pub fn fma(a: f32, b: f32, c: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("fma.rn.f32 %0, %1, %2, %3;", out("=f") r, in("f") a, in("f") b, in("f") c, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn mul(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("mul.rn.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn add(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("add.rn.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn h2f(x: u16) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("cvt.f32.f16 %0, %1;", out("=f") r, in("h") x, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn f2h(x: f32) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("cvt.rn.f16.f32 %0, %1;", out("=h") r, in("f") x, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn bf2f(x: u16) -> f32 {
        f32::from_bits((x as u32) << 16)
    }
    #[inline(always)]
    pub fn f2bf(x: f32) -> u16 {
        let r: u16;
        unsafe { ptx_asm!("cvt.rn.bf16.f32 %0, %1;", out("=h") r, in("f") x, options(register_only)); }
        r
    }
    /// Packed pair `lo` (low half) / `hi` in the element type.
    #[inline(always)]
    pub fn pack2<const DT: u32>(lo: f32, hi: f32) -> u32 {
        let r: u32;
        if DT == DT_F16 {
            unsafe { ptx_asm!("cvt.rn.f16x2.f32 %0, %1, %2;", out("=r") r, in("f") hi, in("f") lo, options(register_only)); }
        } else {
            unsafe { ptx_asm!("cvt.rn.bf16x2.f32 %0, %1, %2;", out("=r") r, in("f") hi, in("f") lo, options(register_only)); }
        }
        r
    }
    /// The two halves of a packed pair as f32 (low first).
    #[inline(always)]
    pub fn unpack2<const DT: u32>(w: u32) -> (f32, f32) {
        if DT == DT_F16 {
            (h2f((w & 0xffff) as u16), h2f((w >> 16) as u16))
        } else {
            (f32::from_bits(w << 16), f32::from_bits(w & 0xffff_0000))
        }
    }
    #[inline(always)]
    pub fn esize<const DT: u32>() -> u64 {
        if DT == DT_F32 { 4 } else { 2 }
    }

    // ---------------------------------------------------------------- global memory
    #[inline(always)]
    pub unsafe fn ldg_u16(p: u64) -> u16 {
        let r: u16;
        ptx_asm!("ld.global.nc.u16 %0, [%1];", out("=h") r, in("l") p);
        r
    }
    #[inline(always)]
    pub unsafe fn ldg_u32(p: u64) -> u32 {
        let r: u32;
        ptx_asm!("ld.global.nc.u32 %0, [%1];", out("=r") r, in("l") p);
        r
    }
    #[inline(always)]
    pub unsafe fn ldg_v4(p: u64) -> (u32, u32, u32, u32) {
        let (a, b, c, d): (u32, u32, u32, u32);
        ptx_asm!("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];", out("=r") a, out("=r") b, out("=r") c, out("=r") d, in("l") p);
        (a, b, c, d)
    }
    /// Coherent loads (C may alias D).
    #[inline(always)]
    pub unsafe fn ld_u16(p: u64) -> u16 {
        let r: u16;
        ptx_asm!("ld.global.u16 %0, [%1];", out("=h") r, in("l") p, clobber("memory"));
        r
    }
    #[inline(always)]
    pub unsafe fn ld_u32(p: u64) -> u32 {
        let r: u32;
        ptx_asm!("ld.global.u32 %0, [%1];", out("=r") r, in("l") p, clobber("memory"));
        r
    }
    #[inline(always)]
    pub unsafe fn st_u16(p: u64, v: u16) {
        ptx_asm!("st.global.u16 [%0], %1;", in("l") p, in("h") v, clobber("memory"));
    }
    #[inline(always)]
    pub unsafe fn st_u32(p: u64, v: u32) {
        ptx_asm!("st.global.u32 [%0], %1;", in("l") p, in("r") v, clobber("memory"));
    }
    #[inline(always)]
    pub unsafe fn st_v2u32(p: u64, a: u32, b: u32) {
        ptx_asm!("st.global.v2.u32 [%0], {%1, %2};", in("l") p, in("r") a, in("r") b, clobber("memory"));
    }
    #[inline(always)]
    pub unsafe fn st_u64(p: u64, v: u64) {
        ptx_asm!("st.global.u64 [%0], %1;", in("l") p, in("l") v, clobber("memory"));
    }
    /// Read-only element (A, B, bias) at byte address `p` as f32.
    #[inline(always)]
    pub unsafe fn ldx<const DT: u32>(p: u64) -> f32 {
        if DT == DT_F32 {
            f32::from_bits(ldg_u32(p))
        } else if DT == DT_F16 {
            h2f(ldg_u16(p))
        } else {
            bf2f(ldg_u16(p))
        }
    }
    /// Coherent element load (C).
    #[inline(always)]
    pub unsafe fn ldc<const DT: u32>(p: u64) -> f32 {
        if DT == DT_F32 {
            f32::from_bits(ld_u32(p))
        } else if DT == DT_F16 {
            h2f(ld_u16(p))
        } else {
            bf2f(ld_u16(p))
        }
    }
    #[inline(always)]
    pub unsafe fn stx<const DT: u32>(p: u64, v: f32) {
        if DT == DT_F32 {
            st_u32(p, v.to_bits())
        } else if DT == DT_F16 {
            st_u16(p, f2h(v))
        } else {
            st_u16(p, f2bf(v))
        }
    }

    /// Output-side parameters shared by every kernel (by value through the helpers).
    #[derive(Clone, Copy)]
    pub struct Epi {
        pub d: u64,
        pub c: u64,
        pub bias: u64,
        pub m: i32,
        pub n: i32,
        pub d_s0: i64,
        pub d_s1: i64,
        pub c_s0: i64,
        pub c_s1: i64,
        pub bias_s0: i64,
        pub bias_s1: i64,
        pub alpha: f32,
        pub beta: f32,
        /// d, c already offset to this batch
        pub pair: bool,
    }

    /// alpha * acc + beta * C(i, j) + bias(i, j), in this order, f32.
    #[inline(always)]
    pub unsafe fn epi_val<const DT: u32>(e: &Epi, i: i32, j: i32, acc: f32) -> f32 {
        let es = esize::<DT>();
        let mut x = mul(e.alpha, acc);
        if e.c != 0 && e.beta != 0.0 {
            let cv = ldc::<DT>(e.c + ((i as i64 * e.c_s0 + j as i64 * e.c_s1) as u64) * es);
            x = fma(e.beta, cv, x);
        }
        if e.bias != 0 {
            x = add(x, ldx::<DT>(e.bias + ((i as i64 * e.bias_s0 + j as i64 * e.bias_s1) as u64) * es));
        }
        x
    }
    #[inline(always)]
    pub unsafe fn epi1<const DT: u32>(e: &Epi, i: i32, j: i32, acc: f32) {
        if i < e.m && j < e.n {
            let x = epi_val::<DT>(e, i, j, acc);
            stx::<DT>(e.d + ((i as i64 * e.d_s0 + j as i64 * e.d_s1) as u64) * esize::<DT>(), x);
        }
    }
    /// D(i, j) and D(i, j + 1); one 2-element store when `e.pair` (d_s1 == 1, even d_s0, aligned base; j even).
    #[inline(always)]
    pub unsafe fn epi2<const DT: u32>(e: &Epi, i: i32, j: i32, a0: f32, a1: f32) {
        if i < e.m {
            if e.pair && j + 1 < e.n {
                let x0 = epi_val::<DT>(e, i, j, a0);
                let x1 = epi_val::<DT>(e, i, j + 1, a1);
                let p = e.d + ((i as i64 * e.d_s0 + j as i64) as u64) * esize::<DT>();
                if DT == DT_F32 {
                    st_v2u32(p, x0.to_bits(), x1.to_bits());
                } else {
                    st_u32(p, pack2::<DT>(x0, x1));
                }
            } else {
                epi1::<DT>(e, i, j, a0);
                epi1::<DT>(e, i, j + 1, a1);
            }
        }
    }

    // ---------------------------------------------------------------- shared memory / tensor cores
    #[inline(always)]
    pub unsafe fn cp_async16(dst: u32, src: u64, bytes: u32) {
        ptx_asm!("cp.async.cg.shared.global [%0], [%1], 16, %2;", in("r") dst, in("l") src, in("r") bytes, clobber("memory"));
    }
    #[inline(always)]
    pub unsafe fn cp_commit() {
        ptx_asm!("cp.async.commit_group;", clobber("memory"));
    }
    #[inline(always)]
    pub unsafe fn cp_wait1() {
        ptx_asm!("cp.async.wait_group 1;", clobber("memory"));
    }
    #[inline(always)]
    pub unsafe fn cp_wait0() {
        ptx_asm!("cp.async.wait_group 0;", clobber("memory"));
    }
    #[inline(always)]
    pub unsafe fn ldsm_x4(addr: u32) -> (u32, u32, u32, u32) {
        let (a, b, c, d): (u32, u32, u32, u32);
        ptx_asm!("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];", out("=r") a, out("=r") b, out("=r") c, out("=r") d, in("r") addr, clobber("memory"));
        (a, b, c, d)
    }
    #[inline(always)]
    pub unsafe fn ldsm_x4_t(addr: u32) -> (u32, u32, u32, u32) {
        let (a, b, c, d): (u32, u32, u32, u32);
        ptx_asm!("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];", out("=r") a, out("=r") b, out("=r") c, out("=r") d, in("r") addr, clobber("memory"));
        (a, b, c, d)
    }
    #[inline(always)]
    pub unsafe fn lds_v4f(addr: u32) -> (f32, f32, f32, f32) {
        let (a, b, c, d): (f32, f32, f32, f32);
        ptx_asm!("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];", out("=f") a, out("=f") b, out("=f") c, out("=f") d, in("r") addr, clobber("memory"));
        (a, b, c, d)
    }
    #[inline(always)]
    pub unsafe fn lds_f(addr: u32) -> f32 {
        let r: f32;
        ptx_asm!("ld.shared.f32 %0, [%1];", out("=f") r, in("r") addr, clobber("memory"));
        r
    }
    #[inline(always)]
    pub unsafe fn sts_f(addr: u32, v: f32) {
        ptx_asm!("st.shared.f32 [%0], %1;", in("r") addr, in("f") v, clobber("memory"));
    }

    /// D += A B, m16n8k16, f32 accumulators; f16 (DT_F16) or bf16 inputs.
    #[inline(always)]
    pub fn mma<const DT: u32>(c: [f32; 4], a: (u32, u32, u32, u32), b0: u32, b1: u32) -> [f32; 4] {
        let (d0, d1, d2, d3): (f32, f32, f32, f32);
        unsafe {
            if DT == DT_F16 {
                ptx_asm!(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};",
                    out("=f") d0, out("=f") d1, out("=f") d2, out("=f") d3,
                    in("r") a.0, in("r") a.1, in("r") a.2, in("r") a.3, in("r") b0, in("r") b1,
                    in("f") c[0], in("f") c[1], in("f") c[2], in("f") c[3],
                    options(register_only),
                );
            } else {
                ptx_asm!(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};",
                    out("=f") d0, out("=f") d1, out("=f") d2, out("=f") d3,
                    in("r") a.0, in("r") a.1, in("r") a.2, in("r") a.3, in("r") b0, in("r") b1,
                    in("f") c[0], in("f") c[1], in("f") c[2], in("f") c[3],
                    options(register_only),
                );
            }
        }
        [d0, d1, d2, d3]
    }

    /// Byte offset of 16-byte chunk `ch` of row `row` in a tile with `cpr` chunks per row: rows of 4 chunks (64 B)
    /// XOR the chunk with (row / 2) % 4, wider rows with row % 8, so that the 8 row addresses of an ldmatrix 8 x 8
    /// matrix (8 consecutive rows from a multiple of 8) hit 8 different 16-byte bank groups.
    #[inline(always)]
    pub fn swz(row: u32, ch: u32, cpr: u32) -> u32 {
        if cpr >= 8 { row * cpr * 16 + ((ch ^ (row & 7)) << 4) } else { row * 64 + ((ch ^ ((row >> 1) & 3)) << 4) }
    }

    /// One pipeline stage: the A tile (BM x 32) and the B tile (32 x 128) of k-tile `k0 / 32` by cp.async, chunks
    /// outside the problem zero-filled without a read. A k-contiguous (!AT): shared [BM][32]; m-contiguous: [32][BM].
    /// B k-contiguous (!BT): shared [128][32]; n-contiguous: [32][128].
    #[inline(always)]
    pub unsafe fn tc_load<const AT: bool, const BT: bool, const BM: i32>(
        sa_: u32, sb_: u32, a: u64, b: u64, m0: i32, n0: i32, k0: i32, m: i32, n: i32, k: i32, lda: i64, ldb: i64, tid: i32,
    ) {
        rep8!(jj: i32, (BM * 4 + 255) / 256, {
            let i = tid + 256 * jj;
            if i < BM * 4 {
                if !AT {
                    let row = i >> 2;
                    let ch = i & 3;
                    let gm = m0 + row;
                    let gk = k0 + ch * 8;
                    let ok = gm < m && gk < k;
                    let src = if ok { a + ((gm as i64 * lda + gk as i64) as u64) * 2 } else { a };
                    cp_async16(sa_ + swz(row as u32, ch as u32, 4), src, if ok { 16 } else { 0 });
                } else {
                    let cpr = BM / 8;
                    let row = i / cpr;
                    let ch = i % cpr;
                    let gk = k0 + row;
                    let gm = m0 + ch * 8;
                    let ok = gk < k && gm < m;
                    let src = if ok { a + ((gk as i64 * lda + gm as i64) as u64) * 2 } else { a };
                    cp_async16(sa_ + swz(row as u32, ch as u32, cpr as u32), src, if ok { 16 } else { 0 });
                }
            }
        });
        rep8!(jj: i32, 2, {
            let i = tid + 256 * jj;
            if !BT {
                let row = i >> 2;
                let ch = i & 3;
                let gn = n0 + row;
                let gk = k0 + ch * 8;
                let ok = gn < n && gk < k;
                let src = if ok { b + ((gn as i64 * ldb + gk as i64) as u64) * 2 } else { b };
                cp_async16(sb_ + swz(row as u32, ch as u32, 4), src, if ok { 16 } else { 0 });
            } else {
                let row = i >> 4;
                let ch = i & 15;
                let gk = k0 + row;
                let gn = n0 + ch * 8;
                let ok = gk < k && gn < n;
                let src = if ok { b + ((gk as i64 * ldb + gn as i64) as u64) * 2 } else { b };
                cp_async16(sb_ + swz(row as u32, ch as u32, 16), src, if ok { 16 } else { 0 });
            }
        });
    }

    /// Tensor-core GEMM body: grid (ceil(n / 128), ceil(m / BM), batch), block 256 (8 warps: 2 along m x 4 along n,
    /// warp tile BM / 2 x 32). `sm` is the shared base (STAGES x (BM x 64 + 8192) bytes).
    #[inline(always)]
    pub unsafe fn tc_body<const DT: u32, const AT: bool, const BT: bool, const BM: i32>(
        sm: u32, a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32,
        a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64,
        bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32,
    ) {
        let tid = thread::threadIdx_x() as i32;
        let lane = tid & 31;
        let wid = tid >> 5;
        let g = lane >> 2;
        let t = lane & 3;
        let q = (lane >> 3) as u32;
        let r = (lane & 7) as u32;
        let n0 = thread::blockIdx_x() as i32 * BN;
        let m0 = thread::blockIdx_y() as i32 * BM;
        let zz = thread::blockIdx_z() as i32;
        let z = (zz / nsplit) as i64;
        let sp = zz % nsplit;
        let ab = a as u64 + (z * sa) as u64 * 2;
        let bb = b as u64 + (z * sb) as u64 * 2;
        // the non-contiguous stride of each operand
        let lda = if AT { a_s1 } else { a_s0 };
        let ldb = if BT { b_s0 } else { b_s1 };
        let a_bytes = (BM * 64) as u32;
        let stage = a_bytes + 8192;
        let wmb = (wid & 1) * (BM / 2);
        let wnb = (wid >> 1) * 32;
        // split-K: this block's k-tiles [t0, t0 + nk)
        let tiles = (k + BK - 1) / BK;
        let per = (tiles + nsplit - 1) / nsplit;
        let t0 = sp * per;
        let nk = if t0 + per < tiles { per } else if tiles > t0 { tiles - t0 } else { 0 };

        let mut acc = [[0f32; 4]; 16];

        rep8!(s: i32, STAGES - 1, {
            if s < nk {
                let base = sm + s as u32 * stage;
                tc_load::<AT, BT, BM>(base, base + a_bytes, ab, bb, m0, n0, (t0 + s) * BK, m, n, k, lda, ldb, tid);
            }
            cp_commit();
        });
        let mut kt = 0;
        while kt < nk {
            cp_wait1();
            thread::sync_threads();
            let nx = kt + STAGES - 1;
            if nx < nk {
                let base = sm + (nx % STAGES) as u32 * stage;
                tc_load::<AT, BT, BM>(base, base + a_bytes, ab, bb, m0, n0, (t0 + nx) * BK, m, n, k, lda, ldb, tid);
            }
            cp_commit();
            let smA = sm + (kt % STAGES) as u32 * stage;
            let smB = smA + a_bytes;
            rep8!(ks: u32, 2, {
                let mut af = [(0u32, 0u32, 0u32, 0u32); 4];
                rep8!(mt: i32, BM / 32, {
                    af[mt as usize] = if !AT {
                        let row = (wmb + mt * 16) as u32 + r + 8 * (q & 1);
                        ldsm_x4(smA + swz(row, 2 * ks + (q >> 1), 4))
                    } else {
                        let krow = 16 * ks + r + 8 * (q >> 1);
                        let mch = ((wmb + mt * 16) / 8) as u32 + (q & 1);
                        ldsm_x4_t(smA + swz(krow, mch, (BM / 8) as u32))
                    };
                });
                let mut bf = [[0u32; 2]; 4];
                rep8!(np: i32, 2, {
                    let (b0, b1, b2, b3) = if !BT {
                        let nrow = (wnb + np * 16) as u32 + r + 8 * (q >> 1);
                        ldsm_x4(smB + swz(nrow, 2 * ks + (q & 1), 4))
                    } else {
                        let krow = 16 * ks + r + 8 * (q & 1);
                        let nch = ((wnb + np * 16) / 8) as u32 + (q >> 1);
                        ldsm_x4_t(smB + swz(krow, nch, 16))
                    };
                    bf[(2 * np) as usize] = [b0, b1];
                    bf[(2 * np + 1) as usize] = [b2, b3];
                });
                rep8!(mt: i32, BM / 32, {
                    rep8!(nt: i32, 4, {
                        let ix = (mt * 4 + nt) as usize;
                        acc[ix] = mma::<DT>(acc[ix], af[mt as usize], bf[nt as usize][0], bf[nt as usize][1]);
                    });
                });
            });
            kt += 1;
        }
        cp_wait0();

        if nsplit > 1 {
            // raw f32 partial sums to ws[sp][z][i][j]; splitk_reduce_* applies the epilogue
            let batch = (thread::gridDim_z() as i32 / nsplit) as i64;
            let wsb = ws as u64 + (((sp as i64 * batch + z) * m as i64 * n as i64) as u64) * 4;
            rep8!(mt: i32, BM / 32, {
                rep8!(nt: i32, 4, {
                    let v = acc[(mt * 4 + nt) as usize];
                    let i0 = m0 + wmb + mt * 16 + g;
                    let j0 = n0 + wnb + nt * 8 + 2 * t;
                    ws_put(wsb, m, n, i0, j0, v[0]);
                    ws_put(wsb, m, n, i0, j0 + 1, v[1]);
                    ws_put(wsb, m, n, i0 + 8, j0, v[2]);
                    ws_put(wsb, m, n, i0 + 8, j0 + 1, v[3]);
                });
            });
            return;
        }
        let es = 2u64;
        let e = Epi {
            d: d as u64 + (z * sd) as u64 * es,
            c: if c.is_null() { 0 } else { c as u64 + (z * sc) as u64 * es },
            bias: bias as u64,
            m,
            n,
            d_s0,
            d_s1,
            c_s0,
            c_s1,
            bias_s0,
            bias_s1,
            alpha,
            beta,
            pair: d_s1 == 1 && (d_s0 & 1) == 0 && (sd & 1) == 0 && (d as u64 & 3) == 0,
        };
        rep8!(mt: i32, BM / 32, {
            rep8!(nt: i32, 4, {
                let v = acc[(mt * 4 + nt) as usize];
                let i0 = m0 + wmb + mt * 16 + g;
                let j0 = n0 + wnb + nt * 8 + 2 * t;
                epi2::<DT>(&e, i0, j0, v[0], v[1]);
                epi2::<DT>(&e, i0 + 8, j0, v[2], v[3]);
            });
        });
    }

    /// One raw f32 partial (split-K) at ws + (i n + j) floats.
    #[inline(always)]
    pub unsafe fn ws_put(wsb: u64, m: i32, n: i32, i: i32, j: i32, v: f32) {
        if i < m && j < n {
            st_u32(wsb + ((i as i64 * n as i64 + j as i64) as u64) * 4, v.to_bits());
        }
    }

    /// Split-K merge: D(i, j) = epilogue(sum over splits s in order of ws[s][z][i][j]); grid (ceil(m n / 256), 1,
    /// batch), block 256.
    #[inline(always)]
    pub unsafe fn reduce_body<const DT: u32>(
        ws: *const f32, nsplit: i32, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, d_s0: i64, d_s1: i64,
        c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sd: i64, sc: i64, alpha: f32, beta: f32,
    ) {
        let e_ = thread::blockIdx_x() as i64 * 256 + thread::threadIdx_x() as i64;
        let mn = m as i64 * n as i64;
        if e_ >= mn {
            return;
        }
        let z = thread::blockIdx_z() as i64;
        let batch = thread::gridDim_z() as i64;
        let i = (e_ / n as i64) as i32;
        let j = (e_ % n as i64) as i32;
        let mut x = 0f32;
        let mut s = 0;
        while s < nsplit {
            x = add(x, f32::from_bits(ld_u32(ws as u64 + (((s as i64 * batch + z) * mn + e_) as u64) * 4)));
            s += 1;
        }
        let es = esize::<DT>();
        let e = Epi {
            d: d as u64 + (z * sd) as u64 * es,
            c: if c.is_null() { 0 } else { c as u64 + (z * sc) as u64 * es },
            bias: bias as u64,
            m,
            n,
            d_s0,
            d_s1,
            c_s0,
            c_s1,
            bias_s0,
            bias_s1,
            alpha,
            beta,
            pair: false,
        };
        epi1::<DT>(&e, i, j, x);
    }

    // ---------------------------------------------------------------- SIMT GEMM (any type, any strides)
    /// Rows (or columns) per thread of a T x T SIMT tile: T / 16 (8 or 4); output row `r` of thread row `ty`.
    #[inline(always)]
    pub fn simt_row(ty: u32, r: u32) -> u32 {
        if r < 4 { ty * 4 + r } else { 64 + ty * 4 + r - 4 }
    }

    /// Global -> registers for one 8-deep k-tile: A (T x 8) and B (8 x T), T / 32 elements each per thread, consecutive
    /// threads along each operand's contiguous dimension (AT / BT select the mapping, any strides work).
    #[inline(always)]
    pub unsafe fn simt_ld<const DT: u32, const AT: bool, const BT: bool, const T: i32>(
        ra: &mut [f32; 4], rb: &mut [f32; 4], ab: u64, bb: u64, m0: i32, n0: i32, k0: i32, m: i32, n: i32, k: i32,
        a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, tid: i32,
    ) {
        let es = esize::<DT>() as i64;
        rep8!(jj: i32, T / 32, {
            let e = tid + 256 * jj;
            let (row, kk) = if !AT { (e >> 3, e & 7) } else { (e % T, e / T) };
            let gm = m0 + row;
            let gk = k0 + kk;
            ra[jj as usize] = if gm < m && gk < k { ldx::<DT>(ab + ((gm as i64 * a_s0 + gk as i64 * a_s1) * es) as u64) } else { 0.0 };
            let (col, kk) = if BT { (e % T, e / T) } else { (e >> 3, e & 7) };
            let gn = n0 + col;
            let gk = k0 + kk;
            rb[jj as usize] = if gn < n && gk < k { ldx::<DT>(bb + ((gk as i64 * b_s0 + gn as i64 * b_s1) * es) as u64) } else { 0.0 };
        });
    }
    /// Registers -> shared [8][T + 4] (k-major) for A and B.
    #[inline(always)]
    pub unsafe fn simt_st<const AT: bool, const BT: bool, const T: i32>(sA: u32, sB: u32, ra: &[f32; 4], rb: &[f32; 4], tid: i32) {
        let srow = (T + 4) as u32;
        rep8!(jj: i32, T / 32, {
            let e = tid + 256 * jj;
            let (row, kk) = if !AT { (e >> 3, e & 7) } else { (e % T, e / T) };
            sts_f(sA + (kk as u32 * srow + row as u32) * 4, ra[jj as usize]);
            let (col, kk) = if BT { (e % T, e / T) } else { (e >> 3, e & 7) };
            sts_f(sB + (kk as u32 * srow + col as u32) * 4, rb[jj as usize]);
        });
    }

    /// SIMT GEMM, T x T tiles (T 128: 8 x 8 per thread; 64: 4 x 4), BK 8, register-staged double-buffered shared tiles.
    /// grid (ceil(n / T), ceil(m / T), batch x nsplit), block 256; `sm`: 2 operands x 2 buffers x 8 x (T + 4) floats.
    #[inline(always)]
    pub unsafe fn simt_body<const DT: u32, const AT: bool, const BT: bool, const T: i32>(
        sm: u32, a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32,
        a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64,
        bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32,
    ) {
        let tid = thread::threadIdx_x() as i32;
        let tx = (tid & 15) as u32;
        let ty = (tid >> 4) as u32;
        let n0 = thread::blockIdx_x() as i32 * T;
        let m0 = thread::blockIdx_y() as i32 * T;
        let zz = thread::blockIdx_z() as i32;
        let z = (zz / nsplit) as i64;
        let sp = zz % nsplit;
        let es = esize::<DT>();
        let ab = a as u64 + (z * sa) as u64 * es;
        let bb = b as u64 + (z * sb) as u64 * es;
        let srow = (T + 4) as u32;
        let buf = 8 * srow * 4; // bytes per (operand, buffer)
        let tiles = (k + 7) / 8;
        let per = (tiles + nsplit - 1) / nsplit;
        let t0 = sp * per;
        let nk = if t0 + per < tiles { per } else if tiles > t0 { tiles - t0 } else { 0 };
        let mut acc = [[0f32; 8]; 8];
        let mut ra = [0f32; 4];
        let mut rb = [0f32; 4];
        if nk > 0 {
            simt_ld::<DT, AT, BT, T>(&mut ra, &mut rb, ab, bb, m0, n0, t0 * 8, m, n, k, a_s0, a_s1, b_s0, b_s1, tid);
            simt_st::<AT, BT, T>(sm, sm + 2 * buf, &ra, &rb, tid);
        }
        thread::sync_threads();
        let mut kt = 0;
        while kt < nk {
            let cur = (kt & 1) as u32;
            if kt + 1 < nk {
                simt_ld::<DT, AT, BT, T>(&mut ra, &mut rb, ab, bb, m0, n0, (t0 + kt + 1) * 8, m, n, k, a_s0, a_s1, b_s0, b_s1, tid);
            }
            let sA = sm + cur * buf;
            let sB = sm + 2 * buf + cur * buf;
            rep8!(kk: u32, 8, {
                let (a0, a1, a2, a3) = lds_v4f(sA + (kk * srow + ty * 4) * 4);
                let (b0, b1, b2, b3) = lds_v4f(sB + (kk * srow + tx * 4) * 4);
                let mut av = [a0, a1, a2, a3, 0.0, 0.0, 0.0, 0.0];
                let mut bv = [b0, b1, b2, b3, 0.0, 0.0, 0.0, 0.0];
                if T == 128 {
                    let (a4, a5, a6, a7) = lds_v4f(sA + (kk * srow + 64 + ty * 4) * 4);
                    let (b4, b5, b6, b7) = lds_v4f(sB + (kk * srow + 64 + tx * 4) * 4);
                    av = [a0, a1, a2, a3, a4, a5, a6, a7];
                    bv = [b0, b1, b2, b3, b4, b5, b6, b7];
                }
                rep8!(i: i32, T / 16, {
                    rep8!(j: i32, T / 16, {
                        acc[i as usize][j as usize] = fma(av[i as usize], bv[j as usize], acc[i as usize][j as usize]);
                    });
                });
            });
            if kt + 1 < nk {
                let nb = 1 - cur;
                simt_st::<AT, BT, T>(sm + nb * buf, sm + 2 * buf + nb * buf, &ra, &rb, tid);
            }
            thread::sync_threads();
            kt += 1;
        }
        if nsplit > 1 {
            let batch = (thread::gridDim_z() as i32 / nsplit) as i64;
            let wsb = ws as u64 + (((sp as i64 * batch + z) * m as i64 * n as i64) as u64) * 4;
            rep8!(i: i32, T / 16, {
                rep8!(j: i32, T / 16, {
                    ws_put(wsb, m, n, m0 + simt_row(ty, i as u32) as i32, n0 + simt_row(tx, j as u32) as i32, acc[i as usize][j as usize]);
                });
            });
            return;
        }
        let e = Epi {
            d: d as u64 + (z * sd) as u64 * es,
            c: if c.is_null() { 0 } else { c as u64 + (z * sc) as u64 * es },
            bias: bias as u64,
            m,
            n,
            d_s0,
            d_s1,
            c_s0,
            c_s1,
            bias_s0,
            bias_s1,
            alpha,
            beta,
            pair: false,
        };
        rep8!(i: i32, T / 16, {
            rep8!(j: i32, T / 16, {
                epi1::<DT>(&e, m0 + simt_row(ty, i as u32) as i32, n0 + simt_row(tx, j as u32) as i32, acc[i as usize][j as usize]);
            });
        });
    }

    // ---------------------------------------------------------------- GEMV (m <= R)
    /// Dot of one 16-byte chunk of A and of B (8 halves or 4 floats) into `acc`, in element order.
    #[inline(always)]
    pub fn dot16<const DT: u32>(acc: f32, a: (u32, u32, u32, u32), b: (u32, u32, u32, u32)) -> f32 {
        if DT == DT_F32 {
            let mut x = fma(f32::from_bits(a.0), f32::from_bits(b.0), acc);
            x = fma(f32::from_bits(a.1), f32::from_bits(b.1), x);
            x = fma(f32::from_bits(a.2), f32::from_bits(b.2), x);
            fma(f32::from_bits(a.3), f32::from_bits(b.3), x)
        } else {
            let (a0, a1) = unpack2::<DT>(a.0);
            let (a2, a3) = unpack2::<DT>(a.1);
            let (a4, a5) = unpack2::<DT>(a.2);
            let (a6, a7) = unpack2::<DT>(a.3);
            let (b0, b1) = unpack2::<DT>(b.0);
            let (b2, b3) = unpack2::<DT>(b.1);
            let (b4, b5) = unpack2::<DT>(b.2);
            let (b6, b7) = unpack2::<DT>(b.3);
            let mut x = fma(a0, b0, acc);
            x = fma(a1, b1, x);
            x = fma(a2, b2, x);
            x = fma(a3, b3, x);
            x = fma(a4, b4, x);
            x = fma(a5, b5, x);
            x = fma(a6, b6, x);
            fma(a7, b7, x)
        }
    }

    /// B contiguous along k. CPB 8: one warp per column j, grid (ceil(n / 8), 1, batch). CPB 1: one block per column,
    /// its 8 warps split k (interleaved 16-byte chunks), partial sums merged in warp order through `red` (shared, 8 x 8
    /// floats), grid (n, 1, batch). Block 256. VEC: 16-byte chunks (A rows and B columns contiguous along k, k a
    /// multiple of the chunk, 16-byte aligned rows), else element loads.
    #[inline(always)]
    pub unsafe fn gemv_k_body<const DT: u32, const R: i32, const VEC: bool, const CPB: i32>(
        red: u32, a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32,
        a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64,
        bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32,
    ) {
        let tid = thread::threadIdx_x() as i32;
        let lane = tid & 31;
        let wid = tid >> 5;
        let ks = 8 / CPB;
        let wk = wid / CPB;
        let j = thread::blockIdx_x() as i32 * CPB + wid % CPB;
        if j >= n {
            return;
        }
        let z = thread::blockIdx_z() as i64;
        let es = esize::<DT>();
        let ab = a as u64 + (z * sa) as u64 * es;
        let bcol = b as u64 + ((z * sb + j as i64 * b_s1) as u64) * es;
        let mut acc = [0f32; 8];
        if VEC {
            let per = (16 / es) as i32;
            let nch = k / per;
            let mut ch = wk * 32 + lane;
            while ch < nch {
                let bv = ldg_v4(bcol + ch as u64 * 16);
                rep8!(i: i32, R, {
                    if i < m {
                        let av = ldg_v4(ab + ((i as i64 * a_s0) as u64) * es + ch as u64 * 16);
                        acc[i as usize] = dot16::<DT>(acc[i as usize], av, bv);
                    }
                });
                ch += 32 * ks;
            }
        } else {
            let mut kk = wk * 32 + lane;
            while kk < k {
                let bv = ldx::<DT>(bcol + ((kk as i64 * b_s0) as u64) * es);
                rep8!(i: i32, R, {
                    if i < m {
                        let av = ldx::<DT>(ab + ((i as i64 * a_s0 + kk as i64 * a_s1) as u64) * es);
                        acc[i as usize] = fma(av, bv, acc[i as usize]);
                    }
                });
                kk += 32 * ks;
            }
        }
        rep8!(i: i32, R, {
            let mut x = acc[i as usize];
            x = add(x, warp::shuffle_xor_f32_sync(0xffff_ffff, x, 16));
            x = add(x, warp::shuffle_xor_f32_sync(0xffff_ffff, x, 8));
            x = add(x, warp::shuffle_xor_f32_sync(0xffff_ffff, x, 4));
            x = add(x, warp::shuffle_xor_f32_sync(0xffff_ffff, x, 2));
            x = add(x, warp::shuffle_xor_f32_sync(0xffff_ffff, x, 1));
            acc[i as usize] = x;
        });
        if CPB == 1 {
            if lane == 0 {
                rep8!(i: i32, R, {
                    sts_f(red + ((wid * 8 + i) * 4) as u32, acc[i as usize]);
                });
            }
            thread::sync_threads();
            if wid == 0 {
                rep8!(i: i32, R, {
                    let mut x = 0f32;
                    rep8!(w: i32, 8, {
                        x = add(x, lds_f(red + ((w * 8 + i) * 4) as u32));
                    });
                    acc[i as usize] = x;
                });
            }
        }
        if lane == 0 && wk == 0 {
            let e = Epi {
                d: d as u64 + (z * sd) as u64 * es,
                c: if c.is_null() { 0 } else { c as u64 + (z * sc) as u64 * es },
                bias: bias as u64,
                m,
                n,
                d_s0,
                d_s1,
                c_s0,
                c_s1,
                bias_s0,
                bias_s1,
                alpha,
                beta,
                pair: false,
            };
            rep8!(i: i32, R, {
                epi1::<DT>(&e, i, j, acc[i as usize]);
            });
        }
    }

    /// B contiguous along n: lane -> column j = 32 blockIdx.x + lane, warp w -> k range w of 8; grid (ceil(n / 32),
    /// 1, batch), block 256; `red`: shared 8 x R x 32 floats.
    #[inline(always)]
    pub unsafe fn gemv_n_body<const DT: u32, const R: i32>(
        red: u32, a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32,
        a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64,
        bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32,
    ) {
        let tid = thread::threadIdx_x() as i32;
        let lane = tid & 31;
        let wid = tid >> 5;
        let j = thread::blockIdx_x() as i32 * 32 + lane;
        let z = thread::blockIdx_z() as i64;
        let es = esize::<DT>();
        let ab = a as u64 + (z * sa) as u64 * es;
        let bb = b as u64 + (z * sb) as u64 * es;
        let per = (k + 7) / 8;
        let kb = wid * per;
        let ke = if kb + per < k { kb + per } else { k };
        let mut acc = [0f32; 8];
        if j < n {
            let mut kk = kb;
            while kk < ke {
                let bv = ldx::<DT>(bb + ((kk as i64 * b_s0 + j as i64 * b_s1) as u64) * es);
                rep8!(i: i32, R, {
                    if i < m {
                        let av = ldx::<DT>(ab + ((i as i64 * a_s0 + kk as i64 * a_s1) as u64) * es);
                        acc[i as usize] = fma(av, bv, acc[i as usize]);
                    }
                });
                kk += 1;
            }
        }
        rep8!(i: i32, R, {
            sts_f(red + ((wid * R + i) * 32 + lane) as u32 * 4, acc[i as usize]);
        });
        thread::sync_threads();
        if wid == 0 && j < n {
            let e = Epi {
                d: d as u64 + (z * sd) as u64 * es,
                c: if c.is_null() { 0 } else { c as u64 + (z * sc) as u64 * es },
                bias: bias as u64,
                m,
                n,
                d_s0,
                d_s1,
                c_s0,
                c_s1,
                bias_s0,
                bias_s1,
                alpha,
                beta,
                pair: false,
            };
            rep8!(i: i32, R, {
                let mut x = 0f32;
                rep8!(w: i32, 8, {
                    x = add(x, lds_f(red + ((w * R + i) * 32 + lane) as u32 * 4));
                });
                epi1::<DT>(&e, i, j, x);
            });
        }
    }

    // ---------------------------------------------------------------- f64 reference (diagnostics)
    /// One thread per output: products and sums in f64 (exact products for f16 / bf16 inputs), the epilogue in
    /// f64, then one rounding to f32 and the element type. grid (ceil(n / 256), m, batch), block 256. Slow: for
    /// TITAN_GEMM_REF A/B runs, not for serving.
    #[inline(always)]
    pub unsafe fn ref_body<const DT: u32>(
        a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32,
        a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64,
        bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32,
    ) {
        let j = (thread::blockIdx_x() * 256 + thread::threadIdx_x()) as i32;
        let i = thread::blockIdx_y() as i32;
        if j >= n || i >= m {
            return;
        }
        let z = thread::blockIdx_z() as i64;
        let es = esize::<DT>();
        let ab = a as u64 + (z * sa) as u64 * es;
        let bb = b as u64 + (z * sb) as u64 * es;
        let mut acc = 0f64;
        let mut kk = 0;
        while kk < k {
            let av = ldx::<DT>(ab + ((i as i64 * a_s0 + kk as i64 * a_s1) as u64) * es) as f64;
            let bv = ldx::<DT>(bb + ((kk as i64 * b_s0 + j as i64 * b_s1) as u64) * es) as f64;
            acc += av * bv;
            kk += 1;
        }
        let mut x = alpha as f64 * acc;
        if !c.is_null() && beta != 0.0 {
            let cp = c as u64 + ((z * sc + i as i64 * c_s0 + j as i64 * c_s1) as u64) * es;
            x += beta as f64 * ldc::<DT>(cp) as f64;
        }
        if !bias.is_null() {
            x += ldx::<DT>(bias as u64 + ((i as i64 * bias_s0 + j as i64 * bias_s1) as u64) * es) as f64;
        }
        stx::<DT>(d as u64 + ((z * sd + i as i64 * d_s0 + j as i64 * d_s1) as u64) * es, x as f32);
    }

    // ---------------------------------------------------------------- Philox4x32-10
    #[inline(always)]
    pub fn mulhilo(a: u32, b: u32) -> (u32, u32) {
        let p = a as u64 * b as u64;
        ((p >> 32) as u32, p as u32)
    }
    /// Philox4x32-10 of counter (lo, hi, 0, 0) and key (seed lo, seed hi).
    #[inline(always)]
    pub fn philox(ctr: u64, seed: u64) -> (u32, u32, u32, u32) {
        let mut c0 = ctr as u32;
        let mut c1 = (ctr >> 32) as u32;
        let mut c2 = 0u32;
        let mut c3 = 0u32;
        let mut k0 = seed as u32;
        let mut k1 = (seed >> 32) as u32;
        let mut r = 0;
        while r < 10 {
            let (hi0, lo0) = mulhilo(0xD251_1F53, c0);
            let (hi1, lo1) = mulhilo(0xCD9E_8D57, c2);
            let n0 = hi1 ^ c1 ^ k0;
            let n2 = hi0 ^ c3 ^ k1;
            c0 = n0;
            c1 = lo1;
            c2 = n2;
            c3 = lo0;
            k0 = k0.wrapping_add(0x9E37_79B9);
            k1 = k1.wrapping_add(0xBB67_AE85);
            r += 1;
        }
        (c0, c1, c2, c3)
    }
    /// (x >> 8 + 1) / 2^24: uniform on (0, 1], exact in f32.
    #[inline(always)]
    pub fn u01f(x: u32) -> f32 {
        ((x >> 8) + 1) as f32 * f32::from_bits(0x3380_0000)
    }
    /// 53 random bits + 1 over 2^53: uniform on (0, 1], exact in f64.
    #[inline(always)]
    pub fn u01d(lo: u32, hi: u32) -> f64 {
        ((((hi as u64) << 21) ^ (lo as u64 >> 11)) + 1) as f64 * f64::from_bits(0x3CA0_0000_0000_0000)
    }

    pub const TWO_PI_F: f32 = 6.283_185_5;
    pub const TWO_PI_D: f64 = 6.283_185_307_179_586;

    // ================================================================ kernels (GENERATED by gen_kernels.py)
    // BEGIN GENERATED
    /// tensor cores, f16, A k-contiguous, B k-contiguous, BM 128
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn tc_f16_kk_128(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<u16, 24576, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        tc_body::<DT_F16, false, false, 128>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// tensor cores, f16, A k-contiguous, B k-contiguous, BM 64
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn tc_f16_kk_64(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<u16, 18432, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        tc_body::<DT_F16, false, false, 64>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// tensor cores, f16, A k-contiguous, B k-contiguous, BM 32
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn tc_f16_kk_32(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<u16, 15360, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        tc_body::<DT_F16, false, false, 32>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// tensor cores, f16, A k-contiguous, B n-contiguous, BM 128
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn tc_f16_kn_128(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<u16, 24576, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        tc_body::<DT_F16, false, true, 128>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// tensor cores, f16, A k-contiguous, B n-contiguous, BM 64
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn tc_f16_kn_64(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<u16, 18432, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        tc_body::<DT_F16, false, true, 64>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// tensor cores, f16, A k-contiguous, B n-contiguous, BM 32
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn tc_f16_kn_32(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<u16, 15360, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        tc_body::<DT_F16, false, true, 32>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// tensor cores, f16, A m-contiguous, B k-contiguous, BM 128
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn tc_f16_mk_128(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<u16, 24576, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        tc_body::<DT_F16, true, false, 128>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// tensor cores, f16, A m-contiguous, B k-contiguous, BM 64
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn tc_f16_mk_64(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<u16, 18432, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        tc_body::<DT_F16, true, false, 64>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// tensor cores, f16, A m-contiguous, B k-contiguous, BM 32
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn tc_f16_mk_32(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<u16, 15360, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        tc_body::<DT_F16, true, false, 32>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// tensor cores, f16, A m-contiguous, B n-contiguous, BM 128
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn tc_f16_mn_128(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<u16, 24576, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        tc_body::<DT_F16, true, true, 128>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// tensor cores, f16, A m-contiguous, B n-contiguous, BM 64
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn tc_f16_mn_64(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<u16, 18432, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        tc_body::<DT_F16, true, true, 64>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// tensor cores, f16, A m-contiguous, B n-contiguous, BM 32
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn tc_f16_mn_32(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<u16, 15360, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        tc_body::<DT_F16, true, true, 32>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// tensor cores, bf16, A k-contiguous, B k-contiguous, BM 128
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn tc_bf16_kk_128(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<u16, 24576, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        tc_body::<DT_BF16, false, false, 128>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// tensor cores, bf16, A k-contiguous, B k-contiguous, BM 64
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn tc_bf16_kk_64(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<u16, 18432, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        tc_body::<DT_BF16, false, false, 64>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// tensor cores, bf16, A k-contiguous, B k-contiguous, BM 32
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn tc_bf16_kk_32(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<u16, 15360, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        tc_body::<DT_BF16, false, false, 32>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// tensor cores, bf16, A k-contiguous, B n-contiguous, BM 128
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn tc_bf16_kn_128(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<u16, 24576, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        tc_body::<DT_BF16, false, true, 128>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// tensor cores, bf16, A k-contiguous, B n-contiguous, BM 64
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn tc_bf16_kn_64(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<u16, 18432, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        tc_body::<DT_BF16, false, true, 64>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// tensor cores, bf16, A k-contiguous, B n-contiguous, BM 32
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn tc_bf16_kn_32(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<u16, 15360, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        tc_body::<DT_BF16, false, true, 32>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// tensor cores, bf16, A m-contiguous, B k-contiguous, BM 128
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn tc_bf16_mk_128(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<u16, 24576, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        tc_body::<DT_BF16, true, false, 128>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// tensor cores, bf16, A m-contiguous, B k-contiguous, BM 64
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn tc_bf16_mk_64(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<u16, 18432, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        tc_body::<DT_BF16, true, false, 64>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// tensor cores, bf16, A m-contiguous, B k-contiguous, BM 32
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn tc_bf16_mk_32(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<u16, 15360, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        tc_body::<DT_BF16, true, false, 32>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// tensor cores, bf16, A m-contiguous, B n-contiguous, BM 128
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn tc_bf16_mn_128(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<u16, 24576, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        tc_body::<DT_BF16, true, true, 128>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// tensor cores, bf16, A m-contiguous, B n-contiguous, BM 64
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn tc_bf16_mn_64(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<u16, 18432, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        tc_body::<DT_BF16, true, true, 64>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// tensor cores, bf16, A m-contiguous, B n-contiguous, BM 32
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn tc_bf16_mn_32(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<u16, 15360, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        tc_body::<DT_BF16, true, true, 32>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// SIMT 128 x 128, f32, thread mapping for A k-contiguous, B k-contiguous
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn simt_f32_kk_128(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 4224, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        simt_body::<DT_F32, false, false, 128>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// SIMT 64 x 64, f32, thread mapping for A k-contiguous, B k-contiguous
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn simt_f32_kk_64(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 2176, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        simt_body::<DT_F32, false, false, 64>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// SIMT 128 x 128, f32, thread mapping for A k-contiguous, B n-contiguous
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn simt_f32_kn_128(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 4224, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        simt_body::<DT_F32, false, true, 128>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// SIMT 64 x 64, f32, thread mapping for A k-contiguous, B n-contiguous
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn simt_f32_kn_64(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 2176, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        simt_body::<DT_F32, false, true, 64>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// SIMT 128 x 128, f32, thread mapping for A m-contiguous, B k-contiguous
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn simt_f32_mk_128(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 4224, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        simt_body::<DT_F32, true, false, 128>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// SIMT 64 x 64, f32, thread mapping for A m-contiguous, B k-contiguous
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn simt_f32_mk_64(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 2176, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        simt_body::<DT_F32, true, false, 64>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// SIMT 128 x 128, f32, thread mapping for A m-contiguous, B n-contiguous
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn simt_f32_mn_128(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 4224, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        simt_body::<DT_F32, true, true, 128>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// SIMT 64 x 64, f32, thread mapping for A m-contiguous, B n-contiguous
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn simt_f32_mn_64(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 2176, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        simt_body::<DT_F32, true, true, 64>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// SIMT 128 x 128, f16, thread mapping for A k-contiguous, B k-contiguous
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn simt_f16_kk_128(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 4224, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        simt_body::<DT_F16, false, false, 128>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// SIMT 64 x 64, f16, thread mapping for A k-contiguous, B k-contiguous
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn simt_f16_kk_64(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 2176, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        simt_body::<DT_F16, false, false, 64>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// SIMT 128 x 128, f16, thread mapping for A k-contiguous, B n-contiguous
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn simt_f16_kn_128(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 4224, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        simt_body::<DT_F16, false, true, 128>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// SIMT 64 x 64, f16, thread mapping for A k-contiguous, B n-contiguous
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn simt_f16_kn_64(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 2176, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        simt_body::<DT_F16, false, true, 64>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// SIMT 128 x 128, f16, thread mapping for A m-contiguous, B k-contiguous
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn simt_f16_mk_128(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 4224, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        simt_body::<DT_F16, true, false, 128>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// SIMT 64 x 64, f16, thread mapping for A m-contiguous, B k-contiguous
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn simt_f16_mk_64(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 2176, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        simt_body::<DT_F16, true, false, 64>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// SIMT 128 x 128, f16, thread mapping for A m-contiguous, B n-contiguous
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn simt_f16_mn_128(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 4224, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        simt_body::<DT_F16, true, true, 128>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// SIMT 64 x 64, f16, thread mapping for A m-contiguous, B n-contiguous
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn simt_f16_mn_64(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 2176, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        simt_body::<DT_F16, true, true, 64>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// SIMT 128 x 128, bf16, thread mapping for A k-contiguous, B k-contiguous
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn simt_bf16_kk_128(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 4224, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        simt_body::<DT_BF16, false, false, 128>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// SIMT 64 x 64, bf16, thread mapping for A k-contiguous, B k-contiguous
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn simt_bf16_kk_64(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 2176, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        simt_body::<DT_BF16, false, false, 64>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// SIMT 128 x 128, bf16, thread mapping for A k-contiguous, B n-contiguous
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn simt_bf16_kn_128(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 4224, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        simt_body::<DT_BF16, false, true, 128>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// SIMT 64 x 64, bf16, thread mapping for A k-contiguous, B n-contiguous
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn simt_bf16_kn_64(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 2176, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        simt_body::<DT_BF16, false, true, 64>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// SIMT 128 x 128, bf16, thread mapping for A m-contiguous, B k-contiguous
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn simt_bf16_mk_128(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 4224, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        simt_body::<DT_BF16, true, false, 128>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// SIMT 64 x 64, bf16, thread mapping for A m-contiguous, B k-contiguous
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn simt_bf16_mk_64(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 2176, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        simt_body::<DT_BF16, true, false, 64>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// SIMT 128 x 128, bf16, thread mapping for A m-contiguous, B n-contiguous
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn simt_bf16_mn_128(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 4224, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        simt_body::<DT_BF16, true, true, 128>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// SIMT 64 x 64, bf16, thread mapping for A m-contiguous, B n-contiguous
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn simt_bf16_mn_64(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 2176, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        simt_body::<DT_BF16, true, true, 64>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta, ws, nsplit);
    }

    /// GEMV, f32, m <= 1, B k-contiguous, a warp per column, 16-byte loads
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_k1_f32_v(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        gemv_k_body::<DT_F32, 1, true, 8>(0, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f32, m <= 1, B k-contiguous, a block per column (8 warps split k), 16-byte loads
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_ks1_f32_v(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 64, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_k_body::<DT_F32, 1, true, 1>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f32, m <= 1, B k-contiguous, a warp per column
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_k1_f32(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        gemv_k_body::<DT_F32, 1, false, 8>(0, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f32, m <= 1, B k-contiguous, a block per column (8 warps split k)
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_ks1_f32(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 64, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_k_body::<DT_F32, 1, false, 1>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f32, m <= 1, B n-contiguous
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_n1_f32(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 2048, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_n_body::<DT_F32, 1>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f32, m <= 2, B k-contiguous, a warp per column, 16-byte loads
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_k2_f32_v(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        gemv_k_body::<DT_F32, 2, true, 8>(0, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f32, m <= 2, B k-contiguous, a block per column (8 warps split k), 16-byte loads
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_ks2_f32_v(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 64, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_k_body::<DT_F32, 2, true, 1>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f32, m <= 2, B k-contiguous, a warp per column
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_k2_f32(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        gemv_k_body::<DT_F32, 2, false, 8>(0, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f32, m <= 2, B k-contiguous, a block per column (8 warps split k)
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_ks2_f32(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 64, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_k_body::<DT_F32, 2, false, 1>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f32, m <= 2, B n-contiguous
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_n2_f32(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 2048, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_n_body::<DT_F32, 2>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f32, m <= 4, B k-contiguous, a warp per column, 16-byte loads
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_k4_f32_v(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        gemv_k_body::<DT_F32, 4, true, 8>(0, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f32, m <= 4, B k-contiguous, a block per column (8 warps split k), 16-byte loads
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_ks4_f32_v(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 64, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_k_body::<DT_F32, 4, true, 1>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f32, m <= 4, B k-contiguous, a warp per column
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_k4_f32(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        gemv_k_body::<DT_F32, 4, false, 8>(0, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f32, m <= 4, B k-contiguous, a block per column (8 warps split k)
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_ks4_f32(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 64, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_k_body::<DT_F32, 4, false, 1>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f32, m <= 4, B n-contiguous
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_n4_f32(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 2048, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_n_body::<DT_F32, 4>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f32, m <= 8, B k-contiguous, a warp per column, 16-byte loads
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_k8_f32_v(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        gemv_k_body::<DT_F32, 8, true, 8>(0, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f32, m <= 8, B k-contiguous, a block per column (8 warps split k), 16-byte loads
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_ks8_f32_v(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 64, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_k_body::<DT_F32, 8, true, 1>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f32, m <= 8, B k-contiguous, a warp per column
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_k8_f32(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        gemv_k_body::<DT_F32, 8, false, 8>(0, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f32, m <= 8, B k-contiguous, a block per column (8 warps split k)
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_ks8_f32(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 64, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_k_body::<DT_F32, 8, false, 1>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f32, m <= 8, B n-contiguous
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_n8_f32(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 2048, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_n_body::<DT_F32, 8>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f16, m <= 1, B k-contiguous, a warp per column, 16-byte loads
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_k1_f16_v(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        gemv_k_body::<DT_F16, 1, true, 8>(0, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f16, m <= 1, B k-contiguous, a block per column (8 warps split k), 16-byte loads
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_ks1_f16_v(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 64, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_k_body::<DT_F16, 1, true, 1>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f16, m <= 1, B k-contiguous, a warp per column
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_k1_f16(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        gemv_k_body::<DT_F16, 1, false, 8>(0, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f16, m <= 1, B k-contiguous, a block per column (8 warps split k)
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_ks1_f16(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 64, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_k_body::<DT_F16, 1, false, 1>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f16, m <= 1, B n-contiguous
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_n1_f16(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 2048, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_n_body::<DT_F16, 1>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f16, m <= 2, B k-contiguous, a warp per column, 16-byte loads
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_k2_f16_v(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        gemv_k_body::<DT_F16, 2, true, 8>(0, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f16, m <= 2, B k-contiguous, a block per column (8 warps split k), 16-byte loads
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_ks2_f16_v(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 64, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_k_body::<DT_F16, 2, true, 1>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f16, m <= 2, B k-contiguous, a warp per column
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_k2_f16(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        gemv_k_body::<DT_F16, 2, false, 8>(0, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f16, m <= 2, B k-contiguous, a block per column (8 warps split k)
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_ks2_f16(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 64, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_k_body::<DT_F16, 2, false, 1>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f16, m <= 2, B n-contiguous
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_n2_f16(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 2048, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_n_body::<DT_F16, 2>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f16, m <= 4, B k-contiguous, a warp per column, 16-byte loads
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_k4_f16_v(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        gemv_k_body::<DT_F16, 4, true, 8>(0, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f16, m <= 4, B k-contiguous, a block per column (8 warps split k), 16-byte loads
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_ks4_f16_v(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 64, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_k_body::<DT_F16, 4, true, 1>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f16, m <= 4, B k-contiguous, a warp per column
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_k4_f16(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        gemv_k_body::<DT_F16, 4, false, 8>(0, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f16, m <= 4, B k-contiguous, a block per column (8 warps split k)
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_ks4_f16(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 64, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_k_body::<DT_F16, 4, false, 1>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f16, m <= 4, B n-contiguous
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_n4_f16(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 2048, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_n_body::<DT_F16, 4>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f16, m <= 8, B k-contiguous, a warp per column, 16-byte loads
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_k8_f16_v(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        gemv_k_body::<DT_F16, 8, true, 8>(0, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f16, m <= 8, B k-contiguous, a block per column (8 warps split k), 16-byte loads
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_ks8_f16_v(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 64, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_k_body::<DT_F16, 8, true, 1>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f16, m <= 8, B k-contiguous, a warp per column
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_k8_f16(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        gemv_k_body::<DT_F16, 8, false, 8>(0, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f16, m <= 8, B k-contiguous, a block per column (8 warps split k)
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_ks8_f16(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 64, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_k_body::<DT_F16, 8, false, 1>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, f16, m <= 8, B n-contiguous
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_n8_f16(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 2048, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_n_body::<DT_F16, 8>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, bf16, m <= 1, B k-contiguous, a warp per column, 16-byte loads
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_k1_bf16_v(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        gemv_k_body::<DT_BF16, 1, true, 8>(0, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, bf16, m <= 1, B k-contiguous, a block per column (8 warps split k), 16-byte loads
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_ks1_bf16_v(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 64, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_k_body::<DT_BF16, 1, true, 1>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, bf16, m <= 1, B k-contiguous, a warp per column
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_k1_bf16(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        gemv_k_body::<DT_BF16, 1, false, 8>(0, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, bf16, m <= 1, B k-contiguous, a block per column (8 warps split k)
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_ks1_bf16(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 64, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_k_body::<DT_BF16, 1, false, 1>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, bf16, m <= 1, B n-contiguous
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_n1_bf16(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 2048, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_n_body::<DT_BF16, 1>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, bf16, m <= 2, B k-contiguous, a warp per column, 16-byte loads
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_k2_bf16_v(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        gemv_k_body::<DT_BF16, 2, true, 8>(0, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, bf16, m <= 2, B k-contiguous, a block per column (8 warps split k), 16-byte loads
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_ks2_bf16_v(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 64, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_k_body::<DT_BF16, 2, true, 1>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, bf16, m <= 2, B k-contiguous, a warp per column
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_k2_bf16(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        gemv_k_body::<DT_BF16, 2, false, 8>(0, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, bf16, m <= 2, B k-contiguous, a block per column (8 warps split k)
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_ks2_bf16(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 64, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_k_body::<DT_BF16, 2, false, 1>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, bf16, m <= 2, B n-contiguous
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_n2_bf16(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 2048, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_n_body::<DT_BF16, 2>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, bf16, m <= 4, B k-contiguous, a warp per column, 16-byte loads
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_k4_bf16_v(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        gemv_k_body::<DT_BF16, 4, true, 8>(0, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, bf16, m <= 4, B k-contiguous, a block per column (8 warps split k), 16-byte loads
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_ks4_bf16_v(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 64, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_k_body::<DT_BF16, 4, true, 1>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, bf16, m <= 4, B k-contiguous, a warp per column
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_k4_bf16(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        gemv_k_body::<DT_BF16, 4, false, 8>(0, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, bf16, m <= 4, B k-contiguous, a block per column (8 warps split k)
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_ks4_bf16(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 64, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_k_body::<DT_BF16, 4, false, 1>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, bf16, m <= 4, B n-contiguous
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_n4_bf16(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 2048, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_n_body::<DT_BF16, 4>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, bf16, m <= 8, B k-contiguous, a warp per column, 16-byte loads
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_k8_bf16_v(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        gemv_k_body::<DT_BF16, 8, true, 8>(0, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, bf16, m <= 8, B k-contiguous, a block per column (8 warps split k), 16-byte loads
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_ks8_bf16_v(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 64, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_k_body::<DT_BF16, 8, true, 1>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, bf16, m <= 8, B k-contiguous, a warp per column
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_k8_bf16(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        gemv_k_body::<DT_BF16, 8, false, 8>(0, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, bf16, m <= 8, B k-contiguous, a block per column (8 warps split k)
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_ks8_bf16(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 64, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_k_body::<DT_BF16, 8, false, 1>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// GEMV, bf16, m <= 8, B n-contiguous
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemv_n8_bf16(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        static mut SMEM: SharedArray<f32, 2048, 16> = SharedArray::UNINIT;
        let sm = cuda_device::shared::cvta_generic_to_shared_u32(SharedArray::as_raw_mut_ptr(&raw mut SMEM) as *const u8);
        gemv_n_body::<DT_BF16, 8>(sm, a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// f64-accumulating reference, f32 (diagnostics)
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemm_ref_f32(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        ref_body::<DT_F32>(a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// f64-accumulating reference, f16 (diagnostics)
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemm_ref_f16(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        ref_body::<DT_F16>(a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// f64-accumulating reference, bf16 (diagnostics)
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn gemm_ref_bf16(a: *const u8, b: *const u8, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, k: i32, a_s0: i64, a_s1: i64, b_s0: i64, b_s1: i64, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sa: i64, sb: i64, sd: i64, sc: i64, alpha: f32, beta: f32, ws: *mut f32, nsplit: i32) {
        ref_body::<DT_BF16>(a, b, d, c, bias, m, n, k, a_s0, a_s1, b_s0, b_s1, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sa, sb, sd, sc, alpha, beta);
    }

    /// split-K merge + epilogue, f32
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn splitk_reduce_f32(ws: *const f32, nsplit: i32, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sd: i64, sc: i64, alpha: f32, beta: f32) {
        reduce_body::<DT_F32>(ws, nsplit, d, c, bias, m, n, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sd, sc, alpha, beta);
    }

    /// split-K merge + epilogue, f16
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn splitk_reduce_f16(ws: *const f32, nsplit: i32, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sd: i64, sc: i64, alpha: f32, beta: f32) {
        reduce_body::<DT_F16>(ws, nsplit, d, c, bias, m, n, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sd, sc, alpha, beta);
    }

    /// split-K merge + epilogue, bf16
    #[kernel]
    #[launch_bounds(256, 1)]
    pub unsafe fn splitk_reduce_bf16(ws: *const f32, nsplit: i32, d: *mut u8, c: *const u8, bias: *const u8, m: i32, n: i32, d_s0: i64, d_s1: i64, c_s0: i64, c_s1: i64, bias_s0: i64, bias_s1: i64, sd: i64, sc: i64, alpha: f32, beta: f32) {
        reduce_body::<DT_BF16>(ws, nsplit, d, c, bias, m, n, d_s0, d_s1, c_s0, c_s1, bias_s0, bias_s1, sd, sc, alpha, beta);
    }
    // END GENERATED

    /// Philox uniform (0, 1] f32: thread t writes elements 4t .. 4t + 3 from counter offset + t; grid ceil(n / 1024),
    /// block 256.
    #[kernel]
    pub unsafe fn philox_uniform_f32(out: *mut f32, n: u64, seed: u64, offset: u64) {
        let t = thread::blockIdx_x() as u64 * 256 + thread::threadIdx_x() as u64;
        let base = 4 * t;
        if base < n {
            let (x0, x1, x2, x3) = philox(offset + t, seed);
            let v = [u01f(x0), u01f(x1), u01f(x2), u01f(x3)];
            let mut e = 0;
            while e < 4 {
                unroll!();
                if base + e < n {
                    st_u32(out as u64 + (base + e) * 4, v[e as usize].to_bits());
                }
                e += 1;
            }
        }
    }

    /// Philox Box-Muller normal f32 (mean + std * N(0, 1)): thread t writes elements 4t .. 4t + 3 (two pairs).
    #[kernel]
    pub unsafe fn philox_normal_f32(out: *mut f32, n: u64, seed: u64, offset: u64, mean: f32, std: f32) {
        let t = thread::blockIdx_x() as u64 * 256 + thread::threadIdx_x() as u64;
        let base = 4 * t;
        if base < n {
            let (x0, x1, x2, x3) = philox(offset + t, seed);
            let r0 = (-2.0f32 * u01f(x0).ln()).sqrt();
            let a0 = TWO_PI_F * u01f(x1);
            let r1 = (-2.0f32 * u01f(x2).ln()).sqrt();
            let a1 = TWO_PI_F * u01f(x3);
            let v = [r0 * a0.cos(), r0 * a0.sin(), r1 * a1.cos(), r1 * a1.sin()];
            let mut e = 0;
            while e < 4 {
                unroll!();
                if base + e < n {
                    st_u32(out as u64 + (base + e) * 4, fma(std, v[e as usize], mean).to_bits());
                }
                e += 1;
            }
        }
    }

    /// Philox uniform (0, 1] f64: thread t writes elements 2t, 2t + 1 (53 bits each).
    #[kernel]
    pub unsafe fn philox_uniform_f64(out: *mut f64, n: u64, seed: u64, offset: u64) {
        let t = thread::blockIdx_x() as u64 * 256 + thread::threadIdx_x() as u64;
        let base = 2 * t;
        if base < n {
            let (x0, x1, x2, x3) = philox(offset + t, seed);
            st_u64(out as u64 + base * 8, u01d(x0, x1).to_bits());
            if base + 1 < n {
                st_u64(out as u64 + (base + 1) * 8, u01d(x2, x3).to_bits());
            }
        }
    }

    /// Philox Box-Muller normal f64: thread t writes elements 2t, 2t + 1 (one pair).
    #[kernel]
    pub unsafe fn philox_normal_f64(out: *mut f64, n: u64, seed: u64, offset: u64, mean: f64, std: f64) {
        let t = thread::blockIdx_x() as u64 * 256 + thread::threadIdx_x() as u64;
        let base = 2 * t;
        if base < n {
            let (x0, x1, x2, x3) = philox(offset + t, seed);
            let r = (-2.0f64 * u01d(x0, x1).ln()).sqrt();
            let a = TWO_PI_D * u01d(x2, x3);
            st_u64(out as u64 + base * 8, (mean + std * (r * a.cos())).to_bits());
            if base + 1 < n {
                st_u64(out as u64 + (base + 1) * 8, (mean + std * (r * a.sin())).to_bits());
            }
        }
    }
}

fn main() {
    std::process::exit(if gate::run() { 0 } else { 1 });
}
