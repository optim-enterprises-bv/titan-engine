//! mistralrs-quant Marlin kernels (marlin_matmul_{f16,bf16,awq_f16,awq_bf16}.cu, marlin_repack.cu) in
//! cuda-oxide: a separate crate only because their 198 instances take ~20 minutes of device
//! codegen. The PTX (`marlin_kernels.ptx`) is loaded by `mistralrs-quant-b`'s launchers.
#![allow(non_snake_case, clippy::missing_safety_doc, clippy::too_many_arguments, unused_unsafe)]

use cuda_device::{kernel, ptx_asm, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(always)]
    pub fn add(a: f32, b: f32) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("add.rn.ftz.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn h2f(h: u16) -> f32 {
        let r: f32;
        unsafe { ptx_asm!("cvt.f32.f16 %0, %1;", out("=f") r, in("h") h, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn bf2f(h: u16) -> f32 {
        f32::from_bits((h as u32) << 16)
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
    #[inline(always)]
    pub fn hfma2(a: u32, b: u32, c: u32) -> u32 {
        let r: u32;
        unsafe { ptx_asm!("fma.rn.f16x2 %0, %1, %2, %3;", out("=r") r, in("r") a, in("r") b, in("r") c, options(register_only)); }
        r
    }
    #[inline(always)]
    pub fn hmul2(a: u32, b: u32) -> u32 {
        let r: u32;
        unsafe { ptx_asm!("mul.rn.f16x2 %0, %1, %2;", out("=r") r, in("r") a, in("r") b, options(register_only)); }
        r
    }
    /// f16 element (bit pattern).
    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct H(pub u16);
    /// bf16 element (bit pattern).
    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct B(pub u16);

    // ------------------------------------------------------------------ Marlin (vLLM marlin_kernel.cuh)
    #[inline(always)]
    pub unsafe fn cp_async4(smem: u32, glob: *const u8) {
        ptx_asm!("cp.async.cg.shared.global [%0], [%1], 16;", in("r") smem, in("l") glob as u64, clobber("memory"));
    }
    #[inline(always)]
    pub unsafe fn cp_async4_pred(smem: u32, glob: *const u8, pred: bool) {
        let p = pred as u32;
        ptx_asm!("{ .reg .pred p; setp.ne.b32 p, %0, 0; @p cp.async.cg.shared.global [%1], [%2], 16; }", in("r") p, in("r") smem, in("l") glob as u64, clobber("memory"));
    }
    #[inline(always)]
    pub unsafe fn cp_async_fence() {
        ptx_asm!("cp.async.commit_group;", clobber("memory"));
    }
    #[inline(always)]
    pub unsafe fn cp_async_wait2() {
        ptx_asm!("cp.async.wait_group 2;", clobber("memory"));
    }
    #[inline(always)]
    pub unsafe fn cp_async_wait0() {
        ptx_asm!("cp.async.wait_group 0;", clobber("memory"));
    }
    #[inline(always)]
    pub unsafe fn ldsm4(smem: u32) -> [u32; 4] {
        let (a, b, c, d): (u32, u32, u32, u32);
        ptx_asm!("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];", out("=r") a, out("=r") b, out("=r") c, out("=r") d, in("r") smem);
        [a, b, c, d]
    }

    /// Element type of the Marlin kernels (f16 or bf16) and its exact ops.
    pub trait Mt: Copy {
        /// `dequant<scalar_t, kU4B8 | kU4>(q)`: AWQ (`ZP`) uses the unbiased constants.
        fn dequant<const ZP: bool>(q: i32) -> [u32; 2];
        fn sub2(a: u32, b: u32) -> u32;
        fn mul2(a: u32, b: u32) -> u32;
        fn f2n(x: f32) -> u16;
        fn n2f(x: u16) -> f32;
        fn mma(c: &mut [f32; 4], a: &[u32; 4], b: &[u32; 2]);
    }
    impl Mt for H {
        #[inline(always)]
        fn dequant<const ZP: bool>(q: i32) -> [u32; 2] {
            let q = q as u32;
            let lo = (q & 0x000f_000f) | 0x6400_6400;
            let hi = (q & 0x00f0_00f0) | 0x6400_6400;
            let (sub, add) = if ZP { (0x6400_6400u32, 0xd400_d400u32) } else { (0x6408_6408u32, 0xd480_d480u32) };
            [Self::sub2(lo, sub), hfma2(hi, 0x2c00_2c00, add)]
        }
        #[inline(always)]
        fn sub2(a: u32, b: u32) -> u32 {
            let r: u32;
            unsafe { ptx_asm!("sub.rn.f16x2 %0, %1, %2;", out("=r") r, in("r") a, in("r") b, options(register_only)); }
            r
        }
        #[inline(always)]
        fn mul2(a: u32, b: u32) -> u32 {
            hmul2(a, b)
        }
        #[inline(always)]
        fn f2n(x: f32) -> u16 {
            f2h(x)
        }
        #[inline(always)]
        fn n2f(x: u16) -> f32 {
            h2f(x)
        }
        #[inline(always)]
        fn mma(c: &mut [f32; 4], a: &[u32; 4], b: &[u32; 2]) {
            unsafe {
                ptx_asm!("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};",
                    inout("+f") c[0], inout("+f") c[1], inout("+f") c[2], inout("+f") c[3],
                    in("r") a[0], in("r") a[1], in("r") a[2], in("r") a[3], in("r") b[0], in("r") b[1]);
            }
        }
    }
    #[inline(always)]
    pub fn bfma2(a: u32, b: u32, c: u32) -> u32 {
        let r: u32;
        unsafe { ptx_asm!("fma.rn.bf16x2 %0, %1, %2, %3;", out("=r") r, in("r") a, in("r") b, in("r") c, options(register_only)); }
        r
    }
    impl Mt for B {
        #[inline(always)]
        fn dequant<const ZP: bool>(q: i32) -> [u32; 2] {
            let lo = ((q as u32) & 0x000f_000f) | 0x4300_4300;
            let q = q >> 4;
            let hi = ((q as u32) & 0x000f_000f) | 0x4300_4300;
            let add = if ZP { 0xC300_C300u32 } else { 0xC308_C308u32 };
            [bfma2(lo, 0x3F80_3F80, add), bfma2(hi, 0x3F80_3F80, add)]
        }
        #[inline(always)]
        fn sub2(a: u32, b: u32) -> u32 {
            let r: u32;
            unsafe { ptx_asm!("sub.rn.bf16x2 %0, %1, %2;", out("=r") r, in("r") a, in("r") b, options(register_only)); }
            r
        }
        #[inline(always)]
        fn mul2(a: u32, b: u32) -> u32 {
            let r: u32;
            unsafe { ptx_asm!("mul.rn.bf16x2 %0, %1, %2;", out("=r") r, in("r") a, in("r") b, options(register_only)); }
            r
        }
        #[inline(always)]
        fn f2n(x: f32) -> u16 {
            f2bf(x)
        }
        #[inline(always)]
        fn n2f(x: u16) -> f32 {
            bf2f(x)
        }
        #[inline(always)]
        fn mma(c: &mut [f32; 4], a: &[u32; 4], b: &[u32; 2]) {
            unsafe {
                ptx_asm!("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};",
                    inout("+f") c[0], inout("+f") c[1], inout("+f") c[2], inout("+f") c[3],
                    in("r") a[0], in("r") a[1], in("r") a[2], in("r") a[3], in("r") b[0], in("r") b[1]);
            }
        }
    }
    #[inline(always)]
    pub fn splat_lo(x: u32) -> u32 {
        (x & 0xffff) | (x << 16)
    }
    #[inline(always)]
    pub fn splat_hi(x: u32) -> u32 {
        (x >> 16) | (x & 0xffff_0000)
    }

    /// `Marlin<scalar_t, THREADS, MB, NB, KB, stages = 4, GB, kU4B8 | kU4, act_order = false, ZP, 4>`.
    /// Pointers in `[u32; 4]` (int4) units like the reference; `shr` is the dynamic shared base.
    #[inline(always)]
    pub unsafe fn marlin<T: Mt, const THREADS: i32, const MB: usize, const NB: i32, const KB: i32, const GB: i32, const ZP: bool>(
        a_in: *const [u32; 4], b_in: *const [u32; 4], c_in: *mut [u32; 4], scales_ptr: *const [u32; 4], zp_ptr: *const [u32; 4],
        prob_m_in: i32, prob_n: i32, prob_k: i32, num_groups: i32, locks_in: *mut i32,
    ) {
        const STAGES: i32 = 4;
        let _ = num_groups;
        let tid = thread::threadIdx_x() as i32;
        let bid = thread::blockIdx_x() as i32;
        let gdim = thread::gridDim_x() as i32;
        let mbi = MB as i32;
        let mut a_g = a_in;
        let mut c_g = c_in;
        let mut locks = locks_in;
        let mut prob_m = prob_m_in;
        let mut parallel = 1i32;
        if prob_m > 16 * mbi {
            parallel = prob_m / (16 * mbi);
            prob_m = 16 * mbi;
        }
        let k_tiles = prob_k / 16 / KB;
        let n_tiles = prob_n / 16 / NB;
        let dc = |a: i32, b: i32| (a + b - 1) / b;
        let mut iters = dc(k_tiles.wrapping_mul(n_tiles).wrapping_mul(parallel), gdim);
        if GB != -1 && GB >= KB {
            iters = (GB / KB) * dc(iters, GB / KB);
        }
        let mut slice_row = (iters.wrapping_mul(bid)) % k_tiles;
        let mut slice_col_par = (iters.wrapping_mul(bid)) / k_tiles;
        let mut slice_col = slice_col_par;
        let mut slice_iters = 0i32;
        let mut slice_count = 0i32;
        let mut slice_idx = 0i32;
        if slice_col_par >= n_tiles {
            a_g = a_g.offset(((slice_col_par / n_tiles) * 16 * mbi * prob_k / 8) as isize);
            c_g = c_g.offset(((slice_col_par / n_tiles) * 16 * mbi * prob_n / 8) as isize);
            locks = locks.offset(((slice_col_par / n_tiles) * n_tiles) as isize);
            slice_col = slice_col_par % n_tiles;
        }
        // init_slice
        macro_rules! init_slice {
            () => {{
                slice_iters = iters * (bid + 1) - (k_tiles * slice_col_par + slice_row);
                if slice_iters < 0 || slice_col_par >= n_tiles * parallel {
                    slice_iters = 0;
                }
                if slice_iters != 0 {
                    if slice_row + slice_iters > k_tiles {
                        slice_iters = k_tiles - slice_row;
                    }
                    slice_count = 1;
                    slice_idx = 0;
                    let col_first = iters * dc(k_tiles * slice_col_par, iters);
                    if col_first <= k_tiles * (slice_col_par + 1) {
                        let col_off = col_first - k_tiles * slice_col_par;
                        slice_count = dc(k_tiles - col_off, iters);
                        if col_off > 0 {
                            slice_count += 1;
                        }
                        let delta_first = iters * bid - col_first;
                        if delta_first < 0 || (col_off == 0 && delta_first == 0) {
                            slice_idx = slice_count - 1;
                        } else {
                            slice_idx = slice_count - 1 - delta_first / iters;
                            if col_off > 0 {
                                slice_idx -= 1;
                            }
                        }
                    }
                    if slice_col == n_tiles {
                        a_g = a_g.offset((16 * mbi * prob_k / 8) as isize);
                        c_g = c_g.offset((16 * mbi * prob_n / 8) as isize);
                        locks = locks.offset(n_tiles as isize);
                        slice_col = 0;
                    }
                }
            }};
        }
        init_slice!();

        // A
        let a_gl_stride = prob_k / 8;
        let a_sh_stride = 16 * KB / 8;
        let a_gl_rd_delta_o = 16 * KB / 8;
        let a_gl_rd_delta_i = a_gl_stride * (THREADS / a_gl_rd_delta_o);
        let a_sh_wr_delta = a_sh_stride * (THREADS / a_gl_rd_delta_o);
        let a_sh_rd_delta_o = 2 * ((THREADS / 32) / (NB / 4));
        let a_sh_rd_delta_i = a_sh_stride * 16;
        let a_sh_stage = a_sh_stride * (16 * mbi);
        let a_sh_wr_iters = dc(a_sh_stage, a_sh_wr_delta);
        // B
        let b_gl_stride = 16 * prob_n / (8 * 4);
        let b_sh_stride = ((NB * 16) * 16 / 8) / 4;
        let b_sh_stride_threads = b_sh_stride;
        let b_gl_rd_delta_o = b_gl_stride * KB;
        let b_gl_rd_delta_i = b_gl_stride * (THREADS / b_sh_stride_threads);
        let b_sh_wr_delta = THREADS;
        let b_sh_rd_delta = THREADS;
        let b_sh_stage = b_sh_stride * KB;
        let b_sh_wr_iters = b_sh_stage / b_sh_wr_delta; // == 2 for every instantiated config
        // scales
        let s_gl_stride = prob_n / 8;
        let s_sh_stride = 16 * NB / 8;
        let s_tb_groups = if GB != -1 && GB < KB { KB / GB } else { 1 };
        let s_sh_stage = s_tb_groups * s_sh_stride;
        let s_gl_rd_delta = s_gl_stride;
        let tb_k = 16 * KB;
        // zero points
        let zp_gl_stride = (prob_n / 8) / 4;
        let zp_sh_stride = ((16 * NB) / 8) / 4;
        let zp_tb_groups = s_tb_groups;
        let zp_sh_stage = if ZP { zp_tb_groups * zp_sh_stride } else { 0 };
        let zp_gl_rd_delta = zp_gl_stride;

        let mut a_gl_rd = a_gl_stride * (tid / a_gl_rd_delta_o) + (tid % a_gl_rd_delta_o);
        a_gl_rd += a_gl_rd_delta_o * slice_row;
        let a_sh_wr = a_sh_stride * (tid / a_gl_rd_delta_o) + (tid % a_gl_rd_delta_o);
        let mut a_sh_rd = a_sh_stride * ((tid % 32) % 16) + (tid % 32) / 16;
        a_sh_rd += 2 * ((tid / 32) / (NB / 4));
        let mut b_gl_rd = b_gl_stride * (tid / b_sh_stride_threads) + (tid % b_sh_stride_threads);
        b_gl_rd += b_sh_stride * slice_col;
        b_gl_rd += b_gl_rd_delta_o * slice_row;
        let b_sh_wr = tid;
        let b_sh_rd = tid;
        let k_iter_size = tb_k / b_sh_wr_iters;

        let mut s_gl_rd = if GB == -1 { s_sh_stride * slice_col + tid } else { s_gl_stride * ((KB * slice_row) / GB) + s_sh_stride * slice_col + tid };
        let s_sh_wr = tid;
        let s_sh_wr_pred = tid < s_sh_stride;
        let mut zp_gl_rd = 0i32;
        if ZP {
            zp_gl_rd = if GB == -1 { zp_sh_stride * slice_col + tid } else { zp_gl_stride * ((KB * slice_row) / GB) + zp_sh_stride * slice_col + tid };
        }
        let zp_sh_wr = tid;
        let zp_sh_wr_pred = tid < zp_sh_stride;
        let s_sh_rd = if GB != -1 { 8 * ((tid / 32) % (NB / 4)) + (tid % 32) / 4 } else { 8 * ((tid / 32) % (NB / 4)) + (tid % 32) % 4 };
        // num_ints_per_thread = 8 / pack_factor = 1
        let zp_sh_rd = 8 * ((tid / 32) % (NB / 4)) + ((tid % 32) / 4);

        let mut a_sh_wr_pred = [false; 8];
        let transform_a = |i: i32| {
            let row = i / a_gl_rd_delta_o;
            (a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o)) ^ row
        };
        let mut a_sh_wr_trans = [0i32; 8];
        let mut i = 0;
        while i < a_sh_wr_iters {
            a_sh_wr_pred[i as usize] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;
            a_sh_wr_trans[i as usize] = transform_a(a_sh_wr_delta * i + a_sh_wr);
            i += 1;
        }
        let mut a_sh_rd_trans = [[0i32; MB]; 2];
        let mut i = 0;
        while i < 2 {
            let mut j = 0;
            while j < MB {
                a_sh_rd_trans[i][j] = transform_a(a_sh_rd_delta_o * i as i32 + a_sh_rd_delta_i * j as i32 + a_sh_rd);
                j += 1;
            }
            i += 1;
        }
        let mut b_ptr = [b_in; 2];
        b_ptr[0] = b_in.offset(b_gl_rd as isize);
        b_ptr[1] = b_in.offset((b_gl_rd_delta_i + b_gl_rd) as isize);

        let sh = cuda_device::DynamicSharedArray::<u8, 16>::get_raw() as usize as *mut [u32; 4];
        let sh_a = sh;
        let sh_b = sh_a.offset((STAGES * a_sh_stage) as isize);
        let sh_zp = sh_b.offset((STAGES * b_sh_stage) as isize);
        let sh_s = sh_zp.offset((STAGES * zp_sh_stage) as isize);
        let sh_red = sh_s.offset((STAGES * s_sh_stage) as isize);
        let sa = |p: *const [u32; 4]| cuda_device::shared::cvta_generic_to_shared_u32(p as *const u8);

        let mut frag_a = [[[0u32; 4]; MB]; 2];
        let mut frag_b_quant = [[0u32; 4]; 2];
        let mut frag_c = [[[[0f32; 4]; 2]; 4]; MB];
        let mut frag_s = [[0u32; 4]; 2];
        let mut frag_qzp = [0i32; 2];

        macro_rules! fetch_to_shared {
            ($pipe:expr, $a_off:expr, $pred:expr) => {{
                let pipe: i32 = $pipe;
                let a_off: i32 = $a_off;
                if $pred {
                    let sh_a_stage = sh_a.offset((a_sh_stage * pipe) as isize);
                    let mut i = 0;
                    while i < a_sh_wr_iters {
                        cp_async4_pred(sa(sh_a_stage.offset(a_sh_wr_trans[i as usize] as isize)),
                            a_g.offset((a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * a_off) as isize) as *const u8, a_sh_wr_pred[i as usize]);
                        i += 1;
                    }
                    let sh_b_stage = sh_b.offset((b_sh_stage * pipe) as isize);
                    let mut i = 0;
                    while i < 2 {
                        cp_async4(sa(sh_b_stage.offset((b_sh_wr_delta * i + b_sh_wr) as isize)), b_ptr[i as usize] as *const u8);
                        b_ptr[i as usize] = b_ptr[i as usize].offset(b_gl_rd_delta_o as isize);
                        i += 1;
                    }
                    if GB != -1 {
                        let sh_s_stage = sh_s.offset((s_sh_stage * pipe) as isize);
                        if GB >= KB {
                            if s_sh_wr_pred {
                                cp_async4(sa(sh_s_stage.offset(s_sh_wr as isize)), scales_ptr.offset(s_gl_rd as isize) as *const u8);
                            }
                            if (pipe + 1) % (GB / KB) == 0 {
                                s_gl_rd += s_gl_rd_delta;
                            }
                        } else {
                            let mut i = 0;
                            while i < s_tb_groups {
                                if s_sh_wr_pred {
                                    cp_async4(sa(sh_s_stage.offset((i * s_sh_stride + s_sh_wr) as isize)), scales_ptr.offset(s_gl_rd as isize) as *const u8);
                                }
                                s_gl_rd += s_gl_rd_delta;
                                i += 1;
                            }
                        }
                    }
                    if ZP && GB != -1 {
                        let sh_zp_stage = sh_zp.offset((zp_sh_stage * pipe) as isize);
                        if GB >= KB {
                            if pipe % (GB / KB) == 0 {
                                if zp_sh_wr_pred {
                                    cp_async4(sa(sh_zp_stage.offset(zp_sh_wr as isize)), zp_ptr.offset(zp_gl_rd as isize) as *const u8);
                                }
                                zp_gl_rd += zp_gl_rd_delta;
                            }
                        } else {
                            let mut i = 0;
                            while i < zp_tb_groups {
                                if zp_sh_wr_pred {
                                    cp_async4(sa(sh_zp_stage.offset((i * zp_sh_stride + zp_sh_wr) as isize)), zp_ptr.offset(zp_gl_rd as isize) as *const u8);
                                }
                                zp_gl_rd += zp_gl_rd_delta;
                                i += 1;
                            }
                        }
                    }
                }
                cp_async_fence();
            }};
        }
        macro_rules! wait_for_stage {
            () => {{
                cp_async_wait2();
                thread::sync_threads();
            }};
        }
        macro_rules! fetch_to_registers {
            ($k:expr, $pipe:expr) => {{
                let k: i32 = $k;
                let pipe: i32 = $pipe;
                let sh_a_stage = sh_a.offset((a_sh_stage * pipe) as isize);
                let mut i = 0;
                while i < MB {
                    frag_a[(k % 2) as usize][i] = ldsm4(sa(sh_a_stage.offset(a_sh_rd_trans[(k % 2) as usize][i] as isize)));
                    i += 1;
                }
                let sh_b_stage = sh_b.offset((b_sh_stage * pipe) as isize);
                frag_b_quant[(k % 2) as usize] = *sh_b_stage.offset((b_sh_rd_delta * (k % 2) + b_sh_rd) as isize);
            }};
        }
        macro_rules! fetch_scales_to_registers {
            ($k:expr, $full_pipe:expr) => {{
                let k: i32 = $k;
                let pipe: i32 = $full_pipe % STAGES;
                if GB != -1 {
                    let sh_s_stage = sh_s.offset((s_sh_stage * pipe) as isize);
                    if GB >= KB {
                        frag_s[(k % 2) as usize] = *sh_s_stage.offset(s_sh_rd as isize);
                    } else {
                        let warp_row = (tid / 32) / (NB / 4);
                        let cur_k = warp_row * 16 + k_iter_size * (k % 2);
                        let cur_group_id = (cur_k / 16) / GB;
                        frag_s[(k % 2) as usize] = *sh_s_stage.offset((s_sh_rd + cur_group_id * s_sh_stride) as isize);
                    }
                }
            }};
        }
        macro_rules! fetch_zp_to_registers {
            ($k:expr, $full_pipe:expr) => {{
                let k: i32 = $k;
                let pipe: i32 = $full_pipe % STAGES;
                if ZP {
                    let base: *const i32 = if GB == -1 {
                        sh_zp as *const i32
                    } else if GB >= KB {
                        sh_zp.offset((zp_sh_stage * ((GB / KB) * (pipe / (GB / KB)))) as isize) as *const i32
                    } else {
                        let warp_row = (tid / 32) / (NB / 4);
                        let cur_k = warp_row * 16 + k_iter_size * (k % 2);
                        let cur_group_id = (cur_k / 16) / GB;
                        sh_zp.offset((zp_sh_stage * pipe + cur_group_id * zp_sh_stride) as isize) as *const i32
                    };
                    frag_qzp[(k % 2) as usize] = *base.offset(zp_sh_rd as isize);
                }
            }};
        }
        macro_rules! matmul {
            ($k:expr) => {{
                let kk = ($k % 2) as usize;
                let mut fzp = [0u32; 4];
                if ZP {
                    let z0 = frag_qzp[kk];
                    let z1 = z0 >> 8;
                    let d0 = T::dequant::<ZP>(z0);
                    let d1 = T::dequant::<ZP>(z1);
                    fzp = [d0[0], d0[1], d1[0], d1[1]];
                }
                let mut j = 0;
                while j < 4 {
                    let bq0 = frag_b_quant[kk][j] as i32;
                    let bq1 = bq0 >> 8;
                    let mut fb0 = T::dequant::<ZP>(bq0);
                    let mut fb1 = T::dequant::<ZP>(bq1);
                    if ZP {
                        let z = splat_lo(fzp[j]);
                        fb0 = [T::sub2(fb0[0], z), T::sub2(fb0[1], z)];
                    }
                    if GB != -1 {
                        let s = splat_lo(frag_s[kk][j]);
                        fb0 = [T::mul2(fb0[0], s), T::mul2(fb0[1], s)];
                    }
                    if ZP {
                        let z = splat_hi(fzp[j]);
                        fb1 = [T::sub2(fb1[0], z), T::sub2(fb1[1], z)];
                    }
                    if GB != -1 {
                        let s = splat_hi(frag_s[kk][j]);
                        fb1 = [T::mul2(fb1[0], s), T::mul2(fb1[1], s)];
                    }
                    let mut i = 0;
                    while i < MB {
                        T::mma(&mut frag_c[i][j][0], &frag_a[kk][i], &fb0);
                        T::mma(&mut frag_c[i][j][1], &frag_a[kk][i], &fb1);
                        i += 1;
                    }
                    j += 1;
                }
            }};
        }
        macro_rules! start_pipes {
            () => {{
                let mut i = 0i32;
                while i < STAGES - 1 {
                    if ZP && GB == -1 && i == 0 {
                        if zp_sh_wr_pred {
                            cp_async4(sa(sh_zp.offset(zp_sh_wr as isize)), zp_ptr.offset(zp_gl_rd as isize) as *const u8);
                        }
                    }
                    fetch_to_shared!(i, i, i < slice_iters);
                    i += 1;
                }
                frag_c = [[[[0f32; 4]; 2]; 4]; MB];
                wait_for_stage!();
                fetch_to_registers!(0, 0);
                fetch_scales_to_registers!(0, 0);
                fetch_zp_to_registers!(0, 0);
                a_gl_rd += a_gl_rd_delta_o * (STAGES - 1);
            }};
        }

        if slice_iters != 0 {
            start_pipes!();
        }
        while slice_iters != 0 {
            let mut pipe = 0i32;
            while pipe < STAGES {
                let mut k = 0i32;
                while k < 2 {
                    fetch_to_registers!(k + 1, pipe % STAGES);
                    fetch_scales_to_registers!(k + 1, pipe);
                    fetch_zp_to_registers!(k + 1, pipe);
                    if k == 0 {
                        fetch_to_shared!((pipe + STAGES - 1) % STAGES, pipe, slice_iters >= STAGES);
                        pipe += 1;
                        wait_for_stage!();
                    }
                    matmul!(k);
                    k += 1;
                }
                slice_iters -= 1;
                if slice_iters == 0 {
                    break;
                }
            }
            a_gl_rd += a_gl_rd_delta_o * STAGES;

            if slice_iters == 0 {
                cp_async_wait0();
                let last = slice_idx == slice_count - 1;
                if GB == -1 && last {
                    if s_sh_wr_pred {
                        cp_async4(sa(sh_s.offset(s_sh_wr as isize)), scales_ptr.offset(s_gl_rd as isize) as *const u8);
                    }
                    cp_async_fence();
                }
                // ---- thread_block_reduce
                let red_off = THREADS / b_sh_stride_threads / 2;
                if red_off >= 1 {
                    let red_idx = tid / b_sh_stride_threads;
                    let red_sh_stride = b_sh_stride_threads * 4 * 2;
                    let red_sh_delta = b_sh_stride_threads;
                    let red_sh_rd = red_sh_stride * (tid / b_sh_stride_threads) + (tid % b_sh_stride_threads);
                    let mut m_block = 0;
                    while m_block < MB {
                        let mut i = red_off;
                        while i > 0 {
                            if i <= red_idx && red_idx < 2 * i {
                                let mut j = 0;
                                while j < 8 {
                                    let red_sh_wr = red_sh_delta * j + (red_sh_rd - red_sh_stride * i);
                                    let fc = &mut frag_c[m_block][(j / 2) as usize][(j % 2) as usize];
                                    if i < red_off {
                                        let c_rd = *sh_red.offset((red_sh_delta * j + red_sh_rd) as isize);
                                        let c_wr = *sh_red.offset(red_sh_wr as isize);
                                        let mut e = 0;
                                        while e < 4 {
                                            fc[e] = add(fc[e], add(f32::from_bits(c_rd[e]), f32::from_bits(c_wr[e])));
                                            e += 1;
                                        }
                                    }
                                    *sh_red.offset(red_sh_wr as isize) = [fc[0].to_bits(), fc[1].to_bits(), fc[2].to_bits(), fc[3].to_bits()];
                                    j += 1;
                                }
                            }
                            thread::sync_threads();
                            i /= 2;
                        }
                        if red_idx == 0 {
                            let mut j = 0;
                            while j < 8 {
                                let c_rd = *sh_red.offset((red_sh_delta * j + red_sh_rd) as isize);
                                let fc = &mut frag_c[m_block][(j / 2) as usize][(j % 2) as usize];
                                let mut e = 0;
                                while e < 4 {
                                    fc[e] = add(fc[e], f32::from_bits(c_rd[e]));
                                    e += 1;
                                }
                                j += 1;
                            }
                        }
                        thread::sync_threads();
                        m_block += 1;
                    }
                }
                if GB == -1 && last {
                    cp_async_wait0();
                    thread::sync_threads();
                    if tid / 32 < NB / 4 {
                        frag_s[0] = *sh_s.offset(s_sh_rd as isize);
                        frag_s[1] = *sh_s.offset((s_sh_rd + 4) as isize);
                    }
                }
                if slice_count > 1 {
                    // barrier_acquire(&locks[slice_col], slice_idx)
                    let lock = locks.offset(slice_col as isize);
                    if tid == 0 {
                        loop {
                            let state: i32;
                            ptx_asm!("ld.global.acquire.gpu.b32 %0, [%1];", out("=r") state, in("l") lock as u64, clobber("memory"));
                            if state == slice_idx {
                                break;
                            }
                        }
                    }
                    thread::sync_threads();
                    // ---- global_reduce(first = slice_idx == 0, last)
                    let first = slice_idx == 0;
                    let active_threads = 32 * NB / 4;
                    if tid < active_threads {
                        let c_gl_stride = prob_n / 8;
                        let c_gl_wr_delta_o = 8 * c_gl_stride;
                        let c_gl_wr_delta_i = 4 * (active_threads / 32);
                        let mut c_gl_wr = c_gl_stride * ((tid % 32) / 4) + 4 * (tid / 32) + tid % 4;
                        c_gl_wr += (2 * NB) * slice_col;
                        let c_sh_wr_delta = active_threads;
                        let c_sh_wr = tid;
                        let row = (tid % 32) / 4;
                        if !first {
                            let mut i = 0;
                            while i < mbi * 4 {
                                cp_async4_pred(sa(sh_red.offset((c_sh_wr + c_sh_wr_delta * i) as isize)),
                                    c_g.offset((c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2)) as isize) as *const u8,
                                    i < (mbi - 1) * 4 || 8 * (i / 2) + row < prob_m);
                                i += 1;
                            }
                            cp_async_fence();
                            cp_async_wait0();
                        }
                        let mut i = 0;
                        while i < mbi * 4 {
                            if i < (mbi - 1) * 4 || 8 * (i / 2) + row < prob_m {
                                let m = (i / 4) as usize;
                                let e = (i % 4) as usize;
                                if !first {
                                    let c_red = *sh_red.offset((c_sh_wr + i * c_sh_wr_delta) as isize);
                                    let mut j = 0;
                                    while j < 8 {
                                        let h = (c_red[j / 2] >> (16 * (j % 2))) as u16;
                                        let (jj, jk) = (j / 2, j % 2);
                                        let v = &mut frag_c[m][jj][jk][e];
                                        *v = add(*v, T::n2f(h));
                                        j += 1;
                                    }
                                }
                                if !last {
                                    let mut c = [0u32; 4];
                                    let mut j = 0;
                                    while j < 8 {
                                        let (jj, jk) = (j / 2, j % 2);
                                        let h = T::f2n(frag_c[m][jj][jk][e]) as u32;
                                        c[j / 2] |= h << (16 * (j % 2));
                                        j += 1;
                                    }
                                    *c_g.offset((c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2)) as isize) = c;
                                }
                            }
                            i += 1;
                        }
                    }
                    // barrier_release(&locks[slice_col], last)
                    thread::sync_threads();
                    if tid == 0 {
                        if last {
                            *lock = 0;
                        } else {
                            ptx_asm!("fence.acq_rel.gpu;", clobber("memory"));
                            ptx_asm!("red.relaxed.gpu.global.add.s32 [%0], 1;", in("l") lock as u64, clobber("memory"));
                        }
                    }
                }
                if last {
                    // ---- write_result
                    let c_gl_stride = prob_n / 8;
                    let c_sh_stride = 2 * NB + 1;
                    let c_gl_wr_delta = c_gl_stride * (THREADS / (2 * NB));
                    let c_sh_rd_delta = c_sh_stride * (THREADS / (2 * NB));
                    let mut c_gl_wr = c_gl_stride * (tid / (2 * NB)) + (tid % (2 * NB));
                    c_gl_wr += (2 * NB) * slice_col;
                    let mut c_sh_wr = (4 * c_sh_stride) * ((tid % 32) / 4) + (tid % 32) % 4;
                    c_sh_wr += 32 * (tid / 32);
                    let mut c_sh_rd = c_sh_stride * (tid / (2 * NB)) + (tid % (2 * NB));
                    let c_gl_wr_end = c_gl_stride * prob_m;
                    let red2 = sh_red as *mut u32;
                    let write = |idx: i32, c0: f32, c1: f32, s: u32| {
                        let mut res = T::f2n(c0) as u32 | ((T::f2n(c1) as u32) << 16);
                        if GB == -1 {
                            res = T::mul2(res, s);
                        }
                        *red2.offset(idx as isize) = res;
                    };
                    if tid / 32 < NB / 4 {
                        let mut i = 0;
                        while i < MB {
                            let mut j = 0;
                            while j < 4 {
                                let wr = c_sh_wr + 8 * j as i32;
                                let s0 = frag_s[j / 2][2 * (j % 2)];
                                let s1 = frag_s[j / 2][2 * (j % 2) + 1];
                                let f = &frag_c[i][j];
                                write(wr, f[0][0], f[0][1], s0);
                                write(wr + (4 * c_sh_stride) * 8, f[0][2], f[0][3], s0);
                                write(wr + 4, f[1][0], f[1][1], s1);
                                write(wr + (4 * c_sh_stride) * 8 + 4, f[1][2], f[1][3], s1);
                                j += 1;
                            }
                            c_sh_wr += 16 * (4 * c_sh_stride);
                            i += 1;
                        }
                    }
                    thread::sync_threads();
                    let n_it = dc(16 * mbi, THREADS / (2 * NB));
                    let mut i = 0;
                    while i < n_it {
                        if c_gl_wr < c_gl_wr_end {
                            *c_g.offset(c_gl_wr as isize) = *sh_red.offset(c_sh_rd as isize);
                            c_gl_wr += c_gl_wr_delta;
                            c_sh_rd += c_sh_rd_delta;
                        }
                        i += 1;
                    }
                }
                slice_row = 0;
                slice_col_par += 1;
                slice_col += 1;
                init_slice!();
                if slice_iters != 0 {
                    a_gl_rd = a_gl_stride * (tid / a_gl_rd_delta_o) + (tid % a_gl_rd_delta_o);
                    let mut i = 0;
                    while i < 2 {
                        b_ptr[i] = b_ptr[i].offset((b_sh_stride - b_gl_rd_delta_o * k_tiles) as isize);
                        if slice_col == 0 {
                            b_ptr[i] = b_ptr[i].offset(-(b_gl_stride as isize));
                        }
                        i += 1;
                    }
                    s_gl_rd = s_sh_stride * slice_col + tid;
                    zp_gl_rd = zp_sh_stride * slice_col + tid;
                    start_pipes!();
                }
            }
        }
    }

    // ---- marlin_repack.cu
    #[inline(always)]
    pub unsafe fn marlin_repack<const BITS: i32, const PERM: bool, const AWQ: bool>(w: *const u32, perm: *const u32, out: *mut u32, size_k: i32, size_n: i32) {
        let pack = 32 / BITS;
        let k_tiles = size_k / 16;
        let n_tiles = size_n / 64;
        let block_k_tiles = (k_tiles + thread::gridDim_x() as i32 - 1) / thread::gridDim_x() as i32;
        let start = (thread::blockIdx_x() as i32).wrapping_mul(block_k_tiles);
        if start >= k_tiles {
            return;
        }
        let finish = if start + block_k_tiles < k_tiles { start + block_k_tiles } else { k_tiles };
        let tid = thread::threadIdx_x() as i32;
        let mask: u32 = (1u32 << BITS) - 1;
        let tile_size = 16 * 64 / pack;
        // Direct (synchronous) gather: the reference's cp.async pipeline only moves data.
        let mut kt = start;
        while kt < finish {
            let mut nt = 0;
            while nt < n_tiles {
                let warp_id = tid / 32;
                let th_id = tid % 32;
                if warp_id < 4 {
                    let tc_col = th_id / 4;
                    let tc_row = (th_id % 4) * 2;
                    let offs = [0, 1, 8, 9];
                    let cur_n = warp_id * 16 + tc_col;
                    let first_n = nt * 64;
                    let mut vals = [0u32; 8];
                    let mut i = 0;
                    while i < 4 {
                        let k_idx = tc_row + offs[i];
                        if AWQ {
                            let cur_n_packed = cur_n / pack;
                            let cur_n_pos = cur_n % pack;
                            let unp = if BITS == 4 { [0, 4, 1, 5, 2, 6, 3, 7][cur_n_pos as usize] } else { [0, 2, 1, 3][cur_n_pos as usize] };
                            let row = w.offset((kt * 16 + k_idx).wrapping_mul(size_n / pack) as isize);
                            let s0 = *row.offset((first_n / pack + cur_n_packed) as isize) as i32;
                            let s1 = *row.offset((first_n / pack + cur_n_packed + 8 / pack) as isize) as i32;
                            vals[i] = ((s0 >> (unp * BITS)) as u32) & mask;
                            vals[4 + i] = ((s1 >> (unp * BITS)) as u32) & mask;
                        } else if PERM {
                            let src_k = *perm.offset((kt * 16 + k_idx) as isize);
                            let src_k_pos = src_k % pack as u32;
                            let row = w.offset(((src_k as i32) / pack).wrapping_mul(size_n) as isize);
                            let b1 = *row.offset((first_n + cur_n) as isize);
                            let b2 = *row.offset((first_n + cur_n + 8) as isize);
                            vals[i] = (b1 >> (src_k_pos * BITS as u32)) & mask;
                            vals[4 + i] = (b2 >> (src_k_pos * BITS as u32)) & mask;
                        } else {
                            let cur_int = k_idx / pack;
                            let cur_pos = k_idx % pack;
                            let row = w.offset((kt * 16 / pack + cur_int).wrapping_mul(size_n) as isize);
                            vals[i] = (*row.offset((first_n + cur_n) as isize) >> (cur_pos * BITS)) & mask;
                            vals[4 + i] = (*row.offset((first_n + cur_n + 8) as isize) >> (cur_pos * BITS)) & mask;
                        }
                        i += 1;
                    }
                    let out_offset = (kt * n_tiles + nt) * tile_size;
                    if BITS == 4 {
                        let pidx = [0, 2, 4, 6, 1, 3, 5, 7];
                        let mut res = 0u32;
                        let mut i = 0;
                        while i < 8 {
                            res |= vals[pidx[i]] << (i * 4);
                            i += 1;
                        }
                        *out.offset((out_offset + th_id * 4 + warp_id) as isize) = res;
                    } else {
                        let pidx = [0, 2, 1, 3];
                        let (mut r1, mut r2) = (0u32, 0u32);
                        let mut i = 0;
                        while i < 4 {
                            r1 |= vals[pidx[i]] << (i * 8);
                            r2 |= vals[4 + pidx[i]] << (i * 8);
                            i += 1;
                        }
                        *out.offset((out_offset + th_id * 8 + warp_id * 2) as isize) = r1;
                        *out.offset((out_offset + th_id * 8 + warp_id * 2 + 1) as isize) = r2;
                    }
                }
                nt += 1;
            }
            kt += 1;
        }
    }


    // GENERATED KERNELS BEGIN
    #[kernel] pub unsafe fn marlin_gptq_f16_t256_m1_n8_k8_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 1, 8, 8, -1, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t256_m1_n8_k8_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 1, 8, 8, 4, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t256_m1_n8_k8_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 1, 8, 8, 8, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t256_m2_n8_k8_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 2, 8, 8, -1, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t256_m2_n8_k8_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 2, 8, 8, 4, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t256_m2_n8_k8_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 2, 8, 8, 8, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t256_m3_n8_k8_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 3, 8, 8, -1, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t256_m3_n8_k8_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 3, 8, 8, 4, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t256_m3_n8_k8_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 3, 8, 8, 8, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t256_m4_n8_k8_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 4, 8, 8, -1, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t256_m4_n8_k8_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 4, 8, 8, 4, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t256_m4_n8_k8_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 4, 8, 8, 8, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t256_m1_n16_k4_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 1, 16, 4, -1, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t256_m1_n16_k4_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 1, 16, 4, 4, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t256_m1_n16_k4_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 1, 16, 4, 8, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t256_m2_n16_k4_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 2, 16, 4, -1, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t256_m2_n16_k4_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 2, 16, 4, 4, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t256_m2_n16_k4_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 2, 16, 4, 8, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t256_m3_n16_k4_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 3, 16, 4, -1, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t256_m3_n16_k4_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 3, 16, 4, 4, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t256_m3_n16_k4_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 3, 16, 4, 8, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t256_m4_n16_k4_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 4, 16, 4, -1, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t256_m4_n16_k4_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 4, 16, 4, 4, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t256_m4_n16_k4_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 4, 16, 4, 8, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t128_m1_n8_k4_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 1, 8, 4, -1, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t128_m1_n8_k4_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 1, 8, 4, 4, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t128_m1_n8_k4_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 1, 8, 4, 8, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t128_m2_n8_k4_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 2, 8, 4, -1, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t128_m2_n8_k4_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 2, 8, 4, 4, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t128_m2_n8_k4_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 2, 8, 4, 8, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t128_m3_n8_k4_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 3, 8, 4, -1, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t128_m3_n8_k4_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 3, 8, 4, 4, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t128_m3_n8_k4_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 3, 8, 4, 8, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t128_m4_n8_k4_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 4, 8, 4, -1, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t128_m4_n8_k4_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 4, 8, 4, 4, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t128_m4_n8_k4_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 4, 8, 4, 8, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t128_m1_n4_k8_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 1, 4, 8, -1, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t128_m1_n4_k8_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 1, 4, 8, 4, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t128_m1_n4_k8_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 1, 4, 8, 8, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t128_m2_n4_k8_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 2, 4, 8, -1, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t128_m2_n4_k8_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 2, 4, 8, 4, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t128_m2_n4_k8_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 2, 4, 8, 8, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t128_m3_n4_k8_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 3, 4, 8, -1, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t128_m3_n4_k8_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 3, 4, 8, 4, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t128_m3_n4_k8_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 3, 4, 8, 8, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t128_m4_n4_k8_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 4, 4, 8, -1, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t128_m4_n4_k8_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 4, 4, 8, 4, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_f16_t128_m4_n4_k8_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 4, 4, 8, 8, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t256_m1_n8_k8_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 1, 8, 8, -1, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t256_m1_n8_k8_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 1, 8, 8, 4, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t256_m1_n8_k8_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 1, 8, 8, 8, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t256_m2_n8_k8_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 2, 8, 8, -1, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t256_m2_n8_k8_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 2, 8, 8, 4, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t256_m2_n8_k8_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 2, 8, 8, 8, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t256_m3_n8_k8_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 3, 8, 8, -1, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t256_m3_n8_k8_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 3, 8, 8, 4, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t256_m3_n8_k8_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 3, 8, 8, 8, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t256_m4_n8_k8_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 4, 8, 8, -1, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t256_m4_n8_k8_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 4, 8, 8, 4, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t256_m4_n8_k8_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 4, 8, 8, 8, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t256_m1_n16_k4_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 1, 16, 4, -1, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t256_m1_n16_k4_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 1, 16, 4, 4, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t256_m1_n16_k4_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 1, 16, 4, 8, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t256_m2_n16_k4_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 2, 16, 4, -1, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t256_m2_n16_k4_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 2, 16, 4, 4, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t256_m2_n16_k4_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 2, 16, 4, 8, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t256_m3_n16_k4_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 3, 16, 4, -1, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t256_m3_n16_k4_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 3, 16, 4, 4, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t256_m3_n16_k4_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 3, 16, 4, 8, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t256_m4_n16_k4_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 4, 16, 4, -1, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t256_m4_n16_k4_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 4, 16, 4, 4, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t256_m4_n16_k4_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 4, 16, 4, 8, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t128_m1_n8_k4_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 1, 8, 4, -1, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t128_m1_n8_k4_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 1, 8, 4, 4, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t128_m1_n8_k4_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 1, 8, 4, 8, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t128_m2_n8_k4_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 2, 8, 4, -1, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t128_m2_n8_k4_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 2, 8, 4, 4, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t128_m2_n8_k4_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 2, 8, 4, 8, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t128_m3_n8_k4_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 3, 8, 4, -1, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t128_m3_n8_k4_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 3, 8, 4, 4, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t128_m3_n8_k4_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 3, 8, 4, 8, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t128_m4_n8_k4_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 4, 8, 4, -1, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t128_m4_n8_k4_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 4, 8, 4, 4, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t128_m4_n8_k4_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 4, 8, 4, 8, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t128_m1_n4_k8_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 1, 4, 8, -1, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t128_m1_n4_k8_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 1, 4, 8, 4, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t128_m1_n4_k8_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 1, 4, 8, 8, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t128_m2_n4_k8_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 2, 4, 8, -1, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t128_m2_n4_k8_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 2, 4, 8, 4, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t128_m2_n4_k8_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 2, 4, 8, 8, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t128_m3_n4_k8_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 3, 4, 8, -1, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t128_m3_n4_k8_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 3, 4, 8, 4, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t128_m3_n4_k8_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 3, 4, 8, 8, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t128_m4_n4_k8_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 4, 4, 8, -1, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t128_m4_n4_k8_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 4, 4, 8, 4, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_gptq_bf16_t128_m4_n4_k8_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 4, 4, 8, 8, false>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t256_m1_n8_k8_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 1, 8, 8, -1, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t256_m1_n8_k8_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 1, 8, 8, 4, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t256_m1_n8_k8_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 1, 8, 8, 8, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t256_m2_n8_k8_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 2, 8, 8, -1, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t256_m2_n8_k8_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 2, 8, 8, 4, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t256_m2_n8_k8_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 2, 8, 8, 8, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t256_m3_n8_k8_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 3, 8, 8, -1, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t256_m3_n8_k8_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 3, 8, 8, 4, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t256_m3_n8_k8_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 3, 8, 8, 8, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t256_m4_n8_k8_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 4, 8, 8, -1, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t256_m4_n8_k8_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 4, 8, 8, 4, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t256_m4_n8_k8_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 4, 8, 8, 8, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t256_m1_n16_k4_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 1, 16, 4, -1, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t256_m1_n16_k4_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 1, 16, 4, 4, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t256_m1_n16_k4_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 1, 16, 4, 8, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t256_m2_n16_k4_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 2, 16, 4, -1, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t256_m2_n16_k4_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 2, 16, 4, 4, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t256_m2_n16_k4_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 2, 16, 4, 8, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t256_m3_n16_k4_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 3, 16, 4, -1, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t256_m3_n16_k4_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 3, 16, 4, 4, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t256_m3_n16_k4_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 3, 16, 4, 8, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t256_m4_n16_k4_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 4, 16, 4, -1, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t256_m4_n16_k4_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 4, 16, 4, 4, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t256_m4_n16_k4_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 256, 4, 16, 4, 8, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t128_m1_n8_k4_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 1, 8, 4, -1, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t128_m1_n8_k4_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 1, 8, 4, 4, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t128_m1_n8_k4_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 1, 8, 4, 8, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t128_m2_n8_k4_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 2, 8, 4, -1, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t128_m2_n8_k4_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 2, 8, 4, 4, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t128_m2_n8_k4_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 2, 8, 4, 8, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t128_m3_n8_k4_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 3, 8, 4, -1, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t128_m3_n8_k4_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 3, 8, 4, 4, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t128_m3_n8_k4_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 3, 8, 4, 8, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t128_m4_n8_k4_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 4, 8, 4, -1, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t128_m4_n8_k4_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 4, 8, 4, 4, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t128_m4_n8_k4_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 4, 8, 4, 8, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t128_m1_n4_k8_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 1, 4, 8, -1, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t128_m1_n4_k8_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 1, 4, 8, 4, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t128_m1_n4_k8_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 1, 4, 8, 8, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t128_m2_n4_k8_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 2, 4, 8, -1, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t128_m2_n4_k8_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 2, 4, 8, 4, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t128_m2_n4_k8_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 2, 4, 8, 8, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t128_m3_n4_k8_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 3, 4, 8, -1, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t128_m3_n4_k8_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 3, 4, 8, 4, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t128_m3_n4_k8_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 3, 4, 8, 8, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t128_m4_n4_k8_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 4, 4, 8, -1, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t128_m4_n4_k8_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 4, 4, 8, 4, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_f16_t128_m4_n4_k8_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<H, 128, 4, 4, 8, 8, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t256_m1_n8_k8_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 1, 8, 8, -1, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t256_m1_n8_k8_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 1, 8, 8, 4, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t256_m1_n8_k8_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 1, 8, 8, 8, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t256_m2_n8_k8_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 2, 8, 8, -1, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t256_m2_n8_k8_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 2, 8, 8, 4, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t256_m2_n8_k8_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 2, 8, 8, 8, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t256_m3_n8_k8_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 3, 8, 8, -1, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t256_m3_n8_k8_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 3, 8, 8, 4, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t256_m3_n8_k8_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 3, 8, 8, 8, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t256_m4_n8_k8_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 4, 8, 8, -1, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t256_m4_n8_k8_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 4, 8, 8, 4, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t256_m4_n8_k8_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 4, 8, 8, 8, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t256_m1_n16_k4_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 1, 16, 4, -1, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t256_m1_n16_k4_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 1, 16, 4, 4, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t256_m1_n16_k4_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 1, 16, 4, 8, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t256_m2_n16_k4_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 2, 16, 4, -1, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t256_m2_n16_k4_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 2, 16, 4, 4, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t256_m2_n16_k4_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 2, 16, 4, 8, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t256_m3_n16_k4_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 3, 16, 4, -1, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t256_m3_n16_k4_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 3, 16, 4, 4, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t256_m3_n16_k4_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 3, 16, 4, 8, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t256_m4_n16_k4_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 4, 16, 4, -1, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t256_m4_n16_k4_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 4, 16, 4, 4, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t256_m4_n16_k4_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 256, 4, 16, 4, 8, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t128_m1_n8_k4_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 1, 8, 4, -1, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t128_m1_n8_k4_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 1, 8, 4, 4, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t128_m1_n8_k4_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 1, 8, 4, 8, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t128_m2_n8_k4_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 2, 8, 4, -1, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t128_m2_n8_k4_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 2, 8, 4, 4, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t128_m2_n8_k4_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 2, 8, 4, 8, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t128_m3_n8_k4_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 3, 8, 4, -1, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t128_m3_n8_k4_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 3, 8, 4, 4, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t128_m3_n8_k4_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 3, 8, 4, 8, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t128_m4_n8_k4_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 4, 8, 4, -1, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t128_m4_n8_k4_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 4, 8, 4, 4, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t128_m4_n8_k4_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 4, 8, 4, 8, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t128_m1_n4_k8_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 1, 4, 8, -1, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t128_m1_n4_k8_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 1, 4, 8, 4, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t128_m1_n4_k8_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 1, 4, 8, 8, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t128_m2_n4_k8_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 2, 4, 8, -1, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t128_m2_n4_k8_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 2, 4, 8, 4, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t128_m2_n4_k8_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 2, 4, 8, 8, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t128_m3_n4_k8_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 3, 4, 8, -1, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t128_m3_n4_k8_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 3, 4, 8, 4, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t128_m3_n4_k8_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 3, 4, 8, 8, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t128_m4_n4_k8_gm1(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 4, 4, 8, -1, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t128_m4_n4_k8_g4(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 4, 4, 8, 4, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn marlin_awq_bf16_t128_m4_n4_k8_g8(a: *const [u32; 4], b: *const [u32; 4], c: *mut [u32; 4], s: *const [u32; 4], zp: *const [u32; 4], g_idx: *const i32, m: i32, n: i32, kk: i32, num_groups: i32, locks: *mut i32) { let _ = g_idx; marlin::<B, 128, 4, 4, 8, 8, true>(a, b, c, s, zp, m, n, kk, num_groups, locks) }
    #[kernel] pub unsafe fn gptq_marlin_repack_4_perm(w: *const u32, perm: *const u32, out: *mut u32, size_k: i32, size_n: i32) { marlin_repack::<4, true, false>(w, perm, out, size_k, size_n) }
    #[kernel] pub unsafe fn gptq_marlin_repack_4_noperm(w: *const u32, perm: *const u32, out: *mut u32, size_k: i32, size_n: i32) { marlin_repack::<4, false, false>(w, perm, out, size_k, size_n) }
    #[kernel] pub unsafe fn awq_marlin_repack_4(w: *const u32, out: *mut u32, size_k: i32, size_n: i32) { marlin_repack::<4, false, true>(w, core::ptr::null(), out, size_k, size_n) }
    #[kernel] pub unsafe fn gptq_marlin_repack_8_perm(w: *const u32, perm: *const u32, out: *mut u32, size_k: i32, size_n: i32) { marlin_repack::<8, true, false>(w, perm, out, size_k, size_n) }
    #[kernel] pub unsafe fn gptq_marlin_repack_8_noperm(w: *const u32, perm: *const u32, out: *mut u32, size_k: i32, size_n: i32) { marlin_repack::<8, false, false>(w, perm, out, size_k, size_n) }
    #[kernel] pub unsafe fn awq_marlin_repack_8(w: *const u32, out: *mut u32, size_k: i32, size_n: i32) { marlin_repack::<8, false, true>(w, core::ptr::null(), out, size_k, size_n) }
    // GENERATED KERNELS END
}

fn main() {}
