#![allow(unsafe_op_in_unsafe_fn)]
#![allow(non_snake_case, clippy::missing_safety_doc, clippy::too_many_arguments, dead_code)]
//! Mapped-memory doorbell for titan's tiered experts (mistralrs-quant titan_doorbell.rs): the GPU publishes a MoE
//! layer's routing ids and q8_1 input into pinned, mapped host memory and bumps a sequence word there; the host
//! answers with the sequence number after writing the CPU experts' rows; the GPU waits for the answer and copies the
//! rows into the expert output. Every argument is a fixed pointer or a shape: the sequence number lives in device
//! memory, so the launches can be captured in a CUDA graph.
//!
//! Control block (host, u32 words, each on its own 64-byte line): PUB at 0, DONE at 16, ERR at 32.
//! Row header (host): [m, task 0, .., task m-1]; rows: m x n floats.
//!
//! `main()` is the functional gate: a host thread plays the CPU side (random miss sets, m = 0 included) against
//! publish -> wait -> scatter and publish -> collect, for many layers in one stream with no host sync in between,
//! and checks every output byte and every published byte.

use cuda_device::{SharedArray, kernel, ptx_asm, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub mod kernels {
    use super::*;

    pub const CTL_PUB: usize = 0;
    pub const CTL_DONE: usize = 16;
    pub const CTL_ERR: usize = 32;
    /// Largest m the scatter's shared task list takes (mistralrs-quant SCATTER_MAX_TASKS).
    pub const MAX_TASKS: usize = 1024;

    #[inline(always)]
    unsafe fn ld_volatile(p: *const u32) -> u32 {
        let r: u32;
        ptx_asm!("ld.volatile.global.u32 %0, [%1];", out("=r") r, in("l") p as u64);
        r
    }
    #[inline(always)]
    unsafe fn st_volatile(p: *mut u32, v: u32) {
        ptx_asm!("st.volatile.global.u32 [%0], %1;", in("l") p as u64, in("r") v, clobber("memory"));
    }
    #[inline(always)]
    unsafe fn fence_sys() {
        ptx_asm!("fence.sc.sys;", clobber("memory"));
    }
    #[inline(always)]
    unsafe fn globaltimer() -> u64 {
        let t: u64;
        ptx_asm!("mov.u64 %0, %%globaltimer;", out("=l") t);
        t
    }
    #[inline(always)]
    unsafe fn nanosleep() {
        ptx_asm!("nanosleep.u32 128;", clobber("memory"));
    }
    #[inline(always)]
    unsafe fn copy16(src: *const u32, dst: *mut u32) {
        let (a, b, c, d): (u32, u32, u32, u32);
        ptx_asm!("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];", out("=r") a, out("=r") b, out("=r") c, out("=r") d, in("l") src as u64);
        ptx_asm!("st.global.v4.u32 [%0], {%1, %2, %3, %4};", in("l") dst as u64, in("r") a, in("r") b, in("r") c, in("r") d, clobber("memory"));
    }
    /// 16 bytes from host memory, bypassing the caches (the host wrote them after the kernel may have cached a line).
    #[inline(always)]
    unsafe fn copy16_cv(src: *const u32, dst: *mut u32) {
        let (a, b, c, d): (u32, u32, u32, u32);
        ptx_asm!("ld.global.cv.v4.u32 {%0, %1, %2, %3}, [%4];", out("=r") a, out("=r") b, out("=r") c, out("=r") d, in("l") src as u64);
        ptx_asm!("st.global.v4.u32 [%0], {%1, %2, %3, %4};", in("l") dst as u64, in("r") a, in("r") b, in("r") c, in("r") d, clobber("memory"));
    }

    /// Spin (thread 0) until DONE equals the device sequence number or the timeout passes (then ERR = seq).
    #[inline(always)]
    unsafe fn spin(dev_seq: *const u32, host_ctl: *mut u32, timeout_ns: u64) {
        let s = ld_volatile(dev_seq);
        let t0 = globaltimer();
        while ld_volatile(host_ctl.add(CTL_DONE)) != s {
            nanosleep();
            if globaltimer() - t0 > timeout_ns {
                st_volatile(host_ctl.add(CTL_ERR), s);
                break;
            }
        }
        fence_sys();
    }

    /// Rows listed in the header into their task rows of `out`; the header goes through shared memory once per block.
    #[inline(always)]
    unsafe fn scatter_rows(hdr: *const u32, rows: *const u32, out: *mut u32, n: u32, max_rows: u32, wait: bool, seq: *const u32, ctl: *mut u32, timeout: u64) {
        static mut M: SharedArray<u32, 1> = SharedArray::UNINIT;
        static mut TASKS: SharedArray<u32, MAX_TASKS> = SharedArray::UNINIT;
        let m_s = SharedArray::as_raw_mut_ptr(&raw mut M);
        let t_s = SharedArray::as_raw_mut_ptr(&raw mut TASKS);
        let tid = thread::threadIdx_x();
        if tid == 0 {
            if wait {
                spin(seq, ctl, timeout);
            }
            let m = ld_volatile(hdr);
            *m_s = if m < max_rows { m } else { max_rows };
        }
        thread::sync_threads();
        let m = *m_s;
        if m == 0 {
            return;
        }
        let mut i = tid;
        while i < m {
            *t_s.add(i as usize) = ld_volatile(hdr.add(1 + i as usize));
            i += thread::blockDim_x();
        }
        thread::sync_threads();
        let n4 = n / 4;
        let total = m * n4;
        let mut idx = thread::blockIdx_x() * thread::blockDim_x() + tid;
        while idx < total {
            let r = idx / n4;
            let c = idx - r * n4;
            let t = *t_s.add(r as usize) as usize;
            copy16_cv(rows.add(4 * idx as usize), out.add(t * n as usize + 4 * c as usize));
            idx += thread::gridDim_x() * thread::blockDim_x();
        }
    }

    /// Copy `n_ids` routing ids and `xq_vec` 16-byte words of the q8_1 input into mapped host memory, then publish the
    /// next sequence number. One block.
    #[kernel]
    pub unsafe fn db_publish(ids: *const u32, n_ids: u32, xq: *const u32, xq_vec: u32, host_ids: *mut u32, host_xq: *mut u32, dev_seq: *mut u32, host_ctl: *mut u32) {
        let tid = thread::threadIdx_x();
        let mut i = tid;
        while i < n_ids {
            st_volatile(host_ids.add(i as usize), ld_volatile(ids.add(i as usize)));
            i += thread::blockDim_x();
        }
        let mut i = tid;
        while i < xq_vec {
            copy16(xq.add(4 * i as usize), host_xq.add(4 * i as usize));
            i += thread::blockDim_x();
        }
        fence_sys();
        thread::sync_threads();
        if tid == 0 {
            let s = ld_volatile(dev_seq) + 1;
            st_volatile(dev_seq, s);
            fence_sys();
            st_volatile(host_ctl.add(CTL_PUB), s);
        }
    }

    /// Wait for the host's answer to the last publish (thread 0 spins).
    #[kernel]
    pub unsafe fn db_wait(dev_seq: *const u32, host_ctl: *mut u32, timeout_ns: u64) {
        if thread::threadIdx_x() == 0 {
            spin(dev_seq, host_ctl, timeout_ns);
        }
    }

    /// `out[task[r] * n ..][..n] = rows[r * n ..][..n]` for the header's m rows; n is a multiple of 4.
    #[kernel]
    pub unsafe fn db_scatter(hdr: *const u32, rows: *const u32, out: *mut u32, n: u32, max_rows: u32) {
        scatter_rows(hdr, rows, out, n, max_rows, false, hdr, core::ptr::null_mut(), 0);
    }

    /// `db_wait` then `db_scatter` in one launch: thread 0 of every block spins, then the blocks copy the rows.
    #[kernel]
    pub unsafe fn db_collect(dev_seq: *const u32, host_ctl: *mut u32, timeout_ns: u64, hdr: *const u32, rows: *const u32, out: *mut u32, n: u32, max_rows: u32) {
        scatter_rows(hdr, rows, out, n, max_rows, true, dev_seq, host_ctl, timeout_ns);
    }
}

mod gate;

fn main() {
    std::process::exit(if gate::run() { 0 } else { 1 });
}
