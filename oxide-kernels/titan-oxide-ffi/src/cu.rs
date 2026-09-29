//! The CUDA driver API subset the launchers use, with the same names and types as
//! `cuda_core::sys` (bindgen of cuda.h), linked dynamically against libcuda (`-lcuda`, as cudarc's
//! `dynamic-linking` feature does for candle).
//!
//! The reference libraries are compiled with `--default-stream per-thread`: to them a null stream
//! is the per-thread default stream. The stream-taking wrappers below therefore turn a null
//! `CUstream` into `CU_STREAM_PER_THREAD` (several ports pass the caller's null stream straight
//! through, which the driver would read as the legacy stream).
#![allow(non_camel_case_types, non_upper_case_globals, non_snake_case, dead_code, clippy::missing_safety_doc)]
use std::ffi::{c_char, c_int, c_uchar, c_uint, c_void};

pub type CUresult = c_uint;
pub type CUdevice = c_int;
pub type CUdeviceptr = u64;
pub type CUdevice_attribute = c_uint;
pub type CUfunction_attribute = c_uint;
#[repr(C)]
pub struct CUctx_st {
    _p: [u8; 0],
}
#[repr(C)]
pub struct CUmod_st {
    _p: [u8; 0],
}
#[repr(C)]
pub struct CUfunc_st {
    _p: [u8; 0],
}
#[repr(C)]
pub struct CUstream_st {
    _p: [u8; 0],
}
pub type CUcontext = *mut CUctx_st;
pub type CUmodule = *mut CUmod_st;
pub type CUfunction = *mut CUfunc_st;
pub type CUstream = *mut CUstream_st;

pub const cudaError_enum_CUDA_SUCCESS: CUresult = 0;
pub const CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: CUfunction_attribute = 8;
pub const CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: CUdevice_attribute = 16;
pub const CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: CUdevice_attribute = 75;
pub const CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: CUdevice_attribute = 97;

/// `CU_STREAM_PER_THREAD`.
pub const STREAM_PER_THREAD: usize = 0x2;

#[inline(always)]
fn ptsz(s: CUstream) -> CUstream {
    if s.is_null() { STREAM_PER_THREAD as CUstream } else { s }
}

mod raw {
    use super::*;
    #[link(name = "cuda")]
    unsafe extern "C" {
        pub fn cuInit(flags: c_uint) -> CUresult;
        pub fn cuProfilerStart() -> CUresult;
        pub fn cuProfilerStop() -> CUresult;
        pub fn cuGetErrorString(error: CUresult, p: *mut *const c_char) -> CUresult;
        pub fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
        pub fn cuDeviceGetAttribute(pi: *mut c_int, attrib: CUdevice_attribute, dev: CUdevice) -> CUresult;
        pub fn cuDevicePrimaryCtxRetain(pctx: *mut CUcontext, dev: CUdevice) -> CUresult;
        pub fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult;
        pub fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
        pub fn cuCtxPushCurrent_v2(ctx: CUcontext) -> CUresult;
        pub fn cuCtxPopCurrent_v2(pctx: *mut CUcontext) -> CUresult;
        pub fn cuCtxGetDevice(device: *mut CUdevice) -> CUresult;
        pub fn cuCtxSynchronize() -> CUresult;
        pub fn cuStreamGetCtx(h: CUstream, pctx: *mut CUcontext) -> CUresult;
        pub fn cuStreamSynchronize(h: CUstream) -> CUresult;
        pub fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
        pub fn cuModuleGetFunction(f: *mut CUfunction, m: CUmodule, name: *const c_char) -> CUresult;
        pub fn cuFuncSetAttribute(f: CUfunction, attrib: CUfunction_attribute, value: c_int) -> CUresult;
        pub fn cuOccupancyMaxActiveBlocksPerMultiprocessor(n: *mut c_int, f: CUfunction, block: c_int, smem: usize) -> CUresult;
        pub fn cuLaunchKernel(
            f: CUfunction, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, smem: c_uint, s: CUstream,
            params: *mut *mut c_void, extra: *mut *mut c_void,
        ) -> CUresult;
        pub fn cuMemAllocAsync(p: *mut CUdeviceptr, bytes: usize, s: CUstream) -> CUresult;
        pub fn cuMemFreeAsync(p: CUdeviceptr, s: CUstream) -> CUresult;
        pub fn cuMemsetD8Async(d: CUdeviceptr, v: c_uchar, n: usize, s: CUstream) -> CUresult;
        pub fn cuMemsetD32Async(d: CUdeviceptr, v: c_uint, n: usize, s: CUstream) -> CUresult;
        pub fn cuMemcpyAsync(dst: CUdeviceptr, src: CUdeviceptr, n: usize, s: CUstream) -> CUresult;
        pub fn cuMemcpyDtoDAsync_v2(dst: CUdeviceptr, src: CUdeviceptr, n: usize, s: CUstream) -> CUresult;
        pub fn cuMemcpyDtoHAsync_v2(dst: *mut c_void, src: CUdeviceptr, n: usize, s: CUstream) -> CUresult;
        pub fn cuMemcpyHtoDAsync_v2(dst: CUdeviceptr, src: *const c_void, n: usize, s: CUstream) -> CUresult;
    }
}

pub use raw::{
    cuCtxGetCurrent, cuCtxGetDevice, cuCtxPopCurrent_v2, cuCtxPushCurrent_v2, cuCtxSetCurrent, cuCtxSynchronize, cuDeviceGet,
    cuDeviceGetAttribute, cuDevicePrimaryCtxRetain, cuFuncSetAttribute, cuGetErrorString, cuInit, cuModuleGetFunction,
    cuModuleLoadData, cuOccupancyMaxActiveBlocksPerMultiprocessor, cuStreamGetCtx,
};

pub unsafe fn cuStreamSynchronize(h: CUstream) -> CUresult {
    unsafe { raw::cuStreamSynchronize(ptsz(h)) }
}
pub unsafe fn cuLaunchKernel(
    f: CUfunction, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, smem: c_uint, s: CUstream,
    params: *mut *mut c_void, extra: *mut *mut c_void,
) -> CUresult {
    unsafe { raw::cuLaunchKernel(f, gx, gy, gz, bx, by, bz, smem, ptsz(s), params, extra) }
}
pub unsafe fn cuMemAllocAsync(p: *mut CUdeviceptr, bytes: usize, s: CUstream) -> CUresult {
    unsafe { raw::cuMemAllocAsync(p, bytes, ptsz(s)) }
}
pub unsafe fn cuMemFreeAsync(p: CUdeviceptr, s: CUstream) -> CUresult {
    unsafe { raw::cuMemFreeAsync(p, ptsz(s)) }
}
pub unsafe fn cuMemsetD8Async(d: CUdeviceptr, v: c_uchar, n: usize, s: CUstream) -> CUresult {
    unsafe { raw::cuMemsetD8Async(d, v, n, ptsz(s)) }
}
pub unsafe fn cuMemsetD32Async(d: CUdeviceptr, v: c_uint, n: usize, s: CUstream) -> CUresult {
    unsafe { raw::cuMemsetD32Async(d, v, n, ptsz(s)) }
}
pub unsafe fn cuMemcpyAsync(dst: CUdeviceptr, src: CUdeviceptr, n: usize, s: CUstream) -> CUresult {
    unsafe { raw::cuMemcpyAsync(dst, src, n, ptsz(s)) }
}
pub unsafe fn cuMemcpyDtoDAsync_v2(dst: CUdeviceptr, src: CUdeviceptr, n: usize, s: CUstream) -> CUresult {
    unsafe { raw::cuMemcpyDtoDAsync_v2(dst, src, n, ptsz(s)) }
}
pub unsafe fn cuMemcpyDtoHAsync_v2(dst: *mut c_void, src: CUdeviceptr, n: usize, s: CUstream) -> CUresult {
    unsafe { raw::cuMemcpyDtoHAsync_v2(dst, src, n, ptsz(s)) }
}
pub unsafe fn cuMemcpyHtoDAsync_v2(dst: CUdeviceptr, src: *const c_void, n: usize, s: CUstream) -> CUresult {
    unsafe { raw::cuMemcpyHtoDAsync_v2(dst, src, n, ptsz(s)) }
}

/// Driver profiler range (for cudart's cudaProfilerStart/Stop twins in lib.rs).
pub unsafe fn profiler_start() -> CUresult {
    unsafe { raw::cuProfilerStart() }
}

pub unsafe fn profiler_stop() -> CUresult {
    unsafe { raw::cuProfilerStop() }
}
