//! A CUDA memory pool of its own for titan's long-lived buffers (tiered expert slots, KV caches).
//!
//! They used to come from the device's default stream-ordered pool, which also serves every step's temporaries.
//! Resizing them (dynamic KV / expert demotion) left their old blocks as holes between temporaries that the pool
//! could neither hand back to the driver nor fit larger allocations into: 2 GiB reserved and unused, and a 47k prompt
//! ran out of memory. Here they sit apart, and with the pool's release threshold at 0 a freed buffer goes back to
//! the driver at the next synchronisation, where any allocation can take it. A `CudaSlice` from this pool is freed
//! as any other (`cuMemFreeAsync` returns the block to the pool it came from).

use candle_core::cuda::cudarc::driver::{sys, CudaSlice, DeviceRepr, ValidAsZeroBits};
use candle_core::cuda::{CudaDevice, WrapErr};
use candle_core::Result;
use std::collections::HashMap;
use std::sync::Mutex;

struct Pool(sys::CUmemoryPool);
unsafe impl Send for Pool {}

static POOLS: Mutex<Option<HashMap<i32, Pool>>> = Mutex::new(None);

fn pool(dev: &CudaDevice) -> Result<sys::CUmemoryPool> {
    let stream = dev.cuda_stream();
    let ctx = stream.context();
    ctx.bind_to_thread().w()?;
    let ordinal = ctx.cu_device();
    let mut g = POOLS.lock().unwrap();
    let map = g.get_or_insert_with(HashMap::new);
    if let Some(p) = map.get(&ordinal) {
        return Ok(p.0);
    }
    let mut props: sys::CUmemPoolProps = unsafe { std::mem::zeroed() };
    props.allocType = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
    props.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    props.location.__bindgen_anon_1.id = ordinal;
    let mut p: sys::CUmemoryPool = std::ptr::null_mut();
    unsafe { sys::cuMemPoolCreate(&mut p, &props) }.result().w()?;
    // release threshold 0 (the default): freed memory returns to the driver at the next sync
    map.insert(ordinal, Pool(p));
    tracing::info!("titan: long-lived buffers (expert slots, KV) get their own CUDA memory pool");
    Ok(p)
}

/// `n` zeroed elements from the long-lived pool, freed like any `CudaSlice`.
pub fn alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(dev: &CudaDevice, n: usize) -> Result<CudaSlice<T>> {
    let p = pool(dev)?;
    let stream = dev.cuda_stream();
    let bytes = (n * std::mem::size_of::<T>()).max(1);
    let mut ptr: sys::CUdeviceptr = 0;
    unsafe { sys::cuMemAllocFromPoolAsync(&mut ptr, bytes, p, stream.cu_stream()) }.result().w()?;
    unsafe { sys::cuMemsetD8Async(ptr, 0, bytes, stream.cu_stream()) }.result().w()?;
    Ok(unsafe { stream.upgrade_device_ptr::<T>(ptr, n) })
}

/// Return the long-lived pool's unused memory to the driver now (after a sync).
pub fn trim(dev: &CudaDevice) {
    if let Ok(p) = pool(dev) {
        let _ = dev.cuda_stream().synchronize();
        unsafe {
            let _ = sys::cuMemPoolTrimTo(p, 0);
        }
    }
}
