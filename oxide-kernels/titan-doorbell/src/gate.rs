//! Functional gate: a host thread answers L layers published back to back in one stream (no host sync between
//! them), with random miss sets (m = 0 included); every published byte and every output byte is checked.
use cuda_core::sys as cu;
use std::ffi::c_void;
use std::sync::atomic::{fence, Ordering};

macro_rules! ck {
    ($e:expr) => {{
        let r = $e;
        assert_eq!(r, cu::cudaError_enum_CUDA_SUCCESS, "{}", stringify!($e));
    }};
}

const LAYERS: usize = 400;
const TOPK: usize = 8;
const N: usize = 2048;
const XQ_BYTES: usize = 2304;
// host regions (bytes)
const CTL: usize = 0;
const IDS: usize = 4096;
const XQ: usize = 8192;
const HDR: usize = 16384;
const ROWS: usize = 20480;
const HOST_LEN: usize = ROWS + TOPK * N * 4;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn load(path: &str) -> cu::CUmodule {
    let mut img = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    img.push(0);
    let mut m: cu::CUmodule = std::ptr::null_mut();
    unsafe { ck!(cu::cuModuleLoadData(&mut m, img.as_ptr() as *const c_void)) };
    m
}
fn func(m: cu::CUmodule, name: &str) -> cu::CUfunction {
    let mut f: cu::CUfunction = std::ptr::null_mut();
    let c = std::ffi::CString::new(name).unwrap();
    unsafe { ck!(cu::cuModuleGetFunction(&mut f, m, c.as_ptr())) };
    f
}
unsafe fn launch(f: cu::CUfunction, grid: u32, block: u32, args: &mut [*mut c_void]) {
    ck!(cu::cuLaunchKernel(f, grid, 1, 1, block, 1, 1, 0, std::ptr::null_mut(), args.as_mut_ptr(), std::ptr::null_mut()));
}
fn dalloc(bytes: usize) -> u64 {
    let mut p: cu::CUdeviceptr = 0;
    unsafe {
        ck!(cu::cuMemAlloc_v2(&mut p, bytes));
        ck!(cu::cuMemsetD8_v2(p, 0, bytes));
    }
    p
}
fn ids_of(l: usize) -> Vec<u32> {
    (0..TOPK).map(|k| (l * 37 + k * 11) as u32 % 256).collect()
}
fn xq_of(l: usize) -> Vec<u8> {
    (0..XQ_BYTES).map(|i| (l * 7 + i * 3) as u8).collect()
}
fn val(l: usize, t: usize, c: usize) -> f32 {
    (l * 100_000 + t * 10_000 + c) as f32
}
fn misses(l: usize) -> Vec<usize> {
    let mut r = Rng(0x9e37_79b9 ^ (l as u64 + 1) * 0x1234_5678);
    let m = (r.next() % 9) as usize;
    let mut t: Vec<usize> = (0..TOPK).collect();
    for i in (1..TOPK).rev() {
        t.swap(i, (r.next() % (i as u64 + 1)) as usize);
    }
    t.truncate(m);
    t
}

pub fn run() -> bool {
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
    let dir = env!("CARGO_MANIFEST_DIR");
    let ptx = std::env::var("OXIDE_PTX").unwrap_or_else(|_| format!("{dir}/titan_doorbell.ptx"));
    let m = load(&ptx);
    let (publish, wait, scatter, collect) = (func(m, "db_publish"), func(m, "db_wait"), func(m, "db_scatter"), func(m, "db_collect"));
    let mut host: *mut c_void = std::ptr::null_mut();
    let mut hdev: cu::CUdeviceptr = 0;
    unsafe {
        ck!(cu::cuMemHostAlloc(&mut host, HOST_LEN, 0x1 | 0x2));
        std::ptr::write_bytes(host as *mut u8, 0, HOST_LEN);
        ck!(cu::cuMemHostGetDevicePointer_v2(&mut hdev, host, 0));
    }
    let hb = host as usize;
    let seq = dalloc(4);
    let ids_d: Vec<u64> = (0..LAYERS).map(|_| dalloc(4 * TOPK)).collect();
    let xq_d: Vec<u64> = (0..LAYERS).map(|_| dalloc(XQ_BYTES)).collect();
    let out_d: Vec<u64> = (0..LAYERS).map(|_| dalloc(4 * TOPK * N)).collect();
    for l in 0..LAYERS {
        unsafe {
            ck!(cu::cuMemcpyHtoD_v2(ids_d[l], ids_of(l).as_ptr() as *const c_void, 4 * TOPK));
            ck!(cu::cuMemcpyHtoD_v2(xq_d[l], xq_of(l).as_ptr() as *const c_void, XQ_BYTES));
        }
    }
    // the CPU side
    let cpu = std::thread::spawn(move || {
        let ctl = |w: usize| (hb + CTL + 4 * w) as *mut u32;
        let mut last = 0u32;
        let mut bad = 0usize;
        for l in 0..LAYERS {
            let t0 = std::time::Instant::now();
            let s = loop {
                let p = unsafe { std::ptr::read_volatile(ctl(0)) };
                if p != last {
                    break p;
                }
                if t0.elapsed().as_secs() > 20 {
                    eprintln!("layer {l}: no publish");
                    return usize::MAX;
                }
                std::hint::spin_loop();
            };
            fence(Ordering::Acquire);
            last = s;
            let ids: Vec<u32> = (0..TOPK).map(|k| unsafe { std::ptr::read_volatile((hb + IDS + 4 * k) as *const u32) }).collect();
            let xq: Vec<u8> = (0..XQ_BYTES).map(|i| unsafe { std::ptr::read_volatile((hb + XQ + i) as *const u8) }).collect();
            bad += (ids != ids_of(l)) as usize + (xq != xq_of(l)) as usize;
            let tasks = misses(l);
            for (r, &t) in tasks.iter().enumerate() {
                for c in 0..N {
                    unsafe { *((hb + ROWS + 4 * (r * N + c)) as *mut f32) = val(l, t, c) };
                }
                unsafe { std::ptr::write_volatile((hb + HDR + 4 * (1 + r)) as *mut u32, t as u32) };
            }
            unsafe { std::ptr::write_volatile((hb + HDR) as *mut u32, tasks.len() as u32) };
            fence(Ordering::Release);
            unsafe { std::ptr::write_volatile(ctl(16), s) };
        }
        bad
    });
    let timeout: u64 = 30_000_000_000;
    let (mut n, mut maxr) = (N as u32, TOPK as u32);
    let mut n_ids = TOPK as u32;
    let mut xq_vec = (XQ_BYTES / 16) as u32;
    let (mut h_ids, mut h_xq, mut h_ctl, mut h_hdr, mut h_rows) = (hdev + IDS as u64, hdev + XQ as u64, hdev + CTL as u64, hdev + HDR as u64, hdev + ROWS as u64);
    let mut seq_p = seq;
    let t0 = std::time::Instant::now();
    for l in 0..LAYERS {
        let (mut ids_p, mut xq_p, mut out_p) = (ids_d[l], xq_d[l], out_d[l]);
        let mut to = timeout;
        unsafe {
            launch(publish, 1, 256, &mut [
                &mut ids_p as *mut u64 as *mut c_void, &mut n_ids as *mut u32 as *mut c_void, &mut xq_p as *mut u64 as *mut c_void,
                &mut xq_vec as *mut u32 as *mut c_void, &mut h_ids as *mut u64 as *mut c_void, &mut h_xq as *mut u64 as *mut c_void,
                &mut seq_p as *mut u64 as *mut c_void, &mut h_ctl as *mut u64 as *mut c_void,
            ]);
            if l % 2 == 0 {
                launch(wait, 1, 32, &mut [&mut seq_p as *mut u64 as *mut c_void, &mut h_ctl as *mut u64 as *mut c_void, &mut to as *mut u64 as *mut c_void]);
                launch(scatter, 16, 256, &mut [
                    &mut h_hdr as *mut u64 as *mut c_void, &mut h_rows as *mut u64 as *mut c_void, &mut out_p as *mut u64 as *mut c_void,
                    &mut n as *mut u32 as *mut c_void, &mut maxr as *mut u32 as *mut c_void,
                ]);
            } else {
                launch(collect, 7, 256, &mut [
                    &mut seq_p as *mut u64 as *mut c_void, &mut h_ctl as *mut u64 as *mut c_void, &mut to as *mut u64 as *mut c_void,
                    &mut h_hdr as *mut u64 as *mut c_void, &mut h_rows as *mut u64 as *mut c_void, &mut out_p as *mut u64 as *mut c_void,
                    &mut n as *mut u32 as *mut c_void, &mut maxr as *mut u32 as *mut c_void,
                ]);
            }
        }
    }
    let enq = t0.elapsed();
    unsafe { ck!(cu::cuCtxSynchronize()) };
    let total = t0.elapsed();
    let bad_pub = cpu.join().unwrap();
    let err = unsafe { std::ptr::read_volatile((hb + CTL + 4 * 32) as *const u32) };
    let mut bad_out = 0usize;
    for l in 0..LAYERS {
        let mut h = vec![0f32; TOPK * N];
        unsafe { ck!(cu::cuMemcpyDtoH_v2(h.as_mut_ptr() as *mut c_void, out_d[l], 4 * TOPK * N)) };
        let tasks = misses(l);
        for t in 0..TOPK {
            for c in 0..N {
                let want = if tasks.contains(&t) { val(l, t, c) } else { 0.0 };
                bad_out += (h[t * N + c].to_bits() != want.to_bits()) as usize;
            }
        }
    }
    let ok = bad_pub == 0 && bad_out == 0 && err == 0;
    println!(
        "titan_doorbell: {LAYERS} layers (publish + wait/scatter or collect, one stream, no host sync), {} missed rows; \
         published mismatches {bad_pub}, output words differing {bad_out}, error word {err}; enqueue {:.1} ms, total {:.1} ms ({:.1} us/layer) -> {}",
        (0..LAYERS).map(|l| misses(l).len()).sum::<usize>(),
        enq.as_secs_f64() * 1e3,
        total.as_secs_f64() * 1e3,
        total.as_secs_f64() * 1e6 / LAYERS as f64,
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}
