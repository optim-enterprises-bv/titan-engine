//! kdiff: differential tester for cuda-oxide ports of nvcc kernels.
//!
//! Loads the reference PTX (exactly as the original project built it) and the oxide
//! PTX into one context, launches the same entry by name in both with identical
//! argument bytes and identical input buffers, and compares every output byte.
//! A port passes only on zero differing bytes.

use cuda_core::{CudaContext, CudaModule, CudaStream, DeviceBuffer};
use std::ffi::c_void;
use std::sync::Arc;

pub use cuda_core;

/// One kernel argument, passed by value exactly as the C ABI expects.
#[derive(Clone)]
pub enum Arg {
    /// Index of a buffer in the launch's buffer list (passed as a device pointer).
    Buf(usize),
    /// A null device pointer.
    Null,
    U64(u64),
    I64(i64),
    U32(u32),
    I32(i32),
    F32(f32),
    F64(f64),
    /// A by-value 2-byte scalar (f16 / bf16 bit pattern).
    B16(u16),
    /// A by-value 1-byte scalar (fp8, u8).
    B8(u8),
}

pub struct Harness {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub reference: Arc<CudaModule>,
    pub oxide: Arc<CudaModule>,
}

/// Result of one differential launch.
pub struct Diff {
    pub bytes: usize,
    pub differing: usize,
    pub first: Option<(usize, usize, u8, u8)>, // (buffer, byte offset, ref, oxide)
}

impl Harness {
    pub fn new(reference_ptx: &str, oxide_ptx: &str) -> Self {
        let ctx = CudaContext::new(0).expect("cuda context");
        let stream = ctx.default_stream();
        let reference = ctx
            .load_module_from_ptx_src(&std::fs::read_to_string(reference_ptx).expect(reference_ptx))
            .expect("reference ptx load");
        let oxide = ctx
            .load_module_from_ptx_src(&std::fs::read_to_string(oxide_ptx).expect(oxide_ptx))
            .expect("oxide ptx load");
        Self { ctx, stream, reference, oxide }
    }

    /// Like `new`, but either side may be a cubin (the driver sniffs PTX text vs ELF). For
    /// references that exist only as SASS, e.g. candle's statically linked FFI kernels.
    pub fn from_files(reference: &str, oxide: &str) -> Self {
        let ctx = CudaContext::new(0).expect("cuda context");
        let stream = ctx.default_stream();
        let reference = ctx.load_module_from_file(reference).unwrap_or_else(|e| panic!("{reference}: {e:?}"));
        let oxide = ctx.load_module_from_file(oxide).unwrap_or_else(|e| panic!("{oxide}: {e:?}"));
        Self { ctx, stream, reference, oxide }
    }

    /// Launch `name` in both modules on private copies of `buffers` (raw bytes) and
    /// compare the buffers afterwards. `outputs` lists which buffers to compare
    /// (usually just the output); inputs are compared too if listed.
    pub fn diff(
        &self,
        name: &str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared: u32,
        args: &[Arg],
        buffers: &[Vec<u8>],
        outputs: &[usize],
    ) -> Diff {
        self.diff_pair(name, name, grid, block, shared, args, buffers, outputs)
    }

    /// `diff` for entries whose names differ, e.g. a mangled C++ template instance vs its port.
    #[allow(clippy::too_many_arguments)]
    pub fn diff_pair(
        &self,
        ref_name: &str,
        oxide_name: &str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared: u32,
        args: &[Arg],
        buffers: &[Vec<u8>],
        outputs: &[usize],
    ) -> Diff {
        let run = |module: &Arc<CudaModule>, name: &str| -> Vec<Vec<u8>> {
            let bufs: Vec<DeviceBuffer<u8>> = buffers
                .iter()
                .map(|b| DeviceBuffer::from_host(&self.stream, if b.is_empty() { &[0u8][..] } else { b }).unwrap())
                .collect();
            let mut vals: Vec<[u8; 8]> = args
                .iter()
                .map(|a| match a {
                    Arg::Buf(i) => bufs[*i].cu_deviceptr().to_le_bytes(),
                    Arg::Null => 0u64.to_le_bytes(),
                    Arg::U64(v) => v.to_le_bytes(),
                    Arg::I64(v) => v.to_le_bytes(),
                    Arg::U32(v) => (*v as u64).to_le_bytes(),
                    Arg::I32(v) => (*v as u32 as u64).to_le_bytes(),
                    Arg::F32(v) => (v.to_bits() as u64).to_le_bytes(),
                    Arg::F64(v) => v.to_bits().to_le_bytes(),
                    Arg::B16(v) => (*v as u64).to_le_bytes(),
                    Arg::B8(v) => (*v as u64).to_le_bytes(),
                })
                .collect();
            // Each param is read from the start of its slot, little-endian, for its own size.
            let mut ptrs: Vec<*mut c_void> = vals.iter_mut().map(|v| v.as_mut_ptr() as *mut c_void).collect();
            let f = module.load_function(name).unwrap_or_else(|e| panic!("{name}: {e:?}"));
            unsafe {
                cuda_core::simt::launch_kernel_on_stream(&f, grid, block, shared, &self.stream, &mut ptrs)
                    .unwrap_or_else(|e| panic!("launch {name}: {e:?}"));
            }
            self.stream.synchronize().unwrap_or_else(|e| panic!("sync after {name}: {e:?}"));
            outputs.iter().map(|&i| bufs[i].to_host_vec(&self.stream).unwrap()).collect()
        };
        let a = run(&self.reference, ref_name);
        let b = run(&self.oxide, oxide_name);
        let mut d = Diff { bytes: 0, differing: 0, first: None };
        for (k, (x, y)) in a.iter().zip(&b).enumerate() {
            let n = buffers[outputs[k]].len();
            d.bytes += n;
            for i in 0..n {
                if x[i] != y[i] {
                    d.differing += 1;
                    if d.first.is_none() {
                        d.first = Some((outputs[k], i, x[i], y[i]));
                    }
                }
            }
        }
        d
    }

    pub fn has(&self, module_is_oxide: bool, name: &str) -> bool {
        let m = if module_is_oxide { &self.oxide } else { &self.reference };
        m.load_function(name).is_ok()
    }
}

/// Deterministic xorshift generator.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
    /// f32 values mixing normal ranges with edge cases (+-0, denormals, inf, nan, huge).
    pub fn f32s(&mut self, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| match self.next() % 16 {
                0 => [0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 1e-40, -1e-42, 3e38, -3e38][i % 9],
                1 => f32::from_bits(self.next() as u32),
                _ => ((self.next() >> 11) as f64 / (1u64 << 53) as f64 * 16.0 - 8.0) as f32,
            })
            .collect()
    }
    pub fn f64s(&mut self, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| match self.next() % 16 {
                0 => [0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 1e-310, 1e308][i % 7],
                1 => f64::from_bits(self.next()),
                _ => (self.next() >> 11) as f64 / (1u64 << 53) as f64 * 16.0 - 8.0,
            })
            .collect()
    }
    /// Raw 16-bit patterns: every f16/bf16 value class shows up (incl. nan, inf, denormals).
    pub fn b16s(&mut self, n: usize) -> Vec<u16> {
        (0..n).map(|_| self.next() as u16).collect()
    }
}

pub fn as_bytes<T: Copy>(v: &[T]) -> Vec<u8> {
    let n = std::mem::size_of_val(v);
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n).to_vec() }
}

/// Candle's `info` array: dims then strides, as usize (u64).
pub fn layout_info(dims: &[usize], strides: &[usize]) -> Vec<u8> {
    let v: Vec<u64> = dims.iter().chain(strides).map(|&x| x as u64).collect();
    as_bytes(&v)
}

/// Contiguous strides for `dims`.
pub fn contiguous(dims: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * dims[i + 1];
    }
    s
}

/// Running tally for a module's gate.
#[derive(Default)]
pub struct Tally {
    pub launches: usize,
    pub bytes: usize,
    pub failures: Vec<String>,
}

impl Tally {
    pub fn record(&mut self, label: &str, d: &Diff) {
        self.launches += 1;
        self.bytes += d.bytes;
        if d.differing > 0 {
            let (b, off, r, o) = d.first.unwrap();
            self.failures.push(format!("{label}: {} of {} bytes differ (first: buf {b} byte {off}: ref {r:#04x} oxide {o:#04x})", d.differing, d.bytes));
        }
    }
    pub fn finish(&self, module: &str) -> bool {
        for f in self.failures.iter().take(40) {
            println!("  FAIL {f}");
        }
        let ok = self.failures.is_empty();
        println!("{module}: {} launches, {} bytes compared, {} failing -> {}", self.launches, self.bytes, self.failures.len(),
                 if ok { "PASS (bit-identical)" } else { "FAIL" });
        ok
    }
}
