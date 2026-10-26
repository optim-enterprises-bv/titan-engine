//! Oxide (nvcc-free) blockwise E4M3 W8A16 GEMM: the cuda-oxide port of the cuTile `fp8_w8a16_block_bf16` path,
//! for the oxide build where cuTile is absent. Tensor-core (`mma.sync.m16n8k16.f32.bf16.bf16`), FP8 weights
//! converted to bf16 on load, per-[128,128] block f32 scale. Kernel source + verification: oxide-kernels/fp8-blockwise.

use candle_core::{
    cuda::{
        cudarc::driver::{LaunchConfig, PushKernelArg},
        CudaStorage, WrapErr,
    },
    Device, Result, Shape, Storage, Tensor,
};
use float8::F8E4M3;

use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

const PTX: &str = include_str!("fp8_blockwise_oxide.ptx");
const MODULE: &str = "fp8_blockwise_oxide";
const BLOCK: usize = 128;

/// y[M,N] = x[M,K] @ dequant_e4m3(w[N,K])^T, with a per-[128,128] weight-block f32 scale `ws[nmb,nkb]`.
/// x: bf16 [M,K] contiguous. w: F8E4M3 [N,K] contiguous. ws: f32 [ceil(N/128), ceil(K/128)] contiguous. Out: bf16 [M,N].
pub fn fp8_w8a16_block_mma(x: &Tensor, w: &Tensor, ws: &Tensor) -> Result<Tensor> {
    let (m, k) = x.dims2()?;
    let (n, kw) = w.dims2()?;
    if k != kw {
        candle_core::bail!("fp8 oxide gemm: x K={k} != w K={kw}");
    }
    let Device::Cuda(dev) = x.device().clone() else {
        candle_core::bail!("fp8 oxide gemm: operands must be CUDA tensors")
    };
    let stream = dev.cuda_stream();

    let (x_storage, x_layout) = x.storage_and_layout();
    let (w_storage, w_layout) = w.storage_and_layout();
    let (s_storage, s_layout) = ws.storage_and_layout();
    let (Storage::Cuda(x_cuda), Storage::Cuda(w_cuda), Storage::Cuda(s_cuda)) =
        (&*x_storage, &*w_storage, &*s_storage)
    else {
        candle_core::bail!("fp8 oxide gemm: operands must be CUDA tensors")
    };
    let (x_addr, _xg) = slice_ptr_on_stream(x_cuda.as_cuda_slice::<half::bf16>()?, x_layout.start_offset(), &stream);
    let (w_addr, _wg) = slice_ptr_on_stream(w_cuda.as_cuda_slice::<F8E4M3>()?, w_layout.start_offset(), &stream);
    let (s_addr, _sg) = slice_ptr_on_stream(s_cuda.as_cuda_slice::<f32>()?, s_layout.start_offset(), &stream);

    let mut out = unsafe { dev.alloc::<half::bf16>(m * n) }?;
    let (out_addr, out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);

    let f = dev.get_or_load_custom_func("fp8_w8a16_block_bf16_mma", MODULE, PTX)?;
    let (mi, ni, ki, bsi) = (m as i32, n as i32, k as i32, BLOCK as i32);
    let ntm = m.div_ceil(16) as u32;
    let ntn = n.div_ceil(8) as u32;
    let mut b = f.builder();
    b.arg(&out_addr).arg(&x_addr).arg(&w_addr).arg(&s_addr).arg(&mi).arg(&ni).arg(&ki).arg(&bsi);
    unsafe {
        b.launch(LaunchConfig { grid_dim: (ntm * ntn, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 })
    }
    .w()?;
    drop(out_guard);

    Ok(Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
        Shape::from_dims(&[m, n]),
    )))
}

/// FP8 W8A16 grouped MoE GEMM, the oxide port of candle-moe's `grouped` path for FP8 weights.
/// `input` bf16 [rows, K]; `w` F8E4M3 [E, N, K]; `w_scales` f32 [E, N/block, K/block]; `sorted`
/// u32 [num_valid] (argsort permutation, the flat routed index per routed row); `expert_ids` u32
/// [num_valid] (sorted expert per routed row); `tw` Some(f32 [num_valid]) applies the top-k weight
/// and indexes `input` directly (down projection), None divides the sorted value by topk (gate_up).
/// Output bf16 [num_valid, N] indexed by the sorted value. Alignment offsets are built on device
/// (moe_count + moe_scan), matching the engine's bf16 moe_gemm path.
#[allow(clippy::too_many_arguments)]
pub fn fp8_moe_grouped(
    input: &Tensor,
    w: &Tensor,
    w_scales: &Tensor,
    tw: Option<&Tensor>,
    sorted: &Tensor,
    expert_ids: &Tensor,
    topk: usize,
    block: usize,
) -> Result<Tensor> {
    let (num_experts, size_n, size_k) = w.dims3()?;
    let num_valid = sorted.dims1()?;
    let Device::Cuda(dev) = w.device().clone() else {
        candle_core::bail!("fp8 oxide moe: operands must be CUDA tensors")
    };
    let stream = dev.cuda_stream();

    let (in_s, in_l) = input.storage_and_layout();
    let (w_s, w_l) = w.storage_and_layout();
    let (ws_s, ws_l) = w_scales.storage_and_layout();
    let (so_s, so_l) = sorted.storage_and_layout();
    let (ei_s, ei_l) = expert_ids.storage_and_layout();
    let (Storage::Cuda(in_c), Storage::Cuda(w_c), Storage::Cuda(ws_c), Storage::Cuda(so_c), Storage::Cuda(ei_c)) =
        (&*in_s, &*w_s, &*ws_s, &*so_s, &*ei_s)
    else {
        candle_core::bail!("fp8 oxide moe: operands must be CUDA tensors")
    };
    let (in_addr, _ig) = slice_ptr_on_stream(in_c.as_cuda_slice::<half::bf16>()?, in_l.start_offset(), &stream);
    let (w_addr, _wg) = slice_ptr_on_stream(w_c.as_cuda_slice::<F8E4M3>()?, w_l.start_offset(), &stream);
    let (ws_addr, _wsg) = slice_ptr_on_stream(ws_c.as_cuda_slice::<f32>()?, ws_l.start_offset(), &stream);
    let (so_addr, _sog) = slice_ptr_on_stream(so_c.as_cuda_slice::<u32>()?, so_l.start_offset(), &stream);
    let (ei_addr, _eig) = slice_ptr_on_stream(ei_c.as_cuda_slice::<u32>()?, ei_l.start_offset(), &stream);

    let tw_pair = tw.map(|t| t.storage_and_layout());
    let mut _tw_guard = None;
    let tw_addr: u64 = if let Some((tw_s, tw_l)) = tw_pair.as_ref() {
        let Storage::Cuda(tw_c) = &**tw_s else {
            candle_core::bail!("fp8 oxide moe: topk_weights must be CUDA")
        };
        let (addr, g) = slice_ptr_on_stream(tw_c.as_cuda_slice::<f32>()?, tw_l.start_offset(), &stream);
        _tw_guard = Some(g);
        addr
    } else {
        0
    };
    let has_tw: i32 = if tw.is_some() { 1 } else { 0 };

    let mut counts = dev.alloc_zeros::<i32>(num_experts)?;
    let mut offsets = unsafe { dev.alloc::<i32>(num_experts + 1) }?;
    let (cnt_addr, cnt_guard) = slice_ptr_mut_on_stream(&mut counts, 0, &stream);
    let (off_addr, off_guard) = slice_ptr_mut_on_stream(&mut offsets, 0, &stream);

    let f_cnt = dev.get_or_load_custom_func("moe_count", MODULE, PTX)?;
    let f_scan = dev.get_or_load_custom_func("moe_scan", MODULE, PTX)?;
    let (nv_i, ne_i) = (num_valid as i32, num_experts as i32);
    let grid_cnt = (num_valid.div_ceil(256) as u32).max(1);
    let mut cb = f_cnt.builder();
    cb.arg(&ei_addr).arg(&cnt_addr).arg(&nv_i);
    unsafe { cb.launch(LaunchConfig { grid_dim: (grid_cnt, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }.w()?;
    let mut sb = f_scan.builder();
    sb.arg(&cnt_addr).arg(&off_addr).arg(&ne_i);
    unsafe { sb.launch(LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 }) }.w()?;

    let mut out = unsafe { dev.alloc::<half::bf16>(num_valid * size_n) }?;
    let (out_addr, out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);

    let f = dev.get_or_load_custom_func("fp8_moe_grouped", MODULE, PTX)?;
    let (tk_i, nn_i, kk_i, bs_i) = (topk as i32, size_n as i32, size_k as i32, block as i32);
    let ntn = size_n.div_ceil(8) as u32;
    let mut b = f.builder();
    b.arg(&out_addr).arg(&in_addr).arg(&w_addr).arg(&ws_addr).arg(&so_addr).arg(&off_addr).arg(&tw_addr)
        .arg(&ne_i).arg(&tk_i).arg(&has_tw).arg(&nn_i).arg(&kk_i).arg(&bs_i);
    unsafe {
        b.launch(LaunchConfig { grid_dim: (num_experts as u32, ntn, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 })
    }
    .w()?;
    drop((cnt_guard, off_guard, out_guard));

    Ok(Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
        Shape::from_dims(&[num_valid, size_n]),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn oxide_fp8_w8a16_block_matches_dequant() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let (m, n, k, bs) = (48usize, 256usize, 256usize, 128usize);
        let (nmb, nkb) = (n.div_ceil(bs), k.div_ceil(bs));

        // Random-ish deterministic inputs.
        let mut s: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut nxt = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (s >> 33) as u32 as f32 / (1u64 << 31) as f32
        };
        let xv: Vec<half::bf16> = (0..m * k).map(|_| half::bf16::from_f32(nxt() * 4.0 - 2.0)).collect();
        let wv: Vec<F8E4M3> = (0..n * k).map(|_| F8E4M3::from_f32(nxt() * 2.0 - 1.0)).collect();
        let sv: Vec<f32> = (0..nmb * nkb).map(|_| 0.25 + nxt() * 1.5).collect();

        let x = Tensor::from_vec(xv.clone(), (m, k), &dev)?;
        let w = Tensor::from_vec(wv.clone(), (n, k), &dev)?;
        let ws = Tensor::from_vec(sv.clone(), (nmb, nkb), &dev)?;

        let y = fp8_w8a16_block_mma(&x, &w, &ws)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;

        // CPU reference.
        let atol = 0.01 * (k as f32).sqrt();
        let mut fails = 0;
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = 0f32;
                for ki in 0..k {
                    acc += xv[mi * k + ki].to_f32()
                        * wv[ni * k + ki].to_f32()
                        * sv[(ni / bs) * nkb + ki / bs];
                }
                let got = y[mi][ni];
                if (got - acc).abs() > atol + 0.02 * acc.abs() {
                    fails += 1;
                }
            }
        }
        assert_eq!(fails, 0, "oxide fp8 gemm mismatched {fails} elements");
        Ok(())
    }
}
