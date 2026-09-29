//! titan `oxide` build: adapters for mistralrs-quant launchers whose C ABI changed between
//! mistral.rs 84b53bf and v0.9.4 (see mistralrs-core/src/titan_oxide.rs). Each keeps the v0.9.4
//! Rust name and signature and calls the 84b53bf kernel that titan-oxide-ffi exports under the C
//! name, for the argument shapes where both compute the same thing.
#![allow(clippy::too_many_arguments)]

use std::ffi::c_void;

use candle_core::cuda::cudarc::driver::sys::CUstream;

extern "C" {
    #[link_name = "fused_glu_f16"]
    fn fused_glu_f16_v084(a: *const c_void, b: *const c_void, o: *mut c_void, n: u32, act: i32, s: CUstream);
    #[link_name = "fused_glu_bf16"]
    fn fused_glu_bf16_v084(a: *const c_void, b: *const c_void, o: *mut c_void, n: u32, act: i32, s: CUstream);
    #[link_name = "fused_glu_f32"]
    fn fused_glu_f32_v084(a: *const c_void, b: *const c_void, o: *mut c_void, n: u32, act: i32, s: CUstream);

    #[link_name = "launch_moe_weighted_reduce_flat"]
    fn moe_weighted_reduce_flat_v084(
        inputs: *const c_void,
        topk_weights: *const f32,
        outputs: *mut c_void,
        num_tokens: i32,
        hidden: i32,
        topk: i32,
        stream: *mut c_void,
    );
    #[link_name = "launch_moe_weighted_reduce_flat_bf16"]
    fn moe_weighted_reduce_flat_bf16_v084(
        inputs: *const c_void,
        topk_weights: *const f32,
        outputs: *mut c_void,
        num_tokens: i32,
        hidden: i32,
        topk: i32,
        stream: *mut c_void,
    );
}

fn nvcc_only(what: &str) -> ! {
    eprintln!(
        "titan oxide build: {what} needs an nvcc-only kernel variant that is not ported to cuda-oxide"
    );
    std::process::abort()
}

/// The 84b53bf fused GLU takes one dense run of `n` elements: rows must be packed.
fn dense_len(rows: u32, cols: u32, a_row_stride: u32, b_row_stride: u32) -> u32 {
    if (a_row_stride != cols || b_row_stride != cols) && rows > 1 {
        nvcc_only("fused_glu over strided rows");
    }
    rows.checked_mul(cols).unwrap_or_else(|| nvcc_only("fused_glu over more than u32::MAX elements"))
}

pub unsafe extern "C" fn fused_glu_f16(
    a: *const c_void,
    b: *const c_void,
    output: *mut c_void,
    rows: u32,
    cols: u32,
    a_row_stride: u32,
    b_row_stride: u32,
    activation: i32,
    stream: CUstream,
) {
    fused_glu_f16_v084(a, b, output, dense_len(rows, cols, a_row_stride, b_row_stride), activation, stream)
}

pub unsafe extern "C" fn fused_glu_bf16(
    a: *const c_void,
    b: *const c_void,
    output: *mut c_void,
    rows: u32,
    cols: u32,
    a_row_stride: u32,
    b_row_stride: u32,
    activation: i32,
    stream: CUstream,
) {
    fused_glu_bf16_v084(a, b, output, dense_len(rows, cols, a_row_stride, b_row_stride), activation, stream)
}

pub unsafe extern "C" fn fused_glu_f32(
    a: *const c_void,
    b: *const c_void,
    output: *mut c_void,
    rows: u32,
    cols: u32,
    a_row_stride: u32,
    b_row_stride: u32,
    activation: i32,
    stream: CUstream,
) {
    fused_glu_f32_v084(a, b, output, dense_len(rows, cols, a_row_stride, b_row_stride), activation, stream)
}

/// v0.9.4 returns a CUDA status; the 84b53bf launcher (same F32-input kernel) returns nothing.
pub unsafe extern "C" fn launch_moe_weighted_reduce_flat(
    inputs: *const c_void,
    topk_weights: *const f32,
    outputs: *mut c_void,
    num_tokens: i32,
    hidden: i32,
    topk: i32,
    stream: *mut c_void,
) -> i32 {
    moe_weighted_reduce_flat_v084(inputs, topk_weights, outputs, num_tokens, hidden, topk, stream);
    0
}

pub unsafe extern "C" fn launch_moe_weighted_reduce_flat_bf16(
    inputs: *const c_void,
    topk_weights: *const f32,
    outputs: *mut c_void,
    num_tokens: i32,
    hidden: i32,
    topk: i32,
    stream: *mut c_void,
) -> i32 {
    moe_weighted_reduce_flat_bf16_v084(inputs, topk_weights, outputs, num_tokens, hidden, topk, stream);
    0
}
