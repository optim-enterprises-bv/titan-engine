//! titan: the GDN launchers at the ABI the fork's hybrid GGUF models were written and gated
//! against (mistral.rs 84b53bf). Upstream v0.9.4 changed these entry points (in-place pooled state,
//! V-major layout, state index tables) under the same C names. In the `oxide` build the plain names
//! are the bit-identical cuda-oxide twins exported by titan-oxide-ffi; in an nvcc build they are the
//! 84b53bf kernels compiled from `cuda/titan_gdn_legacy.cu` under `titan_legacy_*` names.
#![allow(dead_code)]
use std::ffi::c_void;

extern "C" {
    #[cfg_attr(not(feature = "oxide"), link_name = "titan_legacy_gated_delta_rule_recurrence")]
    pub(crate) fn gated_delta_rule_recurrence(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        output: *mut f32,
        bh: i32,
        seq_len: i32,
        k_dim: i32,
        v_dim: i32,
        stream: i64,
    );
    #[cfg_attr(not(feature = "oxide"), link_name = "titan_legacy_warp_gated_delta_rule_recurrence")]
    pub(crate) fn warp_gated_delta_rule_recurrence(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        output: *mut f32,
        bh: i32,
        seq_len: i32,
        k_dim: i32,
        v_dim: i32,
        stream: i64,
    );
    // Chunked GDN recurrence for prefill (processes tokens in BT=64 chunks)
    #[cfg_attr(not(feature = "oxide"), link_name = "titan_legacy_chunked_gated_delta_rule_recurrence")]
    pub(crate) fn chunked_gated_delta_rule_recurrence(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        output: *mut f32,
        bh: i32,
        seq_len: i32,
        k_dim: i32,
        v_dim: i32,
        stream: i64,
    );
    #[cfg_attr(not(feature = "oxide"), link_name = "titan_legacy_causal_conv1d_update")]
    pub(crate) fn causal_conv1d_update(
        x: *const c_void,
        weight: *const c_void,
        conv_state: *mut c_void,
        output: *mut c_void,
        batch_size: i32,
        conv_dim: i32,
        kernel_size: i32,
        dtype: i32,
        stream: i64,
    );
    #[cfg_attr(not(feature = "oxide"), link_name = "titan_legacy_causal_conv1d_full")]
    pub(crate) fn causal_conv1d_full(
        x: *const c_void,
        weight: *const c_void,
        conv_state_out: *mut c_void,
        output: *mut c_void,
        batch_size: i32,
        conv_dim: i32,
        seq_len: i32,
        kernel_size: i32,
        dtype: i32,
        stream: i64,
    );
    #[cfg_attr(not(feature = "oxide"), link_name = "titan_legacy_gdn_rmsnorm_gated")]
    pub(crate) fn gdn_rmsnorm_gated(
        x: *const c_void,
        gate: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        rows: i32,
        hidden_dim: i32,
        eps: f32,
        dtype: i32,
        stream: i64,
    );
    #[cfg_attr(not(feature = "oxide"), link_name = "titan_legacy_fused_gdn_gating")]
    pub(crate) fn fused_gdn_gating(
        b: *const c_void,
        a: *const c_void,
        a_log: *const f32,
        dt_bias: *const f32,
        beta_out: *mut c_void,
        g_out: *mut c_void,
        total_elements: i32,
        num_heads: i32,
        dtype: i32,
        stream: i64,
    );
    #[cfg_attr(not(feature = "oxide"), link_name = "titan_legacy_gdn_prepare_recurrence")]
    pub(crate) fn gdn_prepare_recurrence(
        mixed_qkv: *const c_void,
        b: *const c_void,
        a: *const c_void,
        a_log: *const f32,
        dt_bias: *const f32,
        q_out: *mut f32,
        k_out: *mut f32,
        v_out: *mut f32,
        g_out: *mut f32,
        beta_out: *mut f32,
        batch_size: i32,
        seq_len: i32,
        num_k_heads: i32,
        num_v_heads: i32,
        head_k_dim: i32,
        head_v_dim: i32,
        dtype: i32,
        stream: i64,
    );

    #[cfg_attr(not(feature = "oxide"), link_name = "titan_legacy_gdn_decode_recurrence")]
    pub(crate) fn gdn_decode_recurrence(
        mixed_qkv: *const c_void,
        b: *const c_void,
        a: *const c_void,
        a_log: *const f32,
        dt_bias: *const f32,
        state: *mut f32,
        output: *mut f32,
        batch_size: i32,
        num_k_heads: i32,
        num_v_heads: i32,
        head_k_dim: i32,
        head_v_dim: i32,
        dtype: i32,
        stream: i64,
    );
}
