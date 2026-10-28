// Authoritative reference for the i-quant fused-MoE path.
//
// The diff harness can only compare against nvcc PTX, and no nvcc reference exists for
// indexed_moe_forward_iq*_q8_1 (candle upstream has only the six built-in quant types). So we
// compile one here: candle's own indexed_moe_forward template (candle-kernels/src/quantized.cu,
// verbatim semantics: grid/block mapping, expert stride, kby/kqs, warp reduction) instantiated
// with llama.cpp's vec_dot_iq*_q8_1 and the real block_iq* structs from ggml-common.h.
//
// This is the contract the ported arms must match, so the arms are gated against the reference
// math rather than against my recollection of it.
#include "cuda_fp16.h"
#include <cstdint>
#include <stdio.h>

#define QK_K   256
#define QK8_1  32
#define WARP_SIZE 32

typedef __half   ggml_half;   // ggml-common.h GGML_COMMON_DECL_CUDA: `typedef half ggml_half;`
typedef __half2  ggml_half2;

// ---------------------------------------------------------------- block structs (ggml-common.h)
typedef struct { ggml_half d; uint16_t qs[QK_K/8]; } block_iq2_xxs;
typedef struct { ggml_half d; uint16_t qs[QK_K/8]; uint8_t scales[QK_K/32]; } block_iq2_xs;
typedef struct { ggml_half d; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/32]; uint8_t scales[QK_K/32]; } block_iq2_s;
typedef struct { ggml_half d; uint8_t qs[3*QK_K/8]; } block_iq3_xxs;
typedef struct { ggml_half d; uint16_t scales_h; uint8_t scales_l[QK_K/64]; uint8_t qs[QK_K/2]; } block_iq4_xs;
#define QK4_NL 32
typedef struct { ggml_half d; uint8_t qs[QK4_NL/2]; } block_iq4_nl;
typedef struct {
    union {
        struct { ggml_half d; ggml_half s; };
        ggml_half2 ds;
    };
    uint8_t qs[QK8_1];
} block_q8_1;

#include "iq_ref_tables.h"

// ------------------------------------------------------------------------ get_int_* (vecdotq.cuh)
static __device__ __forceinline__ int get_int_b2(const void * x, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) x;
    int x32 = x16[2*i32 + 0] << 0;
    x32 |= x16[2*i32 + 1] << 16;
    return x32;
}
static __device__ __forceinline__ int get_int_b4(const void * x, const int & i32) {
    return ((const int *) x)[i32];
}
static __device__ __forceinline__ int get_int_b1(const void * x, const int & i32) {
    const uint8_t * x8 = (const uint8_t *) x;
    int x32 = x8[4*i32 + 0] << 0;
    x32 |= x8[4*i32 + 1] << 8;
    x32 |= x8[4*i32 + 2] << 16;
    x32 |= x8[4*i32 + 3] << 24;
    return x32;
}
// common.cuh
// vecdotq.cuh: returns the byte broadcast over the word (`s * 0x01010101`) so that
// 0x08040201 / 0x80402010 act as per-lane selectors. A scalar-byte version makes both
// selectors zero, so NO sign is ever applied.
static __device__ __forceinline__ uint32_t unpack_ksigns(uint32_t v) {
    const uint32_t p = __popc(v & 0x7f) & 1;
    const uint32_t s = (v & 0x7f) ^ (p << 7);
    return s * 0x01010101u;
}
static __device__ __forceinline__ int ggml_cuda_dp4a(int a, int b, int c) { return __dp4a(a, b, c); }
static __device__ __forceinline__ int2 get_int_from_table_16(const int & q4, const int8_t * table) {
    const uint32_t * table32 = (const uint32_t *) table;
    const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
    uint32_t tmp[2];
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low  = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

// ---------------------------------------------------------------- vec_dot_* (vecdotq.cuh, verbatim)
static __device__ __forceinline__ float vec_dot_iq2_xxs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    const block_iq2_xxs * bq2 = (const block_iq2_xxs *) vbq + kbx;
    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t * aux8 = (const uint8_t *) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = ((const uint2*)iq2xxs_grid)[aux8[k0/2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, k0 + 0);
        sumi = ggml_cuda_dp4a(grid0, u0, sumi);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, k0 + 1);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

static __device__ __forceinline__ float vec_dot_iq2_xs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    const block_iq2_xs * bq2 = (const block_iq2_xs *) vbq + kbx;
    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const uint16_t * q2 = (const uint16_t *) &q2_packed;
    const int ls0 = bq2->scales[iqs/2] & 0x0F;
    const int ls1 = bq2->scales[iqs/2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = ((const uint2*)iq2xs_grid)[q2[l0/2] & 0x1FF];
        const uint32_t signs = unpack_ksigns(q2[l0/2] >> 9);
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);
        if (l0 < 4) { sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0); sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0); }
        else        { sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1); sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1); }
    }
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

static __device__ __forceinline__ float vec_dot_iq2_s_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    const block_iq2_s * bq2 = (const block_iq2_s *) vbq + kbx;
    const int qs_packed = get_int_b2(bq2->qs, iqs/2);
    const uint8_t * qs = (const uint8_t *) &qs_packed;
    const int qh = bq2->qh[iqs/2];
    const int signs_packed_32 = get_int_b2(bq2->qs, QK_K/32 + iqs/2);
    const uint8_t * signs_packed_8 = (const uint8_t *) &signs_packed_32;
    const int ls0 = bq2->scales[iqs/2] & 0x0F;
    const int ls1 = bq2->scales[iqs/2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int * grid_pos = (const int *)(iq2s_grid + (qs[l0/2] | ((qh << (8-l0)) & 0x300)));
        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4(grid_pos[0] ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos[1] ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);
        if (l0 < 4) { sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0); sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0); }
        else        { sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1); sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1); }
    }
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

static __device__ __forceinline__ float vec_dot_iq3_xxs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    const block_iq3_xxs * bq3 = (const block_iq3_xxs *) vbq + kbx;
    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs+1));
    const uint8_t * q3 = (const uint8_t *) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K/16 + iqs/2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7*l0/2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls*sumi + sumi/2)/2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

static __device__ __forceinline__ float vec_dot_iq4_xs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    const block_iq4_xs * bq4 = (const block_iq4_xs *) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int u0 = get_int_b4(bq8_1[iqs/4].qs, j + 0);
        const int u1 = get_int_b4(bq8_1[iqs/4].qs, j + 4);
        sumi = ggml_cuda_dp4a(v.x, u0, sumi);
        sumi = ggml_cuda_dp4a(v.y, u1, sumi);
    }
    const int ls = ((bq4->scales_l[iqs/8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs/2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d = __half2float(bq4->d) * __low2float(bq8_1[iqs/4].ds);
    return d * sumi;
}

// vecdotq.cuh vec_dot_iq4_nl_q8_1 (VDR_IQ4_NL_Q8_1_MMVQ = 2), llama.cpp acecd56.
static __device__ __forceinline__ float vec_dot_iq4_nl_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    const block_iq4_nl * bq4 = (const block_iq4_nl *) vbq + kbx;
    const int * q8 = (const int *) bq8_1->qs + iqs;
    int sumi = 0;
#pragma unroll
    for (int l = 0; l < 2; ++l) {
        const int aux_q4 = get_int_b2(bq4->qs, iqs + l);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
    }
    const float d = __half2float(bq4->d) * __low2float(bq8_1->ds);
    return d * sumi;
}

// ------------------------------------------------------------- candle's indexed_moe_forward (verbatim)
static __device__ __forceinline__ float warp_reduce_sum_ref(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    return x;
}

template <int qk, int qi, typename block_q_t, int vdr, float (*vec_dot)(const void*, const block_q8_1*, const int&, const int&)>
__device__ void indexed_moe_forward_ref(
    const void * __restrict__ all_weights, const void * __restrict__ all_inputs,
    const unsigned int * __restrict__ indices, float * __restrict__ all_outputs,
    const int n, const int k, int /*batch*/, int /*topk*/, const int k_padded, const int input_dim1) {
    const int current_batch = blockIdx.y;
    const int current_topk = blockIdx.z;
    const int task_id = current_batch * gridDim.z + current_topk;
    if (task_id >= gridDim.y * gridDim.z) return;
    const int input_idx = (input_dim1 == 1) ? current_batch : task_id;
    const unsigned int expert_id = indices[task_id];
    const size_t weight_block_size = sizeof(block_q_t);
    const size_t input_block_size = sizeof(block_q8_1);
    const size_t weight_expert_stride_bytes = (size_t)(n * k) / qk * weight_block_size;
    const size_t input_task_stride_bytes = (size_t)k_padded / QK8_1 * input_block_size;
    const void * current_input_ptr  = (const char *)all_inputs  + input_idx * input_task_stride_bytes;
    const void * current_weight_ptr = (const char *)all_weights + expert_id * weight_expert_stride_bytes;
    float * current_output_ptr = all_outputs + task_id * n;
    constexpr int nwarps = 4;
    const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
    const int row0 = blockIdx.x;
    if (row0 >= n) return;
    const int blocks_per_row_x = k / qk;
    constexpr int blocks_per_iter = vdr * nwarps * WARP_SIZE / qi;
    float tmp = 0.0f;
    const block_q_t * w = (const block_q_t *) current_weight_ptr;
    const block_q8_1 * x = (const block_q8_1 *) current_input_ptr;
    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (tid % (qi / vdr));
        tmp += vec_dot(&w[kbx + row0 * blocks_per_row_x], &x[kby], 0, kqs);
    }
    __shared__ float tmp_shared[nwarps - 1][WARP_SIZE];
    if (threadIdx.y > 0) tmp_shared[threadIdx.y - 1][threadIdx.x] = tmp;
    __syncthreads();
    if (threadIdx.y == 0) {
        for (int l = 0; l < nwarps - 1; ++l) tmp += tmp_shared[l][threadIdx.x];
        tmp = warp_reduce_sum_ref(tmp);
        if (threadIdx.x == 0) current_output_ptr[row0] = tmp;
    }
}

#define MOE_ENTRY(suffix, qk, qi, blk, vdr, fn)                                              \
extern "C" __global__ void indexed_moe_forward_##suffix(                                     \
    const void * __restrict__ all_weights, const void * __restrict__ all_inputs,             \
    const unsigned int * __restrict__ indices, float * __restrict__ all_outputs,             \
    const int n, const int k, const int batch, const int topk, const int k_padded,           \
    const int input_dim1) {                                                                  \
    indexed_moe_forward_ref<qk, qi, blk, vdr, fn>(all_weights, all_inputs, indices,          \
        all_outputs, n, k, batch, topk, k_padded, input_dim1);                               \
}

MOE_ENTRY(iq2_xxs_q8_1, QK_K, 16, block_iq2_xxs, 2, vec_dot_iq2_xxs_q8_1)
MOE_ENTRY(iq2_xs_q8_1,  QK_K, 16, block_iq2_xs,  2, vec_dot_iq2_xs_q8_1)
MOE_ENTRY(iq2_s_q8_1,   QK_K, 16, block_iq2_s,   2, vec_dot_iq2_s_q8_1)
MOE_ENTRY(iq3_xxs_q8_1, QK_K, 16, block_iq3_xxs, 2, vec_dot_iq3_xxs_q8_1)
MOE_ENTRY(iq4_xs_q8_1,  QK_K, 32, block_iq4_xs,  4, vec_dot_iq4_xs_q8_1)
MOE_ENTRY(iq4_nl_q8_1,  QK4_NL, 4, block_iq4_nl, 2, vec_dot_iq4_nl_q8_1)

// Per-block isolation entry: mirrors the real caller's pointers exactly
// (`&w[kbx]`, `&x[kby]`, third arg 0, then `bq8_1[iqs/2]`), so a single vec_dot value can be
// compared against the port for one (kbx, iqs) pair. Used to localize index bugs.
#define DBG_ENTRY(suffix, blk, fn)                                                          \
extern "C" __global__ void iq_dbg_vdot_##suffix(const void * __restrict__ w,                \
    const void * __restrict__ x, float * __restrict__ out, const int kbx, const int iqs) {   \
    if (threadIdx.x == 0) {                                                                  \
        const char * wb = (const char *) w + (size_t) kbx * sizeof(blk);                     \
        const char * xb = (const char *) x + (size_t) kbx * (QK_K/QK8_1) * sizeof(block_q8_1); \
        out[0] = fn((const void *) wb, (const block_q8_1 *) xb, 0, iqs);                     \
    }                                                                                        \
}

DBG_ENTRY(iq2_xxs_q8_1, block_iq2_xxs, vec_dot_iq2_xxs_q8_1)

// Piece-by-piece dump for IQ2_XXS so a mismatch can be localized to a component.
extern "C" __global__ void iq_probe_iq2_xxs(const void * __restrict__ w, const void * __restrict__ x,
                                            float * __restrict__ out, const int kbx, const int iqs) {
    if (threadIdx.x == 0) {
        const block_iq2_xxs * bq2 = (const block_iq2_xxs *) w + kbx;
        const block_q8_1  * bq8 = (const block_q8_1 *) ((const char *) x + (size_t) kbx * 8 * (int)sizeof(block_q8_1));
        const int q2 = get_int_b2(bq2->qs, iqs);
        const uint8_t * aux8 = (const uint8_t *) &q2;
        const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
        int sumi = 0;
        int first_grid = 0;
        for (int k0 = 0; k0 < 8; k0 += 2) {
            const uint2 grid_pos = ((const uint2*)iq2xxs_grid)[aux8[k0/2]];
            if (k0 == 0) first_grid = (int)grid_pos.x;
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);
            sumi = ggml_cuda_dp4a(grid0, get_int_b4(bq8[iqs/2].qs, k0 + 0), sumi);
            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);
            sumi = ggml_cuda_dp4a(grid1, get_int_b4(bq8[iqs/2].qs, k0 + 1), sumi);
        }
        const int ls = aux32 >> 27 | 1;
        const float d = __half2float(bq2->d) * __low2float(bq8[iqs/2].ds);
        out[0] = d * (sumi * ls / 8);
        out[1] = (float)sumi;
        out[2] = d;
        out[3] = (float)aux32;
        out[4] = (float)aux8[0];
        out[5] = (float)first_grid;
        out[6] = (float)(int)bq8[iqs/2].qs[0];
        out[7] = (float)(int)bq8[iqs/2].qs[4];
        out[8] = (float)(int)__half_as_ushort(bq2->d);      // raw half bits of the weight scale
        out[9] = (float)(int)((const unsigned short *)&bq8[iqs/2].ds)[0]; // raw bits of q8 d
        out[10] = __half2float(bq2->d);                    // weight scale as float
        out[11] = __low2float(bq8[iqs/2].ds);              // q8 scale as float
        {   // k0 = 0 iteration internals
            const uint2 gp0 = ((const uint2*)iq2xxs_grid)[aux8[0]];
            const uint32_t sg0 = unpack_ksigns(aux32 >> 0);
            out[12] = (float)(int)gp0.y;                   // high grid word
            out[13] = (float)(int)sg0;                     // signs (broadcast) for k0=0
            const int vc = __vcmpne4(sg0 & 0x08040201, 0);
            out[14] = (float)vc;                           // __vcmpne4 result
            const int gv = __vsub4(gp0.x ^ vc, vc);
            out[15] = (float)gv;                           // grid word 0 after sign
            out[16] = (float)ggml_cuda_dp4a(gv, get_int_b4(bq8[iqs/2].qs, 0), 0); // single dp4a
        }
    }
}
DBG_ENTRY(iq2_xs_q8_1,  block_iq2_xs,  vec_dot_iq2_xs_q8_1)
DBG_ENTRY(iq2_s_q8_1,   block_iq2_s,   vec_dot_iq2_s_q8_1)
DBG_ENTRY(iq3_xxs_q8_1, block_iq3_xxs, vec_dot_iq3_xxs_q8_1)
DBG_ENTRY(iq4_xs_q8_1,  block_iq4_xs,  vec_dot_iq4_xs_q8_1)
