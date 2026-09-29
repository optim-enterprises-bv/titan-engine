// kbench: llama.cpp (ggml CUDA backend, ~/ai/llama.cpp build) vs titan-engine cuda-oxide kernels on the
// exact matmul shapes of the 35B / 80B services. One process, one CUDA context, the SAME device buffers
// (weights, activations, routing ids) feed both sides, so outputs can be compared bit for bit.
//
// Timing: every measured run sits in an NVTX range "<side>|<case>|<i>" (side L = llama.cpp, O = ours,
// A = ours alternative kernel). Run under nsys; analyze.py sums the GPU kernels inside each range and
// takes the median over runs. Ours is also timed with CUDA events around the launches (printed here).
// Before each run a 256 MiB memset evicts L2, so weights are read cold, as in a real decode step.
//
// Usage: kbench [substring-filter ...]   (no filter: every case)
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// ---- titan-oxide-ffi launchers (the C symbols mistral.rs links; cuda-oxide PTX inside) ----
extern "C" {
void launch_mmvq_gguf_quantize_q8_1_f32(const void *x, void *vy, int kx, int kx_padded, int num_rows, void *stream);
void launch_mmvq_gguf_q8_0_f32_plain(const void *vx, const void *vy, void *dst, int ncols_x, int nrows_x,
                                     int stride_col_y, int stride_col_dst, int b_size, void *stream);
void launch_mmq_quantize_q8_1_D4(const void *x, const int *ids, void *vy, int type_x, int64_t ne00, int64_t s01,
                                 int64_t s02, int64_t s03, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, void *stream);
void launch_mmq_quantize_q8_1_DS4(const void *x, const int *ids, void *vy, int type_x, int64_t ne00, int64_t s01,
                                  int64_t s02, int64_t s03, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, void *stream);
typedef void (*mmq_fn)(void *tmp_fixup, const void *x, const void *y, void *dst, int64_t ncols_x, int64_t nrows_x,
                       int64_t ncols_y, int64_t stride_row_x, int64_t stride_col_dst, int cc, int nsm, int64_t smpbo,
                       int warp_size, int type_dst, void *stream);
typedef void (*mmq_moe_fn)(void *tmp_fixup, const void *x, const void *y, const int *ids_dst, const int *expert_bounds,
                           void *dst, int64_t ncols_x, int64_t nrows_x, int64_t ncols_dst, int64_t stride_row_x,
                           int64_t stride_col_dst, int64_t num_experts, int64_t ncols_max, int cc, int nsm,
                           int64_t smpbo, int warp_size, void *stream);
void launch_mmq_gguf_q8_0(void *, const void *, const void *, void *, int64_t, int64_t, int64_t, int64_t, int64_t, int,
                          int, int64_t, int, int, void *);
void launch_mmq_gguf_q4_k_moe(void *, const void *, const void *, const int *, const int *, void *, int64_t, int64_t,
                              int64_t, int64_t, int64_t, int64_t, int64_t, int, int, int64_t, int, void *);
void launch_mmq_gguf_q5_k_moe(void *, const void *, const void *, const int *, const int *, void *, int64_t, int64_t,
                              int64_t, int64_t, int64_t, int64_t, int64_t, int, int, int64_t, int, void *);
void launch_mmq_gguf_q6_k_moe(void *, const void *, const void *, const int *, const int *, void *, int64_t, int64_t,
                              int64_t, int64_t, int64_t, int64_t, int64_t, int, int, int64_t, int, void *);
}

#define CK(x)                                                                                      \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            fprintf(stderr, "CUDA %s at %s:%d: %s\n", cudaGetErrorString(e_), __FILE__, __LINE__, #x); \
            exit(2);                                                                               \
        }                                                                                          \
    } while (0)
#define CU(x)                                                                                      \
    do {                                                                                           \
        CUresult r_ = (x);                                                                         \
        if (r_ != CUDA_SUCCESS) {                                                                  \
            const char *s_ = nullptr;                                                              \
            cuGetErrorString(r_, &s_);                                                             \
            fprintf(stderr, "CU %s at %s:%d: %s\n", s_ ? s_ : "?", __FILE__, __LINE__, #x);        \
            exit(2);                                                                               \
        }                                                                                          \
    } while (0)

static std::string read_file(const char *p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) {
        fprintf(stderr, "cannot read %s\n", p);
        exit(2);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

enum Kind { DENSE, MOE, FFN };

struct Case {
    std::string name;
    Kind kind;
    ggml_type type;  // weight type (FFN: gate/up type)
    ggml_type type2; // FFN: down type
    int k, n, E, topk, B;
    int xr; // MOE: activation rows per token (1: gate/up share x, topk: down)
};

static const int NRUNS = 50, NWARM = 3;
// KBENCH_REP=<n>: appended to the side in NVTX tags and RES rows ("L2|case|i"), so repeated
// processes under one nsys session stay apart (bench/run.sh runs the kernel tier twice).
static std::string g_rep;

struct Ctx {
    ggml_backend_t be;
    cudaStream_t s;
    void *flush;
    size_t flush_bytes = 256u << 20;
    CUmodule titan, rows;
    int nsm, smpbo;
    void *fixup;
    int flush_val = 0;
};

static void l2_flush(Ctx &c) {
    CK(cudaMemsetAsync(c.flush, (c.flush_val++) & 0xff, c.flush_bytes, c.s));
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? NAN : v[v.size() / 2];
}

// Random weights: quantize `chunk_rows` random rows once, tile them over the whole tensor.
static void fill_weights(ggml_tensor *w, int k, int64_t total_rows, std::mt19937 &rng) {
    const size_t rs = ggml_row_size(w->type, k);
    const int chunk_rows = (int)std::min<int64_t>(total_rows, 512);
    std::normal_distribution<float> nd(0.f, 0.02f);
    std::vector<float> f((size_t)chunk_rows * k);
    for (auto &v : f) v = nd(rng);
    std::vector<uint8_t> q(rs * chunk_rows);
    ggml_quantize_chunk(w->type, f.data(), q.data(), 0, chunk_rows, k, nullptr);
    std::vector<uint8_t> all(ggml_nbytes(w));
    for (int64_t r = 0; r < total_rows; r += chunk_rows) {
        const int64_t m = std::min<int64_t>(chunk_rows, total_rows - r);
        memcpy(all.data() + r * rs, q.data(), m * rs);
    }
    ggml_backend_tensor_set(w, all.data(), 0, all.size());
}

static void fill_f32(ggml_tensor *t, std::mt19937 &rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> v(ggml_nelements(t));
    for (auto &x : v) x = nd(rng);
    ggml_backend_tensor_set(t, v.data(), 0, v.size() * 4);
}

static std::vector<int32_t> make_ids(int B, int topk, int E, std::mt19937 &rng) {
    std::vector<int32_t> ids((size_t)B * topk);
    std::vector<int> perm(E);
    for (int b = 0; b < B; b++) {
        for (int e = 0; e < E; e++) perm[e] = e;
        for (int j = 0; j < topk; j++) {
            std::uniform_int_distribution<int> u(j, E - 1);
            std::swap(perm[j], perm[u(rng)]);
            ids[(size_t)b * topk + j] = perm[j];
        }
    }
    return ids;
}

// llama.cpp side: run the graph NRUNS times, each in an NVTX range, L2 flushed before.
static double run_llama(Ctx &c, ggml_cgraph *gf, const std::string &name) {
    std::vector<double> wall;
    for (int i = 0; i < NWARM + NRUNS; i++) {
        l2_flush(c);
        CK(cudaStreamSynchronize(c.s));
        char tag[256];
        snprintf(tag, sizeof tag, "L%s|%s|%d", g_rep.c_str(), name.c_str(), i - NWARM);
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        if (i >= NWARM) nvtxRangePushA(tag);
        if (ggml_backend_graph_compute(c.be, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "graph compute failed\n");
            exit(3);
        }
        ggml_backend_synchronize(c.be);
        if (i >= NWARM) nvtxRangePop();
        clock_gettime(CLOCK_MONOTONIC, &t1);
        if (i >= NWARM) wall.push_back((t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3);
    }
    return median(wall);
}

// Ours: `launch` enqueues the whole op on c.s; CUDA events around it.
template <class F> static double run_ours(Ctx &c, const std::string &side, const std::string &name, int nruns, F launch) {
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    std::vector<double> ev;
    for (int i = 0; i < NWARM + nruns; i++) {
        l2_flush(c);
        CK(cudaStreamSynchronize(c.s));
        char tag[256];
        snprintf(tag, sizeof tag, "%s%s|%s|%d", side.c_str(), g_rep.c_str(), name.c_str(), i - NWARM);
        if (i >= NWARM) nvtxRangePushA(tag);
        CK(cudaEventRecord(e0, c.s));
        launch();
        CK(cudaEventRecord(e1, c.s));
        CK(cudaStreamSynchronize(c.s));
        if (i >= NWARM) nvtxRangePop();
        CK(cudaGetLastError());
        float ms;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        if (i >= NWARM) ev.push_back(ms * 1e3);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return median(ev);
}

static void launch_cu(CUfunction f, unsigned gx, unsigned gy, unsigned bx, unsigned by, cudaStream_t s, void **args) {
    CU(cuLaunchKernel(f, gx, gy, 1, bx, by, 1, 0, (CUstream)s, args, nullptr));
}

struct Cmp {
    size_t n = 0, eq = 0;
    double max_rel = 0;
};
static Cmp compare(const std::vector<float> &a, const std::vector<float> &b) {
    Cmp r;
    r.n = std::min(a.size(), b.size());
    double amax = 0;
    for (size_t i = 0; i < r.n; i++) amax = std::max(amax, (double)std::fabs(a[i]));
    for (size_t i = 0; i < r.n; i++) {
        if (memcmp(&a[i], &b[i], 4) == 0) r.eq++;
        r.max_rel = std::max(r.max_rel, std::fabs((double)a[i] - b[i]) / (amax > 0 ? amax : 1));
    }
    return r;
}

static const char *moe_kernel_name(ggml_type t, int k) {
    switch (t) {
    case GGML_TYPE_Q4_K: return k <= 512 ? "q4k_q8_1_moe_gemv_w" : "q4k_q8_1_moe_gemv";
    case GGML_TYPE_Q5_K: return k <= 512 ? "q5k_q8_1_moe_gemv_w" : "q5k_q8_1_moe_gemv";
    case GGML_TYPE_Q6_K: return "q6k_q8_1_moe_gemv";
    case GGML_TYPE_IQ4_NL: return "iq4_nl_q8_1_moe_gemv";
    case GGML_TYPE_MXFP4: return "mxfp4_q8_1_moe_gemv";
    default: return nullptr;
    }
}

static mmq_moe_fn mmq_moe_for(ggml_type t) {
    switch (t) {
    case GGML_TYPE_Q4_K: return launch_mmq_gguf_q4_k_moe;
    case GGML_TYPE_Q5_K: return launch_mmq_gguf_q5_k_moe;
    case GGML_TYPE_Q6_K: return launch_mmq_gguf_q6_k_moe;
    default: return nullptr;
    }
}

// Ours, one MoE projection through the tiered path's kernels: quantize_q8_1 (titan) + <type>_q8_1_moe_gemv,
// all experts resident (identity slot map), exactly the arguments titan_tiered.rs passes.
struct OursMoe {
    CUfunction quant, gemv;
    void *xq;
    uint32_t *map;
    int kp, rows;
    uint64_t w_len, x_len, xq_len, ids_len, map_len;
    unsigned gemv_blocks;
};

static void print_row(const char *side, const Case &cs, double us, const char *extra) {
    printf("RES\t%s%s\t%s\t%.2f\t%s\n", side, g_rep.c_str(), cs.name.c_str(), us, extra);
    fflush(stdout);
}

static void run_case(Ctx &c, const Case &cs, std::mt19937 &rng) {
    fprintf(stderr, "== %s\n", cs.name.c_str());
    const size_t mem = ggml_tensor_overhead() * 64 + ggml_graph_overhead() + 1024;
    ggml_init_params ip = {mem, nullptr, true};
    ggml_context *ctx = ggml_init(ip);
    ggml_tensor *W = nullptr, *W2 = nullptr, *W3 = nullptr, *X = nullptr, *ids = nullptr, *out = nullptr;
    if (cs.kind == DENSE) {
        W = ggml_new_tensor_2d(ctx, cs.type, cs.k, cs.n);
        X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cs.k, cs.B);
        out = ggml_mul_mat(ctx, W, X);
    } else if (cs.kind == MOE) {
        W = ggml_new_tensor_3d(ctx, cs.type, cs.k, cs.n, cs.E);
        X = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, cs.k, cs.xr, cs.B);
        ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, cs.topk, cs.B);
        out = ggml_mul_mat_id(ctx, W, X, ids);
    } else { // FFN: llama.cpp build_moe_ffn order (up, gate, swiglu_split, down)
        W = ggml_new_tensor_3d(ctx, cs.type, cs.k, cs.n, cs.E);   // up
        W2 = ggml_new_tensor_3d(ctx, cs.type, cs.k, cs.n, cs.E);  // gate
        W3 = ggml_new_tensor_3d(ctx, cs.type2, cs.n, cs.k, cs.E); // down
        X = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, cs.k, 1, cs.B);
        ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, cs.topk, cs.B);
        ggml_tensor *up = ggml_mul_mat_id(ctx, W, X, ids);
        ggml_tensor *gate = ggml_mul_mat_id(ctx, W2, X, ids);
        ggml_tensor *act = ggml_swiglu_split(ctx, gate, up);
        out = ggml_mul_mat_id(ctx, W3, act, ids);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, c.be);
    if (!buf) {
        fprintf(stderr, "alloc failed for %s\n", cs.name.c_str());
        exit(3);
    }
    ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    const int64_t wrows = (int64_t)cs.n * (cs.kind == DENSE ? 1 : cs.E);
    fill_weights(W, cs.k, wrows, rng);
    if (W2) fill_weights(W2, cs.k, wrows, rng);
    if (W3) fill_weights(W3, cs.n, (int64_t)cs.k * cs.E, rng);
    fill_f32(X, rng);
    std::vector<int32_t> idv;
    if (ids) {
        idv = make_ids(cs.B, cs.topk, cs.E, rng);
        ggml_backend_tensor_set(ids, idv.data(), 0, idv.size() * 4);
    }

    double l_wall = run_llama(c, gf, cs.name);
    std::vector<float> lo(ggml_nelements(out));
    ggml_backend_tensor_get(out, lo.data(), 0, lo.size() * 4);
    {
        char ex[128];
        snprintf(ex, sizeof ex, "wall_median_us=%.2f", l_wall);
        print_row("L", cs, l_wall, ex);
    }

    if (cs.kind == FFN) { // ours is timed per projection in the MOE cases
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return;
    }

    // ---------------- ours ----------------
    std::vector<float> oo(lo.size());
    void *dout;
    CK(cudaMalloc(&dout, lo.size() * 4));
    CK(cudaMemset(dout, 0, lo.size() * 4));
    const int kp = (cs.k + 511) / 512 * 512;
    if (cs.kind == DENSE) {
        const int B = cs.B;
        if (B <= 8) {
            void *xq;
            CK(cudaMalloc(&xq, (size_t)B * kp / 32 * 36));
            CUfunction rowsf = nullptr;
            if (B > 1) {
                char nm[64];
                snprintf(nm, sizeof nm, "mmvq_gguf_q8_0_f32_rows%d", B);
                CU(cuModuleGetFunction(&rowsf, c.rows, nm));
            }
            int ncols = cs.k, nrows = cs.n, scy = kp / 32, scd = cs.n;
            CUdeviceptr wp = (CUdeviceptr)W->data, xp = (CUdeviceptr)xq, op = (CUdeviceptr)dout;
            void *args[] = {&wp, &xp, &op, &ncols, &nrows, &scy, &scd};
            double us = run_ours(c, "O", cs.name, NRUNS, [&] {
                launch_mmvq_gguf_quantize_q8_1_f32(X->data, xq, cs.k, kp, B, c.s);
                if (B == 1)
                    launch_mmvq_gguf_q8_0_f32_plain(W->data, xq, dout, cs.k, cs.n, kp / 32, cs.n, 1, c.s);
                else
                    launch_cu(rowsf, (cs.n + 1) / 2, 1, 32, 8, c.s, args);
            });
            CK(cudaMemcpy(oo.data(), dout, oo.size() * 4, cudaMemcpyDeviceToHost));
            Cmp r = compare(lo, oo);
            char ex[256];
            snprintf(ex, sizeof ex, "path=%s eq=%zu/%zu max_rel=%.3g", B == 1 ? "mmvq_plain_cuda1" : "mmvq_rows", r.eq,
                     r.n, r.max_rel);
            print_row("O", cs, us, ex);
            cudaFree(xq);
        } else {
            const int kpm = (kp + 127) / 128 * 128;
            void *scr;
            CK(cudaMalloc(&scr, (size_t)B * (kpm / 128) * 144 + 128 * 144));
            double us = run_ours(c, "O", cs.name, NRUNS, [&] {
                launch_mmq_quantize_q8_1_D4(X->data, nullptr, scr, 0, cs.k, cs.k, 0, 0, kpm, B, 1, 1, c.s);
                launch_mmq_gguf_q8_0(c.fixup, W->data, scr, dout, cs.k, cs.n, B, cs.k / 32, cs.n, 1200, c.nsm, c.smpbo, 32,
                                     0, c.s);
            });
            CK(cudaMemcpy(oo.data(), dout, oo.size() * 4, cudaMemcpyDeviceToHost));
            Cmp r = compare(lo, oo);
            char ex[256];
            snprintf(ex, sizeof ex, "path=mmq eq=%zu/%zu max_rel=%.3g", r.eq, r.n, r.max_rel);
            print_row("O", cs, us, ex);
            cudaFree(scr);
        }
    } else { // MOE, tiered kernels
        OursMoe m;
        const int tasks = cs.B * cs.topk;
        m.rows = cs.B * cs.xr;
        m.kp = kp;
        const size_t q8rb = (size_t)kp / 32 * 36;
        CK(cudaMalloc(&m.xq, m.rows * q8rb));
        CK(cudaMemset(m.xq, 0, m.rows * q8rb));
        std::vector<uint32_t> map(cs.E);
        for (int e = 0; e < cs.E; e++) map[e] = e;
        CK(cudaMalloc(&m.map, cs.E * 4));
        CK(cudaMemcpy(m.map, map.data(), cs.E * 4, cudaMemcpyHostToDevice));
        const char *gname = moe_kernel_name(cs.type, cs.k);
        CU(cuModuleGetFunction(&m.quant, c.titan, "quantize_q8_1"));
        CU(cuModuleGetFunction(&m.gemv, c.titan, gname));
        const bool warp_rows = cs.k <= 512 && (cs.type == GGML_TYPE_Q4_K || cs.type == GGML_TYPE_Q5_K);
        m.gemv_blocks = warp_rows ? (unsigned)((cs.n + 3) / 4 * tasks) : (unsigned)(cs.n * tasks);
        CUdeviceptr xptr = (CUdeviceptr)X->data, xq = (CUdeviceptr)m.xq, wptr = (CUdeviceptr)W->data,
                    iptr = (CUdeviceptr)ids->data, mptr = (CUdeviceptr)m.map, optr = (CUdeviceptr)dout;
        uint64_t x_len = (uint64_t)m.rows * cs.k, w_len = ggml_nbytes(W) / 4, xq_len = m.rows * q8rb / 4,
                 ids_len = tasks, map_len = cs.E;
        uint32_t kx = cs.k, kpu = kp, n = cs.n, topk = cs.topk, dim1 = cs.xr;
        void *qargs[] = {&xptr, &x_len, &xq, &kx, &kpu};
        void *gargs[] = {&wptr, &w_len, &xq, &xq_len, &iptr, &ids_len, &mptr, &map_len, &optr, &n, &kx, &kpu, &topk, &dim1};
        const unsigned qblocks = (unsigned)((size_t)m.rows * kp / 256);
        const int nruns = cs.B >= 2048 ? 11 : NRUNS;
        double us = run_ours(c, "O", cs.name, nruns, [&] {
            launch_cu(m.quant, qblocks, 1, 256, 1, c.s, qargs);
            launch_cu(m.gemv, m.gemv_blocks, 1, 128, 1, c.s, gargs);
        });
        CK(cudaMemcpy(oo.data(), dout, oo.size() * 4, cudaMemcpyDeviceToHost));
        Cmp r = compare(lo, oo);
        char ex[256];
        snprintf(ex, sizeof ex, "path=tiered:%s runs=%d eq=%zu/%zu max_rel=%.3g", gname, nruns, r.eq, r.n, r.max_rel);
        print_row("O", cs, us, ex);

        // Alternative: our llama.cpp-derived MMQ MoE port (mistralrs-quant mmq, not used by the tiered service)
        mmq_moe_fn mf = mmq_moe_for(cs.type);
        if (cs.B > 8 && mf && !getenv("KBENCH_NO_ALT")) {
            // mm_ids_helper semantics: assignments sorted by expert; src row, dst column, expert bounds
            std::vector<int32_t> isrc, idst, bounds(cs.E + 1);
            int ncols_max = 0;
            for (int e = 0; e < cs.E; e++) {
                bounds[e] = (int)isrc.size();
                for (int b = 0; b < cs.B; b++)
                    for (int j = 0; j < cs.topk; j++)
                        if (idv[(size_t)b * cs.topk + j] == e) {
                            isrc.push_back(b * cs.xr + j % cs.xr);
                            idst.push_back(b * cs.topk + j);
                        }
                ncols_max = std::max(ncols_max, (int)isrc.size() - bounds[e]);
            }
            bounds[cs.E] = (int)isrc.size();
            int *d_isrc, *d_idst, *d_b;
            CK(cudaMalloc(&d_isrc, isrc.size() * 4));
            CK(cudaMalloc(&d_idst, idst.size() * 4));
            CK(cudaMalloc(&d_b, bounds.size() * 4));
            CK(cudaMemcpy(d_isrc, isrc.data(), isrc.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_idst, idst.data(), idst.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_b, bounds.data(), bounds.size() * 4, cudaMemcpyHostToDevice));
            const int kpm = (kp + 127) / 128 * 128;
            void *scr;
            CK(cudaMalloc(&scr, (size_t)tasks * (kpm / 128) * 144 + 128 * 144));
            CK(cudaMemset(dout, 0, lo.size() * 4));
            const int qk = 256;
            double us2 = run_ours(c, "A", cs.name, NRUNS, [&] {
                if (cs.type == GGML_TYPE_Q6_K)
                    launch_mmq_quantize_q8_1_D4(X->data, d_isrc, scr, 0, cs.k, cs.k, 0, 0, kpm, tasks, 1, 1, c.s);
                else
                    launch_mmq_quantize_q8_1_DS4(X->data, d_isrc, scr, 0, cs.k, cs.k, 0, 0, kpm, tasks, 1, 1, c.s);
                mf(c.fixup, W->data, scr, d_idst, d_b, dout, cs.k, cs.n, tasks, cs.k / qk, cs.n, cs.E, ncols_max, 1200,
                   c.nsm, c.smpbo, 32, c.s);
            });
            CK(cudaMemcpy(oo.data(), dout, oo.size() * 4, cudaMemcpyDeviceToHost));
            Cmp r2 = compare(lo, oo);
            snprintf(ex, sizeof ex, "path=mmq_moe(unused-by-tiered) eq=%zu/%zu max_rel=%.3g", r2.eq, r2.n, r2.max_rel);
            print_row("A", cs, us2, ex);
            cudaFree(scr);
            cudaFree(d_isrc);
            cudaFree(d_idst);
            cudaFree(d_b);
        }
        cudaFree(m.xq);
        cudaFree(m.map);
    }
    cudaFree(dout);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main(int argc, char **argv) {
    // absolute: ncu runs this under sudo (HOME=/root)
    std::string gg = std::string(getenv("HOME")) + "/titan-engine/mr-094/mistralrs-quant/src/gguf/";
    // KBENCH_PTX_DIR: the mistralrs-quant/src/gguf of the tree under test (bench/run.sh passes the binary's source)
    if (const char *d = getenv("KBENCH_PTX_DIR")) gg = std::string(d) + "/";
    if (const char *r = getenv("KBENCH_REP")) g_rep = r;

    std::vector<Case> cases;
    const int batches[] = {1, 2, 3, 8, 512, 2048, 8192};
    auto add = [&](std::string nm, Kind k, ggml_type t, ggml_type t2, int K, int N, int Ex, int tk, int xr,
                   std::initializer_list<int> bs) {
        for (int B : bs) {
            Case c{nm + "/b" + std::to_string(B), k, t, t2, K, N, Ex, tk, B, xr};
            cases.push_back(c);
        }
    };
    auto all = {1, 2, 3, 8, 512, 2048, 8192};
    (void)batches;
    // 35B Qwen3.6-35B-A3B-UD-Q4_K_XL: experts
    add("q35.exp_gate_up.q4_K.k2048n512", MOE, GGML_TYPE_Q4_K, GGML_TYPE_COUNT, 2048, 512, 256, 8, 1, all);
    add("q35.exp_down.q5_K.k512n2048", MOE, GGML_TYPE_Q5_K, GGML_TYPE_COUNT, 512, 2048, 256, 8, 8, all);
    add("q35.exp_down.q6_K.k512n2048", MOE, GGML_TYPE_Q6_K, GGML_TYPE_COUNT, 512, 2048, 256, 8, 8, all);
    add("q35.ffn_fused.q4_K+q5_K", FFN, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, 2048, 512, 256, 8, 1, {1, 3, 8, 512});
    // 35B dense Q8_0 (attention / GDN / shared expert / lm_head)
    add("q35.qkv.q8_0.k2048n8192", DENSE, GGML_TYPE_Q8_0, GGML_TYPE_COUNT, 2048, 8192, 1, 1, 1, all);
    add("q35.attn_gate.q8_0.k2048n4096", DENSE, GGML_TYPE_Q8_0, GGML_TYPE_COUNT, 2048, 4096, 1, 1, 1, all);
    add("q35.ssm_out.q8_0.k4096n2048", DENSE, GGML_TYPE_Q8_0, GGML_TYPE_COUNT, 4096, 2048, 1, 1, 1, all);
    add("q35.shexp_gate_up.q8_0.k2048n512", DENSE, GGML_TYPE_Q8_0, GGML_TYPE_COUNT, 2048, 512, 1, 1, 1, all);
    add("q35.shexp_down.q8_0.k512n2048", DENSE, GGML_TYPE_Q8_0, GGML_TYPE_COUNT, 512, 2048, 1, 1, 1, all);
    add("q35.lm_head.q8_0.k2048n248320", DENSE, GGML_TYPE_Q8_0, GGML_TYPE_COUNT, 2048, 248320, 1, 1, 1, {1, 3});
    // 35B MXFP4_MOE file: gate/up experts
    add("q35mx.exp_gate_up.mxfp4.k2048n512", MOE, GGML_TYPE_MXFP4, GGML_TYPE_COUNT, 2048, 512, 256, 8, 1, all);
    // IQ4_NL experts (same shape), cheap extra
    add("q35.exp_gate_up.iq4_nl.k2048n512", MOE, GGML_TYPE_IQ4_NL, GGML_TYPE_COUNT, 2048, 512, 256, 8, 1, {1, 3, 8, 512});
    // 80B qwen3next Q4_K_M: 512 experts, top-10
    add("q80.exp_gate_up.q4_K.k2048n512", MOE, GGML_TYPE_Q4_K, GGML_TYPE_COUNT, 2048, 512, 512, 10, 1, all);
    add("q80.exp_down.q6_K.k512n2048", MOE, GGML_TYPE_Q6_K, GGML_TYPE_COUNT, 512, 2048, 512, 10, 10, all);
    add("q80.exp_down.q4_K.k512n2048", MOE, GGML_TYPE_Q4_K, GGML_TYPE_COUNT, 512, 2048, 512, 10, 10, {1, 3, 8, 512});

    if (argc > 1 && std::string(argv[1]) == "--list") {
        for (auto &c : cases) printf("%s\n", c.name.c_str());
        return 0;
    }

    Ctx c;
    c.be = ggml_backend_cuda_init(0);
    if (!c.be) {
        fprintf(stderr, "no CUDA backend\n");
        return 2;
    }
    CK(cudaSetDevice(0));
    CK(cudaFree(0)); // primary context current on this thread (the one ggml uses)
    CK(cudaStreamCreateWithFlags(&c.s, cudaStreamNonBlocking));
    CK(cudaMalloc(&c.flush, c.flush_bytes));
    CK(cudaDeviceGetAttribute(&c.nsm, cudaDevAttrMultiProcessorCount, 0));
    CK(cudaDeviceGetAttribute(&c.smpbo, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    CK(cudaMalloc(&c.fixup, (size_t)c.nsm * 128 * 128 * 4));
    std::string titan_ptx = read_file((gg + "titan_kernels_oxide.ptx").c_str());
    std::string rows_ptx = read_file((gg + "mmvq_rows_oxide.ptx").c_str());
    CU(cuModuleLoadData(&c.titan, titan_ptx.c_str()));
    CU(cuModuleLoadData(&c.rows, rows_ptx.c_str()));
    fprintf(stderr, "nsm=%d smpbo=%d\n", c.nsm, c.smpbo);

    std::mt19937 rng(1234);
    for (auto &cs : cases) {
        bool sel = argc <= 1;
        for (int i = 1; i < argc; i++)
            if (getenv("KBENCH_EXACT") ? cs.name == argv[i] : cs.name.find(argv[i]) != std::string::npos) sel = true;
        if (sel) run_case(c, cs, rng);
    }
    CK(cudaDeviceSynchronize());
    fprintf(stderr, "done\n");
    return 0;
}
