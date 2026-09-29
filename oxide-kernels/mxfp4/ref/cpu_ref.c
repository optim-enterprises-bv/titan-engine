// MXFP4 CPU oracle: dumps llama.cpp's own results (libggml-base / libggml-cpu, commit acecd56)
// for a seeded input set that candle's test regenerates bit-for-bit.
//   dequantize_row_mxfp4             (ggml_get_type_traits(MXFP4)->to_float)
//   ggml_vec_dot_mxfp4_q8_0 (AVX2)   (ggml_get_type_traits_cpu(MXFP4)->vec_dot)
//   ggml_vec_dot_mxfp4_q8_0_generic  (verbatim copy below, same gcc flags as ggml-cpu)
//   quantize_row_mxfp4_ref           (ggml_get_type_traits(MXFP4)->from_float_ref; the CPU
//                                     from_float quantize_row_mxfp4 calls it too)
// Build: see ref/build_cpu_ref.sh
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <immintrin.h>
#include "ggml.h"
#include "ggml-cpu.h"

#define QK_MXFP4 32
#define QK8_0 32
typedef struct { uint8_t e; uint8_t qs[QK_MXFP4 / 2]; } block_mxfp4;
typedef struct { uint16_t d; int8_t qs[QK8_0]; } block_q8_0;
_Static_assert(sizeof(block_mxfp4) == 17, "mxfp4");
_Static_assert(sizeof(block_q8_0) == 34, "q8_0");
#define GGML_CPU_FP16_TO_FP32(x) _cvtsh_ss(x)
#define GGML_RESTRICT restrict
#define UNUSED(x) (void)(x)
static const int8_t kvalues_mxfp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
static inline float GGML_E8M0_TO_FP32_HALF(uint8_t x) {
    uint32_t bits = x < 2 ? 0x00200000u << x : (uint32_t)(x - 1) << 23;
    float r; memcpy(&r, &bits, 4); return r;
}

// ---- verbatim from ggml/src/ggml-cpu/quants.c (acecd56) ----
static void ggml_vec_dot_mxfp4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK_MXFP4 == 0);
    static_assert(QK_MXFP4 == QK8_0, "QK_MXFP4 and QK8_0 must be the same");

    const block_mxfp4 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    const int nb = n / QK_MXFP4;

    int ib = 0;
    float sumf = 0;

    for (; ib < nb; ++ib) {
        const float d = GGML_CPU_FP16_TO_FP32(y[ib].d)*GGML_E8M0_TO_FP32_HALF(x[ib].e);

        int sumi1 = 0;
        int sumi2 = 0;
        for (int j = 0; j < QK_MXFP4/2; ++j) {
            sumi1 += y[ib].qs[j +          0] * kvalues_mxfp4[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j + QK_MXFP4/2] * kvalues_mxfp4[x[ib].qs[j] >>  4];
        }
        sumf += d * (sumi1 + sumi2);
    }
    *s = sumf;
}
// ---- end verbatim ----

static uint64_t S;
static uint64_t nx(void) { S ^= S << 13; S ^= S >> 7; S ^= S << 17; return S; }
static uint16_t rand_f16(int i, unsigned rate) {
    static const uint16_t sp[8] = {0x0000, 0x8000, 0x7C00, 0xFC00, 0x7E00, 0x0001, 0x83FF, 0x7BFF};
    if (nx() % rate == 0) return sp[i % 8];
    uint16_t v = (uint16_t)(0x2000 + nx() % 0x2800);
    return v | (uint16_t)((nx() & 1) << 15);
}
static uint64_t fnv(const void *p, size_t n, uint64_t h) {
    const uint8_t *b = p; for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 0x100000001b3ULL; } return h;
}

#define NB1 512
#define NT 400
#define NQ 96

int main(int argc, char **argv) {
    ggml_cpu_init();
    S = argc > 1 ? strtoull(argv[1], 0, 0) : 0x9E3779B97F4A7C15ULL;
    static block_mxfp4 x[NB1];
    static block_q8_0 y[NB1];
    for (int i = 0; i < NB1; i++) { x[i].e = (uint8_t)nx(); for (int j = 0; j < 16; j++) x[i].qs[j] = (uint8_t)nx(); }
    for (int i = 0; i < NB1; i++) { y[i].d = rand_f16(i, 512); for (int j = 0; j < 32; j++) y[i].qs[j] = (int8_t)(uint8_t)nx(); }

    const struct ggml_type_traits *tt = ggml_get_type_traits(GGML_TYPE_MXFP4);
    const struct ggml_type_traits_cpu *tc = ggml_get_type_traits_cpu(GGML_TYPE_MXFP4);
    assert(tt->blck_size == 32 && tt->type_size == 17 && tc->vec_dot_type == GGML_TYPE_Q8_0);

    static float deq[NB1 * QK_MXFP4];
    tt->to_float(x, deq, NB1 * QK_MXFP4);
    printf("dequant_fnv 0x%016llx\n", (unsigned long long)fnv(deq, sizeof deq, 0xcbf29ce484222325ULL));

    printf("vecdot");
    for (int t = 0; t < NT; t++) {
        int nb = 1 + (int)(nx() % 64);
        int xo = (int)(nx() % (NB1 - nb + 1));
        int yo = (int)(nx() % (NB1 - nb + 1));
        float a, g;
        tc->vec_dot(nb * QK_MXFP4, &a, 0, &x[xo], 0, &y[yo], 0, 1);
        ggml_vec_dot_mxfp4_q8_0_generic(nb * QK_MXFP4, &g, 0, &x[xo], 0, &y[yo], 0, 1);
        uint32_t ab, gb; memcpy(&ab, &a, 4); memcpy(&gb, &g, 4);
        printf(" %08x:%08x", ab, gb);
    }
    printf("\n");

    // quantize_row_mxfp4_ref on finite data: exact zeros / -0, all-zero blocks, a denormal block
    // (the float -> uint8_t exponent cast wraps), tiny and huge scales.
    static float xs[NQ * QK_MXFP4];
    for (int i = 0; i < NQ * QK_MXFP4; i++) {
        uint64_t r = nx();
        if (r % 23 == 0) xs[i] = (r & 64) ? -0.0f : 0.0f;
        else xs[i] = (float)((double)(nx() >> 11) / 9007199254740992.0 * 16.0 - 8.0) * (float)(1 + (int)(r % 5)) * 0.01f;
    }
    for (int b = 5; b < NQ; b += 17) for (int j = 0; j < QK_MXFP4; j++) xs[b * QK_MXFP4 + j] = (j & 1) ? -0.0f : 0.0f;
    for (int j = 0; j < QK_MXFP4; j++) xs[9 * QK_MXFP4 + j] *= 1e-40f;
    for (int j = 0; j < QK_MXFP4; j++) xs[10 * QK_MXFP4 + j] *= 3e4f;
    for (int j = 0; j < QK_MXFP4; j++) xs[11 * QK_MXFP4 + j] *= 1e-37f;
    for (int j = 0; j < QK_MXFP4; j++) xs[12 * QK_MXFP4 + j] *= 1e36f;
    for (int j = 0; j < QK_MXFP4; j++) xs[13 * QK_MXFP4 + j] = (float)(j - 16) * 0.25f;  // exact grid points, powers of two
    static block_mxfp4 q[NQ];
    tt->from_float_ref(xs, q, NQ * QK_MXFP4);
    printf("quant_fnv 0x%016llx\n", (unsigned long long)fnv(q, sizeof q, 0xcbf29ce484222325ULL));
    static block_mxfp4 q2[NQ];
    tc->from_float(xs, q2, NQ * QK_MXFP4);
    printf("from_float_same %d\n", memcmp(q, q2, sizeof q) == 0);
    return 0;
}
