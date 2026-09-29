// IQ4_NL CPU oracle: dumps llama.cpp's own results (libggml-base / libggml-cpu, commit acecd56)
// for a seeded input set that candle's test regenerates bit-for-bit.
//   dequantize_row_iq4_nl             (ggml_get_type_traits(IQ4_NL)->to_float)
//   ggml_vec_dot_iq4_nl_q8_0 (AVX2)   (ggml_get_type_traits_cpu(IQ4_NL)->vec_dot)
//   ggml_vec_dot_iq4_nl_q8_0_generic  (verbatim copy below, same gcc flags as ggml-cpu)
//   quantize_row_iq4_nl_ref           (ggml_get_type_traits(IQ4_NL)->from_float_ref; the CPU
//                                      from_float quantize_row_iq4_nl calls it too)
// Build: see ref/build_cpu_ref.sh
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <immintrin.h>
#include "ggml.h"
#include "ggml-cpu.h"

#define QK4_NL 32
#define QK8_0 32
typedef struct { uint16_t d; uint8_t qs[QK4_NL / 2]; } block_iq4_nl;
typedef struct { uint16_t d; int8_t qs[QK8_0]; } block_q8_0;
_Static_assert(sizeof(block_iq4_nl) == 18, "iq4_nl");
_Static_assert(sizeof(block_q8_0) == 34, "q8_0");
#define GGML_CPU_FP16_TO_FP32(x) _cvtsh_ss(x)
#define GGML_RESTRICT restrict
#define UNUSED(x) (void)(x)
static const int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// ---- verbatim from ggml/src/ggml-cpu/quants.c (acecd56) ----
static void ggml_vec_dot_iq4_nl_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK4_NL == 0);
    static_assert(QK4_NL == QK8_0, "QK4_NL and QK8_0 must be the same");

    const block_iq4_nl * GGML_RESTRICT x = vx;
    const block_q8_0   * GGML_RESTRICT y = vy;

    const int nb = n / QK4_NL;

    int ib = 0;
    float sumf = 0;

    for (; ib < nb; ++ib) {
        const float d = GGML_CPU_FP16_TO_FP32(y[ib].d)*GGML_CPU_FP16_TO_FP32(x[ib].d);
        int sumi1 = 0, sumi2 = 0;
        for (int j = 0; j < QK4_NL/2; ++j) {
            sumi1 += y[ib].qs[j+       0] * kvalues_iq4nl[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j+QK4_NL/2] * kvalues_iq4nl[x[ib].qs[j] >>  4];
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
    static block_iq4_nl x[NB1];
    static block_q8_0 y[NB1];
    for (int i = 0; i < NB1; i++) { x[i].d = rand_f16(i, 32); for (int j = 0; j < 16; j++) x[i].qs[j] = (uint8_t)nx(); }
    for (int i = 0; i < NB1; i++) { y[i].d = rand_f16(i, 512); for (int j = 0; j < 32; j++) y[i].qs[j] = (int8_t)(uint8_t)nx(); }

    const struct ggml_type_traits *tt = ggml_get_type_traits(GGML_TYPE_IQ4_NL);
    const struct ggml_type_traits_cpu *tc = ggml_get_type_traits_cpu(GGML_TYPE_IQ4_NL);
    assert(tt->blck_size == 32 && tt->type_size == 18 && tc->vec_dot_type == GGML_TYPE_Q8_0);

    static float deq[NB1 * QK4_NL];
    tt->to_float(x, deq, NB1 * QK4_NL);
    printf("dequant_fnv 0x%016llx\n", (unsigned long long)fnv(deq, sizeof deq, 0xcbf29ce484222325ULL));

    printf("vecdot");
    for (int t = 0; t < NT; t++) {
        int nb = 1 + (int)(nx() % 64);
        int xo = (int)(nx() % (NB1 - nb + 1));
        int yo = (int)(nx() % (NB1 - nb + 1));
        float a, g;
        tc->vec_dot(nb * QK4_NL, &a, 0, &x[xo], 0, &y[yo], 0, 1);
        ggml_vec_dot_iq4_nl_q8_0_generic(nb * QK4_NL, &g, 0, &x[xo], 0, &y[yo], 0, 1);
        uint32_t ab, gb; memcpy(&ab, &a, 4); memcpy(&gb, &g, 4);
        printf(" %08x:%08x", ab, gb);
    }
    printf("\n");

    // quantize_row_iq4_nl_ref on finite data: exact zeros / -0, and all-zero blocks (they keep the
    // previous block's indices: the L[] scratch array is reused), never block 0.
    static float xs[NQ * QK4_NL];
    for (int i = 0; i < NQ * QK4_NL; i++) {
        uint64_t r = nx();
        if (r % 23 == 0) xs[i] = (r & 64) ? -0.0f : 0.0f;
        else xs[i] = (float)((double)(nx() >> 11) / 9007199254740992.0 * 16.0 - 8.0) * (float)(1 + (int)(r % 5)) * 0.01f;
    }
    for (int b = 5; b < NQ; b += 17) for (int j = 0; j < QK4_NL; j++) xs[b * QK4_NL + j] = (j & 1) ? -0.0f : 0.0f;
    for (int j = 0; j < QK4_NL; j++) xs[9 * QK4_NL + j] *= 1e-17f;  // below GROUP_MAX_EPS
    for (int j = 0; j < QK4_NL; j++) xs[10 * QK4_NL + j] *= 3e4f;   // large scale
    static block_iq4_nl q[NQ];
    tt->from_float_ref(xs, q, NQ * QK4_NL);
    printf("quant_fnv 0x%016llx\n", (unsigned long long)fnv(q, sizeof q, 0xcbf29ce484222325ULL));
    static block_iq4_nl q2[NQ];
    tc->from_float(xs, q2, NQ * QK4_NL);
    printf("from_float_same %d\n", memcmp(q, q2, sizeof q) == 0);
    return 0;
}
