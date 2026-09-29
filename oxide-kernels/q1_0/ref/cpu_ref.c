// Q1_0 CPU oracle: dumps llama.cpp's own results (libggml-base / libggml-cpu, commit acecd56)
// for a seeded input set that candle's test regenerates bit-for-bit.
//   dequantize_row_q1_0            (ggml_get_type_traits(Q1_0)->to_float)
//   ggml_vec_dot_q1_0_q8_0 (AVX2)  (ggml_get_type_traits_cpu(Q1_0)->vec_dot)
//   ggml_vec_dot_q1_0_q8_0_generic (verbatim copy below, same gcc flags as ggml-cpu)
//   quantize_row_q1_0_ref          (ggml_get_type_traits(Q1_0)->from_float_ref)
// Build: see ref/build_cpu_ref.sh
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <immintrin.h>
#include "ggml.h"
#include "ggml-cpu.h"

#define QK1_0 128
#define QK8_0 32
typedef struct { uint16_t d; uint8_t qs[QK1_0 / 8]; } block_q1_0;
typedef struct { uint16_t d; int8_t qs[QK8_0]; } block_q8_0;
_Static_assert(sizeof(block_q1_0) == 18, "q1_0");
_Static_assert(sizeof(block_q8_0) == 34, "q8_0");
#define GGML_CPU_FP16_TO_FP32(x) _cvtsh_ss(x)
#define GGML_RESTRICT restrict
#define UNUSED(x) (void)(x)

// ---- verbatim from ggml/src/ggml-cpu/quants.c (acecd56) ----
static void ggml_vec_dot_q1_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK1_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q1_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    float sumf = 0.0;

    for (int i = 0; i < nb; i++) {
        const float d0 = GGML_CPU_FP16_TO_FP32(x[i].d);

        float sumi = 0.0f;

        for (int k = 0; k < 4; k++) {
            const block_q8_0 * GGML_RESTRICT yb = &y[i * 4 + k];
            const float d1 = GGML_CPU_FP16_TO_FP32(yb->d);
            int sumi_block = 0;

            const uint8_t * GGML_RESTRICT bits = &x[i].qs[k * 4];
            const int8_t  * GGML_RESTRICT qy   = yb->qs;

            for (int b = 0; b < 4; ++b, qy += 8) {
                const unsigned mask = bits[b];
                sumi_block += ((mask & 0x01) ? qy[0] : -qy[0])
                           +  ((mask & 0x02) ? qy[1] : -qy[1])
                           +  ((mask & 0x04) ? qy[2] : -qy[2])
                           +  ((mask & 0x08) ? qy[3] : -qy[3])
                           +  ((mask & 0x10) ? qy[4] : -qy[4])
                           +  ((mask & 0x20) ? qy[5] : -qy[5])
                           +  ((mask & 0x40) ? qy[6] : -qy[6])
                           +  ((mask & 0x80) ? qy[7] : -qy[7]);
            }

            sumi += d1 * sumi_block;
        }

        sumf += d0 * sumi;
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
    static block_q1_0 x[NB1];
    static block_q8_0 y[NB1 * 4];
    for (int i = 0; i < NB1; i++) { x[i].d = rand_f16(i, 32); for (int j = 0; j < 16; j++) x[i].qs[j] = (uint8_t)nx(); }
    for (int i = 0; i < NB1 * 4; i++) { y[i].d = rand_f16(i, 512); for (int j = 0; j < 32; j++) y[i].qs[j] = (int8_t)(uint8_t)nx(); }

    const struct ggml_type_traits *tt = ggml_get_type_traits(GGML_TYPE_Q1_0);
    const struct ggml_type_traits_cpu *tc = ggml_get_type_traits_cpu(GGML_TYPE_Q1_0);
    assert(tt->blck_size == 128 && tt->type_size == 18 && tc->vec_dot_type == GGML_TYPE_Q8_0);

    static float deq[NB1 * QK1_0];
    tt->to_float(x, deq, NB1 * QK1_0);
    printf("dequant_fnv 0x%016llx\n", (unsigned long long)fnv(deq, sizeof deq, 0xcbf29ce484222325ULL));

    printf("vecdot");
    for (int t = 0; t < NT; t++) {
        int nb = 1 + (int)(nx() % 32);
        int xo = (int)(nx() % (NB1 - nb + 1));
        int yo = (int)(nx() % (NB1 - nb + 1));
        float a, g;
        tc->vec_dot(nb * QK1_0, &a, 0, &x[xo], 0, &y[yo * 4], 0, 1);
        ggml_vec_dot_q1_0_q8_0_generic(nb * QK1_0, &g, 0, &x[xo], 0, &y[yo * 4], 0, 1);
        uint32_t ab, gb; memcpy(&ab, &a, 4); memcpy(&gb, &g, 4);
        printf(" %08x:%08x", ab, gb);
    }
    printf("\n");

    // quantize_row_q1_0_ref on finite data (with exact zeros / -0)
    static float xs[NQ * QK1_0];
    for (int i = 0; i < NQ * QK1_0; i++) {
        uint64_t r = nx();
        if (r % 23 == 0) xs[i] = (r & 64) ? -0.0f : 0.0f;
        else xs[i] = (float)((double)(nx() >> 11) / 9007199254740992.0 * 16.0 - 8.0) * (float)(1 + (int)(r % 5)) * 0.01f;
    }
    static block_q1_0 q[NQ];
    tt->from_float_ref(xs, q, NQ * QK1_0);
    printf("quant_fnv 0x%016llx\n", (unsigned long long)fnv(q, sizeof q, 0xcbf29ce484222325ULL));
    return 0;
}
