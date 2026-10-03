// Reference CPU dequantization from llama.cpp's own ggml (ggml_get_type_traits(t)->to_float, i.e.
// dequantize_row_*): for each requested GGML type id, NB random blocks (xorshift64*, seed = type id)
// -> OUT/<id>.blocks (raw block bytes) and OUT/<id>.f32 (llama.cpp's dequantized values).
// usage: ggml_deq OUTDIR MIN_VALUES TYPE_ID...
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
typedef void (*to_float_t)(const void *, float *, int64_t);
struct traits { const char *type_name; int64_t blck_size; int64_t blck_size_interleave; size_t type_size;
                bool is_quantized; to_float_t to_float; void *from_float_ref; };
const struct traits *ggml_get_type_traits(int type);
static uint64_t s;
static uint64_t rnd(void) { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; return s * 0x2545F4914F6CDD1DULL; }
int main(int argc, char **argv) {
    const char *out = argv[1];
    long minv = atol(argv[2]);
    for (int a = 3; a < argc; a++) {
        int t = atoi(argv[a]);
        const struct traits *tr = ggml_get_type_traits(t);
        if (!tr->to_float) { printf("type %d %s: no to_float, skipped\n", t, tr->type_name); continue; }
        long nb = (minv + tr->blck_size - 1) / tr->blck_size;
        size_t nbytes = (size_t)nb * tr->type_size, nv = (size_t)nb * tr->blck_size;
        uint8_t *x = malloc(nbytes); float *y = malloc(nv * sizeof(float));
        s = 0x9E3779B97F4A7C15ULL ^ (uint64_t)t;
        for (size_t i = 0; i < nbytes; i++) x[i] = (uint8_t)(rnd() >> 56);
        tr->to_float(x, y, (int64_t)nv);
        char p[4096];
        snprintf(p, sizeof p, "%s/%d.blocks", out, t); FILE *f = fopen(p, "wb"); fwrite(x, 1, nbytes, f); fclose(f);
        snprintf(p, sizeof p, "%s/%d.f32", out, t); f = fopen(p, "wb"); fwrite(y, sizeof(float), nv, f); fclose(f);
        printf("type %d %s: %ld blocks x %ld = %zu values, block %zu bytes\n", t, tr->type_name, nb, (long)tr->blck_size, nv, tr->type_size);
        free(x); free(y);
    }
    return 0;
}
