// g4dump: run one prompt through llama.cpp and write the last token's row of selected graph tensors as raw f32
// (DIR/NAME.f32, NAME = the graph name, e.g. l_out-5), the llama.cpp side of titan's TITAN_G4_DUMP.
// usage: g4dump MODEL PROMPT_FILE OUT_DIR NGL FA(0|1) CTK(f16|bf16) REGEX
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

struct ud_t { std::regex re; std::string dir; int n_tokens; int n; bool active; };

static bool cb(struct ggml_tensor * t, bool ask, void * p) {
    auto * u = (ud_t *) p;
    if (ask) return u->active && std::regex_match(t->name, u->re);
    if (!u->active || !std::regex_match(t->name, u->re)) return true;
    // the last token: the highest index of the outermost dim that equals n_tokens
    int tdim = -1;
    for (int d = GGML_MAX_DIMS - 1; d >= 1; --d) if (t->ne[d] == u->n_tokens) { tdim = d; break; }
    if (tdim < 0) return true;
    std::vector<float> out;
    std::vector<uint8_t> buf(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size());
    int64_t ne[4] = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
    for (int64_t i3 = 0; i3 < ne[3]; ++i3) for (int64_t i2 = 0; i2 < ne[2]; ++i2) for (int64_t i1 = 0; i1 < ne[1]; ++i1) {
        int64_t idx[4] = {0, i1, i2, i3};
        if (idx[tdim] != u->n_tokens - 1) continue;
        for (int64_t i0 = 0; i0 < ne[0]; ++i0) {
            size_t off = i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
            float v;
            if (t->type == GGML_TYPE_F32) memcpy(&v, buf.data() + off, 4);
            else if (t->type == GGML_TYPE_F16) v = ggml_fp16_to_fp32(*(ggml_fp16_t *)(buf.data() + off));
            else if (t->type == GGML_TYPE_BF16) { uint32_t b = (uint32_t)(*(uint16_t *)(buf.data() + off)) << 16; memcpy(&v, &b, 4); }
            else return true;
            out.push_back(v);
        }
    }
    std::string f = u->dir + "/" + t->name + ".f32";
    FILE * fp = fopen(f.c_str(), "wb");
    if (fp) { fwrite(out.data(), 4, out.size(), fp); fclose(fp); u->n++; }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 8) { fprintf(stderr, "usage: g4dump MODEL PROMPT_FILE OUT_DIR NGL FA CTK REGEX\n"); return 1; }
    std::ifstream pf(argv[2]); std::stringstream ss; ss << pf.rdbuf(); std::string prompt = ss.str();
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = atoi(argv[4]);
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 2;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(prompt.size() + 16);
    int n = llama_tokenize(vocab, prompt.c_str(), prompt.size(), toks.data(), toks.size(), true, true);
    if (n < 0) return 3;
    toks.resize(n);
    ud_t u{std::regex(argv[7]), argv[3], n, 0, false};
    auto cp = llama_context_default_params();
    cp.n_ctx = n + 64; cp.n_batch = 512; cp.n_ubatch = 512; cp.no_perf = true;
    cp.flash_attn_type = atoi(argv[5]) ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = cp.type_v = strcmp(argv[6], "bf16") == 0 ? GGML_TYPE_BF16 : GGML_TYPE_F16;
    cp.cb_eval = cb; cp.cb_eval_user_data = &u;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 4;
    // 512-token ubatches in order, as llama-server (n_ubatch 512); only the last one (the last token) is dumped
    for (int i = 0; i < n; i += 512) {
        int m = std::min(512, n - i);
        u.active = i + m == n;
        u.n_tokens = m;
        if (llama_decode(ctx, llama_batch_get_one(toks.data() + i, m))) return 5;
    }
    const float * lg = llama_get_logits_ith(ctx, -1);
    int nv = llama_vocab_n_tokens(vocab);
    std::string f = u.dir + "/logits.f32";
    FILE * fp = fopen(f.c_str(), "wb"); fwrite(lg, 4, nv, fp); fclose(fp);
    fprintf(stderr, "g4dump: %d tokens, %d tensors written\n", n, u.n);
    llama_free(ctx); llama_model_free(model);
    return 0;
}
