"""NanoQuant checkpoint (nq_stream q_kd, one .pt per layer) -> titan GGUF.

Metadata, tokenizer, token_embd, output and output_norm come from the llama.cpp Q8_0 conversion of the same HF model;
every per-layer tensor comes from the checkpoint with llama.cpp's qwen35moe transforms (V-head reorders, A_log -> -exp,
+1 on zero-centred norms). A factored linear W = diag(s_post) U V diag(s_pre) (U, V in {-1,+1}, bit packed lsb first,
set bit = -1) is stored as <name>.nq_u ([out, r/32] words), <name>.nq_v ([r, in/32] words), <name>.nq_spre / .nq_spost (F32).
Bit words are typed F32 in the file (candle reads no I32 tensors): raw bytes, never used as floats.
Experts stack these over the expert axis. The MTP block is dropped (nextn_predict_layers 0).

usage: nq2gguf.py <q_kd dir> <Q8_0 gguf> <out gguf> [--check]
"""
import sys, os
import numpy as np
import torch
sys.path.insert(0, os.path.expanduser("~/src/llama.cpp-inkling-atomic/gguf-py"))
import gguf

qdir, ref_path, out_path = sys.argv[1:4]
CHECK = "--check" in sys.argv
ARCH = "qwen35moe"

ref = gguf.GGUFReader(ref_path)
field = lambda k: ref.fields[k].contents()
n_layer = field(f"{ARCH}.block_count") - field(f"{ARCH}.nextn_predict_layers")
NK, NV = field(f"{ARCH}.ssm.group_count"), field(f"{ARCH}.ssm.time_step_rank")
HK = field(f"{ARCH}.ssm.state_size")
HV = field(f"{ARCH}.ssm.inner_size") // NV
VPK = NV // NK
ref_t = {t.name: t for t in ref.tensors}


def v_perm(n_heads_k, head_dim):
    """grouped-by-K -> tiled V head order (llama.cpp _reorder_v_heads) as an index vector"""
    n = n_heads_k * VPK * head_dim
    return torch.arange(n).reshape(n_heads_k, VPK, head_dim).permute(1, 0, 2).reshape(n)


def ident(n):
    return torch.arange(n)


def qkv_rows():
    q = NK * HK
    return torch.cat([ident(2 * q), 2 * q + v_perm(NK, HV)])


def unpack(packed, shape):
    """lsb-first bits -> {-1,+1} int8 (set bit = -1)"""
    p = packed.to(torch.int64) & 0xFFFFFFFF
    bits = (p.unsqueeze(-1) >> torch.arange(32)) & 1
    return (1 - 2 * bits.reshape(p.shape[0], -1)[:, : shape[1]]).to(torch.int8)


def pack(m):
    """{-1,+1} int8 [rows, cols] -> int32 [rows, cols/32], lsb first"""
    bits = ((1 - m.to(torch.int64)) // 2).reshape(m.shape[0], -1, 32)
    w = (bits << torch.arange(32)).sum(-1)
    return torch.where(w >= 2**31, w - 2**32, w).to(torch.int32)


def factored(st, hf, rows=None, cols=None):
    """(U packed, V packed, s_pre, s_post) with an output-row / input-column permutation applied"""
    U, V = st[f"{hf}.U_packed"], st[f"{hf}.V_packed"]
    spre, spost = st[f"{hf}.scale_pre"].float().reshape(-1), st[f"{hf}.scale_post"].float().reshape(-1)
    if rows is not None:
        U, spost = U[rows], spost[rows]
    if cols is not None:
        vshape = tuple(st[f"{hf}.V_shape"].tolist())
        V, spre = pack(unpack(V, vshape)[:, cols]), spre[cols]
    return U.contiguous(), V.contiguous(), spre, spost


def dense(U, V, spre, spost, in_dim):
    u = unpack(U, (U.shape[0], U.shape[1] * 32)).float()
    v = unpack(V, (V.shape[0], in_dim)).float()
    return spost[:, None] * (u @ v) * spre[None, :]


w = gguf.GGUFWriter(out_path, ARCH)
for f in ref.fields.values():
    if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
        continue
    vt = f.types[0]
    sub = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    val = f.contents()
    if f.name == f"{ARCH}.block_count":
        val = n_layer
    elif f.name == f"{ARCH}.nextn_predict_layers":
        val = 0
    elif f.name == f"{ARCH}.attention.recurrent_layers":
        val = val[:n_layer]
    elif f.name == "general.file_type":
        val = 0
    w.add_key_value(f.name, val, vt, sub_type=sub)
w.add_key_value("general.quantization", "nanoquant-1bit", gguf.GGUFValueType.STRING)

out = []  # (name, np array or reference tensor)


def put(name, arr):
    out.append((name, arr))


for n in ("token_embd.weight", "output_norm.weight", "output.weight"):
    put(n, ref_t[n])

checks = []
for L in range(n_layer):
    d = torch.load(f"{qdir}/layer_{L:03d}.pt", map_location="cpu", weights_only=False)
    st = {k.removeprefix("layer."): v for k, v in d["state"].items()}
    p = f"blk.{L}"
    f32 = lambda t: t.float().numpy()

    def lin(gg, hf, rows=None, cols=None):
        U, V, spre, spost = factored(st, hf, rows, cols)
        put(f"{gg}.nq_u", U.numpy().view(np.float32)); put(f"{gg}.nq_v", V.numpy().view(np.float32))
        put(f"{gg}.nq_spre", spre.numpy()); put(f"{gg}.nq_spost", spost.numpy())
        if CHECK and L in (0, 3):
            checks.append((f"{gg}.weight", dense(U, V, spre, spost, spre.numel())))

    def plain(gg, t):
        put(gg, f32(t))
        if CHECK and L in (0, 3):
            checks.append((gg, t.float()))

    plain(f"{p}.attn_norm.weight", st["input_layernorm.weight"].float() + 1)
    plain(f"{p}.post_attention_norm.weight", st["post_attention_layernorm.weight"].float() + 1)
    if "self_attn.q_proj.U_packed" in st:
        lin(f"{p}.attn_q", "self_attn.q_proj")
        lin(f"{p}.attn_k", "self_attn.k_proj")
        lin(f"{p}.attn_v", "self_attn.v_proj")
        lin(f"{p}.attn_output", "self_attn.o_proj")
        plain(f"{p}.attn_q_norm.weight", st["self_attn.q_norm.weight"].float() + 1)
        plain(f"{p}.attn_k_norm.weight", st["self_attn.k_norm.weight"].float() + 1)
    else:
        lin(f"{p}.attn_qkv", "linear_attn.in_proj_qkv", rows=qkv_rows())
        lin(f"{p}.attn_gate", "linear_attn.in_proj_z", rows=v_perm(NK, HV))
        lin(f"{p}.ssm_out", "linear_attn.out_proj", cols=v_perm(NK, HV))
        ab = v_perm(NK, 1)
        plain(f"{p}.ssm_alpha.weight", st["linear_attn.in_proj_a.weight"].float()[ab])
        plain(f"{p}.ssm_beta.weight", st["linear_attn.in_proj_b.weight"].float()[ab])
        plain(f"{p}.ssm_a", -torch.exp(st["linear_attn.A_log"].float())[ab])
        plain(f"{p}.ssm_dt.bias", st["linear_attn.dt_bias"].float()[ab])
        plain(f"{p}.ssm_conv1d.weight", st["linear_attn.conv1d.weight"].float().squeeze(1)[qkv_rows()])
        plain(f"{p}.ssm_norm.weight", st["linear_attn.norm.weight"].float())
    plain(f"{p}.ffn_gate_inp.weight", st["mlp.gate.weight"].float())
    plain(f"{p}.ffn_gate_inp_shexp.weight", st["mlp.shared_expert_gate.weight"].float().reshape(-1))
    lin(f"{p}.ffn_gate_shexp", "mlp.shared_expert.gate_proj")
    lin(f"{p}.ffn_up_shexp", "mlp.shared_expert.up_proj")
    lin(f"{p}.ffn_down_shexp", "mlp.shared_expert.down_proj")
    n_exp = sum(1 for k in st if k.startswith("mlp.experts.") and k.endswith("gate_proj.U_packed"))
    for gg, hf in (("ffn_gate_exps", "gate_proj"), ("ffn_up_exps", "up_proj"), ("ffn_down_exps", "down_proj")):
        parts = [factored(st, f"mlp.experts.experts.{e}.{hf}") for e in range(n_exp)]
        for i, suf in enumerate(("nq_u", "nq_v", "nq_spre", "nq_spost")):
            a = torch.stack([q[i] for q in parts]).numpy()
            put(f"{p}.{gg}.{suf}", a.view(np.float32) if a.dtype == np.int32 else a)
        if CHECK and L in (0, 3):
            checks.append((f"{p}.{gg}.weight[e0]", dense(*parts[0], parts[0][2].numel())))
    print(f"layer {L}: {len(st)} tensors, {n_exp} experts", flush=True)

if CHECK:
    from gguf.quants import dequantize
    for name, mine in checks:
        base, _, sub = name.partition("[")
        t = ref_t[base]
        r = torch.from_numpy(dequantize(t.data, t.tensor_type).astype(np.float32)).reshape(-1)
        if sub:
            r = r.reshape(-1, *mine.shape)[0].reshape(-1)
        a = mine.reshape(-1)
        cos = torch.nn.functional.cosine_similarity(a, r, dim=0).item()
        rel = ((a - r).norm() / r.norm()).item()
        print(f"check {name:40s} cos {cos:.4f} rel {rel:.4f}")
    sys.exit(0)

for name, a in out:
    if isinstance(a, np.ndarray):
        w.add_tensor(name, a)
    else:
        w.add_tensor(name, a.data, raw_shape=a.data.shape, raw_dtype=a.tensor_type)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file(progress=False)
w.close()
print("wrote", out_path)
