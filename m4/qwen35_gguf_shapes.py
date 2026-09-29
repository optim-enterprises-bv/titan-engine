#!/usr/bin/env python3
"""Check a qwen35 (dense) or qwen35moe GGUF against the tensors quantized_qwen35_moe.rs requests.

Reads only the GGUF header (metadata + tensor infos); no tensor data is loaded.

    python3 qwen35_gguf_shapes.py [path.gguf] [--dump]

Shapes are GGUF `ne` order (ne0 first = the contiguous / input dimension),
which is the reverse of candle's shape for the same tensor.
"""
import os
import struct
import sys

DEFAULT = os.path.expanduser("~/ai/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")

GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0",
    35: "TQ2_0", 39: "MXFP4", 41: "Q1_0",
}
# Types candle's GGUF loader (and hence mistral.rs) can load.
CANDLE_TYPES = {"F32", "F16", "BF16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0",
                "Q8_1", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K", "Q1_0"}
TIERED_TYPES = {"Q4_K", "Q5_K"}  # mistralrs_quant::TieredExperts::supports


class Reader:
    def __init__(self, f):
        self.f = f

    def read(self, fmt):
        size = struct.calcsize(fmt)
        return struct.unpack("<" + fmt, self.f.read(size))

    def u32(self):
        return self.read("I")[0]

    def u64(self):
        return self.read("Q")[0]

    def string(self):
        n = self.u64()
        return self.f.read(n).decode("utf-8", errors="replace")

    def value(self, t):
        scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
                  7: "?", 10: "Q", 11: "q", 12: "d"}
        if t in scalar:
            return self.read(scalar[t])[0]
        if t == 8:
            return self.string()
        if t == 9:
            et = self.u32()
            n = self.u64()
            if et in scalar and n > 64:
                # large arrays (tokenizer): skip without materialising strings
                self.f.seek(n * struct.calcsize(scalar[et]), 1)
                return f"<array {n}>"
            if et == 8 and n > 64:
                for _ in range(n):
                    self.f.seek(self.u64(), 1)
                return f"<string array {n}>"
            return [self.value(et) for _ in range(n)]
        raise ValueError(f"unknown gguf value type {t}")


def read_header(path):
    with open(path, "rb") as f:
        r = Reader(f)
        if f.read(4) != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF file")
        version = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()
        meta = {}
        for _ in range(n_kv):
            k = r.string()
            meta[k] = r.value(r.u32())
        tensors = {}
        offsets = {}
        for _ in range(n_tensors):
            name = r.string()
            nd = r.u32()
            dims = [r.u64() for _ in range(nd)]
            ty = r.u32()
            off = r.u64()
            tensors[name] = (dims, GGML_TYPES.get(ty, f"type{ty}"))
            offsets[name] = off
        align = meta.get("general.alignment", 32)
        pos = f.tell()
        data_start = (pos + align - 1) // align * align
    return version, meta, tensors, offsets, data_start


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else DEFAULT
    version, meta, tensors, offsets, data_start = read_header(path)

    if "--dump" in sys.argv:
        print(f"GGUF v{version}, {len(tensors)} tensors")
        for k, v in meta.items():
            if not k.startswith("tokenizer."):
                print(f"  {k} = {v}")
        for n, (d, t) in tensors.items():
            print(f"  {n:45s} {t:6s} {d}")
        return

    arch = meta["general.architecture"]
    assert arch in ("qwen35", "qwen35moe"), arch
    is_moe = arch == "qwen35moe"
    p = lambda k: meta[f"{arch}.{k}"]

    # --- hyper-parameters the loader reads (and the relations it relies on) ---
    n_layer = p("block_count")
    hidden = p("embedding_length")
    n_head = p("attention.head_count")
    n_kv = p("attention.head_count_kv")
    head_dim = p("attention.key_length")
    assert p("attention.value_length") == head_dim
    n_rot = p("rope.dimension_count")
    n_exp = p("expert_count") if is_moe else 0
    n_exp_used = p("expert_used_count") if is_moe else 0
    ff_exp = p("expert_feed_forward_length") if is_moe else 0
    ff_sh = p("expert_shared_feed_forward_length") if is_moe else 0
    n_ff = meta.get(f"{arch}.feed_forward_length")
    d_conv = p("ssm.conv_kernel")
    d_state = p("ssm.state_size")          # head_k_dim == head_v_dim
    n_k_heads = p("ssm.group_count")
    n_v_heads = p("ssm.time_step_rank")
    d_inner = p("ssm.inner_size")
    interval = p("full_attention_interval")
    assert d_inner == n_v_heads * d_state, "loader assumes head_v_dim == ssm.state_size"
    assert n_v_heads % n_k_heads == 0
    assert n_rot % 2 == 0 and n_rot <= head_dim
    sections = meta.get(f"{arch}.rope.dimension_sections")
    recurrent = meta.get(f"{arch}.attention.recurrent_layers")  # optional override
    print(f"arch={arch} layers={n_layer} hidden={hidden} heads={n_head}/{n_kv} head_dim={head_dim} "
          f"n_rot={n_rot} sections={sections} experts={n_exp}x{n_exp_used} ff_exp={ff_exp} ff_sh={ff_sh} ff={n_ff}")
    print(f"gdn: k_heads={n_k_heads} v_heads={n_v_heads} state={d_state} conv={d_conv} interval={interval} "
          f"rope_base={p('rope.freq_base')} eps={p('attention.layer_norm_rms_epsilon')} "
          f"ctx={p('context_length')}")

    key_dim = n_k_heads * d_state
    value_dim = n_v_heads * d_state
    conv_dim = 2 * key_dim + value_dim

    expected = {
        "token_embd.weight": [hidden, None],
        "output_norm.weight": [hidden],
    }
    optional = {"output.weight": [hidden, None]}
    vocab = tensors["token_embd.weight"][0][1]
    expected["token_embd.weight"][1] = vocab
    optional["output.weight"][1] = vocab

    layer_kind = []
    for i in range(n_layer):
        is_recr = bool(recurrent[i]) if isinstance(recurrent, list) else (i + 1) % interval != 0
        layer_kind.append("gdn" if is_recr else "attn")
        b = f"blk.{i}."
        expected[b + "attn_norm.weight"] = [hidden]
        expected[b + "post_attention_norm.weight"] = [hidden]
        if is_recr:
            expected[b + "attn_qkv.weight"] = [hidden, conv_dim]
            expected[b + "attn_gate.weight"] = [hidden, value_dim]
            expected[b + "ssm_a"] = [n_v_heads]
            expected[b + "ssm_alpha.weight"] = [hidden, n_v_heads]
            expected[b + "ssm_beta.weight"] = [hidden, n_v_heads]
            expected[b + "ssm_conv1d.weight"] = [d_conv, conv_dim]
            expected[b + "ssm_dt.bias"] = [n_v_heads]
            expected[b + "ssm_norm.weight"] = [d_state]
            expected[b + "ssm_out.weight"] = [value_dim, hidden]
        else:
            expected[b + "attn_q.weight"] = [hidden, 2 * n_head * head_dim]
            expected[b + "attn_k.weight"] = [hidden, n_kv * head_dim]
            expected[b + "attn_v.weight"] = [hidden, n_kv * head_dim]
            expected[b + "attn_q_norm.weight"] = [head_dim]
            expected[b + "attn_k_norm.weight"] = [head_dim]
            expected[b + "attn_output.weight"] = [n_head * head_dim, hidden]
        if b + "ffn_gate_exps.weight" in tensors:  # the loader picks Moe vs Dense by this tensor
            expected[b + "ffn_gate_inp.weight"] = [hidden, n_exp]
            expected[b + "ffn_gate_exps.weight"] = [hidden, ff_exp, n_exp]
            expected[b + "ffn_up_exps.weight"] = [hidden, ff_exp, n_exp]
            expected[b + "ffn_down_exps.weight"] = [ff_exp, hidden, n_exp]
            expected[b + "ffn_gate_shexp.weight"] = [hidden, ff_sh]
            expected[b + "ffn_up_shexp.weight"] = [hidden, ff_sh]
            expected[b + "ffn_down_shexp.weight"] = [ff_sh, hidden]
            expected[b + "ffn_gate_inp_shexp.weight"] = [hidden]
        else:
            expected[b + "ffn_gate.weight"] = [hidden, n_ff]
            expected[b + "ffn_up.weight"] = [hidden, n_ff]
            expected[b + "ffn_down.weight"] = [n_ff, hidden]

    errors = []
    for name, shape in expected.items():
        if name not in tensors:
            errors.append(f"MISSING {name}")
            continue
        dims, ty = tensors[name]
        if dims != shape:
            errors.append(f"SHAPE {name}: gguf {dims} != expected {shape}")
        if ty not in CANDLE_TYPES:
            errors.append(f"DTYPE {name}: {ty} not loadable by candle")
    for name, shape in optional.items():
        if name in tensors and tensors[name][0] != shape:
            errors.append(f"SHAPE {name}: gguf {tensors[name][0]} != expected {shape}")

    # the V-head un-reorder permutes whole quant blocks: head_v_dim must be block aligned
    blk = {"Q8_0": 32, "Q4_0": 32, "Q4_1": 32, "Q5_0": 32, "Q5_1": 32, "Q8_1": 32, "F32": 1,
           "F16": 1, "BF16": 1, "Q2_K": 256, "Q3_K": 256, "Q4_K": 256, "Q5_K": 256,
           "Q6_K": 256, "Q8_K": 256, "Q1_0": 128}
    for i, kind in enumerate(layer_kind):
        if kind != "gdn":
            continue
        name = f"blk.{i}.ssm_out.weight"
        if name in tensors and d_state % blk.get(tensors[name][1], 1 << 30) != 0:
            errors.append(f"BLOCK {name}: {tensors[name][1]} blocks do not tile head_v_dim={d_state}")

    unused = sorted(set(tensors) - set(expected) - set(optional))
    print(f"layers: {layer_kind.count('gdn')} gdn, {layer_kind.count('attn')} full-attention "
          f"(attn at {[i for i, k in enumerate(layer_kind) if k == 'attn']})")
    print(f"output.weight present: {'output.weight' in tensors}; vocab={vocab}")

    # which layers fall back from TieredExperts to stock experts
    stock = []
    for i in range(n_layer if is_moe else 0):
        tys = [tensors.get(f"blk.{i}.ffn_{p}_exps.weight", (None, None))[1] for p in ("gate", "up", "down")]
        if not all(t in TIERED_TYPES for t in tys):
            stock.append((i, tys))
    print(f"expert dtype fallback to Stock (TITAN_TIERED=1): {len(stock)} layers")
    for i, tys in stock:
        print(f"  blk.{i}: gate/up/down = {tys}")
    types = {}
    for name, (d, t) in tensors.items():
        key = name.split(".", 2)[-1] if name.startswith("blk.") else name
        types.setdefault(key, set()).add(t)
    print("dtypes per tensor kind:")
    for k in sorted(types):
        print(f"  {k:32s} {sorted(types[k])}")
    if unused:
        print(f"tensors present but not loaded ({len(unused)}): {unused[:10]}{' ...' if len(unused) > 10 else ''}")

    # --- value spot-checks (a few KB of F32 data) for the converter transforms the loader undoes ---
    def f32(name):
        dims, ty = tensors[name]
        assert ty == "F32", (name, ty)
        n = 1
        for d in dims:
            n *= d
        with open(path, "rb") as f:
            f.seek(data_start + offsets[name])
            return struct.unpack(f"<{n}f", f.read(4 * n))
    gdn0 = layer_kind.index("gdn")
    att0 = layer_kind.index("attn")
    mean = lambda v: sum(v) / len(v)
    a = f32(f"blk.{gdn0}.ssm_a")
    print(f"ssm_a[blk.{gdn0}] range [{min(a):.4g}, {max(a):.4g}] (converter stores -exp(A_log): must be < 0)")
    if not all(x < 0 for x in a):
        errors.append("VALUE ssm_a not all negative: loader's a_log = ln(-ssm_a) would be NaN")
    for nm in (f"blk.{gdn0}.attn_norm.weight", f"blk.{gdn0}.post_attention_norm.weight",
               f"blk.{att0}.attn_q_norm.weight", f"blk.{att0}.attn_k_norm.weight",
               "output_norm.weight", f"blk.{gdn0}.ssm_norm.weight"):
        v = f32(nm)
        print(f"  mean({nm}) = {mean(v):.4f}")
    # Gemma-style (1+w) norms are baked by the converter (+1); if these sat near 0 the
    # loader would need to add 1 itself.
    if mean(f32(f"blk.{gdn0}.attn_norm.weight")) < 0.2:
        errors.append("VALUE attn_norm mean ~0: +1 offset NOT baked in; loader must add 1")

    if errors:
        print(f"FAIL: {len(errors)} problems")
        for e in errors:
            print("  " + e)
        sys.exit(1)
    print(f"OK: all {len(expected)} required tensors present with expected shapes")


if __name__ == "__main__":
    main()
