#!/usr/bin/env python3
"""Share of GPU kernel time per matmul op class in a service nsys profile (default: the 35B with MTP=2,
m4/graph/out/nsys-mtp2.sqlite, i.e. the titan-mistral config), keyed to the kbench op names.

    python3 weights.py [profile.sqlite]
"""
import collections, json, os, sqlite3, sys

K = os.path.dirname(os.path.abspath(__file__))
DB = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/titan-engine/m4/graph/out/nsys-mtp2.sqlite")
c = sqlite3.connect(DB)
tot = c.execute("select sum(end-start) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]


def op_of(name, gx, gy, gz):
    if name.startswith("q4k_q8_1_moe_gemv") or name == "q5k_q8_1_moe_gemv":  # gate/up (one layer is Q5_K)
        return "q35.exp_gate_up.q4_K.k2048n512"
    if name == "q5k_q8_1_moe_gemv_w":
        return "q35.exp_down.q5_K.k512n2048"
    if name == "q6k_q8_1_moe_gemv":
        return "q35.exp_down.q6_K.k512n2048"
    if name.startswith("indexed_moe_forward"):
        return "mtp_block.experts(stock indexed_moe)"
    if "quantize_q8_1" in name or name.startswith("quantize_mmq"):
        return "activation quantize (q8_1)"
    if name.startswith("mmq_") :
        return "dense prefill mmq (q8_0)"
    if name.startswith("mmvq_gguf_q8_0"):
        dst = "bf16" if "_bf16_" in name else "f32"
        rows = gx * 2
        if rows == 248320:
            return "q35.lm_head.q8_0.k2048n248320"
        if rows == 8192:
            return "q35.qkv.q8_0.k2048n8192"
        if rows == 4096:
            return "q35.attn_gate.q8_0.k2048n4096"
        if rows == 2048:
            # bf16 dst: ssm_out / attn_output (k=4096); f32 dst: shared-expert down (k=512)
            return "q35.ssm_out.q8_0.k4096n2048" if dst == "bf16" else "q35.shexp_down.q8_0.k512n2048"
        if rows == 512:
            return "q35.shexp_gate_up.q8_0.k2048n512"
        return f"mmvq q8_0 rows={rows}"
    return None


share = collections.defaultdict(float)
calls = collections.Counter()
for name, gx, gy, gz, t, n in c.execute("""select s.value, k.gridX, k.gridY, k.gridZ, sum(k.end-k.start), count(*)
        from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id = k.shortName group by s.value, k.gridX, k.gridY, k.gridZ"""):
    op = op_of(name, gx, gy, gz) or "other (non-matmul)"
    share[op] += t / tot
    calls[op] += n
out = {op: round(100 * v, 2) for op, v in sorted(share.items(), key=lambda x: -x[1])}
json.dump({"profile": DB, "total_gpu_ms": tot / 1e6, "share_pct": out}, open(f"{K}/out/weights.json", "w"), indent=1)
print(f"profile {DB}: {tot/1e6:.0f} ms GPU kernel time")
for op, v in out.items():
    print(f"{v:6.2f}%  {calls[op]:7d} launches  {op}")
