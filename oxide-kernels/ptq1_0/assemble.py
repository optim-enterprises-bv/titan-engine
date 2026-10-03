#!/usr/bin/env python3
"""assemble.py: build src/main.rs of the ptq1_0 crate from the q1_0 crate's shared pieces (float helpers,
quantizers, the acecd56 MMQ core) plus src/new_kernels.rs.in (PT quantizer, FWHT, PT mat-vec) and the PTQ1_0
MMQ tile loader, then run gen_kernels.py for the explicit #[kernel] wrappers. One-shot scaffolding: edit
src/main.rs directly afterwards."""
import os, re, subprocess
D = os.path.dirname(os.path.abspath(__file__))
q = open(os.path.join(D, "../q1_0/src/main.rs")).read().split("\n")
seg = lambda a, b: "\n".join(q[a - 1:b]) + "\n"
h1 = seg(51, 134)      # h2f .. Dst/H/B
h2 = seg(360, 445)     # unroll!, Src, absf .. round_mul
qm = seg(508, 568)     # quantize_mmq_d4 + kernels
mmq = seg(570, 1121)   # MMQ core
new = open(os.path.join(D, "src/new_kernels.rs.in")).read()
new = new.replace("let srow = signs.offset(((r as i32) % n_blk.max(1)) as isize * 1024);",
                  "let srow = signs.wrapping_offset(((r as i32) % n_blk.max(1)) as isize * 1024);")
assert new.count("signs.wrapping_offset") == 2
qm = qm.replace("q1_0_quantize_mmq_d4_", "ptq1_0_quantize_mmq_d4_")
qm = qm.replace("ne2: i32, _n_expert_used: i32)", "ne2: i32, _n_expert_used: i32, _gate: u64, _norm_weight: u64, _norm_scale: u64)")
assert qm.count("_norm_scale: u64") == 3
qm = re.sub(r"    #\[kernel\]\n    pub unsafe fn ptq1_0_quantize_mmq_d4_f16\(.*?\n    \}\n", "", qm, flags=re.S)
assert "d4_f16" not in qm
mmq = mmq.replace("x.wrapping_offset(off.wrapping_mul(18) as isize)", "x.wrapping_offset(off.wrapping_mul(28) as isize)")
for a, b in [("block_q1_0 *", "block_ptq1_0 *"), ("GGML_TYPE_Q1_0", "GGML_TYPE_PTQ1_0"), ("two Q1_0 blocks", "two PTQ1_0 blocks"),
             ("MMQ_ITER_K / QK1_0", "MMQ_ITER_K / QK_PTQ1_0"), ("mul_mat_q<Q1_0,", "mul_mat_q<PTQ1_0,"),
             ("mul_mat_q_stream_k_fixup<Q1_0,", "mul_mat_q_stream_k_fixup<PTQ1_0,")]:
    mmq = mmq.replace(a, b)
start = mmq.index("    /// ggml_cuda_mmq_load_tiles_q1_0 (MMA layout).")
end = mmq.index("    /// ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma")
loader = open(os.path.join(D, "src/loader.rs.in")).read()
mmq = mmq[:start] + loader + mmq[end:]
header = open(os.path.join(D, "src/header.rs.in")).read()
footer = """
    // One entry per reference instance (gen_kernels.py).
    // GENERATED KERNELS BEGIN
    // GENERATED KERNELS END
}

fn main() {
    std::process::exit(if gate::run() { 0 } else { 1 });
}
"""
open(os.path.join(D, "src/main.rs"), "w").write(header + h1 + "\n" + h2 + "\n" + new + qm + "\n" + mmq + footer)
subprocess.run(["python3", os.path.join(D, "gen_kernels.py")], check=True, cwd=D)
