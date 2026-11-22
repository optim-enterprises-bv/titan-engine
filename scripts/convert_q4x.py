"""Qwen3.8-Flash-Next (qwen4_exp) HF bf16 -> quantized GGUF in one pass, no Q8_0 intermediate.

llama.cpp's own converter does the name mapping and tensor transforms; its quantize step is redirected to
libggml's ggml_quantize_chunk (threaded over row chunks) with a per-tensor type policy:
  expert gate/up: GATEUP (default Q2_K), expert down (rows of 640): DOWN (default IQ4_NL),
  n-gram table (rows of 160): NGRAM (default IQ4_NL), token_embd / output: Q6_K, everything else 2-D: Q8_0.
IMATRIX=<llama-imatrix .gguf> passes importance weights per tensor (required for the IQ2 / IQ1 types).

usage: convert_q4x.py <hf dir> <out gguf>
"""
import ctypes, os, sys, concurrent.futures as cf
import numpy as np

LLAMA = os.path.expanduser("~/src/llama.cpp-inkling-atomic")
sys.path.insert(0, LLAMA)
sys.path.insert(0, LLAMA + "/gguf-py")
import gguf
import convert_hf_to_gguf as conv
from conversion import qwen4exp

Q = gguf.GGMLQuantizationType
POLICY = {
    "gate_up": Q[os.environ.get("GATEUP", "Q2_K")],
    "down": Q[os.environ.get("DOWN", "IQ4_NL")],
    "ngram": Q[os.environ.get("NGRAM", "IQ4_NL")],
    "embd": Q[os.environ.get("EMBD", "Q6_K")],
}
THREADS = int(os.environ.get("QTHREADS", "12"))
ROWS_PER_JOB = 2048

ggml = ctypes.CDLL(LLAMA + "/build/bin/libggml-base.so")
ggml.ggml_quantize_chunk.restype = ctypes.c_size_t
ggml.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
ggml.ggml_row_size.restype = ctypes.c_size_t
ggml.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
ggml.ggml_quantize_init.argtypes = [ctypes.c_int]
ggml.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
ggml.ggml_quantize_requires_imatrix.argtypes = [ctypes.c_int]

IMATRIX = {}
if os.environ.get("IMATRIX"):
    r = gguf.GGUFReader(os.environ["IMATRIX"])
    for t in r.tensors:
        if t.name.endswith(".in_sum2"):
            IMATRIX[t.name[: -len(".in_sum2")]] = np.asarray(t.data, dtype=np.float32).reshape(-1)
CURRENT = {"name": None}

orig_quantize = gguf.quants.quantize
NATIVE = {Q.F32, Q.F16, Q.BF16, Q.Q8_0}


def ggml_quantize(data, qtype):
    if qtype in NATIVE:
        return orig_quantize(data, qtype)
    name = CURRENT["name"]
    if isinstance(data, gguf.LazyNumpyTensor):
        def shape_fn(shape):
            return (*shape[:-1], ggml.ggml_row_size(int(qtype), shape[-1]))
        lazy = gguf.LazyNumpyTensor._wrap_fn(lambda d: ggml_quantize_eager(d, qtype, name), meta_noop=(np.uint8, shape_fn))
        return lazy(data)
    return ggml_quantize_eager(data, qtype, name)


def ggml_quantize_eager(data: np.ndarray, qtype, name) -> np.ndarray:
    data = np.ascontiguousarray(data, dtype=np.float32)
    n_per_row = data.shape[-1]
    nrows = data.size // n_per_row
    rb = ggml.ggml_row_size(int(qtype), n_per_row)
    out = np.empty((nrows, rb), dtype=np.uint8)
    ggml.ggml_quantize_init(int(qtype))
    imat = IMATRIX.get(name)
    if ggml.ggml_quantize_requires_imatrix(int(qtype)) and imat is None:
        raise RuntimeError(f"{name}: {qtype.name} needs an imatrix")
    flat = data.reshape(nrows, n_per_row)
    experts = data.shape[0] if data.ndim == 3 else 1
    rows_per_expert = nrows // experts

    def job(r0):
        r1 = min(r0 + ROWS_PER_JOB, nrows)
        w = None
        if imat is not None:
            e = r0 // rows_per_expert
            w = imat[e * n_per_row:(e + 1) * n_per_row] if imat.size >= experts * n_per_row and experts > 1 else imat[:n_per_row]
            w = np.ascontiguousarray(w, dtype=np.float32)
        src = flat[r0:r1]
        dst = out[r0:r1]
        ggml.ggml_quantize_chunk(int(qtype), src.ctypes.data, dst.ctypes.data, 0, r1 - r0, n_per_row,
                                 w.ctypes.data if w is not None else None)

    # an expert stack never mixes experts inside one job (per-expert imatrix rows)
    starts = []
    for e0 in range(0, nrows, rows_per_expert):
        starts += list(range(e0, e0 + rows_per_expert, ROWS_PER_JOB))
    with cf.ThreadPoolExecutor(THREADS) as ex:
        list(ex.map(job, starts))
    return out.reshape(*data.shape[:-1], rb)


gguf.quants.quantize = ggml_quantize

base_force = qwen4exp.Qwen4ExpTextModel.tensor_force_quant


def force_quant(self, name, new_name, bid, n_dims):
    CURRENT["name"] = new_name.removesuffix(".weight")
    if "_exps" in new_name and n_dims >= 2:
        return POLICY["down"] if "ffn_down" in new_name else POLICY["gate_up"]
    if new_name.startswith("per_layer_token_embd"):
        return POLICY["ngram"]
    if new_name in ("token_embd.weight", "output.weight"):
        return POLICY["embd"]
    return base_force(self, name, new_name, bid, n_dims)


qwen4exp.Qwen4ExpTextModel.tensor_force_quant = force_quant

base_params = qwen4exp.Qwen4ExpTextModel.set_gguf_parameters


def set_params(self):
    # transformers renames the checkpoint's "full_attention" layers to "indexed_attention" (QSA); the converter only
    # recognises the old name and would write compress ratio 0 (dense attention) for every trunk attention layer
    lt = self.hparams.get("layer_types")
    if lt:
        self.hparams["layer_types"] = ["full_attention" if t == "indexed_attention" else t for t in lt]
    base_params(self)


qwen4exp.Qwen4ExpTextModel.set_gguf_parameters = set_params

if __name__ == "__main__":
    hf, out = sys.argv[1], sys.argv[2]
    sys.argv = ["convert_hf_to_gguf.py", hf, "--outfile", out, "--outtype", "q8_0"] + sys.argv[3:]
    conv.main()
