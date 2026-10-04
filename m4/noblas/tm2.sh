source $(dirname $0)/tmcmp.sh true
MM=$HOME/ai/models; Q=$MM/Qwen3.6-35B-A3B-MTP; O=..
declare -A GG=( [qwen3.6-35b]=$Q/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf [qwen3.6-35b-iq2m]=$MM/q35-lowbit/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
  [qwen3.6-35b-mxfp4]=$Q/Qwen3.6-35B-A3B-MXFP4_MOE.gguf [gpt-oss-20b]=$MM/gpt-oss-20b-F16.gguf [bonsai-27b]=$MM/bonsai/Bonsai-27B-gguf/Bonsai-27B-Q1_0.gguf
  [bonsai2-27b]=$Q/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf [qwen3-next-80b]=$MM/qwen3-next-80b/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf
  [gpt-oss-120b]=$MM/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf )
declare -A RF=( [qwen3.6-35b]=q35 [qwen3.6-35b-iq2m]=iq2m [qwen3.6-35b-mxfp4]=mx [gpt-oss-20b]=oss20 [bonsai-27b]=bonsai [bonsai2-27b]=bonsai2 [qwen3-next-80b]=next [gpt-oss-120b]=oss120 )
for m in "${!RF[@]}"; do
  for t in new old; do for g in g1 g3; do [ -e $O/$t-$m-$g.json ] && cp $O/$t-$m-$g.json . ; done; done
  tm gguf:${GG[$m]} new-$m-g1.json old-$m-g1.json new-$m-g3.json old-$m-g3.json
  python3 $(dirname $0)/../../sp.py new-$m-g1.json old-$m-g1.json new-$m-g3.json old-$m-g3.json
  for t in new old; do for g in g1 g3; do r=$O/lref-${RF[$m]}-$g.json; [ -e $r ] && [ -e $t-$m-$g.json ] && echo "$m $t $(cmpg $g $t-$m-$g.json $r)"; done; done
done
