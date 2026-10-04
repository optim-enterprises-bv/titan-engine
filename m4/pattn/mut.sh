# Mutation checks for the v0.9.4 paged-attn ports (sourced by a window script after lib.sh). Each mutant is a
# plausible wrong version applied with python to a reflink copy of oxide-pattn (oxide-pattn-mut, rebuilt per mutant
# from the current tree), built with a reduced kernel set, and gated on the affected families: the gate MUST fail.
MX=$E/oxide-pattn-mut
mut_prepare() { # CRATE: fresh copy of the crate (+ shared dirs by symlink), reusing its built target if present
  local c=$1
  rm -rf $MX/$c; mkdir -p $MX
  for d in kdiff tools reference; do [ -e $MX/$d ] || ln -s $OX/$d $MX/$d; done
  cp -a --reflink=auto $OX/$c $MX/$c
}
mut_apply() { # CRATE FILE OLD NEW (python literal replace, must match exactly once)
  python3 - "$MX/$1/$2" "$3" "$4" <<'PY'
import sys
p, a, b = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
assert s.count(a) == 1, (p, s.count(a), a)
open(p, "w").write(s.replace(a, b))
PY
}
mut_run() { # NAME CRATE GATE_ENV...: build the mutant and run its gate; PASS of the mutation check = gate FAILS
  local n=$1 c=$2; shift 2
  [ $(left) -gt 420 ] || { echo "MUTANT $n skipped: $(left)s left"; return 1; }
  if ! oxbuild $MX $c pattn-ox 900; then echo "MUTANT $n: build failed (see $O/oxbuild-$(basename $MX)-$c.log)"; return 1; fi
  kgate $MX $c $O/mut-$n.log 900 "$@"
  if grep -q " 0 launcher calls" $O/mut-$n.log; then echo "MUTANT $n: INVALID (the gate launched nothing)"
  elif grep -q -- "-> FAIL" $O/mut-$n.log; then echo "MUTANT $n: DETECTED ($(grep -c 'FAIL\|first diff' $O/mut-$n.log) fail lines)"
  elif grep -q "Rust side: " $O/mut-$n.log; then echo "MUTANT $n: DETECTED (the Rust side crashed: $(grep -o 'Rust side: .*' $O/mut-$n.log | head -1 | cut -c1-120))"
  else echo "MUTANT $n: NOT DETECTED ($(grep -m1 panicked -A1 $O/mut-$n.log | tail -1 | cut -c1-160))"; fi
}
mutants_b_vscale() { # on the f32 head-128 group-4 instance: f32 output keeps the 1-ulp reassociation visible
  local c=mistralrs-paged-attn-b
  # (+ the split-KV merge kernels the decode cases reach)
  local only='ELj2ELj1ELj4ELj32ELj4ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsIfffiEE|KernelMLAILj2ELj16ELj2ELj32ELj8ELj1ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_20BatchDecodeParamsMLAIfffi|PersistentVariableLengthMergeStatesKernel'
  mut_prepare $c; (cd $MX/$c && ONLY="$only" python3 gen_kernels.py > /dev/null)
  mut_apply $c src/main.rs "st_o[i] = fmul(fmul(st_o[i], d_rcp), v_scale);" "st_o[i] = fmul(st_o[i], fmul(d_rcp, v_scale));"
  mut_run b-vscale-reassoc $c MPB_ONLY=decode,edges MPB_HD=128 MPB_QUICK=1
}
mutants_b() {
  local c=mistralrs-paged-attn-b
  # reduced kernel set: every reshape / gather instance, one f16 head-128 group-4 decode instance (no SW / SC)
  # (+ one MLA instance: launch::available() loads the module through it)
  local only='reshape_and_cache|gather_kv_cache|ELj2ELj1ELj8ELj16ELj4ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half|KernelMLAILj2ELj16ELj2ELj32ELj8ELj1ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_20BatchDecodeParamsMLAIfffi'
  mut_prepare $c; (cd $MX/$c && ONLY="$only" python3 gen_kernels.py > /dev/null)
  mut_apply $c src/main.rs "f2e4m3(fdiv(k.to_f32(), k_scale))" "f2e4m3(k.to_f32() / k_scale)"
  mut_run b-fp8write-ieee-div $c MPB_ONLY=reshape
  mut_prepare $c; (cd $MX/$c && ONLY="$only" python3 gen_kernels.py > /dev/null)
  mut_apply $c src/main.rs "O::from_f32(fmul(k_scale, e4m32f(kb)))" "O::from_f32(k_scale * e4m32f(kb))"
  mut_run b-fp8read-noftz $c MPB_ONLY=gather
  mut_prepare $c; (cd $MX/$c && ONLY="$only" python3 gen_kernels.py > /dev/null)
  mut_apply $c src/main.rs "                *k_out.offset(out_idx as isize) = O::zero();
                *v_out.offset(out_idx as isize) = O::zero();" ""
  mut_run b-gather-nozerofill $c MPB_ONLY=gather
  mutants_b_vscale
}
mutants_a() {
  local c=mistralrs-paged-attn-a
  mut_prepare $c; (cd $MX/$c && GEN_HEADS=64 GEN_BLOCKS=16 python3 gen_kernels.py > /dev/null)
  mut_apply $c src/main.rs "{ copy_blocks::<u8>(k, v, m, nk, nv) }" "{ copy_blocks::<i16>(k, v, m, nk, nv) }"
  mut_run a-copyu8-as-i16 $c MPA_ONLY=copy_blocks
  mut_prepare $c; (cd $MX/$c && GEN_HEADS=64 GEN_BLOCKS=16 python3 gen_kernels.py > /dev/null)
  mut_apply $c src/main.rs "        if batch_id >= num_seqs {
            return;
        }
" ""
  mut_run a-gather-noguard $c MPA_ONLY=gather_kv_cache
}
mutants_core() { # v0.9.4 graph / input-packing helpers
  local c=mistralrs-core-cuda
  mut_prepare $c
  mut_apply $c src/main.rs "let source_row = if row < input_rows { row } else { 0 };" "let source_row = if row < input_rows { row } else { input_rows - 1 };"
  mut_run core-pad-lastrow $c GATE_ONLY=graph GATE_ROUNDS=1
  mut_prepare $c
  mut_apply $c src/main.rs "*(staged[row as usize] as *const u32).offset(column.wrapping_sub(host_width) as isize)" "*(staged[(row as usize) % 64] as *const u32).offset(column.wrapping_sub(host_width).wrapping_add(1) as isize)"
  mut_run core-pack-shift $c GATE_ONLY=graph GATE_ROUNDS=1
}
mutants_fp8dec() { # FP8 decode: element order inside a 32-bit word (big- instead of little-endian bytes)
  local c=mistralrs-paged-attn-b
  local only='ELj2ELj1ELj16ELj8ELj4ELj4ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecodeParamsI6__half13__nv_fp8_e4m3|KernelMLAILj2ELj16ELj2ELj32ELj8ELj1ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_20BatchDecodeParamsMLAIfffi|PersistentVariableLengthMergeStatesKernel'
  mut_prepare $c; (cd $MX/$c && ONLY="$only" python3 gen_kernels.py > /dev/null)
  mut_apply $c src/main.rs "e4m32f(((w >> (8 * k)) & 0xff) as u8)" "e4m32f(((w >> (24 - 8 * k)) & 0xff) as u8)"
  mut_run b-fp8dec-byteorder $c MPB_ONLY=kernels
}
