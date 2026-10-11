#!/bin/sh
# Measure your own implementation against the oracle.
#
# usage: MYLLM=yourmodule harness/run_mine.sh [CASES_PER_SEED]
#
# MYLLM names a Python module with the functions of harness/myllm_template.py
# (default: the template itself). Requires a built repository (`lake build`).
# TOL sets the float tolerance (default 1e-6; use 1e-4 for float32/bf16 code).
# Exits non-zero on any divergence, and also if a planted defect goes undetected.
set -eu

cd "$(dirname "$0")/.."
N="${1:-200}"
TOL="${TOL:-1e-6}"
export MYLLM="${MYLLM:-myllm_template}"
OUT=harness/out-mine
mkdir -p "$OUT"
H=harness

merge_seeds() {  # kind outfile
  : > "$2"
  for seed in 1 2 3 4; do
    python3 "$H/gen.py" "$1" "$seed" "$N" | sed "s/^@$1 gen-\([a-z]*\)-/@$1 gen-\1-s$seed-/" >> "$2"
  done
}

cmp() {  # kind prefix [flags...]
  k=$1; p=$2; shift 2
  python3 "$H/compare.py" "$OUT/$k.oracle" "$OUT/$k.mine" "$p" "$@" --label "$MYLLM $k/$p"
}

echo "== implementation: $MYLLM, $N cases per seed, 4 seeds, tolerance $TOL"
status=0
for k in sample bpe attention speculative chat diffusion; do
  merge_seeds "$k" "$OUT/$k.llm"
  lake env lean --run LLM/Driver.lean "$OUT/$k.llm" > "$OUT/$k.oracle"
  python3 "$H/impl_mine.py" "$OUT/$k.llm" > "$OUT/$k.mine"
done

echo "== sampler"
cmp sample p --float "$TOL" --ties-bin || status=1
python3 "$H/explain_ties.py" "$OUT/sample.llm" "$OUT/sample.oracle" "$OUT/sample.mine" "$TOL" || status=1

echo "== tokenizer (passes if one BPE convention agrees on every case)"
bpe_ok=1
for p in tokens tokens-lookup tiktoken; do
  if cmp bpe "$p" > "$OUT/bpe.$p.txt"; then bpe_ok=0; fi
  head -n 1 "$OUT/bpe.$p.txt"
done
[ $bpe_ok -eq 0 ] || { echo "   no convention agrees; details in $OUT/bpe.*.txt"; status=1; }

echo "== attention"
for p in weight out blocked; do cmp attention "$p" --float "$TOL" || status=1; done

echo "== speculative decoding"
for p in rate accept residual out; do cmp speculative "$p" --float "$TOL" || status=1; done

echo "== chat transcripts"
for p in valid complete; do cmp chat "$p" || status=1; done

echo "== masked diffusion"
cmp diffusion marg --float "$TOL" || status=1

echo "== planted defects (must be detected)"
for kp in sample:p attention:weight bpe:tokens; do
  k=${kp%%:*}; p=${kp#*:}
  python3 "$H/impl_mine.py" "$OUT/$k.llm" --planted-defect > "$OUT/$k.planted"
  f="--float $TOL"; [ "$k" = bpe ] && f=""
  # shellcheck disable=SC2086
  if python3 "$H/compare.py" "$OUT/$k.oracle" "$OUT/$k.planted" "$p" $f > /dev/null; then
    echo "FAIL: planted $k defect not detected"; status=1
  else
    echo "ok: planted $k defect detected"
  fi
done

exit $status
