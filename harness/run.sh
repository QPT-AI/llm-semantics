#!/bin/sh
# Measure implementations against the oracle.
#
# usage: harness/run.sh [CASES_PER_SEED]
#
# Requires a built repository (`lake build`), Python with `tokenizers` and
# `tiktoken`, and optionally LLAMA_LIB pointing at a llama.cpp `libllama.so`.
# Exits non-zero on any unexplained divergence, and also if a planted defect goes
# undetected.
set -eu

cd "$(dirname "$0")/.."
N="${1:-500}"
OUT=harness/out
mkdir -p "$OUT"
H=harness

oracle() {
  lake env lean --run LLM/Driver.lean "$1" > "$2"
}

merge_seeds() {  # kind outfile
  : > "$2"
  for seed in 1 2 3 4; do
    python3 "$H/gen.py" "$1" "$seed" "$N" | sed "s/^@$1 gen-\([a-z]*\)-/@$1 gen-\1-s$seed-/" >> "$2"
  done
}

echo "== revisions"
echo "llm-semantics $(git rev-parse --short HEAD 2>/dev/null || echo unversioned)"
echo "lean $(lake env lean --version | head -n 1)"
python3 -c "import tokenizers, tiktoken; print('tokenizers', tokenizers.__version__); print('tiktoken', tiktoken.__version__)"

echo "== generating corpora ($N cases per seed, 4 seeds)"
merge_seeds bpe "$OUT/bpe.llm"
merge_seeds sample "$OUT/sample.llm"

echo "== running the oracle"
oracle "$OUT/bpe.llm" "$OUT/bpe.oracle"
oracle "$OUT/sample.llm" "$OUT/sample.oracle"

status=0

echo "== tokenizers"
python3 "$H/impl_hf.py" "$OUT/bpe.llm" > "$OUT/bpe.hf"
python3 "$H/impl_tiktoken.py" "$OUT/bpe.llm" > "$OUT/bpe.tiktoken"
python3 "$H/compare.py" "$OUT/bpe.oracle" "$OUT/bpe.hf" tokens --label "HF tokenizers, default" || status=1
python3 "$H/compare.py" "$OUT/bpe.oracle" "$OUT/bpe.hf" tokens-lookup --label "HF tokenizers, ignore_merges" || status=1
python3 "$H/compare.py" "$OUT/bpe.oracle" "$OUT/bpe.tiktoken" tiktoken --label "tiktoken" || status=1

echo "== planted defect (must be detected)"
python3 "$H/impl_hf.py" "$OUT/bpe.llm" --planted-defect > "$OUT/bpe.hf-planted"
if python3 "$H/compare.py" "$OUT/bpe.oracle" "$OUT/bpe.hf-planted" tokens --label "planted" > /dev/null; then
  echo "FAIL: the planted tokenizer defect was not detected"; status=1
else
  echo "ok: planted tokenizer defect detected"
fi

if [ -n "${LLAMA_LIB:-}" ]; then
  echo "== llama.cpp samplers ($LLAMA_LIB)"
  python3 "$H/impl_llama.py" "$OUT/sample.llm" > "$OUT/sample.llama"
  python3 "$H/compare.py" "$OUT/sample.oracle" "$OUT/sample.llama" p --float 1e-6 --ties-bin \
    --label "llama.cpp vs tie-keeping semantics" | head -n 1 || true
  python3 "$H/explain_ties.py" "$OUT/sample.llm" "$OUT/sample.oracle" "$OUT/sample.llama" || status=1
  python3 "$H/impl_llama.py" "$OUT/sample.llm" --planted-defect > "$OUT/sample.llama-planted"
  if python3 "$H/explain_ties.py" "$OUT/sample.llm" "$OUT/sample.oracle" "$OUT/sample.llama-planted" > /dev/null; then
    echo "FAIL: the planted sampler defect was not detected"; status=1
  else
    echo "ok: planted sampler defect detected"
  fi
else
  echo "== llama.cpp samplers: SKIPPED (set LLAMA_LIB to libllama.so)"
fi

exit $status
