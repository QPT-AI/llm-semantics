#!/usr/bin/env python3
"""Run Hugging Face `tokenizers` BPE on the @bpe cases of a case file.

Prints `tokens|WORD|t1 t2 ...` rows (default options, `ignore_merges=False`) and
`tokens-lookup|WORD|...` rows (`ignore_merges=True`) in the oracle's protocol. The
tokenizer is the bare BPE model: no normalizer and no pre-tokenizer; the vocabulary
is every character of the case plus every merge result.

usage: impl_hf.py FILE [--planted-defect]
"""
import sys

from tokenizers import Tokenizer, models

from cases import read_cases


def main() -> None:
    path = sys.argv[1]
    planted = "--planted-defect" in sys.argv[2:]
    print(f"== {path}")
    for kind, name, lines in read_cases(path):
        if kind != "bpe":
            continue
        merges, words = [], []
        for line in lines:
            parts = line.split(" ")
            if parts[0] == "merge":
                merges.append((parts[1], parts[2]))
            elif parts[0] == "encode":
                words.append(parts[1])
        vocab = {}
        chars = sorted({ch for w in words for ch in w} | {ch for a, b in merges for ch in a + b})
        for t in chars + [a + b for a, b in merges]:
            vocab.setdefault(t, len(vocab))
        tok = Tokenizer(models.BPE(vocab=vocab, merges=merges))
        tok_lookup = Tokenizer(models.BPE(vocab=vocab, merges=merges, ignore_merges=True))
        print(f"-- {name}")
        for w in words:
            toks = tok.encode(w).tokens
            if planted and len(toks) >= 2:
                # A deliberately broken implementation: glue the last two tokens.
                toks = toks[:-2] + [toks[-2] + toks[-1]]
            print(f"tokens|{w}|{' '.join(toks)}")
            print(f"tokens-lookup|{w}|{' '.join(tok_lookup.encode(w).tokens)}")


if __name__ == "__main__":
    main()
