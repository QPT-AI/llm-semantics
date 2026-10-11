#!/usr/bin/env python3
"""Run `tiktoken` on the @bpe cases of a case file.

Prints `tiktoken|WORD|t1 t2 ...` rows in the oracle's protocol. The encoding is
built in memory: single characters take the lowest ranks, and each merge result is
ranked by the position of the first merge that produces it. The split pattern is
`\\S+`, so each word is a single piece.

usage: impl_tiktoken.py FILE
"""
import sys

import tiktoken

from cases import read_cases


def main() -> None:
    path = sys.argv[1]
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
        chars = sorted({ch for w in words for ch in w} | {ch for a, b in merges for ch in a + b})
        ranks = {}
        for t in chars + [a + b for a, b in merges]:
            ranks.setdefault(t.encode(), len(ranks))
        enc = tiktoken.Encoding(name=f"case-{abs(hash(name))}", pat_str=r"\S+",
                                mergeable_ranks=ranks, special_tokens={})
        inv = {v: k.decode() for k, v in ranks.items()}
        print(f"-- {name}")
        for w in words:
            print(f"tiktoken|{w}|{' '.join(inv[i] for i in enc.encode(w))}")


if __name__ == "__main__":
    main()
