#!/usr/bin/env python3
"""Account for every sampling divergence by the positional tie convention.

The oracle evaluates each sampling chain twice: with filters that keep ties
(rows `p`) and with filters that break ties by position (rows `q`,
`LLM.Dist.topKPos`, `LLM.Dist.topPPos`). Exactly one of the two can keep exactly `k`
tokens (`LLM.no_equivariant_exact_topK`, `LLM.card_topKPosSet`).

An implementation that keeps exactly `k` tokens breaks ties in its own order, which
need not be the oracle's order by position. Every filter treats tied tokens alike
except for that choice, and tied tokens stay tied through temperature and
renormalization, so such an implementation must agree with the `q` rows up to a
permutation of tied tokens. That is, the multiset of (logit, probability) pairs over
the kept tokens must be the same. This script checks exactly that.

Every case lands in one bin: agrees with `p`; agrees with `q` up to tied tokens
(explained by the positional convention); or unexplained. The exit status is 1 if
any case is unexplained.

usage: explain_ties.py CASES ORACLE_OUT IMPL_OUT [TOL]
"""
import sys
from fractions import Fraction

from cases import parse_rows, read_cases


def close(a, b, tol):
    return (a == 0) == (b == 0) and abs(a - b) <= tol


def kept_multiset(z, probs):
    return sorted((z[i], probs[i]) for i in range(len(z)) if probs[i] > 0)


def main() -> None:
    cases_path, oracle_path, impl_path = sys.argv[1:4]
    tol = Fraction(sys.argv[4]) if len(sys.argv) > 4 else Fraction(1, 10**6)
    logits = {}
    for kind, name, lines in read_cases(cases_path):
        if kind == "sample" and lines and lines[0].startswith("logits2 "):
            logits[name] = [int(z) for z in lines[0].split(" ")[1:]]
    oracle, _ = parse_rows(open(oracle_path, encoding="utf-8").read())
    impl, _ = parse_rows(open(impl_path, encoding="utf-8").read())
    agree_p = explained = unexplained = 0
    for name, rows in sorted(oracle.items()):
        if name not in impl or name not in logits:
            continue
        z = logits[name]
        n = len(z)
        po = [Fraction(rows[f"p|{i}"]) for i in range(n)]
        qo = [Fraction(rows[f"q|{i}"]) for i in range(n)]
        li = [Fraction(impl[name][f"p|{i}"]) for i in range(n)]
        if all(close(a, b, tol) for a, b in zip(po, li)):
            agree_p += 1
            continue
        mq, ml = kept_multiset(z, qo), kept_multiset(z, li)
        if len(mq) == len(ml) and all(a[0] == b[0] and close(a[1], b[1], tol) for a, b in zip(mq, ml)):
            explained += 1
        else:
            unexplained += 1
            print(f"UNEXPLAINED {name}: logits {z}\n   q {mq}\n   impl {ml}")
    print(f"agree with tie-keeping semantics {agree_p}; "
          f"agree with positional semantics up to tied tokens {explained}; unexplained {unexplained}")
    sys.exit(1 if unexplained else 0)


if __name__ == "__main__":
    main()
