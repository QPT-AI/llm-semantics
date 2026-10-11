#!/usr/bin/env python3
"""Compare an implementation's rows with the oracle's rows.

usage: compare.py ORACLE_OUT IMPL_OUT PREFIX [--float TOL] [--label LABEL]

Only rows whose tag is PREFIX (for example `tokens`, `tiktoken`, `p`) are compared.
In exact mode the values must be identical strings. In float mode the oracle's value
is an exact rational and the implementation's a float: a zero must be matched by a
zero, and a nonzero value must agree within TOL.

Every case lands in exactly one bin, and nothing is dropped silently:

* `agree`: every compared row matches;
* `diverge`: some row differs (the first differing row is shown);
* `skip-oracle`: the oracle rejected the case;
* `skip-impl`: the implementation reported the case as not comparable;
* `missing`: the case is absent from one side.

With `--ties-bin`, a divergence in a case whose name ends in `-ties` is reported in
a separate bin, `diverge-ties`, because the semantics predicts it: no rule that
treats tied tokens symmetrically can keep exactly `k` of them
(`LLM.no_equivariant_exact_topK`), and an implementation that keeps exactly `k`
must break ties by position. The exit status is 1 if any case is in `diverge` or
`missing`, and 0 otherwise.
"""
import sys
from collections import Counter
from fractions import Fraction

from cases import parse_rows


def same(o, i, tol):
    if tol is None:
        return o == i
    fo, fi = Fraction(o), Fraction(i)
    if (fo == 0) != (fi == 0):
        return False
    return abs(fo - fi) <= Fraction(tol)


def main() -> None:
    oracle_path, impl_path, prefix = sys.argv[1:4]
    args = sys.argv[4:]
    tol = args[args.index("--float") + 1] if "--float" in args else None
    label = args[args.index("--label") + 1] if "--label" in args else impl_path
    ties_bin = "--ties-bin" in args
    with open(oracle_path, encoding="utf-8") as f:
        oracle, oerr = parse_rows(f.read())
    with open(impl_path, encoding="utf-8") as f:
        impl, ierr = parse_rows(f.read())
    bins = Counter()
    details = []
    for name in sorted(set(oracle) | set(oerr) | set(impl) | set(ierr)):
        if name in oerr:
            bins["skip-oracle"] += 1
            continue
        if name in ierr:
            bins["skip-impl"] += 1
            continue
        if name not in oracle or name not in impl:
            bins["missing"] += 1
            details.append(f"missing  {name}")
            continue
        okeys = {k for k in oracle[name] if k.split("|")[0] == prefix}
        ikeys = {k for k in impl[name] if k.split("|")[0] == prefix}
        if not okeys and not ikeys:
            continue
        bad = None
        if okeys != ikeys:
            bad = f"row sets differ: oracle-only {sorted(okeys - ikeys)[:3]}, impl-only {sorted(ikeys - okeys)[:3]}"
        else:
            for k in sorted(okeys):
                if not same(oracle[name][k], impl[name][k], tol):
                    bad = f"{k}: oracle {oracle[name][k]}  impl {impl[name][k]}"
                    break
        if bad is None:
            bins["agree"] += 1
        else:
            b = "diverge-ties" if ties_bin and name.endswith("-ties") else "diverge"
            bins[b] += 1
            details.append(f"{b:<19}{name}: {bad}")
    total = sum(bins.values())
    print(f"[{label}] {total} cases: " + ", ".join(f"{k} {v}" for k, v in sorted(bins.items())))
    limit = 40
    for d in details[:limit]:
        print("   " + d)
    if len(details) > limit:
        print(f"   ... {len(details) - limit} more")
    sys.stdout.flush()
    sys.exit(1 if bins["diverge"] or bins["missing"] else 0)


if __name__ == "__main__":
    main()
