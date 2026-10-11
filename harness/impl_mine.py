#!/usr/bin/env python3
"""Run your own implementation on every case of a case file.

The module named by the environment variable `MYLLM` (default `myllm_template`) is
imported, and each case kind is mapped to one of its functions, whose signatures are
documented in `harness/myllm_template.py`:

    @sample       sample_filter(logits, ops)      -> p|i|PROB
    @bpe          bpe_encode(merges, word)        -> tokens|W|..., tokens-lookup|W|..., tiktoken|W|...
    @attention    attention(scores, values)       -> weight|i|W, out|j|V
                  attention_blocked(s, v, block)  -> blocked|j|V
    @speculative  speculative(target, draft)      -> rate|R, accept|i|A, residual|i|P, out|i|P
    @chat         chat_validate(events)           -> valid|..., complete|...
    @diffusion    diffusion_marginals(data, obs)  -> marg|i|c|P

`logits2 z...` becomes natural logits `z * ln 2`, `probs`/`weights` become
`log p`, `sharpen N` becomes temperature `1/N`, and attention scores `s` become
`s * ln 2`, since the oracle's softmax is base 2. A BPE result is printed under all
three convention tags, so `compare.py` can say which convention your tokenizer
follows. A missing function or an unsupported directive is printed as `!! ERROR NAME: ...`
and lands in `skip-impl`. An exception in your code prints a `crash|...` row, so the
case diverges; nothing is dropped silently.

With `--planted-defect`, the adapter deliberately breaks the mapping (temperature
`N` instead of `1/N`, natural instead of base-2 attention scores, last two BPE tokens
glued) so that the comparison can be observed to fail.

usage: MYLLM=yourmodule impl_mine.py FILE [--planted-defect]
"""
import importlib
import math
import os
import sys
from fractions import Fraction

from cases import read_cases

LN2 = math.log(2.0)


class NotComparable(Exception):
    pass


def num(x):
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        raise NotComparable(f"non-finite value {x}")
    return format(x, ".17g")


def need(mod, name):
    f = getattr(mod, name, None)
    if f is None:
        raise NotComparable(f"{mod.__name__}.{name} not implemented")
    return f


def to_logits(p):
    return [math.log(x) if x > 0 else -math.inf for x in p]


def run_sample(mod, lines, planted):
    f = need(mod, "sample_filter")
    logits, ops, out = None, [], []
    for line in lines:
        w = line.split()
        if w[0] == "logits2":
            logits = [int(z) * LN2 for z in w[1:]]
        elif w[0] == "probs":
            logits = to_logits([float(Fraction(x)) for x in w[1:]])
        elif w[0] == "weights":
            ws = [Fraction(x) for x in w[1:]]
            logits = to_logits([float(x / sum(ws)) for x in ws])
        elif w[0] == "top-k":
            ops.append(("top_k", int(w[1])))
        elif w[0] == "top-p":
            ops.append(("top_p", float(Fraction(w[1]))))
        elif w[0] == "min-p":
            ops.append(("min_p", float(Fraction(w[1]))))
        elif w[0] == "sharpen":
            n = int(w[1])
            if n == 0:
                raise NotComparable("sharpen 0 (temperature infinity)")
            ops.append(("temperature", float(n) if planted else 1.0 / n))
        elif w[0] == "mask":
            ops.append(("mask", [int(i) for i in w[1:]]))
        elif w[0] == "emit":
            out += [f"p|{i}|{num(p)}" for i, p in enumerate(f(logits, list(ops)))]
        elif w[0] == "cdf":
            pass  # not compared
        else:
            raise NotComparable(f"unsupported directive {w[0]}")
    return out + [f"p|{i}|{num(p)}" for i, p in enumerate(f(logits, list(ops)))]


def run_bpe(mod, lines, planted):
    f = need(mod, "bpe_encode")
    merges, out = [], []
    for line in lines:
        w = line.split()
        if w[0] == "merge":
            merges.append((w[1], w[2]))
        elif w[0] == "encode":
            toks = list(f(list(merges), w[1]))
            if planted and len(toks) >= 2:
                toks = toks[:-2] + [toks[-2] + toks[-1]]
            s = " ".join(toks)
            out += [f"tokens|{w[1]}|{s}", f"tokens-lookup|{w[1]}|{s}", f"tiktoken|{w[1]}|{s}"]
    return out


def run_attention(mod, lines, planted):
    f = need(mod, "attention")
    scores, values, split = [], [], None
    for line in lines:
        w = line.split()
        if w[0] == "key":
            scores.append(int(w[1]) * (1.0 if planted else LN2))
            values.append([float(Fraction(x)) for x in w[2:]])
        elif w[0] == "split":
            split = int(w[1])
    weights, o = f(scores, values)
    out = [f"weight|{i}|{num(x)}" for i, x in enumerate(weights)]
    out += [f"out|{j}|{num(x)}" for j, x in enumerate(o)]
    blocked = getattr(mod, "attention_blocked", None)
    if split and blocked is not None:
        out += [f"blocked|{j}|{num(x)}" for j, x in enumerate(blocked(scores, values, split))]
    return out


def run_speculative(mod, lines, planted):
    f = need(mod, "speculative")
    t = d = None
    for line in lines:
        w = line.split()
        xs = [float(Fraction(x)) for x in w[1:]]
        if w[0] == "target":
            t = xs
        elif w[0] == "draft":
            d = xs
    r = f(t, d)
    out = [f"rate|{num(r['rate'])}"] if "rate" in r else []
    for key in ("accept", "residual", "out"):
        if key in r:
            out += [f"{key}|{i}|{num(x)}" for i, x in enumerate(r[key])]
    return out


def run_chat(mod, lines, planted):
    f = need(mod, "chat_validate")
    events = []
    for line in lines:
        kind, _, rest = line.partition(" ")
        if kind in ("call", "result"):
            i, _, text = rest.partition(" ")
            events.append((kind, int(i), text))
        else:
            events.append((kind, None, rest))
    bad, complete = f(events)
    valid = "valid|true" if bad is None else f"valid|false|{bad}"
    return [valid, f"complete|{'true' if complete else 'false'}"]


def run_diffusion(mod, lines, planted):
    f = need(mod, "diffusion_marginals")
    data, obs = [], None
    for line in lines:
        w = line.split()
        if w[0] == "data":
            data.append((w[1], float(Fraction(w[2]))))
        elif w[0] == "observe":
            obs = w[1]
    alphabet = sorted({c for s, _ in data for c in s})
    marg = f(data, obs)
    out = []
    for i, o in enumerate(obs):
        if o == "_":
            out += [f"marg|{i}|{c}|{num(marg.get(i, {}).get(c, 0.0))}" for c in alphabet]
    return out


RUNNERS = {"sample": run_sample, "bpe": run_bpe, "attention": run_attention,
           "speculative": run_speculative, "chat": run_chat, "diffusion": run_diffusion}


def main() -> None:
    path = sys.argv[1]
    planted = "--planted-defect" in sys.argv[2:]
    sys.path.insert(0, os.getcwd())
    mod = importlib.import_module(os.environ.get("MYLLM", "myllm_template"))
    print(f"== {path}")
    for kind, name, lines in read_cases(path):
        try:
            if kind not in RUNNERS:
                raise NotComparable(f"unknown kind @{kind}")
            rows = RUNNERS[kind](mod, lines, planted)
        except NotComparable as e:
            print(f"!! ERROR {name}: not comparable: {e}")
            continue
        except Exception as e:  # a crash in your code is a divergence, not a skip
            rows = [f"crash|{type(e).__name__}: {e}"]
        print(f"-- {name}")
        for r in rows:
            print(r)


if __name__ == "__main__":
    main()
