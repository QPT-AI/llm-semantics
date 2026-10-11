#!/usr/bin/env python3
"""Seeded generator of oracle case files for property testing.

The generator writes inputs only. It never computes or records an expected result:
expected results come from the oracle, and the generator's job is to produce inputs
that exercise the semantics, including the inputs where implementations are known
to have conventions (ties, boundaries, overlapping merges).

usage: gen.py bpe|sample|speculative|chat|attention|diffusion SEED COUNT
"""
import random
import sys
from fractions import Fraction


def gen_bpe(rng: random.Random, idx: int) -> str:
    alphabet = "abcd"[: rng.randint(2, 4)]
    tokens = list(alphabet)
    merges = []
    for _ in range(rng.randint(1, 8)):
        for _attempt in range(20):
            a, b = rng.choice(tokens), rng.choice(tokens)
            if a + b not in tokens and len(a + b) <= 6:
                merges.append((a, b))
                tokens.append(a + b)
                break
    lines = [f"@bpe gen-bpe-{idx}"]
    lines += [f"merge {a} {b}" for a, b in merges]
    words = set()
    for _ in range(6):
        r = rng.random()
        if r < 0.2:
            words.add(rng.choice(tokens))
        elif r < 0.45:
            words.add(rng.choice(tokens) + rng.choice(tokens))
        else:
            words.add("".join(rng.choice(alphabet) for _ in range(rng.randint(1, 9))))
    lines += [f"encode {w}" for w in sorted(words)]
    return "\n".join(lines)


def rat(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def gen_sample(rng: random.Random, idx: int) -> str:
    n = rng.randint(2, 8)
    distinct = rng.random() < 0.5
    if distinct:
        zs = rng.sample(range(-6, 7), n)
    else:
        zs = [rng.randint(-2, 2) for _ in range(n)]
    ops = []
    for _ in range(rng.randint(1, 3)):
        kind = rng.choice(["top-k", "top-p", "min-p", "sharpen"])
        if kind == "top-k":
            ops.append(f"top-k {rng.randint(1, n)}")
        elif kind == "top-p":
            ops.append(f"top-p {rat(Fraction(rng.randint(1, 19), 20))}")
        elif kind == "min-p":
            ops.append(f"min-p {rat(Fraction(rng.randint(1, 9), 10))}")
        else:
            ops.append(f"sharpen {rng.randint(1, 3)}")
    tag = "distinct" if distinct else "ties"
    return "\n".join([f"@sample gen-sample-{idx}-{tag}", "logits2 " + " ".join(map(str, zs))] + ops)


def gen_dist(rng: random.Random, n: int) -> list:
    ws = [rng.randint(0, 6) for _ in range(n)]
    if sum(ws) == 0:
        ws[0] = 1
    total = sum(ws)
    return [Fraction(w, total) for w in ws]


def gen_speculative(rng: random.Random, idx: int) -> str:
    n = rng.randint(2, 6)
    p, q = gen_dist(rng, n), gen_dist(rng, n)
    return "\n".join([f"@speculative gen-spec-{idx}",
                      "target " + " ".join(map(rat, p)),
                      "draft " + " ".join(map(rat, q))])


def gen_chat(rng: random.Random, idx: int) -> str:
    lines = [f"@chat gen-chat-{idx}"]
    next_id = 1
    pending = []
    for _ in range(rng.randint(1, 10)):
        r = rng.random()
        if r < 0.15:
            lines.append("system be brief")
        elif r < 0.35:
            lines.append("user question")
        elif r < 0.5:
            lines.append("assistant answer")
        elif r < 0.75:
            i = next_id if rng.random() < 0.9 else rng.randint(1, max(1, next_id))
            next_id += 1
            pending.append(i)
            lines.append(f"call {i} tool()")
        else:
            if pending and rng.random() < 0.85:
                i = pending.pop(rng.randrange(len(pending)))
            else:
                i = rng.randint(1, next_id + 1)
            lines.append(f"result {i} ok")
    return "\n".join(lines)


def gen_attention(rng: random.Random, idx: int) -> str:
    n = rng.randint(1, 8)
    d = rng.randint(1, 4)
    ties = rng.random() < 0.3
    lines = [f"@attention gen-attn-{idx}"]
    for _ in range(n):
        s = rng.randint(-1, 1) if ties else rng.randint(-6, 6)
        vs = [rat(Fraction(rng.randint(-8, 8), rng.choice([1, 2, 3, 4]))) for _ in range(d)]
        lines.append(f"key {s} " + " ".join(vs))
    if rng.random() < 0.8:
        lines.append(f"split {rng.randint(1, n)}")
    return "\n".join(lines)


def gen_diffusion(rng: random.Random, idx: int) -> str:
    alphabet = "abc"[: rng.randint(2, 3)]
    length = rng.randint(1, 4)
    seqs = {"".join(rng.choice(alphabet) for _ in range(length)) for _ in range(rng.randint(1, 6))}
    seqs = sorted(seqs)
    p = gen_dist(rng, len(seqs))
    lines = [f"@diffusion gen-diff-{idx}"]
    lines += [f"data {s} {rat(x)}" for s, x in zip(seqs, p) if x > 0]
    # observe a sequence with positive mass, so the evidence is positive
    src = rng.choice([s for s, x in zip(seqs, p) if x > 0])
    r = rng.random()
    obs = "".join("_" if (r < 0.2 or rng.random() < 0.5) else c for c in src)
    lines.append(f"observe {obs}")
    return "\n".join(lines)


GENS = {"bpe": gen_bpe, "sample": gen_sample, "speculative": gen_speculative, "chat": gen_chat,
        "attention": gen_attention, "diffusion": gen_diffusion}


def main() -> None:
    kind, seed, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rng = random.Random(f"{kind}:{seed}")
    print(f"# generated: kind={kind} seed={seed} count={count}")
    for i in range(count):
        print(GENS[kind](rng, i))
        print()


if __name__ == "__main__":
    main()
