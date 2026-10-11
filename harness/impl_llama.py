#!/usr/bin/env python3
"""Run llama.cpp's samplers on the @sample cases of a case file.

The samplers are called through `libllama` with ctypes, on a candidate array built
from the case's logits; no model is loaded. A `logits2 z...` case becomes natural
logits `z * ln 2`, so that llama.cpp's softmax computes `2^z / sum 2^z`. The
directives map as follows: `top-k K` to `llama_sampler_init_top_k(K)`, `top-p P` to
`llama_sampler_init_top_p(P, 1)`, `min-p M` to `llama_sampler_init_min_p(M, 1)`, and
`sharpen N` to `llama_sampler_init_temp(1/N)`. After the chain, the probability of a
surviving token is the softmax of the surviving logits, computed in double precision.

Prints `p|i|PROB` rows (floats) in the oracle's protocol. Cases that do not start
with `logits2`, or use other directives, are reported as `!! ERROR` (not comparable).

With `--planted-defect`, `sharpen N` is mapped to temperature `N` instead of `1/N`:
a deliberately broken adapter that the comparison must catch.

usage: impl_llama.py FILE [--planted-defect]   (LLAMA_LIB names libllama.so)
"""
import ctypes
import math
import os
import sys

from cases import read_cases


class TokenData(ctypes.Structure):
    _fields_ = [("id", ctypes.c_int32), ("logit", ctypes.c_float), ("p", ctypes.c_float)]


class TokenDataArray(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(TokenData)), ("size", ctypes.c_size_t),
                ("selected", ctypes.c_int64), ("sorted", ctypes.c_bool)]


def load():
    lib = ctypes.CDLL(os.environ.get("LLAMA_LIB", "libllama.so"))
    lib.llama_sampler_init_top_k.argtypes = [ctypes.c_int32]
    lib.llama_sampler_init_top_p.argtypes = [ctypes.c_float, ctypes.c_size_t]
    lib.llama_sampler_init_min_p.argtypes = [ctypes.c_float, ctypes.c_size_t]
    lib.llama_sampler_init_temp.argtypes = [ctypes.c_float]
    for f in ("top_k", "top_p", "min_p", "temp"):
        getattr(lib, f"llama_sampler_init_{f}").restype = ctypes.c_void_p
    lib.llama_sampler_apply.argtypes = [ctypes.c_void_p, ctypes.POINTER(TokenDataArray)]
    lib.llama_sampler_free.argtypes = [ctypes.c_void_p]
    return lib


def run_case(lib, lines, planted=False):
    first = lines[0].split(" ")
    if first[0] != "logits2":
        raise ValueError("not comparable: llama.cpp starts from logits")
    zs = [int(z) for z in first[1:]]
    n = len(zs)
    buf = (TokenData * n)(*[TokenData(i, z * math.log(2.0), 0.0) for i, z in enumerate(zs)])
    arr = TokenDataArray(ctypes.cast(buf, ctypes.POINTER(TokenData)), n, -1, False)
    for line in lines[1:]:
        op, arg = line.split(" ")
        if op == "top-k":
            s = lib.llama_sampler_init_top_k(int(arg))
        elif op == "top-p":
            s = lib.llama_sampler_init_top_p(float_of(arg), 1)
        elif op == "min-p":
            s = lib.llama_sampler_init_min_p(float_of(arg), 1)
        elif op == "sharpen":
            s = lib.llama_sampler_init_temp(float(int(arg)) if planted else 1.0 / int(arg))
        else:
            raise ValueError(f"not comparable: directive {op}")
        lib.llama_sampler_apply(s, ctypes.byref(arr))
        lib.llama_sampler_free(s)
    kept = {arr.data[i].id: float(arr.data[i].logit) for i in range(arr.size)}
    m = max(kept.values())
    z = sum(math.exp(l - m) for l in kept.values())
    return [kept_prob(kept, i, m, z) for i in range(n)]


def kept_prob(kept, i, m, z):
    return math.exp(kept[i] - m) / z if i in kept else 0.0


def float_of(s):
    if "/" in s:
        a, b = s.split("/")
        return int(a) / int(b)
    return float(s)


def main() -> None:
    path = sys.argv[1]
    planted = "--planted-defect" in sys.argv[2:]
    lib = load()
    print(f"== {path}")
    for kind, name, lines in read_cases(path):
        if kind != "sample":
            continue
        try:
            probs = run_case(lib, lines, planted)
        except ValueError as e:
            print(f"!! ERROR {name}: {e}")
            continue
        print(f"-- {name}")
        for i, p in enumerate(probs):
            print(f"p|{i}|{p!r}")


if __name__ == "__main__":
    main()
