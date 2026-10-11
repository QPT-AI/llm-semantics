"""Shared reader for oracle case files and for the row protocol."""
from fractions import Fraction


def read_cases(path):
    """Yield (kind, name, [lines]) for every case in a case file."""
    kind = name = None
    lines = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n").rstrip("\r")
            if line.startswith("#") or not line.split():
                continue
            if line.startswith("@"):
                if kind is not None:
                    yield kind, name, lines
                head, _, rest = line.partition(" ")
                kind, name, lines = head[1:], rest.strip(), []
            else:
                lines.append(line)
    if kind is not None:
        yield kind, name, lines


def parse_rows(text):
    """Parse protocol output into {case name: {row key: value}} and {case: error}."""
    cases, errors = {}, {}
    current = None
    for line in text.splitlines():
        if line.startswith("== "):
            continue
        if line.startswith("-- "):
            current = line[3:]
            cases[current] = {}
        elif line.startswith("!! ERROR "):
            body = line[len("!! ERROR "):]
            name, _, reason = body.partition(": ")
            errors[name] = reason
            current = None
        elif current is not None and line:
            parts = line.split("|")
            key = "|".join(parts[:-1])
            cases[current][key] = parts[-1]
    return cases, errors


def to_fraction(s):
    return Fraction(s)
