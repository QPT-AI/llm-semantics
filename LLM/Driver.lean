import LLM.Oracle

/-!
# Driver

Run the oracle on case files:

```
lake env lean --run LLM/Driver.lean FILE...
```

Each file's output begins with `== PATH`. The driver runs under the Lean
interpreter, so what executes is a direct reading of the checked definitions; the
build produces no native executable.
-/

open LLM.Oracle

def main (args : List String) : IO UInt32 := do
  if args.isEmpty then
    IO.eprintln "usage: lake env lean --run LLM/Driver.lean FILE..."
    return 2
  let mut code : UInt32 := 0
  for path in args do
    IO.println s!"== {path}"
    match ← (IO.FS.readFile path).toBaseIO with
    | .ok src =>
        for row in runFile src do
          IO.println row
    | .error e =>
        IO.println s!"!! ERROR cannot read {path}: {e}"
        code := 1
  return code
