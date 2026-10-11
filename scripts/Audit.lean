import LLM
import LLM.Oracle
import LLM.Driver

/-!
# Axiom audit

Every declaration in the `LLM` modules may depend only on the three standard axioms
of Lean's mathematics: propositional extensionality, choice, and quotient soundness.
In particular no declaration may depend on `sorryAx` or on `Lean.ofReduceBool`
(which `native_decide` introduces). The audit walks the whole environment, so a new
theorem is covered without being listed here.

Run with `lake env lean scripts/Audit.lean`. The command fails if any declaration
depends on another axiom.
-/

open Lean Elab Command

/-- The axioms the theory is allowed to use. -/
def allowedAxioms : List Name := [``propext, ``Classical.choice, ``Quot.sound]

elab "#audit_llm" : command => do
  let env ← getEnv
  let mut decls := 0
  let mut theorems := 0
  let mut used : NameSet := {}
  let mut bad : Array (Name × Array Name) := #[]
  for (n, ci) in env.constants.toList do
    let some modIdx := env.getModuleIdxFor? n | continue
    let modName := env.header.moduleNames[modIdx.toNat]!
    unless (`LLM).isPrefixOf modName do continue
    if n.isInternal then continue
    decls := decls + 1
    if ci matches .thmInfo _ then theorems := theorems + 1
    let axs ← liftCoreM <| Lean.collectAxioms n
    for a in axs do used := used.insert a
    let extra := axs.filter (fun a => !allowedAxioms.contains a)
    unless extra.isEmpty do bad := bad.push (n, extra)
  logInfo m!"audited {decls} declarations ({theorems} theorems) in LLM.*; axioms used: {used.toList}"
  unless bad.isEmpty do
    for (n, extra) in bad do
      logError m!"{n} depends on {extra}"

#audit_llm
