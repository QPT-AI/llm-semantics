import LLM.Sampling

/-!
# Counterexamples for the sampler algebra

`LLM.Sampling` proves which reorderings of a sampler chain are harmless: top-k
commutes with min-p and with temperature, and both are idempotent. This module
proves, with explicit distributions, that the remaining laws one might expect are
false. Every statement is checked by evaluation in the kernel (`decide`), over the
rationals, with no `native_decide`.

The witness throughout is the distribution `(1/2, 3/10, 1/5)` on three tokens.
-/

namespace LLM.Counterexamples

/-- The witness distribution `(1/2, 3/10, 1/5)`. -/
def p₀ : Dist ℚ (Fin 3) where
  prob := ![1/2, 3/10, 1/5]
  nonneg := by decide +kernel
  sum_eq_one := by decide +kernel

/-- **Top-p is not idempotent.** With `P = 3/5`, one pass keeps tokens `0` and `1`
(renormalized to `5/8, 3/8`); a second pass keeps only token `0`. -/
theorem topP_not_idempotent :
    (p₀.topP (3/5) (by norm_num)).prob 1 = 3/8 ∧
      ((p₀.topP (3/5) (by norm_num)).topP (3/5) (by norm_num)).prob 1 = 0 := by
  decide +kernel

/-- **Top-p does not commute with top-k.** With `k = 2` and `P = 3/5`, filtering by
top-p first keeps two tokens; filtering by top-k first keeps one. -/
theorem topP_topK_not_comm :
    ((p₀.topP (3/5) (by norm_num)).topK 2 (by norm_num)).prob 1 = 3/8 ∧
      ((p₀.topK 2 (by norm_num)).topP (3/5) (by norm_num)).prob 1 = 0 := by
  decide +kernel

/-- **Top-p does not commute with temperature.** At temperature `1/2` (squaring),
filtering first keeps token `1` with probability `9/34`; squaring first removes it. -/
theorem topP_sharpen_not_comm :
    ((p₀.topP (3/5) (by norm_num)).sharpen 2).prob 1 = 9/34 ∧
      ((p₀.sharpen 2).topP (3/5) (by norm_num)).prob 1 = 0 := by
  decide +kernel

/-- **Min-p does not commute with temperature.** With `m = 1/2`, filtering first keeps
token `1` (since `3/10 ≥ 1/2 · 1/2`); squaring first removes it (since
`9/100 < 1/2 · 1/4`). -/
theorem minP_sharpen_not_comm :
    ((p₀.minP (1/2) (by norm_num)).sharpen 2).prob 1 = 9/34 ∧
      ((p₀.sharpen 2).minP (1/2) (by norm_num)).prob 1 = 0 := by
  decide +kernel

end LLM.Counterexamples
