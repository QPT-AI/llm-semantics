import LLM.Attention
import Mathlib.Analysis.Complex.Exponential

/-!
# The real exponential

`Real.exp` is an exponential character. Every theorem of `LLM.Softmax` and
`LLM.Attention` therefore holds, by the same proof, for the softmax of the
literature. This module records the instance and restates the central results in
textbook form.
-/

namespace LLM

open Finset

/-- `Real.exp` as an exponential character. -/
noncomputable def ExpChar.real : ExpChar ℝ ℝ where
  toFun := Real.exp
  map_add' := Real.exp_add
  pos' := Real.exp_pos
  strictMono' := Real.exp_strictMono

variable {ι : Type*} [Fintype ι] [Nonempty ι]

/-- The textbook softmax: `exp (z i) / ∑ j, exp (z j)`. -/
theorem real_softmax_prob (z : ι → ℝ) (i : ι) :
    (softmax ExpChar.real z).prob i = Real.exp (z i) / ∑ j, Real.exp (z j) := rfl

/-- Subtracting the maximum logit (or any constant) does not change the softmax. -/
theorem real_softmax_sub (z : ι → ℝ) (c : ℝ) :
    softmax ExpChar.real (fun i => z i - c) = softmax ExpChar.real z := by
  simpa [sub_eq_add_neg] using softmax_add_const ExpChar.real z (-c)

/-- At every positive temperature, top-k keeps the same tokens as top-k on logits. -/
theorem real_topK_temperature (z : ι → ℝ) {T : ℝ} (hT : 0 < T) (k : ℕ) :
    topKSet k (softmax ExpChar.real (fun i => z i / T)).prob = topKSet k z :=
  topK_temperature ExpChar.real z hT k

variable {M : Type*} [AddCommGroup M] [Module ℝ M]

/-- **FlashAttention computes softmax attention.** For any list of (score, value)
pairs and any split into blocks, the online-softmax reduction returns
`(∑ exp sⱼ)⁻¹ • ∑ exp sⱼ • vⱼ`. -/
theorem real_flash_attention (bs : List (List (ℝ × M))) :
    Online.finalize ExpChar.real ((bs.map (Online.ofList ExpChar.real)).prod) =
      ((bs.flatten.map (fun p => Real.exp p.1)).sum)⁻¹ •
        (bs.flatten.map (fun p => Real.exp p.1 • p.2)).sum :=
  Online.finalize_blocks ExpChar.real bs

end LLM
