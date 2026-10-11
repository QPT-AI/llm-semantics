import LLM.Dist

/-!
# Speculative decoding

A cheap draft model proposes a token `x ~ q`. The target model, with distribution
`p`, accepts it with probability `min(1, p x / q x)`; on rejection a token is drawn
from the residual `max(0, p - q)`, normalized. The procedure is meant to be
*lossless*: the emitted token has law `p`, whatever `q` is.

This module defines the joint law of (draft, emitted token) exactly as the algorithm
produces it and proves:

* `specJoint_fst`: the draft is distributed as `q`;
* `specJoint_snd`: **losslessness**, the emitted token is distributed as `p`;
* `accept_rate`: the probability of acceptance is `1 - tv p q`;
* `coupling_le`: **optimality**. Any procedure that emits exactly `p` from a draft
  `x ~ q` is a coupling of `q` and `p`, and no coupling makes the two coordinates
  agree with probability above `1 - tv p q`. Speculative sampling attains this
  bound, so no verification rule for a single drafted token can accept more often.
-/

open Finset

namespace LLM

universe u v

variable {𝕜 : Type u} [Field 𝕜] [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜]
variable {V : Type v} [Fintype V]

namespace Spec

/-- The acceptance probability of draft token `x`. With Lean's convention `a / 0 = 0`
this is `0` when `q x = 0`; such a token is never drafted. -/
def accept (p q : Dist 𝕜 V) (x : V) : 𝕜 := min 1 (p.prob x / q.prob x)

lemma accept_nonneg (p q : Dist 𝕜 V) (x : V) : 0 ≤ accept p q x :=
  le_min zero_le_one (div_nonneg (p.nonneg x) (q.nonneg x))

lemma accept_le_one (p q : Dist 𝕜 V) (x : V) : accept p q x ≤ 1 := min_le_left _ _

/-- Drafting `x` and then accepting it has probability `min (p x) (q x)`. -/
lemma mul_accept (p q : Dist 𝕜 V) (x : V) :
    q.prob x * accept p q x = min (p.prob x) (q.prob x) := by
  unfold accept
  rcases (q.nonneg x).lt_or_eq with hq | hq
  · rw [mul_min_of_nonneg _ _ hq.le, mul_one, mul_div_cancel₀ _ (ne_of_gt hq), min_comm]
  · rw [← hq, zero_mul]
    exact (min_eq_right (p.nonneg x)).symm

/-- The residual weights `max(0, p - q)`. Their total is `tv p q`. -/
def residualWeight (p q : Dist 𝕜 V) (x : V) : 𝕜 := max 0 (p.prob x - q.prob x)

/-- The residual distribution. When `p = q` the residual is never used; it is set
to `p` so that it is always a distribution. -/
def residual (p q : Dist 𝕜 V) : Dist 𝕜 V :=
  if h : 0 < Dist.tv p q then
    Dist.normalize (residualWeight p q) (fun _ => le_max_left _ _) h
  else p

lemma residual_prob_of_pos (p q : Dist 𝕜 V) (h : 0 < Dist.tv p q) (x : V) :
    (residual p q).prob x = max 0 (p.prob x - q.prob x) / Dist.tv p q := by
  unfold residual; rw [dite_eq_left h]; rfl

lemma residual_of_not_pos (p q : Dist 𝕜 V) (h : ¬ 0 < Dist.tv p q) : residual p q = p := by
  unfold residual; rw [dite_eq_right h]

lemma eq_of_not_tv_pos (p q : Dist 𝕜 V) (h : ¬ 0 < Dist.tv p q) : p = q :=
  (Dist.tv_eq_zero_iff p q).mp (le_antisymm (not_lt.mp h) (Dist.tv_nonneg p q))

/-- A rejected draft is never resampled as itself: if rejection is possible at `x`,
the residual gives `x` no mass. -/
lemma reject_mul_residual_self (p q : Dist 𝕜 V) (x : V) :
    (q.prob x - min (p.prob x) (q.prob x)) * (residual p q).prob x = 0 := by
  rcases le_total (q.prob x) (p.prob x) with h | h
  · rw [min_eq_right h, sub_self, zero_mul]
  · by_cases htv : 0 < Dist.tv p q
    · rw [residual_prob_of_pos p q htv, max_eq_left (by linarith), zero_div, mul_zero]
    · have hpq := eq_of_not_tv_pos p q htv
      subst hpq
      rw [min_self, sub_self, zero_mul]

/-- The total rejection probability is `tv p q`. -/
lemma sum_reject (p q : Dist 𝕜 V) :
    ∑ x, (q.prob x - min (p.prob x) (q.prob x)) = Dist.tv p q := by
  rw [Finset.sum_sub_distrib, q.sum_eq_one, Dist.sum_min_eq]
  ring

variable [DecidableEq V]

/-- The probability of emitting `y` after drafting `x`, times the probability of
drafting `x`. -/
def jointWeight (p q : Dist 𝕜 V) (x y : V) : 𝕜 :=
  q.prob x * (accept p q x * (if y = x then 1 else 0) + (1 - accept p q x) * (residual p q).prob y)

lemma sum_jointWeight (p q : Dist 𝕜 V) (x : V) : ∑ y, jointWeight p q x y = q.prob x := by
  unfold jointWeight
  rw [← Finset.mul_sum, Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum,
    Dist.sum_eq_one, Finset.sum_ite_eq']
  simp

/-- The joint law of (draft token, emitted token), written as the algorithm runs:
draft `x ~ q`; with probability `accept x` emit `x`; otherwise emit a draw from the
residual. -/
def specJoint (p q : Dist 𝕜 V) : Dist 𝕜 (V × V) where
  prob xy := jointWeight p q xy.1 xy.2
  nonneg xy := by
    unfold jointWeight
    apply mul_nonneg (q.nonneg _)
    apply add_nonneg
    · apply mul_nonneg (accept_nonneg p q _); split_ifs <;> norm_num
    · exact mul_nonneg (by linarith [accept_le_one p q xy.1]) ((residual p q).nonneg _)
  sum_eq_one := by
    rw [Fintype.sum_prod_type]
    simp_rw [sum_jointWeight]
    exact q.sum_eq_one

/-- The draft is distributed as `q`. -/
theorem specJoint_fst (p q : Dist 𝕜 V) (x : V) :
    ∑ y, (specJoint p q).prob (x, y) = q.prob x :=
  sum_jointWeight p q x

/-- Unfolded form of the joint law. -/
lemma specJoint_prob (p q : Dist 𝕜 V) (x y : V) :
    (specJoint p q).prob (x, y) =
      (if y = x then min (p.prob x) (q.prob x) else 0) +
        (q.prob x - min (p.prob x) (q.prob x)) * (residual p q).prob y := by
  show jointWeight p q x y = _
  unfold jointWeight
  rw [mul_add, ← mul_assoc, mul_accept, ← mul_assoc, mul_sub, mul_one, mul_accept]
  split_ifs <;> ring

/-- **Losslessness.** The emitted token is distributed as the target `p`. -/
theorem specJoint_snd (p q : Dist 𝕜 V) (y : V) :
    ∑ x, (specJoint p q).prob (x, y) = p.prob y := by
  simp only [specJoint_prob, Finset.sum_add_distrib, ← Finset.sum_mul, sum_reject]
  rw [Finset.sum_ite_eq]
  simp only [mem_univ, ite_true]
  by_cases htv : 0 < Dist.tv p q
  · rw [residual_prob_of_pos p q htv, mul_div_cancel₀ _ (ne_of_gt htv)]
    rcases le_total (p.prob y) (q.prob y) with h | h
    · rw [min_eq_left h, max_eq_left (by linarith), add_zero]
    · rw [min_eq_right h, max_eq_right (by linarith)]; ring
  · have hpq := eq_of_not_tv_pos p q htv
    subst hpq
    rw [Dist.tv_self, min_self, zero_mul, add_zero]

omit [DecidableEq V] in
/-- **Acceptance rate.** A draft is accepted with probability `1 - tv p q`. -/
theorem accept_rate (p q : Dist 𝕜 V) :
    ∑ x, q.prob x * accept p q x = 1 - Dist.tv p q := by
  simp_rw [mul_accept]
  exact Dist.sum_min_eq p q

/-- The emitted token equals the draft with probability exactly `1 - tv p q`. -/
theorem specJoint_diag (p q : Dist 𝕜 V) :
    ∑ x, (specJoint p q).prob (x, x) = 1 - Dist.tv p q := by
  simp only [specJoint_prob, ite_true, reject_mul_residual_self, add_zero]
  exact Dist.sum_min_eq p q

omit [DecidableEq V] in
/-- A coupling of `q` (first coordinate) and `p` (second coordinate). -/
def IsCoupling (π : Dist 𝕜 (V × V)) (q p : Dist 𝕜 V) : Prop :=
  (∀ x, ∑ y, π.prob (x, y) = q.prob x) ∧ (∀ y, ∑ x, π.prob (x, y) = p.prob y)

theorem specJoint_isCoupling (p q : Dist 𝕜 V) : IsCoupling (specJoint p q) q p :=
  ⟨specJoint_fst p q, specJoint_snd p q⟩

omit [DecidableEq V] in
/-- **The coupling inequality.** No coupling of `q` and `p` agrees with probability
above `1 - tv p q`. -/
theorem coupling_le (π : Dist 𝕜 (V × V)) (q p : Dist 𝕜 V) (hπ : IsCoupling π q p) :
    ∑ x, π.prob (x, x) ≤ 1 - Dist.tv p q := by
  rw [← Dist.sum_min_eq p q]
  apply Finset.sum_le_sum
  intro x _
  apply le_min
  · rw [← hπ.2 x]
    exact Finset.single_le_sum (f := fun x' => π.prob (x', x)) (fun _ _ => π.nonneg _)
      (mem_univ x)
  · rw [← hπ.1 x]
    exact Finset.single_le_sum (f := fun y => π.prob (x, y)) (fun _ _ => π.nonneg _)
      (mem_univ x)

/-- **Optimality of speculative sampling.** Among all lossless single-token
verification schemes, speculative sampling maximizes the agreement probability. -/
theorem spec_optimal (p q : Dist 𝕜 V) (π : Dist 𝕜 (V × V)) (hπ : IsCoupling π q p) :
    ∑ x, π.prob (x, x) ≤ ∑ x, (specJoint p q).prob (x, x) := by
  rw [specJoint_diag]; exact coupling_le π q p hπ

end Spec

end LLM
