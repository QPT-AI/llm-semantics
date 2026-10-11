import LLM.Dist
import Mathlib.Data.Fintype.Option
import Mathlib.Data.Fin.Tuple.Basic
import Mathlib.Data.List.OfFn
import Mathlib.Algebra.BigOperators.Fin

/-!
# Autoregressive decoding

An autoregressive language model assigns to every prefix a distribution over the
next token or the end of the sequence (`none`). Generation is the iterated
composition of these choices. This module proves:

* `mass_conservation`: at every depth `n`, the probability of having stopped
  before `n` plus the probability of still generating at `n` is exactly one;
* `alive_le_of_stop`: if every prefix stops with probability at least `ε > 0`, the
  probability of still generating after `n` tokens is at most `(1 - ε)ⁿ`, so the
  model defines a distribution over finite strings (it is *tight*);
* **negative result** `stopped_lt_half`: a model that stops after `n` tokens with
  probability `1/(n+2)²` never places more than half its mass on finite strings.
  An autoregressive model need not define a distribution over strings at all;
* **negative result** `greedy_not_map`: following the most probable token at
  each step does not find the most probable sequence.
-/

open Finset

namespace LLM

universe u v

variable {𝕜 : Type u} [Field 𝕜] [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜]

/-- An autoregressive language model over the vocabulary `V`: for each prefix, a
distribution over the next token, where `none` ends the sequence. -/
abbrev LM (𝕜 : Type u) [Field 𝕜] [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜] (V : Type v)
    [Fintype V] := List V → Dist 𝕜 (Option V)

namespace LM

variable {V : Type v} [Fintype V] (m : LM 𝕜 V)

/-- The probability that, after the prefix `pre`, the model emits `xs` and has not
yet stopped. -/
def alive (pre : List V) : List V → 𝕜
  | [] => 1
  | x :: xs => (m pre).prob (some x) * alive (pre ++ [x]) xs

/-- The probability that, after `pre`, the model emits exactly `xs` and stops. -/
def stopAfter (pre xs : List V) : 𝕜 := m.alive pre xs * (m (pre ++ xs)).prob none

/-- The probability of still generating after `n` more tokens. -/
def aliveMass (pre : List V) (n : ℕ) : 𝕜 := ∑ f : Fin n → V, m.alive pre (List.ofFn f)

/-- The probability of having stopped within `n` more tokens. -/
def stoppedMass (pre : List V) (n : ℕ) : 𝕜 :=
  ∑ k ∈ range n, ∑ f : Fin k → V, m.stopAfter pre (List.ofFn f)

lemma alive_nonneg (pre xs : List V) : 0 ≤ m.alive pre xs := by
  induction xs generalizing pre with
  | nil => exact zero_le_one
  | cons x xs ih => exact mul_nonneg ((m pre).nonneg _) (ih _)

omit [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜] in
lemma sum_fun_succ {n : ℕ} (g : (Fin (n + 1) → V) → 𝕜) :
    ∑ f : Fin (n + 1) → V, g f = ∑ x : V, ∑ f : Fin n → V, g (Fin.cons x f) := by
  rw [← Fintype.sum_prod_type']
  exact (Fintype.sum_equiv (Fin.consEquiv (fun _ => V)) _ _ (fun _ => rfl)).symm

lemma stopAfter_cons (pre : List V) (x : V) (xs : List V) :
    m.stopAfter pre (x :: xs) = (m pre).prob (some x) * m.stopAfter (pre ++ [x]) xs := by
  simp only [stopAfter, alive, List.append_assoc, List.singleton_append, mul_assoc]

lemma aliveMass_succ (pre : List V) (n : ℕ) :
    m.aliveMass pre (n + 1) = ∑ x, (m pre).prob (some x) * m.aliveMass (pre ++ [x]) n := by
  unfold aliveMass
  rw [sum_fun_succ]
  simp only [List.ofFn_succ, Fin.cons_zero, Fin.cons_succ, alive, Finset.mul_sum]

lemma stoppedMass_succ (pre : List V) (n : ℕ) :
    m.stoppedMass pre (n + 1) =
      (m pre).prob none + ∑ x, (m pre).prob (some x) * m.stoppedMass (pre ++ [x]) n := by
  unfold stoppedMass
  rw [Finset.sum_range_succ', add_comm]
  congr 1
  · simp [stopAfter, alive]
  · simp_rw [Finset.mul_sum]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl (fun k _ => ?_)
    rw [sum_fun_succ]
    simp only [List.ofFn_succ, Fin.cons_zero, Fin.cons_succ, stopAfter_cons]

/-- **Mass conservation.** -/
theorem mass_conservation (pre : List V) (n : ℕ) :
    m.stoppedMass pre n + m.aliveMass pre n = 1 := by
  induction n generalizing pre with
  | zero => simp [stoppedMass, aliveMass, alive]
  | succ n ih =>
    rw [stoppedMass_succ, aliveMass_succ, add_assoc, ← Finset.sum_add_distrib]
    simp_rw [← mul_add, ih, mul_one]
    rw [← Fintype.sum_option (fun o => (m pre).prob o)]
    exact (m pre).sum_eq_one

lemma sum_some (pre : List V) : ∑ x, (m pre).prob (some x) = 1 - (m pre).prob none := by
  have := (m pre).sum_eq_one
  rw [Fintype.sum_option] at this
  linarith

/-- **Tightness under a stopping floor.** -/
theorem alive_le_of_stop {ε : 𝕜} (hε : ∀ pre, ε ≤ (m pre).prob none) (pre : List V) (n : ℕ) :
    m.aliveMass pre n ≤ (1 - ε) ^ n := by
  have hε1 : ε ≤ 1 := le_trans (hε []) ((m []).prob_le_one none)
  induction n generalizing pre with
  | zero => simp [aliveMass, alive]
  | succ n ih =>
    rw [aliveMass_succ, pow_succ]
    calc ∑ x, (m pre).prob (some x) * m.aliveMass (pre ++ [x]) n
        ≤ ∑ x, (m pre).prob (some x) * (1 - ε) ^ n :=
          Finset.sum_le_sum (fun x _ => mul_le_mul_of_nonneg_left (ih _) ((m pre).nonneg _))
      _ = (1 - (m pre).prob none) * (1 - ε) ^ n := by rw [← Finset.sum_mul, sum_some]
      _ ≤ (1 - ε) * (1 - ε) ^ n := by
          apply mul_le_mul_of_nonneg_right (by linarith [hε pre]) (pow_nonneg (by linarith) n)
      _ = (1 - ε) ^ n * (1 - ε) := mul_comm _ _

/-- Under a stopping floor, the stopped mass tends to one. -/
theorem stopped_ge_of_stop {ε : 𝕜} (hε : ∀ pre, ε ≤ (m pre).prob none) (pre : List V) (n : ℕ) :
    1 - (1 - ε) ^ n ≤ m.stoppedMass pre n := by
  have := m.mass_conservation pre n
  have := m.alive_le_of_stop hε pre n
  linarith

end LM

/-! ### A model that is not tight -/

section NotTight

/-- The stopping probability `1/(L+2)²` after `L` tokens. -/
def leakStop (L : ℕ) : 𝕜 := 1 / ((L : 𝕜) + 2) ^ 2

lemma leakStop_nonneg (L : ℕ) : (0 : 𝕜) ≤ leakStop L := by unfold leakStop; positivity

lemma leakStop_le_one (L : ℕ) : leakStop (𝕜 := 𝕜) L ≤ 1 := by
  unfold leakStop
  rw [div_le_one (by positivity)]
  have : (0 : 𝕜) ≤ L := Nat.cast_nonneg L
  nlinarith

/-- A one-token model that stops after `L` tokens with probability `1/(L+2)²`. -/
def leaky : LM 𝕜 Unit := fun pre =>
  { prob := fun o => match o with
      | none => leakStop pre.length
      | some _ => 1 - leakStop pre.length
    nonneg := fun o => by
      cases o
      · exact leakStop_nonneg _
      · simp only; linarith [leakStop_le_one (𝕜 := 𝕜) pre.length]
    sum_eq_one := by simp [Fintype.sum_option] }

lemma leaky_alive (pre : List Unit) (n : ℕ) :
    (leaky (𝕜 := 𝕜)).alive pre (List.replicate n ()) =
      ((pre.length : 𝕜) + 1) * ((pre.length : 𝕜) + n + 2) /
        (((pre.length : 𝕜) + 2) * ((pre.length : 𝕜) + n + 1)) := by
  induction n generalizing pre with
  | zero =>
    simp only [List.replicate_zero, LM.alive, Nat.cast_zero, add_zero]
    have : ((pre.length : 𝕜) + 2) * ((pre.length : 𝕜) + 1) ≠ 0 := by positivity
    field_simp
  | succ n ih =>
    rw [List.replicate_succ, LM.alive, ih]
    simp only [leaky, leakStop, List.length_append, List.length_singleton, Nat.cast_succ]
    have h1 : (pre.length : 𝕜) + 2 ≠ 0 := by positivity
    have h2 : (pre.length : 𝕜) + 1 + 2 ≠ 0 := by positivity
    have h3 : (pre.length : 𝕜) + 1 + n + 1 ≠ 0 := by positivity
    have h4 : (pre.length : 𝕜) + (n + 1) + 1 ≠ 0 := by positivity
    field_simp
    ring

lemma leaky_aliveMass (n : ℕ) :
    (leaky (𝕜 := 𝕜)).aliveMass [] n = ((n : 𝕜) + 2) / (2 * ((n : 𝕜) + 1)) := by
  unfold LM.aliveMass
  rw [Fintype.sum_unique]
  have : List.ofFn (default : Fin n → Unit) = List.replicate n () := by
    rw [show (default : Fin n → Unit) = fun _ => () from rfl, List.ofFn_const]
  rw [this, leaky_alive]
  simp only [List.length_nil, Nat.cast_zero, zero_add]
  have : (n : 𝕜) + 1 ≠ 0 := by positivity
  field_simp

/-- **Non-tightness.** The model `leaky` never accumulates as much as half of its
probability on finite strings: for every `n`, the probability of having stopped
within `n` tokens is below `1/2`. -/
theorem stopped_lt_half (n : ℕ) : (leaky (𝕜 := 𝕜)).stoppedMass [] n < 1 / 2 := by
  have h := (leaky (𝕜 := 𝕜)).mass_conservation [] n
  rw [leaky_aliveMass] at h
  have hpos : (0 : 𝕜) < (n : 𝕜) + 1 := by positivity
  have : (1 : 𝕜) / 2 < ((n : 𝕜) + 2) / (2 * ((n : 𝕜) + 1)) := by
    rw [div_lt_div_iff₀ (by norm_num) (by positivity)]
    nlinarith
  linarith

end NotTight

/-! ### Greedy decoding is not maximum a posteriori -/

section Greedy

/-- A two-step model over `Bool` that never stops before two tokens. First token:
`true` with `3/5`, `false` with `2/5`. After `true`: either token with `1/2`. After
`false`: `true` with certainty. -/
def twoStep : LM 𝕜 Bool := fun pre =>
  match pre with
  | [] =>
    { prob := fun o => match o with
        | none => 0 | some true => 3 / 5 | some false => 2 / 5
      nonneg := fun o => by rcases o with _ | _ | _ <;> norm_num
      sum_eq_one := by simp [Fintype.sum_option]; norm_num }
  | [true] =>
    { prob := fun o => match o with
        | none => 0 | some _ => 1 / 2
      nonneg := fun o => by rcases o with _ | _ | _ <;> norm_num
      sum_eq_one := by simp [Fintype.sum_option] }
  | [false] =>
    { prob := fun o => match o with
        | none => 0 | some true => 1 | some false => 0
      nonneg := fun o => by rcases o with _ | _ | _ <;> norm_num
      sum_eq_one := by simp [Fintype.sum_option] }
  | _ =>
    { prob := fun o => match o with
        | none => 1 | some _ => 0
      nonneg := fun o => by rcases o with _ | _ | _ <;> norm_num
      sum_eq_one := by simp [Fintype.sum_option] }

/-- **Greedy decoding is not MAP decoding.** The first token `true` is strictly the
most probable, yet every two-token sequence that starts with it is strictly less
probable than `false true`. No tie-breaking rule rescues greedy decoding here. -/
theorem greedy_not_map :
    (twoStep (𝕜 := 𝕜) []).prob (some false) < (twoStep (𝕜 := 𝕜) []).prob (some true) ∧
      ∀ y : Bool, (twoStep (𝕜 := 𝕜)).alive [] [true, y] < (twoStep (𝕜 := 𝕜)).alive [] [false, true] := by
  refine ⟨by simp [twoStep]; norm_num, fun y => ?_⟩
  cases y <;> simp [LM.alive, twoStep] <;> norm_num

end Greedy

end LLM
