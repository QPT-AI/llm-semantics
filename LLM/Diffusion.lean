import LLM.Dist
import Mathlib.Algebra.Order.Interval.Set.Instances
import Mathlib.CategoryTheory.Endomorphism
import Mathlib.Data.Fintype.Option
import Mathlib.Data.Fin.Tuple.Basic
import Mathlib.Tactic.FinCases

/-!
# Masked (absorbing) diffusion

A masked diffusion language model corrupts a sequence by replacing each token,
independently, with a mask symbol, and learns to reverse the corruption. Tokens are
`some v`, and the mask is `none`.

* `maskK α`: the forward kernel that keeps a token with probability `α`.
  `forward` is a monoid homomorphism from the multiplicative unit interval into the
  endomorphisms of `Option V` in `FinStoch`: masking at rate `α` and then at rate
  `β` is masking at rate `α β`, and masking at rate one is the identity;
* `qSeq_closed`: the closed form of the sequence-level forward process;
* `posterior_time_independent`: **the optimal denoiser does not depend on the
  noise level.** The posterior over clean sequences given a partially masked one
  is the data distribution conditioned on agreeing with the unmasked tokens, the
  same for every `0 < α < 1`;
* `reverse_bayes`: the closed-form reverse step used by samplers is the Bayes
  posterior of the forward process;
* `chain_rule`: for *any* order of positions, the product of exact conditionals
  is the data probability. Unmasking one token at a time with the exact denoiser
  therefore samples the data distribution exactly, whatever the order;
* **negative result** `parallel_decoding_inexact`: unmasking two positions in one
  step from their separate posteriors does not sample the data distribution; it
  can give positive probability to sequences the data never produces.
-/

open Finset

namespace LLM.Diffusion

universe u v w

variable {𝕜 : Type u} [Field 𝕜] [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜]
variable {V : Type v} [Fintype V] [DecidableEq V]

/-! ### The forward kernel -/

/-- The transition probability of the masking kernel with keep rate `α`. -/
def maskProb (α : 𝕜) : Option V → Option V → 𝕜
  | none, none => 1
  | none, some _ => 0
  | some _, none => 1 - α
  | some v, some w => if w = v then α else 0

omit [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜] in
lemma sum_maskProb (α : 𝕜) (x : Option V) : ∑ y, maskProb α x y = 1 := by
  cases x <;> simp [Fintype.sum_option, maskProb]

omit [Fintype V] in
lemma maskProb_nonneg {α : 𝕜} (h0 : 0 ≤ α) (h1 : α ≤ 1) (x y : Option V) :
    0 ≤ maskProb α x y := by
  rcases x with _ | v <;> rcases y with _ | w <;> simp only [maskProb]
  · exact zero_le_one
  · exact le_rfl
  · linarith
  · split_ifs
    · exact h0
    · exact le_rfl

/-- The masking kernel. -/
def maskK (α : 𝕜) (h0 : 0 ≤ α) (h1 : α ≤ 1) (x : Option V) : Dist 𝕜 (Option V) where
  prob y := maskProb α x y
  nonneg y := maskProb_nonneg h0 h1 x y
  sum_eq_one := sum_maskProb α x

omit [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜] in
/-- **Chapman–Kolmogorov for masking.** -/
theorem maskProb_comp (α β : 𝕜) (x z : Option V) :
    ∑ y, maskProb α x y * maskProb β y z = maskProb (α * β) x z := by
  cases x with
  | none => cases z <;> simp [Fintype.sum_option, maskProb]
  | some v =>
    cases z with
    | none => simp [Fintype.sum_option, maskProb]; ring
    | some w =>
      simp only [Fintype.sum_option, maskProb, mul_ite, mul_zero, ite_mul, zero_mul]
      by_cases hw : w = v
      · subst hw; simp
      · simp [hw]

open CategoryTheory

/-- The object `Option V` of `FinStoch`. -/
abbrev obj (𝕜 : Type u) [Field 𝕜] [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜] (V : Type v)
    [Fintype V] [DecidableEq V] : FinStoch.{u, v} 𝕜 := FinStoch.of (Option V)

/-- **The forward process is a monoid action.** Keep rates multiply. -/
def forward : Set.Icc (0 : 𝕜) 1 →* End (obj 𝕜 V) where
  toFun α := fun x => maskK α.1 α.2.1 α.2.2 x
  map_one' := by
    funext x
    ext y
    show maskProb 1 x y = (Dist.pure x).prob y
    cases x <;> cases y <;> simp [maskProb, eq_comm]
  map_mul' α β := by
    funext x
    ext z
    show maskProb (α.1 * β.1) x z = ∑ y, maskProb β.1 x y * maskProb α.1 y z
    rw [maskProb_comp, mul_comm]

/-! ### The sequence-level forward process -/

variable {ι : Type w} [Fintype ι] [DecidableEq ι]

/-- `y` is a masking of `x`: every position is masked or agrees with `x`. -/
def Consistent (x : ι → V) (y : ι → Option V) : Prop := ∀ i, y i = none ∨ y i = some (x i)

instance (x : ι → V) (y : ι → Option V) : Decidable (Consistent x y) := by
  unfold Consistent; infer_instance

/-- The number of masked positions. -/
def numMasked (y : ι → Option V) : ℕ := (univ.filter (fun i => y i = none)).card

/-- The probability that masking `x` at keep rate `α` yields `y`. -/
def qSeq (α : 𝕜) (x : ι → V) (y : ι → Option V) : 𝕜 := ∏ i, maskProb α (some (x i)) (y i)

omit [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜] [Fintype V] in
/-- **Closed form of the forward process.** -/
theorem qSeq_closed (α : 𝕜) (x : ι → V) (y : ι → Option V) :
    qSeq α x y = if Consistent x y then
      (1 - α) ^ numMasked y * α ^ (Fintype.card ι - numMasked y) else 0 := by
  unfold qSeq
  split_ifs with hc
  · have hfac : ∀ i, maskProb α (some (x i)) (y i) = if y i = none then 1 - α else α := by
      intro i
      rcases hc i with h | h <;> simp [h, maskProb]
    simp_rw [hfac]
    rw [Finset.prod_ite, Finset.prod_const, Finset.prod_const]
    congr 2
    rw [Finset.filter_not, Finset.card_sdiff_of_subset (Finset.filter_subset _ _), Finset.card_univ]
    rfl
  · unfold Consistent at hc
    push Not at hc
    obtain ⟨i, hn, hs⟩ := hc
    apply Finset.prod_eq_zero (mem_univ i)
    rcases hy : y i with _ | w
    · exact absurd hy hn
    · rw [hy] at hs
      have : w ≠ x i := fun h => hs (by rw [h])
      simp [maskProb, this]

/-- **The optimal denoiser is time-independent.** For `0 < α < 1`, the posterior over
clean sequences given the masked sequence `y` is the data distribution restricted
to the sequences consistent with `y`, and does not depend on `α`. -/
theorem posterior_time_independent (p : Dist 𝕜 (ι → V)) (y : ι → Option V) {α : 𝕜}
    (h0 : 0 < α) (h1 : α < 1) (x : ι → V) :
    p.prob x * qSeq α x y / ∑ x', p.prob x' * qSeq α x' y =
      (if Consistent x y then p.prob x else 0) / ∑ x', (if Consistent x' y then p.prob x' else 0) := by
  set c : 𝕜 := (1 - α) ^ numMasked y * α ^ (Fintype.card ι - numMasked y)
  have hc : c ≠ 0 := by
    have : 0 < 1 - α := by linarith
    positivity
  have hq : ∀ x', p.prob x' * qSeq α x' y = c * (if Consistent x' y then p.prob x' else 0) := by
    intro x'
    rw [qSeq_closed]
    split_ifs <;> ring
  simp_rw [hq]
  rw [← Finset.mul_sum, mul_div_mul_left _ _ hc]

/-- The evidence for an observation: the data mass consistent with it. -/
def evidence (p : Dist 𝕜 (ι → V)) (y : ι → Option V) : 𝕜 :=
  ∑ x', if Consistent x' y then p.prob x' else 0

/-- The time-independent posterior over clean sequences given a masked one. -/
def posterior (p : Dist 𝕜 (ι → V)) (y : ι → Option V) (x : ι → V) : 𝕜 :=
  (if Consistent x y then p.prob x else 0) / evidence p y

/-- The Bayes posterior at every noise level `0 < α < 1` is `posterior`. -/
theorem bayes_eq_posterior (p : Dist 𝕜 (ι → V)) (y : ι → Option V) {α : 𝕜}
    (h0 : 0 < α) (h1 : α < 1) (x : ι → V) :
    p.prob x * qSeq α x y / ∑ x', p.prob x' * qSeq α x' y = posterior p y x :=
  posterior_time_independent p y h0 h1 x

/-! ### The reverse step -/

/-- The reverse step from keep rate `αt` back to the higher keep rate `αs`, given
the clean token `x₀` and the current symbol `xt`: an unmasked symbol stays, and a
mask is revealed as `x₀` with probability `(αs - αt)/(1 - αt)`. -/
def revProb (αs αt : 𝕜) (x₀ : V) (xt xs : Option V) : 𝕜 :=
  match xt, xs with
  | some w, some w' => if w' = w then 1 else 0
  | some _, none => 0
  | none, none => (1 - αs) / (1 - αt)
  | none, some v => if v = x₀ then (αs - αt) / (1 - αt) else 0

omit [Fintype V] in
/-- **Bayes identity for the reverse step.** Masking to rate `αs` and then on to
`αt` is the same joint law as masking straight to `αt` and then running the reverse
step. -/
theorem reverse_bayes {αs αt : 𝕜} (hs : 0 < αs) (ht1 : αt < 1) (x₀ : V) (xs xt : Option V) :
    maskProb αs (some x₀) xs * maskProb (αt / αs) xs xt =
      maskProb αt (some x₀) xt * revProb αs αt x₀ xt xs := by
  have hne : (1 : 𝕜) - αt ≠ 0 := by linarith
  have hsne : αs ≠ 0 := ne_of_gt hs
  rcases xs with _ | v <;> rcases xt with _ | w
  · simp only [maskProb, revProb, mul_one]
    field_simp
  · simp only [maskProb, revProb, mul_zero]
  · simp only [maskProb, revProb]
    split_ifs with hv
    · field_simp
    · simp
  · simp only [maskProb, revProb]
    by_cases hwv : w = v
    · subst hwv
      by_cases hv : w = x₀
      · subst hv
        simp only [ite_true]
        field_simp
      · simp [hv]
    · have hvw : v ≠ w := fun h => hwv h.symm
      simp [hwv, hvw]

/-- The reverse step is a distribution. -/
theorem sum_revProb {αs αt : 𝕜} (ht1 : αt < 1) (x₀ : V) (xt : Option V) :
    ∑ xs, revProb αs αt x₀ xt xs = 1 := by
  have hne : (1 : 𝕜) - αt ≠ 0 := by linarith
  cases xt with
  | none =>
    simp only [Fintype.sum_option, revProb]
    rw [Finset.sum_ite_eq']
    simp only [mem_univ, ite_true]
    field_simp
    ring
  | some w => simp [Fintype.sum_option, revProb]

/-! ### Any-order chain rule -/

/-- The marginal probability of agreeing with `x` on the positions in `S`. -/
def marg (p : Dist 𝕜 (ι → V)) (S : Finset ι) (x : ι → V) : 𝕜 :=
  ∑ x' ∈ univ.filter (fun x' : ι → V => ∀ i ∈ S, x' i = x i), p.prob x'

lemma marg_empty (p : Dist 𝕜 (ι → V)) (x : ι → V) : marg p ∅ x = 1 := by
  simp [marg, p.sum_eq_one]

lemma marg_univ (p : Dist 𝕜 (ι → V)) (x : ι → V) : marg p univ x = p.prob x := by
  unfold marg
  have : univ.filter (fun x' : ι → V => ∀ i ∈ (univ : Finset ι), x' i = x i) = {x} := by
    ext x'; simp [funext_iff]
  rw [this, Finset.sum_singleton]

lemma marg_nonneg (p : Dist 𝕜 (ι → V)) (S : Finset ι) (x : ι → V) : 0 ≤ marg p S x :=
  Finset.sum_nonneg (fun _ _ => p.nonneg _)

lemma marg_anti (p : Dist 𝕜 (ι → V)) {S T : Finset ι} (h : S ⊆ T) (x : ι → V) :
    marg p T x ≤ marg p S x := by
  apply Finset.sum_le_sum_of_subset_of_nonneg
  · intro x' hx'
    simp only [Finset.mem_filter, mem_univ, true_and] at *
    exact fun i hi => hx' i (h hi)
  · exact fun _ _ _ => p.nonneg _

/-- The exact conditional probability that position `i` holds `x i`, given that the
positions in `S` agree with `x`. -/
def cond (p : Dist 𝕜 (ι → V)) (S : Finset ι) (i : ι) (x : ι → V) : 𝕜 :=
  marg p (insert i S) x / marg p S x

lemma telescope (M : ℕ → 𝕜) (h0 : M 0 = 1) (hanti : ∀ k, M (k + 1) ≤ M k)
    (hnn : ∀ k, 0 ≤ M k) (n : ℕ) : ∏ k ∈ range n, M (k + 1) / M k = M n := by
  induction n with
  | zero => simp [h0]
  | succ n ih =>
    rw [Finset.prod_range_succ, ih]
    by_cases hM : M n = 0
    · rw [hM, div_zero, mul_zero]
      exact (le_antisymm (hM ▸ hanti n) (hnn (n + 1))).symm
    · field_simp

/-- **Any-order chain rule.** For any enumeration `ord` of the positions, the product
of the exact conditionals along `ord` is the probability of `x`. Unmasking one
position at a time with the exact denoiser samples the data distribution in every
order. -/
theorem chain_rule (p : Dist 𝕜 (ι → V)) (ord : List ι) (hcover : ∀ i, i ∈ ord) (x : ι → V) :
    p.prob x = ∏ k ∈ range ord.length,
      if hk : k < ord.length then cond p (ord.take k).toFinset ord[k] x else 1 := by
  have hstep : ∀ k ∈ range ord.length,
      (if hk : k < ord.length then cond p (ord.take k).toFinset ord[k] x else 1) =
        marg p (ord.take (k + 1)).toFinset x / marg p (ord.take k).toFinset x := by
    intro k hk
    have hk' : k < ord.length := Finset.mem_range.mp hk
    rw [dite_eq_left hk', cond]
    have htk : ord.take (k + 1) = ord.take k ++ [ord[k]] := by
      rw [List.take_add_one, List.getElem?_eq_getElem hk']; rfl
    rw [htk]
    congr 2
    ext j
    simp only [List.mem_toFinset, Finset.mem_insert, List.mem_append, List.mem_singleton]
    exact or_comm
  rw [Finset.prod_congr rfl hstep,
    telescope (fun k => marg p (ord.take k).toFinset x) (by simp [marg_empty])
      (fun k => marg_anti p (by
        intro j hj
        simp only [List.mem_toFinset] at *
        exact List.take_subset_take_left _ (Nat.le_succ k) hj) x)
      (fun k => marg_nonneg p _ x)]
  have : (ord.take ord.length).toFinset = univ := by
    ext j; simp [hcover j]
  simp only [this, marg_univ]

/-- The any-order chain rule as a product over the steps of the order. -/
theorem chain_rule_fin (p : Dist 𝕜 (ι → V)) (ord : List ι) (hcover : ∀ i, i ∈ ord) (x : ι → V) :
    p.prob x = ∏ k : Fin ord.length, cond p (ord.take k).toFinset ord[k] x := by
  rw [chain_rule p ord hcover x, ← Fin.prod_univ_eq_prod_range]
  refine Finset.prod_congr rfl (fun k _ => ?_)
  rw [dite_eq_left k.isLt]
  rfl

/-- The masked sequence that reveals exactly the positions in `S`. -/
def reveal (S : Finset ι) (x : ι → V) : ι → Option V := fun i => if i ∈ S then some (x i) else none

omit [Fintype V] [DecidableEq V] [Fintype ι] in
lemma consistent_reveal_iff (S : Finset ι) (x x' : ι → V) :
    Consistent x' (reveal S x) ↔ ∀ i ∈ S, x' i = x i := by
  unfold Consistent reveal
  constructor
  · intro h i hi
    rcases h i with h' | h' <;> simp [hi] at h'
    exact h'.symm
  · intro h i
    by_cases hi : i ∈ S
    · right; simp [hi, h i hi]
    · left; simp [hi]

/-- **The denoiser's marginal is the chain-rule conditional.** The probability the
time-independent posterior assigns to position `i` holding `x i`, given the tokens
of `x` on `S`, is `cond p S i x`. -/
theorem posterior_marginal (p : Dist 𝕜 (ι → V)) (S : Finset ι) (i : ι) (x : ι → V) :
    ∑ x' ∈ univ.filter (fun x' : ι → V => x' i = x i),
        (if Consistent x' (reveal S x) then p.prob x' else 0) /
          ∑ x'', (if Consistent x'' (reveal S x) then p.prob x'' else 0) =
      cond p S i x := by
  simp_rw [consistent_reveal_iff]
  rw [← sum_div_const]
  unfold cond marg
  congr 1
  · rw [Finset.sum_filter, Finset.sum_filter]
    refine Finset.sum_congr rfl (fun x' _ => ?_)
    simp only [Finset.mem_insert, forall_eq_or_imp]
    by_cases h1 : x' i = x i <;> by_cases h2 : ∀ j ∈ S, x' j = x j <;> simp [h1, h2]
  · rw [Finset.sum_filter]

/-! ### Parallel decoding is inexact -/

/-- Two positions holding opposite bits, each pattern with probability `1/2`. -/
def antiCopy : Dist 𝕜 (Fin 2 → Bool) where
  prob x := if x 0 ≠ x 1 then 1 / 2 else 0
  nonneg x := by split_ifs <;> norm_num
  sum_eq_one := by
    rw [← (piFinTwoEquiv (fun _ => Bool)).symm.sum_comp, Fintype.sum_prod_type]
    simp [piFinTwoEquiv]
    norm_num

/-- **Parallel decoding is inexact.** From the fully masked sequence, the exact
posterior of each position is uniform. Unmasking both positions in one step from
these separate posteriors gives the sequence `true true` probability `1/4`, while the
data never produces it. -/
theorem parallel_decoding_inexact :
    let x : Fin 2 → Bool := fun _ => true
    cond antiCopy ∅ 0 x * cond antiCopy ∅ 1 x = (1 : 𝕜) / 4 ∧ (antiCopy (𝕜 := 𝕜)).prob x = 0 := by
  intro x
  constructor
  · have hm : ∀ i : Fin 2, marg (antiCopy (𝕜 := 𝕜)) {i} x = 1 / 2 := by
      intro i
      unfold marg antiCopy
      rw [Finset.sum_filter, ← (piFinTwoEquiv (fun _ => Bool)).symm.sum_comp,
        Fintype.sum_prod_type]
      fin_cases i <;> simp [piFinTwoEquiv, x]
    simp only [cond, Finset.insert_empty, marg_empty, hm]
    norm_num
  · simp [antiCopy, x]

end LLM.Diffusion
