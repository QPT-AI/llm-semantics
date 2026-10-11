import LLM.Sampling
import Mathlib.Algebra.Order.Group.Defs

/-!
# Softmax over an exponential character

Softmax is usually written with `exp`, but nothing in its algebra needs more than
three facts about `exp`: it turns addition into multiplication, it is positive, and
it is strictly increasing. An *exponential character* is any map with these three
properties, from an ordered additive group of scores `G` into an ordered field `𝕜`.

* `Real.exp : ℝ → ℝ` is one (`LLM.Real`); this is the softmax of the literature.
* `z ↦ 2 ^ z : ℤ → ℚ` is another (`ExpChar.base2`); it is exact, and the oracle uses
  it to evaluate attention with no rounding at all.

Every theorem below holds for every exponential character, so it holds for the real
softmax and for the exact one by the same proof.

Results:

* `softmax_add_const`: adding a constant to every logit changes nothing (this is
  what licenses the "subtract the max" trick);
* `softmax_lt_iff`: softmax preserves the order of logits;
* `softmaxOn_eq_restrict`: setting the logits outside `S` to `-∞` is exactly
  conditioning on `S`, so masking at the logit level and filtering at the
  probability level agree;
* `topK_logits`: top-k computed on logits equals top-k computed on probabilities,
  at every temperature;
* `softmax_nsmul`: multiplying logits by `n` is `Dist.sharpen n`, temperature `1/n`.
-/

open Finset

namespace LLM

universe u v w

/-- An exponential character: a strictly increasing map from the additive group of
scores into the positive elements of an ordered field, turning `+` into `*`. -/
structure ExpChar (G : Type v) (𝕜 : Type u) [AddCommGroup G] [LinearOrder G]
    [Field 𝕜] [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜] where
  /-- The underlying function. -/
  toFun : G → 𝕜
  map_add' : ∀ a b, toFun (a + b) = toFun a * toFun b
  pos' : ∀ a, 0 < toFun a
  strictMono' : StrictMono toFun

namespace ExpChar

variable {G : Type v} {𝕜 : Type u} [AddCommGroup G] [LinearOrder G]
variable [Field 𝕜] [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜]

instance : CoeFun (ExpChar G 𝕜) (fun _ => G → 𝕜) := ⟨ExpChar.toFun⟩

variable (E : ExpChar G 𝕜)

lemma map_add (a b : G) : E (a + b) = E a * E b := E.map_add' a b
lemma pos (a : G) : 0 < E a := E.pos' a
lemma ne_zero (a : G) : E a ≠ 0 := ne_of_gt (E.pos a)
lemma strictMono : StrictMono E := E.strictMono'

lemma lt_iff {a b : G} : E a < E b ↔ a < b := E.strictMono.lt_iff_lt
lemma le_iff {a b : G} : E a ≤ E b ↔ a ≤ b := E.strictMono.le_iff_le

@[simp] lemma map_zero : E 0 = 1 := by
  have h := E.map_add 0 0
  rw [add_zero] at h
  exact (mul_left_cancel₀ (E.ne_zero 0) (by rw [mul_one]; exact h)).symm

lemma map_neg (a : G) : E (-a) = (E a)⁻¹ := by
  apply eq_inv_of_mul_eq_one_right
  rw [← E.map_add, add_neg_cancel, E.map_zero]

lemma map_sub (a b : G) : E (a - b) = E a / E b := by
  rw [sub_eq_add_neg, E.map_add, E.map_neg, div_eq_mul_inv]

lemma map_nsmul (n : ℕ) (a : G) : E (n • a) = E a ^ n := by
  induction n with
  | zero => simp
  | succ n ih => rw [succ_nsmul, E.map_add, ih, pow_succ]

/-- The chaining identity behind online softmax. -/
lemma mul_sub_sub (x y z : G) : E (x - y) * E (y - z) = E (x - z) := by
  rw [← E.map_add, sub_add_sub_cancel]

lemma le_one_of_nonpos {a : G} (h : a ≤ 0) : E a ≤ 1 := by
  rw [← E.map_zero]; exact E.le_iff.mpr h

end ExpChar

section Softmax

variable {G : Type v} {𝕜 : Type u} [AddCommGroup G] [LinearOrder G]
variable [Field 𝕜] [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜]
variable (E : ExpChar G 𝕜)
variable {ι : Type w} [Fintype ι]

lemma partition_pos [Nonempty ι] (z : ι → G) : 0 < ∑ j, E (z j) :=
  Finset.sum_pos (fun j _ => E.pos (z j)) univ_nonempty

/-- Softmax of a logit vector, as a distribution. -/
def softmax [Nonempty ι] (z : ι → G) : Dist 𝕜 ι :=
  Dist.normalize (fun i => E (z i)) (fun i => (E.pos (z i)).le) (partition_pos E z)

lemma softmax_prob [Nonempty ι] (z : ι → G) (i : ι) :
    (softmax E z).prob i = E (z i) / ∑ j, E (z j) := rfl

/-- Softmax gives every token positive probability; only masking can exclude one. -/
lemma softmax_pos [Nonempty ι] (z : ι → G) (i : ι) : 0 < (softmax E z).prob i :=
  div_pos (E.pos _) (partition_pos E z)

/-- **Shift invariance.** -/
theorem softmax_add_const [Nonempty ι] (z : ι → G) (c : G) :
    softmax E (fun i => z i + c) = softmax E z := by
  ext i
  simp only [softmax_prob, E.map_add, ← Finset.sum_mul]
  have := partition_pos E z
  have := E.ne_zero c
  field_simp

/-- Ratios of softmax probabilities are exponentials of logit differences. -/
theorem softmax_ratio [Nonempty ι] (z : ι → G) (i j : ι) :
    (softmax E z).prob i / (softmax E z).prob j = E (z i - z j) := by
  simp only [softmax_prob, E.map_sub]
  have := partition_pos E z
  have := E.ne_zero (z j)
  field_simp

/-- Softmax preserves the order of logits. -/
theorem softmax_lt_iff [Nonempty ι] (z : ι → G) (i j : ι) :
    (softmax E z).prob i < (softmax E z).prob j ↔ z i < z j := by
  simp only [softmax_prob]
  rw [div_lt_div_iff_of_pos_right (partition_pos E z), E.lt_iff]

/-- **Top-k on logits equals top-k on probabilities.** -/
theorem topK_logits [Nonempty ι] (z : ι → G) (k : ℕ) :
    topKSet k (softmax E z).prob = topKSet k z :=
  topKSet_congr (fun u v => softmax_lt_iff E z u v) k

/-- **Integer temperature.** Multiplying every logit by `n` (temperature `1/n`) is
`Dist.sharpen n`: each probability raised to the `n`-th power and renormalized. -/
theorem softmax_nsmul [Nonempty ι] (z : ι → G) (n : ℕ) :
    softmax E (fun i => n • z i) = (softmax E z).sharpen n := by
  ext i
  simp only [softmax_prob, Dist.sharpen, Dist.normalize_prob, E.map_nsmul, div_pow]
  rw [← sum_div_const]
  have hZ := partition_pos E z
  have hZn : (∑ j, E (z j)) ^ n ≠ 0 := pow_ne_zero n (ne_of_gt hZ)
  have hS : 0 < ∑ j, E (z j) ^ n := Finset.sum_pos (fun j _ => pow_pos (E.pos _) n) univ_nonempty
  field_simp

variable [DecidableEq ι]

/-- Softmax restricted to a nonempty set of allowed tokens: the meaning of setting
every other logit to `-∞`. -/
def softmaxOn (S : Finset ι) (hS : S.Nonempty) (z : ι → G) : Dist 𝕜 ι where
  prob i := if i ∈ S then E (z i) / ∑ j ∈ S, E (z j) else 0
  nonneg i := by
    split_ifs
    · exact div_nonneg (E.pos _).le (Finset.sum_nonneg (fun j _ => (E.pos _).le))
    · exact le_rfl
  sum_eq_one := by
    classical
    rw [Finset.sum_ite_mem, Finset.univ_inter, ← sum_div_const]
    exact div_self (ne_of_gt (Finset.sum_pos (fun j _ => E.pos _) hS))

omit [DecidableEq ι] in
lemma softmax_mass_pos [Nonempty ι] (z : ι → G) {S : Finset ι} (hS : S.Nonempty) :
    0 < (softmax E z).mass S := by
  obtain ⟨i, hi⟩ := hS
  exact (softmax E z).mass_pos_of_mem hi (softmax_pos E z i)

/-- **Masking is conditioning.** Masking logits to `-∞` outside `S` and then taking
softmax gives the same distribution as taking softmax and then conditioning on `S`. -/
theorem softmaxOn_eq_restrict [Nonempty ι] (S : Finset ι) (hS : S.Nonempty) (z : ι → G) :
    softmaxOn E S hS z = (softmax E z).restrict S (softmax_mass_pos E z hS) := by
  ext i
  simp only [softmaxOn, Dist.restrict_prob, Dist.mass, softmax_prob]
  split_ifs
  · rw [← sum_div_const]
    have := partition_pos E z
    have := Finset.sum_pos (fun j _ => E.pos (z j)) hS
    field_simp
  · rfl

/-- Filtering logits by top-k and then taking softmax is `Dist.topK` of the softmax.
Implementations that filter logits and implementations that filter probabilities
therefore agree. -/
theorem softmaxOn_topK [Nonempty ι] (z : ι → G) {k : ℕ} (hk : 0 < k) (hS : (topKSet k z).Nonempty) :
    softmaxOn E (topKSet k z) hS z = (softmax E z).topK k hk := by
  rw [softmaxOn_eq_restrict]
  unfold Dist.topK
  exact Dist.restrict_eq_of_eq _ (topK_logits E z k).symm _ _

end Softmax

section Temperature

variable {𝕜 : Type u} [Field 𝕜] [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜]
variable {G : Type v} [Field G] [LinearOrder G] [IsStrictOrderedRing G]
variable (E : ExpChar G 𝕜)
variable {ι : Type w} [Fintype ι]

/-- **Top-k does not see temperature.** For every temperature `T > 0`, top-k of
`softmax (z / T)` keeps the same tokens as top-k of the logits. -/
theorem topK_temperature [Nonempty ι] (z : ι → G) {T : G} (hT : 0 < T) (k : ℕ) :
    topKSet k (softmax E (fun i => z i / T)).prob = topKSet k z := by
  rw [topK_logits]
  exact topKSet_congr (fun u v => div_lt_div_iff_of_pos_right hT) k

end Temperature

/-- The exact exponential character `z ↦ 2 ^ z` from integers to rationals. -/
def ExpChar.base2 : ExpChar ℤ ℚ where
  toFun z := (2 : ℚ) ^ z
  map_add' a b := zpow_add₀ (by norm_num) a b
  pos' a := zpow_pos (by norm_num) a
  strictMono' := zpow_right_strictMono₀ (by norm_num)

end LLM
