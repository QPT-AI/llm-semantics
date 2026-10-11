import LLM.Dist
import Mathlib.Logic.Equiv.Basic
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Fin.VecNotation
import Mathlib.Tactic.NormNum

/-!
# Sampling filters

A decoder never samples from the model's distribution directly. It first passes
the distribution through a chain of filters (top-k, nucleus or top-p, min-p,
temperature), each of which keeps a set of tokens and renormalizes. This module
gives each filter a meaning and proves the algebra of the chain:

* every filter is `Dist.restrict` to a kept set, and every kept set contains a most
  probable token, so no filter can produce an empty distribution;
* `topK k ∘ topK k' = topK (min k k')`, `minP m ∘ minP m' = minP (max m m')`, and
  top-k commutes with min-p and with temperature;
* top-p is **not** idempotent, and does not commute with top-k or with
  temperature (`LLM.Counterexamples`);
* **negative result** (`no_equivariant_exact_topK`, `no_equivariant_exact_nucleus`):
  no rule that treats tokens symmetrically can always keep exactly `k` tokens, or
  always return a smallest nucleus. An implementation must either keep ties
  (the rules defined here) or break them by token identity, and it must say which;
* inverse-CDF sampling is exact: the uniform numbers mapped to token `i` form an
  interval of length `p i` (`invCdf_eq_iff`).
-/

open Finset

namespace LLM

universe u v

section Order

variable {V : Type v} [Fintype V]
variable {β : Type*} [LinearOrder β]

/-- The tokens strictly preferred to `v` by the score `p`. -/
def above (p : V → β) (v : V) : Finset V := univ.filter (fun u => p v < p u)

@[simp] lemma mem_above {p : V → β} {v u : V} : u ∈ above p v ↔ p v < p u := by
  simp [above]

lemma not_mem_above_self (p : V → β) (v : V) : v ∉ above p v := by simp

lemma above_eq_empty_of_max {p : V → β} {a : V} (ha : ∀ b, p b ≤ p a) : above p a = ∅ := by
  ext u; simp [ha u]

lemma above_congr {p : V → β} {β' : Type*} [LinearOrder β'] {q : V → β'}
    (h : ∀ u v, p u < p v ↔ q u < q v) (v : V) : above p v = above q v := by
  ext u; simp [h v u]

/-- Top-k with ties kept: a token survives if fewer than `k` tokens are strictly
preferred to it. Defined for any linearly ordered score, so it applies equally to
logits and to probabilities. -/
def topKSet (k : ℕ) (p : V → β) : Finset V := univ.filter (fun v => (above p v).card < k)

@[simp] lemma mem_topKSet {k : ℕ} {p : V → β} {v : V} :
    v ∈ topKSet k p ↔ (above p v).card < k := by simp [topKSet]

/-- Top-k depends only on the order of the scores. -/
lemma topKSet_congr {p : V → β} {β' : Type*} [LinearOrder β'] {q : V → β'}
    (h : ∀ u v, p u < p v ↔ q u < q v) (k : ℕ) : topKSet k p = topKSet k q := by
  ext v; simp only [mem_topKSet, above_congr h v]

lemma argmax_mem_topKSet {k : ℕ} (hk : 0 < k) {p : V → β} {a : V} (ha : ∀ b, p b ≤ p a) :
    a ∈ topKSet k p := by
  simp [above_eq_empty_of_max ha, hk]

/-- A kept set is *upward closed* if it contains every token preferred to a kept token.
All three standard filters have upward-closed kept sets. -/
def UpClosed (p : V → β) (S : Finset V) : Prop := ∀ v ∈ S, ∀ u, p v < p u → u ∈ S

lemma topKSet_upClosed (k : ℕ) (p : V → β) : UpClosed p (topKSet k p) := by
  intro v hv u hvu
  rw [mem_topKSet] at *
  have hsub : above p u ⊆ above p v := fun w hw => by
    rw [mem_above] at *; exact lt_trans hvu hw
  have hss : above p u ⊂ above p v :=
    Finset.ssubset_iff_subset_ne.mpr ⟨hsub, fun heq => by
      have : u ∈ above p v := by simpa using hvu
      rw [← heq] at this; exact not_mem_above_self p u this⟩
  exact lt_trans (Finset.card_lt_card hss) hv

/-- A selection rule is equivariant when relabeling the tokens relabels its output:
the rule treats tokens symmetrically and cannot prefer one by its identity. -/
def Equivariant {γ : Type*} (sel : (V → γ) → Finset V) : Prop :=
  ∀ (σ : Equiv.Perm V) (p : V → γ), sel (p ∘ σ.symm) = (sel p).map σ.toEmbedding

lemma topKSet_equivariant (k : ℕ) : Equivariant (V := V) (topKSet (β := β) k) := by
  intro σ p
  ext v
  simp only [mem_topKSet, Finset.mem_map_equiv]
  have : above (p ∘ σ.symm) v = (above p (σ.symm v)).map σ.toEmbedding := by
    ext u
    simp only [mem_above, Function.comp_apply, Finset.mem_map_equiv]
  rw [this, Finset.card_map]

/-- An equivariant rule must keep nothing or everything on a constant input. -/
lemma equivariant_const [DecidableEq V] {γ : Type*} {sel : (V → γ) → Finset V} (hsel : Equivariant sel)
    (c : γ) : sel (fun _ => c) = ∅ ∨ sel (fun _ => c) = univ := by
  by_cases h : sel (fun _ => c) = ∅
  · exact Or.inl h
  · right
    obtain ⟨v, hv⟩ := Finset.nonempty_iff_ne_empty.mpr h
    ext u
    simp only [mem_univ, iff_true]
    have key := hsel (Equiv.swap v u) (fun _ => c)
    have hcomp : ((fun _ => c) ∘ (Equiv.swap v u).symm : V → γ) = fun _ => c := rfl
    rw [hcomp] at key
    rw [key, Finset.mem_map_equiv]
    simpa [Equiv.swap_apply_right] using hv

end Order

section Field

variable {𝕜 : Type u} [Field 𝕜] [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜]
variable {V : Type v} [Fintype V]

/-- Nucleus (top-p) with ties kept: a token survives if the probability mass strictly
preferred to it is below `P`. For distinct probabilities this is the usual rule "sort
descending and keep the shortest prefix whose mass reaches `P`". -/
def topPSet (P : 𝕜) (p : V → 𝕜) : Finset V :=
  univ.filter (fun v => ∑ u ∈ above p v, p u < P)

omit [IsStrictOrderedRing 𝕜] in
@[simp] lemma mem_topPSet {P : 𝕜} {p : V → 𝕜} {v : V} :
    v ∈ topPSet P p ↔ ∑ u ∈ above p v, p u < P := by simp [topPSet]

/-- Min-p: a token survives if its probability is at least `m` times every
probability, that is, at least `m` times the maximum. -/
def minPSet (m : 𝕜) (p : V → 𝕜) : Finset V := univ.filter (fun v => ∀ u, m * p u ≤ p v)

omit [IsStrictOrderedRing 𝕜] in
@[simp] lemma mem_minPSet {m : 𝕜} {p : V → 𝕜} {v : V} :
    v ∈ minPSet m p ↔ ∀ u, m * p u ≤ p v := by simp [minPSet]

omit [IsStrictOrderedRing 𝕜] in
lemma argmax_mem_topPSet {P : 𝕜} (hP : 0 < P) {p : V → 𝕜} {a : V} (ha : ∀ b, p b ≤ p a) :
    a ∈ topPSet P p := by
  simp [above_eq_empty_of_max ha, hP]

lemma argmax_mem_minPSet {m : 𝕜} (hm : m ≤ 1) {p : V → 𝕜} (hp : ∀ u, 0 ≤ p u) {a : V}
    (ha : ∀ b, p b ≤ p a) : a ∈ minPSet m p := by
  rw [mem_minPSet]
  intro u
  exact le_trans (mul_le_of_le_one_left (hp u) hm) (ha u)

lemma topPSet_upClosed (P : 𝕜) (p : V → 𝕜) (hp : ∀ u, 0 ≤ p u) :
    UpClosed p (topPSet P p) := by
  intro v hv u hvu
  rw [mem_topPSet] at *
  have hsub : above p u ⊆ above p v := fun w hw => by
    rw [mem_above] at *; exact lt_trans hvu hw
  exact lt_of_le_of_lt (Finset.sum_le_sum_of_subset_of_nonneg hsub (fun w _ _ => hp w)) hv

omit [IsStrictOrderedRing 𝕜] in
lemma minPSet_upClosed (m : 𝕜) (p : V → 𝕜) : UpClosed p (minPSet m p) := by
  intro v hv u hvu
  rw [mem_minPSet] at *
  exact fun w => le_trans (hv w) hvu.le

namespace Dist

/-- The positive mass of any set containing a most probable token. -/
lemma mass_pos_of_argmax_mem (d : Dist 𝕜 V) {S : Finset V}
    (h : ∀ a, (∀ b, d.prob b ≤ d.prob a) → a ∈ S) : 0 < d.mass S := by
  obtain ⟨a, hpos, hmax⟩ := d.exists_argmax
  exact d.mass_pos_of_mem (h a hmax) hpos

variable [DecidableEq V]

/-- Top-k filtering, `k ≥ 1`. -/
def topK (k : ℕ) (hk : 0 < k) (d : Dist 𝕜 V) : Dist 𝕜 V :=
  d.restrict (topKSet k d.prob) (d.mass_pos_of_argmax_mem (fun _ ha => argmax_mem_topKSet hk ha))

/-- Nucleus filtering, `P > 0`. -/
def topP (P : 𝕜) (hP : 0 < P) (d : Dist 𝕜 V) : Dist 𝕜 V :=
  d.restrict (topPSet P d.prob) (d.mass_pos_of_argmax_mem (fun _ ha => argmax_mem_topPSet hP ha))

/-- Min-p filtering, `m ≤ 1`. -/
def minP (m : 𝕜) (hm : m ≤ 1) (d : Dist 𝕜 V) : Dist 𝕜 V :=
  d.restrict (minPSet m d.prob)
    (d.mass_pos_of_argmax_mem (fun _ ha => argmax_mem_minPSet hm d.nonneg ha))

/-- Temperature `1/n`: raise every probability to the `n`-th power and renormalize.
This is exactly `softmax(n · z)` for logits `z` (see `LLM.Softmax`), and it is the
form of temperature that stays inside the rationals. -/
def sharpen (n : ℕ) (d : Dist 𝕜 V) : Dist 𝕜 V :=
  normalize (fun v => d.prob v ^ n) (fun v => pow_nonneg (d.nonneg v) n) (by
    obtain ⟨a, hpos, -⟩ := d.exists_argmax
    exact lt_of_lt_of_le (pow_pos hpos n)
      (Finset.single_le_sum (f := fun v => d.prob v ^ n)
        (fun v _ => pow_nonneg (d.nonneg v) n) (mem_univ a)))

lemma restrict_eq_of_eq (d : Dist 𝕜 V) {S S' : Finset V} (h : S = S') (hS : 0 < d.mass S)
    (hS' : 0 < d.mass S') : d.restrict S hS = d.restrict S' hS' := by
  subst h; rfl

/-- Conditioning twice is conditioning once on any set equal to the intersection. -/
lemma restrict_restrict_eq (d : Dist 𝕜 V) {S T U : Finset V} (hS : 0 < d.mass S)
    (hT : 0 < (d.restrict S hS).mass T) (hU : 0 < d.mass U) (h : S ∩ T = U) :
    (d.restrict S hS).restrict T hT = d.restrict U hU := by
  have hpos : 0 < d.mass (S ∩ T) := h ▸ hU
  rw [restrict_restrict _ _ _ hS hT hpos]
  exact restrict_eq_of_eq _ h _ _

/-- After conditioning on an upward-closed set, the tokens preferred to a kept token
are unchanged. -/
lemma above_restrict (d : Dist 𝕜 V) {S : Finset V} (hS : 0 < d.mass S)
    (hup : UpClosed d.prob S) {v : V} (hv : v ∈ S) :
    above (d.restrict S hS).prob v = above d.prob v := by
  ext u
  simp only [mem_above, restrict_prob, hv, ite_true]
  by_cases hu : u ∈ S
  · simp only [hu, ite_true]
    exact div_lt_div_iff_of_pos_right hS
  · simp only [hu, ite_false]
    constructor
    · intro h; exact absurd h (not_lt.mpr (div_nonneg (d.nonneg v) hS.le))
    · intro h; exact absurd (hup v hv u h) hu

/-- **Composition of top-k filters.** -/
theorem topK_topK (d : Dist 𝕜 V) {k k' : ℕ} (hk : 0 < k) (hk' : 0 < k') :
    (d.topK k hk).topK k' hk' = d.topK (min k k') (lt_min hk hk') := by
  unfold topK
  apply restrict_restrict_eq
  ext v
  simp only [Finset.mem_inter, mem_topKSet, lt_min_iff]
  constructor
  · rintro ⟨hv, hv'⟩
    rw [d.above_restrict _ (topKSet_upClosed k _) (mem_topKSet.mpr hv)] at hv'
    exact ⟨hv, hv'⟩
  · rintro ⟨hv, hv'⟩
    refine ⟨hv, ?_⟩
    rw [d.above_restrict _ (topKSet_upClosed k _) (mem_topKSet.mpr hv)]
    exact hv'

/-- Conditioning on a set that contains a most probable token does not change which
tokens pass a min-p test. -/
lemma minP_restrict_iff (d : Dist 𝕜 V) {S : Finset V} (hS : 0 < d.mass S)
    (hmax : ∃ a ∈ S, ∀ b, d.prob b ≤ d.prob a) (m : 𝕜) {v : V} (hv : v ∈ S) :
    (∀ u, m * (d.restrict S hS).prob u ≤ (d.restrict S hS).prob v) ↔
      ∀ u, m * d.prob u ≤ d.prob v := by
  obtain ⟨a, haS, ha⟩ := hmax
  simp only [restrict_prob, hv, ite_true]
  constructor
  · intro h u
    by_cases hu : u ∈ S
    · have := h u
      simp only [hu, ite_true] at this
      rw [mul_div_assoc', div_le_div_iff_of_pos_right hS] at this
      exact this
    · have := h a
      simp only [haS, ite_true] at this
      rw [mul_div_assoc', div_le_div_iff_of_pos_right hS] at this
      rcases le_total 0 m with hm | hm
      · exact le_trans (mul_le_mul_of_nonneg_left (ha u) hm) this
      · exact le_trans (mul_nonpos_of_nonpos_of_nonneg hm (d.nonneg u)) (d.nonneg v)
  · intro h u
    by_cases hu : u ∈ S
    · simp only [hu, ite_true]
      rw [mul_div_assoc', div_le_div_iff_of_pos_right hS]
      exact h u
    · simp only [hu, ite_false, mul_zero]
      exact div_nonneg (d.nonneg v) hS.le

/-- **Composition of min-p filters**: the stricter threshold wins. -/
theorem minP_minP (d : Dist 𝕜 V) {m m' : 𝕜} (hm : m ≤ 1) (hm' : m' ≤ 1) :
    (d.minP m hm).minP m' hm' = d.minP (max m m') (max_le hm hm') := by
  have hmaxS : ∃ a ∈ minPSet m d.prob, ∀ b, d.prob b ≤ d.prob a := by
    obtain ⟨a, -, ha⟩ := d.exists_argmax
    exact ⟨a, argmax_mem_minPSet hm d.nonneg ha, ha⟩
  unfold minP
  apply restrict_restrict_eq
  ext v
  simp only [Finset.mem_inter]
  constructor
  · rintro ⟨hv, hv'⟩
    rw [mem_minPSet, d.minP_restrict_iff _ hmaxS m' hv] at hv'
    rw [mem_minPSet] at hv ⊢
    intro u
    rw [max_mul_of_nonneg _ _ (d.nonneg u)]
    exact max_le (hv u) (hv' u)
  · intro h
    rw [mem_minPSet] at h
    have hv : ∀ u, m * d.prob u ≤ d.prob v := fun u =>
      le_trans (mul_le_mul_of_nonneg_right (le_max_left m m') (d.nonneg u)) (h u)
    have hv' : ∀ u, m' * d.prob u ≤ d.prob v := fun u =>
      le_trans (mul_le_mul_of_nonneg_right (le_max_right m m') (d.nonneg u)) (h u)
    have hvS : v ∈ minPSet m d.prob := mem_minPSet.mpr hv
    exact ⟨hvS, by rw [mem_minPSet, d.minP_restrict_iff _ hmaxS m' hvS]; exact hv'⟩

/-- Min-p is idempotent. -/
theorem minP_idem (d : Dist 𝕜 V) {m : 𝕜} (hm : m ≤ 1) :
    (d.minP m hm).minP m hm = d.minP m hm := by
  rw [minP_minP]
  exact d.restrict_eq_of_eq (by rw [max_self]) _ _

/-- Top-k is idempotent. -/
theorem topK_idem (d : Dist 𝕜 V) {k : ℕ} (hk : 0 < k) :
    (d.topK k hk).topK k hk = d.topK k hk := by
  rw [topK_topK]
  exact d.restrict_eq_of_eq (by rw [min_self]) _ _

/-- **Top-k and min-p commute.** -/
theorem topK_minP_comm (d : Dist 𝕜 V) {k : ℕ} (hk : 0 < k) {m : 𝕜} (hm : m ≤ 1) :
    (d.topK k hk).minP m hm = (d.minP m hm).topK k hk := by
  have hmaxS : ∃ a ∈ topKSet k d.prob, ∀ b, d.prob b ≤ d.prob a := by
    obtain ⟨a, -, ha⟩ := d.exists_argmax
    exact ⟨a, argmax_mem_topKSet hk ha, ha⟩
  have hpos : 0 < d.mass (topKSet k d.prob ∩ minPSet m d.prob) := by
    obtain ⟨a, hpos, ha⟩ := d.exists_argmax
    exact d.mass_pos_of_mem
      (Finset.mem_inter.mpr ⟨argmax_mem_topKSet hk ha, argmax_mem_minPSet hm d.nonneg ha⟩) hpos
  unfold topK minP
  rw [restrict_restrict_eq (U := topKSet k d.prob ∩ minPSet m d.prob) (hU := hpos),
    restrict_restrict_eq (U := topKSet k d.prob ∩ minPSet m d.prob) (hU := hpos)]
  · ext v
    simp only [Finset.mem_inter]
    constructor
    · rintro ⟨hv, hv'⟩
      rw [mem_topKSet, d.above_restrict _ (minPSet_upClosed m _) hv] at hv'
      exact ⟨mem_topKSet.mpr hv', hv⟩
    · rintro ⟨hv, hv'⟩
      refine ⟨hv', ?_⟩
      rw [mem_topKSet, d.above_restrict _ (minPSet_upClosed m _) hv']
      exact mem_topKSet.mp hv
  · ext v
    simp only [Finset.mem_inter]
    constructor
    · rintro ⟨hv, hv'⟩
      rw [mem_minPSet, d.minP_restrict_iff _ hmaxS m hv] at hv'
      exact ⟨hv, mem_minPSet.mpr hv'⟩
    · rintro ⟨hv, hv'⟩
      refine ⟨hv, ?_⟩
      rw [mem_minPSet, d.minP_restrict_iff _ hmaxS m hv]
      exact mem_minPSet.mp hv'

omit [DecidableEq V] in
lemma sharpen_partition_pos (d : Dist 𝕜 V) (n : ℕ) {S : Finset V}
    (hS : ∀ a, (∀ b, d.prob b ≤ d.prob a) → a ∈ S) : 0 < ∑ b ∈ S, d.prob b ^ n := by
  obtain ⟨a, hpos, ha⟩ := d.exists_argmax
  exact lt_of_lt_of_le (pow_pos hpos n)
    (Finset.single_le_sum (f := fun v => d.prob v ^ n)
      (fun v _ => pow_nonneg (d.nonneg v) n) (hS a ha))

omit [DecidableEq V] in
lemma sharpen_prob (d : Dist 𝕜 V) (n : ℕ) (v : V) :
    (d.sharpen n).prob v = d.prob v ^ n / ∑ b, d.prob b ^ n := rfl

omit [DecidableEq V] in
/-- The kept set of top-k is unchanged by temperature. -/
lemma topKSet_sharpen (d : Dist 𝕜 V) (k : ℕ) {n : ℕ} (hn : n ≠ 0) :
    topKSet k (d.sharpen n).prob = topKSet k d.prob := by
  apply topKSet_congr
  intro u v
  have hZ := d.sharpen_partition_pos n (S := univ) (fun a _ => mem_univ a)
  rw [sharpen_prob, sharpen_prob, div_lt_div_iff_of_pos_right hZ,
    pow_lt_pow_iff_left₀ (d.nonneg u) (d.nonneg v) hn]

/-- **Temperature commutes with top-k.** -/
theorem topK_sharpen_comm (d : Dist 𝕜 V) {k : ℕ} (hk : 0 < k) {n : ℕ} (hn : n ≠ 0) :
    (d.sharpen n).topK k hk = (d.topK k hk).sharpen n := by
  set S := topKSet k d.prob with hSdef
  have hargS : ∀ a, (∀ b, d.prob b ≤ d.prob a) → a ∈ S := fun _ ha => argmax_mem_topKSet hk ha
  have hS : 0 < d.mass S := d.mass_pos_of_argmax_mem hargS
  have hZ := d.sharpen_partition_pos n (S := univ) (fun a _ => mem_univ a)
  have hZS := d.sharpen_partition_pos n hargS
  have hmassS : 0 < (d.sharpen n).mass S := by
    unfold mass
    simp only [sharpen_prob, ← sum_div_const]
    exact div_pos hZS hZ
  have hL : (d.sharpen n).topK k hk = (d.sharpen n).restrict S hmassS :=
    restrict_eq_of_eq _ (topKSet_sharpen d k hn) _ _
  rw [hL]
  ext v
  have htopK : ∀ u, (d.topK k hk).prob u = if u ∈ S then d.prob u / d.mass S else 0 :=
    fun u => rfl
  have hpow : ∀ u, (d.topK k hk).prob u ^ n = if u ∈ S then d.prob u ^ n / d.mass S ^ n else 0 := by
    intro u; rw [htopK]; split_ifs
    · rw [div_pow]
    · exact zero_pow hn
  have hsum : ∑ u, (d.topK k hk).prob u ^ n = (∑ b ∈ S, d.prob b ^ n) / d.mass S ^ n := by
    simp_rw [hpow]
    rw [Finset.sum_ite_mem, Finset.univ_inter, sum_div_const]
  rw [restrict_prob, sharpen_prob, sharpen_prob, hpow, hsum]
  have hmass : (d.sharpen n).mass S = (∑ b ∈ S, d.prob b ^ n) / ∑ b, d.prob b ^ n := by
    unfold mass; simp only [sharpen_prob, sum_div_const]
  rw [hmass]
  have h1 : (∑ b, d.prob b ^ n) ≠ 0 := ne_of_gt hZ
  have h2 : (∑ b ∈ S, d.prob b ^ n) ≠ 0 := ne_of_gt hZS
  have h3 : d.mass S ^ n ≠ 0 := pow_ne_zero n (ne_of_gt hS)
  split_ifs
  · field_simp
  · simp

/-! ### Uniform distribution and the tie impossibility theorems -/

omit [DecidableEq V] in
/-- The uniform distribution. -/
def uniform [Nonempty V] : Dist 𝕜 V where
  prob _ := 1 / (Fintype.card V : 𝕜)
  nonneg _ := by positivity
  sum_eq_one := by
    rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
    have : (Fintype.card V : 𝕜) ≠ 0 := Nat.cast_ne_zero.mpr Fintype.card_ne_zero
    field_simp

end Dist

/-- **No equivariant exact top-k.** If `0 < k < |V|`, no selection rule that commutes
with relabeling of tokens returns exactly `k` tokens on every distribution. The
uniform distribution is the witness: symmetry forces the rule to keep all tokens or
none. So "top-k keeps exactly `k` tokens" and "top-k does not depend on token ids"
cannot both hold; an implementation chooses one and must document it. -/
theorem no_equivariant_exact_topK [DecidableEq V] {k : ℕ} (hk0 : 0 < k) (hkn : k < Fintype.card V)
    (sel : (V → 𝕜) → Finset V) (hsel : Equivariant sel) :
    ¬ ∀ d : Dist 𝕜 V, (sel d.prob).card = k := by
  intro h
  have : Nonempty V := Fintype.card_pos_iff.mp (lt_trans hk0 hkn)
  have hk := h Dist.uniform
  rcases equivariant_const hsel (1 / (Fintype.card V : 𝕜)) with h0 | h1
  · have : (sel (Dist.uniform (𝕜 := 𝕜) (V := V)).prob) = ∅ := h0
    rw [this, Finset.card_empty] at hk; omega
  · have : (sel (Dist.uniform (𝕜 := 𝕜) (V := V)).prob) = univ := h1
    rw [this, Finset.card_univ] at hk; omega

/-- A *smallest nucleus* for threshold `P`: a set of mass at least `P` whose
cardinality is minimal among all such sets. -/
def IsSmallestNucleus (P : 𝕜) (d : Dist 𝕜 V) (S : Finset V) : Prop :=
  P ≤ d.mass S ∧ ∀ T : Finset V, P ≤ d.mass T → S.card ≤ T.card

/-- **No equivariant smallest nucleus.** For `0 < P ≤ 1 - 1/|V|`, no equivariant rule
returns a smallest nucleus for every distribution. -/
theorem no_equivariant_exact_nucleus [DecidableEq V] [Nonempty V] {P : 𝕜} (hP0 : 0 < P)
    (hP1 : P ≤ 1 - 1 / (Fintype.card V : 𝕜))
    (sel : (V → 𝕜) → Finset V) (hsel : Equivariant sel) :
    ¬ ∀ d : Dist 𝕜 V, IsSmallestNucleus P d (sel d.prob) := by
  intro h
  obtain ⟨hmass, hmin⟩ := h (Dist.uniform (𝕜 := 𝕜) (V := V))
  rcases equivariant_const hsel (1 / (Fintype.card V : 𝕜)) with h0 | h1
  · have hs : sel (Dist.uniform (𝕜 := 𝕜) (V := V)).prob = ∅ := h0
    rw [hs] at hmass
    simp [Dist.mass] at hmass
    linarith
  · have hs : sel (Dist.uniform (𝕜 := 𝕜) (V := V)).prob = univ := h1
    obtain ⟨v0⟩ := ‹Nonempty V›
    have hn : 1 ≤ Fintype.card V := Fintype.card_pos
    have hT : P ≤ (Dist.uniform (𝕜 := 𝕜) (V := V)).mass (univ.erase v0) := by
      have hc : (Fintype.card V : 𝕜) ≠ 0 := Nat.cast_ne_zero.mpr Fintype.card_ne_zero
      have : (Dist.uniform (𝕜 := 𝕜) (V := V)).mass (univ.erase v0) =
          1 - 1 / (Fintype.card V : 𝕜) := by
        simp only [Dist.mass, Dist.uniform, Finset.sum_const, nsmul_eq_mul,
          Finset.card_erase_of_mem (mem_univ v0), Finset.card_univ]
        rw [Nat.cast_sub hn, Nat.cast_one]
        field_simp
      rw [this]; exact hP1
    have := hmin _ hT
    rw [hs, Finset.card_univ, Finset.card_erase_of_mem (mem_univ v0), Finset.card_univ] at this
    omega

/-! ### Positional tie-breaking

The other side of the impossibility theorems: break ties by token position. The
filters below keep exactly `min k |V|` tokens (`card_topKPosSet`) and coincide with
the tie-keeping filters when no two tokens are tied (`topKPosSet_eq_of_injective`,
`topPPosSet_eq_of_injective`). By `no_equivariant_exact_topK` they are therefore not
equivariant (`topKPosSet_not_equivariant`): which of several tied tokens survives
depends on where the tokens sit, not on their probabilities.
-/

section Positional

variable {W : Type v} [Fintype W] [LinearOrder W]

/-- `u` precedes `v`: `u` is more probable, or equally probable and earlier. -/
def Precedes (p : W → 𝕜) (u v : W) : Prop := p v < p u ∨ (p u = p v ∧ u < v)

instance (p : W → 𝕜) (u v : W) : Decidable (Precedes p u v) := by
  unfold Precedes; infer_instance

omit [Field 𝕜] [IsStrictOrderedRing 𝕜] [Fintype W] in
lemma precedes_irrefl (p : W → 𝕜) (v : W) : ¬ Precedes p v v := by simp [Precedes]

omit [Field 𝕜] [IsStrictOrderedRing 𝕜] [Fintype W] in
lemma precedes_trans {p : W → 𝕜} {u v w : W} (h1 : Precedes p u v) (h2 : Precedes p v w) :
    Precedes p u w := by
  unfold Precedes at *
  rcases h1 with h1 | ⟨h1, h1'⟩ <;> rcases h2 with h2 | ⟨h2, h2'⟩
  · exact Or.inl (lt_trans h2 h1)
  · exact Or.inl (h2 ▸ h1)
  · exact Or.inl (h1 ▸ h2)
  · exact Or.inr ⟨h1.trans h2, lt_trans h1' h2'⟩

omit [Field 𝕜] [IsStrictOrderedRing 𝕜] [Fintype W] in
lemma precedes_total {p : W → 𝕜} {u v : W} (h : u ≠ v) : Precedes p u v ∨ Precedes p v u := by
  unfold Precedes
  rcases lt_trichotomy (p u) (p v) with h1 | h1 | h1
  · exact Or.inr (Or.inl h1)
  · rcases lt_or_gt_of_ne h with h2 | h2
    · exact Or.inl (Or.inr ⟨h1, h2⟩)
    · exact Or.inr (Or.inr ⟨h1.symm, h2⟩)
  · exact Or.inl (Or.inl h1)

/-- The tokens that precede `v`. -/
def posAbove (p : W → 𝕜) (v : W) : Finset W := univ.filter (fun u => Precedes p u v)

omit [Field 𝕜] [IsStrictOrderedRing 𝕜] in
@[simp] lemma mem_posAbove {p : W → 𝕜} {u v : W} : u ∈ posAbove p v ↔ Precedes p u v := by
  simp [posAbove]

omit [Field 𝕜] [IsStrictOrderedRing 𝕜] in
lemma posAbove_ssubset {p : W → 𝕜} {u v : W} (h : Precedes p u v) :
    posAbove p u ⊂ posAbove p v := by
  rw [Finset.ssubset_iff_of_subset]
  · exact ⟨u, by simp [h], by simp [precedes_irrefl]⟩
  · intro w hw
    rw [mem_posAbove] at *
    exact precedes_trans hw h

/-- The position of `v` in the order: the number of tokens that precede it. -/
def posRank (p : W → 𝕜) (v : W) : ℕ := (posAbove p v).card

omit [Field 𝕜] [IsStrictOrderedRing 𝕜] in
lemma posRank_injective (p : W → 𝕜) : Function.Injective (posRank p) := by
  intro u v h
  by_contra hne
  rcases precedes_total (p := p) hne with h1 | h1
  · exact absurd h (ne_of_lt (Finset.card_lt_card (posAbove_ssubset h1)))
  · exact absurd h (ne_of_gt (Finset.card_lt_card (posAbove_ssubset h1)))

omit [Field 𝕜] [IsStrictOrderedRing 𝕜] in
lemma posRank_lt (p : W → 𝕜) (v : W) : posRank p v < Fintype.card W := by
  unfold posRank
  rw [← Finset.card_univ]
  apply Finset.card_lt_card
  rw [Finset.ssubset_iff_of_subset (subset_univ _)]
  exact ⟨v, mem_univ v, by simp [precedes_irrefl]⟩

omit [Field 𝕜] [IsStrictOrderedRing 𝕜] in
lemma image_posRank (p : W → 𝕜) : univ.image (posRank p) = range (Fintype.card W) := by
  apply Finset.eq_of_subset_of_card_le
  · intro r hr
    simp only [Finset.mem_image, mem_univ, true_and] at hr
    obtain ⟨v, rfl⟩ := hr
    exact Finset.mem_range.mpr (posRank_lt p v)
  · rw [Finset.card_image_of_injective _ (posRank_injective p), Finset.card_range,
      Finset.card_univ]

/-- Top-k with positional tie-breaking. -/
def topKPosSet (k : ℕ) (p : W → 𝕜) : Finset W := univ.filter (fun v => posRank p v < k)

/-- Nucleus with positional tie-breaking: keep the shortest prefix of the order whose
mass reaches `P`. -/
def topPPosSet (P : 𝕜) (p : W → 𝕜) : Finset W :=
  univ.filter (fun v => ∑ u ∈ posAbove p v, p u < P)

omit [Field 𝕜] [IsStrictOrderedRing 𝕜] in
/-- **Exactness.** Positional top-k keeps exactly `min k |V|` tokens. -/
theorem card_topKPosSet (k : ℕ) (p : W → 𝕜) :
    (topKPosSet k p).card = min k (Fintype.card W) := by
  rw [← Finset.card_image_of_injective _ (posRank_injective p)]
  have : (topKPosSet k p).image (posRank p) = (range (Fintype.card W)).filter (· < k) := by
    rw [← image_posRank p, Finset.filter_image]
    rfl
  rw [this]
  have : (range (Fintype.card W)).filter (· < k) = range (min k (Fintype.card W)) := by
    ext r; simp [lt_min_iff, and_comm]
  rw [this, Finset.card_range]

omit [Field 𝕜] [IsStrictOrderedRing 𝕜] in
/-- Without ties, positional top-k is top-k. -/
theorem topKPosSet_eq_of_injective {p : W → 𝕜} (hp : Function.Injective p) (k : ℕ) :
    topKPosSet k p = topKSet k p := by
  have : ∀ v, posAbove p v = above p v := by
    intro v; ext u
    simp only [mem_posAbove, mem_above, Precedes]
    constructor
    · rintro (h | ⟨h, h'⟩)
      · exact h
      · exact absurd (hp h) (ne_of_lt h')
    · exact Or.inl
  ext v; simp [topKPosSet, posRank, this]

omit [IsStrictOrderedRing 𝕜] in
/-- Without ties, positional nucleus is nucleus. -/
theorem topPPosSet_eq_of_injective {p : W → 𝕜} (hp : Function.Injective p) (P : 𝕜) :
    topPPosSet P p = topPSet P p := by
  have : ∀ v, posAbove p v = above p v := by
    intro v; ext u
    simp only [mem_posAbove, mem_above, Precedes]
    constructor
    · rintro (h | ⟨h, h'⟩)
      · exact h
      · exact absurd (hp h) (ne_of_lt h')
    · exact Or.inl
  ext v; simp [topPPosSet, this]

/-- **Positional top-k is not equivariant.** -/
theorem topKPosSet_not_equivariant {k : ℕ} (hk0 : 0 < k) (hkn : k < Fintype.card W) :
    ¬ Equivariant (topKPosSet (𝕜 := 𝕜) (W := W) k) := fun h =>
  no_equivariant_exact_topK hk0 hkn _ h (fun d => by rw [card_topKPosSet, min_eq_left hkn.le])

/-- The first token in the order is a most probable token and precedes everything. -/
lemma Dist.exists_first (d : Dist 𝕜 W) : ∃ a, 0 < d.prob a ∧ posAbove d.prob a = ∅ := by
  obtain ⟨a0, hpos, -⟩ := d.exists_argmax
  set S := univ.filter (fun v => ∀ u, d.prob u ≤ d.prob v) with hSdef
  obtain ⟨a1, -, ha1⟩ := d.exists_argmax
  have hS : S.Nonempty := ⟨a1, by simp [S, ha1]⟩
  have hm : S.min' hS ∈ S := S.min'_mem hS
  simp only [S, Finset.mem_filter, mem_univ, true_and] at hm
  refine ⟨S.min' hS, lt_of_lt_of_le hpos (hm a0), ?_⟩
  ext u
  simp only [mem_posAbove, Precedes, Finset.notMem_empty, iff_false, not_or, not_and]
  refine ⟨not_lt.mpr (hm u), fun hu => ?_⟩
  have huS : u ∈ S := by
    simp only [S, Finset.mem_filter, mem_univ, true_and]
    intro w; rw [hu]; exact hm w
  exact not_lt.mpr (S.min'_le u huS)

/-- Top-k with positional tie-breaking, `k ≥ 1`. -/
def Dist.topKPos (k : ℕ) (hk : 0 < k) (d : Dist 𝕜 W) : Dist 𝕜 W :=
  d.restrict (topKPosSet k d.prob) (by
    obtain ⟨a, hpos, ha⟩ := d.exists_first
    refine d.mass_pos_of_mem (Finset.mem_filter.mpr ⟨mem_univ a, ?_⟩) hpos
    show posRank d.prob a < k
    rw [posRank, ha, Finset.card_empty]
    exact hk)

/-- Nucleus with positional tie-breaking, `P > 0`. -/
def Dist.topPPos (P : 𝕜) (hP : 0 < P) (d : Dist 𝕜 W) : Dist 𝕜 W :=
  d.restrict (topPPosSet P d.prob) (by
    obtain ⟨a, hpos, ha⟩ := d.exists_first
    exact d.mass_pos_of_mem (by simp [topPPosSet, ha, hP]) hpos)

end Positional

/-! ### Inverse-CDF sampling -/

section InvCdf

variable {n : ℕ}

/-- The cumulative distribution: the mass of the first `i` tokens. -/
def cdf (p : Fin n → 𝕜) (i : ℕ) : 𝕜 := ∑ j ∈ univ.filter (fun j : Fin n => j.val < i), p j

omit [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜] in
lemma cdf_zero (p : Fin n → 𝕜) : cdf p 0 = 0 := by simp [cdf]

omit [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜] in
lemma cdf_succ (p : Fin n → 𝕜) (i : Fin n) : cdf p (i.val + 1) = cdf p i.val + p i := by
  unfold cdf
  have : univ.filter (fun j : Fin n => j.val < i.val + 1) =
      insert i (univ.filter (fun j : Fin n => j.val < i.val)) := by
    ext j
    simp only [Finset.mem_filter, mem_univ, true_and, Finset.mem_insert]
    constructor
    · intro hj
      rcases Nat.lt_succ_iff_lt_or_eq.mp hj with h | h
      · exact Or.inr h
      · exact Or.inl (Fin.ext h)
    · rintro (rfl | h)
      · exact Nat.lt_succ_self _
      · exact Nat.lt_succ_of_lt h
  rw [this, Finset.sum_insert (by simp), add_comm]

lemma cdf_mono (p : Fin n → 𝕜) (hp : ∀ j, 0 ≤ p j) {i i' : ℕ} (h : i ≤ i') :
    cdf p i ≤ cdf p i' := by
  unfold cdf
  apply Finset.sum_le_sum_of_subset_of_nonneg
  · intro j hj
    simp only [Finset.mem_filter, mem_univ, true_and] at *
    omega
  · intro j _ _; exact hp j

lemma cdf_n (d : Dist 𝕜 (Fin n)) : cdf d.prob n = 1 := by
  unfold cdf
  rw [Finset.filter_true_of_mem (fun j _ => j.isLt)]
  exact d.sum_eq_one

/-- Inverse-CDF sampling: map `u ∈ [0, 1)` to the first token whose cumulative mass
exceeds `u`. -/
def invCdf (d : Dist 𝕜 (Fin n)) (u : 𝕜) (hu : u < 1) : Fin n :=
  have h : ∃ i, i < n ∧ u < cdf d.prob (i + 1) := by
    have hn : 0 < n := by
      rcases Nat.eq_zero_or_pos n with h0 | h0
      · subst h0
        exact absurd (cdf_n d) (by simp [cdf])
      · exact h0
    refine ⟨n - 1, Nat.sub_lt hn one_pos, ?_⟩
    rw [show n - 1 + 1 = n by omega, cdf_n d]
    exact hu
  ⟨Nat.find h, (Nat.find_spec h).1⟩

/-- **Inverse-CDF sampling is exact.** For `0 ≤ u < 1`, `invCdf d u = i` exactly when
`u` lies in `[cdf i, cdf (i+1))`, an interval of length `d.prob i`. A uniform `u`
therefore selects token `i` with probability `d.prob i`. -/
theorem invCdf_eq_iff (d : Dist 𝕜 (Fin n)) {u : 𝕜} (hu0 : 0 ≤ u) (hu : u < 1) (i : Fin n) :
    invCdf d u hu = i ↔ cdf d.prob i.val ≤ u ∧ u < cdf d.prob (i.val + 1) := by
  unfold invCdf
  simp only [Fin.ext_iff]
  rw [Nat.find_eq_iff]
  constructor
  · rintro ⟨⟨-, hlt⟩, hmin⟩
    refine ⟨?_, hlt⟩
    rcases Nat.eq_zero_or_pos i.val with h0 | h0
    · rw [h0, cdf_zero]; exact hu0
    · have := hmin (i.val - 1) (Nat.sub_lt h0 one_pos)
      push Not at this
      have h' := this (lt_of_le_of_lt (Nat.sub_le _ _) i.isLt)
      rwa [show (i : ℕ) - 1 + 1 = i by omega] at h'
  · rintro ⟨hle, hlt⟩
    refine ⟨⟨i.isLt, hlt⟩, ?_⟩
    intro j hj
    push Not
    intro _
    exact le_trans (cdf_mono d.prob d.nonneg (by omega)) hle

/-- The interval selecting token `i` has length `d.prob i`. -/
theorem cdf_interval_length (d : Dist 𝕜 (Fin n)) (i : Fin n) :
    cdf d.prob (i.val + 1) - cdf d.prob i.val = d.prob i := by
  rw [cdf_succ]; ring

end InvCdf

end Field

end LLM
