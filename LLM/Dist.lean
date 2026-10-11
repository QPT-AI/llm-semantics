import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.GroupWithZero.Finset
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Fintype.Pi
import Mathlib.CategoryTheory.Category.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# Finite distributions and the category of stochastic maps

Every object in this repository is a finite probability distribution or a map that
produces one. A distribution is stated over an arbitrary linearly ordered field `𝕜`.
The executable oracle instantiates `𝕜 := ℚ`, so its arithmetic is exact, and the
analytic statements about the exponential instantiate `𝕜 := ℝ`.

The module provides

* `Dist 𝕜 α`, a probability distribution on a finite type `α`;
* `Dist.pure`, `Dist.bind`, the finite distribution monad;
* `Dist.restrict`, conditioning on a set of positive mass (the operation every
  sampling filter and every logit mask performs);
* `Dist.pi`, independent product;
* `Dist.tv`, total variation distance;
* `FinStoch 𝕜`, finite types and stochastic maps, as a Mathlib `Category`.
-/

open Finset

namespace LLM

universe u v w

/-- Division distributes over a finite sum. -/
lemma sum_div_const {ι 𝕜 : Type*} [DivisionSemiring 𝕜] (s : Finset ι) (f : ι → 𝕜) (a : 𝕜) :
    (∑ i ∈ s, f i) / a = ∑ i ∈ s, f i / a := by
  simp only [div_eq_mul_inv, Finset.sum_mul]

/-- A probability distribution on a finite type, with probabilities in an ordered field. -/
@[ext]
structure Dist (𝕜 : Type u) [Field 𝕜] [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜]
    (α : Type v) [Fintype α] where
  /-- The probability of each outcome. -/
  prob : α → 𝕜
  /-- Probabilities are nonnegative. -/
  nonneg : ∀ a, 0 ≤ prob a
  /-- Probabilities sum to one. -/
  sum_eq_one : ∑ a, prob a = 1

namespace Dist

variable {𝕜 : Type u} [Field 𝕜] [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜]
variable {α : Type v} {β : Type w} [Fintype α] [Fintype β]

/-! ### Mass and support -/

/-- The probability of a set of outcomes. -/
def mass (d : Dist 𝕜 α) (S : Finset α) : 𝕜 := ∑ a ∈ S, d.prob a

lemma mass_nonneg (d : Dist 𝕜 α) (S : Finset α) : 0 ≤ d.mass S :=
  Finset.sum_nonneg (fun a _ => d.nonneg a)

@[simp] lemma mass_univ (d : Dist 𝕜 α) : d.mass univ = 1 := d.sum_eq_one

lemma mass_mono (d : Dist 𝕜 α) {S T : Finset α} (h : S ⊆ T) : d.mass S ≤ d.mass T :=
  Finset.sum_le_sum_of_subset_of_nonneg h (fun a _ _ => d.nonneg a)

lemma mass_le_one (d : Dist 𝕜 α) (S : Finset α) : d.mass S ≤ 1 := by
  rw [← d.mass_univ]; exact d.mass_mono (subset_univ S)

lemma prob_le_one (d : Dist 𝕜 α) (a : α) : d.prob a ≤ 1 := by
  have h := d.mass_le_one {a}
  simpa [mass] using h

lemma prob_le_mass (d : Dist 𝕜 α) {S : Finset α} {a : α} (ha : a ∈ S) :
    d.prob a ≤ d.mass S :=
  Finset.single_le_sum (fun b _ => d.nonneg b) ha

lemma mass_pos_of_mem (d : Dist 𝕜 α) {S : Finset α} {a : α} (ha : a ∈ S)
    (hpos : 0 < d.prob a) : 0 < d.mass S :=
  lt_of_lt_of_le hpos (d.prob_le_mass ha)

/-- Some outcome has positive probability. -/
lemma exists_pos (d : Dist 𝕜 α) : ∃ a, 0 < d.prob a := by
  by_contra h
  push Not at h
  have hle : ∑ a, d.prob a ≤ 0 := Finset.sum_nonpos (fun a _ => h a)
  rw [d.sum_eq_one] at hle
  exact absurd hle (by norm_num)

/-- A most probable outcome exists, and it has positive probability. -/
lemma exists_argmax (d : Dist 𝕜 α) : ∃ a, 0 < d.prob a ∧ ∀ b, d.prob b ≤ d.prob a := by
  obtain ⟨a0, ha0⟩ := d.exists_pos
  obtain ⟨a, -, ha⟩ := Finset.exists_max_image univ d.prob ⟨a0, mem_univ a0⟩
  exact ⟨a, lt_of_lt_of_le ha0 (ha a0 (mem_univ _)), fun b => ha b (mem_univ b)⟩

lemma nonempty (d : Dist 𝕜 α) : Nonempty α := ⟨d.exists_pos.choose⟩

/-- The mass of a set equals the mass of its positive-probability part. -/
lemma mass_eq_mass_filter_pos (d : Dist 𝕜 α) (S : Finset α) :
    d.mass S = d.mass (S.filter (fun a => 0 < d.prob a)) := by
  unfold mass
  rw [Finset.sum_filter]
  refine Finset.sum_congr rfl (fun a _ => ?_)
  rcases (d.nonneg a).lt_or_eq with h | h
  · simp [h]
  · simp [← h]

/-! ### The finite distribution monad -/

section Monad

variable [DecidableEq α]

/-- The point mass. -/
def pure (a : α) : Dist 𝕜 α where
  prob b := if b = a then 1 else 0
  nonneg b := by split_ifs <;> norm_num
  sum_eq_one := by simp

@[simp] lemma pure_prob (a b : α) : (pure a : Dist 𝕜 α).prob b = if b = a then 1 else 0 := rfl

end Monad

/-- Sequential composition: draw `a` from `d`, then draw from `f a`. -/
def bind (d : Dist 𝕜 α) (f : α → Dist 𝕜 β) : Dist 𝕜 β where
  prob b := ∑ a, d.prob a * (f a).prob b
  nonneg b := Finset.sum_nonneg (fun a _ => mul_nonneg (d.nonneg a) ((f a).nonneg b))
  sum_eq_one := by
    rw [Finset.sum_comm]
    simp_rw [← Finset.mul_sum, Dist.sum_eq_one, mul_one]
    exact d.sum_eq_one

@[simp] lemma bind_prob (d : Dist 𝕜 α) (f : α → Dist 𝕜 β) (b : β) :
    (d.bind f).prob b = ∑ a, d.prob a * (f a).prob b := rfl

lemma pure_bind [DecidableEq α] (a : α) (f : α → Dist 𝕜 β) : (pure a).bind f = f a := by
  ext b; simp [ite_mul]

lemma bind_pure [DecidableEq α] (d : Dist 𝕜 α) : d.bind pure = d := by
  ext b; simp [mul_ite]

lemma bind_assoc {γ : Type*} [Fintype γ] (d : Dist 𝕜 α) (f : α → Dist 𝕜 β)
    (g : β → Dist 𝕜 γ) : (d.bind f).bind g = d.bind (fun a => (f a).bind g) := by
  ext c
  simp only [bind_prob, Finset.sum_mul, Finset.mul_sum, mul_assoc]
  exact Finset.sum_comm

/-- Pushforward along a function. -/
def map [DecidableEq β] (f : α → β) (d : Dist 𝕜 α) : Dist 𝕜 β := d.bind (fun a => pure (f a))

/-! ### Conditioning on a set -/

section Restrict

variable [DecidableEq α]

/-- Conditioning on a set of positive mass: zero the outcomes outside `S` and
renormalize. Every logit mask, every sampling filter, and every grammar constraint
acts on a distribution through this operation. -/
def restrict (d : Dist 𝕜 α) (S : Finset α) (h : 0 < d.mass S) : Dist 𝕜 α where
  prob a := if a ∈ S then d.prob a / d.mass S else 0
  nonneg a := by
    split_ifs
    · exact div_nonneg (d.nonneg a) h.le
    · exact le_rfl
  sum_eq_one := by
    rw [Finset.sum_ite_mem, Finset.univ_inter, ← sum_div_const]
    exact div_self (ne_of_gt h)

lemma restrict_prob (d : Dist 𝕜 α) (S : Finset α) (h : 0 < d.mass S) (a : α) :
    (d.restrict S h).prob a = if a ∈ S then d.prob a / d.mass S else 0 := rfl

lemma restrict_prob_of_mem (d : Dist 𝕜 α) {S : Finset α} (h : 0 < d.mass S) {a : α}
    (ha : a ∈ S) : (d.restrict S h).prob a = d.prob a / d.mass S := ite_eq_left ha

lemma restrict_prob_of_not_mem (d : Dist 𝕜 α) {S : Finset α} (h : 0 < d.mass S) {a : α}
    (ha : a ∉ S) : (d.restrict S h).prob a = 0 := ite_eq_right ha

lemma mass_restrict (d : Dist 𝕜 α) (S T : Finset α) (h : 0 < d.mass S) :
    (d.restrict S h).mass T = d.mass (S ∩ T) / d.mass S := by
  unfold mass
  simp only [restrict_prob]
  rw [Finset.sum_ite_mem, Finset.inter_comm, ← sum_div_const]
  rfl

/-- Conditioning twice is conditioning on the intersection. -/
lemma restrict_restrict (d : Dist 𝕜 α) (S T : Finset α) (hS : 0 < d.mass S)
    (hT : 0 < (d.restrict S hS).mass T) (hST : 0 < d.mass (S ∩ T)) :
    (d.restrict S hS).restrict T hT = d.restrict (S ∩ T) hST := by
  ext a
  have hm := d.mass_restrict S T hS
  simp only [restrict_prob, Finset.mem_inter, hm]
  by_cases haT : a ∈ T <;> by_cases haS : a ∈ S <;> simp [haT, haS]
  field_simp

/-- Conditioning on the whole space does nothing. -/
lemma restrict_univ (d : Dist 𝕜 α) (h : 0 < d.mass univ) : d.restrict univ h = d := by
  ext a; simp [restrict_prob]

/-- Conditioning depends only on which positive-probability outcomes are kept. -/
lemma restrict_congr (d : Dist 𝕜 α) {S S' : Finset α} (hS : 0 < d.mass S)
    (hS' : 0 < d.mass S') (h : ∀ a, 0 < d.prob a → (a ∈ S ↔ a ∈ S')) :
    d.restrict S hS = d.restrict S' hS' := by
  have hm : d.mass S = d.mass S' := by
    rw [d.mass_eq_mass_filter_pos S, d.mass_eq_mass_filter_pos S']
    congr 1
    ext a
    simp only [Finset.mem_filter]
    constructor
    · rintro ⟨ha, hp⟩; exact ⟨(h a hp).1 ha, hp⟩
    · rintro ⟨ha, hp⟩; exact ⟨(h a hp).2 ha, hp⟩
  ext a
  simp only [restrict_prob, hm]
  rcases (d.nonneg a).lt_or_eq with hp | hp
  · simp [h a hp]
  · simp [← hp]

/-- If every kept outcome of `S` lies in `T`, conditioning on `S` then on `T` is
conditioning on `S`. -/
lemma restrict_restrict_of_subset (d : Dist 𝕜 α) {S T : Finset α}
    (hS : 0 < d.mass S) (hT : 0 < (d.restrict S hS).mass T) (hsub : S ⊆ T) :
    (d.restrict S hS).restrict T hT = d.restrict S hS := by
  have hST : S ∩ T = S := Finset.inter_eq_left.mpr hsub
  have hpos : 0 < d.mass (S ∩ T) := by rw [hST]; exact hS
  rw [d.restrict_restrict S T hS hT hpos]
  exact d.restrict_congr hpos hS (fun a _ => by rw [hST])

end Restrict

/-! ### Normalizing nonnegative weights -/

/-- Normalize a nonnegative weight vector of positive total. -/
def normalize (w : α → 𝕜) (hw : ∀ a, 0 ≤ w a) (hpos : 0 < ∑ a, w a) : Dist 𝕜 α where
  prob a := w a / ∑ b, w b
  nonneg a := div_nonneg (hw a) hpos.le
  sum_eq_one := by rw [← sum_div_const]; exact div_self (ne_of_gt hpos)

@[simp] lemma normalize_prob (w : α → 𝕜) (hw : ∀ a, 0 ≤ w a) (hpos : 0 < ∑ a, w a) (a : α) :
    (normalize w hw hpos).prob a = w a / ∑ b, w b := rfl

/-- Normalization ignores a positive common factor. -/
lemma normalize_smul (w : α → 𝕜) (hw : ∀ a, 0 ≤ w a) (hpos : 0 < ∑ a, w a) (c : 𝕜)
    (hc : 0 < c) (hw' : ∀ a, 0 ≤ c * w a) (hpos' : 0 < ∑ a, c * w a) :
    normalize (fun a => c * w a) hw' hpos' = normalize w hw hpos := by
  ext a
  simp only [normalize_prob, ← Finset.mul_sum]
  have : (∑ b, w b) ≠ 0 := ne_of_gt hpos
  field_simp

/-- A distribution is its own normalization. -/
lemma normalize_prob_self (d : Dist 𝕜 α) (h : 0 < ∑ a, d.prob a) :
    normalize d.prob d.nonneg h = d := by
  ext a; simp [d.sum_eq_one]

/-! ### Independent products -/

/-- The independent product of a family of distributions. -/
def pi {ι : Type*} [Fintype ι] [DecidableEq ι] {γ : ι → Type*} [∀ i, Fintype (γ i)]
    (d : ∀ i, Dist 𝕜 (γ i)) : Dist 𝕜 (∀ i, γ i) where
  prob x := ∏ i, (d i).prob (x i)
  nonneg x := Finset.prod_nonneg (fun i _ => (d i).nonneg (x i))
  sum_eq_one := by
    classical
    have h := Finset.prod_univ_sum (fun i => (univ : Finset (γ i))) (fun i a => (d i).prob a)
    rw [Fintype.piFinset_univ] at h
    rw [← h]
    simp [Dist.sum_eq_one]

@[simp] lemma pi_prob {ι : Type*} [Fintype ι] [DecidableEq ι] {γ : ι → Type*}
    [∀ i, Fintype (γ i)] (d : ∀ i, Dist 𝕜 (γ i)) (x : ∀ i, γ i) :
    (pi d).prob x = ∏ i, (d i).prob (x i) := rfl

/-! ### Total variation -/

/-- Total variation distance, in the form `∑ₐ max(0, p a - q a)`. -/
def tv (p q : Dist 𝕜 α) : 𝕜 := ∑ a, max 0 (p.prob a - q.prob a)

lemma tv_nonneg (p q : Dist 𝕜 α) : 0 ≤ tv p q :=
  Finset.sum_nonneg (fun _ _ => le_max_left _ _)

/-- The overlap `∑ min(p, q)` is `1 - tv p q`. -/
lemma sum_min_eq (p q : Dist 𝕜 α) : ∑ a, min (p.prob a) (q.prob a) = 1 - tv p q := by
  have h : ∀ a, min (p.prob a) (q.prob a) = p.prob a - max 0 (p.prob a - q.prob a) := by
    intro a
    rcases le_total (p.prob a) (q.prob a) with hle | hle
    · rw [min_eq_left hle, max_eq_left (by linarith)]; ring
    · rw [min_eq_right hle, max_eq_right (by linarith)]; ring
  simp_rw [h]
  rw [Finset.sum_sub_distrib, p.sum_eq_one]
  rfl

/-- Total variation is symmetric. -/
lemma tv_comm (p q : Dist 𝕜 α) : tv p q = tv q p := by
  have h1 := sum_min_eq p q
  have h2 := sum_min_eq q p
  simp_rw [min_comm (q.prob _)] at h2
  linarith

/-- The two usual forms of total variation agree: `tv p q = ½ ∑ |p - q|`. -/
lemma tv_eq_half_l1 (p q : Dist 𝕜 α) : tv p q = (∑ a, |p.prob a - q.prob a|) / 2 := by
  have h : ∀ a, |p.prob a - q.prob a| =
      max 0 (p.prob a - q.prob a) + max 0 (q.prob a - p.prob a) := by
    intro a
    rcases le_total (p.prob a) (q.prob a) with hle | hle
    · rw [abs_of_nonpos (by linarith), max_eq_left (by linarith), max_eq_right (by linarith)]
      ring
    · rw [abs_of_nonneg (by linarith), max_eq_right (by linarith), max_eq_left (by linarith)]
      ring
  have h2 : ∑ a, |p.prob a - q.prob a| = tv p q + tv q p := by
    simp_rw [h]; rw [Finset.sum_add_distrib]; rfl
  rw [h2, tv_comm q p]
  ring

lemma tv_self (p : Dist 𝕜 α) : tv p p = 0 := by simp [tv]

lemma tv_eq_zero_iff (p q : Dist 𝕜 α) : tv p q = 0 ↔ p = q := by
  constructor
  · intro h
    rw [tv_eq_half_l1] at h
    have hsum : ∑ a, |p.prob a - q.prob a| = 0 := by linarith
    have hz := (Finset.sum_eq_zero_iff_of_nonneg (fun a _ => abs_nonneg _)).mp hsum
    ext a
    have := hz a (mem_univ a)
    rw [abs_eq_zero, sub_eq_zero] at this
    exact this
  · rintro rfl; exact tv_self p

end Dist

/-! ### The category of finite stochastic maps -/

/-- Objects of `FinStoch 𝕜`: finite types with decidable equality. Morphisms are
stochastic maps `α → Dist 𝕜 β`; composition is `Dist.bind`. -/
structure FinStoch (𝕜 : Type u) [Field 𝕜] [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜] :
    Type (v + 1) where
  of ::
  /-- The underlying finite type. -/
  carrier : Type v
  [instFintype : Fintype carrier]
  [instDecEq : DecidableEq carrier]

namespace FinStoch

attribute [instance] FinStoch.instFintype FinStoch.instDecEq

variable {𝕜 : Type u} [Field 𝕜] [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜]

instance : CoeSort (FinStoch.{u, v} 𝕜) (Type v) := ⟨FinStoch.carrier⟩

open CategoryTheory

instance category : Category.{max u v} (FinStoch.{u, v} 𝕜) where
  Hom X Y := X → Dist 𝕜 Y
  id _ := fun a => Dist.pure a
  comp f g := fun a => (f a).bind g
  id_comp f := by funext a; exact Dist.pure_bind _ _
  comp_id f := by funext a; exact Dist.bind_pure _
  assoc f g h := by funext a; exact Dist.bind_assoc _ _ _

lemma comp_apply {X Y Z : FinStoch.{u, v} 𝕜} (f : X ⟶ Y) (g : Y ⟶ Z) (a : X) :
    (f ≫ g) a = (f a).bind g := rfl

lemma id_apply (X : FinStoch.{u, v} 𝕜) (a : X) : (𝟙 X : X ⟶ X) a = Dist.pure a := rfl

/-- Deterministic maps embed into `FinStoch`. -/
def det {X Y : FinStoch.{u, v} 𝕜} (f : X → Y) : X ⟶ Y := fun a => Dist.pure (f a)

lemma det_id (X : FinStoch.{u, v} 𝕜) : det (id : X → X) = 𝟙 X := rfl

lemma det_comp {X Y Z : FinStoch.{u, v} 𝕜} (f : X → Y) (g : Y → Z) :
    det (g ∘ f) = det f ≫ det g := by
  funext a
  show Dist.pure (g (f a)) = (Dist.pure (f a)).bind (fun b => Dist.pure (g b))
  rw [Dist.pure_bind]

end FinStoch

end LLM
