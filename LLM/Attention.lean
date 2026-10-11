import LLM.Softmax
import Mathlib.Algebra.Module.LinearMap.Defs
import Mathlib.Algebra.Module.BigOperators
import Mathlib.Algebra.BigOperators.GroupWithZero.Action
import Mathlib.Algebra.BigOperators.Ring.List
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.FreeMonoid.Basic
import Mathlib.Tactic.Abel
import Mathlib.Data.List.Induction

/-!
# Attention

The meaning of one attention read is a softmax-weighted average of value vectors.
Everything else (masks, caches, blocked kernels) is an implementation that has to
equal this average.

**Set-level attention** (indexed by any finite type of positions):

* `attend_convex`: the output is a convex combination of the values, so it obeys
  every linear bound the values obey; `attend_const`: attention cannot abstain;
  if every value is `m`, the output is `m` whatever the scores are;
* `selfAttn_equivariant`: self-attention commutes with permuting positions, and so
  does any stack built from it with pointwise maps and residual sums;
* **negative result** `no_equivariant_first`: no permutation-equivariant map, and so
  no transformer without positional information, can output the first token.

**Sequence-level causal attention** (lists):

* `causalAttn_causal`: an output never depends on later inputs;
* `kvDecode_eq`: **KV caching is exact**, meaning incremental decoding that stores
  only past keys and values equals full recomputation;
* `causal_extend`: for any causal stack, extending the input by one token leaves all
  previous outputs unchanged.

**The online-softmax monoid** (FlashAttention, split-K decoding, ring attention):

* `Online E M` is a commutative monoid, a Mathlib `CommMonoid`, of running
  `(max, normalizer, accumulator)` states;
* `finalize_ofList`: folding a key list through the monoid and finalizing gives
  exactly the attention output;
* `finalize_blocks` and `ofList_perm`: any blocking of the keys into tiles, and any
  order of the tiles, gives the same result;
* `ofList_max`: the running maximum is the maximum score, so every exponent the
  kernel evaluates is `≤ 0` and every exponential is `≤ 1`.
-/

open Finset

namespace LLM

universe u v w

variable {G : Type v} {𝕜 : Type u} [AddCommGroup G] [LinearOrder G]
variable [Field 𝕜] [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜]
variable (E : ExpChar G 𝕜)
variable {M : Type*} [AddCommGroup M] [Module 𝕜 M]

/-! ### One attention read -/

section Attend

variable {ι : Type w} [Fintype ι] [Nonempty ι]

/-- Softmax attention: the scores `s` weight the values `v`. -/
def attend (s : ι → G) (v : ι → M) : M := ∑ j, (softmax E s).prob j • v j

/-- Adding a constant to every score does not change the output. -/
theorem attend_shift (s : ι → G) (c : G) (v : ι → M) :
    attend E (fun j => s j + c) v = attend E s v := by
  unfold attend; rw [softmax_add_const]

/-- **Attention cannot abstain.** If every value is `m`, the output is `m`. -/
theorem attend_const (s : ι → G) (v : ι → M) (m : M) (h : ∀ j, v j = m) :
    attend E s v = m := by
  unfold attend
  simp_rw [h, ← Finset.sum_smul, Dist.sum_eq_one, one_smul]

/-- **Convexity.** The output obeys every linear upper bound the values obey. -/
theorem attend_convex (s : ι → G) (v : ι → M) (φ : M →ₗ[𝕜] 𝕜) (c : 𝕜)
    (h : ∀ j, φ (v j) ≤ c) : φ (attend E s v) ≤ c := by
  unfold attend
  rw [map_sum]
  simp_rw [map_smul, smul_eq_mul]
  calc ∑ j, (softmax E s).prob j * φ (v j) ≤ ∑ j, (softmax E s).prob j * c :=
        Finset.sum_le_sum (fun j _ => mul_le_mul_of_nonneg_left (h j) ((softmax E s).nonneg j))
    _ = c := by rw [← Finset.sum_mul, Dist.sum_eq_one, one_mul]

/-- **Convexity**, lower bounds. -/
theorem attend_convex_ge (s : ι → G) (v : ι → M) (φ : M →ₗ[𝕜] 𝕜) (c : 𝕜)
    (h : ∀ j, c ≤ φ (v j)) : c ≤ φ (attend E s v) := by
  unfold attend
  rw [map_sum]
  simp_rw [map_smul, smul_eq_mul]
  calc c = ∑ j, (softmax E s).prob j * c := by rw [← Finset.sum_mul, Dist.sum_eq_one, one_mul]
    _ ≤ ∑ j, (softmax E s).prob j * φ (v j) :=
        Finset.sum_le_sum (fun j _ => mul_le_mul_of_nonneg_left (h j) ((softmax E s).nonneg j))

lemma softmax_comp_perm (s : ι → G) (σ : Equiv.Perm ι) (j : ι) :
    (softmax E (s ∘ σ)).prob j = (softmax E s).prob (σ j) := by
  simp only [softmax_prob, Function.comp_apply]
  congr 1
  exact Equiv.sum_comp σ (fun i => E (s i))

/-- Reindexing the keys of one read does not change it. -/
lemma attend_perm (s : ι → G) (v : ι → M) (σ : Equiv.Perm ι) :
    attend E (s ∘ σ) (v ∘ σ) = attend E s v := by
  unfold attend
  simp_rw [softmax_comp_perm, Function.comp_apply]
  exact Equiv.sum_comp σ (fun i => (softmax E s).prob i • v i)

end Attend

/-! ### Permutation equivariance -/

section Equivariance

variable {ι : Type w} {X : Type*} {Y : Type*} {Z : Type*}

/-- A sequence map is equivariant when permuting its input positions permutes its
output positions in the same way. -/
def SeqEquivariant (f : (ι → X) → (ι → Y)) : Prop :=
  ∀ (σ : Equiv.Perm ι) (x : ι → X), f (x ∘ σ) = f x ∘ σ

theorem pointwise_equivariant (g : X → Y) : SeqEquivariant (ι := ι) (fun x => g ∘ x) :=
  fun _ _ => rfl

theorem equivariant_comp {f : (ι → X) → (ι → Y)} {g : (ι → Y) → (ι → Z)}
    (hf : SeqEquivariant f) (hg : SeqEquivariant g) : SeqEquivariant (g ∘ f) := by
  intro σ x
  simp only [Function.comp_apply, hf σ x, hg σ (f x)]

/-- Sums of equivariant maps (multi-head attention, residual connections) are
equivariant. -/
theorem equivariant_add [AddCommMonoid Y] {f g : (ι → X) → (ι → Y)}
    (hf : SeqEquivariant f) (hg : SeqEquivariant g) : SeqEquivariant (fun x => f x + g x) := by
  intro σ x
  funext i
  simp only [Function.comp_apply, Pi.add_apply]
  rw [hf σ x, hg σ x]
  rfl

theorem equivariant_id : SeqEquivariant (ι := ι) (fun x : ι → X => x) := fun _ _ => rfl

lemma list_sum_equivariant [AddCommMonoid Y] (fs : List ((ι → X) → (ι → Y)))
    (h : ∀ f ∈ fs, SeqEquivariant f) :
    SeqEquivariant (fun x => (fs.map (fun f => f x)).sum) := by
  induction fs with
  | nil => intro σ x; rfl
  | cons f fs ih =>
    simp only [List.map_cons, List.sum_cons]
    exact equivariant_add (h f (by simp)) (ih (fun g hg => h g (by simp [hg])))

/-- Stacks of equivariant layers are equivariant. -/
theorem stack_equivariant (layers : List ((ι → X) → (ι → X)))
    (h : ∀ f ∈ layers, SeqEquivariant f) :
    SeqEquivariant (fun x => layers.foldl (fun acc f => f acc) x) := by
  induction layers using List.reverseRecOn with
  | nil => intro σ x; rfl
  | append_singleton fs f ih =>
    simp only [List.foldl_append, List.foldl_cons, List.foldl_nil]
    exact equivariant_comp (ih (fun g hg => h g (by simp [hg]))) (h f (by simp))

/-- **Positional information is necessary.** If there are two positions and two
distinguishable tokens, no permutation-equivariant map, and so no stack of
`block`s, outputs the token at a fixed position everywhere. -/
theorem no_equivariant_first {i₀ i₁ : ι} (hi : i₀ ≠ i₁) {a b : X} (hab : a ≠ b) :
    ¬ ∃ f : (ι → X) → (ι → X), SeqEquivariant f ∧ ∀ x i, f x i = x i₀ := by
  classical
  rintro ⟨f, hf, hfirst⟩
  let x : ι → X := fun i => if i = i₀ then a else b
  have h1 := congrFun (hf (Equiv.swap i₀ i₁) x) i₀
  simp only [Function.comp_apply, Equiv.swap_apply_left] at h1
  rw [hfirst, hfirst] at h1
  simp [x, Equiv.swap_apply_left, hi.symm] at h1
  exact hab h1.symm

end Equivariance

/-! ### Self-attention -/

section SelfAttention

variable {ι : Type w} [Fintype ι] [Nonempty ι]
variable {X : Type*}

/-- An attention head. The query and key projections are folded into `score`, so
any bilinear or learned score is covered. -/
structure Head (X : Type*) (G : Type v) (M : Type*) where
  /-- The score of a query token against a key token. -/
  score : X → X → G
  /-- The value of a token. -/
  value : X → M

/-- Self-attention with one head and no positional information. -/
def selfAttn (h : Head X G M) (x : ι → X) : ι → M :=
  fun i => attend E (fun j => h.score (x i) (x j)) (fun j => h.value (x j))

/-- **Self-attention is permutation equivariant.** -/
theorem selfAttn_equivariant (h : Head X G M) : SeqEquivariant (selfAttn E h (ι := ι)) := by
  intro σ x
  funext i
  simp only [selfAttn, Function.comp_apply]
  exact attend_perm E (fun j => h.score (x (σ i)) (x j)) (fun j => h.value (x j)) σ

/-- A pre-norm transformer block without positional encodings:
`y = x + attn(norm x)`, `out = y + mlp(norm y)`, with a sum of heads. -/
def block (heads : List (Head M G M)) (norm mlp : M → M) (x : ι → M) : ι → M :=
  let attnOut := fun x : ι → M => (heads.map (fun h => selfAttn E h (norm ∘ x))).sum
  let y := x + attnOut x
  y + mlp ∘ norm ∘ y

/-- **A transformer block without positional encodings is equivariant.** -/
theorem block_equivariant (heads : List (Head M G M)) (norm mlp : M → M) :
    SeqEquivariant (block E heads norm mlp (ι := ι)) := by
  have hattn : SeqEquivariant (fun x : ι → M =>
      (heads.map (fun h => selfAttn E h (norm ∘ x))).sum) := by
    have := list_sum_equivariant (heads.map (fun h => fun x : ι → M => selfAttn E h (norm ∘ x)))
      (by
        intro f hf
        simp only [List.mem_map] at hf
        obtain ⟨h, -, rfl⟩ := hf
        exact equivariant_comp (pointwise_equivariant norm) (selfAttn_equivariant E h))
    simpa [List.map_map, Function.comp_def] using this
  have hy : SeqEquivariant (fun x : ι → M =>
      x + (heads.map (fun h => selfAttn E h (norm ∘ x))).sum) :=
    equivariant_add equivariant_id hattn
  have hmlp : SeqEquivariant (fun x : ι → M =>
      mlp ∘ norm ∘ (x + (heads.map (fun h => selfAttn E h (norm ∘ x))).sum)) :=
    equivariant_comp hy (pointwise_equivariant (mlp ∘ norm))
  exact equivariant_add hy hmlp

end SelfAttention

/-! ### Causal attention and the KV cache -/

section Causal

variable {X Q K : Type*}

/-- One attention read over a list of cached `(key, value)` pairs. -/
def attendKV (score : Q → K → G) (q : Q) (kv : List (K × M)) : M :=
  ((kv.map (fun p => E (score q p.1))).sum)⁻¹ • (kv.map (fun p => E (score q p.1) • p.2)).sum

/-- On a nonempty list the cached read is the softmax read of `attend`. -/
theorem attendKV_ofFn (score : Q → K → G) (q : Q) {n : ℕ} (kv : Fin (n + 1) → K × M) :
    attendKV E score q (List.ofFn kv) = attend E (fun j => score q (kv j).1) (fun j => (kv j).2) := by
  unfold attendKV attend
  simp only [List.map_ofFn, List.sum_ofFn, Function.comp_def, softmax_prob, div_eq_inv_mul,
    mul_smul, ← Finset.smul_sum]

/-- A causal attention head: query, key and value projections. -/
structure CausalHead (X Q K M : Type*) where
  /-- Query projection. -/
  query : X → Q
  /-- Key projection. -/
  key : X → K
  /-- Value projection. -/
  value : X → M

/-- The cache entry a token contributes. -/
def CausalHead.kv (h : CausalHead X Q K M) (x : X) : K × M := (h.key x, h.value x)

/-- The meaning of causal self-attention: position `i` attends over positions
`0, …, i`. -/
def causalAttn (score : Q → K → G) (h : CausalHead X Q K M) (xs : List X) : List M :=
  xs.mapIdx (fun i x => attendKV E score (h.query x) ((xs.take (i + 1)).map h.kv))

/-- Incremental decoding with a KV cache: each step appends one `(key, value)` pair
and reads the cache. Nothing already in the cache is recomputed. -/
def kvDecode (score : Q → K → G) (h : CausalHead X Q K M) : List (K × M) → List X → List M
  | _, [] => []
  | cache, x :: xs =>
      attendKV E score (h.query x) (cache ++ [h.kv x]) :: kvDecode score h (cache ++ [h.kv x]) xs

lemma causalAttn_length (score : Q → K → G) (h : CausalHead X Q K M) (xs : List X) :
    (causalAttn E score h xs).length = xs.length := by
  simp [causalAttn]

lemma kvDecode_eq_drop (score : Q → K → G) (h : CausalHead X Q K M) (pre xs : List X) :
    kvDecode E score h (pre.map h.kv) xs = (causalAttn E score h (pre ++ xs)).drop pre.length := by
  induction xs generalizing pre with
  | nil => simp [kvDecode, causalAttn]
  | cons x xs ih =>
    simp only [kvDecode]
    have hc : pre.map h.kv ++ [h.kv x] = (pre ++ [x]).map h.kv := by simp
    rw [hc, ih (pre ++ [x])]
    have hlen : pre.length < (causalAttn E score h (pre ++ x :: xs)).length := by
      rw [causalAttn_length]; simp
    rw [List.drop_eq_getElem_cons hlen]
    have e1 : pre ++ [x] ++ xs = pre ++ x :: xs := by simp
    rw [e1]
    congr 1
    · simp only [causalAttn, List.getElem_mapIdx]
      have e2 : (pre ++ x :: xs).take (pre.length + 1) = pre ++ [x] := by
        rw [List.take_append]; simp
      simp [e2]
    · simp

/-- **KV caching is exact.** Incremental decoding equals recomputing causal attention
over the whole sequence. -/
theorem kvDecode_eq (score : Q → K → G) (h : CausalHead X Q K M) (xs : List X) :
    kvDecode E score h [] xs = causalAttn E score h xs := by
  simpa using kvDecode_eq_drop E score h [] xs

/-- A sequence map is causal when it preserves length and its output on a prefix is
the prefix of its output. -/
def Causal {X Y : Type*} (f : List X → List Y) : Prop :=
  (∀ xs, (f xs).length = xs.length) ∧ ∀ xs ys, (f (xs ++ ys)).take xs.length = f xs

/-- **Causal attention is causal.** -/
theorem causalAttn_causal (score : Q → K → G) (h : CausalHead X Q K M) :
    Causal (causalAttn E score h) := by
  refine ⟨causalAttn_length E score h, fun xs ys => ?_⟩
  apply List.ext_getElem
  · simp [causalAttn_length]
  · intro i h1 h2
    simp only [List.getElem_take, causalAttn, List.getElem_mapIdx]
    have hi : i < xs.length := by simpa [causalAttn_length] using h2
    rw [List.getElem_append_left hi, List.take_append_of_le_length (by omega)]

theorem causal_comp {X Y Z : Type*} {f : List X → List Y} {g : List Y → List Z}
    (hf : Causal f) (hg : Causal g) : Causal (g ∘ f) := by
  refine ⟨fun xs => by simp [hg.1, hf.1], fun xs ys => ?_⟩
  simp only [Function.comp_apply]
  have hsplit : f (xs ++ ys) = f xs ++ (f (xs ++ ys)).drop xs.length := by
    conv_lhs => rw [← List.take_append_drop xs.length (f (xs ++ ys))]
    rw [hf.2]
  rw [hsplit]
  have := hg.2 (f xs) ((f (xs ++ ys)).drop xs.length)
  rwa [hf.1] at this

theorem causal_map {X Y : Type*} (g : X → Y) : Causal (List.map g) :=
  ⟨fun xs => by simp, fun xs ys => by simp⟩

/-- Residual connections preserve causality. -/
theorem causal_residual {f : List M → List M} (hf : Causal f) :
    Causal (fun xs => List.zipWith (· + ·) xs (f xs)) := by
  refine ⟨fun xs => by simp [hf.1], fun xs ys => ?_⟩
  rw [List.take_zipWith]
  · have := hf.2 xs ys
    rw [List.take_append_of_le_length le_rfl, List.take_length, this]

/-- **Incremental decoding for any causal stack.** Extending the input by one token
leaves every earlier output unchanged, so a decoder only has to compute the last
position. -/
theorem causal_extend {X Y : Type*} {f : List X → List Y} (hf : Causal f) (xs : List X) (x : X) :
    f (xs ++ [x]) = f xs ++ (f (xs ++ [x])).drop xs.length := by
  conv_lhs => rw [← List.take_append_drop xs.length (f (xs ++ [x]))]
  rw [hf.2]

end Causal

/-! ### The online-softmax monoid -/

section Online

/-- A partial attention state over a nonempty block of keys: running maximum `m`,
normalizer `l = ∑ E(s - m)`, accumulator `o = ∑ E(s - m) • v`. -/
@[ext]
structure OState (G : Type v) (𝕜 : Type u) (M : Type*) where
  /-- Running maximum score. -/
  m : G
  /-- Running normalizer. -/
  l : 𝕜
  /-- Running weighted sum of values. -/
  o : M

/-- Merge two partial states, rescaling both to the larger maximum. -/
def mergeS (a b : OState G 𝕜 M) : OState G 𝕜 M where
  m := max a.m b.m
  l := a.l * E (a.m - max a.m b.m) + b.l * E (b.m - max a.m b.m)
  o := E (a.m - max a.m b.m) • a.o + E (b.m - max a.m b.m) • b.o

lemma mergeS_comm (a b : OState G 𝕜 M) : mergeS E a b = mergeS E b a := by
  ext
  · simp [mergeS, max_comm]
  · simp only [mergeS]; rw [max_comm b.m a.m]; ring
  · simp only [mergeS]; rw [max_comm b.m a.m]; abel

lemma mergeS_assoc (a b c : OState G 𝕜 M) :
    mergeS E (mergeS E a b) c = mergeS E a (mergeS E b c) := by
  have key : ∀ x y z : G, E (y - z) * E (x - y) = E (x - z) := fun x y z => by
    rw [mul_comm]; exact E.mul_sub_sub x y z
  ext
  · simp [mergeS, max_assoc]
  · simp only [mergeS, max_assoc, add_mul, mul_assoc, E.mul_sub_sub]
    ring
  · simp only [mergeS, max_assoc, smul_add, smul_smul, key]
    abel

/-- The online-softmax monoid for the exponential character `E`. The empty state is
the identity. -/
@[ext]
structure Online (E : ExpChar G 𝕜) (M : Type*) where
  /-- `none` is the state of an empty block. -/
  val : Option (OState G 𝕜 M)

namespace Online

variable {E}

/-- Merging states. -/
def mul : Online E M → Online E M → Online E M
  | ⟨none⟩, b => b
  | a, ⟨none⟩ => a
  | ⟨some x⟩, ⟨some y⟩ => ⟨some (mergeS E x y)⟩

instance : Mul (Online E M) := ⟨mul⟩
instance : One (Online E M) := ⟨⟨none⟩⟩

omit [AddCommGroup M] [Module 𝕜 M] in
@[simp] lemma one_val : (1 : Online E M).val = none := rfl
@[simp] lemma none_mul (b : Online E M) : (⟨none⟩ : Online E M) * b = b := by
  cases b; rfl
@[simp] lemma mul_none (a : Online E M) : a * ⟨none⟩ = a := by
  rcases a with ⟨_ | a⟩ <;> rfl
@[simp] lemma some_mul_some (x y : OState G 𝕜 M) :
    (⟨some x⟩ : Online E M) * ⟨some y⟩ = ⟨some (mergeS E x y)⟩ := rfl

instance : CommMonoid (Online E M) where
  mul_assoc a b c := by
    rcases a with ⟨_ | a⟩ <;> rcases b with ⟨_ | b⟩ <;> rcases c with ⟨_ | c⟩ <;>
      simp [mergeS_assoc]
  one_mul a := none_mul a
  mul_one a := mul_none a
  mul_comm a b := by
    rcases a with ⟨_ | a⟩ <;> rcases b with ⟨_ | b⟩ <;> simp [mergeS_comm]

variable (E)

/-- The state of a single key with score `s` and value `v`. -/
def single (s : G) (v : M) : Online E M := ⟨some ⟨s, 1, v⟩⟩

/-- Fold a list of `(score, value)` pairs through the monoid. -/
def ofList (xs : List (G × M)) : Online E M := (xs.map (fun p => single E p.1 p.2)).prod

/-- The fold is the free-monoid homomorphism induced by `single`. -/
theorem ofList_eq_lift (xs : List (G × M)) :
    ofList E xs = FreeMonoid.lift (fun p : G × M => single E p.1 p.2) (FreeMonoid.ofList xs) := by
  rw [FreeMonoid.lift_apply]; rfl

/-- Read the attention output off a state. -/
def finalize : Online E M → M
  | ⟨none⟩ => 0
  | ⟨some st⟩ => st.l⁻¹ • st.o

/-- Attention over a list of `(score, value)` pairs. -/
def attendList (xs : List (G × M)) : M :=
  ((xs.map (fun p => E p.1)).sum)⁻¹ • (xs.map (fun p => E p.1 • p.2)).sum

lemma attendKV_eq_attendList {Q K : Type*} (score : Q → K → G) (q : Q) (kv : List (K × M)) :
    attendKV E score q kv = attendList E (kv.map (fun p => (score q p.1, p.2))) := by
  simp [attendKV, attendList, List.map_map, Function.comp_def]

/-- `st` represents the key list `xs`. -/
def Rep (xs : List (G × M)) (st : OState G 𝕜 M) : Prop :=
  st.l = (xs.map (fun p => E (p.1 - st.m))).sum ∧
    st.o = (xs.map (fun p => E (p.1 - st.m) • p.2)).sum

lemma rep_single (s : G) (v : M) : Rep E [(s, v)] ⟨s, 1, v⟩ := by
  simp [Rep]

lemma rep_merge {xs ys : List (G × M)} {a b : OState G 𝕜 M} (ha : Rep E xs a) (hb : Rep E ys b) :
    Rep E (xs ++ ys) (mergeS E a b) := by
  obtain ⟨hal, hao⟩ := ha
  obtain ⟨hbl, hbo⟩ := hb
  have key : ∀ x y z : G, E (y - z) * E (x - y) = E (x - z) := fun x y z => by
    rw [mul_comm]; exact E.mul_sub_sub x y z
  constructor
  · simp only [mergeS, hal, hbl, List.map_append, List.sum_append, ← List.sum_map_mul_right,
      E.mul_sub_sub]
  · simp only [mergeS, hao, hbo, List.map_append, List.sum_append, List.smul_sum, List.map_map,
      Function.comp_def, smul_smul, key]

lemma ofList_some {xs : List (G × M)} (hxs : xs ≠ []) :
    ∃ st, ofList E xs = ⟨some st⟩ ∧ Rep E xs st := by
  induction xs with
  | nil => exact absurd rfl hxs
  | cons p xs ih =>
    by_cases hnil : xs = []
    · subst hnil
      exact ⟨⟨p.1, 1, p.2⟩, by simp [ofList, single], rep_single E p.1 p.2⟩
    · obtain ⟨st, hst, hrep⟩ := ih hnil
      refine ⟨mergeS E ⟨p.1, 1, p.2⟩ st, ?_, ?_⟩
      · simp only [ofList, List.map_cons, List.prod_cons] at hst ⊢
        rw [hst]; rfl
      · exact rep_merge E (rep_single E p.1 p.2) hrep

/-- **Online softmax computes attention.** -/
theorem finalize_ofList (xs : List (G × M)) : finalize E (ofList E xs) = attendList E xs := by
  by_cases hxs : xs = []
  · subst hxs; simp [ofList, finalize, attendList]
  obtain ⟨st, hst, hl, ho⟩ := ofList_some E hxs
  rw [hst]
  simp only [finalize, attendList, hl, ho]
  have hsub : ∀ p : G × M, E (p.1 - st.m) = E p.1 * (E st.m)⁻¹ := fun p => by
    rw [E.map_sub, div_eq_mul_inv]
  have hl' : (xs.map (fun p => E (p.1 - st.m))).sum =
      (xs.map (fun p => E p.1)).sum * (E st.m)⁻¹ := by
    simp_rw [hsub]; exact List.sum_map_mul_right (l := xs) (f := fun p => E p.1) (r := (E st.m)⁻¹)
  have ho' : (xs.map (fun p => E (p.1 - st.m) • p.2)).sum =
      (E st.m)⁻¹ • (xs.map (fun p => E p.1 • p.2)).sum := by
    rw [List.smul_sum, List.map_map]
    congr 1
    apply List.map_congr_left
    intro p _
    simp only [Function.comp_apply, hsub, mul_smul]
    exact smul_comm _ _ _
  rw [hl', ho', smul_smul, mul_inv, inv_inv, mul_assoc, mul_inv_cancel₀ (E.ne_zero st.m),
    mul_one]

/-- The fold respects concatenation. -/
theorem ofList_append (xs ys : List (G × M)) :
    ofList E (xs ++ ys) = ofList E xs * ofList E ys := by
  simp [ofList, List.map_append, List.prod_append]

lemma ofList_flatten (bs : List (List (G × M))) :
    ofList E bs.flatten = (bs.map (ofList E)).prod := by
  induction bs with
  | nil => rfl
  | cons b bs ih => simp [List.flatten_cons, ofList_append, ih]

/-- **Tiling is exact.** Splitting the keys into blocks, reducing each block, and
merging the block states gives attention over all keys. -/
theorem finalize_blocks (bs : List (List (G × M))) :
    finalize E ((bs.map (ofList E)).prod) = attendList E bs.flatten := by
  rw [← ofList_flatten, finalize_ofList]

/-- **Key order is irrelevant.** -/
theorem ofList_perm {xs ys : List (G × M)} (h : xs.Perm ys) : ofList E xs = ofList E ys :=
  (h.map _).prod_eq

/-- **Numerical safety.** The running maximum of a nonempty fold is the maximum
score, so every exponent `s - m` the kernel evaluates is nonpositive. -/
theorem ofList_max {xs : List (G × M)} {st : OState G 𝕜 M} (h : ofList E xs = ⟨some st⟩) :
    (∀ p ∈ xs, p.1 ≤ st.m) ∧ ∃ p ∈ xs, p.1 = st.m := by
  induction xs generalizing st with
  | nil => simp [ofList] at h; cases h
  | cons p xs ih =>
    by_cases hnil : xs = []
    · subst hnil
      simp only [ofList, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one,
        single, Online.mk.injEq, Option.some.injEq] at h
      subst h
      simp
    · obtain ⟨st', hst', -⟩ := ofList_some E hnil
      have hcons : ofList E (p :: xs) = ⟨some (mergeS E ⟨p.1, 1, p.2⟩ st')⟩ := by
        simp only [ofList, List.map_cons, List.prod_cons] at hst' ⊢
        rw [hst']; rfl
      rw [hcons] at h
      simp only [Online.mk.injEq, Option.some.injEq] at h
      subst h
      obtain ⟨hle, p', hp', heq⟩ := ih hst'
      refine ⟨?_, ?_⟩
      · intro q hq
        simp only [List.mem_cons] at hq
        rcases hq with rfl | hq
        · exact le_max_left _ _
        · exact le_trans (hle q hq) (le_max_right _ _)
      · rcases le_total p.1 st'.m with hpm | hpm
        · exact ⟨p', by simp [hp'], by simp [mergeS, max_eq_right hpm, heq]⟩
        · exact ⟨p, by simp, by simp [mergeS, max_eq_left hpm]⟩

theorem ofList_exp_le_one [IsOrderedAddMonoid G] {xs : List (G × M)} {st : OState G 𝕜 M} (h : ofList E xs = ⟨some st⟩)
    (p : G × M) (hp : p ∈ xs) : E (p.1 - st.m) ≤ 1 :=
  E.le_one_of_nonpos (sub_nonpos.mpr ((ofList_max E h).1 p hp))

end Online

end Online

end LLM
