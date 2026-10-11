import Mathlib.Algebra.FreeMonoid.Basic
import Mathlib.Data.List.Infix
import Mathlib.Data.List.Nodup
import Mathlib.Data.Fintype.Basic
import Mathlib.Tactic.Linarith

/-!
# Tokenization

A tokenizer is a pair of maps between strings over an alphabet `C` (bytes) and
sequences of tokens `T`. The meaning of a token is the string it spells, `str t`, and
the meaning of a token sequence is the concatenation of those strings. Decoding is
therefore the free-monoid homomorphism `FreeMonoid T →* FreeMonoid C` induced by
`str` (`decodeHom`).

Results:

* **negative result** `homomorphic_no_compression`: if an encoder inverts decoding
  and respects concatenation, every token it emits spells exactly one character,
  so it compresses nothing. Any tokenizer that compresses must violate
  `enc (a ++ b) = enc a ++ enc b` (`compressing_not_homomorphic`). Tokenization,
  like pricing in the accounting semantics, lives outside the compositional core,
  and the "prompt boundary" artifacts of real systems are a consequence of this
  theorem, not a defect of any particular implementation;
* `decode_not_injective`: with byte fallback and any multi-byte token, one string has
  several token sequences;
* `tokenizations_spec`: an exact enumeration of every token sequence that spells a
  given string (sound, complete, and duplicate-free). The probability of a string
  is the sum of the probabilities of these sequences;
* BPE, defined as implementations run it (repeatedly merge the leftmost occurrence
  of the lowest-ranked adjacent pair), with `bpe_decode` (lossless), `bpe_normal`
  (the output contains no mergeable pair), `bpe_idem`, `bpeStep_spec` (each step is
  the leftmost minimum-rank merge), and `encode_injective`;
* `bpe_not_shortest`: BPE does not find the shortest tokenization.
-/

namespace LLM.Tok

universe u v

variable {C : Type u} {T : Type v}

/-! ### Decoding -/

/-- Decode a token sequence by concatenating the strings of its tokens. -/
def decode (str : T → List C) (ts : List T) : List C := ts.flatMap str

@[simp] lemma decode_nil (str : T → List C) : decode str [] = [] := rfl
@[simp] lemma decode_cons (str : T → List C) (t : T) (ts : List T) :
    decode str (t :: ts) = str t ++ decode str ts := rfl
@[simp] lemma decode_append (str : T → List C) (a b : List T) :
    decode str (a ++ b) = decode str a ++ decode str b := by simp [decode]

/-- Decoding as a homomorphism of free monoids. -/
def decodeHom (str : T → List C) : FreeMonoid T →* FreeMonoid C :=
  FreeMonoid.lift (fun t => FreeMonoid.ofList (str t))

theorem decodeHom_apply (str : T → List C) (w : FreeMonoid T) :
    (decodeHom str w).toList = decode str w.toList := by
  induction w using FreeMonoid.inductionOn' with
  | one => rfl
  | of_mul t w ih =>
    rw [map_mul, FreeMonoid.toList_mul, ih, FreeMonoid.toList_mul, FreeMonoid.toList_of]
    simp [decodeHom, FreeMonoid.lift_eval_of]

lemma length_le_length_decode (str : T → List C) (hne : ∀ t, str t ≠ []) (ts : List T) :
    ts.length ≤ (decode str ts).length := by
  induction ts with
  | nil => simp
  | cons t ts ih =>
    simp only [decode_cons, List.length_append, List.length_cons]
    have := List.length_pos_iff.mpr (hne t)
    omega

lemma decode_eq_nil (str : T → List C) (hne : ∀ t, str t ≠ []) {ts : List T}
    (h : decode str ts = []) : ts = [] := by
  have := length_le_length_decode str hne ts
  rw [h] at this
  exact List.length_eq_zero_iff.mp (by simpa using this)

/-! ### The no-compression theorem -/

/-- **A homomorphic tokenizer does not compress.** Let every token spell a nonempty
string. If `enc` is a right inverse of decoding and respects concatenation, then
`enc` maps every string of length `n` to exactly `n` tokens. -/
theorem homomorphic_no_compression (str : T → List C) (hne : ∀ t, str t ≠ [])
    (enc : List C → List T) (happ : ∀ a b, enc (a ++ b) = enc a ++ enc b)
    (hsec : ∀ s, decode str (enc s) = s) (s : List C) : (enc s).length = s.length := by
  induction s with
  | nil => rw [decode_eq_nil str hne (hsec [])]; rfl
  | cons c s ih =>
    rw [show c :: s = [c] ++ s from rfl, happ, List.length_append, ih]
    have h1 : (enc [c]).length ≤ 1 := by
      have := length_le_length_decode str hne (enc [c])
      rwa [hsec] at this
    have h2 : enc [c] ≠ [] := by
      intro h
      have := hsec [c]
      rw [h] at this
      simp at this
    have := List.length_pos_iff.mpr h2
    simp only [List.length_append, List.length_singleton]
    omega

/-- Every token a homomorphic tokenizer emits spells a single character. -/
theorem homomorphic_tokens_are_characters (str : T → List C) (hne : ∀ t, str t ≠ [])
    (enc : List C → List T) (happ : ∀ a b, enc (a ++ b) = enc a ++ enc b)
    (hsec : ∀ s, decode str (enc s) = s) (s : List C) (t : T) (ht : t ∈ enc s) :
    (str t).length = 1 := by
  have hlen := homomorphic_no_compression str hne enc happ hsec s
  have hdec := congrArg List.length (hsec s)
  -- the total length is the sum of token lengths, each at least one
  have key : ∀ ts : List T, (decode str ts).length = ts.length → ∀ t ∈ ts, (str t).length = 1 := by
    intro ts
    induction ts with
    | nil => simp
    | cons t' ts ih =>
      intro h t'' ht''
      simp only [decode_cons, List.length_append, List.length_cons] at h
      have h1 := List.length_pos_iff.mpr (hne t')
      have h2 := length_le_length_decode str hne ts
      simp only [List.mem_cons] at ht''
      rcases ht'' with rfl | ht''
      · omega
      · exact ih (by omega) t'' ht''
  exact key (enc s) (by rw [hdec, hlen]) t ht

/-- **A compressing tokenizer is not compositional.** -/
theorem compressing_not_homomorphic (str : T → List C) (hne : ∀ t, str t ≠ [])
    (enc : List C → List T) (hsec : ∀ s, decode str (enc s) = s) (s : List C)
    (hs : (enc s).length < s.length) : ∃ a b, enc (a ++ b) ≠ enc a ++ enc b := by
  by_contra h
  push Not at h
  have := homomorphic_no_compression str hne enc h hsec s
  omega

/-- The same statement for monoid homomorphisms. -/
theorem monoidHom_no_compression (str : T → List C) (hne : ∀ t, str t ≠ [])
    (enc : FreeMonoid C →* FreeMonoid T)
    (hsec : ∀ s, (decodeHom str (enc s)).toList = s.toList) (s : FreeMonoid C) :
    (enc s).toList.length = s.toList.length := by
  let e : List C → List T := fun l => (enc (FreeMonoid.ofList l)).toList
  have happ : ∀ a b, e (a ++ b) = e a ++ e b := by
    intro a b
    simp only [e, ← FreeMonoid.toList_mul, ← map_mul]
    rfl
  have hsec' : ∀ l, decode str (e l) = l := by
    intro l
    have := hsec (FreeMonoid.ofList l)
    rw [decodeHom_apply] at this
    simpa using this
  exact homomorphic_no_compression str hne e happ hsec' s.toList

/-- **Tokenizations are not unique.** With a single-character token for every
character and some token spelling two or more characters, decoding is not
injective. -/
theorem decode_not_injective (str : T → List C) (hbyte : ∀ c, ∃ t, str t = [c]) (t₀ : T)
    (h2 : 2 ≤ (str t₀).length) : ¬ Function.Injective (decode str) := by
  intro hinj
  choose byte hb using hbyte
  have hdec : decode str ((str t₀).map byte) = decode str [t₀] := by
    simp only [decode, List.flatMap_map, hb, List.flatMap_singleton', List.flatMap_cons,
      List.flatMap_nil, List.append_nil]
  have := congrArg List.length (hinj hdec)
  simp at this
  omega

/-! ### Enumerating tokenizations -/

section Enumerate

variable [DecidableEq C]

/-- All token sequences over `vocab` that spell `s`, with fuel. -/
def tokzFuel (vocab : List T) (str : T → List C) : ℕ → List C → List (List T)
  | _, [] => [[]]
  | 0, _ :: _ => []
  | n + 1, s =>
      vocab.flatMap (fun t =>
        if str t ≠ [] ∧ str t <+: s then
          (tokzFuel vocab str n (s.drop (str t).length)).map (t :: ·)
        else [])

/-- All token sequences over `vocab`, using only tokens that spell nonempty strings,
whose decoding is `s`. -/
def tokenizations (vocab : List T) (str : T → List C) (s : List C) : List (List T) :=
  tokzFuel vocab str s.length s

lemma mem_tokzFuel (vocab : List T) (str : T → List C) :
    ∀ (n : ℕ) (s : List C), s.length ≤ n → ∀ ts : List T,
      ts ∈ tokzFuel vocab str n s ↔
        (∀ t ∈ ts, t ∈ vocab ∧ str t ≠ []) ∧ decode str ts = s := by
  intro n
  induction n with
  | zero =>
    intro s hs ts
    have : s = [] := List.length_eq_zero_iff.mp (by omega)
    subst this
    simp only [tokzFuel, List.mem_singleton]
    constructor
    · rintro rfl; simp
    · rintro ⟨hall, hdec⟩
      cases ts with
      | nil => rfl
      | cons t ts =>
        simp only [decode_cons, List.append_eq_nil_iff] at hdec
        exact absurd hdec.1 (hall t (by simp)).2
  | succ n ih =>
    intro s hs ts
    cases s with
    | nil =>
      simp only [tokzFuel, List.mem_singleton]
      constructor
      · rintro rfl; simp
      · rintro ⟨hall, hdec⟩
        cases ts with
        | nil => rfl
        | cons t ts =>
          simp only [decode_cons, List.append_eq_nil_iff] at hdec
          exact absurd hdec.1 (hall t (by simp)).2
    | cons c s' =>
      simp only [tokzFuel, List.mem_flatMap]
      constructor
      · rintro ⟨t, htv, hts⟩
        split_ifs at hts with hpre
        · obtain ⟨ts', hts', rfl⟩ := List.mem_map.mp hts
          have hlen : ((c :: s').drop (str t).length).length ≤ n := by
            have := List.length_pos_iff.mpr hpre.1
            simp only [List.length_drop, List.length_cons] at hs ⊢
            omega
          obtain ⟨hall, hdec⟩ := (ih _ hlen ts').mp hts'
          refine ⟨?_, ?_⟩
          · intro t' ht'
            simp only [List.mem_cons] at ht'
            rcases ht' with rfl | ht'
            · exact ⟨htv, hpre.1⟩
            · exact hall t' ht'
          · rw [decode_cons, hdec]
            obtain ⟨u, hu⟩ := hpre.2
            rw [← hu]
            simp
        · simp at hts
      · rintro ⟨hall, hdec⟩
        cases ts with
        | nil => simp at hdec
        | cons t ts' =>
          have ht := hall t (by simp)
          refine ⟨t, ht.1, ?_⟩
          have hpre : str t <+: c :: s' := ⟨decode str ts', by rw [← hdec]; rfl⟩
          rw [ite_eq_left ⟨ht.2, hpre⟩]
          apply List.mem_map.mpr
          refine ⟨ts', ?_, rfl⟩
          have hdrop : (c :: s').drop (str t).length = decode str ts' := by
            rw [← hdec, decode_cons, List.drop_left]
          have hlen : ((c :: s').drop (str t).length).length ≤ n := by
            have := List.length_pos_iff.mpr ht.2
            simp only [List.length_drop, List.length_cons] at hs ⊢
            omega
          rw [ih _ hlen ts']
          exact ⟨fun t' ht' => hall t' (by simp [ht']), hdrop.symm⟩

/-- **The enumeration is sound and complete.** -/
theorem tokenizations_spec (vocab : List T) (str : T → List C) (s : List C) (ts : List T) :
    ts ∈ tokenizations vocab str s ↔ (∀ t ∈ ts, t ∈ vocab ∧ str t ≠ []) ∧ decode str ts = s :=
  mem_tokzFuel vocab str s.length s le_rfl ts

lemma nodup_tokzFuel (vocab : List T) (str : T → List C) (hv : vocab.Nodup) :
    ∀ (n : ℕ) (s : List C), (tokzFuel vocab str n s).Nodup := by
  intro n
  induction n with
  | zero => intro s; cases s <;> simp [tokzFuel]
  | succ n ih =>
    intro s
    cases s with
    | nil => simp [tokzFuel]
    | cons c s' =>
      simp only [tokzFuel]
      rw [List.nodup_flatMap]
      refine ⟨fun t _ => ?_, ?_⟩
      · split_ifs
        · exact (ih _).map (fun a b h => List.cons_injective h)
        · exact List.nodup_nil
      · refine hv.imp_of_mem (fun {a b} _ _ hab => ?_)
        rw [Function.onFun, List.disjoint_left]
        intro l hla hlb
        by_cases h1 : str a ≠ [] ∧ str a <+: c :: s'
        · by_cases h2 : str b ≠ [] ∧ str b <+: c :: s'
          · rw [ite_eq_left h1] at hla
            rw [ite_eq_left h2] at hlb
            obtain ⟨x, -, rfl⟩ := List.mem_map.mp hla
            obtain ⟨y, -, h⟩ := List.mem_map.mp hlb
            exact hab (List.cons.inj h).1.symm
          · rw [ite_eq_right h2] at hlb; simp at hlb
        · rw [ite_eq_right h1] at hla; simp at hla

/-- **The enumeration has no duplicates** when the vocabulary has none. -/
theorem tokenizations_nodup (vocab : List T) (str : T → List C) (hv : vocab.Nodup)
    (s : List C) : (tokenizations vocab str s).Nodup :=
  nodup_tokzFuel vocab str hv _ _

end Enumerate

/-! ### Byte-pair encoding

Every BPE implementation repeats one step: among the adjacent pairs of the current
sequence that may merge, take the one of lowest priority (leftmost on ties) and
replace it by its merged token. Implementations differ only in the *priority* of a
pair. BPE is therefore defined once, for an arbitrary priority function
`f : T → T → Option (ℕ × T)` returning a rank and the merged token, and the two
conventions in use are instances:

* `rankOf ms`: the rank of the merge *rule* `(a, b)`. This is the convention of
  Hugging Face `tokenizers` and the original GPT-2 encoder.
* `concatRank rank`: the rank of the *concatenated token* `a ++ b`. This is the
  convention of `tiktoken`.

`bpeWith_congr` gives a checkable condition under which the two conventions
produce the same output, and `hf_ne_tiktoken` shows that without it they differ.
-/

section BPE

/-! #### The generic algorithm -/

section Generic

variable (f : T → T → Option (ℕ × T))

/-- The priorities of the adjacent pairs of a sequence. -/
def pairsWith : List T → List (Option (ℕ × T))
  | a :: b :: rest => f a b :: pairsWith (b :: rest)
  | _ => []

lemma pairsWith_length : ∀ ts : List T, (pairsWith f ts).length = ts.length - 1
  | [] => rfl
  | [_] => rfl
  | a :: b :: rest => by
    simp only [pairsWith, List.length_cons, pairsWith_length (b :: rest)]
    omega

lemma pairsWith_getElem : ∀ (ts : List T) (i : ℕ) (h : i < (pairsWith f ts).length),
    (pairsWith f ts)[i] = f (ts[i]'(by rw [pairsWith_length] at h; omega))
      (ts[i + 1]'(by rw [pairsWith_length] at h; omega))
  | [], i, h => by simp [pairsWith] at h
  | [_], i, h => by simp [pairsWith] at h
  | a :: b :: rest, 0, _ => rfl
  | a :: b :: rest, i + 1, h => by
    simp only [pairsWith, List.getElem_cons_succ]
    exact pairsWith_getElem (b :: rest) i _

/-- The leftmost occurrence of the minimum rank: `(position, rank, merged token)`. -/
def best : List (Option (ℕ × T)) → Option (ℕ × ℕ × T)
  | [] => none
  | none :: rest => (best rest).map (fun x => (x.1 + 1, x.2))
  | some (r, c) :: rest =>
      match best rest with
      | none => some (0, r, c)
      | some (i, r', c') => if r' < r then some (i + 1, r', c') else some (0, r, c)

omit f in
lemma best_eq_none_iff : ∀ l : List (Option (ℕ × T)), best l = none ↔ ∀ x ∈ l, x = none
  | [] => by simp [best]
  | none :: rest => by simp [best, best_eq_none_iff rest]
  | some (r, c) :: rest => by
    simp only [best, List.mem_cons, forall_eq_or_imp, reduceCtorEq, false_and, iff_false]
    split <;> [simp; split_ifs <;> simp]

omit f in
/-- The specification of `best`: it returns a position holding the minimum rank,
and every earlier position holds a strictly larger rank or none. -/
lemma best_spec : ∀ (l : List (Option (ℕ × T))) (i r : ℕ) (c : T), best l = some (i, r, c) →
    l[i]? = some (some (r, c)) ∧
      (∀ (j r' : ℕ) (c' : T), l[j]? = some (some (r', c')) → r ≤ r') ∧
      (∀ j < i, ∀ (r' : ℕ) (c' : T), l[j]? = some (some (r', c')) → r < r')
  | [], _, _, _, h => by simp [best] at h
  | none :: rest, i, r, c, h => by
    simp only [best] at h
    obtain ⟨⟨i', r', c'⟩, h1, h2⟩ := Option.map_eq_some_iff.mp h
    simp only [Prod.mk.injEq] at h2
    obtain ⟨rfl, rfl, rfl⟩ := h2
    obtain ⟨hget, hmin, hleft⟩ := best_spec rest i' r' c' h1
    refine ⟨by simpa using hget, ?_, ?_⟩
    · intro j r'' c'' hj
      cases j with
      | zero => simp at hj
      | succ j => exact hmin j r'' c'' (by simpa using hj)
    · intro j hj r'' c'' hjr
      cases j with
      | zero => simp at hjr
      | succ j => exact hleft j (by omega) r'' c'' (by simpa using hjr)
  | some (r₀, c₀) :: rest, i, r, c, h => by
    simp only [best] at h
    split at h
    · rename_i hnone
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl, rfl⟩ := h
      have hall := (best_eq_none_iff rest).mp hnone
      refine ⟨rfl, ?_, fun j hj => absurd hj (Nat.not_lt_zero j)⟩
      intro j r' c' hj
      cases j with
      | zero => simp at hj; obtain ⟨rfl, rfl⟩ := hj; exact le_rfl
      | succ j =>
        have hmem : some (r', c') ∈ rest := List.mem_of_getElem? (by simpa using hj)
        exact absurd (hall _ hmem) (by simp)
    · rename_i i' r' c' hsome
      obtain ⟨hget, hmin, hleft⟩ := best_spec rest i' r' c' hsome
      split_ifs at h with hlt
      · simp only [Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl, rfl⟩ := h
        refine ⟨by simpa using hget, ?_, ?_⟩
        · intro j r'' c'' hj
          cases j with
          | zero => simp at hj; obtain ⟨rfl, rfl⟩ := hj; exact hlt.le
          | succ j => exact hmin j r'' c'' (by simpa using hj)
        · intro j hj r'' c'' hjr
          cases j with
          | zero => simp at hjr; obtain ⟨rfl, rfl⟩ := hjr; exact hlt
          | succ j => exact hleft j (by omega) r'' c'' (by simpa using hjr)
      · simp only [Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl, rfl⟩ := h
        push Not at hlt
        refine ⟨rfl, ?_, fun j hj => absurd hj (Nat.not_lt_zero j)⟩
        intro j r'' c'' hj
        cases j with
        | zero => simp at hj; obtain ⟨rfl, rfl⟩ := hj; exact le_rfl
        | succ j => exact le_trans hlt (hmin j r'' c'' (by simpa using hj))

omit f in
lemma best_lt_length (l : List (Option (ℕ × T))) (i r : ℕ) (c : T)
    (h : best l = some (i, r, c)) : i < l.length := by
  have := (best_spec l i r c h).1
  exact (List.getElem?_eq_some_iff.mp this).1

omit f in
/-- Replace positions `i` and `i + 1` by the token `c`. -/
def mergeAt (ts : List T) (i : ℕ) (c : T) : List T := ts.take i ++ c :: ts.drop (i + 2)

omit f in
lemma mem_mergeAt {ts : List T} {i : ℕ} {c t : T} (h : t ∈ mergeAt ts i c) : t ∈ ts ∨ t = c := by
  simp only [mergeAt, List.mem_append, List.mem_cons] at h
  rcases h with h | h | h
  · exact Or.inl (List.mem_of_mem_take h)
  · exact Or.inr h
  · exact Or.inl (List.mem_of_mem_drop h)

/-- One BPE step: merge the leftmost occurrence of the lowest-priority pair. -/
def stepWith (ts : List T) : Option (List T) :=
  (best (pairsWith f ts)).map (fun x => mergeAt ts x.1 x.2.2)

/-- BPE with explicit fuel. -/
def fuelWith : ℕ → List T → List T
  | 0, ts => ts
  | n + 1, ts =>
      match stepWith f ts with
      | none => ts
      | some ts' => fuelWith n ts'

/-- BPE for the priority function `f`: iterate `stepWith` to a fixed point. The fuel
`ts.length` always suffices (`bpeWith_normal`). -/
def bpeWith (ts : List T) : List T := fuelWith f ts.length ts

/-- **Specification of one step.** The step merges a pair of minimum priority, and
every pair to its left has strictly larger priority. -/
theorem stepWith_spec (ts ts' : List T) (h : stepWith f ts = some ts') :
    ∃ (i r : ℕ) (c : T) (hi : i + 1 < ts.length),
      ts' = mergeAt ts i c ∧ f ts[i] ts[i + 1] = some (r, c) ∧
      (∀ (j : ℕ) (hj : j + 1 < ts.length) r' c', f ts[j] ts[j + 1] = some (r', c') → r ≤ r') ∧
      (∀ (j : ℕ) (hj : j + 1 < ts.length), j < i → ∀ r' c',
        f ts[j] ts[j + 1] = some (r', c') → r < r') := by
  unfold stepWith at h
  obtain ⟨⟨i, r, c⟩, hb, rfl⟩ := Option.map_eq_some_iff.mp h
  obtain ⟨hget, hmin, hleft⟩ := best_spec _ i r c hb
  have hil := best_lt_length _ i r c hb
  rw [pairsWith_length] at hil
  have hi : i + 1 < ts.length := by omega
  have hrank : f ts[i] ts[i + 1] = some (r, c) := by
    obtain ⟨hlt, heq⟩ := List.getElem?_eq_some_iff.mp hget
    rw [pairsWith_getElem] at heq
    exact heq
  refine ⟨i, r, c, hi, rfl, hrank, ?_, ?_⟩
  · intro j hj r' c' hr
    apply hmin j r' c'
    rw [List.getElem?_eq_some_iff]
    refine ⟨by rw [pairsWith_length]; omega, ?_⟩
    rw [pairsWith_getElem]; exact hr
  · intro j hj hji r' c' hr
    apply hleft j hji r' c'
    rw [List.getElem?_eq_some_iff]
    refine ⟨by rw [pairsWith_length]; omega, ?_⟩
    rw [pairsWith_getElem]; exact hr

lemma stepWith_length (ts ts' : List T) (h : stepWith f ts = some ts') :
    ts'.length + 1 = ts.length := by
  obtain ⟨i, r, c, hi, rfl, -⟩ := stepWith_spec f ts ts' h
  simp only [mergeAt, List.length_append, List.length_take, List.length_cons, List.length_drop]
  omega

/-- A sequence is in normal form when no adjacent pair may merge. -/
theorem stepWith_eq_none_iff (ts : List T) :
    stepWith f ts = none ↔ ∀ (i : ℕ) (hi : i + 1 < ts.length), f ts[i] ts[i + 1] = none := by
  unfold stepWith
  rw [Option.map_eq_none_iff, best_eq_none_iff]
  constructor
  · intro h i hi
    have hlt : i < (pairsWith f ts).length := by rw [pairsWith_length]; omega
    have := h _ (List.getElem_mem hlt)
    rwa [pairsWith_getElem] at this
  · intro h x hx
    obtain ⟨i, hlt, rfl⟩ := List.getElem_of_mem hx
    rw [pairsWith_getElem]
    exact h i _

lemma fuelWith_normal : ∀ (n : ℕ) (ts : List T), ts.length ≤ n + 1 →
    stepWith f (fuelWith f n ts) = none := by
  intro n
  induction n with
  | zero =>
    intro ts hts
    simp only [fuelWith]
    rw [stepWith_eq_none_iff]
    intro i hi; omega
  | succ n ih =>
    intro ts hts
    simp only [fuelWith]
    split
    · assumption
    · rename_i ts' h
      have := stepWith_length f ts ts' h
      exact ih ts' (by omega)

/-- **BPE reaches a normal form.** -/
theorem bpeWith_normal (ts : List T) : stepWith f (bpeWith f ts) = none :=
  fuelWith_normal f _ ts (by omega)

lemma fuelWith_of_normal (ts : List T) (h : stepWith f ts = none) :
    ∀ n, fuelWith f n ts = ts := by
  intro n; cases n <;> simp [fuelWith, h]

/-- **BPE is idempotent.** -/
theorem bpeWith_idem (ts : List T) : bpeWith f (bpeWith f ts) = bpeWith f ts :=
  fuelWith_of_normal f _ (bpeWith_normal f ts) _

/-- A priority function is sound for the spelling `str` when merging never changes
what the sequence spells. -/
def Sound (str : T → List C) : Prop := ∀ a b r c, f a b = some (r, c) → str c = str a ++ str b

lemma stepWith_decode (str : T → List C) (hf : Sound f str) (ts ts' : List T)
    (h : stepWith f ts = some ts') : decode str ts' = decode str ts := by
  obtain ⟨i, r, c, hi, rfl, hr, -⟩ := stepWith_spec f ts ts' h
  have hc := hf _ _ r c hr
  conv_rhs => rw [← List.take_append_drop i ts]
  rw [List.drop_eq_getElem_cons (by omega), List.drop_eq_getElem_cons (by omega)]
  simp only [mergeAt, decode_append, decode_cons, hc, List.append_assoc]

lemma fuelWith_decode (str : T → List C) (hf : Sound f str) :
    ∀ (n : ℕ) (ts : List T), decode str (fuelWith f n ts) = decode str ts := by
  intro n
  induction n with
  | zero => intro ts; rfl
  | succ n ih =>
    intro ts
    simp only [fuelWith]
    split
    · rfl
    · rename_i ts' h
      rw [ih, stepWith_decode f str hf ts ts' h]

/-- **BPE is lossless** for every sound priority function. -/
theorem bpeWith_decode (str : T → List C) (hf : Sound f str) (ts : List T) :
    decode str (bpeWith f ts) = decode str ts :=
  fuelWith_decode f str hf _ ts

lemma pairsWith_congr (g : T → T → Option (ℕ × T)) (Tk : T → Prop)
    (hfg : ∀ a b, Tk a → Tk b → f a b = g a b) :
    ∀ ts : List T, (∀ t ∈ ts, Tk t) → pairsWith f ts = pairsWith g ts
  | [], _ => rfl
  | [_], _ => rfl
  | a :: b :: rest, h => by
    simp only [pairsWith]
    rw [hfg a b (h a (by simp)) (h b (by simp)),
      pairsWith_congr g Tk hfg (b :: rest) (fun t ht => h t (List.mem_cons_of_mem a ht))]

/-- **Agreement of conventions.** Let `Tk` be a set of tokens closed under the merges
of `f`. If `f` and `g` assign the same priority and result to every pair of tokens
in `Tk`, BPE under `f` and under `g` agree on every sequence of tokens in `Tk`. -/
theorem bpeWith_congr (g : T → T → Option (ℕ × T)) (Tk : T → Prop)
    (hclosed : ∀ a b r c, Tk a → Tk b → f a b = some (r, c) → Tk c)
    (hfg : ∀ a b, Tk a → Tk b → f a b = g a b) (ts : List T) (hts : ∀ t ∈ ts, Tk t) :
    bpeWith f ts = bpeWith g ts := by
  unfold bpeWith
  generalize ts.length = n
  induction n generalizing ts with
  | zero => rfl
  | succ n ih =>
    have hstep : stepWith f ts = stepWith g ts := by
      unfold stepWith; rw [pairsWith_congr f g Tk hfg ts hts]
    simp only [fuelWith]
    rw [← hstep]
    rcases hs : stepWith f ts with _ | ts'
    · rfl
    · simp only
      apply ih
      obtain ⟨i, r, c, hi, rfl, hr, -⟩ := stepWith_spec f ts ts' hs
      intro t ht
      rcases mem_mergeAt ht with ht | rfl
      · exact hts t ht
      · exact hclosed _ _ r t (hts _ (List.getElem_mem _)) (hts _ (List.getElem_mem _)) hr

end Generic

/-! #### Merge-rule ranks (Hugging Face `tokenizers`, GPT-2) -/

section MergeRules

variable [DecidableEq T]

/-- A merge rule `(a, b, c)`: the adjacent pair `a b` may be replaced by `c`. The rank
of a rule is its position in the list. -/
abbrev Merge (T : Type v) := T × T × T

/-- The rank of the pair `(a, b)` and its merged token: the first rule for the pair. -/
def rankOf : List (Merge T) → T → T → Option (ℕ × T)
  | [], _, _ => none
  | (a', b', c) :: ms, a, b =>
      if a' = a ∧ b' = b then some (0, c) else (rankOf ms a b).map (fun rc => (rc.1 + 1, rc.2))

lemma rankOf_spec : ∀ (ms : List (Merge T)) (a b : T) (r : ℕ) (c : T),
    rankOf ms a b = some (r, c) → ms[r]? = some (a, b, c)
  | [], _, _, _, _, h => by simp [rankOf] at h
  | (a', b', c') :: ms, a, b, r, c, h => by
    simp only [rankOf] at h
    split_ifs at h with hab
    · obtain ⟨rfl, rfl⟩ := hab
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      rfl
    · obtain ⟨⟨r', c''⟩, h1, h2⟩ := Option.map_eq_some_iff.mp h
      simp only [Prod.mk.injEq] at h2
      obtain ⟨rfl, rfl⟩ := h2
      simpa using rankOf_spec ms a b r' c'' h1

/-- A merge table is well formed when each rule's result spells the concatenation of
its two inputs. -/
def WellFormed (str : T → List C) (ms : List (Merge T)) : Prop :=
  ∀ m ∈ ms, str m.2.2 = str m.1 ++ str m.2.1

lemma rankOf_sound (str : T → List C) (ms : List (Merge T)) (hwf : WellFormed str ms) :
    Sound (rankOf ms) str := by
  intro a b r c h
  exact hwf _ (List.mem_of_getElem? (rankOf_spec ms a b r c h))

/-- One BPE step under merge-rule ranks. -/
def bpeStep (ms : List (Merge T)) (ts : List T) : Option (List T) := stepWith (rankOf ms) ts

/-- Byte-pair encoding under merge-rule ranks. -/
def bpe (ms : List (Merge T)) (ts : List T) : List T := bpeWith (rankOf ms) ts

/-- **Specification of one step.** It merges the leftmost occurrence of the
lowest-ranked rule. -/
theorem bpeStep_spec (ms : List (Merge T)) (ts ts' : List T) (h : bpeStep ms ts = some ts') :
    ∃ (i r : ℕ) (c : T) (hi : i + 1 < ts.length),
      ts' = mergeAt ts i c ∧ ms[r]? = some (ts[i], ts[i + 1], c) ∧
      (∀ (j : ℕ) (hj : j + 1 < ts.length) r' c', rankOf ms ts[j] ts[j + 1] = some (r', c') → r ≤ r') ∧
      (∀ (j : ℕ) (hj : j + 1 < ts.length), j < i → ∀ r' c',
        rankOf ms ts[j] ts[j + 1] = some (r', c') → r < r') := by
  obtain ⟨i, r, c, hi, h1, h2, h3, h4⟩ := stepWith_spec (rankOf ms) ts ts' h
  exact ⟨i, r, c, hi, h1, rankOf_spec ms _ _ r c h2, h3, h4⟩

/-- **BPE is lossless.** -/
theorem bpe_decode (str : T → List C) (ms : List (Merge T)) (hwf : WellFormed str ms)
    (ts : List T) : decode str (bpe ms ts) = decode str ts :=
  bpeWith_decode _ str (rankOf_sound str ms hwf) ts

/-- **BPE reaches a normal form.** No adjacent pair of the output has a merge rule. -/
theorem bpe_normal (ms : List (Merge T)) (ts : List T) :
    ∀ (i : ℕ) (hi : i + 1 < (bpe ms ts).length), rankOf ms (bpe ms ts)[i] (bpe ms ts)[i + 1] = none :=
  (stepWith_eq_none_iff _ _).mp (bpeWith_normal _ ts)

/-- **BPE is idempotent.** -/
theorem bpe_idem (ms : List (Merge T)) (ts : List T) : bpe ms (bpe ms ts) = bpe ms ts :=
  bpeWith_idem _ ts

/-- The BPE encoder: map every character to its base token, then merge. -/
def encode (byte : C → T) (ms : List (Merge T)) (s : List C) : List T := bpe ms (s.map byte)

/-- **Round trip.** -/
theorem decode_encode (str : T → List C) (byte : C → T) (hbyte : ∀ c, str (byte c) = [c])
    (ms : List (Merge T)) (hwf : WellFormed str ms) (s : List C) :
    decode str (encode byte ms s) = s := by
  unfold encode
  rw [bpe_decode str ms hwf]
  induction s with
  | nil => rfl
  | cons c s ih => simp [hbyte, ih]

/-- The BPE encoder is injective. -/
theorem encode_injective (str : T → List C) (byte : C → T) (hbyte : ∀ c, str (byte c) = [c])
    (ms : List (Merge T)) (hwf : WellFormed str ms) : Function.Injective (encode byte ms) := by
  intro s s' h
  rw [← decode_encode str byte hbyte ms hwf s, ← decode_encode str byte hbyte ms hwf s', h]

end MergeRules

/-! #### Concatenation ranks (`tiktoken`) -/

section Concat

variable [DecidableEq C]

/-- The `tiktoken` convention: tokens are their own spellings, and the priority of an
adjacent pair is the rank of the token it would form. -/
def concatRank (rank : List C → Option ℕ) (a b : List C) : Option (ℕ × List C) :=
  (rank (a ++ b)).map (fun r => (r, a ++ b))

/-- Byte-pair encoding under concatenation ranks. -/
def bpeTik (rank : List C → Option ℕ) (ts : List (List C)) : List (List C) :=
  bpeWith (concatRank rank) ts

omit [DecidableEq C] in
lemma concatRank_sound (rank : List C → Option ℕ) : Sound (concatRank rank) id := by
  intro a b r c h
  obtain ⟨r', -, h2⟩ := Option.map_eq_some_iff.mp h
  simp only [Prod.mk.injEq] at h2
  rw [← h2.2]; rfl

omit [DecidableEq C] in
/-- **`tiktoken` BPE is lossless.** -/
theorem bpeTik_flatten (rank : List C → Option ℕ) (ts : List (List C)) :
    (bpeTik rank ts).flatten = ts.flatten := by
  have := bpeWith_decode (concatRank rank) id (concatRank_sound rank) ts
  simpa [decode, bpeTik] using this

/-- The merge table, with string tokens, that a list of merge pairs denotes. -/
def hfMerges (ms : List (List C × List C)) : List (Merge (List C)) :=
  ms.map (fun m => (m.1, m.2, m.1 ++ m.2))

/-- The rank table `tiktoken` uses for a list of merge pairs: each token is ranked by
the first merge that produces it. -/
def tikRank (ms : List (List C × List C)) (t : List C) : Option ℕ :=
  ms.findIdx? (fun m => m.1 ++ m.2 = t)

/-- **When the conventions agree.** If, on a vocabulary closed under merging, every
pair of tokens has the same rank and result under both conventions, merge-rule BPE
(Hugging Face) and concatenation-rank BPE (the BPE step of `tiktoken`) produce the
same tokens on every sequence over that vocabulary. The
hypotheses are decidable for a finite vocabulary, and the oracle checks them. -/
theorem hf_eq_tiktoken (ms : List (List C × List C)) (vocab : List (List C))
    (hclosed : ∀ a b r c, a ∈ vocab → b ∈ vocab → rankOf (hfMerges ms) a b = some (r, c) → c ∈ vocab)
    (hagree : ∀ a b, a ∈ vocab → b ∈ vocab → rankOf (hfMerges ms) a b = concatRank (tikRank ms) a b)
    (ts : List (List C)) (hts : ∀ t ∈ ts, t ∈ vocab) :
    bpe (hfMerges ms) ts = bpeTik (tikRank ms) ts :=
  bpeWith_congr _ _ (· ∈ vocab) hclosed hagree ts hts

/-! #### Whole-word lookup

`tiktoken` looks every pre-tokenized word up in its vocabulary before running BPE,
and emits the word as a single token when it is one. Hugging Face `tokenizers` does
the same when `ignore_merges` is set. The lookup is a convention on top of BPE, and
it changes results (`lookup_matters`).
-/

/-- Emit a nonempty word that is itself a vocabulary token directly; otherwise run
the encoder `enc`. -/
def lookupFirst (isTok : List C → Bool) (enc : List C → List (List C)) (w : List C) :
    List (List C) :=
  if w ≠ [] ∧ isTok w = true then [w] else enc w

omit [DecidableEq C] in
/-- The lookup preserves losslessness. -/
theorem lookupFirst_flatten (isTok : List C → Bool) (enc : List C → List (List C))
    (henc : ∀ w, (enc w).flatten = w) (w : List C) : (lookupFirst isTok enc w).flatten = w := by
  unfold lookupFirst
  split_ifs
  · simp
  · exact henc w

omit [DecidableEq C] in
lemma flatten_map_singleton (w : List C) : (w.map (fun c => [c])).flatten = w := by
  induction w with
  | nil => rfl
  | cons c w ih => simp [ih]

/-- `tiktoken`'s encoder for one word: lookup, then concatenation-rank BPE from
characters. -/
def tikEncode (ms : List (List C × List C)) (w : List C) : List (List C) :=
  lookupFirst (fun t => t.length == 1 || (tikRank ms t).isSome)
    (fun w => bpeTik (tikRank ms) (w.map (fun c => [c]))) w

/-- Hugging Face's encoder for one word with `ignore_merges`: lookup, then merge-rule
BPE from characters. -/
def hfEncodeLookup (ms : List (List C × List C)) (w : List C) : List (List C) :=
  lookupFirst (fun t => t.length == 1 || (ms.any (fun m => m.1 ++ m.2 == t)))
    (fun w => bpe (hfMerges ms) (w.map (fun c => [c]))) w

/-- `tiktoken`'s encoder is lossless. -/
theorem tikEncode_flatten (ms : List (List C × List C)) (w : List C) :
    (tikEncode ms w).flatten = w :=
  lookupFirst_flatten _ _ (fun w => by
    rw [bpeTik_flatten, flatten_map_singleton]) w

end Concat

end BPE

/-! ### Counterexamples -/

section Counterexample

/-- Characters `a b c` are `0 1 2`. Tokens: `0 1 2` the characters, `3 = bc`,
`4 = ab`, `5 = abc`. -/
def exStr : Fin 6 → List (Fin 3)
  | 0 => [0] | 1 => [1] | 2 => [2] | 3 => [1, 2] | 4 => [0, 1] | 5 => [0, 1, 2]

/-- Merge `b c` first, then `a b`, then `ab c`. -/
def exMerges : List (Merge (Fin 6)) := [(1, 2, 3), (0, 1, 4), (4, 2, 5)]

lemma exMerges_wellFormed : WellFormed exStr exMerges := by
  intro m hm
  simp only [exMerges, List.mem_cons, List.mem_nil_iff, or_false] at hm
  rcases hm with rfl | rfl | rfl <;> rfl

/-- **BPE does not minimize the number of tokens.** On `abc` it returns `a · bc`, two
tokens, while the single token `abc` also spells the string. -/
theorem bpe_not_shortest :
    bpe exMerges [0, 1, 2] = [0, 3] ∧ decode exStr [5] = decode exStr [0, 1, 2] := by
  constructor
  · decide
  · rfl

/-- The same string has four tokenizations over this vocabulary. -/
theorem ex_tokenization_count :
    (tokenizations [0, 1, 2, 3, 4, 5] exStr [0, 1, 2]).length = 4 := by
  decide

/-- The same merges as pairs of strings over `a b c`. -/
def exPairs : List (List (Fin 3) × List (Fin 3)) := [([1], [2]), ([0], [1]), ([0, 1], [2])]

/-- **The two BPE conventions disagree.** On `abc`, merge-rule ranks (Hugging Face)
give `a · bc`, and concatenation ranks (`tiktoken`) give `abc`: the pair `a · bc` is
not a merge rule, but its concatenation is a ranked token. -/
theorem hf_ne_tiktoken :
    bpe (hfMerges exPairs) [[0], [1], [2]] = [[0], [1, 2]] ∧
      bpeTik (tikRank exPairs) [[0], [1], [2]] = [[0, 1, 2]] := by
  decide

/-- Merges `b c`, `a b`, `c d`, `ab cd` over `a b c d`. -/
def exPairs4 : List (List (Fin 4) × List (Fin 4)) :=
  [([1], [2]), ([0], [1]), ([2], [3]), ([0, 1], [2, 3])]

/-- **The whole-word lookup changes results.** The word `abcd` is a token, but BPE
from characters merges `b c` first and gets stuck at `a · bc · d`. -/
theorem lookup_matters :
    tikEncode exPairs4 [0, 1, 2, 3] = [[0, 1, 2, 3]] ∧
      bpeTik (tikRank exPairs4) [[0], [1], [2], [3]] = [[0], [1, 2], [3]] ∧
      bpe (hfMerges exPairs4) [[0], [1], [2], [3]] = [[0], [1, 2], [3]] := by
  decide

end Counterexample

end LLM.Tok
