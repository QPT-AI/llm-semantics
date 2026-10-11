import Mathlib.Data.List.Basic
import Mathlib.Data.List.Induction
import Mathlib.Data.List.Count
import Mathlib.Logic.Function.Basic
import Mathlib.Tactic.Linarith

/-!
# Chat transcripts and tool calls

A chat transcript is a sequence of events: system, user and assistant messages, tool
calls with identifiers, and tool results. This module fixes what a valid transcript
is and what rendering it to a model's input means.

**The protocol** (`Allowed`, `Valid`) is stated per event in terms of the transcript
before it:

* a call needs a fresh identifier, and may start only when no call is pending or
  directly after another call (a batch of parallel calls);
* a result must answer a pending call;
* a system, user or assistant message may appear only when no call is pending.

`check_iff_valid` proves that a single left-to-right pass with a small state decides
exactly this property. `valid_counts` and `result_has_call` derive the matching
discipline: each identifier is called at most once, answered at most once, and
answered only after it is called.

**Rendering.**

* `parse_render`: rendering with out-of-band markers (special symbols that cannot
  occur in content) is invertible, so the conversation can be recovered from the
  model's input, and `render_encode_injective` carries this through any injective
  tokenizer such as BPE;
* **negative result** `inBand_not_injective`: *every* template that writes its
  markers in the same alphabet as the content is ambiguous. For any choice of
  opening and closing strings, a single user message can render to the same input
  as a user message followed by an assistant message. Role separation therefore
  holds only at the token level, and only if user text can never produce a special
  token.
-/

namespace LLM.Chat

universe u

variable {C : Type u}

/-- The kind of an event, with the call identifier where there is one. -/
inductive Head
  | system
  | user
  | assistant
  | call (id : ℕ)
  | result (id : ℕ)
  deriving DecidableEq, Repr

/-- An event: a kind and a body over the content alphabet `C`. -/
structure Event (C : Type u) where
  /-- The kind of event. -/
  head : Head
  /-- The content. -/
  body : List C
  deriving DecidableEq

/-- The number of calls with identifier `i`. -/
def calls (i : ℕ) (es : List (Event C)) : ℕ := es.countP (fun e => e.head == .call i)

/-- The number of results with identifier `i`. -/
def results (i : ℕ) (es : List (Event C)) : ℕ := es.countP (fun e => e.head == .result i)

/-- Whether a kind is a call. -/
def Head.isCall : Head → Bool
  | .call _ => true
  | _ => false

/-- Whether the last event is a call. -/
def lastIsCall (es : List (Event C)) : Bool :=
  match es.getLast? with
  | some e => e.head.isCall
  | none => false

/-- No call is waiting for its result. -/
def NoPending (pre : List (Event C)) : Prop := ∀ i, results i pre = calls i pre

/-- The protocol: may event `e` follow the transcript `pre`? -/
def Allowed (pre : List (Event C)) (e : Event C) : Prop :=
  match e.head with
  | .call i => calls i pre = 0 ∧ (NoPending pre ∨ lastIsCall pre = true)
  | .result i => calls i pre = 1 ∧ results i pre = 0
  | _ => NoPending pre

/-- A transcript is valid when each event is allowed after the events before it. -/
def Valid (es : List (Event C)) : Prop :=
  ∀ (k : ℕ) (hk : k < es.length), Allowed (es.take k) es[k]

/-- A transcript is complete when it is valid and no call is pending. -/
def Complete (es : List (Event C)) : Prop := Valid es ∧ NoPending es

lemma valid_nil : Valid ([] : List (Event C)) := fun k hk => absurd hk (by simp)

lemma valid_append_singleton (pre : List (Event C)) (e : Event C) :
    Valid (pre ++ [e]) ↔ Valid pre ∧ Allowed pre e := by
  constructor
  · intro h
    refine ⟨fun k hk => ?_, ?_⟩
    · have := h k (by simp; omega)
      rwa [List.take_append_of_le_length (by omega), List.getElem_append_left hk] at this
    · have := h pre.length (by simp)
      simpa using this
  · rintro ⟨hpre, he⟩ k hk
    simp only [List.length_append, List.length_singleton] at hk
    rcases Nat.lt_or_ge k pre.length with hlt | hge
    · rw [List.take_append_of_le_length hlt.le, List.getElem_append_left hlt]
      exact hpre k hlt
    · have : k = pre.length := by omega
      subst this
      simpa using he

/-! ### The checker -/

/-- The checker's state: identifiers called, identifiers answered, and whether the
last event was a call. -/
structure St where
  /-- Identifiers that have been called. -/
  called : List ℕ
  /-- Identifiers that have been answered. -/
  answered : List ℕ
  /-- Whether the last event was a call. -/
  lastCall : Bool

/-- No call is pending, as the checker sees it. -/
def St.noPending (st : St) : Bool := st.called.all (fun i => decide (i ∈ st.answered))

/-- One step of the checker. -/
def stepS (st : St) (e : Event C) : Option St :=
  match e.head with
  | .call i =>
      if i ∉ st.called ∧ (st.noPending = true ∨ st.lastCall = true) then
        some ⟨i :: st.called, st.answered, true⟩
      else none
  | .result i =>
      if i ∈ st.called ∧ i ∉ st.answered then some ⟨st.called, i :: st.answered, false⟩
      else none
  | _ => if st.noPending = true then some ⟨st.called, st.answered, false⟩ else none

/-- Run the checker over a transcript. -/
def run (es : List (Event C)) : Option St :=
  es.foldl (fun acc e => acc.bind (fun st => stepS st e)) (some ⟨[], [], false⟩)

/-- Accept or reject a transcript. -/
def check (es : List (Event C)) : Bool := (run es).isSome

/-- The invariant connecting the checker's state to the transcript read so far. -/
structure Inv (pre : List (Event C)) (st : St) : Prop where
  called : ∀ i, i ∈ st.called ↔ 0 < calls i pre
  answered : ∀ i, i ∈ st.answered ↔ 0 < results i pre
  calls_le : ∀ i, calls i pre ≤ 1
  results_le : ∀ i, results i pre ≤ calls i pre
  last : st.lastCall = lastIsCall pre

lemma inv_nil : Inv ([] : List (Event C)) ⟨[], [], false⟩ :=
  ⟨by simp [calls], by simp [results], by simp [calls], by simp [calls, results], rfl⟩

lemma noPending_iff {pre : List (Event C)} {st : St} (h : Inv pre st) :
    st.noPending = true ↔ NoPending pre := by
  simp only [St.noPending, List.all_eq_true, decide_eq_true_eq, NoPending]
  constructor
  · intro hall i
    have hc := h.calls_le i
    have hr := h.results_le i
    rcases Nat.eq_zero_or_pos (calls i pre) with h0 | hpos
    · omega
    · have := (h.answered i).mp (hall i ((h.called i).mpr hpos))
      omega
  · intro hnp i hi
    have := (h.called i).mp hi
    rw [h.answered, hnp i]
    exact this

lemma calls_snoc (i : ℕ) (pre : List (Event C)) (e : Event C) :
    calls i (pre ++ [e]) = calls i pre + if e.head = .call i then 1 else 0 := by
  simp [calls, List.countP_append, List.countP_singleton]

lemma results_snoc (i : ℕ) (pre : List (Event C)) (e : Event C) :
    results i (pre ++ [e]) = results i pre + if e.head = .result i then 1 else 0 := by
  simp [results, List.countP_append, List.countP_singleton]

lemma lastIsCall_snoc (pre : List (Event C)) (e : Event C) :
    lastIsCall (pre ++ [e]) = e.head.isCall := by
  simp [lastIsCall, List.getLast?_append]

/-- The checker's step agrees with the protocol and maintains the invariant. -/
lemma stepS_spec {pre : List (Event C)} {st : St} (h : Inv pre st) (e : Event C) :
    match stepS st e with
    | none => ¬ Allowed pre e
    | some st' => Allowed pre e ∧ Inv (pre ++ [e]) st' := by
  rcases e with ⟨hd, b⟩
  have hnp := noPending_iff h
  cases hd
  case call j =>
    simp only [stepS, Allowed]
    have hcj : j ∉ st.called ↔ calls j pre = 0 := by
      rw [h.called]; omega
    split_ifs with hc
    · have hal : calls j pre = 0 ∧ (NoPending pre ∨ lastIsCall pre = true) :=
        ⟨hcj.mp hc.1, hc.2.imp hnp.mp (fun hl => h.last ▸ hl)⟩
      refine ⟨hal, ⟨?_, ?_, ?_, ?_, ?_⟩⟩
      · intro i
        rw [calls_snoc, List.mem_cons, h.called]
        by_cases hij : i = j
        · subst hij; simp
        · simp [hij, Ne.symm hij]
      · intro i
        rw [results_snoc, h.answered]
        simp
      · intro i
        rw [calls_snoc]
        by_cases hij : i = j
        · subst hij; simp [hal.1]
        · simp [Ne.symm hij, h.calls_le i]
      · intro i
        rw [calls_snoc, results_snoc]
        have := h.results_le i
        split_ifs <;> simp_all
      · rw [lastIsCall_snoc]; rfl
    · intro hal
      apply hc
      exact ⟨hcj.mpr hal.1, hal.2.imp hnp.mpr (fun hl => h.last ▸ hl)⟩
  case result j =>
    simp only [stepS, Allowed]
    split_ifs with hc
    · have hc1 : calls j pre = 1 := by
        have := (h.called j).mp hc.1; have := h.calls_le j; omega
      have hr0 : results j pre = 0 := by
        have := hc.2; rw [h.answered] at this; omega
      refine ⟨⟨hc1, hr0⟩, ⟨?_, ?_, ?_, ?_, ?_⟩⟩
      · intro i; rw [calls_snoc, h.called]; simp
      · intro i
        rw [results_snoc, List.mem_cons, h.answered]
        by_cases hij : i = j
        · subst hij; simp
        · simp [hij, Ne.symm hij]
      · intro i; rw [calls_snoc]; simpa using h.calls_le i
      · intro i
        rw [calls_snoc, results_snoc]
        by_cases hij : i = j
        · subst hij; simp [hc1, hr0]
        · simp [Ne.symm hij, h.results_le i]
      · rw [lastIsCall_snoc]; rfl
    · rintro ⟨hc1, hr0⟩
      apply hc
      refine ⟨(h.called j).mpr (by omega), ?_⟩
      rw [h.answered]; omega
  all_goals
    simp only [stepS, Allowed]
    split_ifs with hc
    · refine ⟨hnp.mp hc, ⟨?_, ?_, ?_, ?_, ?_⟩⟩
      · intro i; rw [calls_snoc, h.called]; simp
      · intro i; rw [results_snoc, h.answered]; simp
      · intro i; rw [calls_snoc]; simpa using h.calls_le i
      · intro i; rw [calls_snoc, results_snoc]; simpa using h.results_le i
      · rw [lastIsCall_snoc]; rfl
    · exact fun hal => hc (hnp.mpr hal)

lemma run_snoc (pre : List (Event C)) (e : Event C) :
    run (pre ++ [e]) = (run pre).bind (fun st => stepS st e) := by
  simp [run, List.foldl_append]

/-- The checker's verdict and state, for every transcript. -/
lemma run_spec (es : List (Event C)) :
    match run es with
    | none => ¬ Valid es
    | some st => Valid es ∧ Inv es st := by
  induction es using List.reverseRecOn with
  | nil => exact ⟨valid_nil, inv_nil⟩
  | append_singleton pre e ih =>
    rw [run_snoc]
    rcases hrun : run pre with _ | st
    · rw [hrun] at ih
      intro hv
      exact ih ((valid_append_singleton pre e).mp hv).1
    · rw [hrun] at ih
      obtain ⟨hv, hinv⟩ := ih
      have hs := stepS_spec hinv e
      simp only [Option.bind_some]
      rcases hstep : stepS st e with _ | st'
      · rw [hstep] at hs
        intro hv'
        exact hs ((valid_append_singleton pre e).mp hv').2
      · rw [hstep] at hs
        exact ⟨(valid_append_singleton pre e).mpr ⟨hv, hs.1⟩, hs.2⟩

/-- **The checker decides the protocol.** -/
theorem check_iff_valid (es : List (Event C)) : check es = true ↔ Valid es := by
  have := run_spec es
  unfold check
  rcases h : run es with _ | st
  · rw [h] at this; simp [this]
  · rw [h] at this; simp [this.1]

/-- The checker decides completeness as well. -/
theorem complete_iff (es : List (Event C)) :
    Complete es ↔ ∃ st, run es = some st ∧ st.noPending = true := by
  have := run_spec es
  constructor
  · rintro ⟨hv, hnp⟩
    rcases h : run es with _ | st
    · rw [h] at this; exact absurd hv this
    · rw [h] at this; exact ⟨st, rfl, (noPending_iff this.2).mpr hnp⟩
  · rintro ⟨st, h, hnp⟩
    rw [h] at this
    exact ⟨this.1, (noPending_iff this.2).mp hnp⟩

instance : DecidablePred (Valid (C := C)) := fun es =>
  decidable_of_iff _ (check_iff_valid es)

/-- **Matching discipline.** In a valid transcript every identifier is called at most
once and answered at most as often as it is called. In a complete one, calls and
results correspond one to one. -/
theorem valid_counts {es : List (Event C)} (h : Valid es) (i : ℕ) :
    results i es ≤ calls i es ∧ calls i es ≤ 1 := by
  have := run_spec es
  rcases hr : run es with _ | st
  · rw [hr] at this; exact absurd h this
  · rw [hr] at this; exact ⟨this.2.results_le i, this.2.calls_le i⟩

/-- **Every result answers an earlier call.** -/
theorem result_has_call {es : List (Event C)} (h : Valid es) (k : ℕ) (hk : k < es.length)
    (i : ℕ) (hres : es[k].head = .result i) :
    ∃ j < k, ∃ hj : j < es.length, es[j].head = .call i := by
  have hal := h k hk
  simp only [Allowed, hres] at hal
  have hpos : 0 < calls i (es.take k) := by omega
  unfold calls at hpos
  obtain ⟨e, he, hce⟩ := List.countP_pos_iff.mp hpos
  obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem he
  simp only [List.length_take] at hj
  refine ⟨j, by omega, by omega, ?_⟩
  rw [List.getElem_take] at hce
  simpa using hce

/-! ### Rendering -/

/-- Symbols of a model's input: content symbols and out-of-band markers. -/
inductive Sym (A : Type u)
  | ch (a : A)
  | open (h : Head)
  | close
  deriving DecidableEq

/-- Render one event with out-of-band markers. -/
def renderOne (e : Event C) : List (Sym C) := .open e.head :: e.body.map .ch ++ [.close]

/-- Render a transcript. -/
def render (es : List (Event C)) : List (Sym C) := es.flatMap renderOne

/-- Read a body up to its closing marker. -/
def takeBody : List (Sym C) → Option (List C × List (Sym C))
  | [] => none
  | .close :: rest => some ([], rest)
  | .ch c :: rest => (takeBody rest).map (fun br => (c :: br.1, br.2))
  | .open _ :: _ => none

/-- Parse a rendered transcript, with fuel. -/
def parseFuel : ℕ → List (Sym C) → Option (List (Event C))
  | _, [] => some []
  | 0, _ :: _ => none
  | n + 1, .open h :: rest =>
      match takeBody rest with
      | none => none
      | some (b, rest') => (parseFuel n rest').map (⟨h, b⟩ :: ·)
  | _ + 1, _ :: _ => none

/-- Parse a rendered transcript. -/
def parse (s : List (Sym C)) : Option (List (Event C)) := parseFuel s.length s

lemma takeBody_render (b : List C) (rest : List (Sym C)) :
    takeBody (b.map .ch ++ .close :: rest) = some (b, rest) := by
  induction b with
  | nil => rfl
  | cons c b ih => simp [takeBody, ih]

lemma parseFuel_render (es : List (Event C)) : ∀ n, es.length ≤ n → parseFuel n (render es) = some es := by
  induction es with
  | nil => intro n _; cases n <;> rfl
  | cons e es ih =>
    intro n hn
    obtain ⟨n, rfl⟩ : ∃ m, n = m + 1 := ⟨n - 1, by simp at hn; omega⟩
    simp only [render, List.flatMap_cons, renderOne, List.cons_append, List.append_assoc,
      parseFuel, takeBody_render, List.nil_append]
    have := ih n (by simp at hn; omega)
    unfold render at this
    rw [this]
    rfl

lemma length_le_render (es : List (Event C)) : es.length ≤ (render es).length := by
  induction es with
  | nil => simp [render]
  | cons e es ih => simp [render, renderOne] at *; omega

/-- **Out-of-band rendering is invertible.** -/
theorem parse_render (es : List (Event C)) : parse (render es) = some es :=
  parseFuel_render es _ (length_le_render es)

theorem render_injective : Function.Injective (render (C := C)) := by
  intro es es' h
  have := parse_render es
  rw [h, parse_render] at this
  exact (Option.some.inj this).symm

/-- Apply a map to every body, for instance a tokenizer from characters to tokens. -/
def Event.mapBody {D : Type*} (f : List C → List D) (e : Event C) : Event D := ⟨e.head, f e.body⟩

/-- **Token-level role separation.** Encoding every body with an injective tokenizer
and rendering with out-of-band special tokens is injective. -/
theorem render_encode_injective {D : Type*} (enc : List C → List D) (henc : Function.Injective enc) :
    Function.Injective (fun es : List (Event C) => render (es.map (Event.mapBody enc))) := by
  intro es es' h
  have h' := render_injective h
  have hinj : Function.Injective (Event.mapBody (C := C) enc) := by
    rintro ⟨h1, b1⟩ ⟨h2, b2⟩ heq
    simp only [Event.mapBody, Event.mk.injEq] at heq
    rw [heq.1, henc heq.2]
  exact List.map_injective_iff.mpr hinj h'

/-- An in-band template writes its markers in the content alphabet. -/
def renderInBand (opener : Head → List C) (closer : List C) (es : List (Event C)) : List C :=
  es.flatMap (fun e => opener e.head ++ e.body ++ closer)

/-- **Every in-band template is ambiguous.** Whatever strings open and close a
message, the single user message whose text is `closer ++ opener assistant`
renders to the same string as an empty user message followed by an empty assistant
message. -/
theorem inBand_not_injective (opener : Head → List C) (closer : List C) :
    ¬ Function.Injective (renderInBand opener closer) := by
  intro hinj
  have h := @hinj [⟨.user, closer ++ opener .assistant⟩] [⟨.user, []⟩, ⟨.assistant, []⟩]
    (by simp [renderInBand])
  simp at h

end LLM.Chat
