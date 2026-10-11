import LLM

/-!
# The oracle

The oracle reads small case files and evaluates them with the definitions the
theorems are about, over the rationals. It performs no floating-point arithmetic
and no rounding, so every number it prints is exact.

## Input

A case file is a sequence of cases. A case starts with a header `@KIND NAME`.
Lines starting with `#` are comments and blank lines are ignored. Arguments are
separated by single spaces. Rationals are written `a/b`, `-1.25`, or `3`.

* `@sample`: `probs r…` | `weights w…` | `logits2 z…` (integer logits under the
  exact base-2 softmax) to start; then any of `top-k K`, `top-p P`, `min-p M`,
  `sharpen N` (temperature `1/N`), `mask i…`, `cdf`, `emit`. The chain is evaluated
  twice: with the tie-keeping filters (rows `p`) and with positional tie-breaking
  (rows `q`).
* `@speculative`: `target r…`, `draft r…`.
* `@bpe`: `merge A B` (rank is order of appearance), `encode WORD`, `count WORD`.
* `@chat`: `system T`, `user T`, `assistant T`, `call ID T`, `result ID T`.
* `@attention`: `key S v…` (integer base-2 score, rational value vector),
  optionally `split B` (also evaluate through the online-softmax monoid in blocks of
  `B` keys).
* `@diffusion`: `data SEQ P` (a sequence of one-character symbols and its
  probability), `observe PATTERN` (`_` marks a masked position).

## Output

Each case prints `-- NAME` followed by rows of the form `TAG|FIELD|…`, or a single
line `!! ERROR NAME: reason`. The rows are described with each runner below.
-/

namespace LLM.Oracle

open Finset

/-! ### Lexical layer -/

/-- Parse a rational written `a/b`, as a decimal, or as an integer. -/
def parseRat? (s : String) : Option ℚ :=
  match s.splitOn "/" with
  | [a, b] => do
      let n ← a.toInt?
      let d ← b.toNat?
      if d = 0 then none else some ((n : ℚ) / (d : ℚ))
  | [a] =>
      match a.splitOn "." with
      | [i] => (i.toInt?).map (fun z => (z : ℚ))
      | [i, f] => do
          let neg := i.startsWith "-"
          let iabs := if neg then (i.drop 1).toString else i
          let ip ← if iabs.isEmpty then some 0 else iabs.toNat?
          if f.isEmpty || f.toList.any (fun c => !c.isDigit) then none else
          let fp ← f.toNat?
          let v : ℚ := (ip : ℚ) + (fp : ℚ) / ((10 : ℚ) ^ f.length)
          some (if neg then -v else v)
      | _ => none
  | _ => none

/-- Print a rational in lowest terms. -/
def fmtRat (q : ℚ) : String :=
  if q.den = 1 then toString q.num else toString q.num ++ "/" ++ toString q.den

/-- The space-separated words of a line. -/
def words (s : String) : List String := (s.splitOn " ").filter (· ≠ "")

/-- The text after the first `k` single-space-separated fields. -/
def restAfter (s : String) (k : ℕ) : String := " ".intercalate ((s.splitOn " ").drop k)

def parseRats (xs : List String) : Except String (List ℚ) :=
  xs.mapM (fun x => match parseRat? x with
    | some q => .ok q
    | none => .error s!"not a rational: {x}")

def parseNat (x : String) : Except String ℕ :=
  match x.toNat? with
  | some n => .ok n
  | none => .error s!"not a natural number: {x}"

def parseInt (x : String) : Except String ℤ :=
  match x.toInt? with
  | some n => .ok n
  | none => .error s!"not an integer: {x}"

/-! ### Cases -/

/-- A case: kind, name, and numbered body lines. -/
structure Case where
  kind : String
  name : String
  lines : List (ℕ × String)

/-- Split a file into cases. -/
def splitCases (src : String) : Except String (List Case) := do
  let raw := (src.splitOn "\n").map (fun l => if l.endsWith "\r" then (l.dropEnd 1).toString else l)
  let mut cases : Array Case := #[]
  let mut cur : Option Case := none
  let mut ln := 0
  for l in raw do
    ln := ln + 1
    if l.startsWith "#" || (words l).isEmpty then continue
    if l.startsWith "@" then
      if let some c := cur then cases := cases.push { c with lines := c.lines.reverse }
      match words l with
      | k :: rest => cur := some ⟨(k.drop 1).toString, " ".intercalate rest, []⟩
      | [] => throw s!"line {ln}: empty header"
    else
      match cur with
      | some c => cur := some { c with lines := (ln, l) :: c.lines }
      | none => throw s!"line {ln}: content before the first @header"
  if let some c := cur then cases := cases.push { c with lines := c.lines.reverse }
  return cases.toList

/-! ### Distributions over `Fin n` -/

/-- Build a distribution from a list of exactly `n` rationals, checking the axioms. -/
def mkProbs (n : ℕ) (xs : List ℚ) : Except String (Dist ℚ (Fin n)) :=
  if hl : xs.length = n then
    let v : Fin n → ℚ := fun i => xs[i.val]'(hl ▸ i.isLt)
    if h : (∀ i, 0 ≤ v i) ∧ ∑ i, v i = 1 then .ok ⟨v, h.1, h.2⟩
    else .error "probabilities must be nonnegative and sum to 1"
  else .error s!"expected {n} values, got {xs.length}"

/-- Rows `tag|i|p` for every token. -/
def distRows {n : ℕ} (tag : String) (d : Dist ℚ (Fin n)) : List String :=
  (List.finRange n).map (fun i => s!"{tag}|{i.val}|{fmtRat (d.prob i)}")

/-! ### `@sample` -/

/-- The sampler state: the same chain under the tie-keeping filters (`p`) and under
positional tie-breaking (`q`). -/
abbrev SampleState := Σ n, Dist ℚ (Fin n) × Dist ℚ (Fin n)

/-- One sampling directive. Rows: `p|i|PROB` and `q|i|PROB` on `emit` and at the end of
the case (`p` keeps ties, `q` breaks them by position), and `cdf|i|LO|HI` on `cdf`
(token `i` is drawn for `u ∈ [LO, HI)` under `p`). -/
def sampleStep (st : Option SampleState) (line : String) :
    Except String (Option SampleState × List String) := do
  let need : Except String SampleState :=
    match st with
    | some d => .ok d
    | none => .error "no distribution yet"
  let both (n : ℕ) (d : Dist ℚ (Fin n)) : Option SampleState := some ⟨n, d, d⟩
  match words line with
  | "probs" :: xs =>
      let qs ← parseRats xs
      let d ← mkProbs qs.length qs
      return (both _ d, [])
  | "weights" :: xs =>
      let qs ← parseRats xs
      let n := qs.length
      let w : Fin n → ℚ := fun i => qs[i.val]
      if h : (∀ i, 0 ≤ w i) ∧ 0 < ∑ i, w i then
        return (both n (Dist.normalize w h.1 h.2), [])
      else throw "weights must be nonnegative with positive total"
  | "logits2" :: xs =>
      let zs ← xs.mapM parseInt
      let n := zs.length
      if hn : 0 < n then
        have : Nonempty (Fin n) := ⟨⟨0, hn⟩⟩
        return (both n (softmax ExpChar.base2 (fun i : Fin n => zs[i.val])), [])
      else throw "logits2 needs at least one logit"
  | ["top-k", k] =>
      let ⟨n, d, e⟩ ← need
      let K ← parseNat k
      if hk : 0 < K then return (some ⟨n, d.topK K hk, e.topKPos K hk⟩, [])
      else throw "top-k needs k ≥ 1"
  | ["top-p", p] =>
      let ⟨n, d, e⟩ ← need
      let some P := parseRat? p | throw s!"not a rational: {p}"
      if hP : 0 < P then return (some ⟨n, d.topP P hP, e.topPPos P hP⟩, [])
      else throw "top-p needs p > 0"
  | ["min-p", m] =>
      let ⟨n, d, e⟩ ← need
      let some M := parseRat? m | throw s!"not a rational: {m}"
      if hM : M ≤ 1 then return (some ⟨n, d.minP M hM, e.minP M hM⟩, [])
      else throw "min-p needs m ≤ 1"
  | ["sharpen", k] =>
      let ⟨n, d, e⟩ ← need
      let N ← parseNat k
      return (some ⟨n, d.sharpen N, e.sharpen N⟩, [])
  | "mask" :: is =>
      let ⟨n, d, e⟩ ← need
      let idx ← is.mapM parseNat
      let S : Finset (Fin n) := univ.filter (fun i => i.val ∈ idx)
      if h : 0 < d.mass S then
        if h' : 0 < e.mass S then return (some ⟨n, d.restrict S h, e.restrict S h'⟩, [])
        else throw "mask keeps no probability mass"
      else throw "mask keeps no probability mass"
  | ["emit"] =>
      let ⟨n, d, e⟩ ← need
      return (some ⟨n, d, e⟩, distRows "p" d ++ distRows "q" e)
  | ["cdf"] =>
      let ⟨n, d, e⟩ ← need
      return (some ⟨n, d, e⟩, (List.finRange n).map (fun i =>
        s!"cdf|{i.val}|{fmtRat (cdf d.prob i.val)}|{fmtRat (cdf d.prob (i.val + 1))}"))
  | _ => throw s!"unknown sample directive: {line}"

def runSample (c : Case) : Except String (List String) := do
  let mut st : Option SampleState := none
  let mut out : Array String := #[]
  for (ln, line) in c.lines do
    match sampleStep st line with
    | .ok (st', rows) => st := st'; out := out ++ rows.toArray
    | .error e => throw s!"line {ln}: {e}"
  match st with
  | some ⟨_, d, e⟩ => return out.toList ++ distRows "p" d ++ distRows "q" e
  | none => throw "no distribution given"

/-! ### `@speculative` -/

/-- Rows: `rate|R` (acceptance probability), `tv|X`, `accept|i|A` (probability of
accepting draft `i`), `residual|i|P`, `out|i|P` (the law of the emitted token). -/
def runSpeculative (c : Case) : Except String (List String) := do
  let mut target : Option (List ℚ) := none
  let mut draft : Option (List ℚ) := none
  for (ln, line) in c.lines do
    match words line with
    | "target" :: xs => target := some (← (parseRats xs).mapError (s!"line {ln}: " ++ ·))
    | "draft" :: xs => draft := some (← (parseRats xs).mapError (s!"line {ln}: " ++ ·))
    | _ => throw s!"line {ln}: unknown speculative directive: {line}"
  let some t := target | throw "no target"
  let some q := draft | throw "no draft"
  let n := t.length
  let p ← mkProbs n t
  let q ← mkProbs n q
  let rate := ∑ x, q.prob x * Spec.accept p q x
  let joint := Spec.specJoint p q
  return [s!"rate|{fmtRat rate}", s!"tv|{fmtRat (Dist.tv p q)}"] ++
    (List.finRange n).map (fun i => s!"accept|{i.val}|{fmtRat (Spec.accept p q i)}") ++
    distRows "residual" (Spec.residual p q) ++
    (List.finRange n).map (fun i => s!"out|{i.val}|{fmtRat (∑ x, joint.prob (x, i))}")

/-! ### `@bpe` -/

/-- Spell a string token. -/
def showTok (t : List Char) : String := String.ofList t

/-- Rows: `tokens|WORD|t₁ t₂ …` under merge-rule ranks (Hugging Face, default
options), `tokens-lookup|WORD|…` with whole-word lookup first (Hugging Face with
`ignore_merges`), `tiktoken|WORD|…` (whole-word lookup, then concatenation ranks),
`count|WORD|N` (the number of token sequences over the vocabulary that spell
`WORD`), and at the end `agree|true` when the hypotheses of `Tok.hf_eq_tiktoken`
hold for the vocabulary, so that the two BPE cores provably agree on every word
over it. The vocabulary is
every character that occurs together with every token a merge names or produces. -/
def runBPE (c : Case) : Except String (List String) := do
  let mut pairs : Array (List Char × List Char) := #[]
  let mut wordsSeen : Array String := #[]
  let mut out : Array String := #[]
  for (ln, line) in c.lines do
    match words line with
    | ["merge", a, b] => pairs := pairs.push (a.toList, b.toList)
    | ["encode", w] =>
        wordsSeen := wordsSeen.push w
        let ms := pairs.toList
        let base := w.toList.map (fun ch => [ch])
        let hf := Tok.bpe (Tok.hfMerges ms) base
        let hfl := Tok.hfEncodeLookup ms w.toList
        let tik := Tok.tikEncode ms w.toList
        out := out.push s!"tokens|{w}|{" ".intercalate (hf.map showTok)}"
        out := out.push s!"tokens-lookup|{w}|{" ".intercalate (hfl.map showTok)}"
        out := out.push s!"tiktoken|{w}|{" ".intercalate (tik.map showTok)}"
    | ["count", w] =>
        wordsSeen := wordsSeen.push w
        let vocab := (w.toList.map (fun ch => [ch]) ++
          pairs.toList.flatMap (fun m => [m.1, m.2, m.1 ++ m.2])).eraseDups
        let n := (Tok.tokenizations vocab id w.toList).length
        out := out.push s!"count|{w}|{n}"
    | _ => throw s!"line {ln}: unknown bpe directive: {line}"
  let ms := pairs.toList
  let vocab : List (List Char) :=
    ((wordsSeen.toList.flatMap (fun w => w.toList)).map (fun ch => [ch]) ++
      ms.flatMap (fun m => (m.1.map (fun ch => [ch])) ++ (m.2.map (fun ch => [ch])) ++
        [m.1, m.2, m.1 ++ m.2])).eraseDups
  let closed := vocab.all (fun a => vocab.all (fun b =>
    match Tok.rankOf (Tok.hfMerges ms) a b with
    | some (_, t) => vocab.contains t
    | none => true))
  let agree := vocab.all (fun a => vocab.all (fun b =>
    Tok.rankOf (Tok.hfMerges ms) a b == Tok.concatRank (Tok.tikRank ms) a b))
  return out.toList ++ [s!"agree|{closed && agree}"]

/-! ### `@chat` -/

/-- Rows: `valid|true` or `valid|false|K` with `K` the first event that is not
allowed after the events before it, and `complete|true|false`. -/
def runChat (c : Case) : Except String (List String) := do
  let mut es : Array (Chat.Event Char) := #[]
  for (ln, line) in c.lines do
    match words line with
    | "system" :: _ => es := es.push ⟨.system, (restAfter line 1).toList⟩
    | "user" :: _ => es := es.push ⟨.user, (restAfter line 1).toList⟩
    | "assistant" :: _ => es := es.push ⟨.assistant, (restAfter line 1).toList⟩
    | "call" :: i :: _ =>
        let n ← (parseNat i).mapError (s!"line {ln}: " ++ ·)
        es := es.push ⟨.call n, (restAfter line 2).toList⟩
    | "result" :: i :: _ =>
        let n ← (parseNat i).mapError (s!"line {ln}: " ++ ·)
        es := es.push ⟨.result n, (restAfter line 2).toList⟩
    | _ => throw s!"line {ln}: unknown chat directive: {line}"
  let evs := es.toList
  let firstBad := (List.range evs.length).find? (fun k => !Chat.check (evs.take (k + 1)))
  let valid := match firstBad with
    | none => "valid|true"
    | some k => s!"valid|false|{k}"
  let complete := match Chat.run evs with
    | some st => st.noPending
    | none => false
  return [valid, s!"complete|{complete}"]

/-! ### `@attention` -/

/-- Rows: `weight|i|W` (softmax weight of key `i`), `out|j|V` (coordinate `j` of the
output), and, with `split B`, `blocked|j|V` computed through the online-softmax
monoid in blocks of `B` keys. -/
def runAttention (c : Case) : Except String (List String) := do
  let mut keys : Array (ℤ × List ℚ) := #[]
  let mut split : Option ℕ := none
  for (ln, line) in c.lines do
    match words line with
    | "key" :: s :: vs =>
        let z ← (parseInt s).mapError (s!"line {ln}: " ++ ·)
        let v ← (parseRats vs).mapError (s!"line {ln}: " ++ ·)
        keys := keys.push (z, v)
    | ["split", b] => split := some (← (parseNat b).mapError (s!"line {ln}: " ++ ·))
    | _ => throw s!"line {ln}: unknown attention directive: {line}"
  let ks := keys.toList
  let some (_, v0) := ks.head? | throw "no keys"
  let d := v0.length
  unless ks.all (fun k => k.2.length == d) do throw "value vectors differ in dimension"
  let xs : List (ℤ × (Fin d → ℚ)) := ks.map (fun k => (k.1, fun j => k.2.getD j.val 0))
  let n := ks.length
  if hn : n = 0 then throw "no keys" else
  have : Nonempty (Fin n) := ⟨⟨0, Nat.pos_of_ne_zero hn⟩⟩
  let w := softmax ExpChar.base2 (fun i : Fin n => (ks.getD i.val (0, [])).1)
  let out := Online.attendList ExpChar.base2 xs
  let rows := distRows "weight" w ++
    (List.finRange d).map (fun j => s!"out|{j.val}|{fmtRat (out j)}")
  match split with
  | none => return rows
  | some 0 => throw "split needs a positive block size"
  | some b =>
      let blocks := xs.toChunks b
      let st := (blocks.map (Online.ofList ExpChar.base2)).prod
      let res := Online.finalize ExpChar.base2 st
      return rows ++ (List.finRange d).map (fun j => s!"blocked|{j.val}|{fmtRat (res j)}")

/-! ### `@diffusion` -/

/-- The posterior with the evidence computed once. Equal to `Diffusion.posterior` by
definition. -/
def posteriorWith {ι V : Type*} [Fintype ι] [DecidableEq ι] [Fintype V] [DecidableEq V]
    (p : Dist ℚ (ι → V)) (y : ι → Option V) (Z : ℚ) (x : ι → V) : ℚ :=
  (if Diffusion.Consistent x y then p.prob x else 0) / Z

theorem posteriorWith_eq {ι V : Type*} [Fintype ι] [DecidableEq ι] [Fintype V] [DecidableEq V]
    (p : Dist ℚ (ι → V)) (y : ι → Option V) (x : ι → V) :
    posteriorWith p y (Diffusion.evidence p y) x = Diffusion.posterior p y x := rfl

/-- Rows: `evidence|Z`, `post|SEQ|P` for every data sequence, and `marg|i|c|P` for
every masked position `i` and symbol `c`. -/
def runDiffusion (c : Case) : Except String (List String) := do
  let mut data : Array (String × ℚ) := #[]
  let mut obs : Option String := none
  for (ln, line) in c.lines do
    match words line with
    | ["data", s, p] =>
        let some q := parseRat? p | throw s!"line {ln}: not a rational: {p}"
        data := data.push (s, q)
    | ["observe", s] => obs := some s
    | _ => throw s!"line {ln}: unknown diffusion directive: {line}"
  let dat := data.toList
  let some y0 := obs | throw "no observation"
  let some (s0, _) := dat.head? | throw "no data"
  let L := s0.length
  unless dat.all (fun e => e.1.length == L) && y0.length == L do
    throw "all sequences and the observation must have the same length"
  let alphabet := (dat.flatMap (fun e => e.1.toList)).eraseDups.mergeSort (fun a b => a ≤ b)
  let k := alphabet.length
  if hk : 0 < k then
    unless k ^ L ≤ 4096 do throw "sequence space too large for the oracle (k^L > 4096)"
    let sym : Char → Fin k := fun ch => ⟨alphabet.idxOf ch % k, Nat.mod_lt _ hk⟩
    let enc : String → (Fin L → Fin k) := fun s i => sym (s.toList.getD i.val ' ')
    let pr : (Fin L → Fin k) → ℚ := fun x =>
      ((dat.filter (fun e => decide (enc e.1 = x))).map Prod.snd).sum
    if h : (∀ x, 0 ≤ pr x) ∧ ∑ x, pr x = 1 then
      let p : Dist ℚ (Fin L → Fin k) := ⟨pr, h.1, h.2⟩
      let mut y : Fin L → Option (Fin k) := fun _ => none
      for i in List.finRange L do
        let ch := y0.toList.getD i.val '_'
        unless ch == '_' || alphabet.contains ch do throw s!"unknown symbol {ch} in observation"
        if ch != '_' then
          let v := sym ch
          y := Function.update y i (some v)
      let Z := Diffusion.evidence p y
      let post := posteriorWith p y Z
      let seqs := (dat.map Prod.fst).eraseDups
      let masked := (List.finRange L).filter (fun i => (y i).isNone)
      return [s!"evidence|{fmtRat Z}"] ++
        seqs.map (fun s => s!"post|{s}|{fmtRat (post (enc s))}") ++
        masked.flatMap (fun i => (List.finRange k).map (fun v =>
          s!"marg|{i.val}|{alphabet.getD v.val ' '}|{fmtRat (∑ x, if x i = v then post x else 0)}"))
    else throw "data probabilities must be nonnegative and sum to 1"
  else throw "empty alphabet"

/-! ### Dispatch -/

/-- Evaluate one case. -/
def runCase (c : Case) : List String :=
  let res := match c.kind with
    | "sample" => runSample c
    | "speculative" => runSpeculative c
    | "bpe" => runBPE c
    | "chat" => runChat c
    | "attention" => runAttention c
    | "diffusion" => runDiffusion c
    | k => .error s!"unknown case kind @{k}"
  match res with
  | .ok rows => s!"-- {c.name}" :: rows
  | .error e => [s!"!! ERROR {c.name}: {e}"]

/-- Evaluate a whole file. -/
def runFile (src : String) : List String :=
  match splitCases src with
  | .ok cases => cases.flatMap runCase
  | .error e => [s!"!! ERROR {e}"]

end LLM.Oracle
