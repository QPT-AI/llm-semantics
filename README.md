# llm-semantics

This repository holds a mathematical semantics of the inference side of large
language models: how text becomes tokens, how attention reads a context, how a
model's next-token distribution becomes a sampled token, how a conversation becomes
a model's input, and how autoregressive and masked-diffusion decoders generate.
The semantics is formalized in Lean 4 and proved against Mathlib. From the same
definitions the repository builds an executable oracle: a program that reads small
case files, evaluates them under the proved meaning in exact rational arithmetic,
and prints the results in a fixed row format. The oracle exists so that an
implementation of any of these components, written in any language, can be tested
against the mathematics of the subject rather than against the behavior of another
implementation. The harness in `harness/` does this today for Hugging Face
`tokenizers`, `tiktoken`, and the samplers of llama.cpp.

The repository is modeled on
[ledger-semantics](https://github.com/ledger/ledger-semantics), which does the same
for double-entry accounting, and it follows the same discipline: denotational design
as Conal Elliott teaches it ("Denotational design with type class morphisms", 2009).
One begins by saying precisely what a thing means, then requires every operation to
preserve that meaning, and finally lets the implementation be whatever the equations
force it to be. A sampling filter means conditioning a distribution on a set. A
KV-cached decoder means causal attention over the whole prefix. A FlashAttention
kernel means a softmax-weighted average. A chat template means an injective encoding
of a transcript. The theorems say when an implementation keeps that meaning, and
several of them say when no implementation can.

## The mathematics

Every object in the theory is a finite probability distribution or a map that
produces one. `Dist 𝕜 α` is a distribution on a finite type with probabilities in an
arbitrary linearly ordered field `𝕜`. The oracle instantiates `𝕜 := ℚ`, so its
arithmetic is exact; the analytic statements instantiate `𝕜 := ℝ`. A single operation,
`Dist.restrict`, conditions a distribution on a set of positive mass. Every logit
mask, every sampling filter and every grammar constraint acts through it.
`Dist.restrict_restrict` states that conditioning twice is conditioning on the
intersection, and most of the algebra below is a consequence. Finite types and
stochastic maps form a category, `FinStoch 𝕜`, given as a Mathlib `Category`
instance whose composition is the bind of the finite distribution monad (`LLM/Dist.lean`).

Softmax is defined over an *exponential character*: any strictly increasing map from
an ordered group of scores into the positive elements of `𝕜` that turns addition into
multiplication. `Real.exp` is one (`LLM/Real.lean`), and `z ↦ 2^z` from integers to
rationals is another (`ExpChar.base2`). Every theorem about softmax and attention is
proved for an arbitrary character, so it holds for the textbook softmax and for the
exact one the oracle evaluates, by the same proof. `softmax_add_const` licenses the
"subtract the maximum" trick. `softmaxOn_eq_restrict` states that setting logits to
`-∞` outside a set and then taking softmax is conditioning on that set, so masking
logits and filtering probabilities are the same operation. `topK_temperature` states
that top-k keeps the same tokens at every positive temperature
(`LLM/Softmax.lean`).

**Sampling** (`LLM/Sampling.lean`, `LLM/Counterexamples.lean`). Top-k, nucleus
(top-p) and min-p are each `restrict` to a kept set, and every kept set contains a
most probable token, so no filter can empty a distribution. The filters obey an
exact algebra: `topK_topK` (top-k after top-k′ is top-min), `minP_minP` (the stricter
threshold wins), `topK_minP_comm`, `topK_sharpen_comm` (top-k commutes with
temperature), `topK_idem` and `minP_idem`. The laws one might expect beyond these
are false, with explicit witnesses checked by evaluation in the Lean kernel: top-p is
not idempotent, and it commutes neither with top-k nor with temperature; min-p does
not commute with temperature. The reorderings a sampler chain may make freely are
therefore exactly the ones that do not move top-p, or move min-p past temperature.
`invCdf_eq_iff` makes sampling itself precise. The uniform numbers that inverse-CDF
sampling maps to token `i` form an interval of length `p i`.

The theory also contains a negative result about ties, and it explains a split
visible in every sampler implementation. `no_equivariant_exact_topK` states that for
`0 < k < |V|` no selection rule that treats tokens symmetrically (commutes with
relabeling them) can keep exactly `k` tokens on every distribution.
`no_equivariant_exact_nucleus` states the same for "a smallest set of mass at least
`p`". The uniform distribution is the witness, because symmetry forces such a rule to
keep everything or nothing. An implementation must choose. The filters above keep
ties and are symmetric (`topKSet_equivariant`). The positional filters
`Dist.topKPos` and `Dist.topPPos` break ties by position, keep exactly `min k |V|`
tokens (`card_topKPosSet`), coincide with the tie-keeping filters whenever no two
tokens are tied, and are therefore not symmetric (`topKPosSet_not_equivariant`).
The implementations in use split along this line. In Hugging Face `transformers`
(read from the source of version 5.19.0, not measured), top-k removes only scores
strictly below the k-th largest, so it keeps ties, while top-p sorts and cuts by
cumulative position, so it breaks them. llama.cpp breaks ties by position in both, as
the measurements below confirm. Each is correct with respect to one of the two
definitions, and no single definition can serve both.

**Speculative decoding** (`LLM/Speculative.lean`). The joint law of (draft token,
emitted token) is written exactly as the algorithm runs: draft `x ~ q`, accept with
probability `min(1, p x / q x)`, otherwise draw from the normalized residual
`max(0, p - q)`. `specJoint_snd` proves losslessness: the emitted token has law `p`
whatever the draft `q` is. `accept_rate` proves the acceptance probability is
`1 - tv p q`. `coupling_le` and `spec_optimal` prove that no lossless verification
rule for a drafted token can accept more often, because every such rule is a coupling
of `q` and `p`.

**Attention** (`LLM/Attention.lean`). One attention read means a softmax-weighted
average of value vectors. `attend_convex` states that it obeys every linear bound
its values obey. `attend_const` states that attention cannot abstain: if every value
is `m`, the output is `m` whatever the scores are. Softmax weights always sum to
one, which is the property that motivates "attention sink" tokens and
softmax-plus-one variants. `selfAttn_equivariant` and `block_equivariant` prove that
self-attention, and any transformer block built from heads, pointwise maps and
residual sums, commutes with permuting positions. The negative result
`no_equivariant_first` follows: without positional information, no such network can
output the token at a fixed position. For sequences, `causalAttn_causal` proves that
an output never depends on later inputs. `kvDecode_eq` proves that incremental
decoding, which stores only past keys and values, equals recomputing causal
attention over the whole sequence, and `causal_extend` carries this to any causal
stack. The running `(max, normalizer, accumulator)` states of online softmax form a
commutative monoid, `Online E M`, given as a Mathlib `CommMonoid` instance, into which
the free monoid of (score, value) pairs maps by `FreeMonoid.lift`. `finalize_ofList`
proves that the reduction computes attention exactly. `finalize_blocks` and
`ofList_perm` prove that every tiling of the keys and every order of the tiles gives
the same result, which is the correctness of FlashAttention, split-K decoding and ring
attention. `ofList_max` proves that the running maximum is the maximum score, so
every exponential the kernel evaluates is at most one.

**Tokenization** (`LLM/Tokenization.lean`). The meaning of a token sequence is the
string it spells, and decoding is the free-monoid homomorphism
`FreeMonoid T →* FreeMonoid C` that `str` induces. The negative result here
parallels the one about pricing in the accounting semantics.
`homomorphic_no_compression` proves that if an encoder inverts decoding and respects
concatenation, every token it emits spells exactly one character, so it compresses
nothing. A tokenizer that compresses must violate `enc (a ++ b) = enc a ++ enc b`
(`compressing_not_homomorphic`). Tokenization therefore lives outside the
compositional core, and the "prompt boundary" artifacts of real systems follow from
this theorem, not from a defect in any one implementation. `decode_not_injective`
shows that one string has many token sequences, and `tokenizations_spec` and
`tokenizations_nodup` give an exact, duplicate-free enumeration of them, over which a
string's probability is a sum.

Byte-pair encoding is defined once, for an arbitrary pair-priority function, as
implementations run it: repeatedly merge the leftmost occurrence of the
lowest-priority adjacent pair. `bpeWith_decode` (lossless), `bpeWith_normal` (no
mergeable pair remains), `bpeWith_idem` and `stepWith_spec` hold for every priority.
Two priorities are in use. Hugging Face `tokenizers` and the original GPT-2 encoder
rank a pair by its merge *rule* (`bpe`). `tiktoken` ranks a pair by the
rank of the *token it would form* (`bpeTik`). `bpeWith_congr` and `hf_eq_tiktoken`
give a decidable condition on a merge table under which the two agree on every word,
and `hf_ne_tiktoken` proves that without it they do not: on the same table, `abc`
encodes as `a · bc` under one convention and as `abc` under the other. Whole-word
lookup, which `tiktoken` performs and Hugging Face performs with `ignore_merges`, is a
third convention, recorded as `lookupFirst`, and it changes results too
(`lookup_matters`). `bpe_not_shortest` shows that BPE does not find the shortest
tokenization.

**Chat** (`LLM/Chat.lean`). A transcript is a sequence of system, user and assistant
messages, tool calls with identifiers, and tool results. The protocol is stated per
event against the events before it: a call needs a fresh identifier and may start
only when nothing is pending or directly after another call (a parallel batch); a
result must answer a pending call; a message may appear only when nothing is pending.
`check_iff_valid` proves that a single left-to-right pass with a small state decides
exactly this property. `valid_counts` and `result_has_call` derive the matching
discipline: each identifier is called at most once, answered at most once, and only
after it is called. `parse_render` proves that rendering with out-of-band markers is
invertible, and `render_encode_injective` carries injectivity through any injective
tokenizer. The negative result is `inBand_not_injective`. *Every* template that
writes its role markers in the same alphabet as the content is ambiguous: whatever
the opening and closing strings, one user message renders to the same input as a
user message followed by an assistant message. Role separation exists only at the
token level, and only while user text cannot produce a special token.

**Autoregressive decoding** (`LLM/Decoding.lean`). A language model assigns every
prefix a distribution over the next token or the end of the sequence.
`mass_conservation` proves that at every depth the probability of having stopped plus
the probability of still generating is one. `alive_le_of_stop` proves that a stopping
probability bounded below by `ε` makes the model tight. The negative result
`stopped_lt_half` exhibits a model that stops after `n` tokens with probability
`1/(n+2)²` and never places half of its mass on finite strings, so an autoregressive
model need not define a distribution over strings at all. `greedy_not_map` shows that
following the most probable token does not find the most probable sequence, under any
tie-breaking.

**Masked diffusion** (`LLM/Diffusion.lean`). The forward process masks each token
independently. `forward` is a Mathlib `MonoidHom` from the multiplicative unit
interval into the endomorphisms of `Option V` in `FinStoch`: masking at keep rate `α`
and then `β` is masking at `αβ`. `qSeq_closed` gives the closed form of the sequence
process. `posterior_time_independent` proves that the optimal denoiser does not
depend on the noise level, since the posterior is the data distribution conditioned
on the unmasked tokens for every `0 < α < 1`. `reverse_bayes` proves that the
closed-form reverse step samplers use is the Bayes posterior. `chain_rule` and
`chain_rule_fin` prove that, for *any* order of positions, the product of exact
conditionals is the data probability, and `posterior_marginal` identifies those
conditionals with the denoiser's marginals. Unmasking one token at a time with the
exact denoiser therefore samples the data exactly, in every order. The negative
result `parallel_decoding_inexact` shows that unmasking two positions in one step
from their separate posteriors gives probability `1/4` to a sequence the data never
produces.

These claims are stated in the standard vocabulary. Stochastic maps form a Mathlib
`Category`, the forward diffusion process is a Mathlib `MonoidHom` into an
endomorphism monoid, online softmax is a Mathlib `CommMonoid` reached by
`FreeMonoid.lift`, decoding is a free-monoid homomorphism, and symmetry is stated with
`Equiv.Perm`.

The build turns every warning into an error, so a `sorry` fails it.
`scripts/Audit.lean` walks every declaration of the `LLM` modules, 853 of them, and
fails unless each depends only on propositional extensionality, choice, and quotient
soundness. The audit has been observed to fail: a planted `native_decide` proof is
rejected. Concrete counterexamples are proved by `decide`, which the kernel checks,
and never by `native_decide`.

## What an implementation must preserve

An implementation of any of these components can be held to the following, each of
which is a theorem above.

A sampler must act on a distribution only by conditioning it on a kept set that
contains a most probable token. It may reorder top-k, min-p and temperature only as
`topK_minP_comm` and `topK_sharpen_comm` allow, and it must not reorder top-p. It must
say whether it keeps ties or breaks them by position, because it cannot do both. A
speculative decoder must emit exactly the target distribution, and its acceptance
rate cannot exceed `1 - tv(p, q)`. An attention kernel, however it tiles, blocks or
caches, must equal the softmax-weighted average. A KV cache must equal recomputation,
and positional information must enter explicitly if the model is to see order.

A tokenizer must be lossless (`decode ∘ encode = id`), it cannot be compositional if
it compresses, and it must state which BPE convention and which lookup convention it
follows, because they produce different tokens. A chat template must be injective,
and only out-of-band special tokens that user text cannot produce achieve that.
Tool-call transcripts must satisfy the protocol that `check` decides. A decoder that
reports sequence probabilities must conserve mass, and a model that is not tight does
not define a distribution over strings. A masked-diffusion sampler that unmasks
several positions at once is an approximation, and one that unmasks one position at a
time with an exact denoiser is not.

## The oracle

`LLM/Driver.lean` turns the semantics into a test instrument. It reads one or more
case files and prints, for each case, rows computed by the proved definitions over
`ℚ`. It runs through Lake:

```
lake env lean --run LLM/Driver.lean FILE.llm ...
```

A case file is a sequence of cases, each starting with a header `@KIND NAME`. Lines
starting with `#` are comments, and arguments are separated by single spaces.
Rationals are written `a/b`, as decimals, or as integers. `LLM/Oracle.lean`
documents each directive.

* `@sample` starts from `probs`, `weights`, or `logits2` (integer logits under the
  exact base-2 softmax) and applies `top-k`, `top-p`, `min-p`, `sharpen N`
  (temperature `1/N`) and `mask`. It prints `p|i|PROB` under the tie-keeping filters,
  `q|i|PROB` under positional tie-breaking, and `cdf|i|LO|HI` on request.
* `@speculative` takes `target` and `draft` and prints `rate`, `tv`, per-token
  `accept`, the `residual`, and the law of the emitted token `out`.
* `@bpe` takes `merge A B` lines and prints, for each `encode WORD`, the tokens
  under merge-rule ranks (`tokens`), with whole-word lookup (`tokens-lookup`), and
  under `tiktoken`'s convention (`tiktoken`). For `count WORD` it prints the number of
  tokenizations, and at the end it prints `agree|true` when the hypotheses of
  `hf_eq_tiktoken` hold for the case's vocabulary.
* `@chat` takes events and prints `valid|true` or `valid|false|K`, where `K` is the
  first offending event, and `complete|…`.
* `@attention` takes `key SCORE v…` lines and prints softmax `weight`s and the
  output `out`, and with `split B` also the result of the online-softmax monoid over
  blocks of `B` keys (`blocked`).
* `@diffusion` takes `data SEQ PROB` lines and an `observe` pattern with `_` for
  masked positions. It prints the `evidence`, the time-independent `post`erior of each
  data sequence, and the posterior `marg`inal of each masked position.

Each file's output begins with `== PATH` and each case with `-- NAME`. A rejected
case prints `!! ERROR NAME: reason` and nothing else, which keeps failure visible.
Every number is printed as an exact rational in lowest terms. The driver runs under
the Lean interpreter, and the build produces no native executable, so what executes
is a direct reading of the checked definitions. `examples/` holds a case file for
each kind together with the oracle's output (`*.expected`).

## Measuring an implementation against the oracle

The procedure of ledger-semantics transfers without alteration. A seeded
generator, `harness/gen.py`, writes case files. It writes inputs only and never
computes or records an expected result. It does favor inputs where implementations
are known to have conventions, such as tied logits, words that are themselves tokens,
and merge tables with several splits of one string. A small adapter per
implementation (`impl_hf.py`, `impl_tiktoken.py`, `impl_llama.py`) runs the same cases
through the implementation and prints the oracle's row format. `compare.py` puts
every case in exactly one named bin (agree, diverge, skip on either side, or missing),
so nothing is dropped silently. Normalization stays small enough to trust. Token
rows are compared as strings, and sampler rows require that a zero be matched by a
zero and that nonzero probabilities agree within `1e-6`, since llama.cpp holds
logits in `float32`.

Every comparator is observed to fail. `harness/run.sh` replays a planted defect for
each implementation: a tokenizer adapter that glues its last two tokens, and a
sampler adapter that inverts temperature. The run fails unless each defect is
detected.

The results below come from `harness/run.sh 600`. That is 2,400 generated cases per
corpus, run on Lean 4.34.1 and Mathlib v4.34.1 against `tokenizers` 0.23.3,
`tiktoken` 0.14.0, and llama.cpp at commit `a9d27ac` (2026-10-11).

| Implementation | Cases | Result |
|---|---|---|
| Hugging Face `tokenizers`, BPE, default options | 2,400 merge tables, 13,779 words | identical to `bpe` on every word |
| Hugging Face `tokenizers`, BPE, `ignore_merges=True` | same | identical to `hfEncodeLookup` on every word |
| `tiktoken` | same | identical to `tikEncode` on every word |
| llama.cpp `top_k`, `top_p`, `min_p`, `temp` | 2,400 sampler chains | 1,216 tie-free chains: same kept tokens in every case, probabilities within `5.1e-8`; 1,184 chains with ties: 939 agree with the tie-keeping filters, the other 245 agree with the positional filters up to a permutation of tied tokens; 0 unexplained |

The log of this run is `harness/results/2026-10-11.txt`. The generator is seeded,
so the same command reproduces the same corpora and the same counts.

The divergences a comparison surfaces are its value, and these three surfaced two.
First, `tiktoken` initially disagreed with the oracle's concatenation-rank BPE on
words that are themselves vocabulary tokens. The disagreement was a convention, the
whole-word lookup, which is now recorded as `lookupFirst` with its own theorems,
after which agreement is exact. Second, on the same random merge tables the two
libraries disagree with each other on 439 of the 13,779 words (3.2%), and with
`ignore_merges` on 279. Every one of these disagreements is predicted by the oracle,
since it reproduces each library exactly, and `hf_eq_tiktoken` states when they
cannot occur. llama.cpp's divergences from the tie-keeping filters are not defects.
They are the positional convention, which `explain_ties.py` checks exactly: an
implementation that keeps exactly `k` tokens may differ from `q` only by which of
several tied tokens it keeps, so the multiset of (logit, probability) pairs over its
kept tokens must equal that of `q`.

These comparisons cover tokenizers and sampler chains. They do not yet cover
attention kernels, speculative decoders, chat-template renderers, or diffusion
samplers. The oracle already evaluates all of these, and a passing result is
meaningful only together with this statement of its limits.

## Building the repository

Install Lean 4.34.1 through elan, then run `lake exe cache get` followed by
`lake build`. The first command downloads prebuilt Mathlib, and the second compiles
this repository's modules, finishing in about a minute once the cache is present. The
lakefile turns warnings into errors, so an unproved theorem fails the build. Then

```
lake env lean scripts/Audit.lean                          # axiom audit
lake env lean --run LLM/Driver.lean examples/*.llm        # the oracle
pip install tokenizers tiktoken
LLAMA_LIB=/path/to/libllama.so harness/run.sh 600         # the measurements
```

`.github/workflows/ci.yml` runs the build, the audit, the examples against their
expected outputs, and the tokenizer measurements.

## Provenance and present limits

The theory was written for this repository, after the structure of ledger-semantics.
Its present limits are stated here so that none of the claims above is read more
broadly than it is proved.

Analysis on the reals appears only through the exponential character. There is no
measure theory, so sampling is made precise by the interval statement of
`invCdf_eq_iff`, and tightness by masses at finite depth. The oracle's temperature is
`1/N` for natural `N`, which keeps it within the rationals. The theorems hold for every
temperature, but the oracle cannot evaluate an irrational softmax. Speculative
decoding is proved for one drafted token; the sequence version follows by induction
and is not yet formalized. Nothing is proved about floating-point arithmetic.
Implementations are compared against exact values within a stated tolerance, and
decisions that sit exactly on a float boundary are possible in principle. None
occurred in these runs. The diffusion theory covers absorbing (masked) diffusion, not
Gaussian diffusion. The chat protocol models tool calling in the common
request-and-result form and is not any vendor's specification. Finally, the CI
workflow has not been run from this repository; the commands it runs were run locally.

## License

BSD three-clause, as `LICENSE.md` states.
