import Lake
open Lake DSL

package «llm-semantics» where
  leanOptions := #[
    ⟨`autoImplicit, false⟩,
    ⟨`relaxedAutoImplicit, false⟩,
    ⟨`warningAsError, true⟩
  ]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.34.1"

/-- The theory and the oracle's definitions. -/
@[default_target]
lean_lib LLM where
  roots := #[`LLM]
  globs := #[.andSubmodules `LLM]
