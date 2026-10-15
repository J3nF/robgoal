# Style Guide

General coding principles, illustrated with generic examples.
Rules, not prose.

## Precedence

- A project's preexisting conventions and APIs win over every rule below, for naming, in-place markers and documentation of existing code.
  New code follows the project's style where it is consistent.
- Never rename or change a preexisting API without asking the user first.
- If there is no established API yet (e.g. a new or small project), ask the user whether changing the API is acceptable before doing so.
- If a prefix or suffix below collides with a language's established convention, pick another simple one, use it consistently, and document it once where it is introduced.

## Naming

- If the code implements a formula or spec from an external source (a paper, an RFC, a textbook), match variable names to that source's notation as closely as the language allows.
  This includes special symbols where supported (e.g. `ε`, `α` in a language with Unicode identifiers; `epsilon`, `alpha` otherwise).
- Give core domain objects short, fixed names and reuse them identically across every function signature — don't rename the same concept per function (e.g. always `graph`/`g`, never `graph` in one function and `network` in the next for the same object).
- Mark hypothetical/candidate values — computed under an assumed change that may be discarded — with a consistent suffix or prefix, distinct from the committed value (e.g. `total_candidate`/`total_tmp` vs. `total`).
- Adopt a verb-prefix convention and apply it uniformly:
  - One prefix for pure computation that returns a value without deciding anything (e.g. `get_*`, `compute_*`)
  - One prefix for functions that evaluate a candidate and decide whether to accept it, returning either the new or the original value (e.g. `check_*`, `attempt_*`)

## Function Shape

- One responsibility per function.
  Separate "compute a candidate" from "decide whether to apply it" from "apply it" — don't fuse these into one function.
- If several functions share parameters, use the same name and the same order for those parameters everywhere.
  Inconsistent ordering is a common source of call-site bugs.
- Prefer returning new values over in-place mutation.
  Where mutation is unavoidable, copy the input explicitly first, then mutate the copy — never mutate a caller's value silently.
- If functions mutate inputs in-place, make this visible: append "!" in Julia (and in Ruby, where a non-bang twin exists).
  In languages where "!" is not allowed in identifiers, append `_inplace`, unless the language already marks it at the call site (e.g. Rust `&mut`) or the ecosystem has a convention (e.g. NumPy `out=`).

## Documentation

- A *short summary* is a one-line summary of the object's purpose: why it exists and what problem it solves, not a mechanical restatement of its steps.
- Every public function gets a doc comment.
  Its layout depends on whether the language's doc tools already show the call signature:
  - Signature expected (Julia): the call signature as the first line, then a blank line, then the short summary.
  - Signature not expected, because the tools show it (Python, Rust, JavaScript/TypeScript, Java, C#, C/C++, Go, R): start with the short summary.
    In Go, the short summary starts with the function's name, per Go convention.
  - Any other language: follow its doc convention; omit the signature if its doc tools show it.
- After the short summary, give one line per non-obvious parameter, explaining its meaning (not just its type).
- Use an explicit, grep-able placeholder (e.g. `TBW`) for documentation not yet written, rather than leaving the function undocumented.
  Visible unfinished work beats silent gaps.
  Never use `TODO` for this without asking the user first.
- Document non-obvious invariants or gotchas in the doc comment, not just what the function computes — e.g. why a threshold check exists, or what would break without it.
- Keep documentation in the code short and to the point.
  Provide concise reasons, no extensive prose.
- Naming a concept isn't enough, for a docstring, function name, or variable name alike.
  State the relationship to it: Is it, does it, computes it, or holds data for it (e.g. "Data for one step of X" vs "Runs one step of X", not just "One step of X").

## Comments

- Comment only the "why", never a mechanical restatement of the "what".
  If a comment restates the code in words, delete it; if it explains a reason the code itself can't express, keep it.
  A short summary satisfies this too — purpose is how "why" doubles as the reader-facing "what".
- Write comments and doc comments with one sentence per line.
- Document a naming or structural convention once, where it's introduced, instead of repeating the explanation at every use site.

## Imports and Dependencies

- Prefer explicit, qualified access to library functions (`module.function()`) over blanket imports, except for the small number of functions used so pervasively that qualifying them everywhere would hurt readability — import those directly, and only those.
  This rule does not apply to Rust, which follows its own `use` conventions.
- Where the language allows precision (explicit numeric types, strict equality, etc.), use it at points where ambiguity could silently produce the wrong result.

## File/Module Layout

- Group and declare the public interface (exports, `__all__`, etc.) in one place near the top, separate from implementation.
- Order top-level definitions by call hierarchy: entry point first, then each function immediately followed by the helpers it calls — so reading top-to-bottom follows execution order.
