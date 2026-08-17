# Style Guide

General coding principles, illustrated with generic examples. Rules, not prose.

## Naming

- If the code implements a formula or spec from an external source (a paper, an RFC, a textbook), match variable names to that source's notation as closely as the language allows, including special symbols where supported (e.g. `ε`, `α` in a language with Unicode identifiers; `epsilon`, `alpha` otherwise).
- Give core domain objects short, fixed names and reuse them identically across every function signature — don't rename the same concept per function (e.g. always `graph`/`g`, never `graph` in one function and `network` in the next for the same object).
- Mark hypothetical/candidate values — computed under an assumed change that may be discarded — with a consistent suffix or prefix, distinct from the committed value (e.g. `total_candidate`/`total_tmp` vs. `total`).
- Adopt a verb-prefix convention and apply it uniformly:
  - a prefix for pure computation that returns a value without deciding anything (e.g. `get_*`, `compute_*`)
  - a prefix for functions that evaluate a candidate and decide whether to accept it, returning either the new or the original value (e.g. `check_*`, `try_*`)

## Function shape

- One responsibility per function. Separate "compute a candidate" from "decide whether to apply it" from "apply it" — don't fuse these into one function.
- If several functions share parameters, use the same name and the same order for those parameters everywhere. Inconsistent ordering is a common source of call-site bugs.
- Prefer returning new values over in-place mutation. Where mutation is unavoidable, copy the input explicitly first, then mutate the copy — never mutate a caller's value silently.
- If functions mutate inputs in-place, make this visible by appending "!" or another conventionalised signifier.
- In search/optimization loops, return as soon as an improving candidate is found rather than scanning exhaustively, when order doesn't affect correctness (e.g. shuffle first, then take the first improvement).

## Documentation

- Every public function gets a doc comment with: the call signature, a one-line summary, and one line per non-obvious parameter explaining its meaning (not just its type).
- Use an explicit, greppable placeholder (e.g. `TBW`, `TODO`) for documentation not yet written, rather than leaving the function undocumented. Visible unfinished work beats silent gaps.
- Document non-obvious invariants or gotchas in the doc comment, not just what the function computes — e.g. why a threshold check exists, or what would break without it.
- Keep documentation in the code short and to the point. Provide concise reasons, no extensive prose.

## Comments

- Comment only the "why", never the "what". If a comment restates the code in words, delete it; if it explains a reason the code itself can't express, keep it.
- Document a naming or structural convention once, where it's introduced, instead of repeating the explanation at every use site.

## Imports and dependencies

- Prefer explicit, qualified access to library functions (`module.function()`) over blanket imports, except for the small number of functions used so pervasively that qualifying them everywhere would hurt readability — import those directly, and only those.
- Where the language allows precision (explicit numeric types, strict equality, etc.), use it at points where ambiguity could silently produce the wrong result.

## File/module layout

- Group and declare the public interface (exports, `__all__`, etc.) in one place near the top, separate from implementation.
- Order top-level definitions by call hierarchy: entry point first, then each function immediately followed by the helpers it calls — so reading top-to-bottom follows execution order.
