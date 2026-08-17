# Writing Style Guide

For prose in Markdown notes/docs (lab books, design notes, project-plan docs).
Not code — see `STYLE.md` for that. Rules, not prose.

## Voice

- Write in first person ("I propose", "I wish to run", "We wish ai2 to...") or, for active objects, the impersonal third person ("the function calls...", "the class covers...").
- A note generally records a person doing research, not a system describing itself.

## Show the reasoning, not just the outcome

- Walk through the problem, the tradeoff, the alternatives considered, *then*
  the decision. Don't jump straight to the settled fact.
- When a choice is arbitrary or provisional, say so explicitly ("Notice how
  none of the following is the *correct* approach, but a choice. Also note
  the implied technical debt.") rather than presenting it as final.
- Surface consequences and risks where they arise in the reasoning ("more
  troublingly, makes some ai2 schemes converge to 0"), not in a separate
  risks/limitations section bolted on afterward.

## Don't cite yourself instead of writing

- Referencing past notes by date instead of restating the point (e.g. "per
  07-06 notes", "07-14 pre-notes") forces the reader to go hunting. If a
  point matters here, restate or link the reasoning inline.

## Hedge honestly

- Use "(?)" to label provisional decisions.
  A lab book is a record of an evolving hypothesis, not a finished spec —
  unhedged declarative sentences imply settled fact, so reserve them for
  things that actually are settled.
- Keep brainstorms and dead ends in the record, labeled as such (e.g.
  "Terrible(?) brainstorms"), rather than showing only the polished result
  that survived.

## Structure by narrative, not by artifact

- Order sections the way the work unfolded (wishlist → notation → candidate
  formulations → experiment design → chosen experiments → implementation
  choices → results), not by a reference-doc taxonomy (data / module /
  metrics / files / verification).
- A compact reference table (e.g. "Implemented tests") belongs at the *end*,
  after the prose has explained the reasoning — it summarizes, it doesn't
  replace, the explanation.

## Use the project's math notation

- Write formulas in LaTeX ($\tilde y_i(t)$, `$$\begin{align}...$$`), not as
  prose paraphrase or code-identifier stand-ins. Define symbols once in a
  "Notation" section, then reuse them.

## Keep open questions inline

- Mark unresolved questions and future work next to the section they arise
  by appending the tag "TODO", instead of batching them into a disconnected backlog at the end.
