# Writing Style Guide

For all prose an agent writes.
Not code — see `CODING_STYLE.md` for that.
Rules, not prose.

## Scope

- Universal rules apply to all prose: conversational replies, commit message bodies, PR descriptions, reviews, prose in comments, and documents.
- Document rules apply only to documents: Markdown, TeX, Typst, and plain-text files (`.md`, `.tex`, `.typ`, `.txt`) that are not part of a reply, commit message, PR description or review.
- Replies, commit messages, PR descriptions and reviews follow `CLAUDE.md` and the global rules.
  Reviews use the global review format (`file:line`, severity, confidence).
  The document rules never apply to them.

## Simple Writing Rules

- Use Title Case in headings.
- Use British English.
- Use Oxford commas.


## One Sentence Per Line

- Every sentence gets its own line.
- A line break follows each full stop, question mark and exclamation mark.
- Break only at true sentence endings, including inside parentheses and quotes.
- Never break after abbreviations ("e.g.", "i.e.", "etc.", "vs.", "Dr."), inside decimals or version numbers, or inside filenames and URLs.
- Never apply the rule to non-prose: tables, code blocks, inline code, headings.
- A list item keeps its marker; further sentences of the item go on indented continuation lines.
- A commit subject line stays one line; the body follows the rule.

> This is wrong. There are multiple sentences in one line.

> This is right.
> Each sentence is on its own line!

## Voice

- Write in first person only to record intent the user has stated ("I propose", "I wish to run", "We wish `example_module` to...").
- When the agent decides something itself, rather than rephrasing intent the user stated, do not write it as the user's decision.
  Label it "Proposed (agent decision, unconfirmed)" in the text and flag it to the user in the reply.
- For active objects, use the impersonal third person with active verbs ("the function calls...", "the module simulates the system...").
  This is strongly preferred over passive voice ("the system is simulated...").

## Don't Cite Yourself Instead of Writing

- Referencing past notes by date instead of restating the point (e.g. "per 07-06 notes", "07-14 pre-notes") forces the reader to go hunting.
  If a point matters here, restate or link the reasoning inline.

## Replies, Commit Messages and PR Descriptions

- In replies, state the outcome first, with its confidence.
- In replies, then give the reasoning briefly, in the order problem, tradeoff, alternatives, what supports the decision.
- In commit messages and PR descriptions, state the outcome first, describing the final diff, without discarded approaches.

## Document Rules

These apply only to documents (see Scope).

### Show the Reasoning, Not Just the Outcome

- Walk through the problem, the tradeoff, the alternatives considered, *then* the decision.
  Don't jump straight to the settled fact.
- When a choice is arbitrary or provisional, say so explicitly ("Notice how none of the following is the *correct* approach, but a choice.
  Also note the implied technical debt.") rather than presenting it as final.
- Surface consequences and risks where they arise in the reasoning ("more troubling, makes some `example_module` schemes converge to 0"), not in a separate risks/limitations section bolted on afterwards.

### Hedge Honestly

- Use "(?)" to label provisional decisions.
  Ongoing notes are often used as a record of an evolving hypothesis, not a finished spec — unhedged declarative sentences imply settled fact, so reserve them for things that actually are settled.
- Keep brainstorms and dead ends in the record, labelled as such (e.g. "Terrible(?) brainstorms"), rather than showing only the polished result that survived.

### Structure by Narrative, Not by Artifact

- Order sections the way the work unfolded (wish list → notation → candidate formulations → experiment design → chosen experiments → implementation choices → results), not by a reference-doc taxonomy (data / module / metrics / files / verification).
- A compact reference table (e.g. "Implemented tests") belongs at the *end*, after the prose has explained the reasoning — it summarizes, it doesn't replace, the explanation.

### Use the Project's Math Notation

- Write formulas in LaTeX ($\tilde y_i(t)$, `$$\begin{align}...$$`), not as prose paraphrase or code-identifier stand-in.
  In Typst files use Typst math syntax; in plain-text files use plain-text notation.
  Define symbols once in a "Notation" section, then reuse them.

### Keep Open Questions Inline

- Mark unresolved questions and future work next to the section they arise by appending the tag "TODO", instead of batching them into a disconnected backlog at the end.
- An agent asks the user before adding any such "TODO", and appends the tag only once the user has agreed.
