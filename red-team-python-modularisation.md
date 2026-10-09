# Red-team: `python-modularisation`

**Disclosure:** `python-modularisation` has zero commits beyond `serious-scan` — confirmed
`git rev-parse python-modularisation` equals `git rev-parse serious-scan` (`43c53dd`), both
locally and on `origin`. There is no modularisation plan or code anywhere on this branch.

Per explicit instruction, this report instead red-teams `IDEAS.md` §1 (refactor options A-H:
A=Python API with keyword overrides, B=grid/one-factor helpers, C=package + entry point
[`src/*.py` → `src/robgoal/`], D=generic CLI overrides + config files, E=model registry,
F=separate training from tracking, G=protocol/optimizer as config, H=speed via preloading
tensors), which lives on `FCNN-debugging`, not this branch — it's the closest thing to a
"plan to improve the codebase" that exists in the repo. Per the review brief, only findings
that go *beyond* what `IDEAS.md` already concedes/caveats are reported; its own
self-acknowledged caveats (e.g. "E and F are worth it only once a third model family is
planned") are not repeated here as findings.

All five findings below were independently re-verified by an adversarial subagent against the
actual `IDEAS.md` text and the code it discusses; verdicts reflect that pass.

## 1. [HIGH] Option C doesn't account for its own path-anchoring breakage

`src/run_experiments.py:20` and `src/logged_training.py:20` both anchor via
`REPO_ROOT = Path(__file__).resolve().parent.parent` — correct today because each file sits
exactly one directory below the repo root (`src/foo.py`: `.parent`=`src/`,
`.parent.parent`=repo root). Option C proposes moving these files to `src/robgoal/foo.py` —
one directory deeper. That silently changes `.parent.parent` to resolve to `src/`, not the
repo root, relocating `RESULTS_DIR`, the MLflow sqlite tracking DB and artifact paths, and the
MNIST download directory into `src/` the next time anything runs. `IDEAS.md`'s option-C text
says nothing about this (it only mentions dropping the `PYTHONPATH`/`sys.path` reliance); no
other part of the document mentions `REPO_ROOT` either. `REPO_ROOT` would need to become
`.parent.parent.parent` as part of this move — currently unplanned.

## 2. [MEDIUM] Option G's optimizer-as-config doesn't flag the risk it reopens

`docs/lab-book.md`, "Optimizers: Adam vs SGD," explicitly says Adam was abandoned because it
made `ai2`-vs-control results "practically indistinguishable," and SGD was hardcoded on
purpose "to make sure ai2 is able to meaningfully change the loss landscape." Option G proposes
`optimizer: "sgd"|"adam"` as a config field purely as a convenience ("the lab-book lr/optimizer
todos become one-line runs"), with no mention that making this freely configurable risks
silently reintroducing the exact problem the project already diagnosed and worked around — a
future run (or an innocuous default change) could flip to Adam and reproduce the
already-known-uninformative "indistinguishable" result, with nothing to warn against it.
`IDEAS.md` §3's own follow-up #1 ("Re-run all sigmoid rows at lr=0.1 (or with Adam)") repeats
the same blind spot rather than flagging it.

## 3. [MEDIUM, quantified] Option H is undersold and dropped from the suggested order with no stated reason

`IDEAS.md` hedges option H ("Speed (unverified)... would probably cut per-run time a lot...
Profile first.") Benchmarked directly instead of trusting the hedge: one real epoch over MNIST
stage 0 (30,596 images) via the current per-epoch `DataLoader`+transform path takes ~2.0s;
preloading all tensors once (a ~3.15s one-time cost) and indexing per epoch thereafter takes
~0.008s — roughly a 250× per-epoch speedup after about 1.5 epochs of amortization. Given
`epochs_per_stage` defaults to 5-10 and the sweep runs 14-35+ configs across up to 6 stages
each, this is a substantially larger, cheaper win than "probably... profile first" conveys —
and yet H is the one option left *out* of the document's own "Suggested order" paragraph
("C, then A and B..., then D and G" — E/F explicitly deferred, H simply absent), with no
reason given for the omission.

## 4. [LOW-MEDIUM, narrowed after verification] CLI/doc migration is only partly addressed

Option D does explicitly cover one existing flag ("Subcommands (`run`, `scan`, `list`, `show`)
replace the `--scan {sweep,size}` switch"), so a blanket "none of the options discuss CLI
migration" claim would overstate it. But the other flags already shipped by
`serious-scan`/`FCNN-debugging` (`--num-seeds`, `--dataset`, `--initial-classes`,
`--classes-per-stage`, `--lr`) and the docs that describe them (`scan-list.md`, README.md's
"Adjusting the sweep") get no mention anywhere in option D or elsewhere in §1. Given this
project's own practice of replacing rather than deprecating (no compatibility shims), the plan
should commit to a single cutover for all of these, not just the one switch it names.

## 5. [LOW] "Roughly independent" sits in tension with "C, then A and B"

§1 opens with "Options, roughly independent and roughly ordered by payoff per effort," then its
own "Suggested order" paragraph reads "C, then A and B (share one `RunConfig`)." Read as a hard
sequencing dependency, that's in tension with "independent" — though the text only says
"roughly," so this is a fair but not airtight reading; it may just mean priority order rather
than a hard prerequisite. Worth a one-line clarification either way, not a blocking issue.

## Additional gap (not in the original draft)

None of options D, E, or G address backward compatibility of *existing* data: `results/*.json`
filenames and the `mlflow.db` run history were produced under the current `RunConfig` schema
(mutually exclusive optional `channels` vs. `hidden_sizes`/`T` fields, no `model_args`, no
stage spec). Each of D (config files), E (`model: str` + `model_args`), and G
(`stages=[[...]]`) changes that schema, with no mention of whether or how historical runs
remain loadable or comparable to new ones after the change — and the lab book's own practice of
comparing current numbers against previously logged results depends on that continuity.
