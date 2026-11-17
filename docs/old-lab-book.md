# Lab book: autaptic inhibition MNIST tests

Lab book recording experiments with *autaptic inhibition AI (ai2)* networks, as sketched out in `Mini_project_meta_continual_learning.md`.

## General contents

### Wishlist

- If ai2 implemented *adaption*, it would behave like a leaky integral of activity.
- We wish ai2 to:
  - Diffuse information across neurons (laterally and longitudinally).
  - Have a neuron self-regulate to, more or less, switch on and off between inputs.
  - Intervene by raising and lowering outputs of neurons (i.e., not only reducing/increasing them).
  - Have rigorous formal basis.

#### Terrible(?) brainstorms

- When using a sigmoid-shaped activation function, set the new midpoint to the last output.

### Notation

- $i =$ neuron of concern;
- $\tilde\bullet =$ variables changed by autaptic inhibition;\
- $\bullet^\text{self}_i = $ parameter or $i$'s autapse;
  => $y_i =$ $i$'s unadapted activity/output (i.e., no effect of ai2);\
  => $\tilde y_i =$ ai2-adapted activity/output;\
  => $y^\text{self}_i =$ neuron $i$'s autapse activity at input time $t$;
- $b_i =$ $i$'s bias.
- An equations or algorithm implementing ai2 is called an ai2 *variant*.

### Possible implementations

$$\begin{align}
  \tilde b_i(t)
    &= b_i(t) - \tilde b_i(t-1)
  \\
  \tilde y_i(t)
    &= y_i(t) - \tilde{y}_i(t-1)
  \\
  \tilde y_i(t)
    &= y_i(t)\cdot\left(
        1 - \tilde{y}_i(t-1)
      \right) \nonumber\\
    &= y_i(t) - \tilde{y}_i(t-1)y_i(t)
\end{align}$$

### Experiment design

- Learning is to take place between "time steps" $t, (t-1)$.
  - Option A: Show the same input $T$ times to give the network time to have time to converge, given ai2 adaptions.
  - Option B: Implement memory (leaky integrate-fire/integrate-inhibit).
  - Initial inhibition is 0.
- (Future control group: Randomly switching subsets of neurons off?)
- Learn subsets of all numbers, adding more and more after having learnt old set.
- Note: "*Having learnt*" is epoch-delimited instead of cost-function delimited.
  - Learning is epoch-delimited because cost function conditions may restrict allowed states too much (i.e., to a sphere in the loss landscape).
    - Checking whether the worry is true is a potential future TODO.
- Note: These ai2 implementations will reduce activity for the vast majority of scenarios.

### Chosen experiments

- Control groups:
  - CNN
  - Fully connected NN
- Our ai2 implementations:
  $$\begin{align}
    \tilde y_i(t)
      &= y_i(t) - \tilde{y}_i(t-1)
    \\
    \tilde y_i(t)
      &= y_i(t)\cdot\left(
          1 - \tilde{y}_i(t-1)
        \right) \nonumber\\
      &= y_i(t) - \tilde{y}_i(t-1)y_i(t)
  \end{align}$$

## Implementation choices and resolved ambiguities

The following describes choices made while translating above's framework into code.
Notice how none of the following is the *correct* approach, but a choice.
Also note the implied technical debt.

### Time-based inhibition in static-image classifiers

Unroll each forward pass over `T` fixed internal steps, feeding the same input every step.
This is done to:

- Allow dynamics to settle,
- Enable autaptic inhibition to take place in the first place.

Note how, for `T=1`, ai2 networks reduce to a plain layer.

### Activation functions: ReLU and Sigmoids

A priori, activation functions bound to $[0,1]$ match intuitions of thinking about relative inhibition, and seem closer to biological spiking neural networks (as in, there is a maximum signal (density)).
Further, some ai2 variants need boundedness to keep the adapted output in the activation function's domain (e.g., keeping $1-y(t-1)$ positive).

Yet, contrasting bound activation functions with ReLU activations used by our controls result in an apples-to-oranges comparison, and, more troublingly, makes some ai2 variants converge to 0 during rollout (again, consider $1-y(t-1)$, which can only decrease $\tilde{y}_(t)$ for positive-only outputs).

Especially the latter incompatibility's problematic character may be interpreted two ways: Either the ai2 formulations, or activation functions, are a bad choice, given the other.

I propose and run two different mitigation strategies:

- First, instead of ReLU, networks use a sigmoid activation functions.
  While this can keep outputs in the same domain, inhibition may still make neural outputs converge to 0.

- The second approach combines ReLU activation with sigmoid-gated inhibition, i.e.
  $$
  \tilde y_i(t) = s(a_\text{ReLU}(x))
  $$
  with a sigmoid function $s: a(X)\to \tilde Y$ yielding ai2 adaptions $\tilde y_i(t)$.

  This should ease comparability to ReLU structures while keeping ai2 implementation outputs in the expected domain ($[0,1)$).

#### Details on implemented activation schemes

I wish to run both of the above-described activation schemes.

**`relu_sigmoid`** choices:

- `y(t) = relu(Wx+b)` and `y_tilde(t) = tanh(y(t-1)/2)`
- The recurrent $\tanh$ inhibition term yields the inhibition of the next step.\
  => The adaption is in the interval $\tilde y \in [0,1[$ and maps 0 to 0.\
  => If using eq. (1)'s difference-based *ai2*, neurons with activation bigger than 1 can't be fully inhibited.

Further defaults:

- Autapse applies to **hidden layers only** — output logits stay a plain `Linear`.
- The unroll depth `T` (unroll depth) is a small default at first; generally, consider it a control parameter.

### Class-incremental data protocol

Each run starts with 5 of the 10 digits, then introduce the remaining 5 one at a time, one per stage — 6 stages total, cumulative (a later stage's training data is the union of every digit introduced so far, not a replacement of the previous set).

The alternative is to reveal the whole held-back set at once, right after the initial 5.
I chose one-by-one instead, without a strong reason.
That being said, a single new digit per stage lets me attribute a stage's change in forgetting/generalization to *that* digit specifically, rather than to an entangled batch of five.

It would be interesting to see how outcomes using batching would differ, though (TODO).

(?) Seeding stage 0 with 5 digits is arbitrary/not reasoned out formally.

### Evaluation metrics

Every stage's evaluation on **both** the subset of *introduced* digits, and all untaught ones.

This makes one accuracy read-out double as both a forgetting probe (accuracy on already-introduced digits, which can drop as new ones are learned) and a generalization probe (accuracy on not-yet-introduced digits) -- i.e., how much a network trained so far already generalizes to unseen classes.

Consequently, the per-stage/per-digit accuracy table is meant to be dense, since every stage evaluates all 10 digits.
A blank cell would mean a caller passed only a some digits -- which, while tolerated, is not wished for.

We also track backwards transfer, i.e., the accuracy drop on the initial digits introduced between stage 0 and the final stage.
TODO Since any digit introduced later has no stage-0 accuracy to regress from, it only makes sense to pass initial digits (which is not checked for currently!).

### Optimizers: Adam vs SGD

Initial tests, using Adam, returned practically indistinguishable of control and ai2 variants.
We suspect Adam's momentum and adaptive learning to cover up the slightly changed loss landscape.

In order to make sure ai2 is able to meaningfully change the loss landscape, we choose SGD explicitly because of its sensitivity to local loss.

### Logging: MLflow

As the heading says, I use [MLflow's PyTorch integration](https://mlflow.org/docs/latest/ml/deep-learning/pytorch/) to log and, ideally, live-view run results.
Concrete mechanics (what's logged, local backend choice, nested-run bookkeeping) are in `implementation_doc.md`, "Logging: `logged_training.py`" -- kept out of here so that mechanical detail doesn't drown the reasoning below.

Differentiating `ai2_variant` / `activation_scheme` / `T` / `epochs_per_stage` / hidden layer sizes: rather than a full Cartesian grid over all five dimensions (144 runs, mostly uninformative combinations), I sweep `T`, `epochs_per_stage`, and `hidden_sizes` one-factor-at-a-time off a single anchor config, on top of the 7 base tests below.
(?) The anchor choice and the sweep values themselves are guesses at an interesting range, not derived from anything.

Every run sets a fixed seed before model construction, so a config's result is at least reproducible run-to-run.
This does not equalize randomness *across* configs, though: data-loader shuffling keeps drawing from the same global RNG stream every epoch, so same-seed runs only match exactly through weight initialization and the first epoch -- from there different configs consume that stream at different rates (worst for the `epochs_per_stage` sweep specifically, since two configs advance it at genuinely different rates from stage 0's second epoch onward; their comparison isn't fully "all else equal").
To get real variance estimates on top of this single-run reproducibility, I can also run each config as several seed replicates instead of one (off by default -- it multiplies total training runs, and I want fast single-seed iteration as the common case).

I also now log each `AutapticLayer`'s inhibition-signal (`y_self`) absolute mean per epoch -- a first, coarse readout toward the distributed-vs-sparse-representations wishlist at the top of this book.
(?) It's a per-epoch snapshot, not a rigorous measure of "how distributed" activity is; still unclear what absolute mean pattern would actually confirm or refute the wishlist.

TODO Ideas surfaced but not (yet) pursued: logging per-layer gradient/weight norms, to check the "SGD is sensitive to loss-landscape changes" reasoning above directly rather than inferring it from accuracy; `mlflow.enable_system_metrics_logging()` for CPU/GPU utilization, likely low value at this scale; a depth (not just width) sweep for `FCNN_AI2`, free now that `hidden_sizes` is a tuple of arbitrary length.

**What moved where, and why (2026-10-14).**
MLflow started as the place where everything about a run lived, and the 560-run scan showed why that does not hold up.
Of 1146 MLflow runs only 14 carried a git commit (MLflow derives it from the *launching script's* path, and the scan was launched from a throwaway worker script outside the repo), none recorded a dirty tree, `initial_classes`/`classes_per_stage` were in neither the params nor any file name, and single-seed reruns silently overwrote `results/{name}.json`, which held nothing but the accuracy matrix.
Worse, a stale artifact path made the last `log_text` call crash after training, so runs finished and wrote zero results: results depended on a viewer.
I therefore flipped the roles.
The canonical record is now one HDF5 file per run, `results/{launch_id}/{run_name}.h5`, holding the full config, our own git provenance (a launch refuses a dirty tree), the accuracy matrix, per-epoch loss and `y_self`, and (when the capture flags are on) the per-neuron `y_self` histories that used to be JSON artifacts in MLflow.
Each launch also appends an entry (goal, method, code changes, written by me at launch time) to the new, append-only `docs/lab-book.md`; this file, `old-lab-book.md`, stays the place for reasoning.
MLflow is kept, default on, as a *viewer* for live supervision (`--no-mlflow` drops it): it is tagged with `launch_id`, `run_id` and the commit from our own provenance, and an MLflow error can no longer lose or block results (the `.h5` is written first; mid-run MLflow errors are reported at the end instead of aborting training).
`logged_training.py` is accordingly split into the always-on recording (`recorded_training.py`) and the optional MLflow adapter (`logged_training.py`); the mechanics are in `implementation_doc.md`.
(?) Whether to keep MLflow at all is open: with the `.h5` files and `plotting_utils.py` the only thing it still offers is the live view, and at 12 parallel workers SQLite contention is a real cost (MLflow hard-codes a 20 s busy timeout per session).
Still unsolved: `results/` is gitignored, so the numbers are not reproducible from the repo alone, but every file now names the commit that produced it. TODO decide whether to commit a results snapshot.

## Implemented tests

All of these can be and are run locally, if not said otherwise.

### 01 Simple runs

3 controls (no self-inhibition) + 2 formula variants × 2 activation schemes.

| # | Model | `Activation_Scheme` | Notes |
| --- | ------- | -------- | ------- |
| 1 | CNN control | ReLU | small conv net (2×conv+pool+FC), no recurrence |
| 2 | FCNN_AI2 control | ReLU | 784→256→128→10, plain ReLU (= `relu_sigmoid` autapse layer at `T=1`) |
| 3 | FCNN_AI2 control | sigmoid | same skeleton, plain sigmoid (= `sigmoid` autapse layer at `T=1`) |
| 4 | FCNN_AI2 + `ai2_diff` | sigmoid | `y(t) = y_raw(t) - y(t-1)` |
| 5 | FCNN_AI2 + `ai2_gate` | sigmoid | `y(t) = y_raw(t) * (1 - y(t-1))` |
| 6 | FCNN_AI2 + `ai2_diff` | ReLU_sigmoid| `y(t) = relu(y_raw(t) - tanh(y(t-1)/2))` |
| 7 | FCNN_AI2 + `ai2_gate` | ReLU_sigmoid| `y(t) = relu(y_raw(t)) * (1 - tanh(y(t-1)/2))` |

On top of these 7, `run_experiments.py` runs 7 more: a one-factor-at-a-time sweep over `T` (`{2,5,8}`), `epochs_per_stage` (`{2,10}`), and `hidden_sizes` (`{(64,32),(512,256)}`), off test 6 as the anchor -- see "Logging: MLflow" above for why one-factor-at-a-time over a full grid, and its caveats.

### MLflow-supported sweep

Differentiating `ai2_variant` / `activation_scheme` / `T` / `epochs_per_stage` / hidden layer sizes: rather than a full cartesian grid over all five dimensions (144 runs, mostly uninformative combinations), I sweep `T`, `epochs_per_stage`, and `hidden_sizes` one-factor-at-a-time off a single anchor config, on top of the 7 base tests below.
(?) The anchor choice and the sweep values themselves are guesses at an interesting range, not derived from anything.

Every run sets a fixed seed before model construction, so a config's result is at least reproducible run-to-run.
This does not equalize randomness *across* configs, though: data-loader shuffling keeps drawing from the same global RNG stream every epoch, so same-seed runs only match exactly through weight init and the first epoch -- from there different configs consume that stream at different rates (worst for the `epochs_per_stage` sweep specifically, since two configs advance it at genuinely different rates from stage 0's second epoch onward; their comparison isn't fully "all else equal").
To get real variance estimates on top of this single-run reproducibility, I can also run each config as several seed replicates instead of one (off by default -- it multiplies total training runs, and I want fast single-seed iteration as the common case).

I also log each `AutapticLayer`'s inhibition-signal (`y_self`) absoute mean per epoch -- a first, coarse readout toward the distributed-vs-sparse-representations wishlist at the top of this book.
(?) It's a per-epoch snapshot, not a rigorous measure of "how distributed" activity is; still unclear what absoute mean pattern would actually confirm or refute the wishlist.

TODO Ideas surfaced but not (yet) pursued: logging per-layer gradient/weight norms, to check the "SGD is sensitive to loss-landscape changes" reasoning above directly rather than inferring it from accuracy; `mlflow.enable_system_metrics_logging()` for CPU/GPU utilization, likely low value at this scale; a depth (not just width) sweep for `FCNN_AI2`, free now that `hidden_sizes` is a tuple of arbitrary length.

## ToDos 2026-08-26 or so

### Present Conditions

- Compare 5 random Random Seeds results for each run!
- For now, do not use sigmoid variants, since they seem to die off.
- We think/hope our network can deal better with difficult/restricted conditions (small network sizes, few epochs, etc.).
 => Implement experiments with such conditions.
- TODO: Implement activity recording, not only `y_self` recording.

### Focused experiments

For all the following, use 5 randomly chosen random seeds: `--num-seeds 5`
(or `num_seeds=5`) now draws that many distinct random seeds per config
(recorded via the `seed` MLflow param and each replicate's results
filename) instead of the sequential `0, 1, ...` used before this was
implemented -- see `implementation_doc.md`, "Experiment runner:
`run_experiments.py`".

- [x] Compare networks of increasingly smaller...
  - [x] Size: `--scan size` (or `run_experiments.run_size_scan()`) runs
    `size_scan.py`'s scan -- ai2 variants, FCNN controls, and the CNN
    control, all shrinking together, 8 steps by default (down to FCNN
    hidden_sizes `(2, 1)` / CNN channels `(1, 1)`, deep enough to overwhelm a
    10-class classifier's capacity). Epoch size isn't
    part of this scan; use `sweep.py`'s existing `EPOCHS_SWEEP` for that, or
    pass a smaller `epochs_per_stage` directly.
  - [ ] Epoch size (10, 5, 2) -- not part of the new size scan, see above.
- [x] Make networks learn one (of, e.g., three representationally different) number(s) at a time (instead of 0-4, 5, 6,...): `--initial-classes`/`--classes-per-stage` (or `initial_classes`/`classes_per_stage`) control `data.build_stage_digits`'s split now -- e.g. `--initial-classes 1 --classes-per-stage 1` starts from a single digit.
- [ ] Try out different learning rates (1e-(2,3,4)).
  - H₁: Higher `lr` prevents convergence?
  - H₂: Too small `lr` prevents sigmoids to converge?

### Dataset: FashionMNIST

Added `data.py`'s `dataset` param (`"mnist"`/`"fashion_mnist"`,
`--dataset`/`dataset=` throughout `run_experiments.py`) as a second
class-incremental benchmark alongside MNIST, same 10-class/1x28x28 shape so
nothing downstream needed to change. Non-`"mnist"` runs get their result
filename/MLflow run name suffixed with the dataset name, so switching
datasets never overwrites the other's results.

## 2026-11-17: Incremental Inhibition Experiments Before the Meeting

### Choosing the Network Width W

Wish: one width $W$ for four equal hidden layers, where the ai2 effect is neither saturated nor floored.
Pilot launch `2026-11-17-14-54_02ca940`: FCNN controls (T=1, both schemes) and matched CNN controls at $W \in \{2, 4, 8, 16, 32, 64, 128\}$, MNIST, 1 initial class, 1 class per stage, 5 seeds, 1 epoch per stage.
The first attempt (`2026-11-17-14-47_6996f49`) was aborted because the working tree was edited during the launch; it is not used.

Definition, written down in the agent's plan before the first pilot launch (the analysis code `pilot_width.py` was committed only after that launch had started, and a fallback was added after seeing the results; agent decision, unconfirmed; the researcher asked for a quantitative definition but did not fix one):

- $D$ is the last value of `metrics.backward_transfer_over_time`: the mean over classes introduced before the last stage of the final accuracy minus the accuracy at the end of the introducing stage.
- Interference is present at a width if the whole 95 % t-interval of $D$ over 5 seeds lies below $-0.05$.
- Interference is absent if the whole interval lies above $-0.02$ and the seed-mean final accuracy is at least 0.8 (so a network that never learned does not count as robust).
- Both FCNN activation schemes must agree; the CNN is reported only.

Result (`results/plots/size_pilot/width_table.csv`, `width_pilot.png`):

- `relu_sigmoid`: $D = -0.173$ at $W=2$ (present, accuracy 0.22), $-0.060$ at 4 and $-0.052$ at 8 (neither: the intervals reach above $-0.05$), $-0.014$ at 16, and $-0.000$ at 64 (absent, accuracy 0.96).
- `sigmoid`: accuracy is 0.10 to 0.12 (chance) at every width, including 128, so $D$ is a floor effect and says nothing about capacity.
- CNN: mean $D$ between $-0.02$ and $+0.04$ at every width, no interval below $-0.05$, accuracy 0.93 to 0.98; the CNN never shows interference in this protocol.

The pre-registered both-scheme rule therefore cannot be satisfied.
Notice that the fallback is a choice, not the correct approach: the rule is applied to the schemes that reach accuracy 0.8 at some width, which is `relu_sigmoid` only.
That gives $W_\text{low} = 2$, $W_\text{high} = 64$, geometric mean 11.3, rounded to a multiple of 4 gives $W = 12$.
Uncertainty: the interval at $W = 4$ and $W = 8$ is wide (5 seeds), so $W_\text{low}$ is really "somewhere between 2 and 8" and $W$ could as well be 8 to 16 (no probability can be worked out from 5 seeds per width).
Notice also that $W_\text{low} = 2$ is a network with seed-mean accuracy 0.22: it counts as interference because it loses what it learned at introduction, but it is close to a floor.
The pilot ran 1 epoch per stage; the exhaustive runs use 5, which probably moves the transition to smaller widths (?).
The exhaustive launches report $D$ with its interval for the controls at $W = 12$, so that can be checked.

Consequence for the sigmoid scheme: its controls are at chance at $W = 12$, and the unrolled gate and leaky variants will be at or below chance too.
Those columns are kept because the researcher asked for both schemes, but they cannot show a robustness effect.

### Metric Definitions in the Figures

Newest-class curves use, per stage $s$, the accuracy on the classes introduced in stage $s$ after the last epoch of that stage, and `metrics.backward_transfer_over_time` read at the same epoch (mean over all classes introduced earlier; undefined in stage 0).
This reading of "backward transfer computed at that stage" is an interpretation (agent decision, unconfirmed).

### Leaky-Integrate Inhibition

Proposed (agent decision, unconfirmed): for sample $i$ of the stream, $I_i = (1-\lambda)\sum_{k\ge1}\lambda^{k-1} a_{i-k}$ per neuron, with $a$ the activation signal of the *uninhibited* output (`sigmoid` output, or $\tanh(y/2)$ for `relu_sigmoid`), output $y\,(1 - w I_i)$, $w = 1$.
The unrolled layer instead feeds back the previous step's *inhibited* output.
Using the inhibited output here would make sample $i$ depend on the gated output of sample $i-1$, a sequential recursion that needs a Python loop over samples; the uninhibited signal allows one lower-triangular matrix product per batch.
Notice that this makes the leaky and unrolled variants differ in two ways at once (signal and dynamics).
The $(1-\lambda)$ factor keeps $I$ in $[0, 1)$, like the unrolled signal; without it $w = 1$ would make the gate negative.
The state after a batch is carried to the next batch, detached; it is kept separately for training and evaluation, cleared at every stage start, and (evaluation state only) at the start of every evaluation pass.
Inside a batch, gradients flow between samples (earlier samples' signals feed later ones).

Primary $\lambda = 0.9$ (memory about 10 samples, so the signal is a smoothed population-activity estimate).
With shuffled training batches neighbouring samples are random classes, so the layer cannot track the newest class specifically; the stage reset only matters for the first ~10 samples of a stage.
The mixed test set uses one fixed random order, so a test stream mixes MNIST and FashionMNIST like the training batches; evaluation still depends on test order and the fixed batch size 256.
The exhaustive launches add $\lambda = 0.5$ and $0.99$ (at most two more, as agreed).

### 20-Class Mixed Protocol

MNIST digits are labels 0 to 9, FashionMNIST classes labels 10 to 19, each dataset normalised with its own mean and standard deviation (as `data.py` does).
Notice the shortcut: a network can in principle tell the datasets apart from intensity statistics alone.
Random class order per seed, recorded as `stage_digits` in each run's meta; all configs of a launch share the seeds, so every model sees the same orders.
Coarse runs use 25 % of every stage's training examples, drawn once per seed.
Because the order differs per seed, the seed-averaged heatmap by label blurs the introduction diagonal; a second heatmap orders each seed's classes by introduction rank.

### Figure Axes

Backward-transfer axes are signed, so they span the data range (negative values included) instead of starting at 0; the plotting rule "axes start at 0" is deviated from there on purpose, and the figure titles say so.
Heatmaps use one fixed 0-1 colour range.

### Exhaustive Results: Does W = 12 Still Sit in the Transition? (5 Epochs per Stage)

The pilot ran 1 epoch per stage, so the width choice needed a check at the 5 epochs of the exhaustive runs.
The FCNN control (`relu_sigmoid`) at W = 12 has $D = -0.021$ with a 95 % interval of $[-0.058, +0.016]$ on MNIST (launch `2026-11-17-16-00_ef03c45`).
That interval straddles both thresholds of the definition above ($-0.05$ for present, $-0.02$ for absent), so interference is neither clearly present nor absent: W = 12 is still in the transition region (from one interval over 5 seeds; no probability can be worked out).
On FashionMNIST the interval is $[-0.099, -0.010]$ (launch `2026-11-17-16-44_ef03c45`) and on the mixed protocol $[-0.052, -0.026]$ (launch `2026-11-17-17-28_ef03c45`).
Neither is below $-0.05$ as a whole, so under the stated definition neither counts as "present" either.
An earlier reading of the FashionMNIST interval as "interference present" used a $-0.01$ threshold that is not part of the definition, and was dropped.

Notice that the width was chosen on MNIST only; the other two datasets were run at the same $W$ without their own pilot (a choice, not a result).

The sigmoid question from the pilot is partly answered.
At 5 epochs the plain `sigmoid` FCNN control learns (accuracy 0.65 MNIST, 0.71 FashionMNIST, 0.80 mixed), so its floor in the pilot probably came from the 1-epoch setting (?).
The unrolled and leaky `sigmoid` layers stay at chance on MNIST and FashionMNIST, and the leaky ones also on mixed, while the unrolled `sigmoid` gate learns on mixed (0.61).
One possible reason (?) is that the `sigmoid` activation lies in $(0, 1)$ with mean about 0.5, so the inhibition signal is near 0.5 everywhere (measured: 0.5 in the leaky layers), which would roughly halve every output; no test of this reason was run.
The positive $D$ of the learning `sigmoid` models comes from low accuracy at introduction (mean newest-class accuracy 0.36 to 0.51 against 0.89 to 0.95 for `relu_sigmoid`), largely, not necessarily entirely, from that and not from retained knowledge (?).

### Re-Rendered Phase 1 Figures

The researcher asked, after the Phase 1 deadline, for all plots to have zero x margin and for the inhibition plots to share one y range.
That required regenerating the coarse and size-pilot figures from the same launches (`2026-11-17-14-54_02ca940`, `2026-11-17-15-15_f7ca382`, `2026-11-17-15-22_f7ca382`, `2026-11-17-15-29_f7ca382`) at the same paths.
The task rule "do not overwrite Phase 1 figures" conflicts with that request; I followed the later explicit request (agent decision on the conflict, flagged in the final reply).
The versions shown at the meeting are not kept, because `results/` is not under version control; the data and all numbers are unchanged.
