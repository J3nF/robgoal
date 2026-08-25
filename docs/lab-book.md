# Lab book: autapsic-inhibition MNIST tests

Lab book recording experiments with *autapsic inhibition AI (ai2 )* networks, as sketched out in `Mini_project_meta_continual_learning.md`.

## General contents

### Wishlist

- If ai2 implemented *adaption*, it would behave like a leaky integral of activity.
- We wish ai2 to:
  - Diffuse information across neurons (laterally and longitudinally).
  - Have a neuron self-regulate to, more or less, switch on and off between inputs.
  - Intervene by raising and lowering outputs of neurons (i.e., not only reducing/increasing them).
  - Have rigorous formal underbuilding.

#### Terrible(?) brainstorms

- When using a sigmoid-shaped activation function, set the new midpoint to the last output.

### Notation

- $i =$ neuron of concern;
- $\tilde\bullet =$ variables changed by autapsic inhibition;\
- $\bullet^\text{self}_i = $ parameter or $i$'s autapsis;
  => $y_i =$ $i$'s unadapted activity/output (i.e., no effect of ai2);\
  => $\tilde y_i =$ ai2-adapted activity/output;\
  => $y^\text{self}_i =$ neuron i's autapsis activity at input time $t$;
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

- Learning is to take place between "timesteps" $t, (t-1)$.
  - Option A: Show the same input $T$ times to give the network time to have time to converge, given ai2 adaptions.
  - Option B: Implement memory (leaky integrate-fire/integrate-inhibit).
  - Initial inhibition is 0.
- (Future control group: Randomly switching subsets of neurons off?)
- Learn subsets of all numbers, adding more and more after having learnt old set.
- Note: "*Having learnt*" is epoch-delimited instead of cost-function delimited.
  - Learning is epoch-delimited because cost function conditions may restrict allowed states too much (i.e., to a sphere in the loss landscape).
    - Checking whether the worry is true is a potential future TODO.
- Note: These ai2 implementations will reduce activity for the vast majority of scenarioes.

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
- Enable autapsic inhibition to take place in the first place.

Note how, for `T=1`, ai2 networks reduce to a plain layer.

### Activation functions: ReLU and Sigmoids

A priori, activation functions bound to $[0,1]$ match intuitions of thinking about relative inhibition, and seem closer to biological spiking neural networks (as in, there is a maximum signal (density)).
Further, some ai2 variants need boundedness to keep the adapted ouput in the activation function's domain (e.g., keeping $1-y(t-1)$ positive).

Yet, contrasting bound activation functions with ReLU activations used by our controls result in an apples-to-oranges comparison, and, more troublingly, makes some ai2 variants converge to 0 during rollout (again, consider $1-y(t-1)$, which can only decrease $\tilde{y}_(t)$ for positive-only outputs).

Especially the latter incompatibility's problematicness may be interpreted two ways: Either the ai2 formulations, or activation functions, are a bad choice, given the other.

I propose and run two different mitigation strategies:

- First, instead of ReLU, networks use a sigmoid activation functions.
  While this can keep ouputs in the same domain, inhibition may still make neural outputs converge to 0.

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

Every stage's evaluation on **both** the subset of *introduced* digits, as well as all untaught ones.

This makes one accuracy read-out double as both a forgetting probe (accuracy on already-introduced digits, which can drop as new ones are learned) and a generalization probe (accuracy on not-yet-introduced digits) -- i.e., how much a network trained so far already generalises to unseen classes.

Consequently, the per-stage/per-digit accuracy table is meant to be dense, since every stage evaluates all 10 digits.
A blank cell would mean a caller passed only a some digits -- which, while tolerated, is not wished for.

We also track backwards transfer, i.e., the accuracy drop on the intial digits introduced between stage 0 and the final stage.
TODO Since any digit introduced later has no stage-0 accuracy to regress from, it only makes sense to pass initial digits (which is not checked for currently!).

### Optimisers: Adam vs SGD

Initial tests, using Adam, returned practically undistinguishible of control and ai2 variants.
We suspect Adam's momentum and adaptive learning to cover up the slightly changed loss landscape.

In order to make sure ai2 is able to meaningfully change the loss landscape, we choose SGD explicitly because of its sensitivity to local loss.

### Logging: MLflow

As the heading says, I use [MLflow's PyTorch integration](https://mlflow.org/docs/latest/ml/deep-learning/pytorch/) to log and, ideally, live-view run results.
Concrete mechanics (what's logged, local backend choice, nested-run bookkeeping) are in `implementation_doc.md`, "Logging: `logged_training.py`".

## Implemented tests

All of these can be and are run locally, if not said otherwise.

### 01 Simple runs

3 controls (no self-inhibition) + 2 formula variants × 2 activation schemes.

| # | Model | Activation_Scheme | Notes |
| --- | ------- | -------- | ------- |
| 1 | CNN control | ReLU | small conv net (2×conv+pool+FC), no recurrence |
| 2 | FCNN_AI2 control | ReLU | 784→256→128→10, plain ReLU (= `relu_sigmoid` autapse layer at `T=1`) |
| 3 | FCNN_AI2 control | sigmoid | same skeleton, plain sigmoid (= `sigmoid` autapse layer at `T=1`) |
| 4 | FCNN_AI2 + `ai2_diff` | sigmoid | `y(t) = y_raw(t) - y(t-1)` |
| 5 | FCNN_AI2 + `ai2_gate` | sigmoid | `y(t) = y_raw(t) * (1 - y(t-1))` |
| 6 | FCNN_AI2 + `ai2_diff` | relu_sigmoid| `y(t) = relu(y_raw(t) - tanh(y(t-1)/2))` |
| 7 | FCNN_AI2 + `ai2_gate` | relu_sigmoid| `y(t) = relu(y_raw(t)) * (1 - tanh(y(t-1)/2))` |

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
- We think/hope our network can deal better with difficult/restricted conditions (small network sizes, few epochs, etc.).\n
 => Implement experiments with such conditions.
- TODO: Implement activity recording, not only y_self recording.

### Focused experiments

- [ ] Compare networks of increasingly smaller size;
- [ ] Make networks learn one (of, e.g., three representationally different) numbers at a time (instead of 0-4, 5, 6,...).
- [ ] Try out different learning rates.
  - H: Higher lr prevents convergence?
