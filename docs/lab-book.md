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
Also note the implied techincal debt.

### Time-based inhibition in static-image classifiers

Unroll each forward pass over `T` fixed internal steps, feeding the same input every step.
This is done to:

- Allow dynamics to settle,
- Enable autapsic inhibition to take place in the first place.

Note how, for `T=1`, ai2 networks reduce to a plain layer.

### Activation functions: ReLU and Sigmoids

A priori, activation functions bound to $[0,1]$ match intuitions of thinking about relative inhibition, and seem closer to biological spiking neural networks (as in, there is a maximum signal (density)).
Further, some ai2 schemes need boundedness to keep the adapted ouput in the activation function's domain (e.g., keeping $1-y(t-1)$ positive).

Yet, contrasting bound activation functions with ReLU activations used by our controls result in an apples-to-oranges comparison, and, more troublingly, makes some ai2 schemes converge to 0 during rollout (again, consider $1-y(t-1)$, which can only decrease $\tilde{y}_(t)$ for positive-only outputs).

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
- The unroll depth `T` (unroll depth) is a small default at first; generally, consider it a controll parameter.

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

## Implemented tests

3 controls (no self-inhibition) + 2 formula variants × 2 activation schemes.

| # | Model | Scheme | Notes |
| --- | ------- | -------- | ------- |
| 1 | CNN control | ReLU | small conv net (2×conv+pool+FC), no recurrence |
| 2 | FCNN control | ReLU | 784→256→128→10, plain ReLU (= `relu_sigmoid` autapse layer at `T=1`) |
| 3 | FCNN control | sigmoid | same skeleton, plain sigmoid (= `sigmoid` autapse layer at `T=1`) |
| 4 | FCNN + `output_diff` | sigmoid | `y(t) = y_raw(t) - y(t-1)` |
| 5 | FCNN + `mult_gate` | sigmoid | `y(t) = y_raw(t) * (1 - y(t-1))` |
| 6 | FCNN + `output_diff` | relu_sigmoid| `y(t) = relu(y_raw(t) - tanh(y(t-1)/2))` |
| 7 | FCNN + `mult_gate` | relu_sigmoid| `y(t) = relu(y_raw(t)) * (1 - tanh(y(t-1)/2))` |
