//#import "@preview/charged-ieee:0.1.4": ieee
#import "conf.typ": ieee
#import "@preview/dashy-todo:0.1.3": todo
#import "@preview/lovelace:0.3.1": pseudocode-list

#set page(numbering: "1", paper: "a4")
#show: ieee.with(
  title: [ Autaptic inhibition's role in continual learning], // TODO
  abstract: [
    How can one design a performant, memory-efficient continual learning framework that operates within a network's fixed capacity?
    We hypothesize that local capacity constraints limit long-term learning performance in sparse networks, and how distributing learning schemes could circumvent this problem. 
    Inspired by biological networks, we propose _autaptic inhibition for artificial intelligence (ai2)_ as a mechanism to motivate networks to distribute representations across many neurons.
    We then lie the theoretical and implementation groundwork to test whether they can outperform their conventionally sparse counterparts by balancing knowledge retaining and information redistribution flexibly.
  ],
  authors: (
    (
      name: "Jens Friedel",
      department: [School of Engineering Mathematics and Technology],
      organization: [University of Bristol],
      location: [Bristol, United Kingdom],
      email: "jens.friedel@bristol.ac.uk"
    ),
    (
      name: "Conor Houghton",
      department: [School of Engineering Mathematics and Technology],
      organization: [University of Bristol],
      location: [Bristol, United Kingdom],
      email: "conor.houghton@bristol.ac.uk"
    ),
    (
      name: "Gabriel Leivas Oliveira",
      department: [School of Engineering Mathematics and Technology],
      organization: [University of Bristol],
      location: [Bristol, United Kingdom],
      email: "gabriel.leivasoliveira@bristol.ac.uk"
    ),
  ),
  index-terms: ("Continual learning", "Catastrophic forgetting", "Neural network architecture", "Biologically inspired systems"),
  paper-size: "a4",
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

= Introduction <sec:Intro>

Continual-learning systems, which are to acquire new representations over time, face _catastrophic forgetting_, where adapting parameters to new data degrades previously learned representations @Son2024MetaLearning.
One framing of memory-efficient continual learning's research question is thus: How can a network of fixed size keep learning without sacrificing past performance?

We hypothesize that a major contributor to forgetting is local capacity exhaustion.
When a single neuron is responsible for differentiating a feature, it can't be used to store novel representations.
At the same time, neurons responsible for detecting features of a particular domain tend to live in similar network regions.
We argue the exhaustion of that neighbourhood's representation capacity suffices to limit network performance irrespective of the total network capacity.
Conversely, sidestepping local capacity exhaustion would unlock more network capacity for representation learning, enabling better continual learning performance.

In contrast to sparse networks, distributing a representation across many neurons ensures the local network part responsible for a feature domain covers a larger part of the network and, subsequently, has access to more of the network's total capacity.

We propose *autaptic inhibition for artificial intelligence* (ai2): a self-recurrent inhibitory feedback term on a layer's neurons, driven by each neuron's own recent output, intended to discourage any one neuron from acting as a stable, unshared feature detector.
We take inspiration from *autapses* @vanderloos1972autapses --- synapses a neuron makes onto itself, first described anatomically in neocortical pyramidal cells.
Researchers understand circuit-level neural inhibition to support continual learning and memory stability in brains @barron2021inhibition; autaptic self-inhibition is a more granular, single-neuron analogue.
To our knowledge, there is no prior work applying self-recurrent, autaptic connections to distribute representation learning in continual learning settings.

Because we actively push the network from its usual learning trajectory, we expect worse initial accuracy.
Yet, we expect the lack of sparsity to enable slower and more graceful forgetting, since no single neuron's reuse can erase a whole feature.
We assume a well-calibrated ai2 implementation balances initial performance loss and a more flexible network storage to improve overall continual learning performance.


= Related Work <sec:RelatedWork>

We use the Avalanche framework @Carta2023Avalanche as a general reference for continual learning, in particular regarding benchmarks and evaluation protocols.

There are works combining meta-learning and continual learning to let a network's own learning rule adapt over a task stream @Son2024MetaLearning .
Our approach is complementary and structural, changing what a neuron's output can represent, rather than how parameters are updated.

Self-recurrent connections on individual neurons also are of interest outside continual learning.
IndRNN gives each neuron its own per-neuron recurrent self-connection to build longer, deeper RNNs @li2018indrnn.
The Equilibrated RNN adds an autapse-inspired, time-delayed self-feedback term that pushes each hidden state toward equilibrium, improving accuracy and stability @zhang2019ernn, whereas we wish to discourage any single neuron from becoming a stable, unshared feature detector.
Zhang and Li apply self-recurrence to spiking neurons via skip connections @zhang2021scsrsnn, and via a two-neuron laterally-inhibited unit, where a self-recurrent excitatory neuron preserves memory while a paired inhibitory neuron acts as a forget gate @zhang2021lisr.
To our knowledge, none of these use self-recurrence to encourage distributed representations for continual learning.

Within continual learning, other approaches instead identify and protect high-utility neurons directly @kim2021neuronimportance, or deliberately learn sparse, interference-resistant representations @javed2019oml, instead of distributing them.

= Method <sec:Method>

== Autaptic Inhibition Mechanism <sec:Mechanism>

An `AutapticLayer` replaces a plain linear-plus-activation hidden layer.
In order to allow the autaptic inhibition to depend on the current input during learning, each forward pass unrolls the same, fixed input over $T$ internal pseudo-time steps.
During each pseudo-time step, a neuron's autapse feeds back the last neural output, which the neuron uses to weaken its own output.

For now, we implement two *variants* combining this feedback with the neuron's original activation, as well as two *schemes* dictating how to integrate the inhibition signal with the to-be-inhibited output.

We write $x$ for a neuron's (unroll-step-invariant) external input and $a$ for the layer's activation function.
We assign $y(t)$ to a neuron's uninhibited activation at step $t$, $y_"self" (t)$ to its autaptic inhibition signal, and $tilde(y)(t)$ to its ai2-adapted output.

The former two are only used within the neuron:
$
y(t) &= a(x) \
y_"self" (t) &= g(u), quad y_"self" (0) = 0 \
$ <eq:ai2-drive>

Under the `sigmoid` scheme, $a$ is a sigmoid function and $g(u) = u$; under `relu_sigmoid`, $a$ is ReLU and $g(u) = tanh(u/2)$, which keeps the feedback signal bounded to $[0,1)$.

The two variants then differ in how $y(t)$ and $y_"self" (t)$ combine:
*ai2_diff* subtracts the feedback,

$ tilde(y)(t) = y(t) - y_"self" (t), $ <eq:ai2diff>

while *ai2_gate* multiplicatively gates it,

$ tilde(y)(t) = y(t) (1 - y_"self" (t)). $ <eq:ai2gate>

A neuron finally outputs $tilde(y)(T-1)$ (@alg:ai2-unroll) to the next layer.

Since $y_"self" (0) = 0$, both variants leave $tilde(y)(0) = y(0)$ unchanged.
Consequently, $T=1$ degenerates *exactly* to a plain activated linear layer.

#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [Forward pass of one `AutapticLayer` (per neuron; every neuron's autapse acts independently). Degenerates to a plain activated linear layer at $T=1$, since $y_"self" (0) = 0$.],
  pseudocode-list[
    + $y_"self" arrow.l 0$
    + $tilde(y) arrow.l a(z)$
    + *for* $t = 1$ *to* $T$ *do*
      + $y_"self" arrow.l g(tilde(y))$
      + $y arrow.l a(z)$
      + *if* variant is ai2_diff *then* $tilde(y) arrow.l y - y_"self"$
      + *else* $tilde(y) arrow.l y (1 - y_"self" )$
      + *end if*
    + *end for*
    + *return* $tilde(y)$
  ],
  placement: auto,
) <alg:ai2-unroll>

== Continual-Learning Protocol <sec:Protocol>

We adopt a class-incremental protocol on MNIST: each run starts with digits ${0,dots,4}$, then introduces one of the remaining five digits per stage in a total of six stages.
A stage's training set is the union of all digits introduced so far, so that the network must genuinely retain earlier digits while learning new ones.

A stage's duration is fixed to a number of epochs instead of a loss threshold, since a loss-based stopping criterion could over-constrain which regions of the loss landscape a run can explore.

Every stage is evaluated on the full test set:
Already-introduced digits give a forgetting read-out, not-yet-introduced digits a generalization read-out.

== Metrics <sec:metrics>

In every stage, we track: the per-digit test-accuracy matrix; average accuracy over introduced digits; backward transfer, the mean accuracy drop between stage 0 and the final stage on stage 0's digits (the standard forgetting measure); and the mean absolute value of each `AutapticLayer`'s feedback signal $y_"self"$ over an epoch.

== Implementation and Experimental Setup <sec:setup>

The pipeline separates data staging, model definitions, the training loop, MLflow-based logging, and sweep/run orchestration into independent modules.

We compare three controls (a CNN; an FCNN per activation scheme) against the $2 times 2$ ai2 variant/scheme grid at a fixed $T$, plus a one-factor-at-a-time sweep over $T$, epochs per stage, and hidden-layer width to keep the search tractable.

Next to the metrics mentioned in @sec:metrics, a run logs the per-epoch loss, per-stage accuracy, and $y_"self"$ absolute means to a local MLflow instance.

= Preliminary Observations <sec:results>

All numbers below are from a single seed each (@sec:limitations), not yet a statistically supported comparison.
On the base $T=3$, `relu_sigmoid` configuration, the two controls slightly outperform both ai2 variants on final average accuracy (`fcnn_control_relu`: 0.898; `ai2_diff`: 0.872; `ai2_gate`: 0.884) and show less backward transfer (`fcnn_control_relu`: $-0.004$; `ai2_diff`: $0.051$; `ai2_gate`: $0.038$) --- i.e. ai2 currently forgets *more*, not less.
@fig:y-self tracks the last hidden layer's $y_"self"$ absolute mean across training.
`ai2_gate`'s inhibition signal rises then plateaus by stage 2 ($approx$epoch 10), consistent with its multiplicative $(1-y_"self")$ term self-limiting once $y_"self"$ approaches 1.
`ai2_diff`'s signal instead keeps climbing for the full run, which is unsurprising since its subtractive update is not bounded.
The first hidden layer follows the same qualitative pattern in both variants.

#figure(
  image("fig/y_self_absmean.pdf", width: 100%),
  caption: [Mean $|y_"self"|$ of the second hidden layer over training, `ai2_diff` vs. `ai2_gate` (both `relu_sigmoid`, $T=3$, single seed). Dotted lines mark new class-incremental stages.],
  placement: auto,
) <fig:y-self>

@tab:acc-matrix gives the per-stage, per-digit test accuracy for `ai2_diff` (the sweep's anchor configuration); zeros are digits not yet introduced.
Accuracy on already-introduced digits erodes gradually rather than collapsing (e.g. digit 3: 0.91 to 0.88 from stage 0 to 5), while newly introduced digits are picked up within one stage (e.g. digit 5 goes from untrained to 0.63 accuracy the stage after introduction).

#figure(
  table(
    columns: 11,
    align: center,
    table.header([stage], [0], [1], [2], [3], [4], [5], [6], [7], [8], [9]),
    [0], [0.98], [0.98], [0.55], [0.91], [0.89], [0.00], [0.00], [0.00], [0.00], [0.00],
    [1], [0.97], [0.99], [0.80], [0.92], [0.96], [0.26], [0.00], [0.00], [0.00], [0.00],
    [2], [0.96], [0.98], [0.85], [0.91], [0.95], [0.63], [0.81], [0.00], [0.00], [0.00],
    [3], [0.96], [0.98], [0.86], [0.90], [0.92], [0.73], [0.90], [0.83], [0.00], [0.00],
    [4], [0.97], [0.98], [0.86], [0.88], [0.94], [0.75], [0.90], [0.88], [0.76], [0.00],
    [5], [0.97], [0.97], [0.87], [0.88], [0.88], [0.79], [0.92], [0.88], [0.80], [0.76],
  ),
  caption: [Per-stage, per-digit test accuracy, `ai2_diff` (`relu_sigmoid`, $T=3$, single seed). Columns are digits, rows are stages; 0.00 marks a not-yet-introduced digit.],
  placement: auto,
) <tab:acc-matrix>

We attribute this gap to the open issues discussed in @sec:limitations, not a fundamental limitation of autaptic inhibition.

= Discussion and Limitations <sec:limitations>

Everything reported so far is implementation and design, not yet a validated result, for the following reasons.

First, every run so far used a single seed.
Data-loader shuffling draws from one global RNG stream that different configurations consume at different rates, so even same-seed runs are only guaranteed to match through weight initialization and the first training epoch.
Seed-replicated runs, already supported via the logging module's replicate-group mechanism, have not yet been collected.

Second, the $y_"self"$ absolute-mean metric is a heuristic to see if autaptic inhibition takes place.
After wrapping up a systematic exploration of system behaviour, we are going to record every neuron's autaptic inhibition to see how local dynamics evolve over time.

Third, several design decisions were made to get results quickly and will be superseded: the sweep's anchor configuration and value grids, and seeding each run with five initial digits; leaky memory of past activity shall supersede $T$ pseudo-time step unrollings; the ai2 variants and schemes are neither final nor well-calibrated.

Also note that an early comparison using Adam produced practically indistinguishable control and ai2 runs, which we attribute to Adam's adaptive learning rate absorbing the comparatively small change ai2 makes to the loss landscape.
As a response, the optimizer switched to plain SGD, in order to have the learning be more sensitive to the local loss shape.

Lastly, past runs used class-incremental stages which revealed the majority of feature space in the first stage.
The control networks may not learn new representations, but only need to change weights, meaning the runs hardly evaluate continual representation learning.

= Conclusion and Future Work <sec:Conclusion>

We implemented and tested a self-recurrent inhibitory mechanism (ai2) with the goal of pushing networks toward distributed, rather than locally concentrated, representations, and built a class-incremental MNIST pipeline to evaluate it.
The pipeline runs end-to-end and is unit- and integration-tested; what remains is a thorough analysis of whether ai2 improves continual-learning outcomes over its controls.

The immediate next steps are seed-replicated runs for variance estimates; sweeps over more constrained network and training budgets, where an ai2 benefit, if any, should be clearest; a learning-rate sweep; and a genuine activity-recording metric alongside $y_"self"$ to analyse local dynamics.
