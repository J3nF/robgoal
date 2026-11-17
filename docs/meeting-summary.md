# Meeting Summary: Incremental Autaptic-Inhibition Experiments

Status: Phase 1 (coarse) is complete; the exhaustive launches for MNIST and FashionMNIST are finished and the mixed protocol (full data, 6 to 8 hours estimated) is still running (see "Phase 2" below).
Setup: width 12, four equal hidden layers, class-incremental protocol with 1 initial class and 1 class per stage, 5 random seeds shared by all models of a launch, 1 epoch per stage.
The mixed protocol additionally uses 25 % of every stage's training examples.
Every figure title says "coarse"; paths are relative to the repository root.

## Read This First

- The `sigmoid` activation scheme stays at chance for every model, including the controls, at every width of the pilot (final accuracy 0.10 to 0.12 on MNIST).
  Its columns cannot show a robustness effect; this may be a result of the coarse 1-epoch setting (the 5-epoch runs will tell).
- Backward transfer is near zero for the unrolled and leaky layers partly because a network that has not learned its newest class has little to forget.
  On the mixed protocol the leaky layer's final backward transfer is +0.49 because its newest-class accuracy is only 0.19 at the end of the introducing stage, so later learning counts as positive transfer.
- The width W=12 comes from the pilot with a fallback rule (the `sigmoid` scheme could not take part) and its interval is wide; see `docs/old-lab-book.md`, "Choosing the Network Width W".

## Figures

| # | Item | Figure (PNG; PDF next to it) | Finding | Launch id | Coarse or exhaustive |
| --- | --- | --- | --- | --- | --- |
| 1 | Size pilot | `results/plots/size_pilot/width_pilot.png`, `width_table.csv` | Only the `relu_sigmoid` FCNN shows interference that shrinks with width (final backward transfer −0.17 at W=2, about 0 at W=64); the CNN never does; W=12 chosen. | `2026-11-17-14-54_02ca940` | coarse |
| 2 | 2.0 Heatmap, MNIST | `results/plots/coarse/heatmap_mnist.png` | The `relu_sigmoid` models keep most old classes (diagonal introduction, slow fade below the diagonal); `sigmoid` models only ever predict class 1. | `2026-11-17-15-15_f7ca382` | coarse |
| 3 | 2.0 Heatmap, FashionMNIST | `results/plots/coarse/heatmap_fashion_mnist.png` | Same pattern with lower accuracy (final average 0.77 to 0.87 for `relu_sigmoid` models and CNN). | `2026-11-17-15-22_f7ca382` | coarse |
| 4 | 2.0 Heatmap, mixed 20 classes | `results/plots/coarse/heatmap_mixed.png` (by label, MNIST and FashionMNIST marked), `heatmap_mixed_by_rank.png` (by introduction rank, diagonal marks) | Controls reach 0.79 (FCNN) and 0.84 (CNN) final accuracy; unrolled gate 0.73; leaky 0.69; `sigmoid` models are at chance (0.05). | `2026-11-17-15-29_f7ca382` | coarse |
| 5 | 2.1 Newest-class curves | `results/plots/coarse/newest_class_mnist.png`, `newest_class_fashion_mnist.png`, `newest_class_mixed.png` | The unrolled gate matches the FCNN control on newest-class accuracy and backward transfer (MNIST final accuracy 0.916 versus 0.920); the leaky layer learns new classes more slowly (MNIST newest-class accuracy 0.83 versus 0.93). | the three launches above | coarse |
| 6 | 2.3 Leaky layer versus unrolled gate | same files as 5 and 2 | Leaky inhibition (λ=0.9) lowers final accuracy (MNIST 0.858 versus 0.916) and keeps backward transfer at or above 0 (+0.018 MNIST, −0.015 FashionMNIST, versus −0.019 and −0.040 for the unrolled gate). | the three launches above | coarse |
| 7 | 2.4 Inhibition per layer | `results/plots/coarse/inhibition_mean_<dataset>_<model>.png`, `inhibition_var_<dataset>_<model>.png`, comparison `inhibition_eval_<dataset>.png` | The unrolled gate's inhibition mean falls slowly with stages (0.13 to 0.07 in layer 3 on MNIST); the leaky layer's rises with depth and stages (mixed: 0.06 to 0.31 in layer 3); the `sigmoid` layers sit at a constant value. | the three launches above | coarse |

## Phase 2: Exhaustive Runs (5 Epochs per Stage, Full Training Sets)

Exhaustive MNIST, launch `2026-11-17-16-00_ef03c45` (55 runs, 11 configs: leaky λ = 0.5, 0.9, 0.99 added); figures in `results/plots/exhaustive/` (same file names as the coarse figures, titles say "exhaustive").

| Model (MNIST, width 12, 5 seeds) | Final accuracy (mean ± std) | Final backward transfer D, 95 % interval |
| --- | --- | --- |
| CNN control | 0.980 ± 0.008 | −0.006 [−0.014, +0.002] |
| FCNN control, `relu_sigmoid` | 0.934 ± 0.017 | −0.021 [−0.058, +0.016] |
| Unrolled `ai2_gate`, `relu_sigmoid` | 0.934 ± 0.003 | −0.027 [−0.034, −0.021] |
| Leaky λ=0.5 / 0.9 / 0.99, `relu_sigmoid` | 0.882 / 0.916 / 0.924 | −0.032 / −0.032 / −0.029 |
| FCNN control, `sigmoid` | 0.649 ± 0.046 | +0.248 [+0.025, +0.471] |
| Unrolled `ai2_gate`, `sigmoid` | 0.105 ± 0.011 | −0.127 [−0.246, −0.009] |
| Leaky, `sigmoid` (all λ) | 0.100 | −0.111 |

Exhaustive FashionMNIST, launch `2026-11-17-16-44_ef03c45` (55 runs), same figure folder (`*_fashion_mnist*`):

| Model (FashionMNIST, width 12, 5 seeds) | Final accuracy (mean ± std) | Final backward transfer D, 95 % interval |
| --- | --- | --- |
| CNN control | 0.892 ± 0.003 | −0.021 [−0.045, +0.003] |
| FCNN control, `relu_sigmoid` | 0.851 ± 0.002 | −0.055 [−0.099, −0.010] |
| Unrolled `ai2_gate`, `relu_sigmoid` | 0.848 ± 0.005 | −0.055 [−0.094, −0.016] |
| Leaky λ=0.5 / 0.9 / 0.99, `relu_sigmoid` | 0.780 / 0.838 / 0.834 | −0.067 / −0.062 / −0.028 |
| FCNN control, `sigmoid` | 0.706 ± 0.060 | +0.172 [−0.003, +0.348] |
| Unrolled `ai2_gate` and leaky, `sigmoid` | 0.100 | −0.22 to −0.29 |

- On FashionMNIST the FCNN control shows interference at W=12 (whole interval below −0.01, upper bound −0.010), unlike on MNIST; unrolled gate and leaky layers do not reduce it, except possibly leaky λ=0.99 (interval includes 0, +0.008 upper bound), which is not distinguishable from the control with 5 seeds.
- At 5 epochs the plain `sigmoid` FCNN control does learn (accuracy 0.65), but the unrolled gate and the leaky layers with `sigmoid` stay at chance: inhibition with this scheme prevents training at width 12 here.
- With `relu_sigmoid`, the unrolled gate and the leaky layers are not more robust than the FCNN control (backward transfer −0.03 against −0.02, intervals overlap), and larger λ gives higher final accuracy (0.88 to 0.92).
- The FCNN control's interval at W=12 straddles the "present" and "absent" thresholds of the pilot (−0.05 and −0.02), so at 5 epochs interference is neither clearly present nor absent, consistent with the width being in the transition region.
- The positive backward transfer of the `sigmoid` control (+0.25) is an artefact of near-chance accuracy at introduction.

## What Was Run

- Pilot: 7 widths × (2 FCNN controls + 1 CNN control) × 5 seeds, 1 epoch per stage, MNIST (105 runs).
- Three coarse launches, 35 runs each: CNN control (matched channels), 2 FCNN controls, 2 unrolled `ai2_gate` (T=5, weight 1), 2 leaky (λ=0.9), for MNIST, FashionMNIST and the mixed protocol.
- `ai2_diff` was not run (TRIAGE.md #1).
- Per-layer inhibition mean and variance were recorded per training batch (with the stage) and per stage end on the test set (also per class), for the unrolled and leaky models only.

## Next

The plan and its red-team record are in `docs/exhaustive-plan.md`.
The mixed exhaustive launch (full data) is estimated at 6 to 8 hours and is the last in the chain.
