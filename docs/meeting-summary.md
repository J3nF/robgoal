# Meeting Summary: Incremental Autaptic-Inhibition Experiments

Status: Phase 1 (coarse) and Phase 2 (exhaustive) are finished for MNIST, FashionMNIST and the mixed 20-class protocol.
Setup: width 12, four equal hidden layers, class-incremental protocol with 1 initial class and 1 class per stage, 5 random seeds shared by all models of a launch.
Coarse runs use 1 epoch per stage (mixed also 25 % of every stage's training examples); exhaustive runs use 5 epochs per stage and all training examples.
Every figure title says "coarse" or "exhaustive"; paths are relative to the repository root.
Spreads are sample standard deviations (n−1) over 5 seeds; intervals are 95 % t-intervals.

## Read This First

- No robustness benefit of inhibition was found at width 12 with `relu_sigmoid`, on any dataset.
  The unrolled gate equals the FCNN control on accuracy and backward transfer; leaky layers are equal or worse, and only λ=0.99 comes close.
- The `sigmoid` scheme is uninformative.
  In the coarse runs (1 epoch) every `sigmoid` model, controls included, is at chance at every pilot width.
  In the exhaustive runs the plain `sigmoid` FCNN control learns (accuracy 0.65 MNIST, 0.71 FashionMNIST, 0.80 mixed), but the unrolled and leaky `sigmoid` models stay at chance on MNIST and FashionMNIST.
  On the mixed protocol the unrolled `sigmoid` gate learns (0.61), and the leaky `sigmoid` layers stay at chance (0.05).
- Backward transfer of `sigmoid` models is positive when they learn (+0.17 to +0.34) because their newest-class accuracy at introduction is low (0.36 to 0.51 for the `sigmoid` FCNN control against 0.89 to 0.95 for `relu_sigmoid`), so later learning counts as gain.
  The coarse mixed leaky layer's +0.49 has the same cause (newest-class accuracy 0.19).
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

Exhaustive MNIST, launch `2026-11-17-16-00_ef03c45` (55 runs, 11 configs: leaky λ = 0.5, 0.9, 0.99 added); figures in `results/plots/exhaustive/` (same file names as the coarse figures).

| Model (MNIST, width 12, 5 seeds) | Final accuracy (mean ± std) | Final backward transfer D, 95 % interval |
| --- | --- | --- |
| CNN control | 0.980 ± 0.009 | −0.006 [−0.014, +0.002] |
| FCNN control, `relu_sigmoid` | 0.934 ± 0.019 | −0.021 [−0.058, +0.016] |
| Unrolled `ai2_gate`, `relu_sigmoid` | 0.934 ± 0.004 | −0.027 [−0.034, −0.021] |
| Leaky λ=0.5, `relu_sigmoid` | 0.882 ± 0.032 | −0.032 [−0.088, +0.024] |
| Leaky λ=0.9, `relu_sigmoid` | 0.916 ± 0.012 | −0.032 [−0.047, −0.018] |
| Leaky λ=0.99, `relu_sigmoid` | 0.924 ± 0.009 | −0.029 [−0.036, −0.022] |
| FCNN control, `sigmoid` | 0.649 ± 0.052 | +0.248 [+0.025, +0.471] |
| Unrolled `ai2_gate`, `sigmoid` | 0.105 ± 0.012 | −0.127 [−0.246, −0.009] |
| Leaky, `sigmoid` (all λ) | 0.100 ± 0.000 | −0.111 [−0.114, −0.110] |

- On MNIST, interference is neither clearly present nor absent for the FCNN control at W=12: its interval [−0.058, +0.016] straddles the pilot thresholds (−0.05 and −0.02).
- The unrolled gate and the leaky layers are not more robust (D −0.027 and −0.029 to −0.032 against −0.021); the leaky λ=0.5 accuracy loss (0.882) is the only clear difference.

Exhaustive FashionMNIST, launch `2026-11-17-16-44_ef03c45` (55 runs), same figure folder (`*_fashion_mnist*`):

| Model (FashionMNIST, width 12, 5 seeds) | Final accuracy (mean ± std) | Final backward transfer D, 95 % interval |
| --- | --- | --- |
| CNN control | 0.892 ± 0.003 | −0.021 [−0.045, +0.003] |
| FCNN control, `relu_sigmoid` | 0.851 ± 0.002 | −0.055 [−0.099, −0.010] |
| Unrolled `ai2_gate`, `relu_sigmoid` | 0.848 ± 0.006 | −0.055 [−0.094, −0.016] |
| Leaky λ=0.5, `relu_sigmoid` | 0.780 ± 0.073 | −0.067 [−0.117, −0.017] |
| Leaky λ=0.9, `relu_sigmoid` | 0.838 ± 0.007 | −0.062 [−0.098, −0.026] |
| Leaky λ=0.99, `relu_sigmoid` | 0.834 ± 0.006 | −0.028 [−0.065, +0.008] |
| FCNN control, `sigmoid` | 0.706 ± 0.067 | +0.172 [−0.003, +0.348] |
| Unrolled `ai2_gate` and leaky, `sigmoid` | 0.100 ± 0.000 | −0.22 to −0.29 (intervals up to [−0.50, −0.03]) |

- On FashionMNIST the FCNN control's interval [−0.099, −0.010] is also neither clearly present nor absent under the pilot definition (present needs the upper bound below −0.05), so the interference contrast with MNIST is not established.
- The unrolled gate has the same backward transfer as the control (−0.055); leaky λ=0.99 has −0.028 with an interval that includes 0, which 5 seeds cannot distinguish from the control.
- Larger λ does not help accuracy here (0.780, 0.838, 0.834), unlike MNIST (0.882, 0.916, 0.924).
- The unrolled gate and the leaky layers with `sigmoid` stay at chance.

Exhaustive mixed (20 classes, labels 0–9 MNIST, 10–19 FashionMNIST, class order random per seed), launch `2026-11-17-17-28_ef03c45` (55 runs, about 2 h 40 min), figures `results/plots/exhaustive/*_mixed*`:

| Model (mixed, width 12, 5 seeds) | Final accuracy (mean ± std) | Final backward transfer D, 95 % interval |
| --- | --- | --- |
| CNN control | 0.932 ± 0.006 | −0.026 [−0.037, −0.016] |
| FCNN control, `relu_sigmoid` | 0.865 ± 0.007 | −0.039 [−0.052, −0.026] |
| Unrolled `ai2_gate`, `relu_sigmoid` | 0.868 ± 0.007 | −0.045 [−0.059, −0.032] |
| Leaky λ=0.5, `relu_sigmoid` | 0.693 ± 0.046 | −0.106 [−0.185, −0.028] |
| Leaky λ=0.9, `relu_sigmoid` | 0.836 ± 0.012 | −0.043 [−0.066, −0.021] |
| Leaky λ=0.99, `relu_sigmoid` | 0.860 ± 0.009 | −0.033 [−0.062, −0.004] |
| FCNN control, `sigmoid` | 0.798 ± 0.033 | +0.276 [+0.193, +0.359] |
| Unrolled `ai2_gate`, `sigmoid` | 0.609 ± 0.120 | +0.340 [+0.234, +0.446] |
| Leaky, `sigmoid` (all λ) | 0.050 ± 0.000 (chance) | −0.21 [−0.33, −0.09] |

- With `relu_sigmoid`, the unrolled gate matches the FCNN control on accuracy (0.868 versus 0.865) and backward transfer (−0.045 versus −0.039, intervals overlap).
- Leaky inhibition ends at or below the control; only λ=0.99 comes close (0.860).
- The `sigmoid` FCNN control and the unrolled gate learn on 20 classes; their positive D comes from low accuracy at introduction.
- The mixed heatmaps are by class label and, in `heatmap_mixed_by_rank.*`, by introduction rank; the per-dataset normalisation could let a model separate the datasets by input statistics alone.
- MLflow logging failed with "database is locked" in 7 of the 55 jobs, starting 17:30, because 12 workers wrote to one SQLite file.
  The `.h5` files of all 55 runs exist and hold 100 epochs each (20 stages × 5 epochs), so the figures are complete.
  The launcher exited with status 1 because of this, so the chain script did not plot automatically; the plots were made by running `src/plot_incremental.py` manually.

### Exhaustive Figures (Items 2.0, 2.1, 2.3, 2.4)

All in `results/plots/exhaustive/` (PNG and PDF), launches as above, all exhaustive; the dataset is in the file name.

| Figure | Finding |
| --- | --- |
| `heatmap_<dataset>.png`, `heatmap_mixed_by_rank.png` | `relu_sigmoid` models keep old classes after introduction (mixed: yellow below the introduction diagonal, with slow fade for leaky λ=0.5); `sigmoid` models are mostly dark, and the leaky `sigmoid` layers on mixed learn only about half a dozen classes. |
| `newest_class_<dataset>.png` | `relu_sigmoid` models learn each new class to 0.85 to 1.0 (MNIST), with leaky λ=0.5 dipping to 0.74 at stage 5; `sigmoid` unrolled and leaky models fall to 0 newest-class accuracy from stage 2 (MNIST). |
| `inhibition_mean_<dataset>_<model>.png`, `inhibition_var_<dataset>_<model>.png`, `inhibition_eval_<dataset>.png` | Test-set inhibition mean in layer 3, first to last stage: unrolled gate falls (MNIST 0.11 to 0.07, mixed 0.13 to 0.06); leaky λ=0.99 rises (MNIST 0.26 to 0.51, FashionMNIST 0.17 to 0.42, mixed 0.25 to 0.50); leaky `sigmoid` stays near 0.5; the unrolled `sigmoid` gate stays near 0.3 at chance but falls to 0.06 on mixed, where it learns. |

## What Was Run

- Pilot: 7 widths × (2 FCNN controls + 1 CNN control) × 5 seeds, 1 epoch per stage, MNIST (105 runs).
- Three coarse launches, 35 runs each: CNN control (matched channels), 2 FCNN controls, 2 unrolled `ai2_gate` (T=5, weight 1), 2 leaky (λ=0.9), for MNIST, FashionMNIST and the mixed protocol.
- `ai2_diff` was not run (TRIAGE.md #1).
- Per-layer inhibition mean and variance were recorded per training batch (with the stage) and per stage end on the test set (also per class), for the unrolled and leaky models only.

## Next

The plan and its red-team record are in `docs/exhaustive-plan.md`.
The exhaustive chain is complete.
