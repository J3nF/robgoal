# Exhaustive Launch Plan (Phase 2)

## Runtime Estimate

Source: summed `train_seconds` of the coarse launches (CPU seconds over all 35 runs) divided by 12 workers.

- MNIST coarse: 4180 s CPU, 6.9 min wall clock (launch 15:15:33 to 15:22:28).
- FashionMNIST coarse: 4231 s CPU, about 7 min.
- Mixed coarse: 10 314 s CPU, about 14 min (25 % of the training sets, 1 epoch).

Scaling (assumption: training time proportional to epochs times training examples, evaluation time proportional to epochs; the evaluation share is not separated):

- 5 epochs: ×5 CPU time.
- Two more leaky decays (0.5 and 0.99): 4 more configs of 7, so about ×1.35 (leaky cost is close to an `T=1` FCNN, between control and unrolled gate cost).
- MNIST exhaustive: 4180 × 5 × 1.35 = 28 200 s CPU, about 39 min on 12 workers.
- FashionMNIST exhaustive: about 40 min.
- Mixed with all training examples: 10 314 × 4 (data) × 5 (epochs) × 1.35 = 278 000 s CPU, about 6.4 h.
- Mixed with half the training examples: about 3.2 h.

Uncertainty: the estimates assume linear scaling; contention with my own work during the coarse launches probably inflated the coarse times, so the real times may be up to 20 % shorter (about as likely as not).

## Configurations

Per dataset, 11 configs × 5 seeds = 55 runs: CNN control (matched channels), 2 FCNN controls, 2 unrolled `ai2_gate` (T=5, weight 1), and leaky layers with λ in {0.5, 0.9, 0.99} for both schemes.
Width 12, 5 epochs per stage, 1 initial class, 1 class per stage, 5 shared seeds, `--split-seeds`, `--workers 12`, MLflow on, goal text says "exhaustive".
The mixed launch has a random class order per seed.

Decision (agent, unconfirmed): the mixed exhaustive launch runs after the two base launches and keeps 5 epochs; the training fraction is the open question.
Full data costs about 6.4 h, half the data about 3.2 h; I plan half the data (`--train-fraction 0.5`) unless the researcher asks for full data, and label it so in the lab book and the figure titles.

## Launch Commands

Run from the clean runner worktree (`/home/vscode/runner`, checked out at the committed HEAD) so that editing the main checkout cannot dirty the launched tree:

```sh
python src/run_experiments.py --scan incremental --width 12 --leaky-decays 0.5 0.9 0.99 \
  --dataset {mnist,fashion_mnist} --initial-classes 1 --classes-per-stage 1 --num-seeds 5 \
  --epochs-per-stage 5 --split-seeds --workers 12 --goal "Exhaustive ..." --method "..." --code-changes "..."
python src/run_experiments.py ... --dataset mixed --train-fraction 0.5 ...
```

Figures go to `results/plots/exhaustive/` (`plot_incremental.py --tag exhaustive`); the coarse figures are not overwritten.

## Checks Built Into the Runs

- The controls' final backward transfer at W=12 with 5 epochs is reported with its 95 % interval, to test whether the pilot's width choice (made at 1 epoch) still sits in the transition region.
- The sigmoid scheme is re-examined: if it trains at 5 epochs the figures become informative, if not it stays a documented floor.
