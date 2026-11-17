# Exhaustive Launch Plan (Phase 2)

## Runtime Estimate

Source: summed `train_seconds` of the coarse launches (wall-clock seconds of the training calls, summed over the 35 runs of a launch, so close to CPU seconds with one thread per worker): 4180 s (MNIST), 4231 s (FashionMNIST), 10 314 s (mixed, 25 % of the training examples).
The coarse MNIST launch took 6 min 55 s against an ideal 4180 / 12 = 348 s, an efficiency of 84 % (scheduling tail and start-up).

Assumptions: training time is proportional to epochs times training examples, evaluation time to epochs only (the share is not separated, so the data scaling is an upper bound).
Cost per config (units of an FCNN control): unrolled `ai2_gate` about 2 (T=5 passes), leaky about 1, CNN about 1.5; coarse mix is 1 CNN + 2 controls + 2 gate + 2 leaky = 1.5 + 2 + 4 + 2 = 9.5.
Adding four leaky configs (+4) gives 13.5, a factor of 1.42 (an assumption about relative cost, not measured).

- MNIST exhaustive: 4180 × 5 × 1.42 = 29 700 s CPU, divided by 12 and by 0.84: about 49 min.
- FashionMNIST exhaustive: about 50 min.
- Mixed, all training examples: 10 314 × 4 × 5 × 1.42 = 293 000 s CPU, about 8.1 h; with the evaluation share not scaling by data the real time is probably 6 to 8 h.

## Configurations

Per dataset, 11 configs × 5 seeds = 55 runs: CNN control (matched channels), 2 FCNN controls, 2 unrolled `ai2_gate` (T=5, weight 1), and leaky layers with λ in {0.5, 0.9, 0.99} for both schemes.
Width 12, 5 epochs per stage, 1 initial class, 1 class per stage, 5 shared seeds, `--split-seeds`, `--workers 12`, MLflow on, goal text says "exhaustive".
The mixed launch has a random class order per seed and uses all training examples (the task says "full 5 epochs per stage"; a reviewer pointed out that subsampling is for coarse runs only).
The mixed launch runs last, because it takes 6 to 8 h; the researcher can stop it, and the two base launches are finished and plotted first.
Memory is not a risk for the full mixed data: the 25 % coarse launch already ran 12 workers on the same dataset objects (only the index lists differ), and torchvision keeps the images as uint8.

## Launch Procedure

The launcher re-checks the git tree before every job, so all launches run from the clean worktree `/home/vscode/runner`, moved with `git -C /home/vscode/runner checkout --detach in_the_weeds` right before the launch (HEAD must be committed and clean).
Its `mlflow.db`, `results/` and `data/` are symlinks into `/workspace`, so the MLflow viewer on port 5000 and all result files are shared.
The authoritative lab book is `/workspace/docs/lab-book.md`: the script `exhaustive.sh` (scratchpad) appends each launch's new lines from the worktree's copy after the launch, and I commit the change on `in_the_weeds`.
The three launches are chained: each starts only after the previous one exited with status 0.
After a kill or a failed launch, a new launch is started (new launch id, new lab-book entry citing the old one); finished runs of the old launch stay but are not merged.

```sh
for dataset in mnist fashion_mnist mixed; do
  python src/run_experiments.py --scan incremental --width 12 --leaky-decays 0.5 0.9 0.99 \
    --dataset $dataset --initial-classes 1 --classes-per-stage 1 --num-seeds 5 \
    --epochs-per-stage 5 --split-seeds --workers 12 \
    --goal "Exhaustive run, Steps 2-6 ..." --method "..." --code-changes "..." || break
done
uv run python src/plot_incremental.py results/<three launch ids> --out-dir results/plots/exhaustive --tag exhaustive
```

Figures go to `results/plots/exhaustive/`; the coarse figures are not overwritten.

## Checks Built Into the Runs

- The controls' final backward transfer at W=12 with 5 epochs is reported with its 95 % interval, to test whether the pilot's width choice (made at 1 epoch) still sits in the transition region.
- The sigmoid scheme is re-examined: if it trains at 5 epochs the figures become informative, if not it stays a documented floor.
