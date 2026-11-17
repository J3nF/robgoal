# Exhaustive Launch Plan (Phase 2)

## Runtime Estimate

Source: summed `train_seconds` of the coarse launches (wall-clock seconds of the training calls, summed over the 35 runs of a launch, so close to CPU seconds with one thread per worker): 4180 s (MNIST), 4231 s (FashionMNIST), 10 314 s (mixed, 25 % of the training examples).
The coarse MNIST launch took 6 min 55 s against an ideal 4180 / 12 = 348 s, an efficiency of 84 % (scheduling tail and start-up); applying it to the 55-job launches is an assumption, because job sizes and the tail differ.

Assumptions: training time is proportional to epochs times training examples, evaluation time to epochs only (the share is not separated, so the data scaling is an upper bound).
Cost per config (units of an FCNN control): unrolled `ai2_gate` about 2 (T=5 passes), leaky about 1, CNN about 1.5; coarse mix is 1 CNN + 2 controls + 2 gate + 2 leaky = 1.5 + 2 + 4 + 2 = 9.5.
Adding four leaky configs (+4) gives 13.5, a factor of 1.42 (an assumption about relative cost, not measured).

- MNIST exhaustive: 4180 × 5 × 1.42 = 29 700 s CPU, divided by 12 and by 0.84: about 49 min.
- FashionMNIST exhaustive: about 50 min.
- Mixed, all training examples: 10 314 × 4 × 5 × 1.42 = 293 000 s CPU, which is 8.1 h wall clock on 12 workers at 84 % efficiency; with the evaluation share not scaling by data the real time is probably 6 to 8 h.

## Configurations

Per dataset, 11 configs × 5 seeds = 55 runs: CNN control (matched channels), 2 FCNN controls, 2 unrolled `ai2_gate` (T=5, weight 1), and leaky layers with λ in {0.5, 0.9, 0.99} for both schemes.
Width 12, 5 epochs per stage, 1 initial class, 1 class per stage, 5 shared seeds, `--split-seeds`, `--workers 12`, MLflow on, goal text says "exhaustive".
The mixed launch has a random class order per seed and uses all training examples (the task says "full 5 epochs per stage"; a reviewer pointed out that subsampling is for coarse runs only).
The mixed launch runs last, because it takes 6 to 8 h; the researcher can stop it, and the two base launches are finished and plotted first.
Memory is not a risk for the full mixed data: every worker process loads its own copy of the dataset (torchvision keeps the images as uint8), and the coarse mixed launch already ran 12 workers; the training fraction only shrinks the per-stage index lists.

## Launch Procedure

Precondition: all Phase 1 figures and `docs/meeting-summary.md` are finished and committed (they are, at commit `0b7e07c` and later).
The launcher re-checks the git tree before each job starts (a dirty tree makes the remaining jobs fail, as happened to the first pilot attempt), so all launches run from the clean worktree `/home/vscode/runner`.
The script `exhaustive.sh` (in the session scratchpad) does, in order:

1. Discards the worktree's lab-book changes (they were copied to `/workspace/docs/lab-book.md` after each earlier launch) and moves the worktree with `git -C /home/vscode/runner checkout --detach in_the_weeds`; HEAD must be committed and clean.
2. For each dataset in the order `mnist`, `fashion_mnist`, `mixed`, runs the launch below, and only if the previous one exited with status 0.
3. After each launch, appends the lines the launcher added to the worktree's lab book to the authoritative `/workspace/docs/lab-book.md` (appended lines only), then plots that launch into `results/plots/exhaustive/`, so the base figures exist before the mixed launch ends.

The worktree's `mlflow.db`, `results/` and `data/` are symlinks into `/workspace`, so the MLflow viewer on port 5000 and all result files are shared.
Python is the uv environment (`UV_PROJECT_ENVIRONMENT=/workspace/.venv uv run --no-sync python`).
After a kill or a failed launch, a new launch is started (new launch id, new lab-book entry citing the old one); finished runs of the old launch stay but are not merged.

```sh
UV_PROJECT_ENVIRONMENT=/workspace/.venv uv run --no-sync python src/run_experiments.py \
  --scan incremental --width 12 --leaky-decays 0.5 0.9 0.99 --dataset <dataset> \
  --initial-classes 1 --classes-per-stage 1 --num-seeds 5 --epochs-per-stage 5 \
  --split-seeds --workers 12 --goal "Exhaustive run, Steps 2-6: ..." \
  --method "Exhaustive: 5 epochs per stage, all training examples, ..." --code-changes "..."
uv run python src/plot_incremental.py results/<launch id> --out-dir results/plots/exhaustive --tag exhaustive
```

The full goal, method and code-changes strings are in `exhaustive.sh` and end up verbatim in the lab book.
Figures go to `results/plots/exhaustive/`; the coarse figures are not overwritten.

## Checks Built Into the Runs

- The controls' final backward transfer at W=12 with 5 epochs is reported with its 95 % interval, to test whether the pilot's width choice (made at 1 epoch) still sits in the transition region.
- The sigmoid scheme is re-examined: if it trains at 5 epochs the figures become informative, if not it stays a documented floor.

## Known Gap: Telling Launches Apart in MLflow

The launch goal text ("coarse" or "exhaustive") goes into the lab book but not into MLflow, which tags runs with `launch_id`, commit and branch only.
In the MLflow UI, coarse and exhaustive runs are told apart by `launch_id` and by the logged parameters `epochs_per_stage` (1 versus 5) and `train_fraction`.
Adding the goal as a tag would need the goal to travel from the launch spec into `Launch`, which is out of scope here.
