# robgoal

A multi-week project on implementing neural networks capable of robust goal-keeping; specifically tackling catastrophic forgetting of features.

More detailed notes can be found in  the `docs` folder, particularly in `docs/Mini_project_meta_continual_learning.md`.

## Set-up

Use [`uv`](https://docs.astral.sh/uv/), together with the preferred Pytorch version (CPU-only or CUDA), to reproduce our Python environment:

```sh
uv sync --extra=cpu

# or, if you want to use CUDA:
uv sync --extra=cu130
```
