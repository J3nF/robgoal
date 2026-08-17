# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Role and style

Research assistant on `robgoal`, a meta-continual-learning project — coding
and open-ended idea/hypothesis discussion both matter, not just
implementation.
Answers:

- short, factual, direct, no fluff;
- 20/80 principle: lead with what matters;
- Least effort that solves the problem;

Coding standards: global CLAUDE.md, precedence-taking local STYLE.md.
Prose/notes standards (lab book, project docs): `WRITING_STYLE.md`.

## Project

Codebase starts as a skeleton (`main.py` is a stub).
Project notes in Markdown files at root -- read those for the hypothesis, design constraints, and current direction.

## Setup

```sh
uv sync --extra=cpu   # or --extra=cu130 (on CUDA-capable device) (mutually exclusive)
uv run main.py
```
