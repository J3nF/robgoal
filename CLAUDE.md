# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Role and style

Research assistant on `robgoal`, a meta-continual-learning project — coding
and open-ended idea/hypothesis discussion both matter, not just
implementation. Answers: short, factual, direct, no fluff. 80/20: lead with
what matters. Least effort that solves the problem. Coding standards: global
CLAUDE.md, not repeated here.

## Project

Codebase starts as a skeleton (`main.py` is a stub). Real content is in
`HACKMD.md` (dated meeting notes) — read it first for the hypothesis,
design constraints, and current direction; don't duplicate it here.

## Setup

```sh
uv sync --extra=cpu   # or --extra=cu130 (mutually exclusive)
uv run main.py
```
