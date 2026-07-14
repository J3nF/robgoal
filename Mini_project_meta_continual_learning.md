---
title: Mini_project_meta_continual_learning

---

# Mini-project on meta-continual learning

## Meeting notes

### 2026-06-19

- Immediate task: Literature / field review (see Zothero)

https://avalanche.continualai.org/ - Standard framework for CL.

Take a look at the new folder "Theoretical works" and on the paper
"When Meta-Learning Meets Online and Continual Learning: A Survey" under Meta Learning + Continual Learning. 

I was reading a paper that is under review for NeurIPS and their idea on meta incremental learning is quite interesting. I've create a new latex in our overleaf under "meta_incremental_learning", take a look and let me know what do you think. 


### 2026-06-26

#### Pre-note

The question we want to answer at the end is similar to the "No Forgetting Learning paper" **"How can we design a memory-efficient CL framework that operates within the fixed capacity of the backbone network without sacrificing performance?"**

#### Notes

- We rather want to focus on how to distribute information across the network (instead of finding novel learning rules).
- We wish to find measures capturing *learning utility* of neurons (to help diagnose "static" learning).
- Agreement about:
    - Fixed capacity
    -  Selective learning activation of neurons/network parts
    -  Looking at *forgetting* as a usable tool for shifting (meta-)representations (never forgetting forces one to stay in local basins, after all)
    -  Model-agnostic solutions!
- Focus on class incremental learning right now (due to time constraint).

Deliverables for next Tuesday: **How do people categorize neuron's informational capacity?**

### 2026-06-30

#### Pre-note

Start actionable methodological implementation discussions.
The coming week is going to include implementation!