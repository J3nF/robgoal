# Mini-project on meta-continual learning

## Line of argument

- We think local information overload is the root of all our troubles (i.e. catastrophic forgetting).
- We therefore wish to distribute information.
- We think this overlaps with keeping neural networks flexible by pushing them out of minima somewhat^[This *somewhat* is doing some heavy lifting to hide my confusion!].
    - A subset of this statement is that forgetting can be looked at as useful for our purposes.
- We believe one way to this goal is to allow different neuronal behaviour given similar/same inputs.
- Therefore, we wish to let network parts^[who knows what this means as of now] have self-recurrent, self-inhibiting connections.
    - As the Bio people scream right now, we wish to use *autapses*.
- By "self-inhibiting", we hope for no single neurons to be solely responsible for elemental concepts.
- Implication: Many neurons share important information.
- Hypothesis: This diffuses information across the network. This diffusion tackles local information overloading. Voila.

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

### Notes

- We discussed papers around measuring informational contributions of single neurons
- Some ways to do so (roughly):
    - Neuron's contribution to cost function.
    - How much of the gradient a neuron "absorbs" when weights are updates.
    - Noisy neurons.
        - Connor: Add noise to weights.
        - Gabriel: Add noise to neuronal outputs.
- Q. Connor: Once we classified neurons' informational contribution, what is the next step?
    - A: Decide which networks contribute most, keep them fixed/learn slower.
        - Underlying Hypothesis: The information-heavy neurons store important representations.

#### Activity-dependent local plasticity

- Connor: Methabolic rate depends on neuron's activity and can be exhausted.
- Jens: Implement inhibition as self-recurrent connection of neuron.
- Both would force a network to spread information, instead of just using single neurons/pathways.
- Right now, focus on inhibition-like mechanism.
- Questions re: implementation details:
    - Inhibit single neurons vs. network parts ( layer level / channel level)?
        - If network parts: What level of granularity?
    - Shape of inhibition?
        - Decaying vs RELU vs ...
        - Note: If implemented as self-recurrent connection, time scale is 1 step if using binary inhibition.
            - This explicitly separates this approach from simple masking.
    - Think about how the inhibition weight would be included in the update mechanism.
    - If not going for single neurons: How to categorise channels / pathways as such?


### 2026-07-06

- Jens wrote down our line of argument (to be added here!!)
- Short literature review found no past work using self-recurrent connections in continual learning.
    - Biological side: These *autapses* contribute to self-regulation of neurons and network-wide dynamics.
    - ANN side: Some people stumbled on the idea and said: "Look, performs slightly better than simple RNNs."
- **Main point discussed**: Neural activity $\neq$ informational importance!
    - Q: How to identify these neurons?
    - After some discussion, we settle on ***focusing on implementation for the coming week, and returning to this point with fresh eyes***.

#### Implementation

- Go for the self-recurrent network as of now.
- Use MNIST data for now.
- Start with 5 digits, see how the network generalises afterwards.
    - Q: Add one by one or the whole held-back set?
    - A: One by one enables clearer results with regard to "novelty" of new data.
- re: Network size
    - If using a size close to capacity exhaustion, we hopt to see more stable learning and less overfitting with our approach.
    - If using huge networks, we would hope to see more network parts being active as we want to diffuse information.
- Which network to use?
    - MLP (as non-convolutional one) (one small, one large)
    - AlexNet (as a large convolutional one), VGG (19, 30 layers)
    - MNIST model (as a small convolutional one)

