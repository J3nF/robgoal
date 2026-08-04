# Mini-project on meta-continual learning

## Line of argument

- We think local information overload is the root of all our troubles (i.e. catastrophic forgetting).
- We therefore wish to distribute information.
- We think this overlaps with keeping neural networks flexible by pushing them out of minima somewhat^[This *somewhat* is doing some heavy lifting to hide my confusion!].
    - As a subset of this statement, we believe forgetting can be useful/necessary for robust learning.
- To achieve this, we allow different neuronal behaviour given similar/same inputs.
- Therefore, we wish to let network parts^[who knows what this means as of now] have self-recurrent, self-inhibiting connections.
    - As the Bio people scream right now, we wish to use *autapses*.
- By "self-inhibiting" neural outputs via autapses, we hope for no single neuron or layer to be solely responsible for critical features.
- I.e.: Many neurons share important information.
- Hypothesis: This diffuses information across the network, which tackles local information overloading. Voila.

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

#### Notes

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

##### Activity-dependent local plasticity

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

### 2026-07-14

#### Pre-meeting notes

- Planning fallacies:
    - I spent more hours than reasonable on trying to download MNIST data for PyTorch.
    - On a different note, I had a fight with Isambard, who, for all practical purposes, has mostly been in a leaky-comatose state.
- Some barebones entropy considerations took place. Basic argument:
    - Assuming neurons have finite """precision""" $p(x|y\neq x) \neq 0$, firing rates 0 and 1 waste information.
    - Our artificial """inhibition""" push neurons away from $p\in \{0,1\}$.

#### Notes

- Jens presented line of argument:
    - Big picture:
        - Sparse representations (1 neuron per concept) either work good or fail hard (catastrophic forgetting) after exhausting local capacity.
        - We hope distributed representations:
            1. fail later, exhausting *local* capacity more slowly;
            1. fail more gracefully.
    - Information theory-inspired:
        - Sparse representations imply firing rates of 0 and 1.
        - Pushing neurons away from 0 and 1 *motivates*^[formally, a network may still reach sparse representations with autapsic inhibition] networks to store information in a distributed manner (yay).
        - Implementation via ***Autapsic Inhibition***™ -- neurons inhibiting their own activation more, the more active they are -- hinders neurons from being perfect sparse perceptrons.
        - => ***Single neurons' job is not to learn by themselves^[thereby making the network depending on them]; learning representations is the networks job.***
    - Note the line of argument is not ad-hoc, but motivated by the autapses idea. There is some overlap, but no perfect isomorphism.


### 2026-07-14

#### Summer School

- Main point: People were broadly interested in our research idea (though not they were not domain experts).
- Got in touch with potentially interesting future contacts (specially in Computational Neuroscience, Information Theory).

#### Current Plan

- Notation:
    - Most importantly, fancy project name: *autapses-inhibited AI (ai2)*.
    - $i =$ neuron of concern;
    - $\bullet^\text{ai2} =$ autapses-related parameter;
    - $y_i =$ $i$'s unadapted output;
    - $b_i =$ $i$'s bias.
- Have run MNIST experiments for
    - CNN
    - Fully connected NN
    - 3 autapsic FCNN implementation.
      $$\begin{align}
        b^\text{ai2}_i(t) &= b_i(t) - b_i(t-1)\\
        y^\text{ai2}_i(t) &= y_i(t) - y_i(t-1)\\
        y^\text{ai2}_i(t) &= y_i(t)\cdot\left(1-y_i(t-1)\right)\\
      \end{align}$$
      with learning taking place between each timestep $t, (t-1)$.
    - (Future control group: Randomly switching subsets of neurons off.)
  -  Learn subsets of all numbers, adding more and more after having learnt old set.
    - Note: "*Having learnt*" is epoch-delimited instead of cost-function delimited.
    - Learning is epoch-delimited because cost function conditions may restrict allowed states too much (i.e., to a sphere in the loss landscape).
        - Checking whether the worry is true is a pot. future TODO.
