# Lab book

Record planned, fired simulations, with parameter choices and underlying reasoning.

No record necessary if runs fail due to programming or technical errors.

## 2026-08-w1

### Week's plan

#### General assumptions

- Networks are deep.
- Networks first learn a subset of numbers.
    - Learning stops after a number of epochs.
    - We choose to stop learning after a number of epochs, because a loss-dependent stopping condition constrains end states.
        - Loss-dependent stopping constrains end states to spheres within the loss landscape.
        - We expect these spheres to limit our system convergence states too much to allow our systems to converge to meaningfully different states.
- After learning that subset, they are to learn further numbers.
- "Further numbers" may or may not be all held-back numbers.
- For autapsic inhibition (*ai*), I normalise neural outputs to $x_i \in [-1,1]\,$.
- For autapsic inhibition, NNs are fully connected.

Consider these network types:

- Control 1: CNN
- Control 2: Fully connected DNN
- AI2.0: Normal learning step, but neuron $i$'s output shrinks relative to last output:
    - $$x^\text{AI}_i(t) = x_i(t)\cdot (1-x_i(t-1))$$ 
    - TODO: Learning must pay respect to this scaling!
    - TODO: Think whether logistic mappings can emerge from such a rule?
    - Note how this allows for "pulsing" dynamics:
        - A neural pathway may osciallate between activation patterns.
        - Some neural pathways may oscillate activation patterns between each other.
- AI2.1: Normal learning step, but set neuron's bias to
    - $$b^\text{AI}_i(t) = -x_i(t-1)$$
    - (Alternative to keep some learning info: $$b^\text{AI}_i(t) = \langle -x_i(t-1), b^text{backprop}_i(t) \rangle$$
    - TODO: Think whether biases are constrained to be within $\pm 1$.
- AI2.2: Normal learning step, but subtract old activation from new activation:
    - $$x^\text{AI}_i(t) = x_i(t) - x_i(t-1)$$.
