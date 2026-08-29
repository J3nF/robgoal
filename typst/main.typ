//#import "@preview/charged-ieee:0.1.4": ieee
#import "conf.typ": ieee
#import "@preview/dashy-todo:0.1.3": todo
#import "@preview/lovelace:0.3.1": pseudocode-list

#set page(numbering: "1", paper: "a4")
#show: ieee.with(
  title: [ Autaptic inhibition's role in continual learning], // TODO
  abstract: [
    TODO this is some sentence talking about how future AI and continual learning are important and causally related.
    How can one design a performant, memory-efficient continual learning framework that operates within a network's fixed capacity?
    We hypothesize that local capacity constraints limit long-term learning performance in sparse networks, and how distributing learning schemes could circumvent this problem. 
    Inspired by biological networks (TODO: nec. start of sentence?)), we then propose autaptic inhibition for artificial intelligence (ai2) as a mechanism to motivate networks to distribute representations across many neurons.
    Such ai2 systems would outperform their conventionally sparse counterparts by balancing knowledge retainment and information redistribution flexibly.
  ],
  authors: (
    (
      name: "Jens Friedel",
      department: [School of Engineering Mathematics and Technology],
      organization: [University of Bristol],
      location: [Bristol, United Kingdom],
      email: "jens.friedel@bristol.ac.uk"
    ),
    (
      name: "Conor Houghton",
      department: [School of Engineering Mathematics and Technology],
      organization: [University of Bristol],
      location: [Bristol, United Kingdom],
      email: "conor.houghton@bristol.ac.uk"
    ),
    (
      name: "Gabriel Leivas Oliveira",
      department: [School of Engineering Mathematics and Technology],
      organization: [University of Bristol],
      location: [Bristol, United Kingdom],
      email: "gabriel.leivasoliveira@bristol.ac.uk"
    ),
  ),
  index-terms: ("Collective dynamics", "Game theory", "Network formation games", "Network theory (graphs)"), //TODO
  paper-size: "a4",
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

= Introduction <sec:Intro>

= Methods

= Results and Discussion 

= Conclusion

Network dynamics plays an ever-important role in describing the modern world, with connective plasticity and local optimizations enabling the emergence of global order.
Since this framework can describe a vast variety of phenomena, researchers employed numerous approaches --- often multidisciplinary ones --- to analyse these systems.
Specifically, networks of human-like agents have been subject of studies ranging from mostly theoretical analysis to phenomenological social science works @Sobkowicz2020WhitherNO @Flache2017ModelsOS.

When agent $i$ sponsors an edge to $j$, we set $a_(i j) = 1$, and $a_(i j) = 0$ otherwise, so that

$ c^"NFG"_i (a_i) = c^d_i (G) + alpha_a norm(a_i) $ <eq:c_i>

returns the cost an agent incurs, with $norm(dot)$ being the L1 norm.

#figure(
  table(
    columns: (12em, 6em, 6em),
    align: (left, center + horizon, left + horizon),
    inset: (x: 8pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    //fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0  { rgb("#efefef") },
    fill: (x, y) => if y > 0 and calc.rem(y, 4) in (1,2)  { rgb("#efefef") },
    
    table.header[*Variable*][*Symbol*][*Domain*],
    [Number of nodes], [$n$], [$NN$],
    [Node of concern], [$i$], [${1, dots, n}$],
    [Set of $i$'s neighbours], [$K_i$], [$NN^norm(k_i)$],
    [Network graph], [$G$], [${0,1}^(n crossmark n)$],
    [Opinion], [$x_i$], [$RR$],
    [Opinion profile], [$x$], [$RR^n$],
    [Action], [$a_i$], [${0,1}^n$],
    [Action profile], [$a$], [${0,1}^(n times n)$],
    [Distance costs], [$c^d$], [ $ZZ^-$ ],
    [Pairwise distance], [$d_(i j)$], [$NN$],
    [Sponsorship costs], [$c^a$],  [ $]-infinity,0[$ ],
    [Costs per edge], [$alpha_a$], [ $]0,infinity[$ ],
    [Opinion costs], [$c^x$], [ $]-infinity,0[$ ],
    [Opinion range], [$alpha_x$], [$RR^+$],
    [Set of $i$'s neighbours], [$K_i$], [$NN$],
    [Communication rate], [$r_x$], [ $[0, infinity[$ ],
    [Action rate], [$r_a$], [${0,1}$],
    [Convergence threshold], [$epsilon$], [ $]0,alpha_a]$ ],
  ),
  caption: [List of variables used in Co-Formation Games],
  placement: auto,
) <tab:variables>


#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [Schematic CFG algorithm pseudocode. Note this algorithm expects node $i$ to have a cost-reducing action step available, whereas the actual implementation looks through nodes until it found one such step or probed all agents.],

  pseudocode-list[
    + Initialize $a, x$
    + $Delta c = 2 epsilon$
    + *while* $(t<t_max and epsilon < Delta c)$ 
      + *if* $mod(t,r_x) = 0 $ *then*
        + select random node $i$
        + $x_i arrow.l$ get opinion update $x_i^u$
      + *end if*
      + *if* $mod(t,r_a) = 0$ *then*
        + select random node $i$
          + $a_i arrow.l$ get action update $a_i^u$
      + *end if*
      + t += 1
      + $c(t) arrow.l$ get new costs $c_t$
      + $Delta c = c_(t-1) - c(t)$
    + *end while*
  ],
  placement: auto,
) <alg:dynamics>

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 3pt,
    grid.cell(
      image("fig/wetter-terrier.png", width: 100%),
    ),
    grid.cell(
      rowspan: 2,
      inset: (y: 40pt),
      image("fig/wetter-terrier.png", width: 100%),
    ),
    grid.cell(
      image("fig/wetter-terrier.png", width: 100%),
    ),
    grid.cell(
      colspan: 2,
        table(
          columns: (7em, 4em, 4em, 4em),
          inset: (x: 8pt, y: 4pt),
          stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
          
          table.header[*$G$*][*$c^d$*][*$c^a slash alpha_a$*][*$c^x slash alpha_x$*],
          [top chain],[$8$],[$2$],[$1.5$],
          [bottom chain],[$8$],[$2$],[$3$],
          [triangle],[$6$],[$3$],[$2$],
        )
    )
  ), 
  caption: [
    Depending on opinion placement, edge costs, and opinion range, costs motivate nodes to drop or add new edges. 
  ],
  placement: auto,
) <fig:n3>


#figure(
  table(
    columns: (8em, 6em, 6em, 6em),
    inset: (x: 8pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },

    table.header[*$(x_1,x_2,x_3)$*][*$c^x_1$*][*$c^x_2$*][*$c^x_3$*],
    [$(0,0,4)$],[$0$],[$2$],[$4$],
    [$(0,2,4)$],[$2$],[$2$],[$2$],
    [$(0,1,4)$],[$1$],[$2$],[$3$],
  ),
  caption: [
    Total costs of ordered chains depend only on outermost opinions only, while agent utilities may differ significantly.
  ]
) <fig:chains>


#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 3pt,
    grid.cell(
      image("fig/wetter-terrier.png", width: 100%),
    ),
    grid.cell(
      image("fig/wetter-terrier.png", width: 100%),
    ),
    grid.cell(
      image("fig/wetter-terrier.png", width: 100%),
    ),
    grid.cell(
      image("fig/wetter-terrier.png", width: 100%),
    ),
    grid.cell(
      colspan: 2,
        table(
          columns: (7em, 4em, 4em, 4em),
          inset: (x: 8pt, y: 4pt),
          stroke: (x, y) => if y <= 1 { (top: 0.5pt) },

          table.header[*$G$*][*$c^d$*][*$c^a slash alpha_a$*][*$c^x slash alpha_x$*],
          [top star],[$18$],[$3$],[$1.33$],
          [top chain],[$20$],[$3$],[$1.5$],
          [bottom chain],[$20$],[$3$],[$1$],
          [bottom star],[$18$],[$3$],[$2.66$],
        )
    )
  ), 
  caption: [
    Chains can offer optimal costs, outdoing star graphs for certain opinion profiles and parameter choices by profiting from shrinking opinion costs when disagreement happens inside the chain.
  ],
  placement: auto,
) <fig:new_optimal>

Lastly, this subsection lists some elementary conjectures of how chains and stars optimize costs given a certain opinion profile.
As the proofs took place while trying to work out more general results, they are denoted as lemmata.

== Lemma 1 -- Optimal costs of chains

_
For $n$-node chains in a Co-Formation Game, the optimal total opinion costs, given an opinion set $x$ reaching from $0$ to $alpha_x$, are\
1. $c^"opt"_0 = 0$ if all $x_i$ are identical,
2. $c^"opt"_"outside" = 3/2 alpha_x$ if all but one $x_i$ being identical, and
3. $c^"opt"_"inside" = alpha_x$ otherwise.
_

=== Proof

_
Case 1 is trivial.
Case 2 is achieved by placing the outlying opinion at the end of the chain, so that only one node suffers from the disagreement costs, which minimizes opinion costs $c^x$, as can be seen in @eq:c_i.
Case 3 is achieved by ordering opinions in the chain: $0 = x_1 <= x_2 <= ... <= x_n = alpha_x$.
Each node $i$'s opinion costs are
$ c^x_i = (x_i-x_(i-1) + x_(i+1)-x_i) / 2 = (x_(i+1)-x_(i-1)) / 2 , $
so that the total opinion costs become
$ c^x = sum_(i=1)^n c^x_i = sum_(i=1)^n (x_(i+1)-x_(i-1)) / 2 = x_n-x_1 = alpha_x $
$qed$.
_


#figure(
  image("fig/wetter-terrier.png", width: 100%),
  caption: [
     CFGs' number of steps until convergence, when starting with sparse topologies, scale with $n!$ when low edge costs favour dense graphs.
  ]
)<fig:benchmark>

#figure(
 grid(
    columns: (1fr),
    gutter: 3pt,
    grid.cell(
      image("fig/wetter-terrier.png", width: 100%),
    ),
    grid.cell(
      image("fig/wetter-terrier.png", width: 100%),
    ),
  ), 
  caption: [
    CFG runs converge hit the convergence threshold before opinion convergence can take place.
  ],
  placement: auto,
) <fig:simulations>
