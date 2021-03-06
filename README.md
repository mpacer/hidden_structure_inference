
Note: As of January 15, 2016 this is still a work in progress, but the basic functionality is present.

# The general approach

This repository contains code to perform hidden structure inference over a set of (possibly cyclic) directed graphical models using data expressed in continuous-time. 

This is accomplished by using a monte-carlo approach to given a distribution on the parameter space to parameterize the models. Then, given a parameterized graph, it uses monte-carlo integration over the joint distribution of observed and hidden events to define a well-formed likelihood for the observed data. This is sufficient to define the marginal likelihood of the observed data given each graph. Then, given a prior distribution over the set of considered graphs, this is sufficient to generate an unnormalized posterior distribution over the set of graphs (given the unnormalized posteriors, normalization is straightforward: divide the unnormalized posteriors by their sum).

## The specific role of this repository

The posteriors generated by this system can then be used to calculate the posterior probability that any particular hidden edge in the set of graphs is or is not present. In particular, this repository represents an attempt to model the inferences detailed in Experiment 1 of [Lagnado & Sloman (2006)][article], in which the results are expressed in terms of the marginal beliefs about the existence of individual edges under different timing conditions. 

Because of this, much of the code in the `likelihood_calculations_shared_parameters.py` will not be easily generalizable to other tasks. In particular, one can only share parameters between graphs if the underlying semantics of the graphs allows that (e.g., the use of superpositioned point processes (as in this case) allows this). Nonetheless, much of the infrastructure will be able to be used in other tasks that involve enumerating and computing functions over sets of graphs.

## Graph Enumeration: cbnx+

This uses an enriched version of the [Causal Bayesian NetworkX][scipy] ([`cbnx`][cbnx]) library to programmatically enumerate graphical models equipped with rich semantics for operating on graphs based on the semantic or structural properties of the graph's edges and nodes. The basic function used to accomplish this enumeration is `generate_graphs()`, which returns a generator that (for now) needs to be turned into a list to be able to interface with the rest of the code.

The basic graph operations used to specify the set of graphs to be enumerated are `filters` and `conditions`. Given a set of nodes, `cbnx` starts with a complete graph between these nodes, and `filters` take that graph and return a graph with a reduced set of edges according to any rules that will prohibit the existence of an edge in graphs that will be enumerated. 

`Conditions`, on the other hand, apply to graphs as a whole to check whether that graph meets or fails to meet a particular condition. Rather than returning a graph, `conditions` return functions that return `Boolean` values that indicate whether a particular enumerated graph did or did not meet that condition.

`registry.py` allows new `filters` and `conditions` to be registered by end-users wishing to adapt this framework to other problems. This is done by importing the `Registry` class and using the `@X.register` decorator before a function, where `X` is replaced by the appropriate class name.

The `filters.py` and `conditions.py` files contain the `filters` and `conditions` needed by the particular application considered in this repository. To register a new `filter`, you can use `@Filter.register`; to register a new `condition` you can use `@Condition.register`.

The node names, particular `filters`, and `conditions` used can be found in `config.py` as the dictionary `generator_dictionary`. 

Additionally, it can be faster and more transparent to explicitly list those edges that are to be enumerated over if those are known in advance. These can be included as the optional argument `query_edge_set` that is passed into the `generate_graphs()` function. This feature should be thought of as complementary to `filters` and `conditions`.

### Graph semantics

Node and Edge properties are currently assigned by applying objects from the `Node_Semantics_Rule` and `Edge_Semantics_Rule` classes found in `node_semantics.py`. Currently, the semantics are defined relative to the naming conventions of the nodes in question. The particular semantics used in this repository can be found in `config.py`, as the dictionaries `edge_semantics` and `node_semantics`.

## Graph classes 

The continuous-time processes are generated by objects in the `graph_local_classes.py` module. 

The `GraphStructure` class encodes graphical structures generated by networkx (or other compatible means) in a form that has a more convenient API. 

The `GraphParams` class parameterizes the edges of the graph encoded in a `GraphStructure` object according to parameter distributions defined by the model. In this case, the `GraphParams` defines parameters for building and sampling a [Finitary Poisson Process][fpps] (a non-homogeneous Poisson Process with a rate function that integrates to a finite value) on each edge of the graph. In particular, this work relies on exponentially decaying rate functions with a maximum rate (𝜓) and a decay rate (r), that are defined relative to a scale-free base rate parameter (λ). Multiple edges leading into a node are considered to be superposed on each other (i.e., multiple parent nodes can induce events independently of each other). 

The `InnerGraphSimulation` class generates events on these parameterized edges. In this particular application, only the first events generated are relevant to the task, which simplifies the generation of events. 

Caveat: A `InnerNodeSimulation` object is capable of sampling from the more general Finitary Poisson Process, but this needs to be pursued carefully. If no cut-off is given to either the number of generated events or the value the events can take on, this can result in a loop of event generation that will run forever for the purposes of computation. This is true, despite it being the case that at any particular point in the progress of the algorithm, only a finite number of events can be expected from the generated processes. Further details on this are forthcoming.

Reference:
Lagnado, D. & Sloman S. (2006) Time as a guide to cause. *Journal of Experimental Psychology: Learning, Memory & Cognition*, 32, 451-460.

[article]: http://www.ucl.ac.uk/lagnado-lab/publications/lagnado/Lagnado_time_%20as_guide_to_cause.pdf
[scipy]: http://conference.scipy.org/proceedings/scipy2015/mike_pacer.html
[cbnx]: https://github.com/michaelpacer/Causal-Bayesian-NetworkX
[fpps]: https://www.youtube.com/watch?v=69mdtQsrBcI&feature=youtu.be&t=42m52s
