# Caspian

## Artificial Ontological Intelligence Through Artificial Reality

> **Can an artificial agent develop useful internal representations of a world through experience, without being explicitly given the human semantic categories used to describe that world?**

---

# 1. Overview

**Caspian** is an experimental artificial agent designed to investigate a particular question about intelligence:

> **Can an AI construct its own useful understanding of an environment through perception, memory, prediction, action, and consequences?**

The project begins with a deliberately small artificial environment rather than a large language model or a complex real-world system.

Caspian will initially inhabit a simple 2D simulated world. It will perceive numerical observations, take actions, observe consequences, maintain internal state, and learn predictive relationships.

The environment may contain entities, resources, obstacles, and rules that are deliberately **not explained to Caspian using human semantic labels**.

For example, the designer may know that an entity increases the agent's energy, but Caspian will not be told:

```text
"This is food."
```

Instead, Caspian experiences:

```text
observation
    ↓
interaction
    ↓
consequence
    ↓
memory
    ↓
prediction
    ↓
future action
```

The purpose is not initially to create AGI, consciousness, or a commercially deployable autonomous system.

The purpose is to investigate the **mechanisms by which an artificial agent can construct internal models of an environment**.

---

# 2. The Central Idea

Most current AI systems are heavily dependent on human-generated representations.

A simplified view is:

$$
\text{Human-generated data}
\rightarrow
\text{Training}
\rightarrow
\text{Model}
\rightarrow
\text{Predictions}
$$

Caspian investigates another possibility:

$$
\text{Environment}
\rightarrow
\text{Experience}
\rightarrow
\text{Internal representation}
\rightarrow
\text{Prediction}
\rightarrow
\text{Action}
\rightarrow
\text{New experience}
$$

The distinction is important.

Humans do not have to explicitly tell Caspian:

* what objects are called,
* which objects belong together,
* which properties matter,
* which relationships are important,
* what causes what.

Instead, the agent should have an opportunity to **discover useful regularities through interaction**.

---

# 3. Research Direction

The broader research direction is provisionally called:

# Artificial Ontological Intelligence (AOI)

The term is used here as a working research label, not as a claim that a formally established scientific field already exists under this exact definition.

### Working definition

> **Artificial Ontological Intelligence is the study and engineering of artificial agents capable of constructing useful internal representations, categories, relationships, and models of an environment through autonomous interaction rather than relying entirely on predefined human semantic categories.**

The central concept is **ontology**.

In this context, ontology means the agent's internal organization of:

* entities,
* properties,
* relationships,
* events,
* causes,
* consequences,
* persistent patterns,
* useful distinctions.

---

# 4. What Caspian Is Trying to Investigate

The central research question is:

$$
\boxed{
\text{Can an artificial agent construct a useful ontology from experience?}
}
$$

More specifically:

> Can an agent discover environmental regularities when the designer provides the world and learning mechanism but does not explicitly provide the semantic interpretation of those regularities?

This produces several sub-questions.

### Q1 — Representation

Can Caspian develop internal representations that correspond to meaningful environmental regularities?

### Q2 — Prediction

Can Caspian predict future environmental states?

### Q3 — Memory

Can previous experience influence future interpretation and behavior?

### Q4 — Adaptation

Can Caspian update its internal model when environmental rules change?

### Q5 — Generalization

Can knowledge learned in one situation transfer to another?

### Q6 — Concept formation

Can Caspian group different experiences according to properties that were not explicitly specified by the designer?

### Q7 — World modeling

Can Caspian construct an internal model of how its environment behaves?

### Q8 — Self-modeling

At later stages, can Caspian construct useful representations of its own state and the consequences of its own actions?

---

# 5. What Caspian Is NOT

Caspian is not initially intended to be:

* AGI
* a chatbot
* an LLM
* a conscious machine
* a digital human
* an autonomous internet agent
* a self-replicating system
* an unrestricted computer-control system
* a replacement for human intelligence

The project must avoid making claims beyond its evidence.

In particular:

$$
\boxed{
\text{Complex behavior} \neq \text{proof of consciousness}
}
$$

A system may possess sophisticated memory, prediction, planning, and self-modeling without providing evidence that it possesses subjective experience.

Consciousness is therefore treated as a separate and unresolved question.

---

# 6. First-Principles Model

The entire system can initially be represented as a dynamical loop.

Let:

$$
E_t
$$

represent the state of the environment at time \(t\).

Caspian receives an observation:

$$
O_t = G(E_t)
$$

where \(G\) is the environment's observation function.

Caspian maintains an internal state:

$$
S_t
$$

which is updated according to:

$$
S_t =
F_\theta(S_{t-1},O_t,A_{t-1})
$$

where:

* \(S_{t-1}\) = previous internal state
* \(O_t\) = current observation
* \(A_{t-1}\) = previous action
* \(F_\theta\) = learned state-transition function
* \(\theta\) = model parameters

Caspian chooses an action:

$$
A_t = \pi_\theta(S_t)
$$

The environment then changes:

$$
E_{t+1}=T(E_t,A_t)
$$

where \(T\) is the environment's transition function.

Caspian then observes:

$$
O_{t+1}=G(E_{t+1})
$$

The complete loop is therefore:

$$
\boxed{
E_t
\rightarrow
O_t
\rightarrow
S_t
\rightarrow
A_t
\rightarrow
E_{t+1}
\rightarrow
O_{t+1}
}
$$

This loop is the foundation of Caspian.

---

# 7. The Artificial World

The first environment will be intentionally simple.

A possible initial world:

```text
┌─────────────────────┐
│ . . . . . . . . . . │
│ . . . . X . . . . . │
│ . . . . . . . . . . │
│ . . . C . . . . . . │
│ . . . . . . . . . . │
│ . . . . . . . . . . │
│ . . . . . . . . . . │
└─────────────────────┘
```

Where:

```text
C = Caspian
X = unknown entity
. = empty space
```

The world may eventually contain:

* entities,
* resources,
* obstacles,
* energy,
* spatial relationships,
* temporal relationships,
* changing rules,
* multiple agents.

However, complexity should increase gradually.

---

# 8. The World Has Its Own Rules

The world is a mathematical system.

For example:

$$
E_{t+1}=T(E_t,A_t)
$$

Suppose Caspian has an energy variable:

$$
e_t
$$

The environment might update it according to:

$$
e_{t+1}
=
e_t+\Delta e
$$

The value of \(\Delta e\) may depend on Caspian's interaction with an entity.

The designer knows the rule.

Caspian does not.

This distinction is fundamental.

---

# 9. Semantic Labels

A major principle of the project is to distinguish between:

### Designer knowledge

The researcher knows:

```text
Entity X increases energy.
```

### Agent experience

Caspian receives something like:

```text
distance = 1
direction = east
energy = 42
action = interact
new energy = 57
```

Caspian is not told:

```text
X = food
```

This allows us to ask whether the agent can infer the underlying relationship.

---

# 10. Caspian's Sensors

The first version should not begin with raw vision.

Structured numerical observations are preferable because they make the experiment interpretable.

For example:

$$
O_t =
[
x_t,
y_t,
d_t,
\theta_t,
e_t,
c_t,
a_{t-1}
]
$$

where:

* \(x_t,y_t\) = position
* \(d_t\) = distance to an entity
* \(\theta_t\) = direction
* \(e_t\) = energy
* \(c_t\) = collision information
* \(a_{t-1}\) = previous action

Later versions may use increasingly rich observations.

Possible progression:

```text
Structured state
      ↓
Local sensory vector
      ↓
Grid representation
      ↓
Images
      ↓
Video
      ↓
Multimodal environment
```

---

# 11. Action Space

Caspian v0.1 can have a small action space:

$$
A=
\{
\text{UP},
\text{DOWN},
\text{LEFT},
\text{RIGHT},
\text{WAIT},
\text{INTERACT}
\}
$$

This is deliberately limited.

The objective is to study learning, not action complexity.

---

# 12. Internal State

A fundamental requirement is that Caspian cannot be purely reactive.

It should have an internal state:

$$
S_t
$$

which depends on previous experience.

A simple recurrent formulation is:

$$
S_t =
f_\theta(S_{t-1},O_t,A_{t-1})
$$

This means:

> What Caspian currently believes/represents depends partly on what it previously experienced.

This creates the possibility of persistent learned representations.

---

# 13. Prediction

A major component of Caspian is a world model.

Given:

$$
S_t,A_t
$$

Caspian attempts to predict:

$$
\hat O_{t+1}
$$

or, at later stages:

$$
\hat E_{t+1}
$$

The model therefore learns:

$$
P(O_{t+1}\mid S_t,A_t)
$$

rather than simply mapping observations directly to actions.

---

# 14. Prediction Error

Let the real next observation be:

$$
O_{t+1}
$$

and Caspian's prediction be:

$$
\hat O_{t+1}
$$

A simple prediction loss can be:

$$
L_{\text{prediction}}
=
D(O_{t+1},\hat O_{t+1})
$$

For numerical variables, mean squared error can be used:

$$
L_{\text{MSE}}
=
\frac{1}{n}
\sum_{i=1}^{n}
(O_i-\hat O_i)^2
$$

The error provides a learning signal.

---

# 15. Neural Networks

Caspian will eventually use neural networks to approximate functions that are difficult to explicitly program.

A simple neural layer:

$$
z=Wx+b
$$

followed by an activation:

$$
h=\sigma(z)
$$

A network might therefore be:

$$
x
\rightarrow
W_1x+b_1
\rightarrow
\sigma
\rightarrow
W_2h+b_2
\rightarrow
output
$$

The first model should remain small enough to understand.

---

# 16. Linear Algebra

Linear algebra provides the basic language for the model.

Observations become vectors:

$$
x\in\mathbb{R}^n
$$

Internal representations become vectors:

$$
h\in\mathbb{R}^d
$$

Parameters become matrices:

$$
W\in\mathbb{R}^{m\times n}
$$

Transformations occur through:

$$
Wx+b
$$

The project therefore relies heavily on:

* vectors,
* matrices,
* dot products,
* matrix multiplication,
* vector spaces,
* projections,
* dimensionality,
* representations.

---

# 17. Probability

Caspian should eventually represent uncertainty.

Instead of:

$$
\text{"X will definitely produce Y"}
$$

the model can represent:

$$
P(Y\mid X)
$$

For example:

$$
P(\text{energy increase}\mid X)=0.91
$$

This is important because environments can be uncertain.

An action may produce different outcomes.

Therefore Caspian needs to reason probabilistically.

---

# 18. Entropy

Uncertainty can be represented using entropy:

$$
H(X)
=
-\sum_xP(x)\log P(x)
$$

If Caspian is highly uncertain about an environmental event, entropy is high.

If it becomes confident about the outcome, entropy decreases.

This gives us another possible research measurement:

$$
\Delta H
=
H_{\text{before}}
-
H_{\text{after}}
$$

How much does an experience reduce Caspian's uncertainty?

---

# 19. Calculus

Learning requires determining how changing parameters changes error.

Given:

$$
L(\theta)
$$

we need:

$$
\nabla_\theta L
$$

The gradient tells us the direction in parameter space in which the loss increases.

Therefore we move in the opposite direction:

$$
\theta_{t+1}
=
\theta_t-\eta\nabla_\theta L
$$

where:

$$
\eta
$$

is the learning rate.

---

# 20. Backpropagation

Backpropagation applies the chain rule to calculate how errors propagate through the network.

If:

$$
L=f(g(h(x)))
$$

then:

$$
\frac{dL}{dx}
=
\frac{dL}{df}
\frac{df}{dg}
\frac{dg}{dh}
\frac{dh}{dx}
$$

For Caspian, this allows the system to determine which parameters contributed to prediction errors.

---

# 21. Optimization

The model attempts to minimize an objective function.

A simplified objective:

$$
J(\theta)
=
L_{\text{prediction}}
$$

Later:

$$
J(\theta)
=
L_{\text{prediction}}
+
\lambda L_{\text{representation}}
-
\beta R
$$

where:

* \(L_{\text{prediction}}\) = prediction error
* \(L_{\text{representation}}\) = optional representation objective
* \(R\) = reward
* \(\lambda,\beta\) = weighting coefficients

The exact objective should evolve experimentally rather than being assumed in advance.

---

# 22. Reinforcement Learning

Eventually Caspian needs to learn from consequences.

Let:

$$
R_t
$$

be the reward received at time \(t\).

The objective can be:

$$
\max_\pi
\mathbb{E}
\left[
\sum_{t=0}^{T}
\gamma^tR_t
\right]
$$

where:

$$
0\leq\gamma\leq1
$$

is the discount factor.

This represents the tradeoff between immediate and future consequences.

However, reinforcement learning should not be introduced prematurely.

First establish that Caspian can learn and predict.

---

# 23. The World Model and the Policy

Caspian eventually has at least two important functions.

### World model

$$
M_\theta:
(S_t,A_t)
\rightarrow
\hat O_{t+1}
$$

It answers:

> What do I expect to happen?

### Policy

$$
\pi_\theta:
S_t
\rightarrow
A_t
$$

It answers:

> What should I do?

These should be conceptually separated.

A system can predict well without acting well.

A system can act well without having an interpretable world model.

The project should investigate both.

---

# 24. Memory

A major research component is memory.

There are at least three possible forms:

### Short-term state

What is happening now?

$$
S_t
$$

### Episodic memory

What happened previously?

$$
M_t=\{e_1,e_2,\ldots,e_n\}
$$

### Learned parameters

What regularities have been incorporated into the model?

$$
\theta
$$

These represent different kinds of persistence.

---

# 25. The Concept of a "Mindset"

The term **mindset** is used informally in this project to describe persistent internal structures that influence interpretation and behavior.

A possible operational definition is:

> A persistent internal representation or learned disposition that changes how an agent interprets observations, predicts outcomes, and chooses actions.

For example:

Initially:

$$
P(Y\mid X)=0.5
$$

After repeated experience:

$$
P(Y\mid X)=0.95
$$

The agent now behaves differently around \(X\).

If this representation persists across time and influences future decisions, it resembles a primitive computational "belief" or "disposition."

This does **not** establish consciousness.

---

# 26. Ontology Formation

The long-term research goal is to investigate whether Caspian can form categories that are not explicitly supplied by the designer.

Suppose the environment contains:

$$
X_1,X_2,X_3,X_4,X_5
$$

The designer does not tell Caspian that:

$$
X_1,X_3,X_5
$$

belong together.

However, Caspian's internal representations may eventually satisfy:

$$
d(h_{X_1},h_{X_3}) \ll
d(h_{X_1},h_{X_2})
$$

where \(d\) is a distance function in representation space.

This could indicate that the model has developed a representation in which certain entities are grouped together.

But we must determine **why**.

It is not enough to observe clustering.

---

# 27. Emergent Concepts

A concept should not be declared "discovered" merely because a visualization looks interesting.

A stronger test requires:

1. A measurable environmental regularity exists.
2. The regularity was not directly labelled to the agent.
3. The agent develops an internal representation correlated with it.
4. That representation influences prediction or behavior.
5. The behavior generalizes to novel examples.
6. The result can be reproduced.

The stronger the evidence across these dimensions, the stronger the claim.

---

# 28. Causal Discovery

Later experiments can investigate whether Caspian learns causal relationships.

Correlation:

$$
X\rightarrow Y
$$

does not necessarily imply:

$$
X\text{ causes }Y
$$

To investigate causality, Caspian needs interventions.

For example:

$$
do(X=x)
$$

and observe:

$$
P(Y\mid do(X=x))
$$

This creates a more powerful learning loop:

```text
Observe
   ↓
Hypothesize
   ↓
Intervene
   ↓
Observe consequence
   ↓
Update model
```

This is potentially much closer to scientific reasoning than passive pattern recognition.

---

# 29. Experimental Progression

The project should grow incrementally.

## Experiment 001 — Prediction

Can Caspian predict the next environmental state?

---

## Experiment 002 — Consequences

Can Caspian learn that an unknown interaction changes a measurable variable?

---

## Experiment 003 — Generalization

Can Caspian apply a learned relationship to an unseen location?

---

## Experiment 004 — Hidden Categories

Can Caspian distinguish entities based on an underlying property without receiving labels?

---

## Experiment 005 — Rule Changes

Change the environment.

Example:

Previously:

$$
X\rightarrow +10\text{ energy}
$$

Later:

$$
X\rightarrow -10\text{ energy}
$$

Measure how quickly Caspian updates.

---

## Experiment 006 — Multiple Worlds

Train in:

$$
W_1
$$

Test in:

$$
W_2
$$

Determine whether Caspian has learned general principles or merely memorized one environment.

---

## Experiment 007 — Memory

Compare:

```text
No memory
```

against:

```text
Short-term memory
```

against:

```text
Long-term memory
```

Measure the effect.

---

## Experiment 008 — Novel Entities

Introduce entities Caspian has never encountered.

Test whether it can place them into useful internal categories.

---

## Experiment 009 — World Model

Ask Caspian to predict future states several steps ahead:

$$
\hat O_{t+1},
\hat O_{t+2},
\hat O_{t+3},...
$$

---

## Experiment 010 — Counterfactual Reasoning

Ask:

> What would have happened if Caspian had taken action \(A'\) instead of \(A\)?

This requires a model capable of simulating alternative trajectories.

---

# 30. Metrics

Every experiment must have measurable outcomes.

### Survival time

$$
T_{\text{survival}}
$$

How long does the agent remain viable?

### Goal completion

$$
G=
\frac{\text{goals completed}}
{\text{goals attempted}}
$$

### Prediction accuracy

$$
Accuracy=
\frac{\text{correct predictions}}
{\text{total predictions}}
$$

### Prediction error

$$
MSE=
\frac{1}{n}
\sum_i(y_i-\hat y_i)^2
$$

### Sample efficiency

How many experiences are required to reach a given performance level?

$$
SE=
\frac{\text{performance}}
{\text{experience}}
$$

### Adaptation speed

How many interactions are required after a rule change before performance recovers?

$$
T_{\text{adapt}}
$$

### Generalization

Performance on unseen environments:

$$
G_{\text{new}}
$$

### Representation similarity

Measure whether experiences with similar functional properties generate similar internal representations.

### Concept discovery

Measure whether latent representations correspond to environmental regularities that were not explicitly labelled.

### Counterfactual accuracy

Compare predicted alternative outcomes with actual outcomes obtained under controlled interventions.

---

# 31. Intelligence Efficiency

An important long-term metric is not merely:

> How big is Caspian?

but:

> How much useful behavior can Caspian produce per unit of model capacity and experience?

For example:

$$
Efficiency=
\frac{\text{task performance}}
{\text{parameters}}
$$

or:

$$
Efficiency=
\frac{\text{performance}}
{\text{experience}}
$$

This allows comparisons between:

```text
1M parameters
10M parameters
100M parameters
1B parameters
```

rather than assuming that larger automatically means better.

---

# 32. Model Size

Approximate neural-network weight storage is:

$$
Size \approx
\frac{N\times b}{8}
$$

where:

* \(N\) = number of parameters
* \(b\) = bits per parameter

For FP16:

$$
Size\approx2N\text{ bytes}
$$

Therefore:

| Parameters | Approx. FP16 weights |
| ---------: | -------------------: |
|       100K |               0.2 MB |
|         1M |                 2 MB |
|        10M |                20 MB |
|       100M |               200 MB |
|         1B |                 2 GB |
|         7B |                14 GB |

These are approximate weight-storage figures and exclude activations, optimizer states, memory, checkpoints, metadata, and runtime overhead.

Caspian should begin in the small range.

---

# 33. Model Size vs Experience

One of the project's interesting questions is whether intelligence depends primarily on parameter count.

We can compare:

$$
\text{Model size}
$$

against:

$$
\text{Experience}
$$

and:

$$
\text{Environment complexity}
$$

For example:

| Model | Parameters | Experience | Performance |
| ----- | ---------: | ---------: | ----------: |
| A     |         1M |   1M steps |           ? |
| B     |        10M |   1M steps |           ? |
| C     |        10M |  10M steps |           ? |
| D     |       100M |   1M steps |           ? |

This could reveal whether improvements arise mainly from:

* scale,
* experience,
* memory,
* architecture,
* environment design,
* learning algorithm,
* or interactions among them.

---

# 34. Why Not Start With an LLM?

Caspian does not initially need an LLM.

A language model primarily learns:

$$
P(x_t\mid x_{<t})
$$

Caspian initially needs to learn:

$$
P(O_{t+1}\mid S_t,A_t)
$$

The two problems are related but not identical.

An LLM can later become a component of Caspian if language becomes useful.

But beginning without language makes it easier to study whether useful internal representations can arise from environmental interaction alone.

---

# 35. Potential Architecture

A mature Caspian architecture could eventually resemble:

```text
                    ARTIFICIAL WORLD
                           │
                           ▼
                     PERCEPTION
                           │
                           ▼
                    REPRESENTATION
                           │
                           ▼
                      MEMORY
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
            WORLD MODEL           SELF MODEL
                 │                   │
                 └─────────┬─────────┘
                           ▼
                        POLICY
                           │
                           ▼
                         ACTION
                           │
                           ▼
                    WORLD CHANGES
                           │
                           └───────────┐
                                       ▼
                                  NEW EXPERIENCE
```

This architecture is conceptual and should evolve based on experimental evidence.

---

# 36. Self-Model

A later research direction is the possibility of representing the agent itself.

The environment contains:

$$
E_t
$$

but Caspian also has an internal state:

$$
S_t
$$

A self-model could attempt to predict:

$$
\hat S_{t+1}
=
M_{\text{self}}(S_t,A_t)
$$

This raises questions such as:

* Can Caspian predict its own future state?
* Can it distinguish itself from other entities?
* Can it model its own capabilities?
* Can it recognize changes in its own internal state?
* Can it predict the consequences of its own actions?

Again:

$$
\text{self-model} \neq \text{proof of consciousness}
$$

---

# 37. Consciousness

Consciousness is **not** an initial target.

A sophisticated self-model could eventually raise philosophical and scientific questions about machine consciousness, but the project must not assume:

$$
\text{intelligence}
\Rightarrow
\text{consciousness}
$$

or:

$$
\text{self-model}
\Rightarrow
\text{consciousness}
$$

Those relationships remain unresolved.

If later experiments produce behavior that appears relevant to theories of consciousness, that becomes a separate research question requiring appropriate scientific and philosophical analysis.

---

# 38. Safety Architecture

Caspian should initially operate inside a strict sandbox.

### Caspian v0.1 should have:

* no unrestricted internet access,
* no unrestricted filesystem access,
* no external device control,
* no autonomous deployment,
* no self-replication,
* no unrestricted self-modification,
* bounded compute,
* bounded memory,
* deterministic experiment configurations where possible,
* complete experiment logging,
* checkpoints,
* explicit shutdown capability.

The environment should be disposable.

If something goes wrong:

$$
\boxed{\text{Terminate simulation}}
$$

No external consequence should occur.

---

# 39. Reproducibility

Every experiment should record:

```text
Experiment ID
Date
Environment version
Random seed
Model architecture
Parameter count
Hyperparameters
Training steps
Dataset/experience configuration
Reward function
Loss function
Evaluation metrics
Checkpoint
Results
Interpretation
Limitations
```

A result that cannot be reproduced should be treated cautiously.

---

# 40. Scientific Discipline

The project must distinguish between:

### Observation

> Caspian's hidden representations clustered these entities.

### Interpretation

> The clustering may represent a functional category.

### Hypothesis

> Caspian has discovered an environmental concept.

### Evidence

> The representation predicts behavior and generalizes to unseen examples.

These are not equivalent.

The project should avoid anthropomorphism.

---

# 41. What Would Count as a Significant Result?

A strong result would not be:

> "Caspian behaved intelligently."

That is too vague.

A stronger result would look like:

> Caspian was trained without semantic labels for a particular environmental property. After interaction, its internal representations encoded that property, the representation predicted behavior, and the learned representation generalized to previously unseen entities and environments.

An even stronger result would demonstrate:

1. No explicit semantic label.
2. Repeated emergence across random seeds.
3. Generalization.
4. Robustness to environmental changes.
5. Predictive usefulness.
6. Causal relevance to behavior.
7. Reproducibility.
8. Comparison against appropriate baselines.

---

# 42. What Would Count as Failure?

Failure is useful.

Examples:

* Caspian memorizes positions.
* Caspian exploits an unintended reward loophole.
* Caspian learns superficial correlations.
* Representations disappear when the environment changes slightly.
* Performance requires enormous amounts of experience.
* Apparent concepts do not generalize.
* The model simply memorizes the training environment.
* Larger models provide no meaningful improvement.
* Removing memory has no effect.
* The world model does not improve prediction.

These results are valuable because they tell us what the proposed mechanism does **not** accomplish.

---

# 43. Baselines

Every major experiment should have comparison systems.

For example:

### Random agent

$$
\pi(a)=\text{random}
$$

### Reactive agent

Uses only the current observation:

$$
A_t=\pi(O_t)
$$

### Memoryless neural agent

No persistent internal state.

### Memory-based agent

$$
A_t=\pi(S_t)
$$

### World-model agent

Predicts future states.

Comparing these systems allows us to determine which component actually matters.

---

# 44. The Core Experimental Principle

Caspian should gradually receive more capability.

For example:

```text
Baseline
   ↓
Prediction
   ↓
Memory
   ↓
Action
   ↓
Learning from consequences
   ↓
World model
   ↓
Generalization
   ↓
Concept formation
   ↓
Causal reasoning
   ↓
Self-model
```

At every step:

> **What changed, and what capability did that change produce?**

---

# 45. Long-Term Vision

The long-term version of Caspian is not simply a larger neural network.

The vision is an artificial agent capable of:

$$
\boxed{
\text{Perceive}
\rightarrow
\text{Remember}
\rightarrow
\text{Predict}
\rightarrow
\text{Act}
\rightarrow
\text{Observe consequences}
\rightarrow
\text{Update}
}
$$

repeated continuously.

The ultimate research question becomes:

> **Can sufficiently capable artificial agents construct increasingly rich internal models of worlds through experience?**

If yes, the implications extend beyond Caspian.

Potential applications could include:

* scientific discovery,
* robotics,
* autonomous exploration,
* simulation,
* engineering,
* environmental modeling,
* adaptive AI,
* scientific experimentation,
* complex-system analysis,
* automated hypothesis generation.

These applications are hypotheses about future usefulness, not established outcomes.

---

# 46. Artificial Reality

A future Caspian environment does not need to resemble Earth.

It could contain:

* unfamiliar physics,
* unfamiliar entities,
* unusual geometry,
* artificial sensory modalities,
* different temporal rules,
* changing laws,
* multiple interacting agents.

For example, the environment might contain a "sun" that is blue.

There is no reason Caspian should be told:

> "This is a sun."

The relevant question is:

> What does the entity do?

Does it provide energy?

Does it alter temperature?

Does it influence other entities?

Does it move?

Does its presence predict environmental changes?

The agent's representation should arise from interaction.

---

# 47. The Blue Sun Principle

A guiding thought experiment for this project is:

> **If we construct a world whose entities and rules do not correspond neatly to human categories, can an artificial agent still construct a useful understanding of that world?**

This prevents the project from becoming merely:

> "Teach an AI our world."

Instead, the experiment becomes:

> **"Give an AI a world and investigate what structure it can discover."**

---

# 48. The Most Important Distinction

Caspian is not primarily attempting to answer:

> How do we make AI bigger?

It is attempting to investigate:

> **How does an artificial system acquire an internal model of reality?**

The difference is:

$$
\boxed{
\text{Scale}
\neq
\text{Understanding}
}
$$

Scale may contribute to capability.

But this project investigates the relationship between:

$$
\text{Experience}
+
\text{Memory}
+
\text{Prediction}
+
\text{Action}
+
\text{Environment}
$$

and:

$$
\text{Internal representation}
$$

---

# 49. Proposed Research Equation

A useful conceptual formulation is:

$$
\boxed{
\text{Experience}
\rightarrow
\text{Representation}
\rightarrow
\text{Prediction}
\rightarrow
\text{Action}
\rightarrow
\text{New Experience}
}
$$

This creates a closed learning loop:

$$
\boxed{
O_t
\rightarrow
S_t
\rightarrow
A_t
\rightarrow
E_{t+1}
\rightarrow
O_{t+1}
\rightarrow
S_{t+1}
}
$$

The system's internal model changes as experience accumulates.

---

# 50. Development Philosophy

Caspian should be developed according to five principles.

## 1. First principles

Understand the mathematics and mechanisms rather than blindly assembling libraries.

## 2. Small before large

Prove a capability with a tiny system before increasing complexity.

## 3. Measure before interpreting

Collect evidence before making claims.

## 4. Separate observation from speculation

Do not confuse interesting behavior with proof of cognition or consciousness.

## 5. Reproduce before scaling

A phenomenon should be reproducible before major computational resources are invested.

---

# 51. Initial Technology Stack

The initial implementation can use:

```text
Python
NumPy
PyTorch
Matplotlib
Git
```

Potential later additions:

```text
Gymnasium or custom environment framework
TensorBoard
Jupyter
CUDA
Weights & Biases
```

However, external frameworks should be introduced only when they improve the experiment.

---

# 52. Initial Repository

```text
caspian/
│
├── README.md
│
├── environment/
│   ├── world.py
│   ├── entities.py
│   ├── physics.py
│   └── rules.py
│
├── agent/
│   ├── caspian.py
│   ├── memory.py
│   ├── encoder.py
│   └── policy.py
│
├── models/
│   ├── world_model.py
│   └── predictor.py
│
├── learning/
│   ├── loss.py
│   ├── optimizer.py
│   └── train.py
│
├── experiments/
│   ├── experiment_001.py
│   ├── experiment_002.py
│   └── ...
│
├── evaluation/
│   ├── metrics.py
│   ├── evaluation.py
│   └── baselines.py
│
├── analysis/
│   ├── representations.py
│   └── visualizations.py
│
├── checkpoints/
│
├── logs/
│
└── docs/
    ├── experiments/
    └── theory/
```

---

# 53. Versioning

The project should use explicit versions.

```text
Caspian v0.1
```

means the initial prototype.

Future versions:

```text
v0.2
v0.3
v0.4
...
v1.0
```

A version should correspond to a meaningful architectural or experimental change.

---

# 54. Caspian v0.1 Specification

### Environment

2D discrete grid.

### Observation

Structured numerical state.

### Actions

```text
UP
DOWN
LEFT
RIGHT
WAIT
INTERACT
```

### Internal state

Small learned recurrent or stateful representation.

### Objective

Predict environmental consequences and eventually learn useful behavior.

### Model

Small neural network.

### Environment complexity

Low.

### External access

None.

### Initial research target

Learn an environmental relationship without an explicit semantic label.

---

# 55. First Milestone

The first milestone is:

$$
\boxed{
\text{Caspian learns an unlabeled environmental relationship.}
}
$$

Example:

The environment contains entity \(X\).

The researcher knows:

$$
X\rightarrow +10\text{ energy}
$$

but Caspian receives no semantic label saying "food."

After experience, Caspian should predict:

$$
P(\Delta E>0\mid X,\text{interaction})
$$

with significantly greater accuracy than an appropriate baseline.

Then we test whether this knowledge:

* persists,
* affects action,
* generalizes,
* survives location changes,
* survives irrelevant visual changes,
* transfers to novel entities with the same underlying property.

Only then should we move toward ontology formation.

---

# 56. The Long-Term Scientific Question

Everything in this repository ultimately points toward one question:

> **Can an artificial agent develop an internally useful understanding of a world through its own interaction with that world, rather than having that understanding fully specified by its designers?**

If the answer is no, we learn where and why the approach fails.

If the answer is yes, we investigate:

* how,
* under what conditions,
* at what scale,
* with what architecture,
* with what amount of experience,
* and whether the learned representations generalize.

That is the purpose of Caspian.

---

# 57. Final Principle

Caspian should never be judged by how impressive it sounds.

It should be judged by what the experiments demonstrate.

The project begins with a tiny world and a tiny model.

From there:

$$
\boxed{
\text{World}
\rightarrow
\text{Experience}
\rightarrow
\text{Learning}
\rightarrow
\text{Representation}
\rightarrow
\text{Prediction}
\rightarrow
\text{Action}
\rightarrow
\text{Discovery}
}
$$

The objective is not to assume that intelligence will emerge.

The objective is to **build the conditions under which we can test whether it does.**

---

## Project Status

**Current stage:** Concept / Experimental Design

**Current model:** Caspian v0.1

**Current environment:** 2D artificial world

**Current research direction:** Artificial Ontological Intelligence

**Immediate objective:** Build the smallest scientifically useful experiment.

**First question:**

> **Can Caspian learn a meaningful environmental relationship from experience without being given its human semantic label?**

**Next step:**

Build the environment before building the model.

