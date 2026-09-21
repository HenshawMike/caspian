# CASPIAN — Agent-Driven Development & Experimental Research Plan

**Project:** Caspian  
**Research direction:** Artificial Ontological Intelligence *(working research label)*  
**Development method:** Agent-Driven Development  
**Initial environment:** Small isolated artificial 2D world  
**Initial agent:** Small neural agent; no LLM required initially  
**Primary objective:** Investigate whether an artificial agent can develop useful internal representations of an environment from perception, action, feedback, memory, and prediction without being explicitly given human semantic categories.

---

## 1. Important Document Architecture

This project uses **two different documentation layers**.

### Layer A — This Markdown file

This is the **living project specification and Agent-Driven Development plan**.

It defines:

- what Caspian is
- what we are trying to discover
- the research questions
- mathematical foundations
- project architecture
- development phases
- agent-development rules
- experiment requirements
- acceptance criteria
- documentation requirements
- safety and reproducibility requirements
- what must happen before moving to the next phase

This file is **not the experiment report**.

It should be updated when the project specification, architecture, phase requirements, or research plan changes.

### Layer B — DOCX Experiment Reports

After **every meaningful experiment**, a separate `.docx` report must be generated.

The DOCX is the permanent scientific record of what actually happened.

Each experiment report must document:

- what was attempted
- why it was attempted
- exact code/configuration used
- environment
- model
- mathematics
- parameters
- training procedure
- random seeds
- commands executed
- results
- graphs/tables
- failures
- unexpected behavior
- observations
- errors
- debugging
- changes made during the experiment
- interpretation
- alternative explanations
- limitations
- conclusions
- whether the hypothesis survived
- whether the next experiment is justified

**Rule:**

> The Markdown plan describes what we intend to do.  
> The DOCX experiment report records what actually happened.

Never rewrite an experiment report to make the experiment look cleaner than it was.

---

# 2. Research Vision

Caspian is an attempt to investigate a question deeper than conventional language-model scaling.

Most AI systems receive representations that humans have already designed:

- objects
- labels
- classes
- words
- categories
- rewards
- tasks
- environments
- goals.

Caspian begins with a much smaller assumption.

We want to create an artificial world in which the agent can perceive and interact with things without being directly told what those things *mean*.

The long-term question is:

> **Can an artificial agent construct useful internal representations of an environment when human semantic categories are not explicitly provided?**

A more ambitious version is:

> **Can an artificial intelligence develop useful categories or ontology from perception-action feedback?**

This project does **not** assume that success would imply consciousness.

Intelligence, representation, adaptation, agency, self-modeling, and subjective experience are separate questions.

---

# 3. North Star

> **Do not begin by trying to make Caspian intelligent. Begin by creating a world in which intelligence can be measured.**

The first milestone should therefore be deliberately small.

Caspian should demonstrate that it can learn a meaningful environmental relationship from interaction without being given the human semantic label for that relationship.

Everything beyond that must be earned through experimental evidence.

---

# 4. Core Research Question

## Primary question

> Can a small artificial agent learn a useful representation of an environmental regularity through perception, action, feedback, memory, and prediction when the human semantic interpretation of that regularity is not directly supplied?

## Phase 1 version

> Can Caspian learn that an unknown interaction reliably changes a measurable state variable?

For example:

An unknown entity `X` exists in the environment.

Caspian can:

- see `X`
- move toward or away from `X`
- interact with `X`
- observe changes in its internal state.

The researcher may know that interacting with `X` increases energy.

Caspian should **not** receive:

```text
food
beneficial
energy_source
reward_object
relative_direction
shape_code
collision
interaction_result
energy
previous_action
```

The difference is fundamental.

We want the agent to learn:

$$ \text{observation} \rightarrow \text{interaction} \rightarrow \text{state transition} $$

rather than:

$$ \text{human label} \rightarrow \text{known meaning} $$

---

# 10. Semantic Firewall

The project must maintain a semantic firewall.

The researcher can know the meaning of environmental variables.

The agent should receive only the information specified by the experiment.

Before every experiment, perform a semantic-leakage audit.

Ask:

- Did we explicitly name the object's function?
- Did we encode its usefulness?
- Did we encode its danger?
- Did we encode the expected outcome?
- Did we provide a reward that directly reveals the intended category?
- Did the observation contain information that makes the answer trivial?
- Did the training data contain labels that should not exist?
- Did the architecture accidentally expose hidden environment state?

Any leakage must be documented.

---

# 11. Mathematical Foundations

The project should be built from mathematical primitives rather than unexplained frameworks.

## 11.1 Linear Algebra

Neural computation fundamentally involves vectors and matrices.

A basic transformation:

$$ z=Wx+b $$

where:

- \(x\) = input vector
- \(W\) = weight matrix
- \(b\) = bias
- \(z\) = transformed representation.

An activation function produces:

$$ h=\sigma(z) $$

This creates a learned representation.

## 11.2 Probability

If Caspian predicts actions or future states, predictions can be represented probabilistically.

For example:

$$ P(O_{t+1}|S_t,A_t) $$

means:

the probability of the next observation given the current internal state and action.

For actions:

$$ P(A_t|S_t) $$

represents the policy.

## 11.3 Calculus

Learning requires measuring how the loss changes as parameters change.

The gradient:

$$ \nabla_\theta L $$

tells us the direction in parameter space associated with increasing loss.

Gradient descent moves approximately in the opposite direction:

$$ \theta'=\theta-\eta\nabla_\theta L $$

## 11.4 Optimization

Training is an optimization problem.

We seek:

$$ \theta^*=\arg\min_\theta L(\theta) $$

The system searches for parameters that produce better predictions or behavior.

## 11.5 Information Theory

Prediction error can also be understood as information mismatch.

For probabilistic predictions, cross-entropy is:

$$ H(p,q) = -\sum_xp(x)\log q(x) $$

This becomes important later when Caspian predicts discrete outcomes.

## 11.6 Statistics

We must distinguish:

Caspian learned something

from:

Caspian happened to perform well once.

Therefore experiments require:

- multiple seeds
- held-out evaluation
- repeated trials
- confidence intervals where appropriate
- baseline comparisons
- variance reporting.

---

# 12. Representation

The most important scientific object may not be the action.

It may be the internal representation.

Suppose:

$$ S_t=f_\theta(O_{\leq t},A_{<t}) $$

Then \(S_t\) represents information the agent has accumulated about its situation.

We want to investigate whether similar environmental conditions create useful structure in \(S_t\).

Possible analysis:

- hidden-state clustering
- representation similarity
- linear probes
- nearest-neighbor structure
- dimensionality reduction
- predictive information
- temporal consistency.

Important:

A cluster is not automatically a concept.

A representation can correlate with an environmental variable without proving that the agent has developed a human-like concept of that variable.

---

# 13. Baselines

Every important experiment needs baselines.

Initial baselines:

**Random baseline**

Chooses actions randomly.

**Persistence baseline**

Predicts that the next state will remain similar to the current state.

**Reactive baseline**

Uses only the current observation.

**Learned Caspian**

Uses learned internal state and experience.

The purpose is to determine whether learning provides an actual advantage.

---

# 14. Initial Metrics

Phase 1 should measure:

## 14.1 Prediction error

$$ L_{\text{pred}} $$

Lower error indicates better prediction under the chosen metric.

## 14.2 Learning curve

Plot:

$$ \text{performance vs interaction count} $$

## 14.3 Sample efficiency

How much experience is required to reach a specified performance level?

## 14.4 Generalization

Evaluate on:

- unseen starting positions
- unseen trajectories
- unseen entity positions
- held-out environment configurations.

## 14.5 Reproducibility

Run multiple random seeds.

## 14.6 Internal representation consistency

Test whether similar functional situations produce similar internal states.

---

# 15. Project Development Method

We use Agent-Driven Development.

The coding agent is not allowed to freely expand the project.

Every task has a contract.

```
TASK
↓
CONTEXT
↓
CONSTRAINTS
↓
ACCEPTANCE TESTS
↓
IMPLEMENTATION
↓
TEST
↓
EXPERIMENT
↓
DOCUMENT
↓
REVIEW
```

The development agent should optimize for:

- correctness
- reproducibility
- simplicity
- tests
- scientific traceability.

Not:

- unnecessary complexity
- impressive architecture
- premature scaling
- hidden assumptions.

---

# 16. Agent Task Contract

Every coding-agent task should contain:

**TASK** — What exactly must be built?

**CONTEXT** — Why does it exist?

**CONSTRAINTS** — What must not change?

**ACCEPTANCE TESTS** — How will we know it works?

**EXPERIMENT** — What experiment should be run?

**DOCUMENTATION** — What must be recorded?

**DO NOT** — What shortcuts or semantic leakage are prohibited?

The coding agent must report:

- files changed
- tests created
- tests passed/failed
- commands executed
- experiment results
- unexpected behavior
- assumptions
- limitations.

---

# 17. Repository Structure

Initial repository:

```
caspian/
│
├── README.md
├── pyproject.toml
│
├── environment/
│   ├── world.py
│   ├── entities.py
│   ├── dynamics.py
│   └── observations.py
│
├── agent/
│   ├── caspian.py
│   ├── policy.py
│   └── memory.py
│
├── models/
│
├── learning/
│   ├── losses.py
│   ├── optimizer.py
│   └── trainer.py
│
├── evaluation/
│   ├── metrics.py
│   └── baselines.py
│
├── experiments/
│
├── analysis/
│
├── docs/
│   └── experiments/
│
├── configs/
│
├── checkpoints/
│
├── logs/
│
└── tests/
```

---

# 18. Experiment Naming

Every experiment receives a permanent identifier.

Example:

```
CSP-P1-E001
CSP-P1-E002
CSP-P1-E003
```

Format:

```
CSP-P<phase>-E<experiment>
```

Example:

```
CSP-P1-E001
```

means:

Caspian, Phase 1, Experiment 001.

If an experiment is repeated with different seeds:

```
CSP-P1-E001-S01
CSP-P1-E001-S02
CSP-P1-E001-S03
```

---

# 19. Experiment Artifact Structure

Each experiment should produce:

```
experiments/
└── CSP-P1-E001/
    ├── config.yaml
    ├── README.md
    ├── commands.txt
    ├── results.json
    ├── metrics.csv
    ├── logs/
    ├── checkpoints/
    ├── plots/
    └── report.docx
```

The report.docx is generated after the experiment.

It is not the project plan.

---

# 20. Mandatory Experiment DOCX

After every meaningful experiment, generate a DOCX.

The document title should contain:

```
CASPIAN EXPERIMENT REPORT
CSP-P1-E001
```

The report must contain the following.

## 20.1 Experiment identity

- Experiment ID
- Date
- Phase
- Git commit
- Configuration version
- Researcher
- Agent/coding model used, if applicable

## 20.2 Research question

What are we trying to discover?

## 20.3 Hypothesis

What do we expect?

## 20.4 Null hypothesis

What would count as evidence that the learning mechanism did not provide meaningful improvement?

## 20.5 Environment

Document:

- dimensions
- entities
- hidden rules
- state variables
- transition rules
- observation function
- action space
- termination conditions.

## 20.6 Semantic firewall audit

Document exactly what the agent could and could not know.

## 20.7 Model

Document:

- architecture
- layers
- parameter count
- activation functions
- memory
- policy
- prediction head.

## 20.8 Mathematics

Show:

- state equations
- observation equations
- prediction equations
- loss
- optimization rule.

## 20.9 Training

Document:

- optimizer
- learning rate
- batch size
- number of episodes
- interaction count
- random seeds
- hardware
- software versions.

## 20.10 Commands

Record the exact commands used.

## 20.11 Results

Include:

- raw numbers
- tables
- graphs
- baseline comparisons.

## 20.12 Failures

Record every important failure.

Do not hide failed experiments.

## 20.13 Unexpected behavior

Document anything surprising.

Examples:

- strange movement
- unstable learning
- repeated behavior
- exploitation of environment bugs
- unexpected correlations
- degenerate strategies.

## 20.14 Debugging history

Record:

```
Problem
→ hypothesis about cause
→ change
→ test
→ result
```

## 20.15 Interpretation

Separate:

**Evidence** — What was actually observed?

**Interpretation** — What might explain it?

**Alternative explanations** — What else could produce the same result?

## 20.16 Limitations

What does this experiment fail to establish?

## 20.17 Conclusion

Use explicit categories:

- SUPPORTED
- PARTIALLY SUPPORTED
- NOT SUPPORTED
- INCONCLUSIVE

Do not force a positive conclusion.

## 20.18 Next step

The next experiment must be justified by evidence from the current experiment.

---

# 21. Phase Roadmap

The project progresses through phases.

```
PHASE 0
Research & Reproducibility Foundation
        ↓
PHASE 1
Predictive Interaction
        ↓
PHASE 2
Memory & Persistence
        ↓
PHASE 3
Generalization
        ↓
PHASE 4
World Model
        ↓
PHASE 5
Emergent Categories
        ↓
PHASE 6
Rule Change & Adaptation
        ↓
PHASE 7
Causal Experimentation
        ↓
PHASE 8
Novel Worlds
        ↓
PHASE 9
Scale Study
```

A phase cannot be advanced merely because the calendar says so.

Progress must be evidence-based.

---

# 22. PHASE 0 — Research & Safety Foundation

## Goal

Build the smallest reproducible experimental infrastructure.

## Tasks

- Create repository.
- Implement deterministic 2D environment.
- Implement Caspian placeholder.
- Implement observation interface.
- Implement action interface.
- Implement environment transition system.
- Implement reset.
- Implement random baseline.
- Implement scripted/oracle baseline.
- Implement trajectory recording.
- Implement configuration files.
- Implement deterministic seeds.
- Implement unit tests.
- Implement experiment logging.

## Required tests

Test:

- movement
- boundaries
- collision
- interaction
- energy/state changes
- reset
- deterministic seeds
- observation generation
- action validity.

## Phase 0 exit criteria

Phase 0 is complete only when:

- the environment is deterministic under a fixed seed
- the oracle behaves according to the known rules
- the random baseline runs
- trajectories can be saved
- experiments can be reproduced
- tests pass
- configuration is externalized
- a new developer can run the baseline experiment.

## Required Phase 0 report

Generate:

```
CSP-P0-E001.docx
```

documenting what was actually built and tested.

---

# 23. PHASE 1 — Predictive Interaction

## Goal

Demonstrate that a small agent can learn an unlabeled environmental relationship from experience.

## Example hidden rule

```
interaction with X
→
energy changes
```

The exact relationship may be changed between experiments.

The agent should not receive the semantic explanation.

---

# 24. Phase 1 Agent

Initial agent:

```
Observation
     ↓
Encoder
     ↓
Internal state
     ↓
Prediction / policy
     ↓
Action
```

Keep the model small.

A possible first network:

$$ h_1=\sigma(W_1O_t+b_1) $$

$$ S_t=\sigma(W_2h_1+b_2) $$

Prediction:

$$ \hat O_{t+1}=W_3S_t+b_3 $$

No Transformer is required.

---

# 25. Phase 1 Experiments

**CSP-P1-E001 — Can Caspian learn?**

Question: Can Caspian predict the relevant environmental change better than a trivial baseline?

**CSP-P1-E002 — Does it beat persistence?**

Compare Caspian against the Persistence baseline.

If Caspian cannot beat a trivial predictor, complexity should not increase.

**CSP-P1-E003 — Does position matter?**

Change entity positions.

Test whether the agent learns a relationship that depends on spatial conditions.

**CSP-P1-E004 — Does experience matter?**

Compare untrained vs trained.

Measure performance as experience increases.

**CSP-P1-E005 — Is learning reproducible?**

Run multiple seeds.

Example: seed 1, seed 2, seed 3, seed 4, seed 5.

Report variance.

**CSP-P1-E006 — Semantic leakage audit**

Remove or modify potential information channels.

Test whether performance depends on accidental labels or hidden state information.

---

# 26. Phase 1 Metrics

Primary:

- prediction error
- held-out prediction error
- learning curve
- sample efficiency
- baseline gap.

Secondary:

- representation similarity
- action-outcome prediction
- adaptation to unseen positions
- performance variance across seeds.

---

# 27. Phase 1 Success Conditions

Phase 1 is considered successful only if evidence supports all or most of the following:

- Caspian learns from experience.
- Performance improves beyond an untrained agent.
- Caspian beats relevant trivial baselines.
- The result survives held-out evaluation.
- The result is reproducible across seeds.
- No major semantic leakage explains the result.
- The learned behavior cannot be explained by a simpler accidental mechanism.

The last condition is particularly important.

---

# 28. Phase 1 Failure Conditions

Failure is scientifically useful.

Possible failure:

- Caspian ≈ random
- or Caspian ≈ persistence baseline
- or training performance ↑ but held-out performance ↓
- or only one random seed works
- or performance disappears after removing a leaked feature.

These are results, not embarrassments.

---

# 29. PHASE 2 — Memory & Persistence

## Goal

Determine whether Caspian can retain information across time.

The agent should encounter situations where current observations alone are insufficient.

For example:

```
Observation at t=1
        ↓
Important event
        ↓
Observation at t=2
        ↓
Correct prediction requires remembering t=1
```

Potential architectures:

- recurrent networks
- gated memory
- explicit memory
- state-space models.

Questions:

- What information is retained?
- For how long?
- Does memory improve prediction?
- Does memory produce useful internal state?

---

# 30. PHASE 3 — Generalization

## Goal

Determine whether Caspian learns rules rather than memorizing configurations.

Test:

- new positions
- new layouts
- new trajectories
- new entity appearances
- new combinations of known properties.

Core question: Does Caspian learn a reusable relationship?

---

# 31. PHASE 4 — World Model

The project moves from simple prediction toward a learned model of environmental dynamics.

The agent should estimate:

$$ P(E_{t+1}|E_t,A_t) $$

or an internal approximation.

The world model can be used to simulate possible futures:

$$ \hat E_{t+1} \rightarrow \hat E_{t+2} \rightarrow \hat E_{t+3} \rightarrow \cdots $$

This introduces planning.

---

# 32. PHASE 5 — Emergent Categories

This is one of the central research phases.

Instead of giving Caspian categories such as food, danger, tool, enemy, resource — we create environmental regularities.

Different entities may have different functional consequences.

Caspian may develop internal distinctions that correlate with those regularities.

We then investigate:

Does the agent's internal representation contain structure corresponding to useful environmental categories that were never explicitly provided?

Important: Do not automatically call these discovered structures "concepts."

The experiment must define measurable criteria for category discovery.

---

# 33. Possible Category-Discovery Tests

A candidate internal category should ideally satisfy several properties.

**Predictiveness** — It should help predict future states.

**Stability** — It should remain meaningful across time.

**Generalization** — It should apply to unseen instances.

**Compression** — It should summarize multiple observations efficiently.

**Intervention sensitivity** — Changing the relevant environmental property should affect the representation.

**Behavioral relevance** — The representation should help produce better predictions or actions.

---

# 34. PHASE 6 — Rule Change & Adaptation

Change the world's rules.

Example:

Before: X → energy increases

Later: X → energy decreases

Question: Can Caspian detect that its existing model is wrong and adapt?

Measure:

- detection time
- adaptation speed
- catastrophic forgetting
- transfer
- old-model persistence
- new-model learning.

---

# 35. PHASE 7 — Causal Experimentation

Move beyond observation.

Caspian should actively intervene.

Instead of merely observing X, it can:

- approach X
- interact with X
- move away from X
- repeat interaction
- compare outcomes

The agent can effectively conduct experiments.

The research question becomes: Can an artificial agent discover causal structure by selecting interventions?

---

# 36. PHASE 8 — Novel Worlds

Introduce worlds that Caspian has never seen.

Change:

- geometry
- entity appearance
- dynamics
- environmental rules
- resource distributions
- action consequences.

The purpose is to test whether learned representations transfer.

---

# 37. PHASE 9 — Scale Study

Only after the smaller experiments are understood should we investigate scale.

Variables could include:

- model size
- memory capacity
- observation richness
- world complexity
- action-space complexity
- training data
- number of entities.

The question becomes: Which capabilities emerge as environmental complexity and model capacity increase?

Scaling is an experiment, not a default strategy.

---

# 38. What Would Count as Strong Evidence?

The strongest evidence would be a chain:

```
Agent receives raw/neutral observations
        ↓
Agent interacts with environment
        ↓
Agent predicts environmental consequences
        ↓
Prediction improves through experience
        ↓
Agent generalizes to unseen situations
        ↓
Agent adapts when rules change
        ↓
Internal representations contain stable
predictive structure
        ↓
Agent can use that structure for intervention
and causal discovery
```

Even this would not automatically establish consciousness.

It would provide evidence about learned world representation and adaptive intelligence.

---

# 39. Claims We Must Not Make Without Evidence

Do not claim "Caspian understands." unless the experiment defines and tests understanding.

Do not claim "Caspian developed concepts." unless measurable criteria for concept formation have been satisfied.

Do not claim "Caspian is conscious." because adaptive behavior or internal representation alone does not establish subjective experience.

Do not claim "Caspian discovered intelligence." Intelligence is multidimensional.

Do not claim "The system discovered reality." It discovered structure in the artificial environment.

Do not claim "The model works." without defining what "works" means, compared with what, under which conditions, with what statistical evidence.

---

# 40. Reproducibility Rules

Every experiment must record:

- code version
- Git commit
- configuration
- environment version
- model architecture
- parameter count
- seed
- hardware
- software versions
- training duration
- number of interactions
- hyperparameters
- dataset/trajectory version
- evaluation procedure.

A future researcher should be able to reconstruct the experiment.

---

# 41. Experiment Integrity Rules

- Never delete failed experiments.
- Never overwrite raw results.
- Never silently change a configuration.
- Never change the hypothesis after seeing the result without recording the change.
- Never hide an unexpected behavior.
- Never interpret a single run as definitive evidence.
- Never introduce semantic labels accidentally.
- Never increase complexity simply because the current experiment failed.
- Never optimize only for a desired result.
- Always preserve the original evidence.

---

# 42. Debugging vs Experimentation

Separate:

**Engineering debugging** — Example: IndexError. Fix it.

**Scientific experimentation** — Example: Agent predicts poorly. Do not immediately modify everything.

First ask:

- Is the observation sufficient?
- Is the target predictable?
- Is the baseline strong?
- Is the model capable?
- Is the learning signal correct?
- Is there leakage?
- Is the environment stochastic?
- Is the hypothesis wrong?

A failed experiment may reveal that the hypothesis is wrong.

---

# 43. Definition of Done

A task is not done because code exists.

It is done when:

- capability implemented
- tests written
- tests passed
- experiment executed
- baseline compared
- raw results preserved
- results analyzed
- limitations documented
- experiment DOCX generated
- next step justified.

---

# 44. Phase Completion Rule

A phase can only be marked complete when:

```
Implementation
+
Tests
+
Experiment
+
Baseline
+
Evaluation
+
Reproducibility
+
Documentation
+
Scientific interpretation
```

are complete.

The phase report must answer: What did we learn? — not merely: What did we build?

---

# 45. Agent-Driven Development Prompt Structure

When delegating work to a coding agent, use:

```
PROJECT:
Caspian

PHASE:
[phase number]

TASK:
[exact task]

RESEARCH PURPOSE:
[why this task exists]

CONTEXT:
[relevant existing system]

CONSTRAINTS:
[technical/scientific constraints]

SEMANTIC FIREWALL:
[what information must not be exposed]

ACCEPTANCE TESTS:
[exact tests]

EXPERIMENT:
[what should be run]

REQUIRED OUTPUTS:
[files/results]

DOCUMENTATION:
[what must be recorded]

DO NOT:
[forbidden shortcuts]

REPORT:
At the end, report:
1. Files changed
2. Tests run
3. Tests passed/failed
4. Commands used
5. Experiment results
6. Unexpected behavior
7. Assumptions
8. Limitations
9. Recommended next step
```

---

# 46. Initial Technology Principle

Use the simplest technology that can answer the current research question.

Initial stack can be:

- Python
- NumPy
- PyTorch
- Matplotlib
- pytest
- Git
- YAML/JSON configuration.

Do not introduce:

- distributed training
- large language models
- complex simulators
- unnecessary databases
- cloud infrastructure

until the experiment requires them.

---

# 47. Suggested Initial Hardware Strategy

The first experiments should be small enough to run locally.

The goal is not computational scale.

The goal is scientific control and interpretability.

A tiny experiment that can be understood completely is more valuable at this stage than a huge experiment that cannot be explained.

---

# 48. Scientific Progression

The project should gradually move through:

```
Prediction
   ↓
Memory
   ↓
Generalization
   ↓
World Modeling
   ↓
Representation
   ↓
Category Discovery
   ↓
Adaptation
   ↓
Causal Reasoning
   ↓
Novel Environments
```

Each stage depends on evidence from the previous stage.

---

# 49. Potential Long-Term Applications

If the research eventually demonstrates strong learned world representations, the ideas could contribute to:

- scientific discovery
- robotics
- autonomous systems
- engineering
- simulation
- drug discovery
- climate modeling
- manufacturing
- space exploration
- cybersecurity
- scientific AI
- adaptive control
- autonomous experimentation.

But artificial-world success does not automatically transfer to real-world environments.

Transfer must be tested separately.

---

# 50. The Deeper Research Direction

The long-term idea can be summarized as:

**Traditional AI**

```
Human knowledge
      ↓
Representation
      ↓
Model
      ↓
Prediction
```

**Caspian investigates:**

```
Artificial world
      ↓
Perception
      ↓
Interaction
      ↓
Prediction
      ↓
Internal representation
      ↓
Category formation
      ↓
World model
      ↓
Experimentation
      ↓
New knowledge
```

This is the conceptual direction of the project.

---

# 51. The Consciousness Question

The project may eventually raise a deeper question:

If an artificial system develops increasingly sophisticated models of the world and itself, what does that imply about consciousness?

At present, the answer is unresolved.

A system may demonstrate self-modeling, prediction, memory, planning, adaptation, communication, self-preservation behavior — without this establishing subjective experience.

Therefore:

```
Behavior
≠
Subjective experience
```

This distinction must remain explicit throughout the project.

---

# 52. Safety and Research Boundaries

The initial project should remain sandboxed.

Requirements:

- isolated artificial environment
- controlled actions
- human-controlled shutdown
- reproducible experiments
- no uncontrolled external-world actions
- no autonomous replication
- no unrestricted network access
- persistent experiment logs.

The initial research objective is understanding learned representations, not creating an uncontrolled autonomous system.

---

# 53. Final Research Standard

Every major claim should answer:

- What exactly happened?
- What evidence supports it?
- What baseline was used?
- Could a simpler explanation account for it?
- Did the result generalize?
- Was it reproducible?
- Was there semantic leakage?
- What remains unknown?
- What experiment would distinguish competing explanations?

---

# 54. First Milestone

The first scientific milestone is intentionally small:

Caspian learns a meaningful environmental relationship from interaction without being given its human semantic label.

If this fails, investigate why.

If it succeeds, do not immediately declare victory.

Ask: Did Caspian actually learn a general relationship, or did we accidentally make the answer easy?

Then design the next experiment.

---

# 55. Immediate Development Order

The project should begin in this order:

1. Repository
2. Environment
3. Deterministic simulation
4. Observation interface
5. Action interface
6. Random baseline
7. Oracle baseline
8. Tests
9. Experiment logger
10. First predictive model
11. Training loop
12. Evaluation
13. First experiment
14. Experiment DOCX
15. Analysis
16. Phase decision

Do not skip directly to a sophisticated agent.

---

# 56. First Experiment Contract

The first actual learning experiment should have a precise contract.

**Input** — Neutral environmental observations.

**Interaction** — Agent can move and interact.

**Hidden rule** — An environmental interaction changes a measurable state variable.

**Learning objective** — Predict the resulting environmental change.

**Baselines** — random, persistence, reactive.

**Evaluation** — Held-out trajectories and multiple seeds.

**Documentation** — Generate a complete DOCX experiment report.

**Decision** — Only after analysis decide whether Phase 1 continues, is revised, or stops.

---

# 57. Project Philosophy

The project follows five principles:

1. **Build small.** Small systems reveal causality.
2. **Measure everything.** Memory is not enough; preserve evidence.
3. **Challenge your own result.** Try to disprove the hypothesis.
4. **Earn complexity.** Every new component needs a reason.
5. **Let the experiment speak.** Do not make the experiment agree with the idea.

---

# 58. Final North Star

Caspian is not primarily a software product. It is an experimental instrument for investigating how artificial agents can construct representations of worlds.

The first world will be tiny.

The first model will be tiny.

The first experiment will be tiny.

That is intentional.

The objective is to create a chain of experiments where every increase in complexity is justified by something learned from the previous experiment.

The project should therefore progress:

```
WORLD
→
OBSERVATION
→
ACTION
→
FEEDBACK
→
PREDICTION
→
LEARNING
→
REPRESENTATION
→
GENERALIZATION
→
ADAPTATION
→
CAUSAL DISCOVERY
→
NEW WORLDS
```

And after every experiment:

```
RUN
↓
RECORD
↓
ANALYZE
↓
GENERATE DOCX REPORT
↓
REVIEW
↓
DECIDE NEXT EXPERIMENT
```

The code is the implementation.
The experiments are the evidence.
The DOCX reports are the scientific record.
This Markdown file is the living research and development plan.
