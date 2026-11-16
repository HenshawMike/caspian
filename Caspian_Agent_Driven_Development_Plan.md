Understood. I'll give you the **actual Markdown content directly in the chat**, not generate it with code or create a separate file.

The document below combines all three sources into one master document. It preserves the original research philosophy, Agent-Driven Development method, mathematical foundations, experimental history through Phase 12, and the roadmap through Phase 20. I've also kept the distinction between demonstrated results and future research targets.

You can copy the Markdown into a file named:

`CASPIAN_MASTER_RESEARCH_AND_DEVELOPMENT_PLAN.md`

# **PROJECT CASPIAN**

# **Consolidated Master Research, Agent-Driven Development and Experimental Roadmap — Phases 0–20**

**Project:** Caspian\
**Research direction:** Artificial Ontological Intelligence (AOI) — working research label\
**Development method:** Agent-Driven Development\
**Initial environment:** Small, isolated artificial 2D world\
**Initial agent:** Small neural agent; no LLM required initially\
**Research scope:** Phase 0–20\
**Document status:** Living research and development specification\
**Last consolidated:** 16 November 2026

# **PART I — PROJECT FOUNDATION**

## **1. Research Vision**

Caspian is an attempt to investigate a question deeper than conventional language-model scaling.

Most contemporary AI systems are trained using representations, categories, labels, relationships, and information that humans have already defined.

These systems learn from large quantities of human-generated information and structured data. Their learning is therefore influenced by the representations and assumptions already present in their training environments.

Caspian explores a different direction.

We want to create an artificial environment in which an agent can perceive and interact with things without being directly told what those things mean.

The agent should encounter environmental regularities through observation, interaction, feedback, memory, and prediction.

The researcher may understand the environment's underlying rules. However, the agent should not automatically receive those rules or their human semantic interpretations.

The central research question is:

> Can an artificial agent construct useful internal representations of an environment through perception, action, feedback, memory, and prediction without being explicitly given human semantic categories?

A more ambitious version is:

> Can an artificial learning system discover useful properties, relationships, and structures of an environment through its own interaction with that environment?

This project does not assume that success would imply consciousness.

Intelligence, representation, adaptation, agency, self-modelling, and subjective experience are separate research questions.

The project must establish its claims experimentally.

## **2. Research North Star**

> Do not begin by trying to make Caspian intelligent. Begin by creating a world in which intelligence can be measured.

The first milestone should be deliberately small.

Caspian should demonstrate that it can learn a meaningful environmental relationship from interaction without being given the human semantic label for that relationship.

Everything beyond that must be earned through experimental evidence.

## **3. Core Research Objectives**

Caspian investigates whether an artificial learner can:

1. Observe an environment without receiving a complete human-defined description of its underlying properties.
2. Learn predictive relationships from observations and actions.
3. Maintain useful internal state across time.
4. Develop representations that preserve environmental regularities.
5. Discover useful properties without being explicitly given property labels.
6. Identify relationships and interactions between environmental properties.
7. Learn from controlled interventions.
8. Generalise learned structures to unfamiliar environments.
9. Retain and update knowledge through continued experience.
10. Predict future environmental states.
11. Evaluate alternative actions and their consequences.
12. Use learned environmental models to support planning.
13. Demonstrate measurable utility in a bounded real-world application.

These are research objectives, not capabilities that Caspian has already demonstrated.

## **4. The Central Research Distinction**

There are two fundamentally different learning situations.

**Human-defined learning**

The system receives information that already communicates the meaning or intended function of something.

For example:

- This object is food.
- This object is dangerous.
- This object increases energy.
- This object is a resource.
- This action is beneficial.

**Environmental learning**

The system observes an entity, interacts with it, and observes the resulting changes.

For example:

- An unknown entity exists.
- The agent moves toward it.
- The agent interacts with it.
- A measurable internal state changes.
- The agent encounters the same entity again.
- The agent learns to predict the consequence of interacting with it.

The research interest is in the second situation.

We want to investigate whether useful internal structure can emerge from experience without explicitly supplying the human interpretation.

## **5. The Initial Research Example**

An unknown entity `X` exists in the environment.

Caspian can:

- observe `X`;
- move toward or away from `X`;
- interact with `X`;
- observe changes in its internal state;
- remember previous interactions;
- predict what may happen after another interaction.

The researcher may know that interacting with `X` increases an internal energy variable.

However, the agent should not receive labels such as:

```
food
beneficial
energy_source
reward_object
useful
dangerous
resource
target
```

The intended learning relationship is:

\\[ \text{Observation} \rightarrow \text{Interaction} \rightarrow \text{State Transition} \rightarrow \text{Prediction} \\]

Not:

\\[ \text{Human Label} \rightarrow \text{Known Meaning} \\]

The environment's internal implementation may contain information that the agent must not observe. The observation interface must strictly control what information reaches the model.

# **PART II — RESEARCH PRINCIPLES**

## **6. Foundational Principles**

These principles govern every phase of Caspian.

### **6.1 Evidence before complexity**

Do not introduce more complex architectures merely because they appear more sophisticated.

Every increase in complexity must answer a research question.

### **6.2 Measure everything**

Every meaningful capability must have measurable criteria.

Avoid relying on impressions such as:

- the agent seems smarter;
- the representation looks meaningful;
- the model appears to understand;
- the architecture looks promising.

Replace these with measurable experiments.

### **6.3 Challenge your own results**

A successful experiment is not the end of an investigation.

Ask:

- Could the result be caused by data leakage?
- Could the model be memorising?
- Is the baseline too weak?
- Could the environment make the task trivial?
- Does the model actually use the representation?
- Does the result survive different random seeds?
- Does the result survive a different environment?
- Could another architecture achieve the same result more simply?

### **6.4 Earn complexity**

Do not scale the system prematurely.

The project should move from simple environments to more complex environments only when the existing research question has been adequately tested.

### **6.5 Preserve negative findings**

Failed experiments are scientific results.

Do not remove failed hypotheses, unsuccessful runs, unexpected behaviour, or inconvenient measurements from the research record.

### **6.6 Separate engineering success from scientific success**

Code can work perfectly while the scientific hypothesis fails.

A model can train successfully and still fail to generalise.

A model can achieve low training loss without discovering the structure we intended to investigate.

### **6.7 Reproducibility**

Important results must survive independent random seeds and controlled repetitions.

A result from one successful run is preliminary evidence, not sufficient confirmation.

### **6.8 The experiment must speak**

The researcher must not force the interpretation of results.

If an experiment contradicts the hypothesis, preserve that contradiction and investigate it.

> Do not optimize for completing phases. Optimize for discovering what is true about the learning system.

## **7. Semantic Firewall**

The project must maintain a semantic firewall.

The researcher can know the meaning of environmental variables.

The agent should receive only the information explicitly permitted by the experiment.

Before every experiment, conduct a semantic-leakage audit.

Ask:

1. Did we explicitly name the object's function?
2. Did we encode its usefulness?
3. Did we encode its danger?
4. Did we encode the expected outcome?
5. Did we provide a reward that directly reveals the intended category?
6. Does the observation make the answer trivial?
7. Does the training data contain labels that should not exist?
8. Does the architecture expose hidden environmental state?
9. Does the numerical encoding accidentally reveal ordering or meaning?
10. Can the agent infer the intended answer directly from the data-generation process?

Any leakage must be documented.

### **7.1 Environment ground truth versus agent observation**

Maintain a strict separation between:

- **Environment ground truth:** Everything that exists internally in the simulated world.
- **Agent observation:** The information actually provided to the agent.
- **Researcher interpretation:** The human explanation of environmental behaviour.
- **Learned representation:** The information encoded by the model through learning.

These are not interchangeable.

### **7.2 Hidden-state protection**

Variables such as:

```
hidden_state_delta
entity_function
is_beneficial
is_dangerous
expected_outcome
semantic_category
```

must not be exposed to the agent unless the experiment explicitly investigates their use.

Even apparently neutral fields such as entity type codes, interaction flags, or numerical identifiers must be audited for accidental information leakage.

The firewall must be tested, not merely described.

# **PART III — MATHEMATICAL FOUNDATIONS**

## **8. Mathematical Framework**

Caspian should be developed from mathematical primitives rather than unexplained frameworks.

### **8.1 Environment**

Let the environment state at time \\(t\\) be:

\\[ E_t \\]

The environment evolves according to:

\\[ E\_{t+1}=T(E_t,A_t) \\]

where:

- \\(E_t\\) is the current environment state;
- \\(A_t\\) is the agent's action;
- \\(T\\) is the environment transition function.

### **8.2 Observation**

The agent does not necessarily observe the complete environment state.

Its observation is:

\\[ O_t=G(E_t) \\]

where \\(G\\) is the observation function.

This distinction is important because the agent should learn from its available information rather than receive the complete environmental implementation.

### **8.3 Internal state**

The agent maintains an internal state:

\\[ S_t=F\_\theta(S\_{t-1},O_t,A\_{t-1}) \\]

where:

- \\(S_t\\) is the internal state;
- \\(F\_\theta\\) is the learned state-transition function;
- \\(\theta\\) represents model parameters;
- \\(O_t\\) is the current observation;
- \\(A\_{t-1}\\) is the previous action.

This allows the model to incorporate historical information.

### **8.4 Policy**

The agent selects actions according to:

\\[ P(A_t\mid S_t) \\]

This represents the probability of selecting an action given the current internal state.

### **8.5 Prediction**

The model predicts future observations:

\\[ \hat O\_{t+1}=M\_\theta(S_t,A_t) \\]

The objective is to investigate whether the model can learn useful environmental dynamics from experience.

### **8.6 Predictive loss**

A basic prediction loss can be defined as:

\\[ L\_{\text{pred}}= \frac{1}{n} \sum\_{i=1}^{n} \left\\| O\_{t+1}^{(i)}-\hat O\_{t+1}^{(i)} \right\\|^2 \\]

The selected loss must match the prediction target.

### **8.7 Optimization**

Training seeks parameters that minimise the selected objective:

\\[ \theta^\*=\arg\min\_\theta L(\theta) \\]

A gradient-descent update is:

\\[ \theta'=\theta-\eta\nabla\_\theta L \\]

where \\(\eta\\) is the learning rate.

### **8.8 Probability**

For probabilistic prediction:

\\[ P(O\_{t+1}\mid S_t,A_t) \\]

For action selection:

\\[ P(A_t\mid S_t) \\]

### **8.9 Information theory**

For probabilistic predictions, cross-entropy can be used:

\\[ H(p,q)=-\sum_x p(x)\log q(x) \\]

### **8.10 Statistics**

Every major experiment should use:

- multiple random seeds;
- held-out evaluation;
- repeated trials;
- appropriate confidence intervals;
- baseline comparisons;
- variance reporting;
- controlled data splits.

The purpose is to distinguish actual learning from chance performance.

## **9. Representation Learning**

The internal representation may be one of the most important scientific objects in Caspian.

Suppose:

\\[ S_t=f\_\theta(O\_{\leq t},A\_{\<t}) \\]

The system accumulates information about its environment through observations and actions.

We want to investigate whether similar environmental conditions produce useful and consistent structures in the internal state.

Possible analyses include:

- hidden-state clustering;
- representation similarity;
- linear probes;
- nearest-neighbour structure;
- dimensionality reduction;
- predictive information;
- temporal consistency;
- intervention sensitivity.

However, these analyses require careful interpretation.

A cluster is not automatically a concept.

A probe decoding environmental information does not prove that the model's prediction head uses that information.

A representation correlating with an environmental property does not prove that the system has developed a human-like concept of that property.

Whenever possible, evaluate representations through actual prediction and behaviour.

## **10. Baselines**

Every important experiment requires appropriate baselines.

Initial baselines:

| Baseline        | Description                                                               |
| --------------- | ------------------------------------------------------------------------- |
| Random          | Selects actions randomly                                                  |
| Persistence     | Predicts that the next state remains similar to the current state         |
| Reactive        | Uses only the current observation                                         |
| Learned Caspian | Uses learned internal state and experience                                |
| Oracle          | Uses privileged information where appropriate as an upper-bound reference |

Baselines must receive comparable information and evaluation opportunities.

A weak baseline is not evidence of a strong learning mechanism.

## **11. Initial Evaluation Metrics**

### **Prediction error**

\\[ L\_{\text{pred}} \\]

### **Learning curve**

Measure:

\\[ \text{Performance versus interaction count} \\]

### **Sample efficiency**

How much experience is required to reach a defined performance level?

### **Generalisation**

Evaluate:

- unseen starting positions;
- unseen trajectories;
- unseen entity positions;
- held-out environmental configurations;
- unfamiliar combinations of properties.

### **Reproducibility**

Repeat experiments across random seeds.

### **Representation consistency**

Test whether similar functional situations produce similar internal representations.

# **PART IV — AGENT-DRIVEN DEVELOPMENT**

## **12. Development Method**

Caspian uses Agent-Driven Development.

The coding agent must not freely expand the project.

Every task has a defined contract.

The development sequence is:

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

The coding agent should optimize for:

- correctness;
- reproducibility;
- simplicity;
- tests;
- scientific traceability.

It should not optimize for:

- unnecessary complexity;
- impressive architecture;
- premature scaling;
- hidden assumptions;
- unrequested features.

## **13. Agent Task Contract**

Every coding-agent task should contain:

**TASK** — What exactly must be built?

**CONTEXT** — Why does it exist?

**CONSTRAINTS** — What must not change?

**ACCEPTANCE TESTS** — How will we know it works?

**EXPERIMENT** — What experiment should be run?

**DOCUMENTATION** — What must be recorded?

**DO NOT** — What shortcuts or semantic leakage are prohibited?

The coding agent must report:

1. Files changed.
2. Tests created.
3. Tests passed and failed.
4. Commands executed.
5. Experiment results.
6. Unexpected behaviour.
7. Assumptions.
8. Limitations.
9. Recommended next step.

## **14. Experiment Documentation**

After every meaningful experiment, generate a separate DOCX report.

The report should contain:

- experiment ID;
- research question;
- hypothesis;
- environment;
- architecture;
- mathematics;
- parameters;
- training procedure;
- random seeds;
- commands executed;
- results;
- graphs and tables;
- failures;
- unexpected behaviour;
- observations;
- errors;
- debugging;
- changes made;
- interpretation;
- alternative explanations;
- limitations;
- conclusions;
- hypothesis verdict;
- justification for the next experiment.

The distinction is fundamental:

> The Markdown plan describes what we intend to do. The DOCX experiment report records what actually happened.

Never rewrite an experiment report to make an experiment look cleaner than it was.

## **15. Initial Repository Structure**

The original development plan proposed:

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
├── analysis/
├── docs/
│   └── experiments/
├── configs/
├── checkpoints/
├── logs/
└── tests/
```

This is an initial design proposal, not a claim about the current repository.

The actual repository should evolve without silently deleting or losing existing experiment artifacts.

## **16. Initial Technology Stack**

Use the simplest technology capable of answering the current research question.

The original starting stack:

- Python;
- NumPy;
- PyTorch;
- Matplotlib;
- pytest;
- Git;
- YAML/JSON configuration.

Do not introduce distributed training, large language models, complex simulators, unnecessary databases, or cloud infrastructure until an experiment requires them.

Early experiments should remain small enough to run locally.

Scientific control and interpretability take priority over computational scale.

## **17. Debugging and Scientific Discipline**

Separate engineering defects from scientific outcomes.

**Engineering debugging**

A reproducible implementation defect, such as an incorrect tensor shape or indexing error, should be fixed and tested.

**Scientific experimentation**

Poor prediction, unstable learning, or failed generalisation must not trigger indiscriminate changes to the model.

Before changing a scientific design, ask:

1. Is the observation sufficient?
2. Is the target predictable from the available information?
3. Is the baseline strong?
4. Is the model capable of representing the relationship?
5. Is the learning signal correct?
6. Is there semantic leakage?
7. Is the environment stochastic?
8. Could the hypothesis itself be wrong?

A failed experiment may reveal a false hypothesis rather than a coding defect.

## **18. Definition of Done**

A development task is not complete merely because code exists.

It is complete when:

- the capability is implemented;
- tests are written and passed;
- the experiment is executed;
- baselines are compared;
- raw results are preserved;
- results are analysed;
- limitations are documented;
- the DOCX report is generated;
- the next step is justified.

A phase requires implementation, testing, experimentation, baseline comparison, evaluation, reproducibility, documentation, and scientific interpretation.

The phase review must answer:

> What did we learn?

Not merely:

> What did we build?

# **PART V — EXPERIMENTAL HISTORY: PHASES 0–12**

The following sections record the research progression and available findings.

The results should be reconciled against the original experiment reports and raw artifacts before being used in a formal publication.

## **19. Phase 0 — Research and Safety Foundation**

**Central question:** Can we establish a controlled, deterministic artificial environment in which learning experiments can be conducted and reproduced?

### **Objectives**

- Create a small artificial 2D world.
- Establish an agent and environment interface.
- Define observations and actions.
- Implement environmental state transitions.
- Establish deterministic execution.
- Create tests for the environment.
- Implement the semantic firewall.
- Establish random-seed control.

### **Initial environment**

- Grid: 5 × 5.
- Agent starting position: (0,0).
- Unknown entity X: (2,2).
- Initial internal state: 100.
- Step penalty: 1.
- Interaction radius: 1.
- Hidden entity state modifier: +10.
- Random seed: 42.

The environment may internally know the function of entity X, but the agent must not automatically receive that information.

### **CSP-P0-E001 findings**

- Determinism verified.
- 100% bitwise trajectory reproducibility.
- Semantic firewall audited and active.
- Zero recorded semantic leakages.
- 29/29 unit tests passed across 12 acceptance criteria.

Trajectory SHA-256:

```
ce150a820737511f8fcaa727c87701e423caa9e70b5e7be6155e85a057957b0b
```

### **Baseline observations**

Across five seeds and 50 steps:

| Metric                  | Random | Oracle |
| ----------------------- | ------ | ------ |
| Average final energy    | 64.0   | 100    |
| Cumulative energy delta | -36.0  | 0      |
| Successful interactions | 1.4    | 47     |
| Collisions              | 6.0    | 0      |

### **Interpretation**

Phase 0 established a deterministic and reproducible experimental environment.

It did not establish learning or intelligence.

**Status:** Foundation established.

## **20. Phase 1 — Predictive Interaction**

**Central question:** Can Caspian learn useful environmental regularities from observations and actions without receiving human semantic labels?

### **Objectives**

- Build an initial predictive neural model.
- Learn relationships between observations, actions, and future environmental states.
- Compare against non-learning baselines.
- Evaluate generalisation to unseen trajectories and entity positions.
- Investigate sample efficiency.

### **Evaluation**

Initial experiment:

- 600 exploratory training steps.
- 200 independent held-out evaluation steps.

### **Held-out results**

| Model                 | MSE       | RMSE   | MAE    | R²      |
| --------------------- | --------- | ------ | ------ | ------- |
| Caspian PredictiveMLP | 2.363496  | 1.5374 | 0.6142 | 0.6026  |
| Persistence           | 6.02      | 2.4536 | 1.53   | -0.0123 |
| Reactive              | 6.050569  | 2.4598 | 1.0567 | -0.0174 |
| Random                | 38.986277 | 6.2439 | 5.4163 | -5.5555 |

### **Extended findings**

- Eight-seed mean R²: 0.4430.
- Reproducibility: 100%.
- Generalisation to unseen entity positions was reported.

### **Sample efficiency findings**

| Training steps | MSE    | R²      |
| -------------- | ------ | ------- |
| 25             | 206.99 | -112.55 |
| 50             | 40.41  | -21.17  |
| 100            | 1.92   | -0.0536 |
| 250            | 1.71   | 0.060   |
| 500            | 2.22   | -0.2197 |
| 1000           | 0.2189 | 0.8799  |

The training curve was non-monotonic.

Removing action information or entity proximity degraded prediction accuracy.

Latent clustering was also reported between functional interactions and ordinary spatial transitions.

### **Interpretation**

The findings support predictive learning under the tested conditions.

However:

- prediction does not establish understanding;
- clustering does not establish concept formation;
- the non-monotonic learning curve requires further investigation;
- more training data should not automatically be assumed to improve performance.

**Status:** Predictive learning supported within the tested setup.

## **21. Phase 2 — Memory and Temporal Persistence**

**Central question:** Can Caspian use information from previous observations when the current observation alone is insufficient?

### **Research motivation**

A memoryless model estimates:

\\[ P(O\_{t+1}\mid O_t,A_t) \\]

A model with temporal information estimates:

\\[ P(O\_{t+1}\mid O\_{\leq t},A\_{\<t}) \\]

The second formulation allows the model to use information accumulated across previous interactions.

### **Objectives**

- Introduce temporal dependencies.
- Compare memoryless and recurrent architectures.
- Test whether historical observations improve prediction.
- Introduce delayed environmental effects.
- Evaluate memory under different delay lengths.
- Test generalisation to unseen delays.
- Ensure hidden state resets correctly between episodes.

### **Proposed comparisons**

- Feedforward model without persistent memory.
- Recurrent neural network.
- GRU-based model.
- Persistence baseline.

### **Important experimental requirement**

Hidden state must be reset at episode boundaries.

Otherwise, information from one episode may leak into another and invalidate the evaluation.

### **Historical evidence note**

The available project history contains later recurrent-model results from Phase 3, including delayed-dependency and temporal generalisation findings.

However, the complete Phase 2 numerical report is not available in the current consolidated source material.

Its exact metrics and hypothesis verdicts must be reconciled against the primary Phase 2 report.

**Status:** Temporal-memory research initiated; full Phase 2 numerical record requires reconciliation.

## **22. Phase 3 — Spatial, Structural and Multi-Entity Generalisation**

**Central question:** Can Caspian learn predictive structures that generalise beyond the exact positions and configurations encountered during training?

### **Objectives**

- Test unseen entity positions.
- Compare recurrent and feedforward architectures.
- Expand grid dimensions.
- Introduce multiple entities.
- Test delayed environmental effects.
- Evaluate performance across independent seeds.

### **E001 — Unseen entity positions**

Five unseen positions were tested.

| Test | GRU MSE  | Matched MLP MSE |
| ---- | -------- | --------------- |
| 1    | 0.708142 | 5.097081        |
| 2    | 0.553643 | 9.9082          |
| 3    | 1.609726 | 6.570701        |
| 4    | 3.567075 | 10.876487       |
| 5    | 3.223729 | 13.388822       |

Mean reported GRU advantage: 79.83%.

Five out of five tests passed.

### **E002 — Grid generalisation**

The environment was expanded from 5 × 5 to:

- 7 × 7;
- 10 × 10.

Reported GRU advantages over matched MLP:

- 82.74%;
- 85.38%.

### **E003 — Multi-entity delayed interactions**

Two entities with delays of 1 and 3 were introduced.

| Model | MSE      |
| ----- | -------- |
| GRU   | 2.351358 |
| MLP   | 4.952039 |

Reported GRU advantage: 52.52%.

### **E004 — Eight-seed evaluation**

Mean combined advantage: 68.91%.

Standard deviation: 10.85%.

Win rate: 8/8.

The multi-entity results varied substantially across seeds:

| Seed | Reported advantage |
| ---- | ------------------ |
| 5    | 95.10%             |
| 2    | 89.23%             |
| 7    | 81.05%             |
| 8    | 75.21%             |
| 3    | 63.20%             |
| 4    | 65.94%             |
| 1    | 34.05%             |
| 6    | 4.24%              |

### **E005 — Acceptance criteria**

Six out of six criteria were reported as passed.

The criteria covered:

- spatial generalisation;
- layout generalisation;
- multi-entity performance;
- eight-seed reproducibility;
- episode-reset correctness;
- semantic-firewall compliance.

Episode reset: 100%.

Firewall violations: zero across 100 transitions.

### **Interpretation**

The findings support recurrent predictive mechanisms under the tested spatial and temporal conditions.

However, multi-entity robustness varied considerably.

The findings do not establish human-like concepts, reasoning, or general intelligence.

**Status:** Spatial and structural generalisation supported within tested environments.

## **23. Phase 4 — World Model, Counterfactual Prediction and Planning**

**Central question:** Can Caspian internally model environmental dynamics and simulate the outcomes of possible actions?

### **Research progression**

The research moves beyond predicting only the next observation.

The intended progression is:

1. One-step prediction.
2. Multi-step prediction.
3. Counterfactual prediction.
4. Model-based planning.
5. Closed-loop action execution.

### **Objectives**

- Learn action-conditioned environmental dynamics.
- Predict future states.
- Simulate alternative actions.
- Evaluate predicted consequences.
- Select actions using the learned model.
- Compare model-based planning with reactive approaches.

### **Historical findings**

The available experiment summary records:

- Counterfactual tests for \\(k=1\\) through \\(k=5\\).
- Mean reported improvement: 2.56%.
- Planning reward delta: +75.
- Eight out of eight seeds.

The full primary report should be checked before using these figures as formal evidence.

### **Scientific cautions**

A model may produce good one-step predictions while accumulating errors during long rollouts.

A planning system may achieve high reward by exploiting weaknesses in its learned model or the simulator.

A counterfactual prediction is not automatically evidence of causal understanding.

### **Interpretation**

Phase 4 introduced the investigation of internal predictive models and action selection.

The important architectural progression is:

\\[ \text{Perceive} \rightarrow \text{Learn Regularities} \rightarrow \text{Internal Model} \rightarrow \text{Predict} \rightarrow \text{Act} \rightarrow \text{Observe Again} \\]

**Status:** Preliminary world-model and planning evidence; primary report reconciliation required.

## **24. Phase 5 — Emergent Functional Structure and Continuous Properties**

**Central question:** Can Caspian develop useful functional representations that generalise across continuous environmental properties?

### **Objectives**

- Introduce continuous environmental properties.
- Evaluate representations across changing property values.
- Investigate functional clustering.
- Test generalisation beyond observed configurations.
- Evaluate whether representations support actual behaviour.

### **Historical findings**

The available summary records:

- 83.99% mean behavioural accuracy.
- 100% functional-cluster assignment across five seeds.

The findings were reported for zero-shot generalisation over continuous environmental properties.

### **Interpretation**

These results are promising evidence of functional representation under the tested conditions.

However:

- clustering does not prove concept formation;
- functional grouping does not prove autonomous ontology discovery;
- behavioural accuracy must be interpreted relative to the exact task and baselines.

The complete Phase 5 report should be reconciled before making stronger claims.

**Status:** Promising functional representation evidence; not proof of ontology.

## **25. Phase 6 — Compositional and Causal World Structure**

**Central question:** Can Caspian learn how multiple environmental properties combine and influence outcomes?

### **Objectives**

- Introduce multiple environmental properties.
- Test independent and interacting effects.
- Investigate unseen property combinations.
- Examine causal dependencies.
- Evaluate compositional generalisation.
- Test whether the model can distinguish relevant from irrelevant information.

### **Research direction**

The phase investigates the difference between learning individual environmental regularities and learning how those regularities combine.

For example, a model might predict the effect of property A and property B independently but fail when both occur together.

The important question is whether the model has learned reusable structure or merely memorised observed combinations.

### **Evidence note**

The complete primary numerical report for Phase 6 is not available in the consolidated source material.

Later Phase 10–12 experiments provide more specific evidence about compositional generalisation and architectural inductive bias, but their findings should not be retroactively attributed to Phase 6.

**Status:** Research direction established; definitive Phase 6 verdict requires primary report reconciliation.

## **26. Phase 7 — Causal Interventions**

**Central question:** Can Caspian distinguish observational relationships from causal relationships through controlled interventions?

### **Objectives**

- Introduce controlled interventions.
- Change environmental variables independently.
- Observe resulting state transitions.
- Compare observational and intervention-based predictions.
- Evaluate whether learned relationships survive changes in correlated variables.

### **Historical issue**

A Phase 7 experiment, E001, had a reproducibility problem.

Episodes began with internal state fixed at 100.

Reported reproducibility was 22.5%.

This issue must remain visible in the historical record.

It must not be hidden by reporting only later corrected experiments.

### **Interpretation**

A causal claim requires more than correlation or predictive accuracy.

The experiment must establish that the model can predict the consequences of controlled changes and distinguish those consequences from merely associated variables.

The available summary is insufficient to claim successful causal structure recovery.

**Status:** Causal research conducted; successful causal discovery not established.

## **27. Phase 8 — Novel Worlds and Environmental Properties**

**Central question:** Can Caspian learn when environmental properties and rules change in unfamiliar worlds?

### **Objectives**

- Introduce novel environments.
- Test hypotheses H2 and H3.
- Investigate how the model responds to changes in environmental properties.
- Separate familiar structure from new environmental information.
- Test adaptation and transfer.

### **Historical development**

Experiment E002 introduced an appearance-property channel.

This expanded the environmental observation design and created another dimension for investigating how the agent represents environmental information.

### **Evidence note**

The exact numerical outcomes and full hypothesis verdicts are not available in the current consolidated source material.

They must be reconciled against the original Phase 8 report.

**Status:** Novel-world research initiated; full findings require primary report reconciliation.

## **28. Phase 9 — Scaling Study**

**Central question:** How do model capacity, training data, and environmental complexity affect predictive learning and generalisation?

### **Objectives**

- Investigate model size.
- Investigate training-data budgets.
- Increase environmental complexity.
- Measure in-distribution performance.
- Measure unseen-position generalisation.
- Identify capacity and data bottlenecks.

### **E001 experimental design**

Six model sizes:

```
157
313
753
1761
4545
13185 parameters
```

Three training budgets:

```
500
2000
8000
```

Three complexity levels:

| Level | Entities | Types |
| ----- | -------- | ----- |
| C1    | 1        | 2     |
| C2    | 3        | 4     |
| C3    | 5        | 8     |

Other details:

- Five seeds.
- Sixty epochs.
- In-distribution interaction prediction R².
- Held-out 2 × 2 corner-block position generalisation.

### **Evaluation bug and correction**

The initial smoke test produced held-out R² of exactly zero.

The cause was a test-world design problem: the held-out agent began at the origin and almost never reached the held-out corner.

Only three interactions occurred across 1,500 steps.

The test agent was subsequently placed beside the held-out block, producing 200–470 interaction events per test set.

The correction was documented before the main run and did not change the research hypothesis.

### **C1 in-distribution R²**

| Budget | Six model results                              |
| ------ | ---------------------------------------------- |
| 500    | 0.3559, 0.5625, 0.5955, 0.5300, 0.6075, 0.6420 |
| 2000   | 0.6719, 0.8898, 0.9231, 0.9157, 0.9110, 0.9143 |
| 8000   | 0.7882, 0.9995, 0.9999, 0.9997, 0.9997, 0.9995 |

### **C1 unseen-position R²**

| Budget | Six model results                              |
| ------ | ---------------------------------------------- |
| 500    | 0.2661, 0.4337, 0.2971, 0.1633, 0.1787, 0.1472 |
| 2000   | 0.6676, 0.8455, 0.5866, 0.4062, 0.4527, 0.4576 |
| 8000   | 0.7705, 0.9991, 0.9908, 0.9775, 0.9931, 0.9953 |

### **C2 in-distribution R²**

| Budget | Six model results                                |
| ------ | ------------------------------------------------ |
| 500    | 0.0196, 0.0512, -0.1793, -0.0672, 0.0074, 0.0113 |
| 2000   | 0.2930, 0.2727, 0.0292, 0.1388, 0.1987, 0.1966   |
| 8000   | 0.4765, 0.6152, 0.8495, 0.8791, 0.8802, 0.8879   |

### **C2 unseen-position R²**

| Budget | Six model results                                    |
| ------ | ---------------------------------------------------- |
| 500    | -0.1108, 0.0222, -0.2187, -0.2982, -0.1617, -0.2128  |
| 2000   | -0.2420, -0.5517, -1.3764, -0.7835, -0.4163, -0.3292 |
| 8000   | 0.2063, -0.1303, 0.3811, 0.4514, 0.5008, 0.5442      |

### **C3 in-distribution R²**

| Budget | Six model results                                    |
| ------ | ---------------------------------------------------- |
| 500    | -0.3312, -0.3164, -0.3616, -0.2653, -0.1898, -0.2000 |
| 2000   | 0.1124, 0.0322, -0.1043, -0.0437, 0.0084, 0.0099     |
| 8000   | 0.3579, 0.3430, 0.0907, 0.0234, 0.1551, 0.1598       |

### **C3 unseen-position R²**

| Budget | Six model results                                    |
| ------ | ---------------------------------------------------- |
| 500    | -1.9428, -0.6133, -0.2753, -0.4925, -0.3768, -0.4984 |
| 2000   | -0.2808, -0.0780, -0.1807, -0.6163, 0.0259, 0.0022   |
| 8000   | 0.2110, 0.1964, 0.0374, -0.2920, 0.0541, 0.2132      |

### **E001 hypothesis verdicts**

- **H1 — More data or model size never hurts:** Not supported.
- **H2 — Data reduces the generalisation gap more than model size:** Weakly supported; observed at two of three complexity levels.
- **H3 — Required capacity increases with complexity:** Supported as a trend, but C3 remained unresolved.

At 8,000 training steps, the minimum model size reaching R² ≥ 0.9 was:

| Complexity | Minimum model size |
| ---------- | ------------------ |
| C1         | 313 parameters     |
| C2         | None               |
| C3         | None               |

### **E002 — Higher data budgets**

C2 was tested at 32,000 steps.

C3 was tested at 32,000 and 96,000 steps.

Models:

```
157
313
1761
13185 parameters
```

Three seeds.

#### **C2 at 32,000 steps**

| Model  | In-distribution R² | Unseen R² |
| ------ | ------------------ | --------- |
| Tiny   | 0.5028             | 0.0927    |
| Small  | 0.8587             | 0.6347    |
| Medium | 0.9362             | 0.6390    |
| XLarge | 0.9582             | 0.7970    |

#### **C3 at 32,000 steps**

| Model  | In-distribution R² | Unseen R² |
| ------ | ------------------ | --------- |
| Tiny   | 0.4608             | -0.1795   |
| Small  | 0.5237             | -0.5363   |
| Medium | 0.5062             | 0.1488    |
| XLarge | 0.4472             | 0.0941    |

#### **C3 at 96,000 steps**

| Model  | In-distribution R² | Unseen R² |
| ------ | ------------------ | --------- |
| Tiny   | 0.5055             | 0.2277    |
| Small  | 0.5896             | 0.4139    |
| Medium | 0.7845             | 0.3877    |
| XLarge | 0.8262             | 0.0126    |

### **E002 hypothesis verdicts**

- H6 — C2 reaches R² ≥ 0.9 at 32,000 steps for medium and XLarge models: Supported.
- H7 — C3 reaches R² ≥ 0.9 at 96,000 steps: Not supported.
- H8 — Tiny models remain below R² 0.9: Supported.

### **Interpretation**

The scaling study suggests:

1. Training data is a major bottleneck.
2. Capacity requirements increase with environmental complexity.
3. Excess capacity does not automatically improve performance.
4. Unseen-position generalisation can lag substantially behind in-distribution performance.
5. The most complex environment remained unresolved.

### **Limitations**

- Fixed or insufficiently tuned MLP architecture.
- Larger models may require different optimization.
- Entity type code used an opaque scalar.
- Training was passive.
- Seed counts were limited.
- Memory, richer action spaces, and richer observation channels were not comprehensively tested.

**Status:** Scaling behaviour characterised within the tested synthetic environments. Emergence was not demonstrated.

## **29. Phase 10 — Ontology Discovery and Compositional Generalisation**

**Central question:** Can Caspian autonomously discover environmental properties and use them compositionally in unfamiliar combinations?

### **Research objectives**

- Investigate autonomous property discovery.
- Test active versus passive exposure.
- Test unseen property combinations.
- Distinguish representation decodability from actual model use.
- Evaluate whether apparent discovery survives coverage controls.
- Test intervention-based advantages.

### **Registration**

Registration commit:

```
02be40c
```

Experiments:

- E001;
- E002.

### **E001 — Active versus passive exposure**

Reported results:

| Condition | Swapped-position R² |
| --------- | ------------------- |
| Active    | 0.9937 / 0.9903     |
| Passive   | 0.4836 / 0.3964     |

Active exposure was associated with substantially stronger transfer.

However, coverage differences remained a confounding explanation.

The result does not establish that active interaction itself caused the improvement independently of exposure coverage.

### **E002 — Compositional evaluation**

Four registered composition conditions were tested:

- C1;
- C2;
- C3;
- C4.

Budgets:

```
1000
2000
4000
8000
```

Ten seeds.

### **Findings**

- C1 failed.
- C2 narrowly failed.
- C3 was uninformative.
- C4 passed.
- Held-out MLP R² was approximately 0.65–0.83.
- Oracle performance was 1.0.
- Coverage baseline was best or comparable.

### **Representation probe findings**

Probes recovered information from raw or untrained inputs at R² values between 0.96 and 1.0.

This demonstrated that the environmental information was already decodable from the inputs.

It did not demonstrate that the trained model had autonomously discovered or used the property compositionally.

### **Intervention findings**

- Intervention advantage was not established.
- Coverage and balance explained apparent benefits.
- A negative control was flawed.

### **Interpretation**

Phase 10 did not establish autonomous ontology discovery.

The results exposed a critical distinction:

> Information being recoverable from a representation does not prove that the model has learned to use that information compositionally.

**Status:** Autonomous ontology discovery not established. Compositional generalisation remained insufficient.

## **30. Phase 11 — Architectural Inductive Bias and Compositional Use**

**Central question:** Does architectural structure determine whether a model can use learned information compositionally?

### **Experimental design**

CSP-P11-E001 compared:

- small MLP;
- large MLP;
- factorised model;
- additive MLP;
- strict additive model.

The study used:

- two synthetic worlds;
- two data levels;
- ten seeds.

### **High-data held-out normalised RMSE**

| Architecture    | World 1 | World 2 |
| --------------- | ------- | ------- |
| Small MLP       | 0.183   | 0.019   |
| Large MLP       | 0.115   | 0.014   |
| Strict additive | 0.001   | 0.000   |

World 1 memorisation baseline: 0.083.

### **Hypothesis findings**

- **H1:** Supported. Strict additive architecture performed best in the tested additive worlds.
- **H2:** Unsupported. Increased capacity did not close the observed gap. This does not establish that capacity can never close such a gap.
- **H3:** Partly supported. Factorised representations helped at high data, but not universally.
- **H5:** Not met.
- **H6:** Partly refuted. The additive model adapted through main-effect distortion, with training-combination error increasing by 0.0911.

### **Probe findings**

Probes recovered outcomes at R² = 1 for factorised and additive-MLP representations, even when the prediction head failed to generalise compositionally.

This reinforced the distinction between:

- information being present in a representation;
- information being accessible to a probe;
- the model actually using that information to make predictions.

### **Interpretation**

Architectural inductive bias matters when it matches the underlying environmental structure.

However, strict additive architectures may fail when the environment contains non-additive interactions.

**Status:** Architecture-specific compositional advantage supported in tested additive worlds, with important limitations.

## **31. Phase 12 — Non-Additive World and Optimizer Sensitivity**

**Central question:** Does an architectural advantage persist when the environment contains genuine non-additive interactions, and is the result sensitive to optimizer choices?

### **Registration**

Commit:

```
294303b
```

Confirmatory seeds:

```
11–20
```

These seeds were disjoint from Phase 11.

### **Non-additive world**

The W3 environmental rule was:

\\[ Y=6A-4B+2C+5AB \\]

Training combinations:

```
000
100
010
110
001
111
```

Held-out combinations:

```
101
011
```

### **High-data held-out normalised RMSE**

| Architecture      | nRMSE |
| ----------------- | ----- |
| Small MLP         | 0.179 |
| Large MLP         | 0.131 |
| Factorised        | 0.101 |
| Additive MLP      | 0.112 |
| Strict additive   | 0.192 |
| Additive residual | 0.037 |

Additive-residual model parameter count:

```
1434
```

Optimizer configuration:

```
Adam
Learning rate: 0.01
Epochs: 200
```

### **Hypothesis findings**

- H1: Not met.
- H2a: Met.
- H2b: Met.
- H3: Met.
- H4: Met.
- H5: Met.
- H6: Not met.

Reported approximate effects:

- Capacity effect: 0.07.
- Structural advantage: 0.16.

The broad architecture ranking remained stable across the tested optimizer grid.

### **Interpretation**

The strict additive advantage observed in Phase 11 did not generalise to the non-additive W3 environment.

The additive-residual model achieved the best result in this registered test.

This suggests that architectural inductive bias should match the structure of the environment.

A model that assumes only additive effects may perform extremely well in additive environments but fail when genuine interactions are present.

### **Limitations**

- Small synthetic worlds.
- High MLP variance.
- Additive-residual design was post hoc relative to the earlier comparison.
- Optimizer grid was small.
- No Phase 10-style probes or property-flip tests.
- Literature comparison remains outstanding.

**Status:** Additive-residual architecture promising in one tested non-additive world; universal architectural superiority not established.

# **PART VI — CURRENT EVIDENCE LEDGER**

## **32. What Caspian Has Demonstrated So Far**

The evidence should be described carefully.

| Research capability                  | Current evidence status                                        |
| ------------------------------------ | -------------------------------------------------------------- |
| Deterministic artificial environment | Established in Phase 0                                         |
| Semantic-firewall implementation     | Established in initial environment tests                       |
| Predictive learning                  | Supported in Phase 1                                           |
| Temporal memory                      | Evidence in Phases 2–3; Phase 2 report requires reconciliation |
| Spatial generalisation               | Supported in Phase 3                                           |
| Multi-entity robustness              | Partial; performance varied across seeds                       |
| World modelling                      | Preliminary evidence in Phase 4                                |
| Counterfactual prediction            | Preliminary evidence; primary report requires reconciliation   |
| Planning                             | Preliminary evidence; requires further validation              |
| Functional representation            | Promising Phase 5 evidence                                     |
| Continuous-property generalisation   | Promising Phase 5 evidence                                     |
| Causal discovery                     | Not established                                                |
| Novel-world adaptation               | Incomplete evidence                                            |
| Scaling behaviour                    | Characterised in Phase 9                                       |
| Autonomous ontology discovery        | Not established in Phase 10                                    |
| Compositional generalisation         | Mixed and unresolved across Phases 10–12                       |
| Architectural inductive bias         | Supported in specific tested synthetic structures              |
| Non-additive interaction modelling   | Additive-residual promising in one W3 experiment               |
| Cross-world transfer                 | Future research target                                         |
| Continual learning                   | Future research target                                         |
| Integrated predictive planning       | Future research target                                         |
| Real-world utility                   | Not yet established                                            |
| Novel scientific contribution        | Not yet established                                            |

## **33. Important Scientific Lessons So Far**

The first twelve phases have revealed several important issues.

### **33.1 Prediction is not understanding**

A model can predict environmental changes without demonstrating that it has formed human-like concepts.

### **33.2 Decodability is not compositional use**

A probe can recover information from a representation even when the model's own prediction head fails to use that information compositionally.

### **33.3 Coverage can imitate discovery**

An agent that interacts with more relevant configurations may outperform a passive agent without necessarily having discovered a better underlying representation.

### **33.4 Capacity is not a universal solution**

Increasing model size does not automatically solve generalisation or compositionality.

### **33.5 Architectural assumptions matter**

An architecture suited to an additive world may perform poorly in a world with non-additive interactions.

### **33.6 Causal claims require interventions**

Observational relationships are not enough to establish causal learning.

### **33.7 Synthetic success is not real-world transfer**

A system that succeeds in a controlled artificial world has not automatically demonstrated practical utility.

These lessons should directly influence the design of the next phases.

# **PART VII — FUTURE RESEARCH ROADMAP: PHASES 13–20**

The next eight phases are intended to move Caspian from its current predictive and representation-learning experiments toward a more complete adaptive world-model architecture.

They are research targets, not guaranteed milestones.

The research should not advance simply because the previous phase has been implemented. Each phase must produce independently reproducible evidence.

## **34. Phase 13 — Autonomous Property Discovery and Representation Learning**

Next research phase

**Central question:**

Can Caspian discover useful properties of its environment without receiving explicitly separated property channels or property labels?

### **Research motivation**

Previous phases showed that Caspian can learn predictive relationships and that architectural choices influence generalisation.

However, Phase 10 did not establish autonomous ontology discovery.

Phase 13 should directly investigate whether the agent can construct useful representations from raw or minimally processed observations rather than relying on manually separated environmental properties.

### **Objectives**

1. Remove manually supplied property channels.
2. Introduce raw environmental observations.
3. Develop a representation-learning component.
4. Investigate latent environmental property discovery.
5. Evaluate representation consistency across seeds.
6. Test whether discovered representations preserve useful environmental structure.
7. Investigate whether the learned representations support actual prediction and behaviour.
8. Establish strong controls against information leakage.

### **Proposed experiments**

**Experiment 13.1 — Raw observation generation**

Create an environment where the agent receives raw or minimally processed observations.

The environment may contain multiple underlying properties, but the observation interface must not explicitly identify them.

**Experiment 13.2 — Latent property discovery**

Investigate whether the model develops representations that preserve information about environmental properties.

**Experiment 13.3 — Representation consistency**

Test whether similar environmental structures produce comparable representations across independent seeds and model initializations.

**Experiment 13.4 — Property recovery and intervention sensitivity**

Investigate whether representations respond systematically when environmental properties are changed.

**Experiment 13.5 — Compositional generalisation**

Test whether the model can use discovered representations in combinations that were not encountered during training.

### **Evaluation**

- Held-out prediction performance.
- Representation consistency.
- Property recovery against appropriate baselines.
- Intervention sensitivity.
- Generalisation to unseen combinations.
- Behavioural performance.
- Seed stability.
- Semantic-firewall compliance.

### **Baselines**

- Random representation.
- Untrained encoder.
- Statistical feature extraction.
- Manually supplied property channels as an upper-bound comparison.
- Appropriate trained predictive baselines.

### **Success criteria**

Phase 13 succeeds only if Caspian:

- develops useful representations without receiving property labels;
- performs above appropriate baselines;
- generalises to unseen property combinations;
- demonstrates that its actual predictive or behavioural system uses the learned structure;
- reproduces the findings across independent seeds;
- passes the semantic-leakage audit.

A successful probe alone is insufficient.

**Expected milestone:** Caspian moves from receiving explicitly separated environmental properties toward learning useful representations from raw observations.

## **35. Phase 14 — Interaction Discovery and Structured Relationships**

**Central question:**

Can Caspian discover how environmental properties interact, including non-additive interactions?

### **Research motivation**

Phase 11 and Phase 12 demonstrated that architectural inductive bias matters.

The strict additive model performed exceptionally well in tested additive worlds but poorly in the non-additive W3 world.

The additive-residual model performed best in that particular non-additive experiment.

Phase 14 should investigate whether Caspian can discover and use interaction structures without being supplied with interaction labels.

### **Objectives**

1. Introduce multiple independent environmental properties.
2. Introduce interacting properties.
3. Investigate sparse interaction structures.
4. Identify independent and dependent effects.
5. Test pairwise interactions.
6. Test higher-order interactions.
7. Evaluate interaction discovery under noisy observations.
8. Investigate whether the additive-residual architecture remains effective.
9. Compare architectures under different environmental structures.

### **Proposed experiments**

**Experiment 14.1 — Independent properties**

Introduce multiple properties whose effects are independent.

Establish whether the model can recover independent predictive relationships.

**Experiment 14.2 — Sparse pairwise interactions**

Introduce selected interactions between properties while leaving others independent.

**Experiment 14.3 — Multiple simultaneous interactions**

Test environments containing several interacting relationships.

**Experiment 14.4 — Higher-order interactions**

Introduce interactions involving three or more properties.

**Experiment 14.5 — Noisy interaction discovery**

Introduce observation noise and investigate whether the model can still identify useful relationships.

**Experiment 14.6 — Unseen interaction combinations**

Hold out combinations during training and test whether the model can predict their outcomes.

### **Evaluation**

- Interaction precision.
- Interaction recall.
- False interaction detection.
- Held-out prediction error.
- Recovery of independent effects.
- Generalisation to unseen combinations.
- Robustness across seeds and environments.

### **Architecture comparisons**

Compare:

- strict additive model;
- additive-residual model;
- factorised model;
- standard MLP;
- larger MLP;
- appropriate interaction-aware architectures.

### **Success criteria**

Caspian must demonstrate that it can identify and use useful interaction structures without being supplied with interaction labels.

The result must survive multiple environmental structures, not just one favourable synthetic world.

**Expected milestone:** Caspian progresses from learning isolated environmental properties toward learning relationships between those properties.

## **36. Phase 15 — Causal Discovery and Intervention Learning**

**Central question:**

Can Caspian distinguish what merely correlates with an outcome from what actually causes it?

### **Research motivation**

Predictive relationships do not automatically establish causal relationships.

A model may learn that two variables frequently occur together without understanding what happens when one is independently changed.

Phase 15 should investigate whether controlled interventions allow Caspian to learn more reliable causal relationships.

### **Objectives**

1. Introduce controlled environmental interventions.
2. Distinguish observational relationships from causal relationships.
3. Learn the consequences of changing individual properties.
4. Investigate causal dependencies.
5. Introduce confounding variables.
6. Investigate hidden-variable effects.
7. Evaluate counterfactual predictions.
8. Test causal structure recovery where identifiable.

### **Proposed experiments**

**Experiment 15.1 — Correlation versus causation**

Create environments where variables are correlated but do not have direct causal relationships.

Test whether Caspian incorrectly treats correlation as causation.

**Experiment 15.2 — Single-variable interventions**

Change one environmental property while holding relevant variables constant.

Measure whether the model predicts the resulting change.

**Experiment 15.3 — Multiple-variable interventions**

Change multiple properties simultaneously.

Investigate whether the model can predict combined effects.

**Experiment 15.4 — Confounding**

Introduce variables that influence multiple observed properties.

Test whether the model confuses common causes with direct relationships.

**Experiment 15.5 — Hidden-variable experiments**

Investigate whether unobserved environmental factors cause misleading observational relationships.

**Experiment 15.6 — Counterfactual outcome prediction**

Evaluate whether Caspian can predict what would have happened under an alternative intervention.

**Experiment 15.7 — Causal structure recovery**

Test recovery of causal relationships where the environment makes them identifiable.

### **Evaluation**

- Causal effect estimation error.
- Intervention prediction accuracy.
- Counterfactual prediction error.
- Causal graph recovery where identifiable.
- Robustness to confounding.
- Performance against observational-only baselines.

### **Baselines**

- Observational-only model.
- Intervention-trained model.
- Random intervention baseline.
- Oracle causal model where appropriate.

### **Success criteria**

Caspian must demonstrate that intervention-informed learning improves predictions of controlled environmental changes compared with suitable observational-only baselines.

Causal graph recovery must only be claimed where the experimental setup makes the underlying structure identifiable.

**Expected milestone:** Caspian progresses from learning environmental relationships toward learning experimentally supported causal relationships.

## **37. Phase 16 — Cross-World Transfer and Structural Generalisation**

**Central question:**

Can Caspian transfer what it has learned in one environment to a different environment without complete retraining?

### **Research motivation**

A model that learns only one environment may simply memorise its specific structure.

The more important question is whether it can retain reusable environmental relationships and apply them in unfamiliar worlds.

### **Objectives**

1. Introduce multiple structurally related worlds.
2. Change environmental properties and rules.
3. Test whether learned representations survive environmental changes.
4. Separate transferable structures from world-specific information.
5. Investigate zero-shot transfer.
6. Investigate few-shot adaptation.
7. Measure negative transfer.
8. Compare transfer against training from scratch.

### **Proposed experiments**

**Experiment 16.1 — Multiple independent worlds**

Introduce several worlds with different environmental configurations.

**Experiment 16.2 — Shared and world-specific properties**

Some properties should remain consistent across worlds while others change.

**Experiment 16.3 — Unseen environmental rules**

Introduce rules that the agent has not previously encountered.

**Experiment 16.4 — Changing observation distributions**

Change the way environmental information is presented without necessarily changing the underlying structure.

**Experiment 16.5 — Transfer without complete retraining**

Evaluate whether Caspian can use previously learned structure in a new environment.

**Experiment 16.6 — Zero-shot and few-shot adaptation**

Compare immediate transfer with adaptation after limited new experience.

### **Evaluation**

- Cross-world prediction accuracy.
- Transfer efficiency.
- Adaptation sample requirements.
- Retention of previous knowledge.
- Out-of-distribution generalisation.
- Negative transfer.
- Performance against retraining baselines.

### **Baselines**

- Train from scratch.
- Fine-tune from previous world.
- Random initialization.
- Direct transfer without adaptation.
- Appropriate oracle references.

### **Success criteria**

Caspian must demonstrate measurable transfer to unfamiliar environments beyond memorisation and direct retraining baselines.

**Expected milestone:** Caspian develops reusable environmental representations that can support learning across multiple worlds.

## **38. Phase 17 — Continual Learning and Persistent Environmental Memory**

**Central question:**

Can Caspian accumulate environmental knowledge over time without continually losing previously learned information?

### **Research motivation**

A useful adaptive system should not need to start from zero every time it encounters a new environment.

However, continually learning new information may interfere with previously learned information.

This is the problem of catastrophic forgetting.

Phase 17 investigates persistent environmental memory and continual adaptation.

### **Objectives**

1. Introduce persistent memory.
2. Learn across sequential environmental experiences.
3. Maintain previously discovered properties and relationships.
4. Adapt to changing environmental rules.
5. Investigate catastrophic forgetting.
6. Evaluate memory retrieval.
7. Measure memory efficiency.
8. Test whether previously learned knowledge can improve new learning.

### **Proposed experiments**

**Experiment 17.1 — Sequential environment exposure**

Expose Caspian to multiple environments in sequence.

**Experiment 17.2 — Persistent latent representations**

Investigate whether useful representations survive between experiences.

**Experiment 17.3 — Memory retrieval**

Test whether previously learned environmental information can be retrieved when needed.

**Experiment 17.4 — Environmental rule changes**

Change rules in previously encountered worlds.

Evaluate whether Caspian can distinguish outdated information from current environmental behaviour.

**Experiment 17.5 — Continual adaptation**

Investigate whether the system can continue learning without complete retraining.

**Experiment 17.6 — Knowledge retention and forgetting**

Measure how learning new environments affects performance on previously encountered environments.

### **Evaluation**

- Retention accuracy.
- Forgetting rate.
- Adaptation speed.
- Memory retrieval accuracy.
- Performance on previously encountered worlds.
- Memory efficiency.
- Interference between experiences.

### **Success criteria**

Caspian must retain useful knowledge while adapting to new environmental information.

The system must demonstrate measurable advantages over appropriate memoryless or reset-based baselines.

**Expected milestone:** Caspian progresses from a model that learns individual environments toward a persistent learner that accumulates environmental knowledge.

## **39. Phase 18 — Integrated Predictive World Model and Autonomous Planning**

**Central question:**

Can Caspian use its learned environmental structure, memory, and causal knowledge to predict future states and select actions?

### **Research motivation**

This phase integrates the capabilities investigated in earlier experiments.

The objective is not simply to build a larger neural network.

It is to determine whether learned representations, environmental memory, predictive models, and intervention-based knowledge can work together to support useful decision-making.

### **Objectives**

1. Integrate discovered representations with the predictive world model.
2. Condition predictions on actions and historical information.
3. Perform multi-step environmental rollouts.
4. Evaluate predictive uncertainty.
5. Use learned models for planning.
6. Compare model-based and reactive decision-making.
7. Monitor model errors.
8. Investigate recovery from inaccurate predictions.
9. Evaluate performance in unfamiliar conditions.

### **Proposed experiments**

**Experiment 18.1 — Integrated state representation**

Combine the validated representation and memory components.

**Experiment 18.2 — Action-conditioned future prediction**

Predict environmental changes under candidate actions.

**Experiment 18.3 — Multi-step prediction**

Investigate prediction over progressively longer time horizons.

**Experiment 18.4 — Counterfactual planning**

Evaluate alternative actions and their predicted consequences.

**Experiment 18.5 — Planning under uncertainty**

Investigate decision-making when predictions are uncertain.

**Experiment 18.6 — Model-based versus reactive decision-making**

Compare decisions produced using the learned world model against reactive approaches.

**Experiment 18.7 — Prediction error monitoring and recovery**

Investigate whether Caspian can detect when its internal model is inaccurate and adapt accordingly.

### **Evaluation**

- One-step prediction accuracy.
- Multi-step prediction accuracy.
- Long-horizon prediction degradation.
- Counterfactual accuracy.
- Planning reward.
- Decision regret.
- Predictive uncertainty calibration.
- Robustness to unfamiliar situations.
- Model exploitation resistance.

### **Baselines**

- Reactive policy.
- Random policy.
- Non-learning planner.
- Model-based planner.
- Oracle planner where appropriate.

### **Success criteria**

The integrated system must use its learned world model to select actions that outperform suitable reactive and non-learning baselines.

The advantage must survive controls against exploiting simulator or model errors.

**Expected milestone:** Caspian progresses from learning environmental structure to using that structure for predictive decision-making.

## **40. Phase 19 — Real-World Transfer and Application Prototype**

**Central question:**

Can the capabilities developed in controlled environments provide measurable value in a real-world problem?

### **Research motivation**

A successful synthetic research system is not automatically a useful product.

Phase 19 investigates whether validated capabilities can transfer to a bounded real-world environment.

This phase should not begin by attempting to build a general-purpose commercial AI.

It should begin with one measurable problem.

### **Objectives**

1. Identify a bounded application domain.
2. Translate the world-model architecture into a practical environment.
3. Define observable states, actions, outcomes, and constraints.
4. Evaluate transfer from synthetic research environments.
5. Build an application prototype.
6. Compare against conventional baseline systems.
7. Evaluate reliability and operational cost.
8. Investigate adaptation to changing conditions.

### **Potential application directions**

These are research candidates, not predetermined product decisions.

**Dynamic cybersecurity environments**

Investigate whether Caspian can learn patterns of system behaviour, detect changes, and support security decision-making.

**Operational decision-support systems**

Investigate whether Caspian can learn permitted operational patterns and predict relevant changes.

**Predictive systems for changing business environments**

Investigate whether Caspian can model workflows, identify changing patterns, and support operational decisions.

**Adaptive systems that learn operational patterns**

Investigate whether a system can learn from permitted interactions rather than requiring every operational relationship to be manually encoded.

**Simulation and planning environments**

Investigate whether Caspian can improve planning and adaptation in dynamic simulated systems.

### **Proposed experiments**

1. Select a domain with measurable outcomes.
2. Define the environmental state and action space.
3. Establish conventional baseline systems.
4. Evaluate transfer from controlled experiments.
5. Test learning from new environmental experiences.
6. Evaluate performance under changing conditions.
7. Conduct controlled application trials.

### **Evaluation**

- Task-specific predictive accuracy.
- Decision quality.
- Adaptation speed.
- Generalisation to unseen conditions.
- Operational reliability.
- Computational requirements.
- Human oversight requirements.
- Privacy and safety.

### **Application constraints**

The real-world environment must not become an uncontrolled experiment.

The system must operate within explicit permissions and boundaries.

For organizational applications:

- Do not perform covert employee profiling.
- Respect privacy.
- Establish consent and transparency where required.
- Use role-based access.
- Maintain appropriate data-retention policies.
- Preserve human oversight for consequential decisions.

### **Success criteria**

Demonstrate a reproducible advantage on at least one bounded real-world task against relevant baselines.

**Expected milestone:** Caspian moves from controlled artificial research environments into a practical, experimentally evaluated application.

## **41. Phase 20 — Integrated System Validation and Research Consolidation**

**Central question:**

What has Caspian actually demonstrated, and how does its architecture compare with existing approaches?

### **Research motivation**

Phase 20 is the major research consolidation phase.

It should not be treated as a celebration of completing twenty phases.

Its purpose is to establish exactly what the research has demonstrated, what remains uncertain, and whether the system offers a meaningful scientific contribution.

### **Objectives**

1. Integrate validated components from previous phases.
2. Conduct end-to-end experiments.
3. Evaluate the complete learning architecture.
4. Perform comprehensive ablation studies.
5. Compare against established methods.
6. Conduct a systematic literature review.
7. Establish the architecture's limitations.
8. Identify the actual research contribution.
9. Produce a reproducible research record.

### **Proposed experiments**

1. Full-system integration.
2. End-to-end environmental learning.
3. Property discovery evaluation.
4. Representation evaluation.
5. Interaction discovery evaluation.
6. Causal intervention evaluation.
7. Cross-world generalisation.
8. Continual learning evaluation.
9. Predictive planning evaluation.
10. Application-level validation, provided Phase 19 succeeds.
11. Component ablation.
12. Computational efficiency evaluation.
13. Reproducibility across independent seeds and environments.
14. Comparison against relevant existing architectures.

### **Evaluation dimensions**

| Dimension      | Research question                           |
| -------------- | ------------------------------------------- |
| Discovery      | Can it discover environmental properties?   |
| Representation | Does it learn useful internal structures?   |
| Interaction    | Can it identify relationships?              |
| Causality      | Can it predict intervention outcomes?       |
| Generalisation | Can it transfer to unfamiliar worlds?       |
| Memory         | Can it retain knowledge over time?          |
| Prediction     | Can it predict future states?               |
| Planning       | Can it use predictions to select actions?   |
| Adaptation     | Can it respond to changing environments?    |
| Application    | Does it provide measurable practical value? |

### **Final deliverables**

1. Integrated Caspian architecture.
2. Reproducible experimental repository.
3. Complete phase-by-phase research documentation.
4. Benchmark and evaluation suite.
5. Ablation and comparative studies.
6. System limitations and failure analysis.
7. Technical research paper.
8. Application prototype and evaluation report, provided Phase 19 succeeds.

### **Final success criteria**

Phase 20 is successful if Caspian demonstrates a coherent, reproducible set of capabilities across discovery, representation, prediction, adaptation, and planning, with clearly measured advantages and documented limitations.

Novelty and scientific contribution must be established through literature comparison, not assumed from the completion of the roadmap.

**Expected milestone:** A scientifically defensible account of what Caspian is, what it can do, how it works, where it fails, and what contribution it makes.

# **PART VIII — OVERALL RESEARCH DESTINATION**

## **42. The Intended Caspian System**

By the end of Phase 20, the intended research outcome is an integrated artificial learning system capable of investigating and demonstrating the following:

1. Observing an environment without receiving a complete human-defined description of its underlying properties.
2. Discovering useful latent environmental representations.
3. Learning relationships and interactions between environmental properties.
4. Investigating causal structure through interventions.
5. Generalising learned structures across unfamiliar environments.
6. Retaining and updating knowledge through continued experience.
7. Predicting future environmental states.
8. Evaluating alternative actions and their consequences.
9. Planning under uncertainty.
10. Demonstrating measurable utility in at least one bounded real-world application.

These are research targets, not established capabilities.

## **43. The Research Progression**

The intended progression is:

```
CONTROLLED ENVIRONMENT
        |
        v
PERCEPTION AND INTERACTION
        |
        v
PREDICTIVE LEARNING
        |
        v
TEMPORAL MEMORY AND GENERALISATION
        |
        v
WORLD MODELLING AND COUNTERFACTUAL PREDICTION
        |
        v
REPRESENTATION AND FUNCTIONAL STRUCTURE
        |
        v
PROPERTY DISCOVERY
        |
        v
INTERACTION DISCOVERY
        |
        v
CAUSAL DISCOVERY AND INTERVENTION
        |
        v
CROSS-WORLD TRANSFER
        |
        v
CONTINUAL LEARNING
        |
        v
INTEGRATED PREDICTIVE PLANNING
        |
        v
REAL-WORLD APPLICATION
        |
        v
INTEGRATED SYSTEM VALIDATION
```

This is a research dependency map, not a guarantee that each capability will emerge automatically from the previous one.

If evidence contradicts the expected progression, the research plan must adapt.

# **PART IX — PRODUCT AND COMMERCIALISATION HYPOTHESIS**

## **44. From Research to Product**

Caspian is primarily a research project.

However, a successful research direction may eventually produce practical applications.

The commercial question is:

> If an artificial system can learn useful models of unfamiliar environments through observation, interaction, memory, prediction, and action, what valuable problems could this capability solve?

Potential application areas include:

- cybersecurity;
- operational intelligence;
- business process adaptation;
- workflow prediction;
- organizational memory;
- anomaly detection;
- adaptive decision support;
- simulation and planning.

These are hypotheses, not validated markets.

### **44.1 Potential organizational application**

One possible direction is an AI system that learns permitted patterns in an organization's operational environment.

For example, an organization may have:

- recurring workflows;
- changing operational conditions;
- dependencies between systems;
- repeated events;
- patterns of resource usage;
- relationships between processes.

A system might learn these patterns and support prediction, adaptation, and decision-making.

### **44.2 What organizations would still provide**

Caspian should not be marketed as a system that requires no instruction or configuration.

Organizations would still need to provide:

- objectives;
- permissions;
- system interfaces;
- operational constraints;
- evaluation criteria;
- safety boundaries;
- relevant data access.

The research opportunity is to investigate whether the system can reduce the amount of manual encoding required to learn useful environmental structure.

### **44.3 Product validation**

A product must demonstrate measurable value.

Possible value measures:

\\[ \text{Value} = \text{Time Saved} + \text{Errors Avoided} + \text{Decisions Improved} + \text{Processes Optimized} \\]

This is a conceptual framework, not a validated financial formula.

The actual commercial opportunity must be established through:

- customer interviews;
- measurable workflow problems;
- prototypes;
- controlled trials;
- reliability testing;
- willingness-to-pay evidence;
- operational cost analysis.

The product decision should be informed by Phase 19 rather than predetermined before the research results exist.

# **PART X — EXECUTION GOVERNANCE**

## **45. Experiment Registration**

Every experiment must have:

- unique experiment ID;
- phase;
- Git commit;
- research question;
- registered hypotheses;
- predicted outcomes;
- success thresholds;
- environment specification;
- observation specification;
- action-space specification;
- semantic-firewall audit;
- architecture;
- optimizer;
- training configuration;
- data splits;
- random seeds;
- baselines;
- evaluation metrics;
- exclusion criteria;
- stopping rules.

Confirmatory experiments must not silently change their hypotheses after observing results.

Exploratory experiments and confirmatory experiments must be distinguished.

## **46. Experiment Preservation**

Preserve:

- raw outputs;
- configuration files;
- source code;
- random seeds;
- logs;
- model checkpoints;
- evaluation results;
- failures;
- deviations;
- plots;
- statistical analysis.

Never preserve only the best run.

Report the variation across seeds and explain exclusions.

## **47. Phase Exit Review**

Every phase requires an explicit evidence-based review.

The review must answer:

1. What exact research claim was tested?
2. What hypotheses were registered?
3. Which predictions passed?
4. Which predictions failed?
5. What did the baselines demonstrate?
6. Were results reproduced across seeds?
7. Were alternative explanations investigated?
8. Was the semantic firewall maintained?
9. What remains unknown?
10. Is the next phase scientifically justified?
11. What result would falsify the next phase's central hypothesis?

A phase must not advance simply because the implementation runs successfully.

## **48. Experiment Report Structure**

Every meaningful experiment should produce a DOCX report containing:

```
1. Experiment identity
2. Abstract
3. Research question
4. Background
5. Hypotheses
6. Experimental methods
7. Environment specification
8. Observation and action specification
9. Architecture
10. Mathematical formulation
11. Training configuration
12. Registration and Git commit
13. Semantic-firewall audit
14. Results
15. Baseline comparisons
16. Ablations
17. Negative controls
18. Failures and unexpected observations
19. Deviations from the plan
20. Limitations
21. Hypothesis verdicts
22. Interpretation
23. Reproduction instructions
24. Next experiment
```

The report must distinguish observed results from researcher interpretation.

## **49. Suggested Research Repository Structure**

The following is a proposed structure for the expanded research programme:

```
caspian/
│
├── README.md
├── ROADMAP.md
├── pyproject.toml
│
├── research/
│   ├── principles.md
│   ├── architecture.md
│   ├── semantic_firewall.md
│   │
│   ├── phase_00/
│   ├── phase_01/
│   ├── phase_02/
│   ├── phase_03/
│   ├── phase_04/
│   ├── phase_05/
│   ├── phase_06/
│   ├── phase_07/
│   ├── phase_08/
│   ├── phase_09/
│   ├── phase_10/
│   ├── phase_11/
│   ├── phase_12/
│   ├── phase_13/
│   ├── phase_14/
│   ├── phase_15/
│   ├── phase_16/
│   ├── phase_17/
│   ├── phase_18/
│   ├── phase_19/
│   └── phase_20/
│
├── environments/
│
├── agents/
│
├── models/
│
├── baselines/
│
├── experiments/
│   ├── configs/
│   ├── registrations/
│   ├── raw_results/
│   └── analysis/
│
├── reports/
│   ├── docx/
│   └── figures/
│
├── tests/
│
└── benchmarks/
```

This is a proposed structure. It should be adapted to the actual repository and must not cause existing research artifacts to be lost.

## **50. Research Risks and Open Questions**

The following questions remain important throughout the project.

### **Representation**

- Does the model learn useful representations?
- Does the model actually use those representations?
- Can the same structure be recovered across different seeds?
- Does representation quality improve real prediction and behaviour?

### **Discovery**

- Can properties be discovered without explicit labels?
- Can the system distinguish meaningful structure from statistical coincidence?
- How can coverage effects be separated from genuine discovery?

### **Architecture**

- Which inductive biases are useful?
- When should the model be additive?
- When should it represent interactions?
- Does a more complex architecture improve generalisation or simply fit the training environment?

### **Causality**

- Can the model distinguish correlation from causal dependence?
- Does intervention experience improve causal prediction?
- How should confounding and hidden variables be handled?

### **Generalisation**

- Can learned structures transfer between different worlds?
- How much new experience is required?
- Can the system avoid negative transfer?

### **Memory**

- Can the system preserve useful knowledge?
- How much information should be retained?
- How should outdated environmental information be updated?
- Can the system learn continuously without catastrophic forgetting?

### **World modelling**

- How accurate are long-horizon predictions?
- How does uncertainty affect planning?
- Can the model detect its own prediction failures?
- Can planning exploit errors in the learned model?

### **Real-world application**

- Which real-world environment is sufficiently bounded?
- What is the measurable value?
- What data and permissions are necessary?
- What are the safety and privacy implications?
- Is there evidence of customer demand?

### **Scientific novelty**

- How does Caspian compare with existing world models?
- How does it compare with model-based reinforcement learning?
- How does it compare with representation learning?
- How does it compare with causal discovery?
- How does it compare with continual learning?
- What is genuinely new in the research question, method, architecture, or experimental findings?

Novelty must be established through literature review and experimental comparison.

# **PART XI — FINAL RESEARCH PRINCIPLE**

## **51. What Caspian Is Ultimately Trying to Discover**

Caspian is not being developed simply to produce another neural network.

The deeper research objective is to investigate how an artificial learner can construct useful internal models of an environment through its own experience.

The research begins with a small artificial world.

It investigates prediction, memory, representation, interaction, causal relationships, transfer, continual learning, and planning.

Each capability must be independently tested.

Each claim must be supported by evidence.

Each failure must be preserved.

Each architectural decision must have a research justification.

The system must not be declared intelligent, conscious, ontological, or generally capable merely because it performs well on a collection of synthetic tasks.

Those are separate claims requiring separate evidence.

The long-term direction remains:

> Develop an artificial learning system that constructs useful models of unfamiliar environments through observation, interaction, memory, prediction, and action.

And the governing principle remains:

# **Do not optimize for completing phases. Optimize for discovering what is true about the learning system.**

**End of consolidated master document.**

One important note before you use this as the permanent master: the Phase 2, 6, 7 and 8 historical sections deliberately identify where the primary numerical reports still need reconciliation. I have not invented missing results to make the roadmap appear complete.