# Hypotheses (registered; see EXPERIMENT_REGISTRY.md for tests and criteria)
H1 Inductive bias: the additive architecture predicts held-out combinations in the additive world better than mlp_small.
H2 Capacity: mlp_large does NOT close the gap (mlp_large is not better than mlp_small by the H1 margin).
H3 Modularity alone: factorised (concat) beats mlp_small without any additivity assumption.
H4 Negative control valid: in the interaction world every architecture trained on data with no AB co-occurrence errs on held-out combos; the additive architecture errs by at least 0.25 nRMSE in W1 (expected ~0.42).
H5 Replication: H1 holds in W2.
H6 Adaptation: after revealing one held-out combination (20 observations), error on the UNREVEALED held-out combination in W1-interaction falls by at least 0.05 nRMSE for mlp_small and additive_mlp; for the strictly additive architecture it does NOT (predicted failure: cannot represent AB).
H7 Mechanism: additive has lower counterfactual-flip error on unseen pairs (W1 additive) than mlp_small; in W2 it has lower distractor sensitivity.
H8 Forgetting (descriptive): fine-tuning without replay raises error on training combinations; replay reduces it.
