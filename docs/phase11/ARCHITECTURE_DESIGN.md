# Architecture design (parameter counts verified by tests, 3 channels)

All models receive only the property-channel vector (one slot per channel), not positions or entity codes. This is an input simplification versus Phase 10, which fed the 37-dim encoding; architecture A1 reproduces the Phase 10 full-input pipeline for continuity.

| Name | Structure | Params (3 ch) | Built-in assumptions |
|---|---|---|---|
| mlp_small (A) | MLP (32,16) | 673 | none beyond MLP |
| mlp_large (capacity control) | MLP (64,32) | 2369 | none; 3.5x params |
| factorised (B) | one encoder per channel (1-16-8), embeddings concatenated, head (16) | 921 | channel slots are separate modules; head may learn any interaction |
| additive (C) | same encoders, embeddings SUMMED, linear head | 513 | strictly additive across channels, IMPOSED not discovered |
| additive_mlp | encoders, SUM, MLP head (16) | 665 | additive in embedding space only |
| oracle (D) | analytical formula `spec.truth`; additive-dummy regression as a fitted reference | n/a | upper bound, not a learner |
| mlp_full_input_phase10 (A1) | Phase 10 (32,16) on the 37-dim encoding, PredictiveTrainer | see log | Phase 10 pipeline |

Training (identical for all): Adam lr 0.01, weight decay 1e-4, MSE, batch 32, 200 epochs, no early stopping, no best-weight restoring, inputs normalised with a training-fit normaliser. Not tuned.
Interpretation rule: success of `additive` shows the usefulness of an imposed bias, not ontology discovery.
