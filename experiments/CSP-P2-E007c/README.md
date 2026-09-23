# CSP-P2-E007c — Representation-Metric Sanity Check

## Phase
Phase 2 Diagnostic Follow-Up (CSP-P2-E007)

## Summary Results
```json
{
  "experiment_id": "CSP-P2-E007c",
  "title": "Representation-Metric Sanity Check",
  "seed": 42,
  "n_delayed_steps": 11,
  "n_normal_steps": 239,
  "representation_analysis": {
    "inter_class_cosine_similarity": 0.94951,
    "inter_class_euclidean_distance": 1.2966,
    "cosine_similarity_threshold": 0.95,
    "cosine_sim_margin_from_threshold": 0.00049,
    "distinct_memory_representations_formed": true,
    "representation_distinctness_confidence": "borderline",
    "delayed_steps_intraclass_variance": 1.5955,
    "normal_steps_intraclass_variance": 1.4429,
    "pca_top2_explained_variance_ratio": [
      0.4816,
      0.1509
    ]
  },
  "threshold_audit": {
    "documented_threshold": 0.95,
    "threshold_logic": "distinct := cos_sim < threshold",
    "original_cos_sim": 0.9495,
    "corrected_cos_sim": 0.94951,
    "original_distinct_flag": true,
    "corrected_distinct_flag": true,
    "original_flag_technically_correct": true,
    "original_flag_scientifically_confident": false,
    "contradiction_resolved": "No logical contradiction: cos_sim=0.9495 < threshold=0.95, so distinct_memory_representations_formed=True is technically correct. However, the margin is only 0.0005, which is within measurement noise for this sample size. The boolean is borderline rather than a confident claim of distinct representations."
  }
}
```
