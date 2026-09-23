# CSP-P2-E008b: Corrected Delay Sweep & Generalization (Side-by-Side Comparison)

**Phase:** Phase 2 — Memory & Persistence (CSP-P2-E008 Reconciliation)

```json
{
  "experiment_id": "CSP-P2-E008b",
  "title": "Corrected Delay Sweep & Generalization (Side-by-Side Comparison)",
  "seed": 42,
  "canonical_delay_sweep": [
    {
      "delay": 0,
      "memory_model_mse": 1.466198,
      "matched_mlp_mse": 1.287785,
      "small_mlp_mse": 0.018874,
      "persistence_mse": 4.52,
      "memory_vs_matched_gap_pct": -13.85,
      "memory_vs_small_gap_pct": -7668.35,
      "beats_matched_mlp": false,
      "beats_small_mlp": false,
      "beats_persistence": true
    },
    {
      "delay": 1,
      "memory_model_mse": 1.324983,
      "matched_mlp_mse": 4.207527,
      "small_mlp_mse": 4.467004,
      "persistence_mse": 5.48,
      "memory_vs_matched_gap_pct": 68.51,
      "memory_vs_small_gap_pct": 70.34,
      "beats_matched_mlp": true,
      "beats_small_mlp": true,
      "beats_persistence": true
    },
    {
      "delay": 2,
      "memory_model_mse": 2.747635,
      "matched_mlp_mse": 4.112186,
      "small_mlp_mse": 4.029934,
      "persistence_mse": 4.52,
      "memory_vs_matched_gap_pct": 33.18,
      "memory_vs_small_gap_pct": 31.82,
      "beats_matched_mlp": true,
      "beats_small_mlp": true,
      "beats_persistence": true
    },
    {
      "delay": 4,
      "memory_model_mse": 3.438655,
      "matched_mlp_mse": 3.401802,
      "small_mlp_mse": 3.385044,
      "persistence_mse": 3.88,
      "memory_vs_matched_gap_pct": -1.08,
      "memory_vs_small_gap_pct": -1.58,
      "beats_matched_mlp": false,
      "beats_small_mlp": false,
      "beats_persistence": true
    },
    {
      "delay": 8,
      "memory_model_mse": 0.091945,
      "matched_mlp_mse": 0.127847,
      "small_mlp_mse": 0.065378,
      "persistence_mse": 1.0,
      "memory_vs_matched_gap_pct": 28.08,
      "memory_vs_small_gap_pct": -40.64,
      "beats_matched_mlp": true,
      "beats_small_mlp": false,
      "beats_persistence": true
    }
  ],
  "canonical_generalization": {
    "train_delays": [
      1,
      2,
      4
    ],
    "test_delays": [
      3,
      6
    ],
    "results": {
      "delay_3": {
        "delay": 3,
        "type": "interpolation",
        "memory_mse": 2.935475,
        "matched_mlp_mse": 2.925422,
        "persistence_mse": 3.56,
        "gap_vs_persistence_pct": 17.54,
        "gap_vs_matched_mlp_pct": -0.34,
        "beats_persistence": true,
        "beats_matched_mlp": false,
        "generalization_successful": true
      },
      "delay_6": {
        "delay": 6,
        "type": "extrapolation",
        "memory_mse": 2.795933,
        "matched_mlp_mse": 2.852854,
        "persistence_mse": 2.92,
        "gap_vs_persistence_pct": 4.25,
        "gap_vs_matched_mlp_pct": 2.0,
        "beats_persistence": true,
        "beats_matched_mlp": true,
        "generalization_successful": true
      }
    },
    "generalizes_to_all_unseen_delays": true
  },
  "side_by_side_comparisons": {
    "delay_sweep": [
      {
        "delay": 0,
        "original_memory_mse": 0.219141,
        "original_no_mem_mse": 0.214152,
        "original_gap_pct": -2.33,
        "e007_matched_gap_pct": "N/A (d=2 only)",
        "corrected_memory_mse": 1.466198,
        "corrected_matched_mlp_mse": 1.287785,
        "corrected_gap_pct": -13.85,
        "beats_matched_mlp": false
      },
      {
        "delay": 1,
        "original_memory_mse": 2.502476,
        "original_no_mem_mse": 2.689035,
        "original_gap_pct": 6.94,
        "e007_matched_gap_pct": "N/A (d=2 only)",
        "corrected_memory_mse": 1.324983,
        "corrected_matched_mlp_mse": 4.207527,
        "corrected_gap_pct": 68.51,
        "beats_matched_mlp": true
      },
      {
        "delay": 2,
        "original_memory_mse": 3.470325,
        "original_no_mem_mse": 3.726696,
        "original_gap_pct": 6.88,
        "e007_matched_gap_pct": 7.6,
        "corrected_memory_mse": 2.747635,
        "corrected_matched_mlp_mse": 4.112186,
        "corrected_gap_pct": 33.18,
        "beats_matched_mlp": true
      },
      {
        "delay": 4,
        "original_memory_mse": 4.887201,
        "original_no_mem_mse": 5.299342,
        "original_gap_pct": 7.78,
        "e007_matched_gap_pct": "N/A (d=2 only)",
        "corrected_memory_mse": 3.438655,
        "corrected_matched_mlp_mse": 3.401802,
        "corrected_gap_pct": -1.08,
        "beats_matched_mlp": false
      },
      {
        "delay": 8,
        "original_memory_mse": 5.863892,
        "original_no_mem_mse": 6.548174,
        "original_gap_pct": 10.45,
        "e007_matched_gap_pct": "N/A (d=2 only)",
        "corrected_memory_mse": 0.091945,
        "corrected_matched_mlp_mse": 0.127847,
        "corrected_gap_pct": 28.08,
        "beats_matched_mlp": true
      }
    ],
    "generalization": {
      "delay_3_interpolation": {
        "original_e004_mse": 5.723145,
        "original_e004_beats_persistence": false,
        "e007_zero_shot_mse": 2.653152,
        "e007_zero_shot_beats_persistence": true,
        "corrected_e008_canonical_mse": 2.935475,
        "corrected_e008_beats_persistence": true,
        "corrected_e008_gap_vs_persistence_pct": 17.54
      },
      "delay_6_extrapolation": {
        "original_e004_mse": 5.378901,
        "original_e004_beats_persistence": false,
        "e007_zero_shot_mse": 4.357915,
        "e007_zero_shot_beats_persistence": true,
        "corrected_e008_canonical_mse": 2.795933,
        "corrected_e008_beats_persistence": true,
        "corrected_e008_gap_vs_persistence_pct": 4.25
      }
    }
  }
}
```
