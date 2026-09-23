# CSP-P2-E007a — Early-Stopped Re-Benchmark

## Phase
Phase 2 Diagnostic Follow-Up (CSP-P2-E007)

## Summary Results
```json
{
  "experiment_id": "CSP-P2-E007a",
  "title": "Early-Stopped Re-Benchmark",
  "delay": 2,
  "seed": 42,
  "early_stopping_patience": 30,
  "memory_model": {
    "best_epoch": 48,
    "epochs_run": 78,
    "train_loss_at_best_epoch": 1.263169,
    "best_val_loss": 3.470325,
    "trainval_gap_ratio": 0.364,
    "num_parameters": 6753
  },
  "no_memory_model": {
    "best_epoch": 82,
    "epochs_run": 112,
    "train_loss_at_best_epoch": 1.914273,
    "best_val_loss": 3.726696,
    "trainval_gap_ratio": 0.5137,
    "num_parameters": 1761
  },
  "benchmark": {
    "memory_model": {
      "mse": 3.470325,
      "rmse": 1.862881,
      "mae": 0.754814,
      "r2_score": 0.174989
    },
    "no_memory_model": {
      "mse": 3.726696,
      "rmse": 1.930465,
      "mae": 0.783014,
      "r2_score": 0.114041
    },
    "persistence_baseline": {
      "mse": 4.52,
      "rmse": 2.126029,
      "mae": 1.352,
      "r2_score": -0.074553
    },
    "random_baseline": {
      "mse": 39.647951,
      "rmse": 6.296662,
      "mae": 5.467288,
      "r2_score": -8.425626
    },
    "reactive_baseline": {
      "mse": 4.257778,
      "rmse": 2.063438,
      "mae": 0.63456,
      "r2_score": -0.012214
    },
    "memory_vs_no_memory_gap_pct": 6.88,
    "memory_vs_persistence_gap_pct": 23.22,
    "memory_vs_random_gap_pct": 91.25,
    "memory_vs_reactive_gap_pct": 18.49,
    "beats_no_memory": true,
    "beats_persistence": true,
    "beats_random": true,
    "beats_reactive": true
  },
  "comparison_vs_original": {
    "original_memory_mse": 3.470325,
    "original_no_memory_mse": 3.726696,
    "original_gap_pct": 6.9,
    "corrected_memory_mse": 3.470325,
    "corrected_no_memory_mse": 3.726696,
    "corrected_gap_pct": 6.88,
    "delta_memory_mse": 0.0,
    "delta_gap_pct": -0.02
  },
  "delay_sweep": [
    {
      "delay": 0,
      "memory_model_mse": 2.127554,
      "no_memory_model_mse": 1.996597,
      "persistence_baseline_mse": 2.92,
      "memory_advantage_pct": -6.56,
      "beats_no_memory": false
    },
    {
      "delay": 1,
      "memory_model_mse": 4.519235,
      "no_memory_model_mse": 4.816968,
      "persistence_baseline_mse": 5.8,
      "memory_advantage_pct": 6.18,
      "beats_no_memory": true
    },
    {
      "delay": 2,
      "memory_model_mse": 3.035854,
      "no_memory_model_mse": 3.535628,
      "persistence_baseline_mse": 3.88,
      "memory_advantage_pct": 14.14,
      "beats_no_memory": true
    },
    {
      "delay": 4,
      "memory_model_mse": 2.150145,
      "no_memory_model_mse": 1.899804,
      "persistence_baseline_mse": 2.6,
      "memory_advantage_pct": -13.18,
      "beats_no_memory": false
    },
    {
      "delay": 8,
      "memory_model_mse": 1.694571,
      "no_memory_model_mse": 1.413546,
      "persistence_baseline_mse": 1.96,
      "memory_advantage_pct": -19.88,
      "beats_no_memory": false
    }
  ],
  "generalization_results": {
    "delay_3": {
      "delay": 3,
      "mse": 2.59978,
      "persistence_mse": 3.56,
      "gap_vs_persistence_pct": 26.97,
      "generalization_successful": true
    },
    "delay_6": {
      "delay": 6,
      "mse": 2.442108,
      "persistence_mse": 2.92,
      "gap_vs_persistence_pct": 16.37,
      "generalization_successful": true
    }
  },
  "d0_anomaly_persists_after_early_stopping": true,
  "generalization_failure_persists_after_early_stopping": false
}
```
