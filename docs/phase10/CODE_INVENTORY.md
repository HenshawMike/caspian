# Code Inventory (Stage 10.0)

Generated from the recovered phase9 zip. Hashes are the first 16 hex chars of SHA-256.

## Source files

| File | Lines | SHA-256 (16) |
|---|---|---|
| agent/__init__.py | 17 | 952d444969809d42 |
| agent/base.py | 60 | cfc3957490b756b1 |
| agent/oracle_agent.py | 78 | 9c62a2a53588c7c9 |
| agent/predictive_agent.py | 106 | 69c7d24be8d19c5a |
| agent/random_agent.py | 41 | 091f21c64dca5a96 |
| agent/recurrent_agent.py | 76 | afde46bd7d5df3af |
| environment/__init__.py | 32 | 22566f6ccbd1ce68 |
| environment/actions.py | 78 | b84ce9bd91c8a71e |
| environment/dynamics.py | 124 | 94db388d63ce456c |
| environment/entities.py | 101 | 9439f0b06f5629c2 |
| environment/observations.py | 217 | 6252c648ff71846a |
| environment/world.py | 368 | f43b1271a47bceac |
| evaluation/__init__.py | 33 | 2ab19b6255788ac0 |
| evaluation/baselines.py | 83 | 741d18721c2e4cf1 |
| evaluation/evaluator.py | 257 | 248baea09be24dd4 |
| evaluation/metrics.py | 71 | 4741fad1d1ab4b34 |
| evaluation/world_model_rollout.py | 130 | 0ea426ef0d501506 |
| experiments/baseline_comparison.py | 140 | 9e34ed7bc541a4c5 |
| experiments/deterministic_baseline.py | 139 | 9137470be5ba5f53 |
| experiments/generate_p0_report.py | 96 | 44ab1e72dd66fd90 |
| experiments/generate_p1_report.py | 261 | ae829213d2f920a1 |
| experiments/generate_p2_report.py | 14 | bf03d39eb0bcf434 |
| experiments/generate_p3_report.py | 14 | 096eed72674f4255 |
| experiments/generate_p4_report.py | 14 | 2271c9d2ee065a06 |
| experiments/generate_p5_report.py | 14 | 088fe29e9768daf6 |
| experiments/generate_p6_report.py | 14 | 9bc2684d991c1c2c |
| experiments/generate_p7_report.py | 13 | 67f15290b6d1ea63 |
| experiments/generate_p8_report.py | 13 | c5f30667355c186c |
| experiments/generate_p9_report.py | 13 | 08298e8d5daf53cb |
| experiments/logger.py | 104 | 88b66c538e6dd99e |
| experiments/phase1_experiments.py | 396 | 96d8528c3df59916 |
| experiments/phase2_experiments.py | 340 | 55e35366523e55f7 |
| experiments/phase3_experiments.py | 251 | 9a56f7839b8efd44 |
| experiments/phase4_experiments.py | 284 | 5cdc22c8c6a365b3 |
| experiments/phase5_experiments.py | 912 | 76dd92fc8f0b02d0 |
| experiments/phase6_experiments.py | 495 | d15ee46f814e7813 |
| experiments/phase7_experiments.py | 998 | a522f48b5a7e60dd |
| experiments/phase8_experiments.py | 263 | c3cc13f13d76ca2f |
| experiments/phase9_experiments.py | 186 | a696e9b9e2787173 |
| experiments/report_generator.py | 2097 | 0b18374290072268 |
| learning/__init__.py | 28 | 4b42746aa655f9df |
| learning/collector.py | 175 | 056063e35641272f |
| learning/dataset.py | 258 | 173fd96fef1683da |
| learning/deepsets_trainer.py | 137 | 97201c3c9cb63d59 |
| learning/online_trainer.py | 156 | 97650c2ca8ac321d |
| learning/recurrent_trainer.py | 139 | 0ca27807ac8211de |
| learning/sequence_dataset.py | 124 | bcd70812e5bb2fc2 |
| learning/trainer.py | 178 | b832fad66b27c5b5 |
| models/__init__.py | 36 | aff9657d8584fdd8 |
| models/activations.py | 98 | e19af0b0a18fa8eb |
| models/deepsets.py | 242 | 86545e5cad1d3204 |
| models/layers.py | 153 | 60f76688378f2799 |
| models/mlp.py | 239 | be530f21b0c45e5f |
| models/normalization.py | 65 | 447cf0d391e2c230 |
| models/optimizers.py | 179 | a36855da03246337 |
| models/rnn.py | 269 | 3fecaa782e060676 |
| tests/__init__.py | 0 | e3b0c44298fc1c14 |
| tests/test_collector_episode.py | 41 | fda7c2586259b2ff |
| tests/test_deepsets_model.py | 154 | 136446bbcc383380 |
| tests/test_entities_flicker.py | 38 | 2030190873cc63af |
| tests/test_entity_properties.py | 127 | 0fbb5ca87afbb117 |
| tests/test_normalization.py | 105 | 27e0de4331d29b1b |
| tests/test_observation_entity_ordering.py | 35 | 1291df6c2cf707ca |
| tests/test_online_trainer.py | 98 | a9d13d28e2d2959e |
| tests/test_optimizer_weight_decay.py | 36 | ed546d3b3d38529c |
| tests/test_phase5_onehot_encoding.py | 64 | 6903e905fc656b62 |
| tests/test_phase7_helpers.py | 117 | 9657e3db37b7fb2d |
| tests/test_phase8.py | 31 | aa06d7b72b918850 |
| tests/test_phase9.py | 20 | c27ecbf514cd6cd6 |
| tests/test_recurrent_trainer.py | 44 | 3985b3ea79f54528 |
| tests/test_rnn_gradient_check.py | 68 | e4d07d36a89d6d5f |
| tests/test_sequence_dataset.py | 67 | 2a771e013054a509 |
| tests/test_world_model_rollout.py | 95 | 6cfcf92db59db92d |
| tests/test_world_model_target.py | 63 | 68999d50b137a150 |
| tests/test_world_observability.py | 126 | 5fc5d9a1d2e37bce |

## Configs

| File | SHA-256 (16) |
|---|---|
| configs/default_world.json | a0da22e5cf3ab79d |
| configs/phase1_model.json | f6508c333621778a |
| configs/phase1_world.json | a0da22e5cf3ab79d |
| configs/phase2_model.json | 33c6603d63285423 |
| configs/phase2_world.json | 52cd6d35ce79737a |
| configs/phase3_world_test_layout.json | 45efa737b7ecc76b |
| configs/phase3_world_test_multi_entity.json | b387b88fc7af37ae |
| configs/phase3_world_test_new_entity_type.json | 0c1c169c6d10d5df |
| configs/phase3_world_test_positions.json | d863b85983f57d21 |
| configs/phase3_world_train_entity.json | 2378834435c031f0 |
| configs/phase3_world_train_layout.json | 2378834435c031f0 |
| configs/phase3_world_train_positions.json | 6eeb5e2f094237c5 |
| configs/phase4_world.json | aab07a9d0ee79d61 |
| configs/phase5_world_novel_beneficial.json | 9dbd0f766a87fa17 |
| configs/phase5_world_novel_harmful.json | d1f82cc8822e1523 |
| configs/phase5_world_property_train.json | 4e864d4dcd50e22a |
| configs/phase5_world_train.json | 77146ebe3c6c77a2 |
| configs/phase5_world_train_v2.json | daa8523f655612ac |
| configs/phase6_world_base.json | daa8523f655612ac |
| configs/phase6_world_baseline.json | dd4266fda3a5cf2e |
| configs/phase7_world_base.json | f7e1856e6a130091 |
| configs/phase7_world_broad_consistent.json | 5269b57c0b3ff19a |
| configs/phase7_world_single_type.json | 9ab25dd1bcb2e627 |

## Logs (raw results, do not overwrite)

| File | Bytes | SHA-256 (16) | Experiments |
|---|---|---|---|
| logs/phase2_all_experiments_results.json | 20382 | eadc0bcc4d491932 | CSP-P2-E001, CSP-P2-E002, CSP-P2-E003, CSP-P2-E004 |
| logs/phase3_all_experiments_results.json | 5296 | 7525a5ef41fecd4b | CSP-P3-E001, CSP-P3-E002, CSP-P3-E003 |
| logs/phase4_all_experiments_results.json | 4307 | 61e6c9f812954644 | CSP-P4-E001, CSP-P4-E002, CSP-P4-E003 |
| logs/phase5_all_experiments_results.json | 14512 | 955c8f4ffc6bc8f0 | CSP-P5-E001, CSP-P5-E002, CSP-P5-E003, CSP-P5-E004, CSP-P5-E004-STABILITY, CSP-P5-E005, CSP-P5-E006, CSP-P5-E007, CSP-P5-E007-STABILITY |
| logs/phase6_all_experiments_results.json | 6367 | d788acaab83cefb2 | CSP-P6-E001, CSP-P6-E002, CSP-P6-E002-STABILITY, CSP-P6-E003, CSP-P6-E004, CSP-P6-E005 |
| logs/phase7_all_experiments_results.json | 48907 | 95e25b781bafb82c | CSP-P7-E001, CSP-P7-E002, CSP-P7-E002-STABILITY, CSP-P7-E003, CSP-P7-E004, CSP-P7-E005, CSP-P7-E005-R2, CSP-P7-E006a, CSP-P7-E006b, CSP-P7-E |
| logs/phase7_e004_results.json | 5102 | bb54f382e3cc0bc3 | experiment_id, title, seeds, budget, arms, per_seed, summary |
| logs/phase7_e005_r2_results.json | 4410 | 15cfd50d6d5703e2 | experiment_id, broad_world, title, seeds, budget, arms, per_seed, summary |
| logs/phase7_e005_results.json | 6864 | 5e79af52574d5aa5 | experiment_id, title, seeds, budget, arms, per_seed, summary |
| logs/phase7_e006_e007_results.json | 12335 | 48d035e1feafb9ea | CSP-P7-E006a, CSP-P7-E006b, CSP-P7-E007 |
| logs/phase8_all_experiments_results.json | 49951 | fc0bcf955e071702 | CSP-P8-E001, CSP-P8-E002 |
| logs/phase8_e002_results.json | 9729 | f276acfd6188f3f7 | experiment_id, title, seeds, budgets, source_in_dist_r2_mean, per_seed, summary, preregistered_checks |
| logs/phase9_all_experiments_results.json | 67178 | 2b1228aa0d71fed5 | CSP-P9-E001, CSP-P9-E002 |
| logs/phase9_e002_results.json | 8231 | e5ee6d722aedc44e | experiment_id, title, seeds, sizes, plan, cells, preregistered_checks |

## Reports

| File | SHA-256 (16) |
|---|---|
| docs/CSP-P1-E001.docx | 9d657dfbc510536a |
| docs/CSP-P2-E001.docx | 5651783ffeba5606 |
| docs/CSP-P3-E001.docx | 05e173d7e9e4ab64 |
| docs/CSP-P4-E001.docx | e97238d4a7656210 |
| docs/CSP-P5-E001.docx | 0d10aee61d2386a5 |
| docs/CSP-P6-E001.docx | fc2a9b605717185c |
| docs/CSP-P7-E001.docx | 0cf20c51a070b88e |
| docs/CSP-P8-E001.docx | 764618658e30cced |
| docs/CSP-P9-E001.docx | 15740322dc092206 |
