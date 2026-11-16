# Known issues relevant to Phase 11
Carried from docs/phase10/KNOWN_ISSUES.md; status for Phase 11:
- #12 flawed Phase 10 negative control: RESOLVED by design. Training combinations never contain A=1 and B=1 together, so the AB term is unidentifiable; unit test asserts rule equality on all training combos and identical training data across variants.
- #2 Phase 9 agent-start confound: not used in Phase 11 (evaluation worlds hold one combination each, agent start fixed at (0,0) for all train and test collection).
- #8 single untuned optimiser setting: still applies; Phase 11 uses one fixed untuned setting for all architectures. A sensitivity check is NOT part of the registered matrix (open).
- #15/#16 literature comparison and replication: replication addressed by W2; literature review still open, so no novelty claim.
- NEW 17: Evaluation rows are interaction events only, taken from passive collection in single-combination worlds; absolute nRMSE depends on this choice.
- NEW 18: Only two held-out combinations in W1 (110, 111); per-combo statistics are thin. W2 has more.
- NEW 19: Pre-registration drafted after code and an exploratory run existed; no confirmatory data had been seen. Hyperparameters were NOT tuned on exploratory results.
- NEW 20: The additive architecture can only be right when the world is additive across properties; W2's quadratic term is within one property so it remains additive across properties by design.
- Phase 12 update: #8 (optimiser sensitivity) PARTLY CLOSED by a 6-configuration grid (lr x epochs) in Phase 12; batch size, weight decay and schedules untested. Non-additive identified world (W3) added.
