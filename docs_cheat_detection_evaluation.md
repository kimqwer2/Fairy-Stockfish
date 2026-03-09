# FJACE Evaluation Report (Template)

This repository patch includes the framework and test hooks for:
- human negative corpus false-positive estimation (ELS>=80 target FPR<=0.5%)
- synthetic injected positives (1/9, 3/9, 5/9) and ROC/AUC
- cross-validation by player
- forced-position filtering checks
- multiple-testing correction reporting (Holm/Bonferroni)

Use `tools/analyze_batch.sh` to generate per-game JSON, then aggregate externally.
