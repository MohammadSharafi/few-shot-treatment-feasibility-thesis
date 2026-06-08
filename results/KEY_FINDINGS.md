## Finding 1: Core Performance
**Result:** Proposed model achieves AUROC=0.611 vs best baseline 0.762.
**Evidence:** Exp1 results.
**Interpretation:** Few-shot alone underperforms the strongest classical baseline in this strict protocol; the final thesis claim should therefore rely on the stacked/fusion pipeline, not on raw few-shot alone.

## Finding 2: Minimum Shots Needed
**Result:** AUROC by K: K=1: 0.564, K=3: 0.564, K=5: 0.560, K=10: 0.557, K=15: 0.557
**Evidence:** Exp2 K-shot sensitivity.
**Interpretation:** Performance is nearly flat across K; simply increasing support shots did not materially improve AUROC.

## Finding 3: Federated Privacy Cost
**Result:** Centralized AUROC=0.552, Federated=0.547, Gap=0.5%.
**Evidence:** Exp3 federated experiment.
**Interpretation:** Federated learning preserves privacy with minimal performance cost.

## Finding 4: Ablation Sensitivity
**Result:** Full model AUROC=0.583.
**Evidence:** Exp4 ablation study.
**Interpretation:** The ablation table documents component sensitivity; do not claim that every added component improves performance unless it is supported by the row-level numbers.

## Finding 5: Cross-Disease Generalization
**Result:** Proposed vs LR improvement: -12.3% mean across test nodes.
**Evidence:** Exp5 cross-disease experiment.
**Interpretation:** Cross-disease transfer remains a limitation: the proposed model trails LR on average, so external/generalization claims should be stated cautiously.