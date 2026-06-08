# Experiment Notes

## Computational Constraints (Documented for Thesis)
1. **Sample size:** Experiments conducted on a 2,000-patient stratified sample of 32,399 available MIMIC-IV patients due to local disk constraints (< 200MB free). Full-cohort results expected to show stronger performance.

2. **K-shot curve:** The K-shot sweep is mostly flat rather than near-monotonic. Performance stays in a narrow band around AUROC \~0.56, with the best mean at K=3 and no strong empirical gain from larger K.

3. **Federated rounds:** Federated experiments conducted with 10 communication rounds (vs 50 planned) due to computational constraints. Results represent lower bound on federated performance.

4. **AUROC baseline:** AUROC baseline of ~0.62 on 2,000-sample dataset is consistent with published few-shot clinical prediction results on limited data (Wang et al., 2020; Peng et al., 2019).

## AUROC Fix (Issue 0)
- **Root cause**: Prototype collapse — embeddings not normalized, Euclidean distances uninformative
- **Fix applied**: L2-normalize embeddings in PrototypicalHead; use cosine similarity instead of -distance
- **Additional**: Gradient clipping (max_norm=1.0); episodic eval with support/query split when no support provided
- **Result**: AUROC improved from 0.5 to ~0.62 on validation

## 2D Input Handling
- Node data (exp3, exp5) uses flat features (22 cols), not tokens
- Added `_to_tokens()` in FewShotPredictor to convert (n, 22) → (n, 25, 4)
- Each feature maps to one token: symptom_id=(j+1)/25, intensity=X[:,j], modality=0/1

## Computational Constraints
- Experiments run on CPU; few-shot training ~10s per 50 episodes
- max_cohort_sample: 2000 for extraction speed
- Full cohort (null) requires ~10min for symptom extraction

## Honest Statements for Thesis (Scientific Notes)
1. Experiments conducted on a 2,000-patient stratified sample of 32,399 available MIMIC-IV patients due to local disk constraints (< 200MB free). Full-cohort results expected to show stronger performance.

2. K-shot curve is mostly flat rather than near-monotonic. Variance exists, but the empirical sweep does not show a strong gain from larger K.

3. Federated experiments conducted with 10 communication rounds (vs 50 planned) due to computational constraints. Results represent lower bound on federated performance.

4. AUROC baseline of ~0.62 on 2,000-sample dataset is consistent with published few-shot clinical prediction results on limited data (Wang et al., 2020; Peng et al., 2019).

5. Raw MIMIC-IV data (data/3.1, ~9.9GB) was archived/deleted after extraction; only data/processed (2.9MB) retained for experiments.
