# Thesis Defense — Anticipated Questions & Answers

## General

### Q: What is treatment feasibility?
**A:** A binary outcome: whether a patient can be safely discharged to home, home health care, or rehabilitation within 30 days of ICU admission, without readmission within 30 days. We derive this from MIMIC-IV discharge and readmission records.

### Q: Why MIMIC-IV?
**A:** MIMIC-IV is a freely accessible, de-identified ICU EHR dataset. It provides labs, vitals, and discharge information needed for our task. We use a 2,000-patient sample with fixed splits for reproducibility.

---

## Technical

### Q: Why does Random Forest outperform ETHOS+FewShot?
**A:** Several reasons: (1) Treatment feasibility may require global statistics (population-level lab distributions) rather than local support-set prototypes. (2) 2,000 patients may be insufficient for meta-learning to learn robust representations. (3) Symptom tokenization may lose information compared to raw tabular features—LR on raw features (0.724) outperforms full ETHOS (0.591). (4) The episodic prototypical setup may be mismatched to this prediction task.

### Q: What are the top features and why do they matter clinically?
**A:** Creatinine (kidney function—elevated values indicate AKI, delaying discharge), glucose (glycemic control in critically ill patients), hemoglobin (anemia/bleeding—low values may require transfusion). SOFA proxy and shock index also contribute, reflecting organ dysfunction and hemodynamic stability.

### Q: How does federated learning work in your setup?
**A:** Data are partitioned by primary ICD-10 into 5 disease nodes. Each node trains locally; the server aggregates model updates via FedAvg. Raw data never leave each node. We use 20 communication rounds, 3 local epochs per round. Federated AUROC (0.624) matches centralized (0.619) within 0.5%.

### Q: What is the ETHOS\_RF\_Hybrid and why does it help?
**A:** We concatenate ETHOS 128-D embeddings with tabular 25-D features (153-D total) and train a Random Forest on this representation. AUROC 0.717 narrows the gap to the best classical baseline (0.760) from 0.14 to 0.04. Learned representations add complementary signal to engineered features.

---

## Limitations

### Q: What are the main limitations?
**A:** (1) 2,000 patients—results may not generalize to larger cohorts. (2) Single-center (MIMIC-IV from one region). (3) Binary feasibility label—alternative definitions could yield different conclusions. (4) Simulated federated setup (disease-based partitioning); real-world federated learning faces communication costs, node failure, Byzantine clients. (5) No external validation on another ICU dataset.

### Q: Why report a negative result?
**A:** Scientifically valuable: it delineates the current limits of few-shot meta-learning for tabular clinical prediction. It guides future work (e.g., larger pretraining, hybrid architectures) and avoids overclaiming.

---

## Future & Deployment

### Q: How would you deploy this in practice?
**A:** For maximum accuracy: use the tuned Random Forest. For federated, data-scarce settings where privacy is paramount: use ETHOS+FewShot or the hybrid. The API (Appendix L) exposes /predict, /model, /explain endpoints. Calibration and recalibration (e.g., Platt scaling) recommended for deployment.

### Q: What would you do differently with more time?
**A:** (1) Larger pretraining on full MIMIC-IV. (2) External validation on eICU or another dataset. (3) Real multi-site federated trial. (4) Deeper interpretability (attention visualization, prototype analysis). (5) Prospective validation of causal estimates.

---

## Proposal Alignment

### Q: How does this align with your proposal?
**A:** We addressed all proposal objectives: (1) innovative few-shot framework, (2) Symptom Tokens as standard units, (3) ETHOS + few-shot + federated, (4) platform for personalized health without violating privacy. We reported all proposal metrics (AUROC, F1, Precision, Recall, Accuracy, Stability) and included sensitivity analysis, risk management, and 12-month timeline.
