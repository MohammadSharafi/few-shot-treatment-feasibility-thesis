"""Downstream utility of a released dataset.

We train a classifier on the *released* (protected) training records and evaluate it on
a held-out set of *real, unprotected* records. This is the "train on anonymized, deploy
on real" contract a data-broker customer actually cares about.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def downstream_utility(
    X_train_released: pd.DataFrame,
    y_train: np.ndarray,
    X_test_real: pd.DataFrame,
    y_test: np.ndarray,
    cols: list[str],
    model: str = "gb",
    seed: int = 42,
) -> dict:
    sc = StandardScaler().fit(X_train_released[cols].values.astype(float))
    Xtr = sc.transform(X_train_released[cols].values.astype(float))
    Xte = sc.transform(X_test_real[cols].values.astype(float))

    if model == "gb":
        clf = GradientBoostingClassifier(random_state=seed)
    else:
        clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(Xtr, y_train)
    p = clf.predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y_test, p)),
        "auprc": float(average_precision_score(y_test, p)),
        "f1": float(f1_score(y_test, pred)),
    }
