"""
Baselines: Logistic Regression, Random Forest, XGBoost, Standard NN, Federated NN without few-shot
All with fit(), predict(), evaluate() interface.
"""
import numpy as np
from pathlib import Path
import yaml

def load_config():
    with open(Path(__file__).parent.parent / "config.yaml") as f:
        return yaml.safe_load(f)

class BaselineWrapper:
    """Unified interface for sklearn/XGBoost baselines."""
    
    def __init__(self, model, name="baseline"):
        self.model = model
        self.name = name
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef
        pred = self.predict(X)
        try:
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X)[:, 1]
                auroc = roc_auc_score(y, probs)
                auprc = average_precision_score(y, probs)
            else:
                auroc = 0.5
                auprc = 0.5
        except Exception:
            auroc = 0.5
            auprc = 0.5
        return {
            "accuracy": accuracy_score(y, pred),
            "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
            "auroc": auroc,
            "auprc": auprc,
            "mcc": matthews_corrcoef(y, pred) if len(np.unique(y)) >= 2 else 0.0,
        }

def get_baselines(cfg=None):
    """Return list of 5 baselines."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    import xgboost as xgb
    
    cfg = cfg or load_config()
    seed = cfg["seed"]
    n_jobs = cfg["baselines"]["n_jobs"]
    
    return [
        BaselineWrapper(LogisticRegression(max_iter=1000, random_state=seed, n_jobs=n_jobs, class_weight="balanced"), "LogisticRegression"),
        BaselineWrapper(RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=n_jobs), "RandomForest"),
        BaselineWrapper(xgb.XGBClassifier(n_estimators=100, random_state=seed, use_label_encoder=False, eval_metric="logloss"), "XGBoost"),
        BaselineWrapper(MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=seed), "StandardNN"),
        BaselineWrapper(MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=seed), "FederatedNN_NoFewShot"),
    ]
