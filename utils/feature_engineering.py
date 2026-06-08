"""
Feature engineering for treatment feasibility prediction.
Adds: shock_index, sofa_proxy, n_abnormal_labs.
Col indices: 15=HR, 16=SBP, 17=DBP, 18=MBP, 19=Temp, 20=SpO2, 21=RR
Labs: 0-14 (creatinine, K, Na, Cl, glucose, Ca, Mg, Hb, plat, WBC, lactate, ALT, AST, BUN, bicarb)
"""
import numpy as np

# Normalized ranges 0-1. "Abnormal" = far from 0.5 (mid-range)
ABNORMAL_THRESHOLD = 0.15  # |x - 0.5| > 0.35 means extreme

def add_engineered_features(X):
    """
    X: (n, 22) normalized features.
    Returns: (n, 25) with 3 extra columns: shock_index, sofa_proxy, n_abnormal.
    """
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.5, posinf=0.5, neginf=0.5)
    n = X.shape[0]
    extra = np.zeros((n, 3), dtype=np.float32)
    
    # Shock index = HR/SBP. Normalized: HR col 15, SBP col 16. Denorm approx: HR~60-120, SBP~80-160
    hr = 60 + 60 * X[:, 15]  # rough denorm
    sbp = 80 + 80 * X[:, 16]
    sbp = np.clip(sbp, 40, 250)
    shock = hr / sbp
    extra[:, 0] = np.clip((shock - 0.5) / 1.5, 0, 1)  # normalize to 0-1
    
    # SOFA-like proxy: creatinine (0), platelets (8), bilirubin proxy (use lactate 10), 
    # resp (SpO2 col 20), coagulopathy (plat)
    creat = X[:, 0]
    plat = X[:, 8]
    spo2 = X[:, 20]
    lactate = X[:, 10]
    sofa = (1 - creat) * 0.25 + (1 - plat) * 0.25 + (1 - spo2) * 0.25 + (1 - lactate) * 0.25
    extra[:, 1] = np.clip(sofa, 0, 1)
    
    # N abnormal labs (0-14)
    abnormal = np.abs(X[:, :15] - 0.5) > 0.35
    n_abn = abnormal.sum(axis=1).astype(np.float32) / 15.0
    extra[:, 2] = np.clip(n_abn, 0, 1)
    
    return np.hstack([X, extra])
