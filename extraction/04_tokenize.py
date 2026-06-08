#!/usr/bin/env python3
"""
Extraction Step 4: Convert measurements to 25-token sequences
Each token: [symptom_id, intensity, time_delta_norm, modality]
Pad with MASK (id=0). Output: tokens.npy + metadata.parquet
"""
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

def load_config():
    with open(Path(__file__).parent.parent / "config.yaml") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    paths = cfg["paths"]
    # Resolve paths relative to project root
    proj_root = Path(__file__).parent.parent
    paths = {k: str(proj_root / v) if isinstance(v, str) and not Path(v).is_absolute() else v for k, v in paths.items()}
    tok_cfg = cfg["tokenization"]
    lab_itemids = cfg["symptoms"]["lab_itemids"]
    vital_itemids = cfg["symptoms"]["vital_itemids"]
    
    seq_len = tok_cfg["seq_length"]
    mask_id = tok_cfg["mask_token_id"]
    
    # symptom_id: 1-15 labs, 16-22 vitals (0 = MASK)
    itemid_to_symptom_id = {}
    for i, itemid in enumerate(lab_itemids, start=1):
        itemid_to_symptom_id[itemid] = i
    for i, itemid in enumerate(vital_itemids, start=len(lab_itemids) + 1):
        itemid_to_symptom_id[itemid] = i
    
    thesis = pd.read_parquet(Path(paths["processed_dir"]) / "thesis_dataset.parquet")
    
    # Feature columns (exclude stay_id, subject_id, hadm_id, feasible, protocol, primary_icd10, los_hours)
    meta_cols = ["stay_id", "subject_id", "hadm_id", "feasible", "protocol", "primary_icd10", "los_hours"]
    feat_cols = [c for c in thesis.columns if c not in meta_cols]
    
    # Map column index to symptom_id (by order: lab 0-14, vital 15-21)
    col_to_symptom_id = []
    for i, col in enumerate(feat_cols):
        # Column names are 0, 1, 2, ... from pivot
        try:
            idx = int(col)
            if idx < len(lab_itemids):
                col_to_symptom_id.append(idx + 1)
            else:
                col_to_symptom_id.append(idx + 1)  # 16-22 for vitals
        except ValueError:
            col_to_symptom_id.append(mask_id)
    
    # Build token sequences: for each patient, create up to seq_len tokens
    # Each token: [symptom_id, intensity, time_delta_norm, modality]
    # Simplified: we have one value per feature (aggregated). Create tokens from non-missing features.
    n_samples = len(thesis)
    n_dims = 4  # symptom_id, intensity, time_delta, modality
    tokens = np.zeros((n_samples, seq_len, n_dims), dtype=np.float32)
    
    X = thesis[feat_cols].values.astype(np.float32)
    
    for i in range(n_samples):
        row = X[i]
        valid = np.where(~np.isnan(row) & (row >= 0) & (row <= 1))[0]
        if len(valid) > seq_len:
            valid = valid[:seq_len]
        
        for j, idx in enumerate(valid):
            if j >= seq_len:
                break
            sid = col_to_symptom_id[idx] if idx < len(col_to_symptom_id) else (idx + 1)
            intensity = float(row[idx])
            time_delta = 0.0  # aggregated, no temporal order
            modality = 0.0 if idx < len(lab_itemids) else 1.0  # 0=lab, 1=vital
            tokens[i, j, 0] = sid
            tokens[i, j, 1] = intensity
            tokens[i, j, 2] = time_delta
            tokens[i, j, 3] = modality
        
        # Pad with MASK
        for j in range(len(valid), seq_len):
            tokens[i, j, 0] = mask_id
            tokens[i, j, 1] = 0.0
            tokens[i, j, 2] = 0.0
            tokens[i, j, 3] = 0.0
    
    # Metadata
    metadata = thesis[meta_cols].copy()
    
    out_dir = Path(paths["processed_dir"])
    np.save(out_dir / "tokens.npy", tokens)
    metadata.to_parquet(out_dir / "tokens_metadata.parquet", index=False)
    
    print("=" * 60)
    print("TOKENIZATION COMPLETE")
    print("=" * 60)
    print(f"Output: {out_dir / 'tokens.npy'}, {out_dir / 'tokens_metadata.parquet'}")
    print(f"Shape: {tokens.shape}")
    print("=" * 60)

if __name__ == "__main__":
    main()
