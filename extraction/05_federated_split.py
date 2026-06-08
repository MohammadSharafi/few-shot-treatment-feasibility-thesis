#!/usr/bin/env python3
"""
Extraction Step 5: Split thesis_dataset by primary ICD-10 into 5 disease nodes
Always output 5 nodes. If node < 50: merge into "other", then split other randomly.
"""
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

def load_config():
    with open(Path(__file__).parent.parent / "config.yaml") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    paths = cfg["paths"]
    proj_root = Path(__file__).parent.parent
    paths = {k: str(proj_root / v) if isinstance(v, str) and not Path(v).is_absolute() else v for k, v in paths.items()}
    out_dir = Path(paths["processed_dir"])
    seed = cfg["seed"]
    
    thesis = pd.read_parquet(out_dir / "thesis_dataset.parquet")
    thesis["icd_norm"] = thesis["primary_icd10"].astype(str).str.replace(".", "").str.upper()
    
    node_to_prefixes = {
        "node_1_sepsis": ["A40", "A41", "R65"],
        "node_2_cardiac": ["I21", "I22", "I50", "I48"],
        "node_3_respiratory": ["J18", "J44", "J96", "J80"],
        "node_4_renal": ["N17", "N18", "N19"],
        "node_5_diabetes": ["E10", "E11", "E13"],
    }
    
    def assign_node(row):
        icd = row["icd_norm"]
        for node_name, prefixes in node_to_prefixes.items():
            for p in prefixes:
                pn = p.replace(".", "")
                if icd.startswith(pn):
                    return node_name
        return "node_0_other"
    
    thesis["node"] = thesis.apply(assign_node, axis=1)
    
    min_per_node = min(50, len(thesis) // 5)
    
    node_dfs = {}
    other_parts = []
    for node_name in ["node_1_sepsis", "node_2_cardiac", "node_3_respiratory", "node_4_renal", "node_5_diabetes"]:
        df = thesis[thesis["node"] == node_name].drop(columns=["node", "icd_norm"])
        if len(df) >= min_per_node:
            node_dfs[node_name] = df
        else:
            other_parts.append(df)
    other_parts.append(thesis[thesis["node"] == "node_0_other"].drop(columns=["node", "icd_norm"]))
    other_df = pd.concat([p for p in other_parts if len(p) > 0], ignore_index=True)
    
    if len(other_df) > 0:
        np.random.seed(seed)
        n_other = len(other_df)
        n_per_slot = max(min_per_node, n_other // 5)
        indices = np.random.permutation(n_other)
        for i, node_name in enumerate(["node_1_sepsis", "node_2_cardiac", "node_3_respiratory", "node_4_renal", "node_5_diabetes"]):
            start = i * n_per_slot
            end = min((i + 1) * n_per_slot, n_other) if i < 4 else n_other
            chunk = other_df.iloc[indices[start:end]]
            if len(chunk) >= min_per_node or len(chunk) > 0:
                if node_name in node_dfs:
                    node_dfs[node_name] = pd.concat([node_dfs[node_name], chunk], ignore_index=True)
                else:
                    node_dfs[node_name] = chunk.reset_index(drop=True)
    
    for node_name in ["node_1_sepsis", "node_2_cardiac", "node_3_respiratory", "node_4_renal", "node_5_diabetes"]:
        if node_name not in node_dfs or len(node_dfs[node_name]) < min_per_node:
            if node_name not in node_dfs:
                node_dfs[node_name] = thesis[thesis["node"] == "node_1_sepsis"].drop(columns=["node", "icd_norm"]).head(min_per_node)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for idx, node_name in enumerate(["node_1_sepsis", "node_2_cardiac", "node_3_respiratory", "node_4_renal", "node_5_diabetes"], 1):
        df = node_dfs.get(node_name)
        if df is not None and len(df) > 0:
            out_path = out_dir / f"node_{idx}.parquet"
            df.to_parquet(out_path, index=False, compression="snappy")
            written.append((f"node_{idx}", len(df), (df["feasible"] == 1).sum(), (df["feasible"] == 0).sum()))
    
    print("=" * 60)
    print("FEDERATED SPLIT COMPLETE")
    print("=" * 60)
    for name, n, pos, neg in written:
        print(f"{name}: {n} patients, feasible={pos}, infeasible={neg}")
    print("=" * 60)

if __name__ == "__main__":
    main()
