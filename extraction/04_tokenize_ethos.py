#!/usr/bin/env python3
"""
ETHOS-aligned tokenizer (faithful to Renc et al., npj Digital Medicine 2024).

The default 04_tokenize.py median-aggregates each variable and stores a normalised
value, discarding the temporal order. ETHOS instead builds a TEMPORAL token
sequence with: (1) a category token per event, (2) a *quantile* (decile, Q1-Q10)
token for its value (quantile edges fit on the TRAINING split only), and (3)
time-interval tokens between consecutive events. This script reproduces that
scheme from the raw first-24h events so the few-shot model is not handicapped by a
lossy representation.

Outputs (into the processed dir):
  tokens_ethos_seq.npy   (N, MAX_LEN) int token-id sequences (padded)
  tokens_ethos_bag.npy   (N, V) bag-of-(category x quantile) counts (model-ready)
  tokens_ethos_vocab.json  token vocabulary + quantile edges
  tokens_ethos_meta.parquet  stay_id / subject_id / hadm_id / feasible aligned to rows

Run where duckdb + pandas are installed (same env as extraction/02_symptoms.py):
  python extraction/04_tokenize_ethos.py
or for the readmission/expanded cohorts:
  THESIS_CONFIG=config_readmit.yaml python extraction/04_tokenize_ethos.py
"""
from __future__ import annotations
import os, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MAX_LEN = 256          # padded sequence length
N_QUANTILES = 10       # deciles, as in ETHOS
SEED = 42
# 13 time-interval bins (minutes), mirroring ETHOS's interval-token idea
INTERVAL_EDGES_MIN = [5, 15, 60, 180, 360, 720, 1440]  # <5m gets no token


def load_config():
    import yaml
    with open(ROOT / os.environ.get("THESIS_CONFIG", "config.yaml")) as f:
        return yaml.safe_load(f)


def resolve(cfg):
    p = {}
    for k, v in cfg["paths"].items():
        q = Path(v); p[k] = q if q.is_absolute() else ROOT / q
    return p


def extract_events(cfg, paths):
    """Raw, time-ordered first-24h lab + vital events (no aggregation)."""
    import duckdb
    cohort = pd.read_parquet(paths["processed_dir"] / "cohort.parquet")
    cohort["intime"] = pd.to_datetime(cohort["intime"])
    cohort["window_end"] = cohort["intime"] + pd.Timedelta(hours=24)
    labs_ids = cfg["symptoms"]["lab_itemids"]; vit_ids = cfg["symptoms"]["vital_itemids"]
    lab_ranges = cfg["symptoms"]["lab_ranges"]; vit_ranges = cfg["symptoms"]["vital_ranges"]
    ct = cohort[["hadm_id", "stay_id", "intime", "window_end"]].drop_duplicates()

    con = duckdb.connect(); con.execute("SET threads TO 4"); con.register("ct", ct)
    labs = con.execute(f"""
        SELECT l.hadm_id, c.stay_id, l.charttime, l.itemid, l.valuenum, c.intime, c.window_end
        FROM read_csv_auto('{paths['hosp_dir']}/labevents.csv.gz', header=true) l
        JOIN ct c ON l.hadm_id=c.hadm_id
        WHERE l.itemid IN ({','.join(map(str,labs_ids))}) AND l.valuenum IS NOT NULL""").fetchdf()
    vit = con.execute(f"""
        SELECT v.hadm_id, c.stay_id, v.charttime, v.itemid, v.valuenum, c.intime, c.window_end
        FROM read_csv_auto('{paths['icu_dir']}/chartevents.csv.gz', header=true) v
        JOIN ct c ON v.stay_id=c.stay_id
        WHERE v.itemid IN ({','.join(map(str,vit_ids))}) AND v.valuenum IS NOT NULL""").fetchdf()
    con.close()

    ev = pd.concat([labs, vit], ignore_index=True)
    for col in ("charttime", "intime", "window_end"):
        ev[col] = pd.to_datetime(ev[col])
    ev["valuenum"] = pd.to_numeric(ev["valuenum"], errors="coerce")
    ev = ev.dropna(subset=["valuenum"])
    ev = ev[(ev.charttime >= ev.intime) & (ev.charttime <= ev.window_end)]
    # clip to clinical ranges
    for itemid, (lo, hi) in {**lab_ranges, **vit_ranges}.items():
        m = ev.itemid == itemid
        ev.loc[m, "valuenum"] = ev.loc[m, "valuenum"].clip(lo, hi)
    ev["minutes"] = (ev.charttime - ev.intime).dt.total_seconds() / 60.0
    return ev.sort_values(["stay_id", "charttime"]).reset_index(drop=True), labs_ids, vit_ids


def main():
    cfg = load_config(); paths = resolve(cfg)
    ev, labs_ids, vit_ids = extract_events(cfg, paths)
    item_order = list(labs_ids) + list(vit_ids)
    cat_of = {itid: i for i, itid in enumerate(item_order)}   # 0..21

    # train split (by stay) for quantile edges — no leakage.
    # NOTE: pandas .to_numpy() can return a read-only view (copy-on-write); copy so
    # that rng.shuffle (in-place) does not raise "array is read-only".
    stays = np.array(ev.stay_id.drop_duplicates().to_numpy(), copy=True)
    rng = np.random.default_rng(SEED); rng.shuffle(stays)
    train_stays = set(stays[: int(0.8 * len(stays))])
    edges = {}
    for itid in item_order:
        vals = ev.loc[(ev.itemid == itid) & (ev.stay_id.isin(train_stays)), "valuenum"].to_numpy()
        edges[itid] = np.quantile(vals, np.linspace(0, 1, N_QUANTILES + 1)) if len(vals) > N_QUANTILES else None

    def quantile_of(itid, v):
        e = edges[itid]
        if e is None:
            return 1
        return int(np.clip(np.searchsorted(e, v, side="right"), 1, N_QUANTILES))

    def interval_token(dmin):
        return int(np.searchsorted(INTERVAL_EDGES_MIN, dmin))  # 0..len(edges)

    # vocab: 0=PAD; category tokens; quantile tokens; interval tokens
    PAD = 0
    cat_base = 1                       # 22 category tokens: 1..22
    q_base = cat_base + len(item_order)        # quantile tokens
    int_base = q_base + N_QUANTILES            # interval tokens
    V = int_base + len(INTERVAL_EDGES_MIN) + 1

    bag_dim = len(item_order) * N_QUANTILES    # (category x quantile) bag

    # --- vectorise category + quantile columns once (no per-row Python work) ---
    ev = ev.copy()
    ev["cat"] = ev["itemid"].map(cat_of).astype("int32")
    qvals = np.ones(len(ev), dtype=np.int32)
    item_arr = ev["itemid"].to_numpy(); val_arr = ev["valuenum"].to_numpy()
    for itid, e in edges.items():
        if e is None:
            continue
        m = item_arr == itid
        qvals[m] = np.clip(np.searchsorted(e, val_arr[m], side="right"), 1, N_QUANTILES)
    ev["q"] = qvals

    stay_ids = ev["stay_id"].drop_duplicates().to_numpy()
    row_of = {sid: r for r, sid in enumerate(stay_ids)}
    seq = np.zeros((len(stay_ids), MAX_LEN), dtype=np.int32)
    bag = np.zeros((len(stay_ids), bag_dim), dtype=np.float32)

    # single pass over events grouped by stay (O(events), not O(stays x events))
    ie0 = INTERVAL_EDGES_MIN[0]; ie_arr = np.asarray(INTERVAL_EDGES_MIN)
    for sid, g in ev.groupby("stay_id", sort=False):
        r = row_of[sid]
        mins = g["minutes"].to_numpy(); cats = g["cat"].to_numpy(); qs = g["q"].to_numpy()
        toks = []; prev = None
        for i in range(len(mins)):
            if prev is not None:
                dt = mins[i] - prev
                if dt >= ie0:
                    toks.append(int_base + int(np.searchsorted(ie_arr, dt)))
            ci = int(cats[i]); qi = int(qs[i])
            toks.append(cat_base + ci); toks.append(q_base + qi - 1)
            bag[r, ci * N_QUANTILES + qi - 1] += 1.0
            prev = mins[i]
        if len(toks) > MAX_LEN:
            toks = toks[:MAX_LEN]
        seq[r, :len(toks)] = toks

    # meta aligned EXACTLY to seq/bag row order (stay_ids)
    hadm_map = ev.drop_duplicates("stay_id").set_index("stay_id")["hadm_id"]
    meta = pd.DataFrame({"stay_id": stay_ids, "hadm_id": hadm_map.loc[stay_ids].to_numpy()})
    try:
        ds = pd.read_parquet(paths["processed_dir"] / "thesis_dataset.parquet")[["stay_id", "subject_id", "feasible"]]
        meta = meta.merge(ds, on="stay_id", how="left")
    except Exception:
        pass

    out = paths["processed_dir"]
    np.save(out / "tokens_ethos_seq.npy", seq)
    np.save(out / "tokens_ethos_bag.npy", bag)
    json.dump({"vocab_size": V, "bag_dim": bag_dim, "n_quantiles": N_QUANTILES,
               "interval_edges_min": INTERVAL_EDGES_MIN, "item_order": item_order},
              open(out / "tokens_ethos_vocab.json", "w"))
    meta.to_parquet(out / "tokens_ethos_meta.parquet", index=False)
    print(f"ETHOS tokens: seq {seq.shape}, bag {bag.shape}, vocab {V}, stays {len(stay_ids):,}")
    print("Wrote tokens_ethos_seq.npy, tokens_ethos_bag.npy, vocab + meta to", out)


if __name__ == "__main__":
    main()
