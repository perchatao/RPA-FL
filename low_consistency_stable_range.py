"""
stable_range.py
Given baseline thresholds (dv0, dt0), compute the largest stable rectangular
interval [dv_min, dv_max] x [dt_min, dt_max] in which the low-consistency
classification result does not change.

Logic:
  For a LOW class N (has at least one triggering pair (v,t): v>=dv0, t<=dt0):
    - Increasing dv causes witnesses to vanish -> dv_max = max v among all witnesses
    - Decreasing dt causes witnesses to vanish -> dt_min = min t among all witnesses
"""

import numpy as np
import os

BASE = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    "DIOR": {
        "visual_file": "dior/dior-real.txt",
        "text_file":   "bert_matrix/dior-bert.txt",
        "novel": ["airport", "basketball court", "ground track field", "windmill"],
    },
    "DOTA": {
        "visual_file": "dota/dota-real.txt",
        "text_file":   "bert_matrix/dota-bert.txt",
        "novel": ["tennis court", "helicopter", "soccer ball field", "swimming pool"],
    },
}

DV0 = 0.8
DT0 = 0.6


def read_matrix_with_header(filepath):
    with open(filepath, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    categories = [c.strip().lower() for c in lines[0].split("\t") if c.strip()]
    matrix = []
    for line in lines[1:]:
        vals = []
        for tok in line.split():
            try:
                vals.append(float(tok))
            except ValueError:
                pass
        if vals:
            matrix.append(vals)
    return categories, np.array(matrix)


def load_all_pairs():
    """Return all novel-class pair records as a list of dicts:
    [{"dataset", "novel", "partner", "v", "t"}, ...]
    """
    pairs = []
    for ds_name, cfg in DATASETS.items():
        cats, V = read_matrix_with_header(os.path.join(BASE, cfg["visual_file"]))
        _,    T = read_matrix_with_header(os.path.join(BASE, cfg["text_file"]))
        for novel in cfg["novel"]:
            n_idx = cats.index(novel.lower())
            for c_idx, c_name in enumerate(cats):
                if c_idx == n_idx:
                    continue
                pairs.append({
                    "dataset": ds_name,
                    "novel":   novel,
                    "partner": c_name,
                    "v":       float(V[n_idx, c_idx]),
                    "t":       float(T[n_idx, c_idx]),
                })
    return pairs


def classify_novel(pairs, dv, dt):
    """Return {(dataset, novel): bool is_low}"""
    status = {}
    for p in pairs:
        key = (p["dataset"], p["novel"])
        if key not in status:
            status[key] = False
        if p["v"] >= dv and p["t"] <= dt:
            status[key] = True
    return status


def main():
    pairs = load_all_pairs()
    baseline = classify_novel(pairs, DV0, DT0)

    low_keys    = {k for k, v in baseline.items() if v}
    normal_keys = {k for k, v in baseline.items() if not v}

    print(f"Baseline thresholds: dv={DV0}, dt={DT0}")
    print(f"LOW    classes: {[f'{ds}:{cls}' for ds,cls in sorted(low_keys)]}")
    print(f"NORMAL classes: {[f'{ds}:{cls}' for ds,cls in sorted(normal_keys)]}")
    print()

    # ── dv range (dt fixed at DT0) ─────────────────────────────────────────
    # dv_max: keep LOW classes LOW -> take min of max(v) over all witness sets
    dv_max_candidates = []
    for key in low_keys:
        ds, novel = key
        witnesses = [p for p in pairs if p["dataset"]==ds and p["novel"]==novel
                     and p["v"] >= DV0 and p["t"] <= DT0]
        max_v = max(p["v"] for p in witnesses)
        dv_max_candidates.append((max_v, key, witnesses))

    dv_max = min(x[0] for x in dv_max_candidates)
    dv_max_limiting = [(k, w) for v, k, w in dv_max_candidates if abs(v - dv_max) < 1e-9]

    # dv_min: keep NORMAL classes NORMAL -> take max of max(v) over near-miss pairs (text<=DT0, visual<DV0)
    dv_min_candidates = []
    for key in normal_keys:
        ds, novel = key
        near = [p for p in pairs if p["dataset"]==ds and p["novel"]==novel
                and p["t"] <= DT0 and p["v"] < DV0]
        if near:
            max_v = max(p["v"] for p in near)
            dv_min_candidates.append((max_v, key, [p for p in near if abs(p["v"]-max_v)<1e-9]))

    if dv_min_candidates:
        dv_min = max(x[0] for x in dv_min_candidates)
        dv_min_limiting = [(k, w) for v, k, w in dv_min_candidates if abs(v - dv_min) < 1e-9]
    else:
        dv_min = 0.0
        dv_min_limiting = []

    # ── dt range (dv fixed at DV0) ─────────────────────────────────────────
    # dt_min: keep LOW classes LOW -> take max of min(t) over all witness sets
    dt_min_candidates = []
    for key in low_keys:
        ds, novel = key
        witnesses = [p for p in pairs if p["dataset"]==ds and p["novel"]==novel
                     and p["v"] >= DV0 and p["t"] <= DT0]
        min_t = min(p["t"] for p in witnesses)
        dt_min_candidates.append((min_t, key, [p for p in witnesses if abs(p["t"]-min_t)<1e-9]))

    dt_min = max(x[0] for x in dt_min_candidates)
    dt_min_limiting = [(k, w) for v, k, w in dt_min_candidates if abs(v - dt_min) < 1e-9]

    # dt_max: keep NORMAL classes NORMAL -> take min of min(t) over near-miss pairs (visual>=DV0, text>DT0)
    dt_max_candidates = []
    for key in normal_keys:
        ds, novel = key
        near = [p for p in pairs if p["dataset"]==ds and p["novel"]==novel
                and p["v"] >= DV0 and p["t"] > DT0]
        if near:
            min_t = min(p["t"] for p in near)
            dt_max_candidates.append((min_t, key, [p for p in near if abs(p["t"]-min_t)<1e-9]))

    if dt_max_candidates:
        dt_max = min(x[0] for x in dt_max_candidates)
        dt_max_limiting = [(k, w) for v, k, w in dt_max_candidates if abs(v - dt_max) < 1e-9]
    else:
        dt_max = float("inf")
        dt_max_limiting = []

    # ── print results ──────────────────────────────────────────────────────
    print("=" * 60)
    print(f"dv stable interval (dt fixed at {DT0}):")
    print(f"  ({dv_min:.4f}, {dv_max:.4f}]")
    print(f"  dv can be any value in this interval without changing the classification")
    print(f"  Lower bound from NORMAL classes (would trigger if dv drops further):")
    for k, ws in dv_min_limiting:
        for w in ws:
            print(f"    [{k[0]}] {k[1]} <- '{w['partner']}': visual={w['v']:.4f}, text={w['t']:.4f}")
    print(f"  Upper bound from LOW classes (would lose all witnesses if dv rises further):")
    for k, ws in dv_max_limiting:
        for w in ws:
            print(f"    [{k[0]}] {k[1]} <- '{w['partner']}': visual={w['v']:.4f}, text={w['t']:.4f}")

    print()
    print(f"dt stable interval (dv fixed at {DV0}):")
    if dt_max == float("inf"):
        print(f"  [{dt_min:.4f}, +inf)")
    else:
        print(f"  [{dt_min:.4f}, {dt_max:.4f})")
    print(f"  dt can be any value in this interval without changing the classification")
    print(f"  Lower bound from LOW classes (would lose all witnesses if dt drops further):")
    for k, ws in dt_min_limiting:
        for w in ws:
            print(f"    [{k[0]}] {k[1]} <- '{w['partner']}': visual={w['v']:.4f}, text={w['t']:.4f}")
    print(f"  Upper bound from NORMAL classes (would trigger if dt rises further):")
    for k, ws in dt_max_limiting:
        for w in ws:
            print(f"    [{k[0]}] {k[1]} <- '{w['partner']}': visual={w['v']:.4f}, text={w['t']:.4f}")


if __name__ == "__main__":
    main()
