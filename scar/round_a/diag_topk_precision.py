#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose Top-25 identity mismatch between export (float32) and verifier (float64).

Read-only.  Reports, per query, whether the exported ids equal the Top-25 under
float32 scoring and under float64 scoring, whether only order or also the set
differs, and the score gap at the K boundary.
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_report", type=int, default=20)
    args = ap.parse_args()

    z = np.load(os.path.join(args.export_dir, "candidates.npz"), allow_pickle=True)
    cc = np.load(os.path.join(args.export_dir, "coarse_enc.npz"), allow_pickle=True)
    ids = np.asarray(z["cell_ids"])
    tg32 = np.asarray(cc["tg"], np.float32)
    cg32 = np.asarray(cc["cg"], np.float32)
    cg_ids = [str(x) for x in cc["cell_ids"]]
    n, K = ids.shape

    # Export used float32 matmul + default (unstable) argsort on -scores.
    s32 = tg32 @ cg32.T
    top32 = np.argsort(-s32, axis=1)[:, :K]
    ids32 = np.array([[cg_ids[j] for j in r] for r in top32])
    s64 = tg32.astype(np.float64) @ cg32.astype(np.float64).T
    top64 = np.argsort(-s64, axis=1)[:, :K]
    ids64 = np.array([[cg_ids[j] for j in r] for r in top64])

    eq32 = (ids32 == ids).all(axis=1)
    eq64 = (ids64 == ids).all(axis=1)
    set_eq = np.array([set(ids[i]) == set(ids64[i]) for i in range(n)])

    # Boundary gaps: stored 25th score minus best excluded score (float64).
    stored = np.asarray(z["coarse_cos"], np.float64)
    gaps = []
    for i in np.where(~set_eq)[0]:
        row = np.zeros(len(cg_ids), bool)
        row[[cg_ids.index(str(c)) for c in ids[i]]] = True
        gaps.append(float(stored[i, -1] - s64[i][~row].max()))

    rows = []
    for i in np.where(~eq64)[0][:args.max_report]:
        row = np.zeros(len(cg_ids), bool)
        row[[cg_ids.index(str(c)) for c in ids[i]]] = True
        rows.append(dict(
            q=int(i), set_equal=bool(set_eq[i]),
            exported_last_float64_score=float(s64[i][row].min()),
            best_excluded_float64_score=float(s64[i][~row].max()),
            gap_float64=float(s64[i][row].min() - s64[i][~row].max()),
            exported_order_matches_float32=bool(eq32[i])))

    report = {
        "n_queries": int(n),
        "queries_identical_float32": int(eq32.sum()),
        "queries_identical_float64": int(eq64.sum()),
        "queries_set_equal_float64": int(set_eq.sum()),
        "queries_set_differs_float64": int((~set_eq).sum()),
        "boundary_gap_float64_max": float(max(gaps)) if gaps else 0.0,
        "boundary_gap_float64_min": float(min(gaps)) if gaps else 0.0,
        "examples": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "examples"},
                     indent=2))
    print(json.dumps(report["examples"][:5], indent=2))


if __name__ == "__main__":
    main()
