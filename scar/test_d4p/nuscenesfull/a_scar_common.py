#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared CPU-only helpers for the A-domain FEATURE_COORD_FIX_v1 round."""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

FEATURE_VERSION = "FEATURE_COORD_FIX_v1"
PROTOCOL = "CORR-D4P-SCAR-FORMAL-v1"
KS_CELL = (1, 2, 3, 5, 10, 25)
KS_LOC = (1, 3, 5, 10)
SEED = 123
RESAMPLES = 2000

DOMAINS = {
    "pandaset": {
        "vendor": "/root/autodl-tmp/text2loc_pandaset/vendor",
        # the frozen PandaSet builder resolves features_fix from baseB
        "feature_paths": ["/root/autodl-tmp/baseB"],
        "expected": {"train": 4000, "val": 2000},
        "n_train_cells": 380, "n_val_cells": 199,
        "cluster_unit": "scene",
    },
    "nuscenesfull": {
        "vendor": "/root/autodl-tmp/text2loc_nuscenes/vendor",
        "feature_paths": [],
        "expected": {"train": 15113, "val": 7635},
        "n_train_cells": 3512, "n_val_cells": 1696,
        "cluster_unit": "scene",
    },
}


def insert_feature_paths(domain):
    """Make ``features_fix`` importable exactly as the frozen builders do."""
    d = domain_cfg(domain)
    sys.path.insert(0, d["vendor"])
    for p in reversed(d.get("feature_paths", [])):
        sys.path.insert(0, p)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def domain_cfg(domain):
    if domain not in DOMAINS:
        raise SystemExit(f"unknown domain {domain!r}; have {sorted(DOMAINS)}")
    return dict(DOMAINS[domain])


def rank_orders(logits, p2):
    out = np.empty((len(logits), p2.shape[1]), np.int16)
    for i in range(len(logits)):
        order_in_p2 = np.argsort(-logits[i][p2[i]], kind="stable")
        out[i] = p2[i][order_in_p2]
    return out


def val_metrics(orders, data, K=25):
    """Cell R@k vs gt_cell and localization Top-k@5m with cross-scene inf."""
    ids = data["cell_ids"]
    gt = data["gt_cell"]
    pw = data["pred_w"].astype(np.float64)
    gp = data["gt_pose_w"].astype(np.float64)
    scene = data["scene"].astype(str)
    n = len(ids)
    rk = {k: np.zeros(n, np.int8) for k in KS_CELL}
    loc = {k: np.zeros(n, np.int8) for k in KS_LOC}
    for i in range(n):
        order = orders[i]
        ranked = ids[i][order]
        for k in KS_CELL:
            rk[k][i] = int(gt[i] in ranked[:k])
        d = np.linalg.norm(pw[i][order] - gp[i][None, :], axis=1)
        sc = np.array([str(c).split("_")[0] for c in ranked])
        d[sc != scene[i]] = np.inf
        for k in KS_LOC:
            loc[k][i] = int(np.min(d[:k]) <= 5.0)
    return ({f"R@{k}": float(rk[k].mean()) for k in KS_CELL},
            {f"top{k}@5m": float(loc[k].mean()) for k in KS_LOC}, rk, loc)


def cluster_bootstrap(hit_a, hit_b, keys, resamples=RESAMPLES, seed=SEED):
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    n_cl = len(uniq)
    diff = np.asarray(hit_a, np.float64) - np.asarray(hit_b, np.float64)
    sum_d = np.bincount(inv, weights=diff, minlength=n_cl)
    cnt = np.bincount(inv, minlength=n_cl).astype(np.float64)
    rng = np.random.RandomState(seed)
    vals = np.empty(resamples)
    for s in range(resamples):
        pick = rng.randint(0, n_cl, n_cl)
        vals[s] = sum_d[pick].sum() / cnt[pick].sum()
    return dict(delta=float(diff.mean()),
                ci95=[float(np.percentile(vals, 2.5)),
                      float(np.percentile(vals, 97.5))],
                n_clusters=int(n_cl), n_units=int(len(diff)),
                wins=int((diff > 0).sum()), losses=int((diff < 0).sum()),
                ties=int((diff == 0).sum()))
