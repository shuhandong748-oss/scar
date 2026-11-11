# -*- coding: utf-8 -*-
"""TEST13: second, independent implementation of the TEST readouts.

Nothing here imports test13_eval or test13_features.  Everything is rebuilt
from the exported artefacts:

  cache/chunks/*.npz        candidate idx/scores, Fine offsets, pred_w, poses
  cache/cell_encodings.npz  gallery descriptors
  features/test_features.npz  only the relation-derived columns (sp, sc_fine,
                            sat_fine, p2_idx) and the label inputs
  heads/scar23_s0.pt        frozen scorer + TRAIN-fitted normalisation

Checks
------
1. cos is recomputed from the descriptors and matched against the export
   (candidate_score) and against the features column.
2. The Top-25 candidate list is recomputed from the full score matrix
   (argsort(-scores)[:25]) and matched against the export.
3. The presence order is recomputed from (cos + 0.5*sp) instead of trusting
   the stored p0_idx.
4. FSR / Rank Fusion / SCAR orders are recomputed and matched to the eval
   per-query record.
5. Distances are recomputed from pred_w and the GT poses in the chunk files,
   metrics are rebuilt, and every number is matched to metrics_test.json.
6. Three paired pose-cluster bootstraps are re-run with the same seed and
   matched to the reported intervals.

Writes out/eval/VERIFY_INDEPENDENT.json and exits non-zero on any mismatch.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test13_common as C  # noqa: E402

KS_CELL = (1, 2, 3, 5, 10, 25)
KS_LOC = (1, 2, 5, 10, 25)
THRESHS = (5.0, 10.0, 15.0)
MAIN = ("original", "presence", "c_borda", "sat_fine", "SCAR")


def hits(orders_row, ids, drow, gt):
    top_cells = [ids[j] for j in orders_row]
    hc = {k: int(gt in set(top_cells[:k])) for k in KS_CELL}
    d = drow[orders_row]
    hl = {k: {t: int(float(np.min(d[:k])) <= t) for t in THRESHS} for k in KS_LOC}
    return hc, hl


def bootstrap(diff, keys, seeds=2000):
    uniq, inv = np.unique(np.round(keys, 6), axis=0, return_inverse=True)
    n_cl = len(uniq)
    sum_d = np.bincount(inv, weights=diff, minlength=n_cl)
    cnt = np.bincount(inv, minlength=n_cl).astype(np.float64)
    rng = np.random.RandomState(123)
    vals = np.empty(seeds)
    for s in range(seeds):
        pick = rng.randint(0, n_cl, n_cl)
        vals[s] = sum_d[pick].sum() / cnt[pick].sum()
    return float(diff.mean()), [float(np.percentile(vals, 2.5)),
                                float(np.percentile(vals, 97.5))]


def main():
    root = os.path.join(C.ROUND, "out")
    cache = os.path.join(root, "cache")
    meta = C.read_json(os.path.join(cache, "test_export_meta.json"))
    prog = C.read_json(os.path.join(root, "state", "progress.json"))
    ev = C.read_json(os.path.join(root, "eval", "metrics_test.json"))
    pq = [json.loads(l) for l in open(os.path.join(root, "eval", "per_query_test.jsonl"),
                                      encoding="utf-8")]
    zf = np.load(os.path.join(root, "features", "test_features.npz"), allow_pickle=True)

    cells_npz = np.load(os.path.join(cache, "cells.npz"), allow_pickle=False)
    gallery = [str(x) for x in cells_npz["cell_ids"]]
    cell_scene = [str(x) for x in cells_npz["cell_scene"]]
    cell_enc = np.load(os.path.join(cache, "cell_encodings.npz"),
                       allow_pickle=False)["cell_enc"]
    gidx = {c: i for i, c in enumerate(gallery)}

    # ---------------------------------------------------------------- chunks
    cand, scores, pred_w, pose_w, text_enc = [], [], [], [], []
    for rec in prog["completed"]:
        z = np.load(os.path.join(root, "chunks", rec["npz"]), allow_pickle=False)
        cand.append(z["candidate_idx"])
        scores.append(z["candidate_score"])
        pred_w.append(z["pred_w"])
        pose_w.append(z["pose_w"])
        text_enc.append(z["text_enc"])
    cand = np.concatenate(cand, 0)
    scores = np.concatenate(scores, 0)
    pred_w = np.concatenate(pred_w, 0)
    pose_w = np.concatenate(pose_w, 0)
    text_enc = np.concatenate(text_enc, 0)
    n, K = cand.shape
    rep = {"n_queries": int(n), "k": int(K), "n_cells": len(gallery),
           "checks": {}, "max_diffs": {}}
    fails = []

    def check(name, ok, detail=None):
        rep["checks"][name] = bool(ok)
        if detail is not None:
            rep["max_diffs"][name] = detail
        if not ok:
            fails.append(name)

    # 1/2. scores and candidates recomputed from the descriptors
    cos32 = np.zeros((n, K), np.float32)   # float32 matmul, as the export scored
    cos64 = np.zeros((n, K), np.float64)   # float64 dot per candidate, as the
                                           # feature builder recomputed cos_v
    top_re = np.zeros((n, K), np.int32)
    max_cos = 0.0
    cand_mismatch = 0
    B = 500
    for b in range(0, n, B):
        e = min(n, b + B)
        full = text_enc[b:e] @ cell_enc.T                       # float32, as exported
        top_re[b:e] = np.argsort(-full, axis=1)[:, :K]
        rows = np.arange(e - b)[:, None]
        cos32[b:e] = full[rows, cand[b:e]]
        max_cos = max(max_cos, float(np.max(np.abs(
            cos32[b:e].astype(np.float64) - scores[b:e].astype(np.float64)))))
        cand_mismatch += int((top_re[b:e] != cand[b:e]).sum())
        del full
        cf = cell_enc[cand[b:e]].astype(np.float64)
        te = text_enc[b:e].astype(np.float64)
        cos64[b:e] = np.einsum("ikj,ij->ik", cf, te)
        del cf, te
    check("candidate_top25_recomputed", cand_mismatch == 0, {"mismatched_cells": cand_mismatch})
    check("cos_matches_export", max_cos <= 2e-5, {"max_abs_diff": max_cos})

    feat_cos = zf["cos"].astype(np.float64)
    max_cos_feat = float(np.max(np.abs(feat_cos - cos64)))
    check("cos_matches_features", max_cos_feat <= 1e-6, {"max_abs_diff": max_cos_feat})

    cids = zf["cids"].astype(str)
    cids_re = np.array([[gallery[j] for j in row] for row in cand], dtype=cids.dtype)
    check("feature_ids_match_export", bool((cids == cids_re).all()))
    check("gt_pose_matches_export",
          float(np.max(np.abs(zf["gt_pose_w"] - pose_w))) <= 1e-9,
          {"max_abs_diff": float(np.max(np.abs(zf["gt_pose_w"] - pose_w)))})
    check("pred_w_matches_export",
          float(np.max(np.abs(zf["pred_w"] - pred_w))) <= 1e-9,
          {"max_abs_diff": float(np.max(np.abs(zf["pred_w"] - pred_w)))})

    # 3. presence order from cos + 0.5*sp
    sp = zf["sp"].astype(np.float64)
    p0_re = np.argsort(-(cos64 + C.LAMBDA * sp), axis=1, kind="stable")
    check("presence_order_recomputed",
          bool((p0_re == zf["p0_idx"]).all()),
          {"mismatched": int((p0_re != zf["p0_idx"]).sum())})

    # 4. FSR / Rank Fusion / SCAR
    p2 = zf["p2_idx"].astype(np.int64)
    sat = zf["sat_fine"].astype(np.float64)
    sc_fine = zf["sc_fine"].astype(np.float64)
    o_sat = np.stack([p2[i][np.argsort(-sat[i, p2[i]], kind="stable")] for i in range(n)])
    cb = np.zeros((n, K), np.int64)
    for i in range(n):
        p = p2[i]
        fo = np.argsort(-sc_fine[i, p], kind="stable")
        fr = np.empty(K, np.int64)
        fr[fo] = np.arange(K)
        borda = (np.arange(K, dtype=np.float64) + fr) / (2.0 * (K - 1))
        cb[i] = p[np.argsort(borda, kind="stable")]
    import torch
    ck = torch.load(C.SCAR_S0, map_location="cpu", weights_only=False)
    zz = zf["feats"].astype(np.float64)[..., ck["keep"]]
    zz = (zz - ck["mean"]) / ck["std"]
    zz[..., ck["pinned"]] = 0.0
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self, nf):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(nf, 64), nn.ReLU(),
                                     nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = Net(len(ck["feature_names"]))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    with torch.no_grad():
        lg = model(torch.from_numpy(zz.astype(np.float32))).numpy()
    scar_re = np.stack([p2[i][np.argsort(-lg[i, p2[i]], kind="stable")] for i in range(n)])

    name_to_order = {"original": np.tile(np.arange(K), (n, 1)),
                     "presence": p0_re, "c_borda": cb,
                     "sat_fine": o_sat, "SCAR": scar_re}
    for mm, o in name_to_order.items():
        same = sum(1 for i in range(n)
                   if [str(cids[i, j]) for j in o[i]] == pq[i]["order"][mm])
        check(f"order_{mm}_matches_eval", same == n, {"mismatched_queries": n - same})

    # 5. metrics rebuilt from the chunk files
    gt_cell = [str(x) for x in zf["gt_cell"]]
    scene = [str(x) for x in zf["scene"]]
    dist = np.linalg.norm(pred_w - pose_w[:, None, :2], axis=2)
    res_cell = {mm: {k: [] for k in KS_CELL} for mm in MAIN}
    res_loc = {mm: {(k, t): [] for k in KS_LOC for t in THRESHS} for mm in MAIN}
    mask_cnt = 0
    for i in range(n):
        ids = cids[i]
        drow = dist[i].copy()
        for j in range(K):
            if cell_scene[gidx[ids[j]]] != scene[i]:
                drow[j] = np.inf
                mask_cnt += 1
        for mm in MAIN:
            hc, hl = hits(name_to_order[mm][i], ids, drow, gt_cell[i])
            for k in KS_CELL:
                res_cell[mm][k].append(hc[k])
            for k in KS_LOC:
                for t in THRESHS:
                    res_loc[mm][(k, t)].append(hl[k][t])
    max_metric_diff = 0.0
    for mm in MAIN:
        for k in KS_CELL:
            got = float(np.mean(res_cell[mm][k]))
            want = ev["metrics"][mm]["cell"][f"R@{k}"]
            max_metric_diff = max(max_metric_diff, abs(got - want))
        for k in KS_LOC:
            for t in THRESHS:
                got = float(np.mean(res_loc[mm][(k, t)]))
                want = ev["metrics"][mm]["loc"][f"top{k}@{int(t)}m"]
                max_metric_diff = max(max_metric_diff, abs(got - want))
    check("metrics_rebuilt", max_metric_diff < 1e-12,
          {"max_abs_diff": max_metric_diff, "cross_scene_masked_positions": mask_cnt})

    # 6. three paired bootstraps re-run with the registered seed
    boot = {}
    keys = pose_w[:, :2]
    for a_m, b_m, metric in (("SCAR", "sat_fine", "top1_5m"),
                             ("SCAR", "sat_fine", "cell_R1"),
                             ("presence", "original", "top1_5m"),
                             ("c_borda", "sat_fine", "cell_R1")):
        if metric == "top1_5m":
            ha = np.array(res_loc[a_m][(1, 5.0)], np.float64)
            hb = np.array(res_loc[b_m][(1, 5.0)], np.float64)
        else:
            ha = np.array(res_cell[a_m][1], np.float64)
            hb = np.array(res_cell[b_m][1], np.float64)
        d, ci = bootstrap(ha - hb, keys)
        ref = ev["comparisons"][a_m][f"vs_{b_m}"][metric]
        ok = abs(d - ref["delta"]) < 1e-12 and abs(ci[0] - ref["ci95"][0]) < 1e-12 \
            and abs(ci[1] - ref["ci95"][1]) < 1e-12
        boot[f"{a_m}_vs_{b_m}_{metric}"] = {"delta": d, "ci95": ci,
                                            "reference": ref,
                                            "matches": bool(ok)}
        check(f"bootstrap_{a_m}_vs_{b_m}_{metric}", ok)

    rep["status"] = "PASS" if not fails else "FAIL"
    rep["failures"] = fails
    rep["source"] = "rebuilt from out/cache/chunks + out/cache/cell_encodings.npz " \
                    "+ out/features/test_features.npz + the frozen head; " \
                    "test13_eval.py was not imported"
    rep["bootstrap_reproduced"] = boot
    rep["exploratory"] = True
    rep["created_at"] = C.now()
    out_p = os.path.join(root, "eval", "VERIFY_INDEPENDENT.json")
    C.write_json_atomic(out_p, rep)
    print(json.dumps({"status": rep["status"], "failures": fails,
                      "checks": rep["checks"], "max_diffs": rep["max_diffs"]},
                     indent=1))
    return 0 if rep["status"] == "PASS" else 3


if __name__ == "__main__":
    sys.exit(main())
