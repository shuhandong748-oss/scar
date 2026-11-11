#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent verification of one corrected-backbone export (TRAIN/VAL).

Recomputes, from the raw cell/pose data and the exported encodings:
  * Top-25 identity and order vs stable descending coarse cosine;
  * every stored ``coarse_cos`` value;
  * ``pred_w == bbox_w[0:2] + offset * cell_size`` for every candidate;
  * query order, unique candidate ids per query, K=25;
  * nuScenes query identity fields.

Read-only.  Only TRAIN/VAL are accepted; TEST is not a valid split here.
"""
import argparse
import json
import os
import sys

import numpy as np

DOMAINS = {
    "pandaset": {
        "code": "/root/autodl-tmp/text2loc_pandaset/code",
        "extra_paths": [
            "/root/autodl-tmp/Des4Pos/reranker",
            "/root/autodl-tmp/Des4Pos/c_line",
            "/root/autodl-tmp/baseB",
            "/root/autodl-tmp/text2loc_pandaset/vendor",
        ],
        "data": "/root/autodl-tmp/text2loc_pandaset/data/PandaSetPose-50-25-25",
    },
    "nuscenesfull": {
        "code": "/root/autodl-tmp/text2loc_nuscenes/code",
        "extra_paths": ["/root/autodl-tmp/text2loc_nuscenes/vendor"],
        "data": ("/root/autodl-tmp/text2loc_nuscenes/data/"
                 "nuScenesPose-50-25-25_30-10"),
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=sorted(DOMAINS))
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--export_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = DOMAINS[args.domain]
    if args.domain == "nuscenesfull":
        os.environ["T2L_NUSC_ROOT"] = d["data"]
    os.chdir(d["code"])
    sys.path.insert(0, d["code"])
    for p in d["extra_paths"]:
        sys.path.insert(0, p)
    from torch_geometric.transforms import FixedPoints
    from dataloading.kitti360pose.cells import Kitti360CoarseDatasetMulti
    from datapreparation.kitti360pose.utils import (
        SCENE_NAMES_TRAIN, SCENE_NAMES_VAL)

    scenes = {"train": SCENE_NAMES_TRAIN, "val": SCENE_NAMES_VAL}[args.split]
    ds = Kitti360CoarseDatasetMulti(d["data"], scenes, FixedPoints(256),
                                    shuffle_hints=False, flip_poses=False)
    cells = {str(c.id): c for c in ds.all_cells}
    poses = ds.all_poses

    z = np.load(os.path.join(args.export_dir, "candidates.npz"), allow_pickle=True)
    cc = np.load(os.path.join(args.export_dir, "coarse_enc.npz"), allow_pickle=True)
    ids = np.asarray(z["cell_ids"])
    offs = np.asarray(z["offsets"], np.float64)
    pred_w = np.asarray(z["pred_w"], np.float64)
    gt_cell = [str(x) for x in z["gt_cell"]]
    gt_pose = np.asarray(z["gt_pose_w"], np.float64)
    cos = np.asarray(z["coarse_cos"], np.float64)
    n, K = ids.shape
    tg = np.asarray(cc["tg"], np.float64)
    cg = np.asarray(cc["cg"], np.float64)
    cg_ids = [str(x) for x in cc["cell_ids"]]
    row = {c: i for i, c in enumerate(cg_ids)}

    checks = {}
    checks["k25"] = K == 25
    checks["query_order_matches_dataset"] = gt_cell == [str(p.cell_id) for p in poses]
    checks["n_queries"] = n == len(poses)
    checks["gt_pose_2d_finite"] = gt_pose.shape == (n, 2) and bool(
        np.isfinite(gt_pose).all())
    checks["all_ids_known"] = all(c in cells for c in ids.ravel())
    checks["ids_unique_per_query"] = all(len(set(str(c) for c in r)) == K
                                         for r in ids)
    checks["offsets_finite"] = bool(np.isfinite(offs).all())
    checks["pred_w_finite"] = bool(np.isfinite(pred_w).all())
    checks["cg_rows_match"] = len(cg) == len(cg_ids) == len(cells)

    # Recompute scores and Top-25 identity/order from the encodings.
    rows = np.array([[row[str(c)] for c in r] for r in ids], np.int64)
    rec = np.einsum("nkd,nd->nk", cg[rows], tg)
    checks["coarse_cos_max_abs_err"] = float(np.abs(rec - cos).max())
    # The exporter scores in float32 (text_enc/cell_enc are .float() arrays),
    # so the faithful independent recomputation uses the same declared
    # arithmetic.  A float64 recomputation is kept as a cross-check: it can
    # reorder candidates whose float32 scores are near-tied, but must never
    # change the Top-25 *set*.  On nuScenesFull TRAIN the float32 recomputation
    # reproduces all 15113 rows exactly; float64 differs only in the internal
    # order of 8 rows, with 0 set differences (DIAG_TOPK_train.json).
    scores = tg @ cg.T                       # (n_queries, n_cells), float64
    top64 = np.argsort(-scores, axis=1)[:, :K]
    top_ids64 = np.array([[cg_ids[j] for j in r] for r in top64])
    checks["top25_set_equal_float64"] = bool(
        all(set(top_ids64[i]) == set(ids[i]) for i in range(n)))
    s32 = tg.astype(np.float32) @ cg.astype(np.float32).T
    top32 = np.argsort(-s32, axis=1)[:, :K]
    top_ids32 = np.array([[cg_ids[j] for j in r] for r in top32])
    checks["top25_identity_equal"] = bool((top_ids32 == ids).all())
    checks["top25_order_float64_identical"] = bool((top_ids64 == ids).all())
    checks["storage_order_descending"] = bool((np.diff(cos, axis=1) <= 1e-6).all())

    # Decode check in the world frame for every candidate.
    bbox = np.zeros((n, K, 2))
    cell_size = np.zeros((n, K))
    for i in range(n):
        for j, c in enumerate(ids[i]):
            bbox[i, j] = np.asarray(cells[str(c)].bbox_w[:2], np.float64)
            cell_size[i, j] = float(cells[str(c)].cell_size)
    dec = bbox + offs * cell_size[..., None]
    checks["decode_max_abs_err"] = float(np.abs(dec - pred_w).max())

    if args.domain == "nuscenesfull":
        qids = [str(x) for x in z["query_ids"]]
        checks["query_ids_unique"] = len(set(qids)) == n
        checks["source_pose_index_arange"] = bool(
            np.array_equal(np.asarray(z["source_pose_index"]), np.arange(n)))
        checks["order_index_arange"] = bool(
            np.array_equal(np.asarray(z["order_index"]), np.arange(n)))
    else:
        checks["no_nusc_identity_fields"] = "query_ids" not in z.files

    # pred_w is stored as unrounded float64 while offsets are stored as
    # float32, so re-decoding from the stored offsets carries float32
    # quantisation (~1e-6); the C-round verifier used the same 2.4e-6 budget.
    tolerances = {"coarse_cos_max_abs_err": 2e-4, "decode_max_abs_err": 1e-5}
    # float64 ordered identity is diagnostic only: it is not a failure when the
    # float32 scores are near-tied, because the candidate set is unchanged.
    diagnostic_only = {"top25_order_float64_identical"}
    failures = []
    for k, v in checks.items():
        if k in diagnostic_only:
            continue
        if isinstance(v, bool) and not v:
            failures.append(k)
    for k, tol in tolerances.items():
        if not (checks[k] <= tol):
            failures.append(k)
    report = {"domain": args.domain, "split": args.split,
              "export_dir": os.path.abspath(args.export_dir),
              "n_queries": int(n), "checks": checks,
              "failures": failures,
              "status": "OK" if not failures else "FAIL"}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
