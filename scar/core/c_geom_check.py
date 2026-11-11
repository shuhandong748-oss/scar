#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Targeted independent geometry check for the frozen C domains (NPM3D/S3DIS).

This does not retrain, rebuild, or modify anything: it reads the frozen VAL
feature/export files and the corrected-backbone cells, then

  * rebuilds object centres straight from ``obj.xyz`` (never via the stored
    ``build_cell_aux`` result) and compares them with ``build_cell_aux``;
  * checks the cell-frame contract (objects inside [0,1]^2), the world-frame
    identity ``bbox + centre*cell_size == mean(bbox + xyz*cell_size)``;
  * recomputes columns 19/20 (``dist_nearest_matched``,
    ``dist_centroid_matched``) from raw centres + matched object ids and
    compares them with the frozen feature array;
  * verifies the 3.0 truncation and empty-match semantics, and the geometric
    upper bound ``dist <= sqrt(2)`` that holds when centres and offsets are
    both in cell coordinates.

Read-only; writes one JSON report.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

CODE = os.environ.get("C_ROUND_CODE",
                      "/root/autodl-tmp/d4p_trainval_20261104/C/code")
sys.path.insert(0, CODE)
import scar_common as S  # noqa: E402


def raw_centres(cell, max_objs=40):
    out = []
    for obj in cell.objects[:max_objs]:
        xyz = np.asarray(obj.xyz, dtype=np.float64)
        if xyz.shape[0] < 1:
            continue
        out.append(xyz[:, :2].mean(axis=0))
    return np.asarray(out, np.float64) if out else np.zeros((0, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["npm3d", "s3dis"])
    ap.add_argument("--run_dir", required=True,
                    help="domain run dir holding features/ and export/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_cells", type=int, default=60)
    ap.add_argument("--n_queries", type=int, default=0,
                    help="0 = every VAL query")
    a = ap.parse_args()

    d = S.domain_cfg(a.domain)
    dom = S.import_adapter(a.domain)
    dom.apply(d["base"], verbose=False)
    COLS, LABS, NQ = S.domain_vocabulary(a.domain)
    os.chdir(dom.CORE)
    sys.path.insert(0, dom.CORE)
    sys.path.insert(0, d["vendor"])
    from dataloading.kitti360pose.base import Kitti360BaseDataset  # noqa: E402
    from dataloading.kitti360pose.cells import Kitti360CoarseDatasetMulti  # noqa: E402
    from torch_geometric.transforms import FixedPoints  # noqa: E402
    import common as C  # noqa: E402
    import sa_geom as G  # noqa: E402

    splits, _ = dom.split_scene_names(d["base"])
    ds = Kitti360CoarseDatasetMulti(d["base"], list(splits["val"]),
                                    FixedPoints(256),
                                    shuffle_hints=False, flip_poses=False)
    poses = ds.all_poses
    cells_dict = {str(c.id): c for c in ds.all_cells}

    feat_path = os.path.join(a.run_dir, "features", "val.npz")
    fz = np.load(feat_path, allow_pickle=True)
    feats = np.asarray(fz["feats"], np.float64)
    fnames = [str(x) for x in fz["feature_names"]]
    i_near = fnames.index("dist_nearest_matched")
    i_cent = fnames.index("dist_centroid_matched")
    f_cells = [[str(c) for c in row] for row in fz["cell_ids"]]
    f_offs = np.asarray(fz["offsets"], np.float64)
    gt_cell = [str(x) for x in fz["gt_cell"]]
    n = len(f_cells)
    checks = {
        "query_rows_match_dataset": n == len(poses),
        "gt_cell_order_matches_dataset": gt_cell == [str(p.cell_id)
                                                     for p in poses],
        "feats_finite": bool(np.isfinite(feats).all()),
    }

    # ---- cell geometry -----------------------------------------------------
    grid = G.grid_points(33)
    color_of = lambda o: o.get_color_text()
    uniq = sorted({c for row in f_cells for c in row})
    pick = np.linspace(0, len(uniq) - 1, min(a.n_cells, len(uniq))).round()
    sample_cells = [uniq[i] for i in dict.fromkeys(pick.astype(int).tolist())]

    eq_err, world_err = [], []
    in_frac, frame_cells, cm_same = [], [], True
    for cid in sample_cells:
        cell = cells_dict[cid]
        aux = C.build_cell_aux(cell, grid, color_of)
        raw = raw_centres(cell)
        cm = G.build_cell_maps(cell, grid, color_of)
        cm_same = cm_same and (aux["cm"].tokens == cm.tokens) and \
            np.array_equal(aux["cm"].maps, cm.maps)
        if raw.shape != aux["centers"].shape:
            eq_err.append(np.inf)
            continue
        eq_err.append(float(np.abs(raw - aux["centers"].astype(np.float64)).max())
                      if raw.size else 0.0)
        if raw.size == 0:
            continue
        inside = ((raw >= 0.0) & (raw <= 1.0)).all(1)
        in_frac.append(float(inside.mean()))
        all_objs_in = True
        for obj in cell.objects[:40]:
            xyz = np.asarray(obj.xyz, np.float64)
            if xyz.shape[0] < 1:
                continue
            all_objs_in = all_objs_in and float(xyz[:, :2].min()) >= -1e-3 \
                and float(xyz[:, :2].max()) <= 1.0 + 1e-3
        frame_cells.append(bool(all_objs_in))
        bbox = np.asarray(cell.bbox_w[:2], np.float64)
        cs = float(cell.cell_size)
        for obj, c in zip([o for o in cell.objects[:40]
                           if np.asarray(o.xyz).shape[0] >= 1], raw):
            xyz = np.asarray(obj.xyz, np.float64)
            wm = (xyz[:, :2] * cs + bbox[None, :]).mean(axis=0)
            wc = bbox + c * cs
            world_err.append(float(np.linalg.norm(wm - wc)))

    cell_stats = {
        "n_cells": len(sample_cells),
        "max_abs_centers_vs_raw_mean": float(max(eq_err)) if eq_err else 0.0,
        "world_identity_max_err": float(max(world_err)) if world_err else 0.0,
        "raw_centres_inside_unit_frac": float(np.mean(in_frac)) if in_frac else 0.0,
        "cells_all_objects_in_unit_frame": int(sum(frame_cells)),
        "cell_maps_identical": bool(cm_same),
    }
    checks.update({
        "centers_equal_raw_mean": cell_stats["max_abs_centers_vs_raw_mean"]
        < 2e-6,
        "world_identity_holds": cell_stats["world_identity_max_err"] < 1e-4,
        "cell_frame_contract": cell_stats["raw_centres_inside_unit_frac"]
        >= 0.99,
        "cell_maps_identical": bool(cm_same),
    })

    # ---- query-level recomputation of columns 19/20 ------------------------
    idx = np.arange(n)
    if a.n_queries:
        idx = np.unique(np.linspace(0, n - 1,
                                    min(a.n_queries, n)).round().astype(int))
    aux_cache = {}

    def aux_of(cid):
        x = aux_cache.get(cid)
        if x is None:
            x = C.build_cell_aux(cells_dict[cid], grid, color_of)
            aux_cache[cid] = x
        return x

    max_dn = max_dc = 0.0
    n_rows = empty_rows = trunc_rows = at3_near = at3_cent = 0
    empty_ok = trunc_ok = True
    max_obs_distance = 0.0
    for i in idx:
        text = " ".join(Kitti360BaseDataset.create_hint_description(
            poses[i], cells_dict[gt_cell[i]]))
        parsed = G.parse_relations(text, COLS, LABS)
        for r, cid in enumerate(f_cells[i]):
            cell = cells_dict[cid]
            aux = aux_of(cid)
            ids_m = C.matched_object_ids(parsed, aux["cm"])
            raw = raw_centres(cell)
            n_rows += 1
            if ids_m:
                cen = raw[ids_m]
                dn = min(float(np.linalg.norm(cen - f_offs[i, r], axis=1).min()),
                         3.0)
                dc = min(float(np.linalg.norm(cen.mean(0) - f_offs[i, r])), 3.0)
                max_obs_distance = max(max_obs_distance, float(
                    np.linalg.norm(cen - f_offs[i, r], axis=1).max()))
                if dn >= 3.0 - 1e-12 or dc >= 3.0 - 1e-12:
                    trunc_rows += 1
                    trunc_ok = trunc_ok and (
                        abs(feats[i, r, i_near] - 3.0) < 1e-6
                        and abs(feats[i, r, i_cent] - 3.0) < 1e-6)
            else:
                empty_rows += 1
                dn = dc = 3.0
                empty_ok = empty_ok and (
                    abs(feats[i, r, i_near] - 3.0) < 1e-6
                    and abs(feats[i, r, i_cent] - 3.0) < 1e-6)
            max_dn = max(max_dn, abs(feats[i, r, i_near] - dn))
            max_dc = max(max_dc, abs(feats[i, r, i_cent] - dc))
            at3_near += int(abs(feats[i, r, i_near] - 3.0) < 1e-6)
            at3_cent += int(abs(feats[i, r, i_cent] - 3.0) < 1e-6)

    qstats = {
        "n_queries_checked": int(len(idx)),
        "n_candidate_rows": int(n_rows),
        "max_abs_diff_vs_independent_nearest": float(max_dn),
        "max_abs_diff_vs_independent_centroid": float(max_dc),
        "empty_match_rows": int(empty_rows),
        "empty_match_all_3m": bool(empty_ok),
        "truncated_rows": int(trunc_rows),
        "truncated_all_3m": bool(trunc_ok),
        "rows_equal_3m_nearest": int(at3_near),
        "rows_equal_3m_centroid": int(at3_cent),
        "max_matched_object_distance": float(max_obs_distance),
        "distance_bound_sqrt2": float(np.sqrt(2.0)),
    }
    checks.update({
        "independent_recompute_matches": max(max_dn, max_dc) < 2e-5,
        "empty_match_semantics": bool(empty_ok),
        "truncation_semantics": bool(trunc_ok),
        "distances_within_cell_frame_bound":
            qstats["max_matched_object_distance"] <= np.sqrt(2.0) + 1e-3,
        "no_rows_clipped_at_3m": (at3_near == empty_rows and
                                  at3_cent == empty_rows),
    })

    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for ch in iter(lambda: f.read(1 << 20), b""):
                h.update(ch)
        return h.hexdigest()

    checks = {k: bool(v) for k, v in checks.items()}
    report = {
        "purpose": "targeted independent geometry check of frozen C VAL features "
                   "(no retrain, no rebuild, no modification)",
        "domain": a.domain, "split": "val",
        "run_dir": os.path.abspath(a.run_dir),
        "feature_file": feat_path, "feature_sha256": sha(feat_path),
        "common_sha256": sha(os.path.join(d["vendor"], "common.py")),
        "sa_geom_sha256": sha(os.path.join(d["vendor"], "sa_geom.py")),
        "cells": cell_stats, "queries": qstats, "checks": checks,
        "test_read": False,
        "status": "OK" if all(checks.values()) else "FAIL",
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
