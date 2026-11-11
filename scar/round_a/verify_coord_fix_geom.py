"""Real TRAIN/VAL geometry validation for FEATURE_COORD_FIX_v1.

For sampled cells: compares the frozen (defective) aux centres with the
corrected ones, checks the world-frame identity
``bbox + centre*cell_size == mean(bbox + xyz*cell_size)``, object truncation
and empty-xyz alignment.

For sampled queries (when ``--features`` is given): independently recomputes
columns 19/20 straight from the raw objects and compares them to the built
feature array, counts empty-match / truncated rows, and measures how many rows
the fix actually changes.
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


def raw_centres(cell, max_objs=40):
    out = []
    for obj in cell.objects[:max_objs]:
        xyz = np.asarray(obj.xyz, np.float64)
        if xyz.shape[0] < 1:
            continue
        out.append(xyz[:, :2].mean(axis=0))
    return np.asarray(out, np.float64) if out else np.zeros((0, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=sorted(DOMAINS))
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--export_dir", required=True)
    ap.add_argument("--features", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_cells", type=int, default=80)
    ap.add_argument("--n_queries", type=int, default=300)
    args = ap.parse_args()

    d = DOMAINS[args.domain]
    if args.domain == "nuscenesfull":
        os.environ["T2L_NUSC_ROOT"] = d["data"]
    os.chdir(d["code"])
    sys.path.insert(0, d["code"])
    for p in d["extra_paths"]:
        sys.path.insert(0, p)
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)

    import feature_coord_fix_v1 as CF
    import common as C
    import sa_geom as G
    from torch_geometric.transforms import FixedPoints
    from dataloading.kitti360pose.cells import Kitti360CoarseDatasetMulti
    from dataloading.kitti360pose.base import Kitti360BaseDataset
    from datapreparation.kitti360pose.utils import (
        COLOR_NAMES, KNOWN_CLASS,
        SCENE_NAMES_TRAIN, SCENE_NAMES_VAL,
    )

    scenes = {"train": SCENE_NAMES_TRAIN, "val": SCENE_NAMES_VAL}[args.split]
    ds = Kitti360CoarseDatasetMulti(d["data"], scenes, FixedPoints(256),
                                    shuffle_hints=False, flip_poses=False)
    cells_dict = {str(c.id): c for c in ds.all_cells}
    poses = ds.all_poses
    grid = G.grid_points(33)
    color_of = lambda o: o.get_color_text()

    z = np.load(os.path.join(args.export_dir, "candidates.npz"), allow_pickle=True)
    pool = [[str(c) for c in row] for row in z["cell_ids"]]
    unique_cells = sorted({c for row in pool for c in row})
    rng = np.random.default_rng(20261104)
    idx = np.linspace(0, len(unique_cells) - 1,
                      min(args.n_cells, len(unique_cells))).round().astype(int)
    sample_cells = [unique_cells[i] for i in dict.fromkeys(idx.tolist())]

    shifts, old_in, new_in, world_err = [], [], [], []
    n_obj = 0
    cm_equal = True
    count_equal = True
    finite = True
    for cid in sample_cells:
        cell = cells_dict[cid]
        a_old = C.build_cell_aux(cell, grid, color_of)
        a_new = CF.build_cell_aux_v1(cell, grid, color_of, G)
        cm_equal = cm_equal and (a_old["cm"].tokens == a_new["cm"].tokens)
        cm_equal = cm_equal and np.array_equal(a_old["cm"].maps, a_new["cm"].maps)
        count_equal = count_equal and (a_old["counts"] == a_new["counts"])
        count_equal = count_equal and (a_old["n_obj"] == a_new["n_obj"])
        if a_old["centers"].shape != a_new["centers"].shape:
            cm_equal = False
            continue
        if a_new["centers"].size == 0:
            continue
        n_obj += len(a_new["centers"])
        delta = a_old["centers"].astype(np.float64) - a_new["centers"]
        shifts.append(np.linalg.norm(delta, axis=1))
        old_in.append(((a_old["centers"] >= 0) & (a_old["centers"] <= 1)).all(1))
        new_in.append(((a_new["centers"] >= 0) & (a_new["centers"] <= 1)).all(1))
        finite = finite and bool(np.isfinite(a_new["centers"]).all())
        for obj, c in zip([o for o in cell.objects[:40]
                           if np.asarray(o.xyz).shape[0] >= 1],
                          a_new["centers"]):
            xyz = np.asarray(obj.xyz, np.float64)
            wm = (xyz[:, :2] * float(cell.cell_size)
                  + np.asarray(cell.bbox_w[:2])[None, :]).mean(axis=0)
            wc = np.asarray(cell.bbox_w[:2]) + c.astype(np.float64) * float(cell.cell_size)
            world_err.append(float(np.linalg.norm(wm - wc)))

    shifts = np.concatenate(shifts) if shifts else np.zeros(0)
    old_in = np.concatenate(old_in) if old_in else np.zeros(0, bool)
    new_in = np.concatenate(new_in) if new_in else np.zeros(0, bool)
    cell_stats = {
        "n_cells": len(sample_cells),
        "n_objects": int(n_obj),
        "mean_shift_cell_units": float(shifts.mean()) if shifts.size else 0.0,
        "max_shift_cell_units": float(shifts.max()) if shifts.size else 0.0,
        "old_inside_unit_frac": float(old_in.mean()) if old_in.size else 0.0,
        "new_inside_unit_frac": float(new_in.mean()) if new_in.size else 0.0,
        "world_identity_max_err": float(max(world_err)) if world_err else 0.0,
        "cm_and_counts_identical": bool(cm_equal and count_equal),
        "new_centres_finite": bool(finite),
    }

    qstats = {}
    checks = {
        "cm_counts_identical": cell_stats["cm_and_counts_identical"],
        "new_centres_finite": cell_stats["new_centres_finite"],
        "world_identity_tiny": cell_stats["world_identity_max_err"] < 1e-4,
        "old_differs_from_new": cell_stats["max_shift_cell_units"] > 1e-6,
    }

    if args.features:
        fz = np.load(args.features, allow_pickle=True)
        feats = np.asarray(fz["feats"], np.float64)
        fnames = [str(x) for x in fz["feature_names"]]
        i_near = fnames.index("dist_nearest_matched")
        i_cent = fnames.index("dist_centroid_matched")
        f_cells = [[str(c) for c in row] for row in fz["cell_ids"]]
        offs = np.asarray(fz["offsets"], np.float64)
        n = len(f_cells)
        if n != len(poses):
            raise SystemExit(f"feature rows {n} != poses {len(poses)}")
        rows = np.unique(np.linspace(0, n - 1, min(n, args.n_queries)).round()
                         .astype(int))
        aux_old, aux_new = {}, {}

        def aux(cid, fixed):
            store = aux_new if fixed else aux_old
            if cid not in store:
                if fixed:
                    store[cid] = CF.build_cell_aux_v1(cells_dict[cid], grid,
                                                      color_of, G)
                else:
                    store[cid] = C.build_cell_aux(cells_dict[cid], grid, color_of)
            return store[cid]

        max_dn = max_dc = 0.0
        old_new_near = old_new_cent = 0
        n_cand = empty_rows = trunc_rows = 0
        empty_ok = trunc_ok = True
        indep_err = []
        for i in rows:
            text = " ".join(Kitti360BaseDataset.create_hint_description(poses[i], None))
            parsed = G.parse_relations(text, COLOR_NAMES, KNOWN_CLASS)
            n_q = max(1, len(parsed))
            for r, cid in enumerate(f_cells[i]):
                a_old, a_new = aux(cid, False), aux(cid, True)
                ids_m = C.matched_object_ids(parsed, a_new["cm"])
                raw = raw_centres(cells_dict[cid])
                n_cand += 1
                if ids_m:
                    cen = raw[ids_m]
                    dn = min(float(np.linalg.norm(cen - offs[i, r], axis=1).min()), 3.0)
                    dc = min(float(np.linalg.norm(cen.mean(0) - offs[i, r])), 3.0)
                    old = a_old["centers"][ids_m]
                    on = min(float(np.linalg.norm(old - offs[i, r], axis=1).min()), 3.0)
                    oc = min(float(np.linalg.norm(old.mean(0) - offs[i, r])), 3.0)
                    if dn > 3.0 - 1e-12 or dc > 3.0 - 1e-12:
                        trunc_rows += 1
                        trunc_ok = trunc_ok and (
                            abs(feats[i, r, i_near] - 3.0) < 1e-6
                            and abs(feats[i, r, i_cent] - 3.0) < 1e-6)
                    else:
                        indep_err.append(max(abs(feats[i, r, i_near] - dn),
                                             abs(feats[i, r, i_cent] - dc)))
                    old_new_near += int(abs(on - feats[i, r, i_near]) > 1e-5)
                    old_new_cent += int(abs(oc - feats[i, r, i_cent]) > 1e-5)
                    max_dn = max(max_dn, abs(feats[i, r, i_near] - dn))
                    max_dc = max(max_dc, abs(feats[i, r, i_cent] - dc))
                else:
                    empty_rows += 1
                    empty_ok = empty_ok and (
                        abs(feats[i, r, i_near] - 3.0) < 1e-6
                        and abs(feats[i, r, i_cent] - 3.0) < 1e-6)
        qstats = {
            "n_queries_sampled": int(len(rows)),
            "n_candidate_rows": int(n_cand),
            "max_abs_diff_vs_independent_near": float(max_dn),
            "max_abs_diff_vs_independent_cent": float(max_dc),
            "old_version_rows_changed_near": int(old_new_near),
            "old_version_rows_changed_cent": int(old_new_cent),
            "empty_match_rows": int(empty_rows),
            "empty_match_all_3m": bool(empty_ok),
            "truncated_rows": int(trunc_rows),
            "truncated_all_3m": bool(trunc_ok),
        }
        checks.update({
            "independent_recompute_matches": max(max_dn, max_dc) < 2e-5,
            "empty_match_semantics": bool(empty_ok),
            "truncation_semantics": bool(trunc_ok),
        })

    checks = {k: bool(v) for k, v in checks.items()}
    report = {"feature_version": CF.FEATURE_VERSION, "domain": args.domain,
              "split": args.split, "cells": cell_stats, "queries": qstats,
              "checks": checks, "ok": bool(all(checks.values()))}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
