#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Five-order, s0-only TEST evaluation for a C domain.

Methods share this round's export cache and Fine coordinates:
  original, presence, rank_fusion, fsr, scar_s0.
s1/s2 are not loaded or evaluated.  Primary contrast is SCAR - FSR on
Cell R@1 and Top-1@5m; all other contrasts/thresholds are reported as
secondary/exploratory point estimates without new interval tests.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c_test_common as T  # noqa: E402

METHOD_LABELS = {
    "original": "original", "presence": "Presence",
    "rank_fusion": "Rank Fusion", "fsr": "FSR", "scar_s0": "SCAR s0",
}


def load_head(path, expected_sha, feature_names):
    import torch
    import torch.nn as nn
    if T.sha256(path) != expected_sha:
        raise SystemExit(f"head SHA mismatch: {path}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if list(ck["feature_names"]) != list(feature_names):
        raise SystemExit("head feature names mismatch")
    if len(ck["feature_names"]) != 23:
        raise SystemExit("head is not canonical-23")
    if ck.get("config", {}).get("k") != 25:
        raise SystemExit("head is not K25")
    if ck.get("config", {}).get("norms") != dict(
            pool_norm=24, rank_norm=24, tied_norm=25):
        raise SystemExit("head norms mismatch")
    if not isinstance(ck.get("mean"), np.ndarray) \
            or not isinstance(ck.get("std"), np.ndarray):
        raise SystemExit("head normalisation missing")
    mean = np.asarray(ck["mean"], np.float32)
    std = np.asarray(ck["std"], np.float32)
    if mean.shape != (23,) or std.shape != (23,) or np.any(std == 0):
        raise SystemExit("head normalisation shape/zero mismatch")

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(23, 64), nn.ReLU(),
                                     nn.Linear(64, 32), nn.ReLU(),
                                     nn.Linear(32, 1))

        def forward(self, z):
            return self.net(z).squeeze(-1)

    model = Net()
    model.load_state_dict(ck["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def logits(x):
        z = (np.asarray(x, np.float32) - mean) / std
        with torch.no_grad():
            return model(torch.from_numpy(z)).numpy()

    return logits, ck


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=sorted(T.DOMAINS))
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--export_dir", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--freeze_list", required=True)
    ap.add_argument("--protocol_sha", required=True)
    ap.add_argument("--freeze_sha", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--marker", default=None)
    a = ap.parse_args(argv)
    for name in ("export_dir", "features", "out_dir", "freeze_list"):
        setattr(a, name, os.path.abspath(getattr(a, name)))
    S = T.frozen_common()
    info = T.domain_info(a.domain)
    freeze, finfo = T.freeze_info(a.freeze_list, a.domain)
    if a.protocol_sha != finfo["protocol_sha256"]:
        raise SystemExit("protocol SHA mismatch with freeze list")
    if a.freeze_sha and a.freeze_sha != finfo["freeze_sha256"]:
        raise SystemExit("freeze SHA mismatch")
    if freeze["main_head"]["tag"] != "s0":
        raise SystemExit("main head is not s0")

    marker = None
    marker_path = a.marker or os.path.join(a.export_dir, "TEST_READ_ONCE.json")
    if a.split == "test":
        if not os.path.isfile(marker_path):
            raise SystemExit("TEST marker missing")
        marker = T.read_json(marker_path)
        if marker.get("status") != "completed" or marker.get("armed") is not True:
            raise SystemExit("TEST marker not completed/armed")
        if marker.get("protocol_freeze_sha256") != a.protocol_sha:
            raise SystemExit("TEST marker protocol mismatch")
        if marker.get("freeze_list_sha256") != finfo["freeze_sha256"]:
            raise SystemExit("TEST marker freeze mismatch")

    em_path = os.path.join(a.export_dir, "EXPORT_META.json")
    em = T.read_json(em_path)
    if em.get("domain") != a.domain or em.get("split") != a.split:
        raise SystemExit("export meta domain/split mismatch")
    if em.get("k") != 25 or em.get("complete") is not True:
        raise SystemExit("export is not a complete K=25 cache")
    for f, h in (em.get("cache_sha256") or {}).items():
        p = os.path.join(a.export_dir, f)
        if not os.path.isfile(p) or T.sha256(p) != h:
            raise SystemExit(f"export cache SHA mismatch: {f}")
    if marker and marker.get("cache_sha256", {}).get("candidates.npz") != \
            em["cache_sha256"]["candidates.npz"]:
        raise SystemExit("TEST marker does not bind the export cache")

    fm_path = a.features + ".meta.json"
    if not os.path.isfile(fm_path):
        raise SystemExit("feature meta missing")
    fm = T.read_json(fm_path)
    if T.sha256(a.features) != fm.get("feature_file_sha256"):
        raise SystemExit("feature file SHA mismatch")
    if fm.get("domain") != a.domain or fm.get("split") != a.split:
        raise SystemExit("feature meta domain/split mismatch")
    if T.sha256(em_path) != fm.get("export_meta_sha256"):
        raise SystemExit("feature meta export SHA mismatch")
    if fm.get("test_opened") != (a.split == "test"):
        raise SystemExit("feature test_opened mismatch")

    z = np.load(os.path.join(a.export_dir, "candidates.npz"), allow_pickle=True)
    ids = np.asarray([[str(c) for c in row] for row in z["cell_ids"]])
    cscene = np.asarray([[str(c) for c in row] for row in z["cell_scene"]])
    pred_w = np.asarray(z["pred_w"], np.float64)
    gt_w = np.asarray(z["gt_pose_w"], np.float64)
    gt_cell = [str(x) for x in z["gt_cell"]]
    scene = np.asarray([str(x) for x in z["scene"]])
    qids = np.asarray([str(x) for x in z["query_ids"]])
    n_all, K = ids.shape
    if K != 25:
        raise SystemExit("export is not K=25")
    if a.limit:
        n = min(n_all, a.limit)
    else:
        n = n_all
    expected_n = info["expected_test"] if a.split == "test" else info["expected_val"]
    expected_c = (info["expected_test_clusters"] if a.split == "test"
                  else info["expected_val_clusters"])
    if not a.limit:
        if n != expected_n:
            raise SystemExit(f"export has {n} queries != expected {expected_n}")
        if len(set(scene.tolist())) != expected_c:
            raise SystemExit(f"export has {len(set(scene.tolist()))} clusters "
                             f"!= expected {expected_c}")
    if len(set(qids.tolist())) != n_all:
        raise SystemExit("query ids are not unique")
    if not all(len(set(row)) == K for row in ids.tolist()):
        raise SystemExit("candidate ids are not unique per query")

    feat = np.load(a.features, allow_pickle=True)
    feats = np.asarray(feat["feats"], np.float32)
    fids = np.asarray([[str(c) for c in row] for row in feat["cell_ids"]])
    fgt = [str(x) for x in feat["gt_cell"]]
    p2 = np.asarray(feat["p2_idx"], np.int64)
    cos = np.asarray(feat["cos"], np.float64)
    sp = np.asarray(feat["sp"], np.float64)
    sc_fine = np.asarray(feat["sc_fine"], np.float64)
    sat = np.asarray(feat["sat_fine"], np.float64)
    fnames = [str(x) for x in feat["feature_names"]]
    valid = (np.asarray(feat["valid"]).astype(bool)
             if "valid" in feat.files else None)
    if len(feats) != n:
        raise SystemExit(f"features {len(feats)} != evaluated queries {n}")
    if not np.array_equal(fids, ids[:n]):
        raise SystemExit("feature candidate order differs from export")
    if fgt != gt_cell[:n]:
        raise SystemExit("feature gt order differs from export")
    if len(fnames) != 23 or list(fnames) != list(freeze["features"]["constants"]
                                                .get("feature_names", fnames)):
        if len(fnames) != 23:
            raise SystemExit("feature names are not canonical 23")
    if valid is not None and valid.shape[0] != n:
        raise SystemExit("feature valid length mismatch")
    for i in range(n):
        if sorted(p2[i].tolist()) != list(range(K)):
            raise SystemExit(f"p2 is not a permutation at query {i}")
        if not np.array_equal(np.argsort(-cos[i], kind="stable"),
                              np.arange(K)):
            raise SystemExit(f"storage order is not descending cosine at query {i}")
    if not np.all(np.isfinite(feats)):
        raise SystemExit("features contain non-finite values")

    head_path = freeze["main_head"]["path"]
    head_logits, head_ck = load_head(head_path, freeze["main_head"]["sha256"],
                                     fnames)
    os.makedirs(a.out_dir, exist_ok=False)
    if os.listdir(a.out_dir):
        raise SystemExit("out_dir was not empty")
    cluster_map = T.write_cluster_mapping(
        os.path.join(a.out_dir, "cluster_mapping.csv"),
        qids[:n].tolist(), scene[:n].tolist(), info["cluster_unit"])

    orders = np.empty((n, K, len(T.METHODS)), np.int64)
    dist_all = np.zeros((n, K), np.float64)
    dist_ordered = np.zeros((n, K, len(T.METHODS)), np.float64)
    hits_cell = {m: {k: np.zeros(n, np.int8) for k in T.KS_CELL}
                 for m in T.METHODS}
    hits_loc = {m: {(k, t): np.zeros(n, np.int8)
                    for k in T.KS_LOC for t in T.THRESHS}
                for m in T.METHODS}
    rows = []
    for i in range(n):
        lg = head_logits(feats[i][None, :, :])[0]
        if not np.all(np.isfinite(lg)):
            raise SystemExit(f"non-finite head logits at query {i}")
        cur = T.five_orders(feats[i], p2[i], cos[i], sp[i], sc_fine[i],
                            sat[i], lg)
        for mi, m in enumerate(T.METHODS):
            o = cur[m]
            if sorted(o.tolist()) != list(range(K)):
                raise SystemExit(f"{m} order is not a permutation at query {i}")
            orders[i, :, mi] = o
        d0 = np.linalg.norm(pred_w[i] - gt_w[i][None, :], axis=1)
        d0[cscene[i] != scene[i]] = np.inf
        dist_all[i] = d0
        for mi, m in enumerate(T.METHODS):
            o = cur[m]
            ranked = [str(ids[i][j]) for j in o]
            for k in T.KS_CELL:
                hits_cell[m][k][i] = int(gt_cell[i] in ranked[:k])
            d = np.linalg.norm(pred_w[i][o] - gt_w[i][None, :], axis=1)
            d[np.asarray([cscene[i][j] for j in o]) != scene[i]] = np.inf
            dist_ordered[i, :, mi] = d
            for k in T.KS_LOC:
                for t in T.THRESHS:
                    hits_loc[m][(k, t)][i] = int(np.min(d[:k]) <= t)
        if len({int(hits_cell[m][25][i]) for m in T.METHODS}) != 1:
            raise SystemExit(f"R@25 differs across methods at query {i}")
        if len({int(hits_loc[m][(25, 5.0)][i]) for m in T.METHODS}) != 1:
            raise SystemExit(f"Top25@5m differs across methods at query {i}")
        rows.append(dict(
            q=i, query_id=str(qids[i]), scene=str(scene[i]),
            gt_cell=str(gt_cell[i]), ids=[str(x) for x in ids[i]],
            orders={m: [int(x) for x in cur[m]] for m in T.METHODS},
            logits=[float(x) for x in lg],
            distances=[float(x) for x in d0],
            hit_cell={m: {f"R{k}": int(hits_cell[m][k][i]) for k in T.KS_CELL}
                      for m in T.METHODS},
            hit_loc={m: {f"Top{k}@{int(t)}m": int(hits_loc[m][(k, t)][i])
                         for k in T.KS_LOC for t in T.THRESHS}
                     for m in T.METHODS},
        ))

    res_cell = {m: {f"R@{k}": float(hits_cell[m][k].mean()) for k in T.KS_CELL}
                for m in T.METHODS}
    res_loc = {m: {f"Top{k}@{int(t)}m": float(hits_loc[m][(k, t)].mean())
                   for k in T.KS_LOC for t in T.THRESHS} for m in T.METHODS}
    primary_metrics = [("Cell_R@1", hits_cell, 1), ("Top1@5m", hits_loc, (1, 5.0))]
    all_metrics = ([(f"Cell_R@{k}", hits_cell, k) for k in T.KS_CELL]
                   + [(f"Top{k}@{int(t)}m", hits_loc, (k, t))
                      for k in T.KS_LOC for t in T.THRESHS])
    keys = scene[:n].astype("<U64").reshape(-1, 1)
    comps = {}
    for baseline in ("fsr", "original", "presence", "rank_fusion"):
        role = "PRIMARY" if baseline == "fsr" else "SECONDARY_EXPLORATORY"
        c = {}
        for name, arr, sel in all_metrics:
            lo = arr["scar_s0"][sel]
            lb = arr[baseline][sel]
            if baseline == "fsr" and name in {x[0] for x in primary_metrics}:
                c[name] = T.cluster_bootstrap(lo, lb, keys)
                c[name]["interval_role"] = "PRIMARY"
            else:
                c[name] = dict(delta=float(lo.mean() - lb.mean()),
                               note="point estimate only; secondary/exploratory")
        comps[f"SCAR_minus_{baseline}"] = dict(
            target="scar_s0", baseline=baseline, role=role,
            metrics=c,
            note=("primary interval contrast" if role == "PRIMARY" else
                  "secondary exploratory control; no new interval test"))
    inv = dict(
        all_queries_denominator=int(n),
        n_candidates=int(K),
        n_clusters=int(len(set(scene[:n].tolist()))),
        R25_equal_across_methods=len(
            {int(hits_cell[m][25].sum()) for m in T.METHODS}) == 1,
        Top25_5m_equal_across_methods=len(
            {int(hits_loc[m][(25, 5.0)].sum()) for m in T.METHODS}) == 1,
        storage_order_is_coarse_cosine=True,
        p2_is_permutation=True,
        s1_s2_evaluated=False,
        invalid_lists_are_failures=True,
    )
    out = dict(
        status="OK", protocol=dict(
            name=T.PROTOCOL_NAME, domain=a.domain, split=a.split,
            n_queries=int(n), k=K, cluster_unit=info["cluster_unit"],
            denominator=info["denominator"],
            bootstrap=dict(resamples=T.BOOTSTRAP_RESAMPLES,
                           seed=T.BOOTSTRAP_SEED, paired=True,
                           resampling="clusters with replacement; all queries "
                                      "inside a cluster; ratio-of-sums",
                           interval="95% percentile, linear"),
            methods={m: METHOD_LABELS[m] for m in T.METHODS},
            primary_metrics=["Cell_R@1", "Top1@5m"],
            s1_s2="not evaluated",
            nature="exploratory on an already-seen project TEST split; "
                   "not a first blind test and not confirmatory",
        ),
        heads=dict(main=dict(tag="s0", path=head_path,
                             sha256=T.sha256(head_path),
                             selection=head_ck.get("selection"),
                             legacy_a=head_ck.get("legacy_a"))),
        features=dict(path=a.features, sha256=T.sha256(a.features),
                      meta_path=fm_path, meta_sha256=T.sha256(fm_path)),
        export=dict(dir=a.export_dir, meta_path=em_path,
                    meta_sha256=T.sha256(em_path),
                    cache_sha256=em.get("cache_sha256"),
                    manifest_sha256=em.get("manifest_sha256")),
        freeze=dict(path=a.freeze_list, file_sha256=T.sha256(a.freeze_list),
                    freeze_sha256=finfo["freeze_sha256"],
                    canonical_sha256=finfo["canonical_sha256"],
                    protocol_sha256=finfo["protocol_sha256"]),
        marker=(dict(path=os.path.abspath(marker_path),
                     sha256=T.sha256(marker_path))
                if marker else None),
        count_provenance=em.get("count_provenance"),
        cluster_mapping=cluster_map,
        invariants=inv,
        results_cell=res_cell, results_loc=res_loc,
        comparisons=comps,
        primary_contrast=dict(contrast="SCAR_minus_fsr",
                              metrics=["Cell_R@1", "Top1@5m"]),
        secondary_note="All other methods and thresholds are secondary "
                       "exploratory point estimates; no new interval tests.",
        code_sha256={f: T.sha256(os.path.join(HERE, f))
                     for f in ("c_eval_test_five.py", "c_test_common.py")},
        frozen_code_sha256={f: T.sha256(os.path.join(T.FROZEN_CODE, f))
                            for f in ("scar_common.py", "d4p_features.py")},
    )
    np.savez_compressed(os.path.join(a.out_dir, "orders.npz"), orders=orders)
    np.savez_compressed(os.path.join(a.out_dir, "hits.npz"),
                        **{f"cell_{m}_R{k}": hits_cell[m][k]
                           for m in T.METHODS for k in T.KS_CELL},
                        **{f"loc_{m}_Top{k}@{int(t)}m": hits_loc[m][(k, t)]
                           for m in T.METHODS for k in T.KS_LOC
                           for t in T.THRESHS})
    np.savez_compressed(os.path.join(a.out_dir, "distances.npz"),
                        dist_all=dist_all, dist_ordered=dist_ordered)
    T.write_json(out, os.path.join(a.out_dir, "metrics.json"))
    with open(os.path.join(a.out_dir, "per_query.jsonl"), "w",
              encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(a.out_dir, "table_cell_retrieval.csv"), "w",
              encoding="utf-8") as f:
        f.write("method," + ",".join(f"R@{k}" for k in T.KS_CELL) + "\n")
        for m in T.METHODS:
            f.write(m + "," + ",".join(
                f"{100*res_cell[m][f'R@{k}']:.2f}" for k in T.KS_CELL) + "\n")
    with open(os.path.join(a.out_dir, "table_localization.csv"), "w",
              encoding="utf-8") as f:
        f.write("method," + ",".join(
            f"Top{k}@{int(t)}m" for k in T.KS_LOC for t in T.THRESHS) + "\n")
        for m in T.METHODS:
            f.write(m + "," + ",".join(
                f"{100*res_loc[m][f'Top{k}@{int(t)}m']:.2f}"
                for k in T.KS_LOC for t in T.THRESHS) + "\n")
    print("invariants", json.dumps(inv), flush=True)
    for m in T.METHODS:
        print(f"{m:12s} " + " ".join(
            f"R{k}={100*res_cell[m][f'R@{k}']:.2f}" for k in T.KS_CELL)
            + " | " + " ".join(
                f"T{k}@{int(t)}={100*res_loc[m][f'Top{k}@{int(t)}m']:.2f}"
                for k in T.KS_LOC for t in T.THRESHS), flush=True)
    for name, c in comps.items():
        print(name, json.dumps({
            k: dict(delta=round(v["delta"] * 100, 2),
                    ci=[round(x * 100, 2) for x in v.get("ci95", [])])
            for k, v in c["metrics"].items() if k in
            ("Cell_R@1", "Top1@5m")}), flush=True)
    print("saved", a.out_dir, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
