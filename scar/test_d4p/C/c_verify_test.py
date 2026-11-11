#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent verifier for one C-domain five-order TEST evaluation.

It uses only the saved export/feature arrays and the frozen s0 checkpoint.  It
recomputes all five orders, Cell/localization hits, the aggregate table, the
primary SCAR-FSR cluster bootstrap, the R@25/Top-25@5m invariants, threshold
monotonicity, the cluster map, and every frozen input/marker hash.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c_test_common as T  # noqa: E402

S = T.frozen_common()
METHODS = list(T.METHODS)
KS_C = tuple(T.KS_CELL)
KS_L = tuple(T.KS_LOC)
TS = tuple(T.THRESHS)


def local_bootstrap(a, b, keys, resamples=2000, seed=123):
    keys = np.asarray(keys).reshape(-1, 1)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    diff = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    sd = np.bincount(inv, weights=diff, minlength=len(uniq))
    cn = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    rng = np.random.RandomState(seed)
    vals = np.empty(resamples)
    for i in range(resamples):
        pick = rng.randint(0, len(uniq), len(uniq))
        vals[i] = sd[pick].sum() / cn[pick].sum()
    return (float(diff.mean()),
            [float(np.percentile(vals, 2.5, method="linear")),
             float(np.percentile(vals, 97.5, method="linear"))])


def load_head(path, expected_sha, feature_names):
    import torch
    import torch.nn as nn
    if T.sha256(path) != expected_sha:
        raise RuntimeError("head sha mismatch")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if list(ck["feature_names"]) != list(feature_names):
        raise RuntimeError("head feature names mismatch")
    mean = np.asarray(ck["mean"], np.float32)
    std = np.asarray(ck["std"], np.float32)

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

    def logits(x):
        z = (np.asarray(x, np.float32) - mean) / std
        with torch.no_grad():
            return model(torch.from_numpy(z)).numpy()

    return logits, ck


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=sorted(T.DOMAINS))
    ap.add_argument("--export_dir", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--freeze_list", required=True)
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--marker", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    checks, failures = [], []

    def chk(name, ok, detail=None):
        checks.append(dict(check=name, pass_=bool(ok), detail=detail))
        if not ok:
            failures.append(name)

    freeze, finfo = T.freeze_info(a.freeze_list, a.domain)
    marker_path = a.marker or os.path.join(a.export_dir, "TEST_READ_ONCE.json")
    marker = T.read_json(marker_path)
    chk("marker_completed", marker.get("status") == "completed"
        and marker.get("armed") is True)
    chk("marker_protocol", marker.get("protocol_freeze_sha256")
        == finfo["protocol_sha256"])
    chk("marker_freeze", marker.get("freeze_list_sha256")
        == finfo["freeze_sha256"])
    chk("marker_freeze_file", marker.get("freeze_file_sha256")
        == T.sha256(a.freeze_list))
    chk("marker_head", marker.get("main_head", {}).get("sha256")
        == freeze["main_head"]["sha256"])
    chk("marker_normalisation", marker.get("normalisation", {}).get("sha256")
        == freeze["normalisation"]["sha256"])
    for tag in ("coarse", "fine"):
        chk(f"marker_{tag}", marker.get("backbones", {}).get(tag, {}).get("sha256")
            == freeze["backbones"][tag]["sha256"])
    chk("s1_s2_not_evaluated",
        marker.get("policy", {}).get("s1_s2") == "not evaluated"
        and all("s1" not in k and "s2" not in k
                for k in marker.get("policy", {}).get("methods", [])))

    em_path = os.path.join(a.export_dir, "EXPORT_META.json")
    em = T.read_json(em_path)
    comp = T.read_json(os.path.join(a.export_dir, "EXPORT_COMPLETE.json"))
    chk("export_complete", comp.get("status") == "complete")
    for name, h in (comp.get("cache_sha256") or {}).items():
        chk(f"cache_sha_{name}", os.path.isfile(os.path.join(a.export_dir, name))
            and T.sha256(os.path.join(a.export_dir, name)) == h)
    chk("marker_binds_cache",
        marker.get("cache_sha256", {}).get("candidates.npz")
        == comp["cache_sha256"]["candidates.npz"])
    chk("freeze_file_unchanged", T.sha256(a.freeze_list)
        == marker.get("freeze_file_sha256"))
    chk("historical_marker_unchanged",
        T.sha256(marker["historical_test_access"]["path"])
        == marker["historical_test_access"]["sha256"])
    chk("prior_record_unchanged",
        T.sha256(marker["prior_test_access"]["path"])
        == marker["prior_test_access"]["sha256"])
    chk("frozen_code_unchanged",
        T.sha256(os.path.join(T.FROZEN_CODE, "scar_common.py"))
        == marker["frozen_code"]["scar_common_sha256"])

    cand = np.load(os.path.join(a.export_dir, "candidates.npz"),
                   allow_pickle=True)
    ids = np.asarray([[str(c) for c in row] for row in cand["cell_ids"]])
    cscene = np.asarray([[str(c) for c in row] for row in cand["cell_scene"]])
    pred_w = np.asarray(cand["pred_w"], np.float64)
    gt_w = np.asarray(cand["gt_pose_w"], np.float64)
    gt_cell = [str(x) for x in cand["gt_cell"]]
    scene = np.asarray([str(x) for x in cand["scene"]])
    qids = np.asarray([str(x) for x in cand["query_ids"]])
    n, K = ids.shape
    info = T.domain_info(a.domain)
    chk("k25", K == 25)
    chk("query_count_expected", n == info["expected_test"], n)
    chk("query_ids_unique", len(set(qids.tolist())) == n)
    chk("candidate_rows_unique", all(len(set(row)) == K for row in ids.tolist()))
    chk("cluster_count_expected",
        len(set(scene.tolist())) == info["expected_test_clusters"],
        len(set(scene.tolist())))
    chk("gt_pose_shape", gt_w.shape == (n, 2))
    chk("export_queries_meta", em.get("n_queries") == n)

    if a.domain == "s3dis":
        cp = marker.get("count_provenance") or {}
        chk("s3dis_count_provenance", cp.get("old_guard") == 544
            and cp.get("effective") == 542
            and cp.get("raw_manifest_sha256") == T.S3DIS_MANIFEST_SHA256)
        chk("s3dis_export_provenance",
            (em.get("count_provenance") or {}).get("effective") == 542)
        rep = T.read_json(cp["report_path"])
        chk("s3dis_report_sha", T.sha256(cp["report_path"])
            == cp["report_sha256"])
        chk("s3dis_report_id", rep.get("id") == T.S3DIS_REPORT_ID)
        chk("s3dis_manifest_sha",
            T.sha256(os.path.join(S.domain_cfg("s3dis")["base"],
                                  "dataset_manifest.json"))
            == T.S3DIS_MANIFEST_SHA256)

    feat = np.load(a.features, allow_pickle=True)
    feats = np.asarray(feat["feats"], np.float32)
    fids = np.asarray([[str(c) for c in row] for row in feat["cell_ids"]])
    p2 = np.asarray(feat["p2_idx"], np.int64)
    cos = np.asarray(feat["cos"], np.float64)
    sp = np.asarray(feat["sp"], np.float64)
    scf = np.asarray(feat["sc_fine"], np.float64)
    sat = np.asarray(feat["sat_fine"], np.float64)
    fnames = [str(x) for x in feat["feature_names"]]
    valid = np.asarray(feat["valid"]).astype(bool) if "valid" in feat.files else None
    chk("feature_queries", len(feats) == n)
    chk("feature_order_matches_export", np.array_equal(fids, ids))
    chk("feature_names23", len(fnames) == 23)
    chk("feature_gt_cell", [str(x) for x in feat["gt_cell"]] == gt_cell)
    chk("feature_sha", T.sha256(a.features)
        == T.read_json(a.features + ".meta.json").get("feature_file_sha256"))
    chk("feature_meta_export_sha", T.read_json(a.features + ".meta.json")
        .get("export_meta_sha256") == T.sha256(em_path))
    chk("feature_test_opened", T.read_json(a.features + ".meta.json")
        .get("test_opened") is True)
    if a.domain == "s3dis":
        chk("feature_count_provenance",
            (T.read_json(a.features + ".meta.json")
             .get("s3dis_count_provenance") or {}).get("effective") == 542)

    head_path = freeze["main_head"]["path"]
    head_logits, head_ck = load_head(head_path, freeze["main_head"]["sha256"],
                                     fnames)
    chk("head_sha", T.sha256(head_path) == freeze["main_head"]["sha256"])
    norm = T.read_json(freeze["normalisation"]["path"])
    chk("norm_matches_head",
        np.allclose(np.asarray(norm["mean"], np.float32),
                    np.asarray(head_ck["mean"], np.float32), atol=0)
        and np.allclose(np.asarray(norm["std"], np.float32),
                        np.asarray(head_ck["std"], np.float32), atol=0))

    orders_saved = np.load(os.path.join(a.eval_dir, "orders.npz"))["orders"]
    chk("orders_shape", orders_saved.shape == (n, K, len(METHODS)))
    dist_saved = np.load(os.path.join(a.eval_dir, "distances.npz"))
    hits_saved = np.load(os.path.join(a.eval_dir, "hits.npz"))
    metrics = T.read_json(os.path.join(a.eval_dir, "metrics.json"))
    chk("eval_marker_sha", metrics["marker"]["sha256"] == T.sha256(marker_path))
    chk("no_s1_s2_results",
        all("s1" not in k and "s2" not in k for k in metrics["results_cell"]))

    hc = {m: {k: np.zeros(n, np.int8) for k in KS_C} for m in METHODS}
    hl = {m: {(k, t): np.zeros(n, np.int8) for k in KS_L for t in TS}
          for m in METHODS}
    max_order_diff = 0
    for i in range(n):
        p2i = p2[i]
        orders = T.five_orders(feats[i], p2i, cos[i], sp[i], scf[i], sat[i],
                               head_logits(feats[i][None, :, :])[0])
        for mi, m in enumerate(METHODS):
            o = orders[m]
            max_order_diff = max(max_order_diff,
                                 int(np.abs(o - orders_saved[i, :, mi]).max()))
            ranked = [str(ids[i][j]) for j in o]
            for k in KS_C:
                hc[m][k][i] = int(gt_cell[i] in ranked[:k])
            d = np.linalg.norm(pred_w[i][o] - gt_w[i][None, :], axis=1)
            d[cscene[i][o] != scene[i]] = np.inf
            for k in KS_L:
                for t in TS:
                    hl[m][(k, t)][i] = int(np.min(d[:k]) <= t)
        d_all = np.linalg.norm(pred_w[i] - gt_w[i][None, :], axis=1)
        d_all[cscene[i] != scene[i]] = np.inf
        if not np.allclose(d_all, dist_saved["dist_all"][i], atol=1e-9):
            failures.append(f"dist_all mismatch q{i}")
        if not np.array_equal(orders["original"],
                              np.argsort(-cos[i], kind="stable")):
            failures.append(f"storage order not cosine q{i}")
    chk("orders_recomputed", max_order_diff == 0, max_order_diff)
    chk("distance_saved", not any(f.startswith("dist_all mismatch")
                                  for f in failures))

    for m in METHODS:
        for k in KS_C:
            chk(f"cell_{m}_R{k}",
                np.array_equal(hc[m][k], hits_saved[f"cell_{m}_R{k}"]))
            if abs(100*hc[m][k].mean()
                   - 100*metrics["results_cell"][m][f"R@{k}"]) > 1e-6:
                failures.append(f"aggregate cell {m} R{k}")
        for k in KS_L:
            for t in TS:
                key = f"loc_{m}_Top{k}@{int(t)}m"
                chk(key, np.array_equal(hl[m][(k, t)], hits_saved[key]))
                if abs(100*hl[m][(k, t)].mean()
                       - 100*metrics["results_loc"][m][f"Top{k}@{int(t)}m"]) > 1e-6:
                    failures.append(f"aggregate loc {key}")

    chk("R25_equal", len({int(hc[m][25].sum()) for m in METHODS}) == 1)
    chk("Top25_5m_equal", len({int(hl[m][(25, 5.0)].sum())
                               for m in METHODS}) == 1)
    mono = True
    for m in METHODS:
        for a_, b_ in zip(KS_C[:-1], KS_C[1:]):
            mono &= hc[m][a_].mean() <= hc[m][b_].mean() + 1e-12
        for t1, t2 in ((1.0, 3.0), (3.0, 5.0), (5.0, 10.0)):
            mono &= hl[m][(1, t1)].mean() <= hl[m][(1, t2)].mean() + 1e-12
        for a_, b_ in zip(KS_L[:-1], KS_L[1:]):
            mono &= hl[m][(a_, 5.0)].mean() <= hl[m][(b_, 5.0)].mean() + 1e-12
    chk("monotonic", mono)
    chk("all_queries_denominator", len(scene) == n
        and (valid is None or len(valid) == n))
    chk("no_s1_s2", set(METHODS) == set(
        metrics["protocol"]["methods"].keys()))

    keys = scene.reshape(-1, 1)
    for metric_name, arr, sel in (
            ("Cell_R@1", hc, 1), ("Top1@5m", hl, (1, 5.0))):
        dlt, ci = local_bootstrap(arr["scar_s0"][sel], arr["fsr"][sel], keys)
        got = metrics["comparisons"]["SCAR_minus_fsr"]["metrics"][metric_name]
        chk(f"bootstrap_{metric_name}",
            abs(dlt - got["delta"]) < 1e-12
            and abs(ci[0] - got["ci95"][0]) < 1e-12
            and abs(ci[1] - got["ci95"][1]) < 1e-12,
            dict(delta=dlt, ci=ci, saved=got.get("ci95")))
    chk("primary_contrast_role",
        metrics["comparisons"]["SCAR_minus_fsr"]["role"] == "PRIMARY")
    chk("secondary_no_new_interval",
        all("ci95" not in v for k, c in metrics["comparisons"].items()
            if k != "SCAR_minus_fsr" for v in c["metrics"].values()))

    cmap = metrics["cluster_mapping"]
    chk("cluster_mapping_sha", T.sha256(cmap["path"]) == cmap["sha256"])
    with open(cmap["path"], encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    chk("cluster_mapping_rows",
        len(rows) == n and len({r["cluster"] for r in rows})
        == len(set(scene.tolist())))
    chk("cluster_unit", cmap["cluster_unit"] == info["cluster_unit"])

    for s in ("train", "val"):
        p = freeze["features"][s]["path"]
        chk(f"frozen_feature_{s}_unchanged",
            T.sha256(p) == freeze["features"][s]["sha256"])

    out = dict(status="PASS" if not failures else "FAIL", domain=a.domain,
               split="test", n_queries=n,
               n_clusters=len(set(scene.tolist())),
               test_is_seen_not_blind=True, exploratory_not_confirmatory=True,
               checks=checks, failures=sorted(set(failures)),
               code_sha256={f: T.sha256(os.path.join(HERE, f))
                            for f in ("c_verify_test.py", "c_test_common.py")})
    T.write_json(out, a.out)
    print(json.dumps(dict(status=out["status"], n_checks=len(checks),
                          failures=out["failures"]), indent=2), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
