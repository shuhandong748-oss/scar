#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent verifier for one A-domain TEST evaluation.

Recomputes, from the saved candidates/features and the frozen s0 head:
  query/candidate identity, five orders, Cell R@k, localization hits, the
  aggregate table, the paired cluster bootstrap for SCAR-FSR, R@25/Top-25@5m
  invariants, threshold monotonicity, cluster mapping, and that every frozen
  input and the marker still hash to the values recorded at arm time.

It does not read the TEST dataset and does not use the eval script's helpers.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a_test_common as S  # noqa: E402

METHODS = ["original", "presence", "rank_fusion", "fsr", "scar_s0"]
KS_C = (1, 3, 5, 10, 25)
KS_L = (1, 5, 10, 25)
TS = (1.0, 3.0, 5.0, 10.0)


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
    if S.sha256(path) != expected_sha:
        raise RuntimeError("head sha mismatch")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if list(ck["feature_names"]) != list(feature_names):
        raise RuntimeError("head feature names mismatch")
    mean = ck["mean"].astype(np.float32)
    std = ck["std"].astype(np.float32)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(len(feature_names), 64),
                                     nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
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
    ap.add_argument("--domain", required=True, choices=sorted(S.DOMAINS))
    ap.add_argument("--export_dir", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--heads_dir", required=True)
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--freeze_list", required=True)
    ap.add_argument("--marker", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    checks = []
    failures = []

    def chk(name, ok, detail=None):
        checks.append(dict(check=name, pass_=bool(ok), detail=detail))
        if not ok:
            failures.append(name)

    freeze = S.read_json(a.freeze_list)
    chk("freeze_canonical_sha",
        S.canonical_sha({k: v for k, v in freeze.items()
                         if k != "freeze_sha256"}) == freeze["freeze_sha256"])
    marker = S.read_json(a.marker)
    chk("marker_completed", marker.get("status") == "completed"
        and marker.get("armed") is True)
    chk("marker_protocol", marker.get("protocol_sha256")
        == freeze["protocol"]["protocol_sha256"])
    chk("marker_freeze", marker.get("freeze_list_sha256")
        == freeze["freeze_sha256"])
    chk("s1_s2_not_evaluated", all("s1" not in k and "s2" not in k
                                   for k in marker.get("policy", {})
                                   .get("methods", [])))

    em_path = os.path.join(a.export_dir, "export_meta.json")
    em = S.read_json(em_path)
    comp = S.read_json(os.path.join(a.export_dir, "EXPORT_COMPLETE.json"))
    chk("export_complete", comp.get("status") == "complete")
    for name in ("candidates.npz", "coarse_enc.npz"):
        chk(f"cache_sha_{name}",
            S.sha256(os.path.join(a.export_dir, name))
            == comp["cache_sha256"][name])
    chk("marker_binds_cache",
        marker.get("export", {}).get("cache_sha256", {}).get("candidates.npz")
        == comp["cache_sha256"]["candidates.npz"])
    chk("freeze_unchanged", S.sha256(a.freeze_list)
        == marker["freeze_list"]["file_sha256"])
    chk("historical_record_unchanged",
        S.sha256(marker["prior_test_access"]["path"])
        == marker["prior_test_access"]["sha256"])

    # arm-time input record
    arm_inputs_path = os.path.join(os.path.dirname(a.marker), "ARM_INPUTS.json")
    if os.path.isfile(arm_inputs_path):
        ai = S.read_json(arm_inputs_path)
        chk("arm_inputs_sha",
            S.sha256(arm_inputs_path) == marker.get("input_record_sha256"))
        for key in ("config", "known_classes"):
            chk(f"arm_{key}_sha", S.sha256(ai[key]["path"]) == ai[key]["sha256"])
        for st in ("coarse", "fine", "ste_stage1"):
            if st in ai.get("backbones", {}):
                chk(f"arm_{st}_sha", S.sha256(ai["backbones"][st]["path"])
                    == ai["backbones"][st]["sha256"])
        chk("arm_head_sha", S.sha256(ai["main_head"]["path"])
            == ai["main_head"]["sha256"])
        chk("arm_norm_sha", S.sha256(ai["normalisation"]["path"])
            == ai["normalisation"]["sha256"])
    else:
        chk("arm_inputs_present", False, arm_inputs_path)

    cand = np.load(os.path.join(a.export_dir, "candidates.npz"),
                   allow_pickle=True)
    ids = np.asarray([[str(c) for c in row] for row in cand["cell_ids"]])
    cscene = np.asarray([[str(c) for c in row] for row in cand["cell_scene"]])
    pred_w = np.asarray(cand["pred_w"], np.float64)
    gt_w = np.asarray(cand["gt_pose_w"], np.float64)
    # export's own recorded coarse-cosine sort key (see a_eval_test_five.py)
    storage_cos = (np.asarray(cand["coarse_cos"], np.float64)
                   if "coarse_cos" in cand.files else None)
    gt_cell = [str(x) for x in cand["gt_cell"]]
    scene = np.asarray([str(x) for x in cand["scene"]])
    qids = np.asarray([str(x) for x in cand["query_ids"]])
    n, K = ids.shape
    chk("k25", K == 25)
    chk("query_count_expected", n == S.DOMAINS[a.domain]["expected_test"],
        n)
    chk("query_ids_unique", len(set(qids.tolist())) == n)
    chk("candidate_rows_unique",
        all(len(set(row)) == K for row in ids.tolist()))
    chk("cluster_count_expected",
        len(set(scene.tolist())) == S.DOMAINS[a.domain]["expected_test_clusters"],
        len(set(scene.tolist())))
    chk("gt_pose_shape", gt_w.shape == (n, 2))

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
    chk("feature_cos_matches_export",
        storage_cos is not None
        and bool(np.allclose(cos, storage_cos, rtol=0.0, atol=1e-5)))
    chk("feature_sha", S.sha256(a.features) == \
        S.read_json(a.features + ".meta.json").get("feature_file_sha256"))

    head_path = os.path.join(a.heads_dir, f"ht23_{a.domain}_s0.pt")
    head_logits, head_ck = load_head(head_path, freeze["main_head"]["sha256"],
                                     fnames)
    chk("head_sha", S.sha256(head_path) == freeze["main_head"]["sha256"])

    orders_saved = np.load(os.path.join(a.eval_dir, "orders.npz"))["orders"]
    chk("orders_shape", orders_saved.shape == (n, K, len(METHODS)))
    dist_saved = np.load(os.path.join(a.eval_dir, "distances.npz"))
    hits_saved = np.load(os.path.join(a.eval_dir, "hits.npz"))
    metrics = S.read_json(os.path.join(a.eval_dir, "metrics.json"))
    chk("eval_marker_sha", metrics["marker"]["sha256"] == S.sha256(a.marker))

    hc = {m: {k: np.zeros(n, np.int8) for k in KS_C} for m in METHODS}
    hl = {m: {(k, t): np.zeros(n, np.int8) for k in KS_L for t in TS}
          for m in METHODS}
    max_order_diff = 0
    for i in range(n):
        p2i = p2[i]
        os_ = np.argsort(-scf[i][p2i], kind="stable")
        rscf = np.empty(K, np.int64)
        rscf[p2i[os_]] = np.arange(K)
        borda = (np.arange(K) + rscf[p2i]) / (2.0 * (K - 1))
        cur = {
            "original": np.arange(K),
            "presence": np.argsort(-(cos[i] + 0.5 * sp[i]), kind="stable"),
            "rank_fusion": p2i[np.argsort(borda, kind="stable")],
            "fsr": p2i[np.argsort(-sat[i][p2i], kind="stable")],
            "scar_s0": p2i[np.argsort(-head_logits(feats[i][None])[0][p2i],
                                      kind="stable")],
        }
        for mi, m in enumerate(METHODS):
            o = cur[m]
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
        order_key = storage_cos if storage_cos is not None else cos
        if not np.array_equal(cur["original"],
                              np.argsort(-order_key[i], kind="stable")):
            failures.append(f"storage order not cosine q{i}")
    chk("orders_recomputed", max_order_diff == 0, max_order_diff)
    chk("distance_saved", "dist_all mismatch" not in " ".join(failures))

    for m in METHODS:
        for k in KS_C:
            chk(f"cell_{m}_R{k}", np.array_equal(
                hc[m][k], hits_saved[f"cell_{m}_R{k}"]))
            if abs(100*hc[m][k].mean() - 100*metrics["results_cell"][m][f"R@{k}"]) > 1e-6:
                failures.append(f"aggregate cell {m} R{k}")
        for k in KS_L:
            for t in TS:
                key = f"loc_{m}_Top{k}@{int(t)}m"
                chk(f"{key}", np.array_equal(hl[m][(k, t)], hits_saved[key]), None)
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
    chk("no_s1_s2", all(not k.endswith(("s1", "s2")) for k in
                        metrics["results_cell"]))

    keys = scene.reshape(-1, 1)
    for metric_name, arr, sel in (
            ("Cell_R@1", hc, 1), ("Top1@5m", hl, (1, 5.0))):
        dlt, ci = local_bootstrap(arr["scar_s0"][sel], arr["fsr"][sel], keys)
        got = metrics["comparisons"]["SCAR_minus_FSR"]["metrics"][metric_name]
        chk(f"bootstrap_{metric_name}",
            abs(dlt - got["delta"]) < 1e-12
            and abs(ci[0] - got["ci95"][0]) < 1e-12
            and abs(ci[1] - got["ci95"][1]) < 1e-12,
            dict(delta=dlt, ci=ci, saved=got["ci95"]))
    chk("primary_contrast_present",
        metrics["comparisons"]["SCAR_minus_FSR"]["role"] == "PRIMARY")

    cmap = metrics["cluster_mapping"]
    chk("cluster_mapping_sha", S.sha256(cmap["path"]) == cmap["sha256"])
    with open(cmap["path"], encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    chk("cluster_mapping_rows",
        len(rows) == n and len({r["cluster"] for r in rows}) == len(set(scene.tolist())))
    chk("cluster_unit", cmap["cluster_unit"] == S.DOMAINS[a.domain]["cluster_unit"])

    # saved feature/reference SHAs must be untouched
    for s in ("train", "val"):
        p = os.path.join(os.path.dirname(a.freeze_list), "features", f"{s}.npz")
        if os.path.isfile(p) and s in freeze.get("features", {}):
            chk(f"frozen_feature_{s}_unchanged",
                S.sha256(p) == freeze["features"][s]["sha256"])

    out = dict(status="PASS" if not failures else "FAIL", domain=a.domain,
               split="test", n_queries=n, n_clusters=len(set(scene.tolist())),
               test_is_seen_not_blind=True, exploratory_not_confirmatory=True,
               checks=checks, failures=sorted(set(failures)),
               code_sha256={f: S.sha256(os.path.join(HERE, f))
                            for f in ("a_verify_test.py", "a_test_common.py")})
    S.write_json(out, a.out)
    print(json.dumps(dict(status=out["status"], n_checks=len(checks),
                          failures=out["failures"]), indent=2), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
