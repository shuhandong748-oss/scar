#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent verification of VAL selection and VAL evaluation.

Selection: recompute all 30 per-epoch VAL metric pairs from the saved weights and
    the TRAIN-fitted normalisers, recompute arg-max, and require the frozen head
    tensors to equal the selected epoch's tensors.
Evaluation: recompute SCAR/baseline orders, logits, per-query hits, aggregates
    and two paired-bootstrap comparisons from the raw features and head weights,
    then compare to the recorded files.

Writes VERIFY_SELECTION_val.json and VERIFY_EVAL_val.json.  VAL only.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a_scar_common as S  # noqa: E402

THRESHS = (5.0, 10.0, 15.0)


class Head:
    def __init__(self, path):
        import torch
        import torch.nn as nn
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.ck = ck
        self.mean = ck["mean"].astype(np.float32)
        self.std = ck["std"].astype(np.float32)
        self.nf = len(ck["feature_names"])

        class Net(nn.Module):
            def __init__(self, nf):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(nf, 64), nn.ReLU(),
                                         nn.Linear(64, 32), nn.ReLU(),
                                         nn.Linear(32, 1))

            def forward(self, z):
                return self.net(z).squeeze(-1)

        self.torch = torch
        self.model = Net(self.nf)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()

    def logits(self, feats):
        x = (feats - self.mean) / self.std
        with self.torch.no_grad():
            return self.model(self.torch.from_numpy(x.astype(np.float32))).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=sorted(S.DOMAINS))
    ap.add_argument("--features", required=True)
    ap.add_argument("--heads_dir", required=True)
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    d = S.domain_cfg(args.domain)
    S.insert_feature_paths(args.domain)
    import features_fix as F
    import torch

    feat = np.load(args.features, allow_pickle=True)
    feats = feat["feats"]
    ids = np.asarray(feat["cell_ids"])
    p2 = feat["p2_idx"].astype(np.int64)
    cos = np.asarray(feat["cos"], np.float64)
    sp = np.asarray(feat["sp"], np.float64)
    sc_fine = np.asarray(feat["sc_fine"], np.float64)
    sat = np.asarray(feat["sat_fine"], np.float64)
    pred_w = np.asarray(feat["pred_w"], np.float64)
    gt_pose = np.asarray(feat["gt_pose_w"], np.float64)
    gt_cell = [str(x) for x in feat["gt_cell"]]
    scene = np.asarray([str(x) for x in feat["scene"]])
    n, K = ids.shape
    if list(feat["feature_names"]) != list(F.FEATURE_NAMES):
        raise SystemExit("feature names not canonical 23")

    sel_path = os.path.join(args.heads_dir, "SELECTION_FORMAL.json")
    sel = S.read_json(sel_path)
    norm = S.read_json(os.path.join(args.heads_dir, "NORMALIZATION.json"))
    mean = np.asarray(norm["mean"], np.float32)
    std = np.asarray(norm["std"], np.float32)

    # ---- selection ----
    sel_checks = {}
    recomputed = {}
    for seed in (0, 1, 2):
        head_path = os.path.join(args.heads_dir, f"ht23_{args.domain}_s{seed}.pt")
        h = Head(head_path)
        if not np.array_equal(h.mean, mean) or not np.array_equal(h.std, std):
            raise SystemExit(f"s{seed}: head normalisers != NORMALIZATION.json")
        epoch_dir = os.path.join(args.heads_dir, "epochs", f"s{seed}")
        files = sorted(f for f in os.listdir(epoch_dir) if f.startswith("ep"))
        if len(files) != 30:
            raise SystemExit(f"s{seed}: {len(files)} epoch files, expected 30")
        hist = []
        for fn in files:
            ep = int(fn[2:4])
            ck = torch.load(os.path.join(epoch_dir, fn), map_location="cpu",
                            weights_only=False)
            h.model.load_state_dict(ck["state_dict"])
            lg = h.logits(feats)
            rm, lm, _, _ = S.val_metrics(S.rank_orders(lg, p2), feat, K)
            hist.append((ep, lm["top1@5m"], rm["R@1"], lm, rm))
        best = max(hist, key=lambda r: (r[1], r[2], -r[0]))
        got = sel["seeds"][f"s{seed}"]["formal_epoch"]
        recomputed[f"s{seed}"] = {"recomputed_epoch": best[0],
                                  "recorded_epoch": got,
                                  "key": [best[1], best[2], -best[0]]}
        if best[0] != got:
            raise SystemExit(f"s{seed}: selection mismatch {best[0]} != {got}")
        # frozen head tensors == selected epoch tensors
        ep_ck = torch.load(os.path.join(epoch_dir, f"ep{best[0]:02d}.pt"),
                           map_location="cpu", weights_only=False)
        head_ck = torch.load(head_path, map_location="cpu", weights_only=False)
        same = all(torch.equal(head_ck["state_dict"][k], v)
                   for k, v in ep_ck["state_dict"].items())
        sel_checks[f"s{seed}_tensors_equal_selected_epoch"] = bool(same)
        sel_checks[f"s{seed}_epoch_matches"] = True
    sel_checks["main_is_s0"] = (
        sel["main_head_sha256"] == sel["seeds"]["s0"]["sha256"]
        and sel["main_head_sha256"] == S.sha256(sel["main_head"]))
    sel_checks["normalization_tied_to_train_npz"] = (
        norm["train_npz_sha256"] == sel["train_features_sha256"])
    sel_checks["feature_version"] = sel.get("feature_version") == S.FEATURE_VERSION
    sel_ok = all(v for v in sel_checks.values() if isinstance(v, bool)) and \
        sel_checks["feature_version"]
    sel_report = {"status": "OK" if sel_ok else "FAIL",
                  "checks": sel_checks, "recomputed": recomputed,
                  "selected": {f"s{s}": sel["seeds"][f"s{s}"]["formal_epoch"]
                               for s in (0, 1, 2)},
                  "main_head_sha256": sel["main_head_sha256"]}
    S.write_json(sel_report, os.path.join(args.out_dir, "VERIFY_SELECTION_val.json"))

    # ---- evaluation ----
    metrics = S.read_json(os.path.join(args.eval_dir, "metrics.json"))
    rows = [json.loads(line) for line in
            open(os.path.join(args.eval_dir, "per_query.jsonl"), encoding="utf-8")]
    heads = {s: Head(os.path.join(args.heads_dir,
                                  f"ht23_{args.domain}_s{s}.pt")) for s in (0, 1, 2)}
    ev_checks = {}
    logit_max = 0.0
    order_errors = 0
    hit_errors = 0
    hit_cell = {m: {k: np.zeros(n, np.int8) for k in S.KS_CELL}
                for m in ("original", "presence", "rank_fusion", "fsr",
                          "scar_s0", "scar_s1", "scar_s2")}
    hit_loc = {m: {(k, t): np.zeros(n, np.int8)
                   for k in S.KS_LOC for t in THRESHS}
               for m in hit_cell}
    for i in range(n):
        p2i = p2[i]
        order_scf = np.argsort(-sc_fine[i][p2i], kind="stable")
        rank_scf = np.empty(K, np.int64)
        rank_scf[p2i[order_scf]] = np.arange(K)
        borda = (np.arange(K, dtype=np.float64) + rank_scf[p2i]) / (2.0 * (K - 1))
        orders = {
            "original": np.arange(K),
            "presence": np.argsort(-(cos[i] + 0.5 * sp[i]), kind="stable"),
            "rank_fusion": p2i[np.argsort(borda, kind="stable")],
            "fsr": p2i[np.argsort(-sat[i][p2i], kind="stable")],
        }
        for s in (0, 1, 2):
            lg = heads[s].logits(feats[i][None])[0]
            logit_max = max(logit_max, float(np.abs(lg - np.asarray(
                rows[i]["logits"][f"s{s}"])).max()))
            orders[f"scar_s{s}"] = F.rank_by_logits(lg, p2i)
        for m, o in orders.items():
            if list(o) != rows[i]["orders"][m]:
                order_errors += 1
            ranked = [str(ids[i][j]) for j in o]
            dist = np.linalg.norm(pred_w[i][o] - gt_pose[i][None], axis=1)
            dist[np.asarray([c.split("_")[0] for c in ranked]) != scene[i]] = np.inf
            for k in S.KS_CELL:
                v = int(gt_cell[i] in ranked[:k])
                hit_cell[m][k][i] = v
                hit_errors += int(v != rows[i]["hit_cell"][m][f"R{k}"])
            for k in S.KS_LOC:
                for t in THRESHS:
                    v = int(np.min(dist[:k]) <= t)
                    hit_loc[m][(k, t)][i] = v
                    hit_errors += int(
                        v != rows[i]["hit_loc"][m][f"top{k}@{int(t)}m"])

    res_cell = {m: {f"R@{k}": float(hit_cell[m][k].mean()) for k in S.KS_CELL}
                for m in hit_cell}
    res_loc = {m: {f"top{k}@{int(t)}m": float(hit_loc[m][(k, t)].mean())
                   for k in S.KS_LOC for t in THRESHS} for m in hit_cell}
    agg_err = max([abs(res_cell[m][f"R@{k}"] - metrics["results_cell"][m][f"R@{k}"])
                   for m in res_cell for k in S.KS_CELL]
                  + [abs(res_loc[m][f"top{k}@{int(t)}m"]
                         - metrics["results_loc"][m][f"top{k}@{int(t)}m"])
                     for m in res_loc for k in S.KS_LOC for t in THRESHS])
    keys = scene.astype("<U32").reshape(-1, 1)
    b_c1 = S.cluster_bootstrap(hit_cell["scar_s0"][1],
                               hit_cell["fsr"][1], keys)
    b_t1 = S.cluster_bootstrap(hit_loc["scar_s0"][(1, 5.0)],
                               hit_loc["fsr"][(1, 5.0)], keys)
    rec_c1 = metrics["comparisons"]["SCAR_minus_FSR"]["cell_R1"]
    rec_t1 = metrics["comparisons"]["SCAR_minus_FSR"]["top1_5m"]
    boot_ok = (abs(b_c1["delta"] - rec_c1["delta"]) < 1e-12
               and abs(b_t1["delta"] - rec_t1["delta"]) < 1e-12
               and np.allclose(b_c1["ci95"], rec_c1["ci95"], atol=1e-12)
               and np.allclose(b_t1["ci95"], rec_t1["ci95"], atol=1e-12))
    ev_checks["row_count"] = len(rows) == n
    ev_checks["logits_match"] = logit_max < 1e-5
    ev_checks["orders_match"] = order_errors == 0
    ev_checks["per_query_hits_match"] = hit_errors == 0
    ev_checks["aggregates_match"] = agg_err < 1e-12
    ev_checks["bootstrap_reproduced"] = bool(boot_ok)
    ev_checks["invariants_present"] = bool(
        metrics["invariants"]["R25_equal"]
        and metrics["invariants"]["top25_5m_equal"]
        and metrics["invariants"]["same_pool"])
    ev_checks["heads_feature_version"] = all(
        heads[s].ck.get("feature_version") == S.FEATURE_VERSION for s in (0, 1, 2))
    ev_ok = all(ev_checks.values())
    ev_report = {"status": "OK" if ev_ok else "FAIL", "checks": ev_checks,
                 "logits_max_abs_diff": logit_max,
                 "order_errors": order_errors, "hit_errors": hit_errors,
                 "aggregate_max_abs_diff": float(agg_err),
                 "bootstrap_delta": b_c1["delta"], "bootstrap_ci": b_c1["ci95"],
                 "n_queries": int(n),
                 "n_clusters": int(len({str(s) for s in scene}))}
    S.write_json(ev_report, os.path.join(args.out_dir, "VERIFY_EVAL_val.json"))
    print(json.dumps({"selection": sel_report, "evaluation": ev_report}, indent=2))
    return 0 if (sel_ok and ev_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
