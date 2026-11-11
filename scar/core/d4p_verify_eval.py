#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent re-derivation of an evaluation run (CPU, no GPU, no export).

Recomputes every method's order from the cache + features + frozen heads,
recomputes the metrics and the paired cluster bootstrap with a second
implementation, and compares the result with the evaluation's own JSON.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scar_common as S  # noqa: E402

KS_CELL = (1, 2, 3, 5, 10, 25)
KS_LOC = (1, 5, 10, 25)
THRESHS = (5.0, 10.0, 15.0)
METHODS = ["original", "presence", "rank_fusion", "fsr", "scar_s0",
           "scar_s1", "scar_s2"]
SEED = 123
RESAMPLES = 2000


def bootstrap_independent(a, b, keys):
    """Same resampling scheme as the evaluator, written as an explicit loop."""
    uniq = sorted({str(x) for x in keys.tolist()})
    pos = {u: j for j, u in enumerate(uniq)}
    g = np.array([pos[str(x)] for x in keys.tolist()], np.int64)
    n_cl = len(uniq)
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    s = np.zeros(n_cl)
    c = np.zeros(n_cl)
    for i in range(len(d)):
        s[g[i]] += d[i]
        c[g[i]] += 1.0
    rng = np.random.RandomState(SEED)
    v = np.empty(RESAMPLES)
    for r in range(RESAMPLES):
        p = rng.randint(0, n_cl, n_cl)
        v[r] = s[p].sum() / c[p].sum()
    return d.mean(), np.percentile(v, 2.5), np.percentile(v, 97.5), n_cl


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["npm3d", "s3dis"])
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--export_dir", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--heads_dir", required=True)
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--protocol_sha", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    for n in ("export_dir", "features", "heads_dir", "eval_dir"):
        setattr(a, n, os.path.abspath(getattr(a, n)))
    d = S.domain_cfg(a.domain)
    checks = {}

    def check(name, ok, **detail):
        checks[name] = dict(PASS=bool(ok), **detail)
        if not ok:
            print(f"FAIL {name}: {detail}", flush=True)
        return ok

    ev = S.read_json(os.path.join(a.eval_dir, "metrics.json"))
    check("eval_status", ev.get("status") == "OK")
    check("eval_protocol_sha", ev.get("protocol_freeze_sha256") == a.protocol_sha)
    check("main_head_is_s0", ev.get("main_head") == "scar_s0")
    check("eval_domain_split",
          ev["protocol"]["domain"] == a.domain and ev["protocol"]["split"] == a.split)
    check("eval_export_dir", os.path.realpath(ev["export_dir"]) ==
          os.path.realpath(a.export_dir))

    sys.path.insert(0, d["vendor"])
    import features_fix as F

    z = np.load(os.path.join(a.export_dir, "candidates.npz"), allow_pickle=True)
    ids = np.asarray([[str(x) for x in r] for r in z["cell_ids"]])
    cscene = np.asarray([[str(x) for x in r] for r in z["cell_scene"]])
    pred_w = np.asarray(z["pred_w"], np.float64)
    gt_cell = [str(x) for x in z["gt_cell"]]
    gt_pose = np.asarray(z["gt_pose_w"], np.float64)
    scene = np.asarray([str(x) for x in z["scene"]])
    n, K = ids.shape
    check("queries", n == d["expected"][a.split], got=n)
    check("clusters_registered",
          len({str(s) for s in scene}) == d["expected_clusters"][a.split],
          got=len({str(s) for s in scene}),
          expected=d["expected_clusters"][a.split])

    ft = np.load(a.features, allow_pickle=True)
    feats = ft["feats"]
    p2 = ft["p2_idx"].astype(np.int64)
    cos = np.asarray(ft["cos"], np.float64)
    sp = np.asarray(ft["sp"], np.float64)
    sc_fine = np.asarray(ft["sc_fine"], np.float64)
    sat_fine = np.asarray(ft["sat_fine"], np.float64)
    check("feature_rows", len(feats) == n)
    check("feature_names",
          list(ft["feature_names"]) == list(F.FEATURE_NAMES))
    check("feature_ids", bool((np.asarray([[str(x) for x in r]
                                           for r in ft["cell_ids"]]) == ids).all()))
    check("feature_gt", [str(x) for x in ft["gt_cell"]] == gt_cell)

    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(23, 64), nn.ReLU(),
                                     nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    heads = {}
    for tag, seed in (("s0", 0), ("s1", 1), ("s2", 2)):
        p = os.path.join(a.heads_dir, f"ht23_{a.domain}_s{seed}.pt")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        check(f"head_s{seed}_sha", S.sha256(p) ==
              (ev.get("heads", {}).get(tag, {}) or {}).get("sha256"))
        net = Net()
        net.load_state_dict(ck["state_dict"])
        net.eval()
        heads[tag] = (net, np.asarray(ck["mean"], np.float32),
                      np.asarray(ck["std"], np.float32), p)

    def logits_of(tag, i):
        net, m, s, _ = heads[tag]
        with torch.no_grad():
            return net(torch.from_numpy(((feats[i] - m) / s)[None]).float()).numpy()[0]

    hit_cell = {m: {k: np.zeros(n, np.int8) for k in KS_CELL} for m in METHODS}
    hit_loc = {m: {(k, t): np.zeros(n, np.int8) for k in KS_LOC for t in THRESHS}
               for m in METHODS}
    orders_all = {m: np.zeros((n, K), np.int64) for m in METHODS}
    logits_all = {tag: np.zeros((n, K)) for tag in heads}
    for i in range(n):
        p2i = np.asarray(p2[i], np.int64)
        order_scf = np.argsort(-sc_fine[i][p2i], kind="stable")
        rank_scf = np.empty(K, np.int64)
        rank_scf[p2i[order_scf]] = np.arange(K)
        borda_c = (np.arange(K, dtype=np.float64) + rank_scf[p2i]) / (2.0 * (K - 1))
        orders = {
            "original": np.arange(K),
            "presence": np.argsort(-(cos[i] + 0.5 * sp[i]), kind="stable"),
            "rank_fusion": p2i[np.argsort(borda_c, kind="stable")],
            "fsr": p2i[np.argsort(-sat_fine[i][p2i], kind="stable")],
        }
        for tag in heads:
            lg = logits_of(tag, i)
            logits_all[tag][i] = lg
            orders[f"scar_{tag}"] = F.rank_by_logits(lg, p2i)
            if set(int(x) for x in orders[f"scar_{tag}"]) != set(int(x) for x in p2i):
                raise SystemExit(f"SCAR order changes the P2 pool at {i}")
        for m, o in orders.items():
            if len(set(int(x) for x in o)) != K:
                raise SystemExit(f"{m}: order is not a permutation at {i}")
            orders_all[m][i] = o
            ranked = ids[i][o]
            for k in KS_CELL:
                hit_cell[m][k][i] = int(gt_cell[i] in ranked[:k].tolist())
            dist = np.linalg.norm(pred_w[i][o] - gt_pose[i][None, :], axis=1)
            dist[cscene[i][o] != scene[i]] = np.inf
            for k in KS_LOC:
                for t in THRESHS:
                    hit_loc[m][(k, t)][i] = int(np.min(dist[:k]) <= t)

    # ---- compare with the evaluator's own outputs ------------------------
    res_cell = ev["results_cell"]
    res_loc = ev["results_loc"]
    check("cell_metrics_match",
          all(abs(res_cell[m][f"R@{k}"] - hit_cell[m][k].mean()) <= 1e-12
              for m in METHODS for k in KS_CELL))
    check("loc_metrics_match",
          all(abs(res_loc[m][f"top{k}@{int(t)}m"] - hit_loc[m][(k, t)].mean()) <= 1e-12
              for m in METHODS for k in KS_LOC for t in THRESHS))

    pq = [json.loads(x) for x in open(os.path.join(a.eval_dir, "per_query.jsonl"),
                                      encoding="utf-8")]
    check("per_query_count", len(pq) == n)
    sample = range(n)
    check("per_query_orders",
          all(pq[i]["orders"][m] == orders_all[m][i].tolist()
              for i in sample for m in METHODS))
    check("per_query_ids", all(pq[i]["ids"] == ids[i].tolist() for i in sample))
    for tag in heads:
        check(f"per_query_logits_{tag}", all(
            float(np.abs(np.asarray(pq[i]["logits"][tag]) - logits_all[tag][i]).max())
            <= 1e-6 for i in sample))

    keys = scene.astype("<U32")
    metric_sel = ([("cell_R1", hit_cell, 1), ("cell_R2", hit_cell, 2),
                   ("cell_R3", hit_cell, 3), ("cell_R5", hit_cell, 5),
                   ("cell_R10", hit_cell, 10)]
                  + [(f"top{k}_{int(t)}m", hit_loc, (k, t))
                     for k in KS_LOC for t in THRESHS])
    label = {"original": "original", "presence": "Presence",
             "rank_fusion": "Rank Fusion", "fsr": "FSR"}
    mism = []
    for target in ("original", "presence", "rank_fusion", "fsr"):
        stored = ev["comparisons"][f"SCAR_minus_{label[target]}"]
        for name, arr, sel in metric_sel:
            delta, lo, hi, ncl = bootstrap_independent(
                arr["scar_s0"][sel], arr[target][sel], keys)
            st = stored[name]
            if (abs(st["delta"] - delta) > 1e-12
                    or abs(st["ci95"][0] - lo) > 1e-9
                    or abs(st["ci95"][1] - hi) > 1e-9
                    or st["n_clusters"] != ncl):
                mism.append(f"{target}:{name}")
    check("bootstrap_reproduced", not mism, mismatches=mism[:10])
    check("bootstrap_clusters",
          all(v["n_clusters"] == len({str(s) for s in scene})
              for block in ev["comparisons"].values() for v in block.values()))

    # coordinate invariance: SCAR must never re-decode; R@25/Top25@5m equal
    check("pool_invariant",
          len({round(res_cell[m]["R@25"], 12) for m in METHODS}) == 1)
    check("top25_invariant",
          len({int(hit_loc[m][(25, 5.0)].sum()) for m in METHODS}) == 1)
    check("storage_order_descending_cosine",
          all(np.array_equal(orders_all["original"][i],
                             np.argsort(-cos[i], kind="stable")) for i in range(n)))

    out = dict(
        status="OK" if all(c["PASS"] for c in checks.values()) else "FAIL",
        domain=a.domain, split=a.split, eval_dir=a.eval_dir,
        eval_metrics_sha256=S.sha256(os.path.join(a.eval_dir, "metrics.json")),
        features_sha256=S.sha256(a.features),
        export_meta_sha256=S.sha256(os.path.join(a.export_dir, "EXPORT_META.json")),
        heads={"s0": S.sha256(heads["s0"][3]), "s1": S.sha256(heads["s1"][3]),
               "s2": S.sha256(heads["s2"][3])},
        gate_binding=S.load_gate(), protocol_freeze_sha256=a.protocol_sha,
        n_queries=n, n_clusters=int(len({str(s) for s in scene})),
        bootstrap=dict(resamples=RESAMPLES, seed=SEED, cluster_unit=d["cluster_unit"]),
        checks=checks, failures=[k for k, v in checks.items() if not v["PASS"]],
        code_sha256={f: S.sha256(os.path.join(HERE, f))
                     for f in ("d4p_verify_eval.py", "scar_common.py")},
    )
    S.write_json(out, a.out)
    print(json.dumps({k: out[k] for k in ("status", "n_queries", "n_clusters",
                                         "failures")}, indent=2), flush=True)
    print("saved", a.out, flush=True)
    return 0 if out["status"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
