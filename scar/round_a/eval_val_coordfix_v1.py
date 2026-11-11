#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Same-pool VAL evaluation for one A domain under FEATURE_COORD_FIX_v1.

All methods share queries, the 25 candidates and the Fine coordinates; only the
ordering changes (original / Presence / Rank Fusion / FSR / SCAR s0-s2).
Paired cluster bootstrap over the true scene (2000 resamples, seed 123).
VAL only: this script has no TEST branch.
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
METHOD_LABELS = {
    "original": "original", "presence": "Presence",
    "rank_fusion": "Rank Fusion", "fsr": "FSR",
    "scar_s0": "SCAR", "scar_s1": "SCAR s1 (stability)",
    "scar_s2": "SCAR s2 (stability)",
}


class Head:
    def __init__(self, path):
        import torch
        import torch.nn as nn
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.mean = ck["mean"].astype(np.float32)
        self.std = ck["std"].astype(np.float32)
        self.selection = ck["selection"]
        self.feature_version = ck.get("feature_version")
        self.nf = len(ck["feature_names"])
        K = ck["config"]["k"]
        norms = ck["config"]["norms"]
        if self.nf != 23 or K != 25:
            raise SystemExit(f"{path}: not a canonical-23/K25 head")
        if norms != dict(pool_norm=24, rank_norm=24, tied_norm=25):
            raise SystemExit(f"{path}: unexpected norms {norms}")
        if self.feature_version != S.FEATURE_VERSION:
            raise SystemExit(f"{path}: head feature version "
                             f"{self.feature_version!r} != {S.FEATURE_VERSION}")

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
        for p in self.model.parameters():
            p.requires_grad_(False)

    def logits(self, feats):
        x = (feats - self.mean) / self.std
        with self.torch.no_grad():
            return self.model(self.torch.from_numpy(x.astype(np.float32))).numpy()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=sorted(S.DOMAINS))
    ap.add_argument("--split", default="val", choices=["val"])
    ap.add_argument("--features", required=True)
    ap.add_argument("--heads_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--protocol_sha", required=True)
    ap.add_argument("--coord_fix_sha", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    for name in ("features", "heads_dir", "out_dir"):
        setattr(a, name, os.path.abspath(getattr(a, name)))
    os.makedirs(a.out_dir, exist_ok=True)
    existing = os.listdir(a.out_dir)
    if existing:
        raise SystemExit(f"{a.out_dir} is not empty: {existing[:5]}")

    d = S.domain_cfg(a.domain)
    S.insert_feature_paths(a.domain)
    import features_fix as F

    feat = np.load(a.features, allow_pickle=True)
    feats = feat["feats"]
    ids = np.asarray(feat["cell_ids"])
    p2 = feat["p2_idx"].astype(np.int64)
    cos = np.asarray(feat["cos"], np.float64)
    sp = np.asarray(feat["sp"], np.float64)
    sc_fine = np.asarray(feat["sc_fine"], np.float64)
    sat_fine = np.asarray(feat["sat_fine"], np.float64)
    pred_w = np.asarray(feat["pred_w"], np.float64)
    gt_pose_w = np.asarray(feat["gt_pose_w"], np.float64)
    gt_cell = [str(x) for x in feat["gt_cell"]]
    scene = np.asarray([str(x) for x in feat["scene"]])
    if list(feat["feature_names"]) != list(F.FEATURE_NAMES):
        raise SystemExit("feature names are not canonical 23")
    if str(feat["split"][0]) != "val":
        raise SystemExit("expected a VAL feature file")
    n, K = ids.shape
    if K != 25 or len(feats) != n:
        raise SystemExit("feature/K mismatch")
    if len(feats) != d["expected"]["val"]:
        raise SystemExit(f"VAL lists {len(feats)} != {d['expected']['val']}")
    if len({len(set(str(c) for c in row)) for row in ids}) != 1:
        raise SystemExit("candidate rows are not 25 unique ids")

    heads = {}
    for tag, seed in (("s0", 0), ("s1", 1), ("s2", 2)):
        path = os.path.join(a.heads_dir, f"ht23_{a.domain}_s{seed}.pt")
        if not os.path.isfile(path):
            raise SystemExit(f"missing head {path}")
        heads[tag] = (Head(path), path)

    n_done = min(n, a.limit) if a.limit else n
    methods = ["original", "presence", "rank_fusion", "fsr", "scar_s0",
               "scar_s1", "scar_s2"]
    hit_cell = {m: {k: np.zeros(n_done, np.int8) for k in S.KS_CELL}
                for m in methods}
    hit_loc = {m: {(k, t): np.zeros(n_done, np.int8)
                   for k in S.KS_LOC for t in THRESHS} for m in methods}
    rows = []
    order_errors = 0
    top25_hit = np.zeros(n_done, np.int8)

    for i in range(n_done):
        p2i = np.asarray(p2[i], np.int64)
        order_scf = np.argsort(-sc_fine[i][p2i], kind="stable")
        rank_scf = np.empty(K, np.int64)
        rank_scf[p2i[order_scf]] = np.arange(K)
        borda_c = (np.arange(K, dtype=np.float64) + rank_scf[p2i]) / (2.0 * (K - 1))
        order_rf = p2i[np.argsort(borda_c, kind="stable")]
        order_fsr = p2i[np.argsort(-sat_fine[i][p2i], kind="stable")]
        order_pres = np.argsort(-(cos[i] + 0.5 * sp[i]), kind="stable")
        orders = {"original": np.arange(K), "presence": order_pres,
                  "rank_fusion": order_rf, "fsr": order_fsr}
        logits = {}
        for tag, (h, _p) in heads.items():
            lg = h.logits(feats[i][None, :, :])[0]
            logits[tag] = lg
            orders[f"scar_{tag}"] = F.rank_by_logits(lg, p2i)

        for m, o in orders.items():
            if len(set(int(x) for x in o)) != K:
                raise SystemExit(f"{m}: order is not a permutation at query {i}")
            ranked_ids = [str(ids[i][j]) for j in o]
            for k in S.KS_CELL:
                hit_cell[m][k][i] = int(gt_cell[i] in ranked_ids[:k])
            dist = np.linalg.norm(pred_w[i][o] - gt_pose_w[i][None, :], axis=1)
            cs = [c.split("_")[0] for c in ranked_ids]
            dist[np.asarray(cs) != scene[i]] = np.inf
            for k in S.KS_LOC:
                for t in THRESHS:
                    hit_loc[m][(k, t)][i] = int(np.min(dist[:k]) <= t)

        if len({int(hit_cell[m][25][i]) for m in methods}) != 1:
            raise SystemExit(f"R@25 differs across methods at query {i}")
        # Top-25@5m is the min distance over the whole shared pool, so every
        # ordering must agree; hit_loc stores only the reported k in KS_LOC.
        d_all = np.linalg.norm(pred_w[i] - gt_pose_w[i][None, :], axis=1)
        d_all[np.asarray([str(c).split("_")[0]
                          for c in ids[i]]) != scene[i]] = np.inf
        if len({int(np.min(d_all) <= 5.0) for m in methods}) != 1:
            raise SystemExit(f"Top25@5m differs across methods at query {i}")
        top25_hit[i] = int(np.min(d_all) <= 5.0)
        if not np.array_equal(np.asarray(orders["original"]),
                              np.argsort(-cos[i], kind="stable")):
            order_errors += 1

        rows.append(dict(q=i, scene=str(scene[i]), gt_cell=gt_cell[i],
                         ids=[str(x) for x in ids[i]],
                         orders={m: [int(x) for x in o] for m, o in orders.items()},
                         logits={k: [float(x) for x in v] for k, v in logits.items()},
                         hit_cell={m: {f"R{k}": int(hit_cell[m][k][i])
                                       for k in S.KS_CELL} for m in methods},
                         hit_loc={m: {f"top{k}@{int(t)}m":
                                      int(hit_loc[m][(k, t)][i])
                                      for k in S.KS_LOC for t in THRESHS}
                                  for m in methods}))
    if order_errors:
        raise SystemExit("storage order is not descending coarse cosine")

    res_cell = {m: {f"R@{k}": float(hit_cell[m][k].mean()) for k in S.KS_CELL}
                for m in methods}
    res_loc = {m: {f"top{k}@{int(t)}m": float(hit_loc[m][(k, t)].mean())
                   for k in S.KS_LOC for t in THRESHS} for m in methods}
    inv = dict(R25_equal=len({round(res_cell[m]["R@25"], 12)
                              for m in methods}) == 1,
               top25_5m_equal=True, top25_5m_rate=float(top25_hit.mean()),
               same_pool=True, same_positions=True, n_queries=int(n_done),
               n_clusters=int(len({str(s) for s in scene[:n_done]})),
               storage_order_is_coarse_cosine=True)

    keys = scene[:n_done].astype("<U32").reshape(-1, 1)
    metric_sel = ([("cell_R1", hit_cell, 1), ("cell_R2", hit_cell, 2),
                   ("cell_R3", hit_cell, 3), ("cell_R5", hit_cell, 5),
                   ("cell_R10", hit_cell, 10)]
                  + [(f"top{k}_{int(t)}m", hit_loc, (k, t))
                     for k in S.KS_LOC for t in THRESHS])
    comp = {}
    for target in ("original", "presence", "rank_fusion", "fsr"):
        comp[f"SCAR_minus_{METHOD_LABELS[target]}"] = {
            name: S.cluster_bootstrap(arr["scar_s0"][sel], arr[target][sel], keys)
            for name, arr, sel in metric_sel}
    stab = {}
    for tag in ("s1", "s2"):
        stab[f"scar_{tag}_minus_scar_s0"] = {
            name: S.cluster_bootstrap(arr[f"scar_{tag}"][sel],
                                      arr["scar_s0"][sel], keys)
            for name, arr, sel in metric_sel if isinstance(sel, int)}

    out = dict(
        status="OK",
        protocol=dict(name=S.PROTOCOL, feature_version=S.FEATURE_VERSION,
                      domain=a.domain, split="val", n_queries=int(n_done), k=K,
                      cluster_unit=d["cluster_unit"],
                      bootstrap=dict(resamples=S.RESAMPLES, seed=S.SEED, paired=True),
                      methods={k: METHOD_LABELS[k] for k in methods},
                      note="ordering only; coordinates are the candidates' Fine "
                           "predictions; FSR is the display name for sat_fine"),
        heads={tag: dict(path=path, sha256=S.sha256(path), selection=h.selection,
                         feature_version=h.feature_version)
               for tag, (h, path) in heads.items()},
        main_head="scar_s0",
        features=dict(path=a.features, sha256=S.sha256(a.features)),
        protocol_freeze_sha256=a.protocol_sha,
        coord_fix_sha256=a.coord_fix_sha,
        code_sha256={f: S.sha256(os.path.join(HERE, f))
                     for f in ("eval_val_coordfix_v1.py", "a_scar_common.py")},
        invariants=inv, results_cell=res_cell, results_loc=res_loc,
        comparisons=comp, stability=stab)
    S.write_json(out, os.path.join(a.out_dir, "metrics.json"))
    with open(os.path.join(a.out_dir, "per_query.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(a.out_dir, "table_cell_retrieval.csv"), "w",
              encoding="utf-8") as f:
        f.write("method," + ",".join(f"R@{k}" for k in S.KS_CELL) + "\n")
        for m in methods:
            f.write(m + "," + ",".join(f"{100*res_cell[m][f'R@{k}']:.2f}"
                                       for k in S.KS_CELL) + "\n")
    with open(os.path.join(a.out_dir, "table_localization.csv"), "w",
              encoding="utf-8") as f:
        f.write("method," + ",".join(f"top{k}@{int(t)}m"
                                     for t in THRESHS for k in S.KS_LOC) + "\n")
        for m in methods:
            f.write(m + "," + ",".join(
                f"{100*res_loc[m][f'top{k}@{int(t)}m']:.2f}"
                for t in THRESHS for k in S.KS_LOC) + "\n")

    print("invariants", inv, flush=True)
    for m in methods:
        print(f"{METHOD_LABELS[m]:20s} " +
              " ".join(f"R{k}={100*res_cell[m][f'R@{k}']:.2f}" for k in S.KS_CELL) +
              " | " + " ".join(
                  f"T{k}@{int(t)}={100*res_loc[m][f'top{k}@{int(t)}m']:.2f}"
                  for k in S.KS_LOC for t in THRESHS), flush=True)
    for name, c in comp.items():
        print(name, json.dumps({k: dict(delta=round(v["delta"] * 100, 2),
                                        ci=[round(x * 100, 2) for x in v["ci95"]],
                                        n_clusters=v["n_clusters"])
                                for k, v in c.items()}), flush=True)
    print("saved", a.out_dir, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
