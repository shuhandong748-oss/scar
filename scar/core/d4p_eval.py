#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate the frozen candidate cache under every method for one domain/split.

All methods share the same queries, the same 25 candidates, and the same Fine
coordinates; only the ordering changes:

  original      coarse cosine order (storage order of the export)
  Presence      stable argsort descending (cos + 0.5*sp)  = P0
  Rank Fusion   ascending (p2_rank + sc_fine_rank)/(2*(K-1)), stable P2
  FSR           descending sat_fine, stable P2 (internal field sat_fine)
  SCAR          main head s0 logits, descending inside P2; s1/s2 stability only

Metrics: Cell R@k vs pose.cell_id for k in {1,2,3,5,10,25}; localization
Top-k@5/10/15 m using exactly the core's rule (pred_w distance, cross-scene
candidates set to inf).  Paired differences use a paired cluster bootstrap
(2000 resamples, seed 123) over the true scene/room identity.

TEST is only read when the export directory carries a *completed* TEST marker.
This script never writes into the export directory.
"""
import argparse
import hashlib
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
SEED = 123
RESAMPLES = 2000
METHOD_LABELS = {
    "original": "original",
    "presence": "Presence",
    "rank_fusion": "Rank Fusion",
    "fsr": "FSR",
    "scar_s0": "SCAR",
    "scar_s1": "SCAR s1 (stability)",
    "scar_s2": "SCAR s2 (stability)",
}


def sha256(path):
    return S.sha256(path)


def cluster_bootstrap(hit_a, hit_b, keys, resamples=RESAMPLES, seed=SEED):
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    n_cl = len(uniq)
    diff = np.asarray(hit_a, np.float64) - np.asarray(hit_b, np.float64)
    sum_d = np.bincount(inv, weights=diff, minlength=n_cl)
    cnt = np.bincount(inv, minlength=n_cl).astype(np.float64)
    rng = np.random.RandomState(seed)
    vals = np.empty(resamples)
    for s in range(resamples):
        pick = rng.randint(0, n_cl, n_cl)
        vals[s] = sum_d[pick].sum() / cnt[pick].sum()
    return dict(delta=float(diff.mean()),
                ci95=[float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
                n_clusters=int(n_cl), n_units=int(len(diff)),
                wins=int((diff > 0).sum()), losses=int((diff < 0).sum()),
                ties=int((diff == 0).sum()),
                cluster_sizes=[int(x) for x in cnt.tolist()])


class Head:
    def __init__(self, path):
        import torch
        import torch.nn as nn
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.mean = ck["mean"].astype(np.float32)
        self.std = ck["std"].astype(np.float32)
        self.selection = ck["selection"]
        self.legacy_a = ck.get("legacy_a")
        self.nf = len(ck["feature_names"])
        K = ck["config"]["k"]
        norms = ck["config"]["norms"]
        if self.nf != 23 or K != 25:
            raise SystemExit(f"{path}: not a canonical-23/K25 head")
        if norms != dict(pool_norm=24, rank_norm=24, tied_norm=25):
            raise SystemExit(f"{path}: unexpected norms {norms}")

        class Net(nn.Module):
            def __init__(self, nf):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(nf, 64), nn.ReLU(),
                                         nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

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
    ap.add_argument("--domain", required=True, choices=["npm3d", "s3dis"])
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--export_dir", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--heads_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--protocol_sha", required=True)
    ap.add_argument("--freeze_sha", default=None)
    ap.add_argument("--allow_test", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    for name in ("export_dir", "features", "heads_dir", "out_dir"):
        setattr(a, name, os.path.abspath(getattr(a, name)))
    if a.split == "test":
        if not a.allow_test:
            raise SystemExit("TEST is gated: pass --allow_test after the freeze")
        marker_path = os.path.join(a.export_dir, "TEST_READ_ONCE.json")
        if not os.path.isfile(marker_path):
            raise SystemExit("no TEST marker in the export dir")
        m = S.read_json(marker_path)
        if m.get("status") != "completed":
            raise SystemExit(f"TEST export is not completed: {m.get('status')}")
        if m.get("protocol_freeze_sha256") != a.protocol_sha:
            raise SystemExit("TEST marker protocol sha mismatch")
        if a.freeze_sha and m.get("freeze_list_sha256") != a.freeze_sha:
            raise SystemExit("TEST marker freeze sha mismatch")
    os.makedirs(a.out_dir, exist_ok=True)
    existing = [n for n in os.listdir(a.out_dir) if n != "attack-note.txt"]
    if existing:
        raise SystemExit(f"{a.out_dir} is not empty: {existing[:5]}")

    d = S.domain_cfg(a.domain)
    gate = S.load_gate()
    export_meta = S.read_json(os.path.join(a.export_dir, "EXPORT_META.json"))
    if export_meta["split"] != a.split or export_meta["domain"] != a.domain:
        raise SystemExit("export meta domain/split mismatch")
    if a.split == "test":
        marker_content = S.read_json(os.path.join(a.export_dir, "TEST_READ_ONCE.json"))
        if marker_content.get("cache_sha256", {}).get("candidates.npz") != \
                export_meta.get("cache_sha256", {}).get("candidates.npz"):
            raise SystemExit("TEST marker cache sha does not match the export")

    sys.path.insert(0, d["vendor"])
    import features_fix as F

    z = np.load(os.path.join(a.export_dir, "candidates.npz"), allow_pickle=True)
    ids = np.asarray(z["cell_ids"])
    cell_scene = np.asarray(z["cell_scene"]) if "cell_scene" in z.files else None
    offsets = np.asarray(z["offsets"], np.float64)
    pred_w = np.asarray(z["pred_w"], np.float64)
    gt_cell = [str(x) for x in z["gt_cell"]]
    gt_pose_w = np.asarray(z["gt_pose_w"], np.float64)
    scene = np.asarray([str(x) for x in z["scene"]])
    n, K = ids.shape
    if K != 25:
        raise SystemExit("cache is not K=25")

    feat = np.load(a.features, allow_pickle=True)
    feats = feat["feats"]
    if len(feats) != n:
        raise SystemExit(f"features {len(feats)} != export {n}")
    p2 = feat["p2_idx"].astype(np.int64)
    cos = np.asarray(feat["cos"], np.float64)
    sp = np.asarray(feat["sp"], np.float64)
    sc_fine = np.asarray(feat["sc_fine"], np.float64)
    sat_fine = np.asarray(feat["sat_fine"], np.float64)
    if [str(x) for x in feat["gt_cell"]] != gt_cell:
        raise SystemExit("feature gt_cell order mismatch")
    if [[str(c) for c in row] for row in feat["cell_ids"]] != \
            [[str(c) for c in row] for row in ids.tolist()]:
        raise SystemExit("feature candidate order mismatch")
    if list(feat["feature_names"]) != list(F.FEATURE_NAMES):
        raise SystemExit("feature names are not canonical 23")

    heads = {}
    for tag, seed in (("s0", 0), ("s1", 1), ("s2", 2)):
        path = os.path.join(a.heads_dir, f"ht23_{a.domain}_s{seed}.pt")
        if not os.path.isfile(path):
            raise SystemExit(f"missing head {path}")
        heads[tag] = (Head(path), path)

    n_done = min(n, a.limit) if a.limit else n
    methods = ["original", "presence", "rank_fusion", "fsr", "scar_s0",
               "scar_s1", "scar_s2"]
    hit_cell = {m: {k: np.zeros(n_done, np.int8) for k in KS_CELL} for m in methods}
    hit_loc = {m: {(k, t): np.zeros(n_done, np.int8) for k in KS_LOC for t in THRESHS}
               for m in methods}
    rows = []
    max_rank_diff = 0.0
    max_order_error = 0

    for i in range(n_done):
        p2i = np.asarray(p2[i], np.int64)
        # Rank Fusion: ascending mean rank inside P2
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
            # cell recall
            for k in KS_CELL:
                hit_cell[m][k][i] = int(gt_cell[i] in ranked_ids[:k])
            # localization, core rule with cross-scene inf
            dist = np.linalg.norm(pred_w[i][o] - gt_pose_w[i][None, :], axis=1)
            if cell_scene is not None:
                cs = [str(cell_scene[i][j]) for j in o]
            else:
                cs = [c.split("_")[0] for c in ranked_ids]
            dist[np.asarray(cs) != scene[i]] = np.inf
            for k in KS_LOC:
                for t in THRESHS:
                    hit_loc[m][(k, t)][i] = int(np.min(dist[:k]) <= t)

        # invariants: every method sees the same pool / decode
        if len({int(hit_cell[m][25][i]) for m in methods}) != 1:
            raise SystemExit(f"R@25 differs across methods at query {i}")
        if len({int(hit_loc[m][(25, 5.0)][i]) for m in methods}) != 1:
            raise SystemExit(f"Top25@5m differs across methods at query {i}")
        # storage order must equal descending cosine
        if not np.array_equal(np.asarray(orders["original"]),
                              np.argsort(-cos[i], kind="stable")):
            max_order_error += 1

        rows.append(dict(q=i, scene=str(scene[i]), gt_cell=gt_cell[i],
                         ids=[str(x) for x in ids[i]],
                         orders={m: [int(x) for x in o] for m, o in orders.items()},
                         logits={k: [float(x) for x in v] for k, v in logits.items()},
                         hit_cell={m: {f"R{k}": int(hit_cell[m][k][i]) for k in KS_CELL}
                                   for m in methods},
                         hit_loc={m: {f"top{k}@{int(t)}m": int(hit_loc[m][(k, t)][i])
                                      for k in KS_LOC for t in THRESHS}
                                  for m in methods}))

    if max_order_error:
        raise SystemExit("storage order is not descending coarse cosine")

    res_cell = {m: {f"R@{k}": float(hit_cell[m][k].mean()) for k in KS_CELL}
                for m in methods}
    res_loc = {m: {f"top{k}@{int(t)}m": float(hit_loc[m][(k, t)].mean())
                   for k in KS_LOC for t in THRESHS} for m in methods}

    inv = dict(R25_equal=len({round(res_cell[m]["R@25"], 12) for m in methods}) == 1,
               top25_5m_equal=len({int(hit_loc[m][(25, 5.0)].sum()) for m in methods}) == 1,
               same_pool=True, same_positions=True, n_queries=int(n_done),
               n_clusters=int(len({str(s) for s in scene[:n_done]})),
               storage_order_is_coarse_cosine=True)

    keys = scene[:n_done].astype("<U32").reshape(-1, 1)
    metric_sel = ([("cell_R1", hit_cell, 1), ("cell_R2", hit_cell, 2),
                   ("cell_R3", hit_cell, 3), ("cell_R5", hit_cell, 5),
                   ("cell_R10", hit_cell, 10)]
                  + [(f"top{k}_{int(t)}m", hit_loc, (k, t))
                     for k in KS_LOC for t in THRESHS])
    comp = {}
    for target in ("original", "presence", "rank_fusion", "fsr"):
        c = {}
        for name, arr, sel in metric_sel:
            c[name] = cluster_bootstrap(arr["scar_s0"][sel], arr[target][sel], keys)
        comp[f"SCAR_minus_{METHOD_LABELS[target]}"] = c
    # stability: s1/s2 vs s0 (no new selection)
    stab = {}
    for tag in ("s1", "s2"):
        stab[f"scar_{tag}_minus_scar_s0"] = {
            name: cluster_bootstrap(hit_cell[f"scar_{tag}"][sel],
                                    hit_cell["scar_s0"][sel], keys)
            for name, arr, sel in metric_sel if isinstance(sel, int)}

    out = dict(
        status="OK", protocol=dict(
            name="CORR-D4P-SCAR-FORMAL-v1", domain=a.domain, split=a.split,
            n_queries=int(n_done), k=K, cluster_unit=d["cluster_unit"],
            bootstrap=dict(resamples=RESAMPLES, seed=SEED, paired=True),
            methods={k: METHOD_LABELS[k] for k in methods},
            note="SCAR changes ordering only; final coordinates are the candidates' "
                 "existing Fine predictions; FSR is the display name for sat_fine"),
        heads={tag: dict(path=path, sha256=sha256(path), selection=h.selection,
                         legacy_a=h.legacy_a) for tag, (h, path) in heads.items()},
        main_head="scar_s0",
        features=dict(path=a.features, sha256=sha256(a.features)),
        export_dir=a.export_dir,
        export_meta_sha256=sha256(os.path.join(a.export_dir, "EXPORT_META.json")),
        cache_sha256=export_meta.get("cache_sha256"),
        dataset_manifest_sha256=export_meta.get("manifest_sha256"),
        gate_binding=gate, protocol_freeze_sha256=a.protocol_sha,
        freeze_list_sha256=a.freeze_sha,
        code_sha256={f: sha256(os.path.join(HERE, f))
                     for f in ("d4p_eval.py", "scar_common.py")},
        invariants=inv, results_cell=res_cell, results_loc=res_loc,
        comparisons=comp, stability=stab,
        test_marker=(S.read_json(os.path.join(a.export_dir, "TEST_READ_ONCE.json"))
                     if a.split == "test" else None),
    )
    S.write_json(out, os.path.join(a.out_dir, "metrics.json"))
    with open(os.path.join(a.out_dir, "per_query.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(a.out_dir, "table_cell_retrieval.csv"), "w",
              encoding="utf-8") as f:
        f.write("method," + ",".join(f"R@{k}" for k in KS_CELL) + "\n")
        for m in methods:
            f.write(m + "," + ",".join(f"{100*res_cell[m][f'R@{k}']:.2f}"
                                       for k in KS_CELL) + "\n")
    with open(os.path.join(a.out_dir, "table_localization.csv"), "w",
              encoding="utf-8") as f:
        f.write("method," + ",".join(f"top{k}@{int(t)}m"
                                     for t in THRESHS for k in KS_LOC) + "\n")
        for m in methods:
            f.write(m + "," + ",".join(f"{100*res_loc[m][f'top{k}@{int(t)}m']:.2f}"
                                       for t in THRESHS for k in KS_LOC) + "\n")

    print("invariants", inv, flush=True)
    for m in methods:
        print(f"{METHOD_LABELS[m]:20s} " +
              " ".join(f"R{k}={100*res_cell[m][f'R@{k}']:.2f}" for k in KS_CELL) +
              " | " + " ".join(f"T{k}@{int(t)}={100*res_loc[m][f'top{k}@{int(t)}m']:.2f}"
                               for k in KS_LOC for t in THRESHS), flush=True)
    for name, c in comp.items():
        print(name, json.dumps({k: dict(delta=round(v["delta"] * 100, 2),
                                        ci=[round(x * 100, 2) for x in v["ci95"]],
                                        n_clusters=v["n_clusters"])
                                for k, v in c.items()}), flush=True)
    print("saved", a.out_dir, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
