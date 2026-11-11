# -*- coding: utf-8 -*-
"""TEST13: five-ranking TEST evaluation on the single Top-25 export.

Rankings on the SAME K=25 candidates and the SAME Fine offsets (defined
exactly as in COMMAND 12 `train12_eval.py`):

  original   storage / coarse-cosine order (arange(K))
  presence   P0 = stable argsort(-(cos + 0.5*sp))
  c_borda    canonical (p2_rank + sc_fine_rank)/(2*(K-1)), stable in P2 order
             (paper name: Rank Fusion)
  sat_fine   descending sat_fine, stable in P2 order (paper name: FSR / Fine-Sat)
  SCAR       frozen scorer seed 0 (epoch 26), rank within P2 order by logits

Only s0 is evaluated on TEST (operator decision); s1/s2 stay untouched.

EXPLORATORY: this is the formal second TEST read.  Nothing here may be used
to tune hyper-parameters or to re-select epochs, and the numbers must never be
merged with the first TEST read (`kitti_bb0_seed0_corev2b`), which used a
different readout.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test13_common as C  # noqa: E402

if C.CORE not in sys.path:
    sys.path.insert(0, C.CORE)

KS_CELL = (1, 2, 3, 5, 10, 25)
KS_LOC = (1, 2, 5, 10, 25)
THRESHS = (5.0, 10.0, 15.0)
MAIN = ("original", "presence", "c_borda", "sat_fine", "SCAR")
DISPLAY = {"original": "original", "presence": "Presence",
           "c_borda": "Rank Fusion", "sat_fine": "FSR",
           "SCAR": "SCAR"}


def load_head(path):
    import torch
    import torch.nn as nn
    ck = torch.load(path, map_location="cpu", weights_only=False)

    class Net(nn.Module):
        def __init__(self, nf, h1=64, h2=32):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(nf, h1), nn.ReLU(),
                                     nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    m = Net(len(ck["feature_names"]))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return ck, m, torch


def head_logits(ck, model, torch, feats):
    z = feats[..., ck["keep"]].astype(np.float64)
    z = (z - ck["mean"]) / ck["std"]
    z[..., ck["pinned"]] = 0.0
    with torch.no_grad():
        return model(torch.from_numpy(z.astype(np.float32))).numpy()


def rank_by_logits(logits, p2_idx):
    out = np.zeros_like(p2_idx)
    for i in range(len(logits)):
        o = p2_idx[i]
        out[i] = o[np.argsort(-logits[i][o], kind="stable")]
    return out


def canonical_cborda(sc_fine, p2_idx, K):
    n = len(sc_fine)
    out = np.zeros((n, K), np.int16)
    for i in range(n):
        p = p2_idx[i]
        sc_p = sc_fine[i, p].astype(np.float64)
        fine_order = np.argsort(-sc_p, kind="stable")
        fin_rank = np.empty(K, np.int64)
        fin_rank[fine_order] = np.arange(K)
        borda = (np.arange(K, dtype=np.float64) + fin_rank) / (2.0 * (K - 1))
        out[i] = p[np.argsort(borda, kind="stable")]
    return out


def cluster_bootstrap(hit_a, hit_b, keys_xy, seeds=2000):
    uniq, inv = np.unique(np.round(np.asarray(keys_xy, np.float64), 6),
                          axis=0, return_inverse=True)
    n_cl = len(uniq)
    diff = np.asarray(hit_a, np.float64) - np.asarray(hit_b, np.float64)
    sum_d = np.bincount(inv, weights=diff, minlength=n_cl)
    cnt = np.bincount(inv, minlength=n_cl).astype(np.float64)
    rng = np.random.RandomState(123)
    vals = np.empty(seeds)
    for s in range(seeds):
        pick = rng.randint(0, n_cl, n_cl)
        vals[s] = sum_d[pick].sum() / cnt[pick].sum()
    return dict(delta=float(diff.mean()),
                ci95=[float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
                n_clusters=int(n_cl),
                cluster_sizes=dict(min=float(cnt.min()), median=float(np.median(cnt)),
                                   max=float(cnt.max()), total=int(cnt.sum())),
                n_boot=int(seeds), seed=123,
                rule="GT pose (x,y) rounded to 6 dp; clusters drawn with "
                     "replacement; queries of drawn clusters aggregated")


def hits_for(order, cids, dists, gt_cell):
    n, K = order.shape
    hc = {k: np.zeros(n, np.int8) for k in KS_CELL}
    hl = {k: {t: np.zeros(n, np.int8) for t in THRESHS} for k in KS_LOC}
    for i in range(n):
        ranked = order[i]
        top_cells = cids[i][ranked]
        for k in KS_CELL:
            hc[k][i] = int(gt_cell[i] in set(top_cells[:k].tolist()))
        d = dists[i][ranked]
        for k in KS_LOC:
            m = float(np.min(d[:k]))
            for t in THRESHS:
                hl[k][t][i] = int(m <= t)
    return hc, hl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=os.path.join(C.ROUND, "out"))
    ap.add_argument("--cross-check", action="store_true",
                    help="re-derive the original-order localisation from the TEST "
                         "dataset poses (CPU) as an outside check")
    a = ap.parse_args()
    root = a.out_root
    fdir = os.path.join(root, "features")
    fpath = os.path.join(fdir, "test_features.npz")
    z = np.load(fpath, allow_pickle=True)
    feats = z["feats"].astype(np.float32)
    cids = z["cids"]
    n, K = feats.shape[0], feats.shape[1]
    if K != 25:
        raise SystemExit(f"TEST feature shape {feats.shape}")
    meta = C.read_json(fpath + ".meta.json")
    if int(meta["queries"]) != n:
        raise SystemExit("feature meta disagrees with the matrix")
    p2_idx = z["p2_idx"].astype(np.int64)
    p0_idx = z["p0_idx"].astype(np.int64)
    sc_fine = z["sc_fine"]
    sat_fine = z["sat_fine"]
    gt_cell = z["gt_cell"].astype(str)
    cell_scene = z["cell_scene"].astype(str)
    scene = z["scene"].astype(str)
    gt_pose = z["gt_pose_w"].astype(np.float64)
    pred_w = z["pred_w"].astype(np.float64)
    if not all(sorted(p2_idx[i].tolist()) == list(range(K)) for i in range(n)):
        raise SystemExit("p2_idx is not a permutation")

    dists = np.linalg.norm(pred_w - gt_pose[:, None, :2], axis=2)
    dists_scene = np.where(cell_scene == scene[:, None], dists, np.inf)

    orders = {"original": np.tile(np.arange(K, dtype=np.int64), (n, 1)),
              "presence": p0_idx,
              "c_borda": canonical_cborda(sc_fine, p2_idx, K).astype(np.int64)}
    o = np.zeros((n, K), np.int64)
    for i in range(n):
        o[i] = p2_idx[i][np.argsort(-sat_fine[i, p2_idx[i]], kind="stable")]
    orders["sat_fine"] = o

    p = C.SCAR_S0
    if not os.path.isfile(p):
        raise SystemExit(f"missing head {p}")
    got = C.sha256_file(p)
    if got != C.SCAR_S0_SHA256:
        raise SystemExit(f"SCAR head sha256 {got} != frozen {C.SCAR_S0_SHA256}")
    ck, m, torch = load_head(p)
    if [str(x) for x in ck["feature_names"]] != list(C.FEATURE_NAMES):
        raise SystemExit("head feature order differs from the frozen 23 names")
    if int(ck["selection"]["selected_epoch"]) != C.SCAR_SELECTED_EPOCH:
        raise SystemExit(f"head epoch {ck['selection']['selected_epoch']}")
    lg = head_logits(ck, m, torch, feats)
    orders["SCAR"] = rank_by_logits(lg, p2_idx)
    heads = {"SCAR": dict(path=p, sha256=got,
                          selected_epoch=int(ck["selection"]["selected_epoch"]),
                          selected_val=ck["selection"]["selected_val"],
                          old_rule_epoch=ck["selection"]["old_rule_epoch"],
                          norm_mean=[float(x) for x in ck["mean"]],
                          norm_std=[float(x) for x in ck["std"]],
                          norm_pinned=[bool(x) for x in ck["pinned"]],
                          norm_source="TRAIN valid lists only, stored in the head "
                                      "(refitting on TEST is forbidden)")}

    methods = list(orders)
    hits = {mm: hits_for(orders[mm], cids, dists_scene, gt_cell) for mm in methods}
    res_cell = {mm: {f"R@{k}": float(hits[mm][0][k].mean()) for k in KS_CELL}
                for mm in methods}
    res_loc = {mm: {f"top{k}@{int(t)}m": float(hits[mm][1][k][t].mean())
                    for k in KS_LOC for t in THRESHS} for mm in methods}

    inv = {
        "n_queries": int(n), "k": K,
        "R@25_equal_main": len({round(res_cell[mm]["R@25"], 12) for mm in MAIN}) == 1,
        "top25@5m_equal_main": len({int(hits[mm][1][25][5.0].sum()) for mm in MAIN}) == 1,
        "every_method_ranks_25": all(o.shape == (n, K) for o in orders.values()),
        "same_candidates_and_fine_positions": True,
        "full_denominator": True,
    }
    if not (inv["R@25_equal_main"] and inv["top25@5m_equal_main"]):
        raise SystemExit(f"@25 coverage invariant violated: {inv}")

    # per-scene breakdown (3 TEST scenes; descriptive, not a CI)
    scenes_sorted = sorted(set(scene.tolist()))
    per_scene = {}
    for sc in scenes_sorted:
        sel = scene == sc
        per_scene[sc] = {
            "n_queries": int(sel.sum()),
            "methods": {mm: {
                "cell_R@1": float(hits[mm][0][1][sel].mean()),
                "cell_R@5": float(hits[mm][0][5][sel].mean()),
                "top1@5m": float(hits[mm][1][1][5.0][sel].mean()),
                "top5@5m": float(hits[mm][1][5][5.0][sel].mean()),
            } for mm in MAIN},
        }
    if len(scenes_sorted) < 2:
        raise SystemExit("TEST must hold more than one scene")

    # paired pose-cluster bootstrap vs the four controls
    comp = {}
    pose_xy = gt_pose[:, :2]
    for mm in MAIN[1:]:
        comp[mm] = {}
        for base in ("original", "presence", "c_borda", "sat_fine"):
            if base == mm:
                continue
            d = {}
            d["cell_R1"] = cluster_bootstrap(hits[mm][0][1], hits[base][0][1], pose_xy)
            d["cell_R5"] = cluster_bootstrap(hits[mm][0][5], hits[base][0][5], pose_xy)
            d["cell_R25"] = cluster_bootstrap(hits[mm][0][25], hits[base][0][25], pose_xy)
            d["top1_5m"] = cluster_bootstrap(hits[mm][1][1][5.0], hits[base][1][1][5.0],
                                             pose_xy)
            d["top5_5m"] = cluster_bootstrap(hits[mm][1][5][5.0], hits[base][1][5][5.0],
                                             pose_xy)
            d["top25_5m"] = cluster_bootstrap(hits[mm][1][25][5.0],
                                              hits[base][1][25][5.0], pose_xy)
            comp[mm][f"vs_{base}"] = d

    # per-query record (one row per query; orders and distances included)
    pq = []
    for i in range(n):
        pq.append(dict(
            q=i, scene=str(scene[i]), gt_cell=str(gt_cell[i]),
            gt_pose_w=[float(x) for x in gt_pose[i]],
            ids=[str(c) for c in cids[i]],
            dist=[float(x) for x in dists[i]],
            dist_scene_masked=[float(x) for x in dists_scene[i]],
            order={mm: [str(cids[i, j]) for j in orders[mm][i]] for mm in methods},
            hit={mm: dict(
                R1=int(hits[mm][0][1][i]), R5=int(hits[mm][0][5][i]),
                R25=int(hits[mm][0][25][i]),
                top1_5m=int(hits[mm][1][1][5.0][i]),
                top5_5m=int(hits[mm][1][5][5.0][i]),
                top25_5m=int(hits[mm][1][25][5.0][i])) for mm in methods},
        ))
    eval_dir = os.path.join(root, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    pq_path = os.path.join(eval_dir, "per_query_test.jsonl")
    C.write_jsonl_atomic(pq_path, pq)

    # first independent aggregate: rebuild the hits from the jsonl alone
    iv = {}
    for mm in methods:
        c1 = c5 = c25 = t1 = t5 = t25 = 0
        for r in pq:
            ids = r["ids"]
            order = r["order"][mm]
            ds = r["dist_scene_masked"]
            idx = [ids.index(c) for c in order]
            c1 += int(r["gt_cell"] in order[:1])
            c5 += int(r["gt_cell"] in order[:5])
            c25 += int(r["gt_cell"] in order[:25])
            t1 += int(min(ds[j] for j in idx[:1]) <= 5.0)
            t5 += int(min(ds[j] for j in idx[:5]) <= 5.0)
            t25 += int(min(ds[j] for j in idx[:25]) <= 5.0)
        iv[mm] = dict(R1=c1 / n, R5=c5 / n, R25=c25 / n,
                      top1_5m=t1 / n, top5_5m=t5 / n, top25_5m=t25 / n)
        ok = (abs(iv[mm]["R1"] - res_cell[mm]["R@1"]) < 1e-12
              and abs(iv[mm]["R5"] - res_cell[mm]["R@5"]) < 1e-12
              and abs(iv[mm]["top1_5m"] - res_loc[mm]["top1@5m"]) < 1e-12
              and abs(iv[mm]["top25_5m"] - res_loc[mm]["top25@5m"]) < 1e-12)
        iv[mm]["matches_primary"] = bool(ok)
        if not ok:
            raise SystemExit(f"independent verification mismatch for {mm}")
    iv["verdict"] = "PASS"

    cross = None
    if a.cross_check:
        os.chdir(C.CORE)
        from torch_geometric.transforms import FixedPoints
        from dataloading.kitti360pose.cells import Kitti360CoarseDatasetMulti
        from datapreparation.kitti360pose.utils import SCENE_NAMES_TEST
        cfg = C.read_json(C.FINE_CFG)
        ds = Kitti360CoarseDatasetMulti(cfg["base_path"], SCENE_NAMES_TEST,
                                        FixedPoints(cfg["pointnet_numpoints"]),
                                        shuffle_hints=False, flip_poses=False)
        cells_dict = {str(c.id): c for c in ds.all_cells}
        poses = ds.all_poses
        if len(poses) != n:
            raise SystemExit(f"cross-check dataset has {len(poses)} poses, features {n}")
        gt_diff = max(float(np.max(np.abs(
            np.asarray(poses[i].pose_w[0:2], np.float64) - gt_pose[i, :2])))
            for i in range(n))
        acc = {(k, 5.0): 0 for k in (1, 5, 10, 25)}
        for i in range(n):
            ranked = [str(cids[i, j]) for j in orders["original"][i]]
            d = np.linalg.norm(
                pred_w[i] - np.asarray(poses[i].pose_w[0:2], np.float64), axis=1)
            d[np.array([str(cells_dict[c].scene_name) for c in ranked])
              != str(poses[i].scene_name)] = np.inf
            for k in (1, 5, 10, 25):
                acc[(k, 5.0)] += int(np.min(d[:k]) <= 5.0)
        cross = {f"top{k}_5m": acc[(k, 5.0)] / n for k in (1, 5, 10, 25)}
        cross["matches_primary"] = bool(
            all(abs(cross[f"top{k}_5m"] - res_loc["original"][f"top{k}@5m"]) < 1e-12
                for k in (1, 5, 10, 25)))
        cross["gt_pose_max_abs_diff_vs_dataset"] = gt_diff
        cross["n_poses_in_dataset"] = len(poses)
        if gt_diff > 1e-6:
            raise SystemExit(f"exported GT poses differ from dataset: {gt_diff}")
        if not cross["matches_primary"]:
            raise SystemExit(f"core cross-check mismatch: {cross}")

    train_log = C.read_json(C.SCAR_TRAIN_LOG)
    sel = train_log["runs"]["scar23_s0"]["selection"]
    core = dict(
        status="OK", protocol="TEST13 exploratory second TEST read",
        split="test", exploratory=True, test_read_number=2,
        label_display={mm: DISPLAY[mm] for mm in MAIN},
        backbone=dict(coarse_sha256=C.EXPECT["coarse_sha256"],
                      coarse_epoch=C.EXPECT["coarse_epoch"],
                      fine_sha256=C.EXPECT["fine_sha256"],
                      fine_epoch=C.EXPECT["fine_epoch"],
                      coarse_light_digest=C.EXPECT["coarse_light_digest"],
                      fine_light_digest=C.EXPECT["fine_light_digest"]),
        test_export=dict(
            meta_sha256=C.sha256_file(C.TEST_META),
            fingerprint_sha256=C.read_json(C.TEST_META)["fingerprint"]["fingerprint_sha256"],
            n_queries=int(n), n_cells=int(meta["cells_gallery"])),
        features_sha256=C.sha256_file(fpath),
        heads=heads,
        selection=sel.get("selection_rule"),
        selection_history=sel,
        methods={mm: DISPLAY[mm] for mm in MAIN},
        label_rule="same scene and ||pred_w - gt_pose_xy|| <= 5 m; GT only for "
                   "labels/metrics, never a feature",
        bootstrap="paired pose-cluster bootstrap (2000, seed 123), GT pose (x,y) "
                  "6 dp; TEST holds 3 scenes, so this is a per-query cluster "
                  "interval, NOT a cross-scene generalisation claim",
        test_read=True, test_read_exploratory=True,
        head_seeds_on_test=[0],
        backbone_retrained=False, head_retrained=False,
        candidates_regenerated=False, epochs_reselected=False,
        forbidden_uses=["hyper-parameter tuning", "epoch re-selection",
                        "merging with the first TEST read"],
    )
    h = __import__("hashlib").sha256(
        json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()
    out = dict(status="OK", config_hash=h, protocol=core, invariants=inv,
               queries=int(n), k=K, n_scenes=len(scenes_sorted),
               metrics={mm: dict(cell=res_cell[mm], loc=res_loc[mm]) for mm in methods},
               per_scene=per_scene,
               comparisons=comp,
               independent_verification=iv,
               core_cross_check=cross,
               note="methods share the same Top-25 candidates and Fine positions; "
                    "R@25 and top25@5m are invariants, not performance. "
                    "EXPLORATORY second TEST read.")
    metrics_p = os.path.join(eval_dir, "metrics_test.json")
    verify_p = os.path.join(eval_dir, "VERIFY_TEST.json")
    C.write_json_atomic(metrics_p, out)
    C.write_json_atomic(verify_p, dict(
        status="PASS", split="test", queries=int(n),
        independent_aggregate=iv, invariants=inv, core_cross_check=cross,
        check_method="hits rebuilt from eval/per_query_test.jsonl alone; core "
                     "distance/scene-mask cross-check on the original order "
                     "(CPU, TEST dataset poses)",
        metrics_sha256=C.sha256_file(metrics_p),
        per_query_sha256=C.sha256_file(pq_path)))
    C.write_json_atomic(os.path.join(eval_dir, "FREEZE_TEST13.json"), {
        "created_at": C.now(), "config_hash": h, "protocol": core,
        "metrics_sha256": C.sha256_file(metrics_p),
        "per_query_sha256": C.sha256_file(pq_path),
        "verification_sha256": C.sha256_file(verify_p),
        "authorisation": "exploratory second TEST read, explicitly approved; "
                         "no further TEST evaluation is authorised by this record",
    })
    print(json.dumps({mm: {"cell": res_cell[mm], "loc": res_loc[mm]} for mm in MAIN},
                     indent=1))
    print(json.dumps({"invariants": inv, "independent": iv["verdict"],
                      "cross_check": None if cross is None else cross["matches_primary"],
                      "config_hash": h, "n_queries": int(n)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
