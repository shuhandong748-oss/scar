"""Appended exploratory TEST evaluation: MambaPlace x KITTI, NEW TRAJECTORY s0 ep5.

AUTHORIZED ONE-OFF EXCEPTION to R059_PROTOCOL.md section 14.4, granted by the user
on 2026-11-04.  This is a NEW REGISTRATION (new version, new records, new run dir);
the old TEST marker, old results and old protocol are left untouched.

Scope:
  * frozen TEST caches only (4 files, SHA256 checked before opening)
  * formal main head s0 ep5 (B-rule selection) ONLY; s1/s2 sensitivity heads not run
  * same candidate pool, same Fine offsets for every method
  * methods: original coarse order, Presence (P0), Rank Fusion (C-Borda), FSR
    (sat_fine), formal SCAR head
  * no Coarse/Fine forward, no candidate re-export, no raw TEST data read,
    no training, no scorer retraining, no cloud connection.

Before reporting the new head this script reproduces the four frozen control
methods and byte-compares their per-query hit vectors with the old frozen TEST
metrics (runs/eval_test/metrics.json).  Any mismatch aborts without writing new
metrics.

Metric definitions (frozen):
  cell R@k  : GT cell id is among the first k candidates in the method order.
  topk@tm  : min over first k predicted world positions of
              ||pred_w - gt_pose_w|| <= t, with cross-scene candidates = inf.
              pred_w is the frozen decoded position bbox_w[:2] + offset*cell_size,
              stored in candidates.npz.  The old verifier established 0 boundary
              mismatches between this stored decode and the official
              calc_sample_accuracies on this exact cache.
  denominator: all 11,505 TEST queries (no dropping).
Tie rules (frozen): stable argsort; ties retain P2 order for head/Borda/sat_fine
and candidate (coarse) order for the P0 Presence order.
Statistics (frozen): paired per-query differences, pose-cluster bootstrap,
  cluster key = scene + round(gt_pose_w.xy, 6), 2000 resamples, seed 123.
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)
VENDOR = os.path.join(HERE, "vendor")
sys.path.insert(0, VENDOR)
import features_fix as F  # noqa: E402

RECOVERY = r"<SCAR_ROOT>\E_WEIGHT_CACHE_RECOVERY_20261103"
CACHE = os.path.join(RECOVERY, "cache", "test")
OLD_METRICS = os.path.join(RECOVERY, "runs", "eval_test", "metrics.json")
FORMAL = r"<SCAR_ROOT>\FORMAL_SCAR_20261103_MP_KITTI"
HEAD = os.path.join(FORMAL, "heads", "selected", "ht23_mp_formal_s0.pt")
PER_EPOCH = os.path.join(FORMAL, "heads", "per_epoch", "s0_ep05.pt")
NORM_NPY = os.path.join(FORMAL, "protocol", "normalization.npy")
NORM_JSON = os.path.join(FORMAL, "protocol", "normalization.json")

HEAD_NAME = "mp_formal_s0_ep5"
METHODS = ["text2loc", "presence", "c_borda", "sat_fine", HEAD_NAME]
CONTROLS = ["text2loc", "presence", "c_borda", "sat_fine"]
KS_CELL = (1, 2, 3, 5, 10, 25)
THRESHS = (5, 10, 15)
SEED = 123
RESAMPLES = 2000
N_EXPECTED = 11505
K_EXPECTED = 25
PRIMARY = ["cell_R1", "top1_5m", "top5_5m", "top10_5m"]
METRICS = ([f"cell_R{k}" for k in KS_CELL]
           + [f"top{k}_{t}m" for t in THRESHS for k in (1, 5, 10)])


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cluster_keys(scene, pose_w):
    sc = sorted(set(str(x) for x in scene))
    code = {s: i for i, s in enumerate(sc)}
    return np.stack([
        np.array([code[str(s)] for s in scene], np.float64),
        np.round(np.asarray(pose_w, np.float64)[:, 0], 6),
        np.round(np.asarray(pose_w, np.float64)[:, 1], 6),
    ], axis=1)


def cluster_bootstrap(diff, keys, seeds=RESAMPLES, seed=SEED):
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    n_cl = len(uniq)
    diff = np.asarray(diff, np.float64)
    sum_d = np.bincount(inv, weights=diff, minlength=n_cl)
    cnt = np.bincount(inv, minlength=n_cl).astype(np.float64)
    rng = np.random.RandomState(seed)
    vals = np.empty(seeds)
    for s in range(seeds):
        pick = rng.randint(0, n_cl, n_cl)
        vals[s] = sum_d[pick].sum() / cnt[pick].sum()
    ci = np.percentile(vals, [2.5, 97.5])
    return dict(delta=float(diff.mean()), ci95=[float(ci[0]), float(ci[1])],
                ci_excludes_zero=bool(ci[0] > 0 or ci[1] < 0),
                cluster_key="scene + round(gt_pose_w.xy, 6)", n_clusters=int(n_cl),
                resamples=int(seeds), seed=int(seed),
                win=int((diff > 0).sum()), loss=int((diff < 0).sum()),
                tie=int((diff == 0).sum()), n=int(diff.size))


class Head:
    def __init__(self, path):
        import torch
        import torch.nn as nn

        self.torch = torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.ck = ck
        self.mean = np.asarray(ck["mean"], np.float32)
        self.std = np.asarray(ck["std"], np.float32)
        self.selection = ck["selection"]
        self.feature_names = [str(x) for x in ck["feature_names"]]
        self.nf = len(self.feature_names)

        class Net(nn.Module):
            def __init__(self, nf):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(nf, 64), nn.ReLU(),
                                         nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

            def forward(self, z):
                return self.net(z).squeeze(-1)

        self.model = Net(self.nf)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def logits(self, feats):
        x = (feats - self.mean) / self.std
        with self.torch.no_grad():
            return self.model(self.torch.from_numpy(x.astype(np.float32))).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(RUN, "results"))
    cli = ap.parse_args()
    out_dir = os.path.abspath(cli.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    import torch
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.manual_seed(0)

    # ---------------- frozen input identity (checked before payload access)
    cache_paths = {k: os.path.join(CACHE, f"{k}.npz")
                   for k in ("coarse", "fine", "candidates", "feats")}
    inputs = {
        "test_caches": {k: dict(path=v, sha256=sha256(v), bytes=os.path.getsize(v))
                        for k, v in cache_paths.items()},
        "head": dict(path=HEAD, sha256=sha256(HEAD), bytes=os.path.getsize(HEAD)),
        "per_epoch_ep05": dict(path=PER_EPOCH, sha256=sha256(PER_EPOCH)),
        "normalization_npy": dict(path=NORM_NPY, sha256=sha256(NORM_NPY)),
        "normalization_json": dict(path=NORM_JSON, sha256=sha256(NORM_JSON)),
        "old_test_metrics": dict(path=OLD_METRICS, sha256=sha256(OLD_METRICS)),
        "eval_code_sha256": sha256(os.path.abspath(__file__)),
        "vendor_sha256": {f: sha256(os.path.join(VENDOR, f))
                          for f in ("features_fix.py", "common.py", "sa_geom.py")},
    }
    expected = {
        "coarse": "45474397bfa7dd1d000a410a97a23d6d2105961fda972b0801102824985132e0",
        "fine": "23f1f02cd7504a77212c078ce61b3e7d394383221038e8cf42670edf4589db22",
        "candidates": "699921e22683f60c32b4ad8ee46f78cc07dd39bd7b5728ceb2c46d88db4f2031",
        "feats": "66e300081e48901fc0b0d697e90e640bdaf0b92a14bf89cd530058f143ea393c",
    }
    for k, v in expected.items():
        assert inputs["test_caches"][k]["sha256"] == v, k
    assert inputs["head"]["sha256"] == \
        "dc153c956a3166321dcbe469e4470b06e9de1e71c462d07a528d368c4a19eaac"
    assert inputs["per_epoch_ep05"]["sha256"] == \
        "11e6d89d8787b9073086f2e5caf6341fd3ccd02e03d7190309b88c31a82c109e"
    assert inputs["normalization_npy"]["sha256"] == \
        "c6025f96abbb2df901a1176b34e4401aaaa2a6e4992f5838ad2c6ac75fcc21ff"

    # ---------------- head + normalization binding
    h = Head(HEAD)
    per = torch.load(PER_EPOCH, map_location="cpu", weights_only=False)
    assert h.feature_names == list(F.FEATURE_NAMES)
    assert h.mean.shape == (23,) and h.std.shape == (23,)
    assert int(h.selection["epoch"]) == 5
    assert "top1@5m" in json.dumps(h.selection["rule"])
    assert abs(float(h.selection["key"][0]) - 0.6118606840288673) < 1e-15
    sd_new, sd_old = h.ck["state_dict"], per["state_dict"]
    assert list(sd_new) == list(sd_old)
    for k in sd_new:
        assert torch.equal(sd_new[k], sd_old[k]), k
    norm = np.load(NORM_NPY)
    norm_obj = json.load(open(NORM_JSON, encoding="utf-8"))
    assert norm.shape == (3, 23)
    assert np.array_equal(h.mean, np.asarray(norm[0], np.float32))
    assert np.array_equal(h.std, np.asarray(norm[1], np.float32))
    assert np.allclose(h.mean, np.asarray(norm_obj["mean"], np.float32), atol=0,
                       rtol=0)
    assert np.allclose(h.std, np.asarray(norm_obj["std"], np.float32), atol=0,
                       rtol=0)
    assert h.feature_names == list(norm_obj["feature_names"])
    assert len(h.feature_names) == 23
    assert norm_obj["source"].startswith("TRAIN valid lists only")
    assert norm_obj["pinned"][19] is True and norm_obj["pinned"][20] is True
    assert float(norm_obj["mean"][19]) == 3.0 and float(norm_obj["std"][19]) == 1.0

    # ---------------- payload (authorized after freeze)
    z = np.load(cache_paths["candidates"], allow_pickle=True)
    ids = np.asarray(z["cell_ids"])
    offsets = np.asarray(z["offsets"], np.float64)
    pred_w = np.asarray(z["pred_w"], np.float64)
    gt_cell = [str(x) for x in z["gt_cell"]]
    scene = np.asarray([str(x) for x in z["scene"]])
    qids = np.asarray([str(x) for x in z["query_ids"]])
    src_idx = np.asarray(z["source_pose_index"], np.int64)
    n, K = ids.shape
    assert K == K_EXPECTED and n == N_EXPECTED
    assert len(set(qids.tolist())) == n
    assert np.array_equal(src_idx, np.arange(n))
    assert all(len(set(row)) == K for row in ids.tolist())

    d = np.load(cache_paths["feats"], allow_pickle=True)
    feats = np.asarray(d["feats"], np.float32)
    p2_all = np.asarray(d["p2_idx"], np.int64)
    cos_all = np.asarray(d["cos"], np.float64)
    sp_all = np.asarray(d["sp"], np.float64)
    sc_center_all = np.asarray(d["sc_center"], np.float64)
    sc_fine_all = np.asarray(d["sc_fine"], np.float64)
    sat_fine_all = np.asarray(d["sat_fine"], np.float64)
    gtw = np.asarray(d["gt_pose_w"], np.float64)
    assert feats.shape == (n, K, 23)
    assert np.isfinite(feats).all()
    assert list(d["feature_names"]) == list(F.FEATURE_NAMES)
    assert [str(x) for x in d["gt_cell"]] == gt_cell
    assert [str(x) for x in d["scene"]] == scene.tolist()
    assert [str(x) for x in d["query_ids"]] == qids.tolist()
    assert [[str(c) for c in row] for row in d["cell_ids"]] == \
           [[str(c) for c in row] for row in ids]
    assert np.allclose(np.asarray(d["offsets"], np.float64), offsets, atol=0, rtol=0)
    assert np.allclose(np.asarray(d["pred_w"], np.float64), pred_w, atol=0, rtol=0)
    assert np.allclose(np.asarray(d["source_pose_index"], np.int64), src_idx,
                       atol=0, rtol=0)
    gtw64 = np.asarray(gtw, np.float64)

    logits_all = h.logits(feats.reshape(-1, 23)).reshape(n, K).astype(np.float64)

    # ---------------- per-query orders, hits, distances
    M = len(METHODS)
    orders = np.zeros((M, n, K), np.int8)
    hit_cell = {m: {k: np.zeros(n, np.int8) for k in KS_CELL} for m in METHODS}
    hit_loc = {m: {k: {t: np.zeros(n, np.int8) for t in THRESHS}
                   for k in (1, 5, 10)} for m in METHODS}
    hit_t25 = {m: {t: np.zeros(n, np.int8) for t in THRESHS} for m in METHODS}
    dist_m = np.zeros((n, K), np.float64)
    cross_pairs = cross_queries = 0

    for i in range(n):
        p2i = np.asarray(p2_all[i], np.int64)
        assert len(set(p2i.tolist())) == K
        p0, p2c, _tied = F.compute_orders(cos_all[i], sp_all[i], sc_center_all[i])
        assert np.array_equal(p2i, p2c)

        rank_scf = np.empty(K, np.int64)
        order_scf = np.argsort(-sc_fine_all[i][p2i], kind="stable")
        rank_scf[p2i[order_scf]] = np.arange(K)
        borda_c = (np.arange(K, dtype=np.float64) + rank_scf[p2i]) / (2.0 * (K - 1))
        order_borda = p2i[np.argsort(borda_c, kind="stable")]
        order_sat = p2i[np.argsort(-sat_fine_all[i][p2i], kind="stable")]
        order_head = F.rank_by_logits(logits_all[i], p2i)
        method_orders = {
            "text2loc": np.arange(K),
            "presence": p0,
            "c_borda": order_borda,
            "sat_fine": order_sat,
            HEAD_NAME: order_head,
        }

        dvec = np.linalg.norm(pred_w[i] - gtw[i][None, :], axis=1)
        xm = np.asarray([str(c).split("_")[0] != str(scene[i]) for c in ids[i]])
        cross_pairs += int(xm.sum())
        cross_queries += int(xm.any())
        dvec[xm] = np.inf
        dist_m[i] = dvec

        for j, m in enumerate(METHODS):
            o = method_orders[m]
            assert sorted(o.tolist()) == list(range(K))
            orders[j, i] = o
            ranked_ids = ids[i][o]
            for k in KS_CELL:
                hit_cell[m][k][i] = int(gt_cell[i] in set(str(c) for c in ranked_ids[:k]))
            rd = dvec[o]
            for k in (1, 5, 10):
                for t in THRESHS:
                    hit_loc[m][k][t][i] = int(np.min(rd[:k]) <= t)
            for t in THRESHS:
                hit_t25[m][t][i] = int(np.min(rd) <= t)
        if (i + 1) % 2000 == 0:
            print(f"{i+1}/{n}", flush=True)

    # ---------------- control reproduction (must pass before writing results)
    old = json.load(open(OLD_METRICS, encoding="utf-8"))
    ctrl = dict(status="PASS", checks=0, mismatches={}, mismatch_examples=[],
                old_metrics_sha256=inputs["old_test_metrics"]["sha256"],
                note="per-query hit vectors byte-compared with the old frozen TEST run")
    for m in CONTROLS:
        for k in KS_CELL:
            a = hit_cell[m][k]
            b = np.asarray(old["hit_vectors"][m][f"R{k}"], np.int8)
            assert a.shape == b.shape
            ctrl["checks"] += 1
            bad = np.flatnonzero(a != b)
            if bad.size:
                ctrl["mismatches"][f"{m}:R{k}"] = int(bad.size)
                for q in bad[:5]:
                    ctrl["mismatch_examples"].append(dict(query=int(q), method=m,
                                                          metric=f"R{k}",
                                                          new=int(a[q]), old=int(b[q])))
        for k in (1, 5, 10):
            for t in THRESHS:
                key = f"top{k}@{t}m"
                a = hit_loc[m][k][t]
                b = np.asarray(old["hit_loc_vectors"][m][key], np.int8)
                ctrl["checks"] += 1
                bad = np.flatnonzero(a != b)
                if bad.size:
                    ctrl["mismatches"][f"{m}:{key}"] = int(bad.size)
                    for q in bad[:5]:
                        ctrl["mismatch_examples"].append(dict(query=int(q), method=m,
                                                              metric=key,
                                                              new=int(a[q]), old=int(b[q])))
    old_cell = {m: {f"R@{k}": float(hit_cell[m][k].mean()) for k in KS_CELL}
                for m in CONTROLS}
    old_loc = {m: {f"top{k}@{t}m": float(hit_loc[m][k][t].mean())
                   for k in (1, 5, 10) for t in THRESHS} for m in CONTROLS}
    ctrl["old_vs_new_aggregate_max_abs_diff"] = {
        m: {"cell": max(abs(old_cell[m][k] - old["results_cell"][m][k])
                        for k in old_cell[m]),
            "loc": max(abs(old_loc[m][k] - old["results_loc"][m][k])
                       for k in old_loc[m])}
        for m in CONTROLS
    }
    if ctrl["mismatches"]:
        ctrl["status"] = "FAIL"
        with open(os.path.join(out_dir, "CONTROL_REPRODUCTION.json"), "w",
                  encoding="utf-8") as f:
            json.dump(ctrl, f, indent=2)
        print("CONTROL REPRODUCTION FAILED", json.dumps(ctrl["mismatches"]), flush=True)
        return 2
    with open(os.path.join(out_dir, "CONTROL_REPRODUCTION.json"), "w",
              encoding="utf-8") as f:
        json.dump(ctrl, f, indent=2)
    print("control reproduction PASS", ctrl["checks"], "checks", flush=True)

    # ---------------- aggregates
    res_cell = {m: {f"R@{k}": float(hit_cell[m][k].mean()) for k in KS_CELL}
                for m in METHODS}
    res_loc = {m: {f"top{k}@{t}m": float(hit_loc[m][k][t].mean())
                   for k in (1, 5, 10) for t in THRESHS} for m in METHODS}
    coverage = {f"top25_{t}m": float(hit_t25["text2loc"][t].mean()) for t in THRESHS}
    keys = cluster_keys(scene, gtw64)
    n_clusters = int(len(np.unique(keys, axis=0)))
    # Old run (dataset float64 pose_w, official loc path): 3932 clusters.
    # Cached float32 gt_pose_w can move a coordinate across the 1e-6 rounding
    # boundary by <1e-6; the true count of THIS run is recorded and reported.
    inv = dict(
        R25_equal=len({round(res_cell[m]["R@25"], 12) for m in METHODS}) == 1,
        top25_5m_equal=len({int(hit_t25[m][5].sum()) for m in METHODS}) == 1,
        top25_10m_equal=len({int(hit_t25[m][10].sum()) for m in METHODS}) == 1,
        top25_15m_equal=len({int(hit_t25[m][15].sum()) for m in METHODS}) == 1,
        coverage=coverage,
        coverage_equals_100pct={k: bool(v == 1.0) for k, v in coverage.items()},
        same_pool=True, same_positions=True, same_query_ids=True,
        query_ids_unique=True, order_permutations=True,
        n_queries=int(n), n_pose_clusters=n_clusters,
        n_scenes=int(len(set(scene.tolist()))),
        cross_scene_pairs=int(cross_pairs), cross_scene_queries=int(cross_queries),
        boundary_loc_mismatches=0,
        boundary_note="loc hits recomputed from frozen stored pred_w; old verifier "
                      "found 0 boundary mismatches vs official calc_sample_accuracies")

    def vec(m, name):
        if name.startswith("cell_R"):
            return hit_cell[m][int(name.split("cell_R")[1])]
        kk, tt = {"top1_5m": (1, 5), "top5_5m": (5, 5), "top10_5m": (10, 5),
                  "top1_10m": (1, 10), "top5_10m": (5, 10), "top10_10m": (10, 10),
                  "top1_15m": (1, 15), "top5_15m": (5, 15),
                  "top10_15m": (10, 15)}[name]
        return hit_loc[m][kk][tt]

    def comparisons(base):
        out = {}
        for m in METHODS:
            if m == base:
                continue
            out[f"{m}_minus_{base}"] = {
                name: cluster_bootstrap(vec(m, name).astype(np.float64)
                                        - vec(base, name).astype(np.float64), keys)
                for name in METRICS}
        return out

    comp_sat = comparisons("sat_fine")
    comp_pres = comparisons("presence")
    comp_t2l = comparisons("text2loc")
    primary = {k: comp_sat[f"{HEAD_NAME}_minus_sat_fine"][k] for k in PRIMARY}
    verdict = dict(
        primary_metric="Top-1@5m",
        head_vs_fsr=primary["top1_5m"],
        beats_fsr_by_CI=primary["top1_5m"]["ci95"][0] > 0,
        beats_presence_by_CI=comp_pres[f"{HEAD_NAME}_minus_presence"]["top1_5m"]["ci95"][0] > 0,
        beats_text2loc_by_CI=comp_t2l[f"{HEAD_NAME}_minus_text2loc"]["top1_5m"]["ci95"][0] > 0,
        note="intervals conditional on this frozen backbone, these 3 scenes and "
             "this one appended exploratory TEST read")

    per_scene = {}
    for sc in sorted(set(scene.tolist())):
        mask = scene == sc
        rec = dict(n=int(mask.sum()),
                   n_pose_clusters=int(len(np.unique(
                       cluster_keys(scene[mask], gtw[mask]), axis=0))))
        for m in METHODS:
            for base in ("text2loc", "presence", "sat_fine"):
                if m == base:
                    continue
                rec[f"{m}_minus_{base}"] = dict(
                    cell_R1=float(hit_cell[m][1][mask].mean()
                                  - hit_cell[base][1][mask].mean()),
                    top1_5m=float(hit_loc[m][1][5][mask].mean()
                                  - hit_loc[base][1][5][mask].mean()),
                    top5_5m=float(hit_loc[m][5][5][mask].mean()
                                  - hit_loc[base][5][5][mask].mean()))
        per_scene[sc] = rec

    # ---------------- per-query artifacts
    np.savez_compressed(
        os.path.join(out_dir, "per_query_orders.npz"),
        orders=orders, method_names=np.asarray(METHODS),
        dist_m=dist_m, logits=logits_all.astype(np.float32),
        cell_ids=ids, gt_cell=np.asarray(gt_cell), scene=scene,
        query_ids=qids, gt_pose_w=gtw,
        hit_cell=np.stack([[hit_cell[m][k] for k in KS_CELL] for m in METHODS]),
        hit_loc=np.stack([[hit_loc[m][k][t] for k in (1, 5, 10) for t in THRESHS]
                          for m in METHODS]),
        hit_t25=np.stack([[hit_t25[m][t] for t in THRESHS] for m in METHODS]),
        cell_ks=np.asarray(KS_CELL), threshs=np.asarray(THRESHS),
    )
    with open(os.path.join(out_dir, "per_query_hits.csv"), "w", encoding="utf-8",
              newline="") as f:
        w = csv.writer(f)
        header = ["q", "query_id", "scene", "gt_cell"]
        for m in METHODS:
            header += [f"{m}_cell_R{k}" for k in KS_CELL]
            header += [f"{m}_top{k}@{t}m" for t in THRESHS for k in (1, 5, 10)]
            header += [f"{m}_top25@{t}m" for t in THRESHS]
        w.writerow(header)
        for i in range(n):
            row = [i, str(qids[i]), str(scene[i]), gt_cell[i]]
            for j, m in enumerate(METHODS):
                row += [int(hit_cell[m][k][i]) for k in KS_CELL]
                row += [int(hit_loc[m][k][t][i]) for t in THRESHS for k in (1, 5, 10)]
                row += [int(hit_t25[m][t][i]) for t in THRESHS]
            w.writerow(row)

    # ---------------- metrics.json
    out = dict(
        status="OK",
        label="NEW TRAJECTORY s0 ep5 — APPENDED EXPLORATORY TEST READ",
        protocol=dict(
            protocol_id="FORMAL-SCAR-MP-KITTI-S0EP5-TEST-APPEND-v1",
            authorization="user authorization 2026-11-04; one-off exception to "
                          "R059_PROTOCOL.md section 14.4 as a NEW REGISTRATION",
            old_protocol_untouched=True,
            old_marker_untouched=True,
            new_head_only="s0 ep5 MAIN (B rule); s1/s2 not evaluated",
            backbone="MambaPlace coarse + fine, self-trained (R059/E), frozen caches",
            dataset="k360_30-10_scG_pd10_pc4_spY_all", split="test",
            n_queries=int(n), k=int(K), gallery_cells=int(len(set(
                str(c) for row in ids.tolist() for c in row))),
            candidate_set="self-trained MambaPlace coarse cosine Top-25 (fixed, shared)",
            fine="self-trained MambaPlace CrossMatch offsets for all 25 candidates",
            method_order=METHODS,
            methods=dict(
                text2loc="coarse cosine order (original)",
                presence="P0 = stable argsort(-(cos + 0.5*sp)); NOT pure presence",
                c_borda="Rank Fusion: ascending (p2_rank + sc_fine_rank)/(2(K-1)), "
                        "stable P2 tie-break",
                sat_fine="FSR: descending sat_fine, stable P2 tie-break",
                mp_formal_s0_ep5="formal SCAR head: canonical 23-feature MLP, "
                                 "TRAIN-only mean/std, B-rule VAL-selected epoch 5"),
            primary_metric="Top-1@5m",
            metric="cell R@k vs pose.cell_id; topk@tm from frozen pred_w; "
                   "cross-scene = inf; denominator all 11505 queries",
            labels="GT used only for hits/metrics; not used for ordering",
            bootstrap=dict(resamples=RESAMPLES, seed=SEED,
                           cluster_key="scene + round(gt_pose_w.xy, 6)",
                           unit="query pose cluster", n_clusters=n_clusters,
                           n_scenes=int(len(set(scene.tolist())))),
            exploratory_test=True,
            first_and_only_appended_read=True,
        ),
        inputs=inputs,
        head_binding=dict(
            path=HEAD, sha256=inputs["head"]["sha256"],
            epoch=int(h.selection["epoch"]), rule=h.selection["rule"],
            val_top1_5m=float(h.selection["key"][0]),
            val_cell_R1=float(h.selection["key"][1]),
            state_dict_identical_to_per_epoch=True,
            per_epoch_sha256=inputs["per_epoch_ep05"]["sha256"],
            feature_names=h.feature_names,
            normalization="TRAIN valid lists only; head mean/std == frozen "
                          "protocol/normalization.npy; 2 pinned columns",
        ),
        control_reproduction=ctrl,
        invariants=inv,
        results_cell=res_cell, results_loc=res_loc,
        comparisons_sat_fine=comp_sat,
        comparisons_presence=comp_pres,
        comparisons_text2loc=comp_t2l,
        primary=primary, verdict=verdict, per_scene=per_scene,
        artifacts=dict(
            per_query_orders="per_query_orders.npz",
            per_query_hits="per_query_hits.csv",
            per_query_orders_sha256=None,
        ),
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    out["artifacts"]["per_query_orders_sha256"] = sha256(
        os.path.join(out_dir, "per_query_orders.npz"))
    out["artifacts"]["per_query_hits_sha256"] = sha256(
        os.path.join(out_dir, "per_query_hits.csv"))
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    with open(os.path.join(out_dir, "paired_intervals.csv"), "w", encoding="utf-8",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "metric", "delta", "ci95_low", "ci95_high",
                    "excludes_zero", "win", "loss", "tie", "n_clusters"])
        for key, comp in (("sat_fine", comp_sat), ("presence", comp_pres),
                          ("text2loc", comp_t2l)):
            for name, rec in comp[f"{HEAD_NAME}_minus_{key}"].items():
                w.writerow([f"{HEAD_NAME}_minus_{key}", name, f"{rec['delta']:.12f}",
                            f"{rec['ci95'][0]:.12f}", f"{rec['ci95'][1]:.12f}",
                            rec["ci_excludes_zero"], rec["win"], rec["loss"],
                            rec["tie"], rec["n_clusters"]])

    with open(os.path.join(out_dir, "table2_cell_retrieval.csv"), "w",
              encoding="utf-8", newline="") as f:
        f.write("method," + ",".join(f"R@{k}" for k in KS_CELL) + "\n")
        for m in METHODS:
            f.write(m + "," + ",".join(f"{100*res_cell[m][f'R@{k}']:.2f}"
                                       for k in KS_CELL) + "\n")
    with open(os.path.join(out_dir, "table3_localization.csv"), "w",
              encoding="utf-8", newline="") as f:
        f.write("method," + ",".join(f"top{k}@{t}m" for t in THRESHS
                                     for k in (1, 5, 10)) + "\n")
        for m in METHODS:
            f.write(m + "," + ",".join(f"{100*res_loc[m][f'top{k}@{t}m']:.2f}"
                                       for t in THRESHS for k in (1, 5, 10)) + "\n")

    print("RESULTS cell", {m: {k: round(100*v, 2) for k, v in res_cell[m].items()}
                           for m in METHODS}, flush=True)
    print("RESULTS loc", {m: {k: round(100*v, 2) for k, v in res_loc[m].items()}
                          for m in METHODS}, flush=True)
    print("PRIMARY", json.dumps(primary, indent=2), flush=True)
    print("VERDICT", json.dumps(verdict, indent=2), flush=True)
    print("saved", out_dir, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
