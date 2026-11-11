"""Evaluate MambaPlace / presence / C-Borda / sat_fine / target-TRAIN SCAR heads.

MambaPlace (self-trained Coarse + self-trained Fine) x KITTI, protocol
R059-KITTI-MAMBA-PLACE-TARGET-HEAD-v1.

All methods share the same frozen Coarse weights, queries, 25 candidates and
self-trained-Fine positions; only the ordering changes.  Paired differences are
computed against original coarse order, presence (cos + 0.5*sp) and sat_fine,
with a pose-cluster bootstrap (2000 resamples, seed 123; cluster key =
scene + round(pose_w.xy, 6)).  KITTI TEST has 3 large scenes, so per-scene
point differences are reported; intervals are conditional on these scenes and
this frozen backbone.

Method/table order is fixed:
  text2loc, presence, c_borda, sat_fine, ht23_cmm_s0, ht23_cmm_s1, ht23_cmm_s2

TEST is an appended exploratory read: it refuses to overwrite this task's
existing test_one_shot marker.  Old TEST files and markers are untouched.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common_mp as C  # noqa: E402

THRESHS = C.THRESHS
KS_CELL = C.KS_CELL
SEED = 123
METHOD_ORDER = ["text2loc", "presence", "c_borda", "sat_fine",
                "ht23_cmm_s0", "ht23_cmm_s1", "ht23_cmm_s2"]
PRIMARY_METRICS = ["cell_R1", "top1_5m", "top5_5m", "top10_5m"]


def sha256(path):
    return C.sha256(path)


def cluster_keys(scene, pose_w):
    """scene code + rounded GT pose xy; float64 authoritative pose."""
    sc = sorted(set(str(x) for x in scene))
    code = {s: i for i, s in enumerate(sc)}
    return np.stack([
        np.array([code[str(s)] for s in scene], np.float64),
        np.round(np.asarray(pose_w, np.float64)[:, 0], 6),
        np.round(np.asarray(pose_w, np.float64)[:, 1], 6),
    ], axis=1)


def cluster_bootstrap(hit_a, hit_b, keys, seeds=2000, seed=SEED):
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    n_cl = len(uniq)
    diff = np.asarray(hit_a, np.float64) - np.asarray(hit_b, np.float64)
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
                cluster_key="scene + round(pose_w.xy, 6)", n_clusters=int(n_cl),
                resamples=int(seeds), seed=int(seed),
                win=int((diff > 0).sum()), loss=int((diff < 0).sum()),
                tie=int((diff == 0).sum()), n=int(diff.size))


class Head:
    def __init__(self, path):
        import torch
        import torch.nn as nn

        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.mean = ck["mean"].astype(np.float32)
        self.std = ck["std"].astype(np.float32)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--export_dir", default=None)
    ap.add_argument("--features", default=None)
    ap.add_argument("--heads", required=True, help="comma list of .pt paths")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    cli = ap.parse_args()
    export_dir = os.path.abspath(cli.export_dir or C.export_dir(cli.split))
    features_path = os.path.abspath(cli.features or os.path.join(export_dir, "feats.npz"))
    out_dir = os.path.abspath(cli.out_dir)
    heads_arg = ",".join(os.path.abspath(p.strip())
                         for p in cli.heads.split(",") if p.strip())
    os.makedirs(out_dir, exist_ok=True)
    test_log = os.path.join(out_dir, "test_one_shot.json")
    if os.path.exists(test_log):
        raise SystemExit(f"REFUSING: this task's TEST read already happened, see {test_log}")

    os.chdir(C.CODE)
    sys.path.insert(0, C.CODE)
    sys.path.insert(0, C.VENDOR)

    import features_fix as F
    from evaluation.utils import calc_sample_accuracies

    z = np.load(os.path.join(export_dir, "candidates.npz"), allow_pickle=True)
    ids = np.asarray(z["cell_ids"])
    offsets = np.asarray(z["offsets"], np.float64)
    pred_w = np.asarray(z["pred_w"], np.float64)
    gt_cell = [str(x) for x in z["gt_cell"]]
    scene = np.asarray([str(x) for x in z["scene"]])
    qids = np.asarray([str(x) for x in z["query_ids"]])
    src_idx = np.asarray(z["source_pose_index"], np.int64)
    n, K = ids.shape
    assert K == C.K
    assert len(set(qids.tolist())) == n
    if cli.limit == 0:
        assert n == C.EXPECTED[cli.split], (n, C.EXPECTED[cli.split])

    d = np.load(features_path, allow_pickle=True)
    feats = d["feats"]
    assert len(feats) <= n
    n = len(feats)
    ids, offsets, pred_w = ids[:n], offsets[:n], pred_w[:n]
    gt_cell, scene, qids, src_idx = gt_cell[:n], scene[:n], qids[:n], src_idx[:n]
    p2 = np.asarray(d["p2_idx"])
    cos = np.asarray(d["cos"], np.float64)
    sp = np.asarray(d["sp"], np.float64)
    sc_center = np.asarray(d["sc_center"], np.float64)
    sc_fine = np.asarray(d["sc_fine"], np.float64)
    sat_fine = np.asarray(d["sat_fine"], np.float64)
    assert [str(x) for x in d["gt_cell"]] == gt_cell
    assert [str(x) for x in d["scene"]] == scene.tolist()
    assert [str(x) for x in d["query_ids"]] == qids.tolist()
    assert np.array_equal(np.asarray(d["source_pose_index"], np.int64), src_idx)
    assert [[str(c) for c in row] for row in d["cell_ids"]] == \
           [[str(c) for c in row] for row in ids.tolist()]
    assert np.allclose(np.asarray(d["offsets"], np.float64), offsets, atol=0, rtol=0)
    assert np.allclose(np.asarray(d["pred_w"], np.float64), pred_w, atol=0, rtol=0)
    assert np.allclose(cos, np.asarray(z["scores"], np.float64), atol=1e-5, rtol=0)
    assert list(d["feature_names"]) == list(F.FEATURE_NAMES)
    assert feats.shape == (n, K, 23)
    n_parsed = np.asarray(d["n_parsed"], np.int64)
    n_mentions = np.asarray(d["num_mentions"], np.int64)
    parse_ok = int((n_parsed == n_mentions).sum())
    assert parse_ok == n, (parse_ok, n)
    fmeta_path = features_path + ".meta.json"
    fmeta = None
    if os.path.exists(fmeta_path):
        with open(fmeta_path, encoding="utf-8") as f:
            fmeta = json.load(f)
        assert int(fmeta["queries"]) == n
        assert bool(fmeta["parse_all_equal"]) is True
    parse_stats = dict(
        queries=int(n), parsed_exactly=parse_ok,
        parse_failures=int(n - parse_ok),
        parse_failure_ratio=float((n - parse_ok) / n),
        fallback_used=False,
        mention_histogram={str(int(k)): int(v)
                           for k, v in zip(*np.unique(n_mentions, return_counts=True))},
        meta_path=fmeta_path if fmeta else None,
        meta_sha256=sha256(fmeta_path) if fmeta else None,
        note="all mentions parsed; a parse failure aborts the build "
             "(assert len(parsed)==num_mentions), so there is no silent fallback")

    ds, poses, cells = C.load_split_dataset(cli.split)
    cells_dict = {str(c.id): c for c in cells}
    assert len(poses) >= n
    poses = poses[:n]
    assert [str(p.cell_id) for p in poses] == gt_cell
    assert [str(p.scene_name) for p in poses] == scene.tolist()
    gt_pose_w64 = np.asarray([p.pose_w[:2] for p in poses], np.float64)
    assert np.allclose(np.asarray(d["gt_pose_w"], np.float64), gt_pose_w64,
                       atol=1e-6, rtol=0)
    assert all(len(set(row)) == K for row in ids.tolist())
    assert not ({c for row in ids.tolist() for c in row} - set(cells_dict))

    n_done = min(n, cli.limit) if cli.limit else n
    heads = []
    for path in heads_arg.split(","):
        path = path.strip()
        if not path:
            continue
        h = Head(path)
        assert h.nf == 23
        assert h.feature_names == list(F.FEATURE_NAMES), h.feature_names
        assert h.mean.shape == (23,) and h.std.shape == (23,)
        heads.append((os.path.basename(path).replace(".pt", ""), h, os.path.abspath(path)))
    head_names = [h[0] for h in heads]
    methods = ["text2loc", "presence", "c_borda", "sat_fine"] + head_names
    assert methods == METHOD_ORDER, methods

    hit_cell = {m: {k: np.zeros(n_done, np.int8) for k in KS_CELL} for m in methods}
    hit_loc = {m: {k: {t: np.zeros(n_done, np.int8) for t in THRESHS}
                   for k in (1, 5, 10)} for m in methods}
    hit_t25 = {m: {t: np.zeros(n_done, np.int8) for t in THRESHS} for m in methods}
    rows = []

    for i in range(n_done):
        p2i = np.asarray(p2[i], np.int64)
        assert len(set(p2i.tolist())) == K
        assert np.array_equal(p2i, F.compute_orders(cos[i], sp[i], sc_center[i])[1])

        rank_scf = np.empty(K, np.int64)
        order_scf = np.argsort(-sc_fine[i][p2i], kind="stable")
        rank_scf[p2i[order_scf]] = np.arange(K)
        borda_c = (np.arange(K, dtype=np.float64) + rank_scf[p2i]) / (2.0 * (K - 1))
        order_borda = p2i[np.argsort(borda_c, kind="stable")]
        order_sat = p2i[np.argsort(-sat_fine[i][p2i], kind="stable")]
        order_pres = F.compute_orders(cos[i], sp[i], sc_center[i])[0]
        orders = {"text2loc": np.arange(K), "presence": order_pres,
                  "c_borda": order_borda, "sat_fine": order_sat}
        logits = {}
        for name, h, _p in heads:
            lg = h.logits(feats[i][None, :, :])[0]
            logits[name] = lg
            orders[name] = F.rank_by_logits(lg, p2i)
        for m, o in orders.items():
            assert len(o) == K and len(set(o.tolist())) == K
            assert sorted(o.tolist()) == list(range(K))

        cids = np.asarray([str(c) for c in ids[i]])
        pose = poses[i]
        gt = gt_cell[i]
        for m, o in orders.items():
            ranked = cids[o]
            roff = offsets[i][o]
            rpred = pred_w[i][o]
            for k in KS_CELL:
                hit_cell[m][k][i] = int(gt in ranked[:k])
            acc = calc_sample_accuracies(pose, [cells_dict[c] for c in ranked[:10]],
                                         roff[:10], (1, 5, 10), THRESHS)
            for k in (1, 5, 10):
                for t in THRESHS:
                    hit_loc[m][k][t][i] = int(acc[k][t])
            dvec = np.linalg.norm(rpred - gt_pose_w64[i][None, :], axis=1)
            dvec[[str(c).split("_")[0] != str(scene[i]) for c in ranked]] = np.inf
            for t in THRESHS:
                hit_t25[m][t][i] = int(np.min(dvec) <= t)

        rows.append(dict(q=i, query_id=str(qids[i]),
                         source_pose_index=int(src_idx[i]),
                         scene=str(scene[i]), gt_cell=gt,
                         num_mentions=len(pose.descriptions),
                         ids=[str(c) for c in cids], p2=[int(x) for x in p2i],
                         cos=[float(x) for x in cos[i]], sp=[float(x) for x in sp[i]],
                         sc_center=[float(x) for x in sc_center[i]],
                         sc_fine=[float(x) for x in sc_fine[i]],
                         sat_fine=[float(x) for x in sat_fine[i]],
                         logits={k: [float(x) for x in v] for k, v in logits.items()},
                         order={m: [int(j) for j in o] for m, o in orders.items()},
                         hit_cell={m: {f"R{k}": int(hit_cell[m][k][i]) for k in KS_CELL}
                                   for m in methods},
                         hit_loc={m: {f"top{k}@{t}m": int(hit_loc[m][k][t][i])
                                      for k in (1, 5, 10) for t in THRESHS}
                                  for m in methods}))
        if (i + 1) % 1000 == 0:
            print(f"{i+1}/{n_done}", flush=True)

    res_cell = {m: {f"R@{k}": float(hit_cell[m][k].mean()) for k in KS_CELL}
                for m in methods}
    res_loc = {m: {f"top{k}@{t}m": float(hit_loc[m][k][t].mean())
                   for k in (1, 5, 10) for t in THRESHS} for m in methods}
    r25 = len({round(res_cell[m]["R@25"], 12) for m in methods}) == 1
    t25 = {t: len({int(hit_t25[m][t].sum()) for m in methods}) == 1 for t in THRESHS}
    coverage = {f"top25_{t}m": float(hit_t25["text2loc"][t].mean()) for t in THRESHS}
    inv = dict(R25_equal=r25, top25_5m_equal=t25[5], top25_10m_equal=t25[10],
               top25_15m_equal=t25[15],
               coverage=coverage,
               coverage_equals_100pct={k: bool(v == 1.0) for k, v in coverage.items()},
               coverage_note="ordering methods share candidate coverage; equality "
                             "does not imply 100% coverage",
               same_pool=True, same_positions=True, same_query_ids=True,
               query_ids_unique=True, order_permutations=True,
               n_queries=int(n_done),
               n_pose_clusters=int(len(np.unique(
                   cluster_keys(scene[:n_done], gt_pose_w64[:n_done]), axis=0))),
               n_scenes=int(len(set(scene[:n_done].tolist()))),
               expected_queries=C.EXPECTED[cli.split] if cli.limit == 0 else None)

    keys = cluster_keys(scene[:n_done], gt_pose_w64[:n_done])
    metrics = [("cell_R1", hit_cell, 1), ("cell_R2", hit_cell, 2),
               ("cell_R3", hit_cell, 3), ("cell_R5", hit_cell, 5),
               ("cell_R10", hit_cell, 10), ("cell_R25", hit_cell, 25),
               ("top1_5m", hit_loc, (1, 5)), ("top5_5m", hit_loc, (5, 5)),
               ("top10_5m", hit_loc, (10, 5)), ("top1_10m", hit_loc, (1, 10)),
               ("top5_10m", hit_loc, (5, 10)), ("top10_10m", hit_loc, (10, 10)),
               ("top1_15m", hit_loc, (1, 15)), ("top5_15m", hit_loc, (5, 15)),
               ("top10_15m", hit_loc, (10, 15))]

    def vec(arr, k):
        if isinstance(k, tuple):
            return arr[k[0]][k[1]]
        return arr[k]

    def comp_against(base):
        out = {}
        for m in methods:
            if m == base:
                continue
            out[f"{m}_minus_{base}"] = {
                name: cluster_bootstrap(vec(arr[m], k), vec(arr[base], k), keys)
                for name, arr, k in metrics}
        return out

    comp = comp_against("text2loc")
    comp_pres = comp_against("presence")
    comp_sat = comp_against("sat_fine")

    per_scene = {}
    for sc in sorted(set(scene[:n_done].tolist())):
        mask = scene[:n_done] == sc
        dsc = dict(n=int(mask.sum()),
                   n_pose_clusters=int(len(np.unique(
                       cluster_keys(scene[:n_done][mask], gt_pose_w64[:n_done][mask]),
                       axis=0))))
        for m in methods:
            dsc[m] = {}
            for base in ("text2loc", "presence", "sat_fine"):
                if m == base:
                    continue
                dsc[m][f"minus_{base}"] = dict(
                    cell_R1=float(hit_cell[m][1][mask].mean() - hit_cell[base][1][mask].mean()),
                    top1_5m=float(hit_loc[m][1][5][mask].mean() - hit_loc[base][1][5][mask].mean()),
                    top5_5m=float(hit_loc[m][5][5][mask].mean() - hit_loc[base][5][5][mask].mean()))
        per_scene[sc] = dsc

    comps = {"text2loc": comp, "presence": comp_pres, "sat_fine": comp_sat}
    primary = {}
    for base in ("text2loc", "presence", "sat_fine"):
        primary[f"ht23_cmm_s0_minus_{base}"] = {
            k: comps[base][f"ht23_cmm_s0_minus_{base}"][k] for k in PRIMARY_METRICS}
    verdict_5m = primary["ht23_cmm_s0_minus_text2loc"]["top1_5m"]
    verdict = dict(
        primary_metric="Top-1@5m",
        head0_vs_text2loc=verdict_5m,
        beats_text2loc_by_CI=verdict_5m["ci95"][0] > 0,
        beats_presence_by_CI=primary["ht23_cmm_s0_minus_presence"]["top1_5m"]["ci95"][0] > 0,
        beats_sat_fine_by_CI=primary["ht23_cmm_s0_minus_sat_fine"]["top1_5m"]["ci95"][0] > 0,
        note="intervals conditional on this frozen backbone and these scenes; "
             "not a broad generalisation claim")

    out = dict(
        status="OK",
        protocol=dict(
            protocol_id="R059-KITTI-MAMBA-PLACE-TARGET-HEAD-v1",
            backbone=C.BACKBONE,
            backbone_note="both stages self-trained in this task; no released checkpoint used",
            dataset="k360_30-10_scG_pd10_pc4_spY_all",
            split=cli.split, n_queries=int(n_done), k=K,
            gallery_cells=int(len(cells)),
            candidate_set="self-trained MambaPlace coarse cosine Top-25 (fixed, shared)",
            fine=C.FINE_LABEL,
            method_order=METHOD_ORDER,
            methods=dict(
                text2loc="coarse cosine order",
                presence="P0 = stable argsort(-(cos + 0.5*sp)); sp = "
                         "matched-object fraction; NOT pure presence",
                c_borda="ascending (p2_rank + sc_fine_rank)/(2(K-1)), stable P2",
                sat_fine="descending sat_fine, stable P2",
                ht="canonical 23-feature MLP; TRAIN-only mean/std; "
                   "VAL-selected epoch; s0 primary"),
            primary_metric="Top-1@5m",
            metric="cell R@k vs pose.cell_id; loc = official calc_sample_accuracies; "
                   "cross-scene distance = inf",
            labels="same scene and ||pred_w - gt_pose_w|| <= 5 m "
                   "(used for training labels and VAL selection only)",
            bootstrap=dict(resamples=2000, seed=SEED,
                           cluster_key="scene + round(pose_w.xy, 6)",
                           unit="query pose cluster",
                           n_clusters=inv["n_pose_clusters"], n_scenes=inv["n_scenes"]),
            exploratory_test=bool(cli.split == "test"),
            expected_queries=C.EXPECTED[cli.split] if cli.limit == 0 else None),
        heads={name: dict(path=path, sha256=sha256(path), selection=h.selection)
               for name, h, path in heads},
        features=dict(path=features_path, sha256=sha256(features_path),
                      meta_sha256=sha256(fmeta_path) if fmeta else None),
        parse_stats=parse_stats,
        export=dict(dir=export_dir,
                    candidates_sha256=sha256(os.path.join(export_dir, "candidates.npz")),
                    coarse_enc_sha256=sha256(os.path.join(export_dir, "coarse_enc.npz"))),
        invariants=inv,
        results_cell=res_cell, results_loc=res_loc,
        comparisons=comp, comparisons_presence=comp_pres,
        comparisons_sat_fine=comp_sat,
        primary=primary, verdict=verdict, per_scene=per_scene,
        hit_vectors={m: {f"R{k}": hit_cell[m][k].tolist() for k in KS_CELL}
                     for m in methods},
        hit_loc_vectors={m: {f"top{k}@{t}m": hit_loc[m][k][t].tolist()
                             for k in (1, 5, 10) for t in THRESHS} for m in methods},
        hit_t25_vectors={m: {f"top25@{t}m": hit_t25[m][t].tolist()
                             for t in THRESHS} for m in methods},
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(out_dir, "per_query.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(out_dir, "table2_cell_retrieval.csv"), "w",
              encoding="utf-8", newline="") as f:
        f.write("method," + ",".join(f"R@{k}" for k in KS_CELL) + "\n")
        for m in methods:
            f.write(m + "," + ",".join(f"{100*res_cell[m][f'R@{k}']:.2f}"
                                       for k in KS_CELL) + "\n")
    with open(os.path.join(out_dir, "table3_localization.csv"), "w",
              encoding="utf-8", newline="") as f:
        f.write("method," + ",".join(f"top{k}@{t}m" for t in THRESHS
                                     for k in (1, 5, 10)) + "\n")
        for m in methods:
            f.write(m + "," + ",".join(f"{100*res_loc[m][f'top{k}@{t}m']:.2f}"
                                       for t in THRESHS for k in (1, 5, 10)) + "\n")
    if cli.split == "test":
        with open(test_log, "w", encoding="utf-8") as f:
            json.dump(dict(status="TEST_EVALUATED_ONCE_EXPLORATORY",
                           timestamp_utc=out["timestamp_utc"],
                           heads=out["heads"], features=out["features"],
                           export=out["export"], n_queries=int(n_done),
                           command=" ".join(sys.argv)), f, indent=2)
    print("invariants", inv, flush=True)
    for m in methods:
        print(f"{m:14s} " + " ".join(f"R{k}={100*res_cell[m][f'R@{k}']:.2f}"
                                     for k in KS_CELL)
              + " | " + " ".join(f"T{k}@{t}={100*res_loc[m][f'top{k}@{t}m']:.2f}"
                                 for k in (1, 5, 10) for t in THRESHS), flush=True)
    print("verdict", json.dumps(verdict, indent=2), flush=True)
    print("saved", out_dir, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
