#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Five-ranking TEST evaluation for one A domain from the single frozen cache.

Methods: Original, Presence, Rank Fusion, FSR, SCAR s0.  Only s0 is read; s1/s2
are out of scope.  Every query is in the denominator; empty/invalid candidate
lists are failures, never filtered.  Coordinates are the candidates' existing
Fine predictions; only the ordering changes.

Primary contrast: SCAR - FSR on Top-1@5m and Cell R@1.  All other contrasts are
secondary/exploratory.  Paired cluster bootstrap: 2000 resamples, seed 123,
RandomState, ratio-of-sums, linear 95% percentile.
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
LABELS = {"original": "Original", "presence": "Presence",
          "rank_fusion": "Rank Fusion", "fsr": "FSR", "scar_s0": "SCAR s0"}
EXPECTED_EPOCHS = {
    "pandaset": {"coarse": 8, "fine": 35},
    "nuscenesfull": {"coarse": 17, "fine": 30},
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=sorted(S.DOMAINS))
    ap.add_argument("--split", default="test", choices=["test"])
    ap.add_argument("--features", required=True)
    ap.add_argument("--export_dir", required=True)
    ap.add_argument("--heads_dir", required=True)
    ap.add_argument("--freeze_list", required=True)
    ap.add_argument("--marker", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    for n in ("features", "export_dir", "heads_dir", "out_dir", "freeze_list",
              "marker"):
        setattr(a, n, os.path.abspath(getattr(a, n)))
    os.makedirs(a.out_dir, exist_ok=False)
    d = S.domain_cfg(a.domain)
    freeze = S.read_json(a.freeze_list)
    if freeze.get("domain") != a.domain:
        raise SystemExit("freeze domain mismatch")
    canon = S.canonical_sha({k: v for k, v in freeze.items()
                             if k != "freeze_sha256"})
    if canon != freeze.get("freeze_sha256"):
        raise SystemExit("freeze canonical SHA mismatch")
    marker = S.require_marker(a.marker, a.domain, "completed",
                              freeze_sha=freeze["freeze_sha256"],
                              proc_sha=freeze["protocol"]["protocol_sha256"])
    em_path = os.path.join(a.export_dir, "export_meta.json")
    em = S.read_json(em_path)
    if em.get("domain") != a.domain or em.get("split") != a.split:
        raise SystemExit("export meta mismatch")
    comp = S.read_json(os.path.join(a.export_dir, "EXPORT_COMPLETE.json"))
    for name in ("candidates.npz", "coarse_enc.npz"):
        if S.sha256(os.path.join(a.export_dir, name)) != \
                comp["cache_sha256"][name]:
            raise SystemExit(f"export cache SHA mismatch: {name}")
    if marker.get("export", {}).get("cache_sha256", {}).get("candidates.npz") != \
            comp["cache_sha256"]["candidates.npz"]:
        raise SystemExit("marker does not bind this export cache")

    fmeta = S.read_json(a.features + ".meta.json")
    if fmeta.get("domain") not in (None, a.domain):
        raise SystemExit("feature meta domain mismatch")
    if fmeta.get("feature_version") != S.FEATURE_VERSION:
        raise SystemExit("feature version mismatch")
    if fmeta.get("export_meta_sha256") != S.sha256(em_path):
        raise SystemExit("features were not built from this export")
    # The feature meta records the marker SHA as it was at ARM time; the
    # builder then completes the same marker in place (status + features
    # block), so the current SHA can never equal that recorded value.  Bind
    # through the completed marker back to these exact feature bytes instead.
    mfeat = marker.get("features", {})
    if marker.get("status") != "completed" or \
            mfeat.get("sha256") != S.sha256(a.features) or \
            mfeat.get("meta_sha256") != S.sha256(a.features + ".meta.json"):
        raise SystemExit("features were bound to another marker")

    feat = np.load(a.features, allow_pickle=True)
    feats = np.asarray(feat["feats"], np.float32)
    ids = np.asarray([[str(c) for c in row] for row in feat["cell_ids"]])
    p2 = np.asarray(feat["p2_idx"], np.int64)
    cos = np.asarray(feat["cos"], np.float64)
    sp = np.asarray(feat["sp"], np.float64)
    sc_fine = np.asarray(feat["sc_fine"], np.float64)
    sat_fine = np.asarray(feat["sat_fine"], np.float64)
    valid = np.asarray(feat["valid"]).astype(bool) if "valid" in feat.files else None
    fnames = [str(x) for x in feat["feature_names"]]
    if len(fnames) != 23:
        raise SystemExit("feature names are not 23")
    if str(feat["split"][0]) != "test":
        raise SystemExit("expected a TEST feature file")

    cand = np.load(os.path.join(a.export_dir, "candidates.npz"),
                   allow_pickle=True)
    cids = np.asarray([[str(c) for c in row] for row in cand["cell_ids"]])
    cscene = np.asarray([[str(c) for c in row] for row in cand["cell_scene"]])
    pred_w = np.asarray(cand["pred_w"], np.float64)
    gt_pose_w = np.asarray(cand["gt_pose_w"], np.float64)
    # storage order is the export's own recorded coarse-cosine sort key; the
    # feature archive stores a recomputed float32 cosine that can differ by a
    # few ulps, so the order invariant is checked against the export key.
    storage_cos = (np.asarray(cand["coarse_cos"], np.float64)
                   if "coarse_cos" in cand.files else cos)
    gt_cell = [str(x) for x in cand["gt_cell"]]
    scene = np.asarray([str(x) for x in cand["scene"]])
    query_ids = np.asarray([str(x) for x in cand["query_ids"]])
    n, K = cids.shape
    if K != 25:
        raise SystemExit("cache is not K=25")
    if len(feats) != n or len(ids) != n:
        raise SystemExit("feature/candidate query count mismatch")
    if not np.array_equal(ids, cids):
        raise SystemExit("feature candidate order differs from the export")
    if not np.allclose(cos, storage_cos, rtol=0.0, atol=1e-5):
        raise SystemExit("feature cos differs from the exported coarse cosine")
    if [str(x) for x in feat["gt_cell"]] != gt_cell:
        raise SystemExit("gt_cell order mismatch")
    if n != d["expected_test"] and not a.limit:
        raise SystemExit(f"{a.domain} TEST: {n} queries != expected "
                         f"{d['expected_test']}")
    n_done = min(n, a.limit) if a.limit else n
    if len(set(query_ids[:n_done].tolist())) != n_done:
        raise SystemExit("query ids are not unique")

    head_path = os.path.join(a.heads_dir, f"ht23_{a.domain}_s0.pt")
    if not os.path.isfile(head_path):
        raise SystemExit(f"missing main head: {head_path}")
    if S.sha256(head_path) != freeze["main_head"]["sha256"]:
        raise SystemExit("main head s0 SHA != freeze")
    head = S.Head(head_path, expected_sha=freeze["main_head"]["sha256"])
    if head.feature_names != fnames:
        raise SystemExit("head feature names differ from the feature file")

    hit_cell = {m: {k: np.zeros(n_done, np.int8) for k in S.KS_CELL}
                for m in METHODS}
    hit_loc = {m: {(k, t): np.zeros(n_done, np.int8)
                   for k in S.KS_LOC for t in S.THRESHS} for m in METHODS}
    orders = np.zeros((n_done, K, len(METHODS)), np.int16)
    dist_all = np.zeros((n_done, K), np.float64)
    logits_s0 = np.zeros((n_done, K), np.float32)
    rows = []
    order_errors = 0

    for i in range(n_done):
        p2i = np.asarray(p2[i], np.int64)
        order_scf = np.argsort(-sc_fine[i][p2i], kind="stable")
        rank_scf = np.empty(K, np.int64)
        rank_scf[p2i[order_scf]] = np.arange(K)
        borda = (np.arange(K, dtype=np.float64) + rank_scf[p2i]) / (2.0 * (K - 1))
        order_rf = p2i[np.argsort(borda, kind="stable")]
        order_fsr = p2i[np.argsort(-sat_fine[i][p2i], kind="stable")]
        order_pres = np.argsort(-(cos[i] + 0.5 * sp[i]), kind="stable")
        lg = head.logits(feats[i][None, :, :])[0]
        order_scar = S.rank_by_logits(lg, p2i)
        cur = {"original": np.arange(K), "presence": order_pres,
               "rank_fusion": order_rf, "fsr": order_fsr,
               "scar_s0": order_scar}
        logits_s0[i] = lg
        d_all = np.linalg.norm(pred_w[i] - gt_pose_w[i][None, :], axis=1)
        d_all[cscene[i] != scene[i]] = np.inf
        dist_all[i] = d_all
        for mi, m in enumerate(METHODS):
            o = np.asarray(cur[m], np.int64)
            if len(set(int(x) for x in o)) != K:
                raise SystemExit(f"{m}: not a permutation at query {i}")
            orders[i, :, mi] = o
            for k, v in S.cell_hits(ids[i], o, gt_cell[i]).items():
                hit_cell[m][k][i] = v
            lh, d_ranked = S.loc_hits(pred_w[i], o, gt_pose_w[i], cscene[i],
                                      scene[i])
            for key, v in lh.items():
                hit_loc[m][key][i] = v
        if len({int(hit_cell[m][25][i]) for m in METHODS}) != 1:
            raise SystemExit(f"R@25 differs across methods at query {i}")
        if len({int(hit_loc[m][(25, 5.0)][i]) for m in METHODS}) != 1:
            raise SystemExit(f"Top-25@5m differs across methods at query {i}")
        if not np.array_equal(np.arange(K),
                              np.argsort(-storage_cos[i], kind="stable")):
            order_errors += 1
        rows.append(dict(
            q=i, query_id=query_ids[i], scene=scene[i], gt_cell=gt_cell[i],
            orders={m: [int(x) for x in cur[m]] for m in METHODS},
            hit_cell={m: {f"R{k}": int(hit_cell[m][k][i]) for k in S.KS_CELL}
                      for m in METHODS},
            hit_loc={m: {f"Top{k}@{int(t)}m": int(hit_loc[m][(k, t)][i])
                         for k in S.KS_LOC for t in S.THRESHS}
                     for m in METHODS},
            dist_all=[float(x) for x in d_all],
            logits_s0=[float(x) for x in lg]))
    if order_errors:
        raise SystemExit("storage order is not stable descending coarse cosine")

    res_cell, res_loc = S.metrics_from_hits(hit_cell, hit_loc, METHODS, n_done)
    mono_bad = S.check_monotonicity(res_cell, res_loc, METHODS)
    if mono_bad:
        raise SystemExit("success-rate monotonicity failed: " + "; ".join(mono_bad))
    inv = dict(
        n_queries=int(n_done), n_clusters=int(len(set(scene[:n_done].tolist()))),
        cluster_unit=d["cluster_unit"], k=K,
        all_queries_in_denominator=True, no_filtering=True,
        valid_lists=(int(valid[:n_done].sum()) if valid is not None else None),
        invalid_lists=(int(n_done - valid[:n_done].sum())
                       if valid is not None else None),
        R25_equal=len({round(res_cell[m]["R@25"], 12) for m in METHODS}) == 1,
        Top25_5m_equal=len({round(res_loc[m]["Top25@5m"], 12)
                            for m in METHODS}) == 1,
        monotonic=True, storage_order_is_coarse_cosine=True,
        permutations=True, same_pool=True, same_coordinates=True,
    )

    keys = scene[:n_done].astype("<U32").reshape(-1, 1)
    metrics = [(f"Cell_R@{k}", hit_cell, k) for k in S.KS_CELL]
    metrics += [(f"Top1@{int(t)}m", hit_loc, (1, t)) for t in S.THRESHS]
    metrics += [(f"Top{k}@5m", hit_loc, (k, 5.0)) for k in (5, 10, 25)]
    primary_names = {"Cell_R@1", "Top1@5m"}
    comparisons = {}
    for target in ("fsr", "original", "presence", "rank_fusion"):
        c = {}
        for name, src, sel in metrics:
            c[name] = S.bootstrap_ci(src["scar_s0"][sel],
                                     src[target][sel], keys)
        comparisons[f"SCAR_minus_{LABELS[target]}"] = dict(
            role=("PRIMARY" if target == "fsr" else "SECONDARY_EXPLORATORY"),
            metrics=c)
    # Report each non-SCAR method against FSR as an exploratory descriptive row.
    descriptive = {}
    for m in ("original", "presence", "rank_fusion"):
        descriptive[f"{LABELS[m]}_minus_FSR"] = {
            name: S.bootstrap_ci(src[m][sel], src["fsr"][sel], keys)
            for name, src, sel in metrics if name in primary_names}

    # cluster mapping
    uniq = sorted(set(scene[:n_done].tolist()))
    cmap_path = os.path.join(a.out_dir, "cluster_mapping.csv")
    counts = {u: int((scene[:n_done] == u).sum()) for u in uniq}
    with open(cmap_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_index", "query_id", "cluster", "cluster_unit",
                    "cluster_size"])
        for i in range(n_done):
            u = str(scene[i])
            w.writerow([i, query_ids[i], u, d["cluster_unit"], counts[u]])
    mapping_sha = S.sha256(cmap_path)
    map_canon_sha = S.canonical_sha(
        [[i, query_ids[i], str(scene[i])] for i in range(n_done)])

    out = dict(
        status="OK", protocol=dict(
            name=S.PROTOCOL, feature_version=S.FEATURE_VERSION,
            domain=a.domain, split="test", n_queries=int(n_done), k=K,
            cluster_unit=d["cluster_unit"],
            primary_metrics=["Top-1@5m", "Cell R@1"],
            primary_contrast="SCAR s0 - FSR",
            other_contrasts="secondary exploratory",
            bootstrap=dict(resamples=S.RESAMPLES, seed=S.SEED, paired=True,
                           rng="RandomState", weighting="ratio-of-sums",
                           percentile="linear 95%"),
            denominator="all queries; invalid/empty lists count as failures",
            methods={m: LABELS[m] for m in METHODS}),
        test_is_seen_not_blind=True,
        exploratory_not_confirmatory=True,
        main_head=dict(tag="s0", path=head_path, sha256=head.sha256,
                       formal_epoch=freeze.get("heads", {}).get("s0", {})
                       .get("formal_epoch")),
        s1_s2_evaluated=False,
        freeze=dict(path=a.freeze_list, sha256=S.sha256(a.freeze_list),
                    canonical_sha256=freeze["freeze_sha256"]),
        marker=dict(path=a.marker, sha256=S.sha256(a.marker)),
        export=dict(path=a.export_dir, meta_sha256=S.sha256(em_path),
                    cache_sha256=comp["cache_sha256"]),
        features=dict(path=a.features, sha256=S.sha256(a.features),
                      meta_sha256=S.sha256(a.features + ".meta.json")),
        cluster_mapping=dict(path=cmap_path, sha256=mapping_sha,
                             canonical_sha256=map_canon_sha,
                             units=int(len(uniq)), cluster_unit=d["cluster_unit"],
                             sizes=counts,
                             small_cluster_warning=bool(len(uniq) < 30)),
        invariants=inv, results_cell=res_cell, results_loc=res_loc,
        comparisons=comparisons, descriptive_vs_fsr=descriptive,
        code_sha256={f: S.sha256(os.path.join(HERE, f))
                     for f in ("a_eval_test_five.py", "a_test_common.py")})
    S.write_json(out, os.path.join(a.out_dir, "metrics.json"))

    np.savez_compressed(
        os.path.join(a.out_dir, "orders.npz"),
        orders=orders, methods=np.asarray(METHODS, dtype="<U32"),
        query_ids=query_ids[:n_done])
    np.savez_compressed(
        os.path.join(a.out_dir, "distances.npz"),
        dist_all=dist_all, pred_w=pred_w[:n_done],
        gt_pose_w=gt_pose_w[:n_done], cell_ids=cids[:n_done],
        cell_scene=cscene[:n_done], scene=scene[:n_done], gt_cell=gt_cell[:n_done],
        query_ids=query_ids[:n_done])
    np.savez_compressed(
        os.path.join(a.out_dir, "hits.npz"),
        **{f"cell_{m}_R{k}": hit_cell[m][k] for m in METHODS for k in S.KS_CELL},
        **{f"loc_{m}_Top{k}@{int(t)}m": hit_loc[m][(k, t)]
           for m in METHODS for k in S.KS_LOC for t in S.THRESHS})
    with open(os.path.join(a.out_dir, "per_query.jsonl"), "w",
              encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    cols = ["method", "Cell_R@1", "Cell_R@3", "Cell_R@5", "Cell_R@10",
            "Cell_R@25", "Top1@1m", "Top1@3m", "Top1@5m", "Top1@10m",
            "Top5@5m", "Top10@5m", "Top25@5m"]
    with open(os.path.join(a.out_dir, "table_metrics.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for m in METHODS:
            row = [LABELS[m]]
            row += [f"{100*res_cell[m][f'R@{k}']:.2f}"
                    for k in S.KS_CELL]
            row += [f"{100*res_loc[m][f'Top1@{int(t)}m']:.2f}"
                    for t in S.THRESHS]
            row += [f"{100*res_loc[m][f'Top{k}@5m']:.2f}" for k in (5, 10, 25)]
            w.writerow(row)
    with open(os.path.join(a.out_dir, "table_primary.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "Cell_R@1_pct", "Top1@5m_pct"])
        for m in METHODS:
            w.writerow([LABELS[m], f"{100*res_cell[m]['R@1']:.2f}",
                        f"{100*res_loc[m]['Top1@5m']:.2f}"])
    with open(os.path.join(a.out_dir, "table_comparisons.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "role", "metric", "delta_pp", "ci95_low_pp",
                    "ci95_high_pp", "n_clusters", "n_units", "wins", "losses",
                    "ties", "small_cluster_warning"])
        for name, block in comparisons.items():
            for metric, v in block["metrics"].items():
                w.writerow([name, block["role"], metric,
                            f"{100*v['delta']:.2f}",
                            f"{100*v['ci95'][0]:.2f}",
                            f"{100*v['ci95'][1]:.2f}", v["n_clusters"],
                            v["n_units"], v["wins"], v["losses"], v["ties"],
                            v["small_cluster_warning"]])
        for name, dct in descriptive.items():
            for metric, v in dct.items():
                w.writerow([name, "SECONDARY_EXPLORATORY", metric,
                            f"{100*v['delta']:.2f}",
                            f"{100*v['ci95'][0]:.2f}",
                            f"{100*v['ci95'][1]:.2f}", v["n_clusters"],
                            v["n_units"], v["wins"], v["losses"], v["ties"],
                            v["small_cluster_warning"]])

    # reload consistency check of the saved arrays
    ord2 = np.load(os.path.join(a.out_dir, "orders.npz"))["orders"]
    if not np.array_equal(ord2, orders):
        raise SystemExit("saved orders do not reload identically")
    print(json.dumps(dict(n_queries=n_done, n_clusters=len(uniq),
                          primary={m: [100*res_cell[m]["R@1"],
                                       100*res_loc[m]["Top1@5m"]]
                                   for m in METHODS}),
                     indent=2), flush=True)
    print("saved", a.out_dir, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
