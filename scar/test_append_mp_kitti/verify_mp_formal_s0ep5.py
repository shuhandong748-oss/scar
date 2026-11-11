"""Independent verifier for the appended exploratory s0-ep5 TEST evaluation.

Does not trust the evaluation script's aggregates.  Rebuilds every per-query
hit from the frozen caches, re-derives every aggregate, re-runs every bootstrap,
and byte-compares the four control methods against the OLD frozen TEST metrics.
Also re-hashes every old asset against the pre-run snapshot.

Exit nonzero on any mismatch.  Writes VERIFY.json.
"""
import argparse
import csv
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
HEAD_NAME = "mp_formal_s0_ep5"
METHODS = ["text2loc", "presence", "c_borda", "sat_fine", HEAD_NAME]
CONTROLS = ["text2loc", "presence", "c_borda", "sat_fine"]
KS_CELL = (1, 2, 3, 5, 10, 25)
THRESHS = (5, 10, 15)
METRIC_SPEC = ([("cell_R1", "cell", 1), ("cell_R2", "cell", 2),
                ("cell_R3", "cell", 3), ("cell_R5", "cell", 5),
                ("cell_R10", "cell", 10), ("cell_R25", "cell", 25)]
               + [(f"top{k}_{t}m", "loc", (k, t)) for t in THRESHS
                  for k in (1, 5, 10)])
PRIMARY = ["cell_R1", "top1_5m", "top5_5m", "top10_5m"]
SEED = 123
RESAMPLES = 2000


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cluster_keys(scene, pose_w):
    sc = sorted(set(str(x) for x in scene))
    code = {s: i for i, s in enumerate(sc)}
    return np.stack([np.array([code[str(s)] for s in scene], np.float64),
                     np.round(np.asarray(pose_w, np.float64)[:, 0], 6),
                     np.round(np.asarray(pose_w, np.float64)[:, 1], 6)], axis=1)


def boot(diff, keys, seeds=RESAMPLES, seed=SEED):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(RUN, "results"))
    ap.add_argument("--out", default=os.path.join(RUN, "verify", "VERIFY.json"))
    cli = ap.parse_args()
    res_dir = os.path.abspath(cli.results)
    out_path = os.path.abspath(cli.out)
    checks = []
    failures = []

    def check(name, cond, detail=None):
        checks.append(dict(check=name, ok=bool(cond), detail=detail))
        if not cond:
            failures.append(dict(check=name, detail=detail))

    with open(os.path.join(res_dir, "metrics.json"), encoding="utf-8") as f:
        M = json.load(f)
    q = np.load(os.path.join(res_dir, "per_query_orders.npz"), allow_pickle=True)
    z = np.load(os.path.join(CACHE, "candidates.npz"), allow_pickle=True)
    d = np.load(os.path.join(CACHE, "feats.npz"), allow_pickle=True)
    old = json.load(open(OLD_METRICS, encoding="utf-8"))

    check("status", M["status"] == "OK")
    check("label", M["label"] == "NEW TRAJECTORY s0 ep5 — APPENDED EXPLORATORY TEST READ")
    check("method_order", M["protocol"]["method_order"] == METHODS)
    check("methods_npz", [str(x) for x in q["method_names"]] == METHODS)

    ids = np.asarray(z["cell_ids"])
    pred_w = np.asarray(z["pred_w"], np.float64)
    gt_cell = [str(x) for x in z["gt_cell"]]
    scene = np.asarray([str(x) for x in z["scene"]])
    qids = np.asarray([str(x) for x in z["query_ids"]])
    n, K = ids.shape
    check("n_queries", n == 11505 and M["protocol"]["n_queries"] == n, n)
    check("k", K == 25)
    check("query_ids_unique", len(set(qids.tolist())) == n)

    gtw = np.asarray(d["gt_pose_w"], np.float64)
    p2_all = np.asarray(d["p2_idx"], np.int64)
    cos = np.asarray(d["cos"], np.float64)
    sp = np.asarray(d["sp"], np.float64)
    sc_center = np.asarray(d["sc_center"], np.float64)
    sc_fine = np.asarray(d["sc_fine"], np.float64)
    sat_fine = np.asarray(d["sat_fine"], np.float64)
    logits = np.asarray(q["logits"], np.float64)

    check("npz_ids", [[str(c) for c in row] for row in q["cell_ids"]] ==
          [[str(c) for c in row] for row in ids])
    check("npz_scene", [str(x) for x in q["scene"]] == scene.tolist())
    check("npz_qids", [str(x) for x in q["query_ids"]] == qids.tolist())
    check("npz_gt_cell", [str(x) for x in q["gt_cell"]] == gt_cell)
    check("npz_gt_pose", np.allclose(np.asarray(q["gt_pose_w"], np.float64), gtw,
                                     atol=0, rtol=0))
    check("npz_methods_shape", q["orders"].shape == (len(METHODS), n, K))

    # ---------------- rebuild orders from frozen scores
    orders = np.asarray(q["orders"], np.int64)
    dvec_all = np.zeros((n, K), np.float64)
    rebuilt = {m: np.zeros((n, K), np.int64) for m in METHODS}
    for i in range(n):
        p2i = np.asarray(p2_all[i], np.int64)
        p0, p2c, _ = F.compute_orders(cos[i], sp[i], sc_center[i])
        assert np.array_equal(p2i, p2c)
        rank_scf = np.empty(K, np.int64)
        order_scf = np.argsort(-sc_fine[i][p2i], kind="stable")
        rank_scf[p2i[order_scf]] = np.arange(K)
        borda_c = (np.arange(K, dtype=np.float64) + rank_scf[p2i]) / (2.0 * (K - 1))
        rebuilt["text2loc"][i] = np.arange(K)
        rebuilt["presence"][i] = p0
        rebuilt["c_borda"][i] = p2i[np.argsort(borda_c, kind="stable")]
        rebuilt["sat_fine"][i] = p2i[np.argsort(-sat_fine[i][p2i], kind="stable")]
        rebuilt[HEAD_NAME][i] = F.rank_by_logits(logits[i], p2i)
        dv = np.linalg.norm(pred_w[i] - gtw[i][None, :], axis=1)
        dv[[str(c).split("_")[0] != str(scene[i]) for c in ids[i]]] = np.inf
        dvec_all[i] = dv

    check("distances_recomputed", np.allclose(dvec_all, np.asarray(q["dist_m"]),
                                              atol=1e-12, rtol=0))
    for j, m in enumerate(METHODS):
        check(f"orders_match_{m}", np.array_equal(orders[j], rebuilt[m]))
        check(f"orders_permutation_{m}",
              all(sorted(orders[j, i].tolist()) == list(range(K)) for i in range(n)))

    # ---------------- rebuild hits
    hit_cell = {m: {k: np.zeros(n, np.int8) for k in KS_CELL} for m in METHODS}
    hit_loc = {m: {k: {t: np.zeros(n, np.int8) for t in THRESHS}
                   for k in (1, 5, 10)} for m in METHODS}
    hit_t25 = {m: {t: np.zeros(n, np.int8) for t in THRESHS} for m in METHODS}
    for j, m in enumerate(METHODS):
        for i in range(n):
            o = orders[j, i]
            ranked = [str(c) for c in ids[i][o]]
            dv = dvec_all[i][o]
            for k in KS_CELL:
                hit_cell[m][k][i] = int(gt_cell[i] in ranked[:k])
            for k in (1, 5, 10):
                for t in THRESHS:
                    hit_loc[m][k][t][i] = int(np.min(dv[:k]) <= t)
            for t in THRESHS:
                hit_t25[m][t][i] = int(np.min(dv) <= t)

    stored_cell = np.asarray(q["hit_cell"], np.int64)
    stored_loc = np.asarray(q["hit_loc"], np.int64)
    check("npz_hit_cell", np.array_equal(stored_cell,
          np.stack([[hit_cell[m][k] for k in KS_CELL] for m in METHODS])))
    check("npz_hit_loc", np.array_equal(stored_loc,
          np.stack([[hit_loc[m][k][t] for k in (1, 5, 10) for t in THRESHS]
                    for m in METHODS])))

    # ---------------- control reproduction vs OLD frozen metrics
    ctrl_checks = 0
    ctrl_bad = []
    for m in CONTROLS:
        for k in KS_CELL:
            a = hit_cell[m][k]
            b = np.asarray(old["hit_vectors"][m][f"R{k}"], np.int64)
            ctrl_checks += 1
            if not np.array_equal(a, b):
                ctrl_bad.append(f"{m}:R{k}:{(a != b).sum()}")
        for k in (1, 5, 10):
            for t in THRESHS:
                key = f"top{k}@{t}m"
                a = hit_loc[m][k][t]
                b = np.asarray(old["hit_loc_vectors"][m][key], np.int64)
                ctrl_checks += 1
                if not np.array_equal(a, b):
                    ctrl_bad.append(f"{m}:{key}:{(a != b).sum()}")
    check("control_reproduction_vs_old_run", not ctrl_bad,
          dict(checks=ctrl_checks, mismatches=ctrl_bad[:20]))
    check("control_reproduction_receipt", M["control_reproduction"]["status"] == "PASS")

    # ---------------- aggregates
    res_cell = {m: {f"R@{k}": float(hit_cell[m][k].mean()) for k in KS_CELL}
                for m in METHODS}
    res_loc = {m: {f"top{k}@{t}m": float(hit_loc[m][k][t].mean())
                   for k in (1, 5, 10) for t in THRESHS} for m in METHODS}
    agg_bad = []
    for m in METHODS:
        for k, v in res_cell[m].items():
            if abs(v - M["results_cell"][m][k]) > 1e-12:
                agg_bad.append(f"{m}:{k}:{v}:{M['results_cell'][m][k]}")
        for k, v in res_loc[m].items():
            if abs(v - M["results_loc"][m][k]) > 1e-12:
                agg_bad.append(f"{m}:{k}:{v}:{M['results_loc'][m][k]}")
    check("aggregates_match_json", not agg_bad, agg_bad[:20])

    # ---------------- CSV cell-by-cell
    with open(os.path.join(res_dir, "per_query_hits.csv"), newline="",
              encoding="utf-8") as f:
        rr = list(csv.reader(f))
    header, body = rr[0], rr[1:]
    check("csv_rows", len(body) == n)
    exp_header = ["q", "query_id", "scene", "gt_cell"]
    for m in METHODS:
        exp_header += [f"{m}_cell_R{k}" for k in KS_CELL]
        exp_header += [f"{m}_top{k}@{t}m" for t in THRESHS for k in (1, 5, 10)]
        exp_header += [f"{m}_top25@{t}m" for t in THRESHS]
    check("csv_header", header == exp_header)
    csv_bad = []
    for i, row in enumerate(body):
        if int(row[0]) != i or row[1] != str(qids[i]) or row[2] != str(scene[i]) \
                or row[3] != gt_cell[i]:
            csv_bad.append(f"row{i}:identity")
            continue
        col = 4
        for m in METHODS:
            for k in KS_CELL:
                if int(row[col]) != int(hit_cell[m][k][i]):
                    csv_bad.append(f"row{i}:{m}:R{k}")
                col += 1
            for t in THRESHS:
                for k in (1, 5, 10):
                    if int(row[col]) != int(hit_loc[m][k][t][i]):
                        csv_bad.append(f"row{i}:{m}:top{k}@{t}m")
                    col += 1
            for t in THRESHS:
                if int(row[col]) != int(hit_t25[m][t][i]):
                    csv_bad.append(f"row{i}:{m}:top25@{t}m")
                col += 1
        if len(csv_bad) > 20:
            break
    check("csv_cells", not csv_bad, csv_bad[:20])

    def read_csv_simple(path):
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        return rows[0], {r[0]: r[1:] for r in rows[1:]}

    h2, t2 = read_csv_simple(os.path.join(res_dir, "table2_cell_retrieval.csv"))
    check("table2_header", h2 == ["method"] + [f"R@{k}" for k in KS_CELL])
    ok2 = all(t2[m] == [f"{100*res_cell[m][f'R@{k}']:.2f}" for k in KS_CELL]
              for m in METHODS)
    check("table2_values", ok2)
    h3, t3 = read_csv_simple(os.path.join(res_dir, "table3_localization.csv"))
    check("table3_header", h3 == ["method"] + [f"top{k}@{t}m" for t in THRESHS
                                               for k in (1, 5, 10)])
    ok3 = all(t3[m] == [f"{100*res_loc[m][f'top{k}@{t}m']:.2f}"
                        for t in THRESHS for k in (1, 5, 10)] for m in METHODS)
    check("table3_values", ok3)

    # ---------------- invariants
    inv = M["invariants"]
    keys = cluster_keys(scene, gtw)
    n_clusters = int(len(np.unique(keys, axis=0)))
    check("R25_equal", len({round(res_cell[m]["R@25"], 12) for m in METHODS}) == 1)
    for t in THRESHS:
        check(f"top25_{t}m_equal",
              len({int(hit_t25[m][t].sum()) for m in METHODS}) == 1)
    coverage = {f"top25_{t}m": float(hit_t25["text2loc"][t].mean())
                for t in THRESHS}
    for t in THRESHS:
        check(f"coverage_{t}m_recorded",
              abs(coverage[f"top25_{t}m"] - inv["coverage"][f"top25_{t}m"]) < 1e-12)
    check("n_clusters_recorded", inv["n_pose_clusters"] == n_clusters,
          dict(recorded=inv["n_pose_clusters"], recomputed=n_clusters))
    check("n_scenes_recorded",
          inv["n_scenes"] == len(set(scene.tolist())))
    check("n_queries_recorded", inv["n_queries"] == n)

    # ---------------- bootstraps recomputed
    def arr(m, name):
        if name.startswith("cell_R"):
            return hit_cell[m][int(name.split("cell_R")[1])]
        _k, _t = name.split("_")
        return hit_loc[m][int(_k[3:])][int(_t[:-1])]

    boot_bad = []
    boot_checked = 0
    for base, jkey in (("sat_fine", "comparisons_sat_fine"),
                       ("presence", "comparisons_presence"),
                       ("text2loc", "comparisons_text2loc")):
        for name, _kind, _k in METRIC_SPEC:
            cj = M[jkey][f"{HEAD_NAME}_minus_{base}"][name]
            cb = boot(arr(HEAD_NAME, name) - arr(base, name), keys)
            for fld in ("delta", "ci95", "win", "loss", "tie", "n_clusters"):
                if fld == "ci95":
                    same = np.allclose(cj[fld], cb[fld], atol=1e-12)
                else:
                    same = cj[fld] == cb[fld]
                if not same:
                    boot_bad.append(f"{base}:{name}:{fld}")
            boot_checked += 1
    check("bootstrap_recomputed", not boot_bad,
          dict(metric_comparisons=boot_checked, bad=boot_bad[:20]))
    for name in PRIMARY:
        check(f"primary_{name}_recorded",
              M["primary"][name]["delta"] ==
              M["comparisons_sat_fine"][f"{HEAD_NAME}_minus_sat_fine"][name]["delta"])

    # ---------------- per-scene deltas recomputed
    scene_bad = []
    for sc, rec in M["per_scene"].items():
        mask = scene == sc
        if rec["n"] != int(mask.sum()):
            scene_bad.append(f"{sc}:n")
        if rec["n_pose_clusters"] != int(len(np.unique(
                cluster_keys(scene[mask], gtw[mask]), axis=0))):
            scene_bad.append(f"{sc}:clusters")
        for m in METHODS:
            for base in ("text2loc", "presence", "sat_fine"):
                key = f"{m}_minus_{base}"
                if m == base:
                    continue
                got = rec[key]
                for nm, a, b in (
                        ("cell_R1", hit_cell[m][1][mask], hit_cell[base][1][mask]),
                        ("top1_5m", hit_loc[m][1][5][mask], hit_loc[base][1][5][mask]),
                        ("top5_5m", hit_loc[m][5][5][mask], hit_loc[base][5][5][mask])):
                    exp = float(np.asarray(a, np.float64).mean()
                                - np.asarray(b, np.float64).mean())
                    if abs(got[nm] - exp) > 1e-12:
                        scene_bad.append(f"{sc}:{key}:{nm}")
    check("per_scene_recomputed", not scene_bad, scene_bad[:20])

    # ---------------- artifact hashes and old-asset integrity
    check("orders_npz_sha", sha256(os.path.join(res_dir, "per_query_orders.npz")) ==
          M["artifacts"]["per_query_orders_sha256"])
    check("hits_csv_sha", sha256(os.path.join(res_dir, "per_query_hits.csv")) ==
          M["artifacts"]["per_query_hits_sha256"])
    check("eval_code_sha",
          sha256(os.path.join(HERE, "eval_mp_formal_s0ep5.py")) ==
          M["inputs"]["eval_code_sha256"])
    for f, h in M["inputs"]["vendor_sha256"].items():
        check(f"vendor_{f}", sha256(os.path.join(VENDOR, f)) == h)

    before_path = os.path.join(RUN, "freeze", "BEFORE_SHA256.json")
    before_bad = []
    if os.path.exists(before_path):
        before = json.load(open(before_path, encoding="utf-8"))
        for rec in before["assets"]:
            cur = sha256(rec["path"])
            if cur != rec["sha256"]:
                before_bad.append(dict(path=rec["path"], before=rec["sha256"],
                                       now=cur))
    check("old_assets_unchanged", not before_bad, before_bad)
    check("old_metrics_sha", sha256(OLD_METRICS) ==
          M["inputs"]["old_test_metrics"]["sha256"])

    out = dict(
        status="PASS" if not failures else "FAIL",
        checks_total=len(checks), checks_failed=len(failures),
        failures=failures,
        control_reproduction=dict(checks=ctrl_checks, mismatches=ctrl_bad),
        n_queries=n, n_pose_clusters=n_clusters,
        n_scenes=int(len(set(scene.tolist()))), coverage=coverage,
        coverage_is_100pct={k: bool(v == 1.0) for k, v in coverage.items()},
        bootstraps_recomputed=boot_checked,
        csv_rows_checked=len(body),
        per_scene_checked=len(M["per_scene"]),
        old_asset_checks=len(before.get("assets", [])) if os.path.exists(before_path) else 0,
        result_sha256={
            f: sha256(os.path.join(res_dir, f))
            for f in ("metrics.json", "per_query_orders.npz", "per_query_hits.csv",
                      "CONTROL_REPRODUCTION.json", "paired_intervals.csv",
                      "table2_cell_retrieval.csv", "table3_localization.csv")
        },
        note="intervals conditional on this frozen backbone, these scenes and "
             "this single appended exploratory TEST read; not a general claim",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)
    print("VERIFY", out["status"], flush=True)
    return 0 if out["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
