"""Independent verification of the LSEL appended TEST evaluation.

Reads results/EVAL_RESULTS.json, results/per_query_*.npz and the frozen caches.
Recomputes every absolute score, paired difference, win/loss/tie count and the
cluster bootstrap with a loop implementation (not the matmul used in
eval_lsel_test.py), and re-derives the sat_fine hit vectors from the cache.
Writes results/VERIFICATION.json.
"""
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = {
    "mncl_s3dis": dict(cache="test_cache/mncl_s3dis/test_ht23.npz",
                       cell_ids="cids", gt_cell="gt_cell", scene="scenes",
                       pred="pred_w", gt="gt_pose_w", sat="sat_fine"),
    "t2l_kitti": dict(cache="test_cache/t2l_kitti/test_ht23.npz",
                      cell_ids="cell_ids", gt_cell="gt_cell", scene="scene",
                      pred="pred_w", gt="gt_pose_w", sat="sat_fine"),
    "t2lpp_kitti": dict(cache="test_cache/t2lpp_kitti/test_feats.npz",
                        cell_ids="cell_ids", gt_cell="gt_cell", scene="scene",
                        pred="pred_w", gt="gt_pose_w", sat="sat_fine"),
    "t2l_s3dis": dict(cache="test_cache/t2l_s3dis/test_ht.npz",
                      cell_ids="cell_ids", gt_cell="gt_cell", scene="scene",
                      pred="pred_w", gt="gt_pose_w", sat="sat_fine"),
    "t2l_pandaset": dict(cache="test_cache/t2l_pandaset/test_ht.npz",
                         cell_ids="cell_ids", gt_cell="gt_cell", scene="scene",
                         pred="pred_w", gt="gt_pose_w", sat="sat_fine"),
}
RESAMPLES = 2000
SEED = 123


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def loop_bootstrap(diff, inv, n_cl, picks):
    """Independent per-replicate loop over the registered re-samples."""
    sum_d = np.bincount(inv, weights=diff, minlength=n_cl)
    cnt = np.bincount(inv, minlength=n_cl).astype(np.float64)
    vals = np.empty(picks.shape[0])
    for k in range(picks.shape[0]):
        p = picks[k].astype(np.int64)
        vals[k] = sum_d[p].sum() / cnt[p].sum()
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def main():
    res = json.load(open(os.path.join(HERE, "results", "EVAL_RESULTS.json"),
                         encoding="utf-8"))
    out = dict(evaluator_code_sha256=res["code"]["sha256"],
               protocol_sha256=res["protocol"]["sha256"], configs={})
    all_ok = True
    for cfg, spec in CONFIGS.items():
        c = res["configs"].get(cfg)
        if not c or c.get("status") != "DONE":
            out["configs"][cfg] = dict(status="NOT_DONE")
            all_ok = False
            continue
        z = np.load(os.path.join(HERE, spec["cache"]), allow_pickle=True)
        cell_ids = z[spec["cell_ids"]]
        gt_cell = z[spec["gt_cell"]]
        pred = np.asarray(z[spec["pred"]], np.float64)[:, :, :2]
        gt = np.asarray(z[spec["gt"]], np.float64)[:, :2]
        scene = np.asarray([str(x) for x in z[spec["scene"]]])
        p2 = np.asarray(z["p2_idx"], np.int64)
        sat = np.asarray(z[spec["sat"]], np.float64)
        zq = np.load(os.path.join(HERE, "results", f"per_query_{cfg}.npz"))
        n = len(cell_ids)
        assert n == c["n"] == len(zq["qidx"])
        d = np.linalg.norm(pred - gt[:, None, :], axis=2)
        same = np.zeros((n, 25), bool)
        if "cell_scene" in z.files:
            cs = z["cell_scene"].astype(str)
            same = cs == scene[:, None]
        else:
            for i in range(n):
                same[i] = np.array([str(x).split("_")[0] for x in cell_ids[i]]) == scene[i]
        # re-derive sat_fine hits from the cache
        s_sat = sat[np.arange(n)[:, None], p2]
        order_sat = p2[np.arange(n)[:, None], np.argsort(-s_sat, axis=1, kind="stable")]
        cells_sat = np.take_along_axis(cell_ids, order_sat, axis=1)
        hit_cell_sat = (cells_sat[:, 0] == gt_cell).astype(np.int8)
        pr = pred[np.arange(n), order_sat[:, 0]]
        sc = np.array([str(x).split("_")[0] for x in cells_sat[:, 0]])
        hit_top1_sat = ((sc == scene) & (np.linalg.norm(pr - gt, axis=1) <= 5.0)).astype(np.int8)

        chk = dict(n=n,
                   sat_hits_identical=bool(np.array_equal(hit_cell_sat, zq["hit_cell_sat"])
                                           and np.array_equal(hit_top1_sat, zq["hit_top1_sat"])),
                   sat_score_matches_results=bool(abs(hit_cell_sat.mean() - c["sat_fine"]["cell_R1"]) < 1e-12
                                                  and abs(hit_top1_sat.mean() - c["sat_fine"]["top1_5m"]) < 1e-12),
                   orders_are_pool_permutations=True,
                   abs_scores_match=True, comparisons_match=True,
                   max_abs_score_diff=0.0, max_abs_bootstrap_diff=0.0)
        # cluster keys (same definitions as the evaluator)
        sc_sorted = sorted(set(scene.tolist()))
        code = {s: i for i, s in enumerate(sc_sorted)}
        if cfg in ("t2l_kitti", "t2lpp_kitti"):
            keys = np.stack([np.array([code[s] for s in scene], np.float64),
                             np.round(gt[:, 0], 6), np.round(gt[:, 1], 6)], axis=1)
        else:
            keys = np.array([code[s] for s in scene], np.float64)
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = np.asarray(inv).ravel()
        n_cl = len(uniq)
        chk["n_clusters"] = n_cl
        chk["n_clusters_match"] = bool(n_cl == c["n_clusters"])
        chk["cluster_unit"] = c["cluster_unit"]
        rng = np.random.RandomState(SEED)
        picks = rng.randint(0, n_cl, size=(RESAMPLES, n_cl))

        per_seed = {}
        for seed, s in c["per_seed"].items():
            seed = int(seed)
            hits = {"sat_fine": dict(cell=zq["hit_cell_sat"], top1=zq["hit_top1_sat"])}
            for tag in ("newA", "newB"):
                order = zq[f"order_{tag}_s{seed}"]
                perm = bool(np.array_equal(np.sort(order.astype(np.int64), axis=1),
                                           np.sort(p2, axis=1)))
                chk["orders_are_pool_permutations"] &= perm
                cells = np.take_along_axis(cell_ids, order.astype(np.int64), axis=1)
                hc = (cells[:, 0] == gt_cell).astype(np.int8)
                pr = pred[np.arange(n), order[:, 0]]
                sc = np.array([str(x).split("_")[0] for x in cells[:, 0]])
                ht = ((sc == scene) & (np.linalg.norm(pr - gt, axis=1) <= 5.0)).astype(np.int8)
                chk["sat_hits_identical"] &= bool(np.array_equal(hc, zq[f"hit_cell_{tag}_s{seed}"])
                                                  and np.array_equal(ht, zq[f"hit_top1_{tag}_s{seed}"]))
                hits[tag] = dict(cell=hc, top1=ht)
                for metric, key in (("cell_R1", "cell"), ("top1_5m", "top1")):
                    diff = abs(float(hits[tag][key].mean()) - s[tag][metric])
                    chk["max_abs_score_diff"] = max(chk["max_abs_score_diff"], diff)
                    chk["abs_scores_match"] &= diff < 1e-12
            comps = {}
            for name, (x, y) in {"newB_minus_newA": ("newB", "newA"),
                                 "newB_minus_sat_fine": ("newB", "sat_fine"),
                                 "newA_minus_sat_fine": ("newA", "sat_fine")}.items():
                comps[name] = {}
                for metric, key in (("top1_5m", "top1"), ("cell_R1", "cell")):
                    diff = hits[x][key].astype(np.float64) - hits[y][key].astype(np.float64)
                    delta, lo, hi = loop_bootstrap(diff, inv, n_cl, picks)
                    st = s["comparisons"][name][metric]
                    w = int((diff > 0).sum()); l = int((diff < 0).sum()); t = int((diff == 0).sum())
                    ok = (abs(delta - st["delta"]) < 1e-12 and
                          abs(lo - st["ci95"][0]) < 1e-9 and
                          abs(hi - st["ci95"][1]) < 1e-9 and
                          (w, l, t) == (st["win"], st["loss"], st["tie"]))
                    chk["comparisons_match"] &= ok
                    chk["max_abs_bootstrap_diff"] = max(
                        chk["max_abs_bootstrap_diff"], abs(lo - st["ci95"][0]),
                        abs(hi - st["ci95"][1]))
                    comps[name][metric] = dict(delta=delta, ci95=[lo, hi],
                                               win=w, loss=l, tie=t)
            per_seed[seed] = comps
        chk["passed"] = bool(chk["sat_hits_identical"] and chk["sat_score_matches_results"]
                             and chk["orders_are_pool_permutations"]
                             and chk["abs_scores_match"] and chk["comparisons_match"]
                             and chk["n_clusters_match"])
        chk["recomputed"] = per_seed
        out["configs"][cfg] = chk
        all_ok &= chk["passed"]
        print("[%s] verify %s  scores %.1e  boot %.1e" %
              (cfg, "PASS" if chk["passed"] else "FAIL",
               chk["max_abs_score_diff"], chk["max_abs_bootstrap_diff"]), flush=True)
    out["all_passed"] = bool(all_ok)
    with open(os.path.join(HERE, "results", "VERIFICATION.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("ALL_PASSED", out["all_passed"], flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
