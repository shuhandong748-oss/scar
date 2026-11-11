"""LSEL fixed-head appended exploratory TEST evaluation (CPU, reuse-only).

Scope: the 20 delivered newA/newB small scorers registered in PROTOCOL_20261028.json.
No Coarse/Fine, no feature regeneration, no training, no ABCDE queue access.

For every config/seed this script
  * verifies head sha256 and cache sha256/md5 against the frozen protocol,
  * checks input invariants (canonical feature order, p2 permutation, labels rule,
    shared Top-25 pool, R@25/top25 coverage identical across methods),
  * gates the sat_fine scorer against the published official row,
  * forwards newA / newB / sat_fine on the frozen cache,
  * writes per-query hit vectors, absolute scores, paired differences with a
    shared per-config cluster bootstrap (2000 resamples, seed 123),
  * independently recomputes rank-1 predictions with an argmax+first-tie path and
    a pure-Python loop over a deterministic 500-query sample.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
HEAD_ROOT = os.path.join(os.path.dirname(HERE), "LSEL_FIXED_HEAD_20261028")
PROTOCOL = os.path.join(HERE, "PROTOCOL_20261028.json")

CONFIGS = {
    "mncl_s3dis": dict(
        cache="test_cache/mncl_s3dis/test_ht23.npz", hash_kind="md5",
        feats="feats", cell_ids="cids", gt_cell="gt_cell", scene="scenes",
        cell_scene="cell_scene", p2="p2_idx", sat="sat_fine",
        pred="pred_w", gt="gt_pose_w", query_ids="query_ids", style="mncl",
        seeds=[0, 1], ref="references/s3dis_mncl_eval_test_metrics.json"),
    "t2l_kitti": dict(
        cache="test_cache/t2l_kitti/test_ht23.npz", hash_kind="sha256",
        feats="feats", cell_ids="cell_ids", gt_cell="gt_cell", scene="scene",
        cell_scene=None, p2="p2_idx", sat="sat_fine",
        pred="pred_w", gt="gt_pose_w", query_ids="query_ids", style="mlp",
        seeds=[0, 1, 2], ref="references/kitti_t2l_eval_test_metrics.json"),
    "t2lpp_kitti": dict(
        cache="test_cache/t2lpp_kitti/test_feats.npz", hash_kind="sha256",
        feats="feats", cell_ids="cell_ids", gt_cell="gt_cell", scene="scene",
        cell_scene=None, p2="p2_idx", sat="sat_fine",
        pred="pred_w", gt="gt_pose_w", query_ids="query_ids", style="mlp",
        seeds=[0, 1, 2], ref="references/kitti_t2lpp_eval_test_metrics.json"),
    "t2l_s3dis": dict(
        cache="test_cache/t2l_s3dis/test_ht.npz", hash_kind="sha256",
        feats="feats", cell_ids="cell_ids", gt_cell="gt_cell", scene="scene",
        cell_scene=None, p2="p2_idx", sat="sat_fine",
        pred="pred_w", gt="gt_pose_w", query_ids=None, style="mlp",
        seeds=[0], ref="references/s3dis_t2l_eval_test_metrics.json"),
    "t2l_pandaset": dict(
        cache="test_cache/t2l_pandaset/test_ht.npz", hash_kind="sha256",
        feats="feats", cell_ids="cell_ids", gt_cell="gt_cell", scene="scene",
        cell_scene=None, p2="p2_idx", sat="sat_fine",
        pred="pred_w", gt="gt_pose_w", query_ids=None, style="mlp",
        seeds=[2], ref="references/pandaset_seed0_eval_test_metrics.json"),
}
RESAMPLES = 2000
SEED = 123


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Net(nn.Module):
    """Checkpoint schema: sequential stored under `net.*`."""

    def __init__(self, nf):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nf, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)


def make_net(nf):
    return Net(nf)


def load_head(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    names = [str(x) for x in ck["feature_names"]]
    net = make_net(len(names))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return ck, net, names


def head_logits(ck, net, feats, style, batch=200000):
    if style == "mncl":
        z = feats[..., ck["keep"]].astype(np.float64)
        z = (z - ck["mean"]) / ck["std"]
        z[..., ck["pinned"]] = 0.0
        x = z.astype(np.float32)
    else:
        mean = np.asarray(ck["mean"], np.float32)
        std = np.asarray(ck["std"], np.float32)
        x = ((feats - mean) / std).astype(np.float32)
    flat = x.reshape(-1, x.shape[-1])
    out = np.empty(len(flat), np.float64)
    with torch.no_grad():
        for b in range(0, len(flat), batch):
            out[b:b + batch] = net(torch.from_numpy(flat[b:b + batch])).numpy()
    return out.reshape(feats.shape[:2])


def order_from_scores(scores, p2):
    s = np.take_along_axis(np.asarray(scores, np.float64), p2, axis=1)
    o = np.argsort(-s, axis=1, kind="stable")
    return np.take_along_axis(p2, o, axis=1)


def rank1_from_scores(scores, p2):
    """Independent path: max value, first occurrence in P2 order."""
    s = np.take_along_axis(np.asarray(scores, np.float64), p2, axis=1)
    first = np.argmax(s == s.max(axis=1, keepdims=True), axis=1)
    return p2[np.arange(len(p2)), first]


def rank1_loop(scores, p2, upto):
    """Pure-Python independent path on the first `upto` queries."""
    out = np.empty(upto, np.int64)
    for i in range(upto):
        p = p2[i]
        s = np.asarray(scores[i], np.float64)[p]
        best, best_j = -np.inf, -1
        for j in range(len(p)):
            if s[j] > best:
                best, best_j = s[j], j
        out[i] = p[best_j]
    return out


def scene_of_cells(cells):
    return np.array([str(c).split("_")[0] for c in cells])


def rank_metrics(order, cell_ids, gt_cell, pred, gt_xy, scene_str):
    n = len(order)
    cells = np.take_along_axis(cell_ids, order, axis=1)
    cell_hit = (cells[:, 0] == gt_cell).astype(np.int8)
    r1 = order[:, 0]
    pr = np.asarray(pred, np.float64)[np.arange(n), r1][:, :2]
    d = np.linalg.norm(pr - np.asarray(gt_xy, np.float64), axis=1)
    sc = scene_of_cells(cells[:, 0])
    top1 = ((sc == scene_str) & (d <= 5.0)).astype(np.int8)
    return dict(cell=cell_hit, top1=top1, cells=cells)


def cluster_keys(scene_str, gt_xy, spec):
    sc = sorted(set(scene_str.tolist()))
    code = {s: i for i, s in enumerate(sc)}
    if spec in ("t2l_kitti", "t2lpp_kitti"):
        keys = np.stack([
            np.array([code[s] for s in scene_str], np.float64),
            np.round(np.asarray(gt_xy, np.float64)[:, 0], 6),
            np.round(np.asarray(gt_xy, np.float64)[:, 1], 6)], axis=1)
        return keys, "scene + round(pose_w.xy, 6)"
    return np.array([code[s] for s in scene_str], np.float64), "scene"


def make_picks(keys):
    """Multipicity matrix of the registered cluster re-samples.

    Returns `mult` with shape (RESAMPLES, n_cl): mult[r, c] is the number of
    times cluster c was drawn in re-sample r.  A matrix product with a matrix
    of *indices* would weight by index value, which is not the bootstrap
    statistic, so the draws are converted to multiplicities first.
    """
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    n_cl = len(uniq)
    cnt = np.bincount(inv, minlength=n_cl).astype(np.float64)
    rng = np.random.RandomState(SEED)
    idx = rng.randint(0, n_cl, size=(RESAMPLES, n_cl))
    mult = np.zeros((RESAMPLES, n_cl), np.float64)
    np.add.at(mult, (np.arange(RESAMPLES)[:, None], idx), 1.0)
    return inv, n_cl, cnt, mult, cnt @ mult.T


def bootstrap(diff, inv, n_cl, mult, boot_cnt):
    sum_d = np.bincount(inv, weights=np.asarray(diff, np.float64),
                        minlength=n_cl)
    vals = (sum_d @ mult.T) / boot_cnt
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return dict(delta=float(np.mean(diff)), ci95=[float(lo), float(hi)],
                ci_excludes_zero=bool(lo > 0 or hi < 0),
                win=int((np.asarray(diff) > 0).sum()),
                loss=int((np.asarray(diff) < 0).sum()),
                tie=int((np.asarray(diff) == 0).sum()),
                n=int(len(diff)))


def ref_values(path):
    j = json.load(open(path, encoding="utf-8"))
    if "results_cell" in j:
        c = j["results_cell"]["sat_fine"]["R@1"]
        t = j["results_loc"]["sat_fine"]["top1@5m"]
    else:
        c = j["metrics"]["sat_fine"]["cell"]["R@1"]
        t = j["metrics"]["sat_fine"]["loc"]["top1@5m"]
    return float(c), float(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="")
    ap.add_argument("--out_dir", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    torch.set_num_threads(10)
    proto = json.load(open(PROTOCOL, encoding="utf-8"))
    models = {(m["config"], int(m["seed"])): m for m in proto["models"]}
    caches = proto["caches"]
    wanted = [c for c in CONFIGS if c in a.configs.split(",")] if a.configs \
        else list(CONFIGS)

    record = dict(protocol=dict(id=proto["id"], sha256=sha256(PROTOCOL)),
                  code=dict(path=os.path.abspath(__file__),
                            sha256=sha256(os.path.abspath(__file__))),
                  torch=torch.__version__, numpy=np.__version__,
                  torch_threads=int(torch.get_num_threads()),
                  started=time.strftime("%Y-%m-%dT%H:%M:%S"),
                  configs={}, gates={})
    heads_ok, caches_ok = [], []
    for (cfg, seed), m in models.items():
        for tag in ("newA", "newB"):
            p = os.path.join(HEAD_ROOT, m[tag]["path"])
            h = sha256(p)
            heads_ok.append(h == m[tag]["sha256"])
            assert heads_ok[-1], (cfg, seed, tag, h, m[tag]["sha256"])
    for cfg, spec in CONFIGS.items():
        p = os.path.join(HERE, spec["cache"])
        h = (md5(p) if spec["hash_kind"] == "md5" else sha256(p))
        caches_ok.append(h == caches[cfg]["hash"])
        assert caches_ok[-1], (cfg, h, caches[cfg]["hash"])
    record["gates"]["head_sha256"] = dict(passed=bool(all(heads_ok)),
                                          n=len(heads_ok))
    record["gates"]["cache_hash"] = dict(passed=bool(all(caches_ok)),
                                         n=len(caches_ok))
    print("gates: heads %d/%d, caches %d/%d" %
          (sum(heads_ok), len(heads_ok), sum(caches_ok), len(caches_ok)),
          flush=True)

    for cfg in wanted:
        spec = CONFIGS[cfg]
        t0 = time.time()
        z = np.load(os.path.join(HERE, spec["cache"]), allow_pickle=True)
        feats = z[spec["feats"]]
        cell_ids = z[spec["cell_ids"]]
        gt_cell = z[spec["gt_cell"]]
        p2 = np.asarray(z[spec["p2"]], np.int64)
        sat = np.asarray(z[spec["sat"]], np.float64)
        pred = z[spec["pred"]]
        gt_xy = np.asarray(z[spec["gt"]], np.float64)[:, :2]
        scene_str = np.asarray([str(x) for x in z[spec["scene"]]])
        labels = np.asarray(z["labels"], np.int64)
        n, K, nf = feats.shape
        fnames = [str(x) for x in z["feature_names"]]
        assert K == 25 and nf == 23 and len(fnames) == 23
        assert n == caches[cfg]["queries"], (n, caches[cfg]["queries"])
        assert np.all(np.sort(p2, axis=1) == np.arange(K)[None, :]), "p2 not a permutation"
        if spec["query_ids"]:
            assert len(set([str(c) for c in z[spec["query_ids"]]])) == n

        if spec["cell_scene"] is not None:
            cell_scene = z[spec["cell_scene"]].astype(str)
            assert cell_scene.shape == (n, K), cell_scene.shape
            same = cell_scene == scene_str[:, None]
        else:
            same = np.zeros((n, K), bool)
            for i in range(n):
                same[i] = scene_of_cells(cell_ids[i]) == scene_str[i]
        d = np.linalg.norm(np.asarray(pred, np.float64)[:, :, :2]
                           - gt_xy[:, None, :], axis=2)
        rule = (same & (d <= 5.0)).astype(np.int64)
        n_lab_mismatch = int((rule != labels).sum())
        near = np.abs(d - 5.0) < 1e-6
        lab_mismatch_far = int(((rule != labels) & ~near).sum())
        pool_has_gt = np.array([gt_cell[i] in set(cell_ids[i].tolist())
                                for i in range(n)], np.int8)
        pool_top25 = ((np.where(same, d, np.inf).min(axis=1) <= 5.0)).astype(np.int8)

        order_sat = order_from_scores(sat, p2)
        h_sat = rank_metrics(order_sat, cell_ids, gt_cell, pred, gt_xy, scene_str)
        r1_sat_argmax = rank1_from_scores(sat, p2)
        assert np.array_equal(r1_sat_argmax, order_sat[:, 0])
        assert np.array_equal(rank1_loop(sat, p2, min(500, n)),
                              order_sat[:min(500, n), 0])
        ref_c, ref_t = ref_values(os.path.join(HERE, spec["ref"]))
        got_c = float(h_sat["cell"].mean())
        got_t = float(h_sat["top1"].mean())
        gate_sat = dict(passed=bool(abs(got_c - ref_c) < 1e-9 and
                                    abs(got_t - ref_t) < 1e-9),
                        got=dict(cell_R1=got_c, top1_5m=got_t),
                        ref=dict(cell_R1=ref_c, top1_5m=ref_t,
                                 path=spec["ref"],
                                 sha256=sha256(os.path.join(HERE, spec["ref"]))),
                        abs_diff=[abs(got_c - ref_c), abs(got_t - ref_t)])

        keys, cluster_label = cluster_keys(scene_str, gt_xy, cfg)
        inv, n_cl, cnt, picks, boot_cnt = make_picks(keys)
        rec = dict(n=n, k=K, n_clusters=n_cl, cluster_unit=cluster_label,
                   resamples=RESAMPLES, rng_seed=SEED,
                   invariants=dict(
                       feature_names=fnames,
                       p2_permutation=True,
                       labels_mismatch=n_lab_mismatch,
                       labels_mismatch_far_from_threshold=lab_mismatch_far,
                       labels_rule="same scene and ||pred_w - gt_pose_w|| <= 5 m",
                       sat_fine_gate=gate_sat,
                       pool_has_gt_R25=float(pool_has_gt.mean()),
                       pool_top25_5m=float(pool_top25.mean())))
        rec["invariants"]["gate_passed"] = bool(gate_sat["passed"] and
                                                lab_mismatch_far == 0)
        record["gates"][cfg] = rec["invariants"]
        print("[%s] n=%d clusters=%d sat_fine gate %s (got %.6f/%.6f ref %.6f/%.6f) "
              "labels mismatch %d (far %d)" %
              (cfg, n, n_cl, "PASS" if gate_sat["passed"] else "FAIL",
               got_c, got_t, ref_c, ref_t, n_lab_mismatch, lab_mismatch_far),
              flush=True)
        if not rec["invariants"]["gate_passed"]:
            rec["status"] = "GATE_FAILED_STOPPED"
            record["configs"][cfg] = rec
            continue

        per_query = dict(qidx=np.arange(n), scene=scene_str,
                         gt_cell=gt_cell.astype(str),
                         cluster=keys, hit_cell_sat=h_sat["cell"],
                         hit_top1_sat=h_sat["top1"],
                         order_sat=order_sat.astype(np.int8))
        rec["per_seed"] = {}
        for seed in spec["seeds"]:
            m = models[(cfg, seed)]
            pair = {}
            for tag in ("newA", "newB"):
                ck, net, names = load_head(os.path.join(HEAD_ROOT, m[tag]["path"]))
                assert names == fnames, (cfg, seed, tag)
                lg = head_logits(ck, net, feats, spec["style"])
                order = order_from_scores(lg, p2)
                assert np.array_equal(np.sort(order, axis=1),
                                      np.sort(p2, axis=1)), "order not a permutation of the pool"
                hh = rank_metrics(order, cell_ids, gt_cell, pred, gt_xy, scene_str)
                assert np.array_equal(rank1_from_scores(lg, p2), order[:, 0]), \
                    (cfg, seed, tag, "argmax path")
                assert np.array_equal(rank1_loop(lg, p2, min(500, n)),
                                      order[:min(500, n), 0]), (cfg, seed, tag, "loop path")
                pair[tag] = dict(epoch=int(m[tag]["epoch"]),
                                 sha256=m[tag]["sha256"],
                                 cell_R1=float(hh["cell"].mean()),
                                 top1_5m=float(hh["top1"].mean()),
                                 hits=hh)
                per_query[f"hit_cell_{tag}_s{seed}"] = hh["cell"]
                per_query[f"hit_top1_{tag}_s{seed}"] = hh["top1"]
                per_query[f"order_{tag}_s{seed}"] = order.astype(np.int8)
                print("  %s s%d %s ep%d cell %.4f @5m %.4f" %
                      (cfg, seed, tag, m[tag]["epoch"],
                       pair[tag]["cell_R1"], pair[tag]["top1_5m"]), flush=True)
            comps = {}
            for name, (x, y) in {
                    "newB_minus_newA": ("newB", "newA"),
                    "newB_minus_sat_fine": ("newB", "sat_fine"),
                    "newA_minus_sat_fine": ("newA", "sat_fine")}.items():
                comps[name] = {}
                for metric, key in (("top1_5m", "top1"), ("cell_R1", "cell")):
                    hb = h_sat if y == "sat_fine" else pair[y]["hits"]
                    ha = h_sat if x == "sat_fine" else pair[x]["hits"]
                    diff = ha[key].astype(np.float64) - hb[key].astype(np.float64)
                    comps[name][metric] = bootstrap(diff, inv, n_cl, picks, boot_cnt)
                    print("    %s %s %+0.4f [%+.4f,%+.4f] w/l/t %d/%d/%d" %
                          (name, metric, comps[name][metric]["delta"],
                           comps[name][metric]["ci95"][0],
                           comps[name][metric]["ci95"][1],
                           comps[name][metric]["win"],
                           comps[name][metric]["loss"],
                           comps[name][metric]["tie"]), flush=True)
            rec["per_seed"][int(seed)] = dict(
                newA=dict(epoch=pair["newA"]["epoch"], sha256=pair["newA"]["sha256"],
                          cell_R1=pair["newA"]["cell_R1"],
                          top1_5m=pair["newA"]["top1_5m"]),
                newB=dict(epoch=pair["newB"]["epoch"], sha256=pair["newB"]["sha256"],
                          cell_R1=pair["newB"]["cell_R1"],
                          top1_5m=pair["newB"]["top1_5m"]),
                comparisons=comps)
        rec["sat_fine"] = dict(cell_R1=got_c, top1_5m=got_t)
        rec["status"] = "DONE"
        rec["runtime_s"] = round(time.time() - t0, 1)
        record["configs"][cfg] = rec
        np.savez_compressed(os.path.join(a.out_dir, f"per_query_{cfg}.npz"),
                            **per_query)
        print("[%s] done in %.1fs" % (cfg, rec["runtime_s"]), flush=True)

    record["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out = os.path.join(a.out_dir, "EVAL_RESULTS.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, ensure_ascii=False)
    print("WROTE", out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
