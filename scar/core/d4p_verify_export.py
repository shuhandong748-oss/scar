#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent verification of one export cache (CPU, no model, no GPU).

Nothing is taken from the exporter's own claims: the coarse scores are
recomputed from the saved encodings, the Top-25 set and order are re-derived,
the Fine decode is re-derived from offsets, and cell recall / localization are
recomputed from the saved ids and coordinates and compared with metrics.json.
An optional --probe_dir checks that a short export and the full export produce
bit-identical arrays for the shared queries (chunk/chunking invariance).
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["npm3d", "s3dis"])
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--export_dir", required=True)
    ap.add_argument("--probe_dir", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    d = S.domain_cfg(a.domain)
    ed = os.path.abspath(a.export_dir)
    checks = {}

    def check(name, ok, **detail):
        checks[name] = dict(PASS=bool(ok), **detail)
        if not ok:
            print(f"FAIL {name}: {detail}", flush=True)
        return ok

    meta_path = os.path.join(ed, "EXPORT_META.json")
    meta = S.read_json(meta_path)
    check("meta_domain_split", meta["domain"] == a.domain and meta["split"] == a.split,
          domain=meta["domain"], split=meta["split"])
    check("meta_complete", meta.get("complete") is True)
    check("k25", meta["k"] == 25)

    for name, want in (meta.get("cache_sha256") or {}).items():
        p = os.path.join(ed, name)
        check(f"sha:{name}", os.path.isfile(p) and S.sha256(p) == want)

    z = np.load(os.path.join(ed, "candidates.npz"), allow_pickle=True)
    cc = np.load(os.path.join(ed, "coarse_enc.npz"), allow_pickle=True)
    ids = np.asarray([[str(x) for x in row] for row in z["cell_ids"]])
    cscene = np.asarray([[str(x) for x in row] for row in z["cell_scene"]])
    cindex = np.asarray(z["cell_index"], np.int64)
    scores = np.asarray(z["scores"], np.float64)
    offsets = np.asarray(z["offsets"], np.float64)
    pred_w = np.asarray(z["pred_w"], np.float64)
    bmin = np.asarray(z["cell_bbox_min"], np.float64)
    csize = np.asarray(z["cell_size"], np.float64)
    gt_cell = [str(x) for x in z["gt_cell"]]
    gt_pose = np.asarray(z["gt_pose_w"], np.float64)
    scene = np.asarray([str(x) for x in z["scene"]])
    qids = [str(x) for x in z["query_ids"]]
    n, K = ids.shape
    check("queries", n == d["expected"][a.split] == meta["n_queries"],
          got=n, expected=d["expected"][a.split])
    check("K", K == 25)
    check("shape_agree", all(x.shape[:2] == (n, K) for x in
                             (scores, offsets, pred_w, bmin, csize, cindex)))
    # Per-query uniqueness is required.  Identical candidate rows *across* queries
    # are legitimate: two poses of the same cell can carry the same hint text, so
    # the same text/cell encodings yield the same Top-25.  The count of distinct
    # rows is therefore reported, not demanded to equal n.
    n_distinct_rows = len(set(map(tuple, ids.tolist())))
    check("unique_ids_per_query", all(len(set(r)) == K for r in ids.tolist()),
          n_distinct_rows=n_distinct_rows, n_queries=n)
    groups = {}
    for i, row in enumerate(ids.tolist()):
        groups.setdefault(tuple(row), []).append(i)
    dup_groups = [dict(queries=v, gt_cells=[gt_cell[i] for i in v],
                       same_gt_cell=len({gt_cell[i] for i in v}) == 1)
                  for v in groups.values() if len(v) > 1]
    check("unique_query_ids", len(set(qids)) == n)
    check("clusters", len({str(s) for s in scene}) == d["expected_clusters"][a.split],
          got=len({str(s) for s in scene}),
          expected=d["expected_clusters"][a.split])

    cells = [str(x) for x in cc["cell_ids"]]
    check("coarse_cells_unique", len(set(cells)) == len(cells))
    tg = np.asarray(cc["tg"], np.float64)
    cg = np.asarray(cc["cg"], np.float64)
    check("enc_shapes", tg.shape[0] == n and cg.shape[0] == len(cells) and
          tg.shape[1] == cg.shape[1])
    idx_ok = (cindex >= 0).all() and (cindex < len(cells)).all()
    check("index_range", bool(idx_ok))
    if idx_ok:
        enc_names = np.asarray(cells)
        check("candidate_ids_match_enc", bool((enc_names[cindex] == ids).all()))
        # recompute cosine from the saved encodings (float64)
        cos_re = np.zeros((n, K))
        for i in range(n):
            cos_re[i] = cg[cindex[i]] @ tg[i]
        diff = float(np.abs(cos_re - scores).max())
        check("scores_recomputed", diff <= 2e-5, max_abs_diff=diff)
        check("order_descending_cosine",
              bool(all((np.diff(scores[i]) <= 1e-12).all() for i in range(n))))
        # re-derive the Top-25 from the full score matrix
        score_full = tg @ cg.T
        top = np.argsort(-score_full, axis=1)[:, :K]
        check("top25_rederived", bool((top == cindex).all()),
              mismatches=int((top != cindex).any(1).sum()))
        check("top25_scores", float(np.abs(
            np.take_along_axis(score_full, top, 1) - scores).max()) <= 2e-5)

    check("pred_w_from_offsets",
          float(np.abs(bmin + offsets * csize[:, :, None] - pred_w).max()) <= 2e-5,
          max_abs_diff=float(np.abs(bmin + offsets * csize[:, :, None] - pred_w).max()))

    # --- same-forward metrics, recomputed independently -------------------
    rec_cell = {k: np.zeros(n, np.int8) for k in KS_CELL}
    rec_loc = {(k, t): np.zeros(n, np.int8) for k in KS_LOC for t in THRESHS}
    for i in range(n):
        for k in KS_CELL:
            rec_cell[k][i] = int(gt_cell[i] in ids[i][:k].tolist())
        dist = np.linalg.norm(pred_w[i] - gt_pose[i][None, :], axis=1)
        dist[cscene[i] != scene[i]] = np.inf
        for k in KS_LOC:
            for t in THRESHS:
                rec_loc[(k, t)][i] = int(np.min(dist[:k]) <= t)
    met = S.read_json(os.path.join(ed, "metrics.json"))
    cell_ok = all(abs(float(met["cell_R"][f"@{k}"]) - rec_cell[k].mean()) <= 1e-12
                  for k in KS_CELL)
    loc_ok = all(abs(float(met["localization"][f"Top{k}@{int(t)}m"])
                      - rec_loc[(k, t)].mean()) <= 1e-12
                  for k in KS_LOC for t in THRESHS)
    check("metrics_cell_recomputed", cell_ok)
    check("metrics_loc_recomputed", loc_ok)

    pq = [json.loads(line) for line in open(os.path.join(ed, "per_query.jsonl"),
                                            encoding="utf-8")]
    check("per_query_count", len(pq) == n)
    sample = list(range(0, n, max(1, n // 50)))
    check("per_query_ids", all(pq[i]["ids"] == ids[i].tolist() for i in sample))
    check("per_query_pred_w", float(max(
        np.abs(np.asarray(pq[i]["pred_w"]) - pred_w[i]).max() for i in sample)) <= 1e-5)
    check("per_query_hits", all(
        all(pq[i]["hit_cell"][f"R{k}"] == int(rec_cell[k][i]) for k in KS_CELL) and
        all(pq[i]["hit_loc"][f"top{k}@{int(t)}m"] == int(rec_loc[(k, t)][i])
            for k in KS_LOC for t in THRESHS) for i in sample))

    # --- probe vs full: identical arrays on the shared prefix -------------
    probe = None
    if a.probe_dir:
        pd = os.path.abspath(a.probe_dir)
        pz = np.load(os.path.join(pd, "candidates.npz"), allow_pickle=True)
        pn = int(np.asarray(pz["cell_ids"]).shape[0])
        pmeta = S.read_json(os.path.join(pd, "EXPORT_META.json"))
        same = dict(
            ids=bool((np.asarray([[str(x) for x in r] for r in pz["cell_ids"]])
                      == ids[:pn]).all()),
            scores=float(np.abs(np.asarray(pz["scores"], np.float64)
                                - scores[:pn]).max()),
            offsets=float(np.abs(np.asarray(pz["offsets"], np.float64)
                                 - offsets[:pn]).max()),
            pred_w=float(np.abs(np.asarray(pz["pred_w"], np.float64)
                                - pred_w[:pn]).max()),
            queries_same=bool((np.asarray([str(x) for x in pz["gt_cell"]])
                               == np.asarray(gt_cell[:pn])).all()),
            seed_same=pmeta.get("seed") == meta.get("seed"),
        )
        probe = same
        # machine-precision tolerance: the short and full runs must agree on the
        # shared queries; only cuDNN-level noise is allowed, not a sampling change
        check("probe_prefix_identical",
              same["ids"] and same["scores"] <= 1e-5 and same["offsets"] <= 1e-6
              and same["pred_w"] <= 1e-5 and same["queries_same"]
              and same["seed_same"], n_probe=pn, **same)

    out = dict(
        status="OK" if all(c["PASS"] for c in checks.values()) else "FAIL",
        domain=a.domain, split=a.split, export_dir=ed,
        export_meta_sha256=S.sha256(meta_path),
        cache_sha256=meta.get("cache_sha256"),
        gate_binding=S.load_gate(),
        protocol_freeze_sha256=meta.get("protocol_freeze_sha256"),
        n_queries=n, n_cells=len(cells), n_clusters=int(len(set(scene.tolist()))),
        duplicate_candidate_rows=dict(n_distinct=n_distinct_rows,
                                      n_groups=len(dup_groups),
                                      groups=dup_groups[:20]),
        recomputed_cell={f"@{k}": float(rec_cell[k].mean()) for k in KS_CELL},
        recomputed_loc={f"Top{k}@{int(t)}m": float(rec_loc[(k, t)].mean())
                        for k in KS_LOC for t in THRESHS},
        probe=probe, checks=checks,
        failures=[k for k, v in checks.items() if not v["PASS"]],
        code_sha256={f: S.sha256(os.path.join(HERE, f))
                     for f in ("d4p_verify_export.py", "scar_common.py")},
    )
    S.write_json(out, a.out)
    print(json.dumps({k: v for k, v in out.items()
                      if k in ("status", "n_queries", "n_cells", "n_clusters",
                               "failures")}, indent=2), flush=True)
    print("saved", a.out, flush=True)
    return 0 if out["status"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
