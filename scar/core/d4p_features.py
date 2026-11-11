#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build canonical-23 candidate lists for one exported domain/split.

Inputs: the export's candidates.npz + coarse_enc.npz and the frozen cells/poses
(as maps only).  Ground truth labels the lists and is never a feature.

Frozen definition (CORR-D4P-SCAR-FORMAL-v1):
  K=25, pool_norm=rank_norm=24, tied_norm=25, lam=0.5, min_match=4
  P0 = stable argsort(-(cos + 0.5*sp))
  P2 = P0 with the maximum-presence tie block reordered by descending sc_center
  anchor = P2 top-1
  label = same scene AND ||pred_w - gt_pose_w|| <= 5 m
  valid = 1 <= positives < 25
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scar_common as S  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["npm3d", "s3dis"])
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--export_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol_sha", required=True)
    ap.add_argument("--freeze_sha", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    a.export_dir = os.path.abspath(a.export_dir)
    a.out = os.path.abspath(a.out)
    d = S.domain_cfg(a.domain)
    gate = S.load_gate()
    dom = S.import_adapter(a.domain)
    adv = dom.apply(d["base"], verbose=False)
    COLS, LABS, NQ = S.domain_vocabulary(a.domain)

    meta_path = os.path.join(a.export_dir, "EXPORT_META.json")
    export_meta = S.read_json(meta_path)
    if export_meta["split"] != a.split or export_meta["domain"] != a.domain:
        raise SystemExit("export meta does not match the requested domain/split")
    if export_meta["k"] != 25:
        raise SystemExit("export is not K=25")

    sys.path.insert(0, d["vendor"])
    import common as C          # noqa: E402
    import sa_geom as G         # noqa: E402
    import features_fix as F    # noqa: E402

    os.chdir(dom.CORE)
    sys.path.insert(0, dom.CORE)
    from dataloading.kitti360pose.base import Kitti360BaseDataset  # noqa: E402
    from dataloading.kitti360pose.cells import Kitti360CoarseDatasetMulti  # noqa: E402
    from torch_geometric.transforms import FixedPoints  # noqa: E402

    splits, man = dom.split_scene_names(d["base"])
    scenes = splits[a.split]

    z = np.load(os.path.join(a.export_dir, "candidates.npz"), allow_pickle=True)
    pool_ids = [[str(c) for c in row] for row in z["cell_ids"]]
    cell_scene = ([[str(c) for c in row] for row in z["cell_scene"]]
                  if "cell_scene" in z.files else None)
    scores_exp = np.asarray(z["scores"], np.float64) if "scores" in z.files else None
    offs = np.asarray(z["offsets"], np.float64)
    pred_w = np.asarray(z["pred_w"], np.float64)
    gt_cell = [str(x) for x in z["gt_cell"]]
    gt_pose_w = np.asarray(z["gt_pose_w"], np.float64)
    q_scene = [str(x) for x in z["scene"]]
    n = len(pool_ids)
    K = len(pool_ids[0])
    if n != d["expected"][a.split]:
        raise SystemExit(f"{a.domain} {a.split}: {n} queries != {d['expected'][a.split]}")
    if not all(len(p) == K for p in pool_ids) or K != 25:
        raise SystemExit("candidate pool is not K=25")
    if not all(len(set(p)) == K for p in pool_ids):
        raise SystemExit("duplicate candidate ids")

    ds = Kitti360CoarseDatasetMulti(d["base"], list(scenes), FixedPoints(256),
                                    shuffle_hints=False, flip_poses=False)
    poses = ds.all_poses
    cells_dict = {str(c.id): c for c in ds.all_cells}
    if len(poses) != n:
        raise SystemExit(f"pose count {len(poses)} != export {n}")
    if [str(p.cell_id) for p in poses] != gt_cell:
        raise SystemExit("query order mismatch")
    if [str(cells_dict[gt_cell[i]].scene_name) for i in range(n)] != q_scene:
        raise SystemExit("query scene mismatch")
    outside = {c for p in pool_ids for c in p if c not in cells_dict}
    if outside:
        raise SystemExit(f"candidate outside split cells: {list(outside)[:5]}")
    if cell_scene is not None:
        if [[str(cells_dict[c].scene_name) for c in row] for row in pool_ids] != cell_scene:
            raise SystemExit("cell_scene column disagrees with frozen cells")

    cc = np.load(os.path.join(a.export_dir, "coarse_enc.npz"), allow_pickle=True)
    tg = np.asarray(cc["tg"], np.float64)
    cg = np.asarray(cc["cg"], np.float64)
    id2row = {str(c): i for i, c in enumerate(cc["cell_ids"])}
    if tg.shape[0] != n:
        raise SystemExit("coarse_enc query count mismatch")

    grid = G.grid_points(33)
    aux = {}

    def aux_of(cid):
        x = aux.get(cid)
        if x is None:
            x = C.build_cell_aux(cells_dict[cid], grid, lambda o: o.get_color_text())
            aux[cid] = x
        return x

    n_done = min(n, a.limit) if a.limit else n
    feats = np.zeros((n_done, K, 23), np.float32)
    labels = np.zeros((n_done, K), np.uint8)
    valid = np.zeros(n_done, bool)
    p2_all = np.zeros((n_done, K), np.int16)
    p0_all = np.zeros((n_done, K), np.int16)
    coarse_rank = np.zeros((n_done, K), np.int16)
    cell_ids = np.empty((n_done, K), dtype="<U32")
    cell_scenes_out = np.empty((n_done, K), dtype="<U32")
    gt_cells = np.empty(n_done, dtype="<U32")
    scenes_out = np.empty(n_done, dtype="<U32")
    all_cos = np.zeros((n_done, K), np.float32)
    sp_arr = np.zeros((n_done, K), np.float32)
    sc_c_arr = np.zeros((n_done, K), np.float32)
    sc_f_arr = np.zeros((n_done, K), np.float32)
    sat_arr = np.zeros((n_done, K), np.float32)
    cos_check = {"checked": 0, "max_abs_diff": 0.0}

    t0 = time.time()
    n_pos = 0
    for i in range(n_done):
        cids = list(pool_ids[i])
        idx = np.array([id2row[c] for c in cids])
        cos_v = (cg[idx] @ tg[i]).astype(np.float64)
        if scores_exp is not None:
            cos_check["max_abs_diff"] = max(
                cos_check["max_abs_diff"], float(np.abs(cos_v - scores_exp[i]).max()))
            cos_check["checked"] += 1
        text = " ".join(Kitti360BaseDataset.create_hint_description(
            poses[i], cells_dict[gt_cell[i]]))
        parsed = G.parse_relations(text, COLS, LABS)
        if len(parsed) != NQ:
            raise SystemExit(f"query {i}: parsed {len(parsed)} relations != {NQ}: {text}")

        sp_v = np.array([C.matched_count(parsed, aux_of(c)["counts"]) / NQ for c in cids])
        sc_f = np.zeros(K)
        sat_f = np.zeros(K)
        sc_c = np.zeros(K)
        for r, cid in enumerate(cids):
            val, info = G.relation_score(parsed, aux_of(cid)["cm"], mode="C",
                                         point=offs[i, r])
            sc_f[r] = val
            sat_f[r] = info["satisfied"] / float(NQ)
            sc_c[r] = G.relation_score(parsed, aux_of(cid)["cm"], mode="C",
                                       point=(0.5, 0.5))[0]

        n_match = np.zeros(K)
        d_near = np.full(K, 3.0)
        d_cent = np.full(K, 3.0)
        n_obj = np.zeros(K)
        cell_c = np.zeros((K, 2))
        for r, cid in enumerate(cids):
            ids_m = C.matched_object_ids(parsed, aux_of(cid)["cm"])
            n_match[r] = len(ids_m) / NQ
            if ids_m:
                cen = aux_of(cid)["centers"][ids_m]
                d_near[r] = min(float(np.linalg.norm(cen - offs[i, r], axis=1).min()), 3.0)
                d_cent[r] = min(float(np.linalg.norm(cen.mean(0) - offs[i, r])), 3.0)
            n_obj[r] = aux_of(cid)["n_obj"] / 40.0
            cell_c[r] = np.asarray(cells_dict[cid].get_center()[0:2], float)

        p0_idx, p2_idx, tied_mask = F.compute_orders(cos_v, sp_v, sc_c, lam=C.LAMBDA)
        p0_rank_raw = np.empty(K, np.int64)
        p0_rank_raw[p0_idx] = np.arange(K)
        rows = np.arange(K, dtype=np.int64)
        feats[i] = F.feature_matrix(
            cos_v=cos_v, sp_v=sp_v, sc_center=sc_c, sc_fine=sc_f, sat_fine=sat_f,
            offs=offs[i], n_obj=n_obj, n_match=n_match, d_near=d_near, d_cent=d_cent,
            pred_w=pred_w[i], cell_c=cell_c, pool_rank=rows.astype(np.float64),
            p0_rank_raw=p0_rank_raw, p2_idx=p2_idx, tied_mask=tied_mask, rows=rows,
            pool_norm=K - 1, rank_norm=K - 1, tied_norm=K, anchor_idx=int(p2_idx[0]))
        p2_all[i] = p2_idx
        p0_all[i] = p0_idx

        gt_scene = q_scene[i]
        cand_scene = ([cell_scene[i][j] for j in range(K)] if cell_scene is not None
                      else [str(c).split("_")[0] for c in cids])
        same_scene = np.array([cs == gt_scene for cs in cand_scene])
        dist = np.linalg.norm(pred_w[i] - gt_pose_w[i][None, :], axis=1)
        lab = (same_scene & (dist <= 5.0)).astype(np.uint8)
        labels[i] = lab
        n_pos += int(lab.sum())
        valid[i] = 1 <= int(lab.sum()) < K

        order = np.argsort(-cos_v, kind="stable")
        coarse_rank[i][order] = np.arange(K)

        cell_ids[i] = np.asarray(cids, dtype="<U32")
        cell_scenes_out[i] = np.asarray(cand_scene, dtype="<U32")
        gt_cells[i] = gt_cell[i]
        scenes_out[i] = gt_scene
        all_cos[i] = cos_v
        sp_arr[i] = sp_v
        sc_c_arr[i] = sc_c
        sc_f_arr[i] = sc_f
        sat_arr[i] = sat_f
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f"{i+1}/{n_done} {el:.0f}s eta {(n_done-i-1)*el/(i+1)/60:.1f} min",
                  flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(
        a.out, feats=feats, labels=labels, valid=valid, p2_idx=p2_all, p0_idx=p0_all,
        coarse_rank=coarse_rank, cell_ids=cell_ids, cell_scene=cell_scenes_out,
        gt_cell=gt_cells, scene=scenes_out, gt_pose_w=gt_pose_w.astype(np.float32),
        cos=all_cos, sp=sp_arr, sc_center=sc_c_arr, sc_fine=sc_f_arr,
        sat_fine=sat_arr, offsets=offs.astype(np.float32),
        pred_w=pred_w.astype(np.float32),
        feature_names=np.asarray(F.FEATURE_NAMES, dtype="<U32"),
        split=np.asarray([a.split], dtype="<U32"))

    degenerate_columns = [n for n, all_zero in
                          (("sc_center", bool(np.allclose(sc_c_arr, 0))),
                           ("sc_fine", bool(np.allclose(sc_f_arr, 0))),
                           ("sat_fine", bool(np.allclose(sat_arr, 0))))
                          if all_zero]
    degeneracy = {
        "sc_center_all_zero": bool(np.allclose(sc_c_arr, 0)),
        "sc_fine_all_zero": bool(np.allclose(sc_f_arr, 0)),
        "sat_fine_all_zero": bool(np.allclose(sat_arr, 0)),
        "p2_equals_p0": bool(np.array_equal(p2_all, p0_all)),
        "num_mentioned": NQ,
        "min_match": 4,
        "degenerate_columns": degenerate_columns,
        "presence_channel_live": bool(np.std(sp_arr) > 0),
        "sp_std": float(sp_arr.std()),
        "note": ("fewer mentioned relations than min_match 4 make the spatial "
                 "channels identically zero; min_match is never lowered and no "
                 "description is invented (presence channels are unaffected)"
                 if degenerate_columns else "no relation channel is degenerate"),
    }
    meta = dict(
        protocol="CORR-D4P-SCAR-FORMAL-v1", domain=a.domain, split=a.split,
        queries=int(n_done), k=int(K), valid_lists=int(valid.sum()),
        valid_frac=float(valid.mean()), positives_total=int(n_pos),
        positives_per_list=float(labels.sum() / n_done),
        positives_per_valid=float(labels.sum() / max(1, valid.sum())),
        empty_lists=int((labels.sum(1) == 0).sum()),
        full_lists=int((labels.sum(1) == K).sum()),
        features=list(F.FEATURE_NAMES),
        norms=dict(pool_norm=K - 1, rank_norm=K - 1, tied_norm=K),
        label="same scene and ||pred_w - gt_pose_w|| <= 5 m",
        hint_text="Kitti360BaseDataset.create_hint_description(pose, cell)",
        parse_vocab=dict(colors=COLS, classes=LABS),
        relation_channel=degeneracy,
        cos_recomputed_from_coarse_enc=cos_check,
        sa_geom_sha256=hashlib.sha256(open(G.__file__, "rb").read()).hexdigest(),
        features_fix_sha256=hashlib.sha256(open(F.__file__, "rb").read()).hexdigest(),
        common_sha256=hashlib.sha256(open(C.__file__, "rb").read()).hexdigest(),
        dataset_manifest_sha256=man["_sha256"], adapter=adv,
        export_dir=a.export_dir,
        export_meta_sha256=S.sha256(meta_path),
        export_cache_sha256=export_meta.get("cache_sha256"),
        gate_binding=gate, protocol_freeze_sha256=a.protocol_sha,
        freeze_list_sha256=a.freeze_sha,
        feature_file_sha256=S.sha256(a.out),
        code_sha256={f: S.sha256(os.path.join(HERE, f))
                     for f in ("d4p_features.py", "scar_common.py")},
        test_opened=a.split == "test",
    )
    S.write_json(meta, a.out + ".meta.json")
    print(json.dumps({k: meta[k] for k in ("domain", "split", "queries", "valid_lists",
                                          "positives_per_list", "relation_channel")},
                     indent=2), flush=True)
    print("saved", a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
