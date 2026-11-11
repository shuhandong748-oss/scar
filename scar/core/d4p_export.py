#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export corrected Des4Pos Top-25 candidates for one domain and split.

One forward path, one cache:
  coarse: core encode_cells / encode_queries, top-25 by descending cosine;
  fine:   for every candidate, objects[:pad_size] padded, FixedPoints(256),
          FineNetwork.predict_with_text with the query's own text encoding;
  decode: pred_w = bbox_w[0:2] + offset * cell_size.

Direct metrics are computed from that same forward (cell recall + the core's
localization_metrics).  No second forward is ever run to "match" numbers; the
verifier recomputes everything from the saved arrays instead.

TEST safety: the output directory and TEST_READ_ONCE.json must be armed by the
orchestrator *before* this script is called; a marker written any later would
be a protocol violation.  This script refuses to read TEST otherwise.
"""
import argparse
import json
import os
import sys
import time
from argparse import Namespace

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
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--protocol_sha", required=True)
    ap.add_argument("--freeze_sha", default=None)
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--cell_batch", type=int, default=8)
    ap.add_argument("--candidate_batch", type=int, default=10)
    ap.add_argument("--limit_queries", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    if a.k != 25:
        raise SystemExit("the frozen protocol fixes K=25")
    if a.split == "test" and not a.freeze_sha:
        raise SystemExit("TEST export needs --freeze_sha (freeze list must exist)")
    a.out_dir = os.path.abspath(a.out_dir)

    marker, mode = S.arm_test_dir(a.out_dir, a.protocol_sha, a.freeze_sha, a.split)
    d = S.domain_cfg(a.domain)
    gate = S.load_gate()
    backbones = S.verify_backbones(a.domain)
    args, cfg = S.inference_config(a.domain)
    cfg["num_stats_path"] = d["num_stats"]
    args.num_stats_path = d["num_stats"]
    args.device = a.device

    dom = S.import_adapter(a.domain)
    rec = dom.apply(d["base"], verbose=False)
    check = dom.check_core_manifest()
    if not check["PASS"]:
        raise SystemExit(f"core MANIFEST mismatch: {check['failures'][:5]}")

    sys.path.insert(0, d["vendor"])
    os.chdir(dom.CORE)
    sys.path.insert(0, dom.CORE)

    import torch
    import torch_geometric.transforms as T
    from dataloading.kitti360pose.cells import Kitti360CoarseDatasetMulti
    from dataloading.kitti360pose.poses import batch_object_points
    from datapreparation.kitti360pose.imports import Object3d
    from datapreparation.kitti360pose.utils import COLOR_NAMES
    from evaluation.decode import cell_recall, localization_metrics
    from evaluation.pipeline import encode_cells, encode_queries
    from models.coarse import CoarseNetwork
    from models.fine import FineNetwork
    from models.num_stats import load_num_stats
    from models.ste import FrozenT5

    stage = S.stage_records(a.domain)
    known_classes = list(stage["coarse"]["init"]["known_classes"])
    num_stats = load_num_stats(d["num_stats"])
    fp = S.num_stats_fingerprint(num_stats)
    for s in ("coarse", "fine"):
        got = stage[s]["result"].get("num_stats_fingerprint")
        if got is not None and got != fp:
            raise SystemExit(f"{s} checkpoint num_stats fingerprint {got} != {fp}")

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")

    t5 = FrozenT5(args.t5_path, device, max_length=args.max_text_len)
    from training.utils import load_prior
    prior = load_prior(args.ste_stage1_path, t5.model.config.d_model,
                       args.clip_dim, num_layers=args.prior_layers,
                       num_heads=args.prior_heads)
    coarse = CoarseNetwork(known_classes, COLOR_NAMES, args, t5, prior).to(device)
    S.load_light(coarse, stage["coarse"]["ckpt"])
    fine = FineNetwork(known_classes, COLOR_NAMES, args, t5, prior).to(device)
    S.load_light(fine, stage["fine"]["ckpt"])
    coarse.eval()
    fine.eval()

    splits, man = dom.split_scene_names(d["base"])
    scenes = splits[a.split]
    transform = T.FixedPoints(args.pointnet_numpoints)
    ds = Kitti360CoarseDatasetMulti(d["base"], list(scenes), transform,
                                    shuffle_hints=False, flip_poses=False)
    poses = ds.all_poses
    hints = [h for x in ds.datasets for h in x.hint_descriptions]
    if len(hints) != len(poses):
        raise SystemExit(f"hint/pose count mismatch: {len(hints)} != {len(poses)}")
    target_ids = [str(p.cell_id) for p in poses]
    n = min(len(poses), a.limit_queries) if a.limit_queries else len(poses)
    expected_n = d["expected"][a.split]
    if not a.limit_queries and n != expected_n:
        raise SystemExit(f"{a.domain} {a.split}: {n} queries != registered {expected_n}")
    poses, hints, target_ids = poses[:n], hints[:n], target_ids[:n]
    cells_list = ds.all_cells
    cells_dict = {str(c.id): c for c in cells_list}
    if a.k > len(cells_list):
        raise SystemExit(f"K={a.k} exceeds {len(cells_list)} cells")
    print(f"[{a.domain}/{a.split}] scenes={len(scenes)} cells={len(cells_list)} "
          f"queries={n} threads={torch.get_num_threads()}", flush=True)

    # --- coarse, one forward per cell/query -------------------------------
    torch.manual_seed(a.seed)          # deterministic anchor for FixedPoints
    t0 = time.time()
    enc_args = Namespace(**vars(args))
    enc_args.batch_size = a.cell_batch
    cell_enc, cell_ids = encode_cells(coarse, ds.get_cell_dataset(), enc_args, device)
    # queries carry no sampling; keep the same anchor
    torch.manual_seed(a.seed)
    text_enc = encode_queries(coarse, hints, args)
    t_encode = time.time() - t0
    scores_all = text_enc @ cell_enc.T
    top_idx = np.argsort(-scores_all, axis=1, kind="stable")[:, :a.k]
    peak = (torch.cuda.max_memory_allocated() / 2**30) if device.type == "cuda" else 0.0
    print(f"coarse enc {t_encode:.1f}s cells {cell_enc.shape} peak {peak:.2f} GiB",
          flush=True)

    K = a.k
    cell_ids_arr = np.empty((n, K), dtype="<U32")
    cell_scene_arr = np.empty((n, K), dtype="<U32")
    cell_ind = np.zeros((n, K), np.int32)
    scores = np.zeros((n, K), np.float32)
    offsets = np.zeros((n, K, 2), np.float32)
    pred_w = np.zeros((n, K, 2), np.float32)
    cell_bbox_min = np.zeros((n, K, 2), np.float32)
    cell_center = np.zeros((n, K, 2), np.float32)
    cell_size = np.zeros((n, K), np.float32)
    gt_cell = np.empty(n, dtype="<U32")
    gt_pose_w = np.zeros((n, 2), np.float32)
    scene = np.empty(n, dtype="<U32")
    hit_cell = {k: np.zeros(n, np.int8) for k in KS_CELL}
    hit_loc = {(k, t): np.zeros(n, np.int8) for k in KS_LOC for t in THRESHS}
    rows = []

    t1 = time.time()
    with torch.inference_mode():
        for qi in range(n):
            idx = top_idx[qi]
            cands = [cells_dict[str(cell_ids[j])] for j in idx]
            ids = [str(c.id) for c in cands]
            if len(set(ids)) != K:
                raise SystemExit(f"duplicate candidate at query {qi}")
            cell_ids_arr[qi] = np.asarray(ids, dtype="<U32")
            cell_scene_arr[qi] = np.asarray([str(c.scene_name) for c in cands], dtype="<U32")
            cell_ind[qi] = idx
            # cosine recomputed from the stored encodings (cache identity)
            scores[qi] = (cell_enc[idx] @ text_enc[qi]).astype(np.float32)
            for r, c in enumerate(cands):
                cell_bbox_min[qi, r] = np.asarray(c.bbox_w[0:2], float)
                cell_center[qi, r] = (np.asarray(c.bbox_w[0:2], float)
                                      + np.asarray(c.bbox_w[3:5], float)) / 2.0
                cell_size[qi, r] = float(c.cell_size)

            torch.manual_seed(a.seed + qi)   # per-query fine sampling anchor
            t_i = fine.encode_text([hints[qi]])
            off = np.zeros((K, 2), np.float32)
            for start in range(0, K, a.candidate_batch):
                chunk = cands[start:start + a.candidate_batch]
                objects, points = [], []
                for j, cell in enumerate(chunk):
                    # seed per candidate: FixedPoints(256) draws then depend only
                    # on (seed, query, candidate rank), never on the chunk layout
                    torch.manual_seed((a.seed * 1000003 + qi * 10007
                                       + (start + j)) % (2 ** 31 - 1))
                    objs = list(cell.objects)[:args.pad_size]
                    while len(objs) < args.pad_size:
                        objs.append(Object3d.create_padding())
                    objects.append(objs)
                    points.append(batch_object_points(objs, transform))
                pred = fine.predict_with_text(objects, points, t_i)
                off[start:start + len(chunk)] = pred.float().cpu().numpy()
            offsets[qi] = off
            pred_w[qi] = cell_bbox_min[qi] + off * cell_size[qi][:, None]

            pose = poses[qi]
            gt_cell[qi] = target_ids[qi]
            gt_pose_w[qi] = np.asarray(pose.pose_w[0:2], np.float32)
            q_scene = str(cells_dict[target_ids[qi]].scene_name)
            scene[qi] = q_scene

            rec_cell = cell_recall(ids, str(target_ids[qi]), KS_CELL)
            for k in KS_CELL:
                hit_cell[k][qi] = int(rec_cell[k])
            acc, _ = localization_metrics(pose, cands, off, top_k=KS_LOC,
                                          thresholds=THRESHS, query_scene=q_scene)
            for key, val in acc.items():
                hit_loc[key][qi] = int(val)
            rows.append(dict(q=qi, scene=q_scene, gt_cell=str(gt_cell[qi]), ids=ids,
                             scores=[float(x) for x in scores[qi]],
                             offsets=[[float(x) for x in o] for o in off],
                             pred_w=[[float(x) for x in p] for p in pred_w[qi]],
                             hit_cell={f"R{k}": int(hit_cell[k][qi]) for k in KS_CELL},
                             hit_loc={f"top{k}@{int(t)}m": int(hit_loc[(k, t)][qi])
                                      for (k, t) in hit_loc}))
            if (qi + 1) % 100 == 0:
                el = time.time() - t1
                print(f"fine {qi+1}/{n} {el:.0f}s eta {(n-qi-1)*el/(qi+1)/60:.1f} min "
                      f"peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB",
                      flush=True)
    t_fine = time.time() - t1

    query_ids = np.asarray([f"{scene[i]}:{i:04d}" for i in range(n)], dtype="<U32")
    cand_path = os.path.join(a.out_dir, "candidates.npz")
    np.savez_compressed(
        cand_path, cell_ids=cell_ids_arr, cell_scene=cell_scene_arr,
        cell_index=cell_ind, scores=scores, offsets=offsets, pred_w=pred_w,
        cell_bbox_min=cell_bbox_min, cell_center=cell_center, cell_size=cell_size,
        gt_cell=gt_cell, gt_pose_w=gt_pose_w, scene=scene, query_ids=query_ids)
    enc_path = os.path.join(a.out_dir, "coarse_enc.npz")
    np.savez_compressed(
        enc_path, tg=text_enc.astype(np.float32), cg=cell_enc.astype(np.float32),
        cell_ids=np.asarray([str(c.id) for c in cells_list], dtype="<U32"))

    metrics = {
        "cell_R": {f"@{k}": float(hit_cell[k].mean()) for k in KS_CELL},
        "localization": {f"Top{k}@{int(t)}m": float(hit_loc[(k, t)].mean())
                         for k in KS_LOC for t in THRESHS},
        "n_queries": n,
        "denominator": "all queries of the split; no filtering",
    }
    meta = {
        "implementation": "Des4Pos paper reimplementation, corrected command-11 backbone",
        "domain": a.domain, "split": a.split, "n_queries": n, "n_cells": len(cells_list),
        "k": K, "cell_batch": a.cell_batch, "candidate_batch": a.candidate_batch,
        "seed": a.seed, "device": str(device),
        "encode_seconds": round(t_encode, 1), "fine_seconds": round(t_fine, 1),
        "torch_peak_GiB": round(peak, 2),
        "rng_policy": {
            "cell_sampling": "torch.manual_seed(seed) immediately before encode_cells; "
                             "FixedPoints(256) draws follow the deterministic cell order",
            "query_text": "torch.manual_seed(seed) before encode_queries (no sampling)",
            "fine_sampling": "torch.manual_seed(seed + qi) before fine.encode_text; "
                             "before every candidate's batch_object_points the RNG is "
                             "re-seeded with f(seed, qi, candidate_rank), so neither "
                             "K nor candidate_batch can change a sampling draw",
            "dataset": "shuffle_hints=False, flip_poses=False",
            "note": "candidate_batch only groups forwards; per-candidate seeding makes "
                    "the Fine inputs invariant to the chunk layout",
        },
        "checkpoints": {
            "coarse": {"path": stage["coarse"]["ckpt"], "sha256": S.sha256(stage["coarse"]["ckpt"]),
                       "epoch": stage["coarse"]["result"]["best_epoch"]},
            "fine": {"path": stage["fine"]["ckpt"], "sha256": S.sha256(stage["fine"]["ckpt"]),
                     "epoch": stage["fine"]["result"]["best_epoch"]},
        },
        "backbone_binding": backbones,
        "num_stats": {"path": d["num_stats"], "sha256": S.sha256(d["num_stats"]),
                      "fingerprint": fp},
        "config": cfg,
        "config_source": os.path.join(d["run"], "coarse", "STAGE_FREEZE.json"),
        "manifest_sha256": man["_sha256"],
        "dataset_manifest_path": os.path.join(d["base"], "dataset_manifest.json"),
        "adapter": rec,
        "core_manifest_sha256": check["manifest_sha256"],
        "core_manifest_checked": check["checked"],
        "core_root": dom.CORE,
        "gate_binding": gate,
        "protocol_freeze_sha256": a.protocol_sha,
        "freeze_list_sha256": a.freeze_sha,
        "code_sha256": {f: S.sha256(os.path.join(HERE, f))
                        for f in ("d4p_export.py", "scar_common.py")},
        "vendor_sha256": {f: S.sha256(os.path.join(d["vendor"], f))
                          for f in ("common.py", "features_fix.py", "sa_geom.py")},
        "features": {"builder": "vendor/features_fix.py", "K": 25,
                     "pool_norm": 24, "rank_norm": 24, "tied_norm": 25,
                     "min_match": 4},
        "gt_used": "labels and metrics only; never an input to coarse or fine",
        "test_opened": a.split == "test",
        "test_marker_mode": mode,
        "complete": True,
    }
    S.write_json(metrics, os.path.join(a.out_dir, "metrics.json"))
    with open(os.path.join(a.out_dir, "per_query.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    meta["cache_sha256"] = {
        "candidates.npz": S.sha256(cand_path), "coarse_enc.npz": S.sha256(enc_path),
        "metrics.json": S.sha256(os.path.join(a.out_dir, "metrics.json")),
        "per_query.jsonl": S.sha256(os.path.join(a.out_dir, "per_query.jsonl")),
    }
    S.write_json(meta, os.path.join(a.out_dir, "EXPORT_META.json"))
    S.complete_test_marker(
        a.out_dir, marker, n_queries=n, coarse_sha256=meta["checkpoints"]["coarse"]["sha256"],
        fine_sha256=meta["checkpoints"]["fine"]["sha256"],
        cache_sha256=meta["cache_sha256"], metrics=metrics,
        n_scenes=len(set(str(x) for x in scene)))

    print("RESULTS cell " + " ".join(f"R{k}={100*metrics['cell_R'][f'@{k}']:.2f}"
                                     for k in KS_CELL), flush=True)
    print("RESULTS loc  " + " ".join(f"{k}={100*v:.2f}"
                                     for k, v in metrics["localization"].items()), flush=True)
    print(f"saved {a.out_dir} fine {t_fine:.0f}s peak {peak:.2f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
