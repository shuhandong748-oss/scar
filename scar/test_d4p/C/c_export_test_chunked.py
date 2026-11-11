#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chunk-resumable C TEST export using the frozen C forward path.

The model/data construction, the RNG policy and the candidate definition are
taken from the frozen d4p_export.py.  The only structural change is that the
Fine loop is split into independent query chunks:

  * each chunk has a SHA sidecar and binds protocol/freeze/checkpoints/seed;
  * a valid chunk is reused verbatim and never recomputed;
  * a malformed/incomplete chunk stops the run (it is quarantined only with an
    explicit --quarantine-incomplete, preserving its bytes and a reason log);
  * the final candidates.npz / coarse_enc.npz / metrics are assembled from the
    chunks, not re-inferred.

The VAL direct-vs-chunked gate is what establishes equivalence; TEST is read
only after the marker is armed.
"""
import argparse
import json
import os
import shutil
import sys
import time
from argparse import Namespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c_test_common as T  # noqa: E402


def _jsonl_write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _chunk_range(start, n, size):
    return start, min(n, start + size)


def _chunk_name(start, end):
    return f"q{start:06d}_{end:06d}"


def _sidecar_ok(chunk_dir, want, model_shas, protocol_sha, freeze_sha):
    p = os.path.join(chunk_dir, "CHUNK_META.json")
    if not os.path.isfile(p):
        return False, "missing CHUNK_META.json"
    try:
        m = T.read_json(p)
    except Exception as e:  # noqa: BLE001
        return False, f"unreadable CHUNK_META.json: {e}"
    z = os.path.join(chunk_dir, "chunk.npz")
    if not os.path.isfile(z):
        return False, "missing chunk.npz"
    if m.get("status") != "complete":
        return False, "status is not complete"
    checks = [
        (m.get("start"), want["start"]), (m.get("end"), want["end"]),
        (m.get("k"), want["k"]), (m.get("seed"), want["seed"]),
        (m.get("protocol_freeze_sha256"), protocol_sha),
        (m.get("freeze_list_sha256"), freeze_sha),
        (m.get("checkpoints"), model_shas),
    ]
    for got, exp in checks:
        if got != exp:
            return False, f"chunk binding mismatch: {got!r} != {exp!r}"
    if T.sha256(z) != m.get("chunk_sha256"):
        return False, "chunk.npz SHA mismatch"
    return True, "ok"


def _quarantine(chunk_dir, reason):
    qroot = os.path.join(os.path.dirname(chunk_dir), "..", "quarantine")
    qroot = os.path.abspath(qroot)
    os.makedirs(qroot, exist_ok=True)
    name = os.path.basename(chunk_dir) + "_" + time.strftime("%Y%m%dT%H%M%S")
    dst = os.path.join(qroot, name)
    os.replace(chunk_dir, dst)
    T.write_json(dict(source=os.path.abspath(chunk_dir), dest=dst,
                      reason=reason, when=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                      policy="bytes preserved; no automatic retry"),
                 os.path.join(qroot, "QUARANTINE_" + name + ".json"))
    log = os.path.join(qroot, "QUARANTINE_LOG.jsonl")
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(source=os.path.abspath(chunk_dir), dest=dst,
                                reason=reason)) + "\n")
    print(f"QUARANTINED {chunk_dir} -> {dst}: {reason}", flush=True)
    return dst


def _build_dataset(S, d, dom, args):
    import torch
    import torch_geometric.transforms as geom_t
    from dataloading.kitti360pose.cells import Kitti360CoarseDatasetMulti

    splits, man = dom.split_scene_names(d["base"])
    scenes = splits[args.split]
    transform = geom_t.FixedPoints(args.pointnet_numpoints)
    ds = Kitti360CoarseDatasetMulti(d["base"], list(scenes), transform,
                                    shuffle_hints=False, flip_poses=False)
    poses = ds.all_poses
    hints = [h for x in ds.datasets for h in x.hint_descriptions]
    if len(hints) != len(poses):
        raise SystemExit(f"hint/pose count mismatch: {len(hints)} != {len(poses)}")
    return ds, poses, hints, transform, man


def _load_enc(path, n, n_cells, cell_ids_expected, enc_meta):
    import numpy as np
    tg = np.load(os.path.join(path, "tg.npy"), allow_pickle=False)
    cg = np.load(os.path.join(path, "cg.npy"), allow_pickle=False)
    cids = [str(x) for x in np.load(os.path.join(path, "cell_ids.npy"),
                                    allow_pickle=True).tolist()]
    if tg.shape[0] != n or cg.shape[0] != n_cells:
        raise SystemExit("encoding cache shape mismatch")
    if cids != cell_ids_expected:
        raise SystemExit("encoding cache cell order mismatch")
    if enc_meta.get("tg_sha256") != T.sha256(os.path.join(path, "tg.npy")) \
            or enc_meta.get("cg_sha256") != T.sha256(os.path.join(path, "cg.npy")):
        raise SystemExit("encoding cache SHA mismatch")
    return tg, cg


def _save_enc(path, tg, cg, cell_ids, meta_base):
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "tg.npy"), np.asarray(tg))
    np.save(os.path.join(path, "cg.npy"), np.asarray(cg))
    np.save(os.path.join(path, "cell_ids.npy"),
            np.asarray(cell_ids, dtype="<U64"))
    meta = dict(meta_base)
    meta.update(tg_sha256=T.sha256(os.path.join(path, "tg.npy")),
                cg_sha256=T.sha256(os.path.join(path, "cg.npy")),
                cell_ids_sha256=T.sha256(os.path.join(path, "cell_ids.npy")),
                tg_shape=list(np.asarray(tg).shape),
                cg_shape=list(np.asarray(cg).shape),
                tg_dtype=str(np.asarray(tg).dtype),
                cg_dtype=str(np.asarray(cg).dtype),
                cell_ids=list(map(str, cell_ids)))
    T.write_json(meta, os.path.join(path, "ENC_META.json"))
    return meta


def _compute_enc(S, args, d, ds, hints, coarse, device, enc_dir, model_shas,
                 protocol_sha, freeze_sha):
    import torch
    from argparse import Namespace
    from evaluation.pipeline import encode_cells, encode_queries
    torch.manual_seed(args.seed)
    ea = Namespace(**vars(args))
    ea.batch_size = args.cell_batch
    cell_enc, cell_ids = encode_cells(coarse, ds.get_cell_dataset(), ea, device)
    torch.manual_seed(args.seed)
    text_enc = encode_queries(coarse, hints, args)
    base = dict(protocol_freeze_sha256=protocol_sha,
                freeze_list_sha256=freeze_sha, seed=args.seed,
                cell_batch=args.cell_batch, n_queries=len(hints),
                n_cells=len(cell_ids), checkpoints=model_shas,
                rng="torch.manual_seed(seed) before encode_cells; "
                    "torch.manual_seed(seed) before encode_queries")
    meta = _save_enc(enc_dir, np.asarray(text_enc), np.asarray(cell_enc),
                     [str(x) for x in cell_ids], base)
    return (np.asarray(text_enc), np.asarray(cell_enc),
            [str(x) for x in cell_ids], meta)


def _compute_chunk(S, args, qi0, qi1, cells_dict, poses, hints, target_ids,
                   top_idx, cell_enc, text_enc, cell_ids_all, cell_scene_all,
                   fine, transform, cells_batch_args, K, chunk_dir, model_shas,
                   protocol_sha, freeze_sha):
    import torch
    import torch_geometric.transforms as geom_t
    from dataloading.kitti360pose.poses import batch_object_points
    from datapreparation.kitti360pose.imports import Object3d
    from evaluation.decode import cell_recall, localization_metrics

    def new_arrays(n):
        return dict(
            cell_ids=np.empty((n, K), dtype="<U64"),
            cell_scene=np.empty((n, K), dtype="<U64"),
            cell_index=np.zeros((n, K), np.int32),
            scores=np.zeros((n, K), np.float32),
            offsets=np.zeros((n, K, 2), np.float32),
            pred_w=np.zeros((n, K, 2), np.float32),
            cell_bbox_min=np.zeros((n, K, 2), np.float32),
            cell_center=np.zeros((n, K, 2), np.float32),
            cell_size=np.zeros((n, K), np.float32),
            gt_cell=np.empty(n, dtype="<U64"),
            gt_pose_w=np.zeros((n, 2), np.float32),
            scene=np.empty(n, dtype="<U64"),
            query_ids=np.empty(n, dtype="<U64"),
            hit_cell=np.zeros((n, len(T.KS_CELL)), np.int8),
            hit_loc=np.zeros((n, len(T.KS_LOC) * len(T.THRESHS)), np.int8),
        )

    n = qi1 - qi0
    A = new_arrays(n)
    rows = []
    with torch.inference_mode():
        for qi in range(qi0, qi1):
            r = qi - qi0
            idx = top_idx[qi]
            cands = [cells_dict[str(cell_ids_all[j])] for j in idx]
            ids = [str(c.id) for c in cands]
            if len(set(ids)) != K:
                raise SystemExit(f"duplicate candidate at query {qi}")
            A["cell_ids"][r] = np.asarray(ids, dtype="<U64")
            A["cell_scene"][r] = np.asarray(
                [str(c.scene_name) for c in cands], dtype="<U64")
            A["cell_index"][r] = idx
            A["scores"][r] = (cell_enc[idx] @ text_enc[qi]).astype(np.float32)
            for j, c in enumerate(cands):
                A["cell_bbox_min"][r, j] = np.asarray(c.bbox_w[0:2], float)
                A["cell_center"][r, j] = (
                    np.asarray(c.bbox_w[0:2], float)
                    + np.asarray(c.bbox_w[3:5], float)) / 2.0
                A["cell_size"][r, j] = float(c.cell_size)

            torch.manual_seed(args.seed + qi)
            t_i = fine.encode_text([hints[qi]])
            off = np.zeros((K, 2), np.float32)
            for start in range(0, K, args.candidate_batch):
                chunk = cands[start:start + args.candidate_batch]
                objects, points = [], []
                for j, cell in enumerate(chunk):
                    torch.manual_seed((args.seed * 1000003 + qi * 10007
                                       + (start + j)) % (2 ** 31 - 1))
                    objs = list(cell.objects)[:args.pad_size]
                    while len(objs) < args.pad_size:
                        objs.append(Object3d.create_padding())
                    objects.append(objs)
                    points.append(batch_object_points(objs, transform))
                pred = fine.predict_with_text(objects, points, t_i)
                off[start:start + len(chunk)] = pred.float().cpu().numpy()
            A["offsets"][r] = off
            A["pred_w"][r] = (A["cell_bbox_min"][r]
                              + off * A["cell_size"][r][:, None])

            pose = poses[qi]
            A["gt_cell"][r] = np.asarray(str(target_ids[qi]), dtype="<U64")
            A["gt_pose_w"][r] = np.asarray(pose.pose_w[0:2], np.float32)
            q_scene = str(cells_dict[str(target_ids[qi])].scene_name)
            A["scene"][r] = np.asarray(q_scene, dtype="<U64")
            A["query_ids"][r] = np.asarray(f"{q_scene}:{qi:04d}",
                                           dtype="<U64")

            rec_cell = cell_recall(ids, str(target_ids[qi]), T.KS_CELL)
            for j, k in enumerate(T.KS_CELL):
                A["hit_cell"][r, j] = int(rec_cell[k])
            acc, _ = localization_metrics(
                pose, cands, off, top_k=T.KS_LOC, thresholds=T.THRESHS,
                query_scene=q_scene)
            for j, key in enumerate((kk, tt) for kk in T.KS_LOC
                                    for tt in T.THRESHS):
                # localization_metrics returns {(top_k, threshold): bool}
                A["hit_loc"][r, j] = int(acc[key])
            rows.append(dict(
                q=qi, query_id=str(A["query_ids"][r]), scene=q_scene,
                gt_cell=str(A["gt_cell"][r]), ids=ids,
                scores=[float(x) for x in A["scores"][r]],
                offsets=[[float(x) for x in o] for o in off],
                pred_w=[[float(x) for x in p] for p in A["pred_w"][r]],
                hit_cell={f"R{k}": int(A["hit_cell"][r, j])
                          for j, k in enumerate(T.KS_CELL)},
                hit_loc={f"top{k}@{int(t)}m": int(
                    A["hit_loc"][r, j]) for j, (k, t) in enumerate(
                        (kk, tt) for kk in T.KS_LOC for tt in T.THRESHS)},
            ))
            if (qi - qi0 + 1) % 50 == 0:
                print(f"chunk {os.path.basename(chunk_dir)} "
                      f"{qi - qi0 + 1}/{n}", flush=True)

    os.makedirs(chunk_dir, exist_ok=True)
    zpath = os.path.join(chunk_dir, "chunk.npz")
    np.savez_compressed(zpath, **A)
    rpath = os.path.join(chunk_dir, "rows.jsonl")
    _jsonl_write(rpath, rows)
    meta = dict(
        status="complete", start=qi0, end=qi1, n=n, k=K,
        seed=args.seed, candidate_batch=args.candidate_batch,
        cell_batch=args.cell_batch, protocol_freeze_sha256=protocol_sha,
        freeze_list_sha256=freeze_sha, checkpoints=model_shas,
        chunk_sha256=T.sha256(zpath), rows_sha256=T.sha256(rpath),
        created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        rng="per-query torch.manual_seed(seed+qi) before fine.encode_text; "
            "per-candidate f(seed,qi,rank) before batch_object_points",
    )
    T.write_json(meta, os.path.join(chunk_dir, "CHUNK_META.json"))
    return meta


def _assemble(out_dir, chunks, K, n, cell_ids_all, cell_enc, text_enc,
              meta_extra):
    z = None
    rows = []
    ys = {}
    for c in chunks:
        p = os.path.join(c, "chunk.npz")
        zz = np.load(p, allow_pickle=True)
        if z is None:
            for k in zz.files:
                ys[k] = [np.asarray(zz[k])]
            z = zz
        else:
            for k in zz.files:
                ys[k].append(np.asarray(zz[k]))
        with open(os.path.join(c, "rows.jsonl"), encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f)
    arr = {k: np.concatenate(v, axis=0) for k, v in ys.items()}
    for key in ("cell_ids", "cell_scene", "gt_cell", "scene", "query_ids"):
        a0 = np.asarray(arr[key])
        if a0.ndim == 2:
            arr[key] = np.asarray([[str(x) for x in row] for row in a0],
                                  dtype="<U64")
        else:
            arr[key] = np.asarray([str(x) for x in a0.reshape(-1)],
                                  dtype="<U64")
    if arr["cell_ids"].shape != (n, K):
        raise SystemExit("assembled candidate shape mismatch")
    if arr["scores"].shape != (n, K):
        raise SystemExit("assembled scores shape mismatch")
    cand_path = os.path.join(out_dir, "candidates.npz")
    np.savez_compressed(
        cand_path,
        cell_ids=arr["cell_ids"], cell_scene=arr["cell_scene"],
        cell_index=arr["cell_index"], scores=arr["scores"],
        offsets=arr["offsets"], pred_w=arr["pred_w"],
        cell_bbox_min=arr["cell_bbox_min"], cell_center=arr["cell_center"],
        cell_size=arr["cell_size"], gt_cell=arr["gt_cell"],
        gt_pose_w=arr["gt_pose_w"], scene=arr["scene"],
        query_ids=arr["query_ids"])
    enc_path = os.path.join(out_dir, "coarse_enc.npz")
    np.savez_compressed(
        enc_path, tg=np.asarray(text_enc, np.float32),
        cg=np.asarray(cell_enc, np.float32),
        cell_ids=np.asarray(cell_ids_all, dtype="<U64"))
    hit_cell = arr["hit_cell"]
    hit_loc = arr["hit_loc"]
    metrics = {
        "cell_R": {f"@{k}": float(hit_cell[:, j].mean())
                   for j, k in enumerate(T.KS_CELL)},
        "localization": {
            f"Top{k}@{int(t)}m": float(hit_loc[:, j].mean())
            for j, (k, t) in enumerate((kk, tt) for kk in T.KS_LOC
                                        for tt in T.THRESHS)},
        "n_queries": int(n),
        "denominator": "all queries of the split; no filtering; "
                       "invalid/empty lists count as failures",
    }
    T.write_json(metrics, os.path.join(out_dir, "metrics.json"))
    _jsonl_write(os.path.join(out_dir, "per_query.jsonl"), rows)
    cache = dict(
        **{"candidates.npz": T.sha256(cand_path),
           "coarse_enc.npz": T.sha256(enc_path),
           "metrics.json": T.sha256(os.path.join(out_dir, "metrics.json")),
           "per_query.jsonl": T.sha256(os.path.join(out_dir, "per_query.jsonl"))}
    )
    return arr, metrics, cache, rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=sorted(T.DOMAINS))
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--protocol_sha", required=True)
    ap.add_argument("--freeze_sha", default=None)
    ap.add_argument("--freeze_list", default=None)
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--cell_batch", type=int, default=8)
    ap.add_argument("--candidate_batch", type=int, default=10)
    ap.add_argument("--chunk_queries", type=int, default=64)
    ap.add_argument("--limit_queries", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--denominator_report", default=None)
    ap.add_argument("--quarantine-incomplete", action="store_true")
    a = ap.parse_args(argv)
    if a.k != 25:
        raise SystemExit("the frozen protocol fixes K=25")
    if a.split == "test" and not a.freeze_sha:
        raise SystemExit("TEST export needs --freeze_sha")
    a.out_dir = os.path.abspath(a.out_dir)
    S = T.frozen_common()
    info = T.domain_info(a.domain)
    d = S.domain_cfg(a.domain)

    freeze_path = a.freeze_list or os.path.join(
        "/root/autodl-tmp/d4p_trainval_20261104/C", a.domain, "FREEZE_LIST.json")
    freeze = None
    if a.freeze_sha or os.path.isfile(freeze_path):
        freeze, finfo = T.freeze_info(freeze_path, a.domain)
        if a.freeze_sha and finfo["freeze_sha256"] != a.freeze_sha:
            raise SystemExit("freeze sha mismatch")
        if a.freeze_sha and finfo["protocol_sha256"] != a.protocol_sha:
            raise SystemExit("protocol sha mismatch")
    if a.split == "test":
        a.freeze_sha = a.freeze_sha or (freeze["freeze_sha256"] if freeze else None)
        if not a.freeze_sha:
            raise SystemExit("TEST export needs a frozen freeze list")
        tm, _ = T.check_test_marker(a.out_dir, a.protocol_sha, a.freeze_sha,
                                    require_clean=False)
        if tm.get("status") == "completed" and not os.path.isfile(
                os.path.join(a.out_dir, "EXPORT_COMPLETE.json")):
            raise SystemExit(
                "TEST marker is already completed but EXPORT_COMPLETE.json is "
                "missing; refusing a second TEST forward/export")
    else:
        if os.path.isfile(os.path.join(a.out_dir, "TEST_READ_ONCE.json")):
            raise SystemExit("VAL output dir carries a TEST marker")
        os.makedirs(a.out_dir, exist_ok=True)
        if os.listdir(a.out_dir) and not os.path.isfile(
                os.path.join(a.out_dir, "EXPORT_COMPLETE.json")):
            raise SystemExit(f"VAL output dir is not empty: {os.listdir(a.out_dir)[:5]}")

    # Reuse a fully assembled export without touching the forward path.
    comp_path = os.path.join(a.out_dir, "EXPORT_COMPLETE.json")
    if os.path.isfile(comp_path):
        comp = T.read_json(comp_path)
        meta_path = os.path.join(a.out_dir, "EXPORT_META.json")
        meta = T.read_json(meta_path)
        ok = (comp.get("status") == "complete"
              and meta.get("domain") == a.domain and meta.get("split") == a.split
              and meta.get("k") == 25 and meta.get("n_queries") == comp.get("n_queries"))
        for f, h in comp.get("cache_sha256", {}).items():
            ok = ok and os.path.isfile(os.path.join(a.out_dir, f)) \
                and T.sha256(os.path.join(a.out_dir, f)) == h
        if not ok:
            raise SystemExit("EXPORT_COMPLETE exists but its cache is invalid; "
                             "refusing silent recomputation")
        if a.split == "test":
            m, _ = T.check_test_marker(a.out_dir, a.protocol_sha, a.freeze_sha,
                                       allow_completed=True)
            if m.get("status") == "completed" and \
                    (m.get("cache_sha256") or {}) != (comp.get("cache_sha256") or {}):
                raise SystemExit("completed TEST marker does not bind this "
                                 "EXPORT_COMPLETE cache")
            if m.get("status") == "armed":
                S.complete_test_marker(
                    a.out_dir, m, n_queries=comp["n_queries"],
                    cache_sha256=comp["cache_sha256"],
                    reuse_no_model_forward=True)
        print("EXPORT_REUSED", a.out_dir, flush=True)
        return 0

    gate = S.load_gate()
    backbones = S.verify_backbones(a.domain)
    args, cfg = S.inference_config(a.domain)
    cfg["num_stats_path"] = d["num_stats"]
    args.num_stats_path = d["num_stats"]
    args.device = a.device
    # This CLI namespace carries only the run controls (split/out_dir/k/seed/
    # batches/chunking).  The architecture keys come from the frozen
    # STAGE_FREEZE config, exactly as in the frozen direct export; without them
    # the dataset transform, encode_queries and the Fine padding would fall back
    # to no config at all.  Only missing keys are copied, so explicit CLI
    # controls always win.
    for _k, _v in vars(args).items():
        if not hasattr(a, _k):
            setattr(a, _k, _v)

    dom = S.import_adapter(a.domain)
    rec = dom.apply(d["base"], verbose=False)
    check = dom.check_core_manifest()
    if not check["PASS"]:
        raise SystemExit(f"core MANIFEST mismatch: {check['failures'][:5]}")
    sys.path.insert(0, d["vendor"])
    os.chdir(dom.CORE)
    sys.path.insert(0, dom.CORE)

    import torch
    import torch_geometric.transforms as geom_t
    from dataloading.kitti360pose.poses import batch_object_points  # noqa: F401
    from datapreparation.kitti360pose.imports import Object3d  # noqa: F401
    from datapreparation.kitti360pose.utils import COLOR_NAMES
    from models.coarse import CoarseNetwork
    from models.fine import FineNetwork
    from models.num_stats import load_num_stats
    from models.ste import FrozenT5

    ds, poses, hints, transform, man = _build_dataset(S, d, dom, a)
    n_all = len(poses)
    expected_n = None
    count_prov = None
    if a.split == "test" and a.domain == "s3dis":
        rep = os.path.abspath(a.denominator_report or
                              os.path.join(T.TEST_ROOT, "records",
                                           "S3DIS_DENOMINATOR_542_544.json"))
        count_prov = T.s3dis_count_provenance(
            rep, os.path.join(d["base"], "dataset_manifest.json"))
        expected_n = count_prov["effective"]
    elif not a.limit_queries:
        expected_n = d["expected"][a.split]
    if a.limit_queries:
        n = min(n_all, a.limit_queries)
    else:
        n = n_all
        if expected_n is not None and n != expected_n:
            raise SystemExit(f"{a.domain} {a.split}: {n} queries != expected "
                             f"{expected_n}")
    poses, hints = poses[:n], hints[:n]
    target_ids = [str(p.cell_id) for p in poses]
    cells_list = ds.all_cells
    cells_dict = {str(c.id): c for c in cells_list}
    cell_ids_all = [str(c.id) for c in cells_list]
    if a.k > len(cells_list):
        raise SystemExit(f"K={a.k} exceeds {len(cells_list)} cells")
    scene = [str(cells_dict[target_ids[i]].scene_name) for i in range(n)]
    n_clusters = len(set(scene))
    if a.split == "test" and a.domain == "s3dis":
        hist = {}
        for s in scene:
            hist[s] = hist.get(s, 0) + 1
        if len(hist) != 68 or sum(1 for v in hist.values() if v == 6) != 1 \
                or sum(1 for v in hist.values() if v == 8) != 67:
            raise SystemExit(f"S3DIS TEST room histogram unexpected: {hist}")
    if a.split == "test" and a.limit_queries == 0 \
            and n_clusters != info["expected_test_clusters"]:
        raise SystemExit(f"cluster count {n_clusters} != "
                         f"{info['expected_test_clusters']}")
    print(f"[{a.domain}/{a.split}] scenes={len(dom.split_scene_names(d['base'])[0][a.split])} "
          f"cells={len(cells_list)} queries={n} clusters={n_clusters} "
          f"chunks={a.chunk_queries}", flush=True)

    stage0 = S.stage_records(a.domain)
    model_shas = {
        "coarse": {"path": backbones["coarse"]["path"],
                   "sha256": backbones["coarse"]["sha256"],
                   "epoch": stage0["coarse"]["result"]["best_epoch"]},
        "fine": {"path": backbones["fine"]["path"],
                 "sha256": backbones["fine"]["sha256"],
                 "epoch": stage0["fine"]["result"]["best_epoch"]},
        "num_stats": {"path": d["num_stats"],
                      "sha256": T.sha256(d["num_stats"]),
                      "fingerprint": backbones["num_stats"]["fingerprint"]},
    }
    enc_dir = os.path.join(a.out_dir, "encode")
    enc_meta_path = os.path.join(enc_dir, "ENC_META.json")
    enc_ok = False
    if os.path.isfile(enc_meta_path):
        em = T.read_json(enc_meta_path)
        enc_ok = (em.get("protocol_freeze_sha256") == a.protocol_sha
                  and em.get("freeze_list_sha256") == a.freeze_sha
                  and em.get("seed") == a.seed
                  and em.get("n_queries") == n
                  and em.get("n_cells") == len(cells_list)
                  and em.get("checkpoints") == model_shas)
        if enc_ok:
            try:
                text_enc, cell_enc = _load_enc(
                    enc_dir, n, len(cells_list), cell_ids_all, em)
            except SystemExit:
                raise
        else:
            raise SystemExit("encoding cache exists but is not bound to this run; "
                             "refusing to overwrite it")
    else:
        if os.path.isdir(enc_dir) and os.listdir(enc_dir):
            raise SystemExit("encoding dir has files but no valid ENC_META; "
                             "refusing to overwrite")
        text_enc = cell_enc = None

    # Decide whether the forward model is needed.
    chunk_root = os.path.join(a.out_dir, "chunks")
    os.makedirs(chunk_root, exist_ok=True)
    ranges = [_chunk_range(s, n, a.chunk_queries)
              for s in range(0, n, a.chunk_queries)]
    chunk_dirs = [os.path.join(chunk_root, _chunk_name(s, e)) for s, e in ranges]
    need_chunks = []
    for (s, e), cd in zip(ranges, chunk_dirs):
        want = dict(start=s, end=e, k=a.k, seed=a.seed)
        ok, why = _sidecar_ok(cd, want, model_shas, a.protocol_sha, a.freeze_sha)
        if ok:
            continue
        if os.path.isdir(cd):
            if a.quarantine_incomplete:
                _quarantine(cd, why)
            else:
                raise SystemExit(
                    f"chunk {cd} is not usable ({why}); its bytes are preserved. "
                    f"Re-run with --quarantine-incomplete after inspecting it.")
        need_chunks.append((s, e, cd))
    if text_enc is None and not need_chunks:
        # A complete chunk set cannot be assembled without the coarse encoding.
        raise SystemExit("chunks are complete but encoding cache is missing; "
                         "no silent re-inference")

    coarse = fine = None
    if text_enc is None or need_chunks:
        stage = S.stage_records(a.domain)
        known_classes = list(stage["coarse"]["init"]["known_classes"])
        num_stats = load_num_stats(d["num_stats"])
        fp = S.num_stats_fingerprint(num_stats)
        for s in ("coarse", "fine"):
            got = stage[s]["result"].get("num_stats_fingerprint")
            if got is not None and got != fp:
                raise SystemExit(f"{s} num_stats fingerprint {got} != {fp}")
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)
        device = torch.device(a.device if torch.cuda.is_available() else "cpu")
        t5 = FrozenT5(args.t5_path, device, max_length=args.max_text_len)
        from training.utils import load_prior
        prior = load_prior(args.ste_stage1_path, t5.model.config.d_model,
                           args.clip_dim, num_layers=args.prior_layers,
                           num_heads=args.prior_heads)
        if text_enc is None:
            coarse = CoarseNetwork(known_classes, COLOR_NAMES, args, t5, prior).to(device)
            S.load_light(coarse, stage["coarse"]["ckpt"])
            coarse.eval()
            text_enc, cell_enc, cell_ids_enc, em = _compute_enc(
                S, a, d, ds, hints, coarse, device, enc_dir, model_shas,
                a.protocol_sha, a.freeze_sha)
            if cell_ids_enc != cell_ids_all:
                raise SystemExit("encoded cell order != dataset cell order")
        if need_chunks:
            fine = FineNetwork(known_classes, COLOR_NAMES, args, t5, prior).to(device)
            S.load_light(fine, stage["fine"]["ckpt"])
            fine.eval()
        else:
            fine = None
        if text_enc is None:
            raise SystemExit("internal: encoding was not produced")
    else:
        device = torch.device(a.device if torch.cuda.is_available() else "cpu")

    if text_enc is not None:
        scores_all = text_enc @ cell_enc.T
        top_idx = np.argsort(-scores_all, axis=1, kind="stable")[:, :a.k]
    else:
        top_idx = None
    if need_chunks:
        if top_idx is None or fine is None:
            raise SystemExit("internal: model/top_idx missing for chunk computation")
        for s, e, cd in need_chunks:
            _compute_chunk(
                S, a, s, e, cells_dict, poses, hints, target_ids, top_idx,
                cell_enc, text_enc, cell_ids_all, None, fine, transform,
                args, a.k, cd, model_shas, a.protocol_sha, a.freeze_sha)

    # Validate all chunks once more before assembly.
    chunks = []
    for (s, e), cd in zip(ranges, chunk_dirs):
        want = dict(start=s, end=e, k=a.k, seed=a.seed)
        ok, why = _sidecar_ok(cd, want, model_shas, a.protocol_sha, a.freeze_sha)
        if not ok:
            raise SystemExit(f"chunk {cd} invalid at assembly: {why}")
        chunks.append(cd)

    arr, metrics, cache, rows = _assemble(
        a.out_dir, chunks, a.k, n, cell_ids_all, cell_enc, text_enc, {})
    # Frozen-style metadata, with the additive provenance recorded.
    meta = {
        "implementation": "Des4Pos paper reimplementation, corrected command-11 backbone",
        "domain": a.domain, "split": a.split, "n_queries": n,
        "n_cells": len(cells_list), "k": a.k,
        "cell_batch": a.cell_batch, "candidate_batch": a.candidate_batch,
        "chunk_queries": a.chunk_queries, "n_chunks": len(chunks),
        "seed": a.seed, "device": str(device),
        "rng_policy": {
            "cell_sampling": "torch.manual_seed(seed) immediately before encode_cells; "
                             "FixedPoints(256) draws follow the deterministic cell order",
            "query_text": "torch.manual_seed(seed) before encode_queries (no sampling)",
            "fine_sampling": "torch.manual_seed(seed + qi) before fine.encode_text; "
                             "before every candidate's batch_object_points the RNG is "
                             "re-seeded with f(seed, qi, candidate_rank), so neither "
                             "K, candidate_batch nor chunk layout can change a draw",
            "dataset": "shuffle_hints=False, flip_poses=False",
            "chunking": "query chunks are independent; valid chunk sidecars are reused",
        },
        "checkpoints": model_shas,
        "backbone_binding": backbones,
        "num_stats": {"path": d["num_stats"], "sha256": T.sha256(d["num_stats"]),
                      "fingerprint": backbones["num_stats"]["fingerprint"]},
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
        "freeze_list_path": freeze_path,
        "code_sha256": {
            f: T.sha256(os.path.join(HERE, f)) for f in (
                "c_export_test_chunked.py", "c_test_common.py",
                "c_arm_test_v2.py")
        },
        "frozen_code_sha256": {
            f: T.sha256(os.path.join(T.FROZEN_CODE, f)) for f in (
                "scar_common.py", "d4p_export.py", "d4p_features.py",
                "d4p_eval.py")
        },
        "vendor_sha256": {f: T.sha256(os.path.join(d["vendor"], f))
                          for f in ("common.py", "features_fix.py", "sa_geom.py")},
        "features": {"builder": "vendor/features_fix.py", "K": 25,
                     "pool_norm": 24, "rank_norm": 24, "tied_norm": 25,
                     "min_match": 4},
        "gt_used": "labels and metrics only; never an input to coarse or fine",
        "test_opened": a.split == "test",
        "test_marker_mode": ("armed-resumed" if a.split == "test" else None),
        "count_provenance": count_prov,
        "expected_queries": expected_n,
        "cluster_unit": info["cluster_unit"],
        "n_clusters": n_clusters,
        "denominator": info["denominator"],
        "chunk_sha256": {os.path.basename(c): T.sha256(
            os.path.join(c, "chunk.npz")) for c in chunks},
        "complete": True,
    }
    meta["cache_sha256"] = cache
    T.write_json(meta, os.path.join(a.out_dir, "EXPORT_META.json"))
    comp = dict(status="complete", domain=a.domain, split=a.split,
                n_queries=n, k=a.k, n_chunks=len(chunks),
                cache_sha256=cache,
                export_meta_sha256=T.sha256(os.path.join(a.out_dir,
                                                         "EXPORT_META.json")),
                test_marker=(T.sha256(os.path.join(a.out_dir,
                                                   "TEST_READ_ONCE.json"))
                             if a.split == "test" else None),
                count_provenance=count_prov,
                created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                code_sha256=meta["code_sha256"])
    T.write_json(comp, os.path.join(a.out_dir, "EXPORT_COMPLETE.json"))
    if a.split == "test":
        m, _ = T.check_test_marker(a.out_dir, a.protocol_sha, a.freeze_sha,
                                   allow_completed=True)
        S.complete_test_marker(
            a.out_dir, m, n_queries=n, coarse_sha256=meta["checkpoints"]["coarse"]["sha256"],
            fine_sha256=meta["checkpoints"]["fine"]["sha256"],
            cache_sha256=cache, metrics=metrics, n_scenes=n_clusters,
            export_meta_sha256=comp["export_meta_sha256"],
            count_provenance=count_prov, completed_by="c_export_test_chunked.py")
    print("RESULTS cell " + " ".join(
        f"R{k}={100*metrics['cell_R'][f'@{k}']:.2f}" for k in T.KS_CELL),
        flush=True)
    print("RESULTS loc  " + " ".join(
        f"{k}={100*v:.2f}" for k, v in metrics["localization"].items()), flush=True)
    print("saved", a.out_dir, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
