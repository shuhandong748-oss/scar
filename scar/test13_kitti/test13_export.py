# -*- coding: utf-8 -*-
"""TEST13: TEST Top-25 candidate + Fine export on the corrected backbone.

This is the FORMAL SECOND TEST READ of the corrected Des4Pos backbone
(Coarse epoch 8 / Fine epoch 24), explicitly approved by the operator and
labelled EXPLORATORY everywhere.  The first read happened in
`kitti_bb0_seed0_corev2b` through `evaluation/pipeline.py`, with a different
readout; the two are not merged and this round is never used to tune
hyper-parameters or re-select epochs.

Inference semantics are inherited unchanged from COMMAND 09 / 12:

  * one global encoding pass over the TEST gallery and TEST hint sequences;
  * scores = text_enc @ cell_enc.T; candidates = np.argsort(-scores)[:25];
  * per query: torch.manual_seed(0 + qi); Fine on the native 10; full RNG
    snapshot; Fine on candidates 11..25; restore the snapshot, so the extra 15
    never advance the sampling stream seen by later queries;
  * chunked, atomic output with a resume path that re-proves weights / code /
    data identity AND RNG continuity.

No backbone training, no head training, no re-selection, no VAL/TRAIN
re-export.  GT poses are used only for the exported `dists_to_gt_xy` column
and for later labels/metrics; they never enter candidates, Fine inputs or
seeds.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch_geometric.transforms as TF
from argparse import Namespace
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test13_common as C  # noqa: E402

sys.path.insert(0, C.CORE)
from dataloading.kitti360pose.cells import (  # noqa: E402
    Kitti360CoarseDataset,
    Kitti360CoarseDatasetMulti,
)
from dataloading.kitti360pose.utils import batch_object_points  # noqa: E402
from datapreparation.kitti360pose.imports import Object3d  # noqa: E402
from datapreparation.kitti360pose.utils import (  # noqa: E402
    COLOR_NAMES,
    SCENE_NAMES_TEST,
    SCENE_NAMES_TRAIN,
    SCENE_NAMES_VAL,
)
from evaluation.decode import decode_cell_offsets  # noqa: E402
from models.coarse import CoarseNetwork  # noqa: E402
from models.fine import FineNetwork  # noqa: E402
from models.num_stats import load_num_stats  # noqa: E402
from models.ste import FrozenT5  # noqa: E402
from training.utils import load_prior, sha256_file  # noqa: E402

CHUNK = 2000
PROBE_QUERIES = 24


# --------------------------------------------------------------- RNG payload
def rng_payload():
    ns = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (ns[0], np.array(ns[1], copy=True), int(ns[2]),
                  float(ns[3]), float(ns[4])),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": ([s.clone() for s in torch.cuda.get_rng_state_all()]
                       if torch.cuda.is_available() else []),
    }


def restore_rng(payload):
    random.setstate(payload["python"])
    ns = payload["numpy"]
    np.random.set_state((ns[0], np.array(ns[1], copy=True), int(ns[2]),
                         float(ns[3]), float(ns[4])))
    torch.set_rng_state(payload["torch_cpu"].cpu().clone())
    if torch.cuda.is_available():
        cuda = payload.get("torch_cuda") or []
        if len(cuda) != torch.cuda.device_count():
            raise RuntimeError(f"CUDA RNG state has {len(cuda)} devices, runtime "
                               f"has {torch.cuda.device_count()}")
        torch.cuda.set_rng_state_all([s.cpu().clone() for s in cuda])


def rng_to_arrays(payload, prefix):
    py = payload["python"]
    out = {
        prefix + "_py_ver": np.int64(py[0]),
        prefix + "_py_keys": np.array(py[1], dtype=np.int64),
        prefix + "_py_gauss_none": np.int64(1 if py[2] is None else 0),
        prefix + "_py_gauss": np.float64(0.0 if py[2] is None else py[2]),
    }
    ns = payload["numpy"]
    out.update({
        prefix + "_np_name": np.array(str(ns[0])),
        prefix + "_np_keys": np.array(ns[1], dtype=np.uint32),
        prefix + "_np_pos": np.int64(ns[2]),
        prefix + "_np_hg": np.int64(ns[3]),
        prefix + "_np_cg": np.float64(ns[4]),
        prefix + "_torch": payload["torch_cpu"].numpy().astype(np.uint8),
        prefix + "_n_cuda": np.int64(len(payload["torch_cuda"])),
    })
    for i, s in enumerate(payload["torch_cuda"]):
        out[f"{prefix}_cuda_{i}"] = s.numpy().astype(np.uint8)
    return out


def arrays_to_rng(arr, prefix):
    py = (int(arr[prefix + "_py_ver"]),
          tuple(int(x) for x in arr[prefix + "_py_keys"]),
          None if int(arr[prefix + "_py_gauss_none"]) else float(arr[prefix + "_py_gauss"]))
    ns = (str(arr[prefix + "_np_name"]),
          np.array(arr[prefix + "_np_keys"], dtype=np.uint32),
          int(arr[prefix + "_np_pos"]), int(arr[prefix + "_np_hg"]),
          float(arr[prefix + "_np_cg"]))
    cpu = torch.from_numpy(np.array(arr[prefix + "_torch"], dtype=np.uint8).copy())
    nc = int(arr[prefix + "_n_cuda"])
    cuda = [torch.from_numpy(np.array(arr[f"{prefix}_cuda_{i}"], dtype=np.uint8).copy())
            for i in range(nc)]
    return {"python": py, "numpy": ns, "torch_cpu": cpu, "torch_cuda": cuda}


# ------------------------------------------------------------------- loading
def build_args(cfg_path, device):
    cfg = C.read_json(cfg_path)
    args = Namespace(**cfg)
    args.device = str(device)
    args.checkpoint_root = "/tmp/des4pos_eval_ckpt_test13"
    args.folder_name = "test13_export"
    if not hasattr(args, "temperature"):
        args.temperature = 0.1
    return args, cfg


def build_dataset(cfg):
    transform = TF.FixedPoints(cfg["pointnet_numpoints"])
    ds = Kitti360CoarseDatasetMulti(
        cfg["base_path"], SCENE_NAMES_TEST, transform,
        shuffle_hints=False, flip_poses=False)
    poses = list(ds.all_poses)
    hints = [h for d in ds.datasets for h in d.hint_descriptions]
    cells = list(ds.all_cells)
    assert len(poses) == len(hints)
    return ds, transform, poses, hints, cells


def check_identity(ds, poses, hints, cells):
    """TEST identity gate.

    There is no pre-existing frozen count for the TEST split, so the gate is:
      * the dataset must hold exactly SCENE_NAMES_TEST, in that order;
      * no TRAIN or VAL scene tag may leak into TEST cells or poses;
      * every query target cell must belong to the TEST gallery;
      * on the first run the resulting identity is recorded under
        out/state/TEST_IDENTITY.json; any later run must reproduce it exactly.
    """
    fp = C.data_fingerprint(ds)
    errs = []
    full_scenes = [str(d.scene_name) for d in ds.datasets]
    dataset_tags = {str(d.scene_name): sorted({str(c.scene_name) for c in d.cells})
                    for d in ds.datasets}
    tags = {str(c.scene_name) for c in cells}
    pose_tags = {str(p.scene_name) for p in poses}
    if full_scenes != list(SCENE_NAMES_TEST):
        errs.append(f"dataset scenes {full_scenes} != {list(SCENE_NAMES_TEST)}")
    if any(len(v) != 1 for v in dataset_tags.values()):
        errs.append(f"dataset holds several cell scene tags: {dataset_tags}")
    tag_union = {t for v in dataset_tags.values() for t in v}
    if (tags | pose_tags) - tag_union:
        errs.append(f"scene tags outside the TEST datasets: "
                    f"{sorted((tags | pose_tags) - tag_union)}")
    train_tags = {s.split("_")[-2] for s in SCENE_NAMES_TRAIN}
    val_tags = {s.split("_")[-2] for s in SCENE_NAMES_VAL}
    if tags & (train_tags | val_tags):
        errs.append(f"TRAIN/VAL scene tags in TEST cells: "
                    f"{sorted(tags & (train_tags | val_tags))}")
    if pose_tags & (train_tags | val_tags):
        errs.append(f"TRAIN/VAL scene tags in TEST poses: "
                    f"{sorted(pose_tags & (train_tags | val_tags))}")
    if {str(p.cell_id) for p in poses} - {str(c.id) for c in cells}:
        errs.append("query target cell outside the TEST gallery")
    if len(hints) != len(poses):
        errs.append(f"hints {len(hints)} != poses {len(poses)}")
    if errs:
        raise SystemExit("TEST identity check failed: " + "; ".join(errs))

    tag_to_full = {t: full for full, ts in dataset_tags.items() for t in ts}
    per_scene = {full: {"n_queries": 0, "n_cells": len(dataset_tags[full])}
                 for full in full_scenes}
    for p in poses:
        per_scene[tag_to_full[str(p.scene_name)]]["n_queries"] += 1
    ident = {
        "n_queries": fp["n_queries"], "n_cells": fp["n_cells"],
        "query_ids_sha256": fp["query_ids_sha256"],
        "gallery_ids_sha256": fp["gallery_ids_sha256"],
        "dataset_full_names": full_scenes,
        "per_scene": per_scene,
    }
    rec_path = os.path.join(C.ROUND, "out", "state", "TEST_IDENTITY.json")
    if os.path.isfile(rec_path):
        old = C.read_json(rec_path)
        for key in ("n_queries", "n_cells", "query_ids_sha256",
                    "gallery_ids_sha256", "dataset_full_names"):
            if old.get(key) != ident[key]:
                errs.append(f"recorded TEST identity {key} changed: "
                            f"{old.get(key)!r} -> {ident[key]!r}")
        if errs:
            raise SystemExit("TEST identity drift: " + "; ".join(errs))
        ident["recorded_at"] = old.get("recorded_at")
        ident["reproduced"] = True
    else:
        C.write_json_atomic(rec_path, {**ident, "recorded_at": C.now(),
                                       "note": "recorded on the first TEST "
                                               "export run of this round"})
        ident["recorded_at"] = C.now()
        ident["reproduced"] = False

    scene = {
        "dataset_full_names": full_scenes,
        "dataset_to_tags": dataset_tags,
        "cell_tags": sorted(tags),
        "pose_tags": sorted(pose_tags),
        "train_scene_tags": sorted(train_tags),
        "val_scene_tags": sorted(val_tags),
        "identity_record": ident,
    }
    return fp, scene


def build_context(device, seed):
    t_start = time.time()
    coarse_args, cfg_coarse = build_args(C.COARSE_CFG, device)
    fine_args, cfg_fine = build_args(C.FINE_CFG, device)
    allowed = {"batch_size", "folder_name", "epochs", "learning_rate",
               "lr_scheduler", "resume", "params_total", "params_trainable",
               "num_stats_path", "n_train_cells"}
    diff = {k: [cfg_coarse.get(k), cfg_fine.get(k)]
            for k in sorted(set(cfg_coarse) & set(cfg_fine))
            if k not in allowed and cfg_coarse.get(k) != cfg_fine.get(k)}
    if diff:
        raise SystemExit(f"coarse/fine config shape disagreement: {diff}")

    for path, sha in ((C.STE_PRIOR, C.EXPECT["ste_prior_sha256"]),
                      (C.POINTNET, C.EXPECT["pointnet_sha256"]),
                      (C.NUM_STATS, C.EXPECT["num_stats_sha256"]),
                      (C.KNOWN_CLASSES, C.EXPECT["known_classes_sha256"])):
        got = sha256_file(path)
        if got != sha:
            raise SystemExit(f"frozen dependency changed: {path} {got} != {sha}")

    num_stats = load_num_stats(C.NUM_STATS)
    if (num_stats["mean"] != C.EXPECT["num_stats_mean"]
            or num_stats["std"] != C.EXPECT["num_stats_std"]):
        raise SystemExit("num_stats values differ from the frozen fingerprint")
    known = C.read_json(C.KNOWN_CLASSES)["known_classes"]

    torch.manual_seed(seed)
    np.random.seed(seed)
    t5 = FrozenT5(cfg_fine["t5_path"], device, max_length=cfg_fine["max_text_len"])
    prior = load_prior(C.STE_PRIOR, t5.model.config.d_model,
                       cfg_fine["clip_dim"], num_layers=cfg_fine["prior_layers"],
                       num_heads=cfg_fine["prior_heads"])
    coarse = CoarseNetwork(known, COLOR_NAMES, coarse_args, t5, prior).to(device)
    fine = FineNetwork(known, COLOR_NAMES, fine_args, t5, prior).to(device)
    audit = {
        "coarse": C.load_light(coarse, C.COARSE_CKPT, prior,
                               C.EXPECT["coarse_sha256"],
                               C.EXPECT["coarse_epoch"],
                               C.EXPECT["coarse_light_digest"], "coarse"),
        "fine": C.load_light(fine, C.FINE_CKPT, prior,
                             C.EXPECT["fine_sha256"],
                             C.EXPECT["fine_epoch"],
                             C.EXPECT["fine_light_digest"], "fine"),
        "num_stats": {"path": C.NUM_STATS, "mean": num_stats["mean"],
                      "std": num_stats["std"],
                      "source": "frozen num_stats.json (light payload carries none)"},
    }
    coarse.eval()
    fine.eval()
    for p in coarse.parameters():
        p.requires_grad_(False)
    for p in fine.parameters():
        p.requires_grad_(False)

    ds, transform, poses, hints, cells = build_dataset(cfg_fine)
    fp, scene = check_identity(ds, poses, hints, cells)
    cells_dict = {c.id: c for c in cells}
    cell_ids = np.array([str(c.id) for c in cells])
    target_ids = np.array([str(p.cell_id) for p in poses])

    t_enc0 = time.time()
    cell_enc, enc_cell_ids = encode_cells(coarse, ds.get_cell_dataset(),
                                          coarse_args, device)
    if list(enc_cell_ids) != list(cell_ids):
        raise SystemExit("cell encoding order differs from gallery order")
    text_enc = encode_queries(coarse, hints, coarse_args.batch_size)
    t_enc = time.time() - t_enc0
    scores = text_enc @ cell_enc.T
    top_idx = np.argsort(-scores, axis=1)[:, :C.TOP_K]
    post_encode_rng = rng_payload()
    hints_sha = __import__("hashlib").sha256(
        json.dumps(hints, ensure_ascii=False).encode("utf-8")).hexdigest()

    return {
        "device": device, "coarse": coarse, "fine": fine, "transform": transform,
        "coarse_args": coarse_args, "fine_args": fine_args,
        "poses": poses, "hints": hints, "cells": cells,
        "cells_dict": cells_dict, "cell_ids": cell_ids,
        "target_ids": target_ids, "cell_enc": cell_enc, "text_enc": text_enc,
        "scores": scores, "top_idx": top_idx, "post_encode_rng": post_encode_rng,
        "audit": audit, "fingerprint": fp, "scene": scene,
        "hints_sha256": hints_sha, "num_stats": num_stats,
        "t_enc": t_enc, "t_setup": time.time() - t_start,
    }


@torch.no_grad()
def encode_cells(coarse, cells_ds, args, device):
    loader = DataLoader(cells_ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=Kitti360CoarseDataset.collate_fn)
    enc, ids = [], []
    for batch in loader:
        e = coarse.encode_objects(batch["objects"], batch["object_points"])
        enc.append(e.float().cpu().numpy())
        ids.extend(str(x) for x in batch["cell_ids"])
    return np.concatenate(enc, 0), np.array(ids)


@torch.no_grad()
def encode_queries(coarse, hints, batch_size):
    enc = []
    for i in range(0, len(hints), batch_size):
        e = coarse.encode_text(hints[i:i + batch_size])
        enc.append(e.float().cpu().numpy())
    return np.concatenate(enc, 0)


@torch.no_grad()
def fine_offsets(fine, cands, t_i, pad_size, transform):
    objects, points = [], []
    for cell in cands:
        objs = list(cell.objects)[:pad_size]
        while len(objs) < pad_size:
            objs.append(Object3d.create_padding())
        objects.append(objs)
        points.append(batch_object_points(objs, transform))
    pred = fine.predict_with_text(objects, points, t_i)
    return pred.float().cpu().numpy()


def query_offsets(ctx, qi, n_cand, restore_extras=True):
    """Offsets for one query: native 10, then extras after an RNG snapshot."""
    pad = ctx["fine_args"].pad_size
    tfm = ctx["transform"]
    fine = ctx["fine"]
    pose_id = ctx["target_ids"][qi]
    idx = ctx["top_idx"][qi][:n_cand]
    cands = [ctx["cells_dict"][ctx["cell_ids"][i]] for i in idx]
    torch.manual_seed(C.SEED + qi)
    t_i = fine.encode_text([ctx["hints"][qi]])
    off_native = fine_offsets(fine, cands[:C.NATIVE_K], t_i, pad, tfm)
    if n_cand > C.NATIVE_K:
        snap = rng_payload()
        off_extra = fine_offsets(fine, cands[C.NATIVE_K:], t_i, pad, tfm)
        if restore_extras:
            restore_rng(snap)
        off = np.concatenate([off_native, off_extra], 0)
    else:
        off = off_native
    return off, cands


def preview_offsets(ctx, qi):
    """First 10 native offsets of query qi, computed without advancing RNG."""
    snap = rng_payload()
    off, _ = query_offsets(ctx, qi, C.NATIVE_K, restore_extras=True)
    restore_rng(snap)
    return off


# ------------------------------------------------------------------- writing
def save_npz_atomic(path, arrays):
    C.save_npz_atomic(path, arrays, compressed=False)


def chunk_path(out_root, k):
    return os.path.join(out_root, "chunks", f"chunk_{k:04d}.npz")


def queries_path(out_root, k):
    return os.path.join(out_root, "chunks", f"queries_{k:04d}.jsonl")


def run_probe(ctx, out_root, n_probe):
    """A/B/C RNG probe on fixed TRAIN queries + measured sample sizes."""
    restore_rng(ctx["post_encode_rng"])
    A = [query_offsets(ctx, qi, C.NATIVE_K)[0] for qi in range(n_probe)]
    restore_rng(ctx["post_encode_rng"])
    B = [query_offsets(ctx, qi, C.TOP_K)[0] for qi in range(n_probe)]
    restore_rng(ctx["post_encode_rng"])
    Cc = [query_offsets(ctx, qi, C.TOP_K, restore_extras=False)[0]
          for qi in range(n_probe)]
    restore_rng(ctx["post_encode_rng"])

    per_q, ok = [], True
    for qi in range(n_probe):
        a, b, c = A[qi], B[qi], Cc[qi]
        row = {
            "query_index": qi,
            "target_cell_id": str(ctx["target_ids"][qi]),
            "A_native10_vs_B_first10_equal": bool(np.array_equal(a, b[:10])),
            "A_native10_vs_C_first10_equal": bool(np.array_equal(a, c[:10])),
            "A_vs_C_max_abs_diff": float(np.max(np.abs(a - c[:10]))),
            "B_vs_C_first10_max_abs_diff": float(np.max(np.abs(b[:10] - c[:10]))),
            "B_vs_C_extras_max_abs_diff": float(np.max(np.abs(b[10:] - c[10:]))),
        }
        ok = ok and row["A_native10_vs_B_first10_equal"]
        per_q.append(row)

    idx = np.arange(n_probe, dtype=np.int32)
    cand = ctx["top_idx"][:n_probe].astype(np.int32)
    arrays = {
        "query_index": idx,
        "candidate_idx": cand,
        "candidate_score": np.take_along_axis(
            ctx["scores"][:n_probe], cand, axis=1).astype(np.float32),
        "offsets_norm": np.stack(B).astype(np.float32),
        "pred_w": np.stack([decode_cell_offsets(
            [ctx["cells_dict"][ctx["cell_ids"][i]] for i in cand[qi]],
            B[qi].astype(np.float64)) for qi in range(n_probe)]),
        "text_enc": ctx["text_enc"][:n_probe].astype(np.float32),
    }
    pos = np.stack([np.asarray(ctx["poses"][qi].pose_w, dtype=np.float64)
                    for qi in range(n_probe)])
    arrays["pose_w"] = pos
    arrays["dists_to_gt_xy"] = np.linalg.norm(
        arrays["pred_w"] - pos[:, None, :2], axis=2)

    sample_dir = os.path.join(out_root, "probe")
    os.makedirs(sample_dir, exist_ok=True)
    save_npz_atomic(os.path.join(sample_dir, "probe_native10.npz"), {
        "query_index": idx,
        "A_offsets": np.stack(A), "B_offsets": np.stack(B), "C_offsets": np.stack(Cc),
        "A_native10": np.stack(A), "B_native10": np.stack([b[:10] for b in B]),
    })
    save_npz_atomic(os.path.join(sample_dir, "sample_arrays.npz"), arrays)
    sample_bytes = os.path.getsize(os.path.join(sample_dir, "sample_arrays.npz"))

    # per-query jsonl sample size (same schema as the real chunk jsonl)
    rows = []
    for qi in range(n_probe):
        cands = [str(ctx["cell_ids"][i]) for i in cand[qi]]
        rows.append({
            "query_index": qi,
            "cell_id": str(ctx["poses"][qi].cell_id),
            "scene": str(ctx["cells_dict"][ctx["target_ids"][qi]].scene_name),
            "scene_full": str(ctx["dataset_tag_to_full"][str(ctx["cells_dict"][ctx["target_ids"][qi]].scene_name)]),
            "pose_w": [float(x) for x in pos[qi]],
            "hints": list(ctx["hints"][qi]),
            "target_cell_id": str(ctx["target_ids"][qi]),
            "candidates": cands,
            "scores": [float(x) for x in arrays["candidate_score"][qi]],
        })
    sample_jsonl = os.path.join(sample_dir, "sample_queries.jsonl")
    C.write_jsonl_atomic(sample_jsonl, rows)
    jsonl_bytes = os.path.getsize(sample_jsonl)

    payload = {
        "purpose": "prove the extra 15 candidates never advance later native queries",
        "modes": {"A": "native K=10 only",
                  "B": "K=25 with RNG snapshot after native 10 and restore (used)",
                  "C": "K=25 without restore (control)"},
        "n_probe_queries": n_probe,
        "probe_queries": list(range(n_probe)),
        "per_query": per_q,
        "A_equals_B_all_queries": bool(ok),
        "first_query_matches_in_all_modes": per_q[0]["A_native10_vs_C_first10_equal"],
        "divergence_when_restore_skipped": {
            "queries_with_A_vs_C_difference": [r["query_index"] for r in per_q
                                               if not r["A_native10_vs_C_first10_equal"]],
            "max_abs_diff_per_query": [r["A_vs_C_max_abs_diff"] for r in per_q],
        },
        "run_fingerprint_sha256": ctx["fp"]["fingerprint_sha256"],
        "verdict": "PASS" if ok else "FAIL",
        "sample_bytes": {"npz": sample_bytes, "jsonl": jsonl_bytes,
                         "n_queries": n_probe},
        "created_at": C.now(),
    }
    C.write_json_atomic(os.path.join(sample_dir, "probe_rng.json"), payload)
    payload["A_offsets"] = np.stack(A)
    print(json.dumps({"probe": payload["verdict"], "A_eq_B": ok,
                      "sample_npz_bytes": sample_bytes,
                      "sample_jsonl_bytes": jsonl_bytes}, indent=1))
    return payload, sample_bytes, jsonl_bytes


def project_budget(ctx, sample_npz_bytes, sample_jsonl_bytes, n_probe):
    """Projected peak of the whole round from measured per-query bytes."""
    nq = len(ctx["poses"])
    nc = len(ctx["cells"])
    per_q_npz = sample_npz_bytes / float(n_probe)
    per_q_jsonl = sample_jsonl_bytes / float(n_probe)
    cache_chunks = int(per_q_npz * nq)
    cache_jsonl = int(per_q_jsonl * nq)
    cell_enc = nc * int(ctx["cell_enc"].shape[1]) * 4 + nc * 32 * 4
    # cell_ids are stored as <U32 (4 B/char); the rest is float64/int32 arrays
    cells_npz = nc * (4 * 32 + 8 + 6 * 8 + 8 + 2 * 8 + 4) + 4096
    cells_jsonl = int(1135 * nc)          # measured on the COMMAND 09 VAL cache
    # NOTE (v1 fix): the v0 formula wrote `... + 1 << 20`, which parses as
    # `(... + 1) << 20` and inflated both feature files by 2**20.  Parenthesised.
    feats_test = (nq * 25 * 23 * 4 + nq * 25 + nq * 25 * 2 * 2
                  + nq * 25 * 4 + nq * 25 * 4 + nq * 4 + nq * 8 * 2
                  + nq * 25 * 5 * 4 + nq * 25 * 2 * 4 + nq * 25 * 2 * 8
                  + (1 << 20))
    heads = 0                        # no head is trained in this round
    misc = 48 << 20                  # logs, probe, eval jsonl, report copy
    atomic = max(feats_test, cache_chunks)  # largest single atomic file
    raw = (cache_chunks + cache_jsonl + cell_enc + cells_npz + cells_jsonl
           + feats_test + heads + misc + atomic)
    safety = 1.15                    # unmeasured metadata/compression spread
    projected = int(raw * safety)
    free = C.disk_report()["free"]
    required = 1 << 30
    payload = {
        "n_queries": nq, "n_cells": nc,
        "per_query_npz_bytes": round(per_q_npz, 1),
        "per_query_jsonl_bytes": round(per_q_jsonl, 1),
        "projected": {
            "cache_chunks_bytes": cache_chunks,
            "cache_queries_jsonl_bytes": cache_jsonl,
            "cell_encodings_bytes": cell_enc,
            "cells_npz_bytes": cells_npz,
            "cells_jsonl_bytes": cells_jsonl,
            "test_features_npz_bytes": feats_test,
            "heads_and_history_bytes": heads,
            "logs_reports_misc_bytes": misc,
            "atomic_write_temp_allowance_bytes": atomic,
            "raw_sum_bytes": raw,
            "safety_factor": safety,
            "peak_total_bytes": projected,
        },
        "free_bytes": free,
        "required_margin_bytes": required,
        "free_after_projected_bytes": free - projected,
        "free_after_projected_at_least_1GiB": (free - projected) >= required,
        "method": "per-query bytes measured on a fixed 24-query TEST sample "
                  "(npz + jsonl), extrapolated to the TEST query count; fixed arrays "
                  "from actual shape/dtype; one atomic temp copy of the largest "
                  "file; 1.15 safety factor; no per-candidate 33x33 support "
                  "fields are stored",
        "code_version": "test13 v1 (derived from train12_export.py; same gate)",
        "created_at": C.now(),
    }
    C.write_json_atomic(os.path.join(C.ROUND, "out", "budget", "DISK_BUDGET.json"),
                        payload)
    return payload


def fingerprint_payload(ctx):
    code_sha = {
        "test13_export.py": sha256_file(os.path.abspath(__file__)),
        "test13_common.py": sha256_file(os.path.join(HERE, "test13_common.py")),
    }
    core = {}
    for rel in ("evaluation/decode.py", "evaluation/pipeline.py",
                "models/coarse.py", "models/fine.py", "models/ste.py",
                "dataloading/kitti360pose/cells.py",
                "dataloading/kitti360pose/utils.py"):
        core[rel] = sha256_file(os.path.join(C.CORE, rel))
    payload = {
        "seed": C.SEED, "top_k": C.TOP_K, "native_k": C.NATIVE_K,
        "chunk": CHUNK,
        "coarse_sha256": ctx["audit"]["coarse"]["sha256"],
        "fine_sha256": ctx["audit"]["fine"]["sha256"],
        "coarse_light_digest": ctx["audit"]["coarse"]["light_state_digest"],
        "fine_light_digest": ctx["audit"]["fine"]["light_state_digest"],
        "prior_state_digest": ctx["audit"]["coarse"]["prior_state_digest"],
        "num_stats_mean": ctx["num_stats"]["mean"],
        "num_stats_std": ctx["num_stats"]["std"],
        "n_queries": ctx["fingerprint"]["n_queries"],
        "n_cells": ctx["fingerprint"]["n_cells"],
        "query_ids_sha256": ctx["fingerprint"]["query_ids_sha256"],
        "gallery_ids_sha256": ctx["fingerprint"]["gallery_ids_sha256"],
        "hints_sha256": ctx["hints_sha256"],
        "dataset_scene_full_names": ctx["scene"]["dataset_full_names"],
        "rng_policy": "torch.manual_seed(seed+qi) per query; snapshot after native "
                      "10, extras, restore; python/numpy/torch CPU+CUDA captured",
        "code_sha256": code_sha,
        "core_sha256": core,
    }
    payload["fingerprint_sha256"] = __import__("hashlib").sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return payload


def progress_path(out_root):
    return os.path.join(out_root, "state", "progress.json")


def load_progress(out_root):
    p = progress_path(out_root)
    if not os.path.isfile(p):
        return None
    return C.read_json(p)


def save_progress(out_root, payload):
    C.write_json_atomic(progress_path(out_root), payload)


def write_chunk(ctx, out_root, k, q0, q1, rng_start, t0):
    n = q1 - q0
    offs = np.zeros((n, C.TOP_K, 2), dtype=np.float32)
    pred_w = np.zeros((n, C.TOP_K, 2), dtype=np.float64)
    cand = ctx["top_idx"][q0:q1].astype(np.int32)
    scores = np.take_along_axis(ctx["scores"][q0:q1], cand, axis=1).astype(np.float32)
    rows = []
    for i, qi in enumerate(range(q0, q1)):
        off, cands = query_offsets(ctx, qi, C.TOP_K, restore_extras=True)
        offs[i] = off
        pred_w[i] = decode_cell_offsets(cands, off.astype(np.float64))
        rows.append({
            "query_index": qi,
            "cell_id": str(ctx["poses"][qi].cell_id),
            "scene": str(ctx["cells_dict"][ctx["target_ids"][qi]].scene_name),
            "scene_full": str(ctx["dataset_tag_to_full"][
                str(ctx["cells_dict"][ctx["target_ids"][qi]].scene_name)]),
            "pose_w": [float(x) for x in np.asarray(ctx["poses"][qi].pose_w)],
            "hints": list(ctx["hints"][qi]),
            "target_cell_id": str(ctx["target_ids"][qi]),
            "candidates": [str(c.id) for c in cands],
            "scores": [float(x) for x in scores[i]],
        })
    rng_end = rng_payload()
    pos = np.stack([np.asarray(ctx["poses"][qi].pose_w, dtype=np.float64)
                    for qi in range(q0, q1)])
    dists = np.linalg.norm(pred_w - pos[:, None, :2], axis=2)

    next_q = q1 if q1 < len(ctx["poses"]) else None
    preview = (preview_offsets(ctx, next_q) if next_q is not None else None)

    payload = {
        "q0": np.int64(q0), "q1": np.int64(q1),
        "query_index": np.arange(q0, q1, dtype=np.int32),
        "candidate_idx": cand,
        "candidate_score": scores,
        "offsets_norm": offs,
        "pred_w": pred_w,
        "dists_to_gt_xy": dists,
        "pose_w": pos,
        "text_enc": ctx["text_enc"][q0:q1].astype(np.float32),
        "fingerprint_sha256": np.array(ctx["fp"]["fingerprint_sha256"]),
        "next_query_index": np.int64(-1 if next_q is None else next_q),
        "rng_end_sha256": np.array(ctx["fp"]["fingerprint_sha256"]),
    }
    payload.update(rng_to_arrays(rng_start, "rng_start"))
    payload.update(rng_to_arrays(rng_end, "rng_end"))
    if preview is not None:
        payload["next_query_preview"] = preview.astype(np.float32)

    cp = chunk_path(out_root, k)
    qp = queries_path(out_root, k)
    save_npz_atomic(cp, payload)
    C.write_jsonl_atomic(qp, rows)
    rec = {
        "chunk": k, "q0": q0, "q1": q1, "n": n,
        "npz": os.path.basename(cp), "npz_sha256": sha256_file(cp),
        "npz_bytes": os.path.getsize(cp),
        "jsonl": os.path.basename(qp), "jsonl_sha256": sha256_file(qp),
        "jsonl_bytes": os.path.getsize(qp),
        "next_query_index": -1 if next_q is None else next_q,
        "seconds": round(time.time() - t0, 1),
        "rng_end_sha256": __import__("hashlib").sha256(
            json.dumps({k2: str(np.asarray(v).tobytes().hex()[:64])
                        for k2, v in rng_to_arrays(rng_end, "r").items()},
                       sort_keys=True).encode()).hexdigest(),
        "finished_at": C.now(),
    }
    return rec, rng_end


def export_full(ctx, out_root, probe_payload, sample_npz, sample_jsonl):
    nq = len(ctx["poses"])
    n_chunks = (nq + CHUNK - 1) // CHUNK
    os.makedirs(os.path.join(out_root, "chunks"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "state"), exist_ok=True)
    if "fp" not in ctx:
        ctx["fp"] = fingerprint_payload(ctx)
    budget = project_budget(ctx, sample_npz, sample_jsonl, PROBE_QUERIES)
    if not budget["free_after_projected_at_least_1GiB"]:
        raise SystemExit("disk budget gate fails: "
                         f"{budget['free_after_projected_bytes']} B free after projected peak")

    # gallery artefacts first (small, deterministic)
    cache = os.path.join(out_root, "cache")
    os.makedirs(cache, exist_ok=True)
    cells = ctx["cells"]
    cell_arrays = {
        "cell_ids": ctx["cell_ids"],
        "cell_scene": np.array([str(c.scene_name) for c in cells]),
        "cell_bbox_w": np.stack([np.asarray(c.bbox_w, dtype=np.float64) for c in cells]),
        "cell_size": np.array([float(c.cell_size) for c in cells], dtype=np.float64),
        "cell_center": np.stack([np.asarray(c.get_center()[0:2], dtype=np.float64)
                                 for c in cells]),
        "cell_n_objects": np.array([len(c.objects) for c in cells], dtype=np.int32),
    }
    save_npz_atomic(os.path.join(cache, "cells.npz"), cell_arrays)
    save_npz_atomic(os.path.join(cache, "cell_encodings.npz"), {
        "cell_enc": ctx["cell_enc"].astype(np.float32),
        "cell_ids": ctx["cell_ids"],
    })
    C.write_jsonl_atomic(os.path.join(cache, "cells.jsonl"), [
        {"cell_id": str(c.id), "scene": str(c.scene_name),
         "bbox_w": [float(x) for x in np.asarray(c.bbox_w, dtype=np.float64)],
         "cell_size": float(c.cell_size),
         "center": [float(x) for x in np.asarray(c.get_center()[0:2], dtype=np.float64)],
         "n_objects": len(c.objects),
         "objects": [{"id": int(o.id), "label": str(o.label),
                      "color": str(o.get_color_text()),
                      "n_points": int(len(o.xyz))} for o in c.objects]}
        for c in cells])

    prog = load_progress(out_root)
    if prog is None:
        prog = {"fingerprint_sha256": ctx["fp"]["fingerprint_sha256"],
                "n_queries": nq, "n_chunks": n_chunks, "chunk_size": CHUNK,
                "completed": [], "started_at": C.now(),
                "probe_verdict": probe_payload["verdict"],
                "budget_peak_bytes": budget["projected"]["peak_total_bytes"]}
        save_progress(out_root, prog)
    else:
        if prog.get("fingerprint_sha256") != ctx["fp"]["fingerprint_sha256"]:
            raise SystemExit("resume refused: run fingerprint differs "
                             f"{prog.get('fingerprint_sha256')} != "
                             f"{ctx['fp']['fingerprint_sha256']}")
        if prog.get("probe_verdict") != "PASS":
            raise SystemExit("resume refused: recorded probe verdict is not PASS")
        print(f"resume: {len(prog['completed'])}/{n_chunks} chunks already complete",
              flush=True)

    done = {int(c["chunk"]) for c in prog["completed"]}
    for rec in prog["completed"]:
        cp = os.path.join(out_root, "chunks", rec["npz"])
        if not os.path.isfile(cp) or sha256_file(cp) != rec["npz_sha256"]:
            raise SystemExit(f"resume refused: chunk {rec['chunk']} artefact mismatch")
        qp = os.path.join(out_root, "chunks", rec["jsonl"])
        if not os.path.isfile(qp) or sha256_file(qp) != rec["jsonl_sha256"]:
            raise SystemExit(f"resume refused: chunk {rec['chunk']} jsonl mismatch")

    if prog["completed"]:
        last = prog["completed"][-1]
        z = np.load(os.path.join(out_root, "chunks", last["npz"]), allow_pickle=False)
        rng = arrays_to_rng(z, "rng_end")
        nxt = int(z["next_query_index"])
        if nxt != last["q1"]:
            raise SystemExit("resume refused: next_query_index disagrees with q1")
        restore_rng(rng)
        got = preview_offsets(ctx, nxt)
        want = z["next_query_preview"].astype(np.float32)
        if not np.array_equal(got.astype(np.float32), want):
            raise SystemExit("resume refused: RNG continuity preview differs")
        print(f"resume: RNG continuity proven at query {nxt} "
              f"(max|diff|={float(np.max(np.abs(got - want)))})", flush=True)
    else:
        rng = ctx["post_encode_rng"]
        restore_rng(rng)

    for k in range(n_chunks):
        if k in done:
            continue
        q0 = k * CHUNK
        q1 = min(nq, q0 + CHUNK)
        t0 = time.time()
        rng_start = rng
        print(f"[chunk {k}] {q0}..{q1}", flush=True)
        rec, rng = write_chunk(ctx, out_root, k, q0, q1, rng_start, t0)
        prog["completed"].append(rec)
        prog["last_update"] = C.now()
        save_progress(out_root, prog)
        if k == 0 and probe_payload.get("A_offsets") is not None:
            z = np.load(chunk_path(out_root, 0), allow_pickle=False)
            n_p = probe_payload["n_probe_queries"]
            if not np.array_equal(z["offsets_norm"][:n_p, :10],
                                  probe_payload["A_offsets"][:, :10].astype(np.float32)):
                raise SystemExit("full export disagrees with the probe on the "
                                 "first native-10 offsets")
        print(f"[chunk {k}] done {rec['seconds']}s {rec['npz_bytes']} B", flush=True)

    # final metadata
    total_bytes = sum(c["npz_bytes"] + c["jsonl_bytes"] for c in prog["completed"])
    meta = {
        "split": "test",
        "test_access": True,
        "test_read_number": 2,
        "exploratory": True,
        "created_at": C.now(),
        "n_queries": nq, "n_cells": len(ctx["cells"]),
        "k": C.TOP_K, "native_k": C.NATIVE_K, "seed": C.SEED,
        "chunks": prog["completed"], "n_chunks": n_chunks, "chunk_size": CHUNK,
        "chunk_bytes_total": total_bytes,
        "fingerprint": ctx["fp"],
        "weights": ctx["audit"],
        "num_stats": ctx["num_stats"],
        "scene": ctx["scene"],
        "hints_sha256": ctx["hints_sha256"],
        "data_fingerprint": ctx["fingerprint"],
        "label_rule": "same scene AND ||pred_w - gt_pose_xy|| <= 5 m (built later; "
                      "GT never enters candidates, Fine inputs or seeds)",
        "rng_policy": {
            "global_seed": C.SEED,
            "init": "torch.manual_seed(0); np.random.seed(0) once before model build",
            "per_query": "torch.manual_seed(0 + qi) before Fine encode_text",
            "fixedpoints": "PyG FixedPoints(replace=True) uses the NumPy global RNG",
            "padding": "Object3d.create_padding() draws np.random.rand(8,3) per slot",
            "extras": "full RNG snapshot after the native 10; extras computed; restored",
            "resume": "rng_end stored per chunk; next-query preview recomputed on resume",
        },
        "budget": budget,
        "timing": {"setup_and_encode_seconds": round(ctx["t_setup"], 1),
                   "encode_seconds": round(ctx["t_enc"], 1)},
        "env": {"hostname": __import__("socket").gethostname(),
                "python": sys.version.split()[0], "numpy": np.__version__,
                "torch": torch.__version__,
                "torch_geometric": __import__("torch_geometric").__version__,
                "transformers": __import__("transformers").__version__,
                "device": str(ctx["device"]),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "disk": C.disk_report(),
        "note_test_read": "formal second TEST read; exploratory; the first read "
                          "(core CLI, runs/kitti_bb0_seed0_corev2b) used a "
                          "different readout and is not comparable",
    }
    C.write_json_atomic(os.path.join(cache, "test_export_meta.json"), meta)
    C.write_json_atomic(os.path.join(out_root, "state", "progress.json"), prog)
    print(json.dumps({"n_chunks": n_chunks, "chunk_bytes_total": total_bytes,
                      "disk": C.disk_report()}, indent=1))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dryload", "probe", "full"], required=True)
    ap.add_argument("--out-root", default=os.path.join(C.ROUND, "out"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--probe-queries", type=int, default=PROBE_QUERIES)
    args = ap.parse_args()

    if args.mode in ("probe", "full"):
        C.write_test_read_marker(args.mode, "started",
                                 {"argv": sys.argv, "device": args.device})

    device = torch.device(args.device if args.device == "cuda"
                          and torch.cuda.is_available() else "cpu")
    if args.mode == "dryload":
        coarse_args, cfg_coarse = build_args(C.COARSE_CFG, device)
        fine_args, cfg_fine = build_args(C.FINE_CFG, device)
        num_stats = load_num_stats(C.NUM_STATS)
        if num_stats["mean"] != C.EXPECT["num_stats_mean"]:
            raise SystemExit("num_stats mismatch")
        torch.manual_seed(C.SEED)
        np.random.seed(C.SEED)
        t5 = FrozenT5(cfg_fine["t5_path"], device, max_length=cfg_fine["max_text_len"])
        prior = load_prior(C.STE_PRIOR, t5.model.config.d_model, cfg_fine["clip_dim"],
                           num_layers=cfg_fine["prior_layers"],
                           num_heads=cfg_fine["prior_heads"])
        known = C.read_json(C.KNOWN_CLASSES)["known_classes"]
        coarse = CoarseNetwork(known, COLOR_NAMES, coarse_args, t5, prior).to(device)
        fine = FineNetwork(known, COLOR_NAMES, fine_args, t5, prior).to(device)
        audit = {
            "coarse": C.load_light(coarse, C.COARSE_CKPT, prior,
                                   C.EXPECT["coarse_sha256"],
                                   C.EXPECT["coarse_epoch"],
                                   C.EXPECT["coarse_light_digest"], "coarse"),
            "fine": C.load_light(fine, C.FINE_CKPT, prior,
                                 C.EXPECT["fine_sha256"],
                                 C.EXPECT["fine_epoch"],
                                 C.EXPECT["fine_light_digest"], "fine"),
        }
        out_dir = os.path.join(args.out_root, "dryload")
        os.makedirs(out_dir, exist_ok=True)
        C.write_json_atomic(os.path.join(out_dir, "dryload_audit.json"), {
            "created_at": C.now(), "device": str(device), "audit": audit,
            "num_stats": {"mean": num_stats["mean"], "std": num_stats["std"],
                          "path": C.NUM_STATS},
        })
        print(json.dumps({"dryload": "OK", "device": str(device),
                          "coarse_missing": len(audit["coarse"]["missing_keys"]),
                          "fine_missing": len(audit["fine"]["missing_keys"])}, indent=1))
        return 0

    ctx = build_context(device, C.SEED)
    print(json.dumps({"stage": "context_ready",
                      "n_queries": len(ctx["poses"]), "n_cells": len(ctx["cells"]),
                      "encode_seconds": round(ctx["t_enc"], 1),
                      "setup_seconds": round(ctx["t_setup"], 1)}, indent=1))
    tag_to_full = {}
    for full, tags in ctx["scene"]["dataset_to_tags"].items():
        for t in tags:
            tag_to_full[t] = full
    ctx["dataset_tag_to_full"] = tag_to_full
    ctx["fp"] = fingerprint_payload(ctx)

    meta_path = os.path.join(args.out_root, "probe", "probe_rng.json")
    probe_payload = None
    sample_npz = sample_jsonl = None
    if os.path.isfile(meta_path):
        prior_probe = C.read_json(meta_path)
        if prior_probe.get("verdict") != "PASS":
            raise SystemExit("existing probe verdict is not PASS")
        if prior_probe.get("run_fingerprint_sha256") != ctx["fp"]["fingerprint_sha256"]:
            fp12 = str(prior_probe.get("run_fingerprint_sha256"))[:12]
            sup = os.path.join(args.out_root, "probe_superseded_" + fp12)
            if os.path.exists(sup):
                raise SystemExit(f"superseded probe already archived at {sup}")
            os.replace(os.path.join(args.out_root, "probe"), sup)
            print(f"existing probe fingerprint {fp12} differs from this run; "
                  f"archived to {sup} and re-running the probe", flush=True)
        else:
            sample_npz = os.path.getsize(os.path.join(args.out_root, "probe",
                                                      "sample_arrays.npz"))
            sample_jsonl = os.path.getsize(os.path.join(args.out_root, "probe",
                                                        "sample_queries.jsonl"))
            probe_payload = prior_probe
            z = np.load(os.path.join(args.out_root, "probe", "probe_native10.npz"),
                        allow_pickle=False)
            probe_payload["A_offsets"] = z["A_offsets"]
            if args.mode == "probe":
                print("probe already present and PASS; nothing to do")
                return 0
    if probe_payload is None:
        if args.mode == "full":
            print("no matching probe; running the probe first in this process",
                  flush=True)
        probe_payload, sample_npz, sample_jsonl = run_probe(
            ctx, args.out_root, args.probe_queries)
        if probe_payload["verdict"] != "PASS":
            return 3
        if args.mode == "probe":
            project_budget(ctx, sample_npz, sample_jsonl, args.probe_queries)
            return 0
    restore_rng(ctx["post_encode_rng"])
    meta = export_full(ctx, args.out_root, probe_payload, sample_npz, sample_jsonl)
    C.write_test_read_marker("full", "completed", {
        "n_queries": meta["n_queries"], "n_cells": meta["n_cells"],
        "run_fingerprint_sha256": meta["fingerprint"]["fingerprint_sha256"],
        "export_meta_sha256": C.sha256_file(os.path.join(
            args.out_root, "cache", "test_export_meta.json")),
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
