#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chunked, resumable A-domain TEST export (also usable on VAL for the gate).

One forward path, one cache.  The arithmetic mirrors the frozen
``export_candidates_coordfix_v1.py`` exactly:
  * storage order = ``np.argsort(-scores, axis=1)[:, :25]`` (default sort);
  * Fine offsets are cell-normalised; ``pred_w = bbox_min + off * cell_size``;
  * per query ``torch.manual_seed(seed + qi)`` before ``fine.encode_text``;
  * candidate chunks of 10; ``torch.no_grad``.
Chunking is by query, which is RNG-safe because every query reseeds.  Chunks
are written atomically with a sidecar SHA; a valid chunk is reused on resume and
a corrupt chunk stops the run without being overwritten.

TEST safety: the marker directory and TEST_READ_ONCE.json must already be armed.
The script verifies the marker before touching the dataset, records a query
identity pre-check, and never retrains, reselects or touches s1/s2.
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
import a_test_common as S  # noqa: E402

EXPECTED_EPOCHS = {
    "pandaset": {"coarse": 8, "fine": 35},
    "nuscenesfull": {"coarse": 17, "fine": 30},
}
CHUNK_DIR = "chunks"
ENCODE_DIR = "encode"
COMPLETE = "EXPORT_COMPLETE.json"
KNOWN_TOP = {COMPLETE, CHUNK_DIR, ENCODE_DIR, "identity_precheck.json",
             "quarantine", "candidates.npz", "coarse_enc.npz",
             "export_meta.json", "RECOMPUTE_LOG.jsonl"}


def parse_cli(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True, choices=sorted(S.DOMAINS))
    p.add_argument("--split", required=True, choices=["test", "val"])
    p.add_argument("--freeze_list", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--coarse_ckpt", required=True)
    p.add_argument("--fine_ckpt", required=True)
    p.add_argument("--known_classes", required=True)
    p.add_argument("--ste_stage1_path", required=True)
    p.add_argument("--frozen_export", default=os.path.join(
        os.path.dirname(S.A_TRAINVAL), "A_round", "code",
        "export_candidates_coordfix_v1.py"),
        help="frozen direct-path exporter; its load_light/bind_best are reused")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--marker", default=None,
                   help="TEST_READ_ONCE.json; required for split=test")
    p.add_argument("--chunk_size", type=int, default=256)
    p.add_argument("--limit_queries", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=25)
    p.add_argument("--candidate_batch", type=int, default=10)
    p.add_argument("--replace_chunk", type=int, default=None,
                   help="explicit chunk start to recompute; needs --reason")
    p.add_argument("--reason", default=None)
    p.add_argument("--no_resume", action="store_true")
    return p.parse_args(argv)


def atomic_savez(path, **arrays):
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def known_entries(out_dir):
    if not os.path.isdir(out_dir):
        return set()
    return set(os.listdir(out_dir)) - KNOWN_TOP


def load_encodings(out_dir, n_queries):
    meta_path = os.path.join(out_dir, ENCODE_DIR, "encode_meta.json")
    cg_path = os.path.join(out_dir, ENCODE_DIR, "cell_enc.npy")
    tg_path = os.path.join(out_dir, ENCODE_DIR, "text_enc.npy")
    if not (os.path.isfile(meta_path) and os.path.isfile(cg_path)
            and os.path.isfile(tg_path)):
        return None
    m = S.read_json(meta_path)
    if S.sha256(cg_path) != m.get("cell_enc_sha256"):
        raise SystemExit("cached cell encoding SHA mismatch; refusing to reuse")
    if S.sha256(tg_path) != m.get("text_enc_sha256"):
        raise SystemExit("cached text encoding SHA mismatch; refusing to reuse")
    tg = np.load(tg_path)
    cg = np.load(cg_path)
    if tg.shape[0] != n_queries or m.get("n_queries") != n_queries:
        raise SystemExit("cached text encoding query count mismatch")
    return dict(meta=m, tg=tg, cg=cg,
                cell_ids=np.asarray(m["cell_ids"], dtype="<U32"))


def save_encodings(out_dir, tg, cg, cell_ids, cell_scene, args, cli, binds):
    d = os.path.join(out_dir, ENCODE_DIR)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "text_enc.npy"), tg.astype(np.float32))
    np.save(os.path.join(d, "cell_enc.npy"), cg.astype(np.float32))
    m = dict(
        n_cells=int(len(cell_ids)), n_queries=int(len(tg)),
        cell_ids=[str(x) for x in cell_ids],
        cell_scene=[str(x) for x in cell_scene],
        seed=int(cli.seed), batch_size=int(args.batch_size),
        cell_enc_sha256=S.sha256(os.path.join(d, "cell_enc.npy")),
        text_enc_sha256=S.sha256(os.path.join(d, "text_enc.npy")),
        rng="torch.manual_seed(seed) before model construction; cell FixedPoints "
            "draws follow the deterministic cell order; text encoding carries no "
            "sampling and is run in one pass with the frozen batch size",
        checkpoints={k: dict(path=v["path"], sha256=v["sha256"])
                     for k, v in binds.items()},
        code_sha256=dict(export=S.sha256(os.path.abspath(__file__)),
                         common=S.sha256(os.path.join(HERE, "a_test_common.py"))),
        created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    S.write_json(m, os.path.join(d, "encode_meta.json"))
    return m


def sidecar_path(chunk_path):
    return chunk_path[:-4] + ".json"


def validate_chunk(chunk_path, start, end, expected_n, bind_sha):
    sc = sidecar_path(chunk_path)
    if not (os.path.isfile(chunk_path) and os.path.isfile(sc)):
        return None, "missing chunk or sidecar"
    m = S.read_json(sc)
    if m.get("status") != "complete":
        return None, f"status={m.get('status')}"
    if m.get("start") != start or m.get("end") != end:
        return None, "range mismatch"
    if m.get("n_queries") != expected_n:
        return None, "n_queries mismatch"
    if m.get("sha256") != S.sha256(chunk_path):
        return None, "file SHA mismatch"
    for k, v in bind_sha.items():
        if m.get(k) != v:
            return None, f"{k} mismatch"
    z = np.load(chunk_path, allow_pickle=True)
    if z["query_index"].tolist() != list(range(start, end)):
        return None, "query_index mismatch"
    return m, None


def quarantine(out_dir, start, reason):
    if not reason:
        raise SystemExit("--replace_chunk requires --reason")
    d = os.path.join(out_dir, "quarantine",
                     time.strftime("%Y%m%dT%H%M%S") + f"__q{start:06d}")
    os.makedirs(d, exist_ok=False)
    moved = []
    for base in sorted(os.listdir(os.path.join(out_dir, CHUNK_DIR))) \
            if os.path.isdir(os.path.join(out_dir, CHUNK_DIR)) else []:
        if base.startswith(f"q{start:06d}_"):
            src = os.path.join(out_dir, CHUNK_DIR, base)
            dst = os.path.join(d, base)
            os.replace(src, dst)
            moved.append(dst)
    rec = dict(start=start, reason=reason, moved=moved,
               at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    S.write_json(rec, os.path.join(d, "QUARANTINE.json"))
    with open(os.path.join(out_dir, "RECOMPUTE_LOG.jsonl"), "a",
              encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return d


def main(argv=None):
    cli = parse_cli(argv)
    if cli.k != 25:
        raise SystemExit("the frozen protocol fixes K=25")
    if cli.chunk_size <= 0:
        raise SystemExit("chunk_size must be positive")
    out_dir = os.path.abspath(cli.out_dir)
    marker = None
    if cli.split == "test":
        if not cli.marker:
            raise SystemExit("split=test requires --marker")
        freeze_peek = S.read_json(cli.freeze_list)
        marker = S.require_marker(
            cli.marker, cli.domain, "armed",
            freeze_sha=freeze_peek.get("freeze_sha256"),
            proc_sha=freeze_peek.get("protocol", {}).get("protocol_sha256"))
        if marker.get("purpose") is None:
            raise SystemExit("marker lacks the purpose statement")

    freeze, fchk = S.verify_freeze(cli.freeze_list, cli.domain,
                                   EXPECTED_EPOCHS[cli.domain])
    binds = dict(
        coarse=dict(path=cli.coarse_ckpt, sha256=S.sha256(cli.coarse_ckpt)),
        fine=dict(path=cli.fine_ckpt, sha256=S.sha256(cli.fine_ckpt)),
        ste=dict(path=cli.ste_stage1_path,
                 sha256=S.sha256(cli.ste_stage1_path)),
    )
    for stage, key in (("coarse", "coarse"), ("fine", "fine")):
        want = freeze["backbones"]["train"][key]["sha256"]
        if binds[key]["sha256"] != want:
            raise SystemExit(f"{key} checkpoint SHA != freeze")
    if binds["ste"]["sha256"] != freeze["backbones"]["train"]["ste_stage1"]["sha256"]:
        raise SystemExit("STE checkpoint SHA != freeze")
    if binds["coarse"]["sha256"] != S.sha256(
            freeze["backbones"]["train"]["coarse"]["path"]):
        raise SystemExit("coarse checkpoint is not the frozen file")

    if os.path.isdir(out_dir):
        if cli.no_resume:
            raise SystemExit(f"{out_dir} exists and --no_resume was given")
        extra = known_entries(out_dir)
        if extra:
            raise SystemExit(f"{out_dir} has foreign entries: {sorted(extra)[:5]}")
        comp_path = os.path.join(out_dir, COMPLETE)
        if os.path.isfile(comp_path):
            comp = S.read_json(comp_path)
            if comp.get("domain") != cli.domain or comp.get("split") != cli.split:
                raise SystemExit("existing export is for another domain/split")
            for name, want in comp.get("cache_sha256", {}).items():
                p = os.path.join(out_dir, name)
                if not os.path.isfile(p) or S.sha256(p) != want:
                    raise SystemExit(f"existing complete export failed SHA: {name}")
            print(f"EXPORT_REUSED {out_dir} (complete, no TEST access)", flush=True)
            return 0
    else:
        os.makedirs(out_dir, exist_ok=False)
    os.makedirs(os.path.join(out_dir, CHUNK_DIR), exist_ok=True)

    bind_sha = {
        "coarse_sha256": binds["coarse"]["sha256"],
        "fine_sha256": binds["fine"]["sha256"],
        "ste_sha256": binds["ste"]["sha256"],
        "freeze_sha256": fchk["freeze_canonical_sha256"],
        "seed": int(cli.seed),
        "k": int(cli.k),
        "candidate_batch": int(cli.candidate_batch),
    }

    # ---- frozen model/data construction, reproduced in the same order ------
    S.install_paths(cli.domain)
    # Compatibility patch 2026-11-04: adapt.install() must run before the core
    # imports, exactly as in the frozen direct exporter; otherwise `models`
    # resolves against the legacy text2loc tree and has no coarse/fine/ste.
    from adapt.domain import get as get_domain, install
    dom_cfg = get_domain(cli.domain)
    install(dom_cfg)
    from torch.utils.data import DataLoader
    import torch
    import torch_geometric.transforms as T
    import datapreparation.kitti360pose.utils as U
    from dataloading.kitti360pose.cells import (
        Kitti360CoarseDataset, Kitti360CoarseDatasetMulti)
    from dataloading.kitti360pose.poses import batch_object_points
    from datapreparation.kitti360pose.imports import Object3d
    from datapreparation.kitti360pose.utils import COLOR_NAMES as COLOR_NAMES_K360
    from models.coarse import CoarseNetwork
    from models.fine import FineNetwork
    from models.ste import FrozenT5
    from training.utils import load_prior

    with open(cli.config, encoding="utf-8") as f:
        cfg = json.load(f)
    args = Namespace(**cfg)
    args.device = cli.device
    if not hasattr(args, "temperature"):
        args.temperature = 0.1
    with open(cli.known_classes, encoding="utf-8") as f:
        known_classes = json.load(f)["known_classes"]
    if len(known_classes) != dom_cfg["labels"]:
        raise SystemExit(f"expected {dom_cfg['labels']} known classes, got "
                         f"{len(known_classes)}")
    dcfg = dict(S.domain_cfg(cli.domain), labels=dom_cfg["labels"],
                manifest_sha256=dom_cfg["manifest_sha256"])
    if not os.path.isfile(cli.frozen_export):
        raise SystemExit(f"frozen exporter not found: {cli.frozen_export}")
    frozen = _load_module(cli.frozen_export, "frozen_export_coordfix_v1")
    frozen_sha = S.sha256(cli.frozen_export)

    device = torch.device(cli.device if cli.device == "cuda"
                          and torch.cuda.is_available() else "cpu")
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    t5 = FrozenT5(args.t5_path, device, max_length=args.max_text_len)
    prior = load_prior(cli.ste_stage1_path, t5.model.config.d_model,
                       args.clip_dim, num_layers=args.prior_layers,
                       num_heads=args.prior_heads)
    coarse = CoarseNetwork(known_classes, COLOR_NAMES_K360, args, t5, prior).to(device)
    coarse_load = frozen.load_light(coarse, cli.coarse_ckpt)
    fine = FineNetwork(known_classes, COLOR_NAMES_K360, args, t5, prior).to(device)
    fine_load = frozen.load_light(fine, cli.fine_ckpt)
    coarse.eval()
    fine.eval()
    coarse_bind = frozen.bind_best(cli.coarse_ckpt)
    fine_bind = frozen.bind_best(cli.fine_ckpt)
    for key, b in (("coarse", coarse_bind), ("fine", fine_bind)):
        fr = freeze["backbones"]["train"][key]
        if b["BEST"].get("light_sha256") != fr["sha256"]:
            raise SystemExit(f"{key}: BEST light sha != freeze")
        if b["BEST"].get("epoch") != EXPECTED_EPOCHS[cli.domain][key]:
            raise SystemExit(f"{key}: BEST epoch != frozen epoch")
        if b["light_state_digest"] != fr["bind"]["light_state_digest"]:
            raise SystemExit(f"{key}: light state digest != freeze")
        if b["light_n_tensors"] != fr["bind"]["light_n_tensors"]:
            raise SystemExit(f"{key}: light tensor count != freeze")

    transform = T.FixedPoints(args.pointnet_numpoints)
    scenes = getattr(U, "SCENE_NAMES_" + cli.split.upper())
    ds = Kitti360CoarseDatasetMulti(args.base_path, scenes, transform,
                                    shuffle_hints=False, flip_poses=False)
    poses = ds.all_poses
    hints = [h for d_ in ds.datasets for h in d_.hint_descriptions]
    target_ids = np.array([p.cell_id for p in poses])
    n_all = len(poses)
    if len(hints) != n_all:
        raise SystemExit(f"hint/pose mismatch {len(hints)} != {n_all}")
    expected = (dcfg["expected_test"] if cli.split == "test"
                else {"pandaset": 2000, "nuscenesfull": 7635}[cli.domain])
    n = min(n_all, cli.limit_queries) if cli.limit_queries else n_all
    if not cli.limit_queries and n != expected:
        raise SystemExit(f"{cli.domain}/{cli.split}: {n} queries != expected {expected}")
    poses, hints, target_ids = poses[:n], hints[:n], target_ids[:n]

    cells_list = ds.all_cells
    cells_dict = {str(c.id): c for c in cells_list}
    cells_ds = ds.get_cell_dataset()
    K = cli.k
    if K > len(cells_list):
        raise SystemExit(f"K={K} exceeds {len(cells_list)} cells")

    ordinals = {}
    query_ids, q_scene = [], []
    for p in poses:
        s = str(p.scene_name)
        k = ordinals.get(s, 0)
        query_ids.append(f"{s}:{k:04d}")
        ordinals[s] = k + 1
        q_scene.append(s)
    if len(set(query_ids)) != n:
        raise SystemExit("query_ids are not unique")
    gt_cell = np.asarray([str(x) for x in target_ids], dtype="<U32")
    gt_pose_w = np.asarray([np.asarray(p.pose_w, np.float64)[0:2] for p in poses])
    scene_arr = np.asarray(q_scene, dtype="<U32")

    identity = dict(
        domain=cli.domain, split=cli.split, n_queries=int(n),
        n_scenes=int(len(set(q_scene))), n_cells=int(len(cells_list)),
        query_ids_unique=bool(len(set(query_ids)) == n),
        query_ids_sha256=S.sha256_bytes(
            ("\n".join(query_ids)).encode()),
        scene_counts={s: q_scene.count(s) for s in sorted(set(q_scene))},
        gt_cells_unique=int(len(set(gt_cell.tolist()))),
        first_query_ids=query_ids[:5], last_query_ids=query_ids[-5:],
        expected_queries=int(expected),
        expected_clusters=(dcfg["expected_test_clusters"]
                           if cli.split == "test" else None),
        cluster_unit=dcfg["cluster_unit"],
        note="identity pre-check only; no metric is computed here",
    )
    S.write_json(identity, os.path.join(out_dir, "identity_precheck.json"))

    enc = None if cli.no_resume else load_encodings(out_dir, n)
    if enc is None:
        t0 = time.time()
        cell_loader = DataLoader(cells_ds, batch_size=args.batch_size,
                                 shuffle=False,
                                 collate_fn=Kitti360CoarseDataset.collate_fn)
        cell_enc, cell_ids = [], []
        with torch.no_grad():
            for batch in cell_loader:
                e = coarse.encode_objects(batch["objects"], batch["object_points"])
                cell_enc.append(e.float().cpu().numpy())
                cell_ids.extend(str(x) for x in batch["cell_ids"])
        cell_enc = np.concatenate(cell_enc, 0)
        cell_ids = np.asarray(cell_ids)
        if len(cell_ids) != len(cells_list):
            raise SystemExit("cell encoding row/count mismatch")
        text_enc = []
        with torch.no_grad():
            for i in range(0, n, args.batch_size):
                e = coarse.encode_text(hints[i:i + args.batch_size])
                text_enc.append(e.float().cpu().numpy())
        text_enc = np.concatenate(text_enc, 0)
        cell_scene = [str(cells_dict[str(x)].scene_name) for x in cell_ids]
        em = save_encodings(out_dir, text_enc, cell_enc, cell_ids, cell_scene,
                            args, cli, binds)
        print(f"ENC compute {time.time()-t0:.1f}s cells={cell_enc.shape} "
              f"queries={text_enc.shape}", flush=True)
    else:
        tg, cg = enc["tg"], enc["cg"]
        cell_ids = enc["cell_ids"]
        cell_scene = enc["meta"]["cell_scene"]
        em = enc["meta"]
        text_enc, cell_enc = tg, cg
        print(f"ENC reused {out_dir}/{ENCODE_DIR}", flush=True)
    cell_ids = np.asarray([str(x) for x in cell_ids])
    cell_scene = np.asarray([str(x) for x in cell_scene])

    scores_all = text_enc @ cell_enc.T
    top_idx = np.argsort(-scores_all, axis=1)[:, :K]
    stored = np.take_along_axis(scores_all, top_idx, axis=1)
    if np.any(np.diff(stored, axis=1) > 0):
        raise SystemExit("storage order is not descending coarse cosine")

    # ---- chunk loop -------------------------------------------------------
    starts = list(range(0, n, cli.chunk_size))
    done = {}
    for start in starts:
        end = min(start + cli.chunk_size, n)
        path = os.path.join(out_dir, CHUNK_DIR, f"q{start:06d}_{end:06d}.npz")
        meta, why = None, None
        if os.path.isfile(path) and not cli.no_resume:
            meta, why = validate_chunk(path, start, end, end - start, bind_sha)
        if meta is not None:
            done[start] = meta
            print(f"CHUNK_REUSED {start}:{end} sha={meta['sha256'][:12]}", flush=True)
            continue
        if cli.replace_chunk is not None and start == cli.replace_chunk:
            qd = quarantine(out_dir, start, cli.reason)
            print(f"CHUNK_QUARANTINED {start} -> {qd}", flush=True)
            meta, why = None, None
        elif os.path.isfile(path):
            raise SystemExit(
                f"chunk {path} failed validation ({why}); preserved in place. "
                f"Re-run with --replace_chunk {start} --reason '<reason>' after "
                f"recording the failure.")
        t0 = time.time()
        rows = []
        for qi in range(start, end):
            torch.manual_seed(cli.seed + qi)
            t_i = fine.encode_text([hints[qi]])
            cids = [str(c) for c in cell_ids[top_idx[qi]]]
            cands = [cells_dict[c] for c in cids]
            chunks = []
            for cstart in range(0, K, cli.candidate_batch):
                chunk = cands[cstart:cstart + cli.candidate_batch]
                objects, points = [], []
                for cell in chunk:
                    objs = list(cell.objects)[:args.pad_size]
                    while len(objs) < args.pad_size:
                        objs.append(Object3d.create_padding())
                    objects.append(objs)
                    points.append(batch_object_points(objs, transform))
                with torch.no_grad():
                    pred = fine.predict_with_text(objects, points, t_i)
                chunks.append(pred.float().cpu().numpy())
            offs = np.concatenate(chunks, 0)
            rows.append(dict(
                qi=qi, cids=cids, cell_index=top_idx[qi].astype(np.int64),
                cell_scene=[str(cells_dict[c].scene_name) for c in cids],
                scores=stored[qi].astype(np.float32),
                offsets=offs.astype(np.float32),
                pred_w=np.asarray([cells_dict[c].bbox_w[0:2] + offs[r] *
                                   cells_dict[c].cell_size
                                   for r, c in enumerate(cids)], np.float64),
                bbox_min=np.asarray([cells_dict[c].bbox_w[0:2] for c in cids],
                                    np.float64),
                cell_center=np.asarray(
                    [(np.asarray(cells_dict[c].bbox_w[0:2], float) +
                      np.asarray(cells_dict[c].bbox_w[3:5], float)) / 2.0
                     for c in cids], np.float64),
                cell_size=np.asarray([float(cells_dict[c].cell_size) for c in cids],
                                     np.float64),
                gt_cell=str(gt_cell[qi]), gt_pose_w=gt_pose_w[qi],
                query_id=query_ids[qi], scene=q_scene[qi],
            ))
        payload = dict(
            query_index=np.arange(start, end, dtype=np.int64),
            cell_ids=np.asarray([r["cids"] for r in rows], dtype="<U32"),
            cell_scene=np.asarray([r["cell_scene"] for r in rows], dtype="<U32"),
            cell_index=np.asarray([r["cell_index"] for r in rows], np.int32),
            scores=np.asarray([r["scores"] for r in rows], np.float32),
            offsets=np.asarray([r["offsets"] for r in rows], np.float32),
            pred_w=np.asarray([r["pred_w"] for r in rows], np.float64),
            cell_bbox_min=np.asarray([r["bbox_min"] for r in rows], np.float64),
            cell_center=np.asarray([r["cell_center"] for r in rows], np.float64),
            cell_size=np.asarray([r["cell_size"] for r in rows], np.float64),
            gt_cell=np.asarray([r["gt_cell"] for r in rows], dtype="<U32"),
            gt_pose_w=np.asarray([r["gt_pose_w"] for r in rows], np.float64),
            query_ids=np.asarray([r["query_id"] for r in rows], dtype="<U32"),
            scene=np.asarray([r["scene"] for r in rows], dtype="<U32"),
        )
        atomic_savez(path, **payload)
        meta = dict(
            status="complete", start=int(start), end=int(end),
            n_queries=int(end - start), sha256=S.sha256(path),
            **bind_sha,
            query_ids_sha256=S.sha256_bytes(
                ("\n".join(payload["query_ids"].tolist())).encode()),
            rng="seed+qi per query; candidate chunks of %d; no cross-query state"
                % cli.candidate_batch,
            code_sha256=dict(export=S.sha256(os.path.abspath(__file__)),
                             common=S.sha256(os.path.join(HERE, "a_test_common.py"))),
            created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )
        S.write_json(meta, sidecar_path(path))
        done[start] = meta
        print(f"CHUNK_DONE {start}:{end} {time.time()-t0:.0f}s "
              f"sha={meta['sha256'][:12]}", flush=True)

    # ---- assemble ---------------------------------------------------------
    missing = [s for s in starts if s not in done]
    if missing:
        raise SystemExit(f"missing chunks: {missing}")
    parts = []
    for start in starts:
        end = min(start + cli.chunk_size, n)
        z = np.load(os.path.join(out_dir, CHUNK_DIR,
                                 f"q{start:06d}_{end:06d}.npz"),
                    allow_pickle=True)
        parts.append({k: z[k] for k in z.files})
    cat = {k: np.concatenate([p[k] for p in parts], 0) for k in parts[0]}
    if cat["query_index"].tolist() != list(range(n)):
        raise SystemExit("assembled query order is not 0..n-1")
    cand_path = os.path.join(out_dir, "candidates.npz")
    atomic_savez(
        cand_path,
        cell_ids=cat["cell_ids"], cell_scene=cat["cell_scene"],
        cell_index=cat["cell_index"], scores=cat["scores"],
        coarse_cos=cat["scores"], offsets=cat["offsets"], pred_w=cat["pred_w"],
        cell_bbox_min=cat["cell_bbox_min"], cell_center=cat["cell_center"],
        cell_size=cat["cell_size"], gt_cell=cat["gt_cell"],
        gt_pose_w=cat["gt_pose_w"], scene=cat["scene"],
        query_ids=cat["query_ids"], query_index=cat["query_index"],
        source_pose_index=cat["query_index"].astype(np.int64),
        order_index=cat["query_index"].astype(np.int64),
        split=np.asarray([cli.split], dtype="<U32"))
    enc_path = os.path.join(out_dir, "coarse_enc.npz")
    atomic_savez(enc_path, tg=text_enc.astype(np.float32),
                 cg=cell_enc.astype(np.float32),
                 cell_ids=cell_ids.astype("<U32"),
                 split=np.asarray([cli.split], dtype="<U32"))

    chunk_manifest = [dict(name=f"q{s:06d}_{min(s+cli.chunk_size,n):06d}.npz",
                           sha256=done[s]["sha256"], start=s,
                           end=min(s + cli.chunk_size, n)) for s in starts]
    chunk_manifest_sha = S.canonical_sha(chunk_manifest)
    marker_sha = S.sha256(cli.marker) if cli.marker else None
    meta = dict(
        implementation="Des4Pos corrected A-domain TEST chunked export",
        protocol=S.PROTOCOL, feature_version=S.FEATURE_VERSION,
        domain=cli.domain, split=cli.split, n_queries=int(n),
        n_scenes=int(len(set(q_scene))), n_cells=int(len(cells_list)), k=int(K),
        chunk_size=int(cli.chunk_size), candidate_batch=int(cli.candidate_batch),
        chunk_manifest=chunk_manifest, chunk_manifest_sha256=chunk_manifest_sha,
        seed=int(cli.seed), device=str(device),
        binding_sha256=bind_sha,
        checkpoints=binds,
        best_binding=dict(coarse=coarse_bind, fine=fine_bind),
        load=dict(coarse=coarse_load, fine=fine_load),
        frozen_export_loader=dict(path=os.path.abspath(cli.frozen_export),
                                  sha256=frozen_sha),
        dataset_manifest_sha256=dcfg["manifest_sha256"],
        labels=int(dcfg["labels"]),
        config=dict(path=os.path.abspath(cli.config),
                    sha256=S.sha256(cli.config)),
        known_classes=dict(path=os.path.abspath(cli.known_classes),
                           sha256=S.sha256(cli.known_classes),
                           n=len(known_classes)),
        freeze=dict(path=os.path.abspath(cli.freeze_list),
                    file_sha256=S.sha256(cli.freeze_list),
                    canonical_sha256=fchk["freeze_canonical_sha256"],
                    protocol_sha256=fchk["protocol_sha256"]),
        encodings=em,
        identity=identity,
        decode="bbox_w[0:2] + offset * cell_size",
        per_query_seed="seed + qi before fine.encode_text",
        rng_policy=dict(
            coarse="torch.manual_seed(seed) before model construction; "
                   "FixedPoints(256) cell draws in deterministic cell order",
            text="single pass at the frozen batch size; no sampling",
            fine="torch.manual_seed(seed + qi) before fine.encode_text; "
                 "candidate chunks of %d; no per-candidate reseed"
                 % cli.candidate_batch,
            chunking="query chunks are independent because each query reseeds",
        ),
        candidate_identity=dict(
            order="descending coarse cosine, np.argsort(-scores) (frozen default)",
            unique_ids_per_query=bool(
                all(len(set(r)) == K for r in cat["cell_ids"].tolist())),
            query_ids_scheme="<scene>:<ordinal:04d> over flattened dataset order",
        ),
        denominator="all queries; no filtering",
        test_marker=dict(path=cli.marker, sha256=marker_sha),
        test_opened=cli.split == "test",
        created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        code_sha256=dict(export=S.sha256(os.path.abspath(__file__)),
                         common=S.sha256(os.path.join(HERE, "a_test_common.py"))),
    )
    S.write_json(meta, os.path.join(out_dir, "export_meta.json"))
    cache_sha = dict(candidates_npz=S.sha256(cand_path),
                     coarse_enc_npz=S.sha256(enc_path),
                     export_meta_json=S.sha256(os.path.join(
                         out_dir, "export_meta.json")))
    comp = dict(status="complete", domain=cli.domain, split=cli.split,
                n_queries=int(n), n_scenes=int(len(set(q_scene))),
                k=int(K), chunk_manifest_sha256=chunk_manifest_sha,
                marker_sha256=marker_sha,
                freeze_canonical_sha256=fchk["freeze_canonical_sha256"],
                cache_sha256={"candidates.npz": cache_sha["candidates_npz"],
                              "coarse_enc.npz": cache_sha["coarse_enc_npz"]},
                meta_sha256=cache_sha["export_meta_json"],
                completed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    S.write_json(comp, os.path.join(out_dir, COMPLETE))

    if cli.split == "test":
        m = S.read_json(cli.marker)
        m["status"] = "exported"
        m["export"] = dict(
            out_dir=out_dir, n_queries=int(n),
            cache_sha256=comp["cache_sha256"], meta_sha256=comp["meta_sha256"],
            chunk_manifest_sha256=chunk_manifest_sha,
            identity_precheck_sha256=S.sha256(
                os.path.join(out_dir, "identity_precheck.json")),
            exported_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        S.write_json(m, cli.marker)
    print(f"EXPORT_COMPLETE {out_dir} n={n} chunks={len(starts)} "
          f"cache={cache_sha['candidates_npz'][:12]}", flush=True)
    return 0


def _load_module(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    sys.exit(main())
