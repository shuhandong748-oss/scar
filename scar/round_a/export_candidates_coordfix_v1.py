"""Export corrected-backbone candidates (K=25) for PandaSet / nuScenesFull.

This export is **coordinate-fix independent**: it only stores Coarse candidate
identity/order/cosine, Fine offsets/pred_w, GT and query identity.  None of
those passes through ``common.build_cell_aux`` (see IMPACT_SCOPE_ZH.md), so the
same cache feeds both the old-defect diagnostic features and
FEATURE_COORD_FIX_v1.

Identity rules copied verbatim from the frozen A exporters:
  * storage order = stable descending coarse cosine;
  * offsets = Fine output in cell-normalised coordinates;
  * pred_w = ``bbox_w[0:2] + offset * cell_size``;
  * per-query RNG seed = ``seed + qi``; text encoded once per query;
  * candidate chunks of 10; ``torch.no_grad``; 2-D ``gt_pose_w``.
nuScenesFull additionally stores ``query_ids`` (<scene>:<ordinal:04d>),
``source_pose_index`` and ``order_index``.

Only TRAIN/VAL are accepted.  TEST does not exist as a valid split here.

The corrected light checkpoints omit the frozen prior tensors; they are loaded
with strict=False and the missing keys must be exactly the prior keys.  Each
file is bound to its BEST.json ``light_sha256`` / ``light_state_digest``.
"""
import argparse
import json
import os
import time
from argparse import Namespace

import numpy as np
import torch

from adapt.domain import DOMAINS, get as get_domain, install


def parse_cli(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True, choices=sorted(DOMAINS))
    p.add_argument("--config", required=True, help="frozen coarse run config.json")
    p.add_argument("--coarse_ckpt", required=True, help="coarse_best_light.pth")
    p.add_argument("--fine_ckpt", required=True, help="fine_best_light.pth")
    p.add_argument("--known_classes", required=True)
    p.add_argument("--ste_stage1_path", required=True)
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--k", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--candidate_batch", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit_queries", type=int, default=None)
    return p.parse_args(argv)


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bind_best(ckpt_path):
    """Read BEST.json beside the light checkpoint and verify its bindings."""
    d = os.path.dirname(os.path.abspath(ckpt_path))
    with open(os.path.join(d, "BEST.json"), encoding="utf-8") as f:
        best = json.load(f)
    file_sha = sha256_file(ckpt_path)
    if best.get("light_sha256") != file_sha:
        raise SystemExit(f"{ckpt_path}: file SHA {file_sha} != BEST.json "
                         f"{best.get('light_sha256')}")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if payload.get("light_state_digest") != best.get("light_state_digest"):
        raise SystemExit(f"{ckpt_path}: payload state digest != BEST.json")
    return {"BEST_path": os.path.join(d, "BEST.json"),
            "BEST_sha256": sha256_file(os.path.join(d, "BEST.json")),
            "BEST": best,
            "light_state_digest": payload.get("light_state_digest"),
            "light_n_tensors": payload.get("n_tensors")}


def load_light(model, ckpt_path):
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload \
        else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise SystemExit(f"{ckpt_path}: unexpected keys {unexpected[:5]}")
    if not missing or not all(
            (".prior." in k) or k.startswith("prior.") for k in missing):
        raise SystemExit(f"{ckpt_path}: missing keys are not exactly prior keys: "
                         f"{missing[:8]}")
    return {"missing_keys": list(missing), "n_missing": len(missing),
            "n_loaded": len(state)}


def main(argv=None):
    cli = parse_cli(argv)
    if cli.k != 25:
        raise SystemExit(f"the frozen builder requires K=25, got {cli.k}")
    d = get_domain(cli.domain)
    out_dir = os.path.abspath(cli.out_dir)
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        raise SystemExit(f"refusing to overwrite non-empty {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    install(d)

    from torch.utils.data import DataLoader
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
    from training.utils import load_prior, save_json

    with open(cli.config, encoding="utf-8") as f:
        cfg = json.load(f)
    args = Namespace(**cfg)
    args.device = cli.device
    if cli.batch_size:
        args.batch_size = cli.batch_size
    if not hasattr(args, "temperature"):
        args.temperature = 0.1

    with open(cli.known_classes, encoding="utf-8") as f:
        known_classes = json.load(f)["known_classes"]
    if len(known_classes) != d["labels"]:
        raise SystemExit(f"expected {d['labels']} target classes, got "
                         f"{len(known_classes)}")

    device = torch.device(cli.device if cli.device == "cuda"
                          and torch.cuda.is_available() else "cpu")
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    t5 = FrozenT5(args.t5_path, device, max_length=args.max_text_len)
    prior = load_prior(cli.ste_stage1_path, t5.model.config.d_model,
                       args.clip_dim, num_layers=args.prior_layers,
                       num_heads=args.prior_heads)
    coarse = CoarseNetwork(known_classes, COLOR_NAMES_K360, args, t5, prior).to(device)
    coarse_load = load_light(coarse, cli.coarse_ckpt)
    fine = FineNetwork(known_classes, COLOR_NAMES_K360, args, t5, prior).to(device)
    fine_load = load_light(fine, cli.fine_ckpt)
    coarse.eval()
    fine.eval()
    coarse_bind = bind_best(cli.coarse_ckpt)
    fine_bind = bind_best(cli.fine_ckpt)
    if coarse_bind["light_n_tensors"] != coarse_load["n_loaded"]:
        raise SystemExit("coarse light n_tensors != loaded tensors")
    if fine_bind["light_n_tensors"] != fine_load["n_loaded"]:
        raise SystemExit("fine light n_tensors != loaded tensors")

    transform = T.FixedPoints(args.pointnet_numpoints)
    scenes = getattr(U, "SCENE_NAMES_" + cli.split.upper())
    ds = Kitti360CoarseDatasetMulti(args.base_path, scenes, transform,
                                    shuffle_hints=False, flip_poses=False)
    poses = ds.all_poses
    hints = [h for d_ in ds.datasets for h in d_.hint_descriptions]
    target_ids = np.array([p.cell_id for p in poses])
    n_all = len(poses)
    assert len(hints) == n_all, (len(hints), n_all)
    expected = d["n_train"] if cli.split == "train" else d["n_val"]
    if cli.limit_queries:
        n = min(n_all, cli.limit_queries)
        poses, hints, target_ids = poses[:n], hints[:n], target_ids[:n]
    else:
        n = n_all
        if n != expected:
            raise SystemExit(f"split {cli.split}: expected {expected} queries, "
                             f"got {n}")

    query_ids = None
    if cli.domain == "nuscenesfull":
        ordinal = {}
        query_ids = []
        for p in poses:
            s = str(p.scene_name)
            k = ordinal.get(s, 0)
            query_ids.append(f"{s}:{k:04d}")
            ordinal[s] = k + 1
        if len(set(query_ids)) != n:
            raise SystemExit("query_ids are not unique")
    source_pose_index = np.arange(n, dtype=np.int64)
    order_index = np.arange(n, dtype=np.int64)

    cells_list = ds.all_cells
    cells_dict = {str(c.id): c for c in cells_list}
    cells_ds = ds.get_cell_dataset()
    K = cli.k
    if K > len(cells_list):
        raise SystemExit(f"K={K} exceeds the {len(cells_list)} cells of {cli.split}")

    cell_loader = DataLoader(cells_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=Kitti360CoarseDataset.collate_fn)
    cell_enc, cell_ids = [], []
    with torch.no_grad():
        for batch in cell_loader:
            e = coarse.encode_objects(batch["objects"], batch["object_points"])
            cell_enc.append(e.float().cpu().numpy())
            cell_ids.extend(str(x) for x in batch["cell_ids"])
    cell_enc = np.concatenate(cell_enc, 0)
    cell_ids = np.array(cell_ids)
    if len(cell_ids) != len(cells_list):
        raise SystemExit(f"cell encoding rows {len(cell_ids)} != cells "
                         f"{len(cells_list)}")

    text_enc = []
    with torch.no_grad():
        for i in range(0, n, args.batch_size):
            e = coarse.encode_text(hints[i:i + args.batch_size])
            text_enc.append(e.float().cpu().numpy())
    text_enc = np.concatenate(text_enc, 0)
    scores = text_enc @ cell_enc.T
    top_idx = np.argsort(-scores, axis=1)[:, :K]
    stored = np.take_along_axis(scores, top_idx, axis=1)
    if np.any(np.diff(stored, axis=1) > 0):
        raise SystemExit("storage order is not descending coarse cosine")

    offsets_all = np.zeros((n, K, 2), np.float64)
    pred_w_all = np.zeros((n, K, 2), np.float64)
    t0 = time.time()
    for qi in range(n):
        torch.manual_seed(cli.seed + qi)
        t_i = fine.encode_text([hints[qi]])
        cids = [str(c) for c in cell_ids[top_idx[qi]]]
        cands = [cells_dict[c] for c in cids]
        chunks = []
        for start in range(0, K, cli.candidate_batch):
            chunk = cands[start:start + cli.candidate_batch]
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
        offsets_all[qi] = offs
        for r, cell in enumerate(cands):
            pred_w_all[qi, r] = cell.bbox_w[0:2] + offs[r] * cell.cell_size
        if (qi + 1) % 200 == 0:
            el = time.time() - t0
            print(f"{qi+1}/{n} {el:.0f}s eta {(n-qi-1)*el/(qi+1)/60:.1f} min",
                  flush=True)

    gt_cell = np.array([str(x) for x in target_ids], dtype="<U32")
    gt_pose_w = np.array([np.asarray(p.pose_w, np.float64)[0:2] for p in poses])
    if gt_pose_w.shape != (n, 2):
        raise SystemExit(f"gt_pose_w shape {gt_pose_w.shape}, expected {(n, 2)}")
    payload = dict(
        cell_ids=np.array([[str(c) for c in row] for row in cell_ids[top_idx]],
                          dtype="<U32"),
        offsets=offsets_all.astype(np.float32),
        pred_w=pred_w_all.astype(np.float64),
        gt_cell=gt_cell,
        gt_pose_w=gt_pose_w.astype(np.float64),
        coarse_cos=stored.astype(np.float32),
        query_index=np.arange(n, dtype=np.int64),
        split=np.asarray([cli.split], dtype="<U32"),
    )
    if query_ids is not None:
        payload.update(
            query_ids=np.asarray(query_ids, dtype="<U24"),
            source_pose_index=source_pose_index,
            order_index=order_index,
        )
    np.savez_compressed(os.path.join(out_dir, "candidates.npz"), **payload)
    np.savez_compressed(
        os.path.join(out_dir, "coarse_enc.npz"),
        tg=text_enc.astype(np.float32),
        cg=cell_enc.astype(np.float32),
        cell_ids=cell_ids.astype("<U32"),
        split=np.asarray([cli.split], dtype="<U32"),
    )

    meta = {
        "implementation": "Des4Pos paper reimplementation core v2 (not authors' code)",
        "protocol": "CORR-D4P-SCAR-FORMAL-v1 corrected-backbone candidates",
        "domain": cli.domain,
        "split": cli.split,
        "n_queries": int(n),
        "n_cells": int(len(cells_list)),
        "k": int(K),
        "candidate_batch": int(cli.candidate_batch),
        "decode": "bbox_w[0:2] + offset * cell_size",
        "per_query_seed": "seed + qi (matches evaluation.pipeline)",
        "seed": int(cli.seed),
        "checkpoints": {
            "coarse": {"path": cli.coarse_ckpt, "sha256": sha256_file(cli.coarse_ckpt),
                       "bind": coarse_bind, "load": coarse_load},
            "fine": {"path": cli.fine_ckpt, "sha256": sha256_file(cli.fine_ckpt),
                     "bind": fine_bind, "load": fine_load},
            "ste_stage1": {"path": cli.ste_stage1_path,
                           "sha256": sha256_file(cli.ste_stage1_path)},
        },
        "known_classes": {"path": cli.known_classes,
                          "sha256": sha256_file(cli.known_classes),
                          "n": len(known_classes)},
        "config": {"path": cli.config, "sha256": sha256_file(cli.config)},
        "data": {"base_path": args.base_path,
                 "manifest_sha256": d["manifest_sha256"]},
        "coord_fix_dependence": "none: candidates/Fine/GT do not use build_cell_aux",
        "read_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "test_read": False,
    }
    if query_ids is not None:
        meta["query_id_scheme"] = "<scene>:<ordinal:04d>; flattened dataset order"
    save_json(meta, os.path.join(out_dir, "export_meta.json"))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
