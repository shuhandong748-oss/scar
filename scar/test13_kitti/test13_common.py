# -*- coding: utf-8 -*-
"""TEST13 round-local constants: corrected KITTI backbone on the TEST split.

Purpose
-------
Formal *second* TEST read of the corrected Des4Pos re-implementation
(Coarse epoch 8, Fine epoch 24) plus the frozen SCAR scorer seed 0
(epoch 26).  Everything is labelled EXPLORATORY: the first TEST read happened
in `kitti_bb0_seed0_corev2b` through `evaluation/pipeline.py` (a different
readout), and this round must never be used to tune hyper-parameters or to
re-select epochs.

This module only holds paths / frozen identities; it imports the frozen core
read-only and writes exclusively under ROUND.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

CORE = "/root/autodl-tmp/des4pos_paper_reimplementation"
RUN = CORE + "/runs/corrected_kitti_s0_20261031"
ROUND = CORE + "/runs/corrected_kitti_test13_20261102"
SCAR_ROUND = CORE + "/runs/corrected_kitti_scar_trainval_20261102"
SCAR_OUT = SCAR_ROUND + "/out"
SCAR_HEADS = SCAR_OUT + "/heads"
SCAR_TRAIN_LOG = SCAR_HEADS + "/SCAR_TRAIN_LOG.json"
SCAR_FREEZE = SCAR_OUT + "/eval/FREEZE.json"
VAL09 = CORE + "/runs/val_validation_command09_20261102"

BASE_PATH = "/root/autodl-tmp/Des4Pos/data/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all"
COARSE_CFG = RUN + "/k360_30-10_scG_pd10_pc4_spY_all/coarse_corrected_s0_20ep/config.json"
FINE_CFG = RUN + "/k360_30-10_scG_pd10_pc4_spY_all/fine_corrected_s0_35ep/config.json"
COARSE_CKPT = RUN + "/sidecar_coarse/coarse_best20_light.pth"
FINE_CKPT = RUN + "/sidecar_fine/fine_best_light.pth"
STE_PRIOR = RUN + "/../kitti_bb0_seed0_corev2b/ste_stage1/ste_stage1_prior.pth"
NUM_STATS = RUN + "/k360_30-10_scG_pd10_pc4_spY_all/fine_corrected_s0_35ep/num_stats.json"
KNOWN_CLASSES = RUN + "/k360_30-10_scG_pd10_pc4_spY_all/fine_corrected_s0_35ep/known_classes.json"
POINTNET = "/root/autodl-tmp/Des4Pos/checkpoints/pointnet_acc0.86_lr1_p256.pth"

# round-local TEST cache (written by test13_export.py, read-only afterwards)
TEST_CACHE = ROUND + "/out/cache"
TEST_CHUNKS = ROUND + "/out/chunks"
TEST_META = TEST_CACHE + "/test_export_meta.json"

EXPECT = {
    "coarse_sha256": "95c8139cb19667bf475da2e7af92cbd367abea9f8a1126ceea6ecf0a91d74a98",
    "fine_sha256": "df47c1e26356bf9f7997abcddd20c6cde8226a51df8e0794feadcd9e76732695",
    "coarse_epoch": 8,
    "fine_epoch": 24,
    "coarse_light_digest": "1832bede0ae3dfcf49cbf4e87d19a726f3a7b3a39e918f6436cf1a4e7bf0bc26",
    "fine_light_digest": "fef42817f6e21abf0bd621312540040ac6a8d7c9099674292ba192bd76dd5b42",
    "ste_prior_sha256": "41220702eaa468c883f567fe9a90522c5fb2af5bc9eebbf9a82c21b41bece4a5",
    "pointnet_sha256": "ef63fdfc9cd10e698181efe415a873e628cf86698266f005351943030016e8e1",
    "num_stats_sha256": "29ed84c63d2af4af495fe4c2ef7917a8afe82923293da433e37f17317d5f9182",
    "known_classes_sha256": "f1ad155e7ccd1db5654c77b4c14c95d60a1664efe7647e4b9896d79728dfe01a",
    "num_stats_mean": 1826.6844940968194,
    "num_stats_std": 2516.8905096993817,
    "n_train_queries": 28807,
    "n_train_cells": 11259,
    "n_val_queries": 3187,
    "n_val_cells": 1434,
    "val_scene": "2013_05_28_drive_0010_sync",
    "test_scenes": ["2013_05_28_drive_0003_sync",
                    "2013_05_28_drive_0005_sync",
                    "2013_05_28_drive_0009_sync"],
}

# Frozen SCAR scorer (formal selection rule B; user decision: s0 only on TEST)
SCAR_S0 = SCAR_HEADS + "/scar23_s0.pt"
SCAR_S0_SHA256 = "4d875d4e654bd929b15cbaaada9529254e9ac536780675887514f320c1562f40"
SCAR_SELECTED_EPOCH = 26
SCAR_SEEDS = (0, 1, 2)          # s1/s2 exist but are NOT used on TEST

SEED = 0
TOP_K = 25
NATIVE_K = 10
CELL_TOP_K = (1, 2, 3, 5, 10, 25)
LOC_TOP_K = (1, 2, 5, 10, 25)
THRESHS = (5.0, 10.0, 15.0)

FEATURE_NAMES = [
    "cos", "cos_rank", "sp", "p0_rank", "p2_rank", "is_tied", "tied_size",
    "sc_center", "sc_center_rank", "off_dx", "off_dy", "off_norm",
    "sc_fine", "sat_fine", "sc_fine_rank", "borda", "c_rank",
    "n_obj", "n_matched", "dist_nearest_matched", "dist_centroid_matched",
    "pred_dist_anchor", "center_dist_anchor",
]
assert len(FEATURE_NAMES) == 23
LAMBDA = 0.5


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_prior_key(key):
    return key == "prior" or key.startswith("prior.") or ".prior." in key


def state_dict_digest(state_dict):
    import torch
    h = hashlib.sha256()
    for k in sorted(state_dict):
        v = state_dict[k]
        h.update(k.encode("utf-8"))
        h.update(b"\0")
        if torch.is_tensor(v):
            t = v.detach().cpu().contiguous()
            h.update(str(tuple(t.shape)).encode())
            h.update(str(t.dtype).encode())
            h.update(t.numpy().tobytes())
        else:
            h.update(repr(v).encode())
        h.update(b"\n")
    return h.hexdigest()


def data_fingerprint(ds):
    poses = list(ds.all_poses)
    cells = list(ds.all_cells)
    qids = [str(p.cell_id) for p in poses]
    cids = [str(c.id) for c in cells]
    return {
        "n_queries": len(poses),
        "n_cells": len(cells),
        "scenes": sorted(str(s) for s in getattr(ds, "scene_names", [])),
        "dataset_full_names": [str(d.scene_name) for d in ds.datasets],
        "query_ids_sha256": hashlib.sha256("\n".join(qids).encode("utf-8")).hexdigest(),
        "gallery_ids_sha256": hashlib.sha256("\n".join(cids).encode("utf-8")).hexdigest(),
    }


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path, obj):
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def write_jsonl_atomic(path, rows):
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(tmp)), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def save_npz_atomic(path, arrays, compressed=False):
    import numpy as np
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp, "wb") as f:
        (np.savez_compressed if compressed else np.savez)(f, **arrays)
    os.replace(tmp, path)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def disk_report(path=ROUND):
    import shutil
    du = shutil.disk_usage(path)
    return {"total": du.total, "used": du.used, "free": du.free,
            "free_GiB": round(du.free / (1 << 30), 3),
            "free_at_least_1GiB": du.free >= 1 << 30}


# ------------------------------------------------------------ TEST read marker
def test_read_marker_path():
    return os.path.join(ROUND, "out", "state", "TEST_READ_2.json")


def write_test_read_marker(stage, status, extra=None):
    """Record that this round reads TEST, why, and at which stage."""
    p = test_read_marker_path()
    cur = read_json(p) if os.path.isfile(p) else {}
    cur.setdefault("test_read_number", 2)
    cur.setdefault("exploratory", True)
    cur.setdefault("scope", "corrected KITTI backbone (Coarse ep8 / Fine ep24) "
                            "Top-25 export + five rerank readouts + SCAR s0 ep26")
    cur.setdefault("authorisation", "explicit operator approval: formal second "
                                    "TEST read, every number labelled exploratory")
    cur.setdefault("first_read",
                   "runs/kitti_bb0_seed0_corev2b/eval_test_once.json (core CLI, "
                   "different readout; NOT comparable to this round)")
    cur.setdefault("restrictions", ["no hyper-parameter tuning",
                                    "no epoch re-selection",
                                    "s1/s2 heads not evaluated on TEST",
                                    "results not merged with the first read"])
    cur["updated_at"] = now()
    cur["stage"] = stage
    cur["status"] = status
    cur.setdefault("history", []).append({"at": cur["updated_at"], "stage": stage,
                                          "status": status})
    if extra:
        cur.update(extra)
    write_json_atomic(p, cur)
    return cur


def load_light(model, path, prior, expect_sha, expect_epoch, expect_digest, label):
    """Strict light-state load; identical semantics to COMMAND 09 / 12."""
    import torch
    got_sha = sha256_file(path)
    if got_sha != expect_sha:
        raise SystemExit(f"{label}: checkpoint sha256 {got_sha} != {expect_sha}")
    payload = torch.load(path, map_location="cpu")
    state = payload["model"]
    digest = state_dict_digest(state)
    if digest != payload.get("light_state_digest"):
        raise SystemExit(f"{label}: light_state_digest mismatch ({digest})")
    if digest != expect_digest:
        raise SystemExit(f"{label}: digest {digest} != expected {expect_digest}")
    if int(payload.get("n_tensors")) != len(state):
        raise SystemExit(f"{label}: n_tensors mismatch")
    extra = payload.get("extra") or {}
    if extra.get("epoch") != expect_epoch:
        raise SystemExit(f"{label}: checkpoint epoch {extra.get('epoch')} != {expect_epoch}")

    sd_before = model.state_dict()
    expected_missing = sorted(k for k in sd_before if is_prior_key(k))
    if not expected_missing:
        raise SystemExit(f"{label}: model exposes no prior tensors; cannot verify "
                         f"frozen-dependency reconstruction")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if sorted(missing) != expected_missing:
        raise SystemExit(f"{label}: missing keys are not exactly the frozen prior: "
                         f"missing={sorted(missing)} expected={expected_missing}")
    if unexpected:
        raise SystemExit(f"{label}: unexpected keys {unexpected}")
    prior_state = prior.state_dict()
    mapped = {}
    for key in expected_missing:
        assert ".prior." in key, key
        mapped[key.split(".prior.", 1)[1]] = key
    if sorted(mapped) != sorted(prior_state):
        raise SystemExit(f"{label}: prior key sets differ: model={sorted(mapped)} "
                         f"file={sorted(prior_state)}")
    sd_after = model.state_dict()
    for sub, key in mapped.items():
        if not torch.equal(sd_after[key].detach().cpu(), prior_state[sub].detach().cpu()):
            raise SystemExit(f"{label}: prior tensor {sub} differs from the frozen file")
    prior_digest = state_dict_digest({k: v.detach().cpu()
                                      for k, v in prior_state.items()})
    return {
        "checkpoint": path, "sha256": got_sha, "bytes": os.path.getsize(path),
        "epoch": extra.get("epoch"), "light_state_digest": digest,
        "n_tensors_loaded": len(state), "n_params_loaded": int(payload.get("n_params")),
        "missing_keys": expected_missing,
        "missing_keys_are_exactly_frozen_prior": True,
        "unexpected_keys": [],
        "prior_tensors_rebuilt": len(mapped),
        "prior_state_digest": prior_digest,
        "prior_equal_to_frozen_file": True,
        "extra_num_stats_present": "num_stats" in extra,
        "extra_keys": sorted(extra.keys()),
    }
