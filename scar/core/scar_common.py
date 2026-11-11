#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared bindings for the corrected Des4Pos formal-SCAR round.

Every consumer (export, features, trainers, evaluation, verifiers) loads the
same frozen backbone records and applies the same checks.  Nothing here
imports the core or touches the GPU.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from argparse import Namespace

ROOT = "/root/autodl-tmp/d4p_trainval_20261104/C"
CORR = "/root/autodl-tmp/des4pos_corrected_C_20261102"
CORE = os.path.join(CORR, "code", "core")
ADAPT = os.path.join(CORR, "code", "adapt")
GATE = os.path.join(ROOT, "records", "GATE_BINDING.json")
PROTOCOL_LOCAL_SHA = None  # filled by the orchestrator after upload

EXPECTED_HOST = "autodl-container-<CONTAINER_ID>"
EXPECTED_GPU = "GPU-<GPU_UUID>"

DOMAINS = {
    "npm3d": {
        "run": os.path.join(CORR, "npm3d_s0"),
        "base": "/root/autodl-tmp/des4pos_npm3d/data/NPM3D",
        "vendor": "/root/autodl-tmp/des4pos_npm3d/vendor",
        "adapter": "npm3d_domain",
        "prior": "/root/autodl-tmp/des4pos_npm3d/runs/R059-C-Des4PosPaper-NPM3D-bb0/ste_stage1/ste_stage1_prior.pth",
        "pointnet": "/root/autodl-tmp/des4pos_npm3d/assets/pointnet/pointnet_acc0.86_lr1_p256.pth",
        "num_stats": "/root/autodl-tmp/des4pos_npm3d/registration/R059-C-Des4PosPaper-NPM3D-bb0/num_stats.json",
        "t5": "/root/autodl-tmp/des4pos_npm3d/assets/t5-large",
        "colors": ["beige", "gray", "gray-green"],
        "coarse_epoch": 20, "fine_epoch": 30,
        "coarse_sha": "3a73ee4282e8cf4fdb575fa5458fffd8be755d83464d87d469c4cde42a627645",
        "fine_sha": "72842717de4e499d337b8e06f5bf6b34071e0d139075f8484e63238f289f1e0c",
        "cluster_unit": "scene",
        "expected": {"train": 120, "val": 128, "test": 116},
        "expected_clusters": {"train": 30, "val": 32, "test": 29},
        "historical_test_marker":
            "/root/autodl-tmp/des4pos_npm3d/runs/R059-C-Des4PosPaper-NPM3D-bb0/"
            "export/test/TEST_READ_ONCE.json",
        "test_marker": None,
    },
    "s3dis": {
        "run": os.path.join(CORR, "s3dis_s0"),
        "base": "/root/autodl-tmp/text2loc_s3dis/data/S3DISPose",
        "vendor": "/root/autodl-tmp/des4pos_s3dis/vendor",
        "adapter": "s3dis_domain",
        "prior": "/root/autodl-tmp/des4pos_s3dis/runs/R060_D4P_S3DIS_bb0/ste_stage1/ste_stage1_prior.pth",
        "pointnet": "/root/autodl-tmp/text2loc_s3dis/code/checkpoints/pointnet_acc0.86_lr1_p256.pth",
        "num_stats": "/root/autodl-tmp/des4pos_s3dis/registration/R060_D4P_S3DIS_bb0/num_stats.json",
        "t5": "/root/autodl-tmp/text2loc_s3dis/assets/t5-large",
        "colors": None,  # adapter S3DIS_COLORS
        "coarse_epoch": 15, "fine_epoch": 34,
        "coarse_sha": "816fafa2a4fb6fb85f04ec1d0bf7676a5f68dae622288c60fe0c975b757e7058",
        "fine_sha": "bc816b5ae083421a9ef4b9966ca71ffa19eeae08523f491875790fa583ec1980",
        "cluster_unit": "room",
        "expected": {"train": 1084, "val": 544, "test": 544},
        "expected_clusters": {"train": 136, "val": 68, "test": 68},
        "historical_test_marker":
            "/root/autodl-tmp/des4pos_s3dis/runs/R060_D4P_S3DIS_bb0/"
            "export/test/TEST_READ_ONCE.json",
        "test_marker": None,
    },
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sha256_tree(root):
    h = hashlib.sha256()
    entries = []
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d != "__pycache__")
        for n in sorted(fns):
            if n.endswith((".pyc", ".pyo")):
                continue
            p = os.path.join(dp, n)
            entries.append((os.path.relpath(p, root).replace(os.sep, "/"), p))
    for rel, p in sorted(entries):
        h.update(rel.encode() + b"\0" + str(os.path.getsize(p)).encode() + b"\0")
        h.update(sha256(p).encode() + b"\n")
    return h.hexdigest(), len(entries)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def num_stats_fingerprint(stats):
    payload = {
        "version": int(stats["version"]), "domain": stats["domain"],
        "split": stats["split"], "provenance": stats["provenance"],
        "mean": float(stats["mean"]), "std": float(stats["std"]),
        "std_ddof": int(stats["std_ddof"]),
        "pinned_zero_variance": bool(stats["pinned_zero_variance"]),
        "counts_sha256": stats.get("counts_sha256"),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def load_gate():
    g = read_json(GATE)
    if g.get("all_pass") is not True:
        raise SystemExit("gate binding is not PASS")
    return {
        "path": GATE,
        "sha256": sha256(GATE),
        "all_pass": True,
        "hostname": (g.get("gate", {}).get("hostname") or {}).get("detail"),
        "domains": {k: bool(v.get("all_pass")) for k, v in g.get("domains", {}).items()},
    }


def domain_cfg(domain):
    if domain not in DOMAINS:
        raise SystemExit(f"unknown domain {domain}")
    cfg = dict(DOMAINS[domain])
    cfg["adapter_path"] = os.path.join(ADAPT, cfg["adapter"] + ".py")
    return cfg


def import_adapter(domain):
    """Import the domain adapter with DES4POS_CORE pointed at the corrected core."""
    import sys
    os.environ["DES4POS_CORE"] = CORE
    if ADAPT not in sys.path:
        sys.path.insert(0, ADAPT)
    mod = __import__(DOMAINS[domain]["adapter"])
    if os.path.realpath(mod.CORE) != os.path.realpath(CORE):
        raise SystemExit(f"adapter core {mod.CORE} != corrected core {CORE}")
    return mod


# The NPM3D adapter defines labels/counts but no colour constant.  These three
# are the colours actually present: read over all 62 NPM3D TRAIN+VAL cells
# (248 descriptions) on 2026-11-04, set equality with the data, sorted order.
# They are not copied from another domain and no colour outside this set occurs.
NPM3D_COLORS = ("beige", "gray", "gray-green")


def domain_vocabulary(domain):
    """Colours, classes and mentioned-relation count exactly as the adapter."""
    mod = import_adapter(domain)
    if domain == "npm3d":
        return (list(NPM3D_COLORS), list(mod.NPM3D_LABELS),
                int(mod.NUM_MENTIONED))
    # the S3DIS adapter has no mentioned-count constant; the frozen protocol
    # (and the historical builder) uses 4
    return list(mod.S3DIS_COLORS), list(mod.S3DIS_LABELS), 4


def stage_records(domain):
    d = DOMAINS[domain]
    out = {}
    for s in ("coarse", "fine"):
        sd = os.path.join(d["run"], s)
        out[s] = {
            "dir": sd,
            "result": read_json(os.path.join(sd, "STAGE_RESULT.json")),
            "freeze": read_json(os.path.join(sd, "STAGE_FREEZE.json")),
            "init": read_json(os.path.join(sd, "INIT_FREEZE.json")),
            "best": read_json(os.path.join(sd, "BEST.json")),
            "ckpt": os.path.join(sd, f"{s}_best.pth"),
        }
    return out


def inference_config(domain):
    """STAGE_FREEZE config of the coarse stage, plus the registered num_stats path.

    Only training-only keys differ between the coarse and fine configs; the
    architecture keys are asserted identical by the gate binding.
    """
    d = DOMAINS[domain]
    rec = stage_records(domain)
    cfg = dict(rec["coarse"]["freeze"]["config"])
    stats = read_json(d["num_stats"])
    cfg["num_stats"] = stats            # the registered, never-refitted values
    cfg["num_stats_path"] = d["num_stats"]
    cfg["allow_test"] = False
    args = Namespace(**cfg)
    args.device = "cuda"
    return args, cfg


def verify_backbones(domain):
    """Hash and fingerprint the two frozen checkpoints; no model is built."""
    import torch
    d = DOMAINS[domain]
    rec = stage_records(domain)
    checks = {}
    for s, want_epoch, want_sha in (
            ("coarse", d["coarse_epoch"], d["coarse_sha"]),
            ("fine", d["fine_epoch"], d["fine_sha"])):
        path = rec[s]["ckpt"]
        got = sha256(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        extra = payload.get("extra", {})
        # epoch/domain live in the stage records; the light payload may or may
        # not repeat them, so a repeated value must merely agree
        epoch_ok = (rec[s]["result"].get("best_epoch") == want_epoch
                    == rec[s]["best"].get("epoch")
                    and extra.get("epoch") in (None, want_epoch))
        domain_ok = (rec[s]["result"].get("domain") == domain
                     and extra.get("domain") in (None, domain))
        checks[s] = {
            "path": path,
            "sha256": got,
            "sha_ok": got == want_sha,
            "result_sha_ok": rec[s]["result"].get("best_file_sha256") == got,
            "epoch_ok": epoch_ok,
            "domain_ok": domain_ok,
            "epoch_source": "STAGE_RESULT/BEST (payload extra if present)",
            "n_tensors": len(payload["model"]),
            "light_state_digest": payload.get("light_state_digest"),
            "num_stats_fingerprint": extra.get("num_stats_fingerprint"),
            "top_keys": list(payload.keys()),
        }
    fp = num_stats_fingerprint(read_json(d["num_stats"]))
    checks["num_stats"] = {"path": d["num_stats"], "sha256": sha256(d["num_stats"]),
                           "fingerprint": fp}
    ok = (checks["coarse"]["sha_ok"] and checks["fine"]["sha_ok"]
          and checks["coarse"]["result_sha_ok"] and checks["fine"]["result_sha_ok"]
          and checks["coarse"]["epoch_ok"] and checks["fine"]["epoch_ok"]
          and checks["coarse"]["domain_ok"] and checks["fine"]["domain_ok"]
          and checks["coarse"]["num_stats_fingerprint"] == fp
          and checks["fine"]["num_stats_fingerprint"] == fp)
    checks["PASS"] = bool(ok)
    if not ok:
        raise SystemExit("backbone binding failed: " + json.dumps(checks, default=str)[:2000])
    return checks


def _is_prior_key(k):
    return k == "prior" or k.startswith("prior.") or ".prior." in k


def load_light(model, path):
    """Strict light-state load: the missing keys must be exactly the frozen prior.

    The prior submodule is excluded from the delivered light state and rebuilt
    from the registered artefact at construction time, so those are the only
    keys allowed to be absent.  Anything else missing or unexpected is a
    construction/checkpoint mismatch and fails the run.
    """
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise SystemExit(f"{path}: not a corrected light-state payload")
    state = payload["model"]
    sd_before = model.state_dict()
    expected_missing = sorted(k for k in sd_before if _is_prior_key(k))
    if not expected_missing:
        raise SystemExit(f"{path}: model exposes no prior tensors")
    if any(_is_prior_key(k) for k in state):
        raise SystemExit(f"{path}: light state unexpectedly contains prior tensors")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if sorted(missing) != expected_missing:
        raise SystemExit(f"{path}: missing keys are not exactly the frozen prior: "
                         f"missing={sorted(missing)[:5]} "
                         f"expected={expected_missing[:5]}")
    if unexpected:
        raise SystemExit(f"{path}: unexpected keys {unexpected[:5]}")
    payload["_light_check"] = dict(
        missing_keys=len(missing), unexpected=0, n_tensors=len(state),
        light_state_digest=payload.get("light_state_digest"))
    return payload


def arm_test_dir(out_dir, protocol_sha, freeze_sha, split):
    """TEST gate: the directory and marker must exist before any TEST read."""
    marker = os.path.join(out_dir, "TEST_READ_ONCE.json")
    if split != "test":
        if os.path.isfile(marker):
            raise SystemExit(f"{out_dir} carries a TEST marker; refusing a non-TEST write")
        existing = [n for n in os.listdir(out_dir)] if os.path.isdir(out_dir) else []
        if existing:
            raise SystemExit(f"{out_dir} is not empty: {existing[:5]}")
        os.makedirs(out_dir, exist_ok=True)
        return None, "new"
    if not os.path.isdir(out_dir):
        raise SystemExit(f"TEST output dir {out_dir} must be created (armed) first")
    if not os.path.isfile(marker):
        raise SystemExit(f"TEST marker {marker} must be armed before reading TEST")
    m = read_json(marker)
    if m.get("armed") is not True or m.get("status") != "armed":
        raise SystemExit(f"TEST marker is not armed: {m}")
    if m.get("protocol_freeze_sha256") != protocol_sha:
        raise SystemExit("TEST marker protocol sha mismatch")
    if freeze_sha and m.get("freeze_list_sha256") != freeze_sha:
        raise SystemExit("TEST marker freeze-list sha mismatch")
    extra = sorted(set(os.listdir(out_dir)) - {"TEST_READ_ONCE.json"})
    if extra:
        raise SystemExit(f"armed TEST dir is not clean: {extra[:5]}")
    return m, "armed"


def complete_test_marker(out_dir, marker, **fields):
    if not marker:
        return
    marker.update(fields)
    marker["status"] = "completed"
    marker["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(marker, os.path.join(out_dir, "TEST_READ_ONCE.json"))
