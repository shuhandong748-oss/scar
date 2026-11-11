#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for the C-domain exploratory TEST round.

The frozen C code is imported, never copied or edited:
  * scar_common.py       -- bindings, hashes, marker gate, backbone checks
  * d4p_export.py        -- reference direct export (used by the VAL gate)
  * d4p_features.py      -- canonical-23 feature builder (called in-process)
  * d4p_eval.py          -- reference evaluation (s0/s1/s2; not used for TEST)

Only two things are added here:
  1. query-level chunking for the export, with per-chunk sidecars;
  2. an additive, in-memory S3DIS TEST count: 544 (frozen guard) -> 542 (the
     released split).  The frozen file and the historical marker are never
     modified.  The 542 is derived and checked against the denominator report
     and the raw dataset manifest SHA.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN_CODE = os.environ.get(
    "D4P_C_FROZEN_CODE", "/root/autodl-tmp/d4p_trainval_20261104/C/code")
TEST_ROOT = os.environ.get(
    "D4P_C_TEST_ROOT", "/root/autodl-tmp/d4p_test_20261104/C")
PROTOCOL_NAME = "CORR-D4P-SCAR-FORMAL-v1"
S3DIS_MANIFEST_SHA256 = \
    "41e00444edf66f72f9e44508c2071e6dd087c0edb7c4a58c0df4b07c59267b15"
S3DIS_ACTUAL_TEST = 542
S3DIS_NOMINAL_TEST = 544
S3DIS_TEST_ROOMS = 68
S3DIS_NON_EIGHT_ROOM = "d092"
S3DIS_REPORT_ID = "S3DIS-QUERY-COUNT-542-vs-544-20261104"

DOMAINS = {
    "npm3d": dict(
        expected_test=116, expected_test_clusters=29, cluster_unit="scene",
        expected_val=128, expected_val_clusters=32,
        historical_marker="PRIOR_TEST_ACCESS_npm3d.json",
        denominator="all 116 released TEST queries; invalid lists count as failures",
        count_provenance=None,
    ),
    "s3dis": dict(
        expected_test=S3DIS_ACTUAL_TEST,
        expected_test_clusters=S3DIS_TEST_ROOMS, cluster_unit="room",
        expected_val=544, expected_val_clusters=68,
        historical_marker="PRIOR_TEST_ACCESS_s3dis.json",
        denominator="all 542 released TEST queries (67 rooms x 8 + d092 x 6); "
                    "invalid/empty lists count as failures",
        count_provenance=dict(
            old_guard=S3DIS_NOMINAL_TEST, effective=S3DIS_ACTUAL_TEST,
            report_id=S3DIS_REPORT_ID,
            raw_manifest_sha256=S3DIS_MANIFEST_SHA256,
            note="544 is nominal 68 rooms x 8; d092 publishes 6 poses. "
                 "No query was filtered, padded or discarded.",
        ),
    ),
}

METHODS = ["original", "presence", "rank_fusion", "fsr", "scar_s0"]
KS_CELL = (1, 3, 5, 10, 25)
KS_LOC = (1, 5, 10, 25)
THRESHS = (1.0, 3.0, 5.0, 10.0)
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 123


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


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


def canonical_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def frozen_common():
    """Load the frozen scar_common.py by path, under its canonical name."""
    path = os.path.join(FROZEN_CODE, "scar_common.py")
    if not os.path.isfile(path):
        raise SystemExit(f"frozen scar_common.py missing: {path}")
    if "scar_common" in sys.modules:
        mod = sys.modules["scar_common"]
        if os.path.realpath(mod.__file__) != os.path.realpath(path):
            raise SystemExit(f"scar_common already loaded from {mod.__file__}")
        return mod
    spec = importlib.util.spec_from_file_location("scar_common", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["scar_common"] = mod
    spec.loader.exec_module(mod)
    return mod


def frozen_module(name):
    """Load one frozen script as a module without executing main()."""
    path = os.path.join(FROZEN_CODE, name + ".py")
    if not os.path.isfile(path):
        raise SystemExit(f"frozen module missing: {path}")
    mod_name = "_frozen_" + name
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def domain_info(domain):
    if domain not in DOMAINS:
        raise SystemExit(f"unknown domain {domain}")
    return DOMAINS[domain]


def freeze_info(path, domain):
    """Verify a frozen FREEZE_LIST; return facts used by gate/eval/verify."""
    S = frozen_common()
    freeze = read_json(path)
    info = dict(path=os.path.abspath(path), file_sha256=sha256(path),
                canonical_sha256=canonical_sha(
                    {k: v for k, v in freeze.items() if k != "freeze_sha256"}),
                freeze_sha256=freeze.get("freeze_sha256"),
                domain=freeze.get("domain"),
                protocol=freeze.get("protocol"))
    if info["canonical_sha256"] != info["freeze_sha256"]:
        raise SystemExit("FREEZE_LIST canonical SHA mismatch")
    if info["domain"] != domain:
        raise SystemExit("FREEZE_LIST domain mismatch")
    if (freeze.get("protocol", {}).get("name") != PROTOCOL_NAME
            or freeze.get("protocol", {}).get("protocol_sha256") is None):
        raise SystemExit("FREEZE_LIST protocol mismatch")
    main = freeze.get("main_head", {})
    if main.get("tag") != "s0" or not main.get("sha256"):
        raise SystemExit("FREEZE_LIST main head is not s0/bound")
    if not os.path.isfile(main["path"]) or sha256(main["path"]) != main["sha256"]:
        raise SystemExit("frozen s0 head missing or SHA mismatch")
    norm = freeze.get("normalisation", {})
    if not os.path.isfile(norm.get("path", "")) \
            or sha256(norm["path"]) != norm.get("sha256"):
        raise SystemExit("frozen normalisation missing or SHA mismatch")
    for tag in ("coarse", "fine"):
        b = freeze.get("backbones", {}).get(tag, {})
        if not os.path.isfile(b.get("path", "")) or sha256(b["path"]) != b.get("sha256"):
            raise SystemExit(f"frozen {tag} backbone missing or SHA mismatch")
    for s in ("train", "val"):
        f = freeze.get("features", {}).get(s, {})
        if not os.path.isfile(f.get("path", "")) or sha256(f["path"]) != f.get("sha256"):
            raise SystemExit(f"frozen {s} feature missing or SHA mismatch")
        if not os.path.isfile(f.get("path", "") + ".meta.json"):
            raise SystemExit(f"frozen {s} feature meta missing")
    info.update(dict(
        main_head=main,
        normalisation=norm,
        backbones={k: freeze["backbones"][k] for k in ("coarse", "fine")},
        num_stats=freeze.get("num_stats"),
        features={s: freeze["features"][s] for s in ("train", "val")},
        constants=freeze["features"]["constants"],
        gate_binding=freeze.get("gate_binding"),
        protocol_sha256=freeze["protocol"]["protocol_sha256"],
    ))
    return freeze, info


def read_test_marker(test_dir):
    path = os.path.join(test_dir, "TEST_READ_ONCE.json")
    if not os.path.isfile(path):
        raise SystemExit(f"TEST marker missing: {path}")
    return read_json(path), path


def check_test_marker(test_dir, protocol_sha, freeze_sha, allow_completed=True,
                     require_clean=False, allowed_extra=()):
    """Validate a new-round marker without ever rewriting the old one.

    On the very first invocation the only entry allowed is the marker.  On a
    resume, chunks/encode/assembly files listed in allowed_extra are allowed.
    """
    m, path = read_test_marker(test_dir)
    if m.get("armed") is not True:
        raise SystemExit("TEST marker is not armed")
    status = m.get("status")
    if status not in ({"armed", "completed"} if allow_completed else {"armed"}):
        raise SystemExit(f"TEST marker status is {status!r}")
    if m.get("protocol_freeze_sha256") != protocol_sha:
        raise SystemExit("TEST marker protocol SHA mismatch")
    if freeze_sha and m.get("freeze_list_sha256") != freeze_sha:
        raise SystemExit("TEST marker freeze-list SHA mismatch")
    if require_clean:
        extra = sorted(set(os.listdir(test_dir))
                       - {"TEST_READ_ONCE.json"} - set(allowed_extra))
        if extra:
            raise SystemExit(f"armed TEST dir is not clean: {extra[:8]}")
    return m, path


def s3dis_count_provenance(report_path, manifest_path):
    """Validate the additive 544 -> 542 evidence before any TEST read."""
    if not os.path.isfile(report_path):
        raise SystemExit(f"S3DIS denominator report missing: {report_path}")
    rep = read_json(report_path)
    if rep.get("id") != S3DIS_REPORT_ID:
        raise SystemExit("S3DIS denominator report id mismatch")
    raw = rep.get("raw_manifest", {})
    if raw.get("sha256") != S3DIS_MANIFEST_SHA256:
        raise SystemExit("S3DIS denominator report manifest SHA mismatch")
    ts = rep.get("test_split", {})
    hist = ts.get("room_pose_histogram", {})
    if (ts.get("scenes") != S3DIS_TEST_ROOMS
            or ts.get("poses_actual") != S3DIS_ACTUAL_TEST
            or ts.get("queries_nominal_8_per_room") != S3DIS_NOMINAL_TEST
            or hist.get("6_poses") != 1 or hist.get("8_poses") != 67):
        raise SystemExit("S3DIS denominator report count facts changed")
    if not os.path.isfile(manifest_path):
        raise SystemExit(f"S3DIS dataset manifest missing: {manifest_path}")
    got = sha256(manifest_path)
    if got != S3DIS_MANIFEST_SHA256:
        raise SystemExit("S3DIS dataset manifest SHA mismatch")
    frozen_path = os.path.join(FROZEN_CODE, "scar_common.py")
    src = open(frozen_path, encoding="utf-8").read()
    old_line = [ln.strip() for ln in src.splitlines()
                if '"test": 544' in ln or "'test': 544" in ln]
    if not old_line:
        raise SystemExit("frozen 544 guard not found; refusing additive override")
    return dict(
        report_path=os.path.abspath(report_path),
        report_sha256=sha256(report_path),
        raw_manifest_path=os.path.abspath(manifest_path),
        raw_manifest_sha256=got,
        old_guard=544, effective=542,
        old_guard_source_line=old_line[0],
        frozen_scar_common_sha256=sha256(frozen_path),
        room_pose_histogram=hist,
        noneight_room=S3DIS_NON_EIGHT_ROOM,
        denominator="all 542 released TEST queries",
        action="in-memory additive override only; frozen bytes and the "
               "historical TEST_READ_ONCE.json are unchanged",
    )


def expected_counts(domain, split, limit=0):
    info = domain_info(domain)
    if split == "test":
        n = info["expected_test"]
        c = info["expected_test_clusters"]
    elif split == "val":
        n = info["expected_val"]
        c = info["expected_val_clusters"]
    else:
        raise SystemExit("only val/test supported here")
    if limit:
        n = min(n, limit)
        c = None
    return n, c


def five_orders(feats, p2, cos, sp, sc_fine, sat_fine, logits):
    """Return the five frozen orderings for one query (all length K)."""
    K = len(p2)
    order_scf = np.argsort(-sc_fine[p2], kind="stable")
    rank_scf = np.empty(K, np.int64)
    rank_scf[p2[order_scf]] = np.arange(K)
    borda = (np.arange(K, dtype=np.float64) + rank_scf[p2]) / (2.0 * (K - 1))
    return {
        "original": np.arange(K),
        "presence": np.argsort(-(cos + 0.5 * sp), kind="stable"),
        "rank_fusion": p2[np.argsort(borda, kind="stable")],
        "fsr": p2[np.argsort(-sat_fine[p2], kind="stable")],
        "scar_s0": p2[np.argsort(-logits[p2], kind="stable")],
    }


def cluster_bootstrap(hit_a, hit_b, keys, resamples=BOOTSTRAP_RESAMPLES,
                      seed=BOOTSTRAP_SEED):
    keys = np.asarray(keys).reshape(-1, 1)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    diff = np.asarray(hit_a, np.float64) - np.asarray(hit_b, np.float64)
    sd = np.bincount(inv, weights=diff, minlength=len(uniq))
    cn = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    rng = np.random.RandomState(seed)
    vals = np.empty(resamples)
    for i in range(resamples):
        pick = rng.randint(0, len(uniq), len(uniq))
        vals[i] = sd[pick].sum() / cn[pick].sum()
    return dict(delta=float(diff.mean()),
                ci95=[float(np.percentile(vals, 2.5, method="linear")),
                      float(np.percentile(vals, 97.5, method="linear"))],
                n_clusters=int(len(uniq)), n_units=int(len(diff)),
                wins=int((diff > 0).sum()), losses=int((diff < 0).sum()),
                ties=int((diff == 0).sum()),
                cluster_sizes=[int(x) for x in cn.tolist()])


def write_cluster_mapping(path, query_ids, scene, cluster_unit):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"q,query_id,cluster,cluster_unit\n")
        for i, (qid, cl) in enumerate(zip(query_ids, scene)):
            f.write(f"{i},{qid},{cl},{cluster_unit}\n")
    return dict(path=os.path.abspath(path), sha256=sha256(path),
                cluster_unit=cluster_unit, unit="query",
                n_rows=len(query_ids),
                n_clusters=len(set(str(x) for x in scene)))
