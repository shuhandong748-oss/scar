#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for the A-domain corrected-backbone TEST round.

Scope: PandaSet and nuScenesFull, FEATURE_COORD_FIX_v1, one exploratory TEST
export per domain, five rankings evaluated from the same cache (Original,
Presence, Rank Fusion, FSR, SCAR s0).  None of the frozen files is modified;
the feature fix is applied in-process by the frozen wrapper mechanism.

No function here opens a TEST file.  The arm/export scripts enforce marker ->
read ordering.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

FEATURE_VERSION = "FEATURE_COORD_FIX_v1"
PROTOCOL = "CORR-D4P-SCAR-FORMAL-v1"
KS_CELL = (1, 3, 5, 10, 25)
KS_LOC = (1, 5, 10, 25)
THRESHS = (1.0, 3.0, 5.0, 10.0)
SEED = 123
RESAMPLES = 2000
PRIMARY_CELL_K = 1
PRIMARY_LOC_K = 1
PRIMARY_THRESH = 5.0

A_CORR = "/root/autodl-tmp/des4pos_corrected_A_20261102"
A_TRAINVAL = "/root/autodl-tmp/d4p_trainval_20261104/A"
A_TEST = "/root/autodl-tmp/d4p_test_20261104/A"

DOMAINS = {
    "pandaset": {
        "code": "/root/autodl-tmp/text2loc_pandaset/code",
        "extra_paths": [
            "/root/autodl-tmp/Des4Pos/reranker",
            "/root/autodl-tmp/Des4Pos/c_line",
            "/root/autodl-tmp/baseB",
            "/root/autodl-tmp/text2loc_pandaset/vendor",
        ],
        "builder": "/root/autodl-tmp/text2loc_pandaset/scripts/build_ht_t2l.py",
        "vendor": "/root/autodl-tmp/text2loc_pandaset/vendor",
        "feature_paths": ["/root/autodl-tmp/baseB"],
        "env": {},
        "ckpt_dir": os.path.join(A_CORR, "pandaset_s0", "ckpt",
                                 "PandaSetPose-50-25-25"),
        "prior": "/root/autodl-tmp/des4pos_pandaset_v2/runs/"
                 "pandaset_bb0_seed0/ste_stage1/ste_stage1_prior.pth",
        "run_dir": os.path.join(A_TRAINVAL, "pandaset"),
        "expected_test": 2000,
        "expected_test_clusters": 25,
        "cluster_unit": "scene",
        "historical_access": os.path.join(
            A_TEST, "records", "PRIOR_TEST_ACCESS_pandaset.json"),
        "historical_statement": (
            "2026-10-27 L0 structural audit only: all 25 TEST scenes / 2000 poses "
            "/ 12000 descriptions were inspected structurally; no model forward, "
            "no candidate export, no metrics. L1-L4 never occurred. The corrected "
            "backbone has never read TEST."),
    },
    "nuscenesfull": {
        "code": "/root/autodl-tmp/text2loc_nuscenes/code",
        "extra_paths": ["/root/autodl-tmp/text2loc_nuscenes/vendor"],
        "builder": "/root/autodl-tmp/text2loc_nuscenes/scripts/"
                   "build_ht_t2l_nuscenes.py",
        "vendor": "/root/autodl-tmp/text2loc_nuscenes/vendor",
        "feature_paths": [],
        "env": {
            "T2L_NUSC_ROOT":
                "/root/autodl-tmp/text2loc_nuscenes/data/"
                "nuScenesPose-50-25-25_30-10",
        },
        "ckpt_dir": os.path.join(A_CORR, "nuscenesfull_s0", "ckpt",
                                 "nuScenesPose-50-25-25_30-10"),
        "prior": "/root/autodl-tmp/des4pos_nuscenesfull_v2/runs/"
                 "nusc_full_bb0_seed0/ste_stage1/ste_stage1_prior.pth",
        "run_dir": os.path.join(A_TRAINVAL, "nuscenesfull"),
        "expected_test": 7403,
        # 210 = TEST scenes that actually carry queries.  The split nominally
        # covers 213 scenes (150 official val + 63 train), but three have none
        # in the processed release; every accepted nuScenesFull TEST run uses
        # 210 query scene clusters (e.g. text2loc_five_datasets_20261026
        # metrics_test.json invariants: n_queries 7403, n_scene_clusters 210).
        "expected_test_clusters": 210,
        "cluster_unit": "scene",
        "historical_access": os.path.join(
            A_TEST, "records", "PRIOR_TEST_ACCESS_nuscenesfull.json"),
        "historical_statement": (
            "2026-10-29 the v2 (pre-correction) backbone performed one TEST export "
            "and one TEST evaluation (7403 queries). That access belongs to a "
            "different backbone. The corrected backbone has never read TEST. "
            "This round is an exploratory second project-level read, not a first "
            "blind test and not confirmatory."),
    },
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, path):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def canonical_sha(obj):
    return sha256_bytes(json.dumps(obj, sort_keys=True,
                                   separators=(",", ":")).encode())


def domain_cfg(domain):
    if domain not in DOMAINS:
        raise SystemExit(f"unknown A domain {domain!r}; have {sorted(DOMAINS)}")
    return dict(DOMAINS[domain])


def install_paths(domain):
    d = domain_cfg(domain)
    for k, v in d["env"].items():
        os.environ[k] = v
    if d["code"] not in sys.path:
        sys.path.insert(0, d["code"])
    for p in d["extra_paths"]:
        if p not in sys.path:
            sys.path.insert(0, p)


def verify_freeze(freeze_path, domain, expected_epochs):
    """Verify the canonical freeze hash, file hash and the fixed bindings."""
    if not os.path.isfile(freeze_path):
        raise SystemExit(f"freeze list missing: {freeze_path}")
    f = read_json(freeze_path)
    if f.get("domain") != domain:
        raise SystemExit(f"freeze list domain {f.get('domain')} != {domain}")
    canon = canonical_sha({k: v for k, v in f.items() if k != "freeze_sha256"})
    if canon != f.get("freeze_sha256"):
        raise SystemExit("freeze canonical content hash drifted")
    if f["protocol"]["name"] != PROTOCOL:
        raise SystemExit(f"unexpected protocol {f['protocol']['name']}")
    if f["protocol"].get("feature_version") != FEATURE_VERSION:
        raise SystemExit(f"unexpected feature version "
                         f"{f['protocol'].get('feature_version')}")
    bt = f["backbones"]["train"]
    checks = {}
    for stage in ("coarse", "fine"):
        rec = bt[stage]
        got = sha256(rec["path"])
        want = rec["sha256"]
        epoch = rec["bind"]["BEST"]["epoch"]
        if got != want:
            raise SystemExit(f"{domain}/{stage}: SHA {got} != freeze {want}")
        if epoch != expected_epochs[stage]:
            raise SystemExit(f"{domain}/{stage}: epoch {epoch} != "
                             f"{expected_epochs[stage]}")
        if rec["bind"]["BEST"].get("light_sha256") != want:
            raise SystemExit(f"{domain}/{stage}: BEST light sha mismatch")
        checks[stage] = dict(path=rec["path"], sha256=got, epoch=epoch,
                             BEST_path=rec["bind"]["BEST_path"],
                             BEST_sha256=rec["bind"]["BEST_sha256"],
                             light_state_digest=rec["bind"]["light_state_digest"],
                             light_n_tensors=rec["bind"]["light_n_tensors"])
    ste = bt["ste_stage1"]
    if sha256(ste["path"]) != ste["sha256"]:
        raise SystemExit(f"{domain}/STE SHA mismatch")
    checks["ste_stage1"] = dict(path=ste["path"], sha256=ste["sha256"])
    chk = {"PASS": True, "freeze_path": os.path.abspath(freeze_path),
           "freeze_file_sha256": sha256(freeze_path),
           "freeze_canonical_sha256": canon,
           "protocol_sha256": f["protocol"]["protocol_sha256"],
           "feature_version": f["protocol"]["feature_version"],
           "backbones": checks}
    if f.get("coord_fix"):
        chk["coord_fix"] = f["coord_fix"]
    return f, chk


def verify_heads_and_norm(freeze, freeze_path):
    """s0 head and NORMALIZATION must be the frozen bytes."""
    d = os.path.dirname(os.path.abspath(freeze_path))
    hd = freeze["main_head"]
    hpath = hd["path"] if os.path.isabs(hd["path"]) else os.path.join(d, hd["path"])
    if not os.path.isfile(hpath):
        alt = os.path.join(d, "heads", f"ht23_{freeze['domain']}_s0.pt")
        hpath = alt
    if sha256(hpath) != hd["sha256"]:
        raise SystemExit(f"main head s0 SHA mismatch: {hpath}")
    nd = freeze["normalisation"]
    npath = nd["path"]
    if not os.path.isfile(npath):
        npath = os.path.join(d, "heads", "NORMALIZATION.json")
    if sha256(npath) != nd["sha256"]:
        raise SystemExit(f"NORMALIZATION SHA mismatch: {npath}")
    return dict(head=dict(path=hpath, sha256=sha256(hpath),
                          formal_epoch=(freeze.get("heads", {})
                                        .get("s0", {}).get("formal_epoch")),
                          tag=hd.get("tag", "s0")),
                normalisation=dict(path=npath, sha256=sha256(npath),
                                   pinned=nd.get("pinned"),
                                   fitted_on=nd.get("fitted_on")))


def input_record(domain, freeze, freeze_path, config, known):
    h = verify_heads_and_norm(freeze, freeze_path)
    out = dict(
        domain=domain, protocol=PROTOCOL, feature_version=FEATURE_VERSION,
        freeze=dict(path=os.path.abspath(freeze_path),
                    file_sha256=sha256(freeze_path),
                    canonical_sha256=freeze["freeze_sha256"],
                    protocol_sha256=freeze["protocol"]["protocol_sha256"]),
        backbones={k: dict(path=v["path"], sha256=v["sha256"])
                   for k, v in freeze["backbones"]["train"].items()},
        epochs={s: freeze["backbones"]["train"][s]["bind"]["BEST"]["epoch"]
                for s in ("coarse", "fine")},
        main_head=h["head"], normalisation=h["normalisation"],
        config=dict(path=os.path.abspath(config), sha256=sha256(config)),
        known_classes=dict(path=os.path.abspath(known), sha256=sha256(known)),
        features_reference={
            s: dict(path=freeze["features"][s].get("path"),
                    sha256=freeze["features"][s].get("sha256"))
            for s in ("train", "val") if s in freeze.get("features", {})},
        expected_test_queries=DOMAINS[domain]["expected_test"],
        expected_test_clusters=DOMAINS[domain]["expected_test_clusters"],
        cluster_unit=DOMAINS[domain]["cluster_unit"],
        historical_access=dict(
            path=DOMAINS[domain]["historical_access"],
            exists=os.path.isfile(DOMAINS[domain]["historical_access"]),
            statement=DOMAINS[domain]["historical_statement"]),
    )
    code_dir = os.path.dirname(os.path.abspath(__file__))
    out["code_sha256"] = {}
    for n in sorted(os.listdir(code_dir)):
        if n.endswith(".py"):
            out["code_sha256"][n] = sha256(os.path.join(code_dir, n))
    # The in-process patch module is part of the frozen A round; bind its bytes.
    patch = os.path.join(os.path.dirname(A_TRAINVAL), "A_round", "code",
                         "feature_coord_fix_v1.py")
    if os.path.isfile(patch):
        out["feature_coord_fix_v1"] = dict(path=patch, sha256=sha256(patch))
    return out


def require_marker(marker_path, domain, stage, freeze_sha=None,
                   proc_sha=None, allow_completed=True):
    """stage='armed' before any TEST access; 'completed' before eval."""
    if not os.path.isfile(marker_path):
        raise SystemExit(f"TEST marker missing: {marker_path}")
    m = read_json(marker_path)
    if m.get("domain") != domain or m.get("split") != "test":
        raise SystemExit("TEST marker domain/split mismatch")
    status = m.get("status")
    if stage == "armed":
        if m.get("armed") is not True or status not in (
                "armed", "exported", "completed"):
            raise SystemExit(f"TEST marker is not armed (status={status})")
    elif stage == "completed":
        if status != "completed" or m.get("armed") is not True:
            raise SystemExit(f"TEST export not completed (status={status})")
    if freeze_sha and m.get("freeze_list_sha256") != freeze_sha:
        raise SystemExit("TEST marker freeze-list SHA mismatch")
    if proc_sha and m.get("protocol_sha256") != proc_sha:
        raise SystemExit("TEST marker protocol SHA mismatch")
    return m


def update_marker(marker_path, **fields):
    m = read_json(marker_path)
    m.update(fields)
    write_json(m, marker_path)
    return m


class Head:
    """s0 scorer only; s1/s2 are explicitly out of scope."""

    def __init__(self, path, expected_sha=None):
        import torch
        import torch.nn as nn
        if expected_sha and sha256(path) != expected_sha:
            raise SystemExit(f"head SHA mismatch: {path}")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.path = path
        self.sha256 = sha256(path)
        self.mean = ck["mean"].astype(np.float32)
        self.std = ck["std"].astype(np.float32)
        self.selection = ck["selection"]
        self.feature_version = ck.get("feature_version")
        self.feature_names = list(ck["feature_names"])
        self.nf = len(self.feature_names)
        K = ck["config"]["k"]
        norms = ck["config"]["norms"]
        if self.nf != 23 or K != 25:
            raise SystemExit(f"{path}: not canonical-23/K25")
        if norms != dict(pool_norm=24, rank_norm=24, tied_norm=25):
            raise SystemExit(f"{path}: unexpected norms {norms}")
        if self.feature_version != FEATURE_VERSION:
            raise SystemExit(f"{path}: feature version {self.feature_version!r}")

        class Net(nn.Module):
            def __init__(self, nf):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(nf, 64), nn.ReLU(),
                                         nn.Linear(64, 32), nn.ReLU(),
                                         nn.Linear(32, 1))

            def forward(self, z):
                return self.net(z).squeeze(-1)

        self.torch = torch
        self.model = Net(self.nf)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def logits(self, feats):
        x = (feats - self.mean) / self.std
        with self.torch.no_grad():
            return self.model(self.torch.from_numpy(x.astype(np.float32))).numpy()


def support_audit(feats, feature_names, n_q, min_match, valid=None):
    """Derive per-candidate matched counts from the frozen n_matched column."""
    names = list(feature_names)
    if "n_matched" not in names:
        raise SystemExit("feature matrix has no n_matched column")
    j = names.index("n_matched")
    frac = np.asarray(feats[:, :, j], np.float64)
    m = np.rint(frac * float(n_q)).astype(np.int64)
    gate = m >= int(min_match)
    out = dict(
        n_queries=int(len(feats)), n_mentioned=int(n_q), min_match=int(min_match),
        m_definition="n_matched * n_mentioned (rounded)",
        m_min=int(m.min()), m_max=int(m.max()),
        per_query_m_min=[int(x) for x in m.min(1)],
        per_query_m_max=[int(x) for x in m.max(1)],
        candidates_meeting_gate=int(gate.sum()),
        candidates_total=int(gate.size),
        queries_with_any_m_ge_gate=int((gate.any(1)).sum()),
        frac_candidates_meeting_gate=float(gate.mean()),
    )
    if valid is not None:
        out["valid_lists"] = int(np.asarray(valid).sum())
        out["invalid_lists"] = int(len(valid) - np.asarray(valid).sum())
    return out


def rank_by_logits(logits, p2):
    """The frozen ranker: stable descending logits inside the P2 pool order."""
    order = np.argsort(-np.asarray(logits)[p2], kind="stable")
    return np.asarray(p2)[order]


def bootstrap_ci(hit_a, hit_b, keys, resamples=RESAMPLES, seed=SEED):
    """Paired cluster bootstrap, ratio-of-sums, linear 95% percentile."""
    keys = np.asarray(keys)
    if keys.ndim == 1:
        keys = keys.reshape(-1, 1)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    n_cl = len(uniq)
    diff = np.asarray(hit_a, np.float64) - np.asarray(hit_b, np.float64)
    sum_d = np.bincount(inv, weights=diff, minlength=n_cl)
    cnt = np.bincount(inv, minlength=n_cl).astype(np.float64)
    rng = np.random.RandomState(seed)
    vals = np.empty(resamples, np.float64)
    for s in range(resamples):
        pick = rng.randint(0, n_cl, n_cl)
        vals[s] = sum_d[pick].sum() / cnt[pick].sum()
    lo, hi = np.percentile(vals, [2.5, 97.5], method="linear")
    return dict(
        delta=float(diff.mean()),
        ci95=[float(lo), float(hi)],
        n_clusters=int(n_cl), n_units=int(len(diff)),
        wins=int((diff > 0).sum()), losses=int((diff < 0).sum()),
        ties=int((diff == 0).sum()),
        cluster_sizes=[int(x) for x in cnt.tolist()],
        cluster_keys_sha256=sha256_bytes(uniq.tobytes()),
        resamples=int(resamples), seed=int(seed),
        rng="numpy.random.RandomState", percentile="linear",
        weighting="ratio-of-sums within resampled clusters",
        pairing="paired query/unit differences",
        small_cluster_warning=bool(n_cl < 30),
    )


def cell_hits(ids_row, order, gt_cell, ks=KS_CELL):
    ranked = [str(ids_row[j]) for j in order]
    return {int(k): int(str(gt_cell) in ranked[:int(k)]) for k in ks}


def loc_hits(pred_w_row, order, gt_pose, cand_scene_row, query_scene,
             ks=KS_LOC, thresholds=THRESHS):
    order = np.asarray(order, np.int64)
    pw = np.asarray(pred_w_row, np.float64)[order]
    d = np.linalg.norm(pw - np.asarray(gt_pose, np.float64)[None, :], axis=1)
    cs = np.asarray([str(x) for x in cand_scene_row])[order]
    d[cs != str(query_scene)] = np.inf
    out = {}
    for k in ks:
        for t in thresholds:
            out[(int(k), float(t))] = int(np.min(d[:int(k)]) <= float(t))
    return out, d


def metrics_from_hits(hit_cell, hit_loc, methods, n):
    res_cell = {m: {f"R@{k}": float(np.asarray(hit_cell[m][k]).mean())
                    for k in KS_CELL} for m in methods}
    res_loc = {m: {f"Top{k}@{int(t)}m":
                   float(np.asarray(hit_loc[m][(k, t)]).mean())
                   for k in KS_LOC for t in THRESHS} for m in methods}
    return res_cell, res_loc


def check_monotonicity(res_cell, res_loc, methods):
    bad = []
    for m in methods:
        for a, b in zip(KS_CELL[:-1], KS_CELL[1:]):
            if res_cell[m][f"R@{a}"] > res_cell[m][f"R@{b}"] + 1e-12:
                bad.append(f"{m}: R@{a} > R@{b}")
        for t1, t2 in ((1.0, 3.0), (3.0, 5.0), (5.0, 10.0)):
            if res_loc[m][f"Top1@{int(t1)}m"] > \
                    res_loc[m][f"Top1@{int(t2)}m"] + 1e-12:
                bad.append(f"{m}: Top1@{int(t1)}m > Top1@{int(t2)}m")
        for a, b in zip(KS_LOC[:-1], KS_LOC[1:]):
            if res_loc[m][f"Top{a}@5m"] > res_loc[m][f"Top{b}@5m"] + 1e-12:
                bad.append(f"{m}: Top{a}@5m > Top{b}@5m")
    return bad


def file_table(rows, path, header, fmt="{:.6f}"):
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
