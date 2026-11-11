#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Arm the single exploratory TEST read for one domain, before any TEST access.

Creates the output directory exclusively and writes an armed marker that binds
the protocol and freeze-list SHAs.  The export refuses to touch TEST unless
this marker exists, so the order "freeze -> arm -> read" is enforced by code.
This script itself reads no dataset.
"""
import argparse
import hashlib
import json
import os
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scar_common as S  # noqa: E402


def canonical_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["npm3d", "s3dis"])
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--protocol_sha", required=True)
    ap.add_argument("--freeze_sha", required=True)
    ap.add_argument("--test_dir", default=None)
    a = ap.parse_args(argv)
    run = os.path.abspath(a.run_dir)
    d = S.domain_cfg(a.domain)
    freeze_path = os.path.join(run, "FREEZE_LIST.json")
    if not os.path.isfile(freeze_path):
        raise SystemExit("freeze list missing; freeze before arming TEST")
    freeze = S.read_json(freeze_path)
    if freeze.get("freeze_sha256") != a.freeze_sha:
        raise SystemExit("freeze sha mismatch with FREEZE_LIST.json")
    if freeze["protocol"]["protocol_sha256"] != a.protocol_sha:
        raise SystemExit("protocol sha mismatch with the freeze list")
    if freeze["domain"] != a.domain:
        raise SystemExit("freeze list belongs to another domain")
    if canonical_sha({k: v for k, v in freeze.items() if k != "freeze_sha256"}) \
            != a.freeze_sha:
        raise SystemExit("freeze list content drifted from its sha")

    out = os.path.abspath(a.test_dir or os.path.join(run, "test"))
    if os.path.exists(out):
        raise SystemExit(f"{out} already exists; a second TEST read is not allowed")
    os.makedirs(out, exist_ok=False)

    hist = d["historical_test_marker"]
    hist_info = dict(path=hist, exists=os.path.isfile(hist),
                     sha256=S.sha256(hist) if os.path.isfile(hist) else None)
    if not hist_info["exists"]:
        raise SystemExit(f"historical TEST marker not found: {hist}; refusing to "
                         f"proceed without the prior-access record")
    marker = dict(
        armed=True, status="armed", domain=a.domain, split="test",
        test_dir=out,
        protocol_freeze_sha256=a.protocol_sha,
        freeze_list_sha256=a.freeze_sha,
        freeze_list_path=freeze_path,
        main_head=freeze["main_head"],
        backbones={k: freeze["backbones"][k] for k in ("coarse", "fine")},
        gate_binding=freeze["gate_binding"],
        historical_test_access=hist_info,
        prior_access_statement=(
            "This project already read the TEST split historically on the old "
            "backbone (see historical_test_access). This second read is an "
            "exploratory read under the corrected backbone; it is NOT a first "
            "project-level TEST and NOT an independent confirmatory experiment."),
        policy=dict(
            exports=1, shared_by_all_methods=True,
            methods=["original", "Presence", "Rank Fusion", "FSR", "SCAR s0"],
            coordinates="SCAR only reorders; coordinates are each candidate's "
                        "existing Fine prediction",
            no_retry="no automatic re-export, re-selection or retraining after "
                     "this read",
            bootstrap="paired cluster bootstrap, 2000 resamples, seed 123, "
                      f"{d['cluster_unit']} clusters"),
        host=socket.gethostname(),
        created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        created_by=S.sha256(os.path.join(HERE, "d4p_arm_test.py")),
    )
    S.write_json(marker, os.path.join(out, "TEST_READ_ONCE.json"))
    print(json.dumps({k: marker[k] for k in
                      ("status", "domain", "test_dir", "freeze_list_sha256")},
                     indent=2), flush=True)
    print("armed", out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
