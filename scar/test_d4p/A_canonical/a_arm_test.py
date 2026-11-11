#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Arm the single exploratory corrected-backbone TEST read for one A domain.

Runs BEFORE any TEST access.  It reads only freeze/checkpoint/head/normalisation
records, creates the marker directory exclusively and writes TEST_READ_ONCE.json
plus ARM_INPUTS.json.  Refuses if a marker already exists, if the freeze says
TEST was read, or if any fixed SHA/epoch differs.
"""
import argparse
import os
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a_test_common as S  # noqa: E402

EXPECTED_EPOCHS = {
    "pandaset": {"coarse": 8, "fine": 35},
    "nuscenesfull": {"coarse": 17, "fine": 30},
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=sorted(S.DOMAINS))
    ap.add_argument("--freeze_list", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--known_classes", required=True)
    ap.add_argument("--marker_dir", required=True)
    ap.add_argument("--gate_json", required=True,
                    help="GATE.json from a_gate_val_small.py (VAL-only, PASS)")
    ap.add_argument("--historical_access", default=None)
    ap.add_argument("--purpose", default=(
        "新增探索性 TEST：corrected 主干在已见项目 TEST 上的一次导出，"
        "五种排序同池比较；非首次盲测、非确认性实验"))
    a = ap.parse_args(argv)
    freeze_path = os.path.abspath(a.freeze_list)
    marker_dir = os.path.abspath(a.marker_dir)

    freeze, fchk = S.verify_freeze(freeze_path, a.domain, EXPECTED_EPOCHS[a.domain])
    if freeze.get("scope", {}).get("test_read") is not False:
        raise SystemExit("freeze scope does not state test_read=false")
    if freeze.get("scope", {}).get("test_armed") is not False:
        raise SystemExit("freeze scope does not state test_armed=false")
    gate_path = os.path.abspath(a.gate_json)
    if not os.path.isfile(gate_path):
        raise SystemExit(f"VAL gate artifact missing: {gate_path}")
    gate = S.read_json(gate_path)
    if gate.get("status") != "PASS" or gate.get("failures"):
        raise SystemExit("VAL gate is not PASS")
    if gate.get("domain") != a.domain or gate.get("split") != "val":
        raise SystemExit("VAL gate domain/split mismatch")
    if gate.get("test_read") is not False or gate.get("test_armed") is not False:
        raise SystemExit("VAL gate does not attest test_read=false/test_armed=false")
    gfreeze = (gate.get("input") or {}).get("freeze") or {}
    if gfreeze.get("canonical_sha256") != freeze["freeze_sha256"]:
        raise SystemExit("VAL gate freeze SHA does not match this freeze list")
    gate_rec = dict(path=gate_path, sha256=S.sha256(gate_path),
                    status=gate["status"], domain=gate["domain"],
                    split=gate["split"])
    for p, what in ((a.config, "config"), (a.known_classes, "known_classes")):
        if not os.path.isfile(p):
            raise SystemExit(f"{what} missing: {p}")
    inputs = S.input_record(a.domain, freeze, freeze_path, a.config,
                            a.known_classes)
    hist_path = a.historical_access or inputs["historical_access"]["path"]
    hist_exists = os.path.isfile(hist_path)
    if not hist_exists:
        raise SystemExit(f"prior TEST access record missing: {hist_path}")
    try:
        hist_obj = S.read_json(hist_path)
    except Exception:
        hist_obj = None
    hist_rec = dict(path=hist_path, sha256=S.sha256(hist_path),
                    exists=True, content=hist_obj)
    inputs["historical_access"] = hist_rec

    if os.path.exists(marker_dir):
        raise SystemExit(f"marker dir already exists: {marker_dir}; a second "
                         f"TEST read is not allowed")
    os.makedirs(marker_dir, exist_ok=False)
    marker_path = os.path.join(marker_dir, "TEST_READ_ONCE.json")
    if os.path.exists(marker_path):
        raise SystemExit("marker race: TEST_READ_ONCE.json already exists")

    marker = dict(
        armed=True, status="armed", split="test", domain=a.domain,
        purpose=a.purpose,
        test_is_seen_not_blind=True,
        exploratory_not_confirmatory=True,
        marker_dir=marker_dir,
        freeze_list=dict(path=freeze_path,
                         file_sha256=inputs["freeze"]["file_sha256"],
                         canonical_sha256=freeze["freeze_sha256"]),
        freeze_list_sha256=freeze["freeze_sha256"],
        protocol=dict(name=S.PROTOCOL,
                      protocol_sha256=freeze["protocol"]["protocol_sha256"],
                      feature_version=S.FEATURE_VERSION),
        protocol_sha256=freeze["protocol"]["protocol_sha256"],
        backbones=inputs["backbones"],
        epochs=inputs["epochs"],
        main_head=inputs["main_head"],
        normalisation=inputs["normalisation"],
        config=inputs["config"],
        known_classes=inputs["known_classes"],
        expected=dict(n_queries=S.DOMAINS[a.domain]["expected_test"],
                      n_clusters=S.DOMAINS[a.domain]["expected_test_clusters"],
                      cluster_unit=S.DOMAINS[a.domain]["cluster_unit"],
                      source="raw release manifest + L0/v2 prior-access records"),
        denominator="all TEST queries; invalid/empty candidate lists count as failures",
        methods=["Original", "Presence", "Rank Fusion", "FSR", "SCAR s0"],
        policy=dict(
            exports=1, retrain=False, reselect=False, s1_s2=False,
            change_thresholds=False, change_features=False,
            shared_cache=True, k=25,
            retry="no automatic re-export; failed chunks are preserved and "
                  "recomputation needs an explicit reason",
            bootstrap="paired cluster bootstrap, 2000 resamples, seed 123, "
                      "RandomState, ratio-of-sums, linear 95% percentile",
            efficiency=False, kitti_test=False, auto_shutdown=False,
            subsequent_tasks=False,
        ),
        input_record_sha256=None,
        pre_test_gate=gate_rec,
        prior_test_access=hist_rec,
        host=socket.gethostname(),
        created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        created_by=S.sha256(os.path.join(HERE, "a_arm_test.py")),
        common_sha256=S.sha256(os.path.join(HERE, "a_test_common.py")),
    )
    S.write_json(inputs, os.path.join(marker_dir, "ARM_INPUTS.json"))
    marker["input_record_sha256"] = S.sha256(
        os.path.join(marker_dir, "ARM_INPUTS.json"))
    S.write_json(marker, marker_path)
    print(S.json.dumps({k: marker[k] for k in
                        ("status", "domain", "marker_dir", "created")},
                       indent=2, ensure_ascii=False))
    print("ARMED", marker_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
