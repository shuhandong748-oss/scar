#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Arm exactly one exploratory TEST read for C, before any TEST access.

Creates a new TEST directory exclusively and writes TEST_READ_ONCE.json with:
protocol/freeze/backbone/head/normalisation/feature SHAs, the historical TEST
record (read only), the S3DIS 544->542 count provenance when applicable, and
the one-read/s0-only policy.  It reads no dataset and no TEST payload.
"""
import argparse
import os
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c_test_common as T  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=sorted(T.DOMAINS))
    ap.add_argument("--test_dir", default=None)
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--protocol_sha", default=None)
    ap.add_argument("--freeze_sha", default=None)
    ap.add_argument("--prior_record", default=None)
    ap.add_argument("--gate_json", required=True,
                    help="GATE.json from c_gate_val_small.py (VAL-only, PASS)")
    ap.add_argument("--denominator_report", default=None)
    a = ap.parse_args(argv)
    S = T.frozen_common()
    info = T.domain_info(a.domain)
    run = os.path.abspath(a.run_dir or
                          os.path.join("/root/autodl-tmp/d4p_trainval_20261104/C",
                                       a.domain))
    freeze_path = os.path.join(run, "FREEZE_LIST.json")
    if not os.path.isfile(freeze_path):
        raise SystemExit(f"freeze list missing: {freeze_path}")
    freeze, finfo = T.freeze_info(freeze_path, a.domain)
    protocol_sha = a.protocol_sha or finfo["protocol_sha256"]
    freeze_sha = a.freeze_sha or finfo["freeze_sha256"]
    if protocol_sha != finfo["protocol_sha256"]:
        raise SystemExit("protocol SHA argument disagrees with the freeze list")
    if freeze_sha != finfo["freeze_sha256"]:
        raise SystemExit("freeze SHA argument disagrees with the freeze list")

    gate_path = os.path.abspath(a.gate_json)
    if not os.path.isfile(gate_path):
        raise SystemExit(f"VAL gate artifact missing: {gate_path}")
    vgate = T.read_json(gate_path)
    if vgate.get("status") != "PASS" or vgate.get("failures"):
        raise SystemExit("VAL gate is not PASS")
    if vgate.get("domain") != a.domain or vgate.get("split") != "val":
        raise SystemExit("VAL gate domain/split mismatch")
    if vgate.get("test_read") is not False or vgate.get("test_armed") is not False:
        raise SystemExit("VAL gate does not attest test_read=false/test_armed=false")
    vfreeze = (vgate.get("inputs") or {}).get("freeze") or {}
    if vfreeze.get("canonical_sha256") != freeze_sha:
        raise SystemExit("VAL gate freeze SHA does not match this freeze list")
    if (vgate.get("inputs") or {}).get("freeze_file_sha256") != finfo["file_sha256"]:
        raise SystemExit("VAL gate freeze file SHA does not match")
    gate_rec = dict(path=gate_path, sha256=T.sha256(gate_path),
                    status=vgate["status"], domain=vgate["domain"],
                    split=vgate["split"])

    out = os.path.abspath(a.test_dir or
                          os.path.join(T.TEST_ROOT, a.domain, "export", "test"))
    if os.path.exists(out):
        raise SystemExit(f"{out} already exists; a second TEST read is not allowed")

    gate = S.load_gate()
    backbones = S.verify_backbones(a.domain)
    d = S.domain_cfg(a.domain)
    prior_path = os.path.abspath(a.prior_record or
                                 os.path.join(T.TEST_ROOT, "records",
                                              info["historical_marker"]))
    if not os.path.isfile(prior_path):
        raise SystemExit(f"prior TEST access record missing: {prior_path}")
    prior = T.read_json(prior_path)
    if prior.get("domain") != a.domain or prior.get("split") != "test":
        raise SystemExit("prior TEST access record domain/split mismatch")
    hist = d["historical_test_marker"]
    hist_info = dict(path=hist, exists=os.path.isfile(hist),
                     sha256=T.sha256(hist) if os.path.isfile(hist) else None)
    if not hist_info["exists"]:
        raise SystemExit(f"historical TEST marker missing: {hist}")

    count_provenance = None
    if a.domain == "s3dis":
        rep = os.path.abspath(a.denominator_report or
                              os.path.join(T.TEST_ROOT, "records",
                                           "S3DIS_DENOMINATOR_542_544.json"))
        count_provenance = T.s3dis_count_provenance(
            rep, os.path.join(d["base"], "dataset_manifest.json"))
        if count_provenance["old_guard"] != 544 \
                or count_provenance["effective"] != 542:
            raise SystemExit("S3DIS count provenance is not 544->542")

    head_path = freeze["main_head"]["path"]
    norm_path = freeze["normalisation"]["path"]
    code_sha = {
        f: T.sha256(os.path.join(HERE, f))
        for f in ("c_arm_test_v2.py", "c_test_common.py")
    }
    marker = dict(
        armed=True, status="armed", domain=a.domain, split="test",
        test_dir=out,
        protocol_freeze_sha256=protocol_sha,
        freeze_list_sha256=freeze_sha,
        freeze_list_path=freeze_path,
        freeze_file_sha256=finfo["file_sha256"],
        freeze_canonical_sha256=finfo["canonical_sha256"],
        main_head=freeze["main_head"],
        main_head_epoch=freeze["heads"]["s0"].get("formal_epoch"),
        normalisation=freeze["normalisation"],
        backbones={k: freeze["backbones"][k] for k in ("coarse", "fine")},
        num_stats=freeze.get("num_stats"),
        features={s: freeze["features"][s] for s in ("train", "val")},
        feature_definition=freeze["features"].get("definition"),
        constants=freeze["features"]["constants"],
        gate_binding=gate,
        pre_test_gate=gate_rec,
        frozen_code={
            "dir": T.FROZEN_CODE,
            "scar_common_sha256": T.sha256(
                os.path.join(T.FROZEN_CODE, "scar_common.py")),
            "d4p_export_sha256": T.sha256(
                os.path.join(T.FROZEN_CODE, "d4p_export.py")),
            "d4p_features_sha256": T.sha256(
                os.path.join(T.FROZEN_CODE, "d4p_features.py")),
            "d4p_eval_sha256": T.sha256(
                os.path.join(T.FROZEN_CODE, "d4p_eval.py")),
        },
        historical_test_access=hist_info,
        prior_test_access=dict(
            path=prior_path, sha256=T.sha256(prior_path),
            statement=prior.get("historical_accesses", [{}])[0].get("statement"),
        ),
        prior_access_statement=(
            "This project read TEST historically on the old backbone; the "
            "historical marker is preserved unchanged. This round is a second, "
            "exploratory read under the corrected backbone, not a first blind "
            "test and not a confirmatory experiment."),
        count_provenance=count_provenance,
        expected_test=dict(
            queries=info["expected_test"],
            clusters=info["expected_test_clusters"],
            denominator=info["denominator"],
            count_source=("additive 542 provenance" if count_provenance
                          else "frozen registered count"),
        ),
        policy=dict(
            exports=1, shared_by_all_methods=True, only_main_head="s0",
            methods=list(T.METHODS), s1_s2="not evaluated",
            coordinates="SCAR reorders only; coordinates are each candidate's "
                        "existing Fine prediction",
            no_retry="no automatic re-export, re-selection or retraining after "
                     "this read",
            no_denominator_change="all released queries; invalid/empty lists "
                                  "count as failures",
            bootstrap="paired cluster bootstrap, 2000 resamples, seed 123, "
                      f"{info['cluster_unit']} clusters",
            chunking="query-level chunks with SHA sidecars; valid chunks are "
                     "reused and not recomputed",
        ),
        host=socket.gethostname(),
        created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        created_by=code_sha,
    )
    # Create the directory only after every input check has passed, so a failed
    # arm never leaves a half-created TEST directory that blocks a clean retry.
    if os.path.exists(out):
        raise SystemExit(f"{out} appeared during validation; refusing to reuse")
    os.makedirs(out, exist_ok=False)
    T.write_json(marker, os.path.join(out, "TEST_READ_ONCE.json"))
    print("armed", out, flush=True)
    print("freeze", freeze_sha, "protocol", protocol_sha, flush=True)
    print("head", head_path, T.sha256(head_path), flush=True)
    print("norm", norm_path, T.sha256(norm_path), flush=True)
    print("histor", hist, hist_info["sha256"], flush=True)
    if count_provenance:
        print("s3dis count 544 -> 542",
              count_provenance["raw_manifest_sha256"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
