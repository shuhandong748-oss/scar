#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-domain TRAIN/VAL report.  Reads only TRAIN/VAL evidence; never TEST.

Writes REPORT.json next to the run directory's evidence.  Every number is
copied from a frozen artefact whose SHA is recorded, so the report can be
rebuilt and compared byte-for-byte.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scar_common as S  # noqa: E402


def maybe(path):
    return S.read_json(path) if os.path.isfile(path) else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["npm3d", "s3dis"])
    ap.add_argument("--run_dir", required=True)
    a = ap.parse_args(argv)
    run = os.path.abspath(a.run_dir)
    d = S.domain_cfg(a.domain)

    status = maybe(os.path.join(run, "DOMAIN_STATUS.json"))
    freeze = maybe(os.path.join(run, "FREEZE_LIST.json"))
    sel = maybe(os.path.join(run, "heads", "SELECTION_FORMAL.json"))
    ev = maybe(os.path.join(run, "eval_val", "metrics.json"))
    vx_tr = maybe(os.path.join(run, "verify", "VERIFY_EXPORT_train.json"))
    vx_va = maybe(os.path.join(run, "verify", "VERIFY_EXPORT_val.json"))
    ve_va = maybe(os.path.join(run, "verify", "VERIFY_EVAL_val.json"))
    vs_va = maybe(os.path.join(run, "verify", "VERIFY_SELECTION_val.json"))
    ft = maybe(os.path.join(run, "features", "train.npz.meta.json"))
    fv = maybe(os.path.join(run, "features", "val.npz.meta.json"))
    policy = maybe(os.path.join(run, "NO_TEST_POLICY.json"))

    missing = [name for name, obj in (
        ("DOMAIN_STATUS", status), ("FREEZE_LIST", freeze),
        ("SELECTION_FORMAL", sel), ("eval_val/metrics", ev),
        ("VERIFY_EXPORT_train", vx_tr), ("VERIFY_EXPORT_val", vx_va),
        ("VERIFY_EVAL_val", ve_va), ("features/train.meta", ft),
        ("features/val.meta", fv)) if obj is None]

    main = {}
    if sel:
        s0 = sel["seeds"]["s0"]
        main = dict(tag="s0", epoch=s0["formal_epoch"], head_sha256=s0["sha256"],
                    val_cell=s0["formal_val_cell"], val_loc=s0["formal_val_loc"],
                    legacy_a_epoch=s0["legacy_a_epoch"],
                    selection_rule=sel["formal_rule"])

    results = {}
    fsr = {}
    if ev:
        results = {"cell": ev["results_cell"], "loc": ev["results_loc"]}
        c = ev["comparisons"].get("SCAR_minus_FSR", {})
        fsr = {k: dict(delta=round(v["delta"] * 100, 2),
                       ci95=[round(x * 100, 2) for x in v["ci95"]],
                       n_clusters=v["n_clusters"])
               for k, v in c.items()}

    constants = {}
    if ft:
        constants = dict(relation_channel=ft["relation_channel"],
                         valid_lists=ft["valid_lists"],
                         positives_per_list=ft["positives_per_list"],
                         features_fix_sha256=ft["features_fix_sha256"],
                         sa_geom_sha256=ft["sa_geom_sha256"],
                         common_sha256=ft["common_sha256"])
    if fv:
        constants["val_valid_lists"] = fv["valid_lists"]
        constants["val_positives_per_list"] = fv["positives_per_list"]

    verification = {
        "export_train": (vx_tr or {}).get("status"),
        "export_val": (vx_va or {}).get("status"),
        "eval_val": (ve_va or {}).get("status"),
        "selection_val": (vs_va or {}).get("status"),
        "probe_prefix_identical": ((vx_va or {}).get("checks", {})
                                   .get("probe_prefix_identical", {}).get("PASS")),
    }

    report = dict(
        domain=a.domain,
        round="TRAIN/VAL only",
        status=status,
        missing=missing,
        main_head=main,
        val_results=results,
        scar_minus_fsr=fsr,
        feature_constants=constants,
        verification=verification,
        protocol_freeze_sha256=(freeze or {}).get("protocol", {}).get("protocol_sha256"),
        freeze_sha256=(freeze or {}).get("freeze_sha256"),
        test_read=False,
        test_armed=False,
        test_evidence=(policy or {}),
        artifacts={
            "freeze_list": os.path.join(run, "FREEZE_LIST.json"),
            "freeze_list_sha256": (S.sha256(os.path.join(run, "FREEZE_LIST.json"))
                                   if os.path.isfile(os.path.join(run,
                                                                  "FREEZE_LIST.json"))
                                   else None),
            "selection": os.path.join(run, "heads", "SELECTION_FORMAL.json"),
            "eval_val": os.path.join(run, "eval_val", "metrics.json"),
            "features_train": os.path.join(run, "features", "train.npz"),
            "features_val": os.path.join(run, "features", "val.npz"),
            "exports": {s: os.path.join(run, "export", s) for s in ("train", "val")},
        },
        code_sha256={f: S.sha256(os.path.join(HERE, f))
                     for f in ("d4p_report_trainval.py", "scar_common.py")},
    )
    out = os.path.join(run, "REPORT.json")
    S.write_json(report, out)
    print(json.dumps({k: report[k] for k in
                      ("domain", "missing", "main_head", "verification",
                       "test_read", "test_armed")}, indent=2), flush=True)
    print("saved", out, "sha", S.sha256(out), flush=True)
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
