#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the frozen canonical-23 feature builder for a C TEST export.

The frozen d4p_features.py is imported and its main() is called in-process.
No line of its source or of vendor/features_fix.py is copied or edited.

The only additions are:
  * a completed-marker/cache gate for TEST (the frozen builder itself does not
    check the marker);
  * the additive S3DIS count 544 -> 542, validated against the denominator
    report and the raw manifest SHA; the frozen 544 bytes are left untouched;
  * a VAL-only gate override (expected=exported n) so the equivalence gate can
    run on a small sample; it is explicitly recorded and is never a TEST path.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c_test_common as T  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=sorted(T.DOMAINS))
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--export_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol_sha", required=True)
    ap.add_argument("--freeze_sha", default=None)
    ap.add_argument("--denominator_report", default=None)
    ap.add_argument("--marker", default=None)
    a = ap.parse_args(argv)
    a.export_dir = os.path.abspath(a.export_dir)
    a.out = os.path.abspath(a.out)
    S = T.frozen_common()
    d = S.domain_cfg(a.domain)
    info = T.domain_info(a.domain)
    export_meta_path = os.path.join(a.export_dir, "EXPORT_META.json")
    if not os.path.isfile(export_meta_path):
        raise SystemExit("EXPORT_META.json missing")
    em = T.read_json(export_meta_path)
    if em.get("domain") != a.domain or em.get("split") != a.split:
        raise SystemExit("export domain/split mismatch")
    if em.get("k") != 25 or em.get("complete") is not True:
        raise SystemExit("export is not a complete K=25 cache")
    n_exp = int(em["n_queries"])

    marker_path = a.marker or os.path.join(a.export_dir, "TEST_READ_ONCE.json")
    marker = None
    if a.split == "test":
        if not os.path.isfile(marker_path):
            raise SystemExit("TEST marker missing")
        marker = T.read_json(marker_path)
        if marker.get("status") != "completed" or marker.get("armed") is not True:
            raise SystemExit("TEST marker is not completed/armed")
        if marker.get("protocol_freeze_sha256") != a.protocol_sha:
            raise SystemExit("TEST marker protocol mismatch")
        if a.freeze_sha and marker.get("freeze_list_sha256") != a.freeze_sha:
            raise SystemExit("TEST marker freeze mismatch")
        if (marker.get("cache_sha256", {}).get("candidates.npz")
                != em.get("cache_sha256", {}).get("candidates.npz")):
            raise SystemExit("TEST marker does not bind this export cache")
        if T.sha256(os.path.join(a.export_dir, "candidates.npz")) != \
                em["cache_sha256"]["candidates.npz"]:
            raise SystemExit("export candidates SHA mismatch")
    else:
        if os.path.isfile(os.path.join(a.export_dir, "TEST_READ_ONCE.json")):
            raise SystemExit("VAL export carries a TEST marker")

    if os.path.exists(a.out) or os.path.exists(a.out + ".meta.json"):
        raise SystemExit(f"{a.out} already exists; refusing to overwrite an old "
                         f"feature file")

    frozen_path = os.path.join(T.FROZEN_CODE, "d4p_features.py")
    frozen_before = T.sha256(frozen_path)
    scar_before = T.sha256(os.path.join(T.FROZEN_CODE, "scar_common.py"))
    count_prov = None
    if a.domain == "s3dis" and a.split == "test":
        rep = os.path.abspath(a.denominator_report or
                              os.path.join(T.TEST_ROOT, "records",
                                           "S3DIS_DENOMINATOR_542_544.json"))
        count_prov = T.s3dis_count_provenance(
            rep, os.path.join(d["base"], "dataset_manifest.json"))
        old = int(S.DOMAINS["s3dis"]["expected"]["test"])
        if old != 544:
            raise SystemExit(f"frozen S3DIS TEST guard changed: {old}")
        if n_exp != 542:
            raise SystemExit(f"export has {n_exp} queries, not 542")
        S.DOMAINS["s3dis"]["expected"]["test"] = 542
        count_prov["old_in_memory_value"] = old
        count_prov["new_in_memory_value"] = 542

    mod = T.frozen_module("d4p_features")
    old_argv = sys.argv
    sys.argv = [frozen_path, "--domain", a.domain, "--split", a.split,
                "--export_dir", a.export_dir, "--out", a.out,
                "--protocol_sha", a.protocol_sha]
    if a.freeze_sha:
        sys.argv += ["--freeze_sha", a.freeze_sha]
    # The count override is in memory only.  The frozen builder always sees the
    # full split, so no truncated export is ever fed to it.
    try:
        rc = mod.main()
    finally:
        sys.argv = old_argv
    if rc not in (0, None):
        raise SystemExit(f"frozen d4p_features returned {rc}")
    if T.sha256(frozen_path) != frozen_before \
            or T.sha256(os.path.join(T.FROZEN_CODE, "scar_common.py")) != scar_before:
        raise SystemExit("frozen source changed during the feature build; "
                         "refusing to bless the output")
    if not os.path.isfile(a.out) or not os.path.isfile(a.out + ".meta.json"):
        raise SystemExit("frozen feature builder did not produce its output")
    meta = T.read_json(a.out + ".meta.json")
    if meta.get("feature_file_sha256") != T.sha256(a.out):
        raise SystemExit("feature file SHA does not match its meta")
    if int(meta.get("queries", -1)) != n_exp:
        raise SystemExit(f"feature queries {meta.get('queries')} != export {n_exp}")
    meta.update(
        feature_entry_point=dict(
            path=frozen_path, sha256=frozen_before, mode="imported in-process; "
            "no source copied or edited",
            expected_guard_after_build=int(S.DOMAINS[a.domain]["expected"][a.split]),
        ),
        test_marker=(dict(path=os.path.abspath(marker_path),
                          sha256=T.sha256(marker_path), status=marker.get("status"))
                     if marker else None),
        s3dis_count_provenance=count_prov,
        gate_limit_override=None,
        c_test_code_sha256={f: T.sha256(os.path.join(HERE, f))
                            for f in ("c_features_test.py", "c_test_common.py")},
        frozen_code_sha256={f: T.sha256(os.path.join(T.FROZEN_CODE, f))
                            for f in ("scar_common.py", "d4p_features.py",
                                      "d4p_export.py")},
        test_opened=a.split == "test",
        denominator=info["denominator"],
        note=("S3DIS 542 is the released count; 544 is nominal. The frozen 544 "
              "source line is preserved and only patched in memory for this "
              "process." if count_prov else None),
    )
    T.write_json(meta, a.out + ".meta.json")
    print("feature meta", a.out + ".meta.json", flush=True)
    print("feature file", a.out, T.sha256(a.out), flush=True)
    if count_prov:
        print("S3DIS expected guard in memory 544 -> 542; frozen bytes unchanged",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
