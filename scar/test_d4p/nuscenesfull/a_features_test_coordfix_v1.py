#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A-domain TEST feature build under FEATURE_COORD_FIX_v1.

This is the TEST-enabled sibling of the frozen TRAIN/VAL wrapper.  It imports
the frozen builder unchanged and applies the frozen ``feature_coord_fix_v1``
patch in-process (``common.py`` on disk is never edited).  The TEST marker must
already be armed.  The feature artifact is written atomically; a successful
artifact is reused when its recorded SHA and input bindings match.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a_test_common as S  # noqa: E402

EXPECTED_EPOCHS = {
    "pandaset": {"coarse": 8, "fine": 35},
    "nuscenesfull": {"coarse": 17, "fine": 30},
}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=sorted(S.DOMAINS))
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--export_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--marker", default=None)
    ap.add_argument("--freeze_list", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--frozen_code_dir", default=os.path.join(
        os.path.dirname(S.A_TRAINVAL), "A_round", "code"))
    ap.add_argument("--no_reuse", action="store_true")
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out)
    export_dir = os.path.abspath(a.export_dir)
    d = S.domain_cfg(a.domain)

    em_path = os.path.join(export_dir, "export_meta.json")
    if not os.path.isfile(em_path):
        raise SystemExit(f"export meta missing: {em_path}")
    em = S.read_json(em_path)
    if em.get("domain") != a.domain or em.get("split") != a.split:
        raise SystemExit("export meta domain/split mismatch")
    if int(em.get("k", 0)) != 25:
        raise SystemExit("export is not K=25")
    marker_sha = None
    if a.split == "test":
        if not a.marker:
            raise SystemExit("TEST feature build requires --marker")
        m = S.require_marker(a.marker, a.domain, "armed",
                             freeze_sha=em["freeze"]["canonical_sha256"],
                             proc_sha=em["freeze"]["protocol_sha256"])
        marker_sha = S.sha256(a.marker)
        comp = S.read_json(os.path.join(export_dir, "EXPORT_COMPLETE.json"))
        for name in ("candidates.npz", "coarse_enc.npz"):
            p = os.path.join(export_dir, name)
            if S.sha256(p) != comp["cache_sha256"][name]:
                raise SystemExit(f"export cache SHA mismatch: {name}")

    if os.path.isfile(out) and os.path.isfile(out + ".meta.json") \
            and not a.no_reuse:
        old = S.read_json(out + ".meta.json")
        same = (old.get("export_meta_sha256") == S.sha256(em_path)
                and old.get("feature_file_sha256") == S.sha256(out)
                and old.get("domain") == a.domain and old.get("split") == a.split
                and old.get("queries") == em.get("n_queries"))
        if same:
            print(f"FEATURES_REUSED {out}", flush=True)
            return 0
    if os.path.exists(out) or os.path.exists(out + ".meta.json"):
        dq = os.path.join(os.path.dirname(out), "failed",
                          time.strftime("%Y%m%dT%H%M%S") + "_features")
        os.makedirs(dq, exist_ok=True)
        for p in (out, out + ".meta.json"):
            if os.path.exists(p):
                shutil.move(p, os.path.join(dq, os.path.basename(p)))
        S.write_json(dict(reason="existing feature artifact did not match its "
                                 "recorded bindings; preserved before rebuild",
                          at=time.strftime("%Y-%m-%dT%H:%M:%S%z")),
                     os.path.join(dq, "FAILURE.json"))

    # ---- mirror the frozen wrapper's import order exactly -----------------
    for k, v in d["env"].items():
        os.environ[k] = v
    os.chdir(d["code"])
    sys.path.insert(0, d["code"])
    for p in d["extra_paths"]:
        sys.path.insert(0, p)
    if os.path.abspath(a.frozen_code_dir) not in sys.path:
        sys.path.insert(0, os.path.abspath(a.frozen_code_dir))
    import feature_coord_fix_v1 as CF
    import common as C
    import sa_geom as G
    import features_fix as F

    patch = CF.apply(C, G)
    patch["applied"] = True
    patch["common_build_cell_aux_source_sha_before"] = patch.get(
        "old_source_sha256")
    patch["common_build_cell_aux_source_sha_after"] = CF._src_sha(
        C.build_cell_aux)
    patch["common_module_file"] = C.__file__
    patch["common_module_sha256"] = S.sha256(C.__file__)
    patch["sa_geom_module_sha256"] = S.sha256(G.__file__)
    patch["features_fix_module_sha256"] = S.sha256(F.__file__)
    patch["feature_coord_fix_module_sha256"] = S.sha256(CF.__file__)

    builder_path = d["builder"]
    builder = load_module(builder_path, "frozen_ht_builder_" + a.domain)
    tmp_out = out + ".building.npz"
    for p in (tmp_out, tmp_out + ".meta.json"):
        if os.path.exists(p):
            os.remove(p)
    sys.argv = [builder_path, "--split", a.split,
                "--export_dir", export_dir, "--out", tmp_out]
    if a.limit:
        sys.argv += ["--limit", str(a.limit)]
    rc = builder.main()
    if rc:
        raise SystemExit(f"frozen builder returned {rc}")
    if not (os.path.isfile(tmp_out) and os.path.isfile(tmp_out + ".meta.json")):
        raise SystemExit("frozen builder did not produce the expected files")

    meta = S.read_json(tmp_out + ".meta.json")
    meta["feature_version"] = CF.FEATURE_VERSION
    meta["coord_fix"] = patch
    meta["builder"] = {"path": builder_path, "sha256": S.sha256(builder_path)}
    meta["wrapper"] = {"path": os.path.abspath(__file__),
                       "sha256": S.sha256(os.path.abspath(__file__))}
    meta["feature_names"] = list(F.FEATURE_NAMES)
    meta["feature_names_sha256"] = hashlib.sha256(
        json.dumps(list(F.FEATURE_NAMES)).encode()).hexdigest()
    meta["export_dir"] = export_dir
    meta["export_meta_sha256"] = S.sha256(em_path)
    meta["export_cache_sha256"] = em.get("chunk_manifest_sha256")
    meta["test_marker"] = a.marker
    meta["test_marker_sha256"] = marker_sha
    meta["test_opened"] = a.split == "test"
    meta["count_provenance"] = dict(
        queries=int(meta.get("queries", 0)),
        export_queries=int(em.get("n_queries", 0)),
        denominator="all queries; no filtering")
    meta["note"] = (
        "FEATURE_COORD_FIX_v1 TEST build: in-process monkey-patch of "
        "common.build_cell_aux only; columns 19/20 change, all other frozen "
        "behaviour is the unchanged builder.  No retraining and no reselection.")
    z = np.load(tmp_out, allow_pickle=True)
    if list(z["feature_names"]) != list(F.FEATURE_NAMES):
        raise SystemExit("feature names in the npz are not canonical-23")
    if int(z["feats"].shape[1]) != 25 or int(z["feats"].shape[2]) != 23:
        raise SystemExit(f"unexpected feature shape {z['feats'].shape}")
    if str(z["split"][0]) != a.split:
        raise SystemExit("feature split mismatch")
    meta["feature_file_sha256"] = S.sha256(tmp_out)

    S.write_json(meta, tmp_out + ".meta.json")
    os.replace(tmp_out, out)
    os.replace(tmp_out + ".meta.json", out + ".meta.json")
    if a.split == "test" and a.marker:
        mm = S.read_json(a.marker)
        mm["status"] = "completed"
        mm["features"] = dict(
            path=out, sha256=S.sha256(out),
            meta_sha256=S.sha256(out + ".meta.json"),
            feature_version=CF.FEATURE_VERSION,
            feature_names_sha256=meta["feature_names_sha256"],
            queries=int(meta.get("queries", 0)),
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        S.write_json(mm, a.marker)
    print(json.dumps(dict(out=out, sha256=S.sha256(out),
                          feature_version=CF.FEATURE_VERSION,
                          queries=meta.get("queries"),
                          marker_sha256=marker_sha), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
