"""Run the frozen A-domain H-T (SCAR) list builder with FEATURE_COORD_FIX_v1.

The frozen builder source is imported unchanged (its SHA is recorded) and its
``main()`` is called with the original argv.  The only behavioural difference is
that ``common.build_cell_aux`` is monkey-patched *in this process* before the
builder imports it, replacing the double-transformed object centre with the
contract-correct cell-frame mean.  Nothing on disk is edited; old caches,
weights and freeze records are not touched.

Usage:
  python build_ht_t2l_coordfix_v1.py --domain pandaset --split train \
      --export_dir ... --out .../features/train.npz
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys

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
        "env": {},
    },
    "nuscenesfull": {
        "code": "/root/autodl-tmp/text2loc_nuscenes/code",
        "extra_paths": ["/root/autodl-tmp/text2loc_nuscenes/vendor"],
        "builder": "/root/autodl-tmp/text2loc_nuscenes/scripts/build_ht_t2l_nuscenes.py",
        "env": {
            "T2L_NUSC_ROOT":
                "/root/autodl-tmp/text2loc_nuscenes/data/"
                "nuScenesPose-50-25-25_30-10",
        },
    },
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=sorted(DOMAINS))
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--export_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--old-defect", action="store_true",
                    help="diagnostic only: run the frozen builder UNPATCHED to "
                         "produce the old-defect feature version for comparison")
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("TEST is out of scope for this round: TRAIN/VAL only")

    d = DOMAINS[args.domain]
    for k, v in d["env"].items():
        os.environ[k] = v
    os.chdir(d["code"])
    # Mirror the frozen builder's sys.path insert order exactly.
    sys.path.insert(0, d["code"])
    for p in d["extra_paths"]:
        sys.path.insert(0, p)

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import feature_coord_fix_v1 as CF
    import common as C
    import sa_geom as G
    import features_fix as F

    if args.old_defect:
        patch = {
            "feature_version": "OLD_DEFECT_DIAGNOSTIC",
            "applied": False,
            "old_source_sha256": CF._src_sha(C.build_cell_aux),
            "old_formula":
                "(mean(obj.xyz[:, :2]) - cell.bbox_w[0:2]) / cell.cell_size",
            "affected_features": list(CF.AFFECTED_FEATURE_NAMES),
        }
    else:
        patch = CF.apply(C, G)
        patch["applied"] = True
        patch["common_build_cell_aux_source_sha_after_patch"] = CF._src_sha(
            C.build_cell_aux)
    patch["common_module_file"] = C.__file__
    patch["common_module_sha256"] = sha256_file(C.__file__)
    patch["sa_geom_module_sha256"] = sha256_file(G.__file__)
    patch["features_fix_module_sha256"] = sha256_file(F.__file__)

    builder = load_module(d["builder"], "frozen_ht_builder_" + args.domain)
    builder_sha = sha256_file(d["builder"])
    sys.argv = [d["builder"], "--split", args.split,
                "--export_dir", args.export_dir, "--out", args.out]
    if args.limit:
        sys.argv += ["--limit", str(args.limit)]
    rc = builder.main()
    if rc:
        raise SystemExit(f"frozen builder returned {rc}")

    meta_path = args.out + ".meta.json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    meta["feature_version"] = ("OLD_DEFECT_DIAGNOSTIC" if args.old_defect
                               else CF.FEATURE_VERSION)
    meta["coord_fix"] = patch
    meta["builder"] = {"path": d["builder"], "sha256": builder_sha}
    meta["wrapper"] = {"path": os.path.abspath(__file__),
                       "sha256": sha256_file(os.path.abspath(__file__))}
    meta["feature_names_sha256"] = hashlib.sha256(
        json.dumps(list(F.FEATURE_NAMES)).encode()).hexdigest()
    meta["note"] = (
        ("DIAGNOSTIC ONLY: frozen builder run unpatched (old-defect centres); "
         "not a deliverable feature version."
         if args.old_defect else
         "FEATURE_COORD_FIX_v1: only aux object centres (and therefore columns "
         "19/20) differ from the frozen builder; candidates, Fine coordinates, "
         "labels, matching, support and relation scores are computed by the "
         "unchanged frozen code path."))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({"feature_version": CF.FEATURE_VERSION,
                      "out": args.out, "meta": meta_path}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
