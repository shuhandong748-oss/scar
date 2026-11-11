#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze one A domain's FEATURE_COORD_FIX_v1 TRAIN/VAL chain.

Refuses unless every independent verification is present and OK, and no TEST
material or marker exists anywhere under the run directory.
"""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a_scar_common as S  # noqa: E402

CODE_FILES = (
    "a_scar_common.py", "feature_coord_fix_v1.py",
    "build_ht_t2l_coordfix_v1.py", "export_candidates_coordfix_v1.py",
    "test_coord_fix_synthetic.py", "verify_coord_fix_geom.py",
    "compare_feature_versions.py", "verify_export_coordfix_v1.py",
    "train_head_coordfix_v1.py", "eval_val_coordfix_v1.py",
    "verify_val_coordfix_v1.py", "freeze_coordfix_v1.py",
)


def canonical_sha(obj):
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=sorted(S.DOMAINS))
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--protocol_sha", required=True)
    args = ap.parse_args()
    run = os.path.abspath(args.run_dir)
    for dp, dns, fns in os.walk(run):
        if "TEST_READ_ONCE.json" in fns:
            raise SystemExit(f"TEST marker exists under {run}: {dp}")
        for bad in ("test", "test_export"):
            if os.path.basename(dp) == bad:
                raise SystemExit(f"TEST material exists at {dp}")

    def req(path, what):
        if not os.path.isfile(path):
            raise SystemExit(f"missing {what}: {path}")
        return S.read_json(path)

    ver = os.path.join(run, "verify")
    checks = {
        "synthetic": req(os.path.join(ver, "VERIFY_COORD_FIX_SYNTHETIC.json"),
                         "synthetic coord tests"),
        "geom_train": req(os.path.join(ver, "VERIFY_COORD_FIX_GEOM_train.json"),
                          "real geometry TRAIN"),
        "geom_val": req(os.path.join(ver, "VERIFY_COORD_FIX_GEOM_val.json"),
                        "real geometry VAL"),
        "cmp_train": req(os.path.join(ver, "COMPARE_FEATURES_train.json"),
                         "feature version compare TRAIN"),
        "cmp_val": req(os.path.join(ver, "COMPARE_FEATURES_val.json"),
                       "feature version compare VAL"),
        "export_train": req(os.path.join(ver, "VERIFY_EXPORT_train.json"),
                            "TRAIN export verify"),
        "export_val": req(os.path.join(ver, "VERIFY_EXPORT_val.json"),
                          "VAL export verify"),
        "eval_val": req(os.path.join(ver, "VERIFY_EVAL_val.json"),
                        "VAL evaluation verify"),
        "selection_val": req(os.path.join(ver, "VERIFY_SELECTION_val.json"),
                             "VAL selection verify"),
    }
    for name, v in checks.items():
        if v.get("status") != "OK" and v.get("ok") is not True:
            raise SystemExit(f"{name} verification failed: "
                             f"{v.get('checks') or v.get('failures')}")

    feat_train = os.path.join(run, "features", "train.npz")
    feat_val = os.path.join(run, "features", "val.npz")
    ex_train = os.path.join(run, "export", "train")
    ex_val = os.path.join(run, "export", "val")
    heads = os.path.join(run, "heads")
    ev = os.path.join(run, "eval_val")
    sel = req(os.path.join(heads, "SELECTION_FORMAL.json"), "selection")
    norm = req(os.path.join(heads, "NORMALIZATION.json"), "normalisation")
    ft = req(feat_train + ".meta.json", "TRAIN feature meta")
    fv = req(feat_val + ".meta.json", "VAL feature meta")
    em_t = req(os.path.join(ex_train, "export_meta.json"), "TRAIN export meta")
    em_v = req(os.path.join(ex_val, "export_meta.json"), "VAL export meta")
    if ft.get("feature_version") != S.FEATURE_VERSION or \
            fv.get("feature_version") != S.FEATURE_VERSION:
        raise SystemExit("feature metas are not FEATURE_COORD_FIX_v1")
    if norm["train_npz_sha256"] != S.sha256(feat_train):
        raise SystemExit("normalisation not fitted on this TRAIN feature file")
    if sel["main_head_sha256"] != sel["seeds"]["s0"]["sha256"]:
        raise SystemExit("main head is not s0")
    metrics = req(os.path.join(ev, "metrics.json"), "VAL metrics")
    binding = req(os.path.join(run, "records", "START_BINDING.json"),
                  "start binding")

    freeze = dict(
        protocol=dict(name=S.PROTOCOL, feature_version=S.FEATURE_VERSION,
                      protocol_sha256=args.protocol_sha),
        domain=args.domain, run_dir=run, identity=binding,
        backbones=dict(
            train=em_t["checkpoints"], val=em_v["checkpoints"]),
        coord_fix=dict(
            version=S.FEATURE_VERSION,
            affected_features=list(ft["coord_fix"]["affected_features"]),
            old_formula=ft["coord_fix"]["old_formula"],
            new_formula=ft["coord_fix"]["new_formula"],
            old_source_sha256=ft["coord_fix"]["old_source_sha256"],
            new_source_sha256=ft["coord_fix"]["new_source_sha256"],
            common_sha256=ft["coord_fix"]["common_module_sha256"],
            sa_geom_sha256=ft["coord_fix"]["sa_geom_module_sha256"],
            features_fix_sha256=ft["coord_fix"]["features_fix_module_sha256"],
            builder_sha256=ft["builder"]["sha256"],
            note="feature implementation version change; not bitwise equivalent "
                 "to the historical A feature definition"),
        heads={f"s{s}": dict(path=sel["seeds"][f"s{s}"]["head"],
                             sha256=sel["seeds"][f"s{s}"]["sha256"],
                             formal_epoch=sel["seeds"][f"s{s}"]["formal_epoch"])
               for s in (0, 1, 2)},
        main_head=dict(tag="s0", path=sel["main_head"],
                       sha256=sel["main_head_sha256"],
                       selection_rule="max VAL Top-1@5m, tie max VAL Cell R@1, "
                                      "then earliest epoch"),
        normalisation=dict(path=os.path.join(heads, "NORMALIZATION.json"),
                           sha256=S.sha256(os.path.join(heads,
                                                        "NORMALIZATION.json")),
                           fitted_on=norm["fitted_on"],
                           pinned=norm["pinned_names"]),
        features=dict(
            train=dict(path=feat_train, sha256=S.sha256(feat_train),
                       meta_sha256=S.sha256(feat_train + ".meta.json"),
                       meta=ft),
            val=dict(path=feat_val, sha256=S.sha256(feat_val),
                     meta_sha256=S.sha256(feat_val + ".meta.json"),
                     meta=fv),
            constants=dict(k=25, pool_norm=24, rank_norm=24, tied_norm=25,
                           lam=0.5, min_match=4,
                           label="same scene and ||pred_w - gt_pose_w|| <= 5 m")),
        exports={s: dict(dir=os.path.join(run, "export", s),
                         meta_sha256=S.sha256(os.path.join(
                             run, "export", s, "export_meta.json")),
                         candidates_sha256=S.sha256(os.path.join(
                             run, "export", s, "candidates.npz")),
                         coarse_enc_sha256=S.sha256(os.path.join(
                             run, "export", s, "coarse_enc.npz")),
                         n_queries=em["n_queries"])
                 for s, em in (("train", em_t), ("val", em_v))},
        val_evaluation=dict(
            dir=ev, metrics_sha256=S.sha256(os.path.join(ev, "metrics.json")),
            per_query_sha256=S.sha256(os.path.join(ev, "per_query.jsonl")),
            results_cell=metrics["results_cell"],
            results_loc=metrics["results_loc"],
            comparisons=metrics["comparisons"]),
        verification={k: v.get("status", "OK") for k, v in checks.items()},
        scope=dict(round="TRAIN/VAL only", test_read=False, test_armed=False,
                   efficiency_started=False, auto_shutdown=False,
                   note="No TEST data or cache was opened or armed; the frozen "
                        "head and TRAIN/VAL results are the deliverable."),
        code_sha256={f: S.sha256(os.path.join(HERE, f)) for f in CODE_FILES
                     if os.path.isfile(os.path.join(HERE, f))},
        frozen_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    freeze["freeze_sha256"] = canonical_sha(freeze)
    out = os.path.join(run, "FREEZE_LIST.json")
    if os.path.exists(out):
        raise SystemExit(f"{out} already exists (frozen evidence)")
    S.write_json(freeze, out)
    with open(os.path.join(run, "FREEZE_LIST.sha256"), "w", encoding="utf-8") as f:
        f.write(freeze["freeze_sha256"] + "  FREEZE_LIST.json\n")
    print("freeze_sha256", freeze["freeze_sha256"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
