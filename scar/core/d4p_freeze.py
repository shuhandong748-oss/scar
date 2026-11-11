#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze one domain's formal-SCAR chain before any TEST access.

Refuses unless the TRAIN/VAL half is complete and independently verified:
  - both exports verified (VERIFY_EXPORT.json PASS)
  - VAL evaluated and independently verified (VERIFY_EVAL.json PASS)
  - three heads with s0 selected as main, normalisation + features bound
  - no TEST directory or marker of this round exists yet
Writes FREEZE_LIST.json and FREEZE_LIST.sha256 (sha over the canonical JSON
without the sha field itself), then this hash is what the TEST marker carries.
"""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scar_common as S  # noqa: E402

CODE_FILES = ("scar_common.py", "d4p_export.py", "d4p_features.py",
              "d4p_train_head.py", "d4p_eval.py", "d4p_verify_export.py",
              "d4p_verify_eval.py", "d4p_freeze.py", "d4p_arm_test.py",
              "gate_bind.py")


def canonical_sha(obj):
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["npm3d", "s3dis"])
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--protocol_sha", required=True)
    a = ap.parse_args(argv)
    run = os.path.abspath(a.run_dir)
    d = S.domain_cfg(a.domain)
    gate = S.load_gate()
    for bad in ("test", "test_export", "export/test"):
        if os.path.exists(os.path.join(run, bad)):
            raise SystemExit(f"TEST material exists at {os.path.join(run, bad)}; "
                             f"freeze must precede it")
    for dp, dns, fns in os.walk(run):
        if "TEST_READ_ONCE.json" in fns:
            raise SystemExit(f"a TEST marker already exists under {run}: {dp}")

    ex_train = os.path.join(run, "export", "train")
    ex_val = os.path.join(run, "export", "val")
    feat_train = os.path.join(run, "features", "train.npz")
    feat_val = os.path.join(run, "features", "val.npz")
    heads = os.path.join(run, "heads")
    ev_val = os.path.join(run, "eval_val")
    ver = os.path.join(run, "verify")

    def load(path, what):
        if not os.path.isfile(path):
            raise SystemExit(f"missing {what}: {path}")
        return S.read_json(path)

    vx_tr = load(os.path.join(ver, "VERIFY_EXPORT_train.json"), "TRAIN export verify")
    vx_va = load(os.path.join(ver, "VERIFY_EXPORT_val.json"), "VAL export verify")
    ve_va = load(os.path.join(ver, "VERIFY_EVAL_val.json"), "VAL evaluation verify")
    for name, v in (("train_export", vx_tr), ("val_export", vx_va),
                    ("val_eval", ve_va)):
        if v.get("status") != "OK":
            raise SystemExit(f"{name} verification is not OK: {v.get('failures')}")

    sel = load(os.path.join(heads, "SELECTION_FORMAL.json"), "head selection")
    if sel.get("main_head_sha256") != sel["seeds"]["s0"]["sha256"]:
        raise SystemExit("main head is not s0")
    norm_path = os.path.join(heads, "NORMALIZATION.json")
    if not os.path.isfile(norm_path):
        raise SystemExit("normalisation missing")
    norm = S.read_json(norm_path)
    if norm["train_npz_sha256"] != S.sha256(feat_train):
        raise SystemExit("normalisation was not fitted on this TRAIN feature file")

    ft = load(feat_train + ".meta.json", "TRAIN feature meta")
    fv = load(feat_val + ".meta.json", "VAL feature meta")
    for m, split in ((ft, "train"), (fv, "val")):
        if m["split"] != split or m["valid_lists"] < 1:
            raise SystemExit(f"{split} feature meta invalid: {m.get('valid_lists')}")
        if m["positives_per_list"] <= 0:
            raise SystemExit(f"{split}: no positive candidates")
    if ft["feature_file_sha256"] != S.sha256(feat_train):
        raise SystemExit("TRAIN feature meta sha mismatch")
    if fv["feature_file_sha256"] != S.sha256(feat_val):
        raise SystemExit("VAL feature meta sha mismatch")

    backbones = S.verify_backbones(a.domain)
    em = {s: S.read_json(os.path.join(run, "export", s, "EXPORT_META.json"))
          for s in ("train", "val")}
    for s in ("train", "val"):
        for name, want in (em[s].get("cache_sha256") or {}).items():
            p = os.path.join(run, "export", s, name)
            if S.sha256(p) != want:
                raise SystemExit(f"{s}/{name} sha drifted since export")

    ev_metrics = S.read_json(os.path.join(ev_val, "metrics.json"))
    gate_record = S.read_json(S.GATE)
    g_checks = gate_record.get("gate", {})
    g_host = (g_checks.get("hostname") or {}).get("detail")
    g_gpu = (g_checks.get("gpu_identity") or {}).get("detail") or []
    g_gpu = g_gpu[0].split(",")[0].strip() if g_gpu else None
    freeze = dict(
        protocol=dict(name="CORR-D4P-SCAR-FORMAL-v1", domain=a.domain,
                      protocol_sha256=a.protocol_sha),
        domain=a.domain, run_dir=run,
        identity=dict(source=S.GATE, sha256=gate["sha256"], hostname=g_host,
                      gpu_uuid=g_gpu),
        gate_binding=gate,
        backbones=backbones,
        num_stats=dict(path=d["num_stats"], sha256=S.sha256(d["num_stats"]),
                       fingerprint=S.num_stats_fingerprint(
                           S.read_json(d["num_stats"]))),
        heads={tag: dict(path=sel["seeds"][tag]["head"],
                         sha256=sel["seeds"][tag]["sha256"],
                         formal_epoch=sel["seeds"][tag]["formal_epoch"],
                         legacy_a_epoch=sel["seeds"][tag]["legacy_a_epoch"])
               for tag in ("s0", "s1", "s2")},
        main_head=dict(tag="s0", path=sel["main_head"],
                       sha256=sel["main_head_sha256"],
                       selection_rule="max VAL Top-1@5m, tie max VAL Cell R@1, "
                                      "then earliest epoch"),
        normalisation=dict(path=norm_path, sha256=S.sha256(norm_path),
                           fitted_on=norm["fitted_on"],
                           pinned=norm["pinned_names"]),
        features=dict(
            definition="vendor/features_fix.py canonical 23; no re-definition",
            train=dict(path=feat_train, sha256=S.sha256(feat_train),
                       meta_sha256=S.sha256(feat_train + ".meta.json"),
                       valid_lists=ft["valid_lists"],
                       positives_per_list=ft["positives_per_list"]),
            val=dict(path=feat_val, sha256=S.sha256(feat_val),
                     meta_sha256=S.sha256(feat_val + ".meta.json"),
                     valid_lists=fv["valid_lists"],
                     positives_per_list=fv["positives_per_list"]),
            constants=dict(k=25, pool_norm=24, rank_norm=24, tied_norm=25,
                           lam=0.5, min_match=4,
                           label="same scene and ||pred_w - gt_pose_w|| <= 5 m",
                           relation_channel=ft["relation_channel"]),
            features_fix_sha256=ft["features_fix_sha256"],
            sa_geom_sha256=ft["sa_geom_sha256"],
            common_sha256=ft["common_sha256"]),
        exports={
            s: dict(dir=os.path.join(run, "export", s),
                    meta_sha256=S.sha256(os.path.join(run, "export", s,
                                                      "EXPORT_META.json")),
                    cache_sha256=em[s].get("cache_sha256"),
                    n_queries=em[s]["n_queries"],
                    checkpoints=em[s]["checkpoints"],
                    rng_policy=em[s]["rng_policy"])
            for s in ("train", "val")},
        val_evaluation=dict(dir=ev_val,
                            metrics_sha256=S.sha256(os.path.join(ev_val,
                                                                 "metrics.json")),
                            per_query_sha256=S.sha256(os.path.join(ev_val,
                                                                   "per_query.jsonl")),
                            results_cell=ev_metrics["results_cell"],
                            results_loc=ev_metrics["results_loc"],
                            comparisons=ev_metrics["comparisons"]),
        verification=dict(
            export_train_sha256=S.sha256(os.path.join(ver,
                                                      "VERIFY_EXPORT_train.json")),
            export_val_sha256=S.sha256(os.path.join(ver,
                                                    "VERIFY_EXPORT_val.json")),
            eval_val_sha256=S.sha256(os.path.join(ver, "VERIFY_EVAL_val.json"))),
        ordering_rules=dict(
            original="descending coarse cosine (storage order)",
            presence="stable descending (cos + 0.5*sp) = P0",
            rank_fusion="ascending (p2_rank + sc_fine_rank)/(2*(K-1)), stable P2",
            fsr="stable descending sat_fine inside P2",
            scar="stable descending s0 logits inside P2; only the order changes, "
                 "coordinates stay the candidates' own Fine predictions"),
        evaluation=dict(
            metrics=["Cell R@1/2/3/5/10/25", "Top-k@5/10/15m for k=1/5/10/25"],
            localization_rule="min distance to the first k decoded candidates, "
                              "cross-cluster candidates set to inf",
            bootstrap=dict(paired=True, resamples=2000, seed=123,
                           cluster_unit=d["cluster_unit"],
                           clusters_train=d["expected_clusters"]["train"],
                           clusters_val=d["expected_clusters"]["val"])),
        scope=dict(round="TRAIN/VAL only", test_read=False, test_armed=False,
                   efficiency_started=False, auto_shutdown=False,
                   note="No TEST data or TEST cache was opened or armed in this "
                        "round; the frozen head and TRAIN/VAL results are the "
                        "deliverable."),
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
    print("saved", out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
