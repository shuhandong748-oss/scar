#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate and protocol binding for corrected Des4Pos x NPM3D / S3DIS formal SCAR.

Read-only: opens metadata and checkpoint payloads, hashes frozen inputs; never
constructs a dataset, never opens TEST cells/poses, never runs a model forward,
never takes the GPU lock (it only probes it with flock -n).

Exit 0 only if every binding check passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time

ROOT = "/root/autodl-tmp/d4p_trainval_20261104/C"
CORR = "/root/autodl-tmp/des4pos_corrected_C_20261102"
CORE = os.path.join(CORR, "code", "core")
ADAPT = os.path.join(CORR, "code", "adapt")
CORRECTED = os.path.join(CORR, "code", "corrected")
EXPECTED_HOST = "autodl-container-<CONTAINER_ID>"
# Current allocation of this container, bound by the nvidia-smi identity check.
EXPECTED_GPU = "GPU-<GPU_UUID>"
# Allocation that actually trained the corrected backbones on 2026-11-02, as
# recorded by both STAGE_RESULT and STAGE_FREEZE.  A container restart migrates
# the same instance to a different GPU UUID, so the stage records are checked
# for internal consistency against TRAIN_GPU, while the live run is bound to
# EXPECTED_GPU above.  The weights themselves are bound by full SHA-256.
TRAIN_GPU = "GPU-<GPU_UUID>"
LOCK = "/root/autodl-tmp/efficiency_audit_20261029/GPU.lock"

DOMAINS = {
    "npm3d": {
        "run": os.path.join(CORR, "npm3d_s0"),
        "base": "/root/autodl-tmp/des4pos_npm3d/data/NPM3D",
        "vendor": "/root/autodl-tmp/des4pos_npm3d/vendor",
        "adapter": "npm3d_domain",
        "prior": "/root/autodl-tmp/des4pos_npm3d/runs/R059-C-Des4PosPaper-NPM3D-bb0/ste_stage1/ste_stage1_prior.pth",
        "prior_sha": "2e3c4cd1aa7fe0598b9d647e02fc7e2f8aecd0572643751225d23b5e4e694fca",
        "pointnet": "/root/autodl-tmp/des4pos_npm3d/assets/pointnet/pointnet_acc0.86_lr1_p256.pth",
        "pointnet_sha": "ef63fdfc9cd10e698181efe415a873e628cf86698266f005351943030016e8e1",
        "num_stats": "/root/autodl-tmp/des4pos_npm3d/registration/R059-C-Des4PosPaper-NPM3D-bb0/num_stats.json",
        "num_stats_sha": "9798c01228dc9d870a2c624d46db1ea6cbdf96abed6d2e88ca7dcadb4aa82c4c",
        "t5": "/root/autodl-tmp/des4pos_npm3d/assets/t5-large",
        "t5_tree_sha": "fefa48c1d2e403b552be6b1a78c5d8681bb81c4cf375710c756f39ea313a7526",
        "manifest_sha": "6132a54031c214014511aa5592bab0d60ac673d9e31fdd14ca11e52b24f3051c",
        "num_mentioned": 3,
        "coarse_epoch": 20,
        "fine_epoch": 30,
        "historical_test_marker": "/root/autodl-tmp/des4pos_npm3d/runs/R059-C-Des4PosPaper-NPM3D-bb0/export/test/TEST_READ_ONCE.json",
        "historical_ckpt": "/root/autodl-tmp/des4pos_npm3d/runs/R059-C-Des4PosPaper-NPM3D-bb0/ckpt",
        "expected_train_cells": 30, "expected_val_cells": 32, "expected_test_cells": 29,
        "bootstrap_cluster": "scene (one cell per scene, 29 TEST clusters)",
    },
    "s3dis": {
        "run": os.path.join(CORR, "s3dis_s0"),
        "base": "/root/autodl-tmp/text2loc_s3dis/data/S3DISPose",
        "vendor": "/root/autodl-tmp/des4pos_s3dis/vendor",
        "adapter": "s3dis_domain",
        "prior": "/root/autodl-tmp/des4pos_s3dis/runs/R060_D4P_S3DIS_bb0/ste_stage1/ste_stage1_prior.pth",
        "prior_sha": "70ce5b1eb67ed001141bc94b998f23eaa4f985dfa50f2efd04b80ee72f4b9739",
        "pointnet": "/root/autodl-tmp/text2loc_s3dis/code/checkpoints/pointnet_acc0.86_lr1_p256.pth",
        "pointnet_sha": "ef63fdfc9cd10e698181efe415a873e628cf86698266f005351943030016e8e1",
        "num_stats": "/root/autodl-tmp/des4pos_s3dis/registration/R060_D4P_S3DIS_bb0/num_stats.json",
        "num_stats_sha": "752dcbf2bc8c9417c3f86eeb4227bd418c4b99879556a4860c80bb3fd3c7fe97",
        "t5": "/root/autodl-tmp/text2loc_s3dis/assets/t5-large",
        "t5_tree_sha": None,
        "manifest_sha": "41e00444edf66f72f9e44508c2071e6dd087c0edb7c4a58c0df4b07c59267b15",
        "num_mentioned": 4,
        "coarse_epoch": 15,
        "fine_epoch": 34,
        "historical_test_marker": "/root/autodl-tmp/des4pos_s3dis/runs/R060_D4P_S3DIS_bb0/export/test/TEST_READ_ONCE.json",
        "historical_ckpt": "/root/autodl-tmp/des4pos_s3dis/runs/R060_D4P_S3DIS_bb0/ckpt",
        "expected_train_cells": 136, "expected_val_cells": 68, "expected_test_cells": 68,
        "bootstrap_cluster": "room (scene_name, 68 TEST clusters)",
    },
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sha256_tree(root):
    h = hashlib.sha256()
    entries = []
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d != "__pycache__")
        for n in sorted(fns):
            if n.endswith((".pyc", ".pyo")):
                continue
            p = os.path.join(dp, n)
            entries.append((os.path.relpath(p, root).replace(os.sep, "/"), p))
    for rel, p in sorted(entries):
        h.update(rel.encode() + b"\0" + str(os.path.getsize(p)).encode() + b"\0")
        h.update(sha256(p).encode() + b"\n")
    return h.hexdigest(), len(entries)


def run(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    return {"rc": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check(cond, name, detail=None):
    ok = False
    try:
        ok = bool(cond)
    except Exception:
        ok = False
    return {"PASS": ok, "detail": detail, "check": name}


def num_stats_fingerprint(stats):
    payload = {
        "version": int(stats["version"]), "domain": stats["domain"],
        "split": stats["split"], "provenance": stats["provenance"],
        "mean": float(stats["mean"]), "std": float(stats["std"]),
        "std_ddof": int(stats["std_ddof"]),
        "pinned_zero_variance": bool(stats["pinned_zero_variance"]),
        "counts_sha256": stats.get("counts_sha256"),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "records", "GATE_BINDING.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    t0 = time.time()
    checks = {}
    info = {}

    checks["hostname"] = check(socket.gethostname() == EXPECTED_HOST, "hostname",
                               socket.gethostname())
    gpu = run("nvidia-smi --query-gpu=uuid,name,memory.used,utilization.gpu "
              "--format=csv,noheader")
    gpu_lines = [x.strip() for x in gpu["stdout"].splitlines() if x.strip()]
    checks["gpu_identity"] = check(
        len(gpu_lines) == 1 and gpu_lines[0].split(",")[0].strip() == EXPECTED_GPU,
        "gpu", gpu_lines)
    apps = run("nvidia-smi --query-compute-apps=pid,process_name,used_memory "
               "--format=csv,noheader")
    checks["gpu_idle"] = check(apps["stdout"] == "", "compute_apps", apps["stdout"])
    info["gpu_allocation"] = {
        "current": gpu_lines,
        "training_recorded": TRAIN_GPU,
        "same_allocation": any(TRAIN_GPU in x for x in gpu_lines),
        "note": "stage records are checked against TRAIN_GPU; this run is bound "
                "to the live EXPECTED_GPU; weights are bound by full SHA-256",
    }
    checks["lock_file_present"] = check(os.path.isfile(LOCK), "lock", LOCK)
    lock = run(f"flock -n {LOCK} true")
    checks["lock_free"] = check(lock["rc"] == 0, "lock_probe", lock["stdout"] or "free")
    disk = run("df -B1 --output=size,used,avail /root/autodl-tmp /")
    rows = [x.split() for x in disk["stdout"].splitlines()[1:] if x.strip()]
    info["disk_bytes"] = rows
    checks["disk_available"] = check(
        len(rows) >= 2 and all(int(r[2]) >= 8 * 1024**3 for r in rows[:2]),
        "avail_bytes", rows)

    code_files = [
        os.path.join(ADAPT, "npm3d_domain.py"), os.path.join(ADAPT, "s3dis_domain.py"),
        os.path.join(ADAPT, "ckpt_patch.py"),
        os.path.join(CORRECTED, "text_alignment.py"),
        os.path.join(CORRECTED, "cmd11_lib.py"), os.path.join(CORRECTED, "common11.py"),
        os.path.join(CORRECTED, "run_coarse_fresh.py"),
        os.path.join(CORRECTED, "run_fine_fresh.py"),
        os.path.join(CORRECTED, "verify_accept.py"),
    ]
    missing_code = [p for p in code_files if not os.path.isfile(p)]
    info["corrected_code"] = {os.path.basename(p): sha256(p)
                              for p in code_files if os.path.isfile(p)}
    checks["corrected_code_present"] = check(not missing_code, "missing", missing_code)

    core_manifest = os.path.join(CORE, "MANIFEST.sha256")
    info["core_manifest_sha256"] = sha256(core_manifest)
    bad = []
    n_core = 0
    with open(core_manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            digest, rel = line.split(None, 1)
            rel = rel.lstrip("*")
            full = os.path.join(CORE, rel)
            n_core += 1
            if not os.path.isfile(full) or sha256(full) != digest:
                bad.append(rel)
    checks["core_manifest"] = check(not bad and n_core == 59, "failures",
                                    {"checked": n_core, "failures": bad[:10]})

    queue = read_json(os.path.join(CORR, "TRAINING_QUEUE_DONE.json"))
    info["training_queue"] = {k: queue.get(k) for k in
                              ("status", "test_access", "scar_training",
                               "efficiency_forward", "auto_shutdown")}
    checks["training_queue_complete"] = check(
        str(queue.get("status")).upper() == "COMPLETE"
        and queue.get("test_access") is False
        and queue.get("scar_training") is False
        and queue.get("efficiency_forward") is False
        and queue.get("auto_shutdown") is False,
        "queue", info["training_queue"])

    domain_out = {}
    for name, cfg in DOMAINS.items():
        rec = {"paths": {}, "shas": {}, "checks": {}, "stages": {}}
        rr = cfg["run"]
        accept = read_json(os.path.join(rr, "ACCEPT.json"))
        rec["accept"] = {k: accept.get(k) for k in
                         ("domain", "run_root", "time", "PASS", "failures",
                          "test_access", "scar_training")}
        rec["checks"]["accept_PASS"] = check(
            accept.get("PASS") is True and accept.get("failures") == [],
            "accept", {"PASS": accept.get("PASS"), "failures": accept.get("failures")})
        rec["checks"]["accept_no_test_no_scar"] = check(
            accept.get("test_access") is False and accept.get("scar_training") is False,
            "flags", [accept.get("test_access"), accept.get("scar_training")])
        rec["checks"]["accept_checks_all_PASS"] = check(
            all(isinstance(v, dict) and v.get("PASS") for v in accept["checks"].values()),
            "fails", [k for k, v in accept["checks"].items()
                      if not (isinstance(v, dict) and v.get("PASS"))])
        # ACCEPT records no manifest hash; bind the split through INIT_FREEZE
        # scene lists / counts against the manifest below (init_data_matches_manifest).

        ini = {}
        for s, epoch in (("coarse", cfg["coarse_epoch"]), ("fine", cfg["fine_epoch"])):
            sd = os.path.join(rr, s)
            res = read_json(os.path.join(sd, "STAGE_RESULT.json"))
            frz = read_json(os.path.join(sd, "STAGE_FREEZE.json"))
            ini[s] = read_json(os.path.join(sd, "INIT_FREEZE.json"))
            best = read_json(os.path.join(sd, "BEST.json"))
            ck = os.path.join(sd, f"{s}_best.pth")
            actual = sha256(ck)
            payload = None
            try:
                import torch
                payload = torch.load(ck, map_location="cpu", weights_only=False)
                priorish = [k for k in payload["model"] if k == "prior"
                            or k.startswith("prior.") or ".prior." in k]
                ck_info = {
                    "top_keys": list(payload.keys()),
                    "n_tensors": len(payload["model"]),
                    "prior_keys_in_light_state": priorish[:5],
                    "light_state_digest": payload.get("light_state_digest"),
                    "extra_keys": list(payload.get("extra", {}).keys()),
                    "extra_num_stats_fingerprint":
                        payload.get("extra", {}).get("num_stats_fingerprint"),
                }
            except Exception as e:  # pragma: no cover
                ck_info = {"error": repr(e)}
            rec["stages"][s] = {
                "result": {k: res.get(k) for k in
                           ("status", "best_epoch", "best_metric", "best_value",
                            "best_file_sha256", "test_access", "gpu_uuid")},
                "freeze": {"argv": frz.get("argv"), "config": frz.get("config"),
                           "frozen_assets": frz.get("frozen_assets"),
                           "code_freeze": frz.get("code_freeze"),
                           "gpu_uuid": frz.get("gpu_uuid"),
                           "test_access": frz.get("test_access")},
                "init_freeze": {"known_classes": ini[s].get("known_classes"),
                                "num_stats": ini[s].get("num_stats"),
                                "num_stats_fingerprint":
                                    ini[s].get("num_stats_fingerprint")},
                "best_json": best,
                "checkpoint": ck, "checkpoint_sha256": actual, "checkpoint_info": ck_info,
                "sha256": {"STAGE_RESULT": sha256(os.path.join(sd, "STAGE_RESULT.json")),
                           "STAGE_FREEZE": sha256(os.path.join(sd, "STAGE_FREEZE.json")),
                           "INIT_FREEZE": sha256(os.path.join(sd, "INIT_FREEZE.json")),
                           "BEST": sha256(os.path.join(sd, "BEST.json")),
                           "METRICS": sha256(os.path.join(sd, "METRICS.jsonl"))},
            }
            rec["checks"][f"{s}_epoch"] = check(
                res.get("best_epoch") == epoch and best.get("epoch") == epoch,
                "epochs", [res.get("best_epoch"), best.get("epoch"), epoch])
            rec["checks"][f"{s}_checkpoint_sha"] = check(
                res.get("best_file_sha256") == actual, "sha",
                [actual, res.get("best_file_sha256")])
            rec["checks"][f"{s}_status_complete"] = check(
                str(res.get("status")).upper() == "COMPLETE", "status", res.get("status"))
            rec["checks"][f"{s}_test_access_false"] = check(
                res.get("test_access") is False and frz.get("test_access") is False,
                "flags", [res.get("test_access"), frz.get("test_access")])
            rec["checks"][f"{s}_gpu_uuid"] = check(
                res.get("gpu_uuid") == frz.get("gpu_uuid") == TRAIN_GPU,
                "gpu_train", [res.get("gpu_uuid"), frz.get("gpu_uuid"), TRAIN_GPU])
            rec["checks"][f"{s}_light_state_no_prior"] = check(
                ck_info.get("prior_keys_in_light_state") == [], "prior_keys",
                ck_info.get("prior_keys_in_light_state"))

        # architecture parity between the two stage configs
        arch_keys = ("base_path", "pad_size", "object_size", "num_mentioned",
                     "pointnet_path", "t5_path", "ste_stage1_path", "coarse_embed_dim",
                     "fine_embed_dim", "pointnet_numpoints", "max_text_len",
                     "cra_steps", "pmc_threshold", "offset_lambda", "prior_layers",
                     "prior_heads", "clip_dim", "mfam_heads", "bilstm_stride",
                     "bilstm_layers", "ste_word_layers", "ste_inter_layers", "ste_heads",
                     "fine_text_layers", "cra_heads", "dropout", "no_pc_augment")
        ccfg = rec["stages"]["coarse"]["freeze"]["config"]
        fcfg = rec["stages"]["fine"]["freeze"]["config"]
        diff_arch = {k: [ccfg.get(k), fcfg.get(k)] for k in arch_keys
                     if ccfg.get(k) != fcfg.get(k)}
        rec["checks"]["arch_config_parity"] = check(not diff_arch, "diff", diff_arch)
        missing_arch = [k for k in arch_keys if k not in ccfg or k not in fcfg]
        rec["checks"]["arch_config_complete"] = check(not missing_arch, "missing", missing_arch)
        rec["checks"]["base_path"] = check(
            ccfg.get("base_path") == cfg["base"] and fcfg.get("base_path") == cfg["base"],
            "base", cfg["base"])
        rec["checks"]["num_mentioned"] = check(
            ccfg.get("num_mentioned") == cfg["num_mentioned"], "num_mentioned",
            [ccfg.get("num_mentioned"), cfg["num_mentioned"]])
        rec["checks"]["prior_path"] = check(
            ccfg.get("ste_stage1_path") == cfg["prior"]
            and fcfg.get("ste_stage1_path") == cfg["prior"], "prior", cfg["prior"])
        rec["checks"]["pointnet_path"] = check(
            ccfg.get("pointnet_path") == cfg["pointnet"]
            and fcfg.get("pointnet_path") == cfg["pointnet"], "pointnet", cfg["pointnet"])
        rec["checks"]["t5_path"] = check(
            ccfg.get("t5_path") == cfg["t5"] and fcfg.get("t5_path") == cfg["t5"],
            "t5", cfg["t5"])

        # frozen input SHAs
        rec["paths"] = {"run_root": rr, "base": cfg["base"], "vendor": cfg["vendor"],
                        "adapter": os.path.join(ADAPT, f"{cfg['adapter']}.py"),
                        "prior": cfg["prior"], "pointnet": cfg["pointnet"],
                        "num_stats": cfg["num_stats"], "t5": cfg["t5"],
                        "manifest": os.path.join(cfg["base"], "dataset_manifest.json")}
        rec["shas"] = {
            "prior": sha256(cfg["prior"]), "pointnet": sha256(cfg["pointnet"]),
            "num_stats": sha256(cfg["num_stats"]),
            "manifest": sha256(rec["paths"]["manifest"]),
            "adapter": sha256(rec["paths"]["adapter"]),
            "vendor": {f: sha256(os.path.join(cfg["vendor"], f))
                       for f in ("common.py", "features_fix.py", "sa_geom.py")},
        }
        t5sha, t5n = sha256_tree(cfg["t5"])
        rec["shas"]["t5_tree"], rec["shas"]["t5_files"] = t5sha, t5n
        rec["checks"]["prior_sha"] = check(rec["shas"]["prior"] == cfg["prior_sha"],
                                           "sha", [rec["shas"]["prior"], cfg["prior_sha"]])
        rec["checks"]["pointnet_sha"] = check(
            rec["shas"]["pointnet"] == cfg["pointnet_sha"], "sha",
            [rec["shas"]["pointnet"], cfg["pointnet_sha"]])
        rec["checks"]["num_stats_sha"] = check(
            rec["shas"]["num_stats"] == cfg["num_stats_sha"], "sha",
            [rec["shas"]["num_stats"], cfg["num_stats_sha"]])
        rec["checks"]["manifest_sha"] = check(
            rec["shas"]["manifest"] == cfg["manifest_sha"], "sha",
            [rec["shas"]["manifest"], cfg["manifest_sha"]])
        if cfg["t5_tree_sha"]:
            rec["checks"]["t5_tree_sha"] = check(
                rec["shas"]["t5_tree"] == cfg["t5_tree_sha"], "sha",
                [rec["shas"]["t5_tree"], cfg["t5_tree_sha"]])
        want_vendor = {
            "common.py": "b069dae07fd20b473111af5f4e062a8d24da75e2891bd12fb463d65c703215bb",
            "features_fix.py": "ea280d679fc64a976407ce6ebacaa2fda797b10eee1657d2b5e07306959f4615",
            "sa_geom.py": "1c4ebdc5a9673b69bb44da4db3163e527330d60da850b5f36b951ad911ccfd45",
        }
        rec["checks"]["vendor_sha"] = check(
            rec["shas"]["vendor"] == want_vendor, "sha", rec["shas"]["vendor"])

        # num_stats: registered == coarse copy, checkpoint fingerprint matches
        fine_cfg = rec["stages"]["fine"]["freeze"]["config"]
        copy_path = fine_cfg.get("num_stats_path")
        rec["paths"]["num_stats_coarse_copy"] = copy_path
        have_copy = bool(copy_path) and os.path.isfile(copy_path)
        rec["shas"]["num_stats_coarse_copy"] = sha256(copy_path) if have_copy else None
        rec["checks"]["num_stats_copy_sha"] = check(
            have_copy and rec["shas"]["num_stats_coarse_copy"] == cfg["num_stats_sha"],
            "sha", [rec["shas"]["num_stats_coarse_copy"], cfg["num_stats_sha"]])
        reg = read_json(cfg["num_stats"])
        copy_eq = have_copy and read_json(copy_path) == reg
        rec["checks"]["num_stats_copy_values"] = check(copy_eq, "equal", bool(copy_eq))
        fp = num_stats_fingerprint(reg)
        rec["num_stats_fingerprint"] = fp
        for s in ("coarse", "fine"):
            got = rec["stages"][s]["checkpoint_info"].get("extra_num_stats_fingerprint")
            rec["checks"][f"{s}_ckpt_num_stats_fp"] = check(got == fp, "fingerprints",
                                                            [got, fp])
            ini_fp = rec["stages"][s]["init_freeze"].get("num_stats_fingerprint")
            rec["checks"][f"{s}_init_num_stats_fp"] = check(ini_fp == fp, "fingerprints",
                                                            [ini_fp, fp])

        # vocabulary binding: INIT_FREEZE known_classes == adapter labels
        sys.path.insert(0, ADAPT)
        mod = __import__(cfg["adapter"])
        labels = list(mod.NPM3D_LABELS if name == "npm3d" else mod.S3DIS_LABELS)
        for s in ("coarse", "fine"):
            got = rec["stages"][s]["init_freeze"]["known_classes"]
            rec["checks"][f"{s}_known_classes"] = check(got == labels, "classes", got)
        if name == "npm3d":
            rec["feature_vocab"] = {"colors": ["beige", "gray", "gray-green"],
                                    "labels": labels}
        else:
            rec["feature_vocab"] = {"colors": list(mod.S3DIS_COLORS), "labels": labels}

        # split counts (manifest only; no cells or poses opened)
        man = read_json(rec["paths"]["manifest"])
        counts = {k: len(v) for k, v in man["splits"].items()}
        rec["split_counts"] = counts
        rec["checks"]["split_counts"] = check(
            counts.get("train") == cfg["expected_train_cells"]
            and counts.get("val") == cfg["expected_val_cells"]
            and counts.get("test") == cfg["expected_test_cells"], "counts", counts)
        rec["checks"]["split_disjoint"] = check(
            not (set(man["splits"]["train"]) & set(man["splits"]["val"])
                 or set(man["splits"]["train"]) & set(man["splits"]["test"])
                 or set(man["splits"]["val"]) & set(man["splits"]["test"])),
            "splits", counts)
        # bind what the corrected training actually loaded to this manifest
        data = ini["coarse"].get("data", {})
        # INIT_FREEZE records only the splits the stage loaded (TRAIN/VAL); the
        # loader sorts scene names, so bind by set identity and counts, not order.
        data_ok = set(data) == {"train", "val"} and all(
            data[s].get("n_cells") == counts.get(s)
            and set(data[s].get("scenes", [])) == set(man["splits"][s])
            for s in ("train", "val"))
        rec["checks"]["init_data_matches_manifest"] = check(
            data_ok, "data",
            {s: {"n": data.get(s, {}).get("n_cells"),
                 "scenes_equal": set(data.get(s, {}).get("scenes", []))
                                 == set(man["splits"][s])} for s in data})

        # historical TEST access, preserved verbatim
        tm = cfg["historical_test_marker"]
        rec["historical_test_marker"] = {
            "path": tm, "exists": os.path.isfile(tm),
            "sha256": sha256(tm) if os.path.isfile(tm) else None,
            "content": read_json(tm) if os.path.isfile(tm) else None,
        }
        rec["checks"]["historical_test_marker"] = check(os.path.isfile(tm), "marker", tm)

        # corrected backbone must differ from the historical old backbone
        hist = cfg["historical_ckpt"]
        found = []
        if os.path.isdir(hist):
            for dp, _dn, fns in os.walk(hist):
                for fn in fns:
                    if fn.endswith(".pth"):
                        found.append(os.path.join(dp, fn))
        rec["historical_ckpt_files"] = found[:10]
        rec["checks"]["historical_ckpt_distinct"] = check(
            not found or all(sha256(p) not in
                             (rec["stages"]["coarse"]["checkpoint_sha256"],
                              rec["stages"]["fine"]["checkpoint_sha256"])
                             for p in found), "files", found[:10])
        rec["checks"]["uses_corrected_run_root"] = check(
            os.path.abspath(rr).startswith(CORR), "root", rr)

        rec["all_pass"] = all(c["PASS"] for c in rec["checks"].values())
        domain_out[name] = rec

    binding = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gate": checks,
        "info": info,
        "domains": domain_out,
        "all_pass": all(c["PASS"] for c in checks.values())
                    and all(d["all_pass"] for d in domain_out.values()),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(binding, f, indent=2, ensure_ascii=False)
    print(json.dumps({
        "all_pass": binding["all_pass"],
        "gate": {k: v["PASS"] for k, v in checks.items()},
        "domains": {k: {"all_pass": v["all_pass"],
                        "fails": [kk for kk, vv in v["checks"].items() if not vv["PASS"]]}
                    for k, v in domain_out.items()},
    }, indent=2))
    print("saved", args.out, "sha", sha256(args.out))
    return 0 if binding["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
