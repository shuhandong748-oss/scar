#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TRAIN/VAL-only pre-TEST gate for one C domain.

No TEST file is opened.  Two kinds of VAL evidence are used:

  * a 64-query prefix is exported three times: frozen direct, chunked layout A
    and chunked layout B.  The prefix is compared with the *already frozen full
    VAL cache* (no new full inference) and the chunked export is re-run once to
    prove valid-chunk reuse;
  * the frozen full VAL cache is passed through the new feature wrapper and the
    new five-order s0 eval, and both are compared with the frozen reference
    outputs.  This exercises the actual frozen feature builder on all VAL
    queries without touching TEST and without another GPU export.

Every check is all-or-nothing.  GATE.json records test_read=false.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c_test_common as T  # noqa: E402

EXACT_CAND = ("cell_ids", "cell_scene", "cell_index", "gt_cell", "scene",
              "query_ids", "cell_bbox_min", "cell_center", "cell_size")
CLOSE_CAND = ("scores", "offsets", "pred_w")
FEATURE_EXACT = ("labels", "valid", "p2_idx", "p0_idx", "coarse_rank",
                 "cell_ids", "cell_scene", "gt_cell", "scene",
                 "feature_names", "split")
FEATURE_CLOSE = ("feats", "cos", "sp", "sc_center", "sc_fine", "sat_fine",
                 "offsets", "pred_w", "gt_pose_w")


def run(cmd, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("COMMAND " + " ".join(cmd) + "\n")
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        f.write(f"\nEXIT {p.returncode}\n")
    if p.returncode != 0:
        tail = open(log_path, encoding="utf-8").read()[-4000:]
        raise SystemExit(f"command failed: {' '.join(cmd)}\n{tail}")
    return log_path


def _one(x, y, k, exact, close, atol, diffs):
    if x.shape != y.shape:
        return f"{k}: shape {x.shape} != {y.shape}"
    if k in exact:
        if x.dtype.kind in "USO":
            return None if np.array_equal(x.astype(str), y.astype(str)) else k
        return None if np.array_equal(x, y) else k
    if k in close:
        d = float(np.max(np.abs(x.astype(np.float64)
                               - y.astype(np.float64))))
        diffs[k] = d
        return None if d <= atol else k
    return None if np.array_equal(x, y) else k


def cmp_arrays(a_path, b_path, exact, close, atol=1e-6):
    a = np.load(a_path, allow_pickle=True)
    b = np.load(b_path, allow_pickle=True)
    if set(a.files) != set(b.files):
        return [f"keys differ: {sorted(set(a.files) ^ set(b.files))}"], {}
    bad, diffs = [], {}
    for k in a.files:
        msg = _one(np.asarray(a[k]), np.asarray(b[k]), k, exact, close, atol,
                   diffs)
        if msg:
            bad.append(msg)
    return bad, diffs


def cmp_prefix(full_path, small_path, n, exact, close, atol=1e-6):
    full = np.load(full_path, allow_pickle=True)
    small = np.load(small_path, allow_pickle=True)
    if set(full.files) != set(small.files):
        return [f"keys differ: {sorted(set(full.files) ^ set(small.files))}"], {}
    bad, diffs = [], {}
    for k in full.files:
        x = np.asarray(full[k])
        if x.ndim:
            x = x[:n]
        msg = _one(x, np.asarray(small[k]), k, exact, close, atol, diffs)
        if msg:
            bad.append(msg)
    return bad, diffs


def head_logits(path, feature_names):
    import torch
    import torch.nn as nn
    ck = torch.load(path, map_location="cpu", weights_only=False)
    mean = np.asarray(ck["mean"], np.float32)
    std = np.asarray(ck["std"], np.float32)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(23, 64), nn.ReLU(),
                                     nn.Linear(64, 32), nn.ReLU(),
                                     nn.Linear(32, 1))

        def forward(self, z):
            return self.net(z).squeeze(-1)

    m = Net()
    m.load_state_dict(ck["state_dict"])
    m.eval()
    if list(ck["feature_names"]) != list(feature_names):
        raise SystemExit("head feature names mismatch")

    def f(x):
        with torch.no_grad():
            return m(torch.from_numpy((np.asarray(x, np.float32) - mean)
                                      / std)).numpy()
    return f, ck


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=sorted(T.DOMAINS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--freeze_list", default=None)
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--limit", type=int, default=64,
                    help="prefix size for the direct-vs-chunked export gate")
    ap.add_argument("--full_export", default=None)
    ap.add_argument("--full_features", default=None)
    ap.add_argument("--chunk_a", type=int, default=7)
    ap.add_argument("--chunk_b", type=int, default=13)
    ap.add_argument("--denominator_report", default=None)
    ap.add_argument("--python", default=None)
    a = ap.parse_args(argv)
    if a.limit <= 0:
        raise SystemExit("the gate requires a positive prefix size")
    out = os.path.abspath(a.out)
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; use a fresh gate directory")
    os.makedirs(out, exist_ok=False)
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    S = T.frozen_common()
    run_dir = os.path.abspath(a.run_dir or
                              os.path.join("/root/autodl-tmp/d4p_trainval_20261104/C",
                                           a.domain))
    freeze_path = os.path.abspath(a.freeze_list or
                                  os.path.join(run_dir, "FREEZE_LIST.json"))
    full_export = os.path.abspath(a.full_export or
                                  os.path.join(run_dir, "export", "val"))
    full_features = os.path.abspath(a.full_features or
                                    os.path.join(run_dir, "features", "val.npz"))
    freeze, finfo = T.freeze_info(freeze_path, a.domain)
    protocol_sha = finfo["protocol_sha256"]
    freeze_sha = finfo["freeze_sha256"]
    py = a.python or sys.executable
    checks, failures = [], []
    scar_before = T.sha256(os.path.join(T.FROZEN_CODE, "scar_common.py"))

    def chk(name, ok, detail=None):
        checks.append(dict(check=name, pass_=bool(ok), detail=detail))
        if not ok:
            failures.append(name)

    # ---- frozen inputs and prior evidence ------------------------------
    chk("freeze_canonical", finfo["canonical_sha256"] == freeze["freeze_sha256"])
    chk("freeze_domain", freeze["domain"] == a.domain)
    chk("protocol", finfo["protocol_sha256"]
        == freeze["protocol"]["protocol_sha256"])
    chk("main_head_s0", freeze["main_head"]["tag"] == "s0")
    chk("head_epoch",
        freeze["heads"]["s0"].get("formal_epoch") is not None,
        freeze["heads"]["s0"].get("formal_epoch"))
    chk("normalisation_sha", T.sha256(freeze["normalisation"]["path"])
        == freeze["normalisation"]["sha256"])
    norm = T.read_json(freeze["normalisation"]["path"])
    chk("norm_len23", len(norm["mean"]) == 23 and len(norm["std"]) == 23)
    chk("norm_train_only", "TRAIN" in norm.get("fitted_on", ""))
    chk("norm_train_npz", norm.get("train_npz_sha256")
        == freeze["features"]["train"]["sha256"])
    chk("norm_pinned_zero_std",
        all(float(norm["std"][j]) == 1.0 for j, p in enumerate(norm["pinned"])
            if p))
    for tag in ("coarse", "fine"):
        b = freeze["backbones"][tag]
        chk(f"{tag}_sha", T.sha256(b["path"]) == b["sha256"])
        chk(f"{tag}_epoch", b.get("epoch_ok") is True)
    chk("num_stats_sha", T.sha256(freeze["num_stats"]["path"])
        == freeze["num_stats"]["sha256"])
    chk("gate_pass", S.load_gate()["all_pass"] is True)
    s3prov = None
    if a.domain == "s3dis":
        rep = os.path.abspath(a.denominator_report or
                              os.path.join(T.TEST_ROOT, "records",
                                           "S3DIS_DENOMINATOR_542_544.json"))
        s3prov = T.s3dis_count_provenance(
            rep, os.path.join(S.domain_cfg("s3dis")["base"],
                              "dataset_manifest.json"))
        chk("s3dis_544_to_542_provenance",
            s3prov["old_guard"] == 544 and s3prov["effective"] == 542)
    chk("full_val_export_present",
        os.path.isfile(os.path.join(full_export, "EXPORT_META.json")))
    chk("full_val_features_present", os.path.isfile(full_features))
    fm_old = T.read_json(full_features + ".meta.json")
    em_full = T.read_json(os.path.join(full_export, "EXPORT_META.json"))
    n_full = int(em_full["n_queries"])
    chk("full_val_not_test", em_full.get("split") == "val"
        and em_full.get("test_opened") is False
        and fm_old.get("test_opened") is False)
    chk("full_val_queries", n_full == T.domain_info(a.domain)["expected_val"])
    chk("full_val_prefix_ok", a.limit <= n_full, dict(limit=a.limit, n=n_full))
    test_markers = [os.path.join(dp, f)
                    for dp, _, fns in os.walk(out)
                    for f in fns if f == "TEST_READ_ONCE.json"]
    chk("no_test_marker_written", not test_markers)

    # ---- 64-query prefix: frozen direct and two chunk layouts ----------
    direct = os.path.join(out, "direct_export")
    run([py, os.path.join(T.FROZEN_CODE, "d4p_export.py"),
         "--domain", a.domain, "--split", "val", "--out_dir", direct,
         "--protocol_sha", protocol_sha, "--freeze_sha", freeze_sha,
         "--limit_queries", str(a.limit)],
        os.path.join(out, "logs", "export_direct.log"))
    em_d = T.read_json(os.path.join(direct, "EXPORT_META.json"))
    chk("direct_n", em_d["n_queries"] == a.limit)
    chk("direct_val", em_d["split"] == "val"
        and em_d["test_opened"] is False)

    ca = os.path.join(out, "chunk_a")
    cb = os.path.join(out, "chunk_b")
    cmd_base = [py, os.path.join(HERE, "c_export_test_chunked.py"),
                "--domain", a.domain, "--split", "val",
                "--protocol_sha", protocol_sha, "--freeze_sha", freeze_sha,
                "--freeze_list", freeze_path, "--limit_queries", str(a.limit)]
    run(cmd_base + ["--out_dir", ca, "--chunk_queries", str(a.chunk_a)],
        os.path.join(out, "logs", "export_chunk_a.log"))
    run(cmd_base + ["--out_dir", cb, "--chunk_queries", str(a.chunk_b)],
        os.path.join(out, "logs", "export_chunk_b.log"))
    run(cmd_base + ["--out_dir", ca, "--chunk_queries", str(a.chunk_a)],
        os.path.join(out, "logs", "export_chunk_a_reuse.log"))
    chk("chunk_reuse", "EXPORT_REUSED" in open(
        os.path.join(out, "logs", "export_chunk_a_reuse.log"),
        encoding="utf-8").read())

    bad, d1 = cmp_arrays(os.path.join(direct, "candidates.npz"),
                         os.path.join(ca, "candidates.npz"),
                         EXACT_CAND, CLOSE_CAND)
    chk("direct_vs_chunk_candidates", not bad, dict(bad=bad, close=d1))
    bad2, d2 = cmp_arrays(os.path.join(ca, "candidates.npz"),
                          os.path.join(cb, "candidates.npz"),
                          EXACT_CAND, CLOSE_CAND)
    chk("chunk_layout_invariant_candidates", not bad2,
        dict(bad=bad2, close=d2))
    bad3, d3 = cmp_arrays(os.path.join(direct, "coarse_enc.npz"),
                          os.path.join(ca, "coarse_enc.npz"),
                          ("cell_ids",), ("tg", "cg"))
    chk("direct_vs_chunk_coarse_enc", not bad3, dict(bad=bad3, close=d3))
    bad4, d4 = cmp_arrays(os.path.join(ca, "coarse_enc.npz"),
                          os.path.join(cb, "coarse_enc.npz"),
                          ("cell_ids",), ("tg", "cg"))
    chk("chunk_layout_invariant_coarse_enc", not bad4,
        dict(bad=bad4, close=d4))
    for tag, p in (("direct", direct), ("chunk_a", ca), ("chunk_b", cb)):
        badp, dp = cmp_prefix(os.path.join(full_export, "candidates.npz"),
                              os.path.join(p, "candidates.npz"), a.limit,
                              EXACT_CAND, CLOSE_CAND, atol=1e-4)
        chk(f"{tag}_is_full_val_prefix", not badp, dict(bad=badp, close=dp))
        z = np.load(os.path.join(p, "candidates.npz"), allow_pickle=True)
        dec = z["cell_bbox_min"] + z["offsets"] * z["cell_size"][:, :, None]
        chk(f"{tag}_fine_decode",
            float(np.max(np.abs(dec - z["pred_w"]))) <= 1e-6,
            float(np.max(np.abs(dec - z["pred_w"]))))
        chk(f"{tag}_k25", np.asarray(z["cell_ids"]).shape[1] == 25)
        chk(f"{tag}_unique_candidates",
            all(len(set(map(str, r))) == 25 for r in z["cell_ids"]))
    # Prefix metric overlap (same cache definition on both paths).
    md = T.read_json(os.path.join(direct, "metrics.json"))
    mc = T.read_json(os.path.join(ca, "metrics.json"))
    for k in T.KS_CELL:
        chk(f"metric_cell_R{k}",
            abs(float(md["cell_R"][f"@{k}"]) - float(mc["cell_R"][f"@{k}"]))
            <= 1e-12)
    for k in T.KS_LOC:
        for t in (5.0, 10.0):
            key = f"Top{k}@{int(t)}m"
            chk(f"metric_loc_{key}",
                abs(float(md["localization"][key])
                    - float(mc["localization"][key])) <= 1e-12)

    # ---- frozen full VAL export through the new feature wrapper --------
    fr_dir = os.path.join(out, "features_rebuilt")
    os.makedirs(fr_dir)
    run([py, os.path.join(HERE, "c_features_test.py"),
         "--domain", a.domain, "--split", "val",
         "--protocol_sha", protocol_sha, "--freeze_sha", freeze_sha,
         "--export_dir", full_export,
         "--out", os.path.join(fr_dir, "val.npz")],
        os.path.join(out, "logs", "features_rebuilt.log"))
    badf, df = cmp_arrays(full_features,
                          os.path.join(fr_dir, "val.npz"),
                          FEATURE_EXACT, FEATURE_CLOSE, atol=1e-5)
    chk("features_match_frozen_frozen_val", not badf,
        dict(bad=badf, close=df))
    fm = T.read_json(os.path.join(fr_dir, "val.npz.meta.json"))
    chk("features_val_not_test", fm.get("test_opened") is False)
    chk("features_queries", fm.get("queries") == n_full)
    chk("features_min_match",
        (fm.get("relation_channel") or {}).get("min_match") == 4)
    chk("features_num_mentioned",
        (fm.get("relation_channel") or {}).get("num_mentioned")
        == (3 if a.domain == "npm3d" else 4))
    chk("features_names23", len(fm.get("features", [])) == 23)
    chk("features_cos_recomputed",
        (fm.get("cos_recomputed_from_coarse_enc") or {}).get("checked")
        == n_full)
    if a.domain == "npm3d":
        chk("npm3d_degenerate_reported",
            set((fm.get("relation_channel") or {})
                .get("degenerate_columns", []))
            == {"sc_center", "sc_fine", "sat_fine"})

    # ---- new eval and frozen reference eval on full VAL ----------------
    ev = os.path.join(out, "eval_five")
    run([py, os.path.join(HERE, "c_eval_test_five.py"),
         "--domain", a.domain, "--split", "val",
         "--export_dir", full_export,
         "--features", os.path.join(fr_dir, "val.npz"),
         "--out_dir", ev, "--freeze_list", freeze_path,
         "--protocol_sha", protocol_sha, "--freeze_sha", freeze_sha],
        os.path.join(out, "logs", "eval_five.log"))
    evf = os.path.join(out, "eval_frozen")
    run([py, os.path.join(T.FROZEN_CODE, "d4p_eval.py"),
         "--domain", a.domain, "--split", "val",
         "--export_dir", full_export,
         "--features", os.path.join(fr_dir, "val.npz"),
         "--heads_dir", os.path.join(run_dir, "heads"),
         "--out_dir", evf, "--protocol_sha", protocol_sha,
         "--freeze_sha", freeze_sha],
        os.path.join(out, "logs", "eval_frozen.log"))
    m5 = T.read_json(os.path.join(ev, "metrics.json"))
    mf = T.read_json(os.path.join(evf, "metrics.json"))
    for k in T.KS_CELL:
        chk(f"eval_matches_frozen_cell_R{k}",
            abs(float(m5["results_cell"]["scar_s0"][f"R@{k}"])
                - float(mf["results_cell"]["scar_s0"][f"R@{k}"])) <= 1e-12)
    for k in T.KS_LOC:
        for t in (5.0, 10.0):
            chk(f"eval_matches_frozen_loc_{k}_{int(t)}",
                abs(float(m5["results_loc"]["scar_s0"][f"Top{k}@{int(t)}m"])
                    - float(mf["results_loc"]["scar_s0"][
                        f"top{k}@{int(t)}m"])) <= 1e-12)

    # ---- local recomputation, saved/reload, primary contrast -----------
    feat = np.load(os.path.join(fr_dir, "val.npz"), allow_pickle=True)
    cand = np.load(os.path.join(full_export, "candidates.npz"),
                   allow_pickle=True)
    ids = np.asarray([[str(x) for x in r] for r in cand["cell_ids"]])
    cscene = np.asarray([[str(x) for x in r] for r in cand["cell_scene"]])
    gt_cell = [str(x) for x in cand["gt_cell"]]
    scene = np.asarray([str(x) for x in cand["scene"]])
    pred_w = np.asarray(cand["pred_w"], np.float64)
    gt_w = np.asarray(cand["gt_pose_w"], np.float64)
    feats = np.asarray(feat["feats"], np.float32)
    p2 = np.asarray(feat["p2_idx"], np.int64)
    cos = np.asarray(feat["cos"], np.float64)
    sp = np.asarray(feat["sp"], np.float64)
    scf = np.asarray(feat["sc_fine"], np.float64)
    sat = np.asarray(feat["sat_fine"], np.float64)
    fnames = [str(x) for x in feat["feature_names"]]
    hl, ck = head_logits(freeze["main_head"]["path"], fnames)
    chk("head_mean_std_match_norm",
        np.allclose(np.asarray(ck["mean"], np.float32),
                    np.asarray(norm["mean"], np.float32), rtol=0, atol=1e-7)
        and np.allclose(np.asarray(ck["std"], np.float32),
                        np.asarray(norm["std"], np.float32), rtol=0,
                        atol=1e-7))
    orders_saved = np.load(os.path.join(ev, "orders.npz"))["orders"]
    hits_saved = np.load(os.path.join(ev, "hits.npz"))
    bad_order = 0
    hc = {m: {k: np.zeros(n_full, np.int8) for k in T.KS_CELL}
          for m in T.METHODS}
    hlh = {m: {(k, t): np.zeros(n_full, np.int8)
               for k in T.KS_LOC for t in T.THRESHS} for m in T.METHODS}
    for i in range(n_full):
        cur = T.five_orders(feats[i], p2[i], cos[i], sp[i], scf[i], sat[i],
                            hl(feats[i][None, :, :])[0])
        for mi, m in enumerate(T.METHODS):
            o = cur[m]
            bad_order += int(not np.array_equal(o, orders_saved[i, :, mi]))
            ranked = [str(ids[i][j]) for j in o]
            for k in T.KS_CELL:
                hc[m][k][i] = int(gt_cell[i] in ranked[:k])
            d = np.linalg.norm(pred_w[i][o] - gt_w[i][None, :], axis=1)
            d[cscene[i][o] != scene[i]] = np.inf
            for k in T.KS_LOC:
                for t in T.THRESHS:
                    hlh[m][(k, t)][i] = int(np.min(d[:k]) <= t)
    chk("orders_saved_reload_match", bad_order == 0, bad_order)
    for m in T.METHODS:
        for k in T.KS_CELL:
            chk(f"hits_reload_{m}_R{k}",
                np.array_equal(hc[m][k], hits_saved[f"cell_{m}_R{k}"]))
        for k in T.KS_LOC:
            for t in T.THRESHS:
                chk(f"hits_reload_{m}_{k}_{int(t)}",
                    np.array_equal(hlh[m][(k, t)],
                                   hits_saved[f"loc_{m}_Top{k}@{int(t)}m"]))
    chk("R25_invariant",
        len({int(hc[m][25].sum()) for m in T.METHODS}) == 1)
    chk("Top25_5m_invariant",
        len({int(hlh[m][(25, 5.0)].sum()) for m in T.METHODS}) == 1)
    chk("no_s1_s2_eval", set(m5["results_cell"]) == set(T.METHODS))
    for name, role in (("SCAR_minus_fsr", "PRIMARY"),
                       ("SCAR_minus_original", "SECONDARY_EXPLORATORY"),
                       ("SCAR_minus_presence", "SECONDARY_EXPLORATORY"),
                       ("SCAR_minus_rank_fusion", "SECONDARY_EXPLORATORY")):
        chk(f"role_{name}", m5["comparisons"][name]["role"] == role)
    for metric in ("Cell_R@1", "Top1@5m"):
        chk(f"primary_interval_{metric}",
            "ci95" in m5["comparisons"]["SCAR_minus_fsr"]["metrics"][metric])

    # ---- final source/hash stability ----------------------------------
    scar_after = T.sha256(os.path.join(T.FROZEN_CODE, "scar_common.py"))
    chk("frozen_scar_common_unchanged", scar_after == scar_before,
        dict(before=scar_before, after=scar_after))
    if s3prov:
        chk("s3dis_frozen_source_sha_matches_provenance",
            s3prov["frozen_scar_common_sha256"] == scar_before)
    status = "PASS" if not failures else "FAIL"
    gate = dict(
        status=status, domain=a.domain, split="val",
        prefix_queries=a.limit, full_val_queries=n_full,
        test_read=False, test_armed=False, test_accessed=False,
        nature="TRAIN/VAL-only equivalence and input gate; no TEST payload",
        acceptance=[
            "the 64-query prefix from frozen direct equals both chunk layouts",
            "the prefix equals the first rows of the frozen full VAL cache",
            "valid chunks are reused (EXPORT_REUSED) without a forward pass",
            "Fine decode pred_w = bbox + offset*cell_size holds",
            "the frozen canonical-23 builder reproduces the frozen VAL "
            "features exactly; support/min_match are not lowered",
            "five orders, hits, primary intervals and saved/reload agree",
            "new s0-only eval agrees with the frozen reference eval",
            "no TEST marker/access occurs",
        ],
        checks=checks, failures=sorted(set(failures)),
        inputs=dict(freeze=finfo, freeze_file_sha256=T.sha256(freeze_path),
                    norm=norm, s3dis_provenance=s3prov,
                    full_export=full_export,
                    full_export_meta_sha256=T.sha256(
                        os.path.join(full_export, "EXPORT_META.json")),
                    full_features=full_features,
                    full_features_sha256=T.sha256(full_features),
                    frozen_code={f: T.sha256(os.path.join(T.FROZEN_CODE, f))
                                 for f in ("scar_common.py", "d4p_export.py",
                                           "d4p_features.py", "d4p_eval.py")}),
        runtime_seconds=round(time.time() - float(os.environ.get(
            "_GATE_T0", time.time())), 1),
        code_sha256={f: T.sha256(os.path.join(HERE, f))
                     for f in ("c_gate_val_small.py", "c_test_common.py",
                               "c_export_test_chunked.py", "c_features_test.py",
                               "c_eval_test_five.py", "c_verify_test.py")},
        next_step="stop; await the user-supplied online connection before TEST",
    )
    T.write_json(gate, os.path.join(out, "GATE.json"))
    lines = []
    for dp, dns, fns in os.walk(out):
        dns[:] = sorted(d for d in dns if d != "__pycache__")
        for n in sorted(fns):
            p = os.path.join(dp, n)
            lines.append(f"{T.sha256(p)}  {os.path.relpath(p, out)}")
    with open(os.path.join(out, "SHA256SUMS.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps(dict(status=status, n_checks=len(checks),
                          failures=sorted(set(failures))), indent=2), flush=True)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
