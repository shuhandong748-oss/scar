#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TRAIN/VAL-only pre-TEST gate for one A domain.

No TEST file is opened.  The gate re-runs a small slice of the frozen VAL
pipeline twice -- once through the frozen direct exporter, once through the new
chunked TEST exporter -- and checks that identity, order, scores, Fine decode,
coarse encodings, features, the in-process coordinate fix, support accounting,
the five rankings and save/reload all agree.

It also re-runs the frozen feature wrapper and the TEST-enabled wrapper on the
same limited export, and the old-defect variant, so the exact scope of the fix
is demonstrated: only columns 19/20 may change.

Acceptance is all-or-nothing.  The gate never arms a TEST marker, never reads
TEST, never trains or reselects anything.
"""
import argparse
import csv
import json
import os
import subprocess
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
METHODS = ["original", "presence", "rank_fusion", "fsr", "scar_s0"]
AFFECTED = ("dist_nearest_matched", "dist_centroid_matched")
UNCHANGED_ARRAYS = [
    "labels", "valid", "p2_idx", "cell_ids", "gt_cell", "scene", "gt_pose_w",
    "cos", "sp", "sc_center", "sc_fine", "sat_fine", "offsets", "pred_w",
    "feature_names", "split", "query_ids", "source_pose_index", "order_index",
    "num_mentions", "n_parsed",
]


def run(cmd, log_path, env):
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("CMD " + json.dumps(cmd) + "\n")
        f.flush()
        t0 = time.time()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        f.write(f"\nEXIT {p.returncode} SECONDS {time.time()-t0:.1f}\n")
    if p.returncode:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            tail = f.read()[-4000:]
        raise SystemExit(f"command failed rc={p.returncode}: {cmd[1]}\n{tail}")
    return log_path


def cmp_arrays(a, b, exact_names, close_names, atol=1e-6):
    """Return (exact_ok, close_ok, detail) for two npz mappings."""
    exact, close, bad = {}, {}, []
    for k in exact_names:
        if k in a and k in b:
            exact[k] = bool(np.array_equal(a[k], b[k]))
            if not exact[k]:
                bad.append(f"{k}: not identical")
        else:
            exact[k] = None
    for k in close_names:
        if k in a and k in b:
            x = np.asarray(a[k], np.float64)
            y = np.asarray(b[k], np.float64)
            if x.shape != y.shape:
                close[k] = False
                bad.append(f"{k}: shape {x.shape} vs {y.shape}")
                continue
            d = float(np.max(np.abs(x - y))) if x.size else 0.0
            close[k] = bool(np.allclose(x, y, rtol=0.0, atol=atol))
            if not close[k]:
                bad.append(f"{k}: max|d|={d:g}")
        else:
            close[k] = None
    return exact, close, bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=sorted(S.DOMAINS))
    ap.add_argument("--freeze_list", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--coarse_ckpt", required=True)
    ap.add_argument("--fine_ckpt", required=True)
    ap.add_argument("--known_classes", required=True)
    ap.add_argument("--ste_stage1_path", required=True)
    ap.add_argument("--heads_dir", required=True)
    ap.add_argument("--frozen_code_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--chunk_size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pythonpath", default="")
    a = ap.parse_args(argv)

    out = os.path.abspath(a.out_dir)
    if os.path.isdir(out) and os.listdir(out):
        raise SystemExit(f"gate out_dir is not empty: {out}")
    os.makedirs(out, exist_ok=True)
    logs = os.path.join(out, "logs")
    os.makedirs(logs, exist_ok=True)
    work = os.path.join(out, "work")
    os.makedirs(work, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    pp = [os.path.abspath(a.frozen_code_dir)]
    if a.pythonpath:
        pp += [os.path.abspath(p) for p in a.pythonpath.split(os.pathsep) if p]
    if env.get("PYTHONPATH"):
        pp.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pp)

    checks, failures = [], []

    def chk(name, ok, detail=None):
        checks.append(dict(check=name, pass_=bool(ok), detail=detail))
        if not ok:
            failures.append(name)

    # ---------------- inputs ------------------------------------------------
    freeze, fchk = S.verify_freeze(a.freeze_list, a.domain,
                                   EXPECTED_EPOCHS[a.domain])
    chk("freeze_canonical", fchk["PASS"] is True)
    chk("protocol", fchk["protocol_sha256"]
        == freeze["protocol"]["protocol_sha256"])
    chk("feature_version", fchk["feature_version"] == S.FEATURE_VERSION)
    chk("freeze_scope_test_not_read",
        freeze.get("scope", {}).get("test_read") is False
        and freeze.get("scope", {}).get("test_armed") is False)
    hinfo = S.verify_heads_and_norm(freeze, a.freeze_list)
    chk("head_sha", S.sha256(hinfo["head"]["path"])
        == freeze["main_head"]["sha256"])
    chk("head_epoch", hinfo["head"]["formal_epoch"]
        == freeze["heads"]["s0"]["formal_epoch"])

    norm = S.read_json(hinfo["normalisation"]["path"])
    mean = np.asarray(norm["mean"], np.float32)
    std = np.asarray(norm["std"], np.float32)
    pinned = np.asarray(norm["pinned"], bool)
    chk("norm_len23", mean.shape == (23,) and std.shape == (23,)
        and pinned.shape == (23,))
    chk("norm_train_only", "TRAIN" in str(norm.get("fitted_on", "")))
    chk("norm_pinned_std_one",
        bool(np.all(std[pinned] == 1.0)))
    chk("norm_pinned_names", list(norm.get("pinned_names", []))
        == [str(x) for x, p in zip(freeze["features"]["val"]["meta"]["features"],
                                   pinned) if p])
    if "train_npz_sha256" in norm and "train" in freeze.get("features", {}):
        chk("norm_bound_to_frozen_train",
            norm["train_npz_sha256"] == freeze["features"]["train"]["sha256"])
    chk("norm_freeze_sha", hinfo["normalisation"]["sha256"]
        == freeze["normalisation"]["sha256"])
    chk("normalisation_pinned_freeze",
        list(freeze["normalisation"].get("pinned", []))
        == list(norm.get("pinned_names", [])))

    head = S.Head(hinfo["head"]["path"],
                  expected_sha=freeze["main_head"]["sha256"])
    chk("head_feature_version", head.feature_version == S.FEATURE_VERSION)
    chk("head_norms", head.nf == 23)
    chk("head_norm_equals_normalisation",
        np.array_equal(head.mean, mean) and np.array_equal(head.std, std))
    chk("head_names_frozen",
        list(head.feature_names) == list(freeze["features"]["val"]
                                         ["meta"]["features"]))

    cfg = S.read_json(a.config)
    chk("config_exists", os.path.isfile(a.config))
    known = S.read_json(a.known_classes)["known_classes"]
    chk("known_classes_nonempty", len(known) > 0)
    chk("ste_exists", os.path.isfile(a.ste_stage1_path))
    patch_path = os.path.join(os.path.abspath(a.frozen_code_dir),
                              "feature_coord_fix_v1.py")
    chk("patch_module_exists", os.path.isfile(patch_path))
    if "feature_coord_fix_v1.py" in freeze.get("code_sha256", {}):
        chk("patch_module_sha", S.sha256(patch_path)
            == freeze["code_sha256"]["feature_coord_fix_v1.py"])
    hist = S.DOMAINS[a.domain]["historical_access"]
    chk("historical_access_record", os.path.isfile(hist))
    chk("historical_access_sha_recorded",
        bool(S.DOMAINS[a.domain]["historical_statement"]))

    # ---------------- exports: frozen direct vs chunked --------------------
    direct = os.path.join(work, "direct_val")
    chunked = os.path.join(work, "chunked_val")
    common = ["--domain", a.domain, "--split", "val",
              "--config", a.config,
              "--coarse_ckpt", a.coarse_ckpt, "--fine_ckpt", a.fine_ckpt,
              "--known_classes", a.known_classes,
              "--ste_stage1_path", a.ste_stage1_path,
              "--limit_queries", str(a.limit), "--device", a.device,
              "--seed", str(a.seed)]
    frozen_export = os.path.join(os.path.abspath(a.frozen_code_dir),
                                 "export_candidates_coordfix_v1.py")
    run([sys.executable, frozen_export] + common
        + ["--out_dir", direct], os.path.join(logs, "export_direct.log"), env)
    run([sys.executable, os.path.join(HERE, "a_export_test_chunked.py")]
        + common + ["--freeze_list", a.freeze_list,
                    "--out_dir", chunked, "--chunk_size", str(a.chunk_size),
                    "--frozen_export", frozen_export],
        os.path.join(logs, "export_chunked.log"), env)
    run([sys.executable, os.path.join(HERE, "a_export_test_chunked.py")]
        + common + ["--freeze_list", a.freeze_list,
                    "--out_dir", chunked, "--chunk_size", str(a.chunk_size),
                    "--frozen_export", frozen_export],
        os.path.join(logs, "export_chunked_reuse.log"), env)
    with open(os.path.join(logs, "export_chunked_reuse.log"),
              encoding="utf-8") as f:
        chk("export_reuse_no_recompute", "EXPORT_REUSED" in f.read())

    de = np.load(os.path.join(direct, "candidates.npz"), allow_pickle=True)
    ce = np.load(os.path.join(chunked, "candidates.npz"), allow_pickle=True)
    den = np.load(os.path.join(direct, "coarse_enc.npz"), allow_pickle=True)
    cen = np.load(os.path.join(chunked, "coarse_enc.npz"), allow_pickle=True)
    n = len(ce["cell_ids"])
    chk("gate_limit_honoured", n == a.limit and len(de["cell_ids"]) == n,
        dict(direct=len(de["cell_ids"]), chunked=n))
    ex, cl, bad = cmp_arrays(
        {k: de[k] for k in de.files}, {k: ce[k] for k in ce.files},
        ["cell_ids", "gt_cell", "gt_pose_w", "query_index", "split"],
        ["offsets", "pred_w", "coarse_cos", "scores"], atol=1e-6)
    chk("export_candidate_identity", all(v is not False for v in ex.values()), ex)
    chk("export_numeric_match", all(v is not False for v in cl.values()), cl)
    chk("export_cell_ids_unique",
        all(len(set(str(c) for c in row)) == 25 for row in ce["cell_ids"]))
    chk("export_query_index_arange",
        np.array_equal(np.asarray(ce["query_index"]), np.arange(n)))
    chk("export_source_pose_index",
        np.array_equal(np.asarray(ce["source_pose_index"]), np.arange(n)))
    chk("export_split_val", str(ce["split"][0]) == "val")
    chk("export_marker_not_opened",
        S.read_json(os.path.join(chunked, "export_meta.json"))
        .get("test_opened") is False)
    chk("direct_meta_no_test",
        S.read_json(os.path.join(direct, "export_meta.json"))
        .get("test_read") is False)

    bbox = np.asarray(ce["cell_bbox_min"], np.float64)
    size = np.asarray(ce["cell_size"], np.float64)
    offs = np.asarray(ce["offsets"], np.float64)
    pred = np.asarray(ce["pred_w"], np.float64)
    decode_max = float(np.max(np.abs(pred - (bbox + offs * size[..., None]))))
    # The frozen exporter stores offsets as float32 and pred_w as float64
    # computed from the pre-cast offsets, so re-decoding from the stored
    # float32 offsets can differ by up to one float32 ulp scaled by the cell
    # size.  The contract is therefore checked at that storage precision.
    decode_tol = float(np.finfo(np.float32).eps
                       * np.max(np.abs(offs * size[..., None])))
    chk("fine_decode_contract", decode_max <= decode_tol + 1e-12,
        dict(max_abs_diff=decode_max, tol=decode_tol))
    chk("fine_offsets_finite", bool(np.isfinite(offs).all()))
    chk("pred_w_finite", bool(np.isfinite(pred).all()))

    ex, cl, bad = cmp_arrays(
        {k: den[k] for k in den.files}, {k: cen[k] for k in cen.files},
        ["cell_ids", "split"], ["tg", "cg"], atol=1e-6)
    chk("coarse_cell_ids", ex.get("cell_ids") is True)
    chk("coarse_enc_match", all(v is not False for v in cl.values()), cl)

    comp = S.read_json(os.path.join(chunked, "EXPORT_COMPLETE.json"))
    for name in ("candidates.npz", "coarse_enc.npz"):
        chk(f"export_cache_{name}",
            S.sha256(os.path.join(chunked, name)) == comp["cache_sha256"][name])
    manifest = S.read_json(os.path.join(chunked, "export_meta.json"))[
        "chunk_manifest"]
    chunk_ok = []
    for rec in manifest:
        p = os.path.join(chunked, "chunks", rec["name"])
        sc = S.read_json(p[:-4] + ".json")
        z = np.load(p, allow_pickle=True)
        chunk_ok.append(
            sc.get("status") == "complete"
            and sc.get("sha256") == S.sha256(p)
            and sc.get("start") == rec["start"] and sc.get("end") == rec["end"]
            and z["query_index"].tolist() == list(range(rec["start"], rec["end"]))
            and sc.get("coarse_sha256") == S.sha256(a.coarse_ckpt)
            and sc.get("fine_sha256") == S.sha256(a.fine_ckpt)
            and sc.get("ste_sha256") == S.sha256(a.ste_stage1_path)
            and sc.get("freeze_sha256") == fchk["freeze_canonical_sha256"]
            and sc.get("seed") == a.seed and sc.get("k") == 25)
    chk("chunk_manifest_valid", all(chunk_ok), dict(chunks=len(manifest)))
    chk("chunk_count", len(manifest) == (n + a.chunk_size - 1) // a.chunk_size)

    # ---------------- features: wrapper equivalence ------------------------
    frozen_wrapper = os.path.join(os.path.abspath(a.frozen_code_dir),
                                  "build_ht_t2l_coordfix_v1.py")
    feat_direct = os.path.join(out, "features_direct_frozen.npz")
    feat_chunk = os.path.join(out, "features_chunk_frozen.npz")
    feat_ours = os.path.join(out, "features_chunk_ours.npz")
    feat_old = os.path.join(out, "features_chunk_old_defect.npz")
    base = ["--domain", a.domain, "--split", "val", "--limit", str(a.limit)]
    run([sys.executable, frozen_wrapper] + base
        + ["--export_dir", direct, "--out", feat_direct],
        os.path.join(logs, "features_direct_frozen.log"), env)
    run([sys.executable, frozen_wrapper] + base
        + ["--export_dir", chunked, "--out", feat_chunk],
        os.path.join(logs, "features_chunk_frozen.log"), env)
    run([sys.executable, os.path.join(HERE, "a_features_test_coordfix_v1.py")]
        + base + ["--export_dir", chunked, "--out", feat_ours,
                  "--frozen_code_dir", os.path.abspath(a.frozen_code_dir)],
        os.path.join(logs, "features_chunk_ours.log"), env)
    run([sys.executable, frozen_wrapper] + base
        + ["--export_dir", chunked, "--out", feat_old, "--old-defect"],
        os.path.join(logs, "features_chunk_old.log"), env)

    fd = np.load(feat_direct, allow_pickle=True)
    fc = np.load(feat_chunk, allow_pickle=True)
    fo = np.load(feat_ours, allow_pickle=True)
    f_old = np.load(feat_old, allow_pickle=True)
    ex, cl, bad = cmp_arrays(
        {k: fd[k] for k in fd.files}, {k: fc[k] for k in fc.files},
        ["labels", "valid", "p2_idx", "cell_ids", "gt_cell", "scene",
         "gt_pose_w", "feature_names", "split", "query_ids",
         "source_pose_index", "order_index", "num_mentions", "n_parsed"],
        ["feats", "cos", "sp", "sc_center", "sc_fine", "sat_fine", "offsets",
         "pred_w"], atol=1e-5)
    chk("features_direct_vs_chunked_exact",
        all(v is not False for v in ex.values()), ex)
    chk("features_direct_vs_chunked_close",
        all(v is not False for v in cl.values()), cl)

    wrapper_names = ["labels", "valid", "p2_idx", "cell_ids", "gt_cell",
                     "scene", "gt_pose_w", "feature_names", "split",
                     "query_ids", "source_pose_index", "order_index",
                     "num_mentions", "n_parsed", "feats", "cos", "sp",
                     "sc_center", "sc_fine", "sat_fine", "offsets", "pred_w"]
    chk("wrapper_key_sets_identical", sorted(fc.files) == sorted(fo.files),
        dict(frozen=sorted(fc.files), ours=sorted(fo.files)))
    ex, cl, bad = cmp_arrays(
        {k: fc[k] for k in fc.files}, {k: fo[k] for k in fo.files},
        [k for k in wrapper_names if k in fc.files], [])
    chk("wrapper_bit_equal", all(v is True for v in ex.values()),
        {k: v for k, v in ex.items() if v is not True})

    meta_o = S.read_json(feat_ours + ".meta.json")
    chk("feature_version", meta_o.get("feature_version") == S.FEATURE_VERSION)
    chk("feature_count", int(len(fo["feats"])) == n
        and fo["feats"].shape[1:] == (25, 23))
    names = [str(x) for x in fo["feature_names"]]
    chk("feature_names_canonical",
        names == list(freeze["features"]["val"]["meta"]["features"])
        and names == list(head.feature_names))
    chk("wrapper_export_binding",
        meta_o.get("export_meta_sha256")
        == S.sha256(os.path.join(chunked, "export_meta.json")))
    patch = meta_o.get("coord_fix", {})
    chk("patch_applied", patch.get("applied") is True)
    chk("patch_only_function",
        patch.get("patched_function") == "build_cell_aux")
    chk("patch_old_source",
        patch.get("common_build_cell_aux_source_sha_before")
        == freeze["coord_fix"]["old_source_sha256"])
    chk("patch_new_source",
        patch.get("common_build_cell_aux_source_sha_after")
        == freeze["coord_fix"]["new_source_sha256"])
    chk("common_module_untouched",
        patch.get("common_module_sha256") == freeze["coord_fix"]["common_sha256"])
    chk("patch_module_sha_value",
        patch.get("feature_coord_fix_module_sha256") == S.sha256(patch_path))
    chk("test_not_opened_features", meta_o.get("test_opened") is False)

    # old vs new: only columns 19/20 may differ
    old = {k: f_old[k] for k in f_old.files}
    new = {k: fo[k] for k in fo.files}
    unchanged = {}
    for k in UNCHANGED_ARRAYS:
        if k in old and k in new:
            unchanged[k] = bool(np.array_equal(old[k], new[k]))
    chk("old_new_unchanged_fields",
        all(unchanged.values()),
        {k: v for k, v in unchanged.items() if not v})
    of = np.asarray(old["feats"], np.float64)
    nf = np.asarray(new["feats"], np.float64)
    col_exact = {nm: bool(np.array_equal(of[..., j], nf[..., j]))
                 for j, nm in enumerate(names)}
    non_affected = all(v for nm, v in col_exact.items() if nm not in AFFECTED)
    changed_mask = np.zeros(nf.shape[:2], bool)
    for nm in AFFECTED:
        j = names.index(nm)
        changed_mask |= np.abs(of[..., j] - nf[..., j]) > 0
    chk("only_affected_columns_change", non_affected,
        {nm: v for nm, v in col_exact.items() if nm in AFFECTED})
    chk("affected_columns_do_change", int(changed_mask.sum()) > 0,
        dict(changed_rows=int(changed_mask.sum())))
    chk("features_finite", bool(np.isfinite(nf).all()))

    # support accounting
    nm_j = names.index("n_matched")
    n_matched = np.asarray(fo["feats"][:, :, nm_j], np.float64)
    if "num_mentions" in fo.files:
        n_q = np.asarray(fo["num_mentions"], np.int64)
        n_q_source = "features.num_mentions"
    else:
        # Fixed-arity domains (PandaSet: 6 descriptions per query) store no
        # per-query count in the feature archive; take the frozen registry.
        from adapt.domain import get as _domain_get
        _arity = list(_domain_get(a.domain).get("arity") or [])
        if len(_arity) != 1:
            raise SystemExit("%s: feature archive has no num_mentions and the "
                             "domain arity is not constant: %r"
                             % (a.domain, _arity))
        n_q = np.full(int(np.asarray(fo["feats"]).shape[0]),
                      int(_arity[0]), np.int64)
        n_q_source = "adapt.domain.get(%r).arity[0]=%d" % (a.domain, _arity[0])
    m = np.rint(n_matched * n_q[:, None]).astype(np.int64)
    min_match = int(freeze["features"]["constants"]["min_match"])
    p2 = np.asarray(fo["p2_idx"], np.int64)
    chk("num_mentions_positive", bool((n_q > 0).all()), dict(n_q=int(n_q.sum())))
    chk("n_matched_unit_range",
        bool((n_matched >= -1e-6).all() and (n_matched <= 1 + 1e-6).all()))
    chk("m_bounds", bool((m >= 0).all() and (m <= n_q[:, None]).all()))
    chk("p2_is_permutation",
        p2.shape == (n, 25)
        and all(sorted(row.tolist()) == list(range(25)) for row in p2))
    support = dict(
        n_mentions_source=n_q_source,
        n_queries=int(n), n_mentioned_total=int(n_q.sum()),
        n_mentioned_min=int(n_q.min()), n_mentioned_max=int(n_q.max()),
        m_min=int(m.min()), m_max=int(m.max()),
        min_match=min_match,
        candidates_meeting_gate=int((m >= min_match).sum()),
        candidates_total=int(m.size),
        queries_with_any_ge_gate=int((m >= min_match).any(1).sum()),
        frac_candidates_meeting_gate=float((m >= min_match).mean()),
        valid_lists=int(np.asarray(fo["valid"]).sum()),
        invalid_lists=int(n - np.asarray(fo["valid"]).sum()),
        p2_permutation=True,
    )
    chk("support_reported", support["n_queries"] == n)

    # ---------------- five rankings and save/reload ------------------------
    def five_orders(feats, cos, sp, scf, sat, p2, head):
        lg = head.logits(feats)
        out = np.empty((len(feats), 25, len(METHODS)), np.int16)
        for i in range(len(feats)):
            p2i = np.asarray(p2[i], np.int64)
            o_scf = np.argsort(-scf[i][p2i], kind="stable")
            r_scf = np.empty(25, np.int64)
            r_scf[p2i[o_scf]] = np.arange(25)
            borda = (np.arange(25, dtype=np.float64) + r_scf[p2i]) / 48.0
            cur = {
                "original": np.arange(25),
                "presence": np.argsort(-(cos[i] + 0.5 * sp[i]), kind="stable"),
                "rank_fusion": p2i[np.argsort(borda, kind="stable")],
                "fsr": p2i[np.argsort(-sat[i][p2i], kind="stable")],
                "scar_s0": S.rank_by_logits(lg[i], p2i),
            }
            for mi, mname in enumerate(METHODS):
                out[i, :, mi] = cur[mname]
        return out, lg

    def load_feat(z, head):
        return (np.asarray(z["feats"], np.float32),
                np.asarray(z["cos"], np.float64),
                np.asarray(z["sp"], np.float64),
                np.asarray(z["sc_fine"], np.float64),
                np.asarray(z["sat_fine"], np.float64),
                np.asarray(z["p2_idx"], np.int64),
                head)

    ord_ours, lg_ours = five_orders(*load_feat(fo, head))
    ord_frozen, lg_frozen = five_orders(*load_feat(fc, head))
    chk("five_orders_match_frozen_wrapper",
        np.array_equal(ord_ours, ord_frozen))
    chk("logits_match_frozen_wrapper",
        np.allclose(lg_ours, lg_frozen, rtol=0.0, atol=1e-6))
    ord_direct, _ = five_orders(*load_feat(fd, head))
    chk("five_orders_direct_path", np.array_equal(ord_ours, ord_direct))
    dirp = os.path.join(out, "orders_gate.npz")
    np.savez_compressed(dirp, orders=ord_ours,
                        methods=np.asarray(METHODS, dtype="<U32"),
                        query_ids=np.asarray(ce["query_ids"], dtype="<U64"))
    reloaded = np.load(dirp)["orders"]
    chk("orders_save_reload", np.array_equal(reloaded, ord_ours))

    ids = np.asarray([[str(c) for c in row] for row in fo["cell_ids"]])
    gt_cell = [str(x) for x in fo["gt_cell"]]
    scene = np.asarray([str(x) for x in fo["scene"]])
    cscene = np.asarray([[str(c) for c in row] for row in ce["cell_scene"]])
    pred_w = np.asarray(fo["pred_w"], np.float64)
    gt_w = np.asarray(fo["gt_pose_w"], np.float64)
    hits_c = {mm: {k: np.zeros(n, np.int8) for k in S.KS_CELL}
              for mm in METHODS}
    hits_l = {mm: {(k, t): np.zeros(n, np.int8)
                   for k in S.KS_LOC for t in S.THRESHS} for mm in METHODS}
    for i in range(n):
        for mi, mm in enumerate(METHODS):
            o = ord_ours[i, :, mi]
            for k, v in S.cell_hits(ids[i], o, gt_cell[i]).items():
                hits_c[mm][k][i] = v
            lh, _ = S.loc_hits(pred_w[i], o, gt_w[i], cscene[i], scene[i])
            for key, v in lh.items():
                hits_l[mm][key][i] = v
    res_c, res_l = S.metrics_from_hits(hits_c, hits_l, METHODS, n)
    mono = S.check_monotonicity(res_c, res_l, METHODS)
    chk("gate_metrics_monotonic", not mono, mono)
    chk("gate_R25_equal",
        len({round(res_c[mm]["R@25"], 12) for mm in METHODS}) == 1)
    chk("gate_Top25_5m_equal",
        len({round(res_l[mm]["Top25@5m"], 12) for mm in METHODS}) == 1)
    keys = scene.reshape(-1, 1)
    boot1 = S.bootstrap_ci(hits_c["scar_s0"][1], hits_c["fsr"][1], keys,
                           resamples=200, seed=S.SEED)
    boot2 = S.bootstrap_ci(hits_c["scar_s0"][1], hits_c["fsr"][1], keys,
                           resamples=200, seed=S.SEED)
    chk("bootstrap_save_reload_deterministic",
        boot1["ci95"] == boot2["ci95"] and boot1["delta"] == boot2["delta"])

    metrics = dict(
        split="val", n_queries=int(n),
        n_clusters=int(len(set(scene.tolist()))),
        results_cell=res_c, results_loc=res_l,
        bootstrap_probe=dict(metric="Cell_R@1", scar_minus_fsr=boot1,
                             resamples=200),
        invariant=dict(all_queries=True, no_filtering=True,
                       R25_equal=True, Top25_5m_equal=True, monotonic=True),
        note="small-sample gate only; not a VAL result and not a TEST result",
    )
    S.write_json(metrics, os.path.join(out, "gate_metrics.json"))

    # ---------------- summary ---------------------------------------------
    payload = dict(
        status="PASS" if not failures else "FAIL",
        domain=a.domain, split="val", limit=int(a.limit),
        chunk_size=int(a.chunk_size), seed=int(a.seed),
        test_read=False, test_armed=False,
        purpose=("pre-TEST TRAIN/VAL small-sample gate: direct exporter vs "
                 "chunked TEST exporter, frozen wrapper vs TEST wrapper, "
                 "old-defect vs FEATURE_COORD_FIX_v1, five orders, save/reload"),
        input=dict(
            freeze=dict(path=a.freeze_list, sha256=S.sha256(a.freeze_list),
                        canonical_sha256=freeze["freeze_sha256"],
                        protocol_sha256=freeze["protocol"]["protocol_sha256"]),
            config=dict(path=a.config, sha256=S.sha256(a.config)),
            known_classes=dict(path=a.known_classes,
                               sha256=S.sha256(a.known_classes)),
            coarse=dict(path=a.coarse_ckpt, sha256=S.sha256(a.coarse_ckpt)),
            fine=dict(path=a.fine_ckpt, sha256=S.sha256(a.fine_ckpt)),
            ste_stage1=dict(path=a.ste_stage1_path,
                            sha256=S.sha256(a.ste_stage1_path)),
            head=dict(path=hinfo["head"]["path"], sha256=head.sha256,
                      formal_epoch=hinfo["head"]["formal_epoch"]),
            normalization=dict(path=hinfo["normalisation"]["path"],
                               sha256=hinfo["normalisation"]["sha256"],
                               pinned=list(norm.get("pinned_names", [])),
                               fitted_on=norm.get("fitted_on")),
            patch_module=dict(path=patch_path, sha256=S.sha256(patch_path)),
            historical_access=dict(path=hist, sha256=S.sha256(hist),
                                   statement=S.DOMAINS[a.domain]
                                   ["historical_statement"]),
        ),
        export=dict(
            direct_dir=direct, chunked_dir=chunked, n_queries=int(n),
            direct_candidate_sha256=S.sha256(
                os.path.join(direct, "candidates.npz")),
            chunked_candidate_sha256=S.sha256(
                os.path.join(chunked, "candidates.npz")),
            chunked_cache=comp["cache_sha256"],
            chunk_manifest_sha256=comp["chunk_manifest_sha256"],
            decode_max_abs_diff=decode_max,
            reuse_no_recompute=True,
        ),
        features=dict(
            direct_frozen=dict(path=feat_direct, sha256=S.sha256(feat_direct)),
            chunk_frozen=dict(path=feat_chunk, sha256=S.sha256(feat_chunk)),
            chunk_ours=dict(path=feat_ours, sha256=S.sha256(feat_ours),
                            meta_sha256=S.sha256(feat_ours + ".meta.json")),
            old_defect=dict(path=feat_old, sha256=S.sha256(feat_old)),
            feature_names=list(names),
            feature_names_sha256=meta_o.get("feature_names_sha256"),
            patch=patch, support=support,
            changed_candidate_rows=int(changed_mask.sum()),
            changed_candidate_frac=float(changed_mask.mean()),
            unchanged_fields=unchanged,
            affected_only=non_affected and int(changed_mask.sum()) > 0,
        ),
        orders=dict(path=dirp, sha256=S.sha256(dirp),
                    methods=METHODS, reload_ok=True,
                    match_frozen_wrapper=True, match_direct_path=True),
        metrics_file=dict(path=os.path.join(out, "gate_metrics.json"),
                          sha256=S.sha256(os.path.join(out,
                                                       "gate_metrics.json"))),
        checks=checks, failures=sorted(set(failures)),
        gate_code_sha256={f: S.sha256(os.path.join(HERE, f))
                          for f in ("a_gate_val_small.py", "a_test_common.py",
                                    "a_export_test_chunked.py",
                                    "a_features_test_coordfix_v1.py")},
        next_step="stop; await user-supplied online connection before TEST",
        created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    S.write_json(payload, os.path.join(out, "GATE.json"))
    lines = []
    for dp, dns, fns in os.walk(out):
        dns[:] = sorted(d for d in dns if d != "__pycache__")
        for n in sorted(fns):
            p = os.path.join(dp, n)
            lines.append(f"{S.sha256(p)}  {os.path.relpath(p, out)}")
    with open(os.path.join(out, "SHA256SUMS.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps(dict(status=payload["status"],
                          domain=a.domain, n_checks=len(checks),
                          failures=payload["failures"]), indent=2),
          flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
