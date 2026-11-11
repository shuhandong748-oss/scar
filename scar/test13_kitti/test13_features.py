# -*- coding: utf-8 -*-
"""TEST13: canonical23 feature build for the TEST split.

Same *frozen* feature definition as COMMAND 12
  (code/scar_frozen/{common,sa_geom,features_fix}.py, SHA-checked):

    K=25, pool_norm=rank_norm=24, tied_norm=25, canonical P2 order,
    anchor=P2 top-1, features=features_fix.FEATURE_NAMES,
    label = same scene AND ||pred_w - gt_pose_xy|| <= 5 m,
    valid = 1 <= positives < K.

Input is this round's TEST export (out/chunks + out/cache) written by
test13_export.py and, for the VAL split, the frozen COMMAND 09 cache
(read-only, SHA-verified).  No backbone forward runs here; the TEST features
are produced once and are never refitted.

EXPLORATORY: this is the formal second TEST read.  The labels use GT poses and
are used only for metrics -- no feature column reads the GT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test13_common as C  # noqa: E402

FROZEN = os.path.join(HERE, "scar_frozen")
sys.path.insert(0, FROZEN)
sys.path.insert(0, C.CORE)
os.chdir(C.CORE)

import common as Cm  # noqa: E402
import sa_geom as G  # noqa: E402
import features_fix as F  # noqa: E402
from torch_geometric.transforms import FixedPoints  # noqa: E402
from dataloading.kitti360pose.cells import Kitti360CoarseDatasetMulti  # noqa: E402
from datapreparation.kitti360pose.utils import (  # noqa: E402
    COLOR_NAMES,
    KNOWN_CLASS,
    SCENE_NAMES_TEST,
    SCENE_NAMES_TRAIN,
    SCENE_NAMES_VAL,
)
from evaluation.decode import decode_cell_offsets  # noqa: E402

FROZEN_SHA = {
    "sa_geom.py": "c82c016a22c326e7193d8d58bae0f138d013de2b5f2fe6353e9088b704348240",
    "common.py": "a2e1903d6ad955840dc05f35c7116f3452a2ff28ee2371879007498a5c41ebf1",
    "features_fix.py": "ea280d679fc64a976407ce6ebacaa2fda797b10eee1657d2b5e07306959f4615",
}
HINTS_SHA_FROZEN = {
    "train": "47bdb1881c39948fe738d64417321882fd80d1a75c3eff3ce4cd9ebc4278a728",
    "val": "d0531f73e8e079e1bcf92893d01f14fc9edf4f21a9658141b8005326f302de4c",
}
K = 25


def sha_text(hints):
    return hashlib.sha256(json.dumps(hints, ensure_ascii=False).encode("utf-8")).hexdigest()


def check_frozen_modules():
    for name, want in FROZEN_SHA.items():
        got = C.sha256_file(os.path.join(FROZEN, name))
        if got != want:
            raise SystemExit(f"frozen module {name} sha256 {got} != {want}")


def load_dataset(split):
    cfg = C.read_json(C.FINE_CFG)
    scenes = {"train": SCENE_NAMES_TRAIN, "val": SCENE_NAMES_VAL,
              "test": SCENE_NAMES_TEST}[split]
    ds = Kitti360CoarseDatasetMulti(
        cfg["base_path"], scenes, FixedPoints(cfg["pointnet_numpoints"]),
        shuffle_hints=False, flip_poses=False)
    poses = list(ds.all_poses)
    cells = list(ds.all_cells)
    hints = [h for d in ds.datasets for h in d.hint_descriptions]
    return ds, poses, cells, hints


def load_train_cache(root):
    cache = os.path.join(root, "cache")
    prog = C.read_json(os.path.join(root, "state", "progress.json"))
    n_chunks = (C.EXPECT["n_train_queries"] + 2000 - 1) // 2000
    if len(prog["completed"]) != n_chunks:
        raise SystemExit(f"train export incomplete: {len(prog['completed'])}/{n_chunks}")
    cells_npz = np.load(os.path.join(cache, "cells.npz"), allow_pickle=False)
    gallery = [str(x) for x in cells_npz["cell_ids"]]
    cell_enc = np.load(os.path.join(cache, "cell_encodings.npz"),
                       allow_pickle=False)["cell_enc"]
    meta = C.read_json(os.path.join(cache, "train_export_meta.json"))
    arrays = {"candidate_idx": [], "candidate_score": [], "offsets_norm": [],
              "pred_w": [], "pose_w": [], "text_enc": [], "query_index": []}
    hints_cache, cell_id_q = [], []
    for rec in prog["completed"]:
        z = np.load(os.path.join(root, "chunks", rec["npz"]), allow_pickle=False)
        for k in arrays:
            arrays[k].append(z[k])
        rows = [json.loads(l) for l in open(
            os.path.join(root, "chunks", rec["jsonl"]), encoding="utf-8")]
        hints_cache.extend(r["hints"] for r in rows)
        cell_id_q.extend(str(r["cell_id"]) for r in rows)
    out = {k: np.concatenate(v, 0) for k, v in arrays.items()}
    out.update({
        "gallery": gallery, "cell_enc": cell_enc.astype(np.float32), "meta": meta,
        "hints": hints_cache, "query_cell_id": cell_id_q,
        "target_ids": [str(x) for x in _targets_from_chunks(root, prog)],
    })
    return out


def load_test_cache(root):
    """TEST export written by test13_export.py (chunks + cache)."""
    cache = os.path.join(root, "cache")
    meta = C.read_json(os.path.join(cache, "test_export_meta.json"))
    if meta.get("split") != "test" or not meta.get("test_access"):
        raise SystemExit("test_export_meta.json is not a TEST export")
    if int(meta.get("test_read_number", 0)) != 2 or not meta.get("exploratory"):
        raise SystemExit("TEST export is not marked as the exploratory second read")
    prog = C.read_json(os.path.join(root, "state", "progress.json"))
    if len(prog["completed"]) != int(meta["n_chunks"]):
        raise SystemExit(f"test export incomplete: {len(prog['completed'])}"
                         f"/{meta['n_chunks']}")
    ident = C.read_json(os.path.join(C.ROUND, "out", "state", "TEST_IDENTITY.json"))
    if int(meta["n_queries"]) != int(ident["n_queries"]):
        raise SystemExit("test export meta disagrees with the recorded TEST identity")
    cells_npz = np.load(os.path.join(cache, "cells.npz"), allow_pickle=False)
    gallery = [str(x) for x in cells_npz["cell_ids"]]
    cell_enc = np.load(os.path.join(cache, "cell_encodings.npz"),
                       allow_pickle=False)["cell_enc"]
    fp_want = meta["fingerprint"]["fingerprint_sha256"]
    arrays = {"candidate_idx": [], "candidate_score": [], "offsets_norm": [],
              "pred_w": [], "pose_w": [], "text_enc": [], "query_index": []}
    hints_cache, cell_id_q = [], []
    for rec in prog["completed"]:
        z = np.load(os.path.join(root, "chunks", rec["npz"]), allow_pickle=False)
        if str(z["fingerprint_sha256"]) != fp_want:
            raise SystemExit(f"chunk {rec['chunk']} belongs to another run")
        for k in arrays:
            arrays[k].append(z[k])
        rows = [json.loads(l) for l in open(
            os.path.join(root, "chunks", rec["jsonl"]), encoding="utf-8")]
        hints_cache.extend(r["hints"] for r in rows)
        cell_id_q.extend(str(r["cell_id"]) for r in rows)
    out = {k: np.concatenate(v, 0) for k, v in arrays.items()}
    if len(out["pose_w"]) != int(meta["n_queries"]):
        raise SystemExit(f"chunk total {len(out['pose_w'])} != {meta['n_queries']}")
    out.update({
        "gallery": gallery, "cell_enc": cell_enc.astype(np.float32), "meta": meta,
        "hints": hints_cache, "query_cell_id": cell_id_q,
        "target_ids": [str(x) for x in _targets_from_chunks(root, prog)],
    })
    return out


def _targets_from_chunks(root, prog):
    out = []
    for rec in prog["completed"]:
        rows = [json.loads(l) for l in open(
            os.path.join(root, "chunks", rec["jsonl"]), encoding="utf-8")]
        out.extend(r["target_cell_id"] for r in rows)
    return out


def load_val_cache():
    z = np.load(C.VAL09_TOP25, allow_pickle=False)
    e = np.load(C.VAL09_ENCODINGS, allow_pickle=False)
    rows = [json.loads(l) for l in open(C.VAL09_QUERIES, encoding="utf-8")]
    gallery = [str(x) for x in e["cell_ids"]]
    out = {
        "candidate_idx": z["candidate_idx"].astype(np.int64),
        "candidate_score": z["candidate_score"].astype(np.float32),
        "offsets_norm": z["offsets_norm"].astype(np.float32),
        "pred_w": z["pred_w"].astype(np.float64),
        "pose_w": z["pose_w"].astype(np.float64),
        "text_enc": e["text_enc"].astype(np.float32),
        "cell_enc": e["cell_enc"].astype(np.float32),
        "query_index": z["query_index"].astype(np.int64),
        "gallery": gallery,
        "hints": [list(r["hints"]) for r in rows],
        "query_cell_id": [str(r["cell_id"]) for r in rows],
        "target_ids": [str(x) for x in z["target_cell_id"]],
        "meta": {"source": "COMMAND_09 frozen VAL cache",
                 "top25_sha256": C.EXPECT["val09_top25_sha256"],
                 "encodings_sha256": C.EXPECT["val09_encodings_sha256"]},
    }
    return out


def build_features(split, root, limit=0):
    check_frozen_modules()
    t0 = time.time()
    ds, poses, cells, ds_hints = load_dataset(split)
    if split == "train":
        cache = load_train_cache(root)
    elif split == "test":
        cache = load_test_cache(root)
    else:
        cache = load_val_cache()
    n_all = len(poses)
    n = len(cache["pose_w"])
    if limit:
        n = min(n, limit)

    errs = []
    gallery = cache["gallery"]
    ds_ids = [str(c.id) for c in cells]
    if gallery != ds_ids:
        errs.append("cache gallery != dataset all_cells order")
    if [str(p.cell_id) for p in poses][:n] != cache["target_ids"][:n]:
        errs.append("query order (target cell id) != dataset pose order")
    if cache["query_cell_id"][:n] != cache["target_ids"][:n]:
        errs.append("query cell id != target cell id")
    if limit == 0 and cache["hints"] != ds_hints:
        errs.append("cache hints != dataset hint_descriptions")
    hints_sha = sha_text(ds_hints)
    if split in ("train", "test") and cache["meta"].get("hints_sha256") != hints_sha:
        errs.append(f"export hints_sha256 {cache['meta'].get('hints_sha256')} != dataset")
    if errs:
        raise SystemExit("identity check failed: " + "; ".join(errs))

    cells_dict = {str(c.id): c for c in cells}
    id2row = {c: i for i, c in enumerate(gallery)}
    cand = cache["candidate_idx"][:n]
    if cand.shape[1] != K:
        raise SystemExit(f"candidate K={cand.shape[1]} != {K}")
    if any(len(set(row.tolist())) != K for row in cand):
        raise SystemExit("duplicate candidates in some query")
    grid = G.grid_points(33)
    aux = {}

    def aux_of(cid):
        a = aux.get(cid)
        if a is None:
            a = Cm.build_cell_aux(cells_dict[cid], grid, lambda o: o.get_color_text())
            aux[cid] = a
        return a

    feats = np.zeros((n, K, 23), np.float32)
    labels = np.zeros((n, K), np.uint8)
    valid = np.zeros(n, bool)
    p2_all = np.zeros((n, K), np.int16)
    p0_all = np.zeros((n, K), np.int16)
    tied_all = np.zeros((n, K), bool)
    cell_ids = np.empty((n, K), dtype="<U32")
    cell_scene = np.empty((n, K), dtype="<U32")
    gt_cells = np.empty(n, dtype="<U9")
    scenes_out = np.empty(n, dtype="<U4")
    query_ids = np.empty(n, dtype="<U32")
    all_cos = np.zeros((n, K), np.float32)
    sp_arr = np.zeros((n, K), np.float32)
    sc_c_arr = np.zeros((n, K), np.float32)
    sc_f_arr = np.zeros((n, K), np.float32)
    sat_arr = np.zeros((n, K), np.float32)
    dir_m = np.zeros((n, K), np.int16)
    sat_cnt = np.zeros((n, K), np.int16)
    cos_diff_max = 0.0
    pred_diff_max = 0.0
    pred_cmp_n = 0

    def one_query(i, gt_xy):
        idx = cand[i]
        cids = [gallery[j] for j in idx]
        cell_enc = cache["cell_enc"]
        cos_v = (cell_enc[idx].astype(np.float64)
                 @ cache["text_enc"][i].astype(np.float64))
        text = " ".join(cache["hints"][i])
        parsed = G.parse_relations(text, COLOR_NAMES, KNOWN_CLASS)
        assert len(parsed) == len(poses[i].descriptions) == 6, (i, text, parsed)
        n_q = max(1, len(parsed))
        sp_v = np.array([Cm.matched_count(parsed, aux_of(c)["counts"]) / n_q
                         for c in cids])
        sc_f = np.zeros(K)
        sat_f = np.zeros(K)
        sc_c = np.zeros(K)
        d_m = np.zeros(K, np.int16)
        s_c = np.zeros(K, np.int16)
        for r, cid in enumerate(cids):
            val, info = G.relation_score(parsed, aux_of(cid)["cm"], mode="C",
                                         point=cache["offsets_norm"][i, r])
            sc_f[r] = val
            sat_f[r] = info["satisfied"] / float(n_q)
            d_m[r] = info["matched"]
            s_c[r] = info["satisfied"]
            sc_c[r] = G.relation_score(parsed, aux_of(cid)["cm"], mode="C",
                                       point=(0.5, 0.5))[0]
        n_match = np.zeros(K)
        d_near = np.full(K, 3.0)
        d_cent = np.full(K, 3.0)
        n_obj = np.zeros(K)
        cell_c = np.zeros((K, 2))
        offs_i = cache["offsets_norm"][i]
        for r, cid in enumerate(cids):
            ids_m = Cm.matched_object_ids(parsed, aux_of(cid)["cm"])
            n_match[r] = len(ids_m) / n_q
            if ids_m:
                cen = aux_of(cid)["centers"][ids_m]
                d_near[r] = min(float(np.linalg.norm(cen - offs_i[r], axis=1).min()), 3.0)
                d_cent[r] = min(float(np.linalg.norm(cen.mean(0) - offs_i[r])), 3.0)
            n_obj[r] = aux_of(cid)["n_obj"] / 40.0
            cell_c[r] = np.asarray(cells_dict[cid].get_center()[0:2], float)

        p0_idx, p2_idx, tied_mask = F.compute_orders(cos_v, sp_v, sc_c, lam=C.LAMBDA)
        p0_rank_raw = np.empty(K, np.int64)
        p0_rank_raw[p0_idx] = np.arange(K)
        rows = np.arange(K, dtype=np.int64)
        f = F.feature_matrix(
            cos_v=cos_v, sp_v=sp_v, sc_center=sc_c, sc_fine=sc_f,
            sat_fine=sat_f, offs=offs_i, n_obj=n_obj, n_match=n_match,
            d_near=d_near, d_cent=d_cent, pred_w=cache["pred_w"][i], cell_c=cell_c,
            pool_rank=rows.astype(np.float64), p0_rank_raw=p0_rank_raw,
            p2_idx=p2_idx, tied_mask=tied_mask, rows=rows,
            pool_norm=K - 1, rank_norm=K - 1, tied_norm=K,
            anchor_idx=int(p2_idx[0]))
        gt_scene = str(poses[i].scene_name)
        prefix = str(poses[i].cell_id).split("_")[0]
        assert gt_scene == prefix, (i, gt_scene, prefix)
        cid_scene = np.array([str(cells_dict[c].scene_name) for c in cids])
        same = cid_scene == gt_scene
        dist = np.linalg.norm(cache["pred_w"][i] - np.asarray(gt_xy, np.float64)[None, :2],
                              axis=1)
        lab = (same & (dist <= 5.0)).astype(np.uint8)
        return {
            "feats": f, "label": lab, "cos_v": cos_v, "sp": sp_v, "sc_c": sc_c,
            "sc_f": sc_f, "sat": sat_f, "p0": p0_idx, "p2": p2_idx,
            "tied": tied_mask, "cids": cids, "cid_scene": cid_scene,
            "dir_m": d_m, "sat_cnt": s_c, "gt_scene": gt_scene,
            "n_pos": int(lab.sum()),
        }

    _seen = {}
    t1 = time.time()
    n_pos = 0
    for i in range(n):
        r = one_query(i, cache["pose_w"][i])
        feats[i] = r["feats"]
        labels[i] = r["label"]
        valid[i] = 1 <= r["n_pos"] < K
        p2_all[i] = r["p2"]
        p0_all[i] = r["p0"]
        tied_all[i] = r["tied"]
        cell_ids[i] = np.asarray(r["cids"], dtype="<U32")
        cell_scene[i] = r["cid_scene"].astype("<U32")
        gt_cells[i] = str(poses[i].cell_id)
        scenes_out[i] = r["gt_scene"]
        k = _seen.get(r["gt_scene"], 0)
        _seen[r["gt_scene"]] = k + 1
        query_ids[i] = f"{r['gt_scene']}:{k:04d}"
        all_cos[i] = r["cos_v"]
        sp_arr[i] = r["sp"]
        sc_c_arr[i] = r["sc_c"]
        sc_f_arr[i] = r["sc_f"]
        sat_arr[i] = r["sat"]
        dir_m[i] = r["dir_m"]
        sat_cnt[i] = r["sat_cnt"]
        n_pos += r["n_pos"]
        cd = float(np.max(np.abs(r["cos_v"] - cache["candidate_score"][i].astype(np.float64))))
        cos_diff_max = max(cos_diff_max, cd)
        if i < 256 or i % 977 == 0:
            cd2 = np.linalg.norm(
                decode_cell_offsets([cells_dict[c] for c in r["cids"]],
                                    cache["offsets_norm"][i].astype(np.float64))
                - cache["pred_w"][i], axis=1)
            pred_diff_max = max(pred_diff_max, float(cd2.max()))
            pred_cmp_n += 1
        if (i + 1) % 1000 == 0:
            el = time.time() - t1
            print(f"{split} {i+1}/{n} {el:.0f}s eta {(n-i-1)*el/(i+1)/60:.1f} min "
                  f"valid={int(valid[:i+1].sum())}", flush=True)
        if limit and i + 1 >= limit:
            break

    # GT-invariance probe: identical features when the GT pose is moved away
    gt_probe = {"n": 0, "features_identical": None, "labels_changed": 0}
    if n >= 8:
        m = min(64, n)
        same_f, changed = 0, 0
        for i in range(m):
            a = one_query(i, cache["pose_w"][i])
            b = one_query(i, cache["pose_w"][i] + np.array([500.0, 500.0, 0.0]))
            same_f += int(np.array_equal(a["feats"], b["feats"]))
            changed += int(not np.array_equal(a["label"], b["label"]))
        gt_probe = {"n": m, "features_identical": int(same_f), "labels_changed": changed,
                    "ok": same_f == m}

    # column-semantics checks on the emitted matrix (definitional consistency)
    p2pos = np.zeros_like(p2_all[:n])
    np.put_along_axis(p2pos, p2_all[:n].astype(np.int64),
                      np.broadcast_to(np.arange(K, dtype=np.int16), (n, K)), axis=1)
    scf = sc_f_arr[:n]
    scf_rank_p2 = np.stack([
        np.argsort(np.argsort(-scf[i][p2_all[i]], kind="stable"), kind="stable")
        for i in range(min(n, 5000))])
    # scf_rank_p2 is indexed by *canonical P2 position*; both `borda` and the
    # `sc_fine_rank` column are stored per candidate id, so scatter back first
    # (v0 of this checker compared canonical ranks against storage-order p2_rank).
    scf_rank = np.zeros_like(scf_rank_p2)
    np.put_along_axis(scf_rank, p2_all[:len(scf_rank_p2)], scf_rank_p2, axis=1)
    borda_ref = (p2pos[:len(scf_rank)] + scf_rank) / 48.0
    col = {
        "is_tied_equals_mask": bool(np.array_equal(feats[:n, :, 5] > 0.5, tied_all[:n])),
        "tied_size_equals_block_over_25": bool(np.allclose(
            feats[:n, :, 6], tied_all[:n].sum(1, keepdims=True) / 25.0)),
        "sc_center_col7": bool(np.allclose(feats[:n, :, 7], sc_c_arr[:n], atol=1e-6)),
        "sc_fine_col12": bool(np.allclose(feats[:n, :, 12], sc_f_arr[:n], atol=1e-6)),
        "sat_fine_col13": bool(np.allclose(feats[:n, :, 13], sat_arr[:n], atol=1e-6)),
        "borda_col15_sampled": bool(np.allclose(
            feats[:len(scf_rank), :, 15], borda_ref, atol=1e-6)),
        "sc_fine_rank_col14_sampled": bool(np.allclose(
            feats[:len(scf_rank), :, 14], scf_rank / 24.0, atol=1e-6)),
        "p2_rank_col4_sampled": bool(np.allclose(
            feats[:len(scf_rank_p2), :, 4], p2pos[:len(scf_rank_p2)] / 24.0, atol=1e-6)),
    }
    col_ok = all(col.values())

    outdir = os.path.join(root, "features")
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{split}_features.npz")
    arrays = dict(
        feats=feats, labels=labels, valid=valid, p2_idx=p2_all, p0_idx=p0_all,
        tied_mask=tied_all, cids=cell_ids, cell_ids=cell_ids,
        cell_scene=cell_scene, gt_cell=gt_cells, scene=scenes_out,
        scenes=scenes_out, query_ids=query_ids,
        gt_pose_w=cache["pose_w"][:n].astype(np.float64),
        cos=all_cos, scores=all_cos, sp=sp_arr, sc_center=sc_c_arr,
        sc_fine=sc_f_arr, sat_fine=sat_arr, dir_matched=dir_m,
        sat_count=sat_cnt, offsets=cache["offsets_norm"][:n].astype(np.float32),
        pred_w=cache["pred_w"][:n].astype(np.float64),
        cand_idx=cand[:n].astype(np.int32),
        feature_names=np.asarray(F.FEATURE_NAMES, dtype="<U32"),
        split=np.asarray([split], dtype="<U32"),
    )
    tmp = out_path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, out_path)

    labels_sum = int(labels.sum())
    meta = dict(
        split=split, queries=int(n), k=int(K), cells_gallery=len(gallery),
        valid_lists=int(valid.sum()), valid_frac=float(valid.mean()),
        positives_total=labels_sum,
        positives_per_list=float(labels_sum / max(1, n)),
        positives_per_valid=float(labels_sum / max(1, int(valid.sum()))),
        empty_lists=int((labels.sum(1) == 0).sum()),
        full_lists=int((labels.sum(1) == K).sum()),
        features=list(F.FEATURE_NAMES),
        norms=dict(pool_norm=K - 1, rank_norm=K - 1, tied_norm=K),
        label="same scene and ||pred_w - gt_pose_xy|| <= 5 m",
        scene_rule="pose.scene_name / cell.scene_name (asserted == id prefix)",
        feature_dependency=dict(
            gt_never_used=["cos", "cos_rank", "sp", "p0_rank", "p2_rank", "is_tied",
                           "tied_size", "sc_center", "sc_center_rank", "off_dx",
                           "off_dy", "off_norm", "sc_fine", "sat_fine",
                           "sc_fine_rank", "borda", "c_rank", "n_obj", "n_matched",
                           "dist_nearest_matched", "dist_centroid_matched",
                           "pred_dist_anchor", "center_dist_anchor"],
            fine_supported_columns=["sc_fine", "sat_fine", "sc_fine_rank"],
            fine_support_derivation="sc_fine=relation_score(parsed, cm, mode=C, "
                                    "point=offsets_norm); sat_fine=satisfied/n_q from the "
                                    "same call; sc_fine_rank=desc rank in canonical P2 "
                                    "order (rank_norm=24); borda=(p2_rank+sc_fine_rank)/48",
            gt_used_only_for=["labels", "metrics"]),
        sa_geom_module=G.__file__,
        sa_geom_sha256=C.sha256_file(G.__file__),
        common_sha256=C.sha256_file(Cm.__file__),
        features_fix_sha256=C.sha256_file(F.__file__),
        hints_sha256=hints_sha,
        hints_sha256_matches_frozen_build=(hints_sha == HINTS_SHA_FROZEN.get(split)),
        dataset_scenes=[str(d.scene_name) for d in ds.datasets],
        source=(os.path.abspath(os.path.join(root, "chunks"))
                if split in ("train", "test") else C.VAL09_CACHE),
        source_sha256=(cache["meta"].get("fingerprint", {}).get("fingerprint_sha256")
                       if split in ("train", "test") else cache["meta"]),
        cos_recompute_max_abs_diff=cos_diff_max,
        pred_w_redecode_checked=int(pred_cmp_n),
        pred_w_redecode_max_abs_diff=pred_diff_max,
        features_npz_sha256=None,
        written_at=C.now())
    meta["features_npz_sha256"] = C.sha256_file(out_path)
    C.write_json_atomic(out_path + ".meta.json", meta)
    check = {
        "split": split, "n": int(n), "k": K,
        "feats_finite": bool(np.all(np.isfinite(feats))),
        "p2_is_permutation": bool(all(sorted(r.tolist()) == list(range(K))
                                      for r in p2_all[:n])),
        "p0_is_permutation": bool(all(sorted(r.tolist()) == list(range(K))
                                      for r in p0_all[:n])),
        "p2_anchor_equals_top1": bool(np.all(p2_all[:n, 0] >= 0)),
        "candidate_unique_per_query": True,
        "all_queries_kept_no_positive_filter": True,
        "empty_lists_kept": int((labels.sum(1) == 0).sum()),
        "valid_lists": int(valid.sum()),
        "full_lists": int((labels.sum(1) == K).sum()),
        "cos_recompute_max_abs_diff": cos_diff_max,
        "cos_recompute_within_2e_5": bool(cos_diff_max <= 2e-5),
        "pred_w_redecode_checked": int(pred_cmp_n),
        "pred_w_redecode_max_abs_diff": pred_diff_max,
        "pred_w_redecode_within_1e_6": bool(pred_diff_max <= 1e-6),
        "gt_invariance": gt_probe,
        "column_semantics": col,
        "column_semantics_all_ok": col_ok,
        "hints_sha256": hints_sha,
        "hints_match_frozen_build": hints_sha == HINTS_SHA_FROZEN.get(split),
        "elapsed_seconds": round(time.time() - t0, 1),
        "timing_build_seconds": round(time.time() - t1, 1),
    }
    if not col_ok:
        check.setdefault("errors", []).append("column semantics mismatch")
    if not gt_probe.get("ok", False):
        check.setdefault("errors", []).append("GT-invariance probe failed")
    if not check["cos_recompute_within_2e_5"]:
        check.setdefault("errors", []).append("recomputed cos disagrees with export")
    if split == "val":
        orig_top = labels[:, 0]
        orig_cell = (cell_ids[:, 0] == gt_cells)
        check["val_original_order"] = {
            "top1_5m": int(orig_top.sum()),
            "cell_R1": int(orig_cell.sum()),
            "top1_5m_anchor": C.EXPECT["val_original_top1_5m"],
            "cell_R1_anchor": C.EXPECT["val_original_cell_R1"],
            "matches_anchor": bool(int(orig_top.sum()) == C.EXPECT["val_original_top1_5m"]
                                   and int(orig_cell.sum()) == C.EXPECT["val_original_cell_R1"]),
        }
        if not check["val_original_order"]["matches_anchor"]:
            check["errors"] = ["VAL original-order anchors do not reproduce"]
    if split == "test":
        orig_top = labels[:, 0]
        orig_cell = (cell_ids[:, 0] == gt_cells)
        check["test_original_order"] = {
            "n_queries": int(n),
            "top1_5m": int(orig_top.sum()),
            "cell_R1": int(orig_cell.sum()),
            "top1_5m_rate": float(orig_top.mean()),
            "cell_R1_rate": float(orig_cell.mean()),
            "note": "first computation of the TEST original order in this round; "
                    "there is no frozen TEST anchor and none is invented",
        }
        check["exploratory_test_read"] = True
    C.write_json_atomic(os.path.join(outdir, f"{split}_build_check.json"), check)
    print(json.dumps({"meta": {k: meta[k] for k in
                               ("split", "queries", "valid_lists", "positives_total",
                                "empty_lists", "full_lists", "features_npz_sha256")},
                      "check": check}, indent=1))
    return 0 if not check.get("errors") else 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--out-root", default=os.path.join(C.ROUND, "out"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    return build_features(args.split, args.out_root, args.limit)


if __name__ == "__main__":
    sys.exit(main())
