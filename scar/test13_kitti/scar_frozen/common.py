"""Shared utilities for the Des4Pos learned candidate reranker.

All features are computable at inference time from
  (query text, candidate cell map, coarse scores, Fine predicted offset).
Ground-truth pose/cell is read ONLY to build training labels and inside the
official metric, never as a model input.
"""
import collections
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import sa_geom as G  # noqa: E402

POOL = 50
LAMBDA = 0.5
TOPK = 10

CAND_FEATURES = [
    "cos",                 # coarse cosine score
    "cos_rank",            # rank in coarse top-50 (pool order)
    "sp",                  # matched presence fraction
    "p0_rank",             # rank by r_P = cos + lambda * sp
    "p2_rank",             # rank after CGSR gate == candidate position
    "is_tied",             # in the top-presence exact-tie block
    "tied_size",           # size of that block / 10
    "sc_center",           # direction consistency at cell centre
    "sc_center_rank",      # rank within the 10 by sc_center
    "off_dx",              # Fine offset relative to cell centre
    "off_dy",
    "off_norm",            # ||offset - centre||
    "sc_fine",             # direction consistency at Fine position
    "sat_fine",            # satisfied relations at Fine / n_q
    "sc_fine_rank",        # rank within the 10 by sc_fine
    "borda",               # (p2_rank + sc_fine_rank) / 18  == C-line score
    "c_rank",              # rank by borda within the 10 (C-line order)
    "n_obj",               # cell object count / 40
    "n_matched",           # matched query tokens / n_q
    "dist_nearest_matched",  # Fine pos -> nearest matched object (cell units)
    "dist_centroid_matched",  # Fine pos -> matched-object centroid
    "pred_dist_anchor",    # predicted world pos -> P2-top1 predicted pos / cell
    "center_dist_anchor",  # cell centre -> P2-top1 cell centre / cell
]
NFEAT = len(CAND_FEATURES)


def matched_count(rel, counts):
    """Greedy (colour,class) token presence count (old C-line definition)."""
    used = collections.defaultdict(int)
    m = 0
    for color, cls, _d in rel:
        k = used[(color, cls)]
        if k < counts.get((color, cls), 0):
            used[(color, cls)] = k + 1
            m += 1
    return m


def matched_object_ids(rel, cm):
    """Indices into cm.tokens/centers of the greedily matched objects."""
    idx = cm.token_index()
    used = collections.defaultdict(int)
    out = []
    for color, cls, _d in rel:
        lst = idx.get((color, cls), [])
        k = used[(color, cls)]
        if k < len(lst):
            out.append(lst[k])
            used[(color, cls)] = k + 1
    return out


def build_cell_aux(cell, grid, color_of, max_objs=40):
    """Direction maps + object centres (cell-normalised) + token counts."""
    cm = G.build_cell_maps(cell, grid, color_of, max_objs=max_objs)
    centers = []
    for obj in cell.objects[:max_objs]:
        xyz = np.asarray(obj.xyz, dtype=np.float64)
        if xyz.shape[0] < 1:
            continue
        c = xyz[:, :2].mean(axis=0)
        centers.append((c - np.asarray(cell.bbox_w[0:2])) / float(cell.cell_size))
    centers = np.asarray(centers, np.float32) if centers else np.zeros((0, 2), np.float32)
    assert len(centers) == len(cm.tokens)
    counts = collections.Counter((o.get_color_text(), o.label) for o in cell.objects)
    return dict(cm=cm, centers=centers, counts=counts, n_obj=len(cell.objects))


def compute_orders(scores_pool, sp, pool_ids, sc_center=None, sc_center_fn=None,
                   lambda_=LAMBDA):
    """P0/P2/Gate exactly as in the frozen C line.

    Returns (p0_idx, p2_idx, tied_mask, sc_center_full) with indices into the
    top-50 pool.  sc_center_full is only filled where needed when a callable is
    given (NaN elsewhere).
    """
    r_p = scores_pool + lambda_ * sp
    p0_idx = np.argsort(-r_p, kind="stable")

    if sc_center is None:
        sc = np.full(len(sp), np.nan, dtype=np.float32)
    else:
        sc = np.asarray(sc_center, dtype=np.float32).copy()

    top = sp.max()
    tied = np.where(sp == top)[0]
    if len(tied) >= 2:
        if sc_center_fn is not None:
            for j in tied:
                if np.isnan(sc[j]):
                    sc[j] = sc_center_fn(pool_ids[j])
        block = p0_idx[np.isin(p0_idx, tied)]
        nb = block[np.argsort(-sc[block], kind="stable")]
        o = p0_idx.copy()
        o[np.isin(o, tied)] = nb
    else:
        o = p0_idx

    p2_idx = o[:TOPK]
    if sc_center_fn is not None:
        for j in p2_idx:
            if np.isnan(sc[j]):
                sc[j] = sc_center_fn(pool_ids[j])
    return p0_idx, p2_idx, (sp == top), sc


def _rank_desc(vals):
    order = np.argsort(-vals, kind="stable")
    rank = np.empty(len(vals), dtype=np.int64)
    rank[order] = np.arange(len(vals))
    return rank


def feature_matrix(parsed, aux, cells_dict, pool_ids, scores_pool, sp, p0_idx,
                   p2_idx, tied_mask, sc_center, fine_offs):
    """(10, NFEAT) inference-only features for one query's P2 top-10."""
    n_q = max(1, len(parsed))
    k = len(p2_idx)
    inv_p0 = np.empty(POOL, dtype=np.int64)
    inv_p0[p0_idx] = np.arange(POOL)

    cids = [pool_ids[j] for j in p2_idx]
    offs = np.asarray(fine_offs, dtype=np.float64).reshape(k, 2)

    cos_v = np.array([scores_pool[j] for j in p2_idx], np.float64)
    sp_v = np.array([sp[j] for j in p2_idx], np.float64)
    sc_c_v = np.array([sc_center[j] for j in p2_idx], np.float64)
    sc_c_v = np.nan_to_num(sc_c_v, nan=0.0)
    p0r_v = np.array([inv_p0[j] for j in p2_idx], np.float64)
    is_t = np.array([1.0 if tied_mask[j] else 0.0 for j in p2_idx])
    tied_size = float(tied_mask.sum()) / TOPK

    sc_f = np.zeros(k, np.float64)
    sat_f = np.zeros(k, np.float64)
    n_match = np.zeros(k, np.float64)
    d_near = np.full(k, 3.0, np.float64)
    d_cent = np.full(k, 3.0, np.float64)
    n_obj = np.zeros(k, np.float64)
    pred_w = np.zeros((k, 2), np.float64)
    cell_c = np.zeros((k, 2), np.float64)

    for r, cid in enumerate(cids):
        a = aux[cid]
        cell = cells_dict[cid]
        val, info = G.relation_score(parsed, a["cm"], mode="C", point=offs[r])
        sc_f[r] = val
        sat_f[r] = info["satisfied"] / float(n_q)
        ids = matched_object_ids(parsed, a["cm"])
        n_match[r] = len(ids) / float(n_q)
        if ids:
            cen = a["centers"][ids]
            d = np.linalg.norm(cen - offs[r], axis=1)
            d_near[r] = min(float(d.min()), 3.0)
            d_cent[r] = min(float(np.linalg.norm(cen.mean(0) - offs[r])), 3.0)
        n_obj[r] = a["n_obj"] / 40.0
        pred_w[r] = np.asarray(cell.bbox_w[0:2], np.float64) + offs[r] * float(cell.cell_size)
        cell_c[r] = np.asarray(cell.get_center()[0:2], np.float64)

    sc_c_rank = _rank_desc(sc_c_v)
    sc_f_rank = _rank_desc(sc_f)
    p2_rank = np.arange(k, dtype=np.float64)
    borda = (p2_rank + sc_f_rank) / (2.0 * (TOPK - 1))
    c_ord = np.argsort(borda, kind="stable")
    c_rank = np.empty(k, np.int64)
    c_rank[c_ord] = np.arange(k)

    csz = 30.0
    d_anchor = np.linalg.norm(pred_w - pred_w[0], axis=1) / csz
    d_anchor_c = np.linalg.norm(cell_c - cell_c[0], axis=1) / csz

    feats = np.stack([
        cos_v,
        np.array([j for j in p2_idx], np.float64) / (POOL - 1),
        sp_v,
        p0r_v / (POOL - 1),
        p2_rank / (TOPK - 1),
        is_t,
        np.full(k, tied_size),
        sc_c_v,
        sc_c_rank / (TOPK - 1),
        offs[:, 0] - 0.5,
        offs[:, 1] - 0.5,
        np.linalg.norm(offs - 0.5, axis=1),
        sc_f,
        sat_f,
        sc_f_rank / (TOPK - 1),
        borda,
        c_rank / (TOPK - 1),
        n_obj,
        n_match,
        d_near,
        d_cent,
        np.minimum(d_anchor, 3.0),
        np.minimum(d_anchor_c, 3.0),
    ], axis=1)
    return feats.astype(np.float32)


def fine_forward_with_hint_encoding(model, objects, object_points, hint_enc):
    """Exact copy of CrossMatch.forward with the text branch precomputed.

    Text is encoded once per query and shared by its candidate cells; this is
    numerically equivalent in eval mode (BatchNorm uses running stats).
    """
    import torch
    import torch.nn.functional as F
    batch_size = len(objects)
    num_objects = len(objects[0])
    if hint_enc.dim() == 2:
        hint_enc = hint_enc.unsqueeze(0)
    if hint_enc.shape[0] != batch_size:
        hint_enc = hint_enc.expand(batch_size, -1, -1).contiguous()

    out = model.object_encoder(objects, object_points)
    object_encodings = out[0] if isinstance(out, tuple) else out
    object_encodings = object_encodings.reshape((batch_size, num_objects, model.embed_dim))
    object_encodings = F.normalize(object_encodings, dim=-1)

    desc0 = object_encodings.transpose(0, 1)
    desc1 = hint_enc.transpose(0, 1)
    if model.cross_objects is not None:
        if len(model.cross_hints) == len(model.cross_objects):
            for i in range(len(model.cross_hints)):
                desc0 = model.cross_objects[i](desc0, desc1)
                desc1 = model.cross_hints[i](desc1, desc0)
        else:
            desc0_new = model.cross_objects[0](desc0, desc1)
            desc1_new = model.cross_hints[0](desc1, desc0)
            desc1 = model.cross_hints[1](desc1_new, desc0_new)
    else:
        desc1 = model.cross_hints(desc1, desc0)
    desc1 = desc1.max(dim=0)[0]
    return model.mlp_offsets(desc1)


def label_and_distance(pose, cell, off):
    """Training label: Fine prediction within 5 m, official cross-scene rule."""
    pred = np.asarray(cell.bbox_w[0:2], np.float64) + np.asarray(off, np.float64) * float(cell.cell_size)
    d = float(np.linalg.norm(np.asarray(pose.pose_w[0:2], np.float64) - pred))
    if str(pose.cell_id).split("_")[0] != str(cell.id).split("_")[0]:
        d = np.inf
    return d, int(d <= 5.0)


def eval_order(order_ids, fine, poses, cells_dict, threshs=(5, 10, 15)):
    """Official Fine metric for a ranked list per query (one metric call each)."""
    from evaluation.utils import calc_sample_accuracies
    n = len(order_ids)
    hits = {k: {t: 0 for t in threshs} for k in (1, 5, 10)}
    hit1 = np.zeros(n, dtype=np.int8)
    for i in range(n):
        ids = list(order_ids[i][:TOPK])
        cells = [cells_dict[c] for c in ids]
        offs = np.stack([fine[(i, c)] for c in ids])
        acc = calc_sample_accuracies(poses[i], cells, offs, (1, 5, 10), threshs)
        for k in (1, 5, 10):
            for t in threshs:
                hits[k][t] += int(acc[k][t])
        hit1[i] = int(acc[1][5])
    out = {f"top{k}@{t}m": hits[k][t] / n for k in (1, 5, 10) for t in threshs}
    return out, hit1


def coarse_recall(order_ids, poses, k):
    return float(np.mean([poses[i].cell_id in order_ids[i][:k] for i in range(len(poses))]))


def paired(hit_a, hit_b):
    d = np.asarray(hit_a, np.int16) - np.asarray(hit_b, np.int16)
    return dict(win=int((d > 0).sum()), loss=int((d < 0).sum()), net=int(d.sum()))


def cluster_bootstrap(hit_a, hit_b, poses, seeds=2000):
    keys = np.array([[round(float(p.pose_w[0]), 6), round(float(p.pose_w[1]), 6)]
                     for p in poses])
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    n_cl = len(uniq)
    diff = np.asarray(hit_a, np.float64) - np.asarray(hit_b, np.float64)
    sum_d = np.bincount(inv, weights=diff, minlength=n_cl)
    cnt = np.bincount(inv, minlength=n_cl).astype(np.float64)
    rng = np.random.RandomState(123)
    vals = np.empty(seeds)
    for s in range(seeds):
        pick = rng.randint(0, n_cl, n_cl)
        vals[s] = sum_d[pick].sum() / cnt[pick].sum()
    return dict(delta=float(diff.mean()),
                ci95=[float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
                n_clusters=int(n_cl))
