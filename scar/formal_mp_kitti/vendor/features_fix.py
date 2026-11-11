"""Canonical 23-feature candidate matrix (single definition, single source of truth).

Every column, direction, normaliser and tie-break follows the training-side
definition in `experiments/Des4Pos_learned_reranker/common.py`:

  feats = [cos, cos_rank, sp, p0_rank, p2_rank, is_tied, tied_size,
           sc_center, sc_center_rank, off_dx, off_dy, off_norm,
           sc_fine, sat_fine, sc_fine_rank, borda, c_rank,
           n_obj, n_matched, dist_nearest_matched, dist_centroid_matched,
           pred_dist_anchor, center_dist_anchor]

Directions / semantics that are easy to get wrong:
  * `cos_rank`  : position of the candidate in the **coarse cosine pool**
                  (NOT the P0 rank).
  * `p0_rank`   : position in the P0 order r_P = cos + lambda*sp.
  * `p2_rank`   : position in the **P2 order**, i.e. P0 after the directional
                  gate reordered the maximum-presence tie block by descending
                  `sc_center` (NOT the P0 rank).
  * `borda`     : (p2_rank + sc_fine_rank) / (2*(K-1)).
  * `c_rank`    : **ascending** rank of `borda` (smaller Borda is better).
  * anchors     : predicted position / cell centre of the **P2 top-1** row.

Tie convention (training): `feature_matrix` receives rows in P2 order and uses
stable sorts, so exact ties in `sc_center`, `sc_fine` and `borda` are resolved
by **P2 order**.  This builder therefore computes every rank-based column in
canonical P2 order and scatters the results back by candidate id; the output
row order (`rows`) only selects which candidates are emitted and in what
storage order.

Normalisers are adapter parameters, never implicit:
  pool_norm  divides `cos_rank` and `p0_rank`
  rank_norm  divides `p2_rank`, `sc_center_rank`, `sc_fine_rank`, `c_rank`
             and the 2*rank_norm denominator of `borda`
  tied_norm  divides the tie-block size
"""
import numpy as np

FEATURE_NAMES = [
    "cos", "cos_rank", "sp", "p0_rank", "p2_rank", "is_tied", "tied_size",
    "sc_center", "sc_center_rank", "off_dx", "off_dy", "off_norm",
    "sc_fine", "sat_fine", "sc_fine_rank", "borda", "c_rank",
    "n_obj", "n_matched", "dist_nearest_matched", "dist_centroid_matched",
    "pred_dist_anchor", "center_dist_anchor",
]
assert len(FEATURE_NAMES) == 23

LAMBDA = 0.5


def rank_desc(v):
    """0 = largest value, stable (ties keep the incoming order)."""
    order = np.argsort(-np.asarray(v), kind="stable")
    rank = np.empty(len(v), np.int64)
    rank[order] = np.arange(len(v))
    return rank


def rank_asc(v):
    """0 = smallest value, stable (training direction of `c_rank`)."""
    order = np.argsort(np.asarray(v), kind="stable")
    rank = np.empty(len(v), np.int64)
    rank[order] = np.arange(len(v))
    return rank


def compute_orders(cos_v, sp_v, sc_center, lam=LAMBDA):
    """P0 and the directional-gate P2 order over the *given candidate set*.

    Mirrors `common.compute_orders`: P0 = stable argsort of r_P; the whole
    maximum-presence tie block is reordered by descending `sc_center`
    (stable), then P2 = that order (the full order over the candidate set).

    Returns (p0_idx, p2_idx, tied_mask) as indices into the candidate set.
    """
    cos_v = np.asarray(cos_v, np.float64)
    sp_v = np.asarray(sp_v, np.float64)
    sc = np.asarray(sc_center, np.float64)
    if not np.all(np.isfinite(sc)):
        raise ValueError("sc_center must be finite for every candidate")
    r_p = cos_v + lam * sp_v
    p0_idx = np.argsort(-r_p, kind="stable")
    top = sp_v.max()
    tied = sp_v == top                       # exact equality, as in training
    if int(tied.sum()) >= 2:
        block = p0_idx[np.isin(p0_idx, np.where(tied)[0])]
        nb = block[np.argsort(-sc[block], kind="stable")]
        o = p0_idx.copy()
        o[np.isin(o, np.where(tied)[0])] = nb
    else:
        o = p0_idx
    return p0_idx, o, tied


def feature_matrix(*, cos_v, sp_v, sc_center, sc_fine, sat_fine, offs,
                   n_obj, n_match, d_near, d_cent, pred_w, cell_c,
                   pool_rank, p0_rank_raw, p2_idx, tied_mask, rows,
                   pool_norm, rank_norm, tied_norm, anchor_idx=None,
                   tied_count=None):
    """Build the (len(rows), 23) float32 matrix.

    Per-candidate inputs are indexed by candidate id (storage order).
    `p2_idx` is the full P2 order (a permutation of the K candidates) and is
    the canonical order for every rank computation; `rows` only selects the
    emitted rows.  `anchor_idx`, when given, must be the P2 top-1.
    """
    p2_idx = np.asarray(p2_idx, np.int64)
    rows = np.asarray(rows, np.int64)
    k = len(rows)
    if sorted(p2_idx.tolist()) != sorted(rows.tolist()):
        raise ValueError("p2_idx must be a permutation of the row candidate set")
    if anchor_idx is not None and int(anchor_idx) != int(p2_idx[0]):
        raise ValueError("anchor_idx must be the P2 top-1 candidate")

    n_all = len(np.asarray(cos_v))
    cos_v = np.asarray(cos_v, np.float64)
    sp_v = np.asarray(sp_v, np.float64)
    pool_rank = np.asarray(pool_rank, np.float64)
    p0_rank_raw = np.asarray(p0_rank_raw, np.float64)
    tied_mask = np.asarray(tied_mask, bool)
    if len(tied_mask) != n_all:
        raise ValueError("tied_mask must have one entry per candidate")

    # ---------------- canonical P2 order: positions and tie-stable ranks
    p2_pos = np.full(n_all, -1, np.int64)
    p2_pos[p2_idx] = np.arange(k)

    sc_c_all = np.nan_to_num(np.asarray(sc_center, np.float64), nan=0.0)
    sc_f_rank_c = rank_desc(np.asarray(sc_fine, np.float64)[p2_idx])
    sc_c_rank_c = rank_desc(sc_c_all[p2_idx])
    borda_c = (np.arange(k, dtype=np.float64) + sc_f_rank_c) / (2.0 * float(rank_norm))
    c_rank_c = rank_asc(borda_c)

    sc_c_rank = np.full(n_all, np.nan)
    sc_c_rank[p2_idx] = sc_c_rank_c
    sc_f_rank = np.full(n_all, np.nan)
    sc_f_rank[p2_idx] = sc_f_rank_c
    borda = np.full(n_all, np.nan)
    borda[p2_idx] = borda_c
    c_rank = np.full(n_all, np.nan)
    c_rank[p2_idx] = c_rank_c

    # ---------------- columns that only need per-candidate values
    cos_rank = pool_rank[rows] / float(pool_norm)
    p0_rank = p0_rank_raw[rows] / float(pool_norm)
    p2_rank = p2_pos[rows] / float(rank_norm)
    is_t = tied_mask[rows].astype(np.float64)
    n_tied = (float(tied_count) if tied_count is not None
              else float(tied_mask.sum()))
    tied_size = n_tied / float(tied_norm)

    offs = np.asarray(offs, np.float64)[rows]
    pred_w = np.asarray(pred_w, np.float64)
    cell_c = np.asarray(cell_c, np.float64)
    anchor = int(p2_idx[0])
    d_anchor = np.linalg.norm(pred_w[rows] - pred_w[anchor], axis=1) / 30.0
    d_anchor_c = np.linalg.norm(cell_c[rows] - cell_c[anchor], axis=1) / 30.0

    feats = np.stack([
        cos_v[rows], cos_rank, sp_v[rows], p0_rank, p2_rank, is_t,
        np.full(k, tied_size), sc_c_all[rows], sc_c_rank[rows] / float(rank_norm),
        offs[:, 0] - 0.5, offs[:, 1] - 0.5, np.linalg.norm(offs - 0.5, axis=1),
        np.asarray(sc_fine, np.float64)[rows], np.asarray(sat_fine, np.float64)[rows],
        sc_f_rank[rows] / float(rank_norm),
        borda[rows], c_rank[rows] / float(rank_norm),
        np.asarray(n_obj, np.float64)[rows], np.asarray(n_match, np.float64)[rows],
        np.asarray(d_near, np.float64)[rows], np.asarray(d_cent, np.float64)[rows],
        np.minimum(d_anchor, 3.0), np.minimum(d_anchor_c, 3.0),
    ], axis=1)
    return feats.astype(np.float32)


def rank_by_logits(logits, p2_idx):
    """Training-equivalent ordering: sort in P2 order, ties keep P2 order."""
    p2_idx = np.asarray(p2_idx, np.int64)
    order_in_p2 = np.argsort(-np.asarray(logits, np.float64)[p2_idx], kind="stable")
    return p2_idx[order_in_p2]