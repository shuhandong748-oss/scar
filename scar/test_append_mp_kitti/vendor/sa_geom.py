"""SpatialAlign (line C) -- leakage-free geometric relation scoring.  [sa_geom]

Phase 1 established that the frozen coarse model does not verify spatial
relations.  Phase 2 asks whether an explicit geometric relation term fixes that
*without any ground truth at inference time*.

THE LEAKAGE PROBLEM
-------------------
In KITTI360Pose the relation in a hint sentence is defined **relative to the
query pose** ("The pose is north of a dark-green pole").  Phase 1's
`relation_check()` therefore needed `pose.pose_w` as the anchor -- that is
ground truth, and using it at inference is exactly the leak we must avoid.

THE FIX
-------
Read the sentence the other way round: 6 relations are 6 **constraints on where
the pose may be**, anchored on objects of the candidate cell:

    "The pose is north of X"   <=>   p.y > X.y  and  |p.x - X.x| <= |p.y - X.y|
    "The pose is on-top of X"  <=>   ||p - nearest_point(X)||_xy < 0.05

The feasible set of p is computable from (query text, candidate cell map) alone.
The relation score is then a property of that feasible set -- no pose is ever
read, only searched.  Three pre-registered variants:

    A  max over p of (#satisfied)/#matched      (gate: #matched >= 4, else 0)
    B  max over p of (#satisfied)/#queried      (a missing object counts as
                                                 unsatisfied; presence-aware)
    C  (#satisfied at the model's OWN predicted pose) / #matched, i.e. at the
       cell centre -- the official coarse localisation prediction of
       Text2Loc++ -- not at the ground-truth pose.

Direction convention: identical to `datapreparation.kitti360pose.select.get_direction`
(``on-top`` iff the 2-D offset to the object's nearest point is < 0.05 cell
units; otherwise the cardinal whose axis dominates).  The only deviation is that
the nearest object point is found in **2-D (xy)** instead of 3-D, which removes
the need to know the pose's z.  `sa5_leak_audit.py` measures this deviation
against the dataset's own labels.

Allowed inputs, exhaustively:
    query text  |  candidate cell map (labels, colours, point coordinates)
    model output (retrieval score; its predicted pose = the cell centre)
Forbidden and never touched: pose.pose, pose.pose_w, pose.cell_id,
pose.descriptions, GT object ids, the query's ground-truth cell.
"""

import re

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None

ONTOP = 0.05
DIR_TO_CODE = {"north": 0, "south": 1, "east": 2, "west": 3, "on-top": 4}
CODE_TO_DIR = {v: k for k, v in DIR_TO_CODE.items()}

_SENT = re.compile(r"The pose is (on-top|north|south|east|west) of an? (.+?)\.\s*")


def parse_relations(text, color_vocab, class_vocab):
    """Parse a text_mode=0 query into [(color, class, direction_code), ...].

    Purely lexical: the raw query string is the only input.  The colour is the
    longest vocabulary prefix of the object phrase (needed because ``gray`` is a
    prefix of ``gray-green``).
    """
    colors = sorted(set(color_vocab), key=len, reverse=True)
    classes = set(class_vocab)
    out = []
    for m in _SENT.finditer(text + " "):
        direction, phrase = m.group(1), m.group(2).strip()
        color = None
        for c in colors:
            if phrase == c or phrase.startswith(c + " "):
                color = c
                break
        if color is None:
            continue
        cls = phrase[len(color):].strip()
        if cls not in classes:
            continue
        out.append((color, cls, DIR_TO_CODE[direction]))
    return out


def grid_points(n=33):
    """Candidate pose positions on the cell's own [0,1]^2 domain."""
    ax = np.linspace(0.0, 1.0, n)
    gx, gy = np.meshgrid(ax, ax, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


def _cardinal_code(d):
    ax, ay = np.abs(d[:, 0]), np.abs(d[:, 1])
    return np.where(ax >= ay,
                    np.where(d[:, 0] >= 0, 2, 3),
                    np.where(d[:, 1] >= 0, 0, 1)).astype(np.uint8)


def direction_field(xyz_xy, pts):
    """Direction code of `pts` relative to one object, dataset convention."""
    tree = cKDTree(xyz_xy)
    _, idx = tree.query(pts, k=1)
    closest = xyz_xy[idx]
    d = pts - closest
    dist = np.linalg.norm(d, axis=1)
    code = _cardinal_code(d)
    return np.where(dist < ONTOP, 4, code).astype(np.uint8)


class CellMaps:
    """Per-cell direction fields: one uint8 row per object, over the grid."""

    __slots__ = ("cell_id", "tokens", "maps", "grid", "_idx")

    def __init__(self, cell_id, tokens, maps, grid):
        self.cell_id = cell_id          # map identity only; never compared to GT
        self.tokens = tokens            # list[(color, class)] per object
        self.maps = maps                # (n_obj, G) uint8
        self.grid = grid                # (G, 2)
        self._idx = None

    def token_index(self):
        if self._idx is None:
            d = {}
            for j, t in enumerate(self.tokens):
                d.setdefault(t, []).append(j)
            self._idx = d
        return self._idx


def build_cell_maps(cell, grid, color_of, max_objs=40):
    """Build the direction field of every object of a cell.  Map data only."""
    tokens, rows = [], []
    for j, obj in enumerate(cell.objects[:max_objs]):
        xyz = np.asarray(obj.xyz, dtype=np.float64)
        if xyz.shape[0] < 1:
            continue
        rows.append(direction_field(xyz[:, :2], grid))
        tokens.append((color_of(obj), obj.label))
    maps = np.stack(rows, 0) if rows else np.zeros((0, len(grid)), np.uint8)
    return CellMaps(str(cell.id), tokens, maps, grid)


def _group_best(maps, obj_ids, dirs):
    """Per grid point, max number of simultaneously satisfiable (obj, dir) pairs.

    Token groups are disjoint object sets, so solving each group independently is
    exact.  `obj_ids` are the candidate objects of one (colour, class) token,
    `dirs` the direction codes queried for that token (usually 1, sometimes 2).

    VENDORED EXACT FAST PATH -- A machine, Text2Loc x PandaSet, 2026-10-24.
    Copied from /root/autodl-tmp/Des4Pos/c_line/sa_geom.py (B's file is left
    untouched).  The reference rule maximises over every choice of k distinct
    objects and every assignment of the k queried directions to them, i.e.
    C(m, k) * k! evaluations.  On PandaSet a hint repeats one (colour, class)
    token up to 5 times, so a cell holding 23 matching objects costs
    C(23,5)*5! ~ 4.0e6 vector steps (measured 8.0 s for a single pair).  That
    maximum equals a maximum bipartite matching between the k direction slots
    and the m candidate objects, which is computed here exactly by a slot-mask
    DP over objects, vectorised across grid points.  Same integers, same
    dtype, no threshold or ordering change.  Witness:
    scripts/verify_sa_geom_equiv.py + spec/T2L_PANDASET_PERF_ADDENDUM.md.
    """
    k = len(dirs)
    m = len(obj_ids)
    if m == 0 or k == 0:
        return None
    if k == 1:
        # one query slot, several objects of that token: satisfied if ANY of them
        # realises the direction.  (Must be `any`, not `sum`: summing would count
        # duplicate objects as if they were extra satisfied query slots.)
        return np.any(maps[obj_ids] == dirs[0], axis=0).astype(np.int16)
    if k > m:
        return None
    if k > 12:  # unreachable for these data (hints have 6 relations); keep the
        # reference enumeration so the public contract is unchanged for any k.
        from itertools import combinations, permutations
        best = None
        for combo in combinations(range(m), k):
            for perm in permutations(range(k)):
                acc = np.zeros(maps.shape[1], dtype=np.int16)
                for slot in range(k):
                    acc += (maps[obj_ids[combo[perm[slot]]]] == dirs[slot])
                best = acc if best is None else np.maximum(best, acc)
        return best
    obj_ids = np.asarray(obj_ids, dtype=np.int64)
    dirs = np.asarray(dirs)
    n_grid = maps.shape[1]
    n_states = 1 << k
    popcount = np.array([bin(s).count("1") for s in range(n_states)], dtype=np.int16)
    sat = maps[obj_ids][None, :, :] == dirs[:, None, None]  # (k, m, n_grid)
    reach = np.zeros((n_states, n_grid), dtype=bool)
    reach[0] = True
    for j in range(m):
        prev = reach.copy()           # this object may fill at most one slot
        for slot in range(k):
            bit = 1 << slot
            sat_j = sat[slot, j]
            if not sat_j.any():
                continue
            for mask in range(n_states):
                if mask & bit:
                    continue
                src = prev[mask]
                if not src.any():
                    continue
                cand = src & sat_j
                if cand.any():
                    reach[mask | bit] |= cand
    best = np.zeros(n_grid, dtype=np.int16)
    for mask in range(1, n_states):
        r = reach[mask]
        if r.any():
            np.maximum(best, popcount[mask], out=best, where=r)
    return best


def satisfaction_profile(relations, cm):
    """Per-grid-point number of simultaneously satisfiable queried relations.

    Returns (total_map | None, matched, n_queried).  This is the single source of
    truth used by both the audit and the reranking experiment.
    """
    n_q = len(relations)
    if n_q == 0 or cm.maps.shape[0] == 0:
        return None, 0, n_q
    idx = cm.token_index()
    groups = {}
    for color, cls, d in relations:
        groups.setdefault((color, cls), []).append(d)
    total = np.zeros(cm.maps.shape[1], dtype=np.int16)
    matched = 0
    for token, dirs in groups.items():
        obj_ids = idx.get(token, [])
        if not obj_ids:
            continue
        g = _group_best(cm.maps, obj_ids, dirs)
        if g is None:
            continue
        total += g
        matched += min(len(dirs), len(obj_ids))
    return total, matched, n_q


def relation_score(relations, cm, mode="A", point=None, min_match=4):
    """Leakage-free relation-consistency score of one (text, candidate cell).

    Parameters
    ----------
    relations : [(color, class, dir_code)]   parsed from the query text only
    cm        : CellMaps                     built from the candidate cell map only
    mode      : "A" | "B" | "C"
    point     : (x, y) cell-normalised pose used by mode "C" -- the model's own
                prediction (cell centre), never a ground-truth pose
    """
    total, matched, n_q = satisfaction_profile(relations, cm)
    info = dict(matched=matched, satisfied=0, n_q=n_q)
    if total is None or matched < min_match:
        return 0.0, info

    if mode == "C":
        if point is None:
            return 0.0, info
        gi = int(np.argmin(((cm.grid - np.asarray(point)) ** 2).sum(axis=1)))
        sat = int(total[gi])
    else:
        sat = int(total.max())
    info["satisfied"] = sat
    denom = matched if mode in ("A", "C") else n_q
    return sat / denom, info
