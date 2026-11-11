"""FEATURE_COORD_FIX_v1: cell-frame object centres for the A-domain SCAR lists.

The frozen A ``common.py`` (SHA a2e1903d...) computed the auxiliary object
centres as ``(mean(obj.xyz[:, :2]) - cell.bbox_w[0:2]) / cell.cell_size`` even
though ``obj.xyz`` is already "cropped and normalized in cell" (Cell contract,
``datapreparation/kitti360pose/imports.py``).  The extra world->cell transform
put the centres in a mixed frame; they feed only columns 19/20
(``dist_nearest_matched``, ``dist_centroid_matched``).

This module provides the corrected implementation as an *independent version*.
It never edits the shared historical files: ``apply()`` monkey-patches the
already-imported ``common`` module in the new builder process only.  All other
semantics (object truncation to ``max_objs``, token alignment, counts, n_obj,
3.0 truncation of the two distances) are byte-for-byte the same logic.

FEATURE_VERSION = "FEATURE_COORD_FIX_v1".
"""
import collections
import hashlib
import inspect

import numpy as np

FEATURE_VERSION = "FEATURE_COORD_FIX_v1"

# Columns whose values depend on ``aux["centers"]``.
AFFECTED_FEATURE_NAMES = (
    "dist_nearest_matched",
    "dist_centroid_matched",
)


def _src_sha(fn):
    try:
        return hashlib.sha256(inspect.getsource(fn).encode("utf-8")).hexdigest()
    except (OSError, TypeError):
        return None


def build_cell_aux_v1(cell, grid, color_of, geom, max_objs=40):
    """``common.build_cell_aux`` with the contract-correct centre.

    ``centers`` are means of the already cell-normalised object points, so they
    live in the same frame as the Fine offsets they are compared with.
    Every other return field is computed exactly as in the frozen code.
    """
    cm = geom.build_cell_maps(cell, grid, color_of, max_objs=max_objs)
    centers = []
    for obj in cell.objects[:max_objs]:
        xyz = np.asarray(obj.xyz, dtype=np.float64)
        if xyz.shape[0] < 1:
            continue
        centers.append(xyz[:, :2].mean(axis=0))
    centers = (np.asarray(centers, np.float32) if centers
               else np.zeros((0, 2), np.float32))
    assert len(centers) == len(cm.tokens)
    counts = collections.Counter((o.get_color_text(), o.label)
                                 for o in cell.objects)
    return dict(cm=cm, centers=centers, counts=counts, n_obj=len(cell.objects))


def apply(common_module, geom_module, max_objs=40):
    """Replace ``common_module.build_cell_aux`` in this process only.

    Returns a provenance dict (old/new source hashes, old/new code equality
    check).  The historical module object is otherwise untouched on disk.
    """
    orig = common_module.build_cell_aux
    orig_sha = _src_sha(orig)

    def build_cell_aux(cell, grid, color_of, max_objs=max_objs):
        return build_cell_aux_v1(cell, grid, color_of, geom_module,
                                 max_objs=max_objs)

    common_module.build_cell_aux = build_cell_aux
    patch = {
        "feature_version": FEATURE_VERSION,
        "patched_function": "build_cell_aux",
        "target_module_file": getattr(common_module, "__file__", None),
        "old_source_sha256": orig_sha,
        "new_source_sha256": _src_sha(build_cell_aux),
        "old_formula": "(mean(obj.xyz[:, :2]) - cell.bbox_w[0:2]) / cell.cell_size",
        "new_formula": "mean(obj.xyz[:, :2])  # already cell-normalised",
        "affected_features": list(AFFECTED_FEATURE_NAMES),
        "geom_module_file": getattr(geom_module, "__file__", None),
    }
    return patch
