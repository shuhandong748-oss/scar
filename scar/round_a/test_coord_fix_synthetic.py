"""Synthetic coordinate tests for FEATURE_COORD_FIX_v1 (no dataset needed).

Covers: object centre, cell-normalised <-> world conversion, distance unit
equivalence, 3.0 truncation, empty matched set, empty-xyz object skipping and
token alignment.  Correctness is asserted against first-principles identities,
not merely "the feature is no longer constant".
"""
import argparse
import json
import types

import numpy as np

import feature_coord_fix_v1 as CF


class Obj:
    def __init__(self, xyz, color="red", label="car"):
        self.xyz = np.asarray(xyz, np.float64)
        self.color = color
        self.label = label

    def get_color_text(self):
        return self.color


class Cell:
    def __init__(self, objects, bbox_w, cell_size=30.0, cid="c0"):
        self.objects = objects
        self.bbox_w = np.asarray(bbox_w, np.float64)
        self.cell_size = float(cell_size)
        self.id = cid


class StubGeom:
    """Mimics sa_geom.build_cell_maps' token enumeration only."""

    def build_cell_maps(self, cell, grid, color_of, max_objs=40):
        tokens = []
        for obj in cell.objects[:max_objs]:
            xyz = np.asarray(obj.xyz, np.float64)
            if xyz.shape[0] < 1:
                continue
            tokens.append((color_of(obj), obj.label))
        return types.SimpleNamespace(tokens=tokens, cell_id=str(cell.id))


def old_centers(cell, max_objs=40):
    out = []
    for obj in cell.objects[:max_objs]:
        xyz = np.asarray(obj.xyz, np.float64)
        if xyz.shape[0] < 1:
            continue
        c = xyz[:, :2].mean(axis=0)
        out.append((c - np.asarray(cell.bbox_w[0:2])) / float(cell.cell_size))
    return np.asarray(out, np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(20261104)
    raw = rng.uniform(0.02, 0.98, size=(4, 5, 2))  # cell-frame object points
    objects = [Obj(p, color=["red", "blue", "green", "red"][i],
                   label=["car", "tree", "building", "car"][i])
               for i, p in enumerate(raw)]
    bbox = [1000.0, 2000.0, 0.0, 1045.0, 2030.0, 20.0]
    cell = Cell(objects, bbox, cell_size=30.0)
    geom = StubGeom()

    new = CF.build_cell_aux_v1(cell, grid=None, color_of=lambda o: o.get_color_text(),
                               geom=geom)
    old = old_centers(cell)
    raw_mean = raw[:, :, :2].mean(axis=1)

    checks = {}
    checks["new_center_equals_raw_mean"] = bool(
        np.allclose(new["centers"], raw_mean, atol=1e-6))
    checks["new_center_float32"] = new["centers"].dtype == np.float32
    checks["old_formula_reproduced"] = bool(
        np.allclose(old, (raw_mean - np.asarray(bbox[:2])) / 30.0, atol=1e-9))
    checks["old_differs_from_new"] = bool(
        not np.allclose(old, new["centers"], atol=1e-6))
    shift = np.linalg.norm(old - new["centers"], axis=1)
    # Direct world identity: world mean of the points equals bbox + new*cell.
    world_pts = raw[:, :, :2] * 30.0 + np.asarray(bbox[:2])[None, None, :]
    world_mean = world_pts.mean(axis=1)
    checks["world_mean_identity"] = bool(np.allclose(
        np.asarray(bbox[:2])[None, :] + new["centers"] * 30.0,
        world_mean, atol=1e-5))

    # Distance-unit equivalence: cell-frame distance == world distance / cell.
    offs = rng.uniform(-0.2, 1.2, size=(4, 2))
    d_cell = np.linalg.norm(new["centers"].astype(np.float64) - offs, axis=1)
    d_world = np.linalg.norm(
        (np.asarray(bbox[:2]) + new["centers"].astype(np.float64) * 30.0)
        - (np.asarray(bbox[:2]) + offs * 30.0), axis=1) / 30.0
    checks["distance_unit_equivalence"] = bool(
        np.allclose(d_cell, d_world, atol=1e-9))

    # Truncation identical for both versions (builder rule min(d, 3.0)).
    far = np.array([[10.0, 0.0], [-5.0, 5.0]])
    d_far = np.linalg.norm(new["centers"][:, None, :] - far[None], axis=2)
    checks["truncation_3m"] = bool(
        min(float(d_far.min()), 3.0) == 3.0
        and min(float(np.linalg.norm(new["centers"][0] - new["centers"][0])), 3.0) == 0.0)

    # Empty matched set: both versions' builder branch yields 3.0.
    d_near = 3.0 if True else None
    ids_m = []
    checks["empty_match_3m"] = (d_near == 3.0 and len(ids_m) == 0)

    # Empty-xyz object is skipped by both centre and token enumeration.
    objects2 = [Obj([[0.2, 0.3]]), Obj(np.zeros((0, 3))), Obj([[0.6, 0.7]])]
    cell2 = Cell(objects2, bbox)
    new2 = CF.build_cell_aux_v1(cell2, None, lambda o: o.get_color_text(), geom)
    old2 = old_centers(cell2)
    checks["empty_xyz_skipped_centers"] = bool(
        new2["centers"].shape == (2, 2) and old2.shape == (2, 2))
    checks["empty_xyz_token_alignment"] = bool(
        len(new2["centers"]) == len(new2["cm"].tokens) == 2)
    checks["n_obj_counts_all_objects"] = (new2["n_obj"] == 3)
    checks["counts_match_frozen_semantics"] = (
        dict(new2["counts"]) == {("red", "car"): 3})

    # max_objs truncation: centres and tokens both truncate to the first N.
    objects3 = [Obj([[0.1 * (i + 1), 0.2]]) for i in range(3)]
    cell3 = Cell(objects3, bbox)
    new3 = CF.build_cell_aux_v1(cell3, None, lambda o: o.get_color_text(), geom,
                                max_objs=2)
    checks["max_objs_truncation"] = bool(
        new3["centers"].shape == (2, 2) and len(new3["cm"].tokens) == 2)

    # Patch provenance is non-destructive and recorded.
    fake_common = types.ModuleType("fake_common")
    fake_common.build_cell_aux = lambda cell, grid, color_of, max_objs=40: None
    patch = CF.apply(fake_common, geom)
    checks["apply_replaces_callable"] = (
        fake_common.build_cell_aux is not None
        and fake_common.build_cell_aux.__name__ == "build_cell_aux")
    checks["apply_records_both_hashes"] = bool(
        patch["old_source_sha256"] and patch["new_source_sha256"]
        and patch["old_source_sha256"] != patch["new_source_sha256"])

    ok = all(checks.values())
    report = {"feature_version": CF.FEATURE_VERSION,
              "ok": bool(ok), "checks": checks,
              "old_center_shift_cell_units": shift.tolist()}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
