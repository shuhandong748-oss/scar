"""Full old-defect vs FEATURE_COORD_FIX_v1 comparison on one split.

The old npz must come from the *unmodified* frozen builder run on the *same*
export.  Here we establish, over the full split:
  * every unaffected field / feature column is bit-identical;
  * only columns 19/20 change, with exact affected-query/candidate counts;
  * no NaN/Inf in the fixed features;
  * query order, Fine coordinates, labels and valid flags are unchanged.
"""
import argparse
import json
import os

import numpy as np

UNCHANGED_ARRAYS = [
    "labels", "valid", "p2_idx", "cell_ids", "gt_cell", "scene", "gt_pose_w",
    "cos", "sp", "sc_center", "sc_fine", "sat_fine", "offsets", "pred_w",
    "feature_names", "split", "query_ids", "source_pose_index", "order_index",
    "num_mentions", "n_parsed",
]
AFFECTED = ("dist_nearest_matched", "dist_centroid_matched")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    old = np.load(args.old, allow_pickle=True)
    new = np.load(args.new, allow_pickle=True)
    fnames = [str(x) for x in new["feature_names"]]
    assert fnames == [str(x) for x in old["feature_names"]], "feature names differ"
    assert list(old.keys()) == list(new.keys()) or set(old.keys()) == set(new.keys()), (
        sorted(set(old.keys()) ^ set(new.keys())))

    unchanged = {}
    for k in UNCHANGED_ARRAYS:
        if k in old and k in new:
            unchanged[k] = bool(np.array_equal(old[k], new[k]))
    missing_old = [k for k in new.keys() if k not in old]

    fo = np.asarray(old["feats"], np.float64)
    fn = np.asarray(new["feats"], np.float64)
    assert fo.shape == fn.shape
    col_exact = {}
    col_stats = {}
    for j, nm in enumerate(fnames):
        same = bool(np.array_equal(fo[..., j], fn[..., j]))
        col_exact[nm] = same
        if not same:
            d = np.abs(fo[..., j] - fn[..., j])
            col_stats[nm] = {
                "changed_rows": int((d > 0).sum()),
                "rows": int(d.size),
                "mean_abs_diff": float(d.mean()),
                "max_abs_diff": float(d.max()),
                "old_unique": int(len(np.unique(fo[..., j]))),
                "new_unique": int(len(np.unique(fn[..., j]))),
                "old_min": float(fo[..., j].min()),
                "old_max": float(fo[..., j].max()),
                "new_min": float(fn[..., j].min()),
                "new_max": float(fn[..., j].max()),
            }

    non_affected_same = all(v for k, v in col_exact.items() if k not in AFFECTED)
    affected_changed = {k: col_exact.get(k, False) for k in AFFECTED}
    finite = bool(np.isfinite(fn).all())
    changed_mask = np.zeros(fn.shape[:2], bool)
    for k in AFFECTED:
        j = fnames.index(k)
        changed_mask |= (np.abs(fo[..., j] - fn[..., j]) > 0)
    changed_queries = int(changed_mask.any(axis=1).sum())
    n_queries = int(fn.shape[0])
    n_rows = int(changed_mask.size)

    checks = {
        "unaffected_arrays_bitwise_equal": all(unchanged.values()),
        "unaffected_columns_bitwise_equal": non_affected_same,
        "affected_columns_do_change": not any(affected_changed.values()),
        "new_features_finite": finite,
        "same_shapes": True,
        "labels_unchanged": unchanged.get("labels", False),
        "valid_unchanged": unchanged.get("valid", False),
        "order_unchanged": unchanged.get("p2_idx", False),
        "cell_ids_unchanged": unchanged.get("cell_ids", False),
        "fine_offsets_unchanged": unchanged.get("offsets", False),
        "pred_w_unchanged": unchanged.get("pred_w", False),
    }
    report = {
        "feature_version": "FEATURE_COORD_FIX_v1",
        "old": args.old, "new": args.new,
        "n_queries": n_queries, "n_candidate_rows": n_rows,
        "changed_candidate_rows": int(changed_mask.sum()),
        "changed_candidate_frac": float(changed_mask.mean()),
        "changed_queries": changed_queries,
        "changed_query_frac": float(changed_queries / max(1, n_queries)),
        "unaffected_arrays": unchanged,
        "unaffected_array_missing_in_old": missing_old,
        "column_exact_equal": col_exact,
        "changed_column_stats": col_stats,
        "checks": checks,
        "ok": bool(all(checks.values())),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
