"""Build results/COMPARISON_TABLE.csv from results/EVAL_RESULTS.json.

Deterministic; reproduces the table format used for the appended exploratory
TEST evaluation. All deltas/CI bounds are percentage points (pp).
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["mncl_s3dis", "t2l_kitti", "t2lpp_kitti", "t2l_s3dis", "t2l_pandaset"]
ROLE = {"0": "main"}
COLS = ["config", "seed", "role", "n", "n_clusters", "cluster_unit",
        "sat_fine_cell", "sat_fine_top1",
        "newA_epoch", "newA_cell", "newA_top1",
        "newB_epoch", "newB_cell", "newB_top1"]
for tag in ("BA_top1", "BA_cell", "BS_top1", "BS_cell"):
    COLS += [f"{tag}_delta_pp", f"{tag}_ci_lo_pp", f"{tag}_ci_hi_pp", f"{tag}_wlt"]


def pp(x):
    return round(100.0 * x, 4)


def main():
    r = json.load(open(os.path.join(HERE, "results", "EVAL_RESULTS.json"),
                       encoding="utf-8"))
    rows = []
    for cfg in ORDER:
        c = r["configs"][cfg]
        if c.get("status") != "DONE":
            continue
        for seed in sorted(c["per_seed"], key=int):
            s = c["per_seed"][seed]
            sat = c["sat_fine"]
            row = dict(config=cfg, seed=seed,
                       role=ROLE.get(seed, "sensitivity"),
                       n=c["n"], n_clusters=c["n_clusters"],
                       cluster_unit=c["cluster_unit"],
                       sat_fine_cell=sat["cell_R1"], sat_fine_top1=sat["top1_5m"],
                       newA_epoch=s["newA"]["epoch"], newA_cell=s["newA"]["cell_R1"],
                       newA_top1=s["newA"]["top1_5m"],
                       newB_epoch=s["newB"]["epoch"], newB_cell=s["newB"]["cell_R1"],
                       newB_top1=s["newB"]["top1_5m"])
            for prefix, name in (("BA", "newB_minus_newA"),
                                 ("BS", "newB_minus_sat_fine")):
                for metric, tag in (("top1_5m", "top1"), ("cell_R1", "cell")):
                    st = s["comparisons"][name][metric]
                    row[f"{prefix}_{tag}_delta_pp"] = pp(st["delta"])
                    row[f"{prefix}_{tag}_ci_lo_pp"] = pp(st["ci95"][0])
                    row[f"{prefix}_{tag}_ci_hi_pp"] = pp(st["ci95"][1])
                    row[f"{prefix}_{tag}_wlt"] = "%d/%d/%d" % (
                        st["win"], st["loss"], st["tie"])
            rows.append(row)
    out = os.path.join(HERE, "results", "COMPARISON_TABLE.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print("WROTE", out, len(rows), "rows")
    for row in rows:
        print("%-13s s%s %-11s BA top1 %+7.2f [%+7.2f,%+7.2f]  cell %+7.2f  | "
              "BS top1 %+7.2f [%+7.2f,%+7.2f]  cell %+7.2f" % (
                  row["config"], row["seed"], row["role"],
                  row["BA_top1_delta_pp"], row["BA_top1_ci_lo_pp"],
                  row["BA_top1_ci_hi_pp"], row["BA_cell_delta_pp"],
                  row["BS_top1_delta_pp"], row["BS_top1_ci_lo_pp"],
                  row["BS_top1_ci_hi_pp"], row["BS_cell_delta_pp"]))


if __name__ == "__main__":
    main()
