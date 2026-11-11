"""Build the formal 3-seed curve CSV and quantify deviation vs the historical
recovered curve.  Read-only over the delivered logs and the recovery dir.
"""
import csv
import json
import os

OUT = r"<SCAR_ROOT>/FORMAL_SCAR_20261103_MP_KITTI"
REC = r"<SCAR_ROOT>/E_WEIGHT_CACHE_RECOVERY_20261103"
NVAL = 3187

COLS = ["seed", "epoch", "loss", "train_top1",
        "val_cell_R@1", "val_cell_R@2", "val_cell_R@3", "val_cell_R@5",
        "val_cell_R@10", "val_cell_R@25",
        "val_top1@5m", "val_top3@5m", "val_top5@5m", "val_top10@5m"]
CELL = ["R@1", "R@2", "R@3", "R@5", "R@10", "R@25"]
LOC = ["top1@5m", "top3@5m", "top5@5m", "top10@5m"]

new_rows = []
for s in (0, 1, 2):
    d = json.load(open(os.path.join(OUT, "logs", f"s{s}.log.json"), encoding="utf-8"))
    for rec in d["history"]:
        row = {"seed": s, "epoch": rec["epoch"], "loss": rec["loss"],
               "train_top1": rec["train_top1"]}
        for k in CELL:
            row[f"val_cell_{k}"] = rec["val_cell"][k]
        for k in LOC:
            row[f"val_{k}"] = rec["val_loc"][k]
        new_rows.append(row)

with open(os.path.join(OUT, "logs", "VAL_CURVES_FORMAL_3SEEDS.csv"), "w",
          newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    for row in new_rows:
        w.writerow({k: row[k] for k in COLS})

hist = {}
with open(os.path.join(REC, "VAL_CURVES_3SEEDS.csv"), encoding="utf-8") as f:
    for row in csv.DictReader(f):
        hist[(int(float(row["seed"])), int(float(row["epoch"])))] = {
            k: float(v) for k, v in row.items() if k not in ("seed", "epoch")}

metrics = ["loss", "train_top1"] + [f"val_cell_{k}" for k in CELL] + [f"val_{k}" for k in LOC]
dev_rows = []
stats = {m: dict(max_abs=0.0, mean_abs=0.0, n_exact=0, n=0) for m in metrics}
for row in new_rows:
    key = (row["seed"], row["epoch"])
    h = hist[key]
    out = {"seed": row["seed"], "epoch": row["epoch"]}
    for m in metrics:
        d = row[m] - h[m]
        out[f"hist_{m}"] = h[m]
        out[f"new_{m}"] = row[m]
        out[f"delta_{m}"] = d
        st = stats[m]
        st["max_abs"] = max(st["max_abs"], abs(d))
        st["mean_abs"] += abs(d)
        st["n_exact"] += int(d == 0.0)
        st["n"] += 1
    dev_rows.append(out)

with open(os.path.join(OUT, "logs", "DEV_VS_HISTORY.csv"), "w", newline="",
          encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(dev_rows[0].keys()))
    w.writeheader()
    w.writerows(dev_rows)

for m in stats:
    stats[m]["mean_abs"] /= stats[m]["n"]
    stats[m]["max_abs_queries"] = stats[m]["max_abs"] * NVAL

summary = dict(
    n_epochs=len(new_rows),
    per_metric=stats,
    selected=[],
    note="deltas are NEW minus HISTORICAL on the same frozen VAL 3187 queries; "
         "1/3187 = 0.0003138",
)
for s in (0, 1, 2):
    rows = [r for r in new_rows if r["seed"] == s]
    b_new = max(rows, key=lambda r: (r["val_top1@5m"], r["val_cell_R@1"], -r["epoch"]))
    b_hist = max([r for r in dev_rows if r["seed"] == s],
                 key=lambda r: (r["hist_val_top1@5m"], r["hist_val_cell_R@1"], -r["epoch"]))
    summary["selected"].append(dict(
        seed=s, new_epoch=b_new["epoch"], historical_epoch=b_hist["epoch"],
        new_top1_5m=b_new["val_top1@5m"], historical_top1_5m=b_hist["hist_val_top1@5m"],
        delta_top1_5m=b_new["val_top1@5m"] - b_hist["hist_val_top1@5m"],
        delta_top1_5m_queries=(b_new["val_top1@5m"] - b_hist["hist_val_top1@5m"]) * NVAL,
        new_cell_R1=b_new["val_cell_R@1"], historical_cell_R1=b_hist["hist_val_cell_R@1"],
        delta_cell_R1=b_new["val_cell_R@1"] - b_hist["hist_val_cell_R@1"],
        delta_cell_R1_queries=(b_new["val_cell_R@1"] - b_hist["hist_val_cell_R@1"]) * NVAL,
    ))

json.dump(summary, open(os.path.join(OUT, "logs", "DEVIATION_SUMMARY.json"), "w",
                        encoding="utf-8"), indent=2, ensure_ascii=False)
print(json.dumps(summary["selected"], indent=1, ensure_ascii=False))
print("max |delta| val_cell_R@1:", stats["val_cell_R@1"]["max_abs"],
      "=", stats["val_cell_R@1"]["max_abs_queries"], "queries")
print("max |delta| val_top1@5m:", stats["val_top1@5m"]["max_abs"],
      "=", stats["val_top1@5m"]["max_abs_queries"], "queries")
print("exact matches val_cell_R@1:", stats["val_cell_R@1"]["n_exact"], "/", stats["val_cell_R@1"]["n"])
print("exact matches val_top1@5m:", stats["val_top1@5m"]["n_exact"], "/", stats["val_top1@5m"]["n"])
