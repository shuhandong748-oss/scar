#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble the per-domain delivery report from the frozen artefacts."""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scar_common as S  # noqa: E402

NAMES = {"original": "original", "presence": "Presence",
         "rank_fusion": "Rank Fusion", "fsr": "FSR", "scar_s0": "SCAR (s0)"}
ORDER = ["original", "presence", "rank_fusion", "fsr", "scar_s0"]


def pct(x):
    return f"{100 * x:.2f}"


def table(ev, split):
    rows = []
    rc, rl = ev["results_cell"], ev["results_loc"]
    rows.append("| method | Cell R@1 | R@5 | R@10 | R@25 | Top-1@5m | Top-5@5m | Top-10@5m |")
    rows.append("|---|---|---|---|---|---|---|---|")
    for m in ORDER:
        rows.append(f"| {NAMES[m]} | {pct(rc[m]['R@1'])} | {pct(rc[m]['R@5'])} | "
                    f"{pct(rc[m]['R@10'])} | {pct(rc[m]['R@25'])} | "
                    f"{pct(rl[m]['top1@5m'])} | {pct(rl[m]['top5@5m'])} | "
                    f"{pct(rl[m]['top10@5m'])} |")
    return "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["npm3d", "s3dis"])
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--protocol_sha", required=True)
    a = ap.parse_args(argv)
    run = os.path.abspath(a.run_dir)
    d = S.domain_cfg(a.domain)
    freeze = S.read_json(os.path.join(run, "FREEZE_LIST.json"))
    ev_val = S.read_json(os.path.join(run, "eval_val", "metrics.json"))
    ev_test = S.read_json(os.path.join(run, "eval_test", "metrics.json"))
    ft = S.read_json(os.path.join(run, "features", "train.npz.meta.json"))
    sel = S.read_json(os.path.join(run, "heads", "SELECTION_FORMAL.json"))
    vx = {s: S.read_json(os.path.join(run, "verify", f"VERIFY_EXPORT_{s}.json"))
          for s in ("train", "val", "test")}
    ve = {s: S.read_json(os.path.join(run, "verify", f"VERIFY_EVAL_{s}.json"))
          for s in ("val", "test")}
    marker = S.read_json(os.path.join(run, "test", "TEST_READ_ONCE.json"))
    deg = ft["relation_channel"]

    # ---- dependency index, with the small config/num_stats files copied ----
    dep_dir = os.path.join(run, "dependencies")
    os.makedirs(dep_dir, exist_ok=True)
    candidates = {
        "num_stats.json": d["num_stats"],
        "coarse_STAGE_FREEZE.json": os.path.join(d["run"], "coarse",
                                                 "STAGE_FREEZE.json"),
        "fine_STAGE_FREEZE.json": os.path.join(d["run"], "fine",
                                               "STAGE_FREEZE.json"),
        "coarse_INIT_FREEZE.json": os.path.join(d["run"], "coarse",
                                                "INIT_FREEZE.json"),
        "fine_INIT_FREEZE.json": os.path.join(d["run"], "fine",
                                              "INIT_FREEZE.json"),
        "coarse_STAGE_RESULT.json": os.path.join(d["run"], "coarse",
                                                 "STAGE_RESULT.json"),
        "fine_STAGE_RESULT.json": os.path.join(d["run"], "fine",
                                               "STAGE_RESULT.json"),
        "coarse_BEST.json": os.path.join(d["run"], "coarse", "BEST.json"),
        "fine_BEST.json": os.path.join(d["run"], "fine", "BEST.json"),
        "ACCEPT.json": os.path.join(d["run"], "ACCEPT.json"),
        "PROTOCOL_FREEZE.json": os.path.join(S.ROOT, "records",
                                             "PROTOCOL_FREEZE.json"),
        "GATE_BINDING.json": S.GATE,
        "adapter.py": d["adapter_path"],
    }
    deps = {}
    for name, src in candidates.items():
        item = {"source": src, "exists": os.path.isfile(src)}
        if item["exists"]:
            item.update(size=os.path.getsize(src), sha256=S.sha256(src))
            if os.path.getsize(src) < 2 ** 22:
                shutil.copy2(src, os.path.join(dep_dir, name))
                item["copied_to"] = os.path.join(dep_dir, name)
        deps[name] = item
    for label, path in (("coarse_checkpoint", freeze["backbones"]["coarse"]["path"]),
                        ("fine_checkpoint", freeze["backbones"]["fine"]["path"]),
                        ("prior", d["prior"]), ("pointnet", d["pointnet"]),
                        ("t5_tree", d["t5"]), ("base", d["base"])):
        deps[label] = {"source": path, "exists": os.path.exists(path),
                       "sha256": (S.sha256_tree(path)[0]
                                  if os.path.isdir(path) else
                                  S.sha256(path) if os.path.isfile(path) else None),
                       "note": "indexed only; not copied (large asset)"}
    S.write_json({"domain": a.domain, "generated": freeze["frozen_at"],
                  "items": deps}, os.path.join(run, "DEPENDENCIES.json"))

    lines = []
    lines.append(f"# Corrected Des4Pos × {a.domain.upper()} — formal SCAR full chain")
    lines.append("")
    lines.append(f"- protocol: `CORR-D4P-SCAR-FORMAL-v1`, protocol sha `{a.protocol_sha}`")
    lines.append(f"- freeze list sha `{freeze['freeze_sha256']}` (written before TEST)")
    lines.append(f"- backbones: coarse ep{d['coarse_epoch']} "
                 f"`{d['coarse_sha'][:12]}` / fine ep{d['fine_epoch']} "
                 f"`{d['fine_sha'][:12]}`")
    lines.append(f"- main head: s0, formal epoch "
                 f"{sel['seeds']['s0']['formal_epoch']}, "
                 f"`{sel['main_head_sha256'][:12]}`; selection = VAL Top-1@5m → "
                 f"VAL Cell R@1 → earliest epoch")
    lines.append(f"- cluster unit: `{d['cluster_unit']}`; "
                 f"TEST clusters {ev_test['invariants']['n_clusters']} / "
                 f"queries {ev_test['protocol']['n_queries']}; "
                 f"VAL clusters {ev_val['invariants']['n_clusters']} / "
                 f"queries {ev_val['protocol']['n_queries']}")
    lines.append("")
    lines.append("**TEST status.** This project has already read the TEST split "
                 "historically on the old backbone. This second read is an "
                 "exploratory read under the corrected backbone: it is not a "
                 "first project-level TEST and not an independent confirmatory "
                 "experiment. One shared Top-25 cache serves all five methods; "
                 "SCAR only reorders and the coordinates remain each candidate's "
                 "own Fine prediction.")
    lines.append("")
    lines.append(f"## VAL ({ev_val['protocol']['n_queries']} queries)")
    lines.append(table(ev_val, "val"))
    lines.append("")
    lines.append(f"## TEST ({ev_test['protocol']['n_queries']} queries)")
    lines.append(table(ev_test, "test"))
    lines.append("")
    lines.append("## Paired cluster bootstrap — SCAR − baseline (pp, 95% CI, "
                 "2000 resamples, seed 123)")
    lines.append("")
    lines.append("| split | baseline | metric | Δ (pp) | 95% CI | clusters |")
    lines.append("|---|---|---|---|---|---|")
    for split, ev in (("VAL", ev_val), ("TEST", ev_test)):
        for target in ("fsr", "rank_fusion", "presence", "original"):
            block = ev["comparisons"][f"SCAR_minus_{NAMES[target]}"]
            for metric in ("top1_5m", "cell_R1", "top5_5m", "cell_R5"):
                v = block[metric]
                lo, hi = (100 * x for x in v["ci95"])
                lines.append(f"| {split} | {NAMES[target]} | {metric} | "
                             f"{100 * v['delta']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | "
                             f"{v['n_clusters']} |")
    lines.append("")
    lines.append("Stability (s1/s2 vs the fixed main s0, TEST, Cell R@1): " +
                 "; ".join(
                     f"{k} {100 * v['cell_R1']['delta']:+.2f}pp"
                     for k, v in ev_test["stability"].items()))
    lines.append("")
    lines.append("## Checks")
    for label, ok in (("TRAIN export verify", vx["train"]["status"]),
                      ("VAL export verify", vx["val"]["status"]),
                      ("TEST export verify", vx["test"]["status"]),
                      ("VAL eval verify", ve["val"]["status"]),
                      ("TEST eval verify", ve["test"]["status"])):
        lines.append(f"- {label}: **{ok}**")
    lines.append(f"- shared pool invariants: R@25 equal across methods "
                 f"= {ev_test['invariants']['R25_equal']}, Top-25@5m equal "
                 f"= {ev_test['invariants']['top25_5m_equal']}, storage order = "
                 f"descending coarse cosine "
                 f"= {ev_test['invariants']['storage_order_is_coarse_cosine']}")
    lines.append("")
    lines.append("## Data caveats")
    lines.append(f"- canonical-23 relation channel: num_mentioned={deg['num_mentioned']}, "
                 f"min_match={deg['min_match']}, sc_center all-zero="
                 f"{deg['sc_center_all_zero']}, sat_fine all-zero="
                 f"{deg['sat_fine_all_zero']}, P2==P0 {deg['p2_equals_p0']}, "
                 f"sp std {deg['sp_std']:.4f}")
    if a.domain == "npm3d":
        lines.append("- NPM3D provides three description relations; the spatial "
                     "channels that depend on more descriptions are degenerate and "
                     "are reported as such rather than re-thresholded.")
    lines.append(f"- historical TEST marker preserved: "
                 f"`{marker['historical_test_access']['path']}` "
                 f"sha `{marker['historical_test_access']['sha256']}`")
    lines.append("")
    lines.append("## Assets")
    lines.append(f"- freeze: `{run}/FREEZE_LIST.json` "
                 f"(`{freeze['freeze_sha256']}`)")
    lines.append(f"- heads: `{run}/heads/` (s0/s1/s2 + every epoch under "
                 f"`heads/epochs/`)")
    lines.append(f"- caches: `{run}/export/{{train,val}}`, `{run}/test`")
    lines.append(f"- features: `{run}/features/*.npz(+.meta.json)`")
    lines.append(f"- evaluation: `{run}/eval_val`, `{run}/eval_test`; "
                 f"verification: `{run}/verify/`")
    lines.append(f"- dependency index: `{run}/DEPENDENCIES.json` "
                 f"(small configs copied to `{run}/dependencies/`)")
    lines.append(f"- code snapshot: `{run}/code_snapshot/`")
    lines.append(f"- SHA manifest: `{run}/SHA256SUMS`; tar: "
                 f"`{run}/delivery/{a.domain}_corrected_scar.tar`")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(run, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write(text)
    S.write_json({
        "domain": a.domain, "protocol_sha256": a.protocol_sha,
        "freeze_sha256": freeze["freeze_sha256"],
        "main_head": sel["main_head"], "main_head_sha256": sel["main_head_sha256"],
        "val": {"cell": ev_val["results_cell"], "loc": ev_val["results_loc"],
                "comparisons": ev_val["comparisons"]},
        "test": {"cell": ev_test["results_cell"], "loc": ev_test["results_loc"],
                 "comparisons": ev_test["comparisons"],
                 "stability": ev_test["stability"]},
        "verification": {"export": {k: v["status"] for k, v in vx.items()},
                         "eval": {k: v["status"] for k, v in ve.items()}},
        "relation_channel": deg,
        "historical_test_access": marker["historical_test_access"],
        "report_md_sha256": S.sha256(os.path.join(run, "REPORT.md")),
    }, os.path.join(run, "REPORT.json"))
    print(text[:2500], flush=True)
    print("saved", os.path.join(run, "REPORT.md"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
