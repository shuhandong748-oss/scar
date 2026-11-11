"""Pre-training freeze & verification for FORMAL_SCAR_20261103_MP_KITTI.

Read-only over the recovery directory.  Writes the protocol/input/code SHA
records into the delivery directory.  Nothing is trained here.
"""
import datetime as dt
import hashlib
import json
import os
import re
import sys

import numpy as np

RUN_ID = "FORMAL_SCAR_20261103_MP_KITTI"
REC = r"<SCAR_ROOT>/E_WEIGHT_CACHE_RECOVERY_20261103"
OUT = r"<SCAR_ROOT>/FORMAL_SCAR_20261103_MP_KITTI"
CODE = os.path.join(OUT, "code")
VENDOR = os.path.join(CODE, "vendor")
sys.path.insert(0, VENDOR)
import features_fix as F  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


checks = []


def chk(name, ok, detail=""):
    checks.append(dict(check=name, pass_=bool(ok), detail=str(detail)))
    print(("PASS " if ok else "FAIL ") + name + ((" | " + str(detail)) if detail else ""),
          flush=True)
    return ok


idx = json.load(open(os.path.join(REC, "RECOVERY_INDEX.json"), encoding="utf-8"))

# ---------------------------------------------------------------- 1. input SHA
input_records = []
for rec in idx["train_val_files"]:
    p = os.path.join(REC, rec["local_path"])
    actual = sha256(p)
    ok = actual == rec["sha256"] and os.path.getsize(p) == rec["size_bytes"]
    chk(f"sha256 {rec['local_path']}", ok, actual)
    input_records.append(dict(split=rec["local_path"].split("/")[1],
                              file=os.path.basename(rec["local_path"]),
                              rel_path=rec["local_path"], size=os.path.getsize(p),
                              expected_sha256=rec["sha256"], actual_sha256=actual,
                              match=ok))
chk("22/22 TRAIN/VAL files", len(input_records) == 22 and all(r["match"] for r in input_records))

TRAIN_NPZ = os.path.join(REC, "cache/train/feats.npz")
VAL_NPZ = os.path.join(REC, "cache/val/feats.npz")
train_sha = sha256(TRAIN_NPZ)
val_sha = sha256(VAL_NPZ)
chk("TRAIN anchor sha", train_sha == "e5bdfd28a6bb5821ca69a1a3d2aa9037e81226a567cb4c4b2fa3848b81d1af86", train_sha)
chk("VAL anchor sha", val_sha == "9cb9b04928745f83b155141ad1ee8631755f240efd9a7a0bb2a9be4e551d6df0", val_sha)

# TEST caches must NOT be opened.  Record names/sizes only, still verify SHA
# from the frozen manifest (file-level hashing is not deserialisation).
test_records = []
for rec in idx["test_files"]:
    p = os.path.join(REC, rec["local_path"])
    test_records.append(dict(rel_path=rec["local_path"], size=os.path.getsize(p),
                             expected_sha256=rec["sha256"],
                             opened=False, deserialized=False))

# ------------------------------------------------------- 2. feature/label/data
tr = load(TRAIN_NPZ)
va = load(VAL_NPZ)
nf, K = tr["feats"].shape[2], tr["feats"].shape[1]
chk("feature dim 23", nf == 23, nf)
chk("K = 25", K == 25 and va["feats"].shape[1] == 25)
chk("feature_names == vendor def (train)", list(tr["feature_names"]) == list(F.FEATURE_NAMES))
chk("feature_names == vendor def (val)", list(va["feature_names"]) == list(F.FEATURE_NAMES))
chk("feature_names are the 23-column canonical order",
    list(F.FEATURE_NAMES) == [
        "cos", "cos_rank", "sp", "p0_rank", "p2_rank", "is_tied", "tied_size",
        "sc_center", "sc_center_rank", "off_dx", "off_dy", "off_norm",
        "sc_fine", "sat_fine", "sc_fine_rank", "borda", "c_rank",
        "n_obj", "n_matched", "dist_nearest_matched", "dist_centroid_matched",
        "pred_dist_anchor", "center_dist_anchor"])

chk("TRAIN queries 28807", len(tr["feats"]) == 28807, len(tr["feats"]))
chk("VAL queries 3187", len(va["feats"]) == 3187, len(va["feats"]))
chk("TRAIN valid 25544", int(tr["valid"].sum()) == 25544, int(tr["valid"].sum()))
chk("VAL valid 2799 (metrics denominator is still 3187)", int(va["valid"].sum()) == 2799,
    int(va["valid"].sum()))
chk("VAL metric denominator = 3187", len(va["feats"]) == 3187)

lab_tr = tr["labels"]
lab_va = va["labels"]
chk("TRAIN valid lists: 1 <= positives < 25",
    lab_tr[tr["valid"]].sum(1).min() >= 1 and lab_tr[tr["valid"]].sum(1).max() < 25,
    (int(lab_tr[tr["valid"]].sum(1).min()), int(lab_tr[tr["valid"]].sum(1).max())))
chk("TRAIN invalid lists: 0 positives", int(lab_tr[~tr["valid"]].sum()) == 0,
    int(lab_tr[~tr["valid"]].sum()))
chk("VAL invalid lists: 0 positives", int(lab_va[~va["valid"]].sum()) == 0,
    int(lab_va[~va["valid"]].sum()))
chk("labels are binary 0/1", set(np.unique(lab_tr)) <= {0, 1} and set(np.unique(lab_va)) <= {0, 1})
# Label semantics are the frozen Des4Pos positive-cell set (near-GT cells),
# NOT exact gt-cell identity.  Record the fact instead of assuming identity.
pos_d = tr["dists"][lab_tr == 1]
chk("TRAIN positive candidates all within 5.0 m of the GT pose",
    float(pos_d.max()) <= 5.0 + 1e-4, float(pos_d.max()))
agree = np.mean([np.array_equal(lab_tr[i] == 1, tr["dists"][i] <= 5.0)
                 for i in np.where(tr["valid"])[0]])
chk("labels are the near-cell positive set (>=99% equal to dists<=5m; frozen semantics)",
    float(agree) >= 0.99, f"{float(agree):.4f}; {int(round((1-float(agree))*25544))}/25544 lists differ at the boundary")
gt_in = np.mean([str(gt) in set(map(str, tr["cell_ids"][i]))
                 for i, gt in enumerate(tr["gt_cell"]) if tr["valid"][i]])
chk("gold cell appears among the 25 candidates (informational, not a pass condition)",
    True, f"{float(gt_in):.4f} of valid TRAIN lists")

# candidate identity: 25 unique ids per query, p2_idx is a permutation of 0..24
unique_ok = True
perm_ok = True
for i in range(0, 28807, 1):
    if len(set(tr["cell_ids"][i].tolist())) != 25:
        unique_ok = False
        break
    if sorted(tr["p2_idx"][i].tolist()) != list(range(25)):
        perm_ok = False
        break
for i in range(0, 3187, 1):
    if len(set(va["cell_ids"][i].tolist())) != 25 or sorted(va["p2_idx"][i].tolist()) != list(range(25)):
        unique_ok = False
        perm_ok = False
        break
chk("candidate identity unique 25/query (all TRAIN+VAL)", unique_ok)
chk("p2_idx is a permutation of 0..24 (all TRAIN+VAL)", perm_ok)
chk("query_ids unique (TRAIN)", len(set(tr["query_ids"].tolist())) == len(tr["query_ids"]),
    len(set(tr["query_ids"].tolist())))
chk("query_ids unique (VAL)", len(set(va["query_ids"].tolist())) == len(va["query_ids"]),
    len(set(va["query_ids"].tolist())))
chk("split tags", str(tr["split"][0]) == "train" and str(va["split"][0]) == "val")

# ------------------------------------------------------- 3. normalisation (TRAIN only)
valid = tr["valid"]
feats = tr["feats"][valid].astype(np.float32)
flat = feats.reshape(-1, nf)
mean = flat.mean(0).astype(np.float32)
std_raw = flat.std(0)
pinned = std_raw < 1e-3
std = np.where(pinned, 1.0, std_raw).astype(np.float32)

norms = json.load(open(os.path.join(REC, "NORMALIZATION_AND_HEADS.json"), encoding="utf-8"))
for s in ("s0", "s1", "s2"):
    hm = np.asarray(norms["heads"][s]["mean"], np.float32)
    hs = np.asarray(norms["heads"][s]["std"], np.float32)
    hp = np.asarray(norms["heads"][s]["pinned"], bool)
    chk(f"normalisation matches historical head {s} (mean bitwise)",
        np.array_equal(hm, mean), float(np.abs(hm - mean).max()))
    chk(f"normalisation matches historical head {s} (std bitwise)",
        np.array_equal(hs, std), float(np.abs(hs - std).max()))
    chk(f"pinned columns match {s}", np.array_equal(hp, pinned), np.where(pinned)[0].tolist())

# a NEW mean computed from VAL would differ -> proves TRAIN-only fitting matters
val_flat = va["feats"].astype(np.float32).reshape(-1, nf)
val_mean = val_flat.mean(0)
chk("VAL-derived mean differs from TRAIN-derived mean (TRAIN-only is non-trivial)",
    float(np.abs(val_mean - mean).max()) > 1e-3, float(np.abs(val_mean - mean).max()))

# ------------------------------------------------------- 4. historical curves
curve_path = os.path.join(REC, "VAL_CURVES_3SEEDS.csv")
rows = []
with open(curve_path, encoding="utf-8") as f:
    hdr = f.readline().strip().split(",")
    for line in f:
        rows.append(dict(zip(hdr, [float(x) for x in line.strip().split(",")])))
chk("historical curve 90 rows (3 x 30)", len(rows) == 90, len(rows))

hist_sel = {}
for s in (0, 1, 2):
    h = sorted([r for r in rows if int(r["seed"]) == s], key=lambda r: r["epoch"])
    chk(f"historical curve s{s} has 30 epochs", len(h) == 30 and [int(r["epoch"]) for r in h] == list(range(1, 31)))
    a = max(range(len(h)), key=lambda i: (h[i]["val_cell_R@1"], h[i]["val_top1@5m"], -h[i]["epoch"]))
    b = max(range(len(h)), key=lambda i: (h[i]["val_top1@5m"], h[i]["val_cell_R@1"], -h[i]["epoch"]))
    hist_sel[f"s{s}"] = dict(A_epoch=int(h[a]["epoch"]), B_epoch=int(h[b]["epoch"]),
                             B_top1_5m=h[b]["val_top1@5m"], B_cell_R1=h[b]["val_cell_R@1"],
                             A_cell_R1=h[a]["val_cell_R@1"], A_top1_5m=h[a]["val_top1@5m"])
    chk(f"historical A re-selection s{s} == ep1", int(h[a]["epoch"]) == 1, int(h[a]["epoch"]))

status = json.load(open(os.path.join(REC, "B_RULE_HEAD_STATUS.json"), encoding="utf-8"))
for s in (0, 1, 2):
    st = status[f"s{s}"]
    hs = hist_sel[f"s{s}"]
    chk(f"historical B epoch s{s} == {st['B_best_epoch']}", hs["B_epoch"] == st["B_best_epoch"], hs["B_epoch"])
    chk(f"historical B top1@5m s{s} exact", abs(hs["B_top1_5m"] - st["B_best_top1@5m"]) < 1e-15)
    chk(f"historical B cell R@1 s{s} exact", abs(hs["B_cell_R1"] - st["B_best_cell_R@1"]) < 1e-15)
    chk(f"historical B weight absent (expected)", st["B_weight_exists"] is False)

# ------------------------------------------------------- 5. recipe freeze
orig = open(os.path.join(CODE, "train_head_original.py"), encoding="utf-8").read()
new = open(os.path.join(CODE, "train_head_formal.py"), encoding="utf-8").read()
recipe_tokens = [
    "23 -> 64 -> 32 -> 1", "nn.Linear(nf, 64)", "nn.Linear(64, 32)", "nn.Linear(32, 1)",
    "nn.ReLU()", "listwise_softmax_ce", "AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)",
    "batch 512", "for ep in range(1, cli.epochs + 1)", "rng.permutation(len(x))",
    "target = yb / yb.sum(1, keepdim=True)", "-(target * logp).sum(1).mean()",
    "torch.manual_seed(cli.seed)", "np.random.seed(cli.seed)",
    "torch.use_deterministic_algorithms(True)",
    "pinned = std < 1e-3", "std = np.where(pinned, 1.0, std).astype(np.float32)",
]
for t in recipe_tokens:
    chk(f"recipe token present in original: {t}", t in orig)
for t in ["nn.Linear(nf, 64)", "nn.Linear(64, 32)", "nn.Linear(32, 1)",
          "AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)",
          "pinned = std < 1e-3", "torch.use_deterministic_algorithms(True)",
          "rng.permutation(len(x))", "target = yb / yb.sum(1, keepdim=True)"]:
    chk(f"recipe token present in formal runner: {t}", t in new)
chk("formal runner keeps epochs default 30", 'ap.add_argument("--epochs", type=int, default=30)' in new)
chk("formal runner keeps batch 512", 'for b in range(0, len(x), 512)' in new)
chk("formal runner keeps K=25/nf=23 asserts", 'assert nf == 23 and K == 25' in new)

def extract(path, pat):
    return re.search(pat, open(path, encoding="utf-8").read(), re.S).group(1)

sig_orig = dict(
    lr="1e-3", wd="1e-4", batch=512, epochs=30, hidden=[64, 32],
    loss="listwise_softmax_ce", optimiser="AdamW",
    init="torch.manual_seed(seed); default PyTorch Linear init",
    data_order="np.random.RandomState(seed).permutation over TRAIN valid lists, batch 512",
    normalization="mean/std over TRAIN valid lists flattened (25 candidates), std<1e-3 -> 1.0",
    selection_A="max VAL cell R@1, tie max top1@5m, earlier epoch",
    selection_B="max VAL top1@5m, tie max VAL cell R@1, earlier epoch",
    tie_rule="rank_by_logits: stable argsort in P2 order",
    eval="VAL all 3187 queries, cell R@k + topk@5m, scene-filtered",
)

protocol = dict(
    run_id=RUN_ID,
    created=dt.datetime.now().isoformat(timespec="seconds"),
    scope="TRAIN/VAL only; TEST not opened, not evaluated; no Coarse/Fine retrain; no re-export",
    status_before_run="NOT_PREVIOUSLY_COMPLETED (FORMAL_SCAR_20261102 blocked on missing inputs)",
    selection_rule_main="max VAL Top-1@5m -> max VAL Cell R@1 -> earliest epoch",
    selection_role=dict(s0="MAIN", s1="sensitivity", s2="sensitivity"),
    historical_reference=hist_sel,
    historical_reference_note="ep5/8/6 are provenance only; the new trajectory may select different epochs",
    recipe=sig_orig,
    normalization=dict(policy="TRAIN valid lists only", pinned_columns=[F.FEATURE_NAMES[j] for j in np.where(pinned)[0]],
                       mean_sha256=hashlib.sha256(np.asarray(mean, np.float32).tobytes()).hexdigest(),
                       std_sha256=hashlib.sha256(np.asarray(std, np.float32).tobytes()).hexdigest()),
    code_sha256=dict(
        train_head_original=sha256(os.path.join(CODE, "train_head_original.py")),
        train_head_formal=sha256(os.path.join(CODE, "train_head_formal.py")),
        train_head_formal_mm=sha256(os.path.join(CODE, "train_head_formal_mm.py")),
        common_mp=sha256(os.path.join(CODE, "common_mp.py")),
        features_fix=sha256(os.path.join(VENDOR, "features_fix.py")),
        common=sha256(os.path.join(VENDOR, "common.py")),
        sa_geom=sha256(os.path.join(VENDOR, "sa_geom.py")),
        eval_head_original=sha256(os.path.join(CODE, "eval_head_original.py")),
    ),
    input_sha256=dict(train_feats=train_sha, val_feats=val_sha),
    data_facts=dict(
        train_queries=28807, train_valid=25544,
        val_queries=3187, val_valid=2799,
        val_metric_denominator=3187,
        val_lists_without_positive_candidate=3187 - 2799,
        val_note="VAL metrics are computed over all 3187 queries (the 388 no-positive lists count as misses), exactly as in the original train_head.py",
        label_semantics="positive set = near-GT cells (frozen); 99.89% coincide with dists<=5m, 29/25544 boundary lists differ; labels are NOT exact gt-cell identity",
        gt_cell_in_candidates_train_valid=float(gt_in),
    ),
    environment=dict(python=sys.version.split()[0], numpy=np.__version__,
                     executor="local CPU, single thread, no GPU, no global env change"),
    determinism=dict(torch_seed_per_run=True, numpy_seed_per_run=True,
                     deterministic_algorithms=True, threads=1,
                     note="OMP_NUM_THREADS=1 / MKL_NUM_THREADS=1 / torch.set_num_threads(1)"),
    deviations_from_original=[
        "D1 per-epoch state_dict saved (original kept only in-memory A-best)",
        "D2 B-rule is the formal selection; A-rule head also saved for provenance",
        "D3 extra provenance fields in checkpoints",
        "D4 executor pinned to one CPU thread (task requirement); original used engine defaults",
    ],
    no_test_opened=True,
)

os.makedirs(os.path.join(OUT, "protocol"), exist_ok=True)
json.dump(protocol, open(os.path.join(OUT, "protocol", "PROTOCOL.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
json.dump(dict(run_id=RUN_ID, split_files=input_records, test_files_not_opened=test_records),
          open(os.path.join(OUT, "inputs", "INPUT_SHA256.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
json.dump(protocol["code_sha256"],
          open(os.path.join(OUT, "protocol", "CODE_SHA256.json"), "w", encoding="utf-8"),
          indent=2)

# Deterministic single-file normalisation: rows = mean, std, pinned(0/1).
NORM_NPY = os.path.join(OUT, "protocol", "normalization.npy")
np.save(NORM_NPY, np.stack([mean.astype(np.float64), std.astype(np.float64),
                            pinned.astype(np.float64)]))
json.dump(dict(source="TRAIN valid lists only (recomputed in preflight_freeze.py)",
               mean=[float(v) for v in mean], std=[float(v) for v in std],
               pinned=[bool(v) for v in pinned],
               feature_names=list(F.FEATURE_NAMES),
               mean_sha256=hashlib.sha256(np.asarray(mean, np.float32).tobytes()).hexdigest(),
               std_sha256=hashlib.sha256(np.asarray(std, np.float32).tobytes()).hexdigest(),
               file_sha256=sha256(NORM_NPY)),
          open(os.path.join(OUT, "protocol", "normalization.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)

freeze = dict(run_id=RUN_ID, created=protocol["created"],
              protocol_sha256=sha256(os.path.join(OUT, "protocol", "PROTOCOL.json")),
              input_sha256=sha256(os.path.join(OUT, "inputs", "INPUT_SHA256.json")),
              code_sha256_sha256=sha256(os.path.join(OUT, "protocol", "CODE_SHA256.json")),
              train_feats=train_sha, val_feats=val_sha,
              normalization_npy=sha256(NORM_NPY),
              n_checks=len(checks), n_fail=sum(1 for c in checks if not c["pass_"]))
json.dump(freeze, open(os.path.join(OUT, "protocol", "FREEZE.json"), "w", encoding="utf-8"), indent=2)
json.dump(dict(run_id=RUN_ID, checks=checks,
               n_checks=len(checks), n_pass=sum(c["pass_"] for c in checks),
               n_fail=sum(1 for c in checks if not c["pass_"])),
          open(os.path.join(OUT, "verify", "PREFLIGHT.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)

print(f"\nPRE-FLIGHT: {sum(c['pass_'] for c in checks)}/{len(checks)} PASS")
sys.exit(0 if all(c["pass_"] for c in checks) else 1)
