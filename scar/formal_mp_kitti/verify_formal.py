"""Independent verification of FORMAL_SCAR_20261103_MP_KITTI.

Checks (all independent of the training-time in-memory state):
  1. B-rule selection recomputed from the delivered 90-row VAL curve CSV;
  2. every per-epoch checkpoint's epoch tag matches filename and curve;
  3. every per-epoch checkpoint reloads and reproduces its recorded VAL metrics
     (cell R@1..R@25 and top1/3/5/10@5m) exactly;
  4. selected heads are the per-epoch weights of their selection epoch, and
     their recorded selection key is the curve maximum under the B rule;
  5. normalisation is the frozen TRAIN-only normalisation (== protocol copy ==
     recovered historical heads), and differs from a VAL-derived mean;
  6. input files, recovered original assets and source-code copies are unchanged
     (SHA256);
  7. all 90 per-epoch weights are present (3 seeds x 30 epochs).

TEST caches are never opened or hashed here.
"""
import csv
import hashlib
import json
import os
import sys

import numpy as np

OUT = r"<SCAR_ROOT>/FORMAL_SCAR_20261103_MP_KITTI"
REC = r"<SCAR_ROOT>/E_WEIGHT_CACHE_RECOVERY_20261103"
WORK = os.path.join(OUT, "work_npy")
sys.path.insert(0, os.path.join(OUT, "code", "vendor"))
import features_fix as F  # noqa: E402

NVAL = 3187


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


checks = []


def chk(name, ok, detail=""):
    checks.append(dict(check=name, pass_=bool(ok), detail=str(detail)))
    print(("PASS " if ok else "FAIL ") + name + ((" | " + str(detail)) if detail else ""),
          flush=True)
    return ok


def val_metrics(logits, p2, ids, gt, scene, pw, gp):
    n = len(logits)
    rk = {k: np.zeros(n, np.int8) for k in (1, 2, 3, 5, 10, 25)}
    loc = {k: np.zeros(n, np.int8) for k in (1, 3, 5, 10)}
    for i in range(n):
        order = F.rank_by_logits(logits[i], p2[i])
        ranked = ids[i][order]
        for k in rk:
            rk[k][i] = int(str(gt[i]) in [str(c) for c in ranked[:k]])
        cand_scene = np.array([str(c).split("_")[0] for c in ranked])
        d = np.linalg.norm(pw[i][order] - gp[i][None, :], axis=1)
        d[cand_scene != str(scene[i])] = np.inf
        for k in loc:
            loc[k][i] = int(np.min(d[:k]) <= 5.0)
    return ({f"R@{k}": float(rk[k].mean()) for k in rk},
            {f"top{k}@5m": float(loc[k].mean()) for k in loc})


# ---------------------------------------------------------------- environment
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

# ---------------------------------------------------------------- curves
curve = {}
with open(os.path.join(OUT, "logs", "VAL_CURVES_FORMAL_3SEEDS.csv"), encoding="utf-8") as f:
    for row in csv.DictReader(f):
        curve[(int(row["seed"]), int(row["epoch"]))] = {k: float(v) for k, v in row.items()
                                                        if k not in ("seed", "epoch")}
chk("curve has 90 rows (3 x 30)", len(curve) == 90, len(curve))
for s in (0, 1, 2):
    eps = sorted(e for (ss, e) in curve if ss == s)
    chk(f"curve s{s} epochs 1..30", eps == list(range(1, 31)), (eps[0], eps[-1]))

# independent B-rule selection from the CSV only
sel_csv = {}
for s in (0, 1, 2):
    rows = [curve[(s, e)] for e in range(1, 31)]
    best = max(range(30), key=lambda i: (rows[i]["val_top1@5m"],
                                         rows[i]["val_cell_R@1"], -(i + 1)))
    sel_csv[s] = dict(epoch=best + 1, top1=rows[best]["val_top1@5m"],
                      cell=rows[best]["val_cell_R@1"])
    chk(f"B selection recomputed from CSV s{s} -> epoch {best+1}",
        True, f"top1@5m={rows[best]['val_top1@5m']:.10f} cell_R@1={rows[best]['val_cell_R@1']:.10f}")

# ---------------------------------------------------------------- normalization
nz = np.load(os.path.join(OUT, "protocol", "normalization.npy"))
mean_p = nz[0].astype(np.float32)
std_p = nz[1].astype(np.float32)
pin_p = nz[2].astype(bool)
norms_hist = json.load(open(os.path.join(REC, "NORMALIZATION_AND_HEADS.json"), encoding="utf-8"))
for s in ("s0", "s1", "s2"):
    hm = np.asarray(norms_hist["heads"][s]["mean"], np.float32)
    hs = np.asarray(norms_hist["heads"][s]["std"], np.float32)
    chk(f"protocol normalisation == recovered TRAIN-only head {s} (bitwise)",
        np.array_equal(hm, mean_p) and np.array_equal(hs, std_p))

# VAL-derived mean must differ
va_files = {k: os.path.join(WORK, "val", k + ".npy")
            for k in ("feats", "p2_idx", "cell_ids", "gt_cell", "scene",
                      "gt_pose_w", "pred_w", "feature_names", "split")}
if not all(os.path.exists(p) for p in va_files.values()):
    from train_extract import extract_members  # not used; fail loudly is better
    raise SystemExit("work_npy missing; rerun training/preflight extraction")
vfeats = np.load(va_files["feats"], mmap_mode="r")
val_mean = np.asarray(vfeats).reshape(-1, 23).mean(0)
chk("VAL-derived mean differs from frozen TRAIN mean (TRAIN-only is non-trivial)",
    float(np.abs(val_mean - mean_p).max()) > 1e-3,
    float(np.abs(val_mean - mean_p).max()))

vp2 = np.load(va_files["p2_idx"], mmap_mode="r")
vids = np.load(va_files["cell_ids"], mmap_mode="r")
vgt = np.load(va_files["gt_cell"], mmap_mode="r")
vscene = np.load(va_files["scene"], mmap_mode="r")
vpw = np.asarray(np.load(va_files["pred_w"], mmap_mode="r")).astype(np.float64)
vgp = np.asarray(np.load(va_files["gt_pose_w"], mmap_mode="r")).astype(np.float64)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(23, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)


def reload_eval(ckpt):
    m = Net()
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    x = torch.from_numpy(((np.asarray(vfeats, dtype=np.float32) - ckpt["mean"]) /
                          ckpt["std"]).astype(np.float32))
    with torch.no_grad():
        logits = m(x).numpy()
    return val_metrics(logits, vp2, vids, vgt, vscene, vpw, vgp)


# ---------------------------------------------------------------- per-epoch
REQ = ["state_dict", "epoch", "seed", "config", "mean", "std", "pinned",
       "feature_names", "loss", "train_top1", "val_cell", "val_loc",
       "norm_binding", "norm_binding_sha256", "train_sha256", "val_sha256",
       "code_sha256", "env"]
code_sha_mm = sha256(os.path.join(OUT, "code", "train_head_formal_mm.py"))
train_sha = sha256(os.path.join(REC, "cache/train/feats.npz"))
val_sha = sha256(os.path.join(REC, "cache/val/feats.npz"))
n_ckpt = 0
reload_max = 0.0
missing = []
for s in (0, 1, 2):
    for ep in range(1, 31):
        p = os.path.join(OUT, "heads", "per_epoch", f"s{s}_ep{ep:02d}.pt")
        if not os.path.exists(p):
            missing.append(p)
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        n_ckpt += 1
        ok = all(k in ck for k in REQ) and int(ck["epoch"]) == ep and int(ck["seed"]) == s
        chk(f"ckpt s{s} ep{ep:02d}: required fields + epoch/seed tag", ok)
        chk(f"ckpt s{s} ep{ep:02d}: input binding",
            ck["train_sha256"] == train_sha and ck["val_sha256"] == val_sha)
        chk(f"ckpt s{s} ep{ep:02d}: normalisation binding TRAIN-only",
            np.array_equal(np.asarray(ck["mean"], np.float32), mean_p) and
            np.array_equal(np.asarray(ck["std"], np.float32), std_p) and
            np.array_equal(np.asarray(ck["pinned"], bool), pin_p))
        chk(f"ckpt s{s} ep{ep:02d}: feature definition bound",
            list(ck["feature_names"]) == list(F.FEATURE_NAMES))
        chk(f"ckpt s{s} ep{ep:02d}: code sha", ck["code_sha256"] == code_sha_mm)
        nb = ck["norm_binding"]
        nb_sha = hashlib.sha256(json.dumps(nb, sort_keys=True,
                                           separators=(",", ":")).encode()).hexdigest()
        chk(f"ckpt s{s} ep{ep:02d}: norm_binding sha", nb_sha == ck["norm_binding_sha256"])
        row = curve[(s, ep)]
        dcel = max(abs(ck["val_cell"][k] - row[f"val_cell_{k}"]) for k in
                   ("R@1", "R@2", "R@3", "R@5", "R@10", "R@25"))
        dloc = max(abs(ck["val_loc"][k] - row[f"val_{k}"]) for k in
                   ("top1@5m", "top3@5m", "top5@5m", "top10@5m"))
        chk(f"ckpt s{s} ep{ep:02d}: metrics match curve", dcel == 0 and dloc == 0,
            f"max={max(dcel, dloc)}")
        rm, lm = reload_eval(ck)
        d1 = max(abs(rm[k] - ck["val_cell"][k]) for k in rm)
        d2 = max(abs(lm[k] - ck["val_loc"][k]) for k in lm)
        reload_max = max(reload_max, d1, d2)
        chk(f"ckpt s{s} ep{ep:02d}: saved->reloaded VAL identical", d1 == 0 and d2 == 0,
            f"max={max(d1, d2)}")
        if s == 0 and ep == 1:
            pass
chk("all 90 per-epoch weights present", n_ckpt == 90 and not missing, n_ckpt)

# ---------------------------------------------------------------- selected heads
for s in (0, 1, 2):
    selp = os.path.join(OUT, "heads", "selected", f"ht23_mp_formal_s{s}.pt")
    ck = torch.load(selp, map_location="cpu", weights_only=False)
    se = ck["selection"]
    rule_ok = se["rule"] == "max VAL top1@5m, tie max VAL cell R@1, earlier epoch"
    chk(f"selected s{s}: role/rule", rule_ok and ck["formal_scar_run"]["run_id"] ==
        "FORMAL_SCAR_20261103_MP_KITTI")
    chk(f"selected s{s}: epoch == CSV recomputation", int(se["epoch"]) == sel_csv[s]["epoch"],
        (se["epoch"], sel_csv[s]["epoch"]))
    chk(f"selected s{s}: selection key == CSV maximum",
        se["key"][0] == sel_csv[s]["top1"] and se["key"][1] == sel_csv[s]["cell"],
        se["key"])
    ep_ck = torch.load(os.path.join(OUT, "heads", "per_epoch",
                                    f"s{s}_ep{se['epoch']:02d}.pt"),
                       map_location="cpu", weights_only=False)
    same = all(torch.equal(ck["state_dict"][k], ep_ck["state_dict"][k])
               for k in ep_ck["state_dict"])
    chk(f"selected s{s}: weights == per-epoch ckpt of selected epoch", same)
    chk(f"selected s{s}: provenance is NEW TRAJECTORY",
        "NEW TRAJECTORY" in ck["formal_scar_run"]["provenance"])

# ---------------------------------------------------------------- asset integrity
idx = json.load(open(os.path.join(REC, "RECOVERY_INDEX.json"), encoding="utf-8"))
n_ok = 0
for rec in idx["train_val_files"]:
    p = os.path.join(REC, rec["local_path"])
    n_ok += int(sha256(p) == rec["sha256"])
chk("22 recovered TRAIN/VAL inputs unchanged", n_ok == 22, n_ok)

# recovered original heads/logs (inside the recovery dir) unchanged
hist_shas = {}
for s in (0, 1, 2):
    for fn in (f"ht23_cmm_s{s}.pt", f"ht23_cmm_s{s}.log.json"):
        p = os.path.join(REC, "runs", "heads", fn)
        hist_shas[fn] = sha256(p)
man = {}
with open(os.path.join(REC, "MANIFEST_SHA256.txt"), encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            man[parts[1].lstrip("*").lstrip("./")] = parts[0]
man_ok = 0
for fn, h in hist_shas.items():
    rel = f"runs/heads/{fn}"
    if rel in man:
        man_ok += int(man[rel] == h)
chk("recovered historical A heads/logs unchanged (6/6 in manifest)",
    man_ok == 6, f"{man_ok}/6")

# code copies identical to the recovered originals
code_pairs = [
    ("code/train_head_original.py", "post/train_head.py"),
    ("code/common_mp.py", "post/common_mp.py"),
    ("code/vendor/features_fix.py", "post/vendor/features_fix.py"),
    ("code/vendor/common.py", "post/vendor/common.py"),
    ("code/vendor/sa_geom.py", "post/vendor/sa_geom.py"),
]
for dst, src in code_pairs:
    chk(f"code copy {dst} byte-identical to recovered {src}",
        sha256(os.path.join(OUT, dst)) == sha256(os.path.join(REC, src)))

result = dict(run_id="FORMAL_SCAR_20261103_MP_KITTI",
              n_checks=len(checks), n_pass=sum(c["pass_"] for c in checks),
              n_fail=sum(1 for c in checks if not c["pass_"]),
              n_per_epoch_ckpts=n_ckpt,
              selected_heads={f"s{s}": dict(file=f"heads/selected/ht23_mp_formal_s{s}.pt",
                                            epoch=sel_csv[s]["epoch"],
                                            val_top1_5m=sel_csv[s]["top1"],
                                            val_cell_R1=sel_csv[s]["cell"],
                                            role="MAIN" if s == 0 else "sensitivity")
                              for s in (0, 1, 2)},
              reload_max_abs_delta=reload_max,
              normalization=dict(train_only=True,
                                 mean_sha256=hashlib.sha256(mean_p.tobytes()).hexdigest(),
                                 std_sha256=hashlib.sha256(std_p.tobytes()).hexdigest(),
                                 pinned=[F.FEATURE_NAMES[j] for j in np.where(pin_p)[0]]),
              inputs=dict(train_sha256=train_sha, val_sha256=val_sha),
              test_caches_opened=False,
              checks=checks)
json.dump(result, open(os.path.join(OUT, "verify", "VERIFY.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
json.dump(dict(sel_csv), open(os.path.join(OUT, "verify", "SELECTION_RECOMPUTE.json"), "w",
                              encoding="utf-8"), indent=2)
print(f"\nVERIFY: {result['n_pass']}/{result['n_checks']} PASS, reload_max_abs_delta={reload_max}")
sys.exit(0 if result["n_fail"] == 0 else 1)
