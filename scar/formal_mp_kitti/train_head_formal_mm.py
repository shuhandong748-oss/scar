"""Formal SCAR (B-rule) head training — memory-mapped streaming variant.

Same recipe, same code path and same per-element float32 operations as
``train_head_formal.py`` (itself a minimal patch of the recovered original
``post/train_head.py``).  The ONLY difference is memory management (D6):

  * the frozen npz members are stream-extracted to .npy files (byte-identical
    payload; .npz SHA is verified before extraction),
  * TRAIN features are read through ``np.memmap`` and normalised per batch,
    so the 58 MB valid-subset array is never materialised,
  * the per-epoch TRAIN top-1 is computed in the same 512-list chunks as the
    optimiser steps (metric-only chunking; weights are unaffected).

Per-element arithmetic is unchanged: ``(raw - mean) / std`` is elementwise, so
per-batch computation is bitwise identical to slicing a pre-normalised array.
mean/std come from the frozen TRAIN-only normalisation written by
``preflight_freeze.py`` (verified bitwise against a fresh TRAIN recomputation).
"""
import argparse
import json
import os
import platform
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common_mp as C  # noqa: E402


def sha256(path):
    return C.sha256(path)


def extract_members(npz_path, out_dir, names):
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    with zipfile.ZipFile(npz_path) as zf:
        for name in names:
            member = name + ".npy"
            dst = os.path.join(out_dir, member)
            with zf.open(member) as src, open(dst, "wb") as f:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            out[name] = dst
    return out


def val_metrics(logits, data, F, pw, gp):
    n = len(logits)
    p2 = data["p2_idx"]
    ids = data["cell_ids"]
    gt = data["gt_cell"]
    scene = data["scene"]
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
            {f"top{k}@5m": float(loc[k].mean()) for k in loc}, rk, loc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_npz", required=True)
    ap.add_argument("--val_npz", required=True)
    ap.add_argument("--norm_npy", required=True)
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--expected_train_sha", default="")
    ap.add_argument("--expected_val_sha", default="")
    ap.add_argument("--code_sha", default="")
    cli = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn
    torch.set_num_threads(1)
    VENDOR = os.path.join(HERE, "vendor")
    if not os.path.isdir(VENDOR):
        VENDOR = C.VENDOR
    sys.path.insert(0, VENDOR)
    import features_fix as F

    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    torch.use_deterministic_algorithms(True)
    os.makedirs(cli.out_dir, exist_ok=True)
    per_epoch_dir = os.path.join(cli.out_dir, "per_epoch")
    os.makedirs(per_epoch_dir, exist_ok=True)

    train_sha = sha256(cli.train_npz)
    val_sha = sha256(cli.val_npz)
    if cli.expected_train_sha:
        assert train_sha == cli.expected_train_sha, (train_sha, cli.expected_train_sha)
    if cli.expected_val_sha:
        assert val_sha == cli.expected_val_sha, (val_sha, cli.expected_val_sha)

    tr_files = extract_members(cli.train_npz, os.path.join(cli.work_dir, "train"),
                               ["feats", "labels", "valid", "feature_names", "split"])
    va_files = extract_members(cli.val_npz, os.path.join(cli.work_dir, "val"),
                               ["feats", "p2_idx", "cell_ids", "gt_cell", "scene",
                                "gt_pose_w", "pred_w", "feature_names", "split"])

    tr = {k: np.load(v, mmap_mode="r") for k, v in tr_files.items()}
    va = {k: np.load(v, mmap_mode="r") for k, v in va_files.items()}
    nf, K = tr["feats"].shape[2], tr["feats"].shape[1]
    assert nf == 23 and K == 25
    assert list(tr["feature_names"]) == list(F.FEATURE_NAMES)
    assert list(va["feature_names"]) == list(F.FEATURE_NAMES)
    assert str(tr["split"][0]) == "train" and str(va["split"][0]) == "val"
    assert len(tr["feats"]) == C.EXPECTED["train"], len(tr["feats"])
    assert len(va["feats"]) == C.EXPECTED["val"], len(va["feats"])

    nz = np.load(cli.norm_npy)
    mean = nz[0].astype(np.float32)
    std = nz[1].astype(np.float32)
    pinned = nz[2].astype(bool)
    assert mean.shape == (23,) and std.shape == (23,)
    norm_sha = sha256(cli.norm_npy)

    valid = np.asarray(tr["valid"]).astype(bool)
    valid_idx = np.where(valid)[0]
    n_train_total = int(len(valid))
    n_train_valid = int(len(valid_idx))
    assert n_train_valid == 25544
    labels_all = tr["labels"]

    n_sub = 0
    n_lab = 0
    for b in range(0, n_train_valid, 4096):
        idx = valid_idx[b:b + 4096]
        l = np.asarray(labels_all[idx]).astype(np.float32)
        n_sub += int(l.sum())
        assert l.sum(1).min() >= 1 and l.sum(1).max() < K
        n_lab += len(l)
    print(f"train lists {n_lab}/{n_train_total} valid | positives/list "
          f"{n_sub / n_lab:.2f}", flush=True)
    print(f"pinned columns: {[F.FEATURE_NAMES[j] for j in np.where(pinned)[0]]}", flush=True)

    feats_mm = tr["feats"]
    norm_binding = dict(
        mean=[float(v) for v in mean], std=[float(v) for v in std],
        pinned=[bool(v) for v in pinned], feature_names=list(F.FEATURE_NAMES),
        source_split="train", source_mask="valid (1 <= positives < 25)",
        n_train_valid=n_train_valid, n_train_total=n_train_total,
        train_sha256=train_sha, val_sha256=val_sha,
        mean_sha256=C.hashlib.sha256(np.asarray(mean, np.float32).tobytes()).hexdigest(),
        std_sha256=C.hashlib.sha256(np.asarray(std, np.float32).tobytes()).hexdigest(),
        definition_sha256=sha256(os.path.join(VENDOR, "features_fix.py")),
        norm_file_sha256=norm_sha,
    )
    norm_binding_sha = C.hashlib.sha256(
        json.dumps(norm_binding, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(nf, 64), nn.ReLU(),
                                     nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, z):
            return self.net(z).squeeze(-1)

    model = Net()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.RandomState(cli.seed)

    vx = np.array(va["feats"], dtype=np.float32)   # 7.3 MB VAL copy
    vx -= mean
    vx /= std
    pw64 = np.asarray(va["pred_w"]).astype(np.float64)
    gp64 = np.asarray(va["gt_pose_w"]).astype(np.float64)

    config = dict(epochs=cli.epochs, lr=1e-3, wd=1e-4, batch=512,
                  hidden=[64, 32], loss="listwise_softmax_ce",
                  seed=cli.seed, k=K, dataset="kitti",
                  backbone=C.BACKBONE_TAG,
                  norms=dict(pool_norm=24, rank_norm=24, tied_norm=25))
    env = dict(python=platform.python_version(), numpy=np.__version__,
               torch=torch.__version__, torch_threads=torch.get_num_threads(),
               cuda_available=torch.cuda.is_available(),
               deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
               omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
               mkl_num_threads=os.environ.get("MKL_NUM_THREADS"),
               platform=platform.platform(), data_loader="np.memmap streaming")

    best_a = None
    best_b = None
    hist = []

    for ep in range(1, cli.epochs + 1):
        perm = rng.permutation(n_train_valid)
        model.train()
        tot, cnt = 0.0, 0
        for b in range(0, n_train_valid, 512):
            idx = valid_idx[perm[b:b + 512]]
            raw = np.asarray(feats_mm[idx], dtype=np.float32)
            xb = (raw - mean) / std
            yb = torch.from_numpy(np.asarray(labels_all[idx], dtype=np.float32))
            xb = torch.from_numpy(xb)
            logits = model(xb)
            logp = Fn.log_softmax(logits, dim=1)
            target = yb / yb.sum(1, keepdim=True)
            loss = -(target * logp).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
            cnt += len(idx)
        model.eval()
        hits = 0
        with torch.no_grad():
            for b in range(0, n_train_valid, 512):
                idx = valid_idx[b:b + 512]
                raw = np.asarray(feats_mm[idx], dtype=np.float32)
                xb = torch.from_numpy((raw - mean) / std)
                pred = model(xb).argmax(1).numpy()[:, None]
                lab = np.asarray(labels_all[idx]) > 0
                hits += int(np.take_along_axis(lab, pred, axis=1).sum())
        tr_hit = hits / n_train_valid
        with torch.no_grad():
            vl = model(torch.from_numpy(vx)).numpy()
        rm, lm, _, _ = val_metrics(vl, va, F, pw64, gp64)
        key_a = (rm["R@1"], lm["top1@5m"])
        key_b = (lm["top1@5m"], rm["R@1"])
        rec = dict(epoch=ep, loss=tot / cnt, train_top1=float(tr_hit),
                   val_cell=rm, val_loc=lm)
        hist.append(rec)
        print(f"epoch {ep:02d}/{cli.epochs} loss {tot/cnt:.4f} train_top1 {tr_hit:.4f} "
              f"| val R1 {rm['R@1']:.4f} R3 {rm['R@3']:.4f} R5 {rm['R@5']:.4f} "
              f"top1@5m {lm['top1@5m']:.4f} top3@5m {lm['top3@5m']:.4f}", flush=True)

        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ep_ckpt = dict(
            state_dict=sd, epoch=ep, seed=cli.seed,
            config=config, mean=mean, std=std, pinned=pinned,
            feature_names=F.FEATURE_NAMES,
            loss=float(tot / cnt), train_top1=float(tr_hit),
            val_cell={k: float(v) for k, v in rm.items()},
            val_loc={k: float(v) for k, v in lm.items()},
            norm_binding=norm_binding, norm_binding_sha256=norm_binding_sha,
            train_sha256=train_sha, val_sha256=val_sha,
            code_sha256=cli.code_sha, env=env)
        torch.save(ep_ckpt, os.path.join(per_epoch_dir, f"s{cli.seed}_ep{ep:02d}.pt"))

        if best_a is None or key_a > best_a["key"]:
            best_a = dict(key=key_a, epoch=ep, val_cell=rm, val_loc=lm, state_dict=sd)
        if best_b is None or key_b > best_b["key"]:
            best_b = dict(key=key_b, epoch=ep, val_cell=rm, val_loc=lm, state_dict=sd)

    def ckpt_for(best, rule):
        return dict(state_dict=best["state_dict"], mean=mean, std=std,
                    pinned=pinned, feature_names=F.FEATURE_NAMES,
                    config=config,
                    selection=dict(rule=rule, epoch=best["epoch"],
                                   key=[float(best["key"][0]), float(best["key"][1])],
                                   val_cell={k: float(v) for k, v in best["val_cell"].items()},
                                   val_loc={k: float(v) for k, v in best["val_loc"].items()}),
                    history=hist, n_train_total=n_train_total,
                    n_train_valid=n_train_valid,
                    train_npz=cli.train_npz, val_npz=cli.val_npz,
                    train_sha256=train_sha, val_sha256=val_sha,
                    norm_binding=norm_binding, norm_binding_sha256=norm_binding_sha,
                    code_sha256=cli.code_sha, env=env,
                    formal_scar_run=dict(
                        run_id="FORMAL_SCAR_20261103_MP_KITTI",
                        selection_rule_B="max VAL top1@5m, tie max VAL cell R@1, earlier epoch",
                        provenance="NEW TRAJECTORY — not a recovered historical B weight"))

    sel = ckpt_for(best_b, "max VAL top1@5m, tie max VAL cell R@1, earlier epoch")
    out = os.path.join(cli.out_dir, f"ht23_mp_formal_s{cli.seed}.pt")
    torch.save(sel, out)
    a_out = os.path.join(cli.out_dir, "a_rule", f"ht23_cmm_s{cli.seed}.pt")
    os.makedirs(os.path.dirname(a_out), exist_ok=True)
    torch.save(ckpt_for(best_a, "max VAL cell R@1, tie max top1@5m, earlier epoch"), a_out)

    with open(os.path.join(cli.out_dir, f"s{cli.seed}.log.json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(selection_B=sel["selection"],
                       selection_A=dict(epoch=best_a["epoch"],
                                        key=[float(best_a["key"][0]), float(best_a["key"][1])],
                                        val_cell={k: float(v) for k, v in best_a["val_cell"].items()},
                                        val_loc={k: float(v) for k, v in best_a["val_loc"].items()}),
                       history=hist, config=config,
                       n_train_total=n_train_total, n_train_valid=n_train_valid,
                       pinned={F.FEATURE_NAMES[j]: True for j in np.where(pinned)[0]},
                       norm_binding=norm_binding, norm_binding_sha256=norm_binding_sha,
                       train_sha256=train_sha, val_sha256=val_sha,
                       code_sha256=cli.code_sha, env=env),
                  f, indent=2)

    print(f"A selected epoch {best_a['epoch']} key={best_a['key']} -> {a_out}", flush=True)
    print(f"B selected epoch {best_b['epoch']} key={best_b['key']} -> {out}", flush=True)
    print("HEAD_SHA256", sha256(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
