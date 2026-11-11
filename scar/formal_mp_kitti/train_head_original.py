"""Train one H-T (SCAR 23-feature) head seed on MambaPlace x KITTI TRAIN lists.

Protocol R057-KITTI-CMMLOC-TARGET-HEAD-v1:
  23 -> 64 -> 32 -> 1, ReLU, listwise softmax CE, AdamW lr 1e-3 wd 1e-4,
  batch 512, 30 epochs, CPU.  Mean/std fitted on TRAIN *valid* lists only
  (1 <= positives < 25); std < 1e-3 pinned to 1.0.
  VAL selection = max cell R@1, tie max top1@5m, then earliest epoch.
VAL labels are used for epoch selection only, never for fitting.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common_mp as C  # noqa: E402


def sha256(path):
    return C.sha256(path)


def load(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def eval_orders(feats, mean, std, model, torch):
    x = torch.from_numpy((feats - mean) / std)
    with torch.no_grad():
        return model(x).numpy()


def val_metrics(logits, data, F):
    n = len(logits)
    p2 = data["p2_idx"]
    ids = data["cell_ids"]
    gt = data["gt_cell"]
    pw = data["pred_w"].astype(np.float64)
    gp = data["gt_pose_w"].astype(np.float64)
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
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    cli = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn
    sys.path.insert(0, C.VENDOR)
    import features_fix as F

    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    torch.use_deterministic_algorithms(True)
    os.makedirs(cli.out_dir, exist_ok=True)

    tr = load(cli.train_npz)
    va = load(cli.val_npz)
    nf = tr["feats"].shape[2]
    K = tr["feats"].shape[1]
    assert nf == 23 and K == 25
    assert list(tr["feature_names"]) == list(F.FEATURE_NAMES)
    assert list(va["feature_names"]) == list(F.FEATURE_NAMES)
    assert str(tr["split"][0]) == "train" and str(va["split"][0]) == "val"
    assert len(tr["feats"]) == C.EXPECTED["train"], len(tr["feats"])
    assert len(va["feats"]) == C.EXPECTED["val"], len(va["feats"])
    valid = tr["valid"]
    feats = tr["feats"][valid].astype(np.float32)
    labels = tr["labels"][valid].astype(np.float32)
    print(f"train lists {len(feats)}/{len(valid)} valid | positives/list "
          f"{labels.sum(1).mean():.2f}", flush=True)
    assert labels.sum(1).min() >= 1
    assert labels.sum(1).max() < K

    flat = feats.reshape(-1, nf)
    mean = flat.mean(0)
    std = flat.std(0)
    pinned = std < 1e-3
    std = np.where(pinned, 1.0, std).astype(np.float32)
    mean = mean.astype(np.float32)
    print(f"pinned columns: {[F.FEATURE_NAMES[j] for j in np.where(pinned)[0]]}", flush=True)

    x = torch.from_numpy((feats - mean) / std)
    y = torch.from_numpy(labels)

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

    vx = va["feats"].astype(np.float32)
    best = None
    hist = []

    for ep in range(1, cli.epochs + 1):
        perm = rng.permutation(len(x))
        model.train()
        tot, cnt = 0.0, 0
        for b in range(0, len(x), 512):
            idx = torch.from_numpy(perm[b:b + 512]).long()
            xb, yb = x[idx], y[idx]
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
        with torch.no_grad():
            tr_logits = model(x)
            tr_pred = tr_logits.argmax(1).numpy()[:, None]
            tr_hit = np.take_along_axis(labels > 0, tr_pred, axis=1).mean()
        vl = eval_orders(vx, mean, std, model, torch)
        rm, lm, _, _ = val_metrics(vl, va, F)
        key = (rm["R@1"], lm["top1@5m"])
        rec = dict(epoch=ep, loss=tot / cnt, train_top1=float(tr_hit),
                   val_cell=rm, val_loc=lm)
        hist.append(rec)
        print(f"epoch {ep:02d}/{cli.epochs} loss {tot/cnt:.4f} train_top1 {tr_hit:.4f} "
              f"| val R1 {rm['R@1']:.4f} R3 {rm['R@3']:.4f} R5 {rm['R@5']:.4f} "
              f"top1@5m {lm['top1@5m']:.4f} top3@5m {lm['top3@5m']:.4f}", flush=True)
        if best is None or key > best["key"]:
            best = dict(key=key, epoch=ep, val_cell=rm, val_loc=lm,
                        state_dict={k: v.detach().clone()
                                    for k, v in model.state_dict().items()})

    ckpt = dict(state_dict=best["state_dict"], mean=mean, std=std,
                pinned=pinned.astype(bool), feature_names=F.FEATURE_NAMES,
                config=dict(epochs=cli.epochs, lr=1e-3, wd=1e-4, batch=512,
                            hidden=[64, 32], loss="listwise_softmax_ce",
                            seed=cli.seed, k=K, dataset="kitti",
                            backbone=C.BACKBONE_TAG,
                            norms=dict(pool_norm=24, rank_norm=24, tied_norm=25)),
                selection=dict(rule="max VAL cell R@1, tie max top1@5m, earlier epoch",
                               epoch=best["epoch"],
                               key=[float(best["key"][0]), float(best["key"][1])],
                               val_cell={k: float(v) for k, v in best["val_cell"].items()},
                               val_loc={k: float(v) for k, v in best["val_loc"].items()}),
                history=hist,
                n_train_total=int(len(valid)), n_train_valid=int(valid.sum()),
                train_npz=cli.train_npz, val_npz=cli.val_npz,
                train_sha256=sha256(cli.train_npz), val_sha256=sha256(cli.val_npz))
    out = os.path.join(cli.out_dir, f"ht23_cmm_s{cli.seed}.pt")
    torch.save(ckpt, out)
    with open(os.path.join(cli.out_dir, f"ht23_cmm_s{cli.seed}.log.json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(selection=ckpt["selection"], history=hist,
                       config=ckpt["config"], n_train_total=ckpt["n_train_total"],
                       n_train_valid=ckpt["n_train_valid"],
                       pinned={F.FEATURE_NAMES[j]: True for j in np.where(pinned)[0]},
                       train_sha256=ckpt["train_sha256"],
                       val_sha256=ckpt["val_sha256"]),
                  f, indent=2)
    print(f"selected epoch {best['epoch']} key={best['key']} -> {out}", flush=True)
    print("HEAD_SHA256", sha256(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
