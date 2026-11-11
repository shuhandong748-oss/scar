"""LSEL fixed-head completion: local CPU replica of the Text2Loc x KITTI ht23 head.

Derived verbatim from the recovered A-machine script `source/train_head_kitti.py`
(sha256 8965e40a...).  Only the following are added:
  * every epoch's state_dict is written to <out_dir>/epochs/s<seed>/epochNN.pt
  * after the 30-epoch run, BOTH selection rules are applied to the SAME new
    trajectory:  A = (cell R@1, top1@5m); B = (top1@5m, cell R@1); ties earliest.
  * newA / newB heads are saved with the original checkpoint schema.
Network, loss, optimiser, batch order, normalisation, epochs and seeds are
unchanged.  VAL is used for epoch selection only.  TEST is never opened.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "source"))
sys.path.insert(0, os.path.join(ROOT, "source", "vendor"))
import common_kitti as C


def sha256(path):
    h = __import__("hashlib").sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
            {f"top{k}@5m": float(loc[k].mean()) for k in loc},
            rk, loc)


def pick(hist, cell_get, top_get):
    best = None
    for r in hist:
        key = (float(cell_get(r)), float(top_get(r)))
        if best is None or key > best[0]:
            best = (key, int(r["epoch"]))
    return best


def build_ckpt(state_dict, mean, std, pinned, seed, K, epochs, hist,
               sel_epoch, sel_key, sel_rule, F, train_npz, val_npz):
    rec = [r for r in hist if r["epoch"] == sel_epoch][0]
    return dict(state_dict=state_dict, mean=mean, std=std,
                pinned=pinned.astype(bool), feature_names=F.FEATURE_NAMES,
                config=dict(epochs=epochs, lr=1e-3, wd=1e-4, batch=512,
                            hidden=[64, 32], loss="listwise_softmax_ce",
                            seed=seed, k=K, dataset="kitti",
                            backbone="original_text2loc",
                            norms=dict(pool_norm=24, rank_norm=24, tied_norm=25),
                            selection=sel_rule),
                selection=dict(rule=sel_rule, epoch=sel_epoch,
                               key=[float(sel_key[0]), float(sel_key[1])],
                               val_cell={k: float(v) for k, v in rec["val_cell"].items()},
                               val_loc={k: float(v) for k, v in rec["val_loc"].items()}),
                history=hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_npz", required=True)
    ap.add_argument("--val_npz", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn
    import features_fix as F

    tr = load(args.train_npz)
    va = load(args.val_npz)
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

    os.makedirs(os.path.join(args.out_dir, "heads"), exist_ok=True)
    vx = va["feats"].astype(np.float32)
    out = dict(recipe=dict(epochs=args.epochs, lr=1e-3, wd=1e-4, batch=512,
                           hidden=[64, 32], loss="listwise_softmax_ce",
                           selection_A="max VAL cell R@1, tie VAL top1@5m, earlier epoch",
                           selection_B="max VAL top1@5m, tie VAL cell R@1, earlier epoch",
                           head_seeds=[int(s) for s in args.seeds.split(",")],
                           device="cpu", torch=torch.__version__,
                           numpy=np.__version__,
                           torch_threads=int(torch.get_num_threads())),
               inputs=dict(train_npz=args.train_npz, val_npz=args.val_npz,
                           train_sha256=sha256(args.train_npz),
                           val_sha256=sha256(args.val_npz)),
               source_script_sha256=sha256(os.path.join(ROOT, "source",
                                                        "train_head_kitti.py")),
               runs={}, trajectory={})

    for seed in [int(s) for s in args.seeds.split(",")]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.use_deterministic_algorithms(True)
        model = Net()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        rng = np.random.RandomState(seed)
        epoch_dir = os.path.join(args.out_dir, "epochs", f"s{seed}")
        os.makedirs(epoch_dir, exist_ok=True)
        best = None
        hist = []

        for ep in range(1, args.epochs + 1):
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
            sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            torch.save(dict(epoch=ep, state_dict=sd, mean=mean, std=std,
                            pinned=pinned.astype(bool), feature_names=F.FEATURE_NAMES,
                            val_record=rec),
                       os.path.join(epoch_dir, "epoch%02d.pt" % ep))
            print(f"epoch {ep:02d}/{args.epochs} loss {tot/cnt:.4f} train_top1 {tr_hit:.4f} "
                  f"| val R1 {rm['R@1']:.4f} R3 {rm['R@3']:.4f} R5 {rm['R@5']:.4f} "
                  f"top1@5m {lm['top1@5m']:.4f} top3@5m {lm['top3@5m']:.4f}", flush=True)
            if best is None or key > best["key"]:
                best = dict(key=key, epoch=ep, val_cell=rm, val_loc=lm)

        b_key, b_ep = pick(hist, lambda r: r["val_loc"]["top1@5m"],
                           lambda r: r["val_cell"]["R@1"])
        for tag, ep, key, rule in (
                ("newA", best["epoch"], best["key"],
                 "max VAL cell R@1, tie VAL top1@5m, earlier epoch"),
                ("newB", b_ep, b_key,
                 "max VAL top1@5m, tie VAL cell R@1, earlier epoch")):
            ep_path = os.path.join(epoch_dir, "epoch%02d.pt" % ep)
            sd = torch.load(ep_path, map_location="cpu", weights_only=False)["state_dict"]
            ckpt = build_ckpt(sd, mean, std, pinned, seed, K, args.epochs, hist,
                              ep, key, rule, F, args.train_npz, args.val_npz)
            ckpt["n_train_total"] = int(len(valid))
            ckpt["n_train_valid"] = int(valid.sum())
            ckpt["train_npz"] = args.train_npz
            ckpt["val_npz"] = args.val_npz
            ckpt["train_sha256"] = sha256(args.train_npz)
            ckpt["val_sha256"] = sha256(args.val_npz)
            p = os.path.join(args.out_dir, "heads", f"{tag}_ht23_t2l_s{seed}.pt")
            torch.save(ckpt, p)
            out["runs"][f"{tag}_ht23_t2l_s{seed}"] = dict(
                path=p, sha256=sha256(p), epoch=ep,
                key=[float(key[0]), float(key[1])],
                selection_rule=rule, epoch_file=ep_path,
                epoch_file_sha256=sha256(ep_path))
            print(f"saved {p} epoch={ep} key={key}", flush=True)
        out["trajectory"][f"s{seed}"] = dict(
            history=hist,
            pinned=[F.FEATURE_NAMES[j] for j in np.where(pinned)[0]])
        with open(os.path.join(args.out_dir, "retrain_log.json"), "w") as f:
            json.dump(out, f, indent=1)
    print("RETRAIN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
