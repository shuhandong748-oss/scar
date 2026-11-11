#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the three SCAR heads for one A domain on FEATURE_COORD_FIX_v1.

Frozen recipe (CORR-D4P-SCAR-FORMAL-v1):
  23 -> 64 -> 32 -> 1, ReLU; listwise softmax CE; AdamW lr 1e-3 wd 1e-4;
  batch 512; 30 epochs; CPU.
  mean/std fitted on TRAIN *valid* lists only, flattened 23, std<1e-3 pinned 1;
  normalisers shared by the three seeds.
Formal selection: max VAL Top-1@5m -> max VAL Cell R@1 -> earliest epoch.
s0 is the main head; s1/s2 are stability only.  TRAIN/VAL only, no TEST.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a_scar_common as S  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=sorted(S.DOMAINS))
    ap.add_argument("--train_npz", required=True)
    ap.add_argument("--val_npz", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--protocol_sha", required=True)
    ap.add_argument("--coord_fix_sha", default=None)
    ap.add_argument("--export_dir", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args(argv)
    out_dir = os.path.abspath(a.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, "SELECTION_FORMAL.json")):
        raise SystemExit(f"{out_dir} already holds a finished head set; refusing to "
                         f"overwrite (frozen evidence)")
    d = S.domain_cfg(a.domain)
    S.insert_feature_paths(a.domain)
    import features_fix as F
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn

    tr = S.load_npz(a.train_npz)
    va = S.load_npz(a.val_npz)
    if tr["feats"].shape[1] != 25 or tr["feats"].shape[2] != 23:
        raise SystemExit(f"unexpected feature shape {tr['feats'].shape}")
    if list(tr["feature_names"]) != list(F.FEATURE_NAMES):
        raise SystemExit("TRAIN feature names != canonical 23")
    if list(va["feature_names"]) != list(F.FEATURE_NAMES):
        raise SystemExit("VAL feature names != canonical 23")
    if str(tr["split"][0]) != "train" or str(va["split"][0]) != "val":
        raise SystemExit("train/val feature files swapped")
    if len(tr["feats"]) != d["expected"]["train"]:
        raise SystemExit(f"TRAIN lists {len(tr['feats'])} != {d['expected']['train']}")
    if len(va["feats"]) != d["expected"]["val"]:
        raise SystemExit(f"VAL lists {len(va['feats'])} != {d['expected']['val']}")
    K, nf = 25, 23

    valid = tr["valid"].astype(bool)
    feats = tr["feats"][valid].astype(np.float32)
    labels = tr["labels"][valid].astype(np.float32)
    if len(feats) == 0:
        raise SystemExit("no valid TRAIN lists")
    pos = labels.sum(1)
    if pos.min() < 1 or pos.max() >= K:
        raise SystemExit("TRAIN valid masks broken")
    print(f"train valid lists {len(feats)}/{len(valid)} | mean positives/list "
          f"{pos.mean():.2f}", flush=True)

    flat = feats.reshape(-1, nf)
    mean = flat.mean(0).astype(np.float32)
    std = flat.std(0)
    pinned = std < 1e-3
    std = np.where(pinned, 1.0, std).astype(np.float32)
    pinned_names = [F.FEATURE_NAMES[j] for j in np.where(pinned)[0]]
    print(f"pinned columns: {pinned_names}", flush=True)
    norm = {
        "mean": mean.tolist(), "std": std.tolist(), "pinned": pinned.tolist(),
        "pinned_names": pinned_names, "feature_names": list(F.FEATURE_NAMES),
        "feature_version": S.FEATURE_VERSION,
        "fitted_on": "TRAIN valid lists only, flattened 23 features, std<1e-3 pinned to 1",
        "train_valid_lists": int(valid.sum()),
        "train_npz_sha256": S.sha256(a.train_npz),
    }
    S.write_json(norm, os.path.join(out_dir, "NORMALIZATION.json"))

    x = torch.from_numpy((feats - mean) / std)
    y = torch.from_numpy(labels)
    vx = torch.from_numpy((va["feats"].astype(np.float32) - mean) / std)
    vp2 = va["p2_idx"].astype(np.int16)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(nf, 64), nn.ReLU(),
                                     nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, z):
            return self.net(z).squeeze(-1)

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    selection = {"main_seed": 0, "seeds": {},
                 "formal_rule": "max VAL Top-1@5m, tie max VAL Cell R@1, "
                                "then earliest epoch",
                 "feature_version": S.FEATURE_VERSION,
                 "protocol_freeze_sha256": a.protocol_sha,
                 "coord_fix_sha256": a.coord_fix_sha}
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = Net()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        rng = np.random.RandomState(seed)
        epoch_dir = os.path.join(out_dir, "epochs", f"s{seed}")
        os.makedirs(epoch_dir, exist_ok=True)
        hist = []
        for ep in range(1, a.epochs + 1):
            perm = rng.permutation(len(x))
            model.train()
            tot, cnt = 0.0, 0
            for b in range(0, len(x), 512):
                idx = torch.from_numpy(perm[b:b + 512]).long()
                logits = model(x[idx])
                logp = Fn.log_softmax(logits, dim=1)
                target = y[idx] / y[idx].sum(1, keepdim=True)
                loss = -(target * logp).sum(1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * len(idx)
                cnt += len(idx)
            model.eval()
            with torch.no_grad():
                tr_pred = model(x).argmax(1).numpy()[:, None]
                tr_hit = float(np.take_along_axis(labels > 0, tr_pred, axis=1).mean())
                v_logits = model(vx).numpy()
            rm, lm, _, _ = S.val_metrics(S.rank_orders(v_logits, vp2), va, K)
            rec = dict(epoch=ep, loss=tot / cnt, train_top1=tr_hit,
                       val_cell=rm, val_loc=lm)
            hist.append(rec)
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            torch.save({"epoch": ep, "state_dict": state, "val_cell": rm,
                        "val_loc": lm, "loss": tot / cnt, "train_top1": tr_hit,
                        "seed": seed, "feature_version": S.FEATURE_VERSION},
                       os.path.join(epoch_dir, f"ep{ep:02d}.pt"))
            print(f"[s{seed}] epoch {ep:02d}/{a.epochs} loss {tot/cnt:.4f} "
                  f"train_top1 {tr_hit:.4f} | val R1 {rm['R@1']:.4f} "
                  f"R2 {rm['R@2']:.4f} R5 {rm['R@5']:.4f} R10 {rm['R@10']:.4f} "
                  f"R25 {rm['R@25']:.4f} | top1@5m {lm['top1@5m']:.4f} "
                  f"top5@5m {lm['top5@5m']:.4f}", flush=True)

        sel_epoch = max(hist, key=lambda r: (r["val_loc"]["top1@5m"],
                                             r["val_cell"]["R@1"], -r["epoch"]))["epoch"]
        best = hist[sel_epoch - 1]
        state = torch.load(os.path.join(epoch_dir, f"ep{sel_epoch:02d}.pt"),
                           map_location="cpu", weights_only=False)["state_dict"]
        ckpt = dict(
            state_dict=state, mean=mean, std=std, pinned=pinned.astype(bool),
            feature_names=list(F.FEATURE_NAMES),
            feature_version=S.FEATURE_VERSION,
            config=dict(domain=a.domain, epochs=a.epochs, lr=1e-3, wd=1e-4,
                        batch=512, hidden=[64, 32], loss="listwise_softmax_ce",
                        seed=seed, k=K,
                        norms=dict(pool_norm=24, rank_norm=24, tied_norm=25)),
            selection=dict(rule_expr="(val_loc.top1@5m, val_cell.R@1, -epoch)",
                           epoch=sel_epoch,
                           key=[best["val_loc"]["top1@5m"],
                                best["val_cell"]["R@1"], -sel_epoch],
                           val_cell=best["val_cell"], val_loc=best["val_loc"]),
            history=hist, train_npz=a.train_npz, val_npz=a.val_npz,
            train_npz_sha256=S.sha256(a.train_npz),
            val_npz_sha256=S.sha256(a.val_npz),
            train_valid_lists=int(valid.sum()),
            normalisation=dict(mean=mean.tolist(), std=std.tolist(),
                               pinned=pinned.tolist(),
                               fitted_on="TRAIN valid lists only, flatten 23, "
                                         "std<1e-3 pinned to 1"),
            protocol_freeze_sha256=a.protocol_sha,
            coord_fix_sha256=a.coord_fix_sha,
            export_dir=a.export_dir,
            code_sha256={f: S.sha256(os.path.join(HERE, f))
                         for f in ("train_head_coordfix_v1.py", "a_scar_common.py",
                                   "feature_coord_fix_v1.py")},
        )
        head_path = os.path.join(out_dir, f"ht23_{a.domain}_s{seed}.pt")
        torch.save(ckpt, head_path)
        S.write_json(dict(selection=ckpt["selection"], history=hist,
                          config=ckpt["config"],
                          normalisation=ckpt["normalisation"],
                          pinned={n: True for n in pinned_names},
                          feature_version=S.FEATURE_VERSION,
                          train_npz=a.train_npz, val_npz=a.val_npz),
                     os.path.join(out_dir, f"ht23_{a.domain}_s{seed}.log.json"))
        selection["seeds"][f"s{seed}"] = {
            "head": head_path, "sha256": S.sha256(head_path),
            "formal_epoch": sel_epoch,
            "formal_key": ckpt["selection"]["key"],
            "formal_val_cell": best["val_cell"],
            "formal_val_loc": best["val_loc"],
            "selection_recomputed": True,
        }
        print(f"[s{seed}] formal epoch {sel_epoch} -> {head_path}", flush=True)

    selection["main_head"] = selection["seeds"]["s0"]["head"]
    selection["main_head_sha256"] = selection["seeds"]["s0"]["sha256"]
    selection["normalization_sha256"] = S.sha256(
        os.path.join(out_dir, "NORMALIZATION.json"))
    selection["train_features_sha256"] = S.sha256(a.train_npz)
    selection["val_features_sha256"] = S.sha256(a.val_npz)
    S.write_json(selection, os.path.join(out_dir, "SELECTION_FORMAL.json"))
    print("saved", os.path.join(out_dir, "SELECTION_FORMAL.json"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
