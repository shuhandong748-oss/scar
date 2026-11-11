"""Formal SCAR (B-rule) head training — MambaPlace x KITTI TRAIN/VAL only.

Derived from the recovered original ``post/train_head.py``
(SHA256 29656fb0... ; recovery id E_WEIGHT_CACHE_RECOVERY_20261103).

MINIMAL, DOCUMENTED DEVIATIONS from the original file (nothing else changed):
  D0  Local path adaptation: resolve the vendored ``features_fix`` module next
      to this file (the recovered ``common_mp.py`` hard-codes the remote
      ``/root/autodl-tmp`` VENDOR path).  The file is byte-identical.
  D1  ``--out_dir`` layout: writes one state_dict per epoch to
      ``<out_dir>/per_epoch/`` (the original kept only the in-memory A-best).
  D2  Formal selection is B-rule: max VAL top1@5m -> tie max VAL cell R@1 ->
      earliest epoch.  The A-rule (max cell R@1 -> tie top1@5m -> earliest) is
      still computed and its head saved under ``<out_dir>/a_rule/`` purely for
      provenance / comparison against the recovered historical heads.
  D3  Extra provenance fields (code SHA, input SHAs, environment, per-epoch
      binding info) are written into every checkpoint.  NO change to the
      model, loss, optimiser, batch, lr, wd, epochs, seeds, normalisation or
      data order.
  D4  Executor pinned to one CPU thread (task requirement).
  D5  Memory-only: npz members are read selectively, and the (x-mean)/std
      normalisation is done in place / once for VAL.  The individual float32
      operations are unchanged.

Recipe (unchanged, verbatim from the original):
  23 -> 64 -> 32 -> 1, ReLU, listwise softmax CE, AdamW lr 1e-3 wd 1e-4,
  batch 512, 30 epochs, CPU.  Mean/std fitted on TRAIN *valid* lists only
  (1 <= positives < 25); std < 1e-3 pinned to 1.0.
  VAL labels are used for epoch selection only, never for fitting.
"""
import argparse
import json
import os
import platform
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common_mp as C  # noqa: E402


def sha256(path):
    return C.sha256(path)


def load(path, keys=None):
    z = np.load(path, allow_pickle=True)
    ks = z.files if keys is None else keys
    out = {k: z[k] for k in ks}
    z.close()
    return out


def eval_orders(feats_norm, model, torch):
    # D5: the caller pre-normalises once, in place, with exactly the same
    # float32 (x - mean) / std operations as the original per-epoch expression.
    x = torch.from_numpy(feats_norm)
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
    ap.add_argument("--expected_train_sha", default="")
    ap.add_argument("--expected_val_sha", default="")
    ap.add_argument("--code_sha", default="")
    cli = ap.parse_args()

    # --- determinism: single CPU thread (task requirement) -------------------
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn
    torch.set_num_threads(1)
    # Local path adaptation (D0): the recovered common_mp.py points VENDOR at
    # /root/autodl-tmp/...; use the byte-identical vendored copy beside this file.
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

    # D5 (memory only): read just the fields this trainer needs.  npz members
    # are decompressed lazily, so unneeded fields (texts/cell_ids on TRAIN,
    # offsets/cos/sp/... on both splits) are never allocated.  Numerics unchanged.
    tr = load(cli.train_npz, ["feats", "labels", "valid", "feature_names", "split"])
    va = load(cli.val_npz, ["feats", "p2_idx", "cell_ids", "gt_cell", "scene",
                            "gt_pose_w", "pred_w", "feature_names", "split"])
    nf = tr["feats"].shape[2]
    K = tr["feats"].shape[1]
    assert nf == 23 and K == 25
    assert list(tr["feature_names"]) == list(F.FEATURE_NAMES)
    assert list(va["feature_names"]) == list(F.FEATURE_NAMES)
    assert str(tr["split"][0]) == "train" and str(va["split"][0]) == "val"
    assert len(tr["feats"]) == C.EXPECTED["train"], len(tr["feats"])
    assert len(va["feats"]) == C.EXPECTED["val"], len(va["feats"])

    train_sha = sha256(cli.train_npz)
    val_sha = sha256(cli.val_npz)
    if cli.expected_train_sha:
        assert train_sha == cli.expected_train_sha, (train_sha, cli.expected_train_sha)
    if cli.expected_val_sha:
        assert val_sha == cli.expected_val_sha, (val_sha, cli.expected_val_sha)

    valid = tr["valid"]
    n_train_total = int(len(valid))
    n_train_valid = int(valid.sum())
    feats = tr["feats"][valid].astype(np.float32)
    labels = tr["labels"][valid].astype(np.float32)
    del tr  # D5: release the 66 MB full TRAIN features now that the copy exists
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

    norm_binding = dict(
        mean=[float(v) for v in mean], std=[float(v) for v in std],
        pinned=[bool(v) for v in pinned], feature_names=list(F.FEATURE_NAMES),
        source_split="train", source_mask="valid (1 <= positives < 25)",
        n_train_valid=n_train_valid, n_train_total=n_train_total,
        train_sha256=train_sha, val_sha256=val_sha,
        mean_sha256=C.hashlib.sha256(
            np.asarray(mean, np.float32).tobytes()).hexdigest(),
        std_sha256=C.hashlib.sha256(
            np.asarray(std, np.float32).tobytes()).hexdigest(),
        definition_sha256=sha256(os.path.join(VENDOR, "features_fix.py")),
    )
    norm_binding_sha = C.hashlib.sha256(
        json.dumps(norm_binding, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    # D5: in-place normalisation (identical float32 subtract/divide; avoids a
    # full-size temporary under the local machine's tight commit limit).
    feats -= mean
    feats /= std
    del flat
    x = torch.from_numpy(feats)
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

    vx = va["feats"]
    if vx.dtype != np.float32:
        vx = vx.astype(np.float32)
    vx -= mean            # D5: pre-normalise once, in place (same operations)
    vx /= std
    best_a = None
    best_b = None
    hist = []

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
               platform=platform.platform())

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
        vl = eval_orders(vx, model, torch)
        rm, lm, _, _ = val_metrics(vl, va, F)
        key_a = (rm["R@1"], lm["top1@5m"])
        key_b = (lm["top1@5m"], rm["R@1"])
        rec = dict(epoch=ep, loss=tot / cnt, train_top1=float(tr_hit),
                   val_cell=rm, val_loc=lm)
        hist.append(rec)
        print(f"epoch {ep:02d}/{cli.epochs} loss {tot/cnt:.4f} train_top1 {tr_hit:.4f} "
              f"| val R1 {rm['R@1']:.4f} R3 {rm['R@3']:.4f} R5 {rm['R@5']:.4f} "
              f"top1@5m {lm['top1@5m']:.4f} top3@5m {lm['top3@5m']:.4f}", flush=True)

        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        # D1: per-epoch weight + binding info (does not touch RNG/optimiser)
        ep_ckpt = dict(
            state_dict=sd, epoch=ep, seed=cli.seed,
            config=config, mean=mean, std=std, pinned=pinned.astype(bool),
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
                    pinned=pinned.astype(bool), feature_names=F.FEATURE_NAMES,
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
    a_out = os.path.join(cli.out_dir, f"a_rule", f"ht23_cmm_s{cli.seed}.pt")
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
