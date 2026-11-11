#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent acceptance of the VAL epoch selection (TRAIN/VAL round).

Nothing is taken from the trainer's own `selection_recomputed` flag:
  - every seed must have all 30 epoch weights and all 30 VAL curve rows;
  - the formal key (VAL Top-1@5m, VAL Cell R@1, -epoch) is recomputed from the
    full curve and the arg-max, and must equal SELECTION_FORMAL.json;
  - the frozen selected head's state_dict must equal that epoch's state_dict
    tensor-for-tensor;
  - the main head must be s0 and its sha256 must be the recorded one;
  - NORMALIZATION.json must have been fitted on this run's TRAIN feature file.
Writes VERIFY_SELECTION_val.json.  No TEST data is touched.
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scar_common as S  # noqa: E402


def state(path):
    o = torch.load(path, map_location="cpu", weights_only=False)
    return o, o.get("state_dict", o)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["npm3d", "s3dis"])
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    run = os.path.abspath(a.run_dir)
    heads = os.path.join(run, "heads")
    sel = S.read_json(os.path.join(heads, "SELECTION_FORMAL.json"))
    norm = S.read_json(os.path.join(heads, "NORMALIZATION.json"))
    checks = {}

    def check(name, ok, **detail):
        checks[name] = dict(PASS=bool(ok), **detail)
        if not ok:
            print(f"FAIL {name}: {detail}", flush=True)
        return ok

    rec = {}
    for tag in ("s0", "s1", "s2"):
        hist_path = os.path.join(heads, f"ht23_{a.domain}_{tag}.log.json")
        hist = S.read_json(hist_path)
        rows = hist["history"]
        epochs = [int(r["epoch"]) for r in rows]
        files = sorted(os.listdir(os.path.join(heads, "epochs", tag)))
        check(f"{tag}_30_curves", len(rows) == 30 and epochs == list(range(1, 31)),
              n_rows=len(rows), epochs=epochs[:3] + epochs[-2:])
        check(f"{tag}_30_weights",
              len(files) == 30 and files[0] == "ep01.pt" and files[-1] == "ep30.pt",
              n_files=len(files))
        # formal: max Top-1@5m, tie max Cell R@1, then earliest epoch
        fkey = lambda r: (float(r["val_loc"]["top1@5m"]),
                          float(r["val_cell"]["R@1"]), -int(r["epoch"]))
        best = max(rows, key=fkey)
        want_epoch = int(sel["seeds"][tag]["formal_epoch"])
        want_key = [round(x, 12) for x in sel["seeds"][tag]["formal_key"]]
        got_key = [round(x, 12) for x in fkey(best)]
        check(f"{tag}_formal_epoch", int(best["epoch"]) == want_epoch,
              recomputed=int(best["epoch"]), recorded=want_epoch)
        check(f"{tag}_formal_key", got_key == want_key,
              recomputed=got_key, recorded=want_key)
        # legacy rule: max Cell R@1, tie Top-1@5m, then earliest
        lkey = lambda r: (float(r["val_cell"]["R@1"]),
                          float(r["val_loc"]["top1@5m"]), -int(r["epoch"]))
        lbest = max(rows, key=lkey)
        check(f"{tag}_legacy_epoch",
              int(lbest["epoch"]) == int(sel["seeds"][tag]["legacy_a_epoch"]),
              recomputed=int(lbest["epoch"]),
              recorded=sel["seeds"][tag]["legacy_a_epoch"])
        # selected head tensors == selected epoch tensors
        hp = os.path.join(heads, f"ht23_{a.domain}_{tag}.pt")
        ep = os.path.join(heads, "epochs", tag, f"ep{want_epoch:02d}.pt")
        ho, hsd = state(hp)
        eo, esd = state(ep)
        same_keys = sorted(hsd) == sorted(esd)
        maxdiff = 0.0
        if same_keys:
            maxdiff = max(float((hsd[k].double() - esd[k].double()).abs().max())
                          for k in hsd)
        check(f"{tag}_head_matches_epoch", same_keys and maxdiff == 0.0,
              n_tensors=len(hsd), keys_equal=same_keys, max_abs_diff=maxdiff)
        check(f"{tag}_head_sha", S.sha256(hp) == sel["seeds"][tag]["sha256"],
              sha256=S.sha256(hp))
        hs = ho["selection"]
        check(f"{tag}_head_val_matches_curve",
              int(hs["epoch"]) == int(best["epoch"])
              and abs(float(hs["val_loc"]["top1@5m"])
                      - float(best["val_loc"]["top1@5m"])) < 1e-12
              and abs(float(hs["val_cell"]["R@1"])
                      - float(best["val_cell"]["R@1"])) < 1e-12)
        rec[tag] = dict(formal_epoch=int(best["epoch"]),
                        formal_key=got_key,
                        head_sha256=S.sha256(hp),
                        n_epoch_history=len(rows), n_epoch_weights=len(files))

    check("main_is_s0", sel.get("main_seed") == 0 and
          sel["main_head_sha256"] == sel["seeds"]["s0"]["sha256"])
    check("main_head_file",
          os.path.realpath(sel["main_head"]) ==
          os.path.realpath(os.path.join(heads, f"ht23_{a.domain}_s0.pt")))
    train_npz = os.path.join(run, "features", "train.npz")
    check("normalisation_fitted_on_train",
          norm["train_npz_sha256"] == S.sha256(train_npz),
          recorded=norm["train_npz_sha256"], actual=S.sha256(train_npz))
    names = list(norm.get("feature_names") or [])
    if not names:
        names = list(S.read_json(os.path.join(run, "features", "train.npz.meta.json"))
                     ["features"])
    pinned = [n for n, p in zip(names, norm["pinned"]) if p]
    check("normalisation_pinned_names", sorted(pinned) == sorted(norm["pinned_names"]),
          pinned=pinned, recorded=norm["pinned_names"])
    freeze = S.read_json(os.path.join(run, "FREEZE_LIST.json"))
    check("protocol_sha", sel["protocol_freeze_sha256"]
          == freeze["protocol"]["protocol_sha256"],
          recorded=sel["protocol_freeze_sha256"])
    out = dict(
        status="OK" if all(v["PASS"] for v in checks.values()) else "FAIL",
        domain=a.domain, run_dir=run,
        selection_sha256=S.sha256(os.path.join(heads, "SELECTION_FORMAL.json")),
        normalisation_sha256=S.sha256(os.path.join(heads, "NORMALIZATION.json")),
        protocol_freeze_sha256=sel["protocol_freeze_sha256"],
        seeds=rec, checks=checks,
        failures=[k for k, v in checks.items() if not v["PASS"]],
        test_read=False,
        code_sha256={f: S.sha256(os.path.join(HERE, f))
                     for f in ("d4p_verify_selection.py", "scar_common.py")},
        verified_at=__import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    S.write_json(out, a.out)
    print(json.dumps({k: out[k] for k in ("status", "domain", "failures", "seeds")},
                     indent=2), flush=True)
    print("saved", a.out, flush=True)
    return 0 if out["status"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
