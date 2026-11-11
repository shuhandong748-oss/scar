"""R059/E post-processing constants: MambaPlace x KITTI360Pose bb0 (KITTI self-trained).

Ported from the R057 CMMLoc post-processing chain (`a_machine/cmmloc_kitti_selftrain/scripts/`).
**Only the backbone adapter differs**: paths, the frozen-checkpoint lookup, and the
identity strings.  The 23-column feature definition, the normalisers, the heads, the
evaluator, the verifier and the export validators are the same files, byte-identical
except for documented label/import lines (see PORT_NOTES.md).

The frozen core (`code/MambaPlace`, 35 controlled files, manifest sha
789988fe124d065f6d0c49bdc812a57c90fceb5db97603548f07027de58d6c01) is read-only here:
nothing in this directory patches it, and the running training dir is never written.

Selection is already fixed in the training logs, not recomputed here:
  coarse = VAL best-cell R@5 (strict >, earliest)  -> epoch 7, acc 0.648
  fine   = min mean VAL pose offset (strict <, earliest) -> filled in at freeze time
The official trainers delete the previous best checkpoint, so exactly one .pth is
left in each folder; `frozen_ckpt()` asserts that rather than guessing.
"""
import hashlib
import os

os.environ.setdefault("MP_SEED", "0")
os.environ.setdefault("MP_NO_TEST", "1")

ROOT = "/root/autodl-tmp/mambaplace_kitti_selftrain"
CODE = ROOT + "/code/MambaPlace"          # frozen core @ f852d9c + COMPAT patches
DATASET_NAME = "k360_30-10_scG_pd10_pc4_spY_all"
DATA = "/root/autodl-tmp/Des4Pos/data/KITTI360Pose/" + DATASET_NAME   # == CODE/data/<ds> symlink
VENDOR = ROOT + "/post/vendor"            # canonical 23-dim feature modules (copied, read-only)
CACHE = ROOT + "/cache"
COARSE_RUN = ROOT + "/runs/R059_MP_KITTI_bb0_coarse_v1"
FINE_RUN = ROOT + "/runs/R059_MP_KITTI_bb0_fine_v1"
COARSE_FOLDER = "R059_MP_KITTI_bb0_coarse_v1"
FINE_FOLDER = "R059_MP_KITTI_bb0_fine_v1"

T5 = "/root/autodl-tmp/Des4Pos/models/t5-large"
POINTNET = CODE + "/checkpoints/pointnet_acc0.86_lr1_p256.pth"

K = 25
THRESHS = (5, 10, 15)
KS_CELL = (1, 2, 3, 5, 10, 25)
EXPECTED = {"train": 28807, "val": 3187, "test": 11505}
EXPECTED_CELLS = {"train": 11259, "val": 1434, "test": 4308}

TRAIN = ["2013_05_28_drive_0000_sync", "2013_05_28_drive_0002_sync",
         "2013_05_28_drive_0004_sync", "2013_05_28_drive_0006_sync",
         "2013_05_28_drive_0007_sync"]
VAL = ["2013_05_28_drive_0010_sync"]
TEST = ["2013_05_28_drive_0003_sync", "2013_05_28_drive_0005_sync",
        "2013_05_28_drive_0009_sync"]

SCENES = {"train": TRAIN, "val": VAL, "test": TEST}

# Provenance strings written into the export/eval metadata.  MambaPlace is a
# self-trained two-stage model here: BOTH stages come from this task's own runs.
BACKBONE = ("MambaPlace (CellRetrievalNetwork coarse + CrossMatch fine, self-trained, "
            "CV4RA/MambaPlace @f852d9c)")
COARSE_LABEL = ("MambaPlace public Coarse (CellRetrievalNetwork, self-trained from "
                "CV4RA/MambaPlace @f852d9c)")
FINE_LABEL = "self-trained MambaPlace CrossMatch offsets for all 25 candidates (shared)"
BACKBONE_TAG = "mambaplace_selftrain_coarse_plus_selftrain_fine"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_all(seed):
    """Torch / CUDA / numpy / python RNG, fixed for per-chunk reproducible forwards."""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scene_of(cell_id):
    return str(cell_id).split("_")[0]


def load_split_dataset(split):
    """Frozen order, CPU, no hint shuffling / pose flipping (official eval flags)."""
    import sys
    sys.path.insert(0, CODE)
    sys.path.insert(0, VENDOR)
    os.chdir(CODE)
    from torch_geometric.transforms import FixedPoints
    from dataloading.kitti360pose.cells import Kitti360CoarseDatasetMulti
    ds = Kitti360CoarseDatasetMulti(DATA, SCENES[split], FixedPoints(256),
                                    shuffle_hints=False, flip_poses=False)
    return ds, ds.all_poses, ds.all_cells


def ckpt_dir(folder):
    return os.path.join(CODE, "checkpoints", DATASET_NAME, folder)


def frozen_ckpt(folder):
    """Exactly one checkpoint is left by the official trainers (previous best deleted)."""
    d = ckpt_dir(folder)
    fs = sorted(f for f in os.listdir(d) if f.endswith(".pth"))
    assert len(fs) == 1, (d, fs)
    return os.path.join(d, fs[0])


def query_ids(poses):
    cursor = {}
    out = []
    for p in poses:
        sc = str(p.scene_name)
        k = cursor.get(sc, 0)
        out.append(f"{sc}:{k:04d}")
        cursor[sc] = k + 1
    assert len(set(out)) == len(out)
    return out


def coarse_npz(split):
    return os.path.join(CACHE, split, "coarse.npz")


def fine_npz(split):
    return os.path.join(CACHE, split, "fine.npz")


def fine_chunks(split):
    return os.path.join(CACHE, split, "fine_chunks")


def export_dir(split):
    return os.path.join(CACHE, split)


def log_event(path, **kw):
    import json
    import time
    kw = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), **kw)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(kw, ensure_ascii=False) + "\n")
    print("EVENT", json.dumps(kw, ensure_ascii=False), flush=True)
