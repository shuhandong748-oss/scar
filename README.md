# SCAR Reproduction Release

SCAR is a **companion reranker** that sits on top of a visual place-recognition (VPR)
backbone. Given the candidate cells produced by the backbone and a text query, it reranks
the candidates with a 23-dimensional canonical feature vector. This repository publishes
the code, configs, normalization files and result-binding manifests for 25 experimental
cells (5 backbones × 5 datasets).

This repository contains **no weights, datasets, feature caches, raw logs or credentials**.
The A-tier weights (74 files) are hosted in a private Hugging Face model repository,
referenced by SHA256 and by exact remote path in `WEIGHT_INDEX.csv`
(see [Weight index and downloads](#weight-index-and-downloads)).

## Layout

```
scar/core/               corrected Des4Pos SCAR pipeline (export → features → train → eval → verify)
scar/round_a/            corrected A-domain (PandaSet / nuScenesFull) coordinate-fix chain
scar/test_d4p/           corrected four-domain TEST export/eval (single-read policy)
scar/formal_mp_kitti/    MambaPlace × KITTI formal SCAR trajectory
scar/test_append_mp_kitti/  MP × KITTI s0 ep5 append-TEST evidence
scar/test13_kitti/       KITTI test13 evidence package
scar/lsel/               LSEL fixed-head retraining and TEST evaluation
scar/tables/             result-table builders
configs/<cell>/          01__ENVIRONMENT.json / 02__SELECTION_RECORD.json / V3_CELL_SUMMARY.json
normalization/<cell>/    NORMALIZATION.json (some cells embed it in the head file)
results/                 RESULTS_SOURCE.csv / TOP1_5M.csv / CELL_R1.csv / multi-threshold long table
provenance/              weight binding for the historical hybrid 41.66% implementation
manifests/               25-cell manifest, role-level gaps, S3DIS 542 closure, cloud/HF manifests
docs/weights_backup_v1/  original notes of the 2026-11-01 weight backup (mixed ZH/EN)
WEIGHT_INDEX.csv         weight index (SELECTED / FULL_ONLY tiers)
SHA256SUMS               SHA256 of every file in this package (except itself)
```

## Find configs by backbone × dataset

| Backbone \ Dataset | KITTI | PandaSet | nuScenes | S3DIS | NPM3D |
|---|---|---|---|---|---|
| Text2Loc | `configs/T2L_KITTI/` | `configs/T2L_PANDA/` | `configs/T2L_NUSC/` | `configs/T2L_S3DIS/` | `configs/T2L_NPM3D/` |
| Text2Loc++ | `configs/T2LPP_KITTI/` | `configs/T2LPP_PANDA/` | `configs/T2LPP_NUSC/` | `configs/T2LPP_S3DIS/` | `configs/T2LPP_NPM3D/` |
| MNCL | `configs/MNCL_KITTI/` | `configs/MNCL_PANDA/` | `configs/MNCL_NUSC/` | `configs/MNCL_S3DIS/` | `configs/MNCL_NPM3D/` |
| MambaPlace | `configs/MP_KITTI/` | `configs/MP_PANDA/` | `configs/MP_NUSC/` | `configs/MP_S3DIS/` | `configs/MP_NPM3D/` |
| Des4Pos (corrected) | `configs/D4P_KITTI/` | `configs/D4P_PANDA/` | `configs/D4P_NUSC/` | `configs/D4P_S3DIS/` | `configs/D4P_NPM3D/` |

Each cell directory contains three files:

- `01__ENVIRONMENT.json` — captured Python/torch/transformers versions and capture status;
- `02__SELECTION_RECORD.json` — the epoch-selection record for the SCAR head;
- `V3_CELL_SUMMARY.json` — coarse/fine/head roles, SHA, byte size and the A/B relation.

Normalization: `normalization/<cell>/NORMALIZATION.json`. If that file is absent, the
mean/variance is **embedded in the head file** (`role=scar_head` in `WEIGHT_INDEX.csv`,
`normalization_mode=EMBEDDED_IN_HEAD` in `MANIFEST_25CELL.csv`).

## Weight index and downloads

`WEIGHT_INDEX.csv` has two tiers:

- `tier=SELECTED`: the minimal set needed to reproduce the paper — 74 files,
  489,060,515 B (~466 MiB). It contains 25 SCAR heads, 25 configs,
  20 normalization files, plus 2 coarse + 2 fine backbone weights for
  MambaPlace×KITTI and MambaPlace×PandaSet.
- `tier=FULL_ONLY`: per-epoch heads, resume states and full backbone weights that exist
  only in the full backup.

The `SELECTED` tier is uploaded to a **private** Hugging Face model repository:

| Field | Value |
|---|---|
| Repository | `Promexile/scar-weights` (private; access must be granted) |
| Revision | `e7e3e1cfbc391b8a91513680ed8a50bf8bed051e` |
| Remote files | 74 (plus the repository's auto-generated `.gitattributes`) |
| Remote layout | `selected/<cell_id>/<role>/<basename>` |
| Verification | 74/74 matched; 29 LFS files checked against `lfs.oid` (SHA256), 45 small files downloaded and re-hashed |
| Per-file detail | `manifests/HF_SELECTED_WEIGHTS.csv`, `manifests/HF_REMOTE_VERIFICATION.json` |

Download (pin the full revision to reproduce exactly):

```bash
pip install huggingface_hub
hf auth login                     # use an account with access to the private repo
hf download Promexile/scar-weights --repo-type model --revision e7e3e1cfbc391b8a91513680ed8a50bf8bed051e --local-dir scar_weights
```

Each `SELECTED` row in `WEIGHT_INDEX.csv` carries:

- `hf_repo` / `hf_path` / `hf_revision` / `hf_verified` — the download location and its
  verification state;
- `path` — the **local target path** for restore (`<SCAR_ROOT>` and similar placeholders);
  place the downloaded file according to `RESTORE.md`.

`tier=FULL_ONLY` is not on Hugging Face; its `path` column is a placeholder that comes
from the cloud full backup.

Full-backup size (`manifests/CLOUD_BACKUP_MEASUREMENT.json`): 57,091,762,928 B
(53.17 GiB) raw, minus 51,387,683 B duplicated in the five-machine archive and
minus 4,449,196,325 B of FB-01 `.git` internals, leaving 52,591,178,920 B
(48.98 GiB) to upload. FB-04 contains `C_DELIVERY_20261102.tar`
(2,158,428,160 B), which duplicates the sibling directory but is kept as a self-contained
deliverable.

## Verification

```bash
sha256sum -c SHA256SUMS          # verify this code package
sha256sum <weight-file>          # compare with the sha256 column of WEIGHT_INDEX.csv
```

Two kinds of hashes appear in the material:

- `sha256` (this package, `WEIGHT_INDEX.csv`, cloud manifests) is the **file-byte SHA256**.
- The legacy `FREEZE_LIST.sha256` stores a **canonicalized content hash**; verifying it
  with `sha256sum -c` fails by design. Both are listed side by side in
  `manifests/ASSET_PATH_RECONCILIATION.csv` (`sha256_canonical` / `sha256_file_bytes`).

## Result conventions

`results/RESULTS_SOURCE.csv` is the single source of truth for the 25 cells: each row
gives the cell, metric, value, result file and entry identifier. `TOP1_5M.csv` and
`CELL_R1.csv` are pivots of it; `SCAR_MULTITHRESHOLD_LONG.csv` is the multi-threshold
long table.

**Two implementation lines must not be mixed:**

| Implementation line | Scope | Notes |
|---|---|---|
| Historical reproduction `HISTORICAL_V3` | 20 cells (T2L/T2LPP/MNCL/MP) | weights from the 2026-11-01 backup |
| Corrected reimplementation `CORRECTED_REIMPL` | 5 cells (Des4Pos) | reimplemented after the coordinate-frame fix; weights bound separately |

The "Des4Pos historical hybrid 41.66%" is a third line, bound only to the weights and
caches under `provenance/`. It **must not** be swapped with the corrected reimplementation
results, nor attributed to the corrected reimplementation. See
`provenance/HISTORICAL_HYBRID_4166_REPORT_ZH.md` (RG-08).

## 25-cell result overview

| cell_id | Backbone | Dataset | SCAR Top-1@5m | SCAR Cell R@1 | Head epoch | Implementation line |
|---|---|---|---|---|---|---|
| `T2L_KITTI` | Text2Loc | KITTI | 59.45 | 43.74 | 17 | Historical (V3) |
| `T2L_PANDA` | Text2Loc | PandaSet | 44.95 | 52.35 | - | Historical (V3) |
| `T2L_NUSC` | Text2Loc | nuScenes | 54.68 | 62.19 | 23 | Historical (V3) |
| `T2L_S3DIS` | Text2Loc | S3DIS | 41.33 | 41.70 | - | Historical (V3) |
| `T2L_NPM3D` | Text2Loc | NPM3D | 26.72 | 31.03 | - | Historical (V3) |
| `T2LPP_KITTI` | Text2Loc++ | KITTI | 64.15 | 45.02 | 7 | Historical (V3) |
| `T2LPP_PANDA` | Text2Loc++ | PandaSet | 44.30 | 49.10 | - | Historical (V3) |
| `T2LPP_NUSC` | Text2Loc++ | nuScenes | 53.36 | 60.56 | 20 | Historical (V3) |
| `T2LPP_S3DIS` | Text2Loc++ | S3DIS | 45.94 | 48.15 | - | Historical (V3) |
| `T2LPP_NPM3D` | Text2Loc++ | NPM3D | 16.38 | 19.83 | - | Historical (V3) |
| `MNCL_KITTI` | MNCL | KITTI | 63.82 | 47.41 | 25 | Historical (V3) |
| `MNCL_PANDA` | MNCL | PandaSet | 42.05 | 47.85 | - | Historical (V3) |
| `MNCL_NUSC` | MNCL | nuScenes | 52.47 | 65.28 | - | Historical (V3) |
| `MNCL_S3DIS` | MNCL | S3DIS | 38.38 | 39.30 | - | Historical (V3) |
| `MNCL_NPM3D` | MNCL | NPM3D | 14.66 | 19.83 | - | Historical (V3) |
| `MP_KITTI` | MambaPlace | KITTI | 57.07 | 40.83 | 5 | Historical (V3) |
| `MP_PANDA` | MambaPlace | PandaSet | 22.15 | 29.50 | 2 | Historical (V3) |
| `MP_NUSC` | MambaPlace | nuScenes | 47.71 | 60.11 | - | Historical (V3) |
| `MP_S3DIS` | MambaPlace | S3DIS | 40.77 | 42.25 | - | Historical (V3) |
| `MP_NPM3D` | MambaPlace | NPM3D | 19.83 | 24.14 | - | Historical (V3) |
| `D4P_KITTI` | Des4Pos (corrected reimpl) | KITTI | 57.84 | 43.11 | 26 | Corrected reimpl |
| `D4P_PANDA` | Des4Pos (corrected reimpl) | PandaSet | 31.55 | 41.55 | 4 | Corrected reimpl |
| `D4P_NUSC` | Des4Pos (corrected reimpl) | nuScenes | 41.44 | 60.11 | 16 | Corrected reimpl |
| `D4P_S3DIS` | Des4Pos (corrected reimpl) | S3DIS | 37.82 | 39.11 | 7 | Corrected reimpl |
| `D4P_NPM3D` | Des4Pos (corrected reimpl) | NPM3D | 23.28 | 27.59 | 24 | Corrected reimpl |

The numbers are SCAR's own ranking metrics, not deltas against a baseline.

## Known gaps (not filled, not substituted)

| ID | Scope | Status |
|---|---|---|
| RG-01 | MP×KITTI historical rule-B head | never saved; the paper cell uses the new formal trajectory s0 ep5 |
| RG-02 | MP×PandaSet historical rule-A/B heads | never saved; the paper cell uses the new formal trajectory s0 ep2 |
| RG-03 | B heads of T2L×KITTI, T2L×S3DIS, T2LPP×KITTI, MNCL×S3DIS | original B never saved; LSEL-retrained copies are used; must not be called "recovered historical B" |
| RG-04 | MNCL×PandaSet/nuScenes/NPM3D | A=B; no independent B evidence |
| RG-05 | T5-large / CLIP-ViT / MambaPlace environment | external public assets; source and version recorded, not packaged |
| RG-06 | corrected-A attempt1/2/3 interrupted weight trees | failed attempts, left on the remote, not migrated |
| RG-07 | D4P corrected C-round 367 files | local only, not in the P3 archive; referenced from local paths in this release |
| RG-08 | historical hybrid 41.66% | launch arguments cannot be recovered verbatim; registered by weight binding only |
| RG-09 | S3DIS TEST 542 vs nominal 544 | **closed**: 67×8 + d092×6 = 542, see `manifests/S3DIS_542_CLOSURE.md` |
| RG-10 | archive index missing chroot prefix | all 2,133 rows re-parsed, see `manifests/ASSET_PATH_RECONCILIATION.csv` |

## Where to start

1. This file and `RESTORE.md` (Chinese; the restore steps are command-oriented);
2. Open `configs/<cell>/` for the cell of interest;
3. Fetch weights via `WEIGHT_INDEX.csv`;
4. Take results from `results/RESULTS_SOURCE.csv`;
5. For gaps and claim boundaries see `manifests/ROLE_LEVEL_GAPS.csv` and
   `docs/weights_backup_v1/PROVENANCE_AND_LIMITS_ZH.md`.

## Path sanitization

Absolute paths in this package were replaced with placeholders (`<SCAR_ROOT>` = the
refine-logs root, `<ARCHIVE_ROOT>` = the five-machine archive root, `<DRIVE_X>/` = other
drives, `<USER_HOME>` = the user profile, `<REMOTE_HOST>:<PORT>` = a remote endpoint,
`<CONTAINER_ID>` / `GPU-<GPU_UUID>` = cloud machine and GPU identifiers). Every rewritten
file records its before/after SHA256 in `SANITIZATION_MAP.csv`. Therefore the frozen SHAs
referenced in `provenance/` (for example `scar_common`) correspond to the **unsanitized**
originals; when checking code identity use `sha256_before` from `SANITIZATION_MAP.csv`.
