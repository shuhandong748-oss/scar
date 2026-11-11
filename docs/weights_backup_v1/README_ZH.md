# SCAR Weights Backup V1 (SCAR_WEIGHTS_BACKUP_20261101)

Private backup of SCAR efficiency weights. Generated from `ASSET_AUDIT_V3_20261101` on 2026-11-01.

## Scope

- 25 cells = 5 methods x 5 datasets (CMMLoc excluded).
- 95 role mappings: coarse, fine, head_A, head_B per cell.
- 81 physical weight entities, deduplicated by content; A=B roles share one file.
- All weights were copied from local originals; originals were not modified or deleted.
- No model source code, dataset, feature cache, TEST asset, token, SSH key, or local absolute path is included.

## Counts

- Physical weight files: 81
- Total weight bytes: 4448794510 (4.143 GiB)
- Formal B: 20 (14 shared A=B, 6 independent B)
- Missing formal B: 5 (`MNCL_NPM3D`, `MNCL_NUSC`, `MNCL_PANDA`, `MP_KITTI`, `MP_PANDA`)
- Architecture config: full for Des4Pos; corrected for MP_NPM3D; MISSING for MP_S3DIS; partial/absent for other MambaPlace and Text2Loc-family cells.

## Important limits

- A matching SHA256 proves byte identity, **not** that the selection epoch is correct.
- The 14 shared A=B cells have no independent B selection evidence; do not describe them as independent A/B comparisons.
- The 5 missing B weights must not be replaced by A.
- MP_S3DIS has no architecture config; MP_NPM3D's architecture registration is corrected in V3; V2 remains untouched.
- The three MNCL missing-B cells have a COMPLETENESS_25 b_status of 本地已核验, but no B epoch and no independent file; V2/V3 classify them MISSING_UNPROVEN.

See `RECOVERY_ZH.md`, `PROVENANCE_AND_LIMITS_ZH.md`, and `manifests/`.
