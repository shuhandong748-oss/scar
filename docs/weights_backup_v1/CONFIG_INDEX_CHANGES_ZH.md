# V3 配置索引变更

V2 目录 `ASSET_AUDIT_V2_20261101` 保持原样，未修改、未覆盖。

## MP_NPM3D

- 实际情况：V2 的 config 行只登记 `C_ACCEPTANCE.json`（验收记录），没有登记架构协议；`weights.normalization` 为 null。
- V3 修复：登记 `R059_NPM3D_PROTOCOL.json` 为 `ARCHITECTURE / YES`（含 `coarse_embed_dim=256`、`fine_embed_dim=128`、`fixed_embedding`），并把 `R059_CORE_PROTOCOL.json` 作为 provenance、`FREEZE.json` 作为 selection evidence。
- 旧登记未删除；V3 的 `CONFIG_INDEX_V3.csv` 同时保留 `C_ACCEPTANCE` 行并标注其非架构属性。

## MP_S3DIS

- 实际情况：V2 只登记 `DATA_STATS.json`（归一化）和 `num_stats_s3dis.json`（候选独立归一化），没有架构超参配置。
- V3 修复：`CELL_CONFIG_STATUS_V3.csv` 明确标记 `ARCH_CONFIG_MISSING`；`R059_S3DIS_PROTOCOL.json` 与 `R059_PROTOCOL.json` 只作为 provenance/hash 证据登记，`architecture_usable=NO`。

## 其他方法

- Des4Pos 五格：`coarse_config.json`/`fine_config.json`/`config.json` 为完整架构配置。
- MambaPlace（除 NPM3D/S3DIS）：FREEZE 记录提供头契约/主干提交，标为 PARTIAL。
- Text2Loc/Text2Loc++/MNCL：本地只有 SELECTION_RECORD 与 ENVIRONMENT，无主干架构配置，标为 ABSENT_LOCAL；本备份不含 vendor 源码。
