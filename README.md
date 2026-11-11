# SCAR 复现发布包

SCAR 是一个挂在视觉定位主干（VPR backbone）输出之上的**伴随重排器**：给定主干产出的
候选 cell 与文本查询，用 23 维规范特征做候选级重排。本仓库发布 5 个主干 × 5 个数据集的
25 个实验结果所对应的代码、配置、归一化与结果绑定清单。

> **English.** SCAR is a companion reranker that sits on top of a visual place-recognition
> backbone. Given the backbone's candidate cells and a text query, it reranks candidates
> with a 23-dimensional canonical feature vector. This repository contains the code,
> configs, normalization files and result-binding manifests for 25 cells
> (5 backbones × 5 datasets). It contains **no weights, datasets, feature caches or logs**;
> weights are referenced by SHA256 in `WEIGHT_INDEX.csv`.

本仓库**不含**：权重、数据集、特征缓存、原始日志、凭据、SSH 信息。权重只按 SHA256 引用。

## 目录结构

```
scar/core/               修正版 Des4Pos SCAR 主流程（export → features → train → eval → verify）
scar/round_a/            修正版 A 域（PandaSet / nuScenesFull）坐标修正链
scar/test_d4p/           修正版四域 TEST 导出/评测（单次读取）
scar/formal_mp_kitti/    MambaPlace × KITTI 正式 SCAR 轨迹
scar/test_append_mp_kitti/  MP × KITTI s0 ep5 追加 TEST
scar/test13_kitti/       KITTI test13 证据包
scar/lsel/               LSEL 固定头复训与 TEST 评测
scar/tables/             结果表构建
configs/<cell>/          01__ENVIRONMENT.json / 02__SELECTION_RECORD.json / V3_CELL_SUMMARY.json
normalization/<cell>/    NORMALIZATION.json（部分 cell 内嵌在头文件里）
results/                 RESULTS_SOURCE.csv / TOP1_5M.csv / CELL_R1.csv / 多阈值长表
provenance/              历史混合实现 41.66% 的权重绑定与说明
manifests/               25 格清单、角色级缺口、S3DIS 542 收口、云端清单副本
docs/weights_backup_v1/  2026-11-01 权重备份的原始说明（中英混排）
WEIGHT_INDEX.csv         权重索引（SELECTED / FULL_ONLY 两档）
SHA256SUMS               本包内全部文件（除自身）的 SHA256
```

## 按「主干 × 数据集」定位配置

| 主干 \ 数据集 | KITTI | PandaSet | nuScenes | S3DIS | NPM3D |
|---|---|---|---|---|---|
| Text2Loc | `configs/T2L_KITTI/` | `configs/T2L_PANDA/` | `configs/T2L_NUSC/` | `configs/T2L_S3DIS/` | `configs/T2L_NPM3D/` |
| Text2Loc++ | `configs/T2LPP_KITTI/` | `configs/T2LPP_PANDA/` | `configs/T2LPP_NUSC/` | `configs/T2LPP_S3DIS/` | `configs/T2LPP_NPM3D/` |
| MNCL | `configs/MNCL_KITTI/` | `configs/MNCL_PANDA/` | `configs/MNCL_NUSC/` | `configs/MNCL_S3DIS/` | `configs/MNCL_NPM3D/` |
| MambaPlace | `configs/MP_KITTI/` | `configs/MP_PANDA/` | `configs/MP_NUSC/` | `configs/MP_S3DIS/` | `configs/MP_NPM3D/` |
| Des4Pos（修正版） | `configs/D4P_KITTI/` | `configs/D4P_PANDA/` | `configs/D4P_NUSC/` | `configs/D4P_S3DIS/` | `configs/D4P_NPM3D/` |

每个目录下的三件套：

- `01__ENVIRONMENT.json` — 采集到的 Python/torch/transformers 版本与采集状态；
- `02__SELECTION_RECORD.json` — SCAR 头的选轮记录；
- `V3_CELL_SUMMARY.json` — 该 cell 的粗/细/头角色、SHA、字节数与 A/B 关系。

归一化：`normalization/<cell>/NORMALIZATION.json`。若该文件不存在，说明均值/方差
**内嵌在头文件里**（`WEIGHT_INDEX.csv` 中该格 `role=scar_head`，`MANIFEST_25CELL.csv`
中 `normalization_mode=EMBEDDED_IN_HEAD`）。

## 权重索引与下载

`WEIGHT_INDEX.csv` 分两档：

- `tier=SELECTED`：论文复现所需的最小集合，74 个文件、489,060,515 B（约 466 MiB）。
  25 个头、25 份配置、20 份归一化，外加 MambaPlace×KITTI 与 MambaPlace×PandaSet 的
  2 个 coarse + 2 个 fine 主干权重。
- `tier=FULL_ONLY`：只出现在全量备份里的逐轮权重、resume、主干全量。

`tier=SELECTED` 已上传到 Hugging Face **私有**模型仓库：

| 字段 | 值 |
|---|---|
| 仓库 | `Promexile/scar-weights`（private，需获授权访问） |
| 提交版本 | `e7e3e1cfbc391b8a91513680ed8a50bf8bed051e` |
| 远程文件 | 74 个（另含仓库自动生成的 `.gitattributes`） |
| 远程布局 | `selected/<cell_id>/<role>/<basename>` |
| 校验结果 | 74/74 通过；29 个 LFS 文件按 `lfs.oid`（SHA256）核对，45 个小文件下载后重算 SHA256 |
| 逐文件明细 | `manifests/HF_SELECTED_WEIGHTS.csv`、`manifests/HF_REMOTE_VERIFICATION.json` |

下载（把 `e7e3e1cfbc39` 替换为完整提交号可锁定版本）：

```bash
pip install huggingface_hub
hf auth login                     # 使用有该私有仓库权限的账号
hf download Promexile/scar-weights --repo-type model --revision e7e3e1cfbc391b8a91513680ed8a50bf8bed051e --local-dir scar_weights
```

`WEIGHT_INDEX.csv` 中 SELECTED 行同时给出：

- `hf_repo` / `hf_path` / `hf_revision` / `hf_verified` —— 可直接下载的远程位置与核验状态；
- `path` —— 恢复时的**本地目标路径**（`<SCAR_ROOT>` 等占位符），下载后按
  `RESTORE.md` 的对应关系放置。

`tier=FULL_ONLY` 不在 HF，`path` 列是占位路径，来自云端全量备份。

全量备份的体量（`manifests/CLOUD_BACKUP_MEASUREMENT.json`）：原始 57,091,762,928 B
（53.17 GiB），扣除与五机归档重复的
51,387,683 B 与 FB-01 的 `.git` 内部对象
4,449,196,325 B 后，净上传 52,591,178,920 B
（48.98 GiB）。FB-04 内含 `C_DELIVERY_20261102.tar`
（2,158,428,160 B），与同名目录重复但作为自包含交付件保留。

## 校验

```bash
sha256sum -c SHA256SUMS          # 校验本包代码
sha256sum <weight-file>          # 与 WEIGHT_INDEX.csv 的 sha256 列比对
```

注意两种哈希：

- `sha256`（本包、`WEIGHT_INDEX.csv`、云端清单）= **文件字节 SHA256**。
- 旧版 `FREEZE_LIST.sha256` 存的是**规范化内容哈希**，用 `sha256sum -c` 校验会按设计失败。
  两者在 `manifests/ASSET_PATH_RECONCILIATION.csv` 中分列（`sha256_canonical` / `sha256_file_bytes`）。

## 结果口径

`results/RESULTS_SOURCE.csv` 是 25 格结果的唯一绑定源：每行给出 cell、指标、数值、
结果文件与条目标识。`TOP1_5M.csv` / `CELL_R1.csv` 是它的透视表，
`SCAR_MULTITHRESHOLD_LONG.csv` 是多阈值长表。

**两条实现线必须分开看**：

| 实现线 | 范围 | 说明 |
|---|---|---|
| 历史复现线 `HISTORICAL_V3` | 20 格（T2L/T2LPP/MNCL/MP） | 权重来自 2026-11-01 备份 |
| 修正版重实现线 `CORRECTED_REIMPL` | 5 格（Des4Pos） | 修正坐标帧后的重实现，权重另行绑定 |

「Des4Pos 历史混合实现 41.66%」是另一条线，只对应 `provenance/` 里的权重与缓存，
**不能**和修正版重实现的结果互换，也不能归因给修正版。详见
`provenance/HISTORICAL_HYBRID_4166_REPORT_ZH.md`（RG-08）。

## 25 格结果总览

| cell_id | 主干 | 数据集 | SCAR Top-1@5m | SCAR Cell R@1 | 头 epoch | 实现线 |
|---|---|---|---|---|---|---|
| `T2L_KITTI` | Text2Loc | KITTI | 59.45 | 43.74 | 17 | 历史复现线 |
| `T2L_PANDA` | Text2Loc | PandaSet | 44.95 | 52.35 | - | 历史复现线 |
| `T2L_NUSC` | Text2Loc | nuScenes | 54.68 | 62.19 | 23 | 历史复现线 |
| `T2L_S3DIS` | Text2Loc | S3DIS | 41.33 | 41.70 | - | 历史复现线 |
| `T2L_NPM3D` | Text2Loc | NPM3D | 26.72 | 31.03 | - | 历史复现线 |
| `T2LPP_KITTI` | Text2Loc++ | KITTI | 64.15 | 45.02 | 7 | 历史复现线 |
| `T2LPP_PANDA` | Text2Loc++ | PandaSet | 44.30 | 49.10 | - | 历史复现线 |
| `T2LPP_NUSC` | Text2Loc++ | nuScenes | 53.36 | 60.56 | 20 | 历史复现线 |
| `T2LPP_S3DIS` | Text2Loc++ | S3DIS | 45.94 | 48.15 | - | 历史复现线 |
| `T2LPP_NPM3D` | Text2Loc++ | NPM3D | 16.38 | 19.83 | - | 历史复现线 |
| `MNCL_KITTI` | MNCL | KITTI | 63.82 | 47.41 | 25 | 历史复现线 |
| `MNCL_PANDA` | MNCL | PandaSet | 42.05 | 47.85 | - | 历史复现线 |
| `MNCL_NUSC` | MNCL | nuScenes | 52.47 | 65.28 | - | 历史复现线 |
| `MNCL_S3DIS` | MNCL | S3DIS | 38.38 | 39.30 | - | 历史复现线 |
| `MNCL_NPM3D` | MNCL | NPM3D | 14.66 | 19.83 | - | 历史复现线 |
| `MP_KITTI` | MambaPlace | KITTI | 57.07 | 40.83 | 5 | 历史复现线 |
| `MP_PANDA` | MambaPlace | PandaSet | 22.15 | 29.50 | 2 | 历史复现线 |
| `MP_NUSC` | MambaPlace | nuScenes | 47.71 | 60.11 | - | 历史复现线 |
| `MP_S3DIS` | MambaPlace | S3DIS | 40.77 | 42.25 | - | 历史复现线 |
| `MP_NPM3D` | MambaPlace | NPM3D | 19.83 | 24.14 | - | 历史复现线 |
| `D4P_KITTI` | Des4Pos (corrected reimpl) | KITTI | 57.84 | 43.11 | 26 | 修正版重实现线 |
| `D4P_PANDA` | Des4Pos (corrected reimpl) | PandaSet | 31.55 | 41.55 | 4 | 修正版重实现线 |
| `D4P_NUSC` | Des4Pos (corrected reimpl) | nuScenes | 41.44 | 60.11 | 16 | 修正版重实现线 |
| `D4P_S3DIS` | Des4Pos (corrected reimpl) | S3DIS | 37.82 | 39.11 | 7 | 修正版重实现线 |
| `D4P_NPM3D` | Des4Pos (corrected reimpl) | NPM3D | 23.28 | 27.59 | 24 | 修正版重实现线 |

数值是 SCAR 排序本身的指标，不是相对基线的差值。

## 已知缺口（不填补、不替代）

| 编号 | 范围 | 状态 |
|---|---|---|
| RG-01 | MP×KITTI 历史 B 规则头 | 从未落盘；论文格用新正式轨迹 s0 ep5 |
| RG-02 | MP×PandaSet 历史 A/B 规则头 | 从未落盘；论文格用新正式轨迹 s0 ep2 |
| RG-03 | T2L×KITTI、T2L×S3DIS、T2LPP×KITTI、MNCL×S3DIS 的 B 头 | 原 B 从未落盘，现用 LSEL 复训副本；不可称"恢复历史 B" |
| RG-04 | MNCL×PandaSet/nuScenes/NPM3D | A=B，无独立 B 证据 |
| RG-05 | T5-large / CLIP-ViT / MambaPlace 环境 | 外部公共资产，记录来源与版本，不打包 |
| RG-06 | A 修正版 attempt1/2/3 中断权重树 | 失败尝试，留在远端，不迁移 |
| RG-07 | D4P corrected C 轮 367 文件 | 仅本地，未入 P3 归档；本次直接引用本地路径 |
| RG-08 | 41.66% 历史混合实现 | 启动参数无法逐字恢复，只按权重绑定登记 |
| RG-09 | S3DIS TEST 542 vs 名义 544 | **已关闭**：67×8 + d092×6 = 542，见 `manifests/S3DIS_542_CLOSURE.md` |
| RG-10 | 归档索引缺 chroot 前缀 | 已对全部 2,133 行重新解析，见 `manifests/ASSET_PATH_RECONCILIATION.csv` |

## 从哪读起

1. 先看本文件与 `RESTORE.md`；
2. 按 cell 打开 `configs/<cell>/`；
3. 权重按 `WEIGHT_INDEX.csv` 取；
4. 结果以 `results/RESULTS_SOURCE.csv` 为准；
5. 缺口与边界看 `manifests/ROLE_LEVEL_GAPS.csv` 与 `docs/weights_backup_v1/PROVENANCE_AND_LIMITS_ZH.md`。

## 路径脱敏

本包内的绝对路径已替换为占位符（`<SCAR_ROOT>` = refine-logs 根，
`<ARCHIVE_ROOT>` = 五机归档根，`<DRIVE_X>/` = 其他盘符，`<USER_HOME>` = 用户目录，
`<REMOTE_HOST>:<PORT>` = 远程端点，`<CONTAINER_ID>` / `GPU-<GPU_UUID>` = 云机与显卡标识）。
每个被改写的文件都记录了
改写前后的 SHA256，见 `SANITIZATION_MAP.csv`。因此 `provenance/` 里引用的
`scar_common` 等冻结 SHA **对应的是未脱敏原件**；校验代码一致性时以
`SANITIZATION_MAP.csv` 的 `sha256_before` 为准。
