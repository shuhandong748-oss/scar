# 恢复与复现步骤

## 0. 前置

- Python 3.10 环境，torch / torch_geometric / numpy / scipy 版本见 `ENVIRONMENT.md`；
- 上游源码（`adapt` / `models` / `dataloading` / `datapreparation` / `evaluation` /
  `training` / `common_kitti` / `train_extract`）**不在本包**，按 `THIRD_PARTY.md` 获取；
- 数据集（KITTI360Pose / PandaSetPose / nuScenesFull / S3DISPose / ParisLillePose）不在本包，
  需按各 cell 的 `dataset_version` 自行准备。

## 1. 放置代码

```bash
git clone <本仓库> scar-release
export PYTHONPATH="$PWD/scar-release/scar/core:$PYTHONPATH"   # 按需追加 vendor 目录
```

## 2. 下载并放置权重

`tier=SELECTED` 的 74 个文件在 Hugging Face 私有模型仓库
`Promexile/scar-weights`，提交版本 `e7e3e1cfbc391b8a91513680ed8a50bf8bed051e`：

```bash
pip install huggingface_hub
hf auth login
hf download Promexile/scar-weights --repo-type model --revision e7e3e1cfbc391b8a91513680ed8a50bf8bed051e --local-dir scar_weights
```

远程布局是 `selected/<cell_id>/<role>/<basename>`。`WEIGHT_INDEX.csv` 的每一行给出：

| 列 | 含义 |
|---|---|
| `hf_path` | 远程相对路径，例如 `selected/T2L_KITTI/scar_head/headB__newB_ht23_t2l_s0.pt` |
| `path` | 恢复时的本地目标路径（`<SCAR_ROOT>` 等占位符） |
| `sha256` | 文件字节 SHA256，放置后用 `sha256sum` 复核 |
| `hf_verified` | 上传后回读核验结果（全部为 `MATCH`） |

历史线头/配置/归一化的本地目标在
`<SCAR_ROOT>/SCAR_WEIGHTS_BACKUP_20261101/repo/{weights,configs,normalization}/<cell_id>/`
下；修正版 Des4Pos 的 coarse/fine 按其 `path` 列自带的目录放置。逐轮权重与主干全量在
`tier=FULL_ONLY`，来自云端全量备份，不在 HF 的 SELECTED 集合里。

## 3. 校验

```bash
sha256sum -c SHA256SUMS                       # 代码包完整性
sha256sum <file>                              # 与 WEIGHT_INDEX.csv 的 sha256 列比对
```

`FREEZE_LIST.sha256` 用的是规范化内容哈希，**不要**用 `sha256sum -c` 校验它。

## 4. 复现一条 cell

以修正版 Des4Pos 为例（历史线结构相同，脚本名不同）：

1. `export`：`scar/core/d4p_export.py` → 生成候选与 coarse 特征缓存；
2. `features`：`scar/core/d4p_features.py` → 生成 23 维 SCAR 特征；
3. `train`：`scar/core/d4p_train_head.py` → 训练 SCAR 头，按
   「VAL Top-1@5m 最大 → VAL Cell R@1 次之 → 更早 epoch」选轮；
4. `eval`：`scar/core/d4p_eval.py` → 六种排序（Original / Presence / Rank Fusion /
   FSR / 历史 A / SCAR）同缓存评测；
5. `verify`：`scar/core/d4p_verify_eval.py`、`d4p_verify_selection.py` → 选轮与产物校验。

各脚本的实参由对应 `configs/<cell>/01__config.json` 决定；先跑 `--help` 确认当前版本参数。

## 5. 对齐结果

用 `results/RESULTS_SOURCE.csv` 找到该 cell 的行，核对：

- `scar_top1_5m` / `scar_cell_r1` 是否与 `TOP1_5M.csv` / `CELL_R1.csv` 一致；
- 选轮 epoch 是否与 `configs/<cell>/02__SELECTION_RECORD.json` 一致；
- 头文件 SHA 是否与 `WEIGHT_INDEX.csv` 一致。

## 6. 不要重跑 TEST

修正版四域 TEST（PandaSet / nuScenesFull / S3DIS / NPM3D）、test13、MP 追加 TEST
均已按**单次读取**政策评价过，结果冻结在 `results/`。重跑会产生新的读取记录，
破坏可审计性。

## 7. 不可恢复项

RG-01、RG-02、RG-03、RG-04 对应的 B 头从未落盘或只有复训副本；不要用 A 头或其他
epoch 顶替。详见 `manifests/ROLE_LEVEL_GAPS.csv`。
