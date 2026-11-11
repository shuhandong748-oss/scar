# 运行环境

环境信息取自 `configs/<cell>/01__ENVIRONMENT.json`，是**备份/恢复机**上只读采集的结果；它不是对原始训练机的推断。标记 CAPTURED 的条目有实测值，UNCAPTURED 的保持原样。

## 已采集环境

| python env | python | torch | numpy | transformers | 采集类型 | 覆盖 cell 数 | 示例 cell |
|---|---|---|---|---|---|---|---|
| conda des4pos | 3.10.22 | 2.5.1+cu124 | 2.2.6 | 4.46.3 | recovery_environment | 7 | T2LPP_KITTI |
| conda mncl | 3.10.22 | 2.1.2+cu118 | 1.26.4 | 4.40.0 | recovery_environment | 5 | MNCL_KITTI |
| conda base (<REMOTE_PATH> | 3.12.3 | 2.5.1+cu124 | 2.1.3 | 4.44.2 | training_environment | 2 | T2L_NPM3D |

## 未采集

- 部分 cell 的原始训练机已下线，`01__ENVIRONMENT.json` 中 `training_environment.status = UNCAPTURED`，未做任何猜测填充。
- GPU 型号与驱动版本不在采集范围内；`torch_build_cuda` 只是 wheel 构建标签。

## 外部依赖（不在本包内）

| 依赖 | 用途 | 状态 |
|---|---|---|
| `torch`, `torch_geometric`, `numpy`, `scipy` | 主干与前向/几何打分 | 按上表版本 |
| `transformers` + T5-large | Text2Loc 系文本编码 | 公共模型，需联网或本地缓存（RG-05） |
| CLIP-ViT | 图像编码（历史线） | 公共模型，需联网或本地缓存（RG-05） |
| MambaPlace conda 环境 | MambaPlace 主干 | 可重建，未归档（RG-05） |
| Des4Pos / Text2Loc / MNCL / MambaPlace 上游源码 | `adapt`, `models`, `dataloading`, `datapreparation`, `evaluation`, `training`, `common_kitti`, `train_extract` | 许可未核实，**不收录**；需自行获取（见 THIRD_PARTY.md） |
