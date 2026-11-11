# 第三方代码与依赖

本包**不收录**任何许可未核实的第三方源码。只收录本项目自己写的 SCAR 相关代码、
配置、归一化与结果清单。

## 未收录、需自行获取的上游模块

以下模块被 `scar/` 代码 import，但不在本包内：

| 模块 | 来源 | 版本 / commit | 许可 |
|---|---|---|---|
| `adapt`, `models`, `dataloading`, `datapreparation`, `evaluation`, `training` | Des4Pos 上游实现（Text2Loc 派生） | 未记录 | **未核实** |
| `common_kitti`, `train_extract` | Text2Loc / Des4Pos 工具模块 | 未记录 | **未核实** |
| MambaPlace 训练/推理栈 | MambaPlace 上游 | 未记录 | **未核实** |
| MNCL 训练/推理栈 | MNCL 上游 | 未记录 | **未核实** |

未核实的原因：这些仓库的 LICENSE 文件没有随权重备份一起归档，归档中也没有
`LICENSE`/`COPYING`/`NOTICE` 文件（已在 `<ARCHIVE_ROOT>\methods` 下复核）。
在补齐许可证据之前，不上传到本仓库；需要复现时请从上游公开仓库取得同一版本。

## 本包内的 `vendor/` 目录

`scar/formal_mp_kitti/vendor/`、`scar/test13_kitti/scar_frozen/`、
`scar/test_append_mp_kitti/vendor/` 是本项目自己的冻结代码快照
（`common.py` / `features_fix.py` / `sa_geom.py`），用于按 SHA 固定评测时的行为，
不是第三方源码。

## Python 依赖版本

见 `ENVIRONMENT.md` 与各 cell 的 `configs/<cell>/01__ENVIRONMENT.json`。
运行时依赖：`torch`、`torch_geometric`、`numpy`、`scipy`、`transformers`。

## 公共模型资产

T5-large、CLIP-ViT、MambaPlace / MNCL 的预训练权重属于上游公共资产，
本包与云端清单都不打包（RG-05）；下载来源与版本以 `ENVIRONMENT.md` 为准。
