# S3DIS TEST 查询数 542 vs 544 —— 关闭记录

- 对应旧缺项：M-36（TEST_COUNT_DISCREPANCY）
- 状态：**CLOSED / 已解释，不阻断**
- 日期：2026-11-11

## 结论

**542 是实际冻结的 TEST 划分；544 只是“68 房间 × 8 查询”的名义算术值。**

- 68 个 TEST 房间中，67 个各发布 8 个 query pose，`d092` 只发布 6 个。
- 67×8 + 6 = **542**。
- 本轮没有丢弃、补造、下采样任何查询。

## 证据链

| 项 | 值 |
|---|---|
| 回执 | `D4P_CORRECTED_TEST_FOUR_DOMAINS_20261104/reports/S3DIS_DENOMINATOR_542_544.json` |
| 说明 | 同目录 `S3DIS_DENOMINATOR_542_544_ZH.md` |
| 数据 manifest | `baseline_recovery_20261102/C_corrected_domains/s3dis/val_inputs/dataset_manifest.json` |
| manifest SHA256 | `41e00444edf66f72f9e44508c2071e6dd087c0edb7c4a58c0df4b07c59267b15` |
| 独立确认 | R060 DELIVERY / MambaPlace R059 协议 / Text2Loc++ 独立加载器 / R060 TEST_READ_ONCE / R060 test.log — 共 5 处 |
| 本机 corrected TEST marker | `D4P_CORRECTED_TEST_FOUR_DOMAINS_20261104/s3dis/export/test/TEST_READ_ONCE.json`（main head `18f61564…` ep7） |

## 分母政策

- 分母 = 全部 **542** 查询；空/无效候选列表计为失败。
- **436** 是“有效候选列表数”，不是分母。
- 代码约束：冻结的 `C_round/code/scar_common.py` 中 `expected.test=544` 会直接中止；处理方式是新增 TEST 入口并在 marker 记录旧 guard 字节哈希、名义值 544、实际值 542 与本回执哈希，**不得修改冻结文件或历史 marker**。

## 影响范围

S3DIS 五格（D4P/T2L/T2LPP/MNCL/MP）全部使用 542 为分母；结果表中的 S3DIS 数值均按 542 计算，无需重算。

## 对 M-36 的处理

在 `MISSING_AND_CONFLICTS_ADDENDUM.csv` 中追加 M-36-CLOSED 一行，**不修改 M-36 原文**。
