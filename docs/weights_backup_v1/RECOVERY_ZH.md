# 恢复说明（SCAR Weights Backup V1）

## 1. 克隆与拉取

```bash
git lfs install
git clone <private-repo-url> SCAR-Weights-Backup
cd SCAR-Weights-Backup
git lfs pull
```

克隆后先确认 81 个权重都是实体文件，而不是 LFS pointer：

```bash
git lfs ls-files | wc -l          # 应为 81
git lfs ls-files | grep -v '^[0-9a-f]* \*' || true
```

## 2. 校验

```bash
sha256sum -c manifests/SHA256SUMS_WEIGHTS_V3.txt
```

Windows PowerShell 可用 `Get-FileHash -Algorithm SHA256` 逐项对照 `manifests/PHYSICAL_WEIGHTS_V3.csv`。

校验通过仅表示字节与 V3 记录一致，不表示选轮正确。

## 3. 角色映射

- 以 `manifests/ROLE_MANIFEST_V3.csv` 为准：95 条角色到 81 个实体。
- `headAB__*` 同时承担 head_A 与 head_B；这是台账记录的 A=B 同权重复用，不是独立 B。
- MP_NPM3D 的粗/细权重已从原 tar 中解出并单独校验；恢复时不需要原 tar。

## 4. 缺失的正式 B

`manifests/MISSING_B_V3.csv` 列出 5 格：

`MNCL_PANDA`, `MNCL_NUSC`, `MNCL_NPM3D`, `MP_KITTI`, `MP_PANDA`

这些 B 没有任何本地权重。**禁止用 head_A 顶替**，也不得把 A/B 写成独立对照。

## 5. 配置与归一化

- `configs/<cell>/V3_CELL_SUMMARY.json` 是精简摘要（权重、角色、epoch、门禁、架构状态）。
- `configs/<cell>/` 下是脱敏后的源配置副本；`CONFIG_INDEX_V3.csv` 给出源/归档双 SHA256。
- `normalization/` 是独立归一化文件；若摘要标为 `EMBEDDED_IN_HEAD`，归一化在头文件内部，没有单独文件。
- MP_NPM3D 架构配置以 `R059_NPM3D_PROTOCOL.json` 为准（V3 修正登记）。
- MP_S3DIS 明确标记 `ARCH_CONFIG_MISSING`；DATA_STATS/norm_stats 只是归一化，不是架构配置。

## 6. 不含内容

不含 vendor 模型源码、训练/评测代码、数据集、特征缓存、TEST 资产、日志全文、Token/SSH 密钥。要复跑需另行恢复代码与数据。
