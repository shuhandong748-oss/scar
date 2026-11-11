# 来源与边界

## 计数

- 角色映射 95 条，实体 81 个，唯一 SHA256 81 个。
- 去重前按角色累计 4449166098 B；去重后 4448794510 B；节省 371588 B。
- 最大权重文件 213672660 B。
- 正式 B 20 格：14 格 A=B 同文件，6 格独立 B。
- 缺失 B 5 格：`MNCL_NPM3D`, `MNCL_NUSC`, `MNCL_PANDA`, `MP_KITTI`, `MP_PANDA`。

## 选轮

- head_A/head_B 的 epoch 来自 `LOCAL_WEIGHT_AUDIT_20261030/COMPLETENESS_25.csv`。
- coarse/fine 没有独立选轮 epoch 记录；仅在全名中可识别 epoch 时记录 `epoch_hint_from_filename`，不声称其为选轮依据。
- A=B 的 14 格只有单头记录，没有独立 B 选轮证据。
- 5 个缺失 B 中，MP_KITTI/MP_PANDA 的记录 epoch 分别为 5/2，但权重从未写盘。

## 哈希解释边界

SHA256 一致只证明复制后的字节与 V2/历史记录一致；不证明选轮正确、不证明配置与代码版本匹配、不证明 A 可替代 B。

## 不含内容

未包含任何 TEST 资产、数据集、特征缓存、模型源码、日志全文、凭据或本地绝对路径。配置副本已做路径/主机/账号脱敏，源文件 SHA256 与脱敏后 SHA256 分列记录。
