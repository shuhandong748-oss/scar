# 推送条件与目标（本地构建时点）

## 目标

- 仓库：`<GitHub-owner>/SCAR-Weights-Backup`
- 可见性：Private
- 2026-11-01 通过 GitHub API 查询为 404（不存在）；未创建、未推送。
- 同名历史仓库不存在；账户下已有私有仓库 `SCAR`（0 KB）与 `Quan`（39 KB）均未改动。

## 账户与额度检查（2026-11-01）

- GitHub 官方计费文档：Free / Pro / Free-for-org 均含 **10 GiB LFS 存储 + 10 GiB 带宽/月**；Team / Enterprise Cloud 为 250 GiB。
- 本机 LFS 实体：81 个对象，4,448,794,510 B = 4.143 GiB（按 1 GiB=2^30 B）。
- 账户下仅 2 个私有仓库；逐仓库检查默认分支树，均无 `.gitattributes` LFS 规则、无 130–134 B 的 LFS pointer blob，未发现既有 LFS 对象。
- 剩余额度无法通过 API 直接读取：当前凭据 scope 为 `gist, read:org, repo`，无 `user`；`/user` 的 `plan` 为 null；billing 端点返回 404。
- 结论：按官方 10 GiB 含额与“未发现既有 LFS 用量”，4.143 GiB 在额度内并留出约 5.86 GiB 余量；不购买 data pack，不做付费操作。

## 推送前必须满足

- 凭据可用（HTTPS/GCM 已验证；SSH 公钥未授权，因此走 HTTPS）。
- 目标仓库仍不存在，避免覆盖历史。
- 推送失败、额度不足或出现付费要求时立即停止并报告。

## 不在推送范围

- 模型源码、数据集、特征缓存、TEST 资产、日志全文、Token/SSH 私钥、本地绝对路径。
- 4.45 GB 本地 ZIP：仅保留在本机 `local_zip/`，不提交、不上传。
