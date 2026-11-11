# 远端核验回执（SCAR-Weights-Backup）

核验时间：2026-11-01T23:23:29+08:00

## 1. 测试方式

- 不是只看 `git push` 退出码：先通过 GitHub REST API 读取远端仓库、提交和树，再逐文件读取 81 个 LFS pointer，最后用 Git LFS batch API 查询 81 个对象的下载动作，并实际下载最小对象复核 SHA256。
- 本回执所核验的内容提交：`c31e61ee10d460d473d7db55dff9040daae59b7a`。
- 该提交之后的元数据回执提交会再次推送；最终 HEAD 以外部最终报告为准。

## 2. 仓库与提交

- 仓库：`<GitHub-owner>/SCAR-Weights-Backup`（Private）。
- `default_branch=main`；`private=true`。
- 远端 `main` HEAD：`c31e61ee10d460d473d7db55dff9040daae59b7a`，与本地 HEAD 一致。
- 远端树未截断；共 269 个 tree 条目，其中 `weights/` 下实体文件 81 个。

## 3. LFS 对象

- 逐文件读取远端 pointer：81/81 的 `oid sha256` 与 `size` 与本地清单一致。
- 81 个 pointer 的 size 合计：4448794510 B（4.143 GiB）。
- LFS batch 下载查询：81/81 返回可用下载动作，缺失 0。
- 抽样实际下载：`weights/MNCL_KITTI/headA__epoch01.pt`，17352 B，SHA256 `946e8536d2b979e458432643863163fa9eac21fe6a9fbdfa9fbb562f75409699`，与预期一致。
- 抽样通过只证明该对象内容；全量结论由 81 个 pointer 与 batch 动作共同支持，未全量下载 4.14 GiB。

## 4. 不包含内容

- 远端树无 `.zip`、无 `.git/`、无 `test/` 路径；未上传本地 ZIP、数据集、特征缓存、源码、Token 或 SSH 私钥。
- 仓库 API 的 `size_kb` 不统计 LFS；LFS 实际占用以 4,448,794,510 B 为准。

## 5. 额度

- GitHub 官方：Free/Pro 含 10 GiB LFS 存储；本次 4.143 GiB。
- 未购买 data pack、未产生付费操作；API 无法读取精确剩余额度，账户下原有 2 个私有仓库均未发现 LFS 文件。

## 6. 结论

PASS。远端提交、81 个 pointer、81 个 LFS 对象及抽样内容均已独立核验。
