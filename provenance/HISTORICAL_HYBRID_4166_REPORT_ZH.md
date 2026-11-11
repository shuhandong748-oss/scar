# Des4Pos 早期 41.66% 结果与权重只读溯源报告

- 日期：2026-11-04
- 范围：<ARCHIVE_ROOT>\inventory\、<DES4POS_GC_ROOT>\、<DRIVE_E_TIKTOK>\_mf\src\working
- 目标：确认本地是否存在产生 KITTI TEST Top-1@5m = 41.66%、Cell R@1 = 32.46% 的 Coarse/Fine 主干权重与完整运行代码（不是 SCAR reranker.pt）
- 模式：只读。未打开 TEST 特征/预测/原始数据载荷，未运行模型、未重算指标、未训练、未推理、未评价、未连接云端、未修改任何论文或结果表、未删除或覆盖文件。
- 方法：记录→manifest→实际命令/配置→代码→权重的反向绑定；以完整 SHA256 相等为唯一绑定依据，不以文件名、时间或指标相近作为匹配证据；本地压缩包仅查成员索引，未重复解包大型数据。
- 关键路径说明：<DRIVE_E_TIKTOK>\_mf\src\working 与 <DES4POS_GC_ROOT>\working 内容相同（仅 CRLF/LF 差异，`diff --strip-trailing-cr` 对已抽查的 6 个文件均为 0 行差异），本报告的代码 SHA 取 LF 版本（_mf/src/working）。

---

## 0. 结论摘要

1. **两份主干权重都在本地**，且只有一个实体，无重名副本、无压缩包成员副本。
2. **完整 SHA256 能绑定到 41.66% 那次运行**：结果文件、运行 manifest、本地权重三方记录的 SHA 完全一致，链上无断点。
3. **实现身份为“混合实现（hybrid）”，已确认**：骨架为官方 Des4Pos 的 ObjectEncoder/PointNet2/ConvBlock，粗检索文本编码器与精定位 CrossMatch 直接来自 Text2Loc 上游，主干为本机自训；不是官方 Des4Pos 复现，也不是“证据不足”。
4. **恢复该版本仍缺**：服务器端当次 T5-large 的确切字节、cache 构建的精确 argv、sa_geom.py 在当次目录中的实体、全树逐文件比对；官方 Des4Pos 的 STE/CRA/mamba 模块与作者权重从未发布，不属于该混合路径但使“官方复现”标签无法成立。

---

## 1. 反向绑定链（结果记录 → manifest → 命令/配置 → 代码 → 权重）

### 1.1 记录层

| 环节 | 文件 | 证据 |
|---|---|---|
| 汇总记录 | <DES4POS_GC_ROOT>\STATUS.md §8.30（2026-10-21，自第 2499 行起） | 记录 baseline 41.66 / cell R@1 32.46，n=11,505；同时记录 gate PASS 与 Top-25 表 |
| 原始控制台 | experiments\Des4Pos_learned_reranker\test_eval_console.txt | `Kitti360CellDatasetMulti: 3 scenes, 11505 descriptions for 3932 unique poses, 4308 cells`；`[result] baseline_des4pos R@1 0.3246 | T1@5m 0.4166 T1@10m 0.5398 T1@15m 0.5635 | T5@5m 0.6973 T10@5m 0.7892` |

这一层只说明“有这个数字”，不构成权重绑定。

### 1.2 结果层（test_results.json）

<DES4POS_GC_ROOT>\experiments\Des4Pos_learned_reranker\test_results.json（7.3 MB，仅读取 protocol/results/路径/SHA 字段，未读取预测数组）：

- protocol：
  - n_queries = 11505，cells = 4308，pool = 50，topk = 10，seed = 0
  - candidate_set = “baseline = coarse top-10；C 与 learned = P2 top-10”
  - fine = “single frozen cache over union(baseline top-10, P2 top-10)”
  - 评价函数 = evaluation.utils.calc_sample_accuracies
  - gt_used_only_for_labels_and_metric = true；test_used_for_training_or_tuning = false
- results.baseline_des4pos：
  - top1@5m = 0.41660147761842675（41.66%）
  - top5@5m = 0.6972620599739244；top10@5m = 0.7892220773576706
  - coarse_R@1 = 0.32455454150369406（32.46%）、coarse_R@5 = 0.6252933507170795、coarse_R@10 = 0.7330725771403738
- 内嵌权重路径（服务器绝对路径）：
  - coarse：/root/autodl-tmp/Des4Pos/working/checkpoints/k360_30-10_scG_pd10_pc4_spY_all/coarse_run1/coarse_contN_epoch9_acc0.710_ecl0_eco0_p256_npa1_loss-contrastive_f-class-color-position-num.pth
  - fine：同前缀 …/fine_run1/fine_contN_epoch25_offset0.095_lr0.0003_obj-6-16_ecl0_eco0_p256_npa1_f-class-color-position-num.pth
  - 两者与 pointnet 的 sha256、以及 test_coarse/test_cands/test_fine 三个 cache 的 sha256 均记录在案（见 §1.3）。

这一层确定“41.66% 是由哪些权重文件、哪套缓存、哪条评价函数算出来的”。

### 1.3 Manifest 层（RUN_MANIFEST.json）

<DES4POS_GC_ROOT>\experiments\THREE_TABLES_20261021\run_20261021_top25\RUN_MANIFEST.json（sha256 f46bc3a4…）：

- working_head = 2185d9ccad17d299ac3c0bd290585219c4825082（运行代码树 git HEAD）
- server = <REMOTE_HOST>:<PORT>
- split = TEST 场景 0003 / 0005 / 0009；n_queries = 11505；n_cells = 4308
- 权重与输入 SHA256：

| 资产 | 服务器路径 | SHA256 |
|---|---|---|
| Coarse | …/coarse_run1/coarse_contN_epoch9_acc0.710_…-num.pth | 3fa5661986687eeb5ad76fb19527751b3d75661917747dd1a1f69e2926c9386c |
| Fine | …/fine_run1/fine_contN_epoch25_offset0.095_…-num.pth | 41a74ff32d487c66e8fc8e8ade6100ee2a1149cf76ef01616c54df7b422d8e2e |
| PointNet | /root/autodl-tmp/Des4Pos/checkpoints/pointnet_acc0.86_lr1_p256.pth | ef63fdfc9cd10e698181efe415a873e628cf86698266f005351943030016e8e1 |
| test_coarse 缓存 | — | 73c6720d417b40617264c64768c3ba74f10c3290fce842f7c0567e396d47eeb6 |
| test_cands 缓存 | — | da71fc0b8ece2fdd158745d9fa8987bdbecc70c5414eb30822d1ce59fd717495 |
| test_fine 缓存 | — | 5d335ba736cc0d89e92cd1a669ddb93e483705fc478b27be8a310cbdc36dfd2f |
| test_results | — | f1eed9c2d90b630c0c63f9e3461aac395482f8e43fa4215b71cd76e12fe6b10d |

- 该 manifest 记录了 gate 复现出的 41.66 / 60.26 / 65.28 / 66.43 与 cell R@1 32.46 / 46.77 / 47.78 / 45.08；“41.66”行即 baseline_des4pos（未重排的原序）。

### 1.4 命令/配置层

- 训练命令（归档）：<ARCHIVE_ROOT>\_incoming\P1\root\autodl-tmp\Des4Pos\logs\coarse_run1.cmd
  - T5 路径 ../models/t5-large；PointNet ../checkpoints/pointnet_acc0.86_lr1_p256.pth；20 epochs；ranking loss = contrastive；fixed embedding（T5 冻结）。
- 训练日志：<DES4POS_GC_ROOT>\artifacts\logs\coarse_run1.log、fine_run1.log
  - coarse：batch_size 64，lr 0.0005，continue_path=None，ranking_loss contrastive，fixed_embedding True，epochs 20，folder coarse_run1；在 VAL acc 0.710 保存 epoch9。
  - fine：batch_size 32，lr 0.0003，continue_path=None，pmc_prob 0.5，fixed_embedding True，epochs 35；epoch25 保存 offset 0.095（epoch17 为 0.096）。
  - 两处 continue_path=None 说明两份权重是在该 working 树内从头训练得到，不是加载他人的预训练权重。
- 推理/评价：build_test_cache.py（stage_coarse 加载 args.path_coarse → CellRetrievalNetwork；stage_fine 加载 args.path_fine → CrossMatch(…)，strict=False，eval），test_build.log 记录 TEST 场景、11505 描述、4308 cells、154,252 条 Fine 偏移。
- **缺口**：test_build.log 未回显 argv，仓库内也未找到当次 build 的 args 落盘文件；精确命令行只能由脚本默认值 + manifest 路径反推，不能声称逐字复原。

### 1.5 代码层

- <DRIVE_E_TIKTOK>\_mf\src\working（git HEAD 2185d9ccad17d299ac3c0bd290585219c4825082）
  - git log：`2185d9c Restore missing modules + PyG>=2.0 / NumPy 2.x compatibility`；父提交 `e8288a8 baseline: verbatim copy of official @ 3cc53c2`
  - <DES4POS_GC_ROOT>\official 为官方 3cc53c2dfb9f64ee18c9f813229dbec5ecb38d3b，缺 models/language_encoder.py 与 models/cross_matcher.py，且 cell_retrieval.py 导入的 STE_2 在官方树内自相矛盾/缺权重。
- 评价逻辑（实验目录 eval_test.py）：
  - `base_order = [[str(db_ids[j]) for j in pool[i, :C.TOPK]] for i in range(n)]`（TOPK=10，pool=50，按 coarse cosine 顺序取前 10）
  - `orders["baseline_des4pos"] = base_order`，再调用 calc_sample_accuracies
  - common.py：POOL=50，LAMBDA=0.5，TOPK=10；eval_order(threshs=(5,10,15))；coarse_recall 单列报告
  - 因此 baseline_des4pos 是“原序（coarse cosine top-10）+ 冻结 Fine 偏移”，不做任何重排。

### 1.6 权重层（本地实体哈希复核）

| 资产 | 本地路径 | 字节 | 本地 SHA256 | 与 manifest/test_results |
|---|---|---|---|---|
| Coarse ep9 | artifacts\backups\checkpoints\coarse_run1\coarse_contN_epoch9_acc0.710_ecl0_eco0_p256_npa1_loss-contrastive_f-class-color-position-num.pth | 72,894,790 | 3fa5661986687eeb5ad76fb19527751b3d75661917747dd1a1f69e2926c9386c | 完全一致 |
| Fine ep25 | artifacts\backups\checkpoints\fine_run1\fine_contN_epoch25_offset0.095_lr0.0003_obj-6-16_ecl0_eco0_p256_npa1_f-class-color-position-num.pth | 61,645,368 | 41a74ff32d487c66e8fc8e8ade6100ee2a1149cf76ef01616c54df7b422d8e2e | 完全一致 |
| PointNet | checkpoints\pointnet_acc0.86_lr1_p256.pth | 5,858,980 | ef63fdfc9cd10e698181efe415a873e628cf86698266f005351943030016e8e1 | 完全一致 |

- 三份缓存本地实体：experiments\Des4Pos_learned_reranker\test_coarse.npz（17,757,058 B）、test_cands.npz（865,959 B）、test_fine.npz（1,438,485 B），哈希与 manifest 记录一致（本轮沿用已核验结果，未重新读取载荷内容）。
- 重名扫描：全盘 Des4Pos-GC 下 `coarse_contN_epoch9*` 与 `fine_contN_epoch25*` 各只有一个实体；未发现同名但不同 SHA 的混淆项（其他方法的同名近似文件 SHA 均不同，未采用）。
- 未发现这三份权重存在于任何本地压缩包的成员索引中，因此不存在“包内副本”需要引用或避免重复解包。

### 1.7 反向链完整性

结果记录（STATUS/console）→ test_results.json（指标 + 权重路径 + 权重 SHA + cache SHA + 评价函数）→ RUN_MANIFEST.json（同一组 SHA + working_head + 场景/查询数）→ 本地文件全 SHA 相等 → 代码树 git HEAD 相等 → 构建/评价脚本可从该树导出模型类与排序逻辑。**七环中有六环是严格相等绑定；命令 argv 一环为“脚本+manifest 可反推但未逐字保存”。**

---

## 2. 41.66% 的定义确认

| 项目 | 结论 | 依据 |
|---|---|---|
| 数据集划分 | KITTI-360 TEST，场景 0003/0005/0009 | test_results.protocol、console、RUN_MANIFEST |
| 查询数 | 全部 11,505 条描述（3,932 个唯一 pose） | protocol n_queries=11505；console |
| 单元数 | 4,308 cells | protocol / console |
| 阈值 | 5 m（同时报 10/15 m） | common.eval_order(threshs=(5,10,15))；数值列 T1@5m |
| 排序 | **原序**：coarse cosine 相似度降序取前 10（pool 前 10），无重排 | eval_test.py base_order；baseline_des4pos 即 base_order |
| Fine 坐标 | 单一冻结 cache，对 baseline top-10 ∪ P2 top-10 的并集算一次，baseline 行取其中 top-10 的偏移 | test_results.protocol |
| 指标函数 | evaluation.utils.calc_sample_accuracies | protocol |
| 是否用 TEST 调参 | false；GT 仅用于标签与度量 | protocol |

补充：另有 eval_full_test.log 记录直接 pipeline 的 Top-1@5m = 0.4148（41.48），与 frozen-cache 路径的 41.66 不同。**41.66 对应的是 reranker 实验路径（冻结缓存 + base_order），不是直接 pipeline 值**；两者差 0.18 个百分点，本报告不对差异原因下结论。

---

## 3. 实现身份核实

### 3.1 模型类与代码来源

| 主干 | 实际类 | 组成 | 来源 |
|---|---|---|---|
| Coarse | models/cell_retrieval.py → CellRetrievalNetwork | ObjectEncoder + PointNet2 + ConvBlock（官方 Des4Pos 组件）+ Text2Loc LanguageEncoder；mamba_model.py 为 59 行无参数占位、forward 抛 NotImplementedError（死代码）；obj_inter_module 存在于权重但不在前向主路径 | 官方 3cc53c2 的 object_encoder/pointcloud + Text2Loc 上游 |
| Fine | models/cross_matcher.py → CrossMatch | object_encoder + cross_hints / cross_objects 两个 TransformerDecoder + mlp_offsets；MSE 直接回归偏移 | Text2Loc 上游 |

Text2Loc 上游确认：experiments\SIX_BACKBONES_20261022\text2loc_orig_src\README.md 自述为 CVPR 2024 Text2Loc 官方实现；refine-logs\b_machine\clone_assets_20261103\posevi\third_party\upstream.lock.json 记录 Text2Loc commit 5d59723a9df42a8454505c8c0774ebe93e7ee4df。

### 3.2 “hybrid”的具体证据（不是重复旧标签）

1. **文件级字节相等**：
   - working/models/language_encoder.py SHA256 = 05205188717f9978a3ebff5f57b840ad7771de399c0ceca23eee94d158cf3c4d —— 与 text2loc_orig_src/models/language_encoder.py 完全相同。
   - working/models/cross_matcher.py SHA256 = e908d8ed9468721ecd4106adbf72fcc8c0bba66a5845e5c7000f969f8770b5a3 —— 与 text2loc_orig_src/models/cross_matcher.py 完全相同。
2. **官方树缺失**：官方 3cc53c2 内不存在上述两个文件；working 的父提交 e8288a8 是“官方逐字拷贝”，子提交 2185d9c 才“恢复缺失模块 + 兼容性修改”，即两个 Text2Loc 模块是**后来补进来的**，不是官方 Des4Pos 内容。
3. **权重键结构独立佐证**（zipfile 静态读取 data.pkl，未运行模型）：
   - Coarse：199 个键 / 18,200,271 参数；前缀计数 object_encoder 122、language_encoder 31、obj_inter_module 24、layers 22。
   - Fine：217 个键 / 15,385,105 参数；前缀计数 object_encoder 122、cross_hints 36、cross_objects 36、language_encoder 19、mlp_offsets 4。
   - 两份权重中 **STE / CRA / mfam / bilstm / prior 相关键为 0**。这与“官方 Des4Pos 的 STE/CRA 精定位”不符，而与 Text2Loc 的 CrossMatch 结构吻合。
4. **训练日志**：continue_path=None、fixed_embedding True、T5-large 路径、PointNet 初始化、contrastive 排序损失——说明主干是本机从零自训（T5 冻结），并非加载作者发布的 Des4Pos 权重（作者也从未发布）。
5. **官方模块不可用**：官方 cell_retrieval.py 导入 models.STE_2，而官方 STE_2 内部签名/依赖不一致且无对应权重，官方提交本身不可直接运行；working 的 cell_retrieval.py 已不依赖 STE 前向。

### 3.3 组件依赖

| 组件 | 是否使用 | 证据 |
|---|---|---|
| Text2Loc | **是**（LanguageEncoder + CrossMatch，源码级组件复用，不是加载 Text2Loc 预训练权重） | 文件字节相等 + 权重键前缀 + 训练日志 continue_path=None |
| MNCL | **否**（模型路径未导入 mncl 目录任何模块） | cell_retrieval/cross_matcher 的 import 链；只存在 mncl/ 旁支目录 |
| T5 | **是**，t5-large，冻结输出 | 训练命令 ../models/t5-large；fixed_embedding True；LanguageEncoder 内部 T5EncoderModel |
| PointNet | **是**，作为物体编码器初始化 | coarse_run1.cmd 指定 pointnet_acc0.86_lr1_p256.pth；SHA 绑定 |
| STE | **否** | 权重无 STE 键；模型文件无 STE 导入 |
| prior（先验） | **否** | 权重无 prior 键；该实现无数学先验分支 |
| sa_geom | **未证**（reranker 目录内不存在该文件；构建/评价是否引用未留下证据） | 同名同内容副本在 experiments/CGSR/des4pos/sa_geom.py 与 experiments/SpatialAlign/sa_geom.py，SHA c82c016a… |

---

## 4. 与当前 corrected KITTI（Coarse ep8 / Fine ep24）的静态对照

| 维度 | 41.66 混合实现 | corrected D4P_REIMPL_CORRECTED_s0 |
|---|---|---|
| 代码基线 | working 2185d9c（官方 3cc53c2 + 补入 Text2Loc 两模块） | work/core_src（论文重实现 core v2 + ROUND04 对齐文本包装 + TRAIN shuffle） |
| 粗检索结构 | CellRetrievalNetwork：ObjectEncoder/PointNet2/ConvBlock + Text2Loc LanguageEncoder | models/coarse.py：mfam + bilstm + cra 等（models/mfam.py、bilstm.py、cra.py） |
| 精定位结构 | Text2Loc CrossMatch（cross_hints/cross_objects）+ mlp_offsets，MSE 直接偏移 | models/fine.py：CRA + head（models/cra.py、head 键） |
| 文本编码 | Text2Loc LanguageEncoder（t5-large 冻结） | models/ste.py：FrozenT5 / FrozenCLIPText / CrossModalPrior / SteppedTextEncoder |
| 先验分支 | 无 | 有（权重含 prior_proj 等键） |
| 权重键前缀 | coarse：object_encoder / language_encoder / obj_inter_module / layers；fine：object_encoder / cross_hints / cross_objects / language_encoder / mlp_offsets | coarse：object_encoder / text_encoder / mfam / bilstm；fine：object_encoder / cra / text_encoder / head；含 prior_proj |
| 训练选择 | VAL acc 0.710 → coarse ep9；VAL offset 0.095 → fine ep25 | VAL cell R@5 0.6075 → coarse ep8；VAL mean offset 0.0917 / 2.7498 m → fine ep24 |
| 权重实体 | 3fa56619…（72,894,790 B）/ 41a74ff3…（61,645,368 B） | 95c8139c…（coarse_best20_light.pth）/ df47c1e2…（fine_best_light.pth = fine_ep24_light.pth） |
| PointNet 初始化 | ef63fdfc…，显式加载 | 未在本轮证据中建立同等 SHA 绑定（core_src 自带 object_encoder/pointcloud） |
| 评价阶段 | TEST 已读并记录 41.66 | **VAL-only**（RESULTS.json test_access=false），TEST 未读 |
| 选择口径 | 早期按 VAL 指标保存，随后在 TEST 上一次性报告 | 正式协议：VAL 选轮，TEST 锁定后单次评价 |
| 已知文档 | prior provenance 文档（2026-10-26）已标 hybrid | protocol/IMPLEMENTATION_DIFF.md 明确“非作者代码，为论文重实现” |

**已证实的区别**：代码来源、文本编码器、精定位头、STE/prior 分支、权重键空间、权重 SHA、选轮指标、评价阶段均不同；两者不是同一实现的不同训练轮次。

**未查清项**：
- corrected 的 verification/coarse_best.pth（f658db75…）与 verification/fine_best.pth（6c4233c8…）的 SHA 与 weights/ 下权威实体不同，原因未查（可能是校验流程重新序列化）；引用时应以 weights/coarse/coarse_best20_light.pth 与 weights/fine/fine_best_light.pth 为准。
- corrected 的 object_encoder 是否与 41.66 的 PointNet 权重同源，未建立 SHA 证据。
- 41.66 → corrected 的性能差异不归因（本报告不做因果推断）。

---

## 5. 未查清项与限制

1. **服务器端 T5-large 确切字节未知**：本地 artifacts\t5-large 的 model.safetensors 为 2,950,736,730 B（F32，510 tensors），归档的服务器 fetch_t5.log 记录下载大小为 2,299,965,440 B（不同）；本地 T5 与当次运行未建立 SHA 绑定。T5-large 为公开模型，属“可替换但未证明同字节”。
2. **cache 构建 argv 未保存**：test_build.log 无命令行回显，仓库内无 args 落盘；只能用脚本默认值 + manifest 路径反推。
3. **sa_geom.py 不在 reranker 目录**：同名同内容副本在 CGSR/SpatialAlign；当次构建/评价是否引用、引用的是哪一份，无证据。
4. **全树逐文件比对未做**：只核验了 git HEAD 相等与 6 个关键文件的 CRLF 等价；未对 <DES4POS_GC_ROOT>\working 全树与服务器 /root/autodl-tmp/Des4Pos/working 做清单级 SHA 比对。
5. **TRAIN 原始数据载荷未重哈希**：本轮不打开数据载荷；data\test_scenes\SHA256.json 的 TEST 场景 SHA 与 RUN_MANIFEST 一致，但 TRAIN 数据只有目录级证据。
6. **官方 Des4Pos 权重/STE/CRA/mamba 从未发布**：这不是“丢失”，而是上游本就没有；因此“官方 Des4Pos 复现”标签不成立，但混合实现的可运行性证据链是闭合的。
7. 本地无 torch，无法反序列化 state_dict；权重身份采用 zipfile 读取 data.pkl 的键名/形状静态核验，未运行模型，符合只读约束。

---

## 6. 四个问题的直接回答

### 1）对应 Coarse/Fine 是否都在本地，实体或包成员路径是什么？

**都在本地，各只有一个实体，未在任何压缩包成员索引中发现副本。**

- Coarse（ep9）：
  `<DES4POS_GC_ROOT>\artifacts\backups\checkpoints\coarse_run1\coarse_contN_epoch9_acc0.710_ecl0_eco0_p256_npa1_loss-contrastive_f-class-color-position-num.pth`
  72,894,790 B，SHA256 `3fa5661986687eeb5ad76fb19527751b3d75661917747dd1a1f69e2926c9386c`
- Fine（ep25）：
  `<DES4POS_GC_ROOT>\artifacts\backups\checkpoints\fine_run1\fine_contN_epoch25_offset0.095_lr0.0003_obj-6-16_ecl0_eco0_p256_npa1_f-class-color-position-num.pth`
  61,645,368 B，SHA256 `41a74ff32d487c66e8fc8e8ade6100ee2a1149cf76ef01616c54df7b422d8e2e`
- 附带 PointNet：
  `<DES4POS_GC_ROOT>\checkpoints\pointnet_acc0.86_lr1_p256.pth`，5,858,980 B，SHA256 `ef63fdfc9cd10e698181efe415a873e628cf86698266f005351943030016e8e1`

### 2）能否用完整 SHA 绑定到 41.66% 那次运行？

**能，且是三方一致的完整绑定。** test_results.json（记录 metric 的同时记录权重路径与 SHA）与 RUN_MANIFEST.json（记录同一组 SHA、working_head、场景与查询数）给出的 coarse/fine/pointnet/三个 cache 的 SHA256，与本地文件逐一完整相等；manifest 的 gate 复现值同为 41.66 / cell R@1 32.46；运行代码树 git HEAD = 2185d9c 与 manifest 的 working_head 相同。唯一非严格绑定的是 cache 构建的逐字命令行（未保存，只能反推），不影响权重身份。

### 3）实现身份是已确认、混合实现，还是证据不足？

**已确认的混合实现。** 证据是文件字节相等（language_encoder.py、cross_matcher.py 与 Text2Loc 上游同名文件 SHA 相同）、官方提交缺这两个文件且是后来补入、权重键结构含 Text2Loc 的 language_encoder/cross_hints/cross_objects/mlp_offsets 而完全不含 STE/CRA/prior、训练日志显示 continue_path=None 的自训流程。它不是官方 Des4Pos 复现（官方 STE/CRA 路径不可用、作者未发布权重），也不是证据不足。

### 4）恢复该版本还缺哪些具体文件？

严格意义上的“逐字节复原”仍缺：

1. **当次 T5-large 的确切权重字节**：本地 artifacts\t5-large\model.safetensors 与服务器 fetch 记录大小不同，未 SHA 绑定（公开模型，可替换，但不等同）。
2. **cache 构建命令的 argv/args 落盘文件**：test_build.log 无回显，仓库内无 args 文件；只能由 build_test_cache.py 默认值与 manifest 路径反推。
3. **当次目录中的 sa_geom.py**：reranker 目录缺失；同内容副本在 experiments/CGSR/des4pos/ 与 experiments/SpatialAlign/，但无法证明当次引用的是哪一份。
4. **服务器 working 树的全量文件清单/SHA**：本地 git 树 HEAD 一致且关键文件内容等价，但未做全树逐文件比对；若需“服务器代码可重建”的强结论，还缺逐文件清单。
5. （若目标是“官方 Des4Pos 复现”而非该混合实现）官方 STE/CRA/mamba 模块与作者发布的 checkpoint——上游从未提供，无法从本地补齐。

**不需要再找的**：Coarse ep9、Fine ep25、PointNet 三份权重与三份 test cache 均已本地在案且 SHA 绑定。

---

## 附录 A：本报告引用的关键文件

| 用途 | 路径 |
|---|---|
| 结果记录 | experiments\Des4Pos_learned_reranker\test_results.json |
| 运行 manifest | experiments\THREE_TABLES_20261021\run_20261021_top25\RUN_MANIFEST.json |
| 控制台输出 | experiments\Des4Pos_learned_reranker\test_eval_console.txt |
| 缓存构建日志 | experiments\Des4Pos_learned_reranker\test_build.log |
| 评价脚本 | experiments\Des4Pos_learned_reranker\eval_test.py、common.py |
| 缓存构建脚本 | experiments\Des4Pos_learned_reranker\build_test_cache.py |
| 运行代码树 | <DRIVE_E_TIKTOK>\_mf\src\working（HEAD 2185d9c） |
| Text2Loc 参照 | experiments\SIX_BACKBONES_20261022\text2loc_orig_src\ |
| 训练日志 | artifacts\logs\coarse_run1.log、fine_run1.log |
| 旧溯源文档 | refine-logs\SCAR_TPAMI_COLLAB_HANDOFF_20261027_v1\evidence\provenance\DES4POS_IMPLEMENTATION_PROVENANCE.md |
| 修正版结果 | refine-logs\baseline_recovery_20261031\corrected_kitti_s0\delivery_final_20261102\RESULTS.json |
