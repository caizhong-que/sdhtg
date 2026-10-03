# SDHTG 补充材料：索引与提交说明

本目录是**可直接用于投稿**的补充材料，放在最终稿目录 `sn-article-template/` 的**上一级**；
模板目录内的文件（`sn-article.tex` / `sn-article.pdf` / `sn-jnl.cls` 等）未被改动。

## 一、已生成的文件

| 文件 | 内容 | 投稿时选择的上传类型 |
|---|---|---|
| `Supplementary_Information.pdf` | **单一 SI PDF（12 页）**：封面（标题、作者、通讯作者）、内容清单、Supplementary Text S1–S2、Supplementary Tables S1–S13、Supplementary Figures S1–S2、Supplementary References | Supplementary material / Supporting information |
| `Supplementary_Information.tex` | SI 的 LaTeX 源文件（供期刊要求提供源文件时上传，或后续修改） | 通常不必上传 |
| `Source_Data_1.xlsx` | **逐种子源数据工作簿**（0.09 MB，11 个工作表，见下） | Source data |
| `tables/S*.csv` | 与 SI 中表格对应的机器可读版本（UTF-8 BOM，可直接用 Excel/pandas 打开） | 可选，通常并入 Source Data |
| `figures/SuppFig1–2.{pdf,png,svg}` | 两张补充图的矢量/位图文件 | 期刊要求提供图件时上传 |
| `protocol/EXPERIMENT_DESIGN.md`、`protocol/data_provenance_and_hashes.csv` | 实验协议与运行清单、数据缓存 SHA-256 | 可选（Supplementary Text 已在 PDF 内） |
| `Supplementary_information_statement.md` | 可直接粘贴到正文 `\bmhead{Supplementary information}` 的英文声明段落 | 粘贴进正文 |

## 二、SI PDF 的结构（12 页）

1. **封面**：文章标题、作者与单位、通讯作者邮箱、本文包含的内容清单。
2. **Supplementary Text S1**：实验设计与复现协议（数据与泄漏控制、划分、模型选择与统计口径、复现性）。
3. **Supplementary Text S2**：数据来源与缓存哈希表（五个数据集缓存的构建时间、代码版本、`sessions.parquet` SHA-256）。
4. **Supplementary Tables S1–S13**：

| 表 | 内容 | 正文对应位置 |
|---|---|---|
| S1 | 配对统计检验（Wilcoxon、Holm q、Cliff's δ，含配对单位与可达成的最小 p） | §5.4 统计口径声明 |
| S2 | 主对比的补充指标（AUROC / Precision / Recall / MCC / 平衡准确率，5 种子） | 表 2 题注承诺 |
| S3 | 不平衡学习与原型配置（SSH，3 种子） | 表 6 题注承诺 |
| S4 | 最终超参数（含"是否做过敏感性扫描"与配置文件出处） | §5.5 实现细节 |
| S5 | 参数敏感性扫描汇总（78 run，逐种子在 Source Data） | §6.7 稳定性陈述 |
| S6 | 语义与实体捷径消融（SSH，5 种子） | §6.4 |
| S7 | 解析噪声配对（改写率 20% 与 40%，同一权重与阈值） | §6.5 |
| S8 | 实体可辨识性（实体留出协议） | §6.5 |
| S9 | 掩码模板预训练基线（LogBERT 路线，统一输入协议） | §5.3 |
| S10 | 文献参照值及其协议差异（明确"不可直接比较"） | §5.3 对比协议 |
| S11 | 基线分类与比较协议（受控 / 复现 / 文献参照） | §5.3 |
| S12 | 对比预训练的负样本策略 | §5.4 |
| S13 | 语义字段来源与泄漏控制 | §5.2 数据协议 |

5. **Supplementary Figures S1–S2**：训练动力学与配对效应量（含 HDFS 单种子晚期失稳）；参数敏感性与解析噪声鲁棒性。
6. **Supplementary References**：S10 所引用文献的完整条目（11 条），使 SI 自成体系。

## 三、Source Data 工作簿（`Source_Data_1.xlsx`）

| 工作表 | 行数 | 内容 |
|---|---|---|
| README | 17 | 各工作表说明、列含义、数据来源与生成脚本 |
| S1 paired statistics | 27 | 三个检验族的 Δ、p、Holm q、Cliff's δ、配对种子数 |
| S2 main secondary metrics | 26 | 5 数据集 × 5 方法的 AUPRC/AUROC/P/R/F1/MCC/平衡准确率（均值±标准差） |
| S3 imbalance prototype | 22 | 逐种子 AUPRC/F1 |
| S5 sensitivity scan | 79 | 78 个敏感性 run 的逐种子验证/测试指标 |
| S6 semantic shortcuts | 46 | 45 个捷径消融 run 的逐种子指标 |
| S7 parsing noise | 33 | 32 个配对噪声 run（受噪与配对干净的 AUPRC/F1） |
| S8 entity identifiability | 13 | 分组指标（seen / holdout / pooled） |
| S9 masked-template baseline | 42 | 41 个 run 的逐种子指标 |
| S12 negative sampling | 37 | 36 个预训练 run 的逐种子指标 |
| all runs per seed | 725 | 仓库内全部 724 个 run 的逐种子指标（含由 P/R 反推的 MCC 与平衡准确率、校准阈值、训练轮数） |

## 四、提交要点（按通用惯例）

- 目标期刊若提供补充材料模板或专门指南，**以期刊模板为准**；本套材料按"无模板"时的通用惯例组织：单一 SI PDF + 独立 Source Data + 代码仓库。
- 正文需**至少一次**引用每一项：目前终稿表 2 与表 6 的题注、以及 §5.4 的统计声明已提到"supplementary material"，建议再把具体编号写入正文，例如 "…are provided in Supplementary Table S2"、"…in Supplementary Table S1"（此项属于正文修改，本轮未改动正文文件）。
- 术语、缩写与正文一致（AUPRC、F1、Cliff's δ、CB-Focal、GRU-flat 等）。
- 文件体积：SI PDF 约 0.2 MB、Source Data 约 0.09 MB、图件合计约 1.5 MB，均远低于常见的单文件限制（如 50 MB）。
- 上传时的类型选择：`Supplementary_Information.pdf` → Supplementary material / Supporting information；`Source_Data_1.xlsx` → Source data；`figures/` 仅在期刊要求单独图件时上传。

## 五、仍需注意（不属于补充材料本身）

1. 正文中 Data/Code Availability 与 Funding 仍为占位符，需填仓库地址与基金编号。
2. 论文中若继续使用 SSH 的"60/20/20 时序划分"表述，需先按仓库 README 的说明重建缓存并重跑；SI 的 Text S1 已按实际协议描述（HDFS/OpenStack 时序，BGL/Thunderbird 随机）。
3. `Supplementary_Information.pdf` 中存在两处 <5 pt 的排版溢出（表格末行），不影响阅读；如需彻底消除可在 SI 源文件中对相应表格再缩小字号。
