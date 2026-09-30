# SDHTG 补充材料索引与提交说明

本目录是**给投稿用的补充/附录材料**，与最终稿所在目录 `sn-article-template/` 平级；
模板目录内的任何文件都未被修改。

最终稿 `sn-article.tex` 现有 **11 张表 + 7 张图**。其表 2 题注承诺
“detailed secondary metrics are retained for the supplementary material”，表 6 题注承诺
“complete seed-level results are provided in the supplementary material”，第 5.4 节声明使用
Wilcoxon + Holm + Cliff's δ——这些内容都由本目录提供。

## 一、材料清单与对应关系

| 编号 | 内容 | 类型 | 支撑正文位置 | 建议投稿形式 |
|---|---|---|---|---|
| S1 | 配对统计检验（Wilcoxon + Holm q + Cliff's δ，三个检验族） | 表 | §5.4 统计方法声明；§6.1–6.3 结论 | Supplementary Table S1 |
| S2 | 主结果完整次要指标（AUROC / Precision / Recall / MCC / 平衡准确率，5 种子） | 表 | 表 2 题注承诺 | Supplementary Table S2 |
| S3 | 不平衡学习与原型配置的逐种子结果 | 表 | 表 6 题注承诺 | Supplementary Table S3 |
| S4 | 最终超参数清单（含“是否做过敏感性扫描”与代码出处） | 表 | §5.5 实现细节 | Supplementary Table S4 |
| S5 | 参数敏感性扫描（78 个 run 的逐种子值） | 表 | §6.7 稳定性陈述 | Supplementary Table S5 |
| S6 | 语义与实体捷径消融（45 个 run） | 表 | §6.4 语义稳健性 | Supplementary Table S6 |
| S7 | 模板级解析噪声配对（改写率 20% 与 40%，32 个 run） | 表 | §6.5 解析噪声 | Supplementary Table S7 |
| S8 | 实体可辨识性对照（实体留出协议） | 表 | §6.5 实体不可辨识 | Supplementary Table S8 |
| S9 | 掩码模板预训练基线（LogBERT 路线，统一输入协议） | 表 | §5.3 预训练路线 | Supplementary Table S9 |
| S10 | 文献参照值（各方法原始协议 + 协议差异说明） | 表 | §5.3 对比协议 | Supplementary Table S10 |
| S11 | 对比基线分类与本文用法 | 表 | §5.3 | Supplementary Table S11 |
| S12 | 不同负样本策略的对比预训练 | 表 | §5.4 预训练增强 | Supplementary Table S12 |
| S13 | 语义字段来源与泄漏控制 | 表 | §5.2 数据协议 | Supplementary Table S13 |
| SuppFig1 | 训练动力学与配对效应量（含 HDFS 单种子晚期失稳） | 图 | §7.7 复现性限制 | Supplementary Figure S1 |
| SuppFig2 | 参数敏感性与解析噪声鲁棒性（两档改写率） | 图 | §5.5 / §6.5 | Supplementary Figure S2 |
| Source Data | 724 个 run 的逐种子指标（含由 P/R 反推的 MCC 与平衡准确率）、统计检验原始输出、效率基准原始记录 | 数据 | 表 2–11、图 2–7 | Source Data（单独上传） |
| Protocol | 实验设计与运行清单、数据/缓存 SHA-256 清单 | 文本 | §5.2、§7.7、Data Availability | Supplementary Text（可选） |

## 二、两种放置方式（二选一）

1. **作为独立补充材料提交（推荐）**：把 `tables/` 与 `figures/` 的内容按 S1–S13、
   SuppFig1–2 的编号整理成一个 PDF（或逐表 CSV + 图 PDF），`source_data/` 单独作为
   Source Data 上传；正文的 `\bmhead{Supplementary information}` 处使用
   `Supplementary_information_statement.md` 中的段落。
2. **放进正文附录**：最终稿的 `\begin{appendices} … \end{appendices}` 目前仍是模板占位
   （“Section title of first appendix”）。若期刊偏好文内附录，可把 `tables/` 的 CSV 转成
   表格粘贴到该处；注意附录表号会与正文表 1–11 分开编号（A1…），图号同理。

## 三、目录结构

```
supplementary/
  README_索引与提交说明.md     本文件（不必随稿提交）
  Supplementary_information_statement.md   可粘贴的正文声明（英文）
  tables/     S1–S13（CSV，UTF-8 BOM，可直接用 Excel/pandas 打开）
  figures/    SuppFig1–2（PDF / PNG / SVG）
  source_data/ per_seed_metrics_all_runs.csv、statistics_report.json、efficiency_benchmark.json
  protocol/   EXPERIMENT_DESIGN.md、data_provenance_and_hashes.csv
```

## 四、作者自查清单（不必随稿提交）

1. **SSH 主协议划分待确认**：`data/processed/ssh` 的实际划分为 1824/259/260
   （77.8% / 11.1% / 11.1%），且不是按时间排序（train 含最晚一天的会话），与配置
   隐含的“60/20/20 时序划分”不一致；重跑前不要按现行文字描述该协议。
2. **跨层消息**：`configs/model/sdhtg.yaml` 与阶梯 L7 均为 `use_cross_level_messages: true`，
   该模块**属于最终模型**（HDFS 上 ΔF1 = +2.17 个百分点）；中文稿中“按协议移除”的表述
   已不适用于终稿，英文终稿请勿沿用。
3. **超参数表**：原稿列出的 $K_m=10$ 在代码中不存在，S4 已按最终代码重写。
4. 正文 Data/Code Availability 与 Funding 处仍为占位符，需填入仓库地址与基金编号。
5. S9 的 BGL 只有 1 个种子、SSH/OpenStack 各 5 个；S7 的解析噪声为 2 个种子；
   引用这些数字时请连同种子数一起写明。
