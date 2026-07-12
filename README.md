# SDHTG

Semantic-Guided Differentiable Hierarchical Temporal Graph Learning for
Class-Imbalanced Log Anomaly Detection.

## 第一批范围

本版本完整实现数据层：

- 用户本地数据登记与可选 LogHub 下载；
- SHA-256、文件大小和数据清单；
- BGL、HDFS、OpenStack、SSH、Thunderbird 严格格式适配；
- Drain3 模板解析；
- 实体、Action、Status 语义先验；
- Fixed-window、Idle-gap、原生会话三类分组；
- 按时间且按会话隔离的 60/20/20 划分；
- 训练集拟合词表，验证/测试未知值映射为 UNK；
- Parquet 缓存、统计报告和数据泄漏检查。

## 安装

```bash
conda env create -f environment.yml
conda activate sdhtg
pip install -e ".[dev]"
pytest -q
```

## 数据准备

数据可由用户自行下载。推荐从 LogHub 官方仓库取得。将文件放入：

```text
data/raw/BGL/BGL.log
data/raw/HDFS/HDFS.log
data/raw/HDFS/anomaly_label.csv
...
```

每个 YAML 的 `files` 必须填写真实相对路径。正式实验必须填写 SHA-256；探索运行可显式传入 `--allow-missing-hash`，清单会记录该非严格状态。

下载器只下载 YAML 中明确声明的 URL，不猜测镜像地址：

```bash
python scripts/download_loghub.py --config configs/data/bgl.yaml
```

预处理：

```bash
python scripts/preprocess.py --config configs/data/bgl.yaml
```

输出包括：

```text
data/processed/bgl/events.parquet
data/processed/bgl/sessions.parquet
data/processed/bgl/vocab.json
data/processed/bgl/drain_state.bin
data/processed/bgl/manifest.json
data/processed/bgl/quality_report.json
```

## 数据集规则

- BGL、Thunderbird：使用官方行首告警标签，`-` 表示正常；实体取官方结构中的 `Node`。不得用 `ERROR/WARN` 文本生成标签。
- HDFS：从内容中提取所有 `blk_-?\\d+`；标签仅来自 `anomaly_label.csv`。一个事件涉及多个块时复制到各块会话，并记录 `source_event_id`，切分以块会话为单位。
- OpenStack：优先使用 Request/Instance ID 形成会话；标签必须来自 YAML 声明的官方/人工标注文件或结构化标签字段。
- SSH：按 Host/IP/User 等配置字段构造实体和会话；标签必须来自明确标签字段或标签文件。
- 所有数据：Drain 只在训练时段事件上拟合，再以冻结状态转换验证和测试，防止模板泄漏。

具体 LogHub 发布版本可能有不同文件名。配置中的 `log_format`、时间格式和标签模式是数据契约，解析不匹配将立即失败。
