# 第二批实现说明

## 完成模块

1. 多源事件编码：Template、Entity、Action、Status、连续时间五源门控融合；
2. 严格因果编码：单向多层 GRU，PackedSequence 屏蔽 Padding；
3. 上下文战略：逐事件因果 Strategy GRU 与恒等初始化 Residual FiLM；
4. 嵌套软边界：`p_entity = p_action × p(entity | action)`；
5. 可微层次聚合：固定候选段软成员矩阵，无训练期硬阈值；
6. PyG 异构图：Status、Action、Entity三类节点和十类关系；
7. 稀疏局部时间边、语义边、双向层间归属边；
8. 边权感知消息传递：软归属权重参与消息计算并保留梯度；
9. 分层检测：三级节点证据、Strategy门控、正常多原型距离；
10. 消融开关：数据源、FiLM、边界、语义边、时间边、跨层消息、原型；
11. 模型测试：嵌套约束、Padding不变性、前缀因果性和端到端梯度。

## 数学约定

边界概率表示“当前事件之前开始新片段”。事件 `t` 对候选片段 `k` 的成员概率为：

`M[t,k] = p[k] × Π(j=k+1..t)(1-p[j])`，其中 `k <= t`。

首事件边界强制为1，剩余概率质量回填首候选，保证每个有效事件的成员概率和为1。
这与软重置思想一致，但避免训练中将边界阈值化。推理时可以另行校准边界阈值并导出硬层次。

## 执行

```bash
cd sdhtg
pip install -r requirements-model.txt
pip install -e ".[dev]"
pytest -q
python scripts/inspect_model.py --config configs/model/sdhtg.yaml
```

## 重要实现边界

- PyG 图的拓扑候选选择是离散的，但所有被选中层间边的权重仍可微；
- Semantic ID 的 `argmax` 仅用于构造候选语义边，不承担特征聚合；
- 图级检测会使用整个有效序列，因此“因果性”测试针对事件编码、Strategy和边界前缀；
- 在线图级推理需要前缀缓存，将在训练/推理批次中实现；
- 语义抽取质量取决于第一批人工覆写或严格审核后的Entity–Action–Status映射。
