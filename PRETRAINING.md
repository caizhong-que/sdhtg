# 对比预训练协议

## 仅正常样本

```bash
python scripts/train.py --config configs/experiment/main.yaml \
  --pretrain-protocol normal_only
```

适合训练标签可信且正常样本充足的场景。该协议仍使用实例级InfoNCE：同一会话的两个增强视图为正对，其他会话为负样本。

## 全训练集

```bash
python scripts/train.py --config configs/experiment/main.yaml \
  --pretrain-protocol all_train
```

适合无标签预训练协议或需要避免标签进入表征学习阶段的实验。标签只用于后续监督微调，不进入InfoNCE。

## 仅执行预训练

```bash
python scripts/train.py --config configs/experiment/main.yaml \
  --pretrain-only --pretrain-protocol normal_only
```

## 恢复预训练

```bash
python scripts/train.py --config configs/experiment/main.yaml \
  --pretrain-only --pretrain-protocol normal_only \
  --pretrain-resume outputs/bgl/main/seed_42/pretraining/pretrain_last.pt
```

输出：

- `pretrain_last.pt`：最近一次完整训练状态；
- `pretrain_best.pt`：最低预训练损失状态；
- `pretrained.pt`：供微调和迁移使用的模型与投影头权重；
- `pretrain_history.json`：逐轮损失；
- `pretrain_result.json`：协议、最佳损失和产物路径。

不同协议的结果必须分别报告，不能在测试集上选择预训练协议。
