# 第三批：训练系统

实现类别有效样本数加权Focal Loss、正常原型间隔、边界熵/稀疏率/嵌套正则、两视图InfoNCE、困难负样本、课程式边界温度和FiLM强度、CUDA AMP、梯度裁剪、早停、验证集阈值校准、原子断点及完整RNG恢复。

主实验种子使用：42、123、256、512、1024、2048、4096、8192、16384、32768。阈值仅由验证集校准，测试集不得重新搜索。

```bash
cd sdhtg
pip install -e ".[dev]"
pytest -q
python scripts/train.py --config configs/experiment/main.yaml
python scripts/run_multiseed.py --config configs/experiment/main.yaml
```

每个种子目录保存训练清单、配置和数据SHA-256、初始模型哈希、last/best断点、训练历史、校准阈值及结果。断点包含Python、NumPy、CPU和所有CUDA RNG状态。
