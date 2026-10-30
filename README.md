# 酒店评论情感分析系统

中文酒店评论情感二分类（好评 / 差评）：TensorFlow + ChnSentiCorp 数据集 + BERT 微调，配套 Web 演示系统。

详细需求见 [docs/需求文档.md](docs/需求文档.md)。

## 目录结构

```
hotel-sentiment-analysis/
├── docs/                 # 需求文档
├── model/                # ★ 模型侧：数据、训练、评估（Python 3.12 独立虚拟环境）
│   ├── requirements.txt  #   Python 依赖清单
│   ├── prepare_data.py   #   ① 下载 ChnSentiCorp，清洗并切分 train / test
│   ├── train_lstm.py     #   ② LSTM 基线：先跑通全流程，拿兜底模型
│   ├── train_bert.py     #   ③ BERT 微调：核心脚本，目标 93%+
│   ├── evaluate.py       #   ④ 统一评估：准确率、混淆矩阵（答辩用图表）
│   ├── data/             #   数据集（脚本自动下载 / 生成）
│   ├── saved/            #   训练产物：可加载的模型文件
│   └── reports/          #   评估报告与图表
└── web/                  # Web 演示系统（前端 + 预测接口）
```

## 模型侧运行顺序

```bash
cd model
uv venv --python 3.12 .venv && source .venv/bin/activate   # Python 3.12 独立环境
uv pip install -r requirements.txt

python prepare_data.py    # ① 数据就绪
python train_lstm.py      # ② LSTM 基线，验证全流程
python train_bert.py      # ③ BERT 微调，冲指标
python evaluate.py        # ④ 生成评估报告
```

## 模型侧 → 系统侧的衔接

训练脚本把模型保存到 `model/saved/`，Web 系统只需加载该目录下的模型文件做预测（推理），
不涉及任何训练逻辑。
