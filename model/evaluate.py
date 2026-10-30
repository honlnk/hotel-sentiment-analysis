"""④ 统一评估：对 saved/ 下的模型在测试集上评估，产出答辩用报告与图表。

产出（reports/ 目录）：
    evaluation_report.md            评估报告：总体对比、详细指标、转折句演示
    confusion_matrix_lstm.png       LSTM 混淆矩阵图
    confusion_matrix_bert.png       BERT 混淆矩阵图

转折句演示用于检验题目硬性要求「结合上下文理解语义和转折」，
是答辩演示环节的素材。

用法：
    python evaluate.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无窗口环境出图
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
LSTM_PATH = BASE_DIR / "saved" / "lstm" / "lstm_model.keras"
BERT_DIR = BASE_DIR / "saved" / "bert"

tf.get_logger().setLevel("ERROR")
# 让图里的中文正常显示（macOS 自带字体）
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

LABEL_NAMES = ["负面", "正面"]

# 转折句示例：检验题目要求「结合上下文理解转折」的能力
DEMO_EXAMPLES = [
    "房间很干净，但是服务态度太差，不会再来了。",
    "位置有点偏，但性价比高，前台也很热情。",
    "早餐丰富，房间宽敞明亮，下次还住这家。",
    "服务很热情，可惜隔音太差，一晚上没睡好。",
    "离地铁站很近，房间却小得转不开身。",
]


def plot_confusion_matrix(cm: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)
    ax.set_xticks([0, 1], labels=LABEL_NAMES)
    ax.set_yticks([0, 1], labels=LABEL_NAMES)
    ax.set_xlabel("模型预测")
    ax.set_ylabel("真实标签")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[图表] {path.name}")


def predict_lstm(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """返回 (预测标签, 正面概率)。向量化层已内置在模型里，直接吃原始文本。

    注意：Keras 3 不接受 numpy unicode 数组，字符串输入必须用 tf.string 张量。
    """
    model = tf.keras.models.load_model(LSTM_PATH)
    probs = model.predict(tf.constant([[t] for t in texts]), verbose=0).ravel()
    return (probs >= 0.5).astype(int), probs


def predict_bert(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    from transformers import BertTokenizerFast, TFBertForSequenceClassification

    tokenizer = BertTokenizerFast.from_pretrained(BERT_DIR)
    model = TFBertForSequenceClassification.from_pretrained(BERT_DIR)
    enc = tokenizer(
        texts, truncation=True, padding="max_length", max_length=128, return_tensors="tf"
    )
    logits = model.predict(dict(enc), verbose=0).logits
    probs = tf.nn.softmax(logits, axis=-1).numpy()[:, 1]
    return (probs >= 0.5).astype(int), probs


def write_report(results: dict, demo_rows: list[tuple[str, np.ndarray]]) -> None:
    lines = [
        "# 模型评估报告",
        "",
        "数据集：ChnSentiCorp 测试集（训练全程不接触）｜目标：准确率 93%+",
        "",
        "## 一、总体对比",
        "",
        "| 模型 | 测试集准确率 |",
        "|------|------------|",
    ]
    for name, r in results.items():
        lines.append(f"| {name} | {r['accuracy']:.2%} |")

    for name, r in results.items():
        lines += ["", f"## {name} 详细指标", "", "```", r["report"].rstrip(), "```"]

    lines += ["", "## 转折句演示（题目硬性要求的检验）", ""]
    lines.append("| 评论 | " + " | ".join(f"{n} 判定（正面概率）" for n, _ in demo_rows) + " |")
    lines.append("|" + "---|" * (1 + len(demo_rows)))
    for i, text in enumerate(DEMO_EXAMPLES):
        cells = [text]
        for _, probs in demo_rows:
            p = float(probs[i])
            cells.append(f"{'好评' if p >= 0.5 else '差评'}（{p:.0%}）")
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## 混淆矩阵", ""]
    for name in results:
        lines.append(f"![{name} 混淆矩阵](confusion_matrix_{name.lower()}.png)")

    (REPORTS_DIR / "evaluation_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("\n[保存] 评估报告已写入 reports/evaluation_report.md")


def main() -> None:
    df = pd.read_csv(PROCESSED_DIR / "test.csv")
    texts = df["review"].astype(str).tolist()
    y_true = df["label"].astype(int).to_numpy()
    print(f"[数据] 测试集 {len(texts)} 条")

    predictors: dict[str, object] = {}
    if LSTM_PATH.exists():
        predictors["LSTM"] = predict_lstm
    if (BERT_DIR / "config.json").exists():
        predictors["BERT"] = predict_bert
    if not predictors:
        raise SystemExit("未找到已训练的模型，请先运行 train_lstm.py / train_bert.py")

    results: dict[str, dict] = {}
    demo_rows: list[tuple[str, np.ndarray]] = []
    for name, predict_fn in predictors.items():
        print(f"\n===== {name} =====")
        y_pred, _ = predict_fn(texts)
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=4)
        cm = confusion_matrix(y_true, y_pred)
        results[name] = {"accuracy": round(float(acc), 4), "report": report}
        print(f"准确率：{acc:.4f}")
        print(report)
        plot_confusion_matrix(cm, f"{name} 混淆矩阵", REPORTS_DIR / f"confusion_matrix_{name.lower()}.png")

        _, demo_probs = predict_fn(DEMO_EXAMPLES)
        demo_rows.append((name, demo_probs))

    write_report(results, demo_rows)


if __name__ == "__main__":
    main()
