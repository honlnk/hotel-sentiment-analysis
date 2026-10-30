"""② LSTM 基线模型：先用最简结构把「数据 → 训练 → 评估 → 保存」全流程跑通。

作用：兜底模型 + 模型对比的基线。若 BERT 环节出问题，始终有可用的模型交付。

流程：
    加载 data/processed/train.csv → 划分 90% 训练 / 10% 验证 →
    字符级向量化（中文按单字切分，统一截断/填充到 128）→
    Embedding + 双向 LSTM + 分类头 → 训练 12 轮（按验证集准确率
    保存最优权重，机制与 BERT 侧一致）→
    用最优权重在 data/processed/test.csv 上计算准确率 →
    模型保存到 saved/lstm/lstm_model.keras（向量化层内置在模型里，
    Web 端加载后可直接对原始文本调用 predict）

用法：
    python train_lstm.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
SAVE_PATH = BASE_DIR / "saved" / "lstm" / "lstm_model.keras"

SEED = 42
MAX_TOKENS = 20000   # 字符词表上限
SEQ_LEN = 128        # 序列统一长度（与 BERT 保持一致，便于对比）
BATCH_SIZE = 64
EPOCHS = 12
LEARNING_RATE = 5e-4  # 128 单元 BiLSTM 在 1e-3 下训练震荡（首跑实测），减半求稳


def load_split(name: str):
    df = pd.read_csv(PROCESSED_DIR / f"{name}.csv")
    texts = df["review"].astype(str).to_numpy().reshape(-1, 1)  # (n, 1)，对应字符串输入层
    labels = df["label"].astype(int).to_numpy()
    return texts, labels


def build_model(vectorizer: tf.keras.layers.Layer) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(1,), dtype=tf.string, name="review"),
            vectorizer,
            tf.keras.layers.Embedding(MAX_TOKENS, 64, mask_zero=True),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, dropout=0.1)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="lstm_sentiment",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


class SaveBest(tf.keras.callbacks.Callback):
    """每个 epoch 结束按验证准确率保存最优权重，机制与 BERT 侧一致。"""

    def __init__(self, save_path: Path):
        super().__init__()
        self.save_path = save_path
        self.best_acc = -1.0
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None) -> None:
        acc = float((logs or {}).get("val_accuracy", -1.0))
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_epoch = epoch + 1
            self.model.save(self.save_path)
            print(f"\n[保存] 第 {epoch + 1} 轮验证准确率 {acc:.4f} 创新高 → 已保存最优权重")


def main() -> None:
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    texts, labels = load_split("train")
    x_tr, x_val, y_tr, y_val = train_test_split(
        texts, labels, test_size=0.1, random_state=SEED, stratify=labels
    )
    print(f"[数据] 训练 {len(x_tr)} 条 / 验证 {len(x_val)} 条")

    # 中文没有空格分隔，按单字切分（字符级）是最简单有效的方案
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        standardize="lower_and_strip_punctuation",
        split="character",
        output_sequence_length=SEQ_LEN,
    )
    vectorizer.adapt(tf.constant(x_tr.ravel()))

    model = build_model(vectorizer)
    model.summary()

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
        .shuffle(2048, seed=SEED)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE)

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)  # 训练中途就会落盘，需提前建目录
    save_best = SaveBest(SAVE_PATH)
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[save_best])
    print(f"[训练] 完成；最优：第 {save_best.best_epoch} 轮（验证准确率 {save_best.best_acc:.4f}）")

    model = tf.keras.models.load_model(SAVE_PATH)  # 用验证集选出的最优权重做测试评估

    x_test, y_test = load_split("test")
    test_loss, test_acc = model.evaluate(
        tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE), verbose=0
    )
    print(f"[结果] 测试集准确率：{test_acc:.4f}（loss={test_loss:.4f}）")

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(SAVE_PATH)
    print(f"[保存] 最优模型已存于 {SAVE_PATH}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model": "LSTM（字符级 + BiLSTM 基线）",
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "best_epoch": save_best.best_epoch,
        "history": {k: [round(float(v), 4) for v in vals] for k, vals in history.history.items()},
        "test_accuracy": round(float(test_acc), 4),
        "test_loss": round(float(test_loss), 4),
    }
    (REPORTS_DIR / "lstm_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[保存] 训练指标已写入 reports/lstm_metrics.json")


if __name__ == "__main__":
    main()
