"""② LSTM 基线模型：先用最简结构把「数据 → 训练 → 评估 → 保存」全流程跑通。

作用：兜底模型 + 练手。若 BERT 环节出问题，始终有可用的模型交付。

流程：
    加载 data/processed/train.csv → 划分 90% 训练 / 10% 验证 →
    字符级向量化（中文按单字切分，统一截断/填充到 128）→
    Embedding + 双向 LSTM + 分类头 → 训练 5 轮 →
    在 data/processed/test.csv 上计算准确率（预期 90% 左右）→
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
EPOCHS = 5


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
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, dropout=0.2)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="lstm_sentiment",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


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

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    x_test, y_test = load_split("test")
    test_loss, test_acc = model.evaluate(
        tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE), verbose=0
    )
    print(f"[结果] 测试集准确率：{test_acc:.4f}（loss={test_loss:.4f}）")

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(SAVE_PATH)
    print(f"[保存] 模型已保存到 {SAVE_PATH}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model": "LSTM（字符级 + BiLSTM 基线）",
        "epochs": EPOCHS,
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
