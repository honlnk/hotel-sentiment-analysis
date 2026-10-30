"""③ BERT 微调（核心脚本）：加载中文预训练模型，接分类头，小学习率微调。

这是题目要求的核心：BERT 结合上下文理解语义与转折，
如「房间很干净，但是服务态度太差」应正确判为差评。

流程：
    加载 data/processed/train.csv → 划分 90% 训练 / 10% 验证 →
    BERT tokenizer 序列化（固定 max_length=128）→
    加载 bert-base-chinese 预训练权重 + 二分类头 →
    warmup + 线性衰减学习率（transformers 官方 create_optimizer）训练 3 轮 →
    每轮结束按验证准确率保存最优权重（防止过拟合轮次混入产物）→
    用保存的最优权重在测试集上评估一次（测试集只在此处评估，目标 93%+）→
    模型与 tokenizer 保存到 saved/bert/（Web 系统加载此目录）

用法：
    python train_bert.py
"""

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras  # transformers 的 TF 模型暂不支持 Keras 3，统一使用这个 Keras 2 兼容层
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizerFast,
    TFBertForSequenceClassification,
    create_optimizer,
)

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
SAVE_DIR = BASE_DIR / "saved" / "bert"

PRETRAINED = "bert-base-chinese"
HF_MIRROR = "https://hf-mirror.com"  # HuggingFace 直连失败时的国内镜像
SEED = 42
MAX_LEN = 128        # 与 LSTM 基线一致，便于对比
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5  # 微调必须用小学习率，否则会破坏预训练权重


def load_split(name: str):
    df = pd.read_csv(PROCESSED_DIR / f"{name}.csv")
    texts = df["review"].astype(str).to_numpy().reshape(-1, 1)
    labels = df["label"].astype(int).to_numpy()
    return texts, labels


def _load():
    tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED)
    # use_safetensors=False：绕开 transformers 4.57 中 safetensors→TF 转换的已知 bug
    # （TypeError: 'builtins.safe_open' object is not iterable），改走 .bin 加载路径
    model = TFBertForSequenceClassification.from_pretrained(
        PRETRAINED, num_labels=2, use_safetensors=False
    )
    return tokenizer, model


def load_pretrained():
    """加载预训练 tokenizer 与模型；默认走 HuggingFace，直连失败自动切国内镜像。"""
    try:
        return _load()
    except Exception as err:
        print(f"[警告] HuggingFace 直连失败（{type(err).__name__}），切换镜像 {HF_MIRROR} 重试")
        os.environ["HF_ENDPOINT"] = HF_MIRROR  # 必须在重新拉取前设置
        return _load()


def make_ds(tokenizer, texts: np.ndarray, labels: np.ndarray, shuffle: bool = False) -> tf.data.Dataset:
    enc = tokenizer(
        list(texts.ravel()),
        truncation=True,
        padding="max_length",  # 固定长度：GPU 上动态 shape 会反复编译，很慢
        max_length=MAX_LEN,
        return_tensors="tf",
    )
    ds = tf.data.Dataset.from_tensor_slices((dict(enc), labels))
    if shuffle:
        ds = ds.shuffle(2048, seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


class SaveBestPretrained(tf_keras.callbacks.Callback):
    """验证准确率创新高时保存当前权重（替代「固定保存最后一轮」，防止过拟合轮次混入产物）。"""

    def __init__(self, save_dir: Path):
        super().__init__()
        self.save_dir = save_dir
        self.best_acc = -1.0
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        acc = float((logs or {}).get("val_accuracy", -1.0))
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_epoch = epoch + 1
            self.model.save_pretrained(self.save_dir)
            print(f"\n[保存] 第 {epoch + 1} 轮验证准确率 {acc:.4f} 创新高 → 已保存最优权重")


def main() -> None:
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    texts, labels = load_split("train")
    x_tr, x_val, y_tr, y_val = train_test_split(
        texts, labels, test_size=0.1, random_state=SEED, stratify=labels
    )
    x_test, y_test = load_split("test")
    print(f"[数据] 训练 {len(x_tr)} / 验证 {len(x_val)} / 测试 {len(x_test)}")

    print(f"[加载] 预训练模型 {PRETRAINED}（首次运行需下载约 400MB，之后走本地缓存）")
    tokenizer, model = load_pretrained()

    train_ds = make_ds(tokenizer, x_tr, y_tr, shuffle=True)
    val_ds = make_ds(tokenizer, x_val, y_val)
    test_ds = make_ds(tokenizer, x_test, y_test)

    steps_per_epoch = math.ceil(len(x_tr) / BATCH_SIZE)
    total_steps = steps_per_epoch * EPOCHS
    # BERT 微调标准配置：前 10% 步线性升温、之后线性衰减，权重衰减 0.01 抑制过拟合
    optimizer, _ = create_optimizer(
        init_lr=LEARNING_RATE,
        num_train_steps=total_steps,
        num_warmup_steps=int(0.1 * total_steps),
        weight_decay_rate=0.01,
    )

    model.compile(
        optimizer=optimizer,
        loss=tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.summary(print_fn=lambda line: line and print(line))

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    save_best = SaveBestPretrained(SAVE_DIR)

    print(f"[训练] 共 {EPOCHS} 轮，预计 20~40 分钟；建议训练期间关闭浏览器等大内存应用")
    start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[save_best])
    print(f"[训练] 完成，耗时 {(time.time() - start) / 60:.1f} 分钟；"
          f"最优：第 {save_best.best_epoch} 轮（验证准确率 {save_best.best_acc:.4f}）")

    # 测试集只在这里评估一次，且用的是验证集选出的最优权重
    tokenizer.save_pretrained(SAVE_DIR)
    model = TFBertForSequenceClassification.from_pretrained(SAVE_DIR)
    model.compile(
        loss=tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"[结果] 最优模型测试集准确率：{test_acc:.4f}（目标 93%+）")
    print(f"[保存] 最优模型与 tokenizer 已保存到 {SAVE_DIR}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model": "BERT（bert-base-chinese 微调）",
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LEN,
        "best_epoch": save_best.best_epoch,
        "best_val_accuracy": round(save_best.best_acc, 4),
        "history": {k: [round(float(v), 4) for v in vals] for k, vals in history.history.items()},
        "test_accuracy": round(float(test_acc), 4),
        "test_loss": round(float(test_loss), 4),
    }
    (REPORTS_DIR / "bert_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[保存] 训练指标已写入 reports/bert_metrics.json")


if __name__ == "__main__":
    main()
